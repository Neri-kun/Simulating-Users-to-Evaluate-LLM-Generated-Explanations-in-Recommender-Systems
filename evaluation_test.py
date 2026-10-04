"""
Movie-recommendation + LLM-explanation pipeline.

Phases
------
1. BPR model predicts top-K movies per user (PyTorch).
2. PyTorch is evicted to free VRAM for vLLM.
3. vLLM (Qwen) generates a natural-language explanation per (user, rec).
4. CF/KG/readability metrics are computed on each explanation and appended to CSV.
5. Generator LLM is evicted; a judge LLM (Llama) is loaded.
6. Judge scores every row for Fluency/Relevance/Faithfulness (regex-guided).
7. Global corpus metrics are printed and the CSV is saved.

This file is a single-file refactor focused on readability and performance;
the output CSV schema, model choices, and overall algorithm are preserved.
"""
from __future__ import annotations

# ------------------------------------------------------------
# Stdlib
# ------------------------------------------------------------
import concurrent.futures
import gc
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Sequence

# ------------------------------------------------------------
# Third-party
# ------------------------------------------------------------
import nltk
import numpy as np
import pandas as pd
import textstat
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# ------------------------------------------------------------
# PyTorch 2.6+ compatibility shim: recbole's checkpoint loader
# depends on the legacy `weights_only=False` default. Patch once,
# BEFORE importing recbole.
# ------------------------------------------------------------
_ORIGINAL_TORCH_LOAD = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _patched_torch_load

from recbole.quick_start import load_data_and_model  # noqa: E402  (must come after patch)
from vllm.distributed.parallel_state import destroy_model_parallel  # noqa: E402


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class Config:
    # Paths
    bpr_model_path: str = "saved/BPR-Sep-15-2026_21-00-22.pth"#"saved/BPR-Jan-18-2026_12-28-14.pth"
    movies_file: str = "dataset/ml-32m/movies.csv"
    #output_csv: str = "all_users_explanations_metrics.csv"
    output_csv: str = "all_users_explanations_metrics_01.05.2026.csv"

    # Recommendation sizes
    top_k: int = 5
    user_batch_size: int = 1000
    bpr_chunk_size: int = 10_000
    user_cf_sample_cap: int = 500
    history_window: int = 5

    # Models
    generator_model: str = "Qwen/Qwen2.5-1.5B"
    judge_model: str = "casperhansen/llama-3.2-1b-instruct-awq"
    # By default the simulated "user" reuses the judge LLM session (saves one
    # VRAM eviction + reload). Set to a different model if you want persona
    # diversity between the two stages.
    simulator_model: str | None = None

    # Generation hyper-params
    gen_temperature: float = 0.6
    gen_top_p: float = 0.9
    gen_max_tokens: int = 100
    judge_max_tokens: int = 60
    # One simulator call now answers for ALL top-K recommendations of a single
    # user (five 6-field blocks ≈ 5 × ~35 tokens + whitespace/labels). 400 gives
    # ample headroom even with regex-guided decoding padding.
    sim_max_tokens: int = 400
    sim_temperature: float = 0.3  # A little variance to avoid identical answers.
    max_model_len: int = 512
    judge_max_model_len: int = 2048
    gpu_mem_util: float = 0.85
    max_num_seqs: int = 1024


CFG = Config()


# ============================================================
# Constants
# ============================================================
CSV_COLUMNS: list[str] = [
    "User ID", "Recommended Movie", "Explanation Text (LLM)",
    "Explanation Text (Item-CF)", "Item-CF Strength", "PN (Necessity Drop)", "PS (Sufficiency)",
    "Explanation Text (User-CF)", "User-CF Strength", "Explanation Text (KG)", "Expl_Feature",
    "FP (Feature Precision)", "FR (Feature Recall)", "FF (Feature F1)", "FMR (Feature Match Ratio)",
    "Gunning Fog", "Flesch Reading Ease", "Flesch-Kincaid Grade", "ARI", "SMOG Index",
]

# Regex enforced at generation time on the judge LLM
JUDGE_OUTPUT_REGEX: str = (
    r"Fluency: (0\.0|0\.25|0\.5|0\.75|1\.0)\n"
    r"Relevance: (0\.0|0\.25|0\.5|0\.75|1\.0)\n"
    r"Faithfulness: (0\.0|0\.25|0\.5|0\.75|1\.0)"
)

# Precompiled regex for parsing judge output (hot-path: called per row)
_RE_FLUENCY = re.compile(r"(?i)Fluency.*?([01](?:\.\d+)?)")
_RE_RELEVANCE = re.compile(r"(?i)Relevance.*?([01](?:\.\d+)?)")
_RE_FAITHFULNESS = re.compile(r"(?i)Faithfulness.*?([01](?:\.\d+)?)")
_RE_FALLBACK_NUMBERS = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b")

# Sentinel used in the Expl_Feature column when a KG feature cannot be found
NO_FEATURE_SENTINEL = "None"
NO_GENRE_TAG = "(no genres listed)"

# ------------------------------------------------------------
# User-simulation interview (Mohseni et al. [34] dimensions: Mental Model,
# HMT Performance, User Satisfaction, User Trust & Confidence — all measured
# in the paper via interviews/questionnaires, so we simulate the interview).
#
# IMPORTANT: every user in the CSV has `top_k` recommendations (five rows each).
# We batch them into a SINGLE interview per user, where the simulated watcher
# scores all K recommendations in one pass. This preserves the persona across
# the K explanations (the real evaluator reads a person's full recommendation
# list, not five independent pop-ups) and cuts LLM calls by a factor of K.
# ------------------------------------------------------------
def _build_sim_output_regex(k: int) -> str:
    """One regex-constrained block per recommendation, separated by blank lines."""
    block = (
        r"Movie_{n}:\s*\n"
        r"Mental_Model: ([1-5])\s*\n"
        r"Satisfaction: ([1-5])\s*\n"
        r"Clarity: ([1-5])\s*\n"
        r"Trust: ([1-5])\s*\n"
        r"Persuasion: ([1-5])\s*\n"
        r"Would_Watch: (yes|no)"
    )
    return "\n\n".join(block.format(n=i + 1) for i in range(k))


# Single pattern to iterate over every Movie_N block in the decoded output
# (parser side — more forgiving than the generation-time regex).
_RE_SIM_BLOCK = re.compile(
    r"Movie_(\d+)\s*:\s*\n"
    r"\s*Mental[_ ]?Model\s*:\s*([1-5])\s*\n"
    r"\s*Satisfaction\s*:\s*([1-5])\s*\n"
    r"\s*Clarity\s*:\s*([1-5])\s*\n"
    r"\s*Trust\s*:\s*([1-5])\s*\n"
    r"\s*Persuasion\s*:\s*([1-5])\s*\n"
    r"\s*Would[_ ]?Watch\s*:\s*(yes|no|y|n)\b",
    re.IGNORECASE,
)

SIM_COLUMNS: list[str] = [
    "Sim_MentalModel", "Sim_Satisfaction", "Sim_Clarity",
    "Sim_Trust", "Sim_Persuasion", "Sim_WouldWatch",
]

# Prompt shown to the judge. Written as a constant so the main loop is readable.
JUDGE_PROMPT_TEMPLATE = (
    "You are an automated, objective grading script for an academic dataset. "
    "You are evaluating the text generated by another AI. You are NOT making the "
    "recommendation yourself, and you MUST NOT endorse or reject the movies. "
    "Do not refuse to evaluate. Do not converse. Do not offer assistance.\n\n"
    "Evaluate EACH of the following three metrics and assign a decimal score "
    "(0.0, 0.25, 0.5, 0.75, or 1.0) based on this EXACT strict rubric:\n\n"
    "1. Fluency (Grammar and Naturalness)\n"
    "   - 0.0: Broken English.\n"
    "   - 0.25: Poor grammar, awkward phrasing.\n"
    "   - 0.5: Grammatically correct but robotic.\n"
    "   - 0.75: Good flow, minor flaws.\n"
    "   - 1.0: Perfect human-like text.\n\n"
    "2. Relevance (Connection to User/Item)\n"
    "   - 0.0: Ignores history or wrong movie.\n"
    "   - 0.25: Generic, barely mentions item.\n"
    "   - 0.5: Vague connection to history.\n"
    "   - 0.75: Connects to a genre/trend.\n"
    "   - 1.0: Perfectly connects to a specific anchor movie.\n\n"
    "3. Faithfulness (Logical Soundness)\n"
    "   - 0.0: Actively contradicts history.\n"
    "   - 0.25: Mostly hallucinated.\n"
    "   - 0.5: Plausible but weak assumption.\n"
    "   - 0.75: Mostly sound logic.\n"
    "   - 1.0: The logic is flawless and directly cites valid history to justify the recommendation.\n\n"
    "CRITICAL INSTRUCTION: You MUST output exactly three lines, one for each metric, "
    "using the exact labels below. NEVER output words like 'I cannot'.\n"
    "Fluency: [Score]\n"
    "Relevance: [Score]\n"
    "Faithfulness: [Score]\n"
    "Evaluate the following data:\n"
    "User History:\n{rating_history}\n\n"
    "Recommended Movie: {rec_movie}\n"
    'Generated Explanation: "{explanation}"\n'
)


# User-simulation prompt. The LLM role-plays a movie watcher whose persona is
# grounded in the real watch history and answers a short 5-point Likert survey
# plus a binary would-watch decision for EACH of the K recommendations the
# system produced for that user. Mapped to the paper's four measures:
#   - Mental_Model      : Mohseni et al. "Mental Model of the User"
#   - Satisfaction      : "User Satisfaction of Explanation"  (usefulness)
#   - Clarity           : "User Satisfaction of Explanation"  (understandability)
#   - Trust             : "User Trust and Confidence"
#   - Persuasion + Would_Watch : "Performance of the Human-Machine Task"
#     (the user decides to watch / not-watch — analogous to a why/why-not task).
#
# The `{numbered_recs}` placeholder is filled per user with K blocks of the form
#     Movie 1: "<title>"
#     Explanation: "<generated text>"
# and the model is forced (via regex) to reply with K "Movie_N:" answer blocks.
SIM_PROMPT_TEMPLATE = (
    "You are an automated survey-filling agent for an academic user study. "
    "You role-play ONE ordinary movie watcher who has just received several "
    "recommendations. Fill in a short rating form for EACH recommendation. "
    "Do NOT explain, do NOT refuse, do NOT write narrative. Output ONLY the "
    "labeled answer blocks in the exact format shown.\n\n"
    "Rating scale (1-5):\n"
    "  1 = strongly disagree / not at all\n"
    "  2 = disagree / a little\n"
    "  3 = neutral / somewhat\n"
    "  4 = agree / quite a bit\n"
    "  5 = strongly agree / completely\n"
    "Be critical: only give 5 when something is truly excellent. Differentiate "
    "between the recommendations — they will not all deserve the same score.\n\n"
    "Each line answers one statement from the 'watcher' point of view:\n"
    "  Mental_Model  — The explanation made it clear HOW the system picked this movie for me.\n"
    "  Satisfaction  — The explanation is useful and satisfying to read.\n"
    "  Clarity       — The explanation is easy to understand at a glance.\n"
    "  Trust         — After reading it I trust the system more.\n"
    "  Persuasion    — It makes me want to watch this recommended movie.\n"
    "  Would_Watch   — yes or no: will the watcher actually watch this movie?\n\n"
    "### EXAMPLE (format only — do not copy the scores):\n"
    "Recently enjoyed:\n"
    "- Toy Story (1995)\n"
    "- Finding Nemo (2003)\n"
    "Recommendations to rate:\n"
    'Movie 1: "Shrek (2001)"\n'
    'Explanation: "Since you liked \'Toy Story\', you will enjoy \'Shrek\' because '
    'both are family-friendly animated comedies."\n'
    'Movie 2: "The Godfather (1972)"\n'
    'Explanation: "Since you liked \'Toy Story\', you will enjoy \'The Godfather\'."\n'
    "Answer:\n"
    "Movie_1:\n"
    "Mental_Model: 4\n"
    "Satisfaction: 4\n"
    "Clarity: 5\n"
    "Trust: 3\n"
    "Persuasion: 4\n"
    "Would_Watch: yes\n\n"
    "Movie_2:\n"
    "Mental_Model: 1\n"
    "Satisfaction: 1\n"
    "Clarity: 2\n"
    "Trust: 1\n"
    "Persuasion: 1\n"
    "Would_Watch: no\n\n"
    "### NOW ANSWER FOR THE REAL CASE:\n"
    "Recently enjoyed:\n{rating_history}\n"
    "Recommendations to rate:\n{numbered_recs}"
    "Answer:\n"
)


# ============================================================
# Global lookup tables (populated by build_lookups)
# ============================================================
movie_title_map: dict[int, str] = {}
item2genres: dict[int, list[str]] = {}
user2items: dict[int, list[int]] = {}
item2users: dict[int, set[int]] = {}
all_unique_genres_in_dataset: set[str] = set()

item_embs_np: np.ndarray | None = None
user_embs_np: np.ndarray | None = None


# ============================================================
# Small helpers
# ============================================================
def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    """Yield successive slices of `seq` of length `size`."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ============================================================
# Explanation generators (CPU; CF + KG baselines)
# ============================================================
def generate_item_based_explanation(user_id: int, rec_id: int) -> tuple[str, float, float, float]:
    """Find the most-similar item in the user's history via item-item cosine."""
    history_item_ids = user2items.get(user_id, [])
    if not history_item_ids:
        return "Because it is highly rated.", 0.0, 0.0, 0.0

    hist_embs = item_embs_np[history_item_ids]
    rec_emb = item_embs_np[rec_id]
    similarities = hist_embs.dot(rec_emb)

    best_idx = int(np.argmax(similarities))
    best_match_internal_id = history_item_ids[best_idx]
    original_sim_score = similarities[best_idx]

    if len(history_item_ids) > 1:
        # Drop the best match to compute the necessity (PN) gap.
        mask = np.ones(len(history_item_ids), dtype=bool)
        mask[best_idx] = False
        second_best_sim = similarities[mask].max()
        pn_drop = original_sim_score - second_best_sim
    else:
        pn_drop = original_sim_score

    hist_name = movie_title_map.get(best_match_internal_id, "Unknown")
    rec_name = movie_title_map.get(rec_id, "Unknown")
    text = f"Since you enjoyed '{hist_name}', we think you might really like '{rec_name}'."

    return (
        text,
        round(float(original_sim_score), 4),
        round(float(pn_drop), 4),
        round(float(original_sim_score), 4),  # PS == original similarity score
    )


def generate_user_cf_explanation(user_id: int, rec_id: int) -> tuple[str, float]:
    """Find the most-similar user who also watched the recommendation."""
    rec_name = movie_title_map.get(rec_id, "Unknown")
    candidates = item2users.get(rec_id, set()) - {user_id}

    if not candidates:
        return f"Viewers are loving '{rec_name}' right now.", 0.0

    candidates_list = list(candidates)
    if len(candidates_list) > CFG.user_cf_sample_cap:
        candidates_list = random.sample(candidates_list, CFG.user_cf_sample_cap)

    cand_embs = user_embs_np[candidates_list]
    tgt_emb = user_embs_np[user_id]
    similarities = cand_embs.dot(tgt_emb)

    best_idx = int(np.argmax(similarities))
    most_similar_user_id = candidates_list[best_idx]
    user_sim_score = similarities[best_idx]

    shared_items = set(user2items.get(user_id, [])) & set(user2items.get(most_similar_user_id, []))
    if shared_items:
        shared_name = movie_title_map.get(next(iter(shared_items)), "Unknown")
        text = (
            f"Users with similar tastes to yours—who also loved '{shared_name}'—"
            f"highly recommend '{rec_name}'."
        )
    else:
        text = f"Viewers with a watch history similar to yours frequently watched '{rec_name}'."

    return text, round(float(user_sim_score), 4)


def generate_feature_explanation(
    user_id: int, rec_id: int
) -> tuple[str, str, float, float, float, float]:
    """Link the recommendation to the user's most-watched matching genre."""
    rec_name = movie_title_map.get(rec_id, "Unknown")
    history_item_ids = user2items.get(user_id, [])

    if not history_item_ids:
        return (
            f"Recommended {rec_name} based on overall popularity.",
            NO_FEATURE_SENTINEL, 0.0, 0.0, 0.0, 0.0,
        )

    # Flatten all historical genres once, then count.
    historical_genres: list[str] = [
        g for iid in history_item_ids for g in item2genres.get(iid, [])
    ]
    genre_counts = Counter(historical_genres)
    rec_genres = item2genres.get(rec_id, [])

    # Pick the genre with the highest overlap, excluding the placeholder tag.
    best_link_genre: str | None = None
    highest_count = 0
    for genre in rec_genres:
        if genre == NO_GENRE_TAG:
            continue
        count = genre_counts.get(genre, 0)
        if count > highest_count:
            highest_count = count
            best_link_genre = genre

    if best_link_genre is None:
        return f"'{rec_name}' is a great addition.", NO_FEATURE_SENTINEL, 0.0, 0.0, 0.0, 0.0

    expl_features = {best_link_genre}
    rec_feature_set = set(rec_genres)
    user_feature_set = set(historical_genres)
    inter_rec = expl_features & rec_feature_set

    fp = len(inter_rec) / len(expl_features)
    fr = len(inter_rec) / len(rec_feature_set) if rec_feature_set else 0.0
    ff = (2 * fp * fr) / (fp + fr) if (fp + fr) > 0 else 0.0
    fmr = len(expl_features & user_feature_set) / len(expl_features)

    # First historical item whose genres include the linking genre.
    anchor_name = "a similar movie"
    for hist_id in history_item_ids:
        if best_link_genre in item2genres.get(hist_id, []):
            anchor_name = movie_title_map.get(hist_id, "Unknown")
            break

    text = (
        f"Because you frequently watch {best_link_genre} movies (like '{anchor_name}'), "
        f"'{rec_name}' is a perfect match."
    )
    return text, best_link_genre, round(fp, 4), round(fr, 4), round(ff, 4), round(fmr, 4)


# ============================================================
# Metrics worker (readability + CF/KG baselines)
# ============================================================
def compute_row_metrics(args: tuple[int, str, int, str]) -> dict:
    """Called per (user, recommendation, generated-text); builds one CSV row."""
    uid, user_token, rec_id, llm_text = args

    item_text, item_score, pn_score, ps_score = generate_item_based_explanation(uid, rec_id)
    user_text, user_score = generate_user_cf_explanation(uid, rec_id)
    kg_text, expl_genre, fp, fr, ff, fmr = generate_feature_explanation(uid, rec_id)

    # textstat barfs on empty strings; guard first, catch unexpected errors second.
    if llm_text:
        try:
            fog = textstat.gunning_fog(llm_text)
            fre = textstat.flesch_reading_ease(llm_text)
            fkgl = textstat.flesch_kincaid_grade(llm_text)
            ari = textstat.automated_readability_index(llm_text)
            smog = textstat.smog_index(llm_text)
        except Exception:  # noqa: BLE001 – textstat has broad failure modes
            fog = fre = fkgl = ari = smog = 0.0
    else:
        fog = fre = fkgl = ari = smog = 0.0

    return {
        "User ID": user_token,
        "Recommended Movie": movie_title_map.get(rec_id, "Unknown"),
        "Explanation Text (LLM)": llm_text,
        "Explanation Text (Item-CF)": item_text,
        "Item-CF Strength": item_score,
        "PN (Necessity Drop)": pn_score,
        "PS (Sufficiency)": ps_score,
        "Explanation Text (User-CF)": user_text,
        "User-CF Strength": user_score,
        "Explanation Text (KG)": kg_text,
        "Expl_Feature": expl_genre,
        "FP (Feature Precision)": fp,
        "FR (Feature Recall)": fr,
        "FF (Feature F1)": ff,
        "FMR (Feature Match Ratio)": fmr,
        "Gunning Fog": fog,
        "Flesch Reading Ease": fre,
        "Flesch-Kincaid Grade": fkgl,
        "ARI": ari,
        "SMOG Index": smog,
    }


# ============================================================
# Global (corpus-level) evaluation
# ============================================================
def calculate_global_metrics(csv_path: str) -> None:
    print("\n==================================================")
    print(" 📊 CALCULATING GLOBAL CORPUS METRICS (XAI & UX)")
    print("==================================================")

    df = pd.read_csv(csv_path)
    total = len(df)
    if total == 0:
        print("Error: No explanations found in the CSV.")
        return

    # --- Diversity / coverage on the LLM text column ---
    usr = df["Explanation Text (LLM)"].nunique() / total

    valid_features = df.loc[df["Expl_Feature"] != NO_FEATURE_SENTINEL, "Expl_Feature"].dropna()
    n_valid = len(valid_features)
    fcr = (
        valid_features.nunique() / len(all_unique_genres_in_dataset)
        if all_unique_genres_in_dataset else 0.0
    )

    # Shannon entropy (nats→bits via np.log2); vectorized on the probability array.
    feature_probs = valid_features.value_counts(normalize=True).to_numpy()
    fd_entropy = float(-(feature_probs * np.log2(feature_probs, where=feature_probs > 0)).sum())

    if n_valid > 1:
        counts_abs = valid_features.value_counts().to_numpy()
        total_intersections = int((counts_abs * (counts_abs - 1)).sum())
        fd_pairwise = total_intersections / (n_valid * (n_valid - 1))
    else:
        fd_pairwise = 0.0

    # --- Simple column means (all in one dict for easy printing) ---
    means = {col: df[col].mean() for col in [
        "FP (Feature Precision)", "FR (Feature Recall)", "FF (Feature F1)",
        "FMR (Feature Match Ratio)", "PN (Necessity Drop)", "PS (Sufficiency)",
        "Gunning Fog", "Flesch Reading Ease", "Flesch-Kincaid Grade", "ARI", "SMOG Index",
    ]}

    print(f"Total Recommendations Evaluated: {total}\n")
    print("--- 1. Text & Coverage Metrics (LLM Based) ---")
    print(f"USR (Unique Sentence Ratio): {usr:.4f}")
    print(f"FCR (Feature Coverage Ratio): {fcr:.4f}")
    print(f"FD (Shannon Entropy): {fd_entropy:.4f}")
    print(f"FD (Paper/Pairwise Overlap): {fd_pairwise:.4f}")

    print("\n--- 2. Feature & Causal Metrics ---")
    print(f"Mean FP: {means['FP (Feature Precision)']:.4f} | "
          f"Mean FR: {means['FR (Feature Recall)']:.4f}")
    print(f"Mean FF: {means['FF (Feature F1)']:.4f} | "
          f"Mean FMR: {means['FMR (Feature Match Ratio)']:.4f}")
    print(f"Mean PN (Necessity): {means['PN (Necessity Drop)']:.4f} | "
          f"Mean PS (Sufficiency): {means['PS (Sufficiency)']:.4f}")

    print("\n--- 3. Readability & UX Metrics (Averages on LLM Text) ---")
    print(f"Flesch Reading Ease: {means['Flesch Reading Ease']:.2f}")
    print(f"Flesch-Kincaid Grade: {means['Flesch-Kincaid Grade']:.2f}")
    print(f"Gunning Fog Index: {means['Gunning Fog']:.2f}")
    print(f"Automated Read. Index: {means['ARI']:.2f}")
    print(f"SMOG Index: {means['SMOG Index']:.2f}")
    print("==================================================\n")


# ============================================================
# Phase 0: one-time setup
# ============================================================
def setup_nltk() -> None:
    print("Verifying NLTK syllable dictionaries...")
    nltk.download("cmudict", quiet=False)
    _ = nltk.corpus.cmudict.dict()


def load_bpr_and_embeddings():
    """Load the BPR model + precomputed L2-normalized embedding arrays."""
    global item_embs_np, user_embs_np

    print("Loading BPR model & precomputing arrays...")
    config, model, dataset, _, _, _ = load_data_and_model(model_file=CFG.bpr_model_path)

    item_embs_np = F.normalize(model.item_embedding.weight.detach(), p=2, dim=1).cpu().numpy()
    user_embs_np = F.normalize(model.user_embedding.weight.detach(), p=2, dim=1).cpu().numpy()
    return config, model, dataset


def build_movie_metadata(dataset) -> None:
    """Populate movie_title_map, item2genres, and all_unique_genres_in_dataset."""
    movies_df = pd.read_csv(CFG.movies_file, sep=",")
    movies_df["movieId"] = movies_df["movieId"].astype(str)
    movies_df["genres"] = movies_df["genres"].fillna("")

    # Vectorized token→internal-id lookup still has to go through recbole; do it once.
    for _, row in movies_df.iterrows():
        try:
            iid = dataset.token2id(dataset.iid_field, row["movieId"])
        except ValueError:
            continue
        movie_title_map[iid] = row["title"]
        genres = row["genres"].split("|") if row["genres"] else []
        item2genres[iid] = genres
        all_unique_genres_in_dataset.update(genres)

    del movies_df
    gc.collect()


def build_interaction_lookups(dataset) -> None:
    """Vectorized user↔item maps via pandas groupby (≫ faster than setdefault loop)."""
    global user2items, item2users

    uids = dataset.inter_feat[dataset.uid_field].numpy()
    iids = dataset.inter_feat[dataset.iid_field].numpy()

    # Drop duplicate (u, i) pairs in first-seen order so slicing history[-5:]
    # meaningfully refers to "most recent" interactions.
    inter_df = pd.DataFrame({"u": uids, "i": iids}).drop_duplicates(["u", "i"], keep="first")

    user2items = {int(u): g["i"].tolist() for u, g in inter_df.groupby("u", sort=False)}
    item2users = {int(i): set(g["u"].tolist()) for i, g in inter_df.groupby("i", sort=False)}


def ensure_output_csv_header() -> None:
    if not os.path.exists(CFG.output_csv):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CFG.output_csv, index=False)


# ============================================================
# Phase 1: BPR top-K prediction
# ============================================================
def predict_top_k_for_all_users(config, model, all_user_ids: list[int]) -> dict[int, np.ndarray]:
    """Return {uid: np.ndarray of top-K recommended internal item ids}."""
    print("\n🔮 PHASE 1: Predicting top 5 movies for all users...")
    device = config["device"]
    model = model.to(device)
    model.eval()

    item_weight = model.item_embedding.weight  # [num_items, d]
    topk_out: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for chunk_uids in chunked(all_user_ids, CFG.bpr_chunk_size):
            uid_tensor = torch.tensor(chunk_uids, device=device)
            user_embs = model.user_embedding(uid_tensor)
            scores = user_embs @ item_weight.T  # [chunk, num_items]

            # Mask already-seen items AND the padding item (id 0).
            rows, cols = [], []
            for row_idx, uid in enumerate(chunk_uids):
                seen = user2items.get(uid, [])
                rows.extend([row_idx] * len(seen))
                cols.extend(seen)
            if rows:
                scores[rows, cols] = -float("inf")
            scores[:, 0] = -float("inf")

            _, topk_indices = torch.topk(scores, k=CFG.top_k)
            topk_indices_np = topk_indices.cpu().numpy()

            for i, uid in enumerate(chunk_uids):
                topk_out[uid] = topk_indices_np[i]

    return topk_out


# ============================================================
# Phase 3: Generator LLM
# ============================================================
def build_generator_prompt(rating_history: str, rec_name: str) -> str:
    """Build one prompt for the explanation generator.

    Prompt design is driven by the XAI criteria from Vultureanu-Albisi & Badica
    (2021) — specifically the six quality criteria (external/internal coherence,
    simplicity, articulation, contrastiveness, interaction) and the SAGES
    guidance (Simple, Adaptable, Grounded, Expandable, Sourced). The goal is
    to obtain explanations that target the seven explanation purposes the
    paper lists: transparency, scrutability, trust, effectiveness,
    persuasiveness, efficiency, and satisfaction.
    """
    return (
        "You are a movie recommendation assistant. Write ONE short, natural-"
        "sounding sentence that explains why the user will enjoy the recommended "
        "movie.\n\n"
        "RULES (follow all):\n"
        "1. GROUNDED & SOURCED — Anchor the explanation to ONE specific title "
        "from the user's history, mentioned by name in single quotes.\n"
        "2. INTERNAL COHERENCE — Name the concrete shared attribute that links "
        "the anchor and the recommendation (a genre, a theme, a tone, an era, a "
        "director/franchise), in plain language.\n"
        "3. EXTERNAL COHERENCE — Only cite real, plausible properties of the "
        "movies; do NOT invent facts, directors, or plot points.\n"
        "4. CONTRASTIVENESS — The shared attribute should make this pick feel "
        "deliberately better than a generic popular movie.\n"
        "5. SIMPLICITY & ARTICULATION — Plain everyday words, one sentence, "
        "no more than 30 words, no bullet points, no preamble.\n"
        "6. FORMAT — Start with: \"Since you liked '<anchor title>',\". Output "
        "the sentence only — nothing before or after it.\n\n"
        f"User's recent watch history (most recent last):\n{rating_history}\n\n"
        f"Recommended movie: {rec_name}\n\n"
        "Explanation:"
    )


def format_user_history(uid: int) -> str:
    history_item_ids = user2items.get(uid, [])
    recent_ids = history_item_ids[-CFG.history_window:]
    titles = [movie_title_map.get(iid, "Unknown") for iid in recent_ids]
    return "\n".join(f"- {t}" for t in titles) if titles else "No history."


def init_generator_llm() -> LLM:
    print("\n🚀 PHASE 3: Initializing vLLM Engine (ULTRA MODE)...")
    return LLM(
        model=CFG.generator_model,
        gpu_memory_utilization=CFG.gpu_mem_util,
        max_model_len=CFG.max_model_len,
        enable_prefix_caching=True,
        trust_remote_code=True,
        max_num_seqs=CFG.max_num_seqs,
        enforce_eager=True,
    )


def run_generation_and_metrics(
    llm: LLM,
    dataset,
    all_user_ids: list[int],
    topk_predictions: dict[int, np.ndarray],
) -> None:
    sampling_params = SamplingParams(
        temperature=CFG.gen_temperature,
        top_p=CFG.gen_top_p,
        max_tokens=CFG.gen_max_tokens,
    )
    max_threads = min(32, (os.cpu_count() or 4) * 2)
    print(f"\n--- Starting Generation (vLLM Batching + {max_threads} CPU Threads) ---")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as pool:
        for batch_uids in chunked(all_user_ids, CFG.user_batch_size):

            start_idx_label = batch_uids[0] if batch_uids else 0
            end_idx_label = batch_uids[-1] + 1 if batch_uids else 0
            batch_start = time.perf_counter()

            raw_prompts: list[str] = []
            flat_tasks: list[tuple[int, str, int]] = []

            for uid in batch_uids:

                user_token = dataset.id2token(dataset.uid_field, uid)
                rating_history = format_user_history(uid)



                for rec_id in topk_predictions[uid]:
                    rec_name = movie_title_map.get(rec_id, "Unknown")
                    raw_prompts.append(build_generator_prompt(rating_history, rec_name))
                    flat_tasks.append((uid, user_token, int(rec_id)))

            print(f"\n🧠 Generating {len(raw_prompts)} LLM explanations concurrently...")
            outputs = llm.generate(
                prompts=raw_prompts, sampling_params=sampling_params, use_tqdm=True
            )

            enriched = [
                (*task, out.outputs[0].text.strip())
                for task, out in zip(flat_tasks, outputs)
            ]

            print("🧮 Calculating Item-CF, User-CF, KG, and Readability metrics...")
            batch_results = list(pool.map(compute_row_metrics, enriched))

            pd.DataFrame(batch_results).to_csv(
                CFG.output_csv, mode="a", header=False, index=False
            )

            elapsed = time.perf_counter() - batch_start
            print(
                f"✅ Completed batch uids[{start_idx_label}..{end_idx_label}) "
                f"in {elapsed:.2f} seconds!"
            )

            # Free memory before next batch.
            del batch_results, raw_prompts, outputs, enriched
            gc.collect()


# ============================================================
# Phase 5/6: Judge LLM
# ============================================================
def free_generator_vram(llm: LLM) -> None:
    """Cleanly tear down vLLM's distributed state so GPU memory is reclaimed."""
    print("\n🧹 PHASE 4: Evicting Qwen Generator to free VRAM for the Judge...")
    try:
        destroy_model_parallel()
    except Exception:  # noqa: BLE001 – OK if already destroyed
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def init_judge_llm() -> LLM:
    print("\n⚖️  Initializing Judge vLLM Engine (Llama)...")
    return LLM(
        model=CFG.judge_model,
        gpu_memory_utilization=CFG.gpu_mem_util,
        max_model_len=CFG.judge_max_model_len,
        enable_prefix_caching=True,
        enforce_eager=True,
    )


def parse_judge_output(text: str) -> tuple[float, float, float]:
    """Pull (fluency, relevance, faithfulness) from a judge response.

    Primary path: labeled regex. Fallback: first three decimals in [0, 1].
    Returns NaN for any score that cannot be recovered.
    """
    f_match = _RE_FLUENCY.search(text)
    r_match = _RE_RELEVANCE.search(text)
    faith_match = _RE_FAITHFULNESS.search(text)

    if f_match and r_match and faith_match:
        return (
            float(f_match.group(1)),
            float(r_match.group(1)),
            float(faith_match.group(1)),
        )

    numbers = _RE_FALLBACK_NUMBERS.findall(text)
    if len(numbers) >= 3:
        return float(numbers[0]), float(numbers[1]), float(numbers[2])

    return float("nan"), float("nan"), float("nan")


def run_judge_evaluation(judge_llm: LLM) -> pd.DataFrame:
    guided_config = {
        "structured_outputs": StructuredOutputsParams(regex=JUDGE_OUTPUT_REGEX)
    }
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=CFG.judge_max_tokens,
        top_p=1.0,
        **guided_config,
    )

    eval_df = pd.read_csv(CFG.output_csv)
    print(f"\n--- Starting LLM-as-a-Judge Evaluation (Batch Size: {CFG.user_batch_size}) ---")

    for start_idx in range(0, len(eval_df), CFG.user_batch_size):
        end_idx = min(start_idx + CFG.user_batch_size, len(eval_df))
        batch_df = eval_df.iloc[start_idx:end_idx]
        batch_start = time.perf_counter()

        # Build prompts for this slice.
        prompts: list[str] = []
        for _, row in batch_df.iterrows():
            uid_raw = row["User ID"]
            try:
                uid_key: int | object = int(uid_raw) if pd.notna(uid_raw) else uid_raw
            except (TypeError, ValueError):
                uid_key = uid_raw

            rating_history = format_user_history(uid_key) if isinstance(uid_key, int) else "No history."
            prompts.append(JUDGE_PROMPT_TEMPLATE.format(
                rating_history=rating_history,
                rec_movie=row["Recommended Movie"],
                explanation=row["Explanation Text (LLM)"],
            ))

        print(
            f"\n🧠 Judging batch {start_idx} to {end_idx} ({len(prompts)} explanations)..."
        )
        outputs = judge_llm.generate(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=True
        )

        fluency_scores, relevance_scores, faithfulness_scores = [], [], []
        for output, (_, row) in zip(outputs, batch_df.iterrows()):
            text = output.outputs[0].text.strip()
            f_s, r_s, faith_s = parse_judge_output(text)
            fluency_scores.append(f_s)
            relevance_scores.append(r_s)
            faithfulness_scores.append(faith_s)

            if any(np.isnan(s) for s in (f_s, r_s, faith_s)):
                print(
                    f"\n⚠️  [WARNING] Failed to parse scores for Movie: "
                    f"{row['Recommended Movie']}"
                )
                print(f"Raw LLM Output:\n{text}")
                print("-" * 50)

        eval_df.loc[batch_df.index, "Judge_Fluency"] = fluency_scores
        eval_df.loc[batch_df.index, "Judge_Relevance"] = relevance_scores
        eval_df.loc[batch_df.index, "Judge_Faithfulness"] = faithfulness_scores

        elapsed = time.perf_counter() - batch_start
        print(
            f"✅ Completed judge batch {start_idx} to {end_idx} in {elapsed:.2f} seconds!"
        )

        del prompts, outputs, batch_df
        gc.collect()

    return eval_df


def print_final_judge_averages(eval_df: pd.DataFrame) -> None:
    print("\n==================================================")
    print(" ⚖️  FINAL LLM JUDGE AVERAGES")
    print("==================================================")
    print(f"Mean Fluency: {eval_df['Judge_Fluency'].mean():.2f} / 1")
    print(f"Mean Relevance: {eval_df['Judge_Relevance'].mean():.2f} / 1")
    print(f"Mean Faithfulness: {eval_df['Judge_Faithfulness'].mean():.2f} / 1")
    print("==================================================\n")


# ============================================================
# Phase 7: Simulated user-interview evaluation
# ============================================================
SimScore = tuple[float, float, float, float, float, float]


def parse_user_simulation_multi(text: str, k: int) -> list[SimScore]:
    """Parse K "Movie_N" answer blocks from one simulator response.

    The regex-guided decoder is set up to emit exactly K blocks in order
    (Movie_1, Movie_2, …, Movie_K). This parser is forgiving: it indexes the
    matches by the captured block number, so a stray extra block or reordered
    output still lands in the correct slot. Missing slots are filled with NaNs.

    Returns a list of length `k`; each element is
    (mental, satisfaction, clarity, trust, persuasion, would_watch)
    with would_watch mapped to 1.0 / 0.0 / NaN.
    """
    nan = float("nan")
    default: SimScore = (nan, nan, nan, nan, nan, nan)
    out: list[SimScore] = [default] * k

    for match in _RE_SIM_BLOCK.finditer(text):
        try:
            n = int(match.group(1))
        except ValueError:
            continue
        idx = n - 1  # Movie_N is 1-indexed in the prompt.
        if not (0 <= idx < k):
            continue

        m = float(match.group(2))
        s = float(match.group(3))
        c = float(match.group(4))
        t = float(match.group(5))
        p = float(match.group(6))
        w_token = match.group(7).lower()
        w = 1.0 if w_token.startswith("y") else 0.0

        out[idx] = (m, s, c, t, p, w)

    return out


def init_simulator_llm(judge_llm: LLM) -> tuple[LLM, bool]:
    """Return (llm, owns_llm). If `simulator_model` is None, reuse the judge
    session. Otherwise evict the judge and load a dedicated simulator.
    """
    if CFG.simulator_model is None or CFG.simulator_model == CFG.judge_model:
        print("\n🎭 Reusing judge LLM as user simulator.")
        return judge_llm, False

    print("\n🧹 Evicting judge LLM to load dedicated simulator...")
    try:
        destroy_model_parallel()
    except Exception:  # noqa: BLE001
        pass
    del judge_llm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🎭 Initializing Simulator vLLM Engine ({CFG.simulator_model})...")
    sim_llm = LLM(
        model=CFG.simulator_model,
        gpu_memory_utilization=CFG.gpu_mem_util,
        max_model_len=CFG.judge_max_model_len,
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    return sim_llm, True


def _format_numbered_recs(rec_titles: Sequence[str], explanations: Sequence[str]) -> str:
    """Render the 'Movie 1: …\nExplanation: …\n…' block for one user."""
    lines: list[str] = []
    for i, (title, expl) in enumerate(zip(rec_titles, explanations), start=1):
        lines.append(f'Movie {i}: "{title}"')
        lines.append(f'Explanation: "{expl}"')
    return "\n".join(lines) + "\n"


def run_user_simulation_evaluation(sim_llm: LLM, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate ONE user-interview per unique user, rating all K recommendations.

    The CSV from phase A has `top_k` (usually 5) consecutive rows per user.
    Earlier versions of this function treated each row as an independent user,
    which is wrong: a real evaluator reads a person's full recommendation list
    in one sitting. Here we group by "User ID", emit one prompt per user with
    all K (rec, explanation) pairs, and fan the K parsed score tuples back
    out to the user's K row indices.
    """
    k = CFG.top_k
    sim_regex = _build_sim_output_regex(k)
    # Regex-guided decoding: the LLM is FORCED to emit exactly K labeled blocks.
    guided_config = {
        "structured_outputs": StructuredOutputsParams(regex=sim_regex),
    }
    sampling_params = SamplingParams(
        temperature=CFG.sim_temperature,
        max_tokens=CFG.sim_max_tokens,
        top_p=0.9,
        **guided_config,
    )

    # Group rows by user WITHOUT reshuffling — preserves the original order of
    # recommendations in the CSV so the output rows line up with "Movie 1..K"
    # in the prompt.
    user_groups: list[tuple[object, pd.DataFrame]] = list(
        eval_df.groupby("User ID", sort=False)
    )
    total_users = len(user_groups)

    # We now measure the batch size in USERS, not rows. Keep roughly the same
    # total prompt count per LLM call as before (user_batch_size rows) so we
    # don't OOM on the scheduler.
    users_per_batch = max(1, CFG.user_batch_size // k)

    print(
        f"\n--- Starting User-Simulation Interview "
        f"({total_users} users × {k} recs = {total_users * k} rows; "
        f"{users_per_batch} users per LLM batch) ---"
    )

    for start_user in range(0, total_users, users_per_batch):
        end_user = min(start_user + users_per_batch, total_users)
        batch_groups = user_groups[start_user:end_user]
        batch_start = time.perf_counter()

        prompts: list[str] = []
        row_index_batches: list[list[int]] = []  # one list-of-indices per user
        user_labels: list[str] = []

        for user_token, group_df in batch_groups:
            # RecBole stores user ids as string tokens. Convert to the internal
            # integer id so format_user_history can do its lookup.
            try:
                uid_key: int | object = int(user_token) if pd.notna(user_token) else user_token
            except (TypeError, ValueError):
                uid_key = user_token

            rating_history = (
                format_user_history(uid_key) if isinstance(uid_key, int) else "No history."
            )

            rec_titles = group_df["Recommended Movie"].tolist()
            explanations = group_df["Explanation Text (LLM)"].fillna("").tolist()

            # Guard: if for any reason a user has fewer than K rows in the CSV,
            # pad with blank lines so the regex (which expects K blocks) still
            # matches — then NaN-truncate after parsing.
            if len(rec_titles) < k:
                pad = k - len(rec_titles)
                rec_titles = rec_titles + ["(n/a)"] * pad
                explanations = explanations + [""] * pad
            elif len(rec_titles) > k:
                rec_titles = rec_titles[:k]
                explanations = explanations[:k]

            numbered_recs = _format_numbered_recs(rec_titles, explanations)

            prompts.append(SIM_PROMPT_TEMPLATE.format(
                rating_history=rating_history,
                numbered_recs=numbered_recs,
            ))
            # `group_df.index` is the position in the original eval_df we must
            # write scores back to.
            row_index_batches.append(list(group_df.index[:k]))
            user_labels.append(str(user_token))

        print(
            f"\n🗣️  Interviewing simulated users {start_user} to {end_user} "
            f"({len(prompts)} users, {sum(len(r) for r in row_index_batches)} rows)..."
        )
        outputs = sim_llm.generate(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=True
        )

        # For each user: parse K (or fewer) score tuples and write them back
        # into the user's row indices.
        for out, row_indices, user_label in zip(outputs, row_index_batches, user_labels):
            text = out.outputs[0].text.strip()
            scored = parse_user_simulation_multi(text, k)

            # Emit parse failures (any NaN in any block) for auditability.
            failures = [i for i, tup in enumerate(scored) if any(np.isnan(x) for x in tup)]
            if failures:
                print(
                    f"\n⚠️  [WARNING] Simulator parse failures for user {user_label} "
                    f"at block(s) {[i + 1 for i in failures]}."
                )
                print(f"Raw LLM Output:\n{text}")
                print("-" * 50)

            # Row i of the user (0..k-1) receives score tuple i.
            for i, df_idx in enumerate(row_indices):
                if i >= len(scored):
                    break
                m, s, c, t, p, w = scored[i]
                eval_df.at[df_idx, "Sim_MentalModel"] = m
                eval_df.at[df_idx, "Sim_Satisfaction"] = s
                eval_df.at[df_idx, "Sim_Clarity"] = c
                eval_df.at[df_idx, "Sim_Trust"] = t
                eval_df.at[df_idx, "Sim_Persuasion"] = p
                eval_df.at[df_idx, "Sim_WouldWatch"] = w

        elapsed = time.perf_counter() - batch_start
        print(
            f"✅ Completed simulation user-batch {start_user} to {end_user} "
            f"in {elapsed:.2f} seconds!"
        )

        del prompts, outputs, row_index_batches, user_labels, batch_groups
        gc.collect()

    return eval_df


def print_final_simulation_averages(eval_df: pd.DataFrame) -> None:
    print("\n==================================================")
    print(" 🗣️  SIMULATED USER-INTERVIEW AVERAGES")
    print(" (Mohseni et al. dimensions; 1-5 Likert scale)")
    print("==================================================")
    print(f"Mean Mental-Model Clarity: {eval_df['Sim_MentalModel'].mean():.2f} / 5")
    print(f"Mean Satisfaction:          {eval_df['Sim_Satisfaction'].mean():.2f} / 5")
    print(f"Mean Clarity:               {eval_df['Sim_Clarity'].mean():.2f} / 5")
    print(f"Mean Trust:                 {eval_df['Sim_Trust'].mean():.2f} / 5")
    print(f"Mean Persuasion:            {eval_df['Sim_Persuasion'].mean():.2f} / 5")
    watch_rate = eval_df["Sim_WouldWatch"].mean()
    print(f"Would-Watch Rate:           {watch_rate:.2%}")
    print("==================================================\n")


# ============================================================
# Orchestration — two subprocess-isolated phases
# ============================================================
# We deliberately do NOT run the generator and the judge/simulator in the same
# Python process. vLLM's first model leaves a residual CUDA context + worker
# buffers (~1 GiB on small GPUs) that `destroy_model_parallel` +
# `empty_cache()` cannot fully reclaim. The only reliable way to release it
# is for the owning process to exit. Each phase therefore lives in its own
# subprocess; the CSV on disk is the hand-off medium.

def phase_generate() -> None:
    """Phase A — BPR top-K + Qwen generator + per-row metrics. Writes CSV."""
    setup_nltk()

    config, model, dataset = load_bpr_and_embeddings()
    build_movie_metadata(dataset)
    build_interaction_lookups(dataset)

    total_users_count = dataset.user_num - 1
    print(f"Total number of unique users: {total_users_count}")
    ensure_output_csv_header()

    all_user_ids = list(range(1, dataset.user_num))

    # Phase 1: top-K predictions ------------------------------
    topk_predictions = predict_top_k_for_all_users(config, model, all_user_ids)

    # Phase 2: evict PyTorch ----------------------------------
    print("🧹 PHASE 2: Evicting PyTorch to free up VRAM for vLLM...")
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: generator --------------------------------------
    llm = init_generator_llm()
    run_generation_and_metrics(llm, dataset, all_user_ids, topk_predictions)
    calculate_global_metrics(CFG.output_csv)

    # Subprocess exit below fully releases the CUDA context used by Qwen,
    # so the 'evaluate' subprocess starts with a clean GPU.


def phase_evaluate() -> None:
    """Phase B — Judge + simulated-user interview. Enriches CSV in place."""
    # We need user2items + movie_title_map for the prompt builders, so we
    # reload the RecBole dataset. The BPR model weights are immediately
    # discarded; we never move them to GPU in this phase.
    _config, model, dataset = load_bpr_and_embeddings()
    del model
    gc.collect()

    build_movie_metadata(dataset)
    build_interaction_lookups(dataset)

    # Phase 5/6: judge ----------------------------------------
    judge_llm = init_judge_llm()
    eval_df = run_judge_evaluation(judge_llm)
    #eval_df = pd.read_csv(CFG.output_csv)
    # Phase 7: simulated-user interviews ----------------------
    sim_llm, owns_sim = init_simulator_llm(judge_llm)
    eval_df = run_user_simulation_evaluation(sim_llm, eval_df)

    # Tidy the LLM(s) before exit (not strictly needed — process exit will
    # do it — but keeps memory graphs clean for anyone who runs this phase
    # interactively in a notebook).
    if owns_sim:
        try:
            destroy_model_parallel()
        except Exception:  # noqa: BLE001
            pass
        del sim_llm
    del judge_llm
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 8: save & report ----------------------------------
    print("\n💾 Saving final evaluated dataset to CSV...")
    eval_df.to_csv(CFG.output_csv, index=False)
    print_final_judge_averages(eval_df)
    print_final_simulation_averages(eval_df)


def main() -> None:
    """Run each GPU phase in a fresh Python subprocess.

    This is the 'cleanest' VRAM fix for small GPUs: the only guaranteed way
    to release the CUDA context after vLLM tears down is for the owning
    process to terminate.
    """
    import subprocess
    import sys

    for phase in ("generate", "evaluate"):
        banner = f"=== Launching '{phase}' phase in a fresh Python subprocess ==="
        print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}\n")
        subprocess.run(
            [sys.executable, __file__, f"--phase={phase}"],
            check=True,
        )


if __name__ == "__main__":
    import argparse

    calculate_global_metrics(CFG.output_csv)

    parser = argparse.ArgumentParser(
        description="Two-phase recsys XAI pipeline (subprocess-isolated).",
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "evaluate", "all"],
        default="generate",
        help=(
            "'generate' = BPR top-K + Qwen explanations + per-row metrics "
            "(writes the base CSV). "
            "'evaluate' = judge + simulated-user interview "
            "(enriches the existing CSV). "
            "'all' = both, each in a fresh subprocess (recommended)."
        ),
    )
    args = parser.parse_args()

    if args.phase == "generate":
        phase_generate()
    elif args.phase == "evaluate":
        phase_evaluate()
    else:  # "all"
        main()