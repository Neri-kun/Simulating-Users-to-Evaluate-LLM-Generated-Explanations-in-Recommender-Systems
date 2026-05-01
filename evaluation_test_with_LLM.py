import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gc
import os
import time
import textstat
import math
import nltk
import concurrent.futures
import random
from collections import Counter
print("Working dir:", os.getcwd())

import platform
import sys

print(platform.system())
print(sys.executable)

# ==========================================
# PyTorch 2.6+ Compatibility Fix
# ==========================================
_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load

from recbole.quick_start import load_data_and_model

# ==========================================
# 1. Global Fast-Lookups
# ==========================================
movie_title_map = {}
item2genres = {}
user2items = {}
item2users = {}  # CRITICAL for User-CF
all_unique_genres_in_dataset = set()

item_embs_np = None
user_embs_np = None  # CRITICAL for User-CF

OUTPUT_CSV = 'all_users_explanations_metrics.csv'
BATCH_SIZE = 5000

# ==========================================
# 2. Ultra-Fast Generators with LOCAL Metrics
# ==========================================
def generate_item_based_explanation(user_id, rec_id):
    history_item_ids = user2items.get(user_id, [])
    if not history_item_ids: return "Because it is highly rated.", 0.0, 0.0, 0.0

    hist_embs = item_embs_np[history_item_ids]
    rec_emb = item_embs_np[rec_id]
    similarities = hist_embs.dot(rec_emb)

    best_idx = np.argmax(similarities)
    best_match_internal_id = history_item_ids[best_idx]
    original_sim_score = similarities[best_idx]

    if len(history_item_ids) > 1:
        mask = np.ones(len(history_item_ids), dtype=bool)
        mask[best_idx] = False
        second_best_sim = np.max(similarities[mask])
        pn_drop = original_sim_score - second_best_sim
    else:
        pn_drop = original_sim_score

    ps_score = original_sim_score

    hist_name = movie_title_map.get(best_match_internal_id, "Unknown")
    rec_name = movie_title_map.get(rec_id, "Unknown")
    text = f"Since you enjoyed '{hist_name}', we think you might really like '{rec_name}'."

    return text, round(original_sim_score.item(), 4), round(pn_drop.item(), 4), round(ps_score.item(), 4)


def generate_user_cf_explanation(user_id, rec_id):
    users_who_watched_rec = list(item2users.get(rec_id, set()) - {user_id})
    rec_name = movie_title_map.get(rec_id, "Unknown")

    if not users_who_watched_rec:
        return f"Viewers are loving '{rec_name}' right now.", 0.0

    # SPEED OPTIMIZATION: Cap the candidate users to 500.
    if len(users_who_watched_rec) > 500:
        users_who_watched_rec = random.sample(users_who_watched_rec, 500)

    cand_embs = user_embs_np[users_who_watched_rec]
    tgt_emb = user_embs_np[user_id]
    similarities = cand_embs.dot(tgt_emb)

    best_idx = np.argmax(similarities)
    most_similar_user_id = users_who_watched_rec[best_idx]
    user_sim_score = similarities[best_idx]

    target_history = set(user2items.get(user_id, []))
    similar_history = set(user2items.get(most_similar_user_id, []))

    shared_items = target_history.intersection(similar_history)

    if shared_items:
        # Get just one shared movie efficiently
        shared_name = movie_title_map.get(next(iter(shared_items)), "Unknown")
        text = f"Users with similar tastes to yours—who also loved '{shared_name}'—highly recommend '{rec_name}'."
    else:
        text = f"Viewers with a watch history similar to yours frequently watched '{rec_name}'."

    return text, round(user_sim_score.item(), 4)


def generate_kg_explanation(user_id, rec_id):
    history_item_ids = user2items.get(user_id, [])
    if not history_item_ids: return f"Recommended {movie_title_map.get(rec_id, 'Unknown')} based on overall popularity.", "None", 0.0, 0.0, 0.0, 0.0

    all_historical_genres = [g for iid in history_item_ids for g in item2genres.get(iid, [])]
    genre_counts = Counter(all_historical_genres)

    rec_genres = item2genres.get(rec_id, [])
    rec_name = movie_title_map.get(rec_id, "Unknown")

    best_link_genre = None
    highest_count = 0

    for genre in rec_genres:
        if genre in genre_counts and genre_counts[genre] > highest_count and genre != "(no genres listed)":
            highest_count = genre_counts[genre]
            best_link_genre = genre

    if not best_link_genre: return f"'{rec_name}' is a great addition.", "None", 0.0, 0.0, 0.0, 0.0

    expl_features = {best_link_genre}
    rec_features = set(rec_genres)
    user_features = set(all_historical_genres)

    fp = len(expl_features.intersection(rec_features)) / len(expl_features)
    fr = len(expl_features.intersection(rec_features)) / len(rec_features) if len(rec_features) > 0 else 0.0
    ff = (2 * fp * fr) / (fp + fr) if (fp + fr) > 0 else 0.0
    fmr = len(expl_features.intersection(user_features)) / len(expl_features)

    anchor_name = "a similar movie"
    for hist_id in history_item_ids:
        if best_link_genre in item2genres.get(hist_id, []):
            anchor_name = movie_title_map.get(hist_id, "Unknown")
            break

    text = f"Because you frequently watch {best_link_genre} movies (like '{anchor_name}'), '{rec_name}' is a perfect match."
    return text, best_link_genre, fp, fr, ff, fmr


# ==========================================
# 3. Thread Pool Worker Function
# ==========================================
def process_single_user(args):
    uid, user_token, topk_rec_ids = args
    user_results = []
    user_start_time = time.perf_counter()

    for rec_id in topk_rec_ids:
        item_text, item_score, pn_score, ps_score = generate_item_based_explanation(uid, rec_id)
        user_text, user_score = generate_user_cf_explanation(uid, rec_id)
        kg_text, expl_genre, fp, fr, ff, fmr = generate_kg_explanation(uid, rec_id)

        try:
            fog = textstat.gunning_fog(kg_text)
            fre = textstat.flesch_reading_ease(kg_text)
            fkgl = textstat.flesch_kincaid_grade(kg_text)
            ari = textstat.automated_readability_index(kg_text)
            smog = textstat.smog_index(kg_text)
        except Exception:
            fog, fre, fkgl, ari, smog = 0.0, 0.0, 0.0, 0.0, 0.0

        user_results.append({
            "User ID": user_token,
            "Recommended Movie": movie_title_map.get(rec_id, "Unknown"),

            "Explanation Text (Item-CF)": item_text,
            "Item-CF Strength": item_score,
            "PN (Necessity Drop)": pn_score,
            "PS (Sufficiency)": ps_score,

            "Explanation Text (User-CF)": user_text,
            "User-CF Strength": user_score,

            "Explanation Text (KG)": kg_text,
            "Expl_Feature": expl_genre,
            "FP (Feature Precision)": round(fp, 4),
            "FR (Feature Recall)": round(fr, 4),
            "FF (Feature F1)": round(ff, 4),
            "FMR (Feature Match Ratio)": round(fmr, 4),

            "Gunning Fog": fog,
            "Flesch Reading Ease": fre,
            "Flesch-Kincaid Grade": fkgl,
            "ARI": ari,
            "SMOG Index": smog
        })

    user_total_time = time.perf_counter() - user_start_time

    print("Time required to generate explanations for user with ID " + str(
        uid) + " : " + f"{user_total_time:.5f}" + " seconds.")

    return user_results


# ==========================================
# 4. Global Evaluation Analyzer
# ==========================================
def calculate_global_metrics(csv_path):
    print("\n==================================================")
    print(" 📊 CALCULATING GLOBAL CORPUS METRICS (XAI & UX)")
    print("==================================================")

    start_time = time.time()

    df = pd.read_csv(csv_path)
    total_explanations = len(df)

    if total_explanations == 0:
        print("Error: No explanations found in the CSV.")
        return

    unique_sentences = df["Explanation Text (Item-CF)"].nunique()
    usr = unique_sentences / total_explanations

    # Exclude invalid features for accurate feature metric calculations
    valid_features = df[df["Expl_Feature"] != "None"]["Expl_Feature"].dropna()
    N_valid = len(valid_features)

    unique_expl_features = valid_features.nunique()
    total_dataset_features = len(all_unique_genres_in_dataset)
    fcr = unique_expl_features / total_dataset_features if total_dataset_features > 0 else 0

    # --- FD Metric 1: Entropy (Global Distribution) ---
    feature_probs = valid_features.value_counts(normalize=True)
    fd_entropy = -sum(p * math.log(p, 2) for p in feature_probs if p > 0)

    # --- FD Metric 2: Paper Version (Pairwise Overlap) ---
    # Optimized for O(N) using frequency permutations to prevent freezing on large CSVs
    if N_valid > 1:
        feature_counts_abs = valid_features.value_counts()
        total_intersections = sum(c * (c - 1) for c in feature_counts_abs)
        fd_pairwise = total_intersections / (N_valid * (N_valid - 1))
    else:
        fd_pairwise = 0.0

    mean_fp = df["FP (Feature Precision)"].mean()
    mean_fr = df["FR (Feature Recall)"].mean()
    mean_ff = df["FF (Feature F1)"].mean()
    mean_fmr = df["FMR (Feature Match Ratio)"].mean()

    mean_pn = df["PN (Necessity Drop)"].mean()
    mean_ps = df["PS (Sufficiency)"].mean()

    df['is_explainable'] = df['Item-CF Strength'] > 0.0
    mep = df['is_explainable'].sum() / total_explanations
    mer = 1.0

    mean_fog = df["Gunning Fog"].mean()
    mean_fre = df["Flesch Reading Ease"].mean()
    mean_fkgl = df["Flesch-Kincaid Grade"].mean()
    mean_ari = df["ARI"].mean()
    mean_smog = df["SMOG Index"].mean()

    print(f"Total Recommendations Evaluated: {total_explanations}\n")
    print("--- 1. Text & Coverage Metrics ---")
    print(f"USR (Unique Sentence Ratio):   {usr:.4f}")
    print(f"FCR (Feature Coverage Ratio):  {fcr:.4f}")
    print(f"FD  (Shannon Entropy):         {fd_entropy:.4f}  <- (Higher is better)")
    print(f"FD  (Paper/Pairwise Overlap):  {fd_pairwise:.4f}  <- (Lower is better)")

    print("\n--- 2. Feature Quality Metrics ---")
    print(f"Mean FP:  {mean_fp:.4f} | Mean FR:  {mean_fr:.4f}")
    print(f"Mean FF:  {mean_ff:.4f} | Mean FMR: {mean_fmr:.4f}")

    print("\n--- 3. Causal Metrics ---")
    print(f"Mean PN (Necessity):   {mean_pn:.4f}")
    print(f"Mean PS (Sufficiency): {mean_ps:.4f}")

    print("\n--- 4. Readability & UX Metrics (Averages) ---")
    print(f"Flesch Reading Ease:   {mean_fre:.2f}")
    print(f"Flesch-Kincaid Grade:  {mean_fkgl:.2f}")
    print(f"Gunning Fog Index:     {mean_fog:.2f}")
    print(f"Automated Read. Index: {mean_ari:.2f}")
    print(f"SMOG Index:            {mean_smog:.2f}")
    print("==================================================\n")


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Verifying NLTK syllable dictionaries...")
    nltk.download('cmudict', quiet=False)
    _ = nltk.corpus.cmudict.dict()

    MODEL_PATH = 'saved/BPR-Jan-18-2026_12-28-14.pth'
    MOVIES_FILE_PATH = 'dataset/ml-32m/movies.csv'

    print("Loading model & precomputing arrays...")
    try:
        config, model, dataset, _, _, _ = load_data_and_model(
            model_file=MODEL_PATH
        )
    except SystemExit:
        print("RecBole attempted to exit — continuing.")

    item_embs_np = F.normalize(model.item_embedding.weight.detach(), p=2, dim=1).cpu().numpy()
    user_embs_np = F.normalize(model.user_embedding.weight.detach(), p=2, dim=1).cpu().numpy()

    movies_df = pd.read_csv(MOVIES_FILE_PATH, sep=',')
    movies_df['movieId'] = movies_df['movieId'].astype(str)

    for _, row in movies_df.iterrows():
        try:
            internal_iid = dataset.token2id(dataset.iid_field, row['movieId'])
            movie_title_map[internal_iid] = row['title']
            genres = row['genres'].split('|') if pd.notna(row['genres']) else []
            item2genres[internal_iid] = genres
            all_unique_genres_in_dataset.update(genres)
        except ValueError:
            pass

    del movies_df
    gc.collect()

    uids = dataset.inter_feat[dataset.uid_field].numpy()
    iids = dataset.inter_feat[dataset.iid_field].numpy()
    total_users_count = dataset.user_num - 1
    print(f"Total number of unique users: {total_users_count}")

    for u, i in zip(uids, iids):
        user2items.setdefault(u, set()).add(i)
        item2users.setdefault(i, set()).add(u)

    for u in user2items:
        user2items[u] = list(user2items[u])

    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=[
            "User ID", "Recommended Movie",
            "Explanation Text (Item-CF)", "Item-CF Strength", "PN (Necessity Drop)", "PS (Sufficiency)",
            "Explanation Text (User-CF)", "User-CF Strength",
            "Explanation Text (KG)", "Expl_Feature",
            "FP (Feature Precision)", "FR (Feature Recall)", "FF (Feature F1)", "FMR (Feature Match Ratio)",
            "Gunning Fog", "Flesch Reading Ease", "Flesch-Kincaid Grade", "ARI", "SMOG Index"
        ]).to_csv(OUTPUT_CSV, index=False)

    all_internal_users = list(range(1, dataset.user_num))
    max_threads = min(32, os.cpu_count() * 2)

    print(f"\n--- Starting Evaluation ({max_threads} Threads) ---")
    model = model.to(config['device'])
    model.eval()

    for start_idx in range(0, len(all_internal_users), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(all_internal_users))
        batch_uids = all_internal_users[start_idx:end_idx]

        with torch.no_grad():
            uid_tensor = torch.tensor(batch_uids).to(config['device'])
            user_embs = model.user_embedding(uid_tensor)
            scores = torch.matmul(user_embs, model.item_embedding.weight.T)

        rows, cols = [], []
        for row_idx, uid in enumerate(batch_uids):
            items = user2items.get(uid, [])
            rows.extend([row_idx] * len(items))
            cols.extend(items)

        scores[rows, cols] = -float('inf')
        scores[:, 0] = -float('inf')

        _, topk_indices = torch.topk(scores, k=5)
        topk_indices = topk_indices.cpu().numpy()

        tasks = [(uid, dataset.id2token(dataset.uid_field, uid), topk_indices[i]) for i, uid in enumerate(batch_uids)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            nested_results = list(executor.map(process_single_user, tasks))

        batch_results = [item for sublist in nested_results for item in sublist]
        pd.DataFrame(batch_results).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

        print(f"\n✅ Completed batch from user index {start_idx} to {end_idx}...")
        del scores, user_embs, uid_tensor, batch_results
        gc.collect()

    # Trigger the evaluation on the final CSV
    calculate_global_metrics(OUTPUT_CSV)