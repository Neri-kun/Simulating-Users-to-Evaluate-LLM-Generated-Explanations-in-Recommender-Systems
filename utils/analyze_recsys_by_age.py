"""
analyze_recsys_by_age.py
------------------------
For each age bucket, take the users who "like action" and compute standard
RecBole-style evaluation metrics against the model's held-out test interactions.

Metrics:
    Per-user (averaged across users in each bucket):
        Recall@K, Precision@K, Hit@K, NDCG@K, MRR@K, MAP@K, AUC
    Pooled across all (pred, label) pairs in each bucket:
        MAE, RMSE, LogLoss
    Per-bucket aggregate (across all users in the bucket):
        Coverage@K  - fraction of catalogue recommended to at least one user
        Gini@K      - concentration of recommendation frequency over the FULL
                      catalogue (0 = perfectly uniform, 1 = all mass on one item)
        Entropy@K   - Shannon entropy of recommendation frequency, in bits

Implementation notes:
- Splits are read from the trained model's dataloaders, so the train/valid/test
  partition is identical to what the checkpoint was selected on.
- Ranking metrics rank over the full item catalogue minus each user's
  train+valid items. Test positives remain rankable.
- AUC is rank-based (Mann-Whitney): test positives vs all non-positive unseen
  items, averaged across users.
- MAE / RMSE / LogLoss are not natural metrics for BPR (the score is an
  uncalibrated dot product, not a probability). We follow RecBole's convention
  of applying sigmoid and comparing against binary labels (1 for test
  positives, 0 for sampled negatives, ``--neg-samples`` per positive — default
  50 to mirror the BPR checkpoint's ``uni50`` eval mode). Their absolute values
  should not be compared across models with different score distributions.
- Negatives sampled for MAE/RMSE/LogLoss use this script's RNG, not RecBole's
  internal sampler, so values won't reproduce ``trainer.evaluate(test_data)``
  exactly but should be close in expectation.
- Coverage / Gini / Entropy are computed AFTER the seen-item filter, i.e. an
  item that every user has in train+valid is structurally unreachable here.
  Denominator for Coverage and Gini/Entropy is the non-PAD catalogue size
  (``n_items - 1``).
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import os

# ---- RecBole compat patches ----
import scipy.sparse as sp
if not hasattr(sp.dok_matrix, "_update"):
    def _update(self, data_dict):
        for (i, j), v in data_dict.items():
            self[i, j] = v
    sp.dok_matrix._update = _update

_original_torch_load = torch.load


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "ml-32m"


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat

from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction


AGE_LABELS = {
    1:  "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

LOG_EPS = 1e-12


# ----------------------------- I/O helpers ----------------------------------

def _sep_for(path: Path) -> str:
    return "::" if path.suffix.lower() == ".dat" else ","


def load_users(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  users.csv columns as loaded: {list(df.columns)}")

    def _find(candidates):
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    uid_col = _find(["UserID", "userId", "userid", "user_id"])
    age_col = _find(["Age", "age"])
    if uid_col is None or age_col is None:
        raise SystemExit(
            f"users.csv must contain a user-id column (UserID/userId/user_id) "
            f"and an age column (Age/age). Found columns: {list(df.columns)}"
        )

    rename = {}
    if uid_col != "UserID":
        rename[uid_col] = "UserID"
    if age_col != "Age":
        rename[age_col] = "Age"
    if rename:
        df = df.rename(columns=rename)
        print(f"  renamed columns: {rename}")

    df["UserID"] = df["UserID"].astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df


def load_movies(path: Path) -> pd.DataFrame:
    sep = _sep_for(path)
    if sep == "::":
        df = pd.read_csv(path, sep="::", engine="python",
                         names=["MovieID", "Title", "Genres"], encoding="latin-1")
    else:
        df = pd.read_csv(path, encoding="utf-8")
        rename = {c: c for c in df.columns}
        for src, dst in [("movieId", "MovieID"), ("title", "Title"), ("genres", "Genres")]:
            if src in df.columns:
                rename[src] = dst
        df = df.rename(columns=rename)[["MovieID", "Title", "Genres"]]
    df["MovieID"] = df["MovieID"].astype(int)
    df["is_action"] = df["Genres"].fillna("").apply(lambda g: "Action" in g.split("|"))
    return df


def load_ratings(path: Path) -> pd.DataFrame:
    sep = _sep_for(path)
    if sep == "::":
        df = pd.read_csv(path, sep="::", engine="python",
                         names=["UserID", "MovieID", "Rating", "Timestamp"],
                         encoding="latin-1")
    else:
        df = pd.read_csv(path)
        rename = {c: c for c in df.columns}
        for src, dst in [("userId", "UserID"), ("movieId", "MovieID"),
                         ("rating", "Rating"), ("timestamp", "Timestamp")]:
            if src in df.columns:
                rename[src] = dst
        df = df.rename(columns=rename)[["UserID", "MovieID", "Rating", "Timestamp"]]
    df["UserID"] = df["UserID"].astype(int)
    df["MovieID"] = df["MovieID"].astype(int)
    df["Rating"] = df["Rating"].astype(float)
    return df


def find_action_lovers(ratings, movies, min_action_ratings, min_like_rating):
    action_ids = set(movies.loc[movies.is_action, "MovieID"])
    liked = ratings[ratings.MovieID.isin(action_ids) & (ratings.Rating >= min_like_rating)]
    counts = liked.groupby("UserID").size()
    return set(counts[counts >= min_action_ratings].index.astype(int))


def extract_split_items(data_loader, uid_field, iid_field):
    """Per-user (internal) item array from a dataloader's underlying split."""
    if hasattr(data_loader, "_dataset") and data_loader._dataset is not None:
        ds = data_loader._dataset
    else:
        ds = data_loader.dataset
    feat = ds.inter_feat
    uids = feat[uid_field]
    iids = feat[iid_field]
    if torch.is_tensor(uids):
        uids = uids.cpu().numpy()
        iids = iids.cpu().numpy()
    else:
        uids = np.asarray(uids)
        iids = np.asarray(iids)
    out = defaultdict(list)
    for u, i in zip(uids.astype(np.int64), iids.astype(np.int64)):
        out[int(u)].append(int(i))
    return {u: np.asarray(items, dtype=np.int64) for u, items in out.items()}


# ----------------------------- per-user metrics -----------------------------

def compute_user_metrics(scores_cpu, seen_items, test_pos, topk, n_neg_samples,
                         rng, n_items):
    """
    All metrics for one user.

    scores_cpu : np.ndarray (n_items,)  — raw model scores, no masking applied.
    seen_items : np.ndarray             — internal item ids from train+valid.
    test_pos   : np.ndarray             — internal item ids from test.

    Returns a dict including ``topk_items`` (np.ndarray of internal item ids)
    so the caller can aggregate coverage / Gini / entropy across users.
    """
    n_pos = int(len(test_pos))
    if n_pos == 0:
        return None

    # Mask: PAD + seen items, but never test positives.
    excluded = np.zeros(n_items, dtype=bool)
    excluded[0] = True
    if seen_items.size:
        excluded[seen_items] = True
    excluded[test_pos] = False

    # ----- Ranking metrics -----
    masked = np.where(excluded, -np.inf, scores_cpu)
    if topk < n_items:
        cand = np.argpartition(-masked, topk - 1)[:topk]
        topk_idx = cand[np.argsort(-masked[cand])]
    else:
        topk_idx = np.argsort(-masked)[:topk]

    test_pos_set = set(test_pos.tolist())
    hits = np.fromiter((1.0 if int(i) in test_pos_set else 0.0 for i in topk_idx),
                      dtype=np.float64, count=topk)
    n_hits = float(hits.sum())

    hit_at_k = 1.0 if n_hits > 0 else 0.0
    recall = n_hits / n_pos
    precision = n_hits / topk

    discounts = 1.0 / np.log2(np.arange(2, topk + 2))
    dcg = float((hits * discounts).sum())
    idcg = float(discounts[:min(n_pos, topk)].sum())
    ndcg = dcg / idcg if idcg > 0 else 0.0

    if n_hits > 0:
        first_hit = int(np.argmax(hits))
        mrr = 1.0 / (first_hit + 1)
        cumhits = np.cumsum(hits)
        precs = cumhits / np.arange(1, topk + 1)
        ap = float((precs * hits).sum() / min(n_pos, topk))
    else:
        mrr = 0.0
        ap = 0.0

    # ----- AUC over all unseen items -----
    unseen_mask = ~excluded
    unseen_indices = np.where(unseen_mask)[0]
    unseen_scores = scores_cpu[unseen_indices]
    n_unseen = unseen_indices.size
    n_neg_for_auc = n_unseen - n_pos

    if n_neg_for_auc > 0:
        sort_idx = np.argsort(unseen_scores, kind="stable")
        asc_ranks = np.empty(n_unseen, dtype=np.float64)
        asc_ranks[sort_idx] = np.arange(1, n_unseen + 1)
        pos_pos = np.searchsorted(unseen_indices, test_pos)
        R_pos = float(asc_ranks[pos_pos].sum())
        auc = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg_for_auc)
    else:
        auc = np.nan

    # ----- MAE / RMSE / LogLoss (pooled later) -----
    cand_neg = unseen_indices[~np.isin(unseen_indices, test_pos)]
    n_neg_total = n_pos * n_neg_samples
    if cand_neg.size >= n_neg_total:
        sampled_neg = rng.choice(cand_neg, size=n_neg_total, replace=False)
    elif cand_neg.size > 0:
        sampled_neg = rng.choice(cand_neg, size=n_neg_total, replace=True)
    else:
        sampled_neg = np.empty(0, dtype=np.int64)

    pos_pred = 1.0 / (1.0 + np.exp(-scores_cpu[test_pos]))
    neg_pred = (1.0 / (1.0 + np.exp(-scores_cpu[sampled_neg]))
                if sampled_neg.size else np.empty(0, dtype=np.float64))

    se_sum = float(((pos_pred - 1.0) ** 2).sum() + (neg_pred ** 2).sum())
    ae_sum = float(np.abs(pos_pred - 1.0).sum() + np.abs(neg_pred).sum())
    pos_clip = np.clip(pos_pred, LOG_EPS, 1 - LOG_EPS)
    neg_clip = np.clip(neg_pred, LOG_EPS, 1 - LOG_EPS)
    ll_sum = float(-np.log(pos_clip).sum() - np.log(1 - neg_clip).sum())
    n_pairs = int(len(pos_pred) + len(neg_pred))

    return {
        "hit": hit_at_k, "recall": recall, "precision": precision,
        "ndcg": ndcg, "mrr": mrr, "map": ap, "auc": auc,
        "se_sum": se_sum, "ae_sum": ae_sum, "ll_sum": ll_sum, "n_pairs": n_pairs,
        "topk_items": topk_idx.astype(np.int64),
    }


# ----------------------------- evaluation loop ------------------------------

def evaluate_users(model_path, action_lovers, topk, n_neg_samples, batch_size, seed):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_path,
    )
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    n_items = dataset.item_num
    device = config["device"]

    print("  extracting per-user splits from dataloaders...")
    train_per_user = extract_split_items(train_data, uid_field, iid_field)
    valid_per_user = extract_split_items(valid_data, uid_field, iid_field)
    test_per_user  = extract_split_items(test_data,  uid_field, iid_field)
    print(f"    train users: {len(train_per_user):,} | "
          f"valid users: {len(valid_per_user):,} | "
          f"test users: {len(test_per_user):,}")

    known = dataset.field2token_id[uid_field]
    valid_internal = []
    int_to_ext = {}
    skipped_unknown = 0
    skipped_no_test = 0
    for ext in action_lovers:
        s = str(ext)
        if s not in known:
            skipped_unknown += 1
            continue
        internal = int(dataset.token2id(uid_field, s))
        if internal not in test_per_user or test_per_user[internal].size == 0:
            skipped_no_test += 1
            continue
        valid_internal.append(internal)
        int_to_ext[internal] = int(ext)
    print(f"  {skipped_unknown:,} action lovers unknown to the model (k-core / unseen)")
    print(f"  {skipped_no_test:,} action lovers had no test interactions")
    print(f"  {len(valid_internal):,} users will be evaluated")
    if not valid_internal:
        raise SystemExit("No evaluable users.")

    rng = np.random.default_rng(seed)
    rows = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(valid_internal), batch_size):
            chunk = valid_internal[i:i + batch_size]
            chunk_t = torch.as_tensor(chunk, device=device, dtype=torch.long)
            interaction = Interaction({uid_field: chunk_t}).to(device)
            scores = model.full_sort_predict(interaction).view(len(chunk), n_items)
            scores_cpu_batch = scores.cpu().numpy()

            for j, uid in enumerate(chunk):
                seen = np.concatenate([
                    train_per_user.get(uid, np.empty(0, dtype=np.int64)),
                    valid_per_user.get(uid, np.empty(0, dtype=np.int64)),
                ]) if (uid in train_per_user or uid in valid_per_user) \
                    else np.empty(0, dtype=np.int64)
                test_pos = test_per_user[uid]
                m = compute_user_metrics(
                    scores_cpu_batch[j], seen, test_pos,
                    topk, n_neg_samples, rng, n_items,
                )
                if m is None:
                    continue
                # Serialize topk to a space-separated string so it survives a
                # CSV round-trip (used for coverage / Gini / entropy in
                # summarise()).
                topk_items = m.pop("topk_items")
                m["userId"] = int_to_ext[uid]
                m["topk_items"] = " ".join(str(int(x)) for x in topk_items)
                rows.append(m)

            done = min(i + batch_size, len(valid_internal))
            if (i // batch_size) % 10 == 0 or done == len(valid_internal):
                print(f"  evaluated {done:,}/{len(valid_internal):,} users")

    return pd.DataFrame(rows), int(n_items)


# ----------------------------- aggregation ----------------------------------

def _parse_topk_string(s):
    if isinstance(s, np.ndarray):
        return s.astype(np.int64)
    if isinstance(s, (list, tuple)):
        return np.asarray(s, dtype=np.int64)
    if not isinstance(s, str) or not s.strip():
        return np.empty(0, dtype=np.int64)
    return np.fromstring(s, sep=" ", dtype=np.int64)


def _coverage_gini_entropy(topk_arrays, n_items, topk):
    """
    topk_arrays : iterable of np.ndarray (internal item ids per user).
    Returns (coverage, gini, entropy) for the bucket.

    Gini and entropy are computed over the full non-PAD catalogue, with
    zero-count items included (so an item that no user got recommended still
    contributes to the inequality measure). This is the standard
    "popularity-bias" Gini in recsys.
    """
    n_real = n_items - 1  # exclude PAD (id 0)
    if n_real <= 0:
        return float("nan"), float("nan"), float("nan")

    counts = Counter()
    for arr in topk_arrays:
        if arr.size:
            counts.update(int(x) for x in arr if x != 0)

    unique_recommended = len(counts)
    coverage = unique_recommended / n_real

    freqs = np.zeros(n_real, dtype=np.float64)
    for item_id, c in counts.items():
        if 1 <= item_id < n_items:
            freqs[item_id - 1] = c

    total = freqs.sum()
    if total <= 0:
        return coverage, float("nan"), float("nan")

    freqs_sorted = np.sort(freqs)
    n = n_real
    gini = float(
        (2.0 * np.arange(1, n + 1) - n - 1).dot(freqs_sorted) / (n * total)
    )

    nz = freqs_sorted[freqs_sorted > 0]
    p = nz / total
    entropy = float(-(p * np.log2(p)).sum())

    return coverage, gini, entropy


def summarise(per_user, users_df, topk, n_items):
    users_slim = users_df[["UserID", "Age"]].rename(
        columns={"UserID": "userId", "Age": "age"}
    )
    df = per_user.merge(users_slim, on="userId", how="left")
    df = df.dropna(subset=["age"])
    df["age"] = df["age"].astype(int)

    has_topk = "topk_items" in df.columns
    if has_topk:
        df = df.copy()
        df["_topk_arr"] = df["topk_items"].apply(_parse_topk_string)
    else:
        print("  [warn] per-user data has no 'topk_items' column; "
              "Coverage/Gini/Entropy will be NaN. "
              "Re-run evaluation to populate it.")

    out = []
    for age, grp in df.groupby("age"):
        n_users = len(grp)
        row = {
            "age": int(age),
            "age_label": AGE_LABELS.get(int(age), str(int(age))),
            "n_users": n_users,
            f"Recall@{topk}":    grp["recall"].mean(),
            f"MRR@{topk}":       grp["mrr"].mean(),
            f"NDCG@{topk}":      grp["ndcg"].mean(),
            f"Hit@{topk}":       grp["hit"].mean(),
            f"MAP@{topk}":       grp["map"].mean(),
            f"Precision@{topk}": grp["precision"].mean(),
            "AUC":               grp["auc"].dropna().mean(),
        }
        total_pairs = grp["n_pairs"].sum()
        if total_pairs > 0:
            row["MAE"]     = grp["ae_sum"].sum() / total_pairs
            row["RMSE"]    = math.sqrt(grp["se_sum"].sum() / total_pairs)
            row["LogLoss"] = grp["ll_sum"].sum() / total_pairs
        else:
            row["MAE"] = row["RMSE"] = row["LogLoss"] = float("nan")

        if has_topk and n_items is not None:
            cov, gini, ent = _coverage_gini_entropy(
                grp["_topk_arr"].tolist(), n_items, topk
            )
        else:
            cov = gini = ent = float("nan")
        row[f"Coverage@{topk}"] = cov
        row[f"Gini@{topk}"]     = gini
        row[f"Entropy@{topk}"]  = ent

        out.append(row)

    summary = pd.DataFrame(out).sort_values("age").reset_index(drop=True)
    cols = ["age", "age_label", "n_users",
            f"Recall@{topk}", f"MRR@{topk}", f"NDCG@{topk}", f"Hit@{topk}",
            f"MAP@{topk}", f"Precision@{topk}",
            "AUC", "MAE", "RMSE", "LogLoss",
            f"Coverage@{topk}", f"Gini@{topk}", f"Entropy@{topk}"]
    return summary[cols]


# ----------------------------- meta sidecar ---------------------------------

def _meta_path_for(per_user_path: Path) -> Path:
    return per_user_path.with_suffix(per_user_path.suffix + ".meta.json")


def save_meta(per_user_path: Path, n_items: int, topk: int):
    meta_path = _meta_path_for(per_user_path)
    with open(meta_path, "w") as f:
        json.dump({"n_items": int(n_items), "topk": int(topk)}, f)
    print(f"  saved metadata    -> {meta_path}")


def load_meta(per_user_path: Path):
    meta_path = _meta_path_for(per_user_path)
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


# ----------------------------- main -----------------------------------------

def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--users", type=Path, default=ROOT / "users.csv")
    p.add_argument("--movies",  type=Path, default=DATA / "movies.csv")
    p.add_argument("--ratings", type=Path, default=DATA / "ratings.csv")
    p.add_argument("--model", default=str(ROOT / "saved" / "BPR-Jan-18-2026_12-28-14.pth"),
                   help="path to trained RecBole .pth checkpoint")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--neg-samples", type=int, default=50,
                   help="negatives per test positive for MAE/RMSE/LogLoss")
    p.add_argument("--min-action-ratings", type=int, default=5)
    p.add_argument("--min-like-rating", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", type=Path, default=ROOT / "metrics_by_age.csv")
    p.add_argument("--per-user-out", type=Path, default=ROOT / "per_user_metrics.csv",
                   help="per-user metrics CSV. Saved automatically BEFORE aggregation "
                        "so a crash in summarise() doesn't lose evaluation work. "
                        "A sidecar <name>.meta.json with n_items/topk is written "
                        "alongside it for use with --from-per-user.")
    p.add_argument("--from-per-user", type=Path, default=None,
                   help="skip evaluation and aggregate from a previously saved "
                        "per-user CSV. Will look for a <name>.meta.json sidecar "
                        "to recover n_items; otherwise pass --n-items.")
    p.add_argument("--n-items", type=int, default=None,
                   help="override / fallback for catalogue size when using "
                        "--from-per-user without a meta sidecar. Includes PAD, "
                        "i.e. dataset.item_num.")
    args = p.parse_args()

    print("Loading users...")
    users = load_users(args.users)
    print(f"  {len(users):,} users")

    if args.from_per_user is not None:
        print(f"Loading per-user metrics from {args.from_per_user} "
              f"(skipping evaluation)...")
        per_user = pd.read_csv(args.from_per_user)
        print(f"  {len(per_user):,} per-user rows")

        meta = load_meta(args.from_per_user)
        if meta is not None:
            n_items = int(meta["n_items"])
            print(f"  loaded sidecar meta: n_items={n_items}, topk={meta.get('topk')}")
        elif args.n_items is not None:
            n_items = int(args.n_items)
            print(f"  no sidecar; using --n-items={n_items}")
        else:
            print("  [warn] no meta sidecar and no --n-items; "
                  "Coverage/Gini/Entropy will be NaN")
            n_items = None
    else:
        if args.movies is None or args.ratings is None:
            raise SystemExit("--movies and --ratings are required when not using "
                             "--from-per-user.")
        print("Loading movies...")
        movies = load_movies(args.movies)
        print(f"  {len(movies):,} movies, "
              f"{int(movies.is_action.sum()):,} flagged as Action")

        print("Loading ratings...")
        ratings = load_ratings(args.ratings)
        print(f"  {len(ratings):,} interactions")

        print(f"Finding action-loving users (>= {args.min_action_ratings} ratings "
              f">= {args.min_like_rating})...")
        action_lovers = find_action_lovers(
            ratings, movies,
            min_action_ratings=args.min_action_ratings,
            min_like_rating=args.min_like_rating,
        )
        print(f"  {len(action_lovers):,} action-loving users")

        print(f"Evaluating with model {args.model} (top-{args.topk}, "
              f"{args.neg_samples} negatives per test positive)...")
        per_user, n_items = evaluate_users(
            model_path=args.model,
            action_lovers=action_lovers,
            topk=args.topk,
            n_neg_samples=args.neg_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"  per-user metric rows: {len(per_user):,}")
        print(f"  catalogue size (incl. PAD): {n_items:,}")

        per_user.to_csv(args.per_user_out, index=False)
        print(f"  saved per-user   -> {args.per_user_out}")
        save_meta(args.per_user_out, n_items=n_items, topk=args.topk)

    summary = summarise(per_user, users, args.topk, n_items)

    print("\n=== Metrics by age (action-loving users only) ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary.to_csv(args.out, index=False)
    print(f"\nSaved summary -> {args.out}")


if __name__ == "__main__":
    main()