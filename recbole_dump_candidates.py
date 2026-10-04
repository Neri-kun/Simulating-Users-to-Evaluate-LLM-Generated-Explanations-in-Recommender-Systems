"""
Stage 1 of the LLM re-ranking pipeline.

Train a RecBole recommender on MovieLens-32M, then for every TEST user dump:
  - the model's top-N candidate items (items the user has NOT interacted with),
  - a few of the user's highly-rated TRAIN movies (context for the LLM),
  - a binary relevance vector (1 if a candidate is in the user's held-out
    test interactions, else 0)  <-- this is the ground truth for re-ranking.

Output: candidates.json  (consumed by llm_rerank_dashboard.py)

The base recommender's own ranking is recorded as `base_order` so the LLM
re-ranking can be scored *against the model it is re-ranking*.
"""

import os
os.environ["OMP_NUM_THREADS"]      = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"]      = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"]  = str(os.cpu_count())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# ----------------------------------------------------------------------
# Compatibility patches (same ones your training script relies on)
# ----------------------------------------------------------------------
import scipy.sparse as sp
if not hasattr(sp.dok_matrix, "_update"):
    def _update(self, data_dict):
        for (i, j), v in data_dict.items():
            self[i, j] = v
    sp.dok_matrix._update = _update

_original_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer
from recbole.utils.case_study import full_sort_topk

# ItemKNN CSR fix (kept so the script works if you pick ItemKNN)
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.general_recommender import itemknn as _itemknn_mod
from recbole.model.general_recommender.itemknn import ComputeSimilarity

def _itemknn_init_csr(self, config, dataset):
    GeneralRecommender.__init__(self, config, dataset)
    self.k = config["k"]
    self.shrink = config["shrink"] if "shrink" in config else 0.0
    self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
    assert self.n_users == self.interaction_matrix.shape[0]
    assert self.n_items == self.interaction_matrix.shape[1]
    _, self.w = ComputeSimilarity(
        self.interaction_matrix, topk=self.k, shrink=self.shrink
    ).compute_similarity("item")
    self.pred_mat = self.interaction_matrix.dot(self.w).tocsr()
    self.fake_loss = torch.nn.Parameter(torch.zeros(1))
    self.other_parameter_name = ["w", "pred_mat"]
_itemknn_mod.ItemKNN.__init__ = _itemknn_init_csr

# ======================================================================
# Settings you will likely touch
# ======================================================================
MODEL_NAME       = "LightGCN"   # any general recommender that supports full_sort_topk
DATASET_NAME     = "ml-32m"
DATA_PATH        = "dataset/"
N_CANDIDATES     = 50           # top-N recommendations dumped per user
N_HISTORY        = 10           # how many liked train titles to include as LLM context
MAX_TEST_USERS   = 3000         # cap for a fast run; set None to dump every test user
TOPK_BATCH       = 256          # users scored per full_sort batch (memory/speed knob)
OUTPUT_JSON      = "candidates.json"


def load_item_titles(config):
    """Read readable titles from the atomic .item file (falls back to movies.csv)."""
    item_path = os.path.join(DATA_PATH, DATASET_NAME, f"{DATASET_NAME}.item")
    try:
        sep = config["field_separator"] or "\t"
        df = pd.read_csv(item_path, sep=sep, engine="python")
        # atomic columns look like 'item_id:token', 'title:token_seq'
        df.columns = [c.split(":")[0] for c in df.columns]
        return dict(zip(df["item_id"].astype(str), df["title"].astype(str)))
    except Exception as e:
        logging.warning(f"Could not read {item_path} ({e}); trying movies.csv")
        try:
            df = pd.read_csv(os.path.join(DATA_PATH, DATASET_NAME, "movies.csv"))
            return dict(zip(df["movieId"].astype(str), df["title"].astype(str)))
        except Exception as e2:
            logging.warning(f"No title source found ({e2}); using raw ids as titles")
            return {}


def build_history(train_data, dataset, title_of, n_hist, rating_field):
    """Per internal user id -> list of their highest-rated TRAIN titles."""
    feat = train_data.dataset.inter_feat
    uid_f, iid_f = dataset.uid_field, dataset.iid_field
    users = feat[uid_f].numpy()
    items = feat[iid_f].numpy()
    if rating_field and rating_field in feat.columns:
        ratings = feat[rating_field].numpy()
    else:
        ratings = np.ones(len(users))  # no rating loaded -> treat all as liked

    liked = defaultdict(list)
    for u, i, r in zip(users, items, ratings):
        liked[int(u)].append((float(r), int(i)))

    history = {}
    for u, pairs in liked.items():
        pairs.sort(reverse=True)                      # highest rated first
        top = pairs[:n_hist]
        titles = [title_of.get(dataset.id2token(iid_f, i), dataset.id2token(iid_f, i))
                  for _, i in top]
        history[u] = titles
    return history


def run():
    config = Config(
        model=MODEL_NAME,
        config_file_list=[f"config/{DATASET_NAME}.yaml"],
        config_dict={
            "use_gpu": True,
            "gpu_id": 0,
            "data_path": DATA_PATH,
            "dataset": DATASET_NAME,
            "worker": 0,
            # keep full ranking so top-N candidates are drawn from the whole catalog
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "TO",
                "mode": "full",
            },
            # a short run is fine — we only need a trained model to rank with
            "epochs": 50,
            "eval_step": 10,
            "stopping_step": 2,
        },
    )

    init_logger(config)
    logger = logging.getLogger()
    logger.info(f"Device: {config['device']}  Model: {MODEL_NAME}")

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    logger.info("Training base recommender...")
    trainer.fit(train_data, valid_data, saved=True)

    # ------------------------------------------------------------------
    # Extract candidates + ground truth
    # ------------------------------------------------------------------
    model.eval()
    uid_f, iid_f = dataset.uid_field, dataset.iid_field

    title_of = load_item_titles(config)
    history  = build_history(train_data, dataset, title_of, N_HISTORY,
                             rating_field=config["RATING_FIELD"])

    # ground-truth positives per internal user, from the TEST split
    tf = test_data.dataset.inter_feat
    gt = defaultdict(set)
    for u, i in zip(tf[uid_f].numpy(), tf[iid_f].numpy()):
        gt[int(u)].add(int(i))

    test_users = np.unique(tf[uid_f].numpy())
    if MAX_TEST_USERS is not None:
        test_users = test_users[:MAX_TEST_USERS]
    logger.info(f"Dumping candidates for {len(test_users)} test users...")

    records = []
    device = config["device"]
    for start in range(0, len(test_users), TOPK_BATCH):
        chunk = test_users[start:start + TOPK_BATCH]
        # full_sort_topk masks out train/valid items -> genuine recommendations
        _, topk_iid = full_sort_topk(chunk, model, test_data, k=N_CANDIDATES, device=device)
        topk_iid = topk_iid.cpu().numpy()

        for row, u_int in enumerate(chunk.tolist()):
            cand_int = topk_iid[row].tolist()
            cand_titles = [title_of.get(dataset.id2token(iid_f, i),
                                        dataset.id2token(iid_f, i)) for i in cand_int]
            rel = [1 if i in gt[u_int] else 0 for i in cand_int]
            records.append({
                "user":         dataset.id2token(uid_f, u_int),
                "candidates":   cand_titles,
                "base_order":   list(range(N_CANDIDATES, 0, -1)),  # model rank (desc)
                "relevance":    rel,
                "history_liked": history.get(u_int, []),
            })

        if (start // TOPK_BATCH) % 5 == 0:
            logger.info(f"  {min(start + TOPK_BATCH, len(test_users))}/{len(test_users)} users")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "n_candidates": N_CANDIDATES, "records": records}, f)

    n_hits = sum(1 for r in records if any(r["relevance"]))
    logger.info(f"Wrote {OUTPUT_JSON}: {len(records)} users, "
                f"{n_hits} with >=1 relevant candidate in top-{N_CANDIDATES}.")


if __name__ == "__main__":
    run()