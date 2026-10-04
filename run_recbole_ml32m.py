# import logging
#
# from recbole.config import Config
# from recbole.data import create_dataset, data_preparation
# from recbole.utils import init_logger, get_model, get_trainer
#
# import torch
#
# _original_torch_load = torch.load
#
# def torch_load_compat(*args, **kwargs):
#     kwargs.setdefault("weights_only", False)
#     return _original_torch_load(*args, **kwargs)
#
# torch.load = torch_load_compat
#
#
# def run_ml32m():
#     config = Config(
#         model="BPR",
#         config_file_list=["config/ml-32m.yaml"]
#     )
#
#     # Initialize RecBole logger
#     init_logger(config)
#     logger = logging.getLogger()
#
#     logger.info("Running Recbole on MovieLens-32M")
#
#     # Dataset
#     dataset = create_dataset(config)
#     logger.info(dataset)
#
#     train_data, valid_data, test_data = data_preparation(config, dataset)
#
#     # Model
#     model = get_model(config["model"])(
#         config, train_data.dataset
#     ).to(config["device"])
#
#     # Trainer
#     trainer = get_trainer(
#         config["MODEL_TYPE"],
#         config["model"]
#     )(config, model)
#     print("Valid metric:", config["valid_metric"])
#     print("Metrics:", config["metrics"])
#     print("TopK:", config["topk"])
#
#     # Train
#     best_valid_score, best_valid_result = trainer.fit(
#         train_data,
#         valid_data,
#         saved=True
#     )
#
#     # Test
#     test_result = trainer.evaluate(test_data)
#
#     logger.info(f"Best validation result: {best_valid_result}")
#     logger.info(f"Test result: {test_result}")
#
#     return test_result
#
#
# if __name__ == "__main__":
#     torch.set_num_threads(8)
#     run_ml32m()

import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())



import logging
import torch

# ---- SciPy compatibility patch for RecBole (SciPy >= 1.11) ----
import scipy.sparse as sp

if not hasattr(sp.dok_matrix, "_update"):
    def _update(self, data_dict):
        # Replicate old SciPy private behavior
        for (i, j), v in data_dict.items():
            self[i, j] = v

    sp.dok_matrix._update = _update


from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer

# ---- PyTorch compatibility fix (PyTorch 2.1+) ----
_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat






# def run_ml32m():
#     NUM_CORES = os.cpu_count()
#
#     torch.set_num_threads(NUM_CORES)
#     torch.set_num_interop_threads(NUM_CORES)
#
#     # config = Config(
#     #     model="SLIMElastic",
#     #     config_file_list=["config/ml-32m.yaml"]
#     # )
#
#     config = Config(
#         model="LightGCN",
#         config_file_list=["config/ml-32m.yaml"],
#         config_dict={
#             "use_gpu": True,
#             "gpu_id": 0,
#             #"enable_amp": True,
#             "num_workers": 4,
#             "train_batch_size": 16384,  # Increase from 4096 to speed up training
#             #"train_batch_size": 8192,  # Increase from 4096 to speed up training
#             #"train_batch_size": 32768,
#             #"pin_memory": True,  # [NEW] Speeds up CPU -> GPU data transfer
#             #"eval_batch_size": 200000,
#             "eval_batch_size": 50000,  # Keep this stable
#             "data_path": "dataset/",
#             "dataset": "ml-32m",
#             "field_separator": "\t",
#             "eval_args":{
#               "mode": "uni50"
#             },
#             # "load_col": {
#             #     "inter": ["user_id", "item_id"]
#             # },
#
#             "load_col": {
#                 "inter": ["user_id", "item_id", "timestamp"]
#             },
#             #"train_neg_sample_args": None,
#
#             # --- THE FIX ---
#             # Default is 50. Lowering to 20 reduces RAM usage by 60%.
#             "MAX_ITEM_LIST_LENGTH": 20,
#             #"eval_step": 5,
#             #"eval_step": 10,  # Check only every 10 epochs (Speed optimization)
#             #"stopping_step": 15,
#             #"stopping_step": 20,  # Patience: Stop if no improvement after 2 checks (20 epochs)
#             #"persistent_workers": True,
#             "eval_step": 20,  # Check every ~50 minutes (20 * 150s)
#             "stopping_step": 2,  # Give it 2 checks (40 epochs) to fail before quitting
#             "epochs":1,
#         }
#         # config_dict={
#         #     "use_gpu": True,
#         #     "gpu_id": 0,
#         #     "data_path": "dataset/",
#         #     "train_neg_sample_args": None,
#         #     #  "train_batch_size": 32768,
#         #     #  "eval_batch_size": 32768,
#         #     # "worker": 0,
#         #     #
#         #     # # --- THE SPEED FIX ---
#         #     # # Instead of ranking against ALL items (Full),
#         #     # # rank against the real item + 100 random negatives.
#         #     # "eval_args": {
#         #     #     "split": {"RS": [0.8, 0.1, 0.1]},
#         #     #     "group_by": "user",
#         #     #     "order": "TO",
#         #     #      "mode": "uni100"  # <--- CHANGED from 'full' to 'uni100' (Uniform sampling 100 negatives)
#         #     #  },
#         #
#         #     #Optional: Only evaluate every 5 epochs to save even more time
#         #     #"eval_step": 5,
#         #     }
#     )
#
#     init_logger(config)
#     logger = logging.getLogger()
#
#     logger.info(f"Running Recbole on MovieLens-32M using {NUM_CORES} CPU cores")
#
#     dataset = create_dataset(config)
#     logger.info(dataset)
#
#     train_data, valid_data, test_data = data_preparation(config, dataset)
#
#     model = get_model(config["model"])(
#         config, train_data.dataset
#     ).to(config["device"])
#
#     trainer = get_trainer(
#         config["MODEL_TYPE"],
#         config["model"]
#     )(config, model)
#
#     print("Valid metric:", config["valid_metric"])
#     print("Metrics:", config["metrics"])
#     print("TopK:", config["topk"])
#
#     best_valid_score, best_valid_result = trainer.fit(
#         train_data, valid_data, saved=True
#     )
#
#     test_result = trainer.evaluate(test_data)
#
#     logger.info(f"Best validation result: {best_valid_result}")
#     logger.info(f"Test result: {test_result}")
#
#     return test_result


# if __name__ == "__main__":
#     run_ml32m()

import logging
import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# ---- SciPy compatibility patch ----
import scipy.sparse as sp
if not hasattr(sp.dok_matrix, "_update"):
    def _update(self, data_dict):
        for (i, j), v in data_dict.items():
            self[i, j] = v
    sp.dok_matrix._update = _update

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer

# (torch.load compat patch is already applied above at lines 96-101 — do not redefine,
# or torch.load ends up calling itself recursively during checkpoint reload.)

# ---- KnowledgeBasedDataLoader.dataset alias ----
# RecBole's AbstractDataLoader inherits torch.utils.data.DataLoader, which
# exposes .dataset for free; KnowledgeBasedDataLoader does NOT inherit from it
# and only stores self._dataset. RecBole's own eval_collector.data_collect
# (and a few other internal sites) call train_data.dataset, which then
# AttributeErrors for KG models. Add a public alias so the internals work.
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader
if not hasattr(KnowledgeBasedDataLoader, "dataset"):
    KnowledgeBasedDataLoader.dataset = property(lambda self: self._dataset)

import numpy as np
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



# ---- Collector.get_data_struct int-vs-tensor fix (RecBole issue #2131) ----
# Metrics like ItemCoverage / GiniIndex / ShannonEntropy register
# 'data.num_items' as a plain int. The stock get_data_struct calls .cpu() on
# every collected value, which crashes on that int. Only coerce tensors.
import copy
from recbole.evaluator.collector import Collector

def _get_data_struct_tensor_safe(self):
    for key in self.data_struct._data_dict:
        value = self.data_struct._data_dict[key]
        if isinstance(value, torch.Tensor):
            self.data_struct._data_dict[key] = value.cpu()
    returned_struct = copy.deepcopy(self.data_struct)
    for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
        if key in self.data_struct:
            del self.data_struct[key]
    return returned_struct

Collector.get_data_struct = _get_data_struct_tensor_safe

#LightGCN - for some reason e mai rapid pe CPU
def run_ml32m():
    # 1. Configuration: Enable GPU here
    # config = Config(
    #     # Switch to a DL model to actually use GPU (SLIMElastic is CPU-only)
    #     model="LightGCN",
    #     config_file_list=["config/ml-32m.yaml"],
    #      config_dict={
    #          "use_gpu": True,
    #          "gpu_id": 0,                # Select specific GPU
    #          "data_path": "dataset/",    # Ensure this points to your data
    #      }
    # )
    # config = Config(
    #     model="LightGCN",
    #     config_file_list=["config/ml-32m.yaml"],
    #     # config_dict={
    #     #     "use_gpu": True,
    #     #     "gpu_id": 0,
    #     #     "data_path": "dataset/",
    #     #
    #     #     # --- Memory & Speed Optimizations ---
    #     #     "embedding_size": 32,  # Half the default; massive VRAM saver
    #     #     "n_layers": 2,  # Reduces the number of matrix multiplications
    #     #     "train_batch_size": 2048,  # Smaller batches to avoid 100% VRAM saturation
    #     #
    #     #     # --- Skipping heavy evaluation ---
    #     #     "eval_step": 10,  # Only evaluate every 10 epochs
    #     #     "metrics": ["Recall"],  # Only track one metric for speed
    #     #     "topk": [10],  # Limit ranking depth
    #     #
    #     #     # --- Data Loading ---
    #     #     "train_num_workers": 4,  # Parallelize data prep
    #     # }
    # )
    # config = Config(
    #     model="BPR",
    #     config_file_list=["config/ml-32m.yaml"],
    #     config_dict={
    #         "use_gpu": True,
    #         "gpu_id": 0,
    #         "data_path": "dataset/",
    #         "train_batch_size": 32768,
    #         "eval_batch_size": 32768,
    #         "worker": 0,
    #
    #         # --- THE SPEED FIX ---
    #         # Instead of ranking against ALL items (Full),
    #         # rank against the real item + 100 random negatives.
    #         "eval_args": {
    #             "split": {"RS": [0.8, 0.1, 0.1]},
    #             "group_by": "user",
    #             "order": "TO",
    #             "mode": "uni100"  # <--- CHANGED from 'full' to 'uni100' (Uniform sampling 100 negatives)
    #         },
    #
    #         # Optional: Only evaluate every 5 epochs to save even more time
    #         "eval_step": 5,
    #     }
    # )
    # config = Config(
    #     model="FPMC",
    #     config_file_list=["config/ml-32m.yaml"],
    #     config_dict={
    #         "use_gpu": True,
    #         "gpu_id": 0,
    #         "data_path": "dataset/",
    #         "train_batch_size": 32768,
    #         "eval_batch_size": 32768,
    #         "worker": 0,
    #
    #         # --- THE SPEED FIX ---
    #         # Instead of ranking against ALL items (Full),
    #         # rank against the real item + 100 random negatives.
    #         "eval_args": {
    #             "split": {"RS": [0.8, 0.1, 0.1]},
    #             "group_by": "user",
    #             "order": "TO",
    #             "mode": "uni100"  # <--- CHANGED from 'full' to 'uni100' (Uniform sampling 100 negatives)
    #         },
    #
    #         # Optional: Only evaluate every 5 epochs to save even more time
    #         "eval_step": 5,
    #         # # --- ADD THIS TO GENERATE THE 'label' FIELD ---
    #         # "threshold": {"rating": 3.0},  # Ratings >= 3 become label 1, others become 0
    #         "train_neg_sample_args": None,  # Disable sampling since threshold provides labels
    #
    #     }
    # )

    # config = Config(
    #     model="SpectralCF",
    #     config_file_list=["config/ml-32m.yaml"],
    #     config_dict={
    #         "use_gpu": True,
    #         "gpu_id": 0,
    #         "data_path": "dataset/",
    #         "accumulation_steps": 8,
    #         "train_batch_size": 4096,
    #         "eval_batch_size": 4096,
    #         "worker": 0,
    #         "selected_features": ["genres"],
    #         "eval_args": {
    #             "split": {"RS": [0.8, 0.1, 0.1]},
    #             "group_by": "user",
    #             "order": "TO",
    #             "mode": "uni100"
    #         },
    #         "eval_step": 1,
    #     }
    # )

    # STAMP: sequential recommender (uses TIME_FIELD).
    # Memory-friendly: embeddings + attention over a per-user item sequence.
    # No negative sampling needed —


    #RepeatNet dureaza foarte mult! Va trebui sa iau in calcul sa incerc pe o platforma de cloud pe o alta placa video sau sa il las balta
    #Update la RepeatNet: fu antrenat, dar ramasai fara memorie la evaluare.....genial

    # Switched off GRU4RecKG: despite its name, it doesn't consume .kg/.link
    # triples — it expects a precomputed per-item embedding matrix in ml-32m.ent
    # (loaded as preload_weight). Building that requires training a separate KG
    # embedding model first.
    #
    # KGAT was the natural fallback (consumes .kg/.link directly), but in this
    # env KGAT crashes during graph construction: it imports DGL, DGL 2.1.0
    # imports torchdata.datapipes, and torchdata 0.11.0 removed that submodule.
    # KGCN is the same family of model — graph aggregation over the KG — but
    # uses dataset.kg_graph(form="coo") (scipy sparse) instead of DGL, so it
    # avoids the broken import path entirely. Run build_ml32m_kg.py once first.

    #de investigat daca merge mai repede SLIMElastic pe un alt GPU. De rulat, ruleaza.
    #desi nu am incercat inca la EASE, RaCT, RecVAE, CDAE, MacridVAE, MultiDAE, MultiVAE, NCL, SGL, SpectralCF, GCMC,
    # DGCF, NGCF posibil sa fie aceeasi problema

    model_name = "FPMC"   #de investigat daca chiar merge sau nu FEARec. pana acuma am avut probleme cu memoria. Update: Acuma nu e problema cu memoria ci cu
                            #timpul necesar doar pentru UN EPOCH
    config_dict = {
        "use_gpu": True,
        "gpu_id": 0,
        "data_path": "dataset/",
        "worker": 0,
        # Aggressive filter: drop users/items with <20 interactions.
        # Cuts ML-32M (~32M rows) to ~10M rows so STAMP sequence build fits in RAM.
        #"user_inter_num_interval": "[20,inf)",
        #"item_inter_num_interval": "[20,inf)",
        "loss_type": "CE",
        "train_neg_sample_args": None,  # CE loss => no negative sampling
        # Smoke test: 1 train epoch + 1 valid eval (every eval_step) + 1 final test eval.
        "epochs": 1,
        "eval_step": 1,
        # GRU4RecF/KG default to selected_features=["class"] (LFM-1b leftover).
        # ML-32M has no "class" field — point at genres (token_seq) instead.
        "selected_features": ["genres"]
    }

    # RecBole "general" recommenders: classic CF over the user-item matrix —
    # no sequences, no KG, no side features required. All consume only .inter.
    GENERAL_MODELS = {
        "Pop", "ItemKNN", "Random",
        "BPR", "NeuMF", "ConvNCF", "DMF", "FISM", "NAIS",
        "SpectralCF", "GCMC", "NGCF", "LightGCN", "DGCF",
        "LINE", "MultiVAE", "MultiDAE", "MacridVAE", "CDAE",
        "ENMF", "NNCF", "RaCT", "RecVAE",
        "EASE", "SLIMElastic", "ADMMSLIM", "NCEPLRec",
        "SGL", "SimpleX", "NCL", "DiffRec", "LDiffRec",
    }

    # Sequential recommenders build per-user item sequences in data_preparation;
    # default MAX_ITEM_LIST_LENGTH=50 OOMs on ML-32M (~31M rows). Cap to 10.
    SEQUENTIAL_MODELS = {
        "STAMP", "GRU4Rec", "SASRec", "NARM", "BERT4Rec", "FOSSIL",
        "Caser", "NextItNet", "FPMC", "SRGNN", "TransRec",
        "HRM", "RepeatNet", "GRU4RecF", "GRU4RecKG",
        "GRU4RecCPR", "SASRecCPR", "CORE", "FEARec", "LightSANs", "SINE",
    }
    if model_name in SEQUENTIAL_MODELS:
        config_dict["MAX_ITEM_LIST_LENGTH"] = 10

    DECISIONTREE_MODELS = {"XGBoost", "LightGBM"}
    if model_name in DECISIONTREE_MODELS:
        # RecBole looks for properties/model/XGBoost.yaml but the file on disk is
        # xgboost.yaml (lowercase). On case-sensitive filesystems (WSL/Linux) the
        # model defaults never load -> xgb_num_boost_round is None -> xgboost.train
        # crashes on range(0, None). Supply the defaults explicitly.
        config_dict["xgb_num_boost_round"] = 100
        config_dict["convert_token_to_onehot"] = False  # keep off: 200k users x 87k items would OOM
        config_dict["xgb_verbose_eval"] = 50
        config_dict["xgb_params"] = {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": 6,
            "eta": 0.1,
            "seed": 2020,
            "tree_method": "hist",
            "device": "cuda",  # GPU. Remove this line (and use CPU) if device isn't recognized
        }

    CONTEXT_MODELS = {
        "LR", "FM", "FFM", "FwFM", "FmFM", "AFM", "NFM", "PNN",
        "DeepFM", "DCN", "DCNV2", "xDeepFM", "WideDeep", "AutoInt",
        "FNN", "DSSM", "FiGNN", "FiBiNET", "KD_DAGFM", "EulerNet",
        "DIN", "DIEN", "XGBoost", "LightGBM"
    }

    if model_name in CONTEXT_MODELS:
        config_dict["threshold"] = {"rating": 4.0}
        config_dict["eval_args"] = {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "TO",
            "group_by": None,
            "mode": "labeled",
        }
        config_dict["metrics"] = ["AUC", "LogLoss","RMSE","MAE"]
        config_dict["valid_metric"] = "AUC"
        config_dict["topk"] = [10, 20]




    # Models that support full-softmax CE loss (no negative sampling needed).
    # Everything else (TransRec, FPMC, Caser, HRM, BPR, LightGCN, ...) uses
    # pairwise BPR and crashes with KeyError: 'neg_item_id' if neg sampling is off.
    # GRU4RecCPR/SASRecCPR are CE-only by design (their config says so).
    CE_LOSS_MODELS = {
        "STAMP", "GRU4Rec", "SASRec", "NARM", "BERT4Rec", "HRM", "HGN", "S3Rec",
        "NextItNet", "SRGNN", "GRU4RecF", "GRU4RecKG", "SHAN", "SASRecF",
        "GRU4RecCPR", "SASRecCPR", "CORE", "FEARec", "LightSANs", "FOSSIL", "NPE", "KSR", "Caser", "GCSAN"
    }
    if model_name not in CE_LOSS_MODELS:
        config_dict.pop("loss_type", None)
        config_dict.pop("train_neg_sample_args", None)

    if model_name == "KSR":
        config_dict["additional_feat_suffix"] = ["ent", "rel"]  # -> ml-32m.ent, ml-32m.rel
        config_dict["alias_of_entity_id"] = ["ent_id"]
        config_dict["alias_of_relation_id"] = ["rel_id"]
        config_dict["preload_weight"] = {"ent_id": "ent_emb", "rel_id": "rel_emb"}

    # Models whose full_sort_predict materializes (batch, n_items, dim) — e.g.
    # TransRec repeats the item-embedding matrix per user, so the default
    # eval_batch_size=4096 blows past VRAM on ML-32M (~12 GB just for that tensor).
    HEAVY_EVAL_MODELS = {"TransRec"}
    if model_name in HEAVY_EVAL_MODELS:
        config_dict["eval_batch_size"] = 1024 #de la 1024 par sa fie probleme, dar mai incerc o singura data

    # selected_features only makes sense for models that consume .item side info
    # via the FeatureSeqEmbLayer (the *F / *KG sequential variants and the FM
    # family). KGAT/KGCN/CFKG/etc. consume the KG instead — leaving the key in
    # for them is harmless but misleading.
    FEATURE_AWARE_MODELS = {"GRU4RecF", "GRU4RecKG"}
    if model_name not in FEATURE_AWARE_MODELS:
        config_dict.pop("selected_features", None)

    # GRU4RecKG calls dataset.get_preload_weight("ent_id") to seed an
    # nn.Embedding(n_items, embedding_size). This needs:
    #   - additional_feat_suffix=[ent]  -> load ml-32m.ent
    #   - alias_of_item_id=[ent_id]    -> share the token vocabulary with
    #                                     item_id, so field2id_token['ent_id']
    #                                     gets populated during remap (without
    #                                     this, _preload_weight_matrix raises
    #                                     KeyError: 'ent_id')
    #   - preload_weight={ent_id: ent_emb} -> wire the float_seq column into
    #                                         the preload matrix; the model
    #                                         then slices [:n_items]
    # Run train_kg_embeddings.py once before this so ml-32m.ent exists.
    if model_name == "GRU4RecKG":
        config_dict["additional_feat_suffix"] = ["ent"]
        config_dict["alias_of_item_id"] = ["ent_id"]
        config_dict["preload_weight"] = {"ent_id": "ent_emb"}

    # if model_name == "BERT4Rec":
    #     config_dict["MAX_ITEM_LIST_LENGTH"] = 5

    config = Config(
        model=model_name,
        config_file_list=["config/ml-32m.yaml"],
        config_dict=config_dict,
    )

    # Initialize Logger
    init_logger(config)
    logger = logging.getLogger()

    # Log device status
    logger.info(f"RecBole is running on device: {config['device']}")

    # Dataset
    dataset = create_dataset(config)
    logger.info(dataset)

    # Data Preparation
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model Initialization
    # .to(config['device']) moves the model to GPU automatically if configured above.
    model = get_model(config["model"])(
        config, train_data.dataset
    ).to(config["device"])

    logger.info(f"Model structure:\n{model}")

    # Trainer
    trainer = get_trainer(
        config["MODEL_TYPE"],
        config["model"]
    )(config, model)

    print("Valid metric:", config["valid_metric"])
    print("Metrics:", config["metrics"])
    print("TopK:", config["topk"])

    # Train
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True
    )

    # Evaluate
    test_result = trainer.evaluate(test_data)

    logger.info(f"Best validation result: {best_valid_result}")
    logger.info(f"Test result: {test_result}")

    return test_result

if __name__ == "__main__":
    # Remove aggressive CPU thread locking if using GPU
    run_ml32m()
