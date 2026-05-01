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

# ---- PyTorch compatibility fix ----
_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat



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

    config = Config(
        model="SpectralCF",
        config_file_list=["config/ml-32m.yaml"],
        config_dict={
            "use_gpu": True,
            "gpu_id": 0,
            "data_path": "dataset/",
            # --- THE PRE-SPLIT OVERRIDE ---
            #"benchmark_filename": ['train', 'valid', 'test'],
            "accumulation_steps": 8,
            "train_batch_size": 4096,  # This makes the REAL batch 16,384 (4096 * 4)
            "eval_batch_size": 4096,
            #"train_batch_size": 16384,  # Lowered for stability on 32M dataset
            #"eval_batch_size": 16384,
            "worker": 0,
            # --- THE FIX FOR KEYERROR ---
            #"train_neg_sample_args":None,
            # "train_neg_sample_args": {
            #     "strategy": "by",  # Sample negatives by a specific distribution
            #     "by": 1,  # 1 negative for every 1 positive interaction
            #     "distribution": "uniform"
            # },
            # --- THE FIX FOR THE KEYERROR FOR GRU4RecF---
            "selected_features": ["genres"],
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "TO",
                "mode": "uni100"
            },

            "eval_step": 1,
        }
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
    # .to(config['device']) moves the model to GPU automatically if configured above
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
