import torch
import os
from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from pathlib import Path



ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
# --- PYTORCH 2.6 FIX ---
# Save the original torch.load function
_original_torch_load = torch.load

# Create a wrapper that forces weights_only=False
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Override torch.load with our patched version
torch.load = patched_torch_load
# -----------------------

# 1. Load the saved model and the dataset it was trained on
model_path = str(ROOT / "saved" / "BPR-Jan-18-2026_12-28-14.pth")
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_path)

# 2. Force the configuration to calculate Precision at 10 and 20
config['metrics'] = ['Precision']
config['topk'] = [10, 20]

# 3. Initialize the trainer with the updated config and your pre-trained model
trainer = Trainer(config, model)

# 4. Run the evaluation on the test data
print("Evaluating model. This may take a moment depending on your hardware...")
test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

print("\n--- Final Test Results ---")
print(test_result)