import pandas as pd
import numpy as np
import gc

# 1. Update this to the exact path of your original ml-32m.inter file
input_path = "dataset/ml-32m/ml-32m.inter"
base_path = "dataset/ml-32m/ml-32m"

print("1. Loading dataset into memory...")
# We load it, but Pandas is much more efficient at storing strings than RecBole's internal classes
df = pd.read_csv(input_path, sep='\t')

# Find the exact column names (RecBole uses suffixes like 'user_id:token')
user_col = [col for col in df.columns if col.startswith('user_id')][0]
time_col = [col for col in df.columns if col.startswith('timestamp')][0]

print("2. Sorting by User and Time (Time-Ordered)...")
df.sort_values(by=[user_col, time_col], inplace=True)

print("3. Calculating 80/10/10 split boundaries...")
# Vectorized calculation: incredibly fast and low RAM
user_counts = df.groupby(user_col).cumcount() + 1
total_counts = df.groupby(user_col)[user_col].transform('count')
ratios = user_counts / total_counts

# Create masks for the splits
train_mask = ratios <= 0.8
valid_mask = (ratios > 0.8) & (ratios <= 0.9)
test_mask = ratios > 0.9

print("4. Saving Train, Valid, and Test files...")
# Save train and free up RAM
df[train_mask].to_csv(f"{base_path}.train.inter", sep='\t', index=False)
del train_mask
gc.collect()

# Save valid and free up RAM
df[valid_mask].to_csv(f"{base_path}.valid.inter", sep='\t', index=False)
del valid_mask
gc.collect()

# Save test
df[test_mask].to_csv(f"{base_path}.test.inter", sep='\t', index=False)

print("\nSuccess! Files created:")
print(f"- {base_path}.train.inter")
print(f"- {base_path}.valid.inter")
print(f"- {base_path}.test.inter")