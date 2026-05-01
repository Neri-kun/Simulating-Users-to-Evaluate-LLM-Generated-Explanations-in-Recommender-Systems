import pandas as pd

# Update this path to exactly where your file is
file_path = "dataset/ml-32m/ml-32m_tab.inter"

try:
    print("Attempting to load the first 5 rows...")
    # This simulates exactly what RecBole tries to do
    df = pd.read_csv(
        file_path,
        delimiter="\t",
        dtype=str,  # Force string to avoid "Complex128" guessing
        nrows=5
    )
    print("Success! Here is how Pandas sees your data:")
    print(df)
    print("\nCheck the columns above. Do they look separated correctly?")
except Exception as e:
    print("\nFAILED. The file is still broken.")
    print(e)