import pandas as pd
from pathlib import Path

RAW_DIR = Path("dataset/ml32m_raw")
OUT_DIR = Path("dataset/ml-32m")

OUT_DIR.mkdir(exist_ok=True)

# -------- INTERACTIONS --------
ratings = pd.read_csv(RAW_DIR / "ratings.csv")

ratings = ratings.rename(columns={
    "userId": "user_id:token",
    "movieId": "item_id:token",
    "rating": "rating:float",
    "timestamp": "timestamp:float"
})

ratings.to_csv(
    OUT_DIR / "ml-32m.inter",
    sep="\t",
    index=False
)

# -------- ITEMS --------
movies = pd.read_csv(RAW_DIR / "movies.csv")

movies = movies.rename(columns={
    "movieId": "item_id:token",
    "title": "title:token_seq",
    "genres": "genres:token_seq"
})

movies.to_csv(
    OUT_DIR / "ml-32m.item",
    sep="\t",
    index=False
)

print("✅ Recbole ML-32M files generated successfully")
