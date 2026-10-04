"""
Identify and characterize cold-start items in MovieLens-32M.

A cold-start item is a movie that appears in movies.csv (the catalogue)
but has zero ratings in ratings.csv.

Usage:
    python find_cold_start_items.py --data_dir /path/to/ml-32m --out_dir ./figures
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "figure.figsize": (5.5, 3.4),
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_dir", required=True, type=Path)
#     p.add_argument("--out_dir", default=Path("./figures"), type=Path)
#     p.add_argument("--chunksize", default=2_000_000, type=int)
#     return p.parse_args()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=ROOT / "dataset" / "ml-32m", type=Path)
    p.add_argument("--out_dir", default=ROOT / "figures", type=Path)
    p.add_argument("--chunksize", default=2_000_000, type=int)
    return p.parse_args()

def extract_year(title):
    if not isinstance(title, str):
        return None
    m = YEAR_RE.search(title.strip())
    return int(m.group(1)) if m else None


def collect_rated_ids(ratings_path, chunksize):
    """Stream ratings.csv and return the set of movieIds with >=1 rating."""
    rated = set()
    n_ratings = 0
    for chunk in pd.read_csv(ratings_path, chunksize=chunksize,
                             usecols=["movieId"]):
        rated.update(chunk["movieId"].values.tolist())
        n_ratings += len(chunk)
    return rated, n_ratings


def summarize_by_genre(df, n_total):
    exploded = df.assign(genres=df["genres"].str.split("|")).explode("genres")
    return exploded["genres"].value_counts() / n_total * 100


def fig_cold_vs_warm_by_year(cold_years, warm_years, out_path):
    bins = np.arange(1900, 2025, 5)
    fig, ax = plt.subplots()
    ax.hist(warm_years, bins=bins, alpha=0.55, label=f"Warm ($\\geq$1 rating)",
            color="#3a6ea5", edgecolor="black", linewidth=0.3, density=True)
    ax.hist(cold_years, bins=bins, alpha=0.65, label="Cold-start (0 ratings)",
            color="#a5363a", edgecolor="black", linewidth=0.3, density=True)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Density")
    ax.set_title("Release-year distribution: cold-start vs. warm items")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(">> Reading movies.csv ...")
    movies = pd.read_csv(args.data_dir / "movies.csv")
    catalogue_ids = set(movies["movieId"].values.tolist())
    n_catalogue = len(catalogue_ids)
    print(f"   Catalogue size: {n_catalogue:,}")

    print(">> Streaming ratings.csv ...")
    rated_ids, n_ratings = collect_rated_ids(
        args.data_dir / "ratings.csv", args.chunksize
    )
    print(f"   Ratings: {n_ratings:,} interactions over {len(rated_ids):,} items")

    cold_ids = catalogue_ids - rated_ids
    warm_ids = catalogue_ids & rated_ids
    n_cold = len(cold_ids)
    n_warm = len(warm_ids)

    print(f"\n=== Cold-start items: {n_cold:,} "
          f"({n_cold / n_catalogue * 100:.2f}% of catalogue) ===")
    print(f"=== Warm items:       {n_warm:,} "
          f"({n_warm / n_catalogue * 100:.2f}% of catalogue) ===\n")

    cold_movies = movies[movies["movieId"].isin(cold_ids)].copy()
    warm_movies = movies[movies["movieId"].isin(warm_ids)].copy()

    cold_movies["year"] = cold_movies["title"].apply(extract_year)
    warm_movies["year"] = warm_movies["title"].apply(extract_year)

    # --- Persist the cold-start list -----------------------------------------
    out_csv = args.out_dir / "ml32m_cold_start_items.csv"
    cold_movies.sort_values("movieId").to_csv(out_csv, index=False)
    print(f">> Full cold-start list written to {out_csv}")

    # --- Genre breakdown -----------------------------------------------------
    print("\n--- Genre share among cold-start items (top 12) ---")
    cold_share = summarize_by_genre(cold_movies, n_cold)
    warm_share = summarize_by_genre(warm_movies, n_warm)
    table = pd.DataFrame({"cold_%": cold_share, "warm_%": warm_share})
    table["delta_pp"] = table["cold_%"] - table["warm_%"]
    table = table.sort_values("cold_%", ascending=False)
    for genre, row in table.head(12).iterrows():
        print(f"  {genre:25s}  cold {row['cold_%']:5.2f}%  "
              f"warm {row['warm_%']:5.2f}%  delta {row['delta_pp']:+5.2f}pp")
    print("\n  (Positive delta = over-represented among cold-start items)")

    # --- Year analysis -------------------------------------------------------
    cold_years = cold_movies["year"].dropna()
    warm_years = warm_movies["year"].dropna()
    print(f"\n--- Release year of cold-start items ---")
    print(f"  Year known for {len(cold_years):,}/{n_cold:,} cold items "
          f"({len(cold_years)/n_cold*100:.1f}%)")
    if len(cold_years):
        print(f"  Min / median / max: "
              f"{int(cold_years.min())} / {int(cold_years.median())} / "
              f"{int(cold_years.max())}")
        decades = (cold_years // 10 * 10).astype(int).value_counts().sort_index()
        print("  Decade breakdown:")
        for decade, count in decades.items():
            pct = count / len(cold_years) * 100
            bar = "#" * int(pct / 2)
            print(f"    {decade}s: {count:5,} ({pct:5.2f}%)  {bar}")

    # --- Recent-vs-old split (proxy for "will warm up" vs "permanent tail") --
    if len(cold_years):
        cutoff = 2020
        recent = (cold_years >= cutoff).sum()
        old = (cold_years < cutoff).sum()
        print(f"\n  Cold items released >= {cutoff}: {recent:,} "
              f"({recent/len(cold_years)*100:.1f}% of dated cold items)")
        print(f"  Cold items released <  {cutoff}: {old:,} "
              f"({old/len(cold_years)*100:.1f}% of dated cold items)")
        print(f"  --> The first group is plausibly 'not yet rated';")
        print(f"      the second is structurally obscure.")

    # --- Sample examples -----------------------------------------------------
    print(f"\n--- 10 random cold-start examples ---")
    sample = cold_movies.sample(min(10, n_cold), random_state=42)
    for _, row in sample.iterrows():
        title = row["title"][:55] if isinstance(row["title"], str) else "(none)"
        print(f"  [{row['movieId']:>7}] {title:<55} | {row['genres']}")

    # --- Figure --------------------------------------------------------------
    if len(cold_years) and len(warm_years):
        fig_path = args.out_dir / "ml32m_cold_vs_warm_year.jpg"
        fig_cold_vs_warm_by_year(cold_years, warm_years, fig_path)
        print(f"\n>> Year-distribution figure written to {fig_path}")


if __name__ == "__main__":
    main()