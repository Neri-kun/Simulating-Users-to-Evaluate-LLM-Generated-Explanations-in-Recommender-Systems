"""
Compute MovieLens-32M descriptive statistics and render LaTeX-ready figures.

Usage:
    python compute_ml32m_stats.py --data_dir /path/to/ml-32m --out_dir ./figures

Expected files inside --data_dir:
    movies.csv   ratings.csv   tags.csv   links.csv
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # PhDMovieDashboard
DATA = ROOT / "dataset" / "ml-32m"

# ----- Plot style: paper-friendly, no seaborn dependency ---------------------
plt.rcParams.update({
    "figure.figsize": (5.0, 3.2),
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def parse_args():
    p = argparse.ArgumentParser()
    #p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--data_dir", default=DATA, type=Path)
    p.add_argument("--out_dir", default=Path("../figures"), type=Path)
    p.add_argument("--chunksize", default=2_000_000, type=int)
    return p.parse_args()


def compute_rating_stats(ratings_path, chunksize):
    """Stream ratings.csv and accumulate per-user, per-item, and value counts."""
    user_counts = Counter()
    item_counts = Counter()
    value_counts = Counter()
    n_ratings = 0
    min_ts, max_ts = None, None

    for chunk in pd.read_csv(ratings_path, chunksize=chunksize):
        n_ratings += len(chunk)
        user_counts.update(chunk["userId"].values.tolist())
        item_counts.update(chunk["movieId"].values.tolist())
        value_counts.update(chunk["rating"].values.tolist())
        cmin, cmax = chunk["timestamp"].min(), chunk["timestamp"].max()
        min_ts = cmin if min_ts is None else min(min_ts, cmin)
        max_ts = cmax if max_ts is None else max(max_ts, cmax)

    return {
        "n_ratings": n_ratings,
        "user_counts": user_counts,
        "item_counts": item_counts,
        "value_counts": value_counts,
        "min_ts": min_ts,
        "max_ts": max_ts,
    }


def compute_genre_stats(movies_path):
    movies = pd.read_csv(movies_path)
    # Genres are pipe-separated
    exploded = movies.assign(genres=movies["genres"].str.split("|")).explode("genres")
    return exploded["genres"].value_counts(), len(movies)


def fig_rating_distribution(value_counts, out_path):
    keys = sorted(value_counts.keys())
    vals = [value_counts[k] for k in keys]
    total = sum(vals)
    pcts = [v / total * 100 for v in vals]

    fig, ax = plt.subplots()
    bars = ax.bar([str(k) for k in keys], pcts,
                  color="#3a6ea5", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Rating value (stars)")
    ax.set_ylabel("Share of ratings (%)")
    ax.set_title("Rating value distribution")
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{p:.1f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def fig_long_tail(item_counts, out_path):
    counts = np.sort(np.array(list(item_counts.values())))[::-1]
    ranks = np.arange(1, len(counts) + 1)

    fig, ax = plt.subplots()
    ax.loglog(ranks, counts, color="#a5363a", linewidth=1.2)
    ax.set_xlabel("Item rank (log scale)")
    ax.set_ylabel("Number of ratings (log scale)")
    ax.set_title("Item popularity long-tail")

    # Annotate the head/tail split (top 20% of items)
    head_cut = int(0.2 * len(counts))
    head_share = counts[:head_cut].sum() / counts.sum() * 100
    ax.axvline(head_cut, color="grey", linestyle=":", linewidth=0.8)
    ax.text(head_cut * 1.15, counts.max() * 0.4,
            f"Top 20% of items\naccount for {head_share:.1f}%\nof all ratings",
            fontsize=8, color="grey")
    fig.savefig(out_path)
    plt.close(fig)


def fig_user_activity(user_counts, out_path):
    counts = np.array(list(user_counts.values()))
    bins = np.logspace(np.log10(max(counts.min(), 1)),
                       np.log10(counts.max()), 40)
    fig, ax = plt.subplots()
    ax.hist(counts, bins=bins, color="#4a8a4a", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ratings per user (log scale)")
    ax.set_ylabel("Number of users (log scale)")
    ax.set_title("Per-user activity distribution")

    median = np.median(counts)
    mean = counts.mean()
    ax.axvline(median, color="black", linestyle="--", linewidth=0.8,
               label=f"median = {median:.0f}")
    ax.axvline(mean, color="red", linestyle="--", linewidth=0.8,
               label=f"mean = {mean:.0f}")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def fig_genre_distribution(genre_counts, n_movies, out_path):
    # Drop the "(no genres listed)" bucket from the visual if you prefer;
    # keeping it makes the data-quality issue visible.
    gc = genre_counts.sort_values(ascending=True)
    pct = gc / n_movies * 100

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.barh(pct.index, pct.values, color="#7a5aa5",
            edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Share of movies tagged with this genre (%)")
    ax.set_title("Genre coverage across the catalogue")
    for i, v in enumerate(pct.values):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=7)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(">> Streaming ratings.csv ...")
    r = compute_rating_stats(args.data_dir / "ratings.csv", args.chunksize)

    print(">> Reading movies.csv ...")
    genre_counts, n_movies = compute_genre_stats(args.data_dir / "movies.csv")

    n_users = len(r["user_counts"])
    n_items_rated = len(r["item_counts"])
    n_ratings = r["n_ratings"]
    density = n_ratings / (n_users * n_movies)

    # ---- Print a compact summary for the LaTeX table -----------------------
    print("\n=== MovieLens-32M descriptive statistics ===")
    print(f"Users (with >=1 rating):     {n_users:,}")
    print(f"Movies in catalogue:         {n_movies:,}")
    print(f"Movies with >=1 rating:      {n_items_rated:,}")
    print(f"Cold-start movies (0 rat.):  {n_movies - n_items_rated:,}")
    print(f"Ratings:                     {n_ratings:,}")
    print(f"Density:                     {density*100:.4f}%")
    print(f"Sparsity:                    {(1-density)*100:.4f}%")
    uc = np.array(list(r['user_counts'].values()))
    ic = np.array(list(r['item_counts'].values()))
    print(f"Ratings per user  (mean/med/min/max): "
          f"{uc.mean():.1f} / {np.median(uc):.0f} / {uc.min()} / {uc.max()}")
    print(f"Ratings per movie (mean/med/min/max): "
          f"{ic.mean():.1f} / {np.median(ic):.0f} / {ic.min()} / {ic.max()}")
    print(f"First rating timestamp:      {pd.to_datetime(r['min_ts'], unit='s')}")
    print(f"Last rating timestamp:       {pd.to_datetime(r['max_ts'], unit='s')}")
    print(f"Mean rating value:           "
          f"{sum(k*v for k,v in r['value_counts'].items())/n_ratings:.3f}")

    # ---- Figures -----------------------------------------------------------
    print("\n>> Rendering figures ...")
    fig_rating_distribution(r["value_counts"], args.out_dir / "ml32m_rating_dist.jpg")
    fig_long_tail(r["item_counts"], args.out_dir / "ml32m_long_tail.jpg")
    fig_user_activity(r["user_counts"], args.out_dir / "ml32m_user_activity.jpg")
    fig_genre_distribution(genre_counts, n_movies, args.out_dir / "ml32m_genres.jpg")
    print(f">> Done. Figures written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()