"""
Build a movieId -> director mapping for MovieLens-32M using IMDb bulk datasets.

Inputs:
  - links.csv from MovieLens-32M (must be in the same directory as this script,
    or pass --links /path/to/links.csv)

Outputs (written next to this script, or to --outdir):
  - movie_directors.csv   : long-format (movieId, director_nconst, director_name)
  - movies_no_director.csv: movies with no director attribution found
  - director_summary.csv  : director_nconst, director_name, num_movies (descending)

Usage:
  python build_movie_directors.py
  python build_movie_directors.py --links /path/to/links.csv --outdir ./out
  python build_movie_directors.py --skip-download   # if you've already downloaded the .tsv.gz files

Notes:
  - Downloads two files from https://datasets.imdbws.com/ (~350MB total compressed).
  - IMDb bulk dumps are refreshed daily, so re-running on different days may
    produce slightly different results. Record the run date for reproducibility.
  - Movies with multiple credited directors will appear on multiple rows in
    movie_directors.csv. This is intentional.
"""

import argparse
import gzip
import os
import sys
import urllib.request
from datetime import datetime

import pandas as pd

IMDB_BASE = "https://datasets.imdbws.com/"
CREW_FILE = "title.crew.tsv.gz"
NAMES_FILE = "name.basics.tsv.gz"


def download_if_missing(filename: str, outdir: str) -> str:
    """Download an IMDb bulk file to outdir if it isn't already there."""
    path = os.path.join(outdir, filename)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [skip] {filename} already exists ({size_mb:.1f} MB)")
        return path
    url = IMDB_BASE + filename
    print(f"  [download] {url}")
    urllib.request.urlretrieve(url, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [done] saved {filename} ({size_mb:.1f} MB)")
    return path


def load_crew(crew_path: str) -> pd.DataFrame:
    """Load title.crew.tsv.gz and return tconst -> director_nconst (exploded)."""
    print("Loading title.crew (this is the big one — be patient)...")
    crew = pd.read_csv(
        crew_path,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "directors"],
        na_values=r"\N",
        dtype={"tconst": "string", "directors": "string"},
    )
    print(f"  {len(crew):,} title rows loaded")

    # Drop rows with no directors listed
    crew = crew.dropna(subset=["directors"])
    print(f"  {len(crew):,} titles with at least one director")

    # Explode comma-separated director nconsts into one row per (tconst, director)
    crew["directors"] = crew["directors"].str.split(",")
    crew = crew.explode("directors").rename(columns={"directors": "director_nconst"})
    crew["director_nconst"] = crew["director_nconst"].str.strip()
    print(f"  {len(crew):,} (title, director) pairs after explode")
    return crew


def load_names(names_path: str, needed_nconsts: set) -> pd.DataFrame:
    """Load name.basics.tsv.gz, keeping only the nconsts we actually need."""
    print("Loading name.basics (filtering to directors we need)...")
    # We chunk this because the full file is large and we only need a subset
    chunks = []
    reader = pd.read_csv(
        names_path,
        sep="\t",
        compression="gzip",
        usecols=["nconst", "primaryName"],
        na_values=r"\N",
        dtype={"nconst": "string", "primaryName": "string"},
        chunksize=500_000,
    )
    for chunk in reader:
        chunks.append(chunk[chunk["nconst"].isin(needed_nconsts)])
    names = pd.concat(chunks, ignore_index=True)
    names = names.rename(columns={"nconst": "director_nconst", "primaryName": "director_name"})
    print(f"  {len(names):,} director names resolved")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--links", default="links.csv", help="Path to MovieLens links.csv")
    parser.add_argument("--outdir", default=".", help="Directory for downloads and outputs")
    parser.add_argument("--skip-download", action="store_true",
                        help="Assume IMDb files are already present in outdir")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Run started at {datetime.now().isoformat(timespec='seconds')}")
    print(f"Output directory: {os.path.abspath(args.outdir)}")
    print()

    # 1. Get the IMDb files
    if args.skip_download:
        crew_path = os.path.join(args.outdir, CREW_FILE)
        names_path = os.path.join(args.outdir, NAMES_FILE)
        for p in (crew_path, names_path):
            if not os.path.exists(p):
                sys.exit(f"ERROR: --skip-download set but {p} is missing")
    else:
        print("Step 1: Downloading IMDb bulk files (will skip if already present)")
        crew_path = download_if_missing(CREW_FILE, args.outdir)
        names_path = download_if_missing(NAMES_FILE, args.outdir)
    print()

    # 2. Load MovieLens links and format IMDb IDs as tt-prefixed 7-digit strings
    print("Step 2: Loading links.csv")
    if not os.path.exists(args.links):
        sys.exit(f"ERROR: {args.links} not found. Use --links to point at it.")
    links = pd.read_csv(args.links)
    print(f"  {len(links):,} rows in links.csv")
    # IMDb format is tt + zero-padded 7-digit (or wider for very recent titles)
    # The bulk dumps use the same canonical formatting, so we replicate it:
    links["tconst"] = "tt" + links["imdbId"].astype(int).astype(str).str.zfill(7)
    print(f"  formatted imdbId -> tconst (sample: {links['tconst'].head(3).tolist()})")
    print()

    # 3. Load and explode the crew table
    print("Step 3: Loading IMDb crew data")
    crew = load_crew(crew_path)
    print()

    # 4. Join links to crew
    print("Step 4: Joining MovieLens movies to IMDb directors")
    movie_directors = links[["movieId", "tconst"]].merge(crew, on="tconst", how="left")
    print(f"  {len(movie_directors):,} (movieId, director) rows after join")
    print()

    # 5. Resolve director names
    print("Step 5: Resolving director names")
    needed = set(movie_directors["director_nconst"].dropna().unique())
    print(f"  {len(needed):,} unique director nconsts to resolve")
    names = load_names(names_path, needed)
    movie_directors = movie_directors.merge(names, on="director_nconst", how="left")
    print()

    # 6. Split into found/missing and write outputs
    print("Step 6: Writing outputs")
    has_director = movie_directors["director_nconst"].notna()
    found = movie_directors[has_director][["movieId", "director_nconst", "director_name"]].copy()
    missing = movie_directors[~has_director][["movieId", "tconst"]].drop_duplicates()

    found_path = os.path.join(args.outdir, "movie_directors.csv")
    missing_path = os.path.join(args.outdir, "movies_no_director.csv")
    summary_path = os.path.join(args.outdir, "director_summary.csv")

    found.to_csv(found_path, index=False)
    missing.to_csv(missing_path, index=False)

    summary = (
        found.groupby(["director_nconst", "director_name"], dropna=False)
        .size()
        .reset_index(name="num_movies")
        .sort_values("num_movies", ascending=False)
    )
    summary.to_csv(summary_path, index=False)

    print(f"  wrote {found_path}      ({len(found):,} rows)")
    print(f"  wrote {missing_path}    ({len(missing):,} rows)")
    print(f"  wrote {summary_path}    ({len(summary):,} unique directors)")
    print()

    # 7. Quick sanity report
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    n_movies_total = links["movieId"].nunique()
    n_movies_with_director = found["movieId"].nunique()
    coverage = n_movies_with_director / n_movies_total
    print(f"Movies in links.csv:                  {n_movies_total:,}")
    print(f"Movies with at least one director:    {n_movies_with_director:,} ({coverage:.1%})")
    print(f"Total (movieId, director) pairs:      {len(found):,}")
    print(f"Unique directors:                     {found['director_nconst'].nunique():,}")
    print()
    print("Top 10 directors by movie count:")
    print(summary.head(10).to_string(index=False))
    print()
    print(f"Run finished at {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()