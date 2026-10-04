"""
Build RecBole knowledge-graph atomic files for MovieLens-32M.

Outputs (written into --dataset-dir, default dataset/ml-32m/):
  ml-32m.kg    head_id:token  relation_id:token  tail_id:token
  ml-32m.link  item_id:token  entity_id:token

Triples produced:
  (m_<movieId>, directed_by, <nconst>)   from IMDb title.crew + name.basics
  (m_<movieId>, has_genre,   g_<genre>)  from ml-32m.item

Items with no triples are still emitted in .link so the entity_id namespace
stays aligned with item_id; KGAT (etc.) then falls back on the user-item
bipartite signal for those movies.

Usage:
  python build_ml32m_kg.py
  python build_ml32m_kg.py --dataset-dir dataset/ml-32m --imdb-dir .
  python build_ml32m_kg.py --skip-download   # IMDb .tsv.gz already present
"""

import argparse
import os
import sys

import pandas as pd

# Reuse the IMDb plumbing from the directors build.
from build_movie_directors import (
    CREW_FILE,
    NAMES_FILE,
    download_if_missing,
    load_crew,
)

REL_DIRECTED_BY = "directed_by"
REL_HAS_GENRE = "has_genre"
GENRE_PLACEHOLDER = "(no genres listed)"  # ML-20M+ marker for missing genres

ITEM_ID_COL = "item_id:token"
ITEM_GENRES_COL = "genres:token_seq"


def _movie_entity(movie_id) -> str:
    return f"m_{movie_id}"


def _genre_entity(genre: str) -> str:
    return f"g_{genre}"


def load_item_genres(item_path: str) -> pd.DataFrame:
    """Read a RecBole .item file -> long DataFrame of (movieId, genre)."""
    print(f"Loading {item_path}")
    df = pd.read_csv(item_path, sep="\t", dtype=str)
    if ITEM_ID_COL not in df.columns or ITEM_GENRES_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{ITEM_ID_COL}' and '{ITEM_GENRES_COL}' in {item_path}, "
            f"got: {list(df.columns)}"
        )
    df = df[[ITEM_ID_COL, ITEM_GENRES_COL]].rename(
        columns={ITEM_ID_COL: "movieId", ITEM_GENRES_COL: "genres"}
    )
    df["movieId"] = df["movieId"].astype(int)
    df = df.dropna(subset=["genres"])
    df["genre"] = df["genres"].str.split("|")
    df = df.explode("genre")
    df["genre"] = df["genre"].str.strip()
    df = df[df["genre"].ne("") & df["genre"].ne(GENRE_PLACEHOLDER)]
    print(
        f"  {df['movieId'].nunique():,} movies, "
        f"{df['genre'].nunique()} unique genres, "
        f"{len(df):,} (movie, genre) pairs"
    )
    return df[["movieId", "genre"]]


def load_movie_directors(links_path: str, crew_path: str) -> pd.DataFrame:
    """Cross-reference MovieLens links.csv with IMDb crew -> (movieId, director_nconst)."""
    print(f"Loading {links_path}")
    links = pd.read_csv(links_path)
    # IMDb canonical: tt + zero-padded 7+ digits. Same formatting build_movie_directors uses.
    links["tconst"] = "tt" + links["imdbId"].astype(int).astype(str).str.zfill(7)

    crew = load_crew(crew_path)  # already exploded to one row per (tconst, director_nconst)
    md = links[["movieId", "tconst"]].merge(crew, on="tconst", how="inner")
    md = md.dropna(subset=["director_nconst"])
    md["director_nconst"] = md["director_nconst"].str.strip()
    md = md[md["director_nconst"].ne("")]
    print(
        f"  {md['movieId'].nunique():,} movies with at least one director, "
        f"{len(md):,} (movie, director) pairs"
    )
    return md[["movieId", "director_nconst"]]


def write_kg(triples: pd.DataFrame, out_path: str) -> None:
    out = triples.rename(
        columns={
            "head": "head_id:token",
            "relation": "relation_id:token",
            "tail": "tail_id:token",
        }
    )
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  wrote {out_path} ({len(out):,} triples)")


def write_link(item_ids, out_path: str) -> None:
    df = pd.DataFrame(
        {
            "item_id:token": list(item_ids),
            "entity_id:token": [_movie_entity(i) for i in item_ids],
        }
    )
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  wrote {out_path} ({len(df):,} item-entity links)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset/ml-32m",
        help="Directory holding ml-32m.item and links.csv; .kg/.link are written here.",
    )
    parser.add_argument(
        "--imdb-dir",
        default=".",
        help="Where to find (or download) name.basics.tsv.gz and title.crew.tsv.gz.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume IMDb files are already present in --imdb-dir.",
    )
    args = parser.parse_args()

    item_path = os.path.join(args.dataset_dir, "ml-32m.item")
    links_path = os.path.join(args.dataset_dir, "links.csv")
    kg_path = os.path.join(args.dataset_dir, "ml-32m.kg")
    link_path = os.path.join(args.dataset_dir, "ml-32m.link")

    for required in (item_path, links_path):
        if not os.path.exists(required):
            sys.exit(f"ERROR: {required} not found")

    if args.skip_download:
        crew_path = os.path.join(args.imdb_dir, CREW_FILE)
        names_path = os.path.join(args.imdb_dir, NAMES_FILE)
        for p in (crew_path, names_path):
            if not os.path.exists(p):
                sys.exit(f"ERROR: --skip-download set but {p} missing")
    else:
        os.makedirs(args.imdb_dir, exist_ok=True)
        crew_path = download_if_missing(CREW_FILE, args.imdb_dir)
        # name.basics is unused here (we don't resolve display names), but the
        # directors script will need it on next run, so prefetch.
        download_if_missing(NAMES_FILE, args.imdb_dir)

    md = load_movie_directors(links_path, crew_path)
    mg = load_item_genres(item_path)

    director_triples = pd.DataFrame(
        {
            "head": [_movie_entity(m) for m in md["movieId"]],
            "relation": REL_DIRECTED_BY,
            "tail": md["director_nconst"].values,
        }
    )
    genre_triples = pd.DataFrame(
        {
            "head": [_movie_entity(m) for m in mg["movieId"]],
            "relation": REL_HAS_GENRE,
            "tail": [_genre_entity(g) for g in mg["genre"]],
        }
    )
    kg = pd.concat([director_triples, genre_triples], ignore_index=True)
    kg = kg.drop_duplicates(ignore_index=True)
    print(
        f"\nKG composed: {len(kg):,} triples "
        f"({len(director_triples):,} directed_by, {len(genre_triples):,} has_genre)"
    )

    all_item_ids = pd.read_csv(item_path, sep="\t", dtype={ITEM_ID_COL: int})[ITEM_ID_COL]

    print("\nWriting outputs:")
    write_kg(kg, kg_path)
    write_link(all_item_ids, link_path)

    n_with_dir = md["movieId"].nunique()
    n_with_genre = mg["movieId"].nunique()
    n_total = all_item_ids.nunique()
    n_entities = pd.concat([kg["head"], kg["tail"]]).nunique()
    print("\nCoverage:")
    print(f"  movies with director triple: {n_with_dir:,}/{n_total:,} ({n_with_dir / n_total:.1%})")
    print(f"  movies with genre triple:    {n_with_genre:,}/{n_total:,} ({n_with_genre / n_total:.1%})")
    print(f"  unique entities:             {n_entities:,}")
    print(f"  unique relations:            {kg['relation'].nunique()}")


if __name__ == "__main__":
    main()
