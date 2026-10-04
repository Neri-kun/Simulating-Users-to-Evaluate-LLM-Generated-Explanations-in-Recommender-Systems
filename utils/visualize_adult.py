"""
visualize_adult_recbole.py
==========================

Visualize the Adult dataset as integrated in the RecBole framework.

Pipeline
--------
1. Build a RecBole `Config` for the Adult dataset (context-aware setting).
2. Load it via RecBole's Dataset API (`create_dataset`).
3. Pull the interaction frame + side features back into pandas.
4. Produce four families of visualizations:
       (a) Basic statistics (summary table + sparsity heatmap-style bar).
       (b) Interaction distributions (per-user / per-item, log-log long-tail).
       (c) Label distribution (income >= 50k, the binary "interaction" target).
       (d) Feature analysis (age, education, occupation, hours-per-week, ...).

Prerequisites
-------------
    pip install recbole pandas numpy matplotlib seaborn

Dataset layout
--------------
RecBole expects atomic files under  <data_path>/<dataset>/<dataset>.{inter,user,item}.
For the Adult dataset, download / generate:

    dataset/adult/adult.inter      # user_id:token  item_id:token  label:float
    dataset/adult/adult.user       # optional user-side features
    dataset/adult/adult.item       # optional item-side features

(The atomic-file scripts live in the RecSysDatasets repo referenced on the
RecBole datasets page.)

Usage
-----
    python visualize_adult_recbole.py \
        --data_path ./dataset \
        --dataset adult \
        --out ./figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# NumPy 2.0 compatibility shim for RecBole.
# RecBole's `compatibility_settings()` does things like `np.float = np.float_`,
# but NumPy 2.0 removed `np.float_`, `np.int_`-style aliases, etc. Re-create
# them before importing RecBole so its monkey-patches don't crash.
# --------------------------------------------------------------------------- #
for _name, _target in [
    ("float_",   np.float64),
    ("int_",     np.int_ if hasattr(np, "int_") else np.int64),
    ("complex_", np.complex128),
    ("bool_",    np.bool_),
    ("object_",  np.object_),
    ("str_",     np.str_),
    ("unicode_", np.str_),
    ("long",     np.int64),
]:
    if not hasattr(np, _name):
        setattr(np, _name, _target)
# Some RecBole builds also touch the bare aliases.
for _name, _target in [("float", float), ("int", int),
                       ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _target)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

# --------------------------------------------------------------------------- #
# torch.distributed shim.
# Recent RecBole versions call `torch.distributed.barrier()` inside dataset
# loading to coordinate multi-worker downloads. In single-process runs there
# is no default process group, so the barrier crashes with
# "Default process group has not been initialized". Initialize a 1-process
# group on a local TCP socket so the barrier becomes a no-op.
# --------------------------------------------------------------------------- #
import torch                          # noqa: E402
import torch.distributed as _dist     # noqa: E402

if not _dist.is_available():
    # Cover the rare case where torch was built without distributed support.
    _dist.barrier = lambda *a, **kw: None  # type: ignore[attr-defined]
elif not _dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    try:
        _dist.init_process_group(backend="gloo", rank=0, world_size=1)
    except Exception as _e:
        # Fallback: just neutralize the barrier so loading proceeds.
        print(f"[warn] could not init torch.distributed ({_e}); "
              f"patching barrier() to a no-op")
        _dist.barrier = lambda *a, **kw: None  # type: ignore[attr-defined]

# RecBole imports — these are the supported public entry points.
from recbole.config import Config           # noqa: E402
from recbole.data import create_dataset     # noqa: E402
from recbole.utils import init_seed         # noqa: E402


# --------------------------------------------------------------------------- #
# 1.  RecBole bootstrap — schema auto-detection + Config
# --------------------------------------------------------------------------- #
def _ensure_user_item_columns(data_path: str, dataset_name: str) -> str:
    """If the dataset's `.inter` file has no user_id/item_id columns
    (Adult-style CTR datasets), synthesize them and write an augmented copy
    under `<dataset>_aug/`. Returns the dataset name to actually load.
    """
    src_dir   = Path(data_path) / dataset_name
    inter_src = src_dir / f"{dataset_name}.inter"
    header    = _read_atomic_header(inter_src)
    if header is None:
        return dataset_name  # Let the caller surface the missing-file error.

    names = [n for n, _ in header]
    if "user_id" in names and "item_id" in names:
        return dataset_name  # Nothing to do.

    new_name  = f"{dataset_name}_aug"
    dst_dir   = Path(data_path) / new_name
    inter_dst = dst_dir / f"{new_name}.inter"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # If we've already augmented (e.g. on a previous run) and the source
    # hasn't grown, reuse the cached file — but only if the cached file is
    # already in the expected shape (user_id + item_id present, no token_seq
    # columns lingering). Otherwise blow it away so the new conversion runs.
    if inter_dst.exists() and inter_dst.stat().st_mtime >= inter_src.stat().st_mtime:
        cached_header = _read_atomic_header(inter_dst) or []
        cached_names = {n for n, _ in cached_header}
        cached_has_seq = any(t == "token_seq" for _, t in cached_header)
        if (
            "user_id" in cached_names
            and "item_id" in cached_names
            and not cached_has_seq
        ):
            print(f"  [aug] reusing existing {inter_dst}")
            return new_name
        print(f"  [aug] cached {inter_dst} is stale (missing ids or "
              "still contains :token_seq) — regenerating")

    print(f"  [aug] {dataset_name}.inter has no user_id/item_id; "
          f"synthesizing -> {inter_dst}")

    # Read with the original (typed) header preserved as plain column names,
    # then prepend our two synthetic id columns and write back with tabs.
    df = pd.read_csv(inter_src, sep="\t")

    # Convert any :token_seq columns to :token by replacing whitespace in
    # values with '_'. RecBole's remap pipeline for token_seq fields
    # mis-handles single-token sequences in some versions and produces
    # "Length of values (1) does not match length of index (N)" errors.
    # For visualization we don't need the sequence semantics.
    for col in list(df.columns):
        if col.endswith(":token_seq"):
            new_col = col.replace(":token_seq", ":token")
            df[col] = (df[col].astype(str)
                              .str.strip()
                              .str.replace(r"\s+", "_", regex=True))
            df = df.rename(columns={col: new_col})
            print(f"  [aug] converted {col} -> {new_col} "
                  "(whitespace in values -> '_')")

    n  = len(df)
    df.insert(0, "item_id:token", ["adult"] * n)
    df.insert(0, "user_id:token", np.arange(1, n + 1))
    df.to_csv(inter_dst, sep="\t", index=False)

    print(f"  [aug] wrote {n:,} rows with synthetic ids "
          f"(user_id = 1..{n}, item_id = 'adult')")
    return new_name


def _read_atomic_header(path: Path) -> list[tuple[str, str]] | None:
    """Read the first line of a RecBole atomic file and return [(name,type)]."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    out = []
    for tok in header.split("\t"):
        if ":" in tok:
            name, typ = tok.split(":", 1)
        else:
            name, typ = tok, "token"
        out.append((name, typ))
    return out


def _discover_schema(data_path: str, dataset_name: str) -> dict:
    """Inspect atomic files and propose USER_ID/ITEM_ID/LABEL + load_col."""
    base = Path(data_path) / dataset_name
    files = {
        "inter": _read_atomic_header(base / f"{dataset_name}.inter"),
        "user":  _read_atomic_header(base / f"{dataset_name}.user"),
        "item":  _read_atomic_header(base / f"{dataset_name}.item"),
    }

    if files["inter"] is None:
        raise FileNotFoundError(
            f"Could not find {base / (dataset_name + '.inter')}. "
            "Make sure RecBole's auto-download finished, or place the "
            "atomic files there manually."
        )

    inter_cols = files["inter"]
    inter_names = [n for n, _ in inter_cols]
    inter_types = dict(inter_cols)
    print(f"  detected {dataset_name}.inter columns: {inter_cols}")

    # Heuristics:
    # - USER_ID_FIELD: prefer literal 'user_id' -> first :token field
    # - ITEM_ID_FIELD: prefer literal 'item_id' -> second :token field
    # - LABEL_FIELD: prefer 'label'/'rating'/'income' -> only :float column
    token_cols = [n for n, t in inter_cols if t == "token"]
    float_cols = [n for n, t in inter_cols if t == "float"]

    user_field = (
        "user_id" if "user_id" in inter_names else
        (token_cols[0] if token_cols else None)
    )
    item_field = (
        "item_id" if "item_id" in inter_names else
        (token_cols[1] if len(token_cols) > 1 else None)
    )

    label_field = next(
        (n for n in ("label", "rating", "income") if n in inter_names),
        float_cols[0] if len(float_cols) == 1 else None,
    )

    load_col = {"inter": inter_names}
    if files["user"] is not None:
        load_col["user"] = [n for n, _ in files["user"]]
        print(f"  detected {dataset_name}.user  columns: {files['user']}")
    if files["item"] is not None:
        load_col["item"] = [n for n, _ in files["item"]]
        print(f"  detected {dataset_name}.item  columns: {files['item']}")

    print(f"  -> USER_ID_FIELD = {user_field!r}")
    print(f"  -> ITEM_ID_FIELD = {item_field!r}")
    print(f"  -> LABEL_FIELD   = {label_field!r}")

    return {
        "user_field":  user_field,
        "item_field":  item_field,
        "label_field": label_field,
        "load_col":    load_col,
        "inter_types": inter_types,
    }


def build_recbole_dataset(data_path: str, dataset_name: str):
    """Configure RecBole and return the loaded Dataset object."""
    # Adult-style datasets ship without user_id/item_id columns. If that's
    # what we're loading, synthesize them on a derived dataset and use it.
    effective_name = _ensure_user_item_columns(data_path, dataset_name)
    if effective_name != dataset_name:
        print(f"  [aug] using derived dataset '{effective_name}' "
              f"(original '{dataset_name}' kept untouched)")
    dataset_name = effective_name

    schema = _discover_schema(data_path, dataset_name)

    # Adult-style datasets (no real user/item) need synthetic ids. If we
    # couldn't find at least USER_ID_FIELD, surface a clear error.
    if schema["user_field"] is None or schema["item_field"] is None:
        raise RuntimeError(
            "Could not infer USER_ID_FIELD / ITEM_ID_FIELD from the atomic "
            "files. Edit build_recbole_dataset() and set them explicitly. "
            "Detected columns: " + str(schema["load_col"].get("inter"))
        )

    config_dict = {
        # environment
        "data_path": data_path,
        "seed": 2024,
        "reproducibility": True,
        # task — FM is just a vehicle to make RecBole load the data
        "model": "FM",
        "dataset": dataset_name,
        # atomic-file schema (auto-detected)
        "USER_ID_FIELD": schema["user_field"],
        "ITEM_ID_FIELD": schema["item_field"],
        "LABEL_FIELD":   schema["label_field"],
        "load_col":      schema["load_col"],
        # don't filter anything out for visualization
        "user_inter_num_interval": "[0,inf)",
        "item_inter_num_interval": "[0,inf)",
    }
    if schema["label_field"] is not None:
        config_dict["threshold"] = {schema["label_field"]: 0.5}

    config = Config(model="FM", dataset=dataset_name, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    print(dataset)  # RecBole prints a nice summary
    return config, dataset


def _feat_to_df(feat) -> pd.DataFrame | None:
    """Normalize a RecBole feat (DataFrame *or* Interaction) into a DataFrame."""
    if feat is None:
        return None
    # Newer RecBole versions: feat is already a pandas DataFrame.
    if isinstance(feat, pd.DataFrame):
        return feat.copy()
    # Older versions: feat is an Interaction with a `.interaction` dict.
    inner = getattr(feat, "interaction", None)
    if inner is not None:
        return pd.DataFrame({k: np.asarray(v) for k, v in inner.items()})
    # Last resort: try to dict-iterate it directly.
    try:
        return pd.DataFrame({k: np.asarray(v) for k, v in dict(feat).items()})
    except Exception as e:
        raise TypeError(f"Don't know how to convert feat of type "
                        f"{type(feat).__name__} to a DataFrame: {e}")


def _invert_token_remap(dataset, df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Replace RecBole's remapped integer ids with the original string tokens.
    RecBole rewrites every `:token` column to dense integers during loading
    (so `work_class` becomes [1,2,1,4,...] instead of ["Private","Self-emp",...]).
    Without inverting that, every categorical plot would be mislabelled.
    """
    if df is None:
        return None
    field2id_token = getattr(dataset, "field2id_token", None)
    if not field2id_token:
        return df
    df = df.copy()
    for field in df.columns:
        if field not in field2id_token:
            continue
        tokens = list(field2id_token[field])  # tokens[i] -> original string
        try:
            ints = df[field].astype(int).to_numpy()
            df[field] = [tokens[i] if 0 <= i < len(tokens) else f"<unk:{i}>"
                         for i in ints]
        except Exception as e:
            print(f"  ! could not invert remap for field '{field}': {e}")
    # Cosmetic: undo the whitespace->underscore swap we did to native_country
    # so display labels read naturally ("United States", not "United_States").
    if "native_country" in df.columns:
        df["native_country"] = (df["native_country"].astype(str)
                                                    .str.replace("_", " "))
    return df


def dataset_to_frames(dataset):
    """Pull RecBole's internal feats back into pandas DataFrames."""
    inter_df = _feat_to_df(getattr(dataset, "inter_feat", None))
    user_df  = _feat_to_df(getattr(dataset, "user_feat", None))
    item_df  = _feat_to_df(getattr(dataset, "item_feat", None))
    # Replace remapped integer ids with their original string tokens so all
    # downstream plots show readable category labels.
    inter_df = _invert_token_remap(dataset, inter_df)
    user_df  = _invert_token_remap(dataset, user_df)
    item_df  = _invert_token_remap(dataset, item_df)
    return inter_df, user_df, item_df


# --------------------------------------------------------------------------- #
# 2.  Visualization helpers
# --------------------------------------------------------------------------- #
def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {path}")


def plot_basic_stats(dataset, inter_df, out_dir: Path):
    """(a) Basic statistics — summary table figure + sparsity bar."""
    n_users  = dataset.user_num
    n_items  = dataset.item_num
    n_inter  = len(inter_df)
    density  = n_inter / max(n_users * n_items, 1)
    sparsity = 1.0 - density

    summary = pd.DataFrame(
        {
            "metric": ["#User", "#Item", "#Interaction",
                       "Density", "Sparsity"],
            "value":  [n_users, n_items, n_inter,
                       f"{density:.6f}", f"{sparsity:.6f}"],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 2 + 0.35 * len(summary)))
    ax.axis("off")
    tbl = ax.table(cellText=summary.values,
                   colLabels=summary.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)
    ax.set_title("Adult dataset — basic statistics", pad=12, fontsize=13)
    _save(fig, out_dir, "01_basic_stats.png")


def plot_interaction_distributions(inter_df, out_dir: Path):
    """(b) Long-tail distributions of interactions per user / per item."""
    if "user_id" not in inter_df.columns or "item_id" not in inter_df.columns:
        print("  !! inter_df missing user_id/item_id — skipping long-tail plots")
        return

    per_user = inter_df.groupby("user_id").size().values
    per_item = inter_df.groupby("item_id").size().values

    # Detect Adult-style synthetic ids — every user has 1 interaction and/or
    # there's only one distinct item. The plots would be degenerate.
    if per_user.max() <= 1 and len(per_item) <= 1:
        print("  !! distributions look degenerate "
              f"(max per-user={per_user.max()}, n_items={len(per_item)}) — "
              "skipping (this is expected for Adult-style CTR datasets).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, data, label in zip(axes, [per_user, per_item],
                               ["per user", "per item"]):
        sorted_counts = np.sort(data)[::-1]
        ax.loglog(np.arange(1, len(sorted_counts) + 1), sorted_counts,
                  marker=".", linestyle="none", alpha=0.6)
        ax.set_xlabel(f"Rank ({label})")
        ax.set_ylabel("Interaction count")
        ax.set_title(f"Interaction long tail — {label}")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Adult dataset — interaction distributions", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "02_interaction_distributions.png")


def plot_label_distribution(inter_df, out_dir: Path):
    """(c) Distribution of the binary label (income >= 50k)."""
    if "label" not in inter_df.columns:
        print("  !! no `label` column in inter_df — skipping label plot")
        return

    counts = inter_df["label"].value_counts().sort_index()
    pct    = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=["#4C78A8", "#F58518"])
    ax.set_xlabel("Label  (0 = income <50k, 1 = income >=50k)")
    ax.set_ylabel("Count")
    ax.set_title("Adult dataset — label distribution")
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{p:.1f}%", ha="center", va="bottom", fontsize=10)
    _save(fig, out_dir, "03_label_distribution.png")


def plot_feature_analysis(user_df, item_df, inter_df, out_dir: Path):
    """(d) Side-feature exploration — categorical bars + numeric histograms."""
    if user_df is None and item_df is None:
        print("  !! no side features available — skipping feature analysis")
        return

    # Join everything onto the interaction frame so we can colour by label.
    df = inter_df.copy()
    if user_df is not None:
        df = df.merge(user_df, on="user_id", how="left")
    if item_df is not None:
        df = df.merge(item_df, on="item_id", how="left")

    categorical = [c for c in ["workclass", "education", "marital_status",
                               "occupation", "relationship", "race", "sex",
                               "native_country"] if c in df.columns]
    numeric     = [c for c in ["age", "capital_gain", "capital_loss",
                               "hours_per_week"] if c in df.columns]

    # ---- categorical: top categories, stacked by label if available ---- #
    for col in categorical:
        top = df[col].value_counts().head(12).index
        sub = df[df[col].isin(top)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if "label" in sub.columns:
            ct = pd.crosstab(sub[col], sub["label"]).loc[top]
            ct.plot(kind="bar", stacked=True, ax=ax,
                    color=["#4C78A8", "#F58518"])
            ax.legend(title="label", labels=["<50k", ">=50k"])
        else:
            sub[col].value_counts().loc[top].plot(kind="bar", ax=ax)
        ax.set_title(f"Adult dataset — {col} (top categories)")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        _save(fig, out_dir, f"04_feature_cat_{col}.png")

    # ---- numeric: histogram split by label ---- #
    for col in numeric:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if "label" in df.columns:
            for lbl, grp in df.groupby("label"):
                ax.hist(grp[col].dropna(), bins=40, alpha=0.55,
                        label=f"label={int(lbl)}")
            ax.legend()
        else:
            ax.hist(df[col].dropna(), bins=40)
        ax.set_title(f"Adult dataset — distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        _save(fig, out_dir, f"04_feature_num_{col}.png")

    # ---- correlation heatmap on numeric columns ---- #
    if numeric:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(df[numeric].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        ax.set_title("Adult dataset — numeric feature correlations")
        _save(fig, out_dir, "04_feature_correlations.png")


# --------------------------------------------------------------------------- #
# Extended statistics — summary CSVs + richer plots
# --------------------------------------------------------------------------- #
DROPPED_FOR_FEATURE_ANALYSIS = {"user_id", "item_id"}


def _classify_columns(inter_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols), excluding ids."""
    numeric_cols, categorical_cols = [], []
    for c in inter_df.columns:
        if c in DROPPED_FOR_FEATURE_ANALYSIS:
            continue
        if pd.api.types.is_numeric_dtype(inter_df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def write_summary_tables(inter_df: pd.DataFrame, out_dir: Path):
    """Emit summary_numeric.csv and summary_categorical.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols, cat_cols = _classify_columns(inter_df)

    if numeric_cols:
        num = inter_df[numeric_cols].describe(
            percentiles=[.05, .25, .5, .75, .95]).T
        num["missing"] = inter_df[numeric_cols].isna().sum()
        num.to_csv(out_dir / "summary_numeric.csv")
        print(f"  -> wrote {out_dir / 'summary_numeric.csv'}")

    if cat_cols:
        rows = []
        for c in cat_cols:
            s  = inter_df[c].astype(str)
            vc = s.value_counts()
            rows.append({
                "column":      c,
                "n_unique":    int(s.nunique()),
                "top":         vc.index[0] if len(vc) else None,
                "top_freq":    int(vc.iloc[0]) if len(vc) else 0,
                "top_pct":     round(float(vc.iloc[0] / len(s) * 100), 2)
                                if len(vc) else 0.0,
                "missing_?":   int((s == "?").sum()),
                "missing_NaN": int(s.isna().sum()),
            })
        pd.DataFrame(rows).to_csv(out_dir / "summary_categorical.csv",
                                  index=False)
        print(f"  -> wrote {out_dir / 'summary_categorical.csv'}")


def plot_missing_values(inter_df: pd.DataFrame, out_dir: Path):
    """Bar chart of missing rate per column ('?' is Adult's NA marker)."""
    rates = {}
    for c in inter_df.columns:
        if c in DROPPED_FOR_FEATURE_ANALYSIS:
            continue
        s = inter_df[c].astype(str)
        rate = ((s == "?").sum() + s.isna().sum()) / max(len(inter_df), 1)
        if rate > 0:
            rates[c] = rate * 100

    if not rates:
        print("  no '?' or NaN values found — skipping missing-values plot")
        return

    s = pd.Series(rates).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(s))))
    ax.barh(s.index, s.values, color="#E45756")
    ax.set_xlabel("Missing rate (%)")
    ax.set_title("Adult — missing values per column ('?' or NaN)")
    for i, v in enumerate(s.values):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=9)
    _save(fig, out_dir, "05_missing_values.png")


def plot_income_rate_by_group(inter_df: pd.DataFrame, out_dir: Path):
    """For each interesting categorical, plot P(label=1 | group) vs baseline."""
    if "label" not in inter_df.columns:
        return
    overall = inter_df["label"].mean() * 100
    candidates = ["sex", "race", "marital_status", "work_class", "education",
                  "occupation", "relationship", "native_country"]
    cols = [c for c in candidates if c in inter_df.columns]

    for col in cols:
        df = inter_df[[col, "label"]].copy()
        df[col] = df[col].astype(str)
        # Drop the '?' missing marker from groupings
        df = df[df[col] != "?"]
        if col == "native_country":
            top = df[col].value_counts().head(15).index
            df = df[df[col].isin(top)]

        rates = df.groupby(col)["label"].agg(["mean", "count"])
        rates = rates[rates["count"] >= 20].sort_values("mean")
        if not len(rates):
            continue

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rates))))
        colors = ["#4C78A8" if v * 100 < overall else "#F58518"
                  for v in rates["mean"]]
        ax.barh(rates.index, rates["mean"] * 100, color=colors)
        ax.axvline(overall, ls="--", color="grey", linewidth=1.2,
                   label=f"Overall: {overall:.1f}%")
        ax.set_xlabel("P(income ≥ $50K | group)  (%)")
        ax.set_title(f"Adult — income rate by {col}")
        for i, (m, c) in enumerate(zip(rates["mean"], rates["count"])):
            ax.text(m * 100 + 0.4, i, f"{m*100:.1f}%  (n={c:,})",
                    va="center", fontsize=8)
        ax.legend(loc="lower right")
        _save(fig, out_dir, f"06_income_rate_by_{col}.png")


def plot_age_buckets(inter_df: pd.DataFrame, out_dir: Path):
    """Count + income rate by age decade (twin-axis chart)."""
    if "age" not in inter_df.columns or "label" not in inter_df.columns:
        return
    df = inter_df[["age", "label"]].copy()
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 20, 30, 40, 50, 60, 70, 100],
        labels=["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        right=False,
    )
    rates = df.groupby("age_bucket", observed=True)["label"].agg(
        ["mean", "count"])

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(rates.index.astype(str), rates["count"],
            color="#B0C4DE", alpha=0.8)
    ax1.set_ylabel("Count", color="#4C78A8")
    ax1.set_xlabel("Age bucket")
    ax2 = ax1.twinx()
    ax2.plot(rates.index.astype(str), rates["mean"] * 100,
             color="#F58518", marker="o", linewidth=2)
    ax2.set_ylabel("P(income ≥ $50K) (%)", color="#F58518")
    ax2.set_ylim(0, max(rates["mean"]) * 100 * 1.25 + 1)
    ax1.set_title("Adult — count and income rate by age bucket")
    fig.tight_layout()
    _save(fig, out_dir, "07_age_buckets.png")


def plot_hours_buckets(inter_df: pd.DataFrame, out_dir: Path):
    """Count + income rate by hours_per_week bucket."""
    if "hours_per_week" not in inter_df.columns or "label" not in inter_df.columns:
        return
    df = inter_df[["hours_per_week", "label"]].copy()
    df["bucket"] = pd.cut(
        df["hours_per_week"],
        bins=[0, 19, 29, 39, 40, 49, 59, 99],
        labels=["1-19", "20-29", "30-39", "40", "41-49", "50-59", "60+"],
    )
    rates = df.groupby("bucket", observed=True)["label"].agg(["mean", "count"])

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(rates.index.astype(str), rates["count"],
            color="#B0C4DE", alpha=0.8)
    ax1.set_ylabel("Count", color="#4C78A8")
    ax1.set_xlabel("Hours per week")
    ax2 = ax1.twinx()
    ax2.plot(rates.index.astype(str), rates["mean"] * 100,
             color="#F58518", marker="o", linewidth=2)
    ax2.set_ylabel("P(income ≥ $50K) (%)", color="#F58518")
    ax2.set_ylim(0, max(rates["mean"]) * 100 * 1.25 + 1)
    ax1.set_title("Adult — count and income rate by hours_per_week")
    fig.tight_layout()
    _save(fig, out_dir, "07_hours_buckets.png")


def plot_capital_specifics(inter_df: pd.DataFrame, out_dir: Path):
    """Capital gain/loss are heavily zero-inflated. Show that explicitly."""
    cg = inter_df["capital_gain"] if "capital_gain" in inter_df.columns else None
    cl = inter_df["capital_loss"] if "capital_loss" in inter_df.columns else None
    if cg is None and cl is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, s, name in zip(axes, [cg, cl], ["capital_gain", "capital_loss"]):
        if s is None:
            ax.set_visible(False)
            continue
        nz = s[s > 0]
        zero_pct = (s == 0).mean() * 100
        if len(nz):
            ax.hist(nz, bins=40, color="#4C78A8", alpha=0.85)
        ax.set_xlabel(f"{name} (nonzero values only)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\n{zero_pct:.1f}% are exactly 0  ·  "
                     f"n={len(nz):,} nonzero  ·  "
                     f"max={int(s.max()):,}")
    fig.suptitle("Adult — capital gain/loss zero-inflation structure",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "08_capital_gain_loss.png")


def plot_sex_race_heatmap(inter_df: pd.DataFrame, out_dir: Path):
    """Income rate heatmap on the sex × race grid (with cell counts)."""
    if not all(c in inter_df.columns for c in ("sex", "race", "label")):
        return
    df = inter_df[["sex", "race", "label"]].copy()
    df["sex"]  = df["sex"].astype(str)
    df["race"] = df["race"].astype(str)
    rate = df.groupby(["race", "sex"])["label"].mean().unstack() * 100
    cnt  = df.groupby(["race", "sex"])["label"].count().unstack()

    annot = rate.copy().astype(object)
    for r in rate.index:
        for s in rate.columns:
            r_val = rate.loc[r, s]
            n_val = cnt.loc[r, s] if s in cnt.columns and r in cnt.index else 0
            annot.loc[r, s] = (f"{r_val:.1f}%\nn={int(n_val):,}"
                               if pd.notna(r_val) else "—")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(rate, annot=annot.values, fmt="", cmap="RdYlGn_r",
                cbar_kws={"label": "P(income ≥ $50K) (%)"},
                ax=ax, vmin=0, vmax=max(60, float(np.nanmax(rate.values))))
    ax.set_title("Adult — income rate by sex × race")
    fig.tight_layout()
    _save(fig, out_dir, "09_sex_race_heatmap.png")


def plot_education_occupation_heatmap(inter_df: pd.DataFrame, out_dir: Path):
    """Income rate heatmap on education × (top) occupation."""
    if not all(c in inter_df.columns for c in ("education", "occupation", "label")):
        return
    df = inter_df[["education", "occupation", "label"]].copy()
    df["education"]  = df["education"].astype(str)
    df["occupation"] = df["occupation"].astype(str)
    df = df[df["occupation"] != "?"]

    top_occ = df["occupation"].value_counts().head(8).index
    df = df[df["occupation"].isin(top_occ)]

    rate = df.groupby(["education", "occupation"])["label"].mean().unstack() * 100
    # Order rows by mean income rate (low -> high)
    row_order = (df.groupby("education")["label"].mean()
                   .sort_values().index)
    rate = rate.reindex(row_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(rate, annot=True, fmt=".0f", cmap="RdYlGn_r",
                cbar_kws={"label": "P(income ≥ $50K) (%)"}, ax=ax,
                vmin=0, vmax=max(80, float(np.nanmax(rate.values))))
    ax.set_title("Adult — income rate by education × occupation (top 8 occ)")
    fig.tight_layout()
    _save(fig, out_dir, "10_education_occupation_heatmap.png")


# --------------------------------------------------------------------------- #
# 3.  Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=str(ROOT / "dataset"),
                        help="Folder containing <dataset>/<dataset>.* atomic files")
    parser.add_argument("--dataset", default="adult",
                        help="RecBole dataset name (folder + filename prefix)")
    parser.add_argument("--out", default=str(ROOT / "figures"),
                        help="Where to write the PNGs")
    args = parser.parse_args()

    # Resolve whatever was passed so relative args also stop depending on cwd.
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    out_dir = Path(args.out).expanduser().resolve()
    sns.set_theme(style="whitegrid")

    print(f"[1/4] Loading `{args.dataset}` from {args.data_path} via RecBole ...")
    config, dataset = build_recbole_dataset(args.data_path, args.dataset)

    print("[2/4] Materializing feats into pandas ...")
    inter_df, user_df, item_df = dataset_to_frames(dataset)
    print(f"      inter_df: {inter_df.shape}, "
          f"user_df: {None if user_df is None else user_df.shape}, "
          f"item_df: {None if item_df is None else item_df.shape}")

    print("[3/4] Plotting basic views ...")
    plot_basic_stats(dataset, inter_df, out_dir)
    plot_interaction_distributions(inter_df, out_dir)
    plot_label_distribution(inter_df, out_dir)
    plot_feature_analysis(user_df, item_df, inter_df, out_dir)

    print("[3.5/4] Extended statistics (summaries + group-conditional plots) ...")
    write_summary_tables(inter_df, out_dir)
    plot_missing_values(inter_df, out_dir)
    plot_income_rate_by_group(inter_df, out_dir)
    plot_age_buckets(inter_df, out_dir)
    plot_hours_buckets(inter_df, out_dir)
    plot_capital_specifics(inter_df, out_dir)
    plot_sex_race_heatmap(inter_df, out_dir)
    plot_education_occupation_heatmap(inter_df, out_dir)

    print(f"[4/4] Done. Figures + CSVs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()