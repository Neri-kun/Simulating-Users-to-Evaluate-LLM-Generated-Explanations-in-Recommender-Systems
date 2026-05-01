#!/usr/bin/env python3
"""
Visualize age and occupation distributions for the synthetic
MovieLens-32M users generated from MovieLens-1M demographics.

Reads users.csv from the same folder as this script and writes two PNG
charts (age_distribution.png, occupation_distribution.png) alongside it.

Requires: pandas, matplotlib.
    pip install pandas matplotlib
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ----
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "users.csv"
OUT_AGE = HERE / "age_distribution.png"
OUT_OCC = HERE / "occupation_distribution.png"
OUT_AGE_GENDER = HERE / "age_by_gender.png"
OUT_OCC_GENDER = HERE / "occupation_by_gender.png"
OUT_HEATMAP    = HERE / "age_by_occupation_heatmap.png"

GENDER_COLORS = {"F": "#d65db1", "M": "#2c8bc9"}

# Code -> label mappings (from MovieLens-1M README)
AGE_LABELS = {
    1:  "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}
AGE_ORDER = [1, 18, 25, 35, 45, 50, 56]  # chronological

OCC_LABELS = {
    0:  "other / not specified",
    1:  "academic/educator",
    2:  "artist",
    3:  "clerical/admin",
    4:  "college/grad student",
    5:  "customer service",
    6:  "doctor/health care",
    7:  "executive/managerial",
    8:  "farmer",
    9:  "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


# ---- Load ----
print(f"Reading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
n = len(df)
print(f"Loaded {n:,} users")


# ---- Helper: annotate each bar with count and % ----
def annotate_bars(ax, counts, total, horizontal=False):
    for i, v in enumerate(counts):
        pct = 100 * v / total
        label = f"{v:,}  ({pct:.1f}%)"
        if horizontal:
            ax.text(v, i, "  " + label, va="center", ha="left", fontsize=9)
        else:
            ax.text(i, v, label, ha="center", va="bottom", fontsize=9)


# ---- Age distribution ----
print("Plotting age distribution...")
age_counts = df["age"].value_counts().reindex(AGE_ORDER, fill_value=0)
age_labels = [AGE_LABELS[c] for c in AGE_ORDER]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(age_labels, age_counts.values, color="steelblue", edgecolor="black")
ax.set_title(f"Age distribution of MovieLens-32M synthetic users (n = {n:,})", fontsize=13)
ax.set_xlabel("Age group")
ax.set_ylabel("Number of users")
ax.set_ylim(0, age_counts.max() * 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.5)
annotate_bars(ax, age_counts.values, n, horizontal=False)
plt.tight_layout()
plt.savefig(OUT_AGE, dpi=150)
plt.close(fig)
print(f"  -> {OUT_AGE}")


# ---- Occupation distribution (sorted by count, descending) ----
print("Plotting occupation distribution...")
occ_counts = df["occupation"].value_counts().sort_values(ascending=True)  # ascending for horizontal bars (largest on top)
occ_names = [OCC_LABELS[c] for c in occ_counts.index]

fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh(occ_names, occ_counts.values, color="darkorange", edgecolor="black")
ax.set_title(f"Occupation distribution of MovieLens-32M synthetic users (n = {n:,})", fontsize=13)
ax.set_xlabel("Number of users")
ax.set_ylabel("Occupation")
ax.set_xlim(0, occ_counts.max() * 1.22)
ax.grid(axis="x", linestyle="--", alpha=0.5)
annotate_bars(ax, occ_counts.values, n, horizontal=True)
plt.tight_layout()
plt.savefig(OUT_OCC, dpi=150)
plt.close(fig)
print(f"  -> {OUT_OCC}")


# ---- Age x Gender (grouped bars) ----
print("Plotting age x gender...")
ag = (
    df.groupby(["age", "gender"]).size().unstack(fill_value=0)
      .reindex(index=AGE_ORDER, columns=["F", "M"], fill_value=0)
)
x = np.arange(len(AGE_ORDER))
width = 0.4

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - width/2, ag["F"].values, width, label="Female",
       color=GENDER_COLORS["F"], edgecolor="black")
ax.bar(x + width/2, ag["M"].values, width, label="Male",
       color=GENDER_COLORS["M"], edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels([AGE_LABELS[c] for c in AGE_ORDER])
ax.set_title(f"Age x Gender of MovieLens-32M synthetic users (n = {n:,})", fontsize=13)
ax.set_xlabel("Age group")
ax.set_ylabel("Number of users")
ax.set_ylim(0, ag.values.max() * 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(title="Gender")
# Annotate with counts
for i, age in enumerate(AGE_ORDER):
    for j, g in enumerate(["F", "M"]):
        v = ag.loc[age, g]
        ax.text(x[i] + (-width/2 if g == "F" else width/2), v,
                f"{v:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_AGE_GENDER, dpi=150)
plt.close(fig)
print(f"  -> {OUT_AGE_GENDER}")


# ---- Occupation x Gender (horizontal grouped bars) ----
print("Plotting occupation x gender...")
og = (
    df.groupby(["occupation", "gender"]).size().unstack(fill_value=0)
      .reindex(columns=["F", "M"], fill_value=0)
)
og["total"] = og["F"] + og["M"]
og = og.sort_values("total", ascending=True)
occ_codes = og.index.tolist()
occ_names = [OCC_LABELS[c] for c in occ_codes]

y = np.arange(len(occ_codes))
height = 0.4

fig, ax = plt.subplots(figsize=(11, 10))
ax.barh(y - height/2, og["F"].values, height, label="Female",
        color=GENDER_COLORS["F"], edgecolor="black")
ax.barh(y + height/2, og["M"].values, height, label="Male",
        color=GENDER_COLORS["M"], edgecolor="black")
ax.set_yticks(y)
ax.set_yticklabels(occ_names)
ax.set_title(f"Occupation x Gender of MovieLens-32M synthetic users (n = {n:,})", fontsize=13)
ax.set_xlabel("Number of users")
ax.set_ylabel("Occupation")
ax.set_xlim(0, max(og["F"].max(), og["M"].max()) * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.legend(title="Gender", loc="lower right")
# Annotate with counts
for i, code in enumerate(occ_codes):
    for j, g in enumerate(["F", "M"]):
        v = og.loc[code, g]
        ax.text(v, y[i] + (-height/2 if g == "F" else height/2),
                f"  {v:,}", va="center", ha="left", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_OCC_GENDER, dpi=150)
plt.close(fig)
print(f"  -> {OUT_OCC_GENDER}")


# ---- Age x Occupation heatmap (row-normalized: age profile per occupation) ----
print("Plotting age x occupation heatmap...")
# Rows: occupations (sorted by total count, largest at top).
# Columns: age groups in chronological order.
# Cell values: row-normalized percentages — what share of each occupation
# falls into each age group. Annotated with raw counts.
cross_counts = (
    df.groupby(["occupation", "age"]).size().unstack(fill_value=0)
      .reindex(columns=AGE_ORDER, fill_value=0)
)
occ_totals = cross_counts.sum(axis=1).sort_values(ascending=False)
cross_counts = cross_counts.reindex(occ_totals.index)
cross_pct = cross_counts.div(cross_counts.sum(axis=1), axis=0) * 100

row_labels = [OCC_LABELS[c] for c in cross_counts.index]
col_labels = [AGE_LABELS[c] for c in AGE_ORDER]

fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(cross_pct.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(np.arange(len(col_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels)
ax.set_title(
    f"Age profile per occupation (row-normalized %, raw counts shown, n = {n:,})",
    fontsize=12,
)
ax.set_xlabel("Age group")
ax.set_ylabel("Occupation (sorted by total)")

# Annotate each cell with count and row-% for quick reading.
for i in range(cross_counts.shape[0]):
    for j in range(cross_counts.shape[1]):
        count = cross_counts.iat[i, j]
        pct = cross_pct.iat[i, j]
        # Pick text color based on cell intensity for legibility.
        color = "white" if pct > 45 else "black"
        ax.text(j, i, f"{count:,}\n{pct:.1f}%", ha="center", va="center",
                color=color, fontsize=7)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("% of occupation (row-normalized)")
plt.tight_layout()
plt.savefig(OUT_HEATMAP, dpi=150)
plt.close(fig)
print(f"  -> {OUT_HEATMAP}")

print("Done.")
