# Simulating Users to Evaluate LLM-Generated Explanations in Recommender Systems

## Data

The datasets are **not included** in this repository. MovieLens may not be
redistributed without permission from GroupLens, and the IMDb files are licensed
for non-commercial use only. Download them from the original sources and rebuild
the files with the scripts below. `dataset/` is listed in `.gitignore`, so
nothing you generate there will be committed.

All commands are run from the repository root.

### Prepared copy (Git LFS)

The prepared `dataset/ml-32m/` folder is stored in this repository with
[Git LFS](https://git-lfs.com/), about 2.7 GB. Install Git LFS before cloning, and
the files download automatically:

```bash
git lfs install
git clone https://github.com/Neri-kun/Simulating-Users-to-Evaluate-LLM-Generated-Explanations-in-Recommender-Systems.git
```

If you cloned without Git LFS, the files will be small pointer files. Run
`git lfs pull` to download the data. To clone only the code, set
`GIT_LFS_SKIP_SMUDGE=1` before cloning.

With the prepared copy you can skip the steps below. The data stays under its original licenses: the
[MovieLens terms of use](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)
(non-commercial use, citation required; see [Citation](#citation)) and the
[IMDb non-commercial terms](https://developer.imdb.com/non-commercial-datasets/)
for the IMDb-derived knowledge-graph files. If you prefer, or the LFS
download is unavailable, rebuild the data from the original sources as described below.

### 1. Download MovieLens-32M

Download `ml-32m.zip` from <https://grouplens.org/datasets/movielens/32m/> and
extract it. Then copy the CSV files to these locations:

| File          | Copy to                                  |
|---------------|------------------------------------------|
| `ratings.csv` | `dataset/ml-32m/` and `dataset/ml32m_raw/` |
| `movies.csv`  | `dataset/ml-32m/` and `dataset/ml32m_raw/` |
| `links.csv`   | `dataset/ml-32m/`                        |

`build_ml32m_recbole.py` reads from `dataset/ml32m_raw/`. The other scripts read
from `dataset/ml-32m/`.

### 2. Build the RecBole files

```bash
python build_ml32m_recbole.py          # -> dataset/ml-32m/ml-32m.inter, ml-32m.item
python create_train_test_validate.py   # -> dataset/ml-32m/ml-32m.{train,valid,test}.inter
```

`create_train_test_validate.py` sorts each user's ratings by time and splits them
80/10/10 per user.

### 3. Build the knowledge graph (only for KG models)

These steps are needed only for the knowledge-graph models in
`run_recbole_ml32m.py`, such as GRU4RecKG. Run them in this order:

```bash
python build_ml32m_kg.py               # -> dataset/ml-32m/ml-32m.kg, ml-32m.link
python utils/train_kg_embeddings.py    # -> dataset/ml-32m/ml-32m.ent  (add --device cpu without a GPU)
python build_ml32m_rel.py              # -> dataset/ml-32m/ml-32m.rel  (needs ml-32m.ent)
```

`build_ml32m_kg.py` downloads `title.crew.tsv.gz` and `name.basics.tsv.gz`
(about 350 MB) from the [IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)
into the repository root, which is git-ignored. If you already have the files,
pass `--imdb-dir <folder> --skip-download`. IMDb refreshes these dumps daily,
so record the download date if you need exact reproducibility.

### 4. User demographics (only for the age analysis)

`utils/analyze_recsys_by_age.py` and `utils/visualize_users.py` read a
`users.csv` file from the repository root with these columns:

```
userId,gender,age,occupation,zip
```

MovieLens-32M has no demographic data, so this file is not produced by the steps
above. <!-- TODO: describe where users.csv comes from -->

### Expected layout

```
dataset/
├── ml32m_raw/
│   ├── ratings.csv
│   └── movies.csv
└── ml-32m/
    ├── ratings.csv
    ├── movies.csv
    ├── links.csv
    ├── ml-32m.inter
    ├── ml-32m.item
    ├── ml-32m.train.inter
    ├── ml-32m.valid.inter
    ├── ml-32m.test.inter
    ├── ml-32m.kg           # step 3
    ├── ml-32m.link         # step 3
    ├── ml-32m.ent          # step 3
    └── ml-32m.rel          # step 3
```

The RecBole configuration is in `config/ml-32m.yaml` (`data_path: dataset/`,
`dataset: ml-32m`).

### Citation

If you use MovieLens, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. *ACM Transactions on Interactive Intelligent Systems* 5, 4,
> Article 19. <https://doi.org/10.1145/2827872>
