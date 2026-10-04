"""
Train TransE entity embeddings on the ML-32M KG and export ml-32m.ent for GRU4RecKG.

Pipeline:
  1. Load ml-32m.kg (triples), ml-32m.link (item_id -> entity_id), ml-32m.item.
  2. Train TransE with margin-ranking loss + uniform negative head/tail corruption.
  3. For every item_id, look up its linked entity's vector (zeros if unlinked)
     and write ml-32m.ent with columns ent_id:token, ent_emb:float_seq.

Why this shape: GRU4RecKG calls
    dataset.get_preload_weight("ent_id")[: self.n_items]
to seed an nn.Embedding(n_items, embedding_size). RecBole aligns preload rows
through the *same* token dictionary as item_id, so the .ent file must be keyed
by item_id (not by KG entity name). Embedding dim must match the model's
embedding_size (32 in config/ml-32m.yaml).

Usage:
  python train_kg_embeddings.py
  python train_kg_embeddings.py --embedding-dim 64 --epochs 30 --device cpu
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def load_kg(kg_path: str) -> pd.DataFrame:
    df = pd.read_csv(kg_path, sep="\t")
    df.columns = ["head", "relation", "tail"]  # drop :token suffixes
    return df


def load_link(link_path: str) -> dict:
    df = pd.read_csv(link_path, sep="\t", dtype=str)
    df.columns = ["item_id", "entity_id"]
    return dict(zip(df["item_id"], df["entity_id"]))


def load_item_ids(item_path: str) -> pd.Series:
    df = pd.read_csv(item_path, sep="\t", dtype=str, usecols=["item_id:token"])
    df.columns = ["item_id"]
    return df["item_id"]


def train_transe(
    kg: pd.DataFrame,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    margin: float,
    lr: float,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    """Train TransE on the KG. Returns (entity_weight [n_ent, dim], entity->idx)."""
    entities = pd.unique(pd.concat([kg["head"], kg["tail"]], ignore_index=True))
    relations = pd.unique(kg["relation"])
    e2i = {e: i for i, e in enumerate(entities)}
    r2i = {r: i for i, r in enumerate(relations)}
    n_ent, n_rel = len(entities), len(relations)
    print(f"  entities={n_ent:,}  relations={n_rel}  triples={len(kg):,}")

    h = torch.tensor(kg["head"].map(e2i).values, dtype=torch.long, device=device)
    r = torch.tensor(kg["relation"].map(r2i).values, dtype=torch.long, device=device)
    t = torch.tensor(kg["tail"].map(e2i).values, dtype=torch.long, device=device)

    ent_emb = nn.Embedding(n_ent, embedding_dim).to(device)
    rel_emb = nn.Embedding(n_rel, embedding_dim).to(device)
    nn.init.xavier_uniform_(ent_emb.weight)
    nn.init.xavier_uniform_(rel_emb.weight)
    optimizer = torch.optim.Adam(
        list(ent_emb.parameters()) + list(rel_emb.parameters()), lr=lr
    )

    n_triples = len(kg)
    for epoch in range(1, epochs + 1):
        # TransE convention: L2-normalize entity embeddings before each pass.
        with torch.no_grad():
            ent_emb.weight.data = nn.functional.normalize(ent_emb.weight.data, p=2, dim=1)

        perm = torch.randperm(n_triples, device=device)
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for start in range(0, n_triples, batch_size):
            idx = perm[start : start + batch_size]
            h_b, r_b, t_b = h[idx], r[idx], t[idx]

            corrupt_head = torch.rand(len(idx), device=device) < 0.5
            h_neg = torch.randint(0, n_ent, (len(idx),), device=device)
            t_neg = torch.randint(0, n_ent, (len(idx),), device=device)
            h_n = torch.where(corrupt_head, h_neg, h_b)
            t_n = torch.where(corrupt_head, t_b, t_neg)

            pos = (ent_emb(h_b) + rel_emb(r_b) - ent_emb(t_b)).norm(p=1, dim=1)
            neg = (ent_emb(h_n) + rel_emb(r_b) - ent_emb(t_n)).norm(p=1, dim=1)
            loss = (margin + pos - neg).clamp(min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(
            f"  epoch {epoch:>3}/{epochs}  loss={total_loss / n_batches:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

    return ent_emb.weight.detach().cpu().numpy(), e2i


def export_ent_file(
    ent_weight: np.ndarray,
    e2i: dict,
    item_ids: pd.Series,
    item_to_entity: dict,
    embedding_dim: int,
    out_path: str,
) -> None:
    n_linked, n_unlinked = 0, 0
    rows_id = []
    rows_vec = []
    for iid in item_ids:
        ent_name = item_to_entity.get(iid)
        if ent_name is not None and ent_name in e2i:
            vec = ent_weight[e2i[ent_name]]
            n_linked += 1
        else:
            vec = np.zeros(embedding_dim, dtype=np.float32)
            n_unlinked += 1
        rows_id.append(iid)
        # space-separated to match RecBole's default seq_separator=" "
        rows_vec.append(" ".join(f"{v:.6f}" for v in vec))

    out = pd.DataFrame({"ent_id:token": rows_id, "ent_emb:float_seq": rows_vec})
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  wrote {out_path}: {len(out):,} rows  "
          f"({n_linked:,} item->entity hits, {n_unlinked:,} zero-init)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    #ap.add_argument("--dataset-dir", default="dataset/ml-32m")
    ap.add_argument("--embedding-dim", type=int, default=32,
                    help="Must match RecBole 'embedding_size' (32 in config/ml-32m.yaml).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu; falls back to cpu if cuda unavailable.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-dir", default=str(ROOT / "dataset" / "ml-32m"))
    args = ap.parse_args()

    args.dataset_dir = str(Path(args.dataset_dir).expanduser().resolve())

    kg_path = os.path.join(args.dataset_dir, "ml-32m.kg")
    link_path = os.path.join(args.dataset_dir, "ml-32m.link")
    item_path = os.path.join(args.dataset_dir, "ml-32m.item")
    ent_path = os.path.join(args.dataset_dir, "ml-32m.ent")

    for p in (kg_path, link_path, item_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Run build_ml32m_kg.py first.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) != args.device:
        print(f"(requested {args.device}, falling back to {device})")

    print("Loading KG inputs...")
    kg = load_kg(kg_path)
    item_to_entity = load_link(link_path)
    item_ids = load_item_ids(item_path)
    print(f"  link entries: {len(item_to_entity):,}")
    print(f"  item universe: {len(item_ids):,}")

    print(f"\nTraining TransE (dim={args.embedding_dim}, epochs={args.epochs}, "
          f"batch={args.batch_size}, margin={args.margin}, lr={args.lr})...")
    ent_weight, e2i = train_transe(
        kg, args.embedding_dim, args.epochs, args.batch_size,
        args.margin, args.lr, device,
    )

    print("\nExporting ml-32m.ent (indexed by item_id)...")
    export_ent_file(
        ent_weight, e2i, item_ids, item_to_entity, args.embedding_dim, ent_path,
    )


if __name__ == "__main__":
    main()
