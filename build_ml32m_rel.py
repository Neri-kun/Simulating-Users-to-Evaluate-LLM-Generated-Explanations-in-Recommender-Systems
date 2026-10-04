# build_ml32m_rel.py
import numpy as np, pandas as pd

D = "dataset/ml-32m"
kg  = pd.read_csv(f"{D}/ml-32m.kg",  sep="\t")
ent = pd.read_csv(f"{D}/ml-32m.ent", sep="\t", nrows=1)

rel_col = next(c for c in kg.columns  if c.startswith("relation_id"))
emb_col = next(c for c in ent.columns if c.startswith("ent_emb"))

relations = kg[rel_col].astype(str).unique().tolist()
dim = len(str(ent[emb_col].iloc[0]).split(" "))
print(f"{len(relations)} relations, dim={dim}: {relations}")

rng = np.random.default_rng(42)
with open(f"{D}/ml-32m.rel", "w") as f:
    f.write("rel_id:token\trel_emb:float_seq\n")
    for r in relations:
        vec = rng.normal(0, 0.1, size=dim)
        f.write(r + "\t" + " ".join(f"{x:.6f}" for x in vec) + "\n")
print("wrote ml-32m.rel")