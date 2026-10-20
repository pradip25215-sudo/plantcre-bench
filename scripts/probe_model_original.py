"""Frozen embeddings + ridge on the ORIGINAL published splits, for any embedding tag.
Same recipe as probe_promoters_original.py / probe_original_all.py, so numbers are comparable.

usage: python probe_model_original.py <tag>      e.g. agront_1b
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "plantcre"
P = ROOT / "data/processed"
TAG = sys.argv[1]
E = ROOT / "embeddings" / TAG
ALPHAS = np.logspace(-2, 5, 15)
layers = json.load(open(E / "meta.json"))["layers_saved"]

TASKS = [
    ("promoters", [("leaf", "split_leaf", "strength_leaf"), ("proto", "split_proto", "strength_proto")]),
    ("terminators", [(a, "split", f"strength_{a}") for a in ["tobacco", "maize"]]),
    ("enhancers", [(a, "split", f"strength_{a}") for a in ["light", "dark", "warm", "cold", "maize"]]),
]


def fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    p = m.predict(sc.transform(Xte))
    return pearsonr(p, yte)[0] ** 2, spearmanr(p, yte)[0]


rows = []
for name, tasks in TASKS:
    df = pd.read_parquet(P / f"{name}.parquet")
    ids = pd.read_parquet(E / f"{name}.ids.parquet")["id"].values
    assert (ids == df["id"].values).all(), f"{name}: embedding rows don't line up!"
    X_all = np.load(E / f"{name}.npy", mmap_mode="r")
    print(f"\n######## {TAG} / {name}", flush=True)
    for li, L in enumerate(layers):
        X = np.asarray(X_all[:, li, :], dtype=np.float32)  # load each layer once, reuse for all assays
        for a, scol, ycol in tasks:
            y = df[ycol].to_numpy(dtype=float)
            sp = df[scol].to_numpy()
            tr = (sp == "train") & np.isfinite(y)
            te = (sp == "test") & np.isfinite(y)
            r2, rho = fit_eval(X[tr], y[tr], X[te], y[te])
            rows.append(dict(model=TAG, dataset=name, assay=a, layer=L, r2=r2, spearman=rho))
            print(f"layer {L:>2} | {a:7s} R2={r2:.3f} rho={rho:.3f}", flush=True)
        del X

out = pd.DataFrame(rows)
out.to_csv(ROOT / f"results/probe_{TAG}_original_split.tsv", sep="\t", index=False)
best = out.loc[out.groupby(["dataset", "assay"])["r2"].idxmax()]
print("\n######## best layer per task")
print(best[["dataset", "assay", "layer", "r2", "spearman"]].round(3).to_string(index=False))
print(f"\nsaved results/probe_{TAG}_original_split.tsv")
