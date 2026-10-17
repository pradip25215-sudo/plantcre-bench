"""First sanity result: promoter strength on the ORIGINAL Jores split.

Compares, on the same test set:
  - frozen NT-v2-100M embeddings (every saved layer) + ridge
  - 6-mer counts + ridge  (the composition baseline)
  - the original Jores CNN predictions (reference number from their repo)
This is the 'week-2 gate': if these look crazy, something upstream is broken.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "plantcre"
P = ROOT / "data/processed"
E = ROOT / "embeddings/ntv2_100m"
ALPHAS = np.logspace(-2, 5, 15)

prom = pd.read_parquet(P / "promoters.parquet")
X_all = np.load(E / "promoters.npy", mmap_mode="r")
ids = pd.read_parquet(E / "promoters.ids.parquet")["id"].values
assert (ids == prom["id"].values).all(), "embedding rows don't line up with the data!"
layers = json.load(open(E / "meta.json"))["layers_saved"]


def kmer_counts(seqs, k=6):
    idx = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
    M = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for r, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            M[r, idx[s[j:j + k]]] += 1
    return np.log1p(M)


def fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    p = m.predict(sc.transform(Xte))
    return pearsonr(p, yte)[0] ** 2, spearmanr(p, yte)[0], m.alpha_


print("building 6-mer features...", flush=True)
K = kmer_counts(prom["sequence"].tolist())

rows = []
for assay in ["leaf", "proto"]:
    sp, y = prom[f"split_{assay}"], prom[f"strength_{assay}"]
    tr = (sp == "train").to_numpy()
    te = (sp == "test").to_numpy()
    print(f"\n=== {assay}: train={tr.sum():,} test={te.sum():,} ===", flush=True)

    r2, rho, a = fit_eval(K[tr], y[tr], K[te], y[te])
    rows.append(dict(assay=assay, model="6-mer ridge", layer="-", r2=r2, spearman=rho, alpha=a))
    print(f"6-mer ridge        R2={r2:.3f}  rho={rho:.3f}", flush=True)

    for li, L in enumerate(layers):
        X = np.asarray(X_all[:, li, :], dtype=np.float32)
        r2, rho, a = fit_eval(X[tr], y[tr], X[te], y[te])
        rows.append(dict(assay=assay, model="NT-v2-100M ridge", layer=L, r2=r2, spearman=rho, alpha=a))
        print(f"NT layer {L:>2} ridge   R2={r2:.3f}  rho={rho:.3f}", flush=True)

    ref = ROOT / f"data/external/jores2021_promoters/CNN/CNN_test_{assay}_pred.tsv"
    pr = pd.read_csv(ref, sep="\t")
    print(f"Jores CNN pred file columns: {list(pr.columns)}", flush=True)
    num = pr.select_dtypes("number")
    if num.shape[1] >= 2:
        c = num.columns
        r = pearsonr(num[c[0]], num[c[1]])[0]
        print(f"Jores CNN ({c[0]} vs {c[1]})  R2={r * r:.3f}", flush=True)
        rows.append(dict(assay=assay, model=f"Jores CNN (published)", layer="-", r2=r * r,
                         spearman=spearmanr(num[c[0]], num[c[1]])[0], alpha=np.nan))

out = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
out.to_csv(ROOT / "results/probe_promoters_original_split.tsv", sep="\t", index=False)
print("\nsaved results/probe_promoters_original_split.tsv")
