"""Same sanity test as for promoters, now for terminators and enhancers (ORIGINAL splits).

For every assay/condition: 6-mer counts + ridge vs frozen NT-v2-100M (each saved layer) + ridge,
plus the published model's own test predictions as the reference number.
Ridge is trained on 'train' only, scored on 'test'.
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
EXT = ROOT / "data/external"
ALPHAS = np.logspace(-2, 5, 15)
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
    return pearsonr(p, yte)[0] ** 2, spearmanr(p, yte)[0]


def r2(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return pearsonr(a[ok], b[ok])[0] ** 2


rows = []
for name, assays in [("terminators", ["tobacco", "maize"]),
                     ("enhancers", ["light", "dark", "warm", "cold", "maize"])]:
    df = pd.read_parquet(P / f"{name}.parquet")
    ids = pd.read_parquet(E / f"{name}.ids.parquet")["id"].values
    assert (ids == df["id"].values).all(), f"{name}: embedding rows don't line up!"
    X_all = np.load(E / f"{name}.npy", mmap_mode="r")
    print(f"\n######## {name}: building 6-mers for {len(df):,} sequences...", flush=True)
    K = kmer_counts(df["sequence"].tolist())
    split = df["split"].to_numpy()

    for a in assays:
        y = df[f"strength_{a}"].to_numpy(dtype=float)
        tr = (split == "train") & np.isfinite(y)
        te = (split == "test") & np.isfinite(y)
        print(f"\n=== {name} / {a}: train={tr.sum():,} test={te.sum():,} ===", flush=True)
        s_r2, s_rho = fit_eval(K[tr], y[tr], K[te], y[te])
        rows.append(dict(dataset=name, assay=a, model="6-mer ridge", layer="-", r2=s_r2, spearman=s_rho))
        print(f"6-mer ridge       R2={s_r2:.3f} rho={s_rho:.3f}", flush=True)
        for li, L in enumerate(layers):
            X = np.asarray(X_all[:, li, :], dtype=np.float32)
            n_r2, n_rho = fit_eval(X[tr], y[tr], X[te], y[te])
            rows.append(dict(dataset=name, assay=a, model="NT-v2-100M ridge", layer=L, r2=n_r2, spearman=n_rho))
            print(f"NT layer {L:>2}       R2={n_r2:.3f} rho={n_rho:.3f}", flush=True)

# ---- published reference numbers ----
print("\n######## published models (their own test predictions)", flush=True)
pg = pd.read_csv(EXT / "plantgrep_enhancers/data/modelling_data/plantGREP_predictions.tsv", sep="\t")
for i, a in enumerate(["light", "dark", "warm", "cold", "maize"]):
    v = r2(pg[f"predictions_{i}"].to_numpy(float), pg[f"target_{i}"].to_numpy(float))
    rows.append(dict(dataset="enhancers", assay=a, model="plantGREP CNN (published)", layer="-", r2=v, spearman=np.nan))
    print(f"plantGREP CNN  {a:6s} R2={v:.3f}  (assuming output order light,dark,warm,cold,maize)", flush=True)
lm = pd.read_csv(EXT / "plantgrep_enhancers/data/modelling_data/lm_kmers_predictions.tsv.gz", sep="\t")
for a in ["light", "dark", "warm", "cold", "maize"]:
    v = r2(lm[f"prediction_{a}"].to_numpy(float), lm[f"enrichment_{a}"].to_numpy(float))
    rows.append(dict(dataset="enhancers", assay=a, model="6-mer linear (published)", layer="-", r2=v, spearman=np.nan))
    print(f"published 6-mer lm {a:6s} R2={v:.3f}", flush=True)
tp = pd.read_csv(EXT / "gorjifard2024_terminators/CNN/terminator_data_predictions.tsv", sep="\t")
print(f"terminator prediction file columns: {list(tp.columns)}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/probe_original_split_terminators_enhancers.tsv", sep="\t", index=False)
print("\nsaved results/probe_original_split_terminators_enhancers.tsv", flush=True)
