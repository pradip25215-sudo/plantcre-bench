"""Experiment 1: how much of the published performance comes from homology leakage?

Feature sets (all frozen + ridge): 6-mer counts, NT-v2-100M, AgroNT-1B.
Splits:
  original     the published split; test R2 also broken down by max homology score to train
  random       5 random 80/10/10 splits (same sizes as the homology ones) = leaky reference
  homology_t40 5 homology-aware splits, no pair with score >= 40 crosses train/test
  homology_t60 same at threshold 60 (where our BLAST recall is exactly 1.0)
Embedding layer is chosen ONCE per model x assay on the val part of random split seed 1
(train subsampled to 50k for speed), then kept fixed everywhere. Test data never picks anything.
Enhancers: random splits are made on base sequences so fwd/rev copies stay together.

usage: python exp1_homology.py <dataset>
"""
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "plantcre"
P, S, EMB = ROOT / "data/processed", ROOT / "data/splits", ROOT / "embeddings"
ALPHAS = np.logspace(-2, 5, 15)
SEEDS = [1, 2, 3, 4, 5]
THRS = [40, 60]
MODELS = ["6mer", "ntv2_100m", "agront_1b"]
BIN_NAMES = ["no hit", "<34 (chance)", "34-40", "40-60", "60-100", ">=100"]

DATASET = sys.argv[1]
ASSAYS = {
    "promoters": {"leaf": ("strength_leaf", "split_leaf"), "proto": ("strength_proto", "split_proto")},
    "terminators": {a: (f"strength_{a}", "split") for a in ["tobacco", "maize"]},
    "enhancers": {a: (f"strength_{a}", "split") for a in ["light", "dark", "warm", "cold", "maize"]},
}[DATASET]
UNIT = "base_id" if DATASET == "enhancers" else "id"
SPLITTAG = os.environ.get("SPLITTAG", "")


def kmer_counts(seqs, k=6):
    idx = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
    M = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for r, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            M[r, idx[s[j:j + k]]] += 1
    return np.log1p(M)


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def metrics(y, p):
    return pearsonr(p, y)[0] ** 2, spearmanr(p, y)[0]


df = pd.read_parquet(P / f"{DATASET}.parquet")
units = df[UNIT].to_numpy()

# ---- split labels, one array per (kind, seed), aligned with df rows
splits = {}
uniq = pd.unique(units)
n = len(uniq)
for s in SEEDS:
    perm = np.random.default_rng(1000 + s).permutation(n)
    lab = np.empty(n, dtype=object)
    lab[perm[:int(0.8 * n)]] = "train"
    lab[perm[int(0.8 * n):int(0.9 * n)]] = "val"
    lab[perm[int(0.9 * n):]] = "test"
    splits[("random", s)] = pd.Series(units).map(dict(zip(uniq, lab))).to_numpy()
for t in THRS:
    for s in SEEDS:
        sp = pd.read_parquet(S / f"{DATASET}{SPLITTAG}_t{t}_s{s}.parquet")[["id", "split"]]
        splits[(f"homology_t{t}", s)] = df[["id"]].merge(sp, on="id", how="left")["split"].to_numpy()

# ---- homology bins of the original test sets
bins = {}
for ycol, scol in ASSAYS.values():
    if scol not in bins:
        b = pd.read_parquet(S / f"{DATASET}_original_{scol}_homology_bins.parquet")
        bins[scol] = df[[UNIT]].merge(b[[UNIT, "bin"]], on=UNIT, how="left")["bin"].to_numpy()


def emb(model):
    ids = pd.read_parquet(EMB / model / f"{DATASET}.ids.parquet")["id"].values
    assert (ids == df["id"].values).all(), f"{model}: rows don't line up"
    return np.load(EMB / model / f"{DATASET}.npy", mmap_mode="r"), json.load(open(EMB / model / "meta.json"))["layers_saved"]


def pick_layers(model):
    X_all, layers = emb(model)
    sp = splits[("random", 1)]
    best = {}
    for li, L in enumerate(layers):
        X = np.asarray(X_all[:, li, :], dtype=np.float32)
        for a, (ycol, _) in ASSAYS.items():
            y = df[ycol].to_numpy(float)
            tr = np.where((sp == "train") & np.isfinite(y))[0]
            va = np.where((sp == "val") & np.isfinite(y))[0]
            if len(tr) > 50000:
                tr = np.sort(np.random.default_rng(0).choice(tr, 50000, replace=False))
            r2, _ = metrics(y[va], fit_predict(X[tr], y[tr], X[va]))
            if a not in best or r2 > best[a][2]:
                best[a] = (li, L, r2)
        del X
    return best


print(f"######## Experiment 1: {DATASET} | units={n:,} | rows={len(df):,}", flush=True)
print("building 6-mers...", flush=True)
K = kmer_counts(df["sequence"].tolist())

rows = []
for model in MODELS:
    if model == "6mer":
        choice = {a: (None, "-", np.nan) for a in ASSAYS}
    else:
        choice = pick_layers(model)
        print(f"\n[{model}] layer chosen on validation: " +
              ", ".join(f"{a}=L{c[1]} (val R2 {c[2]:.3f})" for a, c in choice.items()), flush=True)
    for a, (ycol, scol) in ASSAYS.items():
        li, L, _ = choice[a]
        X = K if model == "6mer" else np.asarray(emb(model)[0][:, li, :], dtype=np.float32)
        y = df[ycol].to_numpy(float)
        fin = np.isfinite(y)
        base = dict(dataset=DATASET, assay=a, model=model, layer=L)

        sp = df[scol].to_numpy()
        tr, te = (sp == "train") & fin, (sp == "test") & fin
        p = fit_predict(X[tr], y[tr], X[te])
        r2, rho = metrics(y[te], p)
        rows.append(dict(base, split="original", seed=0, bin="all", n_test=int(te.sum()), r2=r2, spearman=rho))
        bt = bins[scol][te]
        for bname in BIN_NAMES:
            mk = bt == bname
            if mk.sum() >= 100:
                r2b, rhob = metrics(y[te][mk], p[mk])
                rows.append(dict(base, split="original", seed=0, bin=bname, n_test=int(mk.sum()), r2=r2b, spearman=rhob))

        orig_r2 = r2
        res = {}
        for (kind, seed), sp in splits.items():
            tr, te = (sp == "train") & fin, (sp == "test") & fin
            r2, rho = metrics(y[te], fit_predict(X[tr], y[tr], X[te]))
            rows.append(dict(base, split=kind, seed=seed, bin="all", n_test=int(te.sum()), r2=r2, spearman=rho))
            res.setdefault(kind, []).append(r2)
        print(f"{model:10s} {a:7s} original={orig_r2:.3f} | " +
              " | ".join(f"{k}={np.mean(v):.3f}+-{np.std(v):.3f}" for k, v in res.items()), flush=True)
        if model != "6mer":
            del X
    pd.DataFrame(rows).to_csv(ROOT / f"results/exp1_{DATASET}{SPLITTAG}.tsv", sep="\t", index=False)

out = pd.DataFrame(rows)
out.to_csv(ROOT / f"results/exp1_{DATASET}{SPLITTAG}.tsv", sep="\t", index=False)
print("\n######## mean R2 (sd) over seeds, bin = all")
summ = out[out["bin"] == "all"].groupby(["assay", "model", "split"])["r2"].agg(["mean", "std"]).round(3)
print(summ.unstack("split").to_string())
print("\n######## original split, R2 by homology bin")
ob = out[(out["split"] == "original")].pivot_table(index=["assay", "model"], columns="bin", values="r2").round(3)
print(ob.reindex(columns=["all"] + BIN_NAMES).to_string())
print(f"\nsaved results/exp1_{DATASET}{SPLITTAG}.tsv\nEXP1 DONE {DATASET}", flush=True)
