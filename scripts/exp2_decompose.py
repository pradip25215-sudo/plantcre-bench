"""Experiment 2: how much of what the gLM embeddings carry is just 6-mer composition?

For each model (best layer from Experiment 1) and dataset:
  1. regress the embedding E on standardized 6-mer counts K (multi-output ridge, alpha picked on a
     held-out 10% of train) -> E_hat = composition part, R = E - E_hat = what 6-mers can't explain
  2. ridge probes for strength on: K, E, E_hat, R, and K+R
Reported: share of embedding variance explained by 6-mers, and the R2 that R adds on top of K.
Original published splits. Enhancers: fwd copies only, train subsampled to 100k (memory).
The decomposition never sees strength values, so fitting it on train rows only is just to stay clean.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "plantcre"
P, EMB = ROOT / "data/processed", ROOT / "embeddings"
ALPHAS = np.logspace(-2, 5, 15)
MODELS = {"ntv2_100m": 10, "agront_1b": 12}
TASKS = {
    "promoters": {"leaf": ("strength_leaf", "split_leaf"), "proto": ("strength_proto", "split_proto")},
    "terminators": {a: (f"strength_{a}", "split") for a in ["tobacco", "maize"]},
    "enhancers": {a: (f"strength_{a}", "split") for a in ["light", "dark", "warm", "cold", "maize"]},
}


def kmer_counts(seqs, k=6):
    idx = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
    M = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for r, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            M[r, idx[s[j:j + k]]] += 1
    return np.log1p(M)


def probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return pearsonr(m.predict(sc.transform(Xte)), yte)[0] ** 2


rng = np.random.default_rng(0)
rows = []
for dataset, tasks in TASKS.items():
    df = pd.read_parquet(P / f"{dataset}.parquet")
    keep = np.ones(len(df), bool) if dataset != "enhancers" else (df["orientation"] == "fwd").to_numpy()
    rows_idx = np.where(keep)[0]
    sub = df.iloc[rows_idx].reset_index(drop=True)
    print(f"\n######## {dataset}: {len(sub):,} rows; building 6-mers...", flush=True)
    K = kmer_counts(sub["sequence"].tolist())

    first_scol = list(tasks.values())[0][1]
    tr0 = np.where(sub[first_scol].to_numpy() == "train")[0]
    if len(tr0) > 100_000:
        tr0 = np.sort(rng.choice(tr0, 100_000, replace=False))
    va0 = np.sort(rng.choice(tr0, len(tr0) // 10, replace=False))
    fit0 = np.setdiff1d(tr0, va0)

    kmu, ksd = K[tr0].mean(0), K[tr0].std(0) + 1e-6
    Ks = (K - kmu) / ksd

    for model, layer in MODELS.items():
        ids = pd.read_parquet(EMB / model / f"{dataset}.ids.parquet")["id"].values
        assert (ids == df["id"].values).all()
        li = json.load(open(EMB / model / "meta.json"))["layers_saved"].index(layer)
        E = np.asarray(np.load(EMB / model / f"{dataset}.npy", mmap_mode="r")[rows_idx, li, :], dtype=np.float32)
        emu, esd = E[tr0].mean(0), E[tr0].std(0) + 1e-6
        Es = (E - emu) / esd

        G = Ks[fit0].T.astype(np.float64) @ Ks[fit0].astype(np.float64)
        KE = Ks[fit0].T.astype(np.float64) @ Es[fit0].astype(np.float64)
        best = None
        for a in [1e1, 1e2, 1e3, 1e4, 1e5]:
            B = np.linalg.solve(G + a * np.eye(G.shape[0]), KE)
            pred = Ks[va0] @ B.astype(np.float32)
            ve = 1 - ((Es[va0] - pred) ** 2).sum() / ((Es[va0] - Es[va0].mean(0)) ** 2).sum()
            if best is None or ve > best[1]:
                best = (a, ve)
        G = Ks[tr0].T.astype(np.float64) @ Ks[tr0].astype(np.float64)
        KE = Ks[tr0].T.astype(np.float64) @ Es[tr0].astype(np.float64)
        B = np.linalg.solve(G + best[0] * np.eye(G.shape[0]), KE).astype(np.float32)
        E_hat = Ks @ B
        R = Es - E_hat
        print(f"[{model} L{layer}] 6-mers explain {100 * best[1]:.1f}% of embedding variance (held-out, alpha={best[0]:g})", flush=True)

        for a, (ycol, scol) in tasks.items():
            y = sub[ycol].to_numpy(float)
            sp = sub[scol].to_numpy()
            tr = np.where((sp == "train") & np.isfinite(y))[0]
            if len(tr) > 100_000:
                tr = np.sort(rng.choice(tr, 100_000, replace=False))
            te = np.where((sp == "test") & np.isfinite(y))[0]
            res = {}
            for name, X in [("K", K), ("E", E), ("E_hat", E_hat), ("R", R)]:
                res[name] = probe(X[tr], y[tr], X[te], y[te])
            KR = np.hstack([Ks, R])
            res["K+R"] = probe(KR[tr], y[tr], KR[te], y[te])
            del KR
            rows.append(dict(dataset=dataset, assay=a, model=model, layer=layer,
                             var_explained_by_kmers=best[1], **{f"r2_{k}": v for k, v in res.items()},
                             gain_R_over_K=res["K+R"] - res["K"]))
            print(f"   {a:7s} K={res['K']:.3f} E={res['E']:.3f} E_hat={res['E_hat']:.3f} R={res['R']:.3f} "
                  f"K+R={res['K+R']:.3f} | gain over 6-mers={res['K+R'] - res['K']:+.3f}", flush=True)
        del E, Es, E_hat, R

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/exp2_decompose.tsv", sep="\t", index=False)
print("\n######## gain of the non-composition part over 6-mers (R2 of K+R minus R2 of K)")
print(out.pivot_table(index=["dataset", "assay"], columns="model", values="gain_R_over_K").round(3).to_string())
print("\nEXP2 DONE", flush=True)
