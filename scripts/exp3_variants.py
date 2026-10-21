"""Experiment 3: can a model rank the effect of a single TFBS shuffle?

Ground truth: 741 WT / shuffled-TFBS pairs from the plantGREP validation library, measured in 5 conditions.
Every model is trained on enhancer strength using the FORCED homology split (t40, seed 1). In that split
each variant WT and all its homologs sit in test, so no model has seen the WT or anything close to it.
Predicted effect = f(mutant) - f(WT), compared with the measured delta.
Feature sets (frozen + ridge): 6-mer counts, NT-v2-100M layer 10, AgroNT-1B layer 12
(layers picked on validation in Experiment 1, nothing tuned here).
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
P, S, EMB = ROOT / "data/processed", ROOT / "data/splits", ROOT / "embeddings"
ALPHAS = np.logspace(-2, 5, 15)
COND = ["light", "dark", "warm", "cold", "maize"]
LAYER = {"ntv2_100m": 10, "agront_1b": 12}
STRONG = 0.5  # |measured delta| at or above this counts as a clear effect


def kmer_counts(seqs, k=6):
    idx = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
    M = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for r, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            M[r, idx[s[j:j + k]]] += 1
    return np.log1p(M)


enh = pd.read_parquet(P / "enhancers.parquet")
sp = enh[["id"]].merge(pd.read_parquet(S / "enhancers_t40_s1.parquet")[["id", "split"]],
                       on="id", how="left")["split"].to_numpy()
train = sp == "train"
v = pd.read_parquet(P / "variant_pairs.parquet").reset_index(drop=True)
wt_base = set(v["base_sequence"].str.replace(r"_(fwd|rev)$", "", regex=True))
leak = enh.loc[train, "base_id"].isin(wt_base).sum()
print(f"pairs={len(v)} | train rows={train.sum():,} | variant WT sequences found in TRAIN: {leak} (must be 0)", flush=True)
assert leak == 0


def variant_index(model):
    ids = pd.read_parquet(EMB / model / "variants.ids.parquet")["id"]
    pos = dict(zip(ids, range(len(ids))))
    w = np.array([pos[f"{b}|WT"] for b in v["base_sequence"]])
    m = np.array([pos[f"{b}|{t}|MUT"] for b, t in zip(v["base_sequence"], v["tf_family"])])
    return w, m


def features(model):
    if model == "6mer":
        return K_enh, kmer_counts(v["seq_wt"].tolist()), kmer_counts(v["seq_mut"].tolist())
    li = json.load(open(EMB / model / "meta.json"))["layers_saved"].index(LAYER[model])
    ids = pd.read_parquet(EMB / model / "enhancers.ids.parquet")["id"].values
    assert (ids == enh["id"].values).all()
    Xe = np.asarray(np.load(EMB / model / "enhancers.npy", mmap_mode="r")[:, li, :], dtype=np.float32)
    Xv = np.load(EMB / model / "variants.npy")
    w, m = variant_index(model)
    return Xe, Xv[w, li, :].astype(np.float32), Xv[m, li, :].astype(np.float32)


print("building 6-mers...", flush=True)
K_enh = kmer_counts(enh["sequence"].tolist())

rows, fam_rows = [], []
for model in ["6mer", "ntv2_100m", "agront_1b"]:
    Xe, Xw, Xm = features(model)
    for c in COND:
        y = enh[f"strength_{c}"].to_numpy(float)
        tr = train & np.isfinite(y)
        sc = StandardScaler().fit(Xe[tr])
        r = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xe[tr]), y[tr])
        pw, pm = r.predict(sc.transform(Xw)), r.predict(sc.transform(Xm))
        pdelta = pm - pw
        d = v[f"delta_{c}"].to_numpy(float)
        wt = v[f"strength_{c}_wt"].to_numpy(float)
        ok, okw = np.isfinite(d), np.isfinite(wt)
        strong = ok & (np.abs(d) >= STRONG)
        row = dict(model=model, condition=c, n_pairs=int(ok.sum()),
                   wt_r2=pearsonr(pw[okw], wt[okw])[0] ** 2, wt_spearman=spearmanr(pw[okw], wt[okw])[0],
                   delta_spearman=spearmanr(pdelta[ok], d[ok])[0], delta_pearson=pearsonr(pdelta[ok], d[ok])[0],
                   n_strong=int(strong.sum()),
                   sign_acc_strong=float((np.sign(pdelta[strong]) == np.sign(d[strong])).mean()),
                   pred_delta_sd=float(np.std(pdelta[ok])), measured_delta_sd=float(np.std(d[ok])))
        rows.append(row)
        print(f"{model:10s} {c:6s} | WT R2={row['wt_r2']:.3f} | delta rho={row['delta_spearman']:+.3f} "
              f"r={row['delta_pearson']:+.3f} | sign acc (|d|>={STRONG}, n={row['n_strong']}) = {row['sign_acc_strong']:.2f}",
              flush=True)
        for fam, g in v[ok].groupby("tf_family"):
            ix = g.index.to_numpy()
            if len(ix) >= 20:
                fam_rows.append(dict(model=model, condition=c, tf_family=fam, n=len(ix),
                                     measured_median=float(np.median(d[ix])), predicted_mean=float(np.mean(pdelta[ix])),
                                     delta_spearman=spearmanr(pdelta[ix], d[ix])[0]))
    if model != "6mer":
        del Xe

out = pd.DataFrame(rows)
fam = pd.DataFrame(fam_rows)
out.to_csv(ROOT / "results/exp3_variants.tsv", sep="\t", index=False)
fam.to_csv(ROOT / "results/exp3_variants_by_family.tsv", sep="\t", index=False)
print("\n######## delta Spearman (predicted vs measured effect)")
print(out.pivot(index="model", columns="condition", values="delta_spearman").round(3)[COND].to_string())
print("\n######## sign accuracy on clear effects")
print(out.pivot(index="model", columns="condition", values="sign_acc_strong").round(2)[COND].to_string())
print("\n######## WT strength R2 (held-out, sequence level)")
print(out.pivot(index="model", columns="condition", values="wt_r2").round(3)[COND].to_string())
print("\n######## per TF family, light: measured median vs predicted mean effect")
print(fam[fam["condition"] == "light"].pivot(index="tf_family", columns="model",
      values="predicted_mean").round(3).join(fam[(fam["condition"] == "light") & (fam["model"] == "6mer")]
      .set_index("tf_family")["measured_median"]).sort_values("measured_median").to_string())
print("\nEXP3 DONE", flush=True)
