"""Noise ceiling for Experiment 3, from the authors' replicate-level data (eVal_reps.rds).

For each condition, delta = shuffle - WT is computed separately in every experiment (= replicate).
Replicate-vs-replicate correlation of delta tells how well the assay agrees with itself.
Spearman-Brown turns the mean pairwise r into the reliability of the k-replicate average,
and sqrt(reliability) is the highest correlation any model can reach against that average.
Done at barcode cutoffs 1 / 5 / 10 (min barcodes for both WT and shuffle in every replicate used).
Same thing for WT strength (sequence-level ceiling, as R2).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from scipy.stats import pearsonr, spearmanr

ROOT = Path.home() / "plantcre"
G = ROOT / "data/external/plantgrep_enhancers/data/RData"
COND = ["light", "dark", "warm", "cold", "maize"]

reps = pyreadr.read_r(str(G / "eVal_reps.rds"))[None]
for c in ["experiment", "condition", "id"]:
    reps[c] = reps[c].astype(str)
reps["condition"] = reps["condition"].replace({"35C": "warm", "10C": "cold"})
print("experiments per condition:",
      reps.groupby("condition")["experiment"].unique().apply(sorted).to_dict(), flush=True)

parts = reps["id"].str.extract(r"^(?P<base>.+)_(?P<tf>TF_\d+)_(?P<kind>WT|shuffle)$")
reps = pd.concat([reps, parts], axis=1).dropna(subset=["kind"])

v = pd.read_parquet(ROOT / "data/processed/variant_pairs.parquet")
v["base"] = v["base_sequence"].str.replace(r"_(fwd|rev)$", "", regex=True)
keys = set(zip(v["base"], v["tf_family"]))
reps = reps[[(b, t) in keys for b, t in zip(reps["base"], reps["tf"])]]
print(f"rows matched to our 741 pairs: {len(reps):,} | distinct pairs: {reps[['base', 'tf']].drop_duplicates().shape[0]}",
      flush=True)

# best Experiment 3 model per condition (CNN mean of 3 inits, 6-mer, AgroNT, NT-v2 ridge)
res = pd.concat([pd.read_csv(ROOT / "results/exp3_variants.tsv", sep="\t"),
                 pd.read_csv(ROOT / "results/exp3_cnn.tsv", sep="\t")])
res = res[~res["model"].str.startswith("cnn init")]
best = res.loc[res.groupby("condition")["delta_spearman"].idxmax()].set_index("condition")


def sb(r, k):
    r = max(r, 0.0)
    return k * r / (1 + (k - 1) * r)


rows = []
for c in COND:
    d = reps[reps["condition"] == c]
    W = d.pivot_table(index=["base", "tf"], columns=["experiment", "kind"], values="enrichment")
    N = d.pivot_table(index=["base", "tf"], columns=["experiment", "kind"], values="n_bc")
    exps = [e for e in sorted(d["experiment"].unique()) if (e, "WT") in W and (e, "shuffle") in W]
    k = len(exps)
    for min_bc in [1, 5, 10]:
        ok = {e: (N[(e, "WT")] >= min_bc) & (N[(e, "shuffle")] >= min_bc) for e in exps}
        delta = {e: W[(e, "shuffle")] - W[(e, "WT")] for e in exps}
        r_d, rho_d, r_w, ns = [], [], [], []
        for a, b in combinations(exps, 2):
            m = ok[a] & ok[b] & delta[a].notna() & delta[b].notna()
            if m.sum() >= 30:
                r_d.append(pearsonr(delta[a][m], delta[b][m])[0])
                rho_d.append(spearmanr(delta[a][m], delta[b][m])[0])
                r_w.append(pearsonr(W[(a, "WT")][m], W[(b, "WT")][m])[0])
                ns.append(int(m.sum()))
        if not r_d:
            continue
        rd, rhod, rw = np.mean(r_d), np.mean(rho_d), np.mean(r_w)
        ceil_r, ceil_rho = np.sqrt(sb(rd, k)), np.sqrt(sb(rhod, k))
        row = dict(condition=c, min_bc=min_bc, n_reps=k, n_pairs=min(ns),
                   rep_delta_pearson=rd, rep_delta_spearman=rhod,
                   ceiling_delta_spearman=ceil_rho, ceiling_delta_pearson=ceil_r,
                   rep_wt_pearson=rw, ceiling_wt_r2=sb(rw, k),
                   best_model=best.loc[c, "model"], best_delta_spearman=best.loc[c, "delta_spearman"])
        row["pct_of_ceiling"] = 100 * row["best_delta_spearman"] / ceil_rho if ceil_rho > 0 else np.nan
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/delta_noise_ceiling.tsv", sep="\t", index=False)
cols = ["condition", "min_bc", "n_reps", "n_pairs", "rep_delta_spearman", "ceiling_delta_spearman",
        "best_model", "best_delta_spearman", "pct_of_ceiling", "rep_wt_pearson", "ceiling_wt_r2"]
print("\n######## noise ceiling vs best model (Experiment 3)")
print(out[cols].round(3).to_string(index=False))
print("\nCEILING DONE", flush=True)
