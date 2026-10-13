"""Build WT/mutant pairs from the plantGREP TFBS-shuffling experiment (Experiment 3 ground truth).

Output: data/processed/variant_pairs.parquet
One row per (wild-type, shuffled-TFBS) pair with measured strengths, delta (mut - WT),
mutation span, and the WT's split / strength in the main ACR library.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path.home() / "plantcre"
P = ROOT / "data/processed"
C = ["light", "dark", "warm", "cold", "maize"]
S = [f"strength_{c}" for c in C]

tf = pd.read_parquet(P / "tfval_raw.parquet")
enh = pd.read_parquet(P / "enhancers.parquet")

# ---------- 1. WT and mutant rows ----------
wt = tf[tf["experiment"] == "unmodified control"].copy()
mu = tf[tf["experiment"].str.startswith("TFBS shuffling")].copy()
mu["tf_family"] = mu["experiment"].str.split(":").str[1].str.strip()
print(f"WT rows={len(wt):,} (unique base={wt['base_sequence'].nunique():,}) | "
      f"mutant rows={len(mu):,} (unique base={mu['base_sequence'].nunique():,})")

wt1 = wt.drop_duplicates("base_sequence")[["base_sequence", "sequence"] + S]
pairs = mu[["base_sequence", "tf_family", "sequence"] + S].merge(
    wt1, on="base_sequence", how="left", suffixes=("_mut", "_wt"))
missing = pairs["sequence_wt"].isna().sum()
print(f"mutants without a matching WT: {missing}")
pairs = pairs.dropna(subset=["sequence_wt"]).reset_index(drop=True)
pairs = pairs.rename(columns={"sequence": "seq_mut"}) if "sequence" in pairs else pairs
pairs = pairs.rename(columns={"sequence_mut": "seq_mut", "sequence_wt": "seq_wt"})

# ---------- 2. Where exactly is the mutation? ----------
def diff_info(a, b):
    pos = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not pos:
        return 0, -1, -1
    return len(pos), min(pos), max(pos)

info = [diff_info(a, b) for a, b in zip(pairs["seq_wt"], pairs["seq_mut"])]
pairs["n_diff"] = [i[0] for i in info]
pairs["mut_start"] = [i[1] for i in info]
pairs["mut_end"] = [i[2] for i in info]
pairs["mut_span"] = pairs["mut_end"] - pairs["mut_start"] + 1
print("\n=== mutation size (bases changed) ===")
print(pairs["n_diff"].describe().round(1).to_string())
print(f"pairs with NO difference (should be 0): {(pairs['n_diff'] == 0).sum()}")

# ---------- 3. Measured effect of each mutation ----------
for c in C:
    pairs[f"delta_{c}"] = pairs[f"strength_{c}_mut"] - pairs[f"strength_{c}_wt"]

print("\n=== measured effects: fraction of pairs where shuffling LOWERS strength ===")
for c in C:
    d = pairs[f"delta_{c}"].dropna()
    print(f"{c:6s} n={len(d):4d} | fraction delta<0 = {(d < 0).mean():.2f} | median delta = {d.median():+.2f}")

print("\n=== median delta per TF family (light, maize) ===")
print(pairs.groupby("tf_family")[["delta_light", "delta_maize"]].median().round(2)
      .sort_values("delta_light").to_string())

# ---------- 4. Link WT to the main ACR library ----------
lib = enh[["id", "split", "sequence"] + S].rename(
    columns={"id": "base_sequence", "split": "lib_split", "sequence": "lib_seq",
             **{s: f"lib_{s}" for s in S}})
pairs = pairs.merge(lib, on="base_sequence", how="left")
found = pairs["lib_split"].notna()
print(f"\nWT found in main ACR library: {found.sum()} of {len(pairs)}")
print("WT split in main library:", pairs["lib_split"].value_counts(dropna=False).to_dict())
same = (pairs.loc[found, "lib_seq"] == pairs.loc[found, "seq_wt"]).mean()
print(f"WT sequence identical to library sequence: {100 * same:.1f}%")

lib_seqs = set(enh["sequence"])
print(f"mutant sequences that also appear in the main library (should be 0): "
      f"{pairs['seq_mut'].isin(lib_seqs).sum()}")

# ---------- 5. Noise ceiling: same WT measured in two libraries ----------
print("\n=== assay reproducibility: WT strength, validation library vs ACR library ===")
w = pairs[found].drop_duplicates("base_sequence")
for c in C:
    a, b = w[f"strength_{c}_wt"], w[f"lib_strength_{c}"]
    ok = a.notna() & b.notna()
    r = pearsonr(a[ok], b[ok])[0]
    rho = spearmanr(a[ok], b[ok])[0]
    print(f"{c:6s} n={ok.sum():4d} | Pearson r={r:.3f} (R2={r * r:.3f}) | Spearman rho={rho:.3f}")

pairs = pairs.drop(columns=["lib_seq"])
pairs.to_parquet(P / "variant_pairs.parquet", index=False)
print(f"\nSaved {len(pairs)} pairs -> {P / 'variant_pairs.parquet'}")

# ---------- 6. Quick look at the insertion library (bonus, not paired yet) ----------
ins = tf[tf["exp_group"] == "TFBS insertion"]
bases = ins["base_sequence"].unique()
print(f"\n=== insertion library: {len(ins):,} rows on {len(bases)} random backgrounds ===")
print("all-'none' constructs present:", ins["experiment"].str.contains("none, none, none").any())
other = tf[tf["base_sequence"].isin(bases) & (tf["exp_group"] != "TFBS insertion")]
print("same backgrounds also measured in:", other["experiment"].value_counts().head(5).to_dict())
