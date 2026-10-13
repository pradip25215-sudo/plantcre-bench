"""Parse the plantGREP validation library (Supplementary Data 3) into a clean table.

Output: data/processed/tfval_raw.parquet
Columns: base_sequence, experiment, exp_group, strength_<condition> x5, gc_pct, sequence
"""
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "plantcre"
X = ROOT / "data/external/plantgrep_enhancers/data/supplementary_data/SupplementaryData3.xlsx"
OUT = ROOT / "data/processed"
CONDITIONS = ["light", "dark", "warm", "cold", "maize"]

raw = pd.read_excel(X, sheet_name="validation library", header=None)
r = raw.index[raw[0].astype(str).str.strip() == "base sequence"][0]
h1 = [str(x).strip() for x in raw.iloc[r].tolist()]
h2 = [str(x).strip() for x in raw.iloc[r + 1].tolist()]
has_sub = any(x in CONDITIONS for x in h2)

names = []
for a, b in zip(h1, h2):
    if b in CONDITIONS:
        names.append(f"strength_{b}")
    elif a == "base sequence":
        names.append("base_sequence")
    elif a.startswith("GC"):
        names.append("gc_pct")
    else:
        names.append(a.replace(" ", "_"))
print("column names:", names)

tf = raw.iloc[r + (2 if has_sub else 1):].copy()
tf.columns = names
tf = tf.dropna(how="all").reset_index(drop=True)
for c in [f"strength_{c}" for c in CONDITIONS] + ["gc_pct"]:
    tf[c] = pd.to_numeric(tf[c], errors="coerce")
for c in ["base_sequence", "experiment", "sequence"]:
    tf[c] = tf[c].astype(str).str.strip()
tf["exp_group"] = tf["experiment"].str.split(":").str[0].str.strip()

lens = tf["sequence"].str.len()
print(f"\nrows={len(tf):,} | sequence length min/median/max = {lens.min()}/{int(lens.median())}/{lens.max()}")
print("\n=== experiment groups ===")
print(tf["exp_group"].value_counts().to_string())

non_evo = tf[tf["exp_group"] != "evolution"]
print("\n=== every non-evolution experiment label ===")
print(non_evo["experiment"].value_counts().to_string())

print("\n=== 3 example rows per non-evolution experiment ===")
for label, d in non_evo.groupby("experiment"):
    print(f"\n--- {label}")
    ex = d.head(3)[["base_sequence", "strength_light", "strength_dark", "strength_maize", "sequence"]].copy()
    ex["sequence"] = ex["sequence"].str.slice(0, 30) + "..."
    print(ex.to_string(index=False))

tf.to_parquet(OUT / "tfval_raw.parquet", index=False)
print("\nSaved", OUT / "tfval_raw.parquet")
