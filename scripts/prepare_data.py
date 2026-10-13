"""Build clean, standard tables from the three Plant STARR-seq datasets.

Outputs (data/processed/):
  promoters.parquet    Jores 2021 core promoters (leaf + proto strength, per-assay split)
  enhancers.parquet    plantGREP ACR library (5 conditions, original split, orientation)
  terminators.parquet  Gorjifard 2024 terminators (tobacco + maize strength, split)
  tfval_raw.parquet    plantGREP TF-site validation library, parsed but not yet paired
"""
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "plantcre"
EXT = ROOT / "data/external"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)


def check_seqs(df, name):
    s = df["sequence"].astype(str).str.upper()
    lens = s.str.len()
    bad = ~s.str.fullmatch(r"[ACGT]+")
    print(f"[{name}] rows={len(df):,} | length min/median/max = "
          f"{lens.min()}/{int(lens.median())}/{lens.max()} | sequences with non-ACGT = {bad.sum():,}")


def exact_leak(df, split_col, name):
    train = set(df.loc[df[split_col] == "train", "sequence"])
    test = df.loc[df[split_col] == "test", "sequence"]
    n = test.isin(train).sum()
    print(f"[{name}] test sequences IDENTICAL to a train sequence ({split_col}): "
          f"{n:,} of {len(test):,} ({100 * n / max(len(test), 1):.2f}%)")


def make_unique(ids):
    dup = ids.groupby(ids).cumcount()
    return ids.where(dup == 0, ids + "_dup" + dup.astype(str))


# ---------------- 1. Promoters (Jores 2021) ----------------
print("\n######## PROMOTERS ########")
P = EXT / "jores2021_promoters/CNN"


def load_prom(assay):
    d = pd.concat([pd.read_csv(P / f"CNN_train_{assay}.tsv", sep="\t"),
                   pd.read_csv(P / f"CNN_test_{assay}.tsv", sep="\t")])
    print(f"{assay}: rows={len(d):,} | duplicated (sp, gene, sequence) = "
          f"{d.duplicated(['sp', 'gene', 'sequence']).sum():,}")
    print(d.groupby(["set", "sp"]).size().unstack(fill_value=0).to_string())
    return d.rename(columns={"set": f"split_{assay}", "enrichment": f"strength_{assay}"})


prom = load_prom("leaf").merge(load_prom("proto"), on=["sp", "gene", "sequence"], how="outer")
prom = prom.rename(columns={"sp": "species"})
prom.insert(0, "id", make_unique("PRO_" + prom["species"] + "_" + prom["gene"].astype(str)))
prom.insert(1, "element", "promoter")
both = prom["split_leaf"].notna() & prom["split_proto"].notna()
disagree = both & (prom["split_leaf"] != prom["split_proto"])
print(f"merged rows={len(prom):,} | in both assays={both.sum():,} | "
      f"split differs between leaf and proto={disagree.sum():,}")
check_seqs(prom, "promoters")
exact_leak(prom, "split_leaf", "promoters")
exact_leak(prom, "split_proto", "promoters")
prom.to_parquet(OUT / "promoters.parquet", index=False)

# ---------------- 2. Enhancers (plantGREP) ----------------
print("\n######## ENHANCERS ########")
E = pd.read_csv(EXT / "plantgrep_enhancers/data/modelling_data/modelling_data_tamsACR.tsv.gz", sep="\t")
E = E.rename(columns={c: c.replace("enrichment_", "strength_") for c in E.columns})
E = E.rename(columns={"set": "split"})
E["orientation"] = E["id"].str.extract(r"_(fwd|rev)$")[0]
E["base_id"] = E["id"].str.replace(r"_(fwd|rev)$", "", regex=True)
E["species"] = E["id"].str.extract(r"^(At|Zm|Sb|Sl)-")[0].fillna("other")
E.insert(1, "element", "enhancer")
print(E.groupby(["split", "species"]).size().unstack(fill_value=0).to_string())
print("orientation counts:", E["orientation"].value_counts(dropna=False).to_dict())
mixed = (E.groupby("base_id")["split"].nunique() > 1).sum()
print(f"base sequences whose fwd/rev copies sit in DIFFERENT splits: {mixed:,}")
print("examples of species='other' ids:", E.loc[E["species"] == "other", "id"].head(10).tolist())
check_seqs(E, "enhancers")
exact_leak(E, "split", "enhancers")
E.to_parquet(OUT / "enhancers.parquet", index=False)

# ---------------- 3. Terminators (Gorjifard 2024) ----------------
print("\n######## TERMINATORS ########")
T = pd.read_csv(EXT / "gorjifard2024_terminators/CNN/terminator_data.tsv", sep="\t")
T = T.rename(columns={"enrichment_tobacco": "strength_tobacco",
                      "enrichment_maize": "strength_maize", "set": "split"})
T["is_synthetic_control"] = T["id"].str.contains("random|shuffl|control", case=False)
T.insert(1, "element", "terminator")
print(T.groupby(["split", "species"]).size().unstack(fill_value=0).to_string())
print(f"synthetic/control sequences: {T['is_synthetic_control'].sum():,}")
check_seqs(T, "terminators")
exact_leak(T, "split", "terminators")
T.to_parquet(OUT / "terminators.parquet", index=False)

# ---------------- 4. TF-site validation library (parse only) ----------------
print("\n######## TF VALIDATION LIBRARY ########")
X = EXT / "plantgrep_enhancers/data/supplementary_data/SupplementaryData3.xlsx"
raw = pd.read_excel(X, sheet_name="validation library", header=None)
r = raw.index[raw[0].astype(str).str.strip() == "base sequence"][0]
h1, h2 = raw.iloc[r].tolist(), raw.iloc[r + 1].tolist()
print("header row 1:", h1)
print("header row 2:", h2)
has_sub = pd.isna(h2[0])
names = []
for a, b in zip(h1, h2):
    a = str(a).strip()
    if a.startswith("log2") and has_sub:
        names.append("log2_" + str(b).strip())
    else:
        names.append(a.replace(" ", "_").replace("(%)", "pct").strip("_"))
tf = raw.iloc[r + (2 if has_sub else 1):].copy()
tf.columns = names
tf = tf.dropna(how="all")
for c in tf.columns:
    if c.startswith("log2_") or c.startswith("GC"):
        tf[c] = pd.to_numeric(tf[c], errors="coerce")
    else:
        tf[c] = tf[c].astype(str)
print(f"rows={len(tf):,} | columns={list(tf.columns)}")
print("experiment values:", tf["experiment"].value_counts().head(20).to_dict())
show = tf.head(8).copy()
for c in show.columns:
    show[c] = show[c].astype(str).str.slice(0, 35)
print(show.to_string())
tf.to_parquet(OUT / "tfval_raw.parquet", index=False)

print("\nDONE. Files written to", OUT)
