"""Print columns, row count and first rows of every key data file."""
import pandas as pd
from pathlib import Path

EXT = Path.home() / "plantcre/data/external"
FILES = [
    "jores2021_promoters/CNN/CNN_train_leaf.tsv",
    "jores2021_promoters/CNN/CNN_test_leaf.tsv",
    "jores2021_promoters/CNN/CNN_train_proto.tsv",
    "jores2021_promoters/CNN/CNN_test_proto.tsv",
    "jores2021_promoters/CNN/evolution_data.tsv",
    "jores2021_promoters/data/promoter_annotation/Maize_all_promoters_unique.tsv",
    "plantgrep_enhancers/data/modelling_data/modelling_data_tamsACR.tsv.gz",
    "plantgrep_enhancers/data/modelling_data/plantGREP_predictions.tsv",
    "plantgrep_enhancers/data/modelling_data/lm_kmers_predictions.tsv.gz",
    "plantgrep_enhancers/data/modelling_data/evolution_data.tsv.gz",
    "gorjifard2024_terminators/CNN/terminator_data.tsv",
    "gorjifard2024_terminators/CNN/evolution_data.tsv",
]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

def short(x):
    s = str(x)
    return s[:40] + "..." if len(s) > 40 else s

for f in FILES:
    path = EXT / f
    print("\n" + "=" * 100)
    print(f)
    print("=" * 100)
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        print(f"rows: {len(df):,} | columns: {len(df.columns)}")
        print("columns:", list(df.columns))
        print(df.head(3).apply(lambda col: col.map(short)).to_string())
    except Exception as e:
        print("ERROR:", type(e).__name__, e)

print("\n" + "=" * 100)
print("SupplementaryData3.xlsx sheets")
print("=" * 100)
xlsx = EXT / "plantgrep_enhancers/data/supplementary_data/SupplementaryData3.xlsx"
for name, df in pd.read_excel(xlsx, sheet_name=None, nrows=3).items():
    print(f"\n--- sheet: {name} | columns: {list(df.columns)}")
    print(df.apply(lambda col: col.map(short)).to_string())

print("\n" + "=" * 100)
print("TFval_sequences.fa (first 6 lines)")
print("=" * 100)
with open(EXT / "plantgrep_enhancers/data/refseq/TFval_sequences.fa") as fh:
    for i, line in enumerate(fh):
        if i >= 6:
            break
        print(short(line.strip()))
