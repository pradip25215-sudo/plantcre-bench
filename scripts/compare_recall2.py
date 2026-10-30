"""Recall of fast BLAST settings vs the slow word-7/evalue-100 baseline, at each threshold."""
from pathlib import Path

import pandas as pd

H = Path.home() / "plantcre/data/homology"


def pairs(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["q", "s", "score"])
    for c in ("q", "s"):
        df[c] = df[c].str.replace("_Reversed", "", regex=False)
    df = df[df["q"] != df["s"]]
    a, b = df[["q", "s"]].min(axis=1), df[["q", "s"]].max(axis=1)
    return df.assign(a=a, b=b).groupby(["a", "b"])["score"].max()


base = pairs(H / "pilot_real.work/pilot_real.processed.tsv")
cands = {"w11_e10": "pilot_real_w11.work/pilot_real_w11.processed.tsv",
         "w11_e100": "pilot_real_w11_e100.work/pilot_real_w11_e100.processed.tsv",
         "w11_e1000": "pilot_real_w11_e1000.work/pilot_real_w11_e1000.processed.tsv",
         "w9_e1000": "pilot_real_w9_e1000.work/pilot_real_w9_e1000.processed.tsv"}
print(f"{'setting':>10} | " + " | ".join(f"recall@{t}" for t in (34, 40, 60)))
for name, rel in cands.items():
    c = pairs(H / rel)
    recs = []
    for t in (34, 40, 60):
        s0 = set(base[base >= t].index)
        s1 = set(c[c >= t].index)
        recs.append(len(s0 & s1) / max(len(s0), 1))
    print(f"{name:>10} | " + " | ".join(f"{r:>9.4f}" for r in recs))
