"""Why is the 'no hit' homology bin always the hardest?
Two suspects: (1) different composition (GC, low complexity), (2) smaller spread of measured strength,
which pulls R2 down by itself. Looks at the original test sets only."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "plantcre"
P, S = ROOT / "data/processed", ROOT / "data/splits"
ORDER = ["no hit", "<34 (chance)", "34-40", "40-60", "60-100", ">=100"]


def entropy3(s):
    c = np.array(list(Counter(s[i:i + 3] for i in range(len(s) - 2)).values()), dtype=float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


for dataset, scol, ycol in [("promoters", "split_leaf", "strength_leaf"),
                            ("terminators", "split", "strength_tobacco"),
                            ("enhancers", "split", "strength_light")]:
    unit = "base_id" if dataset == "enhancers" else "id"
    df = pd.read_parquet(P / f"{dataset}.parquet")
    df = df[df[scol] == "test"].copy()
    b = pd.read_parquet(S / f"{dataset}_original_{scol}_homology_bins.parquet")
    df = df.merge(b[[unit, "bin"]], on=unit, how="left")
    df["gc"] = df["sequence"].str.count("[GC]") / df["sequence"].str.len()
    df["entropy3"] = df["sequence"].map(entropy3)
    df["prefix"] = df["id"].astype(str).str.split("[-_]", regex=True).str[0]
    g = df.groupby("bin").agg(n=("id", "size"), gc=("gc", "mean"), entropy3=("entropy3", "mean"),
                              y_mean=(ycol, "mean"), y_sd=(ycol, "std")).reindex(ORDER).round(3)
    print(f"\n######## {dataset} ({ycol}), original test set")
    print(g.to_string())
    top = df[df["bin"] == "no hit"]["prefix"].value_counts().head(6).to_dict()
    allp = df["prefix"].value_counts(normalize=True).head(6).round(3).to_dict()
    print(f"id prefix in 'no hit': {top}")
    print(f"id prefix share in whole test set: {allp}")
