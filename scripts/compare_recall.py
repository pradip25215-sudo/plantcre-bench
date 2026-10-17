"""Word size 7 (slow, max recall) vs word size 11 (fast) on the same pilot.

recall = of all homolog pairs found with word 7 at a given threshold, how many does word 11 also find.
Also checks the 'giant group' risk: with union-find, weak links can chain everything together.
"""
from pathlib import Path

import pandas as pd

H = Path.home() / "plantcre/data/homology"


def pairs(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["q", "s", "score"])
    df["q"] = df["q"].str.replace("_Reversed", "", regex=False)
    df["s"] = df["s"].str.replace("_Reversed", "", regex=False)
    df = df[df["q"] != df["s"]]
    a = df[["q", "s"]].min(axis=1)
    b = df[["q", "s"]].max(axis=1)
    return df.assign(a=a, b=b).groupby(["a", "b"])["score"].max()


def groups(edges, n_total):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    sizes = pd.Series([find(x) for x in list(parent)]).value_counts()
    in_groups = sizes.sum()
    return sizes.max() if len(sizes) else 0, len(sizes), n_total - in_groups


w7 = pairs(H / "pilot_real.work/pilot_real.processed.tsv")
w11 = pairs(H / "pilot_real_w11.work/pilot_real_w11.processed.tsv")
n = 6000
print(f"{'thr':>4} | {'pairs w7':>9} | {'pairs w11':>9} | {'recall':>7} | largest group | n groups | singletons")
for thr in [34, 40, 60]:
    s7 = set(w7[w7 >= thr].index)
    s11 = set(w11[w11 >= thr].index)
    rec = len(s7 & s11) / max(len(s7), 1)
    big, ng, single = groups(s7, n)
    print(f"{thr:>4} | {len(s7):>9,} | {len(s11):>9,} | {rec:>7.4f} | {big:>13,} | {ng:>8,} | {single:>10,}")
