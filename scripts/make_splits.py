"""Homology-aware splits + leakage numbers for the published splits.

For each dataset and threshold (34, 40, 60):
  1. homology groups = connected components of pairs with corrected score >= threshold (union-find)
  2. new splits: whole groups go to train/val/test (80/10/10), 5 seeds
     - enhancers: fwd and rev copies share a split (we work on base sequences)
     - enhancers: every group holding a variant-pair WT goes to TEST (decision D3)
     - promoters: one shared split for leaf and proto (decision D1)
  3. leakage of the ORIGINAL splits: for each original test sequence, best score to any original
     train sequence -> % leaked, and a homology bin for stratified evaluation (Experiment 1)

usage: python make_splits.py <dataset> [<dataset> ...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "plantcre"
H = ROOT / "data/homology"
P = ROOT / "data/processed"
OUT = ROOT / "data/splits"
OUT.mkdir(parents=True, exist_ok=True)
NOFORCE = os.environ.get("NOFORCE") == "1"
TAG = "_nf" if NOFORCE else ""
THRESHOLDS = [34, 40, 60]
SEEDS = [1, 2, 3, 4, 5]
FRAC = {"train": 0.8, "val": 0.1, "test": 0.1}
BINS = [-1, 0, 33.999, 39.999, 59.999, 99.999, 1e9]
BIN_NAMES = ["no hit", "<34 (chance)", "34-40", "40-60", "60-100", ">=100"]


def load_pairs(dataset):
    df = pd.read_csv(H / f"{dataset}_min20.work/{dataset}_min20.processed.tsv",
                     sep="\t", header=None, names=["q", "s", "score"])
    for c in ("q", "s"):
        df[c] = df[c].str.replace("_Reversed", "", regex=False)
    df = df[df["q"] != df["s"]]
    a, b = df[["q", "s"]].min(axis=1), df[["q", "s"]].max(axis=1)
    return df.assign(a=a, b=b).groupby(["a", "b"], as_index=False)["score"].max()


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return pd.Series({n: find(n) for n in nodes})


def assign_groups(group_sizes, forced_test, seed):
    """Greedy: shuffle groups, give each one to the split that is furthest below its target size."""
    rng = np.random.default_rng(seed)
    total = group_sizes.sum()
    fill = {k: 0 for k in FRAC}
    out = {}
    for g in forced_test:
        out[g] = "test"
        fill["test"] += group_sizes[g]
    rest = [g for g in group_sizes.index if g not in forced_test]
    rng.shuffle(rest)
    for g in rest:
        k = min(FRAC, key=lambda s: fill[s] / (FRAC[s] * total))
        out[g] = k
        fill[k] += group_sizes[g]
    return pd.Series(out)


for dataset in sys.argv[1:]:
    print(f"\n################ {dataset}", flush=True)
    idmap = pd.read_parquet(H / f"{dataset}_idmap.parquet")
    h2id = dict(zip(idmap["header"], idmap["orig_id"]))
    pairs = load_pairs(dataset)
    pairs["ia"], pairs["ib"] = pairs["a"].map(h2id), pairs["b"].map(h2id)
    nodes = idmap["orig_id"].tolist()
    print(f"sequences={len(nodes):,} | candidate pairs (score>=20)={len(pairs):,}", flush=True)

    df = pd.read_parquet(P / f"{dataset}.parquet")
    unit = "base_id" if dataset == "enhancers" else "id"

    forced = set()
    if dataset == "enhancers" and not NOFORCE:
        v = pd.read_parquet(P / "variant_pairs.parquet")
        forced = set(v["base_sequence"].str.replace(r"_(fwd|rev)$", "", regex=True))

    for thr in THRESHOLDS:
        e = pairs[pairs["score"] >= thr]
        comp = components(nodes, zip(e["ia"], e["ib"]))
        sizes = comp.value_counts()
        in_multi = (comp.map(sizes) > 1).mean()
        print(f"\n-- threshold {thr}: {len(e):,} homologous pairs | groups={len(sizes):,} | "
              f"largest group={sizes.max():,} ({100 * sizes.max() / len(nodes):.2f}% of data) | "
              f"sequences with >=1 homolog={100 * in_multi:.1f}%", flush=True)

        forced_groups = set(comp[comp.index.isin(forced)].values)
        if forced:
            n_forced = comp.isin(forced_groups).sum()
            print(f"   variant-WT groups forced to test: {len(forced_groups)} groups, {n_forced:,} sequences", flush=True)

        for seed in SEEDS:
            g2s = assign_groups(sizes, forced_groups, seed)
            sp = comp.map(g2s)
            res = pd.DataFrame({unit: comp.index, "group": comp.values, "split": sp.values})
            res = df[["id"] + ([unit] if unit != "id" else [])].merge(res, on=unit, how="left")
            res.to_parquet(OUT / f"{dataset}{TAG}_t{thr}_s{seed}.parquet", index=False)
            if seed == 1:
                frac = res["split"].value_counts(normalize=True).round(3).to_dict()
                s_map = dict(zip(comp.index, sp.values))
                cross = sum(1 for a, b in zip(e["ia"], e["ib"])
                            if {s_map[a], s_map[b]} == {"train", "test"})
                print(f"   seed 1 split fractions={frac} | homologous pairs crossing train/test={cross} (must be 0)", flush=True)

    # leakage in the ORIGINAL published splits
    split_cols = ["split_leaf", "split_proto"] if dataset == "promoters" else ["split"]
    for col in split_cols:
        orig = df.drop_duplicates(unit).set_index(unit)[col]
        tr_set = set(orig[orig == "train"].index)
        te_ids = orig[orig == "test"].index
        best = {}
        for a, b, s in zip(pairs["ia"], pairs["ib"], pairs["score"]):
            if a in tr_set and b not in tr_set:
                best[b] = max(best.get(b, 0), s)
            elif b in tr_set and a not in tr_set:
                best[a] = max(best.get(a, 0), s)
        mx = pd.Series({i: best.get(i, 0.0) for i in te_ids})
        binned = pd.cut(mx.replace(0, -0.5), BINS, labels=BIN_NAMES)
        print(f"\n== ORIGINAL split ({col}): {len(te_ids):,} test sequences", flush=True)
        for t in THRESHOLDS:
            print(f"   test sequences with a train homolog at >= {t}: {(mx >= t).sum():,} ({100 * (mx >= t).mean():.2f}%)", flush=True)
        print("   max-score-to-train bins:", binned.value_counts().reindex(BIN_NAMES).to_dict(), flush=True)
        pd.DataFrame({unit: mx.index, "max_score_to_train": mx.values, "bin": binned.astype(str).values}) \
          .to_parquet(OUT / f"{dataset}_original_{col}_homology_bins.parquet", index=False)

print("\nSPLITS DONE", flush=True)
