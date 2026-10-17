"""Pick the hashFrag homology threshold from the pilot.

For every query we take its best hit to some OTHER sequence (self and own reverse complement removed).
Shuffled sequences tell us what 'chance' looks like; the threshold goes above their tail.
"""
from pathlib import Path

import numpy as np
import pandas as pd

H = Path.home() / "plantcre/data/homology"
mp = pd.read_parquet(H / "pilot_map.parquet")


def best_hits(name):
    df = pd.read_csv(H / f"{name}.work/{name}.processed.tsv", sep="\t", header=None, names=["q", "s", "score"])
    df["q"] = df["q"].str.replace("_Reversed", "", regex=False)
    df["s"] = df["s"].str.replace("_Reversed", "", regex=False)
    df = df[df["q"] != df["s"]]
    return df.groupby("q")["score"].max()


real, shuf = best_hits("pilot_real"), best_hits("pilot_shuf")
qs = [50, 90, 99, 99.9, 100]
print("best-hit score percentiles (sequences with no hit at all count as 0)")
for label, s, heads in [("shuffled", shuf, mp["shuf"]), ("real", real, mp["real"])]:
    full = s.reindex(heads).fillna(0).to_numpy()
    print(f"  {label:8s}: " + " | ".join(f"p{q}={np.percentile(full, q):.0f}" for q in qs))

print("\nper dataset (shuffled p99.9 / real fraction above 40, 60, 80):")
for src, d in mp.groupby("src"):
    sh = shuf.reindex(d["shuf"]).fillna(0).to_numpy()
    re_ = real.reindex(d["real"]).fillna(0).to_numpy()
    print(f"  {src:10s} shuffled p99.9={np.percentile(sh, 99.9):.0f} | "
          f"real >40: {(re_ > 40).mean():.3f}  >60: {(re_ > 60).mean():.3f}  >80: {(re_ > 80).mean():.3f}")

sh_all = shuf.reindex(mp["shuf"]).fillna(0).to_numpy()
thr = int(np.ceil(max(np.percentile(sh_all, 99.9), shuf.max() if len(shuf) else 0)))
print(f"\nSUGGESTED THRESHOLD (max of shuffled scores, rounded up): {thr}")
