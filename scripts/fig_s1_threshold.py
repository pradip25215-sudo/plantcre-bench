"""Supplementary figure: where the homology threshold comes from.
Best-hit alignment score of 6,000 natural sequences and of their dinucleotide-shuffled copies
(2,000 from each dataset), from the pilot BLAST run. Sequences with no hit count as 0."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path.home() / "plantcre"
H, OUT = ROOT / "data/homology", ROOT / "figures"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
mp = pd.read_parquet(H / "pilot_map.parquet")


def best_hits(name):
    df = pd.read_csv(H / f"{name}.work/{name}.processed.tsv", sep="\t", header=None, names=["q", "s", "score"])
    for c in ("q", "s"):
        df[c] = df[c].str.replace("_Reversed", "", regex=False)
    return df[df["q"] != df["s"]].groupby("q")["score"].max()


real, shuf = best_hits("pilot_real"), best_hits("pilot_shuf")
R = real.reindex(mp["real"]).fillna(0).to_numpy()
S = shuf.reindex(mp["shuf"]).fillna(0).to_numpy()
src = mp["src"].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
bins = np.arange(0, 175, 5)
ax = axes[0]
ax.hist(S, bins=bins, color="#9e9e9e", label=f"dinucleotide-shuffled (max {S.max():.0f})")
ax.hist(R, bins=bins, histtype="step", color="#d95f02", lw=1.4, label="natural sequences")
for t, ls in [(34, ":"), (40, "-"), (60, "--")]:
    ax.axvline(t, color="k", lw=0.8, ls=ls)
    ax.text(t, ax.get_ylim()[1] * 0.97, str(t), fontsize=6, ha="center", va="top",
            bbox=dict(fc="w", ec="none", pad=0.6))
ax.set_yscale("log")
ax.set_xlabel("best alignment score to any other sequence")
ax.set_ylabel("sequences (log scale)")
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax.text(-0.14, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)

ax = axes[1]
rows = []
for s in ["promoter", "terminator", "enhancer"]:
    m = src == s
    for t in (34, 40, 60):
        rows.append(dict(src=s, thr=t, pct=100 * (R[m] >= t).mean()))
d = pd.DataFrame(rows)
for k, (t, col) in enumerate([(34, "#fdd0a2"), (40, "#fd8d3c"), (60, "#a63603")]):
    g = d[d["thr"] == t]
    ax.bar(np.arange(3) + (k - 1) * 0.25, g["pct"], width=0.25, color=col, label=f"score >= {t}")
ax.set_xticks(range(3))
ax.set_xticklabels(["promoters", "terminators", "enhancers"])
ax.set_ylabel("% of natural sequences\nwith a homolog in the pilot")
ax.legend(frameon=False, fontsize=6.5)
ax.text(-0.18, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figS1_threshold.{ext}", bbox_inches="tight")
print(f"shuffled: p99.9 = {np.percentile(S, 99.9):.0f}, max = {S.max():.0f}")
print(f"natural:  p99 = {np.percentile(R, 99):.0f}, p99.9 = {np.percentile(R, 99.9):.0f}, max = {R.max():.0f}")
print(d.pivot(index="src", columns="thr", values="pct").round(1).to_string())
print("saved figures/figS1_threshold.png and .pdf")
