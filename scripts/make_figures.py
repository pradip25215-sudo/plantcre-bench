"""Main figures (draft 2). Reads results/*.tsv and data/splits/*_homology_bins.parquet,
writes figures/<name>.png and .pdf.
  fig1_leakage      (a) % of published test sequences with a train homolog
                    (b) test R2 per seed on random vs homology splits, every model (assays averaged)
  fig2_composition  Exp 2: R2 the non-composition part of the embedding adds on top of 6-mers
  fig3_variants     Exp 3: delta Spearman per condition and model, with the replicate noise ceiling
  fig4_wt_vs_delta  % of the reachable Pearson r, WT strength vs mutation effect
Summaries use the median over seeds (one CNN run collapsed; a mean would be dragged by it).
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path.home() / "plantcre"
R, S, OUT = ROOT / "results", ROOT / "data/splits", ROOT / "figures"
OUT.mkdir(exist_ok=True)
COND = ["light", "dark", "warm", "cold", "maize"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
COL = {"ntv2_100m": "#7570b3", "agront_1b": "#e7298a", "6mer": "#66a61e", "cnn": "#e6ab02", "ft": "#1f78b4"}


def rd(name):
    return pd.read_csv(R / name, sep="\t")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}.png and .pdf", flush=True)


# ---------------- Fig 1 ----------------
leak = []
for label, f in [("Promoters\n(leaf)", "promoters_original_split_leaf"),
                 ("Promoters\n(proto)", "promoters_original_split_proto"),
                 ("Terminators", "terminators_original_split"),
                 ("Enhancers", "enhancers_original_split")]:
    b = pd.read_parquet(S / f"{f}_homology_bins.parquet")
    for t in (34, 40, 60):
        leak.append(dict(set=label, thr=t, pct=100 * (b["max_score_to_train"] >= t).mean(), n=len(b)))
leak = pd.DataFrame(leak)


def frozen(tsv):
    x = rd(tsv)
    return x[(x["bin"] == "all") & (x["split"] != "original")][["dataset", "assay", "model", "split", "seed", "r2"]]


def cnn(names):
    x = pd.concat([rd(n) for n in names if (R / n).exists()])
    return x[(x["bin"] == "all") & (x["split"] != "original")][["dataset", "assay", "model", "split", "seed", "r2"]]


def ft(dataset):
    files = sorted(R.glob("finetune_ntv2_terminators*.tsv")) if dataset == "terminators" else [R / "finetune_ntv2.tsv"]
    x = pd.concat([pd.read_csv(f, sep="\t") for f in files])
    x = x[(x["dataset"] == dataset) & (x["metric"] == "test_r2")].rename(columns={"value": "r2"})
    return x.assign(model="ft")[["dataset", "assay", "model", "split", "seed", "r2"]]


E = pd.concat([
    frozen("exp1_promoters.tsv"), frozen("exp1_terminators.tsv"), frozen("exp1_enhancers_nf.tsv"),
    cnn(["cnn_exp1_promoters.tsv", "cnn_exp1_promoters_seeds45.tsv"]),
    cnn(["cnn_exp1_terminators.tsv", "cnn_exp1_terminators_seeds45.tsv"]),
    cnn(["cnn_exp1_enhancers_nf.tsv"]),
    ft("promoters"), ft("terminators"),
])
LABEL = {"6mer": "6-mer\nridge", "ntv2_100m": "NT-v2\nfrozen", "agront_1b": "AgroNT\nfrozen", "cnn": "CNN",
         "ft": "NT-v2\nfine-tuned"}
MODELS = ["6mer", "ntv2_100m", "agront_1b", "cnn", "ft"]
SPLITS = [("random", "#9e9e9e", "random"), ("homology_t40", "#d95f02", "homology t40"),
          ("homology_t60", "#1b9e77", "homology t60")]
G = E.groupby(["dataset", "model", "split", "seed"], as_index=False)["r2"].mean()  # average over assays

fig = plt.figure(figsize=(7.2, 4.8))
ax = fig.add_subplot(2, 1, 1)
sets = list(dict.fromkeys(leak["set"]))
for k, (t, col) in enumerate([(34, "#fdd0a2"), (40, "#fd8d3c"), (60, "#a63603")]):
    d = leak[leak["thr"] == t].set_index("set").loc[sets]
    ax.bar(np.arange(len(sets)) + (k - 1) * 0.25, d["pct"], width=0.25, color=col, label=f"score >= {t}")
ns = leak.drop_duplicates("set").set_index("set").loc[sets, "n"]
ax.set_xticks(np.arange(len(sets)))
ax.set_xticklabels([f"{s}\nn = {n:,}" for s, n in zip(sets, ns)])
ax.set_ylabel("% of published test set\nwith a train homolog")
ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_ylim(0, 52)
ax.text(-0.08, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)

summary = []
for j, ds in enumerate(["promoters", "terminators", "enhancers"]):
    ax = fig.add_subplot(2, 3, 4 + j)
    present = [m for m in MODELS if ((G["dataset"] == ds) & (G["model"] == m)).any()]
    labels = []
    for i, m in enumerate(present):
        n_seeds = G[(G["dataset"] == ds) & (G["model"] == m) & (G["split"] == "random")]["seed"].nunique()
        labels.append(LABEL[m] + (f"\n({n_seeds} seed)" if n_seeds == 1 else ""))
        for k, (sp, col, _) in enumerate(SPLITS):
            v = G[(G["dataset"] == ds) & (G["model"] == m) & (G["split"] == sp)]["r2"].to_numpy()
            if len(v) == 0:
                continue
            x = i * 4 + k
            ax.scatter(np.full(len(v), x) + np.linspace(-0.15, 0.15, len(v)), v, s=6, color=col, zorder=3)
            med = np.median(v)
            ax.plot([x - 0.35, x + 0.35], [med] * 2, color=col, lw=1.5)
            if m == "cnn" and v.min() < med - 0.15:
                ax.annotate("collapsed\nrun", (x, v.min()), xytext=(6, 0), textcoords="offset points",
                            fontsize=5.5, va="center")
            summary.append(dict(dataset=ds, model=m, split=sp, n_seeds=len(v), median_r2=med))
    ax.set_xticks([i * 4 + 1 for i in range(len(present))])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title(ds, loc="left")
    if j == 0:
        ax.set_ylabel("test R2 (assays averaged)")
        ax.text(-0.3, 1.08, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)
handles = [plt.Line2D([], [], color=c, marker="o", ls="-", ms=3, label=l) for _, c, l in SPLITS]
fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, "fig1_leakage")
summ = pd.DataFrame(summary).pivot_table(index=["dataset", "model"], columns="split", values="median_r2")
summ["drop_t40"] = summ["random"] - summ["homology_t40"]
summ["drop_t60"] = summ["random"] - summ["homology_t60"]
print("\nFig 1b medians (assays averaged):\n" + summ.round(3).to_string())
summ.round(4).to_csv(R / "fig1_medians.tsv", sep="\t")

# ---------------- Fig 2 ----------------
x = rd("exp2_decompose.tsv")
fig, ax = plt.subplots(figsize=(4.2, 2.8))
dsets = ["promoters", "terminators", "enhancers"]
for k, (m, lab) in enumerate([("ntv2_100m", "NT-v2 (layer 10)"), ("agront_1b", "AgroNT (layer 12)")]):
    for i, ds in enumerate(dsets):
        d = x[(x["dataset"] == ds) & (x["model"] == m)]
        xx = i + (k - 0.5) * 0.35
        ax.bar(xx, d["gain_R_over_K"].mean(), width=0.35, color=COL[m], label=lab if i == 0 else None)
        ax.scatter(np.full(len(d), xx), d["gain_R_over_K"], s=6, color="k", zorder=3)
        if ds == "terminators":
            for _, r in d.iterrows():
                ax.annotate(r["assay"], (xx, r["gain_R_over_K"]), xytext=(4, 0), textcoords="offset points",
                            fontsize=5.5, va="center")
        ax.text(xx, d["gain_R_over_K"].max() + 0.008, f"{100 * d['var_explained_by_kmers'].mean():.0f}%",
                ha="center", fontsize=6)
ax.set_xticks(range(len(dsets)))
ax.set_xticklabels(dsets)
ax.set_ylabel("R2 added on top of 6-mers\n(6-mers + residual minus 6-mers)")
ax.legend(frameon=False, fontsize=7)
ax.set_ylim(0, 0.22)
fig.tight_layout()
save(fig, "fig2_composition")

# ---------------- Fig 3 ----------------
v, z = rd("exp3_variants.tsv"), rd("exp3_zeroshot.tsv")
c, c100, ftv = rd("exp3_cnn.tsv"), rd("exp3_cnn_sub100k.tsv"), rd("finetune_ntv2.tsv")
ftv = ftv[(ftv["dataset"] == "variants")].pivot_table(index="assay", columns="metric", values="value")


def init_range(t):
    i = t[t["model"].str.startswith("cnn init")]
    return i.groupby("condition")["delta_spearman"].agg(["min", "max"])


M3 = [("NT-v2 zero-shot", z[z["model"] == "ntv2_100m zero-shot"].set_index("condition"), "#cccccc", None),
      ("AgroNT zero-shot", z[z["model"] == "agront_1b zero-shot"].set_index("condition"), "#969696", None),
      ("NT-v2 frozen", v[v["model"] == "ntv2_100m"].set_index("condition"), COL["ntv2_100m"], None),
      ("AgroNT frozen", v[v["model"] == "agront_1b"].set_index("condition"), COL["agront_1b"], None),
      ("6-mer ridge", v[v["model"] == "6mer"].set_index("condition"), COL["6mer"], None),
      ("CNN, 100k rows (ensemble of 3)", c100[c100["model"] == "cnn (mean of 3)"].set_index("condition"), "#a6761d",
       init_range(c100)),
      ("NT-v2 fine-tuned, 100k rows (1 run)", ftv, COL["ft"], None),
      ("CNN, all rows (ensemble of 3)", c[c["model"] == "cnn (mean of 3)"].set_index("condition"), COL["cnn"],
       init_range(c))]
ceil = rd("delta_noise_ceiling.tsv")
ceil = ceil[ceil["min_bc"] == 1].set_index("condition")
fig, ax = plt.subplots(figsize=(7.2, 3.1))
w = 0.8 / len(M3)
for k, (lab, d, col, rng) in enumerate(M3):
    xs = np.arange(5) - 0.4 + w * (k + 0.5)
    ys = np.array([d.loc[cnd, "delta_spearman"] for cnd in COND])
    ax.bar(xs, ys, width=w, color=col, label=lab)
    if rng is not None:
        ax.vlines(xs, [rng.loc[cnd, "min"] for cnd in COND], [rng.loc[cnd, "max"] for cnd in COND],
                  color="k", lw=0.7)
for i, cnd in enumerate(COND):
    ax.plot([i - 0.42, i + 0.42], [ceil.loc[cnd, "ceiling_delta_spearman"]] * 2, color="k", lw=1.2)
    ax.plot([i - 0.42, i + 0.42], [ceil.loc[cnd, "rep_delta_spearman"]] * 2, color="k", lw=0.8, ls="--")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels(COND)
ax.set_ylabel("Spearman, predicted vs measured\neffect of a TFBS shuffle")
ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.24))
ax.text(4.45, ceil.loc["maize", "ceiling_delta_spearman"], "noise ceiling", fontsize=6, va="center")
ax.text(4.45, ceil.loc["maize", "rep_delta_spearman"], "one replicate\nvs another", fontsize=6, va="center")
ax.set_xlim(-0.5, 5.2)
fig.tight_layout()
save(fig, "fig3_variants")

# ---------------- Fig 4 ----------------
pts = []
for lab, d, col, _ in M3:
    if "wt_r2" not in d.columns:
        continue
    for cnd in COND:
        pts.append(dict(model=lab, condition=cnd, color=col,
                        wt=100 * np.sqrt(d.loc[cnd, "wt_r2"]) / np.sqrt(ceil.loc[cnd, "ceiling_wt_r2"]),
                        delta=100 * d.loc[cnd, "delta_pearson"] / ceil.loc[cnd, "ceiling_delta_pearson"]))
pts = pd.DataFrame(pts)
fig, ax = plt.subplots(figsize=(3.6, 3.4))
for lab, g in pts.groupby("model", sort=False):
    col = g["color"].iloc[0]
    nm = g[g["condition"] != "maize"]
    mz = g[g["condition"] == "maize"]
    ax.scatter(nm["wt"], nm["delta"], s=14, color=col, label=lab)
    ax.scatter(mz["wt"], mz["delta"], s=14, facecolors="none", edgecolors=col, lw=1)
ax.scatter([], [], s=14, facecolors="none", edgecolors="k", label="open circle = maize")
ax.plot([0, 100], [0, 100], color="k", lw=0.6, ls=":")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("WT strength: % of reachable Pearson r")
ax.set_ylabel("mutation effect: % of reachable Pearson r")
ax.legend(frameon=False, fontsize=5.5, loc="upper left")
fig.tight_layout()
save(fig, "fig4_wt_vs_delta")
pts.drop(columns="color").round(1).to_csv(R / "fig4_points.tsv", sep="\t", index=False)
print("\nFIGURES DONE", flush=True)
