"""Figure S2 (study workflow) and Figure S3 (CNN architecture), plain black-and-white style.

Text is set in Liberation Serif (same metrics as Times New Roman) for the PNG and PDF; in the SVG the
font name is written as Times New Roman, so Adobe Illustrator uses the real font and keeps it editable.
Layer sizes in Figure S3 are copied from scripts/cnn_exp1.py (class CNN).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "mathtext.fontset": "stix",
                     "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300})
K = "#000000"
GREY = "#555555"


def box(ax, x, y, w, h, title, body, ts=7.0, bs=6.2):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=K, lw=0.6))
    ax.text(x + w / 2, y + h - 0.016, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=K)
    if body:
        ax.text(x + w / 2, y + h - 0.050, body, ha="center", va="top", fontsize=bs, color=K, linespacing=1.35)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.6, color=K,
                                 shrinkA=0, shrinkB=0))


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    svg = (OUT / f"{name}.svg").read_text()
    (OUT / f"{name}.svg").write_text(svg.replace("Liberation Serif", "Times New Roman"))
    print("saved", name)


# =============================================================== Figure S2: workflow
fig = plt.figure(figsize=(7.0, 3.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0.12, 1)
ax.axis("off")

W, GAP = 0.170, 0.031
X = [0.005 + i * (W + GAP) for i in range(5)]
for x, t in zip(X, ["(1) Data", "(2) Homology search", "(3) Splits", "(4) Models", "(5) Experiments"]):
    ax.text(x + W / 2, 0.970, t, ha="center", va="center", fontsize=8, fontweight="bold", color=K)

for y, h, t, b in [(0.770, 0.160, "Core promoters", "Jores et al. (2021)\n76,728 sequences\n170 bp, two assays"),
                   (0.580, 0.160, "Terminators", "Gorjifard et al. (2024)\n53,409 sequences\n170 bp, two assays"),
                   (0.390, 0.160, "Enhancers", "Jores et al. (2026)\n180,921 sequences\nfive conditions"),
                   (0.200, 0.160, "Variant pairs", "plantGREP validation set\n741 WT / shuffle pairs\n16 TF families")]:
    box(ax, X[0], y, W, h, t, b)

box(ax, X[1], 0.700, W, 0.230, "All-vs-all BLAST",
    "hashFrag settings\nword size 7, both strands\n\n$S = rm + px - g_o g - g_e(e-g)$")
box(ax, X[1], 0.465, W, 0.200, "Threshold",
    "shuffled sequences reach\nat most $S = 34$\n\n$\\tau = 40$  (also 34 and 60)")
box(ax, X[1], 0.275, W, 0.155, "Homology groups",
    "edge if $S_{ij} \\geq \\tau$\nconnected components")

box(ax, X[2], 0.800, W, 0.130, "Published split", "as released")
box(ax, X[2], 0.635, W, 0.130, "Random split", "80/10/10, five seeds")
box(ax, X[2], 0.420, W, 0.180, "Homology-aware split",
    "whole groups assigned\n80/10/10, five seeds\nno pair with $S_{ij} \\geq \\tau$\nacross train and test")
box(ax, X[2], 0.200, W, 0.185, "Forced split",
    "every group holding a\nvariant WT goes to test\n(Experiment 3 only)")

mdl = [("6-mer ridge", "4,096 log counts"),
       ("NT-v2-100M, frozen", "layer 10 + ridge"),
       ("AgroNT-1B, frozen", "layer 12 + ridge"),
       ("CNN", "trained from scratch"),
       ("NT-v2-100M, fine-tuned", "all weights"),
       ("Zero-shot score", "no training")]
top, hgt, gap = 0.930, 0.105, 0.0175
for k, (t, b) in enumerate(mdl):
    y = top - hgt - k * (hgt + gap)
    box(ax, X[3], y, W, hgt, t, b, ts=6.8, bs=6.2)

box(ax, X[4], 0.700, W, 0.230, "Experiment 1",
    "homology leakage\n\nR$^2$ on random vs\nhomology-aware splits\n(Fig. 1)")
box(ax, X[4], 0.465, W, 0.200, "Experiment 2",
    "composition\n\n$E = \\hat{E} + R$\n(Fig. 2)")
box(ax, X[4], 0.200, W, 0.230, "Experiment 3",
    "mutation effects\n\n$\\hat{\\Delta} = f(x_{mut}) - f(x_{wt})$\nnoise ceiling from replicates\n(Figs. 3 and 4)")

R = lambda i: X[i] + W  # noqa: E731
L = lambda i: X[i]      # noqa: E731
for y in (0.850, 0.660, 0.470):
    arrow(ax, (R(0), y), (L(1), 0.815))
arrow(ax, (X[1] + W / 2, 0.700), (X[1] + W / 2, 0.665))
arrow(ax, (X[1] + W / 2, 0.465), (X[1] + W / 2, 0.430))
arrow(ax, (R(1), 0.370), (L(2), 0.510))
arrow(ax, (R(1), 0.300), (L(2), 0.300))
arrow(ax, (R(0), 0.225), (L(2), 0.225))
for y in (0.865, 0.700, 0.510, 0.290):
    arrow(ax, (R(2), y), (L(3), y))
arrow(ax, (R(3), 0.815), (L(4), 0.815))
arrow(ax, (R(3), 0.565), (L(4), 0.565))
arrow(ax, (R(3), 0.315), (L(4), 0.315))

ax.text(0.5, 0.150, "All models are trained and scored on the same splits. R$^2$ is the squared Pearson "
        "correlation on held-out sequences; mutation effects are scored by Spearman $\\rho$.",
        ha="center", va="center", fontsize=6.4, color=K)
save(fig, "figS2_workflow")

# =============================================================== Figure S3: CNN
fig = plt.figure(figsize=(7.0, 3.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

yc, TOP, SHAPE, DESC = 0.600, 0.930, 0.300, 0.255
scale = 0.0030

seq = "ACGTTGCAATCG"
x0, cw, ch = 0.040, 0.0075, 0.030
for j, nt in enumerate("ACGT"):
    yy = yc + 2 * ch - (j + 1) * ch
    ax.text(x0 - 0.008, yy + ch / 2, nt, ha="right", va="center", fontsize=6.5, color=K)
    for i, s in enumerate(seq):
        ax.add_patch(Rectangle((x0 + i * cw, yy), cw, ch, fc=K if s == nt else "white", ec=GREY, lw=0.3))
xin = x0 + len(seq) * cw / 2


def label(cx, name, shape, desc):
    ax.text(cx, TOP, name, ha="center", va="top", fontsize=7, fontweight="bold", color=K)
    ax.text(cx, SHAPE, shape, ha="center", va="top", fontsize=7, color=K)
    ax.text(cx, DESC, desc, ha="center", va="top", fontsize=6.2, color=K, linespacing=1.35)


label(xin, "Input", "4 × 170", "one-hot DNA\n(first 12 bases\ndrawn)")


def conv(xc, length, name, shape, desc):
    h = length * scale
    for k in range(3):
        ax.add_patch(Rectangle((xc + k * 0.007, yc - h / 2 - k * 0.010), 0.032, h, fc="white", ec=K, lw=0.6))
    label(xc + 0.023, name, shape, desc)
    return xc + 0.046


def dense(xc, h, name, shape, desc):
    ax.add_patch(Rectangle((xc, yc - h / 2), 0.020, h, fc="white", ec=K, lw=0.6))
    label(xc + 0.010, name, shape, desc)
    return xc + 0.020


e1 = conv(0.190, 170, "Conv block 1", "128 × 170", "Conv1d, width 13\nBatchNorm, ReLU")
e2 = conv(0.335, 85, "Conv block 2", "128 × 85", "Conv1d, width 13\nBatchNorm, ReLU\nmax pool /2")
e3 = conv(0.480, 42, "Conv block 3", "128 × 42", "Conv1d, width 9\nBatchNorm, ReLU\nmax pool /2")
f1 = dense(0.625, 0.44, "Flatten", "5,376", "dropout 0.3")
f2 = dense(0.745, 0.24, "Dense", "256", "ReLU\ndropout 0.3")
f3 = dense(0.870, 0.08, "Output", "2 or 5", "one value\nper assay")

for p, q in ((x0 + len(seq) * cw + 0.012, 0.183), (e1 + 0.006, 0.329), (e2 + 0.006, 0.474),
             (e3 + 0.006, 0.619), (f1 + 0.008, 0.739), (f2 + 0.008, 0.864)):
    arrow(ax, (p, yc), (q, yc))

ax.text(0.5, 0.080,
        "Loss: $L = \\frac{1}{|\\Omega|}\\sum_{(i,a)\\in\\Omega}(\\hat{z}_{ia} - z_{ia})^2$ over measured "
        "(sequence, assay) pairs.  Adam, learning rate 10$^{-3}$, batch 256, early stopping (patience 5).",
        ha="center", va="center", fontsize=6.4, color=K)
save(fig, "figS3_cnn")
