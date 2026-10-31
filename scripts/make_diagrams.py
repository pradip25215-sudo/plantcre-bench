"""Figure S3: the CNN trained from scratch, plain black-and-white style.

Text is set in Liberation Serif (same metrics as Times New Roman) for the PNG and PDF; in the SVG the
font name is written as Times New Roman, so Adobe Illustrator uses the real font and keeps it editable.
Layer sizes are copied from scripts/cnn_exp1.py (class CNN). Figure S2 is drawn by make_s2_icons.py.
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
