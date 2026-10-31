"""Figure S2: study workflow drawn with plain black-and-white symbols.

Datasets are database cylinders, the BLAST step shows two aligned sequences, the threshold step a
histogram with a cut-off line, homology groups a small graph, splits are segmented bars, models have
small icons, and each experiment has a small plot. Text uses Liberation Serif (Times New Roman
metrics); the saved SVG names Times New Roman so Illustrator uses the real font.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "mathtext.fontset": "stix",
                     "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300})
K = "#000000"
LW = 0.6

fig = plt.figure(figsize=(7.2, 4.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0.095, 1)
ax.axis("off")


def txt(x, y, s, size=6.4, weight="normal", ha="center", va="top"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight, color=K, linespacing=1.35)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=LW, color=K,
                                 shrinkA=0, shrinkB=0))


def cylinder(cx, cy, w=0.055, h=0.050):
    """A database symbol."""
    ry = h * 0.17
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + ry), w, h - 2 * ry, fc="white", ec="none"))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2 + ry, cy + h / 2 - ry], color=K, lw=LW)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2 + ry, cy + h / 2 - ry], color=K, lw=LW)
    ax.add_patch(Ellipse((cx, cy - h / 2 + ry), w, 2 * ry, fc="white", ec=K, lw=LW))
    ax.add_patch(Ellipse((cx, cy + h / 2 - ry), w, 2 * ry, fc="white", ec=K, lw=LW))


def seq_pair(cx, cy):
    """Two aligned sequences with match ticks between them."""
    n, cw = 11, 0.0062
    x0 = cx - n * cw / 2
    for row, dy in ((0, 0.013), (1, -0.013)):
        for i in range(n):
            ax.add_patch(Rectangle((x0 + i * cw, cy + dy - 0.005), cw * 0.82, 0.010,
                                   fc=K if (i % 3 == row % 3) else "white", ec=K, lw=0.3))
    for i in range(n):
        if i % 3 == 0:
            ax.plot([x0 + i * cw + cw * 0.41] * 2, [cy - 0.004, cy + 0.004], color=K, lw=0.4)


def histogram(cx, cy):
    """A small histogram with a vertical cut-off line."""
    hs = [0.004, 0.009, 0.016, 0.022, 0.015, 0.008, 0.004, 0.003, 0.004, 0.005]
    n, bw = len(hs), 0.0055
    x0 = cx - n * bw / 2
    base = cy - 0.016
    for i, hh in enumerate(hs):
        ax.add_patch(Rectangle((x0 + i * bw, base), bw * 0.85, hh,
                               fc=K if i < 6 else "white", ec=K, lw=0.3))
    ax.plot([x0 - 0.004, x0 + n * bw], [base, base], color=K, lw=LW)
    xc = x0 + 6.2 * bw
    ax.plot([xc, xc], [base - 0.004, base + 0.030], color=K, lw=0.9, ls=(0, (2, 1.4)))
    ax.text(xc + 0.004, base + 0.030, "$\\tau$", fontsize=6.2, va="top", ha="left", color=K)


def graph_icon(cx, cy):
    """Two connected components."""
    r = 0.0052
    g1 = [(-0.030, 0.012), (-0.046, -0.008), (-0.014, -0.010)]
    g2 = [(0.026, 0.012), (0.044, -0.004), (0.016, -0.012), (0.040, -0.020)]
    for g in (g1, g2):
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                ax.plot([cx + g[i][0], cx + g[j][0]], [cy + g[i][1], cy + g[j][1]], color=K, lw=0.4)
        for px, py in g:
            ax.add_patch(Circle((cx + px, cy + py), r, fc="white", ec=K, lw=LW))


def split_bar(cx, cy, fr, w=0.132, h=0.015, labels=True):
    """A bar split into train / val / test, with the two small parts labelled underneath."""
    x0 = cx - w / 2
    for frac, lab, dy in zip(fr, ("train", "val", "test"), (0.0, 0.012, 0.030)):
        ax.add_patch(Rectangle((x0, cy - h / 2), w * frac, h, fc="white", ec=K, lw=LW))
        if labels:
            if frac > 0.3:
                ax.text(x0 + w * frac / 2, cy, lab, ha="center", va="center", fontsize=5.2, color=K)
            else:
                xm = x0 + w * frac / 2
                ax.plot([xm, xm], [cy - h / 2, cy - h / 2 - dy], color=K, lw=0.35)
                ax.text(xm, cy - h / 2 - dy - 0.002, lab, ha="center", va="top", fontsize=4.8, color=K)
        x0 += w * frac


def grid_icon(cx, cy):
    """A k-mer count grid."""
    n, c = 4, 0.0085
    x0, y0 = cx - n * c / 2, cy - n * c / 2
    fill = {(0, 1), (1, 3), (2, 0), (3, 2), (1, 1)}
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + i * c, y0 + j * c), c, c,
                                   fc=K if (i, j) in fill else "white", ec=K, lw=0.3))


def stack_icon(cx, cy, n=4):
    """Stacked transformer layers."""
    w, h, g = 0.046, 0.0085, 0.0045
    y0 = cy - (n * (h + g)) / 2
    for i in range(n):
        ax.add_patch(Rectangle((cx - w / 2, y0 + i * (h + g)), w, h, fc="white", ec=K, lw=LW))


def conv_icon(cx, cy):
    """Narrowing convolutional blocks."""
    for i, (dx, hh) in enumerate(((-0.020, 0.040), (0.000, 0.028), (0.020, 0.018))):
        ax.add_patch(Rectangle((cx + dx - 0.007, cy - hh / 2), 0.014, hh, fc="white", ec=K, lw=LW))


def scatter_icon(cx, cy, kind):
    """A tiny plot: bars for Exp. 1, split blocks for Exp. 2, scatter for Exp. 3."""
    w, h = 0.070, 0.038
    x0, y0 = cx - w / 2, cy - h / 2
    ax.plot([x0, x0], [y0, y0 + h], color=K, lw=LW)
    ax.plot([x0, x0 + w], [y0, y0], color=K, lw=LW)
    if kind == "bars":
        for i, hh in enumerate((0.030, 0.026, 0.024)):
            ax.add_patch(Rectangle((x0 + 0.010 + i * 0.020, y0), 0.012, hh,
                                   fc=["white", K, "white"][i], ec=K, lw=0.4))
    elif kind == "stack":
        ax.add_patch(Rectangle((x0 + 0.018, y0), 0.030, 0.016, fc=K, ec=K, lw=0.4))
        ax.add_patch(Rectangle((x0 + 0.018, y0 + 0.016), 0.030, 0.014, fc="white", ec=K, lw=0.4))
    else:
        rng = np.random.default_rng(3)
        px = x0 + 0.012 + rng.random(9) * (w - 0.022)
        py = y0 + 0.006 + (px - x0) * 0.32 + rng.random(9) * 0.008
        ax.plot(px, np.clip(py, y0 + 0.003, y0 + h - 0.003), "o", ms=1.6, mfc="white", mec=K, mew=0.4, ls="none")
        ax.plot([x0, x0 + w], [y0 + h - 0.004, y0 + h - 0.004], color=K, lw=0.7)


# ----------------------------------------------------------------- layout
W, GAP = 0.173, 0.034
X = [0.007 + i * (W + GAP) for i in range(5)]
HEAD = 0.975
for x, t in zip(X, ["(1) Data", "(2) Homology search", "(3) Splits", "(4) Models", "(5) Experiments"]):
    txt(x + W / 2, HEAD, t, size=8, weight="bold", va="center")
    ax.plot([x, x + W], [HEAD - 0.022, HEAD - 0.022], color=K, lw=0.8)


def panel(col, y, h, icon, title, body, icon_dy=0.030):
    x = X[col]
    ax.add_patch(Rectangle((x, y), W, h, fc="white", ec=K, lw=LW))
    icon(x + W / 2, y + h - icon_dy)
    txt(x + W / 2, y + h - icon_dy * 2 - 0.016, title, size=6.9, weight="bold")
    txt(x + W / 2, y + h - icon_dy * 2 - 0.042, body, size=6.1)


# column 1
data = [(0.760, "Core promoters", "Jores et al. (2021)\n76,728 sequences, 170 bp"),
        (0.585, "Terminators", "Gorjifard et al. (2024)\n53,409 sequences, 170 bp"),
        (0.410, "Enhancers", "Jores et al. (2026)\n180,921 sequences"),
        (0.215, "Variant pairs", "741 WT / shuffle pairs\n16 TF families")]
for y, t, b in data:
    panel(0, y, 0.150, cylinder, t, b)

# column 2
panel(1, 0.700, 0.215, seq_pair, "All-vs-all BLAST",
      "hashFrag settings, word size 7\n$S = rm + px - g_o g - g_e(e-g)$", icon_dy=0.034)
panel(1, 0.455, 0.215, histogram, "Threshold from chance",
      "shuffled sequences reach $S = 34$\n$\\tau = 40$  (also 34 and 60)", icon_dy=0.040)
panel(1, 0.258, 0.185, graph_icon, "Homology groups",
      "edge if $S_{ij} \\geq \\tau$;\nconnected components", icon_dy=0.034)

# column 3
panel(2, 0.780, 0.135, lambda cx, cy: split_bar(cx, cy, (0.80, 0.10, 0.10), labels=False),
      "Published split", "as released", icon_dy=0.026)
panel(2, 0.615, 0.135, lambda cx, cy: split_bar(cx, cy, (0.80, 0.10, 0.10)),
      "Random split", "80/10/10, five seeds", icon_dy=0.030)
panel(2, 0.395, 0.190, lambda cx, cy: split_bar(cx, cy, (0.80, 0.10, 0.10)),
      "Homology-aware split", "whole groups assigned;\nno pair with $S_{ij} \\geq \\tau$\nacross train and test",
      icon_dy=0.030)
panel(2, 0.200, 0.165, lambda cx, cy: split_bar(cx, cy, (0.74, 0.10, 0.16)),
      "Forced split", "groups holding a variant WT\ngo to test (Experiment 3)", icon_dy=0.030)

# column 4
models = [(0.818, grid_icon, "6-mer ridge", "4,096 log counts"),
          (0.688, stack_icon, "NT-v2-100M, frozen", "layer 10 + ridge"),
          (0.558, stack_icon, "AgroNT-1B, frozen", "layer 12 + ridge"),
          (0.428, conv_icon, "CNN", "trained from scratch"),
          (0.298, stack_icon, "NT-v2-100M, fine-tuned", "all weights"),
          (0.168, stack_icon, "Zero-shot score", "no training")]
for y, ic, t, b in models:
    panel(3, y, 0.112, ic, t, b, icon_dy=0.026)

# column 5
panel(4, 0.700, 0.215, lambda cx, cy: scatter_icon(cx, cy, "bars"), "Experiment 1",
      "homology leakage\nR$^2$ on random vs\nhomology-aware splits (Fig. 1)", icon_dy=0.036)
panel(4, 0.455, 0.215, lambda cx, cy: scatter_icon(cx, cy, "stack"), "Experiment 2",
      "composition\n$E = \\hat{E} + R$  (Fig. 2)", icon_dy=0.036)
panel(4, 0.200, 0.225, lambda cx, cy: scatter_icon(cx, cy, "scatter"), "Experiment 3",
      "mutation effects\n$\\hat{\\Delta} = f(x_{mut}) - f(x_{wt})$\nnoise ceiling (Figs. 3, 4)",
      icon_dy=0.036)

# ----------------------------------------------------------------- arrows
R = lambda i: X[i] + W  # noqa: E731
L = lambda i: X[i]      # noqa: E731
for y in (0.835, 0.660, 0.485):
    arrow((R(0), y), (L(1), 0.808))
arrow((X[1] + W / 2, 0.700), (X[1] + W / 2, 0.673))
arrow((X[1] + W / 2, 0.455), (X[1] + W / 2, 0.443))
arrow((R(1), 0.380), (L(2), 0.490))
arrow((R(1), 0.300), (L(2), 0.290))
arrow((R(0), 0.238), (L(2), 0.238))
for y in (0.848, 0.683, 0.490, 0.283):
    arrow((R(2), y), (L(3), y))
arrow((R(3), 0.808), (L(4), 0.808))
arrow((R(3), 0.560), (L(4), 0.563))
arrow((R(3), 0.224), (L(4), 0.313))

txt(0.5, 0.135, "All models are trained and scored on the same splits. R$^2$ is the squared Pearson "
    "correlation on held-out sequences; mutation effects are scored by Spearman $\\rho$.",
    size=6.3, va="center")

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"figS2_workflow.{ext}", bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.close(fig)
svg = (OUT / "figS2_workflow.svg").read_text()
(OUT / "figS2_workflow.svg").write_text(svg.replace("Liberation Serif", "Times New Roman"))
print("saved figS2_workflow")
