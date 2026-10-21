"""Supervised CNN trained from scratch on the same splits as Experiment 1.
This is the high-capacity model where homology leakage should hurt most, so it's the real test of Q1.
Architecture is a plain Jores/plantGREP-style CNN (3 conv blocks + dense), one-hot input, no RC augmentation
(promoters/terminators are directional). Early stopping on validation loss, patience 5, max 40 epochs.

Splits: original (published; 10% of train held out as val when there is no val), random s1-3,
homology t40 s1-3, homology t60 s1-3. Assays that share a split are trained as one multi-task model.
usage: python cnn_exp1.py <dataset> [split_tag]      split_tag = "" or "_nf"
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

ROOT = Path.home() / "plantcre"
P, S = ROOT / "data/processed", ROOT / "data/splits"
DATASET = sys.argv[1]
SPLITTAG = sys.argv[2] if len(sys.argv) > 2 else ""
SEEDS, THRS = [1, 2, 3], [40, 60]
BIN_NAMES = ["no hit", "<34 (chance)", "34-40", "40-60", "60-100", ">=100"]
ASSAYS = {
    "promoters": {"leaf": ("strength_leaf", "split_leaf"), "proto": ("strength_proto", "split_proto")},
    "terminators": {a: (f"strength_{a}", "split") for a in ["tobacco", "maize"]},
    "enhancers": {a: (f"strength_{a}", "split") for a in ["light", "dark", "warm", "cold", "maize"]},
}[DATASET]
NAMES = list(ASSAYS)
UNIT = "base_id" if DATASET == "enhancers" else "id"
dev = "cuda"
torch.backends.cudnn.benchmark = True

df = pd.read_parquet(P / f"{DATASET}.parquet")
N, L = len(df), len(df["sequence"].iloc[0])
lut = np.full(256, 0, dtype=np.int64)
for i, c in enumerate("ACGT"):
    lut[ord(c)] = i
idx = lut[np.frombuffer("".join(df["sequence"]).encode(), dtype=np.uint8)].reshape(N, L)
X = np.zeros((N, 4, L), dtype=np.float32)
X[np.arange(N)[:, None], idx, np.arange(L)[None, :]] = 1.0
Xt = torch.from_numpy(X)
Y = np.stack([df[ASSAYS[a][0]].to_numpy(np.float32) for a in NAMES], axis=1)
units = df[UNIT].to_numpy()
print(f"######## CNN Experiment 1: {DATASET}{SPLITTAG} | N={N:,} | assays={NAMES}", flush=True)


class CNN(nn.Module):
    def __init__(self, n_out, L):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 128, 13, padding=6), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 13, padding=6), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 9, padding=4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        flat = 128 * ((L // 2) // 2)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(flat, 256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, n_out))

    def forward(self, x):
        return self.head(self.conv(x))


def predict(model, ix):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(ix), 2048):
            j = torch.from_numpy(ix[b:b + 2048])
            out.append(model(Xt[j].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def masked_mse(pred, y):
    m = ~np.isnan(y)
    return float(((pred - np.nan_to_num(y)) ** 2)[m].mean())


def train_eval(sp, cols, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = Y[:, cols]
    has = np.isfinite(y).any(axis=1)
    tr = np.where((sp == "train") & has)[0]
    va = np.where((sp == "val") & has)[0]
    te = np.where((sp == "test") & has)[0]
    mu, sd = np.nanmean(y[tr], 0), np.nanstd(y[tr], 0)
    yn = ((y - mu) / sd).astype(np.float32)
    Yt = torch.from_numpy(yn)
    model = CNN(len(cols), L).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, bad, state, ep = np.inf, 0, None, 0
    for ep in range(1, 41):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), 256):
            j = torch.from_numpy(perm[b:b + 256])
            xb, yb = Xt[j].to(dev, non_blocking=True), Yt[j].to(dev, non_blocking=True)
            m = ~torch.isnan(yb)
            loss = ((model(xb) - torch.nan_to_num(yb)) ** 2)[m].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = masked_mse(predict(model, va), yn[va])
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(state)
    p = predict(model, te) * sd + mu
    return te, p, y[te], ep


def random_split(seed):
    uniq = pd.unique(units)
    n = len(uniq)
    perm = np.random.default_rng(1000 + seed).permutation(n)
    lab = np.empty(n, dtype=object)
    lab[perm[:int(0.8 * n)]] = "train"
    lab[perm[int(0.8 * n):int(0.9 * n)]] = "val"
    lab[perm[int(0.9 * n):]] = "test"
    return pd.Series(units).map(dict(zip(uniq, lab))).to_numpy()


def with_val(sp):
    """Original splits without a val part: hold out 10% of train units (fixed seed) for early stopping."""
    if (sp == "val").any():
        return sp
    sp = sp.copy()
    tr_units = pd.unique(units[sp == "train"])
    rng = np.random.default_rng(0)
    va_units = set(rng.choice(tr_units, int(0.1 * len(tr_units)), replace=False))
    sp[(sp == "train") & pd.Series(units).isin(va_units).to_numpy()] = "val"
    return sp


bins = {}
for _, scol in ASSAYS.values():
    if scol not in bins:
        b = pd.read_parquet(S / f"{DATASET}_original_{scol}_homology_bins.parquet")
        bins[scol] = df[[UNIT]].merge(b[[UNIT, "bin"]], on=UNIT, how="left")["bin"].to_numpy()

configs = [("original", 0, None)]
configs += [("random", s, random_split(s)) for s in SEEDS]
for t in THRS:
    for s in SEEDS:
        sp = pd.read_parquet(S / f"{DATASET}{SPLITTAG}_t{t}_s{s}.parquet")[["id", "split"]]
        configs.append((f"homology_t{t}", s, df[["id"]].merge(sp, on="id", how="left")["split"].to_numpy()))

rows = []
for kind, seed, shared in configs:
    t0 = time.time()
    if kind == "original":   # promoters have a different original split per assay
        groups = {}
        for a, (_, scol) in ASSAYS.items():
            groups.setdefault(scol, []).append(NAMES.index(a))
        jobs = [(with_val(df[scol].to_numpy()), cols, scol) for scol, cols in groups.items()]
    else:
        jobs = [(shared, list(range(len(NAMES))), None)]
    msg = []
    for sp, cols, scol in jobs:
        te, p, yt, ep = train_eval(sp, cols, seed if seed else 7)
        for k, c in enumerate(cols):
            ok = np.isfinite(yt[:, k])
            r2, rho = pearsonr(p[ok, k], yt[ok, k])[0] ** 2, spearmanr(p[ok, k], yt[ok, k])[0]
            rows.append(dict(dataset=DATASET, assay=NAMES[c], model="cnn", split=kind, seed=seed, bin="all",
                             n_test=int(ok.sum()), r2=r2, spearman=rho, epochs=ep))
            msg.append(f"{NAMES[c]}={r2:.3f}")
            if kind == "original":
                bt = bins[scol][te][ok]
                for bname in BIN_NAMES:
                    mk = bt == bname
                    if mk.sum() >= 100:
                        rb = pearsonr(p[ok, k][mk], yt[ok, k][mk])[0] ** 2
                        rows.append(dict(dataset=DATASET, assay=NAMES[c], model="cnn", split=kind, seed=0,
                                         bin=bname, n_test=int(mk.sum()), r2=rb, spearman=np.nan, epochs=ep))
    print(f"cnn {kind:13s} seed {seed} | " + " ".join(msg) + f" | {ep} epochs, {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / f"results/cnn_exp1_{DATASET}{SPLITTAG}.tsv", sep="\t", index=False)

out = pd.DataFrame(rows)
print("\n######## CNN mean R2 (sd), bin = all")
print(out[out["bin"] == "all"].groupby(["assay", "split"])["r2"].agg(["mean", "std"]).round(3).unstack("split").to_string())
print("\n######## CNN original split, R2 by homology bin")
ob = out[out["split"] == "original"].pivot_table(index="assay", columns="bin", values="r2").round(3)
print(ob.reindex(columns=["all"] + BIN_NAMES).to_string())
print(f"\nCNN EXP1 DONE {DATASET}{SPLITTAG}", flush=True)
