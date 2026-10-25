"""Experiment 3 with the CNN: trained on the FORCED split (t40, seed 1) exactly like exp3_variants.py,
so no WT or homolog of a WT is seen in training. 3 random inits, predictions averaged
(one bad run shouldn't decide the result; single-init numbers are printed too)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

ROOT = Path.home() / "plantcre"
P, S = ROOT / "data/processed", ROOT / "data/splits"
COND = ["light", "dark", "warm", "cold", "maize"]
STRONG = 0.5
SUB = int(os.environ.get("SUB", "0"))  # >0: train on the same subsample as finetune_ntv2.py
TAG = f"_sub{SUB // 1000}k" if SUB else ""
dev = "cuda"
torch.backends.cudnn.benchmark = True


def onehot(seqs):
    lut = np.zeros(256, dtype=np.int64)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i
    n, L = len(seqs), len(seqs[0])
    idx = lut[np.frombuffer("".join(seqs).encode(), dtype=np.uint8)].reshape(n, L)
    X = np.zeros((n, 4, L), dtype=np.float32)
    X[np.arange(n)[:, None], idx, np.arange(L)[None, :]] = 1.0
    return torch.from_numpy(X)


class CNN(nn.Module):
    def __init__(self, n_out, L):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 128, 13, padding=6), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 13, padding=6), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 9, padding=4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * ((L // 2) // 2), 256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, n_out))

    def forward(self, x):
        return self.head(self.conv(x))


enh = pd.read_parquet(P / "enhancers.parquet")
sp = enh[["id"]].merge(pd.read_parquet(S / "enhancers_t40_s1.parquet")[["id", "split"]], on="id", how="left")["split"].to_numpy()
v = pd.read_parquet(P / "variant_pairs.parquet").reset_index(drop=True)
wt_base = set(v["base_sequence"].str.replace(r"_(fwd|rev)$", "", regex=True))
assert enh.loc[sp == "train", "base_id"].isin(wt_base).sum() == 0, "a variant WT is in train!"

X = onehot(enh["sequence"].tolist())
L = X.shape[2]
Y = np.stack([enh[f"strength_{c}"].to_numpy(np.float32) for c in COND], axis=1)
Xw, Xm = onehot(v["seq_wt"].tolist()), onehot(v["seq_mut"].tolist())


def predict(model, Xs):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(Xs), 2048):
            out.append(model(Xs[b:b + 2048].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def train(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    has = np.isfinite(Y).any(axis=1)
    tr = np.where((sp == "train") & has)[0]
    va = np.where((sp == "val") & has)[0]
    if SUB:  # same rng and order as finetune_ntv2.py -> exactly the same rows
        r0 = np.random.default_rng(0)
        tr = np.sort(r0.choice(tr, SUB, replace=False))
        va = np.sort(r0.choice(va, 10_000, replace=False))
    mu, sd = np.nanmean(Y[tr], 0), np.nanstd(Y[tr], 0)
    yn = ((Y - mu) / sd).astype(np.float32)
    Yt = torch.from_numpy(yn)
    model = CNN(len(COND), L).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, bad, state = np.inf, 0, None
    for ep in range(1, 41):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), 256):
            j = torch.from_numpy(perm[b:b + 256])
            xb, yb = X[j].to(dev, non_blocking=True), Yt[j].to(dev, non_blocking=True)
            m = ~torch.isnan(yb)
            loss = ((model(xb) - torch.nan_to_num(yb)) ** 2)[m].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = np.concatenate([predict(model, X[torch.from_numpy(va[b:b + 2048])]) for b in range(0, len(va), 2048)])
        vl = float(((pv - np.nan_to_num(yn[va])) ** 2)[~np.isnan(yn[va])].mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(state)
    return model, mu, sd, ep


def score(pw, pm, tag):
    res = []
    for k, c in enumerate(COND):
        d = v[f"delta_{c}"].to_numpy(float)
        wt = v[f"strength_{c}_wt"].to_numpy(float)
        ok, okw = np.isfinite(d), np.isfinite(wt)
        strong = ok & (np.abs(d) >= STRONG)
        pd_ = pm[:, k] - pw[:, k]
        res.append(dict(model=tag, condition=c, n_pairs=int(ok.sum()),
                        wt_r2=pearsonr(pw[okw, k], wt[okw])[0] ** 2,
                        delta_spearman=spearmanr(pd_[ok], d[ok])[0], delta_pearson=pearsonr(pd_[ok], d[ok])[0],
                        n_strong=int(strong.sum()),
                        sign_acc_strong=float((np.sign(pd_[strong]) == np.sign(d[strong])).mean())))
    return res


rows, PW, PM = [], [], []
for seed in [1, 2, 3]:
    model, mu, sd, ep = train(seed)
    pw, pm = predict(model, Xw) * sd + mu, predict(model, Xm) * sd + mu
    PW.append(pw)
    PM.append(pm)
    r = score(pw, pm, f"cnn init {seed}")
    rows += r
    print(f"cnn init {seed} ({ep} epochs) | delta rho: " + " ".join(f"{x['condition']}={x['delta_spearman']:+.3f}" for x in r), flush=True)
rows += score(np.mean(PW, 0), np.mean(PM, 0), "cnn (mean of 3)")

out = pd.DataFrame(rows)
out.to_csv(ROOT / f"results/exp3_cnn{TAG}.tsv", sep="\t", index=False)
for col in ["delta_spearman", "sign_acc_strong", "wt_r2"]:
    print(f"\n######## CNN {col}")
    print(out.pivot(index="model", columns="condition", values=col).round(3)[COND].to_string())
print("\nEXP3 CNN DONE", flush=True)
