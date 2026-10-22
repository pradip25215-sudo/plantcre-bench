"""Fine-tune NT-v2-100M end to end (all weights + a small head).
Frozen probes might undersell the foundation models, so this checks what fine-tuning adds, on the same splits.

Runs:
  promoters: random s1, homology_t40 s1, homology_t60 s1 (multi-task leaf + proto, shared split)
  enhancers: Experiment 3 setting = forced t40 s1 split (no variant WT or homolog in train),
             train subsampled to 100k rows, val 10k, test 20k -> test R2 + variant effects
Head: mean of the last hidden layer over sequence tokens (CLS and padding left out) -> dropout -> linear.
AdamW, lr 2e-5 for the backbone and 1e-3 for the head, batch 64, max 4 epochs, best epoch by val loss.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path.home() / "plantcre"
P, S = ROOT / "data/processed", ROOT / "data/splits"
NAME, SHA = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "f34324c6fde36a4f635f0f1f06cac5d25acd6798"
COND = ["light", "dark", "warm", "cold", "maize"]
STRONG = 0.5
dev = "cuda"


def local_path():
    for root in [Path("/data3/pradip/hf_cache/hub"), Path.home() / ".cache/huggingface/hub"]:
        p = root / ("models--" + NAME.replace("/", "--")) / "snapshots" / SHA
        if p.exists():
            return str(p)
    raise FileNotFoundError("NT-v2 snapshot not found")


PATH = local_path()
tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)


def encode(seqs):
    enc = tok(list(seqs), return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


class Reg(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.esm = AutoModelForMaskedLM.from_pretrained(PATH, trust_remote_code=True).esm
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.esm.config.hidden_size, n_out)

    def forward(self, ids, am):
        hs = self.esm(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.clone()
        m[:, 0] = 0  # leave CLS out of the average
        m = m.unsqueeze(-1).float()
        return self.out(self.drop((hs * m).sum(1) / m.sum(1)))


def predict(model, ids, am, ix):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(ix), 512):
            j = torch.from_numpy(np.asarray(ix[b:b + 512]))
            out.append(model(ids[j].to(dev), am[j].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def train(ids, am, Y, tr, va, tag, seed=1, epochs=4, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = np.nanmean(Y[tr], 0), np.nanstd(Y[tr], 0)
    Yn = ((Y - mu) / sd).astype(np.float32)
    Yt = torch.from_numpy(Yn)
    model = Reg(Y.shape[1]).to(dev)
    opt = torch.optim.AdamW([{"params": model.esm.parameters(), "lr": 2e-5},
                             {"params": model.out.parameters(), "lr": 1e-3}], weight_decay=0.01)
    best, state = np.inf, None
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), bs):
            j = torch.from_numpy(perm[b:b + bs])
            yb = Yt[j].to(dev)
            m = ~torch.isnan(yb)
            loss = ((model(ids[j].to(dev), am[j].to(dev)) - torch.nan_to_num(yb)) ** 2)[m].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = predict(model, ids, am, va)
        vl = float(((pv - np.nan_to_num(Yn[va])) ** 2)[~np.isnan(Yn[va])].mean())
        print(f"  [{tag}] epoch {ep}: val loss {vl:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if vl < best:
            best = vl
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, mu, sd


def r2(p, y):
    ok = np.isfinite(y)
    return pearsonr(p[ok], y[ok])[0] ** 2


rows = []

# ---------- promoters ----------
df = pd.read_parquet(P / "promoters.parquet")
ids, am = encode(df["sequence"])
Y = np.stack([df["strength_leaf"].to_numpy(np.float32), df["strength_proto"].to_numpy(np.float32)], axis=1)
units = df["id"].to_numpy()
perm = np.random.default_rng(1001).permutation(len(units))
rand = np.empty(len(units), dtype=object)
n = len(units)
rand[perm[:int(0.8 * n)]], rand[perm[int(0.8 * n):int(0.9 * n)]], rand[perm[int(0.9 * n):]] = "train", "val", "test"
splits = {"random": rand}
for t in [40, 60]:
    sp = pd.read_parquet(S / f"promoters_t{t}_s1.parquet")[["id", "split"]]
    splits[f"homology_t{t}"] = df[["id"]].merge(sp, on="id", how="left")["split"].to_numpy()
has = np.isfinite(Y).any(axis=1)
print("######## fine-tuning NT-v2-100M: promoters", flush=True)
for kind, sp in splits.items():
    tr, va, te = [np.where((sp == s) & has)[0] for s in ("train", "val", "test")]
    model, mu, sd = train(ids, am, Y, tr, va, f"promoters {kind}")
    p = predict(model, ids, am, te) * sd + mu
    for k, a in enumerate(["leaf", "proto"]):
        v = r2(p[:, k], Y[te, k])
        rows.append(dict(dataset="promoters", split=kind, seed=1, assay=a, metric="test_r2", value=v))
        print(f"promoters {kind:13s} {a:6s} test R2 = {v:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / "results/finetune_ntv2.tsv", sep="\t", index=False)

# ---------- enhancers: Experiment 3 setting ----------
enh = pd.read_parquet(P / "enhancers.parquet")
sp = enh[["id"]].merge(pd.read_parquet(S / "enhancers_t40_s1.parquet")[["id", "split"]], on="id", how="left")["split"].to_numpy()
v = pd.read_parquet(P / "variant_pairs.parquet").reset_index(drop=True)
wt_base = set(v["base_sequence"].str.replace(r"_(fwd|rev)$", "", regex=True))
assert enh.loc[sp == "train", "base_id"].isin(wt_base).sum() == 0, "a variant WT is in train!"
Ye = np.stack([enh[f"strength_{c}"].to_numpy(np.float32) for c in COND], axis=1)
has = np.isfinite(Ye).any(axis=1)
rng = np.random.default_rng(0)
tr = np.sort(rng.choice(np.where((sp == "train") & has)[0], 100_000, replace=False))
va = np.sort(rng.choice(np.where((sp == "val") & has)[0], 10_000, replace=False))
te = np.sort(rng.choice(np.where((sp == "test") & has)[0], 20_000, replace=False))
print("\n######## fine-tuning NT-v2-100M: enhancers (Experiment 3 split)", flush=True)
eids, eam = encode(enh["sequence"])
model, mu, sd = train(eids, eam, Ye, tr, va, "enhancers forced_t40")
p = predict(model, eids, eam, te) * sd + mu
for k, c in enumerate(COND):
    val = r2(p[:, k], Ye[te, k])
    rows.append(dict(dataset="enhancers", split="forced_t40", seed=1, assay=c, metric="test_r2", value=val))
    print(f"enhancers test R2 {c:6s} = {val:.3f}", flush=True)

wids, wam = encode(v["seq_wt"])
mids, mam = encode(v["seq_mut"])
idx = np.arange(len(v))
pw = predict(model, wids, wam, idx) * sd + mu
pm = predict(model, mids, mam, idx) * sd + mu
for k, c in enumerate(COND):
    d = v[f"delta_{c}"].to_numpy(float)
    wt = v[f"strength_{c}_wt"].to_numpy(float)
    ok, okw = np.isfinite(d), np.isfinite(wt)
    strong = ok & (np.abs(d) >= STRONG)
    pdl = pm[:, k] - pw[:, k]
    res = dict(wt_r2=pearsonr(pw[okw, k], wt[okw])[0] ** 2,
               delta_spearman=spearmanr(pdl[ok], d[ok])[0], delta_pearson=pearsonr(pdl[ok], d[ok])[0],
               sign_acc_strong=float((np.sign(pdl[strong]) == np.sign(d[strong])).mean()))
    for mname, mval in res.items():
        rows.append(dict(dataset="variants", split="forced_t40", seed=1, assay=c, metric=mname, value=mval))
    print(f"variants {c:6s} | WT R2={res['wt_r2']:.3f} | delta rho={res['delta_spearman']:+.3f} "
          f"| sign acc={res['sign_acc_strong']:.2f}", flush=True)

pd.DataFrame(rows).to_csv(ROOT / "results/finetune_ntv2.tsv", sep="\t", index=False)
print("\nFINETUNE DONE", flush=True)
