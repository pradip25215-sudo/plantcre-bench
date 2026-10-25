"""Shared pieces for fine-tuning NT-v2-100M (same recipe as finetune_ntv2.py).
Head: mean of the last hidden layer over sequence tokens (CLS and padding left out) -> dropout -> linear.
AdamW, lr 2e-5 backbone / 1e-3 head, batch 64, max 4 epochs, best epoch by val loss."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from transformers import AutoModelForMaskedLM, AutoTokenizer

NAME, SHA = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "f34324c6fde36a4f635f0f1f06cac5d25acd6798"
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
        m[:, 0] = 0
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
