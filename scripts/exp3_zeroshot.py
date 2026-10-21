"""Experiment 3, zero-shot: variant scores straight from the masked language models (no training).
For each WT/mutant pair, the tokens that differ are masked in the WT sequence and
score = sum over those positions of [log p(mutant token) - log p(WT token)].
Negative = the model finds the mutant less likely. Compared with the measured delta.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path.home() / "plantcre"
COND = ["light", "dark", "warm", "cold", "maize"]
STRONG = 0.5
MODELS = {
    "ntv2_100m": ("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "f34324c6fde36a4f635f0f1f06cac5d25acd6798"),
    "agront_1b": ("InstaDeepAI/agro-nucleotide-transformer-1b", "b0e1ea1f53a2bf5bb29f8eab7a7e553bf06c1ab1"),
}


def local_path(name, sha):
    for root in [Path("/data3/pradip/hf_cache/hub"), Path.home() / ".cache/huggingface/hub"]:
        p = root / ("models--" + name.replace("/", "--")) / "snapshots" / sha
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"no local snapshot for {name}")


v = pd.read_parquet(ROOT / "data/processed/variant_pairs.parquet").reset_index(drop=True)
rows = []
for tag, (name, sha) in MODELS.items():
    path = local_path(name, sha)
    print(f"[{tag}] loading {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True).cuda().eval()
    score = np.zeros(len(v))
    ntok = np.zeros(len(v), dtype=int)
    for i, (w, m) in enumerate(zip(v["seq_wt"], v["seq_mut"])):
        tw = tok(w, return_tensors="pt")["input_ids"][0]
        tm = tok(m, return_tensors="pt")["input_ids"][0]
        assert len(tw) == len(tm), "WT and mutant tokenised to different lengths"
        diff = (tw != tm).nonzero().flatten()
        x = tw.clone()
        x[diff] = tok.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=x[None].cuda(), attention_mask=torch.ones_like(x)[None].cuda()).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1).cpu()
        score[i] = (lp[diff, tm[diff]] - lp[diff, tw[diff]]).sum().item()
        ntok[i] = len(diff)
    print(f"[{tag}] tokens changed per pair: median {np.median(ntok):.0f}, max {ntok.max()}", flush=True)
    v[f"zs_{tag}"] = score
    for c in COND:
        d = v[f"delta_{c}"].to_numpy(float)
        ok = np.isfinite(d)
        strong = ok & (np.abs(d) >= STRONG)
        rows.append(dict(model=f"{tag} zero-shot", condition=c, n_pairs=int(ok.sum()),
                         delta_spearman=spearmanr(score[ok], d[ok])[0], delta_pearson=pearsonr(score[ok], d[ok])[0],
                         n_strong=int(strong.sum()),
                         sign_acc_strong=float((np.sign(score[strong]) == np.sign(d[strong])).mean())))
    del model
    torch.cuda.empty_cache()

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/exp3_zeroshot.tsv", sep="\t", index=False)
v[["base_sequence", "tf_family"] + [f"zs_{t}" for t in MODELS]].to_parquet(ROOT / "results/exp3_zeroshot_scores.parquet", index=False)
print("\n######## zero-shot: delta Spearman")
print(out.pivot(index="model", columns="condition", values="delta_spearman").round(3)[COND].to_string())
print("\n######## zero-shot: sign accuracy on clear effects")
print(out.pivot(index="model", columns="condition", values="sign_acc_strong").round(2)[COND].to_string())
print("\nZEROSHOT DONE", flush=True)
