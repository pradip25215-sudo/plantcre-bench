"""Get mean-pooled embeddings from a HuggingFace DNA model (NT family for now).

For each dataset this writes:
  embeddings/<tag>/<dataset>.npy          float16, shape (N, n_saved_layers, hidden)
  embeddings/<tag>/<dataset>.ids.parquet  row order, so we can match back to the data
  embeddings/<tag>/meta.json              model name, pinned revision, which layers, pooling
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path.home() / "plantcre"
P = ROOT / "data/processed"


def load_dataset(name):
    if name in ("promoters", "terminators", "enhancers"):
        return pd.read_parquet(P / f"{name}.parquet", columns=["id", "sequence"]).reset_index(drop=True)
    if name == "variants":
        v = pd.read_parquet(P / "variant_pairs.parquet",
                            columns=["base_sequence", "tf_family", "seq_wt", "seq_mut"])
        wt = pd.DataFrame({"id": v["base_sequence"] + "|WT", "sequence": v["seq_wt"]}).drop_duplicates("id")
        mu = pd.DataFrame({"id": v["base_sequence"] + "|" + v["tf_family"] + "|MUT", "sequence": v["seq_mut"]})
        return pd.concat([wt, mu], ignore_index=True)
    raise ValueError(f"don't know dataset {name}")


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
ap.add_argument("--tag", default="ntv2_100m")
ap.add_argument("--datasets", nargs="+", default=["promoters", "terminators", "variants", "enhancers"])
ap.add_argument("--every", type=int, default=2, help="save every k-th layer (last layer always saved)")
ap.add_argument("--batch", type=int, default=256)
args = ap.parse_args()

# pin the exact model version so the paper numbers can be reproduced later
def get_sha(name):
    """Pinned model version: ask HuggingFace (30 s limit), else read it from the local cache."""
    try:
        return HfApi().model_info(name, timeout=30).sha
    except Exception as e:
        print(f"HF API didn't answer ({type(e).__name__}), using local cache", flush=True)
        ref = Path.home() / ".cache/huggingface/hub" / ("models--" + name.replace("/", "--")) / "refs/main"
        return ref.read_text().strip()


print("loading...", flush=True)
sha = get_sha(args.model)
print(f"model={args.model} | revision={sha}", flush=True)
tok = AutoTokenizer.from_pretrained(args.model, revision=sha, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(args.model, revision=sha, trust_remote_code=True).cuda().eval()

n_layers = model.config.num_hidden_layers
layers = sorted(set(range(args.every, n_layers + 1, args.every)) | {n_layers})
hidden = model.config.hidden_size
out = ROOT / "embeddings" / args.tag
out.mkdir(parents=True, exist_ok=True)
json.dump({"model": args.model, "revision": sha, "layers_saved": layers, "n_layers": n_layers,
           "hidden": hidden, "pooling": "mean over tokens, CLS excluded, padding excluded",
           "dtype": "float16", "date": str(date.today())},
          open(out / "meta.json", "w"), indent=2)
print(f"saving layers {layers} (hidden={hidden})", flush=True)

for name in args.datasets:
    d = load_dataset(name)
    n = len(d)
    arr = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.float16,
                                    shape=(n, len(layers), hidden))
    t0 = time.time()
    for i in tqdm(range(0, n, args.batch), desc=name, mininterval=60):
        seqs = d["sequence"].iloc[i:i + args.batch].tolist()
        enc = tok(seqs, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0  # first token is CLS, we don't want it in the average
        m = mask.unsqueeze(-1).float()
        pooled = torch.stack([(hs[l] * m).sum(1) / m.sum(1) for l in layers], dim=1)
        arr[i:i + len(seqs)] = pooled.cpu().numpy().astype(np.float16)
    arr.flush()
    d[["id"]].to_parquet(out / f"{name}.ids.parquet", index=False)
    print(f"{name}: {n:,} sequences done in {(time.time() - t0) / 60:.1f} min", flush=True)

print("ALL DONE", flush=True)
