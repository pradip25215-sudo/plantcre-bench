"""Fine-tune NT-v2-100M on terminators (tobacco + maize, multi-task) on the Experiment 1 splits:
random, homology_t40, homology_t60, for one seed. Recipe in ft_common.py.
usage: python finetune_ntv2_terminators.py [seed]   (default 1)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import encode, predict, r2, train  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ROOT = Path.home() / "plantcre"
P, S = ROOT / "data/processed", ROOT / "data/splits"
TAG = "" if SEED == 1 else f"_s{SEED}"

df = pd.read_parquet(P / "terminators.parquet")
ids, am = encode(df["sequence"])
Y = np.stack([df["strength_tobacco"].to_numpy(np.float32), df["strength_maize"].to_numpy(np.float32)], axis=1)
n = len(df)
perm = np.random.default_rng(1000 + SEED).permutation(n)
rand = np.empty(n, dtype=object)
rand[perm[:int(0.8 * n)]], rand[perm[int(0.8 * n):int(0.9 * n)]], rand[perm[int(0.9 * n):]] = "train", "val", "test"
splits = {"random": rand}
for t in [40, 60]:
    sp = pd.read_parquet(S / f"terminators_t{t}_s{SEED}.parquet")[["id", "split"]]
    splits[f"homology_t{t}"] = df[["id"]].merge(sp, on="id", how="left")["split"].to_numpy()

has = np.isfinite(Y).any(axis=1)
rows = []
print(f"######## fine-tuning NT-v2-100M: terminators, seed {SEED}", flush=True)
for kind, sp in splits.items():
    tr, va, te = [np.where((sp == s) & has)[0] for s in ("train", "val", "test")]
    model, mu, sd = train(ids, am, Y, tr, va, f"terminators {kind} s{SEED}", seed=SEED)
    p = predict(model, ids, am, te) * sd + mu
    for k, a in enumerate(["tobacco", "maize"]):
        v = r2(p[:, k], Y[te, k])
        rows.append(dict(dataset="terminators", split=kind, seed=SEED, assay=a, metric="test_r2", value=v))
        print(f"terminators {kind:13s} seed {SEED} {a:7s} test R2 = {v:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / f"results/finetune_ntv2_terminators{TAG}.tsv", sep="\t", index=False)
print(f"\nFINETUNE TERMINATORS DONE seed {SEED}", flush=True)
