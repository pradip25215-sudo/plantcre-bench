"""Write FASTA files for hashFrag, plus a small pilot set to pick the homology threshold.

Pilot = 2,000 random sequences from each dataset (real) + a dinucleotide-shuffled copy of each.
Whatever score the shuffled ones reach is just chance, so the threshold goes above that.

Headers are simple numeric ids (P0000001 etc.) because some original ids have brackets in them,
and BLAST can get fussy about that. The mapping is saved next to each FASTA.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "plantcre"
P = ROOT / "data/processed"
OUT = ROOT / "data/homology"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)


def write_fasta(headers, seqs, path):
    with open(path, "w") as fh:
        for h, s in zip(headers, seqs):
            fh.write(f">{h}\n{s}\n")
    print(f"wrote {len(headers):,} sequences -> {path.name}")


def dinuc_shuffle(seq, rng):
    """Shuffle a sequence but keep every dinucleotide count exactly the same.

    Classic trick: treat the sequence as a walk through a graph (nodes = letters, edges = pairs),
    pick a random 'last exit' edge for each node so they form a tree pointing to the final letter,
    shuffle the rest, then walk again. Guarantees a valid walk using all the edges.
    """
    first, last = seq[0], seq[-1]
    edges = {}
    for a, b in zip(seq[:-1], seq[1:]):
        edges.setdefault(a, []).append(b)
    while True:
        pick = {v: int(rng.integers(len(e))) for v, e in edges.items() if v != last}
        ok = True
        for v in pick:  # following last-exit edges from any node must reach `last` with no loop
            seen, u = set(), v
            while u != last:
                if u in seen or u not in pick:
                    ok = False
                    break
                seen.add(u)
                u = edges[u][pick[u]]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, e in edges.items():
        e = list(e)
        if v != last:
            keep = e.pop(pick[v])
            rng.shuffle(e)
            e.append(keep)
        else:
            rng.shuffle(e)
        order[v] = e
    ptr = {v: 0 for v in order}
    out, cur = [first], first
    for _ in range(len(seq) - 1):
        nxt = order[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def dinucs(s):
    return Counter(a + b for a, b in zip(s[:-1], s[1:]))


# ---- load ----
prom = pd.read_parquet(P / "promoters.parquet", columns=["id", "sequence"])
term = pd.read_parquet(P / "terminators.parquet", columns=["id", "sequence", "is_synthetic_control"])
enh = pd.read_parquet(P / "enhancers.parquet", columns=["id", "base_id", "orientation", "sequence"])

# enhancers: one sequence per base_id (fwd if we have it, else rev). hashFrag adds the
# reverse complement itself, so orientation is covered either way.
enh_u = (enh.assign(o=(enh["orientation"] != "fwd").astype(int))
            .sort_values(["base_id", "o"]).drop_duplicates("base_id"))
print(f"enhancers: {len(enh):,} rows -> {len(enh_u):,} unique base sequences")

# ---- full FASTAs + id maps ----
for name, df, idcol, prefix in [("promoters", prom, "id", "P"),
                                ("terminators", term, "id", "T"),
                                ("enhancers", enh_u, "base_id", "E")]:
    df = df.reset_index(drop=True)
    heads = [f"{prefix}{i:07d}" for i in range(len(df))]
    write_fasta(heads, df["sequence"], OUT / f"{name}_all.fa")
    pd.DataFrame({"header": heads, "orig_id": df[idcol].values}).to_parquet(OUT / f"{name}_idmap.parquet", index=False)

# ---- pilot: 2,000 natural sequences per dataset ----
pil = pd.concat([
    prom.sample(2000, random_state=1).assign(src="promoter"),
    term[~term["is_synthetic_control"]].sample(2000, random_state=1).assign(src="terminator"),
    enh_u[enh_u["base_id"].str.match(r"^(At|Zm|Sb|Sl)-")].sample(2000, random_state=1).assign(src="enhancer"),
], ignore_index=True)

shuf = [dinuc_shuffle(s, rng) for s in pil["sequence"]]
bad = sum(dinucs(a) != dinucs(b) or a[0] != b[0] or a[-1] != b[-1] for a, b in zip(pil["sequence"], shuf))
same = sum(a == b for a, b in zip(pil["sequence"], shuf))
print(f"shuffle check: {bad} sequences with changed dinucleotide counts (must be 0), "
      f"{same} identical to original")
assert bad == 0

real_heads = [f"R{i:05d}" for i in range(len(pil))]
shuf_heads = [f"S{i:05d}" for i in range(len(pil))]
write_fasta(real_heads, pil["sequence"], OUT / "pilot_real.fa")
write_fasta(shuf_heads, shuf, OUT / "pilot_shuf.fa")
pd.DataFrame({"real": real_heads, "shuf": shuf_heads, "src": pil["src"]}).to_parquet(OUT / "pilot_map.parquet", index=False)
