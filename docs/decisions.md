# PlantCRE-Bench — design decisions and data findings

## 2026-10-13 — Day 1

### Data
- Promoters (Jores 2021): 76,728 unique 170-bp sequences (At, Zm, Sb); leaf + proto strength.
- Enhancers (plantGREP 2026): 337,342 rows (fwd + rev), 5 conditions, original train/val/test.
- Terminators (Gorjifard 2024): 53,409 rows; 2,199 synthetic controls; species "GC" = synthetic.
- Variant pairs: 741 WT / TFBS-shuffle pairs, 16 TF families, 2–14 bp changed (median 4).

### Findings
- Exact-duplicate leakage between original train and test = 0 in all three datasets.
  Any homology effect we find is therefore near-duplicate leakage, not copy-paste.
- Promoters: 12,776 sequences have a DIFFERENT split in leaf vs proto assays.
  Cross-assay experiments must use one shared split.
- Enhancers: fwd/rev copies always share a split (0 violations) — correct in original data.
- Enhancers: tomato upstream tiles (Solyc*-N and Solyc*-N.5) overlap by design;
  measure how many overlapping tiles straddle train/test.
- Variant effects are small and TF/assay-specific: ~60% of shuffles lower strength,
  median delta -0.06 to -0.15 log2. Strong families (tobacco light): TF_22, TF_15, TF_2, TF_42.
  TF_13 effect is maize-specific. Report results stratified by family.
- Assay reproducibility for WT absolute strength (validation vs ACR library):
  R2 = 0.84–0.92 across conditions. Delta reproducibility still to be computed from replicates.

### Decisions
- D1. Use one shared split for promoters across leaf and proto.
- D2. Exclude controls (35S, AB80, Cab-1, rbcS, random/GC terminators) from benchmark metrics.
- D3. All 741 WT sequences of the variant set, plus their homologs, are held out of
  training in our own splits. 540 of them were in the original training split.
- D4. Report variant results for strong-effect families and null families separately.

### To do next
- Delta noise ceiling from replicate data (eVal_reps.rds).
- hashFrag homology splits.
- Overlap check for tomato tiles.

## 2026-10-17 — Day 2

### Embeddings
- NT-v2-100M (InstaDeepAI/nucleotide-transformer-v2-100m-multi-species), revision
  f34324c6fde36a4f635f0f1f06cac5d25acd6798. Mean-pooled (CLS + padding excluded), layers 2,4,...,22, float16.
- Rule: first download of any model runs in the foreground; every background run uses HF_HUB_OFFLINE=1
  (the HF version check hung on this server without it).

### Homology (hashFrag 1.0.2, BLAST+ 2.17.0 from the hashfrag conda env)
- pip hashFrag 1.0.2 is missing messages.json; copied from the GitHub repo (src/hashFrag/).
- An older BLAST 2.15 in ~/ncbi-blast-2.15.0+ shadows conda's; scripts force the env's BLAST first in PATH.
- Pilot: 2,000 natural sequences per dataset + dinucleotide-shuffled copies; word 7, evalue 100.
  Shuffled best-hit scores: p99.9 = 31, max = 34. Real: p99 = 94, p99.9 = 166.
- D5. Main homology threshold = 40 (above the shuffled max, with margin for the much larger
  full database). Sensitivity analysis at 34 and 60.
- D6. Word size for full runs decided by the recall test (word 11 vs word 7 on the pilot).
- D6 (final). Full all-vs-all search: hashFrag's exact blastn flags (word 7, evalue 100, dust off,
  max_hsps 1) run directly, plus -min_raw_gapped_score 20 (hashFrag's blastn_module can't pass it).
  Pilot vs plain word 7: ~17x faster, output 58x smaller. Recall of homologous pairs: 0.993 at
  threshold 40 (2/303 missed, both truly separated), 0.988 at 34, 1.000 at 60.
  Main conclusions must hold at both 40 and 60 (60 = exact recall).
- HF cache on this server is /data3/pradip/hf_cache (HF_HOME). Models are loaded with --local-path.

## 2026-10-21 — Experiment 1, first results
- Published splits leak: 32-36% of test sequences have a train homolog at score >= 40 (22-28% at >= 60)
  in all three datasets. Saved per-sequence bins in data/splits/*_original_*_homology_bins.parquet.
- Frozen embeddings + ridge (6-mer, NT-v2, AgroNT) lose almost nothing on homology splits for promoters
  and terminators (differences ~0.01 R2, same size as seed noise). Q1 is not supported for linear probes.
  The real test is a high-capacity model -> CNN from scratch on the same splits (cnn_exp1.py).
- D7. Enhancer splits with variant-WT groups forced to test (D3) distort Experiment 1 (t40 test = 21%,
  train = 70%). Experiment 1 uses unforced splits (data/splits/enhancers_nf_*). Experiment 3 keeps the
  forced splits. Old results/exp1_enhancers.tsv (forced) is not used.
- Layer selection is done on validation only. AgroNT best = layer 12, NT-v2 best = layer 10 on every task.
