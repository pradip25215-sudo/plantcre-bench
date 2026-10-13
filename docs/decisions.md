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
