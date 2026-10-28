# Prior art check (proposal section 8), 2026-10-28

Searches run on 2026-10-28: plant MPRA + DNA language model + homology/leakage; plantGREP + foundation model +
variant effect; AgroNT / PlantCAD2 + promoter strength + k-mer baseline / sequence-disjoint; hashFrag + plant;
plant STARR-seq + variant effect + language model; GENEB plant coverage.

Verdict: no paper found that combines (1) homology-controlled evaluation, (2) a composition (k-mer) baseline /
decomposition and (3) scoring of measured single-motif mutation effects for DNA foundation models on plant
MPRA data. Claim to use: "the first evaluation of DNA foundation models on plant MPRA data that controls for
homology, separates composition, and scores measured single-motif mutation effects."
Do NOT claim "first benchmark of foundation models on plant promoter strength" (AgroNT, BOTANIC-1 did that).

## Closest work
- BOTANIC-1 (Barozet et al., Living Models, bioRxiv 10.64898/2026.09.04.749355, Sep 2026).
  Plant gLM family (BiMamba2, 318M-3.2B). Fine-tunes on PGB promoter and terminator strength (Jores / Gorjifard
  STARR-seq), compares with AgroNT and PlantCAD2-L; Botanic1-S Pearson r 0.861 terminators, 0.896 / 0.874
  promoters. Variant benchmark = 545 causal SNP loci, zero-shot LLR ranking (not MPRA effects).
  No homology control, no k-mer baseline, no plantGREP enhancers, no measured TFBS mutations.
  Uses PGB splits; PGB promoter test (7,154) = Jores leaf test = our split_leaf test, where 32.9% of test
  sequences have a train homolog at score >= 40. Main hook for our Introduction.
  They state that current plant VEP benchmarks rely on proxies such as allele frequency -> our Exp 3 fills that.
- BOTANIC-0 (Ogier du Terrail et al., bioRxiv 10.64898/2026.02.23.706817, Feb 2026). Earlier generation.

## Homology / leakage
- hashFrag (Rafi, Kiyota, de Boer lab; bioRxiv 10.1101/2025.01.22.634321, v2 2026). Human MPRA, ChromBPNet,
  Enformer. v2 reports a non-monotonic pattern: better at both homology extremes, worse at intermediate similarity.
  Compare with our homology bins (AgroNT leaf: 34-40 0.549, 40-60 0.518, >=100 0.556). No plant data.

## Benchmarks
- GENEB (Ledneva et al., ICML 2026, arXiv 2606.04525). 40 models, 100 classification tasks, frozen probes, MCC.
  Includes PGB classification tasks only; no strength regression, no homology control.
- Plant Mol Biol 2025 (10.1007/s11103-025-01604-7): k-mer tokenisation matters more than scale on PGB. Random splits.
- Applied Sciences 2026 (10.3390/app16041745): gLM benchmark on human and rice tasks. Classification.

## Variant effects in plants
- Brachypodium gLM paper (bioRxiv 10.64898/2026.02.27.708524, Feb 2026): PlantCaduceus embeddings -> expression,
  validated on 796 Brachypodium mutant lines. Plant variant-level, but expression of mutant lines, not MPRA.
- Deep-Plant (bioRxiv 10.64898/2026.04.06.716755, 2026): supervised chromatin model fine-tuned for enhancer
  activity. Check whether it uses plantGREP data.

## Still to check
- plantGREP preprint (Jores et al. 2026): do the authors report plantGREP CNN predictions vs measured effects
  for the TFBS-shuffle validation library? If yes, that is the reference number for Exp 3.
- Deep-Plant: which enhancer data, which splits.
- Re-run this search right before submitting to bioRxiv.
