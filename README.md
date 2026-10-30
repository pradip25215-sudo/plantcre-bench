# PlantCRE-Bench

Code for "Homology, composition and mutation effects: what DNA foundation models actually learn from plant MPRA data".

We re-evaluate NT-v2-100M and AgroNT-1B, a CNN trained from scratch and a 6-mer ridge on three plant MPRA datasets
(Jores et al. 2021 promoters, Gorjifard et al. 2024 terminators, plantGREP enhancers) with:

1. homology-aware splits built with hashFrag-style BLAST scores (scripts/blast_all.sh, scripts/make_splits*.py)
2. a decomposition of each embedding into its 6-mer part and a residual (scripts/exp2*.py)
3. 741 measured wild-type / TFBS-shuffle pairs scored against a replicate noise ceiling (scripts/exp3*.py)

Data are not in this repository. Download the three source datasets from the original papers, then run the
scripts in the order given in docs/decisions.md. Figures: python scripts/make_figures.py

Environment: conda env create -f environment.yml
