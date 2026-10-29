# Figure captions (draft, 2026-10-29)

Figure 1. Homology between train and test in the published plant MPRA splits, and what it does to model
performance. (a) Share of each published test set with at least one train sequence above a local alignment
score threshold (hashFrag-style BLAST search, scores corrected as in hashFrag). Chance level from
dinucleotide-shuffled sequences is 34 (maximum over 6,000 shuffled sequences; Figure S1). (b) Test R2
(mean over assays) on random splits and on homology-aware splits at threshold 40 and 60, where no
homologous pair crosses train and test. Each dot is one split seed; bars are medians. Frozen models:
mean-pooled embeddings from the layer chosen on validation (NT-v2-100M layer 10, AgroNT-1B layer 12) with
ridge regression. CNN: trained from scratch, 5 seeds for promoters and terminators, 3 for enhancers.
NT-v2 fine-tuned: all weights, 3 seeds for terminators, 1 for promoters. Enhancer splits here do not force
the variant-pair sequences into test.

Figure 2. How much of what the embeddings know is 6-mer composition. Bars: test R2 of ridge on 6-mer
counts plus the embedding residual (the part 6-mers cannot predict), minus test R2 of 6-mers alone,
averaged over assays; dots are single assays. Numbers above bars: share of embedding variance explained by
6-mer counts on held-out training sequences. Published splits; enhancers use forward copies only and a
100k training subsample.

Figure 3. Ranking the measured effect of shuffling one TF binding site (741 WT / shuffle pairs, 16 TF
families, 5 conditions). No WT sequence or homolog of one is in any training set (forced homology split at
threshold 40). Solid line: noise ceiling, the highest Spearman reachable against the replicate-averaged
effect (Spearman-Brown from replicate-level data). Dashed line: agreement between two single replicates.
Zero-shot: masked-LM score of the mutant tokens. CNN bars are ensembles of 3 initialisations; whiskers show
the range of the single initialisations. "100k rows": CNN and fine-tuned NT-v2 trained on the same 100k-row
subsample.

Figure 4. Every model gets closer to the ceiling on WT strength than on the effect of a mutation. Each
point is one model in one condition: x, Pearson r for WT strength as % of its noise ceiling; y, the same
for the WT-to-mutant change. Open circles: maize protoplasts, the noisiest and hardest condition. Dotted
line: y = x.

Figure S1. Where the homology threshold comes from. (a) Best alignment score of each sequence to any
other sequence in a pilot of 2,000 natural sequences per dataset (orange outline) and one
dinucleotide-shuffled copy of each (grey), on a log count axis. Shuffling keeps every dinucleotide count
and the first and last base, so the grey distribution is what sequence composition alone produces.
Sequences with no BLAST hit are counted as 0. Vertical lines: the thresholds used in the paper.
(b) Share of natural sequences in the pilot with at least one homolog at each threshold, per dataset.
