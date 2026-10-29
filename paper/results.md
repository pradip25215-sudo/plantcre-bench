# Results

## A third of each published test set has a homologous sequence in training

We searched each dataset against itself and recorded, for every test sequence in the published split, its
best alignment score to any training sequence. Dinucleotide-shuffled sequences reach at most 34 by chance
(Figure S1), so a score of 40 or above is homology rather than shared composition.

In all three datasets about a third of the published test set has such a homolog: 32.9% for promoters in
tobacco leaves, 34.4% in maize protoplasts, 32.3% for terminators and 36.1% for enhancers (Figure 1a).
About a quarter still has one at a score of 60. Some of these pairs are near-identical over the full
170 bp: 647 promoter, 751 terminator and 2,230 enhancer test sequences score 100 or more. The promoter
test set here is the one in the Plant Genomic Benchmark, on which AgroNT, PlantCAD2 and Botanic1 all report
their promoter-strength numbers.

Exact duplicates are not the cause. No test sequence in any of the three splits is a character-for-character
copy of a training sequence.

## Removing that homology costs models little

We rebuilt the splits so that no homologous pair crosses train and test, assigning whole homology groups to
train, validation or test, and repeated the evaluation with five seeds (Figure 1b). Against random splits of
the same sizes, frozen-embedding and 6-mer models lose at most 0.01 R2, which is the size of the
seed-to-seed spread, and in several cases they score slightly higher on the homology split than on the random
one. The published-split result sits inside the same range.

Models with more capacity lose a little more, and only on two of the three datasets. The CNN trained from
scratch drops by 0.032 R2 on terminators and 0.011 to 0.023 on enhancers, and fine-tuned NT-v2 drops by
0.013 on tobacco and 0.017 on maize terminators, with every one of six seed-by-assay comparisons in the
same direction. On promoters neither the CNN nor the fine-tuned model drops at all.

We also split the published test sets by how similar each sequence is to training. If models were reading
off memorised homologs, the most similar bin would be the easiest. It is not: for AgroNT on tobacco-leaf
promoters the bin scoring 100 or more reaches R2 0.556 against 0.531 for sequences at chance similarity, and
for NT-v2 that bin is the hardest of all (0.395). The hardest bin everywhere is the one with no BLAST hit at
all, which turns out to be a composition effect rather than a homology one: those sequences are GC-richer
(0.49 against 0.44 for promoters), have higher trinucleotide entropy, and are weaker on average.

So homology is pervasive in these benchmarks, but on this data it inflates reported performance by at most
a few hundredths of R2, and only for models trained end to end.

## About half of what the embeddings carry is 6-mer composition

Frozen embeddings are not obviously better than counting 6-mers. On the published splits, AgroNT beats a
6-mer ridge on promoters (0.530 against 0.483 in leaves) and on terminators (0.456 against 0.293 in tobacco),
but loses on enhancers (0.436 against 0.496 in light). NT-v2 is below the 6-mer baseline on promoters and
enhancers.

To separate composition from the rest, we regressed each embedding on 6-mer counts and split it into the part
6-mers predict and a residual. 6-mer counts explain 44% to 52% of embedding variance across models and datasets
(Figure 2). Adding the residual to 6-mer counts improves prediction by 0.14 to 0.20 R2 on terminators, 0.05 to
0.10 on promoters, and 0.006 to 0.024 on enhancers. On enhancers the composition part alone predicts better
than the full embedding (AgroNT 0.466 against 0.432 in light), so what the embedding adds there is noise as
far as a linear probe is concerned.

The terminator result is the clearest case of a foundation model carrying something beyond composition. We
have not established what that is. One possibility is position: 6-mer counts discard it, and terminator
strength depends on polyadenylation signals and cleavage position. A positional k-mer baseline would test this
and is not included here.

## Fine-tuning closes the gap with supervised models

Fine-tuning all of NT-v2-100M lifts promoter R2 from 0.44 and 0.36 as a frozen probe to 0.700 and 0.677 in
leaves and protoplasts, matching the published Jores CNN (0.714 and 0.670) and beating our own CNN
re-implementation (0.668 and 0.650). On terminators it reaches 0.693 and 0.618, above our CNN (0.620 and
0.540) but below the published DenseNet (0.762 and 0.672).

The same picture holds for variant effects, where a fine-tuned NT-v2 trained on 100,000 enhancer rows and a
CNN ensemble trained on the same rows are indistinguishable (Figure 3). Whether a foundation model helps
therefore depends on how it is used, not only on which model it is.

## No model ranks single-motif mutations well, and zero-shot scoring fails outright

We scored 741 wild-type and TFBS-shuffle pairs from the plantGREP validation library, with every wild-type
sequence and its homologs held out of training. Measured effects are small: shuffling lowers strength in 57%
to 64% of pairs with a median effect of -0.06 to -0.15 log2 units, and the effect is family- and
condition-specific. TF_22 shows a median effect of -0.96 in light and -0.07 in maize, while TF_13 shows +0.04
in light and -0.39 in maize.

Zero-shot masked-language-model scores carry no signal: Spearman -0.07 to 0.08 against measured effects, and
sign accuracy 0.43 to 0.54 on pairs with a clear effect, which is chance (Figure 3). Pretraining likelihood
does not know whether removing a binding site raises or lowers enhancer activity.

Trained models do better but not well. The best Spearman in each condition is 0.37 to 0.56, with 6-mer ridge,
the CNN and fine-tuned NT-v2 within 0.03 of one another and the frozen embeddings behind them. Against the
replicate-derived ceiling of 0.85 to 0.94, that is 42% to 60%. A single experimental replicate predicts another
replicate better than any model does in four of the five conditions: 0.72 against 0.52 in light. Only in the
warm condition, the noisiest, does the best model come close to one replicate (0.43 against 0.46).

Put on the same scale (Figure 4), every model reaches 68% to 84% of its ceiling on wild-type strength but only
31% to 63% on the effect of mutating one site, a gap of 20 to 35 points for every model. The ordering of models
is the same for both, so a model that predicts sequence strength well is also the better one at ranking
mutations, but the skill does not transfer in full. Maize protoplasts are hardest on both axes.
