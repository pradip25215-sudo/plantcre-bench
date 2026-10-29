# Discussion

We set out to test three things about DNA foundation models on plant MPRA data, and got a different answer
to each.

Homology between training and test sets is real and widespread in these benchmarks. A third of every
published test set has a homolog in training, and a sizeable minority of those pairs are near-identical
across the full 170 bp. We expected this to inflate the published numbers, and for a 6-mer ridge or a
frozen-embedding probe it does not: the difference between a random split and a homology-aware one is
within the seed-to-seed spread. Models trained end to end lose a little more, 0.01 to 0.03 R2 on
terminators and enhancers, and nothing on promoters. The direction matches what hashFrag reported for human
models, but the size is small, and our explanation is capacity. A ridge with a few thousand effective
degrees of freedom cannot memorise individual training sequences, so it has little to gain from seeing a
near-copy. Whether a much stronger model, such as the published plantGREP CNN or a large fine-tuned gLM,
gains more is open; our CNN is weaker than the published ones, which puts a floor under what we can claim.

The practical reading is that homology-aware splits are worth building, not because they overturn published
numbers but because they put a bound on how much those numbers could have been inflated. Reporting that
bound is cheap once the splits exist.

On composition, the answer depends on the dataset. Roughly half of each embedding's variance is predictable
from 6-mer counts alone. What the rest adds ranges from almost nothing on enhancers, where the
composition-only part of the embedding actually predicts better than the whole, to 0.14 to 0.20 R2 on
terminators, where the foundation models clearly carry something a bag of 6-mers does not. A plant-pretrained
model (AgroNT) carries more of that extra than a general one (NT-v2) on every dataset, which is what
plant-specific pretraining is supposed to buy. We do not know what the extra signal is. Position is the
obvious candidate for terminators, since polyadenylation and cleavage are positional and k-mer counts throw
position away, but we have not run the positional baseline that would test it.

The variant result is the one we think matters most. Zero-shot likelihood scores from both models are at
chance: they cannot say whether removing a transcription factor binding site raises or lowers enhancer
activity. This is not a subtle failure, and it argues against using pretraining likelihood to rank
regulatory edits in plants, which is one of the uses these models are proposed for. Trained models do
predict something, but every one of them reaches only 42% to 60% of the ceiling set by the assay's own
replicates, and in four of five conditions a single experimental replicate predicts another replicate
better than any model predicts either. Put on a common scale, every model is 20 to 35 points further from
the ceiling on mutation effects than on sequence strength. Sequence-level R2 therefore overstates how well
these models would guide a promoter edit.

We also found that fine-tuning changes the ranking. Frozen NT-v2-100M is the weakest model we tested; the
same weights fine-tuned reach the published Jores CNN on promoters and match a supervised CNN on variant
effects. Papers that compare frozen embeddings alone, including ours in its first pass, understate what
these models can do.

Several limitations bound all of this. Our CNN underperforms the published models it stands in for, by 0.14
R2 on terminators, so our leakage estimates for high-capacity models are conservative at best. We tested two
foundation models, not the current best ones: PlantCAD2 and Botanic1 report stronger plant numbers and are
not included here. The variant set is 741 pairs from one library in one species pair, with shuffles designed
by the original authors, and the effects are small, which limits how finely models can be separated. The
composition decomposition is linear, so a model could encode composition non-linearly and be scored as
carrying extra signal. Fine-tuning used one hyperparameter setting and, on promoters, one seed. Finally, all
three datasets measure 170 bp fragments in transient assays, which is not the same thing as a promoter in
its genomic context.

The obvious next steps are a positional k-mer baseline to explain the terminator result, the stronger plant
models on the same splits, and more measured mutation pairs, ideally from an assay designed to separate
model predictions rather than to validate motifs. We release the splits, the homology scores and the variant
pairs so that a model can be scored on them without redoing any of this.
