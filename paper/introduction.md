# Introduction

Changing a few bases in a promoter or enhancer can change how much a gene is expressed, and plant breeders
now edit these regions directly. Editing the promoter of the tomato CLAVATA3 gene changed fruit size and
inflorescence branching, and that work needed saturation mutagenesis followed by selection, because nobody
could say in advance which bases mattered. A model that ranks the effect of candidate edits would replace
much of that screening.

DNA language models are the current candidate for such a model. AgroNT was pretrained on 48 plant genomes,
PlantCaduceus on 16 angiosperms, PlantCAD2 on 65, and Botanic1 on 320, and each reports strong numbers on
plant regulatory tasks. Two of those tasks are plant MPRA regression: the strength of 170 bp core promoters
from Jores et al. and of terminators from Gorjifard et al., both measured by plant STARR-seq in tobacco
leaves and maize protoplasts, and both packaged in the Plant Genomic Benchmark. More recently the plantGREP
library measured enhancer activity for over 350,000 accessible-region sequences from four species in five
conditions, and trained a CNN on them.

Three things are missing from how these numbers are produced. First, every one of them comes from a random
split of sequences drawn from the same genomes. Plant genomes are repeat-rich and heavily paralogous, so
training and test sets can share homologous sequences; hashFrag showed that this inflates reported
performance for human sequence-to-function models, and the effect has not been measured for plant MPRA data.
Second, the baselines are usually a CNN or another language model, not sequence composition. On 57 genomic
classification datasets, k-mer frequencies beat the frozen embeddings of Evo 2, DNABERT-2 and HyenaDNA on
most datasets, and probing work found no advantage of gLM representations over one-hot models for
cell-type-specific regulatory genomics. How much of a plant gLM's score is composition is not known. Third,
sequence-level R2 is not the quantity a crop engineer needs. Editing a promoter means asking what happens
when one binding site changes, which is a question about a pair of sequences, not one sequence. Plant
variant benchmarks instead use proxies such as allele frequency or variant class, which reflect selection
only indirectly.

Here we evaluate frozen embeddings, fine-tuned models, zero-shot likelihood scores, a CNN trained from
scratch and a 6-mer baseline on all three plant MPRA datasets under three conditions that the published
numbers do not meet. We build homology-aware splits with alignment-score thresholds calibrated against
dinucleotide-shuffled sequences, and ask how much performance those splits cost. We decompose each embedding
into the part 6-mer counts predict and the rest, and ask what the rest adds. And we score models on 741
wild-type and single-TFBS-shuffle pairs whose effects were measured in the same assay, with every wild-type
sequence and its homologs held out of training, against a noise ceiling computed from experimental
replicates.
