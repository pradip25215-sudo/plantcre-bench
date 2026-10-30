# Methods

## Data

We used three published plant massively parallel reporter assay (MPRA) datasets, all with 170 bp
sequences measured in the same two assay systems.

Core promoters came from Jores et al. (2021): 170 bp regions from -165 to +5 bp relative to annotated
transcription start sites of Arabidopsis thaliana, Zea mays and Sorghum bicolor, with plant STARR-seq
strength measured in tobacco leaves and in maize protoplasts. We merged the authors' CNN training and
test files for both assay systems into 76,728 unique sequences. 71,238 of these were measured in both
systems. The published split differs between the two systems for 12,776 sequences, so we kept the two
split columns separate and never mixed them.

Terminators came from Gorjifard et al. (2024): 53,409 sequences from Arabidopsis and maize with strength
in the same two systems, including 2,199 synthetic GC-shuffled controls.

Enhancers came from the plantGREP library (Jores et al., 2026): 337,342 rows covering 180,921 distinct
170 bp sequences from accessible chromatin regions of Arabidopsis, tomato, maize and sorghum, each
measured in five conditions (tobacco leaves in light, in the dark, at 35 C and at 10 C, and maize
protoplasts). Each sequence appears in both orientations, which we kept together in every split.

All sequences are exactly 170 bp and contain only A, C, G and T. No test sequence is an exact copy of a
training sequence in any of the three published splits.

## Homology detection and homology-aware splits

We measured homology with BLAST+ 2.17.0 using the parameters of hashFrag (Rafi et al., 2025): word size
7, match 1, mismatch -1, gap open 2, gap extend 1, e-value 100, dust off, one HSP per pair, and the same
correction of the BLAST alignment score, which applies the gap-open penalty only once per gap. Each
dataset was searched against itself, with reverse complements added to the database. For an
alignment with $m$ matching bases, $x$ mismatches, $g$ gap openings and $e$ gapped positions, the score is

$$S = r\,m + p\,x - g_o\,g - g_e\,(e - g)$$

with $r = 1$, $p = -1$, $g_o = 2$ and $g_e = 1$. BLAST charges the gap-extension penalty on the opening
position as well, so we subtract it once per gap, as hashFrag does. We ran blastn
directly rather than through hashFrag's blastn_module, because the module cannot pass
-min_raw_gapped_score; every other setting and all post-processing came from hashFrag.

To set a homology threshold, we took 2,000 natural sequences from each dataset and made a
dinucleotide-shuffled copy of each, preserving every dinucleotide count as well as the first and last
base. For every sequence $i$ we recorded its best alignment score to any other sequence,

$$M_i = \max_{j \neq i} S_{ij},$$

taking $M_i = 0$ when BLAST reports no hit. Shuffled sequences
reached at most 34 (99.9th percentile 33), while 1% of natural sequences reached 94 or more. We used 40
as the main threshold, above the maximum chance score with room for the larger full database, and
repeated every analysis at 34 and at 60.

Two settings were checked on this pilot before the full runs. Word size 11 was 25 times faster but
recovered only 90% of the homologous pairs that word size 7 found at threshold 40, at any e-value, so we
kept word size 7. Adding -min_raw_gapped_score 20 made the search 17 times faster and the output 58 times
smaller while recovering 99.3% of pairs at threshold 40 (2 of 303 missed, both separated from their
partner in the resulting groups) and 100% at threshold 60. We used it for the full runs and report results
at both thresholds.

Homologous groups are the connected components of the graph whose edges are sequence pairs scoring at or
above the threshold, found with union-find. Homology-aware splits assign whole groups to train, validation
and test in 80/10/10 proportions, with five random seeds. No homologous pair crosses train and test in any
split, which we verified for every split we made. For enhancers the unit is the base sequence, so forward
and reverse copies always land together. For Experiment 3 we additionally forced every group containing a
variant wild-type sequence into the test set; those forced splits are used only there, because they change
the split proportions (Experiment 1 uses unforced enhancer splits).

For the published splits we also recorded, for each test sequence, its best alignment score to any training
sequence, and binned test sequences by that score.

## Models

6-mer ridge: counts of all 4,096 6-mers per sequence, log1p transformed, standardised, with ridge
regression and the penalty chosen by generalised cross-validation over 15 values from 1e-2 to 1e5.

Frozen embeddings: NT-v2-100M (InstaDeepAI/nucleotide-transformer-v2-100m-multi-species, revision
f34324c6) and AgroNT-1B (InstaDeepAI/agro-nucleotide-transformer-1b, revision b0e1ea1f). For each
sequence we took the mean of a hidden layer over the sequence tokens, leaving out the CLS token and
padding:

$$e = \frac{\sum_{t=2}^{T} a_t\, h_t}{\sum_{t=2}^{T} a_t}$$

where $h_t$ is the hidden state of token $t$ in the chosen layer, $a_t$ is 1 for a real token and 0 for
padding, and $t$ starts at 2 to skip the CLS token. Embeddings are stored in float16. We saved every second layer for NT-v2 and every fourth for AgroNT. The layer
was chosen once per model and task on the validation part of one random split, with training subsampled
to 50,000 sequences, and then kept fixed for every result in the paper. Layer 10 of 22 was best for NT-v2
and layer 12 of 40 for AgroNT, on every task. The same ridge procedure as above was then used.

CNN: a Jores-style convolutional network trained from scratch on one-hot sequences, with three
convolutional blocks (128 filters, widths 13, 13 and 9, batch normalisation, ReLU, two max-pooling steps)
and a dense head (dropout 0.3, 256 units, dropout 0.3, linear output). Assays measured on the same split
are predicted by one multi-task output with the loss masked where a measurement is missing. Adam at 1e-3,
batch 256, at most 40 epochs, early stopping on validation loss with patience 5.

Fine-tuned NT-v2: all weights of NT-v2-100M plus a head that takes the mean of the last hidden layer
(CLS and padding left out), dropout 0.1 and a linear layer. AdamW with 2e-5 for the backbone and 1e-3 for
the head, weight decay 0.01, gradient clipping at 1.0, batch 64, four epochs, keeping the epoch with the
lowest validation loss.

Zero-shot variant scores: for a wild-type and mutant pair we masked the tokens that differ in the
wild-type sequence and scored the mutant as the sum over those positions of log p(mutant token) minus
log p(wild-type token), using the masked language modelling head with no training.

Targets were standardised using the training mean and standard deviation of each run, and predictions
converted back before scoring.

## Composition decomposition

For each model and dataset we regressed the standardised embedding on standardised 6-mer counts with a
multi-output ridge, choosing the penalty $\lambda$ on a held-out tenth of the training rows:

$$B = (K^{\top}K + \lambda I)^{-1} K^{\top} E, \qquad \hat{E} = KB, \qquad R = E - \hat{E}$$

where $K$ holds the standardised 6-mer counts and $E$ the standardised embeddings. This splits the
embedding into the part 6-mers can predict, $\hat{E}$, and a residual, $R$. We then ran the same ridge probe for strength
on 6-mer counts alone, the full embedding, the predictable part, the residual, and 6-mer counts
concatenated with the residual. The decomposition never sees the strength values. For enhancers we used
forward copies only and subsampled training to 100,000 rows.

## Variant pairs and the noise ceiling

The plantGREP validation library shuffles one high-confidence binding site of one transcription factor
family inside an accessible-region sequence and measures the result. We paired each shuffled sequence
with its unmodified control, giving 741 pairs over 16 TF families; 12 mutants had no matching control and
were dropped. The two sequences of a pair differ in 2 to 14 bases (median 4). The measured effect is the
mutant strength minus the wild-type strength in each of the five conditions. Across pairs, shuffling
lowers strength in 57% to 64% of cases, with a median effect of -0.06 to -0.15 log2 units.

Every model scored here was trained on enhancer strength using the forced homology split at threshold 40,
seed 1, so no wild-type sequence and no homolog of one appears in training; we checked this directly. The
predicted effect is the model's prediction for the mutant minus its prediction for the wild type. We
report the Spearman correlation between predicted and measured effects, and sign accuracy over the pairs
whose measured effect is at least 0.5 log2 units in absolute value.

The noise ceiling uses the authors' replicate-level data, in which each condition was measured in two or
three independent experiments. We computed the effect separately in each replicate, took the mean
correlation between replicate pairs, and applied the Spearman-Brown formula to get the reliability of the
replicate average. With $k$ replicates and mean pairwise correlation $\bar{r}$ between single-replicate
effects,

$$\rho_{kk} = \frac{k\,\bar{r}}{1 + (k-1)\,\bar{r}}, \qquad \text{ceiling} = \sqrt{\rho_{kk}},$$

and that square root is the highest correlation any model can reach against the replicate average. We
also report each model as a percentage of its ceiling, $100 \times r_{\text{model}} / \text{ceiling}$,
which puts wild-type strength and mutation effects on one scale.
At the pair set used here, the ceiling for the effect is 0.85 to 0.94 (Spearman) and a single replicate
predicts another at 0.46 to 0.77. The same calculation on wild-type strength gives a ceiling of 0.96 to
0.99 (R2). Raising the minimum number of barcodes per measurement raises the ceiling and shrinks the pair
set; we report thresholds of 1, 5 and 10 barcodes and use 1 in the figures, which is the set all models
were scored on.

## Evaluation

Performance is reported as the squared Pearson correlation between predicted and measured values on test
sequences, which is how the three source papers report their own results. Frozen-embedding and 6-mer
results use five split seeds, the CNN five seeds on promoters and terminators and three on enhancers, and
fine-tuned NT-v2 three seeds on terminators and one on promoters. We summarise over seeds with the median,
because one CNN run on a promoter split collapsed to a much lower value and a mean would follow it; every
seed is shown in the figures.

Controls (synthetic terminator sequences, viral and other non-plant enhancer controls) are kept in
training and were also kept in the reported metrics: removing them from the test sets changes no R2 by
more than 0.003, which we checked for all three datasets.

## Code and compute

All analyses ran on one machine with a Tesla P4 GPU. Code is at <repository URL>, with one script per
step and a dated decision log. Package versions are pinned in environment.yml and requirements-lock.txt.
Model weights were loaded from pinned revisions and from a local snapshot, so no result depends on a
model being re-downloaded.
