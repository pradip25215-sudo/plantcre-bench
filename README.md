# PlantCRE-Bench

Code for "Homology, composition and mutation effects: what DNA foundation models actually learn from
plant MPRA data".

We re-evaluate NT-v2-100M and AgroNT-1B (frozen embeddings with a ridge probe, and NT-v2 fine-tuned
end to end) against a 6-mer ridge and a CNN trained from scratch, on three plant MPRA datasets:
Jores et al. (2021) core promoters, Gorjifard et al. (2024) terminators, and the plantGREP enhancer
library (Jores et al. 2026). Three things are tested that the original benchmarks do not test:

1. Homology between training and test sequences, measured with hashFrag-style BLAST alignment scores,
   and splits rebuilt so that no homologous pair crosses the train/test line.
2. How much of a frozen embedding is 6-mer composition, by regressing each embedding on 6-mer counts
   and keeping the residual.
3. Whether any model ranks the effect of shuffling a single transcription factor binding site, scored
   against a replicate noise ceiling on 741 measured wild-type / shuffle pairs.

## Data

Not in this repository. Download the three source datasets from the original papers, put them under
`data/`, then run `scripts/prepare_data.py`. The processed splits and the variant pair table are
archived separately (see the data availability section of the paper).

## Environment

```
conda env create -f environment.yml
conda activate plantcre
```

`requirements-lock.txt` has the exact pinned versions that produced the numbers in the paper. BLAST+
2.17.0 is needed for the homology search and comes from the hashfrag conda env.

## Order to run things

```
python scripts/prepare_data.py          # read the three datasets, write a common table
python scripts/make_fastas.py           # one FASTA per dataset for BLAST
bash   scripts/blast_all.sh             # all-vs-all BLAST, both strands, word size 7
python scripts/pilot_threshold.py       # dinucleotide-shuffled control, sets tau
python scripts/make_splits.py           # random, homology-aware and forced splits
python scripts/embed_hf.py              # frozen embeddings from NT-v2 and AgroNT
python scripts/exp1_homology.py         # Experiment 1, ridge probes on both splits
python scripts/cnn_exp1.py              # Experiment 1, CNN
python scripts/finetune_ntv2.py         # Experiment 1, fine-tuned NT-v2
python scripts/exp2_decompose.py        # Experiment 2, composition decomposition
python scripts/prepare_tfval.py         # build the 741 variant pairs
python scripts/exp3_variants.py         # Experiment 3, trained models
python scripts/exp3_zeroshot.py         # Experiment 3, masked-LM scores
python scripts/delta_noise_ceiling.py   # Spearman-Brown ceiling from replicates
```

`docs/decisions.md` is a dated log of what was decided and why, including the settings that were
checked on a pilot before the full run.

## Which script makes which figure

| Figure | Script | Output |
| --- | --- | --- |
| Figure 1 | `scripts/make_figures.py` | `figures/fig1_leakage.{png,pdf}` |
| Figure 2 | `scripts/make_figures.py` | `figures/fig2_composition.{png,pdf}` |
| Figure 3 | `scripts/make_figures.py` | `figures/fig3_variants.{png,pdf}` |
| Figure 4 | `scripts/make_figures.py` | `figures/fig4_wt_vs_delta.{png,pdf}` |
| Figure S1 | `scripts/fig_s1_threshold.py` | `figures/figS1_threshold.{png,pdf}` |
| Figure S2 | `scripts/make_s2_icons.py` | `figures/figS2_workflow.{png,pdf,svg}` |
| Figure S3 | `scripts/make_diagrams.py` | `figures/figS3_cnn.{png,pdf,svg}` |

Figures 1 to S1 read from `results/`, so the experiment scripts have to run first. S2 and S3 are
drawings and need nothing but matplotlib.

## Results tables

Every number in the paper comes from a file in `results/`. The per-figure summaries are
`results/fig1_medians.tsv` and `results/fig4_points.tsv`; the rest are the raw per-seed and
per-condition tables.

## License

MIT, see `LICENSE`.
