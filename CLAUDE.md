# PlantCRE-Bench

## Goal
PlantCRE-Bench is a benchmark of DNA foundation models on plant STARR-seq/MPRA data. It compares
NT-v2-100M and AgroNT-1B (frozen embeddings + ridge, and NT-v2 fine-tuned end to end) with
6-mer ridge and a CNN trained from scratch. Datasets:
- Promoters: Jores 2021 (At, Zm, Sb; leaf + proto strength)
- Enhancers: plantGREP 2026 ACR library (5 conditions: light, dark, warm, cold, maize)
- Terminators: Gorjifard 2024 (tobacco + maize)

Questions:
- Exp 1: how much of the published performance comes from homology leakage between train and test?
- Exp 2: how much of what the gLM embeddings carry is just 6-mer composition?
- Exp 3: can a model rank the effect of a single TFBS shuffle (741 WT / shuffle pairs, 16 TF families)?

Current numbers: `docs/results_summary.md`. Decision log: `docs/decisions.md`. There is no README.

## Layout
- `scripts/`: the whole pipeline (Python + bash)
  - data prep: `prepare_data.py`, `prepare_tfval.py`, `build_variant_pairs.py`, `inspect_data.py`
  - homology: `make_fastas.py`, `run_pilot.sh`, `run_recall_test.sh`, `pilot_threshold.py`,
    `compare_recall.py`, `blast_all.sh`, `run_full_homology.sh`, `make_splits.py`
  - embeddings: `embed_hf.py`
  - Exp 1: `exp1_homology.py`, `cnn_exp1.py`, `probe_promoters_original.py`, `probe_model_original.py`,
    `check_nohit_bin.py`
  - Exp 2: `exp2_decompose.py`
  - Exp 3: `exp3_variants.py`, `exp3_cnn.py`, `exp3_zeroshot.py`, `delta_noise_ceiling.py`
  - fine-tuning: `finetune_ntv2.py` (promoters + the Exp 3 enhancer split)
- `docs/`: `decisions.md` (dated, source of truth), `results_summary.md`, `*_log.txt` (saved run output)
- `results/`: result TSVs (tracked; `results/*.parquet` is ignored)
- `environment.yml`, `requirements-lock.txt`: the `plantcre` conda env

Not in git (see `.gitignore`); these exist only on the server:
- `data/`: `external/` (raw downloads), `processed/` (clean parquet tables), `homology/` (FASTAs, BLAST work dirs),
  `splits/` (homology splits and homology bins)
- `embeddings/<tag>/`: `<dataset>.npy` (float16, N x layers x hidden), `<dataset>.ids.parquet`, `meta.json`
- also ignored: `*.pt`, `*.npy`, `__pycache__/`, `.ipynb_checkpoints/`

## Running code
- All code runs on the server **cells**, in the conda env **`plantcre`**.
- hashFrag 1.0.2 and BLAST+ 2.17.0 run in the conda env **`hashfrag`**. The shell scripts put
  `~/miniforge3/envs/hashfrag/bin` first in PATH, because an old BLAST 2.15 in `~/ncbi-blast-2.15.0+`
  would otherwise run. The pip hashFrag is missing `messages.json`, which was copied in by hand from the GitHub repo.
- Scripts hardcode `ROOT = ~/plantcre`, so they expect the repo checked out there.
- HuggingFace: `HF_HOME=/data3/pradip/hf_cache`.
  - The first download of a model runs in the foreground.
  - **Every background job sets `HF_HUB_OFFLINE=1`** (the HF version check hung on cells without it).
  - **Load models with `--local-path <snapshot dir>`** (`embed_hf.py`). `exp3_zeroshot.py` and
    `finetune_ntv2.py` find the snapshot themselves through `local_path()`. Never load by Hub name in a background job.
  - Pinned revisions: NT-v2-100M `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`
    @ `f34324c6fde36a4f635f0f1f06cac5d25acd6798`; AgroNT-1B `InstaDeepAI/agro-nucleotide-transformer-1b`
    @ `b0e1ea1f53a2bf5bb29f8eab7a7e553bf06c1ab1`.

## Main decisions (full text in docs/decisions.md)
- **D1** Promoters use one shared split for leaf and proto (12,776 sequences differ between the published assay splits).
- **D2** Controls (35S, AB80, Cab-1, rbcS, random/GC terminators) are excluded from benchmark metrics.
  Not applied yet, see Open issues.
- **D3** All 741 variant WTs and their homologs are held out of training in our splits (forced to test).
  540 of them were in the original training split.
- **D4** Variant results are reported separately for strong-effect and null TF families.
- **D5** Main homology threshold = 40 (shuffled-sequence max was 34). Sensitivity checks at 34 and 60.
- **D6** (final) Full all-vs-all BLAST uses hashFrag's exact blastn flags (word 7, evalue 100, dust off, max_hsps 1)
  plus `-min_raw_gapped_score 20`, run directly (`blast_all.sh`). Recall is 0.993 at 40 and 1.000 at 60, so
  main conclusions must hold at both 40 and 60.
- **D7** Exp 1 uses the unforced enhancer splits (`data/splits/enhancers_nf_*`). Exp 3 keeps the forced (D3) splits.
  The old `results/exp1_enhancers.tsv` (forced) is not used.

Also: embedding layers are chosen on validation only (NT-v2 layer 10, AgroNT layer 12). The test set never
picks anything.

## Working rules
- Add a dated entry to `docs/decisions.md` for every new decision or data finding.
- Never tune or select on test data.
- Report variant (Exp 3) results stratified by TF family.

## Open issues
- D2 is not applied yet. The probe (`probe_*`), `exp1_homology` and `cnn_exp1` results on the published splits
  include controls on purpose, so they stay comparable to the published numbers.
- Exp 3 is not affected, because it only uses the 741 variant pairs.
- A D2-filtered rerun of the final tables is still to do.
