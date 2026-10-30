#!/bin/bash
# word 7 / evalue 100 (our final setting) + drop junk hits below raw score 25 at the BLAST level.
# must give recall 1.0 at 34/40/60, with a much smaller output file.
set -euo pipefail
export PATH="$HOME/miniforge3/envs/hashfrag/bin:$PATH"
cd ~/plantcre/data/homology
out=pilot_real_w7_min25.work
rm -rf $out
echo "===== word 7 evalue 100 min_raw_gapped_score 25 ====="
time hashFrag blastn_module --fasta-path pilot_real.fa --word-size 7 --evalue 100 \
  --max-target-seqs 6000 --blastdb-label pilot_real_w7_min25 --threads 16 \
  --blastn-args="-min_raw_gapped_score 25" --output-dir $out
hashFrag process_blast_results_module --blastn-path $out/pilot_real_w7_min25.blastn.out \
  --processed-blastn-path $out/pilot_real_w7_min25.processed.tsv
echo "lines, baseline vs min25:"
wc -l pilot_real.work/pilot_real.processed.tsv $out/pilot_real_w7_min25.processed.tsv
echo "MINSCORE TEST DONE"
