#!/bin/bash
# same pilot, but word size 11 + evalue 10 (much faster) -> does it still catch the real homologs?
set -euo pipefail
export PATH="$HOME/miniforge3/envs/hashfrag/bin:$PATH"
cd ~/plantcre/data/homology
out=pilot_real_w11.work
rm -rf $out
time hashFrag blastn_module --fasta-path pilot_real.fa --word-size 11 --evalue 10 \
  --max-target-seqs 6000 --blastdb-label pilot_real_w11 --threads 16 --output-dir $out
hashFrag process_blast_results_module --blastn-path $out/pilot_real_w11.blastn.out \
  --processed-blastn-path $out/pilot_real_w11.processed.tsv
wc -l $out/pilot_real_w11.processed.tsv
echo "RECALL TEST DONE"
