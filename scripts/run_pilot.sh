#!/bin/bash
# pilot: same BLAST settings we'll use for the real thing, on real vs shuffled sequences
set -euo pipefail
cd ~/plantcre/data/homology
for name in pilot_real pilot_shuf; do
  echo "===== $name ====="
  time hashFrag blastn_module \
    --fasta-path ${name}.fa \
    --word-size 7 \
    --evalue 100 \
    --max-target-seqs 6000 \
    --blastdb-label ${name} \
    --threads 16 \
    --output-dir ${name}.work
  hashFrag process_blast_results_module \
    --blastn-path ${name}.work/${name}.blastn.out \
    --processed-blastn-path ${name}.work/${name}.processed.tsv
  wc -l ${name}.work/${name}.processed.tsv
done
