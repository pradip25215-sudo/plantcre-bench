#!/bin/bash
# which fast setting still finds >=99% of the word-7 homologs? evalue is the main suspect.
# evalue 1000 ~ mimics the full database (which is ~15x bigger than the pilot one)
set -euo pipefail
export PATH="$HOME/miniforge3/envs/hashfrag/bin:$PATH"
cd ~/plantcre/data/homology
for cfg in "11 100" "11 1000" "9 1000"; do
  set -- $cfg; w=$1; e=$2
  out=pilot_real_w${w}_e${e}.work
  rm -rf $out
  echo "===== word $w evalue $e ====="
  { time hashFrag blastn_module --fasta-path pilot_real.fa --word-size $w --evalue $e \
      --max-target-seqs 6000 --blastdb-label pilot_real_w${w}_e${e} --threads 16 --output-dir $out \
      > /dev/null 2>&1 ; } 2>&1 | grep real
  hashFrag process_blast_results_module --blastn-path $out/pilot_real_w${w}_e${e}.blastn.out \
    --processed-blastn-path $out/pilot_real_w${w}_e${e}.processed.tsv > /dev/null 2>&1
done
echo "RECALL TEST 2 DONE"
