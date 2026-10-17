#!/bin/bash
# All-vs-all BLAST done exactly the way hashFrag's blastn_module does it
# (same makeblastdb, same reverse complements, same blastn flags and outfmt),
# plus -min_raw_gapped_score so junk hits never get written (hashFrag's module has no way to pass this).
# usage: blast_all.sh <fasta> <label> <threads> <min_raw_score> <max_target_seqs>
set -euo pipefail
export PATH="$HOME/miniforge3/envs/hashfrag/bin:$PATH"
FA=$(realpath "$1"); LABEL=$2; T=$3; MIN=$4; MAXT=$5
OUT=$(dirname "$FA")/${LABEL}.work
rm -rf "$OUT"; mkdir -p "$OUT"
echo "blastn: $(which blastn) | $(blastn -version | head -1)"
echo "fasta=$FA label=$LABEL threads=$T min_raw=$MIN max_target=$MAXT"

# 1) database with reverse complements (<id>_Reversed), built by hashFrag itself
hashFrag blastn_module --fasta-path "$FA" --blastdb-label "$LABEL" --exec-makeblastdb-only --output-dir "$OUT"

# 2) the search: hashFrag's construct_blastn_command flags, word 7 / evalue 100, + min raw score
time blastn -query "$FA" -db "$OUT/${LABEL}.blastdb" -out "$OUT/${LABEL}.blastn.out" \
  -word_size 7 -gapopen 2 -gapextend 1 -penalty -1 -reward 1 \
  -max_target_seqs "$MAXT" -xdrop_ungap 20 -xdrop_gap 30 -xdrop_gap_final 100 \
  -evalue 100 -dust no -num_threads "$T" -max_hsps 1 \
  -outfmt '6 std score positive gaps' -strand plus -min_raw_gapped_score "$MIN"

# 3) hashFrag's own post-processing (score correction, best hit per pair)
hashFrag process_blast_results_module --blastn-path "$OUT/${LABEL}.blastn.out" \
  --processed-blastn-path "$OUT/${LABEL}.processed.tsv"
wc -l "$OUT/${LABEL}.processed.tsv"
echo "BLAST ALL DONE: $LABEL"
