#!/bin/bash
# All-vs-all BLAST with hashFrag's exact flags + -min_raw_gapped_score.
# Queries are split into chunks and run as independent blastn processes (1 thread each).
# Result is identical to one big run (each query is searched on its own against the same DB),
# but all cores get used and finished chunks show progress.
# usage: blast_all.sh <fasta> <label> <n_parallel> <min_raw_score> <max_target_seqs>
set -euo pipefail
export PATH="$HOME/miniforge3/envs/hashfrag/bin:$PATH"
FA=$(realpath "$1"); LABEL=$2; NP=$3; MIN=$4; MAXT=$5
OUT=$(dirname "$FA")/${LABEL}.work
rm -rf "$OUT"; mkdir -p "$OUT/chunks"
echo "blastn: $(which blastn) | $(blastn -version | head -1)"
echo "fasta=$FA label=$LABEL parallel=$NP min_raw=$MIN max_target=$MAXT"

# 1) database with reverse complements (<id>_Reversed), built by hashFrag itself
hashFrag blastn_module --fasta-path "$FA" --blastdb-label "$LABEL" --exec-makeblastdb-only --output-dir "$OUT"

# 2) split queries round-robin into NP chunks (each record = header line + sequence line)
awk -v n="$NP" -v d="$OUT/chunks" '/^>/{f=sprintf("%s/q_%03d.fa", d, i%n); i++} {print > f}' "$FA"
echo "chunks: $(ls $OUT/chunks/q_*.fa | wc -l)"

# 3) one blastn per chunk, NP at a time; touch .done when a chunk finishes (that's the progress bar)
export DB="$OUT/${LABEL}.blastdb" MIN MAXT
time ls "$OUT"/chunks/q_*.fa | xargs -P "$NP" -I{} bash -c '
  blastn -query "{}" -db "$DB" -out "{}.out" \
    -word_size 7 -gapopen 2 -gapextend 1 -penalty -1 -reward 1 \
    -max_target_seqs "$MAXT" -xdrop_ungap 20 -xdrop_gap 30 -xdrop_gap_final 100 \
    -evalue 100 -dust no -num_threads 1 -max_hsps 1 \
    -outfmt "6 std score positive gaps" -strand plus -min_raw_gapped_score "$MIN" && touch "{}.done"'

n_done=$(ls "$OUT"/chunks/*.done 2>/dev/null | wc -l); n_all=$(ls "$OUT"/chunks/q_*.fa | wc -l)
[ "$n_done" -eq "$n_all" ] || { echo "ERROR: only $n_done of $n_all chunks finished"; exit 1; }
cat "$OUT"/chunks/q_*.fa.out > "$OUT/${LABEL}.blastn.out"

# 4) hashFrag's own post-processing (score correction, best hit per pair)
hashFrag process_blast_results_module --blastn-path "$OUT/${LABEL}.blastn.out" \
  --processed-blastn-path "$OUT/${LABEL}.processed.tsv"
wc -l "$OUT/${LABEL}.processed.tsv"
echo "BLAST ALL DONE: $LABEL"
