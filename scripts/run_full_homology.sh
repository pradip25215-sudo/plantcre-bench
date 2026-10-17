#!/bin/bash
# full all-vs-all BLAST for the three datasets, one after the other
# usage: run_full_homology.sh <min_raw_score>
set -euo pipefail
MIN=$1
H=~/plantcre/data/homology
for d in terminators promoters enhancers; do
  echo "################ $d  ($(date))"
  ~/plantcre/scripts/blast_all.sh $H/${d}_all.fa ${d}_min${MIN} 24 $MIN 10000
done
echo "FULL HOMOLOGY DONE ($(date))"
