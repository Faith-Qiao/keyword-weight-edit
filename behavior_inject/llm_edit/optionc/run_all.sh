#!/bin/bash
# run_all.sh: all Option C jobs, sequentially, unbuffered logs in optionc/logs/. Each job skips itself if already done.
# Parallel on a multi-GPU node: run individual lines with CUDA_VISIBLE_DEVICES=<i> in separate shells / jobs.
set -u
cd "$(dirname "$0")"; mkdir -p logs results
PY=${PY:-python}
for job in "strict_b50_r1b full" "strict_b10_r1 full" "closed_form full" "none full" \
           "strict_b50_r1b attn" "strict_b10_r1 attn" "closed_form attn"; do
  set -- $job
  echo "$(date '+%F %T') start persist $1 $2"
  $PY -u persist.py $1 $2 2>&1 | tee logs/persist_$1_$2.txt
done
echo "$(date '+%F %T') start invert"
$PY -u invert.py 2>&1 | tee logs/invert.txt
echo "$(date '+%F %T') all done"
