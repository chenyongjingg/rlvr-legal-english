#!/bin/bash
# round-7 tail: after the OOD fix chain (run_rerun.sh) reaches RERUN_ALL_DONE,
# re-score ALL OLD stored results with the fixed cleaner + full reward stack
# (NLI included) -> honest per-tag/per-level numbers for the paper tables.
set -u
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/legal_english || exit 1
mkdir -p logs
STAGE=logs/round7_recompute_stage.txt
: > "$STAGE"

echo "waiting for RERUN_ALL_DONE $(date '+%H:%M:%S')" >> "$STAGE"
for i in $(seq 1 240); do
  if grep -q "RERUN_ALL_DONE" logs/round7_rerun_stage.txt 2>/dev/null; then
    echo "RERUN_ALL_DONE seen $(date '+%H:%M:%S')" >> "$STAGE"; break
  fi
  sleep 60
done

echo "=== RECOMPUTE START $(date '+%H:%M:%S') ===" >> "$STAGE"
$PY scripts/recompute_honest.py >> logs/recompute_honest_run.log 2>&1
echo "=== RECOMPUTE DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"
echo "=== RECOMPUTE_ALL_DONE $(date '+%H:%M:%S') ===" >> "$STAGE"
