#!/bin/bash
# round-7 tail RE-RUN (fix): the original run_verify.sh wait loop TIMED OUT at
# 02:49 (180x60s from 23:49) and judged RAW/polluted rewrites because no honest
# files existed yet. This runner waits for RERUN_ALL_DONE (OOD fixes complete),
# THEN judges the CLEANED honest rewrites + recomputes offline metrics, so the
# verifier/metrics numbers are consistent with the honest tables.
set -u
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/legal_english || exit 1
mkdir -p logs
STAGE=logs/round7_verify2_stage.txt
: > "$STAGE"

echo "waiting for RERUN_ALL_DONE $(date '+%H:%M:%S')" >> "$STAGE"
for i in $(seq 1 150); do
  if grep -q "RERUN_ALL_DONE" logs/round7_rerun_stage.txt 2>/dev/null; then
    echo "RERUN_ALL_DONE seen $(date '+%H:%M:%S')" >> "$STAGE"; break
  fi
  sleep 60
done

echo "=== VERIFIER2 START $(date '+%H:%M:%S') ===" >> "$STAGE"
$PY scripts/08_verifier_eval.py --tags L3_max3,G1,B1,ADV,G6_pure_s11,A_full,OOD_doc \
  >> logs/run_round7_verifier2.log 2>&1
echo "=== VERIFIER2 DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"

echo "=== METRICS2 START $(date '+%H:%M:%S') ===" >> "$STAGE"
$PY scripts/11_metrics_honest.py >> logs/run_round7_metrics2.log 2>&1
echo "=== METRICS2 DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"

echo "=== VERIFY2_ALL_DONE $(date '+%H:%M:%S') ===" >> "$STAGE"
