#!/bin/bash
# round-7 tail-2: after recompute_honest.py reaches RECOMPUTE_ALL_DONE, judge the
# CLEANED rewrites with the independent verifier (Qwen3-4B) and recompute the
# offline metrics (SARI/BLEU/readability) on them, so verifier + metrics numbers
# are consistent with the honest tot/FRE reported elsewhere.
set -u
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/legal_english || exit 1
mkdir -p logs
STAGE=logs/round7_verify_stage.txt
: > "$STAGE"

echo "waiting for RECOMPUTE_ALL_DONE $(date '+%H:%M:%S')" >> "$STAGE"
for i in $(seq 1 180); do
  if grep -q "RECOMPUTE_ALL_DONE" logs/round7_recompute_stage.txt 2>/dev/null; then
    echo "RECOMPUTE_ALL_DONE seen $(date '+%H:%M:%S')" >> "$STAGE"; break
  fi
  sleep 60
done

echo "=== VERIFIER START $(date '+%H:%M:%S') ===" >> "$STAGE"
$PY scripts/08_verifier_eval.py --tags L3_max3,G1,B1,ADV,G6_pure_s11,A_full,OOD_doc \
  >> logs/run_round7_verifier.log 2>&1
echo "=== VERIFIER DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"

echo "=== METRICS START $(date '+%H:%M:%S') ===" >> "$STAGE"
$PY scripts/11_metrics_honest.py >> logs/run_round7_metrics.log 2>&1
echo "=== METRICS DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"

echo "=== VERIFY_ALL_DONE $(date '+%H:%M:%S') ===" >> "$STAGE"
