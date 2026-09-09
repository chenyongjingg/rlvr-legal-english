#!/bin/bash
# round-5: full-budget reward ablations (n=40/G=6/6ep, matching G6 v2 budget,
# but from SFT init like the reduced-budget ablations -> clean budget-scaling test)
set -u
PY=/root/miniconda3/bin/python
cd /path/to/legal_english || exit 1
mkdir -p logs
STAGE=logs/round5_stage.txt
: > "$STAGE"

run_stage() {
  local name="$1"; shift
  echo "=== STAGE $name START $(date '+%H:%M:%S') ===" >> "$STAGE"
  "$@" >> "logs/run_round5_${name}.log" 2>&1
  echo "=== STAGE $name DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"
}

# 1) full-budget A_full (full reward stack, from SFT) -- the full-budget reference
run_stage full \
  $PY scripts/05_grpo_train.py --drop "" --diff-w 0.30 \
  --n-train 40 --G 6 --epochs 6 --seed 42 --out models/grpo_abl_full_full

# 2) full-budget A_diff (drop difficulty reward, from SFT)
run_stage diff \
  $PY scripts/05_grpo_train.py --drop diff --diff-w 0.30 \
  --n-train 40 --G 6 --epochs 6 --seed 42 --out models/grpo_abl_full_diff

echo "=== ALL_DONE $(date '+%H:%M:%S') ===" >> "$STAGE"
