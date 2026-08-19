#!/bin/bash
# round-7 continuation: after loop-depth (B) reaches ALL_DONE, run the OOD
# configs under the FIXED framework (commentary-stripping + regenerating gate).
# G6's honest number = B's L3_max3 stage (same config, seed, levels).
# G1 is already honest (SFT emits no commentary, verified).
set -u
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/legal_english || exit 1
mkdir -p logs
STAGE=logs/round7_rerun_stage.txt
: > "$STAGE"

echo "waiting for loop-depth ALL_DONE $(date '+%H:%M:%S')" >> "$STAGE"
for i in $(seq 1 240); do
  if grep -q "ALL_DONE" logs/round7_stage.txt 2>/dev/null; then
    echo "loop-depth ALL_DONE $(date '+%H:%M:%S')" >> "$STAGE"; break
  fi
  sleep 60
done
grep -q "ALL_DONE" logs/round7_stage.txt 2>/dev/null \
  || echo "TIMEOUT waiting loop-depth; continuing $(date '+%H:%M:%S')" >> "$STAGE"

run_stage() {
  local name="$1"; shift
  local tag=""
  local prev=""
  for a in "$@"; do
    if [ "$prev" = "--tag" ]; then tag="$a"; break; fi
    prev="$a"
  done
  if [ -n "$tag" ] && [ -f "results/$tag.jsonl" ]; then
    echo "=== STAGE $name SKIPPED ($tag exists) $(date '+%H:%M:%S') ===" >> "$STAGE"
    return 0
  fi
  echo "=== STAGE $name START $(date '+%H:%M:%S') ===" >> "$STAGE"
  "$@" >> "logs/run_round7_${name}.log" 2>&1
  echo "=== STAGE $name DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"
}

# OOD external (US Code) under the fixed framework
run_stage OOD_ext_fix \
  env LEGAL_SPLIT_PATH=/root/autodl-tmp/legal_english/data/split_ext.json \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag OOD_ext_fix --n-test 50 --levels beginner,intermediate --seed 7 \
  --max-attempts 3 --src-prefix-filter uscode_

# OOD India (case law) under the fixed framework (if corpus is ready)
if [ -f /root/autodl-tmp/legal_english/data/split_ext_in.json ]; then
  run_stage OOD_india_fix \
    env LEGAL_SPLIT_PATH=/root/autodl-tmp/legal_english/data/split_ext_in.json \
    $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
    --tag OOD_india_fix --n-test 50 --levels beginner,intermediate --seed 7 \
    --max-attempts 3 --src-prefix-filter in_
else
  echo "=== STAGE OOD_india_fix SKIPPED (split_ext_in.json missing) $(date '+%H:%M:%S') ===" >> "$STAGE"
fi

echo "=== RERUN_ALL_DONE $(date '+%H:%M:%S') ===" >> "$STAGE"
