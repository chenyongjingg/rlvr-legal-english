#!/bin/bash
# round-7 (experiment B): multi-agent loop depth study.
# Same full-framework protocol as OOD_external (G6 v2, rag+agents, n=50 beg+inter,
# seed 7) but with --max-attempts capped at 1 / 2 / 3 / 5. L3 == the current
# default setting, so it re-measures G6's loop with the same budget -> an internal
# consistency check. Answers "does allowing more revision iterations help?"
set -u
PY=/root/miniconda3/bin/python
cd /path/to/legal_english || exit 1
mkdir -p logs
STAGE=logs/round7_stage.txt
: > "$STAGE"

run_stage() {
  local name="$1"; shift
  local tag=""
  local prev=""
  for a in "$@"; do
    if [ "$prev" = "--tag" ]; then tag="$a"; break; fi
    prev="$a"
  done
  if [ -n "$tag" ] && [ -f "results/$tag.jsonl" ]; then
    echo "=== STAGE $name SKIPPED $(date '+%H:%M:%S') (results/$tag.jsonl exists) ===" >> "$STAGE"
    return 0
  fi
  echo "=== STAGE $name START $(date '+%H:%M:%S') ===" >> "$STAGE"
  "$@" >> "logs/run_round7_${name}.log" 2>&1
  echo "=== STAGE $name DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"
}

run_stage L1 \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag L1_max1 --n-test 50 --levels beginner,intermediate --seed 7 --max-attempts 1

run_stage L2 \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag L2_max2 --n-test 50 --levels beginner,intermediate --seed 7 --max-attempts 2

run_stage L3 \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag L3_max3 --n-test 50 --levels beginner,intermediate --seed 7 --max-attempts 3

run_stage L5 \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag L5_max5 --n-test 50 --levels beginner,intermediate --seed 7 --max-attempts 5

echo "=== ALL_DONE $(date '+%H:%M:%S') ===" >> "$STAGE"
