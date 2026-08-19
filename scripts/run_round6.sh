#!/bin/bash
# round-6: post-ablation evals, queued to start ONLY after round-5 training finishes.
#   R6_abl_full   pure-protocol eval of full-budget A_full  (grpo_abl_full_full)
#   R6_abl_diff   pure-protocol eval of full-budget A_diff  (grpo_abl_full_diff)
#   R6_ext        external-corpus OOD (Cornell US Code, via LEGAL_SPLIT_PATH)
#   R6_size10     10% KB scale point (completes 10/20/50/100% curve)
set -u
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/legal_english || exit 1
mkdir -p logs
STAGE=logs/round6_stage.txt
: > "$STAGE"

# ---- wait for round-5 (A_full + A_diff full-budget ablations) to finish ----
# NOTE: wait budget is 300 polls x 30s = 150 min. If exhausted WITHOUT seeing
# ALL_DONE, we must ABORT (exit 1) -- otherwise round-6 falls through and starts
# the full-budget evals while A_diff@full is still training, and the abl_diff
# stage crashes on the not-yet-saved models/grpo_abl_full_diff adapter.
if ! grep -q "ALL_DONE" logs/round5_stage.txt 2>/dev/null; then
  echo "waiting for round-5 training..." >> "$STAGE"
  found=0
  for i in $(seq 1 300); do
    if grep -q "ALL_DONE" logs/round5_stage.txt 2>/dev/null; then
      echo "round-5 done, starting round-6 $(date '+%H:%M:%S')" >> "$STAGE"
      found=1
      break
    fi
    sleep 30
  done
  if [ "$found" = "0" ]; then
    echo "round-5 NOT done after 300 polls; aborting round-6 $(date '+%H:%M:%S')" >> "$STAGE"
    exit 1
  fi
fi

run_stage() {
  local name="$1"; shift
  # idempotent: skip a stage if its --tag output already exists (the early OOD
  # driver run_early_ood.sh produces OOD_external/OOD_size10 while round-5 trains).
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
  "$@" >> "logs/run_round6_${name}.log" 2>&1
  echo "=== STAGE $name DONE $(date '+%H:%M:%S') rc=$? ===" >> "$STAGE"
}

# 1) pure-protocol evals of the two full-budget ablations (same protocol as the
#    reduced-budget A_* tags: no rag/agents, n_test=50, seed 7, beg+inter)
run_stage abl_full \
  $PY scripts/06_framework.py --model qwen35-grpo \
  --adapter models/grpo_abl_full_full --tag A_full_full_budget

run_stage abl_diff \
  $PY scripts/06_framework.py --model qwen35-grpo \
  --adapter models/grpo_abl_full_diff --tag A_diff_full_budget

# 2) external-corpus OOD: test rows = Cornell US Code (id prefix uscode_),
#    KB = original train+val (861) via split_ext.json, full framework.
run_stage ext \
  env LEGAL_SPLIT_PATH=/root/autodl-tmp/legal_english/data/split_ext.json \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag OOD_external --src-prefix-filter uscode --n-test 50 \
  --levels beginner,intermediate --seed 7

# 3) 10% KB scale point (same protocol as OOD_size / OOD_size50)
run_stage size10 \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag OOD_size10 --kb-frac 0.10 --n-test 50 --seed 7

echo "=== ALL_DONE $(date '+%H:%M:%S') ===" >> "$STAGE"
