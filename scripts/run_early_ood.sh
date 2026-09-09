#!/bin/bash
# Early OOD evals that do NOT depend on round-5 training (both use the G6 adapter
# grpo_qwen35_lora_v2). Run NOW in parallel with round-5 training so their results
# land ~17:30, well before Teacher Zhou returns at 22:00.
#   E1_ext     OOD_external  external-corpus OOD (Cornell US Code, via split_ext.json)
#   E2_size10  OOD_size10    10% KB scale point (completes 10/20/50/100% curve)
set -u
PY=/root/miniconda3/bin/python
cd /path/to/legal_english || exit 1
mkdir -p logs

run_one() {
  local name="$1"; shift
  echo "=== $name START $(date '+%H:%M:%S') ===" >> logs/early_ood.log
  "$@" >> "logs/early_${name}.log" 2>&1
  echo "=== $name DONE $(date '+%H:%M:%S') rc=$? ===" >> logs/early_ood.log
}

# E1: external corpus OOD (Cornell US Code). KB = train+val of split_ext = the
# original 861 EUR-Lex/Oyez passages (split_ext train+val are identical to
# split.json's). Test = 50 of the 70 uscode rows, x2 levels = 100 rows.
run_one ext \
  env LEGAL_SPLIT_PATH=/path/to/legal_english/data/split_ext.json \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag OOD_external --src-prefix-filter uscode --n-test 50 \
  --levels beginner,intermediate --seed 7

# E2: 10% KB scale point (86 passages). Same protocol as OOD_size/OOD_size50.
run_one size10 \
  $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
  --tag OOD_size10 --kb-frac 0.10 --n-test 50 --seed 7

echo "=== EARLY_OOD_ALL_DONE $(date '+%H:%M:%S') ===" >> logs/early_ood.log
