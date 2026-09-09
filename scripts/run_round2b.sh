#!/bin/bash
# round-2 restart: stages B-F only (A ablation training already completed).
# Fixes the 06_framework.py torch-shadowing bug that failed all B/C stages.
# Launch: setsid bash run_round2b.sh > logs/run_round2b.log 2>&1 &
set +e
ROOT=/path/to/legal_english
PY=/root/miniconda3/bin/python
cd "$ROOT"
mkdir -p logs
echo "== run_round2b start $(date '+%F %T') =="

mark() {
  echo "STAGE $1: $2 $(date '+%F %T')" >> logs/round2_stage.txt
  echo "    [$1] $2 ($(date '+%T'))" >> logs/run_round2b.log
}

run_stage() {
  local label="$1"; shift
  echo "== STAGE $label start $(date '+%F %T') ==" >> logs/run_round2b.log
  "$@" >> logs/run_round2b.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then mark "$label" PASS; else mark "$label" FAIL; fi
  echo "== STAGE $label end rc=$rc $(date '+%T') ==" >> logs/run_round2b.log
}

# ---------- B. pure-generator evals (no rag/agents) ----------
run_stage "B_G6_pure" $PY scripts/06_framework.py --model qwen35-grpo --tag G6_pure
run_stage "B_G6_pure_s11" $PY scripts/06_framework.py --model qwen35-grpo \
    --tag G6_pure_s11 --seed 11
run_stage "B_M2" $PY scripts/06_framework.py --model qwen35-sft --tag M2
for x in full diff term copy faith fmt; do
  run_stage "B_A_$x" $PY scripts/06_framework.py --model qwen35-grpo \
      --adapter models/grpo_abl_$x --tag A_$x
done
run_stage "B_ADV" $PY scripts/06_framework.py --model qwen35-grpo --tag ADV \
    --levels beginner,intermediate,advanced

# ---------- C. OOD (full framework) ----------
run_stage "C_OOD_doc" $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
    --tag OOD_doc --kb-exclude-title "2006/123" --title-filter "2006/123"
run_stage "C_OOD_mismatch" $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
    --tag OOD_mismatch --kb-only-prefix oyez --src-prefix-filter eur
run_stage "C_OOD_size" $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
    --tag OOD_size --kb-frac 0.2

# ---------- D. baselines ----------
run_stage "D_B1" $PY scripts/09_baselines.py --method b1 --tag B1
run_stage "D_B2" $PY scripts/09_baselines.py --method b2 --tag B2

# ---------- E. verifier-as-judge ----------
run_stage "E_verifier" $PY scripts/08_verifier_eval.py --tags G6,G1,B1,ADV

# ---------- F. offline metrics + stats ----------
run_stage "F_metrics" $PY scripts/10_metrics.py
run_stage "F_stats" $PY scripts/07_eval.py "$ROOT/results"

echo "== run_round2b done $(date '+%F %T') ==" >> logs/run_round2b.log
mark "ALL_DONE" PASS
