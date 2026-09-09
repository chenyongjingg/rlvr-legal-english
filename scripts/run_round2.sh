#!/bin/bash
# round-2 night batch: reward ablation + baselines + OOD/ADV/multi-seed + verifier
# + offline metrics + stats. Each stage independent; FAIL continues to next stage.
# Launch:  setsid bash run_round2.sh > logs/run_round2.log 2>&1 &
# Stage progress marker written to logs/round2_stage.txt for polling.
set +e
ROOT=/path/to/legal_english
PY=/root/miniconda3/bin/python
cd "$ROOT"
mkdir -p logs
echo "== run_round2 start $(date '+%F %T') =="

mark() { # $1 = stage label, $2 = pass/fail
  echo "STAGE $1: $2 $(date '+%F %T')" >> logs/round2_stage.txt
  echo "    [$1] $2 ($(date '+%T'))" >> logs/run_round2.log
}

run_stage() { # $1 label, rest = command
  local label="$1"; shift
  echo "== STAGE $label start $(date '+%F %T') ==" >> logs/run_round2.log
  "$@" >> logs/run_round2.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then mark "$label" PASS; else mark "$label" FAIL; fi
  echo "== STAGE $label end rc=$rc $(date '+%T') ==" >> logs/run_round2.log
}

# ---------- A. reward ablation GRPO training (start from SFT, same budget) ----------
for x in full diff term copy faith fmt; do
  drop=""
  [ "$x" != "full" ] && drop="$x"
  run_stage "A_$x" $PY scripts/05_grpo_train.py --drop "$drop" --diff-w 0.30 \
      --n-train 24 --G 4 --epochs 4 --seed 42 --out models/grpo_abl_$x
done

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

echo "== run_round2 done $(date '+%F %T') ==" >> logs/run_round2.log
mark "ALL_DONE" PASS
