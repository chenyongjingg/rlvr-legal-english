#!/bin/bash
# round-3 supplementary experiments (uses idle GPU window after round-2).
# Waits for round-2's ALL_DONE marker, then runs in paper-value order:
#   R3_ADV_full      three-tier FULL framework (rag+agents)   ~60min  (RQ1 three-tier e2e)
#   R3_G6_pure_s21   extra seed (multi-seed -> mean±std)      ~17min
#   R3_G6_pure_s42   extra seed                               ~17min
#   R3_verifier_ext  independent judge: A_full,s11,OOD_doc    ~12min  (credibility across variants)
#   R3_G6_full107    headline on full test set (107 src)      ~90min
# Stage markers -> logs/round3_stage.txt ; log -> logs/run_round3.log
set +e
ROOT=/path/to/legal_english
PY=/root/miniconda3/bin/python
cd "$ROOT"
mkdir -p logs
echo "== run_round3 start $(date '+%F %T') (waiting for round-2 ALL_DONE) ==" >> logs/run_round3.log

# --- wait for round-2 batch to finish so the GPU is free (max 6h) ---
for i in $(seq 1 360); do
  if grep -q "STAGE ALL_DONE: PASS" logs/round2_stage.txt 2>/dev/null; then
    echo "== round-2 ALL_DONE seen, starting round-3 $(date '+%F %T') ==" >> logs/run_round3.log
    break
  fi
  sleep 60
done

mark() { echo "STAGE $1: $2 $(date '+%F %T')" >> logs/round3_stage.txt
         echo "    [$1] $2 ($(date '+%T'))" >> logs/run_round3.log; }

run_stage() { local label="$1"; shift
  echo "== STAGE $label start $(date '+%F %T') ==" >> logs/run_round3.log
  "$@" >> logs/run_round3.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then mark "$label" PASS; else mark "$label" FAIL; fi
  echo "== STAGE $label end rc=$rc $(date '+%T') ==" >> logs/run_round3.log; }

run_stage "R3_ADV_full" $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
    --tag ADV_full --levels beginner,intermediate,advanced --seed 7
run_stage "R3_G6_pure_s21" $PY scripts/06_framework.py --model qwen35-grpo \
    --tag G6_pure_s21 --seed 21
run_stage "R3_G6_pure_s42" $PY scripts/06_framework.py --model qwen35-grpo \
    --tag G6_pure_s42 --seed 42
run_stage "R3_verifier_ext" $PY scripts/08_verifier_eval.py --tags A_full,G6_pure_s11,OOD_doc
run_stage "R3_G6_full107" $PY scripts/06_framework.py --model qwen35-grpo --rag --agents \
    --tag G6_full107 --n-test 107 --seed 7

echo "== run_round3 done $(date '+%F %T') ==" >> logs/run_round3.log
mark "ALL_DONE" PASS
