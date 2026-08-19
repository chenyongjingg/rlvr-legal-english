#!/bin/bash
# round-3b: re-runs round-2's FAILED E_verifier stage (--tags G6,G1,B1,ADV)
# after round-3 completes, using the fixed 08_verifier_eval.py (.format bug fixed:
# literal {{ }} in JUDGE_USR JSON example). Stage markers -> logs/round3b_stage.txt.
set +e
ROOT=/root/autodl-tmp/legal_english
PY=/root/miniconda3/bin/python
cd "$ROOT"
mkdir -p logs
echo "== run_round3b start $(date '+%F %T') (waiting round-3 ALL_DONE) ==" >> logs/run_round3b.log

# --- wait for round-3 to finish so the GPU is free (max 5h) ---
for i in $(seq 1 300); do
  if grep -q "STAGE ALL_DONE: PASS" logs/round3_stage.txt 2>/dev/null; then
    echo "== round-3 ALL_DONE seen, starting $(date '+%F %T') ==" >> logs/run_round3b.log
    break
  fi
  sleep 60
done

mark() { echo "STAGE $1: $2 $(date '+%F %T')" >> logs/round3b_stage.txt
         echo "    [$1] $2 ($(date '+%T'))" >> logs/run_round3b.log; }

run_stage() { local label="$1"; shift
  echo "== STAGE $label start $(date '+%F %T') ==" >> logs/run_round3b.log
  "$@" >> logs/run_round3b.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then mark "$label" PASS; else mark "$label" FAIL; fi
  echo "== STAGE $label end rc=$rc $(date '+%T') ==" >> logs/run_round3b.log; }

run_stage "E_verifier_fixed" $PY scripts/08_verifier_eval.py --tags G6,G1,B1,ADV,A_full,G6_pure_s11,OOD_doc

echo "== run_round3b done $(date '+%F %T') ==" >> logs/run_round3b.log
mark "ALL_DONE" PASS
