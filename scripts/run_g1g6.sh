#!/bin/bash
# Run the full G1-G6 experiment matrix sequentially. One config per log.
cd /root/autodl-tmp/legal_english
P=/root/miniconda3/bin/python
S=scripts/06_framework.py
N=50

echo "START $(date +%H:%M:%S)"
$P $S --model qwen3-4b       --n-test $N --seed 7 --tag G1 > logs/G1.log 2>&1
echo "G1 done $(date +%H:%M:%S)"
$P $S --model qwen3-4b --rag  --n-test $N --seed 7 --tag G2 > logs/G2.log 2>&1
echo "G2 done $(date +%H:%M:%S)"
$P $S --model qwen3-4b --agents --n-test $N --seed 7 --tag G3 > logs/G3.log 2>&1
echo "G3 done $(date +%H:%M:%S)"
$P $S --model qwen3-4b --rag --agents --n-test $N --seed 7 --tag G4 > logs/G4.log 2>&1
echo "G4 done $(date +%H:%M:%S)"
$P $S --model qwen35-sft  --rag --agents --n-test $N --seed 7 --tag G5 > logs/G5.log 2>&1
echo "G5 done $(date +%H:%M:%S)"
$P $S --model qwen35-grpo --rag --agents --n-test $N --seed 7 --tag G6 > logs/G6.log 2>&1
echo "G6 done $(date +%H:%M:%S)"
echo "ALL DONE $(date +%H:%M:%S)"
