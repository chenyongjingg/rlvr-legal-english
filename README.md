# Difficulty-Controllable Generation of Legal-English Learning Materials via Reinforcement Learning with Verifiable Rewards

Code and configuration for the paper of the same name (submitted to
Neurocomputing). This repository supports reproduction of the training,
inference, evaluation, and honest-recalculation pipelines described in the
paper.

## Overview

An end-to-end framework that generates legal-English learning materials from
raw legal sources at three target difficulty levels (beginner / intermediate /
advanced). The system combines:

- a small language model (SLM) as generator,
- retrieval-augmented generation (RAG) over a legal-term KB,
- a multi-agent elaboration loop (planner / writer / editor / reviewer),
- reinforcement learning with verifiable rewards (RLVR; GRPO), jointly
  controlling readability (Flesch Reading Ease band), terminology retention,
  anti-copying, and semantic faithfulness (NLI entailment).

Headline results (honest, decontamination-aware protocol, 100 held-out
passages): total verifiable reward 0.800 (beginner 0.802 / intermediate
0.798); three-level readability separated monotonically (Flesch 64.7 / 25.9 /
10.8 at targets 63 / 45 / 26); NLI contradiction at most 8%; robust to
within-corpus shifts and random seeds (0.828 ± 0.001 over four); difficulty
control transferred to two external corpora (U.S. Code 0.66 with domain-matched
retrieval; Indian case law 0.66).

## Environment

- Python 3.10+
- PyTorch 2.8
- Transformers 5.15, TRL 1.10, PEFT 0.20
- A GPU with ≥ 32 GB VRAM (training was run on an NVIDIA RTX 5090)
- FAISS for the retrieval index (`02_build_index.py`)

Models used:
- Generator / SLM: Qwen3.5-4B (Apache-2.0); QLoRA adapters r=16, α=32 on the
  language-module transformer (7 modules, 21.23M trainable params)
- SFT data generation: Qwen3-4B-Instruct-2507
- Verifier / judge: Qwen3-4B-Instruct-2507 (cross-family judge on the Qwen3.5
  generator)
- NLI: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
- Embedding: sentence-transformers/MiniLM-L6-v2

## Data sources

The corpus is built from publicly available legal sources:

| Source | License |
|---|---|
| EUR-Lex | CC BY (metadata CC0) |
| Wex (Cornell) | CC BY-NC-SA |
| Oyez | CC BY-NC-SA |
| Cornell LII U.S. Code | public-domain text + LII mark copyright |
| Indian case law (external-domain evaluation) | public court judgments |

`01_build_corpus.py` / `02_build_index.py` build and index the corpus from
these sources; `build_external_corpus.py` and `build_external_india.py` build
the two external-domain evaluation corpora. The raw scraped data is not shipped
in this repository (see below).

## Reproduction

The numbered scripts form the pipeline. Shell scripts (`run_*.sh`) drive the
batch experiments described in the paper.

```bash
# 1. Build + split the corpus (80/10/10)
python 01_build_corpus.py
python 02_make_split.py
python 02_build_index.py

# 2. SFT: generate teacher data (Qwen3-4B-Instruct) and train the SLM
python 03_gen_sft_data.py
python 04_sft_train.py

# 3. RLVR: GRPO with the five-term verifiable reward stack
python 05_grpo_train.py --init-adapter models/grpo_qwen35_lora \
    --diff-w 0.30 --n-train 40 --G 6 --epochs 6 --seed 42 \
    --out models/grpo_qwen35_lora_v2

# 4. Inference / evaluation framework (G1–G6 matrix, ablations, OOD)
python 06_framework.py ...
python 07_eval.py ...
python 08_verifier_eval.py ...
python 09_baselines.py ...
python 10_metrics.py ...

# 5. Honest recalculation (decontamination-aware protocol, App. A of the paper)
python recompute_honest.py ...
python 11_metrics_honest.py ...
python honest_table.py ...
python cross_check_numbers.py ...
```

Run scripts also exist for the specific experiment matrices (`run_g1g6.sh`,
`run_round2.sh`, `run_round3.sh`, `run_loopdepth.sh`, `run_round5_fullab.sh`,
`run_round6.sh`, `run_recompute.sh`, `run_verify.sh`, ...).

## Notes on weights and data

- Model adapters and the scraped corpus are not included in this GitHub
  repository due to size and licensing. They can be obtained by contacting the
  authors, and the raw data is fully reconstructible from the public sources
  listed above via the build scripts.
- The evaluation follows a decontamination-aware protocol: outputs with pure
  LLM meta-commentary are scored as zero, and the reported numbers are the
  "honest" recomputed values (see the paper's Appendix A).

## Citation

If you use this code, please cite the paper:

```
Difficulty-Controllable Generation of Legal-English Learning Materials via
Reinforcement Learning with Verifiable Rewards. Submitted to Neurocomputing.
```

## License

Code in this repository is released under the MIT License. The underlying model
weights are Apache-2.0 (Qwen family); the legal corpora retain their original
licenses as listed in the table above.
