# When Automatic Rewards, an LLM Judge, and Human Raters Disagree

Code and data for the paper

> **When Automatic Rewards, an LLM Judge, and Human Raters Disagree: Evaluating
> Difficulty-Controlled Generation of Legal-English Learning Materials**
>
> Yongjin Chen, Ziying Xue, Guyue Zhou
>
> (submitted to *Applied Intelligence*)

This repository backs the current version of the manuscript. It ships the
generator pipeline, the evaluation code, and — most importantly — the **shared
multi-lens evaluation set** (40 legal passages × 6 systems) with the anonymized
human ratings, the LLM-judge scores on the identical texts, and the scoring
scripts that reproduce Tables 11–13 of the paper. Trained LoRA adapters are
**not** distributed here (see [Weights](#weights)).

## What the paper studies

Non-native learners of legal English need materials whose difficulty is tunable
to the learner's level while the legal register, terminology, and meaning are
preserved. Whether a rewrite is "good" is not observable from the text alone — it
is measured through one of several lenses: a text-computable verifiable-reward
stack, a reference LLM used as an independent judge, or human raters. Using a
realistic difficulty-controllable generator as a testbed, the paper scores **one
shared blind set with all three lenses** and asks how far the lenses agree:

- **The generator.** A QLoRA-fine-tuned small language model (Qwen3.5-4B, ~21.2 M
  trainable parameters) with retrieval augmentation over a legal-term KB, a
  multi-agent revision loop, and GRPO over a five-dimensional verifiable reward
  stack (readability band, terminology retention, anti-copying, NLI
  faithfulness, format). Under an honest, decontamination-aware protocol the
  framework genuinely controls text statistics: total verifiable reward **0.800**
  in the two-level configuration and **0.841** over the full three-level run,
  which separates the three target Flesch levels monotonically at
  **67.5 / 30.3 / 20.2** against targets 63 / 45 / 26. This capability is real;
  the paper's question is whether the lenses that measure it agree on which
  outputs are good.
- **The three lenses on one blind set.** 40 passages × 6 systems (the source
  text, an SFT fine-tune, Flan-T5-large, BART-large, a lexicon rewriter, and the
  full RLVR framework) were scored by (a) the automatic reward stack,
  (b) an independent cross-lineage LLM judge (Qwen3-4B-Instruct, never
  fine-tuned on our data), and (c) three legal-English teachers (240 blind
  judgments each).
- **Findings.** (1) Human raters barely agree with one another (pairwise
  weighted κ 0.02–0.11; only 4.6–10.8% exact three-way agreement; the three-rater
  mean has ICC(2,3) = 0.10–0.32 over all 240 rated cells, 0.26–0.61 over the 200
  generated cells), so a single rater is not a stable ground truth.
  (2) The panel mean is nonetheless signal: the judge tracks the three-rater mean
  on the same texts on every dimension (pooled Spearman ρ 0.53–0.70, within-system
  0.38–0.53, n = 200) and does so at least as well as the raters recover one
  another (leave-one-out ρ 0.07–0.29). The system-level ranking (ρ = 0.90 over the
  five generated systems) is reported as an ordering rather than a test: at n = 5
  the exact permutation p is .083. (3) The automatic reward stack is the outlier
  lens: its system ranking does not track the human panel (ρ = 0.20) or the judge
  (ρ = 0.10), and it ranks the RLVR framework first (automatic total reward 0.846
  on the shared blind set) on the same outputs that both external lenses place
  near the bottom.

## Repository layout

| Path | Contents |
|---|---|
| `human_eval/` | The multi-lens evaluation study (Tables 11–13, §4.13) |
| `human_eval/samples/` | Shared blind set: `items.jsonl` (40×6 texts + automatic scores), `unblind_key.json` (blind label → system), `rated_R1.csv` / `rated_R2.csv` / `rated_R3.csv` (anonymized human ratings) |
| `human_eval/H1/` | `human40_judge.jsonl` (LLM-judge scores on the identical 40×6 texts), `h1c_judge_human_analysis.py` (three-way alignment analysis), `h1c_interrater.py` (pairwise weighted κ between the human raters, §4.13), `icc_reliability.py` (ICC(2,3)/ICC(3,3) of the three-rater mean, §4.13 / §5.3), `h1c_clustered_stats.py` (passage-level cluster bootstrap, design effect, §4.13 level split and difficulty-gap robustness, §3.5 / §4.13), `h1c_out/` (aggregate JSONs behind Table 12 and §4.13; see the provenance note on `h1c_means.json` below), plus the alternate-judge and order-control set behind Table 14 / §4.5 and Table 15 / §4.6 — see "Alternate judges and the order control" below |
| `human_eval/make_human_eval_samples.py` | Builds the blind set from generator outputs (sampling + blinding) |
| `human_eval/rubric.md` | The rating rubric shown to the human raters |
| `results/` | Verified result artifacts: `honest_table.json` (Table 1 / ablations / OOD / loop depth), `h1a_mainset.json` + `verifier_eval.json` (judge on each system's own main set, Table 13), the honest per-row JSONL for the main chain, loop depth, ablations, OOD and seeds, `examples_qualitative.json` (§4.11 annotation), and the per-system JSONL the blind-set analysis reads. **Read `results/README.md` first** — it records which of the two `verifier_eval` files backs Table 6, and why `G6v4.jsonl` and `G6v4_honest.jsonl` differ |
| `scripts/` | Generator pipeline: corpus build → SFT → GRPO → framework → evaluation → honest re-scoring (`01_*.py` … `11_*.py`, `honest_table.py`, `recompute_honest.py`, `run_*.sh`) |
| `prompts/judge_prompt.txt` | The judge system/user prompt (also embedded verbatim in `scripts/08_verifier_eval.py` and `scripts/09_judge_human_items.py`) |

### Reproducing specific results

| Paper result | Data | Code |
|---|---|---|
| Table 1 (main chain G1–G6, honest totals) | `results/honest_table.json`, `results/{G1..G5}_honest.jsonl`, `results/metrics_honest.json` | `scripts/honest_table.py`, `scripts/recompute_honest.py` |
| Table 3 (OOD, two external domains) | `results/honest_table.json` (OOD tags), `results/OOD_*_honest.jsonl` | `scripts/07_eval.py`, `scripts/build_external_*.py` |
| Table 6 (verifier probe) | `results/verifier_eval.json` (dev probe; **not** `verifier_eval_rerun.json`, see `results/README.md`) | `scripts/08_verifier_eval.py` |
| §4.11 qualitative taxonomy counts | `results/examples_qualitative.json` (released annotation; the labels are a recorded inspection, not an FRE-band rule) | `scripts/09_qualitative_examples.py` |
| Loop depth L1–L5 (§4.10) | `results/{L1_max1,L2_max2,L3_max3,L5_max5}.jsonl` | `scripts/06_framework.py` |
| Table 11 automatic column (honest basis) | `results/G6v4_honest.jsonl` (the honest variant of `G6v4.jsonl`; two pure-commentary rows scored 0) | `scripts/recompute_honest.py` |
| Tables 11–12 (§4.13, three lenses on the blind set) | `human_eval/samples/`, `human_eval/H1/human40_judge.jsonl`, `results/{B2,M2_honest,B3_flant5,B4_bart,G6v4}.jsonl` | `human_eval/H1/h1c_judge_human_analysis.py` |
| Table 13 (judge on each system's own main set) | `results/h1a_mainset.json`, `results/verifier_eval.json` | `scripts/08_verifier_eval.py`, `scripts/09_judge_human_items.py` |
| §3.5 cluster-bootstrap intervals + design effect; §4.13 level split + difficulty-gap robustness | the same blind-set sheets as Tables 11–12 | `human_eval/H1/h1c_clustered_stats.py` |
| Table 14 / §4.5 (three alternate judges on the same 40×6 cells) | `human_eval/H1/h1c_out/altjudge_{gemma3-27b,haiku45,dsv4flash}.jsonl` | `human_eval/H1/h1c_c_altjudge_report.py` |
| Table 15 / §4.6 (block-order control) | `human_eval/H1/h1c_out/{altjudge,swaporder}_{gemma3-27b,haiku45}.jsonl` | `human_eval/H1/h1c_e_swap_report.py` |
| §4.5 judge-gap p-values | the same alternate-judge sheets | `human_eval/H1/h1c_judge_gap_p.py` (and, independently, `h1c_gap_test.py`) |
| §4.6 same-prompt stability (92.9%, 240 of 240) | `human_eval/H1/h1c_out/selfcons_{gemma3-27b,haiku45}.jsonl` | `human_eval/H1/h1c_selfcons_report.py` |
| §4.13 rater leave-one-out ρ 0.15–0.54 | `human_eval/samples/rated_R{1,2,3}.csv` + `unblind_key.json` | `human_eval/H1/h1c_loo_sb.py` |
| Figures 1–2 | `results/honest_table.json` | `scripts/make_framework_figure.py`, `scripts/make_results_figure.py` |

**Provenance note on `h1c_means.json`.** Its `H` and `J` blocks are means over the
shared 40-item blind set (20 passages × 2 levels × one system), and so is Table 11's
automatic column. Its `A` block is **not** on that basis: it is the mean over each
configuration's full set (100 rows; 150 for G6v4, read from the non-honest
`results/G6v4.jsonl`), so it does not reproduce Table 11's automatic column — e.g.
B2 is `faith` 0.714 and `term` 0.970 here against 0.622 and 1.000 on the blind set.
The manuscript's Table 11 uses the blind-set basis and reports both bases side by
side in §4.13 (shared-set / full-set total: 0.727/0.743, 0.756/0.761, 0.798/0.806,
0.714/0.737, 0.846/0.841, the last under the honest rule). Use the per-row files in
`results/` with `human_eval/samples/items.jsonl` as the blind-set filter to reproduce
Table 11's automatic column; do not read it off `h1c_means.json`.

To re-run the three-way analysis from the shipped data alone (CPU only):

```bash
python human_eval/H1/h1c_judge_human_analysis.py
python human_eval/H1/h1c_interrater.py
python human_eval/H1/icc_reliability.py
python human_eval/H1/h1c_clustered_stats.py
```

The first regenerates `human_eval/H1/h1c_out/h1c_verdict.json` (Table 12), the
second `human_eval/H1/h1c_out/h1c_interrater.json` (the mean pairwise weighted
κ and the exact-three-way-agreement rates quoted in §4.13), the third
`human_eval/H1/h1c_out/icc_reliability.json` (the ICC(2,3) = 0.10–0.32 figure in
the abstract and §5.3, and the ICC(3,3) ceiling check), and the fourth
`human_eval/H1/h1c_out/h1c_clustered_stats.json` (the passage-level
cluster-bootstrap intervals and design effect of §3.5, and the §4.13 level split
and difficulty-gap robustness check); all shipped copies reproduce exactly. They
read the rated CSVs and `unblind_key.json` from `human_eval/samples/`, the judge
scores from `human_eval/H1/`, and the automatic per-system scores from
`results/`.

The bootstrap and level-split numbers are **seed-exact**: `h1c_clustered_stats.py`
fixes `seed = 20260911` and `B = 2000` (800 for the gap robustness check) so the
interval endpoints are reproducible rather than one random draw, and the
manuscript quotes the seed beside the interval. The automatic difficulty quantity
is the *stored* `parts.diff` sub-reward as released in the per-row files;
recomputing it from `fre` gives the same quantity to within their 4-decimal
rounding (Pearson 1.000000, max |delta| 2e-4), but rank statistics on the two can
differ by ~0.01 through tie ordering, so the basis is pinned in the script.

### Alternate judges and the order control (Table 14 / §4.5, Table 15 / §4.6)

```bash
python human_eval/H1/h1c_c_altjudge_report.py   # -> h1c_out/h1c_c_altjudge_report.json
python human_eval/H1/h1c_e_swap_report.py       # -> h1c_out/h1c_e_swap_report.json
python human_eval/H1/h1c_judge_gap_p.py         # -> h1c_out/h1c_judge_gap_p.json
python human_eval/H1/h1c_selfcons_report.py     # -> h1c_out/h1c_selfcons_report.json
python human_eval/H1/h1c_loo_sb.py              # -> h1c_out/h1c_loo_sb.json
```

All five read only the JSONL shipped in `human_eval/H1/h1c_out/` plus
`human_eval/H1/human40_judge.jsonl`, need no credentials and no network, and
reproduce the committed JSON byte-for-byte. `h1c_gap_test.py` recomputes the
§4.5 judge-gap p-values by a second, independent route and must agree with
`h1c_judge_gap_p.py`.

**What is and is not reproducible here.** Gemma-3-27B, Claude Haiku 4.5 and
DeepSeek-V4-Flash are hosted models reached over an HTTP API. The *calls* are not
reproducible: they were sampled at a provider-chosen temperature, the providers
may retire these snapshots, and re-issuing a call would not return identical
text. The *analysis* is: every number in Table 14, Table 15 and §4.6 is a
function of the released per-cell rows, so it recomputes exactly from this repo
without contacting anyone. `h1c_altjudge_api.py` is shipped as the record of what
was asked — the exact prompts, the order swap, the retry rule and the response
parser — not as a re-runnable step. It reads its endpoint and key from
`APIN_JUDGE_BASE` / `APIN_JUDGE_KEY` in the environment, holds no credential, and
prints only a masked key prefix. No key is needed to reproduce any number in the
paper.

Two agreement definitions appear in §4.6 and they are not interchangeable, so
`h1c_selfcons_report.py` prints them side by side: *score agreement* (all three
decodes assign the same difficulty score — the 92.9% quoted against the 53.3%
block-order figure, which is the same definition) and *text agreement* (all three
decodes returned byte-identical output). Haiku 4.5 is 100% on both, which is why
§4.6 voids that arm: at a sampled temperature, identical output measures the
provider rather than the judge. Gemma-3-27B is 92.9% on scores and 67.1% on text,
the pattern expected of a model that actually samples.

## Human-eval ethics

The human evaluation involved three volunteer raters (legal-English teachers at
the authors' institution) who gave informed consent; no identifying data were
collected. Ratings are anonymized as R1/R2/R3 throughout. The blind set is
released so the judge and human scores can be compared on the identical texts.

## Data sources and licensing

The corpus is built from publicly available legal sources:

| Source | License |
|---|---|
| EUR-Lex | CC BY (metadata CC0) |
| Wex (Cornell) | CC BY-NC-SA |
| Oyez | CC BY-NC-SA |
| Cornell LII U.S. Code | public-domain text; LII markup copyrighted |
| Indian case law (external-domain evaluation) | public court judgments |

Because parts of the corpus are non-commercial share-alike (Wex, Oyez), the
derived dataset is released for **non-commercial research use** under a
compatible license, and users must observe each source's terms. The raw scraped
texts are not shipped; `scripts/01_build_corpus.py` and the `scripts/build_*`
tools show how they are assembled from the sources above.

## Weights

Trained LoRA adapters (~2 GB) and the embedding/NLI encoders are **not**
distributed in this repository. The adapters are released under an open-source
license at a location named in the paper's data-availability statement (to be
announced at publication). All open-weights models used are publicly available
under their own licenses: Qwen3.5-4B / Qwen3.5-9B / Qwen3-4B(-Instruct)
(Apache-2.0), DeBERTa-v3 NLI (MIT), MiniLM (Apache-2.0).

## Environment and paths

- Python 3.10+, PyTorch 2.8, Transformers 5.x, TRL 1.x, PEFT 0.20; training
  needs a GPU with ≥ 32 GB VRAM (originally run on an NVIDIA RTX 5090).
- FAISS for the retrieval index (`scripts/02_build_index.py`).
- The training/inference scripts were written against a Linux training root and
  still contain a placeholder base path `/path/to/legal_english`. Set the
  `DATA` / `MODELS` / `RESULTS` / `LOGS` constants at the top of each script (and
  the `cd` line in each `run_*.sh`) to your checkout. Corpus provenance and
  model weights are not bundled, so full retraining requires re-downloading the
  sources (see `scripts/01_build_corpus.py`) and the models.

## License

Code is MIT (see `LICENSE`). The human-eval data and derived dataset are
released for non-commercial research use as described above.
