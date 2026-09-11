# `results/` — provenance notes

Every artifact the paper's tables read is here. Three points need stating
explicitly, because a reader who recomputes without knowing them will get
different numbers and conclude the paper is wrong.

## 1. Two `verifier_eval.json` files — Table 6 uses the first

| File | md5 | What it is |
|---|---|---|
| `verifier_eval.json` | `7fe4661601c5307a73860b45e1ad8f86` | The **dev-probe** run. This is the source of Table 6 (ECE 0.25–0.49, ρ 0.40–0.58, self-consistency 0.90–1.00): 7 tags, including the G1 / B1 / ADV / L3_max3 / A_full / G6_pure_s11 / OOD_doc rows the table quotes. |
| `verifier_eval_rerun.json` | `b82c5524355f6f2ddfd2b7736da69ae0` | A **later re-run** over 12 tags with different values. It is released for completeness. **It does not reproduce Table 6.** |

If you recompute Table 6 against `verifier_eval_rerun.json` you will not match the
paper. Use `verifier_eval.json`. The same dev-probe run is recoverable from the
training log line quoted in the manuscript.

## 2. `G6v4.jsonl` is the raw variant; `G6v4_honest.jsonl` is the scored one

`G6v4.jsonl` is the framework's output **before** the honest scoring rule is
applied. Its two pure-commentary rows (empty rewrites) carry their raw totals
(0.8984 / 0.8876) and the file mean is 0.8525.

`G6v4_honest.jsonl` applies the paper's honest rule (App. A): a row whose cleaned
text is empty has every reward component set to 0. That is what the paper reports,
and it is what `honest_table.json` stores under `tags.G6v4`:

```
n = 150   pure = 2   valid_n = 148   tot = 0.840604   fre_mean = 38.9706
```

The derivation is validated: the honest file's mean total reward and pure-commentary
count reproduce the stored `honest_table.json` entry exactly, and the file is
written only if they match.

Note the asymmetry with the other lenses: `M2_honest.jsonl` **is** the honest
variant, so Table 11's M2 column and its G6v4 column are on the same basis only if
`G6v4_honest.jsonl` is used.

## 3. Statistical conventions the tables follow

- **Honest rule.** Pure-commentary rows score 0 on every reward component and count
  in the denominator; FRE means are taken over valid (non-pure) rows only. So
  `tot` and `fre_mean` in `honest_table.json` have different denominators, and
  `tot_valid` is the valid-rows-only figure.
- **`fre_mean` denominators.** For a tag, `fre_mean` is over `valid_n` rows, not
  `n`. For G6v4's beginner level that is 48 of 50 rows.
- **Effect sizes.** Wilcoxon effect sizes are matched-pairs rank-biserial
  correlations computed from the rank sums, not derived from the p-value.
- **Small samples.** The system-level rank correlations in §4.13 are over five
  generated systems; at n = 5 only a perfect rank correlation reaches p < .05
  exactly, so those are reported as orderings.
- **What is *not* here.** Pre-clean text was not retained for the sequence-to-sequence
  baselines (B3/B4), so the meta-commentary rate and ΔFRE of §4.12.1 are not
  reproducible from this repository — by construction, not by omission. The paper
  states this in §5.3.

## 4. Contents

| Group | Files |
|---|---|
| Honest main chain | `G1_honest.jsonl` … `G5_honest.jsonl` |
| Generator variants | `G6_pure_honest.jsonl`, `G6_pure_s{11,21,42}_honest.jsonl` |
| Three-level framework | `G6v4.jsonl`, `G6v4_honest.jsonl` |
| Loop depth | `L1_max1.jsonl`, `L2_max2.jsonl`, `L3_max3.jsonl`, `L5_max5.jsonl` |
| Reward ablations | `A_{full,diff,term,copy,faith,fmt}_honest.jsonl`, `A_{full,diff}_full_budget_honest.jsonl` |
| OOD | `OOD_{doc,mismatch,size,size50,size10,external}_honest.jsonl` |
| Baselines | `B2.jsonl`, `B3_flant5.jsonl`, `B4_bart.jsonl`, `M2_honest.jsonl` |
| Aggregates | `honest_table.json`, `metrics_honest.json`, `h1a_mainset.json`, `verifier_eval.json` |
| Annotation | `examples_qualitative.json` |
