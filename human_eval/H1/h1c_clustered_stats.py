# -*- coding: utf-8 -*-
"""h1c_clustered_stats.py -- the clustered / post-hoc analyses of S3.5 and S4.13.

Added 2026-09-12.  Several numbers printed in the manuscript had no released
producer: the passage-level cluster-bootstrap intervals of S3.5, the design
effect quoted beside them, the level split of S4.13, and the 20-passage
robustness check of the difficulty gap.  This script computes all of them from
the shipped sheets and writes h1c_out/h1c_clustered_stats.json.

Conventions (fixed here so the numbers are exactly reproducible):

  cluster    the source passage.  The 40 blind items are 20 passages rated at
             two levels, so the 200 generated cells form 20 clusters of 10.
  interval   percentile bootstrap over the resampled clusters, B = 2000,
             seed = 20260911 (the same seed as icc_reliability.py).
  raters     the human value of a cell is the mean of the three raters.
  auto       the *stored* difficulty sub-reward, `parts.diff` as released in the
             per-row JSONL.  Recomputing the FRE-band reward from `fre` gives
             the same quantity to within the 4-decimal rounding of the released
             files (Pearson 1.000000, max |delta| = 2e-4); rank correlations
             computed on the two can differ by ~0.01 through tie ordering, which
             is why the basis is pinned here rather than left to the reader.
  Source     excluded throughout (it is the untranslated negative control and
             has no generator configuration to compare).

Inputs (all released):
  human_eval/samples/unblind_key.json
  human_eval/samples/rated_R{1,2,3}.csv
  human_eval/H1/human40_judge.jsonl
  results/{B2,M2_honest,B3_flant5,B4_bart,G6v4}.jsonl

Output:
  h1c_out/h1c_clustered_stats.json
"""
import importlib.util
import json
import os

import numpy as np
from scipy import stats as sc

SEED = 20260911
BOOT_CI = 2000
BOOT_GAP = 800

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h1c_out")

_spec = importlib.util.spec_from_file_location(
    "h1c", os.path.join(HERE, "h1c_judge_human_analysis.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

PASSAGE = lambda ik: ik.rsplit("_", 1)[0]      # noqa: E731
LEVEL = lambda ik: ik.rsplit("_", 1)[-1]       # noqa: E731


def load_all():
    human = M.load_human()
    judge = M.load_judge(os.path.join(HERE, "human40_judge.jsonl"))
    auto = M.load_auto()
    return human, judge, auto


def pairs(human, judge, auto):
    """(passage, level, system, judge_diff, human_diff, auto_diff) per cell."""
    rows = []
    for (ik, sysn), r in judge.items():
        if sysn == "Source" or not r.get("judge"):
            continue
        rs = human.get((ik, sysn))
        if not rs or r["judge"]["difficulty"] is None:
            continue
        a = auto.get((sysn, ik))
        ad = (a or {}).get("parts", {}).get("diff") if a else None
        rows.append((PASSAGE(ik), LEVEL(ik), sysn, r["judge"]["difficulty"],
                     M.mean([x["difficulty"] for x in rs]), ad))
    return rows


def rho(x, y):
    x, y = list(x), list(y)
    if len(x) < 5 or len(set(x)) < 2 or len(set(y)) < 2:
        return None, None
    r, p = sc.spearmanr(x, y)
    return float(r), float(p)


def cluster_boot(rows, ia, ib, B=BOOT_CI, seed=SEED, want_p=False):
    """Percentile bootstrap resampling the 20 passages.  ia/ib index the two
    columns of each row to correlate."""
    groups = {}
    for i, t in enumerate(rows):
        groups.setdefault(t[0], []).append(i)
    keys = sorted(groups)
    x = np.array([t[ia] for t in rows], float)
    y = np.array([t[ib] for t in rows], float)
    rng = np.random.default_rng(seed)
    vals, ps = [], []
    for _ in range(B):
        ix = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        r, p = rho(x[ix], y[ix])
        if r is None:
            continue
        vals.append(r)
        if want_p:
            ps.append(p)
    vals = np.array(vals)
    out = {"n_resamples": int(len(vals)),
           "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
           "frac_ci_above_zero": float((vals > 0).mean()),
           "frac_ci_below_zero": float((vals < 0).mean())}
    if want_p:
        ps = np.array(ps)
        out["frac_p_lt_05"] = float((ps < .05).mean())
        out["median_p"] = float(np.median(ps))
    return out


def design_effect(rows):
    """ICC(1) of the human difficulty passage-mean over the 40 items, five
    systems per item; deff = 1 + (m-1)*ICC(1)."""
    by_item = {}
    for _, lv, _, _, hd, _ in rows:
        by_item.setdefault(lv, {})
    items = {}
    for pas, lv, sysn, _, hd, _ in rows:
        items.setdefault((pas, lv), []).append(hd)
    keys = sorted(items)
    groups_ = [np.array(items[k], float) for k in keys]
    m = min(len(g) for g in groups_)
    groups_ = [g[:m] for g in groups_]
    grand = np.concatenate(groups_).mean()
    msb = m * np.mean([(g.mean() - grand) ** 2 for g in groups_])
    msw = np.mean([((g - g.mean()) ** 2).sum() / (m - 1) for g in groups_])
    icc1 = (msb - msw) / (msb + (m - 1) * msw)
    return {"n_items": len(groups_), "m": m, "icc1": float(icc1),
            "deff": float(1 + (m - 1) * icc1)}


def level_split(rows):
    out = {}
    for lv in ("beginner", "intermediate"):
        sub = [t for t in rows if t[1] == lv and t[5] is not None]
        cell_r, cell_p = rho([t[5] for t in sub], [t[4] for t in sub])
        per = {}
        for t in sub:
            per.setdefault(t[0], []).append((t[5], t[4]))
        keys = sorted(per)
        pm_r, pm_p = rho([np.mean([v[0] for v in per[k]]) for k in keys],
                         [np.mean([v[1] for v in per[k]]) for k in keys])
        out[lv] = {"n_cells": len(sub), "cell_level": {"rho": cell_r, "p": cell_p},
                   "n_passages": len(keys),
                   "passage_mean": {"rho": pm_r, "p": pm_p}}
    return out


def gap(rows, system_hi="G6", system_lo="M2"):
    """The difficulty gap the two external lenses put between M2 and G6v4, at
    the item level and collapsed to the 20 passages, with the passage-level
    bootstrap of the collapsed test."""
    per = {}
    for t in rows:
        if t[2] in (system_hi, system_lo):
            per.setdefault((t[0], t[2]), []).append(t[3:5])   # (judge_diff, human_diff)
    keys = sorted({k[0] for k in per})
    out = {}
    for name, idx in (("human", 1), ("judge", 0)):
        hi = np.array([np.mean([v[idx] for v in per[(k, system_hi)]]) for k in keys])
        lo = np.array([np.mean([v[idx] for v in per[(k, system_lo)]]) for k in keys])
        cell_hi, cell_lo = [], []
        for t in rows:
            if t[2] == system_hi:
                cell_hi.append(t[3] if idx == 0 else t[4])
            elif t[2] == system_lo:
                cell_lo.append(t[3] if idx == 0 else t[4])
        item_p = sc.wilcoxon(cell_hi, cell_lo).pvalue if len(set(np.array(cell_hi) - np.array(cell_lo))) > 1 else None
        cl_p = sc.wilcoxon(hi, lo).pvalue
        rng = np.random.default_rng(SEED)
        ps = []
        for _ in range(BOOT_GAP):
            ix = rng.integers(0, len(keys), len(keys))
            if len(set(hi[ix] - lo[ix])) < 2:
                continue
            ps.append(sc.wilcoxon(hi[ix], lo[ix]).pvalue)
        ps = np.array(ps)
        out[name] = {"mean_hi": float(np.mean(
                            [t[3] if idx == 0 else t[4] for t in rows if t[2] == system_hi])),
                     "mean_lo": float(np.mean(
                            [t[3] if idx == 0 else t[4] for t in rows if t[2] == system_lo])),
                     "p_item_level": float(item_p),
                     "mean_hi_passage": float(hi.mean()), "mean_lo_passage": float(lo.mean()),
                     "p_passage_level": float(cl_p),
                     "bootstrap_passage_level": {"n_resamples": int(len(ps)),
                                                 "frac_p_lt_05": float((ps < .05).mean()),
                                                 "median_p": float(np.median(ps))}}
    return {"hi": system_hi, "lo": system_lo, "n_passages": len(keys), **out}


def main():
    human, judge, auto = load_all()
    rows = pairs(human, judge, auto)
    print("cells: %d   passages: %d" % (len(rows), len({t[0] for t in rows})))

    jh_r, jh_p = rho([t[3] for t in rows], [t[4] for t in rows])
    ha_rows = [t for t in rows if t[5] is not None]
    ha_r, ha_p = rho([t[5] for t in ha_rows], [t[4] for t in ha_rows])
    res = {
        "seed": SEED,
        "headline_item_level": {
            "judge_human_difficulty": {"rho": jh_r, "p": jh_p, "n": len(rows)},
            "human_auto_difficulty": {"rho": ha_r, "p": ha_p, "n": len(ha_rows)}},
        "headline_cluster_bootstrap": {
            "judge_human_difficulty": cluster_boot(rows, 3, 4),
            "human_auto_difficulty": cluster_boot(ha_rows, 5, 4)},
        "design_effect": design_effect(rows),
        "level_split_auto_vs_human": level_split(rows),
        "difficulty_gap": gap(rows),
    }
    print("\nitem level: judge<->human difficulty %+.3f (p=%.4g) | human<->auto %+.3f (p=%.3f)"
          % (jh_r, jh_p, ha_r, ha_p))
    for k, v in res["headline_cluster_bootstrap"].items():
        print("  cluster bootstrap %-26s CI95 [%+.2f, %+.2f]  (%.1f%% of resamples above zero)"
              % (k, v["ci95"][0], v["ci95"][1], 100 * v["frac_ci_above_zero"]))
    print("design effect: ICC(1) = %+.4f over %d items -> deff = %.4f"
          % (res["design_effect"]["icc1"], res["design_effect"]["n_items"],
             res["design_effect"]["deff"]))
    print("\nlevel split (auto difficulty vs human difficulty):")
    for lv, v in res["level_split_auto_vs_human"].items():
        print("  %-13s cell n=%3d rho %+.4f (p=%.4f) | passage mean n=%d rho %+.4f (p=%.4f)"
              % (lv, v["n_cells"], v["cell_level"]["rho"], v["cell_level"]["p"],
                 v["n_passages"], v["passage_mean"]["rho"], v["passage_mean"]["p"]))
    print("\ndifficulty gap %s vs %s:" % (res["difficulty_gap"]["hi"], res["difficulty_gap"]["lo"]))
    for lens in ("human", "judge"):
        v = res["difficulty_gap"][lens]
        print("  %-6s item %.2f vs %.2f p=%.4f | passage %.2f vs %.2f p=%.4f | "
              "bootstrap %.1f%% of %d significant, median p=%.4f"
              % (lens, v["mean_hi"], v["mean_lo"], v["p_item_level"],
                 v["mean_hi_passage"], v["mean_lo_passage"], v["p_passage_level"],
                 100 * v["bootstrap_passage_level"]["frac_p_lt_05"],
                 v["bootstrap_passage_level"]["n_resamples"],
                 v["bootstrap_passage_level"]["median_p"]))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "h1c_clustered_stats.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", p)


if __name__ == "__main__":
    main()
