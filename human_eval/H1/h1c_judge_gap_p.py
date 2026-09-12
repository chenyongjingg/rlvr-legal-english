# -*- coding: utf-8 -*-
"""h1c_judge_gap_p.py -- passage-level paired Wilcoxon for the M2-vs-G6v4
difficulty gap, per judge.

WHY THIS EXISTS.  The manuscript prints four p-values for the same comparison
("M2 above G6v4 on difficulty") in one sentence:

    +0.625 (Gemma-3-27B, p = .0085 over the 20 passages)
    +0.625 (Claude Haiku 4.5, p = .027)
    +0.50  (the paper's judge, p = .0087 on the same unit)
    +0.358 for the human panel (p = .0025)

The four *gaps* were backed by artifacts (h1c_c_altjudge_report.json's
M2_minus_G6_difficulty, and level_cluster.json for the human panel), but the
three judge *p-values* were backed by nothing: ".0087" appears in no file under
human_eval/H1/h1c_out/, and ".0085"/".027" appear only as unrelated *cell-level*
numbers inside h1c_tier1_extra.json (a per-system Spearman p, not this test).
They were produced ad hoc and survived because nothing recomputed them.

This script recomputes them from the released judge records and writes them to
h1c_out/, so the sentence is checked against a producer rather than against
itself.  It deliberately does NOT print a value the manuscript should adopt --
it prints what the data gives; if that disagrees with the manuscript, the
manuscript is what changes.

Unit.  Passage, not item: the 40 items are 20 passages x 2 levels, so the
per-item test treats within-passage replicates as independent.  Each system's
difficulty is collapsed to a per-passage mean (over its two levels) and the test
is a paired Wilcoxon over the 20 passages, matching the unit the manuscript
names in the sentence.

Inputs (all released)
  human40_judge.jsonl                      the paper's judge (Qwen3-4B-Instruct-2507)
  h1c_out/altjudge_gemma3-27b.jsonl        Gemma-3-27B
  h1c_out/altjudge_haiku45.jsonl           Claude Haiku 4.5
  h1c_out/altjudge_dsv4flash.jsonl         DeepSeek V4 Flash (low coverage)
Output: h1c_out/h1c_judge_gap_p.json
"""
import io
import json
import os
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "h1c_out")

JUDGES = [
    ("paper (Qwen3-4B-Instruct-2507)", os.path.join(BASE, "human40_judge.jsonl")),
    ("gemma3-27b", os.path.join(OUT, "altjudge_gemma3-27b.jsonl")),
    ("haiku45", os.path.join(OUT, "altjudge_haiku45.jsonl")),
    ("dsv4flash", os.path.join(OUT, "altjudge_dsv4flash.jsonl")),
]

# A judge must have this many parsed cells per system before its per-passage
# mean means anything.  dsv4flash parsed only 5/40 M2 cells; reporting a
# Wilcoxon p from that would be a number with no measurement behind it.
MIN_CELLS_PER_SYSTEM = 30

SYSTEMS = ("M2", "G6")


def load_scores(path):
    """{(passage_id, level, system): difficulty} for one judge.

    Two record shapes occur -- altjudge files nest the verdict under
    results[0].judge, the paper's judge file has it at judge -- and both are
    read here rather than editing either file to match the other.
    """
    out = {}
    if not os.path.isfile(path):
        return out
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        # mode "c" is the comparable mode (all six systems); the swap runs and
        # other modes are different experiments and must not be pooled in.
        if r.get("mode", "c") != "c" or r.get("swap"):
            continue
        j = r.get("judge")
        if j is None and isinstance(r.get("results"), list) and r["results"]:
            j = r["results"][0].get("judge")
        if not j or j.get("difficulty") is None:
            continue
        out[(r["id"], r["lv"], r["system"])] = float(j["difficulty"])
    return out


def per_passage(scores, system):
    """{passage_id: mean difficulty} over the levels a system was scored at."""
    byp = {}
    for (pid, lv, sysn), v in scores.items():
        if sysn == system:
            byp.setdefault(pid, []).append(v)
    return {p: float(np.mean(v)) for p, v in byp.items()}


def main():
    res = {}
    print("passage-level paired Wilcoxon, M2 vs G6v4, difficulty")
    print("%-32s %6s %7s %7s %9s  %s" % ("judge", "n_pass", "M2", "G6", "M2-G6", "p"))
    for name, path in JUDGES:
        sc = load_scores(path)
        if not sc:
            res[name] = {"status": "NOT-FOUND", "path": os.path.basename(path)}
            print("%-32s  file missing or unreadable" % name)
            continue
        cells = {s: sum(1 for k in sc if k[2] == s) for s in SYSTEMS}
        if min(cells.values()) < MIN_CELLS_PER_SYSTEM:
            res[name] = {"status": "INSUFFICIENT-COVERAGE", "cells": cells,
                         "min_required": MIN_CELLS_PER_SYSTEM}
            print("%-32s  coverage %s < %d -- no test run"
                  % (name, cells, MIN_CELLS_PER_SYSTEM))
            continue
        a, b = per_passage(sc, "M2"), per_passage(sc, "G6")
        pids = sorted(set(a) & set(b))
        m = np.array([a[p] for p in pids])
        g = np.array([b[p] for p in pids])
        d = m - g
        # scipy drops exact zeros; state how many so the n in the test is visible
        nz = int(np.sum(d != 0))
        if nz < 2:
            w_p = None
        else:
            w_p = float(stats.wilcoxon(m, g).pvalue)
        res[name] = {"status": "OK", "n_passages": len(pids),
                     "n_nonzero_diff": nz, "mean_M2": float(m.mean()),
                     "mean_G6": float(g.mean()), "delta": float(d.mean()),
                     "p": w_p, "unit": "passage",
                     "cells": cells}
        print("%-32s %6d %7.3f %7.3f %+9.3f  %s"
              % (name, len(pids), m.mean(), g.mean(), d.mean(),
                 "%.4g" % w_p if w_p is not None else "n/a"))

    os.makedirs(OUT, exist_ok=True)
    pth = os.path.join(OUT, "h1c_judge_gap_p.json")
    with io.open(pth, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\nwritten", pth)
    return res


if __name__ == "__main__":
    main()
