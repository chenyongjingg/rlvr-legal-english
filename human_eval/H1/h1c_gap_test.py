# -*- coding: utf-8 -*-
"""h1c_gap_test.py -- the M2 > G6v4 difficulty gap, per judge, with its test.

WHY THIS EXISTS.  §4.3 and §4.5 of the manuscript report the difficulty gap
between M2 and G6v4 for four judges (the paper's Qwen3-4B-Instruct-2507, Gemma-
3-27B, Claude Haiku 4.5, and the human panel) and attach a p-value to three of
them.  Until 2026-09-12 those p-values existed only in the prose: no stored
artifact produced them, so nothing could catch it if a later edit moved a number
or if the value had been mis-transcribed in the first place.  That is the same
failure class as the MIR "green light with no artifact" incident, and the fix is
the same -- a producer that recomputes the numbers the manuscript prints.

WHAT IT COMPUTES.  For each judge, the mean difficulty of M2 and of G6v4 over
the shared cells, the gap, and a paired Wilcoxon signed-rank on that gap at two
units:

  * item level  -- n = 40 cells (20 passages x 2 levels).  This is the unit most
    of the paper's judge-versus-human comparisons use, and it treats
    within-passage replicates as independent.
  * passage level -- n = 20, each passage's mean over its two levels.  This is
    the unit that does not assume independence, and it is the unit §4.3 uses for
    the human panel because the panel's own gap is reported there.

Both are printed, because the manuscript must name the unit it is quoting: an
"item-level test over 20 passages" is a contradiction, and one earlier draft of
§4.3 contained exactly that.

The gap value itself is identical under both units (the two levels are balanced
within each passage); only n and the p-value change.

A/B ARMS.  altjudge_<tag>.jsonl is the original block order, swaporder_<tag>.jsonl
the reversed one.  Both are reported for the alternate judges so §4.6's claim
that the gap narrows under reversal is backed by the same producer.

Output: h1c_out/h1c_gap_test.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import h1c_loo_basis as L              # noqa: E402

OUT = os.path.join(BASE, "h1c_out")
DIMS = ("difficulty", "faithfulness", "terminology", "fluency")
PAIR = ("M2", "G6")


def load_alt(path):
    """{(item_lv, system): difficulty} from an altjudge_/swaporder_ file."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        js = [x["judge"] for x in r["results"] if x.get("judge")]
        if not js:
            continue
        v = js[0].get("difficulty")
        if v is not None:
            out[(f'{r["id"]}_{r["lv"]}', r["system"])] = float(v)
    return out


def load_published_judge(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        j = r.get("judge")
        if j and j.get("difficulty") is not None:
            out[(f'{r["id"]}_{r["lv"]}', r["system"])] = float(j["difficulty"])
    return out


def load_human():
    """{(item_lv, system): panel-mean difficulty}."""
    rat = L.load()
    cells = set()
    for rn in L.RATERS:
        cells |= set(rat[rn])
    out = {}
    for c in cells:
        v = [rat[rn][c]["difficulty"] for rn in L.RATERS if c in rat[rn]]
        if v:
            out[c] = float(np.mean(v))
    return out


def passage(item_lv):
    """'eur_021_beginner' -> 'eur_021'.  Levels are the only suffix.

    Takes the item_lv KEY, not the (item_lv, system) tuple: the two callers
    below hold keys, and an earlier version indexed [0] here, which turned every
    key into its first character and collapsed all 20 passages into one group
    (n = 1, no test).  The printed n is what caught it.
    """
    return item_lv.rsplit("_", 1)[0]


def gap(values, unit):
    """Paired M2-vs-G6 gap + Wilcoxon on the cells both systems cover.

    `unit` is 'item' (each cell one observation) or 'passage' (each passage's
    mean over its levels).  Returns None when fewer than 2 pairs can be formed,
    because a signed-rank test on one pair is not defined and a p-value there
    would be noise dressed as evidence.
    """
    cells = sorted({c[0] for c in values if c[1] == PAIR[0]}
                   & {c[0] for c in values if c[1] == PAIR[1]})
    if unit == "item":
        pairs = [(c, values[(c, PAIR[0])], values[(c, PAIR[1])]) for c in cells]
    else:
        byp = {}
        for c in cells:
            byp.setdefault(passage(c), []).append(
                (values[(c, PAIR[0])], values[(c, PAIR[1])]))
        pairs = []
        for p, rows in sorted(byp.items()):
            pairs.append((p,
                          float(np.mean([r[0] for r in rows])),
                          float(np.mean([r[1] for r in rows]))))
    a = np.array([x[1] for x in pairs], dtype=float)
    b = np.array([x[2] for x in pairs], dtype=float)
    n = len(pairs)
    res = {"unit": unit, "n": n,
           "mean_M2": float(a.mean()), "mean_G6": float(b.mean()),
           "gap": float(a.mean() - b.mean()), "p": None, "n_nonzero": None}
    if n >= 2 and not np.allclose(a - b, 0.0):
        d = a - b
        res["n_nonzero"] = int((d != 0).sum())
        res["p"] = float(wilcoxon(a, b).pvalue)
    return res


def main():
    judges = {
        "Qwen3-4B-Instruct-2507 (paper's)": load_published_judge(
            os.path.join(BASE, "human40_judge.jsonl")),
        "human panel": load_human(),
    }
    for tag, label in (("gemma3-27b", "Gemma-3-27B"),
                       ("haiku45", "Claude Haiku 4.5")):
        judges[label + " (A: original order)"] = load_alt(
            os.path.join(OUT, "altjudge_%s.jsonl" % tag))
        judges[label + " (B: blocks reversed)"] = load_alt(
            os.path.join(OUT, "swaporder_%s.jsonl" % tag))

    report = {}
    for label, values in judges.items():
        if not values:
            print("%-34s NO DATA" % label)
            continue
        report[label] = {u: gap(values, u) for u in ("item", "passage")}
        for u in ("item", "passage"):
            g = report[label][u]
            if g is None or g["p"] is None:
                print("%-34s %-8s n=%s  gap unavailable" % (label, u, g["n"] if g else "?"))
                continue
            print("%-34s %-8s n=%2d  M2 %.3f  G6 %.3f  gap %+.3f  p=%.4g"
                  % (label, u, g["n"], g["mean_M2"], g["mean_G6"], g["gap"], g["p"]))

    print("\nREAD: the manuscript must quote the unit it names.  The GAP is the\n"
          "      same under both units; only n and p move.  A p-value labelled\n"
          "      'item-level ... over 20 passages' is quoting one unit's n with\n"
          "      the other's test.")
    with open(os.path.join(OUT, "h1c_gap_test.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print("\nwrote h1c_out/h1c_gap_test.json")


if __name__ == "__main__":
    main()
