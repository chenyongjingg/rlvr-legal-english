# -*- coding: utf-8 -*-
"""h1c_e_swap_report.py -- item E, the block-order control, paired per cell.

The paper cites MT-Bench [45] for position bias and then runs no control of its
own.  The producer's prompt puts ORIGINAL LEGAL TEXT before SIMPLIFIED REWRITE;
E re-runs the identical prompt with the two blocks exchanged and pairs the two
readings cell by cell.

Both arms are GREEDY (temperature 0), so the only difference between them is
the order of the two blocks.  That is the whole design: if the arm also sampled,
a score change would be order plus sampling noise and the item would prove
nothing.

Why this matters for C2 specifically.  Item C found that the paper's Qwen judge
scores the untouched `Source` statute joint-HIGHEST on difficulty while humans
score it lowest, and that two different-vendor judges do not.  If reversing the
blocks moves `Source` difficulty upward, then part of that anomaly is a POSITION
effect rather than a property of difficulty judgement -- which is testable by
the author on the real judge in one run, and explains the anomaly without
appealing to the judge's incompetence.

Caveat that travels with every number here: these are the ALTERNATE judges, not
the paper's Qwen3-4B-Instruct-2507.  That checkpoint is not served by this
endpoint and is not on this machine (only LoRA adapters are), so the paper's own
judge's order sensitivity remains UNMEASURED.  These results bound the paradigm;
they do not report the paper's instrument.

Output: h1c_out/h1c_e_swap_report.json
"""
import glob
import json
import os
import sys

from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1c_loo_basis as L              # noqa: E402
import h1c_c_altjudge_report as C       # noqa: E402  (reuse load_alt)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "h1c_out")
DIMS = ("difficulty", "faithfulness", "terminology", "fluency")
SIX = ["Source", "B2", "M2", "FlanT5", "BART", "G6"]


def load(path):
    if not os.path.isfile(path):
        return None
    return C.load_alt(path)     # {(item_lv, system): {dim: val}}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def main():
    rat = L.load()
    RATERS = L.RATERS

    def H(c, d):
        vs = [rat[rn][c][d] for rn in RATERS if c in rat[rn]]
        return sum(vs) / len(vs) if vs else None

    tags = sorted(os.path.basename(p)[len("altjudge_"):-len(".jsonl")]
                  for p in glob.glob(os.path.join(OUT, "altjudge_*.jsonl"))
                  if "pilot" not in p and "selftest" not in p)
    res = {}
    print("order-A (original first) files: %s" % ", ".join(tags))

    for tag in tags:
        a = load(os.path.join(OUT, "altjudge_%s.jsonl" % tag))
        b = load(os.path.join(OUT, "swaporder_%s.jsonl" % tag))
        if not a or not b:
            print("\n%s: no swapped arm, skipped" % tag)
            continue
        keys = [k for k in a if k in b]
        print("\n" + "=" * 84)
        print("E.  %s -- %d paired cells, both arms greedy" % (tag, len(keys)))
        print("=" * 84)

        per = {}
        print("%-14s %9s %9s %9s %9s %9s"
              % ("dimension", "mean A", "mean B", "mean d", "exact%", "rho(A,B)"))
        for d in DIMS:
            xa = [a[k].get(d) for k in keys]
            xb = [b[k].get(d) for k in keys]
            pts = [(u, v) for u, v in zip(xa, xb)
                   if u is not None and v is not None]
            if len(pts) < 5:
                continue
            u = [p[0] for p in pts]
            v = [p[1] for p in pts]
            dv = [p[1] - p[0] for p in pts]
            exact = sum(1 for p in pts if p[0] == p[1])
            rho = (float(spearmanr(u, v)[0])
                   if len(set(u)) > 1 and len(set(v)) > 1 else None)
            print("%-14s %9.2f %9.2f %+9.2f %8.1f%% %9s"
                  % (d, mean(u), mean(v), mean(dv), 100.0 * exact / len(pts),
                     ("%+.3f" % rho) if rho is not None else "n/a"))
            per[d] = {"mean_A": mean(u), "mean_B": mean(v), "mean_delta": mean(dv),
                      "exact_agree_pct": 100.0 * exact / len(pts),
                      "rho_between_orders": rho, "n": len(pts)}

        # the two claims E can move: where Source sits, and the M2-G6 gap
        print("\n   Source difficulty : A %.2f  ->  B %.2f   (%+.2f)"
              % (mean([a[k]["difficulty"] for k in keys if k[1] == "Source"]),
                 mean([b[k]["difficulty"] for k in keys if k[1] == "Source"]),
                 (mean([b[k]["difficulty"] for k in keys if k[1] == "Source"])
                  - mean([a[k]["difficulty"] for k in keys
                          if k[1] == "Source"]))))
        src_a = mean([a[k]["difficulty"] for k in keys if k[1] == "Source"])
        all_a = {s: mean([a[k]["difficulty"] for k in keys if k[1] == s])
                 for s in SIX}
        all_b = {s: mean([b[k]["difficulty"] for k in keys if k[1] == s])
                 for s in SIX}
        rk_a = 1 + sum(1 for s in SIX if all_a[s] is not None
                       and all_a[s] > src_a)
        src_b = mean([b[k]["difficulty"] for k in keys if k[1] == "Source"])
        rk_b = 1 + sum(1 for s in SIX if all_b[s] is not None
                       and all_b[s] > src_b)
        print("   Source rank (1=highest): A %d  ->  B %d" % (rk_a, rk_b))
        print("   M2 - G6 gap        : A %+.2f  ->  B %+.2f"
              % (all_a["M2"] - all_a["G6"], all_b["M2"] - all_b["G6"]))

        # human reference row for the same two quantities
        src_h = mean([H(c, "difficulty") for c in rat[RATERS[0]]
                      if c[1] in L.SOURCE_LABELS])
        print("   (humans: Source difficulty %.2f, rank 6, M2-G6 %+.2f)"
              % (src_h, mean([H(c, "difficulty") for c in rat[RATERS[0]]
                              if c[1] == "M2"])
                 - mean([H(c, "difficulty") for c in rat[RATERS[0]]
                         if c[1] == "G6"])))
        res[tag] = {"n_paired": len(keys), "per_dimension": per,
                    "source_difficulty_A": src_a, "source_difficulty_B": src_b,
                    "source_rank_A": rk_a, "source_rank_B": rk_b,
                    "M2_minus_G6_A": all_a["M2"] - all_a["G6"],
                    "M2_minus_G6_B": all_b["M2"] - all_b["G6"],
                    "human_source_difficulty": src_h}

    p = os.path.join(OUT, "h1c_e_swap_report.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("\nwritten %s" % p)
    print("\nNOTE: these are alternate judges.  The paper's own "
          "Qwen3-4B-Instruct-2507 is\n      neither served by the endpoint nor "
          "present on this machine, so its order\n      sensitivity remains "
          "unmeasured.")


if __name__ == "__main__":
    main()
