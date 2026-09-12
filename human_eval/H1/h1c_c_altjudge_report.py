# -*- coding: utf-8 -*-
"""h1c_c_altjudge_report.py -- item C, the alternate-judge join.

C asks a single question with two opposite failure modes, and both answers are
useful:

  * if a DIFFERENT-VENDOR judge reproduces the `Source` anomaly (untouched
    statute scored joint-HIGHEST on difficulty while humans score it LOWEST),
    then the anomaly is a property of the LLM-as-judge paradigm and the paper
    must stop resting any difficulty claim on this dimension;
  * if it does NOT reproduce, then the anomaly is a property of the paper's own
    Qwen3-4B-Instruct-2507 checkpoint -- which is worse for the judge's standing,
    not better, because the paper's generators ARE Qwen-family, so a
    family-specific scoring quirk is exactly the confound the DA alleges.

Either way the C2 minimum remedy (demote judge difficulty to corroborating) is
what the evidence supports.  This script reports WHICH of the two it is, and
separately checks whether the ORDERING claim (M2 above G6 on difficulty)
survives under the alternate judges -- that claim can be robust even when the
anomaly is not.

Comparability is by construction: every alternate judge was run by
h1c_altjudge_api.py, whose system prompt and rendered user prompt were
AST-verified byte-identical to github_repo/scripts/09_judge_human_items.py
(same 3000-char truncation, greedy first, sampled retry).

Output: h1c_out/h1c_c_altjudge_report.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1c_loo_basis as L              # noqa: E402
import h1c_c2_c15_checks as X          # noqa: E402  (reuse load_auto / zero_order)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "h1c_out")
DIMS = ("difficulty", "faithfulness", "terminology", "fluency")
SIX = ["Source", "B2", "M2", "FlanT5", "BART", "G6"]


def load_alt(path):
    """{(item_lv, system): {dim: value}} from an altjudge_*.jsonl."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        js = [x["judge"] for x in r["results"] if x.get("judge")]
        if not js:
            continue
        j = js[0]
        out[(f'{r["id"]}_{r["lv"]}', r["system"])] = {
            d: j.get(d) for d in DIMS}
    return out


def sysmeans(judge):
    m = {}
    for s in SIX:
        vs = [v["difficulty"] for (ik, sy), v in judge.items()
              if sy == s and v.get("difficulty") is not None]
        m[s] = (sum(vs) / len(vs)) if vs else None
    return m


def syscounts(judge):
    return {s: sum(1 for (ik, sy), v in judge.items()
                   if sy == s and v.get("difficulty") is not None) for s in SIX}


# 2026-09-12: a rank is computed ACROSS the six systems, so a partial judge can
# only be ranked if every system's mean is a mean.  DeepSeek-V4-Flash parsed 98
# of 240 cells unevenly (Source 39, B2 35, but M2 5, G6v4 15, Flan-T5 2, BART 2),
# and this script printed "source rk 6" from three means resting on two to five
# cells.  The manuscript inherited that rank.  Below half the cells a mean is a
# subsample of unknown direction, so no rank is reported and the caller says so.
MIN_RANK_COVERAGE = 20


def main():
    rat = L.load()
    RATERS = L.RATERS
    paper = X.load_judge(os.path.join(BASE, "human40_judge.jsonl"))

    # X.load_judge yields the whole released row; load_alt yields the inner
    # dims dict.  Unwrap here so every judge below has ONE shape -- otherwise
    # v.get(dim) is None for all 240 paper rows and the paper's own judge
    # silently drops out of every table instead of raising.
    paper = {k: v["judge"] for k, v in paper.items()}

    judges = [("Qwen3-4B-Instruct-2507 (paper's)", paper)]
    for p in sorted(glob.glob(os.path.join(OUT, "altjudge_*.jsonl"))):
        tag = os.path.basename(p)[len("altjudge_"):-len(".jsonl")]
        if "pilot" in tag or "selftest" in tag:
            # "pilot" is a 12-cell prefix of a full run; "selftest" is
            # h1c_swap_truejudge.py's known-answer control, whose two arms are
            # identical by construction.  Neither is a judge -- reporting either
            # as one would put a fabricated row in the table.
            continue
        judges.append((tag, load_alt(p)))
    print("judges found: %s" % ", ".join(n for n, _ in judges))

    # the 200 generated cells on the judge basis: human & judge both present.
    # `allc` keeps the Source cells too -- the Source row is the whole point of
    # C, and it is excluded from `cells` by construction.
    allc = sorted(rat[RATERS[0]])
    cells = sorted({c for c in allc if c[1] not in L.SOURCE_LABELS})

    def H(c, d):
        vs = [rat[rn][c][d] for rn in RATERS if c in rat[rn]]
        return sum(vs) / len(vs) if vs else None

    def J(judge, c, d):
        v = judge.get(c)
        return v.get(d) if v else None

    res = {"judges": {}, "source_cell": {}, "ordering": {}}

    print("\n" + "=" * 88)
    print("C.1  per-system difficulty means, all judges on the SAME 40 items/system")
    print("=" * 88)
    print("%-34s" % "judge" + "".join("%9s" % s for s in SIX) + "%9s" % "src rk")
    hmeans = {s: (lambda v: sum(v) / len(v) if v else None)(
        [H(c, "difficulty") for c in allc if c[1] == s]) for s in SIX}
    print("%-34s" % "HUMAN panel (for reference)"
          + "".join("%9s" % ("%.2f" % hmeans[s] if hmeans[s] is not None
                             else "--") for s in SIX)
          + "%9s" % ("%d" % (1 + sum(1 for s in SIX if hmeans[s] is not None
                                     and hmeans[s] > hmeans["Source"]))
                     if hmeans["Source"] is not None else "--"))
    for name, j in judges:
        m = sysmeans(j)
        cnt = syscounts(j)
        have = [s for s in SIX if m[s] is not None]
        if not have:
            continue
        # rank 1 = highest difficulty.  Withheld when any system's mean rests on
        # fewer than MIN_RANK_COVERAGE cells -- see the note on that constant.
        thin = sorted(s for s in have if cnt[s] < MIN_RANK_COVERAGE)
        rk = None
        if m["Source"] is not None and not thin:
            rk = 1 + sum(1 for s in have if m[s] > m["Source"])
        print("%-34s" % name
              + "".join("%9s" % ("%.2f" % m[s] if m[s] is not None else "--")
                        for s in SIX)
              + "%9s" % (rk if rk is not None
                         else ("n/a" if thin else "--")))
        if thin:
            print("%-34s   rank withheld: %s parsed %s of 40"
                  % ("", ", ".join(thin),
                     ", ".join("%s %d" % (s, cnt[s]) for s in thin)))
        res["source_cell"][name] = {
            "source_difficulty": m["Source"], "source_rank_1_is_highest": rk,
            "n_parsed_per_system": cnt,
            "rank_withheld_below_coverage": thin,
            "per_system_difficulty": m}

    print("\n" + "=" * 88)
    print("C.2  does each judge track HUMAN difficulty?  (200 generated cells)")
    print("=" * 88)
    print("%-34s %-24s %-24s %s"
          % ("judge", "pooled rho", "within-system mean", "M2-G6 gap"))
    for name, j in judges:
        xs, ys = [], []
        for c in cells:
            a, b = J(j, c, "difficulty"), H(c, "difficulty")
            if a is not None and b is not None:
                xs.append(a)
                ys.append(b)
        pooled = X.zero_order(xs, ys) if len(xs) > 5 else None
        per = []
        for s in ["B2", "M2", "FlanT5", "BART", "G6"]:
            cc = [c for c in cells if c[1] == s]
            a = [J(j, c, "difficulty") for c in cc]
            b = [H(c, "difficulty") for c in cc]
            pts = [(u, v) for u, v in zip(a, b) if u is not None and v is not None]
            if len(pts) >= 5 and len(set(p[0] for p in pts)) > 1:
                per.append(float(spearmanr([p[0] for p in pts],
                                           [p[1] for p in pts])[0]))
        wm = sum(per) / len(per) if per else None
        m = sysmeans(j)
        gap = (m["M2"] - m["G6"]) if (m["M2"] is not None
                                      and m["G6"] is not None) else None
        print("%-34s %-24s %-24s %s"
              % (name,
                 ("%+.3f p=%.3g" % (pooled[0], pooled[1])) if pooled else "n/a",
                 ("%+.3f (n=%d)" % (wm, len(per))) if wm is not None else "n/a",
                 "%+.2f" % gap if gap is not None else "n/a"))
        res["judges"][name] = {"pooled_rho_vs_human_difficulty": pooled,
                               "within_system_mean": wm, "n_systems": len(per),
                               "M2_minus_G6_difficulty": gap}

    print("\n" + "=" * 88)
    print("C.3  the halo symptom, stated as the DA states it")
    print("=" * 88)
    print("   On the `Source` cell the rewrite IS the original statute, so a judge")
    print("   reading DIFFICULTY should score it near the bottom (humans: %.2f)."
          % hmeans["Source"])
    print("   A judge reading overall QUALITY scores it high instead -- it is clean,")
    print("   fluent, and factually perfect.")
    print("\n%-34s %11s %11s %11s %11s"
          % ("judge", "difficulty", "faithful", "fluency", "overall"))
    for name, j in [("HUMAN panel", None)] + judges:
        if j is None:
            # humans rate the Source cell under its own SOURCE_LABELS name
            row = {}
            for d in list(DIMS) + ["overall"]:
                vs = [H(c, d) for c in rat[RATERS[0]]
                      if c[1] in L.SOURCE_LABELS]
                row[d] = sum(vs) / len(vs) if vs else None
        else:
            vs_by = {d: [] for d in list(DIMS) + ["overall"]}
            for (ik, sy), v in j.items():
                if sy != "Source":
                    continue
                for d in DIMS:
                    if v.get(d) is not None:
                        vs_by[d].append(v[d])
                ds = [v.get(d) for d in DIMS if v.get(d) is not None]
                if ds:
                    vs_by["overall"].append(sum(ds) / len(ds))
            row = {d: (sum(vs_by[d]) / len(vs_by[d])) if vs_by[d] else None
                   for d in vs_by}
        print("%-34s" % name
              + "".join("%11s" % ("%.2f" % row[d] if row.get(d) is not None
                                  else "--")
                        for d in ["difficulty", "faithfulness", "fluency",
                                  "overall"]))
        res["source_cell"].setdefault(name, {})["source_dimensions"] = row

    # ordering: does M2 sit above G6 in the human panel too?
    res["ordering"] = {
        "human_M2": hmeans["M2"], "human_G6": hmeans["G6"],
        "human_gap": (hmeans["M2"] - hmeans["G6"])}

    p = os.path.join(OUT, "h1c_c_altjudge_report.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("\nwritten %s" % p)


if __name__ == "__main__":
    main()
