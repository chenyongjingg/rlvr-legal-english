# -*- coding: utf-8 -*-
"""h1c_selfcons_report.py -- the K=3 same-prompt stability of each alternate judge.

Added 2026-09-12.  S4.6 prints two numbers that had a raw file but no producer:

    "Gemma-3-27B assigns the identical difficulty score on 92.9% of cells when
     the same prompt is decoded three times at temperature 0.6"

    "Claude Haiku 4.5 returned byte-identical output on 240 of 240 cells at
     temperature 0.6"

Both are computed from `h1c_out/selfcons_*.jsonl`, which stores three decodes per
cell, and neither had a script anywhere on the drive.  Without this, the pair of
numbers that carries S4.6's whole argument -- that order sensitivity is larger
than sampling noise -- is unverifiable, and so is the claim that voids the Haiku
run.  A raw file with no producer is indistinguishable from a made-up number.

Two distinct agreement definitions, kept apart because S4.6 uses both:

  score agreement   all three decodes assign the same difficulty score.  This is
                    the "identical difficulty score" of the manuscript, and it is
                    the number comparable to the block-order agreement that
                    h1c_e_swap_report.py computes from swaporder_*.jsonl.
  text agreement    all three decodes returned byte-identical output.  This is
                    what shows Haiku 4.5 ignored the sampling parameter: the
                    score agreement is 100%, but so is the text, so the reading
                    measures the provider rather than the judge.

The two are deliberately reported separately: 92.9% is score agreement, and the
same run's text agreement is much lower, which is the expected pattern for a
model that actually samples.

Inputs
  h1c_out/selfcons_<tag>.jsonl     three decodes per cell, all released

Output
  h1c_out/h1c_selfcons_report.json

Usage: python h1c_selfcons_report.py
"""
import glob
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "h1c_out")
DIMS = ("difficulty", "faithfulness", "terminology", "fluency")
K = 3


def load(path):
    out = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def analyse(recs):
    """Score agreement and text agreement over cells with K valid decodes."""
    n = 0
    same_score = dict((d, 0) for d in DIMS)
    same_text = 0
    per_dim_n = dict((d, 0) for d in DIMS)
    for r in recs:
        reps = [x for x in (r.get("results") or []) if x.get("judge")]
        if len(reps) != K:
            continue
        n += 1
        for d in DIMS:
            vals = [x["judge"].get(d) for x in reps]
            if any(v is None for v in vals):
                continue
            per_dim_n[d] += 1
            if len(set(vals)) == 1:
                same_score[d] += 1
        if len(set(x.get("raw", "") for x in reps)) == 1:
            same_text += 1
    return {
        "n_cells_with_%d_decodes" % K: n,
        "score_agreement_pct": dict(
            (d, 100.0 * same_score[d] / per_dim_n[d] if per_dim_n[d] else None)
            for d in DIMS),
        "n_score_basis": per_dim_n,
        "text_agreement_n": same_text,
        "text_agreement_pct": 100.0 * same_text / n if n else None,
    }


def main():
    paths = sorted(glob.glob(os.path.join(OUT, "selfcons_*.jsonl")))
    judges = []
    for p in paths:
        tag = os.path.basename(p)[len("selfcons_"):-len(".jsonl")]
        if "pilot" in tag or "selftest" in tag:
            # same exclusion rule as h1c_c_altjudge_report.py: a 12-cell prefix
            # and a known-answer control are not judges.
            continue
        judges.append((tag, analyse(load(p))))
    if not judges:
        sys.exit("no selfcons_*.jsonl in %s" % OUT)

    res = {}
    print("K = %d same-prompt decodes, temperature as stored per record" % K)
    print("%-14s %7s %11s %11s %11s" % ("judge", "cells", "diff agree", "text agree", "faith agree"))
    for tag, a in judges:
        res[tag] = a
        n = a["n_cells_with_%d_decodes" % K]
        print("%-14s %7d %10.1f%% %10.1f%% %10.1f%%"
              % (tag, n, a["score_agreement_pct"]["difficulty"],
                 a["text_agreement_pct"], a["score_agreement_pct"]["faithfulness"]))
        if a["text_agreement_pct"] == 100.0 and n:
            print("   NOTE: %d of %d cells returned byte-identical output at a "
                  "sampled temperature.  That measures the provider, not the "
                  "judge, and no self-consistency figure should be taken from it."
                  % (a["text_agreement_n"], n))

    os.makedirs(OUT, exist_ok=True)
    pth = os.path.join(OUT, "h1c_selfcons_report.json")
    with io.open(pth, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1))
    print("\nwritten %s" % pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
