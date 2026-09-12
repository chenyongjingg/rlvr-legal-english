# -*- coding: utf-8 -*-
"""h1c_loo_sb.py -- the target-size asymmetry between the judge and the
leave-one-out rater comparator, quantified instead of asserted.

WHY THIS EXISTS
§4.13 compares two correlations that do not have the same target:

    judge  <-> three-rater mean          (predictor = 1 model,  target = mean of 3)
    rater  <-> mean of the other two     (predictor = 1 rater,  target = mean of 2)

A target built from more raters is a less noisy estimate of the panel's central
tendency, so it is EASIER to predict.  The manuscript states this direction
("the judge has the easier target in that contrast") but never quantifies it, so
a referee cannot tell whether the advantage is 2% or 50%.

Under classical test theory the attenuation of a correlation by measurement
error is multiplicative in sqrt(reliability), so

    rho(X, M_3) = rho(X, M_2) * sqrt(rel_3 / rel_2)

with rel_k the reliability of a k-rater mean.  Spearman-Brown gives
rel_k = k*rbar / (1 + (k-1)*rbar), hence the correction factor

    sqrt( rel_3 / rel_2 ) = sqrt( 3*(1+rbar) / (2*(1+2*rbar)) )

which is ~1.07-1.17 for the inter-rater correlations actually observed here --
NOT the 1.5 that would follow from a naive "1 vs 3 raters" reading, because the
2-rater target is itself already substantially denoised.

rbar is taken from the data rather than assumed: ICC(3,3) is inverted for the
single-rater consistency ICC via  rbar = ICC(3,3) / (3 - 2*ICC(3,3)).

REPRODUCTION CHECKS (all must pass or the numbers below are not usable)
  1. ICC(2,3) and ICC(3,3) on 200 cells -> 0.26-0.61 and 0.30-0.71   (§5.3)
  2. ICC(2,3) on 240 cells              -> 0.10-0.32                 (§5.3)
  3. leave-one-out, 240 cells           -> 0.07/0.24/0.29/0.10/0.18 (§4.13)
  4. judge <-> panel mean, 200 cells    -> 0.536/0.599/0.532/0.699/0.685 (Table 12)

Output: h1c_out/h1c_loo_sb.json
"""
import json
import os

from scipy.stats import spearmanr

import h1c_loo_basis as LB

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1c_out")
RATERS = LB.RATERS
DIMS = LB.DIMS

# h1c_judge_human_analysis.py keys the judge by f'{id}_{lv}' and the humans by
# the CSV item_id, which is the same composite string.
JUDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "human40_judge.jsonl")
JDIM = ("difficulty", "faithfulness", "terminology", "fluency")


def load_judge():
    out = {}
    for line in open(JUDGE, encoding="utf-8"):
        r = json.loads(line)
        if r.get("judge"):
            out[(r["id"] + "_" + r["lv"], r["system"])] = r["judge"]
    return out


def icc(cells, rat, dim):
    """(ICC(2,3), ICC(3,3)) for n subjects x 3 raters, average measures."""
    n, k = len(cells), len(RATERS)
    x = [[rat[rn][c][dim] for rn in RATERS] for c in cells]
    gm = sum(sum(row) for row in x) / (n * k)
    rowm = [sum(row) / k for row in x]
    colm = [sum(r[i] for r in x) / n for i in range(k)]
    msr = k * sum((r - gm) ** 2 for r in rowm) / (n - 1)
    msc = n * sum((c - gm) ** 2 for c in colm) / (k - 1)
    mse = sum((x[i][j] - rowm[i] - colm[j] + gm) ** 2
              for i in range(n) for j in range(k)) / ((n - 1) * (k - 1))
    icc31 = (msr - mse) / msr
    # Shrout & Fleiss ICC(2,k) AVERAGE-measures form, matching
    # icc_reliability.py:95.  The (k-1)*MSE term belongs to the SINGLE-measure
    # ICC(2,1) denominator and does not appear here; mixing the two (as this
    # script first did) understates the index by ~2x and fails reproduction.
    icc23 = (msr - mse) / (msr + (msc - mse) / n)
    return icc23, icc31


def judge_panel(rat, judge, cells, dim):
    """pooled Spearman of judge score vs three-rater mean, Table 12 basis."""
    xs, ys = [], []
    for (ik, sysn) in cells:
        j = judge.get(((ik, sysn)))
        if j is None:
            continue
        jv = (sum(j[d] for d in JDIM) / 4.0) if dim == "overall" else j[dim]
        hv = sum(rat[rn][(ik, sysn)][dim] for rn in RATERS) / 3.0
        xs.append(jv)
        ys.append(hv)
    return float(spearmanr(xs, ys)[0]), len(xs)


def main():
    rat = LB.load()
    judge = load_judge()
    all_cells = sorted(set(rat["R1"]) & set(rat["R2"]) & set(rat["R3"]))
    gen_cells = [c for c in all_cells if c[1] not in LB.SOURCE_LABELS]

    fails = []

    print("=" * 78)
    print("CHECK 1/2/3  reproduction against published values")
    print("=" * 78)
    icc200 = {d: icc(gen_cells, rat, d) for d in DIMS}
    icc240 = {d: icc(all_cells, rat, d) for d in DIMS}
    r23 = [round(icc200[d][0], 2) for d in DIMS]
    r33 = [round(icc200[d][1], 2) for d in DIMS]
    r240 = [round(icc240[d][0], 2) for d in DIMS]
    for label, got, lo, hi in (
            ("ICC(2,3) 200 cells", r23, 0.26, 0.61),
            ("ICC(3,3) 200 cells", r33, 0.30, 0.71),
            ("ICC(2,3) 240 cells", r240, 0.10, 0.32)):
        ok = (abs(min(got) - lo) <= 0.005 and abs(max(got) - hi) <= 0.005)
        print("  %-18s %.2f-%.2f published %.2f-%.2f  %s"
              % (label, min(got), max(got), lo, hi,
                 "ok" if ok else "*** DOES NOT MATCH ***"))
        if not ok:
            fails.append(label)
    print("  per dimension 200-cell: " + "  ".join(
        "%s %.2f/%.2f" % (d[:4], icc200[d][0], icc200[d][1]) for d in DIMS))

    pub_loo = [0.07, 0.24, 0.29, 0.10, 0.18]
    loo240 = [round(LB.loo(rat, all_cells, d)[1], 2) for d in DIMS]
    ok = loo240 == pub_loo
    print("  leave-one-out 240 cells: got %s vs published %s  %s"
          % (loo240, pub_loo, "ok" if ok else "*** DOES NOT MATCH ***"))
    if not ok:
        fails.append("loo240")

    print("\n" + "=" * 78)
    print("JUDGE <-> PANEL  (reproduction against Table 12)")
    print("=" * 78)
    pub_j = {"difficulty": 0.536, "faithfulness": 0.599, "terminology": 0.532,
             "fluency": 0.699, "overall": 0.685}
    jr = {}
    for d in DIMS:
        jr[d] = judge_panel(rat, judge, gen_cells, d)
        print("  %-14s %.3f  (n=%d)   Table 12 %.3f  %s"
              % (d, jr[d][0], jr[d][1], pub_j[d],
                 "ok" if abs(jr[d][0] - pub_j[d]) < 0.0015 else "*** MISMATCH ***"))
        if abs(jr[d][0] - pub_j[d]) >= 0.0015:
            fails.append("judge:" + d)

    print("\n" + "=" * 78)
    print("THE ASYMMETRY, QUANTIFIED")
    print("=" * 78)
    out = {"n_generated": len(gen_cells), "dims": {}, "checks_failed": fails}
    print("%-13s %6s %6s %6s %7s %8s %8s  %s" %
          ("dim", "judge", "LOO", "ICC33", "rbar", "factor", "LOO_sb", "verdict"))
    for d in DIMS:
        loo200 = LB.loo(rat, gen_cells, d)[1]
        i33 = icc200[d][1]
        rbar = i33 / (3.0 - 2.0 * i33)
        rel2 = 2.0 * rbar / (1.0 + rbar)
        rel3 = 3.0 * rbar / (1.0 + 2.0 * rbar)
        factor = (rel3 / rel2) ** 0.5
        loo_sb = loo200 * factor
        j = jr[d][0]
        verdict = "judge leads" if j >= loo_sb else "panel leads"
        out["dims"][d] = {"judge": j, "loo200": loo200, "icc33": i33,
                          "rbar": rbar, "rel2": rel2, "rel3": rel3,
                          "sb_factor": factor, "loo200_sb": loo_sb,
                          "verdict": verdict}
        print("%-13s %6.3f %6.2f %6.2f %7.3f %8.3f %8.3f  %s"
              % (d, j, loo200, i33, rbar, factor, loo_sb, verdict))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "h1c_loo_sb.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwritten " + p)
    print("\nRESULT: %s" % ("ALL CHECKS PASS" if not fails
                            else "*** FAILED: %s ***" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
