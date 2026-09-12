# -*- coding: utf-8 -*-
"""icc_reliability.py -- reliability of the three-rater panel mean (S4.13, S5.3).

Added 2026-09-12.  Until now every ICC(2,3) value printed in the manuscript was
produced by an *unreleased* ad-hoc computation; this script is the missing
producer, and it reproduces all of them from the shipped sheets.

It reports both index families on both cell bases, because the manuscript's
argument in S5.3 turns on the distinction:

  ICC(2,k)  absolute agreement -- charges the panel for systematic per-rater
            level differences (R1 never used the top of the scale).  This is the
            0.10-0.32 (240 cells) / 0.26-0.61 (200 cells) figure in the paper.
  ICC(3,k)  consistency -- does not charge for level offsets, and is therefore
            the index that matches the *rank* correlations of S4.13.  On it no
            dimension's point estimate exceeds its own sqrt(ICC) ceiling.

The 240 vs 200 distinction: the 40 `Source` cells are 20 passages each rated at
*both* target levels, so the same text appears twice inside one target cell.
That duplication inflates within-target dispersion relative to between-target
dispersion -- exactly the contrast ICC(2,k) is built on -- and roughly halves the
240-cell values.  The S4.13 lens correlations are computed on the 200-cell basis,
so 0.26-0.61 is the reliability of their target.

Inputs (all released):
  human_eval/samples/unblind_key.json
  human_eval/samples/rated_R{1,2,3}.csv     (or $RATED_DIR with the same names)

Output:
  h1c_out/icc_reliability.json
"""
import csv
import json
import os

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES = os.path.join(BASE, "samples")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1c_out")
SEARCH = [SAMPLES]
if os.environ.get("RATED_DIR"):
    SEARCH.append(os.path.abspath(os.environ["RATED_DIR"]))

HUMAN_DIM = {"difficulty": "difficulty_1_5", "faithfulness": "meaning_1_5",
             "terminology": "terminology_1_5", "fluency": "fluency_1_5",
             "overall": "overall_1_5"}
DIMS = ["difficulty", "faithfulness", "terminology", "fluency", "overall"]
RATERS = ["R1", "R2", "R3"]
N_BOOT = 2000
SEED = 20260911
# Table 12 pooled judge<->panel Spearman, the statistics the ceiling is read against
RHO = {"difficulty": 0.536, "faithfulness": 0.599, "terminology": 0.532,
       "fluency": 0.699, "overall": 0.685}


def load_ratings():
    """rater -> {(item_key, system): {dim: float}}."""
    key = json.load(open(os.path.join(SAMPLES, "unblind_key.json"), encoding="utf-8"))
    files, seen = {}, set()
    for d in SEARCH:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("rated_R") and f.endswith(".csv") and f not in seen:
                seen.add(f)
                files[f.replace("rated_", "").replace(".csv", "")] = \
                    os.path.join(d, f)
    if len(files) < 3:
        raise SystemExit("need rated_R{1,2,3}.csv; found " + str(sorted(files)))
    rat = {}
    for rn, path in files.items():
        rat[rn] = {}
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            ik, lab = row["item_id"], row["label"]
            m = key.get(rn, {}).get(ik)
            if m is None:
                continue
            cell = (ik, m[lab])
            rat[rn][cell] = {d: float(row[HUMAN_DIM[d]]) for d in DIMS}
    return rat


def icc(M, model):
    """M: n targets x k raters.  model=2 -> ICC(2,k) absolute agreement;
    model=3 -> ICC(3,k) consistency.  Average-of-k form in both cases."""
    n, k = M.shape
    gm = M.mean()
    ms_r = k * ((M.mean(axis=1) - gm) ** 2).sum() / (n - 1)   # between targets
    ms_c = n * ((M.mean(axis=0) - gm) ** 2).sum() / (k - 1)   # between raters
    ss_e = ((M - M.mean(axis=1, keepdims=True)
             - M.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    if model == 2:
        return (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
    return (ms_r - ms_e) / ms_r


def boot_ci(M, model):
    rng = np.random.default_rng(SEED)
    n = M.shape[0]
    v = np.array([icc(M[rng.integers(0, n, n)], model) for _ in range(N_BOOT)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    rat = load_ratings()
    allc = sorted(set(rat["R1"]) & set(rat["R2"]) & set(rat["R3"]))
    gen = [c for c in allc if c[1] != "Source"]
    print("cells: %d all | %d generated | %d Source"
          % (len(allc), len(gen), len(allc) - len(gen)))

    out = {"n_all": len(allc), "n_generated": len(gen),
           "n_resamples": N_BOOT, "seed": SEED, "basis": {}}
    for label, cells in [("240_cells_all", allc), ("200_cells_generated", gen)]:
        out["basis"][label] = {}
        print("\n== %s ==" % label)
        print("  %-13s %9s %9s %20s" % ("dim", "ICC(2,3)", "ICC(3,3)", "ICC(3,3) CI95"))
        for d in DIMS:
            M = np.array([[rat[r][c][d] for r in RATERS] for c in cells])
            k2, k3 = icc(M, 2), icc(M, 3)
            lo, hi = boot_ci(M, 3)
            out["basis"][label][d] = {"icc23": round(k2, 4), "icc33": round(k3, 4),
                                      "icc33_ci95": [round(lo, 4), round(hi, 4)]}
            print("  %-13s %+9.4f %+9.4f   [%+.3f, %+.3f]" % (d, k2, k3, lo, hi))

    print("\n== ceiling check on the 200-cell basis (S4.13 rho vs sqrt(ICC)) ==")
    print("  %-13s %7s %10s %10s %11s" %
          ("dim", "rho", "sqrtI2", "sqrtI3", "exceeds I3"))
    out["ceiling_check_200"] = {}
    for d in DIMS:
        M = np.array([[rat[r][c][d] for r in RATERS] for c in gen])
        c2, c3 = np.sqrt(icc(M, 2)), np.sqrt(icc(M, 3))
        ex = bool(RHO[d] > c3)
        out["ceiling_check_200"][d] = {"rho": RHO[d], "sqrt_icc23": round(c2, 4),
                                       "sqrt_icc33": round(c3, 4),
                                       "rho_exceeds_icc33_ceiling": ex}
        print("  %-13s %7.3f %10.3f %10.3f %11s" % (d, RHO[d], c2, c3, ex))
    out["any_exceeds_icc33"] = any(
        v["rho_exceeds_icc33_ceiling"] for v in out["ceiling_check_200"].values())

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "icc_reliability.json")
    # newline="\n" so the checked-in copy stays LF on Windows
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwritten", p)
    print("no dimension exceeds its ICC(3,3) ceiling: %s" % (not out["any_exceeds_icc33"]))


if __name__ == "__main__":
    main()
