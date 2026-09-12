# -*- coding: utf-8 -*-
"""h1c_loo_basis.py -- leave-one-out rater agreement on BOTH cell bases.

WHY THIS EXISTS
The manuscript compares two numbers that are computed on different cell sets:

  * judge <-> human panel mean: pooled Spearman on the 200-cell basis
    (40 items x 5 GENERATED systems), reported in Table 12;
  * the leave-one-out rater comparator ("a single rater recovers the other two
    raters' mean at rho = 0.07 / 0.24 / 0.29 / 0.10 / 0.18"), computed on all
    240 cells, i.e. including the 40 `Source` cells.

§5.3 states that the 240-cell pool SUPPRESSES rater-rater agreement (mean
pairwise Pearson on difficulty is 0.001 over the six-system set and rises to
0.147 once restricted to the five generated systems, because the same source
text is rated twice under two different targets).  So the comparison runs in
the direction that flatters the judge: the comparator is measured on the
suppressed basis while the judge is measured on the clean one.

This script recomputes the comparator on the SAME basis as the judge.
Step 1 must reproduce the published 240-cell figures exactly; if it does not,
the method here does not match the manuscript's and the 200-cell figures
below are not usable.

Output: h1c_out/h1c_loo_basis.json
"""
import csv
import json
import os

from scipy.stats import spearmanr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.abspath(os.path.join(BASE, ".."))

# `samples/` ships the BLANK scoresheet_R*.csv templates; the FILLED sheets are
# the released `rated_R*.csv`, which live in the paper tree's clone of the
# public repo.  Search both, preferring the filled ones.
SEARCH = [
    os.path.join(PAPER, "github_repo", "human_eval", "samples"),
    os.path.join(BASE, "samples"),
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1c_out")

# same column mapping as h1c_interrater.py
HUMAN_DIM = {"difficulty": "difficulty_1_5", "faithfulness": "meaning_1_5",
             "terminology": "terminology_1_5", "fluency": "fluency_1_5"}
DIMS = ["difficulty", "faithfulness", "terminology", "fluency", "overall"]
RATERS = ["R1", "R2", "R3"]
SOURCE_LABELS = {"Source"}


def load():
    key = None
    for d in SEARCH:
        p = os.path.join(d, "unblind_key.json")
        if os.path.isfile(p):
            key = json.load(open(p, encoding="utf-8"))
            break
    if key is None:
        raise SystemExit("unblind_key.json not found in " + str(SEARCH))

    rat, files = {}, {}
    for d in SEARCH:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("rated_R") and f.endswith(".csv"):
                rn = f.replace("rated_", "").replace(".csv", "")
                files.setdefault(rn, os.path.abspath(os.path.join(d, f)))
    if len(files) < 3:
        raise SystemExit("need rated_R{1,2,3}.csv; found " + str(sorted(files)))
    print("ratings source: %s" % os.path.dirname(list(files.values())[0]))

    for rn, path in files.items():
        rat[rn] = {}
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            ik, lab = row["item_id"], row["label"]
            m = key.get(rn, {}).get(ik)
            if m is None:
                continue
            cell = (ik, m[lab])
            s = {d: float(row[HUMAN_DIM[d]]) for d in HUMAN_DIM}
            s["overall"] = float(row["overall_1_5"])
            rat[rn][cell] = s
    return rat


def loo(rat, cells, d):
    """rho of each rater against the mean of the other two, per rater + mean."""
    per, vals = {}, {}
    for rn in RATERS:
        y = [rat[rn][c][d] for c in cells]
        others = [[rat[o][c][d] for c in cells] for o in RATERS if o != rn]
        om = [(a + b) / 2.0 for a, b in zip(*others)]
        vals[rn] = y
        per[rn] = float(spearmanr(y, om)[0])
    return per, sum(per.values()) / len(per)


def main():
    rat = load()
    all_cells = sorted(set(rat["R1"]) & set(rat["R2"]) & set(rat["R3"]))
    gen_cells = [c for c in all_cells if c[1] not in SOURCE_LABELS]
    print("cells: all=%d  generated=%d  (excluded Source=%d)"
          % (len(all_cells), len(gen_cells), len(all_cells) - len(gen_cells)))

    out = {"n_all": len(all_cells), "n_generated": len(gen_cells), "dims": {}}
    print("\n%-14s %-28s %s" % ("dimension", "240-cell (published basis)",
                                "200-cell (judge's basis)"))
    for d in DIMS:
        per240, m240 = loo(rat, all_cells, d)
        per200, m200 = loo(rat, gen_cells, d)
        out["dims"][d] = {
            "all240": {"per_rater": per240, "mean": m240},
            "gen200": {"per_rater": per200, "mean": m200},
        }
        print("%-14s %.2f  (R1 %.2f R2 %.2f R3 %.2f)   %.2f  (R1 %.2f R2 %.2f R3 %.2f)"
              % (d, m240, per240["R1"], per240["R2"], per240["R3"],
                 m200, per200["R1"], per200["R2"], per200["R3"]))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "h1c_loo_basis.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwritten", p)

    pub = [0.07, 0.24, 0.29, 0.10, 0.18]
    got = [round(out["dims"][d]["all240"]["mean"], 2) for d in DIMS]
    print("\nreproduction check (240-cell mean vs published 0.07/0.24/0.29/0.10/0.18):")
    print("  got      %s" % got)
    print("  MATCH" if got == pub else "  *** DOES NOT MATCH -- method differs ***")


if __name__ == "__main__":
    main()
