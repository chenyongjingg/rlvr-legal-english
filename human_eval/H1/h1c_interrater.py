# -*- coding: utf-8 -*-
"""h1c_interrater.py -- inter-rater agreement on the 240-cell blind set.

Restored 2026-09-11.  The original producing script was lost from disk; this
one reproduces `h1c_out/h1c_interrater.json` value-for-value from the shipped
`rated_R*.csv` + `unblind_key.json`, so the numbers the manuscript cites
(mean pairwise weighted kappa 0.02-0.11; 4.6-10.8% exact three-way agreement)
are reproducible from the released repository.

Method: pairwise linearly weighted Cohen's kappa (sklearn), each pair rounded
to 3 dp before averaging across the three pairs -- this rounding order is what
the shipped JSON reflects.

Inputs (all released):
  human_eval/samples/unblind_key.json
  human_eval/samples/rated_R{1,2,3}.csv    (or $RATED_DIR with the same names)

Output:
  h1c_out/h1c_interrater.json
"""
import csv
import json
import os

from sklearn.metrics import cohen_kappa_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES = os.path.join(BASE, "samples")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1c_out")
# `samples/` ships rated_R{1,2,3}.csv, so nothing else is needed to reproduce the
# shipped JSON.  Set RATED_DIR to a directory holding the same filenames to run
# it against an un-released copy.
SEARCH = [SAMPLES]
if os.environ.get("RATED_DIR"):
    SEARCH.append(os.path.abspath(os.environ["RATED_DIR"]))

HUMAN_DIM = {"difficulty": "difficulty_1_5", "faithfulness": "meaning_1_5",
             "terminology": "terminology_1_5", "fluency": "fluency_1_5"}
DIMS = ["difficulty", "faithfulness", "terminology", "fluency", "overall"]
PAIRS = [("R1", "R2"), ("R1", "R3"), ("R2", "R3")]


def load_ratings():
    """rater -> {(item_key, system): {dim: float}} plus the raw valid_rewrite."""
    key = json.load(open(os.path.join(SAMPLES, "unblind_key.json"),
                         encoding="utf-8"))
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
        raise SystemExit("need rated_R{{1,2,3}}.csv; found " + str(sorted(files)))
    rat, valid = {}, {}
    for rn, path in files.items():
        rat[rn], valid[rn] = {}, {}
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            ik, lab = row["item_id"], row["label"]
            m = key.get(rn, {}).get(ik)
            if m is None:
                continue
            cell = (ik, m[lab])
            rat[rn][cell] = {d: float(row[HUMAN_DIM[d]]) for d in HUMAN_DIM}
            rat[rn][cell]["overall"] = float(row["overall_1_5"])
            valid[rn][cell] = str(row.get("valid_rewrite", "")).strip()[:1].upper()
    return rat, valid


def main():
    rat, valid = load_ratings()
    raters = sorted(rat)
    cells = sorted(set(rat[raters[0]]) & set(rat[raters[1]]) & set(rat[raters[2]]))
    print("raters %s | common cells %d" % (raters, len(cells)))

    out = {"n_points": len(cells)}
    for d in DIMS:
        pairs = {}
        kv = []
        for a, b in PAIRS:
            k = cohen_kappa_score([rat[a][c][d] for c in cells],
                                  [rat[b][c][d] for c in cells],
                                  weights="linear")
            pairs["%s-%s" % (a, b)] = round(k, 3)
            kv.append(round(k, 3))
        n_ident = sum(1 for c in cells
                      if rat["R1"][c][d] == rat["R2"][c][d] == rat["R3"][c][d])
        out[d] = {"n": len(cells),
                  "mean_pairwise_lin_kappa": sum(kv) / len(kv),
                  "pairs": pairs,
                  "pct_3_raters_identical": round(100 * n_ident / len(cells), 1)}

    kr = []
    for a, b in PAIRS:
        cc = [c for c in cells if c in valid[a] and c in valid[b]]
        y1 = [1 if valid[a][c].startswith("Y") else 0 for c in cc]
        y2 = [1 if valid[b][c].startswith("Y") else 0 for c in cc]
        kr.append(round(cohen_kappa_score(y1, y2), 3))
    out["valid_rewrite"] = {"n": len(cells),
                            "mean_pairwise_cohen": sum(kr) / len(kr),
                            "pairs": kr}

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "h1c_interrater.json")
    # newline="\n": Python text mode would otherwise emit CRLF on Windows and
    # the file would no longer be byte-identical to the shipped copy
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("written", p)
    for d in DIMS:
        print("  %-14s mean lin kappa %.4f  3-identical %.1f%%"
              % (d, out[d]["mean_pairwise_lin_kappa"],
                 out[d]["pct_3_raters_identical"]))
    print("  valid_rewrite  mean cohen %.4f  pairs %s"
          % (out["valid_rewrite"]["mean_pairwise_cohen"],
             out["valid_rewrite"]["pairs"]))


if __name__ == "__main__":
    main()
