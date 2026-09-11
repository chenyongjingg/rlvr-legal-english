# -*- coding: utf-8 -*-
"""
analysis.py — analyze the blind human-evaluation results for the legal-English
rewrite paper (Applied Intelligence resubmission).

Reads the filled-in rater CSVs produced by make_human_eval_samples.py (columns:
item_id, level, source_text, label, rewrite, difficulty_1_5, meaning_1_5,
terminology_1_5, fluency_1_5, overall_1_5, valid_rewrite, comment), maps each
anonymized A–F label back to its system via unblind_key.json, and reports:

  1. per-system x per-dimension mean ± std (and % valid_rewrite)
  2. Friedman test across the 6 systems (repeated measures, blocked by item)
  3. Wilcoxon signed-rank post-hoc (G6 vs each baseline, Bonferroni-corrected)
  4. auto-human consistency: Spearman rho between the automatic reward stack
     (tot / r_diff / faith / term) and the corresponding human dimension.

Usage:
  python analysis.py --items items.jsonl --key unblind_key.json \
      --scores rated_R1.csv rated_R2.csv rated_R3.csv
"""
import argparse, csv, json, os, statistics, sys

DIMS = ["difficulty_1_5", "meaning_1_5", "terminology_1_5",
        "fluency_1_5", "overall_1_5"]
DIM_LABEL = {
    "difficulty_1_5": "Difficulty",
    "meaning_1_5": "Faithfulness",
    "terminology_1_5": "Terminology",
    "fluency_1_5": "Fluency",
    "overall_1_5": "Overall",
}
# auto field that mirrors each human dimension (for the consistency test)
AUTO_FIELD = {
    "difficulty_1_5": "r_diff",
    "meaning_1_5": "faith",
    "terminology_1_5": "term",
    "overall_1_5": "tot",
}
LEVEL_TARGET_FRE = {"beginner": 63.0, "intermediate": 45.0, "advanced": 26.0}
SYSTEMS = ["Source", "B2", "M2", "FlanT5", "BART", "G6"]


def mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def r_diff_auto(fre, lv):
    if fre is None or lv not in LEVEL_TARGET_FRE:
        return None
    return max(0.0, 1.0 - abs(fre - LEVEL_TARGET_FRE[lv]) / 20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--key", default="unblind_key.json")
    ap.add_argument("--scores", nargs="+", required=True,
                    help="filled-in rater CSVs (one per rater)")
    ap.add_argument("--focus", default="G6", help="system to test vs baselines")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    item_map = {"%s_%s" % (e["id"], e["lv"]): e for e in items}
    key = json.load(open(args.key, encoding="utf-8"))

    # human ratings: (item_id, system) -> list of per-rater {dim: value}
    human = {}
    rater_names = []
    for sf in args.scores:
        # "rated_R1.csv" -> "R1" ; "R1.csv" -> "R1"
        rname = os.path.splitext(os.path.basename(sf))[0].replace("rated_", "")
        rater_names.append(rname)
        for row in csv.DictReader(open(sf, encoding="utf-8-sig")):
            item_id = row["item_id"]
            lab = row["label"]
            if item_id not in key[rname]:
                # tolerate raters keyed differently; try any rater key
                kk = next((k for k in key.values() if item_id in k), None)
                mapping = kk[item_id] if kk else None
            else:
                mapping = key[rname][item_id]
            if mapping is None:
                continue
            sysname = mapping[lab]
            human.setdefault((item_id, sysname), []).append(
                {d: num(row.get(d)) for d in DIMS})

    # ---- 1. per-system x per-dimension means ----
    print("=" * 78)
    print("1) Per-system human ratings (mean ± std over items x raters)")
    print("=" * 78)
    header = "%-8s" % "system"
    for d in DIMS:
        header += "  %-11s" % DIM_LABEL[d]
    print(header)
    for s in SYSTEMS:
        line = "%-8s" % s
        for d in DIMS:
            vals = [v[d] for (_, sysn), rs in human.items() if sysn == s
                    for v in rs if v.get(d) is not None]
            if vals:
                line += "  %5.2f±%.2f" % (mean(vals), std(vals))
            else:
                line += "  %11s" % "n/a"
        print(line)

    # ---- 2. Friedman + 3. Wilcoxon ----
    try:
        from scipy import stats as sc
    except ImportError:
        print("\n[scipy not installed — skipping significance tests. "
              "pip install scipy]")
        return

    print("\n" + "=" * 78)
    print("2) Friedman test (repeated measures, blocked by item, "
          "raters averaged per item)")
    print("=" * 78)
    for d in DIMS:
        # build item x system matrix of per-item means
        per_item = {}
        for (item_id, sysn), rs in human.items():
            vals = [v[d] for v in rs if v.get(d) is not None]
            if vals:
                per_item.setdefault(item_id, {})[sysn] = mean(vals)
        # keep only items with all 6 systems rated
        full_items = [i for i, m in per_item.items()
                      if all(s in m for s in SYSTEMS)]
        if len(full_items) < 6:
            print(f"  {DIM_LABEL[d]:12s}: only {len(full_items)} complete "
                  f"items — skipped")
            continue
        cols = [[per_item[i][s] for i in full_items] for s in SYSTEMS]
        stat, p = sc.friedmanchisquare(*cols)
        print(f"  {DIM_LABEL[d]:12s}: chi2={stat:.2f}  p={p:.4g}  "
              f"(n={len(full_items)} items)")

    print("\n" + "=" * 78)
    print("3) Wilcoxon signed-rank: %s vs each baseline (Bonferroni, "
          "alpha=0.05/%d)" % (args.focus, len(SYSTEMS) - 1))
    print("=" * 78)
    for d in DIMS:
        base = {}
        for s in SYSTEMS:
            base[s] = {}
            for (item_id, sysn), rs in human.items():
                if sysn == s:
                    vals = [v[d] for v in rs if v.get(d) is not None]
                    if vals:
                        base[s][item_id] = mean(vals)
        out = []
        for s in SYSTEMS:
            if s == args.focus:
                continue
            common = [i for i in base[args.focus] if i in base[s]]
            if len(common) < 6:
                continue
            a = [base[args.focus][i] for i in common]
            b = [base[s][i] for i in common]
            w, p = sc.wilcoxon(a, b, alternative="greater")
            out.append((s, w, p, mean(a), mean(b)))
        print(f"  {DIM_LABEL[d]:12s} (%s > baseline):" % args.focus)
        if not out:
            print("      (insufficient paired data)")
        for s, w, p, ma, mb in out:
            pc = min(1.0, p * (len(SYSTEMS) - 1))
            sig = "*" if pc < 0.05 else "n.s."
            print(f"      vs {s:8s}: W={w:7.1f} p={p:.4g} "
                  f"(corr {pc:.4g}) {sig}   "
                  f"{args.focus}={ma:.2f} {s}={mb:.2f}")

    # ---- 4. auto-human consistency ----
    print("\n" + "=" * 78)
    print("4) Auto-human consistency (Spearman rho), system = %s" % args.focus)
    print("=" * 78)
    for d in DIMS:
        if d == "fluency_1_5":
            continue  # no automatic counterpart
        pairs = []
        for (item_id, sysn), rs in human.items():
            if sysn != args.focus:
                continue
            hm = [v[d] for v in rs if v.get(d) is not None]
            if not hm:
                continue
            e = item_map.get(item_id)
            if e is None:
                continue
            rec = e["systems"].get(args.focus, {})
            if d == "overall_1_5":
                av = rec.get("tot")
            elif d == "difficulty_1_5":
                av = r_diff_auto(rec.get("fre"), e["lv"])
            elif d == "terminology_1_5":
                p = rec.get("parts") or {}
                av = p.get("term")
            else:  # meaning_1_5 -> faithfulness (parts.faith == NLI entailment)
                p = rec.get("parts") or {}
                av = p.get("faith")
            if av is None:
                continue
            pairs.append((av, mean(hm)))
        if len(pairs) < 5:
            print(f"  {DIM_LABEL[d]:12s}: only {len(pairs)} pairs — skipped")
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        rho, p = sc.spearmanr(xs, ys)
        print(f"  {DIM_LABEL[d]:12s}: rho={rho:+.3f}  p={p:.4g}  (n={len(pairs)})")


if __name__ == "__main__":
    main()
