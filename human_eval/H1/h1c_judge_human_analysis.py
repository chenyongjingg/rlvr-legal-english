# -*- coding: utf-8 -*-
"""
h1c_judge_human_analysis.py — three-way alignment: judge vs pilot humans vs auto
stack, on the exact 40-item x 6-system human-eval subset.  Runs LOCALLY (no GPU).

Keying: every item is identified by the composite "{id}_{lv}" (e.g.
"eur_190_beginner") — the rated CSVs already carry that string as item_id, while
the judge jsonl and the config jsonls carry id and lv separately.  All three
views are normalized onto that composite key so the pooled correlations join.

Sources
  judge : human40_judge.jsonl   (H1B output; rows {id,lv,system,tot,fre,judge{..}})
  human : samples/rated_R1..R3.csv decoded by samples/unblind_key.json
  auto  : 5 configs B2 / M2_honest / B3_flant5 / B4_bart / G6v4 (40/40 rewrites
          verified == the pilot systems; carry parts dict values)

Outputs Q1 (per-system means + 3 rankings), Q2 (judge<->human pooled Spearman),
Q3 (judge<->auto and human<->auto pooled Spearman), and h1c_verdict.json.
"""
import json, os, csv, sys, statistics, argparse

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.normpath(os.path.join(BASE, "..", "samples"))
RES = os.path.normpath(os.path.join(BASE, "..", "..", "results"))  # <repo>/results
H1 = os.path.join(BASE, "h1c_out")
os.makedirs(H1, exist_ok=True)

SYSTEMS = ["Source", "B2", "M2", "FlanT5", "BART", "G6"]
NON_SRC = ["B2", "M2", "FlanT5", "BART", "G6"]
CONFIG = {"B2": "B2.jsonl", "M2": "M2_honest.jsonl", "FlanT5": "B3_flant5.jsonl",
          "BART": "B4_bart.jsonl", "G6": "G6v4.jsonl"}
DIMS = ("difficulty", "faithfulness", "terminology", "fluency")
HUMAN_DIM = {"difficulty": "difficulty_1_5", "faithfulness": "meaning_1_5",
             "terminology": "terminology_1_5", "fluency": "fluency_1_5"}
LEVEL_TARGET_FRE = {"beginner": 63.0, "intermediate": 45.0, "advanced": 26.0}
def r_diff_auto(fre, lv):
    if fre is None or lv not in LEVEL_TARGET_FRE:
        return None
    return max(0.0, 1.0 - abs(fre - LEVEL_TARGET_FRE[lv]) / 20.0)
def mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return sum(xs) / len(xs) if xs else None
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_human():
    """(item_key, system) -> list of rater {dim: value} dicts.  item_key already
    equals the CSV item_id string, e.g. 'eur_190_beginner'."""
    key = json.load(open(os.path.join(SAMPLES, "unblind_key.json"), encoding="utf-8"))
    human = {}
    search_dirs = [SAMPLES]
    rater_files, seen = [], set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("rated_R") and f.endswith(".csv") and f not in seen:
                seen.add(f)
                rater_files.append(os.path.join(d, f))
    if not rater_files:
        raise SystemExit("no rated_R*.csv found under " + " / ".join(search_dirs))
    for path in rater_files:
        rname = os.path.basename(path).replace("rated_", "").replace(".csv", "")
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig"))]
        if len(rows) < 10:
            print(f"[warn] {path}: only {len(rows)} rows, skipping")
            continue
        for row in rows:
            item_key = row["item_id"]; lab = row["label"]
            mapping = key.get(rname, {}).get(item_key)
            if mapping is None:
                kk = next((k for k in key.values() if item_key in k), None)
                mapping = kk[item_key] if kk else None
            if mapping is None:
                continue
            human.setdefault((item_key, mapping[lab]), []).append(
                {d: num(row.get(HUMAN_DIM[d])) for d in DIMS}
                | {"overall": num(row.get("overall_1_5"))})
    return human


def load_judge(path):
    """{(item_key, system): row}; item_key = f'{id}_{lv}'."""
    out = {}
    for r in (json.loads(l) for l in open(path, encoding="utf-8")):
        if r.get("judge"):
            out[(f'{r["id"]}_{r["lv"]}', r["system"])] = r
    return out


def load_auto():
    """{(system, item_key): {tot,fre,parts}} from config jsons."""
    out = {}
    for sysname, fn in CONFIG.items():
        p = os.path.join(RES, fn)
        if not os.path.isfile(p):
            print(f"[warn] missing auto config {p}")
            continue
        for r in (json.loads(l) for l in open(p, encoding="utf-8")):
            out[(sysname, f'{r["id"]}_{r["lv"]}')] = {"tot": r.get("tot"),
                                                      "fre": r.get("fre"),
                                                      "parts": r.get("parts", {})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.path.join(BASE, "human40_judge.jsonl"))
    args = ap.parse_args()
    if not os.path.isfile(args.judge):
        raise SystemExit(f"judge jsonl not found: {args.judge}")
    human = load_human()
    judge = load_judge(args.judge)
    auto = load_auto()
    import scipy.stats as sc

    def rho(xs, ys):
        pts = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
        if len(pts) < 5:
            return None
        xv = [p[0] for p in pts]; yv = [p[1] for p in pts]
        if len(set(xv)) < 2 or len(set(yv)) < 2:
            return None
        r, p = sc.spearmanr(xv, yv)
        return (float(r), float(p), len(pts))

    def human_mean(s):
        items = {}
        for (ik, sysn), rs in human.items():
            if sysn != s:
                continue
            items[ik] = {"overall": mean([r["overall"] for r in rs]),
                         "dims": {d: mean([r[d] for r in rs]) for d in DIMS}}
        if not items:
            return None
        return {"overall": mean([v["overall"] for v in items.values()]),
                "dims": {d: mean([v["dims"][d] for v in items.values()
                                  if v["dims"][d] is not None]) for d in DIMS},
                "n": len(items)}

    def judge_mean(s):
        sub = [r for (ik, sysn), r in judge.items() if sysn == s]
        if not sub:
            return None
        return {"overall": mean([mean([r["judge"][d] for d in DIMS]) for r in sub]),
                "dims": {d: mean([r["judge"][d] for r in sub]) for d in DIMS},
                "n": len(sub)}

    def auto_mean(s):
        vals = [v for (sn, _), v in auto.items() if sn == s]
        if not vals:
            return None
        return {"tot": mean([v["tot"] for v in vals]),
                "diff": mean([(v["parts"] or {}).get("diff") for v in vals]),
                "faith": mean([(v["parts"] or {}).get("faith") for v in vals]),
                "term": mean([(v["parts"] or {}).get("term") for v in vals]),
                "n": len(vals)}

    Hm = {s: human_mean(s) for s in SYSTEMS if s != "Source"}
    Hm = {s: m for s, m in Hm.items() if m}
    Jm = {s: judge_mean(s) for s in SYSTEMS if s != "Source"}
    Jm = {s: m for s, m in Jm.items() if m}
    Am = {s: auto_mean(s) for s in NON_SRC}
    Am = {s: m for s, m in Am.items() if m}
    rank = lambda m, key: sorted(m, key=lambda s: m[s][key], reverse=True)

    print("=" * 76)
    print("Q1  Per-system means over the 40 pilot items  (H=human mean, "
          "J=judge, A=auto)")
    print("=" * 76)
    print("%-8s | %-18s | %-18s | %8s %8s" %
          ("system", "H diff/faith/overall", "J diff/faith/overall", "A.diff",
           "A.tot"))
    for s in SYSTEMS:
        if s == "Source":
            continue
        h, j, a = Hm.get(s, {}), Jm.get(s, {}), Am.get(s, {})
        def cell(m, key, fmt="%.2f"):
            if m.get(key) is not None:
                return fmt % m[key]            # top-level key (auto: diff/tot)
            v = (m.get("dims") or {}).get(key)  # per-dim key (human/judge)
            return fmt % v if v is not None else "--"
        print("%-8s | %6s %7s %7s | %6s %7s %7s | %8s %8s" % (
            s, cell(h, "difficulty"), cell(h, "faithfulness"), cell(h, "overall"),
            cell(j, "difficulty"), cell(j, "faithfulness"), cell(j, "overall"),
            cell(a, "diff"), cell(a, "tot")))
    print("\noverall ranking  HUMAN (5): " + " > ".join(
        f"{s}({Hm[s]['overall']:.2f})" for s in rank(Hm, "overall")))
    print("overall ranking  JUDGE (5): " + " > ".join(
        f"{s}({Jm[s]['overall']:.2f})" for s in rank(Jm, "overall")))
    print("tot    ranking  AUTO  (5): " + " > ".join(
        f"{s}({Am[s]['tot']:.2f})" for s in rank(Am, "tot")))

    def rank_rho(mA, kA, mB, kB):
        ss = [s for s in NON_SRC if s in mA and s in mB]
        if len(ss) < 4:
            return None
        r, p = sc.spearmanr([mA[s][kA] for s in ss], [mB[s][kB] for s in ss])
        return (float(r), float(p), len(ss))
    jh, ja, ha = rank_rho(Jm, "overall", Hm, "overall"), \
                 rank_rho(Jm, "overall", Am, "tot"), \
                 rank_rho(Hm, "overall", Am, "tot")
    print("\nranking correlation across the 5 non-Source systems:")
    for name, res in (("judge vs human", jh), ("judge vs auto ", ja),
                      ("human vs auto ", ha)):
        print(f"  {name}: " + (f"rho={res[0]:+.3f}  p={res[1]:.4g}  (n={res[2]})"
                               if res else "n/a"))

    def judge_human_pooled():
        out = {}
        for d in DIMS + ("overall",):
            xs, ys = [], []
            for (ik, sysn), r in judge.items():
                if sysn == "Source":
                    continue
                rs = human.get((ik, sysn))
                if not rs:
                    continue
                jv = mean([r["judge"][dd] for dd in DIMS]) if d == "overall" \
                    else r["judge"][d]
                hm = mean([x["overall"] for x in rs]) if d == "overall" \
                    else mean([x[d] for x in rs])
                if jv is not None and hm is not None:
                    xs.append(jv); ys.append(hm)
            out[d] = rho(xs, ys)
        return out
    print("\n" + "=" * 76)
    print("Q2  pooled judge<->human Spearman over (item x system) points")
    print("=" * 76)
    jh_pool = judge_human_pooled()
    for d, res in jh_pool.items():
        print(f"  {d:14s}: " + (f"rho={res[0]:+.3f}  p={res[1]:.4g}  (n={res[2]})"
                                if res else "insufficient"))

    def judge_auto_pooled():
        out = {}
        for jd, ak in (("difficulty", "diff"), ("faithfulness", "faith"),
                       ("terminology", "term"), ("overall", "tot")):
            xs, ys = [], []
            for (ik, sysn), r in judge.items():
                if sysn == "Source":
                    continue
                a = auto.get((sysn, ik))
                if not a or a["parts"] is None:
                    continue
                av = a["tot"] if ak == "tot" else a["parts"].get(ak)
                jv = mean([r["judge"][d] for d in DIMS]) if jd == "overall" \
                    else r["judge"][jd]
                if av is not None and jv is not None:
                    xs.append(jv); ys.append(av)
            out[jd] = rho(xs, ys)
        return out
    print("\n" + "=" * 76)
    print("Q3  pooled judge<->auto  |  human<->auto  (5 systems)")
    print("=" * 76)
    ja_pool = judge_auto_pooled()
    ha_pool = {}
    for jd, ak in (("difficulty", "diff"), ("faithfulness", "faith"),
                   ("terminology", "term"), ("overall", "tot")):
        xs, ys = [], []
        for (ik, sysn), rs in human.items():
            if sysn == "Source":
                continue
            a = auto.get((sysn, ik))
            if not a or a["parts"] is None:
                continue
            hm = mean([x["overall"] for x in rs]) if jd == "overall" \
                else mean([x[jd] for x in rs])
            av = a["tot"] if ak == "tot" else \
                (r_diff_auto(a["fre"], ik.rsplit("_", 1)[-1]) if ak == "diff"
                 else (a["parts"] or {}).get(ak))
            if hm is not None and av is not None:
                xs.append(hm); ys.append(av)
        ha_pool[jd] = rho(xs, ys)
    for jd in ("difficulty", "faithfulness", "terminology", "overall"):
        for label, table in (("J-A", ja_pool), ("H-A", ha_pool)):
            res = table[jd]
            print(f"  {label} {jd:12s}: " +
                  (f"rho={res[0]:+.3f}  p={res[1]:.4g}  (n={res[2]})"
                   if res else "insufficient"))

    verdict = {"per_system_overall": {
                   "human": {s: Hm.get(s, {}).get("overall") for s in NON_SRC},
                   "judge": {s: Jm.get(s, {}).get("overall") for s in NON_SRC},
                   "auto_tot": {s: Am.get(s, {}).get("tot") for s in NON_SRC}},
               "rank_rho": {"judge_vs_human": jh, "judge_vs_auto": ja,
                            "human_vs_auto": ha},
               "judge_human_pooled": {d: v for d, v in jh_pool.items()},
               "judge_auto_pooled": {d: v for d, v in ja_pool.items()},
               "human_auto_pooled": {d: v for d, v in ha_pool.items()}}
    vp = os.path.join(H1, "h1c_verdict.json")
    json.dump(verdict, open(vp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nverdict -> {vp}")


if __name__ == "__main__":
    main()
