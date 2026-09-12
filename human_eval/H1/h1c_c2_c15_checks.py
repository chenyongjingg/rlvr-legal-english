# -*- coding: utf-8 -*-
"""h1c_c2_c15_checks.py -- the remaining no-human analyses for round C2 / C15.

Three independent checks, all on the released artifacts, no GPU:

  A. C2 (CRITICAL).  The judge's difficulty dimension is currently presented as
     a co-equal second leg for the difficulty claim ("external difficulty
     judgments rank M2 above G6v4: human 3.37 vs 3.01, judge 3.52 vs 3.02").
     The DA's objection is that the untouched `Source` statute gets judge
     difficulty 3.52 (joint-highest) against human 2.12 (lowest) -- i.e. the
     judge may be scoring general quality, not difficulty, on the one dimension
     the central negative result rests on.  Here that halo hypothesis is TESTED
     rather than asserted: Spearman partial correlations of judge<->human
     difficulty controlling for human fluency, faithfulness, and overall.

  B. C15, DA.  The manuscript's own Table 12 caption says the within-system
     mean "is given for the judge <-> human columns only".  The crux null
     (automatic difficulty ~= 0 with both lenses) is therefore reported POOLED
     while everything else is reported both ways.  This publishes the matching
     within-system means for judge<->auto and human<->auto, on the same 200-cell
     basis.  Reproduction check: the judge<->human within-system row must come
     back as 0.375 / 0.387 / 0.513 / 0.431 / 0.528.

  G. C15, DA.  34% of rows sit at the difficulty sub-reward's zero clamp.  A
     "~=0" claim computed on a boundary-saturated sample is weak even if it
     survives; this reports every automatic-difficulty correlation on all 200
     cells and again on the uncensored subset only.

Output: h1c_out/h1c_c2_c15_checks.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1c_loo_basis as L          # noqa: E402  (its load() finds the rated_*.csv)

BASE = os.path.dirname(os.path.abspath(__file__))


def results_dir():
    """Locate results/G6v4.jsonl.

    Walk up from this file so the script runs unmodified in a clone, where the
    directory sits at the repo root.  `LEGAL_ENGLISH_RESULTS` covers running it
    from a tree that keeps the released repo in a subdirectory.  An earlier
    version hardcoded an absolute path, which no reader could satisfy.
    """
    d = BASE
    for _ in range(4):
        cand = os.path.join(d, "results")
        if os.path.isfile(os.path.join(cand, "G6v4.jsonl")):
            return cand
        d = os.path.dirname(d)
    env = os.environ.get("LEGAL_ENGLISH_RESULTS", "").strip()
    if env and os.path.isfile(os.path.join(env, "G6v4.jsonl")):
        return env
    raise SystemExit("no results/ directory with the five config JSONLs "
                     "(checked %s and its parents; set LEGAL_ENGLISH_RESULTS)"
                     % BASE)


RES = results_dir()
OUT = os.path.join(BASE, "h1c_out")

NON_SRC = ["B2", "M2", "FlanT5", "BART", "G6"]
CONFIG = {"B2": "B2.jsonl", "M2": "M2_honest.jsonl", "FlanT5": "B3_flant5.jsonl",
          "BART": "B4_bart.jsonl", "G6": "G6v4.jsonl"}
JDIMS = ("difficulty", "faithfulness", "terminology", "fluency")
# judge dimension -> automatic sub-reward key ("tot" = total reward)
J2A = {"difficulty": "diff", "faithfulness": "faith",
       "terminology": "term", "overall": "tot"}
LEVEL_TARGET_FRE = {"beginner": 63.0, "intermediate": 45.0, "advanced": 26.0}


def r_diff_auto(fre, lv):
    if fre is None or lv not in LEVEL_TARGET_FRE:
        return None
    return max(0.0, 1.0 - abs(fre - LEVEL_TARGET_FRE[lv]) / 20.0)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_judge(path):
    out = {}
    for r in (json.loads(l) for l in open(path, encoding="utf-8")):
        if r.get("judge"):
            out[(f'{r["id"]}_{r["lv"]}', r["system"])] = r
    return out


def load_auto():
    out = {}
    for sysname, fn in CONFIG.items():
        p = os.path.join(RES, fn)
        if not os.path.isfile(p):
            raise SystemExit("missing auto config " + p)
        for r in (json.loads(l) for l in open(p, encoding="utf-8")):
            out[(sysname, f'{r["id"]}_{r["lv"]}')] = {
                "tot": r.get("tot"), "fre": r.get("fre"),
                "parts": r.get("parts", {})}
    return out


def rank(x):
    return rankdata(np.asarray(x, dtype=float)).astype(float)


def partial(x, y, ctrls):
    """Spearman partial correlation of x and y given ctrls (rank-residualised)."""
    n = len(x)
    if n < 6:
        return None
    cols = [rank(c) for c in ctrls] + [np.ones(n)]
    X = np.column_stack(cols)
    xr, yr = rank(x), rank(y)
    bx, *_ = np.linalg.lstsq(X, xr, rcond=None)
    by, *_ = np.linalg.lstsq(X, yr, rcond=None)
    rx, ry = xr - X @ bx, yr - X @ by
    if rx.std() == 0 or ry.std() == 0:
        return None
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - len(ctrls) - 2
    if abs(r) >= 1.0 or dof < 1:
        return (r, 0.0, dof, n)
    t = r * np.sqrt(dof / (1.0 - r * r))
    return (r, float(2 * tdist.sf(abs(t), dof)), dof, n)


def zero_order(x, y):
    if len(x) < 5 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    r, p = spearmanr(x, y)
    return (float(r), float(p), len(x))


def fmt(res, nd=3):
    if res is None:
        return "n/a (insufficient)"
    r, p = res[0], res[1]
    tail = "  dof=%d n=%d" % (res[2], res[3]) if len(res) == 4 else "  n=%d" % res[2]
    return "rho=%+.3f  p=%.3g%s" % (r, p, tail)


def main():
    rat = L.load()
    judge = load_judge(os.path.join(BASE, "human40_judge.jsonl"))
    auto = load_auto()
    RATERS = L.RATERS

    # ---- assemble the 200 generated cells (Source excluded, as the judge basis)
    cells = sorted({c for c in rat[RATERS[0]]
                    if c[1] not in L.SOURCE_LABELS
                    and c in judge and (c[1], c[0]) in auto})
    print("cells joined (human & judge & auto, Source excluded): %d" % len(cells))
    sysc = {}
    for (ik, sysname) in cells:
        sysc.setdefault(sysname, 0)
        sysc[sysname] += 1
    print("per system: %s" % ", ".join("%s=%d" % kv for kv in sorted(sysc.items())))

    def H(c, d):
        vs = [rat[rn][c][d] for rn in RATERS if c in rat[rn]]
        return sum(vs) / len(vs) if vs else None

    def J(c, d):
        r = judge[c]
        if d == "overall":
            vs = [r["judge"][x] for x in JDIMS]
            return sum(vs) / len(vs) if vs else None
        return r["judge"].get(d)

    def A(c, key):
        ik, sysname = c
        a = auto[(sysname, ik)]
        # "fre" is a TOP-LEVEL field of the config row, not a sub-reward in
        # `parts` -- falling through to the parts lookup returns None, which
        # silently dropped the raw-FRE row from earlier runs.
        if key == "fre":
            return a["fre"]
        if key == "diff":
            return r_diff_auto(a["fre"], ik.rsplit("_", 1)[-1])
        if key == "tot":
            return a["tot"]
        return (a["parts"] or {}).get(key)

    out = {}

    # =====================================================================
    print("\n" + "=" * 78)
    print("A.  C2 halo test -- does the judge read DIFFICULTY, or overall quality?")
    print("=" * 78)
    print("   pooled over the %d generated cells; partial correlations are"
          % len(cells))
    print("   rank-residualised Spearman (controls are the HUMAN panel means).")

    halo = {}
    for target in ("difficulty", "faithfulness"):
        x = [J(c, target) for c in cells]
        y = [H(c, target) for c in cells]
        keep = [i for i in range(len(cells)) if x[i] is not None and y[i] is not None]
        x = [x[i] for i in keep]
        y = [y[i] for i in keep]
        other = [d for d in JDIMS if d != target]
        ctrl = {d: [H(cells[i], d) for i in keep] for d in JDIMS}
        ctrl["overall"] = [H(cells[i], "overall") for i in keep]
        ok = [i for i in range(len(x))
              if all(ctrl[d][i] is not None for d in ctrl) and x[i] is not None]
        x, y = [x[i] for i in ok], [y[i] for i in ok]
        ctrl = {d: [ctrl[d][i] for i in ok] for d in ctrl}

        print("\n  judge<->human %s :" % target)
        print("    zero-order                        %s" % fmt(zero_order(x, y)))
        for name, cs in (
                ("c. fluency          ", ["fluency"]),
                ("c. other judged dim ", other),
                ("c. overall (halo)   ", ["overall"]),
                ("c. fluency+overall  ", ["fluency", "overall"]),
                ("c. all 4 other dims ", [d for d in ctrl if d != target])):
            cl = [ctrl[d] for d in cs]
            if any(len(v) != len(x) for v in cl):
                continue
            print("    %s            %s" % (name, fmt(partial(x, y, cl))))
        # the halo symptom itself: does judge difficulty track human OVERALL
        # better than it tracks human DIFFICULTY?
        if target == "difficulty":
            zo_hd = zero_order(x, y)
            zo_ho = zero_order(x, ctrl["overall"])
            print("    -- halo symptom: judge difficulty vs human OVERALL "
                  "%s" % fmt(zo_ho))
            print("       (vs human difficulty %s)" % fmt(zo_hd))
            halo = {"judge_difficulty_vs_human_overall": zo_ho,
                    "judge_difficulty_vs_human_difficulty": zo_hd}
        out["A_%s" % target] = {
            "zero_order": zero_order(x, y),
            "partial_fluency": partial(x, y, [ctrl["fluency"]]),
            "partial_other_dims": partial(x, y, [ctrl[d] for d in other]),
            "partial_overall": partial(x, y, [ctrl["overall"]]),
            "partial_all_dims": partial(x, y, [ctrl[d] for d in ctrl
                                               if d != target]),
            "n": len(x)}
    out["A_halo_symptom"] = halo

    # =====================================================================
    print("\n" + "=" * 78)
    print("B.  C15 -- within-system means for judge<->auto and human<->auto")
    print("=" * 78)
    print("   mean of the 5 per-system Spearman correlations (n = 40 items each)")

    def within(getx, gety, dims):
        res = {}
        for d in dims:
            per, n_ok = {}, 0
            for s in NON_SRC:
                cc = [c for c in cells if c[1] == s]
                xs = [getx(c, d) for c in cc]
                ys = [gety(c, d) for c in cc]
                pts = [(a, b) for a, b in zip(xs, ys)
                       if a is not None and b is not None]
                if len(pts) < 5:
                    continue
                xv = [p[0] for p in pts]
                yv = [p[1] for p in pts]
                if len(set(xv)) < 2 or len(set(yv)) < 2:
                    per[s] = None
                    continue
                per[s] = float(spearmanr(xv, yv)[0])
                n_ok += 1
            vals = [v for v in per.values() if v is not None]
            res[d] = {"per_system": per,
                      "mean": (sum(vals) / len(vals)) if vals else None,
                      "n_systems": n_ok}
        return res

    all_dims = list(JDIMS) + ["overall"]
    jh = within(lambda c, d: J(c, d), lambda c, d: H(c, d), all_dims)
    ja = within(lambda c, d: J(c, d), lambda c, d: A(c, J2A[d]),
                [d for d in all_dims if d in J2A])
    ha = within(lambda c, d: H(c, d), lambda c, d: A(c, J2A[d]),
                [d for d in all_dims if d in J2A])
    fre = within(lambda c, d: A(c, "fre") if d == "difficulty" else None,
                 lambda c, d: H(c, d), ["difficulty"])

    print("\n  %-14s %-28s %-28s %s" % ("dimension", "J<->H (reproduction)",
                                        "J<->A", "H<->A"))
    for d in all_dims:
        def g(t, dd=d):
            # ja/ha have no entry for dimensions with no automatic counterpart
            # (fluency has no automatic field at all), so a missing key is "--".
            if dd not in t:
                return "--"
            m = t[dd]["mean"]
            return "%.3f (n=%d)" % (m, t[dd]["n_systems"]) if m is not None else "--"
        print("  %-14s %-28s %-28s %s" % (d, g(jh), g(ja), g(ha)))
    print("  %-14s %-28s" % ("FRE raw <-> H", g(fre, "difficulty")))

    pub37 = [0.375, 0.387, 0.513, 0.431, 0.528]
    got37 = [round(jh[d]["mean"], 3) for d in all_dims]
    print("\n  reproduction check, J<->H within-system vs published "
          "0.375 / 0.387 / 0.513 / 0.431 / 0.528")
    print("    got  %s" % got37)
    print("    %s" % ("MATCH" if got37 == pub37
                      else "*** DOES NOT MATCH -- basis differs ***"))
    out["B_within_system"] = {"judge_human": jh, "judge_auto": ja,
                              "human_auto": ha, "fre_raw_human": fre,
                              "reproduction": {"published": pub37, "got": got37}}

    # =====================================================================
    print("\n" + "=" * 78)
    print("G.  C15 -- censoring robustness of the automatic-difficulty null")
    print("=" * 78)
    ad = [A(c, "diff") for c in cells]
    clamped = sum(1 for v in ad if v is not None and v <= 1e-12)
    have = sum(1 for v in ad if v is not None)
    print("   rows with an automatic difficulty value : %d" % have)
    print("   at the zero clamp (|FRE-target| >= 20)  : %d  (%.1f%%)"
          % (clamped, 100.0 * clamped / have))

    cens = {}
    for label, keep in (("all %d cells" % have, lambda c: A(c, "diff") is not None),
                        ("uncensored only",
                         lambda c: (A(c, "diff") or 0.0) > 1e-12)):
        cc = [c for c in cells if keep(c)]
        rows = {}
        for name, gx, gy in (
                ("auto.diff vs human.difficulty", lambda c: A(c, "diff"),
                 lambda c: H(c, "difficulty")),
                ("auto.diff vs judge.difficulty", lambda c: A(c, "diff"),
                 lambda c: J(c, "difficulty")),
                ("FRE raw  vs human.difficulty",
                 lambda c: A(c, "fre"), lambda c: H(c, "difficulty")),
                ("auto.diff vs auto.tot", lambda c: A(c, "diff"),
                 lambda c: A(c, "tot"))):
            xs = [gx(c) for c in cc]
            ys = [gy(c) for c in cc]
            pts = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
            if len(pts) < 5:
                rows[name] = None
                continue
            rows[name] = zero_order([p[0] for p in pts], [p[1] for p in pts])
        cens[label] = rows
        print("\n   [%s]  (n=%d)" % (label, len(cc)))
        for name, res in rows.items():
            print("     %-34s %s" % (name, fmt(res)))
    # what the clamped rows look like on the other lenses
    cl = [c for c in cells if (A(c, "diff") or 0.0) <= 1e-12
          and A(c, "diff") is not None]
    un = [c for c in cells if (A(c, "diff") or 0.0) > 1e-12]
    print("\n   clamped rows  : human difficulty mean %.2f  judge difficulty %.2f"
          % (sum(H(c, "difficulty") for c in cl) / len(cl),
             sum(J(c, "difficulty") for c in cl) / len(cl)))
    print("   uncensored    : human difficulty mean %.2f  judge difficulty %.2f"
          % (sum(H(c, "difficulty") for c in un) / len(un),
             sum(J(c, "difficulty") for c in un) / len(un)))
    out["G_censoring"] = {"n_with_value": have, "n_clamped": clamped,
                          "pct_clamped": 100.0 * clamped / have,
                          "tables": {k: {n: r for n, r in v.items()}
                                     for k, v in cens.items()}}

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "h1c_c2_c15_checks.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("\nwritten %s" % p)


if __name__ == "__main__":
    main()
