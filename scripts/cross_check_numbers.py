# -*- coding: utf-8 -*-
"""cross_check_numbers.py — verify every headline number in manuscript.md against
the real (honest) result files: honest_table.json / stats_legal.json /
verifier_eval.json / metrics_honest.json (+ raw jsonl for paired stats that
07_eval does not persist). Reports PASS / DIFF / MISSING for each.

Honest record: this tool does NOT mark sources verified; it only checks that the
number the manuscript quotes equals the number the pipeline produced.

Usage: python cross_check_numbers.py
"""
import json
import os
import sys

import numpy as np
from scipy import stats as sp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAPER = r"D:\周老师\paper"
RESULTS = r"C:\Users\33378\aigc_wechat\results\legal"

stats = json.load(open(os.path.join(RESULTS, "stats_legal.json"), encoding="utf-8"))
ht = json.load(open(os.path.join(RESULTS, "honest_table.json"), encoding="utf-8"))
ver = json.load(open(os.path.join(RESULTS, "verifier_eval.json"), encoding="utf-8"))
met = json.load(open(os.path.join(RESULTS, "metrics_honest.json"), encoding="utf-8"))

cfg = stats["cfg"]
results = []


def tol(a, b, rel=0.005, abs_=0.005):
    return abs(a - b) <= abs_ or abs(a - b) <= rel * max(abs(a), abs(b))


def check(label, got, expect, rel=0.005, abs_=0.005):
    if got is None or (isinstance(got, float) and got != got):  # NaN
        results.append(("MISSING", label, f"expected {expect}"))
    elif tol(got, expect, rel, abs_):
        results.append(("PASS", label, f"{got:.4g} vs {expect}"))
    else:
        results.append(("DIFF", label, f"{got:.4g} vs {expect}"))


def cfg_all(tag, field):
    v = (cfg.get(tag, {}).get("all", {}) or {}).get(field)
    return v


def cfg_lv(tag, lv, field):
    return (cfg.get(tag, {}).get(lv, {}) or {}).get(field)


# ------------------------------------------------------------------ Table 1
T1 = [  # (config, tot, fre_beg, fre_int)
    ("G1", 0.856, 55.9, 34.5),
    ("G2", 0.866, 60.4, 35.3),
    ("G3", 0.874, 54.0, 32.8),
    ("G4", 0.870, 60.4, 35.1),
    ("G5", 0.759, 77.2, 11.1),
    ("L3_max3", 0.800, 66.1, 31.5),   # = corrected G6
]
for tag, tot, fbg, fit in T1:
    check(f"T1 {tag} tot", cfg_all(tag, "tot"), tot)
    check(f"T1 {tag} FRE beg", cfg_lv(tag, "beginner", "fre_mean"), fbg, abs_=0.1)
    check(f"T1 {tag} FRE int", cfg_lv(tag, "intermediate", "fre_mean"), fit, abs_=0.1)
check("G6 diff", cfg_all("L3_max3", "diff"), 0.55, abs_=0.01)
check("G6 term", cfg_all("L3_max3", "term"), 1.00, abs_=0.01)
check("G6 copy", cfg_all("L3_max3", "copy"), 1.00, abs_=0.01)
check("G6 faith", cfg_all("L3_max3", "faith"), 0.82, abs_=0.01)
check("G6 halluc(contra)", cfg_all("L3_max3", "halluc"), 0.030, abs_=0.005)
check("G6 beginner tot", cfg_lv("L3_max3", "beginner", "tot"), 0.802, abs_=0.003)
check("G6 inter tot", cfg_lv("L3_max3", "intermediate", "tot"), 0.798, abs_=0.003)

# ------------------------------------------------------------------ stats
kw = stats["kw"]["tot"]
check("KW H", kw.get("H"), 85.69, abs_=0.05)
check("KW p", kw.get("p"), 5.4e-17, rel=0.1)
check("KW eps2", kw.get("eps2"), 0.143, abs_=0.005)
a_b = stats["anova"]["anova"]["beginner"]
a_i = stats["anova"]["anova"]["intermediate"]
check("ANOVA beg eta2", a_b.get("eta2"), 0.130, abs_=0.004)
check("ANOVA int eta2", a_i.get("eta2"), 0.107, abs_=0.004)
hx = stats.get("halluc_chi2", {})
check("halluc chi2", hx.get("chi2"), 9.64, abs_=0.05)
check("halluc chi2 p", hx.get("p"), 0.086, abs_=0.01)

# paired Wilcoxon persisted by 07_eval
def cp(a, b, mk, field):
    return stats["contrasts"].get(f"{a}-{b}", {}).get(mk, {}).get(field)
check("G1→G2 fre p", cp("G1", "G2", "fre", "p"), 0.016, abs_=0.002)
check("G4→G5 tot p", cp("G4", "G5", "tot", "p"), 3.7e-12, rel=0.3)
check("G4→G5 tot r", cp("G4", "G5", "tot", "r"), 0.70, abs_=0.02)
check("G5→G6 tot p", cp("G5", "L3_max3", "tot", "p"), 3.2e-3, abs_=0.5e-3)

# ------------------------------------------------------- computed inline
def _load(tag):
    for suf in ("_honest", ""):
        p = os.path.join(RESULTS, tag + suf + ".jsonl")
        if os.path.isfile(p):
            return [json.loads(l) for l in open(p, encoding="utf-8")]
    return None


def _pure(r):
    return bool(r.get("pure_commentary", not (r.get("rew") or "")))


def _val(r, mk):
    if mk == "tot":
        return 0.0 if _pure(r) else r["tot"]
    if mk == "diff":
        return 0.0 if _pure(r) else (r.get("parts") or {}).get("diff", 0.0)
    return r.get(mk)


def aligned(A, B, mk):
    da = {r["id"] + "\x00" + r["lv"]: r for r in A}
    db = {r["id"] + "\x00" + r["lv"]: r for r in B}
    kk = [k for k in da if k in db]
    return (np.array([_val(da[k], mk) for k in kk], float),
            np.array([_val(db[k], mk) for k in kk], float))


def wilc(A, B, mk):
    va, vb = aligned(A, B, mk)
    res = sp.wilcoxon(va, vb, zero_method="wilcox")
    return float(res.pvalue), float(np.median(vb - va))


G5 = _load("G5"); L3 = _load("L3_max3"); G4 = _load("G4")
p, _ = wilc(G5, L3, "diff");   check("G5→G6 diff p (inline)", p, 1.1e-9, rel=0.3)
p, _ = wilc(L3, G4, "tot");     check("G6 vs G4 tot p (inline)", p, 8.8e-5, rel=0.3)

def inband_int(rows):
    tgt = 45.0
    fre = [r["fre"] for r in rows if not _pure(r) and r.get("fre") is not None]
    return sum(1 for f in fre if abs(f - tgt) <= 10) / len(fre) if fre else None
check("G5 int in-band", inband_int([r for r in G5 if r["lv"] == "intermediate"]), 0.10, abs_=0.02)
check("G6 int in-band", inband_int([r for r in L3 if r["lv"] == "intermediate"]), 0.43, abs_=0.02)
int5 = [r for r in G5 if r["lv"] == "intermediate" and not _pure(r) and r.get("fre") is not None]
check("G5 int neg-FRE 14/48", sum(1 for r in int5 if r["fre"] < 0), 14, abs_=0)
check("G5 int valid n", len(int5), 48, abs_=0)

# ------------------------------------------------------- loop depth (Table 6)
for tag, v in [("L1_max1", 0.688), ("L2_max2", 0.766), ("L3_max3", 0.800), ("L5_max5", 0.828)]:
    check(f"loop {tag} tot", cfg_all(tag, "tot"), v, abs_=0.003)

# ------------------------------------------------------- seeds
sp_ = stats.get("seed_pair", {})
seeds = [cfg_all(t, "tot") for t in ("G6_pure", "G6_pure_s11", "G6_pure_s21", "G6_pure_s42")]
check("seed mean", sp_.get("mean_of_seed_means"), 0.828, abs_=0.002)
check("seed std", sp_.get("std"), 0.001, abs_=0.0005)
print("   seeds:", [round(s, 4) for s in seeds])

# ------------------------------------------------------- full test set
check("G6_full107 n", cfg_all("G6_full107", "n"), 214, abs_=0)
check("G6_full107 tot", cfg_all("G6_full107", "tot"), 0.719, abs_=0.005)

# ------------------------------------------------------- OOD (Table 3)
for tag, v in [("OOD_doc", 0.686), ("OOD_mismatch", 0.640), ("OOD_size10", 0.721),
               ("OOD_size", 0.676), ("OOD_size50", 0.703), ("OOD_external", 0.527),
               ("OOD_ext_fix", 0.658), ("OOD_india_fix", 0.664)]:
    check(f"OOD {tag} tot", cfg_all(tag, "tot"), v, abs_=0.005)
check("OOD_ext_fix diff", cfg_all("OOD_ext_fix", "diff"), 0.55, abs_=0.02)
check("OOD_ext_fix faith", cfg_all("OOD_ext_fix", "faith"), 0.73, abs_=0.02)
check("OOD_external faith", cfg_all("OOD_external", "faith"), 0.638, abs_=0.01)
check("OOD_india_fix diff", cfg_all("OOD_india_fix", "diff"), 0.62, abs_=0.02)
check("OOD_india_fix faith", cfg_all("OOD_india_fix", "faith"), 0.69, abs_=0.02)
check("OOD_india_fix FRE beg", cfg_lv("OOD_india_fix", "beginner", "fre_mean"), 63.6, abs_=0.1)
check("OOD_india_fix FRE int", cfg_lv("OOD_india_fix", "intermediate", "fre_mean"), 38.3, abs_=0.1)

# ------------------------------------------------------- baselines
check("B1 tot", cfg_all("B1", "tot"), 0.666, abs_=0.005)
check("B2 tot", cfg_all("B2", "tot"), 0.743, abs_=0.005)

# ------------------------------------------------------- ADV three levels
check("ADV FRE beg", cfg_lv("ADV", "beginner", "fre_mean"), 64.7, abs_=0.1)
check("ADV FRE int", cfg_lv("ADV", "intermediate", "fre_mean"), 25.9, abs_=0.1)
check("ADV FRE adv", cfg_lv("ADV", "advanced", "fre_mean"), 10.8, abs_=0.1)
check("ADV n", cfg_all("ADV", "n"), 150, abs_=0)
check("ADV_full tot", cfg_all("ADV_full", "tot"), 0.688, abs_=0.005)

# ------------------------------------------------------- verifier (Table 5)
v6 = ver.get("L3_max3", {})
check("ver G6 ECE", v6.get("ece"), 0.343, abs_=0.01)
check("ver G6 agree", v6.get("agree_frac"), 0.50, abs_=0.02)
check("ver G6 SC", (v6.get("self_consistency") or {}).get("verdict_agree"), 0.933, abs_=0.01)
vadv = ver.get("ADV", {}).get("levels", {})
check("ver ADV diff beg", (vadv.get("beginner") or {}).get("difficulty"), 3.68, abs_=0.05)
check("ver ADV diff int", (vadv.get("intermediate") or {}).get("difficulty"), 3.96, abs_=0.05)
check("ver ADV diff adv", (vadv.get("advanced") or {}).get("difficulty"), 4.94, abs_=0.05)
for cname, rho_exp in [("G6_pure_s11", 0.58), ("A_full", 0.54), ("ADV", 0.40)]:
    rho = ((ver.get(cname, {}).get("spearman") or {}).get("faithfulness_vs_faith") or {}).get("rho")
    check(f"ver faithfulness rho {cname}", rho, rho_exp, abs_=0.03)
check("ver B1 ECE", ver.get("B1", {}).get("ece"), 0.454, abs_=0.01)
check("ver A_full ECE", ver.get("A_full", {}).get("ece"), 0.488, abs_=0.01)

# ------------------------------------------------------- metrics (§4.8)
m = met.get("tags", {}).get("L3_max3", {}).get("levels", {})
check("SARI G6 beg", (m.get("beginner") or {}).get("sari"), 0.437, abs_=0.005)
check("SARI G6 int", (m.get("intermediate") or {}).get("sari"), 0.386, abs_=0.005)
m2 = met.get("tags", {}).get("M2", {}).get("levels", {}).get("intermediate")
check("FK G6 int", (m.get("intermediate") or {}).get("fk_grade"), 13.5, abs_=0.1)
check("FK M2 int", (m2 or {}).get("fk_grade"), 20.4, abs_=0.1)

print("\n=== CROSS-CHECK SUMMARY ===")
for st, label, detail in results:
    print(f"[{st:7s}] {label:30s} {detail}")
npass = sum(1 for r in results if r[0] == "PASS")
print(f"\nPASS {npass} / {len(results)}")
