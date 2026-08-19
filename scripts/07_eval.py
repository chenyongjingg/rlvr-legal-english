# -*- coding: utf-8 -*-
"""
07_eval.py — round-2 statistics (upgraded, task 23).
Input: results/*.jsonl (06_framework / 09_baselines outputs).
Covers:
  * descriptive stats per config x level (halluc from NLI contra when stored)
  * Kruskal-Wallis + one-way ANOVA (per level) + eta^2 across G1-G6
  * Dunn post-hoc (per level) + BH-FDR
  * paired Wilcoxon: G-chain, A_full vs each ablation, G6 vs OOD_*, G6_pure vs seed11
  * hallucination chi-square across G1-G6
  * radar-data export (per config x level reward parts)
Output: stats_legal.json + console markdown.
Usage: python 07_eval.py [results_dir]
"""
import os, sys, json, math
import statistics as st
import scipy.stats as sp
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RD = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\33378\aigc_wechat\results\legal"
OUT = os.path.join(RD, "stats_legal.json")

CONFIG = [
    ("G1", "Qwen3-4B 零样本"),
    ("G2", "G1 + RAG"),
    ("G3", "G2 + 多智能体"),
    ("G4", "G3 + RAG* (完整框架, Qwen3-4B)"),
    ("G5", "完整框架, Qwen3.5-4B+SFT"),
    ("G6", "完整框架, Qwen3.5-4B+GRPO(v2)"),
    ("A_full", "消融: 全奖励 (GRPO 短训)"),
    ("A_diff", "消融: 去 r_diff(难度)"),
    ("A_term", "消融: 去 r_term(术语)"),
    ("A_copy", "消融: 去 r_copy(反抄)"),
    ("A_faith", "消融: 去 r_faith(NLI)"),
    ("A_fmt", "消融: 去 r_fmt(格式)"),
    ("A_full_full_budget", "全预算消融: A_full (n=40/G=6/6ep)"),
    ("A_diff_full_budget", "全预算消融: A_diff (n=40/G=6/6ep)"),
    ("G6_pure", "纯生成器: GRPO(v2) 无RAG/agents"),
    ("G6_pure_s11", "纯生成器: GRPO(v2) seed11"),
    ("G6_pure_s21", "纯生成器: GRPO(v2) seed21"),
    ("G6_pure_s42", "纯生成器: GRPO(v2) seed42"),
    ("M2", "纯生成器: SFT"),
    ("ADV", "三档难度: GRPO(v2) 纯"),
    ("ADV_full", "三档难度: GRPO(v2) 完整框架"),
    ("G6_full107", "全量 test(107源): GRPO(v2) 完整框架"),
    ("OOD_doc", "OOD: 留出文档 2006/123"),
    ("OOD_mismatch", "OOD: 语料错配(oyez KB)"),
    ("OOD_size", "OOD: 20% KB"),
    ("OOD_size50", "OOD: 50% KB"),
    ("OOD_size10", "OOD: 10% KB"),
    ("OOD_external", "OOD: 外部语料 US Code"),
    ("OOD_ext_fix", "OOD: US Code 修正框架(匹配域)"),
    ("OOD_india_fix", "OOD: 印度判例 修正框架"),
    ("L1_max1", "循环深度: L1 (max 1 次尝试)"),
    ("L2_max2", "循环深度: L2 (max 2 次尝试)"),
    ("L3_max3", "循环深度: L3 (max 3 次尝试, =诚实G6)"),
    ("L5_max5", "循环深度: L5 (max 5 次尝试)"),
    ("B1", "Qwen3.5-9B 零样本基线"),
    ("B2", "词典法基线"),
]
# Main chain as reported in Table 1: G6 = CORRECTED framework (L3_max3, the
# decontamination-aware re-run, honest 0.800), NOT the original-framework G6
# (0.705) which appears only as the OOD / decontamination reference below.
MAIN = ["G1", "G2", "G3", "G4", "G5", "L3_max3"]
ABL = ["A_full", "A_diff", "A_term", "A_copy", "A_faith", "A_fmt"]
SUPPL = ["G6_pure", "G6_pure_s11", "G6_pure_s21", "G6_pure_s42", "M2",
         "ADV", "ADV_full", "G6_full107", "OOD_doc", "OOD_mismatch",
         "OOD_size", "OOD_size50", "OOD_size10", "OOD_external",
         "OOD_ext_fix", "OOD_india_fix",
         "L1_max1", "L2_max2", "L3_max3", "L5_max5",
         "A_full_full_budget", "A_diff_full_budget", "B1", "B2"]


def load(tag):
    # prefer the honest (commentary-cleaned) file when present; round-7 tags
    # (L1_max1..L5_max5, OOD_*_fix) are already produced clean, no _honest file.
    for suffix in ("_honest", ""):
        p = os.path.join(RD, f"{tag}{suffix}.jsonl")
        if os.path.isfile(p):
            return [json.loads(l) for l in open(p, encoding="utf-8")]
    return None


def grp(rows):
    lvs = sorted(set(r["lv"] for r in rows))
    return {lv: [r for r in rows if r["lv"] == lv] for lv in lvs}


def _val(r, mkey):
    if mkey == "tot":
        # Failure-weighted: pure-commentary rows score 0 (same semantics as
        # summ/honest_table); round-7 raw files otherwise leak the polluted tot.
        return 0.0 if _pure(r) else r["tot"]
    if mkey == "fre":
        return r["fre"]
    if mkey in r.get("parts", {}):
        return r["parts"][mkey]
    if mkey in r:
        return r[mkey]
    return None


def _pure(r):
    """A row is a pure-commentary failure if no legal rewrite survives cleaning."""
    if "pure_commentary" in r:
        return bool(r["pure_commentary"])
    return not bool(r.get("rew", ""))


def summ(rows):
    n = len(rows)
    if n == 0:
        return None
    # Failure-weighted tot (pure-commentary rows score 0), same semantics as
    # honest_table.py: _honest files bake in tot=0, but round-7 raw files
    # (L1-L5, OOD_*_fix) store the polluted framework tot on pure rows.
    tot = [0.0 if _pure(r) else r["tot"] for r in rows]
    at = [r.get("attempts", 1) for r in rows]
    pure_n = sum(1 for r in rows if _pure(r))
    stripped_n = sum(1 for r in rows if r.get("stripped"))
    valid = [r for r in rows if not _pure(r)]
    # FRE + reward parts over VALID (non-pure) rows only, matching
    # recompute_honest.py / honest_table.py: pure-commentary rows are generation
    # failures (reported as pure%, tot counts them as 0), not readable output.
    fre = [r["fre"] for r in valid if r.get("fre") is not None]
    parts = [r.get("parts", {}) for r in valid]
    # hallucination = mean NLI contradiction over VALID rewrites only (pure-
    # commentary failures are generation failures, reported separately as pure%).
    hall = 0.0
    for r in valid:
        nli = r.get("nli")
        if nli:
            if isinstance(nli, list) and len(nli) >= 3:
                hall += float(nli[2])
            else:
                hall += float(nli.get("contra", 0.0))
        else:
            hall += 1.0 if r.get("parts", {}).get("faith", 0.0) < 0.5 else 0.0
    return {
        "n": n, "valid_n": len(valid),
        "fre_mean": st.mean(fre) if fre else 0.0,
        "fre_med": st.median(fre) if fre else 0.0,
        "tot": st.mean(tot) if tot else 0.0,
        "diff": st.mean([p.get("diff", 0.0) for p in parts]) if parts else 0.0,
        "term": st.mean([p.get("term", 0.0) for p in parts]) if parts else 0.0,
        "copy": st.mean([p.get("copy", 0.0) for p in parts]) if parts else 0.0,
        "faith": st.mean([p.get("faith", 0.0) for p in parts]) if parts else 0.0,
        "halluc": hall / len(valid) if valid else 0.0,
        "pure": pure_n, "pure_rate": pure_n / n if n else 0.0,
        "stripped": stripped_n,
        "attempts": st.mean(at) if at else 1.0,
    }


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NaN"
    s = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return s + stars


def eps2(H, n, k):
    return H / ((n * n - 1.0) / (n + 1.0))


def anova_eta2(groups):
    """one-way ANOVA + eta^2. Returns (F, p, eta2, n_per_group)."""
    F, p = sp.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_tot = float(((allv - grand) ** 2).sum())
    ss_bet = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups)
    eta2 = ss_bet / ss_tot if ss_tot > 0 else 0.0
    return float(F), float(p), float(eta2)


def dunn_posthoc(groups):
    """Dunn's test (no tie correction). Returns (p_matrix, mean_ranks)."""
    k = len(groups)
    pooled = np.concatenate(groups)
    ranks = sp.rankdata(pooled)
    N = len(pooled)
    pos = 0
    mean_ranks, ns = [], []
    for g in groups:
        m = len(g)
        ns.append(m)
        mean_ranks.append(float(ranks[pos:pos + m].mean()))
        pos += m
    p = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            var = (N * (N + 1) / 12.0) * (1.0 / ns[i] + 1.0 / ns[j])
            z = (mean_ranks[i] - mean_ranks[j]) / math.sqrt(var) if var > 0 else 0.0
            pij = 2.0 * (1.0 - sp.norm.cdf(abs(z)))
            p[i, j] = pij; p[j, i] = pij
    return p, mean_ranks


def bh_fdr(pvals):
    p = np.asarray([max(1e-300, float(x)) for x in pvals], float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = np.arange(1, n + 1)
    q = p[order] * n / ranked
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return [float(x) for x in out]


def paired_wilcoxon(va, vb):
    d = np.asarray(vb, float) - np.asarray(va, float)
    if np.all(d == 0) or np.std(d) == 0:
        return 1.0, 0.0, 0.0
    w, p = sp.wilcoxon(va, vb, zero_method="wilcox")
    if np.isnan(p) or p == 0:
        p = 1e-300
    z = sp.norm.ppf(p / 2) * (1 if w >= 0 else -1)
    return float(p), float(w), float(abs(z) / math.sqrt(len(d)))


def aligned(rows_a, rows_b, mkey):
    da = {r["id"] + "\x00" + r["lv"]: r for r in rows_a}
    db = {r["id"] + "\x00" + r["lv"]: r for r in rows_b}
    kk = [k for k in da if k in db]
    va = np.array([_val(da[k], mkey) for k in kk], float)
    vb = np.array([_val(db[k], mkey) for k in kk], float)
    return va, vb


def desc_table(tags, data, stat):
    print("\n## 每配置 × 难度 描述统计")
    print("| cfg | lv | n | FRE_mean | FRE_med | tot | diff | term | copy | faith | 幻觉率 | pure% | attempts |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        rows = data[t]
        by = grp(rows)
        for lv in by:
            s = summ(by[lv])
            stat["cfg"].setdefault(t, {})[lv] = s
            print(f"| {t} | {lv[:4]} | {s['n']} | {s['fre_mean']:.1f} | {s['fre_med']:.1f} "
                  f"| {s['tot']:.3f} | {s['diff']:.2f} | {s['term']:.2f} | {s['copy']:.2f} "
                  f"| {s['faith']:.3f} | {s['halluc']:.3f} | {100*s['pure_rate']:.0f}% "
                  f"| {s['attempts']:.2f} |")
        sc = summ(rows)
        stat["cfg"][t]["all"] = sc
        print(f"| **{t}** | all | {sc['n']} | {sc['fre_mean']:.1f} | {sc['fre_med']:.1f} "
              f"| {sc['tot']:.3f} | {sc['diff']:.2f} | {sc['term']:.2f} | {sc['copy']:.2f} "
              f"| {sc['faith']:.3f} | {sc['halluc']:.3f} | {100*sc['pure_rate']:.0f}% "
              f"| {sc['attempts']:.2f} |")


def kw_block(tags, data, mlabel, metric_names, stat, key):
    """Kruskal-Wallis over the given configs on the common (id,lv) support."""
    keys = [{(r["id"], r["lv"]) for r in data[t]} for t in tags]
    common = set.intersection(*keys)
    print(f"\n## {mlabel} Kruskal-Wallis (n={len(common)} common samples)")
    for mkey, mn in metric_names.items():
        groups = []
        for t in tags:
            kv = {}
            for r in data[t]:
                if (r["id"], r["lv"]) in common:
                    kv[(r["id"], r["lv"])] = _val(r, mkey)
            groups.append([kv[k] for k in sorted(kv)])
        groups = [g for g in groups if g]
        if len(groups) < 2:
            continue
        H, p = sp.kruskal(*groups)
        stat[key][mkey] = {"H": float(H), "p": float(p),
                           "eps2": float(eps2(H, len(groups[0]) * len(tags), len(tags)))}
        print(f"  {mn}({mkey}): H={H:.2f} p={pfmt(p)}  eps²={eps2(H, len(groups[0])*len(tags), len(tags)):.3f}")


def anova_block(tags, data, mlabel, stat, key):
    """One-way ANOVA per level + Dunn post-hoc + BH-FDR (for the reward metric)."""
    keys = [{(r["id"], r["lv"]) for r in data[t]} for t in tags]
    common = set.intersection(*keys)
    levels = sorted(set(lv for _, lv in common))
    print(f"\n## {mlabel} 每难度 ANOVA + Dunn(α=0.05, BH-FDR) — 指标: tot")
    stat[key] = {"anova": {}, "dunn": {}}
    for lv in levels:
        groups = []
        for t in tags:
            vals = []
            for rid, rl in common:
                if rl != lv:
                    continue
                r = next((x for x in data[t] if x["id"] == rid and x["lv"] == rl), None)
                vals.append(_val(r, "tot"))
            groups.append(vals)
        groups = [g_ for g_ in groups if len(g_) == len(groups[0]) and len(g_)]
        if len(groups) < 2 or len(groups[0]) < 2:
            continue
        F, p, eta2 = anova_eta2(groups)
        stat[key]["anova"][lv] = {"F": F, "p": p, "eta2": eta2,
                                  "n_per": len(groups[0]), "k": len(groups)}
        print(f"  {lv}: F={F:.2f} p={pfmt(p)}  η²={eta2:.3f} (n={len(groups[0])}/group)")
        if p < 0.05 and len(groups) <= 10:
            pmat, mrs = dunn_posthoc(groups)
            pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
            pvals = [pmat[i, j] for i, j in pairs]
            qvals = bh_fdr(pvals)
            stat[key]["dunn"][lv] = {"mean_ranks": {tags[i]: mrs[i]
                                                    for i in range(len(tags))},
                                     "pairs": []}
            for (i, j), pv, qv in zip(pairs, pvals, qvals):
                stat[key]["dunn"][lv]["pairs"].append(
                    {"a": tags[i], "b": tags[j], "p": float(pv), "q": float(qv)})
                print(f"      {tags[i]} vs {tags[j]}: p={pfmt(pv)} q={pfmt(qv)}")


def main():
    data = {t: load(t) for t, _ in CONFIG}
    for t, lab in CONFIG:
        if data[t] is None:
            print(f"[warn] missing {t}.jsonl ({lab})", file=sys.stderr)
    data = {t: rows for t, rows in data.items() if rows is not None}

    stat = {"cfg": {}, "kw": {}, "anova": {}, "dunn": {}, "abl_kw": {},
            "abl_anova": {}, "abl_dunn": {}, "abl_contrasts": {},
            "contrasts": {}, "halluc_chi2": {}, "radar": {}, "seed_pair": {}}

    # ---- 主链: 仅 G1-G6 全部存在才做 ----
    main_tags = [t for t in MAIN if t in data]
    if len(main_tags) >= 2:
        desc_table(main_tags, data, stat)
        metric_names = {"tot": "总奖励", "fre": "Flesch", "faith": "NLI 蕴含",
                        "term": "术语保留", "attempts": "反馈轮数"}
        kw_block(main_tags, data, "跨配置", metric_names, stat, "kw")
        anova_block(main_tags, data, "跨配置", stat, "anova")

        # paired contrasts G-chain
        pairs = [("G1", "G2", "RAG 增益"), ("G2", "G3", "多智能体增益"),
                 ("G3", "G4", "RAG+多智能体组合"), ("G4", "G5", "模型升级(SFT)"),
                 ("G5", "L3_max3", "RLVR(GRPO)增益 (L3=诚实G6)")]
        print("\n## 相邻配置配对 Wilcoxon（按样本配对，两尾）")
        print("| 对比 | 指标 | 中位数差 | W | p | r |")
        print("|---|---|---|---|---|---|")
        all_pvals = []
        for a, b, lab in pairs:
            if a not in data or b not in data:
                continue
            for mkey in ("tot", "fre", "faith"):
                va, vb = aligned(data[a], data[b], mkey)
                if len(va) < 2:
                    continue
                p, w, r = paired_wilcoxon(va, vb)
                stat["contrasts"].setdefault(f"{a}-{b}", {"label": lab})[mkey] = {
                    "p": p, "W": float(w), "r": float(r),
                    "med_diff": float(np.median(vb - va))}
                all_pvals.append(p)
                print(f"| {a}→{b} ({lab}) | {mkey} | {np.median(vb-va):+.2f} | "
                      f"{w:.0f} | {pfmt(p)} | {r:.3f} |")
        if all_pvals:
            qs = bh_fdr(all_pvals)
            stat["contrasts"]["_bh_fdr"] = {"n": len(all_pvals),
                                            "qvals": [float(q) for q in qs]}

        # hallucination chi-square across the main chain (contra if stored else
        # faith proxy). nli is a dict {entail/neutral/contra} in *_honest files
        # but a list [entail, neutral, contra] in round-7 raw files; handle both.
        counts = []
        for t in main_tags:
            hall = 0.0
            for r in data[t]:
                nli = r.get("nli")
                if nli:
                    if isinstance(nli, list) and len(nli) >= 3:
                        hall += float(nli[2])
                    else:
                        hall += float(nli.get("contra", 0.0))
                elif r.get("parts", {}).get("faith", 0.0) < 0.5:
                    hall += 1.0
            counts.append([hall, len(data[t]) - hall])
        counts = np.array(counts)
        chi2, p_chi, df, _ = sp.chi2_contingency(counts)
        stat["halluc_chi2"] = {"chi2": float(chi2), "p": float(p_chi), "df": int(df),
                               "table": counts.tolist()}
        print(f"\n## 幻觉率跨配置卡方 (G1-G6)\n  表(幻觉,非幻觉): {counts.tolist()}\n"
              f"  chi2={chi2:.2f} df={df} p={pfmt(p_chi)}")
    else:
        # fallback: describe whatever exists
        present = [t for t, _ in CONFIG if t in data]
        if present:
            desc_table(present, data, stat)

    # ---- 消融块 ----
    abl_tags = [t for t in ABL if t in data]
    if "A_full" in data and len(abl_tags) >= 2:
        desc_table(abl_tags, data, stat)
        kw_block(abl_tags, data, "消融", {"tot": "总奖励", "diff": "难度匹配",
                                         "faith": "NLI 蕴含"}, stat, "abl_kw")
        anova_block(abl_tags, data, "消融", stat, "abl_anova")
        # paired A_full vs each ablation
        print("\n## 消融: A_full vs 各 leave-one-out（配对 Wilcoxon, tot）")
        print("| 对比 | 中位数差 | p | r |")
        print("|---|---|---|---|")
        for t in abl_tags:
            if t == "A_full":
                continue
            va, vb = aligned(data["A_full"], data[t], "tot")
            if len(va) < 2:
                continue
            p, w, r = paired_wilcoxon(va, vb)
            stat["abl_contrasts"][t] = {"p": p, "W": float(w), "r": float(r),
                                        "med_diff": float(np.median(vb - va))}
            print(f"| A_full vs {t} | {np.median(vb-va):+.3f} | {pfmt(p)} | {r:.3f} |")

    # ---- round-2/3 补充配置描述表（纯生成器/OOD/ADV/全量test/基线）----
    suppl_tags = [t for t in SUPPL if t in data]
    if suppl_tags:
        desc_table(suppl_tags, data, stat)

    # ---- 框架纯度 / OOD / 多 seed 对比 ----
    print("\n## 纯度 / OOD / 多 seed 配对对比")
    print("| 对比 | 指标 | 中位数差 | p | r |")
    print("|---|---|---|---|---|")
    extra = [("G6_pure", "G6", "纯生成器 vs 完整框架"),
             ("M2", "G6_pure", "SFT vs GRPO (纯)"),
             ("OOD_doc", "G6", "OOD: 留出文档"),
             ("OOD_mismatch", "G6", "OOD: 语料错配"),
             ("OOD_size", "G6", "OOD: 20% KB"),
             ("OOD_size50", "G6", "OOD: 50% KB"),
             ("OOD_size10", "G6", "OOD: 10% KB"),
             # round-7 循环深度（同 split+seed，可配对）
             ("L2_max2", "L1_max1", "循环 L2 vs L1 (更多尝试)"),
             ("L3_max3", "L2_max2", "循环 L3 vs L2"),
             ("L5_max5", "L3_max3", "循环 L5 vs L3"),
             ("L3_max3", "G6", "循环 L3 (诚实G6) vs 旧G6 重打分"),
             # OOD 修正框架 vs 旧版重打分（同 test 集）
             ("OOD_ext_fix", "OOD_external", "OOD US Code: 修正框架 vs 旧重打分"),
             # 全预算消融同 test 集可配对；外部语料 OOD_external 为不同语料，仅描述性比较
             ("A_diff_full_budget", "A_full_full_budget", "全预算: A_diff vs A_full")]
    for a, b, lab in extra:
        if a not in data or b not in data:
            continue
        for mkey in ("tot", "diff", "faith"):
            va, vb = aligned(data[a], data[b], mkey)
            if len(va) < 2:
                continue
            p, w, r = paired_wilcoxon(va, vb)
            stat["contrasts"].setdefault(f"{a}-{b}", {"label": lab})[mkey] = {
                "p": p, "W": float(w), "r": float(r),
                "med_diff": float(np.median(vb - va))}
            print(f"| {a} vs {b} ({lab}) | {mkey} | {np.median(vb-va):+.3f} | "
                  f"{pfmt(p)} | {r:.3f} |")

    seed_tags = [("G6_pure", "seed7"), ("G6_pure_s11", "seed11"),
                 ("G6_pure_s21", "seed21"), ("G6_pure_s42", "seed42")]
    present_seeds = [(t, lab) for t, lab in seed_tags if t in data]
    if len(present_seeds) >= 2:
        means = {}
        for t, lab in present_seeds:
            means[lab] = float(np.mean([r["tot"] for r in data[t]]))
        vals = list(means.values())
        stat["seed_pair"] = {
            "seeds": {lab: m for lab, m in means.items()},
            "n_seeds": len(vals),
            "mean_of_seed_means": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(min(vals)), "max": float(max(vals)),
            "range": float(max(vals) - min(vals))}
        print(f"\n多 seed (纯, {len(present_seeds)} seeds): " +
              ", ".join(f"{lab}={m:.3f}" for lab, m in means.items()) +
              f" | mean={np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ---- radar 导出（每配置 × 难度 的奖励分量均值）----
    radar_tags = [t for t in ("G1", "G2", "G3", "G4", "G5", "G6", "A_full",
                              "A_diff", "A_term", "A_copy", "A_faith", "A_fmt",
                              "G6_pure", "G6_pure_s21", "G6_pure_s42", "M2",
                              "ADV", "ADV_full", "G6_full107", "B1", "B2")
                  if t in data]
    for t in radar_tags:
        stat["radar"][t] = {}
        by = grp(data[t])
        for lv in by:
            parts = [r.get("parts", {}) for r in by[lv]]
            stat["radar"][t][lv] = {
                "diff": st.mean([p.get("diff", 0.0) for p in parts]),
                "term": st.mean([p.get("term", 0.0) for p in parts]),
                "copy": st.mean([p.get("copy", 0.0) for p in parts]),
                "faith": st.mean([p.get("faith", 0.0) for p in parts]),
                "fmt": st.mean([p.get("fmt", 0.0) for p in parts]),
            }

    json.dump(stat, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n统计 JSON -> {OUT}")


if __name__ == "__main__":
    main()
