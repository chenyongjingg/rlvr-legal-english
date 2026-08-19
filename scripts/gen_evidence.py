# -*- coding: utf-8 -*-
"""Build the evidence record (source_manifest / claims / consistency) for the
legal-English framework manuscript, following scientific-writing skill rules:
  * claims.csv stores SHA-256 of normalized claim text (not raw text)
  * no source is marked verified (human verification pending)
  * every numeric/result record binds evidence IDs
Run: python gen_evidence.py  (writes into D:/周老师/paper/)
"""
import json, hashlib, csv, os

PAPER = r"D:\周老师\paper"
MIRROR = r"C:\Users\33378\aigc_wechat\results\legal"
SRV = "/root/autodl-tmp/legal_english"


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def norm(t):
    return " ".join(t.split())


# ---------------- source manifest ----------------
LOCAL = {
    "stats": ("https://local-file", f"{MIRROR}\\stats_legal.json", f"{SRV}/results/stats_legal.json"),
}
sources = [
    {"e": "E001", "t": "stats_legal.json — 描述统计/统计检验/seed_pair/radar（07_eval 产物，25 cfg）",
     "s": "other", "loc": f"{MIRROR}\\stats_legal.json ; {SRV}/results/stats_legal.json"},
    {"e": "E002", "t": "verifier_eval.json — 跨族裁判聚合（per-level 4维均分/ECE/自一致性/Spearman）",
     "s": "other", "loc": f"{MIRROR}\\verifier_eval.json ; {SRV}/results/verifier_eval.json"},
    {"e": "E003", "t": "metrics_legal.json — SARI/BLEU/rich 可读性/严格矛盾率/效率（10_metrics 产物）",
     "s": "other", "loc": f"{MIRROR}\\metrics_legal.json ; {SRV}/results/metrics_legal.json"},
    {"e": "E004", "t": "verifier_{tag}.jsonl — 逐样本裁判分（G6/G1/B1/ADV/A_full/G6_pure_s11/OOD_doc）",
     "s": "other", "loc": f"{MIRROR}\\verifier_*.jsonl ; {SRV}/results/verifier_*.jsonl"},
    {"e": "E005", "t": "results/{tag}.jsonl — 逐样本生成结果（G1-G6/G6_pure*/M2/ADV*/OOD_*/B1/B2/A_*）",
     "s": "other", "loc": f"{MIRROR}\\*.jsonl ; {SRV}/results/*.jsonl"},
    {"e": "E006", "t": "05_grpo_train.py — GRPO/RLVR 训练与奖励消融（--drop，降预算）",
     "s": "other", "loc": f"{SRV}/scripts/05_grpo_train.py"},
    {"e": "E007", "t": "06_framework.py — 评测框架（RAG/agents/NLI 3分类/三档难度/OOD flags）",
     "s": "other", "loc": f"{SRV}/scripts/06_framework.py"},
    {"e": "E008", "t": "07_eval.py — 统计（KW/ANOVA/Dunn/BH-FDR/配对Wilcoxon/4-seed）",
     "s": "other", "loc": f"{SRV}/scripts/07_eval.py"},
    {"e": "E009", "t": "08_verifier_eval.py — Verifier-as-Judge（Qwen3-4B 跨族裁判）",
     "s": "other", "loc": f"{SRV}/scripts/08_verifier_eval.py"},
    {"e": "E010", "t": "09_baselines.py — B1 9B零样本 / B2 词典法基线",
     "s": "other", "loc": f"{SRV}/scripts/09_baselines.py"},
    {"e": "E011", "t": "10_metrics.py — 离线指标（SARI/BLEU/rich可读性/矛盾率/效率）",
     "s": "other", "loc": f"{SRV}/scripts/10_metrics.py"},
    {"e": "E012", "t": "corpus_v2.json + split.json — EUR-Lex/Wex/Oyez 968 snippets + 49 术语，80/10/10 划分",
     "s": "other", "loc": f"{SRV}/data/corpus_v2.json ; {SRV}/data/split.json"},
    {"e": "E013", "t": "模型与适配器（Qwen3.5-4B + grpo_qwen35_lora_v2；Qwen3-4B-Instruct-2507 裁判；NLI-DeBERTa-v3；MiniLM-L6-v2）",
     "s": "other", "loc": f"{SRV}/models/*"},
    {"e": "E014", "t": "论文设计/计划书 demo_v2026_Neurocomputing_v2.docx（研究设计、RQ、目标难度）",
     "s": "other", "loc": r"D:\周老师\demo_v2026_Neurocomputing_v2.docx"},
]

# Related-work reference sources (E015-E024): real papers located via web search.
# Identifiers recorded so a human can open each; ALL remain unverified by design.
ref_sources = [
    {"e": "E015", "s": "journal_article",
     "t": "F. Alva-Manchego, C. Scarton, L. Specia, Data-Driven Sentence Simplification: Survey and Benchmark, Comput. Linguist. 46(1):135-187, 2020",
     "loc": "https://doi.org/10.1162/COLI_a_00370",
     "doi": "10.1162/COLI_a_00370", "url": "https://direct.mit.edu/coli/article/46/1/135/93232",
     "authors": ["Fernando Alva-Manchego", "Carolina Scarton", "Lucia Specia"], "year": 2020},
    {"e": "E016", "s": "journal_article",
     "t": "W. Xu, C. Napoles, E. Pavlick, Q. Chen, C. Callison-Burch, Optimizing Statistical Machine Translation for Text Simplification, TACL 4:401-415, 2016",
     "loc": "https://doi.org/10.1162/tacl_a_00107",
     "doi": "10.1162/tacl_a_00107", "url": "https://transacl.org/ojs/index.php/tacl/article/view/896",
     "authors": ["Wei Xu", "Courtney Napoles", "Ellie Pavlick", "Quanze Chen", "Chris Callison-Burch"], "year": 2016},
    {"e": "E017", "s": "preprint",
     "t": "Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y.K. Li, Y. Wu, D. Guo, DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv:2402.03300, 2024",
     "loc": "https://arxiv.org/abs/2402.03300",
     "doi": "", "url": "https://arxiv.org/abs/2402.03300",
     "authors": ["Zhihong Shao", "Peiyi Wang", "Qihao Zhu", "Runxin Xu", "Junxiao Song", "Xiao Bi", "Haowei Zhang", "Mingchuan Zhang", "Y. K. Li", "Y. Wu", "Daya Guo"], "year": 2024},
    {"e": "E018", "s": "conference_paper",
     "t": "P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rocktaeschel, S. Riedel, D. Kiela, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020",
     "loc": "https://arxiv.org/abs/2005.11401",
     "doi": "", "url": "https://arxiv.org/abs/2005.11401",
     "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Kuttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"], "year": 2020},
    {"e": "E019", "s": "conference_paper",
     "t": "J. Niklaus, L. Zheng, A.D. McCarthy, C. Hahn, B.M. Rosen, P. Henderson, D.E. Ho, G. Honke, P. Liang, C.D. Manning, LawInstruct: A Resource for Studying Language Model Adaptation to the Legal Domain, Findings of NAACL 2025, pp.127-152",
     "loc": "https://aclanthology.org/2025.findings-naacl.7/",
     "doi": "10.18653/v1/2025.findings-naacl.7", "url": "https://aclanthology.org/2025.findings-naacl.7/",
     "authors": ["Joel Niklaus", "Lucia Zheng", "Arya D. McCarthy", "Christopher Hahn", "Brian M. Rosen", "Peter Henderson", "Daniel E. Ho", "Garrett Honke", "Percy Liang", "Christopher D. Manning"], "year": 2025},
    {"e": "E020", "s": "conference_paper",
     "t": "F. Alva-Manchego, R. Stodden, J.M. Imperial, A. Barayan, K. North, H. Tayyar Madabushi, Findings of the TSAR 2025 Shared Task on Readability-Controlled Text Simplification, TSAR@EMNLP 2025, pp.116-130",
     "loc": "https://aclanthology.org/2025.tsar-1.8/",
     "doi": "10.18653/v1/2025.tsar-1.8", "url": "https://aclanthology.org/2025.tsar-1.8/",
     "authors": ["Fernando Alva-Manchego", "Regina Stodden", "Joseph Marvin Imperial", "Abdullah Barayan", "Kai North", "Harish Tayyar Madabushi"], "year": 2025},
    {"e": "E021", "s": "conference_paper",
     "t": "Z. Wan, Y. Zhang, Y. Wang, F. Cheng, S. Kurohashi, Reformulating Domain Adaptation of Large Language Models as Adapt-Retrieve-Revise: A Case Study on Chinese Legal Domain, Findings of ACL 2024, pp.5030-5041",
     "loc": "https://aclanthology.org/2024.findings-acl.299/",
     "doi": "10.18653/v1/2024.findings-acl.299", "url": "https://aclanthology.org/2024.findings-acl.299/",
     "authors": ["Zhen Wan", "Yating Zhang", "Yexiang Wang", "Fei Cheng", "Sadao Kurohashi"], "year": 2024},
    {"e": "E022", "s": "preprint",
     "t": "N. Guha, J. Nyarko, D.E. Ho, et al., LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models, arXiv:2308.11462, 2023",
     "loc": "https://arxiv.org/abs/2308.11462",
     "doi": "", "url": "https://arxiv.org/abs/2308.11462",
     "authors": ["Neel Guha", "Julian Nyarko", "Daniel E. Ho"], "year": 2023},
    {"e": "E023", "s": "journal_article",
     "t": "Y. Zuo, Automatic generation of ESL learning materials based on CEFR levels using reinforcement-tuned LLMs, Discover Artificial Intelligence 6 (Springer), 2025",
     "loc": "https://doi.org/10.1007/s44163-025-00762-3",
     "doi": "10.1007/s44163-025-00762-3", "url": "https://link.springer.com/article/10.1007/s44163-025-00762-3",
     "authors": ["Yi Zuo"], "year": 2025},
    {"e": "E024", "s": "conference_paper",
     "t": "H. Sun, Y. Chai, S. Wang, Y. Sun, H. Wu, H. Wang, Curiosity-Driven Reinforcement Learning from Human Feedback, ACL 2025 Long Papers, pp.23517-23534",
     "loc": "https://aclanthology.org/2025.acl-long.1146/",
     "doi": "", "url": "https://aclanthology.org/2025.acl-long.1146/",
     "authors": ["Haoran Sun", "Yekun Chai", "Shuohuan Wang", "Yu Sun", "Hua Wu", "Haifeng Wang"], "year": 2025},
]

src_manifest = {
    "schema_version": "1.0",
    "sources": [
        {
            "evidence_id": s["e"],
            "source_type": s["s"],
            "title": s["t"],
            "locator": s["loc"],
            "identifiers": {"doi": "", "isbn": "", "pmcid": "", "pmid": "", "url": ""},
            "authors": [],
            "year": None,
            "confidentiality": "public",
            "verification": {"source_opened": False, "status": "unverified",
                             "verified_by": "", "verified_on": ""},
        }
        for s in sources
    ]
    + [
        {
            "evidence_id": s["e"],
            "source_type": s["s"],
            "title": s["t"],
            "locator": s["loc"],
            "identifiers": {"doi": s.get("doi", ""), "isbn": "", "pmcid": "", "pmid": "",
                            "url": s.get("url", "")},
            "authors": s.get("authors", []),
            "year": s.get("year"),
            "confidentiality": "public",
            "verification": {"source_opened": False, "status": "unverified",
                             "verified_by": "", "verified_on": ""},
        }
        for s in ref_sources
    ],
}
json.dump(src_manifest, open(os.path.join(PAPER, "source_manifest.json"), "w",
          encoding="utf-8"), ensure_ascii=False, indent=2)


# ---------------- claims.csv ----------------
# (claim_id, section, claim_kind, claim_text, evidence_ids, analysis_intent, uncertainty)
claims = [
    ("C001", "results", "factual",
     "G6（完整框架 + GRPO v2）在 100 条测试集上总奖励 0.884，为 G1–G6 全场最高。", "E001", "confirmatory", "moderate"),
    ("C002", "results", "factual",
     "G6 beginner FRE 64.3（目标 63，带 [43,83]）、intermediate FRE 38.1（目标 45，带 [25,65]），均在 ±20 目标带内。", "E001", "confirmatory", "moderate"),
    ("C003", "results", "factual",
     "GRPO/RLVR 是核心贡献：G5→G6 总奖励 0.806→0.884（配对 Wilcoxon p=9.2e-12, r=0.68, 中位数差 +0.08）。", "E001", "confirmatory", "low"),
    ("C004", "results", "factual",
     "干净同模型对比（均 Qwen3.5-4B 纯协议）：G6_pure 总奖励 0.883 显著高于 SFT 基线 M2（tot p=1.2e-12, r=0.71；diff p=7.4e-6；faith p=6.3e-9），无模型身份混杂。", "E001", "confirmatory", "low"),
    ("C005", "results", "factual",
     "三档难度完整框架（ADV_full，150 条）FRE 单调分离 beginner 65.0 / intermediate 37.5 / advanced 30.6，均落在各自目标带内。", "E001", "confirmatory", "moderate"),
    ("C006", "results", "factual",
     "OOD 稳健性：四种分布偏移（文档留出 OOD_doc、语料错配 OOD_mismatch、KB 缩容、外部语料 US Code）下难度控制均保持。KB 规模 10%/20%/50%/100% 四点 tot 0.867/0.877/0.873/0.884，vs 全量配对 p(tot)=0.039/0.392/0.212——仅 10% 点小幅显著回落（中位数差 −0.020，集中于 diff p=0.060，faith p=0.196 保持），20%/50% 无显著差异，FRE 全落 ±20 目标带；外部语料 OOD_external tot 0.841 = 0.884 的 95.2% 保留，beginner/intermediate FRE 62.5/42.8 落目标带。", "E001", "exploratory", "low"),
    ("C007", "results", "factual",
     "4-seed（7/11/21/42）纯生成器总奖励均值 0.853 ± 0.009（std），范围 0.838–0.864，seed 敏感性低。", "E001", "exploratory", "low"),
    ("C008", "results", "factual",
     "全量 test（107 源×2 级 = 214 行）G6_full107 总奖励 0.869 / diff 0.586 / faith 0.843 / 严格矛盾率 0.033，与 100 条子集 headline 一致，主结果稳健复现。", "E001", "confirmatory", "moderate"),
    ("C009", "results", "factual",
     "独立跨族裁判（Qwen3-4B）对 ADV 三档的 difficulty 均分 3.68→3.96→4.94 单调递增，独立于 FRE 验证难度梯度（RQ1）。", "E002", "confirmatory", "moderate"),
    ("C010", "results", "factual",
     "纯协议/消融配置上裁判 faithfulness 与自动 NLI faith 显著正相关（rho 0.40–0.58，p<1e-6），低矛盾率非自动指标假象。", "E002", "exploratory", "low"),
    ("C011", "results", "factual",
     "基线对比：G6 总奖励领先 9B 零样本 B1（0.694/0.733）约 +0.15–0.18；词典法 B2 严格矛盾率全场最高（0.248/0.218），G6 为 0.031/0.024（低一个数量级）。", "E001", "exploratory", "moderate"),
    ("C012", "results", "factual",
     "SFT 单独破坏难度控制：G5 beginner FRE 76.2 / intermediate 15.0（远离目标），GRPO 修复（G6 64.3/38.1）。", "E001", "confirmatory", "moderate"),
    ("C013", "results", "factual",
     "奖励消融（生成器层）：留一法移除任一奖励项均不崩（无单一命门）。降预算（n=24/G=4/4ep）下去 r_diff 反而提升总奖励（A_diff vs A_full tot p=0.0037，0.817 vs 0.776），揭示难度-语义训练期权衡；足量预算（n=40/G=6/6ep，从 SFT 起）下该权衡被化解：A_diff_full_budget vs A_full_full_budget 总奖励 0.828 vs 0.835（配对 p=0.42 不显著），而 A_full 难度匹配显著更优（diff p=0.0001、intermediate FRE 25.0 vs 1.7）、A_diff 语义保持更高（faith p=0.0025）；端到端 G6（含 RAG/agents）兼得两者（tot 0.884、diff 0.625、faith 0.857）。", "E001", "exploratory", "low"),
    ("C014", "results", "factual",
     "严格矛盾率（NLI 矛盾概率）G6 ~2–3%（0.031/0.024），远低于消融（0.05–0.12）与基线（B1 0.15–0.20、B2 0.22–0.25）。", "E003", "confirmatory", "moderate"),
    ("C015", "results", "factual",
     "SARI（ref=9B 教师单条机器简化）：G6 为 0.436/0.413 主要配置最高；BLEU 全表极低（~1e-4）因对合法改写惩罚性，论文以 SARI 为主。", "E003", "exploratory", "moderate"),
    ("C016", "results", "factual",
     "QLoRA（r=16, α=32, 7 模块）可训练参数 21.23M，约为 Qwen3.5-4B 文本模块的 0.8%（sft_train.log 实测 21,233,664 / 2,611,327,488 = 0.813%）。", "E003;E013", "confirmatory", "moderate"),
    ("C017", "results", "factual",
     "生成效率：完整框架 ~16–30 s/条，纯生成器 ~6–17 s/条，词典法 B2 0.02 s/条。", "E003", "exploratory", "moderate"),
    ("C018", "methods", "factual",
     "奖励栈 W={格式0.15, 难度0.20, 术语0.15, 反抄袭0.25, 语义0.25}；GRPO v2 精调 diff-w=0.30, n=40, G=6, 6 epochs, seed 42。", "E006", "confirmatory", "not_applicable"),
    ("C019", "methods", "factual",
     "RAG 知识库 = train+val 861 条段落（防测试泄漏），MiniLM-L6-v2 稠密索引；测试集 100 条（beginner 50 + intermediate 50，seed 7, n_test=50）。", "E012", "confirmatory", "not_applicable"),
    ("C020", "methods", "factual",
     "幻觉检测用 DeBERTa-v3 3 分类（蕴含/中立/矛盾），严格矛盾率 = NLI 矛盾概率均值；NLI 保留完整概率。", "E007", "confirmatory", "not_applicable"),
    ("C021", "results", "factual",
     "裁判校准中等：G6 ECE 0.306 / 与自动一致率 0.570 / 自一致性(K=3) 0.900；弱基线 ECE 明显更差（B1 0.454, A_full 0.488）；框架配置自一致性 0.90–1.00。", "E002", "exploratory", "moderate"),
    ("C022", "results", "factual",
     "多智能体（G2→G3）faith 0.87→0.95（p=1.6e-7, r=0.52），幻觉率 0.06→0.03。", "E001", "confirmatory", "low"),
    ("C023", "results", "factual",
     "RAG（G1→G2）beginner FRE 56.8→62.3 逼近目标（p=0.002），faith 轻微下降（p=0.002）。", "E001", "confirmatory", "low"),
    ("C024", "results", "factual",
     "全配置 tot 差异显著（Kruskal-Wallis H=69.56, p=1.3e-13, ε²=0.116）；beginner/intermediate ANOVA 均显著（η²=0.159/0.105）；幻觉计数跨 G1–G6 卡方不显著（χ²=6.11, p=0.296）。", "E001", "confirmatory", "low"),
    ("C025", "results", "factual",
     "裁判 difficulty 与自动 FRE-based diff 逐样本相关弱/负（多数 ns 或负显著），两套难度口径互补而非等价；但层级间分离一致。", "E002", "exploratory", "low"),
    # ---- Related-Work citation claims (C026-C035): one per reference source ----
    ("C026", "related_work", "factual",
     "Alva-Manchego, Scarton and Specia survey data-driven sentence simplification; SARI is the standard simplification metric.", "E015", "descriptive", "not_applicable"),
    ("C027", "related_work", "factual",
     "Xu et al. introduce the SARI evaluation metric for text simplification.", "E016", "descriptive", "not_applicable"),
    ("C028", "related_work", "factual",
     "Shao et al. introduce GRPO, the critic-free group-relative policy objective used for RLVR here.", "E017", "descriptive", "not_applicable"),
    ("C029", "related_work", "factual",
     "Lewis et al. introduce retrieval-augmented generation (RAG) for grounding generation in external knowledge.", "E018", "descriptive", "not_applicable"),
    ("C030", "related_work", "factual",
     "Niklaus et al. release LawInstruct, a large legal-domain instruction-tuning resource across jurisdictions.", "E019", "descriptive", "not_applicable"),
    ("C031", "related_work", "factual",
     "The TSAR shared task on readability-controlled simplification shows prompting alone is unreliable for target levels and difficulty-fidelity trade-offs persist.", "E020", "descriptive", "not_applicable"),
    ("C032", "related_work", "factual",
     "Wan et al. adapt-retrieve-revise legal LLMs to prevent hallucination in the Chinese legal domain.", "E021", "descriptive", "not_applicable"),
    ("C033", "related_work", "factual",
     "Guha et al. build LegalBench, a broad collaborative legal reasoning benchmark for LLMs.", "E022", "descriptive", "not_applicable"),
    ("C034", "related_work", "factual",
     "Reinforcement-tuned small models generate CEFR-aligned learning materials and beat larger zero-shot baselines.", "E023", "descriptive", "not_applicable"),
    ("C035", "related_work", "factual",
     "Curiosity-driven RLHF addresses the diversity-alignment trade-off of reward-tuned text generation.", "E024", "descriptive", "not_applicable"),
]

with open(os.path.join(PAPER, "claims.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["claim_id", "section", "claim_kind", "claim_text_sha256",
                "evidence_ids", "verification_status", "uncertainty", "analysis_intent"])
    for cid, sec, kind, text, ev, intent, unc in claims:
        w.writerow([cid, sec, kind, sha(norm(text)), ev, "unverified", unc, intent])

# ---------------- consistency manifest ----------------
cm = {
    "schema_version": "1.0",
    "methods": [
        {"method_id": "M001", "name": "GRPO/RLVR 训练（Qwen3.5-4B + QLoRA r16/α32/7模块；五维可验证奖励栈；GRPO v2 diff-w0.30 n=40 G=6 6ep seed42；消融降预算 n=24 G=4 4ep seed42）",
         "protocol_status": "prespecified", "analysis_intent": "confirmatory", "outcome_ids": ["O007"]},
        {"method_id": "M002", "name": "框架评测（06_framework：RAG + 多智能体；NLI 3 分类矛盾率；三档难度目标 63/45/26 ±20；OOD flags；纯协议无 RAG/agents）",
         "protocol_status": "prespecified", "analysis_intent": "confirmatory", "outcome_ids": ["O001", "O002"]},
        {"method_id": "M003", "name": "统计方法（07_eval：Kruskal-Wallis + ANOVA/η² + Dunn/BH-FDR + 配对 Wilcoxon + 幻觉卡方 + 4-seed mean±std + radar）",
         "protocol_status": "prespecified", "analysis_intent": "confirmatory", "outcome_ids": ["O001"]},
        {"method_id": "M004", "name": "Verifier-as-Judge（08_verifier_eval：Qwen3-4B-Instruct-2507 跨族裁判，4 维 1-5 + 置信度；ECE 代理 vs auto tot≥0.80；K=3 自一致性 40 条；Spearman）",
         "protocol_status": "post_hoc", "analysis_intent": "exploratory", "outcome_ids": ["O005"]},
        {"method_id": "M005", "name": "离线指标（10_metrics：SARI vs B1 参考；BLEU；FK/Fog/TTR/词长/术语密度；严格矛盾率重算；效率表）",
         "protocol_status": "post_hoc", "analysis_intent": "exploratory", "outcome_ids": ["O006"]},
        {"method_id": "M006", "name": "外部基线（09_baselines：B1 = Qwen3.5-9B 零样本 + strip_thinking；B2 = 词典法）",
         "protocol_status": "prespecified", "analysis_intent": "exploratory", "outcome_ids": ["O001"]},
        {"method_id": "M007", "name": "数据构建（01-03：EUR-Lex/Wex/Oyez → corpus_v2.json 968 snippets + 49 术语；split 80/10/10 按 id 哈希）",
         "protocol_status": "not_applicable", "analysis_intent": "descriptive", "outcome_ids": ["O008"]},
        {"method_id": "M008", "name": "OOD 与多 seed 评测（06_framework OOD flags：文档留出/语料错配/KB 缩容；4-seed 7/11/21/42 纯协议）",
         "protocol_status": "post_hoc", "analysis_intent": "exploratory", "outcome_ids": ["O001"]},
    ],
    "numeric_facts": [
        {"fact_id": "N001", "concept": "G6 全量 100 条测试集总奖励", "value": 0.8835, "unit": "reward", "sample_size": 100, "denominator": None, "numerator": None, "analysis_set": "test(100)", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N002", "concept": "G6 全量 FRE 均值", "value": 51.2, "unit": "Flesch", "sample_size": 100, "denominator": None, "numerator": None, "analysis_set": "test(100)", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N003", "concept": "G6 难度匹配 diff", "value": 0.625, "unit": "reward", "sample_size": 100, "denominator": None, "numerator": None, "analysis_set": "test(100)", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N004", "concept": "G6 beginner FRE 均值", "value": 64.2886, "unit": "Flesch", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(100)/beginner", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N005", "concept": "G5→G6 GRPO 配对 p", "value": 9.22e-12, "unit": "p-value", "sample_size": 100, "denominator": None, "numerator": None, "analysis_set": "test(100)", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N006", "concept": "G6_pure vs M2 配对 p（tot）", "value": 1.24e-12, "unit": "p-value", "sample_size": 100, "denominator": None, "numerator": None, "analysis_set": "test(100)", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N007", "concept": "ADV_full beginner FRE", "value": 65.0, "unit": "Flesch", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(150)/beginner", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N008", "concept": "4-seed 总奖励 mean", "value": 0.853049, "unit": "reward", "sample_size": 4, "denominator": None, "numerator": None, "analysis_set": "test(100)/4seeds", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N009", "concept": "G6_full107 全量 test 总奖励", "value": 0.8693, "unit": "reward", "sample_size": 214, "denominator": None, "numerator": None, "analysis_set": "test(107×2)", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N010", "concept": "G6 严格矛盾率（contra）", "value": 0.027, "unit": "probability", "sample_size": 100, "denominator": None, "numerator": None, "analysis_set": "test(100)", "section": "results", "evidence_ids": ["E003"]},
        {"fact_id": "N011", "concept": "ADV 裁判 difficulty beginner", "value": 3.68, "unit": "1-5", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(150)/beginner", "section": "results", "evidence_ids": ["E002"]},
        {"fact_id": "N012", "concept": "QLoRA 可训练参数", "value": 18350080, "unit": "params", "sample_size": None, "denominator": None, "numerator": None, "analysis_set": "Qwen3.5-4B", "section": "results", "evidence_ids": ["E003"]},
        {"fact_id": "N013", "concept": "B2 词典法 beginner 严格矛盾率", "value": 0.248, "unit": "probability", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(100)/beginner", "section": "results", "evidence_ids": ["E003"]},
        {"fact_id": "N014", "concept": "G6 beginner SARI", "value": 0.436, "unit": "SARI", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(100)/beginner", "section": "results", "evidence_ids": ["E003"]},
        {"fact_id": "N015", "concept": "RAG KB 段落数（train+val）", "value": 861, "unit": "passages", "sample_size": None, "denominator": None, "numerator": None, "analysis_set": "KB", "section": "methods", "evidence_ids": ["E012"]},
        {"fact_id": "N016", "concept": "ADV_full intermediate FRE", "value": 37.5, "unit": "Flesch", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(150)/intermediate", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N017", "concept": "ADV_full advanced FRE", "value": 30.6, "unit": "Flesch", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(150)/advanced", "section": "results", "evidence_ids": ["E001"]},
        {"fact_id": "N018", "concept": "ADV 裁判 difficulty intermediate", "value": 3.96, "unit": "1-5", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(150)/intermediate", "section": "results", "evidence_ids": ["E002"]},
        {"fact_id": "N019", "concept": "ADV 裁判 difficulty advanced", "value": 4.94, "unit": "1-5", "sample_size": 50, "denominator": None, "numerator": None, "analysis_set": "test(150)/advanced", "section": "results", "evidence_ids": ["E002"]},
        {"fact_id": "N020", "concept": "语料规模 snippets", "value": 968, "unit": "snippets", "sample_size": None, "denominator": None, "numerator": None, "analysis_set": "corpus_v2", "section": "methods", "evidence_ids": ["E012"]},
    ],
    "results": [
        {"result_id": "R001", "method_id": "M002", "outcome_id": "O001", "sample_size": 100, "reported_sections": ["results"], "analysis_intent": "confirmatory", "evidence_ids": ["E001"]},
        {"result_id": "R002", "method_id": "M002", "outcome_id": "O002", "sample_size": 100, "reported_sections": ["results"], "analysis_intent": "confirmatory", "evidence_ids": ["E001"]},
        {"result_id": "R003", "method_id": "M003", "outcome_id": "O001", "sample_size": 100, "reported_sections": ["results"], "analysis_intent": "confirmatory", "evidence_ids": ["E001"]},
        {"result_id": "R004", "method_id": "M002", "outcome_id": "O002", "sample_size": 150, "reported_sections": ["results"], "analysis_intent": "confirmatory", "evidence_ids": ["E001"]},
        {"result_id": "R005", "method_id": "M008", "outcome_id": "O001", "sample_size": 226, "reported_sections": ["results"], "analysis_intent": "exploratory", "evidence_ids": ["E001"]},
        {"result_id": "R006", "method_id": "M006", "outcome_id": "O001", "sample_size": 200, "reported_sections": ["results"], "analysis_intent": "exploratory", "evidence_ids": ["E001"]},
        {"result_id": "R007", "method_id": "M008", "outcome_id": "O001", "sample_size": 400, "reported_sections": ["results"], "analysis_intent": "exploratory", "evidence_ids": ["E001"]},
        {"result_id": "R008", "method_id": "M004", "outcome_id": "O005", "sample_size": 1500, "reported_sections": ["results"], "analysis_intent": "exploratory", "evidence_ids": ["E002"]},
        {"result_id": "R009", "method_id": "M005", "outcome_id": "O006", "sample_size": 100, "reported_sections": ["results"], "analysis_intent": "exploratory", "evidence_ids": ["E003"]},
        {"result_id": "R010", "method_id": "M002", "outcome_id": "O001", "sample_size": 214, "reported_sections": ["results"], "analysis_intent": "confirmatory", "evidence_ids": ["E001"]},
        {"result_id": "R011", "method_id": "M001", "outcome_id": "O007", "sample_size": 40, "reported_sections": ["methods"], "analysis_intent": "confirmatory", "evidence_ids": ["E006"]},
        {"result_id": "R012", "method_id": "M007", "outcome_id": "O008", "sample_size": 968, "reported_sections": ["methods"], "analysis_intent": "descriptive", "evidence_ids": ["E012"]},
    ],
}
json.dump(cm, open(os.path.join(PAPER, "consistency_manifest.json"), "w",
          encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"wrote source_manifest.json ({len(sources)} sources), "
      f"claims.csv ({len(claims)} claims, sha256 stored), "
      f"consistency_manifest.json ({len(cm['methods'])} methods, "
      f"{len(cm['numeric_facts'])} numeric_facts, {len(cm['results'])} results)")
