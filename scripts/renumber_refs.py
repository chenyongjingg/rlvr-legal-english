# -*- coding: utf-8 -*-
"""Renumber the manuscript References [1]-[46] by first appearance and update
render_submission.py's CITE_MAP. Run from D:\\周老师\\paper.

The mapping (E-id -> reference number) is computed from the current in-text
[@E0xx] markers; the reference list below is hand-formatted Elsevier style.
"""
import re

PAPER = r"D:\周老师\paper"
MANUSCRIPT = PAPER + r"\manuscript.md"
RENDER = PAPER + r"\scripts\render_submission.py"

REFS = [
    # 1
    "F. Alva-Manchego, C. Scarton, L. Specia, Data-driven sentence simplification: survey and benchmark, Comput. Linguist. 46 (1) (2020) 135\u2013187. doi:10.1162/COLI_a_00370.",
    # 2
    "J. Niklaus, L. Zheng, A.D. McCarthy, C. Hahn, B.M. Rosen, P. Henderson, D.E. Ho, G. Honke, P. Liang, C.D. Manning, LawInstruct: A resource for studying language model adaptation to the legal domain, in: Findings of the Association for Computational Linguistics: NAACL, 2025, pp. 127\u2013152. doi:10.18653/v1/2025.findings-naacl.7.",
    # 3
    "N. Guha, J. Nyarko, D.E. Ho, et al., LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models, in: Adv. Neural Inf. Process. Syst. 36 (Datasets and Benchmarks Track), 2023. arXiv:2308.11462.",
    # 4
    "A. Fern\u00e1ndez Garc\u00eda, J. de la Rosa, J. Gonzalo, R. Morante, E. Amig\u00f3, A. Benito-Santos, J. Carrillo-de-Albornoz, V. Fresno-Fern\u00e1ndez, A. Ghajari Espinosa, G. Marco, L. Plaza, E. S\u00e1nchez-Salido, BOE-XSUM: Extreme summarization in clear language of Spanish legal decrees and notifications, arXiv:2509.24908 (2025).",
    # 5
    "Z. Wan, Y. Zhang, Y. Wang, F. Cheng, S. Kurohashi, Reformulating domain adaptation of large language models as adapt-retrieve-revise: A case study on Chinese legal domain, in: Findings of the Association for Computational Linguistics: ACL, 2024, pp. 5030\u20135041. doi:10.18653/v1/2024.findings-acl.299.",
    # 6
    "P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt\u00e4schel, S. Riedel, D. Kiela, Retrieval-augmented generation for knowledge-intensive NLP tasks, in: Adv. Neural Inf. Process. Syst. 33, 2020, pp. 9459\u20139474.",
    # 7
    "L. Martin, \u00c9. de la Clergerie, B. Sagot, A. Bordes, Controllable sentence simplification, in: Proc. 12th Language Resources and Evaluation Conference (LREC), 2020, pp. 4689\u20134698.",
    # 8
    "L. Martin, A. Fan, \u00c9. de la Clergerie, A. Bordes, B. Sagot, MUSS: Multilingual unsupervised sentence simplification by mining paraphrases, in: Proc. 13th Language Resources and Evaluation Conference (LREC), 2022, pp. 1651\u20131664. doi:10.18653/v1/2022.lrec-1.176.",
    # 9
    "H. Saggion, S. \u0160tajner, D. Ferr\u00e9s, K.C. Sheang, M. Shardlow, K. North, M. Zampieri, Findings of the TSAR-2022 shared task on multilingual lexical simplification, in: Proc. TSAR-2022 Workshop (with EMNLP 2022), 2022, pp. 271\u2013283. doi:10.18653/v1/2022.tsar-1.31.",
    # 10
    "F. Alva-Manchego, R. Stodden, J.M. Imperial, A. Barayan, K. North, H. Tayyar Madabushi, Findings of the TSAR 2025 shared task on readability-controlled text simplification, in: Proc. 4th Workshop on Text Simplification, Accessibility and Readability (TSAR), 2025, pp. 116\u2013130. doi:10.18653/v1/2025.tsar-1.8.",
    # 11
    "L. Ermakova, \u00c9. SanJuan, S. Huet, H. Azarbonyad, O. Augereau, J. Kamps, Overview of the CLEF 2023 SimpleText lab: Automatic simplification of scientific texts, in: Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2023), LNCS 14163, Springer, 2023, pp. 482\u2013506. doi:10.1007/978-3-031-42448-9_30.",
    # 12
    "L. Ermakova, \u00c9. SanJuan, S. Huet, H. Azarbonyad, G.M. Di Nunzio, F. Vezzani, J. D'Souza, J. Kamps, Overview of the CLEF 2024 SimpleText track: Improving access to scientific texts for everyone, in: Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2024), LNCS 14959, Springer, 2024, pp. 283\u2013307. doi:10.1007/978-3-031-71908-0_13.",
    # 13
    "G. Li, Y. Arase, N. Crespi, Aligning sentence simplification with ESL learner's proficiency for language acquisition, in: Proc. NAACL-HLT, 2025, pp. 492\u2013507. arXiv:2502.11457.",
    # 14
    "D. Sokova, A. Bezobrazova, C. Orasan, SQUREL at TSAR 2025 shared task CEFR-controlled text simplification with prompting and reinforcement fine-tuning, in: Proc. 4th Workshop on Text Simplification, Accessibility and Readability (TSAR 2025), 2025, pp. 242\u2013250. doi:10.18653/v1/2025.tsar-1.21.",
    # 15
    "Y. Zuo, Automatic generation of ESL learning materials based on CEFR levels using reinforcement-tuned LLMs, Discov. Artif. Intell. 6 (2025). doi:10.1007/s44163-025-00762-3.",
    # 16
    "D. Glandorf, D. Meurers, Towards fine-grained pedagogical control over English grammar complexity in educational text generation, in: Proc. 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), 2024, pp. 299\u2013308. doi:10.18653/v1/2024.bea-1.24.",
    # 17
    "M. F\u00e4rber, P. Aghdam, K. Im, M. Tawfelis, H. Ghoshal, SimplifyMyText: An LLM-based system for inclusive plain language text simplification, in: Advances in Information Retrieval (ECIR 2025), LNCS 15575, Springer, 2025, pp. 418\u2013424. doi:10.1007/978-3-031-88717-8_32.",
    # 18
    "R. Flesch, A new readability yardstick, J. Appl. Psychol. 32 (3) (1948) 221\u2013233. doi:10.1037/h0057532.",
    # 19
    "W. Xu, C. Napoles, E. Pavlick, Q. Chen, C. Callison-Burch, Optimizing statistical machine translation for text simplification, Trans. Assoc. Comput. Linguist. 4 (2016) 401\u2013415. doi:10.1162/tacl_a_00107.",
    # 20
    "W. Xu, C. Callison-Burch, C. Napoles, Problems in current text simplification research: New data can help, Trans. Assoc. Comput. Linguist. 3 (2015) 283\u2013297. doi:10.1162/tacl_a_00139.",
    # 21
    "C. Jiang, M. Maddela, W. Lan, Y. Zhong, W. Xu, Neural CRF model for sentence alignment in text simplification (Wiki-Auto), in: Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020, pp. 7943\u20137960. doi:10.18653/v1/2020.acl-main.709.",
    # 22
    "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, Proximal policy optimization algorithms, arXiv:1707.06347 (2017).",
    # 23
    "L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, R. Lowe, Training language models to follow instructions with human feedback, in: Adv. Neural Inf. Process. Syst. 35 (NeurIPS 2022), 2022, pp. 27730\u201327744.",
    # 24
    "R. Rafailov, A. Sharma, E. Mitchell, C.D. Manning, S. Ermon, C. Finn, Direct preference optimization: Your language model is secretly a reward model, in: Adv. Neural Inf. Process. Syst. 36 (NeurIPS 2023), 2023, pp. 53728\u201353741.",
    # 25
    "Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y.K. Li, Y. Wu, D. Guo, DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, arXiv:2402.03300 (2024).",
    # 26
    "DeepSeek-AI, DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, Nature 645 (2025) 633\u2013638. doi:10.1038/s41586-025-09422-z.",
    # 27
    "N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, et al., Tulu 3: Pushing frontiers in open language model post-training, arXiv:2411.15124 (2024).",
    # 28
    "H. Sun, Y. Chai, S. Wang, Y. Sun, H. Wu, H. Wang, Curiosity-driven reinforcement learning from human feedback, in: Proc. 63rd Annual Meeting of the Association for Computational Linguistics (Vol. 1: Long Papers), 2025, pp. 23517\u201323534. doi:10.18653/v1/2025.acl-long.1146.",
    # 29
    "J. Skalse, N.H.R. Howe, D. Krasheninnikov, D. Krueger, Defining and characterizing reward hacking, in: Adv. Neural Inf. Process. Syst. 35 (NeurIPS 2022), 2022.",
    # 30
    "X. Zhang, M. Lapata, Sentence simplification with deep reinforcement learning, in: Proc. EMNLP 2017, 2017, pp. 584\u2013594. doi:10.18653/v1/D17-1062.",
    # 31
    "H. Wang, J.A. Clark, H. McKelvey, L. Sterman, Z. Gao, Z. Tian, X. Liu, Improving scholarship accessibility with reinforcement learning, Inf. Res. 30 (iConf) (2025) 203\u2013218. doi:10.47989/ir30iConf47530.",
    # 32
    "S.D. Krashen, The Input Hypothesis: Issues and Implications, Longman, London/New York, 1985.",
    # 33
    "S. Vajjala, D. Meurers, On improving the accuracy of readability classification using insights from second language acquisition, in: Proc. 7th Workshop on Building Educational Applications Using NLP (BEA), 2012, pp. 163\u2013173. doi:10.5555/2390384.2390404.",
    # 34
    "Qwen Team, Qwen2.5 technical report, arXiv:2412.15115 (2024).",
    # 35
    "N. Reimers, I. Gurevych, Sentence-BERT: Sentence embeddings using Siamese BERT-networks, in: Proc. EMNLP-IJCNLP 2019, 2019, pp. 3982\u20133992. doi:10.18653/v1/D19-1410.",
    # 36
    "V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, W.-t. Yih, Dense passage retrieval for open-domain question answering, in: Proc. EMNLP 2020, 2020, pp. 6769\u20136781. doi:10.18653/v1/2020.emnlp-main.550.",
    # 37
    "Y. Du, S. Li, A. Torralba, J.B. Tenenbaum, I. Mordatch, Improving factuality and reasoning in language models through multiagent debate, in: Proc. ICML 2024 (PMLR 235), 2024.",
    # 38
    "P. He, X. Liu, J. Gao, W. Chen, DeBERTa: Decoding-enhanced BERT with disentangled attention, in: ICLR, 2021. arXiv:2006.03654.",
    # 39
    "A. Williams, N. Nangia, S.R. Bowman, A broad-coverage challenge corpus for sentence understanding through inference, in: Proc. NAACL-HLT 2018, Vol. 1 (Long Papers), 2018, pp. 1112\u20131122. doi:10.18653/v1/N18-1101.",
    # 40
    "Z. Gekhman, J. Herzig, R. Aharoni, C. Elkind, I. Szpektor, TrueTeacher: Learning factual consistency evaluation with large language models, in: Proc. EMNLP 2023, 2023. doi:10.18653/v1/2023.emnlp-main.780.",
    # 41
    "T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, QLoRA: Efficient finetuning of quantized LLMs, in: Adv. Neural Inf. Process. Syst. 36 (NeurIPS 2023), 2023, pp. 10088\u201310115.",
    # 42
    "E.J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of large language models, in: ICLR, 2022. arXiv:2106.09685.",
    # 43
    "L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E.P. Xing, H. Zhang, J.E. Gonzalez, I. Stoica, Judging LLM-as-a-judge with MT-Bench and Chatbot Arena, in: Adv. Neural Inf. Process. Syst. 36 (NeurIPS 2023, Datasets and Benchmarks Track), 2023, pp. 46595\u201346623.",
    # 44
    "Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, C. Zhu, G-Eval: NLG evaluation using GPT-4 with better human alignment, in: Proc. EMNLP 2023, 2023, pp. 2511\u20132522. doi:10.18653/v1/2023.emnlp-main.153.",
    # 45
    "H. Li, Q. Dong, J. Chen, H. Su, Y. Zhou, Q. Ai, Z. Ye, Y. Liu, LLMs-as-judges: A comprehensive survey on LLM-based evaluation methods, arXiv:2412.05579 (2024).",
    # 46
    "T. Guidroz, D. Ardila, J. Li, A. Mansour, P. Jhun, N. Chaparro Gonz\u00e1lez, X. Ji, M. Sanchez, S.S. Kakarmath, M.M.J. Bellaiche, M.A. Garrido, F. Ahmed, D. Choudhary, J. Hartford, C. Xu, H. Echeverr\u00eda, Y. Wang, J. Shaffer, E. Cao, Y. Matias, A. Hassidim, D.R. Webster, Y. Liu, S. Fujiwara, P. Bui, Q.H. Duong, LLM-based text simplification and its effect on user comprehension and cognitive load, arXiv:2505.01980 (2025).",
]


def main() -> int:
    text = open(MANUSCRIPT, encoding="utf-8").read()
    head, _, _ = text.partition("## References")
    # compute first-appearance order
    order = []
    for m in re.finditer(r"\[@(E\d{3})\]", head):
        if m.group(1) not in order:
            order.append(m.group(1))
    assert len(order) == len(REFS), f"{len(order)} in-text refs vs {len(REFS)} entries"
    cite_map = {e: str(i + 1) for i, e in enumerate(order)}

    body = "## References\n\n"
    body += "<!-- Final editing pass: replace each in-text [@Exxx] with the number below "
    body += "([@E015]\u2192[1], ...). Metadata web-verified 2026-08-18 per-reference "
    body += "(independent web-search subagent); Teacher Zhou still confirms each DOI/URL. -->\n\n"
    for i, ref in enumerate(REFS, 1):
        body += f"[{i}] {ref}\n"
    # drop trailing blank lines, keep one
    new_text = head.rstrip("\n") + "\n\n" + body.rstrip("\n") + "\n"
    with open(MANUSCRIPT, "w", encoding="utf-8") as f:
        f.write(new_text)

    # update render_submission.py CITE_MAP
    render = open(RENDER, encoding="utf-8").read()
    lines = []
    for line in render.splitlines():
        if line.strip().startswith("CITE_MAP = {"):
            lines.append("CITE_MAP = {")
            entries = ", ".join(f'"{e}": "{n}"' for e, n in cite_map.items())
            lines.append("    " + entries + ",")
        elif line.strip() == "}":
            continue  # skip old closing brace (rewritten below)
        else:
            lines.append(line)
    out = []
    for line in lines:
        if line.strip() == "CITE_MAP = {":
            out.append(line)
            out.append("}")  # closing brace after inline entries
        else:
            out.append(line)
    # simpler: rebuild from scratch between markers
    pat = re.compile(
        r"CITE_MAP = \{(?:.|\n)*?\n\}", re.MULTILINE
    )
    new_render = pat.sub(
        "CITE_MAP = {\n    "
        + ", ".join(f'"{e}": "{n}"' for e, n in cite_map.items())
        + ",\n}",
        render,
    )
    with open(RENDER, "w", encoding="utf-8") as f:
        f.write(new_render)

    print(f"wrote {len(REFS)} references; CITE_MAP updated")
    print("mapping:", cite_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
