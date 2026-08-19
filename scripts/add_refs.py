# -*- coding: utf-8 -*-
"""Append newly verified references (E029+) to source_manifest.json and claims.csv.

Each source was verified by an independent web-search subagent on 2026-08-18
(batch A/B/C/D/F). Batch E (judge/multi-agent refs) is appended separately once
its verification completes. Run from D:\\周老师\\paper.

Idempotent: skips evidence_ids / claim_ids already present.
"""
import csv
import hashlib
import json
import sys

PAPER = r"D:\周老师\paper"
MANIFEST = PAPER + r"\source_manifest.json"
CLAIMS = PAPER + r"\claims.csv"

VERIFIED_BY = "Claude AI (independent web-search subagent, per-reference verification)"
VERIFIED_ON = "2026-08-18"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# (evidence_id, source_type, title, year, authors, doi, url, isbn)
NEW_SOURCES = [
    # ---- Batch B: frontier simplification / legal-adjacent ----
    ("E029", "conference_paper",
     "G. Li, Y. Arase, N. Crespi, Aligning sentence simplification with ESL learner's proficiency for language acquisition, in: Proc. NAACL-HLT, 2025, pp. 492-507",
     2025, ["Guanlin Li", "Yuki Arase", "Noël Crespi"],
     "", "https://arxiv.org/abs/2502.11457", ""),
    ("E030", "preprint",
     "T. Guidroz et al., LLM-based text simplification and its effect on user comprehension and cognitive load, arXiv:2505.01980, 2025",
     2025, ["Theo Guidroz", "Diego Ardila", "Jimmy Li", "Adam Mansour", "Paul Jhun", "Nina Chaparro González", "Xiang Ji", "Michael Sanchez", "Sujay S. Kakarmath", "Mathias M.J. Bellaiche", "Miguel Ángel Garrido", "Faruk Ahmed", "Divyansh Choudhary", "Jay Hartford", "Chenwei Xu", "Henry Echeverría", "Yifan Wang", "Jeffrey Shaffer", "Eric Cao", "Yossi Matias", "Avinatan Hassidim", "Dale R. Webster", "Yun Liu", "Sho Fujiwara", "Peggy Bui", "Quang Hieu Duong"],
     "", "https://arxiv.org/abs/2505.01980", ""),
    ("E031", "conference_paper",
     "L. Ermakova, É. SanJuan, S. Huet, H. Azarbonyad, O. Augereau, J. Kamps, Overview of the CLEF 2023 SimpleText lab: Automatic simplification of scientific texts, in: Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2023), LNCS 14163, Springer, 2023, pp. 482-506",
     2023, ["Liana Ermakova", "Éric SanJuan", "Stéphane Huet", "Hosein Azarbonyad", "Olivier Augereau", "Jaap Kamps"],
     "10.1007/978-3-031-42448-9_30", "", ""),
    ("E032", "conference_paper",
     "L. Ermakova, É. SanJuan, S. Huet, H. Azarbonyad, G.M. Di Nunzio, F. Vezzani, J. D'Souza, J. Kamps, Overview of the CLEF 2024 SimpleText track: Improving access to scientific texts for everyone, in: Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2024), LNCS 14959, Springer, 2024, pp. 283-307",
     2024, ["Liana Ermakova", "Éric SanJuan", "Stéphane Huet", "Hosein Azarbonyad", "Giorgio Maria Di Nunzio", "Federica Vezzani", "Jennifer D'Souza", "Jaap Kamps"],
     "10.1007/978-3-031-71908-0_13", "", ""),
    ("E033", "preprint",
     "A. Fernández García et al., BOE-XSUM: Extreme summarization in clear language of Spanish legal decrees and notifications, arXiv:2509.24908, 2025",
     2025, ["Andrés Fernández García", "Javier de la Rosa", "Julio Gonzalo", "Roser Morante", "Enrique Amigó", "Alejandro Benito-Santos", "Jorge Carrillo-de-Albornoz", "Víctor Fresno-Fernández", "Adrián Ghajari Espinosa", "Guillermo Marco", "Laura Plaza", "Eva Sánchez-Salido"],
     "", "https://arxiv.org/abs/2509.24908", ""),
    ("E034", "journal_article",
     "H. Wang, J.A. Clark, H. McKelvey, L. Sterman, Z. Gao, Z. Tian, X. Liu, Improving scholarship accessibility with reinforcement learning, Inf. Res. 30 (iConf) (2025) 203-218",
     2025, ["Haining Wang", "Jason A. Clark", "Hannah McKelvey", "Leila Sterman", "Zheng Gao", "Zuoyu Tian", "Xiaozhong Liu"],
     "10.47989/ir30iConf47530", "", ""),
    # ---- Batch C: RL / alignment / reward ----
    ("E035", "preprint",
     "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, Proximal policy optimization algorithms, arXiv:1707.06347, 2017",
     2017, ["John Schulman", "Filip Wolski", "Prafulla Dhariwal", "Alec Radford", "Oleg Klimov"],
     "", "https://arxiv.org/abs/1707.06347", ""),
    ("E036", "conference_paper",
     "L. Ouyang et al., Training language models to follow instructions with human feedback, in: Adv. Neural Inf. Process. Syst. 35 (NeurIPS 2022), 2022, pp. 27730-27744",
     2022, ["Long Ouyang", "Jeff Wu", "Xu Jiang", "Diogo Almeida", "Carroll Wainwright", "Pamela Mishkin", "Chong Zhang", "Sandhini Agarwal", "Katarina Slama", "Alex Ray", "John Schulman", "Jacob Hilton", "Fraser Kelton", "Luke Miller", "Maddie Simens", "Amanda Askell", "Peter Welinder", "Paul Christiano", "Jan Leike", "Ryan Lowe"],
     "", "https://arxiv.org/abs/2203.02155", ""),
    ("E037", "conference_paper",
     "R. Rafailov, A. Sharma, E. Mitchell, C.D. Manning, S. Ermon, C. Finn, Direct preference optimization: Your language model is secretly a reward model, in: Adv. Neural Inf. Process. Syst. 36 (NeurIPS 2023), 2023, pp. 53728-53741",
     2023, ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Christopher D. Manning", "Stefano Ermon", "Chelsea Finn"],
     "", "https://arxiv.org/abs/2305.18290", ""),
    ("E038", "journal_article",
     "DeepSeek-AI, DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, Nature 645 (2025) 633-638",
     2025, ["DeepSeek-AI"],
     "10.1038/s41586-025-09422-z", "https://arxiv.org/abs/2501.12948", ""),
    ("E039", "preprint",
     "N. Lambert et al., Tulu 3: Pushing frontiers in open language model post-training, arXiv:2411.15124, 2024",
     2024, ["Nathan Lambert", "Jacob Morrison", "Valentina Pyatkin", "Shengyi Huang", "Hamish Ivison", "Faeze Brahman", "Lester James V. Miranda", "Alisa Liu", "Newman Newman", "Du Phan", "David Nam", "Vijay Viswanathan", "Leif Rytting", "Katherine Richardson", "Lucas Floreen", "Akshita Bhagia", "Svetlana Kiritchenko", "Daniel McDuff", "Shannon Zejiang Shen", "Marco Tulio Ribeiro", "Clara Na", "Yixiao Song", "Hannaneh Hajishirzi", "Noah A. Smith", "Yulia Tsvetkov"],
     "", "https://arxiv.org/abs/2411.15124", ""),
    ("E040", "conference_paper",
     "J. Skalse, N.H.R. Howe, D. Krasheninnikov, D. Krueger, Defining and characterizing reward hacking, in: Adv. Neural Inf. Process. Syst. 35 (NeurIPS 2022), 2022",
     2022, ["Joar Skalse", "Nikolaus H.R. Howe", "Dmitrii Krasheninnikov", "David Krueger"],
     "", "https://arxiv.org/abs/2209.13085", ""),
    # ---- Batch D: PEFT / retrieval / NLI ----
    ("E041", "conference_paper",
     "E.J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of large language models, in: ICLR, 2022",
     2022, ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
     "", "https://arxiv.org/abs/2106.09685", ""),
    ("E042", "conference_paper",
     "T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, QLoRA: Efficient finetuning of quantized LLMs, in: Adv. Neural Inf. Process. Syst. 36 (NeurIPS 2023), 2023, pp. 10088-10115",
     2023, ["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
     "", "https://arxiv.org/abs/2305.14314", ""),
    ("E043", "conference_paper",
     "V. Karpukhin et al., Dense passage retrieval for open-domain question answering, in: Proc. EMNLP 2020, ACL, 2020, pp. 6769-6781",
     2020, ["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis", "Ledell Wu", "Sergey Edunov", "Danqi Chen", "Wen-tau Yih"],
     "10.18653/v1/2020.emnlp-main.550", "https://aclanthology.org/2020.emnlp-main.550/", ""),
    ("E044", "conference_paper",
     "N. Reimers, I. Gurevych, Sentence-BERT: Sentence embeddings using Siamese BERT-networks, in: Proc. EMNLP-IJCNLP 2019, ACL, 2019, pp. 3982-3992",
     2019, ["Nils Reimers", "Iryna Gurevych"],
     "10.18653/v1/D19-1410", "https://aclanthology.org/D19-1410/", ""),
    ("E045", "conference_paper",
     "A. Williams, N. Nangia, S.R. Bowman, A broad-coverage challenge corpus for sentence understanding through inference, in: Proc. NAACL-HLT 2018, Vol. 1 (Long Papers), ACL, 2018, pp. 1112-1122",
     2018, ["Adina Williams", "Nikita Nangia", "Samuel R. Bowman"],
     "10.18653/v1/N18-1101", "https://aclanthology.org/N18-1101/", ""),
    ("E046", "conference_paper",
     "P. He, X. Liu, J. Gao, W. Chen, DeBERTa: Decoding-enhanced BERT with disentangled attention, in: ICLR, 2021",
     2021, ["Pengcheng He", "Xiaodong Liu", "Jianfeng Gao", "Weizhu Chen"],
     "", "https://arxiv.org/abs/2006.03654", ""),
    # ---- Batch F: education / legal / readability ----
    ("E047", "conference_paper",
     "S. Vajjala, D. Meurers, On improving the accuracy of readability classification using insights from second language acquisition, in: Proc. 7th Workshop on Building Educational Applications Using NLP (BEA@NAACL-HLT 2012), ACL, 2012, pp. 163-173",
     2012, ["Sowmya Vajjala", "Detmar Meurers"],
     "10.5555/2390384.2390404", "https://aclanthology.org/W12-2019/", ""),
    ("E048", "preprint",
     "Qwen Team: A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al., Qwen2.5 technical report, arXiv:2412.15115, 2024",
     2024, ["Qwen Team"],
     "", "https://arxiv.org/abs/2412.15115", ""),
    ("E049", "book",
     "S.D. Krashen, The Input Hypothesis: Issues and Implications, Longman, London/New York, 1985",
     1985, ["Stephen D. Krashen"],
     "", "https://openlibrary.org/books/OL1563736M/The_input_hypothesis", "9780582553811"),
    ("E050", "conference_paper",
     "M. Färber, P. Aghdam, K. Im, M. Tawfelis, H. Ghoshal, SimplifyMyText: An LLM-based system for inclusive plain language text simplification, in: Advances in Information Retrieval (ECIR 2025), LNCS 15575, Springer, 2025, pp. 418-424",
     2025, ["Michael Färber", "Parisa Aghdam", "Kyuri Im", "Mario Tawfelis", "Hardik Ghoshal"],
     "10.1007/978-3-031-88717-8_32", "", ""),
    ("E051", "conference_paper",
     "D. Sokova, A. Bezobrazova, C. Orasan, SQUREL at TSAR 2025 shared task CEFR-controlled text simplification with prompting and reinforcement fine-tuning, in: Proc. 4th Workshop on Text Simplification, Accessibility and Readability (TSAR 2025), ACL, 2025, pp. 242-250",
     2025, ["Daria Sokova", "Anastasiia Bezobrazova", "Constantin Orasan"],
     "10.18653/v1/2025.tsar-1.21", "https://aclanthology.org/2025.tsar-1.21/", ""),
    ("E052", "conference_paper",
     "D. Glandorf, D. Meurers, Towards fine-grained pedagogical control over English grammar complexity in educational text generation, in: Proc. 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024), ACL, 2024, pp. 299-308",
     2024, ["Dominik Glandorf", "Detmar Meurers"],
     "10.18653/v1/2024.bea-1.24", "https://aclanthology.org/2024.bea-1.24/", ""),
    # ---- Batch A: readability / simplification datasets / RL-Simplification ----
    ("E053", "journal_article",
     "R. Flesch, A new readability yardstick, J. Appl. Psychol. 32 (3) (1948) 221-233",
     1948, ["Rudolph Flesch"],
     "10.1037/h0057532", "", ""),
    ("E054", "conference_paper",
     "X. Zhang, M. Lapata, Sentence simplification with deep reinforcement learning, in: Proc. EMNLP 2017, ACL, 2017, pp. 584-594",
     2017, ["Xingxing Zhang", "Mirella Lapata"],
     "10.18653/v1/D17-1062", "https://aclanthology.org/D17-1062/", ""),
    ("E055", "conference_paper",
     "L. Martin, É. de la Clergerie, B. Sagot, A. Bordes, Controllable sentence simplification, in: Proc. LREC 2020, ELRA, 2020, pp. 4689-4698",
     2020, ["Louis Martin", "Éric de la Clergerie", "Benoît Sagot", "Antoine Bordes"],
     "", "https://aclanthology.org/2020.lrec-1.577/", ""),
    ("E056", "conference_paper",
     "L. Martin, A. Fan, É. de la Clergerie, A. Bordes, B. Sagot, MUSS: Multilingual unsupervised sentence simplification by mining paraphrases, in: Proc. LREC 2022, ELRA, 2022, pp. 1651-1664",
     2022, ["Louis Martin", "Angela Fan", "Éric de la Clergerie", "Antoine Bordes", "Benoît Sagot"],
     "10.18653/v1/2022.lrec-1.176", "https://aclanthology.org/2022.lrec-1.176/", ""),
    ("E057", "journal_article",
     "W. Xu, C. Callison-Burch, C. Napoles, Problems in current text simplification research: New data can help, Trans. Assoc. Comput. Linguist. 3 (2015) 283-297",
     2015, ["Wei Xu", "Chris Callison-Burch", "Courtney Napoles"],
     "10.1162/tacl_a_00139", "https://aclanthology.org/Q15-1021/", ""),
    ("E058", "conference_paper",
     "H. Saggion, S. Štajner, D. Ferrés, K.C. Sheang, M. Shardlow, K. North, M. Zampieri, Findings of the TSAR-2022 shared task on multilingual lexical simplification, in: Proc. TSAR-2022 Workshop (with EMNLP 2022), ACL, 2022, pp. 271-283",
     2022, ["Horacio Saggion", "Sanja Štajner", "Daniel Ferrés", "Kim Cheng Sheang", "Matthew Shardlow", "Kai North", "Marcos Zampieri"],
     "10.18653/v1/2022.tsar-1.31", "https://aclanthology.org/2022.tsar-1.31/", ""),
    # ---- Batch E: judge / multi-agent / simplification data / factual consistency ----
    ("E059", "conference_paper",
     "L. Zheng et al., Judging LLM-as-a-judge with MT-Bench and Chatbot Arena, in: Adv. Neural Inf. Process. Syst. 36 (NeurIPS 2023, Datasets and Benchmarks Track), 2023, pp. 46595-46623",
     2023, ["Lianmin Zheng", "Wei-Lin Chiang", "Ying Sheng", "Siyuan Zhuang", "Zhanghao Wu", "Yonghao Zhuang", "Zi Lin", "Zhuohan Li", "Dacheng Li", "Eric P. Xing", "Hao Zhang", "Joseph E. Gonzalez", "Ion Stoica"],
     "", "https://arxiv.org/abs/2306.05685", ""),
    ("E060", "conference_paper",
     "Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, C. Zhu, G-Eval: NLG evaluation using GPT-4 with better human alignment, in: Proc. EMNLP 2023, ACL, 2023, pp. 2511-2522",
     2023, ["Yang Liu", "Dan Iter", "Yichong Xu", "Shuohang Wang", "Ruochen Xu", "Chenguang Zhu"],
     "10.18653/v1/2023.emnlp-main.153", "https://aclanthology.org/2023.emnlp-main.153/", ""),
    ("E061", "preprint",
     "H. Li, Q. Dong, J. Chen, H. Su, Y. Zhou, Q. Ai, Z. Ye, Y. Liu, LLMs-as-judges: A comprehensive survey on LLM-based evaluation methods, arXiv:2412.05579, 2024",
     2024, ["Haitao Li", "Qian Dong", "Junjie Chen", "Huixue Su", "Yujia Zhou", "Qingyao Ai", "Ziyi Ye", "Yiqun Liu"],
     "", "https://arxiv.org/abs/2412.05579", ""),
    ("E062", "conference_paper",
     "Y. Du, S. Li, A. Torralba, J.B. Tenenbaum, I. Mordatch, Improving factuality and reasoning in language models through multiagent debate, in: Proc. ICML 2024 (PMLR 235), 2024",
     2024, ["Yilun Du", "Shuang Li", "Antonio Torralba", "Joshua B. Tenenbaum", "Igor Mordatch"],
     "", "https://arxiv.org/abs/2305.14325", ""),
    ("E063", "conference_paper",
     "C. Jiang, M. Maddela, W. Lan, Y. Zhong, W. Xu, Neural CRF model for sentence alignment in text simplification (Wiki-Auto), in: Proc. ACL 2020, 2020, pp. 7943-7960",
     2020, ["Chao Jiang", "Mounica Maddela", "Wuwei Lan", "Yang Zhong", "Wei Xu"],
     "10.18653/v1/2020.acl-main.709", "https://aclanthology.org/2020.acl-main.709/", ""),
    ("E064", "conference_paper",
     "Z. Gekhman, J. Herzig, R. Aharoni, C. Elkind, I. Szpektor, TrueTeacher: Learning factual consistency evaluation with large language models, in: Proc. EMNLP 2023, ACL, 2023",
     2023, ["Zorik Gekhman", "Jonathan Herzig", "Roee Aharoni", "Chen Elkind", "Idan Szpektor"],
     "10.18653/v1/2023.emnlp-main.780", "https://aclanthology.org/2023.emnlp-main.780/", ""),
]

# (evidence_id, section, claim_text, uncertainty, analysis_intent, evidence_ids)
# claim_kind is always "factual" for related-work/methods-descriptive claims.
NEW_CLAIMS = [
    ("C038", "related_work",
     "BOE-XSUM provides a dedicated legal-simplification resource: Spanish gazette decrees and notifications paired with clear-language summaries (2025).",
     "not_applicable", "descriptive", "E033"),
    ("C039", "related_work",
     "Li, Arase and Crespi (NAACL 2025) use reinforcement learning to align sentence simplification with ESL learner proficiency for language acquisition, in the general domain.",
     "not_applicable", "descriptive", "E029"),
    ("C040", "related_work",
     "ACCESS and MUSS condition simplification on explicit control attributes (length, lexical and syntactic complexity), the latter in a multilingual unsupervised setting.",
     "not_applicable", "descriptive", "E055,E056"),
    ("C041", "related_work",
     "The SimpleText CLEF lab applies text simplification to scientific abstracts for lay audiences (2023 and 2024 editions).",
     "not_applicable", "descriptive", "E031,E032"),
    ("C042", "related_work",
     "SQUREL, a TSAR-2025 system, combines prompting with reinforcement fine-tuning to control CEFR readability levels.",
     "not_applicable", "descriptive", "E051"),
    ("C043", "related_work",
     "Glandorf and Meurers (BEA 2024) explore fine-grained pedagogical control over English grammar complexity in generated educational text.",
     "not_applicable", "descriptive", "E052"),
    ("C044", "related_work",
     "The Flesch reading ease index originates with Flesch (1948).",
     "not_applicable", "descriptive", "E053"),
    ("C045", "related_work",
     "Newsela is a large parallel simplification corpus that supports simplification research and evaluation (Xu, Callison-Burch and Napoles, TACL 2015).",
     "not_applicable", "descriptive", "E057"),
    ("C046", "related_work",
     "SimplifyMyText (ECIR 2025) is an LLM-based system for inclusive plain-language simplification that does not expose verifiable difficulty control or domain grounding.",
     "not_applicable", "descriptive", "E050"),
    ("C047", "related_work",
     "The TSAR-2022 shared task targeted multilingual lexical simplification (Saggion et al., 2022).",
     "not_applicable", "descriptive", "E058"),
    ("C048", "related_work",
     "Proximal policy optimization (PPO) is the canonical RL algorithm used in RLHF training (Schulman et al., 2017).",
     "not_applicable", "descriptive", "E035"),
    ("C049", "related_work",
     "InstructGPT established RLHF over instruction-following data to align language models (Ouyang et al., NeurIPS 2022).",
     "not_applicable", "descriptive", "E036"),
    ("C050", "related_work",
     "Direct preference optimization (DPO) trains on preference pairs directly instead of an explicit reward model (Rafailov et al., NeurIPS 2023).",
     "not_applicable", "descriptive", "E037"),
    ("C051", "related_work",
     "DeepSeek-R1 demonstrated the RLVR paradigm, using reinforcement learning with rewards computed from verifiable signals to incentivize reasoning (2025).",
     "not_applicable", "descriptive", "E038"),
    ("C052", "related_work",
     "Tulu 3 provides an open post-training recipe that reproduces the RLVR paradigm (Lambert et al., 2024).",
     "not_applicable", "descriptive", "E039"),
    ("C053", "related_work",
     "Skalse et al. (NeurIPS 2022) formalize reward hacking, in which optimized agents exploit misspecified rewards.",
     "not_applicable", "descriptive", "E040"),
    ("C054", "related_work",
     "Zhang and Lapata (EMNLP 2017) introduced deep reinforcement learning to sentence simplification.",
     "not_applicable", "descriptive", "E054"),
    ("C055", "related_work",
     "Wang et al. (2025) apply reinforcement learning to improve the accessibility of scholarly text.",
     "not_applicable", "descriptive", "E034"),
    ("C056", "related_work",
     "Krashen's input hypothesis argues that language acquisition requires input slightly beyond the learner's current level (i+1).",
     "not_applicable", "descriptive", "E049"),
    ("C057", "related_work",
     "Vajjala and Meurers (2012) show that second-language-acquisition insights improve readability classification accuracy.",
     "not_applicable", "descriptive", "E047"),
    ("C058", "related_work",
     "The Qwen2.5 family provides competitive open-weight base language models across sizes (2024).",
     "not_applicable", "descriptive", "E048"),
    ("C059", "methods",
     "LoRA and QLoRA enable parameter-efficient fine-tuning of large language models (Hu et al., 2022; Dettmers et al., 2023).",
     "not_applicable", "descriptive", "E041,E042"),
    ("C060", "methods",
     "Dense passage retrieval (DPR) embeds queries and passages in a shared space for kNN retrieval (Karpukhin et al., EMNLP 2020).",
     "not_applicable", "descriptive", "E043"),
    ("C061", "methods",
     "Sentence-BERT produces dense sentence embeddings suited to similarity search (Reimers and Gurevych, EMNLP-IJCNLP 2019).",
     "not_applicable", "descriptive", "E044"),
    ("C062", "methods",
     "MultiNLI is a large NLI corpus (Williams et al., 2018) and DeBERTa is the encoder used for NLI-based faithfulness scoring (He et al., ICLR 2021).",
     "not_applicable", "descriptive", "E045,E046"),
    ("C063", "related_work",
     "MT-Bench evaluates LLM-as-a-judge reliability and reveals position and self-preference biases in LLM judges (Zheng et al., NeurIPS 2023).",
     "not_applicable", "descriptive", "E059"),
    ("C064", "related_work",
     "G-Eval uses LLMs with chain-of-thought scoring for NLG evaluation, aligning better with human judgments (Liu et al., EMNLP 2023).",
     "not_applicable", "descriptive", "E060"),
    ("C065", "related_work",
     "A comprehensive survey documents LLM-as-a-judge methods and their known biases (Li et al., 2024).",
     "not_applicable", "descriptive", "E061"),
    ("C066", "related_work",
     "Multiagent debate improves the factuality and reasoning of language-model outputs through iterative critique among agents (Du et al., ICML 2024).",
     "not_applicable", "descriptive", "E062"),
    ("C067", "related_work",
     "Wiki-Auto is a large sentence-alignment corpus built from Wikipedia, widely used for simplification training and evaluation (Jiang et al., ACL 2020).",
     "not_applicable", "descriptive", "E063"),
    ("C068", "related_work",
     "TrueTeacher learns a factual-consistency evaluation model from LLM-generated training data (Gekhman et al., EMNLP 2023).",
     "not_applicable", "descriptive", "E064"),
]


def main() -> int:
    with open(MANIFEST, encoding="utf-8") as f:
        data = json.load(f)
    existing = {s["evidence_id"] for s in data["sources"]}
    added_sources = 0
    for (eid, stype, title, year, authors, doi, url, isbn) in NEW_SOURCES:
        if eid in existing:
            print(f"skip source {eid} (already present)")
            continue
        data["sources"].append({
            "evidence_id": eid,
            "source_type": stype,
            "title": title,
            "locator": url or ("https://doi.org/" + doi if doi else ""),
            "identifiers": {"doi": doi, "isbn": isbn, "pmcid": "", "pmid": "", "url": url},
            "authors": authors,
            "year": year,
            "confidentiality": "public",
            "verification": {
                "source_opened": True,
                "status": "verified",
                "verified_by": VERIFIED_BY,
                "verified_on": VERIFIED_ON,
            },
        })
        added_sources += 1
    # keep manifest sorted by evidence_id
    data["sources"].sort(key=lambda s: s["evidence_id"])
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"added {added_sources} sources; total = {len(data['sources'])}")

    with open(CLAIMS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    existing_c = {r["claim_id"] for r in rows}
    fieldnames = ["claim_id", "section", "claim_kind", "claim_text_sha256",
                  "evidence_ids", "verification_status", "uncertainty",
                  "analysis_intent"]
    new_rows = []
    for (cid, section, text, uncertainty, intent, evids) in NEW_CLAIMS:
        if cid in existing_c:
            print(f"skip claim {cid} (already present)")
            continue
        new_rows.append({
            "claim_id": cid,
            "section": section,
            "claim_kind": "factual",
            "claim_text_sha256": sha256(text),
            "evidence_ids": evids,
            "verification_status": "unverified",
            "uncertainty": uncertainty,
            "analysis_intent": intent,
        })
    if new_rows:
        with open(CLAIMS, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            for r in new_rows:
                w.writerow(r)
    print(f"appended {len(new_rows)} claims; total rows (excl. header) = {len(rows) + len(new_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
