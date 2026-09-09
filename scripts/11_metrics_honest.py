# -*- coding: utf-8 -*-
"""11_metrics_honest.py — offline metrics on the COMMENTARY-CLEANED rewrites.

Re-runs the SARI/BLEU/readability/contradiction pipeline of 10_metrics.py but on
the *_honest.jsonl records (rew_clean), with {ref}_honest (cleaned) as the SARI
reference, so the §4.8 numbers are consistent with the honest tot/FRE reported
everywhere else. NLI contradiction is taken from the stored nli (recomputed
offline by recompute_honest.py), so no NLI model load is needed. Pure-commentary
rows are excluded from SARI/BLEU/references but their count is reported.

Usage: python 11_metrics_honest.py [--ref B1]
Output: results/metrics_honest.json + console markdown.
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m10 = importlib.import_module("10_metrics")  # sari_doc, rich_readability, tokens
sari_doc = m10.sari_doc
rich_readability = m10.rich_readability
tokens = m10.tokens

RESULTS = "/path/to/legal_english/results"
DATA = "/path/to/legal_english/data"

TAGS = ["G1", "G2", "G3", "G4", "G5", "G6", "A_full", "A_diff", "A_term",
        "A_copy", "A_faith", "A_fmt", "G6_pure", "G6_pure_s11", "M2", "ADV",
        "OOD_doc", "OOD_mismatch", "OOD_size", "B2",
        "L1_max1", "L2_max2", "L3_max3", "L5_max5",
        "OOD_ext_fix", "OOD_india_fix"]


def load_rows(tag):
    # prefer the honest (commentary-cleaned) file; raw only for round-7 tags.
    for suffix in ("_honest", ""):
        p = os.path.join(RESULTS, f"{tag}{suffix}.jsonl")
        if os.path.isfile(p):
            return [json.loads(l) for l in open(p, encoding="utf-8")]
    return None


def _rew(r):
    return (r.get("rew_clean") or r.get("rew") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="B1")
    args = ap.parse_args()

    corpus = json.load(open(os.path.join(DATA, "corpus_v2.json"), encoding="utf-8"))
    raw_terms = corpus["terms"]
    terms = ([t["term"] if isinstance(t, dict) else t for t in raw_terms]
             if isinstance(raw_terms, list) else list(raw_terms.keys()))

    refs = {}
    for r in (load_rows(args.ref) or []):
        rw = _rew(r)
        if rw:
            refs[(r["id"], r["lv"])] = rw
    print("references: %s (cleaned) -> %d pairs" % (args.ref, len(refs)), flush=True)

    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    out = {"tags": {}, "sari_ref": args.ref + "_honest"}
    print("\n| tag | lv | n | SARI | BLEU | FRE | FK | Fog | wLen | TTR | sLen | termDen | contra率 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in TAGS:
        rows = load_rows(tag)
        if not rows:
            continue
        out["tags"][tag] = {"levels": {}, "n": len(rows),
                            "pure": sum(1 for r in rows if not _rew(r))}
        for lv in sorted(set(r["lv"] for r in rows)):
            sub = [r for r in rows if r["lv"] == lv]
            sari_scores, sys_sent, ref_sent = [], [], []
            for r in sub:
                rw = _rew(r)
                if not rw:
                    continue
                ref = refs.get((r["id"], r["lv"]))
                if ref:
                    sari_scores.append(sari_doc(rw, ref, r["src"]))
                sys_sent.append(tokens(rw))
                ref_sent.append([ref] if ref else None)
            kept = [i for i in range(len(sub)) if refs.get((sub[i]["id"], sub[i]["lv"]))]
            ref_f = [x for x in ref_sent if x]
            sys_b = [sys_sent[i] for i in kept]
            bleu = (corpus_bleu(ref_f, sys_b,
                                smoothing_function=SmoothingFunction().method7)
                    if ref_f and sys_b else None)
            rd = [rich_readability(_rew(r), terms) for r in sub if _rew(r)]
            contras = []
            for r in sub:
                if not _rew(r):
                    continue
                nli = r.get("nli")
                if nli:
                    contras.append(float(nli.get("contra", 0.0)))
            agg = {
                "n": len(sub), "pure": sum(1 for r in sub if not _rew(r)),
                "sari": float(sum(sari_scores) / len(sari_scores)) if sari_scores else None,
                "bleu": float(bleu) if bleu is not None else None,
                "fre": float(sum(x["fre"] for x in rd) / len(rd)) if rd else None,
                "fk_grade": float(sum(x["fk_grade"] for x in rd) / len(rd)) if rd else None,
                "gunning_fog": float(sum(x["gunning_fog"] for x in rd) / len(rd)) if rd else None,
                "avg_word_len": float(sum(x["avg_word_len"] for x in rd) / len(rd)) if rd else None,
                "ttr": float(sum(x["ttr"] for x in rd) / len(rd)) if rd else None,
                "avg_sent_len": float(sum(x["avg_sent_len"] for x in rd) / len(rd)) if rd else None,
                "term_density": float(sum(x["term_density"] for x in rd) / len(rd)) if rd else None,
                "contra_rate": float(sum(contras) / len(contras)) if contras else None,
            }
            out["tags"][tag]["levels"][lv] = agg

            def f(v, nd):
                return ("%%.%df" % nd) % v if v is not None else "-"
            print("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                  % (tag, lv[:4], agg["n"], f(agg["sari"], 2), f(agg["bleu"], 3),
                     f(agg["fre"], 1), f(agg["fk_grade"], 1), f(agg["gunning_fog"], 1),
                     f(agg["avg_word_len"], 1), f(agg["ttr"], 2),
                     f(agg["avg_sent_len"], 1), f(agg["term_density"], 3),
                     f(agg["contra_rate"], 3)), flush=True)

    out_path = os.path.join(RESULTS, "metrics_honest.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nmetrics_honest -> %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
