# -*- coding: utf-8 -*-
"""
recompute_honest.py — offline honest re-scoring of stored results.

Why this exists: prior runs stored `rew` = the RAW decoded model output. GRPO
models (qwen35-grpo) emit meta-commentary preambles ("The user wants a legal
text rewritten...", "Drafting:", "**Original Text Analysis:**", ...) that the
old r_fmt() let through, so the stored FRE/metrics were computed on polluted
text. This script re-applies the FIXED cleaner (clean_output from
06_framework.py) and re-scores every stored row with the full reward stack
(including a fresh NLI contradiction score). Rows where no legal prose survives
after cleaning are counted as pure-commentary failures (fmt=0, all parts 0).

Outputs:
  results/{tag}_honest.jsonl    per-row honest record (rew_raw, rew_clean, parts, tot, fre, nli)
  logs/recompute_honest.json    per-tag/per-level aggregates

This runs AFTER the GPU re-runs (B + OOD fixes), which are already clean; it
only re-scores OLD jsonl files, so it never touches new results.
"""
import importlib
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, "/path/to/legal_english/scripts")
_fw = importlib.import_module("06_framework")  # module name starts with a digit
clean_output = _fw.clean_output
stripped_info = _fw.stripped_info
rewards = _fw.rewards
flesch = _fw.flesch
LEVEL_TARGET_FRE = _fw.LEVEL_TARGET_FRE
W = _fw.W

DATA = "/path/to/legal_english/data"
MODELS = "/path/to/legal_english/models"
RESULTS = "/path/to/legal_english/results"
LOGS = "/path/to/legal_english/logs"

# Tags to re-score (old runs; new runs L3_max3/OOD_ext_fix/OOD_india_fix are
# already produced by the fixed framework and are excluded).
ALL_TAGS = [
    "G1", "G2", "G3", "G4", "G5", "G6",
    "A_full", "A_fmt", "A_diff", "A_term", "A_copy", "A_faith",
    "A_full_full_budget", "A_diff_full_budget",
    "ADV", "ADV_full",
    "B1", "B2",
    "G6_pure", "G6_pure_s11", "G6_pure_s21", "G6_pure_s42",
    "M2",
    "OOD_doc", "OOD_mismatch", "OOD_size", "OOD_size10", "OOD_size50",
    "OOD_external",
]


def load_terms():
    corpus = json.load(open(os.path.join(DATA, "corpus_v2.json"), encoding="utf-8"))
    raw = corpus["terms"]
    return ([t["term"] if isinstance(t, dict) else t for t in raw]
            if isinstance(raw, list) else list(raw.keys()))


def main():
    import argparse
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None,
                    help="comma-separated override of ALL_TAGS (for smoke tests)")
    args = ap.parse_args()
    want = [t.strip() for t in args.tags.split(",")] if args.tags else ALL_TAGS
    tags = [t for t in want
            if os.path.exists(os.path.join(RESULTS, t + ".jsonl"))]
    print("tags:", len(tags), file=sys.stderr, flush=True)

    dev = "cuda:0"
    nli = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(MODELS, "NLI-DeBERTa-v3")).to(dev).eval()
    nli_tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "NLI-DeBERTa-v3"))
    terms = load_terms()
    print("terms:", len(terms), file=sys.stderr, flush=True)

    summary = {}
    os.makedirs(os.path.join(RESULTS), exist_ok=True)
    t0 = time.time()
    for ti, tag in enumerate(tags):
        rows = [json.loads(l) for l in open(os.path.join(RESULTS, tag + ".jsonl"),
                                            encoding="utf-8")]
        out_rows = []
        n_pure = n_stripped = 0
        for r in rows:
            raw = r["rew"]; src = r["src"]; lv = r["lv"]
            clean = clean_output(raw)
            if not clean:
                parts = {"fmt": 0.0, "diff": 0.0, "term": 0.0, "copy": 0.0,
                         "faith": 0.0}
                tot, fre, nli3 = 0.0, 0.0, {"entail": 0.0, "neutral": 0.0,
                                            "contra": 1.0}
                n_pure += 1
            else:
                tot, parts, fre, nli3 = rewards(clean, src, lv, terms,
                                                nli, nli_tok, dev)
            stripped = stripped_info(raw)
            n_stripped += int(stripped)
            out_rows.append({
                "id": r["id"], "lv": lv, "src": r["src"],
                "sec": r.get("sec", ""),
                "rew_raw": raw, "rew_clean": clean,
                "stripped": bool(stripped), "pure_commentary": not bool(clean),
                "parts": parts, "tot": round(tot, 4), "fre": round(fre, 2),
                "nli": nli3, "attempts": r.get("attempts", 0),
            })
        with open(os.path.join(RESULTS, tag + "_honest.jsonl"), "w",
                  encoding="utf-8") as f:
            for o in out_rows:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

        # ---- aggregate ----
        agg = {"n": len(out_rows), "pure_commentary": n_pure,
               "stripped": n_stripped, "levels": {}}
        for lv in ["beginner", "intermediate", "advanced"]:
            L = [o for o in out_rows if o["lv"] == lv]
            if not L:
                continue
            frs = [o["fre"] for o in L if o["fre"] > 0]
            tgt = LEVEL_TARGET_FRE.get(lv)
            inb10 = sum(1 for f in frs if abs(f - tgt) <= 10) / max(1, len(frs))
            inb12 = sum(1 for f in frs if abs(f - tgt) <= 12) / max(1, len(frs))
            cons = [o["nli"]["contra"] for o in L]
            agg["levels"][lv] = {
                "n": len(L),
                "fre_mean": round(sum(frs) / max(1, len(frs)), 2),
                "fre_std": (round(float(statistics.pstdev(frs)), 2)
                            if len(frs) > 1 else 0.0),
                "inband_10": round(inb10, 3), "inband_12": round(inb12, 3),
                "tot_mean": round(sum(o["tot"] for o in L) / len(L), 4),
                "fmt": round(sum(o["parts"]["fmt"] for o in L) / len(L), 3),
                "diff": round(sum(o["parts"]["diff"] for o in L) / len(L), 3),
                "term": round(sum(o["parts"]["term"] for o in L) / len(L), 3),
                "copy": round(sum(o["parts"]["copy"] for o in L) / len(L), 3),
                "faith": round(sum(o["parts"]["faith"] for o in L) / len(L), 3),
                "contra_mean": round(sum(cons) / len(cons), 3),
            }
        summary[tag] = agg
        print("[%d/%d] %s n=%d pure=%d stripped=%d (%.1fs)"
              % (ti + 1, len(tags), tag, len(out_rows), n_pure,
                 n_stripped, time.time() - t0), file=sys.stderr, flush=True)

    json.dump(summary, open(os.path.join(LOGS, "recompute_honest.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", os.path.join(LOGS, "recompute_honest.json"),
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
