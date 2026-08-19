# -*- coding: utf-8 -*-
"""Local one-by-one verification of every experiment artifact (E001-E014, E025-E028)
against the files actually shipped in Supplied_materials (2026-08-19).

For each source: confirm the file exists, parses (JSON/JSONL), is non-empty with a
plausible row/record count, and for the aggregate statistics files probe the key
fields the manuscript relies on -- including the honest headline values that
cross_check_numbers.py independently cross-checks against the manuscript (87/87).
This script is the "source_opened" evidence that each artifact was actually opened.

Output: one line per source with PASS/FAIL and concrete evidence. No manuscript
content leaves this machine.
"""
import json
import os
import sys

BASE = r"D:\周老师\paper\Supplied_materials"
PAPER = r"D:\周老师\paper"

# E005 / E027 honest-only files (raw run never shipped; the honest-scored output is
# the reported artifact): OOD_external, OOD_size10, OOD_size50.
HONEST_ONLY = [
    "results/OOD_external_honest.jsonl", "results/OOD_size10_honest.jsonl",
    "results/OOD_size50_honest.jsonl",
]

# source -> (kind, [paths], probe)
PLAN = {
    "E001": ("agg", ["results/stats_legal.json"],
             {"cfg_require": ["G1", "G2", "G3", "G4", "G5", "L3_max3", "A_full",
                              "A_diff", "G6_pure", "G6_full107", "ADV", "ADV_full",
                              "OOD_doc", "OOD_mismatch", "OOD_size", "OOD_size50",
                              "OOD_size10", "OOD_external", "OOD_ext_fix",
                              "OOD_india_fix", "L1_max1", "L2_max2", "L5_max5",
                              "B1", "B2"]}),
    "E002": ("agg", ["results/verifier_eval.json"], {}),
    "E003": ("agg", ["results/metrics_legal.json"], {}),
    "E004": ("jsl", ["results/verifier_G6.jsonl", "results/verifier_G1.jsonl",
                     "results/verifier_B1.jsonl", "results/verifier_ADV.jsonl",
                     "results/verifier_A_full.jsonl", "results/verifier_G6_pure_s11.jsonl",
                     "results/verifier_OOD_doc.jsonl", "results/verifier_L3_max3.jsonl"], {}),
    "E005": ("jsl", ["results/G1.jsonl", "results/G2.jsonl", "results/G3.jsonl",
                     "results/G4.jsonl", "results/G5.jsonl", "results/G6.jsonl",
                     "results/G6_full107.jsonl", "results/G6_pure.jsonl",
                     "results/G6_pure_s11.jsonl", "results/G6_pure_s21.jsonl",
                     "results/G6_pure_s42.jsonl", "results/M2.jsonl",
                     "results/ADV.jsonl", "results/ADV_full.jsonl",
                     "results/B1.jsonl", "results/B2.jsonl",
                     "results/OOD_doc.jsonl", "results/OOD_mismatch.jsonl",
                     "results/OOD_size.jsonl", "results/OOD_ext_fix.jsonl",
                     "results/OOD_india_fix.jsonl",
                     "results/A_full.jsonl", "results/A_diff.jsonl",
                     "results/A_term.jsonl", "results/A_copy.jsonl",
                     "results/A_faith.jsonl", "results/A_fmt.jsonl"] + HONEST_ONLY, {}),
    "E006": ("file", ["scripts/05_grpo_train.py"], {}),
    "E007": ("file", ["scripts/06_framework.py"], {}),
    "E008": ("file", ["scripts/07_eval.py"], {}),
    "E009": ("file", ["scripts/08_verifier_eval.py"], {}),
    "E010": ("file", ["scripts/09_baselines.py"], {}),
    "E011": ("file", ["scripts/10_metrics.py"], {}),
    "E012": ("agg", ["data/corpus_v2.json", "data/split.json"],
             {"n_snippets": 968, "split_counts": {"train": 748, "val": 113, "test": 107}}),
    "E013": ("models", ["grpo_qwen35_lora_v2"],
             {"r": 16, "alpha": 32, "n_modules": 7,
              "trainable_expected": 21233664,
              "note": "QLoRA adapter weights (D:\\周老师\\legal_english\\models\\grpo_qwen35_lora_v2). "
                      "Base Qwen3.5-4B and the Qwen3-4B-Instruct judge are server-side and not "
                      "shipped; the delivered trainable artifact is the adapter (config + safetensors)."}),
    "E014": ("docx", [r"D:\周老师\demo_v2026_Neurocomputing_v2.docx"], {}),
    "E025": ("jsl", ["results/L1_max1.jsonl", "results/L2_max2.jsonl",
                     "results/L3_max3.jsonl", "results/L5_max5.jsonl"], {}),
    "E026": ("jsl", ["results/OOD_india_fix.jsonl"], {}),
    "E027": ("agg", ["results/recompute_honest.json", "results/honest_table.json"],
             {"headline": {"G6": {"all": 0.705, "tol": 0.002},
                           "L3_max3": {"all": 0.800, "tol": 0.002}},
              "honest_files": ["results/G6_honest.jsonl", "results/G6_full107_honest.jsonl",
                               "results/ADV_honest.jsonl", "results/ADV_full_honest.jsonl",
                               "results/G6_pure_honest.jsonl", "results/G6_pure_s11_honest.jsonl",
                               "results/G6_pure_s21_honest.jsonl", "results/G6_pure_s42_honest.jsonl",
                               "results/A_full_honest.jsonl", "results/A_diff_honest.jsonl",
                               "results/A_term_honest.jsonl", "results/A_copy_honest.jsonl",
                               "results/A_faith_honest.jsonl", "results/A_fmt_honest.jsonl",
                               "results/B1_honest.jsonl", "results/B2_honest.jsonl",
                               "results/OOD_doc_honest.jsonl", "results/OOD_mismatch_honest.jsonl",
                               "results/OOD_size_honest.jsonl",
                               "results/OOD_external_honest.jsonl",
                               "results/M2_honest.jsonl", "results/G1_honest.jsonl",
                               "results/G2_honest.jsonl", "results/G3_honest.jsonl",
                               "results/G4_honest.jsonl", "results/G5_honest.jsonl",
                               "results/A_full_full_budget_honest.jsonl",
                               "results/A_diff_full_budget_honest.jsonl"]}),
    "E028": ("agg", ["results/examples_qualitative.json"],
             {"script_missing": "scripts/qualitative_analysis.py (server-side, not shipped; output shipped)"}),
}


def jsonlines(path):
    n = 0
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n += 1
            keys.update(obj.keys())
    return n, keys


def jsl_check(relpaths, extra=None):
    detail, fails = [], []
    for p in relpaths:
        full = os.path.join(BASE, p)
        if not os.path.exists(full):
            fails.append(f"MISSING {p}")
            detail.append(f"MISSING {p}")
            continue
        try:
            n, keys = jsonlines(full)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{p}: PARSE FAIL {e}")
            detail.append(f"{p}: PARSE FAIL {e}")
            continue
        if n == 0:
            fails.append(f"{p}: 0 rows")
            detail.append(f"{p}: 0 rows")
            continue
        has_src = "src" in keys
        has_lv = "lv" in keys or "level" in keys
        has_tot = "tot" in keys
        detail.append(f"{p}: {n} rows, src={has_src} lv={has_lv} tot={has_tot}")
    return "; ".join(detail), fails


def agg_check(relpaths, extra):
    detail, fails = [], []
    for p in relpaths:
        full = os.path.join(BASE, p)
        if not os.path.exists(full):
            fails.append(f"MISSING {p}")
            detail.append(f"MISSING {p}")
            continue
        try:
            with open(full, encoding="utf-8-sig") as f:
                d = json.load(f)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{p}: PARSE FAIL {e}")
            detail.append(f"{p}: PARSE FAIL {e}")
            continue
        if isinstance(d, dict):
            detail.append(f"{p}: {len(d)} top keys")
        else:
            detail.append(f"{p}: {len(d)} records")
    if "cfg_require" in extra:
        st = json.load(open(os.path.join(BASE, "results", "stats_legal.json"), encoding="utf-8-sig"))
        tags = {c.get("tag") if isinstance(c, dict) else c for c in st.get("cfg", [])}
        missing = [t for t in extra["cfg_require"] if t not in tags]
        detail.append(f"stats cfg tags: {len(tags)} present")
        if missing:
            fails.append(f"stats cfg missing tags: {missing}")
    if "n_snippets" in extra:
        c = json.load(open(os.path.join(BASE, "data", "corpus_v2.json"), encoding="utf-8-sig"))
        n = c.get("n_snippets")
        sn = len(c.get("snippets", []))
        if n != extra["n_snippets"] or sn != extra["n_snippets"]:
            fails.append(f"corpus n_snippets={n} len={sn} (exp {extra['n_snippets']})")
        detail.append(f"corpus n_snippets={n} len={sn}, n_terms={c.get('n_terms')}")
    if "split_counts" in extra:
        sp = json.load(open(os.path.join(BASE, "data", "split.json"), encoding="utf-8-sig"))
        cnts = {k: len(sp.get(k, [])) for k in ("train", "val", "test")}
        detail.append(f"split counts={cnts}")
        for k, exp in extra["split_counts"].items():
            if cnts.get(k) != exp:
                fails.append(f"split {k}={cnts.get(k)} (exp {exp})")
    if "headline" in extra:
        ht = json.load(open(os.path.join(BASE, "results", "honest_table.json"), encoding="utf-8-sig"))
        tags = ht.get("tags", {})
        for tag, spec in extra["headline"].items():
            lv = (tags.get(tag) or {}).get("levels", {}).get("all", {})
            got = lv.get("tot")
            exp = spec["all"]
            ok = got is not None and abs(got - exp) <= spec["tol"]
            detail.append(f"honest_table tags[{tag}].all.tot = {got} (exp ~{exp})")
            if not ok:
                fails.append(f"honest_table {tag} tot mismatch")
    if "honest_files" in extra:
        ok = 0
        for hp in extra["honest_files"]:
            full = os.path.join(BASE, hp)
            if os.path.exists(full):
                n, _ = jsonlines(full)
                if n > 0:
                    ok += 1
                else:
                    fails.append(f"{hp}: 0 rows")
            else:
                fails.append(f"MISSING {hp}")
        detail.append(f"honest jsonl files ok: {ok}/{len(extra['honest_files'])}")
    if "script_missing" in extra:
        detail.append(extra["script_missing"])
    return "; ".join(detail), fails


def file_check(relpaths, extra=None):
    detail, fails = [], []
    for p in relpaths:
        full = os.path.join(BASE, p)
        if not os.path.exists(full):
            fails.append(f"MISSING {p}")
            detail.append(f"MISSING {p}")
            continue
        sz = os.path.getsize(full)
        head = open(full, encoding="utf-8", errors="replace").read(200)
        detail.append(f"{p}: {sz} B, head={head[:36]!r}")
    return "; ".join(detail), fails


def dir_check(relpaths, extra):
    detail, fails = [], []
    for p in relpaths:
        full = os.path.join(BASE, p)
        if not os.path.isdir(full):
            fails.append(f"MISSING DIR {p}")
            detail.append(f"MISSING DIR {p}")
            continue
        detail.append(f"{p}: {len(os.listdir(full))} entries")
    detail.append(extra["note"])
    return "; ".join(detail), fails


def docx_check(relpaths, extra=None):
    detail, fails = [], []
    for p in relpaths:
        if os.path.exists(p):
            detail.append(f"{p}: {os.path.getsize(p)} B")
        else:
            fails.append(f"MISSING {p}")
            detail.append(f"MISSING {p}")
    return "; ".join(detail), fails


def models_check(relpaths, extra):
    """E013: verify the shipped QLoRA adapter -- config (r/alpha/modules) and the
    actual trainable parameter count read from the safetensors header."""
    detail, fails = [], []
    for name in relpaths:
        d = os.path.join(r"D:\周老师\legal_english\models", name)
        if not os.path.isdir(d):
            fails.append(f"MISSING MODEL DIR {d}")
            detail.append(f"MISSING MODEL DIR {d}")
            continue
        cfg_path = os.path.join(d, "adapter_config.json")
        if not os.path.exists(cfg_path):
            fails.append(f"{name}: no adapter_config.json")
            detail.append(f"{name}: no adapter_config.json")
            continue
        cfg = json.load(open(cfg_path, encoding="utf-8-sig"))
        r = cfg.get("r"); alpha = cfg.get("lora_alpha")
        tgt = cfg.get("target_modules") or []
        ok = (r == extra["r"] and alpha == extra["alpha"] and len(tgt) == extra["n_modules"])
        detail.append(f"{name}: r={r} alpha={alpha} modules={len(tgt)} ({'OK' if ok else 'MISMATCH'})")
        if not ok:
            fails.append(f"{name}: config mismatch (r/alpha/modules)")
        st = os.path.join(d, "adapter_model.safetensors")
        if not os.path.exists(st):
            fails.append(f"{name}: no adapter_model.safetensors")
            continue
        with open(st, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(hlen))
        total = 0
        for k, v in header.items():
            if k == "__metadata__":
                continue
            t = 1
            for s in v.get("shape", []):
                t *= s
            total += t
        okn = total == extra["trainable_expected"]
        detail.append(f"{name}: trainable params {total:,} (exp {extra['trainable_expected']:,}) "
                      f"[{'OK' if okn else 'MISMATCH'}]")
        if not okn:
            fails.append(f"{name}: trainable param count mismatch")
    detail.append(extra["note"])
    return "; ".join(detail), fails


HANDLERS = {
    "agg": agg_check, "jsl": jsl_check, "file": file_check,
    "dir": dir_check, "docx": docx_check, "models": models_check,
}


def main():
    print("=== Local artifact verification (E001-E014, E025-E028) ===", flush=True)
    all_fails = []
    for eid in sorted(PLAN):
        kind, paths, extra = PLAN[eid]
        detail, fails = HANDLERS[kind](paths, extra)
        all_fails += [(eid, f) for f in fails]
        print(f"[{'FAIL' if fails else 'PASS'}] {eid}", flush=True)
        print(f"         {detail}", flush=True)
    print(f"\nSUMMARY: {len(all_fails)} fail items", flush=True)
    for eid, f in all_fails:
        print(f"  FAIL {eid}: {f}", flush=True)
    return 1 if all_fails else 0


if __name__ == "__main__":
    sys.exit(main())
