# -*- coding: utf-8 -*-
"""
10_metrics.py — offline supplementary metrics (task 22, round-2).
For every results/{tag}.jsonl that exists, computes:
  1. SARI  (sentence-level, output vs 9B-teacher reference, src context)
  2. BLEU  (nltk corpus BLEU vs same reference)
  3. rich readability: Flesch-Kincaid grade, Gunning Fog, avg word length,
     TTR, avg sentence length, term density
  4. hallucination rate = NLI contradiction proportion (3-class, recomputed
     offline for records lacking a stored nli; stored nli reused when present)
  5. efficiency: adapter trainable-param estimate, model disk sizes, mean sec/item
References = results/B1.jsonl (Qwen3.5-9B zero-shot simplifications). Documented
limitation: single machine-generated reference.

Output: results/metrics_legal.json + console markdown.
Usage: python 10_metrics.py [--ref B1]
"""
import os, sys, json, re, argparse
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = "/root/autodl-tmp/legal_english/data"
MODELS = "/root/autodl-tmp/legal_english/models"
RESULTS = "/root/autodl-tmp/legal_english/results"

LEVEL_TARGET_FRE = {"beginner": 63.0, "intermediate": 45.0, "advanced": 26.0}
TAGS = ["G1", "G2", "G3", "G4", "G5", "G6", "A_full", "A_diff", "A_term",
        "A_copy", "A_faith", "A_fmt", "G6_pure", "G6_pure_s11", "M2", "ADV",
        "OOD_doc", "OOD_mismatch", "OOD_size", "B2"]


def _syllables(w):
    w = w.lower()
    if len(w) <= 3:
        return 1
    syl = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e"):
        syl -= 1
    return max(1, syl)


def _num_sents(text):
    punct = len(re.findall(r"[.!?]+", text))
    lines = sum(1 for ln in text.split("\n") if ln.strip())
    return max(punct, lines, 1)


def split_sents(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def tokens(text):
    return re.findall(r"[a-z']+", text.lower())


# ---------------- SARI ----------------
def _ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _f1(c, g):
    if not c and not g:
        return 1.0
    if not c or not g:
        return 0.0
    inter = len(c & g)
    p = inter / len(c); r = inter / len(g)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def sari_sentence(out, ref, src):
    o, r, s = out.split(), ref.split(), src.split()
    adds, keeps, dels = [], [], []
    for n in range(1, 5):
        on, rn, sn = set(_ngrams(o, n)), set(_ngrams(r, n)), set(_ngrams(s, n))
        adds.append(_f1(on - sn, rn - sn))
        keeps.append(_f1(on & sn, rn & sn))
        dels.append(_f1(sn - on, sn - rn))
    return (sum(adds) / 4 + sum(keeps) / 4 + sum(dels) / 4) / 3.0


def _best_align(sent, others):
    best, best_j = 0.0, None
    st = set(tokens(sent))
    for j, o in enumerate(others):
        ot = set(tokens(o))
        if not st or not ot:
            continue
        inter = len(st & ot)
        jac = inter / (len(st) + len(ot) - inter)
        if jac > best:
            best, best_j = jac, j
    return best_j


def sari_doc(out, ref, src):
    so, sr, ss = split_sents(out), split_sents(ref), split_sents(src)
    if not so:
        return 0.0
    scores = []
    for osent in so:
        rj = _best_align(osent, sr) if sr else None
        sj = _best_align(osent, ss) if ss else None
        ref_s = sr[rj] if rj is not None else ref
        src_s = ss[sj] if sj is not None else src
        scores.append(sari_sentence(osent, ref_s, src_s))
    return float(np.mean(scores))


# ---------------- readability ----------------
def rich_readability(text, terms):
    wl = tokens(text)
    n = len(wl)
    if n == 0:
        return dict.fromkeys(["fre", "fk_grade", "gunning_fog", "avg_word_len",
                              "ttr", "avg_sent_len", "term_density"], 0.0)
    sents = _num_sents(text)
    syl = sum(_syllables(w) for w in wl)
    fre = 206.835 - 1.015 * (n / sents) - 84.6 * (syl / n)
    fk = 0.39 * (n / sents) + 11.8 * (syl / n) - 15.59
    complex_w = sum(1 for w in wl if _syllables(w) >= 3)
    fog = 0.4 * (n / sents + 100 * complex_w / n)
    chars = sum(len(w) for w in wl)
    low = text.lower()
    term_hits = sum(1 for t in terms if t.lower() in low)
    return {
        "fre": float(fre), "fk_grade": float(fk), "gunning_fog": float(fog),
        "avg_word_len": float(chars / n), "ttr": float(len(set(wl)) / n),
        "avg_sent_len": float(n / sents), "term_density": float(term_hits / n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="B1")
    args = ap.parse_args()

    corpus = json.load(open(os.path.join(DATA, "corpus_v2.json"), encoding="utf-8"))
    raw_terms = corpus["terms"]
    terms = ([t["term"] if isinstance(t, dict) else t for t in raw_terms]
             if isinstance(raw_terms, list) else list(raw_terms.keys()))

    # ---- load references (B1) ----
    ref_path = os.path.join(RESULTS, f"{args.ref}.jsonl")
    refs = {}
    if os.path.isfile(ref_path):
        for l in open(ref_path, encoding="utf-8"):
            r = json.loads(l)
            refs[(r["id"], r["lv"])] = r["rew"]
    print(f"references: {args.ref} -> {len(refs)} pairs", flush=True)

    # ---- NLI model for contradiction recompute ----
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda:0"
    nli = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(MODELS, "NLI-DeBERTa-v3")).to(dev).eval()
    nli_tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "NLI-DeBERTa-v3"))

    def contra(rec):
        if rec.get("nli"):
            return rec["nli"]["contra"], rec["nli"]["entail"]
        try:
            enc = nli_tok(rec["src"][:4000], rec["rew"][:4000], return_tensors="pt",
                          truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                p = torch.softmax(nli(**enc).logits, -1)[0]
            return float(p[2].item()), float(p[0].item())
        except Exception:
            return None, None

    out = {"tags": {}, "sari_ref": args.ref}
    print("\n## 每配置 x 难度：SARI/BLEU/rich readability/矛盾率")
    print("| tag | lv | n | SARI | BLEU | FRE | FK | Fog | wLen | TTR | sLen | termDen | contra率 | sec |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in TAGS:
        path = os.path.join(RESULTS, f"{tag}.jsonl")
        if not os.path.isfile(path):
            print(f"[warn] missing {path}", flush=True)
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        out["tags"][tag] = {"levels": {}, "n": len(rows)}
        for lv in sorted(set(r["lv"] for r in rows)):
            sub = [r for r in rows if r["lv"] == lv]
            n = len(sub)
            # SARI / BLEU (only where refs exist)
            sari_scores, sys_sent, ref_sent = [], [], []
            for r in sub:
                ref = refs.get((r["id"], r["lv"]))
                if ref:
                    sari_scores.append(sari_doc(r["rew"], ref, r["src"]))
                sys_sent.append(tokens(r["rew"]))
                refs_for_bleu = refs.get((r["id"], r["lv"]))
                ref_sent.append([refs_for_bleu] if refs_for_bleu else None)
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            ref_sent = [x for x in ref_sent if x]
            sys_sent_b = [sys_sent[i] for i in range(len(sub)) if refs.get((sub[i]["id"], sub[i]["lv"]))]
            bleu = corpus_bleu(ref_sent, sys_sent_b,
                               smoothing_function=SmoothingFunction().method7) \
                if ref_sent and sys_sent_b else None
            # rich readability + contradiction
            rd = [rich_readability(r["rew"], terms) for r in sub]
            contras, entails = [], []
            for r in sub:
                c, e = contra(r)
                if c is not None:
                    contras.append(c); entails.append(e)
            secs = [r.get("sec") for r in sub if r.get("sec")]
            agg = {
                "n": n,
                "sari": float(np.mean(sari_scores)) if sari_scores else None,
                "bleu": float(bleu) if bleu is not None else None,
                "fre": float(np.mean([x["fre"] for x in rd])),
                "fk_grade": float(np.mean([x["fk_grade"] for x in rd])),
                "gunning_fog": float(np.mean([x["gunning_fog"] for x in rd])),
                "avg_word_len": float(np.mean([x["avg_word_len"] for x in rd])),
                "ttr": float(np.mean([x["ttr"] for x in rd])),
                "avg_sent_len": float(np.mean([x["avg_sent_len"] for x in rd])),
                "term_density": float(np.mean([x["term_density"] for x in rd])),
                "contra_rate": float(np.mean(contras)) if contras else None,
                "entail": float(np.mean(entails)) if entails else None,
                "mean_sec": float(np.mean(secs)) if secs else None,
            }
            out["tags"][tag]["levels"][lv] = agg
            fmt_row = (f"| {tag} | {lv[:4]} | {n} | {agg['sari'] if agg['sari'] is not None else '-':.2f} "
                       if False else None)
            sari_s = f"{agg['sari']:.2f}" if agg["sari"] is not None else "-"
            bleu_s = f"{agg['bleu']:.3f}" if agg["bleu"] is not None else "-"
            sec_s = f"{agg['mean_sec']:.1f}" if agg["mean_sec"] is not None else "-"
            contra_s = f"{agg['contra_rate']:.3f}" if agg["contra_rate"] is not None else "-"
            print(f"| {tag} | {lv[:4]} | {n} | {sari_s} | {bleu_s} | {agg['fre']:.1f} "
                  f"| {agg['fk_grade']:.1f} | {agg['gunning_fog']:.1f} "
                  f"| {agg['avg_word_len']:.1f} | {agg['ttr']:.2f} "
                  f"| {agg['avg_sent_len']:.1f} | {agg['term_density']:.3f} "
                  f"| {contra_s} | {sec_s} |")

    # ---- efficiency table ----
    print("\n## 效率表")
    print("| adapter | r | alpha | trainable_est | disk_GB |")
    print("|---|---|---|---|---|")
    eff = {"adapters": {}}
    for name in sorted(os.listdir(MODELS)):
        cfgp = os.path.join(MODELS, name, "adapter_config.json")
        if not os.path.isfile(cfgp):
            continue
        try:
            cfg = json.load(open(cfgp, encoding="utf-8"))
        except Exception:
            continue
        r = cfg.get("r"); alpha = cfg.get("lora_alpha")
        mods = cfg.get("target_modules", [])
        # Qwen3.5-4B text: 32 layers, hidden 2560
        train = 32 * len(mods) * r * 2 * 2560 if isinstance(mods, list) and r else None
        disk = sum(os.path.getsize(os.path.join(MODELS, name, f))
                   for f in os.listdir(os.path.join(MODELS, name))) / 1e9
        eff["adapters"][name] = {"r": r, "alpha": alpha, "n_modules": len(mods),
                                 "trainable_est": train, "disk_gb": round(disk, 2)}
        print(f"| {name} | {r} | {alpha} | {train if train else '-'} | {disk:.2f} |")
    for tag in ("Qwen3.5-4B", "Qwen3.5-9B", "Qwen3-4B-Instruct-2507"):
        p = os.path.join(MODELS, tag)
        if os.path.isdir(p):
            disk = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / 1e9
            eff.setdefault("models", {})[tag] = round(disk, 2)

    out["efficiency"] = eff
    out_path = os.path.join(RESULTS, "metrics_legal.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nmetrics -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
