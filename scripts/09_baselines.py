# -*- coding: utf-8 -*-
"""
09_baselines.py — external/ablative baselines (task 21, round-2).
  B1 (--method b1): Qwen3.5-9B zero-shot rewrite (no tuning, no RAG, no agents).
  B2 (--method b2): deterministic lexical baseline (CPU): sentence/clause splitting
      tuned per level + legal-term glosses + small legalese->plain substitution.
Output: results/{tag}.jsonl with the SAME record schema as 06_framework (incl. nli,
sec) so 07_eval / 10_metrics can compare directly with G1-G6.

Usage: python 09_baselines.py --method b1|b2 [--tag B1|B2] [--n-test 50]
"""
import os, sys, json, re, argparse, random, time, types
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = "/path/to/legal_english/data"
MODELS = "/path/to/legal_english/models"
RESULTS = "/path/to/legal_english/results"

LEVEL_TARGET_FRE = {"beginner": 63.0, "intermediate": 45.0, "advanced": 26.0}
W = {"fmt": 0.15, "diff": 0.20, "term": 0.15, "copy": 0.25, "faith": 0.25}
SYSTEM = ("You are a legal English educator. You rewrite legal texts into clear, "
          "level-appropriate versions for English learners. You always output "
          "ONLY the requested rewrite, starting directly with it.")
USER = ("Rewrite the following legal text into a {lv}-level version for English "
        "learners. Preserve legal meaning and key legal terms.\n\n{src}")
THINK_MARKERS = ["thinking process", "analyze the request", "thought process",
                 "let me think", "思考过程", "分析需求", "here's a thinking", "<think"]
META_PATTERNS = ["this rewrite", "this version", "this simplified", "this explanation",
                 "in summary", "to summarize", "in conclusion", "as requested",
                 "i hope this", "hope this helps", "i have rewritten", "i've rewritten",
                 "the above", "this is a simplified", "this keeps the main ideas",
                 "it simplifies", "the original legal text", "i have simplified",
                 "here is", "below is", "this passage", "i tried", "let me know"]

# legalese -> plain substitutions (small, safe) for the lexical baseline
LEGAL2PLAIN = [
    (r"\bpursuant to\b", "under"), (r"\bin accordance with\b", "under"),
    (r"\bprior to\b", "before"), (r"\bnotwithstanding\b", "despite"),
    (r"\bprovided that\b", "but"), (r"\bwhereby\b", "in which"),
    (r"\bhereinafter\b", "later in this text"), (r"\bhereunder\b", "under this"),
    (r"\baforesaid\b", "mentioned earlier"), (r"\bin the event that\b", "if"),
    (r"\bfor the purpose of\b", "to"), (r"\bwithout prejudice to\b", "without affecting"),
]
SPLIT_PATTERNS = [r";", r", and ", r", but ", r" provided that ", r" unless ",
                  r" where ", r" whereas ", r", which ", r" in order to "]


# ---------------- metric / reward helpers (mirror 06_framework) ----------------
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


def flesch(text):
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    sents = _num_sents(text)
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sents) - 84.6 * (syl / len(words))


def r_fmt(rew, src):
    r = 1.0
    if not rew or len(rew.split()) < 5:
        return 0.0
    low = rew.lower()
    if any(m in low for m in THINK_MARKERS):
        r -= 0.5
    if "rewrite the following" in low[:60]:
        r -= 0.3
    hits = sum(1 for m in META_PATTERNS if m in low)
    if hits:
        r -= min(0.6, 0.3 * hits)
    if abs(len(rew.split()) - len(src.split())) > 0.9 * len(src.split()):
        r -= 0.3
    return max(0.0, min(1.0, r))


def r_diff(rew, lv):
    return max(0.0, min(1.0, 1.0 - abs(flesch(rew) - LEVEL_TARGET_FRE[lv]) / 20.0))


def r_term(rew, src, terms):
    src_low, rew_low = src.lower(), rew.lower()
    present = [t for t in terms if t.lower() in src_low]
    if not present:
        return 1.0
    kept = sum(1 for t in present if t.lower() in rew_low)
    return kept / len(present)


def r_copy(rew, src):
    rw = re.findall(r"[a-z']+", rew.lower())
    sw = re.findall(r"[a-z']+", src.lower())
    if len(rw) < 8 or len(sw) < 8:
        return 0.5
    rset = set(tuple(rw[i:i + 3]) for i in range(len(rw) - 2))
    sset = set(tuple(sw[i:i + 3]) for i in range(len(sw) - 2))
    if not sset:
        return 1.0
    overlap = len(rset & sset) / len(sset)
    return max(0.0, min(1.0, 1.0 - max(0.0, (overlap - 0.4) / 0.4)))


def rewards(rew, src, lv, terms, nli_model, nli_tok, nli_dev):
    rf, rd = r_fmt(rew, src), r_diff(rew, lv)
    rt, rc = r_term(rew, src, terms), r_copy(rew, src)
    try:
        enc = nli_tok(src[:4000], rew[:4000], return_tensors="pt",
                      truncation=True, max_length=512).to(nli_dev)
        with torch.no_grad():
            out = nli_model(**enc)
        p = torch.softmax(out.logits, dim=-1)[0].tolist()
        ent, neu, con = float(p[0]), float(p[1]), float(p[2])
    except Exception:
        ent = neu = con = 0.0
    tot = (W["fmt"] * rf + W["diff"] * rd + W["term"] * rt +
           W["copy"] * rc + W["faith"] * ent)
    nli = {"entail": round(ent, 4), "neutral": round(neu, 4), "contra": round(con, 4)}
    return tot, {"fmt": rf, "diff": rd, "term": rt, "copy": rc, "faith": ent}, \
        flesch(rew), nli


# ---------------- Qwen3.5 thinking-preamble stripper ----------------
def strip_thinking(s):
    """Qwen3.5 (enable_thinking=False) can still emit a 'Thinking Process:' block
    on zero-shot calls. Strip it when clearly present."""
    s = s.strip()
    low = s.lower()
    m = low.find("thinking process:")
    if m == -1:
        m = low.find("thought process:")
    if m == -1:
        m = low.find("<think")
    if m == -1:
        return s
    rest = s[m:]
    ti = rest.lower().find("text:")
    if ti != -1 and ti < 5000:
        return rest[ti + 5:].lstrip("\n ")
    nl = rest.find("\n\n")
    if nl != -1:
        return rest[nl + 2:].strip()
    return rest.strip()


# ---------------- B2 lexical baseline ----------------
def split_sents(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def lexical_simplify(src, lv, terms, term_defs):
    tgt_words = {"beginner": 12, "intermediate": 18, "advanced": 26}[lv]
    out = []
    seen_terms = set()
    for sent in split_sents(src):
        sent = sent.strip()
        # split into clauses on the split patterns (respect target length)
        clauses = [sent]
        if lv != "advanced":
            for pat in SPLIT_PATTERNS:
                nxt = []
                for cl in clauses:
                    if len(cl.split()) > tgt_words:
                        pieces = re.split(pat, cl, flags=re.I)
                        if len(pieces) > 1:
                            nxt.extend(pieces)
                        else:
                            nxt.append(cl)
                    else:
                        nxt.append(cl)
                clauses = nxt
        # trim each clause to target length (hard cap), keeping sentence fragments joined
        trimmed = []
        for cl in clauses:
            wl = cl.split()
            if len(wl) > tgt_words:
                wl = wl[:tgt_words]
                cl = " ".join(wl)
            trimmed.append(cl)
        out.extend(trimmed)
    # join clauses into sentences (beginner: one clause per sentence; else group 2)
    group = 1 if lv == "beginner" else (2 if lv == "intermediate" else 3)
    sents = []
    for i in range(0, len(out), group):
        chunk = out[i:i + group]
        text = " ".join(chunk).strip()
        if text and not text.endswith((".", "!", "?")):
            text += "."
        sents.append(text)
    text = " ".join(sents)
    # substitution
    for pat, rep in LEGAL2PLAIN:
        text = re.sub(pat, rep, text, flags=re.I)
    # gloss first occurrence of each key legal term (beginner only)
    low = text.lower()
    if lv == "beginner":
        for t in terms:
            if t.lower() in low and t.lower() not in seen_terms:
                seen_terms.add(t.lower())
                d = term_defs.get(t, "")
                if d:
                    # insert gloss right after the term's first occurrence
                    text = re.sub(r"(?<!\w)" + re.escape(t) + r"(?!\w)",
                                  f"{t} ({d.strip()})", text, count=1, flags=re.I)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["b1", "b2"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--levels", default="beginner,intermediate")
    args = ap.parse_args()
    tag = args.tag or ("B1" if args.method == "b1" else "B2")
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    os.makedirs(RESULTS, exist_ok=True)

    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    corpus = json.load(open(os.path.join(DATA, "corpus_v2.json"), encoding="utf-8"))
    raw_terms = corpus["terms"]
    terms = ([t["term"] if isinstance(t, dict) else t for t in raw_terms]
             if isinstance(raw_terms, list) else list(raw_terms.keys()))
    term_defs = {t["term"]: t.get("definition", "") for t in raw_terms
                 if isinstance(t, dict)}
    test_srcs = split["test"][:args.n_test]
    print(f"{tag}: method={args.method} n_test={len(test_srcs)} "
          f"levels={levels}", flush=True)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    nli_dev = "cuda:0"
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(MODELS, "NLI-DeBERTa-v3")).to(nli_dev).eval()
    nli_tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "NLI-DeBERTa-v3"))

    if args.method == "b1":
        from transformers import (AutoModelForImageTextToText, BitsAndBytesConfig)
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True)
        mid = os.path.join(MODELS, "Qwen3.5-9B")
        model = AutoModelForImageTextToText.from_pretrained(
            mid, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        model.eval()
        tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print("B1: Qwen3.5-9B zero-shot loaded", flush=True)

    out_path = os.path.join(RESULTS, f"{tag}.jsonl")
    fout = open(out_path, "w", encoding="utf-8")
    t0 = time.time(); n_done = 0; n_total = len(test_srcs) * len(levels)
    if args.method == "b1":
        gen_kw = dict(max_new_tokens=320, do_sample=True, temperature=0.9, top_p=0.95,
                      pad_token_id=tok.pad_token_id)
    else:
        gen_kw = None
    for s in test_srcs:
        for lv in levels:
            t_item = time.time()
            src = s["text"]
            if args.method == "b1":
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER.format(lv=lv, src=src)}]
                inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              chat_template_kwargs={"enable_thinking": False},
                                              return_tensors="pt")
                with torch.no_grad():
                    out = model.generate(**inp.to(model.device),
                                         num_return_sequences=1, **gen_kw)
                rew = tok.decode(out[0][inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
                rew = strip_thinking(rew)
            else:
                rew = lexical_simplify(src, lv, terms, term_defs)
            tot, parts, fre, nli = rewards(rew, src, lv, terms,
                                           nli_model, nli_tok, nli_dev)
            rec = {"id": s["id"], "lv": lv, "src": src, "rew": rew,
                   "tot": round(tot, 4), "parts": parts, "fre": round(fre, 2),
                   "attempts": 1, "nli": nli, "sec": round(time.time() - t_item, 2)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 20 == 0 or n_done == n_total:
                print(f"  {n_done}/{n_total} done ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    print(f"done: {out_path}", flush=True)


if __name__ == "__main__":
    main()
