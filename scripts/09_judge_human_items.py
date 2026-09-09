# -*- coding: utf-8 -*-
"""
09_judge_human_items.py — judge the EXACT 40 pilot items x 6 systems that the
human raters saw (Applied Intelligence resubmission, H1B).

Why: the 3-rater blind pilot came back with inter-annotator kappa ~= 0 AND a
system ranking that flips the automatic one (humans put the framework G6v4
near-bottom; auto tot ranks it top). Before the manuscript's machine-side
claims are locked, we run the SAME independent judge used in 08_verifier_eval.py
(Qwen3-4B-Instruct-2507) over the identical (item, system) rewrites the humans
scored, so we can answer:
   (a) does the judge replicate the humans (G6v4 low) or the auto stack (G6v4
       high) on these 40 items?
   (b) how correlated is the judge with the (unreliable) pilot humans at all?
   (c) per-system x dimension judge means -> candidate numbers for the paper's
       "independent evaluator" table.

Judge prompt / parsing / retry are copied verbatim from 08_verifier_eval.py so
scores are directly comparable with the verifier_*.jsonl outputs.

Reads the human-eval sample file (items.jsonl: {id,lv,src,systems{Source,B2,
M2,FlanT5,BART,G6: {rew,fre,tot,...}}}). Judging Source = judging the raw legal
text at the target level (it is an anchor, like the humans rated it).
Every non-empty rewrite is judged exactly once (greedy attempt 0; one sampled
retry on parse failure).

Outputs:
  results/human40_judge.jsonl   per (item, system) judge score
  plus an on-server quick per-system x dimension mean table (so the pivotal
  ordering is visible before any file is downloaded).

Usage (server):
  /root/miniconda3/bin/python 09_judge_human_items.py [--items <path>]
Default --items = /path/to/legal_english/human_eval/items.jsonl
"""
import os, sys, json, re, argparse, random
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = "/path/to/legal_english/data"
MODELS = "/path/to/legal_english/models"
RESULTS = "/path/to/legal_english/results"
LEVEL_LABEL = {"beginner": "beginner", "intermediate": "intermediate",
               "advanced": "advanced"}

JUDGE_SYS = ("You are an expert evaluator of simplified legal texts for English "
             "learners. You rate rewrites strictly and objectively, always returning "
             "a single JSON object and nothing else.")
JUDGE_USR = ("Rate this simplified legal rewrite for a {lv}-level English learner.\n"
             "Dimensions (1 = very poor, 5 = excellent):\n"
             "- difficulty: does the text difficulty match the {lv} level? "
             "(beginner: short everyday sentences; intermediate: clear formal text; "
             "advanced: precise legal register)\n"
             "- faithfulness: are ALL legal facts, obligations and conditions of the "
             "original preserved, with nothing added or dropped?\n"
             "- terminology: are key legal terms kept verbatim and used accurately?\n"
             "- fluency: is the English natural, grammatical and readable?\n"
             "Also output a confidence in [0,1] for your overall assessment.\n"
             "Reply with ONLY this JSON:\n"
             '{{"difficulty": <int 1-5>, "faithfulness": <int 1-5>, '
             '"terminology": <int 1-5>, "fluency": <int 1-5>, '
             '"confidence": <float 0-1>}}\n\n'
             "ORIGINAL LEGAL TEXT:\n{src}\n\n"
             "SIMPLIFIED REWRITE ({lv}):\n{rew}\n")
SYSTEMS = ["Source", "B2", "M2", "FlanT5", "BART", "G6"]
DIMS = ("difficulty", "faithfulness", "terminology", "fluency")


def parse_scores(out):
    if not out:
        return None
    for m in re.findall(r"\{[^{}]*\}", out):
        try:
            d = json.loads(m)
            keys = ("difficulty", "faithfulness", "terminology", "fluency", "confidence")
            if all(k in d for k in keys):
                return {k: d[k] for k in keys}
        except Exception:
            continue
    def _num(k):
        m = re.search(k + r'["\']?\s*[:：]\s*([0-9.]+)', out, re.I)
        return float(m.group(1)) if m else None
    got = {k: _num(k) for k in ("difficulty", "faithfulness", "terminology",
                                "fluency", "confidence")}
    if all(v is not None for v in got.values()):
        return got
    return None


def strip_thinking(text):
    if "Thinking Process:" in text:
        idx = text.find("Thinking Process:")
        rest = text[idx:]
        nl = rest.find("\n\n")
        text = rest[nl + 2:] if nl >= 0 else rest
    return text


def judge_once(model, tok, src, rew, lv, dev, gen_kw):
    msgs = [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": JUDGE_USR.format(
                lv=LEVEL_LABEL[lv], src=src[:3000], rew=rew[:3000])}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  chat_template_kwargs={"enable_thinking": False},
                                  return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp.to(dev), max_new_tokens=512,
                             pad_token_id=tok.pad_token_id, **gen_kw)
    txt = tok.decode(out[0][inp["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    txt = strip_thinking(txt)
    return parse_scores(txt), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(
        DATA.replace("data", "human_eval"), "items.jsonl"),
        help="human-eval items.jsonl (upload from local samples/)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if not os.path.isfile(args.items):
        sys.exit(f"[error] items file not found: {args.items}")

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    # (item, system) -> rew
    jobs = []
    for it in items:
        for sysname in SYSTEMS:
            rec = it["systems"].get(sysname) or {}
            rew = (rec.get("rew") or "").strip()
            jobs.append({"id": it["id"], "lv": it["lv"], "src": it["src"],
                         "system": sysname, "rew": rew,
                         "tot": rec.get("tot"), "fre": rec.get("fre")})
    n_empty = sum(1 for j in jobs if not j["rew"])
    jobs = [j for j in jobs if j["rew"]]
    print(f"human items: {len(items)} items x {len(SYSTEMS)} systems = "
          f"{len(items)*len(SYSTEMS)} rewrites, {n_empty} empty -> {len(jobs)} to judge",
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    mid = os.path.join(MODELS, "Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        mid, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = model.device
    print(f"judge: Qwen3-4B-Instruct-2507 loaded on {dev}", flush=True)

    scored = []
    for j in jobs:
        for at in range(args.max_attempts):
            gen_kw = (dict(do_sample=False) if at == 0
                      else dict(do_sample=True, temperature=0.6, top_p=0.9))
            sc, raw = judge_once(model, tok, j["src"], j["rew"], j["lv"], dev, gen_kw)
            if sc is not None:
                break
        scored.append({"id": j["id"], "lv": j["lv"], "system": j["system"],
                       "src": j["src"], "rew": j["rew"],
                       "tot": j["tot"], "fre": j["fre"],
                       "judge": sc, "raw": raw})
        if (len(scored) % 40) == 0:
            print(f"  ...judged {len(scored)}/{len(jobs)}", flush=True)

    out = os.path.join(RESULTS, "human40_judge.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for x in scored:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    n_ok = sum(1 for x in scored if x["judge"])
    print(f"wrote {out}: {n_ok}/{len(scored)} judged", flush=True)

    # ---- on-server quick per-system x dim means (pivotal ordering) ----
    ok = [x for x in scored if x["judge"]]
    print("\n==== per-system judge means over the 40 pilot items ====", flush=True)
    print("%-9s" % "system" + "".join("%13s" % d for d in DIMS) +
          "%9s" % "overall" + "%9s" % "n", flush=True)
    order = {}
    for s in SYSTEMS:
        sub = [x for x in ok if x["system"] == s]
        if not sub:
            print("%-9s  (no judged rows)" % s, flush=True)
            continue
        row = {}
        for d in DIMS:
            row[d] = float(np.mean([x["judge"][d] for x in sub]))
        row["overall"] = float(np.mean([np.mean([x["judge"][d] for d in DIMS])
                                        for x in sub]))
        order[s] = row["overall"]
        print("%-9s" % s + "".join("%13.2f" % row[d] for d in DIMS) +
              "%9.2f" % row["overall"] + "%9d" % len(sub), flush=True)
    rank = sorted(order, key=order.get, reverse=True)
    print("judge overall ranking (hi->lo): " + " > ".join(
        f"{s}({order[s]:.2f})" for s in rank), flush=True)
    if rank and rank[0] != "G6":
        print("[NOTE] judge does NOT rank G6 (framework) top on the pilot "
              "items -- do NOT report framework superiority on this subset "
              "before seeing H1C.", flush=True)


if __name__ == "__main__":
    main()
