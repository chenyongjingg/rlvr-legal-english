# -*- coding: utf-8 -*-
"""
08_verifier_eval.py — Verifier-as-a-Judge (task 20, round-2).
Judge model: Qwen3-4B-Instruct-2507 (cross-family, independent of the Qwen3.5
generator family; respects enable_thinking=False so it emits clean JSON fast).
Original Qwen3.5-9B emitted an unbounded 'Thinking Process:' preamble that
consumed the token budget (parsed 0/100); switched judge model for reliability.
Judges rewrites on 4 dimensions (1-5) + confidence [0,1].

Outputs:
  results/verifier_{tag}.jsonl   per-sample judge scores
  results/verifier_eval.json     aggregates: per-tag x level score means,
                                 ECE (proxy vs auto reward), self-consistency,
                                 Spearman correlation judge dims <-> auto metrics.

ECE proxy: judge mean score >=3.5 is its "good" verdict; auto tot>=0.80 is the
verifiable ground-truth label; ECE = |mean confidence - accuracy| per bin.
Self-consistency: K re-judgements (temp 0.8) on a seeded subset -> agreement.

Usage: python 08_verifier_eval.py [--tags G6,G1,B1,ADV] [--subset 40] [--k 3]
"""
import os, sys, json, re, argparse, random
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = "/root/autodl-tmp/legal_english/data"
MODELS = "/root/autodl-tmp/legal_english/models"
RESULTS = "/root/autodl-tmp/legal_english/results"

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


def parse_scores(out):
    """Robust JSON extraction: Qwen3.5 emits a thinking preamble before the JSON."""
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
    # fallback keyword scan
    def _num(k):
        m = re.search(k + r'["\']?\s*[:：]\s*([0-9.]+)', out, re.I)
        return float(m.group(1)) if m else None
    got = {k: _num(k) for k in ("difficulty", "faithfulness", "terminology",
                                "fluency", "confidence")}
    if all(v is not None for v in got.values()):
        return got
    return None


def strip_thinking(text):
    """Qwen3.5 emits a 'Thinking Process:' preamble even with enable_thinking=False.
    Drop everything up to the first blank line after the marker so the JSON tail
    survives within the token budget."""
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


def load_rows(tag):
    """Prefer the honest (commentary-cleaned) jsonl; drop pure-commentary rows.

    Honest records may not carry src/sec; merge them from the original jsonl
    by (id, lv) so the judge prompt still shows the source text."""
    # prefer the honest (commentary-cleaned) file; raw only for round-7 tags.
    for suffix in ("_honest", ""):
        p = os.path.join(RESULTS, f"{tag}{suffix}.jsonl")
        if os.path.isfile(p):
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            if suffix == "_honest":
                op = os.path.join(RESULTS, f"{tag}.jsonl")
                if os.path.isfile(op):
                    srcmap = {}
                    for r0 in (json.loads(l) for l in open(op, encoding="utf-8")):
                        srcmap[(r0.get("id"), r0.get("lv"))] = r0
                    for r in rows:
                        if not r.get("src"):
                            r0 = srcmap.get((r.get("id"), r.get("lv")))
                            if r0:
                                r["src"] = r0.get("src", "")
                                r["sec"] = r0.get("sec", "")
            return rows
    return None


def _rew(r):
    return (r.get("rew_clean") or r.get("rew") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="G6,G1,B1,ADV")
    ap.add_argument("--subset", type=int, default=40, help="self-consistency subset size")
    ap.add_argument("--k", type=int, default=3, help="self-consistency repeats")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="re-ask judge on parse failure")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    tags = [t for t in args.tags.split(",") if t]

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

    agg = {}
    for tag in tags:
        rows = load_rows(tag)
        if not rows:
            print(f"[warn] missing {tag} -> skip", flush=True)
            continue
        n_pure = sum(1 for r in rows if not _rew(r))
        rows = [r for r in rows if _rew(r)]
        if not rows:
            print(f"[warn] {tag}: all pure-commentary -> skip", flush=True)
            continue
        print(f"{tag}: {len(rows)} valid rewrites to judge (skipped {n_pure} "
              f"pure-commentary)", flush=True)
        # collect judge scores (with retry)
        scored = []
        for r in rows:
            for at in range(args.max_attempts):
                gen_kw = (dict(do_sample=False) if at == 0
                          else dict(do_sample=True, temperature=0.6, top_p=0.9))
                sc, raw = judge_once(model, tok, r["src"], _rew(r), r["lv"], dev, gen_kw)
                if sc is not None:
                    break
            scored.append({"id": r["id"], "lv": r["lv"], "src": r["src"],
                           "rew": _rew(r), "tot": r["tot"],
                           "parts": r.get("parts", {}), "fre": r.get("fre", 0.0),
                           "judge": sc, "raw": raw})
        out = os.path.join(RESULTS, f"verifier_{tag}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for x in scored:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        n_ok = sum(1 for x in scored if x["judge"] is not None)
        print(f"{tag}: judged {n_ok}/{len(scored)}", flush=True)

        # ---- per-tag x level aggregates ----
        tag_agg = {"levels": {}}
        dims = ("difficulty", "faithfulness", "terminology", "fluency")
        for lv in sorted(set(x["lv"] for x in scored)):
            sub = [x for x in scored if x["lv"] == lv and x["judge"]]
            if not sub:
                continue
            lv_agg = {"n": len(sub)}
            for d in dims:
                lv_agg[d] = float(np.mean([x["judge"][d] for x in sub]))
            lv_agg["confidence"] = float(np.mean([x["judge"]["confidence"] for x in sub]))
            # mean auto tot + diff for reference
            lv_agg["auto_tot"] = float(np.mean([x["tot"] for x in sub]))
            lv_agg["auto_diff"] = float(np.mean([x["parts"].get("diff", 0.0) for x in sub]))
            tag_agg["levels"][lv] = lv_agg

        # ---- ECE proxy over all judged samples ----
        judged = [x for x in scored if x["judge"]]
        if judged:
            confs = np.array([x["judge"]["confidence"] for x in judged], float)
            judge_good = np.array([np.mean([x["judge"][d] for d in dims]) >= 3.5
                                   for x in judged])
            auto_good = np.array([x["tot"] >= 0.80 for x in judged])
            pred_good = judge_good & (confs >= 0.5) | (~judge_good & (confs < 0.5))
            # correctness = judge verdict matches auto label
            correct = (judge_good == auto_good)
            bins = np.linspace(0, 1, 6)      # 5 bins
            ece = 0.0
            bin_stats = []
            for b in range(5):
                m = (confs > bins[b]) & (confs <= bins[b + 1])
                if m.sum() == 0:
                    continue
                acc = correct[m].mean()
                conf_b = confs[m].mean()
                ece += (m.sum() / len(confs)) * abs(acc - conf_b)
                bin_stats.append({"bin": f"({bins[b]:.1f},{bins[b+1]:.1f}]",
                                  "n": int(m.sum()), "acc": float(acc),
                                  "conf": float(conf_b)})
            tag_agg["ece"] = float(ece)
            tag_agg["ece_bins"] = bin_stats
            tag_agg["judge_good_frac"] = float(judge_good.mean())
            tag_agg["auto_good_frac"] = float(auto_good.mean())
            tag_agg["agree_frac"] = float(correct.mean())

            # ---- Spearman: judge dims <-> auto metrics ----
            corr = {}
            for d in dims:
                for mkey, mfun in (("diff", "diff"), ("term", "term"),
                                   ("copy", "copy"), ("faith", "faith")):
                    vals = np.array([x["parts"].get(mfun, 0.0) for x in judged], float)
                    jv = np.array([x["judge"][d] for x in judged], float)
                    if np.std(vals) and np.std(jv):
                        rho, p = sp_spearman(jv, vals)
                        corr[f"{d}_vs_{mkey}"] = {"rho": float(rho), "p": float(p)}
            tag_agg["spearman"] = corr

        # ---- self-consistency: K re-judgements on a seeded subset ----
        if args.k > 1 and judged:
            rng2 = random.Random(args.seed)
            subset = rng2.sample(judged, min(args.subset, len(judged)))
            cons_rows = []
            for x in subset:
                scs = []
                for _ in range(args.k):
                    sc, _ = judge_once(model, tok, x["src"], x["rew"], x["lv"],
                                       dev, dict(do_sample=True, temperature=0.8, top_p=0.9))
                    if sc is not None:
                        scs.append(sc)
                cons_rows.append((x["id"], scs))
            agree = []
            mean_score = []
            for _, scs in cons_rows:
                if len(scs) < 2:
                    continue
                totv = [np.mean([s[d] for d in dims]) for s in scs]
                mean_score.append(totv)
                # pairwise agreement on verdict (mean>=3.5)
                ver = [t >= 3.5 for t in totv]
                pairs = [(i, j) for i in range(len(ver)) for j in range(i + 1, len(ver))]
                if pairs:
                    agree.append(np.mean([1.0 if ver[i] == ver[j] else 0.0
                                          for i, j in pairs]))
            tag_agg["self_consistency"] = {
                "n": len(subset), "k": args.k,
                "verdict_agree": float(np.mean(agree)) if agree else None,
                "mean_score_std": float(np.mean([np.std(t) for t in mean_score]))
                if mean_score else None}
        agg[tag] = tag_agg
        print(f"  {tag}: ECE={tag_agg.get('ece', float('nan')):.3f} "
              f"agree={tag_agg.get('agree_frac', float('nan')):.3f} "
              f"selfcons={tag_agg.get('self_consistency', {}).get('verdict_agree')}",
              flush=True)

    out_path = os.path.join(RESULTS, "verifier_eval.json")
    json.dump(agg, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"verifier aggregate -> {out_path}", flush=True)


def sp_spearman(a, b):
    import scipy.stats as sp
    rho, p = sp.spearmanr(a, b)
    return rho, p


if __name__ == "__main__":
    main()
