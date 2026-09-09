# -*- coding: utf-8 -*-
"""
GRPO / RLVR training for Qwen3.5-4B on legal rewrite (task 15).
Policy: Qwen3.5-4B composite + LoRA, initialized from the SFT adapter.
Verifiable rewards per proposal:
  r_fmt   format compliance (clean rewrite, no reasoning preamble, sane length)
  r_diff  readability difficulty match (Flesch RE band per level)
  r_term  key legal-term preservation (terms from source retained in rewrite)
  r_faith NLI entailment (rewrite entails source; DeBERTa-v3 NLI)
Group z-score advantages + per-token KL penalty vs the SFT reference.
Manual loop (no TRL dependency for the composite model).
"""
import os, sys, json, argparse, random, math, re, types
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = "/path/to/legal_english/data"
MODELS = "/path/to/legal_english/models"

SYSTEM = ("You are a legal English educator. You rewrite legal texts into clear, "
          "level-appropriate versions for language learners. You always output "
          "ONLY the requested rewrite, starting directly with it.")
USER = ("Rewrite the following legal text into a {lv}-level version for English "
        "learners. Preserve legal meaning and key legal terms.\n\n{src}")

LEVEL_TARGET_FRE = {"beginner": 63.0, "intermediate": 45.0}   # recalibrated w/ robust sents (flesch_probe)
W = {"fmt": 0.15, "diff": 0.20, "term": 0.15, "copy": 0.25, "faith": 0.25}
THINK_MARKERS = ["thinking process", "analyze the request", "thought process",
                 "let me think", "思考过程", "分析需求", "here's a thinking", "<think"]
# trailing meta-commentary the model appends after its rewrite (format violation)
META_PATTERNS = [
    "this rewrite", "this version", "this simplified", "this explanation",
    "in summary", "to summarize", "in conclusion", "as requested",
    "i hope this", "hope this helps", "i have rewritten", "i've rewritten",
    "the above", "this is a simplified", "this keeps the main ideas",
    "it simplifies", "the original legal text", "i have simplified",
    "here is", "below is", "this passage", "i tried", "let me know",
    "this is a beginner", "this is an intermediate", "this is a simple",
]


def enable_peft_on_text_model(lm):
    if not hasattr(lm, "prepare_inputs_for_generation"):
        def _prep(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]
            out = {"input_ids": input_ids, "attention_mask": attention_mask,
                   "past_key_values": past_key_values}
            out.update(kwargs)
            return out
        lm.prepare_inputs_for_generation = types.MethodType(_prep, lm)
    return lm


def _num_sents(text):
    """Robust sentence count: punctuation boundaries OR newline-separated lines.
    Legal definitions/lists often carry no terminal period (brittle for FRE)."""
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


def _syllables(w):
    w = w.lower()
    if len(w) <= 3:
        return 1
    syl = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e"):
        syl -= 1
    return max(1, syl)


def r_fmt(rew, src, tok):
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
        r -= 0.3          # implausible length drift
    return max(0.0, min(1.0, r))


def r_copy(rew, src):
    """Anti-copy: fraction of SOURCE trigrams reused verbatim in the rewrite.
    Near-verbatim echoes (the 'intermediate' lazy-copy failure) -> ~0; a genuine
    rewrite shares few source trigrams -> ~1."""
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


def r_diff(rew, lv):
    fre = flesch(rew)
    target = LEVEL_TARGET_FRE[lv]
    return max(0.0, min(1.0, 1.0 - abs(fre - target) / 20.0))


def r_term(rew, src, terms):
    src_low, rew_low = src.lower(), rew.lower()
    present = [t for t in terms if t.lower() in src_low]
    if not present:
        return 1.0           # no key terms in source -> vacuous
    kept = sum(1 for t in present if t.lower() in rew_low)
    return kept / len(present)


def r_faith(rew, src, nli_model, nli_tok, nli_dev):
    """NLI: does the rewrite entail the source? batch not needed (caller batches)."""
    try:
        enc = nli_tok(src[:4000], rew[:4000], return_tensors="pt",
                      truncation=True, max_length=512).to(nli_dev)
        with torch.no_grad():
            out = nli_model(**enc)
        # DeBERTa-v3 NLI labels: entailment=0, neutral=1, contradiction=2
        p = torch.softmax(out.logits, dim=-1)[0]
        return float(p[0].item())
    except Exception:
        return 0.0


def total_reward(rew, src, lv, terms, nli_model, nli_tok, nli_dev, tok):
    rf = r_fmt(rew, src, tok)
    rd = r_diff(rew, lv)
    rt = r_term(rew, src, terms)
    rc = r_copy(rew, src)
    rfa = r_faith(rew, src, nli_model, nli_tok, nli_dev)
    return (W["fmt"] * rf + W["diff"] * rd + W["term"] * rt +
            W["copy"] * rc + W["faith"] * rfa,
            {"fmt": rf, "diff": rd, "term": rt, "copy": rc, "faith": rfa})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--sft-adapter", default=os.path.join(MODELS, "sft_qwen35_lora"),
                    help="start checkpoint: initial LoRA for both policy and reference")
    ap.add_argument("--init-adapter", default=None,
                    help="override start checkpoint (e.g. continue from a GRPO adapter)")
    ap.add_argument("--diff-w", type=float, default=0.20,
                    help="difficulty-match weight; other components renormalize to sum 1")
    ap.add_argument("--drop", default="",
                    help="comma-separated reward components to DROP (fmt,diff,term,copy,faith); "
                         "dropped weights -> 0, retained renormalized to sum 1")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--G", type=int, default=6)          # group size
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.04)  # KL penalty
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=os.path.join(MODELS, "grpo_qwen35_lora"))
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # start checkpoint: default SFT adapter; --init-adapter overrides (extension runs)
    start_ckpt = args.init_adapter or args.sft_adapter
    # reward weights: --diff-w sets difficulty weight (renormalize others); --drop zeroes
    # selected components and renormalizes the retained ones to sum 1 (reward ablation).
    dropped = {x.strip() for x in args.drop.split(",") if x.strip()}
    for d in dropped:
        if d not in W:
            raise SystemExit(f"unknown --drop component: {d}")
    if dropped:
        for d in dropped:
            W[d] = 0.0
        if "diff" in W and "diff" not in dropped:
            W["diff"] = args.diff_w
            rest = {k: v for k, v in W.items() if k != "diff" and k not in dropped}
            scale = (1.0 - args.diff_w) / max(sum(rest.values()), 1e-9)
            for k in rest:
                W[k] *= scale
        else:
            s = sum(W.values())
            for k in W:
                W[k] /= max(s, 1e-9)
    else:
        others = {k: v for k, v in W.items() if k != "diff"}
        scale = (1.0 - args.diff_w) / max(sum(others.values()), 1e-9)
        W["diff"] = args.diff_w
        for k, v in others.items():
            W[k] = v * scale
    print(f"start adapter: {start_ckpt}", flush=True)
    print(f"reward weights: {W}  dropped={sorted(dropped)}", flush=True)

    from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = os.path.join(args.models, "Qwen3.5-4B")
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- policy + reference (both share LoRA from start_ckpt; policy continues training) ---
    print(f"loading policy Qwen3.5-4B + {start_ckpt}...", flush=True)
    ref = AutoModelForImageTextToText.from_pretrained(
        base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    ref.eval()
    lm_ref = ref.model.language_model
    enable_peft_on_text_model(lm_ref)
    from peft import PeftModel
    lm_ref = PeftModel.from_pretrained(lm_ref, start_ckpt, is_trainable=False)
    ref.model.language_model = lm_ref
    for p in ref.parameters():
        p.requires_grad = False
    print("reference ready (start adapter, frozen)", flush=True)

    # policy = fresh copy; we re-load base and attach adapter, then unfreeze for LoRA grad
    policy = AutoModelForImageTextToText.from_pretrained(
        base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    policy.train()
    lm_p = policy.model.language_model
    enable_peft_on_text_model(lm_p)
    lm_p = PeftModel.from_pretrained(lm_p, start_ckpt, is_trainable=True)
    policy.model.language_model = lm_p
    policy.gradient_checkpointing_enable()
    n_tr = sum(p.numel() for p in lm_p.parameters() if p.requires_grad)
    print(f"policy trainable params: {n_tr:,}", flush=True)

    # --- NLI model for r_faith ---
    print("loading NLI (DeBERTa-v3)...", flush=True)
    from transformers import AutoModelForSequenceClassification
    nli_dev = "cuda:0"
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(args.models, "NLI-DeBERTa-v3")).to(nli_dev).eval()
    nli_tok = AutoTokenizer.from_pretrained(os.path.join(args.models, "NLI-DeBERTa-v3"))

    # --- data ---
    split = json.load(open(os.path.join(args.data, "split.json"), encoding="utf-8"))
    corpus = json.load(open(os.path.join(args.data, "corpus_v2.json"), encoding="utf-8"))
    raw_terms = corpus["terms"]
    if isinstance(raw_terms, list):
        terms = [t["term"] if isinstance(t, dict) else t for t in raw_terms]
    else:
        terms = list(raw_terms.keys())
    train = split["train"]
    rng = random.Random(args.seed)
    prompts = rng.sample(train, min(args.n_train, len(train)))
    for i, pr in enumerate(prompts):
        pr["_lv"] = "beginner" if i % 2 == 0 else "intermediate"
    print(f"GRPO prompts: {len(prompts)}, group G={args.G}, terms={len(terms)}", flush=True)

    opt = torch.optim.AdamW([p for p in lm_p.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    # prompt token cache
    prom_cache = []
    for pr in prompts:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(lv=pr["_lv"], src=pr["text"])}]
        inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      chat_template_kwargs={"enable_thinking": False},
                                      return_tensors="pt")
        prom_cache.append((pr, inp))

    gen_kw = dict(max_new_tokens=320, do_sample=True, temperature=0.9, top_p=0.95,
                  pad_token_id=tok.pad_token_id)
    for ep in range(args.epochs):
        ep_stats = {k: 0.0 for k in ("rew", "fmt", "diff", "term", "copy", "faith")}
        ep_loss, n_upd = 0.0, 0
        rng.shuffle(prom_cache)
        for pr, inp in prom_cache:
            src, lv = pr["text"], pr["_lv"]
            # --- rollouts ---
            # generate in EVAL mode: train-mode dropout during autoregressive
            # generation collapses to token soup (observed). Loss forward runs
            # in train mode below so LoRA gradients still flow.
            policy.eval()
            with torch.no_grad():
                gen = policy.generate(**inp.to(policy.device), num_return_sequences=args.G,
                                      **gen_kw)
            policy.train()
            plen = inp["input_ids"].shape[1]
            seq_ids = gen[:, :]                     # [G, plen + T]
            rewrites = [tok.decode(g[plen:], skip_special_tokens=True).strip() for g in gen]
            rewards, reward_parts = [], []
            for rew in rewrites:
                tot, parts = total_reward(rew, src, lv, terms, nli_model, nli_tok, nli_dev, tok)
                rewards.append(tot); reward_parts.append(parts)
            rew_t = torch.tensor(rewards, dtype=torch.float32, device=policy.device)
            adv = (rew_t - rew_t.mean()) / (rew_t.std() + 1e-6)
            if args.verbose:
                for gi, (rew, parts) in enumerate(zip(rewards, reward_parts)):
                    print(f"    [{pr['id']}|{lv}] G{gi} rew={rew:.3f} "
                          f"fre={flesch(rewrites[gi]):.1f} {parts} "
                          f"words={len(rewrites[gi].split())}", flush=True)
                print(f"      G0raw: {rewrites[0][:260].replace(chr(10), ' | ')}", flush=True)

            # --- policy + ref logprobs over rollout tokens ---
            resp_mask = torch.zeros_like(seq_ids, dtype=torch.bool)
            resp_mask[:, plen:] = True
            am = (seq_ids != tok.pad_token_id).long()
            loss_acc = 0.0
            for i in range(args.G):
                logits = policy(input_ids=seq_ids[i:i+1], attention_mask=am[i:i+1]).logits[0]
                logp = torch.log_softmax(logits.float(), dim=-1)       # [T, V]
                with torch.no_grad():
                    logits_ref = ref(input_ids=seq_ids[i:i+1],
                                     attention_mask=am[i:i+1]).logits[0]
                    logp_ref = torch.log_softmax(logits_ref.float(), dim=-1)
                t = seq_ids[i]
                # logits[k] predicts token t[k+1]; response tokens start at index plen
                tok_logp = logp[:-1].gather(1, t[1:].unsqueeze(1)).squeeze(1)
                tok_logp_ref = logp_ref[:-1].gather(1, t[1:].unsqueeze(1)).squeeze(1)
                m = resp_mask[i][1:]
                if m.sum().item() == 0:
                    continue
                lp, lpr = tok_logp[m], tok_logp_ref[m]
                kl = torch.exp(lpr - lp) - (lpr - lp) - 1.0
                n = m.sum().float()
                loss_i = (-adv[i] * lp + args.beta * kl).sum() / n
                loss_acc += loss_i
            loss = loss_acc / args.G
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in lm_p.parameters() if p.requires_grad], max_norm=0.5)
            opt.step(); opt.zero_grad()
            ep_loss += float(loss.item()); n_upd += 1
            ep_stats["rew"] += rew_t.mean().item()
            for k in ("fmt", "diff", "term", "copy", "faith"):
                ep_stats[k] += sum(p[k] for p in reward_parts) / args.G
            if n_upd % 8 == 0:
                print(f"[ep {ep}] up {n_upd}/{len(prom_cache)} loss {ep_loss/n_upd:.4f} "
                      f"mean_rew {rew_t.mean().item():.3f}", flush=True)
        nn = max(n_upd, 1)
        line = (f"[epoch {ep}] loss {ep_loss/nn:.4f} | rew {ep_stats['rew']/nn:.3f} "
                f"| fmt {ep_stats['fmt']/nn:.2f} diff {ep_stats['diff']/nn:.2f} "
                f"term {ep_stats['term']/nn:.2f} copy {ep_stats['copy']/nn:.2f} "
                f"faith {ep_stats['faith']/nn:.2f}")
        print(line, flush=True)

    lm_p.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "grpo_meta.json"), "w") as f:
        json.dump({"epochs": args.epochs, "lr": args.lr, "G": args.G, "beta": args.beta,
                   "n_train": len(prompts), "weights": W, "dropped": sorted(dropped),
                   "seed": args.seed, "start_adapter": start_ckpt,
                   "target_fre": LEVEL_TARGET_FRE}, f, indent=1)
    print(f"saved GRPO LoRA -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
