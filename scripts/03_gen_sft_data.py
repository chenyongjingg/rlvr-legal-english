# -*- coding: utf-8 -*-
"""
SFT synthetic parallel data generation (task 14 part 1).
Teacher: Qwen3-4B-Instruct-2507 (2025-gen, text-only, outputs DIRECT rewrites —
Qwen3.5-9B was found to always emit a verbose reasoning preamble that crowds out
the rewrite, so it is kept as the GRPO verifier instead). Teacher rewrites source
legal snippets into beginner/intermediate versions -> parallel pairs for QLoRA SFT
warm-start of Qwen3.5-4B.

Prompt follows the RAG-aware design from the proposal (retrieved terms + patterns).
Configurable N source snippets x 2 difficulty levels.
"""
import os, sys, json, argparse, random
import torch

DATA = "/root/autodl-tmp/legal_english/data"
MODELS = "/root/autodl-tmp/legal_english/models"

SYSTEM = (
    "You are a legal English educator. You rewrite legal texts into clear, level-appropriate "
    "versions for language learners while preserving the legal meaning and key terminology. "
    "You always output ONLY the requested rewrite."
)

DIFF_PROMPTS = {
    "beginner": (
        "Rewrite the following legal text for BEGINNER learners (CEFR A2-B1). Requirements:\n"
        "- Replace complex legal vocabulary with common words; explain unavoidable legal terms in parentheses.\n"
        "- Keep sentences short (average 10-14 words).\n"
        "- Flesch Reading Ease 60-70.\n"
        "- Do NOT omit the core legal meaning or any key facts.\n"
        "Output only the rewritten text.\n\nSource text:\n"
    ),
    "intermediate": (
        "Rewrite the following legal text for INTERMEDIATE learners (CEFR B1-B2). Requirements:\n"
        "- Retain key legal terms but add a brief gloss in parentheses for the hardest terms.\n"
        "- Keep average sentence length 15-20 words.\n"
        "- Flesch Reading Ease 50-60.\n"
        "- Preserve legal meaning and structure faithfully.\n"
        "Output only the rewritten text.\n\nSource text:\n"
    ),
}


def strip_preamble(out):
    """Safety net in case the teacher ever emits a thinking/planning preamble."""
    if not out:
        return out
    low = out.lower()
    for marker in ("thinking process", "思考过程", "thought process", "let me think",
                   "analyze the request", "分析需求"):
        i = low.find(marker)
        if i >= 0:
            seg = out[i:]
            nl = seg.find("\n\n")
            rest = seg[nl + 2:].strip() if nl >= 0 else ""
            return rest or out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--n-train", type=int, default=100)   # source snippets for training pairs
    ap.add_argument("--levels", default="beginner,intermediate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher", default="Qwen3-4B-Instruct-2507")
    args = ap.parse_args()
    random.seed(args.seed)

    split = json.load(open(os.path.join(args.data, "split.json"), encoding="utf-8"))
    train = split["train"]
    rng = random.Random(args.seed)
    sample = rng.sample(train, min(args.n_train, len(train)))
    print(f"sampled {len(sample)} train snippets for SFT data", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    teacher_id = os.path.join(args.models, args.teacher)
    print(f"loading teacher {args.teacher} (bnb-4bit, text-only)...", flush=True)
    tok = AutoTokenizer.from_pretrained(teacher_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        teacher_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    out_records = []
    with torch.no_grad():
        for si, s in enumerate(sample):
            for lv in levels:
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": DIFF_PROMPTS[lv] + s["text"]}]
                inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_tensors="pt").to(model.device)
                gen = model.generate(**inp, max_new_tokens=420, do_sample=True, temperature=0.8,
                                     top_p=0.95, pad_token_id=tok.pad_token_id,
                                     repetition_penalty=1.05)
                new_ids = gen[0][inp["input_ids"].shape[1]:]
                out = strip_preamble(tok.decode(new_ids, skip_special_tokens=True).strip())
                out_records.append({
                    "source_id": s["id"], "difficulty": lv, "source_text": s["text"],
                    "target_text": out, "title": s["title"],
                })
                print(f"[{si+1}/{len(sample)}] {lv}: {len(s['text'].split())}->{len(out.split())} words",
                      flush=True)
            torch.cuda.empty_cache()

    json.dump(out_records, open(os.path.join(args.data, "sft_pairs.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"saved {len(out_records)} sft pairs -> sft_pairs.json", flush=True)


if __name__ == "__main__":
    main()
