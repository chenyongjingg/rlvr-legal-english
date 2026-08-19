# -*- coding: utf-8 -*-
"""
QLoRA SFT warm-start for Qwen3.5-4B on synthetic parallel pairs (task 14 part 2).
Qwen3.5 is a multimodal composite (Qwen3_5ForConditionalGeneration); we train it
TEXT-ONLY by wrapping only the language_model submodule with LoRA, so vision
params stay frozen. Manual training loop (small dataset; fully reproducible,
no TRL SFTTrainer dependency for the composite model).
Hyperparams per proposal: rank=16, alpha=32, lr=2e-4, epochs=3.
Input: data/sft_pairs.json. Output: models/sft_qwen35_lora/ (adapter only).
"""
import os, sys, json, argparse, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = "/root/autodl-tmp/legal_english/data"
MODELS = "/root/autodl-tmp/legal_english/models"

SYSTEM = ("You are a legal English educator. Rewrite legal texts into clear, "
          "level-appropriate versions for language learners.")


def enable_peft_on_text_model(lm):
    """Qwen3_5TextModel lacks prepare_inputs_for_generation, which get_peft_model
    (PeftModelForCausalLM path) grabs at wrap time. We never generate through the
    peft wrapper (generation goes via the composite), so a minimal stub suffices."""
    if not hasattr(lm, "prepare_inputs_for_generation"):
        import types

        def _prep(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]
            out = {"input_ids": input_ids, "attention_mask": attention_mask,
                   "past_key_values": past_key_values}
            out.update(kwargs)
            return out
        lm.prepare_inputs_for_generation = types.MethodType(_prep, lm)
    return lm

def build_prompt(tok, rec):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Rewrite the following legal text into a "
                                     f"{rec['difficulty']}-level version for English "
                                     f"learners. Preserve legal meaning.\n\n{rec['source_text']}"},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def make_batches(tok, records, max_len, device, batch_size=4):
    """Tokenize prompt+target; labels mask the prompt (=-100)."""
    prompts, targets = [], []
    for r in records:
        p = build_prompt(tok, r)
        t = r["target_text"].strip() + tok.eos_token
        prompts.append(p); targets.append(t)
    pids = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    tids = tok(targets, add_special_tokens=False, padding=True, return_tensors="pt")["input_ids"]
    batch, cur_p, cur_t = [], [], []
    for i in range(len(records)):
        plen = (pids["attention_mask"][i] == 1).sum().item()
        tl = (tids[i] != tok.pad_token_id).sum().item()
        if plen + tl > max_len:  # skip overly long examples (rare)
            continue
        cur_p.append(i); cur_t.append(i)
        if len(cur_p) >= batch_size:
            batch.append((cur_p, cur_t, pids, tids)); cur_p, cur_t = [], []
    if cur_p:
        batch.append((cur_p, cur_t, pids, tids))
    return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs", type=int, default=0)  # probe: limit pairs (0 = all)
    ap.add_argument("--out", default=os.path.join(MODELS, "sft_qwen35_lora"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    pairs = json.load(open(os.path.join(args.data, "sft_pairs.json"), encoding="utf-8"))
    if args.max_pairs and args.max_pairs < len(pairs):
        pairs = pairs[:args.max_pairs]
    print(f"{len(pairs)} sft pairs", flush=True)
    if not pairs:
        print("no pairs, exit"); return
    random.shuffle(pairs)

    from transformers import AutoModelForImageTextToText, Qwen3_5ForConditionalGeneration, \
        AutoTokenizer, BitsAndBytesConfig, get_scheduler
    from peft import LoraConfig, get_peft_model

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model_id = os.path.join(args.models, "Qwen3.5-4B")
    print("loading Qwen3.5-4B (bnb-4bit)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    except Exception:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model.train()
    # activation memory control for long sequences on a 32GB card
    model.gradient_checkpointing_enable()

    # LoRA only on the language (text) module; vision stays frozen
    lm = model.model.language_model
    lora = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    enable_peft_on_text_model(lm)
    plm = get_peft_model(lm, lora)
    model.model.language_model = plm
    n_tr = sum(p.numel() for p in plm.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable LoRA params: {n_tr:,} / total {n_all:,} "
          f"({100*n_tr/n_all:.3f}%)", flush=True)

    batch = make_batches(tok, pairs, args.max_len, model.device, batch_size=args.batch_size)
    n_steps = len(batch)
    print(f"batch count: {n_steps}", flush=True)

    opt = torch.optim.AdamW([p for p in plm.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    total_steps = n_steps * args.epochs // args.accum
    sched = get_scheduler("cosine", optimizer=opt,
                          num_warmup_steps=int(0.03 * total_steps),
                          num_training_steps=total_steps)

    step = 0
    for ep in range(args.epochs):
        random.shuffle(batch)
        tot_loss, n_loss = 0.0, 0
        opt.zero_grad()
        for bi, (idx, _, pids, tids) in enumerate(batch):
            inp = pids["input_ids"][idx].to(model.device)      # [b, P]
            am = pids["attention_mask"][idx].to(model.device)  # [b, P]
            tgt = tids[idx].to(model.device)                   # [b, T] padded
            plen = am.sum(dim=1)                               # [b] prompt length per row
            tmask = (tgt != tok.pad_token_id)                  # [b, T] real target mask
            tl = tmask.sum(dim=1)                              # [b] target length per row
            max_t = tgt.shape[1]
            # full sequence = prompt (real) + target (padded to batch max)
            inp_all = torch.cat([inp, torch.full((len(idx), max_t), tok.pad_token_id,
                                                 dtype=torch.long, device=model.device)], dim=1)
            am_all = torch.cat([am, tmask.to(torch.int64)], dim=1)
            # labels: prompt tokens and target pads -> -100; real target tokens -> id
            labels = torch.full((len(idx), inp.shape[1] + max_t), -100,
                                dtype=torch.long, device=model.device)
            for j in range(len(idx)):
                p = plen[j].item(); t = tl[j].item()
                labels[j, p:p + t] = tgt[j][:t]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp_all, attention_mask=am_all, labels=labels).loss
            loss = loss / args.accum
            loss.backward()
            tot_loss += float(loss.item()) * args.accum
            n_loss += 1
            step += 1
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in plm.parameters() if p.requires_grad],
                                               max_norm=0.3)
                opt.step(); sched.step(); opt.zero_grad()
                lr_now = sched.get_last_lr()[0]
                print(f"epoch {ep} step {bi+1}/{n_steps} loss {tot_loss/n_loss:.4f} "
                      f"lr {lr_now:.2e}", flush=True)
        print(f"[epoch {ep}] avg loss {tot_loss/max(n_loss,1):.4f}", flush=True)

    plm.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "sft_meta.json"), "w") as f:
        json.dump({"epochs": args.epochs, "lr": args.lr, "rank": 16, "alpha": 32,
                   "pairs": len(pairs), "seed": args.seed, "model": "Qwen3.5-4B"}, f, indent=1)
    print(f"saved SFT LoRA -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
