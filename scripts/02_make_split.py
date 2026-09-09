# -*- coding: utf-8 -*-
"""Deterministic train/val/test split of the legal corpus (task 13 -> 14 bridge).
Snippets are independent documents; hash-of-id partition (80/10/10) is stable
across runs and used by all downstream stages (SFT data gen, GRPO, evaluation).
Records: {"id", "title", "text"}.
Input: data/corpus_v2.json   Output: data/split.json
"""
import os, sys, json, hashlib

DATA = "/path/to/legal_english/data"

def partition_bucket(sid):
    h = hashlib.md5(("legal:" + sid).encode("utf-8")).hexdigest()
    return int(h, 16) % 100

def main():
    corpus = json.load(open(os.path.join(DATA, "corpus_v2.json"), encoding="utf-8"))
    snippets = corpus["snippets"]
    buckets = {k: [] for k in ("train", "val", "test")}
    for s in snippets:
        b = partition_bucket(s["id"])
        key = "train" if b < 80 else ("val" if b < 90 else "test")
        buckets[key].append({"id": s["id"], "title": s["title"], "text": s["text"]})
    for k, v in buckets.items():
        v.sort(key=lambda r: r["id"])
    out = {k: v for k, v in buckets.items()}
    json.dump(out, open(os.path.join(DATA, "split.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    total = sum(len(v) for v in buckets.values())
    print(f"split.json: train={len(out['train'])} val={len(out['val'])} "
          f"test={len(out['test'])} (total {total})", flush=True)
    # sanity: every snippet accounted for exactly once
    all_ids = [r["id"] for v in buckets.values() for r in v]
    assert len(all_ids) == len(set(all_ids)) == len(snippets), "split not a clean partition"
    print("partition OK", flush=True)

if __name__ == "__main__":
    main()
