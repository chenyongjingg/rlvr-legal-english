# -*- coding: utf-8 -*-
"""
make_human_eval_samples.py — build blind human-evaluation materials for the
legal-English rewrite paper (Applied Intelligence resubmission).

Joins all comparison systems on (id, level) so every rated item contains the
SAME source rewritten by ALL systems, then anonymizes the systems A–F with a
per-item AND per-rater shuffle (removes position/system-order bias).

Systems (PLAN.md §1):
  Source   original text (no simplification; difficulty anchor)   [from src field]
  B2       dictionary/rule baseline          results/B2_honest.jsonl    (rew_clean)
  M2       supervised SFT LLM                results/M2_honest.jsonl    (rew_clean)
  FlanT5   seq2seq, same SFT data            results/B3_flant5.jsonl    (rew)
  BART     seq2seq, same SFT data            results/B4_bart.jsonl      (rew)
  G6/G6v4  full framework + RLVR             results/{g6}.jsonl         (rew_clean)

Usage (run on the 5090 server, next to results/):
  python make_human_eval_samples.py --n 20 --levels beginner,intermediate \
      --g6 G6_honest.jsonl --raters 3 --seed 7 --outdir human_eval_out

Outputs (into outdir):
  items.jsonl          : 40 rated items (source + every system's rewrite + auto metrics)
  scoresheet_R{n}.csv  : one anonymized CSV per rater (A–F shuffled per item)
  unblind_key.json     : rater -> item_id -> label -> system  (de-anonymizing)
"""
import os, sys, json, csv, random, argparse

R = "/path/to/legal_english/results"
LABELS = ["A", "B", "C", "D", "E", "F"]
RATING_COLS = ["difficulty_1_5", "meaning_1_5", "terminology_1_5",
               "fluency_1_5", "overall_1_5", "valid_rewrite", "comment"]


def get_systems(g6_file):
    # (name, file, rewrite_field, is_source)
    return [
        ("Source",  None,             "src",       True),
        ("B2",      "B2_honest.jsonl", "rew_clean", False),
        ("M2",      "M2_honest.jsonl", "rew_clean", False),
        ("FlanT5",  "B3_flant5.jsonl", "rew",       False),
        ("BART",    "B4_bart.jsonl",   "rew",       False),
        ("G6",      g6_file,           "rew_clean", False),
    ]


def load_system(file, field):
    d = {}
    for line in open(os.path.join(R, file), encoding="utf-8"):
        r = json.loads(line)
        if r.get("pure_commentary"):
            continue  # never show meta-commentary to raters
        # *_honest.jsonl uses `rew_clean`; raw eval output uses `rew` (already
        # cleaned). Fall back to `rew` when the requested field is absent.
        f = field if field in r else ("rew" if "rew" in r else field)
        rew = (r.get(f) or "").strip()
        if not rew:
            continue  # empty rewrite == pure-commentary failure
        d[(r["id"], r["lv"])] = {
            "src": r.get("src", ""),
            "rew": rew,
            "fre": r.get("fre"),
            "tot": r.get("tot"),
            "nli": r.get("nli"),
            "parts": r.get("parts"),
        }
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="source texts per level")
    ap.add_argument("--levels", default="beginner,intermediate")
    ap.add_argument("--g6", default="G6_honest.jsonl")
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="human_eval_out")
    args = ap.parse_args()

    levels = [l.strip() for l in args.levels.split(",")]
    systems = get_systems(args.g6)
    names = [s[0] for s in systems]

    data = {}
    for name, f, field, is_src in systems:
        data[name] = None if is_src else load_system(f, field)

    # join: (id, lv) present in all non-source systems
    non_src = [s[0] for s in systems if not s[3]]
    keys = set(data[non_src[0]].keys())
    for nm in non_src[1:]:
        keys &= set(data[nm].keys())
    keys = {k for k in keys if k[1] in levels}

    # source text: fetch from G6's records (consistent across systems)
    src_lookup = {k: data["G6"][k]["src"] for k in keys}

    # sample N ids that are COMPLETE (valid at every requested level in every
    # system) so each item has all 6 systems present — no partial items.
    ids = sorted({i for i, _ in keys})
    complete_ids = [i for i in ids if all((i, lv) in keys for lv in levels)]
    rng = random.Random(args.seed)
    sampled_ids = rng.sample(complete_ids, min(args.n, len(complete_ids)))
    items = [(i, lv) for i in sampled_ids for lv in levels]

    os.makedirs(args.outdir, exist_ok=True)

    # ---- master items.jsonl (unnanonymized, with auto metrics) ----
    master = []
    for (i, lv) in items:
        entry = {"id": i, "lv": lv, "src": src_lookup[(i, lv)],
                 "systems": {}}
        for nm in names:
            if nm == "Source":
                entry["systems"][nm] = {"rew": src_lookup[(i, lv)]}
            else:
                entry["systems"][nm] = data[nm][(i, lv)]
        master.append(entry)
    with open(os.path.join(args.outdir, "items.jsonl"), "w", encoding="utf-8") as f:
        for e in master:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- per-rater anonymized scoresheets + unblind key ----
    key = {}
    for r in range(1, args.raters + 1):
        rng_r = random.Random(args.seed * 100 + r)
        out = os.path.join(args.outdir, "scoresheet_R%d.csv" % r)
        key["R%d" % r] = {}
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "level", "source_text", "label", "rewrite"]
                       + RATING_COLS)
            for (i, lv) in items:
                item_id = "%s_%s" % (i, lv)
                # shuffle system names -> labels, independent per item per rater
                order = names[:]
                rng_r.shuffle(order)
                mapping = {LABELS[k]: order[k] for k in range(len(order))}
                key["R%d" % r][item_id] = mapping
                src = src_lookup[(i, lv)]
                for lab in LABELS:
                    sysname = mapping[lab]
                    if sysname == "Source":
                        rew = src
                    else:
                        rew = data[sysname][(i, lv)]["rew"]
                    w.writerow([item_id, lv, src, lab, rew] + [""] * len(RATING_COLS))

    with open(os.path.join(args.outdir, "unblind_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)

    print("items: %d  (ids=%d x levels=%s)" % (len(items), len(sampled_ids), levels))
    print("raters: %d" % args.raters)
    print("systems: %s" % ", ".join(names))
    print("wrote -> %s/" % args.outdir)


if __name__ == "__main__":
    main()
