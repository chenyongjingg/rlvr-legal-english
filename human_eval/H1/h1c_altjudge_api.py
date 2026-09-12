# -*- coding: utf-8 -*-
"""h1c_altjudge_api.py -- run the paper's judge prompt against any
OpenAI-compatible endpoint, for the three remaining compute-only items.

The whole point of C is COMPARABILITY: the alternate judge must see the same
prompt, the same truncation and the same decoding as the paper's judge, so that
a difference in the scores is attributable to the model and not to the harness.
JUDGE_SYS / JUDGE_USR below are copied VERBATIM from
github_repo/scripts/09_judge_human_items.py, including the 3000-char truncation
of source and rewrite and the greedy-first / sampled-retry decode.

Modes
  --probe        list the models the endpoint serves (needs no judging)
  --mode c       C: one greedy pass by a DIFFERENT-VENDOR judge over the 240
                 blind cells.  This is the move that answers "your judge is a
                 different checkpoint of the same family as your generators".
  --mode d       D: K repeats with sampling -- self-consistency of the judge on
                 the 240 blind cells (the paper reports K = 3 only on
                 development probes, never on these cells).
  --mode e       E: order-swapped prompt (rewrite before original).  The paper
                 cites MT-Bench for position bias and describes no such control.

Secrets
  Read from the environment ONLY.  This script never writes the key anywhere and
  never echoes it; --probe and the run summary print a masked prefix, so a run
  log is safe to paste into the manuscript's evidence directory.

  APIN_JUDGE_BASE    e.g. https://api.example.com/v1
  APIN_JUDGE_KEY     the credential
  APIN_JUDGE_MODEL   model id as the endpoint names it

Usage
  APIN_JUDGE_BASE=... APIN_JUDGE_KEY=... APIN_JUDGE_MODEL=... \
    python h1c_altjudge_api.py --probe
  ... python h1c_altjudge_api.py --mode c --tag llama31-8b
  ... python h1c_altjudge_api.py --mode d --tag qwen3-4b-2507 --k 3
  ... python h1c_altjudge_api.py --mode e --tag qwen3-4b-2507
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.normpath(os.path.join(BASE, "..", "samples", "items.jsonl"))
OUT = os.path.join(BASE, "h1c_out")
SYSTEMS = ["Source", "B2", "M2", "FlanT5", "BART", "G6"]
DIMS = ("difficulty", "faithfulness", "terminology", "fluency")

# ---- verbatim from github_repo/scripts/09_judge_human_items.py -------------
JUDGE_SYS = ("You are an expert evaluator of simplified legal texts for English "
             "learners. You rate rewrites strictly and objectively, always returning "
             "a single JSON object and nothing else.")
JUDGE_HEAD = ("Rate this simplified legal rewrite for a {lv}-level English learner.\n"
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
              '"confidence": <float 0-1>}}\n\n')
JUDGE_TAIL = ("ORIGINAL LEGAL TEXT:\n{src}\n\n"
              "SIMPLIFIED REWRITE ({lv}):\n{rew}\n")
JUDGE_TAIL_SWAPPED = ("SIMPLIFIED REWRITE ({lv}):\n{rew}\n\n"
                      "ORIGINAL LEGAL TEXT:\n{src}\n")


def build_prompt(lv, src, rew, swap=False):
    tail = JUDGE_TAIL_SWAPPED if swap else JUDGE_TAIL
    return [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": JUDGE_HEAD.format(lv=lv) +
             tail.format(lv=lv, src=src[:3000], rew=rew[:3000])}]


def parse_scores(out):
    """Verbatim behaviour of the producer's parser."""
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


# ---- transport -------------------------------------------------------------
def post(url, key, payload, timeout=180, tries=4):
    body = json.dumps(payload).encode("utf-8")
    last = None
    for t in range(tries):
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            last = "HTTP %d: %s" % (e.code, detail)
            if e.code in (400, 401, 403, 404, 422):   # not retryable
                break
        except Exception as e:                        # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
        time.sleep(1.5 * (t + 1))
    raise RuntimeError(last or "request failed")


def chat(base, key, model, msgs, temperature=None, max_tokens=512):
    payload = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if temperature is None:
        payload["temperature"] = 0
    else:
        payload["temperature"] = temperature
        payload["top_p"] = 0.9
    d = post(base.rstrip("/") + "/chat/completions", key, payload)
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    txt = msg.get("content")
    # Aggregators relay Anthropic-style messages with `content` as a LIST of
    # blocks rather than a string.  Left alone this reaches parse_scores() as a
    # list, finds no match, and reports zero parsed cells -- which is
    # indistinguishable from "the judge refused on every item".  Normalise here.
    if isinstance(txt, list):
        txt = "".join(b.get("text", "") for b in txt
                      if isinstance(b, dict) and b.get("type") in (None, "text"))
    if txt is None:
        txt = ch.get("text") or ""
    if not isinstance(txt, str):
        txt = str(txt)
    return strip_thinking(txt)


def one_cell(base, key, model, job, swap, temperature):
    msgs = build_prompt(job["lv"], job["src"], job["rew"], swap=swap)
    for at in range(2):
        t = None if at == 0 else (temperature if temperature is not None else 0.6)
        raw = chat(base, key, model, msgs, temperature=t)
        sc = parse_scores(raw)
        if sc is not None:
            return sc, raw, at
    return None, raw, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--mode", choices=["c", "d", "e"])
    ap.add_argument("--tag", default="", help="label for the output file/model")
    ap.add_argument("--k", type=int, default=3, help="repeats for mode d")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N cells")
    args = ap.parse_args()

    base = os.environ.get("APIN_JUDGE_BASE", "").strip()
    key = os.environ.get("APIN_JUDGE_KEY", "").strip()
    model = os.environ.get("APIN_JUDGE_MODEL", "").strip()
    if not base or not key:
        sys.exit("set APIN_JUDGE_BASE and APIN_JUDGE_KEY (and APIN_JUDGE_MODEL)")
    masked = key[:6] + "..." + key[-4:] if len(key) > 12 else "***"
    print("endpoint: %s   key: %s   model: %s"
          % (base, masked, model or "(unset)"))

    if args.probe:
        req = urllib.request.Request(base.rstrip("/") + "/models",
                                     headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode("utf-8"))
        ids = sorted(x.get("id", "?") for x in (d.get("data") or []))
        print("\n%d models:" % len(ids))
        for i in ids:
            print("   " + i)
        return

    if not model:
        sys.exit("set APIN_JUDGE_MODEL (use --probe to list what is served)")

    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    jobs = []
    for it in items:
        for s in SYSTEMS:
            rec = it["systems"].get(s) or {}
            rew = (rec.get("rew") or "").strip()
            if rew:
                jobs.append({"id": it["id"], "lv": it["lv"], "src": it["src"],
                             "system": s, "rew": rew})
    if args.limit:
        jobs = jobs[:args.limit]
    reps = args.k if args.mode == "d" else 1
    swap = (args.mode == "e")
    print("mode %s: %d cells x %d repeat(s), swap=%s, workers=%d"
          % (args.mode, len(jobs), reps, swap, args.workers))

    def work(idx_job):
        i, j = idx_job
        recs = []
        for rep in range(reps):
            try:
                # E isolates ORDER, so it must decode exactly as C does
                # (greedy).  Sampling here would confound position bias with
                # sampling noise and make the two modes non-comparable.  Only
                # D -- whose entire point is variability -- samples.
                sc, raw, at = one_cell(base, key, model, j, swap,
                                       temperature=None if args.mode in ("c", "e")
                                       else 0.6)
            except Exception as e:                     # noqa: BLE001
                sc, raw, at = None, "ERROR %s" % e, -1
            recs.append({"rep": rep, "judge": sc, "raw": raw[:2000], "attempt": at})
        return {"id": j["id"], "lv": j["lv"], "system": j["system"],
                "src": j["src"], "rew": j["rew"], "results": recs}

    t0 = time.time()
    done = 0
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, list(enumerate(jobs))):
            rows.append(r)
            done += 1
            if done % 40 == 0:
                print("  ...%d/%d  (%.0fs)" % (done, len(jobs), time.time() - t0),
                      flush=True)

    rows.sort(key=lambda r: (r["id"], SYSTEMS.index(r["system"])))
    os.makedirs(OUT, exist_ok=True)
    tag = args.tag or model.replace("/", "_")
    fn = "%s_%s.jsonl" % ({"c": "altjudge", "d": "selfcons", "e": "swaporder"}[args.mode],
                          tag)
    p = os.path.join(OUT, fn)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"model": model, "mode": args.mode, "swap": swap,
                                **r}, ensure_ascii=False) + "\n")

    ok = sum(1 for r in rows if all(x["judge"] for x in r["results"]))
    print("\nwrote %s" % p)
    print("parsed %d/%d cells  (%.0fs)"
          % (ok, len(rows), time.time() - t0))

    # quick per-system means, same shape as the producer's on-server table
    print("\n==== %s per-system judge means over the 40 items ====" % args.mode)
    print("%-9s" % "system" + "".join("%13s" % d for d in DIMS)
          + "%9s%9s" % ("overall", "n"))
    for s in SYSTEMS:
        sub = [r for r in rows if r["system"] == s
               and all(x["judge"] for x in r["results"])]
        if not sub:
            print("%-9s  (none)" % s)
            continue
        def m(d):
            return sum(sum(x["judge"][d] for x in r["results"]) / len(r["results"])
                       for r in sub) / len(sub)
        ov = sum(sum(sum(x["judge"][d] for d in DIMS) / len(DIMS)
                     for x in r["results"]) / len(r["results"])
                 for r in sub) / len(sub)
        print("%-9s" % s + "".join("%13.2f" % m(d) for d in DIMS)
              + "%9.2f%9d" % (ov, len(sub)))

    if args.mode == "d":
        print("\n==== self-consistency over K=%d repeats ====" % reps)
        for d in DIMS:
            exact = same = 0
            for r in rows:
                jj = [x["judge"][d] for x in r["results"] if x["judge"]]
                if len(jj) < 2:
                    continue
                exact += 1 if len(set(jj)) == 1 else 0
                same += 1
            if same:
                print("  %-14s exact agreement %.1f%%  (%d cells)"
                      % (d, 100.0 * exact / same, same))


if __name__ == "__main__":
    main()
