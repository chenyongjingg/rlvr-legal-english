# -*- coding: utf-8 -*-
"""Build an external OOD corpus from Cornell LII US Code (federal statutes).

Genre genuinely unseen in the legal KB (EUR-Lex EU regs / Wex encyclopedia /
Oyez SCOTUS opinions): US codified federal statutes. Produces data/split_ext.json
with train/val = ORIGINAL (unchanged, so KB identical) and test = external rows
(id prefix 'uscode_'). corpus_v2.json is left untouched (terms unchanged).
"""
import json
import os
import re
import sys
import time
import urllib.request

from bs4 import BeautifulSoup

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA = os.path.normpath(DATA)

# (title, section) — substantive, well-known federal statute sections
SECTIONS = [
    ("18", "1001"), ("18", "1030"), ("18", "1341"), ("18", "1343"),
    ("18", "1349"), ("18", "1621"), ("18", "201"), ("18", "241"),
    ("18", "242"), ("18", "371"), ("18", "641"), ("18", "666"),
    ("18", "1512"), ("18", "1951"), ("18", "2314"), ("18", "924"),
    ("15", "1"), ("15", "2"), ("15", "13"), ("15", "14"),
    ("15", "18"), ("15", "45"), ("15", "77"), ("15", "78"),
    ("15", "78j"), ("15", "78m"), ("15", "78o"), ("15", "205"),
    ("26", "61"), ("26", "62"), ("26", "63"), ("26", "151"),
    ("26", "162"), ("26", "170"), ("26", "401"), ("26", "408"),
    ("26", "501"), ("26", "601"), ("29", "151"), ("29", "157"),
    ("29", "158"), ("29", "201"), ("29", "206"), ("29", "207"),
    ("29", "621"), ("29", "623"), ("42", "1981"), ("42", "1983"),
    ("42", "2000a"), ("42", "2000d"), ("42", "2000e"), ("42", "300a"),
    ("42", "405"), ("42", "423"), ("42", "5101"), ("31", "3729"),
    ("31", "3730"), ("31", "3727"), ("28", "1331"), ("28", "1332"),
    ("28", "1391"), ("28", "1291"), ("11", "101"), ("11", "362"),
    ("11", "727"), ("47", "151"), ("47", "230"), ("20", "1681"),
    ("7", "1"), ("49", "40101"), ("49", "41712"), ("18", "1111"),
]

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_provision(title, section):
    url = "https://www.law.cornell.edu/uscode/text/%s/%s" % (title, section)
    req = urllib.request.Request(url, headers=UA)
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    txt = " ".join(soup.get_text(separator=" ").split())
    # provision body sits after 'prev | next' and before the notes block
    start = txt.find("prev | next")
    if start < 0:
        return ""
    body = txt[start + len("prev | next"):]
    for cut in ("U.S. Code Notes", "Authorities (CFR)", "prev | next"):
        i = body.find(cut)
        if i > 0:
            body = body[:i]
    return body.strip()


def clean(title, section, body):
    body = re.sub(r"\s+", " ", body).strip()
    # drop trailing navigation fragments and truncated tails
    body = re.sub(r"\b(Pub\.? L\.?|Stat\.?)\b.*$", "", body)
    if len(body) < 80:
        return ""
    return body


def main():
    # resume from a previous partial run (cache keyed by section id)
    cache_path = os.path.join(DATA, "corpus_ext.json")
    done = {}
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        for p in cached.get("passages", []):
            done[p["title"]] = p  # title like "18 U.S.C. § 1001"

    passages = []
    fails = []
    for i, (t, s) in enumerate(SECTIONS):
        key = "%s U.S.C. § %s" % (t, s)
        if key in done:
            passages.append(done[key])
            continue
        try:
            body = fetch_provision(t, s)
            body = clean(t, s, body)
            if not body:
                fails.append((t, s, "empty"))
                continue
            passages.append({
                "id": "uscode_%03d" % i,
                "source": "uscode",
                "title": "%s U.S.C. § %s" % (t, s),
                "text": body,
            })
        except Exception as e:
            fails.append((t, s, repr(e)[:80]))
        time.sleep(0.3)

    print("fetched %d / %d passages" % (len(passages), len(SECTIONS)))
    print("fails:", fails[:10])

    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    ext = [{"id": p["id"], "title": p["title"], "text": p["text"]} for p in passages]
    split_ext = {"train": split["train"], "val": split["val"], "test": ext}
    out = os.path.join(DATA, "split_ext.json")
    json.dump(split_ext, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out, "test len:", len(ext))
    # also keep the passages for record
    meta = os.path.join(DATA, "corpus_ext.json")
    json.dump({"source": "uscode-cornell-2026", "n_passages": len(passages),
               "passages": passages}, open(meta, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote", meta)


if __name__ == "__main__":
    main()
