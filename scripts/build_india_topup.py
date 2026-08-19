# -*- coding: utf-8 -*-
"""Top up the Indian case-law corpus with additional judgments, preserving the
existing cached passages (only fetches NEW doc IDs + retries the one that failed).
Rebuilds data/split_ext_in.json with the union of cached + new passages.
"""
import json
import os
import re
import time
import urllib.request

from bs4 import BeautifulSoup

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA = os.path.normpath(DATA)

NEW_DOCS = [
    ("182928719", "Shayara Bano v. Union of India (2017)"),
    ("105939091", "Rajasthan High Court re extra-marital relationships (2022)"),
    ("168671544", "Navtej Singh Johar v. Union of India (2018)"),  # retry
]

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 "
                   "Safari/537.36"),
}
MIN_WORDS = 120
MAX_WORDS = 450
PER_DOC = 15


def fetch_doc(doc_id, tries=3, delay=12):
    url = "https://indiankanoon.org/doc/%s/" % doc_id
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            body = urllib.request.urlopen(req, timeout=40).read()
            if len(body) > 1000:
                return body
        except Exception as e:
            print("    try", i + 1, doc_id, repr(e)[:60], flush=True)
        time.sleep(delay)
    return None


HEADER_RE = re.compile(
    r"(Equivalent citations|IN THE SUPREME COURT OF INDIA|IN THE HIGH COURT OF|"
    r"The Hon.?ble|CORAM|Bench:|^AIR \d{4}|^Reportable|^JUDGMENT|^ORDER|"
    r"^Appeal .* against|^Petitioner|^Respondent|^Counsel|^Advocate|"
    r"^For .*:|^MANU/)", re.I)


def extract_body(html):
    soup = BeautifulSoup(html, "html.parser")
    div = soup.select_one("div.judgments")
    if div is None:
        return []
    paras = [p.get_text(" ", strip=True) for p in div.find_all("p")]
    if not paras:
        paras = [ln.strip() for ln in div.get_text("\n").split("\n") if ln.strip()]
    out = []
    for p in paras:
        p = " ".join(p.split())
        if len(p) < 60:
            continue
        if HEADER_RE.search(p):
            continue
        out.append(p)
    return out


def split_passages(paras):
    passages, cur, cur_w = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur_w + w > MAX_WORDS and cur_w >= MIN_WORDS:
            passages.append(" ".join(cur))
            cur, cur_w = [], 0
        cur.append(p)
        cur_w += w
    if cur_w >= MIN_WORDS:
        passages.append(" ".join(cur))
    return passages


def main():
    cache_path = os.path.join(DATA, "corpus_ext_in.json")
    existing = json.load(open(cache_path, encoding="utf-8"))["passages"]
    seen = set(p["id"] for p in existing)
    print("existing passages:", len(existing), flush=True)

    new_rows = []
    fails = []
    for doc_id, title in NEW_DOCS:
        html = fetch_doc(doc_id)
        if html is None:
            fails.append((doc_id, "fetch fail"))
            continue
        paras = extract_body(html)
        print("doc", doc_id, title[:40], "paras", len(paras), flush=True)
        passages = split_passages(paras)
        added = 0
        for i, psg in enumerate(passages[:PER_DOC]):
            pid = "in_%s_%02d" % (doc_id, i)
            if pid in seen:
                continue
            seen.add(pid)
            new_rows.append({
                "id": pid, "source": "indiankanoon", "title": title,
                "text": psg,
            })
            added += 1
        print("   added", added, flush=True)
        time.sleep(6)

    all_rows = existing + new_rows
    print("total passages:", len(all_rows), "fails:", fails, flush=True)

    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    ext = [{"id": r["id"], "title": r["title"], "text": r["text"]} for r in all_rows]
    out = os.path.join(DATA, "split_ext_in.json")
    json.dump({"train": split["train"], "val": split["val"], "test": ext},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out, "test len", len(ext), flush=True)
    json.dump({"source": "indiankanoon-2026", "n_passages": len(all_rows),
               "passages": all_rows},
              open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", cache_path, flush=True)


if __name__ == "__main__":
    main()
