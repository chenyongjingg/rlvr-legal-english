# -*- coding: utf-8 -*-
"""Build a second external OOD corpus from Indian Supreme Court / High Court
case law (indiankanoon.org). Complements the US Code corpus (build_external_
corpus.py) with a DIFFERENT legal system (India, common-law constitutional
adjudication) AND a different text genre (judicial opinions, not codified
statutes). Produces data/split_ext_in.json with train/val = ORIGINAL (so the
RAG KB stays the 861-passage eur/oyez index) and test = Indian case-law rows
(id prefix 'in_').

Genre note: the KB contains EU regulations (EUR-Lex), an encyclopedia (Wex)
and SCOTUS oral-argument transcripts (Oyez) -- no Indian materials. Every test
row is therefore fully out-of-domain twice over (jurisdiction x genre).
"""
import json
import os
import re
import time
import urllib.request

from bs4 import BeautifulSoup

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA = os.path.normpath(DATA)

# (doc_id, case_title) -- landmark, text-rich Indian judgments.
# doc IDs were located via web search of indiankanoon.org.
DOCS = [
    ("1031794", "Vishaka & Ors v. State of Rajasthan & Ors (1997)"),
    ("168671544", "Navtej Singh Johar v. Union of India (2018)"),
    ("1845552", "M. Nagaraj & Ors v. Union of India (2006)"),
    ("129202312", "Justice K.S. Puttaswamy (Retd.) v. Union of India (2017)"),
    ("924317", "Kesavananda Bharati v. State of Kerala (1973, Art.31C)"),
    ("1147386", "M. Radhakrishna Murthy v. Govt. of A.P. (2001)"),
]

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 "
                   "Safari/537.36"),
}

MIN_WORDS = 120
MAX_WORDS = 450
TARGET_ROWS = 66  # >= n_test 50 with margin; extra rows are ignored by eval


def fetch_doc(doc_id, tries=3, delay=10):
    url = "https://indiankanoon.org/doc/%s/" % doc_id
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            body = urllib.request.urlopen(req, timeout=30).read()
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
        if len(p) < 60:            # drops title/citation one-liners
            continue
        if HEADER_RE.search(p):    # drops header/boilerplate blocks
            continue
        out.append(p)
    return out


def split_passages(paras):
    """Merge consecutive paragraphs into passages of MIN..MAX words."""
    passages = []
    cur = []
    cur_w = 0
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
    cache = os.path.join(DATA, "corpus_ext_in.json")
    done = {}
    if os.path.exists(cache):
        for p in json.load(open(cache, encoding="utf-8")).get("passages", []):
            done[p["id"]] = p

    all_rows = []
    fails = []
    for doc_id, title in DOCS:
        key = "in_" + doc_id
        if key in done:
            all_rows.append(done[key])
            print("cached", key, flush=True)
            continue
        html = fetch_doc(doc_id)
        if html is None:
            fails.append((doc_id, "fetch fail"))
            continue
        paras = extract_body(html)
        print("doc", doc_id, title[:45], "paras", len(paras), flush=True)
        passages = split_passages(paras)
        for i, psg in enumerate(passages[:12]):  # cap per doc to keep balance
            all_rows.append({
                "id": "in_%s_%02d" % (doc_id, i),
                "source": "indiankanoon",
                "title": title,
                "text": psg,
            })
        time.sleep(6)  # be gentle with the rate-limited host

    rows = all_rows[:TARGET_ROWS]
    print("total rows", len(rows), "fails", fails[:5], flush=True)
    # sanity: word-length distribution
    ws = sorted(len(r["text"].split()) for r in rows)
    print("word quantiles", ws[0], ws[len(ws)//4], ws[len(ws)//2],
          ws[3*len(ws)//4], ws[-1], flush=True)

    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    ext = [{"id": r["id"], "title": r["title"], "text": r["text"]} for r in rows]
    out = os.path.join(DATA, "split_ext_in.json")
    json.dump({"train": split["train"], "val": split["val"], "test": ext},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out, "test len", len(ext), flush=True)
    json.dump({"source": "indiankanoon-2026", "n_passages": len(rows),
               "passages": rows},
              open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", cache, flush=True)


if __name__ == "__main__":
    main()
