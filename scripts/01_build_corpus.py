# -*- coding: utf-8 -*-
"""
Legal English corpus & knowledge base builder v2 (LegalGraphRAG data layer).
Sources: EUR-Lex XML (clean full text), Oyez (case references), Wex (legal dictionary).
Output: source text snippets (150-300 words), term dictionary JSON, metadata manifest.
Reusable on server: /path/to/legal_english/scripts/01_build_corpus.py
"""
import os, sys, json, re, time, argparse
import urllib.request
import xml.etree.ElementTree as ET

DATA = "/path/to/legal_english/data"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-crawler"}


def fetch(url, timeout=25, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            print(f"  [fetch {i+1}/{tries}] {url} -> {type(e).__name__}: {e}", flush=True)
            time.sleep(2 * (i + 1))
    return None


def clean_text(s):
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_snippets(text, min_w=150, max_w=300):
    """Split long legal text into ~150-300 word snippets on sentence boundaries."""
    if not text:
        return []
    text = clean_text(text)
    words = text.split()
    if len(words) <= max_w:
        return [text] if len(words) >= min_w else []
    sents = re.split(r"(?<=[.;])\s+", text)
    out, cur = [], []
    for s in sents:
        cur.append(s)
        w = sum(len(x.split()) for x in cur)
        if w >= min_w and w <= max_w + 20:
            out.append(" ".join(cur))
            cur = []
        elif w > max_w + 20 and len(cur) > 1:
            joined = " ".join(cur).split()
            if len(joined) > min_w:
                out.append(" ".join(joined[:max_w]))
            cur = [s]
    if sum(len(x.split()) for x in cur) >= min_w:
        out.append(" ".join(cur))
    return out


# ---------- EUR-Lex (HTML full text via BeautifulSoup) ----------
def fetch_eurlex_html(celex_ids):
    """Pull regulation full text from EUR-Lex HTML page; extract <div id="text" class="panel-body">.
    This container holds the clean legal body from the doc title to the last article."""
    import bs4
    out = []
    for cid in celex_ids:
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{cid}&format=html"
        raw = fetch(url)
        if not raw:
            continue
        html = raw.decode("utf-8", errors="ignore")
        try:
            soup = bs4.BeautifulSoup(html, "html.parser")
            node = soup.find("div", id="text")
            text = clean_text(node.get_text(" ", strip=True)) if node else ""
        except Exception as e:
            print(f"  EUR-Lex {cid}: parse error {e}", flush=True)
            text = ""
        if len(text.split()) < 50:
            print(f"  EUR-Lex {cid}: too short ({len(text.split())} words), skip", flush=True)
            continue
        title_m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
        title = clean_text(title_m.group(1)) if title_m else cid
        out.append((title, text))
        print(f"  EUR-Lex {cid}: {len(text.split())} words", flush=True)
        time.sleep(1.0)
    return out


# ---------- Oyez ----------
def fetch_oyez(n=20):
    """Pull recent SCOTUS case names via Oyez API (reference snippets only)."""
    out = []
    url = f"https://api.oyez.org/cases?per_page={n}&order=decided+desc"
    data = fetch(url)
    if not data:
        return out
    cases = json.loads(data.decode("utf-8", errors="ignore"))
    for c in cases:
        name = c.get("name", "")
        if name:
            out.append((name, f"Case: {name}. Court: Supreme Court of the United States."))
    return out


# ---------- Wex dictionary ----------
WEX_TERMS = [
    "force majeure", "indemnify", "indemnification", "tort", "negligence", "breach of contract",
    "liquidated damages", "consideration", "jurisdiction", "plaintiff", "defendant", "tortfeasor",
    "res ipsa loquitur", "prima facie", "mens rea", "actus reus", "habeas corpus", "certiorari",
    "amicus curiae", "stare decisis", "bona fide", "quantum meruit", "unjust enrichment",
    "fiduciary duty", "due diligence", "injunction", "arbitration", "mediation", "affidavit",
    "deposition", "discovery", "subpoena", "verdict", "remand", "precedent",
    "statute of limitations", "default judgment", "summary judgment", "waiver", "novation",
    "assignment", "subrogation", "hold harmless", "governing law", "venue",
    "forum non conveniens", "presumption", "remittitur", "inchoate", "voidable",
]


def fetch_wex(terms):
    """Fetch Wex term pages; extract definition text from the <main> container (modern Drupal).
    Trim leading nav ('force majeure' breadcrumb) and trailing boilerplate."""
    import bs4
    NAV_TAIL = ("Please help us improve", "Support Us!", "Legal Information Institute", "Wex | US Law")
    out = {}
    for term in terms:
        slug = term.replace(" ", "_")
        url = f"https://www.law.cornell.edu/wex/{slug}"
        html = fetch(url)
        if not html:
            continue
        try:
            soup = bs4.BeautifulSoup(html.decode("utf-8", errors="ignore"), "html.parser")
            main = soup.find("main") or soup.find("article")
            if not main:
                continue
            text = clean_text(main.get_text(" ", strip=True))
        except Exception:
            continue
        # trim trailing boilerplate
        for t in NAV_TAIL:
            i = text.find(t)
            if i > 0:
                text = text[:i]
        words = text.split()
        if words:
            out[term] = " ".join(words[:120])
        print(f"  Wex {term}: {'OK' if term in out else 'EMPTY'}", flush=True)
        time.sleep(0.8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    data = args.data
    os.makedirs(data, exist_ok=True)

    celex_ids = [
        "32004R0883", "32009R0593", "32007R0864", "32002R0044", "32004R0261",
        "31993L0013", "32011L0083", "31985L0374", "32006L0123", "31999L0093",
        "32014R0910", "32015R0848", "32006R1896", "32011R1259", "32010R0978",
        "31997L0007", "32009L0022", "31992L0085", "31989L0665", "32006R1011",
    ]

    print("=== EUR-Lex ===", flush=True)
    eur = fetch_eurlex_html(celex_ids)
    print("=== Oyez ===", flush=True)
    oyez = fetch_oyez(25)
    print("=== Wex ===", flush=True)
    wex = fetch_wex(WEX_TERMS)

    snippets = []
    sid = 0
    for title, text in eur:
        for snip in split_snippets(text):
            snippets.append({"id": f"eur_{sid:03d}", "source": "eurlex",
                             "title": title[:100], "text": snip})
            sid += 1
    for i, (name, txt) in enumerate(oyez):
        snippets.append({"id": f"oyez_{i:03d}", "source": "oyez",
                         "title": name[:100], "text": txt})
    terms = [{"term": t, "definition": d} for t, d in wex.items()]

    print(f"=== SUMMARY: {len(snippets)} snippets, {len(terms)} terms ===", flush=True)
    json.dump({"source": "auto-built-v2", "n_snippets": len(snippets), "n_terms": len(terms),
               "snippets": snippets, "terms": terms},
              open(os.path.join(data, "corpus_v2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved corpus_v2.json", flush=True)


if __name__ == "__main__":
    main()
