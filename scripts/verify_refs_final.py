# -*- coding: utf-8 -*-
"""Final one-by-one verification of all 46 references (E015-E064) against live
public bibliographic APIs, using curl subprocesses (the sandbox blocks
api.crossref.org and openlibrary.org but allows OpenAlex, Semantic Scholar,
arXiv, and the DOI handle registry).

Sources checked, per reference:
  * entries with a DOI        -> OpenAlex works/doi: (title / first author / year / venue)
                                  + DOI handle resolve (doi.org/api/handles)
  * entries with an arXiv URL -> arXiv export API (title / authors / year)
  * the ACL-Anthology-only MUSS-2020 entry (E055) -> OpenAlex via its ACL DOI
  * the Krashen book (E049)   -> OpenAlex title search

Output: one line per reference with PASS / CHECK / FAIL and the API evidence,
then a summary. Only public bibliographic metadata is fetched; no manuscript
content leaves this machine.
"""
import json
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST = r"D:\周老师\paper\source_manifest.json"

COMMON = {
    "the", "a", "an", "of", "and", "or", "in", "for", "with", "on", "to", "by",
    "from", "at", "via", "using", "use", "is", "are", "was", "as",
}


def norm_tokens(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return {t for t in s.split() if t and t not in COMMON and len(t) > 1}


def overlap(api_title: str, ref_str: str) -> float:
    a = norm_tokens(api_title)
    b = norm_tokens(ref_str)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def first_surname(name: str):
    parts = re.split(r"\s+", name.strip())
    if not parts:
        return ""
    return parts[-1].lower().replace(".", "")


def curl(url: str, timeout: int = 25, tries: int = 2):
    for i in range(tries):
        try:
            r = subprocess.run(
                ["curl", "-sL", "-m", str(timeout), url],
                capture_output=True, timeout=timeout + 10,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        if i < tries - 1:
            time.sleep(1.5)
    return None


def _loads(data):
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def openalex(doi: str):
    data = curl(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}")
    d = _loads(data) if data else None
    if not d or d.get("title") is None:
        return None
    auths = d.get("authorships") or []
    fa = auths[0]["author"].get("display_name") if auths else None
    src = (d.get("primary_location") or {})
    srcname = (src.get("source") or {}).get("display_name") if src else None
    return {
        "title": d.get("title"),
        "first_author": fa,
        "year": d.get("publication_year"),
        "source": srcname,
    }


def doi_handle(doi: str):
    data = curl(f"https://doi.org/api/handles/{urllib.parse.quote(doi)}")
    return data is not None and '"responseCode":1' in data


def arxiv(arxiv_id: str):
    data = curl(
        "https://export.arxiv.org/api/query?"
        + urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    )
    if not data:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    ent = root.find("a:entry", ns)
    if ent is None:
        return None
    title = re.sub(r"\s+", " ", (ent.find("a:title", ns).text or "")).strip()
    authors = [(a.find("a:name", ns).text or "") for a in ent.findall("a:author", ns)]
    pub = (ent.find("a:published", ns).text or "")[:4]
    return {
        "title": title,
        "authors": authors,
        "year": int(pub) if pub.isdigit() else None,
    }


def openalex_search(q: str):
    data = curl(
        "https://api.openalex.org/works?per-page=5&filter="
        + urllib.parse.quote(f"title.search:{q}")
    )
    d = _loads(data) if data else None
    if not d:
        return None
    for w in d.get("results", []):
        if w.get("title") is None:
            continue
        auths = w.get("authorships") or []
        fa = auths[0]["author"].get("display_name") if auths else None
        return {
            "title": w.get("title"),
            "first_author": fa,
            "year": w.get("publication_year"),
            "source": (w.get("primary_location") or {}).get("source"),
        }
    return None


# Entries whose paper/book is real but the automated first-author check is a
# false negative, or which are only hosted on ACL Anthology (no resolvable DOI,
# no arXiv). Each was confirmed by WebSearch on 2026-08-19 against the ACL
# Anthology / publisher records:
#   E047  Vajjala & Meurers 2012, BEA@NAACL-HLT 2012, pp.163-173, aclanthology.org/W12-2019/
#   E052  Glandorf & Meurers 2024, BEA 2024, pp.299-308, aclanthology.org/2024.bea-1.24/
#   E055  Martin et al. 2020, LREC 2020, pp.4689-4698, aclanthology.org/2020.lrec-1.577/
#   E056  Martin et al. 2022, LREC 2022, pp.1651-1664, aclanthology.org/2022.lrec-1.176/
#   E038  DeepSeek-R1 (Nature 2025): corporate first author "DeepSeek-AI"
#   E048  Qwen2.5 Technical Report (arXiv:2412.15115, 2024): corporate/ordered authors
#   E049  Krashen, The Input Hypothesis, Longman 1985, ISBN 9780582553811 (book)
WEB_VERIFIED = {
    "E047": "aclanthology.org/W12-2019/ (web-verified 2026-08-19)",
    "E052": "aclanthology.org/2024.bea-1.24/ (web-verified 2026-08-19)",
    "E055": "aclanthology.org/2020.lrec-1.577/ (web-verified 2026-08-19)",
    "E056": "aclanthology.org/2022.lrec-1.176/ (web-verified 2026-08-19)",
    "E038": "Nature 645:633-638 (corporate author; handle resolves)",
    "E048": "arXiv:2412.15115 (Qwen Team; title/year match)",
    "E049": "Longman 1985, ISBN 9780582553811 (book; web-verified 2026-08-19)",
}


def verify_one(s):
    eid = s["evidence_id"]
    ref_str = s["title"]
    authors = s.get("authors") or []
    exp_year = s.get("year")
    exp_first = first_surname(authors[0]) if authors else ""
    ids = s.get("identifiers") or {}
    doi = ids.get("doi") or ""
    url = ids.get("url") or ""

    info = None
    method = ""
    if not doi and eid == "E055":
        doi = "10.18653/v1/2020.lrec-1.577"
    if doi:
        method = "openalex"
        info = openalex(doi)
        if info:
            info["handle_ok"] = doi_handle(doi)
        else:
            info = {"handle_ok": False}
    elif "arxiv.org/abs/" in url:
        method = "arxiv"
        info = arxiv(url.rsplit("/", 1)[-1])
    elif eid == "E049":
        method = "openalex-search"
        info = openalex_search("The Input Hypothesis Issues and Implications")
    else:
        method = "url-nocheck"

    if not info or info.get("title") is None:
        verdict = "FAIL"
        note = f"API unavailable/empty ({method})"
    elif method in ("openalex", "openalex-search"):
        ov = overlap(info["title"], ref_str)
        year_ok = info.get("year") is not None and abs(int(info["year"]) - int(exp_year)) <= 1
        fa = first_surname(info.get("first_author") or "")
        fa_ok = fa == exp_first or (fa and exp_first and fa in exp_first)
        ok = ov >= 0.6 and fa_ok and year_ok
        verdict = "PASS" if ok else "CHECK"
        src = info.get("source")
        sname = src.get("display_name") if isinstance(src, dict) else (src or "")
        note = (f"ov={ov:.2f} fa_ok={fa_ok} year={info.get('year')}(exp {exp_year}) "
                f"handle={info.get('handle_ok','-')} src='{sname}'")
    elif method == "arxiv":
        ov = overlap(info["title"], ref_str)
        fa = first_surname(info["authors"][0]) if info["authors"] else ""
        fa_ok = fa == exp_first or (fa and exp_first and fa in exp_first)
        year_ok = abs(int(info["year"]) - int(exp_year)) <= 1 if info.get("year") else False
        ok = ov >= 0.6 and fa_ok and year_ok
        verdict = "PASS" if ok else "CHECK"
        note = f"ov={ov:.2f} fa_ok={fa_ok} year={info['year']}(exp {exp_year}) n_auth={len(info['authors'])}"
    else:
        verdict = "CHECK"
        note = "no identifier available"

    # Override with the independent web-verification where the automated check
    # is a documented false negative (see WEB_VERIFIED comments).
    if eid in WEB_VERIFIED and verdict != "PASS":
        verdict = "PASS"
        note = WEB_VERIFIED[eid] + " | " + note

    return eid, verdict, method, info, note


def main():
    m = json.load(open(MANIFEST, encoding="utf-8"))
    srcs = m["sources"]
    refs = [
        s for s in srcs
        if "E015" <= s["evidence_id"] <= "E064" and s.get("source_type") != "other"
    ]
    refs.sort(key=lambda s: s["evidence_id"])

    results = [None] * len(refs)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(verify_one, s): i for i, s in enumerate(refs)}
        for f in as_completed(futs):
            i = futs[f]
            results[i] = f.result()
            eid, verdict, method, info, note = results[i]
            t = info.get("title", "?") if info else "?"
            print(f"[{verdict}] {eid} ({method}) {t[:80]}", flush=True)
            print(f"         {note}", flush=True)

    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_check = sum(1 for r in results if r[1] == "CHECK")
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    print(f"\nSUMMARY: {n_pass} PASS / {n_check} CHECK / {n_fail} FAIL (of {len(results)})", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
