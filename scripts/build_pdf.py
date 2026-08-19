# -*- coding: utf-8 -*-
"""Reproducible PDF build for the review manuscript.

Pipeline:
  manuscript.md (audited source of truth)
    -> render_submission.render()   (strip [claim/evidence], @E0xx -> [n], drop mapping comment)
    -> pandoc (standalone fragment, no metadata title)
    -> post-process HTML (tag banner + title, fix ../figures/ image paths)
    -> Microsoft Edge headless --print-to-pdf
  Output: pdf/manuscript_review.pdf

Dependencies: pandoc on PATH; Edge at the default install path.
Usage:  python scripts/build_pdf.py [--keep-html] [--keep-md]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from render_submission import render  # reuse the audited stripping/conversion

BASE = r"D:\周老师\paper"
MD = os.path.join(BASE, "manuscript.md")
PDF_DIR = os.path.join(BASE, "pdf")
HTML_OUT = os.path.join(PDF_DIR, "manuscript.html")
PDF_OUT = os.path.join(PDF_DIR, "manuscript_review.pdf")
CSS = os.path.join(PDF_DIR, "manuscript.css")
FIGS = os.path.join(BASE, "figures")

# Edge default install path (Windows)
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_edge() -> str:
    for c in EDGE_CANDIDATES:
        if os.path.exists(c):
            return c
    raise SystemExit("Edge not found; install Microsoft Edge or set EDGE_PATH.")


def postprocess_html(html: str) -> str:
    # Tag the manuscript title H1 (DRAFT banner removed at finalization).
    # Pandoc may line-wrap the tag as "<h1\n\nid=...", so allow any whitespace
    # (or ">") right after "<h1", not just a space or ">".
    lines = html.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if re.match(r"^\s*<h1(?=[\s>])", ln):
            lines[i] = re.sub(r"<h1(?=[\s>])", r'<h1 class="title"', ln, count=1)
            break
    html = "".join(lines)
    # Figures live at ../figures/ relative to pdf/
    html = html.replace('src="figures/', 'src="../figures/')
    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-html", action="store_true", help="keep intermediate manuscript.html")
    args = ap.parse_args()

    if not os.path.exists(CSS):
        raise SystemExit(f"stylesheet missing: {CSS}")

    text = open(MD, encoding="utf-8").read()
    cleaned = render(text)
    # pandoc reads from stdin; emit standalone fragment without metadata title
    proc = subprocess.run(
        ["pandoc", "-f", "markdown+smart", "-t", "html",
         "--standalone", "--css", "manuscript.css",
         "--metadata", "lang=en"],
        input=cleaned.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.buffer.write(proc.stderr)
        return proc.returncode
    html = proc.stdout.decode("utf-8")
    html = postprocess_html(html)

    os.makedirs(PDF_DIR, exist_ok=True)
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {HTML_OUT} ({len(html)} bytes)")

    edge = find_edge()
    url = HTML_OUT.replace("\\", "/")
    r = subprocess.run(
        [edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF_OUT}", f"file:///{url}"],
        capture_output=True,
    )
    # Edge exits 0 even with console noise (QQBrowser import etc.); verify the file
    if not os.path.exists(PDF_OUT):
        sys.stderr.write(r.stderr.decode("utf-8", "replace"))
        raise SystemExit("Edge did not produce a PDF")
    size = os.path.getsize(PDF_OUT)
    print(f"wrote {PDF_OUT} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
