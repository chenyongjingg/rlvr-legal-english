# -*- coding: utf-8 -*-
"""Render the submission-ready manuscript from manuscript.md.

Removes the scientific-writing audit annotations that must NOT appear in the
submitted text:
  * [claim:Cxxx] and [evidence:Exxx] markers (any number, incl. comma lists)
  * [@E0xx] citation markers -> replaced by the numbered [n] style used in
    the Elsevier Numbered reference list
  * the HTML comment block carrying the [@E0xx] -> [n] mapping

Declaration handling: all six declaration sections are substituted with their
FINAL text from FINAL_DECLARATIONS. Three sections (Ethics and consent, Data
and code availability, Declaration of generative AI) have facts verified from
official licensing pages. Funding and Competing interests use metadata supplied
verbatim by the accountable human authors on 2026-08-18. Author contributions
is an AI-drafted CRediT proposal; the accountable author (Guyue Zhou) delegated
finalizing the role split to AI on 2026-08-19, so the text below is the final
author-approved version. The title-page editorial blockquote is replaced with
the author block (names, affiliations, corresponding author, e-mail addresses)
supplied by the accountable authors. manuscript.md is never modified, so the
fail-closed audit stays clean.

Output: manuscript_submission.md (original manuscript.md is NOT modified).
This is a derived artifact; the audited source of truth stays manuscript.md.
"""
import re
import sys

SRC = r"D:\周老师\paper\manuscript.md"
OUT = r"D:\周老师\paper\manuscript_submission.md"

# The mapping embedded in the manuscript's References comment (verified 2026-08-16).
CITE_MAP = {
    "E015": "1", "E019": "2", "E022": "3", "E033": "4", "E021": "5", "E018": "6", "E055": "7", "E056": "8", "E058": "9", "E020": "10", "E031": "11", "E032": "12", "E029": "13", "E051": "14", "E023": "15", "E052": "16", "E050": "17", "E053": "18", "E016": "19", "E057": "20", "E063": "21", "E035": "22", "E036": "23", "E037": "24", "E017": "25", "E038": "26", "E039": "27", "E024": "28", "E040": "29", "E054": "30", "E034": "31", "E049": "32", "E047": "33", "E048": "34", "E044": "35", "E043": "36", "E062": "37", "E046": "38", "E045": "39", "E064": "40", "E042": "41", "E041": "42", "E059": "43", "E060": "44", "E061": "45", "E030": "46",
}

CLAIM_EV = re.compile(r"\[(?:claim|evidence):[^\]]*\]")
CITE = re.compile(r"\[@(E\d{3})\]")
MAPPING_COMMENT = re.compile(r"<!--\s*Final editing pass:.*?-->\s*", re.S)

# Author block (supplied by the accountable human authors, 2026-08-18; both
# e-mail addresses supplied 2026-08-19). Raw HTML block so pandoc passes it
# through untouched; styled by pdf/manuscript.css p.author-block.
AUTHOR_BLOCK = (
    '<div class="authorblock">'
    '<p class="author-block"><strong>Yongjin Chen</strong> <sup>a</sup>, '
    '<strong>Guyue Zhou</strong> <sup>b</sup>, &#42;</p>'
    '<p class="author-block"><sup>a</sup> School of Public Security Management, '
    "People's Public Security University of China, Beijing, China</p>"
    '<p class="author-block"><sup>b</sup> School of International Police Studies, '
    "People's Public Security University of China, Beijing, China</p>"
    '<p class="author-block">E-mail addresses: 3337851329@qq.com (Y. Chen); '
    'zhouguyue@ppsuc.edu.cn (G. Zhou).</p>'
    '<p class="author-block">&#42; Corresponding author: Guyue Zhou.</p>'
    "</div>"
)
# The only blockquote in the manuscript is the editorial title note, which sits
# between the title H1 and the Abstract; replace it with the author block.
BLOCKQUOTE_BEFORE_ABSTRACT = re.compile(
    r"^>[^\n]*(?:\n>[^\n]*)*\n\n(?=## Abstract)", re.M
)

# Final, ready-to-submit declaration text. Substituted at render time only --
# manuscript.md (the audited source) is left untouched. The Ethics/Data/AI
# statements have facts verified 2026-08-16 from the official licensing pages of
# EUR-Lex / Wex / Oyez / Cornell LII and the model releases (Qwen Apache-2.0,
# DeBERTa-v3 MIT, MiniLM Apache-2.0). Funding and Competing interests use the
# author-supplied metadata (2026-08-18). Author contributions is the AI-drafted
# CRediT proposal finalized 2026-08-19 by delegation from the accountable author.
FINAL_DECLARATIONS = {
    "Author contributions":
        "**Yongjin Chen:** Conceptualization, Methodology, Software, Validation, "
        "Formal analysis, Investigation, Data curation, Writing – original draft.\n"
        "**Guyue Zhou:** Conceptualization, Resources, Supervision, Project "
        "administration, Funding acquisition, Writing – review & editing.",
    "Funding":
        "This study is supported by the school-level teaching research project of "
        "the People's Public Security University of China (Research on "
        "AI-Empowered Training Mode for Foreign Language Ability of Talents in "
        "Foreign-Related Rule of Law) (Project grant no.: 2025JXGG11).",
    "Competing interests":
        "The authors declare that they have no known competing financial interests "
        "or personal relationships that could have appeared to influence the work "
        "reported in this paper.",
    "Ethics and consent":
        "This work uses only publicly available data and open-licensed models and "
        "did not involve human participants; no personal data were collected. The "
        "source corpora are public legal documents — EUR-Lex (Creative Commons "
        "Attribution licensed), Wex (CC BY-NC-SA), Oyez (CC BY-NC-SA), and the "
        "Cornell LII U.S. Code (public-domain text, with LII markup separately "
        "copyrighted) — used in accordance with each source's terms. All models "
        "are released under open licenses: the Qwen3.5-4B, Qwen3.5-9B, and Qwen3-4B "
        "language models are Apache-2.0; the NLI DeBERTa-v3 model is MIT; the MiniLM "
        "embedding encoder is Apache-2.0.",
    "Data and code availability":
        "The code and configuration files used in this study will be made available "
        "on GitHub upon publication. The corpus is built from publicly available "
        "legal sources: EUR-Lex (CC BY licensed; metadata CC0), Wex (CC BY-NC-SA), "
        "Oyez (CC BY-NC-SA), and the Cornell LII U.S. Code (public-domain text; LII "
        "markup copyrighted). Because parts of the corpus are non-commercial "
        "share-alike (Wex, Oyez), the derived dataset is released for non-commercial "
        "research use under a compatible license, and users must observe each "
        "source's terms. The trained LoRA adapters and code are released under an "
        "open-source license. The open-weights SLMs (Qwen3.5-4B, Qwen3.5-9B, "
        "Qwen3-4B; Apache-2.0), NLI model (DeBERTa-v3; MIT), and embedding encoder "
        "(MiniLM; Apache-2.0) are publicly available under their respective "
        "licenses.",
    "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process":
        "During the preparation of this work the author(s) used Claude (Anthropic) "
        "and local open-weights models as drafting and editing assistants in order "
        "to improve language, structure, and formatting. After using these tools, "
        "the author(s) reviewed and edited the content as needed and take(s) full "
        "responsibility for the content of the publication. Generative AI was also "
        "used within the research itself — as the generated learning materials "
        "and as the independent judge model described in the Methods — and no "
        "AI tool is listed as an author.",
}

# Replace "## <header>\n\n[[TODO: ...]]" with "## <header>\n\n<final text>".
DECL_PATTERN = {
    header: re.compile(
        r"(## " + re.escape(header) + r"\n\n)" + r"\[\[TODO.*?\]\]", re.S
    )
    for header in FINAL_DECLARATIONS
}


def render(text: str) -> str:
    # 1) citations first (so a numbered marker is not mistaken for an audit tag)
    text = CITE.sub(lambda m: "[" + CITE_MAP.get(m.group(1), m.group(0)) + "]", text)
    # 2) strip all claim/evidence markers
    text = CLAIM_EV.sub("", text)
    # 3) drop the mapping comment
    text = MAPPING_COMMENT.sub("", text)
    # 4) normalise whitespace left by removed markers: "foo [n] . bar" -> "foo [n]. bar"
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+;", ";", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\] \.", "].", text)
    # collapse stray double spaces (keep newlines)
    text = re.sub(r" +", " ", text)
    # 5) replace the editorial title-note blockquote with the author block
    text = BLOCKQUOTE_BEFORE_ABSTRACT.sub(lambda m: AUTHOR_BLOCK + "\n\n", text)
    # 6) substitute the declaration sections with their final text
    for header, pat in DECL_PATTERN.items():
        text = pat.sub(lambda m, h=header: m.group(1) + FINAL_DECLARATIONS[h], text)
    return text


def main() -> int:
    text = open(SRC, encoding="utf-8").read()
    out = render(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    n_claims = len(CLAIM_EV.findall(text))
    n_cites = len(CITE.findall(text))
    n_ab = len(BLOCKQUOTE_BEFORE_ABSTRACT.findall(text))
    n_decl_subbed = len(
        [h for h in FINAL_DECLARATIONS if re.search(DECL_PATTERN[h], text)]
    )
    print(f"stripped {n_claims} claim/evidence markers, converted {n_cites} citations")
    print(f"author block inserted: {n_ab}")
    print(f"finalised declaration sections: {n_decl_subbed}/{len(FINAL_DECLARATIONS)}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
