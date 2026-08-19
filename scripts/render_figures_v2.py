# -*- coding: utf-8 -*-
"""Figure 1 + Figure 2 (v2 — redrawn with fireworks-tech-graph Flat Icon design system).

Upgrades over the v1 render_figures.py (kept untouched for rollback):
  * icons in every node (inline SVG, Feather-style stroke icons)
  * semantic arrow colours (blue = main/data flow, green = retrieval,
    purple dashed = feedback / revision loop, red dashed = honest re-scoring)
  * swim-lane dashed containers with light tint fills + lane titles
  * soft depth: subtle drop shadows, saturated borders, tinted fills
  * a legend (arrow colour key + box-colour key)
  * Figure 2 gains annotation call-outs (SFT collapse / RLVR restore,
    inside-band note) routed with arrowheads instead of bare text

All content matches manuscript.md Sec 3.1–3.6 / Fig.1–2 captions and
honest_table.json (2026-08-17 numbers, re-verified 2026-08-18).

Outputs (300+ dpi PNG + vector PDF) into D:/周老师/paper/figures/.
Usage: python scripts/render_figures_v2.py
"""
import json
import os
import sys

FIGDIR = r"D:\周老师\paper\figures"
os.makedirs(FIGDIR, exist_ok=True)
HONEST = r"C:\Users\33378\aigc_wechat\results\legal\honest_table.json"

# --------------------------------------------------------------------------
# shared CSS — design tokens adopted from fireworks-tech-graph Style 1
# --------------------------------------------------------------------------
CSS = """
html,body{margin:0;padding:0;background:#ffffff}
body{font-family:'Helvetica Neue',Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif;color:#1f2937}
.fig{position:relative;width:178mm;background:#ffffff}
.lane{position:absolute;border-radius:3mm;border:1.1px dashed #c3cad6;
      background:rgba(249,250,251,.55);box-sizing:border-box}
.lane-title{position:absolute;font-size:8.6pt;font-weight:700;color:#334155;letter-spacing:.15px}
.lane-title .tag{display:inline-block;background:#eef2f7;border:1px solid #d6dde7;
      border-radius:1.2mm;padding:0.1mm 1.6mm;margin-right:1.4mm;color:#475569}
.box{position:absolute;border-radius:2.2mm;border:1.15px solid;display:flex;
     flex-direction:column;justify-content:center;align-items:center;
     text-align:center;padding:.7mm 1.2mm;box-sizing:border-box;overflow:hidden;
     box-shadow:0 .5px 0 rgba(0,0,0,.05),0 1.1px 2.6px rgba(31,41,55,.09)}
.ic{display:block;margin:0 0 .4mm 0;flex:0 0 auto}
.bt{font-size:7.3pt;font-weight:700;color:#111827;line-height:1.14}
.bs{font-size:6.3pt;color:#4b5563;line-height:1.24;margin-top:.2mm}
.c-data {border-color:#1d4ed8;background:linear-gradient(180deg,#eff6ff,#dbeafe)}
.c-data .bt{color:#1e3a8a}
.c-train{border-color:#15803d;background:linear-gradient(180deg,#f0fdf4,#dcfce7)}
.c-train .bt{color:#14532d}
.c-model{border-color:#0f766e;background:linear-gradient(180deg,#f0fdfa,#ccfbf1)}
.c-model .bt{color:#134e4a}
.c-eval {border-color:#c2410c;background:linear-gradient(180deg,#fff7ed,#fed7aa)}
.c-eval .bt{color:#7c2d12}
.c-judge{border-color:#7c3aed;background:linear-gradient(180deg,#faf5ff,#ede9fe)}
.c-judge .bt{color:#4c1d95}
.emph{stroke-width:2.1px;box-shadow:0 0 0 1px rgba(0,0,0,.06),0 1.6px 4.4px rgba(31,41,55,.16)}
.ml{position:absolute;font-size:6.3pt;font-style:italic;color:#6b7280;text-align:center;line-height:1.25}
.legend{position:absolute;border:1px solid #e5e7eb;border-radius:2mm;background:#fafafa;box-sizing:border-box}
.legend .lt{font-size:6.8pt;font-weight:700;color:#374151}
.legend .li{font-size:6.3pt;color:#4b5563;white-space:nowrap}
svg.arrows{position:absolute;left:0;top:0;pointer-events:none}
.fig2-panel{position:absolute;border:1.1px solid #d1d5db;border-radius:2.4mm;background:#ffffff;
            box-shadow:0 .5px 0 rgba(0,0,0,.04),0 1.2px 3px rgba(31,41,55,.07)}
.p2-title{font-size:8.6pt;font-weight:700;color:#111827}
.p2-note{font-size:6.4pt;fill:#4b5563}
"""

# palette (matches the CSS tints)
BLUE, GREEN, TEAL, ORANGE, PURPLE, RED, GRAY = "#2563eb", "#16a34a", "#0d9488", "#ea580c", "#9333ea", "#dc2626", "#6b7280"

# Feather-style stroke icon innards (24x24 viewBox)
ICONS = {
    "book": ('<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
             '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'),
    "db": ('<ellipse cx="12" cy="5" rx="9" ry="3"/>'
           '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
           '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'),
    "branch": ('<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/>'
               '<circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>'),
    "sliders": ('<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
                '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
                '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
                '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
                '<line x1="17" y1="16" x2="23" y2="16"/>'),
    "refresh": ('<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
                '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'),
    "list": ('<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/>'
             '<line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/>'
             '<line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>'),
    "file": ('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
             '<polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>'
             '<line x1="16" y1="17" x2="8" y2="17"/>'),
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "cpu": ('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
            '<line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>'
            '<line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>'
            '<line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>'
            '<line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'),
    "check": ('<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
              '<polyline points="22 4 12 14.01 9 11.01"/>'),
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "arrow": '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>',
}


def icon(name, color, size=4.0):
    """Inline SVG icon sized in mm."""
    return (f'<svg class="ic" viewBox="0 0 24 24" width="{size}mm" height="{size}mm" '
            f'fill="none" stroke="{color}" stroke-width="1.9" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICONS[name]}</svg>')


def _arrow(x1, y1, x2, y2, w=0.55, dash=None, color=BLUE, elbow=None, marker="ah-blue"):
    """SVG arrow primitive; coordinates in mm. elbow=(mx,my,bx,by) routes a right angle."""
    d = "none" if dash is None else dash
    if elbow:
        mx, my, bx, by = elbow
        p = f"M {x1} {y1} L {mx} {my} L {bx} {by}"
        return (f'<path d="{p}" stroke="{color}" stroke-width="{w}" fill="none" '
                f'stroke-dasharray="{d}" marker-end="url(#{marker})"/>')
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}" stroke-dasharray="{d}" marker-end="url(#{marker})"/>')


def _markers():
    m = []
    for name, col in (("ah-blue", BLUE), ("ah-green", GREEN), ("ah-purple", PURPLE),
                      ("ah-red", RED), ("ah-gray", GRAY)):
        m.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                 f'markerWidth="2.1" markerHeight="2.1" orient="auto-start-reverse">'
                 f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>')
    return "".join(m)


def _defs():
    return ("<defs>" + _markers() + "</defs>")


# --------------------------------------------------------------------------
# FIGURE 1 — framework overview (v2)
# --------------------------------------------------------------------------
def figure1_html():
    W, H = 178, 130
    boxes, labels, arrows = [], [], []
    lanes = []

    # ---- lane (a) training ------------------------------------------------
    lanes.append(("(a) Training", 3, 4, 172, 44))
    row_a = [("c-data", 7, 12, 28, 16, "book", "Legal sources", ["EUR-Lex · Wex · Oyez"]),
             ("c-data", 41, 12, 26, 16, "db", "Corpus", ["968 snippets + 49 terms"]),
             ("c-data", 73, 12, 24, 16, "branch", "Split (id-hash)", ["train / val / test", "80 / 10 / 10"]),
             ("c-train", 103, 12, 24, 16, "sliders", "SFT (QLoRA)", ["r = 16, α = 32", "21.23 M params"]),
             ("c-train emph", 133, 12, 38, 16, "refresh", "RLVR / GRPO", ["5-dim verifiable reward"])]
    for cls, x, y, w, h, ic, t, subs in row_a:
        boxes.append((cls, x, y, w, h, ic, t, subs))
    for x1, x2 in ((35, 41), (67, 73), (97, 103), (127, 133)):
        arrows.append(_arrow(x1, 20, x2, 20))
    labels.append((17, 28.9, "public legal text (CC BY / public domain)"))
    labels.append((46, 28.9, "teacher simplifications"))
    # reward stack (wide detail box under the training row)
    boxes.append(("c-train", 41, 33, 130, 11, None, "Reward stack W",
                  ["format 0.15 · difficulty 0.20 · terminology 0.15 · anti-copying 0.25 · faithfulness 0.25",
                   "(v2: difficulty weight 0.30)"]))
    arrows.append(_arrow(152, 28, 152, 32.6, color=BLUE, marker="ah-blue"))
    # GRPO update loop over the RLVR box
    arrows.append(f'<path d="M 133 12 C 137 7.8 167 7.8 171 12" stroke="{PURPLE}" '
                  f'stroke-width=".55" fill="none" stroke-dasharray="3 2.2" marker-end="url(#ah-purple)"/>')
    labels.append((152, 9.8, "GRPO policy update (rewarded)"))

    # ---- lane (b) inference ------------------------------------------------
    lanes.append(("(b) Inference", 3, 50, 172, 39))
    row_b = [("c-data", 7, 56, 30, 16, "file", "Legal passage", ["+ target difficulty level"]),
             ("c-model emph", 43, 56, 64, 16, "cpu", "SLM  Qwen3.5-4B  ·  GRPO v2",
              ["multi-agent loop (4 agents):", "planner · writer · editor · reviewer"]),
             ("c-eval", 113, 56, 58, 16, "check", "Simplified passage", ["target FRE band ±20"])]
    for cls, x, y, w, h, ic, t, subs in row_b:
        boxes.append((cls, x, y, w, h, ic, t, subs))
    arrows.append(_arrow(37, 64, 42.6, 64))
    arrows.append(_arrow(107, 64, 112.6, 64))
    # RAG retrieval feeding the SLM
    boxes.append(("c-model", 7, 76, 48, 12.5, "search", "RAG retrieval",
                  ["MiniLM-L6-v2 · 861-passage KB"]))
    arrows.append(f'<path d="M 31 76 L 31 73.5 L 75 73.5 L 75 72.4" stroke="{GREEN}" '
                  f'stroke-width=".6" fill="none" marker-end="url(#ah-green)"/>')
    labels.append((52, 75.2, "retrieved evidence"))

    # ---- lane (c) evaluation -----------------------------------------------
    lanes.append(("(c) Evaluation", 3, 90, 172, 28))
    row_c = [("c-eval", 7, 100, 78, 16, "activity", "Automatic verifiable rewards",
              ["tot · diff · term · copy · faith · contra (NLI)",
               "FRE / SARI · full-107 replication"]),
             ("c-judge", 89, 100, 82, 16, "award", "Independent cross-family judge",
              ["Qwen3-4B · 4 dims 1–5 + confidence",
               "ECE · self-consistency · mean ± std (4 seeds)"])]
    for cls, x, y, w, h, ic, t, subs in row_c:
        boxes.append((cls, x, y, w, h, ic, t, subs))
    # honest re-scoring arrows from the simplified passage into both panels
    arrows.append(f'<path d="M 142 72 L 142 86 L 46 86 L 46 99.6" stroke="{RED}" '
                  f'stroke-width=".55" fill="none" stroke-dasharray="3 2.2" marker-end="url(#ah-red)"/>')
    arrows.append(f'<path d="M 142 72 L 142 88 L 130 88 L 130 99.6" stroke="{RED}" '
                  f'stroke-width=".55" fill="none" stroke-dasharray="3 2.2" marker-end="url(#ah-red)"/>')
    labels.append((158, 78.5, "honest re-scoring (decontaminated)"))

    # ---- legend ------------------------------------------------------------
    legend = ("<div class='legend' style='left:3mm;top:121.5mm;width:172mm;height:6.4mm'>"
              "<div class='lt' style='position:absolute;left:2.5mm;top:2.1mm'>Arrow key:</div>"
              f"<svg width='18mm' height='4mm' viewBox='0 0 18 4' style='position:absolute;left:20mm;top:1.4mm'>"
              f"<line x1='0' y1='2' x2='11' y2='2' stroke='{BLUE}' stroke-width='.9' marker-end='url(#ah-blue)'/></svg>"
              "<div class='li' style='position:absolute;left:41mm;top:2.1mm'>main / data flow</div>"
              f"<svg width='18mm' height='4mm' viewBox='0 0 18 4' style='position:absolute;left:70mm;top:1.4mm'>"
              f"<line x1='0' y1='2' x2='11' y2='2' stroke='{GREEN}' stroke-width='.9' marker-end='url(#ah-green)'/></svg>"
              "<div class='li' style='position:absolute;left:91mm;top:2.1mm'>retrieval evidence</div>"
              f"<svg width='18mm' height='4mm' viewBox='0 0 18 4' style='position:absolute;left:122mm;top:1.4mm'>"
              f"<line x1='0' y1='2' x2='11' y2='2' stroke='{PURPLE}' stroke-width='.9' "
              f"stroke-dasharray='2.5 1.8' marker-end='url(#ah-purple)'/></svg>"
              "<div class='li' style='position:absolute;left:143mm;top:2.1mm'>feedback / loop</div>"
              "</div>")

    # ---- assemble ----------------------------------------------------------
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><style>", CSS, "</style></head><body>"]
    html.append(f"<div class='fig' style='width:{W}mm;height:{H}mm'>")
    for t, x, y, w, h in lanes:
        html.append(f"<div class='lane' style='left:{x}mm;top:{y}mm;width:{w}mm;height:{h}mm'></div>")
        html.append(f"<div class='lane-title' style='left:{x + 2}mm;top:{y + 1.2}mm'>"
                    f"<span class='tag'>{t.split(' ')[0]}</span>{t[4:]}</div>")
    for cls, x, y, w, h, ic, t, subs in boxes:
        title = f"<div class='bt'>{t}</div>"
        body = "".join(f"<div class='bs'>{s}</div>" for s in subs)
        ic_svg = "" if ic is None else icon(
            ic, {"c-data": BLUE, "c-train": GREEN, "c-model": TEAL,
                 "c-eval": ORANGE, "c-judge": PURPLE}[cls.split()[0]])
        html.append(f"<div class='box {cls}' style='left:{x}mm;top:{y}mm;width:{w}mm;height:{h}mm'>"
                    f"{ic_svg}{title}{body}</div>")
    for x, y, t in labels:
        html.append(f"<div class='ml' style='left:{x - 15}mm;top:{y}mm;width:30mm'>{t}</div>")
    html.append(f"<svg class='arrows' width='{W}mm' height='{H}mm' viewBox='0 0 {W} {H}'>{_defs()}"
                + "".join(arrows) + "</svg>")
    html.append(legend)
    html.append("</div></body></html>")
    return ("".join(html), W, H)


# --------------------------------------------------------------------------
# FIGURE 2 — results (v2 panels, richer annotations)
# --------------------------------------------------------------------------
def _y(v, ytop, ybot, vmax, vmin):
    return ytop + (vmax - v) / (vmax - vmin) * (ybot - ytop)


def figure2_html():
    h = json.load(open(HONEST, encoding="utf-8"))["tags"]
    tots = {g: h[g]["all"]["tot"] for g in ("G1", "G2", "G3", "G4", "G5")}
    tots["G6"] = h["L3_max3"]["all"]["tot"]          # corrected framework
    adv = h["ADV"]["levels"]
    ach = {lv: adv[lv]["fre_mean"] for lv in ("beginner", "intermediate", "advanced")}
    tgt = {"beginner": 63, "intermediate": 45, "advanced": 26}

    S = 72.0 / 25.4
    PXM, PYM = 84, 54
    PX, PY = PXM * S, PYM * S
    x0, x1 = 10 * S, 81.5 * S
    yt, yb = 9 * S, 44 * S
    vmin, vmax = 0.73, 0.92
    svg_a = [f'<svg width="{PXM}mm" height="{PYM}mm" viewBox="0 0 {PX:.1f} {PY:.1f}" '
             f'font-family="Arial,Helvetica,sans-serif">']
    for v in (0.76, 0.80, 0.84, 0.88, 0.92):
        yy = _y(v, yt, yb, vmax, vmin)
        svg_a.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" stroke="#e5e7eb" stroke-width="0.6"/>')
        svg_a.append(f'<text x="{x0 - 3.5}" y="{yy + 1.8:.2f}" font-size="6" text-anchor="end" fill="#6b7280">{v:.2f}</text>')
    svg_a.append(f'<line x1="{x0}" y1="{yt}" x2="{x0}" y2="{yb}" stroke="#9ca3af" stroke-width="0.8"/>')
    svg_a.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="#9ca3af" stroke-width="0.8"/>')
    comp = ["base", "+RAG", "+agents", "+GRPO v1", "+SFT", "+GRPO v2"]
    order = ["G1", "G2", "G3", "G4", "G5", "G6"]
    n = len(order)
    bw = 6.2 * S
    gap = (x1 - x0 - n * bw) / (n - 1)
    for i, g in enumerate(order):
        cx = x0 + bw / 2 + i * (bw + gap)
        v = tots[g]
        top = _y(v, yt, yb, vmax, vmin)
        if g == "G5":
            fill, edge = "#ef9a9a", "#b91c1c"
        elif g == "G6":
            fill, edge = "#6ee7a0", "#15803d"
        else:
            fill, edge = "#cbd5e1", "#64748b"
        svg_a.append(f'<rect x="{cx - bw/2:.2f}" y="{top:.2f}" width="{bw}" height="{yb - top:.2f}" '
                     f'fill="{fill}" stroke="{edge}" stroke-width="1.1" rx="1.8"/>')
        svg_a.append(f'<text x="{cx:.2f}" y="{top - 2.8:.2f}" font-size="6.8" font-weight="bold" '
                     f'text-anchor="middle" fill="#111827">{v:.3f}</text>')
        svg_a.append(f'<text x="{cx:.2f}" y="{yb + 7.6:.2f}" font-size="6.3" text-anchor="middle" '
                     f'fill="#374151" font-style="italic">{comp[i]}</text>')
    # annotation call-outs (elbow arrows to G5 / G6)
    g5 = x0 + bw / 2 + 4 * (bw + gap)
    g6 = x0 + bw / 2 + 5 * (bw + gap)
    svg_a.append(f'<path d="M {g5:.2f} {yb + 17.5} L {g5:.2f} {yb + 11.2}" stroke="#b91c1c" stroke-width="0.8" '
                 f'fill="none" marker-end="url(#ah2-red)"/>')
    svg_a.append(f'<path d="M {g6:.2f} {yb + 24.5} L {g6:.2f} {yb + 11.2}" stroke="#15803d" stroke-width="0.8" '
                 f'fill="none" marker-end="url(#ah2-green)"/>')
    svg_a.append(f'<text x="{g5:.2f}" y="{yb + 20.0}" font-size="6.2" text-anchor="middle" fill="#b91c1c" '
                 f'font-weight="bold">SFT without GRPO collapses difficulty control</text>')
    svg_a.append(f'<text x="{g6:.2f}" y="{yb + 27.0}" font-size="6.2" text-anchor="middle" fill="#15803d" '
                 f'font-weight="bold">RLVR restores (corrected framework)</text>')
    svg_a.append(f'<text x="{PX/2}" y="12.8" font-size="8.6" font-weight="bold" '
                 f'text-anchor="middle" fill="#111827">(a) System progression (G1–G6)</text>')
    svg_a.append(f'<text x="6" y="{yb + 6.4}" font-size="6.2" fill="#4b5563">Total verifiable reward</text>')
    svg_a.append(f'<defs><marker id="ah2-red" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="2.2" '
                 f'markerHeight="2.2" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b91c1c"/></marker>'
                 f'<marker id="ah2-green" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="2.2" '
                 f'markerHeight="2.2" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#15803d"/></marker></defs>')
    svg_a.append("</svg>")

    # ---------- panel (b): difficulty vs targets ----------------------------
    svg_b = [f'<svg width="{PXM}mm" height="{PYM}mm" viewBox="0 0 {PX:.1f} {PY:.1f}" '
             f'font-family="Arial,Helvetica,sans-serif">']
    y0, y1b, vlo, vhi = 9 * S, 44 * S, 0, 100
    for v in (0, 20, 40, 60, 80, 100):
        yy = _y(v, y0, y1b, vhi, vlo)
        svg_b.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" stroke="#e5e7eb" stroke-width="0.6"/>')
        svg_b.append(f'<text x="{x0 - 3.5}" y="{yy + 1.8:.2f}" font-size="6" text-anchor="end" fill="#6b7280">{v}</text>')
    svg_b.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1b}" stroke="#9ca3af" stroke-width="0.8"/>')
    svg_b.append(f'<line x1="{x0}" y1="{y1b}" x2="{x1}" y2="{y1b}" stroke="#9ca3af" stroke-width="0.8"/>')
    lv_order = ["beginner", "intermediate", "advanced"]
    lab = {"beginner": "beginner (63)", "intermediate": "intermediate (45)", "advanced": "advanced (26)"}
    bw2 = 9.5 * S
    gap2 = (x1 - x0 - 3 * bw2) / 2
    inband = True
    for i, lv in enumerate(lv_order):
        cx = x0 + bw2 / 2 + i * (bw2 + gap2)
        t = tgt[lv]
        a = ach[lv]
        yb_hi = _y(t + 20, y0, y1b, vhi, vlo)
        yb_lo = _y(t - 20, y0, y1b, vhi, vlo)
        svg_b.append(f'<rect x="{cx - bw2/2:.2f}" y="{yb_hi:.2f}" width="{bw2}" height="{yb_lo - yb_hi:.2f}" '
                     f'fill="#fde8e8" rx="1.6"/>')
        yy = _y(t, y0, y1b, vhi, vlo)
        svg_b.append(f'<line x1="{cx - bw2/2 + 1.9:.2f}" y1="{yy:.2f}" x2="{cx + bw2/2 - 1.9:.2f}" '
                     f'y2="{yy:.2f}" stroke="#374151" stroke-width="1.5" stroke-dasharray="4 2.8"/>')
        top = _y(a, y0, y1b, vhi, vlo)
        svg_b.append(f'<rect x="{cx - bw2/2:.2f}" y="{top:.2f}" width="{bw2}" height="{y1b - top:.2f}" '
                     f'fill="#bbf7d0" stroke="#15803d" stroke-width="1.1" rx="1.9"/>')
        svg_b.append(f'<text x="{cx:.2f}" y="{top - 2.5:.2f}" font-size="7" font-weight="bold" '
                     f'text-anchor="middle" fill="#14532d">{a:.1f}</text>')
        svg_b.append(f'<text x="{cx:.2f}" y="{yy - 1.8:.2f}" font-size="6" text-anchor="middle" '
                     f'fill="#374151" font-style="italic">target {t}</text>')
        svg_b.append(f'<text x="{cx:.2f}" y="{y1b + 7.6:.2f}" font-size="6.3" text-anchor="middle" '
                     f'fill="#374151">{lab[lv]}</text>')
        if not (t - 20 <= a <= t + 20):
            inband = False
    svg_b.append(f'<text x="{PX/2}" y="12.8" font-size="8.6" font-weight="bold" '
                 f'text-anchor="middle" fill="#111827">(b) Difficulty control vs targets (±20 band)</text>')
    svg_b.append(f'<text x="6" y="{y1b + 6.4}" font-size="6.2" fill="#4b5563">Flesch reading ease</text>')
    svg_b.append(f'<text x="{PX/2}" y="{y0 + 5.8}" font-size="6.3" text-anchor="middle" fill="#15803d" '
                 f'font-weight="bold">{"all three levels inside their ±20 band" if inband else "target bands (±20)"}</text>')
    svg_b.append("</svg>")

    W2, H2 = 178, 62
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><style>", CSS, "</style></head><body>"]
    html.append(f"<div class='fig' style='width:{W2}mm;height:{H2}mm'>")
    html.append(f"<div class='fig2-panel' style='left:2mm;top:2mm;width:{PXM + 4}mm;height:{PYM + 4}mm'>"
                f"<div style='padding:1.5mm'>{''.join(svg_a)}</div></div>")
    html.append(f"<div class='fig2-panel' style='left:{PXM + 10}mm;top:2mm;width:{PXM + 4}mm;height:{PYM + 4}mm'>"
                f"<div style='padding:1.5mm'>{''.join(svg_b)}</div></div>")
    html.append("</div></body></html>")
    return ("".join(html), W2, H2)


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def render(fig, base):
    from playwright.sync_api import sync_playwright
    html_str, w_mm, h_mm = fig
    tmp_html = os.path.join(FIGDIR, base + ".html")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html_str)
    png = os.path.join(FIGDIR, base + ".png")
    pdf = os.path.join(FIGDIR, base + ".pdf")
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        w_css, h_css = int(w_mm / 25.4 * 96), int(h_mm / 25.4 * 96)
        ctx = b.new_context(device_scale_factor=4, viewport={"width": w_css, "height": h_css})
        pg = ctx.new_page()
        pg.goto("file:///" + tmp_html.replace("\\", "/"), wait_until="load")
        pg.wait_for_timeout(300)
        over = pg.evaluate(
            """() => Array.from(document.querySelectorAll('.box,.lane,.legend')).map((el, i) => {
                 const cs = getComputedStyle(el);
                 return { tag: el.className.split(' ')[0], i,
                          h: el.clientHeight, sh: el.scrollHeight,
                          w: el.clientWidth, sw: el.scrollWidth };
               }).filter(o => o.sh > o.h + 0.5 || o.sw > o.w + 0.5)"""
        )
        if over:
            print(f"  [overflow] {base}: {over}")
        pg.screenshot(path=png, full_page=True)
        pg.pdf(path=pdf, width=f"{w_mm}mm", height=f"{h_mm}mm",
               print_background=True, margin={"top": "0", "right": "0",
                                              "bottom": "0", "left": "0"})
        b.close()
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main():
    render(figure1_html(), "framework_overview")
    render(figure2_html(), "results_overview")
    return 0


if __name__ == "__main__":
    sys.exit(main())
