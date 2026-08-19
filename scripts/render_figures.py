# -*- coding: utf-8 -*-
"""Figure 1 + Figure 2 (unified HTML/CSS/SVG style) rendered via Playwright+Edge.

Replaces the old matplotlib schematics with journal-quality layouts:
  * Figure 1 (framework_overview): three-lane architecture diagram (training /
    inference / evaluation) built from styled HTML boxes + an SVG arrow layer,
    all content matching manuscript.md Sec 3.1-3.6 (honest numbers, 2026-08-17).
  * Figure 2 (results_overview): two data panels generated as inline SVG from
    honest_table.json (G1-G6 total reward; three-level Flesch vs targets +-20).

Outputs (300+ dpi PNG + vector PDF) into D:/周老师/paper/figures/.
Usage: python scripts/render_figures.py
"""
import io
import json
import os
import sys

FIGDIR = r"D:\周老师\paper\figures"
os.makedirs(FIGDIR, exist_ok=True)
HONEST = r"C:\Users\33378\aigc_wechat\results\legal\honest_table.json"

MM = 96.0 / 25.4  # CSS px per mm (for the 1:1 mm viewBox figures below, unused)

# --------------------------------------------------------------------------
# shared style
# --------------------------------------------------------------------------
CSS = """
html,body{margin:0;padding:0;background:#ffffff}
body{font-family:Arial,'Helvetica Neue',Helvetica,sans-serif;color:#1f2937}
.fig{position:relative;width:178mm;background:#ffffff}
.lane-title{position:absolute;font-size:10pt;font-weight:bold;color:#111827;letter-spacing:.2px}
.box{position:absolute;border-radius:2.2mm;border:1px solid;display:flex;
     flex-direction:column;justify-content:center;align-items:center;
     text-align:center;padding:1mm 1.5mm;box-sizing:border-box;overflow:hidden}
.bt{font-size:8.0pt;font-weight:bold;color:#111827;line-height:1.22}
.bs{font-size:7.0pt;color:#4b5563;line-height:1.28;margin-top:.5mm}
.c-data {border-color:#2b6cb0;background:#ebf4ff}
.c-train{border-color:#2f855a;background:#effaf1}
.c-model{border-color:#b7791f;background:#fffaeb}
.c-eval {border-color:#c53030;background:#fff5f5}
.c-detail{border-color:#a0aec0;background:#f7fafc}
.emph{border-width:1.4px;box-shadow:0 0 0 .5px rgba(0,0,0,.08)}
.ml{position:absolute;font-size:6.4pt;font-style:italic;color:#6b7280;text-align:center;line-height:1.25}
svg.arrows{position:absolute;left:0;top:0;pointer-events:none}
.fig2-panel{position:absolute;border:1px solid #d1d5db;border-radius:2.2mm;background:#ffffff}
.p2-title{font-size:9.5pt;font-weight:bold;color:#111827}
"""

# --------------------------------------------------------------------------
# FIGURE 1 — framework overview
# --------------------------------------------------------------------------
def _arrow(x1, y1, x2, y2, w=0.32, dash=None, color="#4b5563", elbow=None):
    """One <line> (or elbow <path>) with an arrowhead marker, coordinates in mm."""
    d = "none" if dash is None else dash
    if elbow:
        (mx, my, bx, by) = elbow  # vertical then horizontal path via (mx,my)
        p = f"M {x1} {y1} L {mx} {my} L {bx} {by}"
        return (f'<path d="{p}" stroke="{color}" stroke-width="{w}" fill="none" '
                f'stroke-dasharray="{d}" marker-end="url(#ah)"/>')
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}" stroke-dasharray="{d}" marker-end="url(#ah)"/>')


def figure1_html():
    W, H = 178, 104  # mm canvas
    boxes = []   # (css_class, x, y, w, h, title, [sub_line, ...])
    labels = []  # (x, y, text)
    arrows = []

    # ---- lane (a) training -------------------------------------------------
    boxes.append(("c-data", 4, 6, 26, 18, "Legal sources", ["EUR-Lex / Wex / Oyez"]))
    boxes.append(("c-data", 33, 6, 26, 18, "Corpus", ["968 snippets + 49 terms"]))
    boxes.append(("c-data", 62, 6, 26, 18, "Split (id-hash)", ["train / val / test", "80 / 10 / 10"]))
    boxes.append(("c-train", 91, 6, 26, 18, "SFT (QLoRA)", ["r = 16, α = 32, 7 modules", "21.23 M params"]))
    boxes.append(("c-train emph", 120, 6, 26, 18, "RLVR / GRPO", ["5-dim verifiable reward"]))
    for x in (30, 59, 88, 117):
        arrows.append(_arrow(x, 15, x + 3, 15))
    # reward-stack detail under RLVR/GRPO
    boxes.append(("c-detail", 120, 29, 50, 17, "Reward stack W",
                  ["format .15 · difficulty .20",
                   "terminology .15 · anti-copy .25",
                   "faithfulness .25  (v2 diff .30)"]))
    arrows.append(_arrow(133, 24, 133, 28.6, color="#4b5563"))
    labels.append((16.5, 26.4, "public legal text (CC BY / public domain)"))
    labels.append((46, 26.4, "teacher simplifications"))

    # ---- lane (b) inference ------------------------------------------------
    boxes.append(("c-model", 4, 52, 28, 18, "Legal passage", ["+ target level"]))
    boxes.append(("c-model", 35, 52, 30, 18, "RAG retrieval", ["MiniLM-L6-v2 index", "861-passage KB"]))
    boxes.append(("c-model emph", 68, 52, 44, 18, "SLM Qwen3.5-4B (GRPO v2)",
                  ["multi-agent loop: planner · writer", "editor · reviewer"]))
    boxes.append(("c-eval", 115, 52, 28, 18, "Simplified passage", ["target FRE band ±20"]))
    for x in (32, 65, 112):
        arrows.append(_arrow(x, 61, x + 3, 61))
    labels.append((50, 72.2, "retrieved evidence"))
    labels.append((90, 72.2, "revise toward target band"))

    # ---- lane (c) evaluation -----------------------------------------------
    boxes.append(("c-eval", 4, 80, 86, 20, "Automatic verifiable rewards",
                  ["tot · diff · term · copy · faith · contra (NLI)",
                   "+ FRE / SARI / full-107 replication"]))
    boxes.append(("c-eval", 94, 80, 80, 20, "Independent cross-family judge (Qwen3-4B)",
                  ["4 dims 1–5 + confidence · ECE · self-consistency",
                   "reported mean ± std over 4 seeds"]))
    # arrows from simplified passage down into the two evaluation panels
    arrows.append(_arrow(119, 70, 92, 79.6))
    arrows.append(_arrow(139, 70, 98, 79.6))

    # ---- assemble ----------------------------------------------------------
    lane_titles = [("(a) Training", 2.0), ("(b) Inference", 47), ("(c) Evaluation", 75)]
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><style>", CSS, "</style></head><body>"]
    html.append(f"<div class='fig' style='width:{W}mm;height:{H}mm'>")
    for t, y in lane_titles:
        html.append(f"<div class='lane-title' style='left:4mm;top:{y}mm'>{t}</div>")
    for cls, x, y, w, h, t, subs in boxes:
        title = f"<div class='bt'>{t}</div>" if t else ""
        body = "".join(f"<div class='bs'>{s}</div>" for s in subs)
        html.append(f"<div class='box {cls}' style='left:{x}mm;top:{y}mm;width:{w}mm;height:{h}mm'>"
                    f"{title}{body}</div>")
    for x, y, t in labels:
        html.append(f"<div class='ml' style='left:{x - 14}mm;top:{y}mm;width:28mm'>{t}</div>")
    html.append(f"<svg class='arrows' width='{W}mm' height='{H}mm' viewBox='0 0 {W} {H}'>"
                f"<defs><marker id='ah' viewBox='0 0 10 10' refX='8.5' refY='5' "
                f"markerWidth='1.7' markerHeight='1.7' orient='auto-start-reverse'>"
                f"<path d='M 0 0 L 10 5 L 0 10 z' fill='#4b5563'/></marker></defs>"
                + "".join(arrows) + "</svg>")
    html.append("</div></body></html>")
    return ("".join(html), W, H)


# --------------------------------------------------------------------------
# FIGURE 2 — results (inline SVG panels)
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

    S = 72.0 / 25.4      # pt per mm — all panel coordinates below are in pt
    PXM, PYM = 84, 54    # panel size in mm
    PX, PY = PXM * S, PYM * S
    x0, x1 = 10 * S, 81.5 * S      # plot x range (pt)
    yt, yb = 9 * S, 44 * S         # plot y range (pt)
    vmin, vmax = 0.73, 0.92
    svg_a = []
    svg_a.append(f'<svg width="{PXM}mm" height="{PYM}mm" viewBox="0 0 {PX:.1f} {PY:.1f}" '
                 f'font-family="Arial,Helvetica,sans-serif">')
    # gridlines + y ticks
    for v in (0.76, 0.80, 0.84, 0.88, 0.92):
        yy = _y(v, yt, yb, vmax, vmin)
        svg_a.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" '
                     f'stroke="#e5e7eb" stroke-width="0.5"/>')
        svg_a.append(f'<text x="{x0 - 3.5}" y="{yy + 1.8:.2f}" font-size="6" '
                     f'text-anchor="end" fill="#6b7280">{v:.2f}</text>')
    svg_a.append(f'<line x1="{x0}" y1="{yt}" x2="{x0}" y2="{yb}" stroke="#9ca3af" stroke-width="0.7"/>')
    svg_a.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="#9ca3af" stroke-width="0.7"/>')
    # bars
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
            fill, edge = "#e57373", "#c53030"
        elif g == "G6":
            fill, edge = "#81c784", "#2f855a"
        else:
            fill, edge = "#cbd5e1", "#64748b"
        svg_a.append(f'<rect x="{cx - bw/2:.2f}" y="{top:.2f}" width="{bw}" '
                     f'height="{yb - top:.2f}" fill="{fill}" stroke="{edge}" stroke-width="1" rx="1.7"/>')
        svg_a.append(f'<text x="{cx:.2f}" y="{top - 2.6:.2f}" font-size="6.6" font-weight="bold" '
                     f'text-anchor="middle" fill="#111827">{v:.3f}</text>')
        svg_a.append(f'<text x="{cx:.2f}" y="{yb + 7.4:.2f}" font-size="6.2" '
                     f'text-anchor="middle" fill="#374151" font-style="italic">{comp[i]}</text>')
    # compact notes under G5 / G6 (full-width centered so they never clip)
    svg_a.append(f'<text x="{PX/2}" y="{yb + 17.0}" font-size="6" text-anchor="middle" '
                 f'fill="#c53030">SFT without GRPO collapses difficulty control</text>')
    svg_a.append(f'<text x="{PX/2}" y="{yb + 24.0}" font-size="6" text-anchor="middle" '
                 f'fill="#2f855a">RLVR restores (corrected framework)</text>')
    svg_a.append(f'<text x="{PX/2}" y="12.8" font-size="8.5" font-weight="bold" '
                 f'text-anchor="middle" fill="#111827">(a) System progression (G1–G6)</text>')
    svg_a.append(f'<text x="6" y="{yb + 6.2}" font-size="6.2" fill="#4b5563">Total verifiable reward</text>')
    svg_a.append("</svg>")

    # ---------- panel (b): difficulty vs targets ------------------------------
    svg_b = []
    svg_b.append(f'<svg width="{PXM}mm" height="{PYM}mm" viewBox="0 0 {PX:.1f} {PY:.1f}" '
                 f'font-family="Arial,Helvetica,sans-serif">')
    y0, y1b, vlo, vhi = 9 * S, 44 * S, 0, 100
    for v in (0, 20, 40, 60, 80, 100):
        yy = _y(v, y0, y1b, vhi, vlo)
        svg_b.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" '
                     f'stroke="#e5e7eb" stroke-width="0.5"/>')
        svg_b.append(f'<text x="{x0 - 3.5}" y="{yy + 1.8:.2f}" font-size="6" '
                     f'text-anchor="end" fill="#6b7280">{v}</text>')
    svg_b.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1b}" stroke="#9ca3af" stroke-width="0.7"/>')
    svg_b.append(f'<line x1="{x0}" y1="{y1b}" x2="{x1}" y2="{y1b}" stroke="#9ca3af" stroke-width="0.7"/>')
    lv_order = ["beginner", "intermediate", "advanced"]
    lab = {"beginner": "beginner (63)", "intermediate": "intermediate (45)", "advanced": "advanced (26)"}
    bw2 = 9.5 * S
    gap2 = (x1 - x0 - 3 * bw2) / 2
    for i, lv in enumerate(lv_order):
        cx = x0 + bw2 / 2 + i * (bw2 + gap2)
        t = tgt[lv]
        a = ach[lv]
        # +-20 band
        yb_hi = _y(t + 20, y0, y1b, vhi, vlo)
        yb_lo = _y(t - 20, y0, y1b, vhi, vlo)
        svg_b.append(f'<rect x="{cx - bw2/2:.2f}" y="{yb_hi:.2f}" width="{bw2}" '
                     f'height="{yb_lo - yb_hi:.2f}" fill="#fde8e8" rx="1.4"/>')
        # dashed target
        yy = _y(t, y0, y1b, vhi, vlo)
        svg_b.append(f'<line x1="{cx - bw2/2 + 1.7:.2f}" y1="{yy:.2f}" x2="{cx + bw2/2 - 1.7:.2f}" '
                     f'y2="{yy:.2f}" stroke="#374151" stroke-width="1.4" stroke-dasharray="4 2.8"/>')
        # achieved bar
        top = _y(a, y0, y1b, vhi, vlo)
        svg_b.append(f'<rect x="{cx - bw2/2:.2f}" y="{top:.2f}" width="{bw2}" '
                     f'height="{y1b - top:.2f}" fill="#fef7e0" stroke="#b7791f" stroke-width="1" rx="1.7"/>')
        svg_b.append(f'<text x="{cx:.2f}" y="{top - 2.3:.2f}" font-size="6.8" font-weight="bold" '
                     f'text-anchor="middle" fill="#111827">{a:.1f}</text>')
        svg_b.append(f'<text x="{cx:.2f}" y="{yy - 1.7:.2f}" font-size="6" '
                     f'text-anchor="middle" fill="#374151" font-style="italic">target {t}</text>')
        svg_b.append(f'<text x="{cx:.2f}" y="{y1b + 7.4:.2f}" font-size="6.2" '
                     f'text-anchor="middle" fill="#374151">{lab[lv]}</text>')
    svg_b.append(f'<text x="{PX/2}" y="12.8" font-size="8.5" font-weight="bold" '
                 f'text-anchor="middle" fill="#111827">(b) Difficulty control vs targets (±20 band)</text>')
    svg_b.append(f'<text x="6" y="{y1b + 6.2}" font-size="6.2" fill="#4b5563">Flesch reading ease</text>')
    svg_b.append(f'<text x="{PX/2}" y="{y0 + 5.7}" font-size="6" text-anchor="middle" '
                 f'fill="#2f855a">all three levels inside their ±20 band</text>')
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
    html_str, w_mm, h_mm = fig   # mm canvas size travels with the HTML
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
        pg.wait_for_timeout(250)
        # overflow self-check: report any box/panel whose content is clipped
        over = pg.evaluate(
            """() => Array.from(document.querySelectorAll('.box')).map((el, i) => {
                 const cs = getComputedStyle(el);
                 return { i, text: el.innerText.slice(0, 28),
                          h: el.clientHeight, sh: el.scrollHeight,
                          w: el.clientWidth, sw: el.scrollWidth };
               }).filter(o => o.sh > o.h + 0.5 || o.sw > o.w + 0.5)"""
        )
        if over:
            print(f"  [overflow] {base}: {over}")
        # PNG at 4x CSS scale (device pixels), full content
        pg.screenshot(path=png, full_page=True)
        # vector PDF at exact mm size
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
