# -*- coding: utf-8 -*-
"""Figure 1: framework overview for the legal-English RLVR paper.

Three stacked lanes:
  (a) Training   -- legal sources -> corpus -> split -> SFT (QLoRA) -> RLVR/GRPO,
                    with the 5-dim verifiable reward stack detailed below GRPO.
  (b) Inference  -- legal passage + target level -> RAG (MiniLM, 861-passage KB)
                    -> Qwen3.5-4B (GRPO v2) + multi-agent loop -> simplified passage.
  (c) Evaluation -- automatic verifiable rewards + readability/replication (bottom-left)
                    and independent cross-family judge (bottom-right), both fed by the
                    output passage via elbow arrows.

Every figure value matches manuscript.md Methods 3.1-3.6 (verified 2026-08-16).
Rendered at 300 dpi PNG + PDF into D:\\周老师\\paper\\figures\\.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

FIGDIR = r"D:\周老师\paper\figures"
os.makedirs(FIGDIR, exist_ok=True)

# ---- palette ---------------------------------------------------------------
C_DATA   = "#E8F0FE"   # light blue  - data
C_TRAIN  = "#E6F4EA"   # light green - training
C_MODEL  = "#FEF7E0"   # light amber - inference / model
C_EVAL   = "#FCE8E6"   # light red   - evaluation
C_DETAIL = "#F1F3F4"   # grey        - detail panel
C_EDGE   = "#5F6368"
C_TEXT   = "#202124"
C_ARROW  = "#5F6368"

def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False, tc=C_TEXT, ls="solid", lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       fc=fc, ec=C_EDGE, lw=lw, linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.3, color=C_ARROW, cs=None):
    kw = dict(arrowstyle=style, mutation_scale=13, lw=lw, color=color,
              zorder=1, shrinkA=2, shrinkB=2)
    if cs:
        kw["connectionstyle"] = cs
    a = FancyArrowPatch((x1, y1), (x2, y2), **kw)
    ax.add_patch(a)

def label(ax, x, y, text, fs=8, color=C_TEXT, style="italic", ha="center", va="center"):
    ax.text(x, y, text, fontsize=fs, color=color, style=style, ha=ha, va=va, zorder=3)

fig, ax = plt.subplots(figsize=(11.0, 6.4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")

# ------------------------------------------------------------------ lane titles
ax.text(2, 60.0, "(a) Training", fontsize=10, fontweight="bold", color=C_TEXT)
ax.text(2, 39.0, "(b) Inference", fontsize=10, fontweight="bold", color=C_TEXT)
ax.text(2, 20.0, "(c) Evaluation", fontsize=10, fontweight="bold", color=C_TEXT)

# --------------------------------------------------------------- (a) training ----
YH = 53.5
box(ax, 2, YH, 15, 5.5, "Legal sources\nEUR-Lex / Wex / Oyez", C_DATA, fs=8.5)
box(ax, 20, YH, 14, 5.5, "Corpus\n968 snippets + 49 terms", C_DATA, fs=8.5)
box(ax, 37, YH, 13, 5.5, "Split (id-hash)\ntrain/val/test 80/10/10", C_DATA, fs=8.5)
box(ax, 53, YH, 15, 5.5, "SFT (QLoRA)\nr=16, \u03b1=32, 7 modules, 21.23M", C_TRAIN, fs=8.5)
box(ax, 71, YH, 15, 5.5, "RLVR / GRPO\n5-dim verifiable reward", C_TRAIN, fs=8.5)

# reward stack detail panel directly under GRPO
box(ax, 71, 44, 15, 6.5,
    "Reward stack $W$\n"
    "format .15 \u00b7 difficulty .20\n"
    "terminology .15 \u00b7 anti-copy .25\n"
    "faithfulness .25  (v2 diff .30)",
    C_DETAIL, fs=7)
label(ax, 78.5, 42.4, "verifiable text signals", fs=6.5)

# training flow
for (x1, x2) in [(17, 20), (34, 37), (50, 53), (68, 71)]:
    arrow(ax, x1, YH + 2.75, x2, YH + 2.75)
arrow(ax, 78.5, YH, 78.5, 50.6, lw=1.0)

label(ax, 9.5, 51.0, "public legal text\n(CC BY / public domain)", fs=6.5)
label(ax, 27, 51.0, "teacher simplifications", fs=6.5)

# -------------------------------------------------------------- (b) inference ----
YB = 31.0
box(ax, 2, YB, 15, 6, "Legal passage\n+ target level", C_MODEL, fs=8.5)
box(ax, 20, YB, 16, 6, "RAG retrieval\nMiniLM-L6-v2 \u00b7 861-passage KB", C_MODEL, fs=8.5)
box(ax, 39, YB, 21, 6,
    "SLM Qwen3.5-4B (GRPO v2)\n+ multi-agent loop\nplanner \u00b7 writer \u00b7 editor \u00b7 reviewer",
    C_MODEL, fs=8, bold=True)
box(ax, 63, YB, 15, 6, "Simplified passage\n(target FRE band \u00b120)", C_EVAL, fs=8)

for (x1, x2) in [(17, 20), (36, 39), (60, 63)]:
    arrow(ax, x1, YB + 3, x2, YB + 3)
label(ax, 28, 29.2, "retrieved evidence", fs=7)
label(ax, 49.5, 29.2, "revise toward target band", fs=7)

# output -> evaluation lane (elbow arrows)
arrow(ax, 68, YB, 25, 17.5, lw=1.0, cs="angle,angleA=90,angleB=0")  # elbow -> auto panel (Box 1)
arrow(ax, 75, YB, 75, 17.5, lw=1.0)                                   # straight -> judge panel (Box 2)
# --------------------------------------------------------------- (c) evaluation ----
YE = 8.0
box(ax, 2, YE, 46, 9.5,
    "Automatic verifiable rewards\n"
    "tot \u00b7 diff \u00b7 term \u00b7 copy \u00b7 faith \u00b7 contra (NLI)\n"
    "+ FRE / SARI / full-107 replication",
    C_EVAL, fs=7.5)
box(ax, 51, YE, 47, 9.5,
    "Independent cross-family judge (Qwen3-4B)\n"
    "4 dims 1-5 + confidence  \u00b7  ECE  \u00b7  self-consistency\n"
    "reported mean\u00b1std over 4 seeds",
    C_EVAL, fs=7.5)

plt.tight_layout(pad=0.4)
png = os.path.join(FIGDIR, "framework_overview.png")
pdf = os.path.join(FIGDIR, "framework_overview.pdf")
plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(pdf, bbox_inches="tight", facecolor="white")
print("wrote", png)
print("wrote", pdf)
