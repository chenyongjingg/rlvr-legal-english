# -*- coding: utf-8 -*-
"""Figure 2: results overview for the legal-English RLVR paper.

Panel (a) System progression: total verifiable reward across the six
configurations G1-G6 (Table 1 honest values). G5 (+SFT, no GRPO) collapses;
G6 (corrected framework, = L3_max3) restores difficulty control -> the RLVR
story (RQ2/RQ3). The reward-free framework peaks at G3/G4 (~0.87).

Panel (b) Difficulty control: achieved Flesch reading ease per target level
(ADV pure-generator protocol) vs the 63/45/26 targets and the +/-20 band
(Sec 4.3).

Values come from honest_table.json (the commentary-decontaminated re-scoring;
falls back to the pre-honest literals only if that file is absent), so the
figure is always consistent with the manuscript tables.
Rendered at 300 dpi PNG + PDF into the repository's figures/ directory.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
FIGDIR = os.path.join(_REPO, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# official numbers: honest_table.json if present (Table 1 / Sec 4.3 honest)
_ht = os.path.join(_REPO, "results", "honest_table.json")
_h = json.load(open(_ht, encoding="utf-8")) if os.path.isfile(_ht) else None
def _tot(tag):
    if _h and tag in _h.get("tags", {}):
        v = _h["tags"][tag]["all"].get("tot")
        if v is not None:
            return v
    return _LIT_TOTS.get(tag)
def _tot_any(*tags):
    # first available honest value across alternative tags (e.g. G6 == L3_max3)
    for t in tags:
        v = _tot(t)
        if v is not None:
            return v
    return None
def _fre(tag, lv):
    if _h and tag in _h.get("tags", {}):
        v = _h["tags"][tag]["levels"].get(lv, {}).get("fre_mean")
        if v is not None:
            return v
    return _LIT_ACH.get(lv)

_LIT_TOTS = {"G1": 0.856, "G2": 0.866, "G3": 0.874, "G4": 0.870, "G5": 0.759, "G6": 0.800}
_LIT_ACH = {"beginner": 64.7, "intermediate": 25.9, "advanced": 10.8}
# Table 1 reports the CORRECTED framework as G6 (decontamination-aware generator),
# which corresponds to loop-depth tag L3_max3 in the honest_table; the raw "G6"
# tag is the original framework's honest re-score (0.705) and is NOT the G6 bar.
TOTS = {g: _tot(g) for g in ("G1", "G2", "G3", "G4", "G5")}
TOTS["G6"] = _tot_any("L3_max3", "G6") or _LIT_TOTS["G6"]
COMPONENT = {"G1": "base", "G2": "+RAG", "G3": "+agents", "G4": "+GRPO v1",
             "G5": "+SFT", "G6": "+GRPO v2"}
LEVELS = ["beginner", "intermediate", "advanced"]
ACHIEVED = {lv: _fre("ADV", lv) for lv in LEVELS}
TARGET = {"beginner": 63, "intermediate": 45, "advanced": 26}
BAND = 20

C_GREY = "#B0BEC5"
C_G5   = "#E57373"   # red  - the SFT collapse
C_G6   = "#81C784"   # green- best system
C_TGT  = "#5F6368"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=300)

# ------------------------------------------------ (a) progression ------------
sys_order = ["G1", "G2", "G3", "G4", "G5", "G6"]
x = np.arange(len(sys_order))
colors = [C_G5 if s == "G5" else (C_G6 if s == "G6" else C_GREY) for s in sys_order]
bars = ax1.bar(x, [TOTS[s] for s in sys_order], 0.62, color=colors,
               edgecolor="#5F6368", linewidth=0.8, zorder=3)
for xi, s in zip(x, sys_order):
    v = TOTS[s]
    ax1.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax1.text(xi, 0.747, COMPONENT[s], ha="center", va="top", fontsize=7.5,
             style="italic", color="#37474F")
ax1.set_ylim(0.73, 0.95)
ax1.set_yticks([0.76, 0.80, 0.84, 0.88, 0.92])
ax1.set_ylabel("Total verifiable reward", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(sys_order, fontsize=9)
ax1.set_title("(a) System progression (G1–G6)", fontsize=10, fontweight="bold", pad=10)
ax1.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
ax1.set_axisbelow(True)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
# The G5 collapse / G6 restore message is carried by the bar colour (red/green)
# and stated verbatim in the Figure caption + Sec 4.2 text; in-plot call-out
# boxes here overlapped the value labels, so they are intentionally omitted.

# ------------------------------------------------ (b) difficulty -------------
x2 = np.arange(len(LEVELS))
ach = [ACHIEVED[l] for l in LEVELS]
tgt = [TARGET[l] for l in LEVELS]
bars2 = ax2.bar(x2, ach, 0.5, color="#FEF7E0", edgecolor="#5F6368",
                linewidth=0.8, zorder=3)
# +/-20 band shading
for i, lv in enumerate(LEVELS):
    t = TARGET[lv]
    ax2.axhspan(t - BAND, t + BAND, xmin=i/3, xmax=(i+1)/3,
                color="#FCE8E6", alpha=0.55, zorder=0)
    ax2.hlines(t, i - 0.25, i + 0.25, colors=C_TGT, linestyles="--",
               linewidths=1.4, zorder=4)
    # per-bar "target NN" floating labels were redundant with the x-tick
    # "(target)" annotation and overlapped the value labels -> omitted.
for xi, (a, lv) in enumerate(zip(ach, LEVELS)):
    ax2.text(xi, a + 2.5, f"{a:.1f}", ha="center", fontsize=8.5, fontweight="bold")
ax2.set_ylim(0, 100)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_ylabel("Flesch reading ease", fontsize=9)
ax2.set_xticks(x2)
ax2.set_xticklabels([f"{l}\n({TARGET[l]})" for l in LEVELS], fontsize=9)
ax2.set_title("(b) Difficulty control vs targets (±20 band)", fontsize=10,
              fontweight="bold", pad=10)
ax2.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
ax2.set_axisbelow(True)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
# "all levels inside band" annotation removed: it collided with the target
# markers / value labels; the point is stated in the Figure caption.

plt.tight_layout(pad=1.0)
png = os.path.join(FIGDIR, "results_overview.png")
pdf = os.path.join(FIGDIR, "results_overview.pdf")
plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(pdf, bbox_inches="tight", facecolor="white")
print("wrote", png)
print("wrote", pdf)

# ---- self-check: report any remaining overlapping text boxes (dev aid) ----
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_seen = set()
_items = []
for ax in (ax1, ax2):
    for t in (list(ax.texts) + list(ax.get_xticklabels())
              + list(ax.get_yticklabels())):
        if id(t) in _seen or not isinstance(t, plt.Text) or not t.get_text().strip():
            continue
        _seen.add(id(t))
        bb = t.get_window_extent(renderer=_renderer)
        if bb.width > 0 and bb.height > 0:
            _items.append((bb, "A" if ax is ax1 else "B", t.get_text().replace("\n", "|")))
_over = 0
for i in range(len(_items)):
    for j in range(i + 1, len(_items)):
        a, b = _items[i][0], _items[j][0]
        ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
        if ix > 2 and iy > 2:
            _over += 1
            print("OVERLAP [%s/%s] %r <-> %r (ix=%.0f iy=%.0f)"
                  % (_items[i][1], _items[j][1], _items[i][2], _items[j][2], ix, iy))
print("text-overlap pairs: %d" % _over)
