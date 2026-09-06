"""Poster variant of the system figure: the evolutionary loop plus the gate
and the audit, drawn with few boxes and large type so it reads at 1.5 m
when printed one A0 column (~250 mm) wide.

    uv run python evorank/paper/figures/make_flowchart_poster.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parent

INK = "#222222"
EDGE = "#555555"
FILL = "#F8F8F8"
ACCENT = "#1F4E79"
ACCENT_FILL = "#EAF1F8"
GOLD = "#8A6D1A"
GOLD_FILL = "#FDF8EA"

fig, ax = plt.subplots(figsize=(10.5, 12.5), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

TS, BS = 19, 15.5


def box(x, y, w, h, title, body, edge=EDGE, fill=FILL, tcol=INK, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.014",
                 linewidth=lw, edgecolor=edge, facecolor=fill))
    if body:
        ax.text(x + w / 2, y + h - 0.033, title, ha="center", va="center",
                fontsize=TS, fontweight="bold", color=tcol)
        ax.text(x + w / 2, y + (h - 0.058) / 2, body, ha="center", va="center",
                fontsize=BS, color=INK, linespacing=1.55)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=TS, fontweight="bold", color=tcol)


def arrow(x1, y1, x2, y2, color=INK, lw=3.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=26, linewidth=lw, color=color,
                 shrinkA=4, shrinkB=4, connectionstyle=f"arc3,rad={rad}"))


# 1. search space
box(0.20, 0.905, 0.60, 0.080, "Search space: ranking pipelines", None)
ax.text(0.50, 0.883, "features  +  models  +  losses  +  ensembles",
        ha="center", va="center", fontsize=BS, color=INK)
arrow(0.50, 0.868, 0.50, 0.836)

# 2. gate
box(0.09, 0.700, 0.82, 0.125, "HEADROOM GATE",
    "before any LLM spend: a hand-built candidate\n"
    "must beat the noise floor by 2.5x",
    edge=GOLD, fill=GOLD_FILL, tcol=GOLD, lw=2.6)
arrow(0.50, 0.700, 0.50, 0.655, color=GOLD)
ax.text(0.545, 0.678, "go", fontsize=BS + 1, style="italic", color=GOLD,
        ha="left", va="center", fontweight="bold")
arrow(0.91, 0.762, 0.985, 0.762, color=GOLD, lw=2.2)
ax.text(0.998, 0.712, "no-go: stop,\nkeep the budget", ha="right", va="top",
        fontsize=BS - 1.5, style="italic", color=GOLD)

# 3. the loop: ring of 4 boxes with clear gaps
yT, yB = 0.505, 0.315
h = 0.130
box(0.030, yT, 0.435, h, "1. LLM proposes an edit",
    "one call per iteration;\nonly marked code blocks change")
box(0.535, yT, 0.435, h, "2. Guarded evaluation",
    "small fitness fold; leakage-safe\nstats, whitelists, budgets")
box(0.535, yB, 0.435, h, "3. Scored feedback",
    "noise-gated deltas; broken\ncandidates get an explanation")
box(0.030, yB, 0.435, h, "4. Program archive",
    "Pareto set over relevance,\nconversion, revenue")
arrow(0.465, yT + h / 2, 0.535, yT + h / 2)
arrow(0.7525, yT, 0.7525, yB + h)
arrow(0.535, yB + h / 2, 0.465, yB + h / 2)
arrow(0.2475, yB + h, 0.2475, yT)
ax.text(0.50, (yT + yB + h) / 2, "x 50 iterations (~$10)", ha="center",
        va="center", fontsize=TS - 3, fontweight="bold", color=ACCENT,
        bbox=dict(facecolor="white", edgecolor="none", pad=2))

# 4. transfer audit
arrow(0.2475, yB, 0.2475, 0.260, color=ACCENT)
ax.text(0.265, 0.288, "selected programs", fontsize=BS - 1.5, style="italic",
        color=ACCENT, ha="left", va="center")
box(0.09, 0.130, 0.82, 0.125, "TRANSFER AUDIT",
    "after the search: re-score every selected program\n"
    "on 60k held-out queries the loop never saw",
    edge=ACCENT, fill=ACCENT_FILL, tcol=ACCENT, lw=2.6)

# 5. verdict
arrow(0.50, 0.130, 0.50, 0.085, color=ACCENT)
ax.text(0.50, 0.050, "only gains that survive the audit\ncount as discoveries",
        ha="center", va="center", fontsize=TS, fontweight="bold", color=ACCENT)

fig.savefig(OUT / "flowchart_poster.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "flowchart_poster.png", bbox_inches="tight", facecolor="white")
print("wrote flowchart_poster.pdf/.png")
