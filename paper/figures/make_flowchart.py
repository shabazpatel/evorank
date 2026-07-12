"""Render the EvoRank system figure for the paper (Figure 1).

Academic style: restrained monochrome boxes, one accent color reserved for
the honesty harness (the paper's contribution), three labeled lanes, and the
full pipeline including the headroom gate and the transfer audit.

    uv run python make_flowchart.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#222222"
EDGE = "#555555"
FILL = "#f8f8f8"
ACCENT = "#1f4e79"       # honesty harness
ACCENT_FILL = "#eaf1f8"
GUIDE_EDGE = "#8a6d1a"   # knowledge base (dashed, optional)
GUIDE_FILL = "#fdf8ea"
LANE = "#666666"

fig, ax = plt.subplots(figsize=(13.2, 8.0), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, title, lines, edge=EDGE, fill=FILL, dashed=False, title_color=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.1, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 3)) if dashed else "solid"))
    ax.text(x + w / 2, y + h - 0.030, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=title_color)
    body = "\n".join(lines)
    ax.text(x + w / 2, y + (h - 0.052) / 2, body, ha="center", va="center",
            fontsize=8.8, color=INK, linespacing=1.45)


def arrow(x1, y1, x2, y2, color=INK, dashed=False, lw=1.4, rad=0.0, label=None,
          lx=0.0, ly=0.0, label_color=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                va="center", fontsize=8.2, style="italic",
                color=label_color or color)


def lane(x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=LANE)


# ----------------------------------------------------------------- lanes
lane(0.015, 0.965, "A. Data (frozen)")
lane(0.315, 0.965, "B. Evolution loop (one LLM call per iteration)")
lane(0.015, 0.255, "C. Honesty harness")

# ----------------------------------------------------------------- A: data
box(0.015, 0.760, 0.220, 0.150, "E-commerce search logs",
    ["Expedia ICDM 2013", "9.9M impressions, 399k queries", "clicks, bookings, revenue"])
box(0.015, 0.545, 0.220, 0.150, "Preparation",
    ["labels rel / booking / revenue", "71 numeric features",
     "70 / 15 / 15 split by query"])
box(0.015, 0.330, 0.220, 0.150, "Two evaluation folds",
    ["small fold for the loop", "(fast but noisy scores)",
     "large held-out fold for the audit"])
arrow(0.125, 0.760, 0.125, 0.697)
arrow(0.125, 0.545, 0.125, 0.482)

# ----------------------------------------------------------------- B: loop
bw, bh = 0.205, 0.150
x1, x2 = 0.315, 0.560
ytop, ymid, ylow = 0.760, 0.545, 0.330
box(x1, ytop, bw, bh, "Program archive",
    ["keeps every program that is", "best at some trade-off of", "the three goals"])
box(x2, ytop, bw, bh, "Build the next prompt",
    ["one parent program + a few", "rivals, each shown with its", "scores and feedback"])
box(x2, ymid, bw, bh, "LLM edits the program",
    ["only the marked blocks change:", "campaign 1: the training loss",
     "campaign 2: features + model +", "loss + ensemble choices"])
box(x2, ylow, bw, bh, "Guarded evaluation",
    ["train and score the candidate;", "whitelists, time budget,", "leakage-safe statistics"])
box(x1, ymid, bw, bh, "Score + explain",
    ["relevance, conversion, revenue;", "plain-language feedback on what", "helped and what did not (Fig. 2)"])
box(x1, ylow, bw, bh, "Rejection teaches too",
    ["broken candidates score zero", "and receive an explanation", "the LLM reads next time"])

arrow(x1 + bw, ytop + bh / 2, x2, ytop + bh / 2)                     # db -> prompt
arrow(x2 + bw / 2, ytop, x2 + bw / 2, ymid + bh)                     # prompt -> mutation
arrow(x2 + bw / 2, ymid, x2 + bw / 2, ylow + bh)                     # mutation -> eval
arrow(x2, ylow + bh / 2, x1 + bw, ylow + bh / 2)                     # eval -> rejection lane
arrow(x1 + bw / 2, ylow + bh, x1 + bw / 2, ymid)                     # rejection -> scoring
arrow(x1 + bw / 2, ymid + bh, x1 + bw / 2, ytop)                     # scoring -> db
arrow(0.235, 0.340, x2 + bw / 2 - 0.02, 0.322, color=EDGE, lw=1.1, rad=0.22,
      label="fitness fold", lx=-0.115, ly=-0.052)

# ----------------------------------------------------------------- guidance
box(0.815, 0.760, 0.170, 0.150, "Seeded knowledge",
    ["expert ranking know-how", "in the system prompt", "(optional; ablated in", "campaign 1)"],
    edge=GUIDE_EDGE, fill=GUIDE_FILL, dashed=True)
arrow(0.815, 0.835, x2 + bw + 0.004, 0.835, color=GUIDE_EDGE, dashed=True, lw=1.2)

# ----------------------------------------------------------------- C: harness
hw = 0.300
box(0.015, 0.045, hw, 0.150, "1. Headroom gate (before spend)",
    ["can a hand-built candidate beat the", "measurement noise? if not, the loop",
     "would only select luck: do not run it"], edge=ACCENT, fill=ACCENT_FILL,
    title_color=ACCENT)
box(0.345, 0.045, hw, 0.150, "2. Reference baselines",
    ["tuned LambdaMART, LambdaLoss, and", "random search, given the same data",
     "and the same candidate budget"], edge=ACCENT, fill=ACCENT_FILL,
    title_color=ACCENT)
box(0.675, 0.045, hw, 0.150, "3. Transfer audit (after search)",
    ["winners re-scored on 60k queries the", "loop never saw; only gains that",
     "survive there count as discoveries"], edge=ACCENT, fill=ACCENT_FILL,
    title_color=ACCENT)

arrow(0.165, 0.195, 0.315, 0.300, color=ACCENT, rad=0.15,
      label="go / no-go", lx=0.070, ly=-0.012)
arrow(0.660, 0.330, 0.800, 0.195, color=ACCENT, rad=0.15,
      label="selected programs", lx=0.088, ly=0.016)
arrow(0.648, 0.120, 0.672, 0.120, color=ACCENT, lw=1.1)
arrow(0.342, 0.120, 0.318, 0.120, color=ACCENT, lw=1.1)

fig.savefig("evorank_flowchart.png", bbox_inches="tight", facecolor="white")
fig.savefig("evorank_flowchart.pdf", bbox_inches="tight", facecolor="white")
print("wrote evorank_flowchart.png and .pdf")
