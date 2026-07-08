"""Render the EvoRank system flowchart as high-resolution PNG and vector PDF.

Reproducible paper figure, no browser or mermaid dependency.

    uv run python paper/figures/make_flowchart.py
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent

FILL = {"data": "#eef3fb", "loop": "#eefaf0", "kb": "#fff7e0", "base": "#f6eef9"}
EDGE = {"data": "#4a6fa5", "loop": "#3f8f5f", "kb": "#b8860b", "base": "#7d5a96"}


def box(ax, x, y, w, h, title, lines, kind, dashed=False, title_size=10.5, body_size=9.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=FILL[kind], edgecolor=EDGE[kind],
        linewidth=1.6, linestyle="--" if dashed else "-", zorder=2))
    cx = x + w / 2
    ax.text(cx, y + h - 0.030, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", zorder=3)
    if lines:
        body = "\n".join(lines)
        ax.text(cx, y + h - 0.072, body, ha="center", va="top",
                fontsize=body_size, linespacing=1.35, zorder=3)


def arrow(ax, p0, p1, dashed=False, color="#333333", rad=0.0, lw=1.7):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14,
        linestyle="--" if dashed else "-", color=color,
        linewidth=lw, connectionstyle=f"arc3,rad={rad}", zorder=4))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- data preparation (left column) ----------------
    ax.text(0.135, 0.975, "Data preparation (frozen)", ha="center", fontsize=12,
            fontweight="bold", color=EDGE["data"])
    box(ax, 0.03, 0.80, 0.21, 0.125, "E-commerce search logs",
        ["10M impressions, 400k queries", "clicks, bookings, revenue"], "data")
    box(ax, 0.03, 0.60, 0.21, 0.135, "Prepare",
        ["labels rel / booking / rev", "71 numeric features",
         "70/15/15 split by query id"], "data")
    box(ax, 0.03, 0.40, 0.21, 0.125, "Fast subset",
        ["8k train queries", "one evaluation ~40 s"], "data")
    box(ax, 0.03, 0.22, 0.21, 0.110, "Full split",
        ["final reporting"], "data")
    arrow(ax, (0.135, 0.80), (0.135, 0.737))
    arrow(ax, (0.135, 0.60), (0.135, 0.527))
    # Prepare -> Full split, routed outside the column so it crosses no box
    arrow(ax, (0.03, 0.655), (0.03, 0.29), rad=0.12, color=EDGE["data"])

    # ---------------- evolution loop (center/right) ----------------
    ax.text(0.615, 0.975, "Evolution loop (one LLM call per iteration)",
            ha="center", fontsize=12, fontweight="bold", color=EDGE["loop"])

    box(ax, 0.335, 0.76, 0.205, 0.155, "Pareto program database",
        ["3-objective archive", "2 islands, migration", "keeps non-dominated programs"],
        "loop")
    box(ax, 0.615, 0.76, 0.205, 0.155, "Prompt builder",
        ["parent program", "+ 4 context programs", "with scores and feedback"], "loop")
    box(ax, 0.615, 0.545, 0.205, 0.130, "LLM mutation",
        ["rewrites the EVOLVE block", "structure only, not numbers"], "loop")
    box(ax, 0.615, 0.335, 0.205, 0.135, "Candidate objective",
        ["lambdamart_objective(predt,", "rel, booking, rev, slices)",
         "returns per-row grad, hess"], "loop")
    box(ax, 0.335, 0.335, 0.205, 0.135, "Guardrailed evaluator",
        ["XGBoost, fixed hyperparams", "timeout, non-finite reject",
         "degenerate = zero fitness"], "loop")
    box(ax, 0.335, 0.545, 0.205, 0.130, "Three metrics + feedback",
        ["ndcg | book_ndcg | revenue", "feedback string to next prompt"], "loop")

    # knowledge base (the ablation treatment)
    box(ax, 0.865, 0.76, 0.115, 0.155, "Knowledge base",
        ["curated LTR priors", "memory-ON", "arm only"], "kb", dashed=True,
        title_size=10, body_size=8.5)

    # loop arrows (clockwise)
    arrow(ax, (0.540, 0.838), (0.615, 0.838))                    # DB -> prompt
    arrow(ax, (0.7175, 0.760), (0.7175, 0.675))                  # prompt -> LLM
    arrow(ax, (0.7175, 0.545), (0.7175, 0.470))                  # LLM -> candidate
    arrow(ax, (0.615, 0.4025), (0.540, 0.4025))                  # candidate -> evaluator
    arrow(ax, (0.4375, 0.470), (0.4375, 0.545))                  # evaluator -> metrics
    arrow(ax, (0.4375, 0.675), (0.4375, 0.760))                  # metrics -> DB
    arrow(ax, (0.865, 0.838), (0.820, 0.838), dashed=True, color=EDGE["kb"])
    arrow(ax, (0.240, 0.4625), (0.335, 0.4200), color=EDGE["data"])  # fast subset -> evaluator

    # ---------------- baselines strip (bottom) ----------------
    ax.text(0.660, 0.228, "Reference points (same data, same harness)",
            ha="center", fontsize=12, fontweight="bold", color=EDGE["base"])
    labels = [
        ("LambdaMART", "default"),
        ("LambdaMART", "+ Optuna, 40 trials"),
        ("LambdaLoss", "NDCG-Loss2"),
        ("Random objective search", "equal 40-candidate budget"),
    ]
    xs = [0.300, 0.465, 0.630, 0.795]
    for (t, s), x in zip(labels, xs):
        box(ax, x, 0.065, 0.148, 0.110, t, [s], "base", title_size=9.5, body_size=8.5)
    arrow(ax, (0.4375, 0.335), (0.4375, 0.252), dashed=True, color=EDGE["base"])
    ax.text(0.447, 0.288, "compared against", fontsize=8.5, color=EDGE["base"])

    fig.tight_layout()
    fig.savefig(HERE / "evorank_flowchart.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / "evorank_flowchart.pdf", bbox_inches="tight")
    print("wrote", HERE / "evorank_flowchart.png", "and .pdf")


if __name__ == "__main__":
    main()
