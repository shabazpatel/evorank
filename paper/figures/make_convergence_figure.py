"""Poster figure: continual improvement inside the loop (campaign 2).

Top panel: best-so-far fitness NDCG@10 per iteration for the three pipeline
runs, with the starting pipeline and the Optuna-tuned LambdaMART as
reference lines. Bottom panel: the per-iteration increments of the
best-so-far curve (the improvement events), one bar group per run.

    uv run python paper/figures/make_convergence_figure.py
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent

INK = "#222222"
ACCENT = "#1F4E79"
SEED_COLORS = {"s0": "#7FA6C9", "s1": "#3C6EA5", "s2": "#1F4E79"}
BASELINE = 0.4330   # tuned LambdaMART, fitness-fold validation (pipeline_transfer.csv)
START = 0.4218      # starting pipeline, fitness-fold validation

curves = {}
for s in ("s0", "s1", "s2"):
    rows = list(csv.DictReader(open(ROOT / "runs" / "summary"
                                    / f"convergence_pipeline_{s}.csv")))
    curves[s] = np.array([float(r["best_ndcg"]) for r in rows])

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9.6, 5.3), dpi=300, sharex=True,
    gridspec_kw={"height_ratios": [1.9, 0.85], "hspace": 0.16})

it = np.arange(1, 51)
ax1.axhline(BASELINE, color=INK, lw=1.6, ls=(0, (5, 3)))
ax1.axhline(START, color="#999999", lw=1.4, ls=(0, (2, 3)))
ax1.text(16, BASELINE - 0.0007, "LambdaMART + Optuna (0.4330)", va="top",
         ha="left", fontsize=11, color=INK)
ax1.text(1.0, START + 0.0007, "starting pipeline (0.4218)", va="bottom",
         ha="left", fontsize=11, color="#777777")
for s, c in curves.items():
    ax1.step(it, c, where="post", lw=2.6, color=SEED_COLORS[s],
             label=f"run {s}  (final {c[-1]:.4f})")
    cross = int(np.argmax(c > BASELINE)) + 1
    ax1.plot(cross, c[cross - 1], "o", ms=9, color=SEED_COLORS[s],
             markeredgecolor="white", markeredgewidth=1.2, zorder=5)
ax1.set_ylabel("best-so-far NDCG@10\n(fitness fold)", fontsize=12)
ax1.legend(loc="lower right", fontsize=11, frameon=False)
ax1.set_ylim(0.418, 0.443)
ax1.tick_params(labelsize=11)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title("Every run crosses the tuned baseline (dots), then keeps improving",
              fontsize=13, color=ACCENT, fontweight="bold", loc="left", pad=10)

width = 0.3
for k, (s, c) in enumerate(curves.items()):
    inc = np.diff(np.concatenate([[START], c]))
    inc[inc < 0] = 0.0
    ax2.bar(it + (k - 1) * width, inc, width=width, color=SEED_COLORS[s],
            edgecolor="none")
ax2.set_xlabel("iteration (one LLM call each)", fontsize=12)
ax2.set_ylabel("improvement\nper iteration", fontsize=12)
ax2.set_xlim(0, 51.5)
ax2.tick_params(labelsize=11)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_yticks([0, 0.005, 0.010])

fig.savefig(OUT / "convergence_poster.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "convergence_poster.png", bbox_inches="tight", facecolor="white")
print("wrote convergence_poster.pdf/.png; crossings:",
      {s: int(np.argmax(c > BASELINE)) + 1 for s, c in curves.items()})
