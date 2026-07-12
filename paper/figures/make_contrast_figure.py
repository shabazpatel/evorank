"""Figure 3: the study in one picture. Per-seed NDCG@10 gains over the seed
program, on the selection fold (open) vs the 60k-query held-out fold
(filled), campaign 1 vs campaign 2. Data from runs/summary CSVs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, ACCENT, GRAY = "#222222", "#1f4e79", "#888888"

# (val_gain, test_gain) per seed, vs the same seed-program reference
c1 = [(0.0036, -0.0013), (0.0038, 0.0002), (0.0046, 0.0027),
      (0.0047, 0.0023), (0.0038, 0.0002), (0.0047, 0.0013)]
c2 = [(0.0138, 0.0094), (0.0141, 0.0100), (0.0180, 0.0131)]

fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=300)
ax.axhspan(-0.0012, 0.0012, color="#dddddd", alpha=0.6, zorder=0)
ax.axhline(0, color=GRAY, lw=0.8, zorder=1)

def group(vals, x0, label):
    for i, (v, t) in enumerate(vals):
        x = x0 + i * 0.16
        ax.plot([x, x], [v, t], color=GRAY, lw=1.0, zorder=2)
        ax.plot(x, v, "o", mfc="white", mec=INK, ms=6.5, mew=1.3, zorder=3)
        ax.plot(x, t, "o", color=ACCENT, ms=6.5, zorder=3)
    ax.text(x0 + (len(vals) - 1) * 0.08, -0.0062, label, ha="center",
            fontsize=10, color=INK)

group(c1, 0.0, "Campaign 1: training objectives\n(per-edit effects below noise)")
group(c2, 1.55, "Campaign 2: full pipelines\n(effects above noise)")

ax.plot([], [], "o", mfc="white", mec=INK, mew=1.3, label="gain on selection fold")
ax.plot([], [], "o", color=ACCENT, label="gain on 60k held-out queries")
ax.plot([], [], "s", color="#dddddd", ms=10, label="held-out noise band")
ax.legend(loc="upper left", frameon=False, fontsize=9)

ax.set_ylabel("NDCG@10 gain vs seed program", fontsize=10)
ax.set_xlim(-0.35, 2.15)
ax.set_ylim(-0.0075, 0.020)
ax.set_xticks([])
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("transfer_contrast.png", bbox_inches="tight", facecolor="white")
fig.savefig("transfer_contrast.pdf", bbox_inches="tight", facecolor="white")
print("wrote transfer_contrast.png/.pdf")
