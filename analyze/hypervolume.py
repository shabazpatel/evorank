"""Hypervolume and Pareto-front quality per run (paper ablation endpoint).

Recomputes each evolution run's non-dominated front from the frontier CSVs
(all evaluated candidates), then the dominated 3D hypervolume against a common
reference point. Baselines enter as single points. The reference point is the
component-wise minimum over every point considered, minus a small margin, so
all methods share one scale.

Also reports front size and per-axis bests, since a single scalar can hide
which axis a front is strong on.

    uv run python analyze/hypervolume.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from typing import List, Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SUMMARY = RUNS / "summary"
METRICS = ("ndcg", "book_ndcg", "revenue")


def nondominated(points: Sequence[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    """Maximization in every coordinate."""
    front: List[Tuple[float, ...]] = []
    for p in points:
        if any(all(q[i] >= p[i] for i in range(len(p))) and q != p for q in points):
            continue
        if p not in front:
            front.append(p)
    return front


def hv2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """2D hypervolume, maximization, points assumed >= ref."""
    pts = nondominated(points)
    pts.sort(key=lambda p: p[0], reverse=True)  # descending x
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return hv


def hv3d(points: List[Tuple[float, float, float]], ref: Tuple[float, float, float]) -> float:
    """3D hypervolume by sweeping the third axis (slicing), maximization."""
    pts = [p for p in nondominated(points) if all(p[i] > ref[i] for i in range(3))]
    if not pts:
        return 0.0
    zs = sorted({p[2] for p in pts}, reverse=True)
    hv, prev_z = 0.0, None
    for i, z in enumerate(zs):
        # slab between this z-level and the next lower one (or ref)
        lower = zs[i + 1] if i + 1 < len(zs) else ref[2]
        active = [(p[0], p[1]) for p in pts if p[2] >= z]
        hv += hv2d(active, (ref[0], ref[1])) * (z - lower)
    return hv


def load_run_fronts() -> dict:
    fronts = {}
    for f in sorted(SUMMARY.glob("frontier_*.csv")):
        run = f.stem.replace("frontier_", "")
        pts = []
        for row in csv.DictReader(open(f)):
            pts.append(tuple(float(row[m]) for m in METRICS))
        if pts:
            fronts[run] = nondominated(pts)
    return fronts


def load_baseline_points() -> dict:
    pts = {}
    for f in sorted(RUNS.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        m = rec.get("metrics", {})
        if all(k in m for k in METRICS):
            pts[rec.get("name", f.stem)] = [tuple(float(m[k]) for k in METRICS)]
    return pts


def main() -> None:
    fronts = load_run_fronts()
    baselines = load_baseline_points()
    everything = {**fronts, **baselines}
    if not everything:
        sys.exit("no fronts found; run analyze/aggregate.py first")

    all_pts = [p for pts in everything.values() for p in pts]
    ref = tuple(min(p[i] for p in all_pts) - 0.005 for i in range(3))
    print(f"reference point: ({ref[0]:.4f}, {ref[1]:.4f}, {ref[2]:.4f})\n")

    rows = []
    for name, pts in sorted(everything.items()):
        hv = hv3d(list(pts), ref)
        best = {m: max(p[i] for p in pts) for i, m in enumerate(METRICS)}
        rows.append({"method": name, "kind": "evolution" if name in fronts else "baseline",
                     "front_size": len(pts), "hypervolume": hv, **{
                         f"best_{m}": best[m] for m in METRICS}})

    rows.sort(key=lambda r: -r["hypervolume"])
    print(f"{'method':<28} {'kind':<10} {'front':>5} {'hypervolume':>12} "
          f"{'best_ndcg':>10} {'best_book':>10} {'best_rev':>9}")
    print("-" * 90)
    for r in rows:
        print(f"{r['method']:<28} {r['kind']:<10} {r['front_size']:>5} "
              f"{r['hypervolume']:>12.6f} {r['best_ndcg']:>10.4f} "
              f"{r['best_book_ndcg']:>10.4f} {r['best_revenue']:>9.4f}")

    out = SUMMARY / "hypervolume.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
