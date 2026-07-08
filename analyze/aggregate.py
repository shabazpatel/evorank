"""Aggregate all EvoRank runs into the paper's tables and plot data.

Parses two record sources under runs/:
  1. Baseline records: runs/<name>.json written by baselines/_common.log_run.
  2. Evolution runs: any runs/<dir>/ containing adaevolve_iteration_stats_*.jsonl
     (SkyDiscover adaevolve output). Extracts per-iteration metrics, best-so-far
     convergence, and the final best program metrics.

Outputs, no manual transcription (CLAUDE.md section 13):
  - comparison table printed to stdout (three metrics per method)
  - runs/summary/comparison.csv
  - runs/summary/convergence_<run>.csv (best-so-far ndcg per iteration)
  - runs/summary/frontier_<run>.csv (final Pareto candidate metrics)

    python analyze/aggregate.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SUMMARY = RUNS / "summary"
METRICS = ("ndcg", "book_ndcg", "revenue")


def _metrics_from(obj: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Pull the three metrics out of a record that may nest them differently."""
    for container in (obj, obj.get("metrics", {}), obj.get("child_metrics", {}),
                      obj.get("best_program_metrics", {})):
        if isinstance(container, dict) and all(k in container for k in METRICS):
            try:
                return {k: float(container[k]) for k in METRICS}
            except (TypeError, ValueError):
                continue
    return None


def load_baselines() -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(RUNS.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        m = _metrics_from(rec)
        if m:
            rows.append({"method": rec.get("name", path.stem), "kind": "baseline", **m})
    return rows


def _latest_checkpoint(run_dir: pathlib.Path) -> Optional[pathlib.Path]:
    cps = sorted(run_dir.glob("checkpoints/checkpoint_*"),
                 key=lambda p: int(p.name.rsplit("_", 1)[-1]))
    return cps[-1] if cps else None


def load_evolution_runs() -> List[Dict[str, Any]]:
    rows = []
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        stats_files = sorted(run_dir.glob("adaevolve_iteration_stats_*.jsonl"))
        if not stats_files:
            continue

        # Best-so-far convergence straight from the per-iteration stats:
        # global.best_program.metrics is the running best at each iteration.
        best_so_far: List[Dict[str, Any]] = []
        pareto_ids: List[str] = []
        for sf in stats_files:
            for line in sf.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                g = rec.get("global", {})
                m = _metrics_from(g.get("best_program", {}))
                if m:
                    best_so_far.append({"iteration": rec.get("iteration"), **{
                        f"best_{k}": v for k, v in m.items()}})
                pareto_ids = g.get("global_pareto_front_ids", pareto_ids)
        if not best_so_far:
            continue

        # All evaluated candidates (the frontier data) from the last checkpoint.
        candidates: List[Dict[str, Any]] = []
        cp = _latest_checkpoint(run_dir)
        if cp:
            for pf in sorted((cp / "programs").glob("*.json")):
                try:
                    prog = json.loads(pf.read_text())
                except json.JSONDecodeError:
                    continue
                m = _metrics_from(prog)
                if m:
                    candidates.append({
                        "id": prog.get("id", pf.stem),
                        "iteration": prog.get("iteration_found", ""),
                        "on_pareto_front": prog.get("id", pf.stem) in pareto_ids,
                        **m,
                    })

        SUMMARY.mkdir(parents=True, exist_ok=True)
        _write_csv(SUMMARY / f"convergence_{run_dir.name}.csv",
                   ["iteration", *[f"best_{k}" for k in METRICS]], best_so_far)
        if candidates:
            _write_csv(SUMMARY / f"frontier_{run_dir.name}.csv",
                       ["id", "iteration", "on_pareto_front", *METRICS], candidates)

        final = best_so_far[-1]
        rows.append({"method": run_dir.name, "kind": "evolution",
                     "iterations": len(best_so_far),
                     **{k: final[f"best_{k}"] for k in METRICS}})
    return rows


def _write_csv(path: pathlib.Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    baselines = load_baselines()
    evolution = load_evolution_runs()
    rows = baselines + evolution
    if not rows:
        print("no runs found under", RUNS)
        sys.exit(1)

    SUMMARY.mkdir(parents=True, exist_ok=True)
    fields = ["method", "kind", "ndcg", "book_ndcg", "revenue", "iterations"]
    _write_csv(SUMMARY / "comparison.csv", fields, rows)

    rows.sort(key=lambda r: r["ndcg"], reverse=True)
    print(f"{'method':<28} {'kind':<10} {'ndcg':>8} {'book_ndcg':>10} {'revenue':>8} {'iters':>6}")
    print("-" * 76)
    for r in rows:
        print(f"{r['method']:<28} {r['kind']:<10} {r['ndcg']:>8.4f} "
              f"{r['book_ndcg']:>10.4f} {r['revenue']:>8.4f} {str(r.get('iterations', '')):>6}")
    print(f"\nwrote {SUMMARY / 'comparison.csv'} plus per-run convergence/frontier CSVs")


if __name__ == "__main__":
    main()
