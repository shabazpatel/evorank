"""Data evaluation for EvoRank (Phase 1 gate).

Validates the prepared dataset before any tokens are spent on the evolution
loop. Produces the numbers the paper's Experimental Setup section reports, and
enforces hard acceptance checks. Exit code is nonzero if any hard check fails.

Hard checks (must pass):
  1. Split integrity: no query id appears in more than one fold.
  2. Group contiguity: rows are sorted so each qid forms one contiguous block.
  3. Label domain: rel in {0, 1, 5}; booking in {0, 1}; rev >= 0.
  4. Label consistency: rev > 0 only where booking == 1; rel == 5 iff booking == 1.
  5. Scoreable queries: every fold has queries with idcg > 0 for each metric
     (a fold where no query has a booking cannot measure book_ndcg or revenue).

Soft checks (reported, warn only):
  6. Pareto divergence: share of queries with more than one booking, and share
     where the revenue-ideal ordering differs from the booking-ideal ordering.
     If near zero, book_ndcg and revenue coincide and the three-objective story
     weakens; report this in the paper either way.
  7. Random subset: share of rows and queries with random_flag == 1 (the
     position-unbiased impressions for the optional debiasing analysis).
  8. Evaluation cost: wall time of one evaluate() call on the fast subset.
     Warn above the per-candidate budget (default 60s), since the evolution
     loop performs one training per candidate.

Usage:
    python data/data_report.py            # evaluates the fast subset
    python data/data_report.py --full     # evaluates the full split
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ltr.dataset import get_dataset  # noqa: E402

BUDGET_S = 60.0


class CheckFailure(Exception):
    pass


def _hard(cond: bool, msg: str, failures: list[str]) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        failures.append(msg)


def _fold_stats(name: str, f: pd.DataFrame) -> None:
    n_q = f["qid"].nunique()
    gs = f.groupby("qid", sort=False).size()
    booked_q = f.groupby("qid")["booking"].sum()
    clicked_q = f.groupby("qid")["rel"].apply(lambda r: int((r >= 1).sum()))
    print(f"  {name:5s} queries {n_q:>8}  rows {len(f):>10}")
    print(f"        group size  min {gs.min()}  median {int(gs.median())}  max {gs.max()}")
    print(f"        rel dist    {f['rel'].value_counts().sort_index().to_dict()}")
    print(f"        queries with >=1 booking  {int((booked_q > 0).sum())} "
          f"({(booked_q > 0).mean():.1%}),  >=1 positive  {int((clicked_q > 0).sum())} "
          f"({(clicked_q > 0).mean():.1%})")
    booked_rev = f.loc[f["booking"] == 1, "rev"]
    if len(booked_rev):
        print(f"        rev (booked rows)  mean {booked_rev.mean():.2f}  "
              f"median {booked_rev.median():.2f}  p95 {booked_rev.quantile(0.95):.2f}")


def _divergence(f: pd.DataFrame) -> tuple[float, float]:
    """Share of queries with >1 booking, and share where revenue-ideal ordering
    differs from booking-ideal ordering (restricted to queries with >=1 booking)."""
    multi, diverge, with_book = 0, 0, 0
    for _, g in f.groupby("qid", sort=False):
        b = g["booking"].to_numpy()
        r = g["rev"].to_numpy()
        nb = int(b.sum())
        if nb == 0:
            continue
        with_book += 1
        if nb > 1:
            multi += 1
            # orderings differ if the booked items are not all equal in rev
            if len(np.unique(r[b == 1])) > 1:
                diverge += 1
    if with_book == 0:
        return 0.0, 0.0
    return multi / with_book, diverge / with_book


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="evaluate the full split instead of the fast subset")
    ap.add_argument("--skip-timing", action="store_true", help="skip the evaluate() timing check")
    args = ap.parse_args()

    fast = not args.full
    train_df, val_df, test_df, feats, src = get_dataset(fast=fast)
    folds = {"train": train_df, "val": val_df, "test": test_df}
    failures: list[str] = []

    print(f"source: {src}   subset: {'fast' if fast else 'full'}   features: {len(feats)}")
    print()

    print("Fold statistics")
    for name, f in folds.items():
        _fold_stats(name, f)
    print()

    print("Hard checks")
    tr_q, va_q, te_q = (set(f["qid"]) for f in folds.values())
    _hard(not (tr_q & va_q) and not (tr_q & te_q) and not (va_q & te_q),
          "no query id appears in more than one fold", failures)
    for name, f in folds.items():
        qid = f["qid"].to_numpy()
        change_points = int((qid[1:] != qid[:-1]).sum()) + 1
        _hard(change_points == f["qid"].nunique(),
              f"{name}: each qid forms one contiguous block", failures)
    all_df = pd.concat(folds.values())
    _hard(set(np.unique(all_df["rel"])) <= {0, 1, 5}, "rel values in {0, 1, 5}", failures)
    _hard(set(np.unique(all_df["booking"])) <= {0, 1}, "booking values in {0, 1}", failures)
    _hard(bool((all_df["rev"] >= 0).all()), "rev >= 0 everywhere", failures)
    _hard(bool((all_df.loc[all_df["booking"] == 0, "rev"] == 0).all()),
          "rev == 0 wherever booking == 0", failures)
    _hard(bool(((all_df["rel"] == 5) == (all_df["booking"] == 1)).all()),
          "rel == 5 exactly where booking == 1", failures)
    for name, f in folds.items():
        has_book = (f.groupby("qid")["booking"].sum() > 0).any()
        has_pos = (f.groupby("qid")["rel"].max() > 0).any()
        _hard(bool(has_book and has_pos),
              f"{name}: has queries scoreable on all three metrics", failures)
    print()

    print("Soft checks")
    for name, f in folds.items():
        multi, diverge = _divergence(f)
        note = "" if multi > 0 else "  <- WARN: book_ndcg and revenue will coincide"
        print(f"  {name:5s} queries with >1 booking {multi:.1%}, revenue-vs-booking ordering "
              f"divergence {diverge:.1%}{note}")
    if "random_flag" in all_df.columns:
        rf_rows = all_df["random_flag"].mean()
        rf_q = all_df.groupby("qid")["random_flag"].max().mean()
        print(f"  random subset: {rf_rows:.1%} of rows, {rf_q:.1%} of queries (debiasing asset)")
    else:
        print("  random_flag column absent (synthetic source, or prepared before the fix)")
    print()

    if not args.skip_timing:
        print("Evaluation cost (one evaluate() call on the fast subset)")
        from eval.evaluator import evaluate  # noqa: E402  (import here to avoid cost when skipped)
        t0 = time.time()
        res = evaluate(str(ROOT / "seed" / "initial_program.py"))
        dt = time.time() - t0
        over = "  <- WARN: over per-candidate budget, shrink the fast subset" if dt > BUDGET_S else ""
        print(f"  wall time {dt:.1f}s (budget {BUDGET_S:.0f}s){over}")
        print(f"  seed metrics {res['artifacts']['feedback']}")
        print()

    if failures:
        print(f"RESULT: {len(failures)} hard check(s) FAILED")
        for m in failures:
            print(f"  - {m}")
        raise SystemExit(1)
    print("RESULT: all hard checks passed. Data is cleared for baselines and evolution runs.")


if __name__ == "__main__":
    main()
