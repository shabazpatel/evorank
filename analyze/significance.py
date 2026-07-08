"""Paired bootstrap significance over queries for the key comparisons.

Retrains the compared methods on the fast-subset train fold (deterministic,
same harness as the search), predicts the val fold, computes per-query metric
values, and runs a paired bootstrap over queries (default 10k resamples).
For ndcg and book_ndcg the statistic is the mean over queries valid for that
metric; for revenue it is the pooled ratio sum(realized)/sum(ideal), recomputed
per resample.

Reports the observed delta, a 95 percent CI for the delta, and a two-sided
bootstrap p-value, per metric and comparison. Writes runs/summary/significance.csv.

    uv run python analyze/significance.py --resamples 10000
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys

import numpy as np
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ltr.dataset import FIXED_PARAMS, FIXED_ROUNDS, get_dataset, to_dmatrix  # noqa: E402
from ltr.metrics import group_sizes_of, group_slices, per_query_values  # noqa: E402


def _load_program(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"cand_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _custom_obj(mod, train_df, slices):
    rel = train_df["rel"].to_numpy(dtype=np.float64)
    booking = train_df["booking"].to_numpy(dtype=np.float64)
    rev = train_df["rev"].to_numpy(dtype=np.float64)

    def obj(predt, dtrain):
        g, h = mod.lambdamart_objective(predt, rel, booking, rev, slices)
        return np.asarray(g, dtype=np.float64), np.asarray(h, dtype=np.float64)
    return obj


def predictions(train_df, val_df, feats) -> dict:
    """Train each compared method once, return {name: val predictions}."""
    slices = group_slices(group_sizes_of(train_df))
    dtr = to_dmatrix(train_df, feats)
    dva = to_dmatrix(val_df, feats)
    preds = {}

    def train_custom(name, program_path, params=None, rounds=None):
        mod = _load_program(ROOT / program_path)
        p = params or FIXED_PARAMS
        bst = xgb.train(p, dtr, num_boost_round=rounds or FIXED_ROUNDS,
                        obj=_custom_obj(mod, train_df, slices))
        preds[name] = bst.predict(dva)
        print(f"trained {name}")

    # evolved programs under the fixed search config
    train_custom("seed", "seed/initial_program.py")
    train_custom("evolved_best", "runs/mem_on_s1/checkpoints/checkpoint_40/best_program.py")
    train_custom("random_best", "runs/random_search_s0/best_program.py")
    simplified = ROOT / "runs/summary/mechanism_ablation/no_adaptive.py"
    if simplified.exists():
        train_custom("evolved_simplified", "runs/summary/mechanism_ablation/no_adaptive.py")

    # discovered objective with its post-hoc tuned params
    retune = json.loads((ROOT / "runs/posthoc_retune_best_program.json").read_text())
    bp = dict(retune["extra"]["best_params"])
    n = bp.pop("num_boost_round")
    tuned_params = {"tree_method": "hist", "seed": 0, "base_score": 0.0,
                    "disable_default_eval_metric": 1, **bp}
    train_custom("evolved_best_tuned",
                 "runs/mem_on_s1/checkpoints/checkpoint_40/best_program.py",
                 params=tuned_params, rounds=n)

    # optuna-tuned built-in baseline, from its stored best params
    opt = json.loads((ROOT / "runs/lambdamart_optuna.json").read_text())
    bp = dict(opt["extra"]["best_params"])
    n = bp.pop("num_boost_round")
    bst = xgb.train({"objective": "rank:ndcg", "tree_method": "hist", "seed": 0, **bp},
                    dtr, num_boost_round=n)
    preds["optuna"] = bst.predict(dva)
    print("trained optuna")
    return preds


COMPARISONS = [
    ("evolved_best", "seed"),
    ("evolved_best", "random_best"),
    ("evolved_best", "optuna"),
    ("evolved_best_tuned", "optuna"),
    ("evolved_simplified", "optuna"),
    ("evolved_simplified", "random_best"),
]


def bootstrap(pq_a: dict, pq_b: dict, n_boot: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    nq = len(pq_a["ndcg"])
    rows = []
    for metric in ("ndcg", "book_ndcg", "revenue"):
        if metric == "revenue":
            def stat(idx, pq):
                ideal = pq["rev_ideal"][idx].sum()
                return pq["rev_realized"][idx].sum() / ideal if ideal > 0 else 0.0
            valid = np.arange(nq)
        else:
            valid = np.where(~np.isnan(pq_a[metric]) & ~np.isnan(pq_b[metric]))[0]

            def stat(idx, pq, m=metric):
                return float(np.nanmean(pq[m][idx]))
        obs = stat(valid, pq_a) - stat(valid, pq_b)
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(valid, size=len(valid), replace=True)
            deltas[b] = stat(idx, pq_a) - stat(idx, pq_b)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        # two-sided p: fraction of bootstrap deltas on the other side of zero
        p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        rows.append({"metric": metric, "delta": obs, "ci_lo": lo, "ci_hi": hi,
                     "p_value": min(1.0, max(p, 1.0 / n_boot)),
                     "n_queries": len(valid)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--full", action="store_true",
                    help="train on the full split and bootstrap the full fold (very slow "
                         "for custom objectives; prefer --eval-full)")
    ap.add_argument("--eval-full", action="store_true",
                    help="train on the fast subset (fixed training budget) but bootstrap "
                         "the FULL fold: tight confidence intervals at seconds per method")
    ap.add_argument("--fold", choices=["val", "test"], default="val")
    args = ap.parse_args()

    train_df, val_df, test_df, feats, src = get_dataset(fast=not args.full)
    if args.eval_full:
        _ftr, fval, ftest, _f, fsrc = get_dataset(fast=False)
        val_df, test_df = fval, ftest
        src = f"train: {src} | eval: {fsrc}"
    eval_df = test_df if args.fold == "test" else val_df
    print(f"source: {src}  fold: {args.fold}  queries: {eval_df.qid.nunique():,}")
    preds = predictions(train_df, eval_df, feats)
    pq = {name: per_query_values(eval_df, p) for name, p in preds.items()}

    out_rows = []
    for a, b in COMPARISONS:
        if a not in pq or b not in pq:
            print(f"skip {a} vs {b} (method not available)")
            continue
        for r in bootstrap(pq[a], pq[b], args.resamples):
            sig = "*" if r["p_value"] < 0.05 else " "
            print(f"{a} vs {b:<14} {r['metric']:<10} delta={r['delta']:+.4f} "
                  f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p_value']:.4f}{sig}")
            out_rows.append({"method_a": a, "method_b": b, **r})

    train_tag = "full" if args.full else "fast"
    eval_tag = "fullfold" if args.eval_full else train_tag
    suffix = f"{train_tag}train_{eval_tag}_{args.fold}"
    out = ROOT / "runs" / "summary" / f"significance_{suffix}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
