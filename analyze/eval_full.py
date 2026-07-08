"""Full-split and test-fold evaluation of all key methods (final reporting).

Everything so far was measured on the fast-subset val fold, the same fold the
search selected on. This script retrains every key method on the FULL train
fold (280k queries) and reports both the full val fold (tuning-adjacent) and
the untouched full test fold (the paper's headline numbers).

Appends one CSV row per method as it finishes, so partial progress survives
interruption; already-present methods are skipped on relaunch (resume-aware).

    uv run python analyze/eval_full.py            # all methods
    uv run python analyze/eval_full.py --only seed,optuna
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ltr.dataset import FIXED_PARAMS, FIXED_ROUNDS, get_dataset, to_dmatrix  # noqa: E402
from ltr.metrics import group_sizes_of, group_slices, metrics_bundle  # noqa: E402

OUT = ROOT / "runs" / "summary" / "full_eval.csv"
FIELDS = ["method", "fold", "ndcg", "book_ndcg", "revenue", "train_seconds"]

BEST_PROGRAMS = {
    f"evolved_{name}": f"runs/{name}/checkpoints/checkpoint_40/best_program.py"
    for name in ("mem_on_s0", "mem_on_s1", "mem_on_s2",
                 "mem_off_s0", "mem_off_s1", "mem_off_s2")
}


def _load_program(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"cand_{abs(hash(str(path)))}", path)
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


def _stored_params(record: str, base: dict) -> tuple:
    rec = json.loads((ROOT / record).read_text())
    bp = dict(rec["extra"]["best_params"])
    n = bp.pop("num_boost_round")
    return {**base, **bp}, n


def method_specs() -> dict:
    """name -> (kind, program_path or None, params, rounds)"""
    specs = {
        "seed": ("custom", "seed/initial_program.py", FIXED_PARAMS, FIXED_ROUNDS),
        "random_best": ("custom", "runs/random_search_s0/best_program.py",
                        FIXED_PARAMS, FIXED_ROUNDS),
        "lambdamart_default": ("builtin", None, {
            "objective": "rank:ndcg", "eta": 0.1, "max_depth": 6,
            "min_child_weight": 0.1, "tree_method": "hist", "seed": 0}, FIXED_ROUNDS),
    }
    for name, path in BEST_PROGRAMS.items():
        specs[name] = ("custom", path, FIXED_PARAMS, FIXED_ROUNDS)

    # Simplified variant found by the mechanism ablation (constant alpha blend);
    # best fixed-params program, see runs/summary/mechanism_ablation.csv.
    simplified = ROOT / "runs/summary/mechanism_ablation/no_adaptive.py"
    if simplified.exists():
        specs["evolved_simplified"] = (
            "custom", "runs/summary/mechanism_ablation/no_adaptive.py",
            FIXED_PARAMS, FIXED_ROUNDS)

    params, n = _stored_params("runs/lambdamart_optuna.json",
                               {"objective": "rank:ndcg", "tree_method": "hist", "seed": 0})
    specs["lambdamart_optuna"] = ("builtin", None, params, n)

    params, n = _stored_params(
        "runs/posthoc_retune_best_program.json",
        {"tree_method": "hist", "seed": 0, "base_score": 0.0,
         "disable_default_eval_metric": 1})
    specs["evolved_best_tuned"] = (
        "custom", "runs/mem_on_s1/checkpoints/checkpoint_40/best_program.py", params, n)
    return specs


def done_methods() -> set:
    if not OUT.exists():
        return set()
    return {r["method"] for r in csv.DictReader(open(OUT))}


def append_rows(rows) -> None:
    exists = OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated method subset")
    args = ap.parse_args()

    train_df, val_df, test_df, feats, src = get_dataset(fast=False)
    print(f"source: {src}  train queries {train_df.qid.nunique():,}", flush=True)
    slices = group_slices(group_sizes_of(train_df))
    dtr = to_dmatrix(train_df, feats)
    dva = to_dmatrix(val_df, feats)
    dte = to_dmatrix(test_df, feats)

    specs = method_specs()
    if args.only:
        keep = set(args.only.split(","))
        specs = {k: v for k, v in specs.items() if k in keep}
    skip = done_methods()

    for name, (kind, path, params, rounds) in specs.items():
        if name in skip:
            print(f"skip {name} (already in {OUT.name})", flush=True)
            continue
        t0 = time.time()
        obj = _custom_obj(_load_program(ROOT / path), train_df, slices) if kind == "custom" else None
        bst = xgb.train(params, dtr, num_boost_round=rounds, obj=obj)
        dt = time.time() - t0
        rows = []
        for fold, frame, dmat in (("val", val_df, dva), ("test", test_df, dte)):
            m = metrics_bundle(frame, bst.predict(dmat))
            rows.append({"method": name, "fold": fold, **{k: f"{v:.6f}" for k, v in m.items()},
                         "train_seconds": f"{dt:.0f}"})
            print(f"{name:<22} {fold:<5} ndcg={m['ndcg']:.4f} book={m['book_ndcg']:.4f} "
                  f"rev={m['revenue']:.4f} (train {dt:.0f}s)", flush=True)
        append_rows(rows)
    print("done;", OUT, flush=True)


if __name__ == "__main__":
    main()
