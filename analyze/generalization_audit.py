"""Selection-generalization audit: do the programs each search arm SELECTS
generalize, or did the arm select fitness noise?

For every evolution run, walks the checkpoint trajectory (checkpoint_5, _10,
..., final), takes the best program at each point, retrains it on the fast
subset (the search's own training budget), and evaluates BOTH the fast val
fold (what the search saw) and the FULL test fold (60k unseen queries, tight
CIs). The gap between val gain and test gain, per arm, is the noise-chasing
measurement that the reframed paper turns on.

Identical program sources are cached by content hash, so shared checkpoints
cost one training. Appends rows incrementally; safe to relaunch.

    uv run python analyze/generalization_audit.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import pathlib
import sys
import time

import numpy as np
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ltr.dataset import FIXED_PARAMS, FIXED_ROUNDS, get_dataset, to_dmatrix  # noqa: E402
from ltr.metrics import group_sizes_of, group_slices, metrics_bundle  # noqa: E402

OUT = ROOT / "runs" / "summary" / "generalization_audit.csv"
FIELDS = ["run", "checkpoint", "program_sha", "fold", "ndcg", "book_ndcg", "revenue"]

RUN_DIRS = ["mem_on_s0", "mem_on_s1", "mem_on_s2",
            "mem_off_s0", "mem_off_s1", "mem_off_s2",
            "probe_scores_s0", "probe_diag_s0",
            "probe_scores_s1", "probe_diag_s1",
            "probe_scores_s2", "probe_diag_s2"]


def _load_program(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"cand_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def done_keys() -> set:
    if not OUT.exists():
        return set()
    return {(r["run"], r["checkpoint"], r["fold"]) for r in csv.DictReader(open(OUT))}


def main() -> None:
    train_df, val_df, _t, feats, src = get_dataset(fast=True)
    _ft, _fv, full_test, _f, fsrc = get_dataset(fast=False)
    print(f"train: {src} | audit folds: fast val + full test ({fsrc})", flush=True)

    slices = group_slices(group_sizes_of(train_df))
    rel = train_df["rel"].to_numpy(np.float64)
    booking = train_df["booking"].to_numpy(np.float64)
    rev = train_df["rev"].to_numpy(np.float64)
    dtr = to_dmatrix(train_df, feats)
    dva = to_dmatrix(val_df, feats)
    dte = to_dmatrix(full_test, feats)

    skip = done_keys()
    pred_cache: dict = {}   # program sha -> (val_preds, test_preds)

    exists = OUT.exists()
    out_f = open(OUT, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS)
    if not exists:
        writer.writeheader()

    for run in RUN_DIRS:
        cps = sorted((ROOT / "runs" / run / "checkpoints").glob("checkpoint_*"),
                     key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        for cp in cps:
            prog = cp / "best_program.py"
            if not prog.exists():
                continue
            src_text = prog.read_text()
            sha = hashlib.sha1(src_text.encode()).hexdigest()[:10]
            n = cp.name.rsplit("_", 1)[-1]
            if (run, n, "full_test") in skip:
                continue

            if sha not in pred_cache:
                t0 = time.time()
                mod = _load_program(prog)

                def obj(predt, dtrain, m=mod):
                    g, h = m.lambdamart_objective(predt, rel, booking, rev, slices)
                    return np.asarray(g, np.float64), np.asarray(h, np.float64)
                try:
                    bst = xgb.train(FIXED_PARAMS, dtr, num_boost_round=FIXED_ROUNDS, obj=obj)
                    pred_cache[sha] = (bst.predict(dva), bst.predict(dte))
                    print(f"trained {run}/{cp.name} sha={sha} ({time.time()-t0:.0f}s)", flush=True)
                except Exception as ex:
                    print(f"FAILED {run}/{cp.name}: {type(ex).__name__}: {ex}", flush=True)
                    pred_cache[sha] = None
            if pred_cache[sha] is None:
                continue

            vp, tp = pred_cache[sha]
            for fold, frame, preds in (("fast_val", val_df, vp), ("full_test", full_test, tp)):
                m = metrics_bundle(frame, preds)
                writer.writerow({"run": run, "checkpoint": n, "program_sha": sha,
                                 "fold": fold, **{k: f"{v:.6f}" for k, v in m.items()}})
            out_f.flush()

    out_f.close()
    print("done;", OUT, flush=True)


if __name__ == "__main__":
    main()
