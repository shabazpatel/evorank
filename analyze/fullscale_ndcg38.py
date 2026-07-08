"""Full-scale comparison for the paper: train on the FULL train fold (280k
queries, 6.9M rows), evaluate the full test fold (60k queries), report both
NDCG@10 (our protocol) and NDCG@38 (the ICDM 2013 challenge metric).

Caveat that must accompany any use of these numbers: the ICDM leaderboard
test set is hidden and unreproducible; this is a comparable-protocol
approximation on our own held-out split, never a leaderboard claim.

Methods: seed pipeline, lambdamart_optuna (stored params), pipeline_s0/s1/s2.
Appends incrementally to runs/summary/fullscale_ndcg38.csv; resume-safe.

    uv run python analyze/fullscale_ndcg38.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.evaluator_pipeline import TrainStats, _combine, _train_member, _validate_spec  # noqa: E402
from ltr.dataset import get_dataset  # noqa: E402
from ltr.metrics import group_sizes_of, metrics_bundle  # noqa: E402

OUT = ROOT / "runs" / "summary" / "fullscale_ndcg38.csv"
FIELDS = ["method", "ndcg10", "ndcg38", "book_ndcg10", "revenue10", "train_seconds"]


def semantic(frame):
    fmap = pd.read_csv(ROOT / "data" / "prepared" / "feature_map.csv")
    sem = frame.rename(columns=dict(zip(fmap["f_index"], fmap["original"])))
    view = sem.drop(columns=[c for c in ("rel", "booking", "rev", "random_flag")
                             if c in sem.columns])
    return view


def done() -> set:
    if not OUT.exists():
        return set()
    return {r["method"] for r in csv.DictReader(open(OUT))}


def append(row):
    exists = OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    full_train, _fv, full_test, _f, _s = get_dataset(fast=False)
    print(f"train {full_train.qid.nunique():,} queries / test {full_test.qid.nunique():,}",
          flush=True)
    data = {"train": {"frame": full_train, "view": semantic(full_train),
                      "labels": {"rel": full_train["rel"].to_numpy(np.float64),
                                 "booking": full_train["booking"].to_numpy(np.float64),
                                 "click": (full_train["rel"].to_numpy() >= 1).astype(np.float64),
                                 "rev": full_train["rev"].to_numpy(np.float64)}}}
    test_view = semantic(full_test)
    skip = done()

    def score(name, preds, dt):
        m10 = metrics_bundle(full_test, preds, k=10)
        m38 = metrics_bundle(full_test, preds, k=38)
        row = {"method": name, "ndcg10": f"{m10['ndcg']:.6f}", "ndcg38": f"{m38['ndcg']:.6f}",
               "book_ndcg10": f"{m10['book_ndcg']:.6f}", "revenue10": f"{m10['revenue']:.6f}",
               "train_seconds": f"{dt:.0f}"}
        append(row)
        print(f"{name:<20} NDCG@10 {m10['ndcg']:.4f}  NDCG@38 {m38['ndcg']:.4f} "
              f"(train {dt:.0f}s)", flush=True)

    # optuna baseline (raw features, stored tuned params)
    if "lambdamart_optuna" not in skip:
        t0 = time.time()
        opt = json.loads((ROOT / "runs" / "lambdamart_optuna.json").read_text())
        bp = dict(opt["extra"]["best_params"]); n = bp.pop("num_boost_round")
        Xtr = data["train"]["view"].drop(columns=["qid"]).to_numpy(np.float64)
        dtr = xgb.DMatrix(Xtr, label=data["train"]["labels"]["rel"])
        dtr.set_group(group_sizes_of(full_train))
        bst = xgb.train({"objective": "rank:ndcg", "tree_method": "hist", "seed": 0, **bp},
                        dtr, num_boost_round=n)
        preds = bst.predict(xgb.DMatrix(test_view.drop(columns=["qid"]).to_numpy(np.float64)))
        score("lambdamart_optuna", preds, time.time() - t0)

    programs = {"seed_pipeline": ROOT / "seed" / "initial_pipeline.py"}
    for run in ("pipeline_s0", "pipeline_s1", "pipeline_s2"):
        cps = sorted((ROOT / "runs" / run / "checkpoints").glob("checkpoint_*"),
                     key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        programs[run] = cps[-1] / "best_program.py"

    for name, prog in programs.items():
        if name in skip:
            print(f"skip {name}", flush=True)
            continue
        t0 = time.time()
        spec = importlib.util.spec_from_file_location(f"fs_{name}", prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        members, ens = _validate_spec(mod.PIPELINE)
        stats = TrainStats(data["train"]["view"], data["train"]["labels"])
        stats._mode = "train"
        Xtr = mod.build_features(data["train"]["view"].copy(), stats).fillna(0.0).to_numpy(np.float64)
        stats._mode = "apply"
        Xte = mod.build_features(test_view.copy(), stats).fillna(0.0).to_numpy(np.float64)
        preds = []
        for m in members:
            p = np.asarray(_train_member(m, Xtr, Xte, data), np.float64)
            preds.append(p)
            print(f"  {name} member {m['family']}/{m['objective']} done "
                  f"({time.time()-t0:.0f}s cumulative)", flush=True)
        final = _combine(preds, ens, full_test)
        score(name, final, time.time() - t0)

    print("done;", OUT, flush=True)


if __name__ == "__main__":
    main()
