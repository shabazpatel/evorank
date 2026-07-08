"""Transfer audit for discovered pipelines: fast-train, full-test-fold scoring.

For each pipeline run's final best program (plus the seed pipeline and the
Optuna-tuned raw-feature baseline as references): build features on the fast
train fold (mode train) and on the FULL test fold (mode apply, train stats
only), train the members on fast train, ensemble, and score 60k unseen
queries. The val-vs-test comparison is the noise-chasing measurement.

    uv run python analyze/pipeline_transfer.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.evaluator_pipeline import (TrainStats, _combine, _semantic_frames,  # noqa: E402
                                     _train_member, _validate_spec)
from ltr.dataset import get_dataset  # noqa: E402
from ltr.metrics import metrics_bundle  # noqa: E402

RUNS = ["pipeline_s0", "pipeline_s1", "pipeline_s2"]


def full_test_semantic():
    _t, _v, full_test, _f, _s = get_dataset(fast=False)
    fmap = pd.read_csv(ROOT / "data" / "prepared" / "feature_map.csv")
    sem = full_test.rename(columns=dict(zip(fmap["f_index"], fmap["original"])))
    view = sem.drop(columns=[c for c in ("rel", "booking", "rev", "random_flag")
                             if c in sem.columns])
    return full_test, view


def run_pipeline(program_path: pathlib.Path, data, test_frame, test_view):
    spec = importlib.util.spec_from_file_location(f"p_{program_path.stem}", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    members, ens = _validate_spec(mod.PIPELINE)
    tr, va = data["train"], data["val"]
    stats = TrainStats(tr["view"], tr["labels"])

    stats._mode = "train"
    Xtr = mod.build_features(tr["view"].copy(), stats).fillna(0.0).to_numpy(np.float64)
    stats._mode = "apply"
    Xva = mod.build_features(va["view"].copy(), stats).fillna(0.0).to_numpy(np.float64)
    Xte = mod.build_features(test_view.copy(), stats).fillna(0.0).to_numpy(np.float64)

    val_preds, test_preds = [], []
    for m in members:
        val_preds.append(np.asarray(_train_member(m, Xtr, Xva, data), np.float64))
        test_preds.append(np.asarray(_train_member(m, Xtr, Xte, data), np.float64))
    val_final = _combine(val_preds, ens, va["frame"])
    test_final = _combine(test_preds, ens, test_frame)
    return metrics_bundle(va["frame"], val_final), metrics_bundle(test_frame, test_final)


def main() -> None:
    data = _semantic_frames()
    test_frame, test_view = full_test_semantic()
    print(f"full test fold: {test_frame.qid.nunique():,} queries\n", flush=True)

    rows = []

    def report(name, mv, mt):
        rows.append({"name": name, **{f"val_{k}": v for k, v in mv.items()},
                     **{f"test_{k}": v for k, v in mt.items()}})
        print(f"{name:<22} val ndcg {mv['ndcg']:.4f} | TEST ndcg {mt['ndcg']:.4f} "
              f"book {mt['book_ndcg']:.4f} rev {mt['revenue']:.4f}", flush=True)

    # references
    mv, mt = run_pipeline(ROOT / "seed" / "initial_pipeline.py", data, test_frame, test_view)
    report("seed_pipeline", mv, mt)
    seed_val, seed_test = mv["ndcg"], mt["ndcg"]

    opt = json.loads((ROOT / "runs" / "lambdamart_optuna.json").read_text())
    bp = dict(opt["extra"]["best_params"]); n = bp.pop("num_boost_round")
    tr, va = data["train"], data["val"]
    Xtr = tr["view"].drop(columns=["qid"]).to_numpy(np.float64)
    dtr = xgb.DMatrix(Xtr, label=tr["labels"]["rel"])
    from ltr.metrics import group_sizes_of
    dtr.set_group(group_sizes_of(tr["frame"]))
    bst = xgb.train({"objective": "rank:ndcg", "tree_method": "hist", "seed": 0, **bp},
                    dtr, num_boost_round=n)
    mv = metrics_bundle(va["frame"], bst.predict(
        xgb.DMatrix(va["view"].drop(columns=["qid"]).to_numpy(np.float64))))
    mt = metrics_bundle(test_frame, bst.predict(
        xgb.DMatrix(test_view.drop(columns=["qid"]).to_numpy(np.float64))))
    report("lambdamart_optuna", mv, mt)

    # discovered pipelines
    for run in RUNS:
        cps = sorted((ROOT / "runs" / run / "checkpoints").glob("checkpoint_*"),
                     key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        prog = cps[-1] / "best_program.py"
        mv, mt = run_pipeline(prog, data, test_frame, test_view)
        report(run, mv, mt)

    print(f"\nseed reference: val {seed_val:.4f} test {seed_test:.4f}")
    print("gains vs seed (val -> test):")
    for r in rows[2:]:
        print(f"  {r['name']:<20} {r['val_ndcg']-seed_val:+.4f} -> {r['test_ndcg']-seed_test:+.4f}")

    out = ROOT / "runs" / "summary" / "pipeline_transfer.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
