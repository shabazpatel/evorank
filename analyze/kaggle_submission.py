"""Build Kaggle late-submission files for the original ICDM 2013 competition.

Trains on the FULL train.csv (all 399k queries, nothing held out, matching
what 2013 entrants had) and predicts the official test.csv (266k queries,
hidden labels). Produces two submissions in the required SearchId,PropertyId
format (rows rank-ordered within each search):

  runs/summary/kaggle_submission_pipeline.csv   discovered pipeline (deepcfg)
  runs/summary/kaggle_submission_baseline.csv   full-scale Optuna LambdaMART

Uploading these to the competition's late-submission page scores them with
the official NDCG@38 on the actual leaderboard test set.

    uv run python analyze/kaggle_submission.py
"""
from __future__ import annotations

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

from data.prepare_expedia import _build_labels, _select_features  # noqa: E402
from eval.evaluator_pipeline import TrainStats, _combine, _train_member, _validate_spec  # noqa: E402
from ltr.metrics import group_sizes_of  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "runs" / "summary"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_frames():
    log("reading train.csv (9.9M rows)")
    train = pd.read_csv(RAW / "train.csv")
    train = train.sort_values("srch_id", kind="stable").reset_index(drop=True)
    labels_df = _build_labels(train)
    Xtr, names = _select_features(train)
    train_view = Xtr.astype(np.float32)
    train_view["qid"] = train["srch_id"].to_numpy()
    labels = {"rel": labels_df["rel"].to_numpy(np.float64),
              "booking": labels_df["booking"].to_numpy(np.float64),
              "click": (labels_df["rel"].to_numpy() >= 1).astype(np.float64),
              "rev": labels_df["rev"].to_numpy(np.float64)}
    frame = pd.DataFrame({"qid": train["qid" if "qid" in train else "srch_id"].to_numpy(),
                          "rel": labels["rel"]})
    del train, Xtr

    log("reading test.csv (6.6M rows)")
    test = pd.read_csv(RAW / "test.csv")
    test = test.sort_values("srch_id", kind="stable").reset_index(drop=True)
    Xte, names_te = _select_features(test)
    assert names == names_te, f"feature mismatch: {set(names) ^ set(names_te)}"
    test_view = Xte.astype(np.float32)
    test_view["qid"] = test["srch_id"].to_numpy()
    test_ids = test[["srch_id", "prop_id"]].copy()
    del test, Xte
    return train_view, labels, frame, test_view, test_ids


def write_submission(test_ids: pd.DataFrame, scores: np.ndarray, path: pathlib.Path) -> None:
    sub = test_ids.copy()
    sub["score"] = scores
    sub = sub.sort_values(["srch_id", "score"], ascending=[True, False], kind="stable")
    sub[["srch_id", "prop_id"]].rename(
        columns={"srch_id": "SearchId", "prop_id": "PropertyId"}
    ).to_csv(path, index=False)
    log(f"wrote {path} ({len(sub):,} rows)")


def main() -> None:
    train_view, labels, frame, test_view, test_ids = load_frames()

    # --- baseline: full-scale Optuna LambdaMART on raw features ---
    log("training baseline (optuna_full params, raw features)")
    rec = json.loads((ROOT / "runs" / "lambdamart_optuna_full.json").read_text())
    bp = dict(rec["extra"]["best_params"]); n = bp.pop("num_boost_round")
    feat_cols = [c for c in train_view.columns if c != "qid"]
    dtr = xgb.DMatrix(train_view[feat_cols].to_numpy(np.float32), label=labels["rel"])
    dtr.set_group(group_sizes_of(frame))
    bst = xgb.train({"objective": "rank:ndcg", "tree_method": "hist", "seed": 0, **bp},
                    dtr, num_boost_round=n)
    del dtr
    preds = bst.predict(xgb.DMatrix(test_view[feat_cols].to_numpy(np.float32)))
    write_submission(test_ids, preds, OUT / "kaggle_submission_baseline.csv")
    del bst, preds

    # --- discovered pipeline (config-equalized deepcfg) ---
    log("building pipeline features (train)")
    prog = ROOT / "discovered" / "pipeline_s1_deepcfg.py"
    spec = importlib.util.spec_from_file_location("kg", prog)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    members, ens = _validate_spec(mod.PIPELINE)
    data = {"train": {"frame": frame, "view": train_view, "labels": labels}}
    stats = TrainStats(train_view, labels)
    stats._mode = "train"
    Xtr = mod.build_features(train_view.copy(), stats).fillna(0.0).to_numpy(np.float32)
    stats._mode = "apply"
    log("building pipeline features (test)")
    Xte = mod.build_features(test_view.copy(), stats).fillna(0.0).to_numpy(np.float32)
    preds = []
    for m in members:
        log(f"training member {m['family']}/{m['objective']}")
        preds.append(np.asarray(_train_member(m, Xtr, Xte, data), np.float64))
    test_frame = pd.DataFrame({"qid": test_view["qid"].to_numpy()})
    final = _combine(preds, ens, test_frame)
    write_submission(test_ids, final, OUT / "kaggle_submission_pipeline.csv")
    log("done")


if __name__ == "__main__":
    main()
