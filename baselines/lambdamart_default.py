"""Baseline: XGBoost built-in rank:ndcg, out of the box.

Same fixed training config as the evolution loop's evaluator, so the only
difference vs the seed program is built-in C++ objective vs our Python one.
Logs the three metrics on the fast-subset val fold to runs/.
"""
from __future__ import annotations

import pathlib
import sys

import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines._common import log_run, score  # noqa: E402
from ltr.dataset import FIXED_ROUNDS, get_dataset, to_dmatrix  # noqa: E402


def main() -> None:
    train_df, val_df, _test, feats, src = get_dataset(fast=True)
    params = {
        "objective": "rank:ndcg",
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 0.1,
        "tree_method": "hist",
        "seed": 0,
    }
    bst = xgb.train(params, to_dmatrix(train_df, feats), num_boost_round=FIXED_ROUNDS)
    preds = bst.predict(to_dmatrix(val_df, feats))
    log_run("lambdamart_default", score(val_df, preds), extra={"source": src})


if __name__ == "__main__":
    main()
