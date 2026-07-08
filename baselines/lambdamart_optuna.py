"""Baseline: XGBoost rank:ndcg with Optuna-tuned hyperparameters (REQUIRED).

This is the bar the discovered objectives must beat. Beating only the default
config is not a result. Uses the same search space and budget that
retune_best.py later grants the best discovered objective, so the comparison
is symmetric. Tunes on val ndcg with TPE, seed 0.

    python baselines/lambdamart_optuna.py --trials 40
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines._common import log_run, score  # noqa: E402
from ltr.dataset import get_dataset, to_dmatrix  # noqa: E402
from ltr.metrics import metrics_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=300, help="upper bound for num_boost_round")
    ap.add_argument("--full", action="store_true",
                    help="tune on the FULL train/val folds (strict full-scale baseline)")
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, val_df, _test, feats, src = get_dataset(fast=not args.full)
    dtr = to_dmatrix(train_df, feats)
    dva = to_dmatrix(val_df, feats)

    # Same space as retune_best.py, with the built-in objective instead of a
    # custom one (base_score/disable_default_eval_metric not needed here).
    def objective(trial):
        params = {
            "objective": "rank:ndcg",
            "tree_method": "hist",
            "seed": 0,
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        }
        n = trial.suggest_int("num_boost_round", 50, args.rounds)
        bst = xgb.train(params, dtr, num_boost_round=n)
        return metrics_bundle(val_df, bst.predict(dva))["ndcg"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = dict(study.best_params)
    n = best.pop("num_boost_round")
    params = {"objective": "rank:ndcg", "tree_method": "hist", "seed": 0, **best}
    bst = xgb.train(params, dtr, num_boost_round=n)
    preds = bst.predict(dva)

    log_run("lambdamart_optuna_full" if args.full else "lambdamart_optuna", score(val_df, preds),
            extra={"source": src, "trials": args.trials,
                   "best_params": {**best, "num_boost_round": n}})


if __name__ == "__main__":
    main()
