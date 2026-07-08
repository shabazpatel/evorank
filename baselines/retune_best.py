"""Post-hoc retune of a discovered objective (methodology control).

The fixed training hyperparameters during the search are conservative against
EvoRank: the Optuna baseline gets tuned params while candidates do not. This
script closes that asymmetry after the search: it takes the best discovered
program and gives it the SAME Optuna budget the tuned baseline received. If the
discovered objective's advantage survives re-tuning, the gain is attributable to
the objective, not to an artifact of the fixed config.

    python baselines/retune_best.py --program runs/<best_program>.py --trials 40
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines._common import log_run, score            # noqa: E402
from ltr.dataset import get_dataset, to_dmatrix          # noqa: E402
from ltr.metrics import group_sizes_of, group_slices, metrics_bundle  # noqa: E402


def _load_program(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True, help="path to the discovered program (.py with lambdamart_objective)")
    ap.add_argument("--trials", type=int, default=40, help="use the SAME trial count as lambdamart_optuna")
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        raise SystemExit("optuna not installed. Run: pip install optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mod = _load_program(args.program)
    if not hasattr(mod, "lambdamart_objective"):
        raise SystemExit("program has no lambdamart_objective")

    train_df, val_df, _test, feats, src = get_dataset(fast=True)
    slices = group_slices(group_sizes_of(train_df))
    dtr = to_dmatrix(train_df, feats)
    dva = to_dmatrix(val_df, feats)

    rel = train_df["rel"].to_numpy(dtype=np.float64)
    booking = train_df["booking"].to_numpy(dtype=np.float64)
    rev = train_df["rev"].to_numpy(dtype=np.float64)

    def make_obj():
        def obj(predt, dtrain):
            g, h = mod.lambdamart_objective(predt, rel, booking, rev, slices)
            g = np.asarray(g, dtype=np.float64)
            h = np.asarray(h, dtype=np.float64)
            if not (np.all(np.isfinite(g)) and np.all(np.isfinite(h))):
                raise ValueError("non-finite grad/hess")
            return g, h
        return obj

    def objective(trial):
        params = {
            "tree_method": "hist",
            "seed": 0,
            "base_score": 0.0,
            "disable_default_eval_metric": 1,
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        }
        n = trial.suggest_int("num_boost_round", 50, args.rounds)
        try:
            bst = xgb.train(params, dtr, num_boost_round=n, obj=make_obj())
            return metrics_bundle(val_df, bst.predict(dva))["ndcg"]
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = dict(study.best_params)
    n = best.pop("num_boost_round")
    params = {"tree_method": "hist", "seed": 0, "base_score": 0.0,
              "disable_default_eval_metric": 1, **best}
    bst = xgb.train(params, dtr, num_boost_round=n, obj=make_obj())
    preds = bst.predict(dva)

    name = f"posthoc_retune_{pathlib.Path(args.program).stem}"
    log_run(name, score(val_df, preds),
            extra={"source": src, "program": args.program, "trials": args.trials,
                   "best_params": {**best, "num_boost_round": n}})


if __name__ == "__main__":
    main()
