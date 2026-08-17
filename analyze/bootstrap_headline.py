"""Formal paired query bootstrap for the paper's two headline comparisons.

Regime A (Table 1, the loop's own regime): pipeline_s2 vs lambdamart_optuna,
both fast-trained, scored on the full 60k-query test fold.
Regime B (Table 2, full scale, capacity-equalized): pipeline_s1_deepcfg vs
lambdamart_optuna_full, both trained on the full train fold.

For each pair: per-query NDCG@10 and booking-NDCG@10 differences, resampled
over queries (10,000 draws) for a one-sided p-value (H1: pipeline > baseline)
and a 95 percent CI; pooled revenue is recomputed per resample from the
per-query realized/ideal building blocks.

    uv run python analyze/bootstrap_headline.py
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

from eval.evaluator_pipeline import TrainStats, _combine, _train_member, _validate_spec  # noqa: E402
from ltr.dataset import get_dataset  # noqa: E402
from ltr.metrics import group_sizes_of, per_query_values  # noqa: E402

N_BOOT = 10_000
RNG = np.random.default_rng(0)


def semantic_view(frame):
    fmap = pd.read_csv(ROOT / "data" / "prepared" / "feature_map.csv")
    sem = frame.rename(columns=dict(zip(fmap["f_index"], fmap["original"])))
    return sem.drop(columns=[c for c in ("rel", "booking", "rev", "random_flag")
                             if c in sem.columns])


def labels_of(frame):
    return {"rel": frame["rel"].to_numpy(np.float64),
            "booking": frame["booking"].to_numpy(np.float64),
            "click": (frame["rel"].to_numpy() >= 1).astype(np.float64),
            "rev": frame["rev"].to_numpy(np.float64)}


def pipeline_predict(program_path, train_frame, test_frame):
    spec = importlib.util.spec_from_file_location(f"bp_{program_path.stem}", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    members, ens = _validate_spec(mod.PIPELINE)
    data = {"train": {"frame": train_frame, "view": semantic_view(train_frame),
                      "labels": labels_of(train_frame)}}
    stats = TrainStats(data["train"]["view"], data["train"]["labels"])
    stats._mode = "train"
    Xtr = mod.build_features(data["train"]["view"].copy(), stats).fillna(0.0).to_numpy(np.float64)
    stats._mode = "apply"
    Xte = mod.build_features(semantic_view(test_frame).copy(), stats).fillna(0.0).to_numpy(np.float64)
    preds = [np.asarray(_train_member(m, Xtr, Xte, data), np.float64) for m in members]
    return _combine(preds, ens, test_frame)


def xgb_predict(params_json, train_frame, test_frame, feats):
    rec = json.loads(pathlib.Path(params_json).read_text())
    bp = dict(rec["extra"]["best_params"]); n = bp.pop("num_boost_round")
    dtr = xgb.DMatrix(train_frame[feats].to_numpy(np.float64),
                      label=train_frame["rel"].to_numpy(np.float64))
    dtr.set_group(group_sizes_of(train_frame))
    bst = xgb.train({"objective": "rank:ndcg", "tree_method": "hist", "seed": 0, **bp},
                    dtr, num_boost_round=n)
    return bst.predict(xgb.DMatrix(test_frame[feats].to_numpy(np.float64)))


def paired_bootstrap(name, test_frame, preds_pipe, preds_base):
    a = per_query_values(test_frame, preds_pipe, k=10)
    b = per_query_values(test_frame, preds_base, k=10)
    out = {"comparison": name}
    nq = len(a["ndcg"])
    idx_all = np.arange(nq)
    for metric in ("ndcg", "book_ndcg"):
        da = np.asarray(a[metric], np.float64)
        db = np.asarray(b[metric], np.float64)
        mask = ~(np.isnan(da) | np.isnan(db))
        diff = (da - db)[mask]
        boots = np.empty(N_BOOT)
        for i in range(N_BOOT):
            boots[i] = diff[RNG.integers(0, len(diff), len(diff))].mean()
        p = float((boots <= 0).mean())
        out[f"{metric}_delta"] = f"{diff.mean():.5f}"
        out[f"{metric}_ci95"] = f"[{np.quantile(boots,0.025):.5f},{np.quantile(boots,0.975):.5f}]"
        out[f"{metric}_p_onesided"] = f"{max(p, 1.0/N_BOOT):.4f}"
    # pooled revenue: resample queries, recompute ratio difference
    ra, ia = np.asarray(a["rev_realized"]), np.asarray(a["rev_ideal"])
    rb = np.asarray(b["rev_realized"])
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = RNG.integers(0, nq, nq)
        denom = ia[s].sum() or 1.0
        boots[i] = (ra[s].sum() - rb[s].sum()) / denom
    p = float((boots <= 0).mean())
    out["revenue_delta"] = f"{(ra.sum()-rb.sum())/(ia.sum() or 1.0):.5f}"
    out["revenue_ci95"] = f"[{np.quantile(boots,0.025):.5f},{np.quantile(boots,0.975):.5f}]"
    out["revenue_p_onesided"] = f"{max(p, 1.0/N_BOOT):.4f}"
    print(out, flush=True)
    return out


def main() -> None:
    fast_train, _fv, _ft, feats, _s = get_dataset(fast=True)
    full_train, _v2, full_test, feats_full, _s2 = get_dataset(fast=False)
    if str(_s).startswith("synthetic") or str(_s2).startswith("synthetic"):
        sys.exit("refusing to run on the synthetic fallback; prepare data/prepared first")
    print(f"test fold: {full_test.qid.nunique():,} queries", flush=True)
    rows = []

    # Regime A: fast-train (Table 2 headline)
    pa = pipeline_predict(ROOT / "discovered" / "pipeline_s2_best.py", fast_train, full_test)
    ba = xgb_predict(ROOT / "runs" / "lambdamart_optuna.json", fast_train, full_test, feats)
    rows.append(paired_bootstrap("pipeline_s2_vs_optuna_fastregime", full_test, pa, ba))

    # Regime B: full-scale config-equalized (Table 3 headline)
    pb = pipeline_predict(ROOT / "discovered" / "pipeline_s1_deepcfg.py", full_train, full_test)
    bb = xgb_predict(ROOT / "runs" / "lambdamart_optuna_full.json", full_train, full_test, feats_full)
    rows.append(paired_bootstrap("deepcfg_vs_optuna_full_fullscale", full_test, pb, bb))

    out = ROOT / "runs" / "summary" / "bootstrap_headline.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
