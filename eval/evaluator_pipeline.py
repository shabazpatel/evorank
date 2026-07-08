"""SkyDiscover evaluator for EvoRank-Pipeline (Part 2): feature construction,
model selection, loss selection, and ensemble architecture.

Contract: evaluate(program_path) -> dict with the three metrics,
combined_score, and stage-attributed artifacts.feedback. The evaluator owns
everything unsafe: data, semantic column names, leakage-guarded train-fold
statistics, whitelisted model zoo, ensembling, budgets, and rejection.
Design: paper/pipeline_design.md.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import xgboost as xgb  # noqa: E402

from ltr.dataset import get_dataset  # noqa: E402
from ltr.metrics import (group_sizes_of, group_slices, metrics_bundle,  # noqa: E402
                         per_query_values)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "runs" / "_cache"

LABEL_COLS = ("rel", "booking", "rev", "random_flag")
WALL_BUDGET_S = 150.0
MAX_MEMBERS = 3
MAX_FEATURES = 300
N_BOOT = 300

ALLOWED = {
    "xgb": {
        "objectives": {"rank:ndcg", "rank:pairwise", "reg:squarederror",
                       "binary:logistic", "custom:seed", "custom:simplified"},
        "params": {"eta": (0.005, 0.5), "max_depth": (2, 12),
                   "min_child_weight": (0.0, 50.0), "subsample": (0.4, 1.0),
                   "colsample_bytree": (0.4, 1.0), "gamma": (0.0, 5.0),
                   "lambda": (0.0, 50.0), "num_boost_round": (10, 300)},
    },
    "lgbm": {
        "objectives": {"lambdarank", "regression", "binary"},
        "params": {"learning_rate": (0.005, 0.5), "num_leaves": (7, 255),
                   "min_child_samples": (1, 500), "n_estimators": (10, 300),
                   "subsample": (0.4, 1.0), "colsample_bytree": (0.4, 1.0)},
    },
    "sklearn": {
        "objectives": {"logistic", "rf", "ert", "ridge"},
        "params": {"n_estimators": (10, 150), "max_depth": (2, 24),
                   "min_samples_leaf": (1, 200), "alpha": (1e-4, 100.0),
                   "C": (1e-3, 100.0)},
    },
}

CUSTOM_OBJECTIVE_PATHS = {
    "custom:seed": ROOT / "seed" / "initial_program.py",
    "custom:simplified": ROOT / "runs" / "summary" / "mechanism_ablation" / "no_adaptive.py",
}


class StageError(Exception):
    def __init__(self, stage: str, msg: str):
        super().__init__(f"rejected at {stage}: {msg}")
        self.stage = stage


# ------------------------------------------------------------------ data

_DATA_CACHE: dict = {}


def _semantic_frames():
    """Fast-subset frames with semantic column names; label columns separate."""
    if "frames" in _DATA_CACHE:
        return _DATA_CACHE["frames"]
    train_df, val_df, _test, feats, _src = get_dataset(fast=True)
    fmap = pd.read_csv(ROOT / "data" / "prepared" / "feature_map.csv")
    rename = dict(zip(fmap["f_index"], fmap["original"]))
    out = {}
    for name, frame in (("train", train_df), ("val", val_df)):
        sem = frame.rename(columns=rename)
        labels = {
            "rel": frame["rel"].to_numpy(np.float64),
            "booking": frame["booking"].to_numpy(np.float64),
            "click": (frame["rel"].to_numpy() >= 1).astype(np.float64),
            "rev": frame["rev"].to_numpy(np.float64),
        }
        program_view = sem.drop(columns=[c for c in LABEL_COLS if c in sem.columns])
        out[name] = {"frame": frame, "view": program_view, "labels": labels}
    _DATA_CACHE["frames"] = out
    return out


class TrainStats:
    """Leakage-safe aggregate helper, built from the TRAIN fold only.

    Leakage defenses on rate(), each generation measured before the next:
      1. Statistics never touch val/test labels (train-fold only, smoothed
         toward the train prior, unseen values fall back to the prior).
         Alone this still self-leaks on the train fold (measured -0.10 ndcg).
      2. Leave-one-out was tried and is WORSE in a subtle way: subtracting the
         row's own label creates a tiny per-row offset perfectly correlated
         with that label (booked rows encode lower by 1/(C-1+alpha)), which
         hist-based trees read as a label detector (measured -0.13 on a
         near-constant site CTR column).
      3. Adopted: K-fold out-of-fold encoding, folded BY QUERY (mode "train",
         set by the evaluator): each train row's encoding is computed from the
         other folds only, so no function of the row's own label, or its
         query's labels, enters its feature value."""

    N_FOLDS = 5

    ALPHA = 20.0

    def __init__(self, train_view: pd.DataFrame, labels: Dict[str, np.ndarray]):
        self._train = train_view
        self._labels = labels
        self._maps: dict = {}
        self._stacked_model = None
        self._mode = "apply"  # evaluator sets "train" while building train features

    def count(self, df: pd.DataFrame, col: str) -> pd.Series:
        key = ("count", col)
        if key not in self._maps:
            self._maps[key] = self._train[col].value_counts()
        return df[col].map(self._maps[key]).fillna(0.0).astype(float)

    MIN_ROWS_PER_VALUE = 8.0  # rate() sparsity floor on this fold

    def rate(self, df: pd.DataFrame, col: str, target: str = "booking") -> pd.Series:
        if target not in ("booking", "click"):
            raise StageError("features", f"rate target must be booking or click, got {target}")
        key = ("rate", col, target)
        if key not in self._maps:
            y = pd.Series(self._labels[target], index=self._train.index)
            grp = y.groupby(self._train[col])
            prior = float(y.mean())
            sums, counts = grp.sum(), grp.count()
            avg = float(counts.mean())
            if avg < self.MIN_ROWS_PER_VALUE:
                raise StageError(
                    "features",
                    f"rate({col!r}) is too sparse on this training fold "
                    f"(avg {avg:.1f} rows per value, need >= {self.MIN_ROWS_PER_VALUE:.0f}); "
                    f"target-rate encodings need history. Use stats.count({col!r}) instead, "
                    f"or rate() on a lower-cardinality column such as prop_country_id, "
                    f"site_id, or srch_destination_id")
            smoothed = (sums + self.ALPHA * prior) / (counts + self.ALPHA)
            # per-fold maps for out-of-fold train encoding, folded by query so
            # no row sees any statistic that includes its own query's labels
            folds = (self._train["qid"].astype(np.int64) % self.N_FOLDS
                     if "qid" in self._train.columns
                     else pd.Series(np.arange(len(self._train)) % self.N_FOLDS,
                                    index=self._train.index))
            fold_maps = []
            for k in range(self.N_FOLDS):
                mask = folds != k
                yk = y[mask]
                grpk = yk.groupby(self._train.loc[mask, col])
                sk = (grpk.sum() + self.ALPHA * prior) / (grpk.count() + self.ALPHA)
                fold_maps.append(sk)
            self._maps[key] = (smoothed, fold_maps, folds, prior)
        smoothed, fold_maps, folds, prior = self._maps[key]

        if self._mode == "train" and df.index.equals(self._train.index):
            out = pd.Series(np.nan, index=df.index)
            for k in range(self.N_FOLDS):
                mask = (folds == k).to_numpy() if hasattr(folds, "to_numpy") else folds == k
                out[mask] = df.loc[mask, col].map(fold_maps[k])
            return out.fillna(prior).astype(float)
        return df[col].map(smoothed).fillna(prior).astype(float)

    def quantile(self, col: str, q: float) -> float:
        return float(self._train[col].quantile(q))

    def stacked_score(self, df: pd.DataFrame) -> pd.Series:
        """Ridge score on visitor and query columns, the mechanism behind the
        ICDM winners' fm_score feature. Trained on the train fold only."""
        cols = [c for c in self._train.columns
                if c.startswith(("visitor_", "srch_")) and
                pd.api.types.is_numeric_dtype(self._train[c])]
        if self._stacked_model is None:
            from sklearn.linear_model import Ridge
            X = self._train[cols].fillna(0.0).to_numpy(np.float64)
            m = Ridge(alpha=1.0)
            m.fit(X, self._labels["rel"])
            self._stacked_model = (m, cols)
        m, cols = self._stacked_model
        return pd.Series(m.predict(df[cols].fillna(0.0).to_numpy(np.float64)),
                         index=df.index)


# ------------------------------------------------------------------ models

def _clamp_params(family: str, params: dict) -> dict:
    allowed = ALLOWED[family]["params"]
    out = {}
    for k, v in (params or {}).items():
        if k not in allowed:
            continue
        lo, hi = allowed[k]
        try:
            out[k] = min(max(type(lo)(v), lo), hi)
        except (TypeError, ValueError):
            raise StageError("spec", f"bad param {k}={v!r} for {family}")
    return out


def _validate_spec(pipeline: dict) -> Tuple[List[dict], dict]:
    if not isinstance(pipeline, dict):
        raise StageError("spec", "PIPELINE must be a dict")
    models = pipeline.get("models")
    if not isinstance(models, list) or not (1 <= len(models) <= MAX_MEMBERS):
        raise StageError("spec", f"models must be a list of 1..{MAX_MEMBERS}")
    validated = []
    for i, m in enumerate(models):
        fam, objective = m.get("family"), m.get("objective")
        if fam not in ALLOWED:
            raise StageError("spec", f"member {i}: unknown family {fam!r}")
        if objective not in ALLOWED[fam]["objectives"]:
            raise StageError("spec", f"member {i}: objective {objective!r} not allowed for {fam}")
        validated.append({"family": fam, "objective": objective,
                          "params": _clamp_params(fam, m.get("params", {}))})
    ens = pipeline.get("ensemble", {"type": "single"})
    etype = ens.get("type", "single")
    if etype not in ("single", "zscore_weighted", "rank_mean"):
        raise StageError("spec", f"unknown ensemble type {etype!r}")
    if etype == "single" and len(validated) > 1:
        etype, ens = "zscore_weighted", {"type": "zscore_weighted"}
    if etype == "zscore_weighted":
        w = ens.get("weights") or [1.0] * len(validated)
        if len(w) != len(validated) or not all(np.isfinite(w)) or min(w) < 0 or sum(w) <= 0:
            raise StageError("spec", "bad ensemble weights")
        ens = {"type": etype, "weights": [float(x) / sum(w) for x in w]}
    else:
        ens = {"type": etype}
    return validated, ens


def _load_custom_objective(name: str, train_frame: pd.DataFrame):
    path = CUSTOM_OBJECTIVE_PATHS[name]
    spec = importlib.util.spec_from_file_location(f"cobj_{uuid.uuid4().hex}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    slices = group_slices(group_sizes_of(train_frame))
    rel = train_frame["rel"].to_numpy(np.float64)
    booking = train_frame["booking"].to_numpy(np.float64)
    rev = train_frame["rev"].to_numpy(np.float64)

    def obj(predt, dtrain):
        g, h = mod.lambdamart_objective(predt, rel, booking, rev, slices)
        return np.asarray(g, np.float64), np.asarray(h, np.float64)
    return obj


def _train_member(member: dict, Xtr: np.ndarray, Xva: np.ndarray,
                  data) -> np.ndarray:
    fam, objective = member["family"], member["objective"]
    p = dict(member["params"])
    tr = data["train"]
    groups = group_sizes_of(tr["frame"])

    if fam == "xgb":
        rounds = int(p.pop("num_boost_round", 120))
        base = {"tree_method": "hist", "seed": 0, **p}
        y = tr["labels"]["booking"] if objective == "binary:logistic" else tr["labels"]["rel"]
        dtr = xgb.DMatrix(Xtr, label=y)
        dva = xgb.DMatrix(Xva)
        custom = objective.startswith("custom:")
        if custom:
            base.update({"base_score": 0.0, "disable_default_eval_metric": 1})
            dtr.set_group(groups)
            bst = xgb.train(base, dtr, num_boost_round=rounds,
                            obj=_load_custom_objective(objective, tr["frame"]))
        else:
            base["objective"] = objective
            if objective.startswith("rank:"):
                dtr.set_group(groups)
            bst = xgb.train(base, dtr, num_boost_round=rounds)
        return bst.predict(dva)

    if fam == "lgbm":
        import lightgbm as lgb
        n = int(p.pop("n_estimators", 120))
        common = dict(n_estimators=n, random_state=0, verbose=-1,
                      n_jobs=int(os.environ.get("OMP_NUM_THREADS", "4")), **p)
        if objective == "lambdarank":
            m = lgb.LGBMRanker(objective="lambdarank",
                               label_gain=[2 ** i - 1 for i in range(32)], **common)
            m.fit(Xtr, tr["labels"]["rel"].astype(int), group=groups)
            return m.predict(Xva)
        if objective == "regression":
            m = lgb.LGBMRegressor(**common)
            m.fit(Xtr, tr["labels"]["rel"])
            return m.predict(Xva)
        m = lgb.LGBMClassifier(**common)
        m.fit(Xtr, tr["labels"]["booking"].astype(int))
        return m.predict_proba(Xva)[:, 1]

    # sklearn family
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    n_jobs = int(os.environ.get("OMP_NUM_THREADS", "4"))
    if objective == "ridge":
        m = Ridge(alpha=float(p.get("alpha", 1.0)))
        m.fit(Xtr, tr["labels"]["rel"])
        return m.predict(Xva)
    if objective == "logistic":
        m = LogisticRegression(C=float(p.get("C", 1.0)), max_iter=200, n_jobs=n_jobs)
        m.fit(Xtr, tr["labels"]["booking"].astype(int))
        return m.predict_proba(Xva)[:, 1]
    cls = RandomForestRegressor if objective == "rf" else ExtraTreesRegressor
    m = cls(n_estimators=int(p.get("n_estimators", 100)),
            max_depth=p.get("max_depth"), min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            n_jobs=n_jobs, random_state=0)
    m.fit(Xtr, tr["labels"]["rel"])
    return m.predict(Xva)


def _combine(preds: List[np.ndarray], ens: dict, val_frame: pd.DataFrame) -> np.ndarray:
    if len(preds) == 1:
        return preds[0]
    gs = group_slices(group_sizes_of(val_frame))
    if ens["type"] == "rank_mean":
        out = np.zeros_like(preds[0], dtype=np.float64)
        for p in preds:
            for s, e in gs:
                order = np.argsort(np.argsort(p[s:e]))
                out[s:e] += order / max(e - s - 1, 1)
        return out / len(preds)
    # zscore_weighted
    w = ens.get("weights", [1.0 / len(preds)] * len(preds))
    out = np.zeros_like(preds[0], dtype=np.float64)
    for wi, p in zip(w, preds):
        z = np.empty_like(p, dtype=np.float64)
        for s, e in gs:
            seg = p[s:e]
            sd = seg.std()
            z[s:e] = (seg - seg.mean()) / (sd if sd > 1e-12 else 1.0)
        out += wi * z
    return out


# ------------------------------------------------------------- seed reference

def _seed_reference(data) -> np.ndarray:
    """Val predictions of the seed pipeline, cached on disk."""
    tr, va = data["train"], data["val"]
    sig = f"pipeline:{len(tr['frame'])}:{len(va['frame'])}:{tr['frame'].rev.sum():.2f}"
    cache = CACHE_DIR / f"seed_pipeline_{hashlib.sha1(sig.encode()).hexdigest()[:12]}.npz"
    if cache.exists():
        return np.load(cache)["preds"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Xtr = tr["view"].drop(columns=["qid"]).to_numpy(np.float64)
    Xva = va["view"].drop(columns=["qid"]).to_numpy(np.float64)
    member = {"family": "xgb", "objective": "rank:ndcg",
              "params": {"eta": 0.1, "max_depth": 6, "min_child_weight": 0.1,
                         "num_boost_round": 120}}
    preds = _train_member(member, Xtr, Xva, data)
    np.savez_compressed(cache, preds=preds)
    return preds


# ------------------------------------------------------------------ evaluate

def _paired_delta(pq_a, pq_b, metric, rng):
    if metric == "revenue":
        valid = np.arange(len(pq_a["rev_ideal"]))

        def stat(idx, pq):
            ideal = pq["rev_ideal"][idx].sum()
            return pq["rev_realized"][idx].sum() / ideal if ideal > 0 else 0.0
    else:
        valid = np.where(~np.isnan(pq_a[metric]) & ~np.isnan(pq_b[metric]))[0]

        def stat(idx, pq, m=metric):
            return float(np.mean(pq[m][idx]))
    obs = stat(valid, pq_a) - stat(valid, pq_b)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.choice(valid, size=len(valid), replace=True)
        deltas[i] = stat(idx, pq_a) - stat(idx, pq_b)
    return obs, 1.96 * float(deltas.std())


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location(f"pipe_{uuid.uuid4().hex}", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(program_path: str) -> dict:
    t_start = time.monotonic()
    lines: List[str] = []
    try:
        mod = _load_program(program_path)
        if not hasattr(mod, "build_features") or not hasattr(mod, "PIPELINE"):
            raise StageError("spec", "program must define build_features and PIPELINE")
        members, ens = _validate_spec(mod.PIPELINE)

        data = _semantic_frames()
        tr, va = data["train"], data["val"]
        stats = TrainStats(tr["view"], tr["labels"])

        # stage 1: features
        t0 = time.monotonic()
        try:
            stats._mode = "train"
            Xtr_df = mod.build_features(tr["view"].copy(), stats)
            stats._mode = "apply"
            Xva_df = mod.build_features(va["view"].copy(), stats)
        except StageError:
            raise
        except Exception as ex:
            raise StageError("features", f"{type(ex).__name__}: {ex}")
        for name, X in (("train", Xtr_df), ("val", Xva_df)):
            if not isinstance(X, pd.DataFrame) or len(X) != len(data[name]["frame"]):
                raise StageError("features", f"{name} output misaligned or not a DataFrame")
        if list(Xtr_df.columns) != list(Xva_df.columns):
            raise StageError("features", "train/val column mismatch")
        if Xtr_df.shape[1] > MAX_FEATURES:
            raise StageError("features", f"{Xtr_df.shape[1]} columns exceeds {MAX_FEATURES}")
        if "qid" in Xtr_df.columns:
            Xtr_df = Xtr_df.drop(columns=["qid"]); Xva_df = Xva_df.drop(columns=["qid"])
        Xtr = Xtr_df.fillna(0.0).to_numpy(np.float64)
        Xva = Xva_df.fillna(0.0).to_numpy(np.float64)
        if not (np.isfinite(Xtr).all() and np.isfinite(Xva).all()):
            raise StageError("features", "non-finite values after fill")
        n_raw = tr["view"].shape[1] - 1
        engineered = [c for c in Xtr_df.columns if c not in tr["view"].columns]
        feat_line = (f"features: {Xtr_df.shape[1]} columns ({min(n_raw, Xtr_df.shape[1])} raw"
                     + (f" + {len(engineered)} engineered: "
                        + ", ".join(list(map(str, engineered))[:8]) if engineered else "")
                     + f"), build {time.monotonic() - t0:.1f}s")
        lines.append(feat_line)

        # stage 2: members
        preds, member_lines = [], []
        for i, member in enumerate(members):
            if time.monotonic() - t_start > WALL_BUDGET_S:
                member_lines.append(f"[{i}] {member['family']}/{member['objective']}: SKIPPED, wall budget")
                continue
            t0 = time.monotonic()
            try:
                p = np.asarray(_train_member(member, Xtr, Xva, data), np.float64)
                if not np.isfinite(p).all():
                    raise ValueError("non-finite predictions")
                nd = metrics_bundle(va["frame"], p)["ndcg"]
                preds.append(p)
                member_lines.append(f"[{i}] {member['family']}/{member['objective']}: "
                                    f"val ndcg {nd:.4f} ({time.monotonic() - t0:.0f}s)")
            except Exception as ex:
                member_lines.append(f"[{i}] {member['family']}/{member['objective']}: "
                                    f"FAILED {type(ex).__name__}: {ex}")
        if not preds:
            raise StageError("members", "no member produced predictions; " + " | ".join(member_lines))
        lines.append("members: " + "  ".join(member_lines))

        # stage 3: ensemble
        final = _combine(preds, ens, va["frame"])
        if len(preds) > 1:
            best_single = max(metrics_bundle(va["frame"], p)["ndcg"] for p in preds)
            ens_nd = metrics_bundle(va["frame"], final)["ndcg"]
            lines.append(f"ensemble: {ens['type']} "
                         f"{[round(w, 2) for w in ens.get('weights', [])] or ''} "
                         f"= {ens_nd:.4f}, best single {best_single:.4f} "
                         f"({ens_nd - best_single:+.4f})")

        # stage 4: metrics + noise gate vs seed pipeline
        m = metrics_bundle(va["frame"], final)
        if not all(np.isfinite(v) for v in m.values()):
            raise StageError("metrics", "non-finite metric")
        seed_preds = _seed_reference(data)
        rng = np.random.default_rng(0)
        pq_c = per_query_values(va["frame"], final)
        pq_s = per_query_values(va["frame"], seed_preds)
        parts = []
        for metric in ("ndcg", "book_ndcg", "revenue"):
            delta, hw = _paired_delta(pq_c, pq_s, metric, rng)
            gate = "SIGNIFICANT" if abs(delta) > hw else "within noise"
            parts.append(f"{metric}={m[metric]:.4f} ({delta:+.4f} vs seed, +/-{hw:.4f}, {gate})")
        lines.insert(0, "scores: " + " | ".join(parts))

        # stage 5: segments and churn
        has_b = va["frame"].groupby("qid", sort=False)["booking"].max().to_numpy() > 0
        seg_c = float(np.nanmean(pq_c["ndcg"][has_b])); seg_s = float(np.nanmean(pq_s["ndcg"][has_b]))
        clk_c = float(np.nanmean(pq_c["ndcg"][~has_b])); clk_s = float(np.nanmean(pq_s["ndcg"][~has_b]))
        lines.append(f"segments: booked-q ndcg {seg_c:.3f} ({seg_c - seg_s:+.3f} vs seed) | "
                     f"click-only {clk_c:.3f} ({clk_c - clk_s:+.3f})")

        return {**m, "combined_score": m["ndcg"],
                "artifacts": {"feedback": "\n".join(lines)}}

    except StageError as ex:
        return {"ndcg": 0.0, "book_ndcg": 0.0, "revenue": 0.0, "combined_score": 0.0,
                "artifacts": {"feedback": str(ex)}}
    except Exception as ex:
        return {"ndcg": 0.0, "book_ndcg": 0.0, "revenue": 0.0, "combined_score": 0.0,
                "artifacts": {"feedback": f"rejected: {type(ex).__name__}: {ex}"}}


if __name__ == "__main__":
    import json
    res = evaluate(str(ROOT / "seed" / "initial_pipeline.py"))
    print(json.dumps({k: v for k, v in res.items() if k != "artifacts"}, indent=1))
    print(res["artifacts"]["feedback"])
