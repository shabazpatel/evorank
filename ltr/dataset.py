"""Data contract, loader, and fixed training harness for EvoRank.

The downstream contract every module relies on: a DataFrame with a ``qid``
column, feature columns named ``f0..fK``, and the label columns ``rel`` (graded
5/1/0), ``booking`` (binary), and ``rev`` (booking-weighted revenue proxy). Rows
are sorted so each ``qid`` forms one contiguous block (XGBoost group requirement).

``get_dataset(fast=...)`` returns the prepared Expedia parquet splits when they
exist (written by ``data/prepare_expedia.py``), and otherwise falls back to a
synthetic dataset that mirrors Expedia's structure. The synthetic fallback lets
the evaluator and baselines run end to end before the real data is prepared;
swapping in real data changes nothing downstream.

The training hyperparameters live here and are held fixed, so the search
isolates objective discovery rather than hyperparameter optimization. Ported
from notebooks/foundation.ipynb (Sections A and C).
"""
from __future__ import annotations

import pathlib
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from ltr.metrics import group_sizes_of

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Fixed training config for the custom-objective path. base_score and
# disable_default_eval_metric are required when training with a Python obj.
# Held constant across the search so only the objective varies (CLAUDE.md 7).
FIXED_PARAMS = {
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 0.1,
    "tree_method": "hist",
    "seed": 0,
    "base_score": 0.0,
    "disable_default_eval_metric": 1,
}
FIXED_ROUNDS = 120

_FEAT_RE = re.compile(r"^f(\d+)$")
_CACHE: dict = {}


def feature_columns(frame: pd.DataFrame) -> List[str]:
    """Feature columns ``f0..fK`` in numeric index order."""
    feats = [c for c in frame.columns if _FEAT_RE.match(c)]
    return sorted(feats, key=lambda c: int(_FEAT_RE.match(c).group(1)))


def to_dmatrix(frame: pd.DataFrame, feats: List[str]) -> xgb.DMatrix:
    """Build a grouped DMatrix. Label is graded relevance ``rel``; the other
    label columns are used only for metric computation, never for training."""
    d = xgb.DMatrix(frame[feats].to_numpy(), label=frame["rel"].to_numpy().astype(float))
    d.set_group(group_sizes_of(frame))
    return d


def make_synthetic(n_queries: int = 500, items: int = 15, n_feat: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset mirroring Expedia structure: query groups, graded label
    in {0, 1, 5}, a booking flag, and a revenue column. Some queries get a second
    booking so the revenue objective genuinely differs from the booking objective."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        m = int(rng.integers(items - 4, items + 4))
        X = rng.normal(size=(m, n_feat))
        w = rng.normal(size=n_feat)
        utility = X @ w + rng.normal(scale=0.5, size=m)
        order = np.argsort(-utility)
        rel = np.zeros(m, dtype=int)
        rel[order[0]] = 5 if rng.random() < 0.6 else 1
        if m > 1 and rng.random() < 0.2:
            rel[order[1]] = 5
        for idx in order[2:max(2, m // 4)]:
            rel[idx] = 1 if rng.random() < 0.5 else 0
        price = rng.uniform(80, 600, size=m)
        booking = (rel == 5).astype(int)
        rev = booking * price
        for i in range(m):
            rows.append((q, *X[i], int(rel[i]), int(booking[i]), float(rev[i])))
    cols = ["qid"] + [f"f{i}" for i in range(n_feat)] + ["rel", "booking", "rev"]
    df = pd.DataFrame(rows, columns=cols)
    df["random_flag"] = 0  # synthetic has no position-bias experiment
    return df


def _split_by_qid(df: pd.DataFrame, fracs=(0.70, 0.15), seed: int = 7):
    qids = df["qid"].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    n = len(qids)
    tr_q = set(qids[: int(fracs[0] * n)])
    va_q = set(qids[int(fracs[0] * n): int((fracs[0] + fracs[1]) * n)])
    te_q = set(qids) - tr_q - va_q
    pick = lambda s: df[df.qid.isin(s)].sort_values("qid").reset_index(drop=True)
    return pick(tr_q), pick(va_q), pick(te_q)


def _load_prepared(subset_dir: pathlib.Path):
    frames = {}
    for name in ("train", "val", "test"):
        path = subset_dir / f"{name}.parquet"
        if not path.exists():
            return None
        frames[name] = pd.read_parquet(path).sort_values("qid").reset_index(drop=True)
    return frames["train"], frames["val"], frames["test"]


def get_dataset(fast: bool = True, seed: int = 7):
    """Return ``(train_df, val_df, test_df, feats, src)``.

    Uses prepared Expedia parquet when present (``data/prepared_fast`` for the
    loop, ``data/prepared`` for final reporting), otherwise a synthetic fallback.
    ``src`` names the source so callers and logs can report which was used.
    """
    key = ("prepared", fast, seed)
    if key in _CACHE:
        return _CACHE[key]

    subset_dir = ROOT / ("data/prepared_fast" if fast else "data/prepared")
    prepared = _load_prepared(subset_dir)
    if prepared is not None:
        train_df, val_df, test_df = prepared
        src = f"prepared_fast ({subset_dir.name})" if fast else f"prepared ({subset_dir.name})"
    else:
        # Synthetic fallback. Smaller when fast so evaluate() stays quick.
        n_q = 500 if fast else 1500
        df = make_synthetic(n_queries=n_q, seed=1)
        train_df, val_df, test_df = _split_by_qid(df, seed=seed)
        src = "synthetic (no prepared parquet found)"

    feats = feature_columns(train_df)
    result = (train_df, val_df, test_df, feats, src)
    _CACHE[key] = result
    return result


def train_and_predict(train_df: pd.DataFrame, val_df: pd.DataFrame, feats: List[str], obj) -> np.ndarray:
    """Train with the fixed config and a custom objective, predict on val.

    ``obj`` is an XGBoost objective callable ``(predt, dtrain) -> (grad, hess)``.
    Kept here so every candidate and baseline trains under identical settings.
    """
    dtr = to_dmatrix(train_df, feats)
    dva = to_dmatrix(val_df, feats)
    bst = xgb.train(FIXED_PARAMS, dtr, num_boost_round=FIXED_ROUNDS, obj=obj)
    return bst.predict(dva)


if __name__ == "__main__":
    tr, va, te, feats, src = get_dataset(fast=True)
    print(f"source: {src}")
    print(f"features: {len(feats)}  ({feats[:5]}{'...' if len(feats) > 5 else ''})")
    for name, f in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:5s} queries {f.qid.nunique():>7}  rows {len(f):>9}  "
              f"rel_dist {f['rel'].value_counts().sort_index().to_dict()}")