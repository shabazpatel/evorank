"""Diagnostic feedback for candidate ranking objectives.

Builds the EVORANK_FEEDBACK=diagnostic payload: evidence about what a candidate
objective is doing, not only its scores. All components are computed from
predictions and gradients already available at evaluation time. No additional
model training and no additional LLM calls. Typical overhead 3 to 6 seconds on
the fast subset, against a ~40 second training.

Components (see paper/approach.md section 3):
  scores     three metrics with deltas vs the seed objective and a paired
             bootstrap noise gate, so real movement is separable from luck
  segments   booked vs click-only query decomposition, and booked-item rank on
             the top-revenue-decile queries
  behavior   booked-item rank distribution, score spread (collapse detector),
             top-10 overlap with the seed ranking (edit churn)
  alignment  one virtual gradient step on the training scores: does the
             objective's own gradient move each metric up or down
  usage      share of gradient mass attributable to each behavioral signal
             (rel / booking / rev), detecting functionally dead terms

Seed reference predictions are trained once per data fingerprint and cached in
runs/_cache/, so a run pays the one-off cost only on its first evaluation.
Any exception inside diagnostics degrades to scores-only feedback, never to a
failed evaluation.
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from ltr.dataset import FIXED_PARAMS, FIXED_ROUNDS, to_dmatrix
from ltr.metrics import group_sizes_of, group_slices, metrics_bundle, per_query_values

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "runs" / "_cache"
N_BOOT = 400
K = 10


# ---------------------------------------------------------------- seed cache

def _fingerprint(train_df: pd.DataFrame, val_df: pd.DataFrame) -> str:
    sig = (f"{len(train_df)}:{train_df.qid.nunique()}:{train_df.rev.sum():.2f}:"
           f"{len(val_df)}:{val_df.qid.nunique()}:{val_df.rev.sum():.2f}:"
           f"{FIXED_ROUNDS}")
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def seed_val_predictions(train_df, val_df, feats) -> np.ndarray:
    """Val-fold predictions of the seed objective under the fixed config,
    cached on disk keyed by a data fingerprint."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"seed_val_preds_{_fingerprint(train_df, val_df)}.npz"
    if cache.exists():
        return np.load(cache)["preds"]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seed_ref", ROOT / "seed" / "initial_program.py")
    seed_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_mod)

    slices = group_slices(group_sizes_of(train_df))
    rel = train_df["rel"].to_numpy(dtype=np.float64)
    booking = train_df["booking"].to_numpy(dtype=np.float64)
    rev = train_df["rev"].to_numpy(dtype=np.float64)

    def obj(predt, dtrain):
        g, h = seed_mod.lambdamart_objective(predt, rel, booking, rev, slices)
        return np.asarray(g, np.float64), np.asarray(h, np.float64)

    bst = xgb.train(FIXED_PARAMS, to_dmatrix(train_df, feats),
                    num_boost_round=FIXED_ROUNDS, obj=obj)
    preds = bst.predict(to_dmatrix(val_df, feats))
    np.savez_compressed(cache, preds=preds)
    return preds


# ------------------------------------------------------------- query helpers

def _per_query_tables(val_df: pd.DataFrame) -> dict:
    """Static per-query facts about the val fold (computed once per call)."""
    g = val_df.groupby("qid", sort=False)
    return {
        "has_booking": (g["booking"].max() > 0).to_numpy(),
        "rev_ideal_rank": None,  # filled lazily
    }


def _booked_ranks(val_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Rank (0-based) of the booked item per query; NaN for bookingless queries."""
    gs = group_sizes_of(val_df)
    booking = val_df["booking"].to_numpy()
    out = []
    for s, e in group_slices(gs):
        b = booking[s:e]
        if b.sum() == 0:
            out.append(np.nan)
            continue
        order = np.argsort(-preds[s:e], kind="stable")
        ranks = np.empty(e - s, dtype=int)
        ranks[order] = np.arange(e - s)
        out.append(float(ranks[np.argmax(b)]))
    return np.asarray(out)


def _top10_overlap(val_df: pd.DataFrame, a: np.ndarray, b: np.ndarray) -> float:
    gs = group_sizes_of(val_df)
    overlaps = []
    for s, e in group_slices(gs):
        ta = set(np.argsort(-a[s:e], kind="stable")[:K].tolist())
        tb = set(np.argsort(-b[s:e], kind="stable")[:K].tolist())
        overlaps.append(len(ta & tb) / max(len(ta | tb), 1))
    return float(np.mean(overlaps))


# ----------------------------------------------------------- noise-aware delta

def _paired_delta(pq_a: dict, pq_b: dict, metric: str, rng) -> Tuple[float, float]:
    """(observed delta, bootstrap 95pc half-width) for a paired query bootstrap."""
    if metric == "revenue":
        def stat(idx, pq):
            ideal = pq["rev_ideal"][idx].sum()
            return pq["rev_realized"][idx].sum() / ideal if ideal > 0 else 0.0
        valid = np.arange(len(pq_a["rev_ideal"]))
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


# ----------------------------------------------------------------- main entry

def build_diagnostics(mod, train_df, val_df, feats, bst,
                      val_preds: np.ndarray, dtr) -> str:
    rng = np.random.default_rng(0)
    lines: List[str] = []

    seed_preds = seed_val_predictions(train_df, val_df, feats)
    pq_c = per_query_values(val_df, val_preds, k=K)
    pq_s = per_query_values(val_df, seed_preds, k=K)
    m = metrics_bundle(val_df, val_preds, k=K)

    # 1. scores with noise-gated deltas vs seed
    parts = []
    for metric in ("ndcg", "book_ndcg", "revenue"):
        delta, hw = _paired_delta(pq_c, pq_s, metric, rng)
        gate = "SIGNIFICANT" if abs(delta) > hw else "within noise"
        parts.append(f"{metric}={m[metric]:.4f} ({delta:+.4f} vs seed, "
                     f"noise +/-{hw:.4f}, {gate})")
    lines.append("scores: " + " | ".join(parts))

    # 2. segments: booked vs click-only, and top-revenue-decile booked rank
    has_booking = val_df.groupby("qid", sort=False)["booking"].max().to_numpy() > 0
    seg = []
    for name, mask in (("booked-queries", has_booking),
                       ("click-only", ~has_booking)):
        c = float(np.nanmean(pq_c["ndcg"][mask]))
        s = float(np.nanmean(pq_s["ndcg"][mask]))
        seg.append(f"{name} ndcg {c:.3f} ({c - s:+.3f} vs seed)")
    ranks_c = _booked_ranks(val_df, val_preds)
    ranks_s = _booked_ranks(val_df, seed_preds)
    booked_ideal = pq_c["rev_ideal"]
    thresh = np.nanpercentile(booked_ideal[has_booking], 90)
    top_rev = has_booking & (booked_ideal >= thresh)
    seg.append(f"top-revenue-decile booked-item median rank "
               f"{np.nanmedian(ranks_c[top_rev]):.0f} "
               f"(seed {np.nanmedian(ranks_s[top_rev]):.0f})")
    lines.append("segments: " + " | ".join(seg))

    # 3. behavior: booked ranks, score spread, churn vs seed
    med = np.nanmedian(ranks_c)
    top1 = float(np.nanmean(ranks_c == 0))
    miss10 = float(np.nanmean(ranks_c >= K))
    overlap = _top10_overlap(val_df, val_preds, seed_preds)
    lines.append(f"behavior: booked-item rank median {med:.0f}, top-1 rate {top1:.0%}, "
                 f"outside-top-10 {miss10:.0%} | score std {val_preds.std():.3f} | "
                 f"top-10 overlap with seed ranking {overlap:.0%}")

    # 4. alignment probe: one virtual gradient step on the training scores
    train_preds = bst.predict(dtr)
    slices = group_slices(group_sizes_of(train_df))
    rel = train_df["rel"].to_numpy(dtype=np.float64)
    booking = train_df["booking"].to_numpy(dtype=np.float64)
    rev = train_df["rev"].to_numpy(dtype=np.float64)
    g, _ = mod.lambdamart_objective(train_preds, rel, booking, rev, slices)
    g = np.asarray(g, dtype=np.float64)
    g90 = float(np.percentile(np.abs(g), 90))
    if g90 > 0:
        # Small local step: magnitudes are directional evidence only, so the
        # feedback reports signs and relative sizes, not calibrated gains.
        eta = 0.02 * float(train_preds.std()) / g90
        m0 = metrics_bundle(train_df, train_preds, k=K)
        m1 = metrics_bundle(train_df, train_preds - eta * g, k=K)
        moves = {k: m1[k] - m0[k] for k in m0}
        anti = [k for k, v in moves.items() if v < -1e-5]
        note = f"; ANTI-ALIGNED with {', '.join(anti)}" if anti else "; aligned with all three"
        lines.append("alignment (directional, small virtual step): gradient moves "
                     + " ".join(f"{k} {v:+.4f}" for k, v in moves.items()) + note)
    else:
        lines.append("alignment: gradient is numerically zero, objective produces no learning signal")

    # 5. signal usage: gradient sensitivity to zeroing each behavioral signal.
    # Calibration note: share of gradient mass, not importance. A term moving
    # under 0.1 percent of gradient mass can still shift metrics materially if
    # its push is systematic across boosting rounds (measured: the discovered
    # clipped-log revenue term moves ~0.09 percent of mass yet removing it
    # costs about 0.009 revenue). Only an exactly-zero usage means dead code.
    g_norm = float(np.linalg.norm(g)) + 1e-12
    usage = {}
    for name, (r_, b_, v_) in {
        "rel": (np.zeros_like(rel), booking, rev),
        "booking": (rel, np.zeros_like(booking), rev),
        "rev": (rel, booking, np.zeros_like(rev)),
    }.items():
        g2, _ = mod.lambdamart_objective(train_preds, r_, b_, v_, slices)
        usage[name] = float(np.linalg.norm(g - np.asarray(g2, np.float64))) / g_norm
    dead = [k for k, v in usage.items() if v < 1e-4]
    note = f"; {', '.join(dead)} has NO gradient effect (dead code)" if dead else ""
    lines.append("signal usage (share of gradient mass; small but nonzero can still "
                 "matter if systematic): "
                 + " ".join(f"{k} {v:.2%}" for k, v in usage.items()) + note)

    return "\n".join(lines)
