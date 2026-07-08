"""Query-grouped ranking metrics for EvoRank.

All three metrics are NDCG-style (position discount + ideal-ordering
normalization), at k=10 by default, and all lie in [0, 1]:
  - ``ndcg``      uses graded relevance gain ``2**rel - 1``, averaged equally
                  over queries (relevance objective).
  - ``book_ndcg`` uses the binary ``booking`` flag, averaged equally over queries
                  (pure conversion objective).
  - ``revenue``   an explicit revenue proxy: position-discounted booking revenue
                  captured in top-k, pooled across queries and normalized by the
                  ideal revenue ordering (``revenue_capture_at_k``).

The design tension that makes these genuinely compete: relevance and conversion
weight every query equally, while revenue weights by dollars. Expedia searches
have at most one booking each, so a per-query-averaged NDCG on booking revenue
would collapse onto ``book_ndcg`` (per-query NDCG cancels the magnitude of a lone
positive). Pooling the realized and ideal revenue across queries keeps the
magnitude, so high-value searches dominate the revenue objective and an objective
can trade average conversion against high-value ranking. For single-booking data
this pooled ratio equals a revenue-weighted average of per-query booking-NDCG.

Ported from notebooks/foundation.ipynb (Section B); the graded relevance metric
is validated against XGBoost's built-in rank:ndcg.

Run ``python ltr/metrics.py`` for the unit checks.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def group_sizes_of(frame: pd.DataFrame) -> np.ndarray:
    """Contiguous group sizes per ``qid``, in row order. Assumes the frame is
    already sorted so each qid forms one contiguous block."""
    return frame.groupby("qid", sort=False).size().to_numpy()


def group_slices(group_sizes: np.ndarray) -> List[Tuple[int, int]]:
    """Convert group sizes into (start, end) row ranges, one per query."""
    idx, out = 0, []
    for g in group_sizes:
        out.append((idx, idx + int(g)))
        idx += int(g)
    return out


def grouped_ndcg(gain: np.ndarray, scores: np.ndarray, group_sizes: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k over query groups. Queries with idcg == 0 (no positive gain)
    are skipped, so a fold with no bookings cannot inflate book_ndcg/revenue."""
    vals = []
    for s, e in group_slices(group_sizes):
        g, sc = gain[s:e], scores[s:e]
        order = np.argsort(-sc, kind="stable")
        ranked = g[order][:k]
        dcg = float(np.sum(ranked * (1.0 / np.log2(np.arange(2, len(ranked) + 2)))))
        ideal = np.sort(g)[::-1][:k]
        idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))))
        if idcg > 0:
            vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else 0.0


def revenue_capture_at_k(rev: np.ndarray, scores: np.ndarray, group_sizes: np.ndarray, k: int = 10) -> float:
    """Position-discounted booking revenue captured in top-k, pooled across
    queries and normalized by the ideal revenue ordering. In [0, 1].

    Unlike ``grouped_ndcg`` (which averages per-query ratios equally), the
    numerator and denominator are pooled across queries, so a query's weight is
    its revenue magnitude. This is what makes revenue compete with ``book_ndcg``
    on single-booking data (see module docstring). Queries with no revenue
    (no booking) have ideal 0 and contribute nothing.
    """
    num, den = 0.0, 0.0
    for s, e in group_slices(group_sizes):
        r, sc = rev[s:e], scores[s:e]
        n = len(r)
        disc = 1.0 / np.log2(np.arange(2, n + 2))
        realized = float(np.sum(r[np.argsort(-sc, kind="stable")][:k] * disc[:k]))
        ideal = float(np.sum(np.sort(r)[::-1][:k] * disc[:k]))
        if ideal > 0:
            num += realized
            den += ideal
    return num / den if den > 0 else 0.0


def per_query_values(frame: pd.DataFrame, scores: np.ndarray, k: int = 10) -> dict:
    """Per-query building blocks for bootstrap analysis.

    Returns arrays aligned to the frame's query order:
      ndcg, book_ndcg: per-query NDCG@k, NaN where that query has idcg == 0
      rev_realized, rev_ideal: per-query discounted revenue captured and ideal,
        so pooled revenue = rev_realized.sum() / rev_ideal.sum() and a bootstrap
        can resample queries and recompute the ratio.
    """
    gs = group_sizes_of(frame)
    rel_gain = 2.0 ** frame["rel"].to_numpy() - 1.0
    book = frame["booking"].to_numpy().astype(float)
    rev = frame["rev"].to_numpy().astype(float)

    out = {"ndcg": [], "book_ndcg": [], "rev_realized": [], "rev_ideal": []}
    for s, e in group_slices(gs):
        sc = scores[s:e]
        order = np.argsort(-sc, kind="stable")
        n = e - s
        disc = 1.0 / np.log2(np.arange(2, n + 2))
        for key, gain in (("ndcg", rel_gain[s:e]), ("book_ndcg", book[s:e])):
            dcg = float(np.sum(gain[order][:k] * disc[:k]))
            idcg = float(np.sum(np.sort(gain)[::-1][:k] * disc[:k]))
            out[key].append(dcg / idcg if idcg > 0 else np.nan)
        r = rev[s:e]
        out["rev_realized"].append(float(np.sum(r[order][:k] * disc[:k])))
        out["rev_ideal"].append(float(np.sum(np.sort(r)[::-1][:k] * disc[:k])))
    return {key: np.asarray(v) for key, v in out.items()}


def metrics_bundle(frame: pd.DataFrame, scores: np.ndarray, k: int = 10) -> dict:
    """The three competing objectives for a fold, given predicted scores."""
    gs = group_sizes_of(frame)
    return {
        "ndcg": grouped_ndcg((2.0 ** frame["rel"].to_numpy() - 1.0), scores, gs, k),
        "book_ndcg": grouped_ndcg(frame["booking"].to_numpy().astype(float), scores, gs, k),
        "revenue": revenue_capture_at_k(frame["rev"].to_numpy().astype(float), scores, gs, k),
    }


def _unit_checks() -> None:
    # perfect ranking scores 1.0, reversed ranking scores below 1.0
    g = np.array([3.0, 2.0, 3.0, 0.0, 1.0, 2.0])
    assert abs(grouped_ndcg(g, g, [6], 6) - 1.0) < 1e-9, "perfect order must be 1.0"
    assert grouped_ndcg(g, -g, [6], 6) < 1.0, "reversed order must be < 1.0"

    # group_slices tiles the row range exactly
    assert group_slices([2, 3]) == [(0, 2), (2, 5)]

    # metrics_bundle: perfect scores give ndcg == 1.0 on a one-query frame
    df = pd.DataFrame({
        "qid": [0, 0, 0],
        "rel": [5, 1, 0],
        "booking": [1, 0, 0],
        "rev": [200.0, 0.0, 0.0],
    })
    m = metrics_bundle(df, np.array([3.0, 2.0, 1.0]), k=10)
    assert abs(m["ndcg"] - 1.0) < 1e-9 and abs(m["book_ndcg"] - 1.0) < 1e-9
    assert abs(m["revenue"] - 1.0) < 1e-9

    # revenue must diverge from book_ndcg when bookings have unequal value.
    # q0: a $1000 booking mis-ranked to position 2; q1: a $10 booking at position 1.
    # book_ndcg averages the two queries equally; revenue is dollar-weighted, so
    # it is dragged down by the mis-ranked high-value booking.
    div = pd.DataFrame({
        "qid": [0, 0, 1, 1],
        "rel": [5, 0, 5, 0],
        "booking": [1, 0, 1, 0],
        "rev": [1000.0, 0.0, 10.0, 0.0],
    })
    scores = np.array([1.0, 2.0, 2.0, 1.0])  # q0 booking demoted, q1 booking on top
    md = metrics_bundle(div, scores, k=10)
    assert 0.0 <= md["revenue"] <= 1.0 and 0.0 <= md["book_ndcg"] <= 1.0
    assert abs(md["book_ndcg"] - md["revenue"]) > 0.1, "revenue must not collapse onto book_ndcg"
    print(f"metric unit checks passed "
          f"(book_ndcg={md['book_ndcg']:.4f} vs revenue={md['revenue']:.4f} on the divergence case)")


if __name__ == "__main__":
    _unit_checks()