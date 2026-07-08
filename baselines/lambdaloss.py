"""Baseline: LambdaLoss-style metric-driven objective (REQUIRED).

The hand-derived counterpart to what the evolution discovers, and the most
important comparison (Wang et al. 2018). Implements the NDCG-Loss2 weighting:
pair weight |G_i - G_j| * |1/D(d) - 1/D(d + 1)| where d = |rank_i - rank_j| is
the rank distance, which the LambdaLoss framework shows gives a tighter bound
on NDCG than the classic LambdaMART |delta NDCG| form. Note the classic form
|(G_i - G_j)(D_i - D_j)| factors into |G_i - G_j| * |D_i - D_j|, so a
position-pair variant would be identical to the seed; the rank-distance form
here is the genuinely different one. Trains through the exact same harness and
fixed config as every candidate in the loop.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines._common import log_run, score  # noqa: E402
from ltr.dataset import get_dataset, train_and_predict  # noqa: E402
from ltr.metrics import group_sizes_of, group_slices  # noqa: E402


def lambdaloss_objective(predt, rel, group_slices_list, sigma=1.0):
    """NDCG-Loss2 lambda weighting per the LambdaLoss framework."""
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices_list:
        sc = predt[s:e].astype(np.float64)
        lab = rel[s:e]
        m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))
        M = lab[:, None] > lab[None, :]
        gain = 2.0 ** lab - 1.0
        ranks = np.empty(m, dtype=int)
        ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        # NDCG-Loss2: |G_i - G_j| * |1/D(d) - 1/D(d + 1)| with d = |rank_i - rank_j|.
        # The rank-distance discount is what distinguishes this from LambdaMART.
        d = np.abs(ranks[:, None] - ranks[None, :]).astype(np.float64)
        np.fill_diagonal(d, 1.0)  # avoid log2(1) = 0 on the (unused) diagonal
        delta_disc = np.abs(1.0 / np.log2(d + 1.0 + 1e-12) - 1.0 / np.log2(d + 2.0))
        dZ = np.abs(gain[:, None] - gain[None, :]) * delta_disc / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)


def main() -> None:
    train_df, val_df, _test, feats, src = get_dataset(fast=True)
    slices = group_slices(group_sizes_of(train_df))
    rel = train_df["rel"].to_numpy(dtype=np.float64)

    def obj(predt, dtrain):
        g, h = lambdaloss_objective(predt, rel, slices)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(h))):
            raise ValueError("non-finite grad/hess")
        return g, h

    preds = train_and_predict(train_df, val_df, feats, obj)
    log_run("lambdaloss", score(val_df, preds), extra={"source": src, "variant": "ndcg_loss2"})


if __name__ == "__main__":
    main()
