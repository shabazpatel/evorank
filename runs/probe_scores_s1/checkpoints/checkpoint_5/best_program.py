import numpy as np

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """Group-aware LambdaMART gradient over three behavioral signals.

    Per row the objective may use any of:
      - rel:     graded relevance (5 booked, 1 clicked, 0 neither)
      - booking: binary booking flag
      - rev:     booking revenue in USD (0 when not booked)
    group_slices is a list of (start, end) row ranges, one per query.
    Returns (grad, hess) per row.

    Baseline: pairwise-logistic lambda with |deltaNDCG| weighting on the
    relevance gain (2**rel - 1). This reproduces XGBoost's built-in rank:ndcg
    within noise (the Phase 1 correctness check). booking and rev are handed in
    but unused by the seed; SkyDiscover mutates this body to fold them in and
    trade off relevance, conversion, and revenue. How to weight and transform
    revenue (raw, log, clipped, per-pair) is for the search to discover, not a
    hand-set choice.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        pos_disc = 1.0 / np.log2(np.arange(2, m + 2))

        # --- Term 1: pure relevance lambda (drives ndcg, matches baseline) ---
        gain_rel = 2.0 ** lab - 1.0
        M_rel = (lab[:, None] > lab[None, :])
        idcg_rel = float(np.sum(np.sort(gain_rel)[::-1] * pos_disc)) or 1.0
        dZ_rel = np.abs((gain_rel[:, None] - gain_rel[None, :]) * (disc[:, None] - disc[None, :])) / idcg_rel

        # --- Term 2: pure revenue lambda (drives book_ndcg / revenue) ---
        bk = booking[s:e].astype(np.float64)
        rv = rev[s:e].astype(np.float64)
        lrv = np.log1p(rv)
        lrv_max = lrv.max()
        rv_norm = lrv / lrv_max if lrv_max > 0 else lrv
        gain_rev = bk * rv_norm
        M_rev = (gain_rev[:, None] > gain_rev[None, :])
        idcg_rev = float(np.sum(np.sort(gain_rev)[::-1] * pos_disc)) or 1.0
        dZ_rev = np.abs((gain_rev[:, None] - gain_rev[None, :]) * (disc[:, None] - disc[None, :])) / idcg_rev

        beta = 0.6  # relative weight of revenue term vs relevance term
        lam = (np.where(M_rel, -sigma * rho * dZ_rel, 0.0)
               + beta * np.where(M_rev, -sigma * rho * dZ_rev, 0.0))
        hmat = (np.where(M_rel, (sigma ** 2) * rho * (1.0 - rho) * dZ_rel, 0.0)
                + beta * np.where(M_rev, (sigma ** 2) * rho * (1.0 - rho) * dZ_rev, 0.0))
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
