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
    # Robust composite-gain LambdaMART: fold rel, booking, and revenue into
    # a single per-query gain, then run standard |deltaNDCG| pairwise lambda
    # on that composite gain. Revenue is log1p-compressed and scaled by the
    # 90th-percentile (not max) of the group's log-revenue so a single
    # extreme booking doesn't crush the scale of everyone else's gain -
    # this keeps revenue informative for ordinary pairs while still
    # rewarding big bookings (values above p90 simply saturate near/above 1).
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64)
        lab = rel[s:e].astype(np.float64)
        book = booking[s:e].astype(np.float64)
        rv = rev[s:e].astype(np.float64)
        m = len(sc)
        if m < 2:
            continue
        rel_gain = 2.0 ** lab - 1.0
        rel_max = rel_gain.max()
        rel_norm = rel_gain / rel_max if rel_max > 0 else rel_gain
        rev_log = np.log1p(np.maximum(rv, 0.0))
        rev_scale = np.percentile(rev_log, 90) if rev_log.max() > 0 else 0.0
        rev_scale = rev_scale if rev_scale > 1e-8 else (rev_log.max() or 1.0)
        rev_norm = np.clip(rev_log / rev_scale, 0.0, 1.5)
        gain = rel_norm + 1.2 * book + rev_norm
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (gain[:, None] > gain[None, :])             # i has higher composite gain than j
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
