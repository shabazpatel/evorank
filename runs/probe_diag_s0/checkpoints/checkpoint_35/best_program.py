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
    """Composite-gain pairwise LambdaMART.

    Instead of using discrete rel alone, build a single continuous gain per
    row: gain = (2**rel - 1) + w_book*booking + w_rev*log1p(rev). This gain
    drives both which pairs are "correctly ordered" (M) and the |deltaNDCG|
    lambda weighting (dZ), so a single pairwise-logistic LambdaMART pass
    jointly pushes for relevance ranking, booking conversion, and revenue
    capture instead of optimizing relevance alone.
    """
    """Rank-normalized revenue LambdaMART.

    Instead of log1p + percentile-cap (scale-sensitive across queries with
    very different price ranges), revenue is converted to a per-query
    percentile rank among *booked* items only (rev_rank in (0,1], 0 for
    non-booked). This is scale-invariant: a $50 vs $500 hotel query and a
    $5000 vs $50000 hotel query contribute comparably, removing the need to
    hand-tune a cap. Order (M) is driven by relevance+booking (gain_core)
    plus a tiny rev_rank tie-break so equally-relevant items still get a
    well-defined, revenue-informed order without ever overturning the
    rel/booking ranking. Magnitude (dZ/idcg) uses the full gain (core +
    w_rev*rev_rank), amplifying gradient for revenue-rich pairs.
    """
    """Rank-normalized revenue LambdaMART with unified ordering/weighting.

    Revenue is converted to a per-query percentile rank among booked items
    (scale-invariant across price ranges), then folded into a single
    composite gain with relevance and booking: gain = (2**rel-1) +
    w_book*booking + w_rev*rev_rank. This unified gain drives BOTH the
    pairwise ordering mask (M) and the |deltaNDCG| magnitude (dZ), letting
    revenue actually influence which pairs are treated as "correctly
    ordered" (not just how strongly misordered pairs are penalized), while
    the much larger booking/relevance gains keep those signals dominant.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    w_book, w_rev = 1.0, 0.5
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); m = len(sc)
        if m < 2:
            continue
        r = rel[s:e].astype(np.float64)
        b = booking[s:e].astype(np.float64)
        rv = rev[s:e].astype(np.float64)
        # Per-query percentile rank of revenue among booked (rev>0) items
        # only; non-booked items get 0. Bounded in (0,1], scale-invariant.
        rev_rank = np.zeros(m, dtype=np.float64)
        nz_idx = np.where(rv > 0)[0]
        if nz_idx.size > 0:
            order = np.argsort(rv[nz_idx], kind="mergesort")
            r_rank = np.empty(nz_idx.size, dtype=np.float64)
            r_rank[order] = np.arange(1, nz_idx.size + 1)
            rev_rank[nz_idx] = r_rank / nz_idx.size
        gain = (2.0 ** r - 1.0) + w_book * b + w_rev * rev_rank
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (gain[:, None] > gain[None, :])            # unified relevance+booking+revenue order
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
