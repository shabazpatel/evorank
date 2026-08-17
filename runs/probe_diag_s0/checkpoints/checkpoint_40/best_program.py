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
    """Composite-gain LambdaMART with decoupled order/magnitude.

    Ordering mask M uses only rel+booking (gain_core), so revenue can never
    flip which item is "more relevant" -- protects ndcg/book_ndcg. The
    |deltaNDCG| magnitude (dZ) uses the full gain, adding a per-query
    max-normalized log1p(rev) term so the revenue contribution stays on a
    comparable [0,1] scale across queries with very different price levels
    (instead of raw log1p(rev), which can dominate high-price queries).
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
        gain_core = (2.0 ** r - 1.0) + w_book * b
        logrv = np.log1p(rv)
        rv_max = logrv.max()
        rv_norm = logrv / rv_max if rv_max > 0 else logrv
        gain = gain_core + w_rev * rv_norm
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (gain_core[:, None] > gain_core[None, :])  # order strictly by rel+booking
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
