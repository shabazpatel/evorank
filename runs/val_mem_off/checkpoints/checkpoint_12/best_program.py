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
    """Approach: relevance strictly drives the ranking mask/dZ (preserving
    ndcg/book_ndcg quality). Revenue enters via (1) tie-breaking among
    same-relevance pairs using raw revenue, and (2) a bounded
    consistency boost on relevance-pair lambda magnitude driven by
    per-group RANK-PERCENTILE revenue (robust to outlier high-price
    bookings, unlike log1p) so a handful of very expensive rooms don't
    dominate gradient scale. This trades a different revenue-robustness
    profile than log1p-based variants while keeping the same
    relevance-preserving structure.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    beta_tie = 0.6    # revenue tie-break weight (same-relevance pairs)
    gamma_rel = 0.5   # revenue-consistency magnitude boost for relevance pairs
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        rev_s = rev[s:e].astype(np.float64)
        # rank-percentile revenue: robust to outliers, bounded in [0,1]
        order = np.argsort(rev_s, kind="stable")
        pct = np.empty(m, dtype=np.float64)
        pct[order] = np.arange(m, dtype=np.float64) / max(m - 1, 1)
        gain = (2.0 ** lab - 1.0)                       # pure relevance gain
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0

        M_rel = (lab[:, None] > lab[None, :])          # i strictly more relevant than j
        tie = (lab[:, None] == lab[None, :])
        M_tie = tie & (rev_s[:, None] > rev_s[None, :])  # break same-relevance ties by raw revenue
        M = M_rel | M_tie

        dZ_rel = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        pct_diff = pct[:, None] - pct[None, :]
        consistency = np.tanh(gamma_rel * pct_diff)     # in (-1, 1)
        dZ_rel_boosted = dZ_rel * (1.0 + consistency)   # stays >= 0 since factor in (0, 2)
        dZ_rev = beta_tie * np.abs(pct_diff * (disc[:, None] - disc[None, :])) / idcg
        dZ = np.where(M_rel, dZ_rel_boosted, dZ_rev)

        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
