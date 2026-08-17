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
    """Single-pass LambdaMART with revenue-amplified relevance gain.

    Instead of adding a second, independent revenue-driven lambda pass (as in
    prior variants), this folds revenue directly into the relevance gain used
    for the *one* pairwise |deltaNDCG| computation. For booked rows, the raw
    revenue is converted to a within-query, rank-based percentile (robust to
    outliers/scale, unlike raw or log1p(rev)/max), and the base gain
    (2**rel - 1) is multiplied by (1 + w_rev * revenue_percentile). Because
    binary/graded relevance already places booked rows above clicked/neither
    rows, this amplification only reshapes ordering *within* the booked tier
    (and slightly changes deltaNDCG magnitudes across tiers) -- it cannot
    demote a booked row below a non-booked one, so book_ndcg is essentially
    unaffected (permuting among already-top-ranked booked rows doesn't change
    binary NDCG), while relevance-tier lambdas naturally pull higher-revenue
    bookings toward the top, targeting revenue with a single unified,
    cheaper computation.
    """
    """Two-channel LambdaMART: revenue-amplified relevance gain plus a
    lightweight direct booking channel.

    Channel 1 (cheap, as before): fold revenue into the relevance gain via a
    within-query rank percentile among booked rows, so higher-revenue
    bookings get amplified gain relative to same-tier peers. This drives
    ndcg and revenue jointly with a single pairwise |deltaNDCG| pass.

    Channel 2 (new, small weight): a second pairwise |deltaNDCG| pass using
    raw binary booking as the gain, computed only when a query has a mix of
    booked/unbooked rows. This gives book_ndcg explicit, targeted pairwise
    pressure independent of the amplified-gain reshaping in channel 1, which
    otherwise only reorders within the booked tier and leaves book_ndcg
    largely untouched. The two channels' lambdas/hessians are summed with a
    fixed low weight on the booking channel so relevance/revenue signal
    still dominates.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    w_rev = 0.5
    w_book = 0.3
    w_rev_explicit = 0.2

    def pairwise_lambda(gain, rho, disc, m):
        M = gain[:, None] > gain[None, :]
        if not M.any():
            return None, None
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        return lam, hmat

    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e].astype(np.float64)
        bk = booking[s:e].astype(np.float64); rv = rev[s:e].astype(np.float64)
        m = len(sc)
        if m < 2:
            continue

        rev_boost = np.zeros(m, dtype=np.float64)
        booked_idx = np.where(bk > 0)[0]
        if len(booked_idx) > 1:
            order = np.argsort(rv[booked_idx])
            perc = np.empty(len(booked_idx), dtype=np.float64)
            perc[order] = np.arange(len(booked_idx)) / (len(booked_idx) - 1)
            rev_boost[booked_idx] = perc
        elif len(booked_idx) == 1:
            rev_boost[booked_idx] = 1.0

        base_gain = 2.0 ** lab - 1.0
        gain_rel = base_gain * (1.0 + w_rev * rev_boost)

        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)

        lam_total, hmat_total = pairwise_lambda(gain_rel, rho, disc, m)
        if lam_total is None:
            lam_total = np.zeros((m, m), dtype=np.float64)
            hmat_total = np.zeros((m, m), dtype=np.float64)

        if 0.0 < bk.sum() < m:
            lam_b, hmat_b = pairwise_lambda(bk, rho, disc, m)
            if lam_b is not None:
                lam_total = lam_total + w_book * lam_b
                hmat_total = hmat_total + w_book * hmat_b

        rv_log = np.log1p(rv)
        if rv_log.max() > 0:
            lam_r, hmat_r = pairwise_lambda(rv_log, rho, disc, m)
            if lam_r is not None:
                lam_total = lam_total + w_rev_explicit * lam_r
                hmat_total = hmat_total + w_rev_explicit * hmat_r

        grad[s:e] = lam_total.sum(axis=1) - lam_total.sum(axis=0)
        hess[s:e] = hmat_total.sum(axis=1) + hmat_total.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
