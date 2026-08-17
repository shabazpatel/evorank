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
    """Multi-channel LambdaMART: separate NDCG-style pairwise lambdas for
    relevance, booking, and revenue, summed per query.

    For each query we build the standard pairwise-logistic lambda/hessian
    three times, once per gain definition (graded relevance -> targets
    ndcg, binary booking -> targets book_ndcg, log1p(revenue) -> targets
    revenue), each normalized by its own ideal-DCG so the three channels
    are on comparable scales. The channels are combined with fixed weights
    before being summed into row-level grad/hess. This lets each metric
    receive genuine ranking pressure (not just same-tier tie-breaks),
    while w_rel keeps relevance dominant and w_book/w_rev add targeted
    pressure for the other two objectives.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    w_rel, w_book, w_rev = 1.0, 0.35, 0.45

    def channel_lambda(gain, sc, rho, disc, m):
        M = gain[:, None] > gain[None, :]
        if not M.any():
            return None, None
        order = np.sort(gain)[::-1]
        ideal_disc = 1.0 / np.log2(np.arange(2, m + 2))
        idcg = float(np.sum(order * ideal_disc)) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        return lam, hmat

    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64)
        lab = rel[s:e].astype(np.float64)
        bk = booking[s:e].astype(np.float64)
        rv = np.log1p(rev[s:e].astype(np.float64))
        m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)

        lam_total = np.zeros((m, m), dtype=np.float64)
        hmat_total = np.zeros((m, m), dtype=np.float64)

        gain_rel = 2.0 ** lab - 1.0
        lam, hmat = channel_lambda(gain_rel, sc, rho, disc, m)
        if lam is not None:
            lam_total += w_rel * lam
            hmat_total += w_rel * hmat

        if bk.max() > 0:
            lam, hmat = channel_lambda(bk, sc, rho, disc, m)
            if lam is not None:
                lam_total += w_book * lam
                hmat_total += w_book * hmat

        if rv.max() > 0:
            lam, hmat = channel_lambda(rv, sc, rho, disc, m)
            if lam is not None:
                lam_total += w_rev * lam
                hmat_total += w_rev * hmat

        grad[s:e] = lam_total.sum(axis=1) - lam_total.sum(axis=0)
        hess[s:e] = hmat_total.sum(axis=1) + hmat_total.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
