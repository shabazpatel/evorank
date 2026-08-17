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
    REV_CAP = 5000.0
    LOG_CAP = np.log1p(REV_CAP)
    W_REV = 0.5  # weight of the revenue lambda term added on top of NDCG lambda
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        bk = booking[s:e].astype(np.float64); rv = rev[s:e].astype(np.float64)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (lab[:, None] > lab[None, :])              # i strictly more relevant than j
        gain = (2.0 ** lab - 1.0)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ_ndcg = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg

        # Separate, boundedly-normalized revenue signal: does not touch the
        # relevance gain or the pair mask, so ndcg/book_ndcg structure is kept
        # intact while high-revenue bookings get an extra, self-normalized push.
        rev_gain = bk * np.log1p(np.clip(rv, 0.0, REV_CAP)) / LOG_CAP
        idcg_rev = float(np.sum(np.sort(rev_gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ_rev = np.abs((rev_gain[:, None] - rev_gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg_rev

        dZ = dZ_ndcg + W_REV * dZ_rev
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
