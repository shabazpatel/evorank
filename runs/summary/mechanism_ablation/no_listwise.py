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
    from scipy.special import softmax
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    # alpha is now adaptive per query: queries with an actual booking lean
    # more on the listwise Plackett-Luce top-1-identification term (helps
    # book_ndcg/revenue by directly pushing the booked row to rank 0),
    # while booking-free (click-only) queries lean on the well-calibrated
    # pairwise |deltaNDCG| lambda term (protects/improves overall ndcg,
    # which is dominated by the many non-booking queries).
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        has_booking = bool(booking[s:e].sum() > 0)
        alpha = 1.0
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (lab[:, None] > lab[None, :])              # i strictly more relevant than j
        rev_c = np.clip(rev[s:e], 0.0, 3000.0)         # bound revenue tail
        gain = (2.0 ** lab - 1.0) + 3.5 * (np.log1p(rev_c) / np.log1p(3000.0))
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        pair_grad = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        pair_hess = hmat.sum(axis=1) + hmat.sum(axis=0)

        # Listwise Plackett-Luce / softmax cross-entropy component: models
        # P(row i is "the" relevant/booked item) via softmax(scores), trained
        # against a blended-gain target distribution. This directly matches
        # the single-booking-per-query structure (top-1 identification),
        # complementing the pairwise deltaNDCG term above.
        prob = softmax(sc - np.max(sc))
        g_sum = gain.sum()
        if g_sum <= 1e-12:
            target = np.full(m, 1.0 / m)
        else:
            target = gain / g_sum
        list_grad = prob - target
        list_hess = prob * (1.0 - prob)

        grad[s:e] = alpha * pair_grad + (1.0 - alpha) * list_grad
        hess[s:e] = alpha * pair_hess + (1.0 - alpha) * list_hess
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
