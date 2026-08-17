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
    """Approach: mask/ranking direction driven purely by relevance (as in
    the ndcg-optimal baseline), preserving ndcg/book_ndcg quality. Revenue
    enters two ways: (1) same-relevance ties are broken by higher revenue
    (M_tie), and (2) for strict relevance pairs, the |deltaNDCG| lambda
    magnitude is boosted (via a bounded tanh consistency term) when the
    higher-relevance item also has higher log-revenue, and mildly damped
    when revenue disagrees. This nudges gradient magnitude toward
    revenue-consistent orderings without ever flipping the relevance-based
    ranking direction, aiming to lift revenue while keeping ndcg/book_ndcg
    close to the pure-relevance baseline.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    beta_tie = 0.5   # revenue tie-break weight (same-relevance pairs)
    gamma_rel = 0.35  # revenue-consistency magnitude boost for relevance pairs
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        rev_s = rev[s:e].astype(np.float64)
        rev_gain = np.log1p(np.maximum(rev_s, 0.0))
        gain = (2.0 ** lab - 1.0)                       # pure relevance gain
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0

        M_rel = (lab[:, None] > lab[None, :])          # i strictly more relevant than j
        tie = (lab[:, None] == lab[None, :])
        M_tie = tie & (rev_s[:, None] > rev_s[None, :])  # break same-relevance ties by revenue
        M = M_rel | M_tie

        dZ_rel = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        rev_diff = rev_gain[:, None] - rev_gain[None, :]
        consistency = np.tanh(gamma_rel * rev_diff)     # in (-1, 1); positive when i also has higher revenue
        dZ_rel_boosted = dZ_rel * (1.0 + consistency)   # stays >= 0 since factor in (0, 2)
        dZ_rev = beta_tie * np.abs(rev_diff * (disc[:, None] - disc[None, :])) / idcg
        dZ = np.where(M_rel, dZ_rel_boosted, dZ_rev)

        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
