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
    # Revenue-blended NDCG lambda-gradient (re-tuned): augment the relevance
    # gain (2**rel - 1) with a bounded, log-compressed revenue term so the
    # same |delta NDCG| pairwise machinery pulls high-value bookings toward
    # rank 1 harder than low-value bookings. REV_CAP=5000 sits closer to the
    # bulk of the revenue distribution (p95~1223) than a looser cap, so the
    # log1p compression keeps more resolution in the common range instead of
    # spending dynamic range on the rare 50k+ tail; REV_WEIGHT=10 slightly
    # increases the revenue nudge relative to the booking/click gain gap (31)
    # for a stronger pull on high-value bookings while still blending rather
    # than replacing relevance ranking. idcg uses the same blended gain for
    # self-consistency; pair mask, sigmoid, discount, and hessian formula are
    # unchanged from baseline LambdaMART.
    REV_CAP = 5000.0
    REV_WEIGHT = 10.0
    log_cap = np.log1p(REV_CAP)
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        r = rev[s:e].astype(np.float64)
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (lab[:, None] > lab[None, :])              # i strictly more relevant than j
        rev_term = REV_WEIGHT * (np.log1p(np.clip(r, 0.0, REV_CAP)) / log_cap)
        gain = (2.0 ** lab - 1.0) + rev_term
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
