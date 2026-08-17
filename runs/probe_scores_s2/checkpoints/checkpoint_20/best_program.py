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
    """Confidence-scaled percentile+log revenue bonus on top of relevance gain.

    gain = (2**rel-1) + gamma*conf(n_booked)*31*(0.5*pct_rank+0.5*norm_log_rev)
    over booked rows only. pct_rank is order-only (robust), norm_log_rev is
    magnitude-aware (both bounded [0,1] per query). conf ramps from a
    non-zero floor (single booking still gets a real, if damped, boost -
    it legitimately signals "beats all non-booked rows") up to 1 as more
    bookings appear (denser, more trustworthy ordering signal).
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    gamma = 0.55        # weight of revenue bonus relative to top rel gain (31)
    conf_floor = 0.45   # min confidence with a single booking
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        rv = rev[s:e].astype(np.float64); bk = booking[s:e].astype(np.float64)
        gain = (2.0 ** lab - 1.0)
        idx = np.where(bk > 0)[0]
        if idx.size >= 1:
            sub = rv[idx]
            if idx.size > 1:
                order = np.argsort(sub, kind="stable")
                pct = np.empty(idx.size, dtype=np.float64)
                pct[order] = np.arange(idx.size, dtype=np.float64) / (idx.size - 1)
            else:
                pct = np.array([1.0])
            log_rev = np.log1p(np.maximum(sub, 0.0))
            max_log = log_rev.max()
            norm_log = log_rev / max_log if max_log > 0 else np.zeros_like(log_rev)
            combined_rev = 0.5 * pct + 0.5 * norm_log
            rev_grade = np.zeros(m, dtype=np.float64)
            rev_grade[idx] = combined_rev
            conf = conf_floor + (1.0 - conf_floor) * min(1.0, idx.size / 3.0)
            gain = gain + (gamma * conf) * 31.0 * rev_grade
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (gain[:, None] - gain[None, :]) > 1e-9     # i strictly better than j
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
