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
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        pos_disc = 1.0 / np.log2(np.arange(2, m + 2))

        # --- Term 1: pure relevance lambda (drives ndcg, matches baseline) ---
        gain_rel = 2.0 ** lab - 1.0
        M_rel = (lab[:, None] > lab[None, :])
        idcg_rel = float(np.sum(np.sort(gain_rel)[::-1] * pos_disc)) or 1.0
        dZ_rel = np.abs((gain_rel[:, None] - gain_rel[None, :]) * (disc[:, None] - disc[None, :])) / idcg_rel

        # --- Term 2: pure conversion lambda (binary booking gain). Decoupled
        # from revenue so book_ndcg gets a clean, undiluted conversion signal
        # regardless of price spread within the group. ---
        bk = booking[s:e].astype(np.float64)
        rv = rev[s:e].astype(np.float64)
        gain_conv = bk
        M_conv = (gain_conv[:, None] > gain_conv[None, :])
        idcg_conv = float(np.sum(np.sort(gain_conv)[::-1] * pos_disc)) or 1.0
        dZ_conv = np.abs((gain_conv[:, None] - gain_conv[None, :]) * (disc[:, None] - disc[None, :])) / idcg_conv

        # --- Term 3: revenue lambda using raw log1p(rev) (zero for
        # non-booked rows) as the gain. Magnitude differences between booked
        # rows' revenue directly drive dZ, and idcg_rev auto-normalizes the
        # scale per query -- no rank/percentile compression needed, so large
        # revenue gaps generate proportionally larger gradient signal. ---
        gain_rev = bk * np.log1p(np.maximum(rv, 0.0))
        M_rev = (gain_rev[:, None] > gain_rev[None, :])
        idcg_rev = float(np.sum(np.sort(gain_rev)[::-1] * pos_disc)) or 1.0
        dZ_rev = np.abs((gain_rev[:, None] - gain_rev[None, :]) * (disc[:, None] - disc[None, :])) / idcg_rev

        beta = 0.5   # weight of pure conversion term
        gamma = 0.7  # weight of revenue-magnitude term
        lam = (np.where(M_rel, -sigma * rho * dZ_rel, 0.0)
               + beta * np.where(M_conv, -sigma * rho * dZ_conv, 0.0)
               + gamma * np.where(M_rev, -sigma * rho * dZ_rev, 0.0))
        hmat = (np.where(M_rel, (sigma ** 2) * rho * (1.0 - rho) * dZ_rel, 0.0)
                + beta * np.where(M_conv, (sigma ** 2) * rho * (1.0 - rho) * dZ_conv, 0.0)
                + gamma * np.where(M_rev, (sigma ** 2) * rho * (1.0 - rho) * dZ_rev, 0.0))
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
