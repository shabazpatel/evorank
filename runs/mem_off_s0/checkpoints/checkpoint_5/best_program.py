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
    """Decomposed dual-objective LambdaMART: computes two independent,
    NDCG-consistent lambda/hessian matrices per query -- one over the pure
    relevance grade (identical to the baseline, so ndcg is preserved as
    closely as possible) and one over a booking+revenue "ideal ordering"
    (its own rank-discount and idcg normalization, using log1p-scaled,
    group-max-normalized revenue combined with the binary booking flag as
    the gain). The two lambda matrices are linearly combined (rel term full
    weight, book/revenue term weight alpha) before aggregating into
    per-row grad/hess. Unlike merging gains into one NDCG or only adding
    tie-break pairs among equal-rel rows, this lets the booking/revenue
    signal act on *all* pairs with its own well-calibrated discount curve,
    giving a stronger, independently-tunable push on book_ndcg/revenue
    while leaving the rel-driven ndcg term untouched.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    alpha = 0.6  # weight on booking/revenue lambda term relative to rel term
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e].astype(np.float64); m = len(sc)
        bk = booking[s:e].astype(np.float64); rv = rev[s:e].astype(np.float64)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)

        # --- relevance term (baseline, unchanged) ---
        M_rel = (lab[:, None] > lab[None, :])
        gain_rel = (2.0 ** lab - 1.0)
        idcg_rel = float(np.sum(np.sort(gain_rel)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ_rel = np.abs((gain_rel[:, None] - gain_rel[None, :]) * (disc[:, None] - disc[None, :])) / idcg_rel
        lam_rel = np.where(M_rel, -sigma * rho * dZ_rel, 0.0)
        hmat_rel = np.where(M_rel, (sigma ** 2) * rho * (1.0 - rho) * dZ_rel, 0.0)

        # --- booking/revenue term (independent ideal ordering) ---
        rev_log = np.log1p(rv)
        rmax = rev_log.max()
        rev_norm = rev_log / rmax if rmax > 0 else np.zeros(m)
        gain_bk = bk + rev_norm
        M_bk = (gain_bk[:, None] > gain_bk[None, :])
        idcg_bk = float(np.sum(np.sort(gain_bk)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ_bk = np.abs((gain_bk[:, None] - gain_bk[None, :]) * (disc[:, None] - disc[None, :])) / idcg_bk
        lam_bk = np.where(M_bk, -sigma * rho * dZ_bk, 0.0)
        hmat_bk = np.where(M_bk, (sigma ** 2) * rho * (1.0 - rho) * dZ_bk, 0.0)

        lam = lam_rel + alpha * lam_bk
        hmat = hmat_rel + alpha * hmat_bk
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
