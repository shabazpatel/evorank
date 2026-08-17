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
    """Decomposed dual-objective LambdaMART.

    Computes two independent, NDCG-consistent lambda/hessian matrices per
    query: (1) the pure-relevance term (identical to the original baseline,
    so ndcg is preserved) and (2) a booking/revenue term with its own ideal
    ordering, idcg normalization, and rank-discount. The booking/revenue
    gain uses within-group percentile rank of revenue (robust to outlier
    bookings, avoids raw-scale/log1p sensitivity) combined with the binary
    booking flag (weight 2), plus a small partial-credit bonus for rows
    that were clicked but not booked so book_ndcg benefits from click
    signal too. The two lambda matrices are linearly combined (relevance
    term full weight, booking/revenue term weight alpha=0.75) before
    aggregating into per-row grad/hess, letting the booking/revenue signal
    act on all pairs with its own calibrated discount curve while leaving
    the ndcg-driving relevance term essentially untouched.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    alpha = 0.65  # weight on booking/revenue lambda term relative to rel term
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
        # Log1p + group-max normalization preserves relative revenue
        # magnitude (unlike pure percentile rank), aligning the gradient
        # more directly with the revenue metric while still being robust
        # to scale via the log transform. Booking flag is added at unit
        # weight (not 2x) so within-booked revenue differences are not
        # swamped, and a small click-only bonus keeps some book_ndcg lift
        # from click signal.
        rev_log = np.log1p(rv)
        rmax = rev_log.max()
        rev_norm = rev_log / rmax if rmax > 0 else np.zeros(m)
        click_only = (lab == 1.0) & (bk == 0.0)
        gain_bk = bk + rev_norm + 0.1 * click_only
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
