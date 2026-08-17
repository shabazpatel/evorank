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
    bookings appear (denser, more trustworthy ordering signal). This adaptive
    weighting avoids injecting unreliable revenue-ordering noise into the
    common sparse-booking queries while sharpening revenue signal when
    enough bookings exist to trust the ordering.
    """
    """Two-channel LambdaMART: decoupled relevance-NDCG and revenue-NDCG lambdas.

    Channel 1 (relevance): gain1 = 2**rel-1, the untouched standard pairwise-
    logistic |deltaNDCG| lambda, exactly as in the seed. This alone drives
    ndcg/book_ndcg and is never perturbed by revenue, so relevance ranking
    quality is preserved regardless of how much revenue weight is applied.

    Channel 2 (revenue): restricted to booked rows only, gain2 blends an
    order-robust percentile rank of revenue with a magnitude-aware normalized
    log-revenue (both in [0,1]). It gets its own ideal ordering / IDCG and its
    own |deltaNDCG_rev| pairwise lambda, using the SAME score-based ranks and
    discounts as channel 1 (so channels agree on "where" a position sits, but
    disagree on "what's good" there).

    The two channels' grad/hess are summed with a confidence-scaled weight
    (w_rev * conf(n_booked)): a single booking still gets a damped but
    non-zero revenue push, while queries with several bookings get a fuller,
    more trustworthy revenue-ordering push. Because the channels are additive
    rather than merged into one gain before computing dZ, the relevance
    channel's IDCG/ideal-order is never distorted by revenue, which should
    better protect ndcg/book_ndcg while still moving revenue.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    w_rev = 0.5          # weight of revenue channel relative to relevance channel
    conf_floor = 0.45    # min confidence with a single booking
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        rv = rev[s:e].astype(np.float64); bk = booking[s:e].astype(np.float64)

        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        disc_diff = disc[:, None] - disc[None, :]

        # --- Channel 1: relevance-driven lambda (unchanged from seed) ---
        gain1 = (2.0 ** lab - 1.0)
        M1 = (gain1[:, None] - gain1[None, :]) > 1e-9
        idcg1 = float(np.sum(np.sort(gain1)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ1 = np.abs((gain1[:, None] - gain1[None, :]) * disc_diff) / idcg1
        lam1 = np.where(M1, -sigma * rho * dZ1, 0.0)
        hmat1 = np.where(M1, (sigma ** 2) * rho * (1.0 - rho) * dZ1, 0.0)

        grad_g = lam1.sum(axis=1) - lam1.sum(axis=0)
        hess_g = hmat1.sum(axis=1) + hmat1.sum(axis=0)

        # --- Channel 2: revenue-driven lambda (booked rows only) ---
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
            gain2 = np.zeros(m, dtype=np.float64)
            gain2[idx] = combined_rev
            conf = conf_floor + (1.0 - conf_floor) * min(1.0, idx.size / 3.0)

            M2 = (gain2[:, None] - gain2[None, :]) > 1e-9
            idcg2 = float(np.sum(np.sort(gain2)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
            dZ2 = np.abs((gain2[:, None] - gain2[None, :]) * disc_diff) / idcg2
            lam2 = np.where(M2, -sigma * rho * dZ2, 0.0)
            hmat2 = np.where(M2, (sigma ** 2) * rho * (1.0 - rho) * dZ2, 0.0)

            wgt = w_rev * conf
            grad_g = grad_g + wgt * (lam2.sum(axis=1) - lam2.sum(axis=0))
            hess_g = hess_g + wgt * (hmat2.sum(axis=1) + hmat2.sum(axis=0))

        grad[s:e] = grad_g
        hess[s:e] = hess_g
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
