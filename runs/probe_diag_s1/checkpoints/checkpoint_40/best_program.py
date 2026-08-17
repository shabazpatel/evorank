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
        M = (lab[:, None] > lab[None, :])              # i strictly more relevant than j
        gain_rel = (2.0 ** lab - 1.0)
        # Revenue-aware gain boost: sqrt-compress booking revenue (sub-linear,
        # softer than log1p so mid/high revenue bookings retain more relative
        # weight) and rescale per-query to be comparable to gain_rel's max
        # (~31 at rel=5). Only booked rows get the boost so click/no-click
        # ordering (which drives ndcg/book_ndcg) is preserved as the primary
        # signal, with revenue refining ties among bookings.
        # Rank-based revenue gain: within the booked subset of this query,
        # convert revenue to a fractional rank in (0, 1] (highest revenue
        # booking -> 1.0). This is scale-free and outlier-robust (no percentile
        # or magnitude rescale needed), then lifted to gain_rel's range so it
        # nudges ordering among bookings without dominating relevance/clicks.
        rev_slice = rev[s:e].astype(np.float64)
        book_slice = booking[s:e].astype(np.float64)
        rev_gain = np.zeros(m, dtype=np.float64)
        booked_idx = np.where(book_slice > 0)[0]
        nb = booked_idx.size
        if nb > 0:
            order = np.argsort(rev_slice[booked_idx], kind="stable")
            frac_rank = np.empty(nb, dtype=np.float64)
            frac_rank[order] = (np.arange(1, nb + 1)) / nb
            rev_gain[booked_idx] = frac_rank
        w_rev = 0.45
        gain = gain_rel + w_rev * rev_gain * 31.0
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
