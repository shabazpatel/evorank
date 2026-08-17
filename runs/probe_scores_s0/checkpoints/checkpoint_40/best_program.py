import numpy as np
from scipy.special import softmax as _softmax
from scipy.stats import rankdata

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """Listwise softmax (ListNet-style) gradient with relevance + revenue-rank
    + booking-bonus target distribution.

    Per query group, p = softmax(sigma * predt) is the model's predicted
    top-1 attention distribution. The target distribution t is built from
    three additive components, then renormalized to sum to 1:
      1. gain = 2**rel - 1              (graded relevance signal)
      2. alpha * rank_revenue           (percentile rank of revenue among
                                          booked rows only, bounded [0,1],
                                          robust to price-scale outliers)
      3. gamma * booking                (explicit binary booking bonus,
                                          reinforcing conversion signal
                                          independent of rel's own booked=5)
    grad = p - t (softmax cross-entropy gradient); hess = p*(1-p) floored to
    stay strictly positive. This is a compact, single listwise pass (O(m))
    that separately nudges relevance ordering, revenue ordering among
    bookings, and raw conversion likelihood.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    alpha = 0.45  # weight of rank-normalized-revenue distribution
    beta = 0.20   # weight of explicit booking-bonus distribution
    power = 1.4   # sharpen top-revenue emphasis within the revenue distribution
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; r = rev[s:e]
        bk = booking[s:e].astype(np.float64); m = len(sc)
        if m < 2:
            continue
        p = _softmax(sigma * sc)

        # 1) relevance-gain distribution
        gain = (2.0 ** lab - 1.0)
        gsum = gain.sum()
        t_rel = gain / gsum if gsum > 0 else np.full(m, 1.0 / m)

        # 2) sharpened rank-of-revenue distribution among booked rows only
        booked_mask = r > 0
        n_booked = int(booked_mask.sum())
        r_dist = np.full(m, 1.0 / m)
        if n_booked > 0:
            r_norm = np.zeros(m, dtype=np.float64)
            ranks = rankdata(r[booked_mask], method='average')
            r_norm[booked_mask] = (ranks / n_booked) ** power
            rsum = r_norm.sum()
            if rsum > 0:
                r_dist = r_norm / rsum

        # 3) explicit booking-bonus distribution
        bsum = bk.sum()
        b_dist = bk / bsum if bsum > 0 else np.full(m, 1.0 / m)

        t = (1.0 - alpha - beta) * t_rel + alpha * r_dist + beta * b_dist

        grad[s:e] = p - t
        hess[s:e] = p * (1.0 - p)
    return grad, np.maximum(hess, 1e-3)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
