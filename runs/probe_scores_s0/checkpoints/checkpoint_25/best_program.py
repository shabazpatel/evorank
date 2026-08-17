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
    alpha = 0.5   # weight of rank-normalized-revenue term
    gamma = 0.2   # weight of explicit booking bonus term
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; r = rev[s:e]
        bk = booking[s:e].astype(np.float64); m = len(sc)
        if m < 2:
            continue
        p = _softmax(sigma * sc)

        gain = (2.0 ** lab - 1.0)
        booked_mask = r > 0
        r_norm = np.zeros_like(r, dtype=np.float64)
        n_booked = int(booked_mask.sum())
        if n_booked > 0:
            ranks = rankdata(r[booked_mask], method='average')
            r_norm[booked_mask] = ranks / n_booked
        t_raw = gain + alpha * r_norm + gamma * bk
        t_sum = t_raw.sum()
        if t_sum > 0:
            t = t_raw / t_sum
        else:
            t = np.full(m, 1.0 / m)

        grad[s:e] = p - t
        hess[s:e] = p * (1.0 - p)
    return grad, np.maximum(hess, 1e-3)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
