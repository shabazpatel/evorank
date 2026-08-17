import numpy as np
from scipy.special import softmax as _softmax
from scipy.stats import rankdata

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """Listwise softmax (ListNet-style) gradient with three-way blended target.

    For each query group, p = softmax(sigma * predt) is the model's predicted
    top-1 attention distribution. The target t is a convex combination of
    three per-group distributions, each individually normalized to sum to 1:
      1. t_rel:  relevance-gain distribution, gain = 2**rel - 1
      2. t_rev:  percentile-rank distribution over revenue among booked rows
                 only (rank/n_booked), robust to price-scale outliers; falls
                 back to uniform if nothing booked
      3. t_book: uniform-over-booked-rows distribution (booking indicator
                 normalized), an explicit conversion-only signal independent
                 of rel's own booked=5 grading
    Mixing weights alpha (revenue) and beta (booking) trade off relevance,
    revenue ranking, and conversion. grad = p - t (softmax cross-entropy
    gradient); hess = p*(1-p) floored to stay strictly positive.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    alpha = 0.30  # weight of rank-normalized-revenue distribution
    beta = 0.15   # weight of explicit booking-bonus distribution
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; r = rev[s:e]
        bk = booking[s:e].astype(np.float64); m = len(sc)
        if m < 2:
            continue
        p = _softmax(sigma * sc)

        gain = (2.0 ** lab - 1.0)
        gsum = gain.sum()
        t_rel = gain / gsum if gsum > 0 else np.full(m, 1.0 / m)

        booked = r > 0
        n_booked = int(booked.sum())
        r_dist = np.full(m, 1.0 / m)
        if n_booked > 0:
            r_norm = np.zeros(m, dtype=np.float64)
            ranks = rankdata(r[booked], method='average')
            r_norm[booked] = ranks / n_booked
            rsum = r_norm.sum()
            if rsum > 0:
                r_dist = r_norm / rsum

        bsum = bk.sum()
        b_dist = bk / bsum if bsum > 0 else np.full(m, 1.0 / m)

        t = (1.0 - alpha - beta) * t_rel + alpha * r_dist + beta * b_dist

        grad[s:e] = p - t
        hess[s:e] = p * (1.0 - p)
    return grad, np.maximum(hess, 1e-3)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
