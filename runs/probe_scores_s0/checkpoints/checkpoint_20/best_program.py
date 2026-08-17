import numpy as np
from scipy.special import softmax as _softmax

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """Listwise softmax (ListNet-style) gradient over blended relevance+revenue.

    For each query group, treat p = softmax(sigma * predt) as the model's
    predicted top-1 attention distribution, and build a target distribution
    t proportional to a blended gain (2**rel - 1) + alpha * normalized_revenue,
    renormalized to sum to 1 (falling back to uniform if all gains are zero).
    The gradient is the standard softmax cross-entropy gradient grad = p - t,
    and the Hessian uses the diagonal softmax curvature approximation
    hess = p * (1 - p), floored to stay strictly positive. This replaces the
    O(m^2) pairwise deltaNDCG lambda computation with an O(m) listwise
    formulation that directly matches the predicted distribution to a
    relevance+revenue blended target, which should yield smoother, less
    noisy gradients while still favoring booked/high-revenue rows at the top.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    alpha = 0.5  # weight of normalized-revenue term in the target distribution
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; r = rev[s:e]; m = len(sc)
        if m < 2:
            continue
        p = _softmax(sigma * sc)

        gain = (2.0 ** lab - 1.0)
        rmax = r.max()
        r_norm = r / rmax if rmax > 0 else np.zeros_like(r)
        t_raw = gain + alpha * r_norm
        t_sum = t_raw.sum()
        if t_sum > 0:
            t = t_raw / t_sum
        else:
            t = np.full(m, 1.0 / m)

        grad[s:e] = p - t
        hess[s:e] = p * (1.0 - p)
    return grad, np.maximum(hess, 1e-3)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
