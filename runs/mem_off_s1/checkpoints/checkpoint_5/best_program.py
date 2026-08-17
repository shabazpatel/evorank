import numpy as np

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """LambdaMART with revenue-boosted gain.

    gain = (2**rel - 1) * (1 + w * rev_norm), where rev_norm is the
    per-query log1p(rev) scaled to [0,1]. This keeps relevance tiers
    dominant (0/1/5) but among same-tier items (esp. bookings) pushes
    higher-revenue rows to rank above lower-revenue ones, so a single
    |deltaNDCG|-weighted pairwise-logistic lambda jointly targets ndcg,
    book_ndcg, and revenue.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    w = 0.5
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e].astype(np.float64)
        rv = rev[s:e].astype(np.float64); m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        rv_log = np.log1p(rv)
        rv_max = rv_log.max()
        rv_norm = rv_log / rv_max if rv_max > 0 else np.zeros_like(rv_log)
        gain = (2.0 ** lab - 1.0) * (1.0 + w * rv_norm)
        M = (gain[:, None] > gain[None, :])            # i has strictly higher combined gain
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
