import numpy as np

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """LambdaMART with |deltaNDCG| pair weight, scaled by log-revenue of the
    more-relevant row (booking rows only, since rev=0 otherwise) so
    high-value bookings get pushed harder toward the top without touching
    click-vs-nonclick pairs that drive ndcg/book_ndcg."""
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))
        M = (lab[:, None] > lab[None, :])
        gain = (2.0 ** lab - 1.0)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        rv = rev[s:e].astype(np.float64)
        rw = 1.0 + 0.6 * np.clip(np.log1p(rv) / np.log1p(3000.0), 0.0, 1.0)
        dZ = dZ * rw[:, None]
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
