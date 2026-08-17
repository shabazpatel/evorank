import numpy as np

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.379):
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); r = rel[s:e]; b = booking[s:e]; v = rev[s:e]
        m = len(sc)
        if m < 2:
            continue
        rev_term = np.sqrt(v)
        gain = 0.452 * (2.0 ** r - 1.0) + 2.491 * b + 1.442 * rev_term
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))
        M = (gain[:, None] > gain[None, :])
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0) ** 1.331
        ideal = np.sort(gain)[::-1]
        idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, m + 2)) ** 1.331))) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)
# EVOLVE-BLOCK-END: objective
