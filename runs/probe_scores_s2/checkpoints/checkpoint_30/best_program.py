import numpy as np

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """LambdaMART lambda-gradient with confidence-scaled revenue bonus.

    gain = (2**rel-1) + gamma*conf(n_booked)*31*(0.45*pct_rank+0.55*norm_log_rev)
    Blends order-robust percentile rank with magnitude-aware normalized
    log-revenue among booked rows, damped by a confidence ramp on booking
    count, then drives standard pairwise-logistic |deltaNDCG| lambdas.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    gamma = 0.6         # max weight of revenue bonus relative to top rel gain (31)
    conf_floor = 0.45   # min confidence with a single booking
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        rv = rev[s:e].astype(np.float64); bk = booking[s:e].astype(np.float64)
        gain = (2.0 ** lab - 1.0)
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
            combined_rev = 0.45 * pct + 0.55 * norm_log
            rev_grade = np.zeros(m, dtype=np.float64)
            rev_grade[idx] = combined_rev
            conf = conf_floor + (1.0 - conf_floor) * min(1.0, idx.size / 3.0)
            gain = gain + (gamma * conf) * 31.0 * rev_grade
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (gain[:, None] - gain[None, :]) > 1e-9     # i strictly better than j
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
