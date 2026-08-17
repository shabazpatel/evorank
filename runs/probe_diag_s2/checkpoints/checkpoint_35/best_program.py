import numpy as np
from scipy.special import softmax

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """Blended pairwise LambdaMART + listwise ListNet-style softmax objective.

    For each query group, computes the existing pairwise lambda gradient
    (relevance + booking + log-revenue deltaNDCG weighting, averaged) AND a
    listwise ListNet-style cross-entropy term: softmax probabilities
    p_i = softmax(score)_i are pushed toward a target distribution t_i
    proportional to (2**rel-1 + booking + normalized revenue). The two
    gradient/hessian pairs are blended via a fixed convex combination
    (0.7 pairwise, 0.3 listwise) so the listwise term captures global list
    structure (multiple positives, ties) that pure pairwise lambdas can
    under-weight, while the pairwise term continues to anchor plain ndcg.
    """
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
    """
    Multi-signal LambdaMART: instead of weighting pairs by a single
    relevance-derived deltaNDCG, we compute THREE independent
    deltaNDCG-style pair weights per query -- one from graded relevance
    (2**rel - 1), one from binary booking (targets book_ndcg), and one
    from log1p(revenue) (targets revenue) -- each with its own per-query
    IDCG normalization and its own "i more relevant than j" pair mask.
    The three weighted lambda contributions are summed (equally, then
    averaged) into a single per-row gradient/hessian so a single tree
    ensemble jointly optimizes relevance ranking, booking ranking, and
    revenue ranking. This directly exposes ndcg / book_ndcg / revenue as
    combinable objectives rather than only using rel.
    """
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)

    def _pair_weight(gain, disc, m):
        srt = np.sort(gain)[::-1]
        idcg = float(np.sum(srt * (1.0 / np.log2(np.arange(2, m + 2)))))
        idcg = idcg if idcg > 1e-12 else 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        M = gain[:, None] > gain[None, :]
        return dZ * M

    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64)
        m = len(sc)
        if m < 2:
            continue
        lab_rel = rel[s:e].astype(np.float64)
        lab_book = booking[s:e].astype(np.float64)
        lab_rev = np.log1p(np.maximum(rev[s:e].astype(np.float64), 0.0))

        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)

        ranks = np.empty(m, dtype=int)
        ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)

        gain_rel = 2.0 ** lab_rel - 1.0
        w_rel = _pair_weight(gain_rel, disc, m)
        w_book = _pair_weight(lab_book, disc, m)
        w_rev = _pair_weight(lab_rev, disc, m)

        dZ = (w_rel + w_book + w_rev) / 3.0

        lam = -sigma * rho * dZ
        hmat = (sigma ** 2) * rho * (1.0 - rho) * dZ
        grad_pair = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess_pair = hmat.sum(axis=1) + hmat.sum(axis=0)

        # --- listwise ListNet-style softmax cross-entropy component ---
        r = np.maximum(rev[s:e].astype(np.float64), 0.0)
        rmax = float(r.max())
        r_norm = r / rmax if rmax > 1e-9 else r
        target_raw = gain_rel + lab_book + r_norm
        denom = float(target_raw.sum())
        t = target_raw / denom if denom > 1e-12 else np.full(m, 1.0 / m)
        p = softmax(sc)
        grad_list = p - t
        hess_list = p * (1.0 - p)

        W_PAIR, W_LIST = 0.7, 0.3
        grad[s:e] = W_PAIR * grad_pair + W_LIST * grad_list
        hess[s:e] = W_PAIR * hess_pair + W_LIST * hess_list
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
