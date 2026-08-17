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
    # Revenue-blended NDCG lambda-gradient: augment the relevance gain
    # (2**rel - 1) with a bounded, log-compressed revenue term so the same
    # |delta NDCG| pairwise machinery also pulls high-value bookings toward
    # rank 1 harder than low-value bookings. Revenue is clipped at 5000 (well
    # above p95~1223) and log1p-compressed, then rescaled to a magnitude
    # (<= ~10) smaller than the booking/click gain gap (31), so it nudges
    # ordering among relevant rows without overwhelming rel-based ranking or
    # perturbing non-booked (rev=0) rows. idcg uses the same blended gain for
    # self-consistency. Everything else (pair mask, sigmoid, discount, hess)
    # is unchanged from the baseline.
    # Tuned: lower REV_CAP (closer to p95~1223) sharpens log-resolution across
    # typical bookings instead of compressing them into a narrow band near a
    # far-off cap. Revenue now acts through two independent, smaller levers
    # instead of one large one: (1) an additive gain term (REV_WEIGHT, reduced
    # from 10->6) that shifts idcg/gain modestly, and (2) a multiplicative
    # pair-importance weight (PAIR_REV_SCALE) applied only to pairs whose more
    # relevant row is an actual booking (rel==5), which amplifies the pairwise
    # force for high-revenue bookings without further distorting the gain
    # used for NDCG normalization. Both are capped/log-compressed so no rare
    # huge booking can dominate; combined magnitude stays comparable to the
    # parent's single-lever design to avoid collapsing ndcg/book_ndcg.
    # Decoupled dual-metric lambda-gradient: instead of blending revenue into
    # the relevance gain (which distorts the idcg normalizer used for the
    # ndcg-aligned term and hurts the reported ndcg/book_ndcg metrics), we
    # keep the standard relevance |delta NDCG| term completely untouched --
    # gain = 2**rel - 1, its own idcg -- so it stays faithful to how ndcg and
    # book_ndcg are actually scored. We then add a second, independently
    # normalized |delta metric| term whose "gain" is a bounded, log-
    # compressed revenue fraction (own ideal-DCG idcg_rev), mirroring the
    # exact same lambda-gradient math but targeted purely at revenue. Because
    # only the (single) booked row per query has nonzero revenue, this second
    # term only exerts extra pairwise force pulling that booking above
    # clicked/unclicked rows, proportional to how much money it is worth,
    # without ever touching the relevance gain vector. Summing the two dZ
    # terms (LAMBDA_REV controls relative strength) before the sigmoid/hess
    # step keeps grad/hess consistent and bounded; both terms individually
    # obey the LambdaLoss bound-on-a-metric interpretation, so this is a
    # principled "gain vector reflecting all three metrics" rather than an ad
    # hoc reweighting.
    # Three-way decoupled lambda-gradient (one term per reported metric).
    # Rather than blending revenue/booking into the relevance gain (which
    # distorts the idcg normalizer that ndcg/book_ndcg are actually scored
    # against), we build three *independently normalized* |delta metric|
    # terms, each mirroring the exact metric it targets:
    #   dZ_rel  -> gain = 2**rel-1,        own idcg  (drives "ndcg")
    #   dZ_book -> gain = booking (0/1),   own idcg  (drives "book_ndcg")
    #   dZ_rev  -> gain = log1p-compressed revenue fraction, own idcg
    #              (drives the dollar-weighted "revenue" metric)
    # All three share the same pair mask M (i strictly more relevant than j
    # by graded rel) since booking/revenue are always a subset of the
    # highest-rel row per query, so no extra pairs are needed. Summing the
    # three dZ's before the sigmoid/hess step keeps grad/hess finite and
    # positive while giving each metric its own principled, bounded lever
    # (LambdaLoss-style bound-on-a-metric), rather than one ad hoc blend.
    REV_CAP = 3000.0
    W_REL = 1.0
    W_BOOK = 1.0
    W_REV = 1.2
    log_cap = np.log1p(REV_CAP)
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        b = booking[s:e].astype(np.float64)
        r = rev[s:e].astype(np.float64)
        rev_frac = np.log1p(np.clip(r, 0.0, REV_CAP)) / log_cap
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma * S)
        M = (lab[:, None] > lab[None, :])              # i strictly more relevant than j
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        disc_diff = disc[:, None] - disc[None, :]

        def _dZ(gain):
            idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
            return np.abs((gain[:, None] - gain[None, :]) * disc_diff) / idcg

        gain_rel = (2.0 ** lab - 1.0)
        dZ_rel = _dZ(gain_rel)
        dZ_book = _dZ(b)
        dZ_rev = _dZ(rev_frac)

        dZ = W_REL * dZ_rel + W_BOOK * dZ_book + W_REV * dZ_rev

        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
