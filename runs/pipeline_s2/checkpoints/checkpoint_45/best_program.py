"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Feature set: within-query rank transforms (top documented recipe,
    +0.011 ndcg alone) combined with count encodings of high-cardinality ids
    (+0.013 combined, best hand-built variant), plus a small set of causal
    composite price/location anchors. Kept intentionally lean (~25 cols)
    since prior evidence shows bloat hurts. Location score 2 is imputed with
    a low train-fold quantile instead of zero-fill."""
    out = {}

    # low-quantile imputation for location score 2 (missing -> low value)
    loc2_fill = stats.quantile("prop_location_score2", 0.05)
    loc2 = df["prop_location_score2"].fillna(loc2_fill)
    out["prop_location_score2_imp"] = loc2

    # within-query rank transforms (listwise signal)
    out["price_rank"] = df.groupby("qid")["price_usd"].rank()
    out["starrating_rank"] = df.groupby("qid")["prop_starrating"].rank()
    out["loc_score2_rank"] = loc2.groupby(df["qid"]).rank()
    out["review_score_rank"] = df.groupby("qid")["prop_review_score"].rank()
    out["loc_score1_rank"] = df.groupby("qid")["prop_location_score1"].rank()

    # count encodings of high-cardinality ids
    out["prop_id_count"] = stats.count(df, "prop_id")
    out["dest_id_count"] = stats.count(df, "srch_destination_id")

    # composite price/location anchors with a causal story
    import numpy as np
    out["ump"] = np.exp(df["prop_log_historical_price"]) - df["price_usd"]
    out["price_diff"] = df["visitor_hist_adr_usd"] - df["price_usd"]
    out["starrating_diff"] = df["visitor_hist_starrating"] - df["prop_starrating"]
    # per-query normalized price (cheap diversity signal)
    price_mean = df.groupby("qid")["price_usd"].transform("mean")
    price_std = df.groupby("qid")["price_usd"].transform("std").replace(0, 1)
    out["price_z"] = (df["price_usd"] - price_mean) / price_std

    # stacked score reproducing top-importance mechanism
    out["stacked_score"] = stats.stacked_score(df)

    # revenue-oriented signal: booking-rate encoding on a low-cardinality,
    # non-sparse column (prop_country_id), interacted with price to
    # approximate expected-revenue rather than plain price level.
    country_book_rate = stats.rate(df, "prop_country_id", "booking")
    out["country_book_rate"] = country_book_rate
    out["price_x_country_rate"] = df["price_usd"] * country_book_rate

    # a few raw core features retained directly
    for c in ["price_usd", "prop_starrating", "prop_review_score",
              "prop_location_score1", "promotion_flag", "prop_brand_bool",
              "srch_length_of_stay", "srch_booking_window",
              "srch_saturday_night_bool"]:
        out[c] = df[c]

    import pandas as pd
    return pd.DataFrame(out, index=df.index)
# EVOLVE-BLOCK-END: features

# EVOLVE-BLOCK-START: pipeline
PIPELINE = {
    # Two diverse strong members: LambdaMART (xgb rank:ndcg) as the primary
    # ranker, plus lgbm lambdarank for family/impl diversity, combined via
    # per-query z-score weighted sum (challenge-winning combiner). Weight
    # the stronger member higher per priors.
    "models": [
        {"family": "xgb", "objective": "rank:ndcg",
         "params": {"eta": 0.08, "max_depth": 6, "min_child_weight": 0.5,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "num_boost_round": 200}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.06, "num_leaves": 63,
                     "min_child_samples": 20, "subsample": 0.8,
                     "colsample_bytree": 0.8, "n_estimators": 200}},
        {"family": "sklearn", "objective": "ert",
         "params": {"n_estimators": 140, "max_depth": 9,
                    "min_samples_leaf": 10}},
    ],
    # Genuinely different model family (extremely randomized trees) adds
    # diverse error structure vs. the two GBM rankers, per documented
    # priors that same-family variants (e.g. a second xgb loss) waste
    # ensemble budget. Weighted a bit higher than the closest comparable
    # variant to explore a different point on the book_ndcg/revenue
    # trade-off surface.
    "ensemble": {"type": "zscore_weighted", "weights": [0.43, 0.35, 0.22]},
}
# EVOLVE-BLOCK-END: pipeline
