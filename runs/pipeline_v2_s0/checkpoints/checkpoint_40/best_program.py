"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """qrank + counts + raw anchors + price composites + comp-rate +
    stacked_score (~15 cols)."""
    import numpy as np
    import pandas as pd
    out = {}

    for c in ["price_usd", "prop_starrating", "prop_review_score",
              "prop_location_score1", "prop_location_score2"]:
        out[c + "_qrank"] = df.groupby("qid")[c].rank(method="average",
                                                       na_option="bottom")

    out["prop_id_count"] = stats.count(df, "prop_id")
    out["dest_id_count"] = stats.count(df, "srch_destination_id")

    loc2_fill = stats.quantile("prop_location_score2", 0.05)
    out["price_usd"] = df["price_usd"]
    out["prop_starrating"] = df["prop_starrating"]
    out["prop_review_score"] = df["prop_review_score"]
    out["prop_location_score2"] = df["prop_location_score2"].fillna(loc2_fill)
    out["promotion_flag"] = df["promotion_flag"]
    out["prop_brand_bool"] = df["prop_brand_bool"]

    out["ump"] = np.exp(df["prop_log_historical_price"]) - df["price_usd"]
    out["price_diff"] = df["visitor_hist_adr_usd"] - df["price_usd"]

    # comp*_rate mean: competitor price-competitiveness proxy
    comp_cols = [c for c in df.columns if c.endswith("_rate")
                 and c.startswith("comp")]
    if comp_cols:
        out["comp_rate_mean"] = df[comp_cols].mean(axis=1)

    out["stacked_score"] = stats.stacked_score(df)

    return pd.DataFrame(out, index=df.index)
# EVOLVE-BLOCK-END: features

# EVOLVE-BLOCK-START: pipeline
PIPELINE = {
    # xgb rank:ndcg + lgbm lambdarank (top LambdaMART pair) + sklearn ert
    # (cheap structural diversity, different family/training dynamics).
    "models": [
        {"family": "xgb", "objective": "rank:ndcg",
         "params": {"eta": 0.1, "max_depth": 6, "min_child_weight": 0.1,
                    "num_boost_round": 120}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.08, "num_leaves": 63,
                    "min_child_samples": 20, "n_estimators": 200,
                    "subsample": 0.8, "colsample_bytree": 0.8}},
        {"family": "sklearn", "objective": "ert",
         "params": {"n_estimators": 150, "max_depth": 12,
                    "min_samples_leaf": 3}},
    ],
    # per-query z-score weighted sum; slightly favor the top single-model
    # ranker (xgb rank:ndcg) while keeping lgbm lambdarank and ert for
    # diversity, targeting a small generalizable gain over 0.45/0.4/0.15.
    "ensemble": {"type": "zscore_weighted", "weights": [0.5, 0.35, 0.15]},
}
# EVOLVE-BLOCK-END: pipeline
