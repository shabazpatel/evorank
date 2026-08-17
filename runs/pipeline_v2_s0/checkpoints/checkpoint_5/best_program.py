"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Lean recipe: within-query rank transforms (top documented lift) +
    high-cardinality count encodings (2nd highest lift) + a few raw/derived
    anchors (price, stars, review, location score with low-quantile impute,
    promotion/brand flags) + composite price anchors (ump, price_diff) +
    stacked_score. Kept small (~14 cols) since kitchen-sink sets underperform
    this focused combination on this fold."""
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

    out["stacked_score"] = stats.stacked_score(df)

    return pd.DataFrame(out, index=df.index)
# EVOLVE-BLOCK-END: features

# EVOLVE-BLOCK-START: pipeline
PIPELINE = {
    # 1 to 3 members; family x objective x params covers model, loss, and
    # hyperparameter selection. Families: xgb (rank:ndcg, rank:pairwise,
    # reg:squarederror, binary:logistic, custom:seed, custom:simplified),
    # lgbm (lambdarank, regression, binary), sklearn (logistic, rf, ert,
    # ridge). See the whitelist in eval/evaluator_pipeline.py.
    "models": [
        {"family": "xgb", "objective": "rank:ndcg",
         "params": {"eta": 0.1, "max_depth": 6, "min_child_weight": 0.1,
                    "num_boost_round": 120}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.08, "num_leaves": 63,
                    "min_child_samples": 20, "n_estimators": 200,
                    "subsample": 0.8, "colsample_bytree": 0.8}},
        # extra-trees member for cheap structural diversity (different
        # family than the two boosted rankers, no shared training dynamics).
        {"family": "sklearn", "objective": "ert",
         "params": {"n_estimators": 120, "max_depth": 10,
                    "min_samples_leaf": 5}},
    ],
    # single | zscore_weighted (per-query z-score, weighted sum) | rank_mean
    "ensemble": {"type": "zscore_weighted", "weights": [0.45, 0.4, 0.15]},
}
# EVOLVE-BLOCK-END: pipeline
