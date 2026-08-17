"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Rank+count recipe: ranks + id counts + stacked score + price_diff."""
    out = df[[
        "price_usd", "prop_starrating", "prop_review_score",
        "prop_location_score1", "prop_log_historical_price",
        "promotion_flag", "prop_brand_bool",
    ]].copy()

    loc2 = df["prop_location_score2"]
    out["prop_location_score2"] = loc2.fillna(stats.quantile("prop_location_score2", 0.05))

    out["rank_price"] = df.groupby("qid")["price_usd"].rank()
    out["rank_star"] = df.groupby("qid")["prop_starrating"].rank()
    out["rank_review"] = df.groupby("qid")["prop_review_score"].rank()
    out["rank_loc2"] = out.groupby(df["qid"])["prop_location_score2"].rank()

    out["prop_id_count"] = stats.count(df, "prop_id")
    out["dest_id_count"] = stats.count(df, "srch_destination_id")
    out["stacked"] = stats.stacked_score(df)
    out["price_diff"] = df["visitor_hist_adr_usd"] - df["price_usd"]

    return out
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
         "params": {"eta": 0.08, "max_depth": 7, "min_child_weight": 0.1,
                    "subsample": 0.85, "colsample_bytree": 0.85,
                    "num_boost_round": 180}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.08, "num_leaves": 31,
                    "min_child_samples": 20, "n_estimators": 200,
                    "subsample": 0.8, "colsample_bytree": 0.8}},
        {"family": "lgbm", "objective": "binary",
         "params": {"learning_rate": 0.08, "num_leaves": 31,
                    "min_child_samples": 30, "n_estimators": 200,
                    "subsample": 0.8, "colsample_bytree": 0.8}},
    ],
    # single | zscore_weighted (per-query z-score, weighted sum) | rank_mean
    "ensemble": {"type": "zscore_weighted", "weights": [0.5, 0.32, 0.18]},
}
# EVOLVE-BLOCK-END: pipeline
