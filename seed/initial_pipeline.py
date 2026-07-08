"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Construct the feature matrix. df has one row per impression with
    SEMANTIC columns (price_usd, prop_starrating, prop_location_score2,
    srch_length_of_stay, visitor_hist_adr_usd, comp1_rate, ...) plus qid.
    Label columns are not present. stats is the leakage-safe train-fold
    helper:
      stats.count(df, col)             frequency encoding of a column's values
      stats.rate(df, col, "booking")   smoothed conversion-rate encoding
      stats.rate(df, col, "click")     smoothed click-rate encoding
      stats.quantile(col, q)           train-fold quantile scalar
      stats.stacked_score(df)          score feature from a linear model on
                                       visitor and query columns (train-fold)
    Within-query transforms are allowed, e.g. df.groupby("qid")[c].rank().
    Return a numeric DataFrame aligned to df. The seed passes raw features
    through unchanged."""
    return df.drop(columns=["qid"])
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
    ],
    # single | zscore_weighted (per-query z-score, weighted sum) | rank_mean
    "ensemble": {"type": "single"},
}
# EVOLVE-BLOCK-END: pipeline
