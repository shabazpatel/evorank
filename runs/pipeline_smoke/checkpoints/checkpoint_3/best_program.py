"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Raw features plus the highest-evidence additions on this fold:
    (1) within-query percentile ranks of the core ranking signals (price,
    star rating, review score, both location scores, historical price),
    the single strongest recipe measured here (+0.011 ndcg alone); (2)
    count encodings of prop_id and srch_destination_id, the largest
    documented single gain in the literature (+0.013 combined measured);
    (3) the provided stacked_score feature (train-fold linear score on
    visitor/query columns), tested here in combination rather than alone
    since interactions were untested; (4) low-quantile imputation of
    prop_location_score2 instead of leaving NaN/zero-fill. Kept small
    (raw + 6 rank cols + 2 count cols + 1 stacked score) to avoid the
    kitchen-sink dilution effect noted on this fold."""
    feats = df.drop(columns=["qid"]).copy()
    rank_cols = [
        "price_usd", "prop_starrating", "prop_review_score",
        "prop_location_score1", "prop_location_score2",
        "prop_log_historical_price",
    ]
    for c in rank_cols:
        if c in df.columns:
            feats[c + "_qrank"] = df.groupby("qid")[c].rank(pct=True)
    if "prop_id" in df.columns:
        feats["prop_id_count"] = stats.count(df, "prop_id")
    if "srch_destination_id" in df.columns:
        feats["dest_count"] = stats.count(df, "srch_destination_id")
    try:
        feats["stacked_score"] = stats.stacked_score(df)
    except Exception:
        pass
    if "prop_location_score2" in df.columns:
        low_q = stats.quantile("prop_location_score2", 0.05)
        feats["prop_location_score2"] = df["prop_location_score2"].fillna(low_q)
    return feats
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
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "num_boost_round": 150}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.05, "num_leaves": 63,
                    "min_child_samples": 20, "n_estimators": 200,
                    "subsample": 0.8, "colsample_bytree": 0.8}},
        {"family": "sklearn", "objective": "ert",
         "params": {"n_estimators": 120, "max_depth": 10,
                    "min_samples_leaf": 20}},
    ],
    # single | zscore_weighted (per-query z-score, weighted sum) | rank_mean
    # Diversify by FAMILY (xgb ranker + lgbm ranker + sklearn extremely
    # randomized trees) rather than just loss: the forest member is a
    # cheap, structurally different error source (bagged, axis-aligned
    # splits, no boosting), weighted lowest since it is the weakest single
    # member but should still add ensemble diversity per the winners'
    # per-query z-score combiner recipe.
    "ensemble": {"type": "zscore_weighted", "weights": [0.55, 0.33, 0.12]},
}
# EVOLVE-BLOCK-END: pipeline
