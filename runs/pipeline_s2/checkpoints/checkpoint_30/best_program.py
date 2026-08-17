"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Lean feature recipe per documented headroom gate: within-query rank
    transforms (top single recipe, +0.011 ndcg) plus count encodings of
    high-cardinality ids (+0.013 combined, best hand-built variant), plus a
    few causal composite anchors and the stacked score. Kept to ~20 columns
    to avoid the kitchen-sink regression observed with 16+ engineered cols
    plus full raw passthrough. Location score 2 is imputed with a low
    train-fold quantile instead of zero-fill."""
    import numpy as np
    import pandas as pd
    out = {}

    loc2_fill = stats.quantile("prop_location_score2", 0.05)
    loc2 = df["prop_location_score2"].fillna(loc2_fill)
    out["prop_location_score2_imp"] = loc2

    # within-query rank transforms (listwise signal, strongest documented lift)
    out["price_rank"] = df.groupby("qid")["price_usd"].rank()
    out["starrating_rank"] = df.groupby("qid")["prop_starrating"].rank()
    out["loc_score2_rank"] = loc2.groupby(df["qid"]).rank()
    out["review_score_rank"] = df.groupby("qid")["prop_review_score"].rank()
    out["loc_score1_rank"] = df.groupby("qid")["prop_location_score1"].rank()

    # count encodings of high-cardinality ids (second strongest documented lift)
    out["prop_id_count"] = stats.count(df, "prop_id")
    out["dest_id_count"] = stats.count(df, "srch_destination_id")

    # a few causal composite anchors
    out["ump"] = np.exp(df["prop_log_historical_price"]) - df["price_usd"]
    out["price_diff"] = df["visitor_hist_adr_usd"] - df["price_usd"]
    out["starrating_diff"] = df["visitor_hist_starrating"] - df["prop_starrating"]

    price_mean = df.groupby("qid")["price_usd"].transform("mean")
    price_std = df.groupby("qid")["price_usd"].transform("std").replace(0, 1)
    out["price_z"] = (df["price_usd"] - price_mean) / price_std

    out["stacked_score"] = stats.stacked_score(df)

    for c in ["price_usd", "prop_starrating", "prop_review_score",
              "prop_location_score1", "promotion_flag", "prop_brand_bool",
              "srch_length_of_stay", "srch_booking_window",
              "srch_saturday_night_bool"]:
        out[c] = df[c]

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
         "params": {"eta": 0.08, "max_depth": 6, "min_child_weight": 0.5,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "num_boost_round": 200}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.06, "num_leaves": 63,
                     "min_child_samples": 20, "subsample": 0.8,
                     "colsample_bytree": 0.8, "n_estimators": 200}},
        {"family": "sklearn", "objective": "ert",
         "params": {"n_estimators": 120, "max_depth": 8,
                    "min_samples_leaf": 15}},
    ],
    # zscore_weighted per-query combiner; strongest rankers weighted higher,
    # the diverse forest member (different family entirely, not just a
    # different loss on the same GBM engine) gets a lighter weight for
    # error diversity per the winning-solution combiner pattern.
    "ensemble": {"type": "zscore_weighted", "weights": [0.45, 0.35, 0.2]},
}
# EVOLVE-BLOCK-END: pipeline
