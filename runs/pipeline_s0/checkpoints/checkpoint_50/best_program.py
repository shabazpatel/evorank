"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Rank+count recipe (documented best hand-built variant, +0.013 ndcg)
    extended with a few causal composite anchors and a stacked score.
    Within-query percentile ranks of price/star/review/location-score
    columns give listwise signal; count encodings of prop_id and
    srch_destination_id give the largest single documented gain; ump,
    price_diff, starrating_diff, per_fee and score2ma are causal price/
    quality anchors; stats.stacked_score adds a train-fold linear
    visitor/query score. Kept to ~22 columns to avoid feature-bloat
    dilution per the fold's kitchen-sink failure mode."""
    import numpy as np
    import pandas as pd
    out = pd.DataFrame(index=df.index)

    rank_cols = ["price_usd", "prop_starrating", "prop_review_score",
                 "prop_location_score1", "prop_location_score2"]
    for col in rank_cols:
        if col in df.columns:
            out[col + "_rank"] = df.groupby("qid")[col].rank(pct=True)

    for col in ["prop_id", "srch_destination_id"]:
        if col in df.columns:
            out[col + "_count"] = stats.count(df, col)

    if "prop_log_historical_price" in df.columns and "price_usd" in df.columns:
        out["ump"] = np.exp(df["prop_log_historical_price"]) - df["price_usd"]
    if "visitor_hist_adr_usd" in df.columns and "price_usd" in df.columns:
        out["price_diff"] = df["visitor_hist_adr_usd"] - df["price_usd"]
    if "visitor_hist_starrating" in df.columns and "prop_starrating" in df.columns:
        out["starrating_diff"] = df["visitor_hist_starrating"] - df["prop_starrating"]

    if "prop_location_score2" in df.columns:
        q = stats.quantile("prop_location_score2", 0.05)
        loc2 = df["prop_location_score2"].fillna(q)
        out["prop_location_score2_imp"] = loc2
        if "srch_query_affinity_score" in df.columns:
            out["score2ma"] = loc2 * df["srch_query_affinity_score"].fillna(0)

    if "price_usd" in df.columns and "srch_room_count" in df.columns and \
       "srch_adults_count" in df.columns:
        guests = (df["srch_adults_count"].fillna(1) +
                  df.get("srch_children_count", 0)).clip(lower=1)
        out["per_fee"] = df["price_usd"] * df["srch_room_count"].fillna(1) / guests

    try:
        out["stacked_score"] = stats.stacked_score(df)
    except Exception:
        pass

    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9) if f"comp{i}_rate" in df.columns]
    if comp_rate_cols:
        out["comp_rate_mean"] = df[comp_rate_cols].mean(axis=1)
    comp_inv_cols = [f"comp{i}_inv" for i in range(1, 9) if f"comp{i}_inv" in df.columns]
    if comp_inv_cols:
        out["comp_inv_sum"] = df[comp_inv_cols].sum(axis=1)

    for col in ["price_usd", "prop_starrating", "prop_review_score",
                "promotion_flag", "prop_brand_bool", "prop_location_score1"]:
        if col in df.columns:
            out[col] = df[col]

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
         "params": {"eta": 0.1, "max_depth": 6, "min_child_weight": 0.1,
                    "num_boost_round": 150}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.08, "num_leaves": 63,
                    "min_child_samples": 20, "n_estimators": 150,
                    "subsample": 0.8, "colsample_bytree": 0.8}},
        {"family": "xgb", "objective": "binary:logistic",
         "params": {"eta": 0.1, "max_depth": 5, "min_child_weight": 1.0,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "num_boost_round": 120}},
    ],
    # single | zscore_weighted (per-query z-score, weighted sum) | rank_mean
    "ensemble": {"type": "zscore_weighted", "weights": [0.45, 0.35, 0.2]},
}
# EVOLVE-BLOCK-END: pipeline
