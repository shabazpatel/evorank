"""Seed pipeline for EvoRank-Pipeline (Part 2).

Two evolvable blocks: feature construction (code) and pipeline spec
(declarative, whitelisted). The seed is the raw-feature passthrough trained
with a single default LambdaMART, which reproduces the Part 1 baseline. See
paper/pipeline_design.md for the search-space contract and the stats API.
"""

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    """Lean, validated feature set: within-query rank transforms (top single
    lever on this fold, +0.011 ndcg) plus count encodings of high-cardinality
    ids (+0.013 combined, best hand-built variant), a handful of composite
    price/quality anchors, and the stacked score. Kept well under 150
    columns to avoid the kitchen-sink regression observed on this fold.
    prop_location_score2 missing values are imputed with a low train-fold
    quantile rather than zero."""
    import numpy as np
    import pandas as pd

    out = {}

    raw_cols = [
        "price_usd", "prop_starrating", "prop_review_score",
        "prop_location_score1", "prop_location_score2",
        "prop_log_historical_price", "prop_brand_bool", "promotion_flag",
        "srch_length_of_stay", "srch_booking_window", "srch_adults_count",
        "srch_children_count", "srch_room_count", "srch_saturday_night_bool",
        "visitor_hist_starrating", "visitor_hist_adr_usd",
        "orig_destination_distance", "srch_query_affinity_score",
    ]
    for c in raw_cols:
        if c in df.columns:
            out[c] = df[c]

    if "price_usd" in df.columns:
        out["log_price"] = np.log1p(df["price_usd"].clip(lower=0))

    if "prop_location_score2" in df.columns:
        q = stats.quantile("prop_location_score2", 0.05)
        out["prop_location_score2"] = df["prop_location_score2"].fillna(q)

    rank_cols = ["price_usd", "prop_starrating", "prop_location_score2",
                 "prop_review_score", "prop_location_score1"]
    for c in rank_cols:
        if c in df.columns:
            out[f"{c}_qrank"] = df.groupby("qid")[c].rank(pct=True)

    for c in ["prop_id", "srch_destination_id"]:
        if c in df.columns:
            out[f"{c}_count"] = stats.count(df, c)

    price = df["price_usd"] if "price_usd" in df.columns else None
    if price is not None:
        if "prop_log_historical_price" in df.columns:
            out["ump"] = np.exp(df["prop_log_historical_price"]) - price
        if "visitor_hist_adr_usd" in df.columns:
            out["price_diff"] = df["visitor_hist_adr_usd"] - price
        rooms = df["srch_room_count"] if "srch_room_count" in df.columns else 1
        guests = (df["srch_adults_count"].fillna(0) +
                  df["srch_children_count"].fillna(0)) if \
            ("srch_adults_count" in df.columns and "srch_children_count" in df.columns) else 1
        guests = guests.replace(0, 1) if hasattr(guests, "replace") else guests
        out["per_fee"] = price * rooms / guests
        out["total_fee"] = price * (df["srch_length_of_stay"]
                                     if "srch_length_of_stay" in df.columns else 1)

    if ("visitor_hist_starrating" in df.columns and
            "prop_starrating" in df.columns):
        out["starrating_diff"] = (df["visitor_hist_starrating"] -
                                   df["prop_starrating"])

    if ("prop_location_score2" in df.columns and
            "srch_query_affinity_score" in df.columns):
        out["score2ma"] = (out["prop_location_score2"] *
                            df["srch_query_affinity_score"].fillna(0))

    if ("prop_location_score1" in df.columns and
            "prop_location_score2" in df.columns):
        denom = df["prop_location_score1"].replace(0, np.nan)
        out["score1d2"] = df["prop_location_score2"] / denom

    out["stacked_score"] = stats.stacked_score(df)

    comp_rate_cols = [c for c in df.columns if c.startswith("comp") and c.endswith("_rate")
                      and not c.endswith("_rate_percent_diff")]
    if comp_rate_cols:
        out["comp_rate_mean"] = df[comp_rate_cols].mean(axis=1, skipna=True)

    result = pd.DataFrame(out, index=df.index)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
# EVOLVE-BLOCK-END: features

# EVOLVE-BLOCK-START: pipeline
PIPELINE = {
    # 1 to 3 members; family x objective x params covers model, loss, and
    # hyperparameter selection. Families: xgb (rank:ndcg, rank:pairwise,
    # reg:squarederror, binary:logistic, custom:seed, custom:simplified),
    # lgbm (lambdarank, regression, binary), sklearn (logistic, rf, ert,
    # ridge). See the whitelist in eval/evaluator_pipeline.py.
    # Novel diversity choice vs. prior candidates: swap the third member for
    # a sklearn ert (extremely randomized trees) on the same features -- a
    # genuinely different family (bagged, non-boosted, high-variance trees)
    # rather than another xgb objective variant, per the prior that forest
    # models add cheap diversity to a LambdaMART-family core.
    "models": [
        {"family": "xgb", "objective": "rank:ndcg",
         "params": {"eta": 0.1, "max_depth": 6, "min_child_weight": 0.1,
                    "num_boost_round": 150}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.08, "num_leaves": 63,
                    "min_child_samples": 20, "n_estimators": 200,
                    "subsample": 0.8, "colsample_bytree": 0.8}},
        {"family": "sklearn", "objective": "ert",
         "params": {"n_estimators": 120, "max_depth": 8,
                    "min_samples_leaf": 20}},
    ],
    # single | zscore_weighted (per-query z-score, weighted sum) | rank_mean
    "ensemble": {"type": "zscore_weighted", "weights": [0.5, 0.35, 0.15]},
}
# EVOLVE-BLOCK-END: pipeline
