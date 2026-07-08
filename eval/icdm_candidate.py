"""Hand-written ICDM-style candidate for the headroom gate (Gate 0).

Implements the highest-attribution recipes from Liu et al. 2013 (arXiv
1311.7679) inside the EvoRank-Pipeline contract: composite price anchors,
within-query rank features, count and CTR/CVR encodings, the stacked linear
score, and a two-member z-score ensemble. If this program does not lift ndcg
well above the noise floor, the search space lacks the assumed headroom.
"""
import numpy as np

# EVOLVE-BLOCK-START: features
def build_features(df, stats):
    out = df.drop(columns=["qid"]).copy()
    # composite price anchors (ump, price_diff, per_fee, total_fee)
    out["ump"] = np.exp(df["prop_log_historical_price"]) - df["price_usd"]
    out["price_diff"] = df["visitor_hist_adr_usd"] - df["price_usd"]
    out["starrating_diff"] = df["visitor_hist_starrating"] - df["prop_starrating"]
    out["per_fee"] = (df["price_usd"] * df["srch_room_count"]
                      / (df["srch_adults_count"] + df["srch_children_count"] + 1e-6))
    out["total_fee"] = df["price_usd"] * df["srch_room_count"]
    out["score2ma"] = df["prop_location_score2"] * df["srch_query_affinity_score"]
    out["score1d2"] = (df["prop_location_score2"] + 1e-4) / (df["prop_location_score1"] + 1e-4)
    # within-query rank features (listwise bridge)
    for col in ("price_usd", "prop_starrating", "prop_location_score2"):
        out[f"{col}_rank"] = df.groupby("qid")[col].rank(method="average")
    # count encodings (safe at any cardinality) and rate encodings on
    # low-cardinality columns only; per-prop rates are data-starved on the
    # fast fold and the evaluator rejects them
    for col in ("prop_id", "srch_destination_id", "prop_country_id"):
        out[f"{col}_cnt"] = stats.count(df, col)
    out["country_cvr"] = stats.rate(df, "prop_country_id", "booking")
    out["site_ctr"] = stats.rate(df, "site_id", "click")
    # stacked linear score on visitor/query columns (the fm_score mechanism)
    out["stacked_score"] = stats.stacked_score(df)
    return out
# EVOLVE-BLOCK-END: features

# EVOLVE-BLOCK-START: pipeline
PIPELINE = {
    "models": [
        {"family": "xgb", "objective": "rank:ndcg",
         "params": {"eta": 0.1, "max_depth": 6, "min_child_weight": 0.1,
                    "num_boost_round": 150}},
        {"family": "lgbm", "objective": "lambdarank",
         "params": {"learning_rate": 0.08, "num_leaves": 63, "n_estimators": 200}},
    ],
    "ensemble": {"type": "zscore_weighted", "weights": [0.55, 0.45]},
}
# EVOLVE-BLOCK-END: pipeline
