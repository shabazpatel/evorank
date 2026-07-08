# EvoRank-Pipeline: search space and feedback design (Part 2)

The evolvable artifact is a full ranking pipeline: feature construction, model
selection, loss selection, and ensemble architecture. The evaluator owns
everything unsafe (data, training, leakage guards, timeouts) and returns
stage-attributed feedback so the search can tell WHICH axis earned each gain.

## 1. The evolvable program (two blocks)

Block A, feature construction (code):

    # EVOLVE-BLOCK-START: features
    def build_features(df, stats):
        """df: one row per impression with SEMANTIC columns (price_usd,
        prop_starrating, prop_location_score2, srch_length_of_stay, comp*_*,
        ...) plus qid. Labels are NOT present. stats is the train-fold
        aggregate helper (leakage-safe):
          stats.count(df, col)                value frequency encoding
          stats.rate(df, col, "booking")      smoothed CVR encoding
          stats.rate(df, col, "click")        smoothed CTR encoding
          stats.quantile(col, q)              train-fold quantile scalar
        Within-query transforms are allowed and encouraged, for example
        df.groupby("qid")[col].rank(). Return a numeric DataFrame aligned to
        df (one row per impression)."""
        return df[NUMERIC_RAW_COLUMNS]
    # EVOLVE-BLOCK-END: features

Block B, pipeline spec (declarative, whitelisted):

    # EVOLVE-BLOCK-START: pipeline
    PIPELINE = {
        "models": [
            # 1 to 3 members. family x objective x params = model, loss,
            # hyperparameter selection in one place.
            {"family": "xgb",  "objective": "rank:ndcg",  "params": {"eta": 0.1, "max_depth": 6, "num_boost_round": 120}},
        ],
        # single | zscore_weighted (per-query z-score, weighted sum) | rank_mean
        "ensemble": {"type": "single"},
    }
    # EVOLVE-BLOCK-END: pipeline

Whitelists enforced by the evaluator, never trusted from the program:

| family | objectives (loss selection) | clamped params |
|---|---|---|
| xgb | rank:ndcg, rank:pairwise, reg:squarederror, binary:logistic (on booking), custom:seed, custom:simplified | eta, max_depth <= 12, min_child_weight, subsample, colsample_bytree, gamma, lambda, num_boost_round <= 300 |
| lgbm | lambdarank, regression, binary (on booking) | learning_rate, num_leaves <= 255, min_child_samples, n_estimators <= 300, subsample, colsample_bytree |
| ridge | pointwise squared error on graded rel | alpha |

custom:seed and custom:simplified are the Part 1 objectives, connecting the
two campaigns: the search can select a discovered loss as a component.

## 2. Evaluation pipeline and per-stage feedback

Stage by stage, what runs and what feedback it produces. Every stage can
reject; a rejection short-circuits to zero fitness with the stage name and
reason in the feedback (the loss-rejection analog from Part 1).

Stage 0, program load and spec validation.
  Guards: blocks parse, build_features exists, PIPELINE well-formed, families
  and objectives in whitelist, <= 3 members, params clamped.
  Feedback on failure: "rejected at spec: <reason>" (for example unknown
  family "catboost", or 5 members requested).

Stage 1, feature construction (train fold, then val fold with train stats).
  Guards: no label columns readable (df handed to the program excludes
  rel/booking/rev), output aligned and numeric, column count <= 300, no
  NaN/inf after fill, wall budget.
  Feedback: feature count, count of new columns vs raw passthrough, and a
  compactness signal: "features: 84 columns (71 raw + 13 engineered), build
  4.2s". On failure: "rejected at features: <exception>".

Stage 2, per-member training (each model in PIPELINE.models).
  Guards: per-member wall budget, non-finite predictions rejected.
  Feedback per member, the model and loss selection evidence:
  "member 0 xgb/rank:ndcg: val ndcg .4262 (34s)"
  "member 1 lgbm/lambdarank: val ndcg .4249 (11s)"
  A member that fails does not kill the candidate if at least one member
  survives; the failure is reported: "member 2 xgb/custom:seed: TIMEOUT".

Stage 3, ensemble combine.
  Guards: weights length matches members, weights finite and nonnegative.
  Feedback, the ensemble-architecture evidence: ensemble val ndcg vs the best
  single member, i.e. the marginal value of the architecture itself:
  "ensemble zscore_weighted [.6, .4]: ndcg .4281 = best member +.0019".

Stage 4, metric + noise gate (identical to Part 1 machinery).
  Three-objective bundle on the val fold plus paired query bootstrap vs the
  cached seed-pipeline reference:
  "scores: ndcg .4281 (+.0069 vs seed, noise +/-.0040, SIGNIFICANT) |
   book_ndcg ... | revenue ..."

Stage 5, behavioral segments (from Part 1 diagnostics, model-agnostic parts).
  Booked vs click-only decomposition, booked-item rank distribution, top-10
  churn vs seed ranking. The gradient probes from Part 1 do not apply to
  arbitrary members and are omitted in Part 2.

Assembled artifacts.feedback example (what the LLM sees for one candidate):

    scores: ndcg=0.4281 (+0.0069 vs seed, noise +/-0.0040, SIGNIFICANT) | book_ndcg=0.4467 (+0.0017, within noise) | revenue=0.4131 (+0.0033, within noise)
    features: 84 columns (71 raw + 13 engineered: price_rank_in_query, ump, per_fee, prop_id_count, prop_id_cvr, ...), build 4.2s
    members: [0] xgb/rank:ndcg .4262 (34s)  [1] lgbm/lambdarank .4249 (11s)
    ensemble: zscore_weighted [0.6, 0.4] = .4281, best single +0.0019
    segments: booked-q ndcg .512 (+.011) | click-only .389 (+.004) | booked-item median rank 4, top-1 15%
    churn: top-10 overlap with seed ranking 71%

Each line maps to one decision axis: line 2 attributes to features, line 3 to
model and loss choice, line 4 to ensemble architecture, lines 1 and 5 to
overall fitness and behavior. That is the "understand the optimal solution"
requirement made concrete: the search never has to guess which axis moved.

## 3. System-level feedback (across iterations)

- In-context evolution (SkyDiscover native): each mutation prompt contains
  the parent and 4 archive programs WITH their full feedback blocks, so the
  LLM sees contrastive evidence across pipelines (for example, all high
  scorers so far carry prop_id count encodings; ensembles only pay when
  members disagree).
- Pareto archive over the same three objectives; hypervolume and transfer
  audit reuse Part 1 tooling unchanged.
- Memory arms (the ablation, if run in Part 2): system message seeded with
  knowledge/feature_priors.md, the distilled ICDM 2013 lessons (composite
  price anchors, within-query ranks, count and CTR/CVR encodings, z-score
  per-query ensembling, what did not work: deep models).
- Selection honesty: final selections re-scored on the 60k-query test fold
  by analyze/generalization_audit.py before any claim is made.

## 4. Leakage guards (the campaign-critical risk)

Count/CTR/CVR encodings are the classic leak vector. Defenses, in order:
1. The program never receives label columns; only the evaluator-owned
   TrainStats object touches labels, and it is built from the TRAIN fold only.
2. rate() encodings are smoothed toward the train prior
   (sum + alpha * prior) / (count + alpha), alpha = 20, so rare values
   cannot memorize their own outcome.
3. Unseen values at apply time fall back to the train prior (no test-time
   information enters).
4. Backstop: the transfer audit; a leaked feature shows up as a val-test gap
   and is caught before any claim.

## 5. Budget and gates

- Per-candidate cost: features (seconds) + sum of member trainings (xgb ~35s,
  lgbm ~10s, ridge ~2s at 200k rows), evaluator wall cap 150s.
- Gate 0 (before any LLM call): seed pipeline must reproduce the raw-feature
  baseline exactly, and a hand-written ICDM-style candidate must lift ndcg by
  at least +0.010 on the 6k val fold (2.5x the noise floor). If it does not,
  the search space does not have the headroom this campaign assumes, no-go.
- Campaign: smoke 3 iters, then 2 to 3 seeds x 40 iterations overnight,
  scores-only feedback (the guidance ablation was Part 1's question; Part 2
  optimizes for lift), followed by the transfer audit.
