# Pipeline priors for e-commerce hotel ranking (distilled from the ICDM 2013
# challenge literature, notably Liu et al., arXiv 1311.7679)

Feature construction, where most of the lift lives:
- Count encodings of high-cardinality ids (prop_id, srch_destination_id)
  were the single largest documented gain in the challenge (about +2.4 points
  NDCG on a GBM). Use stats.count.
- Rate encodings (CTR per prop_id, CVR per prop_id or destination) carry
  behavioral history; they are smoothed and train-fold-only via stats.rate.
- Composite price anchors work: ump = exp(prop_log_historical_price) -
  price_usd, price_diff = visitor_hist_adr_usd - price_usd, starrating_diff,
  per_fee = price * rooms / guests, total_fee, score2ma =
  location_score2 * query_affinity, score1d2 = ratio of location scores.
- Within-query rank transforms (rank of price_usd, prop_starrating,
  prop_location_score2 inside the result list) bring listwise information to
  any model. These were called the most important listwise features.
- A stacked linear score on visitor and query columns (stats.stacked_score)
  reproduces the mechanism of the challenge's top-importance feature.
- Missing values: location score 2 imputed with a low train quantile
  (stats.quantile) beats zero-fill.

Model and loss selection:
- LambdaMART-family rankers (xgb rank:ndcg, lgbm lambdarank) were the
  strongest single models. Pointwise GBM with count features was competitive.
- Binary objectives on booking give diverse errors that ensemble well even
  when weaker alone. Linear and forest models add diversity cheaply.
- Deep models underperformed in the challenge; they are not in the whitelist.

Ensembling:
- Per-query z-score normalization then a weighted linear combination was the
  challenge winners' final combiner; gains come from DIVERSE members
  (different family or objective), not from near-duplicates.
- Two or three members are enough; weight the stronger member higher.

Failure modes:
- Leakage through target encodings is guarded (train-fold stats, smoothing,
  prior fallback), but a feature that looks too good on val and adds nothing
  on unseen data is suspect; prefer features with a causal story.
- Feature bloat: more than ~150 columns slows training and dilutes splits.
- Ensembles of near-identical members waste budget for no diversity gain.

Measured on this fold (July 7 headroom gate), strongest first:
- Within-query rank features: +0.011 ndcg alone, the top recipe here.
- Plus count encodings: +0.013 combined, the best hand-built variant.
- Rate encodings are neutral at best on this fold even done correctly
  (K-fold out-of-fold by query); per-prop rates are rejected as too sparse.
- Composite price anchors and the stacked score: neutral alone, may interact.
- More features is not better: the 16-feature kitchen sink underperformed
  ranks plus counts. Prefer few, strong, diverse features.
- Ensembles only pay with strong diverse members; a weak member drags the
  combination below the best single model.
