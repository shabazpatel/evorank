# Algorithm-improvement validation arms (July 8-9)

Two arms tested whether search-algorithm changes improve pipeline discovery,
after studying the AdaEvolve and EvoX papers and the SkyDiscover source.
Same evaluator, seed program, feedback, and data as the original campaign.

- pipeline_v2_s0: AdaEvolve with thresholds recalibrated for NDCG-scale
  gains (configs/evorank_pipeline_v2.yaml), 100 iterations.
- pipeline_evox_s0: EvoX meta-evolution (strategy database rewriting its own
  search policy), 100 iterations (configs/evorank_pipeline_evox.yaml).

## Results (val = 6k-query selection fold; test = 60k-query held-out fold)

| arm | val @50 | val @100 | TEST ndcg | test gain vs seed |
|---|---|---|---|---|
| original s0 (adaevolve, 50 it) | 0.4356 | - | 0.4316 | +0.0094 |
| original s1 (adaevolve, 50 it) | 0.4359 | - | 0.4322 | +0.0100 |
| original s2 (adaevolve, 50 it) | 0.4398 | - | 0.4352 | +0.0131 |
| pipeline_v2_s0 (calibrated) | 0.4337 | 0.4337 | 0.4299 | +0.0077 |
| pipeline_evox_s0 (meta-evo) | 0.4376 | 0.4380 | 0.4316 | +0.0094 |

Verdict: neither arm beat the original campaign, budget-matched or with
double budget. The default-config s2 run remains the best pipeline on both
folds. EvoX matched the original seeds midpack; the calibrated arm was the
weakest evolved arm and gained nothing from iterations 50 to 100.

## Adaptive-machinery telemetry (pipeline_v2_s0 jsonl, 102 records)

- Sampling intensity moved only 0.364 to 0.372 inside the configured
  [0.25, 0.4] band: effectively constant. Cause: the accumulated signal G
  sits at 0.07 to 0.10 all run, dominated by the first-improvement spike
  (the first delta from -inf is capped at 1.0, contributing 0.1 to the EMA);
  real 1-2 percent relative gains cannot move G afterward.
- Island spawning: would_spawn never true (G-derived productivity 0.07+
  never dropped below the 1e-4 threshold); islands stayed at 3. The
  pre-calibration failure mode was spawn-on-every-cooldown; the calibrated
  threshold, chosen from a theoretical G estimate of ~1e-4, produced the
  opposite failure mode because actual G is ~0.08 (initialization artifact).
- Paradigm breakthroughs: improvement_rate never dropped below the 0.2
  threshold (minimum exactly 0.20, trigger requires strictly below); zero
  paradigm events. EvoX by contrast performed 14 strategy rewrites, all
  logged, with no outcome advantage.

## Lessons (for the repo and future work; Discussion sentence in the paper
was intentionally NOT added, per the pre-registered gate: the calibrated arm
did not outperform, so no claim is earned)

1. Measure adaptive-signal distributions from telemetry before calibrating
   thresholds; theoretical estimates missed the initialization-spike artifact
   that actually dominates G.
2. On this problem, seed-to-seed variance exceeds algorithm-to-algorithm
   variance (calibrated AdaEvolve, default AdaEvolve, and EvoX all land
   inside the original seed spread on transfer). Search-space headroom and
   evaluation fidelity govern outcomes; search-algorithm sophistication
   redistributes them.
3. Longer horizons show diminishing returns in this space: +0.000 (v2) and
   +0.0004 (EvoX) from iterations 50 to 100 on the selection fold.
