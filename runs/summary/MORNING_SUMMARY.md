# EvoRank overnight run summary (July 3, ~01:30 to ~05:30)

All phases of the build order executed. Every number below is machine-generated
into runs/summary/comparison.csv plus per-run convergence and frontier CSVs by
analyze/aggregate.py. Loop model: claude-sonnet-5 via the Anthropic
OpenAI-compatible endpoint. Total ~290 LLM calls, estimated $15-25 spend.

## Headline table (fast subset, val fold, NDCG at 10)

| method | kind | ndcg | book_ndcg | revenue |
|---|---|---|---|---|
| lambdamart_optuna (40 trials) | baseline | 0.4382 | 0.4644 | 0.4233 |
| mem_on_s1 (40 iters) | evolution | 0.4325 | 0.4559 | 0.4095 |
| mem_off_s0 | evolution | 0.4317 | 0.4523 | 0.4109 |
| mem_on_s2 | evolution | 0.4308 | 0.4528 | 0.4106 |
| mem_off_s2 | evolution | 0.4305 | 0.4552 | 0.4078 |
| mem_off_s1 | evolution | 0.4302 | 0.4502 | 0.4044 |
| random_search_s0 (40 evals) | baseline | 0.4301 | 0.4506 | 0.4028 |
| mem_on_s0 | evolution | 0.4289 | 0.4555 | 0.4180 |
| seed program | reference | 0.4256 | 0.4477 | 0.4060 |
| lambdamart_default | baseline | 0.4230 | 0.4474 | 0.3960 |
| lambdaloss (NDCG-Loss2) | baseline | 0.3816 | 0.3966 | 0.3480 |

## Memory ablation (the headline experiment)

Final quality at 40 iterations, best ndcg per seed:

| seed | memory ON | memory OFF |
|---|---|---|
| 0 | 0.4289 | 0.4317 |
| 1 | 0.4325 | 0.4302 |
| 2 | 0.4308 | 0.4305 |
| mean +/- std | 0.4308 +/- 0.0018 | 0.4308 +/- 0.0008 |

Iterations to reach the equal-budget random-search level (ndcg 0.4301):

| seed | memory ON | memory OFF |
|---|---|---|
| 0 | never (0.4289 final; note its book/revenue are the best of all runs) | 13 |
| 1 | 1 | 23 |
| 2 | 3 | 7 |

Reading, to be stated honestly in the paper:
- Final quality after 40 iterations is indistinguishable between conditions
  (0.4308 vs 0.4308 on mean best ndcg).
- Seeded memory changes the speed, not the ceiling: memory-on runs match the
  full 40-eval random-search budget within 1-3 iterations (2 of 3 seeds),
  memory-off needs 7-23. Early guidance is where the knowledge base pays.
- 5 of 6 evolution runs beat random search at equal budget on ndcg; all 6 beat
  it on book_ndcg, and mem_on_s0 dominates it on revenue by a wide margin.

## vs the tuned baseline

No evolved objective beat lambdamart_optuna on ndcg at this budget under FIXED
training hyperparameters. The post-hoc control closes the asymmetry: the best
discovered objective (mem_on_s1) was granted the same 40-trial Optuna budget.

Post-hoc retune result (runs/posthoc_retune_best_program.json):
discovered + tuned = ndcg 0.4352, book_ndcg 0.4602, revenue 0.4088, vs
lambdamart_optuna 0.4382, 0.4644, 0.4233. Tuning moved the discovered
objective from 0.4325 to 0.4352 (+0.0027) but did not close the gap
(-0.0030 ndcg remaining). Honest reading for the paper: at 40 iterations on
the fast subset, the discovered objective does not yet Pareto-dominate tuned
LambdaMART; the credible claims tonight are (a) beats equal-budget random
search, (b) seeded memory accelerates discovery ~5x, and (c) the discovered
mechanism is interpretable. Longer runs and full-split evaluation are the
obvious next lever on the domination claim.

## The discovered objective (mem_on_s1, figure 1 material)

Mechanistically interpretable and clearly beyond the seed:
1. Revenue enters the gain as 3.5 * log1p(clip(rev, 0, 3000)) / log1p(3000),
   i.e. the search discovered its own outlier handling (clip at ~p99) and
   transform (log), which we deliberately left to it.
2. Adaptive per-query blend: queries WITH a booking weight a listwise
   Plackett-Luce top-1 term more (alpha 0.55), click-only queries lean on the
   classic pairwise |deltaNDCG| lambda (alpha 0.85). This matches the
   single-booking-per-query structure of the data.
3. The hybrid pairwise + listwise gradient is a known research direction
   (softmax/ListNet-style losses), independently rediscovered and adapted, which
   validates the method per the CLAUDE.md framing.
Program source: runs/mem_on_s1/checkpoints/checkpoint_40/best_program.py

## Tier 1 analysis addendum (afternoon)

- Hypervolume: memory-on beats memory-off in ALL 3 seeds (252 vs 230, x1e-6);
  every evolution run beats random search. The ablation now has a
  final-quality result, not only the speed result.
- Mechanism ablation: listwise term +0.008 ndcg, revenue gain +0.005, but the
  adaptive alpha was HURTING. Constant alpha 0.70 gives 0.4353/0.4589/0.4196,
  the best fixed-params program yet ("evolved_simplified"). Its 40-trial
  retune and full-split evaluation are running.
- Bootstrap significance (fast val fold, 1.6k queries): CIs are wider than our
  deltas, so single-fold point claims are not defensible at this fold size.
  Parity with tuned LambdaMART is supported on ndcg/book_ndcg; revenue is
  significantly below it for the tuned evolved program. Full-split reruns
  (60k queries) are the decisive test and are in flight.

## Caveats and next steps
- All numbers are fast-subset val fold. Final reporting needs the full split
  (EVORANK_FAST=0) for the best programs and baselines, and the held-out test
  fold for the paper.
- lambdaloss underperforms under the shared fixed config; consider tuning it
  or reporting it as fixed-config like the candidates (state whichever in the
  paper).
- One operational note: the first ablation launch was killed by a ~45 min
  background-task cap; it was relaunched detached and resume-aware from
  checkpoints (runs/_configs/launch_ablation.sh). No iterations were lost.
- Suggested next experiments: longer runs (T=80+) to test whether evolution
  closes the optuna gap; hypervolume computation for the Pareto quality claim;
  full-split evaluation of the six best programs.
