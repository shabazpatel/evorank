# When Does LLM-Guided Discovery Actually Discover? An Audited Two-Campaign Study on E-Commerce Learning-to-Rank

(Working draft for a 5-page short paper. Venue withheld in this file. All
numbers are machine-generated from runs/summary/, no manual transcription.)

## Abstract

LLM-guided evolutionary loops in the AlphaEvolve family are increasingly used
to discover machine learning components, and their results are typically
reported on the fold the loop itself selected on. We present an open,
fully audited case study on multi-objective e-commerce Learning-to-Rank that
asks when such loops discover anything real. In a first campaign we evolve
LambdaMART-style training objectives over relevance, conversion, and revenue
signals on the Expedia ICDM 2013 dataset. The loop appears to work, beating
default baselines on its selection fold, yet a transfer audit on 60k held-out
queries shows the gains are almost entirely fitness noise: per-edit effect
sizes (0.002 to 0.005 NDCG) are smaller than the selection-fold noise floor,
and neither seeded domain knowledge nor evidence-rich diagnostic feedback
changes what transfers, although seeded knowledge accelerates selection-fold
convergence about five-fold. In a second campaign we point the same loop at a
search space with measured headroom above the noise floor: feature
construction, model selection, loss selection, and ensemble architecture.
There, three independent seeds converge in 50 iterations (about ten dollars
each) on interpretable pipelines that Pareto-dominate an Optuna-tuned
LambdaMART on all three objectives on held-out data, an advantage that
persists, under config-equalized comparison, when training on the full
6.9M-impression dataset. Our central
finding is diagnostic rather than architectural: search-space headroom
relative to fitness noise, not guidance sophistication, separates apparent
discovery from real discovery. We release the loop, the audit stack, and a
measured catalog of failure modes, including three generations of
target-encoding leakage each caught by the audit.

## 1. Introduction

Automated discovery loops that pair a large language model with evolutionary
search over programs have produced striking results, and industrial systems
such as Meta's Ranking Engineer Agent apply the pattern to ads ranking at
scale. These systems are closed, and published results in the genre share a
methodological gap: gains are overwhelmingly reported on the data the loop
used to select candidates. For ranking systems, where realistic per-candidate
evaluations are noisy by necessity (fast folds, subsampled training), this
gap is not pedantry. Selection under noise systematically promotes lucky
candidates, and a loop can appear to make steady progress while discovering
nothing that transfers.

This paper contributes an open, end-to-end audited account of when the
pattern works and when it silently does not, on a public e-commerce ranking
task with three genuinely competing objectives (relevance, conversion, and
revenue). We make three contributions.

First, a measured failure. Evolving the training objective of a GBDT ranker,
the natural first target given the LambdaMART lineage, produces gains on the
selection fold that a paired-bootstrap transfer audit reveals to be noise:
the best evolved objective beats its seed by +0.0002 NDCG@10 on 60k held-out
queries (p = 0.76), and an equal-budget random search transfers as well or
better. We identify the mechanism quantitatively: the objective family's
effect sizes on this dataset (at most about 0.003) sit below the fitness
noise of any evaluation cheap enough to run inside a loop.

Second, guidance ablations under this microscope. Seeding the search with a
curated domain-knowledge prompt accelerates selection-fold convergence about
five-fold and wins on three-objective hypervolume in every seed, and a
gradient-introspective diagnostic feedback channel (alignment probes,
signal-usage probes, noise-gated deltas) provides the loop with strictly more
information; neither changes what generalizes. Guidance changes the speed and
style of search, not the reality of its discoveries.

Third, a measured success under identical auditing. Re-pointing the same loop
at a search space whose effect sizes clear the noise floor (feature
construction, model and loss selection, ensemble architecture) yields
transfer-audited, Pareto-dominant pipelines within 50 iterations in all three
seeds, at about ten dollars per seed. The discovered pipelines are
interpretable, converge across seeds on the same core recipe (within-query
rank features, count encodings, lean feature sets, diverse three-member
z-score ensembles), independently reconstruct the load-bearing structure of
the ICDM 2013 challenge winners' hand-built solutions, and hold their
advantage when retrained on the full dataset.

## 2. Related work

LLM-guided program evolution: AlphaEvolve-style loops, OpenEvolve,
ShinkaEvolve; industrial ranking agents REA and GEARS (closed, DNN-centric).
Automated loss discovery: GLO, AM-LFS, AutoLoss-Zero, CSE-Autoloss;
hand-derived metric-driven losses (LambdaLoss). Reflective feedback for
LLM-driven optimization: Reflexion, ProTeGi textual gradients, OPRO, GEPA.
LLM feature engineering: CAAFE and successors; classical target-encoding
leakage literature. Expedia ICDM 2013 solutions (Liu et al.) document the
feature recipes and ensembles our loop rediscovers. Our delta: none of the
above audits discovery against held-out transfer under loop-realistic noise,
and to our knowledge gradient-level introspection has not been used as
feedback for loss-function search. (Full citations in the bibliography pass.)

## 3. System and audit methodology

EvoRank wraps an evolutionary program database (Pareto archive over ndcg,
book_ndcg, revenue; island model; one LLM mutation call per iteration) around
guardrailed evaluators. Data is the public Expedia Personalized Hotel Search
training file (9.9M impressions, 399k queries), split 70/15/15 by query with
graded labels (5 booked, 1 clicked, 0 otherwise), a dollar-weighted pooled
revenue metric (per-query revenue NDCG collapses onto booking NDCG in
single-booking data), and a fast fitness fold (8k train queries, about 40s to
evaluate an objective candidate, 5 to 40s for a pipeline candidate).

The audit stack, applied identically to every campaign, is the paper's
methodological core: (a) paired query bootstrap for every comparison, with
noise-gated feedback inside the loop itself; (b) equal-budget random-search
controls; (c) symmetric hyperparameter treatment (post-hoc Optuna retune of
discovered artifacts, matching the tuned baseline's budget); (d) a
selection-generalization audit that rescores every checkpoint's selected
program on 60k held-out queries; (e) three-objective hypervolume as the
multi-objective endpoint; (f) mechanism ablations of discovered artifacts.

## 4. Campaign 1: objective space, a measured negative

Setup. The evolvable artifact is a group-aware LambdaMART gradient
(grad, hess per impression) with access to raw behavioral signals rel,
booking, and rev per row; training configuration, features, and metrics are
frozen. The seed reproduces XGBoost's built-in rank:ndcg within noise.
Conditions: seeded domain knowledge on/off (3 seeds each, 40 iterations),
later scores-only vs gradient-diagnostic feedback (3 seeds each), plus
equal-budget random search and tuned-LambdaMART baselines.

Selection-fold appearance vs transfer reality. On the fold the loop selects
on, evolved objectives beat the default baseline and improve steadily.
The transfer audit reverses the picture (Table 1): final selected objectives
gain +0.0004 to +0.0014 NDCG@10 on held-out queries (July 3 arms), the best
evolved objective beats its own seed by +0.0002 (p = 0.76), random search
transfers as well or better (p = 0.03 in its favor), and every method is
significantly below the tuned baseline (p = 0.0002). The mechanism is
measured, not conjectured: per-edit effect sizes of 0.002 to 0.005 against a
selection-fold noise floor of plus or minus 0.008 to 0.015 (1.6k queries).
Enlarging the fitness fold to 6k queries (nearly free, evaluation cost is
training-dominated) halves the noise and triples mean transferred gain, and
is the only intervention that moved transfer.

Guidance ablations. Seeded knowledge: about five-fold faster to reach the
random-search-budget level on the selection fold (1 to 3 vs 7 to 23
iterations) and higher hypervolume in all three seeds (mean 252 vs 230,
x1e-6), with unchanged transfer. Diagnostic feedback (noise-gated deltas,
segment decomposition, a one-step gradient alignment probe, per-signal
gradient-usage probes): flat across three seeds on every endpoint. A
byproduct worth reporting: the usage probe showed the discovered revenue term
moves only 0.09 percent of gradient mass while removing it costs 0.009
revenue, so small-but-systematic gradient biases compound across boosting
rounds, and gradient-mass share must not be read as importance.

Mechanism ablation beats further search. Knocking out components of the best
discovered objective attributed its behavior (the listwise Plackett-Luce term
contributed most; the clipped-log revenue gain second) and revealed that its
per-query adaptive blending was harmful: a constant blend improved all three
metrics. Analysis outperformed 40 further loop iterations at the cost of five
evaluations, foreshadowing the value of evidence over score-only feedback,
though as the ablation shows, evidence fed back into this low-headroom space
still cannot create transfer where effect sizes are subnoise.

At full scale the family is a plateau: trained on all 6.9M impressions,
default rank:ndcg, the seed, random search's pick, and the best evolved
objective all land within 0.001 NDCG@10 of each other.

## 5. Campaign 2: pipeline space, a measured positive

Search space. Two evolvable blocks. Block A is feature-construction code over
semantically named columns with a leakage-guarded train-fold statistics API
(count, out-of-fold rate, quantile, stacked linear score). Block B is a
declarative, whitelisted pipeline spec: one to three members drawn from
XGBoost (rank:ndcg, rank:pairwise, pointwise, binary, or the campaign-1
custom objectives), LightGBM (lambdarank, regression, binary), and sklearn
(logistic, random forest, extra trees, ridge), combined by per-query z-score
weighted sum or rank averaging. Feedback is stage-attributed so the loop can
tell which axis earned each change: a features line, per-member validation
scores (model and loss evidence), the ensemble's marginal gain over its best
member, noise-gated metric deltas, and segment decompositions.

Headroom gate. Before any LLM spend, a hand-built candidate from the ICDM
2013 literature had to clear 2.5x the noise floor; within-query rank features
plus count encodings did (+0.013 NDCG@10). The gate also surfaced a
three-generation leakage study we report as a practitioner catalog: naive
train-fold target encoding costs -0.10 NDCG (the model memorizes its own
labels); leave-one-out encoding is subtly worse, -0.13, because subtracting
the row's own label plants a per-row offset perfectly correlated with that
label, which histogram-based trees read as a label detector; query-folded
out-of-fold encoding is clean. Each generation was caught by the audit stack
before any search depended on it. A sparsity floor likewise rejects per-item
rate encodings that are data-starved at loop scale, with an explanatory
message the LLM demonstrably learns from.

Results. Three seeds, 50 iterations each, about 150 LLM calls and ten dollars
per campaign. All three independently converged on the same interpretable
core: five within-query percentile ranks (price, star rating, review score,
location scores), count encodings of prop_id and destination, a lean 20-to-30
column feature set whose program comments cite the measured bloat evidence,
and a diverse three-member ensemble (XGBoost rank:ndcg, LightGBM lambdarank,
plus an extra-trees or binary member) under per-query z-score weighting.
Divergent inventions include per-query price z-normalization and quantile
imputation of a heavy-missing location score. Transfer audit on 60k held-out
queries, fast-train protocol (Table 2): +0.0094 to +0.0131 NDCG@10 over the
seed (about 70 percent of selection-fold gains survive), and all three seeds
beat the Optuna-tuned LambdaMART; the best seed dominates it on all three
objectives simultaneously. The full-data regime (Table 3) required its own
rigor step and yields a methodology lesson: against a baseline whose
hyperparameters were tuned in the fast regime the pipelines win easily, but
re-tuning the baseline AT full scale reverses that naive verdict (a deeper
configuration wins), and only a config-equalized comparison isolates the
discovered structure. With the same tuned member configuration transplanted
in (a lower bound, since it was not re-tuned for the pipeline), the
discovered features and ensemble beat the strictly tuned baseline on all
four reported numbers (0.4619 vs 0.4544 NDCG@10, 0.5185 vs 0.5132 NDCG@38),
so the campaign-1 wash-out was a property of the objective family, not of
scale, and full-scale claims require per-regime tuning on both sides.
Relative to the ICDM 2013 challenge
report (0.531 NDCG@38, hidden leaderboard, full-data training, a 12-model
hand-built ensemble), our numbers are a comparable-protocol approximation on
our own held-out split, in the same ballpark and not claimed as superior; the
supportable statement is that the autonomous loop reconstructs and Pareto-
extends the challenge solutions' load-bearing structure in about two hours.

## 6. Discussion

The two campaigns differ in exactly one variable: the search space. The same
loop, model, feedback machinery, guardrails, and audit produced noise in one
and transfer-audited Pareto dominance in the other. The deciding quantity is
measurable in advance: the ratio of achievable effect sizes to fitness-fold
noise. Objective-space edits on this dataset top out near 0.003 against a
noise floor no cheap fold gets below about 0.004; pipeline-space edits reach
0.013 and clear it. We suggest a headroom gate (a hand-built known-good
candidate must significantly clear the noise floor before any LLM spend) as a
cheap preregistration for discovery loops, and transfer-audited selection as
the reporting standard.

The guidance results reframe where effort should go. Prompt-side knowledge
and evidence-rich feedback are speed and interpretability levers, valuable
for cost and for converging on clean artifacts, but in our measurements they
do not convert a subnoise search space into a productive one, and they were
not necessary for the productive space to work. Evaluator-side investments
(bigger fitness folds, leakage-safe statistics, stage-attributed rejection
messages the model can learn from) carried more of the outcome.

Limitations. One dataset and one deadline-bounded compute budget; three seeds
per condition; NDCG@38 comparisons to the 2013 challenge are approximate by
necessity; the full 2x2 of knowledge x diagnostics was gated off after a flat
probe rather than exhausted; revenue remains a proxy metric; and discovered
pipelines were audited for transfer, not for online effects.

## 7. Conclusion

Open-sourcing an audited failure next to an audited success on the same task
is, we believe, more useful to practitioners than either alone. The loop is
not magic and is not broken: it is a search procedure whose discoveries are
real precisely when the search space offers effect sizes the fitness signal
can resolve. Everything needed to check, reuse, or extend these claims (code,
configs, seeds, per-iteration logs, checkpointed programs, and the audit
scripts) is released.

## Tables (to be typeset)

Table 1, campaign 1 transfer audit: runs/summary/significance_fasttrain_fullfold_test.csv
and generalization_audit.csv arm summary.
Table 2, campaign 2 transfer audit: runs/summary/pipeline_transfer.csv.
Table 3, full-scale comparison incl. NDCG@38: runs/summary/fullscale_ndcg38.csv
(complete: includes lambdamart_optuna_full, the strictly full-scale-tuned
baseline, and pipeline_s1_deepcfg, the config-equalized pipeline).
Figure 1, system + discovered pipeline architecture (paper/figures/).
Figure 2, effect size vs noise floor across the two campaigns.
Figure 3, guidance ablations: convergence bands and transfer bars.
