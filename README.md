# EvoRank

REA-style autonomous discovery for Learning-to-Rank pipelines, with a
built-in honesty harness. An LLM-guided evolutionary loop searches feature
construction, model selection, loss selection, and ensemble architecture for
GBDT-family rankers, and every discovery is audited for transfer to held-out
data before it counts.

Built on the SkyDiscover engine (this repository's parent directory). The
research story, including a fully measured negative campaign in objective
space and a positive, transfer-audited campaign in pipeline space, is in
paper/draft.md and runs/summary/.

## The workflow (evorank_cli.py)

```
uv run python evorank_cli.py gate      # headroom check BEFORE any LLM spend
uv run python evorank_cli.py search --seeds 0,1,2 --iterations 50
uv run python evorank_cli.py audit     # transfer audit on the held-out fold
uv run python evorank_cli.py report    # one comparison table for everything
```

- gate: evaluates the seed pipeline and a hand-built reference candidate,
  reports the fitness noise floor, and refuses to bless a search space whose
  known-good candidate cannot beat it. This is the step the discovery-loop
  literature skips; it predicts whether the loop will select real
  improvements or noise on your data.
- search: one LLM call per iteration mutates two blocks, feature-construction
  code and a whitelisted declarative pipeline spec (xgb, lgbm, sklearn
  families; rank, pointwise, binary, and custom losses; z-score or rank-mean
  ensembles). Candidates get stage-attributed feedback: per-member scores,
  the ensemble's margin over its best member, noise-gated metric deltas
  against the seed, and segment decompositions. Resume-aware from checkpoints.
- audit: rescores selected pipelines on a large held-out fold, so
  selection-fold luck cannot masquerade as discovery.
- report: aggregates all runs and baselines.

## Data contract

Bring your own preparation (adapters are optional by design): three parquet
folds, train/val/test, each with a qid column, numeric feature columns, and
label columns, sorted so each qid is contiguous. This project's reference
preparation for the Expedia ICDM 2013 dataset is data/prepare_expedia.py;
the evaluator's guardrails (leakage-safe out-of-fold encodings with a
sparsity floor, parameter clamps, wall budgets, degenerate-candidate
rejection) apply regardless of dataset.

## Headline result (details in runs/summary/PIPELINE_CAMPAIGN.md)

Three independent seeds converged in 50 iterations (about ten dollars each)
on interpretable pipelines that beat an Optuna-tuned LambdaMART baseline on
60k held-out queries on all three objectives (relevance, conversion,
revenue), an advantage that persists when retrained on the full 6.9M-row
dataset. The same loop pointed at objective space alone discovers nothing
that transfers; the difference, measured, is search-space headroom relative
to fitness noise. Run the gate first.

## Repository map

```
evorank_cli.py            the tool: gate / search / audit / report
eval/evaluator_pipeline.py  guardrailed pipeline evaluator + feedback
seed/initial_pipeline.py    seed program (two EVOLVE blocks)
configs/                    campaign configs (system message = search contract)
knowledge/                  optional domain priors injected into the prompt
analyze/                    audit stack: bootstrap significance, transfer
                            audits, hypervolume, mechanism ablation, aggregate
data/                       reference dataset preparation + gate
paper/                      design docs, figures, draft
runs/summary/               machine-generated results, never hand-edited
```

Part 1 artifacts (objective-space campaign: seed/initial_program.py,
eval/evaluator.py with gradient-diagnostic feedback) are retained; they
produced the measured negative result and the diagnostics that motivated the
pipeline campaign.

## Install

EvoRank runs inside a SkyDiscover checkout (the evolutionary engine):

```
git clone <skydiscover repo> && cd skydiscover
uv sync
uv pip install -r evorank/requirements.txt   # rerun after any `uv sync`, it prunes extras
cd evorank
uv run python evorank_cli.py gate
```

Set an LLM key in the repo root .env (ANTHROPIC_API_KEY, OPENAI_API_KEY, or
any endpoint SkyDiscover's OpenAI-compatible client reaches) before `search`.

## Discovered artifacts

discovered/ contains the committed best programs from both campaigns: the
three converged pipelines (pipeline_s*_best.py), the best evolved objective
from the negative campaign, and its ablation-simplified variant. Each is a
runnable candidate under the corresponding evaluator.

## Notes

- LLM: any OpenAI-compatible endpoint SkyDiscover supports; campaigns here
  used claude-sonnet-5 (note: rejects the temperature parameter, configs set
  temperature: null).
- Every number in the paper traces to a CSV under runs/summary/ produced by
  a script under analyze/. No manual transcription.
- Prose in this repository uses commas and periods, no em dashes.
