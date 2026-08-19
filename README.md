# EvoRank

**An autonomous ranking engineer for Learning-to-Rank.** EvoRank proposes
hypotheses, builds candidate ranking pipelines, runs guarded experiments,
learns from stage-attributed feedback, and keeps only what proves itself on
held-out data. You define the objectives. It does the experimentation.

```mermaid
flowchart TB
    G["🚦 <b>gate</b><br/>is this search space worth the spend?"]
    G ==> LOOP
    subgraph LOOP["🔁 <b>search</b> — one LLM call per iteration"]
        direction LR
        P["propose"] --> C["candidate:<br/>features, models,<br/>losses, ensemble"]
        C --> X["guarded<br/>experiment"]
        X --> F["evidence<br/>feedback"]
        F --> AR[("archive")]
        AR -.-> P
    end
    LOOP ==> A["🔍 <b>audit</b><br/>does it transfer to held-out data?"]
    A ==> R["📊 <b>report</b><br/>only proven discoveries count"]

    style G fill:#eef3fb,stroke:#4a6fa5,stroke-width:2px
    style LOOP fill:#f2faf4,stroke:#3f8f5f,stroke-width:2px
    style A fill:#fdf3e7,stroke:#c07a2d,stroke-width:2px
    style R fill:#f6eef9,stroke:#7d5a96,stroke-width:2px
```

```
uv run python evorank_cli.py gate      # will this search space pay off? know BEFORE spending
uv run python evorank_cli.py search --seeds 0,1,2 --iterations 50
uv run python evorank_cli.py audit     # only transfer-proven discoveries count
uv run python evorank_cli.py report
```

## What it did on its first real assignment

Pointed at a public e-commerce hotel-search dataset (9.9M impressions, three
competing objectives: relevance, conversion, revenue), three independent
EvoRank seeds each converged in 50 iterations, roughly two hours and ten
dollars of LLM spend, on ranking pipelines that beat an Optuna-tuned
LambdaMART baseline on 60k held-out queries: significantly better relevance
and conversion in every run, with revenue matching or better (the best run
beats the baseline on all three). The advantage holds at full data scale
under a capacity-equalized comparison:

| trained on 280k queries, tested on 60k | NDCG@10 | NDCG@38 |
|---|---|---|
| LambdaMART, Optuna-tuned at full scale | 0.4544 | 0.5132 |
| EvoRank pipeline (s1 features, tuned XGBoost member, scaled LightGBM member) | **0.4619** | **0.5185** |

The pipeline row is `discovered/pipeline_s1_deepcfg.py`: the s1 recipe with the
baseline's tuned XGBoost settings on its XGBoost member, a scaled-up LightGBM
member, and no third member; it received no search or tuning of its own. The
s1 features alone under the tuned single model reach 0.4587 NDCG@10.

The discovered pipelines are code you can read: within-query rank features,
count encodings, lean feature sets, and a diverse three-member ensemble under
per-query z-score weighting. All three seeds found the same core design
independently, and their own code comments cite the experimental evidence
they learned it from. The programs are in `discovered/`.

## Why it is different: the audit stack

Autonomous experimentation systems have a failure mode nobody talks about:
under noisy evaluation they select lucky candidates and report steady
progress while discovering nothing real. We measured this happening, on this
codebase, before we fixed it. EvoRank ships with the fix built in:

- **The gate.** Before a single LLM call, `evorank gate` measures your
  fitness noise floor and demands that a hand-built reference candidate beat
  it decisively. If your search space has no headroom, EvoRank tells you to
  keep your money.
- **The audit.** `evorank audit` rescores every selected pipeline on a large
  held-out fold. Selection-fold luck does not survive it. Our own first
  campaign did not survive it either, and that story ships in the paper
  rather than in a drawer.
- **Guarded experiments.** Leakage-safe out-of-fold target encodings (we
  cataloged and measured three generations of encoding leakage so you do not
  have to), whitelisted model and loss families, parameter clamps, wall
  budgets, and rejection messages written so the model learns from them.

## How the loop works

One LLM call per iteration mutates two blocks of a candidate program:

1. **Feature construction**, real code over semantically named columns, with
   a train-fold statistics API for counts, out-of-fold rates, quantiles, and
   stacked scores.
2. **Pipeline spec**, a declarative choice of one to three members (XGBoost,
   LightGBM, sklearn families; ranking, pointwise, binary, or custom losses)
   and an ensemble combiner.

Every candidate gets stage-attributed feedback: which features changed, how
each member scored, what the ensemble added over its best member, and
noise-gated metric deltas so the model can tell signal from luck. Survivors
enter a Pareto archive across your objectives. Rinse, repeat, audit.

![system flowchart](paper/figures/evorank_flowchart.png)

## Install

EvoRank runs inside a SkyDiscover checkout (the evolutionary engine):

```
git clone https://github.com/skydiscover-ai/skydiscover.git && cd skydiscover   # the engine
git clone https://github.com/shabazpatel/evorank.git                          # this repo, inside it
uv sync
uv pip install -r evorank/requirements.txt   # rerun after any `uv sync`
cd evorank
uv run python evorank_cli.py gate
```

Set an LLM key in the repo root `.env` before `search`. The shipped configs use
`claude-sonnet-5` through Anthropic's OpenAI-compatible endpoint
(`ANTHROPIC_API_KEY`); any OpenAI-compatible model works by editing
`llm.models` and `llm.api_base` in `configs/*.yaml`.

## Reproduce the paper

Data: download the Kaggle competition `expedia-personalized-sort` (ICDM 2013)
and place `train.csv` (and `test.csv`, needed only for the Kaggle late
submission) under `data/raw/`. Then, from `evorank/`:

```
# folds: 70/15/15 by query; fast subset 8k train / 6k val / 6k test queries
uv run python data/prepare_expedia.py --raw data/raw/train.csv --fast-queries 8000 --fast-eval-queries 6000
uv run python data/data_report.py

# baselines (write runs/*.json; the tuned records are committed)
uv run python baselines/lambdamart_default.py
uv run python baselines/lambdamart_optuna.py            # 40 trials, small fold
uv run python baselines/lambdamart_optuna.py --full     # 40 trials, full scale
uv run python baselines/random_search.py --trials 40    # equal-budget control
uv run python baselines/lambdaloss.py

# campaign 1 (training objectives): seeded vs unseeded, 3 seeds x 40 iterations
EVORANK_FEEDBACK=rich    uv run python evorank_cli.py search --seeds 0,1,2 --iterations 40 \
    --name mem_on  --config configs/evorank_memory_on.yaml  --seed-program seed/initial_program.py --evaluator eval/evaluator.py
EVORANK_FEEDBACK=minimal uv run python evorank_cli.py search --seeds 0,1,2 --iterations 40 \
    --name mem_off --config configs/evorank_memory_off.yaml --seed-program seed/initial_program.py --evaluator eval/evaluator.py

# campaign 2 (full pipelines): gate, then 3 seeds x 50 iterations, then audit
uv run python evorank_cli.py gate
uv run python evorank_cli.py search --seeds 0,1,2 --iterations 50
uv run python evorank_cli.py audit && uv run python evorank_cli.py report

# analyses behind each paper number (all write runs/summary/*.csv)
uv run python analyze/generalization_audit.py           # campaign 1 transfer audit
uv run python analyze/significance.py --eval-full --fold test --resamples 5000
uv run python analyze/mechanism_ablation.py             # component-removal tests
uv run python analyze/hypervolume.py                    # frontier quality
uv run python analyze/pipeline_transfer.py              # Table 1
uv run python analyze/fullscale_ndcg38.py               # Table 2 (all rows)
uv run python analyze/bootstrap_headline.py             # headline CIs and p-values
uv run python analyze/kaggle_submission.py              # Kaggle late-submission files
```

Every run's final `best_program.py`, the tuned-baseline records, and the launch
configs are committed under `runs/`, so the analysis scripts run from a fresh
clone once the data is prepared. Note that scripts without flags run in full
when invoked; there is no `--help` dry run. The double-budget search-algorithm
arms in the Discussion use `configs/evorank_pipeline_v2.yaml` (recalibrated
AdaEvolve) and `configs/evorank_pipeline_evox.yaml` (EvoX).

Environment switches (campaign 1 evaluator, `eval/evaluator.py`): `EVORANK_FAST=0`
scores on the full folds instead of the fast subset; `EVORANK_FEEDBACK=minimal|rich|diagnostic`
selects the feedback channel; `EVORANK_EVAL_TIMEOUT_S` caps one evaluation. The
campaign 2 evaluator (`eval/evaluator_pipeline.py`) always uses the fast subset;
full-fold results come from `analyze/pipeline_transfer.py` and `analyze/eval_full.py`.

## Bring your own data

Three parquet folds (train/val/test) with a `qid` column, numeric features,
and label columns, rows contiguous per query. That is the whole contract.
The reference preparation for the Expedia ICDM 2013 dataset is in
`data/prepare_expedia.py`; heavier adapters are deliberately optional.

## The research behind it

This repository is the full artifact of a two-campaign study: one campaign
where the loop looked successful and provably was not (mechanism measured:
per-edit effect sizes below the fitness noise floor), and one where it
produced the transfer-audited results above. Every number in the paper traces
to a CSV in `runs/summary/` generated by a script in `analyze/`. Draft and
design docs are in `paper/`.

```
evorank_cli.py        gate / search / audit / report
eval/                 guardrailed evaluators + stage-attributed feedback
seed/                 seed programs (the starting points for evolution)
discovered/           what the loop found, as runnable code
analyze/              the audit stack: bootstrap, transfer, hypervolume, ablation
runs/summary/         machine-generated results, never hand-edited
paper/                design docs, figures, draft
```

## Paper

*EvoRank: LLM-Guided Evolution of Multi-Objective Learning-to-Rank Pipelines.*
Rayhan Patel and Shabaz Patel. Accepted at GenAIECommerce 2026, the Third
Workshop on Agentic and Generative AI for E-commerce, co-located with RecSys
2026 (Minneapolis, MN, USA, September 28, 2026). Camera-ready PDF:
[`paper/EvoRank_GenAIECommerce2026.pdf`](paper/EvoRank_GenAIECommerce2026.pdf);
LaTeX source in `paper/latex/`.

```bibtex
@inproceedings{patel2026evorank,
  title     = {EvoRank: LLM-Guided Evolution of Multi-Objective Learning-to-Rank Pipelines},
  author    = {Patel, Rayhan and Patel, Shabaz},
  booktitle = {Proceedings of the Third Workshop on Agentic and Generative AI for
               E-commerce (GenAIECommerce 2026), co-located with RecSys 2026},
  series    = {CEUR Workshop Proceedings},
  publisher = {CEUR-WS.org},
  year      = {2026},
  note      = {To appear. Code: \url{https://github.com/shabazpatel/evorank}}
}
```

## License

Apache-2.0. If this tool or its methodology is useful in your work, please
cite the paper above.
