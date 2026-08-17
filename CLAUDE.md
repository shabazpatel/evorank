# CLAUDE.md — EvoRank (Option 2: Agentic LTR Experimentation Loop)

Project context for Claude Code. Read this fully before writing any code.

## 0. What we are building

EvoRank is an open, reproducible **agentic experimentation loop for supervised, multi-objective Learning-to-Rank (LTR)** on e-commerce search data. It uses the SkyDiscover framework (this repo) as the evolutionary engine and adds an LTR-specific evaluator, a seeded domain-knowledge memory, and a multi-objective Pareto setup over real behavioral signals.

Target venue: a peer-reviewed workshop short paper, hard deadline July 20, 2026 (venue name withheld in-repo). We are writing a 5-page short paper. Scope discipline matters more than completeness.

Inspiration: Meta's Ranking Engineer Agent (REA) and GEARS (closed, internal, DNN-based). Our angle is the open, tree-based (GBDT), reproducible counterpart.

## 0.1 Current repository state (keep this updated as you build)

Status August 17, 2026: the paper was ACCEPTED at GenAIECommerce 2026 (RecSys 2026 workshop, Minneapolis, September 28). Camera-ready is due August 28 on EasyChair; the camera-ready source is `paper/latex/main.tex` (build with tectonic or pdflatex; 8 pages), the PDF is `paper/EvoRank_GenAIECommerce2026.pdf`, the arXiv bundle is `paper/arxiv/`, and every change against the submitted version plus the code-review outcomes are in `paper/camera_ready_changes.md`. `paper/latex/main_anon.tex` is the submitted (double-blind) text and is not edited.

Phases 0 through 4 of the Section 14 build order are COMPLETE as of July 3, 2026. The full directory tree exists and the pipeline runs end to end on real data. Start any new session by reading `runs/summary/MORNING_SUMMARY.md` (headline table, ablation numbers, discovered-objective analysis) and re-running `uv run python analyze/aggregate.py`.

What exists and is verified:
- Phase 1: `ltr/metrics.py` (three genuinely distinct metrics; revenue is pooled dollar-weighted capture, see Section 6), `ltr/dataset.py` (contract + synthetic fallback + fixed training harness), `seed/initial_program.py` (5-arg signature per Section 7, matches built-in rank:ndcg, gap -0.0039), `eval/evaluator.py` (guardrails plus EVORANK_FEEDBACK=rich|minimal for the ablation and EVORANK_FAST=0 for full-split evaluation), `data/` scripts, prepared parquet (fast subset 8k train queries, ~40s per evaluation).
- Phase 2 (all logged to `runs/*.json`): lambdamart_default 0.4230 ndcg, lambdamart_optuna 0.4382 (the bar), lambdaloss (rank-distance NDCG-Loss2) 0.3816, random_search_s0 0.4301 at the 40-eval budget.
- Phase 3/4: `knowledge/ltr_priors.md` (the treatment; do not edit mid-experiment-set), `configs/evorank_memory_{on,off}.yaml` (verified to differ only in system_message), six ablation runs at 40 iterations in `runs/mem_{on,off}_s{0,1,2}` with checkpoints, plus `runs/val_mem_*` (12-iter validation) and `runs/smoke`.
- Key result so far: memory-on and memory-off reach the same final quality (mean best ndcg 0.4308 both) but memory-on reaches the random-search-budget level in 1-3 iterations vs 7-23; no evolved objective beat tuned LambdaMART on ndcg at this budget under fixed hyperparameters (post-hoc retune control addresses the asymmetry). Best discovered objective: `runs/mem_on_s1/checkpoints/checkpoint_40/best_program.py` (clipped-log revenue gain + adaptive pairwise/listwise blend).

Feedback treatment arms (July 3 afternoon onward): `EVORANK_FEEDBACK` selects minimal (scores only), rich (legacy memory-on arm from the July 3 morning runs), or diagnostic (`eval/diagnostics.py`: noise-gated deltas vs a cached seed reference, segment decomposition, behavior stats, gradient alignment probe, per-signal gradient usage). The fast subset was regenerated with 6k-query val/test folds (`data/prepare_expedia.py --fast-only --fast-eval-queries 6000`), same 8k train queries, which halves fitness noise; runs on the new subset are NOT metric-comparable with the July 3 morning runs. Probe launcher for the feedback ablation: `runs/_configs/launch_probe.sh`. Calibration lesson baked into the usage probe: the discovered revenue term moves ~0.09 percent of gradient mass yet shifts the revenue metric by ~0.009, so low gradient-mass share must not be read as unused.

Operational facts that will save you time:
- LLM: claude-sonnet-5 via the Anthropic OpenAI-compatible endpoint (`https://api.anthropic.com/v1/`); it REJECTS the temperature parameter, so configs set `temperature: null`. Key comes from repo-root `.env` (`set -a; source .env; set +a`).
- Run the CLI from the repo root: `uv run python -m skydiscover.cli evorank/seed/initial_program.py evorank/eval/evaluator.py -c <config> -s adaevolve -i N -o evorank/runs/<name>`. The `skydiscover-run` console script is broken; use `python -m`.
- The project venv is uv-managed; `uv sync` PRUNES manually installed packages (pandas, pyarrow, optuna, scikit-learn, xgboost), so re-`uv pip install` them after any sync.
- Long runs: background shell tasks are killed after ~45 minutes; launch multi-hour work detached (`nohup bash runs/_configs/launch_ablation.sh &`, which is resume-aware from checkpoints).

Remaining work for the paper: full-split (EVORANK_FAST=0) and test-fold evaluation of the best programs and baselines, hypervolume for the Pareto-quality claim, possibly longer runs (T=80+) to close the optuna gap, figures from `runs/summary/*.csv`, and the 5-page write-up.

PART 2 (July 7, the positive result): the pipeline search space (features x model x loss x ensemble; `eval/evaluator_pipeline.py`, `seed/initial_pipeline.py`, design in `paper/pipeline_design.md`) produced transfer-audited discoveries: all 3 seeds beat the re-baselined tuned LambdaMART on the 60k-query test fold, pipeline_s2 Pareto-dominates it (see `runs/summary/PIPELINE_CAMPAIGN.md`). The two-part story: objective space had no headroom above fitness noise (Part 1 negative, mechanism measured); pipeline space does (Part 2 positive). Remaining: formal bootstrap on the headline comparison, optional full-train-regime check, figures, write-up.

Paper documentation lives in `paper/approach.md` (method description, mermaid flowchart source, terminology glossary for the planned extensions A1 phase-alternated HPO, B combination stage, C MSLR zero-shot transfer, diagnostic feedback 2x2, plus figure plan and reproducibility inventory). Rendered flowchart: `paper/figures/evorank_flowchart.svg`.

The SkyDiscover engine lives in the parent repo; treat it as a black box per Section 3 and see the repo-root `CLAUDE.md` for its commands and architecture. A second, older SkyDiscover checkout exists at `.claude/worktrees/` (a Claude Code worktree); ignore it, it is not the framework you run against.

### 0.2 Data (resolved): correct dataset is in place

The correct **Expedia Personalized Hotel Search** (`expedia-personalized-sort`, ICDM 2013) `train.csv` (2.2GB, full schema: `srch_id, prop_id, booking_bool, click_bool, gross_bookings_usd, position, random_bool, price_usd, prop_starrating, comp1..8_*, ...`) is extracted to `data/raw/train.csv` (the default `--raw` path for `prepare_expedia.py`). Only the training file is used; the Kaggle `test.csv` has hidden labels (no `*_bool`, no `gross_bookings_usd`), so the test fold comes from splitting `train.csv` 70/15/15 by `srch_id` (Section 5), not from Kaggle's test set. The wrong competition download (Expedia Hotel Recommendations, hotel-cluster classification) still sits in `expedia-hotel-recommendations/`; it is gitignored and unused, and can be deleted to reclaim ~4.5GB.

## 1. Novelty levers (the reason this is a paper, not a demo)

Running SkyDiscover on a new dataset is **not** a contribution, and a reviewer will say so. Frame the work as a finding, not a system: the headline claim is that LLM-guided evolution discovers e-commerce ranking objectives that dominate hand-crafted losses on the relevance, conversion, and revenue frontier, and we explain what makes them work. The loop is only the method that produces that result. The publishable novelty depends entirely on the three things below. Treat them as first-class requirements, not extras. Do not write any docstring, README, or comment that claims the framework itself is the contribution.

1. **Open agentic loop for supervised multi-objective LTR.** REA/GEARS are closed and DNN-based; RankEvolve (arXiv 2602.16932) is unsupervised lexical retrieval with a scalar fitness. We are the open, GBDT, multi-objective version. Reproducibility (seeds, configs, logs) is therefore part of the contribution, not hygiene.
2. **Domain-seeded memory + ablation.** We inject a curated LTR knowledge base as priors into the agent's context. The headline experiment is an ablation: identical run with the knowledge base ON vs OFF, measuring iterations-to-baseline and final Pareto quality. This must be measurable from logs.
3. **Real e-commerce multi-objective Pareto.** Three competing objectives computed from genuine behavioral labels (clicks, bookings, booking revenue), not a simulated proxy.

Positioning to reflect in generated docs (do not overclaim): related and prior work includes RankEvolve (unsupervised lexical, scalar), GEARS/REA (closed, DNN), LambdaLoss (Wang et al. 2018, hand-derived metric-driven loss; our discovered counterpart and a required baseline), Unbiased LambdaMART (Hu et al. 2019; note XGBoost already ships this as `lambdarank_unbiased`, so any debiasing the agent finds is a rediscovery, not new), and the automated loss-search lineage GLO (2019), AM-LFS (2019), AutoLoss-Zero (2021), CSE-Autoloss (2021). Our delta vs that lineage: LLM-guided program evolution (not genetic-primitive), ranking metrics and GBDT lambda-gradients (not vision losses), multi-objective, and memory-driven.

### 1.1 The three figures that carry the paper

Build the logging and analysis so these three artifacts are producible. They are the contribution made visible:

1. **The discovered objective.** The evolved grad/hess code, a plain-language explanation of its mechanism, and an ablation of that mechanism. Highest value, lowest engineering. Rediscovering a known principle (NDCG-style weighting, position discounting) validates the method; finding something humans have not written is new knowledge.
2. **Pareto domination.** The relevance, conversion, revenue frontier with evolved objectives plotted above tuned LambdaMART and LambdaLoss.
3. **Memory ablation curve.** Best-so-far vs iteration, memory-on vs memory-off, with seeds.

The single baseline that makes all of this believable is random search at equal generation budget (Section 11). Beating only default XGBoost reads as "just compute"; beating random-search-at-equal-budget and LambdaLoss shows the guidance and the discovery are both real.

Target abstract sentence: we show that LLM-guided evolutionary search discovers interpretable Learning-to-Rank objectives that Pareto-dominate hand-crafted losses across relevance, conversion, and revenue on real e-commerce search data, and that seeding the search with domain knowledge measurably accelerates discovery.

## 2. Environment and dependencies

- This repo uses `uv` and Python >= 3.10. Confirm with `uv sync` before anything else.
- Set the LLM key: `export OPENAI_API_KEY=...` (or `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`; SkyDiscover resolves LiteLLM `provider/model` strings). I have AWS Bedrock access, so a `bedrock/...` model string is acceptable if cheaper.
- Extra Python deps for our work: `xgboost`, `pandas`, `numpy`, `pyarrow`, `optuna`, `scikit-learn`. Install into the uv environment: `uv pip install xgboost pandas numpy pyarrow optuna scikit-learn`.
- Do NOT modify SkyDiscover's own source (`skydiscover/`, `benchmarks/`, `configs/`). Treat the framework as a black box accessed only through its documented API (Section 3). Put all of our work under `projects/evorank/`.

## 3. SkyDiscover API contract (verified — conform exactly)

Before writing our evaluator, **read these real templates** and mirror their structure:
`benchmarks/math/circle_packing/initial_program.py`, `.../evaluator.py`, `.../config.yaml`.

Key facts:
- Evaluator is a Python file exposing `def evaluate(program_path) -> dict`. The dict must contain `combined_score` (float, maximized) unless Pareto mode supplies objective keys. Optional `artifacts` (dict) is injected into the next LLM prompt as context. This `artifacts` channel is how we feed insights back, so use it deliberately.
- Seed/initial program marks the mutable region with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`. Everything outside is frozen. If no markers, the whole file is mutable (we do not want that — always use markers).
- Pareto mode: set `search.type: adaevolve` and `search.database.pareto_objectives: [...]` in the config, and return those exact metric keys from `evaluate`. In Pareto mode `combined_score` becomes an optional scalar fallback.
- Run via CLI: `uv run skydiscover-run <initial_program.py> <evaluator.py> --config <config.yaml> --search adaevolve --iterations N -o <output_dir>`.
- Or Python API: `from skydiscover import run_discovery; run_discovery(initial_program=..., evaluator=..., search="adaevolve", model=..., iterations=...)`. It returns an object with `.best_score` and `.best_solution`.
- Config keys we use: `max_iterations`, `llm.models: [{name, weight}]`, `search.type`, `search.database.pareto_objectives`, `prompt.system_message`, `monitor: {enabled: true}` (live dashboard + human-steering).
- Use `--search adaevolve` (fast early gains at low budget, T<=50). Do NOT use EvoX for the sprint (it shines only at long horizons) and do NOT reimplement MCGS or any search internals.

## 4. Repository layout to create

```
projects/evorank/
  data/
    prepare_expedia.py        # raw CSV -> clean parquet + splits
    schema.md                 # column meanings, label mapping, decisions
  ltr/
    metrics.py                # ndcg@k, book_ndcg@k, revenue@k (all query-grouped)
    train_xgbranker.py        # shared training harness (Booster API, group-aware)
  seed/
    initial_program.py        # LambdaMART objective with EVOLVE-BLOCKs
  eval/
    evaluator.py              # SkyDiscover evaluate(program_path) wrapper
  configs/
    evorank_memory_on.yaml    # adaevolve + pareto + seeded knowledge base
    evorank_memory_off.yaml   # adaevolve + pareto + generic system message
  baselines/
    lambdamart_default.py
    lambdamart_optuna.py      # tuned baseline (REQUIRED)
    lambdaloss.py             # LambdaLoss objective (REQUIRED)
    random_search.py          # equal-budget control
  knowledge/
    ltr_priors.md             # the seeded domain knowledge base (text)
  notebooks/
    foundation.ipynb          # Phase 0/1 de-risking (provided, verified)
  runs/                       # all outputs, logs, checkpoints (gitignored)
  analyze/
    aggregate.py              # parse runs -> tables/plots for the paper
  README.md
```

## 5. Data: Expedia Personalized Hotel Search (ICDM 2013)

Source: Kaggle `expedia-personalized-sort`. I will place `train.csv` under `projects/evorank/data/raw/`. Do not attempt to download it.

`prepare_expedia.py` must:
- Use the **training file only**. The Kaggle test set has hidden labels, so split the training file by `srch_id` into train/val/test (70/15/15 by query, never split a query across folds).
- Build the graded relevance label per (srch_id, prop_id): `gain = 5 if booking_bool == 1 else (1 if click_bool == 1 else 0)`.
- Keep query groups via `srch_id` (XGBoost needs contiguous group sizes; sort by `srch_id`).
- Feature set: the numeric search/hotel/competitor columns (price_usd, prop_starrating, prop_review_score, prop_location_score1/2, prop_log_historical_price, srch_length_of_stay, srch_booking_window, srch_adults_count, orig_destination_distance, the comp* columns, etc.). Document inclusions/exclusions in `schema.md`. Drop leakage columns: `position`, `gross_bookings_usd` (revenue is a label, never a feature), `click_bool`, `booking_bool`. Handle heavy-missing comp* columns explicitly (impute + missing-indicator).
- Preserve `gross_bookings_usd` and `random_bool` as **label-side** columns in a separate array for metric computation (not in X).
- Write a **fast subset** (sample ~25-40k srch_ids) for the evolution loop, and the **full split** for final reporting. Each candidate evaluation in the loop must run in under ~60s; subsample to hit that.
- Acceptance: print row/query counts per fold, label distribution, feature count, and confirm no query spans folds.

`random_bool == 1` isolates the position-unbiased subset. This is an optional asset for a later debiasing sub-result; do not block the main pipeline on it.

## 6. Metrics (`ltr/metrics.py`)

All metrics are query-grouped (per `srch_id`), averaged over queries, evaluated at k=10. Implement and unit-test each:
- `ndcg_at_k`: graded relevance gain (5/1/0). Primary relevance objective.
- `book_ndcg_at_k`: gain = `booking_bool` (binary). Pure conversion objective.
- `revenue_at_k` (`revenue_capture_at_k` in code): position-discounted booking revenue (`booking_bool * gross_bookings_usd`) captured in top-k, **pooled across queries** and normalized by the ideal revenue ordering. Range [0,1]. This is an explicit revenue **proxy**; label it as such in code and paper. Only booked items contribute. IMPORTANT: pooling (rather than the per-query averaging used by `ndcg`/`book_ndcg`) is deliberate. Expedia searches have at most one booking each, so a per-query-averaged NDCG on booking revenue collapses exactly onto `book_ndcg` (per-query NDCG cancels the magnitude of a lone positive). Pooling keeps the dollar magnitude, so high-value searches dominate the revenue axis and the objective genuinely competes. Verified on the data: the seed scores ndcg 0.418 / book_ndcg 0.439 / revenue 0.389 (three distinct values); with per-query averaging revenue equalled book_ndcg exactly. The revenue metric uses **raw** dollars and is intentionally not winsorized or log-scaled; any such transform is for the search to discover inside the objective, not a hand-set choice in the fitness metric.

These three genuinely compete: relevance and conversion weight every query equally, while revenue weights by dollars, which is what produces a non-trivial Pareto frontier.

## 7. Seed program (`seed/initial_program.py`)

The seed program contains **only the evolvable objective**, wrapped in EVOLVE-BLOCK markers. It does not load data or train models. Keeping data loading and the training loop out of the mutable file means the LLM can only change the objective, which is exactly what we want it to discover. The evaluator (Section 8) owns everything else and imports this function.

Signature and baseline. The objective is handed all three behavioral signals per row (`rel`, `booking`, `rev`), extending the notebook's original `(predt, labels, group_slices)` form. This is the key design decision: the seed body uses only `rel` (so it still reproduces built-in `rank:ndcg` within noise, the Phase 1 correctness check), but `booking` and `rev` are exposed so the search can discover how to fold conversion and revenue into the loss. How to weight and transform revenue (raw, log, clipped, per-pair) is for the search to discover, not a hand-set choice. The authoritative version is `seed/initial_program.py`; the shape is:

```python
import numpy as np

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):
    """Group-aware LambdaMART gradient over three behavioral signals (rel,
    booking, rev), one (start, end) slice per query. Returns (grad, hess) per row.
    Baseline: pairwise-logistic lambda with |deltaNDCG| weighting on the
    relevance gain; booking and rev are exposed but unused in the seed.
    SkyDiscover mutates this body."""
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); lab = rel[s:e]; m = len(sc)
        if m < 2:
            continue
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))          # sigmoid(-sigma*S)
        M = (lab[:, None] > lab[None, :])              # i more relevant than j
        gain = (2.0 ** lab - 1.0)
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0)
        idcg = float(np.sum(np.sort(gain)[::-1] * (1.0 / np.log2(np.arange(2, m + 2))))) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)  # i gets lam, j gets -lam
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)                # keep hessian positive
# EVOLVE-BLOCK-END: objective
```

Training hyperparameters (depth, eta, num_boost_round) stay **fixed in the evaluator**, so the search isolates objective discovery rather than HPO. A second `build_pairs` block (topk vs mean vs novel pair weighting) can be added later; start with the single objective block to keep the search space tight. Set all seeds.

## 8. Evaluator (`eval/evaluator.py`)

Implements the SkyDiscover contract. It imports the candidate program, builds the DMatrix from the prepared fast subset, trains with the candidate objective via `xgb.train(..., obj=...)`, predicts on val, and computes the three metrics. This shape (import-from-path plus the rejection guard) is verified in `notebooks/foundation.ipynb`:

```python
def evaluate(program_path):
    try:
        mod = load_program(program_path)               # importlib from path
        dtr, group_slices_tr, dva, va_df = load_fast_subset()
        rel, booking, rev = train_signals()            # aligned to DMatrix row order
        def obj(predt, dtrain):
            g, h = mod.lambdamart_objective(predt, rel, booking, rev, group_slices_tr)
            if not (np.all(np.isfinite(g)) and np.all(np.isfinite(h))):
                raise ValueError("non-finite grad/hess")
            return g, h
        bst = xgb.train(FIXED_PARAMS, dtr, num_boost_round=FIXED_ROUNDS, obj=obj)
        m = metrics_bundle(va_df, bst.predict(dva))     # ndcg, book_ndcg, revenue
        return {**m, "combined_score": m["ndcg"],
                "artifacts": {"feedback": one_line_diagnostic(m)}}
    except Exception as ex:
        return {"ndcg": 0.0, "book_ndcg": 0.0, "revenue": 0.0, "combined_score": 0.0,
                "artifacts": {"feedback": f"rejected: {type(ex).__name__}: {ex}"}}
```

Guardrails (these are the loss-rejection analog, and they matter):
- Wrap training in a timeout. If it exceeds the budget, return all-zeros with a feedback string.
- Reject NaN/Inf grad/hess or non-finite metrics: return zeros with feedback explaining the failure. Never let a degenerate objective score well.
- Determinism: fixed seed, fixed data subset per run.
- The `artifacts.feedback` is fed to the next LLM step. In the **memory-on** condition make it informative (e.g., "ndcg up 0.4pt but revenue collapsed: objective over-weights exact bookings"); in the **memory-off** condition keep it minimal (scores only). This difference is part of the ablation.

## 9. Configs

`evorank_memory_on.yaml`: `search.type: adaevolve`, `search.database.pareto_objectives: [ndcg, book_ndcg, revenue]`, a model pool, `monitor.enabled: true`, and a `prompt.system_message` that **embeds the contents of `knowledge/ltr_priors.md`** (the seeded domain knowledge). 

`evorank_memory_off.yaml`: identical except `prompt.system_message` is generic ("You are optimizing a ranking objective; improve the metrics.") and contains none of the LTR priors.

Everything else (iterations, model, seed, data subset) must be identical between the two configs. The only difference is the knowledge base. Keep the two files diffable.

## 10. Seeded knowledge base (`knowledge/ltr_priors.md`)

A concise, curated text block of ranking priors the agent can draw on. Include: the LambdaMART lambda-gradient form and why NDCG-delta weighting is used; the LambdaLoss idea that the loss should be tied to the target metric; the topk-vs-mean pair-construction tradeoff and when each helps; monotonic score transforms and position discounting; common failure modes (degenerate constant scores, exploding gradients, over-weighting head bookings at the expense of revenue). Keep it factual and compact (roughly one page). This file IS the experimental treatment, so version it and never edit it mid-experiment-set.

## 11. Baselines (required for a credible result)

- `lambdamart_default.py`: XGBRanker `rank:ndcg`, out of the box.
- `lambdamart_optuna.py`: same model, hyperparameters tuned with Optuna on val. **Beating only the default is not a result; you must beat the tuned baseline.**
- `lambdaloss.py`: a LambdaLoss-style metric-driven objective. This is the hand-derived counterpart to what we discover and the most important comparison.
- `random_search.py`: random objective mutations at the **same generation budget** as the EvoRank run, to show the LLM-guided loop beats blind search.

All baselines log the same three metrics on the same splits, into `runs/` in the same schema as the evolutionary runs.

## 12. The headline experiment: memory ablation

Run matrix:
- `evorank_memory_on.yaml` x 3 seeds
- `evorank_memory_off.yaml` x 3 seeds
- Same iterations (start with 40), same model, same fast subset.

Report from logs:
- **Iterations to reach the tuned-LambdaMART NDCG** (efficiency).
- **Final Pareto quality** (hypervolume over the three objectives, plus best NDCG@10).
- Convergence curves (best-so-far vs iteration), memory-on vs memory-off.

Claim to support: domain-seeded memory makes the autonomous loop converge faster and reach a better frontier. If the data does not support it, report that honestly; a negative result here is still a finding.

## 13. Logging and reproducibility (part of the contribution)

- Every run writes: the exact config, the git commit, the seed, per-iteration metrics for all three objectives, the Pareto frontier at each step, and the final best program's source. SkyDiscover checkpoints to `-o <dir>`; point all runs at `projects/evorank/runs/<run_name>/` and keep them.
- `analyze/aggregate.py` parses `runs/` into the tables and plots the paper needs. No manual transcription of numbers.
- `runs/` and `data/raw/` are gitignored. Configs, code, and `knowledge/ltr_priors.md` are committed.

## 14. Build order (phased, with definition-of-done)

**Phase 0 — Foundation smoke test.** `uv sync`; set key; run the circle_packing example for ~5 iterations to confirm SkyDiscover works end to end. Done when a run completes and writes a checkpoint.

**Phase 1 — Shared foundation.** Do this interactively in `notebooks/foundation.ipynb` first. It already runs end to end on synthetic data and validates the metrics, the group-aware custom objective, and the `evaluate(program_path)` contract. Then port the verified functions into `metrics.py`, `train_xgbranker.py`, `seed/initial_program.py`, and `eval/evaluator.py`, and write `prepare_expedia.py` + `schema.md`. Done when (a) the metric unit checks pass, (b) the custom LambdaMART objective reproduces built-in `rank:ndcg` within noise on the data (the correctness check), and (c) `evaluate("seed/initial_program.py")` returns the three metrics on the real Expedia fast subset. **Stop here and report numbers before going further.**

**Phase 2 — Baselines.** `lambdamart_optuna.py`, `lambdaloss.py`, `random_search.py`, all logging to `runs/`. Done when the tuned baseline and LambdaLoss numbers are recorded.

**Phase 3 — Evolutionary runs.** Both configs, a single short run each (10-15 iterations) to validate the loop, Pareto keys, and `artifacts` injection. Done when both runs complete and `aggregate.py` produces a frontier plot.

**Phase 4 — Ablation.** The full 3-seed matrix at 40 iterations, then `aggregate.py` tables/curves. Done when the memory-on vs memory-off comparison is plotted with seeds.

Do not start Phase N+1 until Phase N's definition-of-done is met. After each phase, print a short status summary and the key numbers.

## 15. Coding constraints

- Treat SkyDiscover as a black box; only touch `projects/evorank/`.
- Keep each candidate evaluation fast (<~60s); subsample aggressively in the loop, evaluate the final best programs on the full split.
- Deterministic everywhere (seed numpy, xgboost, the data sampler). Log seeds.
- Cost: the evolution loop makes one LLM call per iteration. Prefer a cheaper model for the loop than for any writing; expose the model as a config field so it is easy to switch. Estimate token cost before launching the full ablation.
- Small, typed, single-purpose functions. No notebook-style monoliths.
- In any prose you generate (README, docstrings, comments), use commas and periods. Do not use em dashes.
