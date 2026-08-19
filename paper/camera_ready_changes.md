# Camera-ready changes: EvoRank (GenAIECommerce 2026 @ RecSys 2026, submission 26)

Camera-ready due 28 August 2026 on EasyChair. Page allowance unchanged from
submission (CEUR short-paper band, held at 8 physical pages), so the revision
is page-neutral: every addition is offset by a cut elsewhere. Source:
`paper/latex/main.tex`; the submitted anonymized text is preserved in
`paper/latex/main_anon.tex` and `paper/submitted_2026-07-12.pdf` (local).

## Reviewer points → what changed

| Reviewer point | Action | Where |
|---|---|---|
| R2: terms such as "audit stack" and "headroom gate" force the reader back to earlier sections; suggest "auto-research loop audit/stopping criteria", "search-space headroom" | Section 3.4 retitled *The audit stack: acceptance and stopping criteria for the loop*; the headroom gate is now item (a) of the stack (it was only in the figure caption before); first body use in the Introduction reads "its audit stack (the acceptance and stopping criteria applied to every result), including the search-space headroom gate"; Campaign 2 cross-references the section; the lone synonym "honesty harness" (Fig. 1 caption) replaced by "audit stack". The abstract already glossed *selection fold*, *transfer audit*, *fitness noise*, and *search-space headroom*. | `main.tex` Intro contribution (4); §3.4 heading and item (a); Fig. 1 caption; Campaign 2 first paragraph |
| R1, R2: single, 13-year-old dataset; unclear whether the gate/audit generalize to other ranking problems | New paragraph **Limitations and generalization** at the end of Discussion: why the dataset (only public LTR set with conversion and revenue labels), what the audit stack requires (small fitness fold, larger held-out fold, paired bootstrap, one strong tuned baseline), and the gate as a three-step recipe for any query-grouped ranking or recommendation task | Discussion, last paragraph |
| R1: only two search spaces (no intermediate features+loss / model+loss) | Named as untested in the new paragraph, with the gate as the tool for ranking intermediate spaces before spending; experiments deferred | Same paragraph |
| R1: three runs limit statistical power on search variance | Stated explicitly: sufficient for the paired-bootstrap headline (which every run cleared), not for characterizing search variance | Same paragraph |
| R1: no comparison with other LLM-guided optimization frameworks | Stated as not done; the audit is engine-agnostic (EvoX arm as one swap) and a controlled comparison is future work; cites AlphaEvolve and GEPA already in the bibliography (no new references) | Same paragraph |
| Both: dataset age | Acknowledged ("now thirteen years old") | Same paragraph |

## Deferred to a future submission (not in camera-ready)

- Second dataset (MSLR-WEB30K transfer was designed, `paper/approach.md` §C, and shelved as a positive claim; campaign-2 pipelines never evaluated on it).
- Intermediate search spaces (features+loss, model+loss) with the headroom gate applied to each.
- More seeds per condition.
- Head-to-head with other LLM-guided evolution engines under the same audit.

## Other camera-ready edits

- De-anonymized: Rayhan Patel (University of Maryland, ORCID 0009-0004-4882-2123) and Shabaz Patel (Independent Researcher, ORCID 0000-0002-9617-7710, corresponding), equal contribution; repository URL restored in the title footnote and Conclusion.
- Bibliography: all 25 entries fact-checked against primary sources (arXiv, DOI, proceedings pages): authors, order, titles, venues, years, and identifiers confirmed. Edits: REA entry now names its authors (A. Kumar, E. Gao, M. Levi, et al.) and carries the blog URL and access date; GEPA upgraded to ICLR 2026 and ML-Agent to ICML 2026 (both confirmed); RankEvolve kept as arXiv (SIGIR 2026 acceptance appears only on an author page). Entries reordered so numbers appear in first-citation order ([1] to [25] ascending in the text); one hard-coded figure number replaced by a cross-reference.
- Offsetting cuts for page neutrality: Fig. 2 log excerpt drops the feature-stage line (caption now says "excerpted"), duplicate baseline sentence removed from §3.4, the Discussion's research-directions paragraph compressed, the probe-byproduct sentence and the "neither magic nor broken" sentence removed, provenance caveat tightened, minor phrasing trims in Related Work and Campaign 1.
- Numbers checked against `runs/summary/`: Table 1 = `pipeline_transfer.csv`; Table 2 = `fullscale_ndcg38.csv`; bootstrap CIs and p-values = `bootstrap_headline.csv`; Kaggle 0.5196 / 0.5140 and ranks 20 / 30 of 340 = `PIPELINE_CAMPAIGN.md` (official late-submission table). All match.
- Length: 8 physical pages before and after (tectonic build); word count 4321 to 4451 (+3%), the extra lines absorbed by the references page.

## Code review

Full read of `ltr/`, `eval/`, `analyze/`, `baselines/`, `discovered/`, `seed/`, CLI,
data prep, configs, and README against the paper's claims. Reported numbers
were **not** affected by anything found; fixes are guard-hardening and
documentation. Regression check after the fixes: `eval/evaluator_pipeline.py`
on `discovered/pipeline_s2_best.py` reproduces `runs/summary/pipeline_transfer.csv`
bit-exactly (val NDCG@10 = 0.4398464840271819).

### `ltr/`, `eval/`, `seed/`
- **Fixed (guard hardening)** `eval/evaluator_pipeline.py` `TrainStats.rate()`: in
  train mode it silently fell back to the in-sample encoding when called on a
  subset or reordered frame (the exact self-leak the OOF path prevents). Every
  discovered program calls it on the unmodified frame, so no reported number
  changes; it now raises a stage error instead of falling through.
- **Fixed** output-alignment check now requires the input index, not just the
  row count (a reordering `build_features` was previously scored misaligned).
- **Fixed** clean stage error when `data/prepared/feature_map.csv` is missing;
  `ltr/dataset.py` cache key now includes `seed`; `eval/evaluator.py` `__main__`
  creates `runs/` before writing its probe; dead `_per_query_tables` removed.
- **Documented** in the module docstring: `evaluator_pipeline.py` always uses the
  fast subset (`EVORANK_FAST` / `EVORANK_FEEDBACK` apply to campaign 1's
  `evaluator.py`); in-loop noise gate uses 300–400 bootstrap resamples, the
  audit 10,000. README gains an environment-switches note.
- **Paper wording corrected** to match code: §3.2 bootstrap "many times (10,000
  in the audit, a few hundred inside the loop's noise gate)" instead of
  "thousands"; §5 helper list now includes "a train-fold linear score over
  query-level columns" (`stats.stacked_score`, a Ridge fit on the train fold
  used by all three discovered pipelines; it is not out-of-fold, though it uses
  only query-level columns, and the transfer audit on held-out queries is
  unaffected).
- Left as is: campaign-2 wall budget skips later members rather than rejecting
  the candidate (the SkyDiscover-level `evaluator.timeout` still rejects
  runaway evaluations, which is what §3.3 describes); the rich-feedback arm's
  ±0.002 "within noise" label is below the measured fitness noise, an
  observation consistent with the paper's finding that this feedback changed
  nothing.

### `analyze/`, `baselines/`
- **Paper corrections (numbers/wording now match the CSVs)**: Table 1 bold moved
  to s0's test revenue (0.4257 is the column best; s2 still beats the baseline on
  all three objectives); Table 2 last row and its caption now describe
  `discovered/pipeline_s1_deepcfg.py` accurately (the s1 recipe with the
  baseline's full-scale tuned XGBoost settings on its XGBoost member, a scaled-up
  LightGBM member, and no third member; no search or tuning of its own), and the
  text no longer calls it "the same tuned settings" or "a lower bound";
  campaign 1's per-edit effect range now quotes the component-removal tests
  (0.0005 to 0.008, most below 0.005) against the measured NDCG@10 noise
  (+/-0.007 to 0.009); "two of three seeded runs" reach the random-search level
  in 1 to 3 iterations (mem_on_s0 never does, per `convergence_mem_on_s0.csv`);
  campaign 1 p-values marked two-sided and the floored 0.0002 reported as
  p < 0.001.
- **Traceability fixed**: `analyze/fullscale_ndcg38.py` now computes every Table 2
  row (added `lambdamart_optuna_full`, `pipeline_s1_deepcfg`, and a new
  `pipeline_s1_features_only` method for the "features alone reach 0.4587"
  sentence, previously only in a log) with `--methods` / `--out` flags. Rerun
  from scratch: `lambdamart_optuna_full` reproduces the CSV to 6 decimals;
  features-only 0.4587 / 0.5158 / 0.4849 / 0.4595 (now a CSV row);
  `pipeline_s1_deepcfg` gives 0.4621 / 0.5187 / 0.4889 / 0.4586 against the
  recorded 0.4619 / 0.5185 / 0.4883 / 0.4585 (LightGBM thread-level variation;
  the recorded row is kept because the bootstrap and Kaggle runs used that
  training).
- **Fixed** `discovered/pipeline_s1_deepcfg.py` carried the seed docstring; it now
  states its provenance. `baselines/retune_best.py` logged nothing on a crashing
  trial and scored it 0.0; it now prints the error and returns NaN.
  `analyze/pipeline_transfer.py`, `bootstrap_headline.py`, `fullscale_ndcg38.py`
  refuse to run on the synthetic data fallback (a fresh clone without prepared
  data would otherwise overwrite `runs/summary/*.csv` with synthetic numbers).
  Docstring drift fixed in `bootstrap_headline.py` (table numbers),
  `significance.py` (output file names, p-value floor), `mechanism_ablation.py`.
- Known and documented, not changed: `runs/summary/significance_*.csv` and
  `hypervolume.csv` predate the July 7 re-tune of `runs/lambdamart_optuna.json`
  on the 6k validation fold; regenerating them moves the baseline rows slightly
  and does not change any conclusion (the "p < 0.001 below the tuned baseline"
  statement is robust to it). `random_search.py` samples a parametric family
  rather than mutating the seed program; the paper's "blind mutation" phrase is
  kept as shorthand for "unguided sampling at equal budget".

### Repository (reproducibility surface)
- **Fixed** the paper's "code, configs, seeds, logs, discovered programs, and
  audit scripts are released" claim was not fully true: `.gitignore` excluded
  every input the analysis scripts read. Now tracked (about 1.2 MB total): every
  checkpoint's `best_program.py`, `runs/*.json` baseline records (tuned
  hyperparameters), the mechanism-ablation variants, the random-search control's
  programs, and the launch configs (`${ANTHROPIC_API_KEY}` placeholders only).
- **Fixed** README: reproduce-the-paper block (data acquisition, fold
  preparation with the exact fast-subset sizes, baselines, both campaigns,
  every analysis script), LLM named (`claude-sonnet-5`), SkyDiscover clone URL,
  environment switches, "honesty harness" heading renamed to "audit stack",
  the headline paragraph no longer says "on every objective at once" (s1's
  revenue is below the baseline), the full-scale table row is described
  accurately, and a Paper section with BibTeX.
- Verified: `discovered/pipeline_s{0,1,2}_best.py` are byte-identical to
  `checkpoint_50/best_program.py`; `objective_campaign_best.py` to mem_on_s1
  `checkpoint_40`; Fig. 3 excerpt matches `pipeline_s1_best.py` (its editorial
  ellipses and count comment are now labelled as such).
- Kaggle late-submission CSVs (86 MB each) are ignored rather than committed;
  regenerate with `analyze/kaggle_submission.py`; scores are recorded in
  `runs/summary/PIPELINE_CAMPAIGN.md`.
