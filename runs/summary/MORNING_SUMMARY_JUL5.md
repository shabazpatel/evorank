# EvoRank morning summary, July 5

Overnight: selection-generalization audit of all 12 evolution runs, plus the
feedback ablation extended to 3 seeds. Everything below is machine-generated
from runs/summary/generalization_audit.csv and run logs. Seed references
(fast-trained): fast_val ndcg 0.4212, full_test ndcg 0.4220.

## 1. The selection-generalization audit (the decisive table)

Final selected program per run, retrained on the search's own budget, scored
on the selection-side fold (fast val) and 60k unseen queries (full test).

| arm | val gain | test gain | gap | per-seed test gains |
|---|---|---|---|---|
| mem_off (old 1.6k fold) | +0.0011 | +0.0004 | +0.0006 | -5, +14, +4 bp |
| mem_on (old 1.6k fold) | +0.0004 | +0.0006 | -0.0002 | +9, +2, +6 bp |
| probe_scores (6k fold) | +0.0050 | +0.0014 | +0.0035 | +25, +4, +15 bp |
| probe_diag (6k fold) | +0.0046 | +0.0008 | +0.0039 | -11, +4, +29 bp |

Readings, stated honestly:
1. The July 3 runs (old noisy fold) selected essentially nothing that
   transfers: test gains of 0 to 14 basis points, mean ~5 bp.
2. De-noising the fitness fold (the only difference between mem_off and
   probe_scores) roughly tripled mean transferred gain (4 -> 14 bp) and
   produced the only meaningful single-run transfer (+25 bp). Still small in
   absolute terms, but directionally consistent with the noise mechanism we
   measured.
3. Diagnostic feedback showed NO advantage across 3 seeds on any endpoint:
   val convergence (0.4258 vs 0.4262 mean final), final best, or transfer
   (+8 vs +14 bp mean). Per-seed variance dwarfs the arm difference.

## 2. Consolidated verdicts (all evidence to date)

- Discovery claim: at T=40, no arm discovers objectives that beat the seed on
  held-out data by more than ~25 bp, and none approaches tuned LambdaMART
  (which leads by ~55 to 70 bp with p=0.0002 on the 60k-query fold).
- Guidance claims: priors speed val-fold convergence ~5x and win hypervolume
  on the selection fold in all seeds, but neither priors nor diagnostic
  evidence changes what actually transfers. The binding constraint is fitness
  signal quality (fold size) and the small headroom of the objective space on
  this dataset, not the feedback content.
- Regime claim: on the full 280k-query split all objectives, including
  default rank:ndcg (31s training), converge to ~0.440-0.441. Objective
  choice on this dataset matters only in the limited-data regime, and there
  the loop mostly harvests noise unless the fitness fold is large.
- Mechanism analysis remains the strongest positive artifact: gradient-level
  ablation attributed the discovered program's behavior and improved it by
  deletion (constant blend beats adaptive alpha on all three metrics).

## 3. Go / no-go decisions

- Full 2x2 (priors x feedback): NO-GO. The feedback axis is flat across 3
  seeds; adding priors-on diagnostic arms is very unlikely to change the
  conclusion. The existing 3-seeded arms already support the negative result.
- A1 phase-alternated HPO: NO-GO. Both post-hoc retunes gained at most +27 bp
  on the selection fold and 0 bp for the simplified program; hyperparameter
  co-adaptation cannot close a gap this size.
- B combination stage: NO-GO. Nothing real to combine.
- C MSLR zero-shot transfer: NO-GO as a positive claim (the objective does
  not beat its own seed out of domain-transfer would be theater).
- OPTIONAL (only remaining experiment with upside): one long-horizon run pair
  (T=100, de-noised fold, scores feedback, 2 seeds, ~$16) to test whether the
  loop finds transferable gains given a clean signal and more budget. Useful
  either way for the "what would it take" discussion section. Decision left
  to the author.

## 4. The paper this data supports

Title direction: what it takes to make LLM-driven objective discovery real,
an open, fully audited case study on e-commerce LTR.

1. Open, reproducible loop + honest evaluation protocol (fast-train /
   large-fold transfer audit, paired query bootstrap, hypervolume).
2. Measured failure mechanism: fitness noise exceeds per-edit effect sizes,
   the archive selects lottery tickets, val gains do not transfer; a data-side
   fix (6x val fold) triples transferred gain at zero cost.
3. Guidance ablations (3 seeds each): priors accelerate, evidence-rich
   diagnostics do not change outcomes; neither raises the transfer ceiling.
4. Gradient-level diagnosis toolkit (alignment probe, signal-usage probe,
   noise gating) plus the mechanism ablation that improved the discovered
   program by simplification; includes the 0.09 percent gradient mass vs
   0.009 metric effect calibration lesson.
5. Regime finding: objective choice washes out at full data scale on this
   benchmark; discovery loops operate exactly where noise is worst.

All claims currently in paper/approach.md section 4 are consistent with this
framing after tonight's update; MORNING_SUMMARY.md (July 3) remains as the
historical record of the interim readings.

## 5. Cost ledger to date
LLM: ~430 calls total (ablation 240, validation 24, probes 6x40=240 minus
rejections, smoke 3), claude-sonnet-5 intro pricing, cumulative estimate $30
to $45. Compute: everything else CPU-local.
