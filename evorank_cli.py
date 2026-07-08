"""evorank: REA-style autonomous pipeline discovery for Learning-to-Rank,
with a built-in honesty harness.

The workflow this CLI packages (each stage maps to one subcommand):

  gate     Headroom gate. Evaluates the seed pipeline and a hand-built
           reference candidate, reports the fitness noise floor, and tells
           you whether your search space has headroom worth LLM spend
           BEFORE any is spent.
  search   The discovery loop: LLM mutations over feature-construction code
           and a whitelisted pipeline spec (model x loss x ensemble), guided
           by stage-attributed feedback (per-member scores, ensemble margin,
           noise-gated metric deltas, segment decomposition). Resume-aware.
  audit    Transfer audit: rescores selected pipelines on a held-out fold so
           selection-fold noise cannot masquerade as discovery.
  report   Aggregates every run and baseline into one comparison table.

Data contract (bring your own preparation; adapters are optional): three
parquet folds, train/val/test, each with a qid column, numeric feature
columns, and label columns (default: graded rel, binary booking, revenue
rev; configurable in the evaluator for other objective sets). Rows sorted so
each qid is contiguous.

    uv run python evorank_cli.py gate
    uv run python evorank_cli.py search --seeds 0,1,2 --iterations 50
    uv run python evorank_cli.py audit
    uv run python evorank_cli.py report
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent


def _env() -> dict:
    env = dict(os.environ)
    envfile = REPO / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    env.setdefault("EVORANK_EVAL_TIMEOUT_S", "0")
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def cmd_gate(args) -> int:
    """Seed parity + headroom verdict, no LLM calls."""
    sys.path.insert(0, str(ROOT))
    from eval.evaluator_pipeline import evaluate  # noqa: E402

    print("== evorank gate: seed pipeline (parity reference) ==")
    seed = evaluate(str(ROOT / args.seed_program))
    print(seed["artifacts"]["feedback"], "\n")

    print(f"== evorank gate: reference candidate ({args.candidate}) ==")
    cand = evaluate(str(ROOT / args.candidate))
    print(cand["artifacts"]["feedback"], "\n")

    lift = cand["ndcg"] - seed["ndcg"]
    # noise floor is embedded in the feedback line; re-derive the verdict from lift
    verdict = "PASS" if lift >= args.min_lift else "NO-GO"
    print(f"gate verdict: {verdict} (candidate lift {lift:+.4f}, "
          f"required {args.min_lift:+.4f})")
    if verdict == "NO-GO":
        print("This search space does not show headroom above the noise floor "
              "on your fitness fold. Running the loop here is likely to select "
              "noise; enlarge the fitness fold, change the search space, or "
              "improve the reference candidate before spending LLM budget.")
    return 0 if verdict == "PASS" else 1


def cmd_search(args) -> int:
    """Run the discovery loop, one process per seed, resume-aware."""
    seeds = [s.strip() for s in args.seeds.split(",")]
    base_cfg = ROOT / args.config
    for seed in seeds:
        name = f"{args.name}_s{seed}"
        run_dir = ROOT / "runs" / name
        cfg = ROOT / "runs" / "_configs" / f"{name}.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(base_cfg.read_text().replace("random_seed: 0", f"random_seed: {seed}"))

        done = sorted(int(p.name.rsplit("_", 1)[-1])
                      for p in run_dir.glob("checkpoints/checkpoint_*")) or [0]
        remaining = args.iterations - done[-1]
        if remaining <= 0:
            print(f"{name}: complete ({done[-1]}/{args.iterations}), skipping")
            continue
        cmd = ["uv", "run", "python", "-m", "skydiscover.cli",
               str(ROOT / args.seed_program), str(ROOT / args.evaluator),
               "-c", str(cfg), "-s", "adaevolve", "-i", str(remaining),
               "-o", str(run_dir)]
        if done[-1] > 0:
            cmd += ["--checkpoint", str(run_dir / "checkpoints" / f"checkpoint_{done[-1]}")]
        print(f"{name}: running {remaining} iterations "
              f"({'resuming from ' + str(done[-1]) if done[-1] else 'fresh'})")
        rc = subprocess.run(cmd, cwd=REPO, env=_env()).returncode
        if rc != 0:
            print(f"{name}: exited {rc}", file=sys.stderr)
    return 0


def cmd_audit(args) -> int:
    """Transfer audit of selected pipelines on the held-out fold."""
    return subprocess.run(
        ["uv", "run", "python", str(ROOT / "analyze" / "pipeline_transfer.py")],
        cwd=ROOT, env=_env()).returncode


def cmd_report(args) -> int:
    return subprocess.run(
        ["uv", "run", "python", str(ROOT / "analyze" / "aggregate.py")],
        cwd=ROOT, env=_env()).returncode


def main() -> int:
    ap = argparse.ArgumentParser(prog="evorank", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="headroom gate before any LLM spend")
    g.add_argument("--seed-program", default="seed/initial_pipeline.py")
    g.add_argument("--candidate", default="eval/icdm_candidate.py",
                   help="hand-built known-good reference candidate")
    g.add_argument("--min-lift", type=float, default=0.010,
                   help="required ndcg lift over the seed (default 2.5x a "
                        "typical fast-fold noise floor)")
    g.set_defaults(fn=cmd_gate)

    s = sub.add_parser("search", help="run the discovery loop")
    s.add_argument("--seeds", default="0")
    s.add_argument("--iterations", type=int, default=50)
    s.add_argument("--name", default="pipeline")
    s.add_argument("--config", default="configs/evorank_pipeline.yaml")
    s.add_argument("--seed-program", default="seed/initial_pipeline.py")
    s.add_argument("--evaluator", default="eval/evaluator_pipeline.py")
    s.set_defaults(fn=cmd_search)

    a = sub.add_parser("audit", help="transfer audit of selections")
    a.set_defaults(fn=cmd_audit)

    r = sub.add_parser("report", help="aggregate comparison table")
    r.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
