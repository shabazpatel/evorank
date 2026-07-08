"""Baseline: random objective mutations at equal generation budget (REQUIRED).

The control that shows the LLM guidance matters: draws random candidates from
the same objective family the evolution explores (blend weights over rel,
booking, and revenue, a revenue transform, discount shape, sigma), writes each
as a candidate program file, and scores it through the exact same
eval.evaluator.evaluate() guardrails as the loop. Equal budget, no LLM.

    python baselines/random_search.py --trials 40 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines._common import RUNS, log_run  # noqa: E402

CANDIDATE_TEMPLATE = '''import numpy as np

# EVOLVE-BLOCK-START: objective
def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma={sigma}):
    grad = np.zeros_like(predt, dtype=np.float64)
    hess = np.zeros_like(predt, dtype=np.float64)
    for s, e in group_slices:
        sc = predt[s:e].astype(np.float64); r = rel[s:e]; b = booking[s:e]; v = rev[s:e]
        m = len(sc)
        if m < 2:
            continue
        rev_term = {rev_expr}
        gain = {w_rel} * (2.0 ** r - 1.0) + {w_book} * b + {w_rev} * rev_term
        S = sc[:, None] - sc[None, :]
        rho = 1.0 / (1.0 + np.exp(sigma * S))
        M = (gain[:, None] > gain[None, :])
        ranks = np.empty(m, dtype=int); ranks[np.argsort(-sc, kind="stable")] = np.arange(m)
        disc = 1.0 / np.log2(ranks + 2.0) ** {disc_pow}
        ideal = np.sort(gain)[::-1]
        idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, m + 2)) ** {disc_pow}))) or 1.0
        dZ = np.abs((gain[:, None] - gain[None, :]) * (disc[:, None] - disc[None, :])) / idcg
        lam = np.where(M, -sigma * rho * dZ, 0.0)
        hmat = np.where(M, (sigma ** 2) * rho * (1.0 - rho) * dZ, 0.0)
        grad[s:e] = lam.sum(axis=1) - lam.sum(axis=0)
        hess[s:e] = hmat.sum(axis=1) + hmat.sum(axis=0)
    return grad, np.maximum(hess, 1e-6)
# EVOLVE-BLOCK-END: objective
'''

REV_EXPRS = [
    "np.log1p(v)",
    "np.sqrt(v)",
    "np.minimum(v, 2500.0) / 2500.0",
    "v / (np.max(v) + 1e-9)",
]


def draw_candidate(rng: np.random.Generator) -> dict:
    return {
        "w_rel": round(float(rng.uniform(0.0, 2.0)), 3),
        "w_book": round(float(rng.uniform(0.0, 20.0)), 3),
        "w_rev": round(float(rng.uniform(0.0, 5.0)), 3),
        "rev_expr": REV_EXPRS[int(rng.integers(len(REV_EXPRS)))],
        "sigma": round(float(rng.uniform(0.5, 2.0)), 3),
        "disc_pow": round(float(rng.uniform(0.5, 2.0)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40, help="same budget as the evolution runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("EVORANK_EVAL_TIMEOUT_S", "0")  # SIGALRM-free; candidates are bounded
    from eval.evaluator import evaluate  # noqa: E402  (import after env setup)

    rng = np.random.default_rng(args.seed)
    work = RUNS / f"random_search_s{args.seed}"
    work.mkdir(parents=True, exist_ok=True)

    history = []
    best = None
    for i in range(args.trials):
        params = draw_candidate(rng)
        src = CANDIDATE_TEMPLATE.format(**params)
        cand = work / f"candidate_{i:03d}.py"
        cand.write_text(src)
        res = evaluate(str(cand))
        rec = {"iteration": i, "params": params,
               "ndcg": res["ndcg"], "book_ndcg": res["book_ndcg"], "revenue": res["revenue"],
               "feedback": res["artifacts"]["feedback"]}
        history.append(rec)
        if best is None or res["ndcg"] > best["ndcg"]:
            best = {**rec}
            (work / "best_program.py").write_text(src)
        print(f"[random {i + 1:>3}/{args.trials}] ndcg={res['ndcg']:.4f} "
              f"best={best['ndcg']:.4f} ({res['artifacts']['feedback'][:60]})")

    (work / "history.json").write_text(json.dumps(history, indent=2))
    log_run(f"random_search_s{args.seed}",
            {"ndcg": best["ndcg"], "book_ndcg": best["book_ndcg"], "revenue": best["revenue"]},
            extra={"trials": args.trials, "seed": args.seed,
                   "best_iteration": best["iteration"], "best_params": best["params"],
                   "history_file": str(work / "history.json")})


if __name__ == "__main__":
    main()
