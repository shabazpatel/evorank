"""SkyDiscover evaluator for EvoRank.

Implements the ``evaluate(program_path) -> dict`` contract (see repo-root
CLAUDE.md Section 3): import a candidate program, train an XGBoost ranker with
its custom objective under the fixed config, and score the three competing
objectives on the validation fold.

Guardrails (the loss-rejection analog):
  - Per-candidate timeout (``EVORANK_EVAL_TIMEOUT_S``, default 60s). A candidate
    that exceeds it is rejected with all-zero metrics and a feedback string.
  - Non-finite grad/hess or non-finite metrics are rejected, never scored well.
  - Deterministic: fixed seed and fixed data subset per run.

The ``artifacts.feedback`` string is injected into the next LLM prompt. Set
``EVORANK_FAST=0`` to evaluate on the full split (final reporting) instead of the
fast subset (the loop). Ported from notebooks/foundation.ipynb (Section E).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import signal
import sys
import threading
import uuid

import numpy as np

# Make the evorank package root importable no matter how SkyDiscover loads this
# file (it imports the evaluator by path, so the cwd is not evorank/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ltr.dataset import FIXED_PARAMS, FIXED_ROUNDS, get_dataset, to_dmatrix
from ltr.metrics import group_sizes_of, group_slices, metrics_bundle

import xgboost as xgb


def _timeout_s() -> float:
    return float(os.environ.get("EVORANK_EVAL_TIMEOUT_S", "60"))


def _use_fast() -> bool:
    return os.environ.get("EVORANK_FAST", "1") != "0"


class _Timeout(Exception):
    pass


def _run_with_timeout(fn, seconds: float):
    """Run ``fn`` under a wall-clock timeout using SIGALRM. Falls back to running
    without a timeout when not on the main thread (SIGALRM is main-thread only)."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return fn()

    def _handler(signum, frame):
        raise _Timeout(f"evaluation exceeded {seconds:.0f}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _load_program(program_path: str):
    """Import a candidate program from a file path under a unique module name."""
    spec = importlib.util.spec_from_file_location(f"candidate_{uuid.uuid4().hex}", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Seed program reference metrics on the real fast subset (measured Phase 1).
# Used only for the rich feedback deltas in the memory-on condition.
_SEED_REF = {"ndcg": 0.4179, "book_ndcg": 0.4394, "revenue": 0.3888}


def _feedback_mode() -> str:
    """Feedback treatment arm, controlled by EVORANK_FEEDBACK (default minimal):
      minimal     scores only (control arm)
      rich        scores + deltas vs seed + one-line reading (legacy memory-on arm)
      diagnostic  full evidence block from eval/diagnostics.py: noise-gated
                  deltas, segment decomposition, behavior, gradient alignment
                  probe, and per-signal gradient usage
    """
    return os.environ.get("EVORANK_FEEDBACK", "minimal")


def _one_line_diagnostic(m: dict) -> str:
    scores = f"ndcg={m['ndcg']:.4f} book_ndcg={m['book_ndcg']:.4f} revenue={m['revenue']:.4f}"
    if _feedback_mode() != "rich":
        return scores
    deltas = {k: m[k] - _SEED_REF[k] for k in _SEED_REF}
    delta_str = " ".join(f"d_{k}={v:+.4f}" for k, v in deltas.items())
    worst = min(deltas, key=deltas.get)
    best = max(deltas, key=deltas.get)
    if all(abs(v) < 0.002 for v in deltas.values()):
        note = "within noise of the seed objective"
    elif deltas[worst] < -0.002 and deltas[best] > 0.002:
        note = f"trades {worst} away for {best}; check whether the frontier gains"
    elif deltas[worst] < -0.002:
        note = f"{worst} regressed vs seed; objective may over-weight other signals"
    else:
        note = f"improves on seed, strongest on {best}"
    return f"{scores} | vs_seed: {delta_str} | {note}"


def _train_and_score(mod) -> dict:
    train_df, val_df, _test_df, feats, _src = get_dataset(fast=_use_fast())
    slices = group_slices(group_sizes_of(train_df))
    # Behavioral signals aligned to DMatrix/predt row order (train_df is not
    # reordered when building the DMatrix), so the objective can weight relevance,
    # conversion, and revenue however the search discovers.
    rel = train_df["rel"].to_numpy(dtype=np.float64)
    booking = train_df["booking"].to_numpy(dtype=np.float64)
    rev = train_df["rev"].to_numpy(dtype=np.float64)
    dtr = to_dmatrix(train_df, feats)
    dva = to_dmatrix(val_df, feats)

    def obj(predt, dtrain):
        g, h = mod.lambdamart_objective(predt, rel, booking, rev, slices)
        g = np.asarray(g, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        if g.shape != predt.shape or h.shape != predt.shape:
            raise ValueError("grad/hess shape mismatch")
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(h))):
            raise ValueError("non-finite grad/hess")
        return g, h

    bst = xgb.train(FIXED_PARAMS, dtr, num_boost_round=FIXED_ROUNDS, obj=obj)
    val_preds = bst.predict(dva)
    m = metrics_bundle(val_df, val_preds)
    if not all(np.isfinite(v) for v in m.values()):
        raise ValueError("non-finite metric")

    feedback = None
    if _feedback_mode() == "diagnostic":
        # Evidence block; any diagnostics failure degrades to scores-only
        # feedback rather than failing the evaluation.
        try:
            from eval.diagnostics import build_diagnostics
            feedback = build_diagnostics(mod, train_df, val_df, feats, bst,
                                         val_preds, dtr)
        except Exception as ex:  # pragma: no cover
            feedback = None
            print(f"[evaluator] diagnostics failed, falling back to scores: "
                  f"{type(ex).__name__}: {ex}")
    return m, feedback


def evaluate(program_path: str) -> dict:
    """Train the candidate objective and return the three metrics plus
    ``combined_score`` and an ``artifacts.feedback`` string. Any failure
    (import error, non-finite grad/hess/metric, timeout) returns all-zeros with a
    feedback string, so a degenerate objective can never score well."""
    try:
        mod = _load_program(program_path)
        if not hasattr(mod, "lambdamart_objective"):
            raise AttributeError("program has no lambdamart_objective")
        m, feedback = _run_with_timeout(lambda: _train_and_score(mod), _timeout_s())
        if feedback is None:
            feedback = _one_line_diagnostic(m)
        return {**m, "combined_score": m["ndcg"],
                "artifacts": {"feedback": feedback}}
    except Exception as ex:
        return {"ndcg": 0.0, "book_ndcg": 0.0, "revenue": 0.0, "combined_score": 0.0,
                "artifacts": {"feedback": f"rejected: {type(ex).__name__}: {ex}"}}


if __name__ == "__main__":
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    seed_prog = root / "seed" / "initial_program.py"
    print("seed program :", evaluate(str(seed_prog)))

    # degenerate candidate (NaN gradient) must be rejected, not scored
    broken = root / "runs" / "_broken_probe.py"
    broken.write_text(
        "import numpy as np\n"
        "def lambdamart_objective(predt, rel, booking, rev, group_slices, sigma=1.0):\n"
        "    return np.full_like(predt, np.nan), np.ones_like(predt)\n"
    )
    print("broken program:", evaluate(str(broken)))
    broken.unlink(missing_ok=True)