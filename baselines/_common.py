"""Shared helpers for EvoRank baselines and controls.

Every baseline logs the same three metrics on the same splits into ``runs/`` in
one schema, so ``analyze/aggregate.py`` can parse evolutionary runs and baselines
uniformly (CLAUDE.md Sections 11, 13).
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from ltr.metrics import metrics_bundle

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def score(frame: pd.DataFrame, preds: np.ndarray) -> dict:
    """The three competing objectives for a set of predictions on a fold."""
    return metrics_bundle(frame, preds)


def log_run(name: str, metrics: dict, extra: Optional[dict] = None) -> pathlib.Path:
    """Write one run record to ``runs/<name>.json`` and echo a one-line summary.

    Records only picklable/JSON-serializable content: the metric dict plus an
    optional ``extra`` payload (source, program path, tuned params, seed, etc.).
    """
    RUNS.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "metrics": metrics}
    if extra:
        record["extra"] = extra
    out = RUNS / f"{name}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    summary = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
    print(f"[{name}] {summary}  -> {out.relative_to(ROOT)}")
    return out