"""Prepare the Expedia Personalized Hotel Search (ICDM 2013) training file.

Reads ``data/raw/train.csv`` (Kaggle ``expedia-personalized-sort``) and writes
prepared parquet splits with the downstream contract: ``qid``, ``f0..fK``,
``rel`` / ``booking`` / ``rev``.

Design decisions (see also schema.md):
  - Training file only. The Kaggle test labels are hidden, so we split the
    training file by srch_id into train/val/test (70/15/15), never splitting a
    query across folds.
  - Graded label: 5 if booked, else 1 if clicked, else 0.
  - Revenue is a label, never a feature. ``gross_bookings_usd`` is kept only on
    the label side (rev = gross_bookings_usd when booked, else 0).
  - Leakage columns dropped from X: position, gross_bookings_usd, click_bool,
    booking_bool.
  - Heavy-missing comp* columns get a missing indicator plus zero-fill.
  - A fast subset (~30k queries) is written for the evolution loop so each
    candidate evaluation stays fast; the full split is for final reporting.

Usage:
    python data/prepare_expedia.py --raw data/raw/train.csv --fast-queries 30000
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

LABEL_SIDE = {"gross_bookings_usd", "position", "click_bool", "booking_bool", "random_bool", "srch_id"}
# Non-feature identifiers we never rank on.
DROP_ALWAYS = {"srch_id", "date_time", "position", "gross_bookings_usd", "click_bool", "booking_bool"}


def _build_labels(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["rel"] = np.where(raw["booking_bool"] == 1, 5, np.where(raw["click_bool"] == 1, 1, 0))
    raw["booking"] = raw["booking_bool"].astype(int)
    raw["rev"] = np.where(raw["booking_bool"] == 1, raw.get("gross_bookings_usd", 0.0), 0.0)
    raw["rev"] = raw["rev"].fillna(0.0).astype(float)
    # Keep random_bool on the label side: random_bool == 1 marks the
    # position-unbiased impressions used for the optional debiasing analysis.
    if "random_bool" in raw.columns:
        raw["random_flag"] = raw["random_bool"].fillna(0).astype(int)
    else:
        raw["random_flag"] = 0
    return raw.rename(columns={"srch_id": "qid"})


def _select_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Numeric feature matrix with comp* missing-indicators; returns (X, names)."""
    candidate = [
        c for c in raw.columns
        if c not in DROP_ALWAYS
        and c not in {"qid", "rel", "booking", "rev", "random_flag", "random_bool"}
        and pd.api.types.is_numeric_dtype(raw[c])
    ]
    X = raw[candidate].copy()
    # comp* columns are heavily missing: add explicit missing indicators, then zero-fill.
    comp_cols = [c for c in candidate if c.startswith("comp")]
    for c in comp_cols:
        X[f"{c}_isna"] = X[c].isna().astype(int)
    X = X.fillna(0.0)
    names = list(X.columns)
    return X, names


def _split_by_qid(qids: np.ndarray, fracs=(0.70, 0.15), seed: int = 7):
    q = np.array(sorted(set(qids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(q)
    n = len(q)
    tr = set(q[: int(fracs[0] * n)])
    va = set(q[int(fracs[0] * n): int((fracs[0] + fracs[1]) * n)])
    te = set(q) - tr - va
    return tr, va, te


def _write_split(df: pd.DataFrame, out_dir: pathlib.Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values("qid").reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)


def _report(tag: str, frames: dict[str, pd.DataFrame]) -> None:
    print(f"[{tag}]")
    for name, f in frames.items():
        dist = f["rel"].value_counts().to_dict()
        print(f"  {name:5s}  queries {f['qid'].nunique():>7}  rows {len(f):>9}  rel_dist {dist}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "train.csv"))
    ap.add_argument("--fast-queries", type=int, default=30000,
                    help="number of training-file queries to keep for the fast subset")
    ap.add_argument("--fast-eval-queries", type=int, default=None,
                    help="val/test query cap for the fast subset (default fast-queries // 5). "
                         "Larger val folds cut fitness noise at almost no evaluation cost, "
                         "since candidate evaluation is dominated by training time.")
    ap.add_argument("--fast-only", action="store_true",
                    help="derive the fast subset from existing data/prepared parquet and "
                         "leave data/prepared untouched (no raw csv read)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_full = ROOT / "data" / "prepared"
    if args.fast_only:
        if not (out_full / "train.parquet").exists():
            raise SystemExit("--fast-only requires an existing data/prepared split")
        print("loading existing full split from data/prepared ...")
        full = {name: pd.read_parquet(out_full / f"{name}.parquet")
                for name in ("train", "val", "test")}
    else:
        raw_path = pathlib.Path(args.raw)
        if not raw_path.exists():
            raise SystemExit(f"raw file not found: {raw_path}. Place the Kaggle train.csv there.")

        print(f"reading {raw_path} ...")
        raw = pd.read_csv(raw_path)
        raw = _build_labels(raw)
        X, feat_names = _select_features(raw)

        df = pd.concat(
            [raw[["qid", "rel", "booking", "rev", "random_flag"]].reset_index(drop=True),
             X.reset_index(drop=True)],
            axis=1,
        )
        # rename features to the f0..fK contract
        rename = {c: f"f{i}" for i, c in enumerate(feat_names)}
        df = df.rename(columns=rename)
        fcols = [f"f{i}" for i in range(len(feat_names))]
        df = df[["qid", *fcols, "rel", "booking", "rev", "random_flag"]]
        print(f"feature count: {len(fcols)}")

        # full split
        tr_q, va_q, te_q = _split_by_qid(df["qid"].to_numpy(), seed=args.seed)
        full = {
            "train": df[df.qid.isin(tr_q)],
            "val": df[df.qid.isin(va_q)],
            "test": df[df.qid.isin(te_q)],
        }
        assert not (set(full["train"].qid) & set(full["val"].qid) & set(full["test"].qid)), "query spans folds"
        for name, f in full.items():
            _write_split(f, out_full, name)
        _report("full", full)

        # save the feature-name mapping for traceability
        (out_full / "feature_map.csv").write_text(
            "f_index,original\n" + "\n".join(f"f{i},{name}" for i, name in enumerate(feat_names)) + "\n"
        )

    # fast subset: cap the number of training queries; keep val/test smaller too
    eval_cap = args.fast_eval_queries or max(1, args.fast_queries // 5)
    rng = np.random.default_rng(args.seed)
    def cap(frame, n):
        qs = frame["qid"].unique()
        if len(qs) > n:
            keep = set(rng.choice(qs, size=n, replace=False))
            return frame[frame.qid.isin(keep)]
        return frame
    fast = {
        "train": cap(full["train"], args.fast_queries),
        "val": cap(full["val"], eval_cap),
        "test": cap(full["test"], eval_cap),
    }
    out_fast = ROOT / "data" / "prepared_fast"
    for name, f in fast.items():
        _write_split(f, out_fast, name)
    _report("fast", fast)
    print(f"wrote {out_fast}" + ("" if args.fast_only else f" and {out_full}"))


if __name__ == "__main__":
    main()
