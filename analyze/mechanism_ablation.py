"""Mechanism ablation of the best discovered objective (paper figure 1).

Takes the discovered program and knocks out one mechanism at a time by source
transformation, then scores every variant through the exact same evaluate()
guardrails as the search. Attributes metric contributions to mechanisms.

Variants:
  full          the discovered program, unchanged (reference)
  no_revenue    remove the clipped-log revenue term from the gain
  no_listwise   alpha = 1.0 everywhere (pure pairwise |deltaNDCG|)
  no_adaptive   constant alpha = 0.70 (blend kept, per-query adaptivity removed)
  raw_revenue   linear rev_c / 3000 instead of log1p (tests the transform)
  seed          the original seed objective (secondary reference)

    uv run python analyze/mechanism_ablation.py \
        --program runs/mem_on_s1/checkpoints/checkpoint_40/best_program.py
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REV_TERM = " + 3.5 * (np.log1p(rev_c) / np.log1p(3000.0))"
REV_TERM_RAW = " + 3.5 * (rev_c / 3000.0)"
ALPHA_ADAPTIVE = "alpha = 0.55 if has_booking else 0.85"

VARIANTS = {
    "full": lambda src: src,
    "no_revenue": lambda src: src.replace(REV_TERM, ""),
    "no_listwise": lambda src: src.replace(ALPHA_ADAPTIVE, "alpha = 1.0"),
    "no_adaptive": lambda src: src.replace(ALPHA_ADAPTIVE, "alpha = 0.70"),
    "raw_revenue": lambda src: src.replace(REV_TERM, REV_TERM_RAW),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program",
                    default="runs/mem_on_s1/checkpoints/checkpoint_40/best_program.py")
    args = ap.parse_args()

    os.environ.setdefault("EVORANK_EVAL_TIMEOUT_S", "0")
    from eval.evaluator import evaluate  # noqa: E402

    src = (ROOT / args.program).read_text()
    work = ROOT / "runs" / "summary" / "mechanism_ablation"
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, transform in VARIANTS.items():
        variant_src = transform(src)
        if name != "full" and variant_src == src:
            print(f"WARNING: variant {name} did not change the source, "
                  "the anchor string may have drifted; skipping")
            continue
        path = work / f"{name}.py"
        path.write_text(variant_src)
        res = evaluate(str(path))
        rows.append({"variant": name, "ndcg": res["ndcg"],
                     "book_ndcg": res["book_ndcg"], "revenue": res["revenue"],
                     "feedback": res["artifacts"]["feedback"]})
        print(f"{name:<12} ndcg={res['ndcg']:.4f} book={res['book_ndcg']:.4f} "
              f"rev={res['revenue']:.4f}")

    seed_res = evaluate(str(ROOT / "seed" / "initial_program.py"))
    rows.append({"variant": "seed", "ndcg": seed_res["ndcg"],
                 "book_ndcg": seed_res["book_ndcg"], "revenue": seed_res["revenue"],
                 "feedback": seed_res["artifacts"]["feedback"]})
    print(f"{'seed':<12} ndcg={seed_res['ndcg']:.4f} book={seed_res['book_ndcg']:.4f} "
          f"rev={seed_res['revenue']:.4f}")

    out = ROOT / "runs" / "summary" / "mechanism_ablation.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "ndcg", "book_ndcg", "revenue", "feedback"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
