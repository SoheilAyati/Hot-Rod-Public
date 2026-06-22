#!/usr/bin/env python
"""
shadow_compare.py — evidence-based A/B between two forecast strategies.

This is the "shadow test" discipline: you run one model for real and a
candidate on the side, and each day after the actuals settle you ask which
*would* have scored better. After enough days you have a live, paired record to
set beside a backtest — so "should we switch?" is a number, not a feeling.

Give it two folders of daily submission CSVs (``A`` = current, ``B`` =
candidate) and a source of actuals:

    python shadow_compare.py --a submissions/current --b submissions/candidate \
        --actuals actuals/

It prints a per-day table, a running tally, and the adoption verdict: the paired
mean improvement with a moving-block-bootstrap 95% CI, the win rate, and a tail
check (worst-10 day mean and worst rolling-7-day window). The gate Hot Rod uses
to promote a model is encoded in :func:`adoption_verdict` below — a candidate
ships only if it lowers the mean *and* holds the tail.

Offline when given actuals; no API key needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _common as C
from rolling_mae import collect_daily_mae, load_actuals


def adoption_verdict(daily_a: pd.Series, daily_b: pd.Series,
                     block: int = 7, worst_n: int = 10) -> dict:
    """Decide whether candidate B should replace incumbent A.

    The rule (Hot Rod's tail-gated adoption gate):

      1. B lowers the mean daily MAE, and
      2. the paired improvement (A − B) has a moving-block-bootstrap 95% CI
         whose lower bound is above zero (robust, not noise), and
      3. B does not regress the tail: neither the worst-``worst_n``-day mean nor
         the worst rolling-7-day window is worse than A's.

    The mean alone is never sufficient: the score is tail-dominated, and a
    change that trims the average while fattening the worst week loses ranking.
    """
    paired = pd.concat([daily_a.rename("a"), daily_b.rename("b")], axis=1).dropna()
    if paired.empty:
        return {"verdict": "NO DATA", "reasons": ["no overlapping days"]}
    diff = (paired["a"] - paired["b"]).to_numpy()  # >0 means B better
    mean, lo, hi = C.moving_block_bootstrap_ci(diff, block=block)
    ta, tb = C.tail_stats(paired["a"]), C.tail_stats(paired["b"])

    mean_better = tb["mean"] < ta["mean"]
    ci_excludes_zero = lo > 0
    tail_holds = (tb[f"worst_{worst_n}_mean"] <= ta[f"worst_{worst_n}_mean"] + 1e-6
                  and tb["worst_roll7"] <= ta["worst_roll7"] + 1e-6)

    reasons = []
    reasons.append(("mean lower" if mean_better else "mean NOT lower")
                   + f" ({tb['mean']:.1f} vs {ta['mean']:.1f})")
    reasons.append((f"95% CI [{lo:+.1f}, {hi:+.1f}] excludes 0" if ci_excludes_zero
                    else f"95% CI [{lo:+.1f}, {hi:+.1f}] spans 0"))
    reasons.append(("tail holds" if tail_holds else "tail REGRESSES")
                   + f" (worst-{worst_n} {tb[f'worst_{worst_n}_mean']:.0f} vs "
                   f"{ta[f'worst_{worst_n}_mean']:.0f}; "
                   f"worst-roll7 {tb['worst_roll7']:.0f} vs {ta['worst_roll7']:.0f})")

    passed = mean_better and ci_excludes_zero and tail_holds
    return {
        "verdict": "ADOPT B" if passed else "KEEP A",
        "n_days": int(len(paired)),
        "mean_improvement": float(mean), "ci_lo": float(lo), "ci_hi": float(hi),
        "win_rate": float((diff > 0).mean()),
        "reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="folder of incumbent CSVs (A)")
    ap.add_argument("--b", required=True, help="folder of candidate CSVs (B)")
    ap.add_argument("--actuals", required=True, help="actuals folder or combined CSV")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--block", type=int, default=7, help="bootstrap block length")
    args = ap.parse_args()

    actual = load_actuals(args.actuals)
    da = collect_daily_mae(args.a, actual)
    db = collect_daily_mae(args.b, actual)
    paired = pd.concat([da.rename("a"), db.rename("b")], axis=1).dropna()
    if paired.empty:
        print("[fatal] no overlapping scorable days between A and B.")
        return 1

    la, lb = args.label_a, args.label_b
    print("=" * 60)
    print(f"shadow A/B   A={la}  B={lb}")
    print("=" * 60)
    print(f"  {'date':<12}{la:>10}{lb:>10}{'edge(A-B)':>12}  winner")
    print("  " + "-" * 46)
    for d, row in paired.iterrows():
        edge = row["a"] - row["b"]
        win = lb if edge > 0 else la
        print(f"  {d.date()!s:<12}{row['a']:10.1f}{row['b']:10.1f}"
              f"{edge:+12.1f}  {win}")

    v = adoption_verdict(da, db, block=args.block)
    print("  " + "-" * 46)
    print(f"  days paired        : {v['n_days']}")
    print(f"  mean {la:>3} / {lb:<3}    : {paired['a'].mean():.1f} / {paired['b'].mean():.1f} MW")
    print(f"  mean improvement   : {v['mean_improvement']:+.1f} MW "
          f"(95% CI [{v['ci_lo']:+.1f}, {v['ci_hi']:+.1f}])")
    print(f"  {lb} win rate        : {v['win_rate']*100:.0f}%")
    print("\n  gate:")
    for r in v["reasons"]:
        print(f"    - {r}")
    print(f"\n  VERDICT: {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
