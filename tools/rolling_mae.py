#!/usr/bin/env python
"""
rolling_mae.py — reproduce the leaderboard's rolling 7-day mean MAE.

Point it at a folder of daily submission CSVs (named ``YYYY-MM-DD.csv``) and a
folder — or single file — of actuals, and it computes, for every day:

  * that day's MAE (mean abs error over its 24 hours), and
  * the rolling 7-day mean of that daily MAE — the number the leaderboard ranks
    you on (lowest wins).

    python rolling_mae.py --forecasts submissions/hot_rod --actuals actuals/

``--actuals`` may be a directory of ``YYYY-MM-DD.csv`` actual files or one long
CSV covering the whole range. The rolling window (``--window``, default 7) is
defined once a full window of days is available; pass ``--min-periods 1`` to see
a partial roll from day one.

Offline by default — give it actuals and it never touches the network.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import _common as C


def load_actuals(path: str) -> pd.Series:
    """Load actuals from either a directory of daily CSVs or one combined CSV."""
    p = Path(path)
    if p.is_dir():
        parts = [C.read_forecast(f) for f in sorted(p.glob("*.csv"))]
        if not parts:
            raise SystemExit(f"[fatal] no CSVs in actuals dir {p}")
        s = pd.concat(parts)
    else:
        s = C.read_forecast(p)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.rename("actual")


def collect_daily_mae(forecast_dir: str, actual: pd.Series) -> pd.Series:
    rows = {}
    for f in sorted(Path(forecast_dir).glob("*.csv")):
        try:
            fc = C.read_forecast(f)
        except ValueError as exc:
            print(f"[skip] {f.name}: {exc}")
            continue
        day = fc.index[0].tz_convert("UTC").normalize()
        want = C.expected_hours(day)
        m = C.mae(fc.reindex(want), actual.reindex(want))
        if m == m:  # not NaN
            rows[pd.Timestamp(day.date())] = m
    return pd.Series(rows, name="daily_mae").sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", required=True,
                    help="folder of daily submission CSVs (YYYY-MM-DD.csv)")
    ap.add_argument("--actuals", required=True,
                    help="folder of actual CSVs or one combined actuals CSV")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--min-periods", type=int, default=None,
                    help="min days before the roll is defined (default = window)")
    args = ap.parse_args()

    actual = load_actuals(args.actuals)
    daily = collect_daily_mae(args.forecasts, actual)
    if daily.empty:
        print("[fatal] no scorable forecast days found.")
        return 1
    roll = C.rolling_7day_mae(daily, args.window, args.min_periods)
    stats = C.tail_stats(daily)

    print("=" * 60)
    print(f"rolling {args.window}-day mean MAE - {args.forecasts}")
    print("=" * 60)
    print(f"  {'date':<12}{'daily MAE':>12}{'roll-' + str(args.window):>12}")
    print("  " + "-" * 34)
    for d, v in daily.items():
        r = roll.get(d, float('nan'))
        rtxt = "  -" if r != r else f"{r:12.1f}"
        print(f"  {d.date()!s:<12}{v:12.1f}{rtxt}")
    print("  " + "-" * 34)
    print(f"  days scored        : {stats['n_days']}")
    print(f"  mean daily MAE     : {stats['mean']:.1f} MW")
    print(f"  median daily MAE   : {stats['median']:.1f} MW")
    print(f"  worst-10 mean      : {stats['worst_10_mean']:.1f} MW")
    print(f"  worst single day   : {stats['worst_day']:.1f} MW")
    print(f"  worst rolling-{args.window}d   : {stats['worst_roll7']:.1f} MW")
    last = roll.dropna()
    if len(last):
        print(f"\n  latest rolling-{args.window}d score : {last.iloc[-1]:.1f} MW "
              f"(as of {last.index[-1].date()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
