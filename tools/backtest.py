#!/usr/bin/env python
"""
backtest.py — rolling-origin backtest harness with a tail-gated verdict.

A small, model-agnostic framework for the evaluation discipline that decides
what Hot Rod ships. It walks a range of target days; for each day it hands your
forecaster the load history truncated at the publication cutoff, collects the
24-hour forecast, and scores it against the actual. The output is the full
tail-aware summary — mean, median, worst-10-day mean, worst day, worst
rolling-7-day window — and, in compare mode, a paired moving-block-bootstrap
confidence interval plus a PASS/FAIL adoption gate.

Why a harness like this matters: short windows mislead. A change can look great
over a fortnight and quietly fatten the worst week over a full year. Backtesting
over a long, live-replica range with a tail gate is the part of the method that
transfers to any team, independent of the model inside.

The forecaster interface (the only thing you implement)
-------------------------------------------------------
A plain callable, referenced as ``module:function``::

    def forecast(history: pd.Series, target_day: pd.Timestamp) -> pd.Series:
        # history : hourly UTC load up to the publication cutoff (settled data)
        # returns : 24 hourly forecasts for target_day (UTC), MW

Two ready-made ones live in ``baselines.py``.

Examples
--------
    # backtest the seasonal-naive baseline over a year of actuals
    python backtest.py --forecaster baselines:seasonal_naive \
        --actuals actuals_2024.csv --start 2024-07-01 --end 2025-06-30

    # A/B two forecasters with the adoption gate
    python backtest.py --forecaster mymodel:predict \
        --vs baselines:seasonal_naive --actuals actuals_2024.csv

``--actuals`` is both the history the forecaster sees and the truth it is scored
against (a directory of daily CSVs or one long CSV). Offline; no API key.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

import _common as C
from rolling_mae import load_actuals
from shadow_compare import adoption_verdict

Forecaster = Callable[[pd.Series, pd.Timestamp], pd.Series]


def load_forecaster(spec: str) -> Forecaster:
    """Resolve a ``module:function`` string to a callable, searching the tools
    folder and the current directory so user modules are found."""
    if ":" not in spec:
        raise SystemExit(f"[fatal] forecaster '{spec}' must be module:function")
    mod_name, fn_name = spec.split(":", 1)
    for extra in (str(Path(__file__).parent), str(Path.cwd())):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"[fatal] could not load forecaster '{spec}': {exc}")


def run_backtest(forecaster: Forecaster, actual: pd.Series,
                 start: pd.Timestamp | None, end: pd.Timestamp | None,
                 pub_lag_h: int = 24) -> pd.Series:
    """Return a per-day MAE Series for ``forecaster`` over the target range.

    For each day D the forecaster receives history up to
    ``D 00:00 UTC − pub_lag_h`` (the publication cutoff: data settled by submit
    time) and must return 24 hourly values, which are scored against the actual.
    """
    actual = actual.sort_index()
    days = pd.DatetimeIndex(sorted({ts.normalize() for ts in actual.index}))
    if start is not None:
        days = days[days >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        days = days[days <= pd.Timestamp(end, tz="UTC")]

    out = {}
    for day in days:
        cutoff = day - pd.Timedelta(hours=pub_lag_h)
        history = actual.loc[:cutoff].dropna()
        if history.empty:
            continue
        try:
            fc = forecaster(history, day)
        except Exception as exc:  # noqa: BLE001 - one bad day shouldn't kill the run
            print(f"[skip] {day.date()}: forecaster raised {exc!r}")
            continue
        want = C.expected_hours(day)
        m = C.mae(pd.Series(fc).reindex(want), actual.reindex(want))
        if m == m:  # not NaN
            out[pd.Timestamp(day.date())] = m
    return pd.Series(out, name="daily_mae").sort_index()


def _print_stats(label: str, daily: pd.Series) -> None:
    s = C.tail_stats(daily)
    print(f"  {label}")
    print(f"    days={s['n_days']}  mean={s['mean']:.1f}  median={s['median']:.1f}  "
          f"worst10={s['worst_10_mean']:.1f}  worstday={s['worst_day']:.1f}  "
          f"worst-roll7={s['worst_roll7']:.1f}  (MW)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecaster", required=True, help="module:function")
    ap.add_argument("--vs", default=None, help="second forecaster -> compare mode")
    ap.add_argument("--actuals", required=True, help="actuals dir or combined CSV")
    ap.add_argument("--start", default=None, help="first target day YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="last target day YYYY-MM-DD")
    ap.add_argument("--pub-lag", type=int, default=24,
                    help="publication lag in hours (history cutoff before D 00:00)")
    ap.add_argument("--block", type=int, default=7, help="bootstrap block length")
    args = ap.parse_args()

    actual = load_actuals(args.actuals)
    f1 = load_forecaster(args.forecaster)
    d1 = run_backtest(f1, actual, args.start, args.end, args.pub_lag)
    if d1.empty:
        print("[fatal] no scorable days - check the date range and actuals.")
        return 1

    print("=" * 64)
    print(f"backtest: {args.forecaster}")
    print("=" * 64)
    _print_stats(args.forecaster, d1)

    if args.vs:
        f2 = load_forecaster(args.vs)
        d2 = run_backtest(f2, actual, args.start, args.end, args.pub_lag)
        _print_stats(args.vs, d2)
        # treat A = incumbent (--vs), B = candidate (--forecaster)
        v = adoption_verdict(d2, d1, block=args.block)
        print("\n  adoption gate (candidate = --forecaster, incumbent = --vs):")
        print(f"    days paired      : {v['n_days']}")
        print(f"    mean improvement : {v['mean_improvement']:+.1f} MW "
              f"(95% CI [{v['ci_lo']:+.1f}, {v['ci_hi']:+.1f}])")
        print(f"    candidate win    : {v['win_rate']*100:.0f}%")
        for r in v["reasons"]:
            print(f"    - {r}")
        print(f"\n  VERDICT: {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
