#!/usr/bin/env python
"""
baselines.py — reference forecasters you have to beat.

A forecast is only as good as the baseline it improves on. These are the honest
floors for next-day German load, usable two ways:

  * as a library — import a function and call it from your own backtest, or
  * as a CLI — write a baseline submission CSV for a given day.

Each baseline takes the load history (an hourly UTC Series ending at the
publication cutoff) and the target day, and returns 24 hourly forecasts. They
are deterministic and dependency-light (numpy/pandas only); they make no use of
weather or any other team's forecast.

    # write a seasonal-naive submission for one day from a history CSV
    python baselines.py --history load_history.csv --date 2026-06-08 \
        --method seasonal_naive --out baseline_2026-06-08.csv

Methods
-------
seasonal_naive   y_hat(t) = y(t - 168h)        # same hour, one week earlier
weekly_profile   average of the last N matching (weekday, hour) values
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import _common as C


def seasonal_naive(history: pd.Series, target_day: pd.Timestamp) -> pd.Series:
    """Repeat the load from exactly one week (168 h) earlier.

    The classic hard-to-beat baseline for hourly electricity load: it captures
    the weekday-by-hour shape for free and is always available at submit time
    (the lag is a full week, hence settled).
    """
    hours = C.expected_hours(target_day)
    src = hours - pd.Timedelta(hours=168)
    vals = history.reindex(src).to_numpy(dtype=float)
    return pd.Series(vals, index=hours, name=C.FC_COL)


def weekly_profile(history: pd.Series, target_day: pd.Timestamp,
                   n_weeks: int = 4) -> pd.Series:
    """Average the last ``n_weeks`` values at the same weekday-and-hour.

    A denoised seasonal-naive: instead of one week back it averages several,
    which steadies the estimate against a single anomalous week.
    """
    hours = C.expected_hours(target_day)
    out = []
    for h in hours:
        lags = [h - pd.Timedelta(hours=168 * k) for k in range(1, n_weeks + 1)]
        vals = history.reindex(lags).dropna()
        out.append(float(vals.mean()) if len(vals) else np.nan)
    return pd.Series(out, index=hours, name=C.FC_COL)


METHODS = {"seasonal_naive": seasonal_naive, "weekly_profile": weekly_profile}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True,
                    help="CSV of hourly load history (timestamp_utc,value)")
    ap.add_argument("--date", required=True, help="target day YYYY-MM-DD")
    ap.add_argument("--method", choices=sorted(METHODS), default="seasonal_naive")
    ap.add_argument("--out", default=None, help="write a submission CSV here")
    args = ap.parse_args()

    hist = C.read_forecast(args.history)
    fc = METHODS[args.method](hist, pd.Timestamp(args.date))
    if fc.isna().any():
        print(f"[warn] {int(fc.isna().sum())} hour(s) have no history to draw on.")

    if args.out:
        df = fc.rename(C.FC_COL).reset_index()
        df.columns = [C.TS_COL, C.FC_COL]
        df[C.TS_COL] = df[C.TS_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.to_csv(args.out, index=False)
        print(f"[ok] wrote {args.method} baseline -> {args.out}")
    else:
        for ts, v in fc.items():
            print(f"  {ts:%Y-%m-%dT%H:%M:%SZ}  {v:10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
