#!/usr/bin/env python
"""
make_examples.py — regenerate the small example dataset shipped in this folder.

Writes a deterministic, synthetic set so every tool in ``../tools`` can be tried
offline, with no ENTSO-E key:

  actuals_2026.csv          ~6 weeks of hourly "actual" load (the truth)
  sample_submission.csv     one well-formed 24-row daily submission
  submissions_A/*.csv       daily forecasts of strategy A (seasonal-naive)
  submissions_B/*.csv       daily forecasts of strategy B (weekly-profile)

The numbers are illustrative (a synthetic load curve, not real ENTSO-E data) —
they exist so the CLIs produce sensible output you can read, not as a real
forecast. Run from anywhere:  python examples/make_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import _common as C        # noqa: E402
import baselines           # noqa: E402


def synthetic_actuals(start="2026-05-01", days=46, seed=2026) -> pd.Series:
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 24, freq="h")
    h, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    rng = np.random.default_rng(seed)
    base = 12000 * np.sin((h - 4) / 24 * 2 * np.pi) + 52000
    weekend = -5000 * (dow >= 5)
    noise = rng.normal(0, 600, size=len(idx))
    return pd.Series(base + weekend + noise, index=idx, name=C.FC_COL)


def write_submission(series: pd.Series, day: pd.Timestamp, path: Path) -> None:
    hours = C.expected_hours(day)
    df = series.reindex(hours).rename(C.FC_COL).reset_index()
    df.columns = [C.TS_COL, C.FC_COL]
    df[C.TS_COL] = df[C.TS_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df[C.FC_COL] = df[C.FC_COL].round(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    actual = synthetic_actuals()

    # the long actuals file (whole range, one CSV)
    af = actual.rename(C.FC_COL).reset_index()
    af.columns = [C.TS_COL, C.FC_COL]
    af[C.TS_COL] = af[C.TS_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    af[C.FC_COL] = af[C.FC_COL].round(2)
    af.to_csv(HERE / "actuals_2026.csv", index=False)

    # daily forecasts for both strategies, for every day with a full week of history
    rng = np.random.default_rng(7)
    first = actual.index[0].normalize() + pd.Timedelta(days=7)
    last = actual.index[-1].normalize()
    days = pd.date_range(first, last, freq="D", tz="UTC")
    for day in days:
        a = baselines.seasonal_naive(actual, day) + rng.normal(0, 200, 24)
        b = baselines.weekly_profile(actual, day, n_weeks=4) + rng.normal(0, 200, 24)
        write_submission(a, day, HERE / "submissions_A" / f"{day.date()}.csv")
        write_submission(b, day, HERE / "submissions_B" / f"{day.date()}.csv")

    # one standalone sample submission
    sample_day = days[-1]
    write_submission(baselines.seasonal_naive(actual, sample_day), sample_day,
                     HERE / "sample_submission.csv")

    print(f"[ok] wrote actuals_2026.csv ({len(actual)} hours), "
          f"{len(days)} day(s) each in submissions_A/ and submissions_B/, "
          f"and sample_submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
