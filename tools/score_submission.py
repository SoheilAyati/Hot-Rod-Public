#!/usr/bin/env python
"""
score_submission.py — score a forecast CSV against the actual load.

Computes the mean absolute error (MW) of a daily submission against the
ENTSO-E DE actual total load — the same quantity the leaderboard grades you on.
Get the actuals one of two ways:

  * ``--actual-csv FILE`` — an offline CSV of actuals (no network, no key); or
  * live from ENTSO-E (default) — needs ``pip install entsoe-py`` and the
    ``ENTSOE_API_KEY`` environment variable.

Examples
--------
    # offline, against a saved actuals file
    python score_submission.py 2026-06-08.csv --actual-csv actuals_2026-06-08.csv

    # live (pulls the DE control-area actual the leaderboard uses)
    python score_submission.py 2026-06-08.csv --date 2026-06-08

The actuals CSV uses the same schema as a submission; its value column may be
named ``forecast_mw``, ``actual`` or ``value``.

Note on the data series: the leaderboard scores against the DE **control area**
"Actual Total Load", not the DE-LU bidding zone. This tool fetches the control
area; if you supply your own actuals, make sure they are the same series.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import _common as C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="submission CSV to score")
    ap.add_argument("--date", default=None,
                    help="target day YYYY-MM-DD (default: inferred from the file)")
    ap.add_argument("--actual-csv", default=None,
                    help="offline actuals CSV (skips the ENTSO-E download)")
    ap.add_argument("--area", default="DE",
                    help="ENTSO-E area for the live fetch (default DE control area)")
    args = ap.parse_args()

    fc = C.read_forecast(args.csv)
    target = args.date or fc.index[0].tz_convert("UTC").normalize()

    problems = C.validate_forecast(args.csv, args.date)
    if problems:
        print(f"[warn] submission has {len(problems)} format issue(s) "
              f"(run validate_submission.py); scoring the rows present anyway.")

    if args.actual_csv:
        actual = C.read_forecast(args.actual_csv).rename("actual")
    else:
        actual = C.pull_actual(target, area=args.area)

    if actual.dropna().empty:
        print("[fatal] no actual values available for this day "
              "(not yet published, or empty actuals file).")
        return 1

    want = C.expected_hours(target)
    fc = fc.reindex(want)
    actual = actual.reindex(want)
    score = C.mae(fc, actual)
    n = int(pd.concat([fc, actual], axis=1).dropna().shape[0])

    print("=" * 60)
    print(f"score: {args.csv}")
    print("=" * 60)
    print(f"  target day      : {pd.Timestamp(target).date()}")
    print(f"  hours scored    : {n}/24")
    print(f"  MAE             : {score:8.1f} MW")
    if n < 24:
        print(f"  [note] {24 - n} hour(s) not yet settled - partial-day MAE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
