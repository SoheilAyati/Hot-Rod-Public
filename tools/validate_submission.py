#!/usr/bin/env python
"""
validate_submission.py — pre-flight check for a daily forecast CSV.

Run this on your submission *before* you send it. It catches the mistakes that
get a file silently rejected or mis-scored by the leaderboard: wrong row count,
gaps or duplicate hours, off-grid timestamps, NaNs, a wrong target day, or
values that are obviously not load in MW.

    python validate_submission.py path/to/2026-06-08.csv
    python validate_submission.py 2026-06-08.csv --date 2026-06-08

Exit code 0 = clean, 1 = problems found (so it can gate a submit script).
The canonical schema is two columns, 24 hourly UTC rows:

    timestamp_utc,forecast_mw
    2026-06-08T00:00:00Z,39486.42
    ...

No network and no API key required.
"""
from __future__ import annotations

import argparse
import sys

import _common as C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="submission CSV to check")
    ap.add_argument("--date", default=None,
                    help="target day YYYY-MM-DD (default: inferred from the file)")
    args = ap.parse_args()

    problems = C.validate_forecast(args.csv, args.date)
    print("=" * 60)
    print(f"validate: {args.csv}")
    print("=" * 60)
    if not problems:
        try:
            s = C.read_forecast(args.csv)
            day = s.index[0].date()
            print(f"  OK - 24 hourly rows for {day}, "
                  f"range {s.min():.0f}-{s.max():.0f} MW")
        except ValueError:
            print("  OK")
        print("\nPASS: file is a well-formed submission.")
        return 0

    for p in problems:
        print(f"  [problem] {p}")
    print(f"\nFAIL: {len(problems)} issue(s) - fix before submitting.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
