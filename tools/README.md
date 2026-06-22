# Hot Rod forecasting toolkit

A small, dependency-light set of command-line tools any team in the
Lastprognose-Challenge (or any next-day load-forecasting project) can use to
**validate, score, compare and backtest** daily forecasts. They encode the
evaluation discipline described in the [model cards](../model_cards) as runnable
code — most usefully the tail-gated adoption gate.

Everything here needs only `numpy` and `pandas`. Fetching live actuals from
ENTSO-E is the one optional extra (`entsoe-py` + a free API key); every tool also
works fully offline if you hand it an actuals CSV.

## Install

```sh
cd tools
python -m pip install -r requirements.txt          # numpy + pandas
# optional, for live actuals:
python -m pip install entsoe-py
```

The tools import a shared helper module (`_common.py`) that sits next to them, so
run them from inside the `tools/` folder, or as `python tools/<tool>.py` from the
repo root (both work).

## The data model

A daily submission is a 24-row CSV for one target day, hourly, in UTC:

```
timestamp_utc,forecast_mw
2026-06-08T00:00:00Z,39486.42
2026-06-08T01:00:00Z,39730.44
...
```

The score is the **rolling 7-day mean MAE** in MW (lower wins): each day has a
mean-absolute-error over its 24 hours, and the headline number is the mean of that
daily MAE over the trailing seven days. Actuals files use the same schema (the
value column may be named `forecast_mw`, `actual` or `value`).

> **Which actual series?** The leaderboard scores against the DE **control area**
> "Actual Total Load", not the DE-LU bidding zone. The live fetch uses the control
> area; if you supply your own actuals, use the same series or your numbers will
> disagree with the leaderboard.

## The tools

| Tool | What it does |
| --- | --- |
| `validate_submission.py` | Pre-flight a submission CSV: 24 rows, UTC grid, no gaps/dupes/NaNs, right day, sane MW range. Exit code gates a submit script. |
| `score_submission.py` | MAE of one submission against actuals (offline CSV or live ENTSO-E). |
| `rolling_mae.py` | The leaderboard metric: per-day MAE and rolling 7-day mean over a folder of daily CSVs, plus the tail summary. |
| `shadow_compare.py` | A/B two strategies day-by-day with a running tally and the **tail-gated adoption verdict** (paired bootstrap CI + tail check). |
| `backtest.py` | Rolling-origin backtest harness for *your* forecaster (a `module:function`), with the same tail-gated verdict in compare mode. |
| `baselines.py` | Reference forecasters to beat (`seasonal_naive`, `weekly_profile`) — importable and as a CLI. |
| `plot_field.py` | Interactive forecast-vs-actual chart for one day as a self-contained HTML file (Chart.js; no matplotlib/plotly). |

Every tool has `--help`.

## Try it on the bundled example data

The [`../examples`](../examples) folder ships a small synthetic dataset (regenerate
with `python examples/make_examples.py`). From the repo root:

```sh
# 1. is my file well-formed?
python tools/validate_submission.py examples/sample_submission.csv

# 2. what did it score? (offline, against the bundled actuals)
python tools/score_submission.py examples/sample_submission.csv \
    --actual-csv examples/actuals_2026.csv

# 3. the leaderboard metric over a month of submissions
python tools/rolling_mae.py --forecasts examples/submissions_A \
    --actuals examples/actuals_2026.csv

# 4. should I switch from strategy A to strategy B?
python tools/shadow_compare.py --a examples/submissions_A --b examples/submissions_B \
    --actuals examples/actuals_2026.csv --label-a seasonal --label-b weekly

# 5. backtest a forecaster and A/B it against a baseline
python tools/backtest.py --forecaster baselines:weekly_profile \
    --vs baselines:seasonal_naive --actuals examples/actuals_2026.csv

# 6. eyeball where the curves miss
python tools/plot_field.py --actual examples/actuals_2026.csv \
    seasonal=examples/submissions_A/2026-06-15.csv \
    weekly=examples/submissions_B/2026-06-15.csv
```

## Plugging in your own model (`backtest.py`)

The harness only needs a callable referenced as `module:function`:

```python
# mymodel.py
import pandas as pd

def predict(history: pd.Series, target_day: pd.Timestamp) -> pd.Series:
    """history: hourly UTC load up to the publication cutoff.
       return: 24 hourly forecasts for target_day (UTC), MW."""
    ...
```

```sh
python tools/backtest.py --forecaster mymodel:predict \
    --vs baselines:seasonal_naive --actuals path/to/actuals.csv \
    --start 2024-07-01 --end 2025-06-30
```

It walks each target day, hands your function the history truncated at the
publication cutoff (`--pub-lag` hours before midnight), scores the 24 values against
the actual, and prints the tail-aware summary and the adoption verdict.

## The adoption gate, in one place

`shadow_compare.adoption_verdict()` and the compare mode of `backtest.py` apply the
rule Hot Rod uses to promote a model. Candidate **B** replaces incumbent **A** only
if all three hold:

1. **B lowers the mean** daily MAE;
2. the paired improvement (A − B) has a **moving-block-bootstrap 95% CI whose lower
   bound is above zero** (robust to resampling, not noise); and
3. **B does not regress the tail** — neither the worst-10-day mean nor the worst
   rolling-7-day window is worse than A's.

The mean alone is never enough: the competition score is tail-dominated, so a change
that trims the average while fattening the worst week loses ranking.

## Tests

```sh
cd tools
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite is self-contained (a synthetic load series stands in for ENTSO-E), so it
runs offline with no API key. It is also what CI runs on every push and pull request.
