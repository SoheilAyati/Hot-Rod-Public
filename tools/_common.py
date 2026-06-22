"""
_common.py — shared helpers for the Hot Rod public forecasting toolkit.

Pure, dependency-light building blocks used by the command-line tools in this
folder (``score_submission.py``, ``validate_submission.py``, ``rolling_mae.py``,
``shadow_compare.py``, ``backtest.py``). Everything here works on the public
competition data model and needs only ``numpy`` and ``pandas``; the optional
ENTSO-E download lives in :func:`pull_actual` and imports ``entsoe`` lazily, so
the rest of the toolkit runs with no API key and no network.

The data model (the Lastprognose-Challenge submission format)
-------------------------------------------------------------
A daily submission is a 24-row CSV for one target day D:

    timestamp_utc,forecast_mw
    2026-06-08T00:00:00Z,39486.42
    2026-06-08T01:00:00Z,39730.44
    ...                               (24 consecutive hourly rows, UTC)

The score is the **rolling 7-day mean MAE** in MW (lower is better): each day
has a mean-absolute-error over its 24 hours, and the headline number is the
mean of that daily MAE over the trailing seven days.

None of these helpers are specific to any one team's model — they operate on
forecast/actual series only, so any team can reuse them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Columns of the canonical submission CSV.
TS_COL = "timestamp_utc"
FC_COL = "forecast_mw"


# --------------------------------------------------------------------------- #
# Reading and validating submission / forecast CSVs                           #
# --------------------------------------------------------------------------- #
def read_forecast(path: str | Path) -> pd.Series:
    """Read a submission CSV into a tz-aware (UTC) hourly Series of MW.

    Accepts the canonical ``timestamp_utc,forecast_mw`` schema and is lenient
    about a few common variants (a ``value``/``load_mw`` value column, a
    ``timestamp``/``time`` time column). The returned Series is sorted, named
    ``forecast_mw`` and indexed in UTC. Raises ``ValueError`` with a clear
    message when the file cannot be interpreted — never returns a silently
    mangled frame (fail-safe, not fail-quiet).
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name}: file has no rows")

    ts_col = _first_present(df.columns, [TS_COL, "timestamp", "time", "datetime"])
    fc_col = _first_present(df.columns, [FC_COL, "value", "load_mw", "mw", "y"])
    if ts_col is None or fc_col is None:
        raise ValueError(
            f"{path.name}: expected columns '{TS_COL},{FC_COL}' "
            f"but found {list(df.columns)}"
        )

    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if idx.isna().any():
        bad = int(idx.isna().sum())
        raise ValueError(f"{path.name}: {bad} timestamp(s) could not be parsed")
    vals = pd.to_numeric(df[fc_col], errors="coerce")

    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(idx), name=FC_COL)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def _first_present(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def expected_hours(target_date: str | pd.Timestamp) -> pd.DatetimeIndex:
    """The 24 hourly UTC timestamps that a submission for ``target_date`` must cover."""
    t0 = pd.Timestamp(target_date).tz_localize("UTC") if pd.Timestamp(
        target_date
    ).tzinfo is None else pd.Timestamp(target_date).tz_convert("UTC")
    t0 = t0.normalize()
    return pd.date_range(t0, periods=24, freq="h", tz="UTC")


def validate_forecast(path: str | Path,
                      target_date: str | pd.Timestamp | None = None) -> list[str]:
    """Return a list of human-readable problems with a submission CSV.

    An empty list means the file is a well-formed 24-row hourly UTC submission.
    When ``target_date`` is given the hours are checked against that exact day;
    otherwise the day is inferred from the first timestamp. This mirrors the
    checks the leaderboard applies, so teams can catch a rejection *before*
    submitting.
    """
    problems: list[str] = []
    try:
        s = read_forecast(path)
    except ValueError as exc:
        return [str(exc)]

    if target_date is None:
        target_date = s.index[0].tz_convert("UTC").normalize()
    want = expected_hours(target_date)

    if len(s) != 24:
        problems.append(f"expected 24 rows, found {len(s)}")
    if s.isna().any():
        problems.append(f"{int(s.isna().sum())} missing/NaN forecast value(s)")
    if (~s.index.isin(want)).any():
        extra = s.index.difference(want)
        problems.append(
            f"{len(extra)} row(s) outside the target day {want[0].date()} "
            f"(first stray: {extra[0]})" if len(extra) else "timestamps off-grid"
        )
    missing = want.difference(s.index)
    if len(missing):
        problems.append(f"{len(missing)} hour(s) of the target day missing "
                        f"(first: {missing[0]})")
    finite = s.dropna()
    if len(finite) and ((finite <= 0).any() or (finite > 200_000).any()):
        problems.append("forecast value(s) outside a sane MW range (0, 200000)")
    return problems


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def mae(forecast: pd.Series, actual: pd.Series) -> float:
    """Mean absolute error over the hours present in *both* series (MW)."""
    j = pd.concat([forecast.rename("f"), actual.rename("a")], axis=1).dropna()
    if j.empty:
        return float("nan")
    return float((j["f"] - j["a"]).abs().mean())


def daily_mae(forecast: pd.Series, actual: pd.Series) -> pd.Series:
    """Per-day MAE (MW), indexed by calendar date (UTC). One value per day."""
    j = pd.concat([forecast.rename("f"), actual.rename("a")], axis=1).dropna()
    if j.empty:
        return pd.Series(dtype=float, name="daily_mae")
    err = (j["f"] - j["a"]).abs()
    out = err.groupby(j.index.tz_convert("UTC").date).mean()
    out.index = pd.to_datetime(out.index)
    out.name = "daily_mae"
    return out


def rolling_7day_mae(daily: pd.Series, window: int = 7,
                     min_periods: int | None = None) -> pd.Series:
    """Rolling mean of a daily-MAE series — the headline leaderboard metric.

    ``daily`` is one MAE per day (see :func:`daily_mae`). With the default
    ``window=7`` and ``min_periods=window`` the result is defined only once a
    full week of days is available, matching a strict rolling-7-day score.
    """
    if min_periods is None:
        min_periods = window
    return daily.sort_index().rolling(window, min_periods=min_periods).mean()


def tail_stats(daily: pd.Series, worst_n: int = 10) -> dict:
    """Tail summary of a daily-MAE series: mean, median, worst-N mean, worst
    single day, and worst rolling-7-day window. The competition score is
    tail-dominated, so a healthy mean is never enough on its own."""
    d = daily.dropna().sort_index()
    roll = rolling_7day_mae(d, min_periods=1)
    worst = d.sort_values(ascending=False)
    return {
        "n_days": int(len(d)),
        "mean": float(d.mean()) if len(d) else float("nan"),
        "median": float(d.median()) if len(d) else float("nan"),
        f"worst_{worst_n}_mean": float(worst.head(worst_n).mean()) if len(d) else float("nan"),
        "worst_day": float(d.max()) if len(d) else float("nan"),
        "worst_roll7": float(roll.max()) if len(roll.dropna()) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Moving-block bootstrap (paired) — the statistical backbone of adoption gates #
# --------------------------------------------------------------------------- #
def moving_block_bootstrap_ci(diff: np.ndarray | pd.Series,
                              block: int = 7,
                              n_boot: int = 10_000,
                              ci: float = 0.95,
                              seed: int = 2026) -> tuple[float, float, float]:
    """Moving-block-bootstrap confidence interval for the mean of a paired
    difference series.

    Daily errors are autocorrelated (weather regimes persist, the score itself
    is a 7-day roll), so an i.i.d. bootstrap understates the interval. The
    moving-block bootstrap resamples contiguous blocks of length ``block`` to
    preserve that short-range dependence. Returns ``(mean, lo, hi)`` for the
    central ``ci`` interval. A lower bound above zero means the improvement is
    robust to resampling — the precise sense in which Hot Rod calls a gain
    "validated".
    """
    x = np.asarray(diff, dtype=float)
    x = x[~np.isnan(x)]
    nobs = x.size
    if nobs == 0:
        return (float("nan"), float("nan"), float("nan"))
    block = max(1, min(block, nobs))
    n_blocks = int(np.ceil(nobs / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, nobs - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = (starts[b][:, None] + offsets).ravel()[:nobs]
        means[b] = x[idx].mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return (float(x.mean()), lo, hi)


# --------------------------------------------------------------------------- #
# Optional: pull DE actual load from ENTSO-E (lazy import; needs a free key)   #
# --------------------------------------------------------------------------- #
def pull_actual(target_date: str | pd.Timestamp, area: str = "DE") -> pd.Series:
    """Fetch the 24 hourly *actual* DE load values for ``target_date`` (UTC, MW).

    Requires the ``entsoe-py`` package and the ``ENTSOE_API_KEY`` environment
    variable (free registration at https://transparency.entsoe.eu/). The
    leaderboard scores against the DE **control area** "Actual Total Load", so
    ``area`` defaults to ``"DE"`` rather than the ``"DE_LU"`` bidding zone — a
    distinction worth knowing, because the two series differ and using the
    wrong one makes your self-computed MAE disagree with the leaderboard.

    Returns an empty Series (with a warning) if the day has not yet been
    published; never raises on a not-yet-settled day so callers can degrade
    gracefully.
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "pull_actual needs the 'entsoe-py' package: pip install entsoe-py"
        ) from exc

    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("set ENTSOE_API_KEY in the environment to fetch actuals")

    client = EntsoePandasClient(api_key=key)
    start = pd.Timestamp(target_date, tz="UTC") - pd.Timedelta(hours=3)
    end = pd.Timestamp(target_date, tz="UTC") + pd.Timedelta(hours=27)
    last_err = None
    for ar in [area, "DE_LU"]:  # DE first (control area = what the leaderboard scores)
        try:
            raw = client.query_load(ar, start=start, end=end)
            s = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
            s = s.astype(float)
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")
            s.index = s.index.tz_convert("UTC")
            s = s.resample("h").mean()
            want = expected_hours(target_date)
            return s.reindex(want).rename("actual")
        except Exception as exc:  # noqa: BLE001 - try the next area code
            last_err = exc
    print(f"[warn] could not fetch actual load for {target_date}: {last_err}")
    return pd.Series(dtype=float, name="actual")
