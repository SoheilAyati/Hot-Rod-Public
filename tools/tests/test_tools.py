"""
Unit tests for the Hot Rod public forecasting toolkit.

Self-contained: a deterministic synthetic load series stands in for ENTSO-E, so
the suite runs offline with no API key. Covers IO/validation, the metrics, the
moving-block bootstrap, the baselines, the backtest harness, and the adoption
gate.
"""
import numpy as np
import pandas as pd
import pytest

import _common as C
import baselines
import backtest
from shadow_compare import adoption_verdict


# --------------------------------------------------------------------------- #
# Fixtures: a deterministic synthetic hourly load series                      #
# --------------------------------------------------------------------------- #
def synthetic_load(start="2024-01-01", days=120, seed=0):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 24, freq="h")
    h = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    rng = np.random.default_rng(seed)
    daily = 12000 * np.sin((h - 4) / 24 * 2 * np.pi) + 50000
    weekly = -4000 * (dow >= 5)  # weekends lower
    noise = rng.normal(0, 400, size=len(idx))
    return pd.Series(daily + weekly + noise, index=idx, name=C.FC_COL)


@pytest.fixture
def load_series():
    return synthetic_load()


def write_submission(tmp_path, day, series, name=None):
    hours = C.expected_hours(day)
    df = series.reindex(hours).rename(C.FC_COL).reset_index()
    df.columns = [C.TS_COL, C.FC_COL]
    df[C.TS_COL] = df[C.TS_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path = tmp_path / (name or f"{pd.Timestamp(day).date()}.csv")
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# IO + validation                                                             #
# --------------------------------------------------------------------------- #
def test_read_and_validate_good_file(tmp_path, load_series):
    day = "2024-02-01"
    p = write_submission(tmp_path, day, load_series)
    s = C.read_forecast(p)
    assert len(s) == 24
    assert str(s.index.tz) == "UTC"
    assert C.validate_forecast(p, day) == []


def test_validate_flags_wrong_row_count(tmp_path, load_series):
    day = "2024-02-01"
    p = write_submission(tmp_path, day, load_series)
    df = pd.read_csv(p).iloc[:20]            # drop 4 hours
    df.to_csv(p, index=False)
    problems = C.validate_forecast(p, day)
    assert any("24 rows" in x for x in problems)


def test_validate_flags_nan_and_missing(tmp_path, load_series):
    day = "2024-02-01"
    s = load_series.copy()
    p = write_submission(tmp_path, day, s)
    df = pd.read_csv(p)
    df[C.FC_COL] = df[C.FC_COL].astype(object)
    df.loc[3, C.FC_COL] = ""                 # a blank -> NaN after coercion
    df.to_csv(p, index=False)
    problems = C.validate_forecast(p, day)
    assert any("missing" in x or "NaN" in x for x in problems)


def test_read_missing_file_raises():
    with pytest.raises(ValueError):
        C.read_forecast("does_not_exist.csv")


# --------------------------------------------------------------------------- #
# Metrics + bootstrap                                                         #
# --------------------------------------------------------------------------- #
def test_mae_zero_when_identical(load_series):
    s = load_series.iloc[:24]
    assert C.mae(s, s) == pytest.approx(0.0)


def test_daily_and_rolling_mae(load_series):
    actual = load_series
    forecast = load_series + 100.0          # constant +100 MW bias -> MAE 100
    daily = C.daily_mae(forecast, actual)
    assert daily.mean() == pytest.approx(100.0, abs=1e-6)
    roll = C.rolling_7day_mae(daily)
    assert roll.dropna().iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_bootstrap_ci_orders_and_brackets_mean():
    rng = np.random.default_rng(1)
    diff = rng.normal(50, 10, size=200)     # clearly positive improvement
    mean, lo, hi = C.moving_block_bootstrap_ci(diff, block=7, n_boot=2000)
    assert lo < mean < hi
    assert lo > 0                           # robustly positive


def test_bootstrap_ci_spans_zero_for_noise():
    rng = np.random.default_rng(2)
    diff = rng.normal(0, 50, size=200)
    _, lo, hi = C.moving_block_bootstrap_ci(diff, block=7, n_boot=2000)
    assert lo < 0 < hi


# --------------------------------------------------------------------------- #
# Baselines + backtest harness                                                #
# --------------------------------------------------------------------------- #
def test_seasonal_naive_shape_and_values(load_series):
    day = load_series.index[80 * 24].normalize()
    fc = baselines.seasonal_naive(load_series, day)
    assert len(fc) == 24
    # equals the value exactly one week earlier
    week_ago = C.expected_hours(day) - pd.Timedelta(hours=168)
    assert np.allclose(fc.to_numpy(), load_series.reindex(week_ago).to_numpy())


def test_backtest_runs_and_scores(load_series):
    daily = backtest.run_backtest(baselines.seasonal_naive, load_series,
                                  start=None, end=None, pub_lag_h=24)
    assert len(daily) > 50
    # seasonal-naive on this smooth series should be well under the signal scale
    assert 0 < daily.mean() < 5000


def test_weekly_profile_beats_seasonal_naive_on_average(load_series):
    d_sn = backtest.run_backtest(baselines.seasonal_naive, load_series,
                                 None, None, 24)
    d_wp = backtest.run_backtest(baselines.weekly_profile, load_series,
                                 None, None, 24)
    # averaging several weeks cancels noise, so it should not be worse on the mean
    assert d_wp.mean() <= d_sn.mean() + 1e-6


# --------------------------------------------------------------------------- #
# Adoption gate                                                               #
# --------------------------------------------------------------------------- #
def test_adoption_gate_adopts_clear_winner():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    a = pd.Series(np.linspace(1200, 1300, 120), index=idx)
    b = a - 150.0                            # B uniformly 150 MW better
    v = adoption_verdict(a, b)
    assert v["verdict"] == "ADOPT B"
    assert v["ci_lo"] > 0


def test_adoption_gate_keeps_when_tail_regresses():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    a = pd.Series(rng.normal(1300, 100, 120), index=idx)
    b = a - 30.0                             # slightly better on the mean...
    b.iloc[:10] = a.iloc[:10] + 2000         # ...but a much worse tail
    v = adoption_verdict(a, b)
    assert v["verdict"] == "KEEP A"
