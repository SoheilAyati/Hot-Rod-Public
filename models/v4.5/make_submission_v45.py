"""
make_submission_v45.py  —  Voltcrown v4.5 (Team hot_rod), public reference release
==================================================================================

This is the **complete, unmodified** v4.5 submission model, released in full and
open for anyone in the Lastprognose-Challenge community to use. v4.5 is two
generations behind the team's deployed model; everything about it — the exact
hyperparameters, the training windows, the feature set and the bias-correction
algorithm — is open here and in the accompanying model card
(../../model_cards/model_card_v45.md).

The only change from the operational script is portability: the audit-log
directory is taken from the ``HOTROD_LOG_DIR`` environment variable (default
``./logs``) instead of a hard-coded path. The forecast itself is unaffected —
the log location does not enter the computation (CR-2), so the output CSV stays
bit-identical to the operational runs.

v4.5 = v4 (expanded training data: 2018-2019 + 2024-D-1) with tuned
hyperparameters (n_estimators=3000, num_leaves=23, min_child_samples=40),
PLUS an out-of-sample bias-correction layer.

== AUDIT LOG (AI Act Art. 12 record-keeping / PR-4) =========================
On every run we switch on spotforecast2-safe's structured audit log FIRST,
before any work, because the run itself is a log-worthy action. Once on, the
library's own actions (ENTSO-E download, weather fetch, NaN handling,
training, prediction) write into the same file automatically; we add explicit
entries for our custom steps (features, bias estimate, forecast summary).
Logs land in:  $HOTROD_LOG_DIR/make_submission_v45/<name>_<date>_<time>.log
one dated file per run. The log carries timestamps (a record); this does NOT
affect determinism - the forecast CSV stays bit-identical (CR-2).

== THE BIAS CORRECTION ======================================================

Diagnosis: the model systematically UNDER-predicts. Backtest bias ~ -400 MW.
The cause is a mix of (a) 2018-2019 training data having lower absolute load
than the present, and (b) tree models not extrapolating an upward trend. The
model can't see this drift, so we correct for it externally.

How we estimate the bias WITHOUT cheating (in-sample residuals are ~0 and
useless):
  1. Hold out the most recent BIAS_WINDOW_DAYS days.
  2. Train a "bias model" on everything BEFORE the holdout.
  3. Predict the held-out days  -> genuinely out-of-sample residuals.
  4. bias = median(actual - predicted) over those days, EXCLUDING holidays
     (so a public holiday in the window doesn't skew the estimate).
     Positive bias = the model runs low = we shift the forecast UP.

Then:
  5. Train the FINAL model on ALL data (incl. the recent days).
  6. Predict the target day.
  7. corrected = raw_prediction + bias.

The correction is clipped to +/- BIAS_CLIP_MW as a safety rail: if the
estimate is implausibly large, something is wrong and we don't want to
blindly shift the forecast by thousands of MW.

NOTE (honesty for the record): over a full live-replica year this bias layer
turned out to HURT the mean (it improves only the worst-7-day window). It is
left ON here because that is the model as it was deployed; pass
``--no-bias-correction`` for the better-on-average variant. See the model card
§7 for the numbers.

DETERMINISM (CR-2): bias estimation uses a fixed lookback window and the
same seeded LGBM. Given the same data, the correction is reproducible.

All other compliance properties carry over from v4:
  CR-1 auditability, CR-3 fail-safe (raise on missing), CR-4 deps.

REQUIREMENTS:
  pip install "spotforecast2-safe>=15.5,<16" lightgbm holidays numpy pandas
  export ENTSOE_API_KEY="your-token"     # free at https://transparency.entsoe.eu/

USAGE (writes into <repo-root>/submissions/<team>/<date>.csv):
    export PYTHONHASHSEED=0
    python make_submission_v45.py --team hot_rod --target-date 2026-05-29 \
        --repo-root /path/to/your/leaderboard/checkout

To compare against the UNCORRECTED model, run with --no-bias-correction
and diff the two CSVs.
"""

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import holidays as holidays_lib
from lightgbm import LGBMRegressor

from spotforecast2_safe import LinearlyInterpolateTS
from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home
from spotforecast2_safe.downloader.entsoe import download_new_data
from spotforecast2_safe.weather import WeatherService
from spotforecast2_safe.manager.logger import setup_logging


# =============================================================================
# CONFIG
# =============================================================================
DATA_PULL_START = "2018-01-01"
HDD_BASE = 15.0
CDD_BASE = 22.0

KEEP_WINDOWS = [
    ("2018-01-01", "2019-12-31"),
    ("2024-01-01", None),
]

# --- bias correction ---
BIAS_WINDOW_DAYS = 14      # out-of-sample lookback for bias estimation
BIAS_CLIP_MW     = 2500.0  # safety rail: never shift by more than this

# --- audit log location (one dated file per run, in a per-script subfolder) ---
# Portable: override with the HOTROD_LOG_DIR environment variable.
LOG_ROOT    = Path(os.environ.get("HOTROD_LOG_DIR", "logs"))
SCRIPT_NAME = "make_submission_v45"

LGBM_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=23,
    min_child_samples=40,
    random_state=2026,
    verbose=-1,
    n_jobs=-1,
)

CITIES = [
    ("berlin",    52.520,  13.405,  5.0),
    ("hamburg",   53.551,   9.993,  3.4),
    ("munich",    48.137,  11.575,  3.0),
    ("rhineruhr", 50.937,   6.960, 10.0),
    ("frankfurt", 50.110,   8.682,  5.8),
    ("stuttgart", 48.775,   9.183,  2.7),
]
GERMAN_STATES = [
    "BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV",
    "NI", "NW", "RP", "SL", "SN", "ST", "TH", "SH",
]

WX_HIST_CACHE_DIR     = Path("_cache_weather")
WX_FORECAST_CACHE_DIR = Path("_cache_weather_forecasts")
WX_HIST_CACHE_DIR.mkdir(exist_ok=True)
WX_FORECAST_CACHE_DIR.mkdir(exist_ok=True)
FORECAST_WINDOW_DAYS_BACK = 7


# =============================================================================
def compute_keep_mask(idx, cutoff_ts):
    """Rows in KEEP_WINDOWS up to cutoff_ts (the '2024->None' window ends at cutoff)."""
    mask = np.zeros(len(idx), dtype=bool)
    for start_str, end_str in KEEP_WINDOWS:
        s = pd.Timestamp(start_str, tz="UTC")
        e = (cutoff_ts if end_str is None
             else pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(hours=23))
        mask |= (idx >= s) & (idx <= e)
    return mask


def slug(name):
    s = name.lower()
    for k, v in [("ä","ae"),("ö","oe"),("ü","ue"),("ß","ss")]:
        s = s.replace(k, v)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def build_holiday_map(years):
    out = {}
    for state in GERMAN_STATES:
        for date_, name in holidays_lib.Germany(subdiv=state, years=years).items():
            out.setdefault(slug(name), set()).add(date_)
    return out


def any_holiday_set(m):
    s = set()
    for v in m.values():
        s |= v
    return s


def fetch_one_city(name, lat, lon, start, end, use_forecast, cache_path):
    ws = WeatherService(latitude=lat, longitude=lon,
                       cache_path=cache_path, use_forecast=use_forecast)
    df = ws.get_dataframe(start=start, end=end, freq="h", fill_missing=False,
                          fallback_on_failure=False)
    keep = [c for c in ["temperature_2m", "cloud_cover", "wind_speed_10m"]
            if c in df.columns]
    if not keep:
        raise RuntimeError(f"[wx] {name} returned: {list(df.columns)}")
    return df[keep]


def fetch_weather_blended(data_start, target_end, target_date_str):
    cutoff = (pd.Timestamp.now("UTC").normalize()
              - pd.Timedelta(days=FORECAST_WINDOW_DAYS_BACK))
    fc_dir = WX_FORECAST_CACHE_DIR / target_date_str
    fc_dir.mkdir(parents=True, exist_ok=True)

    total_pop = sum(c[3] for c in CITIES)
    weighted = None
    for name, lat, lon, pop in CITIES:
        w = pop / total_pop
        print(f"[wx]   {name:<10s}  weight={w:.2%}")
        hist = fetch_one_city(name, lat, lon, data_start, cutoff,
                              use_forecast=False,
                              cache_path=WX_HIST_CACHE_DIR / f"hist_{name}.parquet")
        fc   = fetch_one_city(name, lat, lon, cutoff, target_end,
                              use_forecast=True,
                              cache_path=fc_dir / f"{name}.parquet")
        combined = pd.concat([hist, fc.loc[~fc.index.isin(hist.index)]]).sort_index()
        wc = combined * w
        weighted = wc if weighted is None else weighted.add(wc, fill_value=0)
    weighted = weighted.rename(columns=lambda c: f"de_{c}")
    print(f"[wx] forecast snapshot dir: {fc_dir.resolve()}")
    return weighted


def build_features(y, weather):
    idx = y.index
    df = pd.DataFrame(index=idx)
    df["lag_168"] = y.shift(168)
    df["lag_336"] = y.shift(336)
    df["lag_504"] = y.shift(504)
    df["lag_672"] = y.shift(672)
    df["roll_mean_last_week"] = y.shift(168).rolling(168).mean()
    df["hour"]       = idx.hour
    df["dayofweek"]  = idx.dayofweek
    df["month"]      = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    years = sorted({d.year for d in idx})
    hmap = build_holiday_map(years)
    all_hol = any_holiday_set(hmap)
    dates = pd.Index(idx.date)
    for hslug, hdates in hmap.items():
        df[f"h_{hslug}"] = dates.isin(hdates).astype(int)
    df["is_any_holiday"]        = dates.isin(all_hol).astype(int)
    df["is_day_before_holiday"] = (
        pd.Index(idx.date + pd.Timedelta(days=1)).isin(all_hol).astype(int)
    )
    df["is_day_after_holiday"]  = (
        pd.Index(idx.date - pd.Timedelta(days=1)).isin(all_hol).astype(int)
    )
    bridge = np.zeros(len(idx), dtype=int)
    for i, ts in enumerate(idx):
        d, wd = ts.date(), ts.dayofweek
        if wd >= 5 or d in all_hol: continue
        if wd == 0 and (d + pd.Timedelta(days=1)) in all_hol: bridge[i] = 1
        elif wd == 4 and (d - pd.Timedelta(days=1)) in all_hol: bridge[i] = 1
    df["is_bridge_day"] = bridge

    df = df.join(weather, how="left")
    if "de_temperature_2m" in df.columns:
        df["de_hdd"] = (HDD_BASE - df["de_temperature_2m"]).clip(lower=0)
        df["de_cdd"] = (df["de_temperature_2m"] - CDD_BASE).clip(lower=0)
    return df


def load_load_series(start_dt, end_dt):
    print(f"[load] fetching ENTSO-E DE load "
          f"{start_dt.date()} -> {end_dt.date()} ...")
    api_key = os.environ["ENTSOE_API_KEY"]
    download_new_data(
        api_key=api_key, country_code="DE",
        start=start_dt.strftime("%Y%m%d%H%M"),
        end=end_dt.strftime("%Y%m%d%H%M"),
        force=True,
    )
    interim = get_data_home() / "interim" / "energy_load.csv"
    df_raw = fetch_data(filename=str(interim))
    df_raw.index = pd.to_datetime(df_raw.index, utc=True)
    load_col = next(c for c in df_raw.columns if "Actual" in c and "Load" in c)
    y = df_raw[load_col].astype(float).rename("load")
    if y.index.inferred_freq != "h":
        y = y.resample("h").mean()
    y = y.loc[start_dt:end_dt]
    y = y.loc[:y.last_valid_index()]
    y = LinearlyInterpolateTS(on_missing="raise").fit_transform(y)
    y.index = pd.DatetimeIndex(y.index, freq="h")
    print(f"[load] {len(y):,} hours from {y.index[0]} to {y.index[-1]}")
    return y


# =============================================================================
# BIAS ESTIMATION (out-of-sample)
# =============================================================================
def estimate_bias(features, y_full, train_end_ts, logger=None):
    """
    Hold out the last BIAS_WINDOW_DAYS days, train on everything before them,
    predict the holdout, and return the median residual (actual - predicted)
    over non-holiday hours. Positive => model runs low => shift forecast UP.
    """
    holdout_start = (train_end_ts
                     - pd.Timedelta(days=BIAS_WINDOW_DAYS)
                     + pd.Timedelta(hours=1))
    bias_train_cutoff = holdout_start - pd.Timedelta(hours=1)

    # Bias model trains on kept windows strictly BEFORE the holdout
    keep = compute_keep_mask(features.index, bias_train_cutoff)
    clean = y_full.notna().to_numpy() & features.notna().all(axis=1).to_numpy()
    bias_train_mask = keep & clean & (features.index <= bias_train_cutoff)

    X_bt = features.loc[bias_train_mask]
    y_bt = y_full.loc[bias_train_mask]
    if len(X_bt) < 5000:
        raise SystemExit(f"[bias] only {len(X_bt)} rows to train bias model")

    print(f"[bias] training holdout model on {len(X_bt):,} rows "
          f"(through {bias_train_cutoff.date()}) ...")
    if logger:
        logger.info(f"bias holdout model | rows={len(X_bt)} "
                    f"train_through={bias_train_cutoff.date()}")
    m_bias = LGBMRegressor(**LGBM_PARAMS)
    m_bias.fit(X_bt, y_bt)

    # Holdout window: the BIAS_WINDOW_DAYS days, non-holiday, clean, known y
    holdout_mask = (
        (features.index >= holdout_start)
        & (features.index <= train_end_ts)
        & clean
        & (features["is_any_holiday"].to_numpy() == 0)
    )
    X_ho = features.loc[holdout_mask]
    y_ho = y_full.loc[holdout_mask]
    if len(X_ho) < 24 * 5:
        print(f"[bias] WARNING: only {len(X_ho)} non-holiday holdout hours; "
              f"bias estimate may be noisy.")
        if logger:
            logger.warning(f"bias holdout thin | non_holiday_hours={len(X_ho)}")

    pred_ho = m_bias.predict(X_ho)
    residuals = y_ho.values - pred_ho            # +ve = under-predicted
    bias_med = float(np.median(residuals))
    bias_mean = float(np.mean(residuals))

    print(f"[bias] holdout window {holdout_start.date()} -> {train_end_ts.date()}"
          f"  ({len(X_ho)} non-holiday hrs)")
    print(f"[bias] median residual = {bias_med:+.1f} MW   "
          f"(mean = {bias_mean:+.1f} MW)")
    if logger:
        logger.info(f"bias holdout | window={holdout_start.date()}..{train_end_ts.date()} "
                    f"non_holiday_hrs={len(X_ho)} median={bias_med:+.1f} mean={bias_mean:+.1f}")

    # Safety rail
    bias_applied = float(np.clip(bias_med, -BIAS_CLIP_MW, BIAS_CLIP_MW))
    if bias_applied != bias_med:
        print(f"[bias] WARNING: clipped bias from {bias_med:+.1f} to "
              f"{bias_applied:+.1f} MW (|bias| exceeded {BIAS_CLIP_MW})")
        if logger:
            logger.warning(f"bias clipped | from={bias_med:+.1f} to={bias_applied:+.1f} "
                           f"clip={BIAS_CLIP_MW}")
    return bias_applied


# =============================================================================
# MAIN
# =============================================================================
def make_submission(team, target_date, repo_root, apply_bias=True):
    # --- switch on the audit log FIRST (before any work) ---
    log_dir = LOG_ROOT / SCRIPT_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logging(level=logging.INFO, log_dir=log_dir)
    logger.info(f"audit run start | script={SCRIPT_NAME}.py model=v4.5 "
                f"team={team} target_date={target_date} apply_bias={apply_bias}")
    logger.info(f"config | lgbm_params={LGBM_PARAMS} keep_windows={KEEP_WINDOWS} "
                f"bias_window_days={BIAS_WINDOW_DAYS} bias_clip_mw={BIAS_CLIP_MW} "
                f"data_pull_start={DATA_PULL_START}")
    if log_path:
        print(f"[log] audit log: {log_path}")

    try:
        return _run_submission(team, target_date, repo_root, apply_bias, logger)
    except BaseException:
        logger.exception("audit run FAILED")
        raise


def _run_submission(team, target_date, repo_root, apply_bias, logger):
    target_hours_start = pd.Timestamp(target_date, tz="UTC")
    target_hours_end   = target_hours_start + pd.Timedelta(hours=23)
    train_end_ts       = target_hours_start - pd.Timedelta(hours=1)
    data_start_ts      = pd.Timestamp(DATA_PULL_START, tz="UTC")

    y_actual = load_load_series(data_start_ts, train_end_ts)
    logger.info(f"load series | hours={len(y_actual)} "
                f"from={y_actual.index[0]} to={y_actual.index[-1]}")

    full_index = pd.date_range(y_actual.index[0], target_hours_end,
                               freq="h", tz="UTC")
    y_full = pd.Series(np.nan, index=full_index, name="load")
    y_full.loc[y_actual.index] = y_actual.values

    print("[wx] fetching multi-city weather (2018 onward + forecast) ...")
    weather = fetch_weather_blended(full_index[0], target_hours_end, target_date)
    weather = weather.reindex(full_index)
    n_missing = int(weather.isna().any(axis=1).sum())
    if n_missing > 0:
        print(f"[wx] WARNING: {n_missing} weather rows missing after reindex")
        logger.warning(f"weather missing | rows={n_missing}")
    logger.info(f"weather assembled | cities={len(CITIES)} missing_rows={n_missing}")

    print("[feat] building feature matrix ...")
    features = build_features(y_full, weather)
    print(f"[feat] shape: {features.shape}")
    logger.info(f"features built | shape={features.shape}")

    X_target = features.loc[target_hours_start:target_hours_end]
    if len(X_target) != 24:
        raise SystemExit(f"[fatal] expected 24 target rows, got {len(X_target)}")
    if X_target.isna().any().any():
        missing = X_target.columns[X_target.isna().any()].tolist()
        raise SystemExit(f"[fatal] target rows missing feature(s): {missing}")

    # --- estimate bias (before training the final model) ---
    bias = 0.0
    if apply_bias:
        bias = estimate_bias(features, y_full, train_end_ts, logger=logger)
        logger.info(f"bias correction | applied={bias:+.1f} MW")
    else:
        print("[bias] bias correction DISABLED (--no-bias-correction)")
        logger.info("bias correction | disabled")

    # --- final model on ALL kept data ---
    keep_mask = compute_keep_mask(features.index, train_end_ts)
    train_mask = (
        keep_mask
        & y_full.notna().to_numpy()
        & features.notna().all(axis=1).to_numpy()
    )
    X_train = features.loc[train_mask]
    y_train = y_full.loc[train_mask]
    print(f"[train] final model training rows: {len(X_train):,}")
    if len(X_train) < 5000:
        raise SystemExit(f"[fatal] only {len(X_train)} clean training rows")

    print("[train] fitting final LightGBM ...")
    logger.info(f"final model training | rows={len(X_train)} "
                f"seed={LGBM_PARAMS['random_state']}")
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    logger.info("final model | fitted")

    y_pred_raw = model.predict(X_target)
    y_pred = y_pred_raw + bias

    if (y_pred <= 0).any() or np.isnan(y_pred).any():
        raise SystemExit(f"[fatal] forecast invalid: {y_pred}")

    logger.info(f"forecast | raw_mean={y_pred_raw.mean():.1f} "
                f"final_mean={y_pred.mean():.1f} min={y_pred.min():.1f} "
                f"max={y_pred.max():.1f} bias={bias:+.1f}")

    submission = pd.DataFrame({
        "timestamp_utc": pd.date_range(target_hours_start, periods=24, freq="h",
                                       tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_mw":   np.round(y_pred, 2),
    })
    submissions_dir = repo_root / "submissions"
    if not submissions_dir.is_dir():
        raise SystemExit(f"Directory '{submissions_dir}' not found.")
    out = submissions_dir / team / f"{target_date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)

    print("\n[ok] forecast summary:")
    print(f"     bias correction applied: {bias:+.1f} MW")
    print(f"     raw   mean = {y_pred_raw.mean():>8.1f} MW")
    print(f"     final mean = {y_pred.mean():>8.1f} MW")
    print(f"     final min  = {y_pred.min():>8.1f} MW")
    print(f"     final max  = {y_pred.max():>8.1f} MW")
    logger.info(f"submission written | path={out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--no-bias-correction", action="store_true",
                        help="disable the bias correction (for comparison)")
    args = parser.parse_args()
    path = make_submission(args.team, args.target_date, args.repo_root.resolve(),
                           apply_bias=not args.no_bias_correction)
    print(f"\nSubmission written: {path}")
