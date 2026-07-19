"""
make_submission_v75.py   (model "v7.5")
=======================================

v7.5 = v7 + MULTI-SEED ENSEMBLE.

(Filename has no second dot - make_submission_v75.py - so the version map and
shell tooling don't choke on "v7.5.py". The model is referred to as v7.5.)

== WHAT v7.5 ADDS vs v7 ====================================================
v7 trains ONE LightGBM. v7.5 trains SEVERAL (default 5) and AVERAGES their
24-hour predictions. The point is tail-day robustness: on a hard day a single
model can make one unlucky large error; independent models rarely all blow up
the same way, so the average is steadier. Expected gain is modest (~20-40 MW
on the mean) and concentrated on the worst days - which is exactly where the
mean-MAE metric is won (your median ~1113 vs mean ~1298 = the tail).

  HONEST CAVEAT - this is NOT a free lunch, and it MUST be backtested before
  deploying. For different seeds to produce *diverse* models worth averaging,
  each member uses row + feature subsampling (subsample / colsample_bytree).
  That subsampling is itself a change from v7 (which uses all rows/features
  per tree): it regularizes each member but makes it individually weaker.
  Whether 5 subsampled members averaged beats 1 full v7 model is an EMPIRICAL
  question - run v7 vs v7.5 on the 60-day live-replica backtest and only
  deploy v7.5 if it wins. Until then v7 stays the shadow candidate.

Determinism (CR-2): the seed list is fixed (base 2026 -> 2026, 2027, ...), so
the ensemble is reproducible; the forecast CSV is bit-identical for a given
--seeds value. Subsampling uses these fixed seeds, so it does not break this.

Everything else is inherited from v7: v6 features + de_shortwave_radiation,
expanded data [2018-2019]U[2024-D-1], per-holiday + bridge, L2 objective,
direct Open-Meteo weather (so radiation is available), and the OPTIONAL
bias correction (--bias-correction, OFF by default - tested-and-rejected
over 60 days; see make_submission_v45.py / the v5 ablation).

Compliance: CR-1, CR-2, CR-3 (raise on missing weather/rows), CR-4
(numpy/pandas/lightgbm/sklearn/holidays/astral/spotforecast2_safe/requests).
Multiple LightGBMs are still only LightGBM - CR-4 clean.

== AUDIT LOG (chapter 01, AI Act Art. 12 / PR-4) ===========================
Same as v7: audit log on FIRST, library actions auto-log, weather fetch is
threaded with the logger (cache-hit vs fetch, 429s), and we log each ensemble
member as it is fitted plus the averaged result and member spread. Logs:
  Soheil_Files/logs/make_submission_v75/<name>_<date>_<time>.log

USAGE:
    cd E:\\DDMO\\challenge-leaderboard
    $env:PYTHONHASHSEED = "0"

    # v7.5 (5-model ensemble) - CANDIDATE, backtest before deploying:
    uv run python E:\\DDMO\\DDMO_Hot_Rod\\scripts\\Soheil_Files\\daily\\live\\make_submission_v75.py ^
        --team hot_rod --target-date 2026-06-02

    # ablation: single subsampled member (isolates subsampling from averaging):
    uv run python ...\\daily\\live\\make_submission_v75.py --team hot_rod ^
        --target-date 2026-06-02 --seeds 1

    # shadow folder:
    ... make_submission_v75.py --team hot_rod --target-date 2026-06-02 ^
        --repo-root E:\\DDMO\\DDMO_Hot_Rod\\shadow
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import holidays as holidays_lib
from lightgbm import LGBMRegressor

from spotforecast2_safe import LinearlyInterpolateTS
from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home
from spotforecast2_safe.downloader.entsoe import download_new_data
from spotforecast2_safe.manager.logger import setup_logging


# =============================================================================
# CONFIG
# =============================================================================
DATA_PULL_START = "2018-01-01"
HDD_BASE = 15.0
CDD_BASE = 22.0
SUN_LAT, SUN_LON = 51.0, 10.0

KEEP_WINDOWS = [
    ("2018-01-01", "2019-12-31"),
    ("2024-01-01", None),
]

# --- multi-seed ensemble (v7.5's defining change) ---
# Each member subsamples rows + features so different seeds give DIVERSE trees
# worth averaging. Subsampling is a change from v7 -> v7.5 MUST be backtested.
# Params below are the tune_v75 winner L31_M40_S0.7_C0.9 (two-fold-robust),
# CONFIRMED at 5 seeds on backtest_v75.py: vs the old 0.8/0.8/leaves=23 defaults
# it lowered mean (1240.6->1231.5), median (1124->1071), worst-10 (2383->2299)
# and worst-day (4513->4387) - better in body AND tail. Prior defaults in [].
ENSEMBLE_SEED_BASE = 2026
DEFAULT_N_SEEDS    = 5
SUBSAMPLE          = 0.7   # bagging_fraction per member   [was 0.8]
SUBSAMPLE_FREQ     = 1     # bag every iteration
COLSAMPLE_BYTREE   = 0.9   # feature_fraction per member   [was 0.8]

LGBM_BASE = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,             # tuned [was 23]
    min_child_samples=40,
    subsample=SUBSAMPLE,
    subsample_freq=SUBSAMPLE_FREQ,
    colsample_bytree=COLSAMPLE_BYTREE,
    verbose=-1,
    n_jobs=-1,
)

# --- optional bias correction (OFF by default; inherited from v7) ---
BIAS_WINDOW_DAYS = 14
BIAS_CLIP_MW     = 2500.0

# --- audit log location ---
LOG_ROOT    = Path(os.environ.get("HOTROD_LOG_DIR", "logs"))  # portable: override via HOTROD_LOG_DIR
SCRIPT_NAME = "make_submission_v75"

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

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS  = "temperature_2m,cloud_cover,wind_speed_10m,shortwave_radiation"
WX_RAW_COLS  = ["temperature_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation"]

WX_HIST_CACHE_DIR     = Path("_cache_weather_solar")
WX_FORECAST_CACHE_DIR = Path("_cache_weather_forecasts_solar")
WX_HIST_CACHE_DIR.mkdir(exist_ok=True)
WX_FORECAST_CACHE_DIR.mkdir(exist_ok=True)
FORECAST_WINDOW_DAYS_BACK = 7


# =============================================================================
def compute_keep_mask(idx, cutoff_ts):
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


def build_sunlight(idx):
    try:
        from astral import LocationInfo
        from astral.sun import sun
    except ImportError as e:
        raise SystemExit(
            "[fatal] astral not importable. Install with: uv add astral\n"
            f"(original error: {e})"
        )
    loc = LocationInfo(name="DE", region="DE", timezone="UTC",
                       latitude=SUN_LAT, longitude=SUN_LON)
    daylight, sr_h, ss_h = {}, {}, {}
    for d in pd.Index(idx.date).unique():
        try:
            s = sun(loc.observer, date=d, tzinfo="UTC")
            sr, ss = s["sunrise"], s["sunset"]
            sr_h[d] = sr.hour + sr.minute / 60.0
            ss_h[d] = ss.hour + ss.minute / 60.0
            daylight[d] = (ss - sr).total_seconds() / 3600.0
        except Exception:
            sr_h[d] = ss_h[d] = daylight[d] = np.nan
    dates = pd.Index(idx.date)
    sr_arr = dates.map(sr_h).astype(float).to_numpy()
    ss_arr = dates.map(ss_h).astype(float).to_numpy()
    dl_arr = dates.map(daylight).astype(float).to_numpy()
    hour_arr = idx.hour.to_numpy()
    is_day = ((hour_arr >= sr_arr) & (hour_arr < ss_arr)).astype(int)
    return dl_arr, is_day


def build_features(y, weather):
    idx = y.index
    df = pd.DataFrame(index=idx)

    df["lag_168"] = y.shift(168)
    df["lag_336"] = y.shift(336)
    df["lag_504"] = y.shift(504)
    df["lag_672"] = y.shift(672)

    base = y.shift(168)
    df["roll_mean_last_week"] = base.rolling(168).mean()
    df["roll_max_168"]        = base.rolling(168).max()
    df["roll_min_168"]        = base.rolling(168).min()
    df["roll_std_168"]        = base.rolling(168).std()

    df["hour"]       = idx.hour
    df["dayofweek"]  = idx.dayofweek
    df["month"]      = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    df["hour_sin"]  = np.sin(2*np.pi*idx.hour/24.0)
    df["hour_cos"]  = np.cos(2*np.pi*idx.hour/24.0)
    df["dow_sin"]   = np.sin(2*np.pi*idx.dayofweek/7.0)
    df["dow_cos"]   = np.cos(2*np.pi*idx.dayofweek/7.0)
    df["month_sin"] = np.sin(2*np.pi*(idx.month-1)/12.0)
    df["month_cos"] = np.cos(2*np.pi*(idx.month-1)/12.0)

    dl, isday = build_sunlight(idx)
    df["daylight_hours"] = dl
    df["is_daylight"]    = isday

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


# =============================================================================
# WEATHER  (direct Open-Meteo, includes shortwave_radiation, throttled)
# =============================================================================
def _to_utc(ts):
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def fetch_openmeteo(base_url, name, lat, lon, start, end, cache_path, logger=None):
    s, e = _to_utc(start), _to_utc(end)
    if cache_path.exists():
        c = pd.read_parquet(cache_path)
        if not c.empty and c.index.min() <= s and c.index.max() >= e:
            if logger:
                logger.info(f"weather cache hit | name={name} endpoint={base_url} "
                            f"rows={len(c)}")
            return c
    params = {
        "latitude": lat, "longitude": lon, "hourly": HOURLY_VARS,
        "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    last = None
    for attempt in range(4):
        try:
            r = requests.get(base_url, params=params, timeout=120)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"[wx]   {name}: 429, waiting {wait}s ...")
                if logger:
                    logger.warning(f"weather rate-limited | name={name} "
                                   f"attempt={attempt+1} wait_s={wait}")
                time.sleep(wait); continue
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame({k: h[k] for k in WX_RAW_COLS},
                              index=pd.to_datetime(h["time"], utc=True))
            df.to_parquet(cache_path)
            if logger:
                logger.info(f"weather fetched | name={name} endpoint={base_url} "
                            f"rows={len(df)}")
            return df
        except Exception as ex:
            last = ex; time.sleep(5)
    raise RuntimeError(f"[wx] {name} fetch failed ({base_url}): {last}")


def fetch_weather_blended(data_start, target_end, target_date_str, logger=None):
    cutoff = (pd.Timestamp.now("UTC").normalize()
              - pd.Timedelta(days=FORECAST_WINDOW_DAYS_BACK))
    fc_dir = WX_FORECAST_CACHE_DIR / target_date_str
    fc_dir.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"weather fetch start | archive={ARCHIVE_URL} "
                    f"forecast={FORECAST_URL} cutoff={cutoff.date()} "
                    f"cities={len(CITIES)} vars={HOURLY_VARS}")

    total_pop = sum(c[3] for c in CITIES)
    weighted = None
    for i, (name, lat, lon, pop) in enumerate(CITIES):
        if i > 0:
            print("[wx]   throttling 5s to avoid rate limit ...")
            time.sleep(5)
        w = pop / total_pop
        print(f"[wx]   {name:<10s}  weight={w:.2%}")
        hist = fetch_openmeteo(ARCHIVE_URL, name, lat, lon, data_start, cutoff,
                               WX_HIST_CACHE_DIR / f"solar_hist_{name}.parquet",
                               logger=logger)
        fc   = fetch_openmeteo(FORECAST_URL, name, lat, lon, cutoff, target_end,
                               fc_dir / f"{name}.parquet", logger=logger)
        combined = pd.concat([hist, fc.loc[~fc.index.isin(hist.index)]]).sort_index()
        wc = combined * w
        weighted = wc if weighted is None else weighted.add(wc, fill_value=0)
    weighted = weighted.rename(columns=lambda c: f"de_{c}")
    print(f"[wx] forecast snapshot dir: {fc_dir.resolve()}")
    if logger:
        logger.info(f"weather forecast snapshot | dir={fc_dir.resolve()}")
    return weighted


def load_load_series(start_dt, end_dt):
    print(f"[load] fetching ENTSO-E DE load "
          f"{start_dt.date()} -> {end_dt.date()} ...")
    api_key = os.environ["ENTSOE_API_KEY"]
    download_new_data(
        api_key=api_key, country_code="DE",
        start=start_dt.strftime("%Y%m%d%H%M"),
        end=end_dt.strftime("%Y%m%d%H%M"),
        force=False,
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
# BIAS ESTIMATION (optional; single subsampled member - bias is just a scalar)
# =============================================================================
def estimate_bias(features, y_full, train_end_ts, logger=None):
    holdout_start = (train_end_ts
                     - pd.Timedelta(days=BIAS_WINDOW_DAYS)
                     + pd.Timedelta(hours=1))
    bias_train_cutoff = holdout_start - pd.Timedelta(hours=1)

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
    m_bias = LGBMRegressor(**LGBM_BASE, random_state=ENSEMBLE_SEED_BASE)
    m_bias.fit(X_bt, y_bt)

    holdout_mask = (
        (features.index >= holdout_start)
        & (features.index <= train_end_ts)
        & clean
        & (features["is_any_holiday"].to_numpy() == 0)
    )
    X_ho = features.loc[holdout_mask]
    y_ho = y_full.loc[holdout_mask]
    if len(X_ho) < 24 * 5:
        print(f"[bias] WARNING: only {len(X_ho)} non-holiday holdout hours.")
        if logger:
            logger.warning(f"bias holdout thin | non_holiday_hours={len(X_ho)}")

    pred_ho = m_bias.predict(X_ho)
    residuals = y_ho.values - pred_ho
    bias_med = float(np.median(residuals))
    bias_mean = float(np.mean(residuals))
    print(f"[bias] holdout {holdout_start.date()} -> {train_end_ts.date()} "
          f"({len(X_ho)} hrs): median {bias_med:+.1f} MW (mean {bias_mean:+.1f})")
    if logger:
        logger.info(f"bias holdout | median={bias_med:+.1f} mean={bias_mean:+.1f} "
                    f"hrs={len(X_ho)}")

    bias_applied = float(np.clip(bias_med, -BIAS_CLIP_MW, BIAS_CLIP_MW))
    if bias_applied != bias_med:
        print(f"[bias] WARNING: clipped {bias_med:+.1f} -> {bias_applied:+.1f}")
        if logger:
            logger.warning(f"bias clipped | from={bias_med:+.1f} to={bias_applied:+.1f}")
    return bias_applied


# =============================================================================
# MAIN  (v7 features + multi-seed ensemble; bias optional, OFF by default)
# =============================================================================
def make_submission(team, target_date, repo_root, apply_bias=False,
                    n_seeds=DEFAULT_N_SEEDS):
    log_dir = LOG_ROOT / SCRIPT_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logging(level=logging.INFO, log_dir=log_dir)
    logger.info(f"audit run start | script={SCRIPT_NAME}.py model=v7.5 "
                f"team={team} target_date={target_date} apply_bias={apply_bias} "
                f"n_seeds={n_seeds}")
    logger.info(f"config | lgbm_base={LGBM_BASE} keep_windows={KEEP_WINDOWS} "
                f"data_pull_start={DATA_PULL_START} hourly_vars={HOURLY_VARS} "
                f"ensemble_seed_base={ENSEMBLE_SEED_BASE} n_seeds={n_seeds} "
                f"subsample={SUBSAMPLE} colsample={COLSAMPLE_BYTREE} "
                f"apply_bias={apply_bias}")
    if log_path:
        print(f"[log] audit log: {log_path}")
    try:
        return _run_submission(team, target_date, repo_root, apply_bias, n_seeds, logger)
    except BaseException:
        logger.exception("audit run FAILED")
        raise


def _run_submission(team, target_date, repo_root, apply_bias, n_seeds, logger):
    if n_seeds < 1:
        raise SystemExit(f"[fatal] --seeds must be >= 1, got {n_seeds}")

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

    print("[wx] fetching multi-city weather + radiation (2018 onward + forecast) ...")
    weather = fetch_weather_blended(full_index[0], target_hours_end, target_date,
                                    logger=logger)
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
    if "de_shortwave_radiation" not in features.columns:
        raise SystemExit("[fatal] de_shortwave_radiation missing - weather fetch "
                         "did not return radiation.")

    X_target = features.loc[target_hours_start:target_hours_end]
    if len(X_target) != 24:
        raise SystemExit(f"[fatal] expected 24 target rows, got {len(X_target)}")
    if X_target.isna().any().any():
        miss = X_target.columns[X_target.isna().any()].tolist()
        raise SystemExit(f"[fatal] target rows missing feature(s): {miss}")

    # --- optional bias correction (OFF by default) ---
    bias = 0.0
    if apply_bias:
        print("[bias] --bias-correction ENABLED (experimental; backtest first)")
        bias = estimate_bias(features, y_full, train_end_ts, logger=logger)
        logger.info(f"bias correction | applied={bias:+.1f} MW")
    else:
        logger.info("bias correction | disabled (default)")

    keep_mask = compute_keep_mask(features.index, train_end_ts)
    train_mask = (
        keep_mask
        & y_full.notna().to_numpy()
        & features.notna().all(axis=1).to_numpy()
    )
    X_train = features.loc[train_mask]
    y_train = y_full.loc[train_mask]
    print(f"[train] training rows: {len(X_train):,}")
    if len(X_train) < 5000:
        raise SystemExit(f"[fatal] only {len(X_train)} clean training rows")

    # --- MULTI-SEED ENSEMBLE: train n_seeds members, average predictions ---
    seeds = [ENSEMBLE_SEED_BASE + i for i in range(n_seeds)]
    print(f"[train] fitting {n_seeds}-model ensemble "
          f"(seeds {seeds[0]}..{seeds[-1]}, subsample={SUBSAMPLE}, "
          f"colsample={COLSAMPLE_BYTREE}) ...")
    logger.info(f"ensemble training | n_seeds={n_seeds} seeds={seeds} "
                f"rows={len(X_train)}")
    member_preds = []
    for s in seeds:
        m = LGBMRegressor(**LGBM_BASE, random_state=s)
        m.fit(X_train, y_train)
        member_preds.append(m.predict(X_target))
        logger.info(f"ensemble member fitted | seed={s}")
    member_preds = np.vstack(member_preds)          # (n_seeds, 24)
    y_pred_raw = member_preds.mean(axis=0)          # average across members
    spread = float(member_preds.std(axis=0).mean()) # mean cross-member std (tail signal)
    print(f"[train] ensemble averaged ({n_seeds} members); "
          f"mean cross-member spread = {spread:.1f} MW")
    logger.info(f"ensemble averaged | n_seeds={n_seeds} member_spread_mean={spread:.1f}")

    y_pred = y_pred_raw + bias
    if (y_pred <= 0).any() or np.isnan(y_pred).any():
        raise SystemExit(f"[fatal] forecast invalid: {y_pred}")

    logger.info(f"forecast | raw_mean={y_pred_raw.mean():.1f} "
                f"final_mean={y_pred.mean():.1f} min={y_pred.min():.1f} "
                f"max={y_pred.max():.1f} bias={bias:+.1f} spread={spread:.1f}")

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

    print(f"\n[ok] forecast summary (v7.5 = v7 + {n_seeds}-model ensemble):")
    if apply_bias:
        print(f"     bias correction applied: {bias:+.1f} MW")
        print(f"     raw   mean = {y_pred_raw.mean():>8.1f} MW")
    print(f"     final mean = {y_pred.mean():>8.1f} MW")
    print(f"     min  = {y_pred.min():>8.1f} MW")
    print(f"     max  = {y_pred.max():>8.1f} MW")
    print(f"     cross-member spread (mean) = {spread:>6.1f} MW")
    logger.info(f"submission written | path={out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS,
                        help="number of LightGBM members to average (default 5; "
                             "1 = single subsampled model, useful as an ablation)")
    parser.add_argument("--bias-correction", action="store_true",
                        help="ENABLE the v45-style bias correction (OFF by default; "
                             "tested-and-rejected over 60 days - backtest first).")
    args = parser.parse_args()
    path = make_submission(args.team, args.target_date, args.repo_root.resolve(),
                           apply_bias=args.bias_correction, n_seeds=args.seeds)
    print(f"\nSubmission written: {path}")