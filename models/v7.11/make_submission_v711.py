"""
make_submission_v711.py   (model "v7.11")
=========================================

**LIVE MODEL** since 2026-07-12 (first live target day 2026-07-13). Promoted
from the v7.10s shadow arm by owner decision (Soheil, 2026-07-12) AHEAD of the
usual ~7-scored-shadow-day gate -- 2 shadow CSVs existed, none scored yet;
the offline evidence is Erik's 165-day feature backtest (see the card §3/§6).
v7.10 retired the same day (script moved to daily/candidates/); **v7.9 was
reinstated as the FIRST backup** (owner decision -- model_card_v79.md §9c;
chain v7.9 -> v7.5 -> v4.5). Promotion record: model_cards/model_card_v711.md.
Provenance: shadow-tested 2026-07-11..07-12 under the arm label "v7.10s"
(script make_submission_v710s.py; shadow CSVs remain in shadow/v710s/, audit
logs in logs/make_submission_v710s/).

    v7.11 = deployed v7.10 (wxnl) + Erik's `solar_elevation`
          = v7.5 base + nowcast lag_24/lag_48 + multi-NWP mean weather
            + exact per-hour sun elevation angle -- NO wxscen, NO tuned params

Attribution arms still running nightly against this model:

    arm            wxscen   solar_elevation
    v7.11 (live)     no          yes
    v7.10e           yes         yes        <- v7.10e - v7.11 = wxscen alone
    v7.9.5 (reverse-shadow of the retired v7.9 line; runs until the 5-seed
            holdout lands)

The `solar_elevation` block is byte-for-byte Erik's v7.10e implementation
(astral.sun.elevation at the DE centroid, clipped to 0 at night, computed from
the timestamp only -- deterministic, CR-2, no publication-state dependency; it
feeds both the main ensemble and the nowcaster via build_features_v75).
Everything else, including the v7.9.5 hour-level corrupt-input guard, is the
previously-deployed v7.10 build unchanged. Card: model_card_v711.md.

--- v7.10 HEADER BELOW (the base of v7.11; v7.10 retired 2026-07-12) ----------

v7.10 is Henning's wxnl structure as a daily submission:

    v7.10 = v7.9 WITHOUT recent_offset
            = v7.5 base + nowcast-imputed lag_24/lag_48 + multi-NWP weather

Evidence: results/benchmarking/SOHEIL_BENCHMARK_SUMMARY.md (holdout
2025-07-01..2026-05-31: wxnl beat v79 on mean and tail, pub_lag 24 and 48,
1-seed) plus 13 clean live-shadow days vs the live arm (2026-06-25..07-09,
mean -6.2 MW, rolling-7 better in 6/7 windows) -- see model_card_v710.md.

== LIVE BUILD: HOUR-LEVEL CORRUPT-INPUT GUARD (ported from v7.9.5) ==========
The shadow arm ingested corrupt ENTSO-E input on 2026-06-27/28 that the live
QC pipeline excluded (see the incident analysis in the model card). The live
build therefore carries the v7.9.5 machinery unchanged:
  * BAD_GROUND_TRUTH_HOURS / --exclude-hours : NaN ONLY corrupt hours (keep
    the rest of the day); feature lags patched from clean prior weeks.
  * AUTO detection (physical bounds + isolated spike + profile deviation),
    disable with --no-auto-detect (the daily_submit_qc.ps1 flow does this and
    passes the human-confirmed hours instead).
With no flagged/passed hours this guard is a NO-OP: the math is byte-for-byte
the shadow-validated v7.10 arm.

--- ORIGINAL v7.9 HEADER BELOW (mechanics unchanged except as noted) ----------

v7.9 = v7.5 + RECENT-REGIME SIGNALS + MULTI-NWP TARGET WEATHER.

(Filename has no second dot, same convention as make_submission_v75.py.)

== WHAT v7.9 ADDS vs v7.5 ==================================================
Three components, validated TOGETHER on the full-year 5-seed live-replica
backtest (backtest_v75_vs_v79.py, 2024-07-01..2025-06-30, 364 days):

  paired vs v7.5: mean +110.9 MW (95% block-CI [+63,+168], excludes zero),
  worst-10 -311, rolling-7-worst -315, worst day flat. Decomposition showed
  the three components are COMPLEMENTARY (each pair fails the tail gate;
  the triple passes with margin) - see docs/v79_design.md.

  1. recent_offset  : trailing 7-day mean of (load - load_1wk_ago), evaluated
                      at the publication anchor. A settled-only regime-level
                      feature (NOT the rejected output-bias correction).
  2. lag_24 / lag_48: short lags made usable by NOWCAST IMPUTATION - an
                      auxiliary LightGBM (v7.5 features + raw lag_48/72)
                      predicts the unpublished recent hours; lags use the
                      real actual where published, else the nowcast.
  3. multi-NWP mean : the TARGET DAY's weather is the population-weighted
                      mean of icon_seamless + ecmwf_ifs025 + gfs_seamless
                      instead of a single best-match forecast. (History and
                      the recent gap stay exactly as v7.5 - validated config.)

== LIVE CENSORING (the part the backtest cannot test - CR-3 / Art. 10) =====
For every row t the publication anchor is
    anchor(t) = min( floor(t to day) - PUB_LAG_HOURS,  last published actual )
On historical training rows this reduces to the fixed rule used in the
backtest (floor - 24h), so there is NO train/serve skew; on the target day it
automatically censors against the ACTUAL last-published ENTSO-E timestamp, so
a late publication day just means more nowcast hours, never a silent leak.
lag_48 is censored-and-imputed by the same rule (live, (D-2, late hours) may
not be published yet even though the settled backtest always had them).

Bias correction is REMOVED entirely (rejected twice over a full year).

Compliance: CR-1, CR-2 (seed base 2026, snapshot caches, deterministic),
CR-3 (raise on missing weather/rows/features), CR-4 (numpy/pandas/lightgbm/
sklearn/holidays/astral/spotforecast2_safe/requests - unchanged stack).

== DEPLOYMENT STATUS =======================================================
LIVE (v7.11, since 2026-07-12). Routine nightly use goes through the QC-gated
wrapper, which scans for corrupt hours, asks the operator to confirm the
exclusion list, then runs this script with --no-auto-detect (auto-detection
remains available for unattended/what-if runs):

    powershell -ExecutionPolicy Bypass -File `
      E:\\DDMO\\DDMO_Hot_Rod\\scripts\\Soheil_Files\\daily\\daily_submit_qc.ps1 -Date 2026-07-13

Direct (non-QC) invocation, e.g. for a re-run or a what-if:

    cd E:\\DDMO\\challenge-leaderboard
    $env:PYTHONHASHSEED = "0"
    uv run python E:\\DDMO\\DDMO_Hot_Rod\\scripts\\Soheil_Files\\daily\\live\\make_submission_v711.py ^
        --team hot_rod --target-date 2026-07-13
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
# CONFIG (v7.5-identical unless marked NEW)
# =============================================================================
DATA_PULL_START = "2018-01-01"
HDD_BASE = 15.0
CDD_BASE = 22.0
SUN_LAT, SUN_LON = 51.0, 10.0

KEEP_WINDOWS = [
    ("2018-01-01", "2019-12-31"),
    ("2024-01-01", None),
]

ENSEMBLE_SEED_BASE = 2026
DEFAULT_N_SEEDS    = 5
LGBM_BASE = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=-1, num_leaves=31,
    min_child_samples=40, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.9, verbose=-1, n_jobs=-1,
)

# --- NEW: v7.9 component config (matches the validated backtest exactly) ---
PUB_LAG_HOURS   = 24       # fixed-rule part of the anchor (backtest value)
NOWCAST_TREES   = 800      # auxiliary nowcaster (1 model, seed 2026)
NWP_MODELS      = ["icon_seamless", "ecmwf_ifs025", "gfs_seamless"]

# --- bad ground-truth days (added 2026-06-10) -------------------------------
# ENTSO-E actuals for these dates were corrupted. They are set to NaN after the
# load fetch so they cannot feed lag_168/336/504/672, the roll_* windows, the
# recency offset, the nowcaster, or training targets. LightGBM handles the
# resulting NaNs natively at predict time (CR-3: explicit, logged exclusion --
# not silent imputation). Maintain this list; extend via --exclude-days,
# UN-exclude per run via --include-days (after verifying with check_days.py).
# !! ENTSO-E DOES correct retroactively: the 06-03/04 corruption was fixed
# upstream ~1 week later (seen 2026-06-11). BUT the local cache is pulled with
# force=False, so corrections only arrive after a forced refresh -- run
# daily/check_days.py to refresh + verify BEFORE removing a day from this
# list. Every run below also re-checks each excluded day against its 4-week
# profile and says so if it now looks clean.
# 2026-06-11: field-wide morning collapse (06-11h, 46.8 GW on a working
# Thursday, +6.4 GW noon snapback; ENTSO-E's own forecast off by 14.3 GW).
# Weather ruled out (check_weather_day 2026-06-12: radiation surprise -34 W/m2).
# 06-03/06-04 REMOVED 2026-06-12: ENTSO-E corrected them upstream; verified
# CLEAN after forced refresh (06-04 is Fronleichnam -- its corrected data is
# holiday-shaped and real). 06-11 stays: the corrupted values are unchanged;
# the earlier "CLEAN" verdict was a holiday-contaminated-profile artifact.
# 2026-06-13 (Soheil): day exclusion DISABLED -- every published day is used
# raw, with NO NaN-ing and NO weekly-persistence lag patch (that patch distorted
# lag sourcing, e.g. the 06-12 morning ramp). ENTSO-E corrects corrupt days
# retroactively and the leaderboard re-scores against the corrected series, so
# excluding them on our side bought little and added distortion. To re-enable,
# list days here again (the machinery below is unchanged, just fed an empty list).
BAD_GROUND_TRUTH_DAYS = []

# HOUR-LEVEL corruption map {'YYYY-MM-DD': [utc_hours]}. NaNs ONLY these
# hours (keeps the rest of the day) -- prefer this over the whole-day list above.
# Known cases: 2026-06-10 -> [5]; 2026-06-11 -> [6,7,8,9,10,11].
# (Ported from v7.9.1/v7.9.5 into the live v7.10 build, 2026-07-10.)
BAD_GROUND_TRUTH_HOURS = {}

# AUTO corrupt-hour detection thresholds (conservative, holiday-aware; from v7.9.5).
AUTO_SCAN_DAYS = 14        # only scan the recent window
AUTO_DEV_MW    = 8000.0    # |load - 6-week holiday-robust profile| threshold
AUTO_RUN_H     = 2         # >= this many CONSECUTIVE suspect hours to act

LOG_ROOT    = Path(os.environ.get("HOTROD_LOG_DIR", "logs"))  # portable: override via HOTROD_LOG_DIR
SCRIPT_NAME = "make_submission_v711"

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
WX_DE_COLS   = [f"de_{c}" for c in WX_RAW_COLS] + ["de_hdd", "de_cdd"]

WX_HIST_CACHE_DIR     = Path("_cache_weather_solar")          # shared with v7.5
WX_FORECAST_CACHE_DIR = Path("_cache_weather_forecasts_v79")  # NEW (per-model snapshots)
WX_HIST_CACHE_DIR.mkdir(exist_ok=True)
WX_FORECAST_CACHE_DIR.mkdir(exist_ok=True)
FORECAST_WINDOW_DAYS_BACK = 7


# =============================================================================
# SHARED HELPERS (v7.5-identical)
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
        from astral.sun import sun, elevation as sun_elevation
    except ImportError as e:
        raise SystemExit(f"[fatal] astral not importable: {e}")
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
    # solar elevation angle per hour (degrees above horizon, clipped to 0 at night)
    # rein deterministisch — kein Leakage-Risiko  (byte-for-byte Erik's v7.10e code)
    solar_elev = np.array([
        max(sun_elevation(loc.observer, dateandtime=ts.to_pydatetime()), 0.0)
        for ts in idx
    ])
    return dl_arr, is_day, solar_elev


def build_features_v75(y, weather):
    """The v7.5 feature matrix, unchanged (the validated base)."""
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
    dl, isday, solar_elev = build_sunlight(idx)
    df["daylight_hours"]  = dl
    df["is_daylight"]     = isday
    df["solar_elevation"] = solar_elev
    years = sorted({d.year for d in idx})
    hmap = build_holiday_map(years)
    all_hol = any_holiday_set(hmap)
    dates = pd.Index(idx.date)
    for hslug, hdates in hmap.items():
        df[f"h_{hslug}"] = dates.isin(hdates).astype(int)
    df["is_any_holiday"]        = dates.isin(all_hol).astype(int)
    df["is_day_before_holiday"] = pd.Index(idx.date + pd.Timedelta(days=1)).isin(all_hol).astype(int)
    df["is_day_after_holiday"]  = pd.Index(idx.date - pd.Timedelta(days=1)).isin(all_hol).astype(int)
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
# NEW: v7.9 COMPONENT 1 - recent-regime signals with LIVE censoring
# =============================================================================
def anchor_per_row(idx, last_actual_ts):
    """anchor(t) = min(floor(t)-PUB_LAG, last published actual). On historical
    rows this equals the backtest's fixed rule -> no train/serve skew; on the
    target day it censors against the REAL publication state.
    Pure pandas ops (the previous int64/asi8 roundtrip mis-parsed on some
    pandas versions -> epoch-garbage anchors -> 2026-06-05 incident)."""
    rule = idx.floor("D") - pd.Timedelta(hours=PUB_LAG_HOURS)
    cap = pd.Timestamp(last_actual_ts)
    return pd.DatetimeIndex(rule.where(rule <= cap, cap))


def holiday_robust_profile(y, allh, weeks=6, min_refs=3):
    """Same-weekday-hour MEDIAN over the last `weeks` weeks, ignoring reference
    days that are public holidays. Added 2026-06-12: the 4-week profile called
    the still-corrupted 06-11 'CLEAN' because 2 of its 4 reference Thursdays
    were Ascension (05-14) and Fronleichnam (06-04), dragging the profile
    ~4 GW low. Excluded days contribute NaN automatically (y is NaN there)."""
    refs = []
    for k in range(1, weeks + 1):
        r = y.shift(168 * k)
        src_hol = pd.Index((y.index - pd.Timedelta(hours=168 * k)).date).isin(allh)
        refs.append(r.mask(src_hol))
    R = pd.concat(refs, axis=1)
    return R.median(axis=1).where(R.notna().sum(axis=1) >= min_refs)


def censored_nowcast_lag(y, nowpred, k, last_actual_ts):
    """lag_k = real y(t-k) where (t-k) <= anchor(t), else the nowcaster's
    estimate. Same rule for training and target rows."""
    idx = y.index
    lagged_ts = idx - pd.Timedelta(hours=k)
    anchor = anchor_per_row(idx, last_actual_ts)
    avail = lagged_ts <= anchor
    return y.shift(k).where(avail, nowpred.shift(k))


def fit_nowcaster(features_v75, y_full, last_actual_ts, logger, y_lag_src=None):
    """Auxiliary LightGBM: v7.5 features + RAW lag_48/lag_72, trained on all
    clean rows up to the last actual; predicts every hour it can (incl. the
    unpublished gap, whose weather comes from the forecast blend).
    y_lag_src: series used for the raw lag FEATURES (the bad-day-patched
    y_feat); the fit TARGET stays y_full so excluded days are never learned."""
    if y_lag_src is None:
        y_lag_src = y_full
    nc = features_v75.copy()
    nc["lag_48_raw"] = y_lag_src.shift(48)
    nc["lag_72_raw"] = y_lag_src.shift(72)
    nc_cols = list(nc.columns)
    keep = compute_keep_mask(nc.index, last_actual_ts)
    fit_rows = (keep & y_full.notna().to_numpy()
                & nc.notna().all(axis=1).to_numpy()
                & (nc.index <= last_actual_ts))
    if int(fit_rows.sum()) < 5000:
        raise SystemExit(f"[fatal] nowcaster has only {int(fit_rows.sum())} rows")
    print(f"[nowcast] fitting aux model on {int(fit_rows.sum()):,} rows "
          f"x {len(nc_cols)} feats ({NOWCAST_TREES} trees) ...")
    logger.info(f"nowcaster fit | rows={int(fit_rows.sum())} cols={len(nc_cols)} "
                f"trees={NOWCAST_TREES} seed={ENSEMBLE_SEED_BASE}")
    m = LGBMRegressor(**{**LGBM_BASE, "n_estimators": NOWCAST_TREES},
                      random_state=ENSEMBLE_SEED_BASE)
    m.fit(nc.loc[fit_rows, nc_cols], y_full.loc[fit_rows])
    pred_ok = nc[nc_cols].notna().all(axis=1).to_numpy()
    nowpred = pd.Series(np.nan, index=y_full.index)
    nowpred.loc[pred_ok] = m.predict(nc.loc[pred_ok, nc_cols])
    n_gap = int(pred_ok[(nc.index > last_actual_ts)].sum())
    logger.info(f"nowcaster predicted | rows={int(pred_ok.sum())} gap_rows={n_gap}")
    return nowpred


# =============================================================================
# WEATHER (v7.5 mechanism + NEW per-model target-day fetch)
# =============================================================================
def _to_utc(ts):
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def fetch_openmeteo(base_url, name, lat, lon, start, end, cache_path,
                    logger=None, model=None):
    s, e = _to_utc(start), _to_utc(end)
    if cache_path.exists():
        c = pd.read_parquet(cache_path)
        if not c.empty and c.index.min() <= s and c.index.max() >= e:
            if logger:
                logger.info(f"weather cache hit | name={name} model={model or 'best'}")
            return c
    params = {
        "latitude": lat, "longitude": lon, "hourly": HOURLY_VARS,
        "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    if model:
        params["models"] = model
    last = None
    for attempt in range(4):
        try:
            r = requests.get(base_url, params=params, timeout=120)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"[wx]   {name}({model or 'best'}): 429, waiting {wait}s ...")
                time.sleep(wait); continue
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame({k: h[k] for k in WX_RAW_COLS},
                              index=pd.to_datetime(h["time"], utc=True))
            df.to_parquet(cache_path)
            if logger:
                logger.info(f"weather fetched | name={name} model={model or 'best'} "
                            f"rows={len(df)}")
            return df
        except Exception as ex:
            last = ex; time.sleep(5)
    raise RuntimeError(f"[wx] {name}({model}) fetch failed ({base_url}): {last}")


def fetch_weather_v79(data_start, target_start, target_end, target_date_str, logger):
    """v7.5's hist+best-match blend for history and the recent gap, PLUS the
    multi-model mean swapped onto the TARGET DAY only (validated config)."""
    cutoff = (pd.Timestamp.now("UTC").normalize()
              - pd.Timedelta(days=FORECAST_WINDOW_DAYS_BACK))
    fc_dir = WX_FORECAST_CACHE_DIR / target_date_str
    fc_dir.mkdir(parents=True, exist_ok=True)
    total_pop = sum(c[3] for c in CITIES)

    blended = None                      # hist + best-match (the v7.5 frame)
    mm = {m: None for m in NWP_MODELS}  # per-model target-window frames
    for i, (name, lat, lon, pop) in enumerate(CITIES):
        if i > 0:
            time.sleep(3)
        w = pop / total_pop
        print(f"[wx]   {name:<10s} weight={w:.2%}")
        hist = fetch_openmeteo(ARCHIVE_URL, name, lat, lon, data_start, cutoff,
                               WX_HIST_CACHE_DIR / f"solar_hist_{name}.parquet",
                               logger=logger)
        fc = fetch_openmeteo(FORECAST_URL, name, lat, lon, cutoff, target_end,
                             fc_dir / f"{name}_best.parquet", logger=logger)
        combined = pd.concat([hist, fc.loc[~fc.index.isin(hist.index)]]).sort_index()
        wc = combined * w
        blended = wc if blended is None else blended.add(wc, fill_value=0)
        for model in NWP_MODELS:
            time.sleep(2)
            fcm = fetch_openmeteo(FORECAST_URL, name, lat, lon, target_start,
                                  target_end, fc_dir / f"{name}_{model}.parquet",
                                  logger=logger, model=model)
            wcm = fcm * w
            mm[model] = wcm if mm[model] is None else mm[model].add(wcm, fill_value=0)

    blended = blended.rename(columns=lambda c: f"de_{c}")
    frames = []
    for model in NWP_MODELS:
        f = mm[model].rename(columns=lambda c: f"de_{c}")
        if f.isna().any().any():
            raise SystemExit(f"[fatal] NWP model {model} returned NaN for the "
                             f"target window (CR-3: refusing to impute)")
        frames.append(f)
    idx = frames[0].index
    for f in frames:
        idx = idx.intersection(f.index)
    mm_mean = sum(f.reindex(idx) for f in frames) / len(frames)

    # swap the multi-model mean onto the TARGET DAY rows only
    tgt_rows = blended.index.intersection(
        mm_mean.loc[target_start:target_end].index)
    if len(tgt_rows) < 24:
        raise SystemExit(f"[fatal] multi-model mean covers only {len(tgt_rows)} "
                         f"target hours (need 24)")
    blended.loc[tgt_rows, mm_mean.columns] = mm_mean.loc[tgt_rows].values
    # Per-member target-window frames are returned for v79 compatibility. v7.10
    # uses only the swapped multi-model mean, not scenario propagation.
    members = {m: f.reindex(idx) for m, f in zip(NWP_MODELS, frames)}
    print(f"[wx] target-day weather = mean({'+'.join(NWP_MODELS)}); "
          f"snapshot: {fc_dir.resolve()}")
    logger.info(f"multi-model swap | models={NWP_MODELS} target_rows={len(tgt_rows)} "
                f"snapshot={fc_dir.resolve()}")
    return blended, members, list(mm_mean.columns)


def load_load_series(start_dt, end_dt):
    print(f"[load] fetching ENTSO-E DE load {start_dt.date()} -> {end_dt.date()} ...")
    api_key = os.environ["ENTSOE_API_KEY"]
    download_new_data(api_key=api_key, country_code="DE",
                      start=start_dt.strftime("%Y%m%d%H%M"),
                      end=end_dt.strftime("%Y%m%d%H%M"), force=False)
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
# HOUR-LEVEL bad-data exclusion (NaN only corrupt hours; keep the rest).
# Ported byte-for-byte from make_submission_v795.py (v7.9.1/v7.9.5 machinery)
# into the live v7.10 build on 2026-07-10. No-op when no hours are flagged.
# =============================================================================
def parse_corrupt_hours(base_map, cli_spec=""):
    """Merge a {'YYYY-MM-DD': [hours]} map and a CLI string
    '2026-06-10:5,2026-06-11:6-11' into a sorted DatetimeIndex of UTC hours."""
    out = []
    def _add(dstr, hours):
        d0 = pd.Timestamp(dstr, tz="UTC")
        for h in hours:
            out.append(d0 + pd.Timedelta(hours=int(h)))
    for dstr, hours in (base_map or {}).items():
        _add(dstr, hours)
    for chunk in (cli_spec or "").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        dstr, hrs = chunk.split(":", 1)
        hs = set()
        for part in hrs.split("+"):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                hs |= set(range(int(a), int(b) + 1))
            elif part:
                hs.add(int(part))
        _add(dstr.strip(), sorted(hs))
    return pd.DatetimeIndex(sorted(set(out)))


def apply_hourly_exclusion(y_full, y_feat, corrupt_ts, allh, logger=None):
    """Hour-level bad-data handling (v7.9.1). For each corrupt UTC hour:
      * y_full -> NaN (out of training targets), keeping the day's OTHER good
        hours (unlike the whole-day BAD_GROUND_TRUTH_DAYS).
      * y_feat -> the clean same-hour value 1..4 weeks earlier (skip a source
        week that is itself corrupt/NaN or a public holiday), so no lag/roll
        column ever sees a NaN at predict (LightGBM reads a never-trained NaN as
        ZERO -> collapse; this patch is the guard).
    Operates on copies; returns (y_full, y_feat). Raises if a cell is unpatchable."""
    corrupt = pd.DatetimeIndex(sorted(set(corrupt_ts))).intersection(y_full.index)
    if len(corrupt) == 0:
        return y_full, y_feat
    y_full = y_full.copy()
    y_feat = y_feat.copy()
    orig = y_feat.copy()                      # clean source values (pre-NaN)
    corrupt_set = set(corrupt)
    n_real = int(orig.reindex(corrupt).notna().sum())
    y_full.loc[corrupt] = np.nan
    patched, unpatched = 0, []
    for t in corrupt:
        val = np.nan
        for k in (168, 336, 504, 672):
            src = t - pd.Timedelta(hours=k)
            if (src not in orig.index) or (src in corrupt_set) or (src.date() in allh):
                continue
            v = orig.get(src, np.nan)
            if pd.notna(v):
                val = v
                break
        if pd.isna(val):
            unpatched.append(t)
        else:
            y_feat.loc[t] = val
            patched += 1
    if unpatched:
        raise SystemExit(f"[fatal] {len(unpatched)} corrupt hour(s) unpatchable "
                         f"(no clean non-holiday same-hour source within 4 weeks): "
                         f"{[str(u) for u in unpatched[:5]]}")
    msg = (f"hour-level exclusion | corrupt_cells={len(corrupt)} "
           f"nan_in_target={n_real} feat_patched={patched}")
    print(f"[excl] {msg}")
    if logger:
        logger.info(msg)
    return y_full, y_feat


PHYS_LO_MW, PHYS_HI_MW, JUMP_MW = 15000.0, 90000.0, 20000.0


def detect_corrupt_hours(y_full, allh, last_actual_ts, scan_days=AUTO_SCAN_DAYS,
                         dev_mw=AUTO_DEV_MW, run_h=AUTO_RUN_H, logger=None):
    """v7.9.5 AUTO-detect -- TWO complementary rules, unioned:

      (A) PHYSICAL single-hour guard (added 2026-06-27 after the 06-25 16:00
          incident): any hour outside [PHYS_LO_MW, PHYS_HI_MW], or an ISOLATED
          spike (jump > JUMP_MW on BOTH sides). ANY duration -> catches a LONE
          impossible value (2026-06-25 16:00 = 1,124,902 MW) that rule (B) misses
          because one hour never reaches a run of run_h. Effectively zero false
          positives: real DE control-area load is ~35-84 GW (verified clean over
          8.5 yrs, 74k hrs -- only the known corruption flags).
      (B) PROFILE deviation (original): |load - holiday-robust 6-week profile| >
          dev_mw for >= run_h CONSECUTIVE hours, holiday days excluded.

    Returns the union as a DatetimeIndex (warn-logged, Art. 14). A false positive
    only swaps an input cell for last week's same hour (far safer than a clamp).
    For the routine daily flow prefer preflight_qc.py + daily_submit_qc.ps1, which
    surface these for human confirmation BEFORE submitting."""
    win_idx = y_full.loc[last_actual_ts - pd.Timedelta(days=scan_days):
                         last_actual_ts].index
    # (A) physical: out-of-bounds OR isolated spike (large jump on both sides).
    ywin = y_full.reindex(win_idx)
    dprev = (y_full - y_full.shift(1)).reindex(win_idx).abs()
    dnext = (y_full.shift(-1) - y_full).reindex(win_idx).abs()
    spike = (dprev > JUMP_MW) & (dnext > JUMP_MW)
    phys = pd.DatetimeIndex(
        win_idx[((ywin < PHYS_LO_MW) | (ywin > PHYS_HI_MW) | spike).fillna(False)])
    # (B) profile deviation, >= run_h consecutive, non-holiday.
    dev = (y_full - holiday_robust_profile(y_full, allh)).abs()
    sus = (dev.reindex(win_idx) > dev_mw).fillna(False)
    prof_flags = pd.DatetimeIndex([])
    if bool(sus.any()):
        runlen = sus.groupby((~sus).cumsum()).transform("sum")
        prof_flags = pd.DatetimeIndex(
            [t for t in sus.index[sus & (runlen >= run_h)] if t.date() not in allh])
    flagged = phys.union(prof_flags).sort_values()
    if len(flagged):
        m = (f"AUTO corrupt-hour detection | flagged={len(flagged)} "
             f"(physical={len(phys)}, profile={len(prof_flags)}) "
             f"first={flagged[0]} last={flagged[-1]}")
        print(f"[excl] {m}")
        if logger:
            logger.warning(m)
    return flagged


# =============================================================================
# MAIN
# =============================================================================
def make_submission(team, target_date, repo_root, n_seeds=DEFAULT_N_SEEDS,
                    exclude_days=None, exclude_hours="", auto_detect=True):
    if exclude_days is None:
        exclude_days = list(BAD_GROUND_TRUTH_DAYS)
    log_dir = LOG_ROOT / SCRIPT_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logging(level=logging.INFO, log_dir=log_dir)
    logger.info(f"audit run start | script={SCRIPT_NAME}.py model=v7.11 "
                f"team={team} target_date={target_date} n_seeds={n_seeds} "
                f"exclude_days={exclude_days} exclude_hours='{exclude_hours}' "
                f"auto_detect={auto_detect}")
    logger.info(f"config | lgbm_base={LGBM_BASE} keep_windows={KEEP_WINDOWS} "
                f"pub_lag_h={PUB_LAG_HOURS} nowcast_trees={NOWCAST_TREES} "
                f"nwp_models={NWP_MODELS} "
                f"seed_base={ENSEMBLE_SEED_BASE}")
    if log_path:
        print(f"[log] audit log: {log_path}")
    try:
        return _run_submission(team, target_date, repo_root, n_seeds, logger,
                               exclude_days, exclude_hours, auto_detect)
    except BaseException:
        logger.exception("audit run FAILED")
        raise


def _run_submission(team, target_date, repo_root, n_seeds, logger, exclude_days=(),
                    exclude_hours="", auto_detect=True):
    if n_seeds < 1:
        raise SystemExit(f"[fatal] --seeds must be >= 1, got {n_seeds}")

    target_hours_start = pd.Timestamp(target_date, tz="UTC")
    target_hours_end   = target_hours_start + pd.Timedelta(hours=23)
    train_end_ts       = target_hours_start - pd.Timedelta(hours=1)
    data_start_ts      = pd.Timestamp(DATA_PULL_START, tz="UTC")

    y_actual = load_load_series(data_start_ts, train_end_ts)
    last_actual_ts = y_actual.index[-1]
    pub_gap_h = int((target_hours_start - last_actual_ts) / pd.Timedelta(hours=1))
    print(f"[load] last published actual: {last_actual_ts} "
          f"({pub_gap_h}h before target start)")
    logger.info(f"publication state | last_actual={last_actual_ts} gap_h={pub_gap_h}")

    full_index = pd.date_range(y_actual.index[0], target_hours_end, freq="h", tz="UTC")
    y_full = pd.Series(np.nan, index=full_index, name="load")
    y_full.loc[y_actual.index] = y_actual.values

    # --- bad ground-truth exclusion (explicit, logged; never silent) ---------
    # Two series, two roles:
    #   y_full : TRAINING TARGET. Bad days -> NaN (never learned).
    #   y_feat : FEATURE SOURCING (lags/rolls/nowcaster). Bad days -> replaced by
    #            their own week-earlier values (weekly persistence). NaN must
    #            NEVER reach a lag/roll column at predict: LightGBM treats a
    #            never-seen-in-training NaN as ZERO (verified 2026-06-10:
    #            lag=NaN scored identically to lag=0 -> forecast collapse).
    # holiday calendar (needed by the profile checks + the patch chain)
    _years = list(range(y_full.index[0].year, y_full.index[-1].year + 1))
    allh = any_holiday_set(build_holiday_map(_years))

    # freshness check FIRST (on the un-excluded data): if an excluded day now
    # matches its HOLIDAY-AWARE 6-week profile, ENTSO-E has likely corrected
    # it upstream -- tell the operator instead of silently discarding good data.
    prof_chk = holiday_robust_profile(y_full, allh)
    for bad in exclude_days:
        d0 = pd.Timestamp(bad, tz="UTC")
        m = (y_full.index >= d0) & (y_full.index < d0 + pd.Timedelta(days=1))
        if not (m & y_full.notna().to_numpy()).any():
            continue
        if d0.date() in allh:
            print(f"[QC] NOTE: excluded day {bad} is itself a public holiday -- "
                  f"profile verdicts are unreliable for it; judge its shape "
                  f"manually (check_days.py prints the hours).")
            continue
        dev_b = (y_full[m] - prof_chk[m]).abs()
        sus_b = dev_b > 8000.0
        runs_b = sus_b.astype(int).groupby((~sus_b).cumsum()).cumsum()
        if not len(runs_b) or runs_b.empty or int(runs_b.max()) < 3:
            print(f"[QC] NOTE: excluded day {bad} looks CLEAN in the current "
                  f"data (max dev {dev_b.max():.0f} MW vs holiday-aware "
                  f"profile) -- ENTSO-E may have corrected it. Verify with "
                  f"check_days.py, then re-run with --include-days {bad}")
            logger.warning(f"excluded day looks clean | day={bad} "
                           f"maxdev={dev_b.max():.0f}")
    for bad in exclude_days:
        d0 = pd.Timestamp(bad, tz="UTC")
        m = (y_full.index >= d0) & (y_full.index < d0 + pd.Timedelta(days=1))
        n_bad = int((m & y_full.notna().to_numpy()).sum())
        if n_bad:
            y_full[m] = np.nan
            print(f"[excl] {bad}: {n_bad} corrupted actual hours -> NaN "
                  f"(removed from training targets + recency offset)")
            logger.info(f"bad ground truth excluded | day={bad} hours={n_bad}")
    y_feat = y_full.copy()
    for bad in exclude_days:
        d0 = pd.Timestamp(bad, tz="UTC")
        m = (y_feat.index >= d0) & (y_feat.index < d0 + pd.Timedelta(days=1))
        if not m.any():
            continue
        # weekly-persistence patch, CHAINED 1->2->3->4 weeks back. A fallback
        # week is skipped if its source day is excluded (NaN in y_full) OR a
        # public holiday (2026-06-12 lesson: with 06-04 back in the data, the
        # naive chain would have patched a normal Thursday with Fronleichnam's
        # holiday profile).
        repl, used = None, None
        for k in (168, 336, 504, 672):
            src_day = (d0 - pd.Timedelta(hours=k)).date()
            if src_day in allh:
                continue
            cand = y_full.shift(k)[m]
            if cand.isna().any():
                continue
            repl, used = cand, f"lag{k}"
            break
        if repl is None:
            raise SystemExit(f"[fatal] cannot patch excluded day {bad}: no clean "
                             f"non-holiday same-weekday profile within 4 weeks -- "
                             f"extend the chain before submitting.")
        y_feat[m] = repl.to_numpy()
        print(f"[excl] {bad}: lag/roll feature source patched ({used})")
        logger.info(f"feature source patched | day={bad} method={used}")

    # --- HOUR-LEVEL bad-data exclusion (additive; runs after the legacy
    # day-level block, which is a no-op while BAD_GROUND_TRUTH_DAYS/--exclude-days
    # are empty). NaNs ONLY corrupt hours and keeps the rest of each day. The
    # 2026-06-27/28 shadow blowouts are the reason this is in the LIVE build. ---
    corrupt_ts = parse_corrupt_hours(BAD_GROUND_TRUTH_HOURS, exclude_hours)
    if auto_detect:
        auto = detect_corrupt_hours(y_full, allh, last_actual_ts, logger=logger)
        corrupt_ts = corrupt_ts.union(auto)
    else:
        print("[excl] auto-detection OFF (--no-auto-detect): manual list only")
    if len(corrupt_ts):
        y_full, y_feat = apply_hourly_exclusion(y_full, y_feat, corrupt_ts, allh, logger)

    # --- automated bad-data TRIPWIRE (warn-only -- Art. 14, operator decides) -
    # Third corruption incident in 9 days (06-03/04, 06-10 05h, 06-11 morning):
    # flag any non-excluded day in the last QC_SCAN_DAYS where load deviates
    # from its 4-week same-weekday-hour MEDIAN profile by > QC_DEV_MW for
    # >= QC_RUN_H consecutive hours. Median across 4 weeks is robust to one
    # corrupted source week. NOTE: genuine holidays can fire this -- check the
    # calendar before excluding.
    QC_SCAN_DAYS, QC_DEV_MW, QC_RUN_H = 14, 8000.0, 3
    prof = holiday_robust_profile(y_full, allh)
    dev_qc = (y_full - prof).abs()
    sus = dev_qc.loc[last_actual_ts - pd.Timedelta(days=QC_SCAN_DAYS):
                     last_actual_ts] > QC_DEV_MW
    if sus.any():
        runs = sus.astype(int).groupby((~sus).cumsum()).cumsum()
        flagged = sorted({h.date() for h in runs[runs >= QC_RUN_H].index})
        flagged = [d for d in flagged
                   if str(d) not in {str(pd.Timestamp(b).date())
                                     for b in exclude_days}]
        hol_flagged = [d for d in flagged if d in allh]
        flagged = [d for d in flagged if d not in allh]
        if hol_flagged:
            print(f"[QC] note: {hol_flagged} deviate from the weekday profile "
                  f"but are public holidays -- expected, no action needed.")
        if flagged:
            print("!" * 72)
            print(f"[QC] TRIPWIRE: suspect ground truth on {flagged} -- load is "
                  f">{QC_DEV_MW:.0f} MW from its 4-week profile for "
                  f">={QC_RUN_H} consecutive hours.")
            print("[QC] If not a holiday/known event, re-run with "
                  f"--exclude-days {','.join(str(d) for d in flagged)}")
            print("!" * 72)
            logger.warning(f"QC tripwire | suspect_days={flagged}")

    print("[wx] fetching weather (hist + best-match + per-model target) ...")
    weather, _wx_members, _wx_swap_cols = fetch_weather_v79(
        full_index[0], target_hours_start, target_hours_end,
        target_date, logger=logger)
    weather = weather.reindex(full_index)
    n_missing = int(weather.isna().any(axis=1).sum())
    if n_missing > 0:
        print(f"[wx] WARNING: {n_missing} weather rows missing after reindex")
        logger.warning(f"weather missing | rows={n_missing}")

    print("[feat] building v7.5 base matrix (lag/roll source = y_feat) ...")
    features = build_features_v75(y_feat, weather)
    if "de_shortwave_radiation" not in features.columns:
        raise SystemExit("[fatal] de_shortwave_radiation missing")
    v75_cols = list(features.columns)

    # --- v7.10 / wxnl: recent_offset is intentionally NOT included. ---------
    # The benchmark arm was v75_cols + lag_24_nc + lag_48 with multi-NWP target
    # weather. Keeping the column out of both train and target features makes
    # this daily path match that arm instead of silently reproducing v7.9.
    print("[feat] recent_offset DISABLED (v7.11 = wxnl + solar_elevation)")

    nowpred = fit_nowcaster(features[v75_cols], y_full, last_actual_ts, logger,
                            y_lag_src=y_feat)
    features["lag_24_nc"] = censored_nowcast_lag(y_feat, nowpred, 24, last_actual_ts)
    features["lag_48"]    = censored_nowcast_lag(y_feat, nowpred, 48, last_actual_ts)

    tgt24 = features.loc[target_hours_start:target_hours_end, "lag_24_nc"]
    tgt48 = features.loc[target_hours_start:target_hours_end, "lag_48"]
    anchor_t = anchor_per_row(tgt24.index, last_actual_ts)
    # sanity guard (2026-06-05 incident): a target anchor must sit within a few
    # days of the target start -- anything else is a computation bug, not data
    if (anchor_t < target_hours_start - pd.Timedelta(days=7)).any() \
            or (anchor_t > last_actual_ts).any():
        raise SystemExit(f"[fatal] anchor sanity check failed: "
                         f"min={anchor_t.min()} max={anchor_t.max()} "
                         f"(last_actual={last_actual_ts})")
    n_now24 = int(((tgt24.index - pd.Timedelta(hours=24)) > anchor_t).sum())
    n_now48 = int(((tgt48.index - pd.Timedelta(hours=48)) > anchor_t).sum())
    print(f"[feat] target lags: lag_24 uses nowcast for {n_now24}/24 hrs, "
          f"lag_48 for {n_now48}/24 hrs")
    logger.info(f"target lag censoring | nowcast_24={n_now24} nowcast_48={n_now48}")
    print(f"[feat] shape: {features.shape}")
    logger.info(f"features built | shape={features.shape}")

    X_target = features.loc[target_hours_start:target_hours_end]
    if len(X_target) != 24:
        raise SystemExit(f"[fatal] expected 24 target rows, got {len(X_target)}")
    if X_target.isna().any().any():
        miss = X_target.columns[X_target.isna().any()].tolist()
        # FATAL, always. The y_feat patching above must keep every lag/roll
        # column dense; a NaN here would be silently read as ZERO by LightGBM
        # (no missing values exist in training -- verified 2026-06-10) and
        # collapse the forecast. If this fires, the exclusion patching has a
        # gap -- fix it, never bypass this check.
        raise SystemExit(f"[fatal] target rows missing feature(s): {miss}")

    keep_mask = compute_keep_mask(features.index, train_end_ts)
    train_mask = (keep_mask & y_full.notna().to_numpy()
                  & features.notna().all(axis=1).to_numpy())
    X_train = features.loc[train_mask]
    y_train = y_full.loc[train_mask]
    print(f"[train] training rows: {len(X_train):,}")
    if len(X_train) < 5000:
        raise SystemExit(f"[fatal] only {len(X_train)} clean training rows")

    seeds = [ENSEMBLE_SEED_BASE + i for i in range(n_seeds)]
    print(f"[train] fitting {n_seeds}-model v7.11 ensemble (seeds {seeds[0]}..{seeds[-1]}) ...")
    logger.info(f"ensemble training | n_seeds={n_seeds} seeds={seeds} rows={len(X_train)} params={LGBM_BASE}")
    fitted = []
    for s in seeds:
        m = LGBMRegressor(**LGBM_BASE, random_state=s)
        m.fit(X_train, y_train)
        fitted.append(m)
        logger.info(f"ensemble member fitted | seed={s}")

    # --- wxmean: predict ONCE on the multi-NWP mean target weather. -----------
    # This is the daily equivalent of the benchmark arm wxnl:
    #   arm_cols["wxnl"] = v75_cols + ["lag_24_nc", "lag_48"]
    #   arm_wx["wxnl"] = "mm"
    # Do not switch this to wxscen here; scenario propagation is a different arm.
    seed_preds = np.vstack([mdl.predict(X_target) for mdl in fitted])
    y_pred = seed_preds.mean(axis=0)
    spread = float(seed_preds.std(axis=0).mean())
    print(f"[train] wxmean: predicted once on mean({'+'.join(NWP_MODELS)}) "
          f"target weather (x{n_seeds} seeds); seed spread = {spread:.1f} MW")

    if (y_pred <= 0).any() or np.isnan(y_pred).any():
        raise SystemExit(f"[fatal] forecast invalid: {y_pred}")
    logger.info(f"forecast | mean={y_pred.mean():.1f} min={y_pred.min():.1f} "
                f"max={y_pred.max():.1f} spread={spread:.1f}")

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

    print(f"\n[ok] v7.11 forecast summary:")
    print(f"     mean = {y_pred.mean():>8.1f} MW   min = {y_pred.min():>8.1f}   "
          f"max = {y_pred.max():>8.1f}")
    print(f"     seed spread = {spread:>6.1f} MW")
    print(f"     nowcast hrs: lag24={n_now24}/24 lag48={n_now48}/24")
    logger.info(f"submission written | path={out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--exclude-days", type=str, default="",
                        help="extra bad-ground-truth days (comma-separated "
                             "YYYY-MM-DD), added to BAD_GROUND_TRUTH_DAYS")
    parser.add_argument("--include-days", type=str, default="",
                        help="days to UN-exclude for this run (after a CLEAN "
                             "verdict from check_days.py)")
    parser.add_argument("--exclude-hours", type=str, default="",
                        help="HOUR-level corruption, e.g. "
                             "'2026-06-10:5,2026-06-11:6-11'. NaNs only those "
                             "hours; merged with BAD_GROUND_TRUTH_HOURS and the "
                             "auto-detected hours.")
    parser.add_argument("--no-auto-detect", dest="auto_detect",
                        action="store_false",
                        help="disable auto corrupt-hour detection (=> manual "
                             "list only; the QC-gated daily flow uses this and "
                             "passes human-confirmed hours instead)")
    args = parser.parse_args()
    incl = {d.strip() for d in args.include_days.split(",") if d.strip()}
    excl = [d for d in (list(BAD_GROUND_TRUTH_DAYS)
                        + [d.strip() for d in args.exclude_days.split(",")
                           if d.strip()])
            if d not in incl]
    if incl:
        print(f"[excl] un-excluded for this run (operator decision): {sorted(incl)}")
    path = make_submission(args.team, args.target_date, args.repo_root.resolve(),
                           n_seeds=args.seeds, exclude_days=excl,
                           exclude_hours=args.exclude_hours,
                           auto_detect=args.auto_detect)
    print(f"\nSubmission written: {path}")
