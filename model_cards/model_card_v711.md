# Model Card — Voltcrown v7.11 (Team hot_rod)

Public model card for **Voltcrown v7.11**, the **final live model** of Team *hot_rod*
in the Lastprognose-Challenge SS26 (TH Köln, *Data-Driven Modeling and Optimization*,
summer 2026). It follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy and is written to stand on its own, so it can be read without reference to
the other cards in this folder.

> **Fully released.** With the challenge now concluded, v7.11 is released **in full** —
> this card discloses every hyperparameter, training window, feature and the complete
> nowcast, weather-blend and corrupt-input machinery, and the **runnable reference
> implementation is included** at
> [`../models/v7.11/make_submission_v711.py`](../models/v7.11/make_submission_v711.py).
> v7.11 was the team's **final live model**; the earlier v4.5, v7.5 and v7.9 cards in
> this folder document the models beneath it in the fallback chain.

## 1. Model details

| Field | Value |
| --- | --- |
| Name | Voltcrown v7.11 (Team hot_rod), `make_submission_v711.py` |
| Status | Final live model (open reference release) |
| Type | Deterministic, daily-retrained, **direct 24-hour multi-output** load forecaster: a **5-seed subsampled LightGBM** ensemble over the v7.5 feature base + nowcast-imputed short lags + multi-NWP mean weather + an exact per-hour solar-elevation angle, with an auxiliary LightGBM **nowcaster** |
| Target | ENTSO-E "DE Actual Total Load", MW, hourly — 24 values for target day D |
| Developed by | Team hot_rod, Lastprognose-Challenge SS26 |
| Score metric | Rolling **7-day mean MAE** (MW); lower is better |
| Cadence | Submit by **D−1 23:59 CET** for target day D |
| Determinism | Reproducible: `PYTHONHASHSEED=0`, seed base 2026 (ensemble 2026–2030, nowcaster 2026), snapshotted inputs |
| Reference code | [`../models/v7.11/make_submission_v711.py`](../models/v7.11/make_submission_v711.py) |
| Built on | `spotforecast2_safe` (T. Bartz-Beielstein) + LightGBM |

The runtime stack is a small, allow-listed set (numpy, pandas, scikit-learn,
lightgbm, holidays, astral, entsoe-py, requests, statsmodels, spotforecast2_safe).
Heavy or non-deterministic libraries (torch, tensorflow, optuna, xgboost, catboost,
matplotlib, plotly) are excluded to keep the dependency and CVE surface small and the
pipeline bit-reproducible. Every model in v7.11, including the nowcaster, is LightGBM.

## 2. Intended use and scope

v7.11 produces one daily competition submission: a 24-value MW forecast of the next
day's German load. It is the team's final deployed model — the v7.5 base extended by
the recent-regime and weather machinery of v7.9/v7.10 and one new deterministic
calendar feature — and is published here as a complete, reusable reference.

Out of scope: not a general forecasting library; DE control area only; point forecasts
with no calibrated uncertainty; must not use realised same-day exogenous data (that
would leak — §6) or be repurposed for grid operation without independent validation.

## 3. How it works (plain language)

The engine is a **gradient-boosted decision tree** ensemble (LightGBM): hundreds of
small yes/no-question trees built in sequence, each correcting earlier mistakes, summed
into a load estimate. The same configuration is trained under 5 random seeds with
row-and-feature subsampling and averaged. All 24 hours of day D are predicted at once
(each from its own known inputs) rather than recursively, so an early-hour error cannot
cascade. The model is retrained from scratch every day.

On top of that v7.5 core, v7.11 carries three inherited ideas and adds one. The base is
**blind to the most recent week** (its autoregressive lags are all ≥1 week back), so an
auxiliary LightGBM **nowcaster** fills the unpublished recent hours and lets a 1- and
2-day lag be used honestly; and because a single weather forecast is noisy, the target
day's weather is the **mean of three independent forecast models**. The new idea is one
calendar feature — the **exact solar elevation angle** each hour — giving the trees the
smooth within-day sun geometry the base model's binary daylight flags cannot express
(§4). Notably, v7.11 carries **neither** the output-bias correction of v4.5 **nor** the
`recent_offset` regime term of v7.9: both were dropped in the lineage as harmful or
unnecessary on the tail.

## 4. Technical specification (complete)

v7.11 = the deployed v7.10 chassis (a.k.a. "wxnl") + one new feature. In lineage terms:

> **v7.11 = v7.5 base + nowcast-imputed `lag_24`/`lag_48` + multi-NWP mean weather
> (`wxmean`) + exact per-hour `solar_elevation` — with the v7.9.5 hour-level
> corrupt-input guard, and with `recent_offset` and `wxscen` intentionally absent and
> no re-tuned parameters.**

### Estimator

The v7.5 configuration, unchanged: a LightGBM regressor trained under 5 seeds with
row+feature subsampling and averaged (no output-bias correction).

| Parameter | Value |
| --- | --- |
| `n_estimators` | 3000 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `min_child_samples` | 40 |
| `max_depth` | −1 (unbounded) |
| `subsample` / `subsample_freq` | 0.7 / 1 |
| `colsample_bytree` | 0.9 |
| `n_seeds` | 5 (seeds 2026–2030, averaged) |
| bias correction | **off** (rejected across the lineage) |

The design intent of the "ensemble" is **subsampling as regularization**, not
averaging: an ablation on the base model showed a single subsampled member captures
most of the gain and the extra seeds only steady the tail slightly. No parameters were
re-tuned for v7.11 — the values above are the tuned v7.5 set carried through unchanged.

### Training window

`KEEP_WINDOWS = [2018-01-01 … 2019-12-31] ∪ [2024-01-01 … D−1]`. Data is pulled from
2018-01-01; rows outside the kept windows (2020–2023, the pandemic and energy-crisis
years whose load behaviour is unlike the current regime) are dropped before training.
The model retrains daily, so the second window always extends to the hour before the
target day.

### Features

The v7.5 feature matrix (~51 columns) is the base:

- **Weekly autoregressive lags** (1–4 weeks back, always settled before submission):
  `lag_168`, `lag_336`, `lag_504`, `lag_672`; plus **rolling statistics** over a 168h
  window shifted 168h back: `roll_mean_last_week`, `roll_max_168`, `roll_min_168`,
  `roll_std_168`.
- **Calendar & sunlight (deterministic):** `hour`, `dayofweek`, `month`, `is_weekend`;
  cyclical `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`; ephemeris `daylight_hours`,
  `is_daylight`.
- **Holidays (all 16 Bundesländer union):** per-holiday `h_*` flags, `is_any_holiday`,
  `is_day_before_holiday`, `is_day_after_holiday`, `is_bridge_day`.
- **Weather (population-weighted over six cities — Berlin 5.0, Hamburg 3.4, Munich 3.0,
  Rhine-Ruhr 10.0, Frankfurt 5.8, Stuttgart 2.7):** `de_temperature_2m`,
  `de_cloud_cover`, `de_wind_speed_10m`, `de_shortwave_radiation`, and degree-days
  `de_hdd = max(0, 15 − T)`, `de_cdd = max(0, T − 22)`.

On top of that base, v7.11 adds three columns and swaps the target-day weather.

**Nowcast-imputed short lags (`lag_24_nc`, `lag_48`).** An auxiliary LightGBM nowcaster
(the v7.5 features plus raw `lag_48`/`lag_72`, 800 trees, seed 2026) predicts the
unpublished recent hours; the short lags then take the **real actual where published,
else the nowcast**, chosen per row by a **publication-time cutoff**
`anchor(t) = min(floor(t to day) − 24h, last published ENTSO-E actual)`. On historical
training rows this reduces to the fixed backtest rule (no train/serve skew); on the
target day it censors against the *actual* publication state — a late ENTSO-E day just
means more nowcast hours, never a silent leak.

**Multi-NWP mean target weather (`wxmean`).** For the target day only, the weather is
the population-weighted **mean of `icon_seamless` + `ecmwf_ifs025` + `gfs_seamless`**
instead of a single best-match forecast; training weather stays observed (live-replica
convention). Scenario propagation (`wxscen`) is **not** used — the model predicts once,
on the mean weather.

**Exact per-hour solar elevation (`solar_elevation`) — the one new feature.** The solar
elevation angle above the horizon, in degrees, at the DE centroid (51.0 °N, 10.0 °E),
computed per hour with `astral.sun.elevation` and clipped to 0 at night. It is added
**inside `build_features`**, so — because the nowcaster consumes the same base matrix —
it feeds **both** the main 5-seed ensemble and the auxiliary nowcaster. It is a
deterministic function of the timestamp alone: no new data dependency, no
publication-state sensitivity, and therefore **no leakage channel**. `astral` is
already on the allow-list, so it costs no new dependency.

**Explicitly absent.** `recent_offset` (the v7.9 trailing-regime term) is **not**
present — it was removed at v7.10 after a holdout showed it amplified the worst-week
tail. `wxscen` (weather-scenario propagation) is **not** present. No parameters were
re-tuned.

### Corrupt-input guard (inherited, byte-for-byte)

v7.11 carries the v7.9.5 **hour-level corrupt-input guard** unchanged: where corrupted
ENTSO-E actuals appear, it NaNs **only the corrupt hours** (not the whole day) in the
training target and patches the feature source for those cells from the clean same-hour
value 1–4 weeks earlier (skipping holiday or corrupt source weeks), so no lag or
rolling column ever sees a NaN at predict. Corrupt hours can be supplied manually or
flagged automatically (physical-bounds, isolated-spike, and 6-week holiday-robust
profile-deviation rules); a false positive only swaps one input cell for last week's
same hour — far milder than an output clamp, which is disallowed. With no flagged hours
the guard is a **no-op** and the arithmetic is byte-for-byte the base chassis.

### Design objectives

Deterministic (same input → same output, bit for bit); leakage-free by construction
(§6); fail-safe (missing or invalid input **raises**, never silently imputed —
including a required complete window from all three NWP models).

## 5. Interfaces and runtime

Inputs: ENTSO-E DE load history from 2018 (via `spotforecast2_safe`/`entsoe-py`);
per-model Open-Meteo weather (three NWP forecasts × six cities for the target day,
snapshotted per run; observed weather for training); deterministic calendar and
ephemeris (`holidays`, `astral`). Output: 24 hourly MW values for day D, written to
`submissions/<team>/<date>.csv`. CPU-only; a forecast takes seconds to a couple of
minutes (the nowcaster fit adds seconds). Each run writes a structured audit log
(EU AI Act Art. 12 record-keeping), including the publication gap, the per-lag nowcast
hour counts, and the weather-blend snapshot.

## 6. Data and operational design domain (the leakage discipline)

Valid for the DE control area with target-day weather available and an intact load
history; gaps **raise** rather than being imputed
(`LinearlyInterpolateTS(on_missing="raise")`). Every input must be genuinely available
at D−1 23:59:

- Weekly lags (≥168h back) are settled and published before submission.
- The short lags `lag_24`/`lag_48` use the real actual only up to the publication
  anchor; everything after it is the nowcaster's estimate, so no unpublished value leaks
  in, and the anchor reduces to the fixed backtest rule on training rows (no skew).
- Weather is the *forecast* vintage for the target day and *observed* for training, and
  is snapshotted so a backtest cannot see a later revision; all three NWP models must
  return a complete target window or the run raises.
- `solar_elevation` is a pure function of the timestamp — no publication state at all.
- No other team's, and no TSO's, load forecast is ever used as an input or additive
  term.

Accuracy degrades on regime shifts, anomalous holiday and bridge days, operating states
scarce in the curated window, and days when the input weather forecast itself is wrong.
The short-lag components degrade gracefully (more nowcast hours) when ENTSO-E publishes
late; first-published actuals feeding the nowcaster may still revise by a day or two.

## 7. Evaluation

v7.11 differs from its deployed base (v7.10 "wxnl") by exactly the `solar_elevation`
feature, so the honest evidence is the evidence *for that feature*, plus the full-year
validation the base chassis inherits.

**The base chassis.** The 5-seed full-year live-replica backtests behind this lineage
are on the v7.5 and v7.9 cards: v7.9 improved the v7.5 base by **+110.9 MW** mean (95%
block-bootstrap CI [+63, +168]) with the tail gate passed; v7.10 then dropped
`recent_offset` after a fresh holdout showed that term amplified the worst-rolling-7
tail. v7.11 inherits that chassis unchanged.

**The new feature — `solar_elevation`.** The offline evidence at deployment was a
**165-day, 1-seed backtest** of the feature (measured on the `wxscen` variant of the
stack): mean **−17.7 MW/day** with the feature winning on **95 of 165 days**, but the
worst-rolling-7 window slightly *worse* (2048 vs 1999 MW) — in one line,
**mean-helpful, tail-neutral-at-best.** On the live `wxmean` chassis it had **little
scored evidence at the time it was deployed** (it was promoted before scored live days
had accrued), so no live MAE for v7.11 is reported here rather than a fabricated one.

**Why it was adopted anyway.** Two structural reasons, not a large measured win.
*Rationale:* within-day sun geometry targets the **morning and evening ramp residuals**
specifically — the smooth intra-day angle that `is_daylight` (binary) and
`daylight_hours` (per-day constant) cannot carry. *Safety:* the construction is a
**deterministic, timestamp-only** function on the *unchanged* live chassis — no new
data dependency, no publication-state sensitivity, no leakage channel, so it cannot
introduce a train/serve skew or a late-data failure mode.

The transferable lesson is the method: judge the tail on a long live-replica window, not
just the mean; and when scored evidence for a marginal change is thin, prefer one whose
*worst case is bounded by construction*.

## 8. Model transparency

Point forecasts only; no native uncertainty quantification (an L1/quantile objective
tested worse than L2 for MAE). White-box: LightGBM ensembles plus a LightGBM nowcaster,
with feature attribution via native TreeSHAP (`predict(pred_contrib=True)`, no
`shap`/`matplotlib` dependency). The autoregressive lags dominate; the new columns to
read are `lag_24_nc`, `lag_48` and `solar_elevation`. Mean|SHAP| is read as *influence*,
not out-of-sample benefit — a diagnostic and Art. 13 record, never a selection gate.

## 9. Operation: monitoring and response

Retrained daily. Monitor input freshness and coverage, the ENTSO-E publication gap and
per-lag nowcast-hour counts, the completeness of all three NWP target windows,
distribution drift against the curated training window, and the corrupt-input guard's
flag list (an unusual number of flagged hours is a warning sign). The overriding
operational rule is to submit early and confirm acceptance: a missed submission carried
forward is the single biggest ranking threat. The final fallback chain behind v7.11 is
**v7.9 → v7.5 → v4.5** — each an emergency submit path if v7.11 generation fails.

## 10. Compliance support

Built to the EU AI Act articles the course emphasises: data governance via the leakage
rules and fail-safe-on-missing (Art. 10); this card and the audit log as the
technical-documentation baseline (Art. 11); per-run structured audit logs (Art. 12);
white-box, re-derivable transforms (Art. 13); deterministic, reproducible computation
(Art. 15). The four code rules apply: CR-1 git-audited submissions, CR-2 determinism
(seed base 2026; `PYTHONHASHSEED=0`; no RNG outside LightGBM's seeded fit and
subsampling), CR-3 raise-on-missing (load rows, weather, and the NWP target window),
CR-4 a LightGBM-only modelling stack with the heavy/non-deterministic deny-list
enforced.

## 11. Limitations and ethical considerations

Forecasts one bidding zone for a competition; must not be repurposed for grid operation
or safety-critical use without independent validation. It can be wrong on rare regimes,
anomalous holiday and bridge days, and days when the input weather forecast is itself
wrong; the short-lag nowcast carries the known slight optimism of first-published
actuals. The one new feature was deployed on structural and offline grounds with thin
*live* scored evidence — an honest limitation, mitigated by its leakage-free,
worst-case-bounded construction. Point forecasts carry no calibrated uncertainty.

## 12. Citation and contact

Team hot_rod, "Voltcrown" model line, Lastprognose-Challenge SS26. Built on
`spotforecast2_safe` (T. Bartz-Beielstein) and LightGBM. See
[`CITATION.cff`](../CITATION.cff).

## 13. Disclaimer

Provided as-is for a forecasting competition. Forecast accuracy is bounded by the data
and the weather forecast; no warranty, and not for safety-critical use.
