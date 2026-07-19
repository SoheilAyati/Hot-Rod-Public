# Model Card — Voltcrown v7.5 (Team hot_rod)

Public model card for **Voltcrown v7.5**, a fallback model of Team *hot_rod* in the
Lastprognose-Challenge SS26 (TH Köln, *Data-Driven Modeling and Optimization*, summer
2026). It follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy and is written to stand on its own, so it can be read without reference to
the other cards in this folder.

> **Fully released.** The challenge has concluded, so v7.5 is released **in full** and
> open for anyone to use — this card discloses every hyperparameter, training window,
> feature and the complete ensemble recipe, and the **runnable reference
> implementation is included** at [`../models/v7.5/make_submission_v75.py`](../models/v7.5/make_submission_v75.py).
> v7.5 is the team's most thoroughly full-year-validated model.

## 1. Model details

| Field | Value |
| --- | --- |
| Name | Voltcrown v7.5 (Team hot_rod), `make_submission_v75.py` |
| Status | Fallback (open reference release) |
| Type | Deterministic, daily-retrained, **direct 24-hour multi-output** load forecaster: a **5-seed subsampled LightGBM ensemble** over calendar, holiday, ephemeris and weather features |
| Target | ENTSO-E "DE Actual Total Load", MW, hourly — 24 values for target day D |
| Developed by | Team hot_rod, Lastprognose-Challenge SS26 |
| Score metric | Rolling **7-day mean MAE** (MW); lower is better |
| Cadence | Submit by **D−1 23:59 CET** for target day D; scored ~07:00 UTC on D+1 |
| Determinism | Reproducible: fixed seed base (2026), `PYTHONHASHSEED=0`, snapshotted inputs |
| Reference code | [`../models/v7.5/make_submission_v75.py`](../models/v7.5/make_submission_v75.py) |
| Built on | `spotforecast2-safe` (T. Bartz-Beielstein) + LightGBM |
| Lineage | v3 → v4.5 → v6 → v7 → **v7.5** (see the per-version cards) |

The runtime stack is a small, allow-listed set (numpy, pandas, scikit-learn,
lightgbm, holidays, astral, entsoe-py, requests, statsmodels, spotforecast2_safe).
Heavy or non-deterministic libraries (torch, tensorflow, optuna, xgboost, catboost,
matplotlib, plotly) are excluded to keep the dependency and CVE surface small and the
pipeline bit-reproducible. LightGBM is the only permitted gradient booster.

## 2. Intended use and scope

v7.5 produces one daily competition submission: a 24-value MW forecast of the next
day's German load. It is tuned for one objective — the lowest rolling 7-day mean MAE —
and, because that mean is **tail-dominated** (median daily MAE ~1100 MW vs mean ~1250),
it is specifically optimised so the *worst* days are as small as possible.

Out of scope: not a general forecasting library; DE control area only; point
forecasts with no calibrated uncertainty; must not use realised same-day exogenous
data (that would leak — §6) or be repurposed for grid operation without independent
validation.

## 3. How it works (plain language)

The engine is a **gradient-boosted decision tree** model (LightGBM): hundreds of small
yes/no-question trees built in sequence, each correcting earlier mistakes, summed into
a load estimate. All 24 hours of day D are predicted at once (each from its own known
inputs) rather than recursively, which prevents an early-hour error from cascading
through the day. The model is retrained from scratch every day.

Two design choices define v7.5. First, a deliberately **curated training window**:
2018–2019 together with 2024 to the present, **skipping 2020–2023** — an older "shape"
window plus the recent regime, dropping the pandemic and energy-crisis years whose
load behaviour is unlike the current one. Second, the **ensemble**: the same tuned
configuration is trained under five random seeds and its 24-hour predictions are
averaged. The point is tail-day robustness — on a hard day a single model can make one
unlucky large error, but independent members rarely all blow up the same way, so the
average is steadier. Crucially, the active ingredient is not the averaging but the
**row-and-feature subsampling** that each member uses to become diverse enough to be
worth averaging (§4): the subsampling is a regularizer, and it, not the five-fold vote,
is what delivers the gain.

## 4. Technical specification (complete)

### Estimator

A LightGBM regressor trained under five seeds with row+feature subsampling, its 24-hour
predictions averaged. No output bias correction (see the rationale below and §7).

| Parameter | Value |
| --- | --- |
| `n_estimators` | 3000 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 (tuned; was 23 in v4.5) |
| `min_child_samples` | 40 |
| `max_depth` | −1 (unbounded) |
| `subsample` (bagging fraction) | 0.7 (row subsampling — regularization) |
| `subsample_freq` | 1 (bag every iteration) |
| `colsample_bytree` (feature fraction) | 0.9 (feature subsampling) |
| seeds | 5 seeds, 2026–2030, predictions averaged |
| bias correction | **OFF** (tested and rejected — §7) |

The tuned values were selected by a two-fold guardrail search (ranked by the *worse* of
a Mar–Apr and a May–Jun fold) and confirmed on a 4-arm backtest. Versus the earlier
`num_leaves=23`, `subsample/colsample=0.8` defaults, the tuned
`L31 / M40 / S0.7 / C0.9` winner lowered the mean (1240.6 → 1231.5), median
(1124 → 1071), worst-10 (2383 → 2299) and worst-day (4513 → 4387) — better in body *and*
tail.

### Training window

`KEEP_WINDOWS = [2018-01-01 … 2019-12-31] ∪ [2024-01-01 … D−1]`. Data is pulled from
2018-01-01; rows outside the kept windows (i.e. 2020–2023) are dropped before training.
The model retrains daily, so the second window always extends to the hour before the
target day.

### Features (~51 columns)

- **Autoregressive lags (weekly-aligned):** `lag_168`, `lag_336`, `lag_504`, `lag_672`
  (1–4 weeks back, same weekday-and-hour). All ≥168h so they are always *settled and
  available* at submission (§6 leakage).
- **Rolling statistics:** `roll_mean_last_week`, `roll_max_168`, `roll_min_168`,
  `roll_std_168` over a 168-hour window, itself shifted 168h back.
- **Calendar (raw):** `hour`, `dayofweek`, `month`, `is_weekend`.
- **Calendar (cyclical):** `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`.
- **Sunlight (ephemeris, deterministic):** `daylight_hours`, `is_daylight`.
- **Holidays (union over all sixteen German *Bundesländer*):** per-holiday one-hot
  `h_*` flags, plus `is_any_holiday`, `is_day_before_holiday`, `is_day_after_holiday`,
  and `is_bridge_day` (a working day wedged between a holiday and the weekend).
- **Weather (population-weighted over six cities):** `de_temperature_2m`,
  `de_cloud_cover`, `de_wind_speed_10m`, degree-days `de_hdd = max(0, 15 − T)` and
  `de_cdd = max(0, T − 22)`, and `de_shortwave_radiation` (solar GHI). Cities and
  population weights: Berlin 5.0, Hamburg 3.4, Munich 3.0, Rhine-Ruhr 10.0, Frankfurt
  5.8, Stuttgart 2.7. Historical weather is observed; the target-day tail is the
  Open-Meteo forecast vintage, snapshotted per run.

The cyclical encodings, the sunlight ephemeris and the solar-radiation channel are
exactly what separate v7.5 from the earlier v4.5.

### Design rationale (subsampling is the active ingredient)

The active ingredient of the "ensemble" is **subsampling (regularization)**, not
averaging. An ablation showed a single subsampled member captures ~82% of the gain;
the five seeds add only ~6.5 MW (a slightly steadier tail). Beyond ~5 seeds the 1/√N
floor makes more seeds pointless. This is why v7.5 is fixed at five seeds, and why the
subsampling — not the vote — is treated as the model's defining change over v7.

### Design objectives

Deterministic (same input → same output, bit for bit; the fixed seed list keeps the
ensemble reproducible and the CSV byte-identical for a given seed count); leakage-free
by construction (§6); fail-safe (missing or invalid input raises, never silently
imputed).

## 5. Interfaces and runtime

Inputs: ENTSO-E DE load history from 2018 (via `spotforecast2_safe`/entsoe-py);
Open-Meteo weather (forecast for the target day, observed for training); deterministic
`holidays`/`astral` calendar and ephemeris. Output: 24 hourly MW values for day D,
written to `<repo-root>/submissions/<team>/<date>.csv`. CPU-only; a forecast takes
seconds to a couple of minutes (weather and ENTSO-E fetches are cached/snapshotted).
Each run writes a structured audit log (EU AI Act Art. 12 record-keeping).

## 6. Data and operational design domain (the leakage discipline)

Valid for the DE control area with target-day weather available and an intact load
history; gaps **raise** rather than being imputed (`LinearlyInterpolateTS(on_missing=
"raise")`). Every input must be genuinely available at D−1 23:59:

- Autoregressive lags are weekly (≥168h), so they are settled and published before
  submission.
- Weather is the *forecast* vintage for the target day and *observed* for training
  (live-replica), and is snapshotted so a backtest cannot accidentally see a later
  revision.
- Any exogenous market signal may use only its **day-ahead** vintage, never the
  realised same-day value.
- No other team's, and no TSO's, load forecast (e.g. the ENTSO-E day-ahead Forecasted
  Load) is ever used as an input or additive term.

Accuracy degrades on regime shifts, anomalous holiday and bridge days, and rare
operating states under-represented in the curated window — these are the tail days the
model is tuned to contain but cannot eliminate, along with days when the input weather
forecast itself is wrong.

## 7. Evaluation

Evaluated with `live-replica` backtests (forecast weather on the target day, observed
weather for training) and moving-block bootstrap confidence intervals. All numbers are
MW MAE; lower is better.

**60-day window (May–Jun 2025), 5-seed, deploy config:**

| metric | v7.5 |
| --- | --- |
| mean | 1231 |
| median | 1071 |
| worst-10 | 2299 |
| worst-day | 4387 |

**Full-year window (1 Jul 2024 – 30 Jun 2025, 364 days), 5-seed, robust:**

| metric | v7.5 | v4.5 (backup) |
| --- | --- | --- |
| mean | **1331** | 1633 |
| worst-10 | **4153** | 5572 |
| worst-day | **5078** | 8855 |
| worst 7-day window | **2621** | 3830 |

Paired (v4.5 − v7.5): **+302 MW** mean, 95% block-bootstrap CI **[+207, +407]**
(excludes zero → robust). v7.5 wins **every season** (largest in winter, +388) and
**every day-type**, and beats v4.5 on **68%** of days. Fresh-day MAPE ≈ **2.3%**
(professional-grade; beats the ENTSO-E TSO reference on most normal days).

**Successor feature search (all rejected against v7.5 on the tail gate):** day-ahead
spot price, polynomial/interaction features, a widened bridge-day rule, SHAP-top-N
selection, the kitchen-sink combination, v4.5-style bias correction, school holidays,
and wind-only. None lowered the mean while holding the tail. The conclusion was that
**v7.5 is the model** — there is "no magic feature at 2.3% MAPE."

The transferable lesson is the method, not the MW: validate on a long, live-replica
window and judge the tail, not just the mean.

## 8. Model transparency

Point forecasts only; no native uncertainty quantification (a quantile/conformal head
would have to be added downstream — kept off the leaderboard path because L1/quantile
objectives tested worse than L2 for MAE). White-box — a gradient-boosted-tree ensemble.
Feature attribution via LightGBM-native TreeSHAP (`predict(pred_contrib=True)`, no
`shap`/`matplotlib` dependency): the **autoregressive lags dominate** (lag_168 alone
≈ 31% of mean|SHAP|, the four lags ≈ 50%), followed by holiday flags, daylight, HDD
and the cyclical hour encodings. Mean|SHAP| measures *influence*, not out-of-sample
benefit — it is used for redundancy diagnosis and the Art. 13 record, never as a
feature-selection gate.

## 9. Operation: monitoring and response

Retrained daily. Monitor input freshness and coverage (ENTSO-E publication lag,
weather coverage), distribution drift against the curated training window, and daily
MAE against the ENTSO-E reference. The cross-member spread reported each run is a
useful tail-day warning signal. The overriding operational rule is to submit early and
confirm acceptance: a missed submission carried forward and scored against the wrong
day is the single biggest ranking threat.

## 10. Compliance support

Built to the EU AI Act articles the course emphasises: data governance via the leakage
rules and fail-safe-on-missing (Art. 10); this card and the audit log as the
technical-documentation baseline (Art. 11); per-run structured audit logs (Art. 12);
white-box, re-derivable transforms (Art. 13); deterministic, reproducible computation
(Art. 15). The four code rules apply: CR-1 git-audited submissions (only the CSV is
committed); CR-2 determinism (seed base 2026; `PYTHONHASHSEED=0`; no RNG outside
LightGBM's seeded fit; snapshotted data); CR-3 raise-on-missing (no silent NaN
imputation); CR-4 a LightGBM-only stack with the heavy/non-deterministic deny-list
enforced.

## 11. Limitations and ethical considerations

Forecasts one bidding zone for a competition; must not be repurposed for grid operation
or safety-critical use without independent validation. It can be wrong on rare regimes,
on anomalous holiday and bridge days, and on days when the input weather forecast is
wrong — these are the tail days the ensemble is tuned to contain but cannot eliminate.
Point forecasts carry no calibrated uncertainty.

## 12. Citation and contact

Team hot_rod, "Voltcrown" model line, Lastprognose-Challenge SS26. Built on
`spotforecast2-safe` (T. Bartz-Beielstein) and LightGBM. See [`CITATION.cff`](../CITATION.cff).

## 13. Disclaimer

Provided as-is for a forecasting competition. Forecast accuracy is bounded by the data
and the weather forecast; no warranty, and not for safety-critical use.
