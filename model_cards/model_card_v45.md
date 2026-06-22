# Model Card — Voltcrown v4.5 (Team hot_rod)

Public model card for **Voltcrown v4.5**, a fallback model of Team *hot_rod* in the
Lastprognose-Challenge SS26 (TH Köln, *Data-Driven Modeling and Optimization*, summer
2026). It follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy and is written to stand on its own, so it can be read without reference to
the other cards in this folder.

> **Fully released.** v4.5 is released **in full** and open for anyone to use — this
> card discloses every hyperparameter, training window,
> feature and the complete bias-correction algorithm, and the **runnable reference
> implementation is included** at [`../models/v4.5/`](../models/v4.5/). (The team's
> current models are documented at the methodology level while in use; v4.5 is two
> generations behind and is opened completely.)

## 1. Model details

| Field | Value |
| --- | --- |
| Name | Voltcrown v4.5 (Team hot_rod), `make_submission_v45.py` |
| Status | Fallback (open reference release) |
| Type | Deterministic, daily-retrained, direct 24-hour load forecaster: a **single tuned LightGBM** over calendar, holiday and weather features, plus a scalar out-of-sample bias-correction layer |
| Target | ENTSO-E "DE Actual Total Load", MW, hourly — 24 values for target day D |
| Developed by | Team hot_rod, Lastprognose-Challenge SS26 |
| Score metric | Rolling **7-day mean MAE** (MW); lower is better |
| Cadence | Submit by **D−1 23:59 CET** for target day D |
| Determinism | Reproducible: fixed seed (2026), `PYTHONHASHSEED=0`, snapshotted inputs |
| Reference code | [`../models/v4.5/make_submission_v45.py`](../models/v4.5/make_submission_v45.py) |
| Built on | `spotforecast2-safe` (T. Bartz-Beielstein) + LightGBM |

The runtime stack is a small, allow-listed set (numpy, pandas, scikit-learn,
lightgbm, holidays, entsoe-py, requests). Heavy or non-deterministic libraries
(torch, tensorflow, optuna, xgboost, catboost, matplotlib, plotly) are excluded to
keep the dependency and CVE surface small and the pipeline bit-reproducible.

## 2. Intended use and scope

v4.5 produces one daily competition submission: a 24-value MW forecast of the next
day's German load. It is the team's deepest fallback — the simplest model that is
still competitive — and is published here as a complete, reusable reference.

Out of scope: not a general forecasting library; DE control area only; point
forecasts with no calibrated uncertainty; must not use realised same-day exogenous
data (that would leak — §6) or be repurposed for grid operation without independent
validation.

## 3. How it works (plain language)

The engine is a single **gradient-boosted decision tree** model (LightGBM): hundreds
of small yes/no-question trees built in sequence, each correcting earlier mistakes,
summed into a load estimate. All 24 hours of day D are predicted at once (each from
its own known inputs) rather than recursively, which prevents an early-hour error
from cascading through the day. The model is retrained from scratch every day.

Two design choices define v4.5. First, a deliberately **curated training window**:
2018–2019 together with 2024 to the present, **skipping 2020–2023** — the pandemic
and energy-crisis years whose load behaviour is unlike the current regime. Including
those years made the model worse. Second, a **scalar bias correction**: a single
additive constant, estimated out-of-sample from recent residuals, that nudges the
whole forecast up or down to remove a persistent level offset. As §7 shows with the
numbers, this term turned out to *hurt* on a full year — a useful cautionary tale
about global corrections, preserved here exactly as it was deployed.

## 4. Technical specification (complete)

### Estimator

A single LightGBM regressor (no ensemble, no subsampling), then a scalar additive
bias correction.

| Parameter | Value |
| --- | --- |
| `n_estimators` | 3000 |
| `learning_rate` | 0.05 |
| `num_leaves` | 23 |
| `min_child_samples` | 40 |
| `max_depth` | −1 (unbounded) |
| `subsample` / `colsample` | none |
| `random_state` | 2026 |

### Training window

`KEEP_WINDOWS = [2018-01-01 … 2019-12-31] ∪ [2024-01-01 … D−1]`. Data is pulled from
2018-01-01; rows outside the kept windows (i.e. 2020–2023) are dropped before
training. The model retrains daily, so the second window always extends to the hour
before the target day.

### Features (~39 columns)

- **Weekly autoregressive lags:** `lag_168`, `lag_336`, `lag_504`, `lag_672` (1–4
  weeks back, always settled before submission).
- **One rolling statistic:** `roll_mean_last_week` — the 168-hour rolling mean of
  `lag_168`.
- **Calendar (raw, no cyclical encodings):** `hour`, `dayofweek`, `month`,
  `is_weekend`.
- **Holidays:** a per-holiday one-hot `h_*` built from the union over all sixteen
  German *Bundesländer*, plus `is_any_holiday`, `is_day_before_holiday`,
  `is_day_after_holiday`, and `is_bridge_day` (a working day wedged between a holiday
  and the weekend).
- **Weather (population-weighted over six cities):** `de_temperature_2m`,
  `de_cloud_cover`, `de_wind_speed_10m`, and degree-days `de_hdd = max(0, 15 − T)`,
  `de_cdd = max(0, T − 22)`. Cities and population weights: Berlin 5.0, Hamburg 3.4,
  Munich 3.0, Rhine-Ruhr 10.0, Frankfurt 5.8, Stuttgart 2.7. Historical weather is
  observed; the target-day tail is the Open-Meteo forecast vintage, snapshotted per
  run.

v4.5 predates the cyclical (sine/cosine) encodings, the sunlight ephemeris, and the
solar-radiation channel of later versions; those are exactly what separate it from
v7.5 and v7.9.

### Bias correction (the defining feature; default on)

A single scalar estimated **out-of-sample** every run, so it is honest rather than a
near-zero in-sample residual:

1. Hold out the last `BIAS_WINDOW_DAYS = 14` days.
2. Train a bias model (same LightGBM config) on the kept windows strictly *before* the
   holdout.
3. Predict the holdout → genuinely out-of-sample residuals.
4. `bias = median(actual − predicted)` over **non-holiday** holdout hours (so a public
   holiday in the window cannot skew it). Positive ⇒ the model runs low ⇒ shift up.
5. Train the final model on **all** kept data and predict the target day.
6. `forecast = raw_prediction + bias`, with `bias` clipped to ±`BIAS_CLIP_MW = 2500`
   as a safety rail.

Disable it with `--no-bias-correction`.

### Design objectives

Deterministic (same input → same output, bit for bit); leakage-free by construction
(§6); fail-safe (missing or invalid input raises, never silently imputed).

## 5. Interfaces and runtime

Inputs: ENTSO-E DE load history from 2018; forecast weather for the target day and
observed weather for training; deterministic calendar. Output: 24 hourly MW values
for day D, written to `<repo-root>/submissions/<team>/<date>.csv`. CPU-only; a
forecast takes seconds to a couple of minutes (the bias layer fits one extra model
per run). Each run writes a structured audit log (EU AI Act Art. 12 record-keeping).

## 6. Data and operational design domain (the leakage discipline)

Valid for the DE control area with target-day weather available and an intact load
history; gaps **raise** rather than being imputed (`LinearlyInterpolateTS(on_missing=
"raise")`). Every input must be genuinely available at D−1 23:59:

- Autoregressive lags are weekly (≥1 week back), so they are settled and published
  before submission.
- Weather is the *forecast* vintage for the target day and *observed* for training,
  and is snapshotted so a backtest cannot accidentally see a later revision.
- The bias correction is estimated only from data before the target day, with an
  out-of-sample holdout — it never peeks at the target.
- No other team's, and no TSO's, load forecast is ever used as an input or additive
  term.

Accuracy degrades on regime shifts, anomalous holiday and bridge days, operating
states scarce in the curated window, and days when the input weather forecast itself
is wrong. The scalar bias term can also mis-fire when the recent residual level is
not representative of the target day.

## 7. Evaluation

Full-year live-replica backtest (1 Jul 2024 – 30 Jun 2025, 364 days; 5-seed harness).
Lower is better; MW.

| arm | mean | worst-10 | worst day | worst rolling-7d |
| --- | --- | --- | --- | --- |
| **v4.5** (bias on, as deployed) | **1633** | **5572** | **8855** | **3830** |
| v4.5 (no bias, diagnostic) | 1549 | 5380 | 7803 | 4236 |
| v7.5 (first fallback) | 1331 | 4153 | 5078 | 2621 |
| v7.9 (deployed) | 1220 | 3842 | 5095 | 2306 |

Two findings worth sharing, both of which overturned a prior belief:

1. **v4.5 is not the "safe" one.** Its catastrophes are far worse than the later
   models' (worst day 8855 MW vs ~5100). The impression that v4.5 was the reliable
   choice came from a short 60-day window and a single settled live day; it does not
   survive a full year. The successor v7.5 beats it by **+302 MW mean** (moving-block
   bootstrap 95% CI [+207, +407]) in every season and day-type, body and tail.
2. **The bias layer hurts.** Over the full year the correction *costs* +85 MW on the
   mean and +1052 MW on the worst day versus the no-bias variant; it improves only the
   worst rolling-7-day window. It is a volatile daily correction (±700–1000 MW swings)
   chasing a moving target — which is exactly why every later Voltcrown version
   dropped output bias correction. It is kept on here because that is the model as it
   was deployed, and the comparison is instructive.

The transferable lesson is the method, not the MW: validate on a long, live-replica
window and judge the tail, not just the mean.

## 8. Model transparency

Point forecasts only; no native uncertainty quantification. White-box — a single
gradient-boosted-tree model plus a transparent scalar offset; feature attribution via
LightGBM's native tree importances, read as influence rather than benefit.

## 9. Operation: monitoring and response

Retrained daily. Monitor input freshness and coverage, distribution drift against the
curated training window, the publication lag of the most recent actuals, and the size
of the scalar bias term (an unusually large or clipped correction is a warning sign).
The overriding operational rule is to submit early and confirm acceptance: a missed
submission carried forward is the single biggest ranking threat.

## 10. Compliance support

Built to the EU AI Act articles the course emphasises: data governance via the
leakage rules and fail-safe-on-missing (Art. 10); this card and the audit log as the
technical-documentation baseline (Art. 11); per-run structured audit logs (Art. 12);
white-box, re-derivable transforms (Art. 13); deterministic, reproducible computation
(Art. 15). The four code rules apply: CR-1 git-audited submissions, CR-2 determinism
(seed 2026; `PYTHONHASHSEED=0`; no RNG outside LightGBM's seeded fit), CR-3
raise-on-missing, CR-4 a LightGBM-only stack with the heavy/non-deterministic
deny-list enforced.

## 11. Limitations and ethical considerations

Forecasts one bidding zone for a competition; must not be repurposed for grid
operation or safety-critical use without independent validation. Its tail is heavier
than the later models', it can be wrong on rare regimes and on days when the input
weather forecast is wrong, and the scalar bias term can mis-fire (and, as §7 shows,
hurts on average). Point forecasts carry no calibrated uncertainty.

## 12. Citation and contact

Team hot_rod, "Voltcrown" model line, Lastprognose-Challenge SS26. Built on
`spotforecast2-safe` (T. Bartz-Beielstein) and LightGBM. See [`CITATION.cff`](../CITATION.cff).

## 13. Disclaimer

Provided as-is for a forecasting competition. Forecast accuracy is bounded by the
data and the weather forecast; no warranty, and not for safety-critical use.
