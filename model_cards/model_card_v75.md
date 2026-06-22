# Model Card — Voltcrown v7.5 (Team hot_rod)

Public model card for **Voltcrown v7.5**, the validated fallback model of Team
*hot_rod* in the Lastprognose-Challenge SS26 (TH Köln, *Data-Driven Modeling and
Optimization*, summer 2026). It follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy and is written to stand on its own, so it can be read without reference to
the other cards in this folder.

> **Disclosure status.** v7.5 remains an active fallback for the deployed model, so
> this card describes it at the **methodology level**: model family, data, feature
> *families*, the leakage discipline, the evaluation protocol and the measured
> results. Exact tuned hyperparameters are deployment specifics, released when the
> model is fully retired (no longer used even as a fallback). Nothing here is false
> about what the model does; it is abstracted, not altered.

## 1. Model details

| Field | Value |
| --- | --- |
| Name | Voltcrown v7.5 (Team hot_rod) |
| Status | Validated fallback (first fallback for the deployed model) |
| Type | Deterministic, daily-retrained, direct 24-hour load forecaster: a multi-seed **subsampled LightGBM ensemble** over calendar, weather and autoregressive features, with tuned hyperparameters and the scalar bias term removed |
| Target | ENTSO-E "DE Actual Total Load", MW, hourly — 24 values for target day D |
| Developed by | Team hot_rod, Lastprognose-Challenge SS26 |
| Score metric | Rolling **7-day mean MAE** (MW); lower is better |
| Cadence | Submit by **D−1 23:59 CET** for target day D |
| Determinism | Reproducible: fixed seeds, `PYTHONHASHSEED=0`, snapshotted inputs |
| Built on | `spotforecast2-safe` (T. Bartz-Beielstein) + LightGBM |

The runtime stack is a small, allow-listed set (numpy, pandas, scikit-learn,
lightgbm, holidays, astral, entsoe-py, requests, statsmodels). Heavy or
non-deterministic libraries (torch, tensorflow, optuna, xgboost, catboost,
matplotlib, plotly) are excluded to keep the dependency and CVE surface small and
the pipeline bit-reproducible.

## 2. Intended use and scope

v7.5 produces one daily competition submission: a 24-value MW forecast of the next
day's German load. It is the model the team falls back to when the deployed model
cannot be generated, and the validated reference any new candidate must beat. As the
score is a tail-dominated rolling mean, the model is tuned to contain the worst days,
not only to be good on average.

Out of scope: not a general forecasting library; DE bidding zone only; point
forecasts with no calibrated uncertainty; must not use realised same-day exogenous
data (that would leak — §6) or be repurposed for grid operation without independent
validation.

## 3. How it works (plain language)

The engine is **gradient-boosted decision trees** (LightGBM): hundreds of small
yes/no-question trees built in sequence, each correcting earlier mistakes, summed
into a load estimate. All 24 hours of day D are predicted at once (each from its own
known inputs) rather than recursively, which prevents an early-hour error from
cascading through the day. The model is retrained from scratch every day: long
history anchors the stable weekly and seasonal patterns, and the newest rows keep it
current.

What distinguishes v7.5 from a single tree model is the **ensemble built by
subsampling**. Several LightGBM models are trained under different fixed seeds, each
on a random subset of rows and features, and their predictions are averaged. This
regularises the forecast — no single model can over-memorise a quirk of the training
data — and steadies the hard days. A finding worth sharing: the averaging itself
contributes only single-digit MW (the 1/√N noise floor); the real benefit is the
subsampling regularisation, not the seed count.

## 4. Technical specification (feature families; exact tuned values withheld)

### Estimator

A multi-seed subsampled LightGBM ensemble with tuned hyperparameters (moderate leaf
count, low learning rate with a high tree count, row and feature subsampling). The
**scalar bias-correction term used by the earlier v4.5 line was removed**: on a
properly regularised ensemble a global additive offset traded body accuracy for tail
risk and did not survive the tail gate.

### Feature families

- **Autoregressive lags**, weekly-aligned (≥1 week back, hence always settled before
  submission), plus **rolling-window statistics** over a one-week window (recent
  level, peak, floor, variability).
- **Calendar**: hour, day-of-week, month and weekend flags, with **cyclical
  (sine/cosine) encodings** so the model treats time as a loop.
- **Sunlight**: deterministic ephemeris (daylight length, day/night) from date and
  coordinates.
- **Holidays**: the union across all sixteen German *Bundesländer*, with
  day-before / day-after / bridge-day flags.
- **Weather**: population-weighted across major German population centres —
  temperature, cloud cover, wind, heating/cooling degree-days, and **solar
  radiation**, a direct read on sunshine that displaces grid load through rooftop PV
  and is the single most valuable weather input.

### Design objectives

Deterministic (same input → same output, bit for bit); leakage-free by construction
(§6); fail-safe (missing or invalid input raises, never silently imputed).

## 5. Interfaces and runtime

Inputs: ENTSO-E DE load history; forecast weather for the target day and observed
weather for training; deterministic calendar and ephemeris. Output: 24 hourly MW
values for day D. CPU-only; seconds to minutes per daily forecast. Each run writes a
structured audit log (EU AI Act Art. 12 record-keeping).

## 6. Data and operational design domain (the leakage discipline)

Valid for the DE control area with target-day weather available and an intact load
history; gaps **raise** rather than being imputed. Every input must be genuinely
available at D−1 23:59:

- Autoregressive lags are weekly (≥1 week back), so they are settled and published
  before submission.
- Weather is the *forecast* vintage for the target day and *observed* for training (a
  live-replica setup), so the model learns the true weather→load map and is scored
  under realistic forecast error.
- No other team's, and no TSO's, load forecast is ever used as an input or additive
  term.

Accuracy degrades on regime shifts, anomalous holiday and bridge days, operating
states scarce in the training data, and days when the input weather forecast itself
is wrong.

## 7. Evaluation

The adoption discipline (recommended to other teams, and shipped as runnable code in
[`../tools`](../tools)):

- **Live-replica backtests over a full year** — forecast weather on the target day,
  observed weather for training. Short windows mislead.
- **Tail-gated adoption**: a change ships only if it lowers the mean *and* holds the
  tail — a paired improvement whose moving-block-bootstrap confidence interval
  excludes zero, with no regression on the worst-10-day mean or the worst
  rolling-7-day window.
- **No bundling**: each candidate is tested in isolation before combining.
- **Influence ≠ benefit**: features are never selected by importance ranking.

### Results

Full-year live-replica backtest (1 Jul 2024 – 30 Jun 2025, 364 days; deploy
configuration). Lower is better; MW.

| model | role | mean MAE | worst-10-day mean | worst single day | worst rolling-7-day |
| --- | --- | --- | --- | --- | --- |
| v4.5 | second fallback | 1633 | 5572 | 8855 | 3830 |
| **v7.5** | **validated fallback** | **1331** | **4153** | **5078** | **2621** |

Paired against the v4.5 fallback (moving-block bootstrap, 95% CI): **+302 MW mean**
(CI [+207, +407]) — an improvement in every season and day-type, in both the body and
the tail. The gain over v4.5 comes from richer features (cyclical encodings,
sunlight, solar radiation, expanded rolling statistics), the removal of the scalar
bias term, and the subsampled multi-seed ensemble that regularises the worst days.
v7.5 is in turn beaten by the deployed v7.9 model by a further +111 MW mean.

## 8. Model transparency

Point forecasts only; no native uncertainty quantification. White-box — gradient-
boosted trees, no opaque weights; feature attribution via LightGBM's native tree
importances, read as influence rather than benefit.

## 9. Operation: monitoring and response

Retrained daily. We monitor input freshness and coverage, distribution drift against
the training period, the publication lag of the most recent actuals, and daily error
against a reference. The overriding operational rule is to submit early and confirm
acceptance: a missed submission carried forward is the single biggest ranking threat.

## 10. Compliance support

Built to the EU AI Act articles the course emphasises: data governance via the
leakage rules and fail-safe-on-missing (Art. 10); this card and internal
documentation as the technical-documentation baseline (Art. 11); per-run structured
audit logs (Art. 12); white-box, re-derivable transforms (Art. 13); deterministic,
reproducible computation (Art. 15). The four code rules apply: CR-1 git-audited
submissions, CR-2 determinism (pinned seeds; no RNG outside LightGBM's seeded
sampling), CR-3 raise-on-missing, CR-4 a LightGBM-only stack with the
heavy/non-deterministic deny-list enforced.

## 11. Limitations and ethical considerations

Forecasts one bidding zone for a competition; must not be repurposed for grid
operation or safety-critical use without independent validation. It can be wrong on
rare regimes and on days when the input weather forecast is wrong. Point forecasts
carry no calibrated uncertainty.

## 12. Citation and contact

Team hot_rod, "Voltcrown" model line, Lastprognose-Challenge SS26. Built on
`spotforecast2-safe` (T. Bartz-Beielstein) and LightGBM. See [`CITATION.cff`](../CITATION.cff).

## 13. Disclaimer

Provided as-is for a forecasting competition. Forecast accuracy is bounded by the
data and the weather forecast; no warranty, and not for safety-critical use.

---

*Intentionally not in this public card:* exact tuned hyperparameters, seeds, lag
offsets and window lengths. These are released when the model is fully retired.
