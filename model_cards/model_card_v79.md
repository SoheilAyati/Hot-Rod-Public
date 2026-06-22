# Model Card — Voltcrown v7.9 (Team hot_rod)

Public model card for **Voltcrown v7.9**, the deployed day-ahead electricity-load
forecasting model of Team *hot_rod* in the Lastprognose-Challenge SS26 (TH Köln,
*Data-Driven Modeling and Optimization*, summer 2026). The card follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy and is written so that other teams can learn from the **methodology and
design decisions** — the part that transfers between problems.

> **Disclosure status.** Voltcrown v7.9 is the live model, so this card describes
> it at the **methodology level**: model family, data, feature *families*, the
> leakage discipline, the evaluation protocol, the measured results, the limits,
> and the compliance posture — everything an auditor or a classmate needs to
> understand *how it works and why*. Exact hyperparameters, the precise feature
> formulas, and any not-yet-deployed candidates are deployment specifics and are
> withheld while this version is in use. The complete technical card is published
> once the version is retired — i.e. when a newer version supersedes it and it is
> no longer used as a fallback. Nothing here is false about what the deployed
> model does; it is abstracted, not altered. A **private determinism audit** of the
> live script, for a selected group, is planned to demonstrate reproducibility in the
> meantime (see §9a).

## 1. Model details

| Field | Value |
| --- | --- |
| Name | Voltcrown v7.9 (Team hot_rod), deployed model |
| Status | Deployed / live |
| Type | Deterministic, daily-retrained, direct 24-hour load forecaster: a multi-seed gradient-boosted-tree (LightGBM) ensemble over calendar, weather and autoregressive features, extended with recent-regime signals, nowcast-imputed short lags, and multi-model weather inputs |
| Target | ENTSO-E "DE Actual Total Load", MW, hourly — 24 values for target day D |
| Developed by | Team hot_rod, Lastprognose-Challenge SS26 |
| Score metric | Rolling **7-day mean MAE** (MW); lower is better |
| Cadence | Submit by **D−1 23:59 CET** for target day D |
| Determinism | Reproducible: fixed seeds, `PYTHONHASHSEED=0`, snapshotted inputs |
| Built on | `spotforecast2-safe` (T. Bartz-Beielstein) + LightGBM |

The runtime stack is a deliberately small, allow-listed set (numpy, pandas,
scikit-learn, lightgbm, holidays, astral, entsoe-py, requests, statsmodels). Heavy
or non-deterministic libraries (torch, tensorflow, optuna, xgboost, catboost,
matplotlib, plotly) are excluded to keep the dependency and CVE surface small and
the pipeline bit-reproducible. Every model in Voltcrown, including the auxiliary
nowcaster described in §4, is LightGBM.

## 2. Intended use and scope

Voltcrown produces one daily competition submission: a 24-value MW forecast of the
next day's German load. Because the score is a tail-dominated rolling mean — a
handful of hard days (regime shifts, anomalous holidays, days when the weather
forecast was wrong) carries the result — the model is tuned to **contain the worst
days**, not merely to be good on average.

Out of scope: this is not a general forecasting library; it covers the DE bidding
zone only; it produces point forecasts with no calibrated uncertainty; it must not
be used with realised same-day exogenous data (that would leak — §6), nor
repurposed for grid operation without independent validation.

## 3. How it works (plain language)

The engine is **gradient-boosted decision trees** (LightGBM): hundreds of small
yes/no-question trees built in sequence, each correcting the previous ones'
mistakes, summed into a load estimate. It is not a neural network — for tabular
data (rows of numbers) tree ensembles are the strongest family, they run on a CPU,
and they are fully inspectable. All 24 hours of day D are predicted **at once**
(each hour from its own known inputs) rather than feeding hour 1 into hour 2, which
stops errors snowballing across the day. The model is **retrained from scratch
every day**, so it is never stale: years of history anchor the stable patterns and
the newest rows keep it current.

## 4. Technical specification (feature families; exact recipe withheld — see disclosure)

### Estimator

A LightGBM regressor trained under several fixed random seeds, each on a random
subset of rows and features ("subsampling"), with the predictions averaged. The
active ingredient is the subsampling **regularization** — it stops any one model
over-memorising — while the multi-seed averaging adds a smaller, tail-steadying
benefit. A result other teams may find useful: seed-averaging alone contributes
only single-digit MW (the 1/√N floor), so the value is in the regularization, not
the number of seeds.

### Feature families

- **Autoregressive lags**, weekly-aligned (≥1 week back) so they are always settled
  and published before submission, plus **rolling-window statistics** over a
  one-week window (recent level, peak, floor, variability).
- **Calendar**: hour, day-of-week, month and weekend flags, plus **cyclical
  (sine/cosine) encodings** so the model treats time as a loop (23:00 next to
  00:00, December next to January).
- **Sunlight**: deterministic ephemeris (daylight length, day/night) from date and
  coordinates.
- **Holidays**: the union across all sixteen German *Bundesländer*, with
  day-before / day-after / bridge-day flags (a weekday holiday makes load behave
  like a Sunday).
- **Weather**: population-weighted across major German population centres —
  temperature, cloud cover, wind, heating/cooling degree-days (demand rises in both
  cold and heat), and **solar radiation** (a direct read on sunshine, which
  displaces grid load through rooftop PV — historically the most valuable single
  weather feature).
- **Recent-regime signal**: a settled-data feature capturing how far recent load
  has been running above or below its weekly-seasonal level — a denoised "what
  regime are we in now" gauge, used as a model *input* (not as an output bias
  correction, which was tested and rejected).
- **Short-horizon autoregressive information** made usable at the publication
  frontier via an auxiliary LightGBM **nowcaster** that estimates the most recent,
  not-yet-published hours; the main model consumes them under a strict
  publication-time cutoff (§6).
- **Multi-model weather inputs**: the target-day weather is a blend of several
  independent numerical-weather-prediction (NWP) models rather than a single
  forecast, which cancels independent forecast errors and helps the weather-driven
  worst days.

### What v7.9 added over its predecessor

Three components, each attacking a structural error source identified in a prior
investigation, and validated together because the decomposition showed they are
complementary rather than redundant: the recent-regime input signal, the
nowcast-imputed short lags, and the multi-model weather blend. Measured in
isolation each component is modest and some even fail the tail check alone; the
three together pass cleanly (see §7) — measured synergy, not a kitchen sink.

### Illustrative configuration

Moderate-depth LightGBM trees, a low learning rate with a high tree count,
row/feature subsampling, predictions averaged over a small number of fixed seeds.
These are the deployment's choices, shown to characterise the model — **not
recommended defaults**; the best values are data- and feature-specific and, in our
experience, mattered far less than the feature and methodology decisions.

### Design objectives

Deterministic (same input → same output, bit for bit); leakage-free by construction
(§6); fail-safe (missing or invalid input raises, it is never silently imputed).

## 5. Interfaces and runtime

Inputs: ENTSO-E DE load history; day-ahead/forecast weather for the target day and
observed weather for training; deterministic calendar and ephemeris. Output: 24
hourly MW values for day D. CPU-only, no GPU; seconds to minutes per daily forecast.
Each run writes a structured audit log (EU AI Act Art. 12 record-keeping).

## 6. Data and operational design domain (the leakage discipline — most worth sharing)

The model is valid for the DE control area with target-day weather available and an
intact load history; gaps **raise** rather than being imputed. Every input must be
genuinely available at D−1 23:59:

- Autoregressive lags are weekly (≥1 week back), so they are settled and published
  before submission.
- Weather is the *forecast* vintage for the target day and *observed* for training
  (a "live-replica" setup), so the model learns the true weather→load map and is
  scored under realistic forecast error.
- The short-horizon nowcast uses a **publication-time cutoff applied identically to
  training and target rows**: a recent value is used only where it would actually
  have been published by submission time; otherwise the nowcaster's estimate stands
  in. The same rule on both sides means the model never learns to rely on data it
  will not have live.
- No other team's, and no TSO's, load forecast is ever used as an input or additive
  term.

Accuracy degrades on regime shifts, anomalous holiday and bridge days, and
operating states scarce in the training data — the tail days the model contains but
cannot eliminate — and on days when the input weather forecast itself is wrong (no
model fixes a wrong input).

## 7. Evaluation

We treat the **evaluation methodology** as the product and recommend it to other
teams:

- **Live-replica backtests over a full year** (forecast weather on the target day,
  observed for training) — short windows misled us more than once.
- **Tail-gated adoption.** A change ships only if it lowers the mean *and* holds the
  tail: a paired improvement whose moving-block-bootstrap confidence interval
  excludes zero, with no regression on the worst-10-day mean or the worst
  rolling-7-day window. The mean alone is never sufficient.
- **No bundling.** Each candidate feature is tested in isolation; only components
  that clear the gate on their own are combined, then re-confirmed. This repeatedly
  caught features that helped the average while quietly worsening the tail.
- **Influence ≠ benefit.** Features are never selected by importance ranking; only
  the tail-gated backtest decides.

The same discipline is shipped as runnable code in [`../tools`](../tools): the
moving-block bootstrap, the tail summary, and the adoption gate are exactly what
`shadow_compare.py` and `backtest.py` apply.

### Lineage results — the deployed model and its two fallbacks

Full-year live-replica backtest (1 Jul 2024 – 30 Jun 2025, 364 days; deploy
configuration, 5 seeds). Lower is better; MW.

| model | role | mean MAE | worst-10-day mean | worst single day | worst rolling-7-day |
| --- | --- | --- | --- | --- | --- |
| v4.5 | second fallback | 1633 | 5572 | 8855 | 3830 |
| v7.5 | validated fallback | 1331 | 4153 | 5078 | 2621 |
| **v7.9** | **deployed** | **1220** | **3842** | 5095 | **2306** |

Paired, robustness-checked (moving-block bootstrap, 95% CI):

- v7.9 over v7.5: **+111 MW mean** (CI [+63, +168]) — tail strongly improved
  (worst-10 −311, worst rolling-7-day −315), win rate 57 %.
- v7.5 over v4.5: **+302 MW mean** (CI [+207, +407]) — better in every season and
  day-type, body and tail.
- Cumulative v4.5 → v7.9 ≈ **−410 MW** mean (~25 % lower error). v7.9's fresh-day
  error is ≈ 2.3 % MAPE, competitive with the TSO's own day-ahead reference on
  normal days.

These are one data vintage on one window; the transferable part is the methodology
above, not the exact MW.

## 8. Model transparency

Point forecasts only; no native uncertainty quantification (a quantile or conformal
layer would be a downstream add-on). White-box — gradient-boosted trees, no opaque
weights; feature attribution via LightGBM's native tree importances, with the
caveat from §7 that importance is influence, not benefit.

## 9. Operation: monitoring and response

The model is retrained daily. We monitor input freshness and coverage, distribution
drift against the training period, the publication lag of the most recent actuals
(which sets how many hours the nowcaster fills), and daily error against a
reference. Responses: submit early and confirm acceptance — a missed submission
carried forward is the single biggest ranking threat — and fall back to the
validated previous version (v7.5, then v4.5) if forecast generation fails.

## 9a. Determinism audit (planned, private)

Voltcrown v7.9 is deterministic by construction (CR-2): fixed seeds,
`PYTHONHASHSEED=0`, snapshotted inputs, and no randomness outside LightGBM's seeded
sampling. So that this can be *verified* independently rather than merely asserted,
the team will make the v7.9 submission script available **at a later date, to a
selected group**, for a reproducibility audit. The auditors will run the script to
produce a forecast for a **past target date that the team already submitted with
v7.9**, and check that it reproduces the **same 24 hourly values and the same MAE** —
confirming the model and pipeline are bit-reproducible end to end.

This audit is conducted **privately, with a selected group**. The full v7.9 script is
not part of this public release while the model is in use (consistent with the
*Disclosure status* above); the private audit is how determinism is demonstrated to
third parties in the meantime. (For a fully open, runnable model, see the v4.5
reference release in [`../models/v4.5/`](../models/v4.5/).)

## 10. Compliance support

Built to the EU AI Act articles the course emphasises: data governance via the
leakage rules and fail-safe-on-missing (Art. 10); this card plus internal
documentation as the technical-documentation baseline (Art. 11); per-run structured
audit logs (Art. 12); white-box, re-derivable transforms (Art. 13); deterministic,
reproducible computation (Art. 15). A small, audited, deterministic dependency stack
keeps the CVE and attack surface small. The project's four "code rules" map onto
this: CR-1 git-audited submissions, CR-2 determinism (pinned seeds, no RNG outside
LightGBM's seeded sampling), CR-3 raise-on-missing throughout, CR-4 a LightGBM-only
modelling stack with the heavy/non-deterministic deny-list enforced.

## 11. Limitations and ethical considerations

Voltcrown forecasts one bidding zone for a competition and must not be repurposed
for grid operation or any safety-critical use without independent validation. It can
be wrong on rare regimes and on days when the input weather forecast is wrong. The
point forecasts carry no calibrated uncertainty.

## 12. Citation and contact

Team hot_rod, "Voltcrown" model line, Lastprognose-Challenge SS26. Built on
`spotforecast2-safe` (T. Bartz-Beielstein) and LightGBM. See [`CITATION.cff`](../CITATION.cff).

## 13. Disclaimer

Provided as-is for a forecasting competition. Forecast accuracy is bounded by the
data and the weather forecast; no warranty, and not for safety-critical use.

---

*Intentionally not in this public card:* exact hyperparameters, seeds, lag offsets
and window lengths, the recency-offset and nowcaster formulas, the specific NWP
models, and any not-yet-deployed candidates under evaluation. These are released
when the model is retired (see *Disclosure status*). The methodology in §6–§7 is the
part we hope is useful.
