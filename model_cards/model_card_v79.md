# Model Card — Voltcrown v7.9 / v7.9.5 (Team hot_rod)

Public model card for **Voltcrown v7.9** and its point-release **v7.9.5**, live models of
Team *hot_rod* in the Lastprognose-Challenge SS26 (TH Köln, *Data-Driven Modeling and
Optimization*, summer 2026). It follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy and is written to stand on its own, so it can be read without reference to the
other cards in this folder. ("Voltcrown" is the team's name for its prediction-model
line; versions keep the v-numbering.)

> **Fully released.** The challenge has concluded, so v7.9 and v7.9.5 are released **in
> full** and open for anyone to use — this card discloses every hyperparameter, training
> window and feature, the **recency-offset** regime signal, the auxiliary **nowcaster**,
> the **multi-NWP weather blend**, and the **v7.9.5 corrupt-hour detector**, and the
> **runnable reference implementation is included** at
> [`../models/v7.9.5/make_submission_v795.py`](../models/v7.9.5/make_submission_v795.py).
> The released script is v7.9.5 — v7.9 with an automatic corrupt-hour detector; run with
> `--no-auto-detect` and no manual exclusions it is byte-identical to v7.9.

## 1. Model details

| Field | Value |
| --- | --- |
| Name | Voltcrown v7.9 / v7.9.5 (Team hot_rod), `make_submission_v795.py` |
| Status | Released reference (was the team's live model, 2026-06 → 07-10) |
| Type | Deterministic, daily-retrained, direct 24-hour multi-output load forecaster: a 5-seed subsampled **LightGBM** ensemble over the v7.5 feature base **+ recent-regime signals + multi-NWP target weather**, with an auxiliary LightGBM **nowcaster** |
| Target | ENTSO-E "DE Actual Total Load", MW, hourly — 24 values for target day D |
| Developed by | Team hot_rod, Lastprognose-Challenge SS26 |
| Score metric | Rolling **7-day mean MAE** (MW); lower is better |
| Cadence | Submit by **D−1 23:59 CET** for target day D |
| Determinism | `PYTHONHASHSEED=0`; seed base 2026 (main ensemble 2026–2030; nowcaster 2026); snapshotted inputs |
| Reference code | [`../models/v7.9.5/make_submission_v795.py`](../models/v7.9.5/make_submission_v795.py) |
| Built on | `spotforecast2_safe` (T. Bartz-Beielstein) + LightGBM |
| Lineage | v4.5 → v6 → v7 → v7.5 → **v7.9 / v7.9.5** |

The runtime stack is a small, allow-listed set (numpy, pandas, scikit-learn, lightgbm,
holidays, astral, entsoe-py, requests, statsmodels, spotforecast2_safe). Heavy or
non-deterministic libraries (torch, tensorflow, optuna, xgboost, catboost, matplotlib,
plotly) are excluded to keep the dependency and CVE surface small and the pipeline
bit-reproducible. Every model in v7.9, **including the nowcaster**, is LightGBM.

## 2. Intended use and scope

v7.9 produces one daily competition submission: a 24-value MW forecast of the next day's
German load. It is a full model — the team's live workhorse for its deployment window —
and is published here as a complete, reusable reference. Because the score is a
tail-dominated rolling mean, the model is tuned to **contain the worst days**, not merely
to be good on average.

Out of scope: not a general forecasting library; DE control area only; point forecasts
with no calibrated uncertainty; must not use realised same-day exogenous data (that would
leak — §7), nor be repurposed for grid operation without independent validation.

## 3. How it works (plain language)

The engine is the same as its predecessor **v7.5**: a **gradient-boosted decision-tree**
model (LightGBM) — hundreds of small yes/no-question trees built in sequence, each
correcting earlier mistakes — trained under five random seeds with row and feature
subsampling and averaged. All 24 hours of day D are predicted at once (each from its own
known inputs) rather than recursively, so an early-hour error cannot cascade, and the model
is retrained from scratch every day on a **curated window** (2018–2019 with 2024 to the
present, skipping the pandemic and energy-crisis years 2020–2023).

v7.9 adds three things on top of that base, each targeting a weakness v7.5 left open — the
model's blindness to the most recent week, and error in the input weather forecast: a
**recency offset** (how far this week's load level has drifted from a week ago, entered as
an *input feature*, not an output correction); a pair of **short lags** made usable by a
small **nowcaster** that fills the hours ENTSO-E has not yet published; and a **multi-NWP
weather blend** for the target day. They were validated *together* — each pair fails the
tail test, the triple passes with margin (§8), measured synergy rather than a kitchen sink.
v7.9.5 then adds an **automatic guard against corrupted input data** (§5) that never touches
the model itself.

## 4. Technical specification (complete)

### Estimator and base

The main estimator is **identical to v7.5**: a LightGBM regressor trained under 5 seeds
with subsampling and averaged. Output bias correction is **removed** (rejected twice over
a full year — the v4.5 lesson).

| Parameter | Value |
| --- | --- |
| `n_estimators` | 3000 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `min_child_samples` | 40 |
| `max_depth` | −1 (unbounded) |
| `subsample` / `subsample_freq` | 0.7 / 1 (row subsampling) |
| `colsample_bytree` | 0.9 (feature subsampling) |
| `n_seeds` | 5 (seeds 2026–2030, predictions averaged) |
| bias correction | **removed** |

### Training window

`KEEP_WINDOWS = [2018-01-01 … 2019-12-31] ∪ [2024-01-01 … D−1]`; data is pulled from
2018-01-01 and rows outside the kept windows (2020–2023) are dropped before training. The
model retrains daily, so the second window always extends to the hour before day D.

### Feature matrix (~54 columns = v7.5's ~51 + three)

The v7.5 base is unchanged: weekly autoregressive lags `lag_168/336/504/672`; rolling
statistics `roll_mean_last_week`, `roll_max_168`, `roll_min_168`, `roll_std_168` (168h
window, itself shifted 168h back); raw calendar (`hour`, `dayofweek`, `month`,
`is_weekend`) and cyclical encodings (`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`);
sunlight ephemeris (`daylight_hours`, `is_daylight`); a per-holiday one-hot `h_*` over all
sixteen German *Bundesländer* plus `is_any_holiday`, `is_day_before_holiday`,
`is_day_after_holiday`, `is_bridge_day`; and population-weighted weather over six cities
(Berlin 5.0, Hamburg 3.4, Munich 3.0, Rhine-Ruhr 10.0, Frankfurt 5.8, Stuttgart 2.7):
`de_temperature_2m`, `de_cloud_cover`, `de_wind_speed_10m`, `de_shortwave_radiation`, and
degree-days `de_hdd = max(0, 15 − T)`, `de_cdd = max(0, T − 22)`. v7.9 adds exactly three
columns, each explained below.

**(a) `recent_offset` — the recency signal (settled-only regime input).** The trailing
7-day mean of the deviation `dev(t) = y(t) − y(t − 168)` (this hour minus the same hour a
week earlier), read at the per-row publication anchor by forward-fill (a rolling mean over
`7 × 24` hours, `min_periods` half that) so it can never see an unpublished value. It gives
a denoised read on how the current regime's level differs from a week ago, and enters as an
**input feature the trees may split on**, not an additive correction bolted onto the output
— the model decides when and how much to trust it (the distinction that matters in §12).

**(b) `lag_24_nc` / `lag_48` — nowcast-imputed short lags.** Lags at 24h and 48h are the
strongest possible predictors, but at submission time the most recent hours are unpublished,
so the direct model cannot use them. v7.9 fills the gap with a small **auxiliary LightGBM
nowcaster** — the v7.5 feature set plus raw `lag_48`/`lag_72`, **800 trees, seed 2026**,
trained on all clean settled rows up to the last published actual — which predicts every
hour it can, including the unpublished recent gap (whose weather comes from the forecast
blend). Each short lag then uses the **real actual where already published, and the nowcast
otherwise**, chosen per row by the censoring rule below.

**(c) `wxmean` — multi-NWP target weather.** For the target day only, the weather is the
population-weighted mean of **three independent numerical weather predictions —
`icon_seamless` + `ecmwf_ifs025` + `gfs_seamless`** — instead of a single best-match
forecast, cancelling each model's independent error. History and the recent gap keep the
observed / best-match vintage exactly as in v7.5 (live-replica convention). All three NWP
models must return a complete target window or the run **raises** (CR-3: no imputation).

### Live censoring (the v7.9-specific leakage rule)

Every row is built against a **publication anchor**
`anchor(t) = min(floor(t to day) − 24h, last published ENTSO-E actual)`, applied
**identically to training and target rows**. On historical training rows this reduces to
the fixed backtest rule (`floor − 24h`), so there is no train/serve skew. On the target day
it censors against the *actual* publication state: a late ENTSO-E day simply means more
nowcast hours are used, never a silent leak. `lag_48` is censored-and-imputed by the same
rule (live, the late hours of D−2 may still be unpublished). The run prints and logs the
publication gap and the per-lag nowcast-hour counts; a sanity guard aborts if any target
anchor lands more than a few days from the target start (a computation-bug tripwire).

### Design objectives

Deterministic (same input → same output, bit for bit); leakage-free by construction (§7);
fail-safe (missing or invalid weather, rows, or features **raise**, never silently imputed).

## 5. The v7.9.5 corrupt-input guard

v7.9.5 keeps the v7.9 model **byte-for-byte identical** and only hardens how *corrupted
ENTSO-E actuals* are handled — a response to a run of upstream data-quality incidents
during the deployment window (a field-wide morning collapse; lone physically impossible
readings). It is the automatic superset of a manual hour-level exclusion.

**Detection (`detect_corrupt_hours`).** Within the last `AUTO_SCAN_DAYS = 14` days it
unions two complementary rules and flags the offending UTC hours:

- **Profile deviation** — hours whose load deviates by more than `AUTO_DEV_MW ≈ 8 GW` from
  the **holiday-robust 6-week profile** (the same-weekday-hour median over the last six
  weeks, ignoring reference days that are public holidays) for **≥ `AUTO_RUN_H = 2`
  consecutive hours**, with public-holiday days themselves excluded. This reuses v7.9's
  proven warn-only tripwire profile but *acts* on it, catching multi-hour runs.
- **Physical guard** — any hour outside a plausible DE control-area band (~15–90 GW) or an
  isolated spike (a large jump on both sides), catching a lone impossible value that never
  reaches a two-hour run.

The auto-flagged hours are **unioned with a manual `--exclude-hours` list** (e.g.
`"2026-06-10:5,2026-06-11:6-11"`) for the single-hour spikes the profile rule cannot see.

**Action.** For each flagged hour the guard (i) sets the **training target and the recency
offset** to `NaN` there, so the corrupt value is never learned and never counted in the
offset, while keeping the rest of that day's good hours; and (ii) **patches the feature
source** (the series feeding the lags, rolls and nowcaster) with the **clean same-hour
value 1–4 weeks earlier**, skipping any source week that is itself corrupt or a public
holiday, so no lag or rolling column ever sees a `NaN` at predict time. (A `NaN` reaching a
never-trained lag column would be read by LightGBM as zero — a forecast collapse — which
this patch exists to prevent; an unpatchable cell raises rather than proceeds.)

**Why this is safe.** A false positive only swaps **one input cell for last week's same
hour** — a far milder failure than a blacklisted **output clamp** (which the team forbids
on principle). The guard cannot move the forecast far even when it is wrong, and every flag
is warn-logged for operator review (AI Act Art. 14). Running with `--no-auto-detect` and no
manual exclusions makes v7.9.5 **≡ v7.9**.

## 6. Interfaces and runtime

Inputs: ENTSO-E DE load history from 2018; per-model Open-Meteo weather (three NWP models
× six cities for the target day, observed archive for training), snapshotted per target
date; deterministic calendar and ephemeris. Output: 24 hourly MW values for day D, written
to `<repo-root>/submissions/<team>/<date>.csv`. CPU-only; a forecast takes minutes (the
extra cost over v7.5 is the nowcaster fit and the per-model weather fetches — seconds
each). Each run writes a structured audit log (EU AI Act Art. 12 record-keeping).

## 7. Data and operational design domain (the leakage discipline)

Valid for the DE control area with target-day weather available and an intact load history;
gaps **raise** rather than being imputed (`LinearlyInterpolateTS(on_missing="raise")`, and
CR-3 raise-on-missing throughout, including the NWP target window). Every input must be
genuinely available at D−1 23:59:

- Weekly autoregressive lags (≥168h) are settled and published before submission.
- The short lags (24h/48h) are governed by the publication anchor: real actual where
  published, nowcast otherwise — and the nowcaster is trained only on settled rows up to
  the last actual, so it never peeks past the anchor.
- `recent_offset` is a trailing mean read at the anchor by forward-fill; its underlying
  deviation is `NaN` beyond the last actual, so the lookup can never touch unpublished data.
- Weather is the *forecast* vintage for the target day and *observed* for training, and is
  snapshotted so a backtest cannot accidentally see a later revision.
- No other team's, and no TSO's, load forecast is ever used as an input or additive term.

Accuracy degrades on regime shifts, anomalous holiday and bridge days, operating states
scarce in the curated window, and days when the input weather forecast itself is wrong. The
short-lag components degrade **gracefully** (more nowcast hours) when ENTSO-E publishes
late; first-published actuals feeding the nowcaster may still revise by 1–2 days, a known
optimism the `--pub-lag 48` cross-check is used to bound.

## 8. Evaluation

Full-year live-replica backtest (2024-07-01 … 2025-06-30, 364 days; 5-seed deploy config;
forecast weather on the target day, observed weather for training; identical training rows;
moving-block bootstrap CIs). Lower is better; MW.

| metric | v7.5 | **Voltcrown v7.9** |
| --- | --- | --- |
| mean | 1331.1 | **1220.2** |
| median | 1134 | **1079** |
| worst-10 | 4153 | **3842** |
| worst day | 5078 | 5095 |
| worst rolling-7d | 2620.9 | **2305.5** |

Paired (v7.5 − v7.9): **+110.9 MW** mean, 95% block-CI **[+63, +168]** (excludes zero);
rolling-7d **+112.8**, CI [+70, +165]; win rate 57%; the tail gate passes with margin
(worst-10 **−311**, roll7-worst **−315**, worst day flat).

**Component decomposition (1-seed, same year).** Alone: `recent_offset` **+30** (CI spans
0, tail −207), nowcast lags **+55** (mean robust but the **tail fails** — worst day +1495),
`wxmean` **+37** (clean adopt). Pairs: weather+offset **+67**, weather+nowlag **+84** —
**both fail the rolling-7 tail gate**. The **triple is +104.5 with the tail strongly
improved**: the components cover each other's failure modes (the settled offset anchors the
model when the nowcast errs, and vice versa), so each pair fails but the triple passes —
**measured synergy, not a kitchen sink**. `wxmean` alone (**+36.9**, CI [+21, +54], tail
holds) is the **validated minimal fallback** — the one component safe to ship on its own.

## 9. Model transparency

Point forecasts only; no native uncertainty quantification (L1/quantile objectives tested
worse than L2 for MAE). White-box: a gradient-boosted-tree ensemble plus one auxiliary
LightGBM nowcaster; feature attribution via LightGBM-native TreeSHAP
(`predict(pred_contrib=True)` — no `shap`/`matplotlib` dependency). The autoregressive lags
dominate the attribution; the new columns to watch are `recent_offset`, `lag_24_nc` and
`lag_48`. Mean|SHAP| measures **influence, not out-of-sample benefit** — used for
redundancy diagnosis and the Art. 13 record, never as a feature-selection gate.

## 10. Operation: monitoring and response

Retrained daily. Monitor input freshness and coverage; distribution drift against the
curated window; the **publication gap** and the **nowcast-hour counts** (lag_24 / lag_48);
the printed `recent_offset` value; and — with v7.9.5 — the corrupt-hour flags. On a day
with an unusually large publication gap, compare that day's error specifically (the regime
the backtest only approximates). The overriding operational rule is to submit early and
confirm acceptance: a missed submission carried forward is the single biggest ranking
threat. Keeping a validated fallback (v7.5) writing to a shadow folder is what lets a live
decision stay evidence-based.

## 11. Compliance support

Built to the EU AI Act articles the course emphasises: data governance via the leakage
rules and fail-safe-on-missing (Art. 10); this card and the audit log as the
technical-documentation baseline (Art. 11); per-run structured audit logs (Art. 12);
white-box, re-derivable transforms (Art. 13); the warn-only corrupt-hour flags surfaced for
operator review (Art. 14); deterministic, reproducible computation (Art. 15). The four code
rules apply: CR-1 git-audited submissions; CR-2 determinism (seed base 2026;
`PYTHONHASHSEED=0`; no RNG outside LightGBM's seeded fit and sampling; snapshotted inputs);
CR-3 raise-on-missing (rows, weather, features, and the NWP target window); CR-4 a
LightGBM-only modelling stack with the heavy/non-deterministic deny-list enforced.

## 12. Limitations and ethical considerations

Forecasts one bidding zone for a competition; must not be repurposed for grid operation or
safety-critical use without independent validation. Point forecasts carry no calibrated
uncertainty. The nowcaster's short lags rely on first-published actuals that can revise, and
the whole model is only as good as the input weather forecast.

**The instructive limitation is `recent_offset`.** On the full-year window above it helped
the mean and the tail. But on a *fresh* later holdout it **amplified the worst-rolling-7-day
tail** (≈ **2399 MW** with the offset vs **2196 MW** without it), which is exactly why the
successor models **dropped it** — keeping the nowcast lags and the multi-NWP weather, which
carried the mean gain without the tail cost. The transferable lesson generalises the v4.5
finding on output bias correction: **a component that improves the mean is never sufficient
justification on its own — validate on a long, out-of-sample, live-replica window and judge
the tail, not just the mean.** A term that looks settled and denoised can still make the
worst days worse, and the worst days are what the score is decided on.

## 13. Citation and contact

Team hot_rod, "Voltcrown" model line, Lastprognose-Challenge SS26. Built on
`spotforecast2_safe` (T. Bartz-Beielstein) and LightGBM. Model lineage and per-version
cards accompany this file; see [`CITATION.cff`](../CITATION.cff).

## 14. Disclaimer

Provided as-is for a forecasting competition. Forecast accuracy is bounded by the data and
the weather forecast; no warranty, and not for safety-critical use.
