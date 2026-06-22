# Model cards — Team hot_rod ("Voltcrown")

Public model cards for Team *hot_rod*'s electricity-load forecasting models in the
Lastprognose-Challenge SS26. Every model forecasts next-day German hourly load
(ENTSO-E "DE Actual Total Load", MW) and is scored on the rolling 7-day mean MAE.
The cards use the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy. Each card is written to be **read on its own** — the shared context
(metric, determinism, leakage discipline, compliance posture) is restated in every
card rather than cross-referenced, so you can open just the one you care about.

## The three published models

| Version | Card | Role | One-line description | mean MAE\* |
| --- | --- | --- | --- | --- |
| v4.5 | [model_card_v45.md](model_card_v45.md) | second fallback | a single tuned LightGBM with a curated training window and a scalar bias term | 1633 |
| v7.5 | [model_card_v75.md](model_card_v75.md) | first fallback | a multi-seed subsampled LightGBM ensemble; bias removed; cyclical + sunlight + solar-radiation features | 1331 |
| **v7.9** | [**model_card_v79.md**](model_card_v79.md) | **deployed** | v7.5 plus a recent-regime signal, nowcast-imputed short lags, and a multi-model weather blend | **1220** |

\* Full-year live-replica backtest, 1 Jul 2024 – 30 Jun 2025 (364 days), deploy
configuration, MW. Lower is better. The full result tables, including the tail
metrics and the paired confidence intervals, are inside each card's §7.

## How the line progressed

The lineage is the story of two ideas compounding: **better features** and **better
regularisation**, each adopted only after it passed a tail-gated full-year backtest.

- **v4.5** establishes the base: a single LightGBM on calendar, per-state holidays
  and population-weighted temperature with heating/cooling degree-days, trained on a
  curated window that skips the anomalous 2020–2023 years, with a scalar bias
  correction. Simple and robust — which is why it is kept as the deepest fallback.
- **v7.5** is the inflection point. It adds cyclical calendar encodings, a sunlight
  ephemeris and, most valuably, **solar radiation** (sunshine displaces grid load
  through rooftop PV); it replaces the single model with a **subsampled multi-seed
  ensemble**; and it **removes the scalar bias term**, which had been trading body
  accuracy for tail risk. The result is +302 MW mean over v4.5, better in every
  season and day-type.
- **v7.9** is the deployed model. On top of v7.5 it adds three complementary
  signals — a denoised recent-regime input, an auxiliary "nowcaster" that fills the
  most recent unpublished hours under a strict publication-time cutoff, and a blend
  of several independent weather models for the target day. Together they cut another
  +111 MW off the mean and, importantly, **improve the tail** (worst-10-day mean
  −311, worst rolling-7-day window −315).

## The transferable part

The methodology matters more than any one feature, and it is the same across all
three cards:

1. **Live-replica backtests over a full year** — forecast-vintage weather on the
   target day, observed weather for training. Short windows mislead.
2. **A tail-gated adoption rule** — a change ships only when it lowers the mean *and*
   holds the tail (a paired moving-block-bootstrap interval that excludes zero, with
   no regression on the worst-10-day mean or the worst rolling-7-day window).
3. **No bundling and no importance-based selection** — each candidate is validated in
   isolation; feature importance is treated as influence, never as proof of benefit.

That rule is not just prose: it ships as runnable code in [`../tools`](../tools)
(`backtest.py` and `shadow_compare.py` apply exactly this gate), so another team can
adopt the discipline directly.

## A note on disclosure

While a model is deployed or kept as a fallback, its card is written at the
methodology level: families of features, the design rationale and the measured
results, but not the exact hyperparameters, seeds or feature formulas. The complete
technical detail for a version is published once it is fully retired (no longer used
even as a fallback). This keeps the cards honest — nothing stated is false about what
the model does — while not handing a live competitor the precise recipe.
