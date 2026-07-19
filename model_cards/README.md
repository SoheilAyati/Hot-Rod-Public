# Model cards — Team hot_rod ("Voltcrown")

Public model cards for Team *hot_rod*'s electricity-load forecasting models in the
Lastprognose-Challenge SS26. Every model forecasts next-day German hourly load
(ENTSO-E "DE Actual Total Load", MW) and is scored on the rolling 7-day mean MAE.
The cards use the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy. Each card is written to be **read on its own** — the shared context
(metric, determinism, leakage discipline, compliance posture) is restated in every
card rather than cross-referenced, so you can open just the one you care about.

> **The challenge has concluded — every model here is now released in full.** Each
> card discloses the complete recipe (exact hyperparameters, seeds, feature formulas
> and algorithms), and each has a **runnable reference implementation** under
> [`../models/`](../models). While a model was live or on the fallback bench its card
> was written at the methodology level; now that the competition is over, that
> reservation is lifted for the whole line.

## The published models

| Version | Card | Role in the final chain | One-line description | mean MAE\* |
| --- | --- | --- | --- | --- |
| v4.5 | [model_card_v45.md](model_card_v45.md) | last-resort fallback | a single tuned LightGBM with a curated training window and a scalar bias term | 1633 |
| v7.5 | [model_card_v75.md](model_card_v75.md) | second fallback | a 5-seed subsampled LightGBM ensemble; bias removed; cyclical + sunlight + solar-radiation features | 1331 |
| v7.9 / v7.9.5 | [model_card_v79.md](model_card_v79.md) | first backup (was live 2026-06 → 07-10) | v7.5 plus a recent-regime signal, nowcast-imputed short lags and a multi-model weather blend, plus an automatic corrupt-input guard | 1220 |
| **v7.11** | [**model_card_v711.md**](model_card_v711.md) | **final live model** (from 2026-07-12) | the v7.10 stack (v7.9 minus the recent-regime signal, multi-NWP mean weather) plus an exact per-hour `solar_elevation` feature | —† |

\* Full-year live-replica backtest, 1 Jul 2024 – 30 Jun 2025 (364 days), deploy
configuration, MW. Lower is better. The full result tables, including the tail
metrics and the paired confidence intervals, are inside each card's evaluation
section. Every model's runnable script is in [`../models/`](../models).

† v7.11 was promoted late in the competition on structural and offline feature
evidence (the `solar_elevation` backtest averaged −17.7 MW/day over 165 days on the
earlier weather chassis) ahead of a full scored-shadow window, so it has no
directly-comparable full-year deploy number of its own — see its card. Its base,
v7.10, is characterised in the v7.9 and v7.11 cards.

## How the line progressed

The lineage is the story of a few ideas compounding — **better features**, **better
regularisation**, and **containing the tail** — each adopted only after it passed a
tail-gated full-year backtest.

- **v4.5** establishes the base: a single LightGBM on calendar, per-state holidays
  and population-weighted temperature with heating/cooling degree-days, trained on a
  curated window that skips the anomalous 2020–2023 years, with a scalar bias
  correction. Simple and robust — which is why it is kept as the deepest fallback.
  (The full year later showed the bias term *hurts* — a preserved cautionary tale.)
- **v7.5** is the inflection point. It adds cyclical calendar encodings, a sunlight
  ephemeris and, most valuably, **solar radiation** (sunshine displaces grid load
  through rooftop PV); it replaces the single model with a **subsampled multi-seed
  ensemble**; and it **removes the scalar bias term**. The result is +302 MW mean
  over v4.5, better in every season and day-type.
- **v7.9 / v7.9.5** adds three complementary signals on top of v7.5 — a denoised
  recent-regime input (`recent_offset`), an auxiliary "nowcaster" that fills the most
  recent unpublished hours under a strict publication-time cutoff, and a blend of
  several independent weather models for the target day. Together they cut another
  +111 MW off the mean and improve the tail (worst-10-day mean −311, worst
  rolling-7-day window −315). v7.9.5 adds an automatic corrupt-hour guard that swaps a
  contaminated input cell for last week's clean value — never an output clamp.
- **v7.11** is the final live model. A fresh holdout showed the recent-regime signal
  *amplified the worst-week tail*, so its successor (v7.10) **removed `recent_offset`**
  and predicted the multi-NWP **mean** weather once; v7.11 then adds a single new
  feature, an exact per-hour **`solar_elevation`** angle, targeting the morning/evening
  ramp. That the mean-improving `recent_offset` was dropped for the tail is the
  line's central lesson in one move.

## The transferable part

The methodology matters more than any one feature, and it is the same across all the
cards:

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

## Disclosure

Now that the competition is over, **the whole Voltcrown line is released in full**:
every card discloses the exact hyperparameters, seeds and feature formulas, and every
model ships a runnable reference script under [`../models/`](../models). During the
competition the cards for live and fallback models were deliberately methodology-level
(families of features, design rationale and measured results, but not the precise
recipe), so nothing published was ever false about what a model did while not handing a
live competitor the exact formula. v4.5 was released in full from the start as an open
reference; the rest of the line now joins it.
