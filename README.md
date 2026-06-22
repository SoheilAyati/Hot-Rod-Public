# Hot Rod — public model cards & forecasting toolkit

[![CI](https://github.com/SoheilAyati/Hot-Rod-Public/actions/workflows/ci.yml/badge.svg)](https://github.com/SoheilAyati/Hot-Rod-Public/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/SoheilAyati/Hot-Rod-Public/badge)](https://scorecard.dev/viewer/?uri=github.com/SoheilAyati/Hot-Rod-Public)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The public home of Team **hot_rod** ("Voltcrown") from the **Lastprognose-Challenge
SS26** — the live electricity-load forecasting challenge in TH Köln's *Data-Driven
Modeling and Optimization* course. Every day the team forecasts the next day's German
hourly grid load (ENTSO-E "DE Actual Total Load", MW), scored on the rolling 7-day
mean MAE.

This repository publishes the two things meant to be shared:

1. **[Model cards](model_cards/)** — what our models are, how they work, and how they
   actually perform, written at a level other teams can learn from.
2. **[A forecasting toolkit](tools/)** — dependency-light tools to validate, score,
   compare and backtest *any* team's daily forecasts, encoding the evaluation
   discipline we recommend.

> The cards are deliberately written at the **methodology level** while a model is in
> use: the design rationale and measured results are here, the exact hyperparameters
> are released when a version is fully retired. See each card's *Disclosure status*.

## The model, in one paragraph

The deployed model, **Voltcrown v7.9**, is a deterministic, daily-retrained LightGBM
ensemble that predicts all 24 hours of the next day at once from calendar, weather,
holiday and autoregressive features, extended with a recent-regime signal, an
auxiliary "nowcaster" for the most recent unpublished hours, and a blend of several
weather models. Over a full live-replica year it scores **1220 MW mean MAE**,
improving on the validated fallback (v7.5, 1331) by +111 MW with a strongly improved
tail. Two earlier models are kept as fallbacks. The full story, with results and
confidence intervals, is in **[model_cards/](model_cards/)**:

| Version | Role | mean MAE\* | Card |
| --- | --- | --- | --- |
| v4.5 | second fallback | 1633 | [model_card_v45.md](model_cards/model_card_v45.md) |
| v7.5 | first fallback | 1331 | [model_card_v75.md](model_cards/model_card_v75.md) |
| **v7.9** | **deployed** | **1220** | [**model_card_v79.md**](model_cards/model_card_v79.md) |

\* Full-year live-replica backtest, 1 Jul 2024 – 30 Jun 2025, MW; lower is better.

## The toolkit

`numpy` + `pandas` only (live ENTSO-E actuals are an optional extra); everything also
runs fully offline. See **[tools/README.md](tools/README.md)** for the full guide.

| Tool | What it does |
| --- | --- |
| `validate_submission.py` | Pre-flight a submission CSV (24 rows, UTC grid, no gaps/NaNs, right day). |
| `score_submission.py` | MAE of a submission against actuals (offline or live). |
| `rolling_mae.py` | The leaderboard metric: per-day and rolling 7-day mean MAE over a folder. |
| `shadow_compare.py` | A/B two strategies with the **tail-gated adoption verdict**. |
| `backtest.py` | Rolling-origin backtest harness for *your* forecaster, with the same gate. |
| `baselines.py` | Reference forecasters to beat (`seasonal_naive`, `weekly_profile`). |
| `plot_field.py` | Interactive forecast-vs-actual chart as a self-contained HTML file. |

### Quickstart

```sh
git clone https://github.com/SoheilAyati/Hot-Rod-Public.git
cd Hot-Rod-Public
python -m pip install -r tools/requirements.txt

# is my submission well-formed, and what would it score?
python tools/validate_submission.py examples/sample_submission.csv
python tools/score_submission.py examples/sample_submission.csv \
    --actual-csv examples/actuals_2026.csv

# should I switch from strategy A to strategy B? (the adoption gate)
python tools/shadow_compare.py --a examples/submissions_A --b examples/submissions_B \
    --actuals examples/actuals_2026.csv --label-a seasonal --label-b weekly
```

The bundled **[examples/](examples/)** dataset is synthetic, so every command above
runs with no API key.

## The idea worth stealing

More than any single feature, what carries the score is the **evaluation discipline**,
and it ships here as code:

1. **Live-replica backtests over a full year** — forecast-vintage weather on the
   target day, observed weather for training. Short windows mislead.
2. **A tail-gated adoption rule** — a change ships only when it lowers the mean *and*
   holds the tail: a paired moving-block-bootstrap interval that excludes zero, with
   no regression on the worst-10-day mean or the worst rolling-7-day window. The
   competition score is tail-dominated, so the mean alone is never enough.
3. **No bundling, no importance-based selection** — validate each candidate in
   isolation; treat feature importance as influence, never as proof of benefit.

`shadow_compare.py` and `backtest.py` apply exactly this gate.

## Repository layout

```
Hot-Rod-Public/
├── model_cards/      self-contained public cards: v4.5, v7.5, v7.9 (+ lineage overview)
├── tools/            the forecasting toolkit (+ tests, pinned requirements)
├── examples/         small synthetic dataset so every tool runs offline
├── .github/          CI, CodeQL (SAST), Dependabot, OpenSSF Scorecard workflows
├── SECURITY.md       how to report a vulnerability
├── CONTRIBUTING.md   how to contribute
└── LICENSE           MIT
```

## How it's built

The same posture runs through the models and this repository: a small, audited,
deterministic dependency surface; reproducible-by-construction computation; and
supply-chain hygiene tracked by the **OpenSSF Scorecard** workflow, with GitHub
Actions pinned by commit SHA, Dependabot watching dependencies, and CodeQL for static
analysis. It reflects the course's emphasis on the EU AI Act (data governance,
record-keeping, transparency, robustness) carried over from the forecasting pipeline
into the code that ships it. See
[docs/openssf-scorecard.md](docs/openssf-scorecard.md) for the per-check status and
the activation steps.

## Security, contributing, citation

- Found a vulnerability? See **[SECURITY.md](SECURITY.md)** (please report privately).
- Want to contribute? See **[CONTRIBUTING.md](CONTRIBUTING.md)** and the
  **[Code of Conduct](CODE_OF_CONDUCT.md)**.
- Using this in your own work? Cite it via **[CITATION.cff](CITATION.cff)**.

## Acknowledgements & license

Built on [`spotforecast2-safe`](https://github.com/sequential-parameter-optimization/spotforecast2-safe)
by T. Bartz-Beielstein and on LightGBM. Released under the [MIT License](LICENSE).

*Provided as-is for a forecasting competition. Forecast accuracy is bounded by the
data and the weather forecast; not for grid operation or any safety-critical use.*
