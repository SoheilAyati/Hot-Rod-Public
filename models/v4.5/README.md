# Voltcrown v4.5 — full reference release

This folder contains the **complete, runnable** v4.5 submission model, released in
full as a good-will contribution to the Lastprognose-Challenge community. Unlike the
team's current models — documented at the methodology level while in use — v4.5 is
**fully open**: every hyperparameter, training window, feature and the
bias-correction algorithm is here in the code and in the
[model card](../../model_cards/model_card_v45.md).

v4.5 is two generations behind the deployed model. It is a single tuned LightGBM
with a curated training window and an out-of-sample scalar bias correction — simple,
robust, and a clean starting point to learn from or to use as a baseline to beat.

## What's here

- `make_submission_v45.py` — the model. Identical to the operational script except
  that the audit-log directory is taken from `HOTROD_LOG_DIR` (default `./logs`)
  instead of a hard-coded path. The forecast is unaffected (the log location does
  not enter the computation), so output CSVs stay bit-identical to operational runs.

## Requirements

```sh
pip install "spotforecast2-safe>=15.5,<16" lightgbm holidays numpy pandas
export ENTSOE_API_KEY="your-token"   # free at https://transparency.entsoe.eu/
```

`spotforecast2-safe` is the open course package the model is built on
([upstream](https://github.com/sequential-parameter-optimization/spotforecast2-safe)).
It supplies the ENTSO-E downloader, the Open-Meteo weather service, the fail-safe
interpolation, and the AI-Act audit logger.

## Run it

```sh
export PYTHONHASHSEED=0                       # determinism (CR-2)
python make_submission_v45.py \
    --team your_team \
    --target-date 2026-05-29 \
    --repo-root /path/to/your/leaderboard/checkout
```

It writes a validated 24-row CSV to `<repo-root>/submissions/<team>/<date>.csv`.
Add `--no-bias-correction` to get the uncorrected variant (which, over a full year,
is actually better on the mean — see the model card §7).

## The recipe, in brief

- **Estimator:** a single LightGBM regressor (no ensemble) — `n_estimators=3000`,
  `learning_rate=0.05`, `num_leaves=23`, `min_child_samples=40`, `max_depth=-1`,
  `random_state=2026`.
- **Training window:** 2018-01-01…2019-12-31 together with 2024-01-01…D−1, deliberately
  skipping the anomalous 2020–2023 years.
- **Features (~39):** weekly lags `lag_168/336/504/672`; `roll_mean_last_week`; raw
  calendar (`hour`, `dayofweek`, `month`, `is_weekend`); per-state holiday one-hots
  plus `is_any_holiday`, day-before/after and `is_bridge_day`; population-weighted
  weather over six cities (`de_temperature_2m`, `de_cloud_cover`, `de_wind_speed_10m`,
  `de_hdd` base 15 °C, `de_cdd` base 22 °C). No cyclical encodings, sunlight or solar
  radiation (those arrived in later versions).
- **Bias correction (default on):** hold out the last 14 days, train a bias model on
  everything before, take `median(actual − predicted)` over non-holiday holdout
  hours, clip to ±2500 MW, and add it to the final forecast.

Full details, results and the leakage/compliance discussion are in the
[model card](../../model_cards/model_card_v45.md).

## License & disclaimer

MIT (see the repository [LICENSE](../../LICENSE)). Provided as-is for a forecasting
competition; not for grid operation or any safety-critical use.
