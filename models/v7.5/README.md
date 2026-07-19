# Voltcrown v7.5 — full reference release

This folder contains the **complete, runnable** v7.5 submission model, released in
full now that the Lastprognose-Challenge SS26 has concluded. Every hyperparameter,
the training window, the full feature set and the ensemble recipe are here in the
code and in the [model card](../../model_cards/model_card_v75.md).

v7.5 is the team's most thoroughly full-year-validated model — a 5-seed subsampled
LightGBM ensemble. It was the live model in 2026-05 → 06 and then served as a
fallback. It is a strong, robust baseline to learn from or to beat.

## What's here

- `make_submission_v75.py` — the model. Identical to the operational script except
  that the audit-log directory is taken from the `HOTROD_LOG_DIR` environment
  variable (default `./logs`) instead of a hard-coded path. The forecast is
  unaffected (the log location does not enter the computation), so output CSVs stay
  bit-identical to operational runs.

## Requirements

```sh
pip install "spotforecast2-safe>=15.5,<16" lightgbm holidays astral numpy pandas requests
export ENTSOE_API_KEY="your-token"   # free at https://transparency.entsoe.eu/
```

`spotforecast2-safe` is the open course package the model is built on
([upstream](https://github.com/sequential-parameter-optimization/spotforecast2-safe)).
It supplies the ENTSO-E downloader, the Open-Meteo weather service, the fail-safe
interpolation, and the AI-Act audit logger.

## Run it

```sh
export PYTHONHASHSEED=0                       # determinism (CR-2)
python make_submission_v75.py \
    --team your_team \
    --target-date 2026-05-29 \
    --repo-root /path/to/your/leaderboard/checkout
```

It writes a validated 24-row CSV to `<repo-root>/submissions/<team>/<date>.csv`.
`--seeds N` changes the ensemble size (default 5). `--bias-correction` enables the
v4.5-style scalar bias layer (off by default — tested and rejected, see the card §7).

## The recipe, in brief

- **Estimator:** a LightGBM regressor trained under **5 fixed seeds (2026–2030)** with
  row and feature subsampling, the 24-hour predictions averaged. The active ingredient
  is the **subsampling regularization**, not the averaging — an ablation showed a
  single subsampled member captures ~82% of the gain and 5 seeds add only ~6.5 MW.
- **Hyperparameters (tuned, deployed):** `n_estimators=3000`, `learning_rate=0.05`,
  `num_leaves=31`, `min_child_samples=40`, `subsample=0.7`, `subsample_freq=1`,
  `colsample_bytree=0.9`, `max_depth=-1`. Bias correction **off**.
- **Training window:** `KEEP_WINDOWS = [2018-01-01…2019-12-31] ∪ [2024-01-01…D−1]`,
  deliberately skipping the anomalous 2020–2023 years.
- **Features (~51):** weekly lags `lag_168/336/504/672`; rolling
  `mean/max/min/std` over a 168h window; raw calendar (`hour`, `dayofweek`, `month`,
  `is_weekend`) plus cyclical sine/cosine encodings; sunlight ephemeris
  (`daylight_hours`, `is_daylight`); the 16-Bundesland holiday union with
  day-before/after and bridge-day flags; population-weighted weather over six cities
  (`de_temperature_2m`, `de_cloud_cover`, `de_wind_speed_10m`, `de_hdd` base 15 °C,
  `de_cdd` base 22 °C, `de_shortwave_radiation` — solar GHI).

Full details, results and the leakage/compliance discussion are in the
[model card](../../model_cards/model_card_v75.md).

## License & disclaimer

MIT (see the repository [LICENSE](../../LICENSE)). Provided as-is for a forecasting
competition; not for grid operation or any safety-critical use.
