# Voltcrown v7.9.5 — full reference release

This folder contains the **complete, runnable** v7.9.5 submission model, released in
full now that the Lastprognose-Challenge SS26 has concluded. Every hyperparameter, the
recency-offset feature, the nowcaster, the multi-NWP weather blend and the automatic
corrupt-input guard are here in the code and in the
[model card](../../model_cards/model_card_v79.md).

v7.9.5 is the **v7.9 model plus an automatic corrupt-hour detector**: it is
byte-for-byte v7.9 in every modelling respect, with an added, safe input-hygiene step.
With `--no-auto-detect` and an empty exclusion list it is identical to v7.9. v7.9 was
the team's live model in 2026-06 → 2026-07-10 and later the first backup.

## What's here

- `make_submission_v795.py` — the model. Identical to the operational script except
  that the audit-log directory is taken from the `HOTROD_LOG_DIR` environment variable
  (default `./logs`) instead of a hard-coded path. The forecast is unaffected, so
  output CSVs stay bit-identical to operational runs.

## Requirements

```sh
pip install "spotforecast2-safe>=15.5,<16" lightgbm holidays astral numpy pandas requests
export ENTSOE_API_KEY="your-token"   # free at https://transparency.entsoe.eu/
```

`spotforecast2-safe` is the open course package the model is built on
([upstream](https://github.com/sequential-parameter-optimization/spotforecast2-safe)):
the ENTSO-E downloader, the Open-Meteo weather service, the fail-safe interpolation,
and the AI-Act audit logger.

## Run it

```sh
export PYTHONHASHSEED=0                       # determinism (CR-2)
python make_submission_v795.py \
    --team your_team \
    --target-date 2026-07-05 \
    --repo-root /path/to/your/leaderboard/checkout
```

It writes a validated 24-row CSV to `<repo-root>/submissions/<team>/<date>.csv`.
Useful flags: `--no-auto-detect` (disable the automatic corrupt-hour detector ⇒ the
run becomes plain v7.9); `--exclude-hours "2026-06-10:5,2026-06-11:6-11"` (manually
NaN specific corrupt hours); `--seeds N` (ensemble size, default 5).

## The recipe, in brief

v7.9.5 = **v7.5 base + three v7.9 components + the v7.9.5 guard**:

- **v7.5 base:** the same 5-seed subsampled LightGBM ensemble (`n_estimators=3000`,
  `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=40`, `subsample=0.7`,
  `colsample_bytree=0.9`, seeds 2026–2030, bias off), same `KEEP_WINDOWS` and the
  ~51-feature matrix.
- **`recent_offset`:** trailing 7-day mean of `y(t) − y(t−168)` evaluated at the
  publication anchor — a settled-only, denoised regime-level *input feature* (not an
  output bias correction).
- **Nowcast-imputed short lags (`lag_24_nc`, `lag_48`):** an auxiliary LightGBM
  (v7.5 features + raw lag_48/72, 800 trees, seed 2026) estimates the most recent
  unpublished hours; the short lags use the real actual where published, else the
  nowcast — under a publication-time cutoff applied identically to training and target.
- **Multi-NWP target weather (`wxmean`):** the target day's weather is the
  population-weighted mean of `icon_seamless + ecmwf_ifs025 + gfs_seamless` instead of
  a single forecast; training weather stays observed (live-replica).
- **v7.9.5 corrupt-input guard:** automatically flags hours deviating > ~8 GW from the
  holiday-robust 6-week profile for ≥ 2 consecutive hours within the last 14 days
  (excluding public holidays), unions them with any manual `--exclude-hours`, NaNs
  exactly those hours and patches the feature source with the clean same-hour value
  1–4 weeks earlier. A false positive only swaps one input cell for last week's same
  hour — far milder than an output clamp.

Full details, the component decomposition and the full-year results are in the
[model card](../../model_cards/model_card_v79.md).

## License & disclaimer

MIT (see the repository [LICENSE](../../LICENSE)). Provided as-is for a forecasting
competition; not for grid operation or any safety-critical use.
