# Voltcrown v7.11 — full reference release

This folder contains the **complete, runnable** v7.11 submission model — the team's
**final live model** — released in full now that the Lastprognose-Challenge SS26 has
concluded. Every hyperparameter, the inherited stack and the one added feature are
here in the code and in the [model card](../../model_cards/model_card_v711.md).

v7.11 = the deployed v7.10 stack **plus one feature, `solar_elevation`**. It carries
no `recent_offset` (removed at v7.10) and no `wxscen`; the weather input is the
multi-NWP mean, and the v7.9.5 hour-level corrupt-input guard is retained.

## What's here

- `make_submission_v711.py` — the model. Identical to the operational script except
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
and the AI-Act audit logger. `astral` (already on the allow-list) supplies the exact
sun geometry.

## Run it

```sh
export PYTHONHASHSEED=0                       # determinism (CR-2)
python make_submission_v711.py \
    --team your_team \
    --target-date 2026-07-13 \
    --repo-root /path/to/your/leaderboard/checkout
```

It writes a validated 24-row CSV to `<repo-root>/submissions/<team>/<date>.csv`.
Useful flags: `--no-auto-detect` (disable the automatic corrupt-hour detector);
`--exclude-hours "YYYY-MM-DD:h[-h]"` (manually NaN corrupt hours); `--seeds N`
(ensemble size, default 5).

## The recipe, in brief

v7.11 = **v7.5 base + nowcast lags + multi-NWP mean weather + `solar_elevation`**
(no `recent_offset`, no `wxscen`, no tuned params), with the v7.9.5 hour-guard:

- **v7.5 base:** the 5-seed subsampled LightGBM ensemble (`n_estimators=3000`,
  `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=40`, `subsample=0.7`,
  `colsample_bytree=0.9`, seeds 2026–2030), `KEEP_WINDOWS` and the ~51-feature matrix.
- **Nowcast-imputed short lags (`lag_24`, `lag_48`):** the auxiliary-LightGBM nowcaster
  fills the most recent unpublished hours under a publication-time cutoff.
- **Multi-NWP target weather (`wxmean`):** population-weighted mean of
  `icon_seamless + ecmwf_ifs025 + gfs_seamless`, predicted once.
- **`solar_elevation` (the one new feature):** the exact solar elevation angle per hour
  (degrees above the horizon at the DE centroid 51.0 °N / 10.0 °E via
  `astral.sun.elevation`, clipped to 0 at night), added inside the feature builder so it
  feeds **both** the main ensemble and the nowcaster. It is computed purely from the
  timestamp: deterministic, no publication-state dependency, no leakage. It targets the
  morning/evening ramp residuals that the binary `is_daylight` and per-day
  `daylight_hours` cannot carry.
- **v7.9.5 hour-level corrupt-input guard:** retained unchanged from v7.9.5.

Full details and the deployment evidence are in the
[model card](../../model_cards/model_card_v711.md).

## License & disclaimer

MIT (see the repository [LICENSE](../../LICENSE)). Provided as-is for a forecasting
competition; not for grid operation or any safety-critical use.
