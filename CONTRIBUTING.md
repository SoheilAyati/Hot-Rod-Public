# Contributing

Thanks for your interest. This is a small public repository from Team *hot_rod* for
the Lastprognose-Challenge SS26: the published model cards and a toolkit other teams
can reuse. Contributions that fix bugs, improve the docs, or make the tools more
portable are very welcome.

## Ground rules

- **Be respectful.** See the [Code of Conduct](CODE_OF_CONDUCT.md).
- **Keep the dependency surface small.** The tools intentionally rely only on
  `numpy` and `pandas` (plus an optional ENTSO-E client). Please do not add heavy or
  non-deterministic dependencies (torch, tensorflow, optuna, xgboost, catboost,
  matplotlib, plotly). The HTML chart in `plot_field.py` is deliberately client-side
  Chart.js for exactly this reason.
- **No competition secrets.** This repo is public during an active competition. The
  model cards are written at the methodology level on purpose (see each card's
  *Disclosure status*); please don't add un-disclosed hyperparameters or not-yet-
  deployed model details.

## Making a change

1. Fork the repository and create a topic branch.
2. Make your change. Keep tools runnable both from `tools/` and as
   `python tools/<tool>.py` from the repo root.
3. Add or update tests under `tools/tests/` and run them:

   ```sh
   cd tools
   python -m pip install -r requirements-dev.txt
   python -m pytest tests -q
   ```

   Keep the suite **offline and deterministic** — the existing tests use a synthetic
   load series rather than the live API, and new tests should too.
4. Keep console output ASCII (it runs on Windows terminals as well), and keep prose
   in clear English.
5. Open a pull request describing *what* changed and *why*. CI (tests, CodeQL) runs
   automatically; please make sure it is green.

## Reporting bugs and ideas

Open a GitHub issue with a clear description and, for a bug, a minimal reproduction
(the smallest CSV and command that shows the problem). For anything security-related,
follow the [security policy](SECURITY.md) instead of opening a public issue.
