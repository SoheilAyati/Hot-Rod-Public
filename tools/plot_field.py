#!/usr/bin/env python
"""
plot_field.py — interactive forecast-vs-actual chart for one day (no heavy deps).

Compare any number of forecasts for a single target day against the actual load
in a self-contained HTML file you open in a browser. Each curve is toggleable
from the legend, and the legend shows each forecast's MAE for the day so you can
rank by eye. Useful for the daily post-mortem: where does my curve miss — the
night trough, the morning ramp, the evening peak?

    python plot_field.py --actual actuals_2026-06-08.csv \
        mine=submissions/mine/2026-06-08.csv \
        baseline=baseline_2026-06-08.csv

Each positional argument is ``label=path.csv`` (a submission CSV). ``--actual``
is optional; with it, MAEs are shown and the actual is drawn in black.

Deliberately no matplotlib and no plotly (kept off the dependency/CVE surface):
the chart is client-side Chart.js loaded from a CDN, and this script only reads
CSVs and writes HTML text. Add ``--no-open`` to skip launching the browser.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

import _common as C

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
           "#17becf", "#bcbd22", "#393b79", "#637939", "#8c6d31", "#843c39",
           "#7b4173", "#3182bd", "#31a354", "#756bb1", "#e6550d", "#636363"]

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Forecast field __DATE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
 body{font-family:system-ui,Segoe UI,sans-serif;margin:18px;color:#1c1c1c;}
 h2{font-weight:600;margin:0 0 2px;} .sub{color:#666;margin:0 0 12px;font-size:14px;}
 .bar{margin:10px 0;} button{margin-right:6px;padding:6px 11px;border:1px solid #ccc;
   border-radius:6px;background:#f6f6f6;cursor:pointer;font-size:13px;}
 button:hover{background:#ececec;} #wrap{max-width:1180px;height:560px;}
 .tip{color:#888;font-size:13px;margin-left:6px;}
</style></head><body>
<h2>Forecast vs actual &mdash; __DATE__</h2>
<p class="sub">__SUB__</p>
<div class="bar">
 <button onclick="setAll(true)">Show all</button>
 <button onclick="setAll(false)">Hide all</button>
 <span class="tip">Click a name in the legend to toggle it.</span>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const LABELS = __LABELS__, DS = __DATASETS__;
const chart = new Chart(document.getElementById('c'), {
  type:'line', data:{labels:LABELS, datasets:DS},
  options:{ responsive:true, maintainAspectRatio:false, spanGaps:true,
    interaction:{mode:'index', intersect:false},
    plugins:{ legend:{position:'right', labels:{boxWidth:12, font:{size:11}}},
      tooltip:{itemSort:(a,b)=>a.parsed.y-b.parsed.y} },
    scales:{ y:{title:{display:true,text:'Load (MW)'}},
             x:{title:{display:true,text:'Hour (UTC)'}} } }
});
function setAll(v){ chart.data.datasets.forEach((d,i)=>chart.setDatasetVisibility(i,v)); chart.update(); }
</script></body></html>
"""


def _arr(series, hours):
    out = []
    for v in series.reindex(hours).to_numpy(dtype=float):
        out.append(None if np.isnan(v) else round(float(v), 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forecasts", nargs="+", help="label=path.csv (one per forecast)")
    ap.add_argument("--actual", default=None, help="actuals CSV for the day")
    ap.add_argument("--date", default=None, help="target day (default: inferred)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    series = {}
    for spec in args.forecasts:
        if "=" not in spec:
            raise SystemExit(f"[fatal] '{spec}' must be label=path.csv")
        label, path = spec.split("=", 1)
        series[label] = C.read_forecast(path)

    date = args.date or next(iter(series.values())).index[0].tz_convert("UTC").normalize()
    hours = C.expected_hours(date)
    actual = C.read_forecast(args.actual).rename("actual") if args.actual else None
    av = actual.reindex(hours).to_numpy(dtype=float) if actual is not None else None

    def mae(s):
        if av is None:
            return float("nan")
        x = s.reindex(hours).to_numpy(dtype=float)
        m = ~np.isnan(x) & ~np.isnan(av)
        return float(np.mean(np.abs(x[m] - av[m]))) if m.sum() else float("nan")

    ranked = sorted(series.items(), key=lambda kv: (np.isnan(mae(kv[1])), mae(kv[1])))

    datasets = []
    if av is not None and np.isfinite(av).any():
        datasets.append(dict(label="ACTUAL", data=_arr(actual, hours),
                             borderColor="#111111", borderWidth=3,
                             pointRadius=0, tension=0.3, order=0))
    for i, (label, s) in enumerate(ranked):
        m = mae(s)
        mtxt = "" if np.isnan(m) else f" ({m:.0f})"
        datasets.append(dict(label=f"{label}{mtxt}", data=_arr(s, hours),
                             borderColor=PALETTE[i % len(PALETTE)],
                             borderWidth=1.8, pointRadius=0, tension=0.3))

    labels = [h.strftime("%H:%M") for h in hours]
    sub = f"{len(series)} forecast(s)"
    if av is not None:
        sub += f" &middot; {int(np.isfinite(av).sum())}/24 actual hours &middot; legend shows MAE (MW)"
    daystr = date if isinstance(date, str) else str(date.date())
    html = (HTML.replace("__DATE__", daystr)
                .replace("__SUB__", sub)
                .replace("__LABELS__", json.dumps(labels))
                .replace("__DATASETS__", json.dumps(datasets)))

    out = Path(args.out) if args.out else Path(f"field_{daystr}.html")
    out.write_text(html, encoding="utf-8")
    print(f"[plot] wrote {out.resolve()}")
    if av is not None:
        print(f"\n  {'forecast':<20}{'MAE':>8}")
        print("  " + "-" * 28)
        for label, s in ranked:
            m = mae(s)
            print(f"  {label:<20}{'n/a' if np.isnan(m) else f'{m:>8.0f}'}")
    if not args.no_open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:  # noqa: BLE001
            print("[plot] open it manually in a browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
