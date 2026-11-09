"""Sensitivity of the deployed models to sensor error (no new data).

The benchmark's models were trained on an Alphasense OPC-N3 and SHT31 inputs; the
Sensa wearable carries a Sensirion SEN55. Until a co-located campaign exists, this
script asks how much the deployed models' outputs move under the kinds of error a
different low-cost sensor introduces: a multiplicative PM gain error, an additive PM
offset, a temperature offset (self-heating in an enclosure) and a relative-humidity
offset. Every test window of every fold is perturbed and run through the C library
(c/aq_native, the code that runs on the board); AP, F2 and the share of changed
decisions are compared with the unperturbed run.

    python analysis/robustness.py [--jobs 2]

Writes analysis/out/robustness_raw.json (one row per fold, model and scenario) and
analysis/out/robustness.json (means over folds).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE = os.path.join(ROOT, "c", "aq_native")
OUT = os.path.join(ROOT, "analysis", "out")
T, RH, AH, PM25, PM10 = range(5)

# name -> (pm_gain, pm_offset_ug, t_offset_C, rh_offset_pct)
SCENARIOS = {
    "baseline": (1.0, 0.0, 0.0, 0.0),
    "pm_gain_0.70": (0.70, 0.0, 0.0, 0.0),
    "pm_gain_0.85": (0.85, 0.0, 0.0, 0.0),
    "pm_gain_1.15": (1.15, 0.0, 0.0, 0.0),
    "pm_gain_1.30": (1.30, 0.0, 0.0, 0.0),
    "pm_gain_1.50": (1.50, 0.0, 0.0, 0.0),
    "pm_offset_-5": (1.0, -5.0, 0.0, 0.0),
    "pm_offset_+5": (1.0, 5.0, 0.0, 0.0),
    "t_offset_+3": (1.0, 0.0, 3.0, 0.0),
    "rh_offset_-10": (1.0, 0.0, 0.0, -10.0),
}


def abs_humidity(t_c, rh):
    """g/m^3 from temperature and relative humidity (Magnus formula)."""
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    return 216.7 * (rh / 100.0) * es / (273.15 + t_c)


def perturb(win, gain, off, dt, drh):
    w = win.astype(np.float64).copy()
    for c in (PM25, PM10):
        w[..., c] = np.clip(w[..., c] * gain + off, 0.0, None)
    if dt or drh:
        # a sensor that reads warm/dry: the reported RH changes with the reported
        # temperature (same vapour), plus any RH offset; AH is recomputed from both
        t0, rh0 = w[..., T], w[..., RH]
        ah_true = abs_humidity(t0, rh0)
        t1 = t0 + dt
        rh1 = np.clip(rh0 * abs_humidity(t0, 100.0) / abs_humidity(t1, 100.0) + drh, 0.0, 100.0)
        w[..., T], w[..., RH] = t1, rh1
        # keep the dataset's own AH scale: rescale its column by the modelled change
        w[..., AH] = w[..., AH] * abs_humidity(t1, rh1) / np.maximum(ah_true, 1e-6)
    return w.astype("<f4")


def run_c(model, wfile):
    out = subprocess.run([NATIVE, "-p", model, wfile], capture_output=True, text=True,
                         check=True).stdout.split()
    a = np.array(out, float).reshape(-1, 2)
    return a[:, 0], a[:, 1] > 0.5


def metrics(y, s, d):
    from sklearn.metrics import average_precision_score
    t = y > 0
    tp, fp, fn = int((d & t).sum()), int((d & ~t).sum()), int((~d & t).sum())
    den = 5 * tp + 4 * fn + fp
    ap = float(average_precision_score(t, s)) if 0 < t.sum() < len(t) else float("nan")
    return ap, (5 * tp / den if den else 0.0)


def job(args):
    h, design, fold, arch, model = args
    d = os.path.join(ROOT, "export_out", f"h{h}", design, fold)
    win = np.fromfile(os.path.join(d, "test_windows.f32"), "<f4").reshape(-1, 24, 5)
    y = np.fromfile(os.path.join(d, "test_y.u8"), np.uint8)
    rows, base = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for name, p in SCENARIOS.items():
            wf = os.path.join(tmp, "w.f32")
            perturb(win, *p).tofile(wf)
            s, dec = run_c(model, wf)
            ap, f2 = metrics(y, s, dec)
            if base is None:
                base = (ap, f2, dec)
            rows.append({"horizon": h, "design": design, "fold": fold, "arch": arch,
                         "scenario": name, "n": int(len(y)), "ap": ap, "f2": f2,
                         "d_ap": ap - base[0], "d_f2": f2 - base[1],
                         "flipped": int((dec != base[2]).sum()),
                         "positives": int(dec.sum())})
    return rows


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = a.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = json.load(open(os.path.join(ROOT, "export_out", "summary.json")))
    raw_path = os.path.join(OUT, "robustness_raw.json")
    rows = json.load(open(raw_path)) if os.path.exists(raw_path) else []
    done = {(r["horizon"], r["design"], r["fold"], r["arch"]) for r in rows}
    jobs = []
    for m in summary.values():
        h, des, fold = m["horizon"], m["design"], m["outer"]
        nd = os.path.join(ROOT, "export_out", f"h{h}", des, fold)
        cd = os.path.join(ROOT, "export_classical", f"h{h}", des, fold)
        models = {m["arch"]: os.path.join(nd, f"{m['arch']}_f32.aqm")}
        if m["arch"] == "GRU":       # classical models once per fold
            for c in ("WHO", "LR", "GBT"):
                models[c] = os.path.join(cd, f"{c}.aqm")
        for arch, path in models.items():
            if (h, des, fold, arch) not in done:
                jobs.append((h, des, fold, arch, path))
    # slow ones first so the pool stays busy
    jobs.sort(key=lambda j: -os.path.getsize(j[4]) * os.path.getsize(
        os.path.join(ROOT, "export_out", f"h{j[0]}", j[1], j[2], "test_windows.f32")))
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows += r
            with open(raw_path, "w") as f:
                json.dump(rows, f)
            print(f"{i + 1}/{len(jobs)} {r[0]['arch']} h{r[0]['horizon']} {r[0]['design']} "
                  f"{r[0]['fold']}", flush=True)
    # means over folds, per model family and scenario
    agg = {}
    for arch in ("WHO", "LR", "GBT", "GRU", "TCN", "FUSION"):
        for sc in SCENARIOS:
            rs = [r for r in rows if r["arch"] == arch and r["scenario"] == sc]
            if not rs:
                continue
            n = sum(r["n"] for r in rs)
            dap = [r["d_ap"] for r in rs if np.isfinite(r["d_ap"])]
            agg[f"{arch}/{sc}"] = {
                "folds": len(rs), "windows": n,
                "mean_d_ap": float(np.mean(dap)), "worst_d_ap": float(np.min(dap)),
                "mean_d_f2": float(np.mean([r["d_f2"] for r in rs])),
                "flipped_pct": 100.0 * sum(r["flipped"] for r in rs) / n}
    with open(os.path.join(OUT, "robustness.json"), "w") as f:
        json.dump(agg, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
