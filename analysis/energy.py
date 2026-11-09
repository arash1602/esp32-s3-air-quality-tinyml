"""Energy per prediction from a meter log and the hil.py energy run.

    python analysis/energy.py --log energy_log.csv --run hil_results_timing/energy.json \
        --latency hil_results_timing_v4/latency.json [--latency ...] -o energy_results.json

For each model, hil.py ran inference back to back for `seconds` (busy), then idled for
`seconds` (idle). The mean power of each period is taken from the log, leaving out
`--trim` seconds at both ends of each period (USB command traffic, period edges).
Energy per prediction = (P_busy - P_idle) * seconds / inferences: the board's constant
draw (regulator, LEDs, USB) cancels. Power is measured at the USB input (5 V side).
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, nargs="+")
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--latency", nargs="*", default=[])
    ap.add_argument("--trim", type=float, default=2.0)
    ap.add_argument("-o", "--out", default="energy_results.json")
    a = ap.parse_args()
    t, p = [], []
    for path in a.log:
        with open(path) as f:
            for r in csv.DictReader(f):
                ts = r.get("epoch_s") or r.get("epoch")        # USB logger or BLE (fnb58) CSV
                if not ts or not r.get("voltage_V") or not r.get("current_A"):
                    continue
                t.append(float(ts))
                p.append(float(r["voltage_V"]) * float(r["current_A"]))
    t, p = np.array(t), np.array(p)
    o = np.argsort(t); t, p = t[o], p[o]
    lat = {}
    for path in a.latency:
        for r in json.load(open(path)):
            if r.get("weights_in", "flash") == "flash":
                lat[r["index"], r.get("fold"), r["arch"], r["precision"]] = r["median_us"]
    out = []
    for path in a.run:
        for r in json.load(open(path)):
            s = r["seconds"]
            def mean(t0):
                m = (t >= t0 + a.trim) & (t <= t0 + s - a.trim)
                return (float(p[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)
            pb, nb = mean(r["busy_start"])
            pi, ni = mean(r["idle_start"])
            dp = pb - pi
            e = dp * s / r["inferences"] if r["inferences"] else float("nan")
            rec = {k: r[k] for k in ("index", "horizon", "design", "arch", "precision") if k in r}
            rec |= {"fold": r.get("fold"), "inferences": r["inferences"], "seconds": s,
                    "p_busy_W": pb, "p_idle_W": pi, "dP_W": dp, "n_busy": nb, "n_idle": ni,
                    "energy_per_pred_J": e,
                    "latency_from_run_us": 1e6 * s / r["inferences"] if r["inferences"] else None}
            k = (r["index"], r.get("fold"), r["arch"], r["precision"])
            if k in lat:
                rec["latency_median_us"] = lat[k]
            out.append(rec)
            print(f"[{r['index']:2d}] {r['arch']:6s} {r['precision']:4s} busy {pb*1000:7.1f} mW "
                  f"idle {pi*1000:7.1f} mW  dP {dp*1000:6.1f} mW  "
                  f"{e*1e3:9.4f} mJ/prediction  ({nb}/{ni} samples)")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
