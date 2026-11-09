"""SEN55-for-OPC-N3 substitution scenarios derived from published co-location data.

AQ-SPEC (South Coast AQMD) evaluated the Alphasense OPC-N3 (Aug-Oct 2018) and the
Sensirion SPS30 (Mar-May 2019) at the same site (Rubidoux) against the same three
reference monitors (GRIMM, BAM, T640), hourly means, fits sensor = a*ref + b. Eliminating
the reference gives SPS30 ~= (a_S/a_O) * OPC + (b_S - a_S/a_O * b_O); the medians over
the 3x3 unit pairs per reference are used. The SPS30 stands in for the SEN55 (reported
6-10 % higher than the SPS30 in a side-by-side study, see the paper).

    python analysis/robustness_aqspec.py [--jobs 2]
writes analysis/out/robustness_aqspec.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness as rb  # noqa: E402

# name -> (gain PM2.5, offset PM2.5, gain PM10, offset PM10), from the 1-h AQ-SPEC fits
SCEN = {
    "via GRIMM": (0.66, -2.5, 0.53, 5.4),
    "via BAM": (0.94, -5.5, 0.89, 1.6),
    "via T640": (1.11, -5.3, 1.27, -5.8),
}


def perturb(win, g25, o25, g10, o10):
    w = win.astype(np.float64).copy()
    w[..., rb.PM25] = np.clip(w[..., rb.PM25] * g25 + o25, 0, None)
    w[..., rb.PM10] = np.clip(w[..., rb.PM10] * g10 + o10, 0, None)
    return w.astype("<f4")


def job(args):
    h, design, fold, arch, model = args
    d = os.path.join(rb.ROOT, "export_out", f"h{h}", design, fold)
    win = np.fromfile(os.path.join(d, "test_windows.f32"), "<f4").reshape(-1, 24, 5)
    y = np.fromfile(os.path.join(d, "test_y.u8"), np.uint8)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        wf = os.path.join(tmp, "w.f32")
        win.astype("<f4").tofile(wf)
        s0, d0 = rb.run_c(model, wf)
        ap0, f20 = rb.metrics(y, s0, d0)
        for name, p in SCEN.items():
            perturb(win, *p).tofile(wf)
            s, dec = rb.run_c(model, wf)
            ap, f2 = rb.metrics(y, s, dec)
            rows.append({"horizon": h, "design": design, "fold": fold, "arch": arch,
                         "scenario": name, "n": int(len(y)), "d_ap": ap - ap0, "d_f2": f2 - f20,
                         "flipped": int((dec != d0).sum())})
    return rows


def main() -> int:
    a = argparse.ArgumentParser(); a.add_argument("--jobs", type=int, default=2)
    args = a.parse_args()
    summary = json.load(open(os.path.join(rb.ROOT, "export_out", "summary.json")))
    jobs = []
    for m in summary.values():
        h, des, fold = m["horizon"], m["design"], m["outer"]
        nd = os.path.join(rb.ROOT, "export_out", f"h{h}", des, fold)
        cd = os.path.join(rb.ROOT, "export_classical", f"h{h}", des, fold)
        jobs.append((h, des, fold, m["arch"], os.path.join(nd, f"{m['arch']}_f32.aqm")))
        if m["arch"] == "GRU":
            for c in ("WHO", "LR", "GBT"):
                jobs.append((h, des, fold, c, os.path.join(cd, f"{c}.aqm")))
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for r in ex.map(job, jobs):
            rows += r
    agg = {}
    for arch in ("WHO", "LR", "GBT", "GRU", "TCN", "FUSION"):
        for sc in SCEN:
            for des in ("all", "single-blocked", "multi"):
                rs = [r for r in rows if r["arch"] == arch and r["scenario"] == sc
                      and (des == "all" or r["design"] == des)]
                n = sum(r["n"] for r in rs)
                agg[f"{arch}/{sc}/{des}"] = {
                    "mean_d_ap": float(np.nanmean([r["d_ap"] for r in rs])),
                    "mean_d_f2": float(np.mean([r["d_f2"] for r in rs])),
                    "flipped_pct": 100.0 * sum(r["flipped"] for r in rs) / n}
    json.dump({"scenarios": SCEN, "rows": rows, "agg": agg},
              open(os.path.join(rb.OUT, "robustness_aqspec.json"), "w"), indent=1)
    for k, v in agg.items():
        if k.endswith("/all"):
            print(f"{k:28s} dAP {100*v['mean_d_ap']:+.1f}  dF2 {100*v['mean_d_f2']:+.1f}  chg {v['flipped_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
