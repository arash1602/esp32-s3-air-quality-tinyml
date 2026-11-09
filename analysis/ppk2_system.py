"""Power of the full Sensa firmware with the hourly warning, from a PPK2 record.

    python analysis/ppk2_system.py <record> [--skip 60] [-o result.json]

<record>.cur/.dig/.json come from tools/ppk2_log.py --minutes N, with the PPK2 supplying
the whole device (development board through its 5 V pin, and the SEN55) and D0 on the
pin that the firmware (firmware_sensa, env sensa-ppk2) raises during each prediction.

Reported: mean current and power of the device over the record (after --skip seconds of
start-up and SEN55 warm-up), the same with the prediction intervals left out, energy per
hour, and per prediction its duration and the energy above the device's power in the
0.5 s before it (the SEN55 and the radio keep drawing, so the local level is used).
A 1-ms-mean trace of the whole record and a full-rate excerpt around one prediction
are saved to <out>_trace.npz for a figure.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def runs(d0):
    e = np.diff(np.concatenate(([0], d0.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--skip", type=float, default=60.0, help="start-up to leave out, s")
    ap.add_argument("--glitch-high", type=int, default=10)
    ap.add_argument("--glitch-low", type=int, default=100,
                    help="low gaps up to this many samples inside a prediction are glitches")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    meta = json.load(open(a.name + ".json"))
    rate, v = meta["rate"], meta["mv"] / 1000.0
    cur = np.memmap(a.name + ".cur", np.float32, "r")
    dig = np.memmap(a.name + ".dig", np.uint8, "r")
    n = min(len(cur), len(dig))
    k0 = int(a.skip * rate)
    if n <= k0 + rate:
        raise SystemExit("record too short")
    c = np.asarray(cur[k0:n], dtype=np.float64) / 1000.0          # mA
    d0 = (np.asarray(dig[k0:n]) & 1).astype(bool)
    s, e = runs(d0)
    keep = (e - s) > a.glitch_high
    s, e = s[keep], e[keep]
    # join a prediction split by a short low glitch on the logic input
    if len(s) > 1:
        join = (s[1:] - e[:-1]) <= a.glitch_low
        s, e = s[np.r_[True, ~join]], e[np.r_[~join, True]]
    t_rec = len(c) / rate
    i_mean = float(c.mean())
    mask = np.ones(len(c), bool)
    pred = []
    for a0, a1 in zip(s, e):
        mask[a0:a1] = False
        b0 = max(0, a0 - int(0.5 * rate))
        base = float(c[b0:a0].mean()) if a0 > b0 else float("nan")
        q = float(c[a0:a1].sum()) / rate                            # mA s
        pred.append({"t_s": a.skip + a0 / rate, "duration_ms": (a1 - a0) / rate * 1e3,
                     "i_mean_mA": float(c[a0:a1].mean()), "i_before_mA": base,
                     "energy_above_before_mJ": v * (q - base * (a1 - a0) / rate)})
    i_wo = float(c[mask].mean())
    res = {"meta": meta, "skip_s": a.skip, "recorded_s": t_rec, "supply_V": v,
           "i_mean_mA": i_mean, "p_mean_mW": v * i_mean,
           "i_without_predictions_mA": i_wo, "p_without_predictions_mW": v * i_wo,
           "energy_per_hour_J": v * i_mean * 1e-3 * 3600,
           "i_p01_p50_p99_mA": [float(x) for x in np.percentile(c[::10], [1, 50, 99])],
           "n_predictions": len(pred), "predictions": pred}
    if pred:
        dur = np.array([p["duration_ms"] for p in pred])
        en = np.array([p["energy_above_before_mJ"] for p in pred])
        res |= {"prediction_ms_median": float(np.median(dur)),
                "prediction_mJ_median": float(np.median(en)),
                "prediction_share_of_hourly_energy": float(np.median(en)) * 1e-3
                / res["energy_per_hour_J"]}
    print(f"{t_rec / 60:.1f} min after {a.skip:.0f} s start-up: mean {i_mean:.2f} mA at "
          f"{v:.2f} V = {v * i_mean:.1f} mW ({res['energy_per_hour_J']:.0f} J per hour)")
    print(f"without the prediction intervals: {i_wo:.2f} mA; "
          f"1/50/99 % of 0.1-ms current: {res['i_p01_p50_p99_mA']}")
    for p in pred:
        print(f"  prediction at {p['t_s']:7.1f} s: {p['duration_ms']:7.2f} ms, "
              f"{p['i_mean_mA']:.1f} mA (before {p['i_before_mA']:.1f} mA), "
              f"{p['energy_above_before_mJ']:.3f} mJ above the level before")
    out = a.out or a.name + "_system.json"
    json.dump(res, open(out, "w"), indent=1)
    # traces for a figure: 1-ms means of everything, full rate around one prediction
    m = len(c) // (rate // 1000)
    ms = c[:m * (rate // 1000)].reshape(m, -1).mean(1).astype(np.float32)
    ex = {}
    if len(s):
        j = len(s) // 2
        x0, x1 = max(0, s[j] - int(0.5 * rate)), min(len(c), e[j] + int(0.5 * rate))
        ex = {"exc_cur_mA": c[x0:x1].astype(np.float32), "exc_d0": d0[x0:x1],
              "exc_t0_s": np.float64(a.skip + x0 / rate)}
    np.savez_compressed(out.replace(".json", "") + "_trace.npz", ms_cur_mA=ms,
                        ms_t0_s=np.float64(a.skip), rate=np.int64(rate), **ex)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
