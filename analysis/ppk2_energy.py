"""Energy per prediction from a PPK2 record of the stand-alone energy run.

    python analysis/ppk2_energy.py ppk_timing --index models_timing_index.json \
        -o ppk2_results_timing.json

Reads <name>.cur (float32 uA), <name>.dig (uint8, bit 0 = D0) and <name>.json from
tools/ppk2_log.py. The D0 marker sequence is described in bench/aq_autorun.c.

Per model:
  P_idle     mean power over the idle reference (CPU in the scheduler), leaving out
             the first 1 s and the last 0.3 s of it;
  per prediction (one D0-high interval) the energy above idle,
             E = V * sum(I - I_idle) * dt over the interval;
  reported: number of predictions, median and mean E with its standard error,
  median D0-high time (the prediction plus two register writes), and, as a
  cross-check, the busy-period method of the FNB58 analysis: (P_busy - P_idle) times
  the busy time over the prediction count.
A short full-rate excerpt around the first predictions of every model is saved to
<out>.npz for a current-trace figure.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def deglitch(s, e, g, g_high):
    """Drop high runs of at most g_high samples, then join highs separated by at most
    g low samples: short glitches on the logic input, seen next to marker edges in
    long records. Every real marked interval lasts at least 1 ms (batching in
    bench/aq_autorun.c), and real gaps at least 50 us."""
    keep = (e - s) > g_high
    s, e = s[keep], e[keep]
    if len(s) < 2:
        return s, e
    join = (s[1:] - e[:-1]) <= g
    return s[np.r_[True, ~join]], e[np.r_[~join, True]]


def runs(d0):
    """Start and end (exclusive) sample of every high run of d0."""
    e = np.diff(np.concatenate(([0], d0.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--index", required=True)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--gap", type=float, default=0.2, help="phase gap, s")
    ap.add_argument("--glitch", type=int, default=4,
                    help="low glitches up to this many samples are filled (default 4)")
    ap.add_argument("--glitch-high", type=int, default=10,
                    help="high glitches up to this many samples are dropped (default 10)")
    a = ap.parse_args()
    meta = json.load(open(a.name + ".json"))
    rate, v = meta["rate"], meta["mv"] / 1000.0
    dt = 1.0 / rate
    cur = np.memmap(a.name + ".cur", np.float32, "r")
    dig = np.memmap(a.name + ".dig", np.uint8, "r")
    n = min(len(cur), len(dig))
    cur, d0 = cur[:n], (np.asarray(dig[:n]) & 1).astype(bool)
    idx = json.load(open(a.index))

    s, e = deglitch(*runs(d0), a.glitch, a.glitch_high)
    if not len(s):
        raise SystemExit("no D0 activity in the record: check the D0 and logic GND wires")
    # group high runs into phases separated by low gaps > --gap
    brk = np.flatnonzero(s[1:] - e[:-1] > a.gap * rate) + 1
    groups = np.split(np.arange(len(s)), brk)

    def kind(g):
        w = e[g] - s[g]
        span = (e[g[-1]] - s[g[0]]) / rate
        mw = np.median(w) / rate
        if len(g) == 1 and w[0] >= 2.5 * rate and g[-1] == len(s) - 1:
            return "end"
        if span < 2.5:                           # marker pulse trains are short
            for name, width in (("header", 0.002), ("batch", 0.003), ("id", 0.005)):
                if abs(mw - width) <= 0.0004:
                    return name
        return "busy"

    def pulses_in(g, width):
        """Pulse count of a marker train (period 2 x width) from its time span, so a
        glitch that splits or adds a pulse does not change the count."""
        return int(round(((e[g[-1]] - s[g[0]]) / rate + width) / (2 * width)))

    kinds = [kind(g) for g in groups]
    if kinds[0] != "header":
        print("warning: the record does not start with the header (started late?)")
    else:
        if pulses_in(groups[0], 0.002) != len(idx):
            print(f"warning: board reports {pulses_in(groups[0], 0.002)} models, index has {len(idx)}: "
                  "is the right model image on the board?")
    results, excerpts = [], {}
    j = 0
    while j < len(groups):
        if kinds[j] != "id":
            j += 1
            continue
        gi = groups[j]
        mi = pulses_in(gi, 0.005) - 1
        id_end = e[gi[-1]]
        kb = 1                                   # predictions per marked interval
        if j + 1 < len(groups) and kinds[j + 1] == "batch":
            kb = pulses_in(groups[j + 1], 0.003)
            id_end = e[groups[j + 1][-1]]
            j += 1
        if j + 1 < len(groups) and kinds[j + 1] == "busy":
            gb = groups[j + 1]
            bs, be = s[gb], e[gb]
            i0, i1 = id_end + int(1.0 * rate), bs[0] - int(0.3 * rate)
            i_idle = float(np.mean(cur[i0:i1], dtype=np.float64))
            # Energy per prediction = mean power above idle while D0 is high, times the
            # median D0-high time. Robust to logic-input glitches that split or merge
            # a few intervals; the per-interval energies are kept for the spread.
            c = np.concatenate(([0.0], np.cumsum(cur[bs[0]:be[-1]], dtype=np.float64)))
            q = c[be - bs[0]] - c[bs - bs[0]]                       # uA * samples
            t_high = float(np.sum(be - bs))
            p_excess = v * (q.sum() / t_high - i_idle) * 1e-6       # W above idle, D0 high
            t_one = float(np.median(be - bs)) * dt / kb             # s per prediction
            n_pred = int(round(t_high * dt / (t_one * kb))) * kb
            ep = v * (q - i_idle * (be - bs)) * dt * 1e-6 / kb      # J, per interval
            ok_w = np.abs((be - bs) - np.median(be - bs)) <= 0.05 * np.median(be - bs) + 2
            busy = float(np.mean(cur[bs[0]:be[-1]], dtype=np.float64))
            t_busy = (be[-1] - bs[0]) * dt
            e_busy = v * (busy - i_idle) * 1e-6 * t_busy / max(n_pred, 1)
            ent = idx[mi] if mi < len(idx) else {"index": mi}
            r = {k: ent[k] for k in ("index", "kind", "horizon", "design", "arch",
                                      "precision", "fold", "bytes") if k in ent}
            r |= {"n_pred": n_pred, "batch": int(kb), "idle_mA": i_idle / 1000,
                  "busy_mA": busy / 1000, "p_idle_mW": v * i_idle / 1000,
                  "dP_high_mW": p_excess * 1000,
                  "high_median_us": t_one * 1e6,
                  "energy_mJ": p_excess * t_one * 1e3,
                  "energy_median_mJ": float(np.median(ep[ok_w]) * 1e3),
                  "energy_se_mJ": float(np.std(ep[ok_w], ddof=1) / np.sqrt(ok_w.sum()) * 1e3)
                  if ok_w.sum() > 1 else float("nan"),
                  "n_clean_intervals": int(ok_w.sum()), "n_intervals": int(len(bs)),
                  "energy_busy_method_mJ": float(e_busy * 1e3)}
            results.append(r)
            k0, k1 = bs[0] - int(0.002 * rate), be[min(2, len(be) - 1)] + int(0.002 * rate)
            step = max(1, (k1 - k0) // 20000)
            excerpts[f"m{mi}_cur"] = np.asarray(cur[k0:k1:step], np.float32)
            excerpts[f"m{mi}_d0"] = d0[k0:k1:step]
            excerpts[f"m{mi}_dt"] = np.float64(step * dt)
            print(f"[{mi:2d}] {r.get('arch', '?'):6s} {r.get('precision', ''):4s} "
                  f"{r['n_pred']:6d} pred  idle {r['p_idle_mW']:6.1f} mW  "
                  f"high {r['high_median_us'] / 1000:9.3f} ms  "
                  f"E {r['energy_mJ']:9.4f} mJ (median {r['energy_median_mJ']:9.4f}, busy method {r['energy_busy_method_mJ']:9.4f})")
            j += 2
        else:
            print(f"[{mi:2d}] no busy period (model could not be loaded)")
            j += 1
    out = a.out or a.name + "_results.json"
    json.dump({"meta": meta, "models": results}, open(out, "w"), indent=1)
    np.savez_compressed(out.replace(".json", "") + "_excerpts.npz", **excerpts)
    print(f"saved {out} ({len(results)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
