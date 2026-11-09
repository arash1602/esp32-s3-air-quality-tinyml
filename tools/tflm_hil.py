"""Board runs of the TensorFlow Lite Micro firmware (firmware_tflm), for the comparison
with aq_infer: decisions, latency and memory of every model in tflm_models.bin.

    python tools/tflm_hil.py --port COM3 --tflm tflm_out -o tflm_board.json

For every model: its arena use; N test windows of its fold streamed through it (default
500, evenly spread), compared with the TensorFlow Lite interpreter on the PC (same file)
and with the NumPy reference (= PyTorch = aq_infer) of the trained network; and its
latency on the window that tools/hil.py times aq_infer on (median of up to 1000
repetitions). Resumes from an existing output file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import ref_models as rm  # noqa: E402
from hil import Board  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--tflm", default="tflm_out")
    ap.add_argument("--image", default="tflm_a", help="image name (tflm_a: TCN+CNN-GRU, tflm_b: GRU)")
    ap.add_argument("--export", default=os.path.join(os.path.dirname(HERE), "export_out"))
    ap.add_argument("--windows", type=int, default=500)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("-o", "--out", default="tflm_board.json")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.tflm, a.image + "_index.json")))
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["index"] for r in res}
    b = Board(a.port)
    print(f"board: {b.n_models} models (index: {len(idx)})", flush=True)
    for e in idx:
        if e["index"] in done:
            continue
        d = os.path.join(a.export, f"h{e['horizon']}", e["design"], e["fold"])
        win = np.fromfile(os.path.join(d, "test_windows.f32"), "<f4").reshape(-1, 24, 5)
        meta = json.load(open(os.path.join(d, f"{e['arch']}.json")))
        thr = float(meta["threshold_logit"])
        sd = rm.folded_state_dict(dict(np.load(os.path.join(d, f"{e['arch']}.npz"))))
        sel = np.unique(np.linspace(0, len(win) - 1, min(a.windows, len(win))).astype(int))
        ref = rm.predict_logit(e["arch"], win[sel], sd, meta["hp"], meta["scaler"]).astype(np.float64)
        pc = np.fromfile(os.path.join(a.tflm, e["key"] + ".f32"), "<f4").astype(np.float64)[sel]
        try:
            info = b.select(e["index"])
        except SystemExit as err:              # e.g. the interpreter's arena is too small
            r = {**e, "error": str(err)}
            res.append(r)
            json.dump(res, open(a.out, "w"), indent=1)
            print(f"[{e['index']:2d}] {e['key']:30s} does not run: {err}", flush=True)
            b.reconnect()
            continue
        s, dec, us = b.predict(win[sel], 1.0)
        b.set_window(win[len(win) // 2] if len(win) < 1000 else
                     np.fromfile(os.path.join(d, "test_windows.f32"), "<f4", count=24 * 5 * 1000)
                     .reshape(-1, 24, 5)[500])
        line = b.bench(a.reps, a.budget)
        m = b.text(b"M").split()
        r = {**e, "arena_bytes": info["n_params"], "windows": int(len(sel)),
             "board_vs_pc_max_abs": float(np.max(np.abs(s - pc))),
             "board_vs_pc_decisions_differ": int(((s > thr) != (pc > thr)).sum()),
             "board_vs_ref_max_abs": float(np.max(np.abs(s - ref))),
             "board_vs_ref_decisions_differ": int((dec != (ref > thr)).sum()),
             "median_us": int(line[3]), "p99_us": int(line[4]), "min_us": int(line[2]),
             "reps": int(line[1]), "free_heap": int(m[1])}
        res.append(r)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"[{e['index']:2d}] {e['key']:30s} arena {r['arena_bytes'] / 1024:6.1f} KB  "
              f"median {r['median_us'] / 1000:8.2f} ms  vs PC {r['board_vs_pc_max_abs']:.1e} "
              f"({r['board_vs_pc_decisions_differ']} dec)  vs reference "
              f"{r['board_vs_ref_decisions_differ']}/{len(sel)} decisions differ", flush=True)
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
