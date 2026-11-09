"""C library vs NumPy reference on EVERY test window of every exported network.

    python tests/verify_export.py [--export export_out]

For each model in export_out/summary.json, both device files (_f32.aqm, _int8.aqm)
are run through the C library (c/aq_native) on the fold's full test set and compared
with the NumPy reference of the same precision (float32 weights, or the int8 round
trip), and with PyTorch's logits. Writes export_out/verify_c.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ref_models as rm  # noqa: E402

NATIVE = os.path.join(ROOT, "c", "aq_native")


def ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--export", default=os.path.join(ROOT, "export_out"))
    args = a.parse_args()
    subprocess.run(["gcc", "-O2", "-std=c99", "-o", NATIVE, os.path.join(ROOT, "c", "aq_native.c"),
                    os.path.join(ROOT, "c", "aq_infer.c"), "-lm"], check=True)
    summary = json.load(open(os.path.join(args.export, "summary.json")))
    out = []
    for key, m in sorted(summary.items()):
        d = os.path.join(args.export, f"h{m['horizon']}", m["design"], m["outer"])
        wpath = os.path.join(d, "test_windows.f32")
        win = np.fromfile(wpath, "<f4").reshape(-1, 24, 5)
        y = np.fromfile(os.path.join(d, "test_y.u8"), np.uint8)
        sd = dict(np.load(os.path.join(d, f"{m['arch']}.npz")))
        torch = np.fromfile(os.path.join(d, f"{m['arch']}_torch.f32"), "<f4")
        thr = m["threshold_logit"]
        for prec in ("f32", "int8"):
            ref_sd = rm.int8_state_dict(sd) if prec == "int8" else rm.folded_state_dict(sd)
            ref = np.asarray(rm.FORWARD[m["arch"]](rm.scale(win, m["scaler"]), ref_sd, m["hp"]),
                             np.float32)
            c = np.array(subprocess.run([NATIVE, os.path.join(d, f"{m['arch']}_{prec}.aqm"), wpath],
                                        capture_output=True, text=True, check=True).stdout.split(),
                         np.float32)
            r = {"key": key, "arch": m["arch"], "horizon": m["horizon"], "design": m["design"],
                 "outer": m["outer"], "precision": prec, "n": int(len(y)),
                 "c_vs_numpy_max_abs": float(np.max(np.abs(c - ref))),
                 "c_vs_numpy_decisions_equal": int(((c > thr) == (ref > thr)).sum()),
                 "c_vs_torch_decisions_equal": int(((c > thr) == (torch > thr)).sum()),
                 "ap_c": ap(y, c), "ap_torch": ap(y, torch), "ap_paper": m["paper_ap"]}
            out.append(r)
        print(f"{key:40s} C-vs-NumPy f32 {out[-2]['c_vs_numpy_max_abs']:.1e} "
              f"int8 {out[-1]['c_vs_numpy_max_abs']:.1e}  decisions "
              f"{out[-2]['c_vs_numpy_decisions_equal']}/{out[-1]['c_vs_numpy_decisions_equal']}"
              f"/{len(y)}", flush=True)
    with open(os.path.join(args.export, "verify_c.json"), "w") as f:
        json.dump(out, f, indent=1)
    for prec in ("f32", "int8"):
        rs = [r for r in out if r["precision"] == prec]
        n = sum(r["n"] for r in rs)
        print(f"{prec}: {len(rs)} models, {n} windows; C vs NumPy max "
              f"{max(r['c_vs_numpy_max_abs'] for r in rs):.2e}, decisions equal "
              f"{sum(r['c_vs_numpy_decisions_equal'] for r in rs)}/{n}; C vs PyTorch decisions "
              f"{sum(r['c_vs_torch_decisions_equal'] for r in rs)}/{n}; largest |AP(C) - AP(torch)| "
              f"{max(abs(r['ap_c'] - r['ap_torch']) for r in rs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
