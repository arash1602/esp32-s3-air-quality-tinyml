"""C library == NumPy reference, for every architecture configuration of paper 1.

    python tests/test_c_vs_numpy.py [--c-binary c/aq_native]

For each of the 18 selected configurations (3 architectures x 3 horizons x 2 designs)
and both precisions, builds a model with PyTorch's parameter names and shapes and
random weights (scaled like a trained network, ~1/sqrt(fan-in)), random scaler
statistics and 200 random windows with realistic magnitudes, writes the blob, runs
the C harness and compares logits with ref_models (on the int8 round trip for int8).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import blob  # noqa: E402
import ref_models as rm  # noqa: E402

RESULTS = os.environ.get("AQM_RESULTS",
                         os.path.join(os.path.dirname(HERE), "..", "repo", "code", "tuning_results"))


def windows(rng, n):
    w = np.empty((n, 24, 5), np.float32)
    w[..., 0] = rng.normal(12, 8, (n, 24))            # temperature, C
    w[..., 1] = rng.uniform(30, 95, (n, 24))          # relative humidity, %
    w[..., 2] = rng.uniform(2, 14, (n, 24))           # absolute humidity
    w[..., 3] = rng.lognormal(2.0, 0.9, (n, 24))      # optical PM2.5
    w[..., 4] = w[..., 3] * rng.uniform(1.2, 4.0, (n, 24))
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-binary", default=os.path.join(os.path.dirname(HERE), "c", "aq_native"))
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    worst = {"f32": 0.0, "int8": 0.0}
    agree = {"f32": [0, 0], "int8": [0, 0]}
    n_cfg = 0
    with tempfile.TemporaryDirectory() as tmp:
        for h in (1, 12, 24):
            for design in ("single", "multi"):
                for arch in ("GRU", "TCN", "FUSION"):
                    with open(os.path.join(RESULTS, f"h{h}",
                                           f"best_{arch}_{design}_ctx24_blocked_logz.json")) as f:
                        hp = json.load(f)["hyperparameters"]
                    sd = rm.random_state_dict(arch, hp, rng)
                    for k, v in sd.items():          # trained-network-like magnitudes
                        if v.ndim >= 2 and "original0" not in k:
                            sd[k] = (v / 0.3 / np.sqrt(np.prod(v.shape[1:]))).astype(np.float32)
                    w = windows(rng, 200)
                    t = rm.scale(w.reshape(-1, 5), {"feature_mean": [0] * 5, "feature_std": [1] * 5,
                                                   "log_channels": [3, 4]})
                    scaler = {"feature_mean": t.mean(0).tolist(), "feature_std": t.std(0).tolist(),
                              "log_channels": [3, 4]}
                    wpath = os.path.join(tmp, "w.f32")
                    w.astype("<f4").tofile(wpath)
                    for prec in ("f32", "int8"):
                        ref_sd = rm.int8_state_dict(sd) if prec == "int8" else rm.folded_state_dict(sd)
                        ref = np.array([rm.FORWARD[arch](rm.scale(x, scaler), ref_sd, hp) for x in w])
                        thr = float(np.median(ref))
                        bpath = os.path.join(tmp, "m.aqm")
                        blob.write_blob(bpath, arch, hp, sd, scaler, thr, precision=prec)
                        out = subprocess.run([args.c_binary, bpath, wpath], capture_output=True,
                                             text=True, check=True).stdout
                        c = np.array([float(x) for x in out.split()])
                        err = np.max(np.abs(c - ref) / np.maximum(1.0, np.abs(ref)))
                        worst[prec] = max(worst[prec], err)
                        same = int(((c > thr) == (ref > thr)).sum())
                        agree[prec][0] += same
                        agree[prec][1] += len(ref)
                        status = "ok" if err < 1e-4 else "FAIL"
                        print(f"h={h:2d} {design:6s} {arch:6s} {prec:4s} max rel err {err:.2e} "
                              f"decisions {same}/{len(ref)} {status}")
                    n_cfg += 1
    print(f"\n{n_cfg} configurations x 2 precisions")
    for p in ("f32", "int8"):
        print(f"  {p}: worst logit error {worst[p]:.2e}, decisions identical "
              f"{agree[p][0]}/{agree[p][1]}")
    ok = all(v < 1e-4 for v in worst.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
