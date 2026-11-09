"""End-to-end test of the board workflow without the board.

Builds a models image exactly as for the Sensa board (tools/build_models_image.py),
runs the board-side benchmark loop (bench/aq_bench.c) on this PC behind a
pseudo-terminal (tests/bench_host.c), and drives it with tools/hil.py: memory,
fidelity on every test window, latency, and a 1-second energy run.

The classical models are the real ones (export_classical/). The network models are
stand-ins with PyTorch's names and shapes and random weights, written in the same
layout as export_models.py writes, so the network path is exercised end to end too.

    python tests/test_hil_host.py --code-dir ../code
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blob  # noqa: E402
import ref_models as rm  # noqa: E402

FOLDS = {"single-blocked": ("single", "Antwerp-b1"), "multi": ("multi", "heldout-Zagreb")}


def fake_export(code_dir, classical, out, h, rng):
    for design, (short, fold) in FOLDS.items():
        src = os.path.join(classical, f"h{h}", design, fold)
        d = os.path.join(out, f"h{h}", design, fold)
        os.makedirs(d, exist_ok=True)
        for f in ("test_windows.f32", "test_y.u8"):
            shutil.copy(os.path.join(src, f), d)
        win = np.fromfile(os.path.join(d, "test_windows.f32"), "<f4").reshape(-1, 24, 5)
        x = win.astype(np.float64)
        x[..., 3:] = np.log1p(np.clip(x[..., 3:], 0, None))
        scaler = {"feature_mean": x.reshape(-1, 5).mean(0).tolist(),
                  "feature_std": x.reshape(-1, 5).std(0).tolist(), "log_channels": [3, 4]}
        for arch in ("GRU", "TCN", "FUSION"):
            with open(os.path.join(code_dir, "tuning_results", f"h{h}",
                                   f"best_{arch}_{short}_ctx24_blocked_logz.json")) as f:
                hp = json.load(f)["hyperparameters"]
            sd = rm.random_state_dict(arch, hp, rng)
            np.savez(os.path.join(d, f"{arch}.npz"), **sd)
            with open(os.path.join(d, f"{arch}.json"), "w") as f:
                json.dump({"arch": arch, "hp": hp, "scaler": scaler, "threshold_logit": 0.0}, f)
            for prec in ("f32", "int8"):
                blob.write_blob(os.path.join(d, f"{arch}_{prec}.aqm"), arch, hp, sd, scaler, 0.0,
                                precision=prec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--classical", default=os.path.join(ROOT, "export_classical"))
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="aq_hil_")
    exp = os.path.join(tmp, "export_out")
    fake_export(args.code_dir, args.classical, exp, args.horizon, np.random.default_rng(1))
    py = sys.executable
    image = os.path.join(tmp, "models.bin")
    subprocess.run([py, os.path.join(ROOT, "tools", "build_models_image.py"), "--export", exp,
                    "--classical", args.classical, "--code-dir", args.code_dir, "-o", image,
                    "--horizons", str(args.horizon)], check=True)
    exe = os.path.join(tmp, "bench_host")
    subprocess.run(["gcc", "-O2", "-std=c11", "-I", os.path.join(ROOT, "c"), "-I",
                    os.path.join(ROOT, "bench"), "-o", exe, os.path.join(ROOT, "tests", "bench_host.c"),
                    os.path.join(ROOT, "bench", "aq_bench.c"), os.path.join(ROOT, "c", "aq_infer.c"),
                    "-lm"], check=True)
    board = subprocess.Popen([exe, image], stdout=subprocess.PIPE, text=True)
    port = board.stdout.readline().strip()
    res = os.path.join(tmp, "hil_results")
    idx = os.path.join(tmp, "models_index.json")
    hil = [py, os.path.join(ROOT, "tools", "hil.py"), "--port", port, "--index", idx,
           "--export", exp, "--classical", args.classical, "--out", res]
    env = dict(os.environ)
    try:
        import serial  # noqa: F401
    except ImportError:          # no pyserial here: use the minimal POSIX stand-in
        env["PYTHONPATH"] = os.path.join(ROOT, "tests", "serial_shim") + os.pathsep + env.get(
            "PYTHONPATH", "")
        print("pyserial not installed: using tests/serial_shim")
    ok = True
    try:
        for cmd in (["memory"], ["fidelity"], ["latency", "--reps", "200"],
                    ["energy", "--seconds", "1"]):
            subprocess.run(hil + cmd, check=True, env=env)
        fid = json.load(open(os.path.join(res, "fidelity.json")))
        for r in fid:
            # trees and the WHO rule: bit-exact; logistic regression: summation order of
            # a 5-term dot product; networks: float32 rounding, which for these random
            # stand-ins (logits up to about +-8) is ~5e-5 on BOTH sides when each is
            # compared with a float64 computation, so only a larger gap means a bug
            tol = {"WHO": 0.0, "GBT": 0.0, "LR": 1e-13}.get(r["arch"], 2e-4)
            good = r["max_rel_diff"] <= tol and r["decisions_equal"] == r["n"]
            ok &= good
            if not good:
                print("MISMATCH", r)
        n_cls = sum(r["precision"] == "f64" for r in fid)
        print(f"\n{len(fid)} models checked ({n_cls} classical), "
              f"{sum(r['n'] for r in fid)} windows: {'PASS' if ok else 'FAIL'}")
    finally:
        board.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
