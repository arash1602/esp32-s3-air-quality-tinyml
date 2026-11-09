"""TensorFlow Lite (Micro) versions of the fidelity networks, for the comparison with aq_infer.

    python tools/tflm_convert.py -o tflm_out            (needs: pip install tensorflow scikit-learn)

For every network of the fidelity image (GRU, TCN, CNN-GRU; 1, 12 and 24 h; within a city
on Antwerp-b1 and across cities on heldout-Zagreb), the trained PyTorch weights in
export_out are rebuilt as a TensorFlow graph of builtin operators only (no custom ops,
no while loops: the 24 recurrent steps are unrolled), so TensorFlow Lite Micro can run it.
The input is the scaled window (scaling, as in aq_infer, is done in C on the device).

Two TensorFlow Lite files per network:
  f32   float32 weights and activations
  int8  full-integer quantization (weights per channel, activations per tensor), the
        mode TensorFlow Lite Micro is optimized for (ESP-NN kernels on the ESP32-S3).
        Activation ranges are calibrated on the test windows of the OTHER folds of the
        same horizon and design, so the evaluated fold is never used for calibration.

Every file is checked on the PC with the TensorFlow Lite interpreter on all test windows
of its fold, against the NumPy reference (ref_models.py, = PyTorch = aq_infer):
largest logit difference, decisions changed, change in AP and F2.

Output (-o): <h>_<design>_<arch>_<prec>.tflite, the PC interpreter logits (.f32), stats.json,
and the board image tflm_models.bin with its index tflm_index.json (tools/tflm_hil.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import ref_models as rm  # noqa: E402

ARCH_ID = {"GRU": 0, "TCN": 1, "FUSION": 2}
DESIGNS = {"single-blocked": "Antwerp-b1", "multi": "heldout-Zagreb"}


# ------------------------------------------------------------------ graph builders
def build(arch, sd, hp):
    """tf.function: scaled window (1, 24, 5) float32 -> logit (1, 1)."""
    import tensorflow as tf
    C = lambda a: tf.constant(np.asarray(a, np.float32))

    def dense(x, w, b):                         # x (1, in), w (out, in)
        return tf.matmul(x, C(w.T)) + C(b)

    def gru_stack(seq, prefix, layers):         # seq: list of (1, in), one per step
        for l in range(layers):
            wih, whh = sd[f"{prefix}.weight_ih_l{l}"], sd[f"{prefix}.weight_hh_l{l}"]
            bih, bhh = sd[f"{prefix}.bias_ih_l{l}"], sd[f"{prefix}.bias_hh_l{l}"]
            H = whh.shape[1]
            Wi, Wh, Bi, Bh = C(wih.T), C(whh.T), C(bih), C(bhh)
            h = tf.zeros((1, H), tf.float32)
            out = []
            for x in seq:
                gi = tf.matmul(x, Wi) + Bi
                gh = tf.matmul(h, Wh) + Bh
                r = tf.sigmoid(gi[:, :H] + gh[:, :H])
                z = tf.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
                n = tf.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
                h = (1.0 - z) * n + z * h
                out.append(h)
            seq = out
        return seq

    def conv(x, w, b, d):                       # x (1, 1, T, Cin) NHWC; w (Cout, Cin, K)
        K = w.shape[2]
        pad = (K - 1) * d
        if pad:
            x = tf.pad(x, [[0, 0], [0, 0], [pad, 0], [0, 0]])
        f = C(np.transpose(w, (2, 1, 0))[None])  # (1, K, Cin, Cout)
        return tf.nn.conv2d(x, f, strides=1, padding="VALID", dilations=[1, 1, d, 1]) + C(b)

    if arch == "GRU":
        def f(x):
            seq = [x[:, t, :] for t in range(24)]
            h = gru_stack(seq, "gru", hp["num_layers"])[-1]
            return dense(h, sd["head.weight"], sd["head.bias"])
    elif arch == "TCN":
        def f(x):
            s = tf.reshape(x, (1, 1, 24, 5))
            for i, _ in enumerate(hp["tcn_channels"]):
                p, d = f"tcn.network.{i}", 2 ** i
                h = tf.nn.relu(conv(s, sd[f"{p}.conv1.weight"], sd[f"{p}.conv1.bias"], d))
                h = tf.nn.relu(conv(h, sd[f"{p}.conv2.weight"], sd[f"{p}.conv2.bias"], d))
                res = (conv(s, sd[f"{p}.downsample.weight"], sd[f"{p}.downsample.bias"], 1)
                       if f"{p}.downsample.weight" in sd else s)
                s = tf.nn.relu(h + res)
            last = tf.reshape(s[:, :, 23, :], (1, -1))
            return dense(last, sd["head.weight"], sd["head.bias"])
    else:
        def f(x):
            s = tf.reshape(x, (1, 1, 24, 5))
            c = tf.nn.relu(conv(s, sd["conv.1.weight"], sd["conv.1.bias"], 1))
            f_cnn = dense(tf.reshape(c[:, :, 23, :], (1, -1)), sd["cnn_proj.weight"], sd["cnn_proj.bias"])
            seq = [x[:, t, :] for t in range(24)]
            h = gru_stack(seq, "gru", hp["num_layers"])[-1]
            f_gru = dense(h, sd["gru_proj.weight"], sd["gru_proj.bias"])
            g = tf.sigmoid(dense(tf.concat([f_cnn, f_gru], axis=1), sd["gate.weight"], sd["gate.bias"]))
            fused = g * f_cnn + (1.0 - g) * f_gru
            return dense(fused, sd["head.weight"], sd["head.bias"])
    return tf.function(f, input_signature=[tf.TensorSpec((1, 24, 5), tf.float32)])


def convert(fn, int8=False, rep=None):
    import tensorflow as tf
    cf = fn.get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf], fn)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([r[None]] for r in rep)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(model, xs):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=model)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = np.empty(len(xs), np.float64)
    for k, x in enumerate(xs):
        if i["dtype"] == np.int8:
            s, z = i["quantization"]
            x = np.clip(np.round(x / s) + z, -128, 127).astype(np.int8)
        it.set_tensor(i["index"], x[None].astype(i["dtype"]))
        it.invoke()
        y = it.get_tensor(o["index"]).reshape(-1)[0]
        if o["dtype"] == np.int8:
            s, z = o["quantization"]
            y = (float(y) - z) * s
        out[k] = float(y)
    io = {"in_scale": float(i["quantization"][0]), "in_zp": int(i["quantization"][1]),
          "out_scale": float(o["quantization"][0]), "out_zp": int(o["quantization"][1])}
    return out, io


def op_names(model):
    ops = set()
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        it = Interpreter(model_content=model)
        for d in it._get_ops_details():
            ops.add(d["op_name"])
    except Exception:
        pass
    return sorted(ops)


def ap_f2(y, s, thr):
    from sklearn.metrics import average_precision_score
    t = y > 0
    d = s > thr
    tp, fp, fn = int((d & t).sum()), int((d & ~t).sum()), int((~d & t).sum())
    den = 5 * tp + 4 * fn + fp
    return float(average_precision_score(t, s)), (5 * tp / den if den else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=os.path.join(ROOT, "export_out"))
    ap.add_argument("-o", "--out", default="tflm_out")
    ap.add_argument("--archs", nargs="+", default=["TCN", "FUSION", "GRU"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    ap.add_argument("--calib", type=int, default=500, help="calibration windows (int8)")
    ap.add_argument("--image-archs", nargs="+", default=None,
                    help="networks in the board image (default: --archs); the 7 MB models "
                         "partition holds TCN+FUSION or GRU, not all three")
    ap.add_argument("--image-name", default="tflm_models")
    a = ap.parse_args()
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    os.makedirs(a.out, exist_ok=True)
    stats_path = os.path.join(a.out, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    rng = np.random.default_rng(0)
    for h in a.horizons:
        for design, fold in DESIGNS.items():
            d = os.path.join(a.export, f"h{h}", design, fold)
            win = np.fromfile(os.path.join(d, "test_windows.f32"), "<f4").reshape(-1, 24, 5)
            y = np.fromfile(os.path.join(d, "test_y.u8"), np.uint8)
            others = [p for p in glob.glob(os.path.join(a.export, f"h{h}", design, "*"))
                      if os.path.isdir(p) and os.path.basename(p) != fold]
            for arch in a.archs:
                name = f"h{h}_{design}_{arch}"
                if f"{name}_int8" in stats:
                    continue
                meta = json.load(open(os.path.join(d, f"{arch}.json")))
                hp, scaler, thr = meta["hp"], meta["scaler"], float(meta["threshold_logit"])
                sd = rm.folded_state_dict(dict(np.load(os.path.join(d, f"{arch}.npz"))))
                xs = rm.scale(win, scaler)
                ref = rm.predict_logit(arch, win, sd, hp, scaler).astype(np.float64)
                cal = np.concatenate([rm.scale(np.fromfile(os.path.join(p, "test_windows.f32"), "<f4")
                                               .reshape(-1, 24, 5), scaler) for p in others])
                cal = cal[rng.choice(len(cal), min(a.calib, len(cal)), replace=False)]
                fn = build(arch, sd, hp)
                ap_ref, f2_ref = ap_f2(y, ref, thr)
                for prec in ("f32", "int8"):
                    t0 = time.time()
                    try:
                        model = convert(fn, prec == "int8", cal)
                    except Exception as e:                    # report, do not stop
                        print(f"{name} {prec}: conversion failed: {str(e)[:300]}", flush=True)
                        stats[f"{name}_{prec}"] = {"error": str(e)[:2000]}
                        continue
                    base = os.path.join(a.out, f"{name}_{prec}")
                    open(base + ".tflite", "wb").write(model)
                    out, io = run_tflite(model, xs)
                    out.astype("<f4").tofile(base + ".f32")
                    apv, f2v = ap_f2(y, out, thr)
                    r = {"arch": arch, "horizon": h, "design": design, "fold": fold,
                         "precision": prec, "bytes": len(model), "threshold": thr,
                         "scaler": scaler, "n_test": int(len(win)), **io,
                         "ops": op_names(model),
                         "max_abs_diff": float(np.max(np.abs(out - ref))),
                         "decisions_changed": int(((out > thr) != (ref > thr)).sum()),
                         "ap_ref": ap_ref, "ap": apv, "f2_ref": f2_ref, "f2": f2v,
                         "secs": time.time() - t0}
                    stats[f"{name}_{prec}"] = r
                    print(f"{name:28s} {prec:4s} {len(model) / 1024:8.1f} KB  max|diff| "
                          f"{r['max_abs_diff']:.2e}  decisions changed {r['decisions_changed']:4d}"
                          f"/{len(win)}  dAP {apv - ap_ref:+.4f}  dF2 {f2v - f2_ref:+.4f}", flush=True)
                    json.dump(stats, open(stats_path, "w"), indent=1)
    # board image: AQIX table; each blob = 64-byte header + .tflite, 16-byte aligned
    entries, blobs = [], []
    order = ["GRU", "TCN", "FUSION"]
    for key in sorted(stats, key=lambda k: (stats[k].get("horizon", 0),
                                            stats[k].get("design", ""),
                                            order.index(stats[k]["arch"]) if "arch" in stats[k] else 9,
                                            stats[k].get("precision", ""))):
        r = stats[key]
        if "error" in r or r["arch"] not in (a.image_archs or a.archs):
            continue
        model = open(os.path.join(a.out, key + ".tflite"), "rb").read()
        mask = sum(1 << c for c in r["scaler"]["log_channels"])
        hdr = struct.pack("<4sIIfI5f5f", b"TFLM", ARCH_ID[r["arch"]], 1 if r["precision"] == "int8" else 0,
                          r["threshold"], mask, *r["scaler"]["feature_mean"],
                          *[s if np.isfinite(s) and s > 0 else 1.0 for s in r["scaler"]["feature_std"]])
        hdr += b"\0" * (64 - len(hdr))
        entries.append({"index": len(entries), "kind": "tflm", "horizon": r["horizon"],
                        "design": r["design"], "arch": r["arch"],
                        "precision": "tfl_" + r["precision"], "fold": r["fold"],
                        "bytes": len(model), "key": key})
        blobs.append(hdr + model)
    off = 8 + 8 * len(blobs)
    off = (off + 15) & ~15
    table, data = b"", b""
    for b in blobs:
        pad = (-(off + len(data))) & 15
        data += b"\0" * pad
        table += struct.pack("<II", off + len(data), len(b))
        data += b
    img = b"AQIX" + struct.pack("<I", len(blobs)) + table
    img += b"\0" * (off - len(img)) + data
    if len(img) > 0x6F0000:
        print(f"WARNING: image {len(img)} bytes is larger than the 0x6F0000-byte partition")
    open(os.path.join(a.out, a.image_name + ".bin"), "wb").write(img)
    json.dump(entries, open(os.path.join(a.out, a.image_name + "_index.json"), "w"), indent=1)
    ops = sorted({o for r in stats.values() for o in r.get("ops", [])})
    print(f"image: {len(blobs)} models, {len(img) / 1024:.0f} KB; operators used: {ops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
