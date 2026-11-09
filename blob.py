"""Pack a trained model into the binary blob that the C library (c/aq_infer.c) reads.

    write_blob(path, arch, hp, state_dict, scaler, threshold, precision="f32"|"int8")

Weight-normalized TCN convolutions are folded to plain weights; int8 uses the same
per-row symmetric quantization as ref_models.quantize_per_row, so the NumPy reference
on ref_models.int8_state_dict(...) is the exact float model of the int8 blob.
"""
from __future__ import annotations

import struct

import numpy as np

import ref_models as rm

ARCH_ID = {"GRU": 0, "TCN": 1, "FUSION": 2}


def _tensor(out, arr, int8):
    a = np.asarray(arr, dtype=np.float32)
    rows = a.shape[0]
    cols = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
    flat = a.reshape(rows, cols)
    if int8 and a.ndim >= 2:
        q, s = rm.quantize_per_row(flat)
        out += struct.pack("<III", 1, rows, cols)
        out += s.astype("<f4").tobytes()
        data = q.astype(np.int8).tobytes()
        out += data + b"\0" * ((-len(data)) % 4)
    else:
        out += struct.pack("<III", 0, rows, cols)
        out += flat.astype("<f4").tobytes()


def _gru(out, sd, prefix, layers, int8):
    for l in range(layers):
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            _tensor(out, sd[f"{prefix}.{name}_l{l}"], int8)


def blob_bytes(arch, hp, sd, scaler, threshold, precision="f32"):
    int8 = precision == "int8"
    sd = rm.folded_state_dict(sd)
    n_in = len(scaler["feature_mean"])
    kernel = int(hp.get("kernel_size", 1))
    out = bytearray(b"AQM1")
    out += struct.pack("<IIIIII", 1, ARCH_ID[arch], int(int8), 24, n_in, kernel)
    out += np.asarray(scaler["feature_mean"], "<f4").tobytes()
    out += np.asarray(scaler["feature_std"], "<f4").tobytes()
    mask = 0
    for c in scaler["log_channels"]:
        mask |= 1 << int(c)
    out += struct.pack("<If", mask, float(threshold))
    if arch == "GRU":
        out += struct.pack("<II", hp["num_layers"], hp["hidden_size"])
        _gru(out, sd, "gru", hp["num_layers"], int8)
    elif arch == "TCN":
        chans = list(hp["tcn_channels"])
        out += struct.pack("<I", len(chans))
        for i, c in enumerate(chans):
            out += struct.pack("<II", c, int(f"tcn.network.{i}.downsample.weight" in sd))
        for i in range(len(chans)):
            p = f"tcn.network.{i}"
            for name in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"):
                _tensor(out, sd[f"{p}.{name}"], int8)
            if f"{p}.downsample.weight" in sd:
                _tensor(out, sd[f"{p}.downsample.weight"], int8)
                _tensor(out, sd[f"{p}.downsample.bias"], int8)
    elif arch == "FUSION":
        out += struct.pack("<IIII", hp["cnn_channels"], hp["gru_hidden"], hp["num_layers"],
                           hp["fusion_dim"])
        _tensor(out, sd["conv.1.weight"], int8)
        _tensor(out, sd["conv.1.bias"], int8)
        _tensor(out, sd["cnn_proj.weight"], int8)
        _tensor(out, sd["cnn_proj.bias"], int8)
        _gru(out, sd, "gru", hp["num_layers"], int8)
        for name in ("gru_proj.weight", "gru_proj.bias", "gate.weight", "gate.bias"):
            _tensor(out, sd[name], int8)
    else:
        raise ValueError(arch)
    _tensor(out, sd["head.weight"], int8)
    _tensor(out, sd["head.bias"], int8)
    return bytes(out)


def write_blob(path, *args, **kw):
    data = blob_bytes(*args, **kw)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


# --------------------------------------------------------------------------- #
# Classical predictors of paper 1 (computed in double on the device)
# --------------------------------------------------------------------------- #
CLASSIC_ID = {"WHO": 3, "LR": 4, "GBT": 5}
N_STATS = 6


def _classic_header(kind, n_in, ge, thr):
    out = bytearray(b"AQM1")
    out += struct.pack("<IIIIII", 1, CLASSIC_ID[kind], 0, 24, n_in, 1)
    out += np.zeros(n_in, "<f4").tobytes() + np.ones(n_in, "<f4").tobytes()
    out += struct.pack("<If", 0, float(thr))
    out += struct.pack("<Id", int(ge), float(thr))
    return out


def who_blob_bytes(i25, i10, lim25=15.0, lim10=45.0, n_in=5):
    """The zero-parameter WHO rule: mean PM2.5 > 15 or mean PM10 > 45 (ug/m3)."""
    out = _classic_header("WHO", n_in, 0, 1.0)
    out += struct.pack("<IIdd", i25, i10, lim25, lim10)
    return bytes(out)


def lr_blob_bytes(idx, mean, scale, coef, intercept, thr_score, n_in=5):
    """Logistic regression on window means. ``thr_score`` is on the decision-function
    scale; the device decides ``score >= thr_score``."""
    n = len(idx)
    out = _classic_header("LR", n_in, 1, thr_score)
    out += struct.pack(f"<I{n}I", n, *[int(i) for i in idx])
    out += np.concatenate([np.asarray(mean, "<f8"), np.asarray(scale, "<f8"),
                           np.asarray(coef, "<f8").reshape(-1),
                           np.asarray([intercept], "<f8").reshape(-1)]).astype("<f8").tobytes()
    return bytes(out)


def gbt_blob_bytes(clf, thr_score, n_in=5):
    """scikit-learn HistGradientBoostingClassifier (binary, numeric features) on the
    30 window statistics; the device decides ``raw score >= thr_score``."""
    preds = [p[0] for p in clf._predictors]
    baseline = float(np.asarray(clf._baseline_prediction).reshape(-1)[0])
    out = _classic_header("GBT", n_in, 1, thr_score)
    out += struct.pack("<Id", len(preds), baseline)
    for p in preds:
        nodes = p.nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported")
        if len(nodes) >= 65535:
            raise ValueError("tree too large")
        out += struct.pack("<I", len(nodes))
        for nd in nodes:
            leaf = bool(nd["is_leaf"])
            out += struct.pack("<dHHBBH",
                               float(nd["value"] if leaf else nd["num_threshold"]),
                               0 if leaf else int(nd["left"]), 0 if leaf else int(nd["right"]),
                               255 if leaf else int(nd["feature_idx"]),
                               int(nd["missing_go_to_left"]), 0)
    return bytes(out)


def ge_threshold_on_raw(prob_thr):
    """Smallest float64 r with expit(r) >= prob_thr, so that on any raw score s,
    ``s >= r`` is exactly scikit-learn's ``expit(s) >= prob_thr``."""
    from scipy.special import expit
    p = float(prob_thr)
    if not np.isfinite(p):
        return float("inf")
    if p <= 0.0:
        return float("-inf")
    if expit(np.float64(np.finfo(np.float64).max)) < p:
        return float("inf")

    def key(x):                     # monotone map of float64 to int64
        i = np.float64(x).view(np.int64)
        return int(i) if i >= 0 else -(int(i) & 0x7FFFFFFFFFFFFFFF)

    def val(k):
        i = k if k >= 0 else -k | (-0x8000000000000000)
        return float(np.int64(i).view(np.float64))

    lo, hi = key(-800.0), key(800.0)    # expit(-800) = 0 < p ; expit(800) = 1 >= p
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if expit(np.float64(val(mid))) >= p:
            hi = mid
        else:
            lo = mid
    return val(hi)
