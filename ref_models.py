"""NumPy reference implementation of the paper-1 models, for embedded deployment.

Every forward pass here reproduces the PyTorch models in AQM_GRU_Model.py,
AQM_model.py (TCN) and AQM_CNNGRU_Model.py (FUSION) in evaluation mode (dropout
off), from the same ``state_dict`` arrays. It is the bridge between PyTorch and the
C library: PyTorch == NumPy is checked on the PC (export_models.py), NumPy == C is
checked here (tests/test_c_vs_numpy.py), so PyTorch == C follows.

Conventions
    window   float32 array (24, 5): the raw hourly means of one input window, in the
             order of senseurcity_dataset.INPUT_FEATURES
             (SHT31TE, SHT31HE, Absolute_humidity, OPCN3PM25, OPCN3PM10)
    scaler   {"feature_mean": [5], "feature_std": [5], "log_channels": [3, 4]}
    output   one logit; the decision is ``logit > threshold`` (nested_worker)
"""
from __future__ import annotations

import numpy as np

F32 = np.float32


def sigmoid(x):
    x = np.asarray(x, dtype=F32)
    with np.errstate(over="ignore"):
        return (1.0 / (1.0 + np.exp(-x))).astype(F32)


# --------------------------------------------------------------------- scaling
def scale(window, scaler):
    x = np.asarray(window, dtype=F32).copy()
    for c in scaler["log_channels"]:
        x[..., c] = np.log1p(np.clip(x[..., c], 0.0, None))
    mean = np.asarray(scaler["feature_mean"], dtype=F32)
    std = np.asarray(scaler["feature_std"], dtype=F32)
    std = np.where(np.isfinite(std) & (std > 0), std, F32(1.0)).astype(F32)
    return ((x - mean) / std).astype(F32)


# --------------------------------------------------------------------- layers
def linear(x, w, b):
    return (x @ w.T + b).astype(F32)


def gru_layer(seq, w_ih, w_hh, b_ih, b_hh):
    """PyTorch GRU, one layer: seq (..., T, in) -> outputs (..., T, H)."""
    H = w_hh.shape[1]
    T = seq.shape[-2]
    h = np.zeros(seq.shape[:-2] + (H,), dtype=F32)
    out = np.empty(seq.shape[:-2] + (T, H), dtype=F32)
    for t in range(T):
        gi = (seq[..., t, :] @ w_ih.T + b_ih).astype(F32)
        gh = (h @ w_hh.T + b_hh).astype(F32)
        r = sigmoid(gi[..., :H] + gh[..., :H])
        z = sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
        n = np.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:]).astype(F32)
        h = ((1.0 - z) * n + z * h).astype(F32)
        out[..., t, :] = h
    return out


def gru_stack(seq, sd, prefix, layers):
    for l in range(layers):
        seq = gru_layer(seq, sd[f"{prefix}.weight_ih_l{l}"], sd[f"{prefix}.weight_hh_l{l}"],
                        sd[f"{prefix}.bias_ih_l{l}"], sd[f"{prefix}.bias_hh_l{l}"])
    return seq


def causal_conv1d(seq, w, b, dilation=1):
    """seq (..., T, Cin), w (Cout, Cin, K) -> (..., T, Cout); left zero padding
    (K-1)*dilation, i.e. PyTorch Conv1d with symmetric padding then a right chomp."""
    T = seq.shape[-2]
    cout, cin, K = w.shape
    pad = (K - 1) * dilation
    xp = np.concatenate([np.zeros(seq.shape[:-2] + (pad, cin), dtype=F32), seq], axis=-2)
    out = np.broadcast_to(b.astype(F32), seq.shape[:-2] + (T, cout)).copy()
    for k in range(K):
        out += xp[..., k * dilation: k * dilation + T, :] @ w[:, :, k].T
    return out.astype(F32)


def weight_norm(g, v):
    """torch.nn.utils.parametrizations.weight_norm, dim=0: w = g * v / ||v||."""
    norm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True))
    return (g * v / norm).astype(F32)


def tcn_conv_weight(sd, key):
    """Effective weight of a (possibly weight-normalized) conv in a state_dict."""
    g = sd.get(f"{key}.parametrizations.weight.original0")
    if g is not None:
        return weight_norm(g, sd[f"{key}.parametrizations.weight.original1"])
    if f"{key}.weight_g" in sd:              # older torch.nn.utils.weight_norm
        return weight_norm(sd[f"{key}.weight_g"], sd[f"{key}.weight_v"])
    return sd[f"{key}.weight"]


# --------------------------------------------------------------------- models
def gru_forward(x, sd, hp):
    out = gru_stack(x, sd, "gru", hp["num_layers"])
    return linear(out[..., -1, :], sd["head.weight"], sd["head.bias"])[..., 0]


def tcn_forward(x, sd, hp):
    chans = list(hp["tcn_channels"])
    seq = x
    for i, c in enumerate(chans):
        p = f"tcn.network.{i}"
        d = 2 ** i
        h = np.maximum(causal_conv1d(seq, tcn_conv_weight(sd, f"{p}.conv1"), sd[f"{p}.conv1.bias"], d), 0)
        h = np.maximum(causal_conv1d(h, tcn_conv_weight(sd, f"{p}.conv2"), sd[f"{p}.conv2.bias"], d), 0)
        if f"{p}.downsample.weight" in sd:
            res = causal_conv1d(seq, sd[f"{p}.downsample.weight"], sd[f"{p}.downsample.bias"], 1)
        else:
            res = seq
        seq = np.maximum(h + res, 0).astype(F32)
    return linear(seq[..., -1, :], sd["head.weight"], sd["head.bias"])[..., 0]


def fusion_forward(x, sd, hp):
    c = np.maximum(causal_conv1d(x, sd["conv.1.weight"], sd["conv.1.bias"], 1), 0)
    f_cnn = linear(c[..., -1, :], sd["cnn_proj.weight"], sd["cnn_proj.bias"])
    out = gru_stack(x, sd, "gru", hp["num_layers"])
    f_gru = linear(out[..., -1, :], sd["gru_proj.weight"], sd["gru_proj.bias"])
    g = sigmoid(linear(np.concatenate([f_cnn, f_gru], axis=-1), sd["gate.weight"], sd["gate.bias"]))
    fused = (g * f_cnn + (1.0 - g) * f_gru).astype(F32)
    return linear(fused, sd["head.weight"], sd["head.bias"])[..., 0]


FORWARD = {"GRU": gru_forward, "TCN": tcn_forward, "FUSION": fusion_forward}


def predict_logit(arch, window, sd, hp, scaler):
    """One window (24, 5) -> float, or a batch (N, 24, 5) -> array (N,)."""
    y = FORWARD[arch](scale(window, scaler), sd, hp)
    return float(y) if np.ndim(y) == 0 else np.asarray(y, dtype=F32)


# ------------------------------------------------------------ int8 weights
def quantize_per_row(w):
    """Symmetric int8, one scale per output channel (first axis). Returns (q, scale)."""
    flat = w.reshape(w.shape[0], -1).astype(np.float64)
    amax = np.abs(flat).max(axis=1)
    s = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.round(flat / s[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
    return q, s.astype(F32)


def dequantize(q, s):
    return (q.astype(F32) * s.reshape((-1,) + (1,) * (q.ndim - 1))).astype(F32)


def int8_state_dict(sd):
    """State dict with every weight matrix replaced by its int8 round trip; biases,
    weight-norm gains and other vectors stay float. Weight-normalized convolutions are
    folded to their effective weight first, which is what the device stores."""
    out = {}
    for k, v in sd.items():
        if k.endswith("parametrizations.weight.original0"):
            continue
        if k.endswith("parametrizations.weight.original1"):
            base = k[: -len(".parametrizations.weight.original1")]
            w = tcn_conv_weight(sd, base)
            q, s = quantize_per_row(w)
            out[f"{base}.weight"] = dequantize(q, s)
            continue
        if v.ndim >= 2:
            q, s = quantize_per_row(v)
            out[k] = dequantize(q, s)
        else:
            out[k] = np.asarray(v, dtype=F32)
    return out


def folded_state_dict(sd):
    """Float state dict with weight-normalized convolutions folded to plain weights."""
    out = {}
    for k, v in sd.items():
        if k.endswith("parametrizations.weight.original0"):
            continue
        if k.endswith("parametrizations.weight.original1"):
            base = k[: -len(".parametrizations.weight.original1")]
            out[f"{base}.weight"] = tcn_conv_weight(sd, base)
            continue
        out[k] = np.asarray(v, dtype=F32)
    return out


# ------------------------------------------------------- random test models
def random_state_dict(arch, hp, rng, n_in=5):
    """A state dict with PyTorch's names and shapes and random values, for tests."""
    r = lambda *s: (rng.standard_normal(s) * 0.3).astype(F32)
    sd = {}

    def gru(prefix, n_in, H, L):
        for l in range(L):
            i = n_in if l == 0 else H
            sd[f"{prefix}.weight_ih_l{l}"] = r(3 * H, i)
            sd[f"{prefix}.weight_hh_l{l}"] = r(3 * H, H)
            sd[f"{prefix}.bias_ih_l{l}"] = r(3 * H)
            sd[f"{prefix}.bias_hh_l{l}"] = r(3 * H)

    if arch == "GRU":
        H = hp["hidden_size"]
        gru("gru", n_in, H, hp["num_layers"])
        sd["head.weight"], sd["head.bias"] = r(1, H), r(1)
    elif arch == "TCN":
        prev, K = n_in, hp["kernel_size"]
        for i, c in enumerate(hp["tcn_channels"]):
            p = f"tcn.network.{i}"
            for j, cin in (("conv1", prev), ("conv2", c)):
                sd[f"{p}.{j}.parametrizations.weight.original0"] = np.abs(r(c, 1, 1)) + 0.5
                sd[f"{p}.{j}.parametrizations.weight.original1"] = r(c, cin, K)
                sd[f"{p}.{j}.bias"] = r(c)
            if prev != c:
                sd[f"{p}.downsample.weight"], sd[f"{p}.downsample.bias"] = r(c, prev, 1), r(c)
            prev = c
        sd["head.weight"], sd["head.bias"] = r(1, prev), r(1)
    elif arch == "FUSION":
        cc, K, gh, fd = hp["cnn_channels"], hp["kernel_size"], hp["gru_hidden"], hp["fusion_dim"]
        sd["conv.1.weight"], sd["conv.1.bias"] = r(cc, n_in, K), r(cc)
        sd["cnn_proj.weight"], sd["cnn_proj.bias"] = r(fd, cc), r(fd)
        gru("gru", n_in, gh, hp["num_layers"])
        sd["gru_proj.weight"], sd["gru_proj.bias"] = r(fd, gh), r(fd)
        sd["gate.weight"], sd["gate.bias"] = r(fd, 2 * fd), r(fd)
        sd["head.weight"], sd["head.bias"] = r(1, fd), r(1)
    return sd
