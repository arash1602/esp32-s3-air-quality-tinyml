"""Every number and table of paper 2 that does not need the board.

    python analysis/paper2_numbers.py      -> analysis/out/numbers.json, tables/*.tex, figures

Inputs: export_out/ (networks, from export_models.py; verify_c.json from
tests/verify_export.py) and export_classical/ (from export_classical.py).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ref_models as rm  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "out")
NETS = ("GRU", "TCN", "FUSION")
CLS = ("WHO", "LR", "GBT")
ALL = CLS + NETS
DESIGNS = ("single-blocked", "multi")
H = (1, 12, 24)
T, C = 24, 5


def f2(y, d):
    t = y > 0
    tp, fp, fn = int((d & t).sum()), int((d & ~t).sum()), int((~d & t).sum())
    den = 5 * tp + 4 * fn + fp
    return 5 * tp / den if den else 0.0


def net_cost(arch, sd, hp):
    """(parameters, multiply-accumulates per prediction, peak activation floats)."""
    f = rm.folded_state_dict(sd)
    # PyTorch's TCN block registers each convolution twice (convN and net.K, the same
    # module inside nn.Sequential), so the state dict holds every TCN weight twice;
    # count each tensor once, as stored in the model file
    params = int(sum(np.asarray(v).size for k, v in f.items() if ".net." not in k))
    macs, peak = 0, 0

    def gru(prefix, n_in, Hh, L):
        m = 0
        for l in range(L):
            i = n_in if l == 0 else Hh
            m += T * (3 * Hh * i + 3 * Hh * Hh)
        return m

    if arch == "GRU":
        Hh, L = hp["hidden_size"], hp["num_layers"]
        macs = gru("gru", C, Hh, L) + Hh
        peak = 2 * T * Hh + 6 * Hh
    elif arch == "TCN":
        K, cin = hp["kernel_size"], C
        for i, c in enumerate(hp["tcn_channels"]):
            macs += T * c * cin * K + T * c * c * K
            if f"tcn.network.{i}.downsample.weight" in f:
                macs += T * c * cin
            cin = c
        macs += cin
        peak = 3 * T * max(hp["tcn_channels"])
    else:
        K, cc, Hh, L, fd = (hp["kernel_size"], hp["cnn_channels"], hp["gru_hidden"],
                            hp["num_layers"], hp["fusion_dim"])
        macs = T * cc * C * K + fd * cc + gru("gru", C, Hh, L) + fd * Hh + 2 * fd * fd + fd
        peak = T * cc + 2 * T * Hh + 6 * Hh + 5 * fd
    return params, macs, peak


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    nets = json.load(open(os.path.join(ROOT, "export_out", "summary.json")))
    ver = json.load(open(os.path.join(ROOT, "export_out", "verify_c.json")))
    cls = json.load(open(os.path.join(ROOT, "export_classical", "summary.json")))
    num = {}

    # ---------------------------------------------------------------- models
    rows = []          # one per (h, design, fold, arch, precision)
    for k, r in nets.items():
        d = os.path.join(ROOT, "export_out", f"h{r['horizon']}", r["design"], r["outer"])
        sd = dict(np.load(os.path.join(d, f"{r['arch']}.npz")))
        p, m, peak = net_cost(r["arch"], sd, r["hp"])
        for prec in ("f32", "int8"):
            rows.append({"h": r["horizon"], "design": r["design"], "fold": r["outer"],
                         "arch": r["arch"], "prec": prec, "params": p, "macs": m,
                         "act_bytes": 4 * peak,
                         "bytes": r[f"blob_bytes_{prec}"],
                         "ap": r["torch_ap"] if prec == "f32" else r["int8_ap"],
                         "f2": r["torch_f2"] if prec == "f32" else r["int8_f2"],
                         "paper_ap": r["paper_ap"], "n": r["n_test"]})
    for k, r in cls.items():
        d = os.path.join(ROOT, "export_classical", f"h{r['horizon']}", r["design"], r["outer"])
        y = np.fromfile(os.path.join(d, "test_y.u8"), np.uint8)
        dec = np.fromfile(os.path.join(d, f"{r['arch']}_ref_dec.u8"), np.uint8).astype(bool)
        rows.append({"h": r["horizon"], "design": r["design"], "fold": r["outer"],
                     "arch": r["arch"], "prec": "f64", "params": r["n_params"], "macs": None,
                     "act_bytes": 30 * 8, "bytes": r["blob_bytes"], "ap": r["c_ap"],
                     "f2": f2(y, dec), "paper_ap": r["paper_ap"], "n": r["n_test"],
                     "trees": r.get("n_trees"), "config": r.get("config")})
    with open(os.path.join(OUT, "models.json"), "w") as f:
        json.dump(rows, f, indent=1)

    # footprint by architecture (over all 54 folds)
    foot = {}
    for a in ALL:
        for prec in (("f64",) if a in CLS else ("f32", "int8")):
            rs = [r for r in rows if r["arch"] == a and r["prec"] == prec]
            b = np.array([r["bytes"] for r in rs])
            pr = np.array([r["params"] for r in rs])
            e = {"n_models": len(rs), "bytes_median": float(np.median(b)), "bytes_min": int(b.min()),
                 "bytes_max": int(b.max()), "params_median": float(np.median(pr)),
                 "params_min": int(pr.min()), "params_max": int(pr.max())}
            if rs[0]["macs"] is not None:
                mm = np.array([r["macs"] for r in rs])
                e |= {"macs_median": float(np.median(mm)), "macs_min": int(mm.min()),
                      "macs_max": int(mm.max()),
                      "act_bytes_max": int(max(r["act_bytes"] for r in rs))}
            if a == "GBT":
                tr = np.array([r["trees"] for r in rs])
                e |= {"trees_min": int(tr.min()), "trees_max": int(tr.max())}
            foot[f"{a}_{prec}"] = e
    num["footprint"] = foot

    # --------------------------------------------------------------- fidelity
    fid = {}
    for a in NETS:
        for prec in ("f32", "int8"):
            rs = [r for r in ver if r["arch"] == a and r["precision"] == prec]
            fid[f"{a}_{prec}"] = {
                "models": len(rs), "windows": sum(r["n"] for r in rs),
                "c_vs_numpy_max_abs": max(r["c_vs_numpy_max_abs"] for r in rs),
                "decisions_equal": sum(r["c_vs_numpy_decisions_equal"] for r in rs)}
    for a in CLS:
        rs = [r for r in cls.values() if r["arch"] == a]
        fid[f"{a}_f64"] = {"models": len(rs), "windows": sum(r["n_test"] for r in rs),
                           "c_vs_sklearn_max_abs": max(r["c_vs_sklearn_max_abs_diff"] for r in rs),
                           "decisions_equal": sum(r["c_decisions_equal"] for r in rs),
                           "f32_vs_f64_decisions_equal": sum(r["f32_vs_f64_decisions_equal"] for r in rs)}
    fid["numpy_vs_exact_max"] = max(r["numpy_vs_f64_max_abs_diff"] for r in nets.values())
    fid["torch_vs_exact_max"] = max(r["torch_vs_f64_max_abs_diff"] for r in nets.values())
    fid["numpy_torch_decisions_equal"] = sum(r["numpy_decisions_equal"] for r in nets.values())
    fid["total_windows_per_set"] = sum(r["n_test"] for r in nets.values()) // 3
    num["fidelity"] = fid

    # ----------------------------------------------------- refit reproduction
    rep = {}
    for a in NETS:
        dd = np.array([r["torch_ap"] - r["paper_ap"] for r in nets.values() if r["arch"] == a])
        rep[a] = {"exact": int((np.abs(dd) < 1e-4).sum()), "n": len(dd),
                  "mean": float(dd.mean()), "mean_abs": float(np.abs(dd).mean()),
                  "max_abs": float(np.abs(dd).max())}
    for a in ("LR", "GBT"):
        dd = np.array([r["refit_ap_f64"] - r["paper_ap"] for r in cls.values() if r["arch"] == a])
        rep[a] = {"exact": int((np.abs(dd) < 1e-12).sum()), "n": len(dd),
                  "max_abs": float(np.abs(dd).max())}
    num["reproduction"] = rep

    # ------------------------------------------------------------ quantization
    q = {}
    for a in NETS:
        rs = [r for r in nets.values() if r["arch"] == a]
        dap = np.array([r["int8_ap"] - r["torch_ap"] for r in rs])
        df2 = np.array([r["int8_f2"] - r["torch_f2"] for r in rs])
        n = sum(r["n_test"] for r in rs)
        kept = sum(r["int8_decisions_equal"] for r in rs)
        ratio = np.array([r["blob_bytes_int8"] / r["blob_bytes_f32"] for r in rs])
        q[a] = {"dap_mean": float(dap.mean()), "dap_max_abs": float(np.abs(dap).max()),
                "df2_mean": float(df2.mean()), "df2_max_abs": float(np.abs(df2).max()),
                "decisions_changed": n - kept, "windows": n,
                "decisions_changed_pct": 100 * (n - kept) / n,
                "size_ratio_min": float(ratio.min()), "size_ratio_max": float(ratio.max())}
    num["quantization"] = q

    # --------------------------------------------- accuracy of deployed models
    acc = {}
    for h in H:
        for dsg in DESIGNS:
            for a in ALL:
                for prec in (("f64",) if a in CLS else ("f32", "int8")):
                    rs = [r for r in rows if r["h"] == h and r["design"] == dsg and r["arch"] == a
                          and r["prec"] == prec]
                    acc[f"h{h}/{dsg}/{a}/{prec}"] = {
                        "ap_mean": float(np.mean([r["ap"] for r in rs])),
                        "f2_mean": float(np.mean([r["f2"] for r in rs])),
                        "bytes_median": float(np.median([r["bytes"] for r in rs])),
                        "folds": len(rs)}
    num["accuracy"] = acc

    # --------------------------------------------------------- static memory
    num["memory"] = {"aq_static_bytes_default": 40928, "aq_static_bytes_classical": 3008,
                     "aq_window_bytes": 568, "window_bytes": T * C * 4}

    with open(os.path.join(OUT, "numbers.json"), "w") as f:
        json.dump(num, f, indent=1)
    write_tables(num)
    figure(rows, acc)
    print(json.dumps({k: num[k] for k in ("reproduction", "quantization")}, indent=1))
    return 0


def kb(x):
    return f"{x / 1024:.1f}" if x < 10 * 1024 else f"{x / 1024:.0f}"


def write_tables(num):
    t = os.path.join(OUT, "tables")
    foot, fid, q = num["footprint"], num["fidelity"], num["quantization"]
    names = {"WHO": "WHO rule", "LR": "Logistic regression", "GBT": "Boosted trees",
             "GRU": "GRU", "TCN": "TCN", "FUSION": "CNN--GRU hybrid"}
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Model & Weights & Parameters & Model file (KB) & MACs per window \\", r"\midrule"]
    for a in ALL:
        for prec in (("f64",) if a in CLS else ("f32", "int8")):
            e = foot[f"{a}_{prec}"]
            par = (f"{e['params_median']:,.0f} ({e['params_min']:,}--{e['params_max']:,})"
                   if e["params_min"] != e["params_max"] else f"{e['params_min']:,}")
            if a == "GBT":
                par = f"{e['params_median']:,.0f} nodes ({e['params_min']:,}--{e['params_max']:,})"
            by = (f"{kb(e['bytes_median'])} ({kb(e['bytes_min'])}--{kb(e['bytes_max'])})"
                  if e["bytes_min"] != e["bytes_max"] else kb(e["bytes_min"]))
            km = lambda x: f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:,.0f}k"
            mac = (f"{km(e['macs_median'])} ({km(e['macs_min'])}--{km(e['macs_max'])})"
                   if "macs_median" in e else "--")
            w = {"f64": "float64", "f32": "float32", "int8": "int8"}[prec]
            L.append(f"{names[a] if prec != 'int8' else ''} & {w} & {par} & {by} & {mac} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(t, "footprint.tex"), "w").write("\n".join(L).replace(",", "{,}"))

    def sci(x):
        if x == 0:
            return "0"
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r" & & & Max.\ absolute & Decisions \\",
         r"Model & Weights & Folds & score error & equal \\", r"\midrule"]
    for a in CLS:
        e = fid[f"{a}_f64"]
        L.append(f"{names[a]} & float64 & {e['models']} & {sci(e['c_vs_sklearn_max_abs'])} & "
                 f"{100 * e['decisions_equal'] / e['windows']:.0f}\\,\\% \\\\")
    for a in NETS:
        for prec in ("f32", "int8"):
            e = fid[f"{a}_{prec}"]
            L.append(f"{names[a] if prec == 'f32' else ''} & {'float32' if prec == 'f32' else 'int8'} & "
                     f"{e['models']} & {sci(e['c_vs_numpy_max_abs'])} & "
                     f"{100 * e['decisions_equal'] / e['windows']:.0f}\\,\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    s = "\n".join(L)
    open(os.path.join(t, "fidelity_pc.tex"), "w").write(s)

    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Model & Size ratio & Mean $\Delta$AP & Max.\ $|\Delta$AP$|$ & Max.\ $|\Delta F_2|$ & Decisions changed \\",
         r"\midrule"]
    for a in NETS:
        e = q[a]
        L.append(f"{names[a]} & {e['size_ratio_min']:.2f}--{e['size_ratio_max']:.2f} & "
                 f"{e['dap_mean']:+.4f} & {e['dap_max_abs']:.4f} & {e['df2_max_abs']:.4f} & "
                 f"{e['decisions_changed']} ({e['decisions_changed_pct']:.2f}\\,\\%) \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(t, "quantization.tex"), "w").write("\n".join(L))

    acc = num["accuracy"]
    L = [r"\begin{tabular}{ll" + "r" * 6 + "}", r"\toprule",
         r" & & \multicolumn{3}{c}{Within city} & \multicolumn{3}{c}{Across cities} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Model & Precision & 1 h & 12 h & 24 h & 1 h & 12 h & 24 h \\", r"\midrule"]
    for a in ALL:
        for prec in (("f64",) if a in CLS else ("f32", "int8")):
            v = [acc[f"h{h}/{d}/{a}/{prec}"]["ap_mean"] for d in DESIGNS for h in H]
            w = {"f64": "float64", "f32": "float32", "int8": "int8"}[prec]
            L.append(f"{names[a] if prec != 'int8' else ''} & {w} & " + " & ".join(f"{x:.3f}" for x in v)
                     + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(t, "accuracy.tex"), "w").write("\n".join(L))


def figure(rows, acc):
    """AP against model-file size, one point per family and precision (mean over the
    three horizons); an arrow joins each
    network's float32 and 8-bit files. Shared AP axis for the two designs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.35), sharey=True)
    col = {"WHO": "#2a78d6", "LR": "#eb6834", "GBT": "#1baf7a", "GRU": "#eda100",
           "TCN": "#e87ba4", "FUSION": "#008300"}
    mk = {"WHO": "s", "LR": "o", "GBT": "^", "GRU": "D", "TCN": "v", "FUSION": "P"}
    name = {"WHO": "WHO", "LR": "LR", "GBT": "GBT", "GRU": "GRU", "TCN": "TCN", "FUSION": "CNN-GRU"}
    ink = "#333333"
    pts = {}
    for ax, dsg, title in zip(axes, DESIGNS, ("Within city", "Across cities")):
        for a in ALL:
            for prec in (("f64",) if a in CLS else ("f32", "int8")):
                es = [acc[f"h{h}/{dsg}/{a}/{prec}"] for h in H]
                x = float(np.median([e["bytes_median"] for e in es]))
                ys = [e["ap_mean"] for e in es]; y = float(np.mean(ys))
                pts[(dsg, a, prec)] = (x, y)
                ax.scatter(x, y, marker=mk[a], s=30, zorder=3, alpha=0.95,
                           facecolors=col[a] if prec != "int8" else "white",
                           edgecolors=col[a], linewidths=1.0)
            if a not in CLS:
                (x0, y0), (x1, y1) = pts[(dsg, a, "f32")], pts[(dsg, a, "int8")]
                ax.annotate("", xy=(x1 * 1.25, y1), xytext=(x0 / 1.25, y0),
                            arrowprops=dict(arrowstyle="->", color=col[a], lw=0.6, alpha=0.8))
        ax.set_xscale("log"); ax.set_xlim(50, 2e6)
        ax.set_title(title, fontsize=7.5, color=ink)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(colors=ink, labelsize=6.5, width=0.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Mean AP over test folds", color=ink)
    fig.supxlabel("Model file on the device (bytes, log scale)", fontsize=7.5, color=ink, y=0.165)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker=mk[a], color=col[a], ms=4, label=name[a]) for a in ALL]
    hs += [Line2D([], [], ls="", marker="o", markerfacecolor="white", color="#666666", ms=4, label="int8")]
    fig.legend(handles=hs, loc="lower center", ncol=7, frameon=False, fontsize=6.3,
               handletextpad=0.1, columnspacing=0.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_ap_vs_size.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
