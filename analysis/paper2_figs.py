"""Result figures of paper 2 (vector PDF, IEEE column sizes).

    python analysis/paper2_figs.py

fig_tradeoff.pdf     AP against model-file size and against energy per prediction, within
                     a city and across cities; float32 -> 8-bit shown by arrows.
fig_speedup.pdf      latency gain of the flash-aware kernels (Table V data).
fig_sensitivity.pdf  sensor-error sensitivity as heatmaps (Table VII data).
fig_trace.pdf        supply current of three TCN predictions, float32 and 8-bit (PPK2).
Inputs: analysis/out/numbers.json, latency_summary.json, robustness.json, board_results.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
B = os.path.join(ROOT, "board_results")
INK, MUTED, GRID = "#222222", "#555555", "#dddddd"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6,
                     "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK,
                     "axes.labelcolor": INK, "pdf.fonttype": 42})
# validated categorical palette (dataviz skill), fixed order
COL = {"WHO": "#2a78d6", "LR": "#eb6834", "GBT": "#1baf7a", "GRU": "#eda100",
       "TCN": "#e87ba4", "FUSION": "#008300"}
MK = {"WHO": "s", "LR": "o", "GBT": "^", "GRU": "D", "TCN": "v", "FUSION": "P"}
NAME = {"WHO": "WHO rule", "LR": "LR", "GBT": "GBT", "GRU": "GRU", "TCN": "TCN",
        "FUSION": "CNN–GRU"}
ARCHS = ["WHO", "LR", "GBT", "GRU", "TCN", "FUSION"]
H = (1, 12, 24)
DES = (("single-blocked", "Within city"), ("multi", "Across cities"))


def energies():
    """incremental energy (J) per (h, design, arch, prec), weights in flash"""
    L = lambda p: json.load(open(os.path.join(B, p)))
    lat = {}
    for img, files in (("timing", ["v3/latency_timing.json", "v4/latency.json"]),
                       ("fidelity", ["v3/latency_fidelity.json"])):
        for f in files:
            for r in L(f):
                if r["weights_in"] == "flash":
                    lat[(img, r["index"])] = r
    # energy per prediction above idle at the 3.3 V supply (PPK2, marked predictions)
    pe = {}
    for img in ("timing", "fidelity"):
        for m in L(f"ppk2/ppk2_results_{img}.json")["models"]:
            pe[(img, m["index"])] = m["energy_mJ"] * 1e-3
    e = {}
    for (img, i), r in lat.items():
        if (img, i) not in pe:
            continue
        if img == "timing" and r["precision"] == "f64":
            continue                     # the extra worst-case GBT
        if img == "fidelity" and r["precision"] != "f64":
            continue
        e[(r["horizon"], r["design"], r["arch"], r["precision"])] = pe[(img, i)]
    return e


def tradeoff():
    acc = json.load(open(os.path.join(OUT, "numbers.json")))["accuracy"]
    en = energies()
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.95), sharey=True)
    for r, (xkind, xlabel) in enumerate((("size", "Model file (bytes)"),
                                         ("energy", "Energy per prediction (mJ)"))):
        for c, (dsg, title) in enumerate(DES):
            ax = axes[r, c]
            pts = {}
            for a in ARCHS:
                for prec in (("f64",) if a in ("WHO", "LR", "GBT") else ("f32", "int8")):
                    es = [acc[f"h{h}/{dsg}/{a}/{prec}"] for h in H]
                    y = float(np.mean([v["ap_mean"] for v in es]))
                    x = (float(np.median([v["bytes_median"] for v in es])) if xkind == "size"
                         else 1e3 * float(np.median([en[(h, dsg, a, prec)] for h in H])))
                    pts[(a, prec)] = (x, y)
                    ax.scatter(x, y, marker=MK[a], s=24, zorder=3, linewidths=0.7,
                               facecolors=COL[a] if prec != "int8" else "white", edgecolors=COL[a])
                if a not in ("WHO", "LR", "GBT"):
                    (x0, y0), (x1, y1) = pts[(a, "f32")], pts[(a, "int8")]
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                                arrowprops=dict(arrowstyle="-|>", color=COL[a], lw=0.6,
                                                shrinkA=3.5, shrinkB=3.5, mutation_scale=6))
            if (r, c) in ((0, 0), (1, 1)):
                x, y = pts[("TCN", "int8")]
                ax.annotate("8-bit TCN", xy=(x, y), xytext=(x * (0.012 if xkind == "size" else 0.05),
                            y + (0.04 if xkind == "size" else 0.075)), fontsize=6.9, color=INK, zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.5, shrinkA=1,
                                            shrinkB=2.2, mutation_scale=5))
            ax.set_xscale("log")
            ax.set_ylim(0.43, 0.80)
            ax.grid(color="#f0f0f0", lw=0.3)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=6.2, width=0.5, length=2.2, pad=1.5)
            ax.text(0.03, 0.97, f"({'abcd'[2 * r + c]})", transform=ax.transAxes, va="top",
                    fontsize=7)
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK, pad=3)
            if c == 0:
                ax.set_xlabel(xlabel, fontsize=7.2, labelpad=1.5, x=1.06)
    fig.text(0.015, 0.56, "Mean AP over test folds", rotation=90, va="center", fontsize=7.5, color=INK)
    fam = [Line2D([], [], ls="", marker=MK[a], color=COL[a], ms=4.5, label=NAME[a]) for a in ARCHS]
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.patches import FancyArrowPatch
    class _Arrow(HandlerPatch):
        def create_artists(self, legend, orig, x0, y0, w, h, fs, trans):
            p = FancyArrowPatch((x0, y0 + h / 2), (x0 + w, y0 + h / 2), arrowstyle="-|>",
                                mutation_scale=6, lw=0.6, color=MUTED)
            p.set_transform(trans)
            return [p]
    arrow = FancyArrowPatch((0, 0), (1, 0))
    prec = [Line2D([], [], ls="", marker="v", color=MUTED, ms=4.5, label="floating-point weights"),
            Line2D([], [], ls="", marker="v", markerfacecolor="white", color=MUTED, ms=4.5,
                   label="8-bit weights"),
            arrow]
    fig.legend(handles=fam, loc="lower center", ncol=6, frameon=False, fontsize=6.9,
               handletextpad=0.1, columnspacing=0.6, bbox_to_anchor=(0.53, 0.035))
    fig.legend(handles=prec, labels=["floating-point weights", "8-bit weights", "quantization"],
               handler_map={FancyArrowPatch: _Arrow()}, loc="lower center", ncol=3, frameon=False,
               fontsize=6.9, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.53, -0.005))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.17, hspace=0.48, wspace=0.12)
    fig.savefig(os.path.join(OUT, "fig_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def speedup():
    s = json.load(open(os.path.join(OUT, "latency_summary.json")))["speedup"]
    rows = [("TCN", "f32"), ("FUSION", "f32"), ("GRU", "int8"), ("GRU", "f32"),
            ("TCN", "int8"), ("FUSION", "int8")]
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    for i, (a, p) in enumerate(rows):
        r = s[f"{a}/{p}"]
        g = r["gain"]
        ax.barh(i, g, height=0.62, color=COL[a] if p == "f32" else "white",
                edgecolor=COL[a], linewidth=1.0, hatch=None if p == "f32" else "////")
        f = lambda ms: f"{ms:,.0f}" if ms >= 100 else (f"{ms:.0f}" if ms >= 10 else f"{ms:.1f}")
        ax.text(g + 0.25, i, f"{g:.1f}×   {f(r['before_ms'])} → {f(r['after_ms'])} ms",
                va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{NAME[a]} {'float32' if p == 'f32' else 'int8'}" for a, p in rows],
                       fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 26)
    ax.axvline(1, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("Speed-up over straightforward kernels (×), weights in flash")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6.5, width=0.5, length=2.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_speedup.pdf"), bbox_inches="tight")
    plt.close(fig)


def sensitivity():
    a = json.load(open(os.path.join(OUT, "robustness.json")))
    SC = [("pm_gain_0.70", "PM ×0.70"), ("pm_gain_0.85", "PM ×0.85"), ("pm_gain_1.15", "PM ×1.15"),
          ("pm_gain_1.30", "PM ×1.30"), ("pm_gain_1.50", "PM ×1.50"), ("pm_offset_-5", "PM −5 µg/m³"),
          ("pm_offset_+5", "PM +5 µg/m³"), ("t_offset_+3", "T +3 °C"), ("rh_offset_-10", "RH −10 %")]
    blue = ["#104281", "#2a78d6", "#86b6ef"]
    red = ["#f4a6a6", "#d03b3b", "#8e1f1f"]
    div = LinearSegmentedColormap.from_list("div", red[::-1] + ["#f0efec"] + blue[::-1])  # loss = red
    seq = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    panels = [("mean_d_ap", 100, "Change in AP (points)", div, True),
              ("mean_d_f2", 100, "Change in $F_2$ (points)", div, True),
              ("flipped_pct", 1, "Decisions changed (%)", seq, False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    for ax, (key, k, title, cmap, diverging) in zip(axes, panels):
        M = np.array([[a[f"{ar}/{sc}"][key] * k for ar in ARCHS] for sc, _ in SC])
        if diverging:
            lim = np.abs(M).max()
            norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
        else:
            norm = plt.Normalize(0, M.max())
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                dark = abs(norm(v) - 0.5) > 0.33 if diverging else norm(v) > 0.55
                ax.text(j, i, ("0.0" if abs(v) < 0.05 else f"{v:.1f}").replace("-", "−"), ha="center", va="center", fontsize=6,
                        color="white" if dark else INK)
        ax.set_xticks(range(len(ARCHS)))
        ax.set_xticklabels([{"WHO": "WHO", "FUSION": "CNN–\nGRU"}.get(x, NAME[x]) for x in ARCHS],
                           fontsize=6.5)
        ax.set_yticks(range(len(SC)))
        ax.set_yticklabels([s for _, s in SC], fontsize=6.8)
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)



def trace():
    """Supply current of three consecutive predictions of the same TCN (Antwerp, h=1),
    float32 and 8-bit weights, from the PPK2 record (fig_trace.pdf)."""
    z = np.load(os.path.join(B, "ppk2", "ppk2_results_timing_excerpts.npz"))
    res = {m["index"]: m for m in json.load(open(os.path.join(
        B, "ppk2", "ppk2_results_timing.json")))["models"]}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.15), sharex=True, sharey=True)
    for ax, (i, lab) in zip(axes, ((2, "(a) float32"), (3, "(b) 8-bit"))):
        c = z[f"m{i}_cur"] / 1000.0
        d = z[f"m{i}_d0"].astype(bool)
        dt = float(z[f"m{i}_dt"])
        k = max(1, int(round(1e-4 / dt)))                 # 0.1 ms means
        n = len(c) // k
        cm = c[:n * k].reshape(n, k).mean(1)
        t = (np.arange(n) * k * dt) * 1e3
        dm = d[:n * k].reshape(n, k).mean(1) > 0.5
        ax.fill_between(t, 30, 92, where=dm, color=COL["TCN"], alpha=0.13, lw=0,
                        step="mid")
        for r0 in np.flatnonzero(np.diff(d.astype(np.int8)) == 1) + 1:   # prediction starts
            ax.axvline(r0 * dt * 1e3, color="#444444", lw=0.6, ls=(0, (2, 1.5)))
        ax.plot(t, cm, color=COL["TCN"] if i == 2 else "#a8426f", lw=0.8)
        idle = res[i]["idle_mA"]
        ax.axhline(idle, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        m = res[i]
        ax.text(0.99, 0.985, f"{lab}: {m['high_median_us'] / 1e3:.1f} ms, "
                f"{m['energy_mJ']:.2f} mJ per prediction", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.4, color=INK,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.set_ylim(30, 92)
        ax.set_yticks([40, 60, 80])
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].text(78, res[3]["idle_mA"] + 1.0, "idle", ha="right", va="bottom",
                 fontsize=7, color=MUTED)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_xlim(-3, 80)
    fig.text(0.015, 0.55, "Supply current (mA)", rotation=90, va="center", fontsize=7.5)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.98, bottom=0.2, hspace=0.12)
    fig.savefig(os.path.join(OUT, "fig_trace.pdf"))
    fig.savefig(os.path.join(OUT, "fig_trace.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    tradeoff()
    speedup()
    sensitivity()
    trace()
    print("figures written to", OUT)
