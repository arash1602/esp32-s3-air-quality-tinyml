"""LaTeX table from analysis/out/robustness.json."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = json.load(open(os.path.join(ROOT, "analysis/out/robustness.json")))
AR = [("WHO", "WHO"), ("LR", "LR"), ("GBT", "GBT"), ("GRU", "GRU"), ("TCN", "TCN"), ("FUSION", "CNN--GRU")]
SC = [("pm_gain_0.70", r"PM $\times0.70$"), ("pm_gain_0.85", r"PM $\times0.85$"),
      ("pm_gain_1.15", r"PM $\times1.15$"), ("pm_gain_1.50", r"PM $\times1.50$"),
      ("pm_offset_-5", r"PM $-5\,\mu$g/m$^3$"), ("pm_offset_+5", r"PM $+5\,\mu$g/m$^3$"),
      ("t_offset_+3", r"$T$ $+3\,^\circ$C"), ("rh_offset_-10", r"RH $-10$\,pp")]
def f(x):
    # in points (x 100), one decimal
    v = round(100 * x, 1)
    if v == 0: return "0.0"
    return (f"{v:+.1f}").replace("-", "$-$")
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Sensitivity of the deployed models to sensor error. Every test window is perturbed as stated and run through the C library. $\Delta$AP, $\Delta F_2$: change in AP and in $F_2$ in points ($\times100$), mean over the 54 test folds; Chg.: share of the 125{,}976 test-window decisions that change (\%).}",
     r"\label{tab:robust}", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
     r"\begin{tabular}{l" + "rrr" * len(AR) + "}", r"\toprule",
     " & " + " & ".join(r"\multicolumn{3}{c}{%s}" % n for _, n in AR) + r" \\",
     "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(AR))),
     "Perturbation & " + " & ".join([r"$\Delta$AP & $\Delta F_2$ & Chg."] * len(AR)) + r" \\", r"\midrule"]
for k, name in SC:
    cells = []
    for ar, _ in AR:
        r = a[f"{ar}/{k}"]
        cells += [f(r["mean_d_ap"]), f(r["mean_d_f2"]), f"{r['flipped_pct']:.1f}"]
    L.append(name + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(ROOT, "analysis/out/tables/robustness.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
