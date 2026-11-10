"""Latency tables from the board runs (final kernels).

Networks: GRU and CNN-GRU from board_results/v4 (row-wise input projections, partial
staging), TCN from v3 (unchanged since); classical models from v3 (fidelity image).
Speed-up table: the naive kernels (v1) against the final ones."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "board_results")
L = lambda p: json.load(open(os.path.join(B, p)))
v1 = L("v1_naive_kernels/latency.json")
v3t, v3f, v4 = L("v3/latency_timing.json"), L("v3/latency_fidelity.json"), L("v4/latency.json")
for r in v4 + v3t: r["image"] = "timing"
for r in v3f: r["image"] = "fidelity"
final = [r for r in v4] + [r for r in v3t if r["arch"] == "TCN"] + \
        [r for r in v3f if r["precision"] == "f64"] + [r for r in v3t if r["precision"] == "f64"]
# energy: per prediction above idle at the 3.3 V supply, from the PPK2 runs (marked
# predictions, analysis/ppk2_energy.py); FNB58 (5 V USB input) kept as a cross-check
PE = {}
for img in ("timing", "fidelity"):
    for m in L(f"ppk2/ppk2_results_{img}.json")["models"]:
        PE[(img, m["index"])] = m
EN = {}
for img, f in (("timing", "energy/energy_results_timing.json"),
               ("fidelity", "energy/energy_results_classical.json")):
    for r in L(f):
        EN[(img, r["index"])] = r["dP_W"]
def energy_mj(r):
    m = PE.get((r["image"], r["index"]))
    return None if m is None else m["energy_mJ"]
NAME = {"WHO": "WHO rule", "LR": "LR", "GBT": "GBT", "GRU": "GRU",
        "TCN": "TCN", "FUSION": "CNN--GRU"}
PREC = {"f64": "float64", "f32": "float32", "int8": "int8"}
def ms(x):
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:,.0f}".replace(",", "{,}"))
def rng(v):
    v = sorted(v)
    a, b = ms(v[0]), ms(v[-1])
    return a if a == b else f"{a}--{b}"
def kb(v):
    v = sorted(v); f = lambda x: f"{x/1024:.1f}" if x < 10240 else f"{x/1024:.0f}"
    return f(v[0]) if v[0] == v[-1] else f"{f(v[0])}--{f(v[-1])}"
rows = []
summary = {}
for arch in ("WHO", "LR", "GBT", "GRU", "TCN", "FUSION"):
    for prec in ("f64", "f32", "int8"):
        rs = [r for r in final if r["arch"] == arch and r["precision"] == prec]
        if not rs: continue
        fl = {(r["index"], r["fold"]): r for r in rs if r["weights_in"] == "flash"}
        ra = {(r["index"], r["fold"]): r for r in rs if r["weights_in"] == "ram"}
        fms = [r["median_us"] / 1e3 for r in fl.values()]
        rms = [r["median_us"] / 1e3 for r in ra.values()]
        nofit = len(fl) - len(ra)
        rcell = rng(rms) if rms else "--"
        if nofit: rcell += f"$^{{({len(ra)})}}$"
        ens = [energy_mj(r) for r in fl.values()]
        ens = [x for x in ens if x is not None]
        ecell = rng(ens) if len(ens) == len(fl) else "--"
        rows.append(f"{NAME[arch]} & {PREC[prec]} & {len(fl)} & {kb([r['blob_bytes'] for r in fl.values()])} & {rng(fms)} & {rcell} & {ecell} \\\\")
        summary[f"{arch}/{prec}"] = {"n": len(fl), "flash_ms": [min(fms), max(fms)],
                                    "ram_ms": [min(rms), max(rms)] if rms else None, "ram_nofit": nofit,
                                    "energy_mJ": [min(ens), max(ens)] if ens else None,
                                    "dP_mW": [min(PE[(r["image"], r["index"])]["dP_high_mW"] for r in fl.values()),
                                              max(PE[(r["image"], r["index"])]["dP_high_mW"] for r in fl.values())]}
T = [r"\begin{table}[t]", r"\centering",
     r"\caption{Latency and energy of one prediction on the ESP32-S3 at 240\,MHz, range over the models measured ($n$). Latency: median over at least 10 (up to 1{,}000) repetitions; Flash: weights read in place, as deployed; RAM: model file copied to RAM first ($^{*}$some models of the row do not fit). Energy: energy above idle at the 3.3\,V supply, Eq.~(\ref{eq:energy}), measured over marked predictions with weights in flash.}",
     r"\label{tab:latency}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{llrrrrr}", r"\toprule",
     r" & & & & \multicolumn{2}{c}{Latency (ms)} & Energy \\", r"\cmidrule(lr){5-6}",
     r"Model & Weights & $n$ & File (KB) & Flash & RAM & (mJ) \\", r"\midrule"] + rows + \
    [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
# speed-up of the flash-aware kernels, weights in flash
v1k = {(r["index"], r["weights_in"]): r for r in v1}
pick = []
for r in final:
    if r["weights_in"] != "flash" or r["precision"] == "f64": continue
    o = v1k.get((r["index"], "flash"))
    if o and o["fold"] == r["fold"]:
        pick.append((o["median_us"] / r["median_us"], r, o))
best = {}
for s, r, o in pick:
    k = (r["arch"], r["precision"])
    if k not in best or s > best[k][0]: best[k] = (s, r, o)
S = [r"\begin{table}[t]", r"\centering",
     r"\caption{Effect of the flash-aware kernels (Section~\ref{sec:flash}), weights read from flash: median latency with straightforward kernels and with the final ones, for the model of each family and precision that gains most. Outputs are bit-identical.}",
     r"\label{tab:speedup}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{llrrrr}", r"\toprule",
     r"Model & Weights & File (KB) & Before (ms) & After (ms) & Gain \\", r"\midrule"]
for arch in ("GRU", "TCN", "FUSION"):
    for prec in ("f32", "int8"):
        if (arch, prec) not in best: continue
        s, r, o = best[(arch, prec)]
        S.append(f"{NAME[arch]} & {PREC[prec]} & {kb([r['blob_bytes']])} & {ms(o['median_us']/1e3)} & {ms(r['median_us']/1e3)} & {s:.1f}$\\times$ \\\\")
S += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
os.makedirs(os.path.join(ROOT, "analysis/out/tables"), exist_ok=True)
open(os.path.join(ROOT, "analysis/out/tables/latency.tex"), "w").write("\n".join(T) + "\n")
open(os.path.join(ROOT, "analysis/out/tables/speedup.tex"), "w").write("\n".join(S) + "\n")
json.dump({"latency": summary, "speedup": {f"{a}/{p}": {"before_ms": o["median_us"]/1e3, "after_ms": r["median_us"]/1e3, "gain": s, "bytes": r["blob_bytes"]} for (a, p), (s, r, o) in best.items()}},
          open(os.path.join(ROOT, "analysis/out/latency_summary.json"), "w"), indent=1)
print("\n".join(rows)); print("\n".join(S[8:-3]))
