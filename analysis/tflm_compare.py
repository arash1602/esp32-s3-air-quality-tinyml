"""aq_infer vs TensorFlow Lite Micro on the same 36 trained networks (fidelity image).

    python analysis/tflm_compare.py      -> analysis/out/tables/tflm.tex, analysis/out/tflm_summary.json
"""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "board_results")
L = lambda p: json.load(open(os.path.join(B, p)))
key = lambda h, d, a: (int(h), d, a)
# aq_infer: PPK2 fidelity run (marked time and energy), file sizes from the fidelity index
aq = {}
fid = {e["index"]: e for e in L("../models_fidelity_index.json")} if False else {e["index"]: e for e in json.load(open(os.path.join(ROOT, "models_fidelity_index.json")))}
for m in L("ppk2/ppk2_results_fidelity.json")["models"]:
    if m["arch"] in ("GRU", "TCN", "FUSION"):
        aq[key(m["horizon"], m["design"], m["arch"]) + (m["precision"],)] = {
            "ms": m["high_median_us"] / 1e3, "mJ": m["energy_mJ"], "kb": fid[m["index"]]["bytes"] / 1024}
# TFLM: board run (decisions, latency, arena), PPK2 energy, PC interpreter stats
tf = {}
stats = L("tflm/pc_stats.json")
for f in ("tflm_board_tflm_a.json", "tflm_board_tflm_b.json"):
    for r in L("tflm/" + f):
        p = r["precision"].replace("tfl_", "")
        k = key(r["horizon"], r["design"], r["arch"]) + (p,)
        s = stats[r["key"]]
        tf[k] = {"kb": s["bytes"] / 1024, "pc_changed": s["decisions_changed"], "n_test": s["n_test"],
                 "dap": s["ap"] - s["ap_ref"]}
        if "error" in r:
            tf[k]["fits"] = False
            continue
        tf[k] |= {"fits": True, "ms": r["median_us"] / 1e3, "arena_kb": r["arena_bytes"] / 1024,
                  "board_vs_ref": r["board_vs_ref_decisions_differ"], "board_n": r["windows"]}
for f, ix in (("ppk2_results_tflm_a.json", "tflm_a_index.json"),
              ("ppk2_results_tflm_b_part.json", "tflm_b_index.json"),
              ("ppk2_results_tflm_b2.json", "tflm_b2_index.json")):
    idx = {e["index"]: e for e in L("tflm/" + ix)}
    for m in L("tflm/" + f)["models"]:
        e = idx[m["index"]]
        k = key(e["horizon"], e["design"], e["arch"]) + (e["precision"].replace("tfl_", ""),)
        tf[k]["mJ"] = m["energy_mJ"]
        tf[k]["ms_ppk2"] = m["high_median_us"] / 1e3
NAME = {"TCN": "TCN", "FUSION": "CNN--GRU", "GRU": "GRU"}
PREC = {"f32": "float32", "int8": "int8"}
rows, summary = [], {}
def rng(v, f):
    v = sorted(v); a, b = f(v[0]), f(v[-1]); return a if a == b else f"{a}--{b}"
f1 = lambda x: f"{x:.1f}" if x < 100 else f"{x:.0f}"
f2 = lambda x: f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")
for arch in ("TCN", "FUSION", "GRU"):
    for p in ("f32", "int8"):
        ks = sorted(k for k in tf if k[2] == arch and k[3] == p)
        ok = [k for k in ks if tf[k]["fits"]]
        sp = [tf[k]["ms"] / aq[k]["ms"] for k in ok]          # >1: TFLM slower
        en = [tf[k]["mJ"] / aq[k]["mJ"] for k in ok if "mJ" in tf[k]]
        ch = sum(tf[k]["pc_changed"] for k in ks); n = sum(tf[k]["n_test"] for k in ks)
        bd = sum(tf[k].get("board_vs_ref", 0) for k in ok); bn = sum(tf[k].get("board_n", 0) for k in ok)
        summary[f"{arch}/{p}"] = {"n": len(ks), "fit": len(ok), "time_ratio": [min(sp), max(sp)],
                                  "energy_ratio": [min(en), max(en)], "pc_changed": ch, "pc_n": n,
                                  "board_changed": bd, "board_n": bn,
                                  "arena_kb": [min(tf[k]["arena_kb"] for k in ok), max(tf[k]["arena_kb"] for k in ok)],
                                  "tf_kb": [min(tf[k]["kb"] for k in ks), max(tf[k]["kb"] for k in ks)],
                                  "aq_kb": [min(aq[k]["kb"] for k in ks), max(aq[k]["kb"] for k in ks)]}
        s = summary[f"{arch}/{p}"]
        fitcell = f"{len(ok)}/{len(ks)}"
        rows.append(f"{NAME[arch]} & {PREC[p]} & {fitcell} & {rng(s['aq_kb'], f1)} & {rng(s['tf_kb'], f1)} & "
                    f"{rng(s['arena_kb'], f1)} & {rng(s['time_ratio'], lambda x: f'{x:.1f}')} & "
                    f"{rng(s['energy_ratio'], lambda x: f'{x:.1f}')} & {ch}/{n:,}".replace(",", "{,}") + r" \\")
T = [r"\begin{table*}[t]", r"\centering",
     r"\caption{The library of this paper (aq\_infer) against TensorFlow Lite Micro (TFLM) on the ESP32-S3: model storage, working memory, latency, energy and numerical fidelity.}",
     r"\label{tab:tflm}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r" & & TFLM & \multicolumn{2}{c}{Model file (KB)} & TFLM arena & \multicolumn{2}{c}{TFLM / aq\_infer} & Decisions changed \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){7-8}",
     r"Network & Weights & fit & aq\_infer & TFLM & (KB) & Time & Energy & by TFLM (PC) \\", r"\midrule"] + rows + \
    [r"\bottomrule", r"\end{tabular}",
     r"",
     r"\vspace{2pt}",
     r"\parbox{\textwidth}{\scriptsize 36 trained networks (the fidelity configurations: three horizons, within and across cities), TFLM esp-tflite-micro~1.4.2 with ESP-NN kernels on ESP-IDF~6.1, same board. TFLM fit: configurations that run with a 256\,KB tensor arena; arena: TFLM's working memory per network. Time and energy: ratio TFLM/aq\_infer per network for the configurations that both run, so values above 1 favour aq\_infer; energy above idle at 3.3\,V (PPK2). Decisions changed: TFLM's PC interpreter against the reference on all 17{,}704 test windows of the six networks of each row; on the board, TFLM's ESP-NN kernels changed 13 further decisions of 17{,}000 sampled (Section~\ref{sec:tflm}). The two int8 schemes differ: aq\_infer quantizes weights only and keeps floating-point activations, TFLM quantizes weights and activations (full-integer). GRUs are unrolled over the 24 time steps in TFLM, which has no GRU operator.}",
     r"\end{table*}"]
os.makedirs(os.path.join(ROOT, "analysis/out/tables"), exist_ok=True)
open(os.path.join(ROOT, "analysis/out/tables/tflm.tex"), "w").write("\n".join(T) + "\n")
json.dump(summary, open(os.path.join(ROOT, "analysis/out/tflm_summary.json"), "w"), indent=1)
print("\n".join(rows))
# detail
for k in sorted(tf):
    t = tf[k]
    if t["fits"]: print(k, f"{aq[k]['ms']:.1f}/{t['ms']:.1f} ms  {aq[k]['mJ']:.2f}/{t.get('mJ', float('nan')):.2f} mJ  ppk2ms {t.get('ms_ppk2')}")
