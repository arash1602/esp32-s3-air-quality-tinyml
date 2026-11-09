"""Export paper 1's classical predictors (WHO rule, logistic regression, gradient-boosted
trees) for the device, and check the C library against scikit-learn on every test fold.

    python export_classical.py --code-dir ../code [--data-dir ...] [--horizons 1 12 24]

No PyTorch or GPU needed. For every horizon, design and outer fold, the configuration
and threshold that paper 1's inner-fold selection chose (``extra_baselines_ctx24.json``)
are refitted on the same ``refit_idx`` training split, and the script checks:

  1. the refit reproduces paper 1's test average precision (deterministic, so exactly);
  2. on the float32 windows the device receives, the C library (c/aq_native) gives the
     same scores (to ~1e-15) and the same decisions as scikit-learn;
  3. how many decisions differ between float32 and float64 windows (the device gets
     float32 readings; paper 1's features were computed in float64).

Writes export_classical/h<h>/<design>/<fold>/{WHO,LR,GBT}.aqm, test_windows.f32,
test_y.u8, <ARCH>.json, the scikit-learn scores and decisions on the float32 windows
(<ARCH>_ref.f64, <ARCH>_ref_dec.u8: the reference for tools/hil.py), and
export_classical/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blob  # noqa: E402

NATIVE = os.path.join(HERE, "c", "aq_native")


def build_native():
    src = [os.path.join(HERE, "c", f) for f in ("aq_native.c", "aq_infer.c")]
    subprocess.run(["gcc", "-O2", "-std=c99", "-o", NATIVE, *src, "-lm"], check=True)


def run_c(blob_path, win_path):
    out = subprocess.run([NATIVE, "-p", blob_path, win_path], check=True,
                         capture_output=True, text=True).stdout.split()
    a = np.asarray(out, dtype=object).reshape(-1, 2)
    return a[:, 0].astype(np.float64), a[:, 1].astype(np.int64).astype(bool)


def raw_windows(ds):
    T = ds.context_size
    return np.stack([ds._series[w.series_idx][0][w.start:w.start + T] for w in ds._windows])


def stats_of(win):
    """paper 1's 30 window statistics (extra_baselines.window_arrays), from windows."""
    x = win.astype(np.float64)
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), x[:, -1], x[:, -1] - x[:, 0]],
                          axis=1)


def stats_rowwise(win):
    """Same, computed per window as extra_baselines does (bit-identical reference)."""
    out = np.empty((len(win), 6 * win.shape[2]))
    for i, w in enumerate(win):
        x = w.astype(np.float64)
        out[i] = np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0), x[-1], x[-1] - x[0]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "export_classical"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    ap.add_argument("--designs", nargs="+", default=["single-blocked", "multi"])
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.code_dir))
    import extra_baselines as eb
    import naive_baselines as nb
    from compare_models import nested_plan
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score

    data_dir = args.data_dir or os.environ.get("AQM_DATA_DIR") or os.path.join(
        args.code_dir, "senseurcity_data_v03", "dataset")
    build_native()
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    lr_cols = [list(nb.INPUT_FEATURES).index(f) for f in nb.LR_FEATURES]
    i25, i10 = (list(nb.INPUT_FEATURES).index(f) for f in ("OPCN3PM25", "OPCN3PM10"))
    t0 = time.time()

    for h in args.horizons:
        nb.HORIZON = h
        common = {"context_size": nb.CONTEXT_SIZE, "overlap": nb.OVERLAP, "horizon": h}
        with open(os.path.join(args.code_dir, "tuning_results", f"h{h}",
                               f"extra_baselines_ctx{nb.CONTEXT_SIZE}.json"), encoding="utf-8") as f:
            paper = {(r["design"], r["outer"]): r for r in json.load(f)}
        for design in args.designs:
            experiment, protocol = nb.EXPERIMENTS[design]
            specs = []
            cities = [[c] for c in nb.CITIES] if experiment == "single" else [list(nb.CITIES)]
            for cs in cities:
                specs += nested_plan(data_dir, experiment=experiment, protocol=protocol, cities=cs,
                                     k_outer=nb.K_OUTER, n_inner=nb.N_INNER,
                                     block_hours=nb.BLOCK_HOURS, **common)
            for spec in specs:
                key = f"h{h}/{design}/{spec.outer_id}"
                if (design, spec.outer_id) not in paper or all(f"{key}/{a}" in summary
                                                                for a in ("WHO", "LR", "GBT")):
                    continue
                prow = paper[(design, spec.outer_id)]
                test_ds = nb._split(data_dir, spec.test_cities, bands=spec.test_bands, **common)
                isp = next(i for i in spec.inner if i.idx == spec.refit_idx)
                refit_ds = nb._split(data_dir, spec.train_cities, bands=isp.train_bands, **common)
                te, rf = eb.window_arrays(test_ds), eb.window_arrays(refit_ds)
                y = te[2]
                win64 = raw_windows(test_ds)
                win32 = win64.astype(np.float32)
                d = os.path.join(args.out, f"h{h}", design, spec.outer_id)
                os.makedirs(d, exist_ok=True)
                wpath = os.path.join(d, "test_windows.f32")
                win32.astype("<f4").tofile(wpath)
                y.astype(np.uint8).tofile(os.path.join(d, "test_y.u8"))
                s32 = stats_rowwise(win32)
                same_stats = bool(np.array_equal(stats_rowwise(win64), te[0]))

                # ---- fit the three predictors exactly as paper 1 refitted them
                cfg_lr = prow["logreg"]["config"]
                sc, lr = nb.fit_logreg(rf[1][:, lr_cols], rf[2], cfg_lr["C"])
                cfg_gb = prow["gbm"]["config"]
                gb = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                                    random_state=nb.SEED, **cfg_gb).fit(rf[0], rf[2])
                models = {
                    "WHO": dict(
                        paper_ap=None,
                        p64=lambda st: np.maximum(st[:, i25] / 15.0, st[:, i10] / 45.0),
                        dec=lambda st: (st[:, i25] > 15.0) | (st[:, i10] > 45.0),
                        raw=lambda st: np.maximum(st[:, i25] / 15.0, st[:, i10] / 45.0),
                        bytes=blob.who_blob_bytes(i25, i10), thr_prob=None, n_params=0),
                    "LR": dict(
                        paper_ap=prow["logreg"]["pr_auc"],
                        p64=lambda st: nb.lr_score(sc, lr, st[:, :5][:, lr_cols]),
                        raw=lambda st: lr.decision_function(sc.transform(st[:, :5][:, lr_cols])),
                        thr_prob=prow["logreg"]["threshold"], n_params=len(lr_cols) + 1),
                    "GBT": dict(
                        paper_ap=prow["gbm"]["pr_auc"],
                        p64=lambda st: gb.predict_proba(st)[:, 1],
                        raw=lambda st: gb._raw_predict(st).ravel(),
                        thr_prob=prow["gbm"]["threshold"],
                        n_params=int(sum(len(p[0].nodes) for p in gb._predictors))),
                }
                for a in ("LR", "GBT"):
                    m = models[a]
                    m["thr_raw"] = blob.ge_threshold_on_raw(m["thr_prob"])
                    m["dec"] = (lambda mm: lambda st: mm["p64"](st) >= mm["thr_prob"])(m)
                models["LR"]["bytes"] = blob.lr_blob_bytes(
                    lr_cols, sc.mean_, sc.scale_, lr.coef_.ravel(), float(lr.intercept_[0]),
                    models["LR"]["thr_raw"])
                models["GBT"]["bytes"] = blob.gbt_blob_bytes(gb, models["GBT"]["thr_raw"])

                line = []
                for a, m in models.items():
                    bpath = os.path.join(d, f"{a}.aqm")
                    with open(bpath, "wb") as f:
                        f.write(m["bytes"])
                    c_score, c_dec = run_c(bpath, wpath)
                    ref_raw32, ref_dec32 = m["raw"](s32), m["dec"](s32)
                    np.asarray(ref_raw32, "<f8").tofile(os.path.join(d, f"{a}_ref.f64"))
                    np.asarray(ref_dec32, np.uint8).tofile(os.path.join(d, f"{a}_ref_dec.u8"))
                    dec64 = m["dec"](te[0])
                    rec = {"arch": a, "horizon": h, "design": design, "outer": spec.outer_id,
                           "n_test": int(len(y)), "prevalence": float(y.mean()),
                           "blob_bytes": len(m["bytes"]), "n_params": m["n_params"],
                           "threshold_prob": m["thr_prob"],
                           "threshold_raw": m.get("thr_raw", 1.0),
                           "paper_ap": m["paper_ap"],
                           "refit_ap_f64": float(average_precision_score(y, m["p64"](te[0]))),
                           "ap_f32": float(average_precision_score(y, ref_raw32)),
                           "c_ap": float(average_precision_score(y, c_score)),
                           "c_vs_sklearn_max_abs_diff": float(np.max(np.abs(c_score - ref_raw32))),
                           "c_decisions_equal": int((c_dec == ref_dec32).sum()),
                           "f32_vs_f64_decisions_equal": int((ref_dec32 == dec64).sum()),
                           "stats_identical_to_paper": same_stats,
                           "positives_predicted": int(c_dec.sum())}
                    if a in ("LR", "GBT"):
                        # sklearn's probability is expit(raw): the raw threshold is exact
                        p = m["p64"](s32)
                        rec["raw_threshold_exact"] = bool(np.array_equal(
                            p >= m["thr_prob"], ref_raw32 >= m["thr_raw"]))
                    if a == "GBT":
                        rec["config"] = cfg_gb
                        rec["n_trees"] = len(gb._predictors)
                    if a == "LR":
                        rec["config"] = cfg_lr
                    with open(os.path.join(d, f"{a}.json"), "w", encoding="utf-8") as f:
                        json.dump(rec, f, indent=1)
                    summary[f"{key}/{a}"] = rec
                    line.append(f"{a} AP {rec['c_ap']:.3f}"
                                + (f" (paper {m['paper_ap']:.3f})" if m["paper_ap"] is not None else "")
                                + f" C-sk {rec['c_vs_sklearn_max_abs_diff']:.0e}"
                                + f" dec {rec['c_decisions_equal']}/{rec['n_test']}")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=1)
                print(f"{key:32s} " + " | ".join(line) + f" | {time.time() - t0:.0f}s", flush=True)

    recs = list(summary.values())
    fit = [r for r in recs if r["paper_ap"] is not None]
    print(f"\n{len(recs)} models")
    print("refit vs paper 1 AP, largest difference:",
          f"{max(abs(r['refit_ap_f64'] - r['paper_ap']) for r in fit):.2e}")
    print("C vs scikit-learn score, largest difference:",
          f"{max(r['c_vs_sklearn_max_abs_diff'] for r in recs):.2e}")
    print("C vs scikit-learn decisions:", sum(r["c_decisions_equal"] for r in recs), "/",
          sum(r["n_test"] for r in recs))
    print("float32 vs float64 windows, decisions equal:",
          sum(r["f32_vs_f64_decisions_equal"] for r in recs), "/", sum(r["n_test"] for r in recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
