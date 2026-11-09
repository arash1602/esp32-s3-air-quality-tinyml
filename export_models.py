"""Retrain paper 1's selected networks, save them, and package them for the ESP32-S3.

Run on a PC with PyTorch and the SensEURCity dataset (the same setup as the seed run):

    python export_models.py --code-dir <path to the paper-1 code folder> --smoke   # 2 min test
    python export_models.py --code-dir <path to the paper-1 code folder>           # full run

For every horizon (1, 12, 24 h), design (within city, leave one city out), outer
fold and architecture (GRU, TCN, FUSION), the configuration and decision threshold
selected in paper 1's sweep are refitted exactly as the sweep refitted them (seed 42,
deterministic kernels), and the script saves:

    export_out/<h>/<design>/<fold>/<ARCH>.npz       trained weights (PyTorch state_dict)
    export_out/<h>/<design>/<fold>/<ARCH>.json      settings, scaler, threshold, scores
    export_out/<h>/<design>/<fold>/<ARCH>_f32.aqm   device model, 32-bit weights
    export_out/<h>/<design>/<fold>/<ARCH>_int8.aqm  device model, 8-bit weights
    export_out/<h>/<design>/<fold>/<ARCH>_torch.f32 PyTorch test logits
    export_out/<h>/<design>/<fold>/test_windows.f32 raw test windows [N][24][5]
    export_out/<h>/<design>/<fold>/test_y.u8        labels
    export_out/summary.json                         every check below

Checks recorded for every model:
  * the refit reproduces paper 1's test average precision (same hardware tolerance
    as the seed run);
  * the NumPy reference (ref_models.py) reproduces PyTorch's test logits, and both are
    compared with a float64 computation of the same network (on a GPU, PyTorch's
    TF32 convolutions and matrix products are the least precise of the three);
  * average precision and F2 of the 8-bit model on the same windows.
Send back the whole export_out folder.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blob  # noqa: E402
import ref_models as rm  # noqa: E402

ARCHS = ("GRU", "TCN", "FUSION")
DESIGNS = ("single-blocked", "multi")


def ap_score(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def f2_at(y, s, thr):
    p = s > thr
    t = y > 0.5
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    d = 5 * tp + 4 * fn + fp
    return 5 * tp / d if d else 0.0


def exact_logits(arch, win, sd, hp, scaler):
    """The same network computed in float64 (reference for both NumPy and PyTorch)."""
    saved = rm.F32
    rm.F32 = np.float64
    try:
        sd64 = rm.folded_state_dict({k: np.asarray(v, np.float64) for k, v in sd.items()})
        return np.asarray(rm.FORWARD[arch](rm.scale(win.astype(np.float64), scaler), sd64, hp),
                          np.float64)
    finally:
        rm.F32 = saved


def raw_windows(ds):
    """Unscaled input windows of a dataset, in sample order (scaling happens on read)."""
    T = ds.context_size
    return np.stack([ds._series[w.series_idx][0][w.start:w.start + T] for w in ds._windows]
                    ).astype(np.float32)


def outer_records(results_dir, h, arch, experiment):
    out = {}
    with open(os.path.join(results_dir, f"h{h}", f"nested5_{arch}_{experiment}_ctx24_blocked.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            with contextlib.suppress(json.JSONDecodeError):
                r = json.loads(line)
                if r.get("kind") == "outer":
                    out[r["outer"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-dir", required=True, help="the paper-1 code folder")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "export_out"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="one fold, TCN only, 3 epochs")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.code_dir))
    import find_best_model as fbm
    from compare_models import materialize_outer, nested_plan
    from nested_worker import Job

    data_dir = args.data_dir or os.environ.get("AQM_DATA_DIR") or os.path.join(
        args.code_dir, "senseurcity_data_v03", "dataset")
    results_dir = os.path.join(args.code_dir, "tuning_results")
    archs, designs, horizons, epochs = ARCHS, DESIGNS, args.horizons, fbm.EPOCHS
    if args.smoke:
        archs, designs, horizons, epochs = ("TCN",), ("single-blocked",), [1], 3
        if args.out == os.path.join(HERE, "export_out"):    # never mix a 3-epoch test
            args.out = os.path.join(HERE, "export_out_smoke")  # into the real results
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    summary = {}
    if os.path.exists(summary_path) and not args.smoke:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    runner = fbm._Runner(args.workers, True)
    tmp_dir = tempfile.mkdtemp(prefix="aq_export_")
    t0 = time.time()
    try:
        for h in horizons:
            fbm.HORIZON = h
            common = {"context_size": fbm.CONTEXT_SIZE, "overlap": fbm.OVERLAP, "horizon": h}
            for design in designs:
                experiment, protocol = fbm.EXPERIMENTS[design]
                recs = {a: outer_records(results_dir, h, a, experiment) for a in archs}
                plan = nested_plan(data_dir, experiment=experiment, protocol=protocol,
                                   cities=list(fbm.CITIES), k_outer=fbm.K_OUTER,
                                   n_inner=fbm.N_INNER, block_hours=fbm.BLOCK_HOURS, **common)
                if args.smoke:
                    plan = plan[:1]
                for spec in plan:
                    fold_dir = os.path.join(args.out, f"h{h}", design, spec.outer_id)
                    todo = [a for a in archs if f"h{h}/{design}/{spec.outer_id}/{a}" not in summary]
                    if not todo:
                        continue
                    os.makedirs(fold_dir, exist_ok=True)
                    print(f"\nh={h} {design} {spec.outer_id}: materializing", flush=True)
                    bundle = materialize_outer(data_dir, spec, **common, scaling=fbm.DEFAULT_SCALING)
                    if bundle is None:
                        continue
                    win = raw_windows(bundle.test)
                    ytest = bundle.test.compute_labels().astype(np.uint8)
                    win.astype("<f4").tofile(os.path.join(fold_dir, "test_windows.f32"))
                    ytest.tofile(os.path.join(fold_dir, "test_y.u8"))
                    ff = fbm._export_outer(bundle, tmp_dir, f"h{h}_{design}_{spec.outer_id}")
                    del bundle
                    scaler = ff.scalers[ff.refit_fold]
                    scaler = {"feature_mean": scaler["feature_mean"], "feature_std": scaler["feature_std"],
                              "log_channels": scaler.get("log_channels", [3, 4])}
                    jobs = []
                    for a in todo:
                        r = recs[a][spec.outer_id]
                        hp = fbm._grid_for(a)[r["winner_combo"]]
                        jobs.append((a, hp, r, Job(arch=a, combo=r["winner_combo"], hp=hp,
                                                   fold=ff.refit_fold, dev_path=ff.dev_path,
                                                   test_path=ff.test_path,
                                                   input_channels=fbm.N_INPUT_CHANNELS, seed=fbm.SEED,
                                                   epochs=epochs, patience=fbm.PATIENCE,
                                                   batch_size=fbm.DEFAULT_BATCH,
                                                   eval_batch=fbm.EVAL_BATCH,
                                                   threshold=r["threshold"]["logit"],
                                                   threshold_beta=fbm.THRESHOLD_BETA)))
                    futures = [(a, hp, r, runner.submit(j)) for a, hp, r, j in jobs]
                    for a, hp, r, fut in futures:
                        _, res = fut.result()
                        sd = {k: np.asarray(v, dtype=np.float32) for k, v in res["best_state"].items()}
                        torch_logits = np.asarray(res["predictions"]["test_logits"], np.float32)
                        y = np.asarray(res["predictions"]["test_y"]).reshape(-1)
                        if not np.array_equal(y.astype(np.uint8), ytest):
                            raise SystemExit(f"{spec.outer_id} {a}: test labels differ from raw windows")
                        thr = float(r["threshold"]["logit"])
                        ref = rm.predict_logit(a, win, sd, hp, scaler)
                        ref64 = exact_logits(a, win, sd, hp, scaler)
                        sd8 = rm.int8_state_dict(sd)
                        ref8 = np.asarray(rm.FORWARD[a](rm.scale(win, scaler), sd8, hp), np.float32)
                        np.savez_compressed(os.path.join(fold_dir, f"{a}.npz"), **sd)
                        torch_logits.astype("<f4").tofile(os.path.join(fold_dir, f"{a}_torch.f32"))
                        nb_f32 = blob.write_blob(os.path.join(fold_dir, f"{a}_f32.aqm"), a, hp, sd,
                                                 scaler, thr, precision="f32")
                        nb_i8 = blob.write_blob(os.path.join(fold_dir, f"{a}_int8.aqm"), a, hp, sd,
                                                scaler, thr, precision="int8")
                        rec = {
                            "arch": a, "horizon": h, "design": design, "outer": spec.outer_id,
                            "hp": hp, "scaler": scaler, "threshold_logit": thr,
                            "n_test": int(len(y)), "prevalence": float(y.mean()),
                            "paper_ap": r["test"]["pr_auc"],
                            "torch_ap": ap_score(y, torch_logits),
                            "numpy_max_abs_diff": float(np.max(np.abs(ref - torch_logits))),
                            "numpy_decisions_equal": int(((ref > thr) == (torch_logits > thr)).sum()),
                            "numpy_vs_f64_max_abs_diff": float(np.max(np.abs(ref - ref64))),
                            "torch_vs_f64_max_abs_diff": float(np.max(np.abs(torch_logits - ref64))),
                            "int8_ap": ap_score(y, ref8),
                            "torch_f2": f2_at(y, torch_logits, thr),
                            "int8_f2": f2_at(y, ref8, thr),
                            "int8_decisions_equal": int(((ref8 > thr) == (torch_logits > thr)).sum()),
                            "blob_bytes_f32": nb_f32, "blob_bytes_int8": nb_i8,
                            "epochs": epochs, "secs": round(res["secs"], 1),
                        }
                        with open(os.path.join(fold_dir, f"{a}.json"), "w", encoding="utf-8") as f:
                            json.dump(rec, f, indent=1)
                        summary[f"h{h}/{design}/{spec.outer_id}/{a}"] = rec
                        with open(summary_path, "w", encoding="utf-8") as f:
                            json.dump(summary, f, indent=1)
                        print(f"  {a:6s} AP torch {rec['torch_ap']:.3f} (paper {rec['paper_ap']:.3f})  "
                              f"vs exact: NumPy {rec['numpy_vs_f64_max_abs_diff']:.1e}, "
                              f"torch {rec['torch_vs_f64_max_abs_diff']:.1e}  "
                              f"int8 AP {rec['int8_ap']:.3f}  decisions kept "
                              f"{rec['int8_decisions_equal']}/{rec['n_test']}", flush=True)
                    for p in (ff.dev_path, ff.test_path):
                        with contextlib.suppress(OSError):
                            os.remove(p)
    finally:
        runner.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    recs = list(summary.values())
    if recs:
        worst = max(r.get("numpy_vs_f64_max_abs_diff", r["numpy_max_abs_diff"]) for r in recs)
        wt = max(r.get("torch_vs_f64_max_abs_diff", r["numpy_max_abs_diff"]) for r in recs)
        dec = sum(r["numpy_decisions_equal"] for r in recs), sum(r["n_test"] for r in recs)
        dev = max(abs(r["torch_ap"] - r["paper_ap"]) for r in recs)
        print(f"\n{len(recs)} models in {(time.time() - t0) / 60:.1f} min -> {args.out}")
        print(f"NumPy reference vs exact (float64): largest logit difference {worst:.2e} "
              + ("(OK)" if worst < 1e-4 else "(CHECK: larger than expected)"))
        print(f"PyTorch vs exact: {wt:.2e} (GPU arithmetic, for information); "
              f"NumPy and PyTorch decisions equal on {dec[0]}/{dec[1]} windows")
        print(f"refit vs paper 1 average precision: largest difference {dev:.4f}")
    if args.smoke:
        print("Smoke test finished. If the NumPy difference says OK, run without --smoke.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
