"""Pack model blobs into the flash image for the board's "models" partition.

    python tools/build_models_image.py --export export_out --code-dir <paper-1 code> -o models.bin

Two kinds of models go into the image, listed in models_index.json:

  timing models   the configuration paper 1 reports for each horizon, design and
                  architecture (its pooled configuration), in float32 and int8. If some
                  fold's trained model has exactly that configuration it is used;
                  otherwise the configuration is built with random weights, which does
                  not change speed, memory or energy.
  fidelity models one trained model per horizon, design and architecture, from a
                  fold chosen in advance (Antwerp-b1 within city, heldout-Zagreb across
                  cities), in both precisions; their test windows are streamed to the
                  board by tools/hil.py and the board's logits are compared with the
                  NumPy reference.
  classical       the WHO rule, logistic regression and gradient-boosted trees of the
                  same folds (from export_classical.py; run on the board in double
                  precision), used for fidelity, latency and energy alike; plus the
                  largest GBT of any fold (most tree nodes) as the worst case for latency.

--flash 4 builds for a board with 4 MB of flash (3 MB models partition); if the image
does not fit, build it in two rounds with --kind timing and --kind fidelity.

Then write the image (port name as on your PC, e.g. COM5 or /dev/ttyACM0):

    python -m esptool --chip esp32s3 --port COM5 write_flash 0x110000 models.bin
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import blob  # noqa: E402
import ref_models as rm  # noqa: E402

ARCHS = ("GRU", "TCN", "FUSION")
DESIGNS = {"single-blocked": ("single", "Antwerp-b1"), "multi": ("multi", "heldout-Zagreb")}
PARTITION_BYTES = {8: 0x6F0000, 16: 0x6F0000, 4: 0x2F0000}
CLASSICAL = ("WHO", "LR", "GBT")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=os.path.join(os.path.dirname(HERE), "export_out"))
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("-o", "--out", default="models.bin")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    ap.add_argument("--classical", default=os.path.join(os.path.dirname(HERE), "export_classical"))
    ap.add_argument("--flash", type=int, choices=sorted(PARTITION_BYTES), default=8,
                    help="board flash size in MB (esptool flash_id reports it)")
    ap.add_argument("--kind", choices=["all", "timing", "fidelity"], default="all",
                    help="build only one kind of model if both do not fit in the partition "
                         "(classical models go with fidelity)")
    args = ap.parse_args()
    partition = PARTITION_BYTES[args.flash]
    rng = np.random.default_rng(0)
    entries, blobs = [], []

    def add(kind, h, design, arch, prec, data, source):
        if args.kind != "all" and kind != args.kind and not (
                kind == "classical" and args.kind == "fidelity"):
            return
        fold = source.split(":", 1)[1].split()[0] if source.startswith("trained:") else None
        entries.append({"index": len(blobs), "kind": kind, "horizon": h, "design": design,
                        "arch": arch, "precision": prec, "bytes": len(data), "source": source,
                        "fold": fold})
        blobs.append(data)

    largest = None
    summ = os.path.join(args.classical, "summary.json")
    if os.path.exists(summ):
        with open(summ) as f:
            gbts = [r for r in json.load(f).values()
                    if r["arch"] == "GBT" and r["horizon"] in args.horizons]
        if gbts:
            largest = max(gbts, key=lambda r: r["n_params"])

    for h in args.horizons:
        for design, (short, fid_fold) in DESIGNS.items():
            for arch in ARCHS:
                with open(os.path.join(args.code_dir, "tuning_results", f"h{h}",
                                       f"best_{arch}_{short}_ctx24_blocked_logz.json")) as f:
                    pooled = json.load(f)["hyperparameters"]
                ddir = os.path.join(args.export, f"h{h}", design)
                match = None
                for fold in sorted(os.listdir(ddir)) if os.path.isdir(ddir) else []:
                    meta = os.path.join(ddir, fold, f"{arch}.json")
                    if os.path.exists(meta):
                        with open(meta) as f:
                            m = json.load(f)
                        if m["hp"] == pooled:
                            match = (fold, m)
                            break
                for prec in ("f32", "int8"):
                    if match:
                        fold, _ = match
                        with open(os.path.join(ddir, fold, f"{arch}_{prec}.aqm"), "rb") as f:
                            add("timing", h, design, arch, prec, f.read(), f"trained:{fold}")
                    else:
                        sd = rm.random_state_dict(arch, pooled, rng)
                        sc = {"feature_mean": [0.0] * 5, "feature_std": [1.0] * 5, "log_channels": [3, 4]}
                        add("timing", h, design, arch, prec,
                            blob.blob_bytes(arch, pooled, sd, sc, 0.0, precision=prec), "random")
                for prec in ("f32", "int8"):
                    p = os.path.join(ddir, fid_fold, f"{arch}_{prec}.aqm")
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            add("fidelity", h, design, arch, prec, f.read(), f"trained:{fid_fold}")
            for arch in CLASSICAL:
                p = os.path.join(args.classical, f"h{h}", design, fid_fold, f"{arch}.aqm")
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        add("classical", h, design, arch, "f64", f.read(), f"trained:{fid_fold}")
    if largest:
        p = os.path.join(args.classical, f"h{largest['horizon']}", largest["design"],
                         largest["outer"], "GBT.aqm")
        with open(p, "rb") as f:
            add("timing", largest["horizon"], largest["design"], "GBT", "f64", f.read(),
                f"trained:{largest['outer']} (largest GBT, {largest['n_params']} nodes)")

    header = 8 + 8 * len(blobs)
    off = (header + 15) & ~15
    index, body = [], bytearray()
    for b in blobs:
        index.append((off + len(body), len(b)))
        body += b + b"\0" * ((-len(b)) % 16)
    image = b"AQIX" + struct.pack("<I", len(blobs))
    image += b"".join(struct.pack("<II", o, n) for o, n in index)
    image += b"\0" * (off - len(image)) + bytes(body)
    if len(image) > partition:
        print(f"image is {len(image) / 1e6:.2f} MB, larger than the {partition / 1e6:.2f} MB "
              "partition: build the timing and fidelity models as two images with --kind, "
              "or fewer horizons at a time (--horizons)", file=sys.stderr)
        return 1
    with open(args.out, "wb") as f:
        f.write(image)
    idx_path = os.path.splitext(args.out)[0] + "_index.json"
    with open(idx_path, "w") as f:
        json.dump(entries, f, indent=1)
    print(f"{len(blobs)} models, {len(image) / 1e6:.2f} MB -> {args.out}; index -> {idx_path}")
    print(f"flash with: python -m esptool --chip esp32s3 --port <PORT> write_flash 0x110000 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
