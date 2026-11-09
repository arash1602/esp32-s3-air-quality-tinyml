"""Hardware-in-the-loop runs on the Sensa board (firmware: firmware/, image: models.bin).

    python tools/hil.py --port COM5 memory
    python tools/hil.py --port COM5 fidelity     # board vs reference on the test windows
    python tools/hil.py --port COM5 latency      # 1000-repetition timing of every model
    python tools/hil.py --port COM5 energy       # alternating busy / idle periods

Results go to hil_results/<command>.json. Needs: pip install pyserial numpy scikit-learn

fidelity: the test windows of the fidelity fold are streamed to the board (all of
them for the classical models, 500 spread over the fold for each network), and the board's score and decision are compared with the
reference: the NumPy model (networks, ref_models.py) or scikit-learn's own scores
saved by export_classical.py (WHO rule, logistic regression, boosted trees). The
board's average precision and F2 on those windows are recorded too.

latency: each model is timed on a real test window (tree paths depend on the data;
networks do not), --reps repetitions, minimum / median / 99th percentile / mean,
twice: with the weights read in place from flash (as deployed), and copied to RAM
when they fit (to separate computation from flash access).

energy: for each timing and classical model the board runs inference back to back
for --seconds, then idles for --seconds, and the script prints the start time of
each period. Log the board's current with your meter over the whole run; the script
records the inference count of each busy period. Energy per inference =
(mean busy power - mean idle power) x busy seconds / inference count. Write the meter
readings into hil_results/energy_meter.csv (columns: index, busy_mA, idle_mA, volts).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import ref_models as rm  # noqa: E402

WIN_BYTES = 24 * 5 * 4


class BoardStalled(Exception):
    """The board stopped answering after `done` windows; results so far attached."""
    def __init__(self, done, score, dec, us):
        super().__init__(f"board stalled after {done} windows")
        self.done, self.score, self.dec, self.us = done, score.copy(), dec.copy(), us.copy()


def predict_robust(board, windows, sec_per_window, idx, tries=5):
    """board.predict, resuming after a stall: the chip is reset over USB, the model
    selected again and the remaining windows sent (prediction has no state)."""
    parts, start = [], 0
    for attempt in range(tries + 1):
        try:
            s, d, u = board.predict(windows[start:], sec_per_window)
            parts.append((s, d, u))
            break
        except BoardStalled as e:
            parts.append((e.score, e.dec, e.us))
            start += e.done
            if attempt == tries:
                raise SystemExit(f"model {idx}: board kept stalling ({tries} resets); "
                                 "press RESET and rerun the same command (it resumes).")
            print(f"      board stalled at window {start} of {len(windows)}: "
                  "resetting it and continuing", flush=True)
            board.reconnect()
            board.select(idx)
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


class Board:
    def __init__(self, port, baud=921600, timeout=120):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.resends = 0
        self._open()

    def reconnect(self):
        """Reset the chip through the USB serial/JTAG port (as esptool does) and
        reconnect; the port disappears for a moment while the chip restarts."""
        try:
            self.s.dtr = False
            self.s.rts = True
            time.sleep(0.2)
            self.s.rts = False
        except Exception:
            pass
        try:
            self.s.close()
        except Exception:
            pass
        deadline = time.time() + 60
        while True:
            time.sleep(1.5)
            try:
                self._open()
                return
            except BaseException:
                if time.time() > deadline:
                    raise SystemExit("board did not come back after the reset: press RESET "
                                     "and rerun the same command (it resumes).")

    def _open(self):
        import serial
        port, baud, timeout = self.port, self.baud, self.timeout
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = port, baud, timeout
        s.dtr = False          # opening the ESP32-S3's USB port with DTR/RTS asserted
        s.rts = False          # can reset the chip
        s.open()
        self.s = s
        line = self._hello(6)
        if not line.startswith("AQ ") and not getattr(self, "_reset_tried", False):
            # board may be stuck in an earlier, interrupted command: reset it over
            # USB (as esptool does) and try once more
            self._reset_tried = True
            print("board did not answer; resetting it over USB ...", flush=True)
            try:
                s.dtr = False; s.rts = True; time.sleep(0.2); s.rts = False
            except Exception:
                pass
            s.close()
            for _ in range(20):
                time.sleep(1.0)
                try:
                    s.open(); break
                except Exception:
                    continue
            time.sleep(1.5)
            line = self._hello(10)
        s.timeout = timeout
        if not line.startswith("AQ "):
            raise SystemExit(f"no reply from the board ({line!r}): is aq_bench flashed, the models "
                             "image written, and no serial monitor open on this port?")
        _, n, mhz = line.split()
        self.n_models, self.cpu_mhz = int(n), int(mhz)

    def _hello(self, seconds):
        s, line = self.s, ""
        deadline = time.time() + seconds
        while time.time() < deadline:
            s.reset_input_buffer()
            s.write(b"I")
            s.timeout = 1.0
            line = self.readline()
            while line == "AQREADY":
                line = self.readline()
            if line.startswith("AQ "):
                break
        return line

    def readline(self):
        return self.s.readline().decode(errors="replace").strip()

    def select(self, idx, ram=False):
        self.s.write((b"R" if ram else b"S") + bytes([idx]))
        line = self.readline()
        if ram and line.startswith("ERR -3"):
            return None                      # does not fit in RAM
        if not line.startswith("OK"):
            raise SystemExit(f"model {idx}: {line}")
        _, _, arch, prec, n_params, nbytes = line.split()
        return {"arch_id": int(arch), "precision_id": int(prec), "n_params": int(n_params),
                "blob_bytes": int(nbytes)}

    def predict(self, windows, sec_per_window=1.0, retries=4):
        """Board score (f64), decision and microseconds for every window, one 'Q'
        command per window. USB CDC on the ESP32-S3 occasionally loses a transfer;
        the board drops a window that arrives incomplete, so a window whose reply
        does not come is simply sent again (prediction has no state)."""
        n = len(windows)
        score, dec, us = np.empty(n), np.empty(n, bool), np.empty(n, np.uint32)
        w = np.ascontiguousarray(windows, "<f4")
        old = self.s.timeout
        self.s.timeout = max(1.0, sec_per_window * 3 + 0.5)
        try:
            for i in range(n):
                msg = b"Q" + w[i].tobytes()
                for attempt in range(retries + 1):
                    self.s.write(msg)
                    data = self.s.read(13)
                    if len(data) == 13:
                        break
                    self.resends += 1
                    time.sleep(0.4)                 # board drops a partial window after 250 ms
                    self.s.reset_input_buffer()     # and any partial reply
                else:
                    raise BoardStalled(i, score[:i], dec[:i], us[:i])
                score[i], dec[i], us[i] = struct.unpack("<dBI", data)
        finally:
            self.s.timeout = old
        return score, dec, us

    def set_window(self, window):
        self.s.write(b"X" + np.ascontiguousarray(window, "<f4").tobytes())
        if self.readline() != "X":
            raise SystemExit("board did not accept the window")

    def text(self, cmd, arg=None, timeout=None):
        self.s.write(cmd + (struct.pack("<I", arg) if arg is not None else b""))
        if timeout is None:
            return self.readline()
        old, self.s.timeout = self.s.timeout, timeout
        try:
            return self.readline()
        finally:
            self.s.timeout = old

    def bench(self, reps, budget_s):
        """Timing with as many repetitions as fit in about budget_s seconds (at most
        reps, at least 10): a 3-repetition probe first, so slow models do not stall."""
        probe = self.text(b"B", 3, timeout=600).split()
        if not probe or probe[0] != "B":
            raise SystemExit(f"board did not answer the timing probe: {probe}")
        t = max(float(probe[5]), 1.0) * 1e-6
        n = int(max(10, min(reps, budget_s / t)))
        line = self.text(b"B", n, timeout=n * t * 2 + 30).split()
        if not line or line[0] != "B":
            raise SystemExit(f"board did not answer: {line}")
        return line


def fold_dir(e, args):
    root = args.classical if e["kind"] == "classical" or e["precision"] == "f64" else args.export
    return os.path.join(root, f"h{e['horizon']}", e["design"], e["fold"])


def ap_f2(y, s, d):
    from sklearn.metrics import average_precision_score
    t = y > 0
    tp, fp, fn = int((d & t).sum()), int((d & ~t).sum()), int((~d & t).sum())
    den = 5 * tp + 4 * fn + fp
    return (float(average_precision_score(y, s)) if 0 < t.sum() < len(t) else float("nan"),
            5 * tp / den if den else 0.0)


def reference(e, d, sel):
    """Reference score and decision of an index entry on windows ``sel``."""
    if e["kind"] == "classical" or e["precision"] == "f64":
        ref = np.fromfile(os.path.join(d, f"{e['arch']}_ref.f64"), "<f8")[sel]
        dec = np.fromfile(os.path.join(d, f"{e['arch']}_ref_dec.u8"), np.uint8)[sel].astype(bool)
        return ref, dec
    meta = json.load(open(os.path.join(d, f"{e['arch']}.json")))
    win = np.fromfile(os.path.join(d, "test_windows.f32"), "<f4").reshape(-1, 24, 5)[sel]
    sd = dict(np.load(os.path.join(d, f"{e['arch']}.npz")))
    ref_sd = rm.int8_state_dict(sd) if e["precision"] == "int8" else rm.folded_state_dict(sd)
    ref = np.asarray(rm.FORWARD[e["arch"]](rm.scale(win, meta["scaler"]), ref_sd, meta["hp"]),
                     np.float32).astype(np.float64)
    return ref, ref > meta["threshold_logit"]


def label(e):
    return f"[{e['index']:2d}] h={e['horizon']:2d} {e['design']:14s} {e['arch']:6s} {e['precision']:4s}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fidelity", "latency", "energy", "memory"])
    ap.add_argument("--port", required=True)
    ap.add_argument("--index", default="models_index.json")
    ap.add_argument("--export", default=os.path.join(ROOT, "export_out"))
    ap.add_argument("--classical", default=os.path.join(ROOT, "export_classical"))
    ap.add_argument("--max-windows", type=int, default=0,
                    help="classical models: windows per model, 0 = every test window")
    ap.add_argument("--net-windows", type=int, default=500,
                    help="networks: windows per model, spread over the fold (0 = all; the "
                         "PC checks every window, the board run confirms the PC on a sample)")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--models", type=int, nargs="*", default=None,
                    help="only these model indices (default: all)")
    ap.add_argument("--fid-budget", type=float, default=240.0,
                    help="fidelity: seconds of board time per model at most (at least 50 windows)")
    ap.add_argument("--budget", type=float, default=30.0,
                    help="latency: seconds per model at most (fewer repetitions for slow models)")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--out", default="hil_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.index) as f:
        idx = json.load(f)
    b = Board(args.port)
    if b.n_models != len(idx):
        raise SystemExit(f"board has {b.n_models} models, index lists {len(idx)}: rewrite models.bin")
    results = []
    path = os.path.join(args.out, f"{args.command}.json")
    done = set()
    if args.command in ("fidelity", "latency", "energy") and os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        done = {r["index"] for r in results}
        print(f"resuming: {len(done)} models already in {path}")

    def save():
        with open(path, "w") as f:
            json.dump(results, f, indent=1)

    print(f"board: {b.n_models} models, CPU {b.cpu_mhz} MHz")

    if args.command == "fidelity":
        for e in idx:
            if e["kind"] not in ("fidelity", "classical"):
                continue
            if args.models and e["index"] not in args.models:
                continue
            if e["index"] in done:
                continue
            d = fold_dir(e, args)
            win = np.fromfile(os.path.join(d, "test_windows.f32"), "<f4").reshape(-1, 24, 5)
            y = np.fromfile(os.path.join(d, "test_y.u8"), np.uint8)
            info = b.select(e["index"])
            resends0 = b.resends
            # time one window first: slow models get fewer windows (--budget seconds)
            _, _, us0 = predict_robust(b, win[len(win) // 2: len(win) // 2 + 1], 30.0, e["index"])
            t1 = max(float(us0[0]) * 1e-6, 1e-5)
            print(f"{label(e)} one window: {t1 * 1000:.1f} ms", flush=True)
            cap = args.max_windows if e["precision"] == "f64" else args.net_windows
            n = len(win) if cap <= 0 else min(cap, len(win))
            n = int(max(min(n, 50), min(n, args.fid_budget / t1)))
            sel = np.linspace(0, len(win) - 1, n).astype(int)
            ref, ref_dec = reference(e, d, sel)
            score, dec, us = predict_robust(b, win[sel], t1, e["index"])
            ap_b, f2_b = ap_f2(y[sel], score, dec)
            ap_r, f2_r = ap_f2(y[sel], ref, ref_dec)
            r = {**e, **info, "n": int(n),
                 "max_abs_diff": float(np.max(np.abs(score - ref))),
                 "max_rel_diff": float(np.max(np.abs(score - ref) / np.maximum(1.0, np.abs(ref)))),
                 "decisions_equal": int((dec == ref_dec).sum()),
                 "ap_board": ap_b, "f2_board": f2_b, "ap_reference": ap_r, "f2_reference": f2_r,
                 "median_us": float(np.median(us)), "max_us": int(us.max()),
                 "usb_resends": int(b.resends - resends0)}
            results.append(r)
            print(f"{label(e)} max |board - reference| {r['max_abs_diff']:.1e} "
                  f"(relative {r['max_rel_diff']:.1e})  decisions "
                  f"{r['decisions_equal']}/{n}  AP {ap_b:.3f} (ref {ap_r:.3f})  "
                  f"median {r['median_us']:.0f} us"
                  + (f"  ({r['usb_resends']} windows resent)" if r["usb_resends"] else ""))
            save()
    elif args.command == "latency":
        for e in idx:
            if e["kind"] not in ("timing", "classical"):
                continue
            if args.models and e["index"] not in args.models:
                continue
            if e["index"] in done:
                continue
            window = None
            if e.get("fold"):
                p = os.path.join(fold_dir(e, args), "test_windows.f32")
                if os.path.exists(p):
                    w = np.fromfile(p, "<f4", count=24 * 5 * 1000).reshape(-1, 24, 5)
                    window = w[len(w) // 2]
            for place in ("flash", "ram"):
                info = b.select(e["index"], ram=place == "ram")
                if info is None:
                    print(f"{label(e)} {place:5s} does not fit in RAM")
                    continue
                if window is not None:
                    b.set_window(window)
                line = b.bench(args.reps, args.budget)
                r = {**e, **info, "weights_in": place, "reps": int(line[1]), "min_us": int(line[2]),
                     "median_us": int(line[3]), "p99_us": int(line[4]), "mean_us": float(line[5])}
                results.append(r)
                print(f"{label(e)} {place:5s} median {r['median_us'] / 1000:.3f} ms  "
                      f"p99 {r['p99_us'] / 1000:.3f} ms  {info['n_params']} params, "
                      f"{info['blob_bytes']} bytes")
                save()
    elif args.command == "energy":
        print("Start logging current now. Each model: busy period, then idle period.")
        for e in idx:
            if e["kind"] not in ("timing", "classical"):
                continue
            if args.models and e["index"] not in args.models:
                continue
            if e["index"] in done:
                continue
            for attempt in range(3):
                try:
                    b.select(e["index"])
                    break
                except SystemExit as err:
                    print(f"      {err}; retrying" + (" after a reset" if attempt else ""),
                          flush=True)
                    if attempt:
                        b.reconnect()
                    time.sleep(1.0)
            else:
                print(f"{label(e)}: skipped (model could not be selected)", flush=True)
                continue
            t_busy = time.time()
            print(f"{label(e)}: BUSY from {time.strftime('%H:%M:%S')}", flush=True)
            count = int(b.text(b"L", args.seconds).split()[1])
            t_idle = time.time()
            print(f"      IDLE from {time.strftime('%H:%M:%S')}", flush=True)
            b.text(b"Z", args.seconds)
            results.append({**e, "busy_start": t_busy, "idle_start": t_idle,
                            "seconds": args.seconds, "inferences": count})
            save()
    else:
        f = b.text(b"M").split()
        results.append({"free_heap": int(f[1]), "min_free_heap": int(f[2]),
                        "stack_free": int(f[3]), "aq_static_bytes": int(f[4]),
                        "cpu_mhz": b.cpu_mhz})
        print(results[-1])
    save()
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
