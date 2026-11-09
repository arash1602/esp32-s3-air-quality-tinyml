"""Record current and the D0 marker from a Nordic Power Profiler Kit II (PPK2) while
the board runs the stand-alone energy firmware (env "ppk2", bench/aq_autorun.c).

    python -m pip install ppk2-api numpy
    python tools/ppk2_log.py -o ppk_timing          (writes ppk_timing.cur and .dig)

The PPK2 supplies the board (source-meter mode, 3.3 V by default) through VOUT and
records 100,000 samples per second. Close nRF Connect's Power Profiler first: only one
program can use the PPK2. The script switches the board on, records until the
end of the run (D0 low for 10 s after the end marker), switches the board off and stops. Ctrl+C
stops it early; the data so far are kept.

Output: <out>.cur  float32 current in microamperes, one value per 10 us
        <out>.dig  uint8 logic byte per sample (bit 0 = D0)
        <out>.json voltage, sample rate, sample count, wall-clock start and end
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

RATE = 100_000


def find_port():
    """The PPK2's serial port: by USB id (Nordic 1915:C00A), since Windows names it
    plain "USB Serial Device", which ppk2-api's own search does not recognise."""
    from serial.tools import list_ports
    ports = [p.device for p in list_ports.comports() if p.vid == 0x1915 and p.pid == 0xC00A]
    if not ports:
        from ppk2_api.ppk2_api import PPK2_API
        for d in PPK2_API.list_devices() or []:
            ports.append(d[0] if isinstance(d, (tuple, list)) else d)
    if not ports:
        raise SystemExit("No PPK2 found: connect its USB DATA/POWER port, switch it ON, and "
                         "close nRF Connect's Power Profiler app.")
    return sorted(ports)[0]


def open_ppk2(port):
    from ppk2_api import ppk2_api as api
    cls = getattr(api, "PPK2_MP", None)          # reads in a separate process if available
    if cls is not None:
        try:
            return cls(port, buffer_max_size_seconds=5, buffer_chunk_seconds=0.02,
                       timeout=1, write_timeout=1, exclusive=True), True
        except TypeError:
            pass
    return api.PPK2_API(port, timeout=1, write_timeout=1, exclusive=True), False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True, help="output name without extension")
    ap.add_argument("--mv", type=int, default=3300, help="supply voltage in mV (default 3300)")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--minutes", type=float, default=0,
                    help="record exactly this many minutes (full-firmware runs); "
                         "otherwise stop at the end of the stand-alone energy run")
    a = ap.parse_args()

    port = find_port()
    ppk, mp = open_ppk2(port)
    print(f"PPK2 on {port}" + (" (multiprocess reader)" if mp else ""), flush=True)
    ppk.get_modifiers()
    ppk.set_source_voltage(a.mv)          # same order as ppk2-api's own example
    ppk.use_source_meter()

    fc = open(a.out + ".cur", "wb")
    fd = open(a.out + ".dig", "wb")
    n = 0
    low_run = 0                                    # samples D0 has been low in a row
    seen_low = seen_activity = False
    checked_power = False
    t0 = time.time()
    ppk.toggle_DUT_power("ON")
    ppk.start_measuring()
    print(f"board powered at {a.mv} mV; recording (the run takes about 10-15 minutes)",
          flush=True)
    t_print, reason = t0 + 5, "time limit"
    try:
        while time.time() - t0 < a.max_minutes * 60:
            data = ppk.get_data()
            if not data:
                time.sleep(0.002)
                continue
            out = ppk.get_samples(data)
            if not isinstance(out, tuple) or len(out) < 2:
                raise SystemExit("this ppk2-api version returns no logic data; "
                                 "run: python -m pip install -U ppk2-api")
            cur = np.asarray(out[0], dtype=np.float32)
            dig = np.asarray(out[1], dtype=np.uint8)
            k = min(len(cur), len(dig))
            cur, dig = cur[:k], dig[:k]
            fc.write(cur.tobytes()); fd.write(dig.tobytes())
            n += k
            d0 = dig & 1
            if (d0 == 0).any():
                seen_low = True
            if seen_low and d0.any():
                seen_activity = True                # D0 went low, then high: firmware runs
            if not checked_power and n >= 2 * RATE:
                checked_power = True
                if float(np.mean(cur)) < 1000:      # under 1 mA: the board is not powered
                    print(f"The board draws only {float(np.mean(cur)):.1f} uA: it is not "
                          "powered. Check that the red wire goes from the PPK2 pin VOUT "
                          "to the ESP32 3V3 row.", flush=True)
                    reason = "board not powered"
                    break
            # end of the run: after the end marker (D0 high 3 s) the firmware keeps D0
            # low for good; nowhere else is D0 low for 10 s (longest idle phase 4.5 s).
            # A long D0-high alone is not the end: one prediction can take seconds.
            if k:
                if not d0.any():
                    low_run += k
                else:
                    low_run = k - 1 - int(np.flatnonzero(d0)[-1])
                if not a.minutes and seen_activity and low_run >= 10 * RATE:
                    reason = "end of run"
                    break
                if a.minutes and n >= a.minutes * 60 * RATE:
                    reason = "end of run"
                    break
            now = time.time()
            if now >= t_print:
                el = now - t0
                print(f"{time.strftime('%H:%M:%S')}  {n / RATE:7.1f} s recorded in {el:6.1f} s"
                      f"  ({n / el / 1000:5.1f} k samples/s)  now {float(cur.mean()) / 1000:7.2f} mA"
                      f"  D0 {'high' if d0[-1] else 'low'}", flush=True)
                if el > 20 and n / el < 0.9 * RATE:
                    print("  WARNING: the PC is not keeping up with the PPK2 "
                          "(close other programs)", flush=True)
                t_print = now + 5
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    t1 = time.time()
    try:
        ppk.toggle_DUT_power("OFF")
        ppk.stop_measuring()
    except Exception:
        pass
    fc.close(); fd.close()
    meta = {"mv": a.mv, "rate": RATE, "samples": n, "start": t0, "end": t1, "stop": reason}
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    print(f"stopped ({reason}); {n} samples = {n / RATE:.1f} s; saved {a.out}.cur/.dig/.json")
    if not seen_activity and not a.minutes:
        print("No marker activity on D0 was recorded: check the D0 wire (to ESP32 pin 4), "
              "the logic GND wire, and that the energy firmware is on the board.")
    return 0 if reason == "end of run" else 1


if __name__ == "__main__":
    sys.exit(main())
