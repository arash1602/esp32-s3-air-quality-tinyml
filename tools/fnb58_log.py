"""Log voltage and current from a FNIRSI FNB58 USB meter to CSV (100 samples/s).

    pip install hidapi
    python tools/fnb58_log.py -o energy_log.csv          (Ctrl+C to stop)

Connect the meter's own "PC" micro-USB port to the computer with a second cable; the
measured path (USB-A plug -> USB-A socket / Type-C OUT) is separate. Works on Windows,
macOS and Linux through the operating system's HID driver, so no driver install.

Protocol from Witold Baryluk's fnirsi-usb-power-data-logger (MIT licence,
https://github.com/baryluk/fnirsi-usb-power-data-logger): 64-byte HID reports; a
data report starts 0xAA 0x04 and holds four samples of 15 bytes (voltage and current
as little-endian u32 in units of 10 uV / 10 uA), one every 10 ms.

Output columns: epoch_s (PC clock, same clock as tools/hil.py), voltage_V, current_A.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

VID, PID = 0x2E3C, 0x5558          # FNB58 (and FNB48S uses PID 0x0049)
PIDS = (0x5558, 0x0049)
SPS = 100


def _is_meter(d):
    txt = f"{d.get('manufacturer_string') or ''} {d.get('product_string') or ''}".lower()
    return (d["vendor_id"] == VID or "fnirsi" in txt or "fnb" in txt
            or (d["vendor_id"] == 0x0483 and d["product_id"] in (0x003A, 0x003B)))


def list_hid():
    import hid
    print("HID devices seen by this computer:")
    for d in hid.enumerate():
        print(f"  {d['vendor_id']:04x}:{d['product_id']:04x}  "
              f"{d.get('manufacturer_string') or ''} | {d.get('product_string') or ''}  "
              f"(interface {d.get('interface_number')}, usage page {d.get('usage_page', 0):04x})")


def open_meter():
    import hid
    devs = [d for d in hid.enumerate() if _is_meter(d)]
    if not devs:
        list_hid()
        raise SystemExit("FNB58 not found: connect its PC (micro-USB) port to this computer "
                         "with a data cable (some micro-USB cables carry power only), and "
                         "close the FNIRSI PC software if it is open.")
    last = None
    for d in devs:                      # some systems list more than one HID interface
        try:
            h = hid.device()
            h.open_path(d["path"])
            print(f"meter: {d['vendor_id']:04x}:{d['product_id']:04x} "
                  f"{d.get('product_string') or ''}", flush=True)
            return h
        except OSError as e:
            last = e
    raise SystemExit(f"could not open the FNB58 ({last})")


def send(h, payload: bytes):
    h.write(b"\x00" + payload)          # leading 0: no report ID


def start(h):
    send(h, b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
    send(h, b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    send(h, b"\xaa\x82" + b"\x00" * 61 + b"\x96")


def keepalive(h):
    send(h, b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


def decode(data, t_recv):
    """Four (t, V, A) samples from one 64-byte data report, or [] for other reports."""
    if len(data) < 62 or data[0] != 0xAA or data[1] != 0x04:
        return []
    out = []
    for i in range(4):
        o = 2 + 15 * i
        v = int.from_bytes(bytes(data[o:o + 4]), "little") / 100000
        a = int.from_bytes(bytes(data[o + 4:o + 8]), "little") / 100000
        out.append((t_recv - (3 - i) / SPS, v, a))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="energy_log.csv")
    args = ap.parse_args()
    h = open_meter()
    start(h)
    n, t_keep, t_print = 0, time.time() + 1.0, time.time() + 5.0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch_s", "voltage_V", "current_A"])
        print(f"logging to {args.out}; press Ctrl+C to stop", flush=True)
        try:
            while True:
                data = h.read(64, 2000)
                now = time.time()
                for t, v, a in decode(data, now):
                    w.writerow([f"{t:.3f}", f"{v:.5f}", f"{a:.5f}"])
                    n += 1
                    last = (v, a)
                if now >= t_keep:
                    keepalive(h)
                    t_keep = now + 1.0
                if now >= t_print and n:
                    f.flush()
                    print(f"{time.strftime('%H:%M:%S')}  {n} samples  last {last[0]:.4f} V "
                          f"{last[1] * 1000:.2f} mA", flush=True)
                    t_print = now + 5.0
                elif now >= t_print:
                    print("no data from the meter yet ...", flush=True)
                    t_print = now + 5.0
        except KeyboardInterrupt:
            pass
    h.close()
    print(f"saved {n} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
