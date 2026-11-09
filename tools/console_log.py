"""Save the board's USB console for a while (memory reports of firmware_sensa).

    python tools/console_log.py --port COM3 --seconds 180 -o mem_sensa-aq.txt

Prints every line as it arrives and writes it, with the PC time, to the output file.
At the end, summarises the "MEM ..." lines: lowest free heap, lowest minimum free heap,
lowest free loop stack, and the number of predictions seen.
"""
from __future__ import annotations

import argparse
import time

import serial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--quiet", action="store_true",
                    help="show only predictions and memory lines (long runs)")
    a = ap.parse_args()
    s = serial.Serial(a.port, 115200, timeout=0.5)
    s.dtr = True
    t_end = time.time() + a.seconds
    mem = []
    last_note = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
      try:
        while time.time() < t_end:
            line = s.readline().decode("utf-8", "replace").rstrip()
            if not line:
                continue
            if not a.quiet or not line.startswith(("[SEN55]", "Classification")):
                print(line[:200], flush=True)
            elif time.time() - last_note > 300:
                last_note = time.time()
                print(f"{time.strftime('%H:%M')}  still logging; {(t_end - time.time()) / 3600:.1f} h left", flush=True)
            f.write(f"{time.time():.3f} {line}\n")
            f.flush()
            if line.startswith("MEM "):
                w = line.split()
                mem.append({w[i]: int(w[i + 1]) for i in range(1, len(w) - 1, 2)})
      except KeyboardInterrupt:
        print("stopped with Ctrl+C; the log so far is saved")
      if mem:
        summary = {k: min(m[k] for m in mem if k in m) for k in mem[0]}
        summary["aq_predictions"] = max(m.get("aq_predictions", 0) for m in mem)
        f.write(f"SUMMARY {summary}\n")
        print("SUMMARY (lowest values):", summary)
      else:
        print("no MEM lines received: is the right firmware on the board?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
