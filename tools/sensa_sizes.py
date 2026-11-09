"""Build firmware_sensa with and without the hourly warning and save the size reports.

    python tools/sensa_sizes.py          (writes sensa_sizes.json and prints the table)

Runs "pio run -e sensa-base" and "-e sensa-aq" and reads the RAM and flash figures
that PlatformIO prints at the end of each build.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"^(RAM|Flash):.*?used (\d+) bytes from (\d+) bytes", re.M)


def main() -> int:
    res = {}
    for env in ("sensa-base", "sensa-aq", "sensa-aq-generic"):
        p = subprocess.run([sys.executable, "-m", "platformio", "run", "-e", env],
                           cwd=os.path.join(ROOT, "firmware_sensa"),
                           capture_output=True, text=True)
        out = p.stdout + p.stderr
        open(os.path.join(ROOT, f"build_{env}.log"), "w").write(out)
        if p.returncode:
            print(out[-3000:])
            raise SystemExit(f"build of {env} failed (full log: build_{env}.log)")
        res[env] = {m[0]: {"used": int(m[1]), "total": int(m[2])} for m in PAT.findall(out)}
        print(env, res[env], flush=True)
    for k in ("RAM", "Flash"):
        res[f"added_{k}"] = res["sensa-aq"][k]["used"] - res["sensa-base"][k]["used"]
        res[f"added_generic_{k}"] = res["sensa-aq-generic"][k]["used"] - res["sensa-base"][k]["used"]
    json.dump(res, open(os.path.join(ROOT, "sensa_sizes.json"), "w"), indent=1)
    print(f"the warning adds {res['added_Flash']} bytes of flash and {res['added_RAM']} bytes "
          f"of static RAM (library sized for the model; {res['added_generic_RAM']} bytes with "
          "the default limits); saved sensa_sizes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
