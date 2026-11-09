# On-device measurements on the ESP32-S3

Needs: the ESP32-S3 development board (recommended) or the Sensa
PCB, the same PlatformIO setup used for the Sensa firmware (VS Code + PlatformIO, or
`pip install platformio`), and Python with `pip install pyserial numpy scikit-learn
esptool`. Everything else (trained models, model images, reference results) is in this
package. For the energy run, a USB power meter (dev board) or a current meter in series
with a 3.3 V lab supply (PCB). About half a day.

## 0. Which board

**Use the ESP32-S3 development board.** It has the same ESP32-S3 chip as the Sensa PCB,
works from one USB cable, and none of the measurements needs the sensor, the motor or
the buzzer. The first-revision Sensa PCB's power stage failed (see the paper), so the
development board was used for all measurements.

1. Plug a USB-C cable into the dev board's **native USB** port (often labelled "USB";
   the other one, labelled "UART" or "COM", goes through a USB-serial chip). If the
   ports are not labelled, use either; if step 3 says "no reply from the board", move
   the cable to the other port and press RESET.
2. Nothing else needs to be connected; take any jumper wires off the board's pins.
3. Read the flash size:

   ```
   python -m esptool --chip esp32s3 --port COM5 flash_id
   ```

   `COM5` is an example (Device Manager shows the port; on Linux/macOS it looks like
   `/dev/ttyACM0` or `/dev/cu.usbmodem*`). The line `Detected flash size:` decides the
   build name in step 1: 8 MB -> `sensa-8mb`, 16 MB -> `sensa-16mb`, 4 MB ->
   `sensa-4mb`. With 8 or 16 MB, the prebuilt model images below are used as they are.
   Note the module name printed on the board's metal can (e.g. ESP32-S3-WROOM-1-N16R8)
   for the paper.

If the Sensa PCB is used instead: it carries an ESP32-S3-WROOM-1-N8 (`sensa-8mb`). Do
**not** connect the LiPo battery. Power it as the team did: a lab supply at 3.30 V
(current limit about 500 mA) on the output node of the failed 3.3 V regulator, USB-C 5 V
routed to the sensor rail, and the USB-C cable to the PC for data.

## 1. Build and flash the benchmark firmware

```
cd firmware
pio run -e sensa-8mb -t upload --upload-port COM5      (or sensa-16mb, from step 0)
```

If the upload does not start, hold BOOT, tap RESET, release BOOT, and run it again.

## 2. Round 1: timing models

All the models do not fit in the flash at once (9.7 MB), so the measurements run in
two rounds with two model images. Both images are already built and included
(`models_timing.bin`, `models_fidelity.bin`, with their `_index.json` files), together
with the trained models they came from (`export_out`, `export_classical`), so nothing
needs the dataset or the paper-1 code. Close any serial monitor first (including
PlatformIO's). From the `edge` folder:

```
python -m esptool --chip esp32s3 --port COM5 write_flash 0x110000 models_timing.bin
```

Press RESET on the board once, then:

```
python tools/hil.py --port COM5 --index models_timing_index.json --out hil_results_timing memory
python tools/hil.py --port COM5 --index models_timing_index.json --out hil_results_timing latency
```

Then the energy run of this round (step 4) with
`--index models_timing_index.json --out hil_results_timing`.

## 3. Round 2: fidelity and classical models

```
python -m esptool --chip esp32s3 --port COM5 write_flash 0x110000 models_fidelity.bin
```

Press RESET, then:

```
python tools/hil.py --port COM5 --index models_fidelity_index.json --out hil_results_fidelity fidelity
python tools/hil.py --port COM5 --index models_fidelity_index.json --out hil_results_fidelity latency
```

and the energy run (step 4) with `--index models_fidelity_index.json --out
hil_results_fidelity` (it covers the WHO rule, logistic regression and boosted trees).

`fidelity` streams test windows through the board (all of them for the classical
models, 500 per network). Expected: board-vs-reference difference of 1e-4 or smaller
for the networks (float32 rounding), 0 for the WHO rule and the boosted trees, below
1e-13 for logistic regression, and all decisions equal. It takes about 10-30 minutes,
mostly for the GRUs.

`latency` times every model twice: weights read from flash, as deployed, and copied to
RAM when they fit (the large GRUs do not; that is expected and recorded).

## 4. Energy

**Dev board:** put a USB power meter (a USB-C inline meter that shows current to
0.001 A or better, or logs it) between the PC and the board's USB cable. Nothing else
may be plugged into the board.

**Sensa PCB:** put a current meter in series with the 3.3 V lab supply.

Then run (once per round, with that round's index and output folder):

```
python tools/hil.py --port COM5 --index <index of the round> --out <folder of the round> --seconds 20 energy
```

For each model the script prints when the busy period and the idle period start. Note
the average current of each period (or read the meter in the middle of each period;
use `--seconds 30` for more time) and write them into `energy_meter.csv` in that round's
folder, with the voltage the meter shows:

```
index,busy_mA,idle_mA,volts
0,58.1,31.4,5.05
...
```

The analysis uses the difference between busy and idle current, which removes the
board's constant draw (LEDs, regulator, USB-serial chip).

## 5. Send back

Zip the two folders `hil_results_timing` and `hil_results_fidelity` and the two
`models_*_index.json` files, plus one photo of the
measurement set-up, the meter's model name, which board was used, the module name on
its metal can, and the flash size from step 0.
