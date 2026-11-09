# aq-edge: verified on-device WHO-exceedance classifiers for an ESP32-S3 wearable

Code and measurement data for the paper *"Verified On-Device Inference of WHO Air
Quality Guideline Exceedance Classifiers for an ESP32-S3 Wearable Monitor"*
(M. Ghamari, IEEE Sensors Journal, submitted).

The toolkit runs all six predictors of the WHO 24-hour PM2.5/PM10 exceedance benchmark
(WHO rule, logistic regression, gradient-boosted trees, GRU, TCN, CNN–GRU) on an
ESP32-S3 from one dependency-free C library and one model-file format, and contains
everything used to verify and measure them: the PC verification chain, the
hardware-in-the-loop protocol, the energy measurements (Nordic PPK2 and FNIRSI FNB58),
the full wearable firmware with the hourly warning, and the comparison with
TensorFlow Lite Micro.

## Contents

| Path | What it is |
| --- | --- |
| `c/aq_infer.[ch]` | portable C99 inference library for all six model families (no dynamic memory; float32 or 8-bit weights read in place from flash, flash-cache-aware kernels) |
| `c/aq_native.c` | runs a model file on a PC, for the fidelity checks |
| `blob.py` | writes a trained model in the library's binary format |
| `ref_models.py` | NumPy reference of the networks, input scaling, 8-bit weight quantization |
| `export_models.py` | retrains the selected networks with the benchmark code, checks NumPy against PyTorch, writes model files |
| `export_classical.py` | WHO rule, LR and GBT as in the benchmark; C against scikit-learn on every test fold |
| `integration/aq_window/` | 1 Hz SEN55 readings to the 24-hour window (hourly means, gaps, snapshot/restore across power cycles), with unit tests |
| `bench/` | board side of the hardware-in-the-loop protocol (`aq_bench`) and the stand-alone energy run with a GPIO marker (`aq_autorun`) |
| `firmware/` | PlatformIO/Arduino benchmark firmware for the ESP32-S3 (env `sensa-8mb`; env `ppk2` for the energy run) |
| `firmware_idf/` | the same benchmark for plain ESP-IDF |
| `firmware_sensa/` | the Sensa wearable firmware (SEN55, AQI pipeline, BLE) with the hourly warning added (envs `sensa-base`, `sensa-aq`, `sensa-aq-generic`, `sensa-ppk2`, `sensa-ppk2-o2`, `sensa-24h`) |
| `firmware_tflm/` | TensorFlow Lite Micro firmware (ESP-IDF, esp-tflite-micro with ESP-NN) with the same protocol and energy marker |
| `tools/hil.py` | fidelity, latency, energy and memory runs over USB |
| `tools/ppk2_log.py`, `tools/fnb58_log.py` | current logging with a Nordic PPK2 (100 kS/s, logic input) or a FNIRSI FNB58 |
| `tools/tflm_convert.py`, `tools/tflm_hil.py` | TensorFlow Lite conversion of the trained networks (checked against the reference on every test window) and their board runs |
| `tools/build_models_image.py`, `tools/blob_to_header.py` | model images for the board's flash partition; one model as a C header |
| `tools/console_log.py`, `tools/sensa_sizes.py` | memory reports and build sizes of the full firmware |
| `analysis/` | scripts that turn the board results into the paper's tables and figures |
| `tests/` | C vs NumPy, the board protocol on a PC (with injected transfer errors), the window logic, the energy-run marker decoding |
| `board_results/` | raw results of every board run reported in the paper (latency, fidelity, memory, FNB58 and PPK2 energy, full firmware, 29-hour SEN55 log, TensorFlow Lite Micro) |
| `models_*.bin`, `models_*_index.json` | the two model images flashed to the board and their index |
| `BOARD_GUIDE.md` | step-by-step guide for the board measurements |

The trained models (`export_out/`, `export_classical/`, about 215 MB) are published as a
release asset of this repository: download `trained_models.zip` from the Releases page
and unzip it into the repository folder.

## Verification chain

PyTorch = NumPy (`export_models.py`, every model) → NumPy = C (`tests/test_c_vs_numpy.py`)
→ scikit-learn = C (`export_classical.py`, every fold) → C on the board = reference
(`tools/hil.py fidelity`; rehearsed on a PC by `tests/test_hil_host.py`).

## Quick start

```
pip install numpy scikit-learn pyserial esptool platformio
gcc -O2 -std=c99 -o c/aq_native c/aq_native.c c/aq_infer.c -lm
python tests/test_c_vs_numpy.py
gcc -O2 -std=c11 -I c -I integration/aq_window -o test_aq_window \
    integration/test_aq_window.c integration/aq_window/aq_window.c c/aq_infer.c -lm && ./test_aq_window
```

Board measurements (ESP32-S3-DevKitC-1, 8 MB flash), from this folder:

```
cd firmware && pio run -e sensa-8mb -t upload && cd ..
python -m esptool --chip esp32s3 --port COM3 write_flash 0x110000 models_timing.bin
python tools/hil.py --port COM3 --index models_timing_index.json --out hil_results latency
```

Energy with a PPK2 (board powered at 3.3 V through its 3V3 pin, USB unplugged, PPK2
logic D0 on GPIO4): flash `firmware` env `ppk2`, then

```
python tools/ppk2_log.py -o ppk_timing
python analysis/ppk2_energy.py ppk_timing --index models_timing_index.json
```

TensorFlow Lite Micro comparison: `python tools/tflm_convert.py -o tflm_out` (needs
`pip install tensorflow`), build and flash `firmware_tflm`, write `tflm_out/tflm_a.bin`
to 0x110000, then `python tools/tflm_hil.py --port COM3 --image tflm_a`.

See `BOARD_GUIDE.md` for details, `integration/INTEGRATION.md` and
`firmware_sensa/platformio.ini` for the full-firmware measurements.

## Data

The SensEURCity data set is available from Zenodo (Van Poppel et al., 2023). The
benchmark code is available with Thyssen and Ghamari (2026).

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## License

MIT (see `LICENSE`). `firmware_sensa/` is built on the Sensa project's firmware;
the Sensirion SEN5x driver and esp-tflite-micro are fetched by the build system under
their own licenses.
