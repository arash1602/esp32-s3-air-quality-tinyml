#pragma once
#include <cstddef>

#include "types.h"

/*
 * =========================================================================
 * On-Device PM2.5 Calibration  (TensorFlow Lite Micro)
 * =========================================================================
 *
 * WHAT THIS MODULE DOES
 *   The SEN55 is an optical particle sensor. Like any optical instrument it
 *   has a systematic bias — it tends to under- or over-read depending on
 *   particle size mix, temperature, and humidity. We trained a tiny neural
 *   network (a 1-D CNN) on paired SEN55 + reference-grade (BAM) measurements
 *   so it learns to "undo" that bias.
 *
 *   Think of it as a calibration curve, except instead of a single gain +
 *   offset it is an 8-input non-linear correction. The trained network is
 *   compressed to an int8 TensorFlow Lite model and runs entirely on the
 *   ESP32-S3 — no Wi-Fi, no cloud.
 *
 * HOW THE MODEL GETS HERE
 *   The Python training pipeline in tools/pytorch_calibration/ produces two
 *   generated C headers that this module #includes:
 *     - include/calib_model.h   — the .tflite model as a C byte array
 *     - include/calib_scaler.h  — the input normalisation constants
 *   Run `python main.py` in that folder to (re)generate them.
 *
 * BUILDS BEFORE THE MODEL EXISTS
 *   Until those two headers are generated, this module compiles in a
 *   "passthrough" mode: calibrate_pm25() simply returns the raw reading
 *   unchanged, so the rest of the firmware still builds and runs. A compiler
 *   #warning tells you when this is happening.
 *
 * RAM / FLASH FOOTPRINT
 *   - Model weights:  stored in flash (~2-9 KB, see calib_model.h).
 *   - Tensor arena:   16 KB of static RAM (see kTensorArenaSize in
 *                     calibrate.cpp). calibrate_arena_used_bytes() reports
 *                     how much of that the model actually needs so the
 *                     arena can be trimmed.
 *   No dynamic allocation (new/malloc) happens in the inference path.
 */

/*
 * calibrate_init()
 * Loads the TFLite model and allocates its tensor arena. Call this exactly
 * once during start-up — pipeline_init() already does this for you.
 *
 * Returns:
 *   true  — model loaded; calibrate_pm25() will run real inference.
 *   false — model unavailable (not generated yet, schema mismatch, or the
 *           tensor arena was too small). The firmware keeps running;
 *           calibrate_pm25() falls back to passthrough.
 */
bool calibrate_init();

/*
 * calibrate_pm25()
 * Runs the calibration model on one raw SEN55 reading.
 *
 * Parameters:
 *   r — a single raw Reading straight from the sensor.
 *
 * Returns:
 *   The corrected PM2.5 value in µg/m³ (never negative). If no model is
 *   loaded, returns r.pm2_5 unchanged.
 *
 * Note: only PM2.5 is calibrated — that is the single value the model was
 * trained to predict. The other PM channels are left untouched.
 */
float calibrate_pm25(const Reading& r);

/*
 * calibrate_model_loaded()
 * Returns true if a real TFLite model is loaded and inference is active,
 * false if the module is in passthrough mode.
 */
bool calibrate_model_loaded();

/*
 * calibrate_arena_used_bytes()
 * Returns how many bytes of the static tensor arena the loaded model
 * actually uses. Useful for documenting/trimming RAM usage. Returns 0 when
 * no model is loaded.
 */
size_t calibrate_arena_used_bytes();
