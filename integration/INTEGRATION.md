# Running a WHO-exceedance classifier inside the Sensa firmware

The benchmark firmware (`../firmware`) measures the models in isolation. This folder
puts one model into the real Sensa application (repository `sensa-senior-project`), so
the paper can also report what the feature costs the shipping firmware: flash and RAM
from PlatformIO's size report, with and without it.

What it adds: every hour, the device averages its SEN55 readings (as paper 1 averaged
SensEURCity's minute data), keeps the last 24 hourly means of temperature, relative
humidity, absolute humidity, PM2.5 and PM10, and runs the classifier, which says
whether the coming 24-hour mean is expected to exceed the WHO limit (PM2.5 15 or
PM10 45 ug/m3) at the chosen horizon. Nothing changes in the existing 1 Hz pipeline.

Mapping of the SEN55 to paper 1's inputs (Alphasense OPC-N3 + Sensirion SHT31):

| paper 1 input | Sensa `Reading` field |
| --- | --- |
| `SHT31TE` temperature | `temp_c` |
| `SHT31HE` relative humidity | `rh` |
| `Absolute_humidity` | computed per reading by `aq_abs_humidity()` (the dataset's own formula) |
| `OPCN3PM25` | `pm2_5` |
| `OPCN3PM10` | `pm10` |

The models were trained on OPC-N3 data. Running them on SEN55 readings is a
deployment demonstration, not a validated prediction: the paper says so.

## Steps (in the Sensa repository)

1. Copy the library files into the Sensa repo, each library in its own folder:

   ```
   lib/aq_infer/src/aq_infer.h, aq_infer.c        (from edge/c)
   lib/aq_window/src/aq_window.h, aq_window.c     (from edge/integration/aq_window)
   ```

2. Pick the model and turn it into a header (from the `edge` folder). The 24-hour
   horizon, leave-one-city-out model is the natural choice for a device that will be
   used in a new place; any `.aqm` works (`LR`, `GBT`, `WHO`, or a network's
   `_int8.aqm` / `_f32.aqm` from `export_out`):

   ```
   python tools/blob_to_header.py export_classical/h24/multi/heldout-Zagreb/LR.aqm -o aq_model_blob.h
   ```

   and copy `aq_model_blob.h` to the Sensa repo's `include/` folder.

3. In `src/main.cpp`:

   ```cpp
   #include "aq_infer.h"
   #include "aq_window.h"
   #include "aq_model_blob.h"

   static aq_model_t aq_model;
   static aq_window_t aq_win;
   static bool aq_ok = false;
   static uint32_t aq_last_ts = 0;
   static int8_t aq_exceed = -1;          // -1 until 24 hours of data exist

   // in setup(), after pipeline_init():
   aq_ok = aq_load(&aq_model, aq_model_blob, aq_model_blob_len) == 0;
   aqw_init(&aq_win);

   // in loop(), right after `Classification result = pipeline_tick();`
   Reading r = pipeline_last_reading();
   if (aq_ok && r.ts_ms != 0 && r.ts_ms != aq_last_ts) {     // a new reading
       aq_last_ts = r.ts_ms;
       if (aqw_push(&aq_win, r.ts_ms / 1000, r.temp_c, r.rh, r.pm2_5, r.pm10) > 0) {
           float w[AQW_HOURS * AQW_CH];
           if (aqw_window(&aq_win, w)) {
               double score;
               aq_exceed = (int8_t)aq_predict(&aq_model, w, &score);
               Serial.printf("WHO 24h exceedance expected: %d (score %.3f)\n", aq_exceed, score);
           }
       }
   }
   ```

   Where the result goes is the team's choice: an alert pattern via
   `maybe_trigger_alert()`, or a fourth BLE characteristic next to the class label.

4. Size the library's working memory in `platformio.ini` (`build_flags` of the
   `sensa-pcb` environment). The default reserves about 41 KB of RAM for the largest
   networks. With a classical model add `-DAQ_MAX_CH=8 -DAQ_MAX_LAYERS=1` (about 3 KB);
   with a network, set `-DAQ_MAX_CH=` to its widest layer (`aq_load` refuses a model
   that does not fit, so a wrong value fails safely: `aq_ok` stays false).

5. Build `sensa-pcb` before and after the change and save both size reports:

   ```
   pio run -e sensa-pcb -v > size_before.txt     (on the unchanged code)
   pio run -e sensa-pcb -v > size_after.txt
   ```

   The lines `RAM: ... used ... bytes` and `Flash: ... used ... bytes` are the numbers
   the paper needs. Send both files back.

## Test on a PC

```
gcc -O2 -std=c11 -I c -I integration/aq_window -o test_aq_window \
    integration/test_aq_window.c integration/aq_window/aq_window.c c/aq_infer.c -lm
./test_aq_window export_classical/h24/multi/heldout-Zagreb/LR.aqm
```

It checks the absolute-humidity formula against rows of the SensEURCity files, the
hourly averaging (invalid readings dropped, missing hours invalidate the window, the
49.7-day `millis()` wrap), and runs the model on a simulated day.
