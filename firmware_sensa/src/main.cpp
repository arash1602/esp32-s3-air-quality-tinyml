/* Sensa firmware (dev-board build) with the hourly WHO-exceedance warning.
 *
 * The application is the Sensa team's firmware (SEN55 at 1 Hz, 10-s AQI pipeline,
 * BLE service), unchanged except that the dev-board heartbeat LED is off (it blinks
 * an RGB LED with blocking delays, which neither the wearable nor a power
 * measurement wants). With AQ_ENABLE, every hour of SEN55 readings is averaged into
 * the 24-hour window (aq_window) and one prediction is run with the deployed model
 * (aq_model_blob.h, an 8-bit TCN); the result is printed and sent over BLE.
 *
 * Build flags (platformio.ini):
 *   AQ_ENABLE      add the hourly warning (without it: the plain Sensa firmware)
 *   AQ_HOUR_S      seconds per "hour" (default 3600; 60 for the power run)
 *   AQ_PREFILL     start with a full 24-hour window (one real test window), so a
 *                  prediction runs at the first hour boundary instead of after 24 h
 *   AQ_MARK_GPIO   pin raised during each prediction (PPK2 logic input D0), default 4
 *   AQ_MEM_PRINT_S period of the memory report on the USB console (default 10 s)
 *   AQ_PRINT_WINDOW print each 24-hour window as it was classified ("[AQW]", 120 float32
 *                  bit patterns in hex), so a PC can re-run the same window exactly
 */
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <soc/gpio_reg.h>

#include "ble/ble_service.h"
#include "processing/pipeline.h"
#include "sensor/sen55_sensor.h"

static Sen55Sensor sensor;
static uint32_t last_tick_ms = 0;
static const uint32_t INTERVAL_MS = 1000;

#ifndef AQ_MARK_GPIO
#define AQ_MARK_GPIO 4
#endif
#ifndef AQ_MEM_PRINT_S
#define AQ_MEM_PRINT_S 10
#endif

#ifdef AQ_ENABLE
#include "aq_infer.h"
#include "aq_window.h"
#include "aq_model_blob.h"
#ifdef AQ_PREFILL
#include "aq_sample_window.h"
#endif
#ifndef AQ_HOUR_S
#define AQ_HOUR_S 3600
#endif

static aq_model_t aq_model;
static aq_window_t aq_win;
static bool aq_ok = false;
static uint32_t aq_last_ts = 0;
static int8_t aq_exceed = -1;          // -1 until 24 hours of data exist
static uint32_t aq_count = 0, aq_last_us = 0;
static uint32_t aq_t0_s = 0;           // offset of the clock given to aq_window

static inline void mark(int on) {
    if (on) REG_WRITE(GPIO_OUT_W1TS_REG, 1u << AQ_MARK_GPIO);
    else REG_WRITE(GPIO_OUT_W1TC_REG, 1u << AQ_MARK_GPIO);
}

// aq_window clock: seconds, scaled so that one "hour" lasts AQ_HOUR_S real seconds
static uint32_t aq_clock_s(uint32_t ms) {
    return aq_t0_s + (uint32_t)((uint64_t)ms * (3600 / AQ_HOUR_S) / 1000);
}

static void aq_setup() {
    pinMode(AQ_MARK_GPIO, OUTPUT);
    mark(0);
    aq_ok = aq_load(&aq_model, aq_model_blob, aq_model_blob_len) == 0;
    aqw_init(&aq_win);
#ifdef AQ_PREFILL
    // 24 hours from one real test window (one reading per hour), then the live clock
    // continues in the 25th hour
    for (int h = 0; h < 24; h++) {
        const float *x = aq_sample_window + h * 5;     // T, RH, AH, PM2.5, PM10
        aqw_push(&aq_win, (uint32_t)h * 3600 + 1800, x[0], x[1], x[3], x[4]);
    }
    aq_t0_s = 24 * 3600 + 1;
#endif
    Serial.printf("[AQ] model %s, %u bytes, %u parameters; library static RAM %u bytes; "
                  "hour = %u s\n", aq_ok ? "loaded" : "FAILED", (unsigned)aq_model_blob_len,
                  (unsigned)aq_model.n_params, (unsigned)aq_static_bytes(), (unsigned)AQ_HOUR_S);
}

static void aq_after_tick() {
    Reading r = pipeline_last_reading();
    if (!aq_ok || r.ts_ms == 0 || r.ts_ms == aq_last_ts) return;   // no new reading
    aq_last_ts = r.ts_ms;
    if (aqw_push(&aq_win, aq_clock_s(r.ts_ms), r.temp_c, r.rh, r.pm2_5, r.pm10) <= 0) return;
    float w[AQW_HOURS * AQW_CH];
    if (!aqw_window(&aq_win, w)) {
        Serial.printf("[AQ] hour closed (%u); window not complete yet\n",
                      (unsigned)aq_win.hours_closed);
        return;
    }
#ifdef AQ_PRINT_WINDOW
    Serial.print("[AQW]");
    for (int i = 0; i < AQW_HOURS * AQW_CH; i++) {
        uint32_t b;
        memcpy(&b, &w[i], 4);
        Serial.printf(" %08x", (unsigned)b);
    }
    Serial.println();
#endif
    double score;
    uint32_t t0 = micros();
    mark(1);
    aq_exceed = (int8_t)aq_predict(&aq_model, w, &score);
    mark(0);
    aq_last_us = micros() - t0;
    aq_count++;
    ble_notify_exceedance(aq_exceed, (float)score);
    uint64_t sb;
    memcpy(&sb, &score, 8);
    Serial.printf("[AQ] prediction %u: WHO 24-h exceedance expected: %d (score %.3f, bits %016llx), %u us\n",
                  (unsigned)aq_count, aq_exceed, score, (unsigned long long)sb, (unsigned)aq_last_us);
}
#endif

static void print_mem() {
    Serial.printf("MEM free_heap %u min_free_heap %u largest_block %u loop_stack_free %u"
#ifdef AQ_ENABLE
                  " aq_predictions %u"
#endif
                  "\n",
                  (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMinFreeHeap(),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
                  (unsigned)uxTaskGetStackHighWaterMark(NULL)
#ifdef AQ_ENABLE
                  , (unsigned)aq_count
#endif
    );
}

void setup() {
    Serial.begin(115200);
    Serial.setTxTimeoutMs(0);          // never wait for a USB host (power runs: none)
    unsigned long t = millis();
    while (!Serial && (millis() - t) < 3000);
    Serial.println("[Sensa] Running on dev board"
#ifdef AQ_ENABLE
                   " with the hourly WHO-exceedance warning"
#endif
                   ".");
    if (!sensor.begin()) {
        Serial.println("ERROR: SEN55 failed to start. Check wiring (SDA=8, SCL=9).");
    }
    pipeline_init(&sensor);
    ble_init();
#ifdef AQ_ENABLE
    aq_setup();
#endif
    print_mem();
}

void loop() {
    static uint32_t last_mem_ms = 0;
    uint32_t now = millis();
    if (now - last_mem_ms >= AQ_MEM_PRINT_S * 1000u) {
        last_mem_ms = now;
        print_mem();
    }
    if (now - last_tick_ms < INTERVAL_MS) {
        delay(10);                     // yield; the original loop blocked in its LED blink
        return;
    }
    last_tick_ms = now;

    Classification result = pipeline_tick();
    ble_notify(pipeline_last_reading(), result);
#ifdef AQ_ENABLE
    aq_after_tick();
#endif
    Serial.printf("Classification: %d\n", (int)result);
}
