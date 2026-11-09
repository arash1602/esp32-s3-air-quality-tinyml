/* TensorFlow Lite Micro benchmark on the ESP32-S3, the counterpart of aq_bench for the
 * comparison with aq_infer. Same board, same "models" partition, same USB protocol
 * (bench/aq_bench.h: I S R Q X B L Z M) and, with TFLM_AUTORUN, the same PPK2 marker
 * sequence as bench/aq_autorun.c, so tools/hil.py, tools/ppk2_log.py and
 * analysis/ppk2_energy.py are used unchanged.
 *
 * Model image (tools/tflm_convert.py): "AQIX" u32 count, count x (u32 offset, u32 length),
 * blobs; each blob is a 64-byte header ("TFLM", u32 arch, u32 int8, f32 threshold,
 * u32 log mask, f32 mean[5], f32 std[5]) and the .tflite flatbuffer, read in place from
 * flash. A prediction takes the raw window, scales it as aq_infer does (log1p, mean,
 * std), quantizes it for int8 models, runs the interpreter and compares the logit with
 * the threshold; the timing covers all of it, as for aq_infer.
 *
 * 'S' answers "OK <idx> <arch> <prec> <arena_bytes> <blob_bytes>": the fourth field is
 * the interpreter's arena use (aq_bench reports the parameter count there).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "driver/usb_serial_jtag.h"
#include "driver/gpio.h"
#include "soc/gpio_reg.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

extern "C" {
#include "aq_sample_window.h"
}

#ifndef TFLM_MARK_GPIO
#define TFLM_MARK_GPIO 4
#endif

/* ---------------------------------------------------------------- USB I/O */
static void io_init(void) {
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 4096;
    cfg.tx_buffer_size = 4096;
    usb_serial_jtag_driver_install(&cfg);
}
static void rd(void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    while (n) {
        int r = usb_serial_jtag_read_bytes(p, n, portMAX_DELAY);
        if (r > 0) { p += r; n -= r; }
    }
}
static int rd_within(void *buf, size_t n, uint32_t ms) {
    uint8_t *p = (uint8_t *)buf;
    int64_t end = esp_timer_get_time() + (int64_t)ms * 1000;
    while (n) {
        int64_t left = end - esp_timer_get_time();
        if (left <= 0) return 0;
        int r = usb_serial_jtag_read_bytes(p, n, pdMS_TO_TICKS(left / 1000 + 1));
        if (r > 0) { p += r; n -= r; }
    }
    return 1;
}
static void wr(const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n) {
        int w = usb_serial_jtag_write_bytes(p, n, portMAX_DELAY);
        if (w > 0) { p += w; n -= w; }
    }
}
static void text(const char *s) { wr(s, strlen(s)); }
static inline int64_t now_us(void) { return esp_timer_get_time(); }

/* ---------------------------------------------------------------- models */
struct Hdr {
    char magic[4];
    uint32_t arch, int8;
    float thr;
    uint32_t log_mask;
    float mean[5], stdv[5];
};

static const uint8_t *g_img;
static uint32_t g_count;
static int g_loaded = -1;
static uint32_t g_len;
static Hdr g_hdr;
static uint8_t *g_arena;
static size_t g_arena_size;
alignas(16) static uint8_t g_interp_mem[sizeof(tflite::MicroInterpreter)];
static tflite::MicroInterpreter *g_it;
static tflite::MicroMutableOpResolver<24> g_ops;
static float g_window[24 * 5];

static void ops_init(void) {
    g_ops.AddAdd();
    g_ops.AddSub();
    g_ops.AddMul();
    g_ops.AddConv2D();
    g_ops.AddFullyConnected();
    g_ops.AddPad();
    g_ops.AddReshape();
    g_ops.AddStridedSlice();
    g_ops.AddConcatenation();
    g_ops.AddLogistic();
    g_ops.AddTanh();
    g_ops.AddRelu();
    g_ops.AddQuantize();
    g_ops.AddDequantize();
    g_ops.AddSplit();
    g_ops.AddSplitV();
    g_ops.AddMaximum();
    g_ops.AddPack();
    g_ops.AddUnpack();
    g_ops.AddExpandDims();
    g_ops.AddSqueeze();
}

static int select_model(uint32_t idx) {
    if (!g_img || idx >= g_count) return -1;
    uint32_t off, len;
    memcpy(&off, g_img + 8 + 8 * idx, 4);
    memcpy(&len, g_img + 12 + 8 * idx, 4);
    if (g_it) { g_it->~MicroInterpreter(); g_it = nullptr; }
    g_loaded = -1;
    memcpy(&g_hdr, g_img + off, sizeof g_hdr);
    if (memcmp(g_hdr.magic, "TFLM", 4) != 0) return -2;
    const tflite::Model *m = tflite::GetModel(g_img + off + 64);
    if (m->version() != TFLITE_SCHEMA_VERSION) return -4;
    g_it = new (g_interp_mem) tflite::MicroInterpreter(m, g_ops, g_arena, g_arena_size);
    if (g_it->AllocateTensors() != kTfLiteOk) { g_it->~MicroInterpreter(); g_it = nullptr; return -5; }
    g_loaded = (int)idx;
    g_len = len - 64;
    return 0;
}

/* raw window -> logit (as aq_infer: scaling, model, dequantized output) */
static double predict(const float *w, uint8_t *dec) {
    TfLiteTensor *in = g_it->input(0);
    const bool q = g_hdr.int8 != 0;
    const float is = q ? in->params.scale : 1.0f;
    const int iz = q ? in->params.zero_point : 0;
    for (int i = 0; i < 24 * 5; i++) {
        int c = i % 5;
        float v = w[i];
        if (g_hdr.log_mask & (1u << c)) v = log1pf(v > 0.0f ? v : 0.0f);
        v = (v - g_hdr.mean[c]) / g_hdr.stdv[c];
        if (q) {
            int x = (int)lroundf(v / is) + iz;
            in->data.int8[i] = (int8_t)(x < -128 ? -128 : x > 127 ? 127 : x);
        } else {
            in->data.f[i] = v;
        }
    }
    g_it->Invoke();
    TfLiteTensor *out = g_it->output(0);
    double y = q ? ((double)out->data.int8[0] - out->params.zero_point) * out->params.scale
                 : (double)out->data.f[0];
    *dec = y > (double)g_hdr.thr;
    return y;
}

/* ---------------------------------------------------------------- protocol */
static uint32_t g_times[5000];
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void bench_loop(void) {
    char line[160];
    text("AQREADY\n");
    for (;;) {
        uint8_t cmd;
        rd(&cmd, 1);
        if (cmd == 'I') {
            snprintf(line, sizeof line, "AQ %lu 240\n", (unsigned long)g_count);
            text(line);
        } else if (cmd == 'S' || cmd == 'R') {
            uint8_t idx;
            rd(&idx, 1);
            if (cmd == 'R') { text("ERR -3\n"); continue; }
            int rc = select_model(idx);
            if (rc) snprintf(line, sizeof line, "ERR %d\n", rc);
            else snprintf(line, sizeof line, "OK %d %lu %lu %lu %lu\n", idx,
                          (unsigned long)g_hdr.arch, (unsigned long)(g_hdr.int8 ? 1 : 0),
                          (unsigned long)g_it->arena_used_bytes(), (unsigned long)g_len);
            text(line);
        } else if (cmd == 'Q') {
            if (!rd_within(g_window, sizeof g_window, 250)) continue;
            double s = 0.0; uint8_t d = 0; uint32_t us = 0;
            if (g_loaded >= 0) {
                int64_t t0 = now_us();
                s = predict(g_window, &d);
                us = (uint32_t)(now_us() - t0);
            }
            uint8_t rec[13];
            memcpy(rec, &s, 8); rec[8] = d; memcpy(rec + 9, &us, 4);
            wr(rec, 13);
        } else if (cmd == 'X') {
            rd(g_window, sizeof g_window);
            text("X\n");
        } else if (cmd == 'B') {
            uint32_t reps;
            rd(&reps, 4);
            if (reps > 5000) reps = 5000;
            if (g_loaded < 0 || reps == 0) { text("ERR no model\n"); continue; }
            uint64_t sum = 0;
            uint8_t d;
            for (uint32_t i = 0; i < reps; i++) {
                int64_t t0 = now_us();
                predict(g_window, &d);
                g_times[i] = (uint32_t)(now_us() - t0);
                sum += g_times[i];
            }
            qsort(g_times, reps, sizeof(uint32_t), cmp_u32);
            snprintf(line, sizeof line, "B %lu %lu %lu %lu %.1f\n", (unsigned long)reps,
                     (unsigned long)g_times[0], (unsigned long)g_times[reps / 2],
                     (unsigned long)g_times[(reps * 99) / 100], (double)sum / reps);
            text(line);
        } else if (cmd == 'L') {
            uint32_t secs, count = 0;
            rd(&secs, 4);
            int64_t end = now_us() + (int64_t)secs * 1000000;
            uint8_t d;
            while (g_loaded >= 0 && now_us() < end) { predict(g_window, &d); count++; }
            snprintf(line, sizeof line, "L %lu %lu\n", (unsigned long)count, (unsigned long)secs);
            text(line);
        } else if (cmd == 'Z') {
            uint32_t secs;
            rd(&secs, 4);
            vTaskDelay(pdMS_TO_TICKS(secs * 1000u));
            snprintf(line, sizeof line, "Z %lu\n", (unsigned long)secs);
            text(line);
        } else if (cmd == 'M') {
            snprintf(line, sizeof line, "M %lu %lu %lu %lu\n",
                     (unsigned long)esp_get_free_heap_size(),
                     (unsigned long)esp_get_minimum_free_heap_size(),
                     (unsigned long)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)),
                     (unsigned long)(g_it ? g_it->arena_used_bytes() : 0));
            text(line);
        }
    }
}

/* ---------------------------------------------------- stand-alone energy run */
static inline void mark(int on) {
    if (on) REG_WRITE(GPIO_OUT_W1TS_REG, 1u << TFLM_MARK_GPIO);
    else REG_WRITE(GPIO_OUT_W1TC_REG, 1u << TFLM_MARK_GPIO);
}
static void spin_us(uint32_t us) { int64_t e = now_us() + us; while (now_us() < e) { } }
static void pulses(uint32_t n, uint32_t hi, uint32_t lo) {
    for (uint32_t k = 0; k < n; k++) { mark(1); spin_us(hi); mark(0); spin_us(lo); }
}
static void sleep_ms(uint32_t ms) { vTaskDelay(pdMS_TO_TICKS(ms)); }

static void autorun(uint32_t busy_s, uint32_t idle_s) {
    gpio_reset_pin((gpio_num_t)TFLM_MARK_GPIO);
    gpio_set_direction((gpio_num_t)TFLM_MARK_GPIO, GPIO_MODE_OUTPUT);
    mark(0);
    memcpy(g_window, aq_sample_window, sizeof g_window);
    sleep_ms(5000);
    pulses(g_count, 2000, 2000);
    sleep_ms(1000);
    for (uint32_t i = 0; i < g_count; i++) {
        pulses(i + 1, 5000, 5000);
        sleep_ms(300);
        int ok = select_model(i) == 0;
        uint32_t k = 1;
        uint8_t d;
        if (ok) {
            int64_t t0 = now_us();
            predict(g_window, &d);
            int64_t one = now_us() - t0;
            if (one < 1000) k = one > 0 ? (uint32_t)((1000 + one - 1) / one) : 200;
            if (k > 200) k = 200;
            pulses(k, 3000, 3000);
        }
        sleep_ms(500 + 1000 * idle_s);
        if (ok) {
            uint32_t n = 0;
            int64_t end = now_us() + (int64_t)busy_s * 1000000;
            while (now_us() < end || n < 3) {
                mark(1);
                for (uint32_t j = 0; j < k; j++) predict(g_window, &d);
                mark(0);
                n++;
                spin_us(200);
            }
        }
        sleep_ms(1000);
    }
    mark(1);
    sleep_ms(3000);
    mark(0);
    for (;;) sleep_ms(1000);
}

extern "C" void app_main(void) {
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "models");
    esp_partition_mmap_handle_t h;
    const void *ptr = nullptr;
    if (!part || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) != ESP_OK)
        ptr = nullptr;
    g_img = (const uint8_t *)ptr;
    if (g_img && memcmp(g_img, "AQIX", 4) == 0) memcpy(&g_count, g_img + 4, 4);
    else g_img = nullptr;
    /* the largest arena the heap allows, up to 256 KB (reported per model by 'S'/'M') */
    for (g_arena_size = 256 * 1024; g_arena_size >= 32 * 1024; g_arena_size -= 16 * 1024) {
        g_arena = (uint8_t *)heap_caps_aligned_alloc(16, g_arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (g_arena) break;
    }
    ops_init();
#ifdef TFLM_AUTORUN
    autorun(8, 3);
#else
    io_init();
    vTaskDelay(pdMS_TO_TICKS(200));
    bench_loop();
#endif
}
