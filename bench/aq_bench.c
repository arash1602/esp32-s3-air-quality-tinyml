/* aq_bench.c -- see aq_bench.h */
#include "aq_bench.h"
#include "aq_infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t *g_models;
static uint32_t g_count, g_blob_len;
static aq_model_t g_model;
static int g_loaded = -1;
static uint8_t *g_ram;                   /* blob copied to RAM by 'R', or NULL */
static float g_window[AQ_MAX_T * AQ_MAX_IN];
static uint32_t g_times[5000];

static void text(const char *s) { aqb_write(s, strlen(s)); }
static uint32_t rd_u32(void) { uint32_t v; aqb_read(&v, 4); return v; }

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int select_model(uint32_t idx, int to_ram) {
    if (!g_models || idx >= g_count) return -1;
    uint32_t off, len;
    memcpy(&off, g_models + 8 + 8 * idx, 4);
    memcpy(&len, g_models + 12 + 8 * idx, 4);
    g_loaded = -1;
    free(g_ram);
    g_ram = NULL;
    const uint8_t *src = g_models + off;
    if (to_ram) {
        g_ram = malloc(len);
        if (!g_ram) return -3;               /* does not fit in RAM */
        memcpy(g_ram, src, len);
        src = g_ram;
    }
    if (aq_load(&g_model, src, len) != 0) return -2;
    g_loaded = (int)idx; g_blob_len = len;
    return 0;
}

void aq_bench_loop(const uint8_t *models) {
    char line[160];
    const size_t win_bytes = (size_t)AQ_MAX_T * AQ_MAX_IN * sizeof(float);
    if (models && memcmp(models, "AQIX", 4) == 0) {
        g_models = models;
        memcpy(&g_count, models + 4, 4);
    }
    memset(g_window, 0, sizeof g_window);
    text("AQREADY\n");

    for (;;) {
        uint8_t cmd;
        aqb_read(&cmd, 1);
        if (cmd == 'I') {
            snprintf(line, sizeof line, "AQ %lu %d\n", (unsigned long)g_count, aqb_cpu_mhz());
            text(line);
        } else if (cmd == 'S' || cmd == 'R') {
            uint8_t idx;
            aqb_read(&idx, 1);
            int rc = select_model(idx, cmd == 'R');
            if (rc) snprintf(line, sizeof line, "ERR %d\n", rc);
            else snprintf(line, sizeof line, "OK %d %ld %ld %lu %lu\n", idx, (long)g_model.arch,
                          (long)g_model.precision, (unsigned long)g_model.n_params,
                          (unsigned long)g_blob_len);
            text(line);
        } else if (cmd == 'W' || cmd == 'P') {
            uint32_t n = rd_u32();
            for (uint32_t i = 0; i < n; i++) {
                aqb_read(g_window, win_bytes);
                double s = 0.0; uint8_t d = 0; uint32_t us = 0;
                if (g_loaded >= 0) {
                    int64_t t0 = aqb_now_us();
                    d = (uint8_t)aq_predict(&g_model, g_window, &s);
                    us = (uint32_t)(aqb_now_us() - t0);
                }
                /* one write per result: a record split over several USB transfers
                   was occasionally cut short by the USB CDC driver */
                uint8_t rec[13]; size_t k = 0;
                if (cmd == 'W') { float f = (float)s; memcpy(rec, &f, 4); k = 4; }
                else { memcpy(rec, &s, 8); rec[8] = d; k = 9; }
                memcpy(rec + k, &us, 4);
                aqb_write(rec, k + 4);
            }
        } else if (cmd == 'Q') {
            /* one window per command: if a transfer is lost on the way in, the
               partial window is dropped; if the reply is lost, the PC resends */
            if (!aqb_read_within(g_window, win_bytes, 250)) continue;
            double s = 0.0; uint8_t d = 0; uint32_t us = 0;
            if (g_loaded >= 0) {
                int64_t t0 = aqb_now_us();
                d = (uint8_t)aq_predict(&g_model, g_window, &s);
                us = (uint32_t)(aqb_now_us() - t0);
            }
            uint8_t rec[13];
            memcpy(rec, &s, 8); rec[8] = d; memcpy(rec + 9, &us, 4);
            aqb_write(rec, 13);
        } else if (cmd == 'X') {
            aqb_read(g_window, win_bytes);
            text("X\n");
        } else if (cmd == 'B') {
            uint32_t reps = rd_u32();
            if (reps > 5000) reps = 5000;
            if (g_loaded < 0 || reps == 0) { text("ERR no model\n"); continue; }
            uint64_t sum = 0;
            volatile double sink = 0.0;
            for (uint32_t i = 0; i < reps; i++) {
                double s;
                int64_t t0 = aqb_now_us();
                aq_predict(&g_model, g_window, &s);
                g_times[i] = (uint32_t)(aqb_now_us() - t0);
                sink += s; sum += g_times[i];
            }
            (void)sink;
            qsort(g_times, reps, sizeof(uint32_t), cmp_u32);
            snprintf(line, sizeof line, "B %lu %lu %lu %lu %.1f\n", (unsigned long)reps,
                     (unsigned long)g_times[0], (unsigned long)g_times[reps / 2],
                     (unsigned long)g_times[(reps * 99) / 100], (double)sum / reps);
            text(line);
        } else if (cmd == 'L') {
            uint32_t secs = rd_u32(), count = 0;
            int64_t end = aqb_now_us() + (int64_t)secs * 1000000;
            volatile double sink = 0.0;
            while (g_loaded >= 0 && aqb_now_us() < end) {
                double s;
                aq_predict(&g_model, g_window, &s);
                sink += s; count++;
            }
            (void)sink;
            snprintf(line, sizeof line, "L %lu %lu\n", (unsigned long)count, (unsigned long)secs);
            text(line);
        } else if (cmd == 'Z') {
            uint32_t secs = rd_u32();
            aqb_sleep_ms(secs * 1000u);
            snprintf(line, sizeof line, "Z %lu\n", (unsigned long)secs);
            text(line);
        } else if (cmd == 'M') {
            uint32_t fh, mfh, st;
            aqb_mem(&fh, &mfh, &st);
            snprintf(line, sizeof line, "M %lu %lu %lu %lu\n", (unsigned long)fh,
                     (unsigned long)mfh, (unsigned long)st, (unsigned long)aq_static_bytes());
            text(line);
        }
    }
}
