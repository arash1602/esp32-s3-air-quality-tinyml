/* aq_autorun.c -- stand-alone energy run for an external power analyser (Nordic PPK2).
 *
 * The board is powered by the analyser at 3.3 V with no USB connection, so nothing is
 * sent to a PC. Every phase is marked on one GPIO that the analyser records together
 * with the current (its logic input D0); tools/ppk2_log.py records, and
 * analysis/ppk2_energy.py turns the record into energy per prediction.
 *
 * Marker sequence (H = pin high, L = pin low):
 *   L 5 s                                   start-up, analyser settles
 *   header: n_models pulses, 2 ms H / 2 ms L, then L 1 s
 *   per model i:
 *     ID: i + 1 pulses, 5 ms H / 5 ms L
 *     L 0.3 s; the model is loaded and one prediction timed (unmarked)
 *     batch size K: K pulses, 3 ms H / 3 ms L. K predictions run back to back in
 *           one marked interval, K = 1 unless one prediction is shorter than
 *           AQ_MIN_HIGH_US, so that every marked interval lasts at least that long
 *           and stays far from glitch scale on the analyser's logic input
 *     L 0.5 s + idle_s                      idle reference (CPU in the scheduler)
 *     busy: per batch H, then L for AQ_GAP_US (busy-wait), until busy_s have
 *           passed and at least 3 batches were made
 *     L 1 s
 *   end: H 3 s, then L forever
 * A model that cannot be loaded gets its ID and idle period but no busy period.
 * Low gaps inside a busy period are AQ_GAP_US, inside an ID 5 ms, inside a batch-size
 * group 3 ms, and between phases at least 0.3 s, so the phases separate without
 * ambiguity.
 */
#include "aq_bench.h"
#include "aq_infer.h"
#include "aq_sample_window.h"

#include <string.h>

#ifndef AQ_GAP_US
#define AQ_GAP_US 200
#endif
#ifndef AQ_MIN_HIGH_US
#define AQ_MIN_HIGH_US 1000
#endif
#define AQ_MAX_BATCH 200

static void spin_us(uint32_t us) {
    int64_t end = aqb_now_us() + us;
    while (aqb_now_us() < end) { }
}

static void pulses(void (*mark)(int), uint32_t n, uint32_t high_us, uint32_t low_us) {
    for (uint32_t k = 0; k < n; k++) {
        mark(1); spin_us(high_us);
        mark(0); spin_us(low_us);
    }
}

void aq_autorun(const uint8_t *models, void (*mark)(int), uint32_t busy_s, uint32_t idle_s) {
    static aq_model_t m;
    static float window[AQ_MAX_T * AQ_MAX_IN];
    uint32_t count = 0;
    mark(0);
    if (models && memcmp(models, "AQIX", 4) == 0) memcpy(&count, models + 4, 4);
    memset(window, 0, sizeof window);
    memcpy(window, aq_sample_window, sizeof aq_sample_window);
    aqb_sleep_ms(5000);
    pulses(mark, count, 2000, 2000);
    aqb_sleep_ms(1000);

    for (uint32_t i = 0; i < count; i++) {
        pulses(mark, i + 1, 5000, 5000);
        aqb_sleep_ms(300);
        uint32_t off, len;
        memcpy(&off, models + 8 + 8 * i, 4);
        memcpy(&len, models + 12 + 8 * i, 4);
        int ok = aq_load(&m, models + off, len) == 0;
        uint32_t k = 1;
        if (ok) {
            double s;
            int64_t t0 = aqb_now_us();
            aq_predict(&m, window, &s);                 /* also warms the flash cache */
            int64_t one = aqb_now_us() - t0;
            if (one < AQ_MIN_HIGH_US)
                k = one > 0 ? (uint32_t)((AQ_MIN_HIGH_US + one - 1) / one) : AQ_MAX_BATCH;
            if (k > AQ_MAX_BATCH) k = AQ_MAX_BATCH;
            pulses(mark, k, 3000, 3000);
        }
        aqb_sleep_ms(500 + 1000 * idle_s);
        if (ok) {
            volatile double sink = 0.0;
            uint32_t n = 0;
            int64_t end = aqb_now_us() + (int64_t)busy_s * 1000000;
            while (aqb_now_us() < end || n < 3) {
                double s;
                mark(1);
                for (uint32_t j = 0; j < k; j++) {
                    aq_predict(&m, window, &s);
                    sink += s;
                }
                mark(0);
                n++;
                spin_us(AQ_GAP_US);
            }
            (void)sink;
        }
        aqb_sleep_ms(1000);
    }
    mark(1);
    aqb_sleep_ms(3000);
    mark(0);
    for (;;) aqb_sleep_ms(1000);
}
