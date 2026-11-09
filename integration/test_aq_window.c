/* Tests for integration/aq_window and the firmware-side use of aq_infer.
 *
 *   gcc -O2 -std=c11 -I c -I integration/aq_window -o test_aq_window \
 *       integration/test_aq_window.c integration/aq_window/aq_window.c c/aq_infer.c -lm
 *   ./test_aq_window [model.aqm]
 *
 * With a model blob, it also builds a window from simulated 1 Hz readings and runs
 * the model on it, as the firmware would every hour.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aq_infer.h"
#include "aq_window.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                           printf(__VA_ARGS__); printf("\n"); } } while (0)

/* {RH %, T degC, Absolute_humidity g/m3} sampled from the SensEURCity CSVs */
static const double AH_ROWS[][3] = {
    { 71.3969635, 14.5830472, 8.92139504 }, { 70.6034943, 11.373312, 7.23219182 },
    { 32.3277638, 15.9636072, 4.3931355 },  { 64.8661021, 16.3000687, 8.99577441 },
    { 50.4234379, 10.2304112, 4.80618061 }, { 68.299382, 4.7909514, 4.57785093 },
    { 75.9853513, 1.02845808, 3.95452966 }, { 57.2579538, 12.8793774, 6.44250745 },
    { 75.944152, 10.7431144, 7.47701454 },  { 51.9005112, 17.6699474, 7.81389811 },
    { 42.0706493, 19.0318151, 6.86690981 }, { 27.3456931, 26.94934, 7.02153697 },
};

static void test_abs_humidity(void) {
    for (size_t i = 0; i < sizeof AH_ROWS / sizeof AH_ROWS[0]; i++) {
        float ah = aq_abs_humidity((float)AH_ROWS[i][1], (float)AH_ROWS[i][0]);
        CHECK(fabs(ah - AH_ROWS[i][2]) < 1e-5 * AH_ROWS[i][2], "AH row %zu: %g vs %g", i,
              (double)ah, AH_ROWS[i][2]);
    }
}

static void test_hours(void) {
    aq_window_t w;
    float win[AQW_HOURS * AQW_CH];
    aqw_init(&w);
    /* 1 Hz readings for 24 h: hour k has PM2.5 = k, PM10 = 2k, T = 20, RH = 50 */
    uint32_t t = 1000;
    for (int k = 0; k < 24; k++)
        for (int s = 0; s < 3600; s++, t++)
            aqw_push(&w, t, 20.0f, 50.0f, (float)k, 2.0f * k);
    CHECK(!aqw_window(&w, win), "window ready before the 24th hour closed");
    CHECK(aqw_push(&w, t, 20.0f, 50.0f, 100.0f, 100.0f) == 1, "24th hour did not close");
    CHECK(aqw_window(&w, win), "window not ready after 24 hours");
    for (int k = 0; k < 24; k++) {
        CHECK(win[k * 5 + 3] == (float)k && win[k * 5 + 4] == 2.0f * k, "hour %d PM wrong", k);
        CHECK(win[k * 5 + 0] == 20.0f && win[k * 5 + 1] == 50.0f, "hour %d T/RH wrong", k);
        CHECK(fabsf(win[k * 5 + 2] - aq_abs_humidity(20.0f, 50.0f)) < 1e-6f, "hour %d AH", k);
    }
    /* invalid readings are dropped, not averaged */
    for (int s = 1; s < 3600; s++, t++)
        aqw_push(&w, t, 20.0f, 50.0f, (s % 2) ? NAN : 100.0f, 100.0f);
    t++;
    aqw_push(&w, t, 20.0f, 50.0f, 0.0f, 0.0f);
    CHECK(aqw_window(&w, win) && win[23 * 5 + 3] == 100.0f, "NaN readings entered the mean");
    /* a whole hour without valid readings invalidates the next 24 windows */
    t += 7200;
    CHECK(aqw_push(&w, t, 20.0f, 50.0f, 5.0f, 5.0f) == 2, "gap of two hours not closed");
    CHECK(!aqw_window(&w, win), "window with a missing hour reported ready");
    /* the millisecond clock wraps after 49.7 days: no endless loop, window invalid */
    CHECK(aqw_push(&w, 5, 20.0f, 50.0f, 5.0f, 5.0f) == AQW_HOURS + 1, "clock wrap not bounded");
    CHECK(!aqw_window(&w, win), "window across a clock wrap reported ready");
    for (uint32_t k = 1; k <= 24 * 3600; k++) aqw_push(&w, 5 + k, 20.0f, 50.0f, 7.0f, 9.0f);
    CHECK(aqw_window(&w, win) && win[23 * 5 + 3] == 7.0f, "no recovery 24 h after a clock wrap");
}

static void *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = aligned_alloc(16, ((size_t)n + 15) & ~(size_t)15);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); return NULL; }
    fclose(f); *len = (size_t)n; return buf;
}

static void test_model(const char *path) {
    size_t len;
    void *blob = slurp(path, &len);
    aq_model_t m;
    CHECK(blob && aq_load(&m, blob, len) == 0, "cannot load %s", path);
    if (!blob) return;
    aq_window_t w;
    float win[AQW_HOURS * AQW_CH];
    aqw_init(&w);
    uint32_t t = 0;
    for (int k = 0; k <= 24; k++)       /* clean air, then polluted air */
        for (int s = 0; s < 3600; s++, t++) {
            float pm = k < 12 ? 5.0f : 60.0f;
            aqw_push(&w, t, 15.0f, 60.0f, pm, 1.5f * pm);
        }
    CHECK(aqw_window(&w, win), "no window");
    double score;
    int d = aq_predict(&m, win, &score);
    printf("model %s (arch %d): score %.6g, decision %d on half a day at 60 ug/m3\n",
           path, (int)m.arch, score, d);
}


static void test_restore(void) {
    aq_window_t w, r;
    aqw_snapshot_t snap;
    float a[AQW_HOURS * AQW_CH], b[AQW_HOURS * AQW_CH];
    aqw_init(&w);
    uint32_t t = 0, unix0 = 1800000000u;
    for (int k = 0; k < 30 * 3600; k++, t++)
        if (aqw_push(&w, t, 20.0f, 50.0f, 10.0f + (float)(k / 3600), 20.0f) > 0)
            aqw_snapshot(&w, unix0 + t, &snap);
    CHECK(aqw_window(&w, a), "window before power-off");
    /* power off 10 min after the last snapshot, back on 20 min later (30 min outage) */
    CHECK(aqw_restore(&r, &snap, snap.unix_s + 1800u, 5u) == 0, "short outage costs no hour");
    for (uint32_t s2 = 5; s2 < 5 + 1800 + 10; s2++) aqw_push(&r, s2, 20.0f, 50.0f, 40.0f, 20.0f);
    CHECK(aqw_window(&r, b), "window after short outage");
    CHECK(memcmp(a + AQW_CH, b, sizeof(float) * AQW_CH * (AQW_HOURS - 1)) == 0, "history kept");
    /* 3 h 20 min outage: three hours missing, no window until they leave the ring */
    CHECK(aqw_restore(&r, &snap, snap.unix_s + 3 * 3600u + 1200u, 0u) == 3, "three hours missing");
    CHECK(!aqw_window(&r, b), "no window with missing hours");
    /* 25 h outage, corrupted snapshot: discarded */
    CHECK(aqw_restore(&r, &snap, snap.unix_s + 25 * 3600u, 0u) == -1, "long outage discarded");
    snap.hours[3][2] += 1.0f;
    CHECK(aqw_restore(&r, &snap, snap.unix_s + 60u, 0u) == -1, "corrupted snapshot discarded");
}

int main(int argc, char **argv) {
    test_abs_humidity();
    test_hours();
    test_restore();
    for (int i = 1; i < argc; i++) test_model(argv[i]);
    printf("%s (aq_window state %zu bytes, aq_infer static %zu bytes)\n", fails ? "FAILED" : "PASS",
           sizeof(aq_window_t), aq_static_bytes());
    return fails != 0;
}
