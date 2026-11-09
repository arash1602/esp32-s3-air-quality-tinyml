/* aq_window.c -- see aq_window.h */
#include "aq_window.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

float aq_abs_humidity(float temp_c, float rh) {
    /* SensEURCity's Absolute_humidity column, reproduced to 1e-13 g/m3 */
    double t = temp_c;
    return (float)(6.112 * exp(17.67 * t / (t + 243.5)) * rh * 2.1674 / (273.15 + t));
}

void aqw_init(aq_window_t *w) { memset(w, 0, sizeof *w); }

static void close_hour(aq_window_t *w) {
    uint32_t s = w->head;
    if (w->n > 0) {
        for (int c = 0; c < AQW_CH; c++) w->hours[s][c] = (float)(w->sum[c] / w->n);
        w->valid[s] = 1;
    } else {
        memset(w->hours[s], 0, sizeof w->hours[s]);
        w->valid[s] = 0;
    }
    w->head = (w->head + 1) % AQW_HOURS;
    w->hours_closed++;
    memset(w->sum, 0, sizeof w->sum);
    w->n = 0;
}

int aqw_push(aq_window_t *w, uint32_t t_s, float temp_c, float rh, float pm2_5, float pm10) {
    int closed = 0;
    if (!w->started) { w->started = 1; w->hour_start_s = t_s; }
    while (t_s - w->hour_start_s >= 3600u) {
        close_hour(w);
        w->hour_start_s += 3600u;
        closed++;
        if (closed > AQW_HOURS) {       /* long gap or clock wrap: every hour missing */
            w->hour_start_s = t_s;
            break;
        }
    }
    if (isfinite(temp_c) && isfinite(rh) && isfinite(pm2_5) && isfinite(pm10) &&
        rh >= 0.0f && rh <= 100.0f && pm2_5 >= 0.0f && pm10 >= 0.0f) {
        float v[AQW_CH] = { temp_c, rh, aq_abs_humidity(temp_c, rh), pm2_5, pm10 };
        for (int c = 0; c < AQW_CH; c++) w->sum[c] += v[c];
        w->n++;
    }
    return closed;
}

int aqw_window(const aq_window_t *w, float *out) {
    if (w->hours_closed < AQW_HOURS) return 0;
    for (int k = 0; k < AQW_HOURS; k++) {
        uint32_t s = (w->head + (uint32_t)k) % AQW_HOURS;   /* oldest first */
        if (!w->valid[s]) return 0;
        memcpy(out + k * AQW_CH, w->hours[s], sizeof w->hours[s]);
    }
    return 1;
}

#define AQW_MAGIC 0x41515731u   /* "AQW1" */

static uint32_t snap_check(const aqw_snapshot_t *s) {
    const uint8_t *p = (const uint8_t *)s;
    uint32_t h = 2166136261u;                   /* FNV-1a over all but the check */
    for (size_t i = 0; i < offsetof(aqw_snapshot_t, check); i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

void aqw_snapshot(const aq_window_t *w, uint32_t unix_s, aqw_snapshot_t *snap) {
    memset(snap, 0, sizeof *snap);
    snap->magic = AQW_MAGIC;
    snap->unix_s = unix_s;
    memcpy(snap->hours, w->hours, sizeof snap->hours);
    memcpy(snap->valid, w->valid, sizeof snap->valid);
    snap->head = w->head;
    snap->hours_closed = w->hours_closed;
    snap->check = snap_check(snap);
}

int aqw_restore(aq_window_t *w, const aqw_snapshot_t *snap, uint32_t unix_now_s, uint32_t t_s) {
    aqw_init(w);
    if (snap->magic != AQW_MAGIC || snap->check != snap_check(snap) || snap->head >= AQW_HOURS ||
        unix_now_s < snap->unix_s || unix_now_s - snap->unix_s >= AQW_HOURS * 3600u)
        return -1;
    uint32_t elapsed = unix_now_s - snap->unix_s;
    memcpy(w->hours, snap->hours, sizeof w->hours);
    memcpy(w->valid, snap->valid, sizeof w->valid);
    w->head = snap->head;
    w->hours_closed = snap->hours_closed;
    uint32_t missed = elapsed / 3600u;          /* whole hours the device was off */
    for (uint32_t k = 0; k < missed; k++) close_hour(w);   /* n == 0: recorded missing */
    /* the current hour started at snapshot + missed hours; keep that alignment */
    w->started = 1;
    w->hour_start_s = t_s - (elapsed % 3600u);
    return (int)missed;
}
