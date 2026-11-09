/* aq_window: turns the Sensa firmware's 1 Hz SEN55 readings into the input of the
 * WHO-threshold classifiers: the last 24 hourly means of temperature, relative
 * humidity, absolute humidity, PM2.5 and PM10 (paper 1's channel order).
 *
 * As in paper 1's preparation of SensEURCity: absolute humidity is derived from each
 * reading's temperature and humidity (same Magnus formula as the dataset), invalid
 * readings are dropped before averaging, an hour's value is the mean of its valid
 * readings, and a window is complete only when all 24 hours have at least one valid
 * reading (nothing is imputed). Hours count from the first reading (no clock needed).
 *
 * Portable C99, no allocation: about 600 bytes of state.
 */
#ifndef AQ_WINDOW_H
#define AQ_WINDOW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AQW_HOURS 24
#define AQW_CH    5          /* T (degC), RH (%), AH (g/m3), PM2.5, PM10 (ug/m3) */

typedef struct {
    float    hours[AQW_HOURS][AQW_CH];  /* ring of hourly means          */
    uint8_t  valid[AQW_HOURS];
    uint32_t head;                       /* slot of the next hour to close */
    double   sum[AQW_CH];
    uint32_t n;                          /* valid readings this hour       */
    uint32_t hour_start_s;               /* time the current hour began    */
    uint32_t hours_closed;
    int      started;
} aq_window_t;

/* Absolute humidity (g/m3) from temperature (degC) and relative humidity (%). */
float aq_abs_humidity(float temp_c, float rh);

void aqw_init(aq_window_t *w);

/* Add one reading taken at t_s seconds (any monotonic clock, e.g. millis()/1000).
 * Returns the number of hours closed by this call (0 almost always; more than 1 only
 * after a gap of whole hours, which are recorded as missing). */
int aqw_push(aq_window_t *w, uint32_t t_s, float temp_c, float rh, float pm2_5, float pm10);

/* 1 if the last 24 closed hours are all valid; then out (24 x 5, oldest first) is
 * filled with the window the classifiers expect. */
int aqw_window(const aq_window_t *w, float *out);

/* Keeping the history across a power cycle (recharge, reset). The firmware saves a
 * snapshot to non-volatile storage whenever an hour closes (aqw_push returned > 0),
 * with the wall-clock time of that moment (e.g. set over BLE by the dashboard), and
 * restores it at start-up with the current wall-clock time. Hours that passed while
 * the device was off are recorded as missing, so a warning resumes only once the
 * last 24 hours are complete again; after an outage of 24 h or more, or without a
 * valid clock, the snapshot is discarded and the device starts over. */
typedef struct {
    uint32_t magic, unix_s;              /* time the snapshot was taken        */
    float    hours[AQW_HOURS][AQW_CH];
    uint8_t  valid[AQW_HOURS];
    uint32_t head, hours_closed, check;
} aqw_snapshot_t;

void aqw_snapshot(const aq_window_t *w, uint32_t unix_s, aqw_snapshot_t *snap);
/* t_s: the monotonic clock aqw_push will be called with. Returns the number of hours
 * the outage cost (recorded as missing), or -1 if the snapshot was discarded. */
int  aqw_restore(aq_window_t *w, const aqw_snapshot_t *snap, uint32_t unix_now_s, uint32_t t_s);

#ifdef __cplusplus
}
#endif
#endif
