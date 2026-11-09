#pragma once
#include <cmath>
#include <cstdint>

#include "aqi_thresholds.h"
#include "buffer.h"
#include "types.h"

/*
 * extract_features()
 * Computes a FeatureVector from the readings currently stored in a RingBuffer.
 *
 * This is the "signal processing" step of the pipeline. Instead of feeding raw
 * sensor samples directly into the ML model, we summarize the window with
 * statistics (mean, std, etc.) that capture both the level and the variability
 * of the signal. This makes the model smaller and more robust.
 *
 * Parameters:
 *   buf  — the ring buffer holding the current window of readings
 *   out  — the FeatureVector to fill in (passed by reference, modified in place)
 *
 * Returns true on success, false if the buffer has fewer than 2 samples
 * (need at least 2 to compute meaningful statistics).
 *
 * NOTE: This function is a template so it works with any buffer size N.
 * The compiler generates the actual code when you call it with a specific N.
 */
template <uint16_t N>
bool extract_features(const RingBuffer<N>& buf, FeatureVector& out) {
    uint16_t n = buf.count();
    if (n < 2) return false;  // not enough data yet

    // Record the time span of this window
    out.window_start_ms = buf[0].ts_ms;
    out.window_end_ms = buf[n - 1].ts_ms;
    out.sample_count = n;

    // --- Pass 1: means, min, max, % above threshold ---
    float sum_pm = 0, sum_temp = 0, sum_rh = 0, sum_voc = 0, sum_nox = 0;
    float mn = buf[0].pm2_5, mx = buf[0].pm2_5;
    uint16_t above35 = 0;
    for (uint16_t i = 0; i < n; i++) {
        float v = buf[i].pm2_5;
        sum_pm += v;
        sum_temp += buf[i].temp_c;
        sum_rh += buf[i].rh;
        sum_voc += buf[i].voc_index;
        sum_nox += buf[i].nox_index;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (v > AQI_PM25_UNHEALTHY_MIN) above35++;
    }
    out.pm2_5_mean = sum_pm / n;
    out.pm2_5_min = mn;
    out.pm2_5_max = mx;
    out.pm2_5_pct_above_35 = (float)above35 / n;
    out.temp_mean = sum_temp / n;
    out.rh_mean = sum_rh / n;
    out.voc_mean = sum_voc / n;
    out.nox_mean = sum_nox / n;

    // --- Pass 2: std deviation ---
    // Measures spread around the mean. sqrt( sum((x-mean)^2) / n )
    float sq_sum = 0;
    for (uint16_t i = 0; i < n; i++) {
        float d = buf[i].pm2_5 - out.pm2_5_mean;
        sq_sum += d * d;
    }
    // Cast to double before sqrt to avoid GCC 13 ambiguous overload error
    out.pm2_5_std = (float)std::sqrt((double)(sq_sum / n));

    // --- Pass 3: slope via least-squares linear regression ---
    // Fits a line y = a*t + b to the PM2.5 values over time.
    // slope = (n*sum(t*y) - sum(t)*sum(y)) / (n*sum(t^2) - sum(t)^2)
    // Time is in seconds relative to the window start to keep numbers small.
    float sum_t = 0, sum_y = 0, sum_ty = 0, sum_t2 = 0;
    for (uint16_t i = 0; i < n; i++) {
        float t = (buf[i].ts_ms - buf[0].ts_ms) / 1000.0f;  // seconds from window start
        float y = buf[i].pm2_5;
        sum_t += t;
        sum_y += y;
        sum_ty += t * y;
        sum_t2 += t * t;
    }
    float denom = n * sum_t2 - sum_t * sum_t;
    out.pm2_5_slope = (denom != 0.0f) ? (n * sum_ty - sum_t * sum_y) / denom : 0.0f;

    // --- Pass 4: lag-1 autocorrelation ---
    // Measures how much each sample resembles the one before it.
    // Value near 1.0 = smooth/stable signal; near 0 = noisy/random.
    // Formula: sum((x[i]-mean)*(x[i-1]-mean)) / sum((x[i]-mean)^2)
    float num_ac = 0, den_ac = 0;
    for (uint16_t i = 0; i < n; i++) {
        float d = buf[i].pm2_5 - out.pm2_5_mean;
        den_ac += d * d;
        if (i > 0) {
            float d_prev = buf[i - 1].pm2_5 - out.pm2_5_mean;
            num_ac += d * d_prev;
        }
    }
    out.pm2_5_autocorr = (den_ac != 0.0f) ? num_ac / den_ac : 0.0f;

    return true;
}
