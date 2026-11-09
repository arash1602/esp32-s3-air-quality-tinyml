#pragma once
#include <cstdint>

/*
 * =========================
 * Raw Sensor Reading
 * =========================
 * Represents a single timestamped sample from the SEN55
 * (or a mock sensor during development).
 */
struct Reading {
    uint32_t ts_ms;  // timestamp (milliseconds since boot)

    // Particulate Matter (ug/m^3)
    float pm1;
    float pm2_5;
    float pm4;
    float pm10;

    // Environmental Readings
    float temp_c;  // temp (C)
    float rh;      // relative humidity (%)

    // Gas indicies
    float voc_index;  // 1-500
    float nox_index;  // 1-500
};

/*
 * =========================
 * Air Quality Classification
 * =========================
 * User-facing exposure level.
 */
enum class Classification : uint8_t {
    GOOD = 0,
    MODERATE = 1,
    UNHEALTHY = 2,
    VERY_UNHEALTHY = 3,
    HAZARDOUS = 4,
    UNKNOWN = 255
};

/*
 * =========================
 * Feature Vector
 * =========================
 * Aggregated statistics over a rolling window
 * (ex:, 60 seconds of 1 Hz data).
 */
struct FeatureVector {
    uint32_t window_start_ms;
    uint32_t window_end_ms;
    uint16_t sample_count;

    // PM2.5 statistics
    float pm2_5_mean;
    float pm2_5_std;
    float pm2_5_min;           // lowest value in the window
    float pm2_5_max;           // highest value in the window
    float pm2_5_slope;         // linear trend in µg/m³ per second (positive = rising)
    float pm2_5_pct_above_35;  // fraction of samples above 35.5 µg/m³ (0.0–1.0)
    float pm2_5_autocorr;      // lag-1 autocorrelation (how much each sample predicts the next)

    // Environmental averages
    float temp_mean;
    float rh_mean;

    // Gas averages
    float voc_mean;
    float nox_mean;
};
