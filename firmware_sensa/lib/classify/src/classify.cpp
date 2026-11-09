#include "classify.h"

#include "aqi_thresholds.h"

Classification classify(const FeatureVector& f) {
    float pm = f.pm2_5_mean;

    // Guard against invalid readings (negative or NaN)
    if (pm < 0.0f || pm != pm) return Classification::UNKNOWN;

    // EPA AQI PM2.5 breakpoints — thresholds defined in include/aqi_thresholds.h
    if (pm <= AQI_PM25_GOOD_MAX)            return Classification::GOOD;
    if (pm <= AQI_PM25_MODERATE_MAX)        return Classification::MODERATE;
    if (pm <= AQI_PM25_UNHEALTHY_MAX)       return Classification::UNHEALTHY;
    if (pm <= AQI_PM25_VERY_UNHEALTHY_MAX)  return Classification::VERY_UNHEALTHY;
    return Classification::HAZARDOUS;
}
