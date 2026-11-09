#pragma once

/*
 * EPA AQI Breakpoints for PM2.5 (µg/m³)
 *
 * These are the official U.S. EPA thresholds that define each air quality
 * category. Think of them like the absolute maximum ratings in a datasheet —
 * they come from a regulatory standard, not from our own measurements.
 *
 * Source: U.S. EPA AQI Technical Assistance Document (PM2.5, 24-hour avg)
 * https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
 *
 * Used in:
 *   lib/classify/src/classify.cpp       — comparator chain for classification
 *   lib/features/src/feature_extraction.h — % of samples above UNHEALTHY threshold
 */

// Upper bound of GOOD range (inclusive)

constexpr float AQI_PM25_GOOD_MAX         =   9.0f;

// Upper bound of MODERATE range (inclusive)
constexpr float AQI_PM25_MODERATE_MAX     =  35.4f;

// Upper bound of UNHEALTHY range (inclusive)
constexpr float AQI_PM25_UNHEALTHY_MAX    = 125.4f;

// Upper bound of VERY_UNHEALTHY range (inclusive)
constexpr float AQI_PM25_VERY_UNHEALTHY_MAX = 225.4f;

// Threshold used as a feature: fraction of window samples above this value.
// Matches the bottom of the UNHEALTHY band (35.5 µg/m³).
constexpr float AQI_PM25_UNHEALTHY_MIN    =  35.5f;
