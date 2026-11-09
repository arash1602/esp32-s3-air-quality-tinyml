#pragma once
#include "types.h"

/*
 * classify()
 * Maps a FeatureVector to a Classification enum using EPA AQI PM2.5 thresholds.
 * Portable — no Arduino dependency.
 */
Classification classify(const FeatureVector& f);
