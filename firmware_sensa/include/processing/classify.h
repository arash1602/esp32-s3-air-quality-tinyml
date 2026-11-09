#pragma once
#include "types.h"

/*
 * classify()
 * Maps a FeatureVector to a Classification enum using EPA AQI PM2.5 thresholds.
 *
 * This is the baseline classifier — no ML required. It looks at the mean PM2.5
 * value over the window and returns the matching exposure category.
 *
 * Parameters:
 *   f — the feature vector computed from the current window of readings
 *
 * Returns the Classification label for the current window.
 */
Classification classify(const FeatureVector& f);
