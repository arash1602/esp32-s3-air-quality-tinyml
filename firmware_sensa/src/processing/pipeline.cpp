#include "processing/pipeline.h"

#include "buffer.h"
#include "classify.h"
#include "feature_extraction.h"
#include "ml/calibrate.h"

// Window size: 10 samples at 1 Hz = 10 seconds of data.
// Adjust N to change the window length.
static RingBuffer<10> s_buf;

// Pointer to whichever sensor was passed to pipeline_init()
static ISensor* s_sensor = nullptr;

// Most recent valid reading (post warm-up), exposed via pipeline_last_reading()
static Reading s_last_reading{};

void pipeline_init(ISensor* sensor) {
    s_sensor = sensor;
    s_buf.clear();
    s_last_reading = Reading{};

    // Load the on-device PM2.5 calibration model (TensorFlow Lite Micro).
    // If the model has not been trained/exported yet, this returns false and
    // the pipeline simply uses raw sensor values — see src/ml/calibrate.cpp.
    calibrate_init();
}

Classification pipeline_tick() {
    if (!s_sensor) return Classification::UNKNOWN;

    // Read one sample from the sensor and add it to the rolling window
    Reading r{};
    if (!s_sensor->read(r)) return Classification::UNKNOWN;

    // Apply the ML calibration correction to PM2.5 BEFORE the reading enters
    // the buffer. Everything downstream — feature extraction, classification,
    // and the BLE notification — therefore operates on the corrected value.
    // Only pm2_5 is corrected; that is the single quantity the model predicts.
    // If no model is loaded, calibrate_pm25() returns the raw value unchanged.
    r.pm2_5 = calibrate_pm25(r);

    s_last_reading = r;  // save for pipeline_last_reading()
    s_buf.push(r);

    // Wait until the buffer has a full window before classifying.
    // Early readings would be based on too few samples to be meaningful.
    if (!s_buf.full()) return Classification::UNKNOWN;

    // Summarise the window into a feature vector, then classify
    FeatureVector f{};
    if (!extract_features(s_buf, f)) return Classification::UNKNOWN;

    return classify(f);
}

Reading pipeline_last_reading() {
    return s_last_reading;
}
