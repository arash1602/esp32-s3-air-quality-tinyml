#pragma once
#include "sensor/sensor.h"
#include "types.h"

// millis() is an Arduino built-in. On the host (native unit-test build) it
// doesn't exist, so we provide a simple stub using the C standard library.
#ifdef ARDUINO
#include <Arduino.h>
#else
#include <cstdint>
#include <ctime>
static uint32_t millis() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif

/*
 * MockSensor
 * Returns fake but realistic Reading values so the full pipeline can be tested
 * on the ESP32 without the SEN55 physically connected.
 *
 * PM2.5 slowly oscillates between ~5 and ~40 µg/m³ to exercise multiple
 * classification thresholds during a normal test run.
 */
class MockSensor : public ISensor {
   public:
    bool read(Reading& out) override {
        out.ts_ms = millis();

        _pm2_5 += 0.5f;
        if (_pm2_5 > 40.0f) _pm2_5 = 5.0f;

        out.pm1 = _pm2_5 * 0.7f;
        out.pm2_5 = _pm2_5;
        out.pm4 = _pm2_5 * 1.2f;
        out.pm10 = _pm2_5 * 1.5f;

        out.temp_c = 22.0f;
        out.rh = 50.0f;
        out.voc_index = 100.0f;
        out.nox_index = 10.0f;

        return true;
    }

   private:
    float _pm2_5 = 5.0f;
};
