#pragma once

// DemoSensor — a fake ISensor that cycles through all AQI levels.
//
// Instead of reading from the real SEN55 hardware, this class generates
// synthetic readings that slowly drift through GOOD → MODERATE → UNHEALTHY
// → VERY_UNHEALTHY → HAZARDOUS → back to GOOD. A small random jitter is
// added each tick so the charts look like real sensor data rather than a
// perfectly smooth ramp.
//
// This is only compiled when DEMO_MODE is defined (the sensa-pcb-demo env).
// The pipeline and BLE code are completely unchanged — they just see an
// ISensor and don't know it's fake.

#include "sensor/sensor.h"
#include <Arduino.h>  // millis(), random()

class DemoSensor : public ISensor {
public:
    DemoSensor() : _pm25(5.0f), _phase(0), _phase_ticks(0) {}

    bool read(Reading& out) override {
        // Advance the PM2.5 target value each tick.
        // Each "phase" holds for ~15 ticks (seconds) before stepping up.
        // After HAZARDOUS we wrap back to GOOD.
        _phase_ticks++;
        if (_phase_ticks >= 15) {
            _phase_ticks = 0;
            _phase = (_phase + 1) % 5;  // 5 phases: GOOD/MOD/UNHEALTHY/VERY/HAZ
        }

        // Target PM2.5 centre for each AQI phase (µg/m³)
        static const float targets[5] = { 5.0f, 20.0f, 80.0f, 175.0f, 260.0f };
        float target = targets[_phase];

        // Nudge current value toward the target, then add a small random jitter.
        // The 0.3 factor controls how quickly we track the target (like an EMA).
        _pm25 += 0.3f * (target - _pm25);
        _pm25 += (random(-100, 100) / 100.0f) * 2.0f;  // ±2 µg/m³ noise
        if (_pm25 < 0.0f) _pm25 = 0.0f;

        // Scale the other PM fractions proportionally to PM2.5 so they look
        // realistic relative to each other.
        out.ts_ms     = millis();
        out.pm2_5     = _pm25;
        out.pm1       = _pm25 * 0.6f;
        out.pm4       = _pm25 * 1.2f;
        out.pm10      = _pm25 * 1.5f;
        out.temp_c    = 22.0f + (random(-50, 50) / 100.0f);  // ~22 °C ±0.5
        out.rh        = 45.0f + (random(-100, 100) / 100.0f); // ~45% ±1
        out.voc_index = 100.0f + (_pm25 * 0.5f);  // loosely correlated with PM
        out.nox_index = 10.0f  + (_pm25 * 0.1f);

        return true;  // always ready — no warm-up needed
    }

private:
    float    _pm25;         // current PM2.5 value, updated each tick
    uint8_t  _phase;        // which AQI band we're currently targeting (0–4)
    uint8_t  _phase_ticks;  // how many ticks we've spent in the current phase
};
