#pragma once
#include "types.h"

/*
 * ISensor — common interface for any sensor (real or mock).
 *
 * By defining a shared interface, pipeline.cpp can call either the real SEN55
 * driver or the mock sensor without knowing which one it's talking to. This is
 * the same idea as a connector standard in hardware: the pipeline only cares
 * about the "pinout" (read), not what's plugged in.
 *
 * To implement a sensor, inherit from this class and override read().
 */
class ISensor {
   public:
    // Attempt to read one sample from the sensor.
    // Fills `out` with the latest values and returns true on success.
    // Returns false if the sensor is not ready or the read failed.
    virtual bool read(Reading& out) = 0;

    virtual ~ISensor() = default;
};
