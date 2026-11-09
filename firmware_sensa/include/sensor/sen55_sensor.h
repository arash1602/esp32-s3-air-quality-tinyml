#pragma once
#include <Arduino.h>
#include <SensirionI2CSen5x.h>
#include <Wire.h>

#include "sensor/sensor.h"
#include "types.h"

// I2C pin assignments.
// The SENSA_PCB flag is set in platformio.ini for the custom PCB build.
// Both boards happen to use the same GPIO8/9 for I2C, but this block makes
// the intent explicit and makes it easy to change either target independently.
#ifdef SENSA_PCB
  // Custom Sensa PCB: SDA=GPIO8, SCL=GPIO9 (pulled up to 3.3V with 10k resistors)
  #define SEN55_SDA_PIN 8
  #define SEN55_SCL_PIN 9
#else
  // RYMCU ESP32-S3-DevKitC-1 dev board
  #define SEN55_SDA_PIN 8
  #define SEN55_SCL_PIN 9
#endif

// The SEN55 I2C address is fixed at 0x69 (set by Sensirion in hardware).
#define SEN55_I2C_ADDR 0x69

// How many readings to discard after startup.
// The SEN55 datasheet says the laser particle counter needs ~30 s to stabilize.
// At 1 Hz that's 30 readings. We discard them so bad data never enters the pipeline.
#define SEN55_WARMUP_SAMPLES 30

/*
 * Sen55Sensor
 *
 * Implements ISensor for the real Sensirion SEN55 hardware.
 * Uses the official Sensirion Arduino library (SensirionI2CSen5x).
 *
 * Usage:
 *   Sen55Sensor sensor;
 *   sensor.begin();          // call once in setup()
 *   sensor.read(reading);    // call at 1–2 Hz in loop()
 */
class Sen55Sensor : public ISensor {
   public:
    // Initializes I2C, starts the SEN55 measurement mode.
    // Returns true on success, false if the sensor didn't respond.
    bool begin();

    // Reads one sample from the SEN55.
    // Fills `out` with PM, VOC, NOx, temp, and RH values.
    // Returns false if the sensor isn't ready yet (warm-up) or a read error occurred.
    bool read(Reading& out) override;

   private:
    SensirionI2CSen5x _sen5x;
    uint16_t _warmup_count = 0;  // counts discarded warm-up readings
    bool _started = false;       // true after begin() succeeds
};
