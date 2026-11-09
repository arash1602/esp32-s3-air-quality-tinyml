#include "sensor/sen55_sensor.h"

#include <Arduino.h>

bool Sen55Sensor::begin() {
    // Initialize the I2C bus on the pins wired to the SEN55.
    // Wire.begin(SDA, SCL) — SDA carries data, SCL is the clock.
    Wire.begin(SEN55_SDA_PIN, SEN55_SCL_PIN);

    // Hand the I2C bus to the Sensirion library.
    _sen5x.begin(Wire);

    // Reset the sensor to a known state in case it was left running
    // from a previous power cycle.
    uint16_t err = _sen5x.deviceReset();
    if (err) {
        Serial.print("[SEN55] Reset failed, error: ");
        Serial.println(err);
        return false;
    }

    // Small delay after reset — the SEN55 datasheet recommends waiting
    // at least 100 ms before sending further commands.
    delay(100);

    // Tell the sensor to start continuous measurement.
    // After this command the sensor takes one reading per second internally.
    err = _sen5x.startMeasurement();
    if (err) {
        Serial.print("[SEN55] startMeasurement failed, error: ");
        Serial.println(err);
        return false;
    }

    _started = true;
    Serial.println("[SEN55] Started. Discarding warm-up readings...");
    return true;
}

bool Sen55Sensor::read(Reading& out) {
    if (!_started) return false;

    // These are the raw scaled integers the Sensirion library returns.
    // The library divides by the right scale factor to give physical units.
    float pm1, pm2_5, pm4, pm10;
    float rh, temp;
    float voc, nox;

    // isDataReady() returns true once the sensor has a fresh 1 Hz sample.
    // Calling readMeasuredValues() before data is ready returns stale data.
    bool data_ready = false;
    uint16_t err = _sen5x.readDataReady(data_ready);
    if (err || !data_ready) return false;

    err = _sen5x.readMeasuredValues(pm1, pm2_5, pm4, pm10, rh, temp, voc, nox);
    if (err) {
        Serial.print("[SEN55] readMeasuredValues failed, error: ");
        Serial.println(err);
        return false;
    }

    // Discard the first SEN55_WARMUP_SAMPLES readings.
    // The laser particle counter needs ~30 s to reach stable operating temperature.
    if (_warmup_count < SEN55_WARMUP_SAMPLES) {
        _warmup_count++;
        Serial.print("[SEN55] Warming up (");
        Serial.print(_warmup_count);
        Serial.print("/");
        Serial.print(SEN55_WARMUP_SAMPLES);
        Serial.println(")");
        return false;  // signal "not ready yet" to the pipeline
    }

    // Print raw sensor values for debugging
    Serial.print("[SEN55] PM1="); Serial.print(pm1);
    Serial.print(" PM2.5=");      Serial.print(pm2_5);
    Serial.print(" PM4=");        Serial.print(pm4);
    Serial.print(" PM10=");       Serial.print(pm10);
    Serial.print(" Temp=");       Serial.print(temp);
    Serial.print(" RH=");         Serial.print(rh);
    Serial.print(" VOC=");        Serial.print(voc);
    Serial.print(" NOx=");        Serial.println(nox);

    out.ts_ms    = millis();
    out.pm1      = pm1;
    out.pm2_5    = pm2_5;
    out.pm4      = pm4;
    out.pm10     = pm10;
    out.temp_c   = temp;
    out.rh       = rh;
    out.voc_index = voc;
    out.nox_index = nox;

    return true;
}
