#pragma once
#include "types.h"

/*
 * ble_init()
 *
 * Sets up the BLE GATT server with a single "Air Quality" service.
 * Call once in setup() after Serial is ready.
 *
 * The device will advertise as "Sensa" and a phone/PC can connect
 * to read or receive notifications on the characteristics below.
 *
 * Service UUID:       4fafc201-1fb5-459e-8fcc-c5c9c331914b
 * Characteristics:
 *   PM2.5  (notify):  beb5483e-36e1-4688-b7f5-ea07361b26a8  — float, µg/m³
 *   Label  (notify):  beb5483e-36e1-4688-b7f5-ea07361b26a9  — uint8 (Classification enum)
 *   Reading(notify):  beb5483e-36e1-4688-b7f5-ea07361b26aa  — packed Reading struct (36 bytes)
 *
 * Reading payload layout (little-endian):
 *   [0..3]   uint32  ts_ms
 *   [4..7]   float   pm1
 *   [8..11]  float   pm2_5
 *   [12..15] float   pm4
 *   [16..19] float   pm10
 *   [20..23] float   temp_c
 *   [24..27] float   rh
 *   [28..31] float   voc_index
 *   [32..35] float   nox_index
 */
void ble_init();

/*
 * ble_notify(reading, result)
 *
 * Pushes the full sensor reading and classification label to any connected BLE client.
 * Call this once per pipeline_tick() that returns a valid result.
 *
 * reading — full Reading struct from the sensor
 * result  — Classification enum value
 */
void ble_notify(const Reading& reading, Classification result);

#ifdef AQ_ENABLE
// Added for the hourly WHO-exceedance warning: decision and score of the last prediction.
void ble_notify_exceedance(int8_t exceed, float score);
#endif
