#include "ble/ble_service.h"

#include <Arduino.h>
#include <BLE2902.h>
#include <BLECharacteristic.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

// UUIDs identify this service and its characteristics to BLE clients.
// These are randomly generated — they just need to be unique and consistent
// between the firmware and whatever app reads them.
#define SERVICE_UUID         "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_UUID_PM25       "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHAR_UUID_LABEL      "beb5483e-36e1-4688-b7f5-ea07361b26a9"
#define CHAR_UUID_READING    "beb5483e-36e1-4688-b7f5-ea07361b26aa"
#define CHAR_UUID_EXCEED     "beb5483e-36e1-4688-b7f5-ea07361b26ab"   // added: WHO warning

// Pointers to characteristics so ble_notify() can update them later.
static BLECharacteristic* s_pm25_char    = nullptr;
static BLECharacteristic* s_label_char   = nullptr;
static BLECharacteristic* s_reading_char = nullptr;
#ifdef AQ_ENABLE
static BLECharacteristic* s_exceed_char  = nullptr;
#endif

// Tracks whether a client is currently connected.
// We only bother sending notifications when someone is listening.
static bool s_connected = false;

// BLE connection callbacks — called automatically by the BLE stack
// when a client connects or disconnects.
class ConnectionCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer*)    override { s_connected = true;  Serial.println("[BLE] Client connected."); }
    void onDisconnect(BLEServer* srv) override {
        s_connected = false;
        Serial.println("[BLE] Client disconnected. Restarting advertising...");
        // Restart advertising so a new client can connect after the previous one leaves.
        srv->startAdvertising();
    }
};

void ble_init() {
    // Initialize the BLE stack and set the device name that appears when scanning.
    BLEDevice::init("Sensa");

    // Create the GATT server and register our connection callbacks.
    BLEServer* server = BLEDevice::createServer();
    server->setCallbacks(new ConnectionCallbacks());

    // Create the Air Quality service.
    BLEService* service = server->createService(SERVICE_UUID);

    // PM2.5 characteristic — kept for backwards compatibility with the Python client.
    s_pm25_char = service->createCharacteristic(
        CHAR_UUID_PM25,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    s_pm25_char->addDescriptor(new BLE2902());

    // Classification label characteristic.
    s_label_char = service->createCharacteristic(
        CHAR_UUID_LABEL,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    s_label_char->addDescriptor(new BLE2902());

    // Full Reading characteristic — sends all sensor fields as a packed 36-byte payload.
    // Layout: uint32 ts_ms, then 8x float (pm1, pm2_5, pm4, pm10, temp_c, rh, voc, nox).
    // The web app unpacks these bytes using a DataView with little-endian offsets.
    s_reading_char = service->createCharacteristic(
        CHAR_UUID_READING,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    s_reading_char->addDescriptor(new BLE2902());

#ifdef AQ_ENABLE
    // WHO 24-h exceedance warning: uint8 decision (1 = exceedance expected), float score.
    s_exceed_char = service->createCharacteristic(
        CHAR_UUID_EXCEED,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    s_exceed_char->addDescriptor(new BLE2902());
#endif

    // Start the service, then begin advertising so clients can find the device.
    service->start();
    BLEDevice::startAdvertising();

    Serial.println("[BLE] Advertising as 'Sensa'.");
}

void ble_notify(const Reading& reading, Classification result) {
    // No client connected — nothing to send.
    if (!s_connected) return;

    // --- PM2.5 (backwards compat) ---
    float pm2_5 = reading.pm2_5;
    s_pm25_char->setValue(reinterpret_cast<uint8_t*>(&pm2_5), sizeof(pm2_5));
    s_pm25_char->notify();

    // --- Label ---
    uint8_t label = static_cast<uint8_t>(result);
    s_label_char->setValue(&label, sizeof(label));
    s_label_char->notify();

    // --- Full Reading (packed struct, 36 bytes) ---
    // We copy field-by-field into a flat byte buffer to avoid any struct padding
    // surprises between the ESP32 compiler and the browser's DataView.
    uint8_t buf[36];
    uint32_t ts = reading.ts_ms;
    float fields[8] = {
        reading.pm1, reading.pm2_5, reading.pm4, reading.pm10,
        reading.temp_c, reading.rh, reading.voc_index, reading.nox_index
    };
    memcpy(buf,      &ts,     4);
    memcpy(buf + 4,  fields, 32);
    s_reading_char->setValue(buf, sizeof(buf));
    s_reading_char->notify();
}

#ifdef AQ_ENABLE
void ble_notify_exceedance(int8_t exceed, float score) {
    if (!s_exceed_char) return;
    uint8_t buf[5];
    buf[0] = (uint8_t)exceed;
    memcpy(buf + 1, &score, 4);
    s_exceed_char->setValue(buf, sizeof(buf));
    if (s_connected) s_exceed_char->notify();
}
#endif
