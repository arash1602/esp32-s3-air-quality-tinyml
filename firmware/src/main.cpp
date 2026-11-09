/* aq_bench for the Sensa board (PlatformIO, Arduino-ESP32), the board side of
 * tools/hil.py. The model blobs live in the "models" flash partition and are
 * memory-mapped, so weights are read in place from flash, as they would be in the
 * Sensa firmware. The console is the ESP32-S3's built-in USB (USB CDC on boot,
 * as in the Sensa firmware); the host's baud-rate setting does not matter.
 *
 * The sensor, BLE radio and alert outputs are left off so that the measurements
 * isolate the classifier.
 */
#include <Arduino.h>
#include <esp_partition.h>
#include <esp_timer.h>
#include <esp_idf_version.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "aq_bench.h"

#ifdef AQ_AUTORUN
#include <soc/gpio_reg.h>
#include <driver/gpio.h>
#ifndef AQ_MARK_GPIO
#define AQ_MARK_GPIO 4
#endif
extern "C" void aq_autorun(const uint8_t *models, void (*mark)(int), uint32_t busy_s,
                           uint32_t idle_s);
/* direct register writes: a marker edge costs a few CPU cycles */
static void mark_pin(int on) {
    if (on) REG_WRITE(GPIO_OUT_W1TS_REG, 1u << AQ_MARK_GPIO);
    else REG_WRITE(GPIO_OUT_W1TC_REG, 1u << AQ_MARK_GPIO);
}
#endif

extern "C" {

void aqb_read(void *buf, size_t n) {
    uint8_t *p = static_cast<uint8_t *>(buf);
    while (n) {
        int a = Serial.available();
        if (a <= 0) { delay(1); continue; }   /* lets the idle task run while waiting */
        size_t k = Serial.readBytes(p, n < (size_t)a ? n : (size_t)a);
        p += k; n -= k;
    }
}

int aqb_read_within(void *buf, size_t n, uint32_t ms) {
    uint8_t *p = static_cast<uint8_t *>(buf);
    uint32_t t0 = millis();
    while (n) {
        int a = Serial.available();
        if (a <= 0) {
            if (millis() - t0 > ms) return 0;
            delay(1);
            continue;
        }
        size_t k = Serial.readBytes(p, n < (size_t)a ? n : (size_t)a);
        p += k; n -= k;
    }
    return 1;
}

void aqb_write(const void *buf, size_t n) {
    const uint8_t *p = static_cast<const uint8_t *>(buf);
    while (n) {
        size_t k = Serial.write(p, n);
        p += k; n -= k;
        if (!k) delay(1);
    }
    Serial.flush();
}

int64_t aqb_now_us(void) { return esp_timer_get_time(); }
void aqb_sleep_ms(uint32_t ms) { vTaskDelay(pdMS_TO_TICKS(ms)); }
int aqb_cpu_mhz(void) { return (int)getCpuFrequencyMhz(); }

void aqb_mem(uint32_t *free_heap, uint32_t *min_free_heap, uint32_t *stack_free) {
    *free_heap = ESP.getFreeHeap();
    *min_free_heap = ESP.getMinFreeHeap();
    *stack_free = uxTaskGetStackHighWaterMark(NULL);   /* bytes on ESP32 FreeRTOS */
}

}  // extern "C"

static const uint8_t *map_models(void) {
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "models");
    if (!part) return nullptr;
    const void *ptr = nullptr;
#if ESP_IDF_VERSION_MAJOR >= 5
    esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) != ESP_OK)
        return nullptr;
#else
    spi_flash_mmap_handle_t h;
    if (esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &ptr, &h) != ESP_OK)
        return nullptr;
#endif
    return static_cast<const uint8_t *>(ptr);
}

void setup() {
    setCpuFrequencyMhz(240);
#ifdef AQ_AUTORUN
    /* stand-alone energy run for the PPK2: no USB, phases marked on AQ_MARK_GPIO */
    pinMode(AQ_MARK_GPIO, OUTPUT);
    digitalWrite(AQ_MARK_GPIO, LOW);
    aq_autorun(map_models(), mark_pin, 8, 3);   /* never returns */
#endif
    Serial.setRxBufferSize(4096);
    Serial.setTimeout(100000);
    Serial.begin(921600);
    uint32_t t = millis();
    while (!Serial && millis() - t < 3000) delay(10);
    delay(200);
    aq_bench_loop(map_models());   /* never returns */
}

void loop() {}
