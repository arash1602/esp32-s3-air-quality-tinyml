/* aq_bench, ESP-IDF variant (the Sensa firmware itself uses PlatformIO/Arduino; see
 * ../firmware). Board side of tools/hil.py: protocol in ../../bench/aq_bench.h. */
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_partition.h"
#include "sdkconfig.h"
#include "aq_bench.h"

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
static void io_init(void) {
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 4096; cfg.tx_buffer_size = 4096;
    usb_serial_jtag_driver_install(&cfg);
}
void aqb_read(void *buf, size_t n) {
    uint8_t *p = buf;
    while (n) { int r = usb_serial_jtag_read_bytes(p, n, portMAX_DELAY); if (r > 0) { p += r; n -= r; } }
}
int aqb_read_within(void *buf, size_t n, uint32_t ms) {
    uint8_t *p = buf;
    int64_t end = esp_timer_get_time() + (int64_t)ms * 1000;
    while (n) {
        int64_t left = end - esp_timer_get_time();
        if (left <= 0) return 0;
        int r = usb_serial_jtag_read_bytes(p, n, pdMS_TO_TICKS(left / 1000 + 1));
        if (r > 0) { p += r; n -= r; }
    }
    return 1;
}
void aqb_write(const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n) { int w = usb_serial_jtag_write_bytes(p, n, portMAX_DELAY); if (w > 0) { p += w; n -= w; } }
}
#else
#include "driver/uart.h"
#define AQ_UART UART_NUM_0
static void io_init(void) {
    uart_config_t c = { .baud_rate = 921600, .data_bits = UART_DATA_8_BITS,
                        .parity = UART_PARITY_DISABLE, .stop_bits = UART_STOP_BITS_1,
                        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE, .source_clk = UART_SCLK_DEFAULT };
    uart_driver_install(AQ_UART, 8192, 8192, 0, NULL, 0);
    uart_param_config(AQ_UART, &c);
}
void aqb_read(void *buf, size_t n) {
    uint8_t *p = buf;
    while (n) { int r = uart_read_bytes(AQ_UART, p, n, portMAX_DELAY); if (r > 0) { p += r; n -= r; } }
}
int aqb_read_within(void *buf, size_t n, uint32_t ms) {
    uint8_t *p = buf;
    int64_t end = esp_timer_get_time() + (int64_t)ms * 1000;
    while (n) {
        int64_t left = end - esp_timer_get_time();
        if (left <= 0) return 0;
        int r = uart_read_bytes(AQ_UART, p, n, pdMS_TO_TICKS(left / 1000 + 1));
        if (r > 0) { p += r; n -= r; }
    }
    return 1;
}
void aqb_write(const void *buf, size_t n) { uart_write_bytes(AQ_UART, buf, n); }
#endif

int64_t aqb_now_us(void) { return esp_timer_get_time(); }
void aqb_sleep_ms(uint32_t ms) { vTaskDelay(pdMS_TO_TICKS(ms)); }
int aqb_cpu_mhz(void) { return CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ; }
void aqb_mem(uint32_t *fh, uint32_t *mfh, uint32_t *st) {
    *fh = esp_get_free_heap_size();
    *mfh = esp_get_minimum_free_heap_size();
    *st = (uint32_t)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t));
}

void app_main(void) {
    io_init();
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "models");
    esp_partition_mmap_handle_t h;
    const void *ptr = NULL;
    if (!part || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) != ESP_OK)
        ptr = NULL;
    vTaskDelay(pdMS_TO_TICKS(200));
    aq_bench_loop(ptr);
}
