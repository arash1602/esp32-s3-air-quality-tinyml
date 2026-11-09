/* aq_bench: the board side of tools/hil.py, independent of the framework.
 *
 * The firmware (PlatformIO/Arduino in firmware/, or ESP-IDF in firmware_idf/)
 * supplies the hooks below and calls aq_bench_loop() with the memory-mapped
 * "models" partition. Protocol (integers little-endian, windows float32 [24][5] in
 * physical units, every text reply ends with '\n'):
 *
 *   'I'                       "AQ <n_models> <cpu_mhz>"
 *   'S' u8 idx                selects a model, weights read in place from flash:
 *                             "OK <idx> <arch> <precision> <n_params> <blob_bytes>"
 *   'R' u8 idx                the same with the model copied to RAM first ("ERR -3"
 *                             if it does not fit)
 *   'W' u32 n, n windows      per window: f32 score, u32 microseconds          (binary)
 *   'P' u32 n, n windows      per window: f64 score, u8 decision, u32 microseconds
 *   'Q' window                one window, the same 13-byte reply as 'P'; a window that
 *                             does not arrive within 250 ms is dropped without a reply,
 *                             so the PC can resend after a lost transfer
 *   'X' window                sets the window used by 'B' and 'L'; "X"
 *   'B' u32 reps              "B <reps> <min_us> <median_us> <p99_us> <mean_us>"
 *   'L' u32 seconds           inference back to back, then "L <count> <seconds>"
 *   'Z' u32 seconds           idle, then "Z <seconds>"
 *   'M'                       "M <free_heap> <min_free_heap> <stack_free> <aq_static_bytes>"
 *
 * Timing covers the whole prediction: input scaling or window statistics, the
 * model, and the decision.
 */
#ifndef AQ_BENCH_H
#define AQ_BENCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* hooks the firmware provides */
void     aqb_read(void *buf, size_t n);         /* blocks until n bytes arrived */
int      aqb_read_within(void *buf, size_t n, uint32_t ms);  /* 1 if all n bytes arrived */
void     aqb_write(const void *buf, size_t n);
int64_t  aqb_now_us(void);
void     aqb_sleep_ms(uint32_t ms);              /* idle, CPU in the scheduler   */
int      aqb_cpu_mhz(void);
void     aqb_mem(uint32_t *free_heap, uint32_t *min_free_heap, uint32_t *stack_free);

/* models: "AQIX" u32 count, count x (u32 offset, u32 length), blobs; or NULL */
void aq_bench_loop(const uint8_t *models);

#ifdef __cplusplus
}
#endif
#endif
