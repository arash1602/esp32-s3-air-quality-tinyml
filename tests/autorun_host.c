/* Host check of the stand-alone energy run: virtual clock; every marker edge is
 * written to stdout as "t_us level". A prediction takes a fake time per model. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "aq_bench.h"
void aq_autorun(const uint8_t *models, void (*mark)(int), uint32_t busy_s, uint32_t idle_s);
static int64_t now;
static int level;
static int64_t t_high;
int64_t aqb_now_us(void) { return now++; }
void aqb_sleep_ms(uint32_t ms) {
    static int ended;
    if (ms == 3000 && level == 1) ended = 1;            /* end marker */
    else if (ended && level == 0) { fflush(stdout); exit(0); }
    now += (int64_t)ms * 1000;
}
static void mark(int on) {
    if (on == 0 && level == 1 && now - t_high < 100) now += 1500;  /* fake prediction time */
    if (on == 1) t_high = now;
    level = on; printf("%lld %d\n", (long long)now, on);
}
void aqb_read(void *b, size_t n) { (void)b; (void)n; }
int aqb_read_within(void *b, size_t n, uint32_t ms) { (void)b; (void)n; (void)ms; return 0; }
void aqb_write(const void *b, size_t n) { (void)b; (void)n; }
int aqb_cpu_mhz(void) { return 240; }
void aqb_mem(uint32_t *a, uint32_t *b, uint32_t *c) { *a = *b = *c = 0; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    uint8_t *buf = malloc(n); if (fread(buf, 1, n, f) != (size_t)n) return 1; fclose(f);
    aq_autorun(buf, mark, 1, 1);
    return 0;
}
