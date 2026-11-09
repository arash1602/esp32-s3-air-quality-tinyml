/* Runs the board-side benchmark loop (bench/aq_bench.c) on a PC behind a
 * pseudo-terminal, so tools/hil.py can be tested end to end without the board:
 *
 *   bench_host models.bin          prints the pty path, e.g. /dev/pts/3
 *   python tools/hil.py --port /dev/pts/3 fidelity
 */
#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE 600
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "aq_bench.h"

static int g_fd = -1;
int64_t aqb_now_us(void);

void aqb_read(void *buf, size_t n) {
    unsigned char *p = buf;
    while (n) {
        ssize_t r = read(g_fd, p, n);
        if (r > 0) { p += r; n -= (size_t)r; }
        else if (r < 0) usleep(1000);
        else usleep(1000);
    }
}
int aqb_read_within(void *buf, size_t n, uint32_t ms) {
    unsigned char *p = buf;
    int64_t end = aqb_now_us() + (int64_t)ms * 1000;
    while (n) {
        if (aqb_now_us() > end) return 0;
        ssize_t r = read(g_fd, p, n);
        if (r > 0) { p += r; n -= (size_t)r; } else usleep(200);
    }
    return 1;
}
/* fault injection for tests: AQ_DROP_EVERY=k drops every k-th 13-byte result */
static long g_drop_every = -1, g_results;
void aqb_write(const void *buf, size_t n) {
    if (g_drop_every < 0) { const char *e = getenv("AQ_DROP_EVERY"); g_drop_every = e ? atol(e) : 0; }
    if (n == 13 && g_drop_every > 0 && ++g_results % g_drop_every == 0) return;
    const unsigned char *p = buf;
    while (n) {
        ssize_t w = write(g_fd, p, n);
        if (w > 0) { p += w; n -= (size_t)w; } else usleep(1000);
    }
}
int64_t aqb_now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}
void aqb_sleep_ms(uint32_t ms) { usleep(ms * 1000u); }
int aqb_cpu_mhz(void) { return 0; }
void aqb_mem(uint32_t *a, uint32_t *b, uint32_t *c) { *a = *b = *c = 0; }

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: bench_host models.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    /* 16-byte aligned, as the flash mapping is */
    unsigned char *img = aligned_alloc(16, ((size_t)n + 15) & ~(size_t)15);
    if (fread(img, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);

    g_fd = posix_openpt(O_RDWR | O_NOCTTY);
    if (g_fd < 0 || grantpt(g_fd) || unlockpt(g_fd)) { perror("pty"); return 1; }
    struct termios t;
    int slave = open(ptsname(g_fd), O_RDWR | O_NOCTTY);   /* keep the pty open */
    tcgetattr(slave, &t); cfmakeraw(&t); tcsetattr(slave, TCSANOW, &t);
    printf("%s\n", ptsname(g_fd));
    fflush(stdout);
    aq_bench_loop(img);
    return 0;
}
