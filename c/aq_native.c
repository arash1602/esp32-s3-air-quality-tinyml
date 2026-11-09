/* Native harness: run a model blob over a file of windows and print one logit per line.
 *
 *   aq_native model.aqm windows.f32 > logits.txt
 *
 * windows.f32 holds N windows as little-endian float32 [N][24][5]. Used to check the
 * C library against the NumPy reference and PyTorch on every test fold, on a PC.
 * With -t, it also times inference on the host. With -p, each line is
 * "score decision", the score to 17 significant digits (classical models are
 * computed in double).
 */
#include "aq_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)n + 4);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f); *len = (size_t)n; return buf;
}

int main(int argc, char **argv) {
    int timing = 0, pred = 0, a = 1;
    if (argc > 1 && strcmp(argv[1], "-t") == 0) { timing = 1; a = 2; }
    else if (argc > 1 && strcmp(argv[1], "-p") == 0) { pred = 1; a = 2; }
    if (argc - a != 2) { fprintf(stderr, "usage: aq_native [-t|-p] model.aqm windows.f32\n"); return 2; }
    size_t blen, wlen;
    void *blob = slurp(argv[a], &blen);
    float *win = (float *)slurp(argv[a + 1], &wlen);
    if (!blob || !win) { fprintf(stderr, "cannot read inputs\n"); return 1; }
    aq_model_t m;
    int rc = aq_load(&m, blob, blen);
    if (rc) { fprintf(stderr, "aq_load failed: %d\n", rc); return 1; }
    size_t per = (size_t)m.T * (size_t)m.n_in, n = wlen / (sizeof(float) * per);
    clock_t t0 = clock();
    for (size_t i = 0; i < n; i++) {
        if (pred) {
            double s; int d = aq_predict(&m, win + i * per, &s);
            printf("%.17g %d\n", s, d);
            continue;
        }
        float y = aq_logit(&m, win + i * per);
        if (!timing) printf("%.9g\n", y);
    }
    if (timing)
        printf("%zu windows, %.3f us per window (host), %zu params\n", n,
               1e6 * (double)(clock() - t0) / CLOCKS_PER_SEC / (double)n, m.n_params);
    return 0;
}
