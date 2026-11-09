/* aq_infer.c -- see aq_infer.h. Blob layout (little-endian, 4-byte aligned):
 *
 *   char magic[4] = "AQM1"; u32 version = 1
 *   u32 arch, precision, T, n_in, kernel
 *   f32 mean[n_in], std[n_in]; u32 log_mask; f32 threshold
 *   arch header:
 *     GRU:    u32 layers, hidden
 *     TCN:    u32 n_blocks; per block u32 cout, has_down
 *     FUSION: u32 cnn_channels, gru_hidden, gru_layers, fusion_dim
 *     classical (WHO, LR, GBT): u32 ge; f64 threshold, then
 *       WHO: u32 i25, i10; f64 lim25, lim10                      (no tensors)
 *       LR:  u32 n; u32 idx[n]; f64 mean[n], scale[n], coef[n], intercept
 *       GBT: u32 n_trees; f64 baseline; per tree u32 n_nodes, then nodes of
 *            { f64 threshold or leaf value; u16 left, right; u8 feature
 *              (255 = leaf), missing_go_left; u16 pad }          (no tensors)
 *   tensors, in model order, each:
 *     u32 kind (0 = f32, 1 = int8 + per-row f32 scale), u32 rows, u32 cols,
 *     [kind 1: f32 scale[rows]], data (f32[rows*cols] or i8[rows*cols] padded to 4)
 */
#include "aq_infer.h"
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ parsing */
typedef struct { const uint8_t *p, *end; int err; } rd_t;

static uint32_t rd_u32(rd_t *r) {
    uint32_t v = 0;
    if (r->p + 4 > r->end) { r->err = 1; return 0; }
    memcpy(&v, r->p, 4); r->p += 4; return v;
}
static float rd_f32(rd_t *r) { uint32_t u = rd_u32(r); float f; memcpy(&f, &u, 4); return f; }
static double rd_f64(rd_t *r) {
    double d = 0.0;
    if (r->p + 8 > r->end) { r->err = 1; return 0.0; }
    memcpy(&d, r->p, 8); r->p += 8; return d;
}
static double get_f64(const uint8_t *p) { double d; memcpy(&d, p, 8); return d; }

static size_t rd_tensor(rd_t *r, aq_mat_t *t) {
    uint32_t kind = rd_u32(r);
    t->rows = (int32_t)rd_u32(r); t->cols = (int32_t)rd_u32(r);
    size_t n = (size_t)t->rows * (size_t)t->cols;
    t->f = NULL; t->q = NULL; t->scale = NULL;
    if (r->err) return 0;
    if (kind == 0) {
        if (r->p + 4 * n > r->end) { r->err = 1; return 0; }
        t->f = (const float *)r->p; r->p += 4 * n;
    } else if (kind == 1) {
        if (r->p + 4 * (size_t)t->rows + n > r->end) { r->err = 1; return 0; }
        t->scale = (const float *)r->p; r->p += 4 * (size_t)t->rows;
        t->q = (const int8_t *)r->p; r->p += (n + 3) & ~(size_t)3;
    } else { r->err = 1; return 0; }
    return n;
}

static size_t rd_gru(rd_t *r, aq_gru_layer_t *g) {
    return rd_tensor(r, &g->w_ih) + rd_tensor(r, &g->w_hh)
         + rd_tensor(r, &g->b_ih) + rd_tensor(r, &g->b_hh);
}

int aq_load(aq_model_t *m, const void *blob, size_t len) {
    rd_t r = { (const uint8_t *)blob, (const uint8_t *)blob + len, 0 };
    memset(m, 0, sizeof *m);
    if (len < 8 || memcmp(blob, "AQM1", 4) != 0) return -1;
    r.p += 4;
    if (rd_u32(&r) != 1) return -2;
    m->arch = (int32_t)rd_u32(&r); m->precision = (int32_t)rd_u32(&r);
    m->T = (int32_t)rd_u32(&r); m->n_in = (int32_t)rd_u32(&r); m->kernel = (int32_t)rd_u32(&r);
    if (m->T > AQ_MAX_T || m->n_in > AQ_MAX_IN || m->kernel > AQ_MAX_K) return -3;
    for (int c = 0; c < m->n_in; c++) m->mean[c] = rd_f32(&r);
    for (int c = 0; c < m->n_in; c++) m->std[c] = rd_f32(&r);
    m->log_mask = rd_u32(&r); m->threshold = rd_f32(&r);
    size_t n = 0;
    if (m->arch == AQ_GRU) {
        m->gru_layers = (int32_t)rd_u32(&r); m->gru_hidden = (int32_t)rd_u32(&r);
        if (m->gru_layers > AQ_MAX_LAYERS || m->gru_hidden > AQ_MAX_CH) return -4;
        for (int l = 0; l < m->gru_layers; l++) n += rd_gru(&r, &m->gru[l]);
    } else if (m->arch == AQ_TCN) {
        m->n_blocks = (int32_t)rd_u32(&r);
        if (m->n_blocks > AQ_MAX_LAYERS) return -4;
        int cin = m->n_in;
        for (int b = 0; b < m->n_blocks; b++) {
            m->blk[b].cout = (int32_t)rd_u32(&r); m->blk[b].has_down = (int32_t)rd_u32(&r);
            m->blk[b].cin = cin; m->blk[b].dilation = 1 << b;
            if (m->blk[b].cout > AQ_MAX_CH) return -4;
            cin = m->blk[b].cout;
        }
        for (int b = 0; b < m->n_blocks; b++) {
            aq_tcn_block_t *k = &m->blk[b];
            n += rd_tensor(&r, &k->w1) + rd_tensor(&r, &k->b1)
               + rd_tensor(&r, &k->w2) + rd_tensor(&r, &k->b2);
            if (k->has_down) n += rd_tensor(&r, &k->wd) + rd_tensor(&r, &k->bd);
        }
    } else if (m->arch == AQ_FUSION) {
        m->cnn_channels = (int32_t)rd_u32(&r); m->gru_hidden = (int32_t)rd_u32(&r);
        m->gru_layers = (int32_t)rd_u32(&r); m->fusion_dim = (int32_t)rd_u32(&r);
        if (m->cnn_channels > AQ_MAX_CH || m->gru_hidden > AQ_MAX_CH ||
            2 * m->fusion_dim > AQ_MAX_CH * 2 || m->gru_layers > AQ_MAX_LAYERS) return -4;
        n += rd_tensor(&r, &m->conv_w) + rd_tensor(&r, &m->conv_b)
           + rd_tensor(&r, &m->cnn_proj_w) + rd_tensor(&r, &m->cnn_proj_b);
        for (int l = 0; l < m->gru_layers; l++) n += rd_gru(&r, &m->gru[l]);
        n += rd_tensor(&r, &m->gru_proj_w) + rd_tensor(&r, &m->gru_proj_b)
           + rd_tensor(&r, &m->gate_w) + rd_tensor(&r, &m->gate_b);
    } else if (m->arch == AQ_WHO || m->arch == AQ_LR || m->arch == AQ_GBT) {
        m->ge = (int32_t)rd_u32(&r); m->thr_d = rd_f64(&r);
        m->threshold = (float)m->thr_d;
        if (m->arch == AQ_WHO) {
            m->who_i25 = (int32_t)rd_u32(&r); m->who_i10 = (int32_t)rd_u32(&r);
            m->who_lim25 = rd_f64(&r); m->who_lim10 = rd_f64(&r);
            if (m->who_i25 >= m->n_in || m->who_i10 >= m->n_in) return -4;
        } else if (m->arch == AQ_LR) {
            m->lr_n = (int32_t)rd_u32(&r);
            if (m->lr_n > AQ_MAX_IN) return -4;
            for (int i = 0; i < m->lr_n; i++) {
                m->lr_idx[i] = (int32_t)rd_u32(&r);
                if (m->lr_idx[i] >= m->n_in) return -4;
            }
            size_t nb = 8 * (3 * (size_t)m->lr_n + 1);
            if (r.p + nb > r.end) return -6;
            m->lr_coef = r.p; r.p += nb;
            n = (size_t)m->lr_n + 1;
        } else {
            m->gbt_trees = (int32_t)rd_u32(&r); m->gbt_baseline = rd_f64(&r);
            m->gbt_data = r.p;
            for (int t = 0; t < m->gbt_trees && !r.err; t++) {
                uint32_t nn = rd_u32(&r);
                if (r.p + 16 * (size_t)nn > r.end) return -6;
                for (uint32_t k = 0; k < nn; k++) {   /* validate links and features */
                    const uint8_t *q = r.p + 16 * (size_t)k;
                    uint16_t lft, rgt; memcpy(&lft, q + 8, 2); memcpy(&rgt, q + 10, 2);
                    if (q[12] != 255 && (q[12] >= AQ_N_STATS * m->n_in || lft >= nn || rgt >= nn))
                        return -4;
                }
                r.p += 16 * (size_t)nn; n += nn;
            }
        }
        m->n_params = n;
        return r.err ? -6 : 0;
    } else return -5;
    n += rd_tensor(&r, &m->head_w) + rd_tensor(&r, &m->head_b);
    m->n_params = n;
    return r.err ? -6 : 0;
}

/* ------------------------------------------------------------------- kernels */
static inline float mat_at(const aq_mat_t *w, int i, int j) {
    return w->f ? w->f[(size_t)i * w->cols + j]
                : (float)w->q[(size_t)i * w->cols + j] * w->scale[i];
}

/* Dot product with four independent accumulators, so consecutive multiply-adds do
 * not wait on each other (the floating-point unit is pipelined). */
static inline float dot_f(const float *w, const float *x, int n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        a0 += w[j] * x[j]; a1 += w[j + 1] * x[j + 1];
        a2 += w[j + 2] * x[j + 2]; a3 += w[j + 3] * x[j + 3];
    }
    for (; j < n; j++) a0 += w[j] * x[j];
    return (a0 + a1) + (a2 + a3);
}

static inline float dot_q(const int8_t *w, const float *x, int n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        a0 += (float)w[j] * x[j]; a1 += (float)w[j + 1] * x[j + 1];
        a2 += (float)w[j + 2] * x[j + 2]; a3 += (float)w[j + 3] * x[j + 3];
    }
    for (; j < n; j++) a0 += (float)w[j] * x[j];
    return (a0 + a1) + (a2 + a3);
}

/* y[i] = W[i,:] . x + b[i]; b may be NULL. Each weight row is read once, in order. */
static void matvec(const aq_mat_t *w, const float *x, const aq_mat_t *b, float *y) {
    for (int i = 0; i < w->rows; i++) {
        float acc = w->f ? dot_f(w->f + (size_t)i * w->cols, x, w->cols)
                         : dot_q(w->q + (size_t)i * w->cols, x, w->cols) * w->scale[i];
        y[i] = acc + (b ? mat_at(b, i, 0) : 0.0f);
    }
}

static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Working memory, static so the ESP32 task stack stays small. */
static float buf_a[AQ_MAX_T * AQ_MAX_CH], buf_b[AQ_MAX_T * AQ_MAX_CH],
             buf_c[AQ_MAX_T * AQ_MAX_CH], buf_x[AQ_MAX_T * AQ_MAX_IN];
static float g_i[3 * AQ_MAX_CH], g_h[3 * AQ_MAX_CH], h_state[AQ_MAX_CH];
static float w_row[AQ_MAX_CH * AQ_MAX_K];     /* one convolution filter, in RAM */

/* Recurrent weights are read at every time step. When they live in flash and are
 * larger than the processor's flash cache they are re-fetched 24 times per
 * prediction, so each GRU layer's input projection is computed for all time steps
 * first (w_ih read once), and the recurrent matrix w_hh is copied into RAM for the
 * duration of the layer when it fits in AQ_STAGE_BYTES. Neither changes the
 * arithmetic: results are bit-identical. */
#if AQ_STAGE_BYTES > 0
static uint32_t stage_buf[(AQ_STAGE_BYTES + 3) / 4];
#endif
static float g_in[AQ_MAX_T * 3 * AQ_MAX_CH];  /* input projections, all time steps */

size_t aq_static_bytes(void) {
    return sizeof buf_a + sizeof buf_b + sizeof buf_c + sizeof buf_x
         + sizeof g_i + sizeof g_h + sizeof h_state + sizeof w_row + sizeof g_in
#if AQ_STAGE_BYTES > 0
         + sizeof stage_buf
#endif
         ;
}

/* PyTorch GRU layer (gate order r, z, n), batch of one. in: [T][din] -> out: [T][H]. */
static void gru_layer(const aq_gru_layer_t *g, const float *in, int T, int din, float *out) {
    int H = g->w_hh.cols, G = 3 * H;
    (void)din;
    /* input projections of all time steps, row by row: each row of w_ih is copied to
       RAM once and applied to all T inputs, so w_ih is read from flash once */
    {
        const aq_mat_t *w = &g->w_ih;
        int C = w->cols;
        for (int i = 0; i < w->rows; i++) {
            float bias = mat_at(&g->b_ih, i, 0);
            if (w->f) {
                memcpy(w_row, w->f + (size_t)i * C, sizeof(float) * (size_t)C);
                for (int t = 0; t < T; t++) {
                    float acc = dot_f(w_row, in + (size_t)t * C, C);
                    g_in[(size_t)t * G + i] = acc + bias;
                }
            } else {
                int8_t *qr = (int8_t *)w_row;
                memcpy(qr, w->q + (size_t)i * C, (size_t)C);
                float sc = w->scale[i];
                for (int t = 0; t < T; t++) {
                    float acc = dot_q(qr, in + (size_t)t * C, C) * sc;
                    g_in[(size_t)t * G + i] = acc + bias;
                }
            }
        }
    }
    /* recurrent matrix: as many leading rows as fit are copied to RAM for the layer
       (all of them for most models); the remaining rows are read in place */
    aq_mat_t whh = g->w_hh, rest = g->w_hh;
    int staged = 0;
#if AQ_STAGE_BYTES > 0
    {
        size_t row_bytes = (size_t)whh.cols * (whh.f ? sizeof(float) : 1u) + (whh.f ? 0u : sizeof(float));
        size_t fit = sizeof stage_buf / row_bytes;
        staged = (int)(fit < (size_t)whh.rows ? fit : (size_t)whh.rows);
        if (!g->b_hh.f) staged = 0;                  /* biases are float; be safe */
        if (staged > 0) {
            size_t n = (size_t)staged * (size_t)whh.cols;
            if (whh.f) {
                memcpy(stage_buf, whh.f, n * sizeof(float));
                whh.f = (const float *)stage_buf;
            } else {
                float *sc = (float *)stage_buf;
                memcpy(sc, g->w_hh.scale, (size_t)staged * sizeof(float));
                memcpy(sc + staged, g->w_hh.q, n);
                whh.scale = sc;
                whh.q = (const int8_t *)(sc + staged);
            }
            whh.rows = staged;
            rest.rows = g->w_hh.rows - staged;
            if (rest.f) rest.f += n; else { rest.q += n; rest.scale += staged; }
        }
    }
#endif
    aq_mat_t bh = g->b_hh, bh_rest = g->b_hh;     /* bias split to match the rows */
    if (staged > 0) { bh.rows = staged; bh_rest.rows -= staged; bh_rest.f += staged; }
    for (int k = 0; k < H; k++) h_state[k] = 0.0f;
    for (int t = 0; t < T; t++) {
        const float *gi = g_in + (size_t)t * G;
        if (staged > 0) {
            matvec(&whh, h_state, &bh, g_h);
            if (rest.rows > 0) matvec(&rest, h_state, &bh_rest, g_h + staged);
        } else {
            matvec(&g->w_hh, h_state, &g->b_hh, g_h);
        }
        for (int k = 0; k < H; k++) {
            float r = sigm(gi[k] + g_h[k]);
            float z = sigm(gi[H + k] + g_h[H + k]);
            float nn = tanhf(gi[2 * H + k] + r * g_h[2 * H + k]);
            h_state[k] = (1.0f - z) * nn + z * h_state[k];
        }
        memcpy(out + (size_t)t * H, h_state, sizeof(float) * (size_t)H);
    }
}

/* Runs the stacked GRU; returns a pointer to the [T][H] output of the last layer. */
static const float *gru_stack(const aq_model_t *m, const float *x, int T) {
    const float *in = x; int din = m->n_in;
    float *bufs[2] = { buf_a, buf_b };
    for (int l = 0; l < m->gru_layers; l++) {
        gru_layer(&m->gru[l], in, T, din, bufs[l & 1]);
        in = bufs[l & 1]; din = m->gru[l].w_hh.cols;
    }
    return in;
}

/* Causal dilated conv: in [T][cin] -> out [T][cout], weights (cout x cin*K) with
 * element [o][c*K + k], left zero padding. Each filter is read from flash once and
 * kept in RAM (w_row) while it slides over the window, so the weights are read once
 * per prediction instead of once per time step; int8 filters are dequantized into
 * w_row, which is exactly the float model the NumPy reference evaluates. */
static void causal_conv(const aq_mat_t *w, const aq_mat_t *b, const float *in, int T,
                        int cin, int K, int dil, int relu, float *out) {
    int cout = w->rows, n = cin * K;
    for (int o = 0; o < cout; o++) {
        if (w->f) memcpy(w_row, w->f + (size_t)o * n, sizeof(float) * (size_t)n);
        else {
            const int8_t *q = w->q + (size_t)o * n;
            float sc = w->scale[o];
            for (int j = 0; j < n; j++) w_row[j] = (float)q[j] * sc;
        }
        float bias = mat_at(b, o, 0);
        for (int t = 0; t < T; t++) {
            float a0 = 0.0f, a1 = 0.0f;
            for (int k = 0; k < K; k++) {
                int src = t - (K - 1 - k) * dil;
                if (src < 0) continue;
                const float *xin = in + (size_t)src * cin;
                const float *wk = w_row + k;
                int c = 0;
                for (; c + 2 <= cin; c += 2) {
                    a0 += wk[c * K] * xin[c];
                    a1 += wk[(c + 1) * K] * xin[c + 1];
                }
                for (; c < cin; c++) a0 += wk[c * K] * xin[c];
            }
            float acc = (a0 + a1) + bias;
            out[(size_t)t * cout + o] = (relu && acc < 0.0f) ? 0.0f : acc;
        }
    }
}

/* ------------------------------------------------------- classical models */
void aq_window_stats(const float *window, int T, int n_in, double *out) {
    /* Same order of operations as NumPy on a float64 copy of the window:
     * sequential sums down the time axis, population standard deviation. */
    for (int c = 0; c < n_in; c++) {
        double s = 0.0, mn = window[c], mx = window[c];
        for (int t = 0; t < T; t++) {
            double v = window[t * n_in + c];
            s += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double mean = s / T, ss = 0.0;
        for (int t = 0; t < T; t++) {
            double d = (double)window[t * n_in + c] - mean;
            ss += d * d;
        }
        double first = window[c], last = window[(T - 1) * n_in + c];
        out[0 * n_in + c] = mean;
        out[1 * n_in + c] = sqrt(ss / T);
        out[2 * n_in + c] = mn;
        out[3 * n_in + c] = mx;
        out[4 * n_in + c] = last;
        out[5 * n_in + c] = last - first;
    }
}

static double gbt_raw(const aq_model_t *m, const double *f) {
    double raw = m->gbt_baseline;
    const uint8_t *p = m->gbt_data;
    for (int t = 0; t < m->gbt_trees; t++) {
        uint32_t nn; memcpy(&nn, p, 4);
        const uint8_t *nodes = p + 4, *q = nodes;
        while (q[12] != 255) {
            double x = f[q[12]];
            uint16_t nxt;
            int left = isnan(x) ? q[13] : (x <= get_f64(q));
            memcpy(&nxt, q + (left ? 8 : 10), 2);
            q = nodes + 16 * (size_t)nxt;
        }
        raw += get_f64(q);
        p = nodes + 16 * (size_t)nn;
    }
    return raw;
}

static int classical(const aq_model_t *m, const float *window, double *score) {
    double f[AQ_N_STATS * AQ_MAX_IN];
    aq_window_stats(window, m->T, m->n_in, f);   /* f[c] is the window mean */
    double s;
    if (m->arch == AQ_WHO) {
        double m25 = f[m->who_i25], m10 = f[m->who_i10];
        double a = m25 / m->who_lim25, b = m10 / m->who_lim10;
        if (score) *score = a > b ? a : b;
        return m25 > m->who_lim25 || m10 > m->who_lim10;   /* the rule itself */
    } else if (m->arch == AQ_LR) {
        const uint8_t *k = m->lr_coef;
        int n = m->lr_n;
        s = 0.0;
        for (int i = 0; i < n; i++) {
            double z = (f[m->lr_idx[i]] - get_f64(k + 8 * i)) / get_f64(k + 8 * (n + i));
            s += z * get_f64(k + 8 * (2 * n + i));
        }
        s += get_f64(k + 8 * (3 * n));
    } else {
        s = gbt_raw(m, f);
    }
    if (score) *score = s;
    return m->ge ? s >= m->thr_d : s > m->thr_d;
}

static float network_logit(const aq_model_t *m, const float *window);

float aq_logit(const aq_model_t *m, const float *window) {
    if (m->arch >= AQ_WHO) { double s; classical(m, window, &s); return (float)s; }
    return network_logit(m, window);
}

int aq_predict(const aq_model_t *m, const float *window, double *score) {
    if (m->arch >= AQ_WHO) return classical(m, window, score);
    float y = network_logit(m, window);
    if (score) *score = y;
    return y > m->threshold;
}

/* ----------------------------------------------------------------- networks */
static float network_logit(const aq_model_t *m, const float *window) {
    const int T = m->T, C = m->n_in;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) {
            float v = window[t * C + c];
            if (m->log_mask & (1u << c)) v = log1pf(v > 0.0f ? v : 0.0f);
            float sd = (m->std[c] > 0.0f && isfinite(m->std[c])) ? m->std[c] : 1.0f;
            buf_x[t * C + c] = (v - m->mean[c]) / sd;
        }
    float y = 0.0f;
    if (m->arch == AQ_GRU) {
        const float *out = gru_stack(m, buf_x, T);
        matvec(&m->head_w, out + (size_t)(T - 1) * m->gru_hidden, &m->head_b, &y);
    } else if (m->arch == AQ_TCN) {
        const float *in = buf_x; int cin = C;
        for (int b = 0; b < m->n_blocks; b++) {
            const aq_tcn_block_t *k = &m->blk[b];
            /* the two working buffers not holding this block's input */
            float *pool[3] = { buf_a, buf_b, buf_c }, *h = NULL, *o = NULL;
            for (int i = 0; i < 3; i++) {
                if (pool[i] == in) continue;
                if (!h) h = pool[i]; else if (!o) o = pool[i];
            }
            causal_conv(&k->w1, &k->b1, in, T, cin, m->kernel, k->dilation, 1, h);
            causal_conv(&k->w2, &k->b2, h, T, k->cout, m->kernel, k->dilation, 1, o);
            /* residual: o = relu(o + res) */
            if (k->has_down) {
                causal_conv(&k->wd, &k->bd, in, T, cin, 1, 1, 0, h);   /* h reused for res */
                for (size_t i = 0; i < (size_t)T * k->cout; i++) {
                    float v = o[i] + h[i]; o[i] = v > 0.0f ? v : 0.0f;
                }
            } else {
                for (size_t i = 0; i < (size_t)T * k->cout; i++) {
                    float v = o[i] + in[i]; o[i] = v > 0.0f ? v : 0.0f;
                }
            }
            in = o; cin = k->cout;   /* block output feeds the next block */
        }
        matvec(&m->head_w, in + (size_t)(T - 1) * cin, &m->head_b, &y);
    } else { /* FUSION */
        int cc = m->cnn_channels, fd = m->fusion_dim;
        float f_cnn[AQ_MAX_CH], f_gru[AQ_MAX_CH], cat[2 * AQ_MAX_CH], g[AQ_MAX_CH], fused[AQ_MAX_CH];
        causal_conv(&m->conv_w, &m->conv_b, buf_x, T, C, m->kernel, 1, 1, buf_c);
        matvec(&m->cnn_proj_w, buf_c + (size_t)(T - 1) * cc, &m->cnn_proj_b, f_cnn);
        const float *out = gru_stack(m, buf_x, T);
        matvec(&m->gru_proj_w, out + (size_t)(T - 1) * m->gru_hidden, &m->gru_proj_b, f_gru);
        memcpy(cat, f_cnn, sizeof(float) * (size_t)fd);
        memcpy(cat + fd, f_gru, sizeof(float) * (size_t)fd);
        matvec(&m->gate_w, cat, &m->gate_b, g);
        for (int k = 0; k < fd; k++) {
            float s = sigm(g[k]);
            fused[k] = s * f_cnn[k] + (1.0f - s) * f_gru[k];
        }
        matvec(&m->head_w, fused, &m->head_b, &y);
    }
    return y;
}
