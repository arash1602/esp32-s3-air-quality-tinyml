/* aq_infer: portable inference for the WHO-threshold air quality classifiers.
 *
 * One library runs every model of the benchmark -- GRU, TCN (weight-normalized
 * convolutions folded), the gated CNN-GRU hybrid (FUSION), and the classical
 * predictors of paper 1 (the zero-parameter WHO rule, logistic regression on the
 * window means, gradient-boosted trees on 30 window statistics) -- from a model
 * "blob" produced by export/blob.py. The same code is compiled natively on a PC
 * (for fidelity checks over every test fold) and for the ESP32-S3 (blob linked
 * into flash), so the two produce the same arithmetic.
 *
 * Weights are float32 or int8 with one float scale per output row, dequantized on
 * the fly; activations are always float32. No dynamic allocation: all working
 * memory is static and sized by the AQ_MAX_* limits below.
 */
#ifndef AQ_INFER_H
#define AQ_INFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Limits; each can be lowered with -D to shrink the static working memory
 * (aq_static_bytes()), e.g. -DAQ_MAX_CH=64 when only smaller networks are used. */
#ifndef AQ_MAX_T
#define AQ_MAX_T      24    /* window length (hours)            */
#endif
#define AQ_MAX_IN     5     /* input channels                   */
#ifndef AQ_MAX_CH
#define AQ_MAX_CH     128   /* widest hidden layer              */
#endif
#ifndef AQ_MAX_K
#define AQ_MAX_K      8     /* convolution kernel size          */
#endif
#ifndef AQ_STAGE_BYTES
#define AQ_STAGE_BYTES 65536 /* RAM for one recurrent matrix (0 = never stage) */
#endif
#ifndef AQ_MAX_LAYERS
#define AQ_MAX_LAYERS 8     /* GRU layers or TCN blocks         */
#endif

enum { AQ_GRU = 0, AQ_TCN = 1, AQ_FUSION = 2, AQ_WHO = 3, AQ_LR = 4, AQ_GBT = 5 };

#define AQ_N_STATS 6        /* mean, std, min, max, last, last - first */

/* A weight matrix of rows x cols: float, or int8 with per-row scales. */
typedef struct {
    int32_t rows, cols;
    const float  *f;       /* float data, or NULL                */
    const int8_t *q;       /* int8 data, or NULL                 */
    const float  *scale;   /* per-row scale when q is used       */
} aq_mat_t;

typedef struct {
    aq_mat_t w_ih, w_hh;   /* (3H x in), (3H x H), rows r,z,n    */
    aq_mat_t b_ih, b_hh;   /* (3H x 1)                           */
} aq_gru_layer_t;

typedef struct {
    int32_t cin, cout, dilation, has_down;
    aq_mat_t w1, b1, w2, b2, wd, bd;   /* conv weights (cout x cin*K)   */
} aq_tcn_block_t;

typedef struct {
    int32_t arch, precision, T, n_in, kernel;
    float mean[AQ_MAX_IN], std[AQ_MAX_IN];
    uint32_t log_mask;                 /* bit c set: log1p on channel c  */
    float threshold;                   /* decision: logit > threshold    */
    /* GRU and FUSION recurrent part */
    int32_t gru_layers, gru_hidden;
    aq_gru_layer_t gru[AQ_MAX_LAYERS];
    /* TCN */
    int32_t n_blocks;
    aq_tcn_block_t blk[AQ_MAX_LAYERS];
    /* FUSION */
    int32_t cnn_channels, fusion_dim;
    aq_mat_t conv_w, conv_b, cnn_proj_w, cnn_proj_b, gru_proj_w, gru_proj_b,
             gate_w, gate_b;
    /* every network */
    aq_mat_t head_w, head_b;
    /* classical predictors: computed in double, exactly as scikit-learn does.
     * Decision: score >= thr_d (ge = 1, LR and GBT, as in paper 1) or > (ge = 0). */
    int32_t ge;
    double thr_d;
    int32_t who_i25, who_i10;          /* WHO rule: channels of PM2.5, PM10 */
    double who_lim25, who_lim10;
    int32_t lr_n, lr_idx[AQ_MAX_IN];   /* LR: window-mean channels, in order */
    const uint8_t *lr_coef;            /* f64 mean[n], scale[n], coef[n], intercept */
    int32_t gbt_trees;                 /* GBT: trees on the 30 window statistics */
    double gbt_baseline;
    const uint8_t *gbt_data;           /* per tree: u32 n_nodes, 16-byte nodes */
    size_t n_params;                   /* weights, or tree nodes for GBT */
} aq_model_t;

/* Parse a blob in place (pointers refer into it; keep it alive). 0 on success. */
int aq_load(aq_model_t *m, const void *blob, size_t len);

/* Raw window, row-major [T][n_in] in physical units -> logit. */
float aq_logit(const aq_model_t *m, const float *window);

/* 1 if the window is predicted to exceed the WHO limit (networks). */
static inline int aq_decide(const aq_model_t *m, float logit) { return logit > m->threshold; }

/* Any model: score (logit for networks, LR and GBT; max(PM2.5/15, PM10/45) for the
 * WHO rule) in *score if not NULL, and the decision as the return value. */
int aq_predict(const aq_model_t *m, const float *window, double *score);

/* Bytes of static working memory the library reserves (all models share it). */
size_t aq_static_bytes(void);

/* The 30 window statistics the GBT uses: stat s of channel c at [s * n_in + c]. */
void aq_window_stats(const float *window, int T, int n_in, double *out);

#ifdef __cplusplus
}
#endif
#endif
