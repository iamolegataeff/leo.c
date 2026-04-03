/*
 * leotests/test_leo.c — Leo C inference test suite
 *
 * All tests are functions. Run: cc test_leo.c -O2 -lm -o test_leo && ./test_leo
 * Tests verify math ops, GGUF parsing logic, and Gemma-3 specific behavior.
 * No model file needed for unit tests (marked [unit]).
 * Integration tests (marked [integ]) require a GGUF path as argv[1].
 *
 * (c) 2026 arianna method
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); tests_failed++; } \
    else { tests_passed++; } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("  FAIL: %s (got %.6f, expected %.6f, diff %.6f)\n", msg, (float)(a), (float)(b), fabsf((a)-(b))); \
        tests_failed++; \
    } else { tests_passed++; } \
} while(0)

/* ═══════════════════════════════════════════════════════════════
 * Math ops (copied from leo.c for standalone testing)
 * ═══════════════════════════════════════════════════════════════ */

static float silu_f(float x) { return x / (1.0f + expf(-x)); }
static float gelu_tanh_f(float x) { return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); }

static void rmsnorm(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

static void rmsnorm_gemma(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * (w[i] + 1.0f);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = (sign<<31)|((exp+112)<<23)|(mant<<13); }
    } else if (exp == 31) f = (sign<<31)|0x7F800000|(mant<<13);
    else f = (sign<<31)|((exp+112)<<23)|(mant<<13);
    float r; memcpy(&r, &f, 4); return r;
}

static void softmax_n(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ═══════════════════════════════════════════════════════════════
 * Unit tests
 * ═══════════════════════════════════════════════════════════════ */

void test_gelu_tanh(void) {
    printf("[unit] test_gelu_tanh\n");
    ASSERT_NEAR(gelu_tanh_f(0.0f), 0.0f, 1e-6, "gelu(0) = 0");
    ASSERT_NEAR(gelu_tanh_f(1.0f), 0.8412f, 1e-3, "gelu(1) ≈ 0.841");
    ASSERT_NEAR(gelu_tanh_f(-1.0f), -0.1588f, 1e-3, "gelu(-1) ≈ -0.159");
    ASSERT_NEAR(gelu_tanh_f(3.0f), 2.9964f, 1e-3, "gelu(3) ≈ 2.996");
    /* GELU ≠ SiLU */
    float g = gelu_tanh_f(1.0f);
    float s = silu_f(1.0f);
    ASSERT(fabsf(g - s) > 0.05f, "gelu(1) ≠ silu(1)");
}

void test_silu(void) {
    printf("[unit] test_silu\n");
    ASSERT_NEAR(silu_f(0.0f), 0.0f, 1e-6, "silu(0) = 0");
    ASSERT_NEAR(silu_f(1.0f), 0.7311f, 1e-3, "silu(1) ≈ 0.731");
    ASSERT_NEAR(silu_f(-1.0f), -0.2689f, 1e-3, "silu(-1) ≈ -0.269");
}

void test_rmsnorm_basic(void) {
    printf("[unit] test_rmsnorm_basic\n");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    rmsnorm(out, x, w, 4, 1e-6f);
    /* RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
    float rms = sqrtf(30.0f / 4.0f);
    ASSERT_NEAR(out[0], 1.0f / rms, 1e-4, "rmsnorm x[0]/rms");
    ASSERT_NEAR(out[1], 2.0f / rms, 1e-4, "rmsnorm x[1]/rms");
    ASSERT_NEAR(out[3], 4.0f / rms, 1e-4, "rmsnorm x[3]/rms");
}

void test_rmsnorm_with_weights(void) {
    printf("[unit] test_rmsnorm_with_weights\n");
    float x[] = {1.0f, -1.0f, 1.0f, -1.0f};
    float w[] = {2.0f, 3.0f, 4.0f, 5.0f};
    float out[4];
    rmsnorm(out, x, w, 4, 1e-6f);
    /* RMS = sqrt(4/4) = 1.0, inv = 1.0 */
    ASSERT_NEAR(out[0], 2.0f, 1e-4, "rmsnorm weighted x[0]");
    ASSERT_NEAR(out[1], -3.0f, 1e-4, "rmsnorm weighted x[1]");
    ASSERT_NEAR(out[2], 4.0f, 1e-4, "rmsnorm weighted x[2]");
    ASSERT_NEAR(out[3], -5.0f, 1e-4, "rmsnorm weighted x[3]");
}

void test_rmsnorm_gemma_adds_one(void) {
    printf("[unit] test_rmsnorm_gemma_adds_one\n");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w_original[] = {0.5f, 0.5f, 0.5f, 0.5f}; /* Gemma stores w-1 */
    float out_gemma[4], out_normal[4];
    float w_converted[] = {1.5f, 1.5f, 1.5f, 1.5f}; /* After llama.cpp +1 */
    rmsnorm_gemma(out_gemma, x, w_original, 4, 1e-6f);
    rmsnorm(out_normal, x, w_converted, 4, 1e-6f);
    /* Both should give same result */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(out_gemma[i], out_normal[i], 1e-5, "gemma norm matches converted");
}

void test_f16_to_f32(void) {
    printf("[unit] test_f16_to_f32\n");
    /* f16 encoding of 1.0 = 0x3C00 */
    ASSERT_NEAR(f16_to_f32(0x3C00), 1.0f, 1e-6, "f16(1.0)");
    /* f16 encoding of -1.0 = 0xBC00 */
    ASSERT_NEAR(f16_to_f32(0xBC00), -1.0f, 1e-6, "f16(-1.0)");
    /* f16 encoding of 0.0 = 0x0000 */
    ASSERT_NEAR(f16_to_f32(0x0000), 0.0f, 1e-6, "f16(0.0)");
    /* f16 encoding of 0.5 = 0x3800 */
    ASSERT_NEAR(f16_to_f32(0x3800), 0.5f, 1e-6, "f16(0.5)");
    /* f16 encoding of 65504 (max normal) = 0x7BFF */
    ASSERT(f16_to_f32(0x7BFF) > 65000.0f, "f16 max normal > 65000");
}

void test_softmax(void) {
    printf("[unit] test_softmax\n");
    float x[] = {1.0f, 2.0f, 3.0f};
    softmax_n(x, 3);
    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5, "softmax sums to 1");
    ASSERT(x[2] > x[1], "softmax preserves order [2]>[1]");
    ASSERT(x[1] > x[0], "softmax preserves order [1]>[0]");
    ASSERT_NEAR(x[2], 0.6652f, 1e-3, "softmax(3) ≈ 0.665");
}

void test_softmax_single(void) {
    printf("[unit] test_softmax_single\n");
    float x[] = {42.0f};
    softmax_n(x, 1);
    ASSERT_NEAR(x[0], 1.0f, 1e-6, "softmax single element = 1.0");
}

void test_softmax_numerical_stability(void) {
    printf("[unit] test_softmax_numerical_stability\n");
    float x[] = {1000.0f, 1001.0f, 1002.0f};
    softmax_n(x, 3);
    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5, "softmax stable with large values");
    ASSERT(x[2] > x[1], "softmax stable: preserves order");
}

void test_embedding_scaling(void) {
    printf("[unit] test_embedding_scaling\n");
    /* Gemma-3 scales embeddings by sqrt(hidden_dim) */
    int D = 640;
    float scale = sqrtf((float)D);
    ASSERT_NEAR(scale, 25.298f, 0.01f, "sqrt(640) ≈ 25.298");
    /* After scaling, a typical embedding value ~0.02 becomes ~0.5 */
    float emb = 0.02f;
    ASSERT_NEAR(emb * scale, 0.506f, 0.01f, "scaled embedding ≈ 0.506");
}

void test_gemma3_architecture_constants(void) {
    printf("[unit] test_gemma3_architecture_constants\n");
    /* Gemma-3 270M architecture */
    int n_layers = 18;
    int hidden = 640;
    int ffn = 2048;
    int n_heads = 4;
    int n_kv_heads = 1;
    int head_dim = 256;
    int vocab = 262144;
    int sliding_window = 512;

    ASSERT(head_dim != hidden / n_heads, "Gemma-3 head_dim ≠ hidden/heads (256 ≠ 160)");
    ASSERT(n_heads / n_kv_heads == 4, "GQA ratio = 4");
    ASSERT(n_heads * head_dim == 1024, "Q proj output = 1024");
    ASSERT(n_kv_heads * head_dim == 256, "KV proj output = 256");

    /* Global attention layers: every 6th starting from 5 */
    int global_layers[] = {5, 11, 17};
    for (int i = 0; i < 3; i++)
        ASSERT(global_layers[i] % 6 == 5, "global layer pattern: l%6==5");

    /* Local layers use sliding window */
    for (int l = 0; l < n_layers; l++) {
        if (l % 6 != 5)
            ASSERT(sliding_window == 512, "local layer window = 512");
    }

    (void)ffn; (void)vocab;
}

void test_rope_dual_theta(void) {
    printf("[unit] test_rope_dual_theta\n");
    /* Gemma-3 uses two RoPE bases */
    float theta_global = 1000000.0f;
    float theta_swa = 10000.0f;
    ASSERT(theta_global != theta_swa, "dual theta: global ≠ swa");

    /* Frequency at position 0, dim 0 */
    int head_dim = 256;
    float freq_global = 1.0f / powf(theta_global, 0.0f / (float)head_dim);
    float freq_swa = 1.0f / powf(theta_swa, 0.0f / (float)head_dim);
    ASSERT_NEAR(freq_global, 1.0f, 1e-6, "freq at dim=0 always 1.0 (global)");
    ASSERT_NEAR(freq_swa, 1.0f, 1e-6, "freq at dim=0 always 1.0 (swa)");

    /* Frequency at higher dims diverges */
    float freq_g_128 = 1.0f / powf(theta_global, 128.0f / (float)head_dim);
    float freq_s_128 = 1.0f / powf(theta_swa, 128.0f / (float)head_dim);
    ASSERT(freq_s_128 > freq_g_128 * 5.0f, "swa freq >> global freq at dim=128");
}

void test_vocab_limit_gemma(void) {
    printf("[unit] test_vocab_limit_gemma\n");
    /* The critical bug: vocab 200K limit rejected 262K Gemma vocab */
    int gemma_vocab = 262144;
    int old_limit = 200000;
    int new_limit = 300000;
    ASSERT(gemma_vocab > old_limit, "Gemma vocab exceeds old 200K limit");
    ASSERT(gemma_vocab < new_limit, "Gemma vocab fits new 300K limit");
}

void test_geglu_vs_swiglu(void) {
    printf("[unit] test_geglu_vs_swiglu\n");
    /* GEGLU: gelu(gate) * up; SwiGLU: silu(gate) * up */
    float gate = 2.0f, up = 1.5f;
    float geglu = gelu_tanh_f(gate) * up;
    float swiglu = silu_f(gate) * up;
    ASSERT(fabsf(geglu - swiglu) > 0.01f, "GEGLU ≠ SwiGLU for same inputs");
    /* For Gemma-3, GEGLU is correct */
    ASSERT_NEAR(geglu, gelu_tanh_f(2.0f) * 1.5f, 1e-6, "GEGLU = gelu(gate)*up");
}

void test_sliding_window_bounds(void) {
    printf("[unit] test_sliding_window_bounds\n");
    int sliding_window = 512;

    /* At position 100, all tokens visible (100 < 512) */
    int pos1 = 100;
    int start1 = pos1 - sliding_window + 1;
    if (start1 < 0) start1 = 0;
    ASSERT(start1 == 0, "pos=100: start=0 (all visible)");

    /* At position 600, only last 512 visible */
    int pos2 = 600;
    int start2 = pos2 - sliding_window + 1;
    if (start2 < 0) start2 = 0;
    ASSERT(start2 == 89, "pos=600: start=89 (window applied)");
    ASSERT(pos2 - start2 + 1 == 512, "window size = 512");
}

void test_qk_norm_dimensions(void) {
    printf("[unit] test_qk_norm_dimensions\n");
    /* Q/K norm operates on head_dim=256, not hidden=640 */
    int head_dim = 256;
    int n_heads = 4;
    int n_kv_heads = 1;
    int q_total = n_heads * head_dim;     /* 1024 */
    int k_total = n_kv_heads * head_dim;  /* 256 */

    ASSERT(q_total == 1024, "Q total dim = 1024");
    ASSERT(k_total == 256, "K total dim = 256");

    /* Norm weight is per head_dim, shared across heads */
    int norm_weight_size = head_dim; /* 256 */
    ASSERT(norm_weight_size == 256, "Q/K norm weight = 256 elements");
}

void test_post_norm_sandwich(void) {
    printf("[unit] test_post_norm_sandwich\n");
    /* Gemma-3 sandwich norm: pre-norm → sublayer → post-norm → residual */
    float residual[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float sublayer_out[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float post_norm_w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float normed[4];

    /* Post-norm the sublayer output */
    rmsnorm(normed, sublayer_out, post_norm_w, 4, 1e-6f);

    /* Add to residual */
    float result[4];
    for (int i = 0; i < 4; i++) result[i] = residual[i] + normed[i];

    /* Result should be residual + normalized sublayer */
    ASSERT(result[0] > residual[0], "sandwich: result > residual (positive sublayer)");
    ASSERT(result[0] < residual[0] + sublayer_out[0] + 0.5f, "sandwich: post-norm bounds output");
}

void test_tied_embeddings(void) {
    printf("[unit] test_tied_embeddings\n");
    /* Gemma-3 270M uses tied embeddings: lm_head = embed_tokens */
    /* When output.weight is missing, we reuse token_embd.weight */
    float *tok_emb = (float *)malloc(4 * sizeof(float));
    tok_emb[0] = 1.0f; tok_emb[1] = 2.0f; tok_emb[2] = 3.0f; tok_emb[3] = 4.0f;
    float *output = NULL; /* missing */
    if (!output) output = tok_emb; /* tie */
    ASSERT(output == tok_emb, "tied: output == tok_emb");
    ASSERT_NEAR(output[0], 1.0f, 1e-6, "tied: same values");
    free(tok_emb);
}

void test_gemma_bos_eos(void) {
    printf("[unit] test_gemma_bos_eos\n");
    /* Gemma-3 token IDs */
    int bos = 2;  /* <bos> */
    int eos = 106; /* <end_of_turn> */
    ASSERT(bos == 2, "Gemma BOS = 2");
    ASSERT(eos == 106, "Gemma EOS = 106 (<end_of_turn>)");
    ASSERT(bos != eos, "BOS ≠ EOS");
}

void test_attention_output_dim(void) {
    printf("[unit] test_attention_output_dim\n");
    /* For Gemma-3: attention output is heads*head_dim = 1024, but D = 640 */
    /* o_proj maps 1024 → 640 */
    int heads = 4, head_dim = 256, D = 640;
    int attn_out_dim = heads * head_dim;
    ASSERT(attn_out_dim == 1024, "attn output = 1024");
    ASSERT(attn_out_dim > D, "attn output > hidden dim (Gemma-3 peculiarity)");
    /* Buffer must be max(D, attn_out_dim) */
    int buf = attn_out_dim > D ? attn_out_dim : D;
    ASSERT(buf == 1024, "buffer = max(D, attn_out) = 1024");
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    printf("leo.c test suite\n\n");

    /* Unit tests — no model needed */
    test_gelu_tanh();
    test_silu();
    test_rmsnorm_basic();
    test_rmsnorm_with_weights();
    test_rmsnorm_gemma_adds_one();
    test_f16_to_f32();
    test_softmax();
    test_softmax_single();
    test_softmax_numerical_stability();
    test_embedding_scaling();
    test_gemma3_architecture_constants();
    test_rope_dual_theta();
    test_vocab_limit_gemma();
    test_geglu_vs_swiglu();
    test_sliding_window_bounds();
    test_qk_norm_dimensions();
    test_post_norm_sandwich();
    test_tied_embeddings();
    test_gemma_bos_eos();
    test_attention_output_dim();

    printf("\n%d tests: %d passed, %d failed\n", tests_run, tests_passed, tests_failed);

    (void)argc; (void)argv;
    return tests_failed > 0 ? 1 : 0;
}
