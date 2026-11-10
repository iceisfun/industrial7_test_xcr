/* simd_jpeg_dec.h - self-contained baseline + progressive Huffman JPEG decoder for the simd_jpeg exe.
 *
 * Platform neutral (stdint/string/stdlib + SSE2/AVX2 intrinsics only) so the very same code is also
 * built natively on the corpus host (corpus/jpeg/gen_synth.c) to produce golden values.
 *
 * Pluggable kernels (jd_kernels): islow integer IDCT, AAN float IDCT, fancy upsampling (h2v1, h2v2,
 * h1v2), YCbCr->RGB / RGBA. Three sets: scalar, SSE2, AVX2. Every set computes exactly the same
 * function (bit-identical output for every input), which the exe checks.
 *
 * Arithmetic contract (suite-defined, chosen so that it equals libjpeg-turbo's jidctint.c / jdsample.c /
 * jdcolor.c results for every valid 8-bit stream, and is well defined for corrupt ones):
 *   islow : dequantised value = sat16(coef * q); all IDCT arithmetic mod 2^32 (libjpeg's MULTIPLY/DESCALE
 *           with CONST_BITS 13, PASS1_BITS 2); pass-1 outputs saturate to int16 (as the SIMD paths of
 *           libjpeg-turbo do); final sample = clamp(v + 128, 0, 255). For |v| < 512 this equals
 *           libjpeg's range_limit table; beyond that libjpeg wraps (C) or saturates (SIMD) and we saturate.
 *   float : AAN (jidctflt.c structure), IEEE single precision, fixed operation order, no FMA, multiplier
 *           table q*aan[r]*aan[c]*0.125f computed in float; sample = clamp(trunc(x + 128.5)) with
 *           cvttps2dq semantics (out-of-range -> INT_MIN -> 0).
 *   upsampling and colour conversion: libjpeg-turbo's integer formulas (see each scalar kernel).
 * Entropy decoding follows libjpeg's jdhuff.c / jdphuff.c behaviour including corrupt-data handling
 * (zero-bit padding at end of data / markers, "insufficient data" skips later MCUs of the scan, bad
 * Huffman code -> 17 bits consumed and symbol 0, restart-marker resync per jpeg_resync_to_restart),
 * so truncated baseline files reproduce djpeg's documented partial output (grey 128 blocks).
 * Not supported (clean error JD_ERR_UNSUPPORTED): arithmetic coding, lossless, hierarchical, 12/16-bit
 * precision, DNL, 2 or 4 components. Progressive block smoothing (libjpeg's do_block_smoothing for
 * incompletely refined images) is NOT implemented: complete images are unaffected. */
#ifndef SIMD_JPEG_DEC_H
#define SIMD_JPEG_DEC_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <immintrin.h>

#define JD_AVX2   __attribute__((target("avx2")))
#define JD_FPSSE  __attribute__((target("sse2,fpmath=sse")))  /* scalar float with SSE semantics on x86 too */
#define JD_INLINE static inline __attribute__((always_inline))

enum { JD_DCT_ISLOW = 0, JD_DCT_FLOAT = 1 };
enum { JD_UP_FANCY = 0, JD_UP_BOX = 1 };
enum { JD_OUT_NATIVE = 0, JD_OUT_RGB = 1, JD_OUT_RGBA = 2 };
enum { JD_CS_GRAY = 0, JD_CS_YCC = 1, JD_CS_RGB = 2 };
enum {
    JD_OK = 0, JD_ERR_ARG = -1, JD_ERR_NOT_JPEG = -2, JD_ERR_TRUNC_HEADER = -3, JD_ERR_BAD_MARKER = -4,
    JD_ERR_UNSUPPORTED = -5, JD_ERR_BAD_DQT = -6, JD_ERR_BAD_DHT = -7, JD_ERR_BAD_SOF = -8,
    JD_ERR_BAD_SOS = -9, JD_ERR_TOO_BIG = -10, JD_ERR_NOMEM = -11, JD_ERR_NO_SCAN = -12, JD_ERR_BAD_DRI = -13
};
/* warning bits (decode still returns JD_OK with a complete, documented output) */
enum {
    JD_W_TRUNCATED = 1,   /* entropy data ran out / hit a marker: rest of that scan decoded as zeros */
    JD_W_CORRUPT = 2,     /* bad Huffman code or coefficient index overrun */
    JD_W_RESTART = 4,     /* restart marker missing / out of sequence (resynchronised) */
    JD_W_EOF = 8,         /* end of data before EOI (fake EOI inserted, like libjpeg's source managers) */
    JD_W_EXTRANEOUS = 16, /* garbage bytes before a marker */
    JD_W_PARTIAL_HDR = 32 /* data ended inside a marker segment after the first scan: later scans dropped
                           * (a malformed - not truncated - segment is a fatal error, as in libjpeg) */
};

static const char *jd_strerror(int e) {
    switch (e) {
    case JD_OK: return "ok"; case JD_ERR_ARG: return "bad argument"; case JD_ERR_NOT_JPEG: return "not a JPEG (no SOI)";
    case JD_ERR_TRUNC_HEADER: return "truncated before image data"; case JD_ERR_BAD_MARKER: return "bad marker";
    case JD_ERR_UNSUPPORTED: return "unsupported JPEG process"; case JD_ERR_BAD_DQT: return "bad/missing DQT";
    case JD_ERR_BAD_DHT: return "bad/missing DHT"; case JD_ERR_BAD_SOF: return "bad SOF";
    case JD_ERR_BAD_SOS: return "bad SOS"; case JD_ERR_TOO_BIG: return "image too big";
    case JD_ERR_NOMEM: return "out of memory"; case JD_ERR_NO_SCAN: return "no scan"; case JD_ERR_BAD_DRI: return "bad DRI";
    }
    return "?";
}

/* ---------------------------------------------------------------------------------------------
 * kernels
 * ------------------------------------------------------------------------------------------- */
typedef struct jd_kernels {
    const char *name;
    /* coef: 64 int16 natural order; q: 64 uint16 natural order; out: 8x8 bytes at stride */
    void (*idct_islow)(const int16_t *coef, const uint16_t *q, uint8_t *out, ptrdiff_t stride);
    void (*idct_float)(const int16_t *coef, const float *fq, uint8_t *out, ptrdiff_t stride);
    /* w >= 2 input samples, 2*w output samples */
    void (*h2v1_fancy)(const uint8_t *in, int w, uint8_t *out);
    /* in0 = nearest row, in1 = next-nearest row (above for the upper output row, below for the lower) */
    void (*h2v2_fancy)(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out);
    /* w >= 1; lower = 0 for the upper output row (bias 1), 1 for the lower one (bias 2) */
    void (*h1v2_fancy)(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out, int lower);
    /* n >= 1 pixels */
    void (*ycc_rgb)(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n);
    void (*ycc_rgba)(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n);
} jd_kernels;

/* ---- shared IDCT 1-D bodies (one source of truth for every kernel set) ---- */
#define JD_ISLOW_1D(T, ADD, SUB, MUL, SHL13, DESC, sh, i0, i1, i2, i3, i4, i5, i6, i7, o0, o1, o2, o3, o4, o5, o6, o7) do { \
    T z1_, z2_, z3_, z4_, z5_, t0_, t1_, t2_, t3_, t10_, t11_, t12_, t13_;                       \
    z2_ = (i2); z3_ = (i6);                                                                          \
    z1_ = MUL(ADD(z2_, z3_), 4433);                                                                  \
    t2_ = ADD(z1_, MUL(z3_, -15137));                                                                \
    t3_ = ADD(z1_, MUL(z2_, 6270));                                                                  \
    t0_ = SHL13(ADD((i0), (i4)));                                                                    \
    t1_ = SHL13(SUB((i0), (i4)));                                                                    \
    t10_ = ADD(t0_, t3_); t13_ = SUB(t0_, t3_); t11_ = ADD(t1_, t2_); t12_ = SUB(t1_, t2_);          \
    t0_ = (i7); t1_ = (i5); t2_ = (i3); t3_ = (i1);                                                  \
    z1_ = ADD(t0_, t3_); z2_ = ADD(t1_, t2_); z3_ = ADD(t0_, t2_); z4_ = ADD(t1_, t3_);              \
    z5_ = MUL(ADD(z3_, z4_), 9633);                                                                  \
    t0_ = MUL(t0_, 2446); t1_ = MUL(t1_, 16819); t2_ = MUL(t2_, 25172); t3_ = MUL(t3_, 12299);       \
    z1_ = MUL(z1_, -7373); z2_ = MUL(z2_, -20995); z3_ = MUL(z3_, -16069); z4_ = MUL(z4_, -3196);    \
    z3_ = ADD(z3_, z5_); z4_ = ADD(z4_, z5_);                                                        \
    t0_ = ADD(t0_, ADD(z1_, z3_)); t1_ = ADD(t1_, ADD(z2_, z4_));                                    \
    t2_ = ADD(t2_, ADD(z2_, z3_)); t3_ = ADD(t3_, ADD(z1_, z4_));                                    \
    o0 = DESC(ADD(t10_, t3_), sh); o7 = DESC(SUB(t10_, t3_), sh);                                    \
    o1 = DESC(ADD(t11_, t2_), sh); o6 = DESC(SUB(t11_, t2_), sh);                                    \
    o2 = DESC(ADD(t12_, t1_), sh); o5 = DESC(SUB(t12_, t1_), sh);                                    \
    o3 = DESC(ADD(t13_, t0_), sh); o4 = DESC(SUB(t13_, t0_), sh);                                    \
} while (0)

#define JD_FLT_1D(T, ADD, SUB, MULK, i0, i1, i2, i3, i4, i5, i6, i7, o0, o1, o2, o3, o4, o5, o6, o7) do { \
    T t0_ = (i0), t1_ = (i2), t2_ = (i4), t3_ = (i6);                                                \
    T t10_ = ADD(t0_, t2_), t11_ = SUB(t0_, t2_);                                                    \
    T t13_ = ADD(t1_, t3_), t12_ = SUB(MULK(SUB(t1_, t3_), 1.414213562f), t13_);                     \
    t0_ = ADD(t10_, t13_); t3_ = SUB(t10_, t13_); t1_ = ADD(t11_, t12_); t2_ = SUB(t11_, t12_);      \
    T t4_ = (i1), t5_ = (i3), t6_ = (i5), t7_ = (i7);                                                \
    T z13_ = ADD(t6_, t5_), z10_ = SUB(t6_, t5_), z11_ = ADD(t4_, t7_), z12_ = SUB(t4_, t7_);        \
    t7_ = ADD(z11_, z13_);                                                                           \
    T u11_ = MULK(SUB(z11_, z13_), 1.414213562f);                                                    \
    T z5_ = MULK(ADD(z10_, z12_), 1.847759065f);                                                     \
    T u10_ = SUB(z5_, MULK(z12_, 1.082392200f));                                                     \
    T u12_ = SUB(z5_, MULK(z10_, 2.613125930f));                                                     \
    t6_ = SUB(u12_, t7_); t5_ = SUB(u11_, t6_); t4_ = SUB(u10_, t5_);                                \
    o0 = ADD(t0_, t7_); o7 = SUB(t0_, t7_); o1 = ADD(t1_, t6_); o6 = SUB(t1_, t6_);                  \
    o2 = ADD(t2_, t5_); o5 = SUB(t2_, t5_); o3 = ADD(t3_, t4_); o4 = SUB(t3_, t4_);                  \
} while (0)

/* ---- scalar kernels ---- */
#define JDS_ADD(a, b)   ((uint32_t)(a) + (uint32_t)(b))
#define JDS_SUB(a, b)   ((uint32_t)(a) - (uint32_t)(b))
#define JDS_MUL(a, c)   ((uint32_t)(a) * (uint32_t)(int32_t)(c))
#define JDS_SHL13(a)    ((uint32_t)(a) << 13)
#define JDS_DESC(x, n)  ((uint32_t)((int32_t)((uint32_t)(x) + (1u << ((n) - 1))) >> (n)))
JD_INLINE int32_t jd_sat16(int32_t v) { return v < -32768 ? -32768 : v > 32767 ? 32767 : v; }
JD_INLINE uint8_t jd_clamp8(int32_t v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }

static void jd_idct_islow_c(const int16_t *coef, const uint16_t *q, uint8_t *out, ptrdiff_t stride) {
    int32_t ws[64];
    for (int c = 0; c < 8; c++) {
        uint32_t in[8], o[8];
        for (int r = 0; r < 8; r++) in[r] = (uint32_t)jd_sat16((int32_t)coef[r * 8 + c] * (int32_t)q[r * 8 + c]);
        JD_ISLOW_1D(uint32_t, JDS_ADD, JDS_SUB, JDS_MUL, JDS_SHL13, JDS_DESC, 11,
                    in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
        for (int r = 0; r < 8; r++) ws[r * 8 + c] = jd_sat16((int32_t)o[r]);
    }
    for (int y = 0; y < 8; y++) {
        const int32_t *w = ws + y * 8;
        uint32_t o[8];
        JD_ISLOW_1D(uint32_t, JDS_ADD, JDS_SUB, JDS_MUL, JDS_SHL13, JDS_DESC, 18,
                    (uint32_t)w[0], (uint32_t)w[1], (uint32_t)w[2], (uint32_t)w[3], (uint32_t)w[4], (uint32_t)w[5],
                    (uint32_t)w[6], (uint32_t)w[7], o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
        for (int x = 0; x < 8; x++) out[y * stride + x] = jd_clamp8((int32_t)o[x] + 128);
    }
}

#define JDF_ADD(a, b)  ((a) + (b))
#define JDF_SUB(a, b)  ((a) - (b))
#define JDF_MULK(a, k) ((a) * (k))
/* cvttps2dq semantics: NaN / out of range -> INT32_MIN */
#define JDF_CVTT(v) (((v) >= 2147483648.0f || !((v) >= -2147483648.0f)) ? INT32_MIN : (int32_t)(v))
JD_FPSSE static void jd_idct_float_c(const int16_t *coef, const float *fq, uint8_t *out, ptrdiff_t stride) {
    float ws[64];
    for (int c = 0; c < 8; c++) {
        float in[8], o[8];
        for (int r = 0; r < 8; r++) in[r] = (float)coef[r * 8 + c] * fq[r * 8 + c];
        JD_FLT_1D(float, JDF_ADD, JDF_SUB, JDF_MULK, in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7],
                  o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
        for (int r = 0; r < 8; r++) ws[r * 8 + c] = o[r];
    }
    for (int y = 0; y < 8; y++) {
        const float *w = ws + y * 8;
        float o[8];
        float b0 = w[0] + 128.5f;
        JD_FLT_1D(float, JDF_ADD, JDF_SUB, JDF_MULK, b0, w[1], w[2], w[3], w[4], w[5], w[6], w[7],
                  o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
        for (int x = 0; x < 8; x++) {
            int32_t v = JDF_CVTT(o[x]);
            out[y * stride + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
}

/* libjpeg-turbo jdsample.c h2v1_fancy_upsample (triangle filter, 3/4 nearer + 1/4 further, alternating
 * rounding bias 1/2; edge samples replicated) */
static void jd_h2v1_fancy_c(const uint8_t *in, int w, uint8_t *out) {
    out[0] = in[0];
    out[1] = (uint8_t)((in[0] * 3 + in[1] + 2) >> 2);
    for (int i = 1; i < w - 1; i++) {
        int v = in[i] * 3;
        out[2 * i] = (uint8_t)((v + in[i - 1] + 1) >> 2);
        out[2 * i + 1] = (uint8_t)((v + in[i + 1] + 2) >> 2);
    }
    out[2 * w - 2] = (uint8_t)((in[w - 1] * 3 + in[w - 2] + 1) >> 2);
    out[2 * w - 1] = in[w - 1];
}
/* h2v2_fancy_upsample: column sums 3*near + far, then the same triangle horizontally, bias 8 / 7 */
static void jd_h2v2_fancy_c(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out) {
    int cs = in0[0] * 3 + in1[0], ns = in0[1] * 3 + in1[1], ls;
    out[0] = (uint8_t)((cs * 4 + 8) >> 4);
    out[1] = (uint8_t)((cs * 3 + ns + 7) >> 4);
    for (int i = 1; i < w - 1; i++) {
        ls = cs; cs = ns; ns = in0[i + 1] * 3 + in1[i + 1];
        out[2 * i] = (uint8_t)((cs * 3 + ls + 8) >> 4);
        out[2 * i + 1] = (uint8_t)((cs * 3 + ns + 7) >> 4);
    }
    ls = cs; cs = ns;
    out[2 * w - 2] = (uint8_t)((cs * 3 + ls + 8) >> 4);
    out[2 * w - 1] = (uint8_t)((cs * 4 + 7) >> 4);
}
/* h1v2_fancy_upsample: (3*near + far + bias) >> 2, bias 1 for the upper row, 2 for the lower */
static void jd_h1v2_fancy_c(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out, int lower) {
    int bias = lower ? 2 : 1;
    for (int i = 0; i < w; i++) out[i] = (uint8_t)((in0[i] * 3 + in1[i] + bias) >> 2);
}
/* jdcolor.c ycc_rgb_convert with SCALEBITS 16: Cr_r = (FIX(1.402)x + 1/2)>>16, Cb_b = (FIX(1.772)x + 1/2)>>16,
 * G = y + ((-FIX(0.34414)cb + 1/2 - FIX(0.71414)cr) >> 16); FIX(1.402)=91881 FIX(1.772)=116130
 * FIX(0.71414)=46802 FIX(0.34414)=22554 */
JD_INLINE void jd_ycc1(int y, int cb, int cr, uint8_t *o) {
    int xb = cb - 128, xr = cr - 128;
    o[0] = jd_clamp8(y + ((91881 * xr + 32768) >> 16));
    o[1] = jd_clamp8(y + ((-22554 * xb - 46802 * xr + 32768) >> 16));
    o[2] = jd_clamp8(y + ((116130 * xb + 32768) >> 16));
}
static void jd_ycc_rgb_c(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) jd_ycc1(y[i], cb[i], cr[i], out + 3 * i);
}
static void jd_ycc_rgba_c(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) { jd_ycc1(y[i], cb[i], cr[i], out + 4 * i); out[4 * i + 3] = 255; }
}

/* ---- SSE2 kernels ---- */
JD_INLINE __m128i jd_mullo32k_sse2(__m128i a, int32_t c) {
    __m128i k = _mm_set1_epi32(c);
    __m128i p02 = _mm_mul_epu32(a, k);
    __m128i p13 = _mm_mul_epu32(_mm_srli_epi64(a, 32), k);
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(p02, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(p13, _MM_SHUFFLE(0, 0, 2, 0)));
}
JD_INLINE __m128i jd_mullo32_sse2(__m128i a, __m128i b) {
    __m128i p02 = _mm_mul_epu32(a, b);
    __m128i p13 = _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32));
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(p02, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(p13, _MM_SHUFFLE(0, 0, 2, 0)));
}
#define JDX_ADD(a, b)  _mm_add_epi32((a), (b))
#define JDX_SUB(a, b)  _mm_sub_epi32((a), (b))
#define JDX_MUL(a, c)  jd_mullo32k_sse2((a), (c))
#define JDX_SHL13(a)   _mm_slli_epi32((a), 13)
#define JDX_DESC(x, n) _mm_srai_epi32(_mm_add_epi32((x), _mm_set1_epi32(1 << ((n) - 1))), (n))
#define JD_TR4_EPI32(a, b, c, d) do {                                              \
    __m128i t0_ = _mm_unpacklo_epi32(a, b), t1_ = _mm_unpacklo_epi32(c, d);        \
    __m128i t2_ = _mm_unpackhi_epi32(a, b), t3_ = _mm_unpackhi_epi32(c, d);        \
    a = _mm_unpacklo_epi64(t0_, t1_); b = _mm_unpackhi_epi64(t0_, t1_);            \
    c = _mm_unpacklo_epi64(t2_, t3_); d = _mm_unpackhi_epi64(t2_, t3_);            \
} while (0)
JD_INLINE __m128i jd_sat16x4_sse2(__m128i v) { __m128i p = _mm_packs_epi32(v, v); return _mm_srai_epi32(_mm_unpacklo_epi16(p, p), 16); }

/* 8x8 transpose of int32 held as m[row][half] */
JD_INLINE void jd_tr8_sse2(__m128i m[8][2]) {
    JD_TR4_EPI32(m[0][0], m[1][0], m[2][0], m[3][0]);
    JD_TR4_EPI32(m[4][1], m[5][1], m[6][1], m[7][1]);
    JD_TR4_EPI32(m[0][1], m[1][1], m[2][1], m[3][1]);
    JD_TR4_EPI32(m[4][0], m[5][0], m[6][0], m[7][0]);
    for (int i = 0; i < 4; i++) { __m128i t = m[i][1]; m[i][1] = m[4 + i][0]; m[4 + i][0] = t; }
}

static void jd_idct_islow_sse2(const int16_t *coef, const uint16_t *q, uint8_t *out, ptrdiff_t stride) {
    __m128i m[8][2], o[8][2];
    const __m128i z = _mm_setzero_si128();
    for (int r = 0; r < 8; r++) {
        __m128i c = _mm_loadu_si128((const __m128i *)(coef + 8 * r));
        __m128i qq = _mm_loadu_si128((const __m128i *)(q + 8 * r));
        __m128i c0 = _mm_srai_epi32(_mm_unpacklo_epi16(c, c), 16), c1 = _mm_srai_epi32(_mm_unpackhi_epi16(c, c), 16);
        __m128i q0 = _mm_unpacklo_epi16(qq, z), q1 = _mm_unpackhi_epi16(qq, z);
        __m128i p = _mm_packs_epi32(jd_mullo32_sse2(c0, q0), jd_mullo32_sse2(c1, q1));
        m[r][0] = _mm_srai_epi32(_mm_unpacklo_epi16(p, p), 16);
        m[r][1] = _mm_srai_epi32(_mm_unpackhi_epi16(p, p), 16);
    }
    for (int h = 0; h < 2; h++) {
        JD_ISLOW_1D(__m128i, JDX_ADD, JDX_SUB, JDX_MUL, JDX_SHL13, JDX_DESC, 11,
                    m[0][h], m[1][h], m[2][h], m[3][h], m[4][h], m[5][h], m[6][h], m[7][h],
                    o[0][h], o[1][h], o[2][h], o[3][h], o[4][h], o[5][h], o[6][h], o[7][h]);
    }
    for (int r = 0; r < 8; r++) { o[r][0] = jd_sat16x4_sse2(o[r][0]); o[r][1] = jd_sat16x4_sse2(o[r][1]); }
    jd_tr8_sse2(o);   /* o[c][half] = column c, rows 4h..4h+3 */
    for (int h = 0; h < 2; h++) {
        JD_ISLOW_1D(__m128i, JDX_ADD, JDX_SUB, JDX_MUL, JDX_SHL13, JDX_DESC, 18,
                    o[0][h], o[1][h], o[2][h], o[3][h], o[4][h], o[5][h], o[6][h], o[7][h],
                    m[0][h], m[1][h], m[2][h], m[3][h], m[4][h], m[5][h], m[6][h], m[7][h]);
    }
    jd_tr8_sse2(m);   /* m[y][half] = row y, columns 4h.. */
    const __m128i k128 = _mm_set1_epi32(128);
    for (int y = 0; y < 8; y++) {
        __m128i p = _mm_packs_epi32(_mm_add_epi32(m[y][0], k128), _mm_add_epi32(m[y][1], k128));
        _mm_storel_epi64((__m128i *)(out + y * stride), _mm_packus_epi16(p, p));
    }
}

#define JDXF_ADD(a, b)  _mm_add_ps((a), (b))
#define JDXF_SUB(a, b)  _mm_sub_ps((a), (b))
#define JDXF_MULK(a, k) _mm_mul_ps((a), _mm_set1_ps(k))
JD_INLINE void jd_tr8f_sse2(__m128 m[8][2]) {
    _MM_TRANSPOSE4_PS(m[0][0], m[1][0], m[2][0], m[3][0]);
    _MM_TRANSPOSE4_PS(m[4][1], m[5][1], m[6][1], m[7][1]);
    _MM_TRANSPOSE4_PS(m[0][1], m[1][1], m[2][1], m[3][1]);
    _MM_TRANSPOSE4_PS(m[4][0], m[5][0], m[6][0], m[7][0]);
    for (int i = 0; i < 4; i++) { __m128 t = m[i][1]; m[i][1] = m[4 + i][0]; m[4 + i][0] = t; }
}
static void jd_idct_float_sse2(const int16_t *coef, const float *fq, uint8_t *out, ptrdiff_t stride) {
    __m128 m[8][2], o[8][2];
    for (int r = 0; r < 8; r++) {
        __m128i c = _mm_loadu_si128((const __m128i *)(coef + 8 * r));
        __m128i c0 = _mm_srai_epi32(_mm_unpacklo_epi16(c, c), 16), c1 = _mm_srai_epi32(_mm_unpackhi_epi16(c, c), 16);
        m[r][0] = _mm_mul_ps(_mm_cvtepi32_ps(c0), _mm_loadu_ps(fq + 8 * r));
        m[r][1] = _mm_mul_ps(_mm_cvtepi32_ps(c1), _mm_loadu_ps(fq + 8 * r + 4));
    }
    for (int h = 0; h < 2; h++)
        JD_FLT_1D(__m128, JDXF_ADD, JDXF_SUB, JDXF_MULK, m[0][h], m[1][h], m[2][h], m[3][h], m[4][h], m[5][h], m[6][h], m[7][h],
                  o[0][h], o[1][h], o[2][h], o[3][h], o[4][h], o[5][h], o[6][h], o[7][h]);
    jd_tr8f_sse2(o);
    for (int h = 0; h < 2; h++) {
        __m128 b0 = _mm_add_ps(o[0][h], _mm_set1_ps(128.5f));
        JD_FLT_1D(__m128, JDXF_ADD, JDXF_SUB, JDXF_MULK, b0, o[1][h], o[2][h], o[3][h], o[4][h], o[5][h], o[6][h], o[7][h],
                  m[0][h], m[1][h], m[2][h], m[3][h], m[4][h], m[5][h], m[6][h], m[7][h]);
    }
    jd_tr8f_sse2(m);
    for (int y = 0; y < 8; y++) {
        __m128i p = _mm_packs_epi32(_mm_cvttps_epi32(m[y][0]), _mm_cvttps_epi32(m[y][1]));
        _mm_storel_epi64((__m128i *)(out + y * stride), _mm_packus_epi16(p, p));
    }
}

JD_INLINE __m128i jd_ld8_sse2(const uint8_t *p) { return _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)p), _mm_setzero_si128()); }
static void jd_h2v1_fancy_sse2(const uint8_t *in, int w, uint8_t *out) {
    out[0] = in[0];
    out[1] = (uint8_t)((in[0] * 3 + in[1] + 2) >> 2);
    int i = 1;
    const __m128i k1 = _mm_set1_epi16(1), k2 = _mm_set1_epi16(2);
    for (; i + 8 <= w - 1; i += 8) {
        __m128i c = jd_ld8_sse2(in + i), p = jd_ld8_sse2(in + i - 1), n = jd_ld8_sse2(in + i + 1);
        __m128i c3 = _mm_add_epi16(c, _mm_add_epi16(c, c));
        __m128i e = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(c3, p), k1), 2);
        __m128i o = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(c3, n), k2), 2);
        __m128i lo = _mm_unpacklo_epi16(e, o), hi = _mm_unpackhi_epi16(e, o);
        _mm_storeu_si128((__m128i *)(out + 2 * i), _mm_packus_epi16(lo, hi));
    }
    for (; i < w - 1; i++) {
        int v = in[i] * 3;
        out[2 * i] = (uint8_t)((v + in[i - 1] + 1) >> 2);
        out[2 * i + 1] = (uint8_t)((v + in[i + 1] + 2) >> 2);
    }
    out[2 * w - 2] = (uint8_t)((in[w - 1] * 3 + in[w - 2] + 1) >> 2);
    out[2 * w - 1] = in[w - 1];
}
static void jd_h2v2_fancy_sse2(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out) {
    int cs0 = in0[0] * 3 + in1[0], cs1 = in0[1] * 3 + in1[1];
    out[0] = (uint8_t)((cs0 * 4 + 8) >> 4);
    out[1] = (uint8_t)((cs0 * 3 + cs1 + 7) >> 4);
    int i = 1;
    const __m128i k8 = _mm_set1_epi16(8), k7 = _mm_set1_epi16(7);
#define JD_CS_SSE2(off) _mm_add_epi16(_mm_add_epi16(jd_ld8_sse2(in0 + (off)), _mm_add_epi16(jd_ld8_sse2(in0 + (off)), jd_ld8_sse2(in0 + (off)))), jd_ld8_sse2(in1 + (off)))
    for (; i + 8 <= w - 1; i += 8) {
        __m128i c = JD_CS_SSE2(i), p = JD_CS_SSE2(i - 1), n = JD_CS_SSE2(i + 1);
        __m128i c3 = _mm_add_epi16(c, _mm_add_epi16(c, c));
        __m128i e = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(c3, p), k8), 4);
        __m128i o = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(c3, n), k7), 4);
        _mm_storeu_si128((__m128i *)(out + 2 * i), _mm_packus_epi16(_mm_unpacklo_epi16(e, o), _mm_unpackhi_epi16(e, o)));
    }
#undef JD_CS_SSE2
    for (; i < w - 1; i++) {
        int ls = in0[i - 1] * 3 + in1[i - 1], cs = in0[i] * 3 + in1[i], ns = in0[i + 1] * 3 + in1[i + 1];
        out[2 * i] = (uint8_t)((cs * 3 + ls + 8) >> 4);
        out[2 * i + 1] = (uint8_t)((cs * 3 + ns + 7) >> 4);
    }
    {
        int ls = in0[w - 2] * 3 + in1[w - 2], cs = in0[w - 1] * 3 + in1[w - 1];
        out[2 * w - 2] = (uint8_t)((cs * 3 + ls + 8) >> 4);
        out[2 * w - 1] = (uint8_t)((cs * 4 + 7) >> 4);
    }
}
static void jd_h1v2_fancy_sse2(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out, int lower) {
    int bias = lower ? 2 : 1, i = 0;
    const __m128i kb = _mm_set1_epi16((short)bias), z = _mm_setzero_si128();
    for (; i + 16 <= w; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(in0 + i)), b = _mm_loadu_si128((const __m128i *)(in1 + i));
        __m128i al = _mm_unpacklo_epi8(a, z), ah = _mm_unpackhi_epi8(a, z), bl = _mm_unpacklo_epi8(b, z), bh = _mm_unpackhi_epi8(b, z);
        __m128i rl = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(al, _mm_add_epi16(al, al)), bl), kb), 2);
        __m128i rh = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(ah, _mm_add_epi16(ah, ah)), bh), kb), 2);
        _mm_storeu_si128((__m128i *)(out + i), _mm_packus_epi16(rl, rh));
    }
    for (; i < w; i++) out[i] = (uint8_t)((in0[i] * 3 + in1[i] + bias) >> 2);
}
/* colour conversion on 8 int16 lanes. Derivation (exact, x = sample-128, |x| <= 128):
 *   (91881x + 32768) >> 16  = x + ((26345x + 32768) >> 16)        (91881 = 65536 + 26345)
 *   (116130x + 32768) >> 16 = 2x + ((-14942x + 32768) >> 16)      (116130 = 131072 - 14942)
 *   (-22554xb - 46802xr + 32768) >> 16 = -xr + ((18734xr - 22554xb + 32768) >> 16)
 *   ((P + 32768) >> 16) for P = c*x computed as mulhi + (mullo >> 15, logical)  */
JD_INLINE void jd_ycc8_sse2(__m128i y, __m128i cb, __m128i cr, __m128i *R, __m128i *G, __m128i *B) {
    const __m128i k128 = _mm_set1_epi16(128);
    __m128i xb = _mm_sub_epi16(cb, k128), xr = _mm_sub_epi16(cr, k128);
    const __m128i kr = _mm_set1_epi16(26345), kb = _mm_set1_epi16(-14942);
    __m128i crr = _mm_add_epi16(xr, _mm_add_epi16(_mm_mulhi_epi16(xr, kr), _mm_srli_epi16(_mm_mullo_epi16(xr, kr), 15)));
    __m128i cbb = _mm_add_epi16(_mm_add_epi16(xb, xb), _mm_add_epi16(_mm_mulhi_epi16(xb, kb), _mm_srli_epi16(_mm_mullo_epi16(xb, kb), 15)));
    const __m128i kg = _mm_set_epi16(-22554, 18734, -22554, 18734, -22554, 18734, -22554, 18734);
    const __m128i half = _mm_set1_epi32(32768);
    __m128i g0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(xr, xb), kg), half), 16);
    __m128i g1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(xr, xb), kg), half), 16);
    __m128i cg = _mm_sub_epi16(_mm_packs_epi32(g0, g1), xr);
    *R = _mm_add_epi16(y, crr); *G = _mm_add_epi16(y, cg); *B = _mm_add_epi16(y, cbb);
}
/* 16 pixels -> R,G,B byte vectors */
JD_INLINE void jd_ycc16_sse2(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, __m128i *R, __m128i *G, __m128i *B) {
    const __m128i z = _mm_setzero_si128();
    __m128i Y = _mm_loadu_si128((const __m128i *)y), U = _mm_loadu_si128((const __m128i *)cb), V = _mm_loadu_si128((const __m128i *)cr);
    __m128i r0, g0, b0, r1, g1, b1;
    jd_ycc8_sse2(_mm_unpacklo_epi8(Y, z), _mm_unpacklo_epi8(U, z), _mm_unpacklo_epi8(V, z), &r0, &g0, &b0);
    jd_ycc8_sse2(_mm_unpackhi_epi8(Y, z), _mm_unpackhi_epi8(U, z), _mm_unpackhi_epi8(V, z), &r1, &g1, &b1);
    *R = _mm_packus_epi16(r0, r1); *G = _mm_packus_epi16(g0, g1); *B = _mm_packus_epi16(b0, b1);
}
JD_INLINE void jd_store_rgba16_sse2(uint8_t *o, __m128i R, __m128i G, __m128i B) {
    const __m128i A = _mm_set1_epi8((char)0xFF);
    __m128i rg0 = _mm_unpacklo_epi8(R, G), rg1 = _mm_unpackhi_epi8(R, G);
    __m128i ba0 = _mm_unpacklo_epi8(B, A), ba1 = _mm_unpackhi_epi8(B, A);
    _mm_storeu_si128((__m128i *)(o + 0), _mm_unpacklo_epi16(rg0, ba0));
    _mm_storeu_si128((__m128i *)(o + 16), _mm_unpackhi_epi16(rg0, ba0));
    _mm_storeu_si128((__m128i *)(o + 32), _mm_unpacklo_epi16(rg1, ba1));
    _mm_storeu_si128((__m128i *)(o + 48), _mm_unpackhi_epi16(rg1, ba1));
}
static void jd_ycc_rgba_sse2(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i R, G, B;
        jd_ycc16_sse2(y + i, cb + i, cr + i, &R, &G, &B);
        jd_store_rgba16_sse2(out + 4 * i, R, G, B);
    }
    for (; i < n; i++) { jd_ycc1(y[i], cb[i], cr[i], out + 4 * i); out[4 * i + 3] = 255; }
}
static void jd_ycc_rgb_sse2(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n) {
    int i = 0;
    uint8_t tmp[64] __attribute__((aligned(16)));
    for (; i + 16 <= n; i += 16) {
        __m128i R, G, B;
        jd_ycc16_sse2(y + i, cb + i, cr + i, &R, &G, &B);
        jd_store_rgba16_sse2(tmp, R, G, B);
        uint8_t *o = out + 3 * i;
        for (int j = 0; j < 16; j++) { o[3 * j] = tmp[4 * j]; o[3 * j + 1] = tmp[4 * j + 1]; o[3 * j + 2] = tmp[4 * j + 2]; }
    }
    for (; i < n; i++) jd_ycc1(y[i], cb[i], cr[i], out + 3 * i);
}

/* ---- AVX2 kernels (only called after the caller checked AVX2 is usable) ---- */
#define JDY_ADD(a, b)  _mm256_add_epi32((a), (b))
#define JDY_SUB(a, b)  _mm256_sub_epi32((a), (b))
#define JDY_MUL(a, c)  _mm256_mullo_epi32((a), _mm256_set1_epi32(c))
#define JDY_SHL13(a)   _mm256_slli_epi32((a), 13)
#define JDY_DESC(x, n) _mm256_srai_epi32(_mm256_add_epi32((x), _mm256_set1_epi32(1 << ((n) - 1))), (n))
#define JD_TR8_AVX2(T, UNLO, UNHI, UNLO64, UNHI64, PERM, r) do {                                        \
    T t0_ = UNLO(r[0], r[1]), t1_ = UNHI(r[0], r[1]), t2_ = UNLO(r[2], r[3]), t3_ = UNHI(r[2], r[3]);    \
    T t4_ = UNLO(r[4], r[5]), t5_ = UNHI(r[4], r[5]), t6_ = UNLO(r[6], r[7]), t7_ = UNHI(r[6], r[7]);    \
    T u0_ = UNLO64(t0_, t2_), u1_ = UNHI64(t0_, t2_), u2_ = UNLO64(t1_, t3_), u3_ = UNHI64(t1_, t3_);     \
    T u4_ = UNLO64(t4_, t6_), u5_ = UNHI64(t4_, t6_), u6_ = UNLO64(t5_, t7_), u7_ = UNHI64(t5_, t7_);     \
    r[0] = PERM(u0_, u4_, 0x20); r[1] = PERM(u1_, u5_, 0x20); r[2] = PERM(u2_, u6_, 0x20); r[3] = PERM(u3_, u7_, 0x20); \
    r[4] = PERM(u0_, u4_, 0x31); r[5] = PERM(u1_, u5_, 0x31); r[6] = PERM(u2_, u6_, 0x31); r[7] = PERM(u3_, u7_, 0x31); \
} while (0)
#define JD_UNLO64I(a, b) _mm256_unpacklo_epi64(a, b)
#define JD_UNHI64I(a, b) _mm256_unpackhi_epi64(a, b)
#define JD_UNLO64F(a, b) _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)))
#define JD_UNHI64F(a, b) _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)))

JD_AVX2 static void jd_store2rows_avx2(uint8_t *o, ptrdiff_t stride, __m256i a, __m256i b) {
    __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), _MM_SHUFFLE(3, 1, 2, 0));
    p = _mm256_packus_epi16(p, p);
    _mm_storel_epi64((__m128i *)o, _mm256_castsi256_si128(p));
    _mm_storel_epi64((__m128i *)(o + stride), _mm256_extracti128_si256(p, 1));
}
JD_AVX2 static void jd_idct_islow_avx2(const int16_t *coef, const uint16_t *q, uint8_t *out, ptrdiff_t stride) {
    __m256i m[8], o[8];
    const __m256i lo16 = _mm256_set1_epi32(-32768), hi16 = _mm256_set1_epi32(32767);
    for (int r = 0; r < 8; r++) {
        __m256i c = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(coef + 8 * r)));
        __m256i qq = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(q + 8 * r)));
        m[r] = _mm256_max_epi32(_mm256_min_epi32(_mm256_mullo_epi32(c, qq), hi16), lo16);
    }
    JD_ISLOW_1D(__m256i, JDY_ADD, JDY_SUB, JDY_MUL, JDY_SHL13, JDY_DESC, 11,
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
    for (int r = 0; r < 8; r++) o[r] = _mm256_max_epi32(_mm256_min_epi32(o[r], hi16), lo16);
    JD_TR8_AVX2(__m256i, _mm256_unpacklo_epi32, _mm256_unpackhi_epi32, JD_UNLO64I, JD_UNHI64I, _mm256_permute2x128_si256, o);
    JD_ISLOW_1D(__m256i, JDY_ADD, JDY_SUB, JDY_MUL, JDY_SHL13, JDY_DESC, 18,
                o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
    JD_TR8_AVX2(__m256i, _mm256_unpacklo_epi32, _mm256_unpackhi_epi32, JD_UNLO64I, JD_UNHI64I, _mm256_permute2x128_si256, m);
    const __m256i k128 = _mm256_set1_epi32(128);
    for (int y = 0; y < 8; y += 2)
        jd_store2rows_avx2(out + y * stride, stride, _mm256_add_epi32(m[y], k128), _mm256_add_epi32(m[y + 1], k128));
}
#define JDYF_ADD(a, b)  _mm256_add_ps((a), (b))
#define JDYF_SUB(a, b)  _mm256_sub_ps((a), (b))
#define JDYF_MULK(a, k) _mm256_mul_ps((a), _mm256_set1_ps(k))
JD_AVX2 static void jd_idct_float_avx2(const int16_t *coef, const float *fq, uint8_t *out, ptrdiff_t stride) {
    __m256 m[8], o[8];
    for (int r = 0; r < 8; r++) {
        __m256i c = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(coef + 8 * r)));
        m[r] = _mm256_mul_ps(_mm256_cvtepi32_ps(c), _mm256_loadu_ps(fq + 8 * r));
    }
    JD_FLT_1D(__m256, JDYF_ADD, JDYF_SUB, JDYF_MULK, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
              o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
    JD_TR8_AVX2(__m256, _mm256_unpacklo_ps, _mm256_unpackhi_ps, JD_UNLO64F, JD_UNHI64F, _mm256_permute2f128_ps, o);
    __m256 b0 = _mm256_add_ps(o[0], _mm256_set1_ps(128.5f));
    JD_FLT_1D(__m256, JDYF_ADD, JDYF_SUB, JDYF_MULK, b0, o[1], o[2], o[3], o[4], o[5], o[6], o[7],
              m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
    JD_TR8_AVX2(__m256, _mm256_unpacklo_ps, _mm256_unpackhi_ps, JD_UNLO64F, JD_UNHI64F, _mm256_permute2f128_ps, m);
    for (int y = 0; y < 8; y += 2)
        jd_store2rows_avx2(out + y * stride, stride, _mm256_cvttps_epi32(m[y]), _mm256_cvttps_epi32(m[y + 1]));
}
JD_AVX2 static inline __m256i jd_ld16_avx2(const uint8_t *p) { return _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p)); }
JD_AVX2 static void jd_h2v1_fancy_avx2(const uint8_t *in, int w, uint8_t *out) {
    out[0] = in[0];
    out[1] = (uint8_t)((in[0] * 3 + in[1] + 2) >> 2);
    int i = 1;
    const __m256i k1 = _mm256_set1_epi16(1), k2 = _mm256_set1_epi16(2);
    for (; i + 16 <= w - 1; i += 16) {
        __m256i c = jd_ld16_avx2(in + i), p = jd_ld16_avx2(in + i - 1), n = jd_ld16_avx2(in + i + 1);
        __m256i c3 = _mm256_add_epi16(c, _mm256_add_epi16(c, c));
        __m256i e = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(c3, p), k1), 2);
        __m256i o = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(c3, n), k2), 2);
        _mm256_storeu_si256((__m256i *)(out + 2 * i), _mm256_packus_epi16(_mm256_unpacklo_epi16(e, o), _mm256_unpackhi_epi16(e, o)));
    }
    for (; i < w - 1; i++) {
        int v = in[i] * 3;
        out[2 * i] = (uint8_t)((v + in[i - 1] + 1) >> 2);
        out[2 * i + 1] = (uint8_t)((v + in[i + 1] + 2) >> 2);
    }
    out[2 * w - 2] = (uint8_t)((in[w - 1] * 3 + in[w - 2] + 1) >> 2);
    out[2 * w - 1] = in[w - 1];
}
JD_AVX2 static void jd_h2v2_fancy_avx2(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out) {
    int cs0 = in0[0] * 3 + in1[0], cs1 = in0[1] * 3 + in1[1];
    out[0] = (uint8_t)((cs0 * 4 + 8) >> 4);
    out[1] = (uint8_t)((cs0 * 3 + cs1 + 7) >> 4);
    int i = 1;
    const __m256i k8 = _mm256_set1_epi16(8), k7 = _mm256_set1_epi16(7);
#define JD_CS_AVX2(off) _mm256_add_epi16(_mm256_add_epi16(jd_ld16_avx2(in0 + (off)), _mm256_add_epi16(jd_ld16_avx2(in0 + (off)), jd_ld16_avx2(in0 + (off)))), jd_ld16_avx2(in1 + (off)))
    for (; i + 16 <= w - 1; i += 16) {
        __m256i c = JD_CS_AVX2(i), p = JD_CS_AVX2(i - 1), n = JD_CS_AVX2(i + 1);
        __m256i c3 = _mm256_add_epi16(c, _mm256_add_epi16(c, c));
        __m256i e = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(c3, p), k8), 4);
        __m256i o = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(c3, n), k7), 4);
        _mm256_storeu_si256((__m256i *)(out + 2 * i), _mm256_packus_epi16(_mm256_unpacklo_epi16(e, o), _mm256_unpackhi_epi16(e, o)));
    }
#undef JD_CS_AVX2
    for (; i < w - 1; i++) {
        int ls = in0[i - 1] * 3 + in1[i - 1], cs = in0[i] * 3 + in1[i], ns = in0[i + 1] * 3 + in1[i + 1];
        out[2 * i] = (uint8_t)((cs * 3 + ls + 8) >> 4);
        out[2 * i + 1] = (uint8_t)((cs * 3 + ns + 7) >> 4);
    }
    {
        int ls = in0[w - 2] * 3 + in1[w - 2], cs = in0[w - 1] * 3 + in1[w - 1];
        out[2 * w - 2] = (uint8_t)((cs * 3 + ls + 8) >> 4);
        out[2 * w - 1] = (uint8_t)((cs * 4 + 7) >> 4);
    }
}
JD_AVX2 static void jd_h1v2_fancy_avx2(const uint8_t *in0, const uint8_t *in1, int w, uint8_t *out, int lower) {
    int bias = lower ? 2 : 1, i = 0;
    const __m256i kb = _mm256_set1_epi16((short)bias);
    for (; i + 16 <= w; i += 16) {
        __m256i a = jd_ld16_avx2(in0 + i), b = jd_ld16_avx2(in1 + i);
        __m256i r = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(a, _mm256_add_epi16(a, a)), b), kb), 2);
        r = _mm256_permute4x64_epi64(_mm256_packus_epi16(r, r), _MM_SHUFFLE(3, 1, 2, 0));
        _mm_storeu_si128((__m128i *)(out + i), _mm256_castsi256_si128(r));
    }
    for (; i < w; i++) out[i] = (uint8_t)((in0[i] * 3 + in1[i] + bias) >> 2);
}
JD_AVX2 static void jd_ycc16_avx2(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, __m128i *R, __m128i *G, __m128i *B) {
    const __m256i k128 = _mm256_set1_epi16(128);
    __m256i Y = jd_ld16_avx2(y);
    __m256i xb = _mm256_sub_epi16(jd_ld16_avx2(cb), k128), xr = _mm256_sub_epi16(jd_ld16_avx2(cr), k128);
    const __m256i kr = _mm256_set1_epi16(26345), kb = _mm256_set1_epi16(-14942);
    __m256i crr = _mm256_add_epi16(xr, _mm256_add_epi16(_mm256_mulhi_epi16(xr, kr), _mm256_srli_epi16(_mm256_mullo_epi16(xr, kr), 15)));
    __m256i cbb = _mm256_add_epi16(_mm256_add_epi16(xb, xb), _mm256_add_epi16(_mm256_mulhi_epi16(xb, kb), _mm256_srli_epi16(_mm256_mullo_epi16(xb, kb), 15)));
    const __m256i kg = _mm256_set1_epi32((int)(((uint32_t)(uint16_t)-22554 << 16) | 18734u));
    const __m256i half = _mm256_set1_epi32(32768);
    __m256i g0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(xr, xb), kg), half), 16);
    __m256i g1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(xr, xb), kg), half), 16);
    __m256i cg = _mm256_sub_epi16(_mm256_packs_epi32(g0, g1), xr);
    __m256i r = _mm256_add_epi16(Y, crr), g = _mm256_add_epi16(Y, cg), b = _mm256_add_epi16(Y, cbb);
    /* packus per 128-bit lane, then gather the low qwords: bytes 0..15 in order */
    *R = _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(r, r), _MM_SHUFFLE(3, 1, 2, 0)));
    *G = _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(g, g), _MM_SHUFFLE(3, 1, 2, 0)));
    *B = _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(b, b), _MM_SHUFFLE(3, 1, 2, 0)));
}
JD_AVX2 static void jd_store_rgba16_avx2(uint8_t *o, __m128i R, __m128i G, __m128i B) {
    const __m128i A = _mm_set1_epi8((char)0xFF);
    __m128i rg0 = _mm_unpacklo_epi8(R, G), rg1 = _mm_unpackhi_epi8(R, G);
    __m128i ba0 = _mm_unpacklo_epi8(B, A), ba1 = _mm_unpackhi_epi8(B, A);
    _mm256_storeu_si256((__m256i *)o, _mm256_set_m128i(_mm_unpackhi_epi16(rg0, ba0), _mm_unpacklo_epi16(rg0, ba0)));
    _mm256_storeu_si256((__m256i *)(o + 32), _mm256_set_m128i(_mm_unpackhi_epi16(rg1, ba1), _mm_unpacklo_epi16(rg1, ba1)));
}
JD_AVX2 static void jd_ycc_rgba_avx2(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i R, G, B;
        jd_ycc16_avx2(y + i, cb + i, cr + i, &R, &G, &B);
        jd_store_rgba16_avx2(out + 4 * i, R, G, B);
    }
    for (; i < n; i++) { jd_ycc1(y[i], cb[i], cr[i], out + 4 * i); out[4 * i + 3] = 255; }
}
JD_AVX2 static void jd_ycc_rgb_avx2(const uint8_t *y, const uint8_t *cb, const uint8_t *cr, uint8_t *out, int n) {
    int i = 0;
    /* RGBX -> RGB inside each 16-byte group (4 pixels -> 12 bytes) */
    const __m128i shuf = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    uint8_t tmp[64] __attribute__((aligned(32)));
    for (; i + 16 <= n; i += 16) {
        __m128i R, G, B;
        jd_ycc16_avx2(y + i, cb + i, cr + i, &R, &G, &B);
        jd_store_rgba16_avx2(tmp, R, G, B);
        uint8_t *o = out + 3 * i;
        for (int j = 0; j < 4; j++) {
            __m128i v = _mm_shuffle_epi8(_mm_load_si128((const __m128i *)(tmp + 16 * j)), shuf);
            uint8_t t12[16];
            _mm_storeu_si128((__m128i *)t12, v);
            memcpy(o + 12 * j, t12, 12);
        }
    }
    for (; i < n; i++) jd_ycc1(y[i], cb[i], cr[i], out + 3 * i);
}

static const jd_kernels jd_kernels_scalar = { "scalar", jd_idct_islow_c, jd_idct_float_c, jd_h2v1_fancy_c, jd_h2v2_fancy_c,
                                              jd_h1v2_fancy_c, jd_ycc_rgb_c, jd_ycc_rgba_c };
static const jd_kernels jd_kernels_sse2 = { "sse2", jd_idct_islow_sse2, jd_idct_float_sse2, jd_h2v1_fancy_sse2, jd_h2v2_fancy_sse2,
                                            jd_h1v2_fancy_sse2, jd_ycc_rgb_sse2, jd_ycc_rgba_sse2 };
static const jd_kernels jd_kernels_avx2 = { "avx2", jd_idct_islow_avx2, jd_idct_float_avx2, jd_h2v1_fancy_avx2, jd_h2v2_fancy_avx2,
                                            jd_h1v2_fancy_avx2, jd_ycc_rgb_avx2, jd_ycc_rgba_avx2 };

/* AAN scale factors cos(k*pi/16)*sqrt(2), k>0; literal so every build uses the same float values */
static const float jd_aan[8] = { 1.0f, 1.387039845f, 1.306562965f, 1.175875602f, 1.0f, 0.785694958f, 0.541196100f, 0.275899379f };
JD_FPSSE static void jd_make_fq(const uint16_t *q, float *fq) {
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) fq[r * 8 + c] = (float)q[r * 8 + c] * jd_aan[r] * jd_aan[c] * 0.125f;
}

/* ---------------------------------------------------------------------------------------------
 * decoder
 * ------------------------------------------------------------------------------------------- */
/* jpeg_natural_order + 16 entries of 63 (libjpeg's guard against k overrun in corrupt data) */
static const uint8_t jd_zz[80] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63 };

/* Annex K.3 Huffman tables (identical to those cjpeg emits without -optimize).
 * The decoder pre-loads them into DC/AC slots 0 and 1 like libjpeg-turbo (std_huff_tables(): Motion-JPEG
 * streams omit DHT) */
static const uint8_t jd_std_dc_lum_bits[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
static const uint8_t jd_std_dc_chr_bits[16] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
static const uint8_t jd_std_dc_vals[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
static const uint8_t jd_std_ac_lum_bits[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
static const uint8_t jd_std_ac_lum_vals[162] = {
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
    0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
    0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
    0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa };
static const uint8_t jd_std_ac_chr_bits[16] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
static const uint8_t jd_std_ac_chr_vals[162] = {
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,
    0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,
    0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,
    0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa };
typedef struct jd_options {
    const jd_kernels *k;     /* NULL = scalar */
    int dct;                 /* JD_DCT_* */
    int upsample;            /* JD_UP_* */
    int out_fmt;             /* JD_OUT_* */
    uint32_t max_pixels;     /* 0 = default (16M) */
    int header_only;
} jd_options;

typedef struct jd_image {
    uint8_t *pix; int w, h, ch; size_t stride;
    int ncomp, progressive, colorspace, hs[4], vs[4];
    int warnings, scans, restart_interval;
    uint32_t mcus_total, mcus_decoded;   /* of the first scan; decoded = before data ran out */
} jd_image;

typedef struct jd_huff {
    int present;
    uint16_t look[512];      /* (len << 8) | sym for codes up to 9 bits, 0 = longer */
    int32_t maxcode[18], valoff[18];
    uint8_t val[256]; int nval; int maxsym;
} jd_huff;

typedef struct jd_comp {
    int id, h, v, tq;
    int bw, bh;              /* allocated block grid (MCU padded) */
    int dsw, dsh, cbw, cbh;  /* downsampled samples and real blocks */
    int16_t *coef;
    uint16_t q[64]; float fq[64]; int q_latched;
    int32_t dcpred;
    int dct, act;            /* table selectors of the current scan */
    uint8_t *plane; int pstride;
} jd_comp;

typedef struct jd_bits { const uint8_t *p, *end; uint64_t acc; int nb; int marker; int eof; int short_data; } jd_bits;

typedef struct jd_dec {
    const uint8_t *d; size_t n;
    const jd_options *opt; jd_image *img;
    uint16_t qt[4][64]; int qtp[4];
    jd_huff ht[2][4];
    int have_sof, progressive, W, H, nc, hmax, vmax, mcux, mcuy;
    jd_comp c[4];
    int ri, jfif, adobe, adobe_tr;
    int ns, sc[4], Ss, Se, Ah, Al;
    jd_bits br; uint32_t eobrun; int rtg, nextrst;
    int nscans, warn;
} jd_dec;

static int jd_build_huff(jd_huff *h, const uint8_t *bits, const uint8_t *vals, int nval) {
    int code = 0, p = 0, huffcode[257], huffsize[257];
    for (int l = 1; l <= 16; l++) {
        for (int i = 0; i < bits[l - 1]; i++) { huffcode[p] = code++; huffsize[p] = l; p++; }
        if (code >= (1 << l)) return -1;         /* jdhuff.c: no code may be all ones / over-subscribed */
        code <<= 1;
    }
    (void)huffsize;
    memset(h, 0, sizeof *h);
    p = 0;
    for (int l = 1; l <= 16; l++) {
        if (bits[l - 1]) {
            h->valoff[l] = p - huffcode[p];
            p += bits[l - 1];
            h->maxcode[l] = huffcode[p - 1];
        } else h->maxcode[l] = -1;
    }
    h->maxcode[17] = 0xFFFFF;
    memcpy(h->val, vals, (size_t)nval);
    h->nval = nval; h->maxsym = 0;
    for (int i = 0; i < nval; i++) if (vals[i] > h->maxsym) h->maxsym = vals[i];
    p = 0;
    for (int l = 1; l <= 9; l++)
        for (int i = 0; i < bits[l - 1]; i++, p++) {
            int c = huffcode[p] << (9 - l);
            for (int j = 0; j < (1 << (9 - l)); j++) h->look[c + j] = (uint16_t)((l << 8) | vals[p]);
        }
    h->present = 1;
    return 0;
}

static void jd_fill(jd_bits *b) {
    while (b->nb <= 56) {
        if (b->marker) return;
        if (b->p >= b->end) { b->marker = 0xD9; b->eof = 1; return; }
        uint8_t c = *b->p++;
        if (c == 0xFF) {
            while (b->p < b->end && *b->p == 0xFF) b->p++;
            if (b->p >= b->end) { b->marker = 0xD9; b->eof = 1; return; }
            uint8_t m = *b->p++;
            if (m != 0) { b->marker = m; return; }
        }
        b->acc = (b->acc << 8) | c;
        b->nb += 8;
    }
}
JD_INLINE uint32_t jd_peek(jd_bits *b, int n) {
    if (b->nb < n) jd_fill(b);
    uint32_t mask = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1);
    if (b->nb >= n) return (uint32_t)(b->acc >> (b->nb - n)) & mask;
    return (uint32_t)(b->acc << (n - b->nb)) & mask;
}
JD_INLINE void jd_skip(jd_bits *b, int n) {
    if (n > b->nb) { b->short_data = 1; b->nb = 0; } else b->nb -= n;
}
JD_INLINE uint32_t jd_getbits(jd_bits *b, int n) { uint32_t v = jd_peek(b, n); jd_skip(b, n); return v; }
JD_INLINE int32_t jd_extend(uint32_t v, int s) { return v < (1u << (s - 1)) ? (int32_t)v - (1 << s) + 1 : (int32_t)v; }

static int jd_huff_dec(jd_dec *d, const jd_huff *h) {
    jd_bits *b = &d->br;
    uint32_t v = jd_peek(b, 17);
    uint16_t e = h->look[v >> 8];
    if (e) { jd_skip(b, e >> 8); return e & 0xFF; }
    for (int l = 10; l <= 16; l++) {
        int32_t code = (int32_t)(v >> (17 - l));
        if (code <= h->maxcode[l]) { jd_skip(b, l); return h->val[(code + h->valoff[l]) & 0xFF]; }
    }
    jd_skip(b, 17);
    d->warn |= JD_W_CORRUPT;
    return 0;
}

/* libjpeg next_marker(): skip garbage, swallow FF fill, ignore FF00. EOF -> fake EOI. */
static int jd_next_marker(jd_dec *d, const uint8_t **pp) {
    const uint8_t *p = *pp, *e = d->d + d->n;
    for (;;) {
        while (p < e && *p != 0xFF) { p++; d->warn |= JD_W_EXTRANEOUS; }
        if (p >= e) { *pp = p; d->warn |= JD_W_EOF; return -1; }
        while (p < e && *p == 0xFF) p++;
        if (p >= e) { *pp = p; d->warn |= JD_W_EOF; return -1; }
        int c = *p++;
        if (c != 0) { *pp = p; return c; }
        d->warn |= JD_W_EXTRANEOUS;
    }
}

/* jpeg_resync_to_restart + process_restart */
static void jd_restart(jd_dec *d) {
    jd_bits *b = &d->br;
    b->nb = 0;
    if (!b->marker) {
        int m = jd_next_marker(d, &b->p);
        if (m < 0) { b->marker = 0xD9; b->eof = 1; } else b->marker = m;
    }
    int desired = d->nextrst;
    if (b->marker == 0xD0 + desired) b->marker = 0;
    else {
        d->warn |= JD_W_RESTART;
        for (;;) {
            int m = b->marker, action;
            if (m < 0xC0) action = 2;
            else if (m < 0xD0 || m > 0xD7) action = 3;
            else if (m == 0xD0 + ((desired + 1) & 7) || m == 0xD0 + ((desired + 2) & 7)) action = 3;
            else if (m == 0xD0 + ((desired - 1) & 7) || m == 0xD0 + ((desired - 2) & 7)) action = 2;
            else action = 1;
            if (action == 1) { b->marker = 0; break; }
            if (action == 3) break;
            if (b->eof) break;
            int n = jd_next_marker(d, &b->p);
            if (n < 0) { b->marker = 0xD9; b->eof = 1; break; }
            b->marker = n;
        }
    }
    d->nextrst = (desired + 1) & 7;
    for (int i = 0; i < d->nc; i++) d->c[i].dcpred = 0;
    d->eobrun = 0;
    d->rtg = d->ri;
    if (b->marker == 0) b->short_data = 0;
}

static void jd_block_seq(jd_dec *d, jd_comp *c, int16_t *blk) {
    jd_bits *b = &d->br;
    int s = jd_huff_dec(d, &d->ht[0][c->dct]);
    int32_t diff = 0;
    if (s) diff = jd_extend(jd_getbits(b, s), s);
    c->dcpred = (int32_t)((uint32_t)c->dcpred + (uint32_t)diff);
    blk[0] = (int16_t)c->dcpred;
    const jd_huff *ac = &d->ht[1][c->act];
    for (int k = 1; k < 64; k++) {
        int rs = jd_huff_dec(d, ac), r = rs >> 4;
        s = rs & 15;
        if (s) {
            k += r;
            if (k > 63) d->warn |= JD_W_CORRUPT;
            blk[jd_zz[k]] = (int16_t)jd_extend(jd_getbits(b, s), s);
        } else {
            if (r != 15) break;
            k += 15;
        }
    }
}
static void jd_block_dc_first(jd_dec *d, jd_comp *c, int16_t *blk) {
    int s = jd_huff_dec(d, &d->ht[0][c->dct]);
    int32_t diff = 0;
    if (s) diff = jd_extend(jd_getbits(&d->br, s), s);
    c->dcpred = (int32_t)((uint32_t)c->dcpred + (uint32_t)diff);
    blk[0] = (int16_t)((uint32_t)c->dcpred << d->Al);
}
static void jd_block_dc_refine(jd_dec *d, int16_t *blk) {
    if (jd_getbits(&d->br, 1)) blk[0] = (int16_t)(blk[0] | (1 << d->Al));
}
static void jd_block_ac_first(jd_dec *d, jd_comp *c, int16_t *blk) {
    if (d->eobrun > 0) { d->eobrun--; return; }
    const jd_huff *ac = &d->ht[1][c->act];
    for (int k = d->Ss; k <= d->Se; k++) {
        int rs = jd_huff_dec(d, ac), r = rs >> 4, s = rs & 15;
        if (s) {
            k += r;
            if (k > 63) d->warn |= JD_W_CORRUPT;
            blk[jd_zz[k]] = (int16_t)((uint32_t)jd_extend(jd_getbits(&d->br, s), s) << d->Al);
        } else {
            if (r == 15) k += 15;
            else {
                d->eobrun = 1u << r;
                if (r) d->eobrun += jd_getbits(&d->br, r);
                d->eobrun--;
                break;
            }
        }
    }
}
static void jd_block_ac_refine(jd_dec *d, jd_comp *c, int16_t *blk) {
    jd_bits *b = &d->br;
    int p1 = 1 << d->Al, m1 = -1 * (1 << d->Al);
    int k = d->Ss;
    if (d->eobrun == 0) {
        const jd_huff *ac = &d->ht[1][c->act];
        for (; k <= d->Se; k++) {
            int rs = jd_huff_dec(d, ac), r = rs >> 4, s = rs & 15;
            if (s) {
                if (s != 1) d->warn |= JD_W_CORRUPT;
                s = jd_getbits(b, 1) ? p1 : m1;
            } else if (r != 15) {
                d->eobrun = 1u << r;
                if (r) d->eobrun += jd_getbits(b, r);
                break;
            }
            do {
                int16_t *t = blk + jd_zz[k];
                if (*t != 0) {
                    if (jd_getbits(b, 1) && (*t & p1) == 0) *t = (int16_t)(*t >= 0 ? *t + p1 : *t + m1);
                } else {
                    if (--r < 0) break;
                }
                k++;
            } while (k <= d->Se);
            if (s) blk[jd_zz[k]] = (int16_t)s;
        }
    }
    if (d->eobrun > 0) {
        for (; k <= d->Se; k++) {
            int16_t *t = blk + jd_zz[k];
            if (*t != 0 && jd_getbits(b, 1) && (*t & p1) == 0) *t = (int16_t)(*t >= 0 ? *t + p1 : *t + m1);
        }
        d->eobrun--;
    }
}

JD_INLINE uint16_t jd_be16(const uint8_t *p) { return (uint16_t)((p[0] << 8) | p[1]); }

/* decode the entropy-coded segment of the current scan starting at *pp; leaves *pp after it and
 * returns the marker that terminated it (or -1 at EOF) */
static int jd_scan(jd_dec *d, const uint8_t **pp) {
    jd_bits *b = &d->br;
    memset(b, 0, sizeof *b);
    b->p = *pp; b->end = d->d + d->n;
    d->eobrun = 0; d->rtg = d->ri; d->nextrst = 0;
    for (int i = 0; i < d->nc; i++) d->c[i].dcpred = 0;
    int mw, mh;
    if (d->ns == 1) { jd_comp *c = &d->c[d->sc[0]]; mw = c->cbw; mh = c->cbh; }
    else { mw = d->mcux; mh = d->mcuy; }
    uint32_t total = (uint32_t)mw * (uint32_t)mh, done = total;
    int mode = !d->progressive ? 0 : (d->Ss == 0 ? (d->Ah == 0 ? 1 : 2) : (d->Ah == 0 ? 3 : 4));
    for (int my = 0; my < mh; my++)
        for (int mx = 0; mx < mw; mx++) {
            if (d->ri) { if (d->rtg == 0) jd_restart(d); d->rtg--; }
            if (b->short_data) { if (done == total) done = (uint32_t)my * (uint32_t)mw + (uint32_t)mx; continue; }
            for (int si = 0; si < d->ns; si++) {
                jd_comp *c = &d->c[d->sc[si]];
                int bh = d->ns == 1 ? 1 : c->v, bwn = d->ns == 1 ? 1 : c->h;
                for (int by = 0; by < bh; by++)
                    for (int bx = 0; bx < bwn; bx++) {
                        int gx = d->ns == 1 ? mx : mx * c->h + bx, gy = d->ns == 1 ? my : my * c->v + by;
                        int16_t *blk = c->coef + ((size_t)gy * (size_t)c->bw + (size_t)gx) * 64;
                        switch (mode) {
                        case 0: jd_block_seq(d, c, blk); break;
                        case 1: jd_block_dc_first(d, c, blk); break;
                        case 2: jd_block_dc_refine(d, blk); break;
                        case 3: jd_block_ac_first(d, c, blk); break;
                        default: jd_block_ac_refine(d, c, blk); break;
                        }
                    }
            }
        }
    if (b->short_data) d->warn |= JD_W_TRUNCATED;
    if (d->nscans == 0) { d->img->mcus_total = total; d->img->mcus_decoded = done; }
    *pp = b->p;
    if (b->eof) { d->warn |= JD_W_EOF; return -1; }
    return b->marker;   /* 0 = none seen yet: caller scans for the next marker */
}

static int jd_parse_sof(jd_dec *d, const uint8_t *s, int len, int prog) {
    if (d->have_sof) return JD_ERR_BAD_SOF;
    if (len < 6) return JD_ERR_BAD_SOF;
    int P = s[0], H = jd_be16(s + 1), W = jd_be16(s + 3), nc = s[5];
    if (P != 8) return JD_ERR_UNSUPPORTED;
    if (H == 0) return JD_ERR_UNSUPPORTED;            /* DNL */
    if (W == 0 || nc == 0) return JD_ERR_BAD_SOF;
    if (len != 6 + 3 * nc) return JD_ERR_BAD_SOF;
    if (nc != 1 && nc != 3) return JD_ERR_UNSUPPORTED;
    uint32_t maxpix = d->opt->max_pixels ? d->opt->max_pixels : (16u << 20);
    if ((uint64_t)W * (uint64_t)H > maxpix) return JD_ERR_TOO_BIG;
    d->W = W; d->H = H; d->nc = nc; d->progressive = prog; d->hmax = d->vmax = 1;
    for (int i = 0; i < nc; i++) {
        jd_comp *c = &d->c[i];
        c->id = s[6 + 3 * i]; c->h = s[7 + 3 * i] >> 4; c->v = s[7 + 3 * i] & 15; c->tq = s[8 + 3 * i];
        if (c->h < 1 || c->h > 4 || c->v < 1 || c->v > 4 || c->tq > 3) return JD_ERR_BAD_SOF;
        for (int j = 0; j < i; j++) if (d->c[j].id == c->id) return JD_ERR_BAD_SOF;
        if (c->h > d->hmax) d->hmax = c->h;
        if (c->v > d->vmax) d->vmax = c->v;
    }
    for (int i = 0; i < nc; i++)   /* only integral upsampling ratios */
        if (d->hmax % d->c[i].h || d->vmax % d->c[i].v) return JD_ERR_UNSUPPORTED;
    d->mcux = (W + 8 * d->hmax - 1) / (8 * d->hmax);
    d->mcuy = (H + 8 * d->vmax - 1) / (8 * d->vmax);
    for (int i = 0; i < nc; i++) {
        jd_comp *c = &d->c[i];
        c->bw = d->mcux * c->h; c->bh = d->mcuy * c->v;
        c->dsw = (int)(((int64_t)W * c->h + d->hmax - 1) / d->hmax);
        c->dsh = (int)(((int64_t)H * c->v + d->vmax - 1) / d->vmax);
        c->cbw = (c->dsw + 7) / 8; c->cbh = (c->dsh + 7) / 8;
    }
    d->have_sof = 1;
    return 0;
}
static int jd_alloc_coefs(jd_dec *d) {
    for (int i = 0; i < d->nc; i++) {
        jd_comp *c = &d->c[i];
        size_t nb = (size_t)c->bw * (size_t)c->bh;
        c->coef = (int16_t *)calloc(nb * 64, sizeof(int16_t));
        if (!c->coef) return JD_ERR_NOMEM;
    }
    return 0;
}

static int jd_parse_sos(jd_dec *d, const uint8_t *s, int len) {
    if (!d->have_sof) return JD_ERR_BAD_SOS;
    if (len < 1) return JD_ERR_BAD_SOS;
    int ns = s[0];
    if (ns < 1 || ns > 4 || len != 4 + 2 * ns) return JD_ERR_BAD_SOS;
    int blocks = 0;
    for (int i = 0; i < ns; i++) {
        int id = s[1 + 2 * i], t = s[2 + 2 * i], ci = -1;
        for (int j = 0; j < d->nc; j++) if (d->c[j].id == id) ci = j;
        if (ci < 0) return JD_ERR_BAD_SOS;
        for (int j = 0; j < i; j++) if (d->sc[j] == ci) return JD_ERR_BAD_SOS;
        d->sc[i] = ci; d->c[ci].dct = t >> 4; d->c[ci].act = t & 15;
        if (d->c[ci].dct > 3 || d->c[ci].act > 3) return JD_ERR_BAD_SOS;
        blocks += d->c[ci].h * d->c[ci].v;
    }
    d->ns = ns;
    if (ns > 1 && blocks > 10) return JD_ERR_BAD_SOS;
    d->Ss = s[1 + 2 * ns]; d->Se = s[2 + 2 * ns]; d->Ah = s[3 + 2 * ns] >> 4; d->Al = s[3 + 2 * ns] & 15;
    if (d->progressive) {
        int bad = 0;
        if (d->Ss == 0) { if (d->Se != 0) bad = 1; }
        else { if (d->Se < d->Ss || d->Se > 63 || ns != 1) bad = 1; }
        if (d->Ah != 0 && d->Al != d->Ah - 1) bad = 1;
        if (d->Al > 13) bad = 1;
        if (bad) return JD_ERR_BAD_SOS;
    } else { d->Ss = 0; d->Se = 63; d->Ah = d->Al = 0; }
    /* tables needed by this scan */
    for (int i = 0; i < ns; i++) {
        jd_comp *c = &d->c[d->sc[i]];
        int need_dc = d->Ss == 0 && d->Ah == 0, need_ac = d->Se > 0;
        if (need_dc) {
            const jd_huff *h = &d->ht[0][c->dct];
            if (!h->present || h->maxsym > 15) return JD_ERR_BAD_DHT;
        }
        if (need_ac && !d->ht[1][c->act].present) return JD_ERR_BAD_DHT;
        if (!c->q_latched) {
            if (!d->qtp[c->tq]) return JD_ERR_BAD_DQT;
            memcpy(c->q, d->qt[c->tq], sizeof c->q);
            jd_make_fq(c->q, c->fq);
            c->q_latched = 1;
        }
    }
    return 0;
}

static void jd_free(jd_image *img) { if (img) { free(img->pix); img->pix = NULL; } }

static void jd_box_row(const uint8_t *in, int rh, int w, uint8_t *out) {
    for (int x = 0; x < w; x++) out[x] = in[x / rh];
}

static int jd_output(jd_dec *d) {
    const jd_kernels *k = d->opt->k ? d->opt->k : &jd_kernels_scalar;
    jd_image *img = d->img;
    int W = d->W, H = d->H, nc = d->nc, ret = JD_OK;
    int cs = nc == 1 ? JD_CS_GRAY : JD_CS_YCC;
    if (nc == 3) {
        if (d->jfif) cs = JD_CS_YCC;
        else if (d->adobe) cs = d->adobe_tr == 0 ? JD_CS_RGB : JD_CS_YCC;
        else if (d->c[0].id == 'R' && d->c[1].id == 'G' && d->c[2].id == 'B') cs = JD_CS_RGB;
    }
    img->colorspace = cs;
    int ch = d->opt->out_fmt == JD_OUT_RGBA ? 4 : d->opt->out_fmt == JD_OUT_RGB ? 3 : (nc == 1 ? 1 : 3);
    img->w = W; img->h = H; img->ch = ch; img->stride = (size_t)W * (size_t)ch;
    img->pix = (uint8_t *)malloc(img->stride * (size_t)H);
    uint8_t *rowbuf[4] = { 0 }, *planes = NULL;
    if (!img->pix) return JD_ERR_NOMEM;
    /* IDCT every real block into component planes */
    size_t tot = 0;
    for (int i = 0; i < nc; i++) { d->c[i].pstride = d->c[i].cbw * 8; tot += (size_t)d->c[i].pstride * (size_t)(d->c[i].cbh * 8); }
    planes = (uint8_t *)malloc(tot ? tot : 1);
    if (!planes) { ret = JD_ERR_NOMEM; goto out; }
    tot = 0;
    for (int i = 0; i < nc; i++) {
        jd_comp *c = &d->c[i];
        c->plane = planes + tot; tot += (size_t)c->pstride * (size_t)(c->cbh * 8);
        if (!c->q_latched) { memset(c->q, 0, sizeof c->q); jd_make_fq(c->q, c->fq); }
        for (int by = 0; by < c->cbh; by++)
            for (int bx = 0; bx < c->cbw; bx++) {
                const int16_t *blk = c->coef + ((size_t)by * (size_t)c->bw + (size_t)bx) * 64;
                uint8_t *o = c->plane + (size_t)by * 8 * (size_t)c->pstride + (size_t)bx * 8;
                if (d->opt->dct == JD_DCT_FLOAT) k->idct_float(blk, c->fq, o, c->pstride);
                else k->idct_islow(blk, c->q, o, c->pstride);
            }
        rowbuf[i] = (uint8_t *)malloc((size_t)W * 4 + 64 + (size_t)c->dsw * 4);
        if (!rowbuf[i]) { ret = JD_ERR_NOMEM; goto out; }
    }
    for (int y = 0; y < H; y++) {
        const uint8_t *row[4];
        for (int i = 0; i < nc; i++) {
            jd_comp *c = &d->c[i];
            int rh = d->hmax / c->h, rv = d->vmax / c->v;
            if (rh == 1 && rv == 1) { row[i] = c->plane + (size_t)y * c->pstride; continue; }
            int iy = y / rv;
            const uint8_t *in0 = c->plane + (size_t)iy * c->pstride;
            int fancy = d->opt->upsample == JD_UP_FANCY;
            if (fancy && rv == 2 && (rh == 2 ? c->dsw > 2 : rh == 1)) {
                int lower = y & 1, ny = lower ? (iy + 1 < c->dsh ? iy + 1 : c->dsh - 1) : (iy > 0 ? iy - 1 : 0);
                const uint8_t *in1 = c->plane + (size_t)ny * c->pstride;
                if (rh == 2) k->h2v2_fancy(in0, in1, c->dsw, rowbuf[i]);
                else k->h1v2_fancy(in0, in1, c->dsw, rowbuf[i], lower);
            } else if (fancy && rv == 1 && rh == 2 && c->dsw > 2) {
                k->h2v1_fancy(in0, c->dsw, rowbuf[i]);
            } else {
                jd_box_row(in0, rh, W, rowbuf[i]);
            }
            row[i] = rowbuf[i];
        }
        uint8_t *o = img->pix + (size_t)y * img->stride;
        if (nc == 1) {
            if (ch == 1) memcpy(o, row[0], (size_t)W);
            else for (int x = 0; x < W; x++) { o[ch * x] = o[ch * x + 1] = o[ch * x + 2] = row[0][x]; if (ch == 4) o[4 * x + 3] = 255; }
        } else if (cs == JD_CS_RGB) {
            for (int x = 0; x < W; x++) { o[ch * x] = row[0][x]; o[ch * x + 1] = row[1][x]; o[ch * x + 2] = row[2][x]; if (ch == 4) o[4 * x + 3] = 255; }
        } else if (ch == 4) k->ycc_rgba(row[0], row[1], row[2], o, W);
        else k->ycc_rgb(row[0], row[1], row[2], o, W);
    }
out:
    for (int i = 0; i < 4; i++) free(rowbuf[i]);
    free(planes);
    if (ret) jd_free(img);
    return ret;
}

static int jd_decode(const uint8_t *data, size_t len, const jd_options *opt, jd_image *img) {
    static const jd_options defopt = { 0 };
    if (!img) return JD_ERR_ARG;
    memset(img, 0, sizeof *img);
    if (!data) return JD_ERR_ARG;
    if (!opt) opt = &defopt;
    jd_dec *d = (jd_dec *)calloc(1, sizeof *d);
    if (!d) return JD_ERR_NOMEM;
    d->d = data; d->n = len; d->opt = opt; d->img = img;
    jd_build_huff(&d->ht[0][0], jd_std_dc_lum_bits, jd_std_dc_vals, 12);
    jd_build_huff(&d->ht[0][1], jd_std_dc_chr_bits, jd_std_dc_vals, 12);
    jd_build_huff(&d->ht[1][0], jd_std_ac_lum_bits, jd_std_ac_lum_vals, 162);
    jd_build_huff(&d->ht[1][1], jd_std_ac_chr_bits, jd_std_ac_chr_vals, 162);
    int ret = JD_OK, pending = 0;
    const uint8_t *p = data + 2, *e = data + len;
    if (len < 2 || data[0] != 0xFF || data[1] != 0xD8) { ret = JD_ERR_NOT_JPEG; goto done; }
    for (;;) {
        int m;
        if (pending) { m = pending; pending = 0; }
        else {
            m = jd_next_marker(d, &p);
            if (m < 0) { if (!d->nscans) { ret = JD_ERR_TRUNC_HEADER; goto done; } break; }
        }
        if (m == 0xD9) { if (!d->nscans) { ret = d->have_sof ? JD_ERR_NO_SCAN : JD_ERR_TRUNC_HEADER; goto done; } break; }
        if ((m >= 0xD0 && m <= 0xD7) || m == 0x01) continue;           /* stray RSTn / TEM: ignored */
        if (m == 0xD8) { ret = JD_ERR_BAD_MARKER; goto done; }           /* duplicate SOI */
        if (m < 0xC0) { ret = JD_ERR_BAD_MARKER; goto done; }
        if (e - p < 2) { if (d->nscans) { d->warn |= JD_W_PARTIAL_HDR; break; } ret = JD_ERR_TRUNC_HEADER; goto done; }
        int L = jd_be16(p);
        if (L < 2) { ret = JD_ERR_BAD_MARKER; goto done; }
        if ((size_t)(e - p) < (size_t)L) { if (d->nscans) { d->warn |= JD_W_PARTIAL_HDR; break; } ret = JD_ERR_TRUNC_HEADER; goto done; }
        const uint8_t *s = p + 2;
        int sl = L - 2;
        p += L;
        int r = 0;
        switch (m) {
        case 0xC0: case 0xC1: case 0xC2:
            r = jd_parse_sof(d, s, sl, m == 0xC2);
            if (!r) {
                img->w = d->W; img->h = d->H; img->ncomp = d->nc; img->progressive = d->progressive;
                for (int i = 0; i < d->nc; i++) { img->hs[i] = d->c[i].h; img->vs[i] = d->c[i].v; }
                if (opt->header_only) goto done;
                r = jd_alloc_coefs(d);
            }
            break;
        case 0xC3: case 0xC5: case 0xC6: case 0xC7: case 0xC9: case 0xCA: case 0xCB: case 0xCD: case 0xCE: case 0xCF:
        case 0xCC: case 0xDC: case 0xDE: case 0xDF:
            r = JD_ERR_UNSUPPORTED; break;   /* lossless / hierarchical / arithmetic / DAC / DNL / DHP / EXP */
        case 0xC4: {
            int q = 0;
            while (q < sl) {
                if (sl - q < 17) { r = JD_ERR_BAD_DHT; break; }
                int tc = s[q] >> 4, th = s[q] & 15, n = 0;
                for (int i = 0; i < 16; i++) n += s[q + 1 + i];
                if (tc > 1 || th > 3 || n > 256 || sl - q - 17 < n) { r = JD_ERR_BAD_DHT; break; }
                if (jd_build_huff(&d->ht[tc][th], s + q + 1, s + q + 17, n)) { r = JD_ERR_BAD_DHT; break; }
                q += 17 + n;
            }
            break;
        }
        case 0xDB: {
            int q = 0;
            while (q < sl) {
                int pq = s[q] >> 4, tq = s[q] & 15, n = pq ? 129 : 65;
                if (pq > 1 || tq > 3 || sl - q < n) { r = JD_ERR_BAD_DQT; break; }
                for (int i = 0; i < 64; i++) d->qt[tq][jd_zz[i]] = pq ? jd_be16(s + q + 1 + 2 * i) : s[q + 1 + i];
                d->qtp[tq] = 1;
                q += n;
            }
            break;
        }
        case 0xDD:
            if (sl != 2) r = JD_ERR_BAD_DRI; else d->ri = jd_be16(s);
            break;
        case 0xDA: {
            r = jd_parse_sos(d, s, sl);
            if (r) break;
            img->restart_interval = d->ri;
            int t = jd_scan(d, &p);
            d->nscans++;
            if (t < 0) goto finish;                  /* EOF inside / right after entropy data */
            pending = t;                             /* 0: find the next marker normally */
            break;
        }
        case 0xE0:
            if (sl >= 5 && !memcmp(s, "JFIF\0", 5)) d->jfif = 1;
            break;
        case 0xEE:
            if (sl >= 12 && !memcmp(s, "Adobe", 5)) { d->adobe = 1; d->adobe_tr = s[11]; }
            break;
        default: break;   /* APPn, COM, JPGn: skipped */
        }
        if (r) { ret = r; goto done; }   /* malformed segment: fatal even after a scan (as libjpeg) */
    }
finish:
    img->scans = d->nscans;
    ret = jd_output(d);
done:
    img->warnings = d->warn;
    for (int i = 0; i < 4; i++) free(d->c[i].coef);
    free(d);
    if (ret) jd_free(img);
    return ret;
}

/* FNV-1a 64 over packed pixels (the corpus generator hashes djpeg's PPM/PGM payload the same way) */
static uint64_t jd_fnv64(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
#endif
