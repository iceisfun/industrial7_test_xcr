/* simd_bench_compute.h - arithmetic kernels for simd_bench (array add, integer multiply, FP throughput /
 * latency, dot product, matrix multiply, popcount, byte sum, conversions, gather).
 * Every SIMD kernel has a scalar or exact integer reference; inputs are chosen so that every FP result
 * is exact (small integers, sums < 2^24) and therefore independent of summation order and of x87 vs SSE
 * evaluation of the scalar C reference on x86. Wide-ISA functions carry an ISA hint in their name
 * (avx/fma/512/...) and T_TARGET; they are called only after cpu_has(). */
#ifndef SIMD_BENCH_COMPUTE_H
#define SIMD_BENCH_COMPUTE_H
#include "simd_bench_util.h"

/* ============================================================================ array add c[i] = a[i] + b[i] */
typedef void (*add_fn)(void *c, const void *a, const void *b, size_t n);   /* n multiple of 64 */

static BM_NOVEC void add_f32_scalar(void *c_, const void *a_, const void *b_, size_t n) {
    float *c = (float *)c_; const float *a = (const float *)a_, *b = (const float *)b_;
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}
static void add_f32_sse(void *c_, const void *a_, const void *b_, size_t n) {
    float *c = (float *)c_; const float *a = (const float *)a_, *b = (const float *)b_;
    for (size_t i = 0; i < n; i += 8) {
        _mm_store_ps(c + i, _mm_add_ps(_mm_load_ps(a + i), _mm_load_ps(b + i)));
        _mm_store_ps(c + i + 4, _mm_add_ps(_mm_load_ps(a + i + 4), _mm_load_ps(b + i + 4)));
    }
}
static T_TARGET("avx") void add_f32_avx(void *c_, const void *a_, const void *b_, size_t n) {
    float *c = (float *)c_; const float *a = (const float *)a_, *b = (const float *)b_;
    for (size_t i = 0; i < n; i += 16) {
        _mm256_store_ps(c + i, _mm256_add_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
        _mm256_store_ps(c + i + 8, _mm256_add_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
    }
    _mm256_zeroupper();
}
static T_TARGET("avx512f") void add_f32_avx512(void *c_, const void *a_, const void *b_, size_t n) {
    float *c = (float *)c_; const float *a = (const float *)a_, *b = (const float *)b_;
    for (size_t i = 0; i < n; i += 32) {
        _mm512_store_ps(c + i, _mm512_add_ps(_mm512_load_ps(a + i), _mm512_load_ps(b + i)));
        _mm512_store_ps(c + i + 16, _mm512_add_ps(_mm512_load_ps(a + i + 16), _mm512_load_ps(b + i + 16)));
    }
    _mm256_zeroupper();
}
static BM_NOVEC void add_i32_scalar(void *c_, const void *a_, const void *b_, size_t n) {
    uint32_t *c = (uint32_t *)c_; const uint32_t *a = (const uint32_t *)a_, *b = (const uint32_t *)b_;
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}
static void add_i32_sse2(void *c_, const void *a_, const void *b_, size_t n) {
    __m128i *c = (__m128i *)c_; const __m128i *a = (const __m128i *)a_, *b = (const __m128i *)b_;
    for (size_t i = 0; i < n / 4; i += 2) {
        _mm_store_si128(c + i, _mm_add_epi32(_mm_load_si128(a + i), _mm_load_si128(b + i)));
        _mm_store_si128(c + i + 1, _mm_add_epi32(_mm_load_si128(a + i + 1), _mm_load_si128(b + i + 1)));
    }
}
static T_TARGET("avx2") void add_i32_avx2(void *c_, const void *a_, const void *b_, size_t n) {
    __m256i *c = (__m256i *)c_; const __m256i *a = (const __m256i *)a_, *b = (const __m256i *)b_;
    for (size_t i = 0; i < n / 8; i += 2) {
        _mm256_store_si256(c + i, _mm256_add_epi32(_mm256_load_si256(a + i), _mm256_load_si256(b + i)));
        _mm256_store_si256(c + i + 1, _mm256_add_epi32(_mm256_load_si256(a + i + 1), _mm256_load_si256(b + i + 1)));
    }
    _mm256_zeroupper();
}
static T_TARGET("avx512f") void add_i32_avx512(void *c_, const void *a_, const void *b_, size_t n) {
    __m512i *c = (__m512i *)c_; const __m512i *a = (const __m512i *)a_, *b = (const __m512i *)b_;
    for (size_t i = 0; i < n / 16; i += 2) {
        _mm512_store_si512(c + i, _mm512_add_epi32(_mm512_load_si512(a + i), _mm512_load_si512(b + i)));
        _mm512_store_si512(c + i + 1, _mm512_add_epi32(_mm512_load_si512(a + i + 1), _mm512_load_si512(b + i + 1)));
    }
    _mm256_zeroupper();
}

/* ============================================================================ integer multiply chains
 * Every lane runs x <- x * C (mod 2^w) K times; the result is seed * C^K, checked exactly. */
#define IM_K 1024
typedef struct im_ctx {
    T_ALIGN(64) uint8_t out[1024];
    uint64_t seed[16];
    uint32_t c32; uint64_t c64; uint16_t c16;
    int K;
} im_ctx;

static BM_NOVEC void imul32_tput_scalar(void *p) {
    im_ctx *x = (im_ctx *)p; uint32_t c = x->c32; BM_HIDE(c);
    uint32_t a0 = (uint32_t)x->seed[0], a1 = (uint32_t)x->seed[1], a2 = (uint32_t)x->seed[2], a3 = (uint32_t)x->seed[3];
    uint32_t a4 = (uint32_t)x->seed[4], a5 = (uint32_t)x->seed[5], a6 = (uint32_t)x->seed[6], a7 = (uint32_t)x->seed[7];
    for (int k = 0; k < x->K; k++) { a0 *= c; a1 *= c; a2 *= c; a3 *= c; a4 *= c; a5 *= c; a6 *= c; a7 *= c; }
    uint32_t *o = (uint32_t *)x->out; o[0] = a0; o[1] = a1; o[2] = a2; o[3] = a3; o[4] = a4; o[5] = a5; o[6] = a6; o[7] = a7;
}
static BM_NOVEC void imul32_lat_scalar(void *p) {
    im_ctx *x = (im_ctx *)p; uint32_t c = x->c32; BM_HIDE(c);
    uint32_t a = (uint32_t)x->seed[0];
    for (int k = 0; k < x->K; k++) { a *= c; BM_HIDE(a); }
    ((uint32_t *)x->out)[0] = a;
}
static BM_NOVEC void imul64_tput_scalar(void *p) {
    im_ctx *x = (im_ctx *)p; uint64_t c = x->c64;
#ifdef T_X64
    BM_HIDE(c);
#else
    BM_BARRIER();
#endif
    uint64_t a0 = x->seed[0], a1 = x->seed[1], a2 = x->seed[2], a3 = x->seed[3];
    uint64_t a4 = x->seed[4], a5 = x->seed[5], a6 = x->seed[6], a7 = x->seed[7];
    for (int k = 0; k < x->K; k++) { a0 *= c; a1 *= c; a2 *= c; a3 *= c; a4 *= c; a5 *= c; a6 *= c; a7 *= c; }
    uint64_t *o = (uint64_t *)x->out; o[0] = a0; o[1] = a1; o[2] = a2; o[3] = a3; o[4] = a4; o[5] = a5; o[6] = a6; o[7] = a7;
}
static BM_NOVEC void imul64_lat_scalar(void *p) {
    im_ctx *x = (im_ctx *)p; uint64_t c = x->c64;
    uint64_t a = x->seed[0];
#ifdef T_X64
    BM_HIDE(c);
    for (int k = 0; k < x->K; k++) { a *= c; BM_HIDE(a); }
#else
    for (int k = 0; k < x->K; k++) { a *= c; BM_BARRIER(); }
#endif
    ((uint64_t *)x->out)[0] = a;
}
/* SSE2 pmuludq: 64-bit lanes, low dword times C -> 64-bit product (4 vectors x 2 lanes) */
static void imul_sse2_pmuludq(void *p) {
    im_ctx *x = (im_ctx *)p;
    __m128i c = _mm_set1_epi32((int)x->c32); BM_HIDEX(c);
    __m128i v0 = _mm_set_epi64x((long long)(uint32_t)x->seed[1], (long long)(uint32_t)x->seed[0]);
    __m128i v1 = _mm_set_epi64x((long long)(uint32_t)x->seed[3], (long long)(uint32_t)x->seed[2]);
    __m128i v2 = _mm_set_epi64x((long long)(uint32_t)x->seed[5], (long long)(uint32_t)x->seed[4]);
    __m128i v3 = _mm_set_epi64x((long long)(uint32_t)x->seed[7], (long long)(uint32_t)x->seed[6]);
    for (int k = 0; k < x->K; k++) { v0 = _mm_mul_epu32(v0, c); v1 = _mm_mul_epu32(v1, c); v2 = _mm_mul_epu32(v2, c); v3 = _mm_mul_epu32(v3, c); }
    __m128i *o = (__m128i *)x->out; _mm_store_si128(o, v0); _mm_store_si128(o + 1, v1); _mm_store_si128(o + 2, v2); _mm_store_si128(o + 3, v3);
}
/* SSE2 pmullw: 16-bit lanes (4 vectors x 8 lanes) */
static void imul_sse2_pmullw(void *p) {
    im_ctx *x = (im_ctx *)p;
    __m128i c = _mm_set1_epi16((short)x->c16); BM_HIDEX(c);
    __m128i v[4];
    for (int j = 0; j < 4; j++) v[j] = _mm_loadu_si128((const __m128i *)&x->seed[j * 2]);
    __m128i v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
    for (int k = 0; k < x->K; k++) { v0 = _mm_mullo_epi16(v0, c); v1 = _mm_mullo_epi16(v1, c); v2 = _mm_mullo_epi16(v2, c); v3 = _mm_mullo_epi16(v3, c); }
    __m128i *o = (__m128i *)x->out; _mm_store_si128(o, v0); _mm_store_si128(o + 1, v1); _mm_store_si128(o + 2, v2); _mm_store_si128(o + 3, v3);
}
/* SSE4.1 pmulld: 32-bit lanes (4 vectors x 4 lanes); seeds as dwords from seed[] bytes */
static T_TARGET("sse4.1") void imul_sse41_pmulld(void *p) {
    im_ctx *x = (im_ctx *)p;
    __m128i c = _mm_set1_epi32((int)x->c32); BM_HIDEX(c);
    const __m128i *s = (const __m128i *)x->seed;
    __m128i v0 = _mm_loadu_si128(s), v1 = _mm_loadu_si128(s + 1), v2 = _mm_loadu_si128(s + 2), v3 = _mm_loadu_si128(s + 3);
    for (int k = 0; k < x->K; k++) { v0 = _mm_mullo_epi32(v0, c); v1 = _mm_mullo_epi32(v1, c); v2 = _mm_mullo_epi32(v2, c); v3 = _mm_mullo_epi32(v3, c); }
    __m128i *o = (__m128i *)x->out; _mm_store_si128(o, v0); _mm_store_si128(o + 1, v1); _mm_store_si128(o + 2, v2); _mm_store_si128(o + 3, v3);
}
/* AVX2 vpmulld ymm: 4 vectors x 8 lanes (seed bytes reused as 32 dwords) */
static T_TARGET("avx2") void imul_avx2_vpmulld(void *p) {
    im_ctx *x = (im_ctx *)p;
    __m256i c = _mm256_set1_epi32((int)x->c32); BM_HIDEX(c);
    const __m256i *s = (const __m256i *)x->seed;
    __m256i v0 = _mm256_loadu_si256(s), v1 = _mm256_loadu_si256(s + 1), v2 = _mm256_loadu_si256(s + 2), v3 = _mm256_loadu_si256(s + 3);
    for (int k = 0; k < x->K; k++) { v0 = _mm256_mullo_epi32(v0, c); v1 = _mm256_mullo_epi32(v1, c); v2 = _mm256_mullo_epi32(v2, c); v3 = _mm256_mullo_epi32(v3, c); }
    __m256i *o = (__m256i *)x->out; _mm256_store_si256(o, v0); _mm256_store_si256(o + 1, v1); _mm256_store_si256(o + 2, v2); _mm256_store_si256(o + 3, v3);
    _mm256_zeroupper();
}
/* AVX-512F vpmulld zmm: 4 vectors x 16 lanes (seed bytes (128) + seed bytes again) */
static T_TARGET("avx512f") void imul_avx512_vpmulld(void *p) {
    im_ctx *x = (im_ctx *)p;
    __m512i c = _mm512_set1_epi32((int)x->c32); BM_HIDEV(c);
    const __m512i *s = (const __m512i *)x->seed;
    __m512i v0 = _mm512_loadu_si512(s), v1 = _mm512_loadu_si512(s + 1), v2 = _mm512_loadu_si512(s), v3 = _mm512_loadu_si512(s + 1);
    for (int k = 0; k < x->K; k++) { v0 = _mm512_mullo_epi32(v0, c); v1 = _mm512_mullo_epi32(v1, c); v2 = _mm512_mullo_epi32(v2, c); v3 = _mm512_mullo_epi32(v3, c); }
    __m512i *o = (__m512i *)x->out; _mm512_store_si512(o, v0); _mm512_store_si512(o + 1, v1); _mm512_store_si512(o + 2, v2); _mm512_store_si512(o + 3, v3);
    _mm256_zeroupper();
}
/* AVX-512DQ vpmullq zmm: 4 vectors x 8 qword lanes */
static T_TARGET("avx512f,avx512dq") void imul_avx512_vpmullq(void *p) {
    im_ctx *x = (im_ctx *)p;
    __m512i c = _mm512_set1_epi64((long long)x->c64); BM_HIDEV(c);
    const __m512i *s = (const __m512i *)x->seed;
    __m512i v0 = _mm512_loadu_si512(s), v1 = _mm512_loadu_si512(s + 1), v2 = _mm512_loadu_si512(s), v3 = _mm512_loadu_si512(s + 1);
    for (int k = 0; k < x->K; k++) { v0 = _mm512_mullo_epi64(v0, c); v1 = _mm512_mullo_epi64(v1, c); v2 = _mm512_mullo_epi64(v2, c); v3 = _mm512_mullo_epi64(v3, c); }
    __m512i *o = (__m512i *)x->out; _mm512_store_si512(o, v0); _mm512_store_si512(o + 1, v1); _mm512_store_si512(o + 2, v2); _mm512_store_si512(o + 3, v3);
    _mm256_zeroupper();
}

/* ============================================================================ FP throughput / latency
 * Accumulator j starts at init[j]; each step is acc = OP(acc, one) with one = 1.0 laundered through asm.
 *   ADD/FMA (acc*1+1): result init[j] + K (exact), MUL/DIV by one, SQRT of 1: result init[j]. */
#define FP_K 1024
enum { FPE_ADD = 0, FPE_SAME = 1 };
typedef struct fp_ctx {
    T_ALIGN(64) uint8_t out[8][64];     /* accumulator j stored at out[j] */
    double init[8];
    int K;
} fp_ctx;

#define CONS_X(v) "+x"(v)
#define CONS_V(v) "+v"(v)
#define FP_TPUT(fname, ATTR, VT, ET, SET1, OPX, STORE, CONS)                                                 \
static ATTR void fname(void *p) {                                                                            \
    fp_ctx *x = (fp_ctx *)p; VT one = SET1((ET)1.0); __asm__ volatile("" : CONS(one));                      \
    VT a0 = SET1((ET)x->init[0]), a1 = SET1((ET)x->init[1]), a2 = SET1((ET)x->init[2]), a3 = SET1((ET)x->init[3]); \
    VT a4 = SET1((ET)x->init[4]), a5 = SET1((ET)x->init[5]), a6 = SET1((ET)x->init[6]), a7 = SET1((ET)x->init[7]); \
    for (int k = 0; k < x->K; k++) {                                                                         \
        a0 = OPX(a0, one); a1 = OPX(a1, one); a2 = OPX(a2, one); a3 = OPX(a3, one);                          \
        a4 = OPX(a4, one); a5 = OPX(a5, one); a6 = OPX(a6, one); a7 = OPX(a7, one);                          \
    }                                                                                                        \
    STORE((ET *)x->out[0], a0); STORE((ET *)x->out[1], a1); STORE((ET *)x->out[2], a2); STORE((ET *)x->out[3], a3); \
    STORE((ET *)x->out[4], a4); STORE((ET *)x->out[5], a5); STORE((ET *)x->out[6], a6); STORE((ET *)x->out[7], a7); \
}
#define FP_LAT(fname, ATTR, VT, ET, SET1, OPX, STORE, CONS)                                                  \
static ATTR void fname(void *p) {                                                                            \
    fp_ctx *x = (fp_ctx *)p; VT one = SET1((ET)1.0); __asm__ volatile("" : CONS(one));                      \
    VT a0 = SET1((ET)x->init[0]);                                                                            \
    for (int k = 0; k < x->K; k++) a0 = OPX(a0, one);                                                        \
    STORE((ET *)x->out[0], a0);                                                                              \
}
/* operation adapters */
#define OP_ADDSS(a, o) _mm_add_ss(a, o)
#define OP_ADDSD(a, o) _mm_add_sd(a, o)
#define OP_MULSD(a, o) _mm_mul_sd(a, o)
#define OP_DIVSD(a, o) _mm_div_sd(a, o)
#define OP_SQRTSD(a, o) _mm_sqrt_sd(o, a)          /* low = sqrt(a.low), high from o */
#define OP_ADDPS(a, o) _mm_add_ps(a, o)
#define OP_MULPS(a, o) _mm_mul_ps(a, o)
#define OP_DIVPS(a, o) _mm_div_ps(a, o)
#define OP_SQRTPS(a, o) _mm_sqrt_ps(a)
#define OP_ADDPD(a, o) _mm_add_pd(a, o)
#define OP_MULPD(a, o) _mm_mul_pd(a, o)
#define OP_VADDPS(a, o) _mm256_add_ps(a, o)
#define OP_VMULPS(a, o) _mm256_mul_ps(a, o)
#define OP_VDIVPS(a, o) _mm256_div_ps(a, o)
#define OP_VADDPD(a, o) _mm256_add_pd(a, o)
#define OP_VMULPD(a, o) _mm256_mul_pd(a, o)
#define OP_FMAPS(a, o)  _mm_fmadd_ps(a, o, o)
#define OP_FMASD(a, o)  _mm_fmadd_sd(a, o, o)
#define OP_VFMAPS(a, o) _mm256_fmadd_ps(a, o, o)
#define OP_VFMAPD(a, o) _mm256_fmadd_pd(a, o, o)
#define OP_ZADDPS(a, o) _mm512_add_ps(a, o)
#define OP_ZMULPD(a, o) _mm512_mul_pd(a, o)
#define OP_ZFMAPS(a, o) _mm512_fmadd_ps(a, o, o)
#define OP_ZFMAPD(a, o) _mm512_fmadd_pd(a, o, o)
#define ST_SS(p, v)  _mm_store_ss(p, v)
#define ST_SD(p, v)  _mm_store_sd(p, v)
#define ST_PS(p, v)  _mm_storeu_ps(p, v)
#define ST_PD(p, v)  _mm_storeu_pd(p, v)
#define ST_YPS(p, v) _mm256_storeu_ps(p, v)
#define ST_YPD(p, v) _mm256_storeu_pd(p, v)
#define ST_ZPS(p, v) _mm512_storeu_ps(p, v)
#define ST_ZPD(p, v) _mm512_storeu_pd(p, v)
#define NOATTR
#define AVX_  T_TARGET("avx")
#define FMA_  T_TARGET("avx,fma")
#define Z512_ T_TARGET("avx512f")

FP_TPUT(fpt_sse_addss,  NOATTR, __m128,  float,  _mm_set1_ps,    OP_ADDSS,  ST_SS,  CONS_X)
FP_TPUT(fpt_sse2_addsd, NOATTR, __m128d, double, _mm_set1_pd,    OP_ADDSD,  ST_SD,  CONS_X)
FP_TPUT(fpt_sse2_mulsd, NOATTR, __m128d, double, _mm_set1_pd,    OP_MULSD,  ST_SD,  CONS_X)
FP_TPUT(fpt_sse_addps,  NOATTR, __m128,  float,  _mm_set1_ps,    OP_ADDPS,  ST_PS,  CONS_X)
FP_TPUT(fpt_sse_mulps,  NOATTR, __m128,  float,  _mm_set1_ps,    OP_MULPS,  ST_PS,  CONS_X)
FP_TPUT(fpt_sse_divps,  NOATTR, __m128,  float,  _mm_set1_ps,    OP_DIVPS,  ST_PS,  CONS_X)
FP_TPUT(fpt_sse_sqrtps, NOATTR, __m128,  float,  _mm_set1_ps,    OP_SQRTPS, ST_PS,  CONS_X)
FP_TPUT(fpt_sse2_addpd, NOATTR, __m128d, double, _mm_set1_pd,    OP_ADDPD,  ST_PD,  CONS_X)
FP_TPUT(fpt_sse2_mulpd, NOATTR, __m128d, double, _mm_set1_pd,    OP_MULPD,  ST_PD,  CONS_X)
FP_TPUT(fpt_avx_addps,  AVX_,   __m256,  float,  _mm256_set1_ps, OP_VADDPS, ST_YPS, CONS_X)
FP_TPUT(fpt_avx_mulps,  AVX_,   __m256,  float,  _mm256_set1_ps, OP_VMULPS, ST_YPS, CONS_X)
FP_TPUT(fpt_avx_divps,  AVX_,   __m256,  float,  _mm256_set1_ps, OP_VDIVPS, ST_YPS, CONS_X)
FP_TPUT(fpt_avx_addpd,  AVX_,   __m256d, double, _mm256_set1_pd, OP_VADDPD, ST_YPD, CONS_X)
FP_TPUT(fpt_avx_mulpd,  AVX_,   __m256d, double, _mm256_set1_pd, OP_VMULPD, ST_YPD, CONS_X)
FP_TPUT(fpt_fma_ps128,  FMA_,   __m128,  float,  _mm_set1_ps,    OP_FMAPS,  ST_PS,  CONS_X)
FP_TPUT(fpt_fma_ps256,  FMA_,   __m256,  float,  _mm256_set1_ps, OP_VFMAPS, ST_YPS, CONS_X)
FP_TPUT(fpt_fma_pd256,  FMA_,   __m256d, double, _mm256_set1_pd, OP_VFMAPD, ST_YPD, CONS_X)
FP_TPUT(fpt_avx512_addps, Z512_, __m512, float,  _mm512_set1_ps, OP_ZADDPS, ST_ZPS, CONS_V)
FP_TPUT(fpt_avx512_mulpd, Z512_, __m512d, double, _mm512_set1_pd, OP_ZMULPD, ST_ZPD, CONS_V)
FP_TPUT(fpt_avx512_fmaps, Z512_, __m512, float,  _mm512_set1_ps, OP_ZFMAPS, ST_ZPS, CONS_V)
FP_TPUT(fpt_avx512_fmapd, Z512_, __m512d, double, _mm512_set1_pd, OP_ZFMAPD, ST_ZPD, CONS_V)

FP_LAT(fpl_sse2_addsd,  NOATTR, __m128d, double, _mm_set1_pd,    OP_ADDSD,  ST_SD,  CONS_X)
FP_LAT(fpl_sse2_mulsd,  NOATTR, __m128d, double, _mm_set1_pd,    OP_MULSD,  ST_SD,  CONS_X)
FP_LAT(fpl_sse2_divsd,  NOATTR, __m128d, double, _mm_set1_pd,    OP_DIVSD,  ST_SD,  CONS_X)
FP_LAT(fpl_sse2_sqrtsd, NOATTR, __m128d, double, _mm_set1_pd,    OP_SQRTSD, ST_SD,  CONS_X)
FP_LAT(fpl_sse_addps,   NOATTR, __m128,  float,  _mm_set1_ps,    OP_ADDPS,  ST_PS,  CONS_X)
FP_LAT(fpl_sse_mulps,   NOATTR, __m128,  float,  _mm_set1_ps,    OP_MULPS,  ST_PS,  CONS_X)
FP_LAT(fpl_avx_addps,   AVX_,   __m256,  float,  _mm256_set1_ps, OP_VADDPS, ST_YPS, CONS_X)
FP_LAT(fpl_fma_sd,      FMA_,   __m128d, double, _mm_set1_pd,    OP_FMASD,  ST_SD,  CONS_X)
FP_LAT(fpl_fma_ps256,   FMA_,   __m256,  float,  _mm256_set1_ps, OP_VFMAPS, ST_YPS, CONS_X)
FP_LAT(fpl_avx512_addps, Z512_, __m512,  float,  _mm512_set1_ps, OP_ZADDPS, ST_ZPS, CONS_V)

/* x87 dependent fadd chain: ST(0) = init + K (exact) */
static T_NOINLINE void fpl_x87_fadd(void *p) {
    fp_ctx *x = (fp_ctx *)p; double r; double init = x->init[0]; long k = x->K;
    __asm__ volatile(
        "fld1\n\t"
        "fldl %[in]\n\t"                 /* st0 = init, st1 = 1 */
        "1:\n\t"
        "fadd %%st(1), %%st\n\t"
        "dec %[k]\n\t"
        "jnz 1b\n\t"
        "fstpl %[r]\n\t"
        "fstp %%st(0)\n\t"
        : [r] "=m"(r), [k] "+r"(k) : [in] "m"(init) : "cc", "st", "st(1)");
    memcpy(x->out[0], &r, 8);
}
/* x87 dependent fmul chain by 1.0: unchanged */
static T_NOINLINE void fpl_x87_fmul(void *p) {
    fp_ctx *x = (fp_ctx *)p; double r; double init = x->init[0]; long k = x->K;
    __asm__ volatile(
        "fld1\n\t" "fldl %[in]\n\t"
        "1:\n\t" "fmul %%st(1), %%st\n\t" "dec %[k]\n\t" "jnz 1b\n\t"
        "fstpl %[r]\n\t" "fstp %%st(0)\n\t"
        : [r] "=m"(r), [k] "+r"(k) : [in] "m"(init) : "cc", "st", "st(1)");
    memcpy(x->out[0], &r, 8);
}
/* MMX paddw throughput: 4 independent chains mm0..mm3 += mm4 (word lanes += 1), ends with EMMS */
static T_NOINLINE void fpt_mmx_paddw(void *p) {
    fp_ctx *x = (fp_ctx *)p; long k = x->K;
    static const uint64_t ones = 0x0001000100010001ull;
    __asm__ volatile(
        "movq %[one], %%mm4\n\t"
        "pxor %%mm0, %%mm0\n\t" "pxor %%mm1, %%mm1\n\t" "pxor %%mm2, %%mm2\n\t" "pxor %%mm3, %%mm3\n\t"
        "1:\n\t"
        "paddw %%mm4, %%mm0\n\t" "paddw %%mm4, %%mm1\n\t" "paddw %%mm4, %%mm2\n\t" "paddw %%mm4, %%mm3\n\t"
        "paddw %%mm4, %%mm0\n\t" "paddw %%mm4, %%mm1\n\t" "paddw %%mm4, %%mm2\n\t" "paddw %%mm4, %%mm3\n\t"
        "dec %[k]\n\t" "jnz 1b\n\t"
        "movq %%mm0, 0(%[o])\n\t" "movq %%mm1, 8(%[o])\n\t" "movq %%mm2, 16(%[o])\n\t" "movq %%mm3, 24(%[o])\n\t"
        "emms\n\t"
        : [k] "+r"(k) : [one] "m"(ones), [o] "r"(x->out[0]) : "cc", "memory", "mm0", "mm1", "mm2", "mm3", "mm4");
}

/* ============================================================================ dot product (exact) */
typedef float (*dot_fn)(const float *a, const float *b, size_t n);   /* n multiple of 64 */
static BM_NOVEC float dot_scalar(const float *a, const float *b, size_t n) {
    float s = 0; for (size_t i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
static float dot_sse(const float *a, const float *b, size_t n) {
    __m128 s0 = _mm_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t i = 0; i < n; i += 16) {
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_load_ps(a + i), _mm_load_ps(b + i)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_load_ps(a + i + 4), _mm_load_ps(b + i + 4)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_load_ps(a + i + 8), _mm_load_ps(b + i + 8)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_load_ps(a + i + 12), _mm_load_ps(b + i + 12)));
    }
    s0 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
    s0 = _mm_add_ps(s0, _mm_movehl_ps(s0, s0));
    s0 = _mm_add_ss(s0, _mm_shuffle_ps(s0, s0, 1));
    return _mm_cvtss_f32(s0);
}
static T_TARGET("avx") float dot_avx(const float *a, const float *b, size_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t i = 0; i < n; i += 32) {
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16)));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24)));
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    float r = _mm_cvtss_f32(h);
    _mm256_zeroupper();
    return r;
}
static T_TARGET("avx2,fma") float dot_fma(const float *a, const float *b, size_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t i = 0; i < n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24), s3);
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    float r = _mm_cvtss_f32(h);
    _mm256_zeroupper();
    return r;
}
static T_TARGET("avx512f") float dot_avx512(const float *a, const float *b, size_t n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t i = 0; i < n; i += 64) {
        s0 = _mm512_fmadd_ps(_mm512_load_ps(a + i), _mm512_load_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(a + i + 16), _mm512_load_ps(b + i + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(a + i + 32), _mm512_load_ps(b + i + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(a + i + 48), _mm512_load_ps(b + i + 48), s3);
    }
    float r = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
    _mm256_zeroupper();
    return r;
}

/* ============================================================================ matrix multiply C = A*B (N x N) */
typedef void (*mm_fn)(float *C, const float *A, const float *B, int N);   /* N multiple of 32 */
static BM_NOVEC void mm_scalar(float *C, const float *A, const float *B, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            float *c = C + i * N; const float *b = B + k * N;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
}
static void mm_sse2(float *C, const float *A, const float *B, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j += 16) {
            __m128 c0 = _mm_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
            for (int k = 0; k < N; k++) {
                __m128 a = _mm_set1_ps(A[i * N + k]);
                const float *b = B + k * N + j;
                c0 = _mm_add_ps(c0, _mm_mul_ps(a, _mm_load_ps(b)));
                c1 = _mm_add_ps(c1, _mm_mul_ps(a, _mm_load_ps(b + 4)));
                c2 = _mm_add_ps(c2, _mm_mul_ps(a, _mm_load_ps(b + 8)));
                c3 = _mm_add_ps(c3, _mm_mul_ps(a, _mm_load_ps(b + 12)));
            }
            float *c = C + i * N + j;
            _mm_store_ps(c, c0); _mm_store_ps(c + 4, c1); _mm_store_ps(c + 8, c2); _mm_store_ps(c + 12, c3);
        }
}
static T_TARGET("avx2,fma") void mm_avx2_fma(float *C, const float *A, const float *B, int N) {
    for (int i = 0; i < N; i += 4)
        for (int j = 0; j < N; j += 16) {
            __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
            for (int k = 0; k < N; k++) {
                const float *b = B + k * N + j;
                __m256 b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8);
                __m256 a0 = _mm256_broadcast_ss(A + (i + 0) * N + k);
                __m256 a1 = _mm256_broadcast_ss(A + (i + 1) * N + k);
                c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
                c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
                __m256 a2 = _mm256_broadcast_ss(A + (i + 2) * N + k);
                __m256 a3 = _mm256_broadcast_ss(A + (i + 3) * N + k);
                c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
                c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
            }
            float *c = C + i * N + j;
            _mm256_store_ps(c, c00);         _mm256_store_ps(c + 8, c01);
            _mm256_store_ps(c + N, c10);     _mm256_store_ps(c + N + 8, c11);
            _mm256_store_ps(c + 2 * N, c20); _mm256_store_ps(c + 2 * N + 8, c21);
            _mm256_store_ps(c + 3 * N, c30); _mm256_store_ps(c + 3 * N + 8, c31);
        }
    _mm256_zeroupper();
}
static T_TARGET("avx512f") void mm_avx512(float *C, const float *A, const float *B, int N) {
    for (int i = 0; i < N; i += 4)
        for (int j = 0; j < N; j += 32) {
            __m512 c00 = _mm512_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
            for (int k = 0; k < N; k++) {
                const float *b = B + k * N + j;
                __m512 b0 = _mm512_load_ps(b), b1 = _mm512_load_ps(b + 16);
                __m512 a0 = _mm512_set1_ps(A[(i + 0) * N + k]), a1 = _mm512_set1_ps(A[(i + 1) * N + k]);
                __m512 a2 = _mm512_set1_ps(A[(i + 2) * N + k]), a3 = _mm512_set1_ps(A[(i + 3) * N + k]);
                c00 = _mm512_fmadd_ps(a0, b0, c00); c01 = _mm512_fmadd_ps(a0, b1, c01);
                c10 = _mm512_fmadd_ps(a1, b0, c10); c11 = _mm512_fmadd_ps(a1, b1, c11);
                c20 = _mm512_fmadd_ps(a2, b0, c20); c21 = _mm512_fmadd_ps(a2, b1, c21);
                c30 = _mm512_fmadd_ps(a3, b0, c30); c31 = _mm512_fmadd_ps(a3, b1, c31);
            }
            float *c = C + i * N + j;
            _mm512_store_ps(c, c00);         _mm512_store_ps(c + 16, c01);
            _mm512_store_ps(c + N, c10);     _mm512_store_ps(c + N + 16, c11);
            _mm512_store_ps(c + 2 * N, c20); _mm512_store_ps(c + 2 * N + 16, c21);
            _mm512_store_ps(c + 3 * N, c30); _mm512_store_ps(c + 3 * N + 16, c31);
        }
    _mm256_zeroupper();
}

/* ============================================================================ popcount / byte sum */
typedef uint64_t (*red_fn)(const uint8_t *p, size_t n);   /* n multiple of 64, p 64-aligned */
static BM_NOVEC uint64_t pop_scalar_swar(const uint8_t *p, size_t n) {
    uint64_t t = 0;
    for (size_t i = 0; i < n; i += 8) {
        uint64_t v; memcpy(&v, p + i, 8);
        v = v - ((v >> 1) & 0x5555555555555555ull);
        v = (v & 0x3333333333333333ull) + ((v >> 2) & 0x3333333333333333ull);
        v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0Full;
        t += (v * 0x0101010101010101ull) >> 56;
    }
    return t;
}
static T_TARGET("popcnt") uint64_t pop_popcnt(const uint8_t *p, size_t n) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
#ifdef T_X64
    for (size_t i = 0; i < n; i += 32) {
        uint64_t a, b, c, d; memcpy(&a, p + i, 8); memcpy(&b, p + i + 8, 8); memcpy(&c, p + i + 16, 8); memcpy(&d, p + i + 24, 8);
        t0 += (uint64_t)__builtin_popcountll(a); t1 += (uint64_t)__builtin_popcountll(b);
        t2 += (uint64_t)__builtin_popcountll(c); t3 += (uint64_t)__builtin_popcountll(d);
    }
#else
    for (size_t i = 0; i < n; i += 16) {
        uint32_t a, b, c, d; memcpy(&a, p + i, 4); memcpy(&b, p + i + 4, 4); memcpy(&c, p + i + 8, 4); memcpy(&d, p + i + 12, 4);
        t0 += (uint32_t)__builtin_popcount(a); t1 += (uint32_t)__builtin_popcount(b);
        t2 += (uint32_t)__builtin_popcount(c); t3 += (uint32_t)__builtin_popcount(d);
    }
#endif
    return t0 + t1 + t2 + t3;
}
static T_TARGET("ssse3") uint64_t pop_ssse3_pshufb(const uint8_t *p, size_t n) {
    const __m128i lut = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4), lo = _mm_set1_epi8(0x0F);
    __m128i acc = _mm_setzero_si128();
    for (size_t i = 0; i < n; i += 64) {          /* per 64 bytes: per-byte counts <= 32, no overflow */
        __m128i c = _mm_setzero_si128();
        for (int j = 0; j < 4; j++) {
            __m128i v = _mm_load_si128((const __m128i *)(p + i + 16 * j));
            c = _mm_add_epi8(c, _mm_add_epi8(_mm_shuffle_epi8(lut, _mm_and_si128(v, lo)),
                                             _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(v, 4), lo))));
        }
        acc = _mm_add_epi64(acc, _mm_sad_epu8(c, _mm_setzero_si128()));
    }
    return (uint64_t)_mm_cvtsi128_si32(acc) + (uint64_t)_mm_cvtsi128_si32(_mm_srli_si128(acc, 8));
}
static T_TARGET("avx2") uint64_t pop_avx2_pshufb(const uint8_t *p, size_t n) {
    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i lo = _mm256_set1_epi8(0x0F);
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = 0; i < n; i += 64) {
        __m256i v0 = _mm256_load_si256((const __m256i *)(p + i)), v1 = _mm256_load_si256((const __m256i *)(p + i + 32));
        __m256i c = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v0, lo)), _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v0, 4), lo)));
        c = _mm256_add_epi8(c, _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v1, lo)), _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v1, 4), lo))));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(c, _mm256_setzero_si256()));
    }
    T_ALIGN(32) uint64_t q[4]; _mm256_store_si256((__m256i *)q, acc);
    _mm256_zeroupper();
    return q[0] + q[1] + q[2] + q[3];
}
static BM_NOVEC uint64_t bsum_scalar(const uint8_t *p, size_t n) { uint64_t s = 0; for (size_t i = 0; i < n; i++) s += p[i]; return s; }
static uint64_t bsum_sse2_psadbw(const uint8_t *p, size_t n) {
    __m128i a0 = _mm_setzero_si128(), a1 = a0, z = a0;
    for (size_t i = 0; i < n; i += 32) {
        a0 = _mm_add_epi64(a0, _mm_sad_epu8(_mm_load_si128((const __m128i *)(p + i)), z));
        a1 = _mm_add_epi64(a1, _mm_sad_epu8(_mm_load_si128((const __m128i *)(p + i + 16)), z));
    }
    a0 = _mm_add_epi64(a0, a1);
    T_ALIGN(16) uint64_t q[2]; _mm_store_si128((__m128i *)q, a0);
    return q[0] + q[1];
}
static T_TARGET("avx2") uint64_t bsum_avx2_psadbw(const uint8_t *p, size_t n) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, z = a0;
    for (size_t i = 0; i < n; i += 64) {
        a0 = _mm256_add_epi64(a0, _mm256_sad_epu8(_mm256_load_si256((const __m256i *)(p + i)), z));
        a1 = _mm256_add_epi64(a1, _mm256_sad_epu8(_mm256_load_si256((const __m256i *)(p + i + 32)), z));
    }
    a0 = _mm256_add_epi64(a0, a1);
    T_ALIGN(32) uint64_t q[4]; _mm256_store_si256((__m256i *)q, a0);
    _mm256_zeroupper();
    return q[0] + q[1] + q[2] + q[3];
}

/* ============================================================================ float -> int32 truncation */
typedef void (*cvt_fn)(int32_t *o, const float *in, size_t n);   /* n multiple of 16 */
static BM_NOVEC void cvt_scalar(int32_t *o, const float *in, size_t n) { for (size_t i = 0; i < n; i++) o[i] = (int32_t)in[i]; }
static void cvt_sse2(int32_t *o, const float *in, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        _mm_store_si128((__m128i *)(o + i), _mm_cvttps_epi32(_mm_load_ps(in + i)));
        _mm_store_si128((__m128i *)(o + i + 4), _mm_cvttps_epi32(_mm_load_ps(in + i + 4)));
    }
}
static T_TARGET("avx") void cvt_avx(int32_t *o, const float *in, size_t n) {
    for (size_t i = 0; i < n; i += 16) {
        _mm256_store_si256((__m256i *)(o + i), _mm256_cvttps_epi32(_mm256_load_ps(in + i)));
        _mm256_store_si256((__m256i *)(o + i + 8), _mm256_cvttps_epi32(_mm256_load_ps(in + i + 8)));
    }
    _mm256_zeroupper();
}

/* ============================================================================ gather: sum table[idx[i]] */
typedef uint64_t (*gat_fn)(const uint32_t *tab, const int32_t *idx, size_t n);   /* n multiple of 32 */
static BM_NOVEC uint64_t gat_scalar(const uint32_t *tab, const int32_t *idx, size_t n) {
    uint64_t s = 0; for (size_t i = 0; i < n; i++) s += tab[idx[i]]; return s;
}
static T_TARGET("avx2") uint64_t gat_avx2(const uint32_t *tab, const int32_t *idx, size_t n) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0;
    for (size_t i = 0; i < n; i += 16) {
        __m256i g0 = _mm256_i32gather_epi32((const int *)tab, _mm256_load_si256((const __m256i *)(idx + i)), 4);
        __m256i g1 = _mm256_i32gather_epi32((const int *)tab, _mm256_load_si256((const __m256i *)(idx + i + 8)), 4);
        a0 = _mm256_add_epi64(a0, _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(g0)), _mm256_cvtepu32_epi64(_mm256_extracti128_si256(g0, 1))));
        a1 = _mm256_add_epi64(a1, _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(g1)), _mm256_cvtepu32_epi64(_mm256_extracti128_si256(g1, 1))));
    }
    a0 = _mm256_add_epi64(a0, a1);
    T_ALIGN(32) uint64_t q[4]; _mm256_store_si256((__m256i *)q, a0);
    _mm256_zeroupper();
    return q[0] + q[1] + q[2] + q[3];
}
static T_TARGET("avx512f") uint64_t gat_avx512(const uint32_t *tab, const int32_t *idx, size_t n) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0;
    for (size_t i = 0; i < n; i += 32) {
        __m512i g0 = _mm512_i32gather_epi32(_mm512_load_si512((const void *)(idx + i)), (const void *)tab, 4);
        __m512i g1 = _mm512_i32gather_epi32(_mm512_load_si512((const void *)(idx + i + 16)), (const void *)tab, 4);
        a0 = _mm512_add_epi64(a0, _mm512_add_epi64(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(g0)), _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(g0, 1))));
        a1 = _mm512_add_epi64(a1, _mm512_add_epi64(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(g1)), _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(g1, 1))));
    }
    uint64_t r = (uint64_t)_mm512_reduce_add_epi64(_mm512_add_epi64(a0, a1));
    _mm256_zeroupper();
    return r;
}
#endif
