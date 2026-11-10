/* simd_dispatch - runtime CPU feature detection and implementation selection, done the way an
 * application does it, and proof that every implementation the dispatcher considers usable produces
 * output bit-identical to the scalar definition.
 *
 * The dispatcher (sd_*) is self-contained application-style code: it reads CPUID itself, executes
 * XGETBV only when CPUID.1:ECX.OSXSAVE[27] is set, and gates every VEX/EVEX tier on XCR0. It is
 * cross-checked against the suite's cpu_has()/cpu_hw(), IsProcessorFeaturePresent() and
 * GetEnabledXStateFeatures().
 *
 * Kernels (each with scalar / SSE2 / SSE4-class / AVX2 / AVX-512 tiers where they make sense):
 *   add_u32  wrapping 32-bit add            scalar, SSE2, AVX2, AVX-512F (masked tail)
 *   add_f32  IEEE binary32 add (MXCSR)      scalar ADDSS, SSE ADDPS, AVX VADDPS, AVX-512F (masked tail)
 *   memcpy   byte copy                      scalar, SSE2, AVX, AVX-512BW (masked tail)
 *   crc32c   Castagnoli, reflected 0x82F63B78   table, SSE4.2 CRC32
 *   adler32  RFC 1950                       scalar, SSE2 (PSADBW/PMADDWD), SSSE3 (PMADDUBSW), AVX2, AVX-512BW
 *   count8   count bytes equal to v         scalar, SSE2, SSE2+POPCNT, AVX2+POPCNT, AVX-512BW+POPCNT
 *
 * Architectural rules used for hard CHECKs (Intel SDM Vol.1 13.2/13.3/14.3/15.2, Vol.2 XGETBV/XSETBV,
 * AMD APM Vol.2 11.5):
 *   - XGETBV is #UD when CR4.OSXSAVE = 0, so it may only run when CPUID.1:ECX[27] = 1.
 *   - CR4.OSXSAVE can only be set when CPUID.1:ECX.XSAVE[26] = 1 (MOV CR4 #GP otherwise).
 *   - XCR0[0] is always 1; XSETBV #GPs for XCR0[2]=1 with XCR0[1]=0 and for partial AVX-512 bits.
 *   - VEX instructions #UD unless XCR0[2:1] = 11b; EVEX instructions #UD unless XCR0[7:5,2:1] all 1.
 *   - x86-64 implies SSE and SSE2 (AMD64 architecture baseline).
 *   - Masked-off lanes of AVX-512 masked loads/stores do not fault (fault suppression).
 * Legacy-encoded SSE..SSE4.2, SSSE3, POPCNT, CRC32 do not depend on XCR0.
 *
 * Windows reference (Windows 10 19041 corpus, decompiled via the reference API):
 *   - kernelbase!GetEnabledXStateFeatures() = RtlGetEnabledExtendedFeatures(~0), or 3 when that is 0,
 *     so XSTATE_LEGACY_FLOATING_POINT|XSTATE_LEGACY_SSE (3) are always reported.
 *   - ntdll!RtlGetEnabledExtendedFeatures(m) = (KUSER_SHARED_DATA.XState.EnabledFeatures |
 *     EnabledUserVisibleSupervisorFeatures) & m  -> a pure mask filter; checked below.
 *   - kernelbase!IsProcessorFeaturePresent(f) = RtlIsProcessorFeaturePresent(f) (reads
 *     KUSER_SHARED_DATA.ProcessorFeatures[f]). PF_SSSE3..PF_AVX512F (36..41) are newer constants:
 *     an older OS returns FALSE for them, so only "PF says present => really usable" is a hard check.
 *   GetEnabledXStateFeatures vs XCR0 disagreement is recorded as NOTE (guidance 10.3). It becomes a
 *   FAIL only if an application trusting the API would select an implementation that then faults.
 *
 * Force / override:
 *   SIMD_DISPATCH_FORCE=<scalar|sse2|sse4|avx2|avx512|auto> caps the global dispatcher of a run.
 *   --child=forced --force=<tier> runs the dispatcher with a cap in a child (the harness rejects
 *   unknown top-level arguments, so --force is accepted as a child argument; the env variable works
 *   for whole runs). A cap above what the machine supports falls back to the best usable tier: a
 *   forced request never makes the dispatcher select an unusable implementation.
 */
#include "t.h"
#include <immintrin.h>

#ifndef PF_XMMI_INSTRUCTIONS_AVAILABLE
#define PF_XMMI_INSTRUCTIONS_AVAILABLE 6
#endif
#ifndef PF_XMMI64_INSTRUCTIONS_AVAILABLE
#define PF_XMMI64_INSTRUCTIONS_AVAILABLE 10
#endif
#ifndef PF_SSE3_INSTRUCTIONS_AVAILABLE
#define PF_SSE3_INSTRUCTIONS_AVAILABLE 13
#endif
#ifndef PF_XSAVE_ENABLED
#define PF_XSAVE_ENABLED 17
#endif
#ifndef PF_SSSE3_INSTRUCTIONS_AVAILABLE
#define PF_SSSE3_INSTRUCTIONS_AVAILABLE 36
#endif
#ifndef PF_SSE4_1_INSTRUCTIONS_AVAILABLE
#define PF_SSE4_1_INSTRUCTIONS_AVAILABLE 37
#endif
#ifndef PF_SSE4_2_INSTRUCTIONS_AVAILABLE
#define PF_SSE4_2_INSTRUCTIONS_AVAILABLE 38
#endif
#ifndef PF_AVX_INSTRUCTIONS_AVAILABLE
#define PF_AVX_INSTRUCTIONS_AVAILABLE 39
#endif
#ifndef PF_AVX2_INSTRUCTIONS_AVAILABLE
#define PF_AVX2_INSTRUCTIONS_AVAILABLE 40
#endif
#ifndef PF_AVX512F_INSTRUCTIONS_AVAILABLE
#define PF_AVX512F_INSTRUCTIONS_AVAILABLE 41
#endif

#ifndef CONTEXT_XSTATE
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)   /* x86 value (winnt.h); mingw defines it for AMD64 only */
#endif
#define SD_STATUS_ILLEGAL 0xC000001Du
#define SD_ENV_FORCE "SIMD_DISPATCH_FORCE"

/* =============================================================================================
 * Capability model
 * =========================================================================================== */
enum {
    SD_SSE = 1u << 0, SD_SSE2 = 1u << 1, SD_SSSE3 = 1u << 2, SD_SSE41 = 1u << 3, SD_SSE42 = 1u << 4,
    SD_POPCNT = 1u << 5, SD_AVX = 1u << 6, SD_AVX2 = 1u << 7, SD_AVX512F = 1u << 8,
    SD_AVX512BW = 1u << 9, SD_AVX512VL = 1u << 10,
};
#define SD_NCAP 11
#define SD_LEGACY_CAPS (SD_SSE | SD_SSE2 | SD_SSSE3 | SD_SSE41 | SD_SSE42 | SD_POPCNT)
#define SD_YMM_CAPS    (SD_AVX | SD_AVX2)
#define SD_ZMM_CAPS    (SD_AVX512F | SD_AVX512BW | SD_AVX512VL)

typedef struct sd_capdef { unsigned bit; const char *name; t_feature f; uint32_t leaf; int reg, pos; } sd_capdef;
static const sd_capdef sd_capdefs[SD_NCAP] = {
    { SD_SSE,      "sse",      F_SSE,      1, CPUID_EDX, 25 },
    { SD_SSE2,     "sse2",     F_SSE2,     1, CPUID_EDX, 26 },
    { SD_SSSE3,    "ssse3",    F_SSSE3,    1, CPUID_ECX, 9 },
    { SD_SSE41,    "sse4.1",   F_SSE41,    1, CPUID_ECX, 19 },
    { SD_SSE42,    "sse4.2",   F_SSE42,    1, CPUID_ECX, 20 },
    { SD_POPCNT,   "popcnt",   F_POPCNT,   1, CPUID_ECX, 23 },
    { SD_AVX,      "avx",      F_AVX,      1, CPUID_ECX, 28 },
    { SD_AVX2,     "avx2",     F_AVX2,     7, CPUID_EBX, 5 },
    { SD_AVX512F,  "avx512f",  F_AVX512F,  7, CPUID_EBX, 16 },
    { SD_AVX512BW, "avx512bw", F_AVX512BW, 7, CPUID_EBX, 30 },
    { SD_AVX512VL, "avx512vl", F_AVX512VL, 7, CPUID_EBX, 31 },
};

/* How the dispatcher decides that the OS enabled the vector state.
 *   CORRECT       CPUID bit AND OSXSAVE AND XCR0 mask (the only correct rule)
 *   NAIVE         CPUID bit only (classic bug: crashes with #UD on an OS/VM that did not enable AVX)
 *   OSXSAVE_ONLY  CPUID + OSXSAVE but no XCR0 check (bug: XSAVE enabled, YMM state not)
 *   YMM_FOR_ZMM   AVX-512 gated on XCR0[2:1] only (bug: opmask/ZMM components not checked)
 *   API           XCR0 replaced by GetEnabledXStateFeatures() (what an app trusting the API does) */
enum { GATE_CORRECT, GATE_NAIVE, GATE_OSXSAVE_ONLY, GATE_YMM_FOR_ZMM, GATE_API };
static const char *sd_gate_name(int g) {
    static const char *n[] = { "correct", "naive_cpuid_only", "osxsave_only", "ymm_for_zmm", "xstate_api" };
    return g >= 0 && g <= GATE_API ? n[g] : "?";
}

/* CPUID/XGETBV source: the real processor, or a synthetic CPU for the decision-logic tests. */
typedef struct sd_src {
    void     (*cpuid)(void *ctx, uint32_t leaf, uint32_t sub, t_cpuid_regs *r);
    uint64_t (*xgetbv)(void *ctx, uint32_t idx);
    void     *ctx;
} sd_src;

typedef struct sd_caps {
    uint32_t maxleaf;
    t_cpuid_regs l1, l7;
    int      osxsave;
    int      xcr0_read;          /* XGETBV executed */
    uint64_t xcr0;               /* XGETBV(0) or the API mask for GATE_API, else 0 */
    int      ymm_ok, zmm_ok;     /* OS state gate result */
    unsigned hw;                 /* raw CPUID bits (SD_*) */
    unsigned usable;             /* hw gated by OS state per the gate rule */
} sd_caps;

static uint64_t sd_xgetbv_raw(uint32_t idx) {
    uint32_t a, d;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(a), "=d"(d) : "c"(idx));   /* XGETBV */
    return ((uint64_t)d << 32) | a;
}
static void real_cpuid(void *ctx, uint32_t leaf, uint32_t sub, t_cpuid_regs *r) { (void)ctx; cpuid_ex(leaf, sub, r); }
static uint64_t real_xgetbv(void *ctx, uint32_t idx) { (void)ctx; return sd_xgetbv_raw(idx); }
static const sd_src sd_real = { real_cpuid, real_xgetbv, NULL };

static void sd_detect(sd_caps *c, const sd_src *src, int gate, uint64_t api_mask) {
    memset(c, 0, sizeof *c);
    t_cpuid_regs r0;
    src->cpuid(src->ctx, 0, 0, &r0);
    c->maxleaf = r0.eax;
    if (c->maxleaf >= 1) src->cpuid(src->ctx, 1, 0, &c->l1);
    if (c->maxleaf >= 7) src->cpuid(src->ctx, 7, 0, &c->l7);
    for (int i = 0; i < SD_NCAP; i++) {
        const sd_capdef *d = &sd_capdefs[i];
        const t_cpuid_regs *r = d->leaf == 1 ? &c->l1 : &c->l7;
        if ((cpuid_reg(r, d->reg) >> d->pos) & 1) c->hw |= d->bit;
    }
    c->osxsave = (int)((c->l1.ecx >> 27) & 1);
    switch (gate) {
    case GATE_CORRECT:
    case GATE_YMM_FOR_ZMM:
        if (c->osxsave) { c->xcr0 = src->xgetbv(src->ctx, 0); c->xcr0_read = 1; }
        c->ymm_ok = c->osxsave && (c->xcr0 & 0x6) == 0x6;
        c->zmm_ok = gate == GATE_CORRECT ? (c->osxsave && (c->xcr0 & 0xE6) == 0xE6) : c->ymm_ok;
        break;
    case GATE_NAIVE:        c->ymm_ok = c->zmm_ok = 1; break;
    case GATE_OSXSAVE_ONLY: c->ymm_ok = c->zmm_ok = c->osxsave; break;
    case GATE_API:
        c->xcr0 = api_mask;
        c->ymm_ok = (api_mask & 0x6) == 0x6;
        c->zmm_ok = (api_mask & 0xE6) == 0xE6;
        break;
    }
    c->usable = c->hw & SD_LEGACY_CAPS;
    if (c->ymm_ok) c->usable |= c->hw & SD_YMM_CAPS;
    if (c->zmm_ok) c->usable |= c->hw & SD_ZMM_CAPS;
}

static char *sd_caps_str(char *buf, size_t cap, unsigned m) {
    size_t o = 0; buf[0] = 0;
    for (int i = 0; i < SD_NCAP; i++)
        if (m & sd_capdefs[i].bit) o += (size_t)snprintf(buf + o, o < cap ? cap - o : 0, "%s%s", o ? "," : "", sd_capdefs[i].name);
    if (!o) snprintf(buf, cap, "none");
    return buf;
}

/* =============================================================================================
 * Implementations
 * =========================================================================================== */
enum { K_ADD_U32, K_ADD_F32, K_MEMCPY, K_CRC32C, K_ADLER32, K_COUNT8, K_COUNT };
static const char *sd_kname[K_COUNT] = { "add_u32", "add_f32", "memcpy", "crc32c", "adler32", "count8" };
enum { TIER_SCALAR, TIER_SSE2, TIER_SSE4, TIER_AVX2, TIER_AVX512, TIER_COUNT };
static const char *sd_tname[TIER_COUNT] = { "scalar", "sse2", "sse4", "avx2", "avx512" };

enum {
    I_ADD_U32_SCALAR, I_ADD_U32_SSE2, I_ADD_U32_AVX2, I_ADD_U32_AVX512,
    I_ADD_F32_SCALAR, I_ADD_F32_SSE, I_ADD_F32_AVX, I_ADD_F32_AVX512,
    I_MEMCPY_SCALAR, I_MEMCPY_SSE2, I_MEMCPY_AVX, I_MEMCPY_AVX512BW,
    I_CRC_SCALAR, I_CRC_SSE42,
    I_ADLER_SCALAR, I_ADLER_SSE2, I_ADLER_SSSE3, I_ADLER_AVX2, I_ADLER_AVX512BW,
    I_COUNT_SCALAR, I_COUNT_SSE2, I_COUNT_POPCNT, I_COUNT_AVX2, I_COUNT_AVX512BW,
    I_N
};

/* Execution evidence: every implementation counts its own entries. */
static volatile LONG g_calls[I_N];
#define SD_ENTER(id) InterlockedIncrement(&g_calls[id])
static void sd_calls_reset(void) { for (int i = 0; i < I_N; i++) InterlockedExchange(&g_calls[i], 0); }

typedef void     (*fn_add_u32)(uint32_t *d, const uint32_t *a, const uint32_t *b, size_t n);
typedef void     (*fn_add_f32)(float *d, const float *a, const float *b, size_t n);
typedef void    *(*fn_memcpy)(void *d, const void *s, size_t n);
typedef uint32_t (*fn_crc)(uint32_t crc, const void *p, size_t n);
typedef uint32_t (*fn_adler)(uint32_t adler, const void *p, size_t n);
typedef size_t   (*fn_count)(const void *p, size_t n, uint8_t v);
typedef void     (*sd_fnptr)(void);

#define SD_SCALAR_ATTR __attribute__((noinline, optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")))
#define SD_ADLER_MOD  65521u
#define SD_ADLER_NMAX 5552u     /* largest n with 255n(n+1)/2 + (n+1)(MOD-1) < 2^32 (zlib) */

/* ---- add_u32 -------------------------------------------------------------------------------- */
static SD_SCALAR_ATTR void add_u32_scalar(uint32_t *d, const uint32_t *a, const uint32_t *b, size_t n) {
    SD_ENTER(I_ADD_U32_SCALAR);
    for (size_t i = 0; i < n; i++) d[i] = a[i] + b[i];
}
static T_NOINLINE void add_u32_sse2(uint32_t *d, const uint32_t *a, const uint32_t *b, size_t n) {
    SD_ENTER(I_ADD_U32_SSE2);
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_storeu_si128((__m128i *)(d + i), _mm_add_epi32(_mm_loadu_si128((const __m128i *)(a + i)), _mm_loadu_si128((const __m128i *)(b + i))));
    for (; i < n; i++) d[i] = a[i] + b[i];
}
static T_TARGET("avx2") T_NOINLINE void add_u32_avx2(uint32_t *d, const uint32_t *a, const uint32_t *b, size_t n) {
    SD_ENTER(I_ADD_U32_AVX2);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_si256((__m256i *)(d + i), _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(a + i)), _mm256_loadu_si256((const __m256i *)(b + i))));
    if (i + 4 <= n) {
        _mm_storeu_si128((__m128i *)(d + i), _mm_add_epi32(_mm_loadu_si128((const __m128i *)(a + i)), _mm_loadu_si128((const __m128i *)(b + i))));
        i += 4;
    }
    for (; i < n; i++) d[i] = a[i] + b[i];
}
static T_TARGET("avx2,avx512f") T_NOINLINE void add_u32_avx512(uint32_t *d, const uint32_t *a, const uint32_t *b, size_t n) {
    SD_ENTER(I_ADD_U32_AVX512);
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_si512((void *)(d + i), _mm512_add_epi32(_mm512_loadu_si512((const void *)(a + i)), _mm512_loadu_si512((const void *)(b + i))));
    if (i < n) {   /* masked tail: inactive lanes must neither load nor store (fault suppression) */
        __mmask16 m = (__mmask16)((1u << (n - i)) - 1u);
        __m512i s = _mm512_add_epi32(_mm512_maskz_loadu_epi32(m, a + i), _mm512_maskz_loadu_epi32(m, b + i));
        _mm512_mask_storeu_epi32(d + i, m, s);
    }
}

/* ---- add_f32 (bit-exact: every tier uses the same IEEE add under the same MXCSR) -------------
 * With two NaN inputs SSE/AVX return the FIRST source operand's NaN (quieted) (SDM Vol.1 4.8.3.5,
 * Table 4-7). C intrinsics such as _mm_add_ps are commutative to the compiler, which may swap the
 * operands, so the adds are written in asm with src1 = a in every tier and in the reference. */
static inline __m128 sd_addss(__m128 a, __m128 b) { __asm__("addss %1, %0" : "+x"(a) : "x"(b)); return a; }
static inline __m128 sd_addps(__m128 a, __m128 b) { __asm__("addps %1, %0" : "+x"(a) : "x"(b)); return a; }
static SD_SCALAR_ATTR void add_f32_scalar(float *d, const float *a, const float *b, size_t n) {
    SD_ENTER(I_ADD_F32_SCALAR);
    for (size_t i = 0; i < n; i++) _mm_store_ss(d + i, sd_addss(_mm_load_ss(a + i), _mm_load_ss(b + i)));
}
static T_NOINLINE void add_f32_sse(float *d, const float *a, const float *b, size_t n) {
    SD_ENTER(I_ADD_F32_SSE);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(d + i, sd_addps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    for (; i < n; i++) _mm_store_ss(d + i, sd_addss(_mm_load_ss(a + i), _mm_load_ss(b + i)));
}
static T_TARGET("avx") T_NOINLINE void add_f32_avx(float *d, const float *a, const float *b, size_t n) {
    SD_ENTER(I_ADD_F32_AVX);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i), r;
        __asm__("vaddps %2, %1, %0" : "=x"(r) : "x"(va), "x"(vb));   /* r = va + vb, src1 = va */
        _mm256_storeu_ps(d + i, r);
    }
    for (; i < n; i++) {
        __m128 va = _mm_load_ss(a + i), vb = _mm_load_ss(b + i), r;
        __asm__("vaddss %2, %1, %0" : "=x"(r) : "x"(va), "x"(vb));
        _mm_store_ss(d + i, r);
    }
}
static T_TARGET("avx2,avx512f") T_NOINLINE void add_f32_avx512(float *d, const float *a, const float *b, size_t n) {
    SD_ENTER(I_ADD_F32_AVX512);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i), vb = _mm512_loadu_ps(b + i), r;
        __asm__("vaddps %2, %1, %0" : "=v"(r) : "v"(va), "v"(vb));
        _mm512_storeu_ps(d + i, r);
    }
    if (i < n) {   /* masked tail: inactive lanes are zero (0+0 raises no flag) and are not stored */
        __mmask16 m = (__mmask16)((1u << (n - i)) - 1u);
        __m512 va = _mm512_maskz_loadu_ps(m, a + i), vb = _mm512_maskz_loadu_ps(m, b + i), r;
        __asm__("vaddps %2, %1, %0" : "=v"(r) : "v"(va), "v"(vb));
        _mm512_mask_storeu_ps(d + i, m, r);
    }
}

/* ---- memcpy ----------------------------------------------------------------------------------- */
static SD_SCALAR_ATTR void *memcpy_scalar(void *d, const void *s, size_t n) {
    SD_ENTER(I_MEMCPY_SCALAR);
    volatile uint8_t *dp = (volatile uint8_t *)d; const uint8_t *sp = (const uint8_t *)s;
    for (size_t i = 0; i < n; i++) dp[i] = sp[i];
    return d;
}
static T_NOINLINE void *memcpy_sse2(void *d, const void *s, size_t n) {
    SD_ENTER(I_MEMCPY_SSE2);
    uint8_t *dp = (uint8_t *)d; const uint8_t *sp = (const uint8_t *)s;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) _mm_storeu_si128((__m128i *)(dp + i), _mm_loadu_si128((const __m128i *)(sp + i)));
    for (; i < n; i++) dp[i] = sp[i];
    return d;
}
static T_TARGET("avx") T_NOINLINE void *memcpy_avx(void *d, const void *s, size_t n) {
    SD_ENTER(I_MEMCPY_AVX);
    uint8_t *dp = (uint8_t *)d; const uint8_t *sp = (const uint8_t *)s;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) _mm256_storeu_si256((__m256i *)(dp + i), _mm256_loadu_si256((const __m256i *)(sp + i)));
    if (i + 16 <= n) { _mm_storeu_si128((__m128i *)(dp + i), _mm_loadu_si128((const __m128i *)(sp + i))); i += 16; }
    for (; i < n; i++) dp[i] = sp[i];
    return d;
}
static T_TARGET("avx2,avx512f,avx512bw") T_NOINLINE void *memcpy_avx512bw(void *d, const void *s, size_t n) {
    SD_ENTER(I_MEMCPY_AVX512BW);
    uint8_t *dp = (uint8_t *)d; const uint8_t *sp = (const uint8_t *)s;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) _mm512_storeu_si512((void *)(dp + i), _mm512_loadu_si512((const void *)(sp + i)));
    if (i < n) {
        __mmask64 m = (__mmask64)(((uint64_t)1 << (n - i)) - 1u);
        _mm512_mask_storeu_epi8(dp + i, m, _mm512_maskz_loadu_epi8(m, sp + i));
    }
    return d;
}

/* ---- crc32c ----------------------------------------------------------------------------------- */
static uint32_t g_crc_tab[256];
static void __attribute__((constructor)) sd_crc_table_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ ((c & 1) ? 0x82F63B78u : 0);
        g_crc_tab[i] = c;
    }
}
static SD_SCALAR_ATTR uint32_t crc32c_scalar(uint32_t crc, const void *p, size_t n) {
    SD_ENTER(I_CRC_SCALAR);
    const uint8_t *b = (const uint8_t *)p; uint32_t c = ~crc;
    for (size_t i = 0; i < n; i++) c = g_crc_tab[(c ^ b[i]) & 0xFF] ^ (c >> 8);
    return ~c;
}
static T_TARGET("sse4.2") T_NOINLINE uint32_t crc32c_sse42(uint32_t crc, const void *p, size_t n) {
    SD_ENTER(I_CRC_SSE42);
    const uint8_t *b = (const uint8_t *)p; uint32_t c = ~crc;
    while (n && ((uintptr_t)b & 7)) { c = _mm_crc32_u8(c, *b++); n--; }
#ifdef T_X64
    uint64_t c64 = c;
    while (n >= 8) { uint64_t v; memcpy(&v, b, 8); c64 = _mm_crc32_u64(c64, v); b += 8; n -= 8; }
    c = (uint32_t)c64;
#endif
    while (n >= 4) { uint32_t v; memcpy(&v, b, 4); c = _mm_crc32_u32(c, v); b += 4; n -= 4; }
    while (n--) c = _mm_crc32_u8(c, *b++);
    return ~c;
}

/* ---- adler32 ---------------------------------------------------------------------------------- */
static SD_SCALAR_ATTR uint32_t adler32_scalar(uint32_t adler, const void *p, size_t n) {
    SD_ENTER(I_ADLER_SCALAR);
    const uint8_t *b = (const uint8_t *)p; uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    while (n) {
        size_t chunk = n < SD_ADLER_NMAX ? n : SD_ADLER_NMAX; n -= chunk;
        while (chunk--) { s1 += *b++; s2 += s1; }
        s1 %= SD_ADLER_MOD; s2 %= SD_ADLER_MOD;
    }
    return s1 | (s2 << 16);
}
static inline uint32_t sd_hsum_epi32(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
    return (uint32_t)_mm_cvtsi128_si32(v);
}
static T_NOINLINE uint32_t adler32_sse2(uint32_t adler, const void *p, size_t n) {
    SD_ENTER(I_ADLER_SSE2);
    const uint8_t *b = (const uint8_t *)p; uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    const __m128i zero = _mm_setzero_si128();
    const __m128i w0 = _mm_setr_epi16(16, 15, 14, 13, 12, 11, 10, 9), w1 = _mm_setr_epi16(8, 7, 6, 5, 4, 3, 2, 1);
    while (n) {
        size_t chunk = n < SD_ADLER_NMAX ? n : SD_ADLER_NMAX; n -= chunk;
        for (; chunk >= 16; chunk -= 16, b += 16) {
            __m128i v = _mm_loadu_si128((const __m128i *)b);
            __m128i sad = _mm_sad_epu8(v, zero);
            uint32_t sum = (uint32_t)_mm_cvtsi128_si32(sad) + (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(sad, 8));
            __m128i m = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi8(v, zero), w0), _mm_madd_epi16(_mm_unpackhi_epi8(v, zero), w1));
            s2 += 16 * s1 + sd_hsum_epi32(m);
            s1 += sum;
        }
        while (chunk--) { s1 += *b++; s2 += s1; }
        s1 %= SD_ADLER_MOD; s2 %= SD_ADLER_MOD;
    }
    return s1 | (s2 << 16);
}
static const int8_t sd_w64[64] T_ALIGN(64) = {
    64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
static T_TARGET("ssse3") T_NOINLINE uint32_t adler32_ssse3(uint32_t adler, const void *p, size_t n) {
    SD_ENTER(I_ADLER_SSSE3);
    const uint8_t *b = (const uint8_t *)p; uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    const __m128i zero = _mm_setzero_si128(), ones = _mm_set1_epi16(1);
    const __m128i w = _mm_load_si128((const __m128i *)(sd_w64 + 48));   /* 16..1 */
    while (n) {
        size_t chunk = n < SD_ADLER_NMAX ? n : SD_ADLER_NMAX; n -= chunk;
        for (; chunk >= 16; chunk -= 16, b += 16) {
            __m128i v = _mm_loadu_si128((const __m128i *)b);
            __m128i sad = _mm_sad_epu8(v, zero);
            uint32_t sum = (uint32_t)_mm_cvtsi128_si32(sad) + (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(sad, 8));
            s2 += 16 * s1 + sd_hsum_epi32(_mm_madd_epi16(_mm_maddubs_epi16(v, w), ones));
            s1 += sum;
        }
        while (chunk--) { s1 += *b++; s2 += s1; }
        s1 %= SD_ADLER_MOD; s2 %= SD_ADLER_MOD;
    }
    return s1 | (s2 << 16);
}
static T_TARGET("avx2") T_NOINLINE uint32_t adler32_avx2(uint32_t adler, const void *p, size_t n) {
    SD_ENTER(I_ADLER_AVX2);
    const uint8_t *b = (const uint8_t *)p; uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    const __m256i zero = _mm256_setzero_si256(), ones = _mm256_set1_epi16(1);
    const __m256i w = _mm256_load_si256((const __m256i *)(sd_w64 + 32));   /* 32..1 */
    while (n) {
        size_t chunk = n < SD_ADLER_NMAX ? n : SD_ADLER_NMAX; n -= chunk;
        for (; chunk >= 32; chunk -= 32, b += 32) {
            __m256i v = _mm256_loadu_si256((const __m256i *)b);
            __m256i sad = _mm256_sad_epu8(v, zero);
            __m256i m = _mm256_madd_epi16(_mm256_maddubs_epi16(v, w), ones);
            __m128i sx = _mm_add_epi32(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
            __m128i mx = _mm_add_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
            s2 += 32 * s1 + sd_hsum_epi32(mx);
            s1 += sd_hsum_epi32(sx);
        }
        while (chunk--) { s1 += *b++; s2 += s1; }
        s1 %= SD_ADLER_MOD; s2 %= SD_ADLER_MOD;
    }
    return s1 | (s2 << 16);
}
static T_TARGET("avx2,avx512f,avx512bw") T_NOINLINE uint32_t adler32_avx512bw(uint32_t adler, const void *p, size_t n) {
    SD_ENTER(I_ADLER_AVX512BW);
    const uint8_t *b = (const uint8_t *)p; uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    const __m512i zero = _mm512_setzero_si512(), ones = _mm512_set1_epi16(1);
    const __m512i w = _mm512_load_si512((const void *)sd_w64);   /* 64..1: pair sums <= 255*127 fit int16 */
    while (n) {
        size_t chunk = n < SD_ADLER_NMAX ? n : SD_ADLER_NMAX; n -= chunk;
        for (; chunk >= 64; chunk -= 64, b += 64) {
            __m512i v = _mm512_loadu_si512((const void *)b);
            uint32_t sum = (uint32_t)_mm512_reduce_add_epi32(_mm512_sad_epu8(v, zero));
            uint32_t ws = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(v, w), ones));
            s2 += 64 * s1 + ws;
            s1 += sum;
        }
        while (chunk--) { s1 += *b++; s2 += s1; }
        s1 %= SD_ADLER_MOD; s2 %= SD_ADLER_MOD;
    }
    return s1 | (s2 << 16);
}

/* ---- count8 ----------------------------------------------------------------------------------- */
static SD_SCALAR_ATTR size_t count8_scalar(const void *p, size_t n, uint8_t v) {
    SD_ENTER(I_COUNT_SCALAR);
    const uint8_t *b = (const uint8_t *)p; size_t c = 0;
    for (size_t i = 0; i < n; i++) c += b[i] == v;
    return c;
}
static T_NOINLINE size_t count8_sse2(const void *p, size_t n, uint8_t v) {
    SD_ENTER(I_COUNT_SSE2);
    const uint8_t *b = (const uint8_t *)p; size_t i = 0, total = 0;
    const __m128i vv = _mm_set1_epi8((char)v), zero = _mm_setzero_si128();
    while (n - i >= 16) {
        __m128i acc = zero;
        for (int it = 0; n - i >= 16 && it < 255; it++, i += 16)
            acc = _mm_sub_epi8(acc, _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(b + i)), vv));
        __m128i s = _mm_sad_epu8(acc, zero);
        total += (uint32_t)_mm_cvtsi128_si32(s) + (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(s, 8));
    }
    for (; i < n; i++) total += b[i] == v;
    return total;
}
static T_TARGET("popcnt") T_NOINLINE size_t count8_popcnt(const void *p, size_t n, uint8_t v) {
    SD_ENTER(I_COUNT_POPCNT);
    const uint8_t *b = (const uint8_t *)p; size_t i = 0, total = 0;
    const __m128i vv = _mm_set1_epi8((char)v);
    for (; i + 16 <= n; i += 16)
        total += (size_t)__builtin_popcount((unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(b + i)), vv)));
    for (; i < n; i++) total += b[i] == v;
    return total;
}
static T_TARGET("avx2,popcnt") T_NOINLINE size_t count8_avx2(const void *p, size_t n, uint8_t v) {
    SD_ENTER(I_COUNT_AVX2);
    const uint8_t *b = (const uint8_t *)p; size_t i = 0, total = 0;
    const __m256i vv = _mm256_set1_epi8((char)v);
    for (; i + 32 <= n; i += 32)
        total += (size_t)__builtin_popcount((unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(b + i)), vv)));
    for (; i < n; i++) total += b[i] == v;
    return total;
}
static T_TARGET("avx2,avx512f,avx512bw,popcnt") T_NOINLINE size_t count8_avx512bw(const void *p, size_t n, uint8_t v) {
    SD_ENTER(I_COUNT_AVX512BW);
    const uint8_t *b = (const uint8_t *)p; size_t i = 0, total = 0;
    const __m512i vv = _mm512_set1_epi8((char)v);
    for (; i + 64 <= n; i += 64)
        total += (size_t)__builtin_popcountll((unsigned long long)_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void *)(b + i)), vv));
    if (i < n) {
        __mmask64 m = (__mmask64)(((uint64_t)1 << (n - i)) - 1u);
        __mmask64 e = _mm512_mask_cmpeq_epi8_mask(m, _mm512_maskz_loadu_epi8(m, b + i), vv);
        total += (size_t)__builtin_popcountll((unsigned long long)e);
    }
    return total;
}

/* ---- implementation table ------------------------------------------------------------------- */
typedef struct sd_impl { int kernel, tier; const char *name; unsigned req; sd_fnptr fn; } sd_impl;
#define FN(f) ((sd_fnptr)(void *)(f))
static const sd_impl g_impl[I_N] = {
    { K_ADD_U32, TIER_SCALAR, "add_u32_scalar",    0,                                     FN(add_u32_scalar) },
    { K_ADD_U32, TIER_SSE2,   "add_u32_sse2",      SD_SSE2,                               FN(add_u32_sse2) },
    { K_ADD_U32, TIER_AVX2,   "add_u32_avx2",      SD_AVX | SD_AVX2,                      FN(add_u32_avx2) },
    { K_ADD_U32, TIER_AVX512, "add_u32_avx512",    SD_AVX | SD_AVX2 | SD_AVX512F,         FN(add_u32_avx512) },
    { K_ADD_F32, TIER_SCALAR, "add_f32_scalar",    SD_SSE,                                FN(add_f32_scalar) },
    { K_ADD_F32, TIER_SSE2,   "add_f32_sse",       SD_SSE,                                FN(add_f32_sse) },
    { K_ADD_F32, TIER_AVX2,   "add_f32_avx",       SD_AVX,                                FN(add_f32_avx) },
    { K_ADD_F32, TIER_AVX512, "add_f32_avx512",    SD_AVX | SD_AVX2 | SD_AVX512F,         FN(add_f32_avx512) },
    { K_MEMCPY,  TIER_SCALAR, "memcpy_scalar",     0,                                     FN(memcpy_scalar) },
    { K_MEMCPY,  TIER_SSE2,   "memcpy_sse2",       SD_SSE2,                               FN(memcpy_sse2) },
    { K_MEMCPY,  TIER_AVX2,   "memcpy_avx",        SD_AVX,                                FN(memcpy_avx) },
    { K_MEMCPY,  TIER_AVX512, "memcpy_avx512bw",   SD_AVX | SD_AVX2 | SD_AVX512F | SD_AVX512BW, FN(memcpy_avx512bw) },
    { K_CRC32C,  TIER_SCALAR, "crc32c_scalar",     0,                                     FN(crc32c_scalar) },
    { K_CRC32C,  TIER_SSE4,   "crc32c_sse42",      SD_SSE42,                              FN(crc32c_sse42) },
    { K_ADLER32, TIER_SCALAR, "adler32_scalar",    0,                                     FN(adler32_scalar) },
    { K_ADLER32, TIER_SSE2,   "adler32_sse2",      SD_SSE2,                               FN(adler32_sse2) },
    { K_ADLER32, TIER_SSE4,   "adler32_ssse3",     SD_SSE2 | SD_SSSE3,                    FN(adler32_ssse3) },
    { K_ADLER32, TIER_AVX2,   "adler32_avx2",      SD_AVX | SD_AVX2,                      FN(adler32_avx2) },
    { K_ADLER32, TIER_AVX512, "adler32_avx512bw",  SD_AVX | SD_AVX2 | SD_AVX512F | SD_AVX512BW, FN(adler32_avx512bw) },
    { K_COUNT8,  TIER_SCALAR, "count8_scalar",     0,                                     FN(count8_scalar) },
    { K_COUNT8,  TIER_SSE2,   "count8_sse2",       SD_SSE2,                               FN(count8_sse2) },
    { K_COUNT8,  TIER_SSE4,   "count8_popcnt",     SD_SSE2 | SD_POPCNT,                   FN(count8_popcnt) },
    { K_COUNT8,  TIER_AVX2,   "count8_avx2",       SD_AVX | SD_AVX2 | SD_POPCNT,          FN(count8_avx2) },
    { K_COUNT8,  TIER_AVX512, "count8_avx512bw",   SD_AVX | SD_AVX2 | SD_AVX512F | SD_AVX512BW | SD_POPCNT, FN(count8_avx512bw) },
};

static const char *sd_iname(int id) { return id >= 0 && id < I_N ? g_impl[id].name : "none"; }

/* ---- independent references (the scalar definitions; never counted) -------------------------- */
static SD_SCALAR_ATTR void ref_add_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (uint32_t)((uint64_t)a[i] + b[i]);
}
static SD_SCALAR_ATTR void ref_add_f32(float *d, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {   /* a + b with src1 = a: NaN-propagation order pinned */
        __m128 va = _mm_load_ss(a + i), vb = _mm_load_ss(b + i);
        __asm__("addss %1, %0" : "+x"(va) : "x"(vb));
        _mm_store_ss(d + i, va);
    }
}
static SD_SCALAR_ATTR void *ref_memcpy(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i++) ((volatile uint8_t *)d)[i] = ((const uint8_t *)s)[i];
    return d;
}
static SD_SCALAR_ATTR uint32_t ref_crc32c(uint32_t crc, const void *p, size_t n) {   /* bitwise */
    uint32_t c = ~crc;
    for (size_t i = 0; i < n; i++) {
        c ^= ((const uint8_t *)p)[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1)));
    }
    return ~c;
}
static SD_SCALAR_ATTR uint32_t ref_adler32(uint32_t adler, const void *p, size_t n) {   /* mod every byte */
    uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    for (size_t i = 0; i < n; i++) { s1 = (s1 + ((const uint8_t *)p)[i]) % SD_ADLER_MOD; s2 = (s2 + s1) % SD_ADLER_MOD; }
    return s1 | (s2 << 16);
}
static SD_SCALAR_ATTR size_t ref_count8(const void *p, size_t n, uint8_t v) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) c += ((const uint8_t *)p)[i] == v;
    return c;
}
static const sd_fnptr g_ref[K_COUNT] = { FN(ref_add_u32), FN(ref_add_f32), FN(ref_memcpy), FN(ref_crc32c), FN(ref_adler32), FN(ref_count8) };

static size_t sd_es(int k) { return (k == K_ADD_U32 || k == K_ADD_F32) ? 4 : 1; }

/* Run kernel k through fn. Array kernels write d; the return value is the reduction result
 * (memcpy: 1 if it returned its destination). */
static uint64_t sd_run(int k, sd_fnptr fn, uint8_t *d, const uint8_t *a, const uint8_t *b, size_t n, uint32_t param) {
    switch (k) {
    case K_ADD_U32: ((fn_add_u32)(void *)fn)((uint32_t *)d, (const uint32_t *)a, (const uint32_t *)b, n); return 0;
    case K_ADD_F32: ((fn_add_f32)(void *)fn)((float *)d, (const float *)a, (const float *)b, n); return 0;
    case K_MEMCPY:  return ((fn_memcpy)(void *)fn)(d, a, n) == (void *)d;
    case K_CRC32C:  return ((fn_crc)(void *)fn)(param, a, n);
    case K_ADLER32: return ((fn_adler)(void *)fn)(param, a, n);
    case K_COUNT8:  return ((fn_count)(void *)fn)(a, n, (uint8_t)param);
    }
    return 0;
}

/* =============================================================================================
 * Dispatcher
 * =========================================================================================== */
static int sd_select(int k, unsigned usable, int cap) {
    int best = -1;
    for (int i = 0; i < I_N; i++) {
        const sd_impl *im = &g_impl[i];
        if (im->kernel != k || im->tier > cap || (im->req & ~usable)) continue;
        if (best < 0 || im->tier > g_impl[best].tier) best = i;
    }
    return best;
}
static int sd_impl_usable(int id, unsigned usable) { return (g_impl[id].req & ~usable) == 0; }

typedef struct sd_disp {
    INIT_ONCE once;
    volatile LONG init_runs;
    int cap, gate;
    uint64_t api_mask;
    const sd_src *src;
    sd_caps caps;
    int sel[K_COUNT];
} sd_disp;

static BOOL CALLBACK sd_init_cb(PINIT_ONCE o, PVOID param, PVOID *ctx) {
    (void)o; (void)ctx;
    sd_disp *d = (sd_disp *)param;
    InterlockedIncrement(&d->init_runs);
    sd_detect(&d->caps, d->src ? d->src : &sd_real, d->gate, d->api_mask);
    for (int k = 0; k < K_COUNT; k++) d->sel[k] = sd_select(k, d->caps.usable, d->cap);
    return TRUE;
}
static void sd_disp_prepare(sd_disp *d, int cap, int gate, uint64_t api_mask) {
    memset(d, 0, sizeof *d);
    InitOnceInitialize(&d->once);
    d->cap = cap; d->gate = gate; d->api_mask = api_mask; d->src = &sd_real;
}
static sd_disp *sd_get(sd_disp *d) { InitOnceExecuteOnce(&d->once, sd_init_cb, d, NULL); return d; }
static void sd_disp_make(sd_disp *d, int cap, int gate, uint64_t api_mask) { sd_disp_prepare(d, cap, gate, api_mask); sd_get(d); }

static int sd_parse_tier(const char *s) {
    if (!s || !*s || !_stricmp(s, "auto") || !_stricmp(s, "best")) return TIER_AVX512;
    if (!_stricmp(s, "scalar") || !_stricmp(s, "c") || !_stricmp(s, "none")) return TIER_SCALAR;
    if (!_stricmp(s, "sse2") || !_stricmp(s, "sse")) return TIER_SSE2;
    if (!_stricmp(s, "sse4") || !_stricmp(s, "sse41") || !_stricmp(s, "sse42") || !_stricmp(s, "ssse3")) return TIER_SSE4;
    if (!_stricmp(s, "avx2") || !_stricmp(s, "avx")) return TIER_AVX2;
    if (!_stricmp(s, "avx512") || !_stricmp(s, "avx-512")) return TIER_AVX512;
    return -1;
}
static int sd_tier_usable(int tier, unsigned usable) {
    for (int i = 0; i < I_N; i++) if (g_impl[i].tier == tier && sd_impl_usable(i, usable)) return 1;
    return 0;
}

/* Global application dispatcher: honours SIMD_DISPATCH_FORCE like an application environment knob. */
static sd_disp g_disp;
static volatile LONG g_disp_ready;
static int g_disp_force_invalid;
static sd_disp *sd_global(void) {
    if (!g_disp_ready) {
        static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
        BOOL pending = FALSE;
        if (InitOnceBeginInitialize(&once, 0, &pending, NULL) && pending) {
            char v[64] = "";
            DWORD n = GetEnvironmentVariableA(SD_ENV_FORCE, v, sizeof v);
            int cap = n && n < sizeof v ? sd_parse_tier(v) : TIER_AVX512;
            if (cap < 0) { g_disp_force_invalid = 1; cap = TIER_AVX512; }
            sd_disp_make(&g_disp, cap, GATE_CORRECT, 0);
            InterlockedExchange(&g_disp_ready, 1);
            InitOnceComplete(&once, 0, NULL);
        }
    }
    return &g_disp;
}
static sd_fnptr sd_fn(const sd_disp *d, int k) { return g_impl[d->sel[k]].fn; }

/* The "machine truth" used to judge the dispatcher: the correct gate on the real CPU. */
static sd_caps g_truth;
static BOOL CALLBACK sd_truth_cb(PINIT_ONCE o, PVOID p, PVOID *ctx) {
    (void)o; (void)p; (void)ctx;
    sd_detect(&g_truth, &sd_real, GATE_CORRECT, 0);
    return TRUE;
}
static const sd_caps *sd_truth(void) {   /* computed once; safe to call first from any thread */
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, sd_truth_cb, NULL, NULL);
    return &g_truth;
}
static int sd_usable_id(int id) { return sd_impl_usable(id, sd_truth()->usable); }
static const char *sd_req_str(int id) { static char b[160]; return sd_caps_str(b, sizeof b, g_impl[id].req); }

/* =============================================================================================
 * Verification engine (buffers, patterns, sweeps)
 * =========================================================================================== */
#define SD_NPAT 5
static const char *sd_pat_name(int k, int p) {
    static const char *common[] = { "random", "zeros", "ones_ff", "ramp" };
    if (p < 4) return common[p];
    switch (k) {
    case K_ADD_U32: return "u32_wrap_edges";
    case K_ADD_F32: return "f32_specials";
    case K_COUNT8:  return "count_targets";
    default:        return "sparse";
    }
}
static const uint32_t sd_f32_specials[] = {
    0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x7F800000u, 0xFF800000u, /* +-0 +-1 +-inf */
    0x7FC00000u, 0x7FC12345u, 0xFFC00001u, 0x7F800001u, 0xFFBFFFFFu,             /* qNaN, payloads, sNaN */
    0x00000001u, 0x807FFFFFu, 0x00400000u,                                       /* denormals */
    0x7F7FFFFFu, 0xFF7FFFFFu, 0x00800000u, 0x80800000u,                          /* max, min normal */
    0x3F800001u, 0x33800000u, 0x4B000000u, 0xCB000001u,                          /* rounding boundaries */
};
static void sd_fill(int k, int pat, uint8_t *buf, size_t cap, uint64_t seed) {
    t_rng r; t_rng_seed(&r, seed * 0x9E3779B97F4A7C15ull + (uint64_t)k * 131 + (uint64_t)pat);
    switch (pat) {
    case 0: t_rng_fill(&r, buf, cap); return;
    case 1: memset(buf, 0, cap); return;
    case 2: memset(buf, 0xFF, cap); return;
    case 3: for (size_t i = 0; i < cap; i++) buf[i] = (uint8_t)(i * 7 + 3); return;
    }
    if (k == K_ADD_U32) {
        static const uint32_t e[] = { 0, 1, 2, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFFu, 0xFFFFFFFEu, 0x55555555u, 0xAAAAAAAAu };
        for (size_t i = 0; i + 4 <= cap; i += 4) {
            uint32_t v = (t_rng_u32(&r) & 3) ? e[t_rng_range(&r, T_COUNTOF(e))] : t_rng_u32(&r);
            memcpy(buf + i, &v, 4);
        }
    } else if (k == K_ADD_F32) {
        for (size_t i = 0; i + 4 <= cap; i += 4) {
            uint32_t v = (t_rng_u32(&r) & 3) ? sd_f32_specials[t_rng_range(&r, T_COUNTOF(sd_f32_specials))] : t_rng_u32(&r);
            memcpy(buf + i, &v, 4);
        }
    } else if (k == K_COUNT8) {
        static const uint8_t e[] = { 0x5A, 0x00, 0xFF, 0x5B };
        for (size_t i = 0; i < cap; i++) buf[i] = e[t_rng_range(&r, 4)];
    } else {
        for (size_t i = 0; i < cap; i++) buf[i] = (t_rng_u32(&r) % 13 == 0) ? (uint8_t)t_rng_u32(&r) : 0;
    }
}
/* kernel parameter for one combination: crc seed, adler state (s1,s2 < 65521), count8 target byte */
static uint32_t sd_param(int k, size_t n, int off) {
    uint32_t h = (uint32_t)t_mix64(((uint64_t)n << 8) ^ (uint64_t)off);
    switch (k) {
    case K_CRC32C:  return (off & 1) ? h : 0;
    case K_ADLER32: return (off & 1) ? (((h >> 16) % SD_ADLER_MOD) << 16) | ((h & 0xFFFF) % SD_ADLER_MOD) : 1;
    case K_COUNT8:  { static const uint8_t v[] = { 0x5A, 0x00, 0xFF, 0x5B }; return v[off & 3]; }
    }
    return 0;
}

typedef struct sd_fail {
    int have, pat, n, oa, ob, od;
    uint32_t param;
} sd_fail;

/* Sweep one implementation: sizes 0..nmax elements, input/dst offsets 0..offmax-1 (elements), all patterns.
 * Destination buffers are pre-filled with a sentinel, so a write outside [od, od+n) is a mismatch too.
 * In-process (do_checks=1): one CHECK per size (first 8 failures printed) and one aggregate CHECK.
 * In a child (do_checks=0): returns the mismatch count only. */
static uint64_t sd_sweep(int id, int nmax, int offmax, uint64_t seed, int do_checks, uint64_t *combos_out) {
    const sd_impl *im = &g_impl[id];
    int k = im->kernel; size_t es = sd_es(k);
    size_t cap = ((size_t)nmax + 64 + 64) * es + 64;
    uint8_t *A[SD_NPAT], *B[SD_NPAT];
    uint8_t *G = (uint8_t *)_aligned_malloc(cap, 64), *R = (uint8_t *)_aligned_malloc(cap, 64);
    int ok_alloc = G && R;
    for (int p = 0; p < SD_NPAT; p++) {
        A[p] = (uint8_t *)_aligned_malloc(cap, 64); B[p] = (uint8_t *)_aligned_malloc(cap, 64);
        if (!A[p] || !B[p]) ok_alloc = 0;
        else { sd_fill(k, p, A[p], cap, seed); sd_fill(k, p, B[p], cap, seed ^ 0xB0B); }
    }
    uint64_t bad = 0, combos = 0;
    int reported = 0;
    sd_fail f; memset(&f, 0, sizeof f);
    if (ok_alloc) {
        for (int n = 0; n <= nmax; n++) {
            uint64_t bad_n = 0, comb_n = 0;
            for (int p = 0; p < SD_NPAT; p++) {
                for (int off = 0; off < offmax; off++) {
                    int oa = off, ob = (off * 13 + 5) & 63, od = (off * 7 + 3) & 63;
                    uint32_t param = sd_param(k, (size_t)n, off);
                    memset(G, 0xCD, cap); memset(R, 0xCD, cap);
                    uint64_t rg = sd_run(k, im->fn, G + od * es, A[p] + oa * es, B[p] + ob * es, (size_t)n, param);
                    uint64_t rr = sd_run(k, g_ref[k], R + od * es, A[p] + oa * es, B[p] + ob * es, (size_t)n, param);
                    comb_n++;
                    if (rg != rr || memcmp(G, R, cap) != 0) {
                        bad_n++;
                        if (!f.have) { f.have = 1; f.pat = p; f.n = n; f.oa = oa; f.ob = ob; f.od = od; f.param = param; }
                    }
                }
            }
            bad += bad_n; combos += comb_n;
            if (do_checks) {
                int ok = bad_n == 0;
                if (ok || reported++ < 8)
                    CHECKF(ok, "%s: n=%d: %llu of %llu (pattern x offset) combinations differ from the scalar reference",
                           im->name, n, (unsigned long long)bad_n, (unsigned long long)comb_n);
            }
        }
    }
    if (do_checks) {
        CHECKF(ok_alloc, "%s: buffer allocation failed", im->name);
        CHECKF(bad == 0, "%s: %llu of %llu combinations mismatched (sizes 0..%d, offsets 0..%d, %d patterns)",
               im->name, (unsigned long long)bad, (unsigned long long)combos, nmax, offmax - 1, SD_NPAT);
        if (f.have) {   /* reproduce the first failure with full diagnostics */
            memset(G, 0xCD, cap); memset(R, 0xCD, cap);
            uint64_t rg = sd_run(k, im->fn, G + f.od * es, A[f.pat] + f.oa * es, B[f.pat] + f.ob * es, (size_t)f.n, f.param);
            uint64_t rr = sd_run(k, g_ref[k], R + f.od * es, A[f.pat] + f.oa * es, B[f.pat] + f.ob * es, (size_t)f.n, f.param);
            t_note("%s first mismatch: pattern=%s n=%d off_a=%d off_b=%d off_d=%d param=0x%x result=0x%llx ref=0x%llx",
                   im->name, sd_pat_name(k, f.pat), f.n, f.oa, f.ob, f.od, f.param, (unsigned long long)rg, (unsigned long long)rr);
            CHECK_MEM(G, R, cap);
        }
    }
    for (int p = 0; p < SD_NPAT; p++) { _aligned_free(A[p]); _aligned_free(B[p]); }
    _aligned_free(G); _aligned_free(R);
    if (combos_out) *combos_out = combos;
    return bad;
}

static void sd_skip_unusable(int id) {
    const sd_caps *c = sd_truth();
    unsigned miss = g_impl[id].req & ~c->usable;
    if (!miss) return;
    char m[160]; sd_caps_str(m, sizeof m, miss);
    if ((g_impl[id].req & ~c->hw) == 0)
        SKIP("%s needs %s: advertised by CPUID but the OS has not enabled the state (OSXSAVE=%d XCR0=0x%llx)",
             g_impl[id].name, m, c->osxsave, (unsigned long long)c->xcr0);
    SKIP("%s needs %s, not present on this CPU", g_impl[id].name, m);
}

/* Full impl case: 0..300 sizes x 0..63 offsets x 5 patterns, compared with the scalar definition. */
static void sd_impl_case(int id) {
    sd_skip_unusable(id);
    sd_calls_reset();
    uint64_t combos = 0;
    sd_sweep(id, 300, 64, 0x5D15A7C4ull + (uint64_t)id, 1, &combos);
    CHECKF(g_calls[id] > 0, "%s was not executed", g_impl[id].name);
    t_info("combinations", "%llu", (unsigned long long)combos);
    t_info("requires", "%s", sd_req_str(id));
}

/* =============================================================================================
 * Children
 * =========================================================================================== */
static void sd_print_disp(const sd_disp *d) {
    printf("GATE=%s\nCAP=%d\nUSABLE=0x%x\nHW=0x%x\nOSXSAVE=%d\nXCR0=0x%llx\n", sd_gate_name(d->gate), d->cap,
           d->caps.usable, d->caps.hw, d->caps.osxsave, (unsigned long long)d->caps.xcr0);
    for (int k = 0; k < K_COUNT; k++) printf("SEL_%s=%d\n", sd_kname[k], d->sel[k]);
}

/* Small dispatched workload through table d; returns mismatches vs reference. */
static uint64_t sd_workload(const sd_disp *d, uint64_t seed, int iters, size_t maxn) {
    size_t cap = (maxn + 128) * 4 + 64;
    uint8_t *a = (uint8_t *)malloc(cap), *b = (uint8_t *)malloc(cap), *g = (uint8_t *)malloc(cap), *r = (uint8_t *)malloc(cap);
    uint64_t bad = 0;
    if (!a || !b || !g || !r) { free(a); free(b); free(g); free(r); return 1; }
    t_rng rng; t_rng_seed(&rng, seed);
    t_rng_fill(&rng, a, cap); t_rng_fill(&rng, b, cap);
    for (int it = 0; it < iters; it++) {
        int k = it % K_COUNT;
        if (d->sel[k] < 0) continue;
        size_t es = sd_es(k), n = (size_t)t_rng_range(&rng, maxn + 1);
        int oa = (int)t_rng_range(&rng, 64), ob = (int)t_rng_range(&rng, 64), od = (int)t_rng_range(&rng, 64);
        uint32_t param = sd_param(k, n, it);
        memset(g, 0xCD, cap); memset(r, 0xCD, cap);
        uint64_t rg = sd_run(k, sd_fn(d, k), g + od * es, a + oa * es, b + ob * es, n, param);
        uint64_t rr = sd_run(k, g_ref[k], r + od * es, a + oa * es, b + ob * es, n, param);
        if (rg != rr || memcmp(g, r, cap)) bad++;
    }
    free(a); free(b); free(g); free(r);
    return bad;
}

/* <exe> --child=decisions : print what the correct dispatcher decides in a fresh process */
CHILD(decisions) {
    (void)argc; (void)argv;
    sd_disp d; sd_disp_make(&d, TIER_AVX512, GATE_CORRECT, 0);
    sd_print_disp(&d);
    return 0;
}

/* <exe> --child=forced [--force=<tier>] : dispatcher capped by --force (else SIMD_DISPATCH_FORCE) */
CHILD(forced) {
    const char *force = NULL, *src = "none";
    char envv[64] = "";
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], "--force=", 8)) { force = argv[i] + 8; src = "cli"; }
    if (!force) {
        DWORD n = GetEnvironmentVariableA(SD_ENV_FORCE, envv, sizeof envv);
        if (n && n < sizeof envv) { force = envv; src = "env"; }
    }
    int cap = sd_parse_tier(force), invalid = 0;
    if (cap < 0) { invalid = 1; cap = TIER_AVX512; }
    sd_disp d; sd_disp_make(&d, cap, GATE_CORRECT, 0);
    printf("FORCE_SOURCE=%s\nFORCE_VALUE=%s\nFORCE_INVALID=%d\nFORCE_HONORED=%d\n", src, force ? force : "",
           invalid, sd_tier_usable(cap, d.caps.usable));
    sd_print_disp(&d);
    fflush(stdout);
    sd_calls_reset();
    uint64_t bad = 0;
    for (int k = 0; k < K_COUNT; k++) if (d.sel[k] >= 0) bad += sd_sweep(d.sel[k], 96, 16, 77 + (uint64_t)k, 0, NULL);
    sd_calls_reset();
    bad += sd_workload(&d, 0xF0CE, 600, 400);
    long unselected = 0, selected_calls = 0;
    for (int i = 0; i < I_N; i++) {
        int is_sel = 0;
        for (int k = 0; k < K_COUNT; k++) if (d.sel[k] == i) is_sel = 1;
        if (is_sel) selected_calls += g_calls[i]; else unselected += g_calls[i];
    }
    printf("VERIFY_MISMATCH=%llu\nUNSELECTED_CALLS=%ld\nSELECTED_CALLS=%ld\nDONE=1\n", (unsigned long long)bad, unselected, selected_calls);
    return 0;
}

/* <exe> --child=global : the process-wide dispatcher as an application initializes it (SIMD_DISPATCH_FORCE) */
CHILD(global) {
    (void)argc; (void)argv;
    sd_disp *d = sd_global();
    printf("FORCE_INVALID=%d\n", g_disp_force_invalid);
    sd_print_disp(d);
    sd_calls_reset();
    uint64_t bad = sd_workload(d, 0x610BA1, 600, 400);
    long unselected = 0;
    for (int i = 0; i < I_N; i++) {
        int is_sel = 0;
        for (int k = 0; k < K_COUNT; k++) if (d->sel[k] == i) is_sel = 1;
        if (!is_sel) unselected += g_calls[i];
    }
    printf("VERIFY_MISMATCH=%llu\nUNSELECTED_CALLS=%ld\nDONE=1\n", (unsigned long long)bad, unselected);
    return 0;
}

/* <exe> --child=run_impl <id> : execute one implementation (used to prove an unusable pick faults) */
CHILD(run_impl) {
    if (argc < 2) return 3;
    int id = atoi(argv[1]);
    if (id < 0 || id >= I_N) return 3;
    static uint8_t a[1024] T_ALIGN(64), b[1024] T_ALIGN(64), d[1024] T_ALIGN(64);
    for (int i = 0; i < 1024; i++) { a[i] = (uint8_t)i; b[i] = (uint8_t)(i * 3); }
    printf("RUN=%s\n", g_impl[id].name); fflush(stdout);
    uint64_t v = sd_run(g_impl[id].kernel, g_impl[id].fn, d, a, b, 200, 1);
    printf("RESULT=0x%llx\n", (unsigned long long)v);
    return 0;
}

/* <exe> --child=misdetect <avx2|avx512> : a dispatcher with a broken gate runs its pick */
CHILD(misdetect) {
    int want512 = argc > 1 && !strcmp(argv[1], "avx512");
    sd_disp d; sd_disp_make(&d, TIER_AVX512, want512 ? GATE_YMM_FOR_ZMM : GATE_NAIVE, 0);
    sd_print_disp(&d);
    int id = d.sel[K_ADD_U32];
    printf("PICK=%s\n", sd_iname(id)); fflush(stdout);
    static uint32_t a[64], b[64], r[64];
    for (int i = 0; i < 64; i++) { a[i] = (uint32_t)i; b[i] = 1000u; }
    if (id >= 0) ((fn_add_u32)(void *)g_impl[id].fn)(r, a, b, 64);
    printf("SURVIVED=1 r63=%u\n", r[63]);
    return 0;
}

/* Front-guarded buffer: n bytes starting right after a PAGE_NOACCESS page. */
static uint8_t *sd_front_guarded(size_t n, void **base) {
    size_t pg = 4096, total = pg + ((n + pg - 1) / pg + 1) * pg;
    uint8_t *p = (uint8_t *)VirtualAlloc(NULL, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    DWORD old;
    if (!p) return NULL;
    if (!VirtualProtect(p, pg, PAGE_NOACCESS, &old)) { VirtualFree(p, 0, MEM_RELEASE); return NULL; }
    *base = p;
    return p + pg;
}

/* <exe> --child=guard_tail : every usable impl with inputs/outputs ending at (and starting right after)
 * an inaccessible page, sizes 0..300. A tail over-read / over-write faults; a masked tail must not. */
CHILD(guard_tail) {
    (void)argc; (void)argv;
    const unsigned usable = sd_truth()->usable;
    const size_t maxb = 300 * 4;
    void *ba, *bb, *bd, *fa, *fb, *fd;
    uint8_t *ea = (uint8_t *)t_guarded_alloc(maxb, &ba), *eb = (uint8_t *)t_guarded_alloc(maxb, &bb), *ed = (uint8_t *)t_guarded_alloc(maxb, &bd);
    uint8_t *sa = sd_front_guarded(maxb, &fa), *sb = sd_front_guarded(maxb, &fb), *sdst = sd_front_guarded(maxb, &fd);
    if (!ea || !eb || !ed || !sa || !sb || !sdst) { printf("ALLOC_FAIL=1\n"); return 4; }
    static uint8_t ref[300 * 4 + 64];
    int done = 0;
    for (int id = 0; id < I_N; id++) {
        if (!sd_impl_usable(id, usable)) continue;
        int k = g_impl[id].kernel; size_t es = sd_es(k);
        printf("BEGIN %s\n", g_impl[id].name); fflush(stdout);
        unsigned long mism = 0;
        for (int n = 0; n <= 300; n++) {
            size_t nb = (size_t)n * es;
            for (int side = 0; side < 2; side++) {
                uint8_t *a = side ? sa : ea + maxb - nb, *b = side ? sb : eb + maxb - nb, *d = side ? sdst : ed + maxb - nb;
                for (size_t i = 0; i < nb; i++) { a[i] = (uint8_t)(i * 31 + n); b[i] = (uint8_t)(i * 17 + 5); d[i] = 0xCD; }
                memset(ref, 0xCD, sizeof ref);
                uint32_t param = sd_param(k, (size_t)n, side);
                uint64_t rg = sd_run(k, g_impl[id].fn, d, a, b, (size_t)n, param);
                uint64_t rr = sd_run(k, g_ref[k], ref, a, b, (size_t)n, param);
                if (k == K_MEMCPY) { rg = 1; rr = 1; }   /* return value compared in the sweeps */
                if (rg != rr || memcmp(d, ref, nb)) mism++;
            }
        }
        printf("OK %s mism=%lu\n", g_impl[id].name, mism); fflush(stdout);
        done++;
    }
    printf("DONE=%d\n", done);
    t_guarded_free(ba); t_guarded_free(bb); t_guarded_free(bd);
    VirtualFree(fa, 0, MEM_RELEASE); VirtualFree(fb, 0, MEM_RELEASE); VirtualFree(fd, 0, MEM_RELEASE);
    return 0;
}

static int sd_find_int(const t_child_result *r, const char *prefix, long long *out) {
    char v[128];
    if (!t_child_find(r, prefix, v, sizeof v)) return 0;
    *out = strtoll(v, NULL, 0);
    return 1;
}

/* =============================================================================================
 * Synthetic CPUs for the decision logic
 * =========================================================================================== */
typedef struct sd_fake {
    uint32_t maxleaf;
    t_cpuid_regs l1, l7;
    uint64_t xcr0;
    int xgetbv_calls, xgetbv_illegal, cpuid_out_of_range;
} sd_fake;
static void fake_cpuid(void *ctx, uint32_t leaf, uint32_t sub, t_cpuid_regs *r) {
    sd_fake *f = (sd_fake *)ctx; (void)sub;
    memset(r, 0, sizeof *r);
    if (leaf > f->maxleaf) { f->cpuid_out_of_range++; r->eax = 0xDEADBEEFu; r->ebx = r->ecx = r->edx = 0xFFFFFFFFu; return; }
    if (leaf == 0) r->eax = f->maxleaf;
    else if (leaf == 1) *r = f->l1;
    else if (leaf == 7) *r = f->l7;
}
static uint64_t fake_xgetbv(void *ctx, uint32_t idx) {
    sd_fake *f = (sd_fake *)ctx; (void)idx;
    f->xgetbv_calls++;
    if (!((f->l1.ecx >> 27) & 1)) { f->xgetbv_illegal++; return 0xFFFFFFFFFFFFFFFFull; }   /* would #UD */
    return f->xcr0;
}
/* Build a fake CPU from SD_* capability bits plus OSXSAVE/XSAVE and XCR0. */
static void fake_make(sd_fake *f, unsigned caps, int osxsave, uint64_t xcr0, uint32_t maxleaf) {
    memset(f, 0, sizeof *f);
    f->maxleaf = maxleaf; f->xcr0 = xcr0;
    for (int i = 0; i < SD_NCAP; i++) {
        if (!(caps & sd_capdefs[i].bit)) continue;
        t_cpuid_regs *r = sd_capdefs[i].leaf == 1 ? &f->l1 : &f->l7;
        uint32_t m = 1u << sd_capdefs[i].pos;
        switch (sd_capdefs[i].reg) { case CPUID_EAX: r->eax |= m; break; case CPUID_EBX: r->ebx |= m; break;
                                     case CPUID_ECX: r->ecx |= m; break; default: r->edx |= m; }
    }
    if (osxsave) f->l1.ecx |= (1u << 27) | (1u << 26);
}
static void fake_detect(sd_fake *f, int gate, sd_caps *c, int sel[K_COUNT], int cap) {
    sd_src s = { fake_cpuid, fake_xgetbv, f };
    sd_detect(c, &s, gate, 0);
    if (sel) for (int k = 0; k < K_COUNT; k++) sel[k] = sd_select(k, c->usable, cap);
}
static int sel_tier(int id) { return id < 0 ? -1 : g_impl[id].tier; }
#define SD_ALL_CAPS ((1u << SD_NCAP) - 1u)

/* =============================================================================================
 * Cases: detection on the real CPU
 * =========================================================================================== */
TEST(detect_cpuid_basic_leaves) {
    const sd_caps *c = sd_truth();
    REQUIREF(cpuid_available(), "CPUID not available");
    CHECKF(c->maxleaf >= 1, "CPUID.0:EAX max basic leaf = %u", c->maxleaf);
    t_cpuid_regs r; cpuid_ex(0, 0, &r);
    char vendor[13]; memcpy(vendor, &r.ebx, 4); memcpy(vendor + 4, &r.edx, 4); memcpy(vendor + 8, &r.ecx, 4); vendor[12] = 0;
    int printable = 1;
    for (int i = 0; i < 12; i++) if (vendor[i] < 0x20 || vendor[i] > 0x7E) printable = 0;
    CHECKF(printable, "vendor string not printable");
    CHECK_EQ(r.eax, c->maxleaf);
    CHECK_EQ(c->maxleaf, cpu()->max_basic);
    t_info("vendor", "%s", vendor);
    t_info("max_basic_leaf", "0x%x", c->maxleaf);
    t_info("leaf1_ecx_edx", "0x%08x 0x%08x", c->l1.ecx, c->l1.edx);
    t_info("leaf7_ebx_ecx_edx", "0x%08x 0x%08x 0x%08x", c->l7.ebx, c->l7.ecx, c->l7.edx);
    if (c->maxleaf < 7) {
        CHECKF((c->hw & (SD_AVX2 | SD_ZMM_CAPS)) == 0, "leaf-7 features reported although max leaf %u < 7", c->maxleaf);
        t_note("max basic leaf < 7: no AVX2/AVX-512 can be detected");
    }
}

TEST(detect_baseline_sse_sse2) {
    const sd_caps *c = sd_truth();
#ifdef T_X64
    /* AMD64 architecture: every x86-64 processor implements SSE and SSE2 */
    CHECKF(c->hw & SD_SSE, "x64 processor without SSE (CPUID.1:EDX[25]=0)");
    CHECKF(c->hw & SD_SSE2, "x64 processor without SSE2 (CPUID.1:EDX[26]=0)");
#else
    /* this x86 build itself uses SSE2 (-msse2): running at all implies it */
    CHECKF(c->hw & SD_SSE2, "SSE2 not advertised but the x86 suite floor is SSE2");
    CHECKF(c->hw & SD_SSE, "SSE2 without SSE advertised");
#endif
    for (int k = 0; k < K_COUNT; k++) {
        int s = sd_select(k, c->usable, TIER_AVX512);
        CHECKF(s >= 0, "kernel %s has no usable implementation", sd_kname[k]);
        CHECKF(sd_select(k, c->usable, TIER_SCALAR) >= 0, "kernel %s has no scalar fallback", sd_kname[k]);
        CHECKF(sel_tier(sd_select(k, c->usable, TIER_SSE2)) >= (k == K_CRC32C ? TIER_SCALAR : TIER_SSE2) ||
               k == K_CRC32C, "SSE2-capable machine but %s does not pick an SSE2 tier", sd_kname[k]);
    }
}

TEST(detect_osxsave_implies_xsave) {
    const sd_caps *c = sd_truth();
    int xsave = (int)((c->l1.ecx >> 26) & 1);
    t_info("xsave_osxsave", "%d %d", xsave, c->osxsave);
    /* MOV to CR4 with OSXSAVE=1 #GPs unless CPUID.1:ECX.XSAVE = 1 (SDM Vol.3 2.5) */
    if (c->osxsave) CHECKF(xsave, "CPUID.1:ECX.OSXSAVE=1 but XSAVE=0");
    if (!xsave) CHECKF(!c->osxsave, "OSXSAVE without XSAVE");
    if (!c->osxsave) {
        CHECKF(!c->xcr0_read, "dispatcher executed XGETBV with OSXSAVE=0 (would #UD)");
        CHECKF((c->usable & (SD_YMM_CAPS | SD_ZMM_CAPS)) == 0, "VEX/EVEX tier usable without OSXSAVE");
        SKIP("OSXSAVE=0: the OS has not enabled XSAVE; only legacy SSE tiers are usable");
    }
    CHECK(c->xcr0_read);
}

TEST(detect_xcr0_architectural_invariants) {
    const sd_caps *c = sd_truth();
    if (!c->osxsave) SKIP("OSXSAVE=0: XGETBV not executable");
    uint64_t x = c->xcr0;
    t_info("xcr0", "0x%llx", (unsigned long long)x);
    CHECKF(x & 1, "XCR0[0] (x87) must always be 1: XCR0=0x%llx", (unsigned long long)x);
    CHECKF(!(x & 4) || (x & 2), "XCR0[2] (AVX) set without XCR0[1] (SSE): XCR0=0x%llx", (unsigned long long)x);
    CHECKF((x & 0xE0) == 0 || (x & 0xE0) == 0xE0, "partial AVX-512 state in XCR0=0x%llx", (unsigned long long)x);
    CHECKF((x & 0xE0) == 0 || (x & 0x6) == 0x6, "AVX-512 state without AVX state: XCR0=0x%llx", (unsigned long long)x);
    CHECK_EQ(x, sd_xgetbv_raw(0));
    CHECK_EQ(x, cpu()->xcr0);
    /* XCR0 may only contain bits the processor supports (CPUID.(0Dh,0) EDX:EAX) */
    if (c->maxleaf >= 0xD) {
        t_cpuid_regs d; cpuid_ex(0xD, 0, &d);
        uint64_t sup = ((uint64_t)d.edx << 32) | d.eax;
        CHECKF((x & ~sup) == 0, "XCR0=0x%llx has bits outside CPUID.(0Dh,0) supported mask 0x%llx", (unsigned long long)x, (unsigned long long)sup);
    }
}

TEST(detect_avx_gate_rule) {
    const sd_caps *c = sd_truth();
    int hw = !!(c->hw & SD_AVX), want = hw && c->osxsave && (c->xcr0 & 6) == 6;
    CHECK_EQ(!!(c->usable & SD_AVX), want);
    CHECK_EQ(!!(c->usable & SD_AVX), cpu_has(F_AVX));
    CHECK_EQ(c->ymm_ok, cpu_ymm_state_enabled());
    t_info("avx", "hw=%d osxsave=%d xcr0=0x%llx usable=%d", hw, c->osxsave, (unsigned long long)c->xcr0, want);
    if (hw && !want) t_note("AVX advertised by CPUID but not enabled by the OS (XCR0[2:1] != 11b): AVX tiers must stay unused");
    if (!hw) SKIP("CPU does not advertise AVX");
}

TEST(detect_avx2_gate_rule) {
    const sd_caps *c = sd_truth();
    int hw = !!(c->hw & SD_AVX2), want = hw && c->ymm_ok;
    CHECK_EQ(!!(c->usable & SD_AVX2), want);
    CHECK_EQ(!!(c->usable & SD_AVX2), cpu_has(F_AVX2));
    if (hw && !(c->hw & SD_AVX)) t_note("CPUID advertises AVX2 without AVX: AVX2 tiers require both and stay unused");
    /* AVX2 tier impls require AVX too */
    for (int k = 0; k < K_COUNT; k++) {
        int s = sd_select(k, c->usable, TIER_AVX512);
        if (sel_tier(s) >= TIER_AVX2) CHECKF(c->ymm_ok, "%s selected without YMM state enabled", g_impl[s].name);
    }
    if (!hw) SKIP("CPU does not advertise AVX2");
}

TEST(detect_avx512_gate_rule) {
    const sd_caps *c = sd_truth();
    int hw = !!(c->hw & SD_AVX512F), want = hw && c->osxsave && (c->xcr0 & 0xE6) == 0xE6;
    CHECK_EQ(!!(c->usable & SD_AVX512F), want);
    CHECK_EQ(!!(c->usable & SD_AVX512F), cpu_has(F_AVX512F));
    CHECK_EQ(!!(c->usable & SD_AVX512BW), cpu_has(F_AVX512BW));
    CHECK_EQ(!!(c->usable & SD_AVX512VL), cpu_has(F_AVX512VL));
    CHECK_EQ(c->zmm_ok, cpu_zmm_state_enabled());
    for (int k = 0; k < K_COUNT; k++) {
        int s = sd_select(k, c->usable, TIER_AVX512);
        if (sel_tier(s) == TIER_AVX512) CHECKF(c->zmm_ok, "%s selected without opmask/ZMM state enabled", g_impl[s].name);
    }
    if (hw && !want) t_note("AVX-512F advertised but XCR0[7:5] not enabled (XCR0=0x%llx): AVX-512 tiers stay unused", (unsigned long long)c->xcr0);
    if (!hw) SKIP("CPU does not advertise AVX-512F");
}

TEST(detect_legacy_sse_tiers_ignore_xcr0) {
    const sd_caps *c = sd_truth();
    /* SSE..SSE4.2, SSSE3, POPCNT: legacy encodings, usable == advertised regardless of XCR0 */
    CHECK_EQ(c->usable & SD_LEGACY_CAPS, c->hw & SD_LEGACY_CAPS);
    sd_caps g[5];
    for (int gate = GATE_CORRECT; gate <= GATE_YMM_FOR_ZMM; gate++) {
        sd_detect(&g[gate], &sd_real, gate, 0);
        CHECKF((g[gate].usable & SD_LEGACY_CAPS) == (c->hw & SD_LEGACY_CAPS), "gate %s changed legacy caps", sd_gate_name(gate));
    }
    sd_detect(&g[4], &sd_real, GATE_API, 0);   /* even a zero API mask leaves legacy SSE usable */
    CHECK_EQ(g[4].usable & SD_LEGACY_CAPS, c->hw & SD_LEGACY_CAPS);
    CHECK_EQ(g[4].usable & (SD_YMM_CAPS | SD_ZMM_CAPS), 0);
}

TEST(detect_matches_suite_feature_detection) {
    const sd_caps *c = sd_truth();
    for (int i = 0; i < SD_NCAP; i++) {
        const sd_capdef *d = &sd_capdefs[i];
        CHECKF(!!(c->hw & d->bit) == !!cpu_hw(d->f), "%s: dispatcher hw=%d suite cpu_hw=%d", d->name, !!(c->hw & d->bit), cpu_hw(d->f));
        CHECKF(!!(c->usable & d->bit) == !!cpu_has(d->f), "%s: dispatcher usable=%d suite cpu_has=%d", d->name, !!(c->usable & d->bit), cpu_has(d->f));
    }
    CHECK_EQ(c->osxsave, cpu()->osxsave);
    char h[200], u[200];
    t_info("hw", "%s", sd_caps_str(h, sizeof h, c->hw));
    t_info("usable", "%s", sd_caps_str(u, sizeof u, c->usable));
}

TEST(detect_repeatable) {
    const sd_caps *c0 = sd_truth();
    int diff = 0;
    for (int i = 0; i < 2000; i++) {
        sd_caps c; sd_detect(&c, &sd_real, GATE_CORRECT, 0);
        if (c.usable != c0->usable || c.hw != c0->hw || c.xcr0 != c0->xcr0 || c.osxsave != c0->osxsave) diff++;
        if (i % 500 == 0) Sleep(0);
    }
    CHECKF(diff == 0, "%d of 2000 repeated detections differed", diff);
    for (int k = 0; k < K_COUNT; k++) {
        sd_disp d; sd_disp_make(&d, TIER_AVX512, GATE_CORRECT, 0);
        CHECK_EQ(d.sel[k], sd_select(k, c0->usable, TIER_AVX512));
        CHECK_EQ(d.init_runs, 1);
    }
}

TEST(detect_identical_on_every_cpu) {
    int ncpu = t_cpu_count();
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) || !pm) SKIP("process affinity mask unavailable");
    const sd_caps *c0 = sd_truth();
    int sel0[K_COUNT];
    for (int k = 0; k < K_COUNT; k++) sel0[k] = sd_select(k, c0->usable, TIER_AVX512);
    uintptr_t prev = 0; int visited = 0, pinfail = 0, elsewhere = 0;
    for (int cpu_i = 0; cpu_i < (int)(sizeof(DWORD_PTR) * 8) && visited < 64; cpu_i++) {
        if (!(pm & ((DWORD_PTR)1 << cpu_i))) continue;
        uintptr_t p = t_pin_thread(cpu_i);
        if (!p) { pinfail++; continue; }
        if (!prev) prev = p;
        SwitchToThread();
        DWORD on = GetCurrentProcessorNumber();
        if ((int)on != cpu_i) elsewhere++;
        sd_caps c; sd_detect(&c, &sd_real, GATE_CORRECT, 0);
        CHECKF(c.usable == c0->usable, "cpu %d: usable caps 0x%x differ from 0x%x", cpu_i, c.usable, c0->usable);
        CHECKF(c.xcr0 == c0->xcr0, "cpu %d: XCR0 0x%llx differs from 0x%llx", cpu_i, (unsigned long long)c.xcr0, (unsigned long long)c0->xcr0);
        CHECKF(c.osxsave == c0->osxsave, "cpu %d: OSXSAVE differs", cpu_i);
        for (int k = 0; k < K_COUNT; k++)
            CHECKF(sd_select(k, c.usable, TIER_AVX512) == sel0[k], "cpu %d: %s decision differs", cpu_i, sd_kname[k]);
        if (c.hw != c0->hw) t_note("cpu %d: raw CPUID feature bits 0x%x differ from 0x%x (not used by the decision)", cpu_i, c.hw, c0->hw);
        if (c.l1.ecx != c0->l1.ecx || c.l7.ebx != c0->l7.ebx || c.l7.ecx != c0->l7.ecx)
            t_note("cpu %d: leaf1.ECX/leaf7 words differ (0x%08x/0x%08x/0x%08x vs 0x%08x/0x%08x/0x%08x)", cpu_i,
                   c.l1.ecx, c.l7.ebx, c.l7.ecx, c0->l1.ecx, c0->l7.ebx, c0->l7.ecx);
        /* the decision made once must be valid wherever the thread runs */
        sd_disp d; sd_disp_make(&d, TIER_AVX512, GATE_CORRECT, 0);
        uint64_t bad = sd_workload(&d, 0xC0 + (uint64_t)cpu_i, 60, 300);
        CHECKF(bad == 0, "cpu %d: %llu dispatched results differ from reference", cpu_i, (unsigned long long)bad);
        visited++;
    }
    t_unpin_thread(prev);
    t_info("cpus", "%d visited of %d (pin failures %d, not-on-pinned-cpu %d)", visited, ncpu, pinfail, elsewhere);
    if (elsewhere) t_note("GetCurrentProcessorNumber differed from the pinned CPU %d times", elsewhere);
    if (cpu_hw(F_HYBRID)) t_note("hybrid processor: decisions must be identical on every core type");
    CHECKF(visited >= 1, "could not pin to any CPU");
}

typedef struct sd_thr_det { const sd_caps *ref; volatile LONG *diff; HANDLE go; } sd_thr_det;
static DWORD WINAPI sd_thread_detect(LPVOID p) {
    sd_thr_det *t = (sd_thr_det *)p;
    WaitForSingleObject(t->go, 5000);
    for (int i = 0; i < 50; i++) {
        sd_caps c; sd_detect(&c, &sd_real, GATE_CORRECT, 0);
        if (c.usable != t->ref->usable || c.xcr0 != t->ref->xcr0) InterlockedIncrement(t->diff);
        if (i % 10 == 0) SwitchToThread();
    }
    return 0;
}
TEST(detect_identical_on_many_threads) {
    enum { NT = 32 };
    HANDLE h[NT]; int nt = 0;
    volatile LONG diff = 0;
    sd_thr_det t = { sd_truth(), &diff, CreateEventA(NULL, TRUE, FALSE, NULL) };
    REQUIRE(t.go);
    for (int i = 0; i < NT; i++) { h[nt] = CreateThread(NULL, 64 * 1024, sd_thread_detect, &t, 0, NULL); if (h[nt]) nt++; }
    if (nt < 2) { SetEvent(t.go); for (int i = 0; i < nt; i++) { WaitForSingleObject(h[i], 10000); CloseHandle(h[i]); } CloseHandle(t.go); SKIP("could not create threads"); }
    SetEvent(t.go);
    DWORD w = WaitForMultipleObjects((DWORD)nt, h, TRUE, 20000);
    CHECKF(w != WAIT_TIMEOUT, "detection threads did not finish");
    for (int i = 0; i < nt; i++) CloseHandle(h[i]);
    CloseHandle(t.go);
    CHECKF(diff == 0, "%ld of %d thread detections differed from the main thread", (long)diff, nt * 50);
    t_info("threads", "%d", nt);
}

TEST(detect_identical_in_child_process) {
    static t_child_result r;
    REQUIRE(t_child_run("decisions", NULL, 15000, &r));
    CHECK(!r.timed_out); CHECK_EQ(r.exit_code, 0); CHECK_EQ(r.exc_code, 0);
    const sd_caps *c = sd_truth();
    long long v;
    CHECK(sd_find_int(&r, "USABLE=", &v) && (unsigned)v == c->usable);
    CHECK(sd_find_int(&r, "HW=", &v) && (unsigned)v == c->hw);
    CHECK(sd_find_int(&r, "XCR0=", &v) && (uint64_t)v == c->xcr0);
    CHECK(sd_find_int(&r, "OSXSAVE=", &v) && v == c->osxsave);
    for (int k = 0; k < K_COUNT; k++) {
        char pfx[32]; snprintf(pfx, sizeof pfx, "SEL_%s=", sd_kname[k]);
        CHECKF(sd_find_int(&r, pfx, &v) && v == sd_select(k, c->usable, TIER_AVX512), "%s decision in child differs", sd_kname[k]);
    }
}

TEST(detect_vs_IsProcessorFeaturePresent) {
    const sd_caps *c = sd_truth();
    struct { DWORD pf; const char *name; unsigned bit; int ancient; } m[] = {
        { PF_XMMI_INSTRUCTIONS_AVAILABLE,   "PF_XMMI (sse)",     SD_SSE,     1 },
        { PF_XMMI64_INSTRUCTIONS_AVAILABLE, "PF_XMMI64 (sse2)",  SD_SSE2,    1 },
        { PF_SSSE3_INSTRUCTIONS_AVAILABLE,  "PF_SSSE3",          SD_SSSE3,   0 },
        { PF_SSE4_1_INSTRUCTIONS_AVAILABLE, "PF_SSE4_1",         SD_SSE41,   0 },
        { PF_SSE4_2_INSTRUCTIONS_AVAILABLE, "PF_SSE4_2",         SD_SSE42,   0 },
        { PF_AVX_INSTRUCTIONS_AVAILABLE,    "PF_AVX",            SD_AVX,     0 },
        { PF_AVX2_INSTRUCTIONS_AVAILABLE,   "PF_AVX2",           SD_AVX2,    0 },
        { PF_AVX512F_INSTRUCTIONS_AVAILABLE, "PF_AVX512F",       SD_AVX512F, 0 },
    };
    for (size_t i = 0; i < T_COUNTOF(m); i++) {
        int pf = IsProcessorFeaturePresent(m[i].pf) ? 1 : 0, u = !!(c->usable & m[i].bit);
        t_info(m[i].name, "pf=%d usable=%d", pf, u);
        /* An application trusting PF must never be told a feature works when it would #UD. */
        CHECKF(!pf || u, "%s reports present but the instructions are not usable (hw=%d osxsave=%d xcr0=0x%llx)",
               m[i].name, !!(c->hw & m[i].bit), c->osxsave, (unsigned long long)c->xcr0);
        if (m[i].ancient) CHECKF(pf == u, "%s=%d but dispatcher usable=%d", m[i].name, pf, u);
        else if (u && !pf) t_note("%s not reported although usable (older OS without this constant?): an app relying on it would under-dispatch", m[i].name);
    }
    int pfx = IsProcessorFeaturePresent(PF_XSAVE_ENABLED) ? 1 : 0;
    t_info("PF_XSAVE_ENABLED", "%d", pfx);
    CHECKF(!pfx || c->osxsave, "PF_XSAVE_ENABLED=1 but CPUID.1:ECX.OSXSAVE=0");
    if (c->osxsave && !pfx) t_note("OSXSAVE=1 but PF_XSAVE_ENABLED=0");
    int pf3 = IsProcessorFeaturePresent(PF_SSE3_INSTRUCTIONS_AVAILABLE) ? 1 : 0, sse3 = (int)(c->l1.ecx & 1);
    CHECKF(!pf3 || sse3, "PF_SSE3_INSTRUCTIONS_AVAILABLE=1 but CPUID.1:ECX[0]=0");
    if (sse3 && !pf3) t_note("SSE3 advertised but PF_SSE3 not reported");
    /* A PF-driven dispatcher (what many apps do) never selects more than the correct one. */
    unsigned pfmask = 0;
    for (size_t i = 0; i < T_COUNTOF(m); i++) if (IsProcessorFeaturePresent(m[i].pf)) pfmask |= m[i].bit;
    if (IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE)) pfmask |= c->usable & (SD_AVX512BW | SD_AVX512VL);
    pfmask |= c->usable & SD_POPCNT;   /* no PF constant for POPCNT */
    for (int k = 0; k < K_COUNT; k++) {
        int s = sd_select(k, pfmask, TIER_AVX512), s0 = sd_select(k, c->usable, TIER_AVX512);
        if (s >= 0) CHECKF(sd_usable_id(s), "PF-driven dispatch picks unusable %s", g_impl[s].name);
        if (sel_tier(s) < sel_tier(s0)) t_note("PF-driven dispatch picks %s instead of %s", sd_iname(s), sd_iname(s0));
    }
}

typedef DWORD64 (WINAPI *pfn_gef_t)(void);
typedef ULONG64 (NTAPI *pfn_rtl_gef_t)(ULONG64);

TEST(detect_vs_GetEnabledXStateFeatures) {
    pfn_gef_t gef = (pfn_gef_t)t_proc("kernel32.dll", "GetEnabledXStateFeatures");
    if (!gef) SKIP("GetEnabledXStateFeatures not exported");
    const sd_caps *c = sd_truth();
    uint64_t api = gef();
    t_info("api_xstate_features", "0x%llx", (unsigned long long)api);
    t_info("xcr0", "0x%llx", (unsigned long long)c->xcr0);
    CHECKF((api & 3) == 3, "GetEnabledXStateFeatures()=0x%llx lacks the legacy x87/SSE bits that are always enabled", (unsigned long long)api);
    CHECK_EQ(api, gef());   /* stable */
    if (t_env_get()->api_xstate_present) CHECK_EQ(api, t_env_get()->api_xstate_features);
    static const struct { uint64_t bit; const char *n; } ub[] = { { 0x4, "AVX" }, { 0x20, "OPMASK" }, { 0x40, "ZMM_Hi256" }, { 0x80, "Hi16_ZMM" } };
    int disagree = 0;
    for (size_t i = 0; i < T_COUNTOF(ub); i++) {
        int a = !!(api & ub[i].bit), x = c->osxsave && (c->xcr0 & ub[i].bit);
        if (a != x) { disagree++; t_note("XSTATE %s: GetEnabledXStateFeatures=%d but XCR0=%d", ub[i].n, a, x); }
    }
    /* What would an application deciding from the API pick? */
    sd_disp ad; sd_disp_make(&ad, TIER_AVX512, GATE_API, api);
    for (int k = 0; k < K_COUNT; k++) {
        int sa = ad.sel[k], s0 = sd_select(k, c->usable, TIER_AVX512);
        t_info(sd_kname[k], "dispatcher=%s api_driven=%s", sd_iname(s0), sd_iname(sa));
        if (sa == s0) continue;
        if (sel_tier(sa) < sel_tier(s0)) { t_note("API-driven dispatch under-selects %s: %s instead of %s (functional, slower)", sd_kname[k], sd_iname(sa), sd_iname(s0)); continue; }
        if (!sd_usable_id(sa)) {
            /* the API overclaims state: find out whether that actually breaks an application */
            static t_child_result r; char arg[16]; snprintf(arg, sizeof arg, "%d", sa);
            REQUIRE(t_child_run("run_impl", arg, 10000, &r));
            CHECKF(!r.timed_out && r.exc_code == 0,
                   "GetEnabledXStateFeatures=0x%llx claims state XCR0=0x%llx lacks; an app trusting it runs %s and dies with 0x%lx",
                   (unsigned long long)api, (unsigned long long)c->xcr0, g_impl[sa].name, (unsigned long)r.exc_code);
            if (!r.exc_code) t_note("%s ran without fault despite XCR0 not enabling its state", g_impl[sa].name);
        }
    }
    t_info("disagreeing_user_components", "%d", disagree);
    if (!disagree) CHECK(1);
}

TEST(detect_RtlGetEnabledExtendedFeatures_mask_filter) {
    pfn_rtl_gef_t rtl = (pfn_rtl_gef_t)t_proc("ntdll.dll", "RtlGetEnabledExtendedFeatures");
    if (!rtl) SKIP("RtlGetEnabledExtendedFeatures not exported");
    uint64_t all = rtl(~0ull);
    t_info("rtl_all", "0x%llx", (unsigned long long)all);
    static const uint64_t masks[] = { 0, 1, 2, 3, 4, 6, 7, 0x20, 0xE0, 0xE6, 0xE7, 0x200, 0x60000, 0x80000,
                                      0xFFFFFFFFull, 0xFFFFFFFF00000000ull, 0x8000000000000000ull, 0x5555555555555555ull };
    for (size_t i = 0; i < T_COUNTOF(masks); i++) {
        uint64_t v = rtl(masks[i]);
        CHECKF(v == (all & masks[i]), "RtlGetEnabledExtendedFeatures(0x%llx)=0x%llx, expected all&mask=0x%llx",
               (unsigned long long)masks[i], (unsigned long long)v, (unsigned long long)(all & masks[i]));
    }
    pfn_gef_t gef = (pfn_gef_t)t_proc("kernel32.dll", "GetEnabledXStateFeatures");
    if (gef) {
        uint64_t g = gef();
        if (g != (all ? all : 3)) t_note("GetEnabledXStateFeatures()=0x%llx differs from RtlGetEnabledExtendedFeatures(~0)=0x%llx", (unsigned long long)g, (unsigned long long)all);
    }
    const sd_caps *c = sd_truth();
    if (c->osxsave && ((all & 0xE6) != (c->xcr0 & 0xE6)))
        t_note("Rtl-reported vector state 0x%llx differs from XCR0 vector state 0x%llx", (unsigned long long)(all & 0xE6), (unsigned long long)(c->xcr0 & 0xE6));
}

/* Compare what an application deciding from mask `api` would pick with the correct dispatcher.
 * Under-selection is a NOTE (functional, slower). Over-selection of an implementation whose state
 * XCR0 lacks is run in a child: a FAIL only if it then really faults. Returns #over-selections. */
static int sd_compare_api_driven(const char *what, uint64_t api) {
    const sd_caps *c = sd_truth();
    sd_disp ad; sd_disp_make(&ad, TIER_AVX512, GATE_API, api);
    int over = 0;
    for (int k = 0; k < K_COUNT; k++) {
        int sa = ad.sel[k], s0 = sd_select(k, c->usable, TIER_AVX512);
        if (sa == s0) continue;
        if (sel_tier(sa) < sel_tier(s0)) { t_note("%s-driven dispatch under-selects %s: %s instead of %s (functional, slower)", what, sd_kname[k], sd_iname(sa), sd_iname(s0)); continue; }
        if (sd_usable_id(sa)) continue;
        over++;
        static t_child_result r; char arg[16]; snprintf(arg, sizeof arg, "%d", sa);
        if (!t_child_run("run_impl", arg, 10000, &r)) { CHECKF(0, "could not start run_impl child"); continue; }
        CHECKF(!r.timed_out && r.exc_code == 0,
               "%s reports state 0x%llx that XCR0=0x%llx lacks; an app trusting it runs %s and dies with 0x%lx",
               what, (unsigned long long)api, (unsigned long long)c->xcr0, g_impl[sa].name, (unsigned long)r.exc_code);
        if (!r.exc_code) t_note("%s ran without fault although XCR0 does not enable its state", g_impl[sa].name);
    }
    return over;
}

/* App-style detection through the CONTEXT API: "is there an AVX / AVX-512 area in an XSTATE context?"
 * kernelbase!LocateXStateFeature (19041 decomp): ids 0/1 -> legacy area; otherwise, when the context
 * has CONTEXT_XSTATE, ntdll!RtlLocateExtendedFeature2(.., &KUSER_SHARED_DATA.XState) returns NULL
 * unless id is in (EnabledFeatures | EnabledUserVisibleSupervisorFeatures), the same mask
 * RtlGetEnabledExtendedFeatures(~0) returns. So: located => GetEnabledXStateFeatures has the bit
 * (hard check). Agreement with XCR0 is a NOTE (guidance 10.3). */
TEST(detect_vs_context_xstate_api) {
    pfn_gef_t gef = (pfn_gef_t)t_proc("kernel32.dll", "GetEnabledXStateFeatures");
    __typeof__(&InitializeContext) ic = T_PROC("kernel32.dll", InitializeContext);
    __typeof__(&LocateXStateFeature) loc = T_PROC("kernel32.dll", LocateXStateFeature);
    if (!gef || !ic || !loc) SKIP("InitializeContext/LocateXStateFeature/GetEnabledXStateFeatures not exported");
    const sd_caps *c = sd_truth();
    uint64_t api = gef();
    if ((api & ~3ull) == 0) {
        t_note("GetEnabledXStateFeatures=0x%llx: no extended state reported, XSTATE contexts carry no AVX area", (unsigned long long)api);
        if (c->ymm_ok) t_note("but XCR0=0x%llx enables YMM state: a context-API-driven app under-selects", (unsigned long long)c->xcr0);
        SKIP("no extended XSTATE features enabled by the OS API");
    }
    DWORD flags = CONTEXT_ALL | CONTEXT_XSTATE, len = 0;
    SetLastError(0);
    BOOL ok = ic(NULL, flags, NULL, &len);
    DWORD err = GetLastError();
    /* documented: NULL Buffer -> FALSE, ERROR_INSUFFICIENT_BUFFER, *ContextLength = required size */
    CHECKF(!ok && err == ERROR_INSUFFICIENT_BUFFER, "InitializeContext(NULL) returned %d err=%lu", ok, (unsigned long)err);
    REQUIREF(len >= sizeof(CONTEXT) && len < (1u << 20), "InitializeContext required length %lu", (unsigned long)len);
    void *buf = malloc(len);
    REQUIRE(buf);
    PCONTEXT ctx = NULL; DWORD len2 = len;
    ok = ic(buf, flags, &ctx, &len2);
    if (!ok || !ctx) { DWORD e = GetLastError(); free(buf); REQUIREF(0, "InitializeContext(CONTEXT_ALL|CONTEXT_XSTATE) failed err=%lu", (unsigned long)e); }
    static const struct { DWORD id; const char *n; } fs[] = { { 2, "AVX" }, { 5, "OPMASK" }, { 6, "ZMM_Hi256" }, { 7, "Hi16_ZMM" } };
    uint64_t located = 0;
    for (size_t i = 0; i < T_COUNTOF(fs); i++) {
        DWORD flen = 0;
        void *p = loc(ctx, fs[i].id, &flen);
        uint64_t bit = 1ull << fs[i].id;
        int a = !!(api & bit), x = c->osxsave && (c->xcr0 & bit);
        t_info(fs[i].n, "located=%d len=%lu api=%d xcr0=%d", p != NULL, (unsigned long)flen, a, x);
        CHECKF(!p || a, "LocateXStateFeature(%s) found an area but GetEnabledXStateFeatures=0x%llx lacks the bit",
               fs[i].n, (unsigned long long)api);
        if (p) {
            located |= bit;
            CHECKF((uint8_t *)p >= (uint8_t *)ctx && (uint8_t *)p + flen <= (uint8_t *)buf + len,
                   "LocateXStateFeature(%s) area %p+%lu outside the context buffer", fs[i].n, p, (unsigned long)flen);
        }
        if (!!p != x) t_note("XSTATE %s: located in a CONTEXT_XSTATE context=%d but XCR0=%d", fs[i].n, p != NULL, x);
    }
    free(buf);
    t_info("context_located_mask", "0x%llx", (unsigned long long)located);
    int over = sd_compare_api_driven("LocateXStateFeature", located | 3);
    t_info("over_selections", "%d", over);
}

/* Windows 11 GetThreadEnabledXStateFeatures: optional API (guidance 10.2). Its value is recorded and
 * compared with XCR0 as NOTE only; an over-claim that would make an app fault is a FAIL. */
TEST(detect_vs_GetThreadEnabledXStateFeatures) {
    pfn_gef_t gte = (pfn_gef_t)t_proc("kernel32.dll", "GetThreadEnabledXStateFeatures");
    if (!gte) SKIP("GetThreadEnabledXStateFeatures not exported (pre-Windows 11 API set)");
    const sd_caps *c = sd_truth();
    uint64_t t = gte();
    t_info("thread_enabled_xstate", "0x%llx", (unsigned long long)t);
    CHECK_EQ(t, gte());
    pfn_gef_t gef = (pfn_gef_t)t_proc("kernel32.dll", "GetEnabledXStateFeatures");
    if (gef && (t & ~gef())) t_note("thread-enabled 0x%llx has bits outside GetEnabledXStateFeatures 0x%llx", (unsigned long long)t, (unsigned long long)gef());
    if (c->osxsave && (t & 0xE6) != (c->xcr0 & 0xE6))
        t_note("thread-enabled vector state 0x%llx differs from XCR0 vector state 0x%llx", (unsigned long long)(t & 0xE6), (unsigned long long)(c->xcr0 & 0xE6));
    sd_compare_api_driven("GetThreadEnabledXStateFeatures", t);
}

TEST(detect_naive_dispatchers_never_safer_than_correct) {
    const sd_caps *c = sd_truth();
    int danger = 0;
    for (int gate = GATE_NAIVE; gate <= GATE_YMM_FOR_ZMM; gate++) {
        sd_caps g; sd_detect(&g, &sd_real, gate, 0);
        CHECKF((c->usable & ~g.usable) == 0, "correct gate enables caps the %s gate does not (0x%x vs 0x%x)", sd_gate_name(gate), c->usable, g.usable);
        for (int k = 0; k < K_COUNT; k++) {
            int s = sd_select(k, g.usable, TIER_AVX512), s0 = sd_select(k, c->usable, TIER_AVX512);
            CHECKF(sel_tier(s) >= sel_tier(s0), "%s gate picks a lower tier for %s", sd_gate_name(gate), sd_kname[k]);
            if (s != s0 && !sd_usable_id(s)) {
                danger++;
                t_note("a %s dispatcher would run %s here, which is not usable (would #UD)", sd_gate_name(gate), g_impl[s].name);
            }
        }
    }
    t_info("unsafe_naive_picks", "%d", danger);
}

/* =============================================================================================
 * Cases: decision logic on synthetic CPUs
 * =========================================================================================== */
TEST(synth_no_osxsave_never_executes_xgetbv) {
    unsigned capsets[] = { SD_ALL_CAPS, SD_SSE | SD_SSE2 | SD_AVX, SD_SSE | SD_SSE2 | SD_AVX | SD_AVX2, SD_ALL_CAPS & ~SD_POPCNT };
    for (size_t i = 0; i < T_COUNTOF(capsets); i++) {
        sd_fake f; fake_make(&f, capsets[i], 0, 0xE7, 0xD);
        sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
        CHECK_EQ(f.xgetbv_calls, 0);
        CHECK_EQ(f.xgetbv_illegal, 0);
        CHECK_EQ(c.usable & (SD_YMM_CAPS | SD_ZMM_CAPS), 0);
        for (int k = 0; k < K_COUNT; k++) CHECKF(sel_tier(sel[k]) <= TIER_SSE4, "caps 0x%x no OSXSAVE: %s picked tier %d", capsets[i], sd_kname[k], sel_tier(sel[k]));
        fake_detect(&f, GATE_NAIVE, &c, sel, TIER_AVX512);
        CHECKF(sel_tier(sel[K_ADD_U32]) > TIER_SSE4 || !(capsets[i] & SD_AVX2), "naive gate should demonstrate the bug");
    }
}

TEST(synth_xcr0_without_ymm_state_disables_avx) {
    static const uint64_t x[] = { 0x1, 0x3, 0x5, 0x7 & ~0x4ull, 0x203, 0x60003, 0x1B };
    for (size_t i = 0; i < T_COUNTOF(x); i++) {
        sd_fake f; fake_make(&f, SD_ALL_CAPS, 1, x[i], 0xD);
        sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
        CHECK_EQ(f.xgetbv_calls, 1);
        CHECKF((c.usable & (SD_YMM_CAPS | SD_ZMM_CAPS)) == 0, "XCR0=0x%llx: VEX/EVEX caps usable", (unsigned long long)x[i]);
        for (int k = 0; k < K_COUNT; k++) CHECKF(sel_tier(sel[k]) <= TIER_SSE4, "XCR0=0x%llx: %s picked %s", (unsigned long long)x[i], sd_kname[k], sd_iname(sel[k]));
        CHECK_EQ(sel_tier(sel[K_CRC32C]), TIER_SSE4);
        CHECK_EQ(sel_tier(sel[K_ADLER32]), TIER_SSE4);
    }
}

TEST(synth_xcr0_ymm_only_disables_avx512) {
    static const uint64_t x[] = { 0x7, 0x27, 0x47, 0x87, 0x67, 0xA7, 0xC7, 0x207, 0x60007 };
    for (size_t i = 0; i < T_COUNTOF(x); i++) {
        sd_fake f; fake_make(&f, SD_ALL_CAPS, 1, x[i], 0xD);
        sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
        CHECKF((c.usable & SD_ZMM_CAPS) == 0, "XCR0=0x%llx: AVX-512 usable", (unsigned long long)x[i]);
        CHECKF((c.usable & SD_YMM_CAPS) == SD_YMM_CAPS, "XCR0=0x%llx: AVX not usable", (unsigned long long)x[i]);
        for (int k = 0; k < K_COUNT; k++) CHECKF(sel_tier(sel[k]) <= TIER_AVX2, "XCR0=0x%llx: %s picked %s", (unsigned long long)x[i], sd_kname[k], sd_iname(sel[k]));
        CHECK_EQ(sel[K_ADD_U32], I_ADD_U32_AVX2);
        CHECK_EQ(sel[K_MEMCPY], I_MEMCPY_AVX);
        /* the YMM_FOR_ZMM bug would pick AVX-512 here */
        int bsel[K_COUNT]; sd_caps bc; fake_detect(&f, GATE_YMM_FOR_ZMM, &bc, bsel, TIER_AVX512);
        CHECK_EQ(bsel[K_ADD_U32], I_ADD_U32_AVX512);
    }
}

TEST(synth_full_avx512_selects_top_tier) {
    sd_fake f; fake_make(&f, SD_ALL_CAPS, 1, 0xE7, 0xD);
    sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
    CHECK_EQ(c.usable, SD_ALL_CAPS);
    CHECK_EQ(sel[K_ADD_U32], I_ADD_U32_AVX512); CHECK_EQ(sel[K_ADD_F32], I_ADD_F32_AVX512);
    CHECK_EQ(sel[K_MEMCPY], I_MEMCPY_AVX512BW); CHECK_EQ(sel[K_CRC32C], I_CRC_SSE42);
    CHECK_EQ(sel[K_ADLER32], I_ADLER_AVX512BW); CHECK_EQ(sel[K_COUNT8], I_COUNT_AVX512BW);
    /* AVX-512F without BW: byte-granular kernels must stay on AVX2 */
    fake_make(&f, SD_ALL_CAPS & ~SD_AVX512BW, 1, 0xE7, 0xD);
    fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
    CHECK_EQ(sel[K_ADD_U32], I_ADD_U32_AVX512);
    CHECK_EQ(sel[K_MEMCPY], I_MEMCPY_AVX); CHECK_EQ(sel[K_ADLER32], I_ADLER_AVX2); CHECK_EQ(sel[K_COUNT8], I_COUNT_AVX2);
    /* XCR0 with extra components (PKRU, AMX, APX) does not matter */
    static const uint64_t extra[] = { 0x2E7, 0x602E7, 0xE02E7, 0xFF };
    for (size_t i = 0; i < T_COUNTOF(extra); i++) {
        fake_make(&f, SD_ALL_CAPS, 1, extra[i], 0xD);
        fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
        CHECK_EQ(sel[K_ADD_U32], I_ADD_U32_AVX512);
    }
}

TEST(synth_max_leaf_below_7_ignores_leaf7) {
    for (uint32_t ml = 1; ml < 7; ml++) {
        sd_fake f; fake_make(&f, SD_ALL_CAPS, 1, 0xE7, ml);
        sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
        CHECK_EQ(f.cpuid_out_of_range, 0);
        CHECK_EQ(c.hw & (SD_AVX2 | SD_ZMM_CAPS), 0);
        CHECK_EQ(sel[K_ADD_U32], I_ADD_U32_SSE2);   /* AVX present but add_u32 needs AVX2 */
        CHECK_EQ(sel[K_ADD_F32], I_ADD_F32_AVX);    /* AVX (leaf 1) still usable */
        CHECK_EQ(sel[K_MEMCPY], I_MEMCPY_AVX);
    }
    sd_fake f; fake_make(&f, SD_ALL_CAPS, 1, 0xE7, 0);   /* no leaf 1 at all */
    sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
    CHECK_EQ(f.cpuid_out_of_range, 0); CHECK_EQ(f.xgetbv_calls, 0);
    CHECK_EQ(c.hw, 0);
    CHECK_EQ(sel[K_ADD_U32], I_ADD_U32_SCALAR); CHECK_EQ(sel[K_ADD_F32], -1);
}

TEST(synth_partial_feature_sets) {
    struct { unsigned caps; uint64_t xcr0; int k; int want; const char *why; } t[] = {
        { SD_ALL_CAPS & ~SD_AVX,       0xE7, K_ADD_U32, I_ADD_U32_SSE2,   "AVX2 bit without AVX bit" },
        { SD_ALL_CAPS & ~SD_AVX2,      0xE7, K_ADD_U32, I_ADD_U32_SSE2,   "AVX-512F without AVX2" },
        { SD_ALL_CAPS & ~SD_AVX2,      0xE7, K_ADD_F32, I_ADD_F32_AVX,    "AVX-only kernel stays on AVX" },
        { SD_ALL_CAPS & ~SD_POPCNT,    0xE7, K_COUNT8,  I_COUNT_SSE2,     "count8 tiers need POPCNT" },
        { SD_ALL_CAPS & ~SD_POPCNT,    0xE7, K_CRC32C,  I_CRC_SSE42,      "CRC32 does not need POPCNT" },
        { SD_ALL_CAPS & ~SD_SSE42,     0xE7, K_CRC32C,  I_CRC_SCALAR,     "no SSE4.2 -> table CRC" },
        { SD_ALL_CAPS & ~SD_SSSE3,     0x3,  K_ADLER32, I_ADLER_SSE2,     "no SSSE3 -> SSE2 adler" },
        { SD_SSE | SD_SSE2,            0x3,  K_ADLER32, I_ADLER_SSE2,     "SSE2-only machine" },
        { SD_SSE | SD_SSE2,            0x3,  K_MEMCPY,  I_MEMCPY_SSE2,    "SSE2-only machine" },
        { SD_SSE,                      0x3,  K_ADD_U32, I_ADD_U32_SCALAR, "SSE without SSE2" },
        { SD_SSE,                      0x3,  K_ADD_F32, I_ADD_F32_SSE,    "SSE add_f32" },
        { 0,                           0x3,  K_ADD_F32, -1,               "no SSE: no add_f32 implementation" },
        { SD_ALL_CAPS & ~SD_AVX512F,   0xE7, K_MEMCPY,  I_MEMCPY_AVX,     "BW without F" },
        { SD_ALL_CAPS,                 0x7,  K_COUNT8,  I_COUNT_AVX2,     "XCR0 YMM only" },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        sd_fake f; fake_make(&f, t[i].caps, 1, t[i].xcr0, 0xD);
        sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
        CHECKF(sel[t[i].k] == t[i].want, "%s: %s picked %s, want %s", t[i].why, sd_kname[t[i].k],
               sd_iname(sel[t[i].k]), sd_iname(t[i].want));
    }
}

TEST(synth_exhaustive_selection_invariants) {
    static const uint64_t xs[] = { 0 /* no OSXSAVE */, 0x3, 0x7, 0x27, 0xE7, 0xE3 };
    unsigned long long combos = 0, bad_req = 0, bad_max = 0, bad_gate = 0, bad_mono = 0, bad_xg = 0;
    for (size_t xi = 0; xi < T_COUNTOF(xs); xi++) {
        for (unsigned caps = 0; caps <= SD_ALL_CAPS; caps++) {
            sd_fake f; fake_make(&f, caps, xs[xi] != 0, xs[xi], 0xD);
            sd_caps c; fake_detect(&f, GATE_CORRECT, &c, NULL, TIER_AVX512);
            int osx = xs[xi] != 0;
            unsigned want = caps & SD_LEGACY_CAPS;
            if (osx && (xs[xi] & 6) == 6) want |= caps & SD_YMM_CAPS;
            if (osx && (xs[xi] & 0xE6) == 0xE6) want |= caps & SD_ZMM_CAPS;
            if (c.usable != want) bad_gate++;
            if (f.xgetbv_illegal || f.xgetbv_calls != osx) bad_xg++;
            for (int k = 0; k < K_COUNT; k++) {
                int prev_tier = -1;
                for (int cap = TIER_SCALAR; cap <= TIER_AVX512; cap++) {
                    int s = sd_select(k, c.usable, cap);
                    combos++;
                    if (s >= 0 && ((g_impl[s].req & ~c.usable) || g_impl[s].tier > cap)) bad_req++;
                    for (int i = 0; i < I_N; i++)   /* nothing better was eligible */
                        if (g_impl[i].kernel == k && g_impl[i].tier <= cap && !(g_impl[i].req & ~c.usable) && g_impl[i].tier > sel_tier(s)) bad_max++;
                    if (sel_tier(s) < prev_tier) bad_mono++;
                    prev_tier = sel_tier(s);
                }
            }
        }
        CHECKF(bad_gate == 0, "XCR0=0x%llx: %llu capability sets gated wrongly", (unsigned long long)xs[xi], bad_gate);
        CHECKF(bad_xg == 0, "XCR0=0x%llx: XGETBV executed wrongly %llu times", (unsigned long long)xs[xi], bad_xg);
        CHECKF(bad_req == 0, "XCR0=0x%llx: %llu selections need missing caps or exceed the cap", (unsigned long long)xs[xi], bad_req);
        CHECKF(bad_max == 0, "XCR0=0x%llx: %llu selections were not the best eligible", (unsigned long long)xs[xi], bad_max);
        CHECKF(bad_mono == 0, "XCR0=0x%llx: %llu non-monotonic cap results", (unsigned long long)xs[xi], bad_mono);
    }
    t_info("combinations", "%llu", combos);
}

TEST(synth_xcr0_high_and_unrelated_bits_ignored) {
    /* Only XCR0[7:5,2:1] gate these tiers; bits in EDX (XCR0[63:32]) or other components must not matter. */
    static const struct { uint64_t x; int ymm, zmm; } t[] = {
        { 0x0000000100000007ull, 1, 0 }, { 0xFFFFFFFF00000003ull, 0, 0 }, { 0x80000000000000E7ull, 1, 1 },
        { 0x0000000000080207ull, 1, 0 }, { 0x00000000000602E3ull, 0, 0 }, { 0xFFFFFFFFFFFFFFFFull, 1, 1 },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        sd_fake f; fake_make(&f, SD_ALL_CAPS, 1, t[i].x, 0xD);
        sd_caps c; int sel[K_COUNT]; fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);
        CHECKF(c.ymm_ok == t[i].ymm && c.zmm_ok == t[i].zmm, "XCR0=0x%llx: ymm=%d zmm=%d, want %d %d",
               (unsigned long long)t[i].x, c.ymm_ok, c.zmm_ok, t[i].ymm, t[i].zmm);
        CHECK_EQ(sel_tier(sel[K_ADD_U32]), t[i].zmm ? TIER_AVX512 : t[i].ymm ? TIER_AVX2 : TIER_SSE2);
    }
}

TEST(synth_gate_bug_variants_demonstrated) {
    /* AVX hardware, OS with XSAVE enabled but YMM state off: only the correct gate is safe */
    sd_fake f; fake_make(&f, SD_ALL_CAPS, 1, 0x3, 0xD);
    int sel[K_COUNT]; sd_caps c;
    fake_detect(&f, GATE_CORRECT, &c, sel, TIER_AVX512);      CHECK_EQ(sel_tier(sel[K_ADD_U32]), TIER_SSE2);
    fake_detect(&f, GATE_NAIVE, &c, sel, TIER_AVX512);        CHECK_EQ(sel_tier(sel[K_ADD_U32]), TIER_AVX512);
    fake_detect(&f, GATE_OSXSAVE_ONLY, &c, sel, TIER_AVX512); CHECK_EQ(sel_tier(sel[K_ADD_U32]), TIER_AVX512);
    fake_detect(&f, GATE_YMM_FOR_ZMM, &c, sel, TIER_AVX512);  CHECK_EQ(sel_tier(sel[K_ADD_U32]), TIER_SSE2);
    /* API gate with a mask that lacks AVX */
    sd_src s = { fake_cpuid, fake_xgetbv, &f };
    f.xgetbv_calls = 0;
    sd_detect(&c, &s, GATE_API, 0x3);  CHECK_EQ(c.usable & SD_YMM_CAPS, 0);
    sd_detect(&c, &s, GATE_API, 0x7);  CHECK_EQ(c.usable & SD_YMM_CAPS, SD_YMM_CAPS); CHECK_EQ(c.usable & SD_ZMM_CAPS, 0);
    sd_detect(&c, &s, GATE_API, 0xE7); CHECK_EQ(c.usable & SD_ZMM_CAPS, SD_ZMM_CAPS);
    CHECK_EQ(f.xgetbv_calls, 0);   /* the API gate does not execute XGETBV */
}

/* =============================================================================================
 * Cases: dispatcher behaviour on the real machine
 * =========================================================================================== */
TEST(dispatch_table_complete_and_reported) {
    sd_disp *d = sd_global();
    const sd_caps *c = sd_truth();
    if (g_disp_force_invalid) t_note("%s has an invalid value; ignored", SD_ENV_FORCE);
    if (d->cap != TIER_AVX512) t_note("%s caps this run at %s", SD_ENV_FORCE, sd_tname[d->cap]);
    for (int k = 0; k < K_COUNT; k++) {
        int s = d->sel[k];
        REQUIREF(s >= 0, "no implementation for %s", sd_kname[k]);
        CHECK_EQ(g_impl[s].kernel, k);
        CHECKF(sd_impl_usable(s, c->usable), "%s selected but needs %s", g_impl[s].name, sd_req_str(s));
        CHECK(g_impl[s].fn != NULL);
        t_info(sd_kname[k], "%s (tier %s)", g_impl[s].name, sd_tname[g_impl[s].tier]);
    }
    /* table sanity: one scalar impl per kernel, no duplicate tiers */
    for (int k = 0; k < K_COUNT; k++) {
        int ns = 0, seen[TIER_COUNT] = { 0 };
        for (int i = 0; i < I_N; i++) if (g_impl[i].kernel == k) { seen[g_impl[i].tier]++; if (g_impl[i].tier == TIER_SCALAR) ns++; }
        CHECK_EQ(ns, 1);
        for (int t = 0; t < TIER_COUNT; t++) CHECKF(seen[t] <= 1, "%s has %d impls at tier %s", sd_kname[k], seen[t], sd_tname[t]);
    }
}

TEST(dispatch_selects_highest_usable_tier) {
    const sd_caps *c = sd_truth();
    sd_disp d; sd_disp_make(&d, TIER_AVX512, GATE_CORRECT, 0);
    for (int k = 0; k < K_COUNT; k++) {
        int best = -1;
        for (int i = 0; i < I_N; i++)
            if (g_impl[i].kernel == k && sd_impl_usable(i, c->usable) && (best < 0 || g_impl[i].tier > g_impl[best].tier)) best = i;
        CHECKF(d.sel[k] == best, "%s: dispatcher picked %s, best usable is %s", sd_kname[k], sd_iname(d.sel[k]), sd_iname(best));
    }
    CHECK_EQ(d.caps.usable, c->usable);
}

TEST(dispatch_cap_matrix_never_unusable) {
    const sd_caps *c = sd_truth();
    for (int cap = TIER_SCALAR; cap <= TIER_AVX512; cap++) {
        sd_disp d; sd_disp_make(&d, cap, GATE_CORRECT, 0);
        for (int k = 0; k < K_COUNT; k++) {
            int s = d.sel[k];
            if (k == K_ADD_F32 && !(c->usable & SD_SSE)) { CHECK_EQ(s, -1); continue; }
            REQUIREF(s >= 0, "cap %s: no impl for %s", sd_tname[cap], sd_kname[k]);
            CHECKF(g_impl[s].tier <= cap, "cap %s: %s exceeds the cap", sd_tname[cap], g_impl[s].name);
            CHECKF(sd_impl_usable(s, c->usable), "cap %s: %s not usable", sd_tname[cap], g_impl[s].name);
            CHECK_EQ(s, sd_select(k, c->usable, cap));
        }
        t_log("cap %s: %s %s %s %s %s %s", sd_tname[cap], sd_iname(d.sel[0]), sd_iname(d.sel[1]), sd_iname(d.sel[2]),
              sd_iname(d.sel[3]), sd_iname(d.sel[4]), sd_iname(d.sel[5]));
    }
}

TEST(dispatch_unselected_tiers_never_execute) {
    for (int cap = TIER_SCALAR; cap <= TIER_AVX512; cap++) {
        sd_disp d; sd_disp_make(&d, cap, GATE_CORRECT, 0);
        sd_calls_reset();
        uint64_t bad = sd_workload(&d, 0xD15 + (uint64_t)cap, 300, 600);
        CHECKF(bad == 0, "cap %s: %llu dispatched results wrong", sd_tname[cap], (unsigned long long)bad);
        for (int i = 0; i < I_N; i++) {
            int selected = 0;
            for (int k = 0; k < K_COUNT; k++) if (d.sel[k] == i) selected = 1;
            if (selected) CHECKF(g_calls[i] > 0, "cap %s: selected %s never ran", sd_tname[cap], g_impl[i].name);
            else CHECKF(g_calls[i] == 0, "cap %s: unselected %s ran %ld times", sd_tname[cap], g_impl[i].name, (long)g_calls[i]);
        }
    }
}

typedef struct sd_race { sd_disp *d; HANDLE go; int sel[K_COUNT]; } sd_race;
static DWORD WINAPI sd_race_thread(LPVOID p) {
    sd_race *r = (sd_race *)p;
    WaitForSingleObject(r->go, 5000);
    sd_disp *d = sd_get(r->d);
    for (int k = 0; k < K_COUNT; k++) r->sel[k] = d->sel[k];
    return 0;
}
TEST(dispatch_lazy_init_race) {
    enum { NT = 64 };
    static sd_race rs[NT]; static sd_disp d;
    HANDLE h[NT]; int nt = 0;
    sd_disp_prepare(&d, TIER_AVX512, GATE_CORRECT, 0);
    HANDLE go = CreateEventA(NULL, TRUE, FALSE, NULL);
    REQUIRE(go);
    for (int i = 0; i < NT; i++) {
        rs[i].d = &d; rs[i].go = go; memset(rs[i].sel, 0xFF, sizeof rs[i].sel);
        h[nt] = CreateThread(NULL, 64 * 1024, sd_race_thread, &rs[i], 0, NULL);
        if (h[nt]) nt++;
    }
    SetEvent(go);
    if (nt) CHECKF(WaitForMultipleObjects((DWORD)nt, h, TRUE, 20000) != WAIT_TIMEOUT, "init threads hung");
    for (int i = 0; i < nt; i++) CloseHandle(h[i]);
    CloseHandle(go);
    if (nt < 2) SKIP("could not create threads");
    CHECK_EQ(d.init_runs, 1);
    int diff = 0;
    for (int i = 0; i < nt; i++) for (int k = 0; k < K_COUNT; k++) if (rs[i].sel[k] != sd_select(k, sd_truth()->usable, TIER_AVX512)) diff++;
    CHECKF(diff == 0, "%d thread views of the dispatch table differ", diff);
    t_info("threads", "%d", nt);
}

typedef struct sd_conc { uint64_t seed; volatile LONG *bad; HANDLE go; } sd_conc;
static DWORD WINAPI sd_conc_thread(LPVOID p) {
    sd_conc *c = (sd_conc *)p;
    WaitForSingleObject(c->go, 5000);
    uint64_t b = sd_workload(sd_global(), c->seed, 240, 1024);
    if (b) InterlockedExchangeAdd(c->bad, (LONG)b);
    return 0;
}
TEST(dispatch_concurrent_use) {
    enum { NT = 32 };
    static sd_conc cs[NT];
    HANDLE h[NT]; int nt = 0;
    volatile LONG bad = 0;
    sd_global();
    HANDLE go = CreateEventA(NULL, TRUE, FALSE, NULL);
    REQUIRE(go);
    for (int i = 0; i < NT; i++) {
        cs[i].seed = 0xCAFE00 + (uint64_t)i; cs[i].bad = &bad; cs[i].go = go;
        h[nt] = CreateThread(NULL, 256 * 1024, sd_conc_thread, &cs[i], 0, NULL);
        if (h[nt]) nt++;
    }
    SetEvent(go);
    if (nt) CHECKF(WaitForMultipleObjects((DWORD)nt, h, TRUE, 30000) != WAIT_TIMEOUT, "worker threads hung");
    for (int i = 0; i < nt; i++) CloseHandle(h[i]);
    CloseHandle(go);
    if (nt < 2) SKIP("could not create threads");
    CHECKF(bad == 0, "%ld dispatched results differed from reference across %d threads", (long)bad, nt);
    t_info("threads", "%d", nt);
}

/* Thread-pool callbacks (OS-created threads the app does not own) use the dispatch table. */
static struct { volatile LONG bad, done; HANDLE ev; } g_tp;
static void CALLBACK sd_tp_cb(PTP_CALLBACK_INSTANCE inst, PVOID ctx) {
    (void)inst;
    uint64_t b = sd_workload(sd_global(), 0x7B00 + (uint64_t)(uintptr_t)ctx, 120, 512);
    if (b) InterlockedExchangeAdd(&g_tp.bad, (LONG)b);
    if (InterlockedIncrement(&g_tp.done) == 64) SetEvent(g_tp.ev);
}
TEST(dispatch_from_threadpool_callbacks) {
    __typeof__(&TrySubmitThreadpoolCallback) sub = T_PROC("kernel32.dll", TrySubmitThreadpoolCallback);
    if (!sub) SKIP("TrySubmitThreadpoolCallback not available");
    sd_global();
    g_tp.bad = 0; g_tp.done = 0;
    g_tp.ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    REQUIRE(g_tp.ev);
    int submitted = 0;
    for (int i = 0; i < 64; i++) if (sub(sd_tp_cb, (PVOID)(uintptr_t)i, NULL)) submitted++;
    if (submitted < 64) {   /* callbacks that did run still count; wait for them before leaving */
        for (int w = 0; w < 200 && g_tp.done < submitted; w++) Sleep(10);
        if (g_tp.done >= submitted) CloseHandle(g_tp.ev);
        SKIP("only %d of 64 thread-pool callbacks could be submitted", submitted);
    }
    DWORD w = WaitForSingleObject(g_tp.ev, 30000);
    CHECKF(w == WAIT_OBJECT_0, "thread-pool callbacks did not finish (%ld of 64 done)", (long)g_tp.done);
    CHECKF(g_tp.bad == 0, "%ld dispatched results in thread-pool callbacks differed from reference", (long)g_tp.bad);
    if (w == WAIT_OBJECT_0) CloseHandle(g_tp.ev);
}

/* --force=<tier> through a child process */
static void sd_force_case(const char *args, int expect_cap, const char *expect_src, int expect_invalid) {
    static t_child_result r;
    REQUIRE(t_child_run("forced", args, 30000, &r));
    CHECK(r.started);
    CHECKF(!r.timed_out, "forced child timed out");
    CHECKF(r.exc_code == 0, "forced child died with 0x%lx", (unsigned long)r.exc_code);
    CHECK_EQ(r.exit_code, 0);
    char src[32] = ""; t_child_find(&r, "FORCE_SOURCE=", src, sizeof src);
    CHECKF(!strcmp(src, expect_src), "force source '%s', expected '%s'", src, expect_src);
    long long v, honored = 0;
    CHECK(sd_find_int(&r, "FORCE_INVALID=", &v) && v == expect_invalid);
    CHECK(sd_find_int(&r, "CAP=", &v) && v == expect_cap);
    CHECK(sd_find_int(&r, "FORCE_HONORED=", &honored));
    const sd_caps *c = sd_truth();
    for (int k = 0; k < K_COUNT; k++) {
        char pfx[32]; snprintf(pfx, sizeof pfx, "SEL_%s=", sd_kname[k]);
        int ok = sd_find_int(&r, pfx, &v);
        CHECKF(ok, "missing %s", pfx);
        if (!ok) continue;
        int want = sd_select(k, c->usable, expect_cap);
        CHECKF(v == want, "%s: child picked %lld, expected %d (%s)", sd_kname[k], v, want, sd_iname(want));
        if (v >= 0 && v < I_N) {
            CHECKF(g_impl[v].tier <= expect_cap, "%s exceeds forced tier", g_impl[v].name);
            CHECKF(sd_usable_id((int)v), "forced child picked unusable %s", g_impl[v].name);
        }
    }
    CHECK(sd_find_int(&r, "VERIFY_MISMATCH=", &v) && v == 0);
    CHECK(sd_find_int(&r, "UNSELECTED_CALLS=", &v) && v == 0);
    CHECK(sd_find_int(&r, "SELECTED_CALLS=", &v) && v > 0);
    CHECK(sd_find_int(&r, "DONE=", &v) && v == 1);
    CHECK_EQ(honored, sd_tier_usable(expect_cap, c->usable));
    if (expect_invalid) t_note("invalid force value ignored: automatic selection used");
    else if (!honored) SKIP("tier %s is not usable here: verified the forced request fell back safely", sd_tname[expect_cap]);
}
TEST(force_cli_scalar) { sd_force_case("--force=scalar", TIER_SCALAR, "cli", 0); }
TEST(force_cli_sse2)   { sd_force_case("--force=sse2", TIER_SSE2, "cli", 0); }
TEST(force_cli_sse4)   { sd_force_case("--force=sse4", TIER_SSE4, "cli", 0); }
TEST(force_cli_avx2)   { sd_force_case("--force=avx2", TIER_AVX2, "cli", 0); }
TEST(force_cli_avx512) { sd_force_case("--force=avx512", TIER_AVX512, "cli", 0); }
TEST(force_cli_invalid_falls_back_to_auto) { sd_force_case("--force=avx9000", TIER_AVX512, "cli", 1); }
TEST(force_env_variable) {
    REQUIRE(SetEnvironmentVariableA(SD_ENV_FORCE, "sse2"));
    sd_force_case("", TIER_SSE2, "env", 0);
    SetEnvironmentVariableA(SD_ENV_FORCE, NULL);
}
TEST(force_cli_overrides_env) {
    REQUIRE(SetEnvironmentVariableA(SD_ENV_FORCE, "avx2"));
    sd_force_case("--force=scalar", TIER_SCALAR, "cli", 0);
    SetEnvironmentVariableA(SD_ENV_FORCE, NULL);
}
/* The env knob read by the process-wide dispatcher (sd_global) in a fresh process. */
static void sd_global_env_case(const char *val, int expect_cap, int expect_invalid) {
    static t_child_result r;
    REQUIRE(SetEnvironmentVariableA(SD_ENV_FORCE, val));
    int started = t_child_run("global", NULL, 30000, &r);
    SetEnvironmentVariableA(SD_ENV_FORCE, NULL);
    REQUIRE(started);
    CHECK(!r.timed_out); CHECK_EQ(r.exc_code, 0); CHECK_EQ(r.exit_code, 0);
    long long v = -1;
    CHECKF(sd_find_int(&r, "FORCE_INVALID=", &v) && v == expect_invalid, "%s=%s: invalid flag", SD_ENV_FORCE, val);
    CHECKF(sd_find_int(&r, "CAP=", &v) && v == expect_cap, "%s=%s: cap", SD_ENV_FORCE, val);
    const sd_caps *c = sd_truth();
    for (int k = 0; k < K_COUNT; k++) {
        char pfx[32]; snprintf(pfx, sizeof pfx, "SEL_%s=", sd_kname[k]);
        int want = sd_select(k, c->usable, expect_cap);
        CHECKF(sd_find_int(&r, pfx, &v) && v == want, "%s=%s: %s picked %lld, expected %s", SD_ENV_FORCE, val, sd_kname[k], v, sd_iname(want));
    }
    CHECK(sd_find_int(&r, "VERIFY_MISMATCH=", &v) && v == 0);
    CHECK(sd_find_int(&r, "UNSELECTED_CALLS=", &v) && v == 0);
    CHECK(sd_find_int(&r, "DONE=", &v) && v == 1);
}
TEST(force_env_global_dispatcher) {
    sd_global_env_case("scalar", TIER_SCALAR, 0);
    sd_global_env_case("SSE4", TIER_SSE4, 0);
    sd_global_env_case("avx2", TIER_AVX2, 0);
    sd_global_env_case("auto", TIER_AVX512, 0);
    sd_global_env_case("bogus-tier", TIER_AVX512, 1);
}
/* A whole run of this exe with the env knob: the dispatcher-level cases must still pass at that cap. */
TEST(force_env_whole_run) {
    static t_child_result r;
    REQUIRE(SetEnvironmentVariableA(SD_ENV_FORCE, "scalar"));
    int started = t_run_self("--filter=dispatch_table_complete_and_reported,dispatch_concurrent_use", 60000, &r);
    SetEnvironmentVariableA(SD_ENV_FORCE, NULL);
    REQUIRE(started);
    CHECK(!r.timed_out); CHECK_EQ(r.exc_code, 0);
    CHECKF(r.exit_code == 0, "run with %s=scalar exited %lu", SD_ENV_FORCE, (unsigned long)r.exit_code);
    CHECKF(strstr(r.out, "caps this run at scalar") != NULL, "the capped run did not report its cap");
    CHECKF(strstr(r.out, "add_u32_scalar") != NULL, "the capped run did not select add_u32_scalar");
    char v[256] = "";
    CHECKF(t_child_find(&r, "SUMMARY ", v, sizeof v) && strstr(v, "total=2 ") && strstr(v, " fail=0 "), "summary: %s", v);
}

TEST(force_parse_names) {
    CHECK_EQ(sd_parse_tier("scalar"), TIER_SCALAR); CHECK_EQ(sd_parse_tier("SSE2"), TIER_SSE2);
    CHECK_EQ(sd_parse_tier("sse4"), TIER_SSE4);     CHECK_EQ(sd_parse_tier("sse42"), TIER_SSE4);
    CHECK_EQ(sd_parse_tier("avx2"), TIER_AVX2);     CHECK_EQ(sd_parse_tier("AVX512"), TIER_AVX512);
    CHECK_EQ(sd_parse_tier("auto"), TIER_AVX512);   CHECK_EQ(sd_parse_tier(NULL), TIER_AVX512);
    CHECK_EQ(sd_parse_tier(""), TIER_AVX512);       CHECK_EQ(sd_parse_tier("avx9000"), -1);
    CHECK_EQ(sd_parse_tier("sse2 "), -1);
}

/* A dispatcher that checks only CPUID runs its pick in a child: it must #UD exactly when XCR0 lacks the state. */
TEST(misdetect_avx2_without_xcr0_check) {
    const sd_caps *c = sd_truth();
    if (!(c->hw & SD_AVX2) || !(c->hw & SD_AVX)) SKIP("CPU does not advertise AVX/AVX2: a CPUID-only dispatcher cannot mis-select AVX2");
    if (c->ymm_ok) SKIP("XCR0=0x%llx enables YMM state: CPUID-only detection happens to be right here (bug not observable)", (unsigned long long)c->xcr0);
    static t_child_result r;
    REQUIRE(t_child_run("misdetect", "avx2", 10000, &r));
    CHECK(!r.timed_out);
    char pick[64] = ""; t_child_find(&r, "PICK=", pick, sizeof pick);
    CHECKF(strstr(pick, "avx") != NULL, "naive dispatcher picked '%s'", pick);
    /* SDM: VEX-encoded instruction with XCR0[2:1] != 11b -> #UD -> STATUS_ILLEGAL_INSTRUCTION */
    CHECKF(r.exc_code == SD_STATUS_ILLEGAL, "expected STATUS_ILLEGAL_INSTRUCTION, child exc=0x%lx exit=0x%lx", (unsigned long)r.exc_code, (unsigned long)r.exit_code);
}
TEST(misdetect_avx512_without_zmm_state_check) {
    const sd_caps *c = sd_truth();
    if (!(c->hw & SD_AVX512F)) SKIP("CPU does not advertise AVX-512F");
    if ((c->hw & (SD_AVX | SD_AVX2)) != (SD_AVX | SD_AVX2))
        SKIP("AVX-512F advertised without AVX/AVX2: the add_u32 AVX-512 implementation (needs all three) is never picked");
    if (c->zmm_ok) SKIP("XCR0=0x%llx enables opmask/ZMM state: the YMM-only gate bug is not observable", (unsigned long long)c->xcr0);
    if (!c->ymm_ok) SKIP("YMM state not enabled either; covered by misdetect_avx2_without_xcr0_check");
    static t_child_result r;
    REQUIRE(t_child_run("misdetect", "avx512", 10000, &r));
    CHECK(!r.timed_out);
    char pick[64] = ""; t_child_find(&r, "PICK=", pick, sizeof pick);
    CHECKF(!strcmp(pick, "add_u32_avx512"), "YMM-only-gated dispatcher picked '%s'", pick);
    CHECKF(r.exc_code == SD_STATUS_ILLEGAL, "EVEX with XCR0[7:5] != 111b must #UD; child exc=0x%lx", (unsigned long)r.exc_code);
}
TEST(misdetect_control_correct_gate_survives) {
    /* same child with the correct table must run and survive: proves the harness path */
    static t_child_result r;
    char arg[16]; snprintf(arg, sizeof arg, "%d", sd_select(K_ADD_U32, sd_truth()->usable, TIER_AVX512));
    REQUIRE(t_child_run("run_impl", arg, 10000, &r));
    CHECK(!r.timed_out); CHECK_EQ(r.exc_code, 0); CHECK_EQ(r.exit_code, 0);
    char v[64]; CHECK(t_child_find(&r, "RESULT=", v, sizeof v));
}

/* =============================================================================================
 * Cases: per-implementation equivalence (0..300 sizes x 0..63 offsets x 5 patterns)
 * =========================================================================================== */
#define IMPL_CASE(name, id) TEST(name) { sd_impl_case(id); }
IMPL_CASE(impl_add_u32_scalar,    I_ADD_U32_SCALAR)
IMPL_CASE(impl_add_u32_sse2,      I_ADD_U32_SSE2)
IMPL_CASE(impl_add_u32_avx2,      I_ADD_U32_AVX2)
IMPL_CASE(impl_add_u32_avx512,    I_ADD_U32_AVX512)
IMPL_CASE(impl_add_f32_scalar,    I_ADD_F32_SCALAR)
IMPL_CASE(impl_add_f32_sse,       I_ADD_F32_SSE)
IMPL_CASE(impl_add_f32_avx,       I_ADD_F32_AVX)
IMPL_CASE(impl_add_f32_avx512,    I_ADD_F32_AVX512)
IMPL_CASE(impl_memcpy_scalar,     I_MEMCPY_SCALAR)
IMPL_CASE(impl_memcpy_sse2,       I_MEMCPY_SSE2)
IMPL_CASE(impl_memcpy_avx,        I_MEMCPY_AVX)
IMPL_CASE(impl_memcpy_avx512bw,   I_MEMCPY_AVX512BW)
IMPL_CASE(impl_crc32c_scalar,     I_CRC_SCALAR)
IMPL_CASE(impl_crc32c_sse42,      I_CRC_SSE42)
IMPL_CASE(impl_adler32_scalar,    I_ADLER_SCALAR)
IMPL_CASE(impl_adler32_sse2,      I_ADLER_SSE2)
IMPL_CASE(impl_adler32_ssse3,     I_ADLER_SSSE3)
IMPL_CASE(impl_adler32_avx2,      I_ADLER_AVX2)
IMPL_CASE(impl_adler32_avx512bw,  I_ADLER_AVX512BW)
IMPL_CASE(impl_count8_scalar,     I_COUNT_SCALAR)
IMPL_CASE(impl_count8_sse2,       I_COUNT_SSE2)
IMPL_CASE(impl_count8_popcnt,     I_COUNT_POPCNT)
IMPL_CASE(impl_count8_avx2,       I_COUNT_AVX2)
IMPL_CASE(impl_count8_avx512bw,   I_COUNT_AVX512BW)

/* =============================================================================================
 * Cases: known vectors, chaining, large inputs, state hygiene
 * =========================================================================================== */
TEST(vectors_crc32c_known_values) {
    uint8_t z[32], ff[32], up[32], dn[32];
    for (int i = 0; i < 32; i++) { z[i] = 0; ff[i] = 0xFF; up[i] = (uint8_t)i; dn[i] = (uint8_t)(31 - i); }
    struct { const void *p; size_t n; uint32_t want; const char *what; } v[] = {
        { "123456789", 9, 0xE3069283u, "\"123456789\" (check value)" },
        { z, 32, 0x8A9136AAu, "32 x 0x00 (RFC 3720 B.4)" }, { ff, 32, 0x62A8AB43u, "32 x 0xFF (RFC 3720 B.4)" },
        { up, 32, 0x46DD794Eu, "0..31 (RFC 3720 B.4)" },    { dn, 32, 0x113FDB5Cu, "31..0 (RFC 3720 B.4)" },
        { "", 0, 0, "empty" }, { "a", 1, 0xC1D04330u, "\"a\"" },
    };
    for (size_t j = 0; j < T_COUNTOF(v); j++) CHECKF(ref_crc32c(0, v[j].p, v[j].n) == v[j].want, "reference crc32c %s", v[j].what);
    int ran = 0;
    for (int id = 0; id < I_N; id++) {
        if (g_impl[id].kernel != K_CRC32C || !sd_usable_id(id)) continue;
        ran++;
        for (size_t j = 0; j < T_COUNTOF(v); j++) {
            uint32_t got = ((fn_crc)(void *)g_impl[id].fn)(0, v[j].p, v[j].n);
            CHECKF(got == v[j].want, "%s(%s) = 0x%08x, want 0x%08x", g_impl[id].name, v[j].what, got, v[j].want);
        }
    }
    t_info("impls", "%d", ran);
}

TEST(vectors_adler32_known_values) {
    static uint8_t ff[100000];
    memset(ff, 0xFF, sizeof ff);
    struct { const void *p; size_t n; uint32_t want; const char *what; } v[] = {
        { "", 0, 0x00000001u, "empty" }, { "a", 1, 0x00620062u, "\"a\"" }, { "abc", 3, 0x024D0127u, "\"abc\"" },
        { "Wikipedia", 9, 0x11E60398u, "\"Wikipedia\"" }, { "message digest", 14, 0x29750586u, "\"message digest\"" },
        { "abcdefghijklmnopqrstuvwxyz", 26, 0x90860B20u, "a..z" },
        { ff, sizeof ff, 0x149A302Cu, "100000 x 0xFF (modulo stress)" },
    };
    for (size_t j = 0; j < T_COUNTOF(v); j++) CHECKF(ref_adler32(1, v[j].p, v[j].n) == v[j].want, "reference adler32 %s", v[j].what);
    for (int id = 0; id < I_N; id++) {
        if (g_impl[id].kernel != K_ADLER32 || !sd_usable_id(id)) continue;
        for (size_t j = 0; j < T_COUNTOF(v); j++) {
            uint32_t got = ((fn_adler)(void *)g_impl[id].fn)(1, v[j].p, v[j].n);
            CHECKF(got == v[j].want, "%s(%s) = 0x%08x, want 0x%08x", g_impl[id].name, v[j].what, got, v[j].want);
        }
    }
}

static void sd_chain_case(int k) {
    static const size_t sizes[] = { 0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 300, 1000, 4096, 5552, 5553, 11105, 20000 };
    size_t maxn = 20000 + 64;
    uint8_t *buf = (uint8_t *)malloc(maxn);
    REQUIRE(buf);
    t_rng r; t_rng_seed(&r, 0xC4A1 + (uint64_t)k); t_rng_fill(&r, buf, maxn);
    uint32_t init = k == K_ADLER32 ? 1 : 0;
    int ids[I_N], nid = 0;
    for (int id = 0; id < I_N; id++) if (g_impl[id].kernel == k && sd_usable_id(id)) ids[nid++] = id;
    for (size_t si = 0; si < T_COUNTOF(sizes); si++) {
        size_t n = sizes[si];
        const uint8_t *p = buf + (si * 5) % 64;
        uint32_t want = (uint32_t)sd_run(k, g_ref[k], NULL, p, NULL, n, init);
        size_t splits[] = { 0, 1, n / 3, n / 2, n > 0 ? n - 1 : 0, n };
        unsigned long bad = 0, tried = 0;
        for (int i = 0; i < nid; i++) for (int j = 0; j < nid; j++) for (size_t s = 0; s < T_COUNTOF(splits); s++) {
            size_t sp = splits[s] > n ? n : splits[s];
            uint32_t part = (uint32_t)sd_run(k, g_impl[ids[i]].fn, NULL, p, NULL, sp, init);
            uint32_t full = (uint32_t)sd_run(k, g_impl[ids[j]].fn, NULL, p + sp, NULL, n - sp, part);
            tried++;
            if (full != want) { if (!bad) t_note("n=%u split=%u: %s then %s gave 0x%08x want 0x%08x", (unsigned)n, (unsigned)sp, g_impl[ids[i]].name, g_impl[ids[j]].name, full, want); bad++; }
        }
        CHECKF(bad == 0, "%s n=%u: %lu of %lu cross-implementation continuations wrong", sd_kname[k], (unsigned)n, bad, tried);
    }
    free(buf);
    t_info("impls", "%d", nid);
}
TEST(chain_crc32c_across_implementations)  { sd_chain_case(K_CRC32C); }
TEST(chain_adler32_across_implementations) { sd_chain_case(K_ADLER32); }

TEST(large_random_buffers_all_impls) {
    const size_t maxn = 256 * 1024;
    size_t cap = (maxn + 128) * 4 + 64;
    uint8_t *a = (uint8_t *)malloc(cap), *b = (uint8_t *)malloc(cap), *g = (uint8_t *)malloc(cap), *rf = (uint8_t *)malloc(cap);
    if (!a || !b || !g || !rf) { free(a); free(b); free(g); free(rf); SKIP("cannot allocate %u bytes", (unsigned)cap); }
    t_rng r; t_rng_seed(&r, 0x1A26E); t_rng_fill(&r, a, cap); t_rng_fill(&r, b, cap);
    for (int id = 0; id < I_N; id++) {
        if (!sd_usable_id(id)) continue;
        int k = g_impl[id].kernel; size_t es = sd_es(k);
        unsigned bad = 0;
        for (int it = 0; it < 12; it++) {
            size_t n = (size_t)t_rng_range(&r, maxn + 1) / es;
            if (it == 0) n = maxn / es; else if (it == 1) n = 5552 * 3 + 17;
            int oa = (int)t_rng_range(&r, 64), ob = (int)t_rng_range(&r, 64), od = (int)t_rng_range(&r, 64);
            uint32_t param = sd_param(k, n, it);
            size_t span = (n + 64) * es + 64;
            memset(g, 0xCD, span); memset(rf, 0xCD, span);
            uint64_t rg = sd_run(k, g_impl[id].fn, g + od * es, a + oa * es, b + ob * es, n, param);
            uint64_t rr = sd_run(k, g_ref[k], rf + od * es, a + oa * es, b + ob * es, n, param);
            if (rg != rr || memcmp(g, rf, span)) { if (!bad) t_note("%s n=%u: result 0x%llx ref 0x%llx", g_impl[id].name, (unsigned)n, (unsigned long long)rg, (unsigned long long)rr); bad++; }
        }
        CHECKF(bad == 0, "%s: %u of 12 large random buffers differ", g_impl[id].name, bad);
    }
    free(a); free(b); free(g); free(rf);
}

TEST(adler32_saturated_input_modulo) {
    const size_t n = 1u << 20;
    uint8_t *p = (uint8_t *)malloc(n + 64);
    REQUIRE(p);
    memset(p, 0xFF, n + 64);
    uint32_t want = ref_adler32(1, p, n), want_st = ref_adler32(0xFFF0FFF0u, p, n);
    for (int id = 0; id < I_N; id++) {
        if (g_impl[id].kernel != K_ADLER32 || !sd_usable_id(id)) continue;
        for (int off = 0; off < 64; off += 21) {
            CHECKF(((fn_adler)(void *)g_impl[id].fn)(1, p + off, n) == want, "%s 1 MiB of 0xFF off=%d", g_impl[id].name, off);
            CHECKF(((fn_adler)(void *)g_impl[id].fn)(0xFFF0FFF0u, p + off, n) == want_st, "%s max state, 1 MiB of 0xFF off=%d", g_impl[id].name, off);
        }
    }
    free(p);
}

TEST(count8_all_byte_values) {
    uint8_t buf[700];
    t_rng r; t_rng_seed(&r, 0xC0C0);
    t_rng_fill(&r, buf, sizeof buf);
    for (int id = 0; id < I_N; id++) {
        if (g_impl[id].kernel != K_COUNT8 || !sd_usable_id(id)) continue;
        unsigned bad = 0;
        for (int v = 0; v < 256; v++)
            for (size_t n = 0; n <= 640; n += 97)
                if (((fn_count)(void *)g_impl[id].fn)(buf + (v & 31), n, (uint8_t)v) != ref_count8(buf + (v & 31), n, (uint8_t)v)) bad++;
        memset(buf, 0x77, sizeof buf);   /* every byte matches: 255-iteration accumulator boundary in SSE2 */
        size_t all = ((fn_count)(void *)g_impl[id].fn)(buf, sizeof buf, 0x77);
        CHECKF(all == sizeof buf, "%s all-match count %u", g_impl[id].name, (unsigned)all);
        t_rng_fill(&r, buf, sizeof buf);
        CHECKF(bad == 0, "%s: %u (value,size) counts wrong", g_impl[id].name, bad);
    }
    uint8_t *big = (uint8_t *)malloc(70000);   /* > 255*16 bytes all matching: SSE2 accumulator flush */
    REQUIRE(big);
    memset(big, 0, 70000);
    for (int id = 0; id < I_N; id++)
        if (g_impl[id].kernel == K_COUNT8 && sd_usable_id(id))
            CHECKF(((fn_count)(void *)g_impl[id].fn)(big, 70000, 0) == 70000, "%s 70000 zero bytes", g_impl[id].name);
    free(big);
}

TEST(vectors_add_f32_nan_propagation) {
    /* SDM Vol.1 Table 4-7 (SSE/AVX): src1 NaN -> src1 quieted; else src2 NaN -> src2 quieted. */
    static const struct { uint32_t a, b, want; const char *what; } v[] = {
        { 0x7FC00011u, 0x7FC00022u, 0x7FC00011u, "QNaN + QNaN -> src1" },
        { 0x7F800011u, 0x7FC00022u, 0x7FC00011u, "SNaN + QNaN -> src1 quieted" },
        { 0x7FC00011u, 0x7F800022u, 0x7FC00011u, "QNaN + SNaN -> src1" },
        { 0xFF800033u, 0x7F800022u, 0xFFC00033u, "SNaN + SNaN -> src1 quieted" },
        { 0x3F800000u, 0x7F800022u, 0x7FC00022u, "1 + SNaN -> src2 quieted" },
        { 0x7F800000u, 0xFFC00044u, 0xFFC00044u, "inf + QNaN -> src2" },
        { 0x7F800000u, 0xFF800000u, 0xFFC00000u, "inf + -inf -> default QNaN" },
        { 0x80000000u, 0x80000000u, 0x80000000u, "-0 + -0 -> -0" },
        { 0x80000000u, 0x00000000u, 0x00000000u, "-0 + +0 -> +0 (RN)" },
    };
    enum { N = 37 };
    float a[N], b[N], d[N];
    int ran = 0;
    for (int id = 0; id < I_N; id++) {
        if (g_impl[id].kernel != K_ADD_F32 || !sd_usable_id(id)) continue;
        ran++;
        for (size_t j = 0; j < T_COUNTOF(v); j++) {
            for (int i = 0; i < N; i++) { memcpy(&a[i], &v[j].a, 4); memcpy(&b[i], &v[j].b, 4); }
            memset(d, 0xCD, sizeof d);
            ((fn_add_f32)(void *)g_impl[id].fn)(d, a, b, N);
            int bad = 0; uint32_t first = 0;
            for (int i = 0; i < N; i++) { uint32_t g; memcpy(&g, &d[i], 4); if (g != v[j].want) { if (!bad) first = g; bad++; } }
            CHECKF(bad == 0, "%s %s: %d of %d lanes wrong (first 0x%08x, want 0x%08x)", g_impl[id].name, v[j].what, bad, N, first, v[j].want);
        }
    }
    t_info("impls", "%d", ran);
}

TEST(add_f32_under_nondefault_mxcsr) {
    /* All tiers use the same MXCSR-controlled IEEE add: RC, FTZ and DAZ must give identical bits. */
    static const unsigned modes[] = { 0x1F80, 0x3F80, 0x5F80, 0x7F80, 0x9FC0, 0xFFC0 & ~0x3Fu };
    const size_t n = 300;
    float *a = (float *)_aligned_malloc((n + 64) * 4, 64), *b = (float *)_aligned_malloc((n + 64) * 4, 64);
    float *g = (float *)_aligned_malloc((n + 64) * 4, 64), *rf = (float *)_aligned_malloc((n + 64) * 4, 64);
    REQUIRE(a && b && g && rf);
    sd_fill(K_ADD_F32, 4, (uint8_t *)a, (n + 64) * 4, 11);
    sd_fill(K_ADD_F32, 4, (uint8_t *)b, (n + 64) * 4, 12);
    /* MXCSR_MASK (FXSAVE bytes 28..31; 0 means the default 0xFFBF): setting a bit outside it makes
     * LDMXCSR #GP (SDM Vol.1 11.6.6). DAZ (bit 6) is not architecturally guaranteed. */
    static uint8_t fx[512] T_ALIGN(64);
    t_fxsave(fx);
    uint32_t mmask; memcpy(&mmask, fx + 28, 4);
    if (!mmask) mmask = 0xFFBF;
    t_info("mxcsr_mask", "0x%x", mmask);
    unsigned saved = t_get_mxcsr();
    int tested = 0;
    for (size_t m = 0; m < T_COUNTOF(modes); m++) {
        if (modes[m] & ~mmask) { t_note("MXCSR mode 0x%04x skipped: bits outside MXCSR_MASK 0x%x", modes[m], mmask); continue; }
        tested++;
        t_set_mxcsr(modes[m]);
        for (int id = 0; id < I_N; id++) {
            if (g_impl[id].kernel != K_ADD_F32 || !sd_usable_id(id)) continue;
            unsigned bad = 0;
            for (size_t len = 0; len <= n; len += 13) for (int off = 0; off < 64; off += 9) {
                memset(g, 0xCD, (n + 64) * 4); memset(rf, 0xCD, (n + 64) * 4);
                ((fn_add_f32)(void *)g_impl[id].fn)(g + off, a + off, b + (off ^ 5), len);
                ref_add_f32(rf + off, a + off, b + (off ^ 5), len);
                if (memcmp(g, rf, (n + 64) * 4)) bad++;
            }
            unsigned now = t_get_mxcsr();
            CHECKF((now & ~0x3Fu) == (modes[m] & ~0x3Fu), "%s changed MXCSR control bits: 0x%x -> 0x%x", g_impl[id].name, modes[m], now);
            CHECKF(bad == 0, "%s under MXCSR=0x%04x: %u mismatches", g_impl[id].name, modes[m], bad);
        }
    }
    t_set_mxcsr(saved);
    _aligned_free(a); _aligned_free(b); _aligned_free(g); _aligned_free(rf);
    CHECKF(tested >= 4, "only %d MXCSR modes were testable (mask 0x%x)", tested, mmask);
}

TEST(impls_preserve_fp_control_state) {
    unsigned short cw = t_get_x87_cw();
    unsigned mx = t_get_mxcsr();
    static uint8_t a[1200] T_ALIGN(64), b[1200] T_ALIGN(64), d[1200] T_ALIGN(64);
    for (int i = 0; i < 1200; i++) { a[i] = (uint8_t)(i * 13); b[i] = (uint8_t)(i * 29 + 1); }
    for (int id = 0; id < I_N; id++) {
        if (!sd_usable_id(id)) continue;
        sd_run(g_impl[id].kernel, g_impl[id].fn, d, a, b, 257, 1);
        CHECKF(t_get_x87_cw() == cw, "%s changed the x87 control word", g_impl[id].name);
        CHECKF((t_get_mxcsr() & ~0x3Fu) == (mx & ~0x3Fu), "%s changed MXCSR control bits", g_impl[id].name);
    }
}

TEST(sse_avx_transitions_between_tiers) {
    /* Alternate legacy-SSE and VEX/EVEX implementations on the same data: each result must match,
     * no matter which encoding ran last (the AVX code must not leave dirty upper state that breaks SSE). */
    const sd_caps *c = sd_truth();
    if (!(c->usable & SD_AVX)) SKIP("no VEX tier usable: nothing to transition between");
    static uint8_t a[2048] T_ALIGN(64), b[2048] T_ALIGN(64), g[2048] T_ALIGN(64), rf[2048] T_ALIGN(64);
    t_rng r; t_rng_seed(&r, 0x7A115); t_rng_fill(&r, a, sizeof a); t_rng_fill(&r, b, sizeof b);
    int ids[I_N], nid = 0;
    for (int id = 0; id < I_N; id++) if (sd_usable_id(id)) ids[nid++] = id;
    unsigned bad = 0, runs = 0;
    for (int round = 0; round < 400; round++) {
        int id = ids[t_rng_range(&r, (uint64_t)nid)];
        int k = g_impl[id].kernel; size_t es = sd_es(k);
        size_t n = (size_t)t_rng_range(&r, 400 / es + 1);
        memset(g, 0xCD, sizeof g); memset(rf, 0xCD, sizeof rf);
        uint64_t x = sd_run(k, g_impl[id].fn, g + 3 * es, a + 1 * es, b + 2 * es, n, 1);
        uint64_t y = sd_run(k, g_ref[k], rf + 3 * es, a + 1 * es, b + 2 * es, n, 1);
        runs++;
        if (x != y || memcmp(g, rf, sizeof g)) bad++;
        if (round % 50 == 0) Sleep(0);   /* also cross a scheduling point with dirty-or-clean upper state */
    }
    CHECKF(bad == 0, "%u of %u interleaved runs mismatched", bad, runs);
    t_info("interleaved_runs", "%u", runs);
}

static void sd_run_all_cb(void *arg) {
    volatile unsigned *bad = (volatile unsigned *)arg;
    static uint8_t a[1300] T_ALIGN(64), b[1300] T_ALIGN(64), g[1300] T_ALIGN(64), rf[1300] T_ALIGN(64);
    for (int i = 0; i < 1300; i++) { a[i] = (uint8_t)(i * 5 + 1); b[i] = (uint8_t)(i * 11 + 7); }
    for (int id = 0; id < I_N; id++) {
        if (!sd_usable_id(id)) continue;
        int k = g_impl[id].kernel;
        memset(g, 0, sizeof g); memset(rf, 0, sizeof rf);
        if (sd_run(k, g_impl[id].fn, g, a, b, 290, 1) != sd_run(k, g_ref[k], rf, a, b, 290, 1) || memcmp(g, rf, sizeof g)) (*bad)++;
    }
}
TEST(impls_preserve_nonvolatile_xmm) {
#ifdef T_X64
    t_vregs in, out;
    volatile unsigned bad = 0;
    t_vregs_pattern(&in, T_VK_XMM, 0x5D);
    REQUIRE(t_vregs_call(T_VK_XMM, &in, &out, sd_run_all_cb, (void *)&bad) == 0);
    /* Windows x64 ABI: XMM6-XMM15 (low 128 bits) are callee-saved */
    t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "after every usable SIMD implementation");
    CHECK_EQ(bad, 0);
#else
    SKIP("x86 ABI: every XMM register is volatile across calls");
#endif
}

TEST(guard_page_tail_handling) {
    static t_child_result r;
    REQUIRE(t_child_run("guard_tail", NULL, 60000, &r));
    CHECKF(!r.timed_out, "guard child timed out");
    if (r.exc_code) {
        const char *last = NULL, *p = r.out;
        while ((p = strstr(p, "BEGIN ")) != NULL) { last = p + 6; p += 6; }
        char who[64] = "?";
        if (last) { size_t i = 0; while (last[i] && last[i] != '\r' && last[i] != '\n' && i < sizeof who - 1) { who[i] = last[i]; i++; } who[i] = 0; }
        CHECKF(0, "guard-page child died with 0x%lx at 0x%llx while running %s (over-read/write past the buffer or masked tail faulted)",
               (unsigned long)r.exc_code, r.exc_addr, who);
    }
    CHECK_EQ(r.exit_code, 0);
    int n = 0;
    for (int id = 0; id < I_N; id++) {
        if (!sd_usable_id(id)) continue;
        char pfx[64]; snprintf(pfx, sizeof pfx, "OK %s mism=", g_impl[id].name);
        long long v = -1;
        CHECKF(sd_find_int(&r, pfx, &v) && v == 0, "%s at page boundaries: %lld mismatches (or did not finish)", g_impl[id].name, v);
        n++;
    }
    long long done = -1;
    CHECK(sd_find_int(&r, "DONE=", &done) && done == n);
    if (!(sd_truth()->usable & SD_AVX512BW)) t_note("AVX-512 masked-tail fault suppression not exercised (no usable AVX-512BW)");
}

/* =============================================================================================
 * PERF: throughput per tier (verified first)
 * =========================================================================================== */
typedef struct sd_bctx { int id; uint8_t *a, *b, *d; size_t n; } sd_bctx;
static void sd_bench_fn(void *p) {
    sd_bctx *c = (sd_bctx *)p;
    T_SINK(sd_run(g_impl[c->id].kernel, g_impl[c->id].fn, c->d, c->a, c->b, c->n, 1));
}
TEST(perf_tiers_throughput) {
    const size_t bytes = 64 * 1024;
    uint8_t *a = (uint8_t *)_aligned_malloc(bytes + 64, 64), *b = (uint8_t *)_aligned_malloc(bytes + 64, 64);
    uint8_t *d = (uint8_t *)_aligned_malloc(bytes + 64, 64), *rf = (uint8_t *)_aligned_malloc(bytes + 64, 64);
    REQUIRE(a && b && d && rf);
    t_rng r; t_rng_seed(&r, 0xBE7C); t_rng_fill(&r, a, bytes + 64); t_rng_fill(&r, b, bytes + 64);
    for (int id = 0; id < I_N; id++) {
        if (!sd_usable_id(id)) continue;
        int k = g_impl[id].kernel;
        if (k == K_ADD_F32) continue;   /* same shape as add_u32 */
        sd_bctx c = { id, a, b, d, bytes / sd_es(k) };
        uint64_t x = sd_run(k, g_impl[id].fn, d, a, b, c.n, 1), y = sd_run(k, g_ref[k], rf, a, b, c.n, 1);
        int ok = x == y && (k == K_CRC32C || k == K_ADLER32 || k == K_COUNT8 || !memcmp(d, rf, bytes));
        CHECKF(ok, "%s wrong on the benchmark input", g_impl[id].name);
        if (!ok) continue;
        t_stats s;
        t_bench(sd_bench_fn, &c, 2, 7, 2.0, &s);
        if (s.median > 0) { char m[64]; snprintf(m, sizeof m, "%s_GBps", g_impl[id].name); t_perf(m, (double)bytes / s.median / 1e9, "GB/s"); }
    }
    _aligned_free(a); _aligned_free(b); _aligned_free(d); _aligned_free(rf);
}
