/* sse_arithmetic - SSE, SSE2, SSE3, SSSE3, SSE4.1 and SSE4.2 arithmetic, integer, conversion,
 * shuffle and string instructions, checked bit-exactly against scalar C reference models.
 *
 * Contract under test: the architectural instruction semantics of Intel SDM Vol.2 (instruction
 * pages) and Vol.1 ch.4/10/11 (MXCSR, NaN rules Table 4-7, rounding, DAZ/FTZ, exception flags),
 * which AMD APM Vol.4 describes identically for these instructions. The OS part of the contract is
 * that an ordinary Windows process can use every instruction that CPUID advertises and gets the
 * documented results and MXCSR flags; no Windows API is involved beyond process start-up.
 *
 * Method:
 *   - every instruction runs in inline asm (the exact encoding under test, never a compiler-chosen
 *     substitute); inputs pass through registers chosen by the compiler, so no constant folding;
 *   - the references are integer-only C models: tests/sse_arithmetic_softfp.h for IEEE binary32/64
 *     with x86 NaN/rounding/flag rules, plain C for integer, shuffle, string and CRC operations;
 *   - inputs: biased random vectors (min/max integers, signs, saturation edges, +-0, +-inf, QNaN,
 *     SNaN, denormals, rounding ties, near-overflow/underflow, cancellation pairs) plus immediate
 *     sweeps and hand-written known-answer vectors (independent of the models);
 *   - floating-point ops run under all four MXCSR rounding modes and DAZ, and the six MXCSR status
 *     flags are compared too (only where the SDM pins them; see softfp.h for the uncertain cases);
 *   - RCPPS/RSQRTPS are approximations: checked only against the documented relative error bound
 *     |rel err| <= 1.5 * 2^-12 and the documented special-value table.
 * Each case is gated by CPUID (cpu_has); a feature that is absent is a SKIP, a wrong result is a FAIL.
 * 64-bit-only forms (REX.W GPR operands) are covered on x64 and noted as not applicable on x86.
 * Deterministic: fixed seeds. Each vector sweep reports at most 6 detailed mismatches and one
 * summary CHECK; known-answer vectors are one CHECK each. */
#include "t.h"
#include <emmintrin.h>
#include "sse_arithmetic_softfp.h"

typedef union {
    uint8_t b[16]; int8_t sb[16]; uint16_t w[8]; int16_t sw[8];
    uint32_t d[4]; int32_t sd[4]; uint64_t q[2]; int64_t sq[2]; __m128i v;
} V128;

#define NEED(f) do { if (!cpu_has(f)) SKIP("%s not usable on this CPU", cpu_feature_name(f)); } while (0)

/* ---------------------------------------------------------------- reporting */
static int g_vec, g_bad;
static long g_total_vec;
#define VFAIL_MAX 6
static void vbegin(void) { g_vec = g_bad = 0; }
#define VCHECK(ok, ...) do { g_vec++; g_total_vec++; if (!(ok)) { if (g_bad++ < VFAIL_MAX) CHECKF(0, __VA_ARGS__); } } while (0)
static void vend_(const char *file, int line, const char *label) {
    t_check(g_bad == 0, file, line, label, "%s: %d of %d vectors mismatched", label, g_bad, g_vec);
}
#define VEND(label) vend_(__FILE__, __LINE__, label)

static char g_hb[16][72];
static int g_hi;
/* lanes printed highest first, like a debugger */
static const char *hx(const void *p, int lane) {
    char *b = g_hb[g_hi++ & 15];
    t_hex_lanes(b, sizeof g_hb[0], p, 16, lane);
    return b;
}

/* ---------------------------------------------------------------- immediate sweeps */
#define I16_(X, h) X(0x##h##0) X(0x##h##1) X(0x##h##2) X(0x##h##3) X(0x##h##4) X(0x##h##5) X(0x##h##6) X(0x##h##7) \
                   X(0x##h##8) X(0x##h##9) X(0x##h##A) X(0x##h##B) X(0x##h##C) X(0x##h##D) X(0x##h##E) X(0x##h##F)
#define I256(X) I16_(X, 0) I16_(X, 1) I16_(X, 2) I16_(X, 3) I16_(X, 4) I16_(X, 5) I16_(X, 6) I16_(X, 7) \
                I16_(X, 8) I16_(X, 9) I16_(X, A) I16_(X, B) I16_(X, C) I16_(X, D) I16_(X, E) I16_(X, F)
#define I128(X) I16_(X, 0) I16_(X, 1) I16_(X, 2) I16_(X, 3) I16_(X, 4) I16_(X, 5) I16_(X, 6) I16_(X, 7)
#define I16(X) I16_(X, 0)

/* ---------------------------------------------------------------- random inputs */
static const uint8_t  E8[]  = { 0, 1, 2, 0x7F, 0x80, 0x81, 0xFE, 0xFF, 0x40, 0x3F, 0xC0 };
static const uint16_t E16[] = { 0, 1, 2, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0xFFFE, 0x00FF, 0xFF00, 0x0080, 0x007F, 0xFF80, 0xFF7F, 0x0100, 0x4000, 0xC000 };
static const uint32_t E32[] = { 0, 1, 2, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFF, 0x10000, 0x8000, 0x7FFF,
                                0xFFFF8000, 0xFFFF7FFF, 0x00007FFF, 0x00FF00FF, 0x40000000, 0xC0000000 };
static const uint64_t E64[] = { 0, 1, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFull,
                                0x100000000ull, 0x80000000ull, 0xFFFFFFFF80000000ull, 0x7FFFFFFFull, 0xFFFFFFFF7FFFFFFFull };

static void gen_int(t_rng *r, V128 *v) {
    t_rng_fill(r, v, 16);
    switch (t_rng_range(r, 7)) {
    case 2: for (int i = 0; i < 16; i++) if (t_rng_range(r, 2)) v->b[i] = E8[t_rng_range(r, T_COUNTOF(E8))]; break;
    case 3: for (int i = 0; i < 8; i++) if (t_rng_range(r, 3)) v->w[i] = E16[t_rng_range(r, T_COUNTOF(E16))]; break;
    case 4: for (int i = 0; i < 4; i++) if (t_rng_range(r, 3)) v->d[i] = E32[t_rng_range(r, T_COUNTOF(E32))]; break;
    case 5: for (int i = 0; i < 2; i++) if (t_rng_range(r, 3)) v->q[i] = E64[t_rng_range(r, T_COUNTOF(E64))]; break;
    case 6: for (int i = 0; i < 16; i++) v->b[i] = (uint8_t)(t_rng_range(r, 9) - 4); break;   /* small +-4 */
    default: break;
    }
}
/* shift counts: the whole low quadword is the count; the high quadword is ignored */
static void gen_count(t_rng *r, V128 *v) {
    static const uint64_t C[] = { 0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256,
                                  0x100000000ull, 0x100000001ull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull, 0x10000000000ull };
    t_rng_fill(r, v, 16);
    if (t_rng_range(r, 4)) v->q[0] = C[t_rng_range(r, T_COUNTOF(C))];
    else v->q[0] = t_rng_range(r, 70);
}

enum { GK_ANY, GK_FIN, GK_CVT, GK_NONAN };
static uint64_t gen_fp(t_rng *r, const sf_fmt *f, int kind) {
    uint64_t S = t_rng_range(r, 2) ? sf_signbit(f) : 0, man = t_rng_next(r) & sf_mmask(f), em = sf_expmax(f);
    for (;;) {
        unsigned c = (unsigned)t_rng_range(r, 18);
        if (kind == GK_CVT && c < 9) {
            /* around the integer range: exponent -2 .. bits+2, few mantissa bits set (ties) */
            int span = f->bits == 32 ? 36 : 68;
            uint64_t e = (uint64_t)(f->bias - 2 + (int)t_rng_range(r, (uint64_t)span));
            uint64_t m = man;
            if (t_rng_range(r, 2)) m &= ~(sf_mmask(f) >> t_rng_range(r, (uint64_t)f->M));
            return S | (e << f->M) | m;
        }
        uint64_t x;
        switch (c) {
        case 0: case 1: x = t_rng_next(r) & (sf_signbit(f) * 2 - 1); break;
        case 2: x = S; break;
        case 3: x = S | (em << f->M); break;
        case 4: x = S | (em << f->M) | (1ull << (f->M - 1)) | man; break;
        case 5: x = S | (em << f->M) | ((man & (sf_mmask(f) >> 1)) ? (man & (sf_mmask(f) >> 1)) : 1); break;
        case 6: x = S | (man ? man : 1); break;
        case 7: {
            static const int pick[] = { 0, 1, 2, 3, 4, 5 };
            switch (pick[t_rng_range(r, 6)]) {
            case 0: x = S | 1; break;                                  /* min denormal */
            case 1: x = S | sf_mmask(f); break;                        /* max denormal */
            case 2: x = S | (1ull << f->M); break;                     /* min normal */
            case 3: x = sf_maxfin(f, (int)(S != 0)); break;            /* max finite */
            case 4: x = S | ((uint64_t)f->bias << f->M); break;        /* 1.0 */
            default: x = S | ((uint64_t)f->bias << f->M) | 1; break;   /* 1.0 + ulp */
            }
            break;
        }
        case 8: case 9: case 10: case 11: case 12:
            x = S | ((uint64_t)(f->bias - 26 + (int)t_rng_range(r, 52)) << f->M) | man; break;
        case 13: case 14: x = S | ((1 + t_rng_range(r, em - 1)) << f->M) | man; break;
        case 15: x = S | ((em - 1 - t_rng_range(r, 3)) << f->M) | man; break;
        case 16: x = S | ((1 + t_rng_range(r, 3)) << f->M) | man; break;
        default: x = S | ((uint64_t)(f->bias + (int)t_rng_range(r, 8)) << f->M) | (man & ~(sf_mmask(f) >> 3)); break;
        }
        if ((kind == GK_FIN || kind == GK_NONAN) && sf_isnan(f, x)) continue;
        if (kind == GK_FIN && sf_isinf(f, x)) continue;
        return x;
    }
}
/* second operand: independent, or related to a (cancellation, nearby exponent) */
static uint64_t gen_fp_b(t_rng *r, const sf_fmt *f, uint64_t a, int kind) {
    if (!sf_isspecial(f, a) && t_rng_range(r, 4) == 0) {
        uint64_t b = a;
        switch (t_rng_range(r, 3)) {
        case 0: b ^= sf_signbit(f); b ^= t_rng_range(r, 8); break;              /* x - (x +- few ulp) */
        case 1: b ^= t_rng_range(r, 4) << (f->M - 2); break;                     /* close values */
        default: {
            int64_t e = (int64_t)sf_exp(f, a) - (int64_t)t_rng_range(r, (uint64_t)f->M + 4);
            if (e < 1) e = 1;
            b = (b & ~(sf_expmax(f) << f->M)) | ((uint64_t)e << f->M);
            if (t_rng_range(r, 2)) b ^= sf_signbit(f);
        }
        }
        return b;
    }
    return gen_fp(r, f, kind);
}

static uint64_t g_mxcsr_mask;   /* from FXSAVE: DAZ supported when bit 6 is set */
static int daz_supported(void) {
    if (!g_mxcsr_mask) {
        static unsigned char area[512 + 16];
        unsigned char *a = (unsigned char *)(((uintptr_t)area + 15) & ~(uintptr_t)15);
        memset(a, 0, 512);
        t_fxsave(a);
        uint32_t m; memcpy(&m, a + 28, 4);
        g_mxcsr_mask = m ? m : 0xFFBF;
    }
    return (g_mxcsr_mask & 0x40) != 0;
}

/* ================================================================ floating point harness */
typedef void (*fop_fn)(V128 *d, const V128 *s, uint32_t *mx, int imm);
/* reference: whole-vector result; care = bytes compared (lanes the architecture leaves open are 0) */
typedef void (*vref_fn)(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm);

#define MX_RUN(insn_body, ...) \
    uint32_t sv_; __asm__ volatile("stmxcsr %1\n\tldmxcsr %2\n\t" insn_body "\n\tstmxcsr %2\n\tldmxcsr %1" __VA_ARGS__)
/* d = d op s */
#define FOP(nm, insn) static T_NOINLINE void fop_##nm(V128 *d, const V128 *s, uint32_t *mx, int imm) { (void)imm; \
    MX_RUN(insn " %3, %0", : "+x"(d->v), "=m"(sv_), "+m"(*mx) : "x"(s->v)); }
#define FOPI_C(n) case n: { MX_RUN(INSN_ " %4, %3, %0", : "+x"(d->v), "=m"(sv_), "+m"(*mx) : "x"(s->v), "i"(n)); break; }
#define FOPI(nm, insn, SWEEP) static T_NOINLINE void fop_##nm(V128 *d, const V128 *s, uint32_t *mx, int imm) { \
    switch (imm) { SWEEP(FOPI_C_##nm) default: break; } }

enum { MODE_DAZ = 4 };
typedef struct {
    const char *name; int feat; fop_fn op; vref_fn ref;
    int ga, gb;           /* generator kinds for a (dest) and b (source) */
    const sf_fmt *fa, *fb;/* lane formats for generation */
    int imm_lo, imm_hi;
    int modes;            /* bitmask of rounding modes 0..3 plus MODE_DAZ (bit 4) */
    int iters;
    int noflags;          /* do not compare flags */
} fop_t;

static void fill_fp(t_rng *r, V128 *v, const sf_fmt *f, int kind, const V128 *rel) {
    int n = 128 / f->bits;
    for (int i = 0; i < n; i++) {
        uint64_t x = rel ? gen_fp_b(r, f, f->bits == 32 ? rel->d[i] : rel->q[i], kind) : gen_fp(r, f, kind);
        if (f->bits == 32) v->d[i] = (uint32_t)x; else v->q[i] = x;
    }
}
static void fill_any(t_rng *r, V128 *v, int kind, const sf_fmt *f, const V128 *rel) {
    if (!f) { gen_int(r, v); return; }
    fill_fp(r, v, f, kind, rel);
}

static const char *RCN[] = { "RN", "RD", "RU", "RZ", "RN+DAZ" };

static void run_fop(const fop_t *o, uint64_t seed) {
    if (!cpu_has(o->feat)) { t_note("%s: %s not usable, not tested", o->name, cpu_feature_name(o->feat)); return; }
    t_rng r; t_rng_seed(&r, seed);
    vbegin();
    for (int imm = o->imm_lo; imm <= o->imm_hi; imm++) {
        for (int m = 0; m < 5; m++) {
            if (!(o->modes & (1 << m))) continue;
            int daz = m == 4, rc = daz ? 0 : m;
            if (daz && !daz_supported()) continue;
            for (int it = 0; it < o->iters; it++) {
                V128 a, b, d, w, care;
                fill_any(&r, &a, o->ga, o->fa, NULL);
                fill_any(&r, &b, o->gb, o->fb, (o->fa && o->fa == o->fb) ? &a : NULL);
                d = a;
                uint32_t mx = 0x1F80u | ((uint32_t)rc << 13) | (daz ? 0x40u : 0);
                o->op(&d, &b, &mx, imm);
                uint32_t fl = 0; int unc = 0;
                memset(&care, 0xFF, sizeof care);
                o->ref(&w, &a, &b, rc, daz, &fl, &unc, &care, imm);
                int ok = 1;
                for (int i = 0; i < 16; i++) if ((d.b[i] ^ w.b[i]) & care.b[i]) ok = 0;
                uint32_t gf = mx & 0x3F;
                int fok = o->noflags || unc || gf == fl;
                int lane = (o->fa && o->fa->bits == 64) ? 8 : 4;
                VCHECK(ok && fok, "%s imm=0x%02x %s: a=%s b=%s got=%s want=%s flags got=0x%02x want=0x%02x%s",
                       o->name, imm, RCN[m], hx(&a, lane), hx(&b, lane), hx(&d, lane), hx(&w, lane), gf, fl,
                       unc ? " (flags not compared)" : "");
                if ((mx & ~0x3Fu) != (0x1F80u | ((uint32_t)rc << 13) | (daz ? 0x40u : 0)))
                    VCHECK(0, "%s: MXCSR control bits changed: 0x%08x", o->name, mx);
            }
        }
    }
    VEND(o->name);
}
static void run_fops(const fop_t *t, size_t n, uint64_t seed) { for (size_t i = 0; i < n; i++) run_fop(&t[i], seed + i * 7919); }

static inline uint64_t lget(const V128 *v, const sf_fmt *f, int i) { return f->bits == 32 ? v->d[i] : v->q[i]; }
static inline void lset(V128 *v, const sf_fmt *f, int i, uint64_t x) { if (f->bits == 32) v->d[i] = (uint32_t)x; else v->q[i] = x; }
static inline uint64_t ones(const sf_fmt *f) { return f->bits == 32 ? 0xFFFFFFFFull : ~0ull; }

/* lane-wise reference generator: x = lane of a (dest), y = lane of b (source) */
#define VLANE(nm, F, SCALAR, expr) \
static void vr_##nm(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc_any, V128 *care, int imm) { \
    const sf_fmt *f = &F; (void)rc; (void)daz; (void)imm; *r = *a; \
    int nl = (SCALAR) ? 1 : 128 / f->bits; \
    for (int i = 0; i < nl; i++) { \
        uint64_t x = lget(a, f, i), y = lget(b, f, i); int unc = 0; (void)x; (void)y; \
        uint64_t res = (expr); lset(r, f, i, res); \
        if (unc) *unc_any = 1; \
        if (unc == 2) lset(care, f, i, 0); \
    } }

/* ROUND*: imm[1:0] mode unless imm[2] (MXCSR.RC); imm[3] suppresses the precision exception */
static uint64_t round_ref(const sf_fmt *f, uint64_t y, int rc, int imm, uint32_t *fl) {
    uint32_t t = 0;
    uint64_t r = sf_round_int(f, y, (imm & 4) ? rc : (imm & 3), &t);
    if (imm & 8) t &= ~(uint32_t)FL_PE;
    *fl |= t;
    return r;
}
#define FP_REFS(S, F) \
VLANE(add##S, F, 0, sf_add(f, x, y, 0, rc, daz, fl, &unc)) \
VLANE(sub##S, F, 0, sf_add(f, x, y, 1, rc, daz, fl, &unc)) \
VLANE(mul##S, F, 0, sf_mul(f, x, y, rc, daz, fl, &unc)) \
VLANE(div##S, F, 0, sf_div(f, x, y, rc, daz, fl, &unc)) \
VLANE(sqrt##S, F, 0, sf_sqrt(f, y, rc, daz, fl, &unc)) \
VLANE(min##S, F, 0, sf_minmax(f, x, y, 0, daz, fl, &unc)) \
VLANE(max##S, F, 0, sf_minmax(f, x, y, 1, daz, fl, &unc)) \
VLANE(and##S, F, 0, x & y) VLANE(andn##S, F, 0, ~x & y & ones(f)) VLANE(or##S, F, 0, x | y) VLANE(xor##S, F, 0, x ^ y) \
VLANE(cmp##S, F, 0, sf_cmp(f, x, y, imm, daz, fl, &unc) ? ones(f) : 0) \
VLANE(adds##S, F, 1, sf_add(f, x, y, 0, rc, daz, fl, &unc)) \
VLANE(subs##S, F, 1, sf_add(f, x, y, 1, rc, daz, fl, &unc)) \
VLANE(muls##S, F, 1, sf_mul(f, x, y, rc, daz, fl, &unc)) \
VLANE(divs##S, F, 1, sf_div(f, x, y, rc, daz, fl, &unc)) \
VLANE(sqrts##S, F, 1, sf_sqrt(f, y, rc, daz, fl, &unc)) \
VLANE(mins##S, F, 1, sf_minmax(f, x, y, 0, daz, fl, &unc)) \
VLANE(maxs##S, F, 1, sf_minmax(f, x, y, 1, daz, fl, &unc)) \
VLANE(cmps##S, F, 1, sf_cmp(f, x, y, imm, daz, fl, &unc) ? ones(f) : 0) \
VLANE(round##S, F, 0, round_ref(f, y, rc, imm, fl)) \
VLANE(rounds##S, F, 1, round_ref(f, y, rc, imm, fl))
FP_REFS(ps, SF32)
FP_REFS(pd, SF64)

FOP(addps, "addps") FOP(subps, "subps") FOP(mulps, "mulps") FOP(divps, "divps") FOP(sqrtps, "sqrtps")
FOP(minps, "minps") FOP(maxps, "maxps") FOP(andps, "andps") FOP(andnps, "andnps") FOP(orps, "orps") FOP(xorps, "xorps")
FOP(addss, "addss") FOP(subss, "subss") FOP(mulss, "mulss") FOP(divss, "divss") FOP(sqrtss, "sqrtss")
FOP(minss, "minss") FOP(maxss, "maxss")
FOP(addpd, "addpd") FOP(subpd, "subpd") FOP(mulpd, "mulpd") FOP(divpd, "divpd") FOP(sqrtpd, "sqrtpd")
FOP(minpd, "minpd") FOP(maxpd, "maxpd") FOP(andpd, "andpd") FOP(andnpd, "andnpd") FOP(orpd, "orpd") FOP(xorpd, "xorpd")
FOP(addsd, "addsd") FOP(subsd, "subsd") FOP(mulsd, "mulsd") FOP(divsd, "divsd") FOP(sqrtsd, "sqrtsd")
FOP(minsd, "minsd") FOP(maxsd, "maxsd")

#define INSN_ "cmpps"
#define FOPI_C_cmpps(n) FOPI_C(n)
FOPI(cmpps, "cmpps", I16)
#undef INSN_
#define INSN_ "cmpss"
#define FOPI_C_cmpss(n) FOPI_C(n)
FOPI(cmpss, "cmpss", I16)
#undef INSN_
#define INSN_ "cmppd"
#define FOPI_C_cmppd(n) FOPI_C(n)
FOPI(cmppd, "cmppd", I16)
#undef INSN_
#define INSN_ "cmpsd"
#define FOPI_C_cmpsd(n) FOPI_C(n)
FOPI(cmpsd, "cmpsd", I16)
#undef INSN_
#define INSN_ "roundps"
#define FOPI_C_roundps(n) FOPI_C(n)
FOPI(roundps, "roundps", I16)
#undef INSN_
#define INSN_ "roundss"
#define FOPI_C_roundss(n) FOPI_C(n)
FOPI(roundss, "roundss", I16)
#undef INSN_
#define INSN_ "roundpd"
#define FOPI_C_roundpd(n) FOPI_C(n)
FOPI(roundpd, "roundpd", I16)
#undef INSN_
#define INSN_ "roundsd"
#define FOPI_C_roundsd(n) FOPI_C(n)
FOPI(roundsd, "roundsd", I16)
#undef INSN_
#define INSN_ "dpps"
#define FOPI_C_dpps(n) FOPI_C(n)
FOPI(dpps, "dpps", I256)
#undef INSN_
#define INSN_ "dppd"
#define FOPI_C_dppd(n) FOPI_C(n)
FOPI(dppd, "dppd", I256)
#undef INSN_

#define ALLM 0x1F
#define F32P(nm, ga, gb, it) { #nm, F_SSE, fop_##nm, vr_##nm, ga, gb, &SF32, &SF32, 0, 0, ALLM, it, 0 }
#define F64P(nm, ga, gb, it) { #nm, F_SSE2, fop_##nm, vr_##nm, ga, gb, &SF64, &SF64, 0, 0, ALLM, it, 0 }
#define F32S(nm, ref, it) { #nm, F_SSE, fop_##nm, vr_##ref, GK_ANY, GK_ANY, &SF32, &SF32, 0, 0, ALLM, it, 0 }
#define F64S(nm, ref, it) { #nm, F_SSE2, fop_##nm, vr_##ref, GK_ANY, GK_ANY, &SF64, &SF64, 0, 0, ALLM, it, 0 }

TEST(sse_fp32_add_sub_packed) {
    NEED(F_SSE);
    static const fop_t t[] = { F32P(addps, GK_ANY, GK_ANY, 1500), F32P(subps, GK_ANY, GK_ANY, 1500) };
    run_fops(t, T_COUNTOF(t), 0x5EA001);
}
TEST(sse_fp32_mul_div_packed) {
    NEED(F_SSE);
    static const fop_t t[] = { F32P(mulps, GK_ANY, GK_ANY, 1500), F32P(divps, GK_ANY, GK_ANY, 1500) };
    run_fops(t, T_COUNTOF(t), 0x5EA002);
}
TEST(sse_fp32_sqrt_packed) {
    NEED(F_SSE);
    static const fop_t t[] = { F32P(sqrtps, GK_ANY, GK_ANY, 2500) };
    run_fops(t, T_COUNTOF(t), 0x5EA003);
}
TEST(sse_fp32_min_max_packed) {
    NEED(F_SSE);
    static const fop_t t[] = { F32P(minps, GK_ANY, GK_ANY, 1500), F32P(maxps, GK_ANY, GK_ANY, 1500) };
    run_fops(t, T_COUNTOF(t), 0x5EA004);
}
TEST(sse_fp32_logic_packed) {
    NEED(F_SSE);
    static const fop_t t[] = { F32P(andps, GK_ANY, GK_ANY, 300), F32P(andnps, GK_ANY, GK_ANY, 300),
                               F32P(orps, GK_ANY, GK_ANY, 300), F32P(xorps, GK_ANY, GK_ANY, 300) };
    run_fops(t, T_COUNTOF(t), 0x5EA005);
}
TEST(sse_fp32_scalar_ops_preserve_upper) {
    NEED(F_SSE);
    static const fop_t t[] = { F32S(addss, addsps, 1200), F32S(subss, subsps, 1200), F32S(mulss, mulsps, 1200),
                               F32S(divss, divsps, 1200), F32S(sqrtss, sqrtsps, 1200), F32S(minss, minsps, 1200),
                               F32S(maxss, maxsps, 1200) };
    run_fops(t, T_COUNTOF(t), 0x5EA006);
}
TEST(sse_fp32_cmpps_cmpss_predicates) {
    NEED(F_SSE);
    static const fop_t t[] = {
        { "cmpps", F_SSE, fop_cmpps, vr_cmpps, GK_ANY, GK_ANY, &SF32, &SF32, 0, 7, ALLM, 400, 0 },
        { "cmpss", F_SSE, fop_cmpss, vr_cmpsps, GK_ANY, GK_ANY, &SF32, &SF32, 0, 7, ALLM, 400, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA007);
}
TEST(sse2_fp64_add_sub_packed) {
    NEED(F_SSE2);
    static const fop_t t[] = { F64P(addpd, GK_ANY, GK_ANY, 2000), F64P(subpd, GK_ANY, GK_ANY, 2000) };
    run_fops(t, T_COUNTOF(t), 0x5EA011);
}
TEST(sse2_fp64_mul_div_packed) {
    NEED(F_SSE2);
    static const fop_t t[] = { F64P(mulpd, GK_ANY, GK_ANY, 2000), F64P(divpd, GK_ANY, GK_ANY, 2000) };
    run_fops(t, T_COUNTOF(t), 0x5EA012);
}
TEST(sse2_fp64_sqrt_packed) {
    NEED(F_SSE2);
    static const fop_t t[] = { F64P(sqrtpd, GK_ANY, GK_ANY, 3000) };
    run_fops(t, T_COUNTOF(t), 0x5EA013);
}
TEST(sse2_fp64_min_max_packed) {
    NEED(F_SSE2);
    static const fop_t t[] = { F64P(minpd, GK_ANY, GK_ANY, 2000), F64P(maxpd, GK_ANY, GK_ANY, 2000) };
    run_fops(t, T_COUNTOF(t), 0x5EA014);
}
TEST(sse2_fp64_logic_packed) {
    NEED(F_SSE2);
    static const fop_t t[] = { F64P(andpd, GK_ANY, GK_ANY, 300), F64P(andnpd, GK_ANY, GK_ANY, 300),
                               F64P(orpd, GK_ANY, GK_ANY, 300), F64P(xorpd, GK_ANY, GK_ANY, 300) };
    run_fops(t, T_COUNTOF(t), 0x5EA015);
}
TEST(sse2_fp64_scalar_ops_preserve_upper) {
    NEED(F_SSE2);
    static const fop_t t[] = { F64S(addsd, addspd, 1200), F64S(subsd, subspd, 1200), F64S(mulsd, mulspd, 1200),
                               F64S(divsd, divspd, 1200), F64S(sqrtsd, sqrtspd, 1200), F64S(minsd, minspd, 1200),
                               F64S(maxsd, maxspd, 1200) };
    run_fops(t, T_COUNTOF(t), 0x5EA016);
}
TEST(sse2_fp64_cmppd_cmpsd_predicates) {
    NEED(F_SSE2);
    static const fop_t t[] = {
        { "cmppd", F_SSE2, fop_cmppd, vr_cmppd, GK_ANY, GK_ANY, &SF64, &SF64, 0, 7, ALLM, 400, 0 },
        { "cmpsd", F_SSE2, fop_cmpsd, vr_cmpspd, GK_ANY, GK_ANY, &SF64, &SF64, 0, 7, ALLM, 400, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA017);
}

/* ---- COMISS/UCOMISS/COMISD/UCOMISD: ZF,PF,CF = 111 unordered, 000 greater, 001 less, 100 equal;
 *      OF, SF, AF cleared; COMI* raise IE on QNaN and SNaN, UCOMI* only on SNaN */
typedef struct { uint8_t zf, pf, cf, of, sf; } eflg;
#define COMI(nm, insn) static T_NOINLINE void ci_##nm(const V128 *a, const V128 *b, uint32_t *mx, eflg *e) { uint32_t sv_; \
    __asm__ volatile("stmxcsr %5\n\tldmxcsr %6\n\t" insn " %8, %7\n\tstmxcsr %6\n\tldmxcsr %5" \
        : "=@ccz"(e->zf), "=@ccp"(e->pf), "=@ccc"(e->cf), "=@cco"(e->of), "=@ccs"(e->sf), "=m"(sv_), "+m"(*mx) : "x"(a->v), "x"(b->v)); }
COMI(comiss, "comiss") COMI(ucomiss, "ucomiss") COMI(comisd, "comisd") COMI(ucomisd, "ucomisd")

static void run_comi(const char *name, int dbl, int quiet, void (*fn)(const V128 *, const V128 *, uint32_t *, eflg *), uint64_t seed) {
    const sf_fmt *f = dbl ? &SF64 : &SF32;
    t_rng r; t_rng_seed(&r, seed);
    vbegin();
    for (int m = 0; m < 2; m++) {
        int daz = m == 1;
        if (daz && !daz_supported()) continue;
        for (int it = 0; it < 3000; it++) {
            V128 a, b; fill_fp(&r, &a, f, GK_ANY, NULL); fill_fp(&r, &b, f, GK_ANY, &a);
            if (it % 5 == 0) lset(&b, f, 0, lget(&a, f, 0));    /* equal operands */
            uint64_t x = lget(&a, f, 0), y = lget(&b, f, 0);
            uint32_t mx = 0x1F80u | (daz ? 0x40u : 0), fl = 0; eflg e; int unc = 0;
            fn(&a, &b, &mx, &e);
            int un = sf_isnan(f, x) || sf_isnan(f, y);
            int lt = 0, eq = 0;
            if (un) { if (!quiet || sf_issnan(f, x) || sf_issnan(f, y)) fl |= FL_IE; if (sf_isden(f, x) || sf_isden(f, y)) unc = 1; }
            else { uint64_t cx = sf_src(f, x, daz, &fl), cy = sf_src(f, y, daz, &fl); lt = sf_lt(f, cx, cy); eq = sf_eq(f, cx, cy); }
            int wz = un || eq, wp = un, wc = un || lt;
            uint32_t gf = mx & 0x3F;
            VCHECK(e.zf == wz && e.pf == wp && e.cf == wc && !e.of && !e.sf && (unc || gf == fl),
                   "%s%s a=%s b=%s: ZF%d PF%d CF%d OF%d SF%d want ZF%d PF%d CF%d; flags 0x%02x want 0x%02x",
                   name, daz ? " DAZ" : "", hx(&a, dbl ? 8 : 4), hx(&b, dbl ? 8 : 4), e.zf, e.pf, e.cf, e.of, e.sf, wz, wp, wc, gf, fl);
        }
    }
    VEND(name);
}
TEST(sse_comiss_ucomiss_flags) {
    NEED(F_SSE);
    run_comi("comiss", 0, 0, ci_comiss, 0x5EA021);
    run_comi("ucomiss", 0, 1, ci_ucomiss, 0x5EA022);
}
TEST(sse2_comisd_ucomisd_flags) {
    NEED(F_SSE2);
    run_comi("comisd", 1, 0, ci_comisd, 0x5EA023);
    run_comi("ucomisd", 1, 1, ci_ucomisd, 0x5EA024);
}

/* ---- known answers: hand-derived, independent of the softfp model (also validates the model) */
typedef struct { const char *op; uint64_t a, b; uint32_t mx; uint64_t want; uint32_t flags; } fka;
static fop_fn fka_op(const char *op, int dbl) {
    static const struct { const char *n; fop_fn s, d; } m[] = {
        { "add", fop_addps, fop_addpd }, { "sub", fop_subps, fop_subpd }, { "mul", fop_mulps, fop_mulpd },
        { "div", fop_divps, fop_divpd }, { "sqrt", fop_sqrtps, fop_sqrtpd }, { "min", fop_minps, fop_minpd },
        { "max", fop_maxps, fop_maxpd } };
    for (size_t i = 0; i < T_COUNTOF(m); i++) if (!strcmp(m[i].n, op)) return dbl ? m[i].d : m[i].s;
    return NULL;
}
static void run_fka(const fka *t, size_t n, int dbl) {
    const sf_fmt *f = dbl ? &SF64 : &SF32;
    for (size_t i = 0; i < n; i++) {
        if ((t[i].mx & 0x40) && !daz_supported()) continue;
        V128 a, b, d;
        for (int l = 0; l < 128 / f->bits; l++) { lset(&a, f, l, t[i].a); lset(&b, f, l, t[i].b); }
        d = a;
        uint32_t mx = t[i].mx;
        fka_op(t[i].op, dbl)(&d, &b, &mx, 0);
        uint64_t got = lget(&d, f, 0), got1 = lget(&d, f, 1);
        CHECKF(got == t[i].want && got1 == t[i].want && (mx & 0x3F) == t[i].flags,
               "#%u %s(0x%llx, 0x%llx) mxcsr=0x%04x: got 0x%llx flags 0x%02x, want 0x%llx flags 0x%02x",
               (unsigned)i, t[i].op, (unsigned long long)t[i].a, (unsigned long long)t[i].b, t[i].mx,
               (unsigned long long)got, mx & 0x3F, (unsigned long long)t[i].want, t[i].flags);
        /* the model must agree with the hand-derived answer too */
        uint32_t fl = 0; int unc = 0; uint64_t mw = 0;
        int rc = (t[i].mx >> 13) & 3, daz = (t[i].mx >> 6) & 1;
        if (!(t[i].mx & 0x8000)) {   /* model does not do FTZ */
            if (!strcmp(t[i].op, "add")) mw = sf_add(f, t[i].a, t[i].b, 0, rc, daz, &fl, &unc);
            else if (!strcmp(t[i].op, "sub")) mw = sf_add(f, t[i].a, t[i].b, 1, rc, daz, &fl, &unc);
            else if (!strcmp(t[i].op, "mul")) mw = sf_mul(f, t[i].a, t[i].b, rc, daz, &fl, &unc);
            else if (!strcmp(t[i].op, "div")) mw = sf_div(f, t[i].a, t[i].b, rc, daz, &fl, &unc);
            else if (!strcmp(t[i].op, "sqrt")) mw = sf_sqrt(f, t[i].b, rc, daz, &fl, &unc);
            else mw = sf_minmax(f, t[i].a, t[i].b, !strcmp(t[i].op, "max"), daz, &fl, &unc);
            CHECKF(mw == t[i].want && fl == t[i].flags, "model self-check #%u %s: model 0x%llx/0x%02x want 0x%llx/0x%02x",
                   (unsigned)i, t[i].op, (unsigned long long)mw, fl, (unsigned long long)t[i].want, t[i].flags);
        }
    }
}
#define MXRN 0x1F80u
#define MXRD 0x3F80u
#define MXRU 0x5F80u
#define MXRZ 0x7F80u
#define MXDAZ 0x1FC0u
#define MXFTZ 0x9F80u
TEST(sse_fp32_known_answers) {
    NEED(F_SSE);
    static const fka t[] = {
        { "add", 0x3F800000, 0x40000000, MXRN, 0x40400000, 0 },                 /* 1 + 2 = 3 */
        { "add", 0x7F7FFFFF, 0x7F7FFFFF, MXRN, 0x7F800000, FL_OE | FL_PE },     /* max+max -> inf */
        { "add", 0x7F7FFFFF, 0x7F7FFFFF, MXRZ, 0x7F7FFFFF, FL_OE | FL_PE },     /* RZ -> max */
        { "add", 0xFF7FFFFF, 0xFF7FFFFF, MXRU, 0xFF7FFFFF, FL_OE | FL_PE },     /* RU, negative -> -max */
        { "add", 0xFF7FFFFF, 0xFF7FFFFF, MXRD, 0xFF800000, FL_OE | FL_PE },
        { "add", 0x3F800000, 0x33800000, MXRN, 0x3F800000, FL_PE },             /* 1 + 2^-24: tie to even */
        { "add", 0x3F800000, 0x33800000, MXRU, 0x3F800001, FL_PE },
        { "add", 0x3F800000, 0x33800001, MXRN, 0x3F800001, FL_PE },             /* just above the tie */
        { "add", 0x3F800001, 0x33800000, MXRN, 0x3F800002, FL_PE },             /* tie, odd -> up */
        { "add", 0x7F800000, 0xFF800000, MXRN, 0xFFC00000, FL_IE },             /* inf - inf: indefinite */
        { "sub", 0x3F800000, 0x3F800000, MXRN, 0x00000000, 0 },                 /* x - x = +0 */
        { "sub", 0x3F800000, 0x3F800000, MXRD, 0x80000000, 0 },                 /*   ... -0 when RD */
        { "add", 0x80000000, 0x80000000, MXRN, 0x80000000, 0 },                 /* -0 + -0 */
        { "add", 0x00000000, 0x80000000, MXRN, 0x00000000, 0 },
        { "add", 0x00000000, 0x80000000, MXRD, 0x80000000, 0 },
        { "add", 0x7FC00001, 0x3F800000, MXRN, 0x7FC00001, 0 },                 /* QNaN propagates, no IE */
        { "add", 0x7F800001, 0x3F800000, MXRN, 0x7FC00001, FL_IE },             /* SNaN quieted, IE */
        { "add", 0x3F800000, 0xFF800003, MXRN, 0xFFC00003, FL_IE },             /* NaN from second source */
        { "add", 0x7FC00002, 0xFF800003, MXRN, 0x7FC00002, FL_IE },             /* both NaN: first source */
        { "add", 0xFF800005, 0x7FC00006, MXRN, 0xFFC00005, FL_IE },
        { "add", 0x00000001, 0x3F800000, MXRN, 0x3F800000, FL_DE | FL_PE },     /* denormal operand */
        { "add", 0x00000001, 0x00000000, MXDAZ, 0x00000000, 0 },                /* DAZ: no DE, zero */
        { "add", 0x80000001, 0x00000000, MXDAZ, 0x00000000, 0 },                /* -0 + +0 = +0 */
        { "add", 0x00000003, 0x00000005, MXRN, 0x00000008, FL_DE },             /* exact denormal sum: no UE */
        { "mul", 0x3F800000, 0x40490FDB, MXRN, 0x40490FDB, 0 },
        { "mul", 0x00000001, 0x3F000000, MXRN, 0x00000000, FL_DE | FL_UE | FL_PE }, /* min_den/2 tie -> 0 */
        { "mul", 0x00000001, 0x3F000000, MXRU, 0x00000001, FL_DE | FL_UE | FL_PE },
        { "mul", 0x00000003, 0x3F000000, MXRN, 0x00000002, FL_DE | FL_UE | FL_PE }, /* 1.5 -> 2 (even) */
        { "mul", 0x00800000, 0x3F000000, MXRN, 0x00400000, 0 },                 /* exact tiny: no UE */
        { "mul", 0x00800001, 0x3F000000, MXRN, 0x00400000, FL_UE | FL_PE },     /* inexact tiny */
        { "mul", 0x00800001, 0x3F000000, MXFTZ, 0x00000000, FL_UE | FL_PE },    /* FTZ */
        { "mul", 0x80800001, 0x3F000000, MXFTZ, 0x80000000, FL_UE | FL_PE },    /* FTZ keeps sign */
        { "mul", 0x00FFFFFF, 0x3F000000, MXRN, 0x00800000, FL_UE | FL_PE },     /* rounds up to min normal:   */
        { "mul", 0x00FFFFFF, 0x3F000000, MXRZ, 0x007FFFFF, FL_UE | FL_PE },     /*  tininess after rounding  */
        { "mul", 0x7F800000, 0x00000000, MXRN, 0xFFC00000, FL_IE },             /* inf * 0 */
        { "mul", 0x7F000000, 0x40000000, MXRN, 0x7F800000, FL_OE | FL_PE },
        { "mul", 0xC0000000, 0x80000000, MXRN, 0x00000000, 0 },                 /* -2 * -0 = +0 */
        { "div", 0x3F800000, 0x40400000, MXRN, 0x3EAAAAAB, FL_PE },             /* 1/3 */
        { "div", 0x3F800000, 0x40400000, MXRZ, 0x3EAAAAAA, FL_PE },
        { "div", 0x3F800000, 0x40400000, MXRD, 0x3EAAAAAA, FL_PE },
        { "div", 0x3F800000, 0x40400000, MXRU, 0x3EAAAAAB, FL_PE },
        { "div", 0xBF800000, 0x40400000, MXRD, 0xBEAAAAAB, FL_PE },
        { "div", 0x3F800000, 0x00000000, MXRN, 0x7F800000, FL_ZE },             /* 1/0 */
        { "div", 0xBF800000, 0x00000000, MXRN, 0xFF800000, FL_ZE },
        { "div", 0x3F800000, 0x80000000, MXRN, 0xFF800000, FL_ZE },
        { "div", 0x00000000, 0x00000000, MXRN, 0xFFC00000, FL_IE },             /* 0/0 */
        { "div", 0x7F800000, 0x7F800000, MXRN, 0xFFC00000, FL_IE },             /* inf/inf */
        { "div", 0x7F800000, 0x00000000, MXRN, 0x7F800000, 0 },                 /* inf/0: no ZE */
        { "div", 0x3F800000, 0x00000000, MXDAZ, 0x7F800000, FL_ZE },
        { "div", 0x3F800000, 0x00000001, MXDAZ, 0x7F800000, FL_ZE },            /* DAZ denormal divisor */
        { "sqrt", 0, 0x40800000, MXRN, 0x40000000, 0 },                          /* sqrt 4 = 2 */
        { "sqrt", 0, 0x40000000, MXRN, 0x3FB504F3, FL_PE },                      /* sqrt 2 */
        { "sqrt", 0, 0x40000000, MXRU, 0x3FB504F4, FL_PE },
        { "sqrt", 0, 0x40000000, MXRZ, 0x3FB504F3, FL_PE },
        { "sqrt", 0, 0xBF800000, MXRN, 0xFFC00000, FL_IE },                      /* sqrt(-1) */
        { "sqrt", 0, 0x80000000, MXRN, 0x80000000, 0 },                          /* sqrt(-0) = -0 */
        { "sqrt", 0, 0x7F800000, MXRN, 0x7F800000, 0 },
        { "sqrt", 0, 0xFF800000, MXRN, 0xFFC00000, FL_IE },
        { "sqrt", 0, 0x00000001, MXRN, 0x1A3504F3, FL_DE | FL_PE },              /* sqrt(2^-149) */
        { "sqrt", 0, 0x80000001, MXDAZ, 0x80000000, 0 },                         /* DAZ: sqrt(-0) */
        { "min", 0x00000000, 0x80000000, MXRN, 0x80000000, 0 },                 /* both zero: second */
        { "min", 0x80000000, 0x00000000, MXRN, 0x00000000, 0 },
        { "max", 0x80000000, 0x00000000, MXRN, 0x00000000, 0 },
        { "max", 0x00000000, 0x80000000, MXRN, 0x80000000, 0 },
        { "min", 0x7FC00000, 0x3F800000, MXRN, 0x3F800000, FL_IE },             /* QNaN first: second */
        { "min", 0x3F800000, 0x7FC00000, MXRN, 0x7FC00000, FL_IE },             /* QNaN second: QNaN */
        { "min", 0x3F800000, 0x7F800001, MXRN, 0x7F800001, FL_IE },             /* SNaN returned unquieted */
        { "max", 0x7F800001, 0x40000000, MXRN, 0x40000000, FL_IE },
        { "max", 0xFF800000, 0x7F800000, MXRN, 0x7F800000, 0 },
        { "min", 0xBF800000, 0x3F800000, MXRN, 0xBF800000, 0 },
        { "max", 0x00000001, 0x00000000, MXRN, 0x00000001, FL_DE },
    };
    run_fka(t, T_COUNTOF(t), 0);
}
TEST(sse2_fp64_known_answers) {
    NEED(F_SSE2);
    static const fka t[] = {
        { "add", 0x3FF0000000000000ull, 0x4000000000000000ull, MXRN, 0x4008000000000000ull, 0 },
        { "add", 0x3FF0000000000000ull, 0x3CA0000000000000ull, MXRN, 0x3FF0000000000000ull, FL_PE },  /* 1+2^-53 tie */
        { "add", 0x3FF0000000000000ull, 0x3CA0000000000000ull, MXRU, 0x3FF0000000000001ull, FL_PE },
        { "add", 0x3FF0000000000001ull, 0x3CA0000000000000ull, MXRN, 0x3FF0000000000002ull, FL_PE },
        { "add", 0x7FEFFFFFFFFFFFFFull, 0x7FEFFFFFFFFFFFFFull, MXRN, 0x7FF0000000000000ull, FL_OE | FL_PE },
        { "add", 0x7FEFFFFFFFFFFFFFull, 0x7FEFFFFFFFFFFFFFull, MXRZ, 0x7FEFFFFFFFFFFFFFull, FL_OE | FL_PE },
        { "sub", 0x4000000000000000ull, 0x4000000000000000ull, MXRD, 0x8000000000000000ull, 0 },
        { "add", 0x7FF0000000000000ull, 0xFFF0000000000000ull, MXRN, 0xFFF8000000000000ull, FL_IE },
        { "add", 0x7FF0000000000001ull, 0x3FF0000000000000ull, MXRN, 0x7FF8000000000001ull, FL_IE },
        { "add", 0x7FF8000000000002ull, 0xFFF0000000000003ull, MXRN, 0x7FF8000000000002ull, FL_IE },
        { "add", 0x0000000000000001ull, 0x0000000000000000ull, MXDAZ, 0x0000000000000000ull, 0 },
        { "mul", 0x0000000000000001ull, 0x3FE0000000000000ull, MXRN, 0x0000000000000000ull, FL_DE | FL_UE | FL_PE },
        { "mul", 0x0010000000000001ull, 0x3FE0000000000000ull, MXFTZ, 0x0000000000000000ull, FL_UE | FL_PE },
        { "mul", 0x001FFFFFFFFFFFFFull, 0x3FE0000000000000ull, MXRN, 0x0010000000000000ull, FL_UE | FL_PE },
        { "mul", 0x7FF0000000000000ull, 0x8000000000000000ull, MXRN, 0xFFF8000000000000ull, FL_IE },
        { "div", 0x3FF0000000000000ull, 0x4008000000000000ull, MXRN, 0x3FD5555555555555ull, FL_PE },  /* 1/3 */
        { "div", 0x3FF0000000000000ull, 0x4008000000000000ull, MXRU, 0x3FD5555555555556ull, FL_PE },
        { "div", 0x4000000000000000ull, 0x4008000000000000ull, MXRN, 0x3FE5555555555555ull, FL_PE },  /* 2/3 */
        { "div", 0x4000000000000000ull, 0x4008000000000000ull, MXRU, 0x3FE5555555555556ull, FL_PE },
        { "div", 0x3FF0000000000000ull, 0x0000000000000000ull, MXRN, 0x7FF0000000000000ull, FL_ZE },
        { "div", 0x0000000000000000ull, 0x8000000000000000ull, MXRN, 0xFFF8000000000000ull, FL_IE },
        { "sqrt", 0, 0x4000000000000000ull, MXRN, 0x3FF6A09E667F3BCDull, FL_PE },  /* sqrt 2 */
        { "sqrt", 0, 0x4000000000000000ull, MXRZ, 0x3FF6A09E667F3BCCull, FL_PE },
        { "sqrt", 0, 0x4022000000000000ull, MXRN, 0x4008000000000000ull, 0 },      /* sqrt 9 = 3 */
        { "sqrt", 0, 0xBFF0000000000000ull, MXRN, 0xFFF8000000000000ull, FL_IE },
        { "sqrt", 0, 0x8000000000000000ull, MXRN, 0x8000000000000000ull, 0 },
        { "min", 0x0000000000000000ull, 0x8000000000000000ull, MXRN, 0x8000000000000000ull, 0 },
        { "max", 0x8000000000000000ull, 0x0000000000000000ull, MXRN, 0x0000000000000000ull, 0 },
        { "min", 0x7FF8000000000000ull, 0x3FF0000000000000ull, MXRN, 0x3FF0000000000000ull, FL_IE },
        { "max", 0x3FF0000000000000ull, 0x7FF0000000000001ull, MXRN, 0x7FF0000000000001ull, FL_IE },
    };
    run_fka(t, T_COUNTOF(t), 1);
}

/* ---- SSE3 / horizontal FP */
static void hpair_ref(const sf_fmt *f, V128 *r, int lane, uint64_t x, uint64_t y, int sub, int rc, int daz, uint32_t *fl, int *unc_any, V128 *care) {
    int unc = 0;
    uint64_t v = sf_add(f, x, y, sub, rc, daz, fl, &unc);
    if (sf_isnan(f, x) && sf_isnan(f, y)) {          /* which NaN wins is not pinned for the horizontal forms */
        lset(care, f, lane, 0); unc = 1;
    }
    if (unc) *unc_any = 1;
    lset(r, f, lane, v);
}
static void vr_addsubps(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)imm;
    for (int i = 0; i < 4; i++) { int u = 0; r->d[i] = (uint32_t)sf_add(&SF32, a->d[i], b->d[i], !(i & 1), rc, daz, fl, &u); if (u) *unc = 1; }
    (void)care;
}
static void vr_addsubpd(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)imm; (void)care;
    for (int i = 0; i < 2; i++) { int u = 0; r->q[i] = sf_add(&SF64, a->q[i], b->q[i], !(i & 1), rc, daz, fl, &u); if (u) *unc = 1; }
}
static void vr_haddps(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)imm;
    hpair_ref(&SF32, r, 0, a->d[0], a->d[1], 0, rc, daz, fl, unc, care); hpair_ref(&SF32, r, 1, a->d[2], a->d[3], 0, rc, daz, fl, unc, care);
    hpair_ref(&SF32, r, 2, b->d[0], b->d[1], 0, rc, daz, fl, unc, care); hpair_ref(&SF32, r, 3, b->d[2], b->d[3], 0, rc, daz, fl, unc, care);
}
static void vr_hsubps(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)imm;
    hpair_ref(&SF32, r, 0, a->d[0], a->d[1], 1, rc, daz, fl, unc, care); hpair_ref(&SF32, r, 1, a->d[2], a->d[3], 1, rc, daz, fl, unc, care);
    hpair_ref(&SF32, r, 2, b->d[0], b->d[1], 1, rc, daz, fl, unc, care); hpair_ref(&SF32, r, 3, b->d[2], b->d[3], 1, rc, daz, fl, unc, care);
}
static void vr_haddpd(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)imm;
    hpair_ref(&SF64, r, 0, a->q[0], a->q[1], 0, rc, daz, fl, unc, care); hpair_ref(&SF64, r, 1, b->q[0], b->q[1], 0, rc, daz, fl, unc, care);
}
static void vr_hsubpd(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)imm;
    hpair_ref(&SF64, r, 0, a->q[0], a->q[1], 1, rc, daz, fl, unc, care); hpair_ref(&SF64, r, 1, b->q[0], b->q[1], 1, rc, daz, fl, unc, care);
}
FOP(addsubps, "addsubps") FOP(addsubpd, "addsubpd") FOP(haddps, "haddps") FOP(hsubps, "hsubps") FOP(haddpd, "haddpd") FOP(hsubpd, "hsubpd")

TEST(sse3_addsub_hadd_hsub) {
    NEED(F_SSE3);
    static const fop_t t[] = {
        { "addsubps", F_SSE3, fop_addsubps, vr_addsubps, GK_ANY, GK_ANY, &SF32, &SF32, 0, 0, ALLM, 1200, 0 },
        { "addsubpd", F_SSE3, fop_addsubpd, vr_addsubpd, GK_ANY, GK_ANY, &SF64, &SF64, 0, 0, ALLM, 1200, 0 },
        { "haddps", F_SSE3, fop_haddps, vr_haddps, GK_ANY, GK_ANY, &SF32, &SF32, 0, 0, ALLM, 1200, 0 },
        { "hsubps", F_SSE3, fop_hsubps, vr_hsubps, GK_ANY, GK_ANY, &SF32, &SF32, 0, 0, ALLM, 1200, 0 },
        { "haddpd", F_SSE3, fop_haddpd, vr_haddpd, GK_ANY, GK_ANY, &SF64, &SF64, 0, 0, ALLM, 1200, 0 },
        { "hsubpd", F_SSE3, fop_hsubpd, vr_hsubpd, GK_ANY, GK_ANY, &SF64, &SF64, 0, 0, ALLM, 1200, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA031);
}

/* ---- SSE4.1 ROUND*: imm[1:0] mode, imm[2] use MXCSR.RC, imm[3] suppress PE; imm[7:4] reserved (not used) */
TEST(sse41_roundps_roundpd_all_modes) {
    NEED(F_SSE41);
    static const fop_t t[] = {
        { "roundps", F_SSE41, fop_roundps, vr_roundps, GK_ANY, GK_CVT, &SF32, &SF32, 0, 15, 0xF, 250, 0 },
        { "roundpd", F_SSE41, fop_roundpd, vr_roundpd, GK_ANY, GK_CVT, &SF64, &SF64, 0, 15, 0xF, 250, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA041);
}
TEST(sse41_roundss_roundsd_preserve_upper) {
    NEED(F_SSE41);
    static const fop_t t[] = {
        { "roundss", F_SSE41, fop_roundss, vr_roundsps, GK_ANY, GK_CVT, &SF32, &SF32, 0, 15, 0xF, 200, 0 },
        { "roundsd", F_SSE41, fop_roundsd, vr_roundspd, GK_ANY, GK_CVT, &SF64, &SF64, 0, 15, 0xF, 200, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA042);
}
TEST(sse41_round_known_answers) {
    NEED(F_SSE41);
    /* value, imm -> result: floor/ceil/trunc/nearest-even of +-2.5, -0.5 -> -0, 2^23+0.5 not representable */
    static const struct { uint32_t x; int imm; uint32_t want; } t[] = {
        { 0x40200000, 0x8, 0x40000000 }, { 0x40200000, 0x9, 0x40000000 }, { 0x40200000, 0xA, 0x40400000 }, { 0x40200000, 0xB, 0x40000000 },
        { 0xC0200000, 0x8, 0xC0000000 }, { 0xC0200000, 0x9, 0xC0400000 }, { 0xC0200000, 0xA, 0xC0000000 }, { 0xC0200000, 0xB, 0xC0000000 },
        { 0x3FC00000, 0x8, 0x40000000 }, { 0xBF000000, 0x8, 0x80000000 }, { 0xBF000000, 0xA, 0x80000000 }, { 0x3F000000, 0x9, 0x00000000 },
        { 0x3E800000, 0xA, 0x3F800000 }, { 0x00000001, 0xA, 0x3F800000 }, { 0x80000001, 0x9, 0xBF800000 }, { 0x4B000001, 0x8, 0x4B000001 },
        { 0x7F800000, 0x8, 0x7F800000 }, { 0xFF800000, 0x9, 0xFF800000 }, { 0x7F800001, 0x8, 0x7FC00001 }, { 0x80000000, 0xA, 0x80000000 },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        V128 a = { 0 }, b; for (int l = 0; l < 4; l++) b.d[l] = t[i].x;
        uint32_t mx = 0x1F80; V128 d = a;
        fop_roundps(&d, &b, &mx, t[i].imm);
        CHECKF(d.d[0] == t[i].want && d.d[3] == t[i].want, "roundps(0x%08x, imm %d) = 0x%08x want 0x%08x", t[i].x, t[i].imm, d.d[0], t[i].want);
        uint32_t want_fl = (t[i].x == 0x7F800001) ? FL_IE : 0;   /* imm bit 3 set: PE suppressed */
        CHECKF((mx & 0x3F) == want_fl, "roundps(0x%08x, imm %d) flags 0x%02x want 0x%02x", t[i].x, t[i].imm, mx & 0x3F, want_fl);
    }
    /* imm[3] = 0: inexact sets PE; imm[2] = 1 takes MXCSR.RC (RU here) */
    V128 b, d = { 0 }; for (int l = 0; l < 4; l++) b.d[l] = 0x40200000;
    uint32_t mx = 0x5F80; fop_roundps(&d, &b, &mx, 0x4);
    CHECKF(d.d[0] == 0x40400000 && (mx & 0x3F) == FL_PE, "roundps imm 4 under RU: 0x%08x flags 0x%02x", d.d[0], mx & 0x3F);
}

/* ---- DPPS/DPPD: per-lane products (imm[7:4] / imm[5:4]) rounded, then the documented add tree,
 *      broadcast to lanes selected by imm[3:0] / imm[1:0], other lanes +0.0. Flags not compared. */
static void vr_dpps(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)care; *unc = 1;
    uint64_t t1[4];
    for (int i = 0; i < 4; i++) { int u = 0; t1[i] = (imm >> (4 + i)) & 1 ? sf_mul(&SF32, a->d[i], b->d[i], rc, daz, fl, &u) : 0; }
    int u = 0;
    uint64_t t2 = sf_add(&SF32, t1[0], t1[1], 0, rc, daz, fl, &u), t3 = sf_add(&SF32, t1[2], t1[3], 0, rc, daz, fl, &u);
    uint64_t t4 = sf_add(&SF32, t2, t3, 0, rc, daz, fl, &u);
    for (int i = 0; i < 4; i++) r->d[i] = (imm >> i) & 1 ? (uint32_t)t4 : 0;
}
static void vr_dppd(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)care; *unc = 1;
    uint64_t t1[2];
    for (int i = 0; i < 2; i++) { int u = 0; t1[i] = (imm >> (4 + i)) & 1 ? sf_mul(&SF64, a->q[i], b->q[i], rc, daz, fl, &u) : 0; }
    int u = 0;
    uint64_t t2 = sf_add(&SF64, t1[0], t1[1], 0, rc, daz, fl, &u);
    for (int i = 0; i < 2; i++) r->q[i] = (imm >> i) & 1 ? t2 : 0;
}
static T_NOINLINE void fop_dppd_valid(V128 *d, const V128 *s, uint32_t *mx, int imm) { fop_dppd(d, s, mx, (imm & 3) | ((imm & 0xC) << 2)); }
static void vr_dppd_valid(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    vr_dppd(r, a, b, rc, daz, fl, unc, care, (imm & 3) | ((imm & 0xC) << 2));
}
TEST(sse41_dpps_dppd) {
    NEED(F_SSE41);
    /* NaN inputs are excluded: with several NaN lanes the winning payload depends on the add tree's
     * operand order, which only the pseudo-code fixes; infinities (-> default NaN) are kept. */
    static const fop_t t[] = {
        { "dpps", F_SSE41, fop_dpps, vr_dpps, GK_NONAN, GK_NONAN, &SF32, &SF32, 0, 255, 0xF, 12, 1 },
        { "dppd", F_SSE41, fop_dppd_valid, vr_dppd_valid, GK_NONAN, GK_NONAN, &SF64, &SF64, 0, 15, 0xF, 150, 1 } };
    run_fops(t, T_COUNTOF(t), 0x5EA051);
    /* known answer: (1,2,3,4).(5,6,7,8) = 70, all lanes */
    V128 a = { .d = { 0x3F800000, 0x40000000, 0x40400000, 0x40800000 } }, b = { .d = { 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000 } };
    uint32_t mx = 0x1F80; V128 d = a;
    fop_dpps(&d, &b, &mx, 0xFF);
    CHECKF(d.d[0] == 0x428C0000 && d.d[1] == 0x428C0000 && d.d[2] == 0x428C0000 && d.d[3] == 0x428C0000, "dpps 0xFF: %s", hx(&d, 4));
    d = a; fop_dpps(&d, &b, &mx, 0x31);
    CHECKF(d.d[0] == 0x41880000 && d.d[1] == 0 && d.d[2] == 0 && d.d[3] == 0, "dpps 0x31 = 17 in lane 0: %s", hx(&d, 4));
}

/* ---- RCPPS / RSQRTPS (+ scalar): approximation, |rel err| <= 1.5*2^-12 (SDM RCPPS/RSQRTPS) */
#define FOPU(nm, insn) static T_NOINLINE void fop_##nm(V128 *d, const V128 *s, uint32_t *mx, int imm) { (void)imm; \
    MX_RUN(insn " %3, %0", : "+x"(d->v), "=m"(sv_), "+m"(*mx) : "x"(s->v)); }
FOPU(rcpps, "rcpps") FOPU(rsqrtps, "rsqrtps") FOPU(rcpss, "rcpss") FOPU(rsqrtss, "rsqrtss")
static double u2d32(uint32_t u) { float f; memcpy(&f, &u, 4); return (double)f; }

TEST(sse_rcpps_rsqrtps_accuracy) {
    NEED(F_SSE);
    const double eb = 1.5 / 4096.0;
    t_rng r; t_rng_seed(&r, 0x5EA061);
    double worst_rcp = 0, worst_rsq = 0;
    vbegin();
    for (int it = 0; it < 20000; it++) {
        V128 b, d = { 0 }, e = { 0 };
        for (int i = 0; i < 4; i++) {
            /* normal inputs whose reciprocal is a normal (exponent 2..251) */
            uint32_t ex = 1 + (uint32_t)t_rng_range(&r, 253), m = t_rng_u32(&r) & 0x7FFFFF;
            if (it < 2048 && i == 0) { ex = 127; m = (uint32_t)it << 12; }   /* sweep the mantissa table */
            b.d[i] = (t_rng_range(&r, 2) ? 0x80000000u : 0) | (ex << 23) | m;
        }
        uint32_t mx = 0x1F80; fop_rcpps(&d, &b, &mx, 0);
        uint32_t mx2 = 0x1F80; V128 bp = b; for (int i = 0; i < 4; i++) bp.d[i] &= 0x7FFFFFFF;
        fop_rsqrtps(&e, &bp, &mx2, 0);
        for (int i = 0; i < 4; i++) {
                uint32_t ax = b.d[i] & 0x7FFFFFFF;
            if (ax >= 0x7E7FE800u) { VCHECK((d.d[i] & 0x7FFFFFFF) == 0 && (d.d[i] >> 31) == (b.d[i] >> 31), "rcpps(0x%08x)=0x%08x want signed 0 (tiny result flushed)", b.d[i], d.d[i]); }
            else if (ax <= 0x00800C01u) { VCHECK(d.d[i] == ((b.d[i] & 0x80000000u) | 0x7F800000u), "rcpps(0x%08x)=0x%08x want signed inf", b.d[i], d.d[i]); }
            else if (ax >= 0x7E000000u || ax < 0x00801000u) { /* between the documented guarantees: not pinned */ }
            else {
                double x = u2d32(b.d[i]), y = u2d32(d.d[i]), err = y * x - 1.0;
                if (err < 0) err = -err;
                if (err > worst_rcp) worst_rcp = err;
                VCHECK(err <= eb && (d.d[i] >> 31) == (b.d[i] >> 31), "rcpps(0x%08x)=0x%08x rel err %.3g > 1.5*2^-12", b.d[i], d.d[i], err);
            }
            double x = u2d32(bp.d[i]), y = u2d32(e.d[i]), p = y * y * x;
            double lo = (1 - eb) * (1 - eb), hi = (1 + eb) * (1 + eb);
            double err = y * __builtin_sqrt(x) - 1.0; if (err < 0) err = -err;
            if (err > worst_rsq) worst_rsq = err;
            VCHECK(p >= lo && p <= hi && !(e.d[i] >> 31), "rsqrtps(0x%08x)=0x%08x rel err %.3g > 1.5*2^-12", bp.d[i], e.d[i], err);
        }
        VCHECK((mx & 0x3F) == 0 && (mx2 & 0x3F) == 0, "rcpps/rsqrtps raised MXCSR flags 0x%02x/0x%02x (they signal none)", mx & 0x3F, mx2 & 0x3F);
    }
    VEND("rcpps/rsqrtps accuracy");
    t_info("rcpps_max_rel_err", "%.4g (bound %.4g)", worst_rcp, eb);
    t_info("rsqrtps_max_rel_err", "%.4g (bound %.4g)", worst_rsq, eb);
    /* scalar forms keep the upper lanes of the destination */
    V128 a = { .d = { 1, 0x11111111, 0x22222222, 0x33333333 } }, b = { .d = { 0x40800000, 0x5, 0x6, 0x7 } }, d = a;
    uint32_t mx = 0x1F80; fop_rcpss(&d, &b, &mx, 0);
    CHECKF(d.d[1] == a.d[1] && d.d[2] == a.d[2] && d.d[3] == a.d[3], "rcpss upper lanes: %s", hx(&d, 4));
    CHECKF(__builtin_fabs(u2d32(d.d[0]) * 4.0 - 1.0) <= eb, "rcpss(4) = 0x%08x", d.d[0]);
    d = a; fop_rsqrtss(&d, &b, &mx, 0);
    CHECKF(d.d[1] == a.d[1] && d.d[2] == a.d[2] && d.d[3] == a.d[3], "rsqrtss upper lanes: %s", hx(&d, 4));
    CHECKF(__builtin_fabs(u2d32(d.d[0]) * 2.0 - 1.0) <= eb, "rsqrtss(4) = 0x%08x", d.d[0]);
}
TEST(sse_rcp_rsqrt_special_values) {
    NEED(F_SSE);
    /* RCPPS: +-0 -> +-inf, +-inf -> +-0, denormal -> +-inf, NaN -> quieted NaN.
     * RSQRTPS: +-0 -> +-inf, +inf -> +0, negative (incl. -inf) -> indefinite, denormal -> +-inf, NaN -> quiet NaN. */
    static const struct { uint32_t x, rcp, rsq; } t[] = {
        { 0x00000000, 0x7F800000, 0x7F800000 }, { 0x80000000, 0xFF800000, 0xFF800000 },
        { 0x7F800000, 0x00000000, 0x00000000 }, { 0xFF800000, 0x80000000, 0xFFC00000 },
        { 0x00000001, 0x7F800000, 0x7F800000 }, { 0x807FFFFF, 0xFF800000, 0xFF800000 },
        { 0x7FC12345, 0x7FC12345, 0x7FC12345 }, { 0x7F812345, 0x7FC12345, 0x7FC12345 },
        { 0xFFC00000, 0xFFC00000, 0xFFC00000 }, { 0xBF800000, 0xBF800000, 0xFFC00000 },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        V128 b, d = { 0 }, e = { 0 };
        for (int l = 0; l < 4; l++) b.d[l] = t[i].x;
        uint32_t mx = 0x1F80, mx2 = 0x1F80;
        fop_rcpps(&d, &b, &mx, 0); fop_rsqrtps(&e, &b, &mx2, 0);
        if (t[i].x == 0xBF800000) {   /* rcp(-1) is approximate: only the bound applies */
            CHECKF(__builtin_fabs(u2d32(d.d[0]) * -1.0 - 1.0) <= 1.5 / 4096, "rcpps(-1)=0x%08x", d.d[0]);
        } else CHECKF(d.d[0] == t[i].rcp && d.d[3] == t[i].rcp, "rcpps(0x%08x)=0x%08x want 0x%08x", t[i].x, d.d[0], t[i].rcp);
        CHECKF(e.d[0] == t[i].rsq && e.d[3] == t[i].rsq, "rsqrtps(0x%08x)=0x%08x want 0x%08x", t[i].x, e.d[0], t[i].rsq);
        CHECKF((mx & 0x3F) == 0 && (mx2 & 0x3F) == 0, "rcp/rsqrt(0x%08x) flags 0x%02x/0x%02x want none", t[i].x, mx & 0x3F, mx2 & 0x3F);
    }
}

/* ================================================================ conversions */
static void vr_cvtdq2ps(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)a; (void)daz; (void)unc; (void)care; (void)imm;
    for (int i = 0; i < 4; i++) r->d[i] = (uint32_t)sf_from_i64(&SF32, b->sd[i], rc, fl);
}
static void vr_cvtps2dq_g(V128 *r, const V128 *b, int rc, int daz, uint32_t *fl) {
    int u = 0; for (int i = 0; i < 4; i++) r->d[i] = (uint32_t)sf_to_int(&SF32, b->d[i], 32, rc, daz, fl, &u);
}
static void vr_cvtps2dq(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) { (void)a; (void)unc; (void)care; (void)imm; vr_cvtps2dq_g(r, b, rc, daz, fl); }
static void vr_cvttps2dq(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) { (void)a; (void)rc; (void)unc; (void)care; (void)imm; vr_cvtps2dq_g(r, b, RC_RZ, daz, fl); }
static void vr_cvtpd2dq_g(V128 *r, const V128 *b, int rc, int daz, uint32_t *fl) {
    int u = 0; for (int i = 0; i < 2; i++) r->d[i] = (uint32_t)sf_to_int(&SF64, b->q[i], 32, rc, daz, fl, &u);
    r->q[1] = 0;
}
static void vr_cvtpd2dq(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) { (void)a; (void)unc; (void)care; (void)imm; vr_cvtpd2dq_g(r, b, rc, daz, fl); }
static void vr_cvttpd2dq(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) { (void)a; (void)rc; (void)unc; (void)care; (void)imm; vr_cvtpd2dq_g(r, b, RC_RZ, daz, fl); }
static void vr_cvtdq2pd(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)a; (void)daz; (void)unc; (void)care; (void)imm;
    int32_t s0 = b->sd[0], s1 = b->sd[1];
    r->q[0] = sf_from_i64(&SF64, s0, rc, fl); r->q[1] = sf_from_i64(&SF64, s1, rc, fl);
}
static void vr_cvtps2pd(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)a; (void)care; (void)imm;
    uint32_t s0 = b->d[0], s1 = b->d[1];
    r->q[0] = sf_convert(&SF32, &SF64, s0, rc, daz, fl, unc); r->q[1] = sf_convert(&SF32, &SF64, s1, rc, daz, fl, unc);
}
static void vr_cvtpd2ps(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)a; (void)care; (void)imm;
    uint64_t s0 = b->q[0], s1 = b->q[1];
    r->d[0] = (uint32_t)sf_convert(&SF64, &SF32, s0, rc, daz, fl, unc); r->d[1] = (uint32_t)sf_convert(&SF64, &SF32, s1, rc, daz, fl, unc);
    r->q[1] = 0;
}
static void vr_cvtss2sd(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)care; (void)imm; *r = *a; r->q[0] = sf_convert(&SF32, &SF64, b->d[0], rc, daz, fl, unc);
}
static void vr_cvtsd2ss(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) {
    (void)care; (void)imm; *r = *a; r->d[0] = (uint32_t)sf_convert(&SF64, &SF32, b->q[0], rc, daz, fl, unc);
}
FOP(cvtdq2ps, "cvtdq2ps") FOP(cvtps2dq, "cvtps2dq") FOP(cvttps2dq, "cvttps2dq") FOP(cvtpd2dq, "cvtpd2dq") FOP(cvttpd2dq, "cvttpd2dq")
FOP(cvtdq2pd, "cvtdq2pd") FOP(cvtps2pd, "cvtps2pd") FOP(cvtpd2ps, "cvtpd2ps") FOP(cvtss2sd, "cvtss2sd") FOP(cvtsd2ss, "cvtsd2ss")

/* GPR <-> scalar: results placed in d->q[0] (GPR result zero-extended, rest of d cleared) */
#define CVT_TO_GPR(nm, insn, T) static T_NOINLINE void fop_##nm(V128 *d, const V128 *s, uint32_t *mx, int imm) { (void)imm; T g; \
    MX_RUN(insn " %3, %0", : "=&r"(g), "=m"(sv_), "+m"(*mx) : "x"(s->v)); memset(d, 0, sizeof *d); d->q[0] = (uint64_t)(uint32_t)g; \
    if (sizeof(T) == 8) d->q[0] = (uint64_t)g; }
CVT_TO_GPR(cvtss2si, "cvtss2si", int32_t) CVT_TO_GPR(cvttss2si, "cvttss2si", int32_t)
CVT_TO_GPR(cvtsd2si, "cvtsd2si", int32_t) CVT_TO_GPR(cvttsd2si, "cvttsd2si", int32_t)
#define CVT_FROM_GPR(nm, insn, T, F) static T_NOINLINE void fop_##nm(V128 *d, const V128 *s, uint32_t *mx, int imm) { (void)imm; T g = (T)s->F; \
    MX_RUN(insn " %3, %0", : "+x"(d->v), "=m"(sv_), "+m"(*mx) : "r"(g)); }
CVT_FROM_GPR(cvtsi2ss, "cvtsi2ss", int32_t, sd[0]) CVT_FROM_GPR(cvtsi2sd, "cvtsi2sd", int32_t, sd[0])
#ifdef T_X64
CVT_TO_GPR(cvtss2si64, "cvtss2si", int64_t) CVT_TO_GPR(cvttss2si64, "cvttss2si", int64_t)
CVT_TO_GPR(cvtsd2si64, "cvtsd2si", int64_t) CVT_TO_GPR(cvttsd2si64, "cvttsd2si", int64_t)
CVT_FROM_GPR(cvtsi2ss64, "cvtsi2ssq", int64_t, sq[0]) CVT_FROM_GPR(cvtsi2sd64, "cvtsi2sdq", int64_t, sq[0])
#endif
#define VR_TOGPR(nm, F, bits, RCX) static void vr_##nm(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) { \
    (void)a; (void)care; (void)imm; (void)rc; memset(r, 0, sizeof *r); r->q[0] = sf_to_int(&F, lget(b, &F, 0), bits, RCX, daz, fl, unc); }
VR_TOGPR(cvtss2si, SF32, 32, rc) VR_TOGPR(cvttss2si, SF32, 32, RC_RZ) VR_TOGPR(cvtsd2si, SF64, 32, rc) VR_TOGPR(cvttsd2si, SF64, 32, RC_RZ)
VR_TOGPR(cvtss2si64, SF32, 64, rc) VR_TOGPR(cvttss2si64, SF32, 64, RC_RZ) VR_TOGPR(cvtsd2si64, SF64, 64, rc) VR_TOGPR(cvttsd2si64, SF64, 64, RC_RZ)
#define VR_FROMGPR(nm, F, v) static void vr_##nm(V128 *r, const V128 *a, const V128 *b, int rc, int daz, uint32_t *fl, int *unc, V128 *care, int imm) { \
    (void)care; (void)imm; (void)daz; (void)unc; *r = *a; lset(r, &F, 0, sf_from_i64(&F, v, rc, fl)); }
VR_FROMGPR(cvtsi2ss, SF32, b->sd[0]) VR_FROMGPR(cvtsi2sd, SF64, b->sd[0]) VR_FROMGPR(cvtsi2ss64, SF32, b->sq[0]) VR_FROMGPR(cvtsi2sd64, SF64, b->sq[0])

#define NODAZ 0xF
TEST(sse2_cvtdq2ps_cvtps2dq_cvttps2dq) {
    NEED(F_SSE2);
    static const fop_t t[] = {
        { "cvtdq2ps", F_SSE2, fop_cvtdq2ps, vr_cvtdq2ps, 0, 0, &SF32, NULL, 0, 0, NODAZ, 1500, 0 },
        { "cvtps2dq", F_SSE2, fop_cvtps2dq, vr_cvtps2dq, GK_ANY, GK_CVT, &SF32, &SF32, 0, 0, NODAZ, 1500, 0 },
        { "cvttps2dq", F_SSE2, fop_cvttps2dq, vr_cvttps2dq, GK_ANY, GK_CVT, &SF32, &SF32, 0, 0, NODAZ, 1500, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA071);
}
TEST(sse2_cvtpd2dq_cvttpd2dq_cvtdq2pd) {
    NEED(F_SSE2);
    static const fop_t t[] = {
        { "cvtpd2dq", F_SSE2, fop_cvtpd2dq, vr_cvtpd2dq, GK_ANY, GK_CVT, &SF64, &SF64, 0, 0, NODAZ, 1500, 0 },
        { "cvttpd2dq", F_SSE2, fop_cvttpd2dq, vr_cvttpd2dq, GK_ANY, GK_CVT, &SF64, &SF64, 0, 0, NODAZ, 1500, 0 },
        { "cvtdq2pd", F_SSE2, fop_cvtdq2pd, vr_cvtdq2pd, 0, 0, &SF64, NULL, 0, 0, NODAZ, 600, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA072);
}
TEST(sse2_cvtps2pd_cvtpd2ps) {
    NEED(F_SSE2);
    static const fop_t t[] = {
        { "cvtps2pd", F_SSE2, fop_cvtps2pd, vr_cvtps2pd, GK_ANY, GK_ANY, &SF32, &SF32, 0, 0, ALLM, 1500, 0 },
        { "cvtpd2ps", F_SSE2, fop_cvtpd2ps, vr_cvtpd2ps, GK_ANY, GK_ANY, &SF64, &SF64, 0, 0, ALLM, 1500, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA073);
}
TEST(sse2_cvtss2sd_cvtsd2ss_preserve_upper) {
    NEED(F_SSE2);
    static const fop_t t[] = {
        { "cvtss2sd", F_SSE2, fop_cvtss2sd, vr_cvtss2sd, GK_ANY, GK_ANY, &SF64, &SF32, 0, 0, ALLM, 1500, 0 },
        { "cvtsd2ss", F_SSE2, fop_cvtsd2ss, vr_cvtsd2ss, GK_ANY, GK_ANY, &SF32, &SF64, 0, 0, ALLM, 1500, 0 } };
    run_fops(t, T_COUNTOF(t), 0x5EA074);
}
TEST(sse_cvtsi2ss_cvtsi2sd) {
    NEED(F_SSE2);
    static const fop_t t[] = {
        { "cvtsi2ss r32", F_SSE, fop_cvtsi2ss, vr_cvtsi2ss, GK_ANY, 0, &SF32, NULL, 0, 0, NODAZ, 1500, 0 },
        { "cvtsi2sd r32", F_SSE2, fop_cvtsi2sd, vr_cvtsi2sd, GK_ANY, 0, &SF64, NULL, 0, 0, NODAZ, 800, 0 },
#ifdef T_X64
        { "cvtsi2ss r64", F_SSE, fop_cvtsi2ss64, vr_cvtsi2ss64, GK_ANY, 0, &SF32, NULL, 0, 0, NODAZ, 1500, 0 },
        { "cvtsi2sd r64", F_SSE2, fop_cvtsi2sd64, vr_cvtsi2sd64, GK_ANY, 0, &SF64, NULL, 0, 0, NODAZ, 1500, 0 },
#endif
    };
    run_fops(t, T_COUNTOF(t), 0x5EA075);
#ifndef T_X64
    t_note("REX.W (64-bit GPR) forms are x64-only: not applicable to this x86 build");
#endif
}
TEST(sse_cvtss2si_cvtsd2si_indefinite) {
    NEED(F_SSE2);
    static const fop_t t[] = {
        { "cvtss2si r32", F_SSE, fop_cvtss2si, vr_cvtss2si, 0, GK_CVT, NULL, &SF32, 0, 0, NODAZ, 1500, 0 },
        { "cvttss2si r32", F_SSE, fop_cvttss2si, vr_cvttss2si, 0, GK_CVT, NULL, &SF32, 0, 0, NODAZ, 1500, 0 },
        { "cvtsd2si r32", F_SSE2, fop_cvtsd2si, vr_cvtsd2si, 0, GK_CVT, NULL, &SF64, 0, 0, NODAZ, 1500, 0 },
        { "cvttsd2si r32", F_SSE2, fop_cvttsd2si, vr_cvttsd2si, 0, GK_CVT, NULL, &SF64, 0, 0, NODAZ, 1500, 0 },
#ifdef T_X64
        { "cvtss2si r64", F_SSE, fop_cvtss2si64, vr_cvtss2si64, 0, GK_CVT, NULL, &SF32, 0, 0, NODAZ, 1500, 0 },
        { "cvttss2si r64", F_SSE, fop_cvttss2si64, vr_cvttss2si64, 0, GK_CVT, NULL, &SF32, 0, 0, NODAZ, 1500, 0 },
        { "cvtsd2si r64", F_SSE2, fop_cvtsd2si64, vr_cvtsd2si64, 0, GK_CVT, NULL, &SF64, 0, 0, NODAZ, 1500, 0 },
        { "cvttsd2si r64", F_SSE2, fop_cvttsd2si64, vr_cvttsd2si64, 0, GK_CVT, NULL, &SF64, 0, 0, NODAZ, 1500, 0 },
#endif
    };
    run_fops(t, T_COUNTOF(t), 0x5EA076);
}
TEST(sse_cvt_known_answers) {
    NEED(F_SSE2);
    static const struct { const char *op; uint64_t x; uint32_t mx; uint64_t want; uint32_t fl; } t[] = {
        { "cvtps2dq", 0x40200000, MXRN, 2, FL_PE },                 /* 2.5 -> 2 (even) */
        { "cvtps2dq", 0x40600000, MXRN, 4, FL_PE },                 /* 3.5 -> 4 */
        { "cvtps2dq", 0xC0200000, MXRN, (uint32_t)-2, FL_PE },
        { "cvtps2dq", 0xC0200000, MXRD, (uint32_t)-3, FL_PE },
        { "cvtps2dq", 0x40200000, MXRU, 3, FL_PE },
        { "cvtps2dq", 0x4EFFFFFF, MXRN, 0x7FFFFF80, 0 },            /* largest float < 2^31 */
        { "cvtps2dq", 0x4F000000, MXRN, 0x80000000, FL_IE },        /* 2^31: out of range -> indefinite */
        { "cvtps2dq", 0xCF000000, MXRN, 0x80000000, 0 },            /* -2^31 exactly fits */
        { "cvtps2dq", 0xCF000001, MXRN, 0x80000000, FL_IE },
        { "cvtps2dq", 0x7FC00000, MXRN, 0x80000000, FL_IE },        /* NaN */
        { "cvtps2dq", 0xFF800000, MXRN, 0x80000000, FL_IE },        /* -inf */
        { "cvtps2dq", 0x00000001, MXRU, 1, FL_PE },                 /* denormal, no DE for CVT*2DQ */
        { "cvtps2dq", 0x80000001, MXRD, (uint32_t)-1, FL_PE },
        { "cvttps2dq", 0x3FFFFFFF, MXRN, 1, FL_PE },                /* 1.99999 truncates */
        { "cvttps2dq", 0xBFFFFFFF, MXRU, (uint32_t)-1, FL_PE },     /* truncation ignores RC */
        { "cvttps2dq", 0x4F000000, MXRN, 0x80000000, FL_IE },
        { "cvtpd2dq", 0x41DFFFFFFFE00000ull, MXRN, 0x80000000, FL_IE },  /* 2147483647.5 -> RN (even) 2^31: out of range */
        { "cvtpd2dq", 0x41DFFFFFFFE00000ull, MXRD, 0x7FFFFFFF, FL_PE },
        { "cvtpd2dq", 0x41DFFFFFFFC00000ull, MXRN, 0x7FFFFFFF, 0 },      /* 2147483647.0 */
        { "cvtpd2dq", 0xC1E0000000000000ull, MXRN, 0x80000000, 0 },      /* -2^31 */
        { "cvtpd2dq", 0xC1E0000000100000ull, MXRN, 0x80000000, FL_PE }, /* -2147483648.5 -> even -2^31 */
        { "cvtpd2dq", 0xC1E0000000200000ull, MXRN, 0x80000000, FL_IE }, /* -2147483649 */
        { "cvttpd2dq", 0x41DFFFFFFFE00000ull, MXRN, 0x7FFFFFFF, FL_PE },
        { "cvtdq2ps", 0x7FFFFFFF, MXRN, 0x4F000000, FL_PE },             /* 2^31-1 rounds to 2^31 */
        { "cvtdq2ps", 0x7FFFFFFF, MXRZ, 0x4EFFFFFF, FL_PE },
        { "cvtdq2ps", 0x80000000, MXRN, 0xCF000000, 0 },
        { "cvtdq2ps", 0x01000001, MXRN, 0x4B800000, FL_PE },             /* 2^24+1: tie to even */
        { "cvtdq2ps", 0x01000003, MXRN, 0x4B800002, FL_PE },             /* 2^24+3: tie up to even */
        { "cvtpd2ps", 0x3FF0000010000000ull, MXRN, 0x3F800000, FL_PE },  /* 1 + 2^-24: tie */
        { "cvtpd2ps", 0x47EFFFFFF0000000ull, MXRN, 0x7F800000, FL_OE | FL_PE }, /* rounds above max float */
        { "cvtpd2ps", 0x7FF0000000000001ull, MXRN, 0x7FC00000, FL_IE }, /* SNaN with low payload: quiet NaN */
        { "cvtpd2ps", 0xFFF4000020000000ull, MXRN, 0xFFE00001, FL_IE }, /* payload top bits kept */
        { "cvtpd2ps", 0x36A0000000000000ull, MXRN, 0x00000001, 0 },     /* 2^-149: exact denormal, no UE */
        { "cvtpd2ps", 0x3690000000000000ull, MXRN, 0x00000000, FL_UE | FL_PE }, /* 2^-150: tie -> 0 */
        { "cvtpd2ps", 0x3690000000000000ull, MXRU, 0x00000001, FL_UE | FL_PE },
        { "cvtpd2ps", 0x3698000000000000ull, MXRN, 0x00000001, FL_UE | FL_PE }, /* 1.5 * 2^-150 -> 2^-149 */
        { "cvtps2pd", 0x7F800001, MXRN, 0x7FF8000020000000ull, FL_IE },  /* SNaN quieted, payload << 29 */
        { "cvtps2pd", 0x00000001, MXRN, 0x36A0000000000000ull, FL_DE },  /* 2^-149 exact, DE */
        { "cvtps2pd", 0x00000001, MXDAZ, 0x0000000000000000ull, 0 },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        if ((t[i].mx & 0x40) && !daz_supported()) continue;
        V128 b, d = { 0 };
        int dsrc = !strncmp(t[i].op, "cvtpd", 5) || !strncmp(t[i].op, "cvttpd", 6);
        if (dsrc) { b.q[0] = b.q[1] = t[i].x; } else { for (int l = 0; l < 4; l++) b.d[l] = (uint32_t)t[i].x; }
        fop_fn fn = !strcmp(t[i].op, "cvtps2dq") ? fop_cvtps2dq : !strcmp(t[i].op, "cvttps2dq") ? fop_cvttps2dq :
                    !strcmp(t[i].op, "cvtpd2dq") ? fop_cvtpd2dq : !strcmp(t[i].op, "cvttpd2dq") ? fop_cvttpd2dq :
                    !strcmp(t[i].op, "cvtdq2ps") ? fop_cvtdq2ps : !strcmp(t[i].op, "cvtpd2ps") ? fop_cvtpd2ps : fop_cvtps2pd;
        uint32_t mx = t[i].mx;
        fn(&d, &b, &mx, 0);
        uint64_t got = !strcmp(t[i].op, "cvtps2pd") ? d.q[0] : d.d[0];
        uint64_t want = t[i].want; uint32_t wfl = t[i].fl;
        CHECKF(got == want && (mx & 0x3F) == wfl, "#%u %s(0x%llx) mxcsr 0x%04x = 0x%llx flags 0x%02x, want 0x%llx flags 0x%02x",
               (unsigned)i, t[i].op, (unsigned long long)t[i].x, t[i].mx, (unsigned long long)got, mx & 0x3F, (unsigned long long)want, wfl);
        if (dsrc && strcmp(t[i].op, "cvtpd2ps") && strcmp(t[i].op, "cvtps2pd")) CHECKF(d.q[1] == 0, "%s upper 64 bits not zeroed: %s", t[i].op, hx(&d, 8));
        if (!strcmp(t[i].op, "cvtpd2ps")) CHECKF(d.q[1] == 0, "cvtpd2ps upper 64 bits not zeroed: %s", hx(&d, 8));
    }
    /* 64-bit integer indefinite and the 2^63 boundary (x64 only) */
#ifdef T_X64
    {
        V128 b = { 0 }, d; uint32_t mx;
        b.d[0] = 0xDF000000; mx = MXRN; fop_cvtss2si64(&d, &b, &mx, 0);   /* -2^63 fits */
        CHECKF(d.q[0] == 0x8000000000000000ull && (mx & 0x3F) == 0, "cvtss2si64(-2^63)=0x%llx fl 0x%x", (unsigned long long)d.q[0], mx & 0x3F);
        b.d[0] = 0x5F000000; mx = MXRN; fop_cvtss2si64(&d, &b, &mx, 0);   /* 2^63 does not */
        CHECKF(d.q[0] == 0x8000000000000000ull && (mx & 0x3F) == FL_IE, "cvtss2si64(2^63)=0x%llx fl 0x%x", (unsigned long long)d.q[0], mx & 0x3F);
        b.q[0] = 0x7FFFFFFFFFFFFFFFull; mx = MXRN; fop_cvtsi2sd64(&d, &b, &mx, 0);
        CHECKF(d.q[0] == 0x43E0000000000000ull && (mx & 0x3F) == FL_PE, "cvtsi2sd64(2^63-1)=0x%llx fl 0x%x", (unsigned long long)d.q[0], mx & 0x3F);
    }
#endif
}

/* ================================================================ integer harness */
typedef void (*iop_fn)(V128 *d, const V128 *s, int imm);
typedef void (*iref_fn)(V128 *r, const V128 *a, const V128 *b, int imm);
enum { GB_INT, GB_COUNT };
typedef struct { const char *name; int feat; iop_fn op; iref_fn ref; int gen_b; int imm_lo, imm_hi; int iters; } iop_t;

static void run_iop(const iop_t *o, uint64_t seed) {
    if (!cpu_has(o->feat)) { t_note("%s: %s not usable, not tested", o->name, cpu_feature_name(o->feat)); return; }
    t_rng r; t_rng_seed(&r, seed);
    vbegin();
    for (int imm = o->imm_lo; imm <= o->imm_hi; imm++)
        for (int it = 0; it < o->iters; it++) {
            V128 a, b, d, w;
            gen_int(&r, &a);
            if (o->gen_b == GB_COUNT) gen_count(&r, &b);
            else if (t_rng_range(&r, 8) == 0) b = a;          /* equal operands */
            else gen_int(&r, &b);
            d = a;
            o->op(&d, &b, imm);
            o->ref(&w, &a, &b, imm);
            VCHECK(!memcmp(&d, &w, 16), "%s imm=0x%02x: a=%s b=%s got=%s want=%s", o->name, imm,
                   hx(&a, 2), hx(&b, 2), hx(&d, 2), hx(&w, 2));
        }
    VEND(o->name);
}
static void run_iops(const iop_t *t, size_t n, uint64_t seed) { for (size_t i = 0; i < n; i++) run_iop(&t[i], seed + i * 104729); }

#define IOP(nm, insn) static T_NOINLINE void op_##nm(V128 *d, const V128 *s, int imm) { (void)imm; \
    __asm__ volatile(insn " %1, %0" : "+x"(d->v) : "x"(s->v)); }
#define IOPI_C(n) case n: __asm__ volatile(INSN_ " %2, %1, %0" : "+x"(d->v) : "x"(s->v), "i"(n)); break;
#define IOPI(nm, SWEEP) static T_NOINLINE void op_##nm(V128 *d, const V128 *s, int imm) { switch (imm) { SWEEP(IOPI_C) default: break; } }
#define IOPS_C(n) case n: __asm__ volatile(INSN_ " %1, %0" : "+x"(d->v) : "i"(n)); break;
#define IOPS(nm, SWEEP) static T_NOINLINE void op_##nm(V128 *d, const V128 *s, int imm) { (void)s; switch (imm) { SWEEP(IOPS_C) default: break; } }

#define LANES_U(nm, N, F, expr) static void ref_##nm(V128 *r, const V128 *a, const V128 *b, int imm) { V128 t; (void)imm; \
    for (int i = 0; i < N; i++) { uint64_t x = a->F[i], y = b->F[i]; (void)x; (void)y; t.F[i] = (expr); } *r = t; }
#define LANES_S(nm, N, F, expr) static void ref_##nm(V128 *r, const V128 *a, const V128 *b, int imm) { V128 t; (void)imm; \
    for (int i = 0; i < N; i++) { int64_t x = a->F[i], y = b->F[i]; (void)x; (void)y; t.F[i] = (expr); } *r = t; }

static inline int64_t clampi(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }
#define SAT_S8(v)  clampi((v), -128, 127)
#define SAT_U8(v)  clampi((v), 0, 255)
#define SAT_S16(v) clampi((v), -32768, 32767)
#define SAT_U16(v) clampi((v), 0, 65535)
#define MN(x, y) ((x) < (y) ? (x) : (y))
#define MX(x, y) ((x) > (y) ? (x) : (y))
#define ABS(x) ((x) < 0 ? -(x) : (x))

/* SSE2 add/sub (wrap-around) */
LANES_U(paddb, 16, b, x + y) LANES_U(paddw, 8, w, x + y) LANES_U(paddd, 4, d, x + y) LANES_U(paddq, 2, q, x + y)
LANES_U(psubb, 16, b, x - y) LANES_U(psubw, 8, w, x - y) LANES_U(psubd, 4, d, x - y) LANES_U(psubq, 2, q, x - y)
IOP(paddb, "paddb") IOP(paddw, "paddw") IOP(paddd, "paddd") IOP(paddq, "paddq")
IOP(psubb, "psubb") IOP(psubw, "psubw") IOP(psubd, "psubd") IOP(psubq, "psubq")
/* saturating */
LANES_S(paddsb, 16, sb, SAT_S8(x + y)) LANES_S(paddsw, 8, sw, SAT_S16(x + y))
LANES_S(psubsb, 16, sb, SAT_S8(x - y)) LANES_S(psubsw, 8, sw, SAT_S16(x - y))
LANES_S(paddusb, 16, b, SAT_U8(x + y)) LANES_S(paddusw, 8, w, SAT_U16(x + y))
LANES_S(psubusb, 16, b, SAT_U8(x - y)) LANES_S(psubusw, 8, w, SAT_U16(x - y))
IOP(paddsb, "paddsb") IOP(paddsw, "paddsw") IOP(psubsb, "psubsb") IOP(psubsw, "psubsw")
IOP(paddusb, "paddusb") IOP(paddusw, "paddusw") IOP(psubusb, "psubusb") IOP(psubusw, "psubusw")
/* multiply */
LANES_S(pmullw, 8, sw, (int16_t)(x * y)) LANES_S(pmulhw, 8, sw, (x * y) >> 16) LANES_U(pmulhuw, 8, w, (x * y) >> 16)
static void ref_pmuludq(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    t.q[0] = (uint64_t)a->d[0] * b->d[0]; t.q[1] = (uint64_t)a->d[2] * b->d[2]; *r = t; }
static void ref_pmaddwd(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    for (int i = 0; i < 4; i++) t.d[i] = (uint32_t)((int64_t)a->sw[2 * i] * b->sw[2 * i] + (int64_t)a->sw[2 * i + 1] * b->sw[2 * i + 1]);
    *r = t; }
IOP(pmullw, "pmullw") IOP(pmulhw, "pmulhw") IOP(pmulhuw, "pmulhuw") IOP(pmuludq, "pmuludq") IOP(pmaddwd, "pmaddwd")
/* average, SAD */
LANES_U(pavgb, 16, b, (x + y + 1) >> 1) LANES_U(pavgw, 8, w, (x + y + 1) >> 1)
static void ref_psadbw(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    for (int h = 0; h < 2; h++) { uint64_t s = 0; for (int i = 0; i < 8; i++) { int dd = a->b[8 * h + i] - b->b[8 * h + i]; s += (uint64_t)ABS(dd); } t.q[h] = s; } *r = t; }
IOP(pavgb, "pavgb") IOP(pavgw, "pavgw") IOP(psadbw, "psadbw")
/* min/max (SSE2 subset) and SSE4.1 full set */
LANES_U(pminub, 16, b, MN(x, y)) LANES_U(pmaxub, 16, b, MX(x, y)) LANES_S(pminsw, 8, sw, MN(x, y)) LANES_S(pmaxsw, 8, sw, MX(x, y))
LANES_S(pminsb, 16, sb, MN(x, y)) LANES_S(pmaxsb, 16, sb, MX(x, y)) LANES_U(pminuw, 8, w, MN(x, y)) LANES_U(pmaxuw, 8, w, MX(x, y))
LANES_S(pminsd, 4, sd, MN(x, y)) LANES_S(pmaxsd, 4, sd, MX(x, y)) LANES_U(pminud, 4, d, MN(x, y)) LANES_U(pmaxud, 4, d, MX(x, y))
IOP(pminub, "pminub") IOP(pmaxub, "pmaxub") IOP(pminsw, "pminsw") IOP(pmaxsw, "pmaxsw")
IOP(pminsb, "pminsb") IOP(pmaxsb, "pmaxsb") IOP(pminuw, "pminuw") IOP(pmaxuw, "pmaxuw")
IOP(pminsd, "pminsd") IOP(pmaxsd, "pmaxsd") IOP(pminud, "pminud") IOP(pmaxud, "pmaxud")
/* compares */
LANES_U(pcmpeqb, 16, b, x == y ? ~0ull : 0) LANES_U(pcmpeqw, 8, w, x == y ? ~0ull : 0) LANES_U(pcmpeqd, 4, d, x == y ? ~0ull : 0)
LANES_U(pcmpeqq, 2, q, x == y ? ~0ull : 0)
LANES_S(pcmpgtb, 16, sb, x > y ? -1 : 0) LANES_S(pcmpgtw, 8, sw, x > y ? -1 : 0) LANES_S(pcmpgtd, 4, sd, x > y ? -1 : 0)
LANES_S(pcmpgtq, 2, sq, x > y ? -1 : 0)
IOP(pcmpeqb, "pcmpeqb") IOP(pcmpeqw, "pcmpeqw") IOP(pcmpeqd, "pcmpeqd") IOP(pcmpeqq, "pcmpeqq")
IOP(pcmpgtb, "pcmpgtb") IOP(pcmpgtw, "pcmpgtw") IOP(pcmpgtd, "pcmpgtd") IOP(pcmpgtq, "pcmpgtq")
/* logic */
LANES_U(pand, 2, q, x & y) LANES_U(pandn, 2, q, ~x & y) LANES_U(por, 2, q, x | y) LANES_U(pxor, 2, q, x ^ y)
IOP(pand, "pand") IOP(pandn, "pandn") IOP(por, "por") IOP(pxor, "pxor")

TEST(sse2_int_add_sub_wraparound) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "paddb", F_SSE2, op_paddb, ref_paddb, GB_INT, 0, 0, 1500 }, { "paddw", F_SSE2, op_paddw, ref_paddw, GB_INT, 0, 0, 1500 },
        { "paddd", F_SSE2, op_paddd, ref_paddd, GB_INT, 0, 0, 1500 }, { "paddq", F_SSE2, op_paddq, ref_paddq, GB_INT, 0, 0, 1500 },
        { "psubb", F_SSE2, op_psubb, ref_psubb, GB_INT, 0, 0, 1500 }, { "psubw", F_SSE2, op_psubw, ref_psubw, GB_INT, 0, 0, 1500 },
        { "psubd", F_SSE2, op_psubd, ref_psubd, GB_INT, 0, 0, 1500 }, { "psubq", F_SSE2, op_psubq, ref_psubq, GB_INT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0001);
}
TEST(sse2_int_saturating_add_sub) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "paddsb", F_SSE2, op_paddsb, ref_paddsb, GB_INT, 0, 0, 1500 }, { "paddsw", F_SSE2, op_paddsw, ref_paddsw, GB_INT, 0, 0, 1500 },
        { "psubsb", F_SSE2, op_psubsb, ref_psubsb, GB_INT, 0, 0, 1500 }, { "psubsw", F_SSE2, op_psubsw, ref_psubsw, GB_INT, 0, 0, 1500 },
        { "paddusb", F_SSE2, op_paddusb, ref_paddusb, GB_INT, 0, 0, 1500 }, { "paddusw", F_SSE2, op_paddusw, ref_paddusw, GB_INT, 0, 0, 1500 },
        { "psubusb", F_SSE2, op_psubusb, ref_psubusb, GB_INT, 0, 0, 1500 }, { "psubusw", F_SSE2, op_psubusw, ref_psubusw, GB_INT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0002);
}
TEST(sse2_int_multiply) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "pmullw", F_SSE2, op_pmullw, ref_pmullw, GB_INT, 0, 0, 1500 }, { "pmulhw", F_SSE2, op_pmulhw, ref_pmulhw, GB_INT, 0, 0, 1500 },
        { "pmulhuw", F_SSE2, op_pmulhuw, ref_pmulhuw, GB_INT, 0, 0, 1500 }, { "pmuludq", F_SSE2, op_pmuludq, ref_pmuludq, GB_INT, 0, 0, 1500 },
        { "pmaddwd", F_SSE2, op_pmaddwd, ref_pmaddwd, GB_INT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0003);
}
TEST(sse2_int_avg_sad) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "pavgb", F_SSE2, op_pavgb, ref_pavgb, GB_INT, 0, 0, 1500 }, { "pavgw", F_SSE2, op_pavgw, ref_pavgw, GB_INT, 0, 0, 1500 },
        { "psadbw", F_SSE2, op_psadbw, ref_psadbw, GB_INT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0004);
}
TEST(sse2_int_min_max) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "pminub", F_SSE2, op_pminub, ref_pminub, GB_INT, 0, 0, 1500 }, { "pmaxub", F_SSE2, op_pmaxub, ref_pmaxub, GB_INT, 0, 0, 1500 },
        { "pminsw", F_SSE2, op_pminsw, ref_pminsw, GB_INT, 0, 0, 1500 }, { "pmaxsw", F_SSE2, op_pmaxsw, ref_pmaxsw, GB_INT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0005);
}
TEST(sse2_int_compare) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "pcmpeqb", F_SSE2, op_pcmpeqb, ref_pcmpeqb, GB_INT, 0, 0, 1500 }, { "pcmpeqw", F_SSE2, op_pcmpeqw, ref_pcmpeqw, GB_INT, 0, 0, 1500 },
        { "pcmpeqd", F_SSE2, op_pcmpeqd, ref_pcmpeqd, GB_INT, 0, 0, 1500 }, { "pcmpgtb", F_SSE2, op_pcmpgtb, ref_pcmpgtb, GB_INT, 0, 0, 1500 },
        { "pcmpgtw", F_SSE2, op_pcmpgtw, ref_pcmpgtw, GB_INT, 0, 0, 1500 }, { "pcmpgtd", F_SSE2, op_pcmpgtd, ref_pcmpgtd, GB_INT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0006);
}
TEST(sse2_int_logic) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "pand", F_SSE2, op_pand, ref_pand, GB_INT, 0, 0, 500 }, { "pandn", F_SSE2, op_pandn, ref_pandn, GB_INT, 0, 0, 500 },
        { "por", F_SSE2, op_por, ref_por, GB_INT, 0, 0, 500 }, { "pxor", F_SSE2, op_pxor, ref_pxor, GB_INT, 0, 0, 500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0007);
}

/* ---- shifts: count = whole low quadword (xmm form) or imm8; count > width-1 -> 0 (logical) or sign fill */
enum { SH_LL, SH_RL, SH_RA };
static void shift_ref(V128 *r, const V128 *a, uint64_t cnt, int width, int kind) {
    V128 t;
    for (int i = 0; i < 128 / width; i++) {
        uint64_t x = width == 16 ? a->w[i] : width == 32 ? a->d[i] : a->q[i], v;
        uint64_t mask = width == 64 ? ~0ull : (1ull << width) - 1;
        if (kind == SH_RA) {
            int64_t sx = width == 16 ? (int64_t)a->sw[i] : width == 32 ? (int64_t)a->sd[i] : a->sq[i];
            uint64_t c = cnt > (uint64_t)width - 1 ? (uint64_t)width - 1 : cnt;
            v = (uint64_t)(sx >> c) & mask;
        } else if (cnt >= (uint64_t)width) v = 0;
        else v = (kind == SH_LL ? x << cnt : x >> cnt) & mask;
        if (width == 16) t.w[i] = (uint16_t)v; else if (width == 32) t.d[i] = (uint32_t)v; else t.q[i] = v;
    }
    *r = t;
}
#define SHREFS(nm, W, K) \
    static void ref_##nm(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; shift_ref(r, a, b->q[0], W, K); } \
    static void ref_##nm##_i(V128 *r, const V128 *a, const V128 *b, int imm) { (void)b; shift_ref(r, a, (uint64_t)imm, W, K); }
SHREFS(psllw, 16, SH_LL) SHREFS(pslld, 32, SH_LL) SHREFS(psllq, 64, SH_LL)
SHREFS(psrlw, 16, SH_RL) SHREFS(psrld, 32, SH_RL) SHREFS(psrlq, 64, SH_RL)
SHREFS(psraw, 16, SH_RA) SHREFS(psrad, 32, SH_RA)
IOP(psllw, "psllw") IOP(pslld, "pslld") IOP(psllq, "psllq") IOP(psrlw, "psrlw") IOP(psrld, "psrld") IOP(psrlq, "psrlq")
IOP(psraw, "psraw") IOP(psrad, "psrad")
#define INSN_ "psllw"
IOPS(psllw_i, I256)
#undef INSN_
#define INSN_ "pslld"
IOPS(pslld_i, I256)
#undef INSN_
#define INSN_ "psllq"
IOPS(psllq_i, I256)
#undef INSN_
#define INSN_ "psrlw"
IOPS(psrlw_i, I256)
#undef INSN_
#define INSN_ "psrld"
IOPS(psrld_i, I256)
#undef INSN_
#define INSN_ "psrlq"
IOPS(psrlq_i, I256)
#undef INSN_
#define INSN_ "psraw"
IOPS(psraw_i, I256)
#undef INSN_
#define INSN_ "psrad"
IOPS(psrad_i, I256)
#undef INSN_
#define INSN_ "pslldq"
IOPS(pslldq, I256)
#undef INSN_
#define INSN_ "psrldq"
IOPS(psrldq, I256)
#undef INSN_
static void ref_pslldq(V128 *r, const V128 *a, const V128 *b, int imm) { (void)b; V128 t; memset(&t, 0, 16);
    { for (int i = 0; i < 16; i++) if (i - imm >= 0) t.b[i] = a->b[i - imm]; }
    *r = t; }
static void ref_psrldq(V128 *r, const V128 *a, const V128 *b, int imm) { (void)b; V128 t; memset(&t, 0, 16);
    { for (int i = 0; i < 16; i++) if (i + imm < 16) t.b[i] = a->b[i + imm]; }
    *r = t; }

TEST(sse2_shift_by_xmm_count) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "psllw xmm", F_SSE2, op_psllw, ref_psllw, GB_COUNT, 0, 0, 1500 }, { "pslld xmm", F_SSE2, op_pslld, ref_pslld, GB_COUNT, 0, 0, 1500 },
        { "psllq xmm", F_SSE2, op_psllq, ref_psllq, GB_COUNT, 0, 0, 1500 }, { "psrlw xmm", F_SSE2, op_psrlw, ref_psrlw, GB_COUNT, 0, 0, 1500 },
        { "psrld xmm", F_SSE2, op_psrld, ref_psrld, GB_COUNT, 0, 0, 1500 }, { "psrlq xmm", F_SSE2, op_psrlq, ref_psrlq, GB_COUNT, 0, 0, 1500 },
        { "psraw xmm", F_SSE2, op_psraw, ref_psraw, GB_COUNT, 0, 0, 1500 }, { "psrad xmm", F_SSE2, op_psrad, ref_psrad, GB_COUNT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0011);
}
TEST(sse2_shift_by_imm_sweep) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "psllw imm", F_SSE2, op_psllw_i, ref_psllw_i, GB_INT, 0, 255, 6 }, { "pslld imm", F_SSE2, op_pslld_i, ref_pslld_i, GB_INT, 0, 255, 6 },
        { "psllq imm", F_SSE2, op_psllq_i, ref_psllq_i, GB_INT, 0, 255, 6 }, { "psrlw imm", F_SSE2, op_psrlw_i, ref_psrlw_i, GB_INT, 0, 255, 6 },
        { "psrld imm", F_SSE2, op_psrld_i, ref_psrld_i, GB_INT, 0, 255, 6 }, { "psrlq imm", F_SSE2, op_psrlq_i, ref_psrlq_i, GB_INT, 0, 255, 6 },
        { "psraw imm", F_SSE2, op_psraw_i, ref_psraw_i, GB_INT, 0, 255, 6 }, { "psrad imm", F_SSE2, op_psrad_i, ref_psrad_i, GB_INT, 0, 255, 6 } };
    run_iops(t, T_COUNTOF(t), 0x1A0012);
}
TEST(sse2_byte_shift_pslldq_psrldq) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "pslldq", F_SSE2, op_pslldq, ref_pslldq, GB_INT, 0, 255, 6 }, { "psrldq", F_SSE2, op_psrldq, ref_psrldq, GB_INT, 0, 255, 6 } };
    run_iops(t, T_COUNTOF(t), 0x1A0013);
}

/* ---- pack / unpack */
static void ref_packsswb(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    for (int i = 0; i < 8; i++) { t.sb[i] = (int8_t)SAT_S8(a->sw[i]); t.sb[8 + i] = (int8_t)SAT_S8(b->sw[i]); } *r = t; }
static void ref_packuswb(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    for (int i = 0; i < 8; i++) { t.b[i] = (uint8_t)SAT_U8(a->sw[i]); t.b[8 + i] = (uint8_t)SAT_U8(b->sw[i]); } *r = t; }
static void ref_packssdw(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    for (int i = 0; i < 4; i++) { t.sw[i] = (int16_t)SAT_S16(a->sd[i]); t.sw[4 + i] = (int16_t)SAT_S16(b->sd[i]); } *r = t; }
static void ref_packusdw(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    for (int i = 0; i < 4; i++) { t.w[i] = (uint16_t)SAT_U16(a->sd[i]); t.w[4 + i] = (uint16_t)SAT_U16(b->sd[i]); } *r = t; }
IOP(packsswb, "packsswb") IOP(packuswb, "packuswb") IOP(packssdw, "packssdw") IOP(packusdw, "packusdw")
#define UNPK(nm, F, N, hi) static void ref_##nm(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t; \
    for (int i = 0; i < N / 2; i++) { t.F[2 * i] = a->F[i + (hi ? N / 2 : 0)]; t.F[2 * i + 1] = b->F[i + (hi ? N / 2 : 0)]; } *r = t; }
UNPK(punpcklbw, b, 16, 0) UNPK(punpckhbw, b, 16, 1) UNPK(punpcklwd, w, 8, 0) UNPK(punpckhwd, w, 8, 1)
UNPK(punpckldq, d, 4, 0) UNPK(punpckhdq, d, 4, 1) UNPK(punpcklqdq, q, 2, 0) UNPK(punpckhqdq, q, 2, 1)
IOP(punpcklbw, "punpcklbw") IOP(punpckhbw, "punpckhbw") IOP(punpcklwd, "punpcklwd") IOP(punpckhwd, "punpckhwd")
IOP(punpckldq, "punpckldq") IOP(punpckhdq, "punpckhdq") IOP(punpcklqdq, "punpcklqdq") IOP(punpckhqdq, "punpckhqdq")
IOP(unpcklps, "unpcklps") IOP(unpckhps, "unpckhps") IOP(unpcklpd, "unpcklpd") IOP(unpckhpd, "unpckhpd")
static void ref_movlhps(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t = *a; t.q[1] = b->q[0]; *r = t; }
static void ref_movhlps(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t = *a; t.q[0] = b->q[1]; *r = t; }
static void ref_movss(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t = *a; t.d[0] = b->d[0]; *r = t; }
static void ref_movsd(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t = *a; t.q[0] = b->q[0]; *r = t; }
static void ref_movq0(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; (void)a; V128 t = { 0 }; t.q[0] = b->q[0]; *r = t; }
IOP(movlhps, "movlhps") IOP(movhlps, "movhlps") IOP(movss_rr, "movss") IOP(movsd_rr, "movsd") IOP(movq_rr, "movq")

TEST(sse2_pack_saturate) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "packsswb", F_SSE2, op_packsswb, ref_packsswb, GB_INT, 0, 0, 1500 }, { "packuswb", F_SSE2, op_packuswb, ref_packuswb, GB_INT, 0, 0, 1500 },
        { "packssdw", F_SSE2, op_packssdw, ref_packssdw, GB_INT, 0, 0, 1500 },
        { "packusdw", F_SSE41, op_packusdw, ref_packusdw, GB_INT, 0, 0, 1500 } };
    run_iops(t, T_COUNTOF(t), 0x1A0021);
}
TEST(sse2_unpack_interleave_and_moves) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "punpcklbw", F_SSE2, op_punpcklbw, ref_punpcklbw, GB_INT, 0, 0, 300 }, { "punpckhbw", F_SSE2, op_punpckhbw, ref_punpckhbw, GB_INT, 0, 0, 300 },
        { "punpcklwd", F_SSE2, op_punpcklwd, ref_punpcklwd, GB_INT, 0, 0, 300 }, { "punpckhwd", F_SSE2, op_punpckhwd, ref_punpckhwd, GB_INT, 0, 0, 300 },
        { "punpckldq", F_SSE2, op_punpckldq, ref_punpckldq, GB_INT, 0, 0, 300 }, { "punpckhdq", F_SSE2, op_punpckhdq, ref_punpckhdq, GB_INT, 0, 0, 300 },
        { "punpcklqdq", F_SSE2, op_punpcklqdq, ref_punpcklqdq, GB_INT, 0, 0, 300 }, { "punpckhqdq", F_SSE2, op_punpckhqdq, ref_punpckhqdq, GB_INT, 0, 0, 300 },
        { "unpcklps", F_SSE, op_unpcklps, ref_punpckldq, GB_INT, 0, 0, 300 }, { "unpckhps", F_SSE, op_unpckhps, ref_punpckhdq, GB_INT, 0, 0, 300 },
        { "unpcklpd", F_SSE2, op_unpcklpd, ref_punpcklqdq, GB_INT, 0, 0, 300 }, { "unpckhpd", F_SSE2, op_unpckhpd, ref_punpckhqdq, GB_INT, 0, 0, 300 },
        { "movlhps", F_SSE, op_movlhps, ref_movlhps, GB_INT, 0, 0, 300 }, { "movhlps", F_SSE, op_movhlps, ref_movhlps, GB_INT, 0, 0, 300 },
        { "movss xmm,xmm (merge)", F_SSE, op_movss_rr, ref_movss, GB_INT, 0, 0, 300 },
        { "movsd xmm,xmm (merge)", F_SSE2, op_movsd_rr, ref_movsd, GB_INT, 0, 0, 300 },
        { "movq xmm,xmm (zero upper)", F_SSE2, op_movq_rr, ref_movq0, GB_INT, 0, 0, 300 } };
    run_iops(t, T_COUNTOF(t), 0x1A0022);
}

/* ---- shuffles with immediate sweeps */
#define INSN_ "pshufd"
IOPI(pshufd, I256)
#undef INSN_
#define INSN_ "pshuflw"
IOPI(pshuflw, I256)
#undef INSN_
#define INSN_ "pshufhw"
IOPI(pshufhw, I256)
#undef INSN_
#define INSN_ "shufps"
IOPI(shufps, I256)
#undef INSN_
#define INSN_ "shufpd"
IOPI(shufpd, I256)
#undef INSN_
static void ref_pshufd(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; V128 t; for (int i = 0; i < 4; i++) t.d[i] = b->d[(imm >> (2 * i)) & 3]; *r = t; }
static void ref_pshuflw(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; V128 t = *b; for (int i = 0; i < 4; i++) t.w[i] = b->w[(imm >> (2 * i)) & 3]; *r = t; }
static void ref_pshufhw(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; V128 t = *b; for (int i = 0; i < 4; i++) t.w[4 + i] = b->w[4 + ((imm >> (2 * i)) & 3)]; *r = t; }
static void ref_shufps(V128 *r, const V128 *a, const V128 *b, int imm) { V128 t;
    t.d[0] = a->d[imm & 3]; t.d[1] = a->d[(imm >> 2) & 3]; t.d[2] = b->d[(imm >> 4) & 3]; t.d[3] = b->d[(imm >> 6) & 3]; *r = t; }
static void ref_shufpd(V128 *r, const V128 *a, const V128 *b, int imm) { V128 t; t.q[0] = a->q[imm & 1]; t.q[1] = b->q[(imm >> 1) & 1]; *r = t; }
TEST(sse2_pshufd_pshuflw_pshufhw_imm_sweep) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "pshufd", F_SSE2, op_pshufd, ref_pshufd, GB_INT, 0, 255, 4 }, { "pshuflw", F_SSE2, op_pshuflw, ref_pshuflw, GB_INT, 0, 255, 4 },
        { "pshufhw", F_SSE2, op_pshufhw, ref_pshufhw, GB_INT, 0, 255, 4 } };
    run_iops(t, T_COUNTOF(t), 0x1A0031);
}
TEST(sse_shufps_shufpd_imm_sweep) {
    NEED(F_SSE2);
    static const iop_t t[] = {
        { "shufps", F_SSE, op_shufps, ref_shufps, GB_INT, 0, 255, 4 },
        { "shufpd", F_SSE2, op_shufpd, ref_shufpd, GB_INT, 0, 3, 50 } };   /* imm[7:2] of SHUFPD reserved */
    run_iops(t, T_COUNTOF(t), 0x1A0032);
}

/* ---- MOVMSKPS / MOVMSKPD / PMOVMSKB: sign bits to a GPR, upper bits zero */
TEST(sse_movmskps_movmskpd_pmovmskb) {
    NEED(F_SSE2);
    t_rng r; t_rng_seed(&r, 0x1A0041);
    vbegin();
    for (int it = 0; it < 2000; it++) {
        V128 a; gen_int(&r, &a);
        uintptr_t m1 = (uintptr_t)-1, m2 = (uintptr_t)-1, m3 = (uintptr_t)-1;
        /* write the full-width register: the 32-bit result must zero-extend */
        __asm__ volatile("movmskps %1, %k0" : "+r"(m1) : "x"(a.v));
        __asm__ volatile("movmskpd %1, %k0" : "+r"(m2) : "x"(a.v));
        __asm__ volatile("pmovmskb %1, %k0" : "+r"(m3) : "x"(a.v));
        uintptr_t w1 = 0, w2 = 0, w3 = 0;
        for (int i = 0; i < 4; i++) w1 |= (uintptr_t)(a.d[i] >> 31) << i;
        for (int i = 0; i < 2; i++) w2 |= (uintptr_t)(a.q[i] >> 63) << i;
        for (int i = 0; i < 16; i++) w3 |= (uintptr_t)(a.b[i] >> 7) << i;
        VCHECK(m1 == w1 && m2 == w2 && m3 == w3, "a=%s movmskps 0x%llx/0x%llx movmskpd 0x%llx/0x%llx pmovmskb 0x%llx/0x%llx",
               hx(&a, 4), (unsigned long long)m1, (unsigned long long)w1, (unsigned long long)m2, (unsigned long long)w2,
               (unsigned long long)m3, (unsigned long long)w3);
    }
    VEND("movmsk");
}

/* ---- MOVD / MOVQ / MOVSS / MOVSD load forms: zero the rest of the register */
TEST(sse2_movd_movq_zero_extension) {
    NEED(F_SSE2);
    t_rng r; t_rng_seed(&r, 0x1A0042);
    vbegin();
    for (int it = 0; it < 500; it++) {
        V128 junk, d; gen_int(&r, &junk);
        uint32_t g32 = t_rng_u32(&r); uint64_t m64 = t_rng_next(&r); uint32_t m32 = t_rng_u32(&r);
        d = junk; __asm__ volatile("movd %1, %0" : "+x"(d.v) : "r"(g32));
        VCHECK(d.d[0] == g32 && d.d[1] == 0 && d.q[1] == 0, "movd xmm, r32: %s", hx(&d, 4));
        d = junk; __asm__ volatile("movq %1, %0" : "+x"(d.v) : "m"(m64));
        VCHECK(d.q[0] == m64 && d.q[1] == 0, "movq xmm, m64: %s", hx(&d, 8));
        d = junk; __asm__ volatile("movd %1, %0" : "+x"(d.v) : "m"(m32));
        VCHECK(d.d[0] == m32 && d.d[1] == 0 && d.q[1] == 0, "movd xmm, m32: %s", hx(&d, 4));
        d = junk; __asm__ volatile("movss %1, %0" : "+x"(d.v) : "m"(m32));
        VCHECK(d.d[0] == m32 && d.d[1] == 0 && d.q[1] == 0, "movss xmm, m32: %s", hx(&d, 4));
        d = junk; __asm__ volatile("movsd %1, %0" : "+x"(d.v) : "m"(m64));
        VCHECK(d.q[0] == m64 && d.q[1] == 0, "movsd xmm, m64: %s", hx(&d, 8));
        uint32_t o32 = 0; __asm__ volatile("movd %1, %0" : "=r"(o32) : "x"(junk.v));
        VCHECK(o32 == junk.d[0], "movd r32, xmm: 0x%x want 0x%x", o32, junk.d[0]);
        uint64_t o64 = 0; __asm__ volatile("movq %1, %0" : "=m"(o64) : "x"(junk.v));
        VCHECK(o64 == junk.q[0], "movq m64, xmm");
        /* MOVLPS/MOVHPS/MOVLPD/MOVHPD loads merge */
        d = junk; __asm__ volatile("movhps %1, %0" : "+x"(d.v) : "m"(m64));
        VCHECK(d.q[0] == junk.q[0] && d.q[1] == m64, "movhps xmm, m64: %s", hx(&d, 8));
        d = junk; __asm__ volatile("movlpd %1, %0" : "+x"(d.v) : "m"(m64));
        VCHECK(d.q[0] == m64 && d.q[1] == junk.q[1], "movlpd xmm, m64: %s", hx(&d, 8));
#ifdef T_X64
        uint64_t g64 = t_rng_next(&r);
        d = junk; __asm__ volatile("movq %1, %0" : "+x"(d.v) : "r"(g64));
        VCHECK(d.q[0] == g64 && d.q[1] == 0, "movq xmm, r64: %s", hx(&d, 8));
        uint64_t o = 0; __asm__ volatile("movq %1, %0" : "=r"(o) : "x"(junk.v));
        VCHECK(o == junk.q[0], "movq r64, xmm");
#endif
    }
    VEND("movd/movq/movss/movsd");
}

/* ---- MASKMOVDQU: byte-masked store to [EDI/RDI]; unselected bytes untouched */
TEST(sse2_maskmovdqu) {
    NEED(F_SSE2);
    t_rng r; t_rng_seed(&r, 0x1A0043);
    void *base = NULL;
    unsigned char *pg = (unsigned char *)t_guarded_alloc(64, &base);   /* the 64 bytes end at a no-access page */
    REQUIRE(pg != NULL);
    vbegin();
    for (int it = 0; it < 1000; it++) {
        V128 data, mask; gen_int(&r, &data); gen_int(&r, &mask);
        int off = (int)t_rng_range(&r, 49);                              /* any alignment, last at 48 = page end */
        unsigned char before[64], want[64];
        t_rng_fill(&r, before, 64);
        memcpy(pg, before, 64); memcpy(want, before, 64);
        for (int i = 0; i < 16; i++) if (mask.b[i] & 0x80) want[off + i] = data.b[i];
        unsigned char *dst = pg + off;
        __asm__ volatile("maskmovdqu %1, %0\n\tsfence" : : "x"(data.v), "x"(mask.v), "D"(dst) : "memory");
        VCHECK(!memcmp(pg, want, 64), "maskmovdqu off=%d mask=%s data=%s", off, hx(&mask, 1), hx(&data, 1));
    }
    VEND("maskmovdqu");
    t_guarded_free(base);
}

/* ---- PEXTRW / PINSRW (SSE2) */
#define PEXTRW_C(n) case n: __asm__ volatile("pextrw %2, %1, %k0" : "+r"(g) : "x"(a->v), "i"(n)); break;
#define PINSRW_C(n) case n: __asm__ volatile("pinsrw %2, %k1, %0" : "+x"(d->v) : "r"(g), "i"(n)); break;
static T_NOINLINE uintptr_t do_pextrw(const V128 *a, int imm) { uintptr_t g = (uintptr_t)-1; switch (imm) { I256(PEXTRW_C) } return g; }
static T_NOINLINE void do_pinsrw(V128 *d, uintptr_t g, int imm) { switch (imm) { I256(PINSRW_C) } }
TEST(sse2_pextrw_pinsrw_imm_sweep) {
    NEED(F_SSE2);
    t_rng r; t_rng_seed(&r, 0x1A0044);
    vbegin();
    for (int imm = 0; imm < 256; imm++)
        for (int it = 0; it < 8; it++) {
            V128 a; gen_int(&r, &a);
            uintptr_t g = do_pextrw(&a, imm);
            VCHECK(g == a.w[imm & 7], "pextrw imm %d: got 0x%llx want 0x%x (imm[2:0] selects, zero-extended)", imm, (unsigned long long)g, a.w[imm & 7]);
            V128 d = a; uintptr_t src = (uintptr_t)t_rng_next(&r);
            do_pinsrw(&d, src, imm);
            V128 w = a; w.w[imm & 7] = (uint16_t)src;
            VCHECK(!memcmp(&d, &w, 16), "pinsrw imm %d src 0x%llx: got %s want %s", imm, (unsigned long long)src, hx(&d, 2), hx(&w, 2));
        }
    VEND("pextrw/pinsrw");
}

/* ================================================================ SSE3 moves */
static void ref_movshdup(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; (void)imm; V128 t = { .d = { b->d[1], b->d[1], b->d[3], b->d[3] } }; *r = t; }
static void ref_movsldup(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; (void)imm; V128 t = { .d = { b->d[0], b->d[0], b->d[2], b->d[2] } }; *r = t; }
static void ref_movddup(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; (void)imm; V128 t = { .q = { b->q[0], b->q[0] } }; *r = t; }
IOP(movshdup, "movshdup") IOP(movsldup, "movsldup") IOP(movddup, "movddup")
TEST(sse3_movdup_lddqu) {
    NEED(F_SSE3);
    static const iop_t t[] = {
        { "movshdup", F_SSE3, op_movshdup, ref_movshdup, GB_INT, 0, 0, 300 }, { "movsldup", F_SSE3, op_movsldup, ref_movsldup, GB_INT, 0, 0, 300 },
        { "movddup", F_SSE3, op_movddup, ref_movddup, GB_INT, 0, 0, 300 } };
    run_iops(t, T_COUNTOF(t), 0x1A0051);
    /* LDDQU: unaligned 16-byte load at every offset, including the last 16 bytes before a no-access page */
    void *base = NULL;
    unsigned char *pg = (unsigned char *)t_guarded_alloc(64, &base);
    REQUIRE(pg != NULL);
    for (int i = 0; i < 64; i++) pg[i] = (unsigned char)(i * 7 + 3);
    for (int off = 0; off <= 48; off++) {
        V128 d;
        __asm__ volatile("lddqu %1, %0" : "=x"(d.v) : "m"(*(const unsigned char (*)[16])(pg + off)));
        CHECKF(!memcmp(&d, pg + off, 16), "lddqu offset %d: %s", off, hx(&d, 1));
    }
    /* MOVDDUP m64 at the page end */
    V128 d; __asm__ volatile("movddup %1, %0" : "=x"(d.v) : "m"(*(const uint64_t *)(pg + 56)));
    uint64_t q; memcpy(&q, pg + 56, 8);
    CHECKF(d.q[0] == q && d.q[1] == q, "movddup m64 at page end: %s", hx(&d, 8));
    t_guarded_free(base);
}

/* ================================================================ SSSE3 */
LANES_S(pabsb, 16, sb, (int8_t)ABS(y)) LANES_S(pabsw, 8, sw, (int16_t)ABS(y)) LANES_S(pabsd, 4, sd, (int32_t)ABS(y))
LANES_S(psignb, 16, sb, y < 0 ? (int8_t)-x : y == 0 ? 0 : x) LANES_S(psignw, 8, sw, y < 0 ? (int16_t)-x : y == 0 ? 0 : x)
LANES_S(psignd, 4, sd, y < 0 ? (int32_t)-x : y == 0 ? 0 : x)
IOP(pabsb, "pabsb") IOP(pabsw, "pabsw") IOP(pabsd, "pabsd") IOP(psignb, "psignb") IOP(psignw, "psignw") IOP(psignd, "psignd")
#define HOR(nm, F, N, T, expr) static void ref_##nm(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t; \
    for (int i = 0; i < N / 2; i++) { int64_t x = a->F[2 * i], y = a->F[2 * i + 1]; t.F[i] = (T)(expr); } \
    for (int i = 0; i < N / 2; i++) { int64_t x = b->F[2 * i], y = b->F[2 * i + 1]; t.F[N / 2 + i] = (T)(expr); } *r = t; }
HOR(phaddw, sw, 8, int16_t, x + y) HOR(phaddd, sd, 4, int32_t, x + y) HOR(phaddsw, sw, 8, int16_t, SAT_S16(x + y))
HOR(phsubw, sw, 8, int16_t, x - y) HOR(phsubd, sd, 4, int32_t, x - y) HOR(phsubsw, sw, 8, int16_t, SAT_S16(x - y))
IOP(phaddw, "phaddw") IOP(phaddd, "phaddd") IOP(phaddsw, "phaddsw") IOP(phsubw, "phsubw") IOP(phsubd, "phsubd") IOP(phsubsw, "phsubsw")
static void ref_pmaddubsw(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    { for (int i = 0; i < 8; i++) t.sw[i] = (int16_t)SAT_S16((int64_t)a->b[2 * i] * b->sb[2 * i] + (int64_t)a->b[2 * i + 1] * b->sb[2 * i + 1]); }
    *r = t; }
LANES_S(pmulhrsw, 8, sw, (int16_t)((((x * y) >> 14) + 1) >> 1))
static void ref_pshufb(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    { for (int i = 0; i < 16; i++) t.b[i] = (b->b[i] & 0x80) ? 0 : a->b[b->b[i] & 15]; }
    *r = t; }
IOP(pmaddubsw, "pmaddubsw") IOP(pmulhrsw, "pmulhrsw") IOP(pshufb, "pshufb")
#define INSN_ "palignr"
IOPI(palignr, I256)
#undef INSN_
static void ref_palignr(V128 *r, const V128 *a, const V128 *b, int imm) {
    unsigned char cat[32]; memcpy(cat, b, 16); memcpy(cat + 16, a, 16);   /* dest:src, shifted right imm bytes */
    V128 t; for (int i = 0; i < 16; i++) t.b[i] = imm + i < 32 ? cat[imm + i] : 0; *r = t;
}
TEST(ssse3_abs_sign) {
    NEED(F_SSSE3);
    static const iop_t t[] = {
        { "pabsb", F_SSSE3, op_pabsb, ref_pabsb, GB_INT, 0, 0, 1000 }, { "pabsw", F_SSSE3, op_pabsw, ref_pabsw, GB_INT, 0, 0, 1000 },
        { "pabsd", F_SSSE3, op_pabsd, ref_pabsd, GB_INT, 0, 0, 1000 }, { "psignb", F_SSSE3, op_psignb, ref_psignb, GB_INT, 0, 0, 1000 },
        { "psignw", F_SSSE3, op_psignw, ref_psignw, GB_INT, 0, 0, 1000 }, { "psignd", F_SSSE3, op_psignd, ref_psignd, GB_INT, 0, 0, 1000 } };
    run_iops(t, T_COUNTOF(t), 0x1A0061);
}
TEST(ssse3_horizontal_add_sub) {
    NEED(F_SSSE3);
    static const iop_t t[] = {
        { "phaddw", F_SSSE3, op_phaddw, ref_phaddw, GB_INT, 0, 0, 1000 }, { "phaddd", F_SSSE3, op_phaddd, ref_phaddd, GB_INT, 0, 0, 1000 },
        { "phaddsw", F_SSSE3, op_phaddsw, ref_phaddsw, GB_INT, 0, 0, 1000 }, { "phsubw", F_SSSE3, op_phsubw, ref_phsubw, GB_INT, 0, 0, 1000 },
        { "phsubd", F_SSSE3, op_phsubd, ref_phsubd, GB_INT, 0, 0, 1000 }, { "phsubsw", F_SSSE3, op_phsubsw, ref_phsubsw, GB_INT, 0, 0, 1000 } };
    run_iops(t, T_COUNTOF(t), 0x1A0062);
}
TEST(ssse3_pmaddubsw_pmulhrsw) {
    NEED(F_SSSE3);
    static const iop_t t[] = {
        { "pmaddubsw", F_SSSE3, op_pmaddubsw, ref_pmaddubsw, GB_INT, 0, 0, 2000 }, { "pmulhrsw", F_SSSE3, op_pmulhrsw, ref_pmulhrsw, GB_INT, 0, 0, 2000 } };
    run_iops(t, T_COUNTOF(t), 0x1A0063);
}
TEST(ssse3_pshufb) {
    NEED(F_SSSE3);
    static const iop_t t[] = { { "pshufb", F_SSSE3, op_pshufb, ref_pshufb, GB_INT, 0, 0, 3000 } };
    run_iops(t, T_COUNTOF(t), 0x1A0064);
    /* identity, reverse, broadcast, all-zero (bit 7) controls */
    V128 a; for (int i = 0; i < 16; i++) a.b[i] = (uint8_t)(0xA0 + i);
    V128 c, d, w;
    for (int i = 0; i < 16; i++) c.b[i] = (uint8_t)(15 - i);
    d = a; op_pshufb(&d, &c, 0); ref_pshufb(&w, &a, &c, 0);
    CHECKF(!memcmp(&d, &w, 16) && d.b[0] == 0xAF, "pshufb reverse: %s", hx(&d, 1));
    for (int i = 0; i < 16; i++) c.b[i] = (uint8_t)(0x80 | i);
    d = a; op_pshufb(&d, &c, 0);
    CHECKF(d.q[0] == 0 && d.q[1] == 0, "pshufb bit7 controls must zero: %s", hx(&d, 1));
    for (int i = 0; i < 16; i++) c.b[i] = (uint8_t)(0x70 | 5);   /* bits 6:4 ignored */
    d = a; op_pshufb(&d, &c, 0);
    CHECKF(d.b[0] == 0xA5 && d.b[15] == 0xA5, "pshufb ignores control bits 6:4: %s", hx(&d, 1));
}
TEST(ssse3_palignr_imm_sweep) {
    NEED(F_SSSE3);
    static const iop_t t[] = { { "palignr", F_SSSE3, op_palignr, ref_palignr, GB_INT, 0, 255, 8 } };
    run_iops(t, T_COUNTOF(t), 0x1A0065);
}

/* ================================================================ SSE4.1 integer and blends */
LANES_S(pmulld, 4, sd, (int32_t)(uint32_t)(x * y))
static void ref_pmuldq(V128 *r, const V128 *a, const V128 *b, int imm) { (void)imm; V128 t;
    t.sq[0] = (int64_t)a->sd[0] * b->sd[0]; t.sq[1] = (int64_t)a->sd[2] * b->sd[2]; *r = t; }
IOP(pmulld, "pmulld") IOP(pmuldq, "pmuldq")
TEST(sse41_pmulld_pmuldq) {
    NEED(F_SSE41);
    static const iop_t t[] = {
        { "pmulld", F_SSE41, op_pmulld, ref_pmulld, GB_INT, 0, 0, 2000 }, { "pmuldq", F_SSE41, op_pmuldq, ref_pmuldq, GB_INT, 0, 0, 2000 } };
    run_iops(t, T_COUNTOF(t), 0x1A0071);
}
TEST(sse41_min_max_all_widths) {
    NEED(F_SSE41);
    static const iop_t t[] = {
        { "pminsb", F_SSE41, op_pminsb, ref_pminsb, GB_INT, 0, 0, 1000 }, { "pmaxsb", F_SSE41, op_pmaxsb, ref_pmaxsb, GB_INT, 0, 0, 1000 },
        { "pminuw", F_SSE41, op_pminuw, ref_pminuw, GB_INT, 0, 0, 1000 }, { "pmaxuw", F_SSE41, op_pmaxuw, ref_pmaxuw, GB_INT, 0, 0, 1000 },
        { "pminsd", F_SSE41, op_pminsd, ref_pminsd, GB_INT, 0, 0, 1000 }, { "pmaxsd", F_SSE41, op_pmaxsd, ref_pmaxsd, GB_INT, 0, 0, 1000 },
        { "pminud", F_SSE41, op_pminud, ref_pminud, GB_INT, 0, 0, 1000 }, { "pmaxud", F_SSE41, op_pmaxud, ref_pmaxud, GB_INT, 0, 0, 1000 } };
    run_iops(t, T_COUNTOF(t), 0x1A0072);
}
TEST(sse41_pcmpeqq_sse42_pcmpgtq) {
    NEED(F_SSE41);
    static const iop_t t[] = {
        { "pcmpeqq", F_SSE41, op_pcmpeqq, ref_pcmpeqq, GB_INT, 0, 0, 2000 }, { "pcmpgtq", F_SSE42, op_pcmpgtq, ref_pcmpgtq, GB_INT, 0, 0, 2000 } };
    run_iops(t, T_COUNTOF(t), 0x1A0073);
}
#define INSN_ "pblendw"
IOPI(pblendw, I256)
#undef INSN_
#define INSN_ "blendps"
IOPI(blendps, I16)
#undef INSN_
#define INSN_ "blendpd"
IOPI(blendpd, I16)
#undef INSN_
#define INSN_ "insertps"
IOPI(insertps, I256)
#undef INSN_
#define INSN_ "mpsadbw"
IOPI(mpsadbw, I16)
#undef INSN_
static void ref_pblendw(V128 *r, const V128 *a, const V128 *b, int imm) { V128 t; for (int i = 0; i < 8; i++) t.w[i] = (imm >> i) & 1 ? b->w[i] : a->w[i]; *r = t; }
static void ref_blendps(V128 *r, const V128 *a, const V128 *b, int imm) { V128 t; for (int i = 0; i < 4; i++) t.d[i] = (imm >> i) & 1 ? b->d[i] : a->d[i]; *r = t; }
static void ref_blendpd(V128 *r, const V128 *a, const V128 *b, int imm) { V128 t; for (int i = 0; i < 2; i++) t.q[i] = (imm >> i) & 1 ? b->q[i] : a->q[i]; *r = t; }
static void ref_insertps(V128 *r, const V128 *a, const V128 *b, int imm) {
    V128 t = *a; t.d[(imm >> 4) & 3] = b->d[(imm >> 6) & 3];
    for (int i = 0; i < 4; i++) if ((imm >> i) & 1) t.d[i] = 0;
    *r = t;
}
static void ref_mpsadbw(V128 *r, const V128 *a, const V128 *b, int imm) {
    int o1 = ((imm >> 2) & 1) * 4, o2 = (imm & 3) * 4; V128 t;
    for (int i = 0; i < 8; i++) { int s = 0; for (int k = 0; k < 4; k++) { int dd = a->b[o1 + i + k] - b->b[o2 + k]; s += ABS(dd); } t.w[i] = (uint16_t)s; }
    *r = t;
}
TEST(sse41_blend_imm_sweep) {
    NEED(F_SSE41);
    static const iop_t t[] = {
        { "pblendw", F_SSE41, op_pblendw, ref_pblendw, GB_INT, 0, 255, 6 },
        { "blendps", F_SSE41, op_blendps, ref_blendps, GB_INT, 0, 15, 40 },
        { "blendpd", F_SSE41, op_blendpd, ref_blendpd, GB_INT, 0, 3, 100 } };
    run_iops(t, T_COUNTOF(t), 0x1A0074);
}
TEST(sse41_insertps_imm_sweep) {
    NEED(F_SSE41);
    static const iop_t t[] = { { "insertps", F_SSE41, op_insertps, ref_insertps, GB_INT, 0, 255, 8 } };
    run_iops(t, T_COUNTOF(t), 0x1A0075);
    /* memory form: loads 32 bits, imm[7:6] ignored */
    V128 a = { .d = { 1, 2, 3, 4 } }; uint32_t m = 0xDEADBEEF;
    __asm__ volatile("insertps $0xE0, %1, %0" : "+x"(a.v) : "m"(m));   /* count_s=3 ignored, count_d=2 */
    CHECKF(a.d[0] == 1 && a.d[1] == 2 && a.d[2] == 0xDEADBEEF && a.d[3] == 4, "insertps m32 imm 0xE0: %s", hx(&a, 4));
}
TEST(sse41_mpsadbw) {
    NEED(F_SSE41);
    static const iop_t t[] = { { "mpsadbw", F_SSE41, op_mpsadbw, ref_mpsadbw, GB_INT, 0, 7, 300 } };
    run_iops(t, T_COUNTOF(t), 0x1A0076);
}

/* PBLENDVB / BLENDVPS / BLENDVPD: implicit XMM0 mask, selected by the top bit of each element */
#define BLENDV(nm, insn) static T_NOINLINE void bv_##nm(V128 *d, const V128 *s, const V128 *m) { \
    __asm__ volatile("movdqu %2, %%xmm0\n\t" insn " %%xmm0, %1, %0" : "+x"(d->v) : "x"(s->v), "m"(m->v) : "xmm0"); }
BLENDV(pblendvb, "pblendvb") BLENDV(blendvps, "blendvps") BLENDV(blendvpd, "blendvpd")
TEST(sse41_blendv_xmm0_mask) {
    NEED(F_SSE41);
    t_rng r; t_rng_seed(&r, 0x1A0077);
    vbegin();
    for (int it = 0; it < 3000; it++) {
        V128 a, b, m, d, w; gen_int(&r, &a); gen_int(&r, &b); gen_int(&r, &m);
        d = a; bv_pblendvb(&d, &b, &m);
        for (int i = 0; i < 16; i++) w.b[i] = (m.b[i] & 0x80) ? b.b[i] : a.b[i];
        VCHECK(!memcmp(&d, &w, 16), "pblendvb a=%s b=%s m=%s got=%s", hx(&a, 1), hx(&b, 1), hx(&m, 1), hx(&d, 1));
        d = a; bv_blendvps(&d, &b, &m);
        for (int i = 0; i < 4; i++) w.d[i] = (m.d[i] >> 31) ? b.d[i] : a.d[i];
        VCHECK(!memcmp(&d, &w, 16), "blendvps a=%s b=%s m=%s got=%s", hx(&a, 4), hx(&b, 4), hx(&m, 4), hx(&d, 4));
        d = a; bv_blendvpd(&d, &b, &m);
        for (int i = 0; i < 2; i++) w.q[i] = (m.q[i] >> 63) ? b.q[i] : a.q[i];
        VCHECK(!memcmp(&d, &w, 16), "blendvpd a=%s b=%s m=%s got=%s", hx(&a, 8), hx(&b, 8), hx(&m, 8), hx(&d, 8));
    }
    VEND("blendv");
}

/* PMOVSX / PMOVZX (all 12 forms; register and m64/m32/m16 sources) */
#define PMOVX(nm, insn) IOP(nm, insn)
PMOVX(pmovsxbw, "pmovsxbw") PMOVX(pmovsxbd, "pmovsxbd") PMOVX(pmovsxbq, "pmovsxbq") PMOVX(pmovsxwd, "pmovsxwd")
PMOVX(pmovsxwq, "pmovsxwq") PMOVX(pmovsxdq, "pmovsxdq") PMOVX(pmovzxbw, "pmovzxbw") PMOVX(pmovzxbd, "pmovzxbd")
PMOVX(pmovzxbq, "pmovzxbq") PMOVX(pmovzxwd, "pmovzxwd") PMOVX(pmovzxwq, "pmovzxwq") PMOVX(pmovzxdq, "pmovzxdq")
#define XREF(nm, DF, N, SF, T) static void ref_##nm(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; (void)imm; V128 t; \
    { for (int i = 0; i < N; i++) t.DF[i] = (T)b->SF[i]; } *r = t; }
XREF(pmovsxbw, sw, 8, sb, int16_t) XREF(pmovsxbd, sd, 4, sb, int32_t) XREF(pmovsxbq, sq, 2, sb, int64_t)
XREF(pmovsxwd, sd, 4, sw, int32_t) XREF(pmovsxwq, sq, 2, sw, int64_t) XREF(pmovsxdq, sq, 2, sd, int64_t)
XREF(pmovzxbw, w, 8, b, uint16_t) XREF(pmovzxbd, d, 4, b, uint32_t) XREF(pmovzxbq, q, 2, b, uint64_t)
XREF(pmovzxwd, d, 4, w, uint32_t) XREF(pmovzxwq, q, 2, w, uint64_t) XREF(pmovzxdq, q, 2, d, uint64_t)
TEST(sse41_pmovsx_pmovzx) {
    NEED(F_SSE41);
    static const iop_t t[] = {
        { "pmovsxbw", F_SSE41, op_pmovsxbw, ref_pmovsxbw, GB_INT, 0, 0, 400 }, { "pmovsxbd", F_SSE41, op_pmovsxbd, ref_pmovsxbd, GB_INT, 0, 0, 400 },
        { "pmovsxbq", F_SSE41, op_pmovsxbq, ref_pmovsxbq, GB_INT, 0, 0, 400 }, { "pmovsxwd", F_SSE41, op_pmovsxwd, ref_pmovsxwd, GB_INT, 0, 0, 400 },
        { "pmovsxwq", F_SSE41, op_pmovsxwq, ref_pmovsxwq, GB_INT, 0, 0, 400 }, { "pmovsxdq", F_SSE41, op_pmovsxdq, ref_pmovsxdq, GB_INT, 0, 0, 400 },
        { "pmovzxbw", F_SSE41, op_pmovzxbw, ref_pmovzxbw, GB_INT, 0, 0, 400 }, { "pmovzxbd", F_SSE41, op_pmovzxbd, ref_pmovzxbd, GB_INT, 0, 0, 400 },
        { "pmovzxbq", F_SSE41, op_pmovzxbq, ref_pmovzxbq, GB_INT, 0, 0, 400 }, { "pmovzxwd", F_SSE41, op_pmovzxwd, ref_pmovzxwd, GB_INT, 0, 0, 400 },
        { "pmovzxwq", F_SSE41, op_pmovzxwq, ref_pmovzxwq, GB_INT, 0, 0, 400 }, { "pmovzxdq", F_SSE41, op_pmovzxdq, ref_pmovzxdq, GB_INT, 0, 0, 400 } };
    run_iops(t, T_COUNTOF(t), 0x1A0078);
    /* memory forms read only 8/4/2 bytes: place the source right before a no-access page */
    void *base = NULL;
    unsigned char *pg = (unsigned char *)t_guarded_alloc(16, &base);
    REQUIRE(pg != NULL);
    for (int i = 0; i < 16; i++) pg[i] = (unsigned char)(0xF0 + i);
    V128 d, b, w; memset(&b, 0, sizeof b);
    memcpy(&b, pg + 8, 8);
    __asm__ volatile("pmovsxbw %1, %0" : "=x"(d.v) : "m"(*(const uint64_t *)(pg + 8))); ref_pmovsxbw(&w, &b, &b, 0);
    CHECKF(!memcmp(&d, &w, 16), "pmovsxbw m64 at page end: %s", hx(&d, 2));
    memset(&b, 0, sizeof b); memcpy(&b, pg + 12, 4);
    __asm__ volatile("pmovzxbd %1, %0" : "=x"(d.v) : "m"(*(const uint32_t *)(pg + 12))); ref_pmovzxbd(&w, &b, &b, 0);
    CHECKF(!memcmp(&d, &w, 16), "pmovzxbd m32 at page end: %s", hx(&d, 4));
    memset(&b, 0, sizeof b); memcpy(&b, pg + 14, 2);
    __asm__ volatile("pmovsxbq %1, %0" : "=x"(d.v) : "m"(*(const uint16_t *)(pg + 14))); ref_pmovsxbq(&w, &b, &b, 0);
    CHECKF(!memcmp(&d, &w, 16), "pmovsxbq m16 at page end: %s", hx(&d, 8));
    t_guarded_free(base);
}

/* PTEST: ZF = (src AND dst) == 0, CF = (src AND NOT dst) == 0; AF, OF, PF, SF cleared */
TEST(sse41_ptest_flags) {
    NEED(F_SSE41);
    t_rng r; t_rng_seed(&r, 0x1A0079);
    vbegin();
    for (int it = 0; it < 4000; it++) {
        V128 a, b; gen_int(&r, &a); gen_int(&r, &b);
        switch (it % 5) { case 1: memset(&b, 0, 16); break; case 2: b = a; break; case 3: for (int i = 0; i < 2; i++) b.q[i] = ~a.q[i]; break;
                          case 4: for (int i = 0; i < 2; i++) b.q[i] &= a.q[i]; break; }
        uint8_t zf, cf, of, sf, pf;
        __asm__ volatile("ptest %6, %5" : "=@ccz"(zf), "=@ccc"(cf), "=@cco"(of), "=@ccs"(sf), "=@ccp"(pf) : "x"(a.v), "x"(b.v));
        int wz = !((a.q[0] & b.q[0]) | (a.q[1] & b.q[1])), wc = !((~a.q[0] & b.q[0]) | (~a.q[1] & b.q[1]));
        VCHECK(zf == wz && cf == wc && !of && !sf && !pf, "ptest dst=%s src=%s: ZF%d CF%d OF%d SF%d PF%d want ZF%d CF%d",
               hx(&a, 8), hx(&b, 8), zf, cf, of, sf, pf, wz, wc);
    }
    VEND("ptest");
}

/* PHMINPOSUW: min unsigned word -> bits 15:0, its lowest index -> bits 18:16, rest zero */
static void ref_phminposuw(V128 *r, const V128 *a, const V128 *b, int imm) { (void)a; (void)imm;
    int idx = 0; for (int i = 1; i < 8; i++) if (b->w[i] < b->w[idx]) idx = i;
    V128 t = { 0 }; t.w[0] = b->w[idx]; t.w[1] = (uint16_t)idx; *r = t; }
IOP(phminposuw, "phminposuw")
TEST(sse41_phminposuw) {
    NEED(F_SSE41);
    static const iop_t t[] = { { "phminposuw", F_SSE41, op_phminposuw, ref_phminposuw, GB_INT, 0, 0, 4000 } };
    run_iops(t, T_COUNTOF(t), 0x1A007A);
}

/* EXTRACTPS / PEXTRB / PEXTRD / PEXTRQ / PINSRB / PINSRD / PINSRQ (index = low imm bits; rest ignored) */
#define PEXB_C(n) case n: __asm__ volatile("pextrb %2, %1, %k0" : "+r"(g) : "x"(a->v), "i"(n)); break;
#define PEXD_C(n) case n: __asm__ volatile("pextrd %2, %1, %k0" : "+r"(g) : "x"(a->v), "i"(n)); break;
#define EXPS_C(n) case n: __asm__ volatile("extractps %2, %1, %k0" : "+r"(g) : "x"(a->v), "i"(n)); break;
#define PEXWM_C(n) case n: __asm__ volatile("pextrw %2, %1, %0" : "=m"(m16) : "x"(a->v), "i"(n)); break;
#define PINB_C(n) case n: __asm__ volatile("pinsrb %2, %k1, %0" : "+x"(d->v) : "r"(g), "i"(n)); break;
#define PIND_C(n) case n: __asm__ volatile("pinsrd %2, %k1, %0" : "+x"(d->v) : "r"(g), "i"(n)); break;
static T_NOINLINE uintptr_t do_pextrb(const V128 *a, int imm) { uintptr_t g = (uintptr_t)-1; switch (imm) { I256(PEXB_C) } return g; }
static T_NOINLINE uintptr_t do_pextrd(const V128 *a, int imm) { uintptr_t g = (uintptr_t)-1; switch (imm) { I256(PEXD_C) } return g; }
static T_NOINLINE uintptr_t do_extractps(const V128 *a, int imm) { uintptr_t g = (uintptr_t)-1; switch (imm) { I256(EXPS_C) } return g; }
static T_NOINLINE uint16_t do_pextrw_m(const V128 *a, int imm) { uint16_t m16 = 0xABCD; switch (imm) { I256(PEXWM_C) } return m16; }
static T_NOINLINE void do_pinsrb(V128 *d, uintptr_t g, int imm) { switch (imm) { I256(PINB_C) } }
static T_NOINLINE void do_pinsrd(V128 *d, uintptr_t g, int imm) { switch (imm) { I256(PIND_C) } }
#ifdef T_X64
#define PEXQ_C(n) case n: __asm__ volatile("pextrq %2, %1, %0" : "=r"(g) : "x"(a->v), "i"(n)); break;
#define PINQ_C(n) case n: __asm__ volatile("pinsrq %2, %1, %0" : "+x"(d->v) : "r"(g), "i"(n)); break;
static T_NOINLINE uint64_t do_pextrq(const V128 *a, int imm) { uint64_t g = 0; switch (imm) { I256(PEXQ_C) } return g; }
static T_NOINLINE void do_pinsrq(V128 *d, uint64_t g, int imm) { switch (imm) { I256(PINQ_C) } }
#endif
TEST(sse41_extract_insert_imm_sweep) {
    NEED(F_SSE41);
    t_rng r; t_rng_seed(&r, 0x1A007B);
    vbegin();
    for (int imm = 0; imm < 256; imm++)
        for (int it = 0; it < 4; it++) {
            V128 a; gen_int(&r, &a);
            uintptr_t g;
            g = do_pextrb(&a, imm); VCHECK(g == a.b[imm & 15], "pextrb imm %d: 0x%llx want 0x%x", imm, (unsigned long long)g, a.b[imm & 15]);
            g = do_pextrd(&a, imm); VCHECK(g == a.d[imm & 3], "pextrd imm %d: 0x%llx want 0x%x", imm, (unsigned long long)g, a.d[imm & 3]);
            g = do_extractps(&a, imm); VCHECK(g == a.d[imm & 3], "extractps imm %d: 0x%llx want 0x%x", imm, (unsigned long long)g, a.d[imm & 3]);
            uint16_t m = do_pextrw_m(&a, imm); VCHECK(m == a.w[imm & 7], "pextrw m16 (SSE4.1 form) imm %d: 0x%x want 0x%x", imm, m, a.w[imm & 7]);
            uintptr_t src = (uintptr_t)t_rng_next(&r);
            V128 d = a, w = a; do_pinsrb(&d, src, imm); w.b[imm & 15] = (uint8_t)src;
            VCHECK(!memcmp(&d, &w, 16), "pinsrb imm %d: got %s want %s", imm, hx(&d, 1), hx(&w, 1));
            d = a; w = a; do_pinsrd(&d, src, imm); w.d[imm & 3] = (uint32_t)src;
            VCHECK(!memcmp(&d, &w, 16), "pinsrd imm %d: got %s want %s", imm, hx(&d, 4), hx(&w, 4));
#ifdef T_X64
            uint64_t q = do_pextrq(&a, imm); VCHECK(q == a.q[imm & 1], "pextrq imm %d", imm);
            d = a; w = a; uint64_t s64 = t_rng_next(&r); do_pinsrq(&d, s64, imm); w.q[imm & 1] = s64;
            VCHECK(!memcmp(&d, &w, 16), "pinsrq imm %d: got %s want %s", imm, hx(&d, 8), hx(&w, 8));
#endif
        }
    VEND("extract/insert");
#ifndef T_X64
    t_note("PEXTRQ/PINSRQ need REX.W: x64-only, not applicable to this x86 build");
#endif
}

/* MOVNTDQA from ordinary write-back memory behaves as an aligned 16-byte load */
TEST(sse41_movntdqa_wb_memory) {
    NEED(F_SSE41);
    static V128 buf[64];
    t_rng r; t_rng_seed(&r, 0x1A007C);
    t_rng_fill(&r, buf, sizeof buf);
    for (int i = 0; i < 64; i++) {
        V128 d;
        __asm__ volatile("movntdqa %1, %0" : "=x"(d.v) : "m"(buf[i].v));
        CHECKF(!memcmp(&d, &buf[i], 16), "movntdqa [%d]: %s want %s", i, hx(&d, 8), hx(&buf[i], 8));
    }
}
