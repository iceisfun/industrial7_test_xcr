/* sse_arithmetic_softfp.h - integer-only IEEE-754 binary32/binary64 reference model with x86 SSE
 * semantics, used by tests/sse_arithmetic.c as the exact-bit oracle.
 *
 * Everything works on raw bit patterns (uint64_t) and integer arithmetic only: no float/double
 * variables, so the model is independent of the FPU it checks and SNaN payloads never pass through
 * x87 registers on the 32-bit build.
 *
 * x86 SSE rules modelled (Intel SDM Vol.1 4.8.3.5 Table 4-7, 10.2.3, 11.5.2; AMD APM Vol.1 4.4):
 *   - NaN operand(s): result is the FIRST source operand if it is a NaN, else the second; an SNaN is
 *     quieted (bit M-1 set) and raises IE. Invalid operation without NaN input -> default NaN
 *     ("real indefinite": sign=1, quiet bit only: 0xFFC00000 / 0xFFF8000000000000).
 *   - Rounding: MXCSR.RC (0 RN-even, 1 toward -inf, 2 toward +inf, 3 toward zero).
 *   - Overflow: RN->inf, RZ->max finite, RD-> -inf/+max, RU-> +inf/-max. Flags OE|PE.
 *   - Underflow flag (masked UM): tiny (detected AFTER rounding, i.e. with unbounded exponent) AND
 *     inexact. Gradual underflow to denormals.
 *   - FTZ (MXCSR[15], set the model's sf_ftz): when the result is tiny (same after-rounding test) it is
 *     replaced by a zero of the result's sign and UE|PE are set, even when the tiny result was exact
 *     (SDM Vol.1 10.2.3.3: "returns a zero result with the sign of the true result; sets the
 *     precision and underflow exception flags"). FTZ never changes operands (that is DAZ) and does
 *     not touch MIN/MAX/compare/convert-to-integer results, which cannot underflow.
 *   - DAZ: denormal sources become signed zero before any computation, DE is not set.
 *   - Exact zero sum of opposite-signed operands is +0, or -0 when RC = toward -inf.
 * The model returns the flags it is sure about and sets *unc when the architecture leaves room
 * (e.g. DE ordering when a NaN/special and a denormal meet in one lane); callers then skip the flag
 * comparison for that vector but still compare the result bits. */
#ifndef SSE_ARITHMETIC_SOFTFP_H
#define SSE_ARITHMETIC_SOFTFP_H
#include <stdint.h>

enum { FL_IE = 1, FL_DE = 2, FL_ZE = 4, FL_OE = 8, FL_UE = 16, FL_PE = 32 };
enum { RC_RN = 0, RC_RD = 1, RC_RU = 2, RC_RZ = 3 };

typedef struct sf_fmt { int M, E, bias, bits; } sf_fmt;
static const sf_fmt SF32 = { 23, 8, 127, 32 };
static const sf_fmt SF64 = { 52, 11, 1023, 64 };

static inline uint64_t sf_signbit(const sf_fmt *f) { return 1ull << (f->M + f->E); }
static inline uint64_t sf_expmax(const sf_fmt *f) { return (1ull << f->E) - 1; }
static inline uint64_t sf_mmask(const sf_fmt *f) { return (1ull << f->M) - 1; }
static inline uint64_t sf_exp(const sf_fmt *f, uint64_t x) { return (x >> f->M) & sf_expmax(f); }
static inline uint64_t sf_man(const sf_fmt *f, uint64_t x) { return x & sf_mmask(f); }
static inline int sf_sign(const sf_fmt *f, uint64_t x) { return (x & sf_signbit(f)) != 0; }
static inline int sf_isnan(const sf_fmt *f, uint64_t x) { return sf_exp(f, x) == sf_expmax(f) && sf_man(f, x) != 0; }
static inline int sf_issnan(const sf_fmt *f, uint64_t x) { return sf_isnan(f, x) && !(x & (1ull << (f->M - 1))); }
static inline int sf_isinf(const sf_fmt *f, uint64_t x) { return sf_exp(f, x) == sf_expmax(f) && sf_man(f, x) == 0; }
static inline int sf_iszero(const sf_fmt *f, uint64_t x) { return (x & ~sf_signbit(f)) == 0; }
static inline int sf_isden(const sf_fmt *f, uint64_t x) { return sf_exp(f, x) == 0 && sf_man(f, x) != 0; }
static inline int sf_isspecial(const sf_fmt *f, uint64_t x) { return sf_iszero(f, x) || sf_exp(f, x) == sf_expmax(f); }
static inline uint64_t sf_quiet(const sf_fmt *f, uint64_t x) { return x | (1ull << (f->M - 1)); }
static inline uint64_t sf_defnan(const sf_fmt *f) { return sf_signbit(f) | (sf_expmax(f) << f->M) | (1ull << (f->M - 1)); }
static inline uint64_t sf_inf(const sf_fmt *f, int s) { return (s ? sf_signbit(f) : 0) | (sf_expmax(f) << f->M); }
static inline uint64_t sf_zero(const sf_fmt *f, int s) { return s ? sf_signbit(f) : 0; }
static inline uint64_t sf_maxfin(const sf_fmt *f, int s) { return (s ? sf_signbit(f) : 0) | (((sf_expmax(f) - 1) << f->M) | sf_mmask(f)); }

static int sf_ftz;   /* models MXCSR.FTZ for the next operations (single-threaded use) */

static inline int sf_clz64(uint64_t x) { int n = 0; if (!x) return 64; while (!(x >> 63)) { x <<= 1; n++; } return n; }

/* value of a finite nonzero x = sig * 2^e */
static inline void sf_unpack(const sf_fmt *f, uint64_t x, uint64_t *sig, int *e) {
    uint64_t ex = sf_exp(f, x);
    *sig = sf_man(f, x) | (ex ? (1ull << f->M) : 0);
    *e = (int)(ex ? ex : 1) - f->bias - f->M;
}

/* DAZ / DE bookkeeping for one source operand of an arithmetic op */
static inline uint64_t sf_src(const sf_fmt *f, uint64_t x, int daz, uint32_t *fl) {
    if (sf_isden(f, x)) {
        if (daz) return x & sf_signbit(f);
        *fl |= FL_DE;
    }
    return x;
}

/* Round sig*2^e (+ sticky below bit 0) to format f. sig != 0. */
static uint64_t sf_round_pack(const sf_fmt *f, int sign, int e, uint64_t sig, int sticky, int rc, uint32_t *fl) {
    int lz = sf_clz64(sig);
    sig <<= lz; e -= lz;                    /* sig in [2^63, 2^64) */
    int be = e + 63 + f->bias;              /* biased exponent of the leading bit */
    uint64_t sb = sign ? sf_signbit(f) : 0;
    int nshift = 63 - f->M;
    if (be >= (int)sf_expmax(f)) goto overflow;
    {
        int shift = be >= 1 ? nshift : nshift + (1 - be);
        uint64_t q, guard, rest;
        if (shift > 64) { q = 0; guard = 0; rest = 1; }
        else if (shift == 64) { q = 0; guard = sig >> 63; rest = (sig << 1) != 0 || sticky; }
        else { q = sig >> shift; guard = (sig >> (shift - 1)) & 1; rest = (sig & ((1ull << (shift - 1)) - 1)) != 0 || sticky; }
        int inexact = guard || rest;
        int inc = 0;
        switch (rc) {
        case RC_RN: inc = guard && (rest || (q & 1)); break;
        case RC_RD: inc = sign && inexact; break;
        case RC_RU: inc = !sign && inexact; break;
        default: inc = 0;
        }
        q += inc;
        if (inexact) *fl |= FL_PE;
        if (be < 1) {
            /* tininess after rounding: round to full precision with unbounded exponent */
            int tiny = 1;
            if (be == 0) {
                uint64_t qn = sig >> nshift, gn = (sig >> (nshift - 1)) & 1;
                int rn = (sig & ((1ull << (nshift - 1)) - 1)) != 0 || sticky, in2 = 0;
                switch (rc) {
                case RC_RN: in2 = gn && (rn || (qn & 1)); break;
                case RC_RD: in2 = sign && (gn || rn); break;
                case RC_RU: in2 = !sign && (gn || rn); break;
                }
                if (qn + in2 == (1ull << (f->M + 1))) tiny = 0;
            }
            if (tiny && sf_ftz) { *fl |= FL_UE | FL_PE; return sb; }
            if (tiny && inexact) *fl |= FL_UE;
            return sb | q;
        }
        uint64_t bits = ((uint64_t)(be - 1) << f->M) + q;
        if ((bits >> f->M) >= sf_expmax(f)) goto overflow;
        return sb | bits;
    }
overflow:
    *fl |= FL_OE | FL_PE;
    switch (rc) {
    case RC_RN: return sf_inf(f, sign);
    case RC_RZ: return sf_maxfin(f, sign);
    case RC_RD: return sign ? sf_inf(f, 1) : sf_maxfin(f, 0);
    default:    return sign ? sf_maxfin(f, 1) : sf_inf(f, 0);
    }
}

/* two-operand NaN rule (first source wins) */
static inline int sf_nan2(const sf_fmt *f, uint64_t a, uint64_t b, uint64_t *r, uint32_t *fl) {
    if (!sf_isnan(f, a) && !sf_isnan(f, b)) return 0;
    if (sf_issnan(f, a) || sf_issnan(f, b)) *fl |= FL_IE;
    *r = sf_isnan(f, a) ? sf_quiet(f, a) : sf_quiet(f, b);
    return 1;
}

/* the DE flag is only asserted by the model when every operand is finite and nonzero */
static inline void sf_unc_den2(const sf_fmt *f, uint64_t a, uint64_t b, int daz, int *unc) {
    if (daz) return;
    if ((sf_isden(f, a) || sf_isden(f, b)) && (sf_isspecial(f, a) || sf_isspecial(f, b))) *unc = 1;
}

static uint64_t sf_add(const sf_fmt *f, uint64_t a, uint64_t b, int sub, int rc, int daz, uint32_t *fl, int *unc) {
    uint64_t r;
    sf_unc_den2(f, a, b, daz, unc);
    if (sf_nan2(f, a, b, &r, fl)) return r;
    a = sf_src(f, a, daz, fl); b = sf_src(f, b, daz, fl);
    if (sub) b ^= sf_signbit(f);
    int sa = sf_sign(f, a), sb = sf_sign(f, b);
    if (sf_isinf(f, a) || sf_isinf(f, b)) {
        if (sf_isinf(f, a) && sf_isinf(f, b) && sa != sb) { *fl |= FL_IE; return sf_defnan(f); }
        return sf_isinf(f, a) ? a : b;
    }
    if (sf_iszero(f, a) && sf_iszero(f, b)) return sf_zero(f, sa == sb ? sa : rc == RC_RD);
    /* x + 0: the result is x itself, which is tiny when x is a denormal (FTZ flushes it, UE|PE) */
    if (sf_iszero(f, a) || sf_iszero(f, b)) {
        uint64_t x = sf_iszero(f, a) ? b : a;
        if (sf_ftz && sf_isden(f, x)) { *fl |= FL_UE | FL_PE; return x & sf_signbit(f); }
        return x;
    }
    uint64_t ma, mb; int ea, eb;
    sf_unpack(f, a, &ma, &ea); sf_unpack(f, b, &mb, &eb);
    ma <<= 9; ea -= 9; mb <<= 9; eb -= 9;
    /* X = larger magnitude */
    if (eb > ea || (eb == ea && mb > ma)) { uint64_t tm = ma; ma = mb; mb = tm; int te = ea; ea = eb; eb = te; int ts = sa; sa = sb; sb = ts; }
    int d = ea - eb, sticky = 0;
    if (d > 63) { sticky = mb != 0; mb = 0; }
    else if (d > 0) { sticky = (mb & ((1ull << d) - 1)) != 0; mb >>= d; }
    uint64_t m;
    if (sa == sb) m = ma + mb;
    else {
        m = ma - mb - (sticky ? 1 : 0);
        if (m == 0 && !sticky) return sf_zero(f, rc == RC_RD);
    }
    return sf_round_pack(f, sa, ea, m, sticky, rc, fl);
}

/* 64x64 -> 128 */
static inline void sf_mul64(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) {
    uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t mid = (ll >> 32) + (uint32_t)lh + (uint32_t)hl;
    *lo = (mid << 32) | (uint32_t)ll;
    *hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

static uint64_t sf_mul(const sf_fmt *f, uint64_t a, uint64_t b, int rc, int daz, uint32_t *fl, int *unc) {
    uint64_t r;
    sf_unc_den2(f, a, b, daz, unc);
    if (sf_nan2(f, a, b, &r, fl)) return r;
    a = sf_src(f, a, daz, fl); b = sf_src(f, b, daz, fl);
    int s = sf_sign(f, a) ^ sf_sign(f, b);
    if (sf_isinf(f, a) || sf_isinf(f, b)) {
        if (sf_iszero(f, a) || sf_iszero(f, b)) { *fl |= FL_IE; return sf_defnan(f); }
        return sf_inf(f, s);
    }
    if (sf_iszero(f, a) || sf_iszero(f, b)) return sf_zero(f, s);
    uint64_t ma, mb, hi, lo; int ea, eb;
    sf_unpack(f, a, &ma, &ea); sf_unpack(f, b, &mb, &eb);
    sf_mul64(ma, mb, &hi, &lo);
    int e = ea + eb, sticky = 0;
    if (hi) {
        int sh = 64 - sf_clz64(hi);         /* bits of hi in use */
        sticky = (lo & ((1ull << sh) - 1)) != 0;
        lo = (lo >> sh) | (hi << (64 - sh));
        e += sh;
    }
    return sf_round_pack(f, s, e, lo, sticky, rc, fl);
}

static inline void sf_norm52(uint64_t *m, int *e) { while (*m < (1ull << 52)) { *m <<= 1; (*e)--; } while (*m >= (1ull << 53)) { /* not reached for these formats */ *m >>= 1; (*e)++; } }

static uint64_t sf_div(const sf_fmt *f, uint64_t a, uint64_t b, int rc, int daz, uint32_t *fl, int *unc) {
    uint64_t r;
    sf_unc_den2(f, a, b, daz, unc);
    if (sf_nan2(f, a, b, &r, fl)) return r;
    a = sf_src(f, a, daz, fl); b = sf_src(f, b, daz, fl);
    int s = sf_sign(f, a) ^ sf_sign(f, b);
    if (sf_isinf(f, a)) { if (sf_isinf(f, b)) { *fl |= FL_IE; return sf_defnan(f); } return sf_inf(f, s); }
    if (sf_isinf(f, b)) return sf_zero(f, s);
    if (sf_iszero(f, b)) { if (sf_iszero(f, a)) { *fl |= FL_IE; return sf_defnan(f); } *fl |= FL_ZE; return sf_inf(f, s); }
    if (sf_iszero(f, a)) return sf_zero(f, s);
    uint64_t ma, mb, q = 0; int ea, eb;
    sf_unpack(f, a, &ma, &ea); sf_unpack(f, b, &mb, &eb);
    sf_norm52(&ma, &ea); sf_norm52(&mb, &eb);
    uint64_t rem = ma;
    for (int i = 0; i < 63; i++) { q <<= 1; if (rem >= mb) { rem -= mb; q |= 1; } rem <<= 1; }
    return sf_round_pack(f, s, ea - eb - 62, q, rem != 0, rc, fl);
}

static uint64_t sf_sqrt(const sf_fmt *f, uint64_t a, int rc, int daz, uint32_t *fl, int *unc) {
    if (sf_isnan(f, a)) { if (sf_issnan(f, a)) *fl |= FL_IE; return sf_quiet(f, a); }
    if (!daz && sf_isden(f, a) && sf_sign(f, a)) *unc = 1;   /* DE vs IE ordering for sqrt(-denormal) */
    a = sf_src(f, a, daz, fl);
    if (sf_iszero(f, a)) return a;
    if (sf_sign(f, a)) { *fl |= FL_IE; return sf_defnan(f); }
    if (sf_isinf(f, a)) return a;
    uint64_t m; int e;
    sf_unpack(f, a, &m, &e);
    sf_norm52(&m, &e);
    if (e & 1) { m <<= 1; e -= 1; }
    /* root = floor(sqrt(m * 2^64)); radicand bits: m (high 64) then 64 zero bits */
    uint64_t root = 0, rem = 0;
    for (int i = 0; i < 64; i++) {
        uint64_t two = i < 32 ? (m >> (62 - 2 * i)) & 3 : 0;
        rem = (rem << 2) | two;
        uint64_t trial = (root << 2) | 1;
        root <<= 1;
        if (rem >= trial) { rem -= trial; root |= 1; }
    }
    return sf_round_pack(f, 0, (e - 64) / 2, root, rem != 0, rc, fl);
}

/* a < b for non-NaN values (DAZ applied by caller) */
static inline int sf_lt(const sf_fmt *f, uint64_t a, uint64_t b) {
    if (sf_iszero(f, a) && sf_iszero(f, b)) return 0;
    int sa = sf_sign(f, a), sb = sf_sign(f, b);
    uint64_t ma = a & ~sf_signbit(f), mb = b & ~sf_signbit(f);
    if (sa != sb) return sa;              /* negative < positive (zeros handled above) */
    return sa ? ma > mb : ma < mb;
}
static inline int sf_eq(const sf_fmt *f, uint64_t a, uint64_t b) {
    return (sf_iszero(f, a) && sf_iszero(f, b)) || a == b;
}

/* MINPS/MAXPS: NaN (either, quiet or signalling) or both zero -> second operand unchanged; IE on any NaN.
 * Under DAZ the operands are converted first, so a denormal second operand comes back as signed zero
 * (observed on Intel silicon; consistent with SDM 10.2.3.4 "converts all denormal source operands"). */
static uint64_t sf_minmax(const sf_fmt *f, uint64_t a, uint64_t b, int ismax, int daz, uint32_t *fl, int *unc) {
    if (daz) { a = sf_src(f, a, 1, fl); b = sf_src(f, b, 1, fl); }
    if (sf_isnan(f, a) || sf_isnan(f, b)) {
        *fl |= FL_IE;
        if (sf_isden(f, a) || sf_isden(f, b)) *unc = 1;
        return b;
    }
    a = sf_src(f, a, daz, fl); b = sf_src(f, b, daz, fl);
    int pick_a = ismax ? sf_lt(f, b, a) : sf_lt(f, a, b);
    return pick_a ? a : b;
}

/* CMPPS predicate 0..7 -> 1/0; signalling predicates (LT, LE, NLT, NLE) raise IE on QNaN too */
static int sf_cmp(const sf_fmt *f, uint64_t a, uint64_t b, int pred, int daz, uint32_t *fl, int *unc) {
    int un = sf_isnan(f, a) || sf_isnan(f, b);
    if (un) {
        int sig = pred == 1 || pred == 2 || pred == 5 || pred == 6;
        if (sig || sf_issnan(f, a) || sf_issnan(f, b)) *fl |= FL_IE;
        if (sf_isden(f, a) || sf_isden(f, b)) *unc = 1;
    } else {
        a = sf_src(f, a, daz, fl); b = sf_src(f, b, daz, fl);
    }
    int lt = !un && sf_lt(f, a, b), eq = !un && sf_eq(f, a, b);
    switch (pred & 7) {
    case 0: return eq;
    case 1: return lt;
    case 2: return lt || eq;
    case 3: return un;
    case 4: return !eq;
    case 5: return !lt;
    case 6: return !(lt || eq);
    default: return !un;
    }
}

/* int -> float (value fits in int64) */
static uint64_t sf_from_i64(const sf_fmt *f, int64_t v, int rc, uint32_t *fl) {
    if (v == 0) return 0;
    int s = v < 0;
    uint64_t m = s ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
    return sf_round_pack(f, s, 0, m, 0, rc, fl);
}

/* Round a finite value to an integer according to rc (or truncate). Returns 0 when it does not fit in
 * 64 bits magnitude (*big = 1). *inexact set when a fraction was discarded. */
static uint64_t sf_to_intmag(const sf_fmt *f, uint64_t a, int rc, int *big, int *inexact) {
    uint64_t m; int e; int s = sf_sign(f, a);
    *big = 0; *inexact = 0;
    if (sf_iszero(f, a)) return 0;
    sf_unpack(f, a, &m, &e);
    if (e >= 0) {
        if (e + (64 - sf_clz64(m)) > 64) { *big = 1; return 0; }
        return m << e;
    }
    uint64_t q, guard, rest;
    int sh = -e;
    if (sh > 64) { q = 0; guard = 0; rest = 1; }
    else if (sh == 64) { q = 0; guard = m >> 63; rest = (m << 1) != 0; }
    else { q = m >> sh; guard = (m >> (sh - 1)) & 1; rest = (m & ((1ull << (sh - 1)) - 1)) != 0; }
    *inexact = guard || rest;
    int inc = 0;
    switch (rc) {
    case RC_RN: inc = guard && (rest || (q & 1)); break;
    case RC_RD: inc = s && *inexact; break;
    case RC_RU: inc = !s && *inexact; break;
    }
    return q + inc;
}

/* CVT(T)SS2SI / CVT(T)PS2DQ etc.: bits = 32 or 64. NaN/out of range -> integer indefinite (1 << bits-1), IE */
static uint64_t sf_to_int(const sf_fmt *f, uint64_t a, int bits, int rc, int daz, uint32_t *fl, int *unc) {
    uint64_t indef = 1ull << (bits - 1);
    (void)unc;
    if (sf_isnan(f, a) || sf_isinf(f, a)) { *fl |= FL_IE; return indef; }
    if (daz && sf_isden(f, a)) a &= sf_signbit(f);   /* CVT*2SI lists no #D: DE is never set */
    int big, inex, s = sf_sign(f, a);
    uint64_t mag = sf_to_intmag(f, a, rc, &big, &inex);
    uint64_t lim = s ? indef : indef - 1;
    if (big || mag > lim) { *fl |= FL_IE; return indef; }
    if (inex) *fl |= FL_PE;
    uint64_t v = s ? (uint64_t)0 - mag : mag;
    return bits == 32 ? (uint32_t)v : v;
}

/* format conversion f -> g (CVTSS2SD / CVTSD2SS) */
static uint64_t sf_convert(const sf_fmt *f, const sf_fmt *g, uint64_t a, int rc, int daz, uint32_t *fl, int *unc) {
    (void)unc;
    int s = sf_sign(f, a);
    if (sf_isnan(f, a)) {
        if (sf_issnan(f, a)) *fl |= FL_IE;
        uint64_t pay = sf_man(f, a);
        pay = g->M >= f->M ? pay << (g->M - f->M) : pay >> (f->M - g->M);
        return sf_quiet(g, (s ? sf_signbit(g) : 0) | (sf_expmax(g) << g->M) | pay);
    }
    if (sf_isinf(f, a)) return sf_inf(g, s);
    a = sf_src(f, a, daz, fl);
    if (sf_iszero(f, a)) return sf_zero(g, s);
    uint64_t m; int e;
    sf_unpack(f, a, &m, &e);
    return sf_round_pack(g, s, e, m, 0, rc, fl);
}

/* ROUNDPS/ROUNDSS: round to integral value in the same format; no DE; PE unless suppressed */
static uint64_t sf_round_int(const sf_fmt *f, uint64_t a, int rc, uint32_t *fl) {
    if (sf_isnan(f, a)) { if (sf_issnan(f, a)) *fl |= FL_IE; return sf_quiet(f, a); }
    if (sf_isinf(f, a) || sf_iszero(f, a)) return a;
    uint64_t m; int e; int s = sf_sign(f, a);
    sf_unpack(f, a, &m, &e);
    if (e >= 0) return a;
    int big, inex;
    uint64_t q = sf_to_intmag(f, a, rc, &big, &inex);
    if (inex) *fl |= FL_PE;
    if (q == 0) return sf_zero(f, s);
    uint32_t dummy = 0;
    return sf_round_pack(f, s, 0, q, 0, RC_RN, &dummy);
}
#endif
