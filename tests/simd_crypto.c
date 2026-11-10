/* simd_crypto - AES-NI, PCLMULQDQ, SHA extensions, GFNI, VAES/VPCLMULQDQ and SHA512 extensions:
 * single-instruction semantics against bit-exact C models, complete algorithms against published
 * known-answer vectors, and the OS-facing part (state survives thread switches, suspension and
 * Get/SetThreadContext round trips while a crypto loop holds keys in vector registers).
 *
 * Contract under test
 *   - Instruction semantics: Intel SDM Vol.2 (AESENC/AESENCLAST/AESDEC/AESDECLAST/AESIMC/
 *     AESKEYGENASSIST, PCLMULQDQ, SHA1RNDS4/SHA1NEXTE/SHA1MSG1/SHA1MSG2, SHA256RNDS2/SHA256MSG1/
 *     SHA256MSG2, GF2P8MULB/GF2P8AFFINEQB/GF2P8AFFINEINVQB, VAES*, VPCLMULQDQ, VSHA512RNDS2/
 *     VSHA512MSG1/VSHA512MSG2); AMD APM Vol.4 describes the shared instructions identically.
 *     Legacy-SSE m128 operands of these instructions are "Exceptions Type 4": #GP on a misaligned
 *     16-byte memory operand; the VEX forms have no alignment requirement.
 *   - Algorithms: FIPS-197 (AES, Appendix A key expansion and Appendix B/C vectors), NIST SP 800-38A
 *     (ECB/CBC/CTR F.1/F.2/F.5 vectors), SP 800-38D and the GCM submission test cases 1-4, 13, 14, 16,
 *     FIPS 180-4 (SHA-1/SHA-256/SHA-512, "abc", empty, 448-bit, 896-bit, one million 'a'),
 *     CRC-32C (Castagnoli, RFC 3720 B.4 vectors, check value of "123456789") and CRC-32 (IEEE).
 *   - OS: an ordinary Windows process can use every instruction CPUID advertises (legacy-SSE forms
 *     need no XCR0 state; VEX forms need XCR0[2:1]; EVEX zmm forms need XCR0[7:5]); vector state used
 *     by these loops is per-thread and survives preemption, SuspendThread/ResumeThread and a
 *     GetThreadContext/SetThreadContext round trip of the unchanged context; a misaligned legacy
 *     m128 operand is reported to the process as an exception (the code is recorded; ACCESS_VIOLATION
 *     is what Windows reports for #GP).
 *
 * Method
 *   - Software references are table/bit-level C written from the standards and checked first against
 *     the published vectors (cases *_sw_reference_*), so a hardware mismatch is attributable.
 *   - Single-instruction models are checked against the hardware on biased random vectors (fixed seeds)
 *     and every imm8 where the immediate matters; at most 6 detailed mismatches per sweep plus one
 *     summary CHECK.
 *   - Model-composed algorithms (SHA-1/256/512 built from the instruction models in the same order the
 *     hardware path uses) must reproduce the FIPS 180-4 digests: this pins the model semantics even for
 *     instructions the build host cannot execute (SHA512 extensions, EVEX VAES).
 *   - Feature absent -> SKIP (per feature). Advertised but wrong -> FAIL.
 * Deterministic: fixed seeds, bounded loops, at most 8 worker threads, < 4 MB of buffers. */
#include "t.h"
#include <immintrin.h>

typedef union { uint8_t b[16]; uint32_t d[4]; uint64_t q[2]; __m128i v; } V128;
typedef struct { T_ALIGN(32) uint8_t b[32]; } B256;
typedef struct { T_ALIGN(64) uint8_t b[64]; } B512;

#define NEED(f) do { if (!cpu_has(f)) SKIP("%s not usable on this CPU (cpuid=%d)", cpu_feature_name(f), cpu_hw(f)); } while (0)

/* ================================================================ reporting helpers */
static int g_vec, g_bad;
#define VFAIL_MAX 6
static void vbegin(void) { g_vec = g_bad = 0; }
#define VCHECK(ok, ...) do { g_vec++; if (!(ok)) { if (g_bad++ < VFAIL_MAX) CHECKF(0, __VA_ARGS__); } } while (0)
static void vend_(const char *file, int line, const char *label) {
    t_check(g_bad == 0, file, line, label, "%s: %d of %d vectors mismatched", label, g_bad, g_vec);
    t_log("%s: %d vectors", label, g_vec);
}
#define VEND(label) vend_(__FILE__, __LINE__, label)

static char g_hb[16][144];
static int g_hi;
static const char *hx(const void *p, size_t n) {           /* bytes in memory order */
    char *b = g_hb[g_hi++ & 15];
    const uint8_t *s = (const uint8_t *)p;
    size_t i;
    if (n > 64) n = 64;
    for (i = 0; i < n; i++) sprintf(b + 2 * i, "%02x", s[i]);
    b[2 * i] = 0;
    return b;
}
/* labelled byte-exact check */
static int chk_mem_(const void *got, const void *want, size_t n, const char *file, int line, const char *fmt, ...) {
    char lbl[200];
    va_list ap; va_start(ap, fmt); vsnprintf(lbl, sizeof lbl, fmt, ap); va_end(ap);
    return t_check_mem(got, want, n, file, line, lbl);
}
#define CHECK_BLK(got, want, n, ...) chk_mem_((got), (want), (n), __FILE__, __LINE__, __VA_ARGS__)

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
/* hex string (spaces ignored) -> bytes in the same order; returns the byte count */
static size_t unhex(const char *s, uint8_t *o, size_t cap) {
    size_t n = 0; int hi = -1;
    for (; *s; s++) {
        int v = hexval(*s);
        if (v < 0) continue;
        if (hi < 0) hi = v; else { if (n < cap) o[n] = (uint8_t)(hi << 4 | v); n++; hi = -1; }
    }
    return n;
}
/* a 128-bit number written most significant byte first (Intel white-paper notation) -> register image */
static V128 be128(const char *s) {
    uint8_t t[16]; V128 v; size_t n = unhex(s, t, 16);
    memset(&v, 0, sizeof v);
    for (size_t i = 0; i < n && i < 16; i++) v.b[i] = t[n - 1 - i];
    return v;
}
static uint32_t rd32be(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static uint64_t rd64be(const uint8_t *p) { return (uint64_t)rd32be(p) << 32 | rd32be(p + 4); }
static void wr32be(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static void wr64be(uint8_t *p, uint64_t v) { wr32be(p, (uint32_t)(v >> 32)); wr32be(p + 4, (uint32_t)v); }
static uint32_t rol32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
static uint32_t ror32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static uint64_t ror64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
static V128 vxor(V128 a, V128 b) { V128 r; r.q[0] = a.q[0] ^ b.q[0]; r.q[1] = a.q[1] ^ b.q[1]; return r; }

/* biased random 128-bit inputs: random, all-zero, all-ones, single bits, byte patterns */
static void gen128(t_rng *r, V128 *v) {
    t_rng_fill(r, v, 16);
    switch (t_rng_range(r, 10)) {
    case 0: memset(v, 0, 16); break;
    case 1: memset(v, 0xFF, 16); break;
    case 2: memset(v, 0, 16); v->b[t_rng_range(r, 16)] = (uint8_t)(1u << t_rng_range(r, 8)); break;
    case 3: memset(v, (int)t_rng_range(r, 256), 16); break;
    case 4: v->q[1] = 0; break;
    case 5: v->q[0] = ~0ull; break;
    default: break;
    }
}

/* ================================================================ AES software reference (FIPS-197) */
static uint8_t g_sbox[256], g_isbox[256];
static uint32_t g_crc32c_tab[256], g_crc32_tab[256];
static int g_init;

static uint8_t gf_mul8(uint8_t a, uint8_t b) {           /* GF(2^8) mod x^8+x^4+x^3+x+1 (0x11B) */
    uint8_t p = 0;
    while (b) { if (b & 1) p ^= a; a = (uint8_t)((a << 1) ^ ((a & 0x80) ? 0x1B : 0)); b >>= 1; }
    return p;
}
static uint8_t gf_inv8(uint8_t a) {                        /* a^254; inv(0) = 0 */
    uint8_t r = 1, x = a; int e = 254;
    if (!a) return 0;
    while (e) { if (e & 1) r = gf_mul8(r, x); x = gf_mul8(x, x); e >>= 1; }
    return r;
}
static uint8_t rotl8(uint8_t x, int n) { return (uint8_t)((x << n) | (x >> (8 - n))); }
static void crc_table(uint32_t poly_reflected, uint32_t *tab) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ poly_reflected : c >> 1;
        tab[i] = c;
    }
}
static void crypto_init(void) {
    if (g_init) return;
    for (int x = 0; x < 256; x++) {   /* S-box from its definition (FIPS-197 5.1.1), not a typed table */
        uint8_t b = gf_inv8((uint8_t)x);
        uint8_t s = (uint8_t)(b ^ rotl8(b, 1) ^ rotl8(b, 2) ^ rotl8(b, 3) ^ rotl8(b, 4) ^ 0x63);
        g_sbox[x] = s; g_isbox[s] = (uint8_t)x;
    }
    crc_table(0x82F63B78u, g_crc32c_tab);
    crc_table(0xEDB88320u, g_crc32_tab);
    g_init = 1;
}

/* state byte i = FIPS input byte i: s[r][c] = b[r + 4c] */
static V128 sw_shiftrows(V128 s)    { V128 o; for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) o.b[r + 4 * c] = s.b[r + 4 * ((c + r) & 3)]; return o; }
static V128 sw_invshiftrows(V128 s) { V128 o; for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) o.b[r + 4 * c] = s.b[r + 4 * ((c - r) & 3)]; return o; }
static V128 sw_subbytes(V128 s)     { for (int i = 0; i < 16; i++) s.b[i] = g_sbox[s.b[i]]; return s; }
static V128 sw_invsubbytes(V128 s)  { for (int i = 0; i < 16; i++) s.b[i] = g_isbox[s.b[i]]; return s; }
static V128 sw_mixcolumns(V128 s) {
    V128 o;
    for (int c = 0; c < 4; c++) {
        uint8_t a0 = s.b[4 * c], a1 = s.b[4 * c + 1], a2 = s.b[4 * c + 2], a3 = s.b[4 * c + 3];
        o.b[4 * c]     = (uint8_t)(gf_mul8(a0, 2) ^ gf_mul8(a1, 3) ^ a2 ^ a3);
        o.b[4 * c + 1] = (uint8_t)(a0 ^ gf_mul8(a1, 2) ^ gf_mul8(a2, 3) ^ a3);
        o.b[4 * c + 2] = (uint8_t)(a0 ^ a1 ^ gf_mul8(a2, 2) ^ gf_mul8(a3, 3));
        o.b[4 * c + 3] = (uint8_t)(gf_mul8(a0, 3) ^ a1 ^ a2 ^ gf_mul8(a3, 2));
    }
    return o;
}
static V128 sw_invmixcolumns(V128 s) {
    V128 o;
    for (int c = 0; c < 4; c++) {
        uint8_t a0 = s.b[4 * c], a1 = s.b[4 * c + 1], a2 = s.b[4 * c + 2], a3 = s.b[4 * c + 3];
        o.b[4 * c]     = (uint8_t)(gf_mul8(a0, 14) ^ gf_mul8(a1, 11) ^ gf_mul8(a2, 13) ^ gf_mul8(a3, 9));
        o.b[4 * c + 1] = (uint8_t)(gf_mul8(a0, 9) ^ gf_mul8(a1, 14) ^ gf_mul8(a2, 11) ^ gf_mul8(a3, 13));
        o.b[4 * c + 2] = (uint8_t)(gf_mul8(a0, 13) ^ gf_mul8(a1, 9) ^ gf_mul8(a2, 14) ^ gf_mul8(a3, 11));
        o.b[4 * c + 3] = (uint8_t)(gf_mul8(a0, 11) ^ gf_mul8(a1, 13) ^ gf_mul8(a2, 9) ^ gf_mul8(a3, 14));
    }
    return o;
}
/* SDM models of the single-round instructions */
static V128 m_aesenc(V128 s, V128 k)     { return vxor(sw_mixcolumns(sw_subbytes(sw_shiftrows(s))), k); }
static V128 m_aesenclast(V128 s, V128 k) { return vxor(sw_subbytes(sw_shiftrows(s)), k); }
static V128 m_aesdec(V128 s, V128 k)     { return vxor(sw_invmixcolumns(sw_invsubbytes(sw_invshiftrows(s))), k); }
static V128 m_aesdeclast(V128 s, V128 k) { return vxor(sw_invsubbytes(sw_invshiftrows(s)), k); }
static V128 m_aesimc(V128 s)             { return sw_invmixcolumns(s); }
static uint32_t subword(uint32_t w) {
    return (uint32_t)g_sbox[w & 0xFF] | (uint32_t)g_sbox[(w >> 8) & 0xFF] << 8 |
           (uint32_t)g_sbox[(w >> 16) & 0xFF] << 16 | (uint32_t)g_sbox[w >> 24] << 24;
}
/* AESKEYGENASSIST: X3=src[127:96], X1=src[63:32]; RotWord = ROR 8 of the little-endian dword */
static V128 m_aeskeygenassist(V128 s, uint8_t imm) {
    V128 o;
    o.d[0] = subword(s.d[1]);
    o.d[1] = ror32(subword(s.d[1]), 8) ^ imm;
    o.d[2] = subword(s.d[3]);
    o.d[3] = ror32(subword(s.d[3]), 8) ^ imm;
    return o;
}

typedef struct { int nr; V128 rk[15]; } aes_sched;
/* FIPS-197 5.2 KeyExpansion, word by word */
static void sw_expand(const uint8_t *key, int bits, aes_sched *ks) {
    int nk = bits / 32, nr = nk + 6, total = 4 * (nr + 1);
    uint32_t w[60]; uint8_t rcon = 1;
    for (int i = 0; i < nk; i++) w[i] = (uint32_t)key[4 * i] | (uint32_t)key[4 * i + 1] << 8 | (uint32_t)key[4 * i + 2] << 16 | (uint32_t)key[4 * i + 3] << 24;
    for (int i = nk; i < total; i++) {
        uint32_t t = w[i - 1];
        if (i % nk == 0) { t = subword(ror32(t, 8)) ^ rcon; rcon = gf_mul8(rcon, 2); }
        else if (nk > 6 && i % nk == 4) t = subword(t);
        w[i] = w[i - nk] ^ t;
    }
    ks->nr = nr;
    for (int r = 0; r <= nr; r++) for (int j = 0; j < 4; j++) ks->rk[r].d[j] = w[4 * r + j];
}
/* FIPS-197 5.1 Cipher and 5.3 InvCipher (straight form, not the equivalent inverse cipher) */
static V128 sw_encrypt(const aes_sched *ks, V128 s) {
    s = vxor(s, ks->rk[0]);
    for (int r = 1; r < ks->nr; r++) s = vxor(sw_mixcolumns(sw_shiftrows(sw_subbytes(s))), ks->rk[r]);
    return vxor(sw_shiftrows(sw_subbytes(s)), ks->rk[ks->nr]);
}
static V128 sw_decrypt(const aes_sched *ks, V128 s) {
    s = vxor(s, ks->rk[ks->nr]);
    for (int r = ks->nr - 1; r >= 1; r--) s = sw_invmixcolumns(vxor(sw_invsubbytes(sw_invshiftrows(s)), ks->rk[r]));
    return vxor(sw_invsubbytes(sw_invshiftrows(s)), ks->rk[0]);
}
static V128 ld128(const uint8_t *p) { V128 v; memcpy(v.b, p, 16); return v; }
static void inc128be(uint8_t c[16]) { for (int i = 15; i >= 0; i--) if (++c[i]) break; }
static void inc32be(uint8_t c[16])  { for (int i = 15; i >= 12; i--) if (++c[i]) break; }
static void sw_ecb(const aes_sched *ks, const uint8_t *in, uint8_t *out, size_t nb, int dec) {
    for (size_t i = 0; i < nb; i++) { V128 b = ld128(in + 16 * i); b = dec ? sw_decrypt(ks, b) : sw_encrypt(ks, b); memcpy(out + 16 * i, b.b, 16); }
}
static void sw_cbc(const aes_sched *ks, const uint8_t iv[16], const uint8_t *in, uint8_t *out, size_t nb, int dec) {
    V128 c = ld128(iv);
    for (size_t i = 0; i < nb; i++) {
        V128 x = ld128(in + 16 * i), y;
        if (!dec) { y = sw_encrypt(ks, vxor(x, c)); c = y; }
        else { y = vxor(sw_decrypt(ks, x), c); c = x; }
        memcpy(out + 16 * i, y.b, 16);
    }
}
static void sw_ctr(const aes_sched *ks, const uint8_t ctr0[16], const uint8_t *in, uint8_t *out, size_t n, int inc32) {
    uint8_t c[16]; memcpy(c, ctr0, 16);
    for (size_t off = 0; off < n; off += 16) {
        V128 k = sw_encrypt(ks, ld128(c));
        size_t m = n - off < 16 ? n - off : 16;
        for (size_t j = 0; j < m; j++) out[off + j] = in[off + j] ^ k.b[j];
        if (inc32) inc32be(c); else inc128be(c);
    }
}

/* ================================================================ AES-NI hardware paths (legacy SSE forms) */
enum { OP_ENC, OP_ENCLAST, OP_DEC, OP_DECLAST, OP_IMC };
static const char *const g_aesop_name[] = { "aesenc", "aesenclast", "aesdec", "aesdeclast", "aesimc" };
static V128 m_aesop(int op, V128 a, V128 k) {
    switch (op) { case OP_ENC: return m_aesenc(a, k); case OP_ENCLAST: return m_aesenclast(a, k);
                  case OP_DEC: return m_aesdec(a, k); case OP_DECLAST: return m_aesdeclast(a, k); default: return m_aesimc(a); }
}
T_TARGET("aes") T_NOINLINE static void hw_aesop(int op, const V128 *a, const V128 *k, V128 *o) {
    switch (op) {
    case OP_ENC:     o->v = _mm_aesenc_si128(a->v, k->v); break;
    case OP_ENCLAST: o->v = _mm_aesenclast_si128(a->v, k->v); break;
    case OP_DEC:     o->v = _mm_aesdec_si128(a->v, k->v); break;
    case OP_DECLAST: o->v = _mm_aesdeclast_si128(a->v, k->v); break;
    default:         o->v = _mm_aesimc_si128(a->v); break;
    }
}
/* same instructions with an (aligned) m128 source operand, written in asm so the encoding is fixed */
T_NOINLINE static void hw_aesop_mem(int op, const V128 *a, const V128 *k, V128 *o) {
    __m128i x = a->v;
    switch (op) {
    case OP_ENC:     __asm__ volatile("aesenc %1, %0" : "+x"(x) : "m"(*k)); break;
    case OP_ENCLAST: __asm__ volatile("aesenclast %1, %0" : "+x"(x) : "m"(*k)); break;
    case OP_DEC:     __asm__ volatile("aesdec %1, %0" : "+x"(x) : "m"(*k)); break;
    case OP_DECLAST: __asm__ volatile("aesdeclast %1, %0" : "+x"(x) : "m"(*k)); break;
    default:         __asm__ volatile("aesimc %1, %0" : "=x"(x) : "m"(*a)); break;
    }
    o->v = x;
}

#define I16_(X, h) X(0x##h##0) X(0x##h##1) X(0x##h##2) X(0x##h##3) X(0x##h##4) X(0x##h##5) X(0x##h##6) X(0x##h##7) \
                   X(0x##h##8) X(0x##h##9) X(0x##h##A) X(0x##h##B) X(0x##h##C) X(0x##h##D) X(0x##h##E) X(0x##h##F)
#define I256(X) I16_(X, 0) I16_(X, 1) I16_(X, 2) I16_(X, 3) I16_(X, 4) I16_(X, 5) I16_(X, 6) I16_(X, 7) \
                I16_(X, 8) I16_(X, 9) I16_(X, A) I16_(X, B) I16_(X, C) I16_(X, D) I16_(X, E) I16_(X, F)

#define KGA_CASE(i) case i: o->v = _mm_aeskeygenassist_si128(a->v, i); break;
T_TARGET("aes") T_NOINLINE static void hw_kga(const V128 *a, int imm, V128 *o) {
    switch (imm) { I256(KGA_CASE) default: break; }
}
T_NOINLINE static void hw_kga1_mem(const V128 *a, V128 *o) {
    __m128i x;
    __asm__ volatile("aeskeygenassist $0x1b, %1, %0" : "=x"(x) : "m"(*a));
    o->v = x;
}

/* Intel AES-NI white paper style key expansion for 128 and 256-bit keys */
T_TARGET("aes") static inline __m128i k128_step(__m128i t1, __m128i t2) {
    __m128i t3;
    t2 = _mm_shuffle_epi32(t2, 0xFF);
    t3 = _mm_slli_si128(t1, 4); t1 = _mm_xor_si128(t1, t3);
    t3 = _mm_slli_si128(t3, 4); t1 = _mm_xor_si128(t1, t3);
    t3 = _mm_slli_si128(t3, 4); t1 = _mm_xor_si128(t1, t3);
    return _mm_xor_si128(t1, t2);
}
#define K128(i, rc) t2 = _mm_aeskeygenassist_si128(t1, rc); t1 = k128_step(t1, t2); ks->rk[i].v = t1;
T_TARGET("aes") T_NOINLINE static void hw_expand128(const uint8_t *key, aes_sched *ks) {
    __m128i t1 = _mm_loadu_si128((const __m128i *)key), t2;
    ks->nr = 10; ks->rk[0].v = t1;
    K128(1, 0x01) K128(2, 0x02) K128(3, 0x04) K128(4, 0x08) K128(5, 0x10)
    K128(6, 0x20) K128(7, 0x40) K128(8, 0x80) K128(9, 0x1B) K128(10, 0x36)
}
T_TARGET("aes") static inline __m128i k256_a(__m128i t1, __m128i t2) { return k128_step(t1, t2); }
T_TARGET("aes") static inline __m128i k256_b(__m128i t1, __m128i t3) {
    __m128i t4 = _mm_aeskeygenassist_si128(t1, 0x00), t2 = _mm_shuffle_epi32(t4, 0xAA);
    t4 = _mm_slli_si128(t3, 4); t3 = _mm_xor_si128(t3, t4);
    t4 = _mm_slli_si128(t4, 4); t3 = _mm_xor_si128(t3, t4);
    t4 = _mm_slli_si128(t4, 4); t3 = _mm_xor_si128(t3, t4);
    return _mm_xor_si128(t3, t2);
}
#define K256(i, rc) t2 = _mm_aeskeygenassist_si128(t3, rc); t1 = k256_a(t1, t2); ks->rk[i].v = t1; \
                    t3 = k256_b(t1, t3); ks->rk[i + 1].v = t3;
T_TARGET("aes") T_NOINLINE static void hw_expand256(const uint8_t *key, aes_sched *ks) {
    __m128i t1 = _mm_loadu_si128((const __m128i *)key), t3 = _mm_loadu_si128((const __m128i *)(key + 16)), t2;
    ks->nr = 14; ks->rk[0].v = t1; ks->rk[1].v = t3;
    K256(2, 0x01) K256(4, 0x02) K256(6, 0x04) K256(8, 0x08) K256(10, 0x10) K256(12, 0x20)
    t2 = _mm_aeskeygenassist_si128(t3, 0x40); t1 = k256_a(t1, t2); ks->rk[14].v = t1;
}
/* generic word-wise expansion for every key size: AESKEYGENASSIST supplies SubWord/RotWord/Rcon */
#define RC_CASE(rc) case rc: o = _mm_aeskeygenassist_si128(x, rc); break;
T_TARGET("aes") T_NOINLINE static uint32_t hw_kga_word(uint32_t w, int rcon, int rot) {
    __m128i x = _mm_set_epi32(0, 0, (int)w, 0), o = x;
    switch (rcon) { RC_CASE(0x00) RC_CASE(0x01) RC_CASE(0x02) RC_CASE(0x04) RC_CASE(0x08) RC_CASE(0x10)
                    RC_CASE(0x20) RC_CASE(0x40) RC_CASE(0x80) RC_CASE(0x1B) RC_CASE(0x36) default: break; }
    V128 r; r.v = o;
    return rot ? r.d[1] : r.d[0];
}
static void hw_expand_wordwise(const uint8_t *key, int bits, aes_sched *ks) {
    static const int rcons[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36 };
    int nk = bits / 32, nr = nk + 6, total = 4 * (nr + 1), ri = 0;
    uint32_t w[60];
    for (int i = 0; i < nk; i++) memcpy(&w[i], key + 4 * i, 4);
    for (int i = nk; i < total; i++) {
        uint32_t t = w[i - 1];
        if (i % nk == 0) t = hw_kga_word(t, rcons[ri++], 1);
        else if (nk > 6 && i % nk == 4) t = hw_kga_word(t, 0, 0);
        w[i] = w[i - nk] ^ t;
    }
    ks->nr = nr;
    for (int r = 0; r <= nr; r++) memcpy(ks->rk[r].b, &w[4 * r], 16);
}
static void hw_expand(const uint8_t *key, int bits, aes_sched *ks) {
    if (bits == 128) hw_expand128(key, ks);
    else if (bits == 256) hw_expand256(key, ks);
    else hw_expand_wordwise(key, bits, ks);
}
/* decryption schedule for the equivalent inverse cipher (AESDEC): reversed, AESIMC on the middle keys */
T_TARGET("aes") T_NOINLINE static void hw_dec_sched(const aes_sched *ks, aes_sched *dk) {
    int nr = ks->nr;
    dk->nr = nr; dk->rk[0] = ks->rk[nr]; dk->rk[nr] = ks->rk[0];
    for (int i = 1; i < nr; i++) dk->rk[i].v = _mm_aesimc_si128(ks->rk[nr - i].v);
}
T_TARGET("aes") T_NOINLINE static void hw_ecb_enc(const aes_sched *ks, const uint8_t *in, uint8_t *out, size_t nb) {
    for (size_t i = 0; i < nb; i++) {
        __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in + 16 * i)), ks->rk[0].v);
        for (int r = 1; r < ks->nr; r++) x = _mm_aesenc_si128(x, ks->rk[r].v);
        _mm_storeu_si128((__m128i *)(out + 16 * i), _mm_aesenclast_si128(x, ks->rk[ks->nr].v));
    }
}
T_TARGET("aes") T_NOINLINE static void hw_ecb_dec(const aes_sched *dk, const uint8_t *in, uint8_t *out, size_t nb) {
    for (size_t i = 0; i < nb; i++) {
        __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in + 16 * i)), dk->rk[0].v);
        for (int r = 1; r < dk->nr; r++) x = _mm_aesdec_si128(x, dk->rk[r].v);
        _mm_storeu_si128((__m128i *)(out + 16 * i), _mm_aesdeclast_si128(x, dk->rk[dk->nr].v));
    }
}
T_TARGET("aes") T_NOINLINE static void hw_cbc_enc(const aes_sched *ks, const uint8_t iv[16], const uint8_t *in, uint8_t *out, size_t nb) {
    __m128i c = _mm_loadu_si128((const __m128i *)iv);
    for (size_t i = 0; i < nb; i++) {
        __m128i x = _mm_xor_si128(_mm_xor_si128(_mm_loadu_si128((const __m128i *)(in + 16 * i)), c), ks->rk[0].v);
        for (int r = 1; r < ks->nr; r++) x = _mm_aesenc_si128(x, ks->rk[r].v);
        c = _mm_aesenclast_si128(x, ks->rk[ks->nr].v);
        _mm_storeu_si128((__m128i *)(out + 16 * i), c);
    }
}
/* CBC decryption, 4 blocks in flight (the common software pipeline shape) */
T_TARGET("aes") T_NOINLINE static void hw_cbc_dec(const aes_sched *dk, const uint8_t iv[16], const uint8_t *in, uint8_t *out, size_t nb) {
    __m128i prev = _mm_loadu_si128((const __m128i *)iv);
    size_t i = 0;
    for (; i + 4 <= nb; i += 4) {
        __m128i c0 = _mm_loadu_si128((const __m128i *)(in + 16 * i)), c1 = _mm_loadu_si128((const __m128i *)(in + 16 * i + 16));
        __m128i c2 = _mm_loadu_si128((const __m128i *)(in + 16 * i + 32)), c3 = _mm_loadu_si128((const __m128i *)(in + 16 * i + 48));
        __m128i x0 = _mm_xor_si128(c0, dk->rk[0].v), x1 = _mm_xor_si128(c1, dk->rk[0].v);
        __m128i x2 = _mm_xor_si128(c2, dk->rk[0].v), x3 = _mm_xor_si128(c3, dk->rk[0].v);
        for (int r = 1; r < dk->nr; r++) {
            x0 = _mm_aesdec_si128(x0, dk->rk[r].v); x1 = _mm_aesdec_si128(x1, dk->rk[r].v);
            x2 = _mm_aesdec_si128(x2, dk->rk[r].v); x3 = _mm_aesdec_si128(x3, dk->rk[r].v);
        }
        x0 = _mm_aesdeclast_si128(x0, dk->rk[dk->nr].v); x1 = _mm_aesdeclast_si128(x1, dk->rk[dk->nr].v);
        x2 = _mm_aesdeclast_si128(x2, dk->rk[dk->nr].v); x3 = _mm_aesdeclast_si128(x3, dk->rk[dk->nr].v);
        _mm_storeu_si128((__m128i *)(out + 16 * i), _mm_xor_si128(x0, prev));
        _mm_storeu_si128((__m128i *)(out + 16 * i + 16), _mm_xor_si128(x1, c0));
        _mm_storeu_si128((__m128i *)(out + 16 * i + 32), _mm_xor_si128(x2, c1));
        _mm_storeu_si128((__m128i *)(out + 16 * i + 48), _mm_xor_si128(x3, c2));
        prev = c3;
    }
    for (; i < nb; i++) {
        __m128i c = _mm_loadu_si128((const __m128i *)(in + 16 * i)), x = _mm_xor_si128(c, dk->rk[0].v);
        for (int r = 1; r < dk->nr; r++) x = _mm_aesdec_si128(x, dk->rk[r].v);
        _mm_storeu_si128((__m128i *)(out + 16 * i), _mm_xor_si128(_mm_aesdeclast_si128(x, dk->rk[dk->nr].v), prev));
        prev = c;
    }
}
/* CTR, 8 blocks in flight (uses every XMM register on x64); ctr is advanced; tail handled */
T_TARGET("aes") T_NOINLINE static void hw_ctr(const aes_sched *ks, uint8_t ctr[16], const uint8_t *in, uint8_t *out, size_t n, int inc32, int ways) {
    const int nr = ks->nr;
    if (ways == 8) {
        while (n >= 128) {
            __m128i b[8];
            for (int j = 0; j < 8; j++) {
                b[j] = _mm_xor_si128(_mm_loadu_si128((const __m128i *)ctr), ks->rk[0].v);
                if (inc32) inc32be(ctr); else inc128be(ctr);
            }
            for (int r = 1; r < nr; r++) {
                __m128i k = ks->rk[r].v;
                b[0] = _mm_aesenc_si128(b[0], k); b[1] = _mm_aesenc_si128(b[1], k); b[2] = _mm_aesenc_si128(b[2], k); b[3] = _mm_aesenc_si128(b[3], k);
                b[4] = _mm_aesenc_si128(b[4], k); b[5] = _mm_aesenc_si128(b[5], k); b[6] = _mm_aesenc_si128(b[6], k); b[7] = _mm_aesenc_si128(b[7], k);
            }
            for (int j = 0; j < 8; j++)
                _mm_storeu_si128((__m128i *)(out + 16 * j), _mm_xor_si128(_mm_aesenclast_si128(b[j], ks->rk[nr].v), _mm_loadu_si128((const __m128i *)(in + 16 * j))));
            in += 128; out += 128; n -= 128;
        }
    }
    while (n) {
        __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i *)ctr), ks->rk[0].v);
        for (int r = 1; r < nr; r++) x = _mm_aesenc_si128(x, ks->rk[r].v);
        V128 k; k.v = _mm_aesenclast_si128(x, ks->rk[nr].v);
        size_t m = n < 16 ? n : 16;
        for (size_t j = 0; j < m; j++) out[j] = in[j] ^ k.b[j];
        if (inc32) inc32be(ctr); else inc128be(ctr);
        in += m; out += m; n -= m;
    }
}

/* ================================================================ carry-less multiply and GHASH */
static void sw_clmul64(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi) {
    uint64_t l = 0, h = 0;
    for (int i = 0; i < 64; i++) if ((b >> i) & 1) { l ^= a << i; if (i) h ^= a >> (64 - i); }
    *lo = l; *hi = h;
}
static V128 m_pclmul(V128 a, V128 b, int imm) {
    V128 r; sw_clmul64(a.q[imm & 1], b.q[(imm >> 4) & 1], &r.q[0], &r.q[1]); return r;
}
#define CLM_CASE(i) case i: o->v = _mm_clmulepi64_si128(a->v, b->v, i); break;
T_TARGET("pclmul") T_NOINLINE static void hw_pclmul(const V128 *a, const V128 *b, int imm, V128 *o) {
    switch (imm) { I256(CLM_CASE) default: break; }
}
T_NOINLINE static void hw_pclmul_mem(const V128 *a, const V128 *b, int sel, V128 *o) {
    __m128i x = a->v;
    switch (sel) {
    case 0: __asm__ volatile("pclmulqdq $0x00, %1, %0" : "+x"(x) : "m"(*b)); break;
    case 1: __asm__ volatile("pclmulqdq $0x01, %1, %0" : "+x"(x) : "m"(*b)); break;
    case 2: __asm__ volatile("pclmulqdq $0x10, %1, %0" : "+x"(x) : "m"(*b)); break;
    default: __asm__ volatile("pclmulqdq $0x11, %1, %0" : "+x"(x) : "m"(*b)); break;
    }
    o->v = x;
}
/* GF(2^128) multiply per SP 800-38D 6.3 Algorithm 1 (GCM bit order: bit 0 = MSB of byte 0) */
static void sw_gf128_mul(const uint8_t X[16], const uint8_t Y[16], uint8_t Z[16]) {
    uint8_t z[16] = { 0 }, v[16];
    memcpy(v, Y, 16);
    for (int i = 0; i < 128; i++) {
        if ((X[i >> 3] >> (7 - (i & 7))) & 1) for (int j = 0; j < 16; j++) z[j] ^= v[j];
        int lsb = v[15] & 1;
        for (int j = 15; j > 0; j--) v[j] = (uint8_t)((v[j] >> 1) | (v[j - 1] << 7));
        v[0] >>= 1;
        if (lsb) v[0] ^= 0xE1;
    }
    memcpy(Z, z, 16);
}
/* Gueron/Kounavis gfmul on byte-reflected operands: four PCLMULQDQ, 1-bit shift, reduction */
T_TARGET("pclmul") static inline __m128i gfmul_pclmul(__m128i a, __m128i b) {
    __m128i t2, t3, t4, t5, t6, t7, t8, t9;
    t3 = _mm_clmulepi64_si128(a, b, 0x00);
    t4 = _mm_clmulepi64_si128(a, b, 0x10);
    t5 = _mm_clmulepi64_si128(a, b, 0x01);
    t6 = _mm_clmulepi64_si128(a, b, 0x11);
    t4 = _mm_xor_si128(t4, t5);
    t5 = _mm_slli_si128(t4, 8); t4 = _mm_srli_si128(t4, 8);
    t3 = _mm_xor_si128(t3, t5); t6 = _mm_xor_si128(t6, t4);
    t7 = _mm_srli_epi32(t3, 31); t8 = _mm_srli_epi32(t6, 31);
    t3 = _mm_slli_epi32(t3, 1);  t6 = _mm_slli_epi32(t6, 1);
    t9 = _mm_srli_si128(t7, 12); t8 = _mm_slli_si128(t8, 4); t7 = _mm_slli_si128(t7, 4);
    t3 = _mm_or_si128(t3, t7); t6 = _mm_or_si128(t6, t8); t6 = _mm_or_si128(t6, t9);
    t7 = _mm_slli_epi32(t3, 31); t8 = _mm_slli_epi32(t3, 30); t9 = _mm_slli_epi32(t3, 25);
    t7 = _mm_xor_si128(t7, t8); t7 = _mm_xor_si128(t7, t9);
    t8 = _mm_srli_si128(t7, 4); t7 = _mm_slli_si128(t7, 12);
    t3 = _mm_xor_si128(t3, t7);
    t2 = _mm_srli_epi32(t3, 1); t4 = _mm_srli_epi32(t3, 2); t5 = _mm_srli_epi32(t3, 7);
    t2 = _mm_xor_si128(t2, t4); t2 = _mm_xor_si128(t2, t5); t2 = _mm_xor_si128(t2, t8);
    t3 = _mm_xor_si128(t3, t2);
    return _mm_xor_si128(t6, t3);
}
static V128 rev16(const uint8_t *p) { V128 v; for (int i = 0; i < 16; i++) v.b[i] = p[15 - i]; return v; }
T_TARGET("pclmul") T_NOINLINE static void hw_gf128_mul(const uint8_t X[16], const uint8_t Y[16], uint8_t Z[16]) {
    V128 a = rev16(X), b = rev16(Y), r;
    r.v = gfmul_pclmul(a.v, b.v);
    for (int i = 0; i < 16; i++) Z[i] = r.b[15 - i];
}
typedef void (*gf_mul_fn)(const uint8_t *, const uint8_t *, uint8_t *);
static void ghash(gf_mul_fn mul, const uint8_t H[16], const uint8_t *A, size_t alen, const uint8_t *C, size_t clen, uint8_t S[16]) {
    uint8_t y[16] = { 0 }, blk[16];
    for (int part = 0; part < 2; part++) {
        const uint8_t *p = part ? C : A; size_t n = part ? clen : alen;
        for (size_t off = 0; off < n; off += 16) {
            size_t m = n - off < 16 ? n - off : 16;
            memset(blk, 0, 16); memcpy(blk, p + off, m);
            for (int j = 0; j < 16; j++) y[j] ^= blk[j];
            mul(y, H, y);
        }
    }
    wr64be(blk, (uint64_t)alen * 8); wr64be(blk + 8, (uint64_t)clen * 8);
    for (int j = 0; j < 16; j++) y[j] ^= blk[j];
    mul(y, H, y);
    memcpy(S, y, 16);
}
/* AES-GCM with a 96-bit IV (SP 800-38D 7.1/7.2); hw = AES-NI + PCLMULQDQ, else software */
static void aes_gcm(int hw, const uint8_t *key, int bits, const uint8_t iv[12], const uint8_t *A, size_t alen,
                    const uint8_t *in, size_t n, uint8_t *out, uint8_t tag[16], int decrypt) {
    aes_sched ks; uint8_t H[16] = { 0 }, J0[16], ctr[16], S[16];
    V128 z, e;
    memset(&z, 0, sizeof z);
    if (hw) { hw_expand(key, bits, &ks); hw_ecb_enc(&ks, z.b, H, 1); }
    else { sw_expand(key, bits, &ks); e = sw_encrypt(&ks, z); memcpy(H, e.b, 16); }
    memcpy(J0, iv, 12); J0[12] = J0[13] = J0[14] = 0; J0[15] = 1;
    memcpy(ctr, J0, 16); inc32be(ctr);
    if (hw) hw_ctr(&ks, ctr, in, out, n, 1, 8); else sw_ctr(&ks, ctr, in, out, n, 1);
    ghash(hw ? hw_gf128_mul : sw_gf128_mul, H, A, alen, decrypt ? in : out, n, S);
    if (hw) hw_ecb_enc(&ks, J0, e.b, 1); else e = sw_encrypt(&ks, ld128(J0));
    for (int j = 0; j < 16; j++) tag[j] = S[j] ^ e.b[j];
}

/* ================================================================ CRC-32 / CRC-32C: table, PCLMUL folding, SSE4.2 */
static uint32_t crc_raw(const uint32_t *tab, uint32_t c, const uint8_t *p, size_t n) {
    while (n--) c = tab[(c ^ *p++) & 0xFF] ^ (c >> 8);
    return c;
}
static uint32_t crc_sw(const uint32_t *tab, const uint8_t *p, size_t n) { return ~crc_raw(tab, 0xFFFFFFFFu, p, n); }
/* Fold constant for the reflected domain: clmul(q, k) reflected = x * rev64(q) * rev64(k), so rev64(k) =
 * x^(e) mod P with e = shift - 1; the 32-bit remainder (bit d = x^d) is placed at bit 63-d of k. */
static uint64_t fold_const(uint32_t poly_normal, int e) {
    uint64_t r = 1, k = 0;
    for (int i = 0; i < e; i++) { r <<= 1; if (r & 0x100000000ull) r ^= 0x100000000ull | poly_normal; }
    for (int d = 0; d < 32; d++) if ((r >> d) & 1) k |= 1ull << (63 - d);
    return k;
}
typedef struct { V128 k128, k512; const uint32_t *tab; } crc_fold;
static void crc_fold_init(crc_fold *f, uint32_t poly_normal, const uint32_t *tab) {
    f->k128.q[0] = fold_const(poly_normal, 128 + 64 - 1); f->k128.q[1] = fold_const(poly_normal, 128 - 1);
    f->k512.q[0] = fold_const(poly_normal, 512 + 64 - 1); f->k512.q[1] = fold_const(poly_normal, 512 - 1);
    f->tab = tab;
}
T_TARGET("pclmul") static inline __m128i fold16(__m128i x, __m128i k) {
    return _mm_xor_si128(_mm_clmulepi64_si128(x, k, 0x00), _mm_clmulepi64_si128(x, k, 0x11));
}
T_TARGET("pclmul") T_NOINLINE static uint32_t hw_crc_fold(const crc_fold *f, const uint8_t *p, size_t n, int by4) {
    if (n < 16) return crc_sw(f->tab, p, n);
    const __m128i k1 = f->k128.v, k4 = f->k512.v;
    __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i *)p), _mm_cvtsi32_si128(-1));
    p += 16; n -= 16;
    if (by4 && n >= 48) {
        __m128i x1 = _mm_loadu_si128((const __m128i *)p), x2 = _mm_loadu_si128((const __m128i *)(p + 16));
        __m128i x3 = _mm_loadu_si128((const __m128i *)(p + 32));
        p += 48; n -= 48;
        while (n >= 64) {
            x  = _mm_xor_si128(fold16(x, k4),  _mm_loadu_si128((const __m128i *)p));
            x1 = _mm_xor_si128(fold16(x1, k4), _mm_loadu_si128((const __m128i *)(p + 16)));
            x2 = _mm_xor_si128(fold16(x2, k4), _mm_loadu_si128((const __m128i *)(p + 32)));
            x3 = _mm_xor_si128(fold16(x3, k4), _mm_loadu_si128((const __m128i *)(p + 48)));
            p += 64; n -= 64;
        }
        x = _mm_xor_si128(fold16(x, k1), x1);
        x = _mm_xor_si128(fold16(x, k1), x2);
        x = _mm_xor_si128(fold16(x, k1), x3);
    }
    while (n >= 16) { x = _mm_xor_si128(fold16(x, k1), _mm_loadu_si128((const __m128i *)p)); p += 16; n -= 16; }
    V128 r; r.v = x;
    return ~crc_raw(f->tab, crc_raw(f->tab, 0, r.b, 16), p, n);
}
T_TARGET("sse4.2") T_NOINLINE static uint32_t hw_crc32c_sse42(const uint8_t *p, size_t n, int width) {
    uint32_t c = 0xFFFFFFFFu;
#ifdef T_X64
    if (width == 8) { uint64_t c64 = c; for (; n >= 8; p += 8, n -= 8) { uint64_t v; memcpy(&v, p, 8); c64 = _mm_crc32_u64(c64, v); } c = (uint32_t)c64; }
#endif
    if (width >= 4) for (; n >= 4; p += 4, n -= 4) { uint32_t v; memcpy(&v, p, 4); c = _mm_crc32_u32(c, v); }
    if (width >= 2) for (; n >= 2; p += 2, n -= 2) { uint16_t v; memcpy(&v, p, 2); c = _mm_crc32_u16(c, v); }
    for (; n; p++, n--) c = _mm_crc32_u8(c, *p);
    return ~c;
}

/* ================================================================ SHA software references (FIPS 180-4) */
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
static const uint64_t K512[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,
    0x3956c25bf348b538ull, 0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,
    0xd807aa98a3030242ull, 0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
    0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, 0xc19bf174cf692694ull,
    0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
    0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
    0x983e5152ee66dfabull, 0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull,
    0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull, 0x06ca6351e003826full, 0x142929670a0e6e70ull,
    0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
    0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
    0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull,
    0xd192e819d6ef5218ull, 0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
    0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, 0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull,
    0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull,
    0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull,
    0xca273eceea26619cull, 0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull,
    0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull, 0x113f9804bef90daeull, 0x1b710b35131c471bull,
    0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull, 0x431d67c49c100d4cull,
    0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull };

static uint32_t S0_256(uint32_t x) { return ror32(x, 2) ^ ror32(x, 13) ^ ror32(x, 22); }
static uint32_t S1_256(uint32_t x) { return ror32(x, 6) ^ ror32(x, 11) ^ ror32(x, 25); }
static uint32_t s0_256(uint32_t x) { return ror32(x, 7) ^ ror32(x, 18) ^ (x >> 3); }
static uint32_t s1_256(uint32_t x) { return ror32(x, 17) ^ ror32(x, 19) ^ (x >> 10); }
static uint64_t S0_512(uint64_t x) { return ror64(x, 28) ^ ror64(x, 34) ^ ror64(x, 39); }
static uint64_t S1_512(uint64_t x) { return ror64(x, 14) ^ ror64(x, 18) ^ ror64(x, 41); }
static uint64_t s0_512(uint64_t x) { return ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7); }
static uint64_t s1_512(uint64_t x) { return ror64(x, 19) ^ ror64(x, 61) ^ (x >> 6); }
#define CH(e, f, g)  (((e) & (f)) ^ (~(e) & (g)))
#define MAJ(a, b, c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))

enum { ALG_SHA1, ALG_SHA256, ALG_SHA512 };
typedef void (*compress_fn)(void *st, const uint8_t *p, size_t nblocks);

static void sw_sha1_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint32_t *st = (uint32_t *)stv;
    for (; nb; nb--, p += 64) {
        uint32_t w[80], a = st[0], b = st[1], c = st[2], d = st[3], e = st[4];
        for (int t = 0; t < 16; t++) w[t] = rd32be(p + 4 * t);
        for (int t = 16; t < 80; t++) w[t] = rol32(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);
        for (int t = 0; t < 80; t++) {
            uint32_t f, k;
            if (t < 20) { f = CH(b, c, d); k = 0x5A827999; }
            else if (t < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (t < 60) { f = MAJ(b, c, d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            uint32_t tmp = rol32(a, 5) + f + e + k + w[t];
            e = d; d = c; c = rol32(b, 30); b = a; a = tmp;
        }
        st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e;
    }
}
static void sw_sha256_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint32_t *st = (uint32_t *)stv;
    for (; nb; nb--, p += 64) {
        uint32_t w[64], v[8];
        for (int t = 0; t < 16; t++) w[t] = rd32be(p + 4 * t);
        for (int t = 16; t < 64; t++) w[t] = s1_256(w[t - 2]) + w[t - 7] + s0_256(w[t - 15]) + w[t - 16];
        memcpy(v, st, sizeof v);
        for (int t = 0; t < 64; t++) {
            uint32_t t1 = v[7] + S1_256(v[4]) + CH(v[4], v[5], v[6]) + K256[t] + w[t];
            uint32_t t2 = S0_256(v[0]) + MAJ(v[0], v[1], v[2]);
            memmove(v + 1, v, 7 * sizeof v[0]);
            v[4] += t1; v[0] = t1 + t2;
        }
        for (int i = 0; i < 8; i++) st[i] += v[i];
    }
}
static void sw_sha512_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint64_t *st = (uint64_t *)stv;
    for (; nb; nb--, p += 128) {
        uint64_t w[80], v[8];
        for (int t = 0; t < 16; t++) w[t] = rd64be(p + 8 * t);
        for (int t = 16; t < 80; t++) w[t] = s1_512(w[t - 2]) + w[t - 7] + s0_512(w[t - 15]) + w[t - 16];
        memcpy(v, st, sizeof v);
        for (int t = 0; t < 80; t++) {
            uint64_t t1 = v[7] + S1_512(v[4]) + CH(v[4], v[5], v[6]) + K512[t] + w[t];
            uint64_t t2 = S0_512(v[0]) + MAJ(v[0], v[1], v[2]);
            memmove(v + 1, v, 7 * sizeof v[0]);
            v[4] += t1; v[0] = t1 + t2;
        }
        for (int i = 0; i < 8; i++) st[i] += v[i];
    }
}
/* Merkle-Damgard padding + digest output; the compression function is pluggable (sw / model / hw) */
static size_t md_digest(int alg, compress_fn fn, const uint8_t *msg, size_t len, uint8_t *out) {
    uint32_t s32[8]; uint64_t s64[8];
    uint8_t tail[256];
    size_t bs = alg == ALG_SHA512 ? 128 : 64, lenfield = alg == ALG_SHA512 ? 16 : 8;
    void *st;
    if (alg == ALG_SHA1) { static const uint32_t iv[5] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0 }; memcpy(s32, iv, sizeof iv); st = s32; }
    else if (alg == ALG_SHA256) { static const uint32_t iv[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 }; memcpy(s32, iv, sizeof iv); st = s32; }
    else { static const uint64_t iv[8] = { 0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                                          0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull }; memcpy(s64, iv, sizeof iv); st = s64; }
    size_t full = len / bs;
    if (full) fn(st, msg, full);
    size_t rem = len - full * bs, tl = (rem + 1 + lenfield <= bs) ? bs : 2 * bs;
    memset(tail, 0, sizeof tail);
    memcpy(tail, msg + full * bs, rem);
    tail[rem] = 0x80;
    wr64be(tail + tl - 8, (uint64_t)len * 8);
    fn(st, tail, tl / bs);
    if (alg == ALG_SHA1) { for (int i = 0; i < 5; i++) wr32be(out + 4 * i, s32[i]); return 20; }
    if (alg == ALG_SHA256) { for (int i = 0; i < 8; i++) wr32be(out + 4 * i, s32[i]); return 32; }
    for (int i = 0; i < 8; i++) wr64be(out + 8 * i, s64[i]);
    return 64;
}

/* ---- SDM models of the SHA instructions (dword/qword lanes, lane 0 = bits 31:0) ---- */
static uint32_t sha1_f(int fn, uint32_t b, uint32_t c, uint32_t d) {
    return fn == 0 ? CH(b, c, d) : fn == 2 ? MAJ(b, c, d) : (b ^ c ^ d);
}
static const uint32_t SHA1_K[4] = { 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6 };
static V128 m_sha1rnds4(V128 s1, V128 s2, int imm) {
    int fn = imm & 3;
    uint32_t A = s1.d[3], B = s1.d[2], C = s1.d[1], D = s1.d[0], E = 0, W[4] = { s2.d[3], s2.d[2], s2.d[1], s2.d[0] };
    for (int i = 0; i < 4; i++) {
        uint32_t nA = sha1_f(fn, B, C, D) + rol32(A, 5) + W[i] + (i ? E : 0) + SHA1_K[fn];
        E = D; D = C; C = rol32(B, 30); B = A; A = nA;
    }
    V128 o; o.d[3] = A; o.d[2] = B; o.d[1] = C; o.d[0] = D; return o;
}
static V128 m_sha1nexte(V128 s1, V128 s2) { V128 o = s2; o.d[3] = s2.d[3] + rol32(s1.d[3], 30); return o; }
static V128 m_sha1msg1(V128 s1, V128 s2) {
    V128 o; o.d[3] = s1.d[1] ^ s1.d[3]; o.d[2] = s1.d[0] ^ s1.d[2]; o.d[1] = s2.d[3] ^ s1.d[1]; o.d[0] = s2.d[2] ^ s1.d[0]; return o;
}
static V128 m_sha1msg2(V128 s1, V128 s2) {
    uint32_t w16 = rol32(s1.d[3] ^ s2.d[2], 1), w17 = rol32(s1.d[2] ^ s2.d[1], 1);
    uint32_t w18 = rol32(s1.d[1] ^ s2.d[0], 1), w19 = rol32(s1.d[0] ^ w16, 1);
    V128 o; o.d[3] = w16; o.d[2] = w17; o.d[1] = w18; o.d[0] = w19; return o;
}
static V128 m_sha256rnds2(V128 s1, V128 s2, V128 wk) {
    uint32_t A = s2.d[3], B = s2.d[2], E = s2.d[1], F = s2.d[0], C = s1.d[3], D = s1.d[2], G = s1.d[1], H = s1.d[0];
    for (int i = 0; i < 2; i++) {
        uint32_t t1 = CH(E, F, G) + S1_256(E) + wk.d[i] + H, nA = t1 + MAJ(A, B, C) + S0_256(A), nE = t1 + D;
        H = G; G = F; F = E; E = nE; D = C; C = B; B = A; A = nA;
    }
    V128 o; o.d[3] = A; o.d[2] = B; o.d[1] = E; o.d[0] = F; return o;
}
static V128 m_sha256msg1(V128 s1, V128 s2) {
    V128 o; o.d[3] = s1.d[3] + s0_256(s2.d[0]); o.d[2] = s1.d[2] + s0_256(s1.d[3]);
    o.d[1] = s1.d[1] + s0_256(s1.d[2]); o.d[0] = s1.d[0] + s0_256(s1.d[1]); return o;
}
static V128 m_sha256msg2(V128 s1, V128 s2) {
    uint32_t w16 = s1.d[0] + s1_256(s2.d[2]), w17 = s1.d[1] + s1_256(s2.d[3]);
    uint32_t w18 = s1.d[2] + s1_256(w16), w19 = s1.d[3] + s1_256(w17);
    V128 o; o.d[0] = w16; o.d[1] = w17; o.d[2] = w18; o.d[3] = w19; return o;
}
typedef struct { uint64_t q[4]; } Q4;
static Q4 m_sha512msg1(Q4 s1, uint64_t w4) {
    Q4 o; for (int i = 0; i < 4; i++) o.q[i] = s1.q[i] + s0_512(i < 3 ? s1.q[i + 1] : w4); return o;
}
static Q4 m_sha512msg2(Q4 s1, Q4 s2) {
    Q4 o; o.q[0] = s1.q[0] + s1_512(s2.q[2]); o.q[1] = s1.q[1] + s1_512(s2.q[3]);
    o.q[2] = s1.q[2] + s1_512(o.q[0]); o.q[3] = s1.q[3] + s1_512(o.q[1]); return o;
}
static Q4 m_sha512rnds2(Q4 s1, Q4 s2, uint64_t wk0, uint64_t wk1) {
    uint64_t A = s2.q[3], B = s2.q[2], E = s2.q[1], F = s2.q[0], C = s1.q[3], D = s1.q[2], G = s1.q[1], H = s1.q[0];
    uint64_t wk[2] = { wk0, wk1 };
    for (int i = 0; i < 2; i++) {
        uint64_t t1 = CH(E, F, G) + S1_512(E) + wk[i] + H, nA = t1 + MAJ(A, B, C) + S0_512(A), nE = t1 + D;
        H = G; G = F; F = E; E = nE; D = C; C = B; B = A; A = nA;
    }
    Q4 o; o.q[3] = A; o.q[2] = B; o.q[1] = E; o.q[0] = F; return o;
}
static V128 v_add32(V128 a, V128 b) { V128 o; for (int i = 0; i < 4; i++) o.d[i] = a.d[i] + b.d[i]; return o; }
static V128 v_alignr4(V128 hi, V128 lo) { V128 o; o.d[0] = lo.d[1]; o.d[1] = lo.d[2]; o.d[2] = lo.d[3]; o.d[3] = hi.d[0]; return o; }
static V128 v_set4(uint32_t d3, uint32_t d2, uint32_t d1, uint32_t d0) { V128 o; o.d[3] = d3; o.d[2] = d2; o.d[1] = d1; o.d[0] = d0; return o; }

/* model-composed compression functions: identical dataflow to the hardware paths below */
static void model_sha1_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint32_t *st = (uint32_t *)stv;
    V128 abcd = v_set4(st[0], st[1], st[2], st[3]), e0 = v_set4(st[4], 0, 0, 0);
    for (; nb; nb--, p += 64) {
        V128 W[4], save = abcd, esave = e0, prev = abcd, e;
        for (int i = 0; i < 4; i++) W[i] = v_set4(rd32be(p + 16 * i), rd32be(p + 16 * i + 4), rd32be(p + 16 * i + 8), rd32be(p + 16 * i + 12));
        for (int g = 0; g < 20; g++) {
            V128 w;
            if (g < 4) w = W[g];
            else { w = m_sha1msg2(vxor(m_sha1msg1(W[g & 3], W[(g + 1) & 3]), W[(g + 2) & 3]), W[(g + 3) & 3]); W[g & 3] = w; }
            e = g == 0 ? v_add32(e0, w) : m_sha1nexte(prev, w);
            prev = abcd; abcd = m_sha1rnds4(abcd, e, g / 5);
        }
        e0 = m_sha1nexte(prev, esave);
        abcd = v_add32(abcd, save);
        e0.d[2] = e0.d[1] = e0.d[0] = 0;
    }
    st[0] = abcd.d[3]; st[1] = abcd.d[2]; st[2] = abcd.d[1]; st[3] = abcd.d[0]; st[4] = e0.d[3];
}
static void model_sha256_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint32_t *st = (uint32_t *)stv;
    V128 s0 = v_set4(st[0], st[1], st[4], st[5]), s1 = v_set4(st[2], st[3], st[6], st[7]);
    for (; nb; nb--, p += 64) {
        V128 W[4], sv0 = s0, sv1 = s1;
        for (int i = 0; i < 4; i++) W[i] = v_set4(rd32be(p + 16 * i + 12), rd32be(p + 16 * i + 8), rd32be(p + 16 * i + 4), rd32be(p + 16 * i));
        for (int g = 0; g < 16; g++) {
            if (g >= 4) W[g & 3] = m_sha256msg2(v_add32(m_sha256msg1(W[g & 3], W[(g + 1) & 3]), v_alignr4(W[(g + 3) & 3], W[(g + 2) & 3])), W[(g + 3) & 3]);
            V128 msg = v_add32(W[g & 3], v_set4(K256[4 * g + 3], K256[4 * g + 2], K256[4 * g + 1], K256[4 * g]));
            s1 = m_sha256rnds2(s1, s0, msg);
            V128 hi = msg; hi.d[0] = msg.d[2]; hi.d[1] = msg.d[3];
            s0 = m_sha256rnds2(s0, s1, hi);
        }
        s0 = v_add32(s0, sv0); s1 = v_add32(s1, sv1);
    }
    st[0] = s0.d[3]; st[1] = s0.d[2]; st[4] = s0.d[1]; st[5] = s0.d[0];
    st[2] = s1.d[3]; st[3] = s1.d[2]; st[6] = s1.d[1]; st[7] = s1.d[0];
}
static Q4 q4_add(Q4 a, Q4 b) { Q4 o; for (int i = 0; i < 4; i++) o.q[i] = a.q[i] + b.q[i]; return o; }
static void model_sha512_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint64_t *st = (uint64_t *)stv;
    Q4 s0 = { { st[5], st[4], st[1], st[0] } }, s1 = { { st[7], st[6], st[3], st[2] } };
    for (; nb; nb--, p += 128) {
        Q4 W[4], sv0 = s0, sv1 = s1;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) W[i].q[j] = rd64be(p + 32 * i + 8 * j);
        for (int g = 0; g < 20; g++) {
            if (g >= 4) {
                Q4 lo = W[(g + 2) & 3], hi = W[(g + 3) & 3], al = { { lo.q[1], lo.q[2], lo.q[3], hi.q[0] } };
                W[g & 3] = m_sha512msg2(q4_add(m_sha512msg1(W[g & 3], W[(g + 1) & 3].q[0]), al), hi);
            }
            Q4 k = { { K512[4 * g], K512[4 * g + 1], K512[4 * g + 2], K512[4 * g + 3] } }, msg = q4_add(W[g & 3], k);
            s1 = m_sha512rnds2(s1, s0, msg.q[0], msg.q[1]);
            s0 = m_sha512rnds2(s0, s1, msg.q[2], msg.q[3]);
        }
        s0 = q4_add(s0, sv0); s1 = q4_add(s1, sv1);
    }
    st[0] = s0.q[3]; st[1] = s0.q[2]; st[4] = s0.q[1]; st[5] = s0.q[0];
    st[2] = s1.q[3]; st[3] = s1.q[2]; st[6] = s1.q[1]; st[7] = s1.q[0];
}

/* ---- SHA-NI hardware paths (legacy SSE encodings; only SSE2 + SHA are used) ---- */
#define RNDS4(a, e, f) ((f) == 0 ? _mm_sha1rnds4_epu32(a, e, 0) : (f) == 1 ? _mm_sha1rnds4_epu32(a, e, 1) : \
                        (f) == 2 ? _mm_sha1rnds4_epu32(a, e, 2) : _mm_sha1rnds4_epu32(a, e, 3))
T_TARGET("sha") T_NOINLINE static void hw_sha1_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint32_t *st = (uint32_t *)stv;
    __m128i abcd = _mm_set_epi32((int)st[0], (int)st[1], (int)st[2], (int)st[3]), e0 = _mm_set_epi32((int)st[4], 0, 0, 0);
    for (; nb; nb--, p += 64) {
        __m128i W[4], save = abcd, esave = e0, prev = abcd, e;
        for (int i = 0; i < 4; i++)
            W[i] = _mm_set_epi32((int)rd32be(p + 16 * i), (int)rd32be(p + 16 * i + 4), (int)rd32be(p + 16 * i + 8), (int)rd32be(p + 16 * i + 12));
        for (int g = 0; g < 20; g++) {
            __m128i w;
            if (g < 4) w = W[g];
            else { w = _mm_sha1msg2_epu32(_mm_xor_si128(_mm_sha1msg1_epu32(W[g & 3], W[(g + 1) & 3]), W[(g + 2) & 3]), W[(g + 3) & 3]); W[g & 3] = w; }
            e = g == 0 ? _mm_add_epi32(e0, w) : _mm_sha1nexte_epu32(prev, w);
            prev = abcd; abcd = RNDS4(abcd, e, g / 5);
        }
        e0 = _mm_sha1nexte_epu32(prev, esave);
        abcd = _mm_add_epi32(abcd, save);
    }
    V128 a, x; a.v = abcd; x.v = e0;
    st[0] = a.d[3]; st[1] = a.d[2]; st[2] = a.d[1]; st[3] = a.d[0]; st[4] = x.d[3];
}
T_TARGET("sha") T_NOINLINE static void hw_sha256_blocks(void *stv, const uint8_t *p, size_t nb) {
    uint32_t *st = (uint32_t *)stv;
    __m128i s0 = _mm_set_epi32((int)st[0], (int)st[1], (int)st[4], (int)st[5]), s1 = _mm_set_epi32((int)st[2], (int)st[3], (int)st[6], (int)st[7]);
    for (; nb; nb--, p += 64) {
        __m128i W[4], sv0 = s0, sv1 = s1;
        for (int i = 0; i < 4; i++)
            W[i] = _mm_set_epi32((int)rd32be(p + 16 * i + 12), (int)rd32be(p + 16 * i + 8), (int)rd32be(p + 16 * i + 4), (int)rd32be(p + 16 * i));
        for (int g = 0; g < 16; g++) {
            if (g >= 4) {
                __m128i al = _mm_or_si128(_mm_srli_si128(W[(g + 2) & 3], 4), _mm_slli_si128(W[(g + 3) & 3], 12));
                W[g & 3] = _mm_sha256msg2_epu32(_mm_add_epi32(_mm_sha256msg1_epu32(W[g & 3], W[(g + 1) & 3]), al), W[(g + 3) & 3]);
            }
            __m128i msg = _mm_add_epi32(W[g & 3], _mm_loadu_si128((const __m128i *)&K256[4 * g]));
            s1 = _mm_sha256rnds2_epu32(s1, s0, msg);
            s0 = _mm_sha256rnds2_epu32(s0, s1, _mm_shuffle_epi32(msg, 0x0E));
        }
        s0 = _mm_add_epi32(s0, sv0); s1 = _mm_add_epi32(s1, sv1);
    }
    V128 a, b; a.v = s0; b.v = s1;
    st[0] = a.d[3]; st[1] = a.d[2]; st[4] = a.d[1]; st[5] = a.d[0];
    st[2] = b.d[3]; st[3] = b.d[2]; st[6] = b.d[1]; st[7] = b.d[0];
}
enum { SH_RNDS4_0, SH_RNDS4_1, SH_RNDS4_2, SH_RNDS4_3, SH_NEXTE, SH_MSG1, SH_MSG2, SH256_RNDS2, SH256_MSG1, SH256_MSG2 };
static const char *const g_shaop_name[] = { "sha1rnds4/0", "sha1rnds4/1", "sha1rnds4/2", "sha1rnds4/3", "sha1nexte",
                                            "sha1msg1", "sha1msg2", "sha256rnds2", "sha256msg1", "sha256msg2" };
static V128 m_shaop(int op, V128 a, V128 b, V128 k) {
    switch (op) {
    case SH_RNDS4_0: case SH_RNDS4_1: case SH_RNDS4_2: case SH_RNDS4_3: return m_sha1rnds4(a, b, op);
    case SH_NEXTE: return m_sha1nexte(a, b);
    case SH_MSG1: return m_sha1msg1(a, b);
    case SH_MSG2: return m_sha1msg2(a, b);
    case SH256_RNDS2: return m_sha256rnds2(a, b, k);
    case SH256_MSG1: return m_sha256msg1(a, b);
    default: return m_sha256msg2(a, b);
    }
}
T_TARGET("sha") T_NOINLINE static void hw_shaop(int op, const V128 *a, const V128 *b, const V128 *k, V128 *o) {
    switch (op) {
    case SH_RNDS4_0: o->v = _mm_sha1rnds4_epu32(a->v, b->v, 0); break;
    case SH_RNDS4_1: o->v = _mm_sha1rnds4_epu32(a->v, b->v, 1); break;
    case SH_RNDS4_2: o->v = _mm_sha1rnds4_epu32(a->v, b->v, 2); break;
    case SH_RNDS4_3: o->v = _mm_sha1rnds4_epu32(a->v, b->v, 3); break;
    case SH_NEXTE: o->v = _mm_sha1nexte_epu32(a->v, b->v); break;
    case SH_MSG1: o->v = _mm_sha1msg1_epu32(a->v, b->v); break;
    case SH_MSG2: o->v = _mm_sha1msg2_epu32(a->v, b->v); break;
    case SH256_RNDS2: o->v = _mm_sha256rnds2_epu32(a->v, b->v, k->v); break;
    case SH256_MSG1: o->v = _mm_sha256msg1_epu32(a->v, b->v); break;
    default: o->v = _mm_sha256msg2_epu32(a->v, b->v); break;
    }
}
/* explicit-encoding forms: register operands fixed by asm, sha256rnds2 implicit XMM0, aligned m128 sources */
T_NOINLINE static void hw_shaop_mem(int op, const V128 *a, const V128 *b, const V128 *k, V128 *o) {
    switch (op) {
    case SH_RNDS4_2: __asm__ volatile("movdqa %1, %%xmm1\n\tsha1rnds4 $2, %2, %%xmm1\n\tmovdqa %%xmm1, %0" : "=m"(*o) : "m"(*a), "m"(*b) : "xmm1"); break;
    case SH_NEXTE:   __asm__ volatile("movdqa %1, %%xmm1\n\tsha1nexte %2, %%xmm1\n\tmovdqa %%xmm1, %0" : "=m"(*o) : "m"(*a), "m"(*b) : "xmm1"); break;
    case SH_MSG1:    __asm__ volatile("movdqa %1, %%xmm1\n\tsha1msg1 %2, %%xmm1\n\tmovdqa %%xmm1, %0" : "=m"(*o) : "m"(*a), "m"(*b) : "xmm1"); break;
    case SH_MSG2:    __asm__ volatile("movdqa %1, %%xmm1\n\tsha1msg2 %2, %%xmm1\n\tmovdqa %%xmm1, %0" : "=m"(*o) : "m"(*a), "m"(*b) : "xmm1"); break;
    case SH256_RNDS2:__asm__ volatile("movdqa %3, %%xmm0\n\tmovdqa %1, %%xmm2\n\tsha256rnds2 %%xmm0, %2, %%xmm2\n\tmovdqa %%xmm2, %0"
                                      : "=m"(*o) : "m"(*a), "m"(*b), "m"(*k) : "xmm0", "xmm2"); break;
    case SH256_MSG1: __asm__ volatile("movdqa %1, %%xmm1\n\tsha256msg1 %2, %%xmm1\n\tmovdqa %%xmm1, %0" : "=m"(*o) : "m"(*a), "m"(*b) : "xmm1"); break;
    default:         __asm__ volatile("movdqa %1, %%xmm1\n\tsha256msg2 %2, %%xmm1\n\tmovdqa %%xmm1, %0" : "=m"(*o) : "m"(*a), "m"(*b) : "xmm1"); break;
    }
}

/* ---- SHA512 extensions (VEX.256; AVX state) ---- */
enum { S512_RNDS2, S512_MSG1, S512_MSG2 };
T_TARGET("avx2") T_NOINLINE static void hw_sha512op_ymm(int op, const Q4 *a, const Q4 *b, const uint64_t *k2, Q4 *o) {
    __m256i x = _mm256_loadu_si256((const __m256i *)a), y = _mm256_loadu_si256((const __m256i *)b);
    __m128i k = _mm_loadu_si128((const __m128i *)k2);
    if (op == S512_RNDS2) __asm__("vsha512rnds2 %2, %1, %0" : "+x"(x) : "x"(y), "x"(k));
    else if (op == S512_MSG1) __asm__("vsha512msg1 %1, %0" : "+x"(x) : "x"(_mm256_castsi256_si128(y)));
    else __asm__("vsha512msg2 %1, %0" : "+x"(x) : "x"(y));
    _mm256_storeu_si256((__m256i *)o, x);
    _mm256_zeroupper();
}
T_TARGET("avx2") T_NOINLINE static void hw_sha512_blocks_avx(void *stv, const uint8_t *p, size_t nb) {
    uint64_t *st = (uint64_t *)stv;
    __m256i s0 = _mm256_set_epi64x((long long)st[0], (long long)st[1], (long long)st[4], (long long)st[5]);
    __m256i s1 = _mm256_set_epi64x((long long)st[2], (long long)st[3], (long long)st[6], (long long)st[7]);
    for (; nb; nb--, p += 128) {
        __m256i W[4], sv0 = s0, sv1 = s1;
        for (int i = 0; i < 4; i++)
            W[i] = _mm256_set_epi64x((long long)rd64be(p + 32 * i + 24), (long long)rd64be(p + 32 * i + 16), (long long)rd64be(p + 32 * i + 8), (long long)rd64be(p + 32 * i));
        for (int g = 0; g < 20; g++) {
            if (g >= 4) {
                __m256i x = W[g & 3], al = _mm256_permute4x64_epi64(_mm256_blend_epi32(W[(g + 2) & 3], W[(g + 3) & 3], 0x03), 0x39);
                __asm__("vsha512msg1 %1, %0" : "+x"(x) : "x"(_mm256_castsi256_si128(W[(g + 1) & 3])));
                x = _mm256_add_epi64(x, al);
                __asm__("vsha512msg2 %1, %0" : "+x"(x) : "x"(W[(g + 3) & 3]));
                W[g & 3] = x;
            }
            __m256i msg = _mm256_add_epi64(W[g & 3], _mm256_loadu_si256((const __m256i *)&K512[4 * g]));
            __asm__("vsha512rnds2 %1, %2, %0" : "+x"(s1) : "x"(_mm256_castsi256_si128(msg)), "x"(s0));
            __asm__("vsha512rnds2 %1, %2, %0" : "+x"(s0) : "x"(_mm256_extracti128_si256(msg, 1)), "x"(s1));
        }
        s0 = _mm256_add_epi64(s0, sv0); s1 = _mm256_add_epi64(s1, sv1);
    }
    uint64_t a[4], b[4];
    _mm256_storeu_si256((__m256i *)a, s0); _mm256_storeu_si256((__m256i *)b, s1);
    st[0] = a[3]; st[1] = a[2]; st[4] = a[1]; st[5] = a[0];
    st[2] = b[3]; st[3] = b[2]; st[6] = b[1]; st[7] = b[0];
    _mm256_zeroupper();
}

/* ================================================================ GFNI */
static uint8_t m_affine_byte(uint64_t A, uint8_t x, uint8_t b) {
    uint8_t r = 0;
    for (int k = 0; k < 8; k++) {
        uint8_t row = (uint8_t)(A >> (8 * (7 - k))), v = row & x, par = 0;
        while (v) { par ^= 1; v &= (uint8_t)(v - 1); }
        r |= (uint8_t)((par ^ ((b >> k) & 1)) << k);
    }
    return r;
}
static V128 m_gf2p8affine(V128 x, V128 A, uint8_t b, int inv) {
    V128 o;
    for (int i = 0; i < 16; i++) o.b[i] = m_affine_byte(A.q[i >> 3], inv ? gf_inv8(x.b[i]) : x.b[i], b);
    return o;
}
static V128 m_gf2p8mul(V128 a, V128 b) { V128 o; for (int i = 0; i < 16; i++) o.b[i] = gf_mul8(a.b[i], b.b[i]); return o; }
#define GF_IMMS(X) X(0x00) X(0x63) X(0xFF) X(0x5A) X(0x01) X(0x80)
static const uint8_t g_gf_imms[] = { 0x00, 0x63, 0xFF, 0x5A, 0x01, 0x80 };
#define GFA_CASE(i) case i: o->v = inv ? _mm_gf2p8affineinv_epi64_epi8(x->v, A->v, i) : _mm_gf2p8affine_epi64_epi8(x->v, A->v, i); break;
T_TARGET("gfni") T_NOINLINE static void hw_gfni_affine(const V128 *x, const V128 *A, int imm, int inv, V128 *o) {
    switch (imm) { GF_IMMS(GFA_CASE) default: break; }
}
T_TARGET("gfni") T_NOINLINE static void hw_gfni_mul(const V128 *a, const V128 *b, V128 *o) { o->v = _mm_gf2p8mul_epi8(a->v, b->v); }

/* ================================================================ VAES / VPCLMULQDQ (VEX.256, EVEX.512) */
T_TARGET("aes,avx") T_NOINLINE static void hw_aesop_vex128_avx(int op, const uint8_t *a, const uint8_t *k_unaligned, uint8_t *o) {
    __m128i x = _mm_loadu_si128((const __m128i *)a), r;
    /* VEX forms take an unaligned m128 operand without faulting */
    switch (op) {
    case OP_ENC:     __asm__("vaesenc %2, %1, %0" : "=x"(r) : "x"(x), "m"(*(const uint8_t (*)[16])k_unaligned)); break;
    case OP_ENCLAST: __asm__("vaesenclast %2, %1, %0" : "=x"(r) : "x"(x), "m"(*(const uint8_t (*)[16])k_unaligned)); break;
    case OP_DEC:     __asm__("vaesdec %2, %1, %0" : "=x"(r) : "x"(x), "m"(*(const uint8_t (*)[16])k_unaligned)); break;
    case OP_DECLAST: __asm__("vaesdeclast %2, %1, %0" : "=x"(r) : "x"(x), "m"(*(const uint8_t (*)[16])k_unaligned)); break;
    default:         __asm__("vaesimc %1, %0" : "=x"(r) : "m"(*(const uint8_t (*)[16])k_unaligned)); break;
    }
    _mm_storeu_si128((__m128i *)o, r);
}
T_TARGET("pclmul,avx") T_NOINLINE static void hw_pclmul_vex128_avx(const uint8_t *a, const uint8_t *b_unaligned, int sel, uint8_t *o) {
    __m128i x = _mm_loadu_si128((const __m128i *)a), r;
    const uint8_t (*m)[16] = (const uint8_t (*)[16])b_unaligned;
    switch (sel) {
    case 0: __asm__("vpclmulqdq $0x00, %2, %1, %0" : "=x"(r) : "x"(x), "m"(*m)); break;
    case 1: __asm__("vpclmulqdq $0x01, %2, %1, %0" : "=x"(r) : "x"(x), "m"(*m)); break;
    case 2: __asm__("vpclmulqdq $0x10, %2, %1, %0" : "=x"(r) : "x"(x), "m"(*m)); break;
    default: __asm__("vpclmulqdq $0x11, %2, %1, %0" : "=x"(r) : "x"(x), "m"(*m)); break;
    }
    _mm_storeu_si128((__m128i *)o, r);
}
T_TARGET("avx2,vaes") T_NOINLINE static void hw_vaes_ymm(int op, const uint8_t *a, const uint8_t *k, uint8_t *o) {
    __m256i x = _mm256_loadu_si256((const __m256i *)a), y = _mm256_loadu_si256((const __m256i *)k), r;
    switch (op) {
    case OP_ENC: r = _mm256_aesenc_epi128(x, y); break;
    case OP_ENCLAST: r = _mm256_aesenclast_epi128(x, y); break;
    case OP_DEC: r = _mm256_aesdec_epi128(x, y); break;
    default: r = _mm256_aesdeclast_epi128(x, y); break;
    }
    _mm256_storeu_si256((__m256i *)o, r);
    _mm256_zeroupper();
}
#define VCLM_SEL(f, ...) switch (sel) { case 0: r = f(__VA_ARGS__, 0x00); break; case 1: r = f(__VA_ARGS__, 0x01); break; \
                                         case 2: r = f(__VA_ARGS__, 0x10); break; default: r = f(__VA_ARGS__, 0x11); break; }
static const int g_sel_imm[4] = { 0x00, 0x01, 0x10, 0x11 };
T_TARGET("avx2,vpclmulqdq") T_NOINLINE static void hw_vpclmul_ymm(const uint8_t *a, const uint8_t *b, int sel, uint8_t *o) {
    __m256i x = _mm256_loadu_si256((const __m256i *)a), y = _mm256_loadu_si256((const __m256i *)b), r;
    VCLM_SEL(_mm256_clmulepi64_epi128, x, y)
    _mm256_storeu_si256((__m256i *)o, r);
    _mm256_zeroupper();
}
/* VAES CTR with two blocks per YMM register (counter blocks built in memory) */
T_TARGET("avx2,vaes") T_NOINLINE static void hw_vaes_ctr_ymm(const aes_sched *ks, uint8_t ctr[16], const uint8_t *in, uint8_t *out, size_t nblocks) {
    __m256i rk[15];
    for (int r = 0; r <= ks->nr; r++) rk[r] = _mm256_broadcastsi128_si256(ks->rk[r].v);
    for (; nblocks >= 2; nblocks -= 2, in += 32, out += 32) {
        T_ALIGN(32) uint8_t c[32];
        memcpy(c, ctr, 16); inc128be(ctr); memcpy(c + 16, ctr, 16); inc128be(ctr);
        __m256i x = _mm256_xor_si256(_mm256_load_si256((const __m256i *)c), rk[0]);
        for (int r = 1; r < ks->nr; r++) x = _mm256_aesenc_epi128(x, rk[r]);
        x = _mm256_aesenclast_epi128(x, rk[ks->nr]);
        _mm256_storeu_si256((__m256i *)out, _mm256_xor_si256(x, _mm256_loadu_si256((const __m256i *)in)));
    }
    _mm256_zeroupper();
    if (nblocks) hw_ctr(ks, ctr, in, out, 16, 0, 1);
}
T_TARGET("avx512f,vaes") T_NOINLINE static void hw_vaes_zmm512(int op, const uint8_t *a, const uint8_t *k, uint8_t *o) {
    __m512i x = _mm512_loadu_si512(a), y = _mm512_loadu_si512(k), r;
    switch (op) {
    case OP_ENC: r = _mm512_aesenc_epi128(x, y); break;
    case OP_ENCLAST: r = _mm512_aesenclast_epi128(x, y); break;
    case OP_DEC: r = _mm512_aesdec_epi128(x, y); break;
    default: r = _mm512_aesdeclast_epi128(x, y); break;
    }
    _mm512_storeu_si512(o, r);
    _mm256_zeroupper();
}
T_TARGET("avx512f,vpclmulqdq") T_NOINLINE static void hw_vpclmul_zmm512(const uint8_t *a, const uint8_t *b, int sel, uint8_t *o) {
    __m512i x = _mm512_loadu_si512(a), y = _mm512_loadu_si512(b), r;
    VCLM_SEL(_mm512_clmulepi64_epi128, x, y)
    _mm512_storeu_si512(o, r);
    _mm256_zeroupper();
}
#ifdef T_X64
/* EVEX.256 encodings forced by using YMM16-YMM19 (AVX512VL + VAES / VPCLMULQDQ) */
T_TARGET("avx512f,avx512vl") T_NOINLINE static void hw_vaes_evex256_ymm16(const uint8_t *a, const uint8_t *k, uint8_t *o) {
    __asm__ volatile("vmovdqu64 %1, %%ymm16\n\tvmovdqu64 %2, %%ymm17\n\tvaesenc %%ymm17, %%ymm16, %%ymm18\n\t"
                     "vaesdeclast %%ymm17, %%ymm18, %%ymm19\n\tvmovdqu64 %%ymm18, %0\n\tvmovdqu64 %%ymm19, 32%0\n\tvzeroupper"
                     : "=m"(*(uint8_t (*)[64])o) : "m"(*(const uint8_t (*)[32])a), "m"(*(const uint8_t (*)[32])k)
                     : "xmm16", "xmm17", "xmm18", "xmm19");
}
T_TARGET("avx512f,avx512vl") T_NOINLINE static void hw_vpclmul_evex256_ymm16(const uint8_t *a, const uint8_t *b, uint8_t *o) {
    __asm__ volatile("vmovdqu64 %1, %%ymm16\n\tvmovdqu64 %2, %%ymm17\n\tvpclmulqdq $0x10, %%ymm17, %%ymm16, %%ymm18\n\t"
                     "vmovdqu64 %%ymm18, %0\n\tvzeroupper"
                     : "=m"(*(uint8_t (*)[32])o) : "m"(*(const uint8_t (*)[32])a), "m"(*(const uint8_t (*)[32])b)
                     : "xmm16", "xmm17", "xmm18");
}
#endif

/* ================================================================ known-answer data */
/* FIPS-197 */
static const char *const FIPS_PT = "00112233445566778899aabbccddeeff";
static const struct { int bits; const char *key, *ct, *last_rk, *rk1; } FIPS_C[3] = {
    { 128, "000102030405060708090a0b0c0d0e0f", "69c4e0d86a7b0430d8cdb78070b4c55a", "13111d7fe3944a17f307a78b4d2b30c5", "d6aa74fdd2af72fadaa678f1d6ab76fe" },
    { 192, "000102030405060708090a0b0c0d0e0f1011121314151617", "dda97ca4864cdfe06eaf70a0ec0d7191", "a4970a331a78dc09c418c271e3a41d5d", "10111213141516175846f2f95c43f4fe" },
    { 256, "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "8ea2b7ca516745bfeafc49904b496089", "24fc79ccbf0979e9371ac23c6d68de36", "101112131415161718191a1b1c1d1e1f" },
};
/* FIPS-197 Appendix A key-expansion examples: key, w[Nk..Nk+3] (first derived words), last round key */
static const struct { int bits; const char *key, *first, *last; } FIPS_A[3] = {
    { 128, "2b7e151628aed2a6abf7158809cf4f3c", "a0fafe1788542cb123a339392a6c7605", "d014f9a8c9ee2589e13f0cc8b6630ca6" },
    { 192, "8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b", "fe0c91f72402f5a5ec12068e6c827f6b", "e98ba06f448c773c8ecc720401002202" },
    { 256, "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", "9ba354118e6925afa51a8b5f2067fcde", "fe4890d1e6188d0b046df344706c631e" },
};
/* NIST SP 800-38A F.1/F.2/F.5 */
static const char *const SP_PT = "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
                                 "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710";
static const char *const SP_KEY[3] = { "2b7e151628aed2a6abf7158809cf4f3c", "8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b",
                                       "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4" };
static const char *const SP_ECB[3] = {
    "3ad77bb40d7a3660a89ecaf32466ef97f5d3d58503b9699de785895a96fdbaaf43b1cd7f598ece23881b00e3ed0306887b0c785e27e8ad3f8223207104725dd4",
    "bd334f1d6e45f25ff712a214571fa5cc974104846d0ad3ad7734ecb3ecee4eefef7afd2270e2e60adce0ba2face6444e9a4b41ba738d6c72fb16691603c18e0e",
    "f3eed1bdb5d2a03c064b5a7e3db181f8591ccb10d410ed26dc5ba74a31362870b6ed21b99ca6f4f9f153e7b1beafed1d23304b7a39f9f3ff067d8d8f9e24ecc7" };
static const char *const SP_CBC_IV = "000102030405060708090a0b0c0d0e0f";
static const char *const SP_CBC[3] = {
    "7649abac8119b246cee98e9b12e9197d5086cb9b507219ee95db113a917678b273bed6b8e3c1743b7116e69e222295163ff1caa1681fac09120eca307586e1a7",
    "4f021db243bc633d7178183a9fa071e8b4d9ada9ad7dedf4e5e738763f69145a571b242012fb7ae07fa9baac3df102e008b0e27988598881d920a9e64f5615cd",
    "f58c4c04d6e5f1ba779eabfb5f7bfbd69cfc4e967edb808d679f777bc6702c7d39f23369a9d9bacfa530e26304231461b2eb05e2c39be9fcda6c19078c6a9d1b" };
static const char *const SP_CTR_IV = "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";
static const char *const SP_CTR[3] = {
    "874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee",
    "1abc932417521ca24f2b0459fe7e6e0b090339ec0aa6faefd5ccc2c6f4ce8e941e36b26bd1ebc670d1bd1d665620abf74f78a7f6d29809585a97daec58c6b050",
    "601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c52b0930daa23de94ce87017ba2d84988ddfc9c58db67aada613c2dd08457941a6" };
/* GCM specification test cases (McGrew/Viega), also the NIST CAVS examples */
typedef struct { const char *name; int bits; const char *key, *iv, *pt, *aad, *ct, *tag; } gcm_kat;
#define GCM_P64 "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255"
#define GCM_P60 "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"
#define GCM_A20 "feedfacedeadbeeffeedfacedeadbeefabaddad2"
#define GCM_K   "feffe9928665731c6d6a8f9467308308"
static const gcm_kat GCM_KATS[] = {
    { "tc1",  128, "00000000000000000000000000000000", "000000000000000000000000", "", "", "", "58e2fccefa7e3061367f1d57a4e7455a" },
    { "tc2",  128, "00000000000000000000000000000000", "000000000000000000000000", "00000000000000000000000000000000", "",
      "0388dace60b6a392f328c2b971b2fe78", "ab6e47d42cec13bdf53a67b21257bddf" },
    { "tc3",  128, GCM_K, "cafebabefacedbaddecaf888", GCM_P64, "",
      "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985",
      "4d5c2af327cd64a62cf35abd2ba6fab4" },
    { "tc4",  128, GCM_K, "cafebabefacedbaddecaf888", GCM_P60, GCM_A20,
      "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091",
      "5bc94fbc3221a5db94fae95ae7121a47" },
    { "tc13", 256, "0000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000", "", "", "",
      "530f8afbc74536b9a963b4f1c4cb738b" },
    { "tc14", 256, "0000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000",
      "00000000000000000000000000000000", "", "cea7403d4d606b6e074ec5d3baf39d18", "d0d1c8a799996bf0265b98b5d48ab919" },
    { "tc16", 256, GCM_K GCM_K, "cafebabefacedbaddecaf888", GCM_P60, GCM_A20,
      "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",
      "76fc6ece0f4e1768cddf8853bb2d551b" },
};
/* SHA known answers (FIPS 180-4 examples / NIST CSRC example values) */
static const char *const SHA_MSG56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
static const char *const SHA_MSG112 = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
typedef struct { int alg; int msg; const char *hex; } sha_kat;   /* msg: 0 "", 1 "abc", 2 56-byte, 3 112-byte, 4 million 'a' */
static const sha_kat SHA_KATS[] = {
    { ALG_SHA1, 0, "da39a3ee5e6b4b0d3255bfef95601890afd80709" },
    { ALG_SHA1, 1, "a9993e364706816aba3e25717850c26c9cd0d89d" },
    { ALG_SHA1, 2, "84983e441c3bd26ebaae4aa1f95129e5e54670f1" },
    { ALG_SHA1, 3, "a49b2446a02c645bf419f995b67091253a04a259" },
    { ALG_SHA1, 4, "34aa973cd4c4daa4f61eeb2bdbad27316534016f" },
    { ALG_SHA256, 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
    { ALG_SHA256, 1, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
    { ALG_SHA256, 2, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    { ALG_SHA256, 3, "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" },
    { ALG_SHA256, 4, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" },
    { ALG_SHA512, 0, "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e" },
    { ALG_SHA512, 1, "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f" },
    { ALG_SHA512, 2, "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445" },
    { ALG_SHA512, 3, "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909" },
    { ALG_SHA512, 4, "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b" },
};
static const char *const g_alg_name[] = { "SHA-1", "SHA-256", "SHA-512" };
static const char *const g_msg_name[] = { "\"\"", "\"abc\"", "448-bit", "896-bit", "1M x 'a'" };
static uint8_t *g_million;   /* 1,000,000 'a' (allocated once) */
static const uint8_t *sha_msg(int m, size_t *len) {
    switch (m) {
    case 0: *len = 0; return (const uint8_t *)"";
    case 1: *len = 3; return (const uint8_t *)"abc";
    case 2: *len = 56; return (const uint8_t *)SHA_MSG56;
    case 3: *len = 112; return (const uint8_t *)SHA_MSG112;
    default:
        if (!g_million) { g_million = (uint8_t *)malloc(1000000); if (g_million) memset(g_million, 'a', 1000000); }
        *len = g_million ? 1000000 : 0;
        return g_million;
    }
}
/* run the KAT table for one algorithm through a compression function; returns mismatches */
static void sha_kat_run(int alg, compress_fn fn, const char *impl) {
    for (size_t i = 0; i < T_COUNTOF(SHA_KATS); i++) {
        if (SHA_KATS[i].alg != alg) continue;
        size_t len; const uint8_t *m = sha_msg(SHA_KATS[i].msg, &len);
        uint8_t want[64], got[64];
        REQUIREF(m != NULL, "out of memory for the million-'a' message");
        size_t n = unhex(SHA_KATS[i].hex, want, sizeof want);
        size_t dn = md_digest(alg, fn, m, len, got);
        CHECK_EQ(dn, n);
        CHECK_BLK(got, want, n, "%s %s(%s)", impl, g_alg_name[alg], g_msg_name[SHA_KATS[i].msg]);
    }
}

/* ================================================================================================
 *                                        TEST CASES
 * ================================================================================================ */

/* ---------------------------------------------------------------- inventory and references */
TEST(crypto_feature_inventory) {
    static const t_feature fs[] = { F_AES, F_PCLMULQDQ, F_SHA, F_GFNI, F_VAES, F_VPCLMULQDQ, F_SHA512, F_SM3, F_SM4, F_KL, F_SSE42, F_AVX, F_AVX512F, F_AVX512VL };
    char buf[512]; size_t o = 0;
    for (size_t i = 0; i < T_COUNTOF(fs); i++) {
        o += (size_t)snprintf(buf + o, sizeof buf - o, "%s%s=%s", i ? " " : "", cpu_feature_name(fs[i]),
                              cpu_has(fs[i]) ? "usable" : cpu_hw(fs[i]) ? "cpuid-only" : "absent");
        if (o >= sizeof buf) break;
    }
    t_info("crypto_features", "%s", buf);
    /* the legacy-SSE crypto instructions need no XCR0 state: usable must equal advertised */
    CHECK_EQ(cpu_has(F_AES), cpu_hw(F_AES));
    CHECK_EQ(cpu_has(F_PCLMULQDQ), cpu_hw(F_PCLMULQDQ));
    CHECK_EQ(cpu_has(F_SHA), cpu_hw(F_SHA));
    CHECK_EQ(cpu_has(F_GFNI), cpu_hw(F_GFNI));
    if (cpu_hw(F_VAES) && !cpu_hw(F_AES)) t_note("VAES advertised without AES (unusual CPUID combination)");
    if (cpu_hw(F_VAES) && !cpu_has(F_VAES)) t_note("VAES advertised but YMM state not enabled by the OS (XCR0=0x%llx)", (unsigned long long)cpu()->xcr0);
}

TEST(aes_sw_reference_fips197) {
    crypto_init();
    /* S-box spot values from FIPS-197 Figure 7 / 5.1.1 example */
    CHECK_EQ(g_sbox[0x00], 0x63); CHECK_EQ(g_sbox[0x01], 0x7c); CHECK_EQ(g_sbox[0x53], 0xed);
    CHECK_EQ(g_sbox[0xff], 0x16); CHECK_EQ(g_isbox[0x63], 0x00); CHECK_EQ(g_isbox[0x16], 0xff);
    CHECK_EQ(gf_mul8(0x57, 0x83), 0xc1);   /* FIPS-197 4.2 example */
    CHECK_EQ(gf_mul8(0x57, 0x13), 0xfe);
    for (int i = 0; i < 3; i++) {
        uint8_t key[32], want[16], pt[16], rk[16]; aes_sched ks;
        size_t kl = unhex(FIPS_A[i].key, key, sizeof key);
        CHECK_EQ(kl * 8, (size_t)FIPS_A[i].bits);
        sw_expand(key, FIPS_A[i].bits, &ks);
        unhex(FIPS_A[i].first, rk, 16);
        CHECK_BLK((uint8_t *)ks.rk + kl, rk, 16, "sw AES-%d w[Nk..Nk+3]", FIPS_A[i].bits);
        unhex(FIPS_A[i].last, rk, 16);
        CHECK_BLK(ks.rk[ks.nr].b, rk, 16, "sw AES-%d last round key", FIPS_A[i].bits);
        unhex(FIPS_C[i].key, key, sizeof key); unhex(FIPS_PT, pt, 16); unhex(FIPS_C[i].ct, want, 16);
        sw_expand(key, FIPS_C[i].bits, &ks);
        V128 c = sw_encrypt(&ks, ld128(pt)), p = sw_decrypt(&ks, c);
        CHECK_BLK(c.b, want, 16, "sw AES-%d FIPS-197 C encrypt", FIPS_C[i].bits);
        CHECK_BLK(p.b, pt, 16, "sw AES-%d FIPS-197 C decrypt", FIPS_C[i].bits);
    }
    /* FIPS-197 Appendix B */
    uint8_t key[16], pt[16], want[16]; aes_sched ks;
    unhex("2b7e151628aed2a6abf7158809cf4f3c", key, 16); unhex("3243f6a8885a308d313198a2e0370734", pt, 16);
    unhex("3925841d02dc09fbdc118597196a0b32", want, 16);
    sw_expand(key, 128, &ks);
    V128 c = sw_encrypt(&ks, ld128(pt));
    CHECK_BLK(c.b, want, 16, "sw AES-128 FIPS-197 Appendix B");
}

TEST(aes_sw_reference_sp800_38a_gcm) {
    crypto_init();
    uint8_t pt[64], want[64], got[64], key[32], iv[16];
    unhex(SP_PT, pt, 64);
    for (int k = 0; k < 3; k++) {
        aes_sched ks; int bits = 128 + 64 * k;
        unhex(SP_KEY[k], key, sizeof key); sw_expand(key, bits, &ks);
        unhex(SP_ECB[k], want, 64); sw_ecb(&ks, pt, got, 4, 0); CHECK_BLK(got, want, 64, "sw ECB-AES%d", bits);
        unhex(SP_CBC_IV, iv, 16); unhex(SP_CBC[k], want, 64); sw_cbc(&ks, iv, pt, got, 4, 0); CHECK_BLK(got, want, 64, "sw CBC-AES%d", bits);
        unhex(SP_CTR_IV, iv, 16); unhex(SP_CTR[k], want, 64); sw_ctr(&ks, iv, pt, got, 64, 0); CHECK_BLK(got, want, 64, "sw CTR-AES%d", bits);
    }
    for (size_t i = 0; i < T_COUNTOF(GCM_KATS); i++) {
        const gcm_kat *g = &GCM_KATS[i];
        uint8_t gk[32], giv[12], gp[64], ga[32], gc[64], gt[16], oc[64], ot[16];
        unhex(g->key, gk, 32); unhex(g->iv, giv, 12);
        size_t pl = unhex(g->pt, gp, 64), al = unhex(g->aad, ga, 32);
        unhex(g->ct, gc, 64); unhex(g->tag, gt, 16);
        aes_gcm(0, gk, g->bits, giv, ga, al, gp, pl, oc, ot, 0);
        CHECK_BLK(oc, gc, pl, "sw GCM %s ciphertext", g->name);
        CHECK_BLK(ot, gt, 16, "sw GCM %s tag", g->name);
    }
    /* GHASH of GCM test case 2 (H = E(0^128), C = 0388...fe78): f38cbb1a d69223dc c3457ae5 b6b0f885 */
    uint8_t H[16], C[16], S[16], wantS[16];
    unhex("66e94bd4ef8a2c3b884cfa59ca342b2e", H, 16); unhex("0388dace60b6a392f328c2b971b2fe78", C, 16);
    unhex("f38cbb1ad69223dcc3457ae5b6b0f885", wantS, 16);
    ghash(sw_gf128_mul, H, NULL, 0, C, 16, S);
    CHECK_BLK(S, wantS, 16, "sw GHASH(TC2)");
}

TEST(sha_sw_reference_fips180) {
    sha_kat_run(ALG_SHA1, sw_sha1_blocks, "sw");
    sha_kat_run(ALG_SHA256, sw_sha256_blocks, "sw");
    sha_kat_run(ALG_SHA512, sw_sha512_blocks, "sw");
}

/* The instruction models composed in the hardware dataflow must give the FIPS digests. This validates
 * the models (and therefore the per-instruction checks) independently of any CPU. */
TEST(sha_instruction_models_compose_to_fips180) {
    sha_kat_run(ALG_SHA1, model_sha1_blocks, "model");
    sha_kat_run(ALG_SHA256, model_sha256_blocks, "model");
    sha_kat_run(ALG_SHA512, model_sha512_blocks, "model");
}

TEST(crc_sw_reference_check_values) {
    crypto_init();
    const uint8_t *s = (const uint8_t *)"123456789";
    CHECK_EQ(crc_sw(g_crc32c_tab, s, 9), 0xE3069283u);
    CHECK_EQ(crc_sw(g_crc32_tab, s, 9), 0xCBF43926u);
    /* RFC 3720 B.4 CRC32C examples */
    uint8_t b[32];
    memset(b, 0, 32);    CHECK_EQ(crc_sw(g_crc32c_tab, b, 32), 0x8A9136AAu);
    memset(b, 0xFF, 32); CHECK_EQ(crc_sw(g_crc32c_tab, b, 32), 0x62A8AB43u);
    for (int i = 0; i < 32; i++) b[i] = (uint8_t)i;
    CHECK_EQ(crc_sw(g_crc32c_tab, b, 32), 0x46DD794Eu);
    for (int i = 0; i < 32; i++) b[i] = (uint8_t)(31 - i);
    CHECK_EQ(crc_sw(g_crc32c_tab, b, 32), 0x113FDB5Cu);
}

TEST(clmul_sw_reference_identities) {
    uint64_t lo, hi;
    sw_clmul64(3, 3, &lo, &hi); CHECK_EQ(lo, 5); CHECK_EQ(hi, 0);
    sw_clmul64(0x87, 0x87, &lo, &hi); CHECK_EQ(lo, 0x4015); CHECK_EQ(hi, 0);
    sw_clmul64(~0ull, ~0ull, &lo, &hi); CHECK_EQ(lo, 0x5555555555555555ull); CHECK_EQ(hi, 0x5555555555555555ull);
    sw_clmul64(1ull << 63, 1ull << 63, &lo, &hi); CHECK_EQ(lo, 0); CHECK_EQ(hi, 1ull << 62);
    uint8_t X[16], Y[16], Z[16], one[16] = { 0x80 };
    t_rng r; t_rng_seed(&r, 11);
    t_rng_fill(&r, X, 16); t_rng_fill(&r, Y, 16);
    sw_gf128_mul(X, one, Z); CHECK_BLK(Z, X, 16, "GF(2^128): X * 1 == X");
    uint8_t a[16], b[16];
    sw_gf128_mul(X, Y, a); sw_gf128_mul(Y, X, b); CHECK_BLK(a, b, 16, "GF(2^128) commutative");
}

/* ---------------------------------------------------------------- AES-NI single instructions */
/* Intel AES-NI white paper (Gueron, rev 3.0) Figure "AES instructions examples" */
static void aes_wp_kat(int op, const char *res) {
    NEED(F_AES); crypto_init();
    V128 st = be128("7b5b54657374566563746f725d53475d"), rk = be128("48692853686179295b477565726f6e5d"), want = be128(res), got, mod;
    if (op == OP_IMC) st = rk;
    hw_aesop(op, &st, &rk, &got);
    mod = m_aesop(op, st, rk);
    CHECK_BLK(mod.b, want.b, 16, "model %s vs white-paper vector", g_aesop_name[op]);
    CHECK_BLK(got.b, want.b, 16, "%s white-paper vector (state %s)", g_aesop_name[op], hx(st.b, 16));
}
TEST(aesni_aesenc_known_answer)     { aes_wp_kat(OP_ENC,     "a8311c2f9fdba3c58b104b58ded7e595"); }
TEST(aesni_aesenclast_known_answer) { aes_wp_kat(OP_ENCLAST, "c7fb881e938c5964177ec42553fdc611"); }
TEST(aesni_aesdec_known_answer)     { aes_wp_kat(OP_DEC,     "138ac342faea2787b58eb95eb730392a"); }
TEST(aesni_aesdeclast_known_answer) { aes_wp_kat(OP_DECLAST, "c5a391ef6b317f95d410637b72a593d0"); }
TEST(aesni_aesimc_known_answer) {
    /* AESIMC = InvMixColumns: verify on the FIPS-197 example columns (MixColumns of db135345 -> 8e4da1bc) */
    NEED(F_AES); crypto_init();
    V128 mixed = be128("00000000000000000000000000000000"), plain, got;
    uint8_t in[16] = { 0x8e, 0x4d, 0xa1, 0xbc, 0x9f, 0xdc, 0x58, 0x9d, 0x01, 0x01, 0x01, 0x01, 0xd5, 0xd5, 0xd7, 0xd6 };
    uint8_t out[16] = { 0xdb, 0x13, 0x53, 0x45, 0xf2, 0x0a, 0x22, 0x5c, 0x01, 0x01, 0x01, 0x01, 0xd4, 0xd4, 0xd4, 0xd5 };
    memcpy(mixed.b, in, 16);
    hw_aesop(OP_IMC, &mixed, &mixed, &got);
    memcpy(plain.b, out, 16);
    CHECK_BLK(got.b, plain.b, 16, "aesimc of MixColumns test columns");
    V128 back = sw_mixcolumns(plain);
    CHECK_BLK(back.b, in, 16, "sw MixColumns test columns");
}
TEST(aesni_aeskeygenassist_known_answer) {
    NEED(F_AES); crypto_init();
    V128 src = be128("3c4fcf098815f7aba6d2ae2816157e2b"), want = be128("01eb848beb848a013424b5e524b5e434"), got;
    hw_kga(&src, 0x01, &got);
    V128 mod = m_aeskeygenassist(src, 1);
    CHECK_BLK(mod.b, want.b, 16, "model aeskeygenassist white-paper vector");
    CHECK_BLK(got.b, want.b, 16, "aeskeygenassist imm=1 white-paper vector");
}
static void aes_random_vs_model(int op, uint64_t seed) {
    NEED(F_AES); crypto_init();
    t_rng r; t_rng_seed(&r, seed);
    vbegin();
    for (int i = 0; i < 3000; i++) {
        V128 a, k, got, want;
        gen128(&r, &a); gen128(&r, &k);
        hw_aesop(op, &a, &k, &got);
        want = m_aesop(op, a, k);
        VCHECK(!memcmp(got.b, want.b, 16), "%s(%s, %s) = %s, model %s", g_aesop_name[op], hx(a.b, 16), hx(k.b, 16), hx(got.b, 16), hx(want.b, 16));
    }
    VEND(g_aesop_name[op]);
}
TEST(aesni_aesenc_random_vs_model)     { aes_random_vs_model(OP_ENC, 101); }
TEST(aesni_aesenclast_random_vs_model) { aes_random_vs_model(OP_ENCLAST, 102); }
TEST(aesni_aesdec_random_vs_model)     { aes_random_vs_model(OP_DEC, 103); }
TEST(aesni_aesdeclast_random_vs_model) { aes_random_vs_model(OP_DECLAST, 104); }
TEST(aesni_aesimc_random_vs_model)     { aes_random_vs_model(OP_IMC, 105); }

TEST(aesni_aeskeygenassist_all_imm8_vs_model) {
    NEED(F_AES); crypto_init();
    t_rng r; t_rng_seed(&r, 106);
    vbegin();
    for (int imm = 0; imm < 256; imm++) for (int i = 0; i < 12; i++) {
        V128 a, got, want;
        gen128(&r, &a);
        hw_kga(&a, imm, &got);
        want = m_aeskeygenassist(a, (uint8_t)imm);
        VCHECK(!memcmp(got.b, want.b, 16), "aeskeygenassist(%s, 0x%02x) = %s, model %s", hx(a.b, 16), imm, hx(got.b, 16), hx(want.b, 16));
    }
    VEND("aeskeygenassist imm8 sweep");
}

TEST(aesni_round_identities) {
    NEED(F_AES); crypto_init();
    t_rng r; t_rng_seed(&r, 107);
    vbegin();
    V128 zero; memset(&zero, 0, sizeof zero);
    for (int i = 0; i < 500; i++) {
        V128 x, k, y, z, w;
        gen128(&r, &x); gen128(&r, &k);
        /* AESDECLAST(AESENCLAST(x, 0), 0) == x (InvSubBytes/InvShiftRows undo SubBytes/ShiftRows) */
        hw_aesop(OP_ENCLAST, &x, &zero, &y); hw_aesop(OP_DECLAST, &y, &zero, &z);
        VCHECK(!memcmp(z.b, x.b, 16), "declast(enclast(%s,0),0) = %s", hx(x.b, 16), hx(z.b, 16));
        /* round key is a plain final XOR */
        hw_aesop(OP_ENC, &x, &zero, &y); hw_aesop(OP_ENC, &x, &k, &z); w = vxor(y, k);
        VCHECK(!memcmp(z.b, w.b, 16), "aesenc(x,k) != aesenc(x,0)^k for x=%s k=%s", hx(x.b, 16), hx(k.b, 16));
        hw_aesop(OP_DEC, &x, &zero, &y); hw_aesop(OP_DEC, &x, &k, &z); w = vxor(y, k);
        VCHECK(!memcmp(z.b, w.b, 16), "aesdec(x,k) != aesdec(x,0)^k for x=%s", hx(x.b, 16));
        /* AESIMC undoes the MixColumns step: InvMixColumns(aesenc(x,0)) == aesenclast(x,0) */
        hw_aesop(OP_ENC, &x, &zero, &y); hw_aesop(OP_IMC, &y, &y, &z); hw_aesop(OP_ENCLAST, &x, &zero, &w);
        VCHECK(!memcmp(z.b, w.b, 16), "aesimc(aesenc(x,0)) != aesenclast(x,0) for x=%s", hx(x.b, 16));
    }
    VEND("AES round identities");
}

TEST(aesni_m128_memory_operand_forms) {
    NEED(F_AES); crypto_init();
    t_rng r; t_rng_seed(&r, 108);
    vbegin();
    for (int op = OP_ENC; op <= OP_IMC; op++) for (int i = 0; i < 200; i++) {
        V128 a, k, g1, g2;
        gen128(&r, &a); gen128(&r, &k);
        hw_aesop(op, &a, &k, &g1); hw_aesop_mem(op, &a, &k, &g2);
        VCHECK(!memcmp(g1.b, g2.b, 16), "%s m128 form %s != register form %s", g_aesop_name[op], hx(g2.b, 16), hx(g1.b, 16));
    }
    for (int i = 0; i < 200; i++) {
        V128 a, g1, g2; gen128(&r, &a);
        hw_kga(&a, 0x1b, &g1); hw_kga1_mem(&a, &g2);
        VCHECK(!memcmp(g1.b, g2.b, 16), "aeskeygenassist m128 form %s != register form %s", hx(g2.b, 16), hx(g1.b, 16));
    }
    VEND("AES m128 forms");
}

/* ---------------------------------------------------------------- AES-NI key schedules and block cipher */
static void aes_key_expansion_case(int idx) {
    NEED(F_AES); crypto_init();
    uint8_t key[32], first[16], last[16];
    int bits = FIPS_A[idx].bits, nk = bits / 32;
    unhex(FIPS_A[idx].key, key, 32); unhex(FIPS_A[idx].first, first, 16); unhex(FIPS_A[idx].last, last, 16);
    aes_sched sw, hw1, hw2;
    sw_expand(key, bits, &sw);
    hw_expand(key, bits, &hw1);
    hw_expand_wordwise(key, bits, &hw2);
    CHECK_EQ(hw1.nr, sw.nr); CHECK_EQ(hw2.nr, sw.nr);
    for (int r = 0; r <= sw.nr; r++) {
        CHECK_BLK(hw1.rk[r].b, sw.rk[r].b, 16, "AES-%d %s round key %d", bits, bits == 192 ? "word-wise" : "white-paper", r);
        CHECK_BLK(hw2.rk[r].b, sw.rk[r].b, 16, "AES-%d word-wise round key %d", bits, r);
    }
    CHECK_BLK((uint8_t *)hw1.rk + 4 * nk, first, 16, "AES-%d FIPS-197 A w[Nk..Nk+3]", bits);
    CHECK_BLK(hw1.rk[hw1.nr].b, last, 16, "AES-%d FIPS-197 A last round key", bits);
}
TEST(aesni_key_expansion_128_fips197) { aes_key_expansion_case(0); }
TEST(aesni_key_expansion_192_fips197) { aes_key_expansion_case(1); }
TEST(aesni_key_expansion_256_fips197) { aes_key_expansion_case(2); }

static void aes_fips_c_case(int idx) {
    NEED(F_AES); crypto_init();
    uint8_t key[32], pt[16], want[16], ct[16], back[16], rk1[16], rkl[16];
    int bits = FIPS_C[idx].bits;
    unhex(FIPS_C[idx].key, key, 32); unhex(FIPS_PT, pt, 16); unhex(FIPS_C[idx].ct, want, 16);
    unhex(FIPS_C[idx].rk1, rk1, 16); unhex(FIPS_C[idx].last_rk, rkl, 16);
    aes_sched ks, dk;
    hw_expand(key, bits, &ks);
    CHECK_BLK(ks.rk[1].b, rk1, 16, "AES-%d FIPS-197 C round[1].k_sch", bits);
    CHECK_BLK(ks.rk[ks.nr].b, rkl, 16, "AES-%d FIPS-197 C round[%d].k_sch", bits, ks.nr);
    hw_ecb_enc(&ks, pt, ct, 1);
    CHECK_BLK(ct, want, 16, "AES-%d FIPS-197 C encrypt", bits);
    hw_dec_sched(&ks, &dk);
    hw_ecb_dec(&dk, want, back, 1);
    CHECK_BLK(back, pt, 16, "AES-%d FIPS-197 C decrypt", bits);
    if (idx == 0) {   /* Appendix B */
        uint8_t k2[16], p2[16], c2[16], w2[16];
        unhex("2b7e151628aed2a6abf7158809cf4f3c", k2, 16); unhex("3243f6a8885a308d313198a2e0370734", p2, 16);
        unhex("3925841d02dc09fbdc118597196a0b32", w2, 16);
        hw_expand(k2, 128, &ks); hw_ecb_enc(&ks, p2, c2, 1);
        CHECK_BLK(c2, w2, 16, "AES-128 FIPS-197 Appendix B encrypt");
    }
}
TEST(aesni_aes128_fips197_encrypt_decrypt) { aes_fips_c_case(0); }
TEST(aesni_aes192_fips197_encrypt_decrypt) { aes_fips_c_case(1); }
TEST(aesni_aes256_fips197_encrypt_decrypt) { aes_fips_c_case(2); }

TEST(aesni_equivalent_inverse_schedule_vs_sw) {
    NEED(F_AES); crypto_init();
    t_rng r; t_rng_seed(&r, 109);
    vbegin();
    for (int i = 0; i < 60; i++) {
        int bits = 128 + 64 * (i % 3);
        uint8_t key[32]; aes_sched ks, dk;
        t_rng_fill(&r, key, 32);
        sw_expand(key, bits, &ks); hw_dec_sched(&ks, &dk);
        for (int j = 1; j < ks.nr; j++) {
            V128 want = sw_invmixcolumns(ks.rk[ks.nr - j]);
            VCHECK(!memcmp(dk.rk[j].b, want.b, 16), "AES-%d dk[%d] = %s, InvMixColumns(rk[%d]) = %s", bits, j, hx(dk.rk[j].b, 16), ks.nr - j, hx(want.b, 16));
        }
    }
    VEND("AESIMC decryption schedule");
}

static void sp_mode_case(int mode) {   /* 0 ECB, 1 CBC, 2 CTR */
    NEED(F_AES); crypto_init();
    uint8_t pt[64], want[64], got[64], back[64], key[32], iv[16], ctr[16];
    unhex(SP_PT, pt, 64);
    for (int k = 0; k < 3; k++) {
        int bits = 128 + 64 * k; aes_sched ks, dk;
        unhex(SP_KEY[k], key, 32); hw_expand(key, bits, &ks); hw_dec_sched(&ks, &dk);
        if (mode == 0) {
            unhex(SP_ECB[k], want, 64);
            hw_ecb_enc(&ks, pt, got, 4); hw_ecb_dec(&dk, want, back, 4);
            for (int b = 0; b < 4; b++) {
                CHECK_BLK(got + 16 * b, want + 16 * b, 16, "ECB-AES%d.Encrypt block %d (F.1.%d)", bits, b + 1, 2 * k + 1);
                CHECK_BLK(back + 16 * b, pt + 16 * b, 16, "ECB-AES%d.Decrypt block %d (F.1.%d)", bits, b + 1, 2 * k + 2);
            }
        } else if (mode == 1) {
            unhex(SP_CBC_IV, iv, 16); unhex(SP_CBC[k], want, 64);
            hw_cbc_enc(&ks, iv, pt, got, 4); hw_cbc_dec(&dk, iv, want, back, 4);
            for (int b = 0; b < 4; b++) {
                CHECK_BLK(got + 16 * b, want + 16 * b, 16, "CBC-AES%d.Encrypt block %d (F.2.%d)", bits, b + 1, 2 * k + 1);
                CHECK_BLK(back + 16 * b, pt + 16 * b, 16, "CBC-AES%d.Decrypt block %d (F.2.%d)", bits, b + 1, 2 * k + 2);
            }
        } else {
            unhex(SP_CTR[k], want, 64);
            for (int ways = 1; ways <= 8; ways += 7) {
                unhex(SP_CTR_IV, ctr, 16); hw_ctr(&ks, ctr, pt, got, 64, 0, ways);
                unhex(SP_CTR_IV, ctr, 16); hw_ctr(&ks, ctr, want, back, 64, 0, ways);
                for (int b = 0; b < 4; b++) {
                    CHECK_BLK(got + 16 * b, want + 16 * b, 16, "CTR-AES%d.Encrypt block %d (F.5.%d, %d-way)", bits, b + 1, 2 * k + 1, ways);
                    CHECK_BLK(back + 16 * b, pt + 16 * b, 16, "CTR-AES%d.Decrypt block %d (F.5.%d, %d-way)", bits, b + 1, 2 * k + 2, ways);
                }
            }
        }
    }
}
TEST(aesni_ecb_sp800_38a) { sp_mode_case(0); }
TEST(aesni_cbc_sp800_38a) { sp_mode_case(1); }
TEST(aesni_ctr_sp800_38a) { sp_mode_case(2); }

TEST(aesni_random_keys_modes_vs_software) {
    NEED(F_AES); crypto_init();
    t_rng r; t_rng_seed(&r, 110);
    enum { NB = 37 };
    static uint8_t pt[NB * 16 + 16], a[NB * 16 + 16], b[NB * 16 + 16];
    vbegin();
    for (int i = 0; i < 24; i++) {
        int bits = 128 + 64 * (i % 3); uint8_t key[32], iv[16], ctr[16];
        aes_sched sk, hk, dk;
        t_rng_fill(&r, key, 32); t_rng_fill(&r, iv, 16); t_rng_fill(&r, pt, sizeof pt);
        if (i == 5) memset(iv + 8, 0xFF, 8);                 /* 64-bit carry in the CTR counter */
        sw_expand(key, bits, &sk); hw_expand(key, bits, &hk); hw_dec_sched(&hk, &dk);
        size_t nb = 1 + t_rng_range(&r, NB);
        sw_ecb(&sk, pt, a, nb, 0); hw_ecb_enc(&hk, pt, b, nb);
        VCHECK(!memcmp(a, b, nb * 16), "ECB-AES%d enc %zu blocks key %s", bits, nb, hx(key, bits / 8));
        sw_ecb(&sk, pt, a, nb, 1); hw_ecb_dec(&dk, pt, b, nb);
        VCHECK(!memcmp(a, b, nb * 16), "ECB-AES%d dec %zu blocks", bits, nb);
        sw_cbc(&sk, iv, pt, a, nb, 0); hw_cbc_enc(&hk, iv, pt, b, nb);
        VCHECK(!memcmp(a, b, nb * 16), "CBC-AES%d enc %zu blocks", bits, nb);
        sw_cbc(&sk, iv, pt, a, nb, 1); hw_cbc_dec(&dk, iv, pt, b, nb);
        VCHECK(!memcmp(a, b, nb * 16), "CBC-AES%d dec %zu blocks (4-way pipeline)", bits, nb);
        size_t n = t_rng_range(&r, NB * 16 + 1);
        sw_ctr(&sk, iv, pt, a, n, 0); memcpy(ctr, iv, 16); hw_ctr(&hk, ctr, pt, b, n, 0, 8);
        VCHECK(!memcmp(a, b, n), "CTR-AES%d %zu bytes (8-way)", bits, n);
    }
    VEND("AES random keys and modes");
}

/* XMM usage: 8 blocks in flight + round key; on x64 the compiler uses XMM6-15, so this also exercises the
 * callee-saved XMM spill/restore path through the case harness */
TEST(aesni_ctr_8way_vs_1way_long) {
    NEED(F_AES); crypto_init();
    enum { N = 64 * 1024 + 23 };
    uint8_t *in = (uint8_t *)malloc(N), *o1 = (uint8_t *)malloc(N), *o8 = (uint8_t *)malloc(N);
    if (!in || !o1 || !o8) { free(in); free(o1); free(o8); SKIP("out of memory"); }
    t_rng r; t_rng_seed(&r, 111); t_rng_fill(&r, in, N);
    uint8_t key[32], c0[16], c1[16]; aes_sched ks;
    t_rng_fill(&r, key, 32); t_rng_fill(&r, c0, 16);
    hw_expand(key, 256, &ks);
    memcpy(c1, c0, 16); hw_ctr(&ks, c1, in, o1, N, 0, 1);
    uint8_t c8[16]; memcpy(c8, c0, 16); hw_ctr(&ks, c8, in, o8, N, 0, 8);
    CHECK_BLK(o8, o1, N, "CTR-AES256 8-way vs 1-way, %d bytes", N);
    CHECK_BLK(c8, c1, 16, "final counter");
    aes_sched sk; sw_expand(key, 256, &sk);
    sw_ctr(&sk, c0, in, o1, 4096, 0);
    CHECK_BLK(o8, o1, 4096, "first 4 KiB vs software");
    free(in); free(o1); free(o8);
}

/* ---------------------------------------------------------------- GCM (AES-NI + PCLMULQDQ) */
static void gcm_case(const char *n1, const char *n2) {
    NEED(F_AES); NEED(F_PCLMULQDQ); crypto_init();
    for (size_t i = 0; i < T_COUNTOF(GCM_KATS); i++) {
        const gcm_kat *g = &GCM_KATS[i];
        if (strcmp(g->name, n1) && strcmp(g->name, n2)) continue;
        uint8_t gk[32], giv[12], gp[64], ga[32], gc[64], gt[16], oc[64], ot[16], op[64];
        unhex(g->key, gk, 32); unhex(g->iv, giv, 12);
        size_t pl = unhex(g->pt, gp, 64), al = unhex(g->aad, ga, 32);
        unhex(g->ct, gc, 64); unhex(g->tag, gt, 16);
        aes_gcm(1, gk, g->bits, giv, ga, al, gp, pl, oc, ot, 0);
        CHECK_BLK(oc, gc, pl, "GCM %s ciphertext (AES-%d, %zu-byte P, %zu-byte A)", g->name, g->bits, pl, al);
        CHECK_BLK(ot, gt, 16, "GCM %s tag", g->name);
        aes_gcm(1, gk, g->bits, giv, ga, al, gc, pl, op, ot, 1);
        CHECK_BLK(op, gp, pl, "GCM %s decrypt", g->name);
        CHECK_BLK(ot, gt, 16, "GCM %s decrypt tag", g->name);
    }
}
TEST(aesni_pclmul_gcm_tc1_tc2_aes128_zero) { gcm_case("tc1", "tc2"); }
TEST(aesni_pclmul_gcm_tc3_tc4_aes128)      { gcm_case("tc3", "tc4"); }
TEST(aesni_pclmul_gcm_tc13_tc14_aes256)    { gcm_case("tc13", "tc14"); }
TEST(aesni_pclmul_gcm_tc16_aes256_aad)     { gcm_case("tc16", "tc16"); }

TEST(aesni_pclmul_gcm_tamper_and_random_vs_sw) {
    NEED(F_AES); NEED(F_PCLMULQDQ); crypto_init();
    t_rng r; t_rng_seed(&r, 112);
    static uint8_t p[300], a[70], c1[300], c2[300];
    vbegin();
    for (int i = 0; i < 40; i++) {
        uint8_t key[32], iv[12], t1[16], t2[16];
        int bits = 128 + 64 * (i % 3);
        size_t pl = t_rng_range(&r, sizeof p + 1), al = t_rng_range(&r, sizeof a + 1);
        t_rng_fill(&r, key, 32); t_rng_fill(&r, iv, 12); t_rng_fill(&r, p, sizeof p); t_rng_fill(&r, a, sizeof a);
        aes_gcm(0, key, bits, iv, a, al, p, pl, c1, t1, 0);
        aes_gcm(1, key, bits, iv, a, al, p, pl, c2, t2, 0);
        VCHECK(!memcmp(c1, c2, pl) && !memcmp(t1, t2, 16), "GCM-AES%d P=%zu A=%zu: hw tag %s sw tag %s", bits, pl, al, hx(t2, 16), hx(t1, 16));
        if (pl) {   /* one flipped ciphertext bit must change the tag computed on decryption */
            uint8_t back[300], t3[16];
            c2[t_rng_range(&r, pl)] ^= (uint8_t)(1u << t_rng_range(&r, 8));
            aes_gcm(1, key, bits, iv, a, al, c2, pl, back, t3, 1);
            VCHECK(memcmp(t3, t1, 16) != 0, "tampered ciphertext produced the original tag");
        }
    }
    VEND("GCM random vs software");
}

/* ---------------------------------------------------------------- PCLMULQDQ */
static void pclmul_select_case(int imm, uint64_t seed) {
    NEED(F_PCLMULQDQ);
    t_rng r; t_rng_seed(&r, seed);
    vbegin();
    for (int i = 0; i < 3000; i++) {
        V128 a, b, got, want;
        gen128(&r, &a); gen128(&r, &b);
        hw_pclmul(&a, &b, imm, &got);
        want = m_pclmul(a, b, imm);
        VCHECK(!memcmp(got.b, want.b, 16), "pclmulqdq $0x%02x (%s, %s) = %s, model %s", imm, hx(a.b, 16), hx(b.b, 16), hx(got.b, 16), hx(want.b, 16));
    }
    char lbl[48]; snprintf(lbl, sizeof lbl, "pclmulqdq imm=0x%02x", imm);
    VEND(lbl);
}
TEST(pclmul_imm00_lo_lo) { pclmul_select_case(0x00, 201); }
TEST(pclmul_imm01_hi_lo) { pclmul_select_case(0x01, 202); }
TEST(pclmul_imm10_lo_hi) { pclmul_select_case(0x10, 203); }
TEST(pclmul_imm11_hi_hi) { pclmul_select_case(0x11, 204); }

TEST(pclmul_all_imm8_only_bits_0_and_4_matter) {
    NEED(F_PCLMULQDQ);
    t_rng r; t_rng_seed(&r, 205);
    vbegin();
    for (int imm = 0; imm < 256; imm++) for (int i = 0; i < 8; i++) {
        V128 a, b, got, want;
        gen128(&r, &a); gen128(&r, &b);
        hw_pclmul(&a, &b, imm, &got);
        want = m_pclmul(a, b, imm);
        VCHECK(!memcmp(got.b, want.b, 16), "pclmulqdq $0x%02x = %s, model (selects %d,%d) %s", imm, hx(got.b, 16), imm & 1, (imm >> 4) & 1, hx(want.b, 16));
    }
    VEND("pclmulqdq imm8 sweep");
}

TEST(pclmul_known_answers) {
    NEED(F_PCLMULQDQ);
    static const struct { uint64_t a, b, lo, hi; } kat[] = {
        { 3, 3, 5, 0 },
        { 0x87, 0x87, 0x4015, 0 },
        { 0xFFFFFFFFFFFFFFFFull, 1, 0xFFFFFFFFFFFFFFFFull, 0 },
        { 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull, 0x5555555555555555ull, 0x5555555555555555ull },
        { 0x8000000000000000ull, 0x8000000000000000ull, 0, 0x4000000000000000ull },
        { 0x8000000000000000ull, 2, 0, 1 },
        { 0, 0x123456789ABCDEFull, 0, 0 },
        { 0x100000001ull, 0x100000001ull, 1, 1 },   /* (x^32+1)^2 = x^64+1 */
    };
    for (size_t i = 0; i < T_COUNTOF(kat); i++) {
        uint64_t lo = kat[i].lo, hi = kat[i].hi;
        for (int sel = 0; sel < 4; sel++) {
            V128 a, b, o;
            a.q[0] = a.q[1] = 0xDEADBEEFDEADBEEFull; b.q[0] = b.q[1] = 0x0BADF00D0BADF00Dull;
            a.q[g_sel_imm[sel] & 1] = kat[i].a; b.q[(g_sel_imm[sel] >> 4) & 1] = kat[i].b;
            hw_pclmul(&a, &b, g_sel_imm[sel], &o);
            CHECKF(o.q[0] == lo && o.q[1] == hi, "clmul(0x%llx, 0x%llx) imm=0x%02x = %016llx:%016llx want %016llx:%016llx",
                   (unsigned long long)kat[i].a, (unsigned long long)kat[i].b, g_sel_imm[sel],
                   (unsigned long long)o.q[1], (unsigned long long)o.q[0], (unsigned long long)hi, (unsigned long long)lo);
        }
    }
}

TEST(pclmul_algebraic_properties) {
    NEED(F_PCLMULQDQ);
    t_rng r; t_rng_seed(&r, 206);
    vbegin();
    for (int i = 0; i < 1000; i++) {
        V128 a, b, c, ab, ba, ac, bxc, abxc, x;
        gen128(&r, &a); gen128(&r, &b); gen128(&r, &c);
        hw_pclmul(&a, &b, 0x00, &ab); hw_pclmul(&b, &a, 0x00, &ba);
        VCHECK(!memcmp(ab.b, ba.b, 16), "clmul not commutative for a=%s b=%s", hx(a.b, 8), hx(b.b, 8));
        bxc = vxor(b, c);
        hw_pclmul(&a, &c, 0x00, &ac); hw_pclmul(&a, &bxc, 0x00, &abxc); x = vxor(ab, ac);
        VCHECK(!memcmp(abxc.b, x.b, 16), "clmul not distributive over xor for a=%s", hx(a.b, 8));
        VCHECK((ab.q[1] >> 63) == 0, "bit 127 of a 64x64 carry-less product must be 0, got %s", hx(ab.b, 16));
    }
    VEND("clmul algebra");
}

TEST(pclmul_m128_memory_operand_form) {
    NEED(F_PCLMULQDQ);
    t_rng r; t_rng_seed(&r, 207);
    vbegin();
    for (int sel = 0; sel < 4; sel++) for (int i = 0; i < 200; i++) {
        V128 a, b, g1, g2;
        gen128(&r, &a); gen128(&r, &b);
        hw_pclmul(&a, &b, g_sel_imm[sel], &g1); hw_pclmul_mem(&a, &b, sel, &g2);
        VCHECK(!memcmp(g1.b, g2.b, 16), "pclmulqdq $0x%02x m128 %s != reg %s", g_sel_imm[sel], hx(g2.b, 16), hx(g1.b, 16));
    }
    VEND("pclmulqdq m128");
}

TEST(pclmul_ghash_multiply_vs_bitwise) {
    NEED(F_PCLMULQDQ); crypto_init();
    t_rng r; t_rng_seed(&r, 208);
    vbegin();
    for (int i = 0; i < 2000; i++) {
        V128 x, y; uint8_t a[16], b[16];
        gen128(&r, &x); gen128(&r, &y);
        sw_gf128_mul(x.b, y.b, a); hw_gf128_mul(x.b, y.b, b);
        VCHECK(!memcmp(a, b, 16), "GF(2^128) %s * %s: pclmul %s, bitwise %s", hx(x.b, 16), hx(y.b, 16), hx(b, 16), hx(a, 16));
    }
    VEND("GHASH multiply");
    uint8_t H[16], C[16], S[16], want[16];
    unhex("66e94bd4ef8a2c3b884cfa59ca342b2e", H, 16); unhex("0388dace60b6a392f328c2b971b2fe78", C, 16);
    unhex("f38cbb1ad69223dcc3457ae5b6b0f885", want, 16);
    ghash(hw_gf128_mul, H, NULL, 0, C, 16, S);
    CHECK_BLK(S, want, 16, "GHASH(H, {}, C) of GCM test case 2 with PCLMULQDQ");
}

static void crc_fold_case(int castagnoli, uint64_t seed) {
    NEED(F_PCLMULQDQ); crypto_init();
    crc_fold f;
    const uint32_t *tab = castagnoli ? g_crc32c_tab : g_crc32_tab;
    crc_fold_init(&f, castagnoli ? 0x1EDC6F41u : 0x04C11DB7u, tab);
    const uint8_t *s = (const uint8_t *)"123456789";
    uint32_t check = castagnoli ? 0xE3069283u : 0xCBF43926u;
    CHECK_EQ(hw_crc_fold(&f, s, 9, 1), check);
    /* "123456789" repeated to 9*16 bytes so the folding path runs */
    uint8_t rep[144]; for (int i = 0; i < 16; i++) memcpy(rep + 9 * i, s, 9);
    CHECK_EQ(hw_crc_fold(&f, rep, sizeof rep, 0), crc_sw(tab, rep, sizeof rep));
    CHECK_EQ(hw_crc_fold(&f, rep, sizeof rep, 1), crc_sw(tab, rep, sizeof rep));
    if (castagnoli) {
        uint8_t b[32];
        memset(b, 0, 32);    CHECK_EQ(hw_crc_fold(&f, b, 32, 0), 0x8A9136AAu);
        memset(b, 0xFF, 32); CHECK_EQ(hw_crc_fold(&f, b, 32, 0), 0x62A8AB43u);
        for (int i = 0; i < 32; i++) b[i] = (uint8_t)i;
        CHECK_EQ(hw_crc_fold(&f, b, 32, 0), 0x46DD794Eu);
        for (int i = 0; i < 32; i++) b[i] = (uint8_t)(31 - i);
        CHECK_EQ(hw_crc_fold(&f, b, 32, 0), 0x113FDB5Cu);
    }
    enum { MAXN = 4099 };
    static uint8_t buf[MAXN + 1];
    t_rng r; t_rng_seed(&r, seed);
    t_rng_fill(&r, buf, sizeof buf);
    vbegin();
    for (size_t n = 0; n <= 300; n++) for (int by4 = 0; by4 < 2; by4++) {
        uint32_t want = crc_sw(tab, buf + 1, n), got = hw_crc_fold(&f, buf + 1, n, by4);
        VCHECK(got == want, "%s len=%zu by4=%d: pclmul 0x%08x table 0x%08x", castagnoli ? "CRC32C" : "CRC32", n, by4, got, want);
    }
    for (int i = 0; i < 40; i++) {
        size_t n = t_rng_range(&r, MAXN + 1);
        uint32_t want = crc_sw(tab, buf, n), got = hw_crc_fold(&f, buf, n, 1);
        VCHECK(got == want, "%s len=%zu: pclmul fold-by-4 0x%08x table 0x%08x", castagnoli ? "CRC32C" : "CRC32", n, got, want);
    }
    VEND(castagnoli ? "CRC32C pclmul folding" : "CRC32 pclmul folding");
}
TEST(pclmul_crc32c_folding_vs_table) { crc_fold_case(1, 209); }
TEST(pclmul_crc32_ieee_folding_vs_table) { crc_fold_case(0, 210); }

TEST(crc32c_pclmul_vs_sse42_crc32_instruction) {
    NEED(F_SSE42); crypto_init();
    t_rng r; t_rng_seed(&r, 211);
    static uint8_t buf[2048 + 8];
    t_rng_fill(&r, buf, sizeof buf);
    CHECK_EQ(hw_crc32c_sse42((const uint8_t *)"123456789", 9, 1), 0xE3069283u);
    CHECK_EQ(hw_crc32c_sse42((const uint8_t *)"123456789", 9, 4), 0xE3069283u);
    crc_fold f; crc_fold_init(&f, 0x1EDC6F41u, g_crc32c_tab);
    int have_clmul = cpu_has(F_PCLMULQDQ);
    if (!have_clmul) t_note("PCLMULQDQ absent: SSE4.2 CRC32 compared with the table only");
    vbegin();
    for (int i = 0; i < 300; i++) {
        size_t off = t_rng_range(&r, 8), n = t_rng_range(&r, 2048);
        uint32_t want = crc_sw(g_crc32c_tab, buf + off, n);
        for (int w = 1; w <= 8; w *= 2) {
            uint32_t got = hw_crc32c_sse42(buf + off, n, w);
            VCHECK(got == want, "crc32 instruction width %d len=%zu off=%zu: 0x%08x table 0x%08x", w, n, off, got, want);
        }
        if (have_clmul) { uint32_t got = hw_crc_fold(&f, buf + off, n, 1); VCHECK(got == want, "pclmul CRC32C len=%zu: 0x%08x table 0x%08x", n, got, want); }
    }
    VEND("CRC32C SSE4.2/pclmul/table");
}

/* ---------------------------------------------------------------- SHA extensions */
static void sha_op_random(int op_lo, int op_hi, uint64_t seed) {
    NEED(F_SHA);
    t_rng r; t_rng_seed(&r, seed);
    for (int op = op_lo; op <= op_hi; op++) {
        vbegin();
        for (int i = 0; i < 3000; i++) {
            V128 a, b, k, got, want;
            gen128(&r, &a); gen128(&r, &b); gen128(&r, &k);
            hw_shaop(op, &a, &b, &k, &got);
            want = m_shaop(op, a, b, k);
            VCHECK(!memcmp(got.b, want.b, 16), "%s(%s, %s%s%s) = %s, model %s", g_shaop_name[op], hx(a.b, 16), hx(b.b, 16),
                   op == SH256_RNDS2 ? ", xmm0=" : "", op == SH256_RNDS2 ? hx(k.b, 8) : "", hx(got.b, 16), hx(want.b, 16));
        }
        VEND(g_shaop_name[op]);
    }
}
TEST(sha1rnds4_all_functions_vs_model) { sha_op_random(SH_RNDS4_0, SH_RNDS4_3, 301); }
TEST(sha1nexte_vs_model)               { sha_op_random(SH_NEXTE, SH_NEXTE, 302); }
TEST(sha1msg1_vs_model)                { sha_op_random(SH_MSG1, SH_MSG1, 303); }
TEST(sha1msg2_vs_model)                { sha_op_random(SH_MSG2, SH_MSG2, 304); }
TEST(sha256rnds2_implicit_xmm0_vs_model) { sha_op_random(SH256_RNDS2, SH256_RNDS2, 305); }
TEST(sha256msg1_vs_model)              { sha_op_random(SH256_MSG1, SH256_MSG1, 306); }
TEST(sha256msg2_vs_model)              { sha_op_random(SH256_MSG2, SH256_MSG2, 307); }

TEST(sha1rnds4_m128_fixed_register_form) {
    /* sha1rnds4 with an aligned m128 source and fixed registers (asm-encoded) == intrinsic form */
    NEED(F_SHA);
    t_rng r; t_rng_seed(&r, 308);
    vbegin();
    for (int i = 0; i < 500; i++) {
        V128 a, b, k, g1, g2;
        gen128(&r, &a); gen128(&r, &b); memset(&k, 0, sizeof k);
        hw_shaop(SH_RNDS4_2, &a, &b, &k, &g1); hw_shaop_mem(SH_RNDS4_2, &a, &b, &k, &g2);
        VCHECK(!memcmp(g1.b, g2.b, 16), "sha1rnds4 $2 m128 %s != reg %s", hx(g2.b, 16), hx(g1.b, 16));
    }
    VEND("sha1rnds4 m128 form");
}

TEST(sha_m128_memory_operand_forms) {
    NEED(F_SHA);
    t_rng r; t_rng_seed(&r, 309);
    static const int ops[] = { SH_NEXTE, SH_MSG1, SH_MSG2, SH256_RNDS2, SH256_MSG1, SH256_MSG2 };
    vbegin();
    for (size_t j = 0; j < T_COUNTOF(ops); j++) for (int i = 0; i < 300; i++) {
        V128 a, b, k, g1, g2;
        gen128(&r, &a); gen128(&r, &b); gen128(&r, &k);
        hw_shaop(ops[j], &a, &b, &k, &g1); hw_shaop_mem(ops[j], &a, &b, &k, &g2);
        VCHECK(!memcmp(g1.b, g2.b, 16), "%s m128/fixed-register form %s != intrinsic form %s", g_shaop_name[ops[j]], hx(g2.b, 16), hx(g1.b, 16));
    }
    VEND("SHA m128 forms");
}

TEST(sha1ni_digest_known_answers) { NEED(F_SHA); sha_kat_run(ALG_SHA1, hw_sha1_blocks, "SHA-NI"); }
TEST(sha256ni_digest_known_answers) { NEED(F_SHA); sha_kat_run(ALG_SHA256, hw_sha256_blocks, "SHA-NI"); }

static void sha_random_lengths(int alg, compress_fn hw, compress_fn sw, uint64_t seed) {
    t_rng r; t_rng_seed(&r, seed);
    static uint8_t buf[1100];
    t_rng_fill(&r, buf, sizeof buf);
    vbegin();
    for (size_t n = 0; n <= 300; n++) {
        uint8_t a[64], b[64];
        size_t dn = md_digest(alg, sw, buf, n, a); md_digest(alg, hw, buf, n, b);
        VCHECK(!memcmp(a, b, dn), "%s len=%zu: hw %s sw %s", g_alg_name[alg], n, hx(b, dn), hx(a, dn));
    }
    for (int i = 0; i < 30; i++) {
        uint8_t a[64], b[64]; size_t n = t_rng_range(&r, sizeof buf + 1);
        size_t dn = md_digest(alg, sw, buf, n, a); md_digest(alg, hw, buf, n, b);
        VCHECK(!memcmp(a, b, dn), "%s len=%zu: hw %s sw %s", g_alg_name[alg], n, hx(b, dn), hx(a, dn));
    }
    VEND(g_alg_name[alg]);
}
TEST(sha1ni_random_lengths_vs_software)   { NEED(F_SHA); sha_random_lengths(ALG_SHA1, hw_sha1_blocks, sw_sha1_blocks, 310); }
TEST(sha256ni_random_lengths_vs_software) { NEED(F_SHA); sha_random_lengths(ALG_SHA256, hw_sha256_blocks, sw_sha256_blocks, 311); }

/* ---------------------------------------------------------------- SHA512 extensions (VEX.256) */
static void sha512_op_case(int op, uint64_t seed) {
    NEED(F_SHA512);
    t_rng r; t_rng_seed(&r, seed);
    static const char *const nm[] = { "vsha512rnds2", "vsha512msg1", "vsha512msg2" };
    vbegin();
    for (int i = 0; i < 2000; i++) {
        Q4 a, b, got, want; uint64_t k[2];
        t_rng_fill(&r, &a, sizeof a); t_rng_fill(&r, &b, sizeof b); t_rng_fill(&r, k, sizeof k);
        if (i % 7 == 0) memset(&a, 0, sizeof a);
        hw_sha512op_ymm(op, &a, &b, k, &got);
        want = op == S512_RNDS2 ? m_sha512rnds2(a, b, k[0], k[1]) : op == S512_MSG1 ? m_sha512msg1(a, b.q[0]) : m_sha512msg2(a, b);
        VCHECK(!memcmp(&got, &want, 32), "%s(%s, %s) = %s, model %s", nm[op], hx(&a, 32), hx(&b, 32), hx(&got, 32), hx(&want, 32));
    }
    VEND(nm[op]);
}
TEST(sha512ext_vsha512rnds2_vs_model) { sha512_op_case(S512_RNDS2, 401); }
TEST(sha512ext_vsha512msg1_vs_model)  { sha512_op_case(S512_MSG1, 402); }
TEST(sha512ext_vsha512msg2_vs_model)  { sha512_op_case(S512_MSG2, 403); }
TEST(sha512ext_digest_known_answers) {
    NEED(F_SHA512);
    sha_kat_run(ALG_SHA512, hw_sha512_blocks_avx, "VSHA512");
    sha_random_lengths(ALG_SHA512, hw_sha512_blocks_avx, sw_sha512_blocks, 404);
}

/* ---------------------------------------------------------------- GFNI */
TEST(gfni_gf2p8mulb_vs_model) {
    NEED(F_GFNI); crypto_init();
    vbegin();
    for (int a = 0; a < 256; a++) for (int b0 = 0; b0 < 256; b0 += 16) {   /* all 65536 products, 16 per vector */
        V128 x, y, got, want;
        memset(x.b, a, 16);
        for (int j = 0; j < 16; j++) y.b[j] = (uint8_t)(b0 + j);
        hw_gfni_mul(&x, &y, &got);
        want = m_gf2p8mul(x, y);
        VCHECK(!memcmp(got.b, want.b, 16), "gf2p8mulb(0x%02x, %s) = %s, model %s", a, hx(y.b, 16), hx(got.b, 16), hx(want.b, 16));
    }
    VEND("gf2p8mulb exhaustive");
    V128 x, y, o; memset(x.b, 0x57, 16); memset(y.b, 0x83, 16);
    hw_gfni_mul(&x, &y, &o);
    CHECK_EQ(o.b[7], 0xc1);   /* FIPS-197 4.2 example */
}
TEST(gfni_affineinv_computes_aes_sbox) {
    NEED(F_GFNI); crypto_init();
    V128 A; A.q[0] = A.q[1] = 0xF1E3C78F1F3E7CF8ull;   /* AES affine matrix in GF2P8AFFINE row order */
    uint8_t sb[256];
    for (int b0 = 0; b0 < 256; b0 += 16) {
        V128 x, o;
        for (int j = 0; j < 16; j++) x.b[j] = (uint8_t)(b0 + j);
        hw_gfni_affine(&x, &A, 0x63, 1, &o);
        memcpy(sb + b0, o.b, 16);
    }
    CHECK_BLK(sb, g_sbox, 256, "gf2p8affineinvqb(x, AES matrix, 0x63) == AES S-box");
    CHECK_EQ(sb[0x53], 0xed);
}
TEST(gfni_affine_and_affineinv_vs_model) {
    NEED(F_GFNI); crypto_init();
    t_rng r; t_rng_seed(&r, 501);
    vbegin();
    for (int inv = 0; inv < 2; inv++) for (size_t ii = 0; ii < T_COUNTOF(g_gf_imms); ii++) for (int i = 0; i < 300; i++) {
        V128 x, A, got, want;
        gen128(&r, &x); gen128(&r, &A);
        if (i == 0) { A.q[0] = A.q[1] = 0x0102040810204080ull; }   /* identity matrix */
        hw_gfni_affine(&x, &A, g_gf_imms[ii], inv, &got);
        want = m_gf2p8affine(x, A, g_gf_imms[ii], inv);
        VCHECK(!memcmp(got.b, want.b, 16), "gf2p8affine%sqb(%s, %s, 0x%02x) = %s, model %s", inv ? "inv" : "", hx(x.b, 16), hx(A.b, 16),
               g_gf_imms[ii], hx(got.b, 16), hx(want.b, 16));
        if (i == 0 && !inv) { V128 e = x; for (int j = 0; j < 16; j++) e.b[j] ^= g_gf_imms[ii];
            VCHECK(!memcmp(got.b, e.b, 16), "identity matrix affine must give x ^ imm, got %s", hx(got.b, 16)); }
    }
    VEND("GF2P8AFFINE(INV)QB");
}

/* ---------------------------------------------------------------- VEX.128 forms, VAES and VPCLMULQDQ */
TEST(avx_vex128_aes_pclmul_unaligned_m128) {
    NEED(F_AES); NEED(F_AVX); crypto_init();
    t_rng r; t_rng_seed(&r, 601);
    static uint8_t mem[64];
    vbegin();
    for (int i = 0; i < 400; i++) {
        V128 a, k, want; uint8_t got[16];
        int off = 1 + (int)t_rng_range(&r, 15);
        gen128(&r, &a); gen128(&r, &k);
        memcpy(mem + off, k.b, 16);
        int op = i % 5;
        hw_aesop_vex128_avx(op, a.b, mem + off, got);
        want = op == OP_IMC ? m_aesimc(k) : m_aesop(op, a, k);
        VCHECK(!memcmp(got, want.b, 16), "v%s with m128 at offset %d = %s, model %s", g_aesop_name[op], off, hx(got, 16), hx(want.b, 16));
        if (cpu_has(F_PCLMULQDQ)) {
            int sel = i & 3;
            hw_pclmul_vex128_avx(a.b, mem + off, sel, got);
            want = m_pclmul(a, k, g_sel_imm[sel]);
            VCHECK(!memcmp(got, want.b, 16), "vpclmulqdq $0x%02x m128 offset %d = %s, model %s", g_sel_imm[sel], off, hx(got, 16), hx(want.b, 16));
        }
    }
    VEND("VEX.128 AES/PCLMULQDQ unaligned");
}

TEST(vaes_ymm_per_lane_vs_model) {
    NEED(F_VAES); crypto_init();
    t_rng r; t_rng_seed(&r, 602);
    vbegin();
    for (int op = OP_ENC; op <= OP_DECLAST; op++) for (int i = 0; i < 500; i++) {
        B256 a, k, o;
        V128 a0, a1, k0, k1;
        gen128(&r, &a0); gen128(&r, &a1); gen128(&r, &k0); gen128(&r, &k1);
        memcpy(a.b, a0.b, 16); memcpy(a.b + 16, a1.b, 16); memcpy(k.b, k0.b, 16); memcpy(k.b + 16, k1.b, 16);
        hw_vaes_ymm(op, a.b, k.b, o.b);
        V128 w0 = m_aesop(op, a0, k0), w1 = m_aesop(op, a1, k1);
        VCHECK(!memcmp(o.b, w0.b, 16) && !memcmp(o.b + 16, w1.b, 16), "v%s ymm lanes = %s|%s, model %s|%s", g_aesop_name[op],
               hx(o.b, 16), hx(o.b + 16, 16), hx(w0.b, 16), hx(w1.b, 16));
    }
    VEND("VAES ymm");
}
TEST(vaes_ymm_ctr_vs_aesni_and_sp800_38a) {
    NEED(F_VAES); crypto_init();
    uint8_t pt[64], want[64], got[64], key[32], ctr[16];
    unhex(SP_PT, pt, 64);
    for (int k = 0; k < 3; k++) {
        aes_sched ks; int bits = 128 + 64 * k;
        unhex(SP_KEY[k], key, 32); sw_expand(key, bits, &ks);
        unhex(SP_CTR[k], want, 64); unhex(SP_CTR_IV, ctr, 16);
        hw_vaes_ctr_ymm(&ks, ctr, pt, got, 4);
        CHECK_BLK(got, want, 64, "VAES ymm CTR-AES%d (SP 800-38A F.5.%d)", bits, 2 * k + 1);
    }
    if (!cpu_has(F_AES)) { t_note("AES absent: VAES CTR long-buffer check vs AES-NI skipped"); return; }
    enum { N = 4096 };
    static uint8_t in[N], o1[N], o2[N];
    t_rng r; t_rng_seed(&r, 603); t_rng_fill(&r, in, N);
    uint8_t c0[16], c1[16]; aes_sched ks; t_rng_fill(&r, key, 32); t_rng_fill(&r, c0, 16);
    sw_expand(key, 256, &ks);
    memcpy(c1, c0, 16); hw_ctr(&ks, c1, in, o1, N - 16, 0, 8);
    memcpy(c1, c0, 16); hw_vaes_ctr_ymm(&ks, c1, in, o2, N / 16 - 1);
    CHECK_BLK(o2, o1, N - 16, "VAES ymm CTR vs AES-NI 8-way (odd block count)");
}
TEST(vpclmulqdq_ymm_per_lane_vs_model) {
    NEED(F_VPCLMULQDQ);
    t_rng r; t_rng_seed(&r, 604);
    vbegin();
    for (int sel = 0; sel < 4; sel++) for (int i = 0; i < 500; i++) {
        V128 a0, a1, b0, b1; B256 a, b, o;
        gen128(&r, &a0); gen128(&r, &a1); gen128(&r, &b0); gen128(&r, &b1);
        memcpy(a.b, a0.b, 16); memcpy(a.b + 16, a1.b, 16); memcpy(b.b, b0.b, 16); memcpy(b.b + 16, b1.b, 16);
        hw_vpclmul_ymm(a.b, b.b, sel, o.b);
        V128 w0 = m_pclmul(a0, b0, g_sel_imm[sel]), w1 = m_pclmul(a1, b1, g_sel_imm[sel]);
        VCHECK(!memcmp(o.b, w0.b, 16) && !memcmp(o.b + 16, w1.b, 16), "vpclmulqdq ymm $0x%02x = %s|%s, model %s|%s", g_sel_imm[sel],
               hx(o.b, 16), hx(o.b + 16, 16), hx(w0.b, 16), hx(w1.b, 16));
    }
    VEND("VPCLMULQDQ ymm");
}
TEST(vaes_zmm_per_lane_vs_model) {
    NEED(F_VAES); NEED(F_AVX512F); crypto_init();
    t_rng r; t_rng_seed(&r, 605);
    vbegin();
    for (int op = OP_ENC; op <= OP_DECLAST; op++) for (int i = 0; i < 300; i++) {
        B512 a, k, o; V128 al[4], kl[4];
        for (int l = 0; l < 4; l++) { gen128(&r, &al[l]); gen128(&r, &kl[l]); memcpy(a.b + 16 * l, al[l].b, 16); memcpy(k.b + 16 * l, kl[l].b, 16); }
        hw_vaes_zmm512(op, a.b, k.b, o.b);
        for (int l = 0; l < 4; l++) {
            V128 w = m_aesop(op, al[l], kl[l]);
            VCHECK(!memcmp(o.b + 16 * l, w.b, 16), "v%s zmm lane %d = %s, model %s", g_aesop_name[op], l, hx(o.b + 16 * l, 16), hx(w.b, 16));
        }
    }
    VEND("VAES zmm");
}
TEST(vpclmulqdq_zmm_per_lane_vs_model) {
    NEED(F_VPCLMULQDQ); NEED(F_AVX512F);
    t_rng r; t_rng_seed(&r, 606);
    vbegin();
    for (int sel = 0; sel < 4; sel++) for (int i = 0; i < 300; i++) {
        B512 a, b, o; V128 al[4], bl[4];
        for (int l = 0; l < 4; l++) { gen128(&r, &al[l]); gen128(&r, &bl[l]); memcpy(a.b + 16 * l, al[l].b, 16); memcpy(b.b + 16 * l, bl[l].b, 16); }
        hw_vpclmul_zmm512(a.b, b.b, sel, o.b);
        for (int l = 0; l < 4; l++) {
            V128 w = m_pclmul(al[l], bl[l], g_sel_imm[sel]);
            VCHECK(!memcmp(o.b + 16 * l, w.b, 16), "vpclmulqdq zmm $0x%02x lane %d = %s, model %s", g_sel_imm[sel], l, hx(o.b + 16 * l, 16), hx(w.b, 16));
        }
    }
    VEND("VPCLMULQDQ zmm");
}
TEST(vaes_vpclmul_evex256_upper_registers_x64) {
#ifndef T_X64
    SKIP("YMM16-31 (EVEX register extension) exist only in 64-bit mode");
#else
    NEED(F_VAES); NEED(F_AVX512VL); crypto_init();
    t_rng r; t_rng_seed(&r, 607);
    vbegin();
    for (int i = 0; i < 300; i++) {
        V128 a0, a1, k0, k1; B256 a, k; B512 o;
        gen128(&r, &a0); gen128(&r, &a1); gen128(&r, &k0); gen128(&r, &k1);
        memcpy(a.b, a0.b, 16); memcpy(a.b + 16, a1.b, 16); memcpy(k.b, k0.b, 16); memcpy(k.b + 16, k1.b, 16);
        hw_vaes_evex256_ymm16(a.b, k.b, o.b);
        V128 e0 = m_aesenc(a0, k0), e1 = m_aesenc(a1, k1), d0 = m_aesdeclast(e0, k0), d1 = m_aesdeclast(e1, k1);
        VCHECK(!memcmp(o.b, e0.b, 16) && !memcmp(o.b + 16, e1.b, 16), "EVEX vaesenc ymm18 = %s|%s, model %s|%s", hx(o.b, 16), hx(o.b + 16, 16), hx(e0.b, 16), hx(e1.b, 16));
        VCHECK(!memcmp(o.b + 32, d0.b, 16) && !memcmp(o.b + 48, d1.b, 16), "EVEX vaesdeclast ymm19 mismatch");
        if (cpu_has(F_VPCLMULQDQ)) {
            B256 c;
            hw_vpclmul_evex256_ymm16(a.b, k.b, c.b);
            V128 w0 = m_pclmul(a0, k0, 0x10), w1 = m_pclmul(a1, k1, 0x10);
            VCHECK(!memcmp(c.b, w0.b, 16) && !memcmp(c.b + 16, w1.b, 16), "EVEX vpclmulqdq ymm18 = %s|%s", hx(c.b, 16), hx(c.b + 16, 16));
        }
    }
    VEND("EVEX.256 VAES/VPCLMULQDQ on YMM16-19");
#endif
}

/* ---------------------------------------------------------------- faults (child processes) */
CHILD(aes_misaligned_m128) {
    static T_ALIGN(16) uint8_t buf[64];
    __asm__ volatile("pxor %%xmm1, %%xmm1\n\taesenc 1(%0), %%xmm1" : : "r"(buf) : "xmm1", "memory");
    printf("NOFAULT=1\n");
    return 0;
}
CHILD(aes_aligned_m128) {
    static T_ALIGN(16) uint8_t buf[64];
    __asm__ volatile("pxor %%xmm1, %%xmm1\n\taesenc 16(%0), %%xmm1" : : "r"(buf) : "xmm1", "memory");
    printf("NOFAULT=1\n");
    return 0;
}
CHILD(sha_misaligned_m128) {
    static T_ALIGN(16) uint8_t buf[64];
    __asm__ volatile("pxor %%xmm1, %%xmm1\n\tsha256msg1 4(%0), %%xmm1" : : "r"(buf) : "xmm1", "memory");
    printf("NOFAULT=1\n");
    return 0;
}
static t_child_result g_cr;
#define XP_STATUS_ACCESS_VIOLATION 0xC0000005u
#define XP_STATUS_DATATYPE_MISALIGNMENT 0x80000002u
static void misaligned_case(const char *child, t_feature f) {
    NEED(f);
    if (t_get_mxcsr() & (1u << 17)) SKIP("MXCSR.MM (AMD misaligned SSE mode) is set: misaligned legacy m128 operands are allowed");
    if (cpu_has(F_AES)) {
        REQUIREF(t_child_run("aes_aligned_m128", "", 5000, &g_cr), "child did not start (error %lu)", g_cr.spawn_error);
        CHECKF(g_cr.exc_code == 0 && g_cr.exit_code == 0, "aligned control child failed: exit 0x%lx exc 0x%lx", g_cr.exit_code, g_cr.exc_code);
    }
    REQUIREF(t_child_run(child, "", 5000, &g_cr), "child did not start (error %lu)", g_cr.spawn_error);
    CHECKF(!g_cr.timed_out, "child hung");
    CHECKF(g_cr.exc_code != 0, "misaligned legacy-SSE m128 operand did not fault (exit 0x%lx, output: %.80s)", g_cr.exit_code, g_cr.out);
    if (g_cr.exc_code) {
        t_info("exception_code", "0x%08lx", g_cr.exc_code);
        CHECKF(g_cr.exc_code == XP_STATUS_ACCESS_VIOLATION || g_cr.exc_code == XP_STATUS_DATATYPE_MISALIGNMENT,
               "#GP from a misaligned m128 reported as 0x%08lx (expected STATUS_ACCESS_VIOLATION)", g_cr.exc_code);
    }
}
TEST(aesni_legacy_misaligned_m128_faults_child) { misaligned_case("aes_misaligned_m128", F_AES); }
TEST(sha_legacy_misaligned_m128_faults_child) { misaligned_case("sha_misaligned_m128", F_SHA); }

CHILD(probe_unadvertised) {
    const char *w = argc > 1 ? argv[1] : "";
    if (!strcmp(w, "aes")) __asm__ volatile("pxor %%xmm1, %%xmm1\n\taesenc %%xmm1, %%xmm1" ::: "xmm1");
    else if (!strcmp(w, "pclmul")) __asm__ volatile("pxor %%xmm1, %%xmm1\n\tpclmulqdq $0, %%xmm1, %%xmm1" ::: "xmm1");
    else if (!strcmp(w, "sha")) __asm__ volatile("pxor %%xmm1, %%xmm1\n\tsha256msg1 %%xmm1, %%xmm1" ::: "xmm1");
    else if (!strcmp(w, "gfni")) __asm__ volatile("pxor %%xmm1, %%xmm1\n\tgf2p8mulb %%xmm1, %%xmm1" ::: "xmm1");
    else return 3;
    printf("PROBE_EXECUTED=1\n");
    return 0;
}
/* Diagnostic only: what happens when a crypto instruction is executed although CPUID does not advertise it
 * (a hypervisor may hide a feature the silicon still implements, so no hard expectation). */
TEST(crypto_unadvertised_instruction_probe_child) {
    static const struct { t_feature f; const char *arg; } p[] = { { F_AES, "aes" }, { F_PCLMULQDQ, "pclmul" }, { F_SHA, "sha" }, { F_GFNI, "gfni" } };
    int probed = 0;
    for (size_t i = 0; i < T_COUNTOF(p); i++) {
        if (cpu_hw(p[i].f)) continue;
        probed++;
        if (!t_child_run("probe_unadvertised", p[i].arg, 5000, &g_cr)) { t_note("probe child for %s did not start", p[i].arg); continue; }
        if (g_cr.exc_code) t_info(p[i].arg, "exception 0x%08lx (0xC000001D = STATUS_ILLEGAL_INSTRUCTION)", g_cr.exc_code);
        else t_info(p[i].arg, "executed without fault although CPUID bit is clear (exit %lu)", g_cr.exit_code);
        CHECKF(!g_cr.timed_out, "probe child for %s hung", p[i].arg);
    }
    if (!probed) SKIP("AES, PCLMULQDQ, SHA and GFNI are all advertised: nothing to probe");
}

/* ---------------------------------------------------------------- OS contract: threads, suspension, context */
typedef struct {
    aes_sched ks; uint8_t ctr0[16]; const uint8_t *in; const uint8_t *expect; uint8_t *out; size_t len;
    int iters; volatile long bad, done_iters; int use_sha; uint8_t sha_expect[32]; volatile long *stop;
} crypto_job;
static DWORD WINAPI crypto_worker(LPVOID pv) {
    crypto_job *j = (crypto_job *)pv;
    for (int it = 0; it < j->iters || (j->stop && !*j->stop); it++) {
        uint8_t c[16]; memcpy(c, j->ctr0, 16);
        hw_ctr(&j->ks, c, j->in, j->out, j->len, 0, 8);
        if (memcmp(j->out, j->expect, j->len)) InterlockedIncrement(&j->bad);
        if (j->use_sha) {
            uint8_t d[32]; md_digest(ALG_SHA256, hw_sha256_blocks, j->in, j->len, d);
            if (memcmp(d, j->sha_expect, 32)) InterlockedIncrement(&j->bad);
        }
        InterlockedIncrement(&j->done_iters);
        if (!j->stop) { if (it & 1) Sleep(0); else SwitchToThread(); }
        if (j->stop && it > 100000) break;
    }
    return 0;
}
TEST(crypto_threads_per_thread_key_isolation) {
    NEED(F_AES); crypto_init();
    enum { NT = 8, LEN = 8192 + 48 };
    static crypto_job jobs[NT];
    static uint8_t in[NT][LEN], exp_[NT][LEN], out[NT][LEN];
    HANDLE h[NT]; int started = 0;
    int sha = cpu_has(F_SHA);
    t_rng r; t_rng_seed(&r, 701);
    for (int i = 0; i < NT; i++) {
        crypto_job *j = &jobs[i]; uint8_t key[32];
        memset(j, 0, sizeof *j);
        t_rng_fill(&r, key, 32); t_rng_fill(&r, j->ctr0, 16); t_rng_fill(&r, in[i], LEN);
        sw_expand(key, 128 + 64 * (i % 3), &j->ks);
        sw_ctr(&j->ks, j->ctr0, in[i], exp_[i], LEN, 0);
        j->in = in[i]; j->expect = exp_[i]; j->out = out[i]; j->len = LEN; j->iters = 40; j->use_sha = sha;
        if (sha) md_digest(ALG_SHA256, sw_sha256_blocks, in[i], LEN, j->sha_expect);
    }
    for (int i = 0; i < NT; i++) {
        h[i] = CreateThread(NULL, 0, crypto_worker, &jobs[i], 0, NULL);
        if (!h[i]) break;
        started++;
    }
    if (started < 2) { for (int i = 0; i < started; i++) { WaitForSingleObject(h[i], 30000); CloseHandle(h[i]); } SKIP("could not create worker threads"); }
    DWORD w = WaitForMultipleObjects((DWORD)started, h, TRUE, 30000);
    CHECKF(w != WAIT_TIMEOUT, "workers did not finish within 30 s");
    if (w == WAIT_TIMEOUT) for (int i = 0; i < started; i++) TerminateThread(h[i], 1);
    long total = 0;
    for (int i = 0; i < started; i++) {
        CloseHandle(h[i]);
        CHECKF(jobs[i].bad == 0, "thread %d: %ld of %ld iterations produced wrong AES-CTR%s output", i, jobs[i].bad, jobs[i].done_iters, sha ? "/SHA-256" : "");
        total += jobs[i].done_iters;
    }
    CHECK_EQ(total, (long)started * 40);
    t_info("threads", "%d threads x 40 iterations (AES-CTR %s)", started, sha ? "+ SHA-NI SHA-256" : "only");
}

TEST(crypto_state_survives_suspend_getsetcontext) {
    NEED(F_AES); crypto_init();
    enum { LEN = 4096 };
    static crypto_job job;
    static uint8_t in[LEN], exp_[LEN], out[LEN];
    volatile long stop = 0;
    t_rng r; t_rng_seed(&r, 702);
    uint8_t key[32];
    memset(&job, 0, sizeof job);
    t_rng_fill(&r, key, 32); t_rng_fill(&r, job.ctr0, 16); t_rng_fill(&r, in, LEN);
    sw_expand(key, 256, &job.ks); sw_ctr(&job.ks, job.ctr0, in, exp_, LEN, 0);
    job.in = in; job.expect = exp_; job.out = out; job.len = LEN; job.iters = 0; job.stop = &stop;
    job.use_sha = cpu_has(F_SHA);
    if (job.use_sha) md_digest(ALG_SHA256, sw_sha256_blocks, in, LEN, job.sha_expect);
    HANDLE h = CreateThread(NULL, 0, crypto_worker, &job, 0, NULL);
    if (!h) SKIP("CreateThread failed (%lu)", GetLastError());
    DWORD t0 = GetTickCount();
    while (job.done_iters < 2 && GetTickCount() - t0 < 5000) Sleep(1);
    int sus = 0, gets = 0, sets = 0, fails = 0;
    for (int i = 0; i < 300; i++) {
        if (SuspendThread(h) == (DWORD)-1) { fails++; continue; }
        sus++;
        CONTEXT ctx; memset(&ctx, 0, sizeof ctx);
#ifdef T_X64
        ctx.ContextFlags = CONTEXT_FULL | CONTEXT_FLOATING_POINT;
#else
        ctx.ContextFlags = CONTEXT_FULL | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS;
#endif
        if (GetThreadContext(h, &ctx)) {
            gets++;
            if (i & 1) { if (SetThreadContext(h, &ctx)) sets++; else fails++; }
        } else fails++;
        ResumeThread(h);
        if (i % 8 == 0) Sleep(0);
    }
    long before = job.done_iters;
    t0 = GetTickCount();
    while (job.done_iters < before + 3 && GetTickCount() - t0 < 5000) Sleep(1);
    stop = 1;
    DWORD w = WaitForSingleObject(h, 10000);
    CHECKF(w == WAIT_OBJECT_0, "worker did not stop");
    if (w != WAIT_OBJECT_0) TerminateThread(h, 1);
    CloseHandle(h);
    CHECKF(fails == 0, "%d Suspend/Get/SetThreadContext calls failed", fails);
    CHECKF(job.bad == 0, "%ld of %ld crypto iterations wrong while being suspended / context round-tripped", job.bad, job.done_iters);
    CHECKF(job.done_iters >= before + 3, "worker made no progress after the context round trips");
    t_info("suspend_rounds", "%d suspends, %d GetThreadContext, %d SetThreadContext, %ld iterations", sus, gets, sets, job.done_iters);
}

/* ---------------------------------------------------------------- PERF (verified before reporting) */
typedef struct { const aes_sched *ks; const uint8_t *in; uint8_t *out; size_t nb; int hw; } aes_bench_ctx;
static void aes_bench_fn(void *p) {
    aes_bench_ctx *c = (aes_bench_ctx *)p;
    if (c->hw) hw_ecb_enc(c->ks, c->in, c->out, c->nb); else sw_ecb(c->ks, c->in, c->out, c->nb, 0);
    t_sink_ptr(c->out);
}
TEST(perf_aes128_ecb_software_vs_aesni) {
    NEED(F_AES); crypto_init();
    enum { NB = 1024 };
    static uint8_t in[NB * 16], o1[NB * 16], o2[NB * 16];
    uint8_t key[16]; aes_sched ks; t_stats s;
    t_rng r; t_rng_seed(&r, 801); t_rng_fill(&r, in, sizeof in); t_rng_fill(&r, key, 16);
    sw_expand(key, 128, &ks);
    aes_bench_ctx c1 = { &ks, in, o1, 64, 0 }, c2 = { &ks, in, o2, NB, 1 };
    aes_bench_fn(&c1); aes_bench_fn(&c2);
    REQUIRE(!memcmp(o1, o2, 64 * 16));
    if (t_bench(aes_bench_fn, &c1, 1, 9, 2.0, &s) > 0 && s.median > 0) t_perf("aes128_ecb_sw_MBps", 64 * 16 / s.median / 1e6, "MB/s");
    if (t_bench(aes_bench_fn, &c2, 2, 15, 2.0, &s) > 0 && s.median > 0) t_perf("aes128_ecb_aesni_MBps", NB * 16 / s.median / 1e6, "MB/s");
}
typedef struct { const crc_fold *f; const uint8_t *p; size_t n; int mode; uint32_t r; } crc_bench_ctx;
static void crc_bench_fn(void *pv) {
    crc_bench_ctx *c = (crc_bench_ctx *)pv;
    c->r = c->mode == 0 ? crc_sw(g_crc32c_tab, c->p, c->n) : c->mode == 1 ? hw_crc_fold(c->f, c->p, c->n, 1) : hw_crc32c_sse42(c->p, c->n, 8);
    T_SINK(c->r);
}
TEST(perf_crc32c_table_vs_pclmul_vs_sse42) {
    NEED(F_PCLMULQDQ); crypto_init();
    enum { N = 64 * 1024 };
    static uint8_t buf[N];
    crc_fold f; t_stats s;
    crc_fold_init(&f, 0x1EDC6F41u, g_crc32c_tab);
    t_rng r; t_rng_seed(&r, 802); t_rng_fill(&r, buf, N);
    uint32_t want = crc_sw(g_crc32c_tab, buf, N);
    REQUIRE(hw_crc_fold(&f, buf, N, 1) == want);
    static const char *const nm[] = { "crc32c_table_MBps", "crc32c_pclmul_fold4_MBps", "crc32c_sse42_MBps" };
    for (int m = 0; m < 3; m++) {
        if (m == 2 && !cpu_has(F_SSE42)) continue;
        crc_bench_ctx c = { &f, buf, N, m, 0 };
        if (t_bench(crc_bench_fn, &c, 1, 11, 2.0, &s) > 0 && s.median > 0) {
            CHECK_EQ(c.r, want);
            t_perf(nm[m], N / s.median / 1e6, "MB/s");
        }
    }
}
typedef struct { compress_fn fn; const uint8_t *p; size_t n; uint8_t d[32]; } sha_bench_ctx;
static void sha_bench_fn(void *pv) { sha_bench_ctx *c = (sha_bench_ctx *)pv; md_digest(ALG_SHA256, c->fn, c->p, c->n, c->d); t_sink_ptr(c->d); }
TEST(perf_sha256_software_vs_shani) {
    NEED(F_SHA);
    enum { N = 64 * 1024 };
    static uint8_t buf[N];
    t_stats s; uint8_t want[32];
    t_rng r; t_rng_seed(&r, 803); t_rng_fill(&r, buf, N);
    md_digest(ALG_SHA256, sw_sha256_blocks, buf, N, want);
    sha_bench_ctx c1 = { sw_sha256_blocks, buf, N, { 0 } }, c2 = { hw_sha256_blocks, buf, N, { 0 } };
    sha_bench_fn(&c2);
    REQUIRE(!memcmp(c2.d, want, 32));
    if (t_bench(sha_bench_fn, &c1, 1, 7, 2.0, &s) > 0 && s.median > 0) t_perf("sha256_sw_MBps", N / s.median / 1e6, "MB/s");
    if (t_bench(sha_bench_fn, &c2, 1, 11, 2.0, &s) > 0 && s.median > 0) t_perf("sha256_shani_MBps", N / s.median / 1e6, "MB/s");
}
