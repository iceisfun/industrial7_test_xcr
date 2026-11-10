/* simd_bench_mem.h - memcpy / memset / strlen kernels, CRC32C, AES-128, carry-less multiply for simd_bench.
 * Every kernel handles any length (heads/tails), so the correctness sweeps exercise them with odd sizes
 * and misalignments against guard bytes before any number is reported. */
#ifndef SIMD_BENCH_MEM_H
#define SIMD_BENCH_MEM_H
#include "simd_bench_util.h"

/* ============================================================================ memcpy */
typedef void (*cpy_fn)(void *d, const void *s, size_t n);
static void *(*volatile g_crt_memcpy)(void *, const void *, size_t) = memcpy;   /* real CRT call, not a builtin */
static void *(*volatile g_crt_memset)(void *, int, size_t) = memset;
static size_t (*volatile g_crt_strlen)(const char *) = strlen;

static void cpy_crt(void *d, const void *s, size_t n) { g_crt_memcpy(d, s, n); }
static BM_NOVEC void cpy_bytes(uint8_t *d, const uint8_t *s, size_t n) { for (size_t i = 0; i < n; i++) d[i] = s[i]; }
static void cpy_sse2(void *d_, const void *s_, size_t n) {
    uint8_t *d = (uint8_t *)d_; const uint8_t *s = (const uint8_t *)s_;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m128i a = _mm_loadu_si128((const __m128i *)(s + i)), b = _mm_loadu_si128((const __m128i *)(s + i + 16));
        __m128i c = _mm_loadu_si128((const __m128i *)(s + i + 32)), e = _mm_loadu_si128((const __m128i *)(s + i + 48));
        _mm_storeu_si128((__m128i *)(d + i), a); _mm_storeu_si128((__m128i *)(d + i + 16), b);
        _mm_storeu_si128((__m128i *)(d + i + 32), c); _mm_storeu_si128((__m128i *)(d + i + 48), e);
    }
    for (; i + 16 <= n; i += 16) _mm_storeu_si128((__m128i *)(d + i), _mm_loadu_si128((const __m128i *)(s + i)));
    cpy_bytes(d + i, s + i, n - i);
}
/* non-temporal stores: align the destination to 16, MOVNTDQ the body, SFENCE (weakly ordered stores) */
static void cpy_sse2_nt(void *d_, const void *s_, size_t n) {
    uint8_t *d = (uint8_t *)d_; const uint8_t *s = (const uint8_t *)s_;
    size_t head = (16 - ((uintptr_t)d & 15)) & 15;
    if (head > n) head = n;
    cpy_bytes(d, s, head); d += head; s += head; n -= head;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m128i a = _mm_loadu_si128((const __m128i *)(s + i)), b = _mm_loadu_si128((const __m128i *)(s + i + 16));
        __m128i c = _mm_loadu_si128((const __m128i *)(s + i + 32)), e = _mm_loadu_si128((const __m128i *)(s + i + 48));
        _mm_stream_si128((__m128i *)(d + i), a); _mm_stream_si128((__m128i *)(d + i + 16), b);
        _mm_stream_si128((__m128i *)(d + i + 32), c); _mm_stream_si128((__m128i *)(d + i + 48), e);
    }
    for (; i + 16 <= n; i += 16) _mm_stream_si128((__m128i *)(d + i), _mm_loadu_si128((const __m128i *)(s + i)));
    _mm_sfence();
    cpy_bytes(d + i, s + i, n - i);
}
static T_TARGET("avx") void cpy_avx(void *d_, const void *s_, size_t n) {
    uint8_t *d = (uint8_t *)d_; const uint8_t *s = (const uint8_t *)s_;
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(s + i)), b = _mm256_loadu_si256((const __m256i *)(s + i + 32));
        __m256i c = _mm256_loadu_si256((const __m256i *)(s + i + 64)), e = _mm256_loadu_si256((const __m256i *)(s + i + 96));
        _mm256_storeu_si256((__m256i *)(d + i), a); _mm256_storeu_si256((__m256i *)(d + i + 32), b);
        _mm256_storeu_si256((__m256i *)(d + i + 64), c); _mm256_storeu_si256((__m256i *)(d + i + 96), e);
    }
    for (; i + 32 <= n; i += 32) _mm256_storeu_si256((__m256i *)(d + i), _mm256_loadu_si256((const __m256i *)(s + i)));
    _mm256_zeroupper();
    for (; i < n; i++) d[i] = s[i];
}
static T_TARGET("avx512f") void cpy_avx512(void *d_, const void *s_, size_t n) {
    uint8_t *d = (uint8_t *)d_; const uint8_t *s = (const uint8_t *)s_;
    size_t i = 0;
    for (; i + 256 <= n; i += 256) {
        __m512i a = _mm512_loadu_si512(s + i), b = _mm512_loadu_si512(s + i + 64);
        __m512i c = _mm512_loadu_si512(s + i + 128), e = _mm512_loadu_si512(s + i + 192);
        _mm512_storeu_si512(d + i, a); _mm512_storeu_si512(d + i + 64, b);
        _mm512_storeu_si512(d + i + 128, c); _mm512_storeu_si512(d + i + 192, e);
    }
    for (; i + 64 <= n; i += 64) _mm512_storeu_si512(d + i, _mm512_loadu_si512(s + i));
    for (; i < n; i++) d[i] = s[i];
    _mm256_zeroupper();
}
static void cpy_rep_movsb(void *d, const void *s, size_t n) {
    __asm__ volatile("cld\n\trep movsb" : "+D"(d), "+S"(s), "+c"(n) :: "memory");
}
/* rep movsq (x64) / rep movsd (x86) for the body, rep movsb for the tail */
static void cpy_rep_movs_word(void *d, const void *s, size_t n) {
#ifdef T_X64
    size_t w = n >> 3, t = n & 7;
    __asm__ volatile("cld\n\trep movsq" : "+D"(d), "+S"(s), "+c"(w) :: "memory");
#else
    size_t w = n >> 2, t = n & 3;
    __asm__ volatile("cld\n\trep movsl" : "+D"(d), "+S"(s), "+c"(w) :: "memory");
#endif
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(t) :: "memory");
}

/* ============================================================================ memset */
typedef void (*set_fn)(void *d, int v, size_t n);
static void set_crt(void *d, int v, size_t n) { g_crt_memset(d, v, n); }
static BM_NOVEC void set_bytes(uint8_t *d, int v, size_t n) { for (size_t i = 0; i < n; i++) d[i] = (uint8_t)v; }
static void set_sse2(void *d_, int v, size_t n) {
    uint8_t *d = (uint8_t *)d_; __m128i x = _mm_set1_epi8((char)v); size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        _mm_storeu_si128((__m128i *)(d + i), x); _mm_storeu_si128((__m128i *)(d + i + 16), x);
        _mm_storeu_si128((__m128i *)(d + i + 32), x); _mm_storeu_si128((__m128i *)(d + i + 48), x);
    }
    for (; i + 16 <= n; i += 16) _mm_storeu_si128((__m128i *)(d + i), x);
    set_bytes(d + i, v, n - i);
}
static void set_sse2_nt(void *d_, int v, size_t n) {
    uint8_t *d = (uint8_t *)d_; __m128i x = _mm_set1_epi8((char)v);
    size_t head = (16 - ((uintptr_t)d & 15)) & 15;
    if (head > n) head = n;
    set_bytes(d, v, head); d += head; n -= head;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        _mm_stream_si128((__m128i *)(d + i), x); _mm_stream_si128((__m128i *)(d + i + 16), x);
        _mm_stream_si128((__m128i *)(d + i + 32), x); _mm_stream_si128((__m128i *)(d + i + 48), x);
    }
    for (; i + 16 <= n; i += 16) _mm_stream_si128((__m128i *)(d + i), x);
    _mm_sfence();
    set_bytes(d + i, v, n - i);
}
static T_TARGET("avx") void set_avx(void *d_, int v, size_t n) {
    uint8_t *d = (uint8_t *)d_; __m256i x = _mm256_set1_epi8((char)v); size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        _mm256_storeu_si256((__m256i *)(d + i), x); _mm256_storeu_si256((__m256i *)(d + i + 32), x);
        _mm256_storeu_si256((__m256i *)(d + i + 64), x); _mm256_storeu_si256((__m256i *)(d + i + 96), x);
    }
    for (; i + 32 <= n; i += 32) _mm256_storeu_si256((__m256i *)(d + i), x);
    _mm256_zeroupper();
    for (; i < n; i++) d[i] = (uint8_t)v;
}
static T_TARGET("avx512f,avx512bw") void set_avx512(void *d_, int v, size_t n) {
    uint8_t *d = (uint8_t *)d_; __m512i x = _mm512_set1_epi8((char)v); size_t i = 0;
    for (; i + 256 <= n; i += 256) {
        _mm512_storeu_si512(d + i, x); _mm512_storeu_si512(d + i + 64, x);
        _mm512_storeu_si512(d + i + 128, x); _mm512_storeu_si512(d + i + 192, x);
    }
    for (; i + 64 <= n; i += 64) _mm512_storeu_si512(d + i, x);
    if (i < n) _mm512_mask_storeu_epi8(d + i, (__mmask64)(~0ull >> (64 - (n - i))), x);   /* masked tail */
    _mm256_zeroupper();
}
static void set_rep_stosb(void *d, int v, size_t n) {
    __asm__ volatile("cld\n\trep stosb" : "+D"(d), "+c"(n) : "a"(v) : "memory");
}

/* ============================================================================ strlen */
typedef size_t (*len_fn)(const char *s);
static size_t len_crt(const char *s) { return g_crt_strlen(s); }
/* SSE2: aligned 16-byte loads only (an aligned load never crosses a page, so reading past the
 * terminator inside the same block is safe); bytes before s in the first block are masked off. */
static size_t len_sse2(const char *s) {
    uintptr_t a = (uintptr_t)s & ~(uintptr_t)15; unsigned mis = (unsigned)((uintptr_t)s & 15);
    __m128i z = _mm_setzero_si128();
    unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128((const __m128i *)a), z)) >> mis;
    if (m) return (size_t)__builtin_ctz(m);
    for (a += 16;; a += 16) {
        m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128((const __m128i *)a), z));
        if (m) return (size_t)(a - (uintptr_t)s) + (size_t)__builtin_ctz(m);
    }
}
/* SSE4.2 PCMPISTRI (implicit length): bytewise head to 16-byte alignment, then aligned blocks.
 * imm = UBYTE_OPS | CMP_EQUAL_EACH | MASKED_NEGATIVE_POLARITY: valid (pre-terminator) positions -> 0,
 * invalid -> 1, so the least significant set index is the terminator position (16 if none). */
static T_TARGET("sse4.2") size_t len_sse42(const char *s) {
    const char *p = s;
    while ((uintptr_t)p & 15) { if (!*p) return (size_t)(p - s); p++; }
    for (;; p += 16) {
        __m128i v = _mm_load_si128((const __m128i *)p);
        int i = _mm_cmpistri(v, v, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MASKED_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
        if (i < 16) return (size_t)(p - s) + (size_t)i;
    }
}
static T_TARGET("avx2") size_t len_avx2(const char *s) {
    uintptr_t a = (uintptr_t)s & ~(uintptr_t)31; unsigned mis = (unsigned)((uintptr_t)s & 31);
    __m256i z = _mm256_setzero_si256();
    unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((const __m256i *)a), z)) >> mis;
    size_t r;
    if (m) { r = (size_t)__builtin_ctz(m); goto out; }
    for (a += 32;; a += 32) {
        m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((const __m256i *)a), z));
        if (m) { r = (size_t)(a - (uintptr_t)s) + (size_t)__builtin_ctz(m); goto out; }
    }
out:
    _mm256_zeroupper();
    return r;
}

/* ============================================================================ CRC32C (Castagnoli, reflected 0x82F63B78) */
static uint32_t g_crc_tab[8][256];
static void crc_init(void) {
    if (g_crc_tab[0][1]) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1)));
        g_crc_tab[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; i++)
        for (int t = 1; t < 8; t++) g_crc_tab[t][i] = (g_crc_tab[t - 1][i] >> 8) ^ g_crc_tab[0][g_crc_tab[t - 1][i] & 0xFF];
}
typedef uint32_t (*crc_fn)(const uint8_t *p, size_t n);     /* full CRC: init ~0, final xor ~0 */
static BM_NOVEC uint32_t crc_bitwise(const uint8_t *p, size_t n) {       /* definition, table-free */
    uint32_t c = ~0u;
    for (size_t i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1))); }
    return ~c;
}
static BM_NOVEC uint32_t crc_sw_bytewise(const uint8_t *p, size_t n) {
    uint32_t c = ~0u;
    for (size_t i = 0; i < n; i++) c = (c >> 8) ^ g_crc_tab[0][(c ^ p[i]) & 0xFF];
    return ~c;
}
static BM_NOVEC uint32_t crc_sw_slice8(const uint8_t *p, size_t n) {
    uint32_t c = ~0u; size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint32_t lo, hi; memcpy(&lo, p + i, 4); memcpy(&hi, p + i + 4, 4);
        lo ^= c;
        c = g_crc_tab[7][lo & 0xFF] ^ g_crc_tab[6][(lo >> 8) & 0xFF] ^ g_crc_tab[5][(lo >> 16) & 0xFF] ^ g_crc_tab[4][lo >> 24] ^
            g_crc_tab[3][hi & 0xFF] ^ g_crc_tab[2][(hi >> 8) & 0xFF] ^ g_crc_tab[1][(hi >> 16) & 0xFF] ^ g_crc_tab[0][hi >> 24];
    }
    for (; i < n; i++) c = (c >> 8) ^ g_crc_tab[0][(c ^ p[i]) & 0xFF];
    return ~c;
}
static T_TARGET("sse4.2") uint32_t crc_sse42_u8(const uint8_t *p, size_t n) {
    uint32_t c = ~0u;
    for (size_t i = 0; i < n; i++) c = _mm_crc32_u8(c, p[i]);
    return ~c;
}
/* widest CRC32 form: CRC32 r64,m64 on x64, r32,m32 on x86 */
static T_TARGET("sse4.2") uint32_t crc_sse42_wide(const uint8_t *p, size_t n) {
    size_t i = 0;
#ifdef T_X64
    uint64_t c = ~0u;
    for (; i + 8 <= n; i += 8) { uint64_t v; memcpy(&v, p + i, 8); c = _mm_crc32_u64(c, v); }
    uint32_t c32 = (uint32_t)c;
#else
    uint32_t c32 = ~0u;
    for (; i + 4 <= n; i += 4) { uint32_t v; memcpy(&v, p + i, 4); c32 = _mm_crc32_u32(c32, v); }
#endif
    for (; i < n; i++) c32 = _mm_crc32_u8(c32, p[i]);
    return ~c32;
}
/* three independent streams (latency hiding) joined with the zero-extension operator: for raw CRC
 * registers, crc(A||B) = Z^|B| * crc(A) ^ crc_from_0(B), Z = "append one zero byte" (linear over GF(2)).
 * Z^n is built by repeated squaring (zlib crc32_combine construction, reflected Castagnoli polynomial)
 * and cached for the last n, so the timed loop pays for it once. */
static uint32_t gf2_times(const uint32_t *mat, uint32_t vec) { uint32_t s = 0; while (vec) { if (vec & 1) s ^= *mat; vec >>= 1; mat++; } return s; }
static void gf2_square(uint32_t *sq, const uint32_t *mat) { for (int n = 0; n < 32; n++) sq[n] = gf2_times(mat, mat[n]); }
static void gf2_apply(uint32_t *op, const uint32_t *m) { for (int n = 0; n < 32; n++) op[n] = gf2_times(m, op[n]); }
static void crc_zeros_op(size_t nbytes, uint32_t op[32]) {
    uint32_t odd[32], even[32], row = 1;
    for (int i = 0; i < 32; i++) op[i] = 1u << i;          /* identity */
    odd[0] = 0x82F63B78u;                                    /* one zero bit */
    for (int i = 1; i < 32; i++) { odd[i] = row; row <<= 1; }
    gf2_square(even, odd);                                   /* two zero bits */
    gf2_square(odd, even);                                   /* four zero bits */
    while (nbytes) {
        gf2_square(even, odd); if (nbytes & 1) gf2_apply(op, even); nbytes >>= 1; if (!nbytes) break;
        gf2_square(odd, even); if (nbytes & 1) gf2_apply(op, odd); nbytes >>= 1;
    }
}
static uint32_t crc_shift_zeros(uint32_t c, size_t nbytes) {
    static size_t cached_n = (size_t)-1; static uint32_t op[32];
    if (nbytes != cached_n) { crc_zeros_op(nbytes, op); cached_n = nbytes; }
    return gf2_times(op, c);
}
static T_TARGET("sse4.2") uint32_t crc_sse42_3way(const uint8_t *p, size_t n) {
#ifdef T_X64
    typedef uint64_t cw_t;
#define CRC_W(c, v) _mm_crc32_u64(c, v)
#else
    typedef uint32_t cw_t;                             /* x86: CRC32 r32, m32 streams */
#define CRC_W(c, v) _mm_crc32_u32(c, v)
#endif
    const size_t W = sizeof(cw_t);
    size_t blk = (n / (3 * W)) * W;                    /* bytes per stream, multiple of the word size */
    if (blk < 64) return crc_sse42_wide(p, n);
    cw_t c0 = ~0u, c1 = 0, c2 = 0;
    const uint8_t *p0 = p, *p1 = p + blk, *p2 = p + 2 * blk;
    for (size_t i = 0; i < blk; i += W) {
        cw_t a, b, d; memcpy(&a, p0 + i, W); memcpy(&b, p1 + i, W); memcpy(&d, p2 + i, W);
        c0 = CRC_W(c0, a); c1 = CRC_W(c1, b); c2 = CRC_W(c2, d);
    }
    /* crc(A||B) raw = shift(crcA, |B|) ^ crcB_raw(start 0) */
    uint32_t c = crc_shift_zeros((uint32_t)c0, blk) ^ (uint32_t)c1;
    c = crc_shift_zeros(c, blk) ^ (uint32_t)c2;
    size_t i = 3 * blk;
    for (; i + W <= n; i += W) { cw_t v; memcpy(&v, p + i, W); c = (uint32_t)CRC_W(c, v); }
    for (; i < n; i++) c = _mm_crc32_u8(c, p[i]);
#undef CRC_W
    return ~c;
}

/* ============================================================================ AES-128 */
static uint8_t g_sbox[256], g_inv_sbox[256];
static uint32_t g_te[4][256];
static uint8_t aes_xtime(uint8_t x) { return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1B : 0)); }
static uint8_t aes_gmul(uint8_t a, uint8_t b) { uint8_t r = 0; while (b) { if (b & 1) r ^= a; a = aes_xtime(a); b >>= 1; } return r; }
/* S-box from its definition (FIPS-197 5.1.1): multiplicative inverse in GF(2^8) then the affine map */
static void aes_init(void) {
    if (g_sbox[0] == 0x63) return;
    for (int x = 0; x < 256; x++) {
        uint8_t inv = 0;
        if (x) for (int y = 1; y < 256; y++) if (aes_gmul((uint8_t)x, (uint8_t)y) == 1) { inv = (uint8_t)y; break; }
        uint8_t s = inv, r = inv;
        for (int k = 0; k < 4; k++) { r = (uint8_t)((r << 1) | (r >> 7)); s ^= r; }
        g_sbox[x] = (uint8_t)(s ^ 0x63);
    }
    for (int x = 0; x < 256; x++) g_inv_sbox[g_sbox[x]] = (uint8_t)x;
    for (int x = 0; x < 256; x++) {
        uint8_t s = g_sbox[x], s2 = aes_xtime(s), s3 = (uint8_t)(s2 ^ s);
        uint32_t t = (uint32_t)s2 | (uint32_t)s << 8 | (uint32_t)s << 16 | (uint32_t)s3 << 24;   /* column bytes 0..3 */
        for (int r = 0; r < 4; r++) g_te[r][x] = (t << (8 * r)) | (r ? t >> (32 - 8 * r) : 0);
    }
}
typedef struct aes_key { T_ALIGN(16) uint8_t rk[11][16]; T_ALIGN(16) uint8_t dk[11][16]; } aes_key;
static void aes_expand_sw(aes_key *k, const uint8_t key[16]) {
    static const uint8_t rcon[10] = { 1, 2, 4, 8, 16, 32, 64, 128, 0x1B, 0x36 };
    uint8_t w[176]; memcpy(w, key, 16);
    for (int i = 4; i < 44; i++) {
        uint8_t t[4]; memcpy(t, w + 4 * (i - 1), 4);
        if (i % 4 == 0) {
            uint8_t u = t[0]; t[0] = (uint8_t)(g_sbox[t[1]] ^ rcon[i / 4 - 1]); t[1] = g_sbox[t[2]]; t[2] = g_sbox[t[3]]; t[3] = g_sbox[u];
        }
        for (int j = 0; j < 4; j++) w[4 * i + j] = (uint8_t)(w[4 * (i - 4) + j] ^ t[j]);
    }
    memcpy(k->rk, w, 176);
}
/* byte-oriented reference (FIPS-197 5.1): SubBytes, ShiftRows, MixColumns, AddRoundKey; state column-major */
static BM_NOVEC void aes_enc_sw_bytewise(const aes_key *k, const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16], t[16];
    for (int i = 0; i < 16; i++) s[i] = (uint8_t)(in[i] ^ k->rk[0][i]);
    for (int r = 1; r <= 10; r++) {
        for (int c = 0; c < 4; c++) for (int row = 0; row < 4; row++) t[4 * c + row] = g_sbox[s[4 * ((c + row) & 3) + row]];
        if (r != 10)
            for (int c = 0; c < 4; c++) {
                uint8_t a0 = t[4 * c], a1 = t[4 * c + 1], a2 = t[4 * c + 2], a3 = t[4 * c + 3], x = (uint8_t)(a0 ^ a1 ^ a2 ^ a3);
                t[4 * c]     = (uint8_t)(a0 ^ x ^ aes_xtime((uint8_t)(a0 ^ a1)));
                t[4 * c + 1] = (uint8_t)(a1 ^ x ^ aes_xtime((uint8_t)(a1 ^ a2)));
                t[4 * c + 2] = (uint8_t)(a2 ^ x ^ aes_xtime((uint8_t)(a2 ^ a3)));
                t[4 * c + 3] = (uint8_t)(a3 ^ x ^ aes_xtime((uint8_t)(a3 ^ a0)));
            }
        for (int i = 0; i < 16; i++) s[i] = (uint8_t)(t[i] ^ k->rk[r][i]);
    }
    memcpy(out, s, 16);
}
static BM_NOVEC void aes_dec_sw_bytewise(const aes_key *k, const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16], t[16];
    for (int i = 0; i < 16; i++) s[i] = (uint8_t)(in[i] ^ k->rk[10][i]);
    for (int r = 9; r >= 0; r--) {
        for (int c = 0; c < 4; c++) for (int row = 0; row < 4; row++) t[4 * ((c + row) & 3) + row] = g_inv_sbox[s[4 * c + row]];
        for (int i = 0; i < 16; i++) t[i] ^= k->rk[r][i];
        if (r)
            for (int c = 0; c < 4; c++) {
                uint8_t a0 = t[4 * c], a1 = t[4 * c + 1], a2 = t[4 * c + 2], a3 = t[4 * c + 3];
                t[4 * c]     = (uint8_t)(aes_gmul(a0, 14) ^ aes_gmul(a1, 11) ^ aes_gmul(a2, 13) ^ aes_gmul(a3, 9));
                t[4 * c + 1] = (uint8_t)(aes_gmul(a0, 9) ^ aes_gmul(a1, 14) ^ aes_gmul(a2, 11) ^ aes_gmul(a3, 13));
                t[4 * c + 2] = (uint8_t)(aes_gmul(a0, 13) ^ aes_gmul(a1, 9) ^ aes_gmul(a2, 14) ^ aes_gmul(a3, 11));
                t[4 * c + 3] = (uint8_t)(aes_gmul(a0, 11) ^ aes_gmul(a1, 13) ^ aes_gmul(a2, 9) ^ aes_gmul(a3, 14));
            }
        memcpy(s, t, 16);
    }
    memcpy(out, s, 16);
}
/* 32-bit T-table implementation (little-endian column words) */
static BM_NOVEC void aes_enc_sw_ttable(const aes_key *k, const uint8_t in[16], uint8_t out[16]) {
    uint32_t s[4], t[4], rk[4];
    memcpy(s, in, 16); memcpy(rk, k->rk[0], 16);
    for (int i = 0; i < 4; i++) s[i] ^= rk[i];
    for (int r = 1; r < 10; r++) {
        memcpy(rk, k->rk[r], 16);
        for (int c = 0; c < 4; c++)
            t[c] = g_te[0][s[c] & 0xFF] ^ g_te[1][(s[(c + 1) & 3] >> 8) & 0xFF] ^ g_te[2][(s[(c + 2) & 3] >> 16) & 0xFF] ^ g_te[3][s[(c + 3) & 3] >> 24] ^ rk[c];
        memcpy(s, t, 16);
    }
    memcpy(rk, k->rk[10], 16);
    for (int c = 0; c < 4; c++)
        t[c] = ((uint32_t)g_sbox[s[c] & 0xFF] | (uint32_t)g_sbox[(s[(c + 1) & 3] >> 8) & 0xFF] << 8 |
                (uint32_t)g_sbox[(s[(c + 2) & 3] >> 16) & 0xFF] << 16 | (uint32_t)g_sbox[s[(c + 3) & 3] >> 24] << 24) ^ rk[c];
    memcpy(out, t, 16);
}
/* AES-NI key schedule (Intel AES-NI white paper, AES_128_Key_Expansion) */
#define AESNI_EXP(k, rcon) do { __m128i t_ = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(k, rcon), 0xFF); \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4)); k = _mm_xor_si128(k, _mm_slli_si128(k, 4)); \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4)); k = _mm_xor_si128(k, t_); } while (0)
static T_TARGET("aes,sse2") void aes_expand_ni(aes_key *k, const uint8_t key[16]) {
    __m128i x = _mm_loadu_si128((const __m128i *)key);
    _mm_store_si128((__m128i *)k->rk[0], x);
    AESNI_EXP(x, 0x01); _mm_store_si128((__m128i *)k->rk[1], x);
    AESNI_EXP(x, 0x02); _mm_store_si128((__m128i *)k->rk[2], x);
    AESNI_EXP(x, 0x04); _mm_store_si128((__m128i *)k->rk[3], x);
    AESNI_EXP(x, 0x08); _mm_store_si128((__m128i *)k->rk[4], x);
    AESNI_EXP(x, 0x10); _mm_store_si128((__m128i *)k->rk[5], x);
    AESNI_EXP(x, 0x20); _mm_store_si128((__m128i *)k->rk[6], x);
    AESNI_EXP(x, 0x40); _mm_store_si128((__m128i *)k->rk[7], x);
    AESNI_EXP(x, 0x80); _mm_store_si128((__m128i *)k->rk[8], x);
    AESNI_EXP(x, 0x1B); _mm_store_si128((__m128i *)k->rk[9], x);
    AESNI_EXP(x, 0x36); _mm_store_si128((__m128i *)k->rk[10], x);
    /* equivalent inverse cipher keys: dk[0] = rk[10], dk[i] = InvMixColumns(rk[10-i]), dk[10] = rk[0] */
    _mm_store_si128((__m128i *)k->dk[0], _mm_load_si128((const __m128i *)k->rk[10]));
    for (int i = 1; i < 10; i++) _mm_store_si128((__m128i *)k->dk[i], _mm_aesimc_si128(_mm_load_si128((const __m128i *)k->rk[10 - i])));
    _mm_store_si128((__m128i *)k->dk[10], _mm_load_si128((const __m128i *)k->rk[0]));
}
typedef void (*aes_buf_fn)(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nblocks);
static void aes_ecb_sw_bytewise(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) { for (size_t i = 0; i < nb; i++) aes_enc_sw_bytewise(k, in + 16 * i, out + 16 * i); }
static void aes_ecb_sw_ttable(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) { for (size_t i = 0; i < nb; i++) aes_enc_sw_ttable(k, in + 16 * i, out + 16 * i); }
static T_TARGET("aes,sse2") void aes_ecb_ni_1way(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    __m128i r[11]; for (int i = 0; i < 11; i++) r[i] = _mm_load_si128((const __m128i *)k->rk[i]);
    for (size_t b = 0; b < nb; b++) {
        __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in + 16 * b)), r[0]);
        for (int i = 1; i < 10; i++) x = _mm_aesenc_si128(x, r[i]);
        _mm_storeu_si128((__m128i *)(out + 16 * b), _mm_aesenclast_si128(x, r[10]));
    }
}
#define AES_X8(OP, k) do { x0 = OP(x0, k); x1 = OP(x1, k); x2 = OP(x2, k); x3 = OP(x3, k); \
                             x4 = OP(x4, k); x5 = OP(x5, k); x6 = OP(x6, k); x7 = OP(x7, k); } while (0)
#define AES_LD8(src, k0) const __m128i *ip_ = (const __m128i *)(src); \
    __m128i x0 = _mm_xor_si128(_mm_loadu_si128(ip_ + 0), k0), x1 = _mm_xor_si128(_mm_loadu_si128(ip_ + 1), k0); \
    __m128i x2 = _mm_xor_si128(_mm_loadu_si128(ip_ + 2), k0), x3 = _mm_xor_si128(_mm_loadu_si128(ip_ + 3), k0); \
    __m128i x4 = _mm_xor_si128(_mm_loadu_si128(ip_ + 4), k0), x5 = _mm_xor_si128(_mm_loadu_si128(ip_ + 5), k0); \
    __m128i x6 = _mm_xor_si128(_mm_loadu_si128(ip_ + 6), k0), x7 = _mm_xor_si128(_mm_loadu_si128(ip_ + 7), k0)
#define AES_ST8(dst) do { __m128i *op_ = (__m128i *)(dst); \
    _mm_storeu_si128(op_ + 0, x0); _mm_storeu_si128(op_ + 1, x1); _mm_storeu_si128(op_ + 2, x2); _mm_storeu_si128(op_ + 3, x3); \
    _mm_storeu_si128(op_ + 4, x4); _mm_storeu_si128(op_ + 5, x5); _mm_storeu_si128(op_ + 6, x6); _mm_storeu_si128(op_ + 7, x7); } while (0)
/* eight independent blocks in flight (AESENC latency hidden); round keys loaded per round */
static T_TARGET("aes,sse2") void aes_ecb_ni_8way(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    const __m128i *rk = (const __m128i *)k->rk;
    size_t b = 0;
    for (; b + 8 <= nb; b += 8) {
        AES_LD8(in + 16 * b, _mm_load_si128(rk));
        for (int i = 1; i < 10; i++) { __m128i ki = _mm_load_si128(rk + i); AES_X8(_mm_aesenc_si128, ki); }
        __m128i kl = _mm_load_si128(rk + 10); AES_X8(_mm_aesenclast_si128, kl);
        AES_ST8(out + 16 * b);
    }
    if (b < nb) aes_ecb_ni_1way(k, in + 16 * b, out + 16 * b, nb - b);
}
static T_TARGET("aes,sse2") void aes_ecb_ni_dec_1way(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    const __m128i *dk = (const __m128i *)k->dk;
    for (size_t b = 0; b < nb; b++) {
        __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in + 16 * b)), _mm_load_si128(dk));
        for (int i = 1; i < 10; i++) x = _mm_aesdec_si128(x, _mm_load_si128(dk + i));
        _mm_storeu_si128((__m128i *)(out + 16 * b), _mm_aesdeclast_si128(x, _mm_load_si128(dk + 10)));
    }
}
static T_TARGET("aes,sse2") void aes_ecb_ni_dec_8way(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    const __m128i *dk = (const __m128i *)k->dk;
    size_t b = 0;
    for (; b + 8 <= nb; b += 8) {
        AES_LD8(in + 16 * b, _mm_load_si128(dk));
        for (int i = 1; i < 10; i++) { __m128i ki = _mm_load_si128(dk + i); AES_X8(_mm_aesdec_si128, ki); }
        __m128i kl = _mm_load_si128(dk + 10); AES_X8(_mm_aesdeclast_si128, kl);
        AES_ST8(out + 16 * b);
    }
    if (b < nb) aes_ecb_ni_dec_1way(k, in + 16 * b, out + 16 * b, nb - b);
}
/* CBC encryption: every block depends on the previous ciphertext (latency-bound), IV = 0 */
static void aes_cbc_sw_ttable(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    uint8_t iv[16] = { 0 }, x[16];
    for (size_t b = 0; b < nb; b++) {
        for (int i = 0; i < 16; i++) x[i] = (uint8_t)(in[16 * b + i] ^ iv[i]);
        aes_enc_sw_ttable(k, x, out + 16 * b); memcpy(iv, out + 16 * b, 16);
    }
}
static T_TARGET("aes,sse2") void aes_cbc_ni(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    __m128i r[11]; for (int i = 0; i < 11; i++) r[i] = _mm_load_si128((const __m128i *)k->rk[i]);
    __m128i iv = _mm_setzero_si128();
    for (size_t b = 0; b < nb; b++) {
        __m128i x = _mm_xor_si128(_mm_xor_si128(_mm_loadu_si128((const __m128i *)(in + 16 * b)), iv), r[0]);
        for (int i = 1; i < 10; i++) x = _mm_aesenc_si128(x, r[i]);
        iv = _mm_aesenclast_si128(x, r[10]);
        _mm_storeu_si128((__m128i *)(out + 16 * b), iv);
    }
}
/* VAES on YMM: two blocks per register, 4 registers (8 blocks) per iteration */
static T_TARGET("avx2,vaes") void aes_ecb_vaes_ymm(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    __m256i r[11]; for (int i = 0; i < 11; i++) r[i] = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)k->rk[i]));
    size_t b = 0;
    for (; b + 8 <= nb; b += 8) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(in + 16 * b)), r[0]);
        __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(in + 16 * b + 32)), r[0]);
        __m256i x2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(in + 16 * b + 64)), r[0]);
        __m256i x3 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(in + 16 * b + 96)), r[0]);
        for (int i = 1; i < 10; i++) {
            x0 = _mm256_aesenc_epi128(x0, r[i]); x1 = _mm256_aesenc_epi128(x1, r[i]);
            x2 = _mm256_aesenc_epi128(x2, r[i]); x3 = _mm256_aesenc_epi128(x3, r[i]);
        }
        _mm256_storeu_si256((__m256i *)(out + 16 * b), _mm256_aesenclast_epi128(x0, r[10]));
        _mm256_storeu_si256((__m256i *)(out + 16 * b + 32), _mm256_aesenclast_epi128(x1, r[10]));
        _mm256_storeu_si256((__m256i *)(out + 16 * b + 64), _mm256_aesenclast_epi128(x2, r[10]));
        _mm256_storeu_si256((__m256i *)(out + 16 * b + 96), _mm256_aesenclast_epi128(x3, r[10]));
    }
    _mm256_zeroupper();
    if (b < nb) aes_ecb_sw_ttable(k, in + 16 * b, out + 16 * b, nb - b);
}
/* VAES on ZMM: four blocks per register */
static T_TARGET("avx512f,vaes") void aes_ecb_vaes_zmm(const aes_key *k, const uint8_t *in, uint8_t *out, size_t nb) {
    __m512i r[11]; for (int i = 0; i < 11; i++) r[i] = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)k->rk[i]));
    size_t b = 0;
    for (; b + 16 <= nb; b += 16) {
        __m512i x0 = _mm512_xor_si512(_mm512_loadu_si512(in + 16 * b), r[0]);
        __m512i x1 = _mm512_xor_si512(_mm512_loadu_si512(in + 16 * b + 64), r[0]);
        __m512i x2 = _mm512_xor_si512(_mm512_loadu_si512(in + 16 * b + 128), r[0]);
        __m512i x3 = _mm512_xor_si512(_mm512_loadu_si512(in + 16 * b + 192), r[0]);
        for (int i = 1; i < 10; i++) {
            x0 = _mm512_aesenc_epi128(x0, r[i]); x1 = _mm512_aesenc_epi128(x1, r[i]);
            x2 = _mm512_aesenc_epi128(x2, r[i]); x3 = _mm512_aesenc_epi128(x3, r[i]);
        }
        _mm512_storeu_si512(out + 16 * b, _mm512_aesenclast_epi128(x0, r[10]));
        _mm512_storeu_si512(out + 16 * b + 64, _mm512_aesenclast_epi128(x1, r[10]));
        _mm512_storeu_si512(out + 16 * b + 128, _mm512_aesenclast_epi128(x2, r[10]));
        _mm512_storeu_si512(out + 16 * b + 192, _mm512_aesenclast_epi128(x3, r[10]));
    }
    _mm256_zeroupper();
    if (b < nb) aes_ecb_sw_ttable(k, in + 16 * b, out + 16 * b, nb - b);
}

/* ============================================================================ carry-less multiply 64x64 -> 128 */
typedef void (*clmul_fn)(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n);   /* out[2i], out[2i+1] */
static BM_NOVEC void clmul_sw(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint64_t x = a[i], y = b[i], lo = 0, hi = 0;
        for (int k = 0; k < 64; k++)
            if ((y >> k) & 1) { lo ^= x << k; if (k) hi ^= x >> (64 - k); }
        out[2 * i] = lo; out[2 * i + 1] = hi;
    }
}
static T_TARGET("pclmul,sse2") void clmul_pclmul(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n) {
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128i x = _mm_loadu_si128((const __m128i *)(a + i)), y = _mm_loadu_si128((const __m128i *)(b + i));
        _mm_storeu_si128((__m128i *)(out + 2 * i), _mm_clmulepi64_si128(x, y, 0x00));
        _mm_storeu_si128((__m128i *)(out + 2 * i + 2), _mm_clmulepi64_si128(x, y, 0x11));
    }
    for (; i < n; i++) {
        __m128i x = _mm_loadl_epi64((const __m128i *)(a + i)), y = _mm_loadl_epi64((const __m128i *)(b + i));
        _mm_storeu_si128((__m128i *)(out + 2 * i), _mm_clmulepi64_si128(x, y, 0x00));
    }
}
#endif
