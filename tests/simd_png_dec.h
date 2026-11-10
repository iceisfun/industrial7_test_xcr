/* simd_png_dec.h - self-contained zlib inflate + PNG decoder with scalar / SSE2 / SSSE3 / AVX2 kernels.
 *
 * Inflate (RFC 1950 zlib wrapper, RFC 1951 deflate): stored, fixed and dynamic Huffman blocks, decoded
 * into one caller-sized output buffer (no window copy). Validation follows zlib's inflate:
 *  - header: CM 8, CINFO <= 7, FCHECK, FDICT 0 (PNG forbids a preset dictionary);
 *  - dynamic headers: HLIT <= 286, HDIST <= 30, complete code-length code, repeat 16 needs a previous
 *    length, repeats may not overrun HLIT+HDIST, symbol 256 must have a code, over-subscribed codes are
 *    rejected, incomplete literal/length or distance codes are rejected unless the code is a single
 *    length-1 code; an all-zero distance code is allowed (any distance symbol is then invalid);
 *  - literal/length symbols 286/287 and distance symbols 30/31 are invalid; distance <= bytes produced;
 *  - Adler-32 trailer is checked.
 * Input that ends early: everything that can be decoded from complete symbols is produced (stored blocks
 * copy what is present), then PD_ZE_IN. This is the same output zlib's inflate produces for a prefix.
 *
 * PNG (ISO/IEC 15948, W3C PNG 2nd ed.): signature, chunk lengths <= 2^31-1, chunk type letters, CRC-32
 * of every chunk (type + data), IHDR first with a legal colour type / bit depth / method combination,
 * PLTE (1..256 entries, <= 2^depth for colour type 3, forbidden for grey types), tRNS for types 0/2/3,
 * IDAT data concatenated across chunks (zero-length chunks allowed), unknown ancillary chunks skipped,
 * unknown critical chunks rejected, IEND required. Filter types 0..4, Adam7 interlace.
 *
 * Output: always RGBA. Bit depth <= 8: 8 bits per channel (grey 1/2/4 bit scaled by 255/(2^d-1),
 * i.e. bit replication); 16-bit images: 16 bits per channel in host (little-endian) order. Alpha from
 * the alpha channel, from tRNS (palette alpha; grey/RGB key compared against the original sample
 * value: key match -> 0, otherwise max), else opaque. Palette indices >= the PLTE entry count give
 * opaque black (0,0,0,max). No gamma / colour management is applied.
 *
 * Kernel sets (pd_kernels): unfilter (Sub/Up/Avg/Paeth, 1..8 bytes per pixel), row expansion to RGBA,
 * Adler-32, CRC-32 and the LZ77 match copy. Every kernel set must produce bit-identical results; each
 * falls back to scalar code for the shapes it does not vectorize (documented per kernel). */
#ifndef SIMD_PNG_DEC_H
#define SIMD_PNG_DEC_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

#define PD_TGT(s) __attribute__((target(s)))

/* ---------------------------------------------------------------------------------------------- */
/* status codes                                                                                    */
/* ---------------------------------------------------------------------------------------------- */
enum {
    PD_OK = 0,
    /* inflate */
    PD_ZE_IN = -1,       /* input ended before the stream (incl. Adler trailer) was complete */
    PD_ZE_HDR = -2,      /* bad zlib header */
    PD_ZE_BTYPE = -3,    /* block type 3 */
    PD_ZE_STORED = -4,   /* stored LEN != ~NLEN */
    PD_ZE_CODES = -5,    /* invalid dynamic code description */
    PD_ZE_SYM = -6,      /* invalid literal/length or distance symbol / code */
    PD_ZE_DIST = -7,     /* distance beyond the start of the output */
    PD_ZE_OUT = -8,      /* output larger than the buffer */
    PD_ZE_ADLER = -9,    /* Adler-32 mismatch */
    /* PNG */
    PD_E_ARG = -20, PD_E_SIG = -21, PD_E_CHUNK = -22, PD_E_CRC = -23, PD_E_IHDR = -24, PD_E_PLTE = -25,
    PD_E_LIMIT = -26, PD_E_NOMEM = -27, PD_E_TRUNC = -28, PD_E_ORDER = -29, PD_E_FILTER = -30,
    PD_E_ZSHORT = -31,   /* zlib stream complete but the image needs more data */
    PD_E_ZLONG = -32,    /* zlib stream holds more data than the image needs */
    PD_E_CRITICAL = -33, /* unknown critical chunk */
    PD_E_NODATA = -34,   /* IEND without IDAT */
    /* PD_E_ZLIB_BASE + inflate code: zlib-level error inside IDAT */
    PD_E_ZLIB_BASE = -100,
};
static const char *pd_strerror(int rc) {
    switch (rc) {
    case PD_OK: return "ok";
    case PD_ZE_IN: return "inflate: input truncated"; case PD_ZE_HDR: return "inflate: bad zlib header";
    case PD_ZE_BTYPE: return "inflate: block type 3"; case PD_ZE_STORED: return "inflate: stored LEN/NLEN";
    case PD_ZE_CODES: return "inflate: bad code description"; case PD_ZE_SYM: return "inflate: invalid symbol";
    case PD_ZE_DIST: return "inflate: distance too far back"; case PD_ZE_OUT: return "inflate: output overflow";
    case PD_ZE_ADLER: return "inflate: adler32 mismatch";
    case PD_E_ARG: return "bad argument"; case PD_E_SIG: return "bad signature"; case PD_E_CHUNK: return "bad chunk";
    case PD_E_CRC: return "chunk CRC mismatch"; case PD_E_IHDR: return "bad IHDR"; case PD_E_PLTE: return "bad PLTE";
    case PD_E_LIMIT: return "image too large"; case PD_E_NOMEM: return "out of memory"; case PD_E_TRUNC: return "file truncated";
    case PD_E_ORDER: return "chunk order"; case PD_E_FILTER: return "bad filter type";
    case PD_E_ZSHORT: return "not enough image data"; case PD_E_ZLONG: return "too much image data";
    case PD_E_CRITICAL: return "unknown critical chunk"; case PD_E_NODATA: return "no IDAT";
    default: break;
    }
    if (rc <= PD_E_ZLIB_BASE && rc > PD_E_ZLIB_BASE - 20) return "zlib error in IDAT";
    return "?";
}

/* ---------------------------------------------------------------------------------------------- */
/* format description for row expansion                                                            */
/* ---------------------------------------------------------------------------------------------- */
typedef struct pd_fmt {
    int ctype, depth;           /* PNG colour type 0/2/3/4/6, bit depth */
    int has_trns;               /* grey/RGB key present (palette alpha lives in pal[]) */
    uint16_t key[3];            /* grey key in key[0]; RGB key */
    uint32_t pal[256];          /* palette as RGBA bytes in memory order (little-endian u32 r|g<<8|b<<16|a<<24) */
} pd_fmt;

typedef struct pd_kernels {
    const char *name;
    /* ft 1..4 (0 is a no-op handled by the caller); cur/prev hold n bytes; bpp 1..8 */
    void (*unfilter)(int ft, uint8_t *cur, const uint8_t *prev, size_t n, int bpp);
    /* row of w pixels (exactly the PNG row bytes, no slack) -> w RGBA pixels (4 or 8 bytes each) */
    void (*expand)(const pd_fmt *f, const uint8_t *row, uint32_t w, uint8_t *out);
    uint32_t (*adler32)(uint32_t adler, const uint8_t *p, size_t n);
    uint32_t (*crc32)(uint32_t crc, const uint8_t *p, size_t n);     /* zlib convention: crc32(0, ..) */
    /* out[pos .. pos+len) = out[pos-dist .. ) byte by byte (overlap repeats); never writes past pos+len */
    void (*copy_match)(uint8_t *out, size_t pos, size_t dist, size_t len);
} pd_kernels;

static uint64_t pd_fnv64(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

/* ============================================================================================== */
/* checksums                                                                                       */
/* ============================================================================================== */
#define PD_ADLER_BASE 65521u
#define PD_ADLER_NMAX 5552u
static uint32_t pd_adler_scalar(uint32_t adler, const uint8_t *p, size_t n) {
    uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    while (n) {
        size_t k = n < PD_ADLER_NMAX ? n : PD_ADLER_NMAX;
        n -= k;
        while (k--) { s1 += *p++; s2 += s1; }
        s1 %= PD_ADLER_BASE; s2 %= PD_ADLER_BASE;
    }
    return s1 | (s2 << 16);
}
/* block-sum identity used by all vector versions (B = block bytes, b_i the bytes of block k):
 *   s2 += B*s1_before_k + sum_i (B - i) * b_i,   s1 += sum_i b_i
 * Over m blocks: s2 += m*B*s1_0 + B*sum_k S1_k + sum_k W_k, with S1_k the partial byte sum before block k. */
static uint32_t pd_adler_finish(uint32_t s1, uint32_t s2, uint64_t blocks, uint64_t B, uint64_t ps, uint64_t bsum, uint64_t wsum) {
    uint64_t t2 = (uint64_t)s2 + (blocks * B % PD_ADLER_BASE) * s1 + B * (ps % PD_ADLER_BASE) + wsum;
    uint64_t t1 = (uint64_t)s1 + bsum;
    return (uint32_t)(t1 % PD_ADLER_BASE) | ((uint32_t)(t2 % PD_ADLER_BASE) << 16);
}
static uint32_t pd_hsum32_sse2(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(v);
}
static uint32_t pd_adler_sse2(uint32_t adler, const uint8_t *p, size_t n) {
    uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    const __m128i z = _mm_setzero_si128();
    const __m128i whi = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16), wlo = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
    while (n >= 16) {
        size_t blocks = n / 16;
        if (blocks > PD_ADLER_NMAX / 16) blocks = PD_ADLER_NMAX / 16;
        __m128i vps = z, vs1 = z, vs2 = z;
        for (size_t b = 0; b < blocks; b++, p += 16) {
            __m128i x = _mm_loadu_si128((const __m128i *)p);
            vps = _mm_add_epi32(vps, vs1);
            vs1 = _mm_add_epi32(vs1, _mm_sad_epu8(x, z));
            vs2 = _mm_add_epi32(vs2, _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi8(x, z), whi), _mm_madd_epi16(_mm_unpackhi_epi8(x, z), wlo)));
        }
        uint32_t r = pd_adler_finish(s1, s2, blocks, 16, pd_hsum32_sse2(vps), pd_hsum32_sse2(vs1), pd_hsum32_sse2(vs2));
        s1 = r & 0xFFFF; s2 = r >> 16;
        n -= blocks * 16;
    }
    return pd_adler_scalar(s1 | (s2 << 16), p, n);
}
PD_TGT("ssse3") static uint32_t pd_adler_ssse3(uint32_t adler, const uint8_t *p, size_t n) {
    uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    const __m128i z = _mm_setzero_si128(), ones = _mm_set1_epi16(1);
    const __m128i w = _mm_setr_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    while (n >= 16) {
        size_t blocks = n / 16;
        if (blocks > PD_ADLER_NMAX / 16) blocks = PD_ADLER_NMAX / 16;
        __m128i vps = z, vs1 = z, vs2 = z;
        for (size_t b = 0; b < blocks; b++, p += 16) {
            __m128i x = _mm_loadu_si128((const __m128i *)p);
            vps = _mm_add_epi32(vps, vs1);
            vs1 = _mm_add_epi32(vs1, _mm_sad_epu8(x, z));
            vs2 = _mm_add_epi32(vs2, _mm_madd_epi16(_mm_maddubs_epi16(x, w), ones));
        }
        __m128i t;
        uint32_t a[3];
        t = vps; t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E)); t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1)); a[0] = (uint32_t)_mm_cvtsi128_si32(t);
        t = vs1; t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E)); t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1)); a[1] = (uint32_t)_mm_cvtsi128_si32(t);
        t = vs2; t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E)); t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1)); a[2] = (uint32_t)_mm_cvtsi128_si32(t);
        uint32_t r = pd_adler_finish(s1, s2, blocks, 16, a[0], a[1], a[2]);
        s1 = r & 0xFFFF; s2 = r >> 16;
        n -= blocks * 16;
    }
    return pd_adler_scalar(s1 | (s2 << 16), p, n);
}
PD_TGT("avx2") static uint32_t pd_hsum32_avx2(__m256i v) {
    __m128i t = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
    return (uint32_t)_mm_cvtsi128_si32(t);
}
PD_TGT("avx2") static uint32_t pd_adler_avx2(uint32_t adler, const uint8_t *p, size_t n) {
    uint32_t s1 = adler & 0xFFFF, s2 = adler >> 16;
    const __m256i z = _mm256_setzero_si256(), ones = _mm256_set1_epi16(1);
    const __m256i w = _mm256_setr_epi8(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                                       16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    while (n >= 32) {
        size_t blocks = n / 32;
        if (blocks > PD_ADLER_NMAX / 32) blocks = PD_ADLER_NMAX / 32;
        __m256i vps = z, vs1 = z, vs2 = z;
        for (size_t b = 0; b < blocks; b++, p += 32) {
            __m256i x = _mm256_loadu_si256((const __m256i *)p);
            vps = _mm256_add_epi32(vps, vs1);
            vs1 = _mm256_add_epi32(vs1, _mm256_sad_epu8(x, z));
            vs2 = _mm256_add_epi32(vs2, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones));
        }
        uint32_t r = pd_adler_finish(s1, s2, blocks, 32, pd_hsum32_avx2(vps), pd_hsum32_avx2(vs1), pd_hsum32_avx2(vs2));
        s1 = r & 0xFFFF; s2 = r >> 16;
        n -= blocks * 32;
    }
    return pd_adler_sse2(s1 | (s2 << 16), p, n);   /* < 32 bytes left; legacy-SSE tail after VEX code is fine: */
}                                                   /* the AVX2 kernels end with vzeroupper (compiler-emitted) */

/* CRC-32 (ISO-HDLC, reflected 0xEDB88320) */
static uint32_t pd_crc_tab[8][256];
static void __attribute__((constructor)) pd_crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        pd_crc_tab[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; i++)
        for (int t = 1; t < 8; t++) pd_crc_tab[t][i] = (pd_crc_tab[t - 1][i] >> 8) ^ pd_crc_tab[0][pd_crc_tab[t - 1][i] & 0xFF];
}
static uint32_t pd_crc_bytewise(uint32_t crc, const uint8_t *p, size_t n) {
    uint32_t c = ~crc;
    while (n--) c = pd_crc_tab[0][(c ^ *p++) & 0xFF] ^ (c >> 8);
    return ~c;
}
static uint32_t pd_crc_slice8(uint32_t crc, const uint8_t *p, size_t n) {
    uint32_t c = ~crc;
    while (n && ((uintptr_t)p & 7)) { c = pd_crc_tab[0][(c ^ *p++) & 0xFF] ^ (c >> 8); n--; }
    while (n >= 8) {
        uint32_t lo, hi;
        memcpy(&lo, p, 4); memcpy(&hi, p + 4, 4);
        lo ^= c;
        c = pd_crc_tab[7][lo & 0xFF] ^ pd_crc_tab[6][(lo >> 8) & 0xFF] ^ pd_crc_tab[5][(lo >> 16) & 0xFF] ^ pd_crc_tab[4][lo >> 24] ^
            pd_crc_tab[3][hi & 0xFF] ^ pd_crc_tab[2][(hi >> 8) & 0xFF] ^ pd_crc_tab[1][(hi >> 16) & 0xFF] ^ pd_crc_tab[0][hi >> 24];
        p += 8; n -= 8;
    }
    while (n--) c = pd_crc_tab[0][(c ^ *p++) & 0xFF] ^ (c >> 8);
    return ~c;
}
/* PCLMULQDQ folding (Intel, "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction",
 * reflected form; same constants as zlib-chromium crc32_simd.c). Folds 64-byte blocks; the caller passes
 * len >= 64, len % 16 == 0 and handles the rest. Uses only SSE2 + PCLMULQDQ (no SSE4.1 PEXTRD). */
static const uint64_t pd_k1k2[2] __attribute__((aligned(16))) = { 0x0154442bd4ull, 0x01c6e41596ull };
static const uint64_t pd_k3k4[2] __attribute__((aligned(16))) = { 0x01751997d0ull, 0x00ccaa009eull };
static const uint64_t pd_k5k0[2] __attribute__((aligned(16))) = { 0x0163cd6124ull, 0 };
static const uint64_t pd_poly[2] __attribute__((aligned(16))) = { 0x01db710641ull, 0x01f7011641ull };
PD_TGT("pclmul") static uint32_t pd_crc_fold_pclmul(const uint8_t *buf, size_t len, uint32_t crc /* pre-inverted */) {
    __m128i x0, x1, x2, x3, x4, x5, x6, x7, x8;
    x1 = _mm_loadu_si128((const __m128i *)(buf + 0x00));
    x2 = _mm_loadu_si128((const __m128i *)(buf + 0x10));
    x3 = _mm_loadu_si128((const __m128i *)(buf + 0x20));
    x4 = _mm_loadu_si128((const __m128i *)(buf + 0x30));
    x1 = _mm_xor_si128(x1, _mm_cvtsi32_si128((int)crc));
    x0 = _mm_load_si128((const __m128i *)pd_k1k2);
    buf += 64; len -= 64;
    while (len >= 64) {
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00); x6 = _mm_clmulepi64_si128(x2, x0, 0x00);
        x7 = _mm_clmulepi64_si128(x3, x0, 0x00); x8 = _mm_clmulepi64_si128(x4, x0, 0x00);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x11); x2 = _mm_clmulepi64_si128(x2, x0, 0x11);
        x3 = _mm_clmulepi64_si128(x3, x0, 0x11); x4 = _mm_clmulepi64_si128(x4, x0, 0x11);
        x1 = _mm_xor_si128(_mm_xor_si128(x1, x5), _mm_loadu_si128((const __m128i *)(buf + 0x00)));
        x2 = _mm_xor_si128(_mm_xor_si128(x2, x6), _mm_loadu_si128((const __m128i *)(buf + 0x10)));
        x3 = _mm_xor_si128(_mm_xor_si128(x3, x7), _mm_loadu_si128((const __m128i *)(buf + 0x20)));
        x4 = _mm_xor_si128(_mm_xor_si128(x4, x8), _mm_loadu_si128((const __m128i *)(buf + 0x30)));
        buf += 64; len -= 64;
    }
    x0 = _mm_load_si128((const __m128i *)pd_k3k4);
    x5 = _mm_clmulepi64_si128(x1, x0, 0x00); x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(_mm_xor_si128(x1, x2), x5);
    x5 = _mm_clmulepi64_si128(x1, x0, 0x00); x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(_mm_xor_si128(x1, x3), x5);
    x5 = _mm_clmulepi64_si128(x1, x0, 0x00); x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(_mm_xor_si128(x1, x4), x5);
    while (len >= 16) {
        x2 = _mm_loadu_si128((const __m128i *)buf);
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00); x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x1 = _mm_xor_si128(_mm_xor_si128(x1, x2), x5);
        buf += 16; len -= 16;
    }
    x2 = _mm_clmulepi64_si128(x1, x0, 0x10);
    x3 = _mm_setr_epi32(~0, 0, ~0, 0);
    x1 = _mm_srli_si128(x1, 8);
    x1 = _mm_xor_si128(x1, x2);
    x0 = _mm_loadl_epi64((const __m128i *)pd_k5k0);
    x2 = _mm_srli_si128(x1, 4);
    x1 = _mm_and_si128(x1, x3);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_xor_si128(x1, x2);
    x0 = _mm_load_si128((const __m128i *)pd_poly);
    x2 = _mm_and_si128(x1, x3);
    x2 = _mm_clmulepi64_si128(x2, x0, 0x10);
    x2 = _mm_and_si128(x2, x3);
    x2 = _mm_clmulepi64_si128(x2, x0, 0x00);
    x1 = _mm_xor_si128(x1, x2);
    return (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(x1, 4));
}
static uint32_t pd_crc_pclmul(uint32_t crc, const uint8_t *p, size_t n) {
    if (n >= 64) {
        size_t chunk = n & ~(size_t)15;
        crc = ~pd_crc_fold_pclmul(p, chunk, ~crc);
        p += chunk; n -= chunk;
    }
    return pd_crc_slice8(crc, p, n);
}

/* ============================================================================================== */
/* LZ77 match copy                                                                                 */
/* ============================================================================================== */
static void pd_copy_scalar(uint8_t *out, size_t pos, size_t dist, size_t len) {
    uint8_t *d = out + pos;
    const uint8_t *s = d - dist;
    if (dist >= len) { memcpy(d, s, len); return; }
    while (len--) *d++ = *s++;
}
static void pd_copy_sse2(uint8_t *out, size_t pos, size_t dist, size_t len) {
    uint8_t *d = out + pos;
    const uint8_t *s = d - dist;
    if (dist >= 16) {       /* each 16-byte source chunk lies completely before its destination */
        while (len >= 16) { _mm_storeu_si128((__m128i *)d, _mm_loadu_si128((const __m128i *)s)); d += 16; s += 16; len -= 16; }
    } else if (dist == 1) {
        memset(d, s[0], len); return;
    }
    while (len--) *d++ = *s++;
}
PD_TGT("avx2") static void pd_copy_avx2(uint8_t *out, size_t pos, size_t dist, size_t len) {
    uint8_t *d = out + pos;
    const uint8_t *s = d - dist;
    if (dist >= 32) {
        while (len >= 32) { _mm256_storeu_si256((__m256i *)d, _mm256_loadu_si256((const __m256i *)s)); d += 32; s += 32; len -= 32; }
    }
    if (dist >= 16) {
        while (len >= 16) { _mm_storeu_si128((__m128i *)d, _mm_loadu_si128((const __m128i *)s)); d += 16; s += 16; len -= 16; }
    } else if (dist == 1) {
        memset(d, s[0], len); return;
    }
    while (len--) *d++ = *s++;
}

/* ============================================================================================== */
/* unfilter                                                                                        */
/* ============================================================================================== */
static inline uint8_t pd_paeth(int a, int b, int c) {
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return (uint8_t)(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}
/* scalar tails: continue a filter from byte i (bytes before i are final) */
static void pd_unf_tail(int ft, uint8_t *c, const uint8_t *p, size_t i, size_t n, size_t bpp) {
    switch (ft) {
    case 1: for (; i < n; i++) if (i >= bpp) c[i] = (uint8_t)(c[i] + c[i - bpp]); break;
    case 2: for (; i < n; i++) c[i] = (uint8_t)(c[i] + p[i]); break;
    case 3: for (; i < n; i++) c[i] = (uint8_t)(c[i] + (((i >= bpp ? c[i - bpp] : 0) + p[i]) >> 1)); break;
    case 4: for (; i < n; i++) c[i] = (uint8_t)(c[i] + (i >= bpp ? pd_paeth(c[i - bpp], p[i], p[i - bpp]) : p[i])); break;
    default: break;
    }
}
static void pd_unf_scalar(int ft, uint8_t *c, const uint8_t *p, size_t n, int bpp) { pd_unf_tail(ft, c, p, 0, n, (size_t)bpp); }

/* Sub: in-register prefix sum. CH = bytes stored per step (16, or 12 for bpp 3/6); the 16-byte load
 * stays inside the row (i + 16 <= n). The carry is the last pixel of the previous step. */
#define PD_SUB_BODY(BPP, CH) do { \
    __m128i carry = _mm_setzero_si128(); \
    const __m128i msk = _mm_srli_si128(_mm_set1_epi8(-1), 16 - (BPP)); \
    for (; i + 16 <= n; i += (CH)) { \
        __m128i v = _mm_add_epi8(_mm_loadu_si128((const __m128i *)(c + i)), carry); \
        v = _mm_add_epi8(v, _mm_slli_si128(v, (BPP))); \
        if (2 * (BPP) < (CH)) v = _mm_add_epi8(v, _mm_slli_si128(v, (2 * (BPP)) & 15)); \
        if (4 * (BPP) < (CH)) v = _mm_add_epi8(v, _mm_slli_si128(v, (4 * (BPP)) & 15)); \
        if (8 * (BPP) < (CH)) v = _mm_add_epi8(v, _mm_slli_si128(v, (8 * (BPP)) & 15)); \
        if ((CH) == 16) _mm_storeu_si128((__m128i *)(c + i), v); \
        else { uint32_t t4 = (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(v, 8)); _mm_storel_epi64((__m128i *)(c + i), v); memcpy(c + i + 8, &t4, 4); } \
        carry = _mm_and_si128(_mm_srli_si128(v, (CH) - (BPP)), msk); \
    } } while (0)
/* one pixel of BPP (3, 4, 6, 8) bytes <-> low lanes of an XMM register, through GPRs (no store-forwarding
 * stalls through a stack temporary) and never touching bytes outside the pixel */
#define PD_LDPX(dst, src, BPP) do { \
    const uint8_t *s_ = (src); \
    if ((BPP) == 8) dst = _mm_loadl_epi64((const __m128i *)s_); \
    else { uint32_t lo_; \
           if ((BPP) == 3) { uint16_t w_; memcpy(&w_, s_, 2); lo_ = (uint32_t)w_ | (uint32_t)s_[2] << 16; } else memcpy(&lo_, s_, 4); \
           dst = _mm_cvtsi32_si128((int)lo_); \
           if ((BPP) == 6) { uint16_t hi_; memcpy(&hi_, s_ + 4, 2); dst = _mm_insert_epi16(dst, hi_, 2); } } \
    } while (0)
#define PD_STPX(dstp, v, BPP) do { \
    uint8_t *d_ = (dstp); \
    if ((BPP) == 8) _mm_storel_epi64((__m128i *)d_, (v)); \
    else { uint32_t lo_ = (uint32_t)_mm_cvtsi128_si32(v); \
           if ((BPP) == 3) { uint16_t w_ = (uint16_t)lo_; memcpy(d_, &w_, 2); d_[2] = (uint8_t)(lo_ >> 16); } else memcpy(d_, &lo_, 4); \
           if ((BPP) == 6) { uint16_t hi_ = (uint16_t)_mm_extract_epi16((v), 2); memcpy(d_ + 4, &hi_, 2); } } \
    } while (0)
/* Avg per pixel: floor((a + b) / 2) = pavgb(a, b) - ((a ^ b) & 1) */
#define PD_AVG_BODY(BPP) do { \
    __m128i a = _mm_setzero_si128(), one = _mm_set1_epi8(1); \
    for (; i + (BPP) <= n; i += (BPP)) { \
        __m128i b, x; PD_LDPX(b, p + i, BPP); PD_LDPX(x, c + i, BPP); \
        __m128i av = _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), one)); \
        a = _mm_add_epi8(x, av); PD_STPX(c + i, a, BPP); \
    } } while (0)
/* Paeth per pixel in 16-bit lanes: pa = |b-c|, pb = |a-c|, pc = |a+b-2c| */
#define PD_PAETH_BODY(BPP, ABS) do { \
    const __m128i z = _mm_setzero_si128(); \
    __m128i a = z, cc = z; \
    for (; i + (BPP) <= n; i += (BPP)) { \
        __m128i b, x; PD_LDPX(b, p + i, BPP); PD_LDPX(x, c + i, BPP); b = _mm_unpacklo_epi8(b, z); \
        __m128i db = _mm_sub_epi16(b, cc), da = _mm_sub_epi16(a, cc); \
        __m128i pa = ABS(db), pb = ABS(da), pc = ABS(_mm_add_epi16(db, da)); \
        __m128i nota = _mm_or_si128(_mm_cmpgt_epi16(pa, pb), _mm_cmpgt_epi16(pa, pc)); \
        __m128i usec = _mm_cmpgt_epi16(pb, pc); \
        __m128i bc = _mm_or_si128(_mm_and_si128(usec, cc), _mm_andnot_si128(usec, b)); \
        __m128i pr = _mm_or_si128(_mm_and_si128(nota, bc), _mm_andnot_si128(nota, a)); \
        __m128i r = _mm_add_epi8(x, _mm_packus_epi16(pr, pr)); \
        PD_STPX(c + i, r, BPP); a = _mm_unpacklo_epi8(r, z); cc = b; \
    } } while (0)
#define PD_ABS_SSE2(v) _mm_max_epi16((v), _mm_sub_epi16(_mm_setzero_si128(), (v)))
#define PD_ABS_SSSE3(v) _mm_abs_epi16(v)
/* the vectorized part handles: Up any bpp; Sub bpp 1,2,3,4,6,8; Avg / Paeth bpp 3,4,6,8 (bpp 1/2 is a
 * byte-serial dependency chain and stays scalar, as in libpng's SIMD filters). */
#define PD_UNF_FUNC(NAME, TGT, ABS, UPWIDE) \
TGT static void NAME(int ft, uint8_t *c, const uint8_t *p, size_t n, int bpp) { \
    size_t i = 0; \
    switch (ft) { \
    case 1: switch (bpp) { case 1: PD_SUB_BODY(1, 16); break; case 2: PD_SUB_BODY(2, 16); break; case 3: PD_SUB_BODY(3, 12); break; \
                           case 4: PD_SUB_BODY(4, 16); break; case 6: PD_SUB_BODY(6, 12); break; case 8: PD_SUB_BODY(8, 16); break; default: break; } \
            break; \
    case 2: UPWIDE; for (; i + 16 <= n; i += 16) _mm_storeu_si128((__m128i *)(c + i), _mm_add_epi8(_mm_loadu_si128((const __m128i *)(c + i)), _mm_loadu_si128((const __m128i *)(p + i)))); break; \
    case 3: switch (bpp) { case 3: PD_AVG_BODY(3); break; case 4: PD_AVG_BODY(4); break; case 6: PD_AVG_BODY(6); break; case 8: PD_AVG_BODY(8); break; default: break; } break; \
    case 4: switch (bpp) { case 3: PD_PAETH_BODY(3, ABS); break; case 4: PD_PAETH_BODY(4, ABS); break; case 6: PD_PAETH_BODY(6, ABS); break; case 8: PD_PAETH_BODY(8, ABS); break; default: break; } break; \
    default: return; \
    } \
    pd_unf_tail(ft, c, p, i, n, (size_t)bpp); \
}
PD_UNF_FUNC(pd_unf_sse2, , PD_ABS_SSE2, (void)0)
PD_UNF_FUNC(pd_unf_ssse3, PD_TGT("ssse3"), PD_ABS_SSSE3, (void)0)
#define PD_UP_AVX2 for (; i + 32 <= n; i += 32) _mm256_storeu_si256((__m256i *)(c + i), _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(c + i)), _mm256_loadu_si256((const __m256i *)(p + i))))
PD_UNF_FUNC(pd_unf_avx2, PD_TGT("avx2"), PD_ABS_SSSE3, PD_UP_AVX2)

/* ============================================================================================== */
/* row expansion to RGBA                                                                           */
/* ============================================================================================== */
static inline void pd_put16(uint8_t *o, unsigned v) { o[0] = (uint8_t)v; o[1] = (uint8_t)(v >> 8); }
static void pd_expand_scalar(const pd_fmt *f, const uint8_t *row, uint32_t w, uint8_t *out) {
    const int d = f->depth;
    if (d == 16) {
        for (uint32_t x = 0; x < w; x++, out += 8) {
            unsigned r, g, b, a = 65535;
            switch (f->ctype) {
            case 0: r = g = b = (unsigned)row[2 * x] << 8 | row[2 * x + 1]; if (f->has_trns && r == f->key[0]) a = 0; break;
            case 2: {
                const uint8_t *s = row + 6 * (size_t)x;
                r = (unsigned)s[0] << 8 | s[1]; g = (unsigned)s[2] << 8 | s[3]; b = (unsigned)s[4] << 8 | s[5];
                if (f->has_trns && r == f->key[0] && g == f->key[1] && b == f->key[2]) a = 0;
                break;
            }
            case 4: { const uint8_t *s = row + 4 * (size_t)x; r = g = b = (unsigned)s[0] << 8 | s[1]; a = (unsigned)s[2] << 8 | s[3]; break; }
            default: { const uint8_t *s = row + 8 * (size_t)x; r = (unsigned)s[0] << 8 | s[1]; g = (unsigned)s[2] << 8 | s[3]; b = (unsigned)s[4] << 8 | s[5]; a = (unsigned)s[6] << 8 | s[7]; break; }
            }
            pd_put16(out, r); pd_put16(out + 2, g); pd_put16(out + 4, b); pd_put16(out + 6, a);
        }
        return;
    }
    const unsigned maxv = (1u << d) - 1, scale = 255 / maxv;
    for (uint32_t x = 0; x < w; x++, out += 4) {
        switch (f->ctype) {
        case 0: case 3: {
            unsigned s;
            if (d == 8) s = row[x];
            else { size_t bit = (size_t)x * (unsigned)d; s = (row[bit >> 3] >> (8 - d - (bit & 7))) & maxv; }
            if (f->ctype == 3) { memcpy(out, &f->pal[s], 4); break; }
            out[0] = out[1] = out[2] = (uint8_t)(s * scale);
            out[3] = f->has_trns && s == f->key[0] ? 0 : 255;
            break;
        }
        case 2: { const uint8_t *s = row + 3 * (size_t)x; out[0] = s[0]; out[1] = s[1]; out[2] = s[2];
                  out[3] = f->has_trns && s[0] == f->key[0] && s[1] == f->key[1] && s[2] == f->key[2] ? 0 : 255; break; }
        case 4: out[0] = out[1] = out[2] = row[2 * (size_t)x]; out[3] = row[2 * (size_t)x + 1]; break;
        default: memcpy(out, row + 4 * (size_t)x, 4); break;
        }
    }
}
/* SSE2 fast paths: grey8 (no key), grey+alpha 8, RGBA8 (copy), 16-bit RGBA / grey+alpha / grey (no key).
 * Everything else (sub-byte depths, palette, RGB, keyed tRNS) and every tail: scalar. */
#define PD_SWAP16(x) _mm_or_si128(_mm_slli_epi16((x), 8), _mm_srli_epi16((x), 8))
#define PD_EXPAND_SSE2_CASES \
    if (f->depth == 8 && f->ctype == 6) { memcpy(out, row, 4 * (size_t)w); return; } \
    if (f->depth == 8 && f->ctype == 0 && !f->has_trns) { \
        const __m128i ff = _mm_set1_epi8(-1); \
        for (; x + 16 <= w; x += 16) { \
            __m128i g = _mm_loadu_si128((const __m128i *)(row + x)); \
            __m128i gg = _mm_unpacklo_epi8(g, g), ga = _mm_unpacklo_epi8(g, ff); \
            _mm_storeu_si128((__m128i *)(out + 4 * (size_t)x), _mm_unpacklo_epi16(gg, ga)); \
            _mm_storeu_si128((__m128i *)(out + 4 * (size_t)x + 16), _mm_unpackhi_epi16(gg, ga)); \
            gg = _mm_unpackhi_epi8(g, g); ga = _mm_unpackhi_epi8(g, ff); \
            _mm_storeu_si128((__m128i *)(out + 4 * (size_t)x + 32), _mm_unpacklo_epi16(gg, ga)); \
            _mm_storeu_si128((__m128i *)(out + 4 * (size_t)x + 48), _mm_unpackhi_epi16(gg, ga)); \
        } \
    } else if (f->depth == 8 && f->ctype == 4) { \
        const __m128i lo = _mm_set1_epi16(0xFF); \
        for (; x + 8 <= w; x += 8) { \
            __m128i v = _mm_loadu_si128((const __m128i *)(row + 2 * (size_t)x)); \
            __m128i g = _mm_and_si128(v, lo), gg = _mm_or_si128(g, _mm_slli_epi16(g, 8)); \
            _mm_storeu_si128((__m128i *)(out + 4 * (size_t)x), _mm_unpacklo_epi16(gg, v)); \
            _mm_storeu_si128((__m128i *)(out + 4 * (size_t)x + 16), _mm_unpackhi_epi16(gg, v)); \
        } \
    } else if (f->depth == 16 && f->ctype == 6) { \
        for (; x + 2 <= w; x += 2) { \
            __m128i v = _mm_loadu_si128((const __m128i *)(row + 8 * (size_t)x)); \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x), PD_SWAP16(v)); \
        } \
    } else if (f->depth == 16 && f->ctype == 4) { \
        for (; x + 4 <= w; x += 4) { \
            __m128i s = _mm_loadu_si128((const __m128i *)(row + 4 * (size_t)x)); s = PD_SWAP16(s); \
            __m128i q = _mm_unpacklo_epi32(s, s); \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x), _mm_shufflehi_epi16(_mm_shufflelo_epi16(q, _MM_SHUFFLE(1, 0, 0, 0)), _MM_SHUFFLE(1, 0, 0, 0))); \
            q = _mm_unpackhi_epi32(s, s); \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x + 16), _mm_shufflehi_epi16(_mm_shufflelo_epi16(q, _MM_SHUFFLE(1, 0, 0, 0)), _MM_SHUFFLE(1, 0, 0, 0))); \
        } \
    } else if (f->depth == 16 && f->ctype == 0 && !f->has_trns) { \
        const __m128i ones = _mm_set1_epi8(-1); \
        for (; x + 8 <= w; x += 8) { \
            __m128i s = _mm_loadu_si128((const __m128i *)(row + 2 * (size_t)x)); s = PD_SWAP16(s); \
            __m128i gg = _mm_unpacklo_epi16(s, s), gf = _mm_unpacklo_epi16(s, ones); \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x), _mm_unpacklo_epi32(gg, gf)); \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x + 16), _mm_unpackhi_epi32(gg, gf)); \
            gg = _mm_unpackhi_epi16(s, s); gf = _mm_unpackhi_epi16(s, ones); \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x + 32), _mm_unpacklo_epi32(gg, gf)); \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x + 48), _mm_unpackhi_epi32(gg, gf)); \
        } \
    }
/* SSSE3 adds PSHUFB paths for RGB8 / RGB16 without a key (12 input bytes per step, 16-byte load inside the row) */
#define PD_EXPAND_SSSE3_CASES \
    if (f->depth == 8 && f->ctype == 2 && !f->has_trns) { \
        const __m128i m = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1), al = _mm_set1_epi32((int)0xFF000000u); \
        for (; x + 4 <= w && 3 * (size_t)x + 16 <= 3 * (size_t)w; x += 4) \
            _mm_storeu_si128((__m128i *)(out + 4 * (size_t)x), _mm_or_si128(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(row + 3 * (size_t)x)), m), al)); \
    } else if (f->depth == 16 && f->ctype == 2 && !f->has_trns) { \
        const __m128i m = _mm_setr_epi8(1, 0, 3, 2, 5, 4, -1, -1, 7, 6, 9, 8, 11, 10, -1, -1), al = _mm_set_epi32((int)0xFFFF0000u, 0, (int)0xFFFF0000u, 0); \
        for (; x + 2 <= w && 6 * (size_t)x + 16 <= 6 * (size_t)w; x += 2) \
            _mm_storeu_si128((__m128i *)(out + 8 * (size_t)x), _mm_or_si128(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(row + 6 * (size_t)x)), m), al)); \
    }
static void pd_expand_tail(const pd_fmt *f, const uint8_t *row, uint32_t x, uint32_t w, uint8_t *out) {
    if (x >= w) return;
    int bypp = f->depth == 16 ? 8 : 4;
    size_t inpp = (size_t)(f->ctype == 0 ? 1 : f->ctype == 2 ? 3 : f->ctype == 4 ? 2 : f->ctype == 6 ? 4 : 1) * (f->depth == 16 ? 2 : 1);
    pd_expand_scalar(f, row + inpp * x, w - x, out + (size_t)bypp * x);   /* only reached for depth >= 8 */
}
static void pd_expand_sse2(const pd_fmt *f, const uint8_t *row, uint32_t w, uint8_t *out) {
    if (f->depth < 8 || f->ctype == 3) { pd_expand_scalar(f, row, w, out); return; }
    uint32_t x = 0;
    PD_EXPAND_SSE2_CASES
    pd_expand_tail(f, row, x, w, out);
}
PD_TGT("ssse3") static void pd_expand_ssse3(const pd_fmt *f, const uint8_t *row, uint32_t w, uint8_t *out) {
    if (f->depth < 8 || f->ctype == 3) { pd_expand_scalar(f, row, w, out); return; }
    uint32_t x = 0;
    PD_EXPAND_SSSE3_CASES
    else { PD_EXPAND_SSE2_CASES }
    pd_expand_tail(f, row, x, w, out);
}
/* AVX2: palette 8-bit via VPGATHERDD from the 256-entry RGBA table (indices >= PLTE count hit the opaque
 * black filler, so tRNS needs no special case), grey8 / grey+alpha8 via VPMULLD, RGB8 via VPSHUFB on two
 * 12-byte halves, RGBA16 byte swap via VPSHUFB; the rest as SSSE3 (VEX-encoded). */
PD_TGT("avx2") static void pd_expand_avx2(const pd_fmt *f, const uint8_t *row, uint32_t w, uint8_t *out) {
    uint32_t x = 0;
    if (f->ctype == 3 && f->depth == 8) {
        for (; x + 8 <= w; x += 8) {
            __m256i idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(row + x)));
            _mm256_storeu_si256((__m256i *)(out + 4 * (size_t)x), _mm256_i32gather_epi32((const int *)f->pal, idx, 4));
        }
        for (; x < w; x++) memcpy(out + 4 * (size_t)x, &f->pal[row[x]], 4);
        return;
    }
    if (f->depth < 8 || f->ctype == 3) { pd_expand_scalar(f, row, w, out); return; }
    if (f->depth == 8 && f->ctype == 0 && !f->has_trns) {
        const __m256i mul = _mm256_set1_epi32(0x010101), al = _mm256_set1_epi32((int)0xFF000000u);
        for (; x + 8 <= w; x += 8) {
            __m256i g = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(row + x)));
            _mm256_storeu_si256((__m256i *)(out + 4 * (size_t)x), _mm256_or_si256(_mm256_mullo_epi32(g, mul), al));
        }
    } else if (f->depth == 8 && f->ctype == 4) {
        const __m256i mul = _mm256_set1_epi32(0x010101), lo = _mm256_set1_epi32(0xFF);
        for (; x + 8 <= w; x += 8) {
            __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(row + 2 * (size_t)x)));
            __m256i g = _mm256_and_si256(v, lo), a = _mm256_slli_epi32(_mm256_srli_epi32(v, 8), 24);
            _mm256_storeu_si256((__m256i *)(out + 4 * (size_t)x), _mm256_or_si256(_mm256_mullo_epi32(g, mul), a));
        }
    } else if (f->depth == 8 && f->ctype == 2 && !f->has_trns) {
        const __m256i m = _mm256_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1, 0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
        const __m256i al = _mm256_set1_epi32((int)0xFF000000u);
        for (; x + 8 <= w && 3 * (size_t)x + 28 <= 3 * (size_t)w; x += 8) {
            const uint8_t *s = row + 3 * (size_t)x;
            __m256i v = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)s)), _mm_loadu_si128((const __m128i *)(s + 12)), 1);
            _mm256_storeu_si256((__m256i *)(out + 4 * (size_t)x), _mm256_or_si256(_mm256_shuffle_epi8(v, m), al));
        }
    } else if (f->depth == 16 && f->ctype == 6) {
        const __m256i m = _mm256_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        for (; x + 4 <= w; x += 4)
            _mm256_storeu_si256((__m256i *)(out + 8 * (size_t)x), _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(row + 8 * (size_t)x)), m));
    } else {
        PD_EXPAND_SSSE3_CASES
        else { PD_EXPAND_SSE2_CASES }
    }
    pd_expand_tail(f, row, x, w, out);
}

/* ============================================================================================== */
/* kernel sets                                                                                     */
/* ============================================================================================== */
static const pd_kernels pd_kernels_scalar = { "scalar", pd_unf_scalar, pd_expand_scalar, pd_adler_scalar, pd_crc_bytewise, pd_copy_scalar };
static const pd_kernels pd_kernels_sse2 = { "sse2", pd_unf_sse2, pd_expand_sse2, pd_adler_sse2, pd_crc_slice8, pd_copy_sse2 };
static const pd_kernels pd_kernels_ssse3 = { "ssse3", pd_unf_ssse3, pd_expand_ssse3, pd_adler_ssse3, pd_crc_slice8, pd_copy_sse2 };
/* AVX2 set: CRC-32 by PCLMULQDQ folding (the caller supplies the PCLMUL variant only when PCLMULQDQ is
 * usable; see pd_kernels_avx2_nopclmul) */
static const pd_kernels pd_kernels_avx2 = { "avx2", pd_unf_avx2, pd_expand_avx2, pd_adler_avx2, pd_crc_pclmul, pd_copy_avx2 };
static const pd_kernels pd_kernels_avx2_nopclmul = { "avx2", pd_unf_avx2, pd_expand_avx2, pd_adler_avx2, pd_crc_slice8, pd_copy_avx2 };

/* ============================================================================================== */
/* inflate                                                                                         */
/* ============================================================================================== */
#define PD_FASTB 10
typedef struct pd_huff {
    uint16_t fast[1 << PD_FASTB];   /* sym | len << 9 for codes <= PD_FASTB bits; 0 = use the slow path */
    uint16_t count[16];
    uint16_t sym[288];
} pd_huff;
/* returns 0 complete, >0 incomplete, -1 over-subscribed */
static int pd_huff_build(pd_huff *h, const uint8_t *len, int n) {
    uint16_t offs[16];
    memset(h->count, 0, sizeof h->count);
    memset(h->fast, 0, sizeof h->fast);
    for (int i = 0; i < n; i++) h->count[len[i]]++;
    h->count[0] = 0;
    int left = 1;
    for (int l = 1; l < 16; l++) { left <<= 1; left -= h->count[l]; if (left < 0) return -1; }
    offs[1] = 0;
    for (int l = 1; l < 15; l++) offs[l + 1] = (uint16_t)(offs[l] + h->count[l]);
    for (int i = 0; i < n; i++) if (len[i]) h->sym[offs[len[i]]++] = (uint16_t)i;
    /* canonical codes in (length, symbol) order -> bit-reversed fast table */
    unsigned code = 0;
    int k = 0;
    for (int l = 1; l <= PD_FASTB; l++) {
        for (int j = 0; j < h->count[l]; j++, k++, code++) {
            unsigned r = 0;
            for (int b = 0; b < l; b++) r |= ((code >> b) & 1u) << (l - 1 - b);
            for (unsigned e = r; e < (1u << PD_FASTB); e += 1u << l) h->fast[e] = (uint16_t)(h->sym[k] | (l << 9));
        }
        code <<= 1;
    }
    return left;
}
typedef struct pd_bits { const uint8_t *in; size_t n, pos; uint64_t buf; unsigned cnt; } pd_bits;
static inline void pd_refill(pd_bits *b) {
    if (b->pos + 8 <= b->n) {
        uint64_t v; memcpy(&v, b->in + b->pos, 8);
        b->buf |= v << b->cnt;
        b->pos += (63 - b->cnt) >> 3;
        b->cnt |= 56;
    } else {
        while (b->cnt <= 56 && b->pos < b->n) { b->buf |= (uint64_t)b->in[b->pos++] << b->cnt; b->cnt += 8; }
    }
}
static inline int pd_need(pd_bits *b, unsigned k) { if (b->cnt < k) pd_refill(b); return b->cnt >= k; }
static inline unsigned pd_take(pd_bits *b, unsigned k) { unsigned v = (unsigned)(b->buf & ((1ull << k) - 1)); b->buf >>= k; b->cnt -= k; return v; }
static inline int pd_decode(pd_bits *b, const pd_huff *h) {
    if (b->cnt < 15) pd_refill(b);
    unsigned e = h->fast[b->buf & ((1u << PD_FASTB) - 1)];
    if (e) {
        unsigned l = e >> 9;
        if (l > b->cnt) return PD_ZE_IN;
        b->buf >>= l; b->cnt -= l;
        return (int)(e & 511);
    }
    int code = 0, first = 0, index = 0;
    for (unsigned l = 1; l < 16; l++) {
        if (l > b->cnt) return PD_ZE_IN;
        code |= (int)((b->buf >> (l - 1)) & 1);
        int count = h->count[l];
        if (code - first < count) { b->buf >>= l; b->cnt -= l; return h->sym[index + code - first]; }
        index += count; first += count; first <<= 1; code <<= 1;
    }
    return PD_ZE_SYM;
}
static const uint16_t pd_lbase[29] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
static const uint8_t pd_lext[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
static const uint16_t pd_dbase[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
static const uint8_t pd_dext[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
static pd_huff pd_fixed_lit, pd_fixed_dist;
static void __attribute__((constructor)) pd_fixed_init(void) {
    uint8_t l[288];
    int i = 0;
    for (; i < 144; i++) l[i] = 8;
    for (; i < 256; i++) l[i] = 9;
    for (; i < 280; i++) l[i] = 7;
    for (; i < 288; i++) l[i] = 8;
    pd_huff_build(&pd_fixed_lit, l, 288);
    for (i = 0; i < 30; i++) l[i] = 5;
    l[30] = l[31] = 5;
    pd_huff_build(&pd_fixed_dist, l, 32);
}
/* incomplete codes are accepted only for a single length-1 code (zlib inftrees.c rule) */
static int pd_huff_ok(int r, const uint8_t *len, int n) {
    if (r < 0) return 0;
    if (r == 0) return 1;
    int max = 0;
    for (int i = 0; i < n; i++) if (len[i] > max) max = len[i];
    return max <= 1;   /* max == 0: no codes at all (allowed for distances, caught for literals by len[256]) */
}
typedef struct pd_zstate { pd_huff lit, dist, cl; } pd_zstate;
/* inflate a zlib stream into out[0..cap); *produced = bytes written. flags: PD_F_IGNORE_ADLER */
#define PD_F_IGNORE_CRC 1
#define PD_F_IGNORE_ADLER 2
#define PD_F_KEEP_RAW 4
static int pd_inflate_zlib(const pd_kernels *k, const uint8_t *in, size_t inlen, uint8_t *out, size_t cap, size_t *produced, int flags) {
    pd_bits b = { in, inlen, 0, 0, 0 };
    size_t op = 0;
    int rc = PD_OK;
    pd_zstate *z = NULL;
    if (!k) k = &pd_kernels_scalar;
    *produced = 0;
    if (!pd_need(&b, 16)) return PD_ZE_IN;
    {
        unsigned cmf = pd_take(&b, 8), flg = pd_take(&b, 8);
        if ((cmf & 15) != 8 || (cmf >> 4) > 7 || ((cmf << 8) | flg) % 31 || (flg & 0x20)) return PD_ZE_HDR;
    }
    for (int last = 0; !last;) {
        if (!pd_need(&b, 3)) { rc = PD_ZE_IN; goto done; }
        last = (int)pd_take(&b, 1);
        unsigned type = pd_take(&b, 2);
        if (type == 0) {
            pd_take(&b, b.cnt & 7);
            if (!pd_need(&b, 32)) { rc = PD_ZE_IN; goto done; }
            unsigned len = pd_take(&b, 16), nlen = pd_take(&b, 16);
            if (len != (~nlen & 0xFFFF)) { rc = PD_ZE_STORED; goto done; }
            b.pos -= b.cnt >> 3; b.buf = 0; b.cnt = 0;       /* return whole buffered bytes to the input */
            size_t avail = b.n - b.pos, m = len;
            int short_in = 0, short_out = 0;
            if (m > avail) { m = avail; short_in = 1; }
            if (m > cap - op) { m = cap - op; short_out = 1; }
            memcpy(out + op, b.in + b.pos, m);
            op += m; b.pos += m;
            if (short_out) { rc = PD_ZE_OUT; goto done; }
            if (short_in) { rc = PD_ZE_IN; goto done; }
            continue;
        }
        if (type == 3) { rc = PD_ZE_BTYPE; goto done; }
        const pd_huff *hl = &pd_fixed_lit, *hd = &pd_fixed_dist;
        if (type == 2) {
            if (!z && !(z = (pd_zstate *)malloc(sizeof *z))) { rc = PD_E_NOMEM; goto done; }
            if (!pd_need(&b, 14)) { rc = PD_ZE_IN; goto done; }
            unsigned hlit = pd_take(&b, 5) + 257, hdist = pd_take(&b, 5) + 1, hclen = pd_take(&b, 4) + 4;
            if (hlit > 286 || hdist > 30) { rc = PD_ZE_CODES; goto done; }
            static const uint8_t ord[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
            uint8_t cl[19] = { 0 }, ll[320];
            for (unsigned i = 0; i < hclen; i++) {
                if (!pd_need(&b, 3)) { rc = PD_ZE_IN; goto done; }
                cl[ord[i]] = (uint8_t)pd_take(&b, 3);
            }
            if (pd_huff_build(&z->cl, cl, 19) != 0) { rc = PD_ZE_CODES; goto done; }   /* must be complete */
            for (unsigned i = 0; i < hlit + hdist;) {
                int s = pd_decode(&b, &z->cl);
                if (s < 0) { rc = s == PD_ZE_IN ? PD_ZE_IN : PD_ZE_CODES; goto done; }
                if (s < 16) { ll[i++] = (uint8_t)s; continue; }
                unsigned rep, val = 0;
                if (s == 16) {
                    if (i == 0) { rc = PD_ZE_CODES; goto done; }
                    if (!pd_need(&b, 2)) { rc = PD_ZE_IN; goto done; }
                    val = ll[i - 1]; rep = 3 + pd_take(&b, 2);
                } else if (s == 17) {
                    if (!pd_need(&b, 3)) { rc = PD_ZE_IN; goto done; }
                    rep = 3 + pd_take(&b, 3);
                } else {
                    if (!pd_need(&b, 7)) { rc = PD_ZE_IN; goto done; }
                    rep = 11 + pd_take(&b, 7);
                }
                if (i + rep > hlit + hdist) { rc = PD_ZE_CODES; goto done; }
                while (rep--) ll[i++] = (uint8_t)val;
            }
            if (!ll[256]) { rc = PD_ZE_CODES; goto done; }
            if (!pd_huff_ok(pd_huff_build(&z->lit, ll, (int)hlit), ll, (int)hlit)) { rc = PD_ZE_CODES; goto done; }
            if (!pd_huff_ok(pd_huff_build(&z->dist, ll + hlit, (int)hdist), ll + hlit, (int)hdist)) { rc = PD_ZE_CODES; goto done; }
            hl = &z->lit; hd = &z->dist;
        }
        for (;;) {
            int s = pd_decode(&b, hl);
            if (s < 0) { rc = s; goto done; }
            if (s < 256) {
                if (op >= cap) { rc = PD_ZE_OUT; goto done; }
                out[op++] = (uint8_t)s;
                continue;
            }
            if (s == 256) break;
            s -= 257;
            if (s >= 29) { rc = PD_ZE_SYM; goto done; }
            unsigned len = pd_lbase[s];
            if (pd_lext[s]) { if (!pd_need(&b, pd_lext[s])) { rc = PD_ZE_IN; goto done; } len += pd_take(&b, pd_lext[s]); }
            int ds = pd_decode(&b, hd);
            if (ds < 0) { rc = ds; goto done; }
            if (ds >= 30) { rc = PD_ZE_SYM; goto done; }
            unsigned dist = pd_dbase[ds];
            if (pd_dext[ds]) { if (!pd_need(&b, pd_dext[ds])) { rc = PD_ZE_IN; goto done; } dist += pd_take(&b, pd_dext[ds]); }
            if (dist > op) { rc = PD_ZE_DIST; goto done; }
            if (len > cap - op) { k->copy_match(out, op, dist, cap - op); op = cap; rc = PD_ZE_OUT; goto done; }
            k->copy_match(out, op, dist, len);
            op += len;
        }
    }
    /* Adler-32 trailer, byte aligned, big-endian */
    pd_take(&b, b.cnt & 7);
    if (!pd_need(&b, 32)) { rc = PD_ZE_IN; goto done; }
    {
        uint32_t want = 0;
        for (int i = 0; i < 4; i++) want = (want << 8) | pd_take(&b, 8);
        if (!(flags & PD_F_IGNORE_ADLER) && k->adler32(1, out, op) != want) rc = PD_ZE_ADLER;
    }
done:
    free(z);
    *produced = op;
    return rc;
}

/* ============================================================================================== */
/* PNG                                                                                             */
/* ============================================================================================== */
typedef struct pd_options {
    const pd_kernels *k;      /* NULL = scalar */
    uint32_t max_pixels;      /* 0 = 1 << 24 */
    int flags;                /* PD_F_* */
} pd_options;
typedef struct pd_image {
    uint32_t w, h;
    int depth, ctype, interlace, bps;    /* bps: bytes per output channel (1 or 2) */
    size_t stride;                       /* w * 4 * bps */
    uint8_t *pix;                        /* RGBA; NULL on error */
    uint8_t *raw;                        /* with PD_F_KEEP_RAW: filtered-stream layout, rows unfiltered in place */
    size_t raw_len, raw_produced, raw_complete;   /* expected size, inflated bytes, bytes up to the last unfiltered row */
    int zstatus;                         /* inflate status */
    uint32_t n_idat, n_chunks;
    int has_trns, npal;
} pd_image;
static void pd_free(pd_image *im) { free(im->pix); free(im->raw); im->pix = im->raw = NULL; }

static const uint8_t pd_a7_x0[7] = { 0, 4, 0, 2, 0, 1, 0 }, pd_a7_dx[7] = { 8, 8, 4, 4, 2, 2, 1 };
static const uint8_t pd_a7_y0[7] = { 0, 0, 4, 0, 2, 0, 1 }, pd_a7_dy[7] = { 8, 8, 8, 4, 4, 2, 2 };
static uint32_t pd_pass_dim(uint32_t n, unsigned o, unsigned d) { return n > o ? (n - o + d - 1) / d : 0; }
static int pd_channels(int ctype) { return ctype == 0 || ctype == 3 ? 1 : ctype == 2 ? 3 : ctype == 4 ? 2 : 4; }
static uint32_t pd_be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static int pd_valid_depth(int ctype, int depth) {
    switch (ctype) {
    case 0: return depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16;
    case 3: return depth == 1 || depth == 2 || depth == 4 || depth == 8;
    case 2: case 4: case 6: return depth == 8 || depth == 16;
    default: return 0;
    }
}

static int pd_decode_png(const uint8_t *d, size_t n, const pd_options *opt, pd_image *im) {
    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    const pd_kernels *k = opt && opt->k ? opt->k : &pd_kernels_scalar;
    int flags = opt ? opt->flags : 0;
    uint64_t maxpx = opt && opt->max_pixels ? opt->max_pixels : (1u << 24);
    pd_fmt *f = NULL;
    uint8_t *idat = NULL, *zero = NULL, *tmp = NULL;
    size_t idat_len = 0, idat_cap = 0;
    int rc = PD_OK, seen_ihdr = 0, seen_plte = 0, seen_trns = 0, seen_idat = 0, seen_iend = 0;
    if (!im) return PD_E_ARG;
    memset(im, 0, sizeof *im);
    if (!d && n) return PD_E_ARG;
    if (n < 8 || memcmp(d, sig, 8)) return PD_E_SIG;
    f = (pd_fmt *)calloc(1, sizeof *f);
    if (!f) return PD_E_NOMEM;
    for (int i = 0; i < 256; i++) f->pal[i] = 0xFF000000u;   /* opaque black filler */
    size_t pos = 8;
    for (;;) {
        if (pos == n || n - pos < 8) { rc = PD_E_TRUNC; break; }
        uint32_t len = pd_be32(d + pos);
        const uint8_t *type = d + pos + 4, *data = d + pos + 8;
        if (len > 0x7FFFFFFFu) { rc = PD_E_CHUNK; break; }
        int letters = 1;
        for (int i = 0; i < 4; i++) letters &= (type[i] >= 'A' && type[i] <= 'Z') || (type[i] >= 'a' && type[i] <= 'z');
        if (!letters) { rc = PD_E_CHUNK; break; }
        int is_idat = !memcmp(type, "IDAT", 4);
        size_t avail = n - pos - 8;
        if (avail < (size_t)len + 4) {
            /* truncated chunk: an IDAT contributes the data bytes that are present */
            if (is_idat && seen_ihdr) {
                size_t m = avail < len ? avail : len;
                if (idat_len + m > idat_cap) {
                    size_t nc = idat_len + m;
                    uint8_t *ni = (uint8_t *)realloc(idat, nc ? nc : 1);
                    if (!ni) { rc = PD_E_NOMEM; break; }
                    idat = ni; idat_cap = nc;
                }
                memcpy(idat + idat_len, data, m); idat_len += m;
                im->n_idat++;
            }
            rc = PD_E_TRUNC;
            break;
        }
        im->n_chunks++;
        if (!(flags & PD_F_IGNORE_CRC) && k->crc32(0, type, (size_t)len + 4) != pd_be32(data + len)) { rc = PD_E_CRC; break; }
        pos += (size_t)len + 12;
        if (!seen_ihdr && memcmp(type, "IHDR", 4)) { rc = PD_E_ORDER; break; }
        if (!memcmp(type, "IHDR", 4)) {
            if (seen_ihdr) { rc = PD_E_ORDER; break; }
            if (len != 13) { rc = PD_E_IHDR; break; }
            seen_ihdr = 1;
            im->w = pd_be32(data); im->h = pd_be32(data + 4);
            im->depth = data[8]; im->ctype = data[9]; im->interlace = data[12];
            if (!im->w || !im->h || im->w > 0x7FFFFFFFu || im->h > 0x7FFFFFFFu || !pd_valid_depth(im->ctype, im->depth) ||
                data[10] || data[11] || data[12] > 1) { rc = PD_E_IHDR; break; }
            f->ctype = im->ctype; f->depth = im->depth;
        } else if (!memcmp(type, "PLTE", 4)) {
            if (seen_plte || seen_idat || seen_trns) { rc = PD_E_ORDER; break; }
            if (im->ctype == 0 || im->ctype == 4) { rc = PD_E_PLTE; break; }
            if (len == 0 || len % 3 || len / 3 > 256 || (im->ctype == 3 && len / 3 > (1u << im->depth))) { rc = PD_E_PLTE; break; }
            seen_plte = 1;
            im->npal = (int)(len / 3);
            for (int i = 0; i < im->npal; i++)
                f->pal[i] = (uint32_t)data[3 * i] | (uint32_t)data[3 * i + 1] << 8 | (uint32_t)data[3 * i + 2] << 16 | 0xFF000000u;
        } else if (!memcmp(type, "tRNS", 4)) {
            if (seen_trns || seen_idat) { rc = PD_E_ORDER; break; }
            seen_trns = 1;
            if (im->ctype == 0) {
                if (len != 2) { rc = PD_E_CHUNK; break; }
                f->has_trns = 1; f->key[0] = (uint16_t)(data[0] << 8 | data[1]);
            } else if (im->ctype == 2) {
                if (len != 6) { rc = PD_E_CHUNK; break; }
                f->has_trns = 1;
                for (int i = 0; i < 3; i++) f->key[i] = (uint16_t)(data[2 * i] << 8 | data[2 * i + 1]);
            } else if (im->ctype == 3) {
                if (!seen_plte) { rc = PD_E_ORDER; break; }
                if ((int)len > im->npal) { rc = PD_E_CHUNK; break; }
                for (uint32_t i = 0; i < len; i++) f->pal[i] = (f->pal[i] & 0x00FFFFFFu) | (uint32_t)data[i] << 24;
            }   /* types 4/6: tRNS is not allowed with an alpha channel; ignored (libpng: benign error) */
            im->has_trns = 1;
        } else if (is_idat) {
            if (im->ctype == 3 && !seen_plte) { rc = PD_E_PLTE; break; }
            seen_idat = 1;
            im->n_idat++;
            if (idat_len + len > idat_cap) {
                size_t nc = idat_cap ? idat_cap : 4096;
                while (nc < idat_len + len) nc *= 2;
                uint8_t *ni = (uint8_t *)realloc(idat, nc);
                if (!ni) { rc = PD_E_NOMEM; break; }
                idat = ni; idat_cap = nc;
            }
            memcpy(idat + idat_len, data, len); idat_len += len;
        } else if (!memcmp(type, "IEND", 4)) {
            seen_iend = 1;
            break;
        } else if (!(type[0] & 0x20)) {
            rc = PD_E_CRITICAL; break;
        }
    }
    if (rc == PD_OK && !seen_idat) rc = PD_E_NODATA;
    if (rc != PD_OK && rc != PD_E_TRUNC) goto out;
    if (!seen_ihdr) goto out;
    (void)seen_iend;
    {
        /* raw layout */
        int ch = pd_channels(im->ctype), bitspp = ch * im->depth, bpp = (bitspp + 7) / 8;
        if ((uint64_t)im->w * im->h > maxpx) { rc = PD_E_LIMIT; goto out; }
        uint64_t raw = 0, maxrow = 0;
        int npass = im->interlace ? 7 : 1;
        for (int ps = 0; ps < npass; ps++) {
            uint32_t pw = im->interlace ? pd_pass_dim(im->w, pd_a7_x0[ps], pd_a7_dx[ps]) : im->w;
            uint32_t ph = im->interlace ? pd_pass_dim(im->h, pd_a7_y0[ps], pd_a7_dy[ps]) : im->h;
            if (!pw || !ph) continue;
            uint64_t rb = ((uint64_t)pw * (unsigned)bitspp + 7) / 8;
            raw += (rb + 1) * ph;
            if (rb > maxrow) maxrow = rb;
        }
        if (raw > (uint64_t)(SIZE_MAX / 4)) { rc = PD_E_LIMIT; goto out; }
        im->bps = im->depth == 16 ? 2 : 1;
        im->stride = (size_t)im->w * 4 * (size_t)im->bps;
        im->raw_len = (size_t)raw;
        im->raw = (uint8_t *)malloc(im->raw_len ? im->raw_len : 1);
        zero = (uint8_t *)calloc((size_t)maxrow + 1, 1);
        if (!im->raw || !zero) { rc = PD_E_NOMEM; goto out; }
        size_t prod = 0;
        int zs = pd_inflate_zlib(k, idat, idat_len, im->raw, im->raw_len, &prod, flags);
        im->zstatus = zs; im->raw_produced = prod;
        if (rc == PD_OK) {
            if (zs == PD_ZE_OUT) rc = PD_E_ZLONG;
            else if (zs == PD_E_NOMEM) rc = PD_E_NOMEM;
            else if (zs != PD_OK) rc = PD_E_ZLIB_BASE + zs;
            else if (prod != im->raw_len) rc = PD_E_ZSHORT;
        }
        /* unfilter every complete row (stop at a bad filter type) */
        size_t off = 0;
        int stop = 0;
        for (int ps = 0; ps < npass && !stop; ps++) {
            uint32_t pw = im->interlace ? pd_pass_dim(im->w, pd_a7_x0[ps], pd_a7_dx[ps]) : im->w;
            uint32_t ph = im->interlace ? pd_pass_dim(im->h, pd_a7_y0[ps], pd_a7_dy[ps]) : im->h;
            if (!pw || !ph) continue;
            size_t rb = ((size_t)pw * (unsigned)bitspp + 7) / 8;
            const uint8_t *prev = zero + 1;
            for (uint32_t y = 0; y < ph; y++) {
                if (off + rb + 1 > prod) { stop = 1; break; }
                uint8_t ft = im->raw[off];
                if (ft > 4) { if (rc == PD_OK) rc = PD_E_FILTER; stop = 1; break; }
                if (ft) k->unfilter(ft, im->raw + off + 1, prev, rb, bpp);
                prev = im->raw + off + 1;
                off += rb + 1;
            }
        }
        im->raw_complete = off;
        if (rc != PD_OK) goto out;
        /* expand to RGBA */
        im->pix = (uint8_t *)malloc(im->stride * im->h);
        if (!im->pix) { rc = PD_E_NOMEM; goto out; }
        off = 0;
        if (!im->interlace) {
            size_t rb = ((size_t)im->w * (unsigned)bitspp + 7) / 8;
            for (uint32_t y = 0; y < im->h; y++, off += rb + 1) k->expand(f, im->raw + off + 1, im->w, im->pix + y * im->stride);
        } else {
            size_t bypp = 4 * (size_t)im->bps;
            tmp = (uint8_t *)malloc(im->stride ? im->stride : 1);
            if (!tmp) { rc = PD_E_NOMEM; goto out; }
            for (int ps = 0; ps < 7; ps++) {
                uint32_t pw = pd_pass_dim(im->w, pd_a7_x0[ps], pd_a7_dx[ps]), ph = pd_pass_dim(im->h, pd_a7_y0[ps], pd_a7_dy[ps]);
                if (!pw || !ph) continue;
                size_t rb = ((size_t)pw * (unsigned)bitspp + 7) / 8;
                for (uint32_t y = 0; y < ph; y++, off += rb + 1) {
                    k->expand(f, im->raw + off + 1, pw, tmp);
                    uint8_t *dst = im->pix + (size_t)(pd_a7_y0[ps] + y * pd_a7_dy[ps]) * im->stride + (size_t)pd_a7_x0[ps] * bypp;
                    for (uint32_t x = 0; x < pw; x++) memcpy(dst + (size_t)x * pd_a7_dx[ps] * bypp, tmp + x * bypp, bypp);
                }
            }
        }
    }
out:
    free(f); free(idat); free(zero); free(tmp);
    if (rc != PD_OK) { free(im->pix); im->pix = NULL; }
    if (!(flags & PD_F_KEEP_RAW)) { free(im->raw); im->raw = NULL; }
    return rc;
}
#endif
