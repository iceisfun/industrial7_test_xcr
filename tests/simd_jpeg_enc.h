/* simd_jpeg_enc.h - tiny deterministic baseline JPEG encoder + synthetic test-image generator.
 *
 * Integer-only (separable FDCT with a literal 13-bit cosine table, int64 accumulation, libjpeg's
 * jccolor.c fixed-point RGB->YCbCr, box downsampling with libjpeg's alternating bias, Annex K tables),
 * so the bytes it produces are identical on every platform and compiler. The corpus host builds the
 * same header natively (corpus/jpeg/gen_synth.c), encodes the synthetic images and records the
 * resulting file hash and djpeg's decoded-pixel hash in corpus/reference_vectors; the exe regenerates
 * the files at run time instead of embedding megabytes. */
#ifndef SIMD_JPEG_ENC_H
#define SIMD_JPEG_ENC_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "simd_jpeg_dec.h"   /* Annex K Huffman tables (jd_std_*) */

/* deterministic "photo-like" RGB test image: gradients, rings, hard edges, fine texture, noise */
static uint32_t je_hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
static void je_synth_rgb(uint8_t *rgb, int w, int h, uint32_t seed) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int fx = (int)((int64_t)x * 1024 / (w > 1 ? w - 1 : 1)), fy = (int)((int64_t)y * 1024 / (h > 1 ? h - 1 : 1));
            int dx = fx - 512, dy = fy - 400;
            int r2 = (dx * dx + dy * dy) >> 6;                    /* 0..~10000 */
            int ring = (r2 / 97) & 1;
            int n = (int)(je_hash32((uint32_t)(y * 65599 + x) ^ seed) & 31) - 16;
            int r = (fx >> 2) + (ring ? 40 : -40) + n;
            int g = (fy >> 2) + ((x / 8 + y / 8) & 1 ? 25 : -25) + n / 2;
            int b = 255 - ((fx + fy) >> 3) + ((x * 7 + y * 3) % 23) - 11;
            if (fx > 700 && fy < 300) { r = 250; g = 30; b = 20; }  /* flat saturated patch with hard edge */
            if (((x ^ y) & 31) == 0) { r = g = b = 0; }             /* thin dark lines */
            uint8_t *o = rgb + ((size_t)y * w + x) * 3;
            o[0] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
            o[1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
            o[2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
}

static const uint8_t je_std_lum_q[64] = {   /* natural order, Annex K.1 */
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };
static const uint8_t je_std_chr_q[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99 };
/* round(8192 * C(u)/2 * cos((2x+1)u pi/16)) */
static const int16_t je_k[8][8] = {
    { 2896, 2896, 2896, 2896, 2896, 2896, 2896, 2896 }, { 4017, 3406, 2276, 799, -799, -2276, -3406, -4017 },
    { 3784, 1567, -1567, -3784, -3784, -1567, 1567, 3784 }, { 3406, -799, -4017, -2276, 2276, 4017, 799, -3406 },
    { 2896, -2896, -2896, 2896, 2896, -2896, -2896, 2896 }, { 2276, -4017, 799, 3406, -3406, -799, 4017, -2276 },
    { 1567, -3784, 3784, -1567, -1567, 3784, -3784, 1567 }, { 799, -2276, 3406, -4017, 4017, -3406, 2276, -799 } };
static const uint8_t je_zz[64] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

typedef struct je_buf { uint8_t *p; size_t n, cap; int fail; uint32_t acc; int nb; } je_buf;
static void je_put(je_buf *b, uint8_t v) {
    if (b->n == b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 65536;
        uint8_t *q = (uint8_t *)realloc(b->p, nc);
        if (!q) { b->fail = 1; return; }
        b->p = q; b->cap = nc;
    }
    b->p[b->n++] = v;
}
static void je_put16(je_buf *b, int v) { je_put(b, (uint8_t)(v >> 8)); je_put(b, (uint8_t)v); }
static void je_bits(je_buf *b, uint32_t code, int len) {
    for (int i = len - 1; i >= 0; i--) {
        b->acc = (b->acc << 1) | ((code >> i) & 1);
        if (++b->nb == 8) { uint8_t c = (uint8_t)b->acc; je_put(b, c); if (c == 0xFF) je_put(b, 0); b->acc = 0; b->nb = 0; }
    }
}
static void je_flush(je_buf *b) { while (b->nb) je_bits(b, 1, 1); }

typedef struct je_huff { uint16_t code[256]; uint8_t len[256]; } je_huff;
static void je_make_huff(je_huff *h, const uint8_t *bits, const uint8_t *vals) {
    int code = 0, p = 0;
    memset(h, 0, sizeof *h);
    for (int l = 1; l <= 16; l++) {
        for (int i = 0; i < bits[l - 1]; i++) { h->code[vals[p]] = (uint16_t)code++; h->len[vals[p]] = (uint8_t)l; p++; }
        code <<= 1;
    }
}
static void je_dht(je_buf *b, int cls, int id, const uint8_t *bits, const uint8_t *vals) {
    int n = 0;
    for (int i = 0; i < 16; i++) n += bits[i];
    je_put16(b, 0xFFC4); je_put16(b, 2 + 1 + 16 + n); je_put(b, (uint8_t)(cls << 4 | id));
    for (int i = 0; i < 16; i++) je_put(b, bits[i]);
    for (int i = 0; i < n; i++) je_put(b, vals[i]);
}
/* 8x8 FDCT of level-shifted samples, quantise, zigzag */
static void je_fdct_quant(const int *s, const uint8_t *q, int *out) {
    int64_t t[8][8];
    for (int y = 0; y < 8; y++)
        for (int u = 0; u < 8; u++) { int64_t a = 0; for (int x = 0; x < 8; x++) a += (int64_t)je_k[u][x] * s[y * 8 + x]; t[y][u] = a; }
    for (int v = 0; v < 8; v++)
        for (int u = 0; u < 8; u++) {
            int64_t a = 0;
            for (int y = 0; y < 8; y++) a += (int64_t)je_k[v][y] * t[y][u];
            int64_t F = a >= 0 ? (a + (1 << 25)) >> 26 : -((-a + (1 << 25)) >> 26);
            int64_t qq = q[v * 8 + u], Q = F >= 0 ? (F + qq / 2) / qq : -((-F + qq / 2) / qq);
            out[v * 8 + u] = (int)Q;
        }
}
static int je_cat(int v) { int a = v < 0 ? -v : v, n = 0; while (a) { n++; a >>= 1; } return n; }
static void je_block(je_buf *b, const int *coef, int *pred, const je_huff *dc, const je_huff *ac) {
    int diff = coef[0] - *pred; *pred = coef[0];
    int n = je_cat(diff);
    je_bits(b, dc->code[n], dc->len[n]);
    if (n) je_bits(b, (uint32_t)(diff < 0 ? diff - 1 : diff) & ((1u << n) - 1), n);
    int run = 0;
    for (int k = 1; k < 64; k++) {
        int v = coef[je_zz[k]];
        if (!v) { run++; continue; }
        while (run > 15) { je_bits(b, ac->code[0xF0], ac->len[0xF0]); run -= 16; }
        n = je_cat(v);
        int sym = (run << 4) | n;
        je_bits(b, ac->code[sym], ac->len[sym]);
        je_bits(b, (uint32_t)(v < 0 ? v - 1 : v) & ((1u << n) - 1), n);
        run = 0;
    }
    if (run) je_bits(b, ac->code[0], ac->len[0]);
}

/* sub: 0 = grayscale, 1 = 4:4:4, 2 = 4:2:0. restart: MCUs per interval (0 = none).
 * Returns malloc'd JPEG bytes (free()) and *outlen, or NULL. */
static uint8_t *je_encode(const uint8_t *rgb, int w, int h, int sub, int quality, int restart, size_t *outlen) {
    int nc = sub == 0 ? 1 : 3, hs = sub == 2 ? 2 : 1;
    int mw = 8 * hs, mh = 8 * hs;
    int mx = (w + mw - 1) / mw, my = (h + mh - 1) / mh;
    int pw = mx * mw, ph = my * mh;
    uint8_t q[2][64];
    int scale = quality < 50 ? 5000 / quality : 200 - quality * 2;
    for (int i = 0; i < 64; i++) {
        int a = (je_std_lum_q[i] * scale + 50) / 100, c = (je_std_chr_q[i] * scale + 50) / 100;
        q[0][i] = (uint8_t)(a < 1 ? 1 : a > 255 ? 255 : a); q[1][i] = (uint8_t)(c < 1 ? 1 : c > 255 ? 255 : c);
    }
    /* full-resolution padded planes (edge replicated) */
    uint8_t *pl = (uint8_t *)malloc((size_t)pw * ph * 3);
    if (!pl) return NULL;
    for (int y = 0; y < ph; y++)
        for (int x = 0; x < pw; x++) {
            const uint8_t *s = rgb + ((size_t)(y < h ? y : h - 1) * w + (x < w ? x : w - 1)) * 3;
            int R = s[0], G = s[1], B = s[2];
            size_t i = (size_t)y * pw + x;
            if (nc == 1) pl[i] = (uint8_t)((19595 * R + 38470 * G + 7471 * B + 32768) >> 16);
            else {
                pl[i] = (uint8_t)((19595 * R + 38470 * G + 7471 * B + 32768) >> 16);
                pl[(size_t)pw * ph + i] = (uint8_t)((-11059 * R - 21709 * G + 32768 * B + (128 << 16) + 32767) >> 16);
                pl[(size_t)pw * ph * 2 + i] = (uint8_t)((32768 * R - 27439 * G - 5329 * B + (128 << 16) + 32767) >> 16);
            }
        }
    je_buf b = { 0 };
    je_put16(&b, 0xFFD8);
    je_put16(&b, 0xFFE0); je_put16(&b, 16);
    { static const uint8_t jfif[14] = { 'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0 }; for (int i = 0; i < 14; i++) je_put(&b, jfif[i]); }
    for (int t = 0; t < (nc == 1 ? 1 : 2); t++) {
        je_put16(&b, 0xFFDB); je_put16(&b, 67); je_put(&b, (uint8_t)t);
        for (int i = 0; i < 64; i++) je_put(&b, q[t][je_zz[i]]);
    }
    je_put16(&b, 0xFFC0); je_put16(&b, 8 + 3 * nc); je_put(&b, 8); je_put16(&b, h); je_put16(&b, w); je_put(&b, (uint8_t)nc);
    for (int c = 0; c < nc; c++) { je_put(&b, (uint8_t)(c + 1)); je_put(&b, (uint8_t)(c == 0 ? (hs << 4 | hs) : 0x11)); je_put(&b, (uint8_t)(c ? 1 : 0)); }
    je_dht(&b, 0, 0, jd_std_dc_lum_bits, jd_std_dc_vals); je_dht(&b, 1, 0, jd_std_ac_lum_bits, jd_std_ac_lum_vals);
    if (nc == 3) { je_dht(&b, 0, 1, jd_std_dc_chr_bits, jd_std_dc_vals); je_dht(&b, 1, 1, jd_std_ac_chr_bits, jd_std_ac_chr_vals); }
    if (restart) { je_put16(&b, 0xFFDD); je_put16(&b, 4); je_put16(&b, restart); }
    je_put16(&b, 0xFFDA); je_put16(&b, 6 + 2 * nc); je_put(&b, (uint8_t)nc);
    for (int c = 0; c < nc; c++) { je_put(&b, (uint8_t)(c + 1)); je_put(&b, (uint8_t)(c ? 0x11 : 0x00)); }
    je_put(&b, 0); je_put(&b, 63); je_put(&b, 0);
    je_huff hd[2], ha[2];
    je_make_huff(&hd[0], jd_std_dc_lum_bits, jd_std_dc_vals); je_make_huff(&ha[0], jd_std_ac_lum_bits, jd_std_ac_lum_vals);
    je_make_huff(&hd[1], jd_std_dc_chr_bits, jd_std_dc_vals); je_make_huff(&ha[1], jd_std_ac_chr_bits, jd_std_ac_chr_vals);
    int pred[3] = { 0, 0, 0 }, s[64], co[64], mcu = 0, rst = 0;
    for (int yy = 0; yy < my; yy++)
        for (int xx = 0; xx < mx; xx++) {
            if (restart && mcu && mcu % restart == 0) {
                je_flush(&b); je_put(&b, 0xFF); je_put(&b, (uint8_t)(0xD0 + (rst & 7))); rst++;
                pred[0] = pred[1] = pred[2] = 0;
            }
            mcu++;
            for (int by = 0; by < hs; by++)
                for (int bx = 0; bx < hs; bx++) {
                    for (int i = 0; i < 64; i++)
                        s[i] = pl[(size_t)(yy * mh + by * 8 + i / 8) * pw + xx * mw + bx * 8 + i % 8] - 128;
                    je_fdct_quant(s, q[0], co);
                    je_block(&b, co, &pred[0], &hd[0], &ha[0]);
                }
            for (int c = 1; c < nc; c++) {
                const uint8_t *P = pl + (size_t)pw * ph * c;
                for (int i = 0; i < 64; i++) {
                    int y0 = yy * mh + (i / 8) * hs, x0 = xx * mw + (i % 8) * hs;
                    if (hs == 1) s[i] = P[(size_t)y0 * pw + x0] - 128;
                    else {   /* jcsample.c h2v2_downsample: bias alternates 1,2 across columns */
                        int sum = P[(size_t)y0 * pw + x0] + P[(size_t)y0 * pw + x0 + 1] + P[(size_t)(y0 + 1) * pw + x0] + P[(size_t)(y0 + 1) * pw + x0 + 1];
                        s[i] = ((sum + 1 + (i & 1)) >> 2) - 128;
                    }
                }
                je_fdct_quant(s, q[1], co);
                je_block(&b, co, &pred[c], &hd[1], &ha[1]);
            }
        }
    je_flush(&b);
    je_put16(&b, 0xFFD9);
    free(pl);
    if (b.fail) { free(b.p); return NULL; }
    *outlen = b.n;
    return b.p;
}

/* ---- deterministic kernel test vectors (shared with the host golden generator) ---- */
static uint64_t sjv_next(uint64_t *s) {   /* splitmix64 */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
enum { SJV_REALISTIC = 0, SJV_DENSE = 1, SJV_EXTREME = 2, SJV_DCONLY = 3, SJV_KINDS = 4 };
/* one coefficient block (natural order) + quant table of the given kind */
static void sjv_block(uint64_t *s, int kind, int16_t *coef, uint16_t *q) {
    memset(coef, 0, 64 * sizeof *coef);
    for (int i = 0; i < 64; i++) {
        uint64_t r = sjv_next(s);
        switch (kind) {
        case SJV_REALISTIC: q[i] = (uint16_t)(1 + r % (i == 0 ? 16 : 40)); break;
        case SJV_DENSE: q[i] = (uint16_t)(1 + r % 64); break;
        case SJV_EXTREME: q[i] = (uint16_t)((r & 3) == 0 ? 65535 : 1 + (r >> 8) % 65535); break;
        default: q[i] = (uint16_t)(1 + r % 255); break;
        }
    }
    for (int k = 0; k < 64; k++) {
        uint64_t r = sjv_next(s);
        int pos = je_zz[k];
        switch (kind) {
        case SJV_REALISTIC:
            if (k == 0) coef[0] = (int16_t)((int)(r % 2048) - 1024) / q[0];
            else if ((int)(r % 80) > k + 8) coef[pos] = (int16_t)(((int)((r >> 16) % 129) - 64) * 8 / (k + 4));
            break;
        case SJV_DENSE: coef[pos] = (int16_t)((int)(r % 1024) - 512); break;
        case SJV_EXTREME: {
            int sel = (int)(r & 7);
            coef[pos] = (int16_t)(sel == 0 ? 32767 : sel == 1 ? -32768 : sel == 2 ? 0 : (int)((r >> 8) & 0xFFFF) - 32768);
            break;
        }
        default: if (k == 0) coef[0] = (int16_t)((int)(r % 4096) - 2048); break;
        }
    }
}
#endif
