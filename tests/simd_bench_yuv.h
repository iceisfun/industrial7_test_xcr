/* simd_bench_yuv.h - planar RGB <-> YCbCr (BT.601 full range, 8-bit fixed point) for simd_bench.
 * The integer formulas below ARE the reference; the SSE2 and AVX2 versions must match the scalar code
 * bit for bit (pmaddwd keeps every intermediate in 32 bits, packssdw/packuswb implement the clamp):
 *   Y  = clamp((  77 R + 150 G +  29 B + 128) >> 8)
 *   Cb = clamp(( -43 R -  85 G + 128 B + 32896) >> 8)        32896 = 128*256 + 128
 *   Cr = clamp(( 128 R - 107 G -  21 B + 32896) >> 8)
 *   R  = clamp(Y + ((359 Cr' + 128) >> 8)),  G = clamp(Y + ((-88 Cb' - 183 Cr' + 128) >> 8)),
 *   B  = clamp(Y + ((454 Cb' + 128) >> 8)),  Cb' = Cb - 128, Cr' = Cr - 128, >> arithmetic. */
#ifndef SIMD_BENCH_YUV_H
#define SIMD_BENCH_YUV_H
#include "simd_bench_util.h"

typedef struct yuv_img { const uint8_t *a, *b, *c; uint8_t *x, *y, *z; size_t n; } yuv_img;   /* n multiple of 32 */
typedef void (*yuv_fn)(const yuv_img *im);

static inline uint8_t yuv_clamp(int v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
static BM_NOVEC void rgb2yuv_scalar(const yuv_img *im) {
    for (size_t i = 0; i < im->n; i++) {
        int r = im->a[i], g = im->b[i], b = im->c[i];
        im->x[i] = yuv_clamp((77 * r + 150 * g + 29 * b + 128) >> 8);
        im->y[i] = yuv_clamp((-43 * r - 85 * g + 128 * b + 32896) >> 8);
        im->z[i] = yuv_clamp((128 * r - 107 * g - 21 * b + 32896) >> 8);
    }
}
static BM_NOVEC void yuv2rgb_scalar(const yuv_img *im) {
    for (size_t i = 0; i < im->n; i++) {
        int y = im->a[i], cb = im->b[i] - 128, cr = im->c[i] - 128;
        im->x[i] = yuv_clamp(y + ((359 * cr + 128) >> 8));
        im->y[i] = yuv_clamp(y + ((-88 * cb - 183 * cr + 128) >> 8));
        im->z[i] = yuv_clamp(y + ((454 * cb + 128) >> 8));
    }
}
#define YUV_PAIR(lo, hi) _mm_set1_epi32((int)((uint32_t)(uint16_t)(int16_t)(lo) | (uint32_t)(uint16_t)(int16_t)(hi) << 16))
#define YUV_PAIR256(lo, hi) _mm256_set1_epi32((int)((uint32_t)(uint16_t)(int16_t)(lo) | (uint32_t)(uint16_t)(int16_t)(hi) << 16))

/* --- SSE2: 8 pixels (words) -> 8 words of (u.v pairs . c_uv + w.0 pairs . c_w + off) >> 8 (+ add) */
static inline __m128i yuv_sse2_8(__m128i u, __m128i v, __m128i w, __m128i cuv, __m128i cw, __m128i off, __m128i add) {
    __m128i z = _mm_setzero_si128();
    __m128i lo = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(u, v), cuv), _mm_madd_epi16(_mm_unpacklo_epi16(w, z), cw));
    __m128i hi = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(u, v), cuv), _mm_madd_epi16(_mm_unpackhi_epi16(w, z), cw));
    lo = _mm_add_epi32(_mm_srai_epi32(_mm_add_epi32(lo, off), 8), _mm_unpacklo_epi16(add, z));
    hi = _mm_add_epi32(_mm_srai_epi32(_mm_add_epi32(hi, off), 8), _mm_unpackhi_epi16(add, z));
    return _mm_packs_epi32(lo, hi);
}
static void rgb2yuv_sse2(const yuv_img *im) {
    const __m128i z = _mm_setzero_si128();
    const __m128i cy = YUV_PAIR(77, 150), cyb = YUV_PAIR(29, 0), oy = _mm_set1_epi32(128);
    const __m128i cu = YUV_PAIR(-43, -85), cub = YUV_PAIR(128, 0), cv = YUV_PAIR(128, -107), cvb = YUV_PAIR(-21, 0), oc = _mm_set1_epi32(32896);
    for (size_t i = 0; i < im->n; i += 16) {
        __m128i R = _mm_loadu_si128((const __m128i *)(im->a + i)), G = _mm_loadu_si128((const __m128i *)(im->b + i)), B = _mm_loadu_si128((const __m128i *)(im->c + i));
        __m128i Rl = _mm_unpacklo_epi8(R, z), Rh = _mm_unpackhi_epi8(R, z), Gl = _mm_unpacklo_epi8(G, z), Gh = _mm_unpackhi_epi8(G, z);
        __m128i Bl = _mm_unpacklo_epi8(B, z), Bh = _mm_unpackhi_epi8(B, z);
        _mm_storeu_si128((__m128i *)(im->x + i), _mm_packus_epi16(yuv_sse2_8(Rl, Gl, Bl, cy, cyb, oy, z), yuv_sse2_8(Rh, Gh, Bh, cy, cyb, oy, z)));
        _mm_storeu_si128((__m128i *)(im->y + i), _mm_packus_epi16(yuv_sse2_8(Rl, Gl, Bl, cu, cub, oc, z), yuv_sse2_8(Rh, Gh, Bh, cu, cub, oc, z)));
        _mm_storeu_si128((__m128i *)(im->z + i), _mm_packus_epi16(yuv_sse2_8(Rl, Gl, Bl, cv, cvb, oc, z), yuv_sse2_8(Rh, Gh, Bh, cv, cvb, oc, z)));
    }
}
static void yuv2rgb_sse2(const yuv_img *im) {
    const __m128i z = _mm_setzero_si128(), k128 = _mm_set1_epi16(128), o = _mm_set1_epi32(128);
    const __m128i cr = YUV_PAIR(0, 359), cg = YUV_PAIR(-88, -183), cb = YUV_PAIR(454, 0);
    for (size_t i = 0; i < im->n; i += 16) {
        __m128i Y = _mm_loadu_si128((const __m128i *)(im->a + i)), U = _mm_loadu_si128((const __m128i *)(im->b + i)), V = _mm_loadu_si128((const __m128i *)(im->c + i));
        __m128i Yl = _mm_unpacklo_epi8(Y, z), Yh = _mm_unpackhi_epi8(Y, z);
        __m128i Ul = _mm_sub_epi16(_mm_unpacklo_epi8(U, z), k128), Uh = _mm_sub_epi16(_mm_unpackhi_epi8(U, z), k128);
        __m128i Vl = _mm_sub_epi16(_mm_unpacklo_epi8(V, z), k128), Vh = _mm_sub_epi16(_mm_unpackhi_epi8(V, z), k128);
        _mm_storeu_si128((__m128i *)(im->x + i), _mm_packus_epi16(yuv_sse2_8(Ul, Vl, z, cr, z, o, Yl), yuv_sse2_8(Uh, Vh, z, cr, z, o, Yh)));
        _mm_storeu_si128((__m128i *)(im->y + i), _mm_packus_epi16(yuv_sse2_8(Ul, Vl, z, cg, z, o, Yl), yuv_sse2_8(Uh, Vh, z, cg, z, o, Yh)));
        _mm_storeu_si128((__m128i *)(im->z + i), _mm_packus_epi16(yuv_sse2_8(Ul, Vl, z, cb, z, o, Yl), yuv_sse2_8(Uh, Vh, z, cb, z, o, Yh)));
    }
}

/* --- AVX2: 16 pixels in words (vpmovzxbw keeps order); in-lane unpack + in-lane pack restore the order,
 * the final vpackuswb of two 16-pixel halves is fixed up with vpermq (3,1,2,0). */
static T_TARGET("avx2") inline __m256i yuv_avx2_16(__m256i u, __m256i v, __m256i w, __m256i cuv, __m256i cw, __m256i off, __m256i add) {
    __m256i z = _mm256_setzero_si256();
    __m256i lo = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(u, v), cuv), _mm256_madd_epi16(_mm256_unpacklo_epi16(w, z), cw));
    __m256i hi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(u, v), cuv), _mm256_madd_epi16(_mm256_unpackhi_epi16(w, z), cw));
    lo = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(lo, off), 8), _mm256_unpacklo_epi16(add, z));
    hi = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(hi, off), 8), _mm256_unpackhi_epi16(add, z));
    return _mm256_packs_epi32(lo, hi);
}
#define YUV_LD16(p) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(p)))
#define YUV_ST32(p, w0, w1) _mm256_storeu_si256((__m256i *)(p), _mm256_permute4x64_epi64(_mm256_packus_epi16(w0, w1), 0xD8))
static T_TARGET("avx2") void rgb2yuv_avx2(const yuv_img *im) {
    const __m256i z = _mm256_setzero_si256();
    const __m256i cy = YUV_PAIR256(77, 150), cyb = YUV_PAIR256(29, 0), oy = _mm256_set1_epi32(128);
    const __m256i cu = YUV_PAIR256(-43, -85), cub = YUV_PAIR256(128, 0), cv = YUV_PAIR256(128, -107), cvb = YUV_PAIR256(-21, 0), oc = _mm256_set1_epi32(32896);
    for (size_t i = 0; i < im->n; i += 32) {
        __m256i R0 = YUV_LD16(im->a + i), R1 = YUV_LD16(im->a + i + 16), G0 = YUV_LD16(im->b + i), G1 = YUV_LD16(im->b + i + 16);
        __m256i B0 = YUV_LD16(im->c + i), B1 = YUV_LD16(im->c + i + 16);
        YUV_ST32(im->x + i, yuv_avx2_16(R0, G0, B0, cy, cyb, oy, z), yuv_avx2_16(R1, G1, B1, cy, cyb, oy, z));
        YUV_ST32(im->y + i, yuv_avx2_16(R0, G0, B0, cu, cub, oc, z), yuv_avx2_16(R1, G1, B1, cu, cub, oc, z));
        YUV_ST32(im->z + i, yuv_avx2_16(R0, G0, B0, cv, cvb, oc, z), yuv_avx2_16(R1, G1, B1, cv, cvb, oc, z));
    }
    _mm256_zeroupper();
}
static T_TARGET("avx2") void yuv2rgb_avx2(const yuv_img *im) {
    const __m256i z = _mm256_setzero_si256(), k128 = _mm256_set1_epi16(128), o = _mm256_set1_epi32(128);
    const __m256i cr = YUV_PAIR256(0, 359), cg = YUV_PAIR256(-88, -183), cb = YUV_PAIR256(454, 0);
    for (size_t i = 0; i < im->n; i += 32) {
        __m256i Y0 = YUV_LD16(im->a + i), Y1 = YUV_LD16(im->a + i + 16);
        __m256i U0 = _mm256_sub_epi16(YUV_LD16(im->b + i), k128), U1 = _mm256_sub_epi16(YUV_LD16(im->b + i + 16), k128);
        __m256i V0 = _mm256_sub_epi16(YUV_LD16(im->c + i), k128), V1 = _mm256_sub_epi16(YUV_LD16(im->c + i + 16), k128);
        YUV_ST32(im->x + i, yuv_avx2_16(U0, V0, z, cr, z, o, Y0), yuv_avx2_16(U1, V1, z, cr, z, o, Y1));
        YUV_ST32(im->y + i, yuv_avx2_16(U0, V0, z, cg, z, o, Y0), yuv_avx2_16(U1, V1, z, cg, z, o, Y1));
        YUV_ST32(im->z + i, yuv_avx2_16(U0, V0, z, cb, z, o, Y0), yuv_avx2_16(U1, V1, z, cb, z, o, Y1));
    }
    _mm256_zeroupper();
}
#endif
