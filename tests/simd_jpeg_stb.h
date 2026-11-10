/* simd_jpeg_stb.h - OPTIONAL second, independent in-process JPEG decoder for simd_jpeg (guidance section 6:
 * "stb_image as independent reference").
 *
 * Vendored source: third_party/stb/stb_image.h, stb_image v2.30 (public domain / MIT, see the end of
 * that file), unmodified, from https://github.com/nothings/stb commit
 * 013ac3beddff3dbffafd5177e7972067cd2b5083, sha256 594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3.
 *
 * Compile-time optional, to keep licensing and size under control:
 *   - build with -DSJ_NO_STB (e.g. make CFLAGS="-O2 -DSJ_NO_STB"), or delete third_party/stb/, and
 *     SJ_HAVE_STB is 0: the stb_* cases of simd_jpeg then SKIP with that reason.
 *
 * Only the JPEG loader is compiled (STBI_ONLY_JPEG, from memory, no stdio). stb_image picks its kernels
 * (integer IDCT, h2v2 fancy upsampling, YCbCr->RGBA) once per decode in stbi__setup_jpeg() through
 * stbi__sse2_available(). Its own comments state the SSE2 IDCT "produces bit-identical results to the
 * generic C version"; the SSE2 h2v2 upsampler and YCbCr->RGB (step 4 only) are integer re-implementations
 * of the same formulas. To run BOTH paths from one translation unit without editing the vendored file, the
 * macro below reroutes the call site to sj_stb_sse2_wanted(), which reads a per-thread switch:
 *     definition  static int stbi__sse2_available(void)  -> static int stbi__sse2_available_implvoid() (unused)
 *     call site   stbi__sse2_available()                 -> stbi__sse2_available_impl()  (our switch)
 * On GCC the SSE2 path is compiled whenever -msse2 is on (always for x64; this suite builds x86 with
 * -msse2 -mstackrealign, which is exactly the condition stb wants for STBI_MINGW_ENABLE_SSE2).
 * The "scalar" path is stb's portable C code compiled for the baseline ISA (the compiler may still use
 * SSE2 for it, as for any C code here).
 *
 * The stb decoder's JPEG path is integer-only (IDCT 12-bit fixed point, YCbCr 20-bit fixed point), so its
 * output is a pure function of the input bytes: golden hashes computed on the corpus host
 * (corpus/jpeg/gen_stb_refs.c -> tests/simd_jpeg_stb_ref.h) must be reproduced exactly on any correct
 * x86 CPU + OS. */
#ifndef SIMD_JPEG_STB_H
#define SIMD_JPEG_STB_H

#if !defined(SJ_NO_STB) && defined(__has_include)
#if __has_include("../third_party/stb/stb_image.h")
#define SJ_HAVE_STB 1
#endif
#endif
#ifndef SJ_HAVE_STB
#define SJ_HAVE_STB 0
#endif

#if SJ_HAVE_STB
#include <stdint.h>
#include <stdlib.h>

/* per-thread kernel choice: 0 = stb portable C kernels, 1 = stb SSE2 kernels (when compiled in) */
static __thread int sj_stb_simd_tls;
/* stb internal invariant failures (STBI_ASSERT): counted, never abort the process */
static volatile long sj_stb_assert_count;
static void sj_stb_assert_fail(void) { __sync_fetch_and_add(&sj_stb_assert_count, 1); }
static int stbi__sse2_available_impl(void) { return sj_stb_simd_tls; }
#define stbi__sse2_available(...) stbi__sse2_available_impl##__VA_ARGS__()

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_MINGW_ENABLE_SSE2
#define STBI_MAX_DIMENSIONS 8192          /* the 8K synthetic image is 7680x4320; bounds a crafted header's allocation */
#define STBI_ASSERT(x) ((x) ? (void)0 : sj_stb_assert_fail())
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wold-style-declaration"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#include "../third_party/stb/stb_image.h"
#pragma GCC diagnostic pop
#undef stbi__sse2_available

#ifdef STBI_SSE2
#define SJ_STB_HAVE_SSE2 1
#else
#define SJ_STB_HAVE_SSE2 0
#endif
#define SJ_STB_VERSION "stb_image v2.30 (nothings/stb 013ac3b)"

/* decode with the chosen stb kernel set; req = 0 (native components), 1, 3 or 4. Returns the pixels
 * (free with stbi_image_free) or NULL; *w, *h, *nc (components in the file) and *outc (per pixel) */
static uint8_t *sj_stb_decode(const uint8_t *d, size_t n, int simd, int req, int *w, int *h, int *nc, int *outc) {
    int prev = sj_stb_simd_tls;
    sj_stb_simd_tls = simd;
    *w = *h = *nc = 0;
    uint8_t *p = n > 0x7FFFFFFF ? NULL : stbi_load_from_memory(d, (int)n, w, h, nc, req);
    sj_stb_simd_tls = prev;
    if (outc) *outc = req ? req : *nc;
    return p;
}
#endif /* SJ_HAVE_STB */
#endif
