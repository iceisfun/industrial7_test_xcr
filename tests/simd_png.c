/* simd_png - real-world SIMD workload: an in-tree zlib inflate + PNG decoder (tests/simd_png_dec.h)
 * with scalar / SSE2 / SSSE3 / AVX2 kernel sets for unfiltering (Sub/Up/Avg/Paeth at 1..8 bytes per
 * pixel), RGBA expansion (grey / grey+alpha / RGB / RGBA / palette, 1..16 bits, tRNS), Adler-32,
 * CRC-32 (table, slicing-by-8, PCLMULQDQ folding) and the LZ77 match copy.
 *
 * What is checked (wrong pixels or a crash are a FAIL):
 *  - Adler-32 / CRC-32 of every variant against Python zlib vectors (corpus/png/gen_png.py), published
 *    check values, and against each other for random lengths / alignments / incremental updates;
 *  - every unfilter / expand / match-copy kernel bit-identical to scalar (seeded random and tie-heavy
 *    rows, all filter types, all bytes-per-pixel classes, lengths 0..300, misaligned) and scalar against
 *    an independent reference written here; filter -> unfilter round trips;
 *  - inflate against hand-assembled and Python-zlib streams (stored / fixed / dynamic, distance 32768,
 *    length 258 via code 284, single-code and code-less distance trees, every zlib-rejected header / code
 *    description / symbol / distance error, Adler mismatch, output overflow, truncation at every byte
 *    with the exact partial output zlib produces);
 *  - PNG corpus (every colour type and bit depth, odd sizes 1x1 .. 4097 wide / tall, Adam7, every filter
 *    type, zlib levels 0/1/6/9, strategies, window sizes, split / empty IDATs, ancillary chunks, tRNS,
 *    palette out-of-range) decoded by every kernel set to the RGBA and raw-scanline hashes of the
 *    generator's independent Python decoder; corrupt input (signature, CRC, Adler, IHDR, chunk order,
 *    PLTE/tRNS, unknown critical chunks, filter type, short / long streams, truncation) gives the expected
 *    status and exactly the expected partial data;
 *  - inputs and kernel buffers that end at a PAGE_NOACCESS page (no over-read / over-write), seeded
 *    mutation fuzzing (in process and in a child process), identical results for every kernel set;
 *  - 16 and 256 threads decoding concurrently with mixed kernel sets; decodes stay exact while another
 *    thread suspends the decoders and captures / round-trips their (XSTATE) context; thread-pool callbacks;
 *  - the decoder (integer only) leaves MXCSR and the x87 environment untouched and does not depend on them.
 * PERF: decode MPix/s and inflate MB/s per kernel set (run-time encoded 1024x768 RGBA, 640x480 Adam7 RGB,
 * a Python level-9 stream) and kernel throughput. */
#include "t.h"
#include "simd_png_dec.h"
#include "simd_png_corpus.h"

#define SP_UNCHECKED 0xFFFFFFFFu

/* ============================================================================================ */
/* kernel sets                                                                                   */
/* ============================================================================================ */
typedef struct sp_tier { const pd_kernels *k; int usable; const char *why; } sp_tier;
static sp_tier g_t[4];
enum { SP_SCALAR, SP_SSE2, SP_SSSE3, SP_AVX2, SP_NT };
static void sp_tiers(void) {
    g_t[0].k = &pd_kernels_scalar; g_t[0].usable = 1; g_t[0].why = "";
    g_t[1].k = &pd_kernels_sse2; g_t[1].usable = cpu_has(F_SSE2); g_t[1].why = "SSE2 not usable";
    g_t[2].k = &pd_kernels_ssse3; g_t[2].usable = cpu_has(F_SSE2) && cpu_has(F_SSSE3); g_t[2].why = "SSSE3 not usable";
    g_t[3].k = cpu_has(F_PCLMULQDQ) ? &pd_kernels_avx2 : &pd_kernels_avx2_nopclmul;
    g_t[3].usable = cpu_has(F_AVX2) && cpu_has(F_SSSE3);
    g_t[3].why = "AVX2 not usable (CPUID bit or XCR0 YMM state missing)";
}
static int sp_usable(const pd_kernels **out) {
    int n = 0;
    sp_tiers();
    for (int i = 0; i < SP_NT; i++) if (g_t[i].usable) out[n++] = g_t[i].k;
    return n;
}
static const pd_kernels *sp_tier_or_skip(int idx) {
    sp_tiers();
    if (!g_t[idx].usable) SKIP("%s", g_t[idx].why);
    return g_t[idx].k;
}
static const sp_img *sp_find(const char *name) {
    for (size_t i = 0; i < T_COUNTOF(sp_images); i++)
        if (!strcmp(sp_images[i].name, name)) return &sp_images[i];
    return NULL;
}
static int sp_dec(const uint8_t *d, size_t n, const pd_kernels *k, int flags, uint32_t maxpx, pd_image *im) {
    pd_options o;
    o.k = k; o.max_pixels = maxpx; o.flags = flags;
    return pd_decode_png(d, n, &o, im);
}
static uint64_t sp_hpix(const pd_image *im) { return im->pix ? pd_fnv64(im->pix, im->stride * im->h) : 0; }
static uint64_t sp_fnv_upd(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
static uint8_t sp_ck_byte(uint32_t i) { return (uint8_t)((((uint32_t)(i * 2654435761u)) >> 13 ^ (i >> 8)) & 0xFF); }

TEST(kernel_sets_available) {
    sp_tiers();
    for (int i = 0; i < SP_NT; i++) t_info(g_t[i].k->name, "%s", g_t[i].usable ? "usable" : g_t[i].why);
#ifdef T_X64
    CHECKF(cpu_has(F_SSE2), "SSE2 is architectural on x86-64 but is not usable");
#endif
    t_info("crc32_pclmul", "%s", cpu_has(F_PCLMULQDQ) ? "usable (AVX2 set folds CRC-32 with PCLMULQDQ)" : "not usable");
    t_info("corpus", "%u images, %u inflate vectors, %u checksum vectors; Pillow cross-checked %d", (unsigned)T_COUNTOF(sp_images),
           (unsigned)T_COUNTOF(sp_zvecs), (unsigned)T_COUNTOF(sp_cks), SP_PIL_CHECKED);
    if (cpu_hw(F_AVX2) && !cpu_has(F_AVX2)) t_note("AVX2 advertised by CPUID but YMM state not enabled by the OS: AVX2 kernel cases SKIP");
}

/* ============================================================================================ */
/* Adler-32 / CRC-32                                                                             */
/* ============================================================================================ */
typedef uint32_t (*sp_sum_fn)(uint32_t, const uint8_t *, size_t);
typedef struct sp_sum { const char *name; sp_sum_fn fn; int usable; uint32_t init; } sp_sum;
static int sp_adlers(sp_sum *s) {
    sp_tiers();
    s[0] = (sp_sum){ "scalar", pd_adler_scalar, 1, 1 };
    s[1] = (sp_sum){ "sse2", pd_adler_sse2, g_t[1].usable, 1 };
    s[2] = (sp_sum){ "ssse3", pd_adler_ssse3, g_t[2].usable, 1 };
    s[3] = (sp_sum){ "avx2", pd_adler_avx2, g_t[3].usable, 1 };
    return 4;
}
static int sp_crcs(sp_sum *s) {
    s[0] = (sp_sum){ "bytewise", pd_crc_bytewise, 1, 0 };
    s[1] = (sp_sum){ "slice8", pd_crc_slice8, 1, 0 };
    s[2] = (sp_sum){ "pclmul", pd_crc_pclmul, cpu_has(F_PCLMULQDQ) && cpu_has(F_SSE2), 0 };
    return 3;
}
static uint8_t *g_ckbuf;
static size_t g_ckbuf_n;
static const uint8_t *sp_ck_data(int ff, size_t n) {
    size_t need = 3u << 20;
    if (!g_ckbuf) {
        g_ckbuf = (uint8_t *)malloc(need * 2);
        if (!g_ckbuf) return NULL;
        g_ckbuf_n = need;
        for (size_t i = 0; i < need; i++) g_ckbuf[i] = sp_ck_byte((uint32_t)i);
        memset(g_ckbuf + need, 0xFF, need);
    }
    return n <= g_ckbuf_n ? (ff ? g_ckbuf + need : g_ckbuf) : NULL;
}
static void sp_ck_vectors(int crc) {
    sp_sum s[4];
    int ns = crc ? sp_crcs(s) : sp_adlers(s);
    int any = 0;
    for (int v = 0; v < ns; v++) {
        if (!s[v].usable) { t_note("%s: not usable here", s[v].name); continue; }
        any++;
        int bad = 0;
        for (size_t i = 0; i < T_COUNTOF(sp_cks); i++) {
            const uint8_t *p = sp_ck_data(sp_cks[i].ff, sp_cks[i].len);
            REQUIRE(p);
            uint32_t got = s[v].fn(s[v].init, p, sp_cks[i].len), want = crc ? sp_cks[i].crc : sp_cks[i].adler;
            if (got != want && bad++ < 4)
                CHECKF(0, "%s %s(len %u%s) = 0x%08x, zlib 0x%08x", s[v].name, crc ? "crc32" : "adler32", sp_cks[i].len, sp_cks[i].ff ? ", all 0xFF" : "", got, want);
        }
        CHECKF(bad == 0, "%s: %d of %u vectors wrong", s[v].name, bad, (unsigned)T_COUNTOF(sp_cks));
        /* published check values */
        if (crc) {
            CHECK_EQ(s[v].fn(0, (const uint8_t *)"123456789", 9), 0xCBF43926u);
            CHECK_EQ(s[v].fn(0, (const uint8_t *)"The quick brown fox jumps over the lazy dog", 43), 0x414FA339u);
        } else {
            CHECK_EQ(s[v].fn(1, (const uint8_t *)"Wikipedia", 9), 0x11E60398u);
        }
    }
    t_info("variants_tested", "%d", any);
}
TEST(adler32_zlib_vectors_all_tiers) { sp_ck_vectors(0); }
TEST(crc32_zlib_vectors_all_variants) { sp_ck_vectors(1); }
TEST(crc32_pclmul_available_matches_table) {
    if (!cpu_has(F_PCLMULQDQ)) SKIP("PCLMULQDQ not usable");
    t_rng r; t_rng_seed(&r, 77);
    static uint8_t buf[70000];
    t_rng_fill(&r, buf, sizeof buf);
    int bad = 0;
    for (size_t n = 0; n < 1200; n++) {
        size_t off = n % 61;
        uint32_t a = pd_crc_bytewise(0x1234567u * (uint32_t)n, buf + off, n), b = pd_crc_pclmul(0x1234567u * (uint32_t)n, buf + off, n);
        if (a != b && bad++ < 3) CHECKF(0, "crc32 pclmul len %u off %u: 0x%08x vs table 0x%08x", (unsigned)n, (unsigned)off, b, a);
    }
    CHECK_EQ(pd_crc_pclmul(0, buf, sizeof buf), pd_crc_bytewise(0, buf, sizeof buf));
    CHECK_EQ(bad, 0);
}
/* random lengths, alignments and incremental splits: every variant equals the scalar one */
static void sp_ck_random(int crc) {
    sp_sum s[4];
    int ns = crc ? sp_crcs(s) : sp_adlers(s);
    t_rng r; t_rng_seed(&r, crc ? 31 : 32);
    static uint8_t buf[40000];
    int bad = 0;
    for (int it = 0; it < 3000; it++) {
        int mode = it % 3;
        size_t n = mode == 0 ? (size_t)t_rng_range(&r, 300) : mode == 1 ? (size_t)t_rng_range(&r, 20000) : 5552 * (1 + t_rng_range(&r, 4)) + t_rng_range(&r, 64) - 32;
        size_t off = (size_t)t_rng_range(&r, 64);
        if (it % 7 == 0) memset(buf + off, 0xFF, n); else t_rng_fill(&r, buf + off, n);
        uint32_t init = crc ? t_rng_u32(&r) : (t_rng_u32(&r) % 65521) | (t_rng_u32(&r) % 65521) << 16;
        uint32_t want = s[0].fn(init, buf + off, n);
        size_t cut = n ? (size_t)t_rng_range(&r, (uint32_t)n + 1) : 0;
        for (int v = 0; v < ns; v++) {
            if (!s[v].usable) continue;
            uint32_t got = s[v].fn(init, buf + off, n);
            uint32_t inc = s[v].fn(s[v].fn(init, buf + off, cut), buf + off + cut, n - cut);
            if ((got != want || inc != want) && bad++ < 4)
                CHECKF(0, "%s len %u off %u init 0x%08x: 0x%08x (split at %u: 0x%08x), scalar 0x%08x", s[v].name, (unsigned)n, (unsigned)off, init, got, (unsigned)cut, inc, want);
        }
    }
    CHECK_EQ(bad, 0);
}
TEST(adler32_random_lengths_alignment_incremental) { sp_ck_random(0); }
TEST(crc32_random_lengths_alignment_incremental) { sp_ck_random(1); }
TEST(checksums_guard_page_tails) {
    sp_sum a[4], c[3];
    int na = sp_adlers(a), nc = sp_crcs(c);
    int bad = 0;
    for (size_t n = 0; n <= 300; n++) {
        void *base;
        uint8_t *p = (uint8_t *)t_guarded_alloc(n ? n : 1, &base);
        REQUIRE(p);
        p += n ? 0 : 1;   /* n == 0: pointer right at the guard page */
        for (size_t i = 0; i < n; i++) p[i] = sp_ck_byte((uint32_t)(i * 7 + n));
        uint32_t wa = pd_adler_scalar(1, p, n), wc = pd_crc_bytewise(0, p, n);
        for (int v = 0; v < na; v++) if (a[v].usable && a[v].fn(1, p, n) != wa) bad++;
        for (int v = 0; v < nc; v++) if (c[v].usable && c[v].fn(0, p, n) != wc) bad++;
        t_guarded_free(base);
    }
    CHECKF(bad == 0, "%d checksum results differ on buffers that end at a guard page", bad);
}

/* ============================================================================================ */
/* unfilter                                                                                      */
/* ============================================================================================ */
/* independent reference (PNG spec 9.2-9.4, written with ints and the spec's Paeth predictor text) */
static int sp_ref_paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = p > a ? p - a : a - p, pb = p > b ? p - b : b - p, pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}
static void sp_ref_unfilter(int ft, uint8_t *x, const uint8_t *prior, size_t n, int bpp) {
    for (size_t i = 0; i < n; i++) {
        int a = i >= (size_t)bpp ? x[i - bpp] : 0, b = prior[i], c = i >= (size_t)bpp ? prior[i - bpp] : 0, pr = 0;
        switch (ft) { case 1: pr = a; break; case 2: pr = b; break; case 3: pr = (a + b) / 2; break; case 4: pr = sp_ref_paeth(a, b, c); break; default: break; }
        x[i] = (uint8_t)((x[i] + pr) & 0xFF);
    }
}
static void sp_filter_row(int ft, const uint8_t *cur, const uint8_t *prior, uint8_t *out, size_t n, int bpp) {
    for (size_t i = 0; i < n; i++) {
        int a = i >= (size_t)bpp ? cur[i - bpp] : 0, b = prior[i], c = i >= (size_t)bpp ? prior[i - bpp] : 0, pr = 0;
        switch (ft) { case 1: pr = a; break; case 2: pr = b; break; case 3: pr = (a + b) / 2; break; case 4: pr = sp_ref_paeth(a, b, c); break; default: break; }
        out[i] = (uint8_t)((cur[i] - pr) & 0xFF);
    }
}
static const int g_bpps[] = { 1, 2, 3, 4, 6, 8 };
static void sp_fill_row(t_rng *r, uint8_t *p, size_t n, int kind) {
    static const uint8_t ties[] = { 0, 1, 2, 127, 128, 129, 254, 255 };
    for (size_t i = 0; i < n; i++)
        p[i] = kind == 0 ? (uint8_t)t_rng_u32(r) : kind == 1 ? ties[t_rng_range(r, 8)] : (uint8_t)(i * 3 + (t_rng_range(r, 4)));
}
TEST(unfilter_scalar_vs_spec_reference) {
    t_rng r; t_rng_seed(&r, 5);
    uint8_t prior[400], a[400], b[400];
    int bad = 0;
    for (int it = 0; it < 4000; it++) {
        int ft = 1 + it % 4, bpp = g_bpps[(it / 4) % 6];
        size_t n = (size_t)t_rng_range(&r, 300);
        sp_fill_row(&r, prior, n, it % 3); sp_fill_row(&r, a, n, (it / 3) % 3);
        memcpy(b, a, n);
        pd_unf_scalar(ft, a, prior, n, bpp);
        sp_ref_unfilter(ft, b, prior, n, bpp);
        if (memcmp(a, b, n) && bad++ < 3) CHECKF(0, "scalar unfilter ft %d bpp %d n %u differs from the reference", ft, bpp, (unsigned)n);
    }
    CHECK_EQ(bad, 0);
}
/* filter with the reference encoder, unfilter with each kernel set: the original row comes back */
TEST(unfilter_roundtrip_all_tiers) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    t_rng r; t_rng_seed(&r, 6);
    uint8_t prior[400], cur[400], f[400];
    int bad = 0;
    for (int it = 0; it < 3000; it++) {
        int ft = 1 + it % 4, bpp = g_bpps[(it / 4) % 6];
        size_t n = (size_t)t_rng_range(&r, 320);
        sp_fill_row(&r, prior, n, it % 3); sp_fill_row(&r, cur, n, (it / 5) % 3);
        sp_filter_row(ft, cur, prior, f, n, bpp);
        for (int k = 0; k < nk; k++) {
            uint8_t t[400];
            memcpy(t, f, n);
            ks[k]->unfilter(ft, t, prior, n, bpp);
            if (memcmp(t, cur, n) && bad++ < 4) CHECKF(0, "%s: filter %d bpp %d n %u does not round-trip", ks[k]->name, ft, bpp, (unsigned)n);
        }
    }
    CHECK_EQ(bad, 0);
}
static void sp_unfilter_tier(int idx) {
    const pd_kernels *k = sp_tier_or_skip(idx);
    t_rng r; t_rng_seed(&r, 100 + idx);
    static uint8_t prior[600], a[600], b[600];
    int bad = 0, n_cmp = 0;
    for (int ft = 1; ft <= 4; ft++)
        for (int bi = 0; bi < 6; bi++)
            for (size_t n = 0; n <= 300; n++)
                for (int kind = 0; kind < 3; kind++) {
                    size_t ao = (size_t)t_rng_range(&r, 32), po = (size_t)t_rng_range(&r, 32);
                    sp_fill_row(&r, prior + po, n, kind); sp_fill_row(&r, a + ao, n, (kind + 1) % 3);
                    memcpy(b + ao, a + ao, n);
                    a[ao + n] = b[ao + n] = 0x5A;
                    pd_unf_scalar(ft, b + ao, prior + po, n, g_bpps[bi]);
                    k->unfilter(ft, a + ao, prior + po, n, g_bpps[bi]);
                    n_cmp++;
                    if ((memcmp(a + ao, b + ao, n) || a[ao + n] != 0x5A) && bad++ < 4) {
                        size_t at = 0; while (at < n && a[ao + at] == b[ao + at]) at++;
                        CHECKF(0, "%s unfilter ft %d bpp %d n %u: first difference at byte %u (got 0x%02x, scalar 0x%02x)%s", k->name, ft, g_bpps[bi],
                               (unsigned)n, (unsigned)at, at < n ? a[ao + at] : 0, at < n ? b[ao + at] : 0, a[ao + n] != 0x5A ? "; wrote past the row" : "");
                    }
                }
    t_info("rows", "%d", n_cmp);
    CHECK_EQ(bad, 0);
}
TEST(unfilter_sse2_vs_scalar)  { sp_unfilter_tier(SP_SSE2); }
TEST(unfilter_ssse3_vs_scalar) { sp_unfilter_tier(SP_SSSE3); }
TEST(unfilter_avx2_vs_scalar)  { sp_unfilter_tier(SP_AVX2); }
/* cur and prior rows each end at a PAGE_NOACCESS page: the kernels touch exactly n bytes */
TEST(unfilter_guard_page_bounds) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    t_rng r; t_rng_seed(&r, 8);
    int bad = 0;
    for (size_t n = 1; n <= 200; n++) {
        void *b1, *b2;
        uint8_t *cur = (uint8_t *)t_guarded_alloc(n, &b1), *pri = (uint8_t *)t_guarded_alloc(n, &b2);
        REQUIRE(cur && pri);
        uint8_t want[256], src[256];
        sp_fill_row(&r, pri, n, 0); sp_fill_row(&r, src, n, 0);
        for (int ft = 1; ft <= 4; ft++)
            for (int bi = 0; bi < 6; bi++) {
                memcpy(want, src, n);
                pd_unf_scalar(ft, want, pri, n, g_bpps[bi]);
                for (int k = 0; k < nk; k++) {
                    memcpy(cur, src, n);
                    ks[k]->unfilter(ft, cur, pri, n, g_bpps[bi]);
                    if (memcmp(cur, want, n)) bad++;
                }
            }
        t_guarded_free(b1); t_guarded_free(b2);
    }
    CHECKF(bad == 0, "%d guarded unfilter results differ", bad);
}

/* ============================================================================================ */
/* row expansion                                                                                 */
/* ============================================================================================ */
typedef struct sp_ffmt { int ctype, depth; } sp_ffmt;
static const sp_ffmt g_fmts[] = { { 0, 1 }, { 0, 2 }, { 0, 4 }, { 0, 8 }, { 0, 16 }, { 2, 8 }, { 2, 16 }, { 3, 1 }, { 3, 2 }, { 3, 4 }, { 3, 8 },
                                  { 4, 8 }, { 4, 16 }, { 6, 8 }, { 6, 16 } };
static size_t sp_rowbytes(int ctype, int depth, uint32_t w) { return ((size_t)w * pd_channels(ctype) * depth + 7) / 8; }
static void sp_make_fmt(t_rng *r, pd_fmt *f, int ctype, int depth, int trns) {
    memset(f, 0, sizeof *f);
    f->ctype = ctype; f->depth = depth;
    for (int i = 0; i < 256; i++) f->pal[i] = 0xFF000000u;
    if (ctype == 3) {
        int np = 1 + (int)t_rng_range(r, 1u << depth);
        for (int i = 0; i < np; i++) f->pal[i] = t_rng_u32(r) | (trns ? 0 : 0xFF000000u);
    }
    if (trns && (ctype == 0 || ctype == 2)) {
        f->has_trns = 1;
        unsigned m = (1u << depth) - 1;
        for (int i = 0; i < 3; i++) f->key[i] = (uint16_t)(t_rng_u32(r) & m & (depth == 16 ? 0x3 : 0xFFFF));
    }
}
/* reference expansion written from the PNG spec text (sample extraction by bit position) */
static unsigned sp_sample(const uint8_t *row, size_t idx, int depth) {
    if (depth == 16) return (unsigned)row[2 * idx] << 8 | row[2 * idx + 1];
    if (depth == 8) return row[idx];
    size_t bit = idx * (size_t)depth;
    return (row[bit / 8] >> (8 - depth - bit % 8)) & ((1u << depth) - 1);
}
static void sp_ref_expand(const pd_fmt *f, const uint8_t *row, uint32_t w, uint8_t *out) {
    int ch = pd_channels(f->ctype), d = f->depth;
    unsigned mx = (1u << d) - 1;
    for (uint32_t x = 0; x < w; x++) {
        unsigned s[4] = { 0 }, o[4];
        for (int c = 0; c < ch; c++) s[c] = sp_sample(row, (size_t)x * ch + c, d);
        if (f->ctype == 3) {
            memcpy(out + 4 * (size_t)x, &f->pal[s[0]], 4); continue;
        }
        unsigned omax = d == 16 ? 65535 : 255;
        unsigned sc = d == 16 ? 1 : 255 / mx;
        switch (f->ctype) {
        case 0: o[0] = o[1] = o[2] = s[0] * sc; o[3] = f->has_trns && s[0] == f->key[0] ? 0 : omax; break;
        case 2: o[0] = s[0]; o[1] = s[1]; o[2] = s[2]; o[3] = f->has_trns && s[0] == f->key[0] && s[1] == f->key[1] && s[2] == f->key[2] ? 0 : omax; break;
        case 4: o[0] = o[1] = o[2] = s[0]; o[3] = s[1]; break;
        default: o[0] = s[0]; o[1] = s[1]; o[2] = s[2]; o[3] = s[3]; break;
        }
        for (int c = 0; c < 4; c++) {
            if (d == 16) { out[8 * (size_t)x + 2 * c] = (uint8_t)o[c]; out[8 * (size_t)x + 2 * c + 1] = (uint8_t)(o[c] >> 8); }
            else out[4 * (size_t)x + c] = (uint8_t)o[c];
        }
    }
}
static void sp_expand_check(int idx, int guard) {
    const pd_kernels *k = idx < 0 ? NULL : sp_tier_or_skip(idx);
    t_rng r; t_rng_seed(&r, 200 + idx + 10 * guard);
    static pd_fmt f;
    int bad = 0, n = 0;
    for (size_t fi = 0; fi < T_COUNTOF(g_fmts); fi++)
        for (int trns = 0; trns < 2; trns++)
            for (uint32_t w = 1; w <= (guard ? 70u : 130u); w++) {
                int ct = g_fmts[fi].ctype, d = g_fmts[fi].depth;
                size_t rb = sp_rowbytes(ct, d, w), ob = (size_t)w * (d == 16 ? 8 : 4);
                void *b1 = NULL, *b2 = NULL;
                uint8_t *row, *out;
                static uint8_t rowbuf[1200], outbuf[1200], want[1200];
                if (guard) { row = (uint8_t *)t_guarded_alloc(rb, &b1); out = (uint8_t *)t_guarded_alloc(ob, &b2); REQUIRE(row && out); }
                else { row = rowbuf + (w % 13); out = outbuf + (w % 7); out[ob] = 0xA5; }
                sp_make_fmt(&r, &f, ct, d, trns);
                /* tie-heavy samples so that tRNS keys and out-of-range palette indices really occur */
                for (size_t i = 0; i < rb; i++) row[i] = (i & 3) == 0 ? (uint8_t)t_rng_u32(&r) : (uint8_t)(t_rng_range(&r, 4) * (d == 16 ? 0 : 1));
                if (ct == 2 || ct == 0) for (size_t i = 0; i + 1 < rb && d == 8; i += 3) if (t_rng_range(&r, 3) == 0) { row[i] = (uint8_t)f.key[0]; if (ct == 2 && i + 2 < rb) { row[i + 1] = (uint8_t)f.key[1]; row[i + 2] = (uint8_t)f.key[2]; } }
                sp_ref_expand(&f, row, w, want);
                if (k) k->expand(&f, row, w, out); else pd_expand_scalar(&f, row, w, out);
                n++;
                if ((memcmp(out, want, ob) || (!guard && out[ob] != 0xA5)) && bad++ < 4) {
                    size_t at = 0; while (at < ob && out[at] == want[at]) at++;
                    CHECKF(0, "%s expand ctype %d depth %d%s w %u: first difference at byte %u%s", k ? k->name : "scalar", ct, d, trns ? " tRNS" : "", w,
                           (unsigned)at, !guard && out[ob] != 0xA5 ? " (wrote past the row)" : "");
                }
                if (guard) { t_guarded_free(b1); t_guarded_free(b2); }
            }
    t_info("rows", "%d", n);
    CHECK_EQ(bad, 0);
}
TEST(expand_scalar_vs_spec_reference) { sp_expand_check(-1, 0); }
TEST(expand_sse2_vs_reference)  { sp_expand_check(SP_SSE2, 0); }
TEST(expand_ssse3_vs_reference) { sp_expand_check(SP_SSSE3, 0); }
TEST(expand_avx2_vs_reference)  { sp_expand_check(SP_AVX2, 0); }
TEST(expand_guard_page_bounds_all_tiers) {
    sp_expand_check(-1, 1);
    sp_tiers();
    for (int i = 1; i < SP_NT; i++) if (g_t[i].usable) sp_expand_check(i, 1);
}

/* ============================================================================================ */
/* LZ77 match copy                                                                               */
/* ============================================================================================ */
TEST(copy_match_tiers_vs_bytewise) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    t_rng r; t_rng_seed(&r, 9);
    int bad = 0;
    for (size_t dist = 1; dist <= 70; dist++)
        for (size_t len = 0; len <= 300; len += 1 + len / 16) {
            size_t pos = dist + (size_t)t_rng_range(&r, 40);
            void *gb;
            uint8_t *g = (uint8_t *)t_guarded_alloc(pos + len, &gb);   /* the match ends at the guard page */
            uint8_t want[700];
            REQUIRE(g);
            t_rng_fill(&r, g, pos);
            memcpy(want, g, pos);
            for (size_t i = 0; i < len; i++) want[pos + i] = want[pos + i - dist];
            for (int k = 0; k < nk; k++) {
                memset(g + pos, 0xEE, len);
                ks[k]->copy_match(g, pos, dist, len);
                if (memcmp(g, want, pos + len) && bad++ < 4) CHECKF(0, "%s copy_match dist %u len %u wrong", ks[k]->name, (unsigned)dist, (unsigned)len);
            }
            t_guarded_free(gb);
        }
    CHECK_EQ(bad, 0);
}

/* ============================================================================================ */
/* inflate                                                                                       */
/* ============================================================================================ */
static void sp_inflate_vectors(int idx) {
    const pd_kernels *k = sp_tier_or_skip(idx);
    int bad = 0, n_ok = 0, n_err = 0;
    for (size_t i = 0; i < T_COUNTOF(sp_zvecs); i++) {
        const sp_zvec *z = &sp_zvecs[i];
        void *bi, *bo;
        uint8_t *in = (uint8_t *)t_guarded_alloc(z->len ? z->len : 1, &bi);
        uint8_t *out = (uint8_t *)t_guarded_alloc(z->cap ? z->cap : 1, &bo);   /* output ends at a guard page too */
        REQUIRE(in && out);
        if (!z->len) in++;
        if (!z->cap) out++;
        memcpy(in, z->data, z->len);
        size_t prod = 0;
        int st = pd_inflate_zlib(k, in, z->len, out, z->cap, &prod, 0);
        int ok = st == z->st;
        if (ok && z->out_len != SP_UNCHECKED) ok = prod == z->out_len && pd_fnv64(out, prod) == z->h_out;
        if (!ok && bad++ < 6)
            CHECKF(0, "%s: inflate %s -> status %d (%s), %u bytes; expected %d (%s)%s%u bytes", z->name, k->name, st, pd_strerror(st), (unsigned)prod,
                   z->st, pd_strerror(z->st), z->out_len == SP_UNCHECKED ? ", any length" : ", ", z->out_len == SP_UNCHECKED ? 0 : z->out_len);
        if (z->st == PD_OK) n_ok++; else n_err++;
        t_guarded_free(bi); t_guarded_free(bo);
    }
    t_info("vectors", "%d valid, %d invalid/truncated", n_ok, n_err);
    CHECK_EQ(bad, 0);
}
TEST(inflate_vectors_scalar) { sp_inflate_vectors(SP_SCALAR); }
TEST(inflate_vectors_sse2)   { sp_inflate_vectors(SP_SSE2); }
TEST(inflate_vectors_ssse3)  { sp_inflate_vectors(SP_SSSE3); }
TEST(inflate_vectors_avx2)   { sp_inflate_vectors(SP_AVX2); }

/* ---- run-time encoder (fixed-Huffman deflate with greedy LZ77, PNG writer) for synthetic workloads */
typedef struct sp_bw { uint8_t *p; size_t n; uint64_t acc; int cnt; } sp_bw;
static void sp_put(sp_bw *w, uint32_t v, int nb) {
    w->acc |= (uint64_t)v << w->cnt; w->cnt += nb;
    while (w->cnt >= 8) { w->p[w->n++] = (uint8_t)w->acc; w->acc >>= 8; w->cnt -= 8; }
}
static void sp_huff(sp_bw *w, uint32_t code, int len) {
    uint32_t r = 0;
    for (int i = 0; i < len; i++) r |= ((code >> i) & 1u) << (len - 1 - i);
    sp_put(w, r, len);
}
static void sp_fixed_sym(sp_bw *w, int s) {
    if (s < 144) sp_huff(w, 0x30u + (unsigned)s, 8);
    else if (s < 256) sp_huff(w, 0x190u + (unsigned)(s - 144), 9);
    else if (s < 280) sp_huff(w, (unsigned)(s - 256), 7);
    else sp_huff(w, 0xC0u + (unsigned)(s - 280), 8);
}
/* returns a malloc'd zlib stream; *zn = its length */
static uint8_t *sp_deflate_fixed(const uint8_t *d, size_t n, size_t *zn) {
    sp_bw w = { (uint8_t *)malloc(n + n / 4 + 64), 0, 0, 0 };
    int32_t *head = (int32_t *)malloc(sizeof(int32_t) << 15);
    if (!w.p || !head) { free(w.p); free(head); return NULL; }
    for (int i = 0; i < (1 << 15); i++) head[i] = -1;
    w.p[w.n++] = 0x78; w.p[w.n++] = 0x01;
    sp_put(&w, 1, 1); sp_put(&w, 1, 2);
    size_t i = 0;
    while (i < n) {
        size_t best = 0, dist = 0;
        if (i + 3 <= n) {
            uint32_t h = ((uint32_t)d[i] << 16 | (uint32_t)d[i + 1] << 8 | d[i + 2]) * 2654435761u >> 17;
            int32_t c = head[h];
            head[h] = (int32_t)i;
            if (c >= 0 && i - (size_t)c <= 32768) {
                size_t m = 0, lim = n - i < 258 ? n - i : 258;
                while (m < lim && d[c + m] == d[i + m]) m++;
                if (m >= 3) { best = m; dist = i - (size_t)c; }
            }
        }
        if (!best) { sp_fixed_sym(&w, d[i]); i++; continue; }
        int li = 28;
        while (pd_lbase[li] > best) li--;
        sp_fixed_sym(&w, 257 + li);
        if (pd_lext[li]) sp_put(&w, (uint32_t)(best - pd_lbase[li]), pd_lext[li]);
        int di = 29;
        while (pd_dbase[di] > dist) di--;
        sp_huff(&w, (uint32_t)di, 5);
        if (pd_dext[di]) sp_put(&w, (uint32_t)(dist - pd_dbase[di]), pd_dext[di]);
        i += best;
    }
    sp_fixed_sym(&w, 256);
    if (w.cnt) sp_put(&w, 0, 8 - w.cnt);
    uint32_t a = pd_adler_scalar(1, d, n);
    for (int s = 24; s >= 0; s -= 8) w.p[w.n++] = (uint8_t)(a >> s);
    free(head);
    *zn = w.n;
    return w.p;
}
static size_t sp_chunk(uint8_t *o, const char *type, const uint8_t *data, uint32_t len) {
    o[0] = (uint8_t)(len >> 24); o[1] = (uint8_t)(len >> 16); o[2] = (uint8_t)(len >> 8); o[3] = (uint8_t)len;
    memcpy(o + 4, type, 4);
    if (len) memcpy(o + 8, data, len);
    uint32_t c = pd_crc_bytewise(0, o + 4, (size_t)len + 4);
    o[8 + len] = (uint8_t)(c >> 24); o[9 + len] = (uint8_t)(c >> 16); o[10 + len] = (uint8_t)(c >> 8); o[11 + len] = (uint8_t)c;
    return (size_t)len + 12;
}
/* synthetic image (8-bit samples): gradients, a few flat areas and sparse noise; filtered with every
 * filter type (row-cycled) and fixed-Huffman deflated. Returns the PNG and the expected RGBA. */
typedef struct sp_synth { uint8_t *png; size_t len; uint8_t *rgba; uint32_t w, h; int ctype, interlace; uint8_t *z; size_t zn; size_t raw_len; } sp_synth;
static int sp_synth_make(sp_synth *s, uint32_t w, uint32_t h, int ctype, int interlace) {
    memset(s, 0, sizeof *s);
    s->w = w; s->h = h; s->ctype = ctype; s->interlace = interlace;
    int ch = pd_channels(ctype);
    uint8_t *px = (uint8_t *)malloc((size_t)w * h * ch);
    s->rgba = (uint8_t *)malloc((size_t)w * h * 4);
    if (!px || !s->rgba) { free(px); return 0; }
    t_rng r; t_rng_seed(&r, w * 31 + h);
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++)
            for (int c = 0; c < ch; c++) {
                unsigned v = (x * (2 + c) + y * (3 - c / 2) + ((x / 64 + y / 48) & 1) * 40) & 255;
                if (((x >> 5) ^ (y >> 5)) % 7 == 3) v = 200 - c * 30;   /* flat blocks */
                else if (t_rng_range(&r, 5) == 0) v = (v + t_rng_range(&r, 9)) & 255;
                px[((size_t)y * w + x) * ch + c] = (uint8_t)v;
            }
    for (size_t i = 0; i < (size_t)w * h; i++) {
        const uint8_t *p = px + i * ch;
        uint8_t *o = s->rgba + 4 * i;
        if (ch == 1) { o[0] = o[1] = o[2] = p[0]; o[3] = 255; }
        else if (ch == 3) { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; o[3] = 255; }
        else if (ch == 2) { o[0] = o[1] = o[2] = p[0]; o[3] = p[1]; }
        else memcpy(o, p, 4);
    }
    /* filtered stream */
    size_t raw_cap = (size_t)(w * ch + 1) * h + 7 * ((size_t)h + 1) + 64;
    uint8_t *raw = (uint8_t *)malloc(raw_cap), *prev = (uint8_t *)calloc((size_t)w * ch + 1, 1), *cur = (uint8_t *)malloc((size_t)w * ch + 1);
    if (!raw || !prev || !cur) { free(px); free(raw); free(prev); free(cur); return 0; }
    size_t rn = 0;
    int row = 0;
    for (int ps = 0; ps < (interlace ? 7 : 1); ps++) {
        uint32_t pw = interlace ? pd_pass_dim(w, pd_a7_x0[ps], pd_a7_dx[ps]) : w, ph = interlace ? pd_pass_dim(h, pd_a7_y0[ps], pd_a7_dy[ps]) : h;
        if (!pw || !ph) continue;
        size_t rb = (size_t)pw * ch;
        memset(prev, 0, rb);
        for (uint32_t yy = 0; yy < ph; yy++, row++) {
            uint32_t y = interlace ? pd_a7_y0[ps] + yy * pd_a7_dy[ps] : yy;
            for (uint32_t i = 0; i < pw; i++) memcpy(cur + (size_t)i * ch, px + ((size_t)y * w + (interlace ? pd_a7_x0[ps] + i * pd_a7_dx[ps] : i)) * ch, ch);
            int ft = row % 5;
            raw[rn++] = (uint8_t)ft;
            sp_filter_row(ft, cur, prev, raw + rn, rb, ch);
            rn += rb;
            memcpy(prev, cur, rb);
        }
    }
    free(px); free(prev); free(cur);
    s->raw_len = rn;
    s->z = sp_deflate_fixed(raw, rn, &s->zn);
    free(raw);
    if (!s->z) return 0;
    s->png = (uint8_t *)malloc(s->zn + 128);
    if (!s->png) return 0;
    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    uint8_t ih[13] = { (uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w, (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h,
                       8, (uint8_t)ctype, 0, 0, (uint8_t)interlace };
    memcpy(s->png, sig, 8);
    size_t n = 8;
    n += sp_chunk(s->png + n, "IHDR", ih, 13);
    n += sp_chunk(s->png + n, "IDAT", s->z, (uint32_t)s->zn);
    n += sp_chunk(s->png + n, "IEND", NULL, 0);
    s->len = n;
    return 1;
}
static void sp_synth_free(sp_synth *s) { free(s->png); free(s->rgba); free(s->z); memset(s, 0, sizeof *s); }
static sp_synth g_syn[2];
static int g_syn_state[2];
static const sp_synth *sp_synth_get(int which) {
    if (!g_syn_state[which]) {
        g_syn_state[which] = which == 0 ? (sp_synth_make(&g_syn[0], 1024, 768, 6, 0) ? 1 : -1) : (sp_synth_make(&g_syn[1], 640, 480, 2, 1) ? 1 : -1);
    }
    return g_syn_state[which] > 0 ? &g_syn[which] : NULL;
}
TEST(inflate_runtime_fixed_huffman_roundtrip) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    t_rng r; t_rng_seed(&r, 10);
    static uint8_t src[50000], out[50001];
    int bad = 0;
    for (int it = 0; it < 60; it++) {
        size_t n = (size_t)t_rng_range(&r, sizeof src);
        int kind = it % 3;
        for (size_t i = 0; i < n; i++) src[i] = kind == 0 ? (uint8_t)t_rng_u32(&r) : kind == 1 ? (uint8_t)("abcab"[t_rng_range(&r, 5)]) : (uint8_t)(i / 300);
        size_t zn;
        uint8_t *z = sp_deflate_fixed(src, n, &zn);
        REQUIRE(z);
        for (int k = 0; k < nk; k++) {
            size_t prod;
            int st = pd_inflate_zlib(ks[k], z, zn, out, n, &prod, 0);
            if ((st != PD_OK || prod != n || memcmp(out, src, n)) && bad++ < 4) CHECKF(0, "%s: round trip of %u bytes: status %d, %u bytes", ks[k]->name, (unsigned)n, st, (unsigned)prod);
        }
        free(z);
    }
    CHECK_EQ(bad, 0);
}

/* ============================================================================================ */
/* PNG corpus                                                                                    */
/* ============================================================================================ */
/* 0 = as expected; otherwise writes a reason */
static int sp_check_img(const sp_img *c, const uint8_t *data, const pd_kernels *k, char *why, size_t cap) {
    pd_image im;
    int rc = sp_dec(data, c->len, k, c->flags | PD_F_KEEP_RAW, c->max_pixels, &im);
    int bad = 0;
    if (rc != c->rc) { snprintf(why, cap, "status %d (%s), expected %d (%s)", rc, pd_strerror(rc), c->rc, pd_strerror(c->rc)); bad = 1; }
    else if (c->rc == PD_OK) {
        if (im.w != c->w || im.h != c->h) { snprintf(why, cap, "size %ux%u, expected %ux%u", im.w, im.h, c->w, c->h); bad = 1; }
        else if (im.raw_complete != im.raw_len || pd_fnv64(im.raw, im.raw_complete) != c->h_raw) { snprintf(why, cap, "unfiltered scanlines differ from the reference"); bad = 1; }
        else if (sp_hpix(&im) != c->h_pix) { snprintf(why, cap, "RGBA pixels differ from the reference (raw scanlines match: expansion)"); bad = 1; }
    } else {
        if (c->raw_produced != SP_UNCHECKED && im.raw_produced != c->raw_produced) {
            snprintf(why, cap, "inflated %u bytes, reference %u", (unsigned)im.raw_produced, c->raw_produced); bad = 1;
        } else if (c->raw_complete != SP_UNCHECKED && (im.raw_complete != c->raw_complete || pd_fnv64(im.raw, im.raw_complete) != c->h_raw)) {
            snprintf(why, cap, "partial scanlines: %u bytes (reference %u) %s", (unsigned)im.raw_complete, c->raw_complete,
                     im.raw_complete == c->raw_complete ? "with different content" : ""); bad = 1;
        }
        if (im.pix) { snprintf(why, cap, "pixels returned with an error status"); bad = 1; }
    }
    pd_free(&im);
    return bad;
}
static void sp_corpus(int idx, int errors) {
    const pd_kernels *k = sp_tier_or_skip(idx);
    int bad = 0, n = 0;
    char why[200];
    for (size_t i = 0; i < T_COUNTOF(sp_images); i++) {
        const sp_img *c = &sp_images[i];
        if ((c->rc != PD_OK) != errors) continue;
        n++;
        if (sp_check_img(c, c->data, k, why, sizeof why)) { if (bad++ < 8) CHECKF(0, "%s [%s]: %s", c->name, k->name, why); }
    }
    t_info("images", "%d", n);
    CHECKF(bad == 0, "%d of %d images decoded differently from the reference", bad, n);
}
TEST(corpus_valid_scalar)  { sp_corpus(SP_SCALAR, 0); }
TEST(corpus_valid_sse2)    { sp_corpus(SP_SSE2, 0); }
TEST(corpus_valid_ssse3)   { sp_corpus(SP_SSSE3, 0); }
TEST(corpus_valid_avx2)    { sp_corpus(SP_AVX2, 0); }
TEST(corpus_corrupt_scalar) { sp_corpus(SP_SCALAR, 1); }
TEST(corpus_corrupt_sse2)   { sp_corpus(SP_SSE2, 1); }
TEST(corpus_corrupt_ssse3)  { sp_corpus(SP_SSSE3, 1); }
TEST(corpus_corrupt_avx2)   { sp_corpus(SP_AVX2, 1); }
/* every corpus file decoded from a buffer that ends at a guard page, with every kernel set */
TEST(corpus_guard_page_input_all_tiers) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    int bad = 0;
    char why[200];
    for (size_t i = 0; i < T_COUNTOF(sp_images); i++) {
        const sp_img *c = &sp_images[i];
        void *gb;
        uint8_t *d = (uint8_t *)t_guarded_alloc(c->len ? c->len : 1, &gb);
        REQUIRE(d);
        if (!c->len) d++;
        memcpy(d, c->data, c->len);
        for (int k = 0; k < nk; k++)
            if (sp_check_img(c, d, ks[k], why, sizeof why) && bad++ < 6) CHECKF(0, "%s [%s, guarded]: %s", c->name, ks[k]->name, why);
        t_guarded_free(gb);
    }
    CHECK_EQ(bad, 0);
}
/* every prefix of three files: never a crash, status SIG / TRUNC, the partial scanlines are a prefix
 * of the full image's scanlines, and every kernel set agrees */
static void sp_every_prefix(const char *name) {
    const sp_img *c = sp_find(name);
    REQUIRE(c);
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    pd_image full;
    REQUIRE(sp_dec(c->data, c->len, NULL, PD_F_KEEP_RAW, 0, &full) == PD_OK);
    int bad = 0;
    for (uint32_t n = 0; n < c->len; n++) {
        void *gb;
        uint8_t *d = (uint8_t *)t_guarded_alloc(n ? n : 1, &gb);
        REQUIRE(d);
        if (!n) d++;
        memcpy(d, c->data, n);
        int rc0 = 0; size_t rp0 = 0, rc0n = 0; uint64_t h0 = 0;
        for (int k = 0; k < nk; k++) {
            pd_image im;
            int rc = sp_dec(d, n, ks[k], PD_F_KEEP_RAW, 0, &im);
            uint64_t h = pd_fnv64(im.raw, im.raw_complete);
            if (k == 0) {
                rc0 = rc; rp0 = im.raw_produced; rc0n = im.raw_complete; h0 = h;
                int want = n < 8 ? PD_E_SIG : PD_E_TRUNC;
                if (rc != want && bad++ < 4) CHECKF(0, "%s prefix %u: status %d (%s), expected %d", name, n, rc, pd_strerror(rc), want);
                if (im.raw_complete && (im.raw_complete > full.raw_len || memcmp(im.raw, full.raw, im.raw_complete)) && bad++ < 4)
                    CHECKF(0, "%s prefix %u: partial scanlines are not a prefix of the full image's", name, n);
            } else if ((rc != rc0 || im.raw_produced != rp0 || im.raw_complete != rc0n || h != h0) && bad++ < 4) {
                CHECKF(0, "%s prefix %u: %s differs from %s", name, n, ks[k]->name, ks[0]->name);
            }
            pd_free(&im);
        }
        t_guarded_free(gb);
    }
    pd_free(&full);
    CHECK_EQ(bad, 0);
}
TEST(trunc_every_prefix_rgb8) { sp_every_prefix("base_rgb8_29x13"); }
TEST(trunc_every_prefix_adam7) { sp_every_prefix("base_rgba8_31x17_i"); }
TEST(trunc_every_prefix_palette_split_idat) { sp_every_prefix("split7_ga16_19x5_i"); sp_every_prefix("base_p4_45x7"); }

/* seeded mutations (CRC checking off so that they reach inflate / unfilter): identical results for every
 * kernel set, inputs at a guard page, deterministic */
static const char *const g_fuzz_bases[] = { "base_rgb8_29x13", "base_rgba8_31x17_i", "base_p4_45x7", "fmt_g16_17x9", "fmt_ga8_17x13_i",
                                            "filtbest_bpp6_57x9_i", "zstrat_fixed_rgb8_64x40", "zlvl0_g1_33x7_i", "trns_pal8_partial_50x10",
                                            "split1_rgb8_23x11", "fmt_p1_33x4", "filt4_bpp8_41x6" };
typedef struct sp_fuzz_stats { int n, ok, err, mismatch; uint64_t agg; } sp_fuzz_stats;
static uint64_t sp_next(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s * 0x2545F4914F6CDD1Dull; }
static size_t sp_mutate(const uint8_t *base, size_t blen, uint64_t *s, uint8_t *out) {
    size_t n = blen;
    memcpy(out, base, n);
    int op = (int)(sp_next(s) % 9);
    size_t lo = blen > 41 ? 33 : 8;   /* mostly hit the data after IHDR */
    size_t pos = lo + (size_t)(sp_next(s) % (n - lo));
    switch (op) {
    case 0: out[pos] ^= (uint8_t)(1u << (sp_next(s) & 7)); break;
    case 1: for (int i = 0, m = 1 + (int)(sp_next(s) % 8); i < m; i++) out[lo + sp_next(s) % (n - lo)] ^= (uint8_t)(1u << (sp_next(s) & 7)); break;
    case 2: out[pos] = (uint8_t)sp_next(s); break;
    case 3: { size_t k = 1 + (size_t)(sp_next(s) % 24); if (pos + k > n) k = n - pos; memmove(out + pos, out + pos + k, n - pos - k); n -= k; break; }
    case 4: { size_t k = 1 + (size_t)(sp_next(s) % 40); if (pos + k > n) k = n - pos; memmove(out + pos + k, out + pos, n - pos); n += k; break; }
    case 5: n = 1 + (size_t)(sp_next(s) % blen); break;
    case 6: out[8 + sp_next(s) % 25] = (uint8_t)sp_next(s); break;    /* IHDR fields */
    case 7: { size_t k = 1 + (size_t)(sp_next(s) % 16); if (pos + k > n) k = n - pos; memset(out + pos, (int)(sp_next(s) & 1 ? 0xFF : 0), k); break; }
    default: out[pos] = (uint8_t)(out[pos] + 1 - 2 * (int)(sp_next(s) & 1)); break;
    }
    return n;
}
static void sp_fuzz_one(const pd_kernels *const *ks, int nk, const uint8_t *base, size_t blen, uint64_t seed, sp_fuzz_stats *st) {
    static uint8_t buf[70000];
    if (blen + 64 > sizeof buf) return;
    uint64_t s = seed | 1;
    size_t n = sp_mutate(base, blen, &s, buf);
    void *gb;
    uint8_t *d = (uint8_t *)t_guarded_alloc(n, &gb);
    if (!d) return;
    memcpy(d, buf, n);
    int flags = PD_F_KEEP_RAW | PD_F_IGNORE_CRC | (seed % 5 == 0 ? PD_F_IGNORE_ADLER : 0);
    uint64_t rec0[5] = { 0 };
    for (int k = 0; k < nk; k++) {
        pd_image im;
        int rc = sp_dec(d, n, ks[k], flags, 1u << 20, &im);
        uint64_t rec[5] = { (uint64_t)(int64_t)rc, im.raw_produced, im.raw_complete, pd_fnv64(im.raw, im.raw_complete), sp_hpix(&im) };
        if (k == 0) { memcpy(rec0, rec, sizeof rec); if (rc) st->err++; else st->ok++; }
        else if (memcmp(rec, rec0, sizeof rec)) st->mismatch++;
        pd_free(&im);
    }
    st->agg = sp_fnv_upd(st->agg, rec0, sizeof rec0);
    st->n++;
    t_guarded_free(gb);
}
static void sp_fuzz_run(uint64_t seed, int count, sp_fuzz_stats *st) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    memset(st, 0, sizeof *st);
    st->agg = 0xcbf29ce484222325ull;
    for (int i = 0; i < count; i++) {
        const sp_img *c = sp_find(g_fuzz_bases[i % T_COUNTOF(g_fuzz_bases)]);
        if (c) sp_fuzz_one(ks, nk, c->data, c->len, seed * 1000003u + (uint64_t)i, st);
    }
}
TEST(fuzz_mutations_all_tiers_agree) {
    sp_fuzz_stats a, b;
    double t0 = t_now_ms();
    sp_fuzz_run(1, 1500, &a);
    t_info("mutations", "%d: %d decoded, %d errors, %.0f ms", a.n, a.ok, a.err, t_now_ms() - t0);
    CHECK_EQ(a.n, 1500);
    CHECKF(a.mismatch == 0, "%d mutations decoded differently by some kernel set", a.mismatch);
    sp_fuzz_run(1, 1500, &b);
    CHECKF(a.agg == b.agg, "fuzz results not deterministic");
}
CHILD(fuzz) {
    uint64_t seed = argc > 1 ? strtoull(argv[1], NULL, 0) : 5;
    int count = argc > 2 ? atoi(argv[2]) : 100;
    sp_fuzz_stats st;
    sp_fuzz_run(seed, count, &st);
    printf("FUZZ_N=%d\nFUZZ_MISMATCH=%d\nFUZZ_HASH=%016llx\n", st.n, st.mismatch, (unsigned long long)st.agg);
    fflush(stdout);
    return 0;
}
TEST(fuzz_child_process_consistent) {
    static t_child_result r1;
    REQUIREF(t_child_run("fuzz", "77 800", 120000, &r1), "child spawn failed (error %lu)", (unsigned long)r1.spawn_error);
    CHECKF(!r1.timed_out, "fuzz child timed out");
    CHECKF(r1.exit_code == 0 && r1.exc_code == 0, "fuzz child exit 0x%lx exception 0x%lx at 0x%llx", (unsigned long)r1.exit_code,
           (unsigned long)r1.exc_code, (unsigned long long)r1.exc_addr);
    char n[32] = "", mis[32] = "", h1[32] = "";
    CHECK(t_child_find(&r1, "FUZZ_N=", n, sizeof n) && atoi(n) == 800);
    CHECK(t_child_find(&r1, "FUZZ_MISMATCH=", mis, sizeof mis) && atoi(mis) == 0);
    CHECK(t_child_find(&r1, "FUZZ_HASH=", h1, sizeof h1));
    sp_fuzz_stats st;
    sp_fuzz_run(77, 800, &st);
    char me[32];
    snprintf(me, sizeof me, "%016llx", (unsigned long long)st.agg);
    CHECKF(!strcmp(h1, me), "child %s vs in-process %s", h1, me);
    t_info("child_ms", "%.0f", r1.ms);
}

/* ============================================================================================ */
/* API contract and corpus files                                                                 */
/* ============================================================================================ */
TEST(api_invalid_arguments) {
    const sp_img *c = sp_find("base_rgb8_29x13");
    pd_image im;
    CHECK_EQ(pd_decode_png(c->data, c->len, NULL, NULL), PD_E_ARG);
    CHECK_EQ(pd_decode_png(NULL, 10, NULL, &im), PD_E_ARG);
    CHECK_EQ(pd_decode_png(NULL, 0, NULL, &im), PD_E_SIG);
    CHECK(im.pix == NULL && im.raw == NULL);
    size_t prod = 1;
    uint8_t o[4];
    CHECK_EQ(pd_inflate_zlib(NULL, NULL, 0, o, 4, &prod, 0), PD_ZE_IN);
    CHECK_EQ(prod, 0);
}
TEST(api_max_pixels_limit) {
    const sp_img *c = sp_find("mt_rgb8_160x120");
    pd_image im;
    CHECK_EQ(sp_dec(c->data, c->len, NULL, 0, 160 * 120 - 1, &im), PD_E_LIMIT);
    CHECK(im.pix == NULL);
    CHECK_EQ(sp_dec(c->data, c->len, NULL, 0, 160 * 120, &im), PD_OK);
    CHECK(sp_hpix(&im) == c->h_pix);
    pd_free(&im);
}
TEST(api_image_metadata) {
    const sp_img *c = sp_find("wide_rgba16_4097x1");
    pd_image im;
    REQUIRE(sp_dec(c->data, c->len, NULL, 0, 0, &im) == PD_OK);
    CHECK(im.w == 4097 && im.h == 1 && im.depth == 16 && im.ctype == 6 && im.bps == 2 && im.stride == 4097 * 8 && im.interlace == 0);
    pd_free(&im);
    c = sp_find("split1_rgb8_23x11");
    REQUIRE(sp_dec(c->data, c->len, NULL, 0, 0, &im) == PD_OK);
    CHECKF(im.n_idat > 100, "1-byte IDAT chunks: %u", im.n_idat);
    pd_free(&im);
}
TEST(corpus_files_match_embedded) {
    char dir[MAX_PATH * 2] = "", path[MAX_PATH * 2];
    static const char *const rel[] = { "..\\..\\corpus\\png\\", "corpus\\png\\", "..\\corpus\\png\\" };
    char base[MAX_PATH * 2];
    snprintf(base, sizeof base, "%s", t_self_path());
    char *sl = strrchr(base, '\\'); if (!sl) sl = strrchr(base, '/');
    if (sl) sl[1] = 0; else base[0] = 0;
    for (size_t i = 0; i < T_COUNTOF(rel) && !dir[0]; i++) {
        snprintf(path, sizeof path, "%s%sbase_rgb8_29x13.png", base, rel[i]);
        if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) snprintf(dir, sizeof dir, "%s%s", base, rel[i]);
    }
    if (!dir[0]) SKIP("corpus/png not found next to the exe (embedded copies are used)");
    int found = 0;
    for (size_t i = 0; i < T_COUNTOF(sp_images); i++) {
        const sp_img *c = &sp_images[i];
        if (!strncmp(c->name, "trunc_", 6)) continue;   /* prefixes of the base files, not stored */
        snprintf(path, sizeof path, "%s%s.png", dir, c->name);
        FILE *f = fopen(path, "rb");
        if (!f) { CHECKF(0, "%s missing", path); continue; }
        uint8_t *b = (uint8_t *)malloc(c->len + 1);
        size_t n = b ? fread(b, 1, c->len + 1, f) : 0;
        fclose(f);
        CHECKF(n == c->len && b && !memcmp(b, c->data, c->len), "%s differs from the embedded copy", c->name);
        free(b);
        found++;
    }
    t_info("files", "%d", found);
}
TEST(synth_decode_matches_source_all_tiers) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    for (int s = 0; s < 2; s++) {
        const sp_synth *sy = sp_synth_get(s);
        REQUIRE(sy);
        for (int k = 0; k < nk; k++) {
            pd_image im;
            int rc = sp_dec(sy->png, sy->len, ks[k], 0, 0, &im);
            CHECKF(rc == PD_OK && im.pix && !memcmp(im.pix, sy->rgba, (size_t)sy->w * sy->h * 4), "%s: synthetic %ux%u%s: status %d or wrong pixels",
                   ks[k]->name, sy->w, sy->h, sy->interlace ? " Adam7" : "", rc);
            pd_free(&im);
        }
    }
}

/* ============================================================================================ */
/* multi-threaded decode                                                                         */
/* ============================================================================================ */
static const char *const g_mt_names[] = { "mt_rgba8_96x64_i", "mt_rgb8_160x120", "mt_p8_200x100", "mt_rgba16_64x48", "mt_g8_96x96_lvl1",
                                          "trns_rgb16_key_40x8_i", "wide_g1_4097x2", "fmt_p2_17x13_i" };
typedef struct sp_mt_shared { const sp_img *img[8]; int nimg; const pd_kernels *ks[SP_NT]; int nk; int iters;
                              HANDLE go; volatile LONG bad, decodes, failed_rc; } sp_mt_shared;
typedef struct sp_mt_arg { sp_mt_shared *sh; int tid; } sp_mt_arg;
static int sp_mt_one(const sp_img *c, const pd_kernels *k) {
    pd_image im;
    int rc = sp_dec(c->data, c->len, k, 0, 0, &im);
    int r = rc ? 2 : sp_hpix(&im) != c->h_pix ? 1 : 0;
    pd_free(&im);
    return r;
}
static DWORD WINAPI sp_mt_worker(void *p) {
    sp_mt_arg *a = (sp_mt_arg *)p;
    sp_mt_shared *sh = a->sh;
    WaitForSingleObject(sh->go, 30000);
    for (int it = 0; it < sh->iters; it++) {
        int r = sp_mt_one(sh->img[(a->tid + it) % sh->nimg], sh->ks[(a->tid / sh->nimg + it) % sh->nk]);
        if (r == 2) InterlockedIncrement(&sh->failed_rc); else if (r) InterlockedIncrement(&sh->bad);
        InterlockedIncrement(&sh->decodes);
        if (it & 1) SwitchToThread();
    }
    return 0;
}
static void sp_mt_case(int nthreads, int iters, int nimg) {
    static sp_mt_shared sh;
    memset((void *)&sh, 0, sizeof sh);
    sh.nk = sp_usable(sh.ks); sh.iters = iters; sh.nimg = nimg;
    for (int i = 0; i < nimg; i++) { sh.img[i] = sp_find(g_mt_names[i]); REQUIRE(sh.img[i]); }
    sh.go = CreateEventA(NULL, TRUE, FALSE, NULL);
    REQUIRE(sh.go);
    HANDLE *th = (HANDLE *)calloc((size_t)nthreads, sizeof(HANDLE));
    sp_mt_arg *args = (sp_mt_arg *)calloc((size_t)nthreads, sizeof(sp_mt_arg));
    REQUIRE(th && args);
    int created = 0;
    for (int i = 0; i < nthreads; i++) {
        args[i].sh = &sh; args[i].tid = i;
        th[i] = CreateThread(NULL, 256 * 1024, sp_mt_worker, &args[i], STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (!th[i]) { t_note("CreateThread failed after %d threads (error %lu)", created, GetLastError()); break; }
        created++;
    }
    if (created < 2) {
        SetEvent(sh.go);
        for (int i = 0; i < created; i++) { WaitForSingleObject(th[i], 60000); CloseHandle(th[i]); }
        CloseHandle(sh.go); free(th); free(args);
        SKIP("could not create 2 threads");
    }
    t_info("threads", "%d of %d", created, nthreads);
    double t0 = t_now_ms();
    SetEvent(sh.go);
    int hung = 0;
    for (int i = 0; i < created; i++) {
        double left = t0 + 120000 - t_now_ms();
        if (WaitForSingleObject(th[i], left > 0 ? (DWORD)left : 0) != WAIT_OBJECT_0) hung++;
        else CloseHandle(th[i]);
    }
    REQUIREF(hung == 0, "%d decoder threads did not finish within 120 s", hung);
    CloseHandle(sh.go);
    free(th); free(args);
    t_info("decodes", "%ld in %.0f ms", (long)sh.decodes, t_now_ms() - t0);
    CHECK_EQ(sh.decodes, (LONG)(created * iters));
    CHECKF(sh.failed_rc == 0, "%ld concurrent decodes failed", (long)sh.failed_rc);
    CHECKF(sh.bad == 0, "%ld of %ld concurrent decodes produced different pixels", (long)sh.bad, (long)sh.decodes);
}
TEST(mt_decode_16_threads) { sp_mt_case(16, 60, 8); }
TEST(mt_decode_256_threads) { sp_mt_case(256, 8, 6); }

/* decode while another thread keeps suspending the decoders and capturing (and round-tripping) their
 * context. Documented contract: SuspendThread / GetThreadContext / SetThreadContext / ResumeThread on a
 * thread of the same process; the context set is exactly the one read (floating-point + XSTATE parts
 * only). The controller makes no heap or CRT calls while a worker is suspended. */
typedef struct sp_sus_shared { const sp_img *img[8]; int nimg; const pd_kernels *ks[SP_NT]; int nk;
                               volatile LONG stop, bad, failed, decodes, per[8]; } sp_sus_shared;
static sp_sus_shared g_sus;
static DWORD WINAPI sp_sus_worker(void *p) {
    int tid = (int)(intptr_t)p;
    for (int it = 0; !g_sus.stop; it++) {
        int r = sp_mt_one(g_sus.img[(tid + it) % g_sus.nimg], g_sus.ks[(tid + it / g_sus.nimg) % g_sus.nk]);
        if (r == 2) InterlockedIncrement(&g_sus.failed); else if (r) InterlockedIncrement(&g_sus.bad);
        InterlockedIncrement(&g_sus.decodes);
        InterlockedIncrement(&g_sus.per[tid]);
    }
    return 0;
}
typedef BOOL (WINAPI *sp_initctx_fn)(PVOID, DWORD, PCONTEXT *, PDWORD);
typedef BOOL (WINAPI *sp_setxmask_fn)(PCONTEXT, DWORD64);
typedef DWORD64 (WINAPI *sp_getxen_fn)(void);
#ifndef CONTEXT_XSTATE
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)
#endif
#ifdef T_X64
#define SP_CTX_FP (CONTEXT_FLOATING_POINT)
#else
#define SP_CTX_FP (CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS)
#endif
static void sp_suspend_case(int roundtrip) {
    memset((void *)&g_sus, 0, sizeof g_sus);
    g_sus.nk = sp_usable(g_sus.ks);
    g_sus.nimg = 5;
    for (int i = 0; i < g_sus.nimg; i++) { g_sus.img[i] = sp_find(g_mt_names[i]); REQUIRE(g_sus.img[i]); }
    /* XSTATE-aware context (InitializeContext + SetXStateFeaturesMask, documented for AVX state) when the
     * OS enables AVX / AVX-512 state, so a round trip covers the whole vector state */
    sp_getxen_fn gx = (sp_getxen_fn)t_proc("kernel32.dll", "GetEnabledXStateFeatures");
    sp_initctx_fn ic = (sp_initctx_fn)t_proc("kernel32.dll", "InitializeContext");
    sp_setxmask_fn sm = (sp_setxmask_fn)t_proc("kernel32.dll", "SetXStateFeaturesMask");
    DWORD64 xmask = gx ? gx() & 0xE4ull : 0;
    static CONTEXT plain T_ALIGN(16);
    CONTEXT *ctx = &plain;
    void *xbuf = NULL;
    DWORD getflags = CONTEXT_FULL | SP_CTX_FP, setflags = SP_CTX_FP;
    if (xmask) {
        DWORD len = 0;
        if (!ic || !sm) {
            if (roundtrip) SKIP("XSTATE enabled (mask 0x%llx) but InitializeContext/SetXStateFeaturesMask missing", (unsigned long long)xmask);
        } else {
            ic(NULL, CONTEXT_FULL | SP_CTX_FP | CONTEXT_XSTATE, NULL, &len);
            xbuf = len ? malloc(len) : NULL;
            CONTEXT *c2 = NULL;
            if (xbuf && ic(xbuf, CONTEXT_FULL | SP_CTX_FP | CONTEXT_XSTATE, &c2, &len) && sm(c2, xmask)) {
                ctx = c2; getflags = CONTEXT_FULL | SP_CTX_FP | CONTEXT_XSTATE; setflags = SP_CTX_FP | CONTEXT_XSTATE;
            } else {
                t_note("InitializeContext(CONTEXT_XSTATE) failed (error %lu)", GetLastError());
                free(xbuf); xbuf = NULL;
                if (roundtrip) SKIP("no XSTATE-aware CONTEXT: a full-state round trip is impossible");
            }
        }
    }
    t_info("context", "%s, xstate mask 0x%llx", ctx == &plain ? "legacy CONTEXT" : "InitializeContext XSTATE", (unsigned long long)(ctx == &plain ? 0 : xmask));
    enum { NW = 4 };
    HANDLE th[NW]; int created = 0;
    for (int i = 0; i < NW; i++) {
        th[i] = CreateThread(NULL, 0, sp_sus_worker, (void *)(intptr_t)i, 0, NULL);
        if (!th[i]) { t_note("CreateThread failed after %d (error %lu)", created, GetLastError()); break; }
        created++;
    }
    if (created < 1) { free(xbuf); SKIP("could not create a decode thread"); }
    long ops = 0, sus_fail = 0, get_fail = 0, set_fail = 0, res_fail = 0, first_get_err = 0;
    double t0 = t_now_ms();
    for (int round = 0;; round++) {
        double el = t_now_ms() - t0;
        int all = 1;
        for (int i = 0; i < created; i++) all &= g_sus.per[i] >= 3;
        if ((el > 1500 && all) || el > 15000) break;
        for (int i = 0; i < created; i++) {
            if (SuspendThread(th[i]) == (DWORD)-1) { sus_fail++; continue; }
            ctx->ContextFlags = getflags;
            if (!GetThreadContext(th[i], ctx)) { if (!get_fail++) first_get_err = (long)GetLastError(); }
            else if (roundtrip) {
                ctx->ContextFlags = setflags;
                if (!SetThreadContext(th[i], ctx)) set_fail++;
            }
            if (ResumeThread(th[i]) == (DWORD)-1) res_fail++;
            ops++;
        }
        if (round & 1) SwitchToThread();
    }
    g_sus.stop = 1;
    int hung = 0;
    for (int i = 0; i < created; i++) { if (WaitForSingleObject(th[i], 60000) != WAIT_OBJECT_0) hung++; else CloseHandle(th[i]); }
    free(xbuf);
    REQUIREF(hung == 0, "%d decode threads did not stop", hung);
    t_info("suspend_cycles", "%ld in %.0f ms, %ld decodes on %d threads", ops, t_now_ms() - t0, (long)g_sus.decodes, created);
    CHECKF(sus_fail == 0 && res_fail == 0, "SuspendThread failed %ld, ResumeThread failed %ld times", sus_fail, res_fail);
    CHECKF(get_fail == 0, "GetThreadContext(flags 0x%lx) failed %ld of %ld times (first error %ld)", (unsigned long)getflags, get_fail, ops, first_get_err);
    if (roundtrip) CHECKF(set_fail == 0, "SetThreadContext(flags 0x%lx) of the context just read failed %ld of %ld times", (unsigned long)setflags, set_fail, ops);
    CHECKF(g_sus.failed == 0, "%ld decodes returned an error", (long)g_sus.failed);
    CHECKF(g_sus.bad == 0, "%ld of %ld decodes produced wrong pixels while being suspended/context-%s", (long)g_sus.bad, (long)g_sus.decodes, roundtrip ? "round-tripped" : "captured");
}
TEST(mt_decode_while_suspended_getcontext) { sp_suspend_case(0); }
TEST(mt_decode_while_context_roundtrip)    { sp_suspend_case(1); }

typedef struct sp_tp_shared { const sp_img *img; const pd_kernels *ks[SP_NT]; int nk; volatile LONG next, left, bad; HANDLE done; } sp_tp_shared;
static sp_tp_shared g_tp;
static void CALLBACK sp_tp_cb(PTP_CALLBACK_INSTANCE inst, void *ctx) {
    (void)inst; (void)ctx;
    LONG i = InterlockedIncrement(&g_tp.next) - 1;
    if (sp_mt_one(g_tp.img, g_tp.ks[i % g_tp.nk])) InterlockedIncrement(&g_tp.bad);
    if (InterlockedDecrement(&g_tp.left) == 0) SetEvent(g_tp.done);
}
TEST(threadpool_callbacks_decode) {
    memset((void *)&g_tp, 0, sizeof g_tp);
    g_tp.img = sp_find("mt_p8_200x100");
    g_tp.nk = sp_usable(g_tp.ks);
    g_tp.done = CreateEventA(NULL, TRUE, FALSE, NULL);
    REQUIRE(g_tp.img && g_tp.done);
    enum { N = 160 };
    g_tp.left = N;
    int submitted = 0;
    for (int i = 0; i < N; i++) {
        if (!TrySubmitThreadpoolCallback(sp_tp_cb, NULL, NULL)) { t_note("TrySubmitThreadpoolCallback failed after %d (error %lu)", submitted, GetLastError()); break; }
        submitted++;
    }
    if (submitted < N && InterlockedExchangeAdd(&g_tp.left, -(N - submitted)) == N - submitted) SetEvent(g_tp.done);
    if (submitted == 0) { CloseHandle(g_tp.done); SKIP("thread pool unavailable"); }
    REQUIREF(WaitForSingleObject(g_tp.done, 60000) == WAIT_OBJECT_0, "thread-pool decodes did not finish within 60 s (%ld left)", (long)g_tp.left);
    CloseHandle(g_tp.done);
    t_info("callbacks", "%d", submitted);
    CHECKF(g_tp.bad == 0, "%ld of %d callback decodes wrong", (long)g_tp.bad, submitted);
}

/* ============================================================================================ */
/* FP environment                                                                                */
/* ============================================================================================ */
/* the decoder is integer-only: integer SIMD and GPR instructions neither read MXCSR / the x87 control
 * word nor set their status flags (SDM Vol. 1 10.2.3, 11.5; x87 state is untouched by non-x87 code) */
TEST(fpenv_decode_preserves_mxcsr_and_x87) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    const sp_img *c = sp_find("mt_rgba8_96x64_i");
    unsigned m0 = t_get_mxcsr() & ~0x3Fu;
    for (int k = 0; k < nk; k++) {
        t_fnclex();
        t_set_mxcsr(m0);
        unsigned cw0 = t_get_x87_cw(), sw0 = t_get_x87_sw(), tw0 = t_get_x87_tw();
        CHECK_EQ(sp_mt_one(c, ks[k]), 0);
        unsigned m1 = t_get_mxcsr();
        CHECKF(m1 == m0, "%s: MXCSR 0x%x -> 0x%x across an integer-only decode", ks[k]->name, m0, m1);
        CHECKF(t_get_x87_cw() == cw0 && t_get_x87_sw() == sw0 && t_get_x87_tw() == tw0, "%s: x87 CW/SW/TW 0x%x/0x%x/0x%x -> 0x%x/0x%x/0x%x", ks[k]->name,
               cw0, sw0, tw0, t_get_x87_cw(), t_get_x87_sw(), t_get_x87_tw());
    }
}
/* odd FP modes must not matter: FTZ|DAZ (where MXCSR_MASK allows DAZ), every rounding mode, all MXCSR
 * exceptions unmasked (any SSE FP arithmetic with a flag would fault), x87 24-bit precision + round up */
TEST(fpenv_results_independent_of_fp_modes) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    static uint8_t fx[512] T_ALIGN(16);
    t_fxsave(fx);
    uint32_t mmask; memcpy(&mmask, fx + 28, 4);
    if (!mmask) mmask = 0xFFBF;
    unsigned m0 = t_get_mxcsr(), cw0 = t_get_x87_cw();
    unsigned modes[] = { 0x8040, 0x6000, 0x2000, 0x4000, 0x0000 /* all exceptions unmasked */ };
    if (!(mmask & 0x40)) { modes[0] = 0x8000; t_note("MXCSR_MASK 0x%x: DAZ not supported, FTZ only", mmask); }
    int bad = 0;
    for (size_t mi = 0; mi < T_COUNTOF(modes); mi++)
        for (int k = 0; k < nk; k++)
            for (int i = 0; i < 4; i++) {
                const sp_img *c = sp_find(g_mt_names[i]);
                unsigned m = mi == 4 ? (m0 & ~0x1FBFu) : ((m0 & ~0xE040u) | modes[mi]);
                t_set_mxcsr(m & ~0x3Fu);
                t_set_x87_cw((cw0 & ~0xF00u) | 0x800 /* RC up */ | 0x000 /* PC 24-bit */);
                int r = sp_mt_one(c, ks[k]);
                t_set_x87_cw(cw0);
                t_set_mxcsr(m0);
                if (r && bad++ < 4) CHECKF(0, "%s %s under MXCSR 0x%x: %s", ks[k]->name, c->name, m & ~0x3Fu, r == 2 ? "error" : "wrong pixels");
            }
    CHECK_EQ(bad, 0);
}

/* ============================================================================================ */
/* PERF                                                                                          */
/* ============================================================================================ */
typedef struct sp_bench { const uint8_t *d; size_t n; const pd_kernels *k; uint64_t want; int bad; uint8_t *out; size_t cap; } sp_bench;
static void sp_bench_decode(void *p) {
    sp_bench *b = (sp_bench *)p;
    pd_image im;
    if (sp_dec(b->d, b->n, b->k, 0, 0, &im) != PD_OK || (b->want && sp_hpix(&im) != b->want)) b->bad++;
    T_SINK(im.pix ? im.pix[im.stride] : 0);
    pd_free(&im);
}
static void sp_bench_inflate(void *p) {
    sp_bench *b = (sp_bench *)p;
    size_t prod;
    if (pd_inflate_zlib(b->k, b->d, b->n, b->out, b->cap, &prod, 0) != PD_OK || prod != b->cap) b->bad++;
    T_SINK(b->out[prod / 2]);
}
static void sp_perf_decode(int idx, int which) {
    const pd_kernels *k = sp_tier_or_skip(idx);
    const sp_synth *s = sp_synth_get(which);
    REQUIRE(s);
    sp_bench b = { s->png, s->len, k, 0, 0, NULL, 0 };
    pd_image im;
    REQUIRE(sp_dec(s->png, s->len, k, 0, 0, &im) == PD_OK);
    REQUIREF(!memcmp(im.pix, s->rgba, (size_t)s->w * s->h * 4), "decode result wrong: refusing to report a number");
    b.want = sp_hpix(&im);
    pd_free(&im);
    t_stats st;
    CHECK(t_bench(sp_bench_decode, &b, 1, 7, 0.0, &st) == 7 && b.bad == 0);
    char m[64];
    snprintf(m, sizeof m, "decode_%s_%ux%u_%s", which ? "rgb8_adam7" : "rgba8", s->w, s->h, k->name);
    t_perf(m, (double)s->w * s->h / 1e6 / st.median, "MPix/s");
    t_info("median_ms", "%.2f (min %.2f max %.2f)", st.median * 1e3, st.min * 1e3, st.max * 1e3);
}
TEST(perf_decode_rgba8_scalar) { sp_perf_decode(SP_SCALAR, 0); }
TEST(perf_decode_rgba8_sse2)   { sp_perf_decode(SP_SSE2, 0); }
TEST(perf_decode_rgba8_ssse3)  { sp_perf_decode(SP_SSSE3, 0); }
TEST(perf_decode_rgba8_avx2)   { sp_perf_decode(SP_AVX2, 0); }
TEST(perf_decode_adam7_scalar) { sp_perf_decode(SP_SCALAR, 1); }
TEST(perf_decode_adam7_sse2)   { sp_perf_decode(SP_SSE2, 1); }
TEST(perf_decode_adam7_ssse3)  { sp_perf_decode(SP_SSSE3, 1); }
TEST(perf_decode_adam7_avx2)   { sp_perf_decode(SP_AVX2, 1); }
/* inflate only: run-time fixed-Huffman stream (3 MB) and a Python zlib level-9 dynamic stream */
static const uint8_t *sp_idat_of(const sp_img *c, size_t *zn, uint8_t **owned) {
    size_t pos = 8, tot = 0;
    uint8_t *z = (uint8_t *)malloc(c->len);
    if (!z) return NULL;
    while (pos + 12 <= c->len) {
        uint32_t len = pd_be32(c->data + pos);
        if (!memcmp(c->data + pos + 4, "IDAT", 4)) { memcpy(z + tot, c->data + pos + 8, len); tot += len; }
        pos += (size_t)len + 12;
    }
    *zn = tot; *owned = z;
    return z;
}
static void sp_perf_inflate(int idx) {
    const pd_kernels *k = sp_tier_or_skip(idx);
    const sp_synth *s = sp_synth_get(0);
    REQUIRE(s);
    sp_bench b = { s->z, s->zn, k, 0, 0, (uint8_t *)malloc(s->raw_len), s->raw_len };
    REQUIRE(b.out);
    sp_bench_inflate(&b);
    REQUIREF(b.bad == 0, "inflate failed: refusing to report a number");
    t_stats st;
    CHECK(t_bench(sp_bench_inflate, &b, 1, 7, 0.0, &st) == 7 && b.bad == 0);
    char m[64];
    snprintf(m, sizeof m, "inflate_fixed_%s", k->name);
    t_perf(m, (double)s->raw_len / 1e6 / st.median, "MB/s");
    free(b.out);
    /* dynamic Huffman (Python zlib -9) */
    const sp_img *c = sp_find("mt_rgb8_160x120");
    uint8_t *owned; size_t zn;
    REQUIRE(sp_idat_of(c, &zn, &owned));
    size_t rawn = (size_t)(160 * 3 + 1) * 120;
    sp_bench b2 = { owned, zn, k, 0, 0, (uint8_t *)malloc(rawn), rawn };
    REQUIRE(b2.out);
    sp_bench_inflate(&b2);
    REQUIRE(b2.bad == 0);
    CHECK(t_bench(sp_bench_inflate, &b2, 2, 9, 2.0, &st) == 9 && b2.bad == 0);
    snprintf(m, sizeof m, "inflate_dynamic_%s", k->name);
    t_perf(m, (double)rawn / 1e6 / st.median, "MB/s");
    free(b2.out); free(owned);
}
TEST(perf_inflate_scalar) { sp_perf_inflate(SP_SCALAR); }
TEST(perf_inflate_sse2)   { sp_perf_inflate(SP_SSE2); }
TEST(perf_inflate_ssse3)  { sp_perf_inflate(SP_SSSE3); }
TEST(perf_inflate_avx2)   { sp_perf_inflate(SP_AVX2); }

typedef struct sp_kb { const pd_kernels *k; int what, ft, bpp; uint8_t *a, *b, *o; size_t n; pd_fmt *f; } sp_kb;
enum { SP_KB_UNF, SP_KB_ADLER, SP_KB_CRC, SP_KB_EXP };
#define SP_KB_N (1u << 16)
static void sp_kb_fn(void *p) {
    sp_kb *b = (sp_kb *)p;
    switch (b->what) {
    case SP_KB_UNF: for (int r = 0; r < 4; r++) b->k->unfilter(b->ft, b->a, b->b, b->n, b->bpp); break;
    case SP_KB_ADLER: T_SINK(b->k->adler32(1, b->a, b->n)); break;
    case SP_KB_CRC: T_SINK(b->k->crc32(0, b->a, b->n)); break;
    default: b->k->expand(b->f, b->a, (uint32_t)(b->n / (b->f->ctype == 2 ? 3 : 1)), b->o); break;
    }
    T_SINK(b->a[7]);
}
static void sp_perf_kernel(int what, int ft, int bpp, int ctype, const char *metric, double reps_per_call) {
    const pd_kernels *ks[SP_NT]; int nk = sp_usable(ks);
    static pd_fmt f;
    sp_kb b;
    memset(&b, 0, sizeof b);
    b.n = SP_KB_N; b.what = what; b.ft = ft; b.bpp = bpp; b.f = &f;
    b.a = (uint8_t *)malloc(SP_KB_N + 64); b.b = (uint8_t *)malloc(SP_KB_N + 64); b.o = (uint8_t *)malloc(SP_KB_N * 4 + 64);
    REQUIRE(b.a && b.b && b.o);
    t_rng r; t_rng_seed(&r, 4);
    t_rng_fill(&r, b.a, SP_KB_N); t_rng_fill(&r, b.b, SP_KB_N);
    sp_make_fmt(&r, &f, ctype, 8, 0);
    double base = 0;
    for (int k = 0; k < nk; k++) {
        b.k = ks[k];
        t_stats st;
        CHECK(t_bench(sp_kb_fn, &b, 2, 9, 2.0, &st) == 9);
        double v = (double)SP_KB_N * reps_per_call / 1e6 / st.median;
        char m[64];
        snprintf(m, sizeof m, "%s_%s", metric, ks[k]->name);
        t_perf(m, v, "MB/s");
        if (k == 0) base = v; else { char key[48]; snprintf(key, sizeof key, "speedup_%s", ks[k]->name); t_info(key, "%.2fx", v / base); }
    }
    free(b.a); free(b.b); free(b.o);
}
TEST(perf_kernel_unfilter_sub_bpp4)   { sp_perf_kernel(SP_KB_UNF, 1, 4, 0, "unfilter_sub_bpp4", 4); }
TEST(perf_kernel_unfilter_up)         { sp_perf_kernel(SP_KB_UNF, 2, 3, 0, "unfilter_up", 4); }
TEST(perf_kernel_unfilter_avg_bpp3)   { sp_perf_kernel(SP_KB_UNF, 3, 3, 0, "unfilter_avg_bpp3", 4); }
TEST(perf_kernel_unfilter_paeth_bpp4) { sp_perf_kernel(SP_KB_UNF, 4, 4, 0, "unfilter_paeth_bpp4", 4); }
TEST(perf_kernel_unfilter_paeth_bpp1) { sp_perf_kernel(SP_KB_UNF, 4, 1, 0, "unfilter_paeth_bpp1", 4); }
TEST(perf_kernel_adler32)             { sp_perf_kernel(SP_KB_ADLER, 0, 0, 0, "adler32", 1); }
TEST(perf_kernel_crc32)               { sp_perf_kernel(SP_KB_CRC, 0, 0, 0, "crc32", 1); }
TEST(perf_kernel_expand_rgb8)         { sp_perf_kernel(SP_KB_EXP, 0, 0, 2, "expand_rgb8_in", 1); }
TEST(perf_kernel_expand_palette8)     { sp_perf_kernel(SP_KB_EXP, 0, 0, 3, "expand_pal8_in", 1); }
