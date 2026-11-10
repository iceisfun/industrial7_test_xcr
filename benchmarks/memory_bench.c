/* memory_bench - memcpy / memset bandwidth across a 4 KB .. 256 MB size sweep (docs/GUIDANCE.md section 5).
 *
 * Kernels (tests/simd_bench_mem.h, shared with simd_bench, plus a GPR-word scalar loop defined here):
 *   copy: scalar (native-word GPR loop), crt (msvcrt memcpy through a volatile pointer), sse2 (MOVDQU),
 *         sse2_nt (MOVNTDQ + SFENCE, destination aligned to 16), avx (YMM), avx512 (ZMM), rep_movsb
 *   fill: scalar, crt, sse2, sse2_nt, avx, avx512 (masked tail, needs AVX512BW), rep_stosb
 * Every case sweeps the size (powers of 4 from 4 KB to 256 MB; SIMD_BENCH_MODE=full adds every power of 2)
 * twice: "aligned" (source and destination page aligned) and "misaligned" (source +1, destination +3).
 * Copies stop at 128 MB: source + destination of 256 MB each would exceed the suite's 512 MB budget.
 * GB/s counts the bytes copied or filled once (not read + write traffic).
 *
 * Cache-resident vs DRAM: each row gets an INFO "<metric>.level" = L1 / L2 / L3 / DRAM, from the working
 * set (2n for a copy, n for a fill) against the cache sizes Windows reports
 * (GetLogicalProcessorInformation; CPUID leaf 4 / 0x8000001D only when the OS reports none). The level
 * is a label for the reader, never a check.
 *
 * Pass/fail: slowness never fails. A case FAILs only when a kernel's output is wrong (bytes differ, or
 * a guard byte before/after the destination changed) before or after timing, or when a documented API
 * contract the report relies on is broken (GetLogicalProcessorInformation buffer negotiation: MSDN
 * "If the buffer is not large enough ... GetLastError returns ERROR_INSUFFICIENT_BUFFER", ReturnedLength
 * receives the required size). Feature absent -> SKIP. Measurement methodology (pinning, warm-up,
 * median of >= 9 samples, quartiles in the JSON): tests/simd_bench_util.h. */
#include "simd_bench_util.h"
#include "simd_bench_mem.h"

/* ============================================================================ scalar GPR kernels */
/* native-word loads/stores; no vectorisation and no memcpy/memset idiom recognition */
#define MB_SCALAR __attribute__((noinline, optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-tree-loop-distribute-patterns")))
typedef size_t __attribute__((may_alias, aligned(1))) mb_uw;
static MB_SCALAR void cpy_scalar(void *d_, const void *s_, size_t n) {
    uint8_t *d = (uint8_t *)d_; const uint8_t *s = (const uint8_t *)s_;
    const size_t W = sizeof(size_t);
    size_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) {
        size_t a = *(const mb_uw *)(s + i), b = *(const mb_uw *)(s + i + W);
        size_t c = *(const mb_uw *)(s + i + 2 * W), e = *(const mb_uw *)(s + i + 3 * W);
        *(mb_uw *)(d + i) = a; *(mb_uw *)(d + i + W) = b; *(mb_uw *)(d + i + 2 * W) = c; *(mb_uw *)(d + i + 3 * W) = e;
    }
    for (; i < n; i++) { d[i] = s[i]; BM_BARRIER(); }
}
static MB_SCALAR void set_scalar(void *d_, int v, size_t n) {
    uint8_t *d = (uint8_t *)d_;
    const size_t W = sizeof(size_t);
    size_t w = (size_t)(uint8_t)v * (size_t)0x0101010101010101ull;
    BM_HIDE(w);
    size_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) { *(mb_uw *)(d + i) = w; *(mb_uw *)(d + i + W) = w; *(mb_uw *)(d + i + 2 * W) = w; *(mb_uw *)(d + i + 3 * W) = w; }
    for (; i < n; i++) { d[i] = (uint8_t)v; BM_BARRIER(); }
}

/* ============================================================================ cache sizes */
static size_t g_mb_cache[4];      /* [1] L1 data, [2] L2, [3] L3 (largest instance), bytes; 0 unknown */
static const char *g_mb_cache_src = "none";
static void mb_cpuid_caches(void) {
    unsigned leaf = 0;
    if (cpu()->max_basic >= 4 && !strcmp(cpu()->vendor, "GenuineIntel")) leaf = 4;
    else if (cpu()->max_ext >= 0x8000001D && (cpu()->e1.ecx & (1u << 22))) leaf = 0x8000001D;   /* AMD TOPOEXT */
    if (!leaf) return;
    for (unsigned sub = 0; sub < 16; sub++) {
        t_cpuid_regs r; cpuid_ex(leaf, sub, &r);
        unsigned type = r.eax & 31, level = (r.eax >> 5) & 7;
        if (!type) break;
        if (type == 2 || level < 1 || level > 3) continue;          /* instruction cache */
        size_t sz = (size_t)((r.ebx >> 22) + 1) * (((r.ebx >> 12) & 0x3FF) + 1) * ((r.ebx & 0xFFF) + 1) * ((size_t)r.ecx + 1);
        if (sz > g_mb_cache[level]) g_mb_cache[level] = sz;
        g_mb_cache_src = leaf == 4 ? "cpuid_leaf4" : "cpuid_8000001D";
    }
}
static int g_mb_caches_done;
static void mb_caches(void) {
    if (g_mb_caches_done) return;
    g_mb_caches_done = 1;
    DWORD len = 0;
    if (!GetLogicalProcessorInformation(NULL, &len) && GetLastError() == ERROR_INSUFFICIENT_BUFFER && len) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION *p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
        if (p && GetLogicalProcessorInformation(p, &len)) {
            for (DWORD i = 0; i < len / sizeof *p; i++) {
                if (p[i].Relationship != RelationCache) continue;
                const CACHE_DESCRIPTOR *c = &p[i].Cache;
                if (c->Level < 1 || c->Level > 3 || c->Type == CacheInstruction) continue;
                if (c->Size > g_mb_cache[c->Level]) g_mb_cache[c->Level] = c->Size;
                g_mb_cache_src = "GetLogicalProcessorInformation";
            }
        }
        free(p);
    }
    if (!g_mb_cache[1] && !g_mb_cache[2] && !g_mb_cache[3]) mb_cpuid_caches();
}
static const char *mb_level(size_t working_set) {
    mb_caches();
    for (int l = 1; l <= 3; l++) if (g_mb_cache[l] && working_set <= g_mb_cache[l]) return l == 1 ? "L1" : l == 2 ? "L2" : "L3";
    if (!g_mb_cache[1] && !g_mb_cache[2] && !g_mb_cache[3]) return "unknown";
    return "DRAM";
}

/* ============================================================================ buffers */
#define MB_MAX_SET  ((size_t)256 << 20)
#define MB_MAX_COPY ((size_t)128 << 20)
#define MB_GUARD 4096                       /* slack before and after each buffer for guard bytes */
static uint8_t *g_mb_src, *g_mb_dst;        /* page aligned; usable bytes [MB_GUARD, MB_GUARD + cap) */
static size_t g_mb_src_cap, g_mb_dst_cap;
static int g_mb_alloc_done;
static void mb_alloc(void) {
    if (g_mb_alloc_done) return;
    g_mb_alloc_done = 1;
    for (size_t n = MB_MAX_SET; n >= ((size_t)1 << 20) && !g_mb_dst; n >>= 1)
        if ((g_mb_dst = (uint8_t *)bm_alloc(n + 2 * MB_GUARD)) != NULL) g_mb_dst_cap = n;
    for (size_t n = MB_MAX_COPY; n >= ((size_t)1 << 20) && !g_mb_src; n >>= 1)
        if ((g_mb_src = (uint8_t *)bm_alloc(n + 2 * MB_GUARD)) != NULL) g_mb_src_cap = n;
    if (g_mb_src) { t_rng r; t_rng_seed(&r, 0x5EED1234u); t_rng_fill(&r, g_mb_src, g_mb_src_cap + 2 * MB_GUARD); }
    if (g_mb_dst) memset(g_mb_dst, 0x5A, g_mb_dst_cap + 2 * MB_GUARD);     /* commit every page before timing */
}

/* sizes: powers of 4 from 4 KB to 256 MB (quick) or every power of 2 (full) */
static int mb_sizes(size_t *out, int cap) {
    int n = 0;
    for (size_t s = 4096; s <= MB_MAX_SET && n < cap; s <<= (bm_full() ? 1 : 2)) out[n++] = s;
    return n;
}

/* ============================================================================ verification */
#define MB_GB 64                           /* guard bytes checked on each side of the destination */
static int mb_guards_ok(const uint8_t *d, size_t n, uint8_t g) {
    for (int k = 1; k <= MB_GB; k++) if (d[-k] != g || d[n + (size_t)k - 1] != g) return 0;
    return 1;
}
static size_t mb_count_not(const uint8_t *d, size_t n, uint8_t v) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += d[i] != v;
    return bad;
}

typedef struct mb_ctx { cpy_fn c; set_fn s; uint8_t *d; const uint8_t *src; size_t n; int v; } mb_ctx;
static void mb_cpy_bench(void *p) { mb_ctx *x = (mb_ctx *)p; x->c(x->d, x->src, x->n); }
static void mb_set_bench(void *p) { mb_ctx *x = (mb_ctx *)p; x->s(x->d, x->v, x->n); }

/* one kernel over the whole sweep; copy when c != NULL, fill otherwise */
static void mb_case(const char *op, const char *kname, cpy_fn c, set_fn s) {
    mb_alloc();
    if (!g_mb_dst || (c && !g_mb_src)) SKIP("cannot allocate the benchmark buffers (resource, not capability)");
    size_t cap = c ? (g_mb_src_cap < g_mb_dst_cap ? g_mb_src_cap : g_mb_dst_cap) : g_mb_dst_cap;
    if (cap < (c ? MB_MAX_COPY : MB_MAX_SET)) t_note("largest buffer that could be allocated: %u MB", (unsigned)(cap >> 20));
    if (c) t_note("copies stop at %u MB (source + destination stay within the 512 MB budget)", (unsigned)(cap >> 20));
    size_t sizes[32];
    int ns = mb_sizes(sizes, 32);
    while (ns > 0 && sizes[ns - 1] > cap) ns--;
    if (ns < 32 && (ns == 0 || sizes[ns - 1] < cap)) sizes[ns++] = cap;   /* the sweep ends at the cap (128 MB copies) */
    bm_pin();
    double first = 0, last = 0;
    for (int al = 0; al < 2; al++) {
        const size_t soff = al ? 1 : 0, doff = al ? 3 : 0;
        for (int i = 0; i < ns; i++) {
            size_t n = sizes[i];
            if (n > cap) break;
            char m[96], sz[32];
            snprintf(m, sizeof m, "%s_%s_%s_%s", op, kname, bm_size_name(n, sz, sizeof sz), al ? "misaligned" : "aligned");
            uint8_t *d = g_mb_dst + MB_GUARD + doff;
            const uint8_t *src = g_mb_src ? g_mb_src + MB_GUARD + soff : NULL;
            const uint8_t guard = 0xE7;
            memset(d - MB_GB, guard, n + 2 * MB_GB);
            mb_ctx x = { c, s, d, src, n, 0x3C + i };
            int ok;
            if (c) {
                c(d, src, n);
                ok = !memcmp(d, src, n) && mb_guards_ok(d, n, guard);
            } else {
                s(d, x.v, n);
                ok = mb_count_not(d, n, (uint8_t)x.v) == 0 && mb_guards_ok(d, n, guard);
            }
            CHECKF(ok, "%s: wrong result before timing (%s)", m, c ? "copy differs or guard bytes changed" : "fill byte wrong or guard bytes changed");
            if (!ok) continue;
            double v = bm_run(m, c ? mb_cpy_bench : mb_set_bench, &x, (double)n, 1e9, "GB/s", BM_RATE);
            ok = c ? (!memcmp(d, src, n) && mb_guards_ok(d, n, guard)) : (mb_count_not(d, n, (uint8_t)x.v) == 0 && mb_guards_ok(d, n, guard));
            CHECKF(ok, "%s: wrong result after timing", m);
            char key[128];
            snprintf(key, sizeof key, "%s.level", m);
            t_info(key, "%s", mb_level(c ? 2 * n : n));
            if (!al) { if (i == 0) first = v; last = v; }
        }
    }
    if (first > 0 && last > 0) t_info("largest_vs_4KB_aligned", "%.3f", last / first);
    bm_unpin();
}

/* ============================================================================ cases */
TEST_MANUAL(bench_mode_full) {
    g_bm_full = 1;
    t_note("full mode selected: every power of 2 from 4 KB to 256 MB, %d samples of >= %.0f ms", bm_reps(), bm_min_ms());
    CHECK(bm_full());
}

TEST(mem_env_report) {
    /* GetLogicalProcessorInformation buffer negotiation (documented contract) */
    DWORD len = 0;
    SetLastError(0);
    BOOL r = GetLogicalProcessorInformation(NULL, &len);
    DWORD e = GetLastError();
    CHECKF(!r && e == ERROR_INSUFFICIENT_BUFFER && len >= sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION),
           "GetLogicalProcessorInformation(NULL, &0) = %d, error %lu, length %lu (want FALSE, ERROR_INSUFFICIENT_BUFFER, required size)", r, e, len);
    CHECKF(len % sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) == 0, "required length %lu is not a multiple of the %u-byte record", len,
           (unsigned)sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    mb_caches();
    t_info("cache_source", "%s", g_mb_cache_src);
    t_info("l1d_bytes", "%u", (unsigned)g_mb_cache[1]);
    t_info("l2_bytes", "%u", (unsigned)g_mb_cache[2]);
    t_info("l3_bytes", "%u", (unsigned)g_mb_cache[3]);
    if (!g_mb_cache[1] && !g_mb_cache[2] && !g_mb_cache[3]) t_note("no cache sizes from the OS or CPUID: rows are labelled level=unknown");
    /* CPUID view as well, so a disagreement with the OS report is visible (INFO only) */
    size_t os[4]; memcpy(os, g_mb_cache, sizeof os);
    memset(g_mb_cache, 0, sizeof g_mb_cache);
    mb_cpuid_caches();
    t_info("cpuid_l1d_l2_l3_bytes", "%u %u %u", (unsigned)g_mb_cache[1], (unsigned)g_mb_cache[2], (unsigned)g_mb_cache[3]);
    if (os[1] && g_mb_cache[1] && (os[1] != g_mb_cache[1] || os[2] != g_mb_cache[2])) t_note("OS-reported and CPUID cache sizes differ (L1d %u vs %u, L2 %u vs %u; CPUID describes the core this ran on)",
        (unsigned)os[1], (unsigned)g_mb_cache[1], (unsigned)os[2], (unsigned)g_mb_cache[2]);
    memcpy(g_mb_cache, os, sizeof os);
    if (!os[1] && !os[2] && !os[3]) mb_cpuid_caches();
    SYSTEM_INFO si; GetSystemInfo(&si);
    t_info("page_size", "%lu", si.dwPageSize);
    t_info("large_page_minimum", "%llu", (unsigned long long)GetLargePageMinimum());
    MEMORYSTATUSEX ms; ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) t_info("total_phys_mb", "%llu", (unsigned long long)(ms.ullTotalPhys >> 20));
    t_info("erms", "%d", cpu_hw(F_ERMS));
    t_info("fsrm", "%d", cpu_hw(F_FSRM));
    t_info("cpu", "%s", cpu()->brand);
    t_info("mode", "%s", bm_full() ? "full" : "quick");
    t_info("bench_cpu", "%d", bm_choose_cpu());
    if (t_is_wine()) t_note("running under Wine %s: numbers describe the host + Wine, not Windows", t_wine_version());
}

/* small-size sweep against guard bytes, before any number is accepted (the shared kernels have their own
 * sweep in simd_bench; the scalar ones are new here) */
TEST(mem_scalar_kernels_correctness) {
    static T_ALIGN(64) uint8_t src[600], dst[800];
    t_rng r; t_rng_seed(&r, 99);
    t_rng_fill(&r, src, sizeof src);
    for (int so = 0; so < 9; so++) for (int dof = 0; dof < 9; dof++) for (size_t n = 0; n <= 300; n++) {
        memset(dst, 0xCC, sizeof dst);
        cpy_scalar(dst + 64 + dof, src + so, n);
        int ok = !memcmp(dst + 64 + dof, src + so, n) && mb_guards_ok(dst + 64 + dof, n, 0xCC);
        if (!ok) { CHECKF(0, "cpy_scalar src+%d dst+%d size %u wrong", so, dof, (unsigned)n); return; }
        memset(dst, 0xCC, sizeof dst);
        set_scalar(dst + 64 + dof, 0x30 + (int)n % 100, n);
        ok = mb_count_not(dst + 64 + dof, n, (uint8_t)(0x30 + n % 100)) == 0 && mb_guards_ok(dst + 64 + dof, n, 0xCC);
        if (!ok) { CHECKF(0, "set_scalar dst+%d size %u wrong", dof, (unsigned)n); return; }
    }
}

TEST(memcpy_sweep_scalar)    { mb_case("memcpy", "scalar", cpy_scalar, NULL); }
TEST(memcpy_sweep_crt)       { mb_case("memcpy", "crt", cpy_crt, NULL); }
TEST(memcpy_sweep_sse2)      { BM_NEED(SSE2); mb_case("memcpy", "sse2", cpy_sse2, NULL); }
TEST(memcpy_sweep_nt_sse2)   { BM_NEED(SSE2); mb_case("memcpy", "sse2_nt", cpy_sse2_nt, NULL); }
TEST(memcpy_sweep_avx_ymm)     { BM_NEED(AVX); mb_case("memcpy", "avx", cpy_avx, NULL); }
TEST(memcpy_sweep_avx512_zmm)  { BM_NEED(AVX512F); mb_case("memcpy", "avx512", cpy_avx512, NULL); }
TEST(memcpy_sweep_rep_movsb) { mb_case("memcpy", "rep_movsb", cpy_rep_movsb, NULL); }
TEST(memset_sweep_scalar)    { mb_case("memset", "scalar", NULL, set_scalar); }
TEST(memset_sweep_crt)       { mb_case("memset", "crt", NULL, set_crt); }
TEST(memset_sweep_sse2)      { BM_NEED(SSE2); mb_case("memset", "sse2", NULL, set_sse2); }
TEST(memset_sweep_nt_sse2)   { BM_NEED(SSE2); mb_case("memset", "sse2_nt", NULL, set_sse2_nt); }
TEST(memset_sweep_avx_ymm)     { BM_NEED(AVX); mb_case("memset", "avx", NULL, set_avx); }
TEST(memset_sweep_avx512_zmm)  { BM_NEED(AVX512F); BM_NEED(AVX512BW); mb_case("memset", "avx512", NULL, set_avx512); }
TEST(memset_sweep_rep_stosb) { mb_case("memset", "rep_stosb", NULL, set_rep_stosb); }
