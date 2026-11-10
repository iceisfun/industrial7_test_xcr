/* simd_bench - small, repeatable scalar / SSE / AVX / AVX-512 benchmarks with verified results.
 *
 * What this exe reports (docs/GUIDANCE.md section 5): PERF rows (median of >= 9 samples, with min, max,
 * mean, stddev and stddev %) for
 *   array add (GB/s), integer multiply (Mmul/s), FP throughput (GFLOP/s, independent chains) and latency
 *   (ns/op, dependent chain), dot product, 64/256 matrix multiply (GFLOP/s), popcount, byte sum, float->int
 *   conversion, gather, denormal penalty, memcpy / memset (CRT, SSE2, SSE2 non-temporal, AVX, AVX-512,
 *   REP MOVSB/STOSB) for cache-resident and memory-bound sizes, strlen, CRC32C (software vs SSE4.2),
 *   AES-128 (software vs AES-NI vs VAES), carry-less multiply, planar RGB <-> YCbCr, AVX <-> SSE
 *   transition costs, FXSAVE / XSAVE / XSAVEOPT / XSAVEC costs, context capture (RtlCaptureContext,
 *   Get/SetThreadContext normal vs XSTATE, InitializeContext, CopyContext, RtlCaptureContext2), scheduler
 *   round trips with live SIMD state, thread state stress (same-processor spinners re-verifying their
 *   vector registers under preemption), thread creation (new thread gets the default MXCSR / FCW).
 * Pass/fail policy:
 *   - slowness is NEVER a failure. A case FAILs only when a computed result is wrong, or when a documented
 *     contract the measurement relies on is violated (QPC monotonic, affinity honoured, a context API
 *     returning the state the thread holds, SIMD state surviving context switches / ABI calls).
 *   - feature absent -> SKIP; advertised but wrong -> FAIL. Frequencies, ratios and API behaviour that
 *     varies between machines -> INFO / NOTE.
 *   - TSC ticks are not core cycles; RDTSC numbers are INFO estimates only. QPC is the time base.
 *   - AVX-512 may lower the core clock: a slower AVX-512 number is not an OS defect.
 * Windows contracts relied on (MSDN; Windows 10 19041 reference decompilation, see context_xstate.c):
 *   - QueryPerformanceFrequency is fixed at boot and QueryPerformanceCounter is monotonic ("Acquiring
 *     high-resolution time stamps").
 *   - SetThreadAffinityMask returns the previous mask; a thread whose new mask excludes the current
 *     processor is rescheduled onto an allowed one before the call returns.
 *   - InitializeContext(NULL, ...) fails with ERROR_INSUFFICIENT_BUFFER and returns the required length
 *     (kernelbase!InitializeContext2); LocateXStateFeature(XSTATE_AVX) returns the YMM upper halves,
 *     16 bytes per register; ntdll!RtlCaptureContext (x64) sets ContextFlags = 0x10000F and captures
 *     MXCSR and the caller's RSP.
 *   - Win x64 ABI: XMM6-15 low 128 bits are callee-saved across any call, including blocking Windows APIs;
 *     YMM upper halves and XMM0-5 are volatile. On x86 every XMM register is volatile.
 * Modes: quick (default) / full via SIMD_BENCH_MODE=full or `--include-manual` (case bench_mode_full runs
 * first). The common CLI rejects unknown switches, so --quick/--full are not accepted on the command line. */
#include "simd_bench_util.h"
#include "simd_bench_compute.h"
#include "simd_bench_mem.h"
#include "simd_bench_yuv.h"
#include "simd_bench_asm.h"

#ifndef CONTEXT_XSTATE
#ifdef T_X64
#define CONTEXT_XSTATE (CONTEXT_AMD64 | 0x40L)
#else
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)
#endif
#endif
#ifndef XSTATE_MASK_AVX512
#define XSTATE_MASK_AVX512 (0xE0ull)
#endif
#ifdef T_X64
#define CX_FP_FLAGS (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT)
#else
#define CX_FP_FLAGS (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS)
#endif

/* =====================================================================================================
 * mode / environment / methodology
 * ===================================================================================================== */
TEST_MANUAL(bench_mode_full) {
    g_bm_full = 1;
    t_note("full mode selected: %d samples of >= %.0f ms, %u MB memory-bound buffers", bm_reps(), bm_min_ms(), (unsigned)(bm_big_size() >> 20));
    CHECK(bm_full());
}

static int bm_other_cpu(int not_this) {
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return -1;
    for (int i = (int)(sizeof(DWORD_PTR) * 8) - 1; i >= 0; i--) if ((pm >> i) & 1 && i != not_this) return i;
    return -1;
}

TEST(bench_env_report) {
    const t_env *e = t_env_get();
    LARGE_INTEGER f1, f2;
    CHECK(QueryPerformanceFrequency(&f1));
    CHECKF(f1.QuadPart > 0, "QueryPerformanceFrequency = %lld", (long long)f1.QuadPart);
    CHECK(QueryPerformanceFrequency(&f2));
    CHECKF(f1.QuadPart == f2.QuadPart, "QPC frequency changed: %lld -> %lld (documented fixed at boot)", (long long)f1.QuadPart, (long long)f2.QuadPart);
    CHECK_EQ(t_qpc_freq(), (uint64_t)f1.QuadPart);
    DWORD_PTR pm = 0, sm = 0;
    CHECK(GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm));
    CHECKF((pm & sm) == pm, "process affinity 0x%llx not a subset of the system mask 0x%llx", (unsigned long long)pm, (unsigned long long)sm);
    t_info("mode", "%s", bm_full() ? "full" : "quick");
    t_info("samples", "%d x >= %.1f ms after %d warm-up calls", bm_reps(), bm_min_ms(), bm_warmup());
    t_info("qpc_freq_hz", "%lld", (long long)f1.QuadPart);
    t_info("process_affinity", "0x%llx", (unsigned long long)pm);
    t_info("system_affinity", "0x%llx", (unsigned long long)sm);
    t_info("ncpu", "%d", t_cpu_count());
    t_info("active_groups", "%d", (int)e->active_groups);
    t_info("cpu", "%s", cpu()->brand);
    char buf[2048]; cpu_features_string(buf, sizeof buf, 1);
    t_info("features_usable", "%s", buf);
    t_info("membound_buffer_mb", "%u", (unsigned)(bm_big_size() >> 20));
    int bcpu = bm_choose_cpu();
    t_info("bench_cpu", "%d", bcpu);
    if (cpu_hw(F_HYBRID)) t_note("hybrid CPU: every case is pinned to processor %d (performance core preferred)", bcpu);
    if (t_is_wine()) t_note("running under Wine %s: numbers describe the host + Wine, not Windows", t_wine_version());
}

TEST(bench_stats_selftest) {
    const double v[5] = { 5, 1, 4, 2, 3 };
    t_stats s; t_stats_compute(v, 5, &s);
    CHECK_EQ(s.n, 5);
    CHECK(s.min == 1 && s.max == 5 && s.median == 3 && s.mean == 3);
    CHECKF(fabs(s.stddev - sqrt(2.5)) < 1e-12, "stddev %.17g", s.stddev);
    const double w[4] = { 4, 1, 3, 2 };
    t_stats_compute(w, 4, &s);
    CHECKF(s.median == 2.5, "even-n median %.17g", s.median);
    CHECKF(v[0] == 5 && v[1] == 1, "t_stats_compute modified its input");
    CHECK_EQ(bm_pow32(3, 5), 243u);
    CHECK_EQ(bm_pow64(0x10001ull, 2), 0x100020001ull);
    CHECK_EQ(bm_pow16(0xFFFF, 3), 0xFFFFu);
    CHECK_EQ(bm_pow32(0x9E3779B1u, 0), 1u);
}

/* QPC: monotonic across many reads, call cost, observed resolution */
static void qpc_fn(void *p) { uint64_t *x = (uint64_t *)p; for (int i = 0; i < 64; i++) x[0] += t_qpc(); }
TEST(bench_qpc_contract) {
    bm_pin();
    uint64_t prev = t_qpc(), back = 0, min_delta = ~0ull, n = bm_full() ? 2000000 : 200000;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t t = t_qpc();
        if (t < prev) back++;
        else if (t > prev && t - prev < min_delta) min_delta = t - prev;
        prev = t;
    }
    CHECKF(back == 0, "QueryPerformanceCounter went backwards %llu times in %llu reads", (unsigned long long)back, (unsigned long long)n);
    if (min_delta != ~0ull) t_info("qpc_min_nonzero_delta_ns", "%.1f", (double)min_delta * 1e9 / (double)t_qpc_freq());
    uint64_t sink = 0;
    bm_run("qpc_call", qpc_fn, &sink, 64, 1e9, "ns/call", BM_TIME);
    T_SINK(sink);
    bm_unpin();
}

TEST(bench_tsc_estimate) {
    BM_NEED(TSC);
    bm_pin();
    double ms = bm_full() ? 200 : 25, est[5];
    for (int r = 0; r < 5; r++) {
        uint64_t q0 = t_qpc(), c0 = t_rdtsc(), q1;
        do { q1 = t_qpc(); } while (t_qpc_ms(q0, q1) < ms / 5);
        uint64_t c1 = t_rdtsc();
        CHECKF(c1 > c0, "RDTSC did not advance over %.1f ms (0x%llx -> 0x%llx)", t_qpc_ms(q0, q1), (unsigned long long)c0, (unsigned long long)c1);
        est[r] = (double)(c1 - c0) / t_qpc_sec(q0, q1) / 1e9;
    }
    t_stats s; t_stats_compute(est, 5, &s);
    t_info("tsc_ghz_estimate", "%.3f (min %.3f max %.3f) - TSC rate, NOT the core clock", s.median, s.min, s.max);
    t_info("invariant_tsc", "%d", cpu_hw(F_INVARIANT_TSC));
    if (cpu_has(F_RDTSCP)) { uint32_t aux = 0; t_rdtscp(&aux); t_info("rdtscp_aux", "0x%x", aux); }
    bm_unpin();
}

TEST(bench_affinity_contract) {
    DWORD_PTR pm = 0, sm = 0;
    REQUIRE(GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm));
    if (t_env_get()->active_groups > 1) t_note("more than one processor group: checks restricted to the current group");
    DWORD_PTR orig = SetThreadAffinityMask(GetCurrentThread(), pm);
    REQUIREF(orig != 0, "SetThreadAffinityMask(process mask) failed: %lu", GetLastError());
    DWORD_PTR last = pm;
    int tested = 0;
    for (int i = 0; i < (int)(sizeof(DWORD_PTR) * 8) && tested < 16; i++) {
        if (!((pm >> i) & 1)) continue;
        DWORD_PTR m = (DWORD_PTR)1 << i;
        DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), m);
        CHECKF(prev == last, "SetThreadAffinityMask returned 0x%llx, previous mask was 0x%llx", (unsigned long long)prev, (unsigned long long)last);
        if (!prev) continue;
        last = m;
        PROCESSOR_NUMBER pn; GetCurrentProcessorNumberEx(&pn);
        int bad = 0;
        for (int k = 0; k < 8; k++) { if (k) SwitchToThread(); GetCurrentProcessorNumberEx(&pn); if (pn.Number != i) bad++; }
        if (t_env_get()->active_groups <= 1)
            CHECKF(bad == 0, "thread restricted to cpu %d observed on cpu %u (%d/8 samples)", i, pn.Number, bad);
        tested++;
    }
    SetThreadAffinityMask(GetCurrentThread(), orig);
    t_info("cpus_checked", "%d", tested);
}

/* =====================================================================================================
 * array add
 * ===================================================================================================== */
typedef struct add_ctx { add_fn f; void *a, *b, *c; size_t n; } add_ctx;
static void add_bench(void *p) { add_ctx *x = (add_ctx *)p; x->f(x->c, x->a, x->b, x->n); }
static size_t add_verify(const void *c_, size_t n, int is_int, size_t *first) {
    size_t bad = 0;
    if (is_int) {
        const uint32_t *c = (const uint32_t *)c_;
        for (size_t i = 0; i < n; i++) if (c[i] != (uint32_t)(i * 2654435761u) + ~(uint32_t)i) { if (!bad) *first = i; bad++; }
    } else {
        const float *c = (const float *)c_;
        for (size_t i = 0; i < n; i++) if (c[i] != (float)(i & 1023) + (float)((i * 7) & 511) * 0.5f) { if (!bad) *first = i; bad++; }
    }
    return bad;
}
static void arradd_case(const char *tag, add_fn f, int is_int) {
    size_t nmem = bm_full() ? ((size_t)8 << 20) : ((size_t)4 << 20);
    size_t stride = nmem * 4 + 4096 + 192;              /* different offsets mod 4 KB: no 4K aliasing */
    uint8_t *mem = (uint8_t *)bm_alloc(3 * stride);
    if (!mem) SKIP("cannot allocate %u MB", (unsigned)(3 * stride >> 20));
    void *a = mem, *b = mem + stride, *c = mem + 2 * stride;
    if (is_int) for (size_t i = 0; i < nmem; i++) { ((uint32_t *)a)[i] = (uint32_t)(i * 2654435761u); ((uint32_t *)b)[i] = ~(uint32_t)i; }
    else for (size_t i = 0; i < nmem; i++) { ((float *)a)[i] = (float)(i & 1023); ((float *)b)[i] = (float)((i * 7) & 511) * 0.5f; }
    bm_pin();
    const size_t ns[3] = { 2048, 16384, nmem };
    for (int s = 0; s < 3; s++) {
        size_t n = ns[s], first = 0;
        memset(c, 0xFF, n * 4);
        f(c, a, b, n);
        size_t bad = add_verify(c, n, is_int, &first);
        CHECKF(bad == 0, "%s n=%u: %u wrong elements, first at %u", tag, (unsigned)n, (unsigned)bad, (unsigned)first);
        if (bad) continue;
        char m[96], sz[32];
        snprintf(m, sizeof m, "%s_%s", tag, bm_size_name(3 * n * 4, sz, sizeof sz));
        add_ctx x = { f, a, b, c, n };
        bm_run(m, add_bench, &x, 12.0 * (double)n, 1e9, "GB/s", BM_RATE);
        bad = add_verify(c, n, is_int, &first);
        CHECKF(bad == 0, "%s n=%u after timing: %u wrong elements", tag, (unsigned)n, (unsigned)bad);
    }
    bm_unpin();
    bm_free(mem);
}
TEST(arradd_f32_scalar) { arradd_case("add_f32_scalar", add_f32_scalar, 0); }
TEST(arradd_f32_sse)    { BM_NEED(SSE); arradd_case("add_f32_sse", add_f32_sse, 0); }
TEST(arradd_f32_avx)    { BM_NEED(AVX); arradd_case("add_f32_avx", add_f32_avx, 0); }
TEST(arradd_f32_avx512) { BM_NEED(AVX512F); arradd_case("add_f32_avx512", add_f32_avx512, 0); }
TEST(arradd_i32_scalar) { arradd_case("add_i32_scalar", add_i32_scalar, 1); }
TEST(arradd_i32_sse2)   { BM_NEED(SSE2); arradd_case("add_i32_sse2", add_i32_sse2, 1); }
TEST(arradd_i32_avx2)   { BM_NEED(AVX2); arradd_case("add_i32_avx2", add_i32_avx2, 1); }
TEST(arradd_i32_avx512) { BM_NEED(AVX512F); arradd_case("add_i32_avx512", add_i32_avx512, 1); }

/* =====================================================================================================
 * integer multiply
 * ===================================================================================================== */
enum { IMK_U32T, IMK_U32L, IMK_U64T, IMK_U64L, IMK_PMULUDQ, IMK_PMULLW, IMK_PMULLD128, IMK_PMULLD256, IMK_PMULLD512, IMK_PMULLQ512 };
static void imul_case(const char *metric, t_bench_fn fn, int kind) {
    static im_ctx x;
    memset(&x, 0, sizeof x);
    for (int i = 0; i < 16; i++) x.seed[i] = t_mix64((uint64_t)i + 1) | 0x0001000100010001ull;   /* odd in every 16/32/64-bit lane */
    x.c32 = 0x9E3779B1u; x.c64 = 0x9E3779B97F4A7C15ull; x.c16 = 0x9E37u; x.K = IM_K;
    fn(&x);
    const uint16_t *s16 = (const uint16_t *)x.seed; const uint32_t *s32 = (const uint32_t *)x.seed;
    const uint16_t *o16 = (const uint16_t *)x.out;  const uint32_t *o32 = (const uint32_t *)x.out; const uint64_t *o64 = (const uint64_t *)x.out;
    uint32_t p32 = bm_pow32(x.c32, IM_K), p32m = bm_pow32(x.c32, IM_K - 1);
    uint64_t p64 = bm_pow64(x.c64, IM_K); uint16_t p16 = bm_pow16(x.c16, IM_K);
    int lanes = 0, bad = 0, first = -1; uint64_t got = 0, want = 0;
#define IM_CMP(g, w) do { uint64_t g_ = (g), w_ = (w); if (g_ != w_) { if (first < 0) { first = l; got = g_; want = w_; } bad++; } } while (0)
    switch (kind) {
    case IMK_U32T: lanes = 8; for (int l = 0; l < 8; l++) IM_CMP(o32[l], (uint32_t)x.seed[l] * p32); break;
    case IMK_U32L: lanes = 1; { int l = 0; IM_CMP(o32[0], (uint32_t)x.seed[0] * p32); } break;
    case IMK_U64T: lanes = 8; for (int l = 0; l < 8; l++) IM_CMP(o64[l], x.seed[l] * p64); break;
    case IMK_U64L: lanes = 1; { int l = 0; IM_CMP(o64[0], x.seed[0] * p64); } break;
    case IMK_PMULUDQ: lanes = 8; for (int l = 0; l < 8; l++) IM_CMP(o64[l], (uint64_t)((uint32_t)x.seed[l] * p32m) * x.c32); break;
    case IMK_PMULLW: lanes = 32; for (int l = 0; l < 32; l++) IM_CMP(o16[l], (uint16_t)(s16[l] * p16)); break;
    case IMK_PMULLD128: lanes = 16; for (int l = 0; l < 16; l++) IM_CMP(o32[l], s32[l] * p32); break;
    case IMK_PMULLD256: lanes = 32; for (int l = 0; l < 32; l++) IM_CMP(o32[l], s32[l] * p32); break;
    case IMK_PMULLD512: lanes = 64; for (int l = 0; l < 64; l++) IM_CMP(o32[l], s32[l % 32] * p32); break;
    case IMK_PMULLQ512: lanes = 32; for (int l = 0; l < 32; l++) IM_CMP(o64[l], x.seed[l % 16] * p64); break;
    }
#undef IM_CMP
    CHECKF(bad == 0, "%s: %d/%d lanes wrong, first lane %d got 0x%llx want 0x%llx", metric, bad, lanes, first, (unsigned long long)got, (unsigned long long)want);
    if (bad) return;
    bm_pin();
    bm_run(metric, fn, &x, (double)lanes * IM_K, 1e6, kind == IMK_U32L || kind == IMK_U64L ? "ns/mul" : "Mmul/s", BM_RATE);
    bm_unpin();
}
static void imul_lat_case(const char *metric, t_bench_fn fn, int kind) {
    static im_ctx x;
    memset(&x, 0, sizeof x);
    x.seed[0] = 0x123456789ABCDEF1ull; x.c32 = 0x9E3779B1u; x.c64 = 0x9E3779B97F4A7C15ull; x.K = IM_K;
    fn(&x);
    if (kind == IMK_U32L) CHECK_EQ(((uint32_t *)x.out)[0], (uint32_t)x.seed[0] * bm_pow32(x.c32, IM_K));
    else CHECK_EQ(((uint64_t *)x.out)[0], x.seed[0] * bm_pow64(x.c64, IM_K));
    bm_pin();
    bm_run(metric, fn, &x, IM_K, 1e9, "ns/mul", BM_TIME);
    bm_unpin();
}
TEST(imul_scalar32_tput) { imul_case("imul32_scalar_8chains", imul32_tput_scalar, IMK_U32T); }
TEST(imul_scalar32_lat)  { imul_lat_case("imul32_scalar_latency", imul32_lat_scalar, IMK_U32L); }
TEST(imul_scalar64_tput) {
#ifndef T_X64
    t_note("x86: 64-bit multiply is a compiler sequence of 32-bit MULs");
#endif
    imul_case("imul64_scalar_8chains", imul64_tput_scalar, IMK_U64T);
}
TEST(imul_scalar64_lat)  { imul_lat_case("imul64_scalar_latency", imul64_lat_scalar, IMK_U64L); }
TEST(imul_sse2_pmullw)   { BM_NEED(SSE2); imul_case("pmullw_sse2_16bit", imul_sse2_pmullw, IMK_PMULLW); }
TEST(imul_sse2_pmuludq)  { BM_NEED(SSE2); imul_case("pmuludq_sse2_32x32to64", imul_sse2_pmuludq, IMK_PMULUDQ); }
TEST(imul_sse41_pmulld)  { BM_NEED(SSE41); imul_case("pmulld_sse41", imul_sse41_pmulld, IMK_PMULLD128); }
TEST(imul_avx2_vpmulld)  { BM_NEED(AVX2); imul_case("vpmulld_avx2_ymm", imul_avx2_vpmulld, IMK_PMULLD256); }
TEST(imul_avx512_vpmulld) { BM_NEED(AVX512F); imul_case("vpmulld_avx512_zmm", imul_avx512_vpmulld, IMK_PMULLD512); }
TEST(imul_avx512_vpmullq) { BM_NEED(AVX512F); BM_NEED(AVX512DQ); imul_case("vpmullq_avx512dq_zmm", imul_avx512_vpmullq, IMK_PMULLQ512); }

/* =====================================================================================================
 * FP throughput (8 independent chains) and latency (1 dependent chain)
 * ===================================================================================================== */
/* naccs accumulators x lanes verified lanes; expect ADD: init+K, SAME: init */
static void fp_case(const char *metric, t_bench_fn fn, int lanes, int naccs, int flops_per_lane_op, int expect, int is_double, int sqrt_init, int latency) {
    static fp_ctx x;
    memset(&x, 0, sizeof x);
    for (int j = 0; j < 8; j++) x.init[j] = sqrt_init ? 1.0 : (double)(j + 1);
    x.K = FP_K;
    fn(&x);
    int bad = 0; double gotv = 0, wantv = 0; int fj = -1, fl = -1;
    for (int j = 0; j < naccs; j++)
        for (int l = 0; l < lanes; l++) {
            double g = is_double ? ((const double *)x.out[j])[l] : (double)((const float *)x.out[j])[l];
            double w = expect == FPE_ADD ? x.init[j] + FP_K : x.init[j];
            if (g != w) { if (!bad) { gotv = g; wantv = w; fj = j; fl = l; } bad++; }
        }
    CHECKF(bad == 0, "%s: %d wrong lanes, first acc %d lane %d got %.17g want %.17g", metric, bad, fj, fl, gotv, wantv);
    if (bad) return;
    bm_pin();
    if (latency) bm_run(metric, fn, &x, FP_K, 1e9, "ns/op", BM_TIME);
    else bm_run(metric, fn, &x, (double)lanes * naccs * FP_K * flops_per_lane_op, 1e9, "GFLOP/s", BM_RATE);
    bm_unpin();
}
#define FPT(name, feat, fn, lanes, flops, expect, dbl, sq) TEST(name) { BM_NEED(feat); fp_case(#name, fn, lanes, 8, flops, expect, dbl, sq, 0); }
#define FPL(name, feat, fn, lanes, expect, dbl, sq)        TEST(name) { BM_NEED(feat); fp_case(#name, fn, lanes, 1, 1, expect, dbl, sq, 1); }
FPT(fpt_sse_addss_scalar,  SSE,     fpt_sse_addss,      1, 1, FPE_ADD,  0, 0)
FPT(fpt_sse2_addsd_scalar, SSE2,    fpt_sse2_addsd,     1, 1, FPE_ADD,  1, 0)
FPT(fpt_sse2_mulsd_scalar, SSE2,    fpt_sse2_mulsd,     1, 1, FPE_SAME, 1, 0)
FPT(fpt_sse_addps,         SSE,     fpt_sse_addps,      4, 1, FPE_ADD,  0, 0)
FPT(fpt_sse_mulps,         SSE,     fpt_sse_mulps,      4, 1, FPE_SAME, 0, 0)
FPT(fpt_sse_divps,         SSE,     fpt_sse_divps,      4, 1, FPE_SAME, 0, 0)
FPT(fpt_sse_sqrtps,        SSE,     fpt_sse_sqrtps,     4, 1, FPE_SAME, 0, 1)
FPT(fpt_sse2_addpd,        SSE2,    fpt_sse2_addpd,     2, 1, FPE_ADD,  1, 0)
FPT(fpt_sse2_mulpd,        SSE2,    fpt_sse2_mulpd,     2, 1, FPE_SAME, 1, 0)
FPT(fpt_avx_addps_ymm,     AVX,     fpt_avx_addps,      8, 1, FPE_ADD,  0, 0)
FPT(fpt_avx_mulps_ymm,     AVX,     fpt_avx_mulps,      8, 1, FPE_SAME, 0, 0)
FPT(fpt_avx_divps_ymm,     AVX,     fpt_avx_divps,      8, 1, FPE_SAME, 0, 0)
FPT(fpt_avx_addpd_ymm,     AVX,     fpt_avx_addpd,      4, 1, FPE_ADD,  1, 0)
FPT(fpt_avx_mulpd_ymm,     AVX,     fpt_avx_mulpd,      4, 1, FPE_SAME, 1, 0)
FPT(fpt_fma_ps_xmm,        FMA,     fpt_fma_ps128,      4, 2, FPE_ADD,  0, 0)
FPT(fpt_fma_ps_ymm,        FMA,     fpt_fma_ps256,      8, 2, FPE_ADD,  0, 0)
FPT(fpt_fma_pd_ymm,        FMA,     fpt_fma_pd256,      4, 2, FPE_ADD,  1, 0)
FPT(fpt_avx512_addps_zmm,  AVX512F, fpt_avx512_addps,  16, 1, FPE_ADD,  0, 0)
FPT(fpt_avx512_mulpd_zmm,  AVX512F, fpt_avx512_mulpd,   8, 1, FPE_SAME, 1, 0)
FPT(fpt_avx512_fmaps_zmm,  AVX512F, fpt_avx512_fmaps,  16, 2, FPE_ADD,  0, 0)
FPT(fpt_avx512_fmapd_zmm,  AVX512F, fpt_avx512_fmapd,   8, 2, FPE_ADD,  1, 0)
FPL(fplat_sse2_addsd,      SSE2,    fpl_sse2_addsd,     1, FPE_ADD,  1, 0)
FPL(fplat_sse2_mulsd,      SSE2,    fpl_sse2_mulsd,     1, FPE_SAME, 1, 0)
FPL(fplat_sse2_divsd,      SSE2,    fpl_sse2_divsd,     1, FPE_SAME, 1, 0)
FPL(fplat_sse2_sqrtsd,     SSE2,    fpl_sse2_sqrtsd,    1, FPE_SAME, 1, 1)
FPL(fplat_sse_addps,       SSE,     fpl_sse_addps,      4, FPE_ADD,  0, 0)
FPL(fplat_sse_mulps,       SSE,     fpl_sse_mulps,      4, FPE_SAME, 0, 0)
FPL(fplat_avx_addps_ymm,   AVX,     fpl_avx_addps,      8, FPE_ADD,  0, 0)
FPL(fplat_fma_sd,          FMA,     fpl_fma_sd,         1, FPE_ADD,  1, 0)
FPL(fplat_fma_ps_ymm,      FMA,     fpl_fma_ps256,      8, FPE_ADD,  0, 0)
FPL(fplat_avx512_addps_zmm, AVX512F, fpl_avx512_addps, 16, FPE_ADD,  0, 0)
FPL(fplat_x87_fadd,        FPU,     fpl_x87_fadd,       1, FPE_ADD,  1, 0)
FPL(fplat_x87_fmul,        FPU,     fpl_x87_fmul,       1, FPE_SAME, 1, 0)

TEST(fpt_mmx_paddw) {
    BM_NEED(MMX);
    static fp_ctx x; memset(&x, 0, sizeof x); x.K = FP_K;
    fpt_mmx_paddw(&x);
    const uint16_t *w = (const uint16_t *)x.out[0];
    int bad = 0; for (int i = 0; i < 16; i++) if (w[i] != 2 * FP_K) bad++;
    CHECKF(bad == 0, "MMX paddw: %d of 16 word lanes wrong (lane0 %u want %u)", bad, w[0], 2 * FP_K);
    CHECKF(t_get_x87_tw() == 0xFFFF, "x87 tag word 0x%04x after EMMS (want 0xFFFF: all empty)", t_get_x87_tw());
    if (bad) return;
    bm_pin();
    bm_run("mmx_paddw_4chains", fpt_mmx_paddw, &x, 4.0 * 2 * FP_K, 1e9, "Ginstr/s", BM_RATE);
    bm_unpin();
}

/* Denormal operands: default MXCSR (assists) vs FTZ|DAZ. x * 1.0 with x denormal is exact:
 * default -> result x, DE set, UE/PE clear (SDM vol.1 4.9.1.5, 10.2.3.4: UE needs tiny AND inexact when
 * masked); DAZ -> operand treated as +-0, result 0, DE NOT set (SDM vol.1 10.2.3.4). */
TEST(fp_denormal_penalty) {
    BM_NEED(SSE2);
    T_ALIGN(16) uint8_t fx[512]; memset(fx, 0, sizeof fx);
    t_fxsave(fx);
    uint32_t mask; memcpy(&mask, fx + 28, 4);
    if (!mask) mask = 0xFFBF;                                    /* SDM: MXCSR_MASK 0 means 0xFFBF (no DAZ) */
    t_info("mxcsr_mask", "0x%08x", mask);
    static fp_ctx x; memset(&x, 0, sizeof x);
    for (int j = 0; j < 8; j++) x.init[j] = ldexp((double)(j + 1), -140);   /* float denormals, exact in double->float (no PE/UE from the conversion) */
    x.K = FP_K;
    unsigned saved = t_get_mxcsr();
    t_set_mxcsr(0x1F80);
    fpt_sse_mulps(&x);
    unsigned f = t_get_mxcsr();
    t_set_mxcsr(saved);
    int bad = 0;
    for (int j = 0; j < 8; j++) for (int l = 0; l < 4; l++) if (((const float *)x.out[j])[l] != (float)x.init[j]) bad++;
    CHECKF(bad == 0, "denormal * 1.0 changed the value in %d lanes", bad);
    CHECKF(f & 0x02, "MXCSR.DE not set after denormal operands (mxcsr 0x%04x)", f);
    CHECKF(!(f & 0x10), "MXCSR.UE set although every result was exact (mxcsr 0x%04x)", f);
    CHECKF(!(f & 0x20), "MXCSR.PE set although every result was exact (mxcsr 0x%04x)", f);
    bm_pin();
    t_set_mxcsr(0x1F80);
    double slow = bm_run("mulps_denormal_default", fpt_sse_mulps, &x, 32.0 * FP_K, 1e9, "ns/mul", BM_TIME);
    t_set_mxcsr(saved);
    if (mask & 0x40) {
        /* correctness with DAZ ALONE: the kernel's own double->float conversion then still produces the
         * float denormals (FTZ would already flush them there), so only DAZ can turn the operands into 0 */
        t_set_mxcsr(0x1FC0);                                     /* DAZ | all masked, FTZ clear */
        fpt_sse_mulps(&x);
        f = t_get_mxcsr();
        t_set_mxcsr(0x9FC0);                                     /* FTZ | DAZ | all masked */
        double fast = bm_run("mulps_denormal_ftz_daz", fpt_sse_mulps, &x, 32.0 * FP_K, 1e9, "ns/mul", BM_TIME);
        t_set_mxcsr(saved);
        bad = 0;
        for (int j = 0; j < 8; j++) for (int l = 0; l < 4; l++) {
            float g = ((const float *)x.out[j])[l]; uint32_t u; memcpy(&u, &g, 4);
            if (u != 0) bad++;                                   /* +denormal under DAZ -> +0.0 */
        }
        CHECKF(bad == 0, "DAZ: %d lanes not +0.0", bad);
        CHECKF(!(f & 0x02), "MXCSR.DE set with DAZ=1 (mxcsr 0x%04x)", f);
        if (fast > 0) t_info("denormal_slowdown", "%.2fx", slow / fast);
    } else t_note("DAZ not supported (MXCSR_MASK bit 6 clear): FTZ|DAZ row skipped");
    bm_unpin();
}

/* =====================================================================================================
 * dot product, matrix multiply
 * ===================================================================================================== */
typedef struct dot_ctx { dot_fn f; const float *a, *b; size_t n; float r; } dot_ctx;
static void dot_bench(void *p) { dot_ctx *x = (dot_ctx *)p; x->r = x->f(x->a, x->b, x->n); T_SINK(x->r); }
static void dot_case(const char *tag, dot_fn f) {
    size_t nmax = (size_t)4 << 20;                       /* 2 x 16 MB, exact: sum < 2^24 */
    float *a = (float *)bm_alloc(nmax * 4 + 4096 + 64), *b = (float *)bm_alloc(nmax * 4 + 64);
    if (!a || !b) { bm_free(a); bm_free(b); SKIP("cannot allocate 32 MB"); }
    for (size_t i = 0; i < nmax; i++) { a[i] = (float)(i % 3); b[i] = (float)(i % 5); }
    bm_pin();
    const size_t ns[3] = { 4096, 262144, nmax };
    for (int s = 0; s < 3; s++) {
        size_t n = ns[s];
        uint64_t want = 0; for (size_t i = 0; i < n; i++) want += (uint64_t)(i % 3) * (i % 5);
        float r = f(a, b, n);
        CHECKF(r == (float)want, "%s n=%u: got %.9g want %llu", tag, (unsigned)n, r, (unsigned long long)want);
        if (r != (float)want) continue;
        char m[96], sz[32]; snprintf(m, sizeof m, "%s_%s", tag, bm_size_name(8 * n, sz, sizeof sz));
        dot_ctx x = { f, a, b, n, 0 };
        bm_run(m, dot_bench, &x, (double)n, 1e6, "Melem/s", BM_RATE);
        CHECKF(x.r == (float)want, "%s n=%u after timing: %.9g", tag, (unsigned)n, x.r);
    }
    bm_unpin();
    bm_free(a); bm_free(b);
}
TEST(dot_scalar_f32) { dot_case("dot_scalar", dot_scalar); }
TEST(dot_sse_f32)    { BM_NEED(SSE); dot_case("dot_sse", dot_sse); }
TEST(dot_avx_f32)    { BM_NEED(AVX); dot_case("dot_avx", dot_avx); }
TEST(dot_fma_f32)    { BM_NEED(AVX2); BM_NEED(FMA); dot_case("dot_avx2_fma", dot_fma); }
TEST(dot_avx512_f32) { BM_NEED(AVX512F); dot_case("dot_avx512", dot_avx512); }

typedef struct mm_ctx { mm_fn f; float *C; const float *A, *B; int N; } mm_ctx;
static void mm_bench(void *p) { mm_ctx *x = (mm_ctx *)p; x->f(x->C, x->A, x->B, x->N); }
static void mm_case(const char *tag, mm_fn f) {
    const int Ns[2] = { 64, 256 };
    float *A = (float *)bm_alloc(256 * 256 * 4), *B = (float *)bm_alloc(256 * 256 * 4 + 4096), *C = (float *)bm_alloc(256 * 256 * 4);
    int32_t *R = (int32_t *)bm_alloc(256 * 256 * 4);
    if (!A || !B || !C || !R) { bm_free(A); bm_free(B); bm_free(C); bm_free(R); SKIP("cannot allocate matrices"); }
    bm_pin();
    for (int s = 0; s < 2; s++) {
        int N = Ns[s];
        for (int i = 0; i < N * N; i++) { A[i] = (float)((i * 7 + 3) % 4); B[i] = (float)((i * 5 + 1) % 4); }
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            int32_t t = 0; for (int k = 0; k < N; k++) t += ((i * N + k) * 7 + 3) % 4 * (((k * N + j) * 5 + 1) % 4);
            R[i * N + j] = t;
        }
        for (int i = 0; i < N * N; i++) C[i] = -1.0f;
        f(C, A, B, N);
        int bad = 0, first = -1;
        for (int i = 0; i < N * N; i++) if (C[i] != (float)R[i]) { if (first < 0) first = i; bad++; }
        CHECKF(bad == 0, "%s N=%d: %d wrong elements, first [%d][%d] got %.9g want %d", tag, N, bad,
               first < 0 ? 0 : first / N, first < 0 ? 0 : first % N, first < 0 ? 0.0 : C[first], first < 0 ? 0 : R[first]);
        if (bad) continue;
        char m[96]; snprintf(m, sizeof m, "%s_%dx%d", tag, N, N);
        mm_ctx x = { f, C, A, B, N };
        bm_run(m, mm_bench, &x, 2.0 * N * N * N, 1e9, "GFLOP/s", BM_RATE);
        bad = 0; for (int i = 0; i < N * N; i++) if (C[i] != (float)R[i]) bad++;
        CHECKF(bad == 0, "%s N=%d after timing: %d wrong elements", tag, N, bad);
    }
    bm_unpin();
    bm_free(A); bm_free(B); bm_free(C); bm_free(R);
}
TEST(matmul_scalar)   { mm_case("matmul_scalar", mm_scalar); }
TEST(matmul_sse2)     { BM_NEED(SSE2); mm_case("matmul_sse2", mm_sse2); }
TEST(matmul_avx2_fma) { BM_NEED(AVX2); BM_NEED(FMA); mm_case("matmul_avx2_fma", mm_avx2_fma); }
TEST(matmul_avx512)   { BM_NEED(AVX512F); mm_case("matmul_avx512", mm_avx512); }

/* =====================================================================================================
 * popcount, byte sum, conversion, gather
 * ===================================================================================================== */
typedef struct red_ctx { red_fn f; const uint8_t *p; size_t n; uint64_t r; } red_ctx;
static void red_bench(void *p) { red_ctx *x = (red_ctx *)p; x->r = x->f(x->p, x->n); T_SINK(x->r); }
static void red_case(const char *tag, red_fn f, int popcount) {
    uint8_t *src, *dst;
    if (!bm_big(&src, &dst)) SKIP("cannot allocate the memory-bound buffers");
    bm_pin();
    const size_t ns[2] = { 64 * 1024, (size_t)8 << 20 };
    for (int s = 0; s < 2; s++) {
        size_t n = ns[s]; uint64_t want = 0;
        if (popcount) {
            static uint8_t pc[256]; for (int i = 0; i < 256; i++) { int c = 0; for (int b = 0; b < 8; b++) c += (i >> b) & 1; pc[i] = (uint8_t)c; }
            for (size_t i = 0; i < n; i++) want += pc[src[i]];
        } else for (size_t i = 0; i < n; i++) want += src[i];
        uint64_t r = f(src, n);
        CHECKF(r == want, "%s n=%u: got %llu want %llu", tag, (unsigned)n, (unsigned long long)r, (unsigned long long)want);
        if (r != want) continue;
        char m[96], sz[32]; snprintf(m, sizeof m, "%s_%s", tag, bm_size_name(n, sz, sizeof sz));
        red_ctx x = { f, src, n, 0 };
        bm_run(m, red_bench, &x, (double)n, 1e9, "GB/s", BM_RATE);
        CHECK_EQ(x.r, want);
    }
    bm_unpin();
}
TEST(popcount_scalar_swar)  { red_case("popcount_scalar_swar", pop_scalar_swar, 1); }
TEST(popcount_popcnt)       { BM_NEED(POPCNT); red_case("popcount_popcnt", pop_popcnt, 1); }
TEST(popcount_ssse3_pshufb) { BM_NEED(SSSE3); red_case("popcount_ssse3_pshufb", pop_ssse3_pshufb, 1); }
TEST(popcount_avx2_pshufb)  { BM_NEED(AVX2); red_case("popcount_avx2_vpshufb", pop_avx2_pshufb, 1); }
TEST(bytesum_scalar)        { red_case("bytesum_scalar", bsum_scalar, 0); }
TEST(bytesum_sse2_psadbw)   { BM_NEED(SSE2); red_case("bytesum_sse2_psadbw", bsum_sse2_psadbw, 0); }
TEST(bytesum_avx2_psadbw)   { BM_NEED(AVX2); red_case("bytesum_avx2_vpsadbw", bsum_avx2_psadbw, 0); }

typedef struct cvt_ctx { cvt_fn f; int32_t *o; const float *in; size_t n; } cvt_ctx;
static void cvt_bench(void *p) { cvt_ctx *x = (cvt_ctx *)p; x->f(x->o, x->in, x->n); }
static void cvt_case(const char *tag, cvt_fn f) {
    enum { N = 65536 };
    float *in = (float *)bm_alloc(N * 4); int32_t *o = (int32_t *)bm_alloc(N * 4);
    if (!in || !o) { bm_free(in); bm_free(o); SKIP("cannot allocate"); }
    t_rng r; t_rng_seed(&r, 77);
    for (int i = 0; i < N; i++) in[i] = (float)((double)(int64_t)t_rng_range(&r, 4000000) / 2.0 - 1000000.0 + (double)(i % 7) * 0.125);
    in[0] = -0.75f; in[1] = 0.75f; in[2] = -1.0f; in[3] = 2147483520.0f; in[4] = -2147483648.0f; in[5] = 0.0f; in[6] = -0.0f; in[7] = 16777215.5f;
    memset(o, 0xA5, N * 4);
    f(o, in, N);
    int bad = 0, first = -1;
    for (int i = 0; i < N; i++) { int32_t w = (int32_t)(double)in[i]; if (o[i] != w) { if (first < 0) first = i; bad++; } }
    CHECKF(bad == 0, "%s: %d wrong, first in[%d]=%.9g got %d want %d", tag, bad, first, first < 0 ? 0.0 : in[first],
           first < 0 ? 0 : o[first], first < 0 ? 0 : (int32_t)(double)in[first]);
    if (!bad) {
        bm_pin();
        cvt_ctx x = { f, o, in, N };
        bm_run(tag, cvt_bench, &x, N, 1e6, "Mconv/s", BM_RATE);
        bm_unpin();
    }
    bm_free(in); bm_free(o);
}
TEST(cvt_f32_i32_scalar) {
#ifndef T_X64
    t_note("x86 scalar float->int truncation is x87 FISTP with a control-word switch (compiler sequence)");
#endif
    cvt_case("cvttss_scalar", cvt_scalar);
}
TEST(cvt_f32_i32_sse2)   { BM_NEED(SSE2); cvt_case("cvttps2dq_sse2", cvt_sse2); }
TEST(cvt_f32_i32_avx)    { BM_NEED(AVX); cvt_case("vcvttps2dq_avx_ymm", cvt_avx); }

typedef struct gat_ctx { gat_fn f; const uint32_t *tab; const int32_t *idx; size_t n; uint64_t r; } gat_ctx;
static void gat_bench(void *p) { gat_ctx *x = (gat_ctx *)p; x->r = x->f(x->tab, x->idx, x->n); T_SINK(x->r); }
static void gat_case(const char *tag, gat_fn f) {
    enum { T = 4096, N = 65536 };
    uint32_t *tab = (uint32_t *)bm_alloc(T * 4); int32_t *idx = (int32_t *)bm_alloc(N * 4);
    if (!tab || !idx) { bm_free(tab); bm_free(idx); SKIP("cannot allocate"); }
    t_rng r; t_rng_seed(&r, 99);
    for (int i = 0; i < T; i++) tab[i] = t_rng_u32(&r);
    for (int i = 0; i < N; i++) idx[i] = (int32_t)t_rng_range(&r, T);
    uint64_t want = 0; for (int i = 0; i < N; i++) want += tab[idx[i]];
    uint64_t got = f(tab, idx, N);
    CHECKF(got == want, "%s: sum 0x%llx want 0x%llx", tag, (unsigned long long)got, (unsigned long long)want);
    if (got == want) {
        bm_pin();
        gat_ctx x = { f, tab, idx, N, 0 };
        bm_run(tag, gat_bench, &x, N, 1e6, "Melem/s", BM_RATE);
        CHECK_EQ(x.r, want);
        bm_unpin();
    }
    bm_free(tab); bm_free(idx);
}
TEST(gather_scalar)      { gat_case("gather_scalar_loads", gat_scalar); }
TEST(gather_avx2)        { BM_NEED(AVX2); gat_case("gather_avx2_vpgatherdd", gat_avx2); }
TEST(gather_avx512)      { BM_NEED(AVX512F); gat_case("gather_avx512_vpgatherdd", gat_avx512); }

/* =====================================================================================================
 * memcpy / memset / strlen
 * ===================================================================================================== */
static const size_t g_sweep_sizes[] = { 255, 256, 257, 511, 1000, 1023, 1024, 1025, 4095, 4096, 4097 };
static const int g_sweep_so[] = { 0, 1, 3, 8, 15, 32, 63 };
static const int g_sweep_do[] = { 0, 1, 7, 16, 31, 33, 64 };
#define SWEEP_NSIZES (130 + (int)T_COUNTOF(g_sweep_sizes))
static size_t sweep_size(int i) { return i < 130 ? (size_t)i : g_sweep_sizes[i - 130]; }

static void cpy_sweep(const char *name, cpy_fn f) {
    enum { CAP = 4097 + 64 + 128 + 64 };
    static T_ALIGN(64) uint8_t src[CAP + 64], dst[CAP + 192];
    t_rng r; t_rng_seed(&r, 4242);
    t_rng_fill(&r, src, sizeof src);
    for (size_t a = 0; a < T_COUNTOF(g_sweep_so); a++)
        for (size_t b = 0; b < T_COUNTOF(g_sweep_do); b++) {
            int so = g_sweep_so[a], dof = g_sweep_do[b];
            long bad_size = -1; const char *what = "";
            for (int si = 0; si < SWEEP_NSIZES && bad_size < 0; si++) {
                size_t n = sweep_size(si);
                memset(dst, 0xCC, sizeof dst);
                uint8_t *d = dst + 64 + dof;
                f(d, src + so, n);
                for (int k = 0; k < 64 + dof; k++) if (dst[k] != 0xCC) { bad_size = (long)n; what = "wrote before the destination"; break; }
                if (bad_size < 0 && memcmp(d, src + so, n)) { bad_size = (long)n; what = "copied bytes differ"; }
                if (bad_size < 0) for (size_t k = 0; k < 64; k++) if (d[n + k] != 0xCC) { bad_size = (long)n; what = "wrote past the end"; break; }
            }
            CHECKF(bad_size < 0, "%s src+%d dst+%d size %ld: %s", name, so, dof, bad_size, what);
        }
}
static void set_sweep(const char *name, set_fn f) {
    static T_ALIGN(64) uint8_t dst[4097 + 256];
    for (size_t b = 0; b < T_COUNTOF(g_sweep_do); b++) {
        int dof = g_sweep_do[b];
        long bad_size = -1; const char *what = "";
        for (int si = 0; si < SWEEP_NSIZES && bad_size < 0; si++) {
            size_t n = sweep_size(si); int v = 0x30 + si % 200;
            memset(dst, 0xCC, sizeof dst);
            uint8_t *d = dst + 64 + dof;
            f(d, v, n);
            for (int k = 0; k < 64 + dof; k++) if (dst[k] != 0xCC) { bad_size = (long)n; what = "wrote before the destination"; break; }
            if (bad_size < 0) for (size_t k = 0; k < n; k++) if (d[k] != (uint8_t)v) { bad_size = (long)n; what = "wrong fill byte"; break; }
            if (bad_size < 0) for (size_t k = 0; k < 64; k++) if (d[n + k] != 0xCC) { bad_size = (long)n; what = "wrote past the end"; break; }
        }
        CHECKF(bad_size < 0, "%s dst+%d size %ld: %s", name, dof, bad_size, what);
    }
}
TEST(memcpy_kernels_correctness) {
    cpy_sweep("crt_memcpy", cpy_crt);
    cpy_sweep("sse2", cpy_sse2);
    cpy_sweep("sse2_nt", cpy_sse2_nt);
    cpy_sweep("rep_movsb", cpy_rep_movsb);
    cpy_sweep("rep_movs_word", cpy_rep_movs_word);
    if (cpu_has(F_AVX)) cpy_sweep("avx", cpy_avx); else t_note("AVX not usable: avx kernel not swept");
    if (cpu_has(F_AVX512F)) cpy_sweep("avx512", cpy_avx512); else t_note("AVX-512F not usable: avx512 kernel not swept");
    t_info("erms", "%d", cpu_hw(F_ERMS));
    t_info("fsrm", "%d", cpu_hw(F_FSRM));
}
TEST(memset_kernels_correctness) {
    set_sweep("crt_memset", set_crt);
    set_sweep("sse2", set_sse2);
    set_sweep("sse2_nt", set_sse2_nt);
    set_sweep("rep_stosb", set_rep_stosb);
    if (cpu_has(F_AVX)) set_sweep("avx", set_avx);
    if (cpu_has(F_AVX512F) && cpu_has(F_AVX512BW)) set_sweep("avx512_masked_tail", set_avx512);
}

typedef struct cpy_ctx { cpy_fn f; set_fn s; uint8_t *d; const uint8_t *src; size_t n; int v; } cpy_ctx;
static void cpy_bench(void *p) { cpy_ctx *x = (cpy_ctx *)p; x->f(x->d, x->src, x->n); }
static void set_bench(void *p) { cpy_ctx *x = (cpy_ctx *)p; x->s(x->d, x->v, x->n); }
static void mem_case(const char *tag, cpy_fn f, set_fn sf) {
    uint8_t *src, *dst;
    if (!bm_big(&src, &dst)) SKIP("cannot allocate 2 x %u MB", (unsigned)(bm_big_size() >> 20));
    bm_pin();
    const size_t ns[3] = { 32 * 1024, 256 * 1024, g_bm_big_n };
    for (int s = 0; s < 3; s++) {
        size_t n = ns[s];
        char m[96], sz[32]; snprintf(m, sizeof m, "%s_%s", tag, bm_size_name(n, sz, sizeof sz));
        cpy_ctx x = { f, sf, dst, src, n, 0x3C + s };
        if (f) {
            memset(dst, 0, n);
            f(dst, src, n);
            int ok = !memcmp(dst, src, n);
            CHECKF(ok, "%s %s: destination differs from source", tag, sz);
            if (!ok) continue;
            bm_run(m, cpy_bench, &x, (double)n, 1e9, "GB/s", BM_RATE);
            CHECKF(!memcmp(dst, src, n), "%s %s: destination differs after timing", tag, sz);
        } else {
            memset(dst, ~x.v, n);
            sf(dst, x.v, n);
            size_t bad = 0; for (size_t i = 0; i < n; i++) bad += dst[i] != (uint8_t)x.v;
            CHECKF(bad == 0, "%s %s: %u bytes not 0x%02x", tag, sz, (unsigned)bad, x.v);
            if (bad) continue;
            bm_run(m, set_bench, &x, (double)n, 1e9, "GB/s", BM_RATE);
            bad = 0; for (size_t i = 0; i < n; i++) bad += dst[i] != (uint8_t)x.v;
            CHECKF(bad == 0, "%s %s: %u bytes wrong after timing", tag, sz, (unsigned)bad);
        }
    }
    bm_unpin();
}
TEST(memcpy_crt)         { mem_case("memcpy_crt", cpy_crt, NULL); }
TEST(memcpy_sse2)        { BM_NEED(SSE2); mem_case("memcpy_sse2", cpy_sse2, NULL); }
TEST(memcpy_sse2_nt)     { BM_NEED(SSE2); mem_case("memcpy_sse2_movntdq", cpy_sse2_nt, NULL); }
TEST(memcpy_avx)         { BM_NEED(AVX); mem_case("memcpy_avx_ymm", cpy_avx, NULL); }
TEST(memcpy_avx512)      { BM_NEED(AVX512F); mem_case("memcpy_avx512_zmm", cpy_avx512, NULL); }
TEST(memcpy_rep_movsb)   { mem_case("memcpy_rep_movsb", cpy_rep_movsb, NULL); }
TEST(memcpy_rep_movs_word) { mem_case(sizeof(void *) == 8 ? "memcpy_rep_movsq" : "memcpy_rep_movsd", cpy_rep_movs_word, NULL); }
TEST(memset_crt)         { mem_case("memset_crt", NULL, set_crt); }
TEST(memset_sse2)        { BM_NEED(SSE2); mem_case("memset_sse2", NULL, set_sse2); }
TEST(memset_sse2_nt)     { BM_NEED(SSE2); mem_case("memset_sse2_movntdq", NULL, set_sse2_nt); }
TEST(memset_avx)         { BM_NEED(AVX); mem_case("memset_avx_ymm", NULL, set_avx); }
TEST(memset_avx512)      { BM_NEED(AVX512F); BM_NEED(AVX512BW); mem_case("memset_avx512_zmm", NULL, set_avx512); }
TEST(memset_rep_stosb)   { mem_case("memset_rep_stosb", NULL, set_rep_stosb); }

/* strlen: strings that end exactly at a PAGE_NOACCESS page. Aligned SIMD loads never cross a page, so a
 * correct kernel reads no byte of the guard page whatever the string's alignment. */
TEST(strlen_kernels_correctness) {
    void *base = NULL;
    uint8_t *buf = (uint8_t *)t_guarded_alloc(8192, &base);
    REQUIREF(buf != NULL, "t_guarded_alloc failed");
    struct { const char *n; len_fn f; int ok; } k[] = {
        { "crt", len_crt, 1 }, { "sse2", len_sse2, cpu_has(F_SSE2) }, { "sse42_pcmpistri", len_sse42, cpu_has(F_SSE42) }, { "avx2", len_avx2, cpu_has(F_AVX2) } };
    t_rng r; t_rng_seed(&r, 31337);
    for (size_t i = 0; i < T_COUNTOF(k); i++) {
        if (!k[i].ok) { t_note("%s kernel not usable here", k[i].n); continue; }
        long bad = -1; size_t got = 0;
        for (size_t len = 0; len <= 300 && bad < 0; len++) {         /* terminator is the last byte before the guard */
            char *s = (char *)buf + 8192 - 1 - len;
            for (size_t j = 0; j < len; j++) s[j] = (char)(1 + (j * 37 + len) % 255);
            s[len] = 0;
            got = k[i].f(s);
            if (got != len) bad = (long)len;
        }
        CHECKF(bad < 0, "%s strlen at page end: length %ld returned %u", k[i].n, bad, (unsigned)got);
        bad = -1;
        for (int t = 0; t < 200 && bad < 0; t++) {                  /* interior strings, random start and length */
            size_t off = (size_t)t_rng_range(&r, 4096), len = (size_t)t_rng_range(&r, 3000);
            char *s = (char *)buf + off;
            for (size_t j = 0; j < len; j++) s[j] = (char)(1 + t_rng_range(&r, 255));
            s[len] = 0;
            for (size_t j = len + 1; j < len + 40 && off + j < 8192; j++) s[j] = (char)t_rng_range(&r, 3);   /* zeros after it too */
            got = k[i].f(s);
            if (got != len) bad = (long)len;
        }
        CHECKF(bad < 0, "%s strlen interior: length %ld returned %u", k[i].n, bad, (unsigned)got);
    }
    t_guarded_free(base);
}
typedef struct len_ctx { len_fn f; const char *s; size_t r; } len_ctx;
static void len_bench(void *p) { len_ctx *x = (len_ctx *)p; x->r = x->f(x->s); T_SINK(x->r); }
static void strlen_case(const char *tag, len_fn f) {
    enum { N = 64 * 1024 };
    char *s = (char *)bm_alloc(N + 64);
    if (!s) SKIP("cannot allocate");
    memset(s, 'x', N - 1); s[N - 1] = 0;
    size_t r = f(s);
    CHECKF(r == N - 1, "%s: %u want %u", tag, (unsigned)r, (unsigned)(N - 1));
    if (r == N - 1) {
        bm_pin();
        len_ctx x = { f, s, 0 };
        bm_run(tag, len_bench, &x, N, 1e9, "GB/s", BM_RATE);
        CHECK_EQ(x.r, N - 1);
        bm_unpin();
    }
    bm_free(s);
}
TEST(strlen_crt)            { strlen_case("strlen_crt_64KB", len_crt); }
TEST(strlen_sse2)           { BM_NEED(SSE2); strlen_case("strlen_sse2_64KB", len_sse2); }
TEST(strlen_sse42_pcmpistri) { BM_NEED(SSE42); strlen_case("strlen_sse42_pcmpistri_64KB", len_sse42); }
TEST(strlen_avx2)           { BM_NEED(AVX2); strlen_case("strlen_avx2_64KB", len_avx2); }

/* =====================================================================================================
 * CRC32C
 * ===================================================================================================== */
TEST(crc32c_known_answers) {
    crc_init();
    struct { const char *n; crc_fn f; int ok; } k[] = {
        { "bitwise", crc_bitwise, 1 }, { "bytewise", crc_sw_bytewise, 1 }, { "slice8", crc_sw_slice8, 1 },
        { "sse42_u8", crc_sse42_u8, cpu_has(F_SSE42) }, { "sse42_wide", crc_sse42_wide, cpu_has(F_SSE42) },
        { "sse42_3way", crc_sse42_3way, cpu_has(F_SSE42) } };
    uint8_t v[5][32]; memset(v[0], 0, 32); memset(v[1], 0xFF, 32);
    for (int i = 0; i < 32; i++) { v[2][i] = (uint8_t)i; v[3][i] = (uint8_t)(31 - i); }
    /* RFC 3720 B.4 (iSCSI) test vectors and the "123456789" check value */
    static const uint32_t want[4] = { 0x8A9136AAu, 0x62A8AB43u, 0x46DD794Eu, 0x113FDB5Cu };
    static T_ALIGN(64) uint8_t buf[8192];
    t_rng r; t_rng_seed(&r, 0xC5C);
    t_rng_fill(&r, buf, sizeof buf);
    for (size_t i = 0; i < T_COUNTOF(k); i++) {
        if (!k[i].ok) { t_note("%s not usable (SSE4.2 absent)", k[i].n); continue; }
        uint32_t c = k[i].f((const uint8_t *)"123456789", 9);
        CHECKF(c == 0xE3069283u, "%s(\"123456789\") = 0x%08x want 0xE3069283", k[i].n, c);
        for (int t = 0; t < 4; t++) {
            c = k[i].f(v[t], 32);
            CHECKF(c == want[t], "%s RFC3720 vector %d = 0x%08x want 0x%08x", k[i].n, t, c, want[t]);
        }
        CHECKF(k[i].f(buf, 0) == 0, "%s(empty) = 0x%08x want 0", k[i].n, k[i].f(buf, 0));
        long bad = -1; uint32_t g = 0, w = 0;
        for (size_t n = 1; n <= 600 && bad < 0; n++) {
            size_t off = n % 13;
            g = k[i].f(buf + off, n); w = crc_bitwise(buf + off, n);
            if (g != w) bad = (long)n;
        }
        CHECKF(bad < 0, "%s length %ld: 0x%08x want 0x%08x", k[i].n, bad, g, w);
        bad = -1;
        for (size_t n = 601; n <= 8000 && bad < 0; n += 97) { g = k[i].f(buf, n); w = crc_sw_bytewise(buf, n); if (g != w) bad = (long)n; }
        CHECKF(bad < 0, "%s long length %ld: 0x%08x want 0x%08x", k[i].n, bad, g, w);
    }
}
typedef struct crc_ctx { crc_fn f; const uint8_t *p; size_t n; uint32_t r; } crc_ctx;
static void crc_bench(void *p) { crc_ctx *x = (crc_ctx *)p; x->r = x->f(x->p, x->n); T_SINK(x->r); }
static void crc_case(const char *tag, crc_fn f) {
    crc_init();
    uint8_t *src, *dst;
    if (!bm_big(&src, &dst)) SKIP("cannot allocate");
    bm_pin();
    const size_t ns[2] = { 16 * 1024, (size_t)4 << 20 };
    for (int s = 0; s < 2; s++) {
        size_t n = ns[s];
        uint32_t want = crc_sw_slice8(src, n), got = f(src, n);
        CHECKF(got == want, "%s n=%u: 0x%08x want 0x%08x", tag, (unsigned)n, got, want);
        if (got != want) continue;
        char m[96], sz[32]; snprintf(m, sizeof m, "%s_%s", tag, bm_size_name(n, sz, sizeof sz));
        crc_ctx x = { f, src, n, 0 };
        bm_run(m, crc_bench, &x, (double)n, 1e6, "MB/s", BM_RATE);
        CHECK_EQ(x.r, want);
    }
    bm_unpin();
}
TEST(crc32c_sw_bytewise)  { crc_case("crc32c_sw_table", crc_sw_bytewise); }
TEST(crc32c_sw_slice8)    { crc_case("crc32c_sw_slice8", crc_sw_slice8); }
TEST(crc32c_sse42_u8)     { BM_NEED(SSE42); crc_case("crc32c_sse42_crc32b", crc_sse42_u8); }
TEST(crc32c_sse42_wide)   { BM_NEED(SSE42); crc_case(sizeof(void *) == 8 ? "crc32c_sse42_crc32q" : "crc32c_sse42_crc32d", crc_sse42_wide); }
TEST(crc32c_sse42_3way)   { BM_NEED(SSE42); crc_case(sizeof(void *) == 8 ? "crc32c_sse42_3way_crc32q" : "crc32c_sse42_3way_crc32d", crc_sse42_3way); }

/* =====================================================================================================
 * AES-128
 * ===================================================================================================== */
static void hex16(const char *h, uint8_t *o) { for (int i = 0; i < 16; i++) { unsigned v; sscanf(h + 2 * i, "%2x", &v); o[i] = (uint8_t)v; } }
TEST(aes_known_answers) {
    aes_init();
    CHECK_EQ(g_sbox[0x00], 0x63); CHECK_EQ(g_sbox[0x01], 0x7C); CHECK_EQ(g_sbox[0x53], 0xED); CHECK_EQ(g_sbox[0xFF], 0x16);
    CHECK_EQ(g_inv_sbox[0x63], 0x00); CHECK_EQ(g_inv_sbox[0x16], 0xFF);
    /* FIPS-197 appendix C.1 and appendix B */
    static const char *kv[2][3] = {
        { "000102030405060708090a0b0c0d0e0f", "00112233445566778899aabbccddeeff", "69c4e0d86a7b0430d8cdb78070b4c55a" },
        { "2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734", "3925841d02dc09fbdc118597196a0b32" } };
    int ni = cpu_has(F_AES), vy = cpu_has(F_VAES) && cpu_has(F_AVX2), vz = cpu_has(F_VAES) && cpu_has(F_AVX512F);
    if (!ni) t_note("AES-NI not usable: software only");
    for (int t = 0; t < 2; t++) {
        uint8_t key[16], pt[16], ct[16], out[16];
        hex16(kv[t][0], key); hex16(kv[t][1], pt); hex16(kv[t][2], ct);
        static aes_key ks, kn;
        aes_expand_sw(&ks, key);
        if (t == 1) { uint8_t w40[16]; hex16("d014f9a8c9ee2589e13f0cc8b6630ca6", w40); CHECK_MEM(ks.rk[10], w40, 16); }  /* FIPS-197 A.1 */
        aes_enc_sw_bytewise(&ks, pt, out); CHECK_MEM(out, ct, 16);
        aes_enc_sw_ttable(&ks, pt, out);   CHECK_MEM(out, ct, 16);
        aes_dec_sw_bytewise(&ks, ct, out); CHECK_MEM(out, pt, 16);
        if (ni) {
            aes_expand_ni(&kn, key);
            CHECK_MEM(kn.rk, ks.rk, 176);
            aes_ecb_ni_1way(&kn, pt, out, 1); CHECK_MEM(out, ct, 16);
            uint8_t p8[16 * 16], c8[16 * 16], o8[16 * 16];
            for (int i = 0; i < 16; i++) { memcpy(p8 + 16 * i, pt, 16); memcpy(c8 + 16 * i, ct, 16); }
            aes_ecb_ni_8way(&kn, p8, o8, 16); CHECK_MEM(o8, c8, sizeof o8);
            aes_ecb_ni_dec_8way(&kn, c8, o8, 16); CHECK_MEM(o8, p8, sizeof o8);
            if (vy) { memset(o8, 0, sizeof o8); aes_ecb_vaes_ymm(&kn, p8, o8, 16); CHECK_MEM(o8, c8, sizeof o8); }
            if (vz) { memset(o8, 0, sizeof o8); aes_ecb_vaes_zmm(&kn, p8, o8, 16); CHECK_MEM(o8, c8, sizeof o8); }
        }
    }
    /* random 4 KB: every implementation equals the byte-oriented reference; CBC sw == AES-NI */
    static uint8_t in[4096], ref[4096], o[4096];
    static aes_key k1, k2;
    t_rng r; t_rng_seed(&r, 0xAE5); t_rng_fill(&r, in, sizeof in);
    uint8_t key[16]; t_rng_fill(&r, key, 16);
    aes_expand_sw(&k1, key);
    aes_ecb_sw_bytewise(&k1, in, ref, 256);
    aes_ecb_sw_ttable(&k1, in, o, 256); CHECK_MEM(o, ref, sizeof o);
    if (ni) {
        aes_expand_ni(&k2, key);
        CHECK_MEM(k2.rk, k1.rk, 176);
        memset(o, 0, sizeof o); aes_ecb_ni_1way(&k2, in, o, 256); CHECK_MEM(o, ref, sizeof o);
        memset(o, 0, sizeof o); aes_ecb_ni_8way(&k2, in, o, 253); CHECK_MEM(o, ref, 253 * 16);
        memset(o, 0, sizeof o); aes_ecb_ni_dec_8way(&k2, ref, o, 253); CHECK_MEM(o, in, 253 * 16);
        static uint8_t c1[4096], c2[4096];
        aes_cbc_sw_ttable(&k1, in, c1, 256); aes_cbc_ni(&k2, in, c2, 256); CHECK_MEM(c2, c1, sizeof c1);
        if (vy) { memset(o, 0, sizeof o); aes_ecb_vaes_ymm(&k2, in, o, 251); CHECK_MEM(o, ref, 251 * 16); }
        if (vz) { memset(o, 0, sizeof o); aes_ecb_vaes_zmm(&k2, in, o, 250); CHECK_MEM(o, ref, 250 * 16); }
    }
}
typedef struct aes_ctx { aes_buf_fn f; const aes_key *k; const uint8_t *in; uint8_t *out; size_t nb; } aes_ctx;
static void aes_bench(void *p) { aes_ctx *x = (aes_ctx *)p; x->f(x->k, x->in, x->out, x->nb); }
static void aes_case(const char *tag, aes_buf_fn f, int use_ni_keys, int decrypt, int cbc) {
    aes_init();
    enum { NB = 1024 };                                      /* 16 KB, cache resident */
    static uint8_t in[NB * 16], out[NB * 16], ref[NB * 16];
    static aes_key k;
    t_rng r; t_rng_seed(&r, 0x5EED); t_rng_fill(&r, in, sizeof in);
    uint8_t key[16]; t_rng_fill(&r, key, 16);
    if (use_ni_keys) aes_expand_ni(&k, key); else aes_expand_sw(&k, key);
    if (cbc) aes_cbc_sw_ttable(&k, in, ref, NB);
    else if (decrypt) memcpy(ref, in, sizeof ref);
    else aes_ecb_sw_ttable(&k, in, ref, NB);
    const uint8_t *src = in;
    static uint8_t ctext[NB * 16];
    if (decrypt) { aes_ecb_sw_ttable(&k, in, ctext, NB); src = ctext; }
    memset(out, 0, sizeof out);
    f(&k, src, out, NB);
    int ok = !memcmp(out, ref, sizeof out);
    CHECKF(ok, "%s: output differs from the software reference", tag);
    if (!ok) return;
    bm_pin();
    aes_ctx x = { f, &k, src, out, NB };
    bm_run(tag, aes_bench, &x, NB * 16.0, 1e6, "MB/s", BM_RATE);
    bm_unpin();
    CHECKF(!memcmp(out, ref, sizeof out), "%s: output differs after timing", tag);
}
TEST(aes_sw_bytewise_ecb) { aes_case("aes128_sw_bytewise_ecb_16KB", aes_ecb_sw_bytewise, 0, 0, 0); }
TEST(aes_sw_ttable_ecb)   { aes_case("aes128_sw_ttable_ecb_16KB", aes_ecb_sw_ttable, 0, 0, 0); }
TEST(aes_sw_ttable_cbc)   { aes_case("aes128_sw_ttable_cbc_enc_16KB", aes_cbc_sw_ttable, 0, 0, 1); }
TEST(aesni_ecb_1way)      { BM_NEED(AES); aes_case("aes128_ni_ecb_1way_16KB", aes_ecb_ni_1way, 1, 0, 0); }
TEST(aesni_ecb_8way)      { BM_NEED(AES); aes_case("aes128_ni_ecb_8way_16KB", aes_ecb_ni_8way, 1, 0, 0); }
TEST(aesni_ecb_dec_8way)  { BM_NEED(AES); aes_case("aes128_ni_ecb_dec_8way_16KB", aes_ecb_ni_dec_8way, 1, 1, 0); }
TEST(aesni_cbc_enc)       { BM_NEED(AES); aes_case("aes128_ni_cbc_enc_16KB", aes_cbc_ni, 1, 0, 1); }
TEST(vaes_ecb_ymm)        { BM_NEED(AES); BM_NEED(VAES); BM_NEED(AVX2); aes_case("aes128_vaes_ymm_ecb_16KB", aes_ecb_vaes_ymm, 1, 0, 0); }
TEST(vaes_ecb_zmm)        { BM_NEED(AES); BM_NEED(VAES); BM_NEED(AVX512F); aes_case("aes128_vaes_zmm_ecb_16KB", aes_ecb_vaes_zmm, 1, 0, 0); }

/* =====================================================================================================
 * carry-less multiply
 * ===================================================================================================== */
TEST(clmul_known_answers) {
    uint64_t a[4] = { 3, ~0ull, 1ull << 63, 0x87 }, b[4] = { 3, ~0ull, 1ull << 63, 0x100000001ull }, o[8];
    clmul_sw(a, b, o, 4);
    CHECK_EQ(o[0], 5); CHECK_EQ(o[1], 0);
    CHECK_EQ(o[2], 0x5555555555555555ull); CHECK_EQ(o[3], 0x5555555555555555ull);
    CHECK_EQ(o[4], 0); CHECK_EQ(o[5], 1ull << 62);
    CHECK_EQ(o[6], 0x8700000087ull); CHECK_EQ(o[7], 0);
    if (!cpu_has(F_PCLMULQDQ)) SKIP("PCLMULQDQ not usable (software reference checked)");
    uint64_t p[8]; clmul_pclmul(a, b, p, 4);
    CHECK_MEM(p, o, sizeof o);
    static uint64_t x[1024], y[1024], r1[2048], r2[2048];
    t_rng r; t_rng_seed(&r, 0xC1); t_rng_fill(&r, x, sizeof x); t_rng_fill(&r, y, sizeof y);
    clmul_sw(x, y, r1, 1023); clmul_pclmul(x, y, r2, 1023);
    CHECK_MEM(r2, r1, 1023 * 16);
}
typedef struct cl_ctx { clmul_fn f; const uint64_t *a, *b; uint64_t *o; size_t n; } cl_ctx;
static void cl_bench(void *p) { cl_ctx *x = (cl_ctx *)p; x->f(x->a, x->b, x->o, x->n); }
static void clmul_case(const char *tag, clmul_fn f) {
    enum { N = 1024 };
    static uint64_t a[N], b[N], o[2 * N], ref[2 * N];
    t_rng r; t_rng_seed(&r, 0xC2); t_rng_fill(&r, a, sizeof a); t_rng_fill(&r, b, sizeof b);
    clmul_sw(a, b, ref, N);
    f(a, b, o, N);
    CHECK_MEM(o, ref, sizeof o);
    bm_pin();
    cl_ctx x = { f, a, b, o, N };
    bm_run(tag, cl_bench, &x, N, 1e6, "Mclmul/s", BM_RATE);
    bm_unpin();
}
TEST(clmul_sw_shift_xor) { clmul_case("clmul64_sw", clmul_sw); }
TEST(clmul_pclmulqdq)    { BM_NEED(PCLMULQDQ); clmul_case("clmul64_pclmulqdq", clmul_pclmul); }

/* =====================================================================================================
 * RGB <-> YCbCr
 * ===================================================================================================== */
TEST(yuv_known_values) {
    enum { N = 32 };
    static uint8_t a[N], b[N], c[N], x[N], y[N], z[N];
    static const uint8_t rgb[6][3] = { { 255, 255, 255 }, { 0, 0, 0 }, { 255, 0, 0 }, { 0, 0, 255 }, { 0, 255, 0 }, { 128, 128, 128 } };
    static const uint8_t ycc[6][3] = { { 255, 128, 128 }, { 0, 128, 128 }, { 77, 85, 255 }, { 29, 255, 107 }, { 149, 43, 21 }, { 128, 128, 128 } };
    struct { const char *n; yuv_fn f; int ok; } fw[] = { { "scalar", rgb2yuv_scalar, 1 }, { "sse2", rgb2yuv_sse2, cpu_has(F_SSE2) }, { "avx2", rgb2yuv_avx2, cpu_has(F_AVX2) } };
    for (size_t k = 0; k < T_COUNTOF(fw); k++) {
        if (!fw[k].ok) continue;
        for (int i = 0; i < N; i++) { a[i] = rgb[i % 6][0]; b[i] = rgb[i % 6][1]; c[i] = rgb[i % 6][2]; }
        memset(x, 0xEE, N); memset(y, 0xEE, N); memset(z, 0xEE, N);
        yuv_img im = { a, b, c, x, y, z, N };
        fw[k].f(&im);
        for (int i = 0; i < 6; i++)
            CHECKF(x[i] == ycc[i][0] && y[i] == ycc[i][1] && z[i] == ycc[i][2], "rgb2yuv %s (%u,%u,%u) -> (%u,%u,%u) want (%u,%u,%u)", fw[k].n,
                   rgb[i][0], rgb[i][1], rgb[i][2], x[i], y[i], z[i], ycc[i][0], ycc[i][1], ycc[i][2]);
    }
    static const uint8_t yin[4][3] = { { 128, 128, 128 }, { 255, 128, 128 }, { 0, 128, 128 }, { 128, 0, 255 } };
    static const uint8_t rout[4][3] = { { 128, 128, 128 }, { 255, 255, 255 }, { 0, 0, 0 }, { 255, 81, 0 } };
    struct { const char *n; yuv_fn f; int ok; } bw[] = { { "scalar", yuv2rgb_scalar, 1 }, { "sse2", yuv2rgb_sse2, cpu_has(F_SSE2) }, { "avx2", yuv2rgb_avx2, cpu_has(F_AVX2) } };
    for (size_t k = 0; k < T_COUNTOF(bw); k++) {
        if (!bw[k].ok) continue;
        for (int i = 0; i < N; i++) { a[i] = yin[i % 4][0]; b[i] = yin[i % 4][1]; c[i] = yin[i % 4][2]; }
        yuv_img im = { a, b, c, x, y, z, N };
        bw[k].f(&im);
        for (int i = 0; i < 4; i++)
            CHECKF(x[i] == rout[i][0] && y[i] == rout[i][1] && z[i] == rout[i][2], "yuv2rgb %s (%u,%u,%u) -> (%u,%u,%u) want (%u,%u,%u)", bw[k].n,
                   yin[i][0], yin[i][1], yin[i][2], x[i], y[i], z[i], rout[i][0], rout[i][1], rout[i][2]);
    }
}
/* every one of the 2^24 inputs, both directions: SIMD == scalar bit for bit */
TEST(yuv_exhaustive_simd_vs_scalar) {
    enum { N = 65536 };
    uint8_t *m = (uint8_t *)bm_alloc(N * 12);
    REQUIREF(m != NULL, "cannot allocate");
    uint8_t *a = m, *b = m + N, *c = m + 2 * N, *rx = m + 3 * N, *ry = m + 4 * N, *rz = m + 5 * N, *sx = m + 6 * N, *sy = m + 7 * N, *sz = m + 8 * N;
    uint8_t *qx = m + 9 * N, *qy = m + 10 * N, *qz = m + 11 * N;
    int has2 = cpu_has(F_SSE2), hasa = cpu_has(F_AVX2);
    long bad[2][2] = { { 0, 0 }, { 0, 0 } }; long first[2][2] = { { -1, -1 }, { -1, -1 } };
    int maxerr = 0;
    for (int dir = 0; dir < 2; dir++)
        for (int hi = 0; hi < 256; hi++) {
            for (int i = 0; i < N; i++) { a[i] = (uint8_t)hi; b[i] = (uint8_t)(i >> 8); c[i] = (uint8_t)i; }
            yuv_img ref = { a, b, c, rx, ry, rz, N }, s2 = { a, b, c, sx, sy, sz, N }, s3 = { a, b, c, qx, qy, qz, N };
            (dir ? yuv2rgb_scalar : rgb2yuv_scalar)(&ref);
            if (has2) {
                (dir ? yuv2rgb_sse2 : rgb2yuv_sse2)(&s2);
                for (int i = 0; i < N; i++) if (sx[i] != rx[i] || sy[i] != ry[i] || sz[i] != rz[i]) { if (first[dir][0] < 0) first[dir][0] = (long)hi << 16 | i; bad[dir][0]++; }
            }
            if (hasa) {
                (dir ? yuv2rgb_avx2 : rgb2yuv_avx2)(&s3);
                for (int i = 0; i < N; i++) if (qx[i] != rx[i] || qy[i] != ry[i] || qz[i] != rz[i]) { if (first[dir][1] < 0) first[dir][1] = (long)hi << 16 | i; bad[dir][1]++; }
            }
            if (!dir) {                                   /* round-trip error of the formula pair (INFO only) */
                yuv_img back = { rx, ry, rz, sx, sy, sz, N };
                yuv2rgb_scalar(&back);
                for (int i = 0; i < N; i++) {
                    int e = abs(sx[i] - a[i]); if (abs(sy[i] - b[i]) > e) e = abs(sy[i] - b[i]); if (abs(sz[i] - c[i]) > e) e = abs(sz[i] - c[i]);
                    if (e > maxerr) maxerr = e;
                }
            }
        }
    static const char *dn[2] = { "rgb2yuv", "yuv2rgb" };
    for (int dir = 0; dir < 2; dir++) {
        if (has2) CHECKF(bad[dir][0] == 0, "%s sse2: %ld of 2^24 inputs differ from scalar, first input 0x%06lx", dn[dir], bad[dir][0], first[dir][0]);
        if (hasa) CHECKF(bad[dir][1] == 0, "%s avx2: %ld of 2^24 inputs differ from scalar, first input 0x%06lx", dn[dir], bad[dir][1], first[dir][1]);
    }
    if (!hasa) t_note("AVX2 not usable: only SSE2 compared");
    t_info("roundtrip_max_abs_error", "%d", maxerr);
    bm_free(m);
}
typedef struct yb_ctx { yuv_fn f; yuv_img im; } yb_ctx;
static void yuv_bench(void *p) { yb_ctx *x = (yb_ctx *)p; x->f(&x->im); }
static void yuv_case(const char *tag, yuv_fn f, yuv_fn ref) {
    enum { W = 1920, H = 1080, N = W * H };
    uint8_t *m = (uint8_t *)bm_alloc((size_t)N * 9 + 4096);
    if (!m) SKIP("cannot allocate");
    uint8_t *p[9]; for (int i = 0; i < 9; i++) p[i] = m + (size_t)i * N;
    t_rng r; t_rng_seed(&r, 1080); t_rng_fill(&r, m, (size_t)N * 3);
    yuv_img want = { p[0], p[1], p[2], p[3], p[4], p[5], N }, got = { p[0], p[1], p[2], p[6], p[7], p[8], N };
    ref(&want); f(&got);
    int ok = !memcmp(p[3], p[6], N) && !memcmp(p[4], p[7], N) && !memcmp(p[5], p[8], N);
    CHECKF(ok, "%s: 1920x1080 output differs from the scalar reference", tag);
    if (ok) {
        bm_pin();
        yb_ctx x = { f, got };
        bm_run(tag, yuv_bench, &x, N, 1e6, "MPix/s", BM_RATE);
        bm_unpin();
        CHECKF(!memcmp(p[3], p[6], N) && !memcmp(p[4], p[7], N) && !memcmp(p[5], p[8], N), "%s: output differs after timing", tag);
    }
    bm_free(m);
}
TEST(yuv_rgb2yuv_scalar) { yuv_case("rgb2ycbcr_scalar_1080p", rgb2yuv_scalar, rgb2yuv_scalar); }
TEST(yuv_rgb2yuv_sse2)   { BM_NEED(SSE2); yuv_case("rgb2ycbcr_sse2_1080p", rgb2yuv_sse2, rgb2yuv_scalar); }
TEST(yuv_rgb2yuv_avx2)   { BM_NEED(AVX2); yuv_case("rgb2ycbcr_avx2_1080p", rgb2yuv_avx2, rgb2yuv_scalar); }
TEST(yuv_yuv2rgb_scalar) { yuv_case("ycbcr2rgb_scalar_1080p", yuv2rgb_scalar, yuv2rgb_scalar); }
TEST(yuv_yuv2rgb_sse2)   { BM_NEED(SSE2); yuv_case("ycbcr2rgb_sse2_1080p", yuv2rgb_sse2, yuv2rgb_scalar); }
TEST(yuv_yuv2rgb_avx2)   { BM_NEED(AVX2); yuv_case("ycbcr2rgb_avx2_1080p", yuv2rgb_avx2, yuv2rgb_scalar); }

/* =====================================================================================================
 * AVX <-> SSE transitions
 * ===================================================================================================== */
TEST(transition_avx_sse_semantics) {
    BM_NEED(AVX);
    T_ALIGN(32) uint8_t pat[2][32], out[6][32], want[32];
    t_pattern_reg(pat[0], 32, 0xA7, 0); t_pattern_reg(pat[1], 32, 0xA7, 1);
    memset(out, 0xEE, sizeof out);
    trans_avx_semantics((const uint8_t (*)[32])pat, out);
    CHECK_MEM(out[0], pat[0], 32);                                      /* legacy SSE keeps bits 255:128 */
    memcpy(want, pat[0], 16); memset(want + 16, 0, 16);
    CHECK_MEM(out[1], want, 32);                                        /* VEX.128 zeroes bits 255:128 */
    memcpy(want, pat[0], 16); memcpy(want + 16, pat[1] + 16, 16);
    CHECK_MEM(out[2], want, 32);                                        /* legacy load: low replaced, upper kept */
    memcpy(want, pat[0], 16); memset(want + 16, 0, 16);
    CHECK_MEM(out[3], want, 32);                                        /* VZEROUPPER */
    memset(want, 0, 32);
    CHECK_MEM(out[4], want, 32);                                        /* VZEROALL */
    memset(want, 0, 16); memcpy(want + 16, pat[1] + 16, 16);
    CHECK_MEM(out[5], want, 32);                                        /* legacy MOVAPS after reload */
}
typedef struct tr_case { const char *metric; t_bench_fn fn; int kind; double ops; } tr_case;
TEST(transition_avx_sse_penalty) {
    BM_NEED(AVX);
    static tr_ctx x;
    for (int i = 0; i < 8; i++) x.ones[i] = 1.0f;
    x.K = 4096;
    const tr_case c[5] = {
        { "sse_vex128_addps", trans_avx_vex128, 0, 4.0 * 4096 },
        { "sse_legacy_addps_clean_upper", trans_avx_legacy_clean, 0, 4.0 * 4096 },
        { "sse_legacy_addps_dirty_upper", trans_avx_legacy_dirty, 0, 4.0 * 4096 },
        { "avx256_then_legacy_no_vzeroupper", trans_avx_alternate_novzu, 1, 4096 },
        { "avx256_vzeroupper_legacy", trans_avx_alternate_vzu, 2, 4096 } };
    double v[5] = { 0 };
    bm_pin();
    for (int i = 0; i < 5; i++) {
        memset(x.out, 0, sizeof x.out);
        c[i].fn(&x);
        int bad = 0;
        if (c[i].kind == 0) { for (int r = 0; r < 4; r++) for (int l = 0; l < 4; l++) bad += x.out[r][l] != 4096.0f; }
        else {
            int up = c[i].kind == 1 ? 8 : 4;                            /* kind 2: VZEROUPPER cleared the upper lanes */
            for (int l = 0; l < up; l++) bad += x.out[0][l] != 4096.0f;
            for (int l = 0; l < 4; l++) bad += x.out[1][l] != 4096.0f;
        }
        CHECKF(bad == 0, "%s: %d wrong accumulator lanes (lane0 %.9g, legacy lane0 %.9g)", c[i].metric, bad, x.out[0][0], x.out[1][0]);
        if (bad) continue;
        v[i] = bm_run(c[i].metric, c[i].fn, &x, c[i].ops, 1e9, "ns/op", BM_TIME);
    }
    bm_unpin();
    if (v[1] > 0) t_info("dirty_vs_clean_legacy_ratio", "%.2f", v[2] / v[1]);
    if (v[4] > 0) t_info("alternate_novzu_vs_vzu_ratio", "%.2f", v[3] / v[4]);
}
/* AVX code -> call an SSE-only routine with dirty upper state -> back: result right, ABI state kept */
typedef struct sse_rt { T_ALIGN(64) float a[256], b[256], c[256]; int calls; } sse_rt;
static void sse_routine(void *p) { sse_rt *r = (sse_rt *)p; add_f32_sse(r->c, r->a, r->b, 256); r->calls++; }
typedef struct vcall_ctx { int kind; t_vregs *in, *out; sse_rt *rt; } vcall_ctx;
static void vcall_bench(void *p) { vcall_ctx *v = (vcall_ctx *)p; t_vregs_call(v->kind, v->in, v->out, sse_routine, v->rt); }
TEST(transition_call_sse_routine) {
    BM_NEED(SSE2);
    static t_vregs in, out;
    static sse_rt rt;
    for (int i = 0; i < 256; i++) { rt.a[i] = (float)i; rt.b[i] = (float)(3 * i) * 0.25f; }
    int kinds[2] = { T_VK_XMM, T_VK_YMM }, nk = cpu_has(F_AVX) ? 2 : 1;
    if (nk == 1) t_note("AVX not usable: only the clean (XMM) variant runs");
    bm_pin();
    for (int k = 0; k < nk; k++) {
        t_vregs_pattern(&in, kinds[k], 0x5E0 + (uint32_t)k);
        memset(&out, 0, sizeof out); memset(rt.c, 0, sizeof rt.c);
        REQUIRE(t_vregs_call(kinds[k], &in, &out, sse_routine, &rt) == 0);
        int bad = 0; for (int i = 0; i < 256; i++) bad += rt.c[i] != rt.a[i] + rt.b[i];
        CHECKF(bad == 0, "SSE routine computed %d wrong elements with %s state live", bad, kinds[k] == T_VK_YMM ? "dirty YMM" : "XMM");
#ifdef T_X64
        t_vregs_check(&out, &in, T_VK_XMM, 6, 15, kinds[k] == T_VK_YMM ? "xmm6-15 low across an SSE routine (YMM dirty)" : "xmm6-15 across an SSE routine");
#endif
        vcall_ctx v = { kinds[k], &in, &out, &rt };
        bm_run(kinds[k] == T_VK_YMM ? "call_sse_routine_ymm_dirty" : "call_sse_routine_clean", vcall_bench, &v, 1, 1e9, "ns/call", BM_TIME);
        bad = 0; for (int i = 0; i < 256; i++) bad += rt.c[i] != rt.a[i] + rt.b[i];
        CHECKF(bad == 0, "SSE routine wrong after timing (%d)", bad);
#ifdef T_X64
        t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "xmm6-15 after timed calls");
#endif
    }
    if (cpu_has(F_AVX)) t_vzeroupper();
    bm_unpin();
}

/* =====================================================================================================
 * FXSAVE / XSAVE family
 * ===================================================================================================== */
typedef struct xs_ctx { uint8_t *area; uint64_t rfbm; int op; } xs_ctx;
static void xs_bench(void *p) {
    xs_ctx *x = (xs_ctx *)p;
    switch (x->op) {
    case 0: t_fxsave(x->area); t_fxrstor(x->area); break;
    case 1: t_xsave(x->area, x->rfbm); t_xrstor(x->area, x->rfbm); break;
    case 2: t_xsaveopt(x->area, x->rfbm); t_xrstor(x->area, x->rfbm); break;
    case 3: t_xsavec(x->area, x->rfbm); t_xrstor(x->area, x->rfbm); break;
    }
}
TEST(xsave_fxsave_fxrstor) {
    BM_NEED(FXSR);
    uint8_t *a = (uint8_t *)bm_alloc(4096);
    REQUIRE(a);
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    t_set_mxcsr(0x3F80); t_set_x87_cw(0x037F);
    t_fxsave(a);
    t_set_mxcsr(0x1F80); t_set_x87_cw(0x027F);
    t_fxrstor(a);
    unsigned mx2 = t_get_mxcsr(); unsigned short cw2 = t_get_x87_cw();
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    CHECKF(mx2 == 0x3F80, "MXCSR after FXRSTOR 0x%04x want 0x3F80", mx2);
    CHECKF(cw2 == 0x037F, "x87 CW after FXRSTOR 0x%04x want 0x037F", cw2);
    uint32_t smx; memcpy(&smx, a + 24, 4); CHECKF(smx == 0x3F80, "FXSAVE image MXCSR 0x%04x", smx);
    uint16_t scw; memcpy(&scw, a, 2); CHECKF(scw == 0x037F, "FXSAVE image FCW 0x%04x", scw);
    bm_pin();
    xs_ctx x = { a, 0, 0 };
    bm_run("fxsave_fxrstor_pair", xs_bench, &x, 1, 1e9, "ns/pair", BM_TIME);
    bm_unpin();
    bm_free(a);
}
static void xsave_case(int op, const char *metric) {
    BM_NEED(XSAVE);
    if (op == 2) BM_NEED(XSAVEOPT);
    if (op == 3) BM_NEED(XSAVEC);
    uint64_t xcr0 = cpu()->xcr0, rfbm = xcr0 & ~(uint64_t)T_XS_AMX_STATE;   /* XFD-armed AMX state is opt-in */
    uint32_t size = op == 3 ? xsave_compacted_size(rfbm) : xsave_std_size(rfbm);
    t_info("rfbm", "0x%llx", (unsigned long long)rfbm);
    t_info("area_size", "%u", size);
    uint8_t *a = (uint8_t *)bm_alloc(size + 4096);
    REQUIRE(a);
    unsigned mx = t_get_mxcsr();
    if (op == 2) t_xsave(a, rfbm);                         /* XSAVEOPT needs an area initialised by XSAVE */
    t_set_mxcsr(0x3F80);
    if (op == 1 || op == 2) (op == 1 ? t_xsave : t_xsaveopt)(a, rfbm); else t_xsavec(a, rfbm);
    t_set_mxcsr(0x1F80);
    t_xrstor(a, rfbm);
    unsigned mx2 = t_get_mxcsr();
    t_set_mxcsr(mx);
    CHECKF(mx2 == 0x3F80, "MXCSR after XRSTOR 0x%04x want 0x3F80 (XRSTOR loads MXCSR when RFBM[1] or RFBM[2])", mx2);
    uint64_t bv, comp; memcpy(&bv, a + 512, 8); memcpy(&comp, a + 520, 8);
    t_info("xstate_bv", "0x%llx", (unsigned long long)bv);
    CHECKF((bv & ~rfbm) == 0, "XSTATE_BV 0x%llx has bits outside RFBM 0x%llx", (unsigned long long)bv, (unsigned long long)rfbm);
    if (op == 3) CHECKF(comp == ((1ull << 63) | rfbm), "XSAVEC XCOMP_BV 0x%llx want 0x%llx (bit 63 | RFBM)", (unsigned long long)comp, (unsigned long long)((1ull << 63) | rfbm));
    else CHECKF(comp == 0, "standard-format XCOMP_BV 0x%llx want 0 (XSAVE/XSAVEOPT do not write it)", (unsigned long long)comp);
    uint32_t smx; memcpy(&smx, a + 24, 4);
    CHECKF(smx == 0x3F80, "saved MXCSR 0x%04x want 0x3F80", smx);
    bm_pin();
    xs_ctx x = { a, rfbm, op };
    bm_run(metric, xs_bench, &x, 1, 1e9, "ns/pair", BM_TIME);
    bm_unpin();
    t_set_mxcsr(mx);
    bm_free(a);
}
TEST(xsave_xsave_xrstor)    { xsave_case(1, "xsave_xrstor_pair"); }
TEST(xsave_xsaveopt_xrstor) { xsave_case(2, "xsaveopt_xrstor_pair"); }
TEST(xsave_xsavec_xrstor)   { xsave_case(3, "xsavec_xrstor_pair"); }

/* =====================================================================================================
 * context capture
 * ===================================================================================================== */
typedef struct ctx_worker { bx_spin s; HANDLE th; int ymm; } ctx_worker;
static DWORD WINAPI ctx_worker_proc(LPVOID p) {
    ctx_worker *w = (ctx_worker *)p;
    if (w->ymm) bx_spin_ymm(&w->s); else bx_spin_xmm(&w->s);
    return 0;
}
/* start a worker holding a pattern in XMM/YMM 0..N-1; cpu < 0: no pinning; idle: idle priority */
static int ctx_worker_start(ctx_worker *w, int ymm, uint32_t tag, int cpu, int idle) {
    memset(w, 0, sizeof *w);
    w->ymm = ymm;
    for (int i = 0; i < 16; i++) t_pattern_reg(w->s.vin[i], 32, tag, (uint32_t)i);
    w->th = CreateThread(NULL, 64 * 1024, ctx_worker_proc, w, CREATE_SUSPENDED, NULL);
    if (!w->th) return 0;
    if (cpu >= 0) SetThreadAffinityMask(w->th, (DWORD_PTR)1 << cpu);
    if (idle) SetThreadPriority(w->th, THREAD_PRIORITY_IDLE);
    ResumeThread(w->th);
    double t0 = t_now_ms();
    while (!w->s.ready && t_now_ms() - t0 < 5000) Sleep(1);
    return w->s.ready != 0;
}
static int ctx_worker_stop(ctx_worker *w) {
    if (!w->th) return 0;
    InterlockedExchange(&w->s.stop, 1);
    DWORD r = WaitForSingleObject(w->th, 5000);
    if (r != WAIT_OBJECT_0) TerminateThread(w->th, 1);
    CloseHandle(w->th); w->th = NULL;
    return r == WAIT_OBJECT_0;
}
static int ctx_worker_regs_ok(const ctx_worker *w, const char *label) {
    int bad = 0;
    for (int i = 0; i < T_NXMM; i++) if (memcmp(w->s.vout[i], w->s.vin[i], w->ymm ? 32 : 16)) { if (!bad) CHECK_REG(label, w->s.vout[i], w->s.vin[i], w->ymm ? 32 : 16); bad++; }
    CHECKF(bad == 0, "%s: %d of %d %s registers changed", label, bad, T_NXMM, w->ymm ? "YMM" : "XMM");
    return bad == 0;
}
static const uint8_t *ctx_xmm(const CONTEXT *c, int i) {
#ifdef T_X64
    return (const uint8_t *)&c->FltSave.XmmRegisters[i];
#else
    return (const uint8_t *)c->ExtendedRegisters + 160 + 16 * i;
#endif
}
/* XMM (and, with ymm_hi, YMM upper halves) in a context equal the worker's pattern */
static int ctx_check_regs(const CONTEXT *c, const ctx_worker *w, const uint8_t *ymm_hi, const char *label) {
    int bad = 0;
    for (int i = 0; i < T_NXMM; i++) {
        if (memcmp(ctx_xmm(c, i), w->s.vin[i], 16)) bad++;
        if (ymm_hi && memcmp(ymm_hi + 16 * i, w->s.vin[i] + 16, 16)) bad++;
    }
    if (bad) {
        for (int i = 0; i < T_NXMM; i++) if (memcmp(ctx_xmm(c, i), w->s.vin[i], 16)) { CHECK_REG(label, ctx_xmm(c, i), w->s.vin[i], 16); break; }
        for (int i = 0; ymm_hi && i < T_NXMM; i++) if (memcmp(ymm_hi + 16 * i, w->s.vin[i] + 16, 16)) { CHECK_REG(label, ymm_hi + 16 * i, w->s.vin[i] + 16, 16); break; }
    }
    CHECKF(bad == 0, "%s: %d register halves differ from the worker's pattern", label, bad);
    return bad == 0;
}

typedef struct cap_ctx { CONTEXT *c; } cap_ctx;
static void cap_bench(void *p) { RtlCaptureContext(((cap_ctx *)p)->c); }
TEST(ctx_rtlcapturecontext) {
    static T_ALIGN(64) CONTEXT c;
    memset(&c, 0, sizeof c);
    RtlCaptureContext(&c);
    NT_TIB *tib;                                            /* TEB self pointer (NT_TIB.Self) */
#ifdef T_X64
    __asm__("movq %%gs:0x30, %0" : "=r"(tib));
#else
    __asm__("movl %%fs:0x18, %0" : "=r"(tib));
#endif
#ifdef T_X64
    CHECKF(c.ContextFlags == 0x10000F, "ContextFlags 0x%lx want 0x10000F (CONTROL|INTEGER|SEGMENTS|FLOATING_POINT)", c.ContextFlags);
    CHECKF(c.MxCsr == t_get_mxcsr(), "captured MxCsr 0x%lx, live 0x%x", c.MxCsr, t_get_mxcsr());
    CHECKF(c.FltSave.MxCsr == c.MxCsr, "FltSave.MxCsr 0x%lx != MxCsr 0x%lx", c.FltSave.MxCsr, c.MxCsr);
    uintptr_t sp = (uintptr_t)c.Rsp, ip = (uintptr_t)c.Rip;
#else
    CHECKF((c.ContextFlags & (CONTEXT_i386 | 3)) == (CONTEXT_i386 | 3), "ContextFlags 0x%lx lacks i386|CONTROL|INTEGER", c.ContextFlags);
    uintptr_t sp = (uintptr_t)c.Esp, ip = (uintptr_t)c.Eip;
#endif
    t_info("context_flags", "0x%lx", c.ContextFlags);
    CHECKF(sp >= (uintptr_t)tib->StackLimit && sp < (uintptr_t)tib->StackBase, "captured SP %p outside the thread stack [%p, %p)", (void *)sp, tib->StackLimit, tib->StackBase);
    CHECKF(ip != 0, "captured IP is 0");
    bm_pin();
    cap_ctx x = { &c };
    bm_run("rtlcapturecontext", cap_bench, &x, 1, 1e6, "Mcaptures/s", BM_RATE);
    bm_unpin();
}

/* x64 RtlCaptureContext captures the caller's non-volatile XMM6-15 (the unwinder restores them from this
 * record: RtlUnwindEx / RtlRestoreContext). ntdll 19041 RtlCaptureContext: FXSAVE [rcx+0x100] before any
 * other vector use, STMXCSR [rcx+0x34], ContextFlags = 0x10000F (corpus explorer, ntdll!RtlCaptureContext
 * asm). Only XMM6-15 / MXCSR / FCW are hard checks; XMM0-5 are volatile at the call (INFO). */
#ifdef T_X64
static const uint8_t *fx_xmm(const XMM_SAVE_AREA32 *f, int i) { return (const uint8_t *)&f->XmmRegisters[i]; }
static int cap_xmm_compare(const CONTEXT *c, const uint8_t (*pat)[32], const char *label) {
    int bad = 0, vol = 0;
    for (int i = 0; i < 16; i++) {
        int d = memcmp(fx_xmm(&c->FltSave, i), pat[i], 16) != 0;
        if (i >= 6) { if (d && !bad) CHECK_REG(label, fx_xmm(&c->FltSave, i), pat[i], 16); bad += d; }
        else vol += d;
    }
    CHECKF(bad == 0, "%s: %d of XMM6-15 differ from the caller's registers", label, bad);
    t_info("volatile_xmm0_5_differing", "%d", vol);
    return bad == 0;
}
#endif
TEST(ctx_rtlcapturecontext_nonvolatile_xmm) {
#ifndef T_X64
    SKIP("x86 RtlCaptureContext captures CONTROL|INTEGER|SEGMENTS only (no floating-point state)");
#else
    static T_ALIGN(64) CONTEXT c;
    static T_ALIGN(32) uint8_t pat[16][32];
    for (int i = 0; i < 16; i++) t_pattern_reg(pat[i], 32, 0xCA, (uint32_t)i);
    memset(&c, 0, sizeof c);
    cap_call_xmm((void *)RtlCaptureContext, &c, (const uint8_t (*)[32])pat);
    CHECKF(c.ContextFlags == 0x10000F, "ContextFlags 0x%lx want 0x10000F", c.ContextFlags);
    cap_xmm_compare(&c, (const uint8_t (*)[32])pat, "RtlCaptureContext FltSave XMM6-15");
    CHECKF(c.FltSave.ControlWord == t_get_x87_cw(), "FltSave.ControlWord 0x%04x, live FCW 0x%04x", c.FltSave.ControlWord, t_get_x87_cw());
    CHECKF(c.MxCsr == t_get_mxcsr(), "MxCsr 0x%lx, live 0x%x", c.MxCsr, t_get_mxcsr());
    /* a non-default MXCSR / FCW must be captured as is, not normalised */
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    t_set_mxcsr(0x7F80 | 0x8000); t_set_x87_cw(0x0B7F);
    memset(&c, 0, sizeof c);
    cap_call_xmm((void *)RtlCaptureContext, &c, (const uint8_t (*)[32])pat);
    unsigned short fcw = c.FltSave.ControlWord; DWORD m1 = c.MxCsr, m2 = c.FltSave.MxCsr;
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    CHECKF(m1 == 0xFF80 && m2 == 0xFF80, "MxCsr 0x%lx / FltSave.MxCsr 0x%lx want 0xFF80 (FZ | RC=11 | masks)", m1, m2);
    CHECKF(fcw == 0x0B7F, "FltSave.ControlWord 0x%04x want 0x0B7F", fcw);
#endif
}


typedef struct sr_ctx { HANDLE th; int fails; } sr_ctx;
static void sr_bench(void *p) {
    sr_ctx *x = (sr_ctx *)p;
    DWORD a = SuspendThread(x->th), b = ResumeThread(x->th);
    if (a != 0 || b != 1) x->fails++;
}
TEST(ctx_suspend_resume) {
    int cpu = bm_pin();
    ctx_worker *w = (ctx_worker *)bm_alloc(sizeof *w);
    REQUIRE(w);
    if (!ctx_worker_start(w, 0, 0x51, bm_other_cpu(cpu), 0)) { ctx_worker_stop(w); bm_free(w); bm_unpin(); SKIP("could not start a worker thread"); }
    sr_ctx x = { w->th, 0 };
    bm_run("suspend_resume_pair", sr_bench, &x, 1, 1e6, "us/pair", BM_TIME);
    CHECKF(x.fails == 0, "SuspendThread/ResumeThread returned unexpected previous counts %d times (want 0 and 1)", x.fails);
    CHECK(ctx_worker_stop(w));
    ctx_worker_regs_ok(w, "worker XMM after suspend/resume cycles");
    bm_free(w);
    bm_unpin();
}

typedef struct gc_ctx {
    HANDLE th; CONTEXT *c; DWORD flags; DWORD64 mask; int fails, set;
    BOOL (WINAPI *setmask)(PCONTEXT, DWORD64);
} gc_ctx;
static void gc_bench(void *p) {
    gc_ctx *x = (gc_ctx *)p;
    if (SuspendThread(x->th) == (DWORD)-1) { x->fails++; return; }
    x->c->ContextFlags = x->flags;
    if (x->setmask && !x->setmask(x->c, x->mask)) x->fails++;
    if (!GetThreadContext(x->th, x->c)) x->fails++;
    else if (x->set && !SetThreadContext(x->th, x->c)) x->fails++;
    ResumeThread(x->th);
}
/* XSTATE CONTEXT through the documented negotiation */
typedef struct xctx { void *buf; CONTEXT *c; DWORD len; } xctx;
static int xctx_init(xctx *x, DWORD flags, DWORD64 mask) {
    __typeof__(&InitializeContext) pInit = T_PROC("kernel32.dll", InitializeContext);
    __typeof__(&SetXStateFeaturesMask) pSet = T_PROC("kernel32.dll", SetXStateFeaturesMask);
    memset(x, 0, sizeof *x);
    if (!pInit || !pSet) return 0;
    DWORD len = 0;
    SetLastError(0);
    BOOL ok = pInit(NULL, flags, NULL, &len);
    DWORD e = GetLastError();
    CHECKF(!ok && e == ERROR_INSUFFICIENT_BUFFER && len > 0, "InitializeContext(NULL) ok=%d err=%lu len=%lu (want FALSE, ERROR_INSUFFICIENT_BUFFER, len > 0)", ok, e, len);
    if (ok || !len) return 0;
    x->buf = malloc(len);
    if (!x->buf) return 0;
    x->len = len;
    if (!pInit(x->buf, flags, &x->c, &len)) { CHECKF(0, "InitializeContext(buffer of %lu) failed: %lu", x->len, GetLastError()); return 0; }
    if (!pSet(x->c, mask)) { CHECKF(0, "SetXStateFeaturesMask(0x%llx) failed: %lu", (unsigned long long)mask, GetLastError()); return 0; }
    return 1;
}
static void get_ctx_case(const char *metric, int xstate, DWORD64 mask, int set) {
    int ymm = xstate && cpu_has(F_AVX);
    if (xstate) {
        if (!cpu_has(F_AVX)) SKIP("AVX not usable: no YMM state to capture");
        if (mask & XSTATE_MASK_AVX512) BM_NEED(AVX512F);
        if (!T_PROC("kernel32.dll", InitializeContext) || !T_PROC("kernel32.dll", LocateXStateFeature)) SKIP("XSTATE context APIs not exported (pre-Windows 7 SP1)");
        __typeof__(&GetEnabledXStateFeatures) pEn = T_PROC("kernel32.dll", GetEnabledXStateFeatures);
        DWORD64 en = pEn ? pEn() : 0;
        t_info("enabled_xstate_features", "0x%llx", (unsigned long long)en);
        if ((en & mask) != mask) {
            if (t_is_wine()) XFAIL("Wine reports enabled XSTATE features 0x%llx, lacking 0x%llx", (unsigned long long)en, (unsigned long long)mask);
            CHECKF(0, "CPU state usable (XCR0 0x%llx) but GetEnabledXStateFeatures = 0x%llx lacks 0x%llx",
                   (unsigned long long)cpu()->xcr0, (unsigned long long)en, (unsigned long long)mask);
            return;
        }
    }
    int cpu = bm_pin();
    static T_ALIGN(64) CONTEXT plain;
    xctx xc = { 0 };
    CONTEXT *c = &plain;
    DWORD flags = CX_FP_FLAGS;
    if (xstate) {
        flags |= CONTEXT_XSTATE;
        if (!xctx_init(&xc, flags, mask)) { free(xc.buf); bm_unpin(); return; }
        c = xc.c;
    }
    ctx_worker *w = (ctx_worker *)bm_alloc(sizeof *w);
    REQUIRE(w);
    if (!ctx_worker_start(w, ymm, 0xC0 + (uint32_t)xstate, bm_other_cpu(cpu), 0)) { ctx_worker_stop(w); bm_free(w); free(xc.buf); bm_unpin(); SKIP("could not start a worker thread"); }
    gc_ctx x = { w->th, c, flags, mask, 0, set, xstate ? T_PROC("kernel32.dll", SetXStateFeaturesMask) : NULL };
    __typeof__(&LocateXStateFeature) pLoc = T_PROC("kernel32.dll", LocateXStateFeature);
    __typeof__(&GetXStateFeaturesMask) pGetMask = T_PROC("kernel32.dll", GetXStateFeaturesMask);
    for (int pass = 0; pass < 2; pass++) {
        if (pass) bm_run(metric, gc_bench, &x, 1, 1e6, "us/op", BM_TIME);
        else gc_bench(&x);
        CHECKF(x.fails == 0, "%s: Suspend/Get%sThreadContext failed %d times (last error %lu)", metric, set ? "/Set" : "", x.fails, GetLastError());
        const uint8_t *hi = NULL;
        if (xstate && pLoc) {
            DWORD len = 0;
            hi = (const uint8_t *)pLoc(c, XSTATE_AVX, &len);
            CHECKF(hi != NULL && len >= 16u * T_NXMM, "LocateXStateFeature(XSTATE_AVX) = %p len %lu (want >= %u)", (const void *)hi, len, 16u * T_NXMM);
            if (hi && len < 16u * T_NXMM) hi = NULL;
            DWORD64 m = 0;
            if (pGetMask && pGetMask(c, &m)) CHECKF(m & XSTATE_MASK_AVX, "GetXStateFeaturesMask = 0x%llx: AVX not reported for a thread with non-zero YMM upper halves", (unsigned long long)m);
        }
        ctx_check_regs(c, w, hi, pass ? "context after timed captures" : "first captured context");
        if (x.fails) break;
    }
    CHECK(ctx_worker_stop(w));
    ctx_worker_regs_ok(w, set ? "worker registers after Get/SetThreadContext round trips" : "worker registers after GetThreadContext calls");
    bm_free(w);
    free(xc.buf);
    bm_unpin();
}
TEST(ctx_getthreadcontext_fp)           { get_ctx_case("getthreadcontext_full_fp", 0, 0, 0); }
TEST(ctx_getthreadcontext_xstate_avx)   { get_ctx_case("getthreadcontext_xstate_avx", 1, XSTATE_MASK_AVX, 0); }
TEST(ctx_getthreadcontext_xstate_avx512) { get_ctx_case("getthreadcontext_xstate_avx512", 1, XSTATE_MASK_AVX | XSTATE_MASK_AVX512, 0); }
TEST(ctx_setthreadcontext_fp_roundtrip) { get_ctx_case("get_set_threadcontext_full_fp", 0, 0, 1); }
TEST(ctx_setthreadcontext_xstate_roundtrip) { get_ctx_case("get_set_threadcontext_xstate_avx", 1, XSTATE_MASK_AVX, 1); }

TEST(ctx_getthreadcontext_control) {
    int cpu = bm_pin();
    ctx_worker *w = (ctx_worker *)bm_alloc(sizeof *w);
    REQUIRE(w);
    if (!ctx_worker_start(w, 0, 0xC7, bm_other_cpu(cpu), 0)) { ctx_worker_stop(w); bm_free(w); bm_unpin(); SKIP("could not start a worker thread"); }
    static T_ALIGN(64) CONTEXT c;
    gc_ctx x = { w->th, &c, CONTEXT_CONTROL, 0, 0, 0, NULL };
    bm_run("getthreadcontext_control", gc_bench, &x, 1, 1e6, "us/op", BM_TIME);
    CHECKF(x.fails == 0, "Suspend/GetThreadContext(CONTEXT_CONTROL) failed %d times", x.fails);
#ifdef T_X64
    uintptr_t ip = (uintptr_t)c.Rip, sp = (uintptr_t)c.Rsp;
#else
    uintptr_t ip = (uintptr_t)c.Eip, sp = (uintptr_t)c.Esp;
#endif
    CHECKF(ip >= (uintptr_t)bx_spin_xmm && ip < (uintptr_t)bx_spin_xmm + 4096, "worker IP %p is not inside its spin routine (%p)", (void *)ip, (void *)bx_spin_xmm);
    CHECKF(sp != 0, "worker SP is 0");
    CHECK(ctx_worker_stop(w));
    ctx_worker_regs_ok(w, "worker registers after CONTEXT_CONTROL captures");
    bm_free(w);
    bm_unpin();
}

typedef struct ic_ctx { void *buf; DWORD len, flags; int fails; } ic_ctx;
static void ic_bench(void *p) {
    ic_ctx *x = (ic_ctx *)p;
    static __typeof__(&InitializeContext) pInit; static __typeof__(&SetXStateFeaturesMask) pSet; static __typeof__(&LocateXStateFeature) pLoc;
    if (!pInit) { pInit = T_PROC("kernel32.dll", InitializeContext); pSet = T_PROC("kernel32.dll", SetXStateFeaturesMask); pLoc = T_PROC("kernel32.dll", LocateXStateFeature); }
    CONTEXT *c = NULL; DWORD len = x->len, l2 = 0;
    if (!pInit(x->buf, x->flags, &c, &len) || !pSet(c, XSTATE_MASK_AVX) || !pLoc(c, XSTATE_AVX, &l2)) x->fails++;
}
TEST(ctx_initializecontext_xstate) {
    BM_NEED(AVX);
    if (!T_PROC("kernel32.dll", InitializeContext) || !T_PROC("kernel32.dll", LocateXStateFeature)) SKIP("XSTATE context APIs not exported");
    xctx xc;
    if (!xctx_init(&xc, CX_FP_FLAGS | CONTEXT_XSTATE, XSTATE_MASK_AVX)) { free(xc.buf); return; }
    t_info("xstate_context_length", "%lu", xc.len);
    bm_pin();
    ic_ctx x = { xc.buf, xc.len, CX_FP_FLAGS | CONTEXT_XSTATE, 0 };
    bm_run("initializecontext_setmask_locate", ic_bench, &x, 1, 1e9, "ns/op", BM_TIME);
    CHECKF(x.fails == 0, "InitializeContext/SetXStateFeaturesMask/LocateXStateFeature failed %d times", x.fails);
    bm_unpin();
    free(xc.buf);
}

typedef struct cc_ctx { CONTEXT *dst, *src; DWORD flags; int fails; } cc_ctx;
static void cc_bench(void *p) {
    cc_ctx *x = (cc_ctx *)p;
    static __typeof__(&CopyContext) pCopy;
    if (!pCopy) pCopy = T_PROC("kernel32.dll", CopyContext);
    if (!pCopy(x->dst, x->flags, x->src)) x->fails++;
}
TEST(ctx_copycontext_xstate) {
    BM_NEED(AVX);
    if (!T_PROC("kernel32.dll", CopyContext) || !T_PROC("kernel32.dll", LocateXStateFeature)) SKIP("CopyContext / XSTATE APIs not exported");
    DWORD flags = CX_FP_FLAGS | CONTEXT_XSTATE;
    xctx s, d;
    if (!xctx_init(&s, flags, XSTATE_MASK_AVX) || !xctx_init(&d, flags, XSTATE_MASK_AVX)) { free(s.buf); free(d.buf); return; }
    int cpu = bm_pin();
    ctx_worker *w = (ctx_worker *)bm_alloc(sizeof *w);
    REQUIRE(w);
    if (!ctx_worker_start(w, 1, 0xCC, bm_other_cpu(cpu), 0)) { ctx_worker_stop(w); bm_free(w); free(s.buf); free(d.buf); bm_unpin(); SKIP("could not start a worker thread"); }
    gc_ctx g = { w->th, s.c, flags, XSTATE_MASK_AVX, 0, 0, T_PROC("kernel32.dll", SetXStateFeaturesMask) };
    gc_bench(&g);
    CHECK(ctx_worker_stop(w));
    CHECKF(g.fails == 0, "capturing the source context failed");
    __typeof__(&LocateXStateFeature) pLoc = T_PROC("kernel32.dll", LocateXStateFeature);
    DWORD ls = 0, ld = 0;
    const uint8_t *hs = (const uint8_t *)pLoc(s.c, XSTATE_AVX, &ls);
    if (!g.fails) ctx_check_regs(s.c, w, hs, "source context");
    cc_ctx x = { d.c, s.c, flags, 0 };
    cc_bench(&x);
    CHECKF(x.fails == 0, "CopyContext(XSTATE) failed: %lu", GetLastError());
    const uint8_t *hd = (const uint8_t *)pLoc(d.c, XSTATE_AVX, &ld);
    CHECKF(hd && ld == ls, "destination AVX area %p len %lu (source len %lu)", (const void *)hd, ld, ls);
    if (!x.fails && hd && hs && ld == ls) ctx_check_regs(d.c, w, hd, "CopyContext destination");
    bm_run("copycontext_xstate_avx", cc_bench, &x, 1, 1e9, "ns/op", BM_TIME);
    CHECKF(x.fails == 0, "CopyContext failed %d times", x.fails);
    bm_free(w);
    free(s.buf); free(d.buf);
    bm_unpin();
}

/* RtlCaptureContext2 (ntdll, Windows 10 2004+, x64): RtlCaptureContext plus extended state. ntdll 19041
 * (corpus explorer): FXSAVE + STMXCSR as RtlCaptureContext; then RtlpSanitizeContextFlags(&Flags, 1); if
 * (Flags & 0x100040) == 0x100040 RtlpMergeContextXState merges ONLY the CET user shadow-stack component
 * (XSTATE 11); finally Flags |= 0x10000F. It does not XSAVE the AVX component in 19041, so the YMM upper
 * halves in the record are INFO, never a check. Hard checks: XMM6-15, MXCSR, flags 0x10000F. */
typedef VOID (WINAPI *rcc2_fn)(PCONTEXT);
typedef struct cap2_ctx { rcc2_fn f; CONTEXT *c; DWORD flags; } cap2_ctx;
static void cap2_bench(void *p) { cap2_ctx *x = (cap2_ctx *)p; x->c->ContextFlags = x->flags; x->f(x->c); }
TEST(ctx_rtlcapturecontext2_xstate) {
#ifndef T_X64
    SKIP("RtlCaptureContext2 is an x64 export");
#else
    rcc2_fn f = (rcc2_fn)t_proc("ntdll.dll", "RtlCaptureContext2");
    if (!f) SKIP("ntdll!RtlCaptureContext2 not exported (before Windows 10 2004)");
    BM_NEED(AVX);
    if (!T_PROC("kernel32.dll", InitializeContext) || !T_PROC("kernel32.dll", LocateXStateFeature)) SKIP("XSTATE context APIs not exported");
    DWORD flags = CX_FP_FLAGS | CONTEXT_XSTATE;
    xctx xc;
    if (!xctx_init(&xc, flags, XSTATE_MASK_AVX)) { free(xc.buf); return; }
    static T_ALIGN(32) uint8_t pat[16][32];
    for (int i = 0; i < 16; i++) t_pattern_reg(pat[i], 32, 0xC2, (uint32_t)i);
    xc.c->ContextFlags = flags;
    cap_call_ymm((void *)f, xc.c, (const uint8_t (*)[32])pat);
    t_info("context_flags_after", "0x%lx", xc.c->ContextFlags);
    CHECKF((xc.c->ContextFlags & 0x10000F) == 0x10000F, "ContextFlags 0x%lx lacks CONTROL|INTEGER|SEGMENTS|FLOATING_POINT", xc.c->ContextFlags);
    if (!(xc.c->ContextFlags & 0x40)) t_note("CONTEXT_XSTATE dropped from ContextFlags (0x%lx)", xc.c->ContextFlags);
    cap_xmm_compare(xc.c, (const uint8_t (*)[32])pat, "RtlCaptureContext2 FltSave XMM6-15");
    CHECKF(xc.c->MxCsr == t_get_mxcsr(), "MxCsr 0x%lx, live 0x%x", xc.c->MxCsr, t_get_mxcsr());
    __typeof__(&LocateXStateFeature) pLoc = T_PROC("kernel32.dll", LocateXStateFeature);
    DWORD len = 0; const uint8_t *hi = (const uint8_t *)pLoc(xc.c, XSTATE_AVX, &len);
    int same = 0;
    if (hi && len >= 256) for (int i = 6; i < 16; i++) same += !memcmp(hi + 16 * i, pat[i] + 16, 16);
    t_info("ymm6_15_upper_captured", "%d of 10 (19041 does not capture AVX here; INFO only)", same);
    bm_pin();
    cap2_ctx x = { f, xc.c, flags };
    bm_run("rtlcapturecontext2_xstate", cap2_bench, &x, 1, 1e6, "Mcaptures/s", BM_RATE);
    bm_unpin();
    free(xc.buf);
#endif
}

/* =====================================================================================================
 * scheduler: yields, event ping-pong (with live SIMD state), thread creation
 * ===================================================================================================== */
static void sleep0_bench(void *p) { (void)p; for (int i = 0; i < 16; i++) Sleep(0); }
static void yield_bench(void *p) { (void)p; for (int i = 0; i < 16; i++) SwitchToThread(); }
TEST(sched_sleep0_rate)        { bm_pin(); bm_run("sleep0_alone", sleep0_bench, NULL, 16, 1e6, "Mcalls/s", BM_RATE); bm_unpin(); }
TEST(sched_switchtothread_rate) { bm_pin(); bm_run("switchtothread_alone", yield_bench, NULL, 16, 1e6, "Mcalls/s", BM_RATE); bm_unpin(); }

typedef struct pp_ctx { HANDLE e1, e2, th; volatile LONG quit; volatile LONG served; LONG sent; int fails; int k; } pp_ctx;
static DWORD WINAPI pp_worker(LPVOID p) {
    pp_ctx *x = (pp_ctx *)p;
    for (;;) {
        if (WaitForSingleObject(x->e1, 5000) != WAIT_OBJECT_0) { InterlockedIncrement((volatile LONG *)&x->fails); return 1; }
        if (x->quit) return 0;
        InterlockedIncrement(&x->served);
        SetEvent(x->e2);
    }
}
static void pp_bench(void *p) {
    pp_ctx *x = (pp_ctx *)p;
    for (int i = 0; i < x->k && !x->fails; i++) {
        x->sent++;
        SetEvent(x->e1);
        if (WaitForSingleObject(x->e2, 5000) != WAIT_OBJECT_0) x->fails++;
    }
}
static int pp_start(pp_ctx *x, int cpu, int k) {
    memset(x, 0, sizeof *x);
    x->k = k;
    x->e1 = CreateEventA(NULL, FALSE, FALSE, NULL); x->e2 = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (!x->e1 || !x->e2) return 0;
    x->th = CreateThread(NULL, 64 * 1024, pp_worker, x, CREATE_SUSPENDED, NULL);
    if (!x->th) return 0;
    if (cpu >= 0) SetThreadAffinityMask(x->th, (DWORD_PTR)1 << cpu);
    ResumeThread(x->th);
    return 1;
}
static void pp_stop(pp_ctx *x) {
    if (x->th) {
        x->quit = 1; SetEvent(x->e1);
        if (WaitForSingleObject(x->th, 5000) != WAIT_OBJECT_0) { TerminateThread(x->th, 1); x->fails++; }
        CloseHandle(x->th);
    }
    if (x->e1) CloseHandle(x->e1);
    if (x->e2) CloseHandle(x->e2);
}
static void pp_case(const char *metric, int same_cpu) {
    int cpu = bm_pin();
    if (cpu < 0) SKIP("cannot pin the case thread");
    int wcpu = same_cpu ? cpu : bm_other_cpu(cpu);
    if (wcpu < 0) { bm_unpin(); SKIP("only one processor in the affinity mask"); }
    static pp_ctx x;
    if (!pp_start(&x, wcpu, 64)) { pp_stop(&x); bm_unpin(); SKIP("cannot create events/thread: %lu", GetLastError()); }
    bm_run(metric, pp_bench, &x, 64, 1e3, "k_roundtrips/s", BM_RATE);
    pp_stop(&x);
    CHECKF(x.fails == 0, "%d event waits timed out (5 s)", x.fails);
    CHECKF(x.served == x.sent, "worker served %ld of %ld round trips", x.served, x.sent);
    bm_unpin();
}
TEST(sched_event_pingpong_same_cpu)  { pp_case("event_pingpong_same_cpu", 1); }
TEST(sched_event_pingpong_cross_cpu) { pp_case("event_pingpong_cross_cpu", 0); }

/* SwitchToThread ping-pong on one processor: the documented behaviour is only "yield to a ready thread
 * on the current processor", with no fairness guarantee, so lack of progress is a NOTE, not a FAIL. */
typedef struct yp_ctx { volatile LONG turn, quit; LONG served; HANDLE th; int stalled; } yp_ctx;
static DWORD WINAPI yp_worker(LPVOID p) {
    yp_ctx *x = (yp_ctx *)p;
    while (!x->quit) { if (x->turn == 1) { x->served++; InterlockedExchange(&x->turn, 0); } else SwitchToThread(); }
    return 0;
}
static void yp_bench(void *p) {
    yp_ctx *x = (yp_ctx *)p;
    for (int i = 0; i < 64 && !x->stalled; i++) {
        InterlockedExchange(&x->turn, 1);
        double t0 = t_now_ms();
        while (x->turn == 1) { SwitchToThread(); if (t_now_ms() - t0 > 2000) { x->stalled = 1; break; } }
    }
}
TEST(sched_switchtothread_pingpong) {
    int cpu = bm_pin();
    if (cpu < 0) SKIP("cannot pin the case thread");
    static yp_ctx x; memset((void *)&x, 0, sizeof x);
    x.th = CreateThread(NULL, 64 * 1024, yp_worker, &x, CREATE_SUSPENDED, NULL);
    if (!x.th) { bm_unpin(); SKIP("CreateThread failed: %lu", GetLastError()); }
    SetThreadAffinityMask(x.th, (DWORD_PTR)1 << cpu);
    ResumeThread(x.th);
    yp_bench(&x);
    if (!x.stalled) bm_run("switchtothread_pingpong_same_cpu", yp_bench, &x, 64, 1e3, "k_roundtrips/s", BM_RATE);
    if (x.stalled) t_note("SwitchToThread did not hand the processor to the ready peer within 2 s (no fairness guarantee; not a failure)");
    x.quit = 1;
    CHECK(WaitForSingleObject(x.th, 5000) == WAIT_OBJECT_0);
    CloseHandle(x.th);
    bm_unpin();
}

/* Event ping-pong while (a) the case thread runs it inside t_vregs_call with YMM (or XMM) state live,
 * and (b) an idle-priority thread on the SAME processor holds a full register image in a spin loop.
 * Contract: (a) Win x64 ABI keeps XMM6-15 low halves across the blocking calls; (b) every register of
 * the spinner survives all the context switches on its processor. */
typedef struct ppv_ctx { int kind; t_vregs *in, *out; pp_ctx *pp; } ppv_ctx;
static void ppv_inner(void *p) { pp_bench(p); }
static void ppv_bench(void *p) { ppv_ctx *v = (ppv_ctx *)p; t_vregs_call(v->kind, v->in, v->out, ppv_inner, v->pp); }
TEST(sched_pingpong_simd_live) {
    int cpu = bm_pin();
    if (cpu < 0) SKIP("cannot pin the case thread");
    int ymm = cpu_has(F_AVX), kind = ymm ? T_VK_YMM : T_VK_XMM;
    ctx_worker *spin = (ctx_worker *)bm_alloc(sizeof *spin);
    REQUIRE(spin);
    if (!ctx_worker_start(spin, ymm, 0x5B, cpu, 1)) { ctx_worker_stop(spin); bm_free(spin); bm_unpin(); SKIP("could not start the spinning thread"); }
    static pp_ctx x;
    if (!pp_start(&x, cpu, 16)) { pp_stop(&x); ctx_worker_stop(spin); bm_free(spin); bm_unpin(); SKIP("cannot create events/thread"); }
    static t_vregs in, out;
    t_vregs_pattern(&in, kind, 0x77);
    ppv_ctx v = { kind, &in, &out, &x };
    memset(&out, 0, sizeof out);
    ppv_bench(&v);
#ifdef T_X64
    t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "xmm6-15 low across blocking event waits");
#endif
    bm_run(ymm ? "event_pingpong_ymm_live_same_cpu" : "event_pingpong_xmm_live_same_cpu", ppv_bench, &v, 16, 1e3, "k_roundtrips/s", BM_RATE);
#ifdef T_X64
    t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "xmm6-15 low after timed round trips");
#endif
    if (ymm) t_vzeroupper();
    pp_stop(&x);
    CHECKF(x.fails == 0, "%d event waits timed out", x.fails);
    CHECKF(x.served == x.sent, "worker served %ld of %ld round trips", x.served, x.sent);
    CHECK(ctx_worker_stop(spin));
    ctx_worker_regs_ok(spin, "same-processor spinner registers after the ping-pong");
    bm_free(spin);
    bm_unpin();
}

/* Thread state stress (GUIDANCE 5 "thread state stress iterations/s"): N threads pinned to ONE processor,
 * each holding its own XMM/YMM image and re-verifying it against memory in a tight loop. They never block
 * or yield, so every switch between them is an involuntary (timer/quantum) preemption; the OS must save
 * and restore each thread's full vector state each time. PERF = verified iterations/s over all threads;
 * any mismatch is a FAIL. The number of threads that made progress is INFO (no fairness contract). */
static DWORD WINAPI bv_proc(LPVOID p) {
    bv_spin *s = (bv_spin *)p;
    if (cpu_has(F_AVX2)) bv_spin_ymm(s); else bv_spin_xmm(s);
    return 0;
}
TEST(sched_preempt_simd_verify) {
    int cpu = bm_pin();
    if (cpu < 0) SKIP("cannot pin the case thread");
    enum { NT = 4 };
    int ymm = cpu_has(F_AVX2), width = ymm ? 32 : 16;
    bv_spin *s = (bv_spin *)bm_alloc(sizeof(bv_spin) * NT);
    REQUIRE(s);
    /* detector self-test: change the reference in memory under a running spinner; it must notice and
     * leave its loop by itself (bad = 1), proving the in-loop comparison is live */
    {
        for (int i = 0; i < 16; i++) t_pattern_reg(s[0].vin[i], 32, 0xBF, (uint32_t)i);
        HANDLE h = CreateThread(NULL, 64 * 1024, bv_proc, &s[0], CREATE_SUSPENDED, NULL);
        if (!h) { bm_free(s); bm_unpin(); SKIP("CreateThread failed: %lu", GetLastError()); }
        SetThreadAffinityMask(h, (DWORD_PTR)1 << cpu);
        ResumeThread(h);
        double t0 = t_now_ms();
        while (!s[0].ready && t_now_ms() - t0 < 5000) Sleep(1);
        s[0].vin[BV_NPAT - 1][ymm ? 31 : 15] ^= 0x01;           /* the last compared byte of the last register */
        DWORD w = WaitForSingleObject(h, 5000);
        if (w != WAIT_OBJECT_0) { InterlockedExchange(&s[0].stop, 1); WaitForSingleObject(h, 5000); }
        CloseHandle(h);
        CHECKF(w == WAIT_OBJECT_0 && s[0].bad == 1, "self-test: spinner did not detect a changed reference (wait %lu, bad %ld)", w, s[0].bad);
        memset(s, 0, sizeof(bv_spin) * NT);
    }
    HANDLE th[NT] = { 0 };
    int n = 0;
    for (int t = 0; t < NT; t++) {
        for (int i = 0; i < 16; i++) t_pattern_reg(s[t].vin[i], 32, 0xB0 + (uint32_t)t, (uint32_t)i);
        th[t] = CreateThread(NULL, 64 * 1024, bv_proc, &s[t], CREATE_SUSPENDED, NULL);
        if (!th[t]) break;
        SetThreadAffinityMask(th[t], (DWORD_PTR)1 << cpu);
        n++;
    }
    if (n < 2) {
        for (int t = 0; t < n; t++) { TerminateThread(th[t], 1); CloseHandle(th[t]); }
        bm_free(s); bm_unpin(); SKIP("could not create 2 spinner threads: %lu", GetLastError());
    }
    for (int t = 0; t < n; t++) ResumeThread(th[t]);
    double t0 = t_now_ms();
    int ready = 0;
    while (t_now_ms() - t0 < 5000) { ready = 0; for (int t = 0; t < n; t++) ready += s[t].ready != 0; if (ready == n) break; Sleep(5); }
    double run_ms = bm_full() ? 2000 : 300;
    uint64_t q0 = t_qpc();
    LONG it0[NT]; for (int t = 0; t < n; t++) it0[t] = s[t].iters;
    Sleep((DWORD)run_ms);
    LONG it1[NT]; for (int t = 0; t < n; t++) it1[t] = s[t].iters;
    uint64_t q1 = t_qpc();
    for (int t = 0; t < n; t++) InterlockedExchange(&s[t].stop, 1);
    DWORD w = WaitForMultipleObjects((DWORD)n, th, TRUE, 5000);
    CHECKF(w < WAIT_OBJECT_0 + (DWORD)n, "spinner threads did not stop within 5 s (wait %lu)", w);
    for (int t = 0; t < n; t++) { if (w >= WAIT_OBJECT_0 + (DWORD)n) TerminateThread(th[t], 1); CloseHandle(th[t]); }
    CHECKF(ready == n, "only %d of %d spinners started within 5 s", ready, n);
    double total = 0; int progressed = 0, bad = 0;
    for (int t = 0; t < n; t++) {
        total += (double)(uint32_t)(it1[t] - it0[t]);
        progressed += it1[t] != it0[t];
        int diff = -1;
        for (int i = 0; i < BV_NPAT && diff < 0; i++) if (memcmp(s[t].vout[i], s[t].vin[i], (size_t)width)) diff = i;
        if (diff >= 0) { char l[64]; snprintf(l, sizeof l, "spinner %d %s%d at exit", t, ymm ? "ymm" : "xmm", diff); CHECK_REG(l, s[t].vout[diff], s[t].vin[diff], (size_t)width); }
        if (s[t].bad || diff >= 0) bad++;
    }
    CHECKF(bad == 0, "%d of %d same-processor spinners saw their %s registers change under preemption", bad, n, ymm ? "YMM" : "XMM");
    double sec = t_qpc_sec(q0, q1);
    if (sec > 0) t_perf(ymm ? "preempt_ymm_verify_iters" : "preempt_xmm_verify_iters", total / sec / 1e6, "Miter/s");
    t_info("threads", "%d on cpu %d, %d made progress in %.0f ms", n, cpu, progressed, run_ms);
    t_info("registers_per_thread", "%d x %d bytes", BV_NPAT, width);
    bm_free(s);
    bm_unpin();
}

typedef struct tc_ctx { int fails; unsigned mxcsr; unsigned short cw; } tc_ctx;
static DWORD WINAPI tc_proc(LPVOID p) { tc_ctx *x = (tc_ctx *)p; x->mxcsr = t_get_mxcsr(); x->cw = t_get_x87_cw(); return 0x1234; }
static void tc_bench(void *p) {
    tc_ctx *x = (tc_ctx *)p;
    HANDLE h = CreateThread(NULL, 64 * 1024, tc_proc, x, 0, NULL);
    if (!h) { x->fails++; return; }
    DWORD ec = 0;
    if (WaitForSingleObject(h, 5000) != WAIT_OBJECT_0 || !GetExitCodeThread(h, &ec) || ec != 0x1234) x->fails++;
    CloseHandle(h);
}
/* New threads start with the default FP environment, not the creator's: ntoskrnl 19041
 * KiInitializeContextThread stores MXCSR 0x1F80 and FCW 0x27F into the initial frame (corpus explorer). */
TEST(sched_thread_create_rate) {
    bm_pin();
    static tc_ctx x; memset(&x, 0, sizeof x);
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    t_set_mxcsr(0xFFC0 & ~0x40u); t_set_x87_cw(0x0F7F);      /* creator: FZ, RC=toward zero, PC=64 bits */
    tc_bench(&x);
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    if (x.fails) { bm_unpin(); SKIP("CreateThread failed: %lu", GetLastError()); }
    CHECKF(x.mxcsr == 0x1F80, "new thread MXCSR 0x%04x want 0x1F80 (default, independent of the creator's 0xFF80)", x.mxcsr);
#ifdef T_X64
    CHECKF(x.cw == 0x027F, "new thread x87 FCW 0x%04x want 0x027F (Win64 default)", x.cw);
#else
    if (x.cw != 0x027F) t_note("new x86 thread x87 FCW 0x%04x (Windows default 0x027F)", x.cw);
#endif
    bm_run("createthread_wait_close", tc_bench, &x, 1, 1e3, "kthreads/s", BM_RATE);
    CHECKF(x.fails == 0, "%d thread create/wait/exit-code failures", x.fails);
    t_info("new_thread_mxcsr", "0x%04x", x.mxcsr);
    t_info("new_thread_x87_cw", "0x%04x", x.cw);
    bm_unpin();
}
