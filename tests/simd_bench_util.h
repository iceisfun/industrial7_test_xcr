/* simd_bench_util.h - measurement methodology shared by every simd_bench case (included by simd_bench.c
 * only; the Makefile builds tests/<name>.c, so helpers live in tests/simd_bench_*.h, never in common/).
 *
 * Methodology (docs/GUIDANCE.md section 5):
 *   - QueryPerformanceCounter is the only time base. RDTSC is reported as an INFO estimate only.
 *   - every row: warm-up calls, then >= 9 timed samples (21 in full mode); each sample repeats the kernel
 *     until min_sample_ms elapsed (>= 1 call). The PERF row carries median (value), min, max, mean,
 *     stddev; an INFO "<metric>.sd_pct" gives stddev as % of the mean.
 *   - every case pins its thread to ONE processor chosen once per run (the processor the first case ran
 *     on; on a hybrid CPU the highest-numbered performance core, CPUID.1Ah core type 0x40), so rows of
 *     different cases are comparable; the processor, its core type and the affinity are recorded.
 *   - kernels take their inputs from memory and write their outputs to memory; constants that would
 *     let the compiler fold the work are laundered through empty asm (BM_HIDE*), "scalar" kernels are
 *     compiled without auto-vectorisation (BM_NOVEC).
 *   - every kernel's output is verified before its numbers are reported; a wrong result is a FAIL,
 *     slowness never is.
 * Modes: quick (default, roughly 10 s total natively) and full (longer samples, larger memory-bound
 * buffers). The common CLI has no --quick/--full switch (unknown arguments are a harness error), so full
 * mode is selected with the environment variable SIMD_BENCH_MODE=full or by running the manual case
 * bench_mode_full first:  simd_bench.exe --include-manual   (it is the first case in source order). */
#ifndef SIMD_BENCH_UTIL_H
#define SIMD_BENCH_UTIL_H
#include "t.h"
#include <math.h>
#include <immintrin.h>

/* "scalar" reference kernels: no loop or SLP vectorisation, so they really are scalar */
#define BM_NOVEC __attribute__((noinline, optimize("no-tree-vectorize", "no-tree-slp-vectorize")))
/* opaque values: the compiler can no longer prove anything about x */
#define BM_HIDE(x)   __asm__ volatile("" : "+r"(x))
#define BM_HIDEX(x)  __asm__ volatile("" : "+x"(x))
#define BM_HIDEV(x)  __asm__ volatile("" : "+v"(x))
#define BM_BARRIER() __asm__ volatile("" ::: "memory")

/* ------------------------------------------------------------------------------------------- mode */
static int g_bm_full = -1;
static int bm_full(void) {
    if (g_bm_full < 0) {
        char b[32]; DWORD n = GetEnvironmentVariableA("SIMD_BENCH_MODE", b, sizeof b);
        g_bm_full = (n > 0 && n < sizeof b && !_stricmp(b, "full")) ? 1 : 0;
    }
    return g_bm_full;
}
static int    bm_warmup(void) { return bm_full() ? 5 : 2; }
static int    bm_reps(void)   { return bm_full() ? 21 : 9; }
static double bm_min_ms(void) { return bm_full() ? 25.0 : 1.5; }

/* ------------------------------------------------------------------------------------------- pinning */
static uintptr_t g_bm_prev_mask;
static int       g_bm_cpu = -1;
/* Hybrid parts (CPUID.7.0:EDX[15]): CPUID.1Ah:EAX[31:24] = 0x20 efficiency (Atom) core, 0x40 performance
 * (Core) core. Returns 0 if not reported. */
static unsigned bm_core_type_here(void) {
    if (!cpu_hw(F_HYBRID) || cpu()->max_basic < 0x1A) return 0;
    t_cpuid_regs r; cpuid_ex(0x1A, 0, &r);
    return r.eax >> 24;
}
/* One processor for the whole run, so every row of every case is measured on the same core:
 * the processor the first case ran on, or on a hybrid part the highest-numbered performance core. */
static int g_bm_chosen = -2;
static unsigned g_bm_core_type;
static int bm_choose_cpu(void) {
    if (g_bm_chosen != -2) return g_bm_chosen;
    DWORD_PTR pm = 0, sm = 0;
    int pick = -1;
    if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) && pm) {
        PROCESSOR_NUMBER pn; GetCurrentProcessorNumberEx(&pn);
        if (pn.Number < sizeof(DWORD_PTR) * 8 && (pm & ((DWORD_PTR)1 << pn.Number))) pick = pn.Number;
        else for (int i = 0; i < (int)(sizeof(DWORD_PTR) * 8); i++) if (pm & ((DWORD_PTR)1 << i)) { pick = i; break; }
        if (pick >= 0 && cpu_hw(F_HYBRID) && cpu()->max_basic >= 0x1A) {
            for (int i = (int)(sizeof(DWORD_PTR) * 8) - 1; i >= 0; i--) {
                if (!(pm & ((DWORD_PTR)1 << i))) continue;
                DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << i);
                if (!prev) continue;
                unsigned ct = bm_core_type_here();
                SetThreadAffinityMask(GetCurrentThread(), prev);
                if (ct == 0x40) { pick = i; break; }
            }
        }
    }
    g_bm_chosen = pick;
    return pick;
}
static int bm_pin(void) {
    int cpu = bm_choose_cpu();
    if (cpu < 0) { t_note("pinning: no processor in the process affinity mask; measuring unpinned"); return -1; }
    g_bm_prev_mask = t_pin_thread(cpu);
    if (!g_bm_prev_mask) { t_note("pinning: SetThreadAffinityMask(cpu %d) failed (%lu); measuring unpinned", cpu, GetLastError()); return -1; }
    g_bm_cpu = cpu;
    g_bm_core_type = bm_core_type_here();
    t_info("pinned_cpu", "%d%s", cpu, g_bm_core_type == 0x40 ? " (performance core)" : g_bm_core_type == 0x20 ? " (efficiency core)" : "");
    return cpu;
}
static void bm_unpin(void) { if (g_bm_prev_mask) t_unpin_thread(g_bm_prev_mask); g_bm_prev_mask = 0; g_bm_cpu = -1; }

/* ------------------------------------------------------------------------------------------- runner */
enum { BM_RATE = 0, BM_TIME = 1 };
/* Runs fn(ctx) repeatedly and records a PERF row.
 *   BM_RATE: value = work * calls / seconds / scale   (e.g. bytes, scale 1e9 -> GB/s)
 *   BM_TIME: value = seconds / (calls * work) * scale  (e.g. ops, scale 1e9 -> ns/op)
 * Returns the median. */
static double bm_run(const char *metric, t_bench_fn fn, void *ctx, double work, double scale, const char *unit, int mode) {
    double v[64];
    int reps = bm_reps();
    if (reps > 64) reps = 64;
    for (int i = 0; i < bm_warmup(); i++) fn(ctx);
    uint64_t f = t_qpc_freq(), need = (uint64_t)(bm_min_ms() * (double)f / 1000.0);
    for (int r = 0; r < reps; r++) {
        uint64_t calls = 0, t0 = t_qpc(), t1;
        do { fn(ctx); calls++; t1 = t_qpc(); } while (t1 - t0 < need);
        double sec = t_qpc_sec(t0, t1);
        if (sec <= 0) sec = 1.0 / (double)f;          /* QPC did not advance in one call: one tick */
        v[r] = mode == BM_RATE ? work * (double)calls / sec / scale : sec / ((double)calls * work) * scale;
    }
    t_stats s; t_stats_compute(v, reps, &s);
    t_perf_stats(metric, &s, unit);
    char key[160]; snprintf(key, sizeof key, "%s.sd_pct", metric);
    t_info(key, "%.2f", s.mean > 0 ? 100.0 * s.stddev / s.mean : 0.0);
    return s.median;
}

/* ------------------------------------------------------------------------------------------- memory */
/* page-aligned zeroed memory; NULL on failure (the caller SKIPs: resource, not capability) */
static void *bm_alloc(size_t n) { return VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }
static void  bm_free(void *p)   { if (p) VirtualFree(p, 0, MEM_RELEASE); }

/* Large buffers for memory-bound rows, allocated once and reused (src and dst, bm_big_size() each). */
static uint8_t *g_bm_big[2];
static size_t   g_bm_big_n;
static size_t bm_big_size(void) { return bm_full() ? ((size_t)64 << 20) : ((size_t)32 << 20); }
static int bm_big(uint8_t **src, uint8_t **dst) {
    if (!g_bm_big[0]) {
        size_t n = bm_big_size();
        g_bm_big[0] = (uint8_t *)bm_alloc(n + 4096);
        g_bm_big[1] = (uint8_t *)bm_alloc(n + 4096);
        if (!g_bm_big[0] || !g_bm_big[1]) { bm_free(g_bm_big[0]); bm_free(g_bm_big[1]); g_bm_big[0] = g_bm_big[1] = NULL; return 0; }
        t_rng r; t_rng_seed(&r, 0xB16B00B5u);
        t_rng_fill(&r, g_bm_big[0], n);
        memset(g_bm_big[1], 0x5A, n);                /* commit every page before timing */
        g_bm_big_n = n;
    }
    *src = g_bm_big[0]; *dst = g_bm_big[1];
    return 1;
}

static const char *bm_size_name(size_t bytes, char *buf, size_t cap) {
    if (bytes >= ((size_t)1 << 20) && !(bytes & (((size_t)1 << 20) - 1))) snprintf(buf, cap, "%uMB", (unsigned)(bytes >> 20));
    else if (bytes >= 1024 && !(bytes & 1023)) snprintf(buf, cap, "%uKB", (unsigned)(bytes >> 10));
    else snprintf(buf, cap, "%uB", (unsigned)bytes);
    return buf;
}

/* ------------------------------------------------------------------------------------------- misc */
static uint32_t bm_pow32(uint32_t b, uint64_t e) { uint32_t r = 1; while (e) { if (e & 1) r *= b; b *= b; e >>= 1; } return r; }
static uint64_t bm_pow64(uint64_t b, uint64_t e) { uint64_t r = 1; while (e) { if (e & 1) r *= b; b *= b; e >>= 1; } return r; }
static uint16_t bm_pow16(uint16_t b, uint64_t e) { uint16_t r = 1; while (e) { if (e & 1) r = (uint16_t)(r * b); b = (uint16_t)(b * b); e >>= 1; } return r; }

#define BM_NEED(f) do { if (!cpu_has(F_##f)) SKIP(#f " not usable (CPUID bit %d, gate %s)", cpu_hw(F_##f), cpu_feature_gate_name(F_##f)); } while (0)
#endif
