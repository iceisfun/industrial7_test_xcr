/* simd_bench_ctx.h - thread-context helpers shared by simd_bench (tests/) and context_bench (benchmarks/):
 * a worker thread that holds a known XMM/YMM image in a spin loop, the documented XSTATE CONTEXT
 * negotiation (InitializeContext(NULL) -> ERROR_INSUFFICIENT_BUFFER + length, then SetXStateFeaturesMask),
 * and the Suspend/Get[/Set]ThreadContext/Resume measurement that verifies the captured registers.
 * Contracts: see the header comment of tests/simd_bench.c. Include after simd_bench_util.h / _asm.h. */
#ifndef SIMD_BENCH_CTX_H
#define SIMD_BENCH_CTX_H
#include "simd_bench_util.h"
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

static int bm_other_cpu(int not_this) {
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return -1;
    for (int i = (int)(sizeof(DWORD_PTR) * 8) - 1; i >= 0; i--) if ((pm >> i) & 1 && i != not_this) return i;
    return -1;
}

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
#endif
