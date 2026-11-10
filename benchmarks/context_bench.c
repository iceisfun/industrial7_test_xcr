/* context_bench - cost of the OS operations that capture, restore or switch processor state
 * (docs/GUIDANCE.md section 5 "context capture normal/XSTATE ops/s", section 3 fibers/threads):
 *   - RtlCaptureContext, RtlCaptureContext2 (x64, CONTEXT_XSTATE)
 *   - Suspend + GetThreadContext (+ SetThreadContext) + Resume on a worker that holds a known XMM/YMM
 *     image: CONTEXT_CONTROL, full FP (CONTROL|INTEGER|FLOATING_POINT[|EXTENDED_REGISTERS]), and XSTATE
 *     (AVX, AVX + AVX-512) through the documented InitializeContext / SetXStateFeaturesMask negotiation
 *   - exception round trips: RaiseException continued by a vectored handler (in process), and a hardware
 *     #UD (UD2) continued by a vectored handler that advances the IP (in a CHILD process: deliberate fault)
 *   - SwitchToFiber round trips with and without FIBER_FLAG_FLOAT_SWITCH
 *   - thread create + wait + close, one at a time and in batches of 64
 * Every number is accepted only after the state it moved was verified; slowness never fails.
 *
 * Contracts checked (hard CHECKs), and where they come from:
 *   - captured contexts hold the worker's registers, and the worker's registers are unchanged after the
 *     timed Get/SetThreadContext loop (MSDN GetThreadContext/SetThreadContext; the helpers and the
 *     InitializeContext(NULL) -> ERROR_INSUFFICIENT_BUFFER + length negotiation are in
 *     tests/simd_bench_ctx.h, decomp of kernelbase!InitializeContext2 19041 cited there)
 *   - x64 RtlCaptureContext: ContextFlags = 0x10000F (ntdll 19041 asm, see tests/simd_bench.c)
 *   - RaiseException: "If the exception is continuable and a handler returns EXCEPTION_CONTINUE_EXECUTION,
 *     RaiseException returns" (MSDN); each raise reaches the vectored handler exactly once. Win x64 ABI:
 *     RaiseException is a call, so XMM6-15 survive it.
 *   - hardware exception continued with the unmodified record: the interrupted MXCSR is restored
 *     (the context record is the full interrupted state; NtContinue restores it). x64: XMM6-15 too.
 *   - fibers (see tests/fiber_xstate.c for the decomp): with FIBER_FLAG_FLOAT_SWITCH on both fibers the
 *     MXCSR control bits are per fiber (CreateFiberEx doc: "the floating-point state is switched");
 *     without the flag, x64 still switches them (SwitchToFiber is a call; MXCSR control is nonvolatile in
 *     the x64 ABI) and x86 does not have to (INFO). x64: XMM6-15 survive SwitchToFiber.
 *   - CreateThread / WaitForSingleObject / GetExitCodeThread: the exit code is the thread's return value.
 * Modes and methodology: tests/simd_bench_util.h (pinning, warm-up, median of >= 9 samples, quartiles). */
#include "simd_bench_util.h"
#include "simd_bench_asm.h"
#include "simd_bench_ctx.h"

#ifndef FIBER_FLAG_FLOAT_SWITCH
#define FIBER_FLAG_FLOAT_SWITCH 0x1
#endif

TEST_MANUAL(bench_mode_full) {
    g_bm_full = 1;
    t_note("full mode selected: %d samples of >= %.0f ms", bm_reps(), bm_min_ms());
    CHECK(bm_full());
}

TEST(ctx_env_report) {
    __typeof__(&GetEnabledXStateFeatures) pEn = T_PROC("kernel32.dll", GetEnabledXStateFeatures);
    __typeof__(&InitializeContext) pInit = T_PROC("kernel32.dll", InitializeContext);
    t_info("enabled_xstate_features", pEn ? "0x%llx" : "(GetEnabledXStateFeatures not exported)", pEn ? (unsigned long long)pEn() : 0ull);
    if (pInit) {
        DWORD lp = 0, lx = 0;
        pInit(NULL, CX_FP_FLAGS, NULL, &lp);
        pInit(NULL, CX_FP_FLAGS | CONTEXT_XSTATE, NULL, &lx);
        t_info("context_length_fp", "%lu", lp);
        t_info("context_length_xstate", "%lu", lx);
    }
    t_info("sizeof_CONTEXT", "%u", (unsigned)sizeof(CONTEXT));
    t_info("cpu", "%s", cpu()->brand);
    t_info("ncpu", "%d", t_cpu_count());
    t_info("bench_cpu", "%d", bm_choose_cpu());
    t_info("mode", "%s", bm_full() ? "full" : "quick");
    if (t_is_wine()) t_note("running under Wine %s: numbers describe the host + Wine, not Windows", t_wine_version());
}

/* ============================================================================ capture */
typedef struct cap_ctx { CONTEXT *c; } cap_ctx;
static void cap_bench(void *p) { RtlCaptureContext(((cap_ctx *)p)->c); }
TEST(ctx_rtlcapturecontext) {
    static T_ALIGN(64) CONTEXT c;
    memset(&c, 0, sizeof c);
    RtlCaptureContext(&c);
#ifdef T_X64
    CHECKF(c.ContextFlags == 0x10000F, "ContextFlags 0x%lx want 0x10000F (CONTROL|INTEGER|SEGMENTS|FLOATING_POINT)", c.ContextFlags);
    CHECKF(c.MxCsr == t_get_mxcsr(), "captured MxCsr 0x%lx, live 0x%x", c.MxCsr, t_get_mxcsr());
    CHECK(c.Rip != 0 && c.Rsp != 0);
#else
    CHECKF((c.ContextFlags & (CONTEXT_i386 | 3)) == (CONTEXT_i386 | 3), "ContextFlags 0x%lx lacks i386|CONTROL|INTEGER", c.ContextFlags);
    CHECK(c.Eip != 0 && c.Esp != 0);
#endif
    bm_pin();
    cap_ctx x = { &c };
    bm_run("rtlcapturecontext", cap_bench, &x, 1, 1e6, "Mops/s", BM_RATE);
    bm_unpin();
}

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
    if (!T_PROC("kernel32.dll", InitializeContext) || !T_PROC("kernel32.dll", SetXStateFeaturesMask)) SKIP("XSTATE context APIs not exported");
    DWORD flags = CX_FP_FLAGS | CONTEXT_XSTATE;
    xctx xc;
    if (!xctx_init(&xc, flags, XSTATE_MASK_AVX)) { free(xc.buf); return; }
    cap2_ctx x = { f, xc.c, flags };
    cap2_bench(&x);
    CHECKF((xc.c->ContextFlags & 0x10000F) == 0x10000F, "ContextFlags 0x%lx lacks CONTROL|INTEGER|SEGMENTS|FLOATING_POINT", xc.c->ContextFlags);
    CHECKF(xc.c->MxCsr == t_get_mxcsr(), "MxCsr 0x%lx, live 0x%x", xc.c->MxCsr, t_get_mxcsr());
    t_info("context_flags_after", "0x%lx", xc.c->ContextFlags);
    bm_pin();
    bm_run("rtlcapturecontext2_xstate", cap2_bench, &x, 1, 1e6, "Mops/s", BM_RATE);
    bm_unpin();
    free(xc.buf);
#endif
}

/* ============================================================================ Get/SetThreadContext */
TEST(ctx_get_control) {
    int cpu = bm_pin();
    ctx_worker *w = (ctx_worker *)bm_alloc(sizeof *w);
    REQUIRE(w);
    if (!ctx_worker_start(w, 0, 0xB1, bm_other_cpu(cpu), 0)) { ctx_worker_stop(w); bm_free(w); bm_unpin(); SKIP("could not start a worker thread"); }
    static T_ALIGN(64) CONTEXT c;
    gc_ctx x = { w->th, &c, CONTEXT_CONTROL, 0, 0, 0, NULL };
    gc_bench(&x);
#ifdef T_X64
    uintptr_t ip = (uintptr_t)c.Rip;
#else
    uintptr_t ip = (uintptr_t)c.Eip;
#endif
    CHECKF(x.fails == 0, "Suspend/GetThreadContext(CONTEXT_CONTROL) failed: %lu", GetLastError());
    CHECKF(ip >= (uintptr_t)bx_spin_xmm && ip < (uintptr_t)bx_spin_xmm + 4096, "worker IP %p is not inside its spin routine (%p)", (void *)ip, (void *)bx_spin_xmm);
    if (!x.fails) bm_run("getthreadcontext_control", gc_bench, &x, 1, 1e6, "us/op", BM_TIME);
    CHECKF(x.fails == 0, "Suspend/GetThreadContext(CONTEXT_CONTROL) failed %d times", x.fails);
    CHECK(ctx_worker_stop(w));
    ctx_worker_regs_ok(w, "worker registers after CONTEXT_CONTROL captures");
    bm_free(w);
    bm_unpin();
}
TEST(ctx_get_full_fp)           { get_ctx_case("getthreadcontext_full_fp", 0, 0, 0); }
TEST(ctx_get_xstate_avx)        { get_ctx_case("getthreadcontext_xstate_avx", 1, XSTATE_MASK_AVX, 0); }
TEST(ctx_get_xstate_avx512)     { get_ctx_case("getthreadcontext_xstate_avx512", 1, XSTATE_MASK_AVX | XSTATE_MASK_AVX512, 0); }
TEST(ctx_getset_full_fp)        { get_ctx_case("get_set_threadcontext_full_fp", 0, 0, 1); }
TEST(ctx_getset_xstate_avx)     { get_ctx_case("get_set_threadcontext_xstate_avx", 1, XSTATE_MASK_AVX, 1); }
TEST(ctx_getset_xstate_avx512)  { get_ctx_case("get_set_threadcontext_xstate_avx512", 1, XSTATE_MASK_AVX | XSTATE_MASK_AVX512, 1); }

/* ============================================================================ exceptions */
#define CB_SW_CODE 0xE0C0FFEEu
static volatile LONG g_cb_sw_hits;
static LONG CALLBACK cb_sw_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != CB_SW_CODE) return EXCEPTION_CONTINUE_SEARCH;
    g_cb_sw_hits++;
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void cb_raise_once(void *p) { (void)p; RaiseException(CB_SW_CODE, 0, 0, NULL); }
static void cb_raise_bench(void *p) { (void)p; for (int i = 0; i < 8; i++) RaiseException(CB_SW_CODE, 0, 0, NULL); }
TEST(ctx_exception_raise_veh_roundtrip) {
    void *h = AddVectoredExceptionHandler(1, cb_sw_veh);
    REQUIREF(h != NULL, "AddVectoredExceptionHandler failed: %lu", GetLastError());
    g_cb_sw_hits = 0;
    cb_raise_once(NULL);
    CHECKF(g_cb_sw_hits == 1, "vectored handler ran %ld times for one RaiseException (want 1)", g_cb_sw_hits);
#ifdef T_X64
    {   /* RaiseException is an ordinary call: XMM6-15 (nonvolatile) survive the round trip */
        static t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, 0xE1);
        CHECK(t_vregs_call(T_VK_XMM, &in, &out, cb_raise_once, NULL) == 0);
        t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "XMM6-15 after a continued RaiseException");
    }
#endif
    bm_pin();
    g_cb_sw_hits = 0;
    bm_run("raiseexception_veh_continue", cb_raise_bench, NULL, 8, 1e3, "kexceptions/s", BM_RATE);
    LONG hits = g_cb_sw_hits;
    bm_unpin();
    RemoveVectoredExceptionHandler(h);
    CHECKF(hits > 0 && hits % 8 == 0, "handler hits %ld not a positive multiple of the 8 raises per call", hits);
}

/* Hardware #UD (UD2, 0F 0B) continued by a VEH that advances the IP by 2. Deliberate fault -> child. */
static volatile LONG g_cb_hw_hits, g_cb_hw_bad_mx;
static unsigned g_cb_hw_mx, g_cb_hw_seen_mx, g_cb_hw_seen_flt;
static LONG CALLBACK cb_hw_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != STATUS_ILLEGAL_INSTRUCTION) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord;
#ifdef T_X64
    const uint8_t *ip = (const uint8_t *)c->Rip;
    if ((c->MxCsr & 0xFFC0) != (g_cb_hw_mx & 0xFFC0)) {   /* control bits; the timing loop sets PE */
        g_cb_hw_bad_mx++; g_cb_hw_seen_mx = c->MxCsr; g_cb_hw_seen_flt = c->FltSave.MxCsr; }
#else
    const uint8_t *ip = (const uint8_t *)c->Eip;
#endif
    if (ip[0] != 0x0F || ip[1] != 0x0B) return EXCEPTION_CONTINUE_SEARCH;
#ifdef T_X64
    c->Rip += 2;
#else
    c->Eip += 2;
#endif
    g_cb_hw_hits++;
    return EXCEPTION_CONTINUE_EXECUTION;
}
static T_NOINLINE void cb_ud2(void *p) { (void)p; __asm__ volatile("ud2" ::: "memory"); }
static void cb_ud2_bench(void *p) { (void)p; for (int i = 0; i < 8; i++) cb_ud2(NULL); }
CHILD(hw_ud2) {
    void *h = AddVectoredExceptionHandler(1, cb_hw_veh);
    if (!h) { printf("ERROR=AddVectoredExceptionHandler %lu\n", GetLastError()); return 3; }
    unsigned mx0 = t_get_mxcsr();
    g_cb_hw_mx = 0x3F80;                  /* RC = round down, all masked: a non-default value to carry */
    t_set_mxcsr(g_cb_hw_mx);
    cb_ud2(NULL);
    unsigned mx1 = t_get_mxcsr();
    printf("HITS1=%ld\nMXCSR_AFTER=0x%x\n", g_cb_hw_hits, mx1);
#ifdef T_X64
    static t_vregs in, out;
    t_vregs_pattern(&in, T_VK_XMM, 0xE2);
    t_vregs_call(T_VK_XMM, &in, &out, cb_ud2, NULL);
    int bad = 0;
    for (int i = 6; i < 16; i++) bad += memcmp(out.v[i], in.v[i], 16) != 0;
    printf("XMM6_15_BAD=%d\n", bad);
#endif
    /* timing: same sampling as bm_run, printed for the parent */
    double v[64]; int reps = bm_reps();
    for (int i = 0; i < bm_warmup(); i++) cb_ud2_bench(NULL);
    uint64_t f = t_qpc_freq(), need = (uint64_t)(bm_min_ms() * (double)f / 1000.0);
    LONG h0 = g_cb_hw_hits; uint64_t calls_total = 0;
    for (int r = 0; r < reps && r < 64; r++) {
        uint64_t calls = 0, t0 = t_qpc(), t1;
        do { cb_ud2_bench(NULL); calls++; t1 = t_qpc(); } while (t1 - t0 < need);
        calls_total += calls;
        double sec = t_qpc_sec(t0, t1);
        v[r] = 8.0 * (double)calls / (sec > 0 ? sec : 1.0 / (double)f) / 1e3;
    }
    printf("TIMED_HITS=%ld\nTIMED_WANT=%llu\nBAD_MXCSR_IN_HANDLER=%ld\nSEEN_MXCSR=0x%x/0x%x\n", g_cb_hw_hits - h0, (unsigned long long)(8 * calls_total), g_cb_hw_bad_mx, g_cb_hw_seen_mx, g_cb_hw_seen_flt);
    printf("SAMPLES=");
    for (int r = 0; r < reps && r < 64; r++) printf("%s%.6g", r ? "," : "", v[r]);
    printf("\n");
    t_set_mxcsr(mx0);
    RemoveVectoredExceptionHandler(h);
    fflush(stdout);
    return 0;
}
TEST(ctx_exception_hw_ud2_veh_roundtrip) {
    static t_child_result r;
    t_child_run("hw_ud2", "", 60000, &r);
    if (!r.started) SKIP("could not start the child process (error %lu)", r.spawn_error);
    REQUIREF(!r.timed_out, "child timed out");
    char val[4096];
    REQUIREF(r.exit_code == 0, "child exit 0x%lx exception 0x%lx: %.300s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    REQUIRE(t_child_find(&r, "HITS1=", val, sizeof val));
    CHECKF(atol(val) == 1, "UD2 reached the vectored handler %s times (want 1)", val);
    REQUIRE(t_child_find(&r, "MXCSR_AFTER=", val, sizeof val));
    CHECKF((strtoul(val, NULL, 16) & 0xFFC0) == 0x3F80, "MXCSR after a continued #UD is %s, want the interrupted 0x3F80", val);
#ifdef T_X64
    if (t_child_find(&r, "XMM6_15_BAD=", val, sizeof val)) CHECKF(atoi(val) == 0, "%s of XMM6-15 changed across a continued #UD", val);
    if (t_child_find(&r, "BAD_MXCSR_IN_HANDLER=", val, sizeof val)) CHECKF(atol(val) == 0, "%s handler invocations saw ContextRecord->MxCsr control bits != the interrupted 0x3F80", val);
#endif
    char want[64];
    REQUIRE(t_child_find(&r, "TIMED_HITS=", val, sizeof val) && t_child_find(&r, "TIMED_WANT=", want, sizeof want));
    CHECKF(strtoull(val, NULL, 10) == strtoull(want, NULL, 10), "timed loop: %s handler hits for %s UD2 executions", val, want);
    REQUIRE(t_child_find(&r, "SAMPLES=", val, sizeof val));
    double v[64]; int n = 0;
    for (char *p = val; *p && n < 64;) { v[n++] = strtod(p, &p); if (*p == ',') p++; else break; }
    REQUIREF(n >= 3, "child printed %d samples", n);
    t_stats s; t_stats_compute(v, n, &s);
    t_perf_stats("ud2_veh_continue", &s, "kexceptions/s");
    t_info("child_ms", "%.0f", r.ms);
}

/* ============================================================================ fibers */
typedef struct fb_ctx {
    void *main, *other;
    DWORD flags;
    unsigned mx_main, mx_other;
    volatile LONG other_bad, main_bad, rounds;
    int created;
} fb_ctx;
static void fb_clobber_nonvolatile(void) {
#ifdef T_X64
    __asm__ volatile("pcmpeqd %%xmm6, %%xmm6\n\t" "pcmpeqd %%xmm7, %%xmm7\n\t" "pxor %%xmm8, %%xmm8\n\t" "pxor %%xmm9, %%xmm9\n\t"
                     "pcmpeqd %%xmm10, %%xmm10\n\t" "pcmpeqd %%xmm11, %%xmm11\n\t" "pxor %%xmm12, %%xmm12\n\t" "pxor %%xmm13, %%xmm13\n\t"
                     "pcmpeqd %%xmm14, %%xmm14\n\t" "pxor %%xmm15, %%xmm15\n\t"
                     ::: "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
#endif
}
static VOID WINAPI fb_other_proc(LPVOID p) {
    fb_ctx *x = (fb_ctx *)p;
    t_set_mxcsr(x->mx_other);
    for (;;) {
        fb_clobber_nonvolatile();
        SwitchToFiber(x->main);
        if ((t_get_mxcsr() & 0xFFC0) != (x->mx_other & 0xFFC0)) x->other_bad++;   /* control bits only */
        t_set_mxcsr(x->mx_other);
        x->rounds++;
    }
}
static void fb_round(void *p) {
    fb_ctx *x = (fb_ctx *)p;
    for (int i = 0; i < 16; i++) {
        SwitchToFiber(x->other);
        if ((t_get_mxcsr() & 0xFFC0) != (x->mx_main & 0xFFC0)) { x->main_bad++; t_set_mxcsr(x->mx_main); }
    }
}
static void fb_round_once(void *p) { SwitchToFiber(((fb_ctx *)p)->other); }
typedef struct fb_thread { fb_ctx x; int ok; DWORD err; int cpu; double median; } fb_thread;
static DWORD WINAPI fb_thread_proc(LPVOID p) {
    fb_thread *t = (fb_thread *)p;
    fb_ctx *x = &t->x;
    if (t->cpu >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << t->cpu);
    x->main = ConvertThreadToFiberEx(NULL, x->flags);
    if (!x->main) { t->err = GetLastError(); return 1; }
    x->other = CreateFiberEx(0, 64 * 1024, x->flags, fb_other_proc, x);
    if (!x->other) { t->err = GetLastError(); ConvertFiberToThread(); return 1; }
    t->ok = 1;
    unsigned mx0 = t_get_mxcsr();
    t_set_mxcsr(x->mx_main);
    fb_round(x);                                  /* first switch starts the other fiber */
#ifdef T_X64
    {
        static t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, 0xF1 + x->flags);
        t_vregs_call(T_VK_XMM, &in, &out, fb_round_once, x);
        t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "XMM6-15 after a SwitchToFiber round trip (other fiber clobbers them)");
        t_set_mxcsr(x->mx_main);
    }
#endif
    x->main_bad = x->other_bad = 0;
    LONG r0 = x->rounds;
    t->median = bm_run(x->flags ? "switchtofiber_roundtrip_float_switch" : "switchtofiber_roundtrip_noflag", fb_round, x, 16, 1e6, "Mroundtrips/s", BM_RATE);
    if (x->rounds == r0) x->other_bad = -1;
    t_set_mxcsr(mx0);
    DeleteFiber(x->other);
    ConvertFiberToThread();
    return 0;
}
static void fb_case(DWORD flags) {
    static fb_thread t;
    memset(&t, 0, sizeof t);
    t.x.flags = flags;
    t.x.mx_main = 0x1F80;                         /* default: round to nearest, all masked */
    t.x.mx_other = 0x3F80 | 0x8000;               /* RC = round down, FZ set, all masked */
    t.cpu = bm_choose_cpu();
    HANDLE h = CreateThread(NULL, 0, fb_thread_proc, &t, 0, NULL);
    if (!h) SKIP("CreateThread failed: %lu", GetLastError());
    DWORD w = WaitForSingleObject(h, 120000);
    CloseHandle(h);
    REQUIREF(w == WAIT_OBJECT_0, "fiber thread did not finish");
    if (!t.ok) SKIP("fiber creation failed (error %lu)", t.err);
    CHECKF(t.x.other_bad != -1, "the other fiber never ran during the timed loop");
    int must = flags || sizeof(void *) == 8;
    if (must && (t.x.main_bad || t.x.other_bad) && t_is_wine()) t_expect_fail("Wine: SwitchToFiber does not switch MXCSR / x87 control word per fiber");
    if (must) {
        CHECKF(t.x.main_bad == 0, "%ld of the timed round trips came back with a foreign MXCSR on the main fiber (want 0x%04x control bits)", t.x.main_bad, t.x.mx_main);
        CHECKF(t.x.other_bad == 0, "%ld resumptions of the other fiber saw a foreign MXCSR (want 0x%04x)", t.x.other_bad, t.x.mx_other);
    } else {
        t_info("mxcsr_shared_without_flag", "main saw foreign MXCSR %ld times, other %ld times (x86 without FIBER_FLAG_FLOAT_SWITCH: not switched per docs)", t.x.main_bad, t.x.other_bad);
    }
}
TEST(ctx_fiber_switch_noflag)       { fb_case(0); }
TEST(ctx_fiber_switch_float_switch) { fb_case(FIBER_FLAG_FLOAT_SWITCH); }

/* ============================================================================ threads */
typedef struct tj_ctx { int fails; } tj_ctx;
static DWORD WINAPI tj_proc(LPVOID p) { return (DWORD)(uintptr_t)p ^ 0x5A5A0000u; }
static void tj_one(void *p) {
    tj_ctx *x = (tj_ctx *)p;
    HANDLE h = CreateThread(NULL, 64 * 1024, tj_proc, (void *)(uintptr_t)0x1234, 0, NULL);
    if (!h) { x->fails++; return; }
    DWORD ec = 0;
    if (WaitForSingleObject(h, 5000) != WAIT_OBJECT_0 || !GetExitCodeThread(h, &ec) || ec != (0x1234u ^ 0x5A5A0000u)) x->fails++;
    CloseHandle(h);
}
#define TJ_BATCH 64
static void tj_batch(void *p) {
    tj_ctx *x = (tj_ctx *)p;
    HANDLE h[TJ_BATCH]; int n = 0;
    for (int i = 0; i < TJ_BATCH; i++) {
        h[n] = CreateThread(NULL, 64 * 1024, tj_proc, (void *)(uintptr_t)i, 0, NULL);
        if (!h[n]) { x->fails++; break; }
        n++;
    }
    if (n && WaitForMultipleObjects((DWORD)n, h, TRUE, 10000) != WAIT_OBJECT_0) x->fails++;
    for (int i = 0; i < n; i++) {
        DWORD ec = 0;
        if (!GetExitCodeThread(h[i], &ec) || ec != ((DWORD)i ^ 0x5A5A0000u)) x->fails++;
        CloseHandle(h[i]);
    }
}
TEST(ctx_thread_create_join) {
    tj_ctx x = { 0 };
    tj_one(&x);
    if (x.fails) SKIP("CreateThread failed: %lu", GetLastError());
    bm_pin();
    bm_run("createthread_wait_close", tj_one, &x, 1, 1e3, "kthreads/s", BM_RATE);
    bm_unpin();
    CHECKF(x.fails == 0, "%d create/wait/exit-code failures", x.fails);
}
TEST(ctx_thread_create_join_batch64) {
    tj_ctx x = { 0 };
    tj_batch(&x);
    if (x.fails) SKIP("could not create and join %d threads", TJ_BATCH);
    bm_run("createthread_batch64_wait_close", tj_batch, &x, TJ_BATCH, 1e3, "kthreads/s", BM_RATE);
    CHECKF(x.fails == 0, "%d create/wait/exit-code failures in batches of %d", x.fails, TJ_BATCH);
}
