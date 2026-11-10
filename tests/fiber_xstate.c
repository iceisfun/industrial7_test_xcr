/* fiber_xstate - fibers and the processor state they carry: ConvertThreadToFiber(Ex), CreateFiber(Ex),
 * SwitchToFiber, DeleteFiber, ConvertFiberToThread, GetCurrentFiber/GetFiberData, IsThreadAFiber,
 * fiber local storage, per-fiber FP control state (MXCSR, x87 CW), nonvolatile XMM/GPR preservation,
 * fiber stacks, fibers migrating between threads, and thread-context visibility of a running fiber.
 *
 * Contract under test (what is a hard CHECK and why):
 *   - Microsoft Docs "Fibers": "The only state information maintained for a fiber is its stack, a subset
 *     of its registers, and the fiber data ... The saved registers are the set of registers typically
 *     preserved across a function call."  => SwitchToFiber behaves like a call for the nonvolatile set:
 *       x64: RBX RBP RSI RDI R12-R15, XMM6-15 (low 128 bits), MXCSR control bits, x87 control word
 *            (Windows x64 ABI: these are nonvolatile; MXCSR status bits and XMM0-5/upper YMM are volatile)
 *       x86: EBX ESI EDI EBP (no XMM register is nonvolatile in the x86 ABI)
 *   - CreateFiberEx / ConvertThreadToFiberEx docs: "If this parameter is zero, the floating-point state on
 *     x86 systems is not switched ... If this parameter is FIBER_FLAG_FLOAT_SWITCH, the floating-point
 *     state is switched for the fiber."  => x86: flag => MXCSR control + x87 CW per fiber; no flag on
 *     both sides => state shared.  On x64 the state is always switched (see decomp below).
 *   - Checked decomp, Windows 10 19041 kernelbase (corpus explorer, /api/function/kernelbase/...):
 *       SwitchToFiber: if (lpFiber == TEB->FiberData) return (no-op); validates a cookie, then
 *         SwitchToFiberContext, which UNCONDITIONALLY saves/restores rbx rbp rsi rdi r12-r15 rsp,
 *         xmm6-xmm15 (movaps), MXCSR (stmxcsr/ldmxcsr, whole register), x87 CW (fnstcw/fldcw, after
 *         fnclex), and swaps TEB StackBase/StackLimit/DeallocationStack/FlsData/GuaranteedStackBytes/
 *         ActivationContextStack.  => on x64 FIBER_FLAG_FLOAT_SWITCH changes nothing observable.
 *       ConvertThreadToFiberEx: dwFlags & ~1 -> ERROR_INVALID_PARAMETER (checked first); already a fiber
 *         -> ERROR_ALREADY_FIBER (1280).  ConvertThreadToFiber: already a fiber -> 1280.
 *       ConvertFiberToThread: not a fiber -> FALSE, ERROR_ALREADY_THREAD (1281).
 *       CreateFiberEx: dwFlags & ~1 -> ERROR_INVALID_PARAMETER; RtlCreateUserStack failure -> NT error.
 *       BaseInitializeFiberContext: new fiber starts with MXCSR 0x1F80, x87 CW 0x27F (recorded as INFO
 *         only: not a documented contract).
 *       DeleteFiber: current fiber -> RtlExitUserThread(1) (doc: "its thread calls ExitThread");
 *         otherwise frees stack and runs FLS callbacks (RtlProcessFlsData) for the deleted fiber.
 *       FlsGetValue/FlsSetValue/FlsFree: index 0 or >= 0xFF0 -> STATUS_INVALID_PARAMETER -> error 87.
 *       IsThreadAFiber: TEB SameTebFlags.HasFiberData.
 *   - SetThreadStackGuarantee doc: the guarantee set from a fiber "is used by the fiber no matter which
 *     thread executes the fiber" (per-fiber value).
 *   - FlsCallback doc: called "on fiber deletion, thread exit, and when an FLS index is freed";
 *     FlsFree doc: callback "called for each fiber, if the corresponding FLS slot contains a non-NULL value".
 *   - Fibers doc: TLS accessed by a fiber is the TLS of the thread running it; ExitThread from a fiber
 *     exits the thread; "If your fiber function returns, the thread running the fiber exits."
 *
 * Not checked (INFO/NOTE only): SwitchToFiber(GetCurrentFiber()) (doc: "unpredictable"), SwitchToFiber(NULL)
 * or from a non-fiber thread (undefined), MXCSR/x87 status flags, volatile XMM, upper YMM/ZMM, the
 * initial FP state of a new fiber, thread exit codes the docs do not specify.
 *
 * Fiber work runs on worker threads so the case thread is never converted. CHECK is thread-safe;
 * no SKIP/REQUIRE is used while running on a fiber. */
#include "t.h"

/* mingw-w64 intrin-impl.h (__readgsqword/__readfsdword behind NtCurrentTeb/GetCurrentFiber) trips -Warray-bounds */
#pragma GCC diagnostic ignored "-Warray-bounds"

#ifndef FIBER_FLAG_FLOAT_SWITCH
#define FIBER_FLAG_FLOAT_SWITCH 0x1
#endif
#ifndef ERROR_ALREADY_FIBER
#define ERROR_ALREADY_FIBER 1280L
#endif
#ifndef ERROR_ALREADY_THREAD
#define ERROR_ALREADY_THREAD 1281L
#endif

#define MXC_CTRL 0xFFC0u     /* DAZ, exception masks, RC, FTZ (nonvolatile in the x64 ABI) */
#define MXC_FLAGS 0x003Fu    /* sticky status flags (volatile) */
#define CW_CTRL  0x0F3Fu     /* x87 masks, PC, RC */

/* ======================================================================== */
/*                               helpers                                    */
/* ======================================================================== */

typedef void (*body_fn)(void *);
typedef struct { body_fn fn; void *arg; } thr_arg;
static DWORD WINAPI thr_tramp(void *p) { thr_arg *a = (thr_arg *)p; a->fn(a->arg); return 0; }

/* Run fn(arg) on a fresh thread. Returns 0 ok, -1 CreateThread failed, -2 timeout (thread terminated). */
static int run_thread_ex(body_fn fn, void *arg, SIZE_T stack, DWORD tmo, DWORD *code) {
    thr_arg a = { fn, arg };
    HANDLE h = CreateThread(NULL, stack, thr_tramp, &a, 0, NULL);
    if (!h) return -1;
    DWORD w = WaitForSingleObject(h, tmo);
    if (w != WAIT_OBJECT_0) {
        CHECKF(0, "worker thread did not finish within %lu ms", (unsigned long)tmo);
        TerminateThread(h, 0xDEADu);
        WaitForSingleObject(h, 2000);
        CloseHandle(h);
        return -2;
    }
    if (code) GetExitCodeThread(h, code);
    CloseHandle(h);
    return 0;
}
#define RUN_WORKER(fn, arg) do { int rc_ = run_thread_ex((fn), (arg), 0, 60000, NULL); \
    if (rc_ == -1) SKIP("CreateThread failed: %lu", (unsigned long)GetLastError()); } while (0)

static void sw_to(void *f) { SwitchToFiber(f); }

static BOOL is_fiber(void) {
    BOOL (WINAPI *p)(void) = T_PROC("kernel32.dll", IsThreadAFiber);
    return p ? p() : -1;
}

/* reserved stack range of the running fiber/thread [lo, hi) */
static void stack_range(uintptr_t *lo, uintptr_t *hi) {
    void (WINAPI *p)(PULONG_PTR, PULONG_PTR) = T_PROC("kernel32.dll", GetCurrentThreadStackLimits);
    if (p) { ULONG_PTR l = 0, h = 0; p(&l, &h); *lo = l; *hi = h; return; }
    NT_TIB *tib = (NT_TIB *)NtCurrentTeb();
    *lo = (uintptr_t)tib->StackLimit; *hi = (uintptr_t)tib->StackBase;
}

static t_vregs *vr_alloc(void) { t_vregs *v = (t_vregs *)_aligned_malloc(sizeof(t_vregs), 64); if (v) memset(v, 0, sizeof *v); return v; }
static void vr_free(t_vregs *v) { if (v) _aligned_free(v); }
/* bytes [off, off+len) of registers first..last equal */
static int vr_eq(const t_vregs *a, const t_vregs *b, int first, int last, int off, int len) {
    for (int i = first; i <= last; i++) if (memcmp(a->v[i] + off, b->v[i] + off, (size_t)len)) return 0;
    return 1;
}

/* nonvolatile GPR round trip through fn(arg): loads in[] into the callee-saved GPRs, calls fn, stores them.
 * x64 order: rbx rbp rsi rdi r12 r13 r14 r15 (8); x86: ebx esi edi ebp (4). */
#ifdef T_X64
#define NGPR 8
static const char *const k_gpr_names[NGPR] = { "rbx", "rbp", "rsi", "rdi", "r12", "r13", "r14", "r15" };
__asm__(
    ".globl fx_gpr_call\n.def fx_gpr_call; .scl 2; .type 32; .endef\n.seh_proc fx_gpr_call\nfx_gpr_call:\n"
    "  pushq %rbp\n  .seh_pushreg %rbp\n  pushq %rbx\n  .seh_pushreg %rbx\n"
    "  pushq %rsi\n  .seh_pushreg %rsi\n  pushq %rdi\n  .seh_pushreg %rdi\n"
    "  pushq %r12\n  .seh_pushreg %r12\n  pushq %r13\n  .seh_pushreg %r13\n"
    "  pushq %r14\n  .seh_pushreg %r14\n  pushq %r15\n  .seh_pushreg %r15\n"
    "  subq $40, %rsp\n  .seh_stackalloc 40\n  .seh_endprologue\n"
    "  movq %rdx, 32(%rsp)\n  movq %rcx, %r10\n  movq %r8, %rax\n  movq %r9, %rcx\n"
    "  movq 0(%r10), %rbx\n  movq 8(%r10), %rbp\n  movq 16(%r10), %rsi\n  movq 24(%r10), %rdi\n"
    "  movq 32(%r10), %r12\n  movq 40(%r10), %r13\n  movq 48(%r10), %r14\n  movq 56(%r10), %r15\n"
    "  callq *%rax\n"
    "  movq 32(%rsp), %rax\n"
    "  movq %rbx, 0(%rax)\n  movq %rbp, 8(%rax)\n  movq %rsi, 16(%rax)\n  movq %rdi, 24(%rax)\n"
    "  movq %r12, 32(%rax)\n  movq %r13, 40(%rax)\n  movq %r14, 48(%rax)\n  movq %r15, 56(%rax)\n"
    "  addq $40, %rsp\n"
    "  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n  popq %rdi\n  popq %rsi\n  popq %rbx\n  popq %rbp\n"
    "  ret\n.seh_endproc\n"
    /* fiber start routine probe: record RSP at entry, then tail-jump to the C body */
    ".globl fx_entry_probe\n.def fx_entry_probe; .scl 2; .type 32; .endef\nfx_entry_probe:\n"
    "  movq %rsp, fx_entry_sp(%rip)\n  jmp fx_entry_c\n");
#else
#define NGPR 4
static const char *const k_gpr_names[NGPR] = { "ebx", "esi", "edi", "ebp" };
__asm__(
    ".globl _fx_gpr_call\n_fx_gpr_call:\n"
    "  pushl %ebp\n  pushl %ebx\n  pushl %esi\n  pushl %edi\n"
    "  movl 20(%esp), %eax\n  movl 28(%esp), %ecx\n  movl 32(%esp), %edx\n"
    "  pushl 24(%esp)\n  pushl %edx\n"
    "  movl 0(%eax), %ebx\n  movl 4(%eax), %esi\n  movl 8(%eax), %edi\n  movl 12(%eax), %ebp\n"
    "  call *%ecx\n"
    "  addl $4, %esp\n  popl %eax\n"
    "  movl %ebx, 0(%eax)\n  movl %esi, 4(%eax)\n  movl %edi, 8(%eax)\n  movl %ebp, 12(%eax)\n"
    "  popl %edi\n  popl %esi\n  popl %ebx\n  popl %ebp\n  ret\n"
    ".globl _fx_entry_probe@4\n_fx_entry_probe@4:\n"
    "  movl %esp, _fx_entry_sp\n  jmp _fx_entry_c@4\n");
#endif
void fx_gpr_call(const uintptr_t *in, uintptr_t *out, void (*fn)(void *), void *arg);
void WINAPI fx_entry_probe(void *p);
volatile uintptr_t fx_entry_sp;
static volatile long g_entry_ran;
static void *volatile g_entry_ret;
void WINAPI fx_entry_c(void *p) {
    (void)p;
    InterlockedIncrement(&g_entry_ran);
    for (;;) SwitchToFiber(g_entry_ret);
}

static void gpr_pattern(uintptr_t *v, uint32_t tag) {
    for (int i = 0; i < NGPR; i++) v[i] = (uintptr_t)t_pattern_word(tag, 0x30u + (uint32_t)i, 0);
}

/* ---- FP modes and an observable FP "signature" -------------------------- */
static unsigned g_mxcsr_mask;  /* from FXSAVE; DAZ usable iff bit 6 */
static void init_mxcsr_mask(void) {
    static T_ALIGN(16) uint8_t a[512];
    memset(a, 0, sizeof a);
    t_fxsave(a);
    uint32_t m; memcpy(&m, a + 28, 4);
    g_mxcsr_mask = m ? m : 0xFFBFu;
}
static int daz_ok(void) { if (!g_mxcsr_mask) init_mxcsr_mask(); return (g_mxcsr_mask & 0x40) != 0; }

typedef struct fp_sig {
    uint32_t sse_pos, sse_neg, sse_ftz, sse_daz;   /* 1/3, -1/3, FLT_MIN*0.5, denormal+0 */
    uint32_t sse_tenth;                            /* 1/10 (round-to-nearest rounds up, unlike chop) */
    uint8_t  x87_pos[10], x87_neg[10];             /* 1/3, -1/3 at the current PC/RC as 80-bit */
    uint8_t  x87_tenth[10], x87_mtenth[10];        /* 1/10, -1/10: distinguishes nearest from chop */
} fp_sig;

static T_NOINLINE void fp_sig_now(fp_sig *s) {
    memset(s, 0, sizeof *s);
    float a, b, r;
    uint32_t bits;
    a = 1.0f; b = 3.0f;  __asm__ volatile("divss %2, %0" : "=x"(r) : "0"(a), "x"(b)); memcpy(&s->sse_pos, &r, 4);
    a = -1.0f;           __asm__ volatile("divss %2, %0" : "=x"(r) : "0"(a), "x"(b)); memcpy(&s->sse_neg, &r, 4);
    bits = 0x00800000u; memcpy(&a, &bits, 4); b = 0.5f;          /* FLT_MIN * 0.5 -> denormal or 0 (FTZ) */
    __asm__ volatile("mulss %2, %0" : "=x"(r) : "0"(a), "x"(b)); memcpy(&s->sse_ftz, &r, 4);
    bits = 0x00000005u; memcpy(&a, &bits, 4); bits = 0x00000003u; memcpy(&b, &bits, 4);  /* den + den: DAZ -> 0 */
    __asm__ volatile("addss %2, %0" : "=x"(r) : "0"(a), "x"(b)); memcpy(&s->sse_daz, &r, 4);
    a = 1.0f; b = 10.0f; __asm__ volatile("divss %2, %0" : "=x"(r) : "0"(a), "x"(b)); memcpy(&s->sse_tenth, &r, 4);
    static const double one = 1.0, mone = -1.0, three = 3.0, ten = 10.0;
    t_f80 x;
    __asm__ volatile("fldl %1\n\tfdivl %2\n\tfstpt %0" : "=m"(x) : "m"(one), "m"(three)); memcpy(s->x87_pos, x.b, 10);
    __asm__ volatile("fldl %1\n\tfdivl %2\n\tfstpt %0" : "=m"(x) : "m"(mone), "m"(three)); memcpy(s->x87_neg, x.b, 10);
    __asm__ volatile("fldl %1\n\tfdivl %2\n\tfstpt %0" : "=m"(x) : "m"(one), "m"(ten)); memcpy(s->x87_tenth, x.b, 10);
    __asm__ volatile("fldl %1\n\tfdivl %2\n\tfstpt %0" : "=m"(x) : "m"(mone), "m"(ten)); memcpy(s->x87_mtenth, x.b, 10);
}
/* signature produced by mxcsr/cw (computed on the calling thread, state restored) */
static void fp_ref(unsigned mx, unsigned short cw, fp_sig *s) {
    unsigned m0 = t_get_mxcsr(); unsigned short c0 = t_get_x87_cw();
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    fp_sig_now(s);
    t_set_mxcsr(m0); t_set_x87_cw(c0);
}
static int fp_sig_eq(const fp_sig *a, const fp_sig *b) { return !memcmp(a, b, sizeof *a); }

/* Known Wine gaps only pick an XFAIL reason; on Windows the same checks are hard failures. */
static void wine_gap(int failing, const char *why) { if (failing && t_is_wine()) t_expect_fail("Wine: %s", why); }

/* 1: FP control state must be per fiber; 0: must be shared; -1: unspecified (INFO only) */
static int fp_expect(int flagged) {
#ifdef T_X64
    (void)flagged; return 1;          /* x64: nonvolatile MXCSR control / x87 CW always switched */
#else
    return flagged ? 1 : 0;           /* x86: CreateFiberEx/ConvertThreadToFiberEx docs */
#endif
}

/* ======================================================================== */
/*          "duo": a converted thread fiber A ping-pongs with fiber B        */
/* ======================================================================== */
enum { RK_NONE = 0, RK_XMM, RK_YMM, RK_ZMM, RK_GPR };
typedef struct duo {
    /* config */
    int conv;                 /* 0 ConvertThreadToFiber, 1 Ex(0), 2 Ex(FLOAT_SWITCH) */
    int flagB;                /* CreateFiberEx flags for B */
    unsigned mx[2];           /* 0 = leave alone; else set before every switch */
    unsigned short cw[2];
    int regs;                 /* REG_* round trip around every switch */
    int sig;                  /* compute fp signature after every switch */
    long iters;
    uint32_t tag[2];
    /* runtime */
    void *f[2];
    long count, ret[2];
    t_vregs *in[2], *out[2], *bad[2];
    uintptr_t gin[2][NGPR], gout[2][NGPR], gbad[2][NGPR];
    fp_sig ref[2];
    /* results */
    int err; DWORD err_code;
    long mx_bad[2], cw_bad[2], sig_bad[2], nv_bad[2], vol_bad[2], hi_bad[2];
    long mx_other[2], cw_other[2];        /* observed value == the other side's (shared state) */
    unsigned mx_seen[2]; unsigned short cw_seen[2];
    double sec;
} duo;

static int duo_alloc(duo *d) {
    for (int s = 0; s < 2; s++) {
        d->in[s] = vr_alloc(); d->out[s] = vr_alloc(); d->bad[s] = vr_alloc();
        if (!d->in[s] || !d->out[s] || !d->bad[s]) return 0;
        int kind = d->regs == RK_YMM ? T_VK_YMM : d->regs == RK_ZMM ? T_VK_ZMM : T_VK_XMM;
        if (!d->tag[s]) d->tag[s] = 0x100u + (uint32_t)s;
        t_vregs_pattern(d->in[s], kind, d->tag[s]);
        gpr_pattern(d->gin[s], d->tag[s]);
    }
    if (d->sig) for (int s = 0; s < 2; s++) fp_ref(d->mx[s] ? d->mx[s] : 0x1F80, d->cw[s] ? d->cw[s] : 0x27F, &d->ref[s]);
    return 1;
}
static void duo_free(duo *d) { for (int s = 0; s < 2; s++) { vr_free(d->in[s]); vr_free(d->out[s]); vr_free(d->bad[s]); } }

static void duo_switch(duo *d, int s) {
    void *to = d->f[!s];
    int kind = d->regs == RK_YMM ? T_VK_YMM : d->regs == RK_ZMM ? T_VK_ZMM : T_VK_XMM;
    if (d->regs == RK_GPR) {
        fx_gpr_call(d->gin[s], d->gout[s], sw_to, to);
        if (memcmp(d->gin[s], d->gout[s], sizeof d->gin[s])) {
            if (!d->nv_bad[s]) memcpy(d->gbad[s], d->gout[s], sizeof d->gout[s]);
            d->nv_bad[s]++;
        }
        return;
    }
    if (d->regs == RK_NONE) { SwitchToFiber(to); return; }
    t_vregs_call(kind, d->in[s], d->out[s], sw_to, to);
    int n = t_vregs_count(kind);
#ifdef T_X64
    /* nonvolatile: low 128 bits of xmm6-15 */
    if (!vr_eq(d->out[s], d->in[s], 6, 15, 0, 16)) { if (!d->nv_bad[s]) memcpy(d->bad[s], d->out[s], sizeof(t_vregs)); d->nv_bad[s]++; }
    if (!vr_eq(d->out[s], d->in[s], 0, 5, 0, 16)) d->vol_bad[s]++;
    if (kind != T_VK_XMM && !vr_eq(d->out[s], d->in[s], 0, 15, 16, kind - 16)) d->hi_bad[s]++;
    if (kind == T_VK_ZMM && !vr_eq(d->out[s], d->in[s], 16, n - 1, 0, kind)) d->hi_bad[s]++;
#else
    if (!vr_eq(d->out[s], d->in[s], 0, n - 1, 0, 16)) d->vol_bad[s]++;
    if (kind != T_VK_XMM && !vr_eq(d->out[s], d->in[s], 0, n - 1, 16, kind - 16)) d->hi_bad[s]++;
#endif
}

static void duo_body(duo *d, int s) {
    for (;;) {
        if (s == 0 && d->count >= d->iters) return;
        if (d->mx[s]) t_set_mxcsr(d->mx[s]);
        if (d->cw[s]) t_set_x87_cw(d->cw[s]);
        duo_switch(d, s);
        d->ret[s]++;
        if (s == 0) d->count++;
        unsigned m = t_get_mxcsr(); unsigned short c = t_get_x87_cw();
        d->mx_seen[s] = m; d->cw_seen[s] = c;
        if (d->mx[s] && (m & MXC_CTRL) != (d->mx[s] & MXC_CTRL)) {
            d->mx_bad[s]++;
            if (d->mx[!s] && (m & MXC_CTRL) == (d->mx[!s] & MXC_CTRL)) d->mx_other[s]++;
        }
        if (d->cw[s] && (c & CW_CTRL) != (d->cw[s] & CW_CTRL)) {
            d->cw_bad[s]++;
            if (d->cw[!s] && (c & CW_CTRL) == (d->cw[!s] & CW_CTRL)) d->cw_other[s]++;
        }
        if (d->sig) { fp_sig g; fp_sig_now(&g); if (!fp_sig_eq(&g, &d->ref[s])) d->sig_bad[s]++; }
    }
}
static void WINAPI duo_fiber_b(void *p) { duo *d = (duo *)p; duo_body(d, 1); for (;;) SwitchToFiber(d->f[0]); }

static void duo_thread(void *p) {
    duo *d = (duo *)p;
    if (d->conv == 0) d->f[0] = ConvertThreadToFiber(d);
    else d->f[0] = ConvertThreadToFiberEx(d, d->conv == 2 ? FIBER_FLAG_FLOAT_SWITCH : 0);
    if (!d->f[0]) { d->err = 1; d->err_code = GetLastError(); return; }
    d->f[1] = CreateFiberEx(0, 0, (DWORD)d->flagB, duo_fiber_b, d);
    if (!d->f[1]) { d->err = 2; d->err_code = GetLastError(); ConvertFiberToThread(); return; }
    LARGE_INTEGER t0, t1; QueryPerformanceCounter(&t0);
    duo_body(d, 0);
    QueryPerformanceCounter(&t1);
    d->sec = t_qpc_sec(t0.QuadPart, t1.QuadPart);
    DeleteFiber(d->f[1]);
    if (!ConvertFiberToThread()) d->err = 3;
}

/* run a configured duo on a worker thread; returns 0 when it could not run (case already SKIPped) */
static void duo_run(duo *d) {
    if (!duo_alloc(d)) { duo_free(d); SKIP("allocation failed"); }
    int rc = run_thread_ex(duo_thread, d, 0, 120000, NULL);
    if (rc == -1) { duo_free(d); SKIP("CreateThread failed"); }
    CHECKF(d->err == 0, "duo setup error %d (GetLastError %lu)", d->err, (unsigned long)d->err_code);
    CHECK_EQ(d->count, d->iters);
}

static const char *side_name(int s) { return s ? "fiber B" : "fiber A"; }

/* hard checks on a finished duo according to expectations exp[s] (1 per-fiber, 0 shared, -1 info) */
static void duo_report_fp(duo *d, const int exp[2]) {
    int failing = 0;
    for (int s = 0; s < 2; s++) {
        if (exp[s] == 1) failing |= (d->mx[s] && d->mx_bad[s]) || (d->cw[s] && d->cw_bad[s]) || (d->sig && d->sig_bad[s]);
        if (exp[s] == 0) failing |= (d->mx[s] && d->mx_other[s] != d->ret[s]) || (d->cw[s] && d->cw_other[s] != d->ret[s]);
    }
    wine_gap(failing, "SwitchToFiber does not switch MXCSR / x87 control word per fiber");
    for (int s = 0; s < 2; s++) {
        long r = d->ret[s];
        t_info(s ? "B_switch_returns" : "A_switch_returns", "%ld", r);
        if (d->mx[s]) {
            if (exp[s] == 1) CHECKF(d->mx_bad[s] == 0, "%s: MXCSR control 0x%04x expected per fiber, wrong after %ld of %ld switches (last seen 0x%04x)",
                                    side_name(s), d->mx[s] & MXC_CTRL, d->mx_bad[s], r, d->mx_seen[s]);
            else if (exp[s] == 0) CHECKF(d->mx_other[s] == r, "%s: MXCSR expected shared (not switched): saw the other fiber's value %ld of %ld times (last 0x%04x)",
                                         side_name(s), d->mx_other[s], r, d->mx_seen[s]);
            else t_info(s ? "B_mxcsr_switched" : "A_mxcsr_switched", "%s (%ld of %ld wrong)", d->mx_bad[s] ? "no" : "yes", d->mx_bad[s], r);
        }
        if (d->cw[s]) {
            if (exp[s] == 1) CHECKF(d->cw_bad[s] == 0, "%s: x87 CW 0x%04x expected per fiber, wrong after %ld of %ld switches (last seen 0x%04x)",
                                    side_name(s), d->cw[s] & CW_CTRL, d->cw_bad[s], r, d->cw_seen[s]);
            else if (exp[s] == 0) CHECKF(d->cw_other[s] == r, "%s: x87 CW expected shared (not switched): saw the other fiber's value %ld of %ld times (last 0x%04x)",
                                         side_name(s), d->cw_other[s], r, d->cw_seen[s]);
            else t_info(s ? "B_x87cw_switched" : "A_x87cw_switched", "%s (%ld of %ld wrong)", d->cw_bad[s] ? "no" : "yes", d->cw_bad[s], r);
        }
        if (d->sig && exp[s] == 1)
            CHECKF(d->sig_bad[s] == 0, "%s: arithmetic result does not match the fiber's own rounding/FTZ/DAZ/PC mode %ld of %ld times",
                   side_name(s), d->sig_bad[s], r);
        else if (d->sig) t_info(s ? "B_sig_mismatch" : "A_sig_mismatch", "%ld", d->sig_bad[s]);
    }
}

static void duo_report_regs(duo *d) {
    for (int s = 0; s < 2; s++) {
        long r = d->ret[s];
        char lbl[96];
        if (d->regs == RK_GPR) {
            CHECKF(d->nv_bad[s] == 0, "%s: callee-saved GPRs changed across SwitchToFiber %ld of %ld times", side_name(s), d->nv_bad[s], r);
            if (d->nv_bad[s])
                for (int i = 0; i < NGPR; i++)
                    CHECKF(d->gbad[s][i] == d->gin[s][i], "%s %s: got 0x%llx want 0x%llx", side_name(s), k_gpr_names[i],
                           (unsigned long long)d->gbad[s][i], (unsigned long long)d->gin[s][i]);
            continue;
        }
#ifdef T_X64
        CHECKF(d->nv_bad[s] == 0, "%s: xmm6-15 (low 128) changed across SwitchToFiber %ld of %ld times", side_name(s), d->nv_bad[s], r);
        if (d->nv_bad[s]) {
            snprintf(lbl, sizeof lbl, "%s after SwitchToFiber (first bad)", side_name(s));
            for (int i = 6; i <= 15; i++)
                if (memcmp(d->bad[s]->v[i], d->in[s]->v[i], 16)) {
                    char w[128]; snprintf(w, sizeof w, "xmm%d %s", i, lbl);
                    CHECK_REG(w, d->bad[s]->v[i], d->in[s]->v[i], 16);
                }
        }
        t_info(s ? "B_xmm0_5_changed" : "A_xmm0_5_changed", "%ld of %ld", d->vol_bad[s], r);
#else
        (void)lbl;
        t_info(s ? "B_xmm_changed" : "A_xmm_changed", "%ld of %ld (all XMM volatile in the x86 ABI)", d->vol_bad[s], r);
#endif
        if (d->regs == RK_YMM || d->regs == RK_ZMM)
            t_info(s ? "B_upper_changed" : "A_upper_changed", "%ld of %ld (upper YMM/ZMM bits, xmm16-31: volatile)", d->hi_bad[s], r);
    }
}

/* ======================================================================== */
/*                          API basics and errors                           */
/* ======================================================================== */

typedef struct basic_res {
    int r[64]; DWORD e[64]; void *p[64]; int n;
} basic_res;

static void w_initial(void *p) {
    basic_res *b = (basic_res *)p;
    b->r[0] = is_fiber();
    SetLastError(0);
    b->r[1] = ConvertFiberToThread(); b->e[1] = GetLastError();
}
TEST(fiber_api_initial_not_fiber) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_initial, &b);
    if (b.r[0] == -1) t_note("IsThreadAFiber not exported");
    else CHECKF(b.r[0] == FALSE, "IsThreadAFiber on a fresh thread = %d", b.r[0]);
    wine_gap(b.r[1] != FALSE || b.e[1] != ERROR_ALREADY_THREAD, "ConvertFiberToThread succeeds on a thread that is not a fiber");
    CHECKF(b.r[1] == FALSE, "ConvertFiberToThread on a non-fiber thread returned %d", b.r[1]);
    CHECKF(b.e[1] == ERROR_ALREADY_THREAD, "GetLastError %lu want ERROR_ALREADY_THREAD (1281)", (unsigned long)b.e[1]);
}

static void w_convert_basic(void *p) {
    basic_res *b = (basic_res *)p;
    void *param = (void *)(uintptr_t)0x5EED1234u;
    void *f = ConvertThreadToFiber(param);
    b->p[0] = f; b->e[0] = GetLastError();
    if (!f) return;
    b->p[1] = GetCurrentFiber();
    b->p[2] = GetFiberData();
    b->r[3] = is_fiber();
    b->r[4] = ConvertFiberToThread();
    b->r[5] = is_fiber();
    SetLastError(0);
    b->r[6] = ConvertFiberToThread(); b->e[6] = GetLastError();
}
TEST(fiber_convert_basic) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_convert_basic, &b);
    REQUIREF(b.p[0] != NULL, "ConvertThreadToFiber failed, GetLastError %lu", (unsigned long)b.e[0]);
    CHECK(b.p[1] == b.p[0]);
    CHECKF(b.p[2] == (void *)(uintptr_t)0x5EED1234u, "GetFiberData %p", b.p[2]);
    if (b.r[3] != -1) { CHECK_EQ(b.r[3], TRUE); CHECK_EQ(b.r[5], FALSE); }
    CHECK_EQ(b.r[4], TRUE);
    wine_gap(b.r[6] != FALSE || b.e[6] != ERROR_ALREADY_THREAD, "ConvertFiberToThread succeeds on a thread that is not a fiber");
    CHECK_EQ(b.r[6], FALSE);
    CHECK_EQ(b.e[6], ERROR_ALREADY_THREAD);
}

static void w_convert_ex_basic(void *p) {
    basic_res *b = (basic_res *)p;
    for (int i = 0; i < 2; i++) {
        void *param = (void *)(uintptr_t)(0xABC0u + (unsigned)i);
        void *f = ConvertThreadToFiberEx(param, i ? FIBER_FLAG_FLOAT_SWITCH : 0);
        b->p[i * 4 + 0] = f; b->e[i * 4] = GetLastError();
        if (!f) continue;
        b->p[i * 4 + 1] = GetCurrentFiber();
        b->p[i * 4 + 2] = GetFiberData();
        b->r[i * 4 + 3] = is_fiber();
        b->r[i * 4 + 0] = ConvertFiberToThread();
    }
}
TEST(fiber_convert_ex_basic) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_convert_ex_basic, &b);
    for (int i = 0; i < 2; i++) {
        CHECKF(b.p[i * 4] != NULL, "ConvertThreadToFiberEx(flags=%d) failed: %lu", i, (unsigned long)b.e[i * 4]);
        if (!b.p[i * 4]) continue;
        CHECK(b.p[i * 4 + 1] == b.p[i * 4]);
        CHECK(b.p[i * 4 + 2] == (void *)(uintptr_t)(0xABC0u + (unsigned)i));
        if (b.r[i * 4 + 3] != -1) CHECK_EQ(b.r[i * 4 + 3], TRUE);
        CHECK_EQ(b.r[i * 4 + 0], TRUE);
    }
}

static void w_convert_twice(void *p) {
    basic_res *b = (basic_res *)p;
    void *f = ConvertThreadToFiber((void *)0x11);
    b->p[0] = f;
    if (!f) return;
    SetLastError(0);
    b->p[1] = ConvertThreadToFiber((void *)0x22); b->e[1] = GetLastError();
    SetLastError(0);
    b->p[2] = ConvertThreadToFiberEx((void *)0x33, 0); b->e[2] = GetLastError();
    SetLastError(0);
    b->p[3] = ConvertThreadToFiberEx((void *)0x44, FIBER_FLAG_FLOAT_SWITCH); b->e[3] = GetLastError();
    b->p[4] = GetCurrentFiber();
    b->p[5] = GetFiberData();
    b->r[6] = ConvertFiberToThread();
}
TEST(fiber_convert_twice_fails) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_convert_twice, &b);
    REQUIRE(b.p[0] != NULL);
    for (int i = 1; i <= 3; i++) {
        CHECKF(b.p[i] == NULL, "second conversion #%d returned %p (want NULL)", i, b.p[i]);
        CHECKF(b.e[i] == ERROR_ALREADY_FIBER, "second conversion #%d: GetLastError %lu want ERROR_ALREADY_FIBER (1280)", i, (unsigned long)b.e[i]);
    }
    CHECK(b.p[4] == b.p[0]);
    CHECKF(b.p[5] == (void *)0x11, "fiber data changed by failed conversion: %p", b.p[5]);
    CHECK_EQ(b.r[6], TRUE);
}

static const DWORD k_bad_flags[] = { 2, 4, 0x80000000u, 0xFFFFFFFFu, 0xFFFFFFFEu, 0x100u };
static void w_convert_bad_flags(void *p) {
    basic_res *b = (basic_res *)p;
    for (int i = 0; i < (int)T_COUNTOF(k_bad_flags); i++) {
        SetLastError(0);
        b->p[i] = ConvertThreadToFiberEx((void *)0x1, k_bad_flags[i]);
        b->e[i] = GetLastError();
        b->r[i] = is_fiber();
        if (b->p[i]) ConvertFiberToThread();
    }
    /* already a fiber + bad flags: the result must still be a failure */
    void *f = ConvertThreadToFiber(NULL);
    b->p[40] = f;
    if (f) {
        SetLastError(0);
        b->p[41] = ConvertThreadToFiberEx(NULL, 2); b->e[41] = GetLastError();
        b->r[42] = ConvertFiberToThread();
    }
}
TEST(fiber_convert_ex_bad_flags) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_convert_bad_flags, &b);
    { int bad = 0; for (int i = 0; i < (int)T_COUNTOF(k_bad_flags); i++) bad |= b.p[i] != NULL;
      wine_gap(bad, "ConvertThreadToFiberEx does not reject unknown dwFlags bits"); }
    for (int i = 0; i < (int)T_COUNTOF(k_bad_flags); i++) {
        CHECKF(b.p[i] == NULL, "ConvertThreadToFiberEx(flags=0x%lx) succeeded", (unsigned long)k_bad_flags[i]);
        CHECKF(b.e[i] == ERROR_INVALID_PARAMETER, "flags=0x%lx: GetLastError %lu want 87", (unsigned long)k_bad_flags[i], (unsigned long)b.e[i]);
        if (b.r[i] != -1) CHECKF(b.r[i] == FALSE, "thread became a fiber after rejected flags 0x%lx", (unsigned long)k_bad_flags[i]);
    }
    REQUIRE(b.p[40] != NULL);
    CHECK(b.p[41] == NULL);
    t_info("already_fiber_bad_flags_error", "%lu", (unsigned long)b.e[41]);
    CHECK_EQ(b.r[42], TRUE);
}

static void w_convert_cycle(void *p) {
    basic_res *b = (basic_res *)p;
    for (int i = 0; i < 300; i++) {
        void *param = (void *)(uintptr_t)(0x1000u + (unsigned)i);
        void *f = (i % 3 == 0) ? ConvertThreadToFiber(param) : ConvertThreadToFiberEx(param, (i % 3 == 2) ? FIBER_FLAG_FLOAT_SWITCH : 0);
        if (!f) { b->r[0]++; continue; }
        if (GetCurrentFiber() != f) b->r[1]++;
        if (GetFiberData() != param) b->r[2]++;
        if (is_fiber() == FALSE) b->r[3]++;
        if (!ConvertFiberToThread()) b->r[4]++;
        if (is_fiber() == TRUE) b->r[5]++;
        b->n++;
    }
}
TEST(fiber_convert_cycle_300) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_convert_cycle, &b);
    CHECKF(b.r[0] == 0, "%d conversions failed", b.r[0]);
    CHECKF(b.r[1] == 0, "GetCurrentFiber wrong %d times", b.r[1]);
    CHECKF(b.r[2] == 0, "GetFiberData wrong %d times", b.r[2]);
    CHECKF(b.r[3] == 0, "IsThreadAFiber FALSE after conversion %d times", b.r[3]);
    CHECKF(b.r[4] == 0, "ConvertFiberToThread failed %d times", b.r[4]);
    CHECKF(b.r[5] == 0, "IsThreadAFiber TRUE after ConvertFiberToThread %d times", b.r[5]);
    CHECK_EQ(b.n, 300);
}

typedef struct fpenv_res { unsigned mx[8]; unsigned short cw[8]; int ok; } fpenv_res;
static void w_convert_fp(void *p) {
    fpenv_res *r = (fpenv_res *)p;
    t_set_mxcsr(0x7F80 | (daz_ok() ? 0x40 : 0)); t_set_x87_cw(0x0E7F);   /* RC zero (+DAZ), CW chop */
    r->mx[0] = t_get_mxcsr(); r->cw[0] = t_get_x87_cw();
    void *f = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    r->mx[1] = t_get_mxcsr(); r->cw[1] = t_get_x87_cw();
    if (f) { ConvertFiberToThread(); r->ok++; }
    r->mx[2] = t_get_mxcsr(); r->cw[2] = t_get_x87_cw();
    f = ConvertThreadToFiber(NULL);
    r->mx[3] = t_get_mxcsr(); r->cw[3] = t_get_x87_cw();
    if (f) { ConvertFiberToThread(); r->ok++; }
    r->mx[4] = t_get_mxcsr(); r->cw[4] = t_get_x87_cw();
}
TEST(fiber_convert_preserves_fp_env) {
    init_mxcsr_mask();
    fpenv_res r; memset(&r, 0, sizeof r);
    RUN_WORKER(w_convert_fp, &r);
    CHECK_EQ(r.ok, 2);
    static const char *what[] = { "", "after ConvertThreadToFiberEx", "after ConvertFiberToThread", "after ConvertThreadToFiber", "after ConvertFiberToThread #2" };
    for (int i = 1; i <= 4; i++) {
        CHECKF((r.mx[i] & MXC_CTRL) == (r.mx[0] & MXC_CTRL), "MXCSR 0x%04x %s, was 0x%04x", r.mx[i], what[i], r.mx[0]);
        CHECKF((r.cw[i] & CW_CTRL) == (r.cw[0] & CW_CTRL), "x87 CW 0x%04x %s, was 0x%04x", r.cw[i], what[i], r.cw[0]);
    }
}

static void WINAPI fp_never(void *p) { (void)p; }
static void w_create_bad_flags(void *p) {
    basic_res *b = (basic_res *)p;
    for (int i = 0; i < (int)T_COUNTOF(k_bad_flags); i++) {
        SetLastError(0);
        b->p[i] = CreateFiberEx(0, 0, k_bad_flags[i], fp_never, NULL);
        b->e[i] = GetLastError();
        if (b->p[i]) DeleteFiber(b->p[i]);
    }
    void *f = ConvertThreadToFiber(NULL);   /* same from a fiber */
    if (f) {
        for (int i = 0; i < (int)T_COUNTOF(k_bad_flags); i++) {
            SetLastError(0);
            b->p[20 + i] = CreateFiberEx(0, 0, k_bad_flags[i], fp_never, NULL);
            b->e[20 + i] = GetLastError();
            if (b->p[20 + i]) DeleteFiber(b->p[20 + i]);
        }
        ConvertFiberToThread();
        b->r[0] = 1;
    }
}
TEST(fiber_create_ex_bad_flags) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_create_bad_flags, &b);
    CHECK_EQ(b.r[0], 1);
    { int bad = 0; for (int i = 0; i < 40; i++) bad |= b.p[i] != NULL;
      wine_gap(bad, "CreateFiberEx does not reject unknown dwFlags bits"); }
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < (int)T_COUNTOF(k_bad_flags); i++) {
            CHECKF(b.p[k * 20 + i] == NULL, "CreateFiberEx(flags=0x%lx) succeeded (%s)", (unsigned long)k_bad_flags[i], k ? "fiber" : "thread");
            CHECKF(b.e[k * 20 + i] == ERROR_INVALID_PARAMETER, "CreateFiberEx(flags=0x%lx): GetLastError %lu want 87",
                   (unsigned long)k_bad_flags[i], (unsigned long)b.e[k * 20 + i]);
        }
}

/* basic created fiber: parameter, identity, IsThreadAFiber inside */
typedef struct cb_ctx {
    void *main, *self, *param_arg, *data_in, *cur_in; int isf_in; volatile long ran; DWORD tid_in;
} cb_ctx;
static void WINAPI fp_basic(void *p) {
    cb_ctx *c = (cb_ctx *)p;
    c->param_arg = p;
    c->data_in = GetFiberData();
    c->cur_in = GetCurrentFiber();
    c->isf_in = is_fiber();
    c->tid_in = GetCurrentThreadId();
    c->ran++;
    for (;;) SwitchToFiber(c->main);
}
static void w_create_basic(void *p) {
    cb_ctx *c = (cb_ctx *)p;
    c->main = ConvertThreadToFiber(NULL);
    if (!c->main) return;
    c->self = CreateFiber(0, fp_basic, c);
    if (c->self) { SwitchToFiber(c->self); DeleteFiber(c->self); }
    ConvertFiberToThread();
}
TEST(fiber_create_basic_param) {
    cb_ctx c; memset(&c, 0, sizeof c);
    RUN_WORKER(w_create_basic, &c);
    REQUIRE(c.main != NULL);
    REQUIRE(c.self != NULL);
    CHECK_EQ(c.ran, 1);
    CHECK(c.param_arg == &c);
    CHECK(c.data_in == &c);
    CHECK(c.cur_in == c.self);
    if (c.isf_in != -1) CHECK_EQ(c.isf_in, TRUE);
}

static void w_create_nonfiber(void *p) {
    basic_res *b = (basic_res *)p;
    void *f = CreateFiber(0, fp_never, NULL);
    b->p[0] = f; b->e[0] = GetLastError();
    b->r[1] = is_fiber();
    if (f) DeleteFiber(f);
    b->r[2] = is_fiber();
    b->r[3] = 1;   /* DeleteFiber of a non-current fiber must return */
}
TEST(fiber_create_from_non_fiber_thread) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_create_nonfiber, &b);
    CHECKF(b.p[0] != NULL, "CreateFiber on a plain thread failed: %lu", (unsigned long)b.e[0]);
    if (b.r[1] != -1) { CHECK_EQ(b.r[1], FALSE); CHECK_EQ(b.r[2], FALSE); }
    CHECK_EQ(b.r[3], 1);
}

static void w_create_many_unrun(void *p) {
    basic_res *b = (basic_res *)p;
    static void *f[256];
    for (int i = 0; i < 256; i++) {
        f[i] = CreateFiberEx(4096, 64 * 1024, (i & 1) ? FIBER_FLAG_FLOAT_SWITCH : 0, fp_never, (void *)(uintptr_t)i);
        if (f[i]) b->n++; else b->e[0] = GetLastError();
    }
    for (int i = 0; i < 256; i++) if (f[i]) DeleteFiber(f[i]);
    /* reuse after delete */
    for (int i = 0; i < 256; i++) {
        void *g = CreateFiberEx(4096, 64 * 1024, 0, fp_never, NULL);
        if (g) { b->r[0]++; DeleteFiber(g); }
    }
}
TEST(fiber_create_delete_unrun_256) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_create_many_unrun, &b);
    CHECKF(b.n == 256, "created %d of 256 fibers (last error %lu)", b.n, (unsigned long)b.e[0]);
    CHECK_EQ(b.r[0], 256);
}

/* fiber data values */
typedef struct fd_ctx { void *main; void *f[8]; void *seen[8]; void *cur[8]; int ran[8]; } fd_ctx;
static fd_ctx *g_fd;
static void WINAPI fp_data(void *p) {
    void *d = GetFiberData();
    void *cur = GetCurrentFiber();
    for (int i = 0; i < 8; i++) if (g_fd->f[i] == cur) { g_fd->seen[i] = d; g_fd->cur[i] = p; g_fd->ran[i]++; }
    for (;;) SwitchToFiber(g_fd->main);
}
static void *const k_fd_vals[8] = { NULL, (void *)1, (void *)(uintptr_t)0x7FFFFFFF, (void *)~(uintptr_t)0,
                                    (void *)(uintptr_t)0x80000000u, (void *)&g_fd, (void *)(uintptr_t)0xFFFF, (void *)(uintptr_t)0x12345678 };
static void w_fiber_data(void *p) {
    fd_ctx *c = (fd_ctx *)p; g_fd = c;
    c->main = ConvertThreadToFiber((void *)0x77);
    if (!c->main) return;
    for (int i = 0; i < 8; i++) c->f[i] = CreateFiber(0, fp_data, k_fd_vals[i]);
    for (int i = 0; i < 8; i++) if (c->f[i]) SwitchToFiber(c->f[i]);
    for (int i = 7; i >= 0; i--) if (c->f[i]) SwitchToFiber(c->f[i]);   /* resume: no re-entry of the start routine */
    if (GetFiberData() != (void *)0x77) c->ran[0] += 100;
    for (int i = 0; i < 8; i++) if (c->f[i]) DeleteFiber(c->f[i]);
    ConvertFiberToThread();
}
TEST(fiber_data_values) {
    fd_ctx c; memset(&c, 0, sizeof c);
    RUN_WORKER(w_fiber_data, &c);
    REQUIRE(c.main != NULL);
    for (int i = 0; i < 8; i++) {
        REQUIREF(c.f[i] != NULL, "CreateFiber #%d failed", i);
        CHECKF(c.ran[i] == 1, "fiber %d start routine ran %d times", i, c.ran[i]);
        CHECKF(c.seen[i] == k_fd_vals[i], "fiber %d: GetFiberData %p want %p", i, c.seen[i], k_fd_vals[i]);
        CHECKF(c.cur[i] == k_fd_vals[i], "fiber %d: start routine parameter %p want %p", i, c.cur[i], k_fd_vals[i]);
    }
}

/* chain A -> B -> C -> A with identity checks */
typedef struct chain_ctx { void *f[4]; char log[64]; int nlog; int bad; } chain_ctx;
static chain_ctx *g_ch;
static void WINAPI fp_chain(void *p) {
    int me = (int)(uintptr_t)p;
    for (int round = 0;; round++) {
        if (GetCurrentFiber() != g_ch->f[me]) g_ch->bad++;
        if ((int)(uintptr_t)GetFiberData() != me) g_ch->bad++;
        if (g_ch->nlog < 63) g_ch->log[g_ch->nlog++] = (char)('A' + me);
        SwitchToFiber(g_ch->f[(me + 1) % 3]);
    }
}
static void w_chain(void *p) {
    chain_ctx *c = (chain_ctx *)p; g_ch = c;
    c->f[0] = ConvertThreadToFiber((void *)0);
    if (!c->f[0]) return;
    c->f[1] = CreateFiber(0, fp_chain, (void *)1);
    c->f[2] = CreateFiber(0, fp_chain, (void *)2);
    if (c->f[1] && c->f[2])
        for (int round = 0; round < 5; round++) {
            if (c->nlog < 63) c->log[c->nlog++] = 'A';
            if (GetCurrentFiber() != c->f[0]) c->bad++;
            SwitchToFiber(c->f[1]);
        }
    if (c->f[1]) DeleteFiber(c->f[1]);
    if (c->f[2]) DeleteFiber(c->f[2]);
    ConvertFiberToThread();
}
TEST(fiber_chain_three) {
    chain_ctx c; memset(&c, 0, sizeof c);
    RUN_WORKER(w_chain, &c);
    REQUIRE(c.f[0] && c.f[1] && c.f[2]);
    CHECK_EQ(c.bad, 0);
    c.log[c.nlog] = 0;
    CHECKF(!strcmp(c.log, "ABCABCABCABCABC"), "switch order %s", c.log);
}

/* nested creation: each fiber creates the next one */
typedef struct nest_ctx { void *main; void *f[17]; int depth_seen[17]; int max; int bad; } nest_ctx;
static nest_ctx *g_ne;
static void WINAPI fp_nest(void *p) {
    int d = (int)(uintptr_t)p;
    g_ne->depth_seen[d]++;
    if (GetCurrentFiber() != g_ne->f[d]) g_ne->bad++;
    if (d < g_ne->max) {
        g_ne->f[d + 1] = CreateFiberEx(0, 128 * 1024, FIBER_FLAG_FLOAT_SWITCH, fp_nest, (void *)(uintptr_t)(d + 1));
        if (g_ne->f[d + 1]) SwitchToFiber(g_ne->f[d + 1]);
        else g_ne->bad += 1000;
        /* unwinding: control returns here when fiber d+1 switches back */
        if (GetCurrentFiber() != g_ne->f[d]) g_ne->bad++;
        g_ne->depth_seen[d] += 10;
        SwitchToFiber(d ? g_ne->f[d - 1] : g_ne->main);
    } else {
        SwitchToFiber(g_ne->f[d - 1]);
    }
    for (;;) SwitchToFiber(g_ne->main);
}
static void w_nest(void *p) {
    nest_ctx *c = (nest_ctx *)p; g_ne = c;
    c->main = ConvertThreadToFiber(NULL);
    if (!c->main) return;
    c->f[0] = CreateFiber(0, fp_nest, (void *)0);
    if (c->f[0]) SwitchToFiber(c->f[0]);
    for (int i = 0; i <= c->max; i++) if (c->f[i]) DeleteFiber(c->f[i]);
    ConvertFiberToThread();
}
TEST(fiber_nested_create_depth16) {
    nest_ctx c; memset(&c, 0, sizeof c); c.max = 16;
    RUN_WORKER(w_nest, &c);
    REQUIRE(c.main != NULL);
    CHECK_EQ(c.bad, 0);
    for (int i = 0; i < 16; i++) CHECKF(c.depth_seen[i] == 11, "depth %d visited %d (want start + resume = 11)", i, c.depth_seen[i]);
    CHECK_EQ(c.depth_seen[16], 1);
}

/* ======================================================================== */
/*                       ping-pong and FP state                             */
/* ======================================================================== */

TEST(fiber_ping_pong_100000_noflag) {
    static duo d; memset(&d, 0, sizeof d);
    d.conv = 1; d.flagB = 0; d.iters = 100000;
    duo_run(&d);
    CHECK_EQ(d.ret[0], 100000);
    CHECK_EQ(d.ret[1], 99999);   /* B's first activation is its start, not a return */
    duo_free(&d);
}
TEST(fiber_ping_pong_100000_float_switch) {
    static duo d; memset(&d, 0, sizeof d);
    d.conv = 2; d.flagB = FIBER_FLAG_FLOAT_SWITCH; d.iters = 100000;
    d.mx[0] = 0x3F80; d.mx[1] = 0x5F80; d.cw[0] = 0x067F; d.cw[1] = 0x0A7F;
    duo_run(&d);
    CHECK_EQ(d.ret[0], 100000);
    int exp[2] = { 1, 1 };
    duo_report_fp(&d, exp);
    duo_free(&d);
}

TEST(fp_sig_modes_distinct) {
    /* Sanity for the observable-mode checks below: the four rounding modes, FTZ, DAZ and x87 PC must
     * produce distinct results (SDM Vol.1 4.8.4 / 10.2.3 / 8.1.5). */
    init_mxcsr_mask();
    fp_sig s[4], x[4], ftz, daz, pc24, pc64;
    for (int rc = 0; rc < 4; rc++) {
        fp_ref(0x1F80u | ((unsigned)rc << 13), 0x027F, &s[rc]);
        fp_ref(0x1F80u, (unsigned short)(0x027F | (rc << 10)), &x[rc]);
    }
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++) {
            CHECKF(s[i].sse_pos != s[j].sse_pos || s[i].sse_neg != s[j].sse_neg, "SSE RC %d and %d indistinguishable", i, j);
            CHECKF(memcmp(x[i].x87_pos, x[j].x87_pos, 10) || memcmp(x[i].x87_neg, x[j].x87_neg, 10) ||
                   memcmp(x[i].x87_tenth, x[j].x87_tenth, 10) || memcmp(x[i].x87_mtenth, x[j].x87_mtenth, 10), "x87 RC %d and %d indistinguishable", i, j);
            CHECKF(!fp_sig_eq(&s[i], &s[j]), "SSE signatures of RC %d and %d equal", i, j);
        }
    CHECK_EQ(s[0].sse_pos, 0x3EAAAAABu);   /* 1/3 round-to-nearest */
    CHECK_EQ(s[1].sse_pos, 0x3EAAAAAAu);   /* round down */
    fp_ref(0x9F80, 0x027F, &ftz);
    CHECK_EQ(s[0].sse_ftz, 0x00400000u);   /* FLT_MIN/2 = denormal */
    CHECK_EQ(ftz.sse_ftz, 0u);             /* FTZ flushes */
    CHECK_EQ(s[0].sse_daz, 0x00000008u);
    if (daz_ok()) { fp_ref(0x1FC0, 0x027F, &daz); CHECK_EQ(daz.sse_daz, 0u); }
    else t_note("DAZ not supported (MXCSR_MASK 0x%x)", g_mxcsr_mask);
    fp_ref(0x1F80, 0x007F, &pc24); fp_ref(0x1F80, 0x037F, &pc64);
    CHECK(memcmp(pc24.x87_pos, s[0].x87_pos, 10) != 0);
    CHECK(memcmp(pc64.x87_pos, s[0].x87_pos, 10) != 0);
    t_info("mxcsr_mask", "0x%x", g_mxcsr_mask);
}

static void fp_case(int conv, int flagB, unsigned mxA, unsigned mxB, unsigned short cwA, unsigned short cwB,
                    long iters, const int exp[2]) {
    init_mxcsr_mask();
    static duo d; memset(&d, 0, sizeof d);
    d.conv = conv; d.flagB = flagB; d.iters = iters; d.sig = 1;
    d.mx[0] = mxA; d.mx[1] = mxB; d.cw[0] = cwA; d.cw[1] = cwB;
    duo_run(&d);
    duo_report_fp(&d, exp);
    duo_free(&d);
}

TEST(fiber_mxcsr_rc_per_fiber_float_switch) {
    int exp[2] = { 1, 1 };
    fp_case(2, FIBER_FLAG_FLOAT_SWITCH, 0x3F80, 0x5F80, 0, 0, 2000, exp);   /* RC down vs RC up */
}
TEST(fiber_mxcsr_rc_all_pairs_float_switch) {
    init_mxcsr_mask();
    for (unsigned a = 0; a < 4; a++)
        for (unsigned b = 0; b < 4; b++) {
            if (a == b) continue;
            static duo d; memset(&d, 0, sizeof d);
            d.conv = 2; d.flagB = FIBER_FLAG_FLOAT_SWITCH; d.iters = 200; d.sig = 1;
            d.mx[0] = 0x1F80u | (a << 13); d.mx[1] = 0x1F80u | (b << 13);
            duo_run(&d);
            int exp[2] = { 1, 1 };
            duo_report_fp(&d, exp);
            duo_free(&d);
        }
}
TEST(fiber_mxcsr_rc_per_fiber_noflag) {
    int exp[2] = { fp_expect(0), fp_expect(0) };
    fp_case(1, 0, 0x3F80, 0x5F80, 0, 0, 2000, exp);
}
TEST(fiber_mxcsr_ftz_daz_per_fiber_float_switch) {
    init_mxcsr_mask();
    unsigned b = 0x9F80u | (daz_ok() ? 0x40u : 0);
    int exp[2] = { 1, 1 };
    fp_case(2, FIBER_FLAG_FLOAT_SWITCH, 0x1F80, b, 0, 0, 2000, exp);
}
TEST(fiber_mxcsr_ftz_daz_per_fiber_noflag) {
    init_mxcsr_mask();
    unsigned b = 0x9F80u | (daz_ok() ? 0x40u : 0);
    int exp[2] = { fp_expect(0), fp_expect(0) };
    fp_case(1, 0, 0x1F80, b, 0, 0, 2000, exp);
}
TEST(fiber_mxcsr_mask_bits_per_fiber_float_switch) {
    /* exception mask bits differ (DM unmasked in B); no denormal operands are used while it is unmasked */
    int exp[2] = { 1, 1 };
    static duo d; memset(&d, 0, sizeof d);
    d.conv = 2; d.flagB = FIBER_FLAG_FLOAT_SWITCH; d.iters = 2000;
    d.mx[0] = 0x1F80; d.mx[1] = 0x1E80;
    duo_run(&d);
    duo_report_fp(&d, exp);
    duo_free(&d);
}
TEST(fiber_x87_cw_rc_per_fiber_float_switch) {
    int exp[2] = { 1, 1 };
    fp_case(2, FIBER_FLAG_FLOAT_SWITCH, 0, 0, 0x067F, 0x0A7F, 2000, exp);
}
TEST(fiber_x87_cw_pc_per_fiber_float_switch) {
    int exp[2] = { 1, 1 };
    fp_case(2, FIBER_FLAG_FLOAT_SWITCH, 0, 0, 0x007F, 0x037F, 2000, exp);   /* PC 24 vs PC 64 */
}
TEST(fiber_x87_cw_per_fiber_noflag) {
    int exp[2] = { fp_expect(0), fp_expect(0) };
    fp_case(1, 0, 0, 0, 0x047F, 0x0B7F, 2000, exp);
}
TEST(fiber_fp_full_mode_mix_float_switch) {
    init_mxcsr_mask();
    int exp[2] = { 1, 1 };
    fp_case(2, FIBER_FLAG_FLOAT_SWITCH, 0x7F80u | (daz_ok() ? 0x40u : 0), 0xBF80, 0x0E7F, 0x007F, 5000, exp);
}
TEST(fiber_fp_mixed_flags_a_flagged) {
    /* A (converted with FIBER_FLAG_FLOAT_SWITCH) keeps its state; B (no flag) is per-fiber only on x64 */
#ifdef T_X64
    int exp[2] = { 1, 1 };
#else
    int exp[2] = { 1, -1 };
#endif
    fp_case(2, 0, 0x3F80, 0x5F80, 0x067F, 0x0A7F, 2000, exp);
}
TEST(fiber_fp_mixed_flags_b_flagged) {
#ifdef T_X64
    int exp[2] = { 1, 1 };
#else
    int exp[2] = { -1, 1 };
#endif
    fp_case(1, FIBER_FLAG_FLOAT_SWITCH, 0x3F80, 0x5F80, 0x067F, 0x0A7F, 2000, exp);
}
TEST(fiber_fp_plain_convert_thread_to_fiber) {
    /* ConvertThreadToFiber has no flags parameter: x64 always switches; on x86 it is not specified */
#ifdef T_X64
    int exp[2] = { 1, 1 };
#else
    int exp[2] = { -1, 1 };
#endif
    fp_case(0, FIBER_FLAG_FLOAT_SWITCH, 0x7F80, 0x3F80, 0x0E7F, 0x067F, 2000, exp);
}

typedef struct mxst_ctx { void *main, *f; unsigned mx_b_start, mx_a_after, mx_b_after; unsigned short cw_b_start, sw_a_after; } mxst_ctx;
static mxst_ctx *g_ms;
static void WINAPI fp_mxst(void *p) {
    (void)p;
    g_ms->mx_b_start = t_get_mxcsr(); g_ms->cw_b_start = t_get_x87_cw();
    t_set_mxcsr(t_get_mxcsr() & ~MXC_FLAGS);
    SwitchToFiber(g_ms->main);
    g_ms->mx_b_after = t_get_mxcsr();
    for (;;) SwitchToFiber(g_ms->main);
}
static void w_mxst(void *p) {
    mxst_ctx *c = (mxst_ctx *)p; g_ms = c;
    c->main = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!c->main) return;
    t_set_mxcsr(0x3F80 | 0x20);           /* creator: RC down + PE flag */
    t_set_x87_cw(0x0E7F);
    c->f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, fp_mxst, NULL);
    if (c->f) {
        SwitchToFiber(c->f);
        c->mx_a_after = t_get_mxcsr();
        fp_sig s; fp_sig_now(&s);         /* raises PE in A */
        SwitchToFiber(c->f);
        unsigned short sw; __asm__ volatile("fnstsw %0" : "=m"(sw)); c->sw_a_after = sw;
        DeleteFiber(c->f);
    }
    ConvertFiberToThread();
}
TEST(fiber_new_fiber_initial_fp_and_flags_info) {
    mxst_ctx c; memset(&c, 0, sizeof c);
    RUN_WORKER(w_mxst, &c);
    REQUIRE(c.main && c.f);
    t_info("new_fiber_mxcsr", "0x%04x (creator had 0x3FA0)", c.mx_b_start);
    t_info("new_fiber_x87_cw", "0x%04x (creator had 0x0E7F)", c.cw_b_start);
    t_info("A_mxcsr_after_B_cleared_flags", "0x%04x", c.mx_a_after);
    t_info("B_mxcsr_after_A_raised_PE", "0x%04x", c.mx_b_after);
    t_info("A_x87_sw_after_switch", "0x%04x", c.sw_a_after);
    /* the creator's control bits must come back to it (x64 always; x86 both flagged) */
    CHECKF((c.mx_a_after & MXC_CTRL) == 0x3F80, "creator MXCSR control 0x%04x after the new fiber ran", c.mx_a_after & MXC_CTRL);
}

/* ---- registers ----------------------------------------------------------- */
static void reg_case(int regs, int conv, int flagB, long iters) {
    static duo d; memset(&d, 0, sizeof d);
    d.conv = conv; d.flagB = flagB; d.iters = iters; d.regs = regs;
    d.tag[0] = 0xF1B0u + (uint32_t)regs; d.tag[1] = 0xF1C0u + (uint32_t)regs;
    duo_run(&d);
    duo_report_regs(&d);
    duo_free(&d);
}
TEST(fiber_xmm6_15_preserved_noflag) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
#ifdef T_X86
    t_note("x86 ABI: no XMM register is callee-saved; observing only");
#endif
    reg_case(RK_XMM, 1, 0, 10000);
}
TEST(fiber_xmm6_15_preserved_float_switch) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    reg_case(RK_XMM, 2, FIBER_FLAG_FLOAT_SWITCH, 10000);
}
TEST(fiber_xmm6_15_preserved_plain_convert) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    reg_case(RK_XMM, 0, 0, 10000);
}
TEST(fiber_ymm_low128_preserved) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    reg_case(RK_YMM, 2, FIBER_FLAG_FLOAT_SWITCH, 5000);
}
TEST(fiber_zmm_low128_preserved) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    reg_case(RK_ZMM, 2, FIBER_FLAG_FLOAT_SWITCH, 5000);
}
TEST(fiber_gpr_nonvolatile_preserved_noflag) { reg_case(RK_GPR, 1, 0, 20000); }
TEST(fiber_gpr_nonvolatile_preserved_float_switch) { reg_case(RK_GPR, 2, FIBER_FLAG_FLOAT_SWITCH, 20000); }

/* volatile state across a switch is not specified; record it */
TEST(fiber_volatile_xmm_info) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static duo d; memset(&d, 0, sizeof d);
    d.conv = 1; d.iters = 100; d.regs = RK_XMM;
    duo_run(&d);
#ifdef T_X64
    t_info("xmm0_5_survive_switch", "%s", d.vol_bad[0] ? "no" : "yes");
#else
    t_info("xmm0_7_survive_switch", "%s", d.vol_bad[0] ? "no" : "yes");
#endif
    duo_free(&d);
}

/* ---- perf ----------------------------------------------------------------- */
TEST(fiber_switch_perf) {
    static duo d;
    double ns[2] = { 0, 0 };
    for (int k = 0; k < 2; k++) {
        memset(&d, 0, sizeof d);
        d.conv = k ? 2 : 1; d.flagB = k ? FIBER_FLAG_FLOAT_SWITCH : 0; d.iters = 200000;
        duo_run(&d);
        CHECK_EQ(d.ret[0], 200000);
        if (d.count == d.iters && d.sec > 0) ns[k] = d.sec * 1e9 / (2.0 * (double)d.iters);
        duo_free(&d);
    }
    if (ns[0] > 0) t_perf("switch_noflag", ns[0], "ns/switch");
    if (ns[1] > 0) t_perf("switch_float_switch", ns[1], "ns/switch");
}

/* ======================================================================== */
/*                     termination and deletion                             */
/* ======================================================================== */
typedef struct term_ctx { void *main, *f; volatile long ran, after, after_fiber; DWORD code; int mode; } term_ctx;
static term_ctx *g_tc;
static void WINAPI fp_term(void *p) {
    (void)p;
    g_tc->ran = 1;
    if (g_tc->mode == 1) ExitThread(0x1234);
    if (g_tc->mode == 2) DeleteFiber(GetCurrentFiber());
    if (g_tc->mode == 3) { SwitchToFiber(g_tc->main); g_tc->after_fiber = 1; for (;;) SwitchToFiber(g_tc->main); }
    /* mode 0: return from the fiber function -> the thread exits */
}
static DWORD WINAPI th_term(void *p) {
    term_ctx *c = (term_ctx *)p; g_tc = c;
    c->main = ConvertThreadToFiber(NULL);
    if (!c->main) return 0xBAD1;
    c->f = CreateFiber(0, fp_term, NULL);
    if (!c->f) return 0xBAD2;
    if (c->mode == 4) { c->ran = 1; DeleteFiber(c->main); c->after = 1; return 0xBAD3; }
    SwitchToFiber(c->f);
    c->after = 1;
    return 0xBAD4;
}
static void term_run(term_ctx *c) {
    HANDLE h = CreateThread(NULL, 0, th_term, c, 0, NULL);
    if (!h) SKIP("CreateThread failed");
    DWORD w = WaitForSingleObject(h, 10000);
    CHECKF(w == WAIT_OBJECT_0, "thread did not exit (wait %lu)", (unsigned long)w);
    if (w != WAIT_OBJECT_0) { TerminateThread(h, 1); WaitForSingleObject(h, 2000); }
    GetExitCodeThread(h, &c->code);
    CloseHandle(h);
    CHECK_EQ(c->ran, 1);
    CHECKF(c->after == 0, "code after the terminating call ran");
}
TEST(fiber_return_exits_thread) {
    term_ctx c; memset(&c, 0, sizeof c); c.mode = 0;
    term_run(&c);
    t_info("thread_exit_code", "0x%lx", (unsigned long)c.code);
    CHECK(c.code != 0xBAD4u && c.code != STILL_ACTIVE);
}
TEST(fiber_exitthread_in_fiber) {
    term_ctx c; memset(&c, 0, sizeof c); c.mode = 1;
    term_run(&c);
    CHECK_EQ(c.code, 0x1234u);   /* ExitThread(dwExitCode) */
}
TEST(fiber_delete_current_exits_thread) {
    term_ctx c; memset(&c, 0, sizeof c); c.mode = 2;
    term_run(&c);
    t_info("thread_exit_code", "0x%lx", (unsigned long)c.code);
    CHECK(c.code != 0xBAD4u && c.code != STILL_ACTIVE);
}
TEST(fiber_delete_current_primary_exits_thread) {
    term_ctx c; memset(&c, 0, sizeof c); c.mode = 4;
    term_run(&c);
    t_info("thread_exit_code", "0x%lx", (unsigned long)c.code);
    CHECK(c.code != 0xBAD3u && c.code != STILL_ACTIVE);
}

/* delete a fiber suspended in the middle of its function, then keep using fibers */
typedef struct dso_ctx { void *main, *b, *c; int b_ran, b_resumed, c_ran; int del_ok; } dso_ctx;
static dso_ctx *g_dso;
static void WINAPI fp_dso_b(void *p) { (void)p; g_dso->b_ran++; volatile char big[16384]; big[0] = 1; big[16383] = big[0]; SwitchToFiber(g_dso->main); g_dso->b_resumed++; for (;;) SwitchToFiber(g_dso->main); }
static void WINAPI fp_dso_c(void *p) { (void)p; g_dso->c_ran++; for (;;) SwitchToFiber(g_dso->main); }
static void w_dso(void *p) {
    dso_ctx *c = (dso_ctx *)p; g_dso = c;
    c->main = ConvertThreadToFiber(NULL);
    if (!c->main) return;
    for (int i = 0; i < 50; i++) {
        c->b = CreateFiber(0, fp_dso_b, NULL);
        if (!c->b) break;
        SwitchToFiber(c->b);
        DeleteFiber(c->b);
        c->c = CreateFiber(0, fp_dso_c, NULL);
        if (!c->c) break;
        SwitchToFiber(c->c);
        DeleteFiber(c->c);
        c->del_ok++;
    }
    ConvertFiberToThread();
}
TEST(fiber_delete_suspended_other) {
    dso_ctx c; memset(&c, 0, sizeof c);
    RUN_WORKER(w_dso, &c);
    REQUIRE(c.main != NULL);
    CHECK_EQ(c.del_ok, 50);
    CHECK_EQ(c.b_ran, 50);
    CHECK_EQ(c.b_resumed, 0);
    CHECK_EQ(c.c_ran, 50);
}

/* ======================================================================== */
/*                          fibers across threads                           */
/* ======================================================================== */
typedef struct xt_ctx {
    void *f; void *ret; HANDLE ev_go, ev_done; int ran; DWORD tid_creator, tid_runner, tid_in; void *cur_in, *data_in;
    int created; int deleted_by_other;
} xt_ctx;
static xt_ctx *g_xt;
static void WINAPI fp_xt(void *p) {
    g_xt->ran++;
    g_xt->tid_in = GetCurrentThreadId();
    g_xt->cur_in = GetCurrentFiber();
    g_xt->data_in = GetFiberData();
    (void)p;
    for (;;) SwitchToFiber(g_xt->ret);
}
static void w_xt_creator(void *p) {
    xt_ctx *c = (xt_ctx *)p;
    c->tid_creator = GetCurrentThreadId();
    c->f = CreateFiber(0, fp_xt, (void *)0xF00D);
    c->created = c->f != NULL;
}
static void w_xt_runner(void *p) {
    xt_ctx *c = (xt_ctx *)p;
    c->tid_runner = GetCurrentThreadId();
    c->ret = ConvertThreadToFiber(NULL);
    if (!c->ret) return;
    SwitchToFiber(c->f);
    ConvertFiberToThread();
}
static void w_xt_deleter(void *p) { xt_ctx *c = (xt_ctx *)p; DeleteFiber(c->f); c->deleted_by_other = 1; }
TEST(fiber_run_on_other_thread) {
    xt_ctx c; memset(&c, 0, sizeof c); g_xt = &c;
    RUN_WORKER(w_xt_creator, &c);
    REQUIRE(c.created);
    RUN_WORKER(w_xt_runner, &c);
    CHECK_EQ(c.ran, 1);
    CHECK(c.tid_in == c.tid_runner);
    CHECK(c.tid_in != c.tid_creator);
    CHECK(c.cur_in == c.f);
    CHECK(c.data_in == (void *)0xF00D);
    RUN_WORKER(w_xt_deleter, &c);   /* deleted by a third thread while not running */
    CHECK_EQ(c.deleted_by_other, 1);
}

/* fiber F hops between 4 threads (strict baton: never concurrently) */
#define MIG_T 4
typedef struct mig_ctx {
    void *f; void *ret; HANDLE ev[MIG_T]; DWORD tid_expect; volatile long hops; long max_hops;
    DWORD fls, tls; uintptr_t stk_lo, stk_hi; unsigned mxF; unsigned short cwF;
    unsigned mxT[MIG_T]; unsigned short cwT[MIG_T];
    long bad_tid, bad_cur, bad_data, bad_fls, bad_tls, bad_mx, bad_cw, bad_stack, bad_xmm, bad_tmx, bad_tcw, bad_timeout;
    long tid_changes; DWORD last_tid;
    t_vregs *in, *out;
    int threads_ok;
} mig_ctx;
static mig_ctx *g_mg;
static void mig_check(mig_ctx *m) {
    DWORD tid = GetCurrentThreadId();
    if (tid != m->tid_expect) m->bad_tid++;
    if (tid != m->last_tid) m->tid_changes++;
    m->last_tid = tid;
    if (GetCurrentFiber() != m->f) m->bad_cur++;
    if (GetFiberData() != (void *)m) m->bad_data++;
    if (FlsGetValue(m->fls) != (void *)0xF15F15) m->bad_fls++;
    if ((DWORD)(uintptr_t)TlsGetValue(m->tls) != tid) m->bad_tls++;
    if ((t_get_mxcsr() & MXC_CTRL) != m->mxF) m->bad_mx++;
    if ((t_get_x87_cw() & CW_CTRL) != (m->cwF & CW_CTRL)) m->bad_cw++;
    uintptr_t lo, hi; stack_range(&lo, &hi);
    if (lo != m->stk_lo || hi != m->stk_hi || (uintptr_t)&lo < lo || (uintptr_t)&lo >= hi) m->bad_stack++;
}
static void WINAPI fp_mig(void *p) {
    mig_ctx *m = (mig_ctx *)p;
    FlsSetValue(m->fls, (void *)0xF15F15);
    stack_range(&m->stk_lo, &m->stk_hi);
    t_set_mxcsr(m->mxF); t_set_x87_cw(m->cwF);
    m->last_tid = GetCurrentThreadId();
    for (;;) {
        mig_check(m);
        if (cpu_has(F_SSE2)) {
            t_vregs_call(T_VK_XMM, m->in, m->out, sw_to, m->ret);
#ifdef T_X64
            if (!vr_eq(m->out, m->in, 6, 15, 0, 16)) m->bad_xmm++;
#endif
        } else SwitchToFiber(m->ret);
    }
}
typedef struct mig_thr { mig_ctx *m; int idx; } mig_thr;
static DWORD WINAPI th_mig(void *p) {
    mig_thr *t = (mig_thr *)p; mig_ctx *m = t->m; int k = t->idx;
    TlsSetValue(m->tls, (void *)(uintptr_t)GetCurrentThreadId());
    void *me = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!me) return 1;
    t_set_mxcsr(m->mxT[k]); t_set_x87_cw(m->cwT[k]);
    for (;;) {
        if (WaitForSingleObject(m->ev[k], 10000) != WAIT_OBJECT_0) { m->bad_timeout++; break; }
        if (m->hops >= m->max_hops) { SetEvent(m->ev[(k + 1) % MIG_T]); break; }
        m->ret = me; m->tid_expect = GetCurrentThreadId();
        SwitchToFiber(m->f);
        if ((t_get_mxcsr() & MXC_CTRL) != m->mxT[k]) m->bad_tmx++;
        if ((t_get_x87_cw() & CW_CTRL) != (m->cwT[k] & CW_CTRL)) m->bad_tcw++;
        m->hops++;
        SetEvent(m->ev[(k + 1) % MIG_T]);
    }
    ConvertFiberToThread();
    return 0;
}
TEST(fiber_migrate_threads_1000) {
    static mig_ctx m; memset(&m, 0, sizeof m); g_mg = &m;
    m.max_hops = 1000; m.mxF = 0x5F80; m.cwF = 0x0A7F;
    static const unsigned mxt[MIG_T] = { 0x1F80, 0x3F80, 0x7F80, 0x9F80 };
    static const unsigned short cwt[MIG_T] = { 0x027F, 0x067F, 0x0E7F, 0x007F };
    for (int i = 0; i < MIG_T; i++) { m.mxT[i] = mxt[i]; m.cwT[i] = cwt[i]; }
    m.fls = FlsAlloc(NULL); m.tls = TlsAlloc();
    REQUIRE(m.fls != FLS_OUT_OF_INDEXES && m.tls != TLS_OUT_OF_INDEXES);
    m.in = vr_alloc(); m.out = vr_alloc(); REQUIRE(m.in && m.out);
    t_vregs_pattern(m.in, T_VK_XMM, 0x3161);
    m.f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, fp_mig, &m);
    REQUIRE(m.f != NULL);
    HANDLE th[MIG_T]; mig_thr ta[MIG_T]; int nth = 0;
    for (int i = 0; i < MIG_T; i++) m.ev[i] = CreateEventA(NULL, FALSE, FALSE, NULL);
    for (int i = 0; i < MIG_T; i++) {
        ta[i].m = &m; ta[i].idx = i;
        th[i] = CreateThread(NULL, 0, th_mig, &ta[i], 0, NULL);
        if (th[i]) nth++;
    }
    if (nth != MIG_T) {
        /* cannot run the baton: release whatever started (they time out on their event) */
        m.max_hops = 0;
        for (int i = 0; i < MIG_T; i++) SetEvent(m.ev[i]);
        for (int i = 0; i < MIG_T; i++) if (th[i]) { WaitForSingleObject(th[i], 15000); CloseHandle(th[i]); }
        SKIP("could not create %d threads", MIG_T);
    }
    SetEvent(m.ev[0]);
    DWORD w = WaitForMultipleObjects(MIG_T, th, TRUE, 60000);
    CHECKF(w < WAIT_OBJECT_0 + MIG_T, "migration threads did not finish (%lu)", (unsigned long)w);
    for (int i = 0; i < MIG_T; i++) { if (w >= WAIT_OBJECT_0 + MIG_T) TerminateThread(th[i], 1); CloseHandle(th[i]); CloseHandle(m.ev[i]); }
    CHECK_EQ(m.hops, 1000);
    CHECK_EQ(m.bad_timeout, 0);
    CHECKF(m.tid_changes >= 999, "fiber changed threads only %ld times", m.tid_changes);
    CHECK_EQ(m.bad_tid, 0); CHECK_EQ(m.bad_cur, 0); CHECK_EQ(m.bad_data, 0);
    CHECKF(m.bad_fls == 0, "FLS value lost after migration %ld times", m.bad_fls);
    CHECKF(m.bad_tls == 0, "TLS seen by fiber was not the running thread's %ld times", m.bad_tls);
    CHECKF(m.bad_stack == 0, "fiber stack limits wrong after migration %ld times", m.bad_stack);
    wine_gap(m.bad_mx || m.bad_cw || m.bad_tmx || m.bad_tcw, "SwitchToFiber does not switch MXCSR / x87 control word per fiber");
    CHECKF(m.bad_mx == 0, "fiber MXCSR control lost across threads %ld times", m.bad_mx);
    CHECKF(m.bad_cw == 0, "fiber x87 CW lost across threads %ld times", m.bad_cw);
    CHECKF(m.bad_tmx == 0, "thread primary fiber MXCSR wrong after switch back %ld times", m.bad_tmx);
    CHECKF(m.bad_tcw == 0, "thread primary fiber x87 CW wrong after switch back %ld times", m.bad_tcw);
#ifdef T_X64
    CHECKF(m.bad_xmm == 0, "xmm6-15 of the migrating fiber changed %ld times", m.bad_xmm);
#endif
    DeleteFiber(m.f);
    FlsFree(m.fls); TlsFree(m.tls);
    vr_free(m.in); vr_free(m.out);
}

/* many threads, each with its own fiber pair, distinct modes and register patterns */
#define MT_N 16
TEST(fiber_threads_16x_ping_pong) {
    init_mxcsr_mask();
    static duo d[MT_N];
    HANDLE th[MT_N]; static thr_arg ta[MT_N];
    int n = 0;
    for (int i = 0; i < MT_N; i++) {
        memset(&d[i], 0, sizeof d[i]);
        d[i].conv = 2; d[i].flagB = FIBER_FLAG_FLOAT_SWITCH; d[i].iters = 5000;
        d[i].regs = cpu_has(F_SSE2) ? RK_XMM : RK_NONE;
        d[i].mx[0] = 0x1F80u | ((unsigned)(i & 3) << 13); d[i].mx[1] = 0x1F80u | ((unsigned)((i + 1) & 3) << 13) | ((i & 4) ? 0x8000u : 0);
        d[i].cw[0] = (unsigned short)(0x027F | ((i & 3) << 10)); d[i].cw[1] = (unsigned short)(0x007F | (((i + 2) & 3) << 10));
        d[i].tag[0] = 0x5000u + (uint32_t)i * 2; d[i].tag[1] = 0x5001u + (uint32_t)i * 2;
        if (!duo_alloc(&d[i])) { for (int j = 0; j <= i; j++) duo_free(&d[j]); SKIP("allocation failed"); }
    }
    for (int i = 0; i < MT_N; i++) {
        ta[i].fn = duo_thread; ta[i].arg = &d[i];
        th[n] = CreateThread(NULL, 0, thr_tramp, &ta[i], 0, NULL);
        if (!th[n]) break;
        n++;
    }
    for (int i = 0; i < n; i++) {
        DWORD w = WaitForSingleObject(th[i], 60000);
        CHECKF(w == WAIT_OBJECT_0, "thread %d did not finish", i);
        if (w != WAIT_OBJECT_0) TerminateThread(th[i], 1);
        CloseHandle(th[i]);
    }
    if (n < 2) { for (int i = 0; i < MT_N; i++) duo_free(&d[i]); SKIP("could not create threads"); }
    t_info("threads", "%d", n);
    { int fpbad = 0; for (int i = 0; i < n; i++) for (int s = 0; s < 2; s++) fpbad |= d[i].mx_bad[s] || d[i].cw_bad[s];
      wine_gap(fpbad, "SwitchToFiber does not switch MXCSR / x87 control word per fiber"); }
    for (int i = 0; i < n; i++) {
        CHECKF(d[i].err == 0 && d[i].count == 5000, "thread %d: err %d count %ld", i, d[i].err, d[i].count);
        for (int s = 0; s < 2; s++) {
            CHECKF(d[i].mx_bad[s] == 0, "thread %d %s: MXCSR wrong %ld times (last 0x%04x)", i, side_name(s), d[i].mx_bad[s], d[i].mx_seen[s]);
            CHECKF(d[i].cw_bad[s] == 0, "thread %d %s: x87 CW wrong %ld times (last 0x%04x)", i, side_name(s), d[i].cw_bad[s], d[i].cw_seen[s]);
#ifdef T_X64
            CHECKF(d[i].nv_bad[s] == 0, "thread %d %s: xmm6-15 changed %ld times", i, side_name(s), d[i].nv_bad[s]);
#endif
        }
    }
    for (int i = 0; i < MT_N; i++) duo_free(&d[i]);
}

/* 1024 fibers in a ring on one thread */
#define RR_MAX 1024
typedef struct rr_ctx {
    void *main; void *f[RR_MAX]; int n; int rounds; t_vregs *in, *out;
    unsigned mx[RR_MAX]; unsigned short cw[RR_MAX];
    long visits[RR_MAX]; long bad_mx, bad_cw, bad_xmm, bad_cur, main_bad;
} rr_ctx;
static rr_ctx *g_rr;
static void WINAPI fp_rr(void *p) {
    int i = (int)(uintptr_t)p;
    rr_ctx *r = g_rr;
    t_vregs *in = (t_vregs *)((uint8_t *)r->in + (size_t)i * sizeof(t_vregs));
    t_vregs *out = (t_vregs *)((uint8_t *)r->out + (size_t)i * sizeof(t_vregs));
    /* first activation chains fiber -> fiber (round 0); later the main fiber resumes each one */
    void *next = (i + 1 < r->n) ? r->f[i + 1] : r->main;
    for (;;) {
        t_set_mxcsr(r->mx[i]); t_set_x87_cw(r->cw[i]);
        t_vregs_call(T_VK_XMM, in, out, sw_to, next);
        next = r->main;
        r->visits[i]++;
        if (GetCurrentFiber() != r->f[i]) r->bad_cur++;
        if ((t_get_mxcsr() & MXC_CTRL) != (r->mx[i] & MXC_CTRL)) r->bad_mx++;
        if ((t_get_x87_cw() & CW_CTRL) != (r->cw[i] & CW_CTRL)) r->bad_cw++;
#ifdef T_X64
        if (!vr_eq(out, in, 6, 15, 0, 16)) r->bad_xmm++;
#endif
    }
}
static void w_rr(void *p) {
    rr_ctx *r = (rr_ctx *)p; g_rr = r;
    r->main = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!r->main) return;
    int n = 0;
    for (int i = 0; i < RR_MAX; i++) {
        r->f[i] = CreateFiberEx(4096, 64 * 1024, FIBER_FLAG_FLOAT_SWITCH, fp_rr, (void *)(uintptr_t)i);
        if (!r->f[i]) break;
        n++;
    }
    r->n = n;
    t_set_mxcsr(0x1F80); t_set_x87_cw(0x027F);
    if (n > 0) {
        /* round 0 starts every fiber in turn (each switches to the next before returning to main);
         * later rounds resume each fiber individually */
        SwitchToFiber(r->f[0]);
        for (int round = 0; round < r->rounds; round++) {
            for (int i = 0; i < n; i++) {
                SwitchToFiber(r->f[i]);
                if ((t_get_mxcsr() & MXC_CTRL) != 0x1F80 || (t_get_x87_cw() & CW_CTRL) != (0x027F & CW_CTRL)) r->main_bad++;
            }
        }
    }
    for (int i = 0; i < n; i++) DeleteFiber(r->f[i]);
    ConvertFiberToThread();
}
TEST(fiber_many_1024_ring) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    init_mxcsr_mask();
    static rr_ctx r; memset(&r, 0, sizeof r);
    r.rounds = 4;
    r.in = (t_vregs *)_aligned_malloc(sizeof(t_vregs) * RR_MAX, 64);
    r.out = (t_vregs *)_aligned_malloc(sizeof(t_vregs) * RR_MAX, 64);
    if (!r.in || !r.out) { if (r.in) _aligned_free(r.in); if (r.out) _aligned_free(r.out); SKIP("allocation failed"); }
    for (int i = 0; i < RR_MAX; i++) {
        t_vregs_pattern(&r.in[i], T_VK_XMM, 0x9000u + (uint32_t)i);
        unsigned rc = (unsigned)(i & 3), ftz = (unsigned)((i >> 2) & 1), daz = daz_ok() ? (unsigned)((i >> 3) & 1) : 0;
        r.mx[i] = 0x1F80u | (rc << 13) | (ftz << 15) | (daz << 6);
        r.cw[i] = (unsigned short)(0x007F | (((i >> 4) & 3) << 8) | (((i + 1) & 3) << 10));
        if (((r.cw[i] >> 8) & 3) == 1) r.cw[i] = (unsigned short)(r.cw[i] | 0x0200);   /* PC=01 reserved -> 11 */
    }
    int rc = run_thread_ex(w_rr, &r, 0, 120000, NULL);
    _aligned_free(r.in); _aligned_free(r.out);
    if (rc == -1) SKIP("CreateThread failed");
    REQUIRE(r.main != NULL);
    t_info("fibers_created", "%d", r.n);
    if (r.n < 64) SKIP("only %d fibers could be created", r.n);
    if (r.n < RR_MAX) t_note("created %d of %d fibers", r.n, RR_MAX);
    for (int i = 0; i < r.n; i++) if (r.visits[i] != r.rounds) { CHECKF(0, "fiber %d resumed %ld times, want %d", i, r.visits[i], r.rounds); break; }
    wine_gap(r.bad_mx || r.bad_cw || r.main_bad, "SwitchToFiber does not switch MXCSR / x87 control word per fiber");
    CHECK_EQ(r.bad_cur, 0);
    CHECKF(r.bad_mx == 0, "per-fiber MXCSR wrong %ld times", r.bad_mx);
    CHECKF(r.bad_cw == 0, "per-fiber x87 CW wrong %ld times", r.bad_cw);
    CHECKF(r.main_bad == 0, "main fiber FP control wrong %ld times", r.main_bad);
#ifdef T_X64
    CHECKF(r.bad_xmm == 0, "xmm6-15 wrong %ld times", r.bad_xmm);
#endif
}

/* FP state follows a fiber to another thread */
typedef struct fpx_ctx { void *f, *ret; unsigned mx_in[2]; unsigned short cw_in[2]; unsigned mx_t[2]; unsigned short cw_t[2]; int step; } fpx_ctx;
static fpx_ctx *g_fpx;
static void WINAPI fp_fpx(void *p) {
    (void)p;
    t_set_mxcsr(0x5F80); t_set_x87_cw(0x0A7F);       /* RC up, set on thread 1 */
    SwitchToFiber(g_fpx->ret);
    g_fpx->mx_in[1] = t_get_mxcsr(); g_fpx->cw_in[1] = t_get_x87_cw();   /* now on thread 2 */
    for (;;) SwitchToFiber(g_fpx->ret);
}
static void w_fpx_run(void *p) {
    fpx_ctx *c = (fpx_ctx *)p;
    int k = c->step++;
    c->ret = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!c->ret) return;
    unsigned mx = k ? 0x3F80 : 0x7F80; unsigned short cw = k ? 0x067F : 0x0E7F;
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    SwitchToFiber(c->f);
    c->mx_t[k] = t_get_mxcsr(); c->cw_t[k] = t_get_x87_cw();
    ConvertFiberToThread();
}
TEST(fiber_fp_state_follows_fiber_across_threads) {
    fpx_ctx c; memset(&c, 0, sizeof c); g_fpx = &c;
    c.f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, fp_fpx, NULL);
    REQUIRE(c.f != NULL);
    RUN_WORKER(w_fpx_run, &c);
    RUN_WORKER(w_fpx_run, &c);
    DeleteFiber(c.f);
    wine_gap((c.mx_in[1] & MXC_CTRL) != 0x5F80 || (c.mx_t[0] & MXC_CTRL) != 0x7F80, "SwitchToFiber does not switch MXCSR / x87 control word per fiber");
    CHECKF((c.mx_in[1] & MXC_CTRL) == 0x5F80, "fiber MXCSR on thread 2: 0x%04x want 0x5F80", c.mx_in[1]);
    CHECKF((c.cw_in[1] & CW_CTRL) == (0x0A7F & CW_CTRL), "fiber x87 CW on thread 2: 0x%04x want 0x0A7F", c.cw_in[1]);
    CHECKF((c.mx_t[0] & MXC_CTRL) == 0x7F80, "thread 1 MXCSR after switch back 0x%04x", c.mx_t[0]);
    CHECKF((c.mx_t[1] & MXC_CTRL) == 0x3F80, "thread 2 MXCSR after switch back 0x%04x", c.mx_t[1]);
    CHECKF((c.cw_t[0] & CW_CTRL) == (0x0E7F & CW_CTRL), "thread 1 x87 CW after switch back 0x%04x", c.cw_t[0]);
    CHECKF((c.cw_t[1] & CW_CTRL) == (0x067F & CW_CTRL), "thread 2 x87 CW after switch back 0x%04x", c.cw_t[1]);
}

/* blocking calls inside a fiber */
static void blk_sleep(void *p) { (void)p; Sleep(2); }
static void blk_yield(void *p) { (void)p; SwitchToThread(); Sleep(0); }
static void blk_wait(void *p) { WaitForSingleObject((HANDLE)p, 3); }
typedef struct blk_ctx { void *main, *f; HANDLE ev; long bad_nv, bad_mx, bad_cw, n; t_vregs *in, *out; } blk_ctx;
static blk_ctx *g_blk;
static void WINAPI fp_blk(void *p) {
    (void)p;
    blk_ctx *b = g_blk;
    t_set_mxcsr(0xDF80); t_set_x87_cw(0x0B7F);
    for (int i = 0; i < 30; i++) {
        void (*fn)(void *) = (i % 3 == 0) ? blk_sleep : (i % 3 == 1) ? blk_yield : blk_wait;
        t_vregs_call(T_VK_XMM, b->in, b->out, fn, b->ev);
#ifdef T_X64
        if (!vr_eq(b->out, b->in, 6, 15, 0, 16)) b->bad_nv++;
#endif
        if ((t_get_mxcsr() & MXC_CTRL) != 0xDF80) b->bad_mx++;
        if ((t_get_x87_cw() & CW_CTRL) != (0x0B7F & CW_CTRL)) b->bad_cw++;
        b->n++;
    }
    for (;;) SwitchToFiber(b->main);
}
static void w_blk(void *p) {
    blk_ctx *b = (blk_ctx *)p; g_blk = b;
    b->main = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!b->main) return;
    b->f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, fp_blk, NULL);
    if (b->f) { SwitchToFiber(b->f); DeleteFiber(b->f); }
    ConvertFiberToThread();
}
TEST(fiber_blocking_calls_inside_fiber) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    blk_ctx b; memset(&b, 0, sizeof b);
    b.ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    b.in = vr_alloc(); b.out = vr_alloc();
    REQUIRE(b.ev && b.in && b.out);
    t_vregs_pattern(b.in, T_VK_XMM, 0xB10C);
    RUN_WORKER(w_blk, &b);
    CloseHandle(b.ev); vr_free(b.in); vr_free(b.out);
    REQUIRE(b.main && b.f);
    CHECK_EQ(b.n, 30);
    CHECKF(b.bad_nv == 0, "xmm6-15 changed across blocking calls in a fiber %ld times", b.bad_nv);
    CHECKF(b.bad_mx == 0, "fiber MXCSR changed across blocking calls %ld times", b.bad_mx);
    CHECKF(b.bad_cw == 0, "fiber x87 CW changed across blocking calls %ld times", b.bad_cw);
}

/* ======================================================================== */
/*                      thread context of a running fiber                   */
/* ======================================================================== */
typedef struct ctx_fb {
    void *main, *f; HANDLE ready; volatile long stop; t_vregs *in, *out; uintptr_t lo, hi; unsigned mx; int done;
} ctx_fb;
static ctx_fb *g_cf;
static void WINAPI fp_ctx(void *p) {
    (void)p;
    ctx_fb *c = g_cf;
    stack_range(&c->lo, &c->hi);
    t_set_mxcsr(c->mx);
    SetEvent(c->ready);
    t_vregs_spin(T_VK_XMM, c->in, c->out, &c->stop, 0x7FFFFFFFul);
    c->done = 1;
    for (;;) SwitchToFiber(c->main);
}
static DWORD WINAPI th_ctx(void *p) {
    ctx_fb *c = (ctx_fb *)p;
    c->main = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!c->main) { SetEvent(c->ready); return 1; }
    c->f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, fp_ctx, NULL);
    if (!c->f) { SetEvent(c->ready); ConvertFiberToThread(); return 2; }
    SwitchToFiber(c->f);
    DeleteFiber(c->f);
    ConvertFiberToThread();
    return 0;
}
static int ctx_xmm_get(const CONTEXT *x, int i, uint8_t out16[16]) {
#ifdef T_X64
    memcpy(out16, &x->FltSave.XmmRegisters[i], 16); return 1;
#else
    memcpy(out16, x->ExtendedRegisters + 160 + 16 * i, 16); return 1;
#endif
}
static void ctx_xmm_set(CONTEXT *x, int i, const uint8_t in16[16]) {
#ifdef T_X64
    memcpy(&x->FltSave.XmmRegisters[i], in16, 16);
#else
    memcpy(x->ExtendedRegisters + 160 + 16 * i, in16, 16);
#endif
}
static void ctx_case(int do_set) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static ctx_fb c; memset(&c, 0, sizeof c); g_cf = &c;
    static CONTEXT x __attribute__((aligned(16)));
    c.mx = 0x3F80 | 0x8000;   /* RC down + FTZ */
    c.ready = CreateEventA(NULL, TRUE, FALSE, NULL);
    c.in = vr_alloc(); c.out = vr_alloc();
    REQUIRE(c.ready && c.in && c.out);
    t_vregs_pattern(c.in, T_VK_XMM, 0xC7C7);
    HANDLE h = CreateThread(NULL, 0, th_ctx, &c, 0, NULL);
    if (!h) SKIP("CreateThread failed");
    WaitForSingleObject(c.ready, 10000);
    int found = 0, n = t_vregs_count(T_VK_XMM), tries = 0;
    uint8_t newv[16];
    t_pattern_reg(newv, 16, 0xC8C8, 7);
    for (; tries < 400 && c.f && !found; tries++) {
        if (SuspendThread(h) == (DWORD)-1) break;
        memset(&x, 0, sizeof x);
#ifdef T_X64
        x.ContextFlags = CONTEXT_FULL | CONTEXT_FLOATING_POINT;
#else
        x.ContextFlags = CONTEXT_FULL | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS;
#endif
        if (GetThreadContext(h, &x)) {
            int ok = 1;
            for (int i = 0; i < n && ok; i++) { uint8_t v[16]; ctx_xmm_get(&x, i, v); if (memcmp(v, c.in->v[i], 16)) ok = 0; }
            if (ok) {
                found = 1;
#ifdef T_X64
                uintptr_t sp = (uintptr_t)x.Rsp; unsigned cmx = x.FltSave.MxCsr;
                t_info("context_MxCsr_field", "0x%04x", (unsigned)x.MxCsr);
#else
                uintptr_t sp = (uintptr_t)x.Esp; uint32_t cmx; memcpy(&cmx, x.ExtendedRegisters + 24, 4);
#endif
                CHECKF(sp >= c.lo && sp < c.hi, "context stack pointer %p outside the running fiber's stack [%p,%p)",
                       (void *)sp, (void *)c.lo, (void *)c.hi);
                CHECKF((cmx & MXC_CTRL) == (c.mx & MXC_CTRL), "context MXCSR 0x%04x, fiber set 0x%04x", cmx, c.mx);
                if (do_set) {
                    ctx_xmm_set(&x, 7, newv);
#ifdef T_X64
                    x.ContextFlags = CONTEXT_FLOATING_POINT;
#else
                    x.ContextFlags = CONTEXT_EXTENDED_REGISTERS;
#endif
                    CHECKF(SetThreadContext(h, &x), "SetThreadContext failed: %lu", (unsigned long)GetLastError());
                }
            }
        }
        ResumeThread(h);
        if (!found) Sleep(5);
    }
    InterlockedExchange(&c.stop, 1);
    DWORD w = WaitForSingleObject(h, 10000);
    CHECK(w == WAIT_OBJECT_0);
    if (w != WAIT_OBJECT_0) TerminateThread(h, 1);
    CloseHandle(h); CloseHandle(c.ready);
    CHECKF(found, "GetThreadContext never showed the fiber's XMM pattern (%d tries)", tries);
    CHECK_EQ(c.done, 1);
    if (found) {
        for (int i = 0; i < n; i++) {
            char w2[64]; snprintf(w2, sizeof w2, "xmm%d after context %s", i, do_set ? "set" : "get");
            if (do_set && i == 7) CHECK_REG(w2, c.out->v[7], newv, 16);
            else CHECK_REG(w2, c.out->v[i], c.in->v[i], 16);
        }
    }
    vr_free(c.in); vr_free(c.out);
}
TEST(fiber_get_thread_context_sees_fiber) { ctx_case(0); }
TEST(fiber_set_thread_context_modifies_fiber) { ctx_case(1); }

/* ======================================================================== */
/*                        exceptions inside a fiber                         */
/* ======================================================================== */
#define EXC_FIBER 0xE0F1BE01u
static volatile uintptr_t g_exc_sp;
static volatile long g_exc_hits;
static LONG CALLBACK veh_fiber(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXC_FIBER) return EXCEPTION_CONTINUE_SEARCH;
#ifdef T_X64
    g_exc_sp = (uintptr_t)ep->ContextRecord->Rsp;
#else
    g_exc_sp = (uintptr_t)ep->ContextRecord->Esp;
#endif
    InterlockedIncrement(&g_exc_hits);
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void raise_fiber(void *p) { (void)p; RaiseException(EXC_FIBER, 0, 0, NULL); }
typedef struct exf_ctx { void *main, *f; uintptr_t lo, hi, sp; int cur_ok, bad_nv, mx_ok; t_vregs *in, *out; } exf_ctx;
static exf_ctx *g_exf;
static void WINAPI fp_exf(void *p) {
    (void)p;
    exf_ctx *e = g_exf;
    stack_range(&e->lo, &e->hi);
    t_set_mxcsr(0x5F80);
    t_vregs_call(T_VK_XMM, e->in, e->out, raise_fiber, NULL);
    e->sp = g_exc_sp;
#ifdef T_X64
    e->bad_nv = !vr_eq(e->out, e->in, 6, 15, 0, 16);
#endif
    e->cur_ok = GetCurrentFiber() == e->f;
    e->mx_ok = (t_get_mxcsr() & MXC_CTRL) == 0x5F80;
    for (;;) SwitchToFiber(e->main);
}
static void w_exf(void *p) {
    exf_ctx *e = (exf_ctx *)p; g_exf = e;
    e->main = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!e->main) return;
    e->f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, fp_exf, NULL);
    if (e->f) { SwitchToFiber(e->f); DeleteFiber(e->f); }
    ConvertFiberToThread();
}
TEST(fiber_exception_veh_in_fiber) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    exf_ctx e; memset(&e, 0, sizeof e);
    e.in = vr_alloc(); e.out = vr_alloc(); REQUIRE(e.in && e.out);
    t_vregs_pattern(e.in, T_VK_XMM, 0xE7E7);
    g_exc_hits = 0; g_exc_sp = 0;
    void *veh = AddVectoredExceptionHandler(1, veh_fiber);
    REQUIRE(veh != NULL);
    RUN_WORKER(w_exf, &e);
    RemoveVectoredExceptionHandler(veh);
    vr_free(e.in); vr_free(e.out);
    REQUIRE(e.main && e.f);
    CHECK_EQ(g_exc_hits, 1);
    CHECKF(e.sp >= e.lo && e.sp < e.hi, "exception context SP %p outside the fiber stack [%p,%p)", (void *)e.sp, (void *)e.lo, (void *)e.hi);
    CHECK(e.cur_ok);
    CHECK(e.mx_ok);
#ifdef T_X64
    CHECKF(!e.bad_nv, "xmm6-15 changed by an exception handled inside a fiber");
#endif
}

#ifdef T_X86
static uintptr_t seh_head(void) { uintptr_t h; __asm__ volatile("movl %%fs:0, %0" : "=r"(h)); return h; }
typedef struct seh_ctx { void *main, *f; uintptr_t head_f, lo_f, hi_f, head_main_before, head_main_after; } seh_ctx;
static seh_ctx *g_seh;
static void WINAPI fp_seh(void *p) {
    (void)p;
    NT_TIB *tib = (NT_TIB *)NtCurrentTeb();
    g_seh->head_f = seh_head(); g_seh->lo_f = (uintptr_t)tib->StackLimit; g_seh->hi_f = (uintptr_t)tib->StackBase;
    for (;;) SwitchToFiber(g_seh->main);
}
static void w_seh(void *p) {
    seh_ctx *s = (seh_ctx *)p; g_seh = s;
    s->main = ConvertThreadToFiber(NULL);
    if (!s->main) return;
    s->f = CreateFiber(0, fp_seh, NULL);
    s->head_main_before = seh_head();
    if (s->f) { SwitchToFiber(s->f); s->head_main_after = seh_head(); DeleteFiber(s->f); }
    ConvertFiberToThread();
}
#endif
TEST(fiber_x86_seh_chain_per_fiber) {
#ifdef T_X64
    SKIP("x86 only (x64 uses table-based unwinding, no fs:[0] chain)");
#else
    /* RtlDispatchException only accepts registration records on the current stack, so the chain head
     * of a fiber must be on the fiber's stack (or the end marker), and the primary fiber's chain must
     * come back unchanged. */
    seh_ctx s; memset(&s, 0, sizeof s);
    RUN_WORKER(w_seh, &s);
    REQUIRE(s.main && s.f);
    CHECKF(s.head_f == (uintptr_t)-1 || (s.head_f >= s.lo_f && s.head_f < s.hi_f),
           "fiber fs:[0] = %p, fiber stack [%p,%p)", (void *)s.head_f, (void *)s.lo_f, (void *)s.hi_f);
    CHECKF(s.head_main_after == s.head_main_before, "primary fiber fs:[0] %p after switch, was %p",
           (void *)s.head_main_after, (void *)s.head_main_before);
#endif
}

/* ======================================================================== */
/*                               stacks                                     */
/* ======================================================================== */
typedef struct stk_ctx {
    void *main, *f[8]; int n; uintptr_t lo[9], hi[9], local[9], tib_base[9], tib_limit[9]; uintptr_t main_after_lo, main_after_hi;
    SIZE_T commit, reserve; DWORD err;
} stk_ctx;
static stk_ctx *g_stk;
static void stk_record(int i) {
    volatile int here = i;
    NT_TIB *tib = (NT_TIB *)NtCurrentTeb();
    stack_range(&g_stk->lo[i], &g_stk->hi[i]);
    g_stk->local[i] = (uintptr_t)&here;
    g_stk->tib_base[i] = (uintptr_t)tib->StackBase; g_stk->tib_limit[i] = (uintptr_t)tib->StackLimit;
}
static void WINAPI fp_stk(void *p) { int i = (int)(uintptr_t)p; stk_record(i); for (;;) SwitchToFiber(g_stk->main); }
static void w_stk(void *p) {
    stk_ctx *s = (stk_ctx *)p; g_stk = s;
    s->main = ConvertThreadToFiber(NULL);
    if (!s->main) return;
    stk_record(8);
    for (int i = 0; i < s->n; i++) {
        s->f[i] = CreateFiberEx(s->commit, s->reserve, 0, fp_stk, (void *)(uintptr_t)i);
        if (!s->f[i]) { s->err = GetLastError(); break; }
        SwitchToFiber(s->f[i]);
    }
    stack_range(&s->main_after_lo, &s->main_after_hi);
    for (int i = 0; i < s->n; i++) if (s->f[i]) DeleteFiber(s->f[i]);
    ConvertFiberToThread();
}
TEST(fiber_stacks_distinct_and_restored) {
    stk_ctx s; memset(&s, 0, sizeof s); s.n = 8; s.reserve = 256 * 1024;
    RUN_WORKER(w_stk, &s);
    REQUIRE(s.main != NULL);
    for (int i = 0; i < 8; i++) REQUIREF(s.f[i] != NULL, "CreateFiberEx #%d failed (%lu)", i, (unsigned long)s.err);
    for (int i = 0; i <= 8; i++) {
        CHECKF(s.local[i] >= s.lo[i] && s.local[i] < s.hi[i], "fiber %d: local %p outside its stack [%p,%p)", i, (void *)s.local[i], (void *)s.lo[i], (void *)s.hi[i]);
        CHECKF(s.local[i] >= s.tib_limit[i] && s.local[i] < s.tib_base[i], "fiber %d: local %p outside NT_TIB [StackLimit %p, StackBase %p)",
               i, (void *)s.local[i], (void *)s.tib_limit[i], (void *)s.tib_base[i]);
        CHECKF(s.tib_base[i] == s.hi[i], "fiber %d: NT_TIB StackBase %p != stack limits high %p", i, (void *)s.tib_base[i], (void *)s.hi[i]);
        for (int j = i + 1; j <= 8; j++)
            CHECKF(s.hi[i] <= s.lo[j] || s.hi[j] <= s.lo[i], "stacks of %d [%p,%p) and %d [%p,%p) overlap",
                   i, (void *)s.lo[i], (void *)s.hi[i], j, (void *)s.lo[j], (void *)s.hi[j]);
    }
    CHECKF(s.main_after_lo == s.lo[8] && s.main_after_hi == s.hi[8], "primary stack limits not restored: [%p,%p) was [%p,%p)",
           (void *)s.main_after_lo, (void *)s.main_after_hi, (void *)s.lo[8], (void *)s.hi[8]);
}

static SIZE_T page_size(void) { SYSTEM_INFO si; GetSystemInfo(&si); return si.dwPageSize; }
TEST(fiber_stack_reserve_honored) {
    if (!T_PROC("kernel32.dll", GetCurrentThreadStackLimits)) SKIP("GetCurrentThreadStackLimits not exported");
    static const SIZE_T res[] = { 64 * 1024, 256 * 1024, 1024 * 1024 + 1, 3 * 1024 * 1024 };
    for (int k = 0; k < (int)T_COUNTOF(res); k++) {
        stk_ctx s; memset(&s, 0, sizeof s); s.n = 1; s.commit = 16 * 1024; s.reserve = res[k];
        RUN_WORKER(w_stk, &s);
        REQUIRE(s.main != NULL);
        CHECKF(s.f[0] != NULL, "CreateFiberEx(commit 16K, reserve %lu) failed: %lu", (unsigned long)res[k], (unsigned long)s.err);
        if (!s.f[0]) continue;
        SIZE_T got = s.hi[0] - s.lo[0];
        CHECKF(got >= res[k], "reserve %lu requested, stack range is %lu bytes", (unsigned long)res[k], (unsigned long)got);
        CHECKF(got % page_size() == 0, "stack reservation %lu not a page multiple", (unsigned long)got);
        char key[32]; snprintf(key, sizeof key, "reserve_%lu", (unsigned long)res[k]);
        t_info(key, "%lu", (unsigned long)got);
    }
}
TEST(fiber_stack_default_reserve) {
    if (!T_PROC("kernel32.dll", GetCurrentThreadStackLimits)) SKIP("GetCurrentThreadStackLimits not exported");
    /* CreateFiberEx doc: dwStackReserveSize 0 -> the default reserve size of the executable */
    const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)GetModuleHandleA(NULL);
    const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const uint8_t *)dos + dos->e_lfanew);
    SIZE_T img = (SIZE_T)nt->OptionalHeader.SizeOfStackReserve;
    stk_ctx s; memset(&s, 0, sizeof s); s.n = 1;
    RUN_WORKER(w_stk, &s);
    REQUIRE(s.main && s.f[0]);
    SIZE_T got = s.hi[0] - s.lo[0];
    t_info("image_stack_reserve", "%lu", (unsigned long)img);
    t_info("fiber_stack_reserve", "%lu", (unsigned long)got);
    CHECKF(got >= img, "default fiber reserve %lu < image SizeOfStackReserve %lu", (unsigned long)got, (unsigned long)img);
}

static volatile long g_deep_limit = 1 << 30;
static T_NOINLINE int deep(int d, int max) {
    volatile char buf[2048];
    buf[0] = (char)d; buf[2047] = (char)d;
    if (d >= max || d >= g_deep_limit) return buf[0];
    return deep(d + 1, max) + buf[2047] - (char)d;
}
typedef struct deep_ctx { void *main, *f; int result, max; } deep_ctx;
static deep_ctx *g_deep;
static void WINAPI fp_deep(void *p) { (void)p; g_deep->result = deep(0, g_deep->max); for (;;) SwitchToFiber(g_deep->main); }
static void w_deep(void *p) {
    deep_ctx *d = (deep_ctx *)p; g_deep = d;
    d->main = ConvertThreadToFiber(NULL);
    if (!d->main) return;
    d->f = CreateFiberEx(8192, 1024 * 1024, FIBER_FLAG_FLOAT_SWITCH, fp_deep, NULL);
    if (d->f) { SwitchToFiber(d->f); DeleteFiber(d->f); }
    ConvertFiberToThread();
}
TEST(fiber_stack_grows_on_demand) {
    /* 1 MB reserve, 8 KB commit: ~600 KB of recursion must commit pages through the guard page */
    deep_ctx d; memset(&d, 0, sizeof d); d.max = 256; d.result = -1;
    RUN_WORKER(w_deep, &d);
    REQUIRE(d.main && d.f);
    CHECK_EQ(d.result, 0);   /* deep() returns (char)max + sum of zeros */
}

/* stack overflow in a fiber: child process */
static uintptr_t g_ovf_lo, g_ovf_hi;
static LONG CALLBACK veh_ovf(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != STATUS_STACK_OVERFLOW) return EXCEPTION_CONTINUE_SEARCH;
#ifdef T_X64
    uintptr_t sp = (uintptr_t)ep->ContextRecord->Rsp;
#else
    uintptr_t sp = (uintptr_t)ep->ContextRecord->Esp;
#endif
    char buf[160];
    int n = snprintf(buf, sizeof buf, "OVF_SP_IN_FIBER=%d\n", sp >= g_ovf_lo && sp < g_ovf_hi);
    DWORD wr; WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buf, (DWORD)n, &wr, NULL);
    TerminateProcess(GetCurrentProcess(), STATUS_STACK_OVERFLOW);
    return EXCEPTION_CONTINUE_SEARCH;
}
static T_NOINLINE int ovf(int d) {
    volatile char buf[4096];
    buf[0] = (char)d; buf[4095] = (char)d;
    if (d >= g_deep_limit) return 0;
    return ovf(d + 1) + buf[4095];
}
static void WINAPI fp_ovf(void *p) {
    (void)p;
    ULONG g = 32 * 1024;
    printf("GUARANTEE_OK=%d\n", SetThreadStackGuarantee(&g) ? 1 : 0);
    stack_range(&g_ovf_lo, &g_ovf_hi);
    printf("FIBER_STACK=%p-%p\n", (void *)g_ovf_lo, (void *)g_ovf_hi);
    fflush(stdout);
    ovf(0);
    printf("NOT_REACHED\n");
    ExitProcess(3);
}
CHILD(fiber_overflow) {
    (void)argc; (void)argv;
    AddVectoredExceptionHandler(1, veh_ovf);
    void *m = ConvertThreadToFiber(NULL);
    if (!m) return 10;
    void *f = CreateFiberEx(0, 256 * 1024, 0, fp_ovf, NULL);
    if (!f) return 11;
    SwitchToFiber(f);
    return 12;
}
TEST(fiber_stack_overflow_child) {
    static t_child_result r;
    REQUIRE(t_child_run("fiber_overflow", NULL, 20000, &r));
    CHECK(!r.timed_out);
    t_info("child_exit", "0x%lx", (unsigned long)r.exit_code);
    CHECKF(r.exit_code == STATUS_STACK_OVERFLOW || r.exc_code == STATUS_STACK_OVERFLOW,
           "stack overflow in a fiber: exit 0x%lx exc 0x%lx (want STATUS_STACK_OVERFLOW)", (unsigned long)r.exit_code, (unsigned long)r.exc_code);
    CHECK(strstr(r.out, "NOT_REACHED") == NULL);
    char v[32];
    if (t_child_find(&r, "OVF_SP_IN_FIBER=", v, sizeof v)) CHECKF(!strcmp(v, "1"), "overflow context SP not on the fiber's stack");
    else t_note("overflow handler output missing (process terminated before the VEH ran)");
}

/* SetThreadStackGuarantee is per fiber */
typedef struct sg_ctx { void *main, *b; ULONG a0, a1, b0, b_prev, b1, b2, a2; int set_ok; ULONG b_other_thread; } sg_ctx;
static sg_ctx *g_sg;
static ULONG sg_query(void) { ULONG v = 0; SetThreadStackGuarantee(&v); return v; }
static void WINAPI fp_sg(void *p) {
    (void)p;
    g_sg->b0 = sg_query();
    ULONG v = 48 * 1024;
    g_sg->set_ok = SetThreadStackGuarantee(&v) != 0;
    g_sg->b_prev = v;
    g_sg->b1 = sg_query();
    SwitchToFiber(g_sg->main);
    g_sg->b2 = sg_query();
    SwitchToFiber(g_sg->main);
    g_sg->b_other_thread = sg_query();
    for (;;) SwitchToFiber(g_sg->main);
}
static void w_sg(void *p) {
    sg_ctx *s = (sg_ctx *)p; g_sg = s;
    s->main = ConvertThreadToFiber(NULL);
    if (!s->main) return;
    s->a0 = sg_query();
    s->b = CreateFiberEx(0, 1024 * 1024, 0, fp_sg, NULL);
    if (s->b) {
        SwitchToFiber(s->b);
        s->a1 = sg_query();
        SwitchToFiber(s->b);
        s->a2 = sg_query();
    }
    ConvertFiberToThread();
}
static void w_sg2(void *p) {
    sg_ctx *s = (sg_ctx *)p;
    s->main = ConvertThreadToFiber(NULL);
    if (!s->main) return;
    SwitchToFiber(s->b);
    ConvertFiberToThread();
}
TEST(fiber_stack_guarantee_per_fiber) {
    sg_ctx s; memset(&s, 0, sizeof s);
    RUN_WORKER(w_sg, &s);
    REQUIRE(s.main && s.b);
    RUN_WORKER(w_sg2, &s);   /* run B on another thread */
    DeleteFiber(s.b);
    t_info("guarantee_A", "%lu", (unsigned long)s.a0);
    t_info("guarantee_B_initial", "%lu", (unsigned long)s.b0);
    t_info("guarantee_B_after_set", "%lu", (unsigned long)s.b1);
    CHECK(s.set_ok);
    wine_gap(s.a1 != s.a0 || s.a2 != s.a0 || s.b2 != s.b1 || s.b_other_thread != s.b1, "SetThreadStackGuarantee value is not per fiber");
    CHECKF(s.b1 >= 48 * 1024, "fiber B guarantee %lu after setting 48K", (unsigned long)s.b1);
    CHECKF(s.a1 == s.a0, "fiber A guarantee changed from %lu to %lu by fiber B", (unsigned long)s.a0, (unsigned long)s.a1);
    CHECKF(s.a2 == s.a0, "fiber A guarantee %lu after second switch, was %lu", (unsigned long)s.a2, (unsigned long)s.a0);
    CHECKF(s.b2 == s.b1, "fiber B guarantee %lu after switching away and back, was %lu", (unsigned long)s.b2, (unsigned long)s.b1);
    CHECKF(s.b_other_thread == s.b1, "fiber B guarantee on another thread %lu, was %lu", (unsigned long)s.b_other_thread, (unsigned long)s.b1);
}

/* fiber start routine is entered like any function: x64 ABI requires RSP+8 16-byte aligned at entry */
static void w_align(void *p) {
    int *ok = (int *)p;
    g_entry_ret = ConvertThreadToFiber(NULL);
    if (!g_entry_ret) return;
    for (int i = 0; i < 4; i++) {
        void *f = CreateFiberEx(0, (SIZE_T)(64 * 1024) << i, i & 1, fx_entry_probe, NULL);
        if (!f) continue;
        fx_entry_sp = 0;
        SwitchToFiber(f);
        ok[i] = 1; ok[4 + i] = (int)(fx_entry_sp & 0xF); ok[8 + i] = fx_entry_sp != 0;
        DeleteFiber(f);
    }
    ConvertFiberToThread();
}
TEST(fiber_stack_alignment_at_entry) {
    int ok[12]; memset(ok, 0, sizeof ok);
    g_entry_ran = 0;
    RUN_WORKER(w_align, ok);
    REQUIRE(g_entry_ret != NULL);
    CHECK_EQ(g_entry_ran, 4);
    for (int i = 0; i < 4; i++) {
        CHECKF(ok[i] && ok[8 + i], "fiber %d did not run", i);
#ifdef T_X64
        CHECKF(ok[4 + i] == 8, "fiber %d: RSP mod 16 at start routine entry = %d (want 8)", i, ok[4 + i]);
#else
        CHECKF((ok[4 + i] & 3) == 0, "fiber %d: ESP mod 4 at start routine entry = %d", i, ok[4 + i] & 3);
        t_info("esp_mod16_at_entry", "%d", ok[4 + i]);
#endif
    }
}

static void w_huge(void *p) {
    basic_res *b = (basic_res *)p;
#ifdef T_X64
    SIZE_T huge = (SIZE_T)1 << 50;              /* 1 PB > any user address space */
#else
    SIZE_T huge = (SIZE_T)0xFFF00000u;          /* ~4 GB > any 32-bit user address space */
#endif
    SetLastError(0);
    b->p[0] = CreateFiberEx(0, huge, 0, fp_never, NULL); b->e[0] = GetLastError();
    if (b->p[0]) DeleteFiber(b->p[0]);
    SetLastError(0);
    b->p[1] = CreateFiberEx(huge, 0, 0, fp_never, NULL); b->e[1] = GetLastError();
    if (b->p[1]) DeleteFiber(b->p[1]);
    b->p[2] = CreateFiber(0, fp_never, NULL);  /* still usable */
    if (b->p[2]) DeleteFiber(b->p[2]);
}
TEST(fiber_create_huge_stack_fails) {
    basic_res b; memset(&b, 0, sizeof b);
    RUN_WORKER(w_huge, &b);
    CHECKF(b.p[0] == NULL, "CreateFiberEx with an impossible reserve succeeded");
    CHECKF(b.e[0] != 0, "no error code for the impossible reserve");
    CHECKF(b.p[1] == NULL, "CreateFiberEx with an impossible commit succeeded");
    CHECKF(b.e[1] != 0, "no error code for the impossible commit");
    t_info("reserve_error", "%lu", (unsigned long)b.e[0]);
    t_info("commit_error", "%lu", (unsigned long)b.e[1]);
    CHECK(b.p[2] != NULL);
}

/* ======================================================================== */
/*                        fiber local storage (FLS)                         */
/* ======================================================================== */
#define FLS_LOG 64
static volatile long g_fls_n;
static void *volatile g_fls_vals[FLS_LOG];
static DWORD volatile g_fls_tids[FLS_LOG];
static void NTAPI fls_cb(void *v) {
    long i = InterlockedIncrement(&g_fls_n) - 1;
    if (i < FLS_LOG) { g_fls_vals[i] = v; g_fls_tids[i] = GetCurrentThreadId(); }
}
static void fls_reset(void) { g_fls_n = 0; for (int i = 0; i < FLS_LOG; i++) { g_fls_vals[i] = NULL; g_fls_tids[i] = 0; } }
static int fls_seen(void *v) { int c = 0; for (int i = 0; i < g_fls_n && i < FLS_LOG; i++) if (g_fls_vals[i] == v) c++; return c; }

TEST(fls_alloc_set_get_free) {
    DWORD i = FlsAlloc(NULL);
    REQUIREF(i != FLS_OUT_OF_INDEXES, "FlsAlloc failed: %lu", (unsigned long)GetLastError());
    CHECK(FlsGetValue(i) == NULL);            /* "an FLS index initialized to zero" */
    CHECK(FlsSetValue(i, (void *)0x1234));
    CHECK(FlsGetValue(i) == (void *)0x1234);
    CHECK(FlsSetValue(i, NULL));
    CHECK(FlsGetValue(i) == NULL);
    CHECK(FlsFree(i));
    DWORD j[32]; int n = 0;
    for (int k = 0; k < 32; k++) { j[k] = FlsAlloc(fls_cb); if (j[k] != FLS_OUT_OF_INDEXES) n++; }
    CHECK_EQ(n, 32);
    for (int a = 0; a < 32; a++) for (int b = a + 1; b < 32; b++) if (j[a] != FLS_OUT_OF_INDEXES) CHECK(j[a] != j[b]);
    for (int k = 0; k < 32; k++) if (j[k] != FLS_OUT_OF_INDEXES) CHECK(FlsFree(j[k]));
}

TEST(fls_invalid_index) {
    /* RtlFlsGetValue/SetValue/Free: index 0 or >= 0xFF0 -> STATUS_INVALID_PARAMETER (kernelbase maps to 87) */
    static const DWORD bad[] = { FLS_OUT_OF_INDEXES, 0x7FFFFFFFu, 0x10000u };
    for (int k = 0; k < (int)T_COUNTOF(bad); k++) {
        SetLastError(0);
        CHECK(FlsGetValue(bad[k]) == NULL);
        CHECKF(GetLastError() == ERROR_INVALID_PARAMETER, "FlsGetValue(0x%lx): error %lu", (unsigned long)bad[k], (unsigned long)GetLastError());
        SetLastError(0);
        CHECK(!FlsSetValue(bad[k], (void *)1));
        CHECKF(GetLastError() == ERROR_INVALID_PARAMETER, "FlsSetValue(0x%lx): error %lu", (unsigned long)bad[k], (unsigned long)GetLastError());
        SetLastError(0);
        CHECK(!FlsFree(bad[k]));
        CHECKF(GetLastError() == ERROR_INVALID_PARAMETER, "FlsFree(0x%lx): error %lu", (unsigned long)bad[k], (unsigned long)GetLastError());
    }
}

static void w_fls_plain(void *p) {
    basic_res *b = (basic_res *)p;
    DWORD i = (DWORD)b->r[0];
    b->p[0] = FlsGetValue(i);
    b->r[1] = FlsSetValue(i, (void *)(uintptr_t)GetCurrentThreadId());
    b->p[1] = FlsGetValue(i);
    b->r[2] = (int)GetCurrentThreadId();
}
TEST(fls_without_fibers_is_per_thread) {
    DWORD i = FlsAlloc(NULL);
    REQUIRE(i != FLS_OUT_OF_INDEXES);
    CHECK(FlsSetValue(i, (void *)0xCA5E));
    basic_res b; memset(&b, 0, sizeof b); b.r[0] = (int)i;
    RUN_WORKER(w_fls_plain, &b);
    CHECKF(b.p[0] == NULL, "new thread sees another thread's FLS value %p", b.p[0]);
    CHECK(b.r[1]);
    CHECK(b.p[1] == (void *)(uintptr_t)(DWORD)b.r[2]);
    CHECK(FlsGetValue(i) == (void *)0xCA5E);
    CHECK(FlsSetValue(i, NULL));
    FlsFree(i);
}

typedef struct flsf_ctx { void *main, *b; DWORD idx; void *b_initial, *b_seen, *a_seen; long bad_a, bad_b; int iters; } flsf_ctx;
static flsf_ctx *g_flf;
static void WINAPI fp_flsf(void *p) {
    (void)p;
    flsf_ctx *c = g_flf;
    c->b_initial = FlsGetValue(c->idx);
    FlsSetValue(c->idx, (void *)0xB0B0);
    for (;;) {
        SwitchToFiber(c->main);
        c->b_seen = FlsGetValue(c->idx);
        if (c->b_seen != (void *)0xB0B0) c->bad_b++;
    }
}
static void w_flsf(void *p) {
    flsf_ctx *c = (flsf_ctx *)p; g_flf = c;
    c->main = ConvertThreadToFiber(NULL);
    if (!c->main) return;
    FlsSetValue(c->idx, (void *)0xA0A0);
    c->b = CreateFiber(0, fp_flsf, NULL);
    if (c->b) {
        for (int i = 0; i < c->iters; i++) {
            SwitchToFiber(c->b);
            c->a_seen = FlsGetValue(c->idx);
            if (c->a_seen != (void *)0xA0A0) c->bad_a++;
        }
        DeleteFiber(c->b);
    }
    FlsSetValue(c->idx, NULL);
    ConvertFiberToThread();
}
TEST(fls_per_fiber_values) {
    flsf_ctx c; memset(&c, 0, sizeof c); c.iters = 1000;
    c.idx = FlsAlloc(NULL);
    REQUIRE(c.idx != FLS_OUT_OF_INDEXES);
    RUN_WORKER(w_flsf, &c);
    FlsFree(c.idx);
    REQUIRE(c.main && c.b);
    CHECKF(c.b_initial == NULL, "new fiber starts with FLS value %p (creator had 0xA0A0)", c.b_initial);
    CHECKF(c.bad_a == 0, "fiber A FLS value wrong %ld times (last %p)", c.bad_a, c.a_seen);
    CHECKF(c.bad_b == 0, "fiber B FLS value wrong %ld times (last %p)", c.bad_b, c.b_seen);
}

typedef struct flcb_ctx { void *main, *b; DWORD idx; int mode; long n_after_switch, n_after_delete; DWORD tid; } flcb_ctx;
static flcb_ctx *g_flcb;
static void WINAPI fp_flcb(void *p) {
    (void)p;
    flcb_ctx *c = g_flcb;
    if (c->mode != 2) FlsSetValue(c->idx, (void *)0xDE1E7E);
    for (;;) SwitchToFiber(c->main);
}
static void w_flcb(void *p) {
    flcb_ctx *c = (flcb_ctx *)p; g_flcb = c;
    c->tid = GetCurrentThreadId();
    c->main = ConvertThreadToFiber(NULL);
    if (!c->main) return;
    FlsSetValue(c->idx, (void *)0xAAAA);
    c->b = CreateFiber(0, fp_flcb, NULL);
    if (c->b) {
        SwitchToFiber(c->b);
        c->n_after_switch = g_fls_n;
        if (c->mode == 1) { if (FlsFree(c->idx)) c->idx = FLS_OUT_OF_INDEXES; c->n_after_delete = g_fls_n; }
        DeleteFiber(c->b);
        if (c->mode != 1) c->n_after_delete = g_fls_n;
    }
    if (c->idx != FLS_OUT_OF_INDEXES) FlsSetValue(c->idx, NULL);
    ConvertFiberToThread();
}
TEST(fls_callback_on_delete_fiber) {
    fls_reset();
    flcb_ctx c; memset(&c, 0, sizeof c); c.mode = 0;
    c.idx = FlsAlloc(fls_cb);
    REQUIRE(c.idx != FLS_OUT_OF_INDEXES);
    RUN_WORKER(w_flcb, &c);
    REQUIRE(c.main && c.b);
    CHECKF(c.n_after_switch == 0, "FLS callback ran %ld times before any deletion", c.n_after_switch);
    CHECKF(c.n_after_delete == 1, "DeleteFiber ran the FLS callback %ld times (want 1)", c.n_after_delete);
    CHECKF(g_fls_vals[0] == (void *)0xDE1E7E, "callback value %p want the deleted fiber's 0xDE1E7E", g_fls_vals[0]);
    CHECKF(g_fls_tids[0] == c.tid, "callback ran on thread %lu, deleting thread %lu", (unsigned long)g_fls_tids[0], (unsigned long)c.tid);
    CHECKF(fls_seen((void *)0xAAAA) == 0, "the surviving fiber's value was passed to the callback");
    FlsFree(c.idx);
}
TEST(fls_callback_not_called_for_null_slot) {
    fls_reset();
    flcb_ctx c; memset(&c, 0, sizeof c); c.mode = 2;
    c.idx = FlsAlloc(fls_cb);
    REQUIRE(c.idx != FLS_OUT_OF_INDEXES);
    RUN_WORKER(w_flcb, &c);
    REQUIRE(c.main && c.b);
    CHECKF(c.n_after_delete == 0, "callback ran %ld times for a fiber whose slot is NULL", c.n_after_delete);
    FlsFree(c.idx);
}
TEST(fls_callback_on_flsfree_all_fibers) {
    fls_reset();
    flcb_ctx c; memset(&c, 0, sizeof c); c.mode = 1;
    c.idx = FlsAlloc(fls_cb);
    REQUIRE(c.idx != FLS_OUT_OF_INDEXES);
    RUN_WORKER(w_flcb, &c);
    REQUIRE(c.main && c.b);
    CHECKF(c.idx == FLS_OUT_OF_INDEXES, "FlsFree failed");
    CHECKF(c.n_after_delete == 2, "FlsFree ran the callback %ld times (want 2: running fiber + suspended fiber)", c.n_after_delete);
    CHECKF(fls_seen((void *)0xAAAA) == 1, "running fiber's value not passed to the callback");
    CHECKF(fls_seen((void *)0xDE1E7E) == 1, "suspended fiber's value not passed to the callback");
    if (c.idx != FLS_OUT_OF_INDEXES) FlsFree(c.idx);
}

static void w_fls_exit(void *p) {
    DWORD i = *(DWORD *)p;
    int as_fiber = (int)((DWORD *)p)[1];
    ((DWORD *)p)[2] = GetCurrentThreadId();
    if (as_fiber) ConvertThreadToFiber(NULL);      /* exit while still a fiber */
    FlsSetValue(i, (void *)(uintptr_t)(0xE0E0 + as_fiber));
}
TEST(fls_callback_on_thread_exit) {
    for (int as_fiber = 0; as_fiber < 2; as_fiber++) {
        fls_reset();
        DWORD a[3] = { FlsAlloc(fls_cb), (DWORD)as_fiber, 0 };
        REQUIRE(a[0] != FLS_OUT_OF_INDEXES);
        RUN_WORKER(w_fls_exit, a);
        CHECKF(fls_seen((void *)(uintptr_t)(0xE0E0 + as_fiber)) == 1, "%s thread exit: FLS callback calls with its value = %d (total %ld)",
               as_fiber ? "fiber" : "plain", fls_seen((void *)(uintptr_t)(0xE0E0 + as_fiber)), g_fls_n);
        if (g_fls_n > 0) t_info(as_fiber ? "fiber_exit_cb_on_exiting_thread" : "plain_exit_cb_on_exiting_thread", "%s", g_fls_tids[0] == a[2] ? "yes" : "no");
        FlsFree(a[0]);
    }
}

typedef struct flsm_ctx { void *main, *f[4]; DWORD idx[64]; int n; long bad; } flsm_ctx;
static flsm_ctx *g_flm;
static void WINAPI fp_flsm(void *p) {
    int me = (int)(uintptr_t)p;
    flsm_ctx *c = g_flm;
    for (int k = 0; k < c->n; k++) FlsSetValue(c->idx[k], (void *)(uintptr_t)(((unsigned)me << 16) | (unsigned)k | 0x1000000u));
    for (;;) {
        SwitchToFiber(c->main);
        for (int k = 0; k < c->n; k++) if (FlsGetValue(c->idx[k]) != (void *)(uintptr_t)(((unsigned)me << 16) | (unsigned)k | 0x1000000u)) c->bad++;
    }
}
static void w_flsm(void *p) {
    flsm_ctx *c = (flsm_ctx *)p; g_flm = c;
    c->main = ConvertThreadToFiber(NULL);
    if (!c->main) return;
    for (int i = 0; i < 4; i++) c->f[i] = CreateFiber(0, fp_flsm, (void *)(uintptr_t)i);
    for (int r = 0; r < 50; r++) for (int i = 0; i < 4; i++) if (c->f[i]) SwitchToFiber(c->f[i]);
    for (int k = 0; k < c->n; k++) if (FlsGetValue(c->idx[k]) != NULL) c->bad += 1000;
    for (int i = 0; i < 4; i++) if (c->f[i]) DeleteFiber(c->f[i]);
    ConvertFiberToThread();
}
TEST(fls_many_indexes_many_fibers) {
    flsm_ctx c; memset(&c, 0, sizeof c);
    for (int k = 0; k < 64; k++) { c.idx[k] = FlsAlloc(NULL); if (c.idx[k] == FLS_OUT_OF_INDEXES) break; c.n++; }
    if (c.n < 8) { for (int k = 0; k < c.n; k++) FlsFree(c.idx[k]); SKIP("only %d FLS indexes available", c.n); }
    RUN_WORKER(w_flsm, &c);
    for (int k = 0; k < c.n; k++) FlsFree(c.idx[k]);
    REQUIRE(c.main && c.f[0] && c.f[3]);
    CHECKF(c.bad == 0, "FLS values crossed between fibers %ld times (%d indexes)", c.bad, c.n);
}

/* ======================================================================== */
/*                     misc observations / undefined uses                   */
/* ======================================================================== */
typedef struct misc_ctx { void *main, *f; int frames_f; DWORD le_before, le_in_b, le_after; void *cur_nonfiber; int isf; } misc_ctx;
static misc_ctx *g_mc;
static void WINAPI fp_misc(void *p) {
    (void)p;
    void *fr[62];
    g_mc->frames_f = CaptureStackBackTrace(0, 62, fr, NULL);
    g_mc->le_in_b = GetLastError();
    SetLastError(0xB0B);
    for (;;) SwitchToFiber(g_mc->main);
}
static void w_misc(void *p) {
    misc_ctx *m = (misc_ctx *)p; g_mc = m;
    m->cur_nonfiber = (void *)((NT_TIB *)NtCurrentTeb())->FiberData;
    m->main = ConvertThreadToFiber(NULL);
    if (!m->main) return;
    m->f = CreateFiber(0, fp_misc, NULL);
    if (m->f) {
        SetLastError(0xA0A);
        SwitchToFiber(m->f);
        m->le_after = GetLastError();
        DeleteFiber(m->f);
    }
    ConvertFiberToThread();
}
TEST(fiber_backtrace_and_lasterror_info) {
    misc_ctx m; memset(&m, 0, sizeof m);
    RUN_WORKER(w_misc, &m);
    REQUIRE(m.main && m.f);
    CHECKF(m.frames_f >= 1, "CaptureStackBackTrace inside a fiber returned %d frames", m.frames_f);
    t_info("backtrace_frames_in_fiber", "%d", m.frames_f);
    t_info("lasterror_seen_by_new_fiber", "0x%lx (switcher set 0xA0A)", (unsigned long)m.le_in_b);
    t_info("lasterror_after_switch_back", "0x%lx (fiber set 0xB0B)", (unsigned long)m.le_after);
    t_info("nt_tib_fiberdata_non_fiber_thread", "%p", m.cur_nonfiber);
}

/* SwitchToFiber(GetCurrentFiber()): docs say "unpredictable"; observe in a child */
CHILD(fiber_switch_self) {
    (void)argc; (void)argv;
    void *m = ConvertThreadToFiber((void *)0x5E1F);
    if (!m) return 10;
    unsigned mx = 0x7F80; t_set_mxcsr(mx);
    SwitchToFiber(GetCurrentFiber());
    printf("SELF_RETURNED=1\nSELF_CUR_OK=%d\nSELF_DATA_OK=%d\nSELF_MXCSR_OK=%d\n", GetCurrentFiber() == m,
           GetFiberData() == (void *)0x5E1F, (t_get_mxcsr() & MXC_CTRL) == mx);
    ConvertFiberToThread();
    return 0;
}
TEST(fiber_switch_to_self_child_info) {
    static t_child_result r;
    REQUIRE(t_child_run("fiber_switch_self", NULL, 10000, &r));
    CHECK(!r.timed_out);
    char v[32];
    t_info("returned", "%s", t_child_find(&r, "SELF_RETURNED=", v, sizeof v) ? v : "0");
    if (t_child_find(&r, "SELF_CUR_OK=", v, sizeof v)) t_info("current_fiber_ok", "%s", v);
    t_info("exit_code", "0x%lx", (unsigned long)r.exit_code);
}

CHILD(fiber_switch_null) {
    (void)argc; (void)argv;
    if (!ConvertThreadToFiber(NULL)) return 10;
    fflush(stdout);
    SwitchToFiber(NULL);
    printf("NULL_RETURNED=1\n");
    return 0;
}
TEST(fiber_switch_null_child_info) {
    static t_child_result r;
    REQUIRE(t_child_run("fiber_switch_null", NULL, 10000, &r));
    CHECK(!r.timed_out);
    t_info("exit_code", "0x%lx", (unsigned long)r.exit_code);
    t_info("exception", "0x%lx", (unsigned long)r.exc_code);
    t_info("returned", "%s", strstr(r.out, "NULL_RETURNED=1") ? "yes" : "no");
}

static void WINAPI fp_child_ran(void *p) { (void)p; printf("NONFIBER_FIBER_RAN=1\n"); fflush(stdout); ExitProcess(0x55); }
CHILD(fiber_switch_from_thread) {
    (void)argc; (void)argv;
    void *f = CreateFiber(0, fp_child_ran, NULL);
    if (!f) return 10;
    fflush(stdout);
    SwitchToFiber(f);           /* the calling thread was never converted */
    printf("NONFIBER_RETURNED=1\n");
    return 0;
}
TEST(fiber_switch_from_non_fiber_thread_child_info) {
    static t_child_result r;
    REQUIRE(t_child_run("fiber_switch_from_thread", NULL, 10000, &r));
    CHECK(!r.timed_out);
    t_info("exit_code", "0x%lx", (unsigned long)r.exit_code);
    t_info("exception", "0x%lx", (unsigned long)r.exc_code);
    t_info("fiber_ran", "%s", strstr(r.out, "NONFIBER_FIBER_RAN=1") ? "yes" : "no");
}
