/* thread_xstate - per-thread SIMD / FP state isolation under the Windows scheduler.
 *
 * What the OS must guarantee (the contract under test, not the processor implementation):
 *   1. Every thread owns a private copy of the user-visible register state: XMM/YMM/ZMM, opmask k0-7,
 *      x87 data registers + control word, MMX, MXCSR. Whatever the scheduler does (preemption, blocking
 *      waits, APC delivery, migration between CPUs, SuspendThread, thread exit/terminate) a thread
 *      sees exactly its own values, and never a value another thread loaded ("leak").
 *      Checked with GPR-only spin loops (t_vregs_spin: nothing but preemption touches the registers,
 *      so ALL registers at FULL width must survive) on oversubscribed, pinned or migrating threads.
 *   2. Windows x64 ABI (docs "x64 calling convention" / "x64 software conventions"): XMM6-XMM15 low
 *      128 bits are nonvolatile across calls; XMM0-5, the upper YMM/ZMM bits, XMM16-31 and k0-7 are
 *      volatile. The control bits of MXCSR (15:6) and the x87 control word are nonvolatile; the MXCSR
 *      status flags (5:0) are volatile. The 32-bit ABI makes every XMM register volatile, but MXCSR
 *      control and the x87 CW are preserved there too. So around every blocking API call
 *      (Sleep, SwitchToThread, waits, SleepEx+APC, SRW/condvar, WaitOnAddress, pipes, barriers...)
 *      we hard-check only XMM6-15[127:0] (x64) and MXCSR control / x87 CW (both archs); volatile
 *      registers are only scanned for FOREIGN patterns (another thread's value is never legitimate).
 *   3. New threads start from the documented initial FP environment, independent of the creator:
 *      MXCSR = 0x1F80, x87 CW = 0x027F, empty x87 stack (tag word 0xFFFF; also required by the ABI at
 *      the call of the thread procedure). Checked in the Windows 10 19041 reference corpus:
 *      ntoskrnl!KiInitializeContextThread zero-fills the new thread's XSAVE area, then stores
 *      FCW = 0x27F and MXCSR = 0x1F80 (and XSTATE_BV = 1); ntdll!RtlInitializeContext (x64) also
 *      writes ControlWord 0x27F / MxCsr 0x1F80 with a zeroed FltSave. Register CONTENTS at thread
 *      entry are not hard-checked (loader code runs first) except that they must not hold another
 *      thread's pattern; zero-ness is recorded as INFO.
 *   4. Thread-pool callbacks (CreateThreadpoolWork, TrySubmitThreadpoolCallback, QueueUserWorkItem,
 *      pool timers / waits, RegisterWaitForSingleObject) run on pool threads whose FP environment is
 *      the initial one (every callback, ours included, restores what it changed: ABI), keep their own
 *      register state, and never disturb the submitting thread's nonvolatile state.
 *
 *   5. Observers and asynchronous delivery: GetThreadContext of a SUSPENDED spinning worker must show that
 *      worker's own XMM / x87 / MMX values, never another thread's (and the worker's registers survive
 *      the suspend/get/resume); the whole MXCSR, sticky status flags included, survives preemption;
 *      alertable kernel-object waits (WaitForSingleObjectEx) deliver APCs like SleepEx; Windows 11
 *      special user APCs (QueueUserAPC2) interrupt a NON-alertable spin and must restore its legacy
 *      state exactly (extended YMM/ZMM state across them is INFO only: not documented).
 *
 * Patterns: t_pattern_word(tag, reg, word) - every qword identifies (tag, reg, word), so a leaked value
 * is decoded in the failure notes ("pattern of worker 17 (phase 2)"). Tags change every iteration
 * (4 phases) so a stale restore of the thread's own older state is detected too.
 *
 * Sizes N = 2, 16, 64, 256, 1024 workers. A size SKIPs when CreateThread fails (resource limit).
 * Stack reservation per thread 128 KB (1024 threads = 128 MB of address space, also fine on x86).
 * Durations are bounded (0.25-0.7 s per run, THREAD_XSTATE_SCALE=<percent> env var scales them);
 * every wait has a timeout; iterations/s are reported as PERF.
 * x86: XMM ABI checks do not apply (all volatile) - spin (preemption) checks and leak scans still do. */
#include "t.h"
#include <stddef.h>

#define TX_STACK      (128u * 1024u)
#define MX_CTRL       0xFFC0u          /* MXCSR control bits: DAZ(6), masks(12:7), RC(14:13), FTZ(15) */
#define MX_DEFAULT    0x1F80u
#define CW_DEFAULT    0x027Fu
#define REPORT_MAX    6                /* detailed CHECK_REG reports per case (the rest is counted) */
#define JOIN_MS       30000u

/* ======================================================================================
 * small environment helpers
 * ==================================================================================== */
static int scale_pct(void) {
    static int s = -1;
    if (s < 0) {
        char b[32]; DWORD k = GetEnvironmentVariableA("THREAD_XSTATE_SCALE", b, sizeof b);
        s = (k && k < sizeof b) ? atoi(b) : 100;
        if (s < 10) s = 10;
        if (s > 2000) s = 2000;
    }
    return s;
}
static DWORD dur_ms(int n) {
    DWORD b = n <= 2 ? 250 : n <= 16 ? 300 : n <= 64 ? 400 : n <= 256 ? 500 : 700;
    return b * (DWORD)scale_pct() / 100u;
}
static int best_kind(void) {
    return cpu_has(F_AVX512F) ? T_VK_ZMM : cpu_has(F_AVX) ? T_VK_YMM : T_VK_XMM;
}
static const char *kind_name(int k) { return k == T_VK_ZMM ? "zmm" : k == T_VK_YMM ? "ymm" : "xmm"; }

/* DAZ is only legal in MXCSR when MXCSR_MASK (FXSAVE byte 28; 0 means 0xFFBF) has bit 6 (SDM 10.2.3.1). */
static int daz_ok(void) {
    static int v = -1;
    if (v < 0) {
        static __attribute__((aligned(16))) uint8_t a[512];
        memset(a, 0, sizeof a);
        t_fxsave(a);
        uint32_t m; memcpy(&m, a + 28, 4);
        if (!m) m = 0xFFBF;
        v = (m & 0x40) != 0;
    }
    return v;
}
/* Per-thread FP-environment variants: all exceptions masked (no faults), different RC/FTZ/DAZ. */
static unsigned mxcsr_variant(unsigned k) {
    unsigned v = 0x1F80u | ((k & 3u) << 13) | (((k >> 2) & 1u) << 15);
    if (daz_ok() && ((k >> 3) & 1u)) v |= 0x40u;
    return v;
}
/* x87 CW: masks 0x3F + reserved bit 6 (reads back as written on every implementation we know of: FNINIT
 * yields 0x37F), PC in {00,10,11} (01 reserved), RC any. */
static unsigned short cw_variant(unsigned k) {
    static const unsigned pc[3] = { 0, 2, 3 };
    return (unsigned short)(0x7Fu | (pc[k % 3] << 8) | (((k / 3) & 3u) << 10));
}

static int cpu_masks(DWORD_PTR *m, int cap) {
    DWORD_PTR pm = 0, sm = 0; int k = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return 0;
    for (int i = 0; i < (int)(sizeof(DWORD_PTR) * 8) && k < cap; i++)
        if (pm & ((DWORD_PTR)1 << i)) m[k++] = (DWORD_PTR)1 << i;
    return k;
}
static HANDLE mk_thread(LPTHREAD_START_ROUTINE fn, void *arg, DWORD flags, DWORD *tid) {
    DWORD t = 0;
    HANDLE h = CreateThread(NULL, TX_STACK, fn, arg, flags | STACK_SIZE_PARAM_IS_A_RESERVATION, &t);
    if (tid) *tid = t;
    return h;
}
static void *big_alloc(size_t n) { return VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }
static void big_free(void *p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }

/* ======================================================================================
 * tag domains: which pattern tags a thread may legitimately hold; decoding of leaked values
 * ==================================================================================== */
typedef struct vdom {
    uint32_t max_tag;
    uint32_t eng_n;                     /* engine layout: worker i phase p = 1+i+n*p (p<4); APC of i = 1+4n+i; main = 5n+1 */
    uint32_t hold_base, hold_n;         /* spinning "holder" threads */
    uint32_t cb_base, cb_n;             /* callbacks / short-lived threads */
    uint32_t dirty_base, dirty_n;       /* threads that exit / are terminated with live state */
    uint32_t main_tag;
    volatile LONG reported;
} vdom;

static int dom_owner_engine(const vdom *d, uint32_t tag) {
    uint32_t n = d->eng_n;
    if (!n || tag < 1) return -2;
    if (tag <= 4 * n) return (int)((tag - 1) % n);
    if (tag <= 5 * n) return (int)(tag - 1 - 4 * n);
    return -2;
}
/* idx: engine worker index (or -1); self_tag: the tag the thread currently uses */
static int dom_own(const vdom *d, int idx, uint32_t self_tag, uint32_t tag) {
    if (tag == self_tag) return 1;
    if (idx >= 0 && dom_owner_engine(d, tag) == idx) return 1;
    return 0;
}
static void dom_describe(const vdom *d, uint32_t tag, char *b, size_t cap) {
    uint32_t n = d->eng_n;
    if (n && tag >= 1 && tag <= 4 * n) snprintf(b, cap, "worker %u (phase %u)", (tag - 1) % n, (tag - 1) / n);
    else if (n && tag > 4 * n && tag <= 5 * n) snprintf(b, cap, "the APC routine run for worker %u", tag - 1 - 4 * n);
    else if (tag == d->main_tag) snprintf(b, cap, "the main (case) thread");
    else if (d->hold_n && tag >= d->hold_base && tag < d->hold_base + d->hold_n) snprintf(b, cap, "holder thread %u", tag - d->hold_base);
    else if (d->cb_n && tag >= d->cb_base && tag < d->cb_base + d->cb_n) snprintf(b, cap, "callback/short thread #%u", tag - d->cb_base);
    else if (d->dirty_n && tag >= d->dirty_base && tag < d->dirty_base + d->dirty_n) snprintf(b, cap, "exited/terminated thread #%u", tag - d->dirty_base);
    else snprintf(b, cap, "pattern tag %u", tag);
}
static int qw_patternlike(uint64_t v) { return (v >> 63) && ((v >> 48) & 0xC0) == 0x40; }

/* Count qwords in p[0..bytes) that decode to a known pattern the thread does not own. */
static int scan_foreign(const vdom *d, int idx, uint32_t self_tag, const void *p, size_t bytes,
                        uint64_t *fv, uint32_t *ft, size_t *fo) {
    int c = 0;
    for (size_t o = 0; o + 8 <= bytes; o += 8) {
        uint64_t v; memcpy(&v, (const uint8_t *)p + o, 8);
        if (!qw_patternlike(v)) continue;
        uint32_t tg, r, w;
        if (!t_pattern_explain(v, d->max_tag, &tg, &r, &w)) continue;
        if (dom_own(d, idx, self_tag, tg)) continue;
        if (!c) { *fv = v; *ft = tg; *fo = o; }
        c++;
    }
    return c;
}
/* Detailed report (first REPORT_MAX per case): CHECK_REG hexdump + decode of the first bad qword. */
static void report_reg(vdom *d, const char *who, const char *rname, int reg, const void *got, const void *want, size_t n) {
    if (InterlockedIncrement(&d->reported) > REPORT_MAX) return;
    char what[200]; snprintf(what, sizeof what, "%s: %s%d", who, rname, reg);
    CHECK_REG(what, got, want, n);
    for (size_t o = 0; o + 8 <= n; o += 8) {
        uint64_t g, w; memcpy(&g, (const uint8_t *)got + o, 8); memcpy(&w, (const uint8_t *)want + o, 8);
        if (g == w) continue;
        uint32_t tg, r, wd; char desc[96];
        if (t_pattern_explain(g, d->max_tag, &tg, &r, &wd)) {
            dom_describe(d, tg, desc, sizeof desc);
            t_note("%s: %s%d qword %u = 0x%016llx is the pattern of %s (reg %u word %u)", who, rname, reg,
                   (unsigned)(o / 8), (unsigned long long)g, desc, r, wd);
        } else {
            t_note("%s: %s%d qword %u = 0x%016llx is no known pattern (%s)", who, rname, reg, (unsigned)(o / 8),
                   (unsigned long long)g, g == 0 ? "zero: state lost / reinitialised" : "garbage");
        }
        break;
    }
}
static void report_leak(vdom *d, const char *who, const char *rname, int reg, uint64_t v, uint32_t tag, size_t off) {
    if (InterlockedIncrement(&d->reported) > REPORT_MAX) return;
    char desc[96]; dom_describe(d, tag, desc, sizeof desc);
    t_check(0, __FILE__, __LINE__, "no foreign register state", "%s: %s%d qword %u = 0x%016llx is the pattern of %s",
            who, rname, reg, (unsigned)(off / 8), (unsigned long long)v, desc);
}
/* Full compare of a register image (all regs of kind at full width, + k for ZMM). Returns bad registers. */
static int verify_full(vdom *d, const char *who, int kind, const t_vregs *got, const t_vregs *want) {
    int n = t_vregs_count(kind), bad = 0;
    for (int i = 0; i < n; i++)
        if (memcmp(got->v[i], want->v[i], (size_t)kind)) { bad++; report_reg(d, who, kind_name(kind), i, got->v[i], want->v[i], (size_t)kind); }
    if (kind == T_VK_ZMM) {
        uint64_t m = cpu_has(F_AVX512BW) ? ~0ull : 0xFFFFull;
        for (int i = 0; i < 8; i++)
            if ((got->k[i] & m) != (want->k[i] & m)) {
                bad++;
                if (InterlockedIncrement(&d->reported) <= REPORT_MAX)
                    t_check(0, __FILE__, __LINE__, "k reg preserved", "%s: k%d got 0x%llx want 0x%llx", who, i,
                            (unsigned long long)(got->k[i] & m), (unsigned long long)(want->k[i] & m));
            }
    }
    return bad;
}
/* After a CALL: ABI-nonvolatile part must match; everything is scanned for foreign values. */
static void verify_call(vdom *d, const char *who, int kind, int idx, uint32_t self_tag, const t_vregs *got,
                        const t_vregs *want, unsigned long long *bad_abi, unsigned long long *bad_leak,
                        unsigned long long *vol_changed) {
    int n = t_vregs_count(kind);
    for (int i = 0; i < n; i++) {
        if (!memcmp(got->v[i], want->v[i], (size_t)kind)) continue;
#ifdef T_X64
        if (i >= 6 && i <= 15 && memcmp(got->v[i], want->v[i], 16)) {
            (*bad_abi)++;
            report_reg(d, who, "xmm(nonvolatile)", i, got->v[i], want->v[i], 16);
        } else
#endif
            (*vol_changed)++;
        uint64_t fv = 0; uint32_t ft = 0; size_t fo = 0;
        int f = scan_foreign(d, idx, self_tag, got->v[i], (size_t)kind, &fv, &ft, &fo);
        if (f) { *bad_leak += (unsigned long long)f; report_leak(d, who, kind_name(kind), i, fv, ft, fo); }
    }
}

/* ======================================================================================
 * assembly: thread-entry capture, dirty exit, upper-lane peek
 *   cap layout: t_vregs at 0 (v[i] at i*64), MXCSR at 2112, FCW at 2116, FSW at 2118, FNSTENV at 2120
 * ==================================================================================== */
typedef struct cap {
    t_vregs r;
    uint32_t mx;
    uint16_t cw, sw;
    uint8_t env[28];
    /* C side */
    int mode;
    uint32_t tag;
    int kind;
    unsigned dirty_mx; unsigned short dirty_cw;
    volatile LONG bad, ran;
    vdom *d;
    t_vregs in, out;
} cap;
_Static_assert(offsetof(cap, mx) == 2112, "cap layout");
_Static_assert(offsetof(cap, cw) == 2116, "cap layout");
_Static_assert(offsetof(cap, env) == 2120, "cap layout");
/* dirty-exit block: registers at 0, MXCSR at 2112, FCW at 2116 */
typedef struct dirty {
    t_vregs r;
    uint32_t mx;
    uint16_t cw, pad;
    int kind;
} dirty;
_Static_assert(offsetof(dirty, mx) == 2112, "dirty layout");

DWORD WINAPI te_cap_after(void *p) __asm__("te_cap_after") __attribute__((used));
DWORD WINAPI te_cap_entry_xmm(void *p) __asm__("te_cap_entry_xmm");
DWORD WINAPI te_cap_entry_ymm_avx(void *p) __asm__("te_cap_entry_ymm_avx");
DWORD WINAPI te_cap_entry_zmm_avx512(void *p) __asm__("te_cap_entry_zmm_avx512");
void te_dirty_exit_xmm(const dirty *d) __asm__("te_dirty_exit_xmm") __attribute__((noreturn));
void te_dirty_exit_ymm_avx(const dirty *d) __asm__("te_dirty_exit_ymm_avx") __attribute__((noreturn));
void te_dirty_exit_zmm_avx512(const dirty *d) __asm__("te_dirty_exit_zmm_avx512") __attribute__((noreturn));
void te_peek_ymm_avx(t_vregs *out) __asm__("te_peek_ymm_avx");
void te_peek_zmm_avx512(t_vregs *out) __asm__("te_peek_zmm_avx512");

#define TX_FPCAP(r) \
    "  stmxcsr 2112(" r ")\n  fnstcw 2116(" r ")\n  fnstsw 2118(" r ")\n" \
    "  fnstenv 2120(" r ")\n  fldenv 2120(" r ")\n"
#define TX_DIRTYFP(r) "  ldmxcsr 2112(" r ")\n  fldcw 2116(" r ")\n  fld1\n  fldpi\n  fldl2t\n"
#define TX_FN(name) ".globl " name "\n.p2align 4\n" name ":\n"

#ifdef T_X64
#define TX_R16 "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#define TX_R32 TX_R16 ",16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
__asm__(
    ".text\n"
    TX_FN("te_cap_entry_xmm")
    "  .irp i," TX_R16 "\n  movdqu %xmm\\i, \\i*64(%rcx)\n  .endr\n" TX_FPCAP("%rcx") "  jmp te_cap_after\n"
    TX_FN("te_cap_entry_ymm_avx")
    "  .irp i," TX_R16 "\n  vmovdqu %ymm\\i, \\i*64(%rcx)\n  .endr\n" TX_FPCAP("%rcx") "  jmp te_cap_after\n"
    TX_FN("te_cap_entry_zmm_avx512")
    "  .irp i," TX_R32 "\n  vmovdqu64 %zmm\\i, \\i*64(%rcx)\n  .endr\n" TX_FPCAP("%rcx") "  jmp te_cap_after\n"
    /* load everything, leave the x87 stack non-empty, then ExitThread(0) without restoring anything */
    TX_FN("te_dirty_exit_xmm")
    "  .irp i," TX_R16 "\n  movdqu \\i*64(%rcx), %xmm\\i\n  .endr\n" TX_DIRTYFP("%rcx")
    "  subq $40, %rsp\n  xorl %ecx, %ecx\n  call *__imp_ExitThread(%rip)\n  ud2\n"
    TX_FN("te_dirty_exit_ymm_avx")
    "  .irp i," TX_R16 "\n  vmovdqu \\i*64(%rcx), %ymm\\i\n  .endr\n" TX_DIRTYFP("%rcx")
    "  subq $40, %rsp\n  xorl %ecx, %ecx\n  call *__imp_ExitThread(%rip)\n  ud2\n"
    TX_FN("te_dirty_exit_zmm_avx512")
    "  .irp i," TX_R32 "\n  vmovdqu64 \\i*64(%rcx), %zmm\\i\n  .endr\n" TX_DIRTYFP("%rcx")
    "  subq $40, %rsp\n  xorl %ecx, %ecx\n  call *__imp_ExitThread(%rip)\n  ud2\n"
    TX_FN("te_peek_ymm_avx")
    "  .irp i," TX_R16 "\n  vmovdqu %ymm\\i, \\i*64(%rcx)\n  .endr\n  ret\n"
    TX_FN("te_peek_zmm_avx512")
    "  .irp i," TX_R32 "\n  vmovdqu64 %zmm\\i, \\i*64(%rcx)\n  .endr\n  ret\n"
);
#else
#define TX_R8 "0,1,2,3,4,5,6,7"
__asm__(
    ".text\n"
    TX_FN("te_cap_entry_xmm")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  movdqu %xmm\\i, \\i*64(%eax)\n  .endr\n" TX_FPCAP("%eax") "  jmp te_cap_after\n"
    TX_FN("te_cap_entry_ymm_avx")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  vmovdqu %ymm\\i, \\i*64(%eax)\n  .endr\n" TX_FPCAP("%eax") "  jmp te_cap_after\n"
    TX_FN("te_cap_entry_zmm_avx512")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  vmovdqu64 %zmm\\i, \\i*64(%eax)\n  .endr\n" TX_FPCAP("%eax") "  jmp te_cap_after\n"
    TX_FN("te_dirty_exit_xmm")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  movdqu \\i*64(%eax), %xmm\\i\n  .endr\n" TX_DIRTYFP("%eax")
    "  andl $-16, %esp\n  subl $12, %esp\n  pushl $0\n  call *__imp__ExitThread@4\n  ud2\n"
    TX_FN("te_dirty_exit_ymm_avx")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  vmovdqu \\i*64(%eax), %ymm\\i\n  .endr\n" TX_DIRTYFP("%eax")
    "  andl $-16, %esp\n  subl $12, %esp\n  pushl $0\n  call *__imp__ExitThread@4\n  ud2\n"
    TX_FN("te_dirty_exit_zmm_avx512")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  vmovdqu64 \\i*64(%eax), %zmm\\i\n  .endr\n" TX_DIRTYFP("%eax")
    "  andl $-16, %esp\n  subl $12, %esp\n  pushl $0\n  call *__imp__ExitThread@4\n  ud2\n"
    TX_FN("te_peek_ymm_avx")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  vmovdqu %ymm\\i, \\i*64(%eax)\n  .endr\n  ret\n"
    TX_FN("te_peek_zmm_avx512")
    "  movl 4(%esp), %eax\n  .irp i," TX_R8 "\n  vmovdqu64 %zmm\\i, \\i*64(%eax)\n  .endr\n  ret\n"
);
#endif

static LPTHREAD_START_ROUTINE cap_entry(int kind) {
    return kind == T_VK_ZMM ? te_cap_entry_zmm_avx512 : kind == T_VK_YMM ? te_cap_entry_ymm_avx : te_cap_entry_xmm;
}
static void peek_wide(int kind, t_vregs *out) {
    memset(out, 0, sizeof *out);
    if (kind == T_VK_ZMM) te_peek_zmm_avx512(out);
    else if (kind == T_VK_YMM) te_peek_ymm_avx(out);
}

enum { CAP_PLAIN = 0, CAP_CHURN = 1 };
/* Continuation of the capture stub (runs as the thread procedure after the raw state was saved). */
DWORD WINAPI te_cap_after(void *p) {
    cap *c = (cap *)p;
    c->ran = 1;
    if (c->mode == CAP_CHURN) {
        /* own state round trip under the thread's own FP environment, then leave it dirty on exit */
        t_vregs_pattern(&c->in, c->kind, c->tag);
        t_vregs_spin(c->kind, &c->in, &c->out, NULL, 300);
        if (verify_full(c->d, "churn thread", c->kind, &c->out, &c->in)) c->bad++;
        t_set_mxcsr(c->dirty_mx);
        t_set_x87_cw(c->dirty_cw);
    }
    return 0x600Du;
}

typedef struct capstats {
    int n, bad_mx, bad_cw, bad_tw, leaks, zero_regs, total_regs, upper_nonzero, mx_flags_set, ran_bad;
    unsigned first_mx; unsigned short first_cw, first_tw;
} capstats;
/* Analyse one captured thread-entry state (contract 3 + leak scan). */
static void cap_analyse(cap *c, vdom *d, int kind, capstats *s, const char *label) {
    s->n++;
    uint16_t tw; memcpy(&tw, c->env + 8, 2);
    if (s->n == 1) { s->first_mx = c->mx; s->first_cw = c->cw; s->first_tw = tw; }
    if ((c->mx & MX_CTRL) != MX_DEFAULT) {
        s->bad_mx++;
        if (s->bad_mx <= 3) t_check(0, __FILE__, __LINE__, "new thread MXCSR control == 0x1F80", "%s: MXCSR 0x%04x at thread entry", label, c->mx);
    } else CHECK(1);
    if (c->mx & 0x3F) s->mx_flags_set++;
    if (c->cw != CW_DEFAULT) {
        s->bad_cw++;
        if (s->bad_cw <= 3) t_check(0, __FILE__, __LINE__, "new thread x87 CW == 0x027F", "%s: FCW 0x%04x at thread entry", label, c->cw);
    } else CHECK(1);
    if (tw != 0xFFFF) {
        s->bad_tw++;
        if (s->bad_tw <= 3) t_check(0, __FILE__, __LINE__, "new thread x87 stack empty (FTW 0xFFFF)", "%s: FTW 0x%04x FSW 0x%04x at thread entry", label, tw, c->sw);
    } else CHECK(1);
    int n = t_vregs_count(kind);
    for (int i = 0; i < n; i++) {
        uint64_t fv = 0; uint32_t ft = 0; size_t fo = 0;
        int f = scan_foreign(d, -1, 0xFFFFFFFFu, c->r.v[i], (size_t)kind, &fv, &ft, &fo);
        if (f) { s->leaks += f; report_leak(d, label, kind_name(kind), i, fv, ft, fo); }
        static const uint8_t z[64];
        s->total_regs++;
        if (!memcmp(c->r.v[i], z, (size_t)kind)) s->zero_regs++;
        if (kind > 16 && memcmp(c->r.v[i] + 16, z, (size_t)kind - 16)) s->upper_nonzero++;
    }
    CHECKF(c->ran, "%s: thread procedure did not run", label);
}
static void capstats_report(const capstats *s, const char *what) {
    CHECKF(s->leaks == 0, "%s: %d foreign pattern qwords in new threads' registers", what, s->leaks);
    CHECKF(s->bad_mx == 0, "%s: %d/%d new threads with MXCSR control != 0x1F80", what, s->bad_mx, s->n);
    CHECKF(s->bad_cw == 0, "%s: %d/%d new threads with x87 CW != 0x027F", what, s->bad_cw, s->n);
    CHECKF(s->bad_tw == 0, "%s: %d/%d new threads with a non-empty x87 stack", what, s->bad_tw, s->n);
    t_info("threads", "%d", s->n);
    t_info("entry_mxcsr", "0x%04x", s->first_mx);
    t_info("entry_fcw", "0x%04x", s->first_cw);
    t_info("entry_ftw", "0x%04x", s->first_tw);
    t_info("entry_mxcsr_flags_set", "%d", s->mx_flags_set);
    t_info("entry_zero_regs", "%d/%d", s->zero_regs, s->total_regs);
    t_info("entry_upper_nonzero_regs", "%d", s->upper_nonzero);
}
static cap *cap_new(void) {
    cap *c = (cap *)_aligned_malloc(sizeof(cap), 64);
    if (c) memset(c, 0, sizeof *c);
    return c;
}
/* Create one capture thread (optionally pinned), wait for it, check its exit code. 0 = create failed. */
static int cap_run(cap *c, int kind, DWORD_PTR pin, DWORD *err) {
    HANDLE h = mk_thread(cap_entry(kind), c, CREATE_SUSPENDED, NULL);
    if (!h) { *err = GetLastError(); return 0; }
    if (pin) SetThreadAffinityMask(h, pin);
    ResumeThread(h);
    DWORD w = WaitForSingleObject(h, 10000);
    CHECKF(w == WAIT_OBJECT_0, "capture thread did not finish (wait 0x%lx)", w);
    DWORD ec = 0; GetExitCodeThread(h, &ec);
    CHECKF(ec == 0x600Du, "capture thread exit code 0x%lx", ec);
    CloseHandle(h);
    return 1;
}

/* ======================================================================================
 * holder threads: spin with full-width patterns while something else happens
 * ==================================================================================== */
typedef struct holder {
    t_vregs in, out;
    vdom *d;
    uint32_t tag;
    int kind;
    volatile LONG *stop;
    HANDLE h;
    unsigned long long chunks, bad, bad_env;
} holder;
static DWORD WINAPI holder_main(void *p) {
    holder *hd = (holder *)p;
    unsigned mx = mxcsr_variant(hd->tag); unsigned short cw = cw_variant(hd->tag);
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    t_vregs_pattern(&hd->in, hd->kind, hd->tag);
    do {
        t_vregs_spin(hd->kind, &hd->in, &hd->out, hd->stop, 4000);
        hd->chunks++;
        if (verify_full(hd->d, "holder thread", hd->kind, &hd->out, &hd->in)) hd->bad++;
        if (((t_get_mxcsr() ^ mx) & MX_CTRL) || t_get_x87_cw() != cw) hd->bad_env++;
    } while (!*hd->stop);
    return 0;
}
static int holders_start(holder *hs, int k, vdom *d, int kind, volatile LONG *stop, DWORD_PTR pin) {
    int started = 0;
    for (int i = 0; i < k; i++) {
        hs[i].d = d; hs[i].tag = d->hold_base + (uint32_t)i; hs[i].kind = kind; hs[i].stop = stop;
        hs[i].h = mk_thread(holder_main, &hs[i], CREATE_SUSPENDED, NULL);
        if (!hs[i].h) break;
        if (pin) SetThreadAffinityMask(hs[i].h, pin);
        ResumeThread(hs[i].h);
        started++;
    }
    return started;
}
static void holders_stop(holder *hs, int k, volatile LONG *stop) {
    *stop = 1;
    for (int i = 0; i < k; i++) {
        DWORD w = WaitForSingleObject(hs[i].h, 10000);
        if (w != WAIT_OBJECT_0) t_retire_process("thread_xstate: holder thread did not stop within 10 s (no TerminateThread)");
        CloseHandle(hs[i].h);
        CHECKF(hs[i].bad == 0 && hs[i].bad_env == 0 && hs[i].chunks > 0,
               "holder %d: %llu/%llu spin chunks with corrupted registers, %llu with changed MXCSR/FCW",
               i, hs[i].bad, hs[i].chunks, hs[i].bad_env);
    }
}
static holder *holders_alloc(int k) {
    holder *h = (holder *)_aligned_malloc(sizeof(holder) * (size_t)k, 64);
    if (h) memset(h, 0, sizeof(holder) * (size_t)k);
    return h;
}

/* ======================================================================================
 * the isolation engine
 * ==================================================================================== */
enum { OP_NONE, OP_SLEEP0, OP_SWITCH, OP_EVENT_RING, OP_MUTEX, OP_SEM, OP_APC, OP_CONDVAR, OP_SRW, OP_CRITSEC,
       OP_WAITADDR, OP_PIPE, OP_TIMER, OP_WFMO, OP_SOAW, OP_MIGRATE, OP_SYNCBAR, OP_CBAR, OP_APC_WAITEX, OP_COUNT };
static const char *const g_opname[OP_COUNT] = {
    "preemption only", "Sleep(0)", "SwitchToThread", "event ring wait", "mutex wait", "semaphore wait",
    "SleepEx alertable + APC", "SleepConditionVariableSRW", "SRW lock", "critical section", "WaitOnAddress",
    "pipe ReadFile", "waitable timer wait", "WaitForMultipleObjects", "SignalObjectAndWait",
    "SetThreadAffinityMask migration", "EnterSynchronizationBarrier", "SRW/condvar barrier",
    "WaitForSingleObjectEx alertable + APC" };
enum { PIN_NONE, PIN_ONE, PIN_SPREAD };
enum { RF_VEC, RF_XMM_PEEK, RF_X87, RF_MMX, RF_MIX };

typedef BOOL (WINAPI *woa_t)(volatile VOID *, PVOID, SIZE_T, DWORD);
typedef VOID (WINAPI *wba_t)(PVOID);
typedef BOOL (WINAPI *isb_t)(void *, LONG, LONG);
typedef BOOL (WINAPI *esb_t)(void *, DWORD);
typedef BOOL (WINAPI *dsb_t)(void *);
typedef BOOL (WINAPI *qapc2_t)(PAPCFUNC, HANDLE, ULONG_PTR, int);   /* Windows 11: QueueUserAPC2 */

typedef struct cbar { SRWLOCK l; CONDITION_VARIABLE cv; LONG count, n, gen, broken; } cbar;

struct scen;
typedef struct worker {
    t_vregs in, out, apc_in, apc_out, peek;
    struct scen *sc;
    int idx, rf, kind;
    HANDLE h; DWORD tid;
    volatile LONG finished;
    volatile LONG phase;
    uint32_t cur_tag;
    unsigned cur_mx; unsigned short cur_cw;
    unsigned long long iters, spins, bad_spin, bad_abi, bad_leak, bad_mx, bad_cw, vol_changed, timeouts,
                       apc_runs, cpu_mismatch, bad_mxflags;
    t_f80 xin[8], xout[8];
    uint64_t mmin[8], mmout[8];
} worker;

typedef struct scen {
    vdom d;
    int op, n, pin, rf, kind, prio_mix, susp, mig_ext, getctx, mxflags;
    unsigned spin;
    unsigned long long ctx_n, ctx_fail, ctx_own, ctx_leak;
    int fixed_iters;
    DWORD ms;
    volatile LONG stop, abort;
    worker *w; size_t wbytes;
    int created; DWORD create_err;
    HANDLE *ev, ev_never, mutex, sem; LONG sem_cap;
    HANDLE *pr, *pw, *timers;
    SRWLOCK srw; CONDITION_VARIABLE cv; CRITICAL_SECTION cs; int cs_init;
    volatile LONG inside, readers, writer, excl_viol, api_fail, gen, apc_wrong, bar_viol, aff_fail, hung, resume_fail;
    __attribute__((aligned(16))) unsigned char syncbar[64];
    cbar cb;
    woa_t woa; wba_t wba; esb_t esb; dsb_t dsb; int syncbar_init;
    DWORD_PTR cpus[64]; int ncpu;
    double secs;
    unsigned long long susp_ops, mig_ops;
} scen;

static void excl_enter(scen *sc) { if (InterlockedIncrement(&sc->inside) != 1) InterlockedIncrement(&sc->excl_viol); }
static void excl_leave(scen *sc) { InterlockedDecrement(&sc->inside); }
static void wait_result(scen *sc, worker *w, DWORD r) {
    if (r == WAIT_TIMEOUT) w->timeouts++;
    else InterlockedIncrement(&sc->api_fail);
}

static void op_sleep0(void *p) { (void)p; Sleep(0); }
static void op_switch(void *p) { (void)p; SwitchToThread(); }
static void op_event_ring(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    DWORD r = WaitForSingleObject(sc->ev[w->idx], 50);
    if (r == WAIT_OBJECT_0) { if (!SetEvent(sc->ev[(w->idx + 1) % sc->n])) InterlockedIncrement(&sc->api_fail); }
    else wait_result(sc, w, r);
}
static void op_mutex(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    DWORD r = WaitForSingleObject(sc->mutex, 100);
    if (r == WAIT_OBJECT_0) {
        excl_enter(sc); SwitchToThread(); excl_leave(sc);
        if (!ReleaseMutex(sc->mutex)) InterlockedIncrement(&sc->api_fail);
    } else wait_result(sc, w, r);
}
static void op_sem(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    DWORD r = WaitForSingleObject(sc->sem, 100);
    if (r == WAIT_OBJECT_0) {
        if (InterlockedIncrement(&sc->inside) > sc->sem_cap) InterlockedIncrement(&sc->excl_viol);
        Sleep(0);
        InterlockedDecrement(&sc->inside);
        if (!ReleaseSemaphore(sc->sem, 1, NULL)) InterlockedIncrement(&sc->api_fail);
    } else wait_result(sc, w, r);
}
/* APC routine: runs on the TARGET thread inside its alertable SleepEx. Loads its own pattern into every
 * vector register and its own FP environment, verifies them, and restores the environment (ABI). */
static void CALLBACK apc_fn(ULONG_PTR p) {
    worker *t = (worker *)p; scen *sc = t->sc;
    if (GetCurrentThreadId() != t->tid) { InterlockedIncrement(&sc->apc_wrong); return; }
    t->apc_runs++;
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    unsigned amx = mxcsr_variant((unsigned)t->idx * 7u + 3u); unsigned short acw = cw_variant((unsigned)t->idx + 5u);
    t_set_mxcsr(amx); t_set_x87_cw(acw);
    int kind = t->rf == RF_VEC ? t->kind : T_VK_XMM;
    t_vregs_pattern(&t->apc_in, kind, 1u + 4u * (uint32_t)sc->n + (uint32_t)t->idx);
    t_vregs_spin(kind, &t->apc_in, &t->apc_out, NULL, 200);
    if (verify_full(&sc->d, "APC routine", kind, &t->apc_out, &t->apc_in)) t->bad_spin++;
    if (((t_get_mxcsr() ^ amx) & MX_CTRL) || t_get_x87_cw() != acw) t->bad_mx++;
    t_set_mxcsr(mx); t_set_x87_cw(cw);
}
static void op_apc(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    worker *nx = &sc->w[(w->idx + 1) % sc->n];
    if (!QueueUserAPC(apc_fn, nx->h, (ULONG_PTR)nx) && !sc->stop) InterlockedIncrement(&sc->api_fail);
    DWORD r = SleepEx(50, TRUE);
    if (r == 0) w->timeouts++;
    else if (r != WAIT_IO_COMPLETION) InterlockedIncrement(&sc->api_fail);
}
/* same, but the alertable wait is a kernel object wait (never-signalled event): the APC is delivered from
 * inside NtWaitForSingleObject and the wait returns WAIT_IO_COMPLETION (WaitForSingleObjectEx docs). */
static void op_apc_waitex(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    worker *nx = &sc->w[(w->idx + 1) % sc->n];
    if (!QueueUserAPC(apc_fn, nx->h, (ULONG_PTR)nx) && !sc->stop) InterlockedIncrement(&sc->api_fail);
    DWORD r = WaitForSingleObjectEx(sc->ev_never, 50, TRUE);
    if (r == WAIT_TIMEOUT) w->timeouts++;
    else if (r != WAIT_IO_COMPLETION) InterlockedIncrement(&sc->api_fail);
}
static void op_condvar(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    AcquireSRWLockExclusive(&sc->srw);
    excl_enter(sc); sc->gen++; excl_leave(sc);
    WakeConditionVariable(&sc->cv);
    if (!SleepConditionVariableSRW(&sc->cv, &sc->srw, 20, 0)) {
        if (GetLastError() == ERROR_TIMEOUT) w->timeouts++; else InterlockedIncrement(&sc->api_fail);
    }
    excl_enter(sc); excl_leave(sc);           /* lock is held again after the wait */
    ReleaseSRWLockExclusive(&sc->srw);
}
static void op_srw(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    if (w->iters & 1) {
        AcquireSRWLockShared(&sc->srw);
        InterlockedIncrement(&sc->readers);
        if (sc->writer) InterlockedIncrement(&sc->excl_viol);
        Sleep(0);
        InterlockedDecrement(&sc->readers);
        ReleaseSRWLockShared(&sc->srw);
    } else {
        AcquireSRWLockExclusive(&sc->srw);
        if (InterlockedExchange(&sc->writer, 1) != 0 || sc->readers != 0) InterlockedIncrement(&sc->excl_viol);
        excl_enter(sc); SwitchToThread(); excl_leave(sc);
        InterlockedExchange(&sc->writer, 0);
        ReleaseSRWLockExclusive(&sc->srw);
    }
}
static void op_critsec(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    EnterCriticalSection(&sc->cs);
    excl_enter(sc); SwitchToThread(); excl_leave(sc);
    LeaveCriticalSection(&sc->cs);
}
static void op_waitaddr(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    LONG v = InterlockedIncrement(&sc->gen);
    sc->wba((PVOID)&sc->gen);
    LONG cmp = v;
    if (!sc->woa(&sc->gen, &cmp, sizeof cmp, 20)) {
        if (GetLastError() == ERROR_TIMEOUT) w->timeouts++; else InterlockedIncrement(&sc->api_fail);
    }
}
static void op_pipe(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    char b = 0; DWORD got = 0, wr = 0;
    if (!ReadFile(sc->pr[w->idx], &b, 1, &got, NULL) || got != 1) { InterlockedIncrement(&sc->api_fail); return; }
    if (!WriteFile(sc->pw[(w->idx + 1) % sc->n], &b, 1, &wr, NULL) || wr != 1) InterlockedIncrement(&sc->api_fail);
}
static void op_timer(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    LARGE_INTEGER due; due.QuadPart = -10000;   /* 1 ms relative */
    if (!SetWaitableTimer(sc->timers[w->idx], &due, 0, NULL, NULL, FALSE)) { InterlockedIncrement(&sc->api_fail); return; }
    DWORD r = WaitForSingleObject(sc->timers[w->idx], 1000);
    if (r != WAIT_OBJECT_0) wait_result(sc, w, r);
}
static void op_wfmo(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    HANDLE hs[2] = { sc->ev[w->idx], sc->ev_never };
    DWORD r = WaitForMultipleObjects(2, hs, FALSE, 50);
    if (r == WAIT_OBJECT_0) { if (!SetEvent(sc->ev[(w->idx + 1) % sc->n])) InterlockedIncrement(&sc->api_fail); }
    else wait_result(sc, w, r);                /* WAIT_OBJECT_0+1 (never-signalled event) is a failure */
}
static void op_soaw(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    DWORD r = SignalObjectAndWait(sc->ev[(w->idx + 1) % sc->n], sc->ev[w->idx], 50, FALSE);
    if (r != WAIT_OBJECT_0) wait_result(sc, w, r);
}
static void op_migrate(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    DWORD_PTR m = sc->cpus[(unsigned)(w->idx + (int)w->iters) % (unsigned)sc->ncpu];
    if (!SetThreadAffinityMask(GetCurrentThread(), m)) { InterlockedIncrement(&sc->api_fail); return; }
    /* "If the new thread affinity mask does not specify the processor that is currently running the
     * thread, the thread is rescheduled on one of the allowable processors." (SetThreadAffinityMask docs) */
    PROCESSOR_NUMBER pn; GetCurrentProcessorNumberEx(&pn);
    if (pn.Number >= sizeof(DWORD_PTR) * 8 || !(((DWORD_PTR)1 << pn.Number) & m)) w->cpu_mismatch++;
    Sleep(0);
}
/* barrier phase check: each thread publishes "arrived for phase p" (= p+1) before the barrier; after it
 * the neighbour must have arrived for p and cannot be past p+1. */
static void bar_after(scen *sc, worker *w) {
    LONG mine = (LONG)w->iters + 1;
    LONG nb = sc->w[(w->idx + 1) % sc->n].phase;
    if (nb < mine || nb > mine + 1) InterlockedIncrement(&sc->bar_viol);
}
static void op_syncbar(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    w->phase = (LONG)w->iters + 1;
    sc->esb(sc->syncbar, 0);
    bar_after(sc, w);
}
static void cbar_wait(cbar *b) {
    AcquireSRWLockExclusive(&b->l);
    LONG g = b->gen;
    if (++b->count == b->n) { b->count = 0; b->gen++; WakeAllConditionVariable(&b->cv); }
    else {
        DWORD t0 = GetTickCount();
        while (g == b->gen && !b->broken) {
            SleepConditionVariableSRW(&b->cv, &b->l, 1000, 0);
            if (g == b->gen && GetTickCount() - t0 > 20000) b->broken = 1;   /* a participant is gone */
        }
    }
    ReleaseSRWLockExclusive(&b->l);
}
static void op_cbar(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    w->phase = (LONG)w->iters + 1;
    cbar_wait(&sc->cb);
    bar_after(sc, w);
}
static void (*const g_ops[OP_COUNT])(void *) = {
    NULL, op_sleep0, op_switch, op_event_ring, op_mutex, op_sem, op_apc, op_condvar, op_srw, op_critsec,
    op_waitaddr, op_pipe, op_timer, op_wfmo, op_soaw, op_migrate, op_syncbar, op_cbar, op_apc_waitex };

static void make_f80(t_f80 *out, uint32_t tag) {
    for (int i = 0; i < 8; i++) {
        uint64_t m = t_pattern_word(tag, 0x20u + (uint32_t)i, 0);     /* bit 63 set: normal, explicit J-bit */
        uint16_t e = (uint16_t)(0x3FF0u + ((m >> 8) & 0x1Fu)) | (uint16_t)((tag & 1u) << 15);
        memcpy(out[i].b, &m, 8); memcpy(out[i].b + 8, &e, 2);
    }
}
static void check_env(worker *w, const char *after) {
    scen *sc = w->sc;
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    if ((mx ^ w->cur_mx) & MX_CTRL) {
        w->bad_mx++;
        if (InterlockedIncrement(&sc->d.reported) <= REPORT_MAX)
            t_check(0, __FILE__, __LINE__, "MXCSR control bits preserved", "worker %d after %s: MXCSR 0x%04x, set 0x%04x",
                    w->idx, after, mx, w->cur_mx);
    }
    if (cw != w->cur_cw) {
        w->bad_cw++;
        if (InterlockedIncrement(&sc->d.reported) <= REPORT_MAX)
            t_check(0, __FILE__, __LINE__, "x87 control word preserved", "worker %d after %s: FCW 0x%04x, set 0x%04x",
                    w->idx, after, cw, w->cur_cw);
    }
}

static void worker_iter(worker *w, uint32_t tag) {
    scen *sc = w->sc;
    volatile LONG *stop = sc->fixed_iters ? NULL : &sc->stop;
    const char *opn = g_opname[sc->op];
    char who[96];
    switch (w->rf) {
    case RF_VEC:
    case RF_XMM_PEEK: {
        int kind = w->rf == RF_VEC ? w->kind : T_VK_XMM;
        t_vregs_pattern(&w->in, kind, tag);
        if (sc->spin) {
            /* mxflags: the sticky MXCSR status flags (5:0) are thread state too. Across a GPR-only spin
             * (nothing but preemption in between, no SSE arithmetic) the WHOLE MXCSR must survive. */
            unsigned fmx = 0;
            if (sc->mxflags) {
                fmx = w->cur_mx | ((((unsigned)w->idx * 7u + (unsigned)w->iters) % 63u) + 1u);
                t_set_mxcsr(fmx);
            }
            t_vregs_spin(kind, &w->in, &w->out, stop, sc->spin);
            if (sc->mxflags) {
                unsigned g = t_get_mxcsr();
                t_set_mxcsr(w->cur_mx);
                if (g != fmx) {
                    w->bad_mxflags++;
                    if (InterlockedIncrement(&sc->d.reported) <= REPORT_MAX)
                        t_check(0, __FILE__, __LINE__, "whole MXCSR (incl. status flags) survives preemption",
                                "worker %d: MXCSR 0x%04x after spin, loaded 0x%04x", w->idx, g, fmx);
                }
            }
            w->spins++;
            snprintf(who, sizeof who, "worker %d after preemptible spin", w->idx);
            if (verify_full(&sc->d, who, kind, &w->out, &w->in)) w->bad_spin++;
        }
        if (sc->op != OP_NONE) {
            t_vregs_call(kind, &w->in, &w->out, g_ops[sc->op], w);
            snprintf(who, sizeof who, "worker %d after %s", w->idx, opn);
            verify_call(&sc->d, who, kind, w->idx, tag, &w->out, &w->in, &w->bad_abi, &w->bad_leak, &w->vol_changed);
        }
        if (w->rf == RF_XMM_PEEK && w->kind > T_VK_XMM) {
            /* This thread only ever executes legacy SSE on XMM: its upper YMM/ZMM bits may hold its own
             * (initial / library) values but never another thread's pattern. */
            peek_wide(w->kind, &w->peek);
            int n = t_vregs_count(w->kind);
            for (int i = 0; i < n; i++) {
                uint64_t fv = 0; uint32_t ft = 0; size_t fo = 0;
                int f = scan_foreign(&sc->d, w->idx, tag, w->peek.v[i], (size_t)w->kind, &fv, &ft, &fo);
                if (f) { w->bad_leak += (unsigned long long)f; report_leak(&sc->d, "SSE-only worker upper lanes", kind_name(w->kind), i, fv, ft, fo); }
            }
        }
        break;
    }
    case RF_X87:
        make_f80(w->xin, tag);
        t_x87_spin(w->xin, w->xout, stop, sc->spin ? sc->spin : 1000);
        w->spins++;
        for (int i = 0; i < 8; i++)
            if (memcmp(w->xin[i].b, w->xout[i].b, 10)) {
                w->bad_spin++;
                snprintf(who, sizeof who, "worker %d x87 stack after preemptible spin", w->idx);
                report_reg(&sc->d, who, "st", i, w->xout[i].b, w->xin[i].b, 8);
            }
        if (sc->op != OP_NONE) g_ops[sc->op](w);
        break;
    case RF_MMX:
        for (int i = 0; i < 8; i++) w->mmin[i] = t_pattern_word(tag, 0x30u + (uint32_t)i, 0);
        t_mmx_spin(w->mmin, w->mmout, stop, sc->spin ? sc->spin : 1000);
        w->spins++;
        for (int i = 0; i < 8; i++)
            if (w->mmin[i] != w->mmout[i]) {
                w->bad_spin++;
                snprintf(who, sizeof who, "worker %d MMX after preemptible spin", w->idx);
                report_reg(&sc->d, who, "mm", i, &w->mmout[i], &w->mmin[i], 8);
            }
        if (sc->op != OP_NONE) g_ops[sc->op](w);
        break;
    }
    check_env(w, opn);
}

static DWORD WINAPI worker_main(void *p) {
    worker *w = (worker *)p; scen *sc = w->sc;
    if (sc->abort) { w->finished = 1; return 0; }
    for (;;) {
        if (sc->fixed_iters && w->iters >= (unsigned long long)sc->fixed_iters) break;
        uint32_t tag = 1u + (uint32_t)w->idx + (uint32_t)sc->n * (uint32_t)(w->iters & 3);
        unsigned k = (unsigned)w->idx * 5u + (unsigned)w->iters;
        w->cur_tag = tag;
        w->cur_mx = mxcsr_variant(k); w->cur_cw = cw_variant(k);
        t_set_mxcsr(w->cur_mx); t_set_x87_cw(w->cur_cw);
        worker_iter(w, tag);
        w->iters++;
        if (!sc->fixed_iters && sc->stop) break;
    }
    t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
    w->finished = 1;
    return 0;
}

/* GetThreadContext of a SUSPENDED worker (GetThreadContext also waits until the asynchronous suspension
 * is complete). The legacy XMM image in the context must be the worker's own register state: never another
 * thread's pattern (e.g. the controller's, whose pattern is live while it calls GetThreadContext).
 * x64: CONTEXT.FltSave.XmmRegisters[16]; x86: CONTEXT_EXTENDED_REGISTERS = FXSAVE image, XMM0 at byte 160.
 * Only called from the controller thread (static buffer). */
static void ctx_probe(scen *sc, worker *w) {
    static CONTEXT cx __attribute__((aligned(64)));
    memset(&cx, 0, sizeof cx);
#ifdef T_X64
    cx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT;
#else
    cx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS;
#endif
    if (!GetThreadContext(w->h, &cx)) { sc->ctx_fail++; return; }
    sc->ctx_n++;
#ifdef T_X64
    const uint8_t *fx = (const uint8_t *)&cx.FltSave; int nx = 16;
#else
    const uint8_t *fx = (const uint8_t *)cx.ExtendedRegisters; int nx = 8;
#endif
    const uint8_t *x = fx + 160;
    int own = 0;
    /* ST0-7 / MM0-7 slots (FXSAVE bytes 32..159; MMX and x87-mantissa patterns of RF_MIX workers) */
    for (int i = 0; i < 8; i++) {
        uint64_t fv = 0; uint32_t ft = 0; size_t fo = 0;
        int f = scan_foreign(&sc->d, w->idx, w->cur_tag, fx + 32 + 16 * i, 8, &fv, &ft, &fo);
        if (f) { sc->ctx_leak += (unsigned long long)f; report_leak(&sc->d, "GetThreadContext(suspended worker)", "st/mm", i, fv, ft, fo); }
    }
    for (int i = 0; i < nx; i++) {
        uint64_t fv = 0; uint32_t ft = 0; size_t fo = 0;
        int f = scan_foreign(&sc->d, w->idx, w->cur_tag, x + 16 * i, 16, &fv, &ft, &fo);
        if (f) { sc->ctx_leak += (unsigned long long)f; report_leak(&sc->d, "GetThreadContext(suspended worker)", "xmm", i, fv, ft, fo); }
        for (int q = 0; q < 2; q++) {
            uint64_t v; memcpy(&v, x + 16 * i + 8 * q, 8);
            uint32_t tg, r, wd;
            if (qw_patternlike(v) && t_pattern_explain(v, sc->d.max_tag, &tg, &r, &wd) && dom_own(&sc->d, w->idx, w->cur_tag, tg)) own++;
        }
    }
    if (own) sc->ctx_own++;
}

/* runs on the case thread INSIDE t_vregs_call (main pattern live in the registers): no SKIP/REQUIRE here */
static void run_body(void *p) {
    scen *sc = (scen *)p;
    /* the controller must not be starved by spinning workers (few-CPU hosts, priority mix) */
    int prio0 = GetThreadPriority(GetCurrentThread());
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    for (int i = 0; i < sc->n; i++) {
        worker *w = &sc->w[i];
        w->h = mk_thread(worker_main, w, CREATE_SUSPENDED, &w->tid);
        if (!w->h) { sc->create_err = GetLastError(); break; }
        sc->created++;
    }
    if (sc->created < sc->n) {
        sc->abort = 1;
        for (int i = 0; i < sc->created; i++) ResumeThread(sc->w[i].h);
    } else {
        for (int i = 0; i < sc->n; i++) {
            worker *w = &sc->w[i];
            DWORD_PTR m = 0;
            if (sc->pin == PIN_ONE) m = sc->cpus[0];
            else if (sc->pin == PIN_SPREAD) m = sc->cpus[i % sc->ncpu];
            if (m && !SetThreadAffinityMask(w->h, m)) InterlockedIncrement(&sc->aff_fail);
            if (sc->prio_mix) {
                static const int pr[4] = { THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL,
                                           THREAD_PRIORITY_ABOVE_NORMAL };
                if (!SetThreadPriority(w->h, pr[i % 4])) InterlockedIncrement(&sc->api_fail);
            }
        }
        /* tokens for ring operations */
        int tokens = sc->n <= 16 ? 1 : sc->n / 16;
        if (sc->op == OP_EVENT_RING || sc->op == OP_WFMO || sc->op == OP_SOAW)
            for (int t = 0; t < tokens; t++) SetEvent(sc->ev[(t * 16) % sc->n]);
        if (sc->op == OP_PIPE)
            for (int t = 0; t < tokens; t++) { DWORD wr; char b = 'T'; WriteFile(sc->pw[(t * 16) % sc->n], &b, 1, &wr, NULL); }
    }
    uint64_t t0 = t_qpc();
    if (!sc->abort)
        for (int i = 0; i < sc->n; i++) if (ResumeThread(sc->w[i].h) == (DWORD)-1) InterlockedIncrement(&sc->resume_fail);
    if (!sc->abort && !sc->fixed_iters) {
        unsigned rot = 0;
        while (t_qpc_ms(t0, t_qpc()) < (double)sc->ms) {
            if (sc->mig_ext) {
                /* migrate workers under their feet: they spin with live registers */
                for (int k = 0; k < 64 && k < sc->n; k++, rot++) {
                    worker *w = &sc->w[rot % (unsigned)sc->n];
                    if (!SetThreadAffinityMask(w->h, sc->cpus[(rot / (unsigned)sc->n + (unsigned)w->idx) % (unsigned)sc->ncpu]))
                        InterlockedIncrement(&sc->aff_fail);
                    sc->mig_ops++;
                }
                Sleep(1);
            } else if (sc->susp) {
                for (int k = 0; k < 16 && k < sc->n; k++, rot++) {
                    worker *w = &sc->w[rot % (unsigned)sc->n];
                    if (SuspendThread(w->h) == (DWORD)-1) { InterlockedIncrement(&sc->api_fail); continue; }
                    if (sc->getctx) ctx_probe(sc, w);
                    SwitchToThread();
                    if (ResumeThread(w->h) == (DWORD)-1) InterlockedIncrement(&sc->resume_fail);
                    sc->susp_ops++;
                }
                Sleep(0);
            } else Sleep(10);
        }
        sc->stop = 1;
        if (sc->ev) for (int i = 0; i < sc->n; i++) SetEvent(sc->ev[i]);
        if (sc->pw) for (int i = 0; i < sc->n; i++) { DWORD wr; char b = 'S'; WriteFile(sc->pw[i], &b, 1, &wr, NULL); }
    }
    DWORD tstart = GetTickCount();
    for (int i = 0; i < sc->created; i++) {
        DWORD el = GetTickCount() - tstart, rem = el < JOIN_MS ? JOIN_MS - el : 0;
        DWORD r = WaitForSingleObject(sc->w[i].h, rem);
        if (r != WAIT_OBJECT_0) {
            /* a stuck worker would keep running into the next case and still uses sc: retire the process */
            InterlockedIncrement(&sc->hung);
            t_retire_process("thread_xstate: worker thread did not finish within the join budget (no TerminateThread)");
        }
    }
    sc->secs = t_qpc_sec(t0, t_qpc());
    SetThreadPriority(GetCurrentThread(), prio0 == THREAD_PRIORITY_ERROR_RETURN ? THREAD_PRIORITY_NORMAL : prio0);
}

typedef struct cfg {
    int op, n, pin, rf, prio_mix, susp, mig_ext, fixed_iters, getctx, mxflags;
    unsigned spin;
} cfg;

static void scen_free(scen *sc) {
    if (sc->w) {
        for (int i = 0; i < sc->n; i++) if (sc->w[i].h) CloseHandle(sc->w[i].h);
        big_free(sc->w);
    }
    if (sc->ev) { for (int i = 0; i < sc->n; i++) if (sc->ev[i]) CloseHandle(sc->ev[i]); free(sc->ev); }
    if (sc->pr) { for (int i = 0; i < sc->n; i++) { if (sc->pr[i]) CloseHandle(sc->pr[i]); if (sc->pw[i]) CloseHandle(sc->pw[i]); } free(sc->pr); free(sc->pw); }
    if (sc->timers) { for (int i = 0; i < sc->n; i++) if (sc->timers[i]) CloseHandle(sc->timers[i]); free(sc->timers); }
    if (sc->ev_never) CloseHandle(sc->ev_never);
    if (sc->mutex) CloseHandle(sc->mutex);
    if (sc->sem) CloseHandle(sc->sem);
    if (sc->cs_init) DeleteCriticalSection(&sc->cs);
    if (sc->syncbar_init && sc->dsb) sc->dsb(sc->syncbar);   /* releases the barrier's internal events */
    sc->syncbar_init = 0;
}

static void run_cfg(const cfg *c) {
    static scen S;          /* large; one case at a time */
    scen *sc = &S;
    memset(sc, 0, sizeof *sc);
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
#ifdef T_X86
    if (t_sde_model() && c->n > 256)   /* emulator artifact, README "Emulated validation"; not run: it kills the process */
        XFAIL("Intel SDE (%s): 32-bit Pin cannot allocate its per-thread data for %d threads (Pin assertion)", t_sde_model(), c->n);
#endif
    sc->op = c->op; sc->n = c->n; sc->pin = c->pin; sc->rf = c->rf; sc->prio_mix = c->prio_mix;
    sc->susp = c->susp; sc->mig_ext = c->mig_ext; sc->fixed_iters = c->fixed_iters;
    sc->getctx = c->getctx; sc->mxflags = c->mxflags;
    sc->spin = c->spin ? c->spin : (c->op == OP_NONE ? 30000u : 3000u);
    sc->ms = dur_ms(c->n);
    sc->kind = best_kind();
    sc->ncpu = cpu_masks(sc->cpus, 64);
    if (sc->ncpu < 1) { sc->ncpu = 1; sc->cpus[0] = 1; }
    sc->d.eng_n = (uint32_t)c->n;
    sc->d.main_tag = 5u * (uint32_t)c->n + 1u;
    sc->d.max_tag = 5u * (uint32_t)c->n + 2u;
    if ((c->rf == RF_MMX || c->rf == RF_MIX) && !cpu_has(F_MMX)) SKIP("MMX not available");
    if ((c->op == OP_MIGRATE || c->mig_ext) && sc->ncpu < 2) SKIP("needs >= 2 CPUs in the process affinity mask (have %d)", sc->ncpu);
    if (c->op == OP_WAITADDR) {
        sc->woa = (woa_t)t_proc("kernelbase.dll", "WaitOnAddress");
        sc->wba = (wba_t)t_proc("kernelbase.dll", "WakeByAddressSingle");
        if (!sc->woa || !sc->wba) SKIP("WaitOnAddress/WakeByAddressSingle not exported (pre-Windows 8)");
    }
    if (c->op == OP_SYNCBAR) {
        isb_t isb = (isb_t)t_proc("kernel32.dll", "InitializeSynchronizationBarrier");
        sc->esb = (esb_t)t_proc("kernel32.dll", "EnterSynchronizationBarrier");
        sc->dsb = (dsb_t)t_proc("kernel32.dll", "DeleteSynchronizationBarrier");
        if (!isb || !sc->esb || !sc->dsb) SKIP("SynchronizationBarrier API not exported (pre-Windows 8)");
        REQUIREF(isb(sc->syncbar, c->n, -1), "InitializeSynchronizationBarrier err %lu", GetLastError());
        sc->syncbar_init = 1;
    }
    if (c->op == OP_CBAR) { InitializeSRWLock(&sc->cb.l); InitializeConditionVariable(&sc->cb.cv); sc->cb.n = c->n; }

    sc->wbytes = sizeof(worker) * (size_t)c->n;
    sc->w = (worker *)big_alloc(sc->wbytes);
    REQUIREF(sc->w != NULL, "VirtualAlloc %zu bytes failed", sc->wbytes);
    for (int i = 0; i < c->n; i++) {
        worker *w = &sc->w[i];
        w->sc = sc; w->idx = i; w->kind = sc->kind;
        w->rf = c->rf;
        if (c->rf == RF_MIX) { static const int m[4] = { RF_VEC, RF_XMM_PEEK, RF_X87, RF_MMX }; w->rf = m[i % 4]; }
    }
    int fail_res = 0;
    if (c->op == OP_EVENT_RING || c->op == OP_WFMO || c->op == OP_SOAW) {
        sc->ev = (HANDLE *)calloc((size_t)c->n, sizeof(HANDLE));
        for (int i = 0; sc->ev && i < c->n; i++) if (!(sc->ev[i] = CreateEventW(NULL, FALSE, FALSE, NULL))) fail_res = 1;
        sc->ev_never = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (!sc->ev || !sc->ev_never) fail_res = 1;
    }
    if (c->op == OP_APC_WAITEX && !(sc->ev_never = CreateEventW(NULL, TRUE, FALSE, NULL))) fail_res = 1;
    if (c->op == OP_MUTEX && !(sc->mutex = CreateMutexW(NULL, FALSE, NULL))) fail_res = 1;
    if (c->op == OP_SEM) {
        sc->sem_cap = c->n / 4 > 1 ? c->n / 4 : 1;
        if (!(sc->sem = CreateSemaphoreW(NULL, sc->sem_cap, sc->sem_cap, NULL))) fail_res = 1;
    }
    if (c->op == OP_CONDVAR || c->op == OP_SRW) { InitializeSRWLock(&sc->srw); InitializeConditionVariable(&sc->cv); }
    if (c->op == OP_CRITSEC) { InitializeCriticalSectionAndSpinCount(&sc->cs, 0); sc->cs_init = 1; }
    if (c->op == OP_PIPE) {
        sc->pr = (HANDLE *)calloc((size_t)c->n, sizeof(HANDLE)); sc->pw = (HANDLE *)calloc((size_t)c->n, sizeof(HANDLE));
        for (int i = 0; sc->pr && sc->pw && i < c->n; i++) if (!CreatePipe(&sc->pr[i], &sc->pw[i], NULL, 4096)) fail_res = 1;
        if (!sc->pr || !sc->pw) fail_res = 1;
    }
    if (c->op == OP_TIMER) {
        sc->timers = (HANDLE *)calloc((size_t)c->n, sizeof(HANDLE));
        for (int i = 0; sc->timers && i < c->n; i++) if (!(sc->timers[i] = CreateWaitableTimerW(NULL, FALSE, NULL))) fail_res = 1;
        if (!sc->timers) fail_res = 1;
    }
    if (fail_res) { DWORD e = GetLastError(); scen_free(sc); SKIP("could not create %d synchronisation objects (err %lu)", c->n, e); }

    /* main (case) thread: own pattern in the registers + own FP environment for the whole scenario */
    static t_vregs min_, mout;
    unsigned mmx = mxcsr_variant(13); unsigned short mcw = cw_variant(7);
    t_vregs_pattern(&min_, sc->kind, sc->d.main_tag);
    t_set_mxcsr(mmx); t_set_x87_cw(mcw);
    t_vregs_call(sc->kind, &min_, &mout, run_body, sc);
    unsigned mx_after = t_get_mxcsr(); unsigned short cw_after = t_get_x87_cw();
    t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);

    if (sc->created < sc->n) {
        int created = sc->created; DWORD e = sc->create_err;
        scen_free(sc);
        SKIP("CreateThread failed after %d of %d threads (err %lu)", created, c->n, e);
    }
    CHECKF(((mx_after ^ mmx) & MX_CTRL) == 0, "main thread MXCSR 0x%04x after scenario, set 0x%04x", mx_after, mmx);
    CHECKF(cw_after == mcw, "main thread FCW 0x%04x after scenario, set 0x%04x", cw_after, mcw);
#ifdef T_X64
    t_vregs_check(&mout, &min_, T_VK_XMM, 6, 15, "main thread (nonvolatile) across the whole scenario");
#endif
    if (!sc->getctx) {
        /* (getctx: the controller itself copies the workers' context images through its own volatile
         * registers, so worker values there are expected; its XMM6-15 are still checked above) */
        unsigned long long ml = 0, ma = 0, mv = 0;
        verify_call(&sc->d, "main thread", sc->kind, -1, sc->d.main_tag, &mout, &min_, &ma, &ml, &mv);
        CHECKF(ml == 0, "main thread registers hold %llu foreign pattern qwords", ml);
    }

    unsigned long long it = 0, bs = 0, ba = 0, bl = 0, bm = 0, bc = 0, vc = 0, to = 0, ar = 0, cm = 0, sp = 0, bf = 0, minit = ~0ull;
    for (int i = 0; i < sc->n; i++) {
        worker *w = &sc->w[i];
        CHECKF(w->finished && !w->bad_spin && !w->bad_abi && !w->bad_leak && !w->bad_mx && !w->bad_cw && !w->bad_mxflags,
               "worker %d: finished=%ld iters=%llu spin_bad=%llu abi_bad=%llu leaked=%llu mxcsr_bad=%llu fcw_bad=%llu mxcsr_flags_bad=%llu",
               i, (long)w->finished, w->iters, w->bad_spin, w->bad_abi, w->bad_leak, w->bad_mx, w->bad_cw, w->bad_mxflags);
        it += w->iters; bs += w->bad_spin; ba += w->bad_abi; bl += w->bad_leak; bm += w->bad_mx; bc += w->bad_cw;
        bf += w->bad_mxflags;
        vc += w->vol_changed; to += w->timeouts; ar += w->apc_runs; cm += w->cpu_mismatch; sp += w->spins;
        if (w->iters < minit) minit = w->iters;
    }
    CHECKF(bs == 0, "%llu preemptible spins ended with corrupted registers (all registers, full width)", bs);
    CHECKF(ba == 0, "%llu calls of %s lost nonvolatile XMM6-15 state", ba, g_opname[c->op]);
    CHECKF(bl == 0, "%llu register qwords held another thread's pattern", bl);
    CHECKF(bm == 0, "%llu iterations lost MXCSR control bits", bm);
    CHECKF(bc == 0, "%llu iterations lost the x87 control word", bc);
    CHECKF(sc->hung == 0, "%ld workers did not finish within %u ms after stop (terminated)", (long)sc->hung, JOIN_MS);
    CHECKF(sc->api_fail == 0, "%ld unexpected API failures in %s", (long)sc->api_fail, g_opname[c->op]);
    CHECKF(sc->resume_fail == 0, "%ld ResumeThread failures", (long)sc->resume_fail);
    CHECKF(sc->aff_fail == 0, "%ld SetThreadAffinityMask failures", (long)sc->aff_fail);
    if (c->op == OP_MUTEX || c->op == OP_SEM || c->op == OP_CONDVAR || c->op == OP_SRW || c->op == OP_CRITSEC)
        CHECKF(sc->excl_viol == 0, "%ld mutual-exclusion violations in %s", (long)sc->excl_viol, g_opname[c->op]);
    if (sc->mxflags) CHECKF(bf == 0, "%llu preemptible spins changed the MXCSR status flags", bf);
    if (sc->getctx) {
        CHECKF(sc->ctx_fail == 0, "GetThreadContext of a suspended worker failed %llu times", sc->ctx_fail);
        CHECKF(sc->ctx_leak == 0, "%llu XMM qwords in GetThreadContext results held another thread's pattern", sc->ctx_leak);
        /* method sanity: workers spend nearly all their time inside the spin with the pattern loaded */
        if (sc->ctx_n >= 50)
            CHECKF(sc->ctx_own > 0, "no GetThreadContext result (of %llu) showed the worker's own XMM pattern", sc->ctx_n);
        t_info("contexts_captured", "%llu", sc->ctx_n);
        t_info("contexts_with_own_pattern", "%llu", sc->ctx_own);
    }
    if (c->op == OP_APC || c->op == OP_APC_WAITEX) {
        CHECKF(sc->apc_wrong == 0, "%ld APCs ran on a thread other than the target", (long)sc->apc_wrong);
        CHECKF(ar > 0, "no APC was delivered during alertable SleepEx");
        t_info("apc_runs", "%llu", ar);
    }
    if (c->op == OP_SYNCBAR || c->op == OP_CBAR) {
        CHECKF(sc->bar_viol == 0, "%ld barrier phase violations", (long)sc->bar_viol);
        CHECKF(!sc->cb.broken, "custom barrier broke (participant missing for 20 s)");
        CHECKF(it == (unsigned long long)c->fixed_iters * (unsigned long long)c->n, "phases completed %llu, want %llu",
               it, (unsigned long long)c->fixed_iters * (unsigned long long)c->n);
    }
    if (c->op == OP_MIGRATE) {
        if (cm && t_is_wine()) t_expect_fail("wine: GetCurrentProcessorNumberEx right after SetThreadAffinityMask(self) (%llu mismatches)", cm);
        CHECKF(cm == 0, "%llu times the thread ran outside its new affinity mask right after SetThreadAffinityMask", cm);
    }
    t_info("workers", "%d", c->n);
    t_info("kind", "%s", kind_name(sc->kind));
    t_info("op", "%s", g_opname[c->op]);
    t_info("cpus", "%d", sc->ncpu);
    t_info("iterations", "%llu", it);
    t_info("min_worker_iterations", "%llu", minit);
    t_info("spin_chunks", "%llu", sp);
    t_info("wait_timeouts", "%llu", to);
    t_info("volatile_regs_changed_by_call", "%llu", vc);
    if (sc->mig_ext) t_info("external_migrations", "%llu", sc->mig_ops);
    if (sc->susp) { t_info("suspend_resume_ops", "%llu", sc->susp_ops); CHECKF(sc->susp_ops > 0, "no SuspendThread/ResumeThread performed"); }
    if (sc->secs > 0) t_perf("iterations_per_s", (double)it / sc->secs, "iter/s");
    scen_free(sc);
}

#define ISO(name, ...) TEST(name) { cfg c_ = { __VA_ARGS__ }; run_cfg(&c_); }

/* ======================================================================================
 * 0. self-test of the leak decoder (pure computation: proves a foreign value would be recognised)
 * ==================================================================================== */
TEST(selftest_leak_decoder) {
    static vdom d; memset((void *)&d, 0, sizeof d);
    d.eng_n = 16; d.main_tag = 81; d.max_tag = 82; d.hold_base = 0; d.hold_n = 0;
    __attribute__((aligned(64))) uint8_t r[64];
    uint64_t fv = 0; uint32_t ft = 0; size_t fo = 0;
    t_pattern_reg(r, 64, 1u + 3u + 16u * 2u, 9);              /* worker 3, phase 2, register 9 */
    CHECK_EQ(scan_foreign(&d, 3, 1u + 3u, r, 64, &fv, &ft, &fo), 0);          /* own (older phase) */
    CHECK_EQ(scan_foreign(&d, 4, 1u + 4u, r, 64, &fv, &ft, &fo), 8);          /* another worker */
    CHECK_EQ(ft, 1u + 3u + 32u);
    CHECK_EQ(fo, 0);
    CHECK_EQ(scan_foreign(&d, -1, d.main_tag, r, 64, &fv, &ft, &fo), 8);      /* the main thread */
    t_pattern_reg(r, 64, 1u + 64u + 5u, 2);                    /* APC tag of worker 5 */
    CHECK_EQ(scan_foreign(&d, 5, 1u + 5u, r, 64, &fv, &ft, &fo), 0);
    CHECK_EQ(scan_foreign(&d, 6, 1u + 6u, r, 64, &fv, &ft, &fo), 8);
    t_pattern_reg(r, 64, d.main_tag, 0);
    CHECK_EQ(scan_foreign(&d, 0, 1u, r, 64, &fv, &ft, &fo), 8);
    memset(r, 0, sizeof r);
    CHECK_EQ(scan_foreign(&d, 0, 1u, r, 64, &fv, &ft, &fo), 0);               /* zero is not a pattern */
    char b[96];
    dom_describe(&d, 1u + 3u + 32u, b, sizeof b); CHECK(!strcmp(b, "worker 3 (phase 2)"));
    dom_describe(&d, 81, b, sizeof b);            CHECK(!strcmp(b, "the main (case) thread"));
    CHECK_EQ(mxcsr_variant(0), 0x1F80);
    CHECK_EQ(mxcsr_variant(3) & MX_CTRL, 0x7F80);
    CHECK_EQ(cw_variant(0), 0x007F);
    CHECK_EQ(cw_variant(1), 0x027F);
    for (unsigned k = 0; k < 24; k++) {                        /* every variant round-trips through the CPU */
        unsigned mx = mxcsr_variant(k); unsigned short cw = cw_variant(k);
        t_set_mxcsr(mx); t_set_x87_cw(cw);
        unsigned mx2 = t_get_mxcsr(); unsigned short cw2 = t_get_x87_cw();
        t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
        CHECKF(mx2 == mx && cw2 == cw, "variant %u: MXCSR 0x%04x->0x%04x FCW 0x%04x->0x%04x", k, mx, mx2, cw, cw2);
    }
    t_info("daz_supported", "%d", daz_ok());
}

/* ======================================================================================
 * 1. initial FP environment of new threads, no inheritance, no leak
 * ==================================================================================== */
TEST(init_new_thread_fp_env_default) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static vdom d; memset((void *)&d, 0, sizeof d); d.max_tag = 64;
    int kind = best_kind();
    capstats s; memset(&s, 0, sizeof s);
    for (int i = 0; i < 16; i++) {
        cap *c = cap_new(); REQUIRE(c);
        DWORD err = 0;
        if (!cap_run(c, kind, 0, &err)) { _aligned_free(c); SKIP("CreateThread failed (err %lu)", err); }
        char lab[64]; snprintf(lab, sizeof lab, "new thread #%d", i);
        cap_analyse(c, &d, kind, &s, lab);
        _aligned_free(c);
    }
    capstats_report(&s, "default");
}

TEST(init_new_thread_fp_env_not_inherited) {
    /* The creator's MXCSR / x87 CW (RC, FTZ, DAZ, precision) must not be copied into the new thread. */
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static vdom d; memset((void *)&d, 0, sizeof d); d.max_tag = 64;
    int kind = best_kind();
    capstats s; memset(&s, 0, sizeof s);
    for (unsigned v = 1; v < 16; v++) {
        cap *c = cap_new(); REQUIRE(c);
        /* creator: exceptions stay masked (no faults), RC/FTZ/DAZ vary, sticky flags set on odd variants */
        unsigned pmx = mxcsr_variant(v) | (v & 1u ? 0x3Fu : 0);
        unsigned short pcw = cw_variant(v + 1);
        t_set_mxcsr(pmx); t_set_x87_cw(pcw);
        DWORD err = 0;
        int ok = cap_run(c, kind, 0, &err);
        unsigned mx2 = t_get_mxcsr(); unsigned short cw2 = t_get_x87_cw();
        t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
        if (!ok) { _aligned_free(c); SKIP("CreateThread failed (err %lu)", err); }
        CHECKF(((mx2 ^ pmx) & MX_CTRL) == 0 && cw2 == pcw, "creator FP env changed by CreateThread/Wait: MXCSR 0x%04x (0x%04x) FCW 0x%04x (0x%04x)",
               mx2, pmx, cw2, pcw);
        char lab[96]; snprintf(lab, sizeof lab, "thread created under MXCSR 0x%04x FCW 0x%04x", pmx, pcw);
        cap_analyse(c, &d, kind, &s, lab);
        _aligned_free(c);
    }
    capstats_report(&s, "creator variants");
}

TEST(init_new_thread_regs_no_leak_from_spinners) {
    /* Holder threads keep distinctive patterns live in every register while new threads start on the
     * same CPUs: a lazily / wrongly restored context would hand their values to the new thread. */
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static vdom d; memset((void *)&d, 0, sizeof d);
    int kind = best_kind();
    DWORD_PTR cpus[64]; int ncpu = cpu_masks(cpus, 64); if (ncpu < 1) ncpu = 1;
    int k = ncpu * 2; if (k < 2) k = 2; if (k > 32) k = 32;
    d.hold_base = 100; d.hold_n = (uint32_t)k; d.main_tag = 300; d.max_tag = 301;
    holder *hs = holders_alloc(k); REQUIRE(hs);
    static volatile LONG stop; stop = 0;
    int started = holders_start(hs, k, &d, kind, &stop, 0);
    capstats s; memset(&s, 0, sizeof s);
    int fails = 0; DWORD err = 0;
    for (int i = 0; i < 200 && started == k; i++) {
        cap *c = cap_new(); if (!c) break;
        if (!cap_run(c, kind, (i & 1) ? cpus[i % ncpu] : 0, &err)) { fails++; _aligned_free(c); break; }
        char lab[64]; snprintf(lab, sizeof lab, "new thread #%d", i);
        cap_analyse(c, &d, kind, &s, lab);
        _aligned_free(c);
    }
    holders_stop(hs, started, &stop);
    _aligned_free(hs);
    if (started < k || fails) SKIP("CreateThread failed (holders %d/%d, err %lu)", started, k, err);
    t_info("holders", "%d", k);
    capstats_report(&s, "with spinning holders");
}

/* thread that loads a pattern + dirty FP env + non-empty x87 stack and exits without cleaning up */
static DWORD WINAPI dirty_thread(void *p) {
    const dirty *dd = (const dirty *)p;
    if (dd->kind == T_VK_ZMM) te_dirty_exit_zmm_avx512(dd);
    else if (dd->kind == T_VK_YMM) te_dirty_exit_ymm_avx(dd);
    else te_dirty_exit_xmm(dd);
}
static unsigned dirty_mxcsr(void) { return (0xFFFFu & ~0x40u) | (daz_ok() ? 0x40u : 0); }   /* RC=11, FTZ, flags set, masked */

TEST(init_new_thread_after_dirty_exit) {
    /* A thread exits (ExitThread) with patterns in every vector register, MXCSR = RC/FTZ/(DAZ)+all flags,
     * FCW 0x0F7F and three values on the x87 stack. The next thread on the same CPU must start clean. */
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static vdom d; memset((void *)&d, 0, sizeof d);
    int kind = best_kind();
    d.dirty_base = 1000; d.dirty_n = 64; d.max_tag = 1064;
    DWORD_PTR cpus[64]; int ncpu = cpu_masks(cpus, 64); if (ncpu < 1) { ncpu = 1; cpus[0] = 1; }
    dirty *dd = (dirty *)_aligned_malloc(sizeof(dirty), 64); REQUIRE(dd);
    capstats s; memset(&s, 0, sizeof s);
    for (int i = 0; i < 64; i++) {
        memset(dd, 0, sizeof *dd);
        t_vregs_pattern(&dd->r, kind, d.dirty_base + (uint32_t)i);
        dd->mx = dirty_mxcsr(); dd->cw = 0x0F7F; dd->kind = kind;
        DWORD_PTR pin = (i & 1) ? 0 : cpus[(i / 2) % ncpu];
        HANDLE h = mk_thread(dirty_thread, dd, CREATE_SUSPENDED, NULL);
        if (!h) { DWORD e = GetLastError(); _aligned_free(dd); SKIP("CreateThread failed (err %lu)", e); }
        if (pin) SetThreadAffinityMask(h, pin);
        ResumeThread(h);
        DWORD w = WaitForSingleObject(h, 10000);
        DWORD ec = 0xFFFFFFFF; GetExitCodeThread(h, &ec);
        CloseHandle(h);
        CHECKF(w == WAIT_OBJECT_0 && ec == 0, "dirty thread #%d: wait 0x%lx exit code 0x%lx", i, w, ec);
        cap *c = cap_new(); REQUIRE(c);
        DWORD err = 0;
        if (!cap_run(c, kind, pin, &err)) { _aligned_free(c); _aligned_free(dd); SKIP("CreateThread failed (err %lu)", err); }
        char lab[64]; snprintf(lab, sizeof lab, "thread after dirty exit #%d", i);
        cap_analyse(c, &d, kind, &s, lab);
        _aligned_free(c);
    }
    _aligned_free(dd);
    capstats_report(&s, "after dirty ExitThread");
}

typedef struct spinner { t_vregs in, out; int kind; volatile LONG started; } spinner;
static volatile LONG g_never;
static DWORD WINAPI spinner_main(void *p) {
    spinner *s = (spinner *)p;
    t_set_mxcsr(dirty_mxcsr()); t_set_x87_cw(0x0F7F);
    s->started = 1;
    for (;;) t_vregs_spin(s->kind, &s->in, &s->out, &g_never, 0x7FFFFFFFu);
    return 0;
}
TEST(init_new_thread_after_terminate_spinning) {
    /* TerminateThread on a thread that holds live register patterns, then start a fresh thread on the
     * same CPU: the terminated state must be discarded, not inherited. */
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static vdom d; memset((void *)&d, 0, sizeof d);
    int kind = best_kind();
    d.dirty_base = 2000; d.dirty_n = 32; d.max_tag = 2032;
    DWORD_PTR cpus[64]; int ncpu = cpu_masks(cpus, 64); if (ncpu < 1) { ncpu = 1; cpus[0] = 1; }
    spinner *sp = (spinner *)_aligned_malloc(sizeof(spinner), 64); REQUIRE(sp);
    capstats s; memset(&s, 0, sizeof s);
    for (int i = 0; i < 32; i++) {
        memset(sp, 0, sizeof *sp);
        sp->kind = kind;
        t_vregs_pattern(&sp->in, kind, d.dirty_base + (uint32_t)i);
        DWORD_PTR pin = cpus[i % ncpu];
        HANDLE h = mk_thread(spinner_main, sp, CREATE_SUSPENDED, NULL);
        if (!h) { DWORD e = GetLastError(); _aligned_free(sp); SKIP("CreateThread failed (err %lu)", e); }
        SetThreadAffinityMask(h, pin);
        ResumeThread(h);
        for (int k = 0; k < 2000 && !sp->started; k++) Sleep(1);
        Sleep(2);
        CHECKF(TerminateThread(h, 0xDEADu), "TerminateThread err %lu", GetLastError());
        DWORD w = WaitForSingleObject(h, 10000);
        DWORD ec = 0; GetExitCodeThread(h, &ec);
        CloseHandle(h);
        CHECKF(w == WAIT_OBJECT_0 && ec == 0xDEADu, "terminated thread #%d: wait 0x%lx exit code 0x%lx", i, w, ec);
        cap *c = cap_new(); REQUIRE(c);
        DWORD err = 0;
        if (!cap_run(c, kind, pin, &err)) { _aligned_free(c); _aligned_free(sp); SKIP("CreateThread failed (err %lu)", err); }
        char lab[64]; snprintf(lab, sizeof lab, "thread after TerminateThread #%d", i);
        cap_analyse(c, &d, kind, &s, lab);
        _aligned_free(c);
    }
    _aligned_free(sp);
    capstats_report(&s, "after TerminateThread");
}

TEST(init_new_thread_create_remote_thread_self) {
    /* Same contract through the other documented creation path: CreateRemoteThreadEx on the current
     * process (kernelbase builds the initial context itself, then NtCreateThreadEx). The creator runs
     * with a non-default FP environment. */
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static vdom d; memset((void *)&d, 0, sizeof d); d.max_tag = 64;
    int kind = best_kind();
    capstats s; memset(&s, 0, sizeof s);
    for (unsigned v = 0; v < 16; v++) {
        cap *c = cap_new(); REQUIRE(c);
        unsigned pmx = mxcsr_variant(v + 1) | 0x3Fu; unsigned short pcw = cw_variant(v + 2);
        t_set_mxcsr(pmx); t_set_x87_cw(pcw);
        DWORD tid = 0;
        HANDLE h = CreateRemoteThreadEx(GetCurrentProcess(), NULL, TX_STACK, cap_entry(kind), c,
                                        STACK_SIZE_PARAM_IS_A_RESERVATION, NULL, &tid);
        DWORD e = GetLastError();
        DWORD w = h ? WaitForSingleObject(h, 10000) : WAIT_FAILED;
        unsigned mx2 = t_get_mxcsr(); unsigned short cw2 = t_get_x87_cw();
        t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
        if (!h) { _aligned_free(c); SKIP("CreateRemoteThreadEx(self) failed (err %lu)", e); }
        DWORD ec = 0; GetExitCodeThread(h, &ec);
        CloseHandle(h);
        CHECKF(w == WAIT_OBJECT_0 && ec == 0x600Du, "remote-created thread #%u: wait 0x%lx exit code 0x%lx", v, w, ec);
        CHECKF(((mx2 ^ pmx) & MX_CTRL) == 0 && cw2 == pcw, "creator FP env changed: MXCSR 0x%04x (0x%04x) FCW 0x%04x (0x%04x)",
               mx2, pmx, cw2, pcw);
        char lab[64]; snprintf(lab, sizeof lab, "CreateRemoteThreadEx thread #%u", v);
        cap_analyse(c, &d, kind, &s, lab);
        _aligned_free(c);
    }
    capstats_report(&s, "CreateRemoteThreadEx(self)");
}

/* ======================================================================================
 * 1b. special user-mode APC (Windows 11 QueueUserAPC2 + QUEUE_USER_APC_FLAGS_SPECIAL_USER_APC):
 *     delivered ASYNCHRONOUSLY to a thread that is not alertable - here a thread in the middle of a
 *     GPR-only spin with live patterns. The APC routine loads its own pattern into every register.
 *     Hard: the interrupted thread's integer-visible legacy state (XMM0-15 low 128 bits on x64,
 *     XMM0-7 on x86, MXCSR, x87 CW) is back exactly, no register ever holds another thread's pattern,
 *     the APC runs on its target. Upper YMM/ZMM bits and XMM16-31/k: Microsoft documents nothing
 *     specific for asynchronous APCs, so a loss is recorded as INFO (a FOREIGN value is still a FAIL).
 * ==================================================================================== */
typedef struct sapc {
    t_vregs in, out, ain, aout;
    vdom *d;
    int idx, kind;
    HANDLE h; DWORD tid;
    volatile LONG *stop;
    volatile LONG apc_runs, apc_wrong, apc_bad;
    unsigned long long chunks, bad_low, bad_ext, bad_leak, bad_env, ext_apc_value;
    unsigned mx; unsigned short cw;
    void *hj_buf[64];                  /* emulated special APC: one saved-context buffer per hijack (no reuse) */
    int hj_n;
    volatile LONG hj_entered;
} sapc;
static void CALLBACK sapc_fn(ULONG_PTR p) {
    sapc *s = (sapc *)p;
    if (GetCurrentThreadId() != s->tid) { InterlockedIncrement(&s->apc_wrong); return; }
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    t_set_mxcsr(mxcsr_variant((unsigned)s->idx + 9u)); t_set_x87_cw(cw_variant((unsigned)s->idx + 4u));
    t_vregs_spin(s->kind, &s->ain, &s->aout, NULL, 50);
    int n = t_vregs_count(s->kind);
    for (int i = 0; i < n; i++) if (memcmp(s->ain.v[i], s->aout.v[i], (size_t)s->kind)) { InterlockedIncrement(&s->apc_bad); break; }
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    InterlockedIncrement(&s->apc_runs);
}
static DWORD WINAPI sapc_main(void *p) {
    sapc *s = (sapc *)p;
    t_set_mxcsr(s->mx); t_set_x87_cw(s->cw);
    unsigned n = (unsigned)s->d->eng_n;
    int nr = t_vregs_count(s->kind);
    for (unsigned long long it = 0; !*s->stop; it++) {
        uint32_t tag = 1u + (uint32_t)s->idx + n * (uint32_t)(it & 3);
        t_vregs_pattern(&s->in, s->kind, tag);
        t_vregs_spin(s->kind, &s->in, &s->out, s->stop, 20000);
        if (((t_get_mxcsr() ^ s->mx) & MX_CTRL) || t_get_x87_cw() != s->cw) {
            s->bad_env++;
            t_set_mxcsr(s->mx); t_set_x87_cw(s->cw);
        }
        s->chunks++;
        for (int i = 0; i < nr; i++) {
            uint64_t fv = 0; uint32_t ft = 0; size_t fo = 0;
            int f = scan_foreign(s->d, s->idx, tag, s->out.v[i], (size_t)s->kind, &fv, &ft, &fo);
            if (f) { s->bad_leak += (unsigned long long)f; report_leak(s->d, "spinner interrupted by special APCs", kind_name(s->kind), i, fv, ft, fo); }
            int legacy = i < T_NXMM;            /* XMM0-15 (x64) / XMM0-7 (x86): in every CONTEXT */
            if (legacy && memcmp(s->out.v[i], s->in.v[i], 16)) {
                s->bad_low++;
                report_reg(s->d, "spinner interrupted by special APCs", "xmm", i, s->out.v[i], s->in.v[i], 16);
            } else if (memcmp(s->out.v[i], s->in.v[i], (size_t)s->kind)) {
                s->bad_ext++;
                uint64_t v; memcpy(&v, s->out.v[i] + (legacy ? 16 : 0), 8);
                uint32_t tg, r, wd;
                if (t_pattern_explain(v, s->d->max_tag, &tg, &r, &wd) && tg == 1u + 4u * n + (uint32_t)s->idx) s->ext_apc_value++;
            }
        }
        if (s->kind == T_VK_ZMM) {
            uint64_t m = cpu_has(F_AVX512BW) ? ~0ull : 0xFFFFull;
            for (int i = 0; i < 8; i++) if ((s->out.k[i] ^ s->in.k[i]) & m) s->bad_ext++;
        }
    }
    t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
    return 0;
}
/* Emulated special user APC (self-check path that needs no QueueUserAPC2): what the kernel does for a special
 * user APC is to interrupt the thread wherever it is in user mode, run KiUserApcDispatcher -> the routine on the
 * thread's own stack, and resume the interrupted context with NtContinue. The emulation does the same from user
 * mode: SuspendThread; GetThreadContext(FULL | FLOATING_POINT [| EXTENDED_REGISTERS on x86] [| XSTATE]) into a
 * buffer used only once; SetThreadContext(CONTROL | INTEGER) with the IP at sapc_hijack_entry, the SP 512 bytes
 * below the interrupted one, the arguments (s, saved context); ResumeThread. The entry runs the same routine as
 * the real APC and NtContinue(saved, FALSE). Every check of the real case applies unchanged, so a PASS here
 * validates the detection logic and the Get/SetThreadContext + NtContinue round trip of an asynchronously
 * interrupted non-alertable thread; the QueueUserAPC2 delivery itself still needs the case above. */
typedef LONG (NTAPI *sapc_ntcontinue_t)(PCONTEXT, BOOLEAN);
static sapc_ntcontinue_t g_sapc_ntcontinue;
static void __cdecl sapc_hijack_entry(sapc *s, CONTEXT *c) {
    InterlockedIncrement(&s->hj_entered);
    sapc_fn((ULONG_PTR)s);
    g_sapc_ntcontinue(c, FALSE);
    ExitThread(0xDEADu);               /* NtContinue does not return on success */
}
#ifdef T_X64
#define SAPC_HJ_FLAGS (CONTEXT_FULL | CONTEXT_FLOATING_POINT)
#define SAPC_CONTEXT_XSTATE 0x100040u
#else
#define SAPC_HJ_FLAGS (CONTEXT_FULL | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS)
#define SAPC_CONTEXT_XSTATE 0x10040u
#endif
/* returns 1 hijacked, 0 skipped (pool exhausted / thread gone), -1 a context API failed (err in *err) */
static int sapc_hijack(sapc *s, DWORD64 xmask, DWORD *err) {
    static T_ALIGN(64) CONTEXT nc;
    if (s->hj_n >= (int)T_COUNTOF(s->hj_buf)) return 0;
    __typeof__(&InitializeContext) ic = xmask ? T_PROC("kernel32.dll", InitializeContext) : NULL;
    __typeof__(&SetXStateFeaturesMask) sm = xmask ? T_PROC("kernel32.dll", SetXStateFeaturesMask) : NULL;
    int xs = ic && sm;   /* not (flags & SAPC_CONTEXT_XSTATE): that shares the architecture bit with every CONTEXT_* */
    DWORD flags = SAPC_HJ_FLAGS | (xs ? SAPC_CONTEXT_XSTATE : 0), len = 0;
    void *buf; CONTEXT *c = NULL;
    if (xs) {
        ic(NULL, flags, NULL, &len);
        buf = len ? calloc(1, len) : NULL;
        if (!buf || !ic(buf, flags, &c, &len) || !c || !sm(c, xmask)) { free(buf); *err = GetLastError(); return -1; }
    } else {
        buf = _aligned_malloc(sizeof(CONTEXT), 64);
        if (!buf) return 0;
        memset(buf, 0, sizeof(CONTEXT)); c = (CONTEXT *)buf; c->ContextFlags = flags;
    }
    s->hj_buf[s->hj_n++] = buf;
    if (SuspendThread(s->h) == (DWORD)-1) { *err = GetLastError(); return 0; }
    int r = 1;
    if (!GetThreadContext(s->h, c)) { *err = GetLastError(); r = -1; }
    else {
        memcpy(&nc, c, sizeof nc);
        nc.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
#ifdef T_X64
        uintptr_t sp = ((uintptr_t)c->Rsp - 512u) & ~(uintptr_t)15;
        sp -= 8; *(uintptr_t *)sp = 0;                  /* entry: RSP = 8 mod 16, [RSP] = return address */
        nc.Rsp = sp; nc.Rip = (DWORD64)(uintptr_t)sapc_hijack_entry;
        nc.Rcx = (DWORD64)(uintptr_t)s; nc.Rdx = (DWORD64)(uintptr_t)c;
#else
        uintptr_t sp = ((uintptr_t)c->Esp - 512u) & ~(uintptr_t)15;
        sp -= 12; ((uintptr_t *)sp)[0] = 0; ((uintptr_t *)sp)[1] = (uintptr_t)s; ((uintptr_t *)sp)[2] = (uintptr_t)c;
        nc.Esp = (DWORD)sp; nc.Eip = (DWORD)(uintptr_t)sapc_hijack_entry;
#endif
        if (!SetThreadContext(s->h, &nc)) { *err = GetLastError(); r = -1; }
    }
    ResumeThread(s->h);
    return r;
}
static void sapc_case(int emulate) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    qapc2_t q2 = NULL;
    DWORD64 xmask = 0;
    if (!emulate) {
        q2 = (qapc2_t)t_proc("kernel32.dll", "QueueUserAPC2");
        if (!q2) q2 = (qapc2_t)t_proc("kernelbase.dll", "QueueUserAPC2");
        if (!q2) SKIP("QueueUserAPC2 not exported (special user APCs need Windows 11, build 22000, or Server 2022, build 20348); "
                      "special_user_apc_emulated_context_hijack covers the restore path");
    } else {
        g_sapc_ntcontinue = (sapc_ntcontinue_t)t_proc("ntdll.dll", "NtContinue");
        if (!g_sapc_ntcontinue) SKIP("ntdll!NtContinue not exported");
        if (cpu()->osxsave) xmask = t_env_get()->api_xstate_features & cpu()->xcr0 & ~3ull;
    }
    int kind = best_kind();
    DWORD_PTR cpus[64]; int ncpu = cpu_masks(cpus, 64); if (ncpu < 1) ncpu = 1;
    int k = ncpu + 1; if (k < 2) k = 2; if (k > 8) k = 8;
    static vdom d; memset((void *)&d, 0, sizeof d);
    d.eng_n = (uint32_t)k; d.main_tag = 5u * (uint32_t)k + 1u; d.max_tag = d.main_tag + 1u;
    sapc *ss = (sapc *)_aligned_malloc(sizeof(sapc) * (size_t)k, 64); REQUIRE(ss);
    memset(ss, 0, sizeof(sapc) * (size_t)k);
    static volatile LONG stop; stop = 0;
    int started = 0;
    for (int i = 0; i < k; i++) {
        sapc *s = &ss[i];
        s->d = &d; s->idx = i; s->kind = kind; s->stop = &stop;
        s->mx = mxcsr_variant((unsigned)i + 1u); s->cw = cw_variant((unsigned)i + 2u);
        t_vregs_pattern(&s->ain, kind, 1u + 4u * (uint32_t)k + (uint32_t)i);
        s->h = mk_thread(sapc_main, s, CREATE_SUSPENDED, &s->tid);
        if (!s->h) break;
        ResumeThread(s->h);
        started++;
    }
    /* queue from the case thread for a bounded time; each spinner gets one special APC per round */
    unsigned long long queued = 0; DWORD qerr = 0; int unsupported = 0, hj_fail = 0;
    if (started == k) {
        uint64_t t0 = t_qpc();
        while (!unsupported && t_qpc_ms(t0, t_qpc()) < (double)dur_ms(16)) {
            for (int i = 0; i < k; i++) {
                if (emulate) {
                    int r = sapc_hijack(&ss[i], xmask, &qerr);
                    if (r > 0) queued++;
                    else if (r < 0) { hj_fail++; if (!queued) unsupported = 1; }
                } else if (q2(sapc_fn, ss[i].h, (ULONG_PTR)&ss[i], 1 /* QUEUE_USER_APC_FLAGS_SPECIAL_USER_APC */)) queued++;
                else { qerr = GetLastError(); if (!queued) unsupported = 1; }
            }
            Sleep(emulate ? 3 : 1);
        }
    }
    stop = 1;
    int hung = 0;
    unsigned long long entered = 0;
    for (int i = 0; i < started; i++) {
        if (WaitForSingleObject(ss[i].h, 10000) != WAIT_OBJECT_0) { hung++; t_retire_process("thread_xstate: APC spinner did not finish within 10 s (no TerminateThread)"); }
        DWORD ec = 0; GetExitCodeThread(ss[i].h, &ec);
        if (emulate) CHECKF(ec != 0xDEADu, "spinner %d: NtContinue of the saved context returned (emulated APC)", i);
        CloseHandle(ss[i].h);
    }
    for (int i = 0; i < k; i++) {
        entered += (unsigned long long)ss[i].hj_entered;
        for (int j = 0; j < ss[i].hj_n; j++) {
            if (xmask && T_PROC("kernel32.dll", InitializeContext)) free(ss[i].hj_buf[j]); else _aligned_free(ss[i].hj_buf[j]);
        }
    }
    if (started < k) { _aligned_free(ss); SKIP("CreateThread failed after %d of %d spinners", started, k); }
    if (unsupported && !emulate) {
        _aligned_free(ss);
        SKIP("QueueUserAPC2(SPECIAL_USER_APC) rejected (err %lu%s): special user APCs need a native x64 process on Windows 11 (build "
             "22000) or Server 2022 (build 20348); special_user_apc_emulated_context_hijack covers the restore path", qerr,
             qerr == ERROR_NOT_SUPPORTED ? ", ERROR_NOT_SUPPORTED" : "");
    }
    CHECKF(!unsupported, "context hijack failed from the start: Suspend/Get/SetThreadContext error %lu", qerr);
    if (emulate) {
        CHECKF(hj_fail == 0, "%d Get/SetThreadContext calls failed during the emulated APCs (last error %lu)", hj_fail, qerr);
        CHECKF(entered == queued, "%llu hijacks, %llu entries into the emulated APC", queued, entered);
        t_info("hijack_context", "flags 0x%lx xstate_mask 0x%llx", (unsigned long)(SAPC_HJ_FLAGS | (xmask ? SAPC_CONTEXT_XSTATE : 0)),
               (unsigned long long)xmask);
    }
    unsigned long long runs = 0, wrong = 0, abad = 0, low = 0, ext = 0, leak = 0, env = 0, chunks = 0, extapc = 0;
    for (int i = 0; i < k; i++) {
        sapc *s = &ss[i];
        runs += (unsigned long long)s->apc_runs; wrong += (unsigned long long)s->apc_wrong; abad += (unsigned long long)s->apc_bad;
        low += s->bad_low; ext += s->bad_ext; leak += s->bad_leak; env += s->bad_env; chunks += s->chunks; extapc += s->ext_apc_value;
    }
    _aligned_free(ss);
    CHECKF(hung == 0, "%d spinners did not stop", hung);
    CHECKF(qerr == 0, "QueueUserAPC2 failed after %llu successful calls (err %lu)", queued, qerr);
    CHECKF(wrong == 0, "%llu special APCs ran on a thread other than the target", wrong);
    CHECKF(abad == 0, "%llu special APC routines saw their own registers corrupted", abad);
    CHECKF(leak == 0, "%llu register qwords held another thread's pattern", leak);
    CHECKF(low == 0, "%llu legacy XMM registers of the interrupted spin not restored (low 128 bits)", low);
    CHECKF(env == 0, "%llu spin chunks with MXCSR control / x87 CW changed by an asynchronous APC", env);
    if (runs == 0 && !emulate && t_is_wine()) t_expect_fail("wine: special user APC not delivered to a non-alertable thread");
    CHECKF(runs > 0, "no special user APC was delivered to a spinning (non-alertable) thread (%llu queued)", queued);
    if (emulate) CHECKF(runs == queued, "%llu emulated APCs completed of %llu hijacks", runs, queued);
    t_info("kind", "%s", kind_name(kind));
    t_info("spinners", "%d", k);
    t_info("apcs_queued", "%llu", queued);
    t_info("apcs_run", "%llu", runs);
    t_info("spin_chunks", "%llu", chunks);
    t_info("extended_state_mismatches", "%llu (upper YMM/ZMM, XMM16-31, k: not hard-checked)", ext);
    t_info("extended_state_holding_apc_values", "%llu", extapc);
    if (ext) t_note("extended (non-legacy) vector state of the interrupted thread changed across special APCs: %llu registers", ext);
}
TEST(special_user_apc_async_interrupt) { sapc_case(0); }
/* The hijack suspends and re-points live threads of the process: on Windows 10 it takes the whole exe down in about half of the boots
 * (0xC0000005, once STATUS_GUARD_PAGE_VIOLATION). It therefore runs in a child process (this exe re-run on the manual inner case);
 * a crash becomes a recorded outcome and the rest of the exe survives. */
TEST_MANUAL(sapc_hijack_inner_manual) { sapc_case(1); }
TEST(special_user_apc_emulated_context_hijack) {
    static t_child_result r;
    REQUIREF(t_run_self("--include-manual --filter=sapc_hijack_inner_manual", 180000, &r), "spawn failed: error %lu", r.spawn_error);
    if (r.timed_out) {
        t_note("emulated context hijack child hung and was killed: %.200s", r.out);
        XFAIL("unstable on Windows 10: emulated context-hijack child hung (killed)");
    }
    if (r.exit_code != 0 || r.exc_code != 0) {
        t_note("emulated context hijack child ended with exit 0x%08lx exc 0x%08lx at 0x%llx: %.300s", r.exit_code, r.exc_code, r.exc_addr, r.out);
        XFAIL("unstable on Windows 10: emulated context-hijack child crashed (exit 0x%08lx, exc 0x%08lx)", r.exit_code, r.exc_code);
    }
    if (strstr(r.out, "SKIP")) t_note("child output reports a skip: %.200s", r.out);
    t_info("hijack_child_exit", "0x%08lx", r.exit_code);
}

/* ======================================================================================
 * 2. thread churn: create + join 1000 threads; parent state unchanged, children start clean
 * ==================================================================================== */
typedef struct churn {
    vdom *d; int kind, batch, total, done, create_fail; DWORD err;
    capstats s;
    unsigned long long child_bad, wait_bad;
    double secs;
} churn;
static void churn_body(void *p) {
    churn *ch = (churn *)p;
    cap **caps = (cap **)calloc((size_t)ch->batch, sizeof(cap *));
    HANDLE *hs = (HANDLE *)calloc((size_t)ch->batch, sizeof(HANDLE));
    if (!caps || !hs) { ch->create_fail = 1; free(caps); free(hs); return; }
    for (int j = 0; j < ch->batch; j++) if (!(caps[j] = cap_new())) ch->create_fail = 1;
    uint64_t t0 = t_qpc();
    for (int base = 0; base < ch->total && !ch->create_fail; base += ch->batch) {
        int m = 0;
        for (int j = 0; j < ch->batch && base + j < ch->total; j++) {
            cap *c = caps[j];
            memset(c, 0, offsetof(cap, mode));
            c->mode = CAP_CHURN; c->kind = ch->kind; c->d = ch->d; c->ran = 0; c->bad = 0;
            c->tag = ch->d->cb_base + (uint32_t)(base + j);
            c->dirty_mx = mxcsr_variant((unsigned)(base + j)) | 0x3Fu; c->dirty_cw = cw_variant((unsigned)(base + j) + 1);
            hs[j] = mk_thread(cap_entry(ch->kind), c, 0, NULL);
            if (!hs[j]) { ch->create_fail = 1; ch->err = GetLastError(); break; }
            m++;
        }
        for (int j = 0; j < m; j++) {
            DWORD w = WaitForSingleObject(hs[j], 10000);
            DWORD ec = 0; GetExitCodeThread(hs[j], &ec);
            if (w != WAIT_OBJECT_0 || ec != 0x600Du) ch->wait_bad++;
            CloseHandle(hs[j]);
            char lab[64]; snprintf(lab, sizeof lab, "churn thread #%d", base + j);
            cap_analyse(caps[j], ch->d, ch->kind, &ch->s, lab);
            if (caps[j]->bad) ch->child_bad++;
            ch->done++;
        }
    }
    ch->secs = t_qpc_sec(t0, t_qpc());
    for (int j = 0; j < ch->batch; j++) if (caps[j]) _aligned_free(caps[j]);
    free(caps); free(hs);
}
static void churn_case(int batch) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static vdom d; memset((void *)&d, 0, sizeof d);
    int kind = best_kind();
    d.cb_base = 1; d.cb_n = 1000; d.hold_base = 1100; d.hold_n = 2; d.main_tag = 1200; d.max_tag = 1201;
    holder *hs = holders_alloc(2); REQUIRE(hs);
    static volatile LONG stop; stop = 0;
    int hstarted = holders_start(hs, 2, &d, kind, &stop, 0);
    static churn ch; memset(&ch, 0, sizeof ch);
    ch.d = &d; ch.kind = kind; ch.batch = batch; ch.total = 1000;
    static t_vregs min_, mout;
    unsigned mmx = mxcsr_variant(9); unsigned short mcw = cw_variant(4);
    t_vregs_pattern(&min_, kind, d.main_tag);
    t_set_mxcsr(mmx); t_set_x87_cw(mcw);
    if (hstarted == 2) t_vregs_call(kind, &min_, &mout, churn_body, &ch);
    unsigned mx2 = t_get_mxcsr(); unsigned short cw2 = t_get_x87_cw();
    t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
    holders_stop(hs, hstarted, &stop);
    _aligned_free(hs);
    if (hstarted < 2) SKIP("CreateThread failed for holder threads");
    if (ch.create_fail && ch.done == 0) SKIP("CreateThread failed (err %lu)", ch.err);
    if (ch.create_fail) t_note("CreateThread failed after %d threads (err %lu): checked those only", ch.done, ch.err);
    CHECKF(((mx2 ^ mmx) & MX_CTRL) == 0, "parent MXCSR 0x%04x after 1000 create/join, set 0x%04x", mx2, mmx);
    CHECKF(cw2 == mcw, "parent FCW 0x%04x after 1000 create/join, set 0x%04x", cw2, mcw);
#ifdef T_X64
    t_vregs_check(&mout, &min_, T_VK_XMM, 6, 15, "parent (nonvolatile) across 1000 CreateThread/WaitForSingleObject");
#endif
    unsigned long long ba = 0, bl = 0, vc = 0;
    verify_call(&d, "parent", kind, -1, d.main_tag, &mout, &min_, &ba, &bl, &vc);
    CHECKF(bl == 0, "parent registers hold %llu foreign pattern qwords after churn", bl);
    CHECKF(ch.child_bad == 0, "%llu churn threads saw their own registers corrupted", ch.child_bad);
    CHECKF(ch.wait_bad == 0, "%llu churn threads did not finish / wrong exit code", ch.wait_bad);
    CHECKF(ch.done == ch.total || ch.create_fail, "only %d of %d threads ran", ch.done, ch.total);
    capstats_report(&ch.s, "churn");
    if (ch.secs > 0) t_perf("threads_per_s", ch.done / ch.secs, "thread/s");
}
TEST(churn_create_join_1000_serial) { churn_case(1); }
TEST(churn_create_join_1000_batch32) { churn_case(32); }

/* ======================================================================================
 * 3. isolation matrix: N workers x blocking primitive
 * ==================================================================================== */
ISO(iso_sleep0_n2,    .op = OP_SLEEP0, .n = 2)
ISO(iso_sleep0_n16,   .op = OP_SLEEP0, .n = 16)
ISO(iso_sleep0_n64,   .op = OP_SLEEP0, .n = 64)
ISO(iso_sleep0_n256,  .op = OP_SLEEP0, .n = 256)
ISO(iso_sleep0_n1024, .op = OP_SLEEP0, .n = 1024)
ISO(iso_switch_n2,    .op = OP_SWITCH, .n = 2)
ISO(iso_switch_n16,   .op = OP_SWITCH, .n = 16)
ISO(iso_switch_n64,   .op = OP_SWITCH, .n = 64)
ISO(iso_switch_n256,  .op = OP_SWITCH, .n = 256)
ISO(iso_switch_n1024, .op = OP_SWITCH, .n = 1024)
ISO(iso_event_ring_n2,    .op = OP_EVENT_RING, .n = 2)
ISO(iso_event_ring_n16,   .op = OP_EVENT_RING, .n = 16)
ISO(iso_event_ring_n64,   .op = OP_EVENT_RING, .n = 64)
ISO(iso_event_ring_n256,  .op = OP_EVENT_RING, .n = 256)
ISO(iso_event_ring_n1024, .op = OP_EVENT_RING, .n = 1024)
ISO(iso_mutex_n2,    .op = OP_MUTEX, .n = 2)
ISO(iso_mutex_n16,   .op = OP_MUTEX, .n = 16)
ISO(iso_mutex_n64,   .op = OP_MUTEX, .n = 64)
ISO(iso_mutex_n256,  .op = OP_MUTEX, .n = 256)
ISO(iso_mutex_n1024, .op = OP_MUTEX, .n = 1024)
ISO(iso_semaphore_n2,    .op = OP_SEM, .n = 2)
ISO(iso_semaphore_n16,   .op = OP_SEM, .n = 16)
ISO(iso_semaphore_n64,   .op = OP_SEM, .n = 64)
ISO(iso_semaphore_n256,  .op = OP_SEM, .n = 256)
ISO(iso_semaphore_n1024, .op = OP_SEM, .n = 1024)
ISO(iso_apc_n2,    .op = OP_APC, .n = 2)
ISO(iso_apc_n16,   .op = OP_APC, .n = 16)
ISO(iso_apc_n64,   .op = OP_APC, .n = 64)
ISO(iso_apc_n256,  .op = OP_APC, .n = 256)
ISO(iso_apc_n1024, .op = OP_APC, .n = 1024)
ISO(iso_condvar_n2,    .op = OP_CONDVAR, .n = 2)
ISO(iso_condvar_n16,   .op = OP_CONDVAR, .n = 16)
ISO(iso_condvar_n64,   .op = OP_CONDVAR, .n = 64)
ISO(iso_condvar_n256,  .op = OP_CONDVAR, .n = 256)
ISO(iso_condvar_n1024, .op = OP_CONDVAR, .n = 1024)
ISO(iso_srw_n2,    .op = OP_SRW, .n = 2)
ISO(iso_srw_n16,   .op = OP_SRW, .n = 16)
ISO(iso_srw_n64,   .op = OP_SRW, .n = 64)
ISO(iso_srw_n256,  .op = OP_SRW, .n = 256)
ISO(iso_srw_n1024, .op = OP_SRW, .n = 1024)
/* further primitives */
ISO(iso_critsec_n16,  .op = OP_CRITSEC, .n = 16)
ISO(iso_critsec_n256, .op = OP_CRITSEC, .n = 256)
ISO(iso_waitaddr_n16,  .op = OP_WAITADDR, .n = 16)
ISO(iso_waitaddr_n256, .op = OP_WAITADDR, .n = 256)
ISO(iso_pipe_ring_n2,  .op = OP_PIPE, .n = 2)
ISO(iso_pipe_ring_n64, .op = OP_PIPE, .n = 64)
ISO(iso_waitable_timer_n16, .op = OP_TIMER, .n = 16)
ISO(iso_wfmo_n16,  .op = OP_WFMO, .n = 16)
ISO(iso_wfmo_n256, .op = OP_WFMO, .n = 256)
ISO(iso_signal_and_wait_n2,  .op = OP_SOAW, .n = 2)
ISO(iso_signal_and_wait_n64, .op = OP_SOAW, .n = 64)
/* scheduling variants: pure preemption, pinning, migration, suspension, priorities */
ISO(iso_preempt_pin_one_n16,   .op = OP_NONE, .n = 16,  .pin = PIN_ONE)
ISO(iso_preempt_pin_one_n256,  .op = OP_NONE, .n = 256, .pin = PIN_ONE)
ISO(iso_preempt_unpinned_n64,  .op = OP_NONE, .n = 64)
ISO(iso_preempt_unpinned_n1024, .op = OP_NONE, .n = 1024)
ISO(iso_sleep0_pin_one_n16,    .op = OP_SLEEP0, .n = 16, .pin = PIN_ONE)
ISO(iso_sleep0_pin_spread_n64, .op = OP_SLEEP0, .n = 64, .pin = PIN_SPREAD)
ISO(iso_switch_pin_one_n64,    .op = OP_SWITCH, .n = 64, .pin = PIN_ONE)
ISO(iso_event_ring_pin_one_n16, .op = OP_EVENT_RING, .n = 16, .pin = PIN_ONE)
ISO(iso_migrate_self_n16, .op = OP_MIGRATE, .n = 16)
ISO(iso_migrate_self_n64, .op = OP_MIGRATE, .n = 64)
ISO(iso_migrate_external_n16, .op = OP_NONE, .n = 16, .mig_ext = 1, .spin = 5000)
ISO(iso_migrate_external_n64, .op = OP_NONE, .n = 64, .mig_ext = 1, .spin = 5000)
ISO(iso_suspend_resume_n16,  .op = OP_NONE, .n = 16, .susp = 1, .spin = 5000)
ISO(iso_suspend_resume_n64,  .op = OP_NONE, .n = 64, .susp = 1, .spin = 5000)
ISO(iso_suspend_getcontext_n16,  .op = OP_NONE, .n = 16, .susp = 1, .getctx = 1, .spin = 5000)
ISO(iso_suspend_getcontext_pin_one_n64, .op = OP_NONE, .n = 64, .pin = PIN_ONE, .susp = 1, .getctx = 1, .spin = 5000)
ISO(iso_suspend_getcontext_mixed_n64, .op = OP_NONE, .n = 64, .susp = 1, .getctx = 1, .rf = RF_MIX, .spin = 5000)
ISO(iso_preempt_mxcsr_flags_pin_one_n16, .op = OP_NONE, .n = 16, .pin = PIN_ONE, .mxflags = 1, .spin = 5000)
ISO(iso_preempt_mxcsr_flags_n256,  .op = OP_NONE, .n = 256, .mxflags = 1, .spin = 5000)
ISO(iso_apc_waitex_n2,    .op = OP_APC_WAITEX, .n = 2)
ISO(iso_apc_waitex_n64,   .op = OP_APC_WAITEX, .n = 64)
ISO(iso_apc_waitex_n1024, .op = OP_APC_WAITEX, .n = 1024)
ISO(iso_priority_mix_n64,    .op = OP_SLEEP0, .n = 64, .prio_mix = 1)
/* other register files and mixtures */
ISO(iso_x87_stack_pin_one_n16,  .op = OP_NONE, .n = 16, .pin = PIN_ONE, .rf = RF_X87, .spin = 5000)
ISO(iso_x87_stack_sleep0_n256,  .op = OP_SLEEP0, .n = 256, .rf = RF_X87)
ISO(iso_mmx_pin_one_n16,        .op = OP_NONE, .n = 16, .pin = PIN_ONE, .rf = RF_MMX, .spin = 5000)
ISO(iso_mmx_switch_n256,        .op = OP_SWITCH, .n = 256, .rf = RF_MMX)
ISO(iso_sse_only_upper_lanes_pin_one_n16, .op = OP_SLEEP0, .n = 16, .pin = PIN_ONE, .rf = RF_MIX)
ISO(iso_mixed_regfiles_n256,    .op = OP_SWITCH, .n = 256, .rf = RF_MIX)
ISO(iso_mixed_regfiles_event_ring_n64, .op = OP_EVENT_RING, .n = 64, .rf = RF_MIX)

/* ======================================================================================
 * 4. barrier-synchronised phases (deterministic phase count; every thread changes tag per phase)
 * ==================================================================================== */
ISO(barrier_syncbarrier_n16,  .op = OP_SYNCBAR, .n = 16,  .fixed_iters = 300, .spin = 500)
ISO(barrier_syncbarrier_n64,  .op = OP_SYNCBAR, .n = 64,  .fixed_iters = 120, .spin = 500)
ISO(barrier_custom_n256,      .op = OP_CBAR,    .n = 256, .fixed_iters = 40,  .spin = 500)
ISO(barrier_custom_n1024,     .op = OP_CBAR,    .n = 1024, .fixed_iters = 12, .spin = 500)

/* ======================================================================================
 * 5. thread-pool callbacks
 * ==================================================================================== */
typedef struct tpctx {
    vdom d; int kind;
    volatile LONG started, done, target, stop_rearm;
    HANDLE all_done, ev;
    PTP_WAIT tpw;
    volatile LONG bad_entry_mx, bad_entry_cw, bad_spin, bad_abi, bad_leak, bad_env, wrong_ctx;
    volatile LONG first_entry_mx, first_entry_cw;
    volatile LONG tids_seen;
} tpctx;
#define TP_CB_N 4096u
static void tp_body(tpctx *c) {
    LONG k = InterlockedIncrement(&c->started);
    unsigned emx = t_get_mxcsr(); unsigned short ecw = t_get_x87_cw();
    if ((emx & MX_CTRL) != MX_DEFAULT) { InterlockedIncrement(&c->bad_entry_mx); InterlockedCompareExchange(&c->first_entry_mx, (LONG)emx, 0); }
    if (ecw != CW_DEFAULT) { InterlockedIncrement(&c->bad_entry_cw); InterlockedCompareExchange(&c->first_entry_cw, (LONG)ecw, 0); }
    unsigned mx = mxcsr_variant((unsigned)k); unsigned short cw = cw_variant((unsigned)k);
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    uint32_t tag = c->d.cb_base + (uint32_t)(k - 1) % TP_CB_N;
    __attribute__((aligned(64))) t_vregs in, out;
    t_vregs_pattern(&in, c->kind, tag);
    t_vregs_spin(c->kind, &in, &out, NULL, 1500);
    if (verify_full(&c->d, "pool callback after preemptible spin", c->kind, &out, &in)) InterlockedIncrement(&c->bad_spin);
    t_vregs_call(c->kind, &in, &out, op_sleep0, NULL);
    unsigned long long ba = 0, bl = 0, vc = 0;
    verify_call(&c->d, "pool callback after Sleep(0)", c->kind, -1, tag, &out, &in, &ba, &bl, &vc);
    if (ba) InterlockedIncrement(&c->bad_abi);
    if (bl) InterlockedIncrement(&c->bad_leak);
    if (((t_get_mxcsr() ^ mx) & MX_CTRL) || t_get_x87_cw() != cw) InterlockedIncrement(&c->bad_env);
    t_set_mxcsr(emx); t_set_x87_cw(ecw);          /* ABI: leave the pool thread as we found it */
    if (InterlockedIncrement(&c->done) == c->target && c->all_done) SetEvent(c->all_done);
}
static void CALLBACK tp_work_cb(PTP_CALLBACK_INSTANCE inst, PVOID ctx, PTP_WORK work) { (void)inst; (void)work; tp_body((tpctx *)ctx); }
static void CALLBACK tp_simple_cb(PTP_CALLBACK_INSTANCE inst, PVOID ctx) { (void)inst; tp_body((tpctx *)ctx); }
static DWORD WINAPI qwi_cb(LPVOID ctx) { tp_body((tpctx *)ctx); return 0; }
static void CALLBACK tp_timer_cb(PTP_CALLBACK_INSTANCE inst, PVOID ctx, PTP_TIMER t) { (void)inst; (void)t; tp_body((tpctx *)ctx); }
static void CALLBACK tp_wait_cb(PTP_CALLBACK_INSTANCE inst, PVOID ctx, PTP_WAIT w, TP_WAIT_RESULT r) {
    (void)inst; tpctx *c = (tpctx *)ctx;
    if (r != WAIT_OBJECT_0) InterlockedIncrement(&c->wrong_ctx);
    tp_body(c);
    if (!c->stop_rearm) SetThreadpoolWait(w, c->ev, NULL);
}
static void CALLBACK rw_cb(PVOID ctx, BOOLEAN timed_out) {
    tpctx *c = (tpctx *)ctx;
    if (timed_out) InterlockedIncrement(&c->wrong_ctx);
    tp_body(c);
}

enum { TPM_WORK, TPM_WORK_PRIVATE, TPM_SIMPLE, TPM_QWI, TPM_QWI_LONG, TPM_TIMER, TPM_WAIT, TPM_REGWAIT };
typedef struct tpdrive { tpctx *c; int mode, m, api_fail, timed_out; } tpdrive;
static int wait_done(tpctx *c, LONG want, DWORD ms) {
    DWORD t0 = GetTickCount();
    while (c->done < want) { if (GetTickCount() - t0 > ms) return 0; Sleep(1); }
    return 1;
}
/* inside t_vregs_call on the case thread (main pattern live) */
/* blocking thread-pool waits run through t_run_worker: a hang retires the process instead of wedging the case */
static void tpw_wait_work(void *p)  { WaitForThreadpoolWorkCallbacks((PTP_WORK)p, FALSE); }
static void tpw_wait_timer(void *p) { WaitForThreadpoolTimerCallbacks((PTP_TIMER)p, TRUE); }
static void tpw_wait_wait(void *p)  { WaitForThreadpoolWaitCallbacks((PTP_WAIT)p, TRUE); }
static void tpw_close_pool(void *p) { CloseThreadpool((PTP_POOL)p); }
typedef struct tpw_unreg_arg { HANDLE hw; int ok; } tpw_unreg_arg;
static void tpw_unreg(void *p) { tpw_unreg_arg *a = (tpw_unreg_arg *)p; a->ok = UnregisterWaitEx(a->hw, INVALID_HANDLE_VALUE) ? 1 : 0; }
static void tpw_run(t_worker_fn fn, void *arg) { t_run_worker(fn, arg, 0, 30000, NULL); }
static void tp_drive(void *p) {
    tpdrive *dr = (tpdrive *)p; tpctx *c = dr->c;
    switch (dr->mode) {
    case TPM_WORK:
    case TPM_WORK_PRIVATE: {
        PTP_POOL pool = NULL; TP_CALLBACK_ENVIRON env; PTP_CALLBACK_ENVIRON pe = NULL;
        if (dr->mode == TPM_WORK_PRIVATE) {
            pool = CreateThreadpool(NULL);
            if (!pool) { dr->api_fail++; return; }
            SetThreadpoolThreadMaximum(pool, 4);
            if (!SetThreadpoolThreadMinimum(pool, 1)) dr->api_fail++;
            InitializeThreadpoolEnvironment(&env);
            SetThreadpoolCallbackPool(&env, pool);
            pe = &env;
        }
        PTP_WORK w = CreateThreadpoolWork(tp_work_cb, c, pe);
        if (!w) { dr->api_fail++; } else {
            for (int i = 0; i < dr->m; i++) SubmitThreadpoolWork(w);
            tpw_run(tpw_wait_work, w);
            CloseThreadpoolWork(w);
        }
        if (pool) { DestroyThreadpoolEnvironment(&env); tpw_run(tpw_close_pool, pool); }
        break;
    }
    case TPM_SIMPLE:
        for (int i = 0; i < dr->m; i++) if (!TrySubmitThreadpoolCallback(tp_simple_cb, c, NULL)) { dr->api_fail++; InterlockedDecrement(&c->target); }
        if (WaitForSingleObject(c->all_done, 20000) != WAIT_OBJECT_0) dr->timed_out++;
        break;
    case TPM_QWI:
    case TPM_QWI_LONG:
        for (int i = 0; i < dr->m; i++)
            if (!QueueUserWorkItem(qwi_cb, c, dr->mode == TPM_QWI_LONG ? WT_EXECUTELONGFUNCTION : WT_EXECUTEDEFAULT)) { dr->api_fail++; InterlockedDecrement(&c->target); }
        if (WaitForSingleObject(c->all_done, 20000) != WAIT_OBJECT_0) dr->timed_out++;
        break;
    case TPM_TIMER: {
        PTP_TIMER t = CreateThreadpoolTimer(tp_timer_cb, c, NULL);
        if (!t) { dr->api_fail++; break; }
        ULARGE_INTEGER due; due.QuadPart = (ULONGLONG)-10000LL;     /* 1 ms relative */
        FILETIME ft; ft.dwLowDateTime = due.LowPart; ft.dwHighDateTime = due.HighPart;
        SetThreadpoolTimer(t, &ft, 10, 0);
        if (WaitForSingleObject(c->all_done, 20000) != WAIT_OBJECT_0) dr->timed_out++;
        SetThreadpoolTimer(t, NULL, 0, 0);
        tpw_run(tpw_wait_timer, t);
        CloseThreadpoolTimer(t);
        break;
    }
    case TPM_WAIT: {
        PTP_WAIT w = CreateThreadpoolWait(tp_wait_cb, c, NULL);
        if (!w) { dr->api_fail++; break; }
        SetThreadpoolWait(w, c->ev, NULL);
        for (int i = 0; i < dr->m; i++) { SetEvent(c->ev); if (!wait_done(c, i + 1, 5000)) { dr->timed_out++; break; } }
        c->stop_rearm = 1;
        SetThreadpoolWait(w, NULL, NULL);
        tpw_run(tpw_wait_wait, w);
        CloseThreadpoolWait(w);
        break;
    }
    case TPM_REGWAIT: {
        HANDLE hw = NULL;
        if (!RegisterWaitForSingleObject(&hw, c->ev, rw_cb, c, INFINITE, WT_EXECUTEDEFAULT)) { dr->api_fail++; break; }
        for (int i = 0; i < dr->m; i++) { SetEvent(c->ev); if (!wait_done(c, i + 1, 5000)) { dr->timed_out++; break; } }
        { tpw_unreg_arg ua = { hw, 0 }; tpw_run(tpw_unreg, &ua); if (!ua.ok) dr->api_fail++; }
        break;
    }
    }
}
static void tp_case(int mode, int m) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    static tpctx c; memset((void *)&c, 0, sizeof c);
    c.kind = best_kind();
    c.d.cb_base = 1; c.d.cb_n = TP_CB_N; c.d.hold_base = 5000; c.d.hold_n = 4; c.d.main_tag = 6000; c.d.max_tag = 6001;
    c.target = m;
    c.all_done = CreateEventW(NULL, TRUE, FALSE, NULL);
    c.ev = CreateEventW(NULL, FALSE, FALSE, NULL);
    REQUIRE(c.all_done && c.ev);
    holder *hs = holders_alloc(2); REQUIRE(hs);
    static volatile LONG stop; stop = 0;
    int hstarted = holders_start(hs, 2, &c.d, c.kind, &stop, 0);
    static t_vregs min_, mout;
    unsigned mmx = mxcsr_variant(11); unsigned short mcw = cw_variant(5);
    t_vregs_pattern(&min_, c.kind, c.d.main_tag);
    tpdrive dr = { &c, mode, m, 0, 0 };
    uint64_t t0 = t_qpc();
    t_set_mxcsr(mmx); t_set_x87_cw(mcw);
    t_vregs_call(c.kind, &min_, &mout, tp_drive, &dr);
    unsigned mx2 = t_get_mxcsr(); unsigned short cw2 = t_get_x87_cw();
    t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
    double secs = t_qpc_sec(t0, t_qpc());
    holders_stop(hs, hstarted, &stop);
    _aligned_free(hs);
    CloseHandle(c.all_done); CloseHandle(c.ev);
    if (hstarted < 2) SKIP("CreateThread failed for holder threads");
    CHECKF(dr.api_fail == 0, "%d thread-pool API failures", dr.api_fail);
    CHECKF(dr.timed_out == 0, "callbacks did not complete in time (%ld of %ld done)", (long)c.done, (long)c.target);
    if (mode == TPM_TIMER) CHECKF(c.done >= c.target, "timer callbacks %ld < %ld", (long)c.done, (long)c.target);
    else CHECKF(c.done == m, "callbacks completed %ld, submitted %d", (long)c.done, m);
    CHECKF(((mx2 ^ mmx) & MX_CTRL) == 0, "submitting thread MXCSR 0x%04x, set 0x%04x", mx2, mmx);
    CHECKF(cw2 == mcw, "submitting thread FCW 0x%04x, set 0x%04x", cw2, mcw);
#ifdef T_X64
    t_vregs_check(&mout, &min_, T_VK_XMM, 6, 15, "submitting thread (nonvolatile) across submit + wait");
#endif
    unsigned long long ba = 0, bl = 0, vc = 0;
    verify_call(&c.d, "submitting thread", c.kind, -1, c.d.main_tag, &mout, &min_, &ba, &bl, &vc);
    CHECKF(bl == 0, "submitting thread registers hold %llu foreign pattern qwords", bl);
    CHECKF(c.bad_spin == 0, "%ld callbacks saw their registers corrupted while preempted", (long)c.bad_spin);
    CHECKF(c.bad_abi == 0, "%ld callbacks lost nonvolatile XMM6-15 across Sleep(0)", (long)c.bad_abi);
    CHECKF(c.bad_leak == 0, "%ld callbacks saw another thread's register state", (long)c.bad_leak);
    CHECKF(c.bad_env == 0, "%ld callbacks lost their MXCSR/FCW", (long)c.bad_env);
    CHECKF(c.wrong_ctx == 0, "%ld callbacks reported an unexpected wait result / timeout", (long)c.wrong_ctx);
    CHECKF(c.bad_entry_mx == 0, "%ld callbacks entered with MXCSR control != 0x1F80 (first 0x%04lx)", (long)c.bad_entry_mx, (long)c.first_entry_mx);
    CHECKF(c.bad_entry_cw == 0, "%ld callbacks entered with FCW != 0x027F (first 0x%04lx)", (long)c.bad_entry_cw, (long)c.first_entry_cw);
    t_info("kind", "%s", kind_name(c.kind));
    t_info("callbacks", "%ld", (long)c.done);
    if (secs > 0) t_perf("callbacks_per_s", c.done / secs, "cb/s");
}
TEST(tp_work_default_pool)        { tp_case(TPM_WORK, 2000); }
TEST(tp_work_private_pool_max4)   { tp_case(TPM_WORK_PRIVATE, 1000); }
TEST(tp_try_submit_callback)      { tp_case(TPM_SIMPLE, 1000); }
TEST(tp_queue_user_work_item)     { tp_case(TPM_QWI, 1000); }
TEST(tp_queue_user_work_item_long){ tp_case(TPM_QWI_LONG, 256); }
TEST(tp_timer_periodic)           { tp_case(TPM_TIMER, 100); }
TEST(tp_wait_rearm)               { tp_case(TPM_WAIT, 200); }
TEST(tp_register_wait)            { tp_case(TPM_REGWAIT, 200); }

/* ======================================================================================
 * 6. observations (INFO only: not guaranteed by any contract)
 * ==================================================================================== */
TEST(info_upper_lanes_across_sleep0) {
    /* The ABI makes upper YMM/ZMM bits volatile across calls; record what this OS actually does. Only a
     * foreign value would be a failure. */
    int kind = best_kind();
    if (kind == T_VK_XMM) SKIP("AVX not usable: no upper lanes");
    static vdom d; memset((void *)&d, 0, sizeof d); d.main_tag = 1; d.max_tag = 2;
    static t_vregs in, out;
    t_vregs_pattern(&in, kind, 1);
    int kept = 0, zeroed = 0, other = 0;
    unsigned long long ba = 0, bl = 0, vc = 0;
    for (int i = 0; i < 200; i++) {
        t_vregs_call(kind, &in, &out, (i & 1) ? op_switch : op_sleep0, NULL);
        verify_call(&d, "case thread", kind, -1, 1, &out, &in, &ba, &bl, &vc);
        for (int r = 0; r < t_vregs_count(kind); r++) {
            static const uint8_t z[64];
            if (!memcmp(out.v[r] + 16, in.v[r] + 16, (size_t)kind - 16)) kept++;
            else if (!memcmp(out.v[r] + 16, z, (size_t)kind - 16)) zeroed++;
            else other++;
        }
    }
    CHECKF(bl == 0, "%llu foreign pattern qwords", bl);
#ifdef T_X64
    CHECKF(ba == 0, "%llu nonvolatile XMM6-15 losses", ba);
#endif
    t_info("kind", "%s", kind_name(kind));
    t_info("upper_kept", "%d", kept);
    t_info("upper_zeroed", "%d", zeroed);
    t_info("upper_other", "%d", other);
}

static volatile LONG g_apc_ran;
static void CALLBACK apc_dirty_fn(ULONG_PTR p) {
    (void)p;
    t_set_mxcsr(0x7F80u);           /* RC = toward zero, not restored (deliberate ABI violation) */
    t_set_x87_cw(0x0C7Fu);
    g_apc_ran = 1;
}
TEST(info_apc_dirty_fp_env) {
    /* An APC routine that leaves MXCSR/FCW changed: whether the APC dispatcher restores the interrupted
     * context's FP environment is an implementation detail (x64 Windows resumes via NtContinue). */
    g_apc_ran = 0;
    t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
    CHECK(QueueUserAPC(apc_dirty_fn, GetCurrentThread(), 0));
    DWORD r = SleepEx(1000, TRUE);
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    t_set_mxcsr(MX_DEFAULT); t_set_x87_cw(CW_DEFAULT);
    CHECK_EQ(r, WAIT_IO_COMPLETION);
    CHECK(g_apc_ran);
    t_info("mxcsr_after_dirty_apc", "0x%04x (%s)", mx, (mx & MX_CTRL) == MX_DEFAULT ? "restored" : "APC value kept");
    t_info("fcw_after_dirty_apc", "0x%04x (%s)", cw, cw == CW_DEFAULT ? "restored" : "APC value kept");
}
