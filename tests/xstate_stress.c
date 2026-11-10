/* xstate_stress - many-thread extended-state contamination and context-switch stress.
 *
 * Contract under test (the OS side of per-thread processor state):
 *   - Every thread owns its x87 stack / MMX registers, XMM0-15 (x86: XMM0-7), the YMM upper halves, MXCSR and
 *     the x87 control word. Whatever the OS does to a thread - preemption, voluntary yields (SwitchToThread,
 *     Sleep(0), Sleep(1), event ping-pong, waitable timers, user APCs, lock contention, blocking system calls),
 *     CPU migration (SetThreadAffinityMask), priority changes, SuspendThread / GetThreadContext /
 *     SetThreadContext from another thread, timer callbacks (CreateTimerQueueTimer, timeSetEvent) - the thread
 *     finds exactly the state it left, and never another thread's.
 *   - Each worker has a unique 64-bit seed. Every iteration it derives a new "generation" of register
 *     patterns (vector words, x87 values, MMX words, MXCSR with rounding mode / FTZ / DAZ / sticky flags,
 *     x87 CW with precision + rounding control) from a PRNG, publishes it under a seqlock, and runs one asm
 *     routine that loads the state, holds it in a GPR-only PAUSE loop (or runs an integer SIMD chain, or calls
 *     a yielding C function), stores it, and restores the thread's own MXCSR / CW "mode" (which also differs
 *     per thread). The worker checks the stored state bit-exactly against the generation, plus two rounding
 *     probes (1/10 and -1/10 by DIVSD and FIDIV) whose results depend only on the published RC / PC.
 *   - Context-capture oracle: a controller thread suspends random workers and captures their context
 *     (CONTEXT_FULL + FP, optionally CONTEXT_XSTATE). When the captured IP lies inside a hold loop, the captured
 *     RCX/ECX must point at one of the thread's published blocks and XMM/YMM/MXCSR/FCW/x87/MMX must equal that
 *     block's generation (re-derived from the seed, not read back). Outside the asm, the captured MXCSR control
 *     bits and FCW must be the thread's mode (or, inside a call made with a generation loaded, that
 *     generation's). Every mismatch is FAIL with role, index, thread id, generation, expected and observed bits,
 *     and a decode of the observed word (patterns embed marker / thread index / register / word).
 *
 *   - Forced time slicing with live state: preempt_* cases pin several workers to one CPU (or two per CPU) and
 *     size the hold loop (calibrated PAUSE count) to tens of ms, so the scheduler must switch between threads
 *     whose XMM/YMM/x87/MMX/MXCSR/FCW hold distinct patterns mid-hold; ctl_migrate_* moves victims between CPUs
 *     during multi-ms holds. How often a hold was descheduled is INFO only (quantum length is policy).
 *   - Thread-pool work items (QueueUserWorkItem, queued from timer ticks) run the same register hold.
 *   - A Wine-gap XFAIL is raised only when the case recorded no failure of its own (g_st_fails), so it never
 *     turns a genuine FAIL into XFAIL.
 * Windows behaviour checked in the Windows 10 22H2 (19041) reference decompilation (docker1 corpus explorer):
 *   kernelbase!SuspendThread = NtSuspendThread, returns the previous count or 0xFFFFFFFF + LastError.
 *   kernelbase!GetThreadContext / SetThreadContext = Nt{Get,Set}ContextThread, BOOL from NTSTATUS.
 *   ntoskrnl!PspGetContextThreadInternal: for another thread it queues PspGetSetContextSpecialApc
 *     (KeInitializeApc + KeInsertQueueApc) and waits (KeWaitForGate) until the target captured its own
 *     state, so after GetThreadContext returns the target executes no further user code while it stays
 *     suspended: reading the worker's published seqlock fields after the capture is consistent with it.
 *   x64 ABI (documented): MXCSR bits 6-15 and the x87 CW are nonvolatile, XMM6-15 low 128 bits are
 *     nonvolatile, YMM upper halves / XMM0-5 volatile. x86: every XMM register volatile.
 *   SDM: MMX instructions set TOP=0 and all tags valid, bits 79:64 of the aliased register to 1s; FXSAVE
 *     stores ST(i) in 16-byte slots and the abridged tag word; FNINIT empties the stack.
 * Result policy: feature absent -> SKIP; advertised but broken -> FAIL; variable facts -> t_info / t_note.
 * Wine: t_is_wine() only selects an XFAIL reason for documented Wine gaps; checks never change for Windows.
 * Helpers this file adds locally (not in common/): seqlock pattern slots, st_* asm routines, XSTATE context
 * allocation (same two-phase InitializeContext negotiation as context_xstate.c). */
#include "t.h"

/* Failure counter local to this file: every CHECK* of this file passes through these wrappers (single
 * evaluation of the condition), so a case can tell whether it already failed. A Wine-gap XFAIL is raised
 * only when nothing else failed: t_xfail() would otherwise reclassify a genuine FAIL as XFAIL. */
static volatile LONG g_st_fails;
static int st_cnt(int ok) { if (!ok) InterlockedIncrement(&g_st_fails); return ok; }
#define t_check(ok, ...)        t_check(st_cnt(ok), __VA_ARGS__)
#define t_check_reg_at(...)     st_cnt(t_check_reg_at(__VA_ARGS__))
#define t_check_mem(...)        st_cnt(t_check_mem(__VA_ARGS__))

/* ======================================================================================================
 * Architecture glue
 * ====================================================================================================== */
#ifndef CONTEXT_XSTATE
#ifdef T_X64
#define CONTEXT_XSTATE (CONTEXT_AMD64 | 0x40L)
#else
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)
#endif
#endif
#ifdef T_X64
#define ST_CF_ALL (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT)
#define ST_CF_FP  CONTEXT_FLOATING_POINT
#else
#define ST_CF_ALL (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS)
#define ST_CF_FP  (CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS)
#endif
#define ST_NV       T_NXMM          /* XMM/YMM registers in this mode: 16 (x64) / 8 (x86) */
#define ST_XS_AVX   2               /* XSTATE_AVX feature id */
#define ST_MAXW     256             /* worker cap */
#define ST_NCB      4               /* timer-callback register slots */
#define MX_CTRL     0xFFC0u         /* MXCSR control bits (DAZ, masks, RC, FTZ) */

/* ======================================================================================================
 * Block shared with the asm routines (fixed layout)
 * ====================================================================================================== */
#define SO_VIN    0
#define SO_VOUT   512
#define SO_FIN    1024
#define SO_FOUT   1152
#define SO_MMIN   1280
#define SO_MMOUT  1344
#define SO_MXGEN  1408
#define SO_MXMODE 1412
#define SO_FCWGEN 1416
#define SO_FCWMODE 1418
#define SO_FLAGS  1420
#define SO_SPIN   1424
#define SO_ROUNDS 1428
#define SO_MXOUT  1432
#define SO_FCWOUT 1436
#define SO_FSWOUT 1438
#define SO_PSSE   1440
#define SO_PX87   1456
#define SO_ONE    1488
#define SO_MONE   1496
#define SO_TEN    1504
#define SO_ITEN   1512
#define SO_FN     1520
#define SO_ARG    1528
#define SO_COUT   1536

typedef struct st_blk {
    uint8_t  vin[16][32];        /* register i: low 16 bytes XMM, bytes 16..31 YMM upper */
    uint8_t  vout[16][32];
    uint8_t  fin[8][16];         /* 80-bit values, fin[0] -> ST(0) */
    uint8_t  fout[8][16];
    uint64_t mmin[8];
    uint64_t mmout[8];
    uint32_t mx_gen, mx_mode;
    uint16_t fcw_gen, fcw_mode;
    uint32_t flags;              /* 1 = x87 stack, 2 = MMX */
    uint32_t spin, rounds, mx_out;
    uint16_t fcw_out, fsw_out;
    uint64_t psse[2];            /* DIVSD 1/10, -1/10 */
    uint8_t  px87[2][16];        /* FIDIV 1/10, -1/10 (80-bit) */
    double   one, mone, ten;
    int32_t  iten, pad0;
    uint64_t fn, arg;            /* st_call: fn(arg) */
    uint8_t  cout[16][32];       /* st_compute output */
} __attribute__((aligned(64))) st_blk;
_Static_assert(offsetof(st_blk, vout) == SO_VOUT, "vout");
_Static_assert(offsetof(st_blk, fin) == SO_FIN, "fin");
_Static_assert(offsetof(st_blk, fout) == SO_FOUT, "fout");
_Static_assert(offsetof(st_blk, mmin) == SO_MMIN, "mmin");
_Static_assert(offsetof(st_blk, mmout) == SO_MMOUT, "mmout");
_Static_assert(offsetof(st_blk, mx_gen) == SO_MXGEN, "mxgen");
_Static_assert(offsetof(st_blk, mx_mode) == SO_MXMODE, "mxmode");
_Static_assert(offsetof(st_blk, fcw_gen) == SO_FCWGEN, "fcwgen");
_Static_assert(offsetof(st_blk, fcw_mode) == SO_FCWMODE, "fcwmode");
_Static_assert(offsetof(st_blk, flags) == SO_FLAGS, "flags");
_Static_assert(offsetof(st_blk, spin) == SO_SPIN, "spin");
_Static_assert(offsetof(st_blk, rounds) == SO_ROUNDS, "rounds");
_Static_assert(offsetof(st_blk, mx_out) == SO_MXOUT, "mxout");
_Static_assert(offsetof(st_blk, fcw_out) == SO_FCWOUT, "fcwout");
_Static_assert(offsetof(st_blk, fsw_out) == SO_FSWOUT, "fswout");
_Static_assert(offsetof(st_blk, psse) == SO_PSSE, "psse");
_Static_assert(offsetof(st_blk, px87) == SO_PX87, "px87");
_Static_assert(offsetof(st_blk, one) == SO_ONE, "one");
_Static_assert(offsetof(st_blk, mone) == SO_MONE, "mone");
_Static_assert(offsetof(st_blk, ten) == SO_TEN, "ten");
_Static_assert(offsetof(st_blk, iten) == SO_ITEN, "iten");
_Static_assert(offsetof(st_blk, fn) == SO_FN, "fn");
_Static_assert(offsetof(st_blk, arg) == SO_ARG, "arg");
_Static_assert(offsetof(st_blk, cout) == SO_COUT, "cout");
_Static_assert(sizeof(st_blk) == 2048, "st_blk size");

/* asm entry points (exact assembler names on both architectures) */
void st_hold_xmm(st_blk *b) __asm__("st_hold_xmm");
void st_hold_ymm(st_blk *b) __asm__("st_hold_ymm");
void st_compute_xmm(st_blk *b) __asm__("st_compute_xmm");
void st_compute_ymm(st_blk *b) __asm__("st_compute_ymm");
void st_call_xmm(st_blk *b) __asm__("st_call_xmm");
void st_dump_xmm(void *out) __asm__("st_dump_xmm");
void st_dump_ymm(void *out) __asm__("st_dump_ymm");
extern const char st_text_begin[] __asm__("st_text_begin");
extern const char st_text_end[] __asm__("st_text_end");
extern const char st_hold_xmm_begin[] __asm__("st_hold_xmm_begin");
extern const char st_hold_xmm_end[] __asm__("st_hold_xmm_end");
extern const char st_hold_ymm_begin[] __asm__("st_hold_ymm_begin");
extern const char st_hold_ymm_end[] __asm__("st_hold_ymm_end");

#define ST_S_(x) #x
#define ST_S(x) ST_S_(x)
#define A(o) ST_S(o)
#define ST_G(n) ".globl " n "\n" n ":\n"

/* Rounding probes: DIVSD 1/10, -1/10 under MXCSR = gen, FIDIV 1/10, -1/10 under FCW = gen. Afterwards
 * MXCSR is reloaded (exact generation value, sticky flags included) and x87 exceptions cleared. */
#define ST_PROBES(b) \
    "  ldmxcsr " A(SO_MXGEN) "(" b ")\n  fninit\n  fldcw " A(SO_FCWGEN) "(" b ")\n" \
    "  movsd " A(SO_ONE) "(" b "), %xmm0\n  divsd " A(SO_TEN) "(" b "), %xmm0\n  movsd %xmm0, " A(SO_PSSE) "(" b ")\n" \
    "  movsd " A(SO_MONE) "(" b "), %xmm0\n  divsd " A(SO_TEN) "(" b "), %xmm0\n  movsd %xmm0, " A(SO_PSSE) "+8(" b ")\n" \
    "  fld1\n  fidivl " A(SO_ITEN) "(" b ")\n  fstpt " A(SO_PX87) "(" b ")\n" \
    "  fld1\n  fchs\n  fidivl " A(SO_ITEN) "(" b ")\n  fstpt " A(SO_PX87) "+16(" b ")\n" \
    "  ldmxcsr " A(SO_MXGEN) "(" b ")\n  fnclex\n"
#define ST_FPLOAD(b) \
    "  testl $1, " A(SO_FLAGS) "(" b ")\n  jz 71f\n" \
    "  fldt " A(SO_FIN) "+112(" b ")\n  fldt " A(SO_FIN) "+96(" b ")\n  fldt " A(SO_FIN) "+80(" b ")\n" \
    "  fldt " A(SO_FIN) "+64(" b ")\n  fldt " A(SO_FIN) "+48(" b ")\n  fldt " A(SO_FIN) "+32(" b ")\n" \
    "  fldt " A(SO_FIN) "+16(" b ")\n  fldt " A(SO_FIN) "+0(" b ")\n71:\n" \
    "  testl $2, " A(SO_FLAGS) "(" b ")\n  jz 72f\n" \
    "  .irp i,0,1,2,3,4,5,6,7\n  movq " A(SO_MMIN) "+\\i*8(" b "), %mm\\i\n  .endr\n72:\n"
#define ST_FPSTORE(b) \
    "  stmxcsr " A(SO_MXOUT) "(" b ")\n  fnstcw " A(SO_FCWOUT) "(" b ")\n  fnstsw " A(SO_FSWOUT) "(" b ")\n" \
    "  testl $1, " A(SO_FLAGS) "(" b ")\n  jz 73f\n" \
    "  fstpt " A(SO_FOUT) "+0(" b ")\n  fstpt " A(SO_FOUT) "+16(" b ")\n  fstpt " A(SO_FOUT) "+32(" b ")\n" \
    "  fstpt " A(SO_FOUT) "+48(" b ")\n  fstpt " A(SO_FOUT) "+64(" b ")\n  fstpt " A(SO_FOUT) "+80(" b ")\n" \
    "  fstpt " A(SO_FOUT) "+96(" b ")\n  fstpt " A(SO_FOUT) "+112(" b ")\n73:\n" \
    "  testl $2, " A(SO_FLAGS) "(" b ")\n  jz 74f\n" \
    "  .irp i,0,1,2,3,4,5,6,7\n  movq %mm\\i, " A(SO_MMOUT) "+\\i*8(" b ")\n  .endr\n  emms\n74:\n"
#define ST_RESTORE(b) \
    "  ldmxcsr " A(SO_MXMODE) "(" b ")\n  fnclex\n  fldcw " A(SO_FCWMODE) "(" b ")\n"

/* integer chain x[i] += x[i+1] (sequential, x[n-1] uses the updated x[0]) and XOR chain for YMM */
#define CHX(a, b) "  paddd %xmm" #b ", %xmm" #a "\n"
#define CHY(a, b) "  vxorps %ymm" #b ", %ymm" #a ", %ymm" #a "\n"

#ifdef T_X64
#define ST_R  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#define ST_CHAIN_X CHX(0,1) CHX(1,2) CHX(2,3) CHX(3,4) CHX(4,5) CHX(5,6) CHX(6,7) CHX(7,8) \
                   CHX(8,9) CHX(9,10) CHX(10,11) CHX(11,12) CHX(12,13) CHX(13,14) CHX(14,15) CHX(15,0)
#define ST_CHAIN_Y CHY(0,1) CHY(1,2) CHY(2,3) CHY(3,4) CHY(4,5) CHY(5,6) CHY(6,7) CHY(7,8) \
                   CHY(8,9) CHY(9,10) CHY(10,11) CHY(11,12) CHY(12,13) CHY(13,14) CHY(14,15) CHY(15,0)
/* Prologue: save all nonvolatile GPRs and XMM6-15, 32-byte shadow space, blk pointer at 32(%rsp). */
#define ST_PRO(name) \
    ".globl " name "\n.def " name "; .scl 2; .type 32; .endef\n.seh_proc " name "\n" name ":\n" \
    "  pushq %rbp\n  .seh_pushreg %rbp\n  pushq %rbx\n  .seh_pushreg %rbx\n  pushq %rsi\n  .seh_pushreg %rsi\n" \
    "  pushq %rdi\n  .seh_pushreg %rdi\n  pushq %r12\n  .seh_pushreg %r12\n  pushq %r13\n  .seh_pushreg %r13\n" \
    "  pushq %r14\n  .seh_pushreg %r14\n  pushq %r15\n  .seh_pushreg %r15\n" \
    "  subq $216, %rsp\n  .seh_stackalloc 216\n" \
    "  movdqa %xmm6, 48(%rsp)\n  .seh_savexmm %xmm6, 48\n  movdqa %xmm7, 64(%rsp)\n  .seh_savexmm %xmm7, 64\n" \
    "  movdqa %xmm8, 80(%rsp)\n  .seh_savexmm %xmm8, 80\n  movdqa %xmm9, 96(%rsp)\n  .seh_savexmm %xmm9, 96\n" \
    "  movdqa %xmm10, 112(%rsp)\n  .seh_savexmm %xmm10, 112\n  movdqa %xmm11, 128(%rsp)\n  .seh_savexmm %xmm11, 128\n" \
    "  movdqa %xmm12, 144(%rsp)\n  .seh_savexmm %xmm12, 144\n  movdqa %xmm13, 160(%rsp)\n  .seh_savexmm %xmm13, 160\n" \
    "  movdqa %xmm14, 176(%rsp)\n  .seh_savexmm %xmm14, 176\n  movdqa %xmm15, 192(%rsp)\n  .seh_savexmm %xmm15, 192\n" \
    "  .seh_endprologue\n  movq %rcx, 32(%rsp)\n"
#define ST_EPI \
    "  movdqa 48(%rsp), %xmm6\n  movdqa 64(%rsp), %xmm7\n  movdqa 80(%rsp), %xmm8\n  movdqa 96(%rsp), %xmm9\n" \
    "  movdqa 112(%rsp), %xmm10\n  movdqa 128(%rsp), %xmm11\n  movdqa 144(%rsp), %xmm12\n  movdqa 160(%rsp), %xmm13\n" \
    "  movdqa 176(%rsp), %xmm14\n  movdqa 192(%rsp), %xmm15\n" \
    "  addq $216, %rsp\n  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n" \
    "  popq %rdi\n  popq %rsi\n  popq %rbx\n  popq %rbp\n  ret\n.seh_endproc\n"
#define ST_HOLD(name, LD, ST, TAIL) \
    ST_PRO(name) ST_PROBES("%rcx") ST_FPLOAD("%rcx") LD \
    "  movl " A(SO_SPIN) "(%rcx), %eax\n" \
    ST_G(name "_begin") "  pause\n  decl %eax\n  jnz " name "_begin\n" ST_G(name "_end") \
    ST ST_FPSTORE("%rcx") ST_RESTORE("%rcx") TAIL ST_EPI

__asm__(
    ".text\n"
    ST_G("st_text_begin")
    ST_HOLD("st_hold_xmm",
            "  .irp i," ST_R "\n  movdqu \\i*32(%rcx), %xmm\\i\n  .endr\n",
            "  .irp i," ST_R "\n  movdqu %xmm\\i, " A(SO_VOUT) "+\\i*32(%rcx)\n  .endr\n", "")
    ST_HOLD("st_hold_ymm",
            "  .irp i," ST_R "\n  vmovdqu \\i*32(%rcx), %ymm\\i\n  .endr\n",
            "  .irp i," ST_R "\n  vmovdqu %ymm\\i, " A(SO_VOUT) "+\\i*32(%rcx)\n  .endr\n", "  vzeroupper\n")

    ST_PRO("st_compute_xmm") ST_PROBES("%rcx")
    "  .irp i," ST_R "\n  movdqu \\i*32(%rcx), %xmm\\i\n  .endr\n"
    "  movl " A(SO_ROUNDS) "(%rcx), %eax\n"
    "75:\n" ST_CHAIN_X
    "  .irp i," ST_R "\n  pshufd $0x39, %xmm\\i, %xmm\\i\n  .endr\n"
    "  decl %eax\n  jnz 75b\n"
    "  .irp i," ST_R "\n  movdqu %xmm\\i, " A(SO_COUT) "+\\i*32(%rcx)\n  .endr\n"
    ST_FPSTORE("%rcx") ST_RESTORE("%rcx") ST_EPI

    ST_PRO("st_compute_ymm") ST_PROBES("%rcx")
    "  .irp i," ST_R "\n  vmovdqu \\i*32(%rcx), %ymm\\i\n  .endr\n"
    "  movl " A(SO_ROUNDS) "(%rcx), %eax\n"
    "76:\n" ST_CHAIN_Y
    "  .irp i," ST_R "\n  vpermilps $0x39, %ymm\\i, %ymm\\i\n  .endr\n"
    "  vperm2f128 $0x01, %ymm0, %ymm0, %ymm0\n"
    "  decl %eax\n  jnz 76b\n"
    "  .irp i," ST_R "\n  vmovdqu %ymm\\i, " A(SO_COUT) "+\\i*32(%rcx)\n  .endr\n"
    "  vzeroupper\n"
    ST_FPSTORE("%rcx") ST_RESTORE("%rcx") ST_EPI

    /* st_call_xmm: probes, load XMM0-15 + MXCSR/FCW generation, call fn(arg), store, restore mode */
    ST_PRO("st_call_xmm") ST_PROBES("%rcx")
    "  .irp i," ST_R "\n  movdqu \\i*32(%rcx), %xmm\\i\n  .endr\n"
    "  movq " A(SO_FN) "(%rcx), %rax\n  movq " A(SO_ARG) "(%rcx), %rcx\n"
    "  callq *%rax\n"
    "  movq 32(%rsp), %rcx\n"
    "  .irp i," ST_R "\n  movdqu %xmm\\i, " A(SO_VOUT) "+\\i*32(%rcx)\n  .endr\n"
    "  stmxcsr " A(SO_MXOUT) "(%rcx)\n  fnstcw " A(SO_FCWOUT) "(%rcx)\n  fnstsw " A(SO_FSWOUT) "(%rcx)\n"
    ST_RESTORE("%rcx") ST_EPI

    ST_G("st_dump_xmm")
    "  .irp i," ST_R "\n  movdqu %xmm\\i, \\i*32(%rcx)\n  .endr\n  ret\n"
    ST_G("st_dump_ymm")
    "  .irp i," ST_R "\n  vmovdqu %ymm\\i, \\i*32(%rcx)\n  .endr\n  vzeroupper\n  ret\n"
    ST_G("st_text_end")
    "  nop\n"
);
#else /* x86 (cdecl, blk pointer in ECX for the hold loop, loop counter in EAX) */
#define ST_R  "0,1,2,3,4,5,6,7"
#define ST_CHAIN_X CHX(0,1) CHX(1,2) CHX(2,3) CHX(3,4) CHX(4,5) CHX(5,6) CHX(6,7) CHX(7,0)
#define ST_CHAIN_Y CHY(0,1) CHY(1,2) CHY(2,3) CHY(3,4) CHY(4,5) CHY(5,6) CHY(6,7) CHY(7,0)
#define ST_HOLD(name, LD, ST, TAIL) \
    ST_G(name) "  movl 4(%esp), %ecx\n" ST_PROBES("%ecx") ST_FPLOAD("%ecx") LD \
    "  movl " A(SO_SPIN) "(%ecx), %eax\n" \
    ST_G(name "_begin") "  pause\n  decl %eax\n  jnz " name "_begin\n" ST_G(name "_end") \
    ST ST_FPSTORE("%ecx") ST_RESTORE("%ecx") TAIL "  ret\n"

__asm__(
    ".text\n"
    ST_G("st_text_begin")
    ST_HOLD("st_hold_xmm",
            "  .irp i," ST_R "\n  movdqu \\i*32(%ecx), %xmm\\i\n  .endr\n",
            "  .irp i," ST_R "\n  movdqu %xmm\\i, " A(SO_VOUT) "+\\i*32(%ecx)\n  .endr\n", "")
    ST_HOLD("st_hold_ymm",
            "  .irp i," ST_R "\n  vmovdqu \\i*32(%ecx), %ymm\\i\n  .endr\n",
            "  .irp i," ST_R "\n  vmovdqu %ymm\\i, " A(SO_VOUT) "+\\i*32(%ecx)\n  .endr\n", "  vzeroupper\n")

    ST_G("st_compute_xmm") "  movl 4(%esp), %ecx\n" ST_PROBES("%ecx")
    "  .irp i," ST_R "\n  movdqu \\i*32(%ecx), %xmm\\i\n  .endr\n"
    "  movl " A(SO_ROUNDS) "(%ecx), %eax\n"
    "75:\n" ST_CHAIN_X
    "  .irp i," ST_R "\n  pshufd $0x39, %xmm\\i, %xmm\\i\n  .endr\n"
    "  decl %eax\n  jnz 75b\n"
    "  .irp i," ST_R "\n  movdqu %xmm\\i, " A(SO_COUT) "+\\i*32(%ecx)\n  .endr\n"
    ST_FPSTORE("%ecx") ST_RESTORE("%ecx") "  ret\n"

    ST_G("st_compute_ymm") "  movl 4(%esp), %ecx\n" ST_PROBES("%ecx")
    "  .irp i," ST_R "\n  vmovdqu \\i*32(%ecx), %ymm\\i\n  .endr\n"
    "  movl " A(SO_ROUNDS) "(%ecx), %eax\n"
    "76:\n" ST_CHAIN_Y
    "  .irp i," ST_R "\n  vpermilps $0x39, %ymm\\i, %ymm\\i\n  .endr\n"
    "  vperm2f128 $0x01, %ymm0, %ymm0, %ymm0\n"
    "  decl %eax\n  jnz 76b\n"
    "  .irp i," ST_R "\n  vmovdqu %ymm\\i, " A(SO_COUT) "+\\i*32(%ecx)\n  .endr\n"
    "  vzeroupper\n"
    ST_FPSTORE("%ecx") ST_RESTORE("%ecx") "  ret\n"

    ST_G("st_call_xmm") "  pushl %ebx\n  movl 8(%esp), %ebx\n" ST_PROBES("%ebx")
    "  .irp i," ST_R "\n  movdqu \\i*32(%ebx), %xmm\\i\n  .endr\n"
    "  pushl " A(SO_ARG) "(%ebx)\n  call *" A(SO_FN) "(%ebx)\n  addl $4, %esp\n"
    "  .irp i," ST_R "\n  movdqu %xmm\\i, " A(SO_VOUT) "+\\i*32(%ebx)\n  .endr\n"
    "  stmxcsr " A(SO_MXOUT) "(%ebx)\n  fnstcw " A(SO_FCWOUT) "(%ebx)\n  fnstsw " A(SO_FSWOUT) "(%ebx)\n"
    ST_RESTORE("%ebx") "  popl %ebx\n  ret\n"

    ST_G("st_dump_xmm") "  movl 4(%esp), %ecx\n"
    "  .irp i," ST_R "\n  movdqu %xmm\\i, \\i*32(%ecx)\n  .endr\n  ret\n"
    ST_G("st_dump_ymm") "  movl 4(%esp), %ecx\n"
    "  .irp i," ST_R "\n  vmovdqu %ymm\\i, \\i*32(%ecx)\n  .endr\n  vzeroupper\n  ret\n"
    ST_G("st_text_end")
    "  nop\n"
);
#endif

/* ======================================================================================================
 * CONTEXT accessors (FXSAVE-format legacy area on both architectures)
 * ====================================================================================================== */
#ifdef T_X64
static uintptr_t st_ip(const CONTEXT *c)  { return (uintptr_t)c->Rip; }
static uintptr_t st_cxr(const CONTEXT *c) { return (uintptr_t)c->Rcx; }
static uint64_t  st_axr(const CONTEXT *c) { return c->Rax; }
static uint8_t  *st_fx(CONTEXT *c)        { return (uint8_t *)&c->FltSave; }
#else
static uintptr_t st_ip(const CONTEXT *c)  { return c->Eip; }
static uintptr_t st_cxr(const CONTEXT *c) { return c->Ecx; }
static uint64_t  st_axr(const CONTEXT *c) { return c->Eax; }
static uint8_t  *st_fx(CONTEXT *c)        { return c->ExtendedRegisters; }
#endif
static uint32_t st_ctx_mx(CONTEXT *c)  { uint32_t v; memcpy(&v, st_fx(c) + 24, 4); return v; }
static uint16_t st_ctx_fcw(CONTEXT *c) { uint16_t v; memcpy(&v, st_fx(c), 2); return v; }
static uint16_t st_ctx_fsw(CONTEXT *c) { uint16_t v; memcpy(&v, st_fx(c) + 2, 2); return v; }

/* ======================================================================================================
 * XSTATE API (resolved at run time)
 * ====================================================================================================== */
static struct {
    int resolved;
    __typeof__(&InitializeContext) InitializeContext;
    __typeof__(&LocateXStateFeature) LocateXStateFeature;
    __typeof__(&GetXStateFeaturesMask) GetXStateFeaturesMask;
    __typeof__(&SetXStateFeaturesMask) SetXStateFeaturesMask;
    __typeof__(&GetEnabledXStateFeatures) GetEnabledXStateFeatures;
} X;
static int st_xapi(void) {
    if (!X.resolved) {
        X.InitializeContext = T_PROC("kernel32.dll", InitializeContext);
        X.LocateXStateFeature = T_PROC("kernel32.dll", LocateXStateFeature);
        X.GetXStateFeaturesMask = T_PROC("kernel32.dll", GetXStateFeaturesMask);
        X.SetXStateFeaturesMask = T_PROC("kernel32.dll", SetXStateFeaturesMask);
        X.GetEnabledXStateFeatures = T_PROC("kernel32.dll", GetEnabledXStateFeatures);
        X.resolved = 1;
    }
    return X.InitializeContext && X.LocateXStateFeature && X.GetXStateFeaturesMask && X.SetXStateFeaturesMask &&
           X.GetEnabledXStateFeatures;
}

static void *st_valloc(size_t n) { return VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }
static void st_vfree(void *p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }

/* XSTATE-aware CONTEXT: InitializeContext(NULL) -> ERROR_INSUFFICIENT_BUFFER + length, then initialize. */
typedef struct st_xc { void *buf; CONTEXT *ctx; DWORD len; DWORD err; } st_xc;
static int st_xc_init(st_xc *x, DWORD flags) {
    memset(x, 0, sizeof *x);
    DWORD len = 0;
    SetLastError(0);
    BOOL ok = X.InitializeContext(NULL, flags, NULL, &len);
    if (ok || GetLastError() != ERROR_INSUFFICIENT_BUFFER || !len) { x->err = ok ? 0xFFFFFFFFu : GetLastError(); return 0; }
    x->buf = st_valloc(len);
    if (!x->buf) { x->err = ERROR_OUTOFMEMORY; return 0; }
    x->len = len;
    if (!X.InitializeContext(x->buf, flags, &x->ctx, &x->len)) { x->err = GetLastError(); st_vfree(x->buf); x->buf = NULL; return 0; }
    return 1;
}
static void st_xc_free(st_xc *x) { st_vfree(x->buf); memset(x, 0, sizeof *x); }

/* ======================================================================================================
 * Patterns: word = marker(8) | index(10) | register(4) | word(2) | check(8) | random(32)
 * ====================================================================================================== */
enum { MK_MAIN = 0xA5, MK_APC = 0x95, MK_CB = 0xB5, MK_MMX = 0xD5, MK_X87 = 0xE5 };
static uint64_t st_word(unsigned mk, unsigned idx, unsigned reg, unsigned w, uint64_t rnd) {
    uint64_t hdr = ((uint64_t)(mk & 0xFF) << 56) | ((uint64_t)(idx & 0x3FF) << 46) | ((uint64_t)(reg & 0xF) << 42) |
                   ((uint64_t)(w & 3) << 40);
    return hdr | ((t_mix64(hdr) & 0xFF) << 32) | (rnd & 0xFFFFFFFFull);
}
static int st_decode(uint64_t v, unsigned *mk, unsigned *idx, unsigned *reg, unsigned *w) {
    unsigned m = (unsigned)(v >> 56);
    if (m != MK_MAIN && m != MK_APC && m != MK_CB && m != MK_MMX && m != MK_X87) return 0;
    uint64_t hdr = v & 0xFFFFFF0000000000ull;
    if (((v >> 32) & 0xFF) != (t_mix64(hdr) & 0xFF)) return 0;
    *mk = m; *idx = (unsigned)(v >> 46) & 0x3FF; *reg = (unsigned)(v >> 42) & 0xF; *w = (unsigned)(v >> 40) & 3;
    return 1;
}
static const char *st_mk_name(unsigned mk) {
    switch (mk) { case MK_MAIN: return "worker-main"; case MK_APC: return "worker-APC"; case MK_CB: return "timer-callback";
                  case MK_MMX: return "MMX"; case MK_X87: return "x87"; default: return "?"; }
}
static char *st_explain(uint64_t v, char *buf, size_t cap) {
    unsigned mk, idx, reg, w;
    if (!v) snprintf(buf, cap, "zero");
    else if (st_decode(v, &mk, &idx, &reg, &w)) snprintf(buf, cap, "%s pattern of #%u reg %u word %u", st_mk_name(mk), idx, reg, w);
    else snprintf(buf, cap, "not a suite pattern");
    return buf;
}
/* Explain the first differing qword of got vs want. */
static char *st_explain_diff(const void *got, const void *want, int n, char *buf, size_t cap) {
    const uint8_t *g = (const uint8_t *)got, *w = (const uint8_t *)want;
    for (int o = 0; o + 8 <= n; o += 8) {
        if (memcmp(g + o, w + o, 8)) {
            uint64_t v; memcpy(&v, g + o, 8);
            char e[96];
            snprintf(buf, cap, "observed qword %d = %s", o / 8, st_explain(v, e, sizeof e));
            return buf;
        }
    }
    snprintf(buf, cap, "difference below qword granularity");
    return buf;
}

enum { OP_HOLD, OP_X87, OP_MMX, OP_COMPUTE, OP_CALL, OP_N };
static const char *const st_op_name[OP_N] = { "hold", "hold+x87", "hold+mmx", "compute", "call" };

static int g_daz = -1;
static int st_daz(void) {
    if (g_daz < 0) {
        static uint8_t a[512] __attribute__((aligned(16)));
        memset(a, 0, sizeof a);
        t_fxsave(a);
        uint32_t m; memcpy(&m, a + 28, 4);
        if (!m) m = 0xFFBF;          /* SDM: MXCSR_MASK 0 means the default mask 0xFFBF (no DAZ) */
        g_daz = (m & 0x40) != 0;
    }
    return g_daz;
}

/* Deterministic generation contents from (seed, gen, idx, marker, op). */
static void st_fill(st_blk *b, uint64_t seed, uint32_t gen, unsigned idx, unsigned mk, int op) {
    t_rng r;
    t_rng_seed(&r, seed ^ t_mix64(((uint64_t)gen << 8) ^ mk));
    for (int i = 0; i < 16; i++)
        for (int w = 0; w < 4; w++) { uint64_t v = st_word(mk, idx, (unsigned)i, (unsigned)w, t_rng_next(&r)); memcpy(b->vin[i] + 8 * w, &v, 8); }
    for (int i = 0; i < 8; i++) {
        uint64_t x = t_rng_next(&r);
        uint64_t m = st_word(MK_X87, idx, (unsigned)i, 0, x);                 /* bit 63 set: normal */
        uint16_t se = (uint16_t)(0x3F00 + (unsigned)((x >> 40) % 0x200)) | (uint16_t)((x >> 52) & 1 ? 0x8000 : 0);
        memset(b->fin[i], 0, 16);
        memcpy(b->fin[i], &m, 8); memcpy(b->fin[i] + 8, &se, 2);
    }
    for (int i = 0; i < 8; i++) b->mmin[i] = st_word(MK_MMX, idx, (unsigned)i, 0, t_rng_next(&r));
    uint64_t x = t_rng_next(&r);
    static const uint16_t pcs[3] = { 0, 2, 3 };
    b->mx_gen = 0x1F80u | (((idx + gen) & 3u) << 13) | ((x & 1) ? 0x8000u : 0) | ((st_daz() && (x & 2)) ? 0x40u : 0) |
                (uint32_t)((x >> 8) & 0x3F);
    b->fcw_gen = (uint16_t)(0x7F | (pcs[(x >> 16) % 3] << 8) | (((idx * 3 + gen) & 3u) << 10));
    b->flags = op == OP_X87 ? 1u : op == OP_MMX ? 2u : 0u;
    b->rounds = 1 + (uint32_t)((x >> 24) % 6);
    b->one = 1.0; b->mone = -1.0; b->ten = 10.0; b->iten = 10; b->pad0 = 0;
}

/* Expected rounding-probe results. RC: 0 nearest, 1 down, 2 up, 3 toward zero (MXCSR and x87 alike). */
static int st_up(unsigned rc, int neg) { return neg ? (rc == 0 || rc == 1) : (rc == 0 || rc == 2); }
static uint64_t st_exp_sse(unsigned rc, int neg) {
    return (neg ? 0xBFB9999999999999ull : 0x3FB9999999999999ull) + (uint64_t)st_up(rc, neg);
}
static void st_exp_x87(unsigned pc, unsigned rc, int neg, uint8_t out[10]) {
    int up = st_up(rc, neg);
    uint64_t m;
    if (pc == 0) m = up ? 0xCCCCCD0000000000ull : 0xCCCCCC0000000000ull;          /* 24-bit */
    else if (pc == 2) m = up ? 0xCCCCCCCCCCCCD000ull : 0xCCCCCCCCCCCCC800ull;     /* 53-bit */
    else m = up ? 0xCCCCCCCCCCCCCCCDull : 0xCCCCCCCCCCCCCCCCull;                  /* 64-bit */
    uint16_t e = (uint16_t)(0x3FFB | (neg ? 0x8000 : 0));
    memcpy(out, &m, 8); memcpy(out + 8, &e, 2);
}

/* Reference for st_compute_*: XMM: x[i] += x[i+1] dword-wise, then PSHUFD 0x39; YMM: x[i] ^= x[i+1], VPERMILPS
 * 0x39 per lane, swap halves of x[0]. Both sequential in i, n = ST_NV, repeated `rounds` times. */
static void st_ref_compute(const st_blk *b, int vk, uint8_t out[16][32]) {
    uint32_t x[16][8];
    int n = ST_NV;
    memset(x, 0, sizeof x);
    for (int i = 0; i < n; i++) memcpy(x[i], b->vin[i], (size_t)vk);
    for (uint32_t r = 0; r < b->rounds; r++) {
        if (vk == 16) {
            for (int i = 0; i < n; i++) for (int d = 0; d < 4; d++) x[i][d] += x[(i + 1) % n][d];
            for (int i = 0; i < n; i++) { uint32_t t = x[i][0]; x[i][0] = x[i][1]; x[i][1] = x[i][2]; x[i][2] = x[i][3]; x[i][3] = t; }
        } else {
            for (int i = 0; i < n; i++) for (int d = 0; d < 8; d++) x[i][d] ^= x[(i + 1) % n][d];
            for (int i = 0; i < n; i++) for (int l = 0; l < 8; l += 4) {
                uint32_t t = x[i][l]; x[i][l] = x[i][l + 1]; x[i][l + 1] = x[i][l + 2]; x[i][l + 2] = x[i][l + 3]; x[i][l + 3] = t;
            }
            for (int d = 0; d < 4; d++) { uint32_t t = x[0][d]; x[0][d] = x[0][d + 4]; x[0][d + 4] = t; }
        }
    }
    for (int i = 0; i < n; i++) memcpy(out[i], x[i], (size_t)vk);
}

/* ======================================================================================================
 * Configuration, workers, run state
 * ====================================================================================================== */
#define OPB(x) (1u << (x))
#define OPS_HOLDS (OPB(OP_HOLD) | OPB(OP_X87) | OPB(OP_MMX))
#define OPS_ALL   ((1u << OP_N) - 1)
enum { K_XMM = 1, K_YMM = 2, K_MIX = 3 };
enum { Y_SPIN, Y_SWITCH, Y_SLEEP0, Y_SLEEP1, Y_PINGPONG, Y_WTIMER, Y_APC, Y_ALERT, Y_CS, Y_SRW, Y_SEMA, Y_SYSCALL,
       Y_HOP, Y_PRIO, Y_N };
#define YB(x) (1u << (x))
#define Y_ALL ((1u << Y_N) - 1)
static const char *const st_y_name[Y_N] = { "spin", "SwitchToThread", "Sleep(0)", "Sleep(1)", "event-pingpong",
    "waitable-timer", "self-APC", "alertable-SleepEx", "critical-section", "SRW-lock", "semaphore", "syscalls",
    "affinity-hop", "priority-change" };
enum { C_SUSPEND = 1, C_GET = 2, C_GETX = 4, C_XOPT = 8, C_SET = 16, C_AFF = 32, C_PRIO = 64, C_APC = 128 };
#define C_CAPTURE (C_SUSPEND | C_GET | C_GETX | C_XOPT | C_SET)
enum { T_TQ = 1, T_MM = 2, TA_APC = 4, TA_WAKE = 8, TA_KICK = 16, TA_REGS = 32, TA_POOL = 64 };
enum { PH_C, PH_ASM, PH_CALL };
enum { T_2XCPU = -1 };

typedef struct st_cfg {
    int threads;        /* > 0 fixed count, T_2XCPU = 2 x logical CPUs (<= 256) */
    int ms;             /* run time */
    unsigned ops, yields, ctl, tmr;
    int kind;
    unsigned spin;      /* PAUSE iterations per hold: [spin, 2*spin) */
    int mutate;         /* 1 xmm3 low bit, 2 MXCSR RC, 3 ymm5 upper bit (self-test of the oracle) */
    int churn;          /* create short-lived threads during the run and check their initial state */
    int need_hold;      /* CHECK that the oracle caught the worker inside a hold loop */
    int perf;           /* report rates */
    int pin;            /* PIN_ONE: every worker on one CPU; PIN_PAIRS: two workers per CPU (forced time slicing) */
    int hold_ms;        /* > 0: spin calibrated so one hold lasts about this long (overrides spin) */
} st_cfg;
enum { PIN_NONE, PIN_ONE, PIN_PAIRS };

typedef struct st_slot {
    st_blk b;
    volatile LONG seq, gen, op, vk;     /* seqlock: seq odd while (re)publishing */
    uint64_t seed;
    unsigned mk, idx;
} __attribute__((aligned(64))) st_slot;

struct st_run;
typedef struct st_worker {
    st_slot main, apc;
    struct st_run *S;
    const char *role;
    int idx;
    HANDLE h; DWORD tid;
    uint64_t seed; t_rng rng;
    uint32_t mx_mode; uint16_t fcw_mode;
    volatile LONG phase, started, finished;
    volatile LONG susp_lock;
    volatile LONG apc_queued, apc_done;
    HANDLE ev, wt;
    struct st_worker *peer;
    int call_yield;
    LONG64 iters;
    LONG bad, details, mode_bad, apc_iters;
    uint32_t ops_done[OP_N], yields_done[Y_N];
    uintptr_t pin_mask;          /* nonzero: affinity the worker sets for itself at start */
    LONG holds_timed, holds_stretched;
    int nmut;
    struct { LONG gen; int what, reg, n; uint8_t got[32], want[32]; } mut[4];
} __attribute__((aligned(64))) st_worker;

typedef struct st_run {
    st_cfg c;
    const char *name;
    uint64_t seed;
    int n, avx, xs, min_iters;
    st_worker *w;                       /* n workers + ST_NCB timer-callback pseudo workers */
    st_worker *cbw;
    volatile LONG stop, hard_stop, ctl_stop, abort;
    HANDLE ev_go, sema, ev_set;
    CRITICAL_SECTION cs; SRWLOCK srw;
    uintptr_t procmask; int ncpu;
    /* controller */
    CONTEXT *ctx; st_xc xc; st_blk *scratch;
    LONG64 caps, susp_ops, hold_hits, hold_hits_apc, asm_other, outside, out_checked, apc_skip, sets, aff_ops, prio_ops,
           apc_ops, upper_sys, x87_nonempty, ctl_bad;
    double cap_ms;
    /* timers */
    volatile LONG tmr_on, tmr_inflight, tmr_ticks, tmr_apc, tmr_wake, tmr_kick, tmr_regs, tmr_bad, tmr_pool, tmr_pool_fail;
    LONG64 stretch_qpc;                 /* QPC ticks: a hold longer than this was descheduled (INFO only) */
    volatile LONG cb_lock[ST_NCB];
    /* mutation self-test */
    volatile LONG mut_done; LONG mut_gen; int mut_reg; uint8_t mut_old[16], mut_new[16]; uint32_t mut_mx_old, mut_mx_new;
    volatile LONG fail_details;
    int wine; LONG64 wine_zero_fp, wine_host_ip;
} st_run;
static st_run g_S;

static int st_pick_bit(unsigned mask, unsigned r) {
    int cnt = 0;
    for (unsigned m = mask; m; m &= m - 1) cnt++;
    if (!cnt) return 0;
    int k = (int)(r % (unsigned)cnt);
    for (int b = 0; b < 32; b++) if (mask & (1u << b)) { if (!k--) return b; }
    return 0;
}
static int st_pick_vk(const st_run *S, uint64_t r) {
    if (S->c.kind == K_YMM) return 32;
    if (S->c.kind == K_MIX && S->avx) return (r & 1) ? 32 : 16;
    return 16;
}
static uintptr_t st_nth_cpu(uintptr_t mask, unsigned k) {
    for (int b = 0; b < (int)(8 * sizeof mask); b++) if (mask & ((uintptr_t)1 << b)) { if (!k--) return (uintptr_t)1 << b; }
    return mask;
}
static int st_detail_ok(st_run *S, st_worker *w) {
    if (w && w->details >= 4) return 0;
    if (InterlockedIncrement(&S->fail_details) > 30) return 0;
    if (w) w->details++;
    return 1;
}

static void st_publish(st_slot *s, LONG gen, int op, int vk, uint32_t mxm, uint16_t fcwm, unsigned spin) {
    InterlockedIncrement(&s->seq);                      /* odd: publishing */
    s->gen = gen; s->op = op; s->vk = vk;
    st_fill(&s->b, s->seed, (uint32_t)gen, s->idx, s->mk, op);
    s->b.spin = spin ? spin : 1;
    s->b.mx_mode = mxm; s->b.fcw_mode = fcwm;
    InterlockedIncrement(&s->seq);                      /* even: stable */
}

/* ======================================================================================================
 * Worker-side verification
 * ====================================================================================================== */
static void st_wbad(st_worker *w, const st_slot *s, int what, int reg, const void *got, const void *want, int n,
                    const char *label) {
    st_run *S = w->S;
    w->bad++;
    if (S->c.mutate) {
        if (w->nmut < 4) {
            w->mut[w->nmut].gen = s->gen; w->mut[w->nmut].what = what; w->mut[w->nmut].reg = reg; w->mut[w->nmut].n = n;
            memcpy(w->mut[w->nmut].got, got, (size_t)n); memcpy(w->mut[w->nmut].want, want, (size_t)n);
        }
        w->nmut++;
        return;
    }
    if (!st_detail_ok(S, w)) return;
    char lab[320], ex[128];
    snprintf(lab, sizeof lab, "%s #%d tid %lu gen %ld (%s slot, op %s, %d-byte vectors): %s [%s]", w->role, w->idx,
             (unsigned long)w->tid, (long)s->gen, s == &w->apc ? "APC" : "main", st_op_name[s->op % OP_N], (int)s->vk,
             label, st_explain_diff(got, want, n, ex, sizeof ex));
    CHECK_REG(lab, got, want, (size_t)n);
}
static void st_wbad_val(st_worker *w, const st_slot *s, int what, const char *label, uint32_t got, uint32_t want) {
    st_run *S = w->S;
    w->bad++;
    if (S->c.mutate) {
        if (w->nmut < 4) {
            w->mut[w->nmut].gen = s->gen; w->mut[w->nmut].what = what; w->mut[w->nmut].reg = -1; w->mut[w->nmut].n = 4;
            memcpy(w->mut[w->nmut].got, &got, 4); memcpy(w->mut[w->nmut].want, &want, 4);
        }
        w->nmut++;
        return;
    }
    if (!st_detail_ok(S, w)) return;
    CHECKF(0, "%s #%d tid %lu gen %ld (%s slot, op %s): %s = 0x%04x, expected 0x%04x", w->role, w->idx, (unsigned long)w->tid,
           (long)s->gen, s == &w->apc ? "APC" : "main", st_op_name[s->op % OP_N], label, got, want);
}
enum { W_VEC = 1, W_MX, W_FCW, W_FSW, W_ST, W_MM, W_PSSE, W_PX87, W_COMP };

static void st_verify(st_worker *w, st_slot *s) {
    st_blk *b = &s->b;
    int op = s->op, vk = s->vk, n = ST_NV;
    char lab[64];
    if (op == OP_COMPUTE) {
        uint8_t ref[16][32];
        st_ref_compute(b, vk, ref);
        for (int i = 0; i < n; i++) if (memcmp(b->cout[i], ref[i], (size_t)vk)) {
            snprintf(lab, sizeof lab, "%s%d after compute chain", vk == 32 ? "ymm" : "xmm", i);
            st_wbad(w, s, W_COMP, i, b->cout[i], ref[i], vk, lab);
        }
    } else if (op == OP_CALL) {
#ifdef T_X64
        for (int i = 6; i < 16; i++) if (memcmp(b->vout[i], b->vin[i], 16)) {       /* nonvolatile low 128 bits */
            snprintf(lab, sizeof lab, "nonvolatile xmm%d across yielding call", i);
            st_wbad(w, s, W_VEC, i, b->vout[i], b->vin[i], 16, lab);
        }
#endif
    } else {
        for (int i = 0; i < n; i++) if (memcmp(b->vout[i], b->vin[i], (size_t)vk)) {
            snprintf(lab, sizeof lab, "%s%d after hold", vk == 32 ? "ymm" : "xmm", i);
            st_wbad(w, s, W_VEC, i, b->vout[i], b->vin[i], vk, lab);
        }
    }
    if (op == OP_CALL) {
        if ((b->mx_out ^ b->mx_gen) & MX_CTRL) st_wbad_val(w, s, W_MX, "MXCSR control bits after call", b->mx_out, b->mx_gen);
    } else if (b->mx_out != b->mx_gen) st_wbad_val(w, s, W_MX, "MXCSR (exact, sticky flags included)", b->mx_out, b->mx_gen);
    if (b->fcw_out != b->fcw_gen) st_wbad_val(w, s, W_FCW, "x87 control word", b->fcw_out, b->fcw_gen);
    if (op != OP_CALL && (b->fsw_out & 0x38FF))
        st_wbad_val(w, s, W_FSW, "x87 status word (TOP / exception flags)", b->fsw_out, 0);
    if (op == OP_X87) for (int i = 0; i < 8; i++) if (memcmp(b->fout[i], b->fin[i], 10)) {
        snprintf(lab, sizeof lab, "ST(%d) after hold", i);
        st_wbad(w, s, W_ST, i, b->fout[i], b->fin[i], 10, lab);
    }
    if (op == OP_MMX) for (int i = 0; i < 8; i++) if (b->mmout[i] != b->mmin[i]) {
        snprintf(lab, sizeof lab, "mm%d after hold", i);
        st_wbad(w, s, W_MM, i, &b->mmout[i], &b->mmin[i], 8, lab);
    }
    unsigned rc = (b->mx_gen >> 13) & 3;
    for (int k = 0; k < 2; k++) {
        uint64_t e = st_exp_sse(rc, k);
        if (b->psse[k] != e) {
            snprintf(lab, sizeof lab, "DIVSD %s1/10 under MXCSR RC=%u", k ? "-" : "", rc);
            st_wbad(w, s, W_PSSE, k, &b->psse[k], &e, 8, lab);
        }
        uint8_t ex[10];
        unsigned pc = (b->fcw_gen >> 8) & 3, xrc = (b->fcw_gen >> 10) & 3;
        st_exp_x87(pc, xrc, k, ex);
        if (memcmp(b->px87[k], ex, 10)) {
            snprintf(lab, sizeof lab, "FIDIV %s1/10 under FCW PC=%u RC=%u", k ? "-" : "", pc, xrc);
            st_wbad(w, s, W_PX87, k, b->px87[k], ex, 10, lab);
        }
    }
}

static void st_mode_check(st_worker *w, const char *where) {
    uint32_t mx = t_get_mxcsr();
    uint16_t cw = t_get_x87_cw();
    if (!((mx ^ w->mx_mode) & MX_CTRL) && cw == w->fcw_mode) return;
    w->mode_bad++; w->bad++;
    if (!w->S->c.mutate && st_detail_ok(w->S, w))
        CHECKF(0, "%s #%d tid %lu %s: MXCSR 0x%04x (thread mode 0x%04x, control bits compared) FCW 0x%04x (thread mode 0x%04x)",
               w->role, w->idx, (unsigned long)w->tid, where, mx, w->mx_mode, cw, w->fcw_mode);
    t_set_mxcsr(w->mx_mode); t_set_x87_cw(w->fcw_mode);
}

/* ======================================================================================================
 * Worker: iterations, yields, APCs
 * ====================================================================================================== */
static void st_run_slot(st_slot *s) {
    switch (s->op) {
    case OP_COMPUTE: if (s->vk == 32) st_compute_ymm(&s->b); else st_compute_xmm(&s->b); break;
    case OP_CALL:    st_call_xmm(&s->b); break;
    default:         if (s->vk == 32) st_hold_ymm(&s->b); else st_hold_xmm(&s->b); break;
    }
}

/* User APC (queued by the worker itself, the controller or a timer callback): hold a fresh APC-slot generation
 * and verify it. It restores the MXCSR / CW it found (it is an ordinary function). */
static VOID CALLBACK st_apc(ULONG_PTR p) {
    st_worker *w = (st_worker *)p;
    st_run *S = w->S;
    uint64_t x = t_rng_next(&w->rng);
    st_slot *s = &w->apc;
    int op = (x & 6) == 0 ? OP_X87 : (x & 6) == 2 ? OP_MMX : OP_HOLD;
    st_publish(s, s->gen + 1, op, st_pick_vk(S, x >> 8), t_get_mxcsr(), t_get_x87_cw(), S->c.spin / 2 + 1);
    st_run_slot(s);
    st_verify(w, s);
    w->apc_iters++;
    InterlockedIncrement(&w->apc_done);
}
static void st_queue_apc(st_worker *w) {
    InterlockedIncrement(&w->apc_queued);
    if (!QueueUserAPC(st_apc, w->h, (ULONG_PTR)w)) InterlockedIncrement(&w->apc_done);
}

static void st_yield(st_worker *w, int y) {
    st_run *S = w->S;
    uint64_t x = t_rng_next(&w->rng);
    w->yields_done[y]++;
    switch (y) {
    case Y_SPIN: { volatile unsigned k = 0; for (unsigned i = 0; i < 2000; i++) k += i; break; }
    case Y_SWITCH: SwitchToThread(); break;
    case Y_SLEEP0: Sleep(0); break;
    case Y_SLEEP1: Sleep(1); break;
    case Y_PINGPONG:
        if (w->peer) SetEvent(w->peer->ev);
        WaitForSingleObjectEx(w->ev, 2, TRUE);
        break;
    case Y_WTIMER: {
        LARGE_INTEGER due; due.QuadPart = -5000;      /* 0.5 ms relative */
        if (w->wt && SetWaitableTimer(w->wt, &due, 0, NULL, NULL, FALSE)) WaitForSingleObjectEx(w->wt, 50, TRUE);
        else Sleep(0);
        break;
    }
    case Y_APC:
        InterlockedIncrement(&w->apc_queued);
        if (QueueUserAPC(st_apc, GetCurrentThread(), (ULONG_PTR)w)) SleepEx(0, TRUE);
        else InterlockedIncrement(&w->apc_done);
        break;
    case Y_ALERT: SleepEx((DWORD)(x & 1), TRUE); break;
    case Y_CS:
        EnterCriticalSection(&S->cs);
        { volatile unsigned k = 0; for (unsigned i = 0; i < 300; i++) k += i; }
        LeaveCriticalSection(&S->cs);
        break;
    case Y_SRW:
        if (x & 1) { AcquireSRWLockShared(&S->srw); SwitchToThread(); ReleaseSRWLockShared(&S->srw); }
        else { AcquireSRWLockExclusive(&S->srw); { volatile unsigned k = 0; for (unsigned i = 0; i < 300; i++) k += i; } ReleaseSRWLockExclusive(&S->srw); }
        break;
    case Y_SEMA:
        if (WaitForSingleObjectEx(S->sema, 2, TRUE) == WAIT_OBJECT_0) { SwitchToThread(); ReleaseSemaphore(S->sema, 1, NULL); }
        break;
    case Y_SYSCALL: {
        void *p = VirtualAlloc(NULL, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (p) { ((volatile char *)p)[(x >> 8) % 65536] = 1; VirtualFree(p, 0, MEM_RELEASE); }
        FILETIME a, b2, c2, d;
        GetThreadTimes(GetCurrentThread(), &a, &b2, &c2, &d);
        WaitForSingleObject(S->ev_set, 0);
        HANDLE dup = NULL;
        if (DuplicateHandle(GetCurrentProcess(), S->ev_set, GetCurrentProcess(), &dup, 0, FALSE, DUPLICATE_SAME_ACCESS)) CloseHandle(dup);
        break;
    }
    case Y_HOP:
        if (S->ncpu > 1) {
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)st_nth_cpu(S->procmask, (unsigned)((x >> 8) % (unsigned)S->ncpu)));
            SwitchToThread();
            if (x & 1) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)S->procmask);
        } else SwitchToThread();
        break;
    case Y_PRIO:      /* only lower / normal: never starve the case thread or the controller */
        SetThreadPriority(GetCurrentThread(), -(int)((x >> 8) % 3));
        break;
    }
}

static void st_call_body(void *p) {
    st_worker *w = (st_worker *)p;
    st_yield(w, w->call_yield);
}

static void st_iter(st_worker *w) {
    st_run *S = w->S;
    uint64_t x = t_rng_next(&w->rng);
    int op = st_pick_bit(S->c.ops, (unsigned)x);
    int vk = op == OP_CALL ? 16 : st_pick_vk(S, x >> 20);
    st_slot *s = &w->main;
    st_publish(s, s->gen + 1, op, vk, w->mx_mode, w->fcw_mode, S->c.spin + (unsigned)((x >> 32) % S->c.spin));
    if (op == OP_CALL) {
        w->call_yield = st_pick_bit(S->c.yields, (unsigned)(x >> 40));
        s->b.fn = (uint64_t)(uintptr_t)st_call_body;
        s->b.arg = (uint64_t)(uintptr_t)w;
        w->phase = PH_CALL;
    } else w->phase = PH_ASM;
    /* Integer QPC deltas only: this C code runs under the thread's x87 mode (PC may be 24-bit on x86). */
    LARGE_INTEGER q0 = { 0 }, q1 = { 0 };
    int timed = S->stretch_qpc && (op == OP_HOLD || op == OP_X87 || op == OP_MMX);
    if (timed) QueryPerformanceCounter(&q0);
    st_run_slot(s);
    if (timed) {
        QueryPerformanceCounter(&q1);
        w->holds_timed++;
        if (q1.QuadPart - q0.QuadPart > S->stretch_qpc) w->holds_stretched++;
    }
    w->phase = PH_C;
    st_verify(w, s);
    w->ops_done[op]++;
    st_mode_check(w, "after asm routine");
    st_yield(w, st_pick_bit(S->c.yields, (unsigned)(x >> 48)));
    st_mode_check(w, "after yield");
    w->iters++;
}

static DWORD WINAPI st_worker_main(void *p) {
    st_worker *w = (st_worker *)p;
    st_run *S = w->S;
    w->tid = GetCurrentThreadId();
    if (w->pin_mask) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)w->pin_mask);
    t_set_mxcsr(w->mx_mode);
    t_set_x87_cw(w->fcw_mode);
    InterlockedExchange(&w->started, 1);
    WaitForSingleObject(S->ev_go, 15000);
    if (!S->abort) {
        while (!S->hard_stop) {
            if (S->stop && w->iters >= S->min_iters) break;
            st_iter(w);
        }
        SleepEx(0, TRUE);                               /* deliver pending APCs while the run state is alive */
        st_mode_check(w, "at worker exit");
    }
    InterlockedExchange(&w->finished, 1);
    return 0;
}

/* ======================================================================================================
 * Controller: suspend / capture / oracle / set unchanged / resume, affinity, priority, APC injection
 * ====================================================================================================== */
enum { R_NONE, R_HOLD, R_ASM, R_OUT };
enum { E_RCX = 1, E_SEQ, E_KIND, E_COUNT, E_REGEN, E_VLO, E_VHI, E_MX, E_MX2, E_FCW, E_TOP, E_TAG, E_ST, E_MM, E_MASK,
       E_LOC, E_FOREIGN, E_OUT_MX, E_OUT_FCW, E_XMASK };
typedef struct st_err { int code, reg, n; uint64_t a, b; uint8_t got[16], want[16]; } st_err;
typedef struct st_rep {
    int region, apc_slot, nerr, nall, oracle;
    LONG gen, op, vk, phase;
    uintptr_t ip;
    DWORD prev, resume, get_err, set_err, susp_err;
    int got_ctx, did_set, mutated, wine_zero;
    st_err e[4];
} st_rep;
static void rep_err(st_rep *r, int code, int reg, const void *got, const void *want, int n, uint64_t a, uint64_t b) {
    r->nall++;
    if (r->nerr >= 4) return;
    st_err *e = &r->e[r->nerr++];
    e->code = code; e->reg = reg; e->n = n; e->a = a; e->b = b;
    if (n) { memcpy(e->got, got, (size_t)n); memcpy(e->want, want, (size_t)n); }
}

static int st_nonzero(const uint8_t *p, int n) { for (int i = 0; i < n; i++) if (p[i]) return 1; return 0; }

/* Runs while the target is suspended: no heap, no locks, no CRT formatting. */
static void st_analyze(st_run *S, st_worker *w, CONTEXT *c, st_rep *r) {
    uintptr_t ip = st_ip(c);
    uint8_t *fx = st_fx(c);
    r->ip = ip;
    int hk = 0;
    if (ip >= (uintptr_t)st_hold_xmm_begin && ip <= (uintptr_t)st_hold_xmm_end) hk = 16;
    else if (ip >= (uintptr_t)st_hold_ymm_begin && ip <= (uintptr_t)st_hold_ymm_end) hk = 32;
    if (hk) {
        r->region = R_HOLD;
        uintptr_t bp = st_cxr(c);
        st_slot *s = bp == (uintptr_t)&w->main.b ? &w->main : bp == (uintptr_t)&w->apc.b ? &w->apc : NULL;
        if (!s) { rep_err(r, E_RCX, -1, NULL, NULL, 0, bp, (uintptr_t)&w->main.b); return; }
        r->apc_slot = s == &w->apc;
        r->oracle = 1;
        LONG seq = s->seq;
        r->gen = s->gen; r->op = s->op; r->vk = s->vk;
        if (seq & 1) rep_err(r, E_SEQ, -1, NULL, NULL, 0, (uint64_t)seq, 0);
        if (r->vk != hk) rep_err(r, E_KIND, -1, NULL, NULL, 0, (uint64_t)hk, (uint64_t)r->vk);
        uint64_t cnt = st_axr(c) & 0xFFFFFFFFull;
        if (cnt > s->b.spin) rep_err(r, E_COUNT, -1, NULL, NULL, 0, cnt, s->b.spin);
        st_blk *e = S->scratch;
        st_fill(e, s->seed, (uint32_t)r->gen, s->idx, s->mk, r->op);
        if (memcmp(e->vin, s->b.vin, sizeof e->vin) || e->mx_gen != s->b.mx_gen || e->fcw_gen != s->b.fcw_gen ||
            memcmp(e->fin, s->b.fin, sizeof e->fin) || memcmp(e->mmin, s->b.mmin, sizeof e->mmin))
            rep_err(r, E_REGEN, -1, NULL, NULL, 0, 0, 0);
        for (int i = 0; i < ST_NV; i++)
            if (memcmp(fx + 160 + 16 * i, e->vin[i], 16)) rep_err(r, E_VLO, i, fx + 160 + 16 * i, e->vin[i], 16, 0, 0);
        uint32_t mx = st_ctx_mx(c);
        if (mx != e->mx_gen) rep_err(r, E_MX, -1, NULL, NULL, 0, mx, e->mx_gen);
#ifdef T_X64
        if (c->MxCsr != mx) rep_err(r, E_MX2, -1, NULL, NULL, 0, c->MxCsr, mx);
#endif
        uint16_t fcw = st_ctx_fcw(c), fsw = st_ctx_fsw(c);
        if (fcw != e->fcw_gen) rep_err(r, E_FCW, -1, NULL, NULL, 0, fcw, e->fcw_gen);
        if ((fsw >> 11) & 7) rep_err(r, E_TOP, -1, NULL, NULL, 0, fsw, 0);
        uint8_t tag = fx[4], wtag = (r->op == OP_X87 || r->op == OP_MMX) ? 0xFF : 0x00;
        if (tag != wtag) rep_err(r, E_TAG, -1, NULL, NULL, 0, tag, wtag);
        if (r->op == OP_X87) {
            for (int i = 0; i < 8; i++)
                if (memcmp(fx + 32 + 16 * i, e->fin[i], 10)) rep_err(r, E_ST, i, fx + 32 + 16 * i, e->fin[i], 10, 0, 0);
        } else if (r->op == OP_MMX) {
            for (int i = 0; i < 8; i++) {
                uint8_t want[10];
                memcpy(want, &e->mmin[i], 8); want[8] = 0xFF; want[9] = 0xFF;
                if (memcmp(fx + 32 + 16 * i, want, 10)) rep_err(r, E_MM, i, fx + 32 + 16 * i, want, 10, 0, 0);
            }
        }
        if (S->xs) {
            DWORD64 m = 0;
            if (!X.GetXStateFeaturesMask(c, &m)) { rep_err(r, E_XMASK, -1, NULL, NULL, 0, GetLastError(), 0); return; }
            DWORD len = 0;
            uint8_t *up = (uint8_t *)X.LocateXStateFeature(c, ST_XS_AVX, &len);
            if (r->vk == 32) {
                if (!(m & T_XS_AVX)) rep_err(r, E_MASK, -1, NULL, NULL, 0, m, T_XS_AVX);
                else if (!up) rep_err(r, E_LOC, -1, NULL, NULL, 0, len, 0);
                else for (int i = 0; i < ST_NV; i++)
                    if (memcmp(up + 16 * i, e->vin[i] + 16, 16)) rep_err(r, E_VHI, i, up + 16 * i, e->vin[i] + 16, 16, 0, 0);
            } else if ((m & T_XS_AVX) && up) {
                /* XMM-only hold: the upper halves were cleared by VZEROUPPER and legacy SSE never writes them.
                 * System code may leave its own data there (volatile), but never a suite pattern. */
                for (int i = 0; i < ST_NV; i++) {
                    if (!st_nonzero(up + 16 * i, 16)) continue;
                    int foreign = 0;
                    for (int q = 0; q < 2; q++) {
                        uint64_t v; unsigned a, b2, c2, d2;
                        memcpy(&v, up + 16 * i + 8 * q, 8);
                        if (st_decode(v, &a, &b2, &c2, &d2)) foreign = 1;
                    }
                    if (foreign) { static const uint8_t z[16]; rep_err(r, E_FOREIGN, i, up + 16 * i, z, 16, 0, 0); }
                    else S->upper_sys++;
                }
            }
        }
    } else if (ip >= (uintptr_t)st_text_begin && ip < (uintptr_t)st_text_end) {
        r->region = R_ASM;
    } else {
        r->region = R_OUT;
        if (w->apc_queued != w->apc_done) { S->apc_skip++; return; }   /* an APC may be in delivery */
        r->oracle = 1;
        r->phase = w->phase;
        uint32_t mx = st_ctx_mx(c);
        uint16_t fcw = st_ctx_fcw(c);
        int okmx = !((mx ^ w->mx_mode) & MX_CTRL), okcw = fcw == w->fcw_mode;
        if (r->phase == PH_CALL) {
            okmx = okmx || !((mx ^ w->main.b.mx_gen) & MX_CTRL);
            okcw = okcw || fcw == w->main.b.fcw_gen;
        }
        if ((!okmx || !okcw) && S->wine) {
            /* Known Wine gaps, applied only to a capture that would otherwise fail (Windows: never applied):
             * x86 - a thread captured inside a system call reports an all-zero legacy FP area (no thread here
             *       can hold FCW 0: every mode masks all x87 exceptions);
             * x64 - a thread captured while running host-side (Unix) code of a system call reports that code's
             *       IP and FP environment instead of the PE-side frame (IP outside any MEM_IMAGE mapping).
             * VirtualQuery is a plain system call without user-mode locks: safe while the target is suspended. */
            MEMORY_BASIC_INFORMATION mbi;
            if (fcw == 0 && mx == 0) { r->oracle = 0; r->wine_zero = 1; S->wine_zero_fp++; return; }
            if (!VirtualQuery((const void *)ip, &mbi, sizeof mbi) || mbi.Type != MEM_IMAGE) {
                r->oracle = 0; r->wine_zero = 1; S->wine_host_ip++; return;
            }
        }
        if (!okmx) rep_err(r, E_OUT_MX, -1, NULL, NULL, 0, mx, w->mx_mode);
        if (!okcw) rep_err(r, E_OUT_FCW, -1, NULL, NULL, 0, fcw, w->fcw_mode);
        S->out_checked++;
        if (fx[4] != 0) S->x87_nonempty++;
    }
}

static const char *st_ecode(int c) {
    switch (c) {
    case E_RCX: return "captured RCX/ECX inside the hold loop is not one of the thread's blocks";
    case E_SEQ: return "seqlock odd while the thread sits in a hold loop";
    case E_KIND: return "hold routine width differs from the published generation";
    case E_COUNT: return "captured loop counter (RAX/EAX) above the published spin count";
    case E_REGEN: return "published generation differs from its re-derivation from the seed";
    case E_MX: return "captured MXCSR (FXSAVE image)";
    case E_MX2: return "CONTEXT.MxCsr differs from FltSave.MxCsr";
    case E_FCW: return "captured x87 control word";
    case E_TOP: return "captured x87 TOP field not 0";
    case E_TAG: return "captured abridged x87 tag word";
    case E_MASK: return "XSTATE feature mask lacks AVX while the upper halves hold a pattern";
    case E_LOC: return "LocateXStateFeature(AVX) returned NULL";
    case E_OUT_MX: return "captured MXCSR control bits outside the asm are not the thread's mode/generation";
    case E_OUT_FCW: return "captured x87 CW outside the asm is not the thread's mode/generation";
    case E_XMASK: return "GetXStateFeaturesMask failed";
    default: return "?";
    }
}

static void st_ctl_report(st_run *S, st_worker *w, const st_rep *r) {
    if (r->susp_err) { S->ctl_bad++; CHECKF(0, "SuspendThread(%s #%d tid %lu) failed, GLE=%lu", w->role, w->idx, (unsigned long)w->tid, (unsigned long)r->susp_err); return; }
    if (r->prev != 0) { S->ctl_bad++; CHECKF(0, "SuspendThread(%s #%d) previous count %lu, expected 0", w->role, w->idx, (unsigned long)r->prev); }
    if (r->resume != 1) { S->ctl_bad++; CHECKF(0, "ResumeThread(%s #%d) previous count %lu, expected 1 (GLE=%lu)", w->role, w->idx, (unsigned long)r->resume, (unsigned long)GetLastError()); }
    if (r->get_err) { S->ctl_bad++; if (st_detail_ok(S, NULL)) CHECKF(0, "GetThreadContext(%s #%d tid %lu) failed, GLE=%lu", w->role, w->idx, (unsigned long)w->tid, (unsigned long)r->get_err); }
    if (r->set_err) { S->ctl_bad++; if (st_detail_ok(S, NULL)) CHECKF(0, "SetThreadContext(unchanged, %s #%d tid %lu, %s) failed, GLE=%lu", w->role, w->idx, (unsigned long)w->tid, r->region == R_HOLD ? "full" : "FP only", (unsigned long)r->set_err); }
    if (!r->oracle) return;
    S->ctl_bad += r->nall;
    char where[256];
    if (r->region == R_HOLD)
        snprintf(where, sizeof where, "ctl oracle: %s #%d tid %lu captured in %s hold loop at IP %p, gen %ld (%s slot, op %s)",
                 w->role, w->idx, (unsigned long)w->tid, r->vk == 32 ? "YMM" : "XMM", (void *)r->ip, (long)r->gen,
                 r->apc_slot ? "APC" : "main", st_op_name[r->op % OP_N]);
    else
        snprintf(where, sizeof where, "ctl oracle: %s #%d tid %lu captured outside the asm at IP %p (phase %ld)",
                 w->role, w->idx, (unsigned long)w->tid, (void *)r->ip, (long)r->phase);
    CHECKF(r->nall == 0, "%s: %d mismatch(es)", where, r->nall);
    if (!r->nall) return;
    for (int i = 0; i < r->nerr; i++) {
        if (!st_detail_ok(S, NULL)) break;
        const st_err *e = &r->e[i];
        char lab[480], ex[128];
        switch (e->code) {
        case E_VLO: case E_VHI: case E_ST: case E_MM: case E_FOREIGN: {
            char reg[96];
            if (e->code == E_ST) snprintf(reg, sizeof reg, "ST(%d) (FXSAVE image)", e->reg);
            else if (e->code == E_MM) snprintf(reg, sizeof reg, "mm%d (ST(%d) image, exponent must be 0xFFFF)", e->reg, e->reg);
            else if (e->code == E_VLO) snprintf(reg, sizeof reg, "xmm%d (low 128)", e->reg);
            else if (e->code == E_VHI) snprintf(reg, sizeof reg, "ymm%d upper 128", e->reg);
            else snprintf(reg, sizeof reg, "ymm%d upper 128 holds a suite pattern during an XMM-only hold (stale/foreign state)", e->reg);
            snprintf(lab, sizeof lab, "%s: %s [%s]", where, reg, st_explain_diff(e->got, e->want, e->n, ex, sizeof ex));
            CHECK_REG(lab, e->got, e->want, (size_t)e->n);
            break;
        }
        default:
            CHECKF(0, "%s: %s: observed 0x%llx expected 0x%llx", where, st_ecode(e->code), (unsigned long long)e->a, (unsigned long long)e->b);
            break;
        }
    }
}

/* Apply the configured one-shot mutation to the captured context (self-test of the worker check). */
static int st_mutate(st_run *S, st_worker *w, CONTEXT *c, const st_rep *r) {
    if (S->mut_done || r->region != R_HOLD || r->apc_slot || r->nall || r->op != OP_HOLD) return 0;
    uint8_t *fx = st_fx(c);
    if (S->c.mutate == 1) {
        S->mut_reg = 3;
        memcpy(S->mut_old, fx + 160 + 16 * 3, 16);
        fx[160 + 16 * 3 + 5] ^= 0x10;
        memcpy(S->mut_new, fx + 160 + 16 * 3, 16);
    } else if (S->c.mutate == 2) {
        uint32_t mx = st_ctx_mx(c);
        S->mut_mx_old = mx; S->mut_mx_new = mx ^ 0x6000u;          /* flip both RC bits */
        memcpy(fx + 24, &S->mut_mx_new, 4);
#ifdef T_X64
        c->MxCsr = S->mut_mx_new;
#endif
    } else if (S->c.mutate == 3) {
        if (!S->xs || r->vk != 32) return 0;
        uint8_t *up = (uint8_t *)X.LocateXStateFeature(c, ST_XS_AVX, NULL);
        if (!up) return 0;
        S->mut_reg = 5;
        memcpy(S->mut_old, up + 16 * 5, 16);
        up[16 * 5 + 2] ^= 0x01;
        memcpy(S->mut_new, up + 16 * 5, 16);
    } else return 0;
    S->mut_gen = r->gen;
    return 1;
}

static void st_ctl_capture(st_run *S, st_worker *w) {
    if (InterlockedCompareExchange(&w->susp_lock, 1, 0) != 0) return;
    st_rep r;
    memset(&r, 0, sizeof r);
    double t0 = S->c.perf ? t_now_ms() : 0;
    DWORD prev = SuspendThread(w->h);
    if (prev == (DWORD)-1) {
        r.susp_err = GetLastError();
        InterlockedExchange(&w->susp_lock, 0);
        st_ctl_report(S, w, &r);
        return;
    }
    r.prev = prev;
    S->susp_ops++;
    if (S->c.ctl & (C_GET | C_GETX | C_XOPT | C_SET)) {
        CONTEXT *c = S->ctx;
        c->ContextFlags = ST_CF_ALL | (S->xs ? CONTEXT_XSTATE : 0);
        if (S->xs) X.SetXStateFeaturesMask(c, T_XS_AVX);
        if (!GetThreadContext(w->h, c)) r.get_err = GetLastError() ? GetLastError() : 0xFFFFFFFFu;
        else {
            r.got_ctx = 1;
            S->caps++;
            st_analyze(S, w, c, &r);
            if (r.region == R_HOLD) { S->hold_hits++; if (r.apc_slot) S->hold_hits_apc++; }
            else if (r.region == R_ASM) S->asm_other++;
            else S->outside++;
            int mut = S->c.mutate ? st_mutate(S, w, c, &r) : 0;
            if (((S->c.ctl & C_SET) && !r.wine_zero) || mut) {
                c->ContextFlags = (r.region == R_HOLD ? ST_CF_ALL : ST_CF_FP) | (S->xs ? CONTEXT_XSTATE : 0);
                if (!SetThreadContext(w->h, c)) r.set_err = GetLastError() ? GetLastError() : 0xFFFFFFFFu;
                else { S->sets++; r.did_set = 1; if (mut) { r.mutated = 1; InterlockedExchange(&S->mut_done, 1); } }
            }
        }
    }
    r.resume = ResumeThread(w->h);
    InterlockedExchange(&w->susp_lock, 0);
    if (S->c.perf) S->cap_ms += t_now_ms() - t0;
    st_ctl_report(S, w, &r);
}

static DWORD WINAPI st_ctl_main(void *p) {
    st_run *S = (st_run *)p;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    t_rng r;
    t_rng_seed(&r, S->seed ^ 0xC0C0C0C0ull);
    unsigned it = 0;
    while (!S->ctl_stop) {
        uint64_t x = t_rng_next(&r);
        st_worker *w = &S->w[x % (unsigned)S->n];
        if (!w->started || w->finished || (S->c.mutate && S->mut_done)) { Sleep(0); continue; }
        unsigned acts = S->c.ctl & (C_AFF | C_PRIO | C_APC);
        if (acts && ((x >> 32) & 3) == 0) {
            int a = st_pick_bit(acts, (unsigned)(x >> 36));
            if ((1u << a) == C_AFF) {
                uintptr_t m = ((x >> 40) & 3) == 0 ? S->procmask : st_nth_cpu(S->procmask, (unsigned)((x >> 44) % (unsigned)S->ncpu));
                SetThreadAffinityMask(w->h, (DWORD_PTR)m);
                S->aff_ops++;
            } else if ((1u << a) == C_PRIO) {
                SetThreadPriority(w->h, -(int)((x >> 40) % 3));
                S->prio_ops++;
            } else {
                st_queue_apc(w);
                S->apc_ops++;
            }
        }
        if (S->c.ctl & C_CAPTURE) st_ctl_capture(S, w);
        if ((++it & 3) == 0 || S->ncpu <= 2) SwitchToThread();
    }
    return 0;
}

/* ======================================================================================================
 * Timers: CreateTimerQueueTimer / timeSetEvent callbacks
 * ====================================================================================================== */
typedef UINT (WINAPI *st_tse_fn)(UINT, UINT, void (CALLBACK *)(UINT, UINT, DWORD_PTR, DWORD_PTR, DWORD_PTR), DWORD_PTR, UINT);
typedef UINT (WINAPI *st_tke_fn)(UINT);
typedef UINT (WINAPI *st_tper_fn)(UINT);
#define ST_TIME_PERIODIC 0x0001
#define ST_TIME_CALLBACK_FUNCTION 0x0000
#define ST_TIME_KILL_SYNCHRONOUS 0x0100

/* Callback-thread register check: hold a callback-slot generation and verify it. */
static void st_cb_regs(st_run *S, LONG t, uint64_t x) {
    int k = (int)(t % ST_NCB);
    if (InterlockedCompareExchange(&S->cb_lock[k], 1, 0) != 0) return;
    st_worker *w = &S->cbw[k];
    w->tid = GetCurrentThreadId();
    w->mx_mode = t_get_mxcsr(); w->fcw_mode = t_get_x87_cw();
    st_slot *s = &w->main;
    int op = (x & 3) == 0 ? OP_X87 : (x & 3) == 1 ? OP_MMX : (x & 3) == 2 ? OP_COMPUTE : OP_HOLD;
    st_publish(s, s->gen + 1, op, st_pick_vk(S, x >> 8), w->mx_mode, w->fcw_mode, 200);
    st_run_slot(s);
    st_verify(w, s);
    st_mode_check(w, "after timer-callback hold");
    w->iters++;
    InterlockedIncrement(&S->tmr_regs);
    InterlockedExchange(&S->cb_lock[k], 0);
}

/* Thread-pool work item (QueueUserWorkItem) queued by a timer tick: same register hold on a pool thread.
 * Counted in tmr_inflight from queueing to return, so the driver's drain loop also waits for these. */
static DWORD WINAPI st_pool_cb(void *p) {
    st_run *S = &g_S;
    LONG t = (LONG)(intptr_t)p;
    if (S->tmr_on) {
        st_cb_regs(S, t, t_mix64(S->seed ^ 0x9001ull ^ (uint64_t)t));
        InterlockedIncrement(&S->tmr_pool);
    }
    InterlockedDecrement(&S->tmr_inflight);
    return 0;
}

static void st_timer_tick(st_run *S) {
    InterlockedIncrement(&S->tmr_inflight);
    if (S->tmr_on) {
        LONG t = InterlockedIncrement(&S->tmr_ticks);
        uint64_t x = t_mix64(S->seed ^ (uint64_t)t);
        if (S->c.tmr & TA_REGS) st_cb_regs(S, t, x >> 3);
        if ((S->c.tmr & TA_POOL) && S->tmr_inflight < 64) {
            InterlockedIncrement(&S->tmr_inflight);
            if (!QueueUserWorkItem(st_pool_cb, (void *)(intptr_t)t, WT_EXECUTEDEFAULT)) {
                InterlockedDecrement(&S->tmr_inflight);
                InterlockedIncrement(&S->tmr_pool_fail);
            }
        }
        st_worker *w = &S->w[(x >> 16) % (unsigned)S->n];
        unsigned acts = S->c.tmr & (TA_APC | TA_WAKE | TA_KICK);
        if (acts && w->started && !w->finished) {
            int a = st_pick_bit(acts, (unsigned)(x >> 40));
            if ((1u << a) == TA_APC) { st_queue_apc(w); InterlockedIncrement(&S->tmr_apc); }
            else if ((1u << a) == TA_WAKE) { SetEvent(w->ev); InterlockedIncrement(&S->tmr_wake); }
            else if (InterlockedCompareExchange(&w->susp_lock, 1, 0) == 0) {
                /* kick: suspend, force the state save with a CONTEXT_CONTROL capture, resume */
                CONTEXT c __attribute__((aligned(16)));
                DWORD p0 = SuspendThread(w->h), r1 = (DWORD)-1;
                int got = 1;
                if (p0 != (DWORD)-1) {
                    memset(&c, 0, sizeof c);
                    c.ContextFlags = CONTEXT_CONTROL;
                    got = GetThreadContext(w->h, &c);
                    r1 = ResumeThread(w->h);
                }
                InterlockedExchange(&w->susp_lock, 0);
                InterlockedIncrement(&S->tmr_kick);
                if (p0 != 0 || r1 != 1 || !got) {
                    InterlockedIncrement(&S->tmr_bad);
                    CHECKF(0, "timer kick of worker #%d: SuspendThread prev %lu, GetThreadContext %d, ResumeThread prev %lu",
                           w->idx, (unsigned long)p0, got, (unsigned long)r1);
                }
            }
        }
    }
    InterlockedDecrement(&S->tmr_inflight);
}
static VOID CALLBACK st_tq_cb(PVOID p, BOOLEAN fired) { (void)fired; st_timer_tick((st_run *)p); }
static void CALLBACK st_mm_cb(UINT id, UINT msg, DWORD_PTR user, DWORD_PTR r1, DWORD_PTR r2) {
    (void)id; (void)msg; (void)r1; (void)r2;
    st_timer_tick((st_run *)user);
}

/* ======================================================================================================
 * XSTATE capture availability (cached): 1 works, 0 not applicable (reason), -1 broken (reason)
 * ====================================================================================================== */
static int g_xprobe = 2;
static char g_xwhy[192];
static DWORD WINAPI st_wait_thread(void *p) { WaitForSingleObject((HANDLE)p, 10000); return 0; }
static int st_xstate_probe(void) {
    if (g_xprobe != 2) return g_xprobe;
    if (!cpu_has(F_AVX)) { snprintf(g_xwhy, sizeof g_xwhy, "AVX not usable (CPUID/XCR0)"); return g_xprobe = 0; }
    if (!st_xapi()) { snprintf(g_xwhy, sizeof g_xwhy, "XSTATE context API not exported"); return g_xprobe = 0; }
    if (!(X.GetEnabledXStateFeatures() & T_XS_AVX)) { snprintf(g_xwhy, sizeof g_xwhy, "GetEnabledXStateFeatures lacks AVX"); return g_xprobe = 0; }
    st_xc x;
    if (!st_xc_init(&x, ST_CF_ALL | CONTEXT_XSTATE)) {
        snprintf(g_xwhy, sizeof g_xwhy, "InitializeContext(CONTEXT_XSTATE) failed, GLE=%lu", (unsigned long)x.err);
        return g_xprobe = -1;
    }
    HANDLE ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    HANDLE h = ev ? CreateThread(NULL, 0, st_wait_thread, ev, 0, NULL) : NULL;
    if (!h) { if (ev) CloseHandle(ev); st_xc_free(&x); snprintf(g_xwhy, sizeof g_xwhy, "CreateThread failed"); return g_xprobe = 0; }
    Sleep(20);
    int res = 1;
    if (SuspendThread(h) == (DWORD)-1) { res = -1; snprintf(g_xwhy, sizeof g_xwhy, "SuspendThread failed, GLE=%lu", (unsigned long)GetLastError()); }
    else {
        if (!X.SetXStateFeaturesMask(x.ctx, T_XS_AVX)) { res = -1; snprintf(g_xwhy, sizeof g_xwhy, "SetXStateFeaturesMask(AVX) failed, GLE=%lu", (unsigned long)GetLastError()); }
        else if (!GetThreadContext(h, x.ctx)) { res = -1; snprintf(g_xwhy, sizeof g_xwhy, "GetThreadContext(CONTEXT_XSTATE) failed, GLE=%lu", (unsigned long)GetLastError()); }
        else if (!X.LocateXStateFeature(x.ctx, ST_XS_AVX, NULL)) { res = -1; snprintf(g_xwhy, sizeof g_xwhy, "LocateXStateFeature(AVX) NULL after GetThreadContext"); }
        ResumeThread(h);
    }
    SetEvent(ev);
    WaitForSingleObject(h, 5000);
    CloseHandle(h); CloseHandle(ev);
    st_xc_free(&x);
    if (res == 1) snprintf(g_xwhy, sizeof g_xwhy, "ok");
    return g_xprobe = res;
}

/* ======================================================================================================
 * New-thread churn: initial registers of fresh threads never carry any suite pattern
 * ====================================================================================================== */
typedef struct st_churn { uint8_t regs[16][32] __attribute__((aligned(64))); uint32_t mx; uint16_t cw; int ymm; } st_churn;
static DWORD WINAPI st_churn_main(void *p) {
    st_churn *c = (st_churn *)p;
    if (c->ymm) st_dump_ymm(c->regs); else st_dump_xmm(c->regs);
    c->mx = t_get_mxcsr();
    c->cw = t_get_x87_cw();
    return 0;
}
static int st_churn_one(st_churn *c, int ymm) {
    memset(c, 0, sizeof *c);
    c->ymm = ymm;
    HANDLE h = CreateThread(NULL, 0, st_churn_main, c, 0, NULL);
    if (!h) return 0;
    DWORD r = WaitForSingleObject(h, 3000);
    CloseHandle(h);
    return r == WAIT_OBJECT_0;
}

/* ======================================================================================================
 * Run driver
 * ====================================================================================================== */
static uint64_t st_hash(const char *s) { uint64_t h = 0xcbf29ce484222325ull; while (*s) { h ^= (uint8_t)*s++; h *= 0x100000001b3ull; } return h; }

static void st_init_worker(st_run *S, st_worker *w, int idx, const char *role, unsigned mk) {
    memset(w, 0, sizeof *w);
    w->S = S; w->idx = idx; w->role = role;
    w->seed = t_mix64(S->seed ^ ((uint64_t)(idx + 1) * 0x9E3779B97F4A7C15ull) ^ mk);
    t_rng_seed(&w->rng, w->seed ^ 0x5151);
    static const uint16_t pcs[3] = { 0, 2, 3 };
    w->mx_mode = 0x1F80u | (((unsigned)idx & 3u) << 13) | ((idx & 4) ? 0x8000u : 0) | (((idx & 8) && st_daz()) ? 0x40u : 0);
    w->fcw_mode = (uint16_t)(0x7F | (pcs[((unsigned)idx >> 2) % 3] << 8) | ((((unsigned)idx * 5u + 1u) & 3u) << 10));
    w->main.seed = w->seed; w->main.mk = mk; w->main.idx = (unsigned)idx;
    w->apc.seed = w->seed ^ 0xA9C0A9C0A9C0ull; w->apc.mk = MK_APC; w->apc.idx = (unsigned)idx;
}

static void st_run_cfg(const st_cfg *cfg, const char *name) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    LONG fails0 = g_st_fails;
    st_run *S = &g_S;
    st_cfg c = *cfg;
    if (!c.threads) c.threads = 16;
    if (!c.ms) c.ms = 120;
    if (!c.ops) c.ops = OPB(OP_HOLD) | OPB(OP_COMPUTE);
    if (!c.yields) c.yields = YB(Y_SLEEP0);
    if (!c.kind) c.kind = K_XMM;
    if (!c.spin) c.spin = 2000;
    /* Emulator artifact (README "Emulated validation"), never on real hardware: under Intel SDE a controller
     * that keeps suspending/capturing threads which run emulated code deadlocks Pin at random (seen in
     * ctl_suspend_resume_only and ctl_apc_injection: the whole process hangs until the run timeout; the
     * x86 runs show "controller did not stop"). Not run there; tools/wine_run.sh covers these cases. */
    if (t_sde_model() && c.ctl)
        XFAIL("Intel SDE (%s): a thread-suspending controller can deadlock Pin (process hangs); not run", t_sde_model());
    st_daz();
    int avx = cpu_has(F_AVX);
    if (c.kind == K_YMM && !avx) SKIP("AVX not usable (CPUID/XCR0): YMM patterns need it");
    if (c.mutate == 3 && !avx) SKIP("AVX not usable");
    int xs = 0;
    if (c.ctl & (C_GETX | C_XOPT) || c.mutate == 3) {
        int p = st_xstate_probe();
        if (p == 1) xs = 1;
        else if ((c.ctl & C_GETX) || c.mutate == 3) {
            if (p == 0) SKIP("XSTATE context capture not applicable: %s", g_xwhy);
            if (t_is_wine()) XFAIL("Wine: XSTATE thread-context capture unusable (%s)", g_xwhy);
            SKIP("XSTATE thread-context capture broken (%s); reported as FAIL by xstate_capture_probe", g_xwhy);
        }
    }
    st_tse_fn tse = NULL; st_tke_fn tke = NULL; st_tper_fn tbp = NULL, tep = NULL;
    if (c.tmr & T_MM) {
        tse = (st_tse_fn)t_proc("winmm.dll", "timeSetEvent");
        tke = (st_tke_fn)t_proc("winmm.dll", "timeKillEvent");
        tbp = (st_tper_fn)t_proc("winmm.dll", "timeBeginPeriod");
        tep = (st_tper_fn)t_proc("winmm.dll", "timeEndPeriod");
        if (!tse || !tke) SKIP("winmm timeSetEvent/timeKillEvent not available");
    }
    const t_env *env = t_env_get();
    int ncpu = t_cpu_count();
    if (ncpu < 1) ncpu = 1;
    int n = c.threads == T_2XCPU ? 2 * ncpu : c.threads;
    if (n > ST_MAXW) n = ST_MAXW;
    if (n < 1) n = 1;

    memset(S, 0, sizeof *S);
    S->c = c; S->name = name; S->seed = st_hash(name); S->n = n; S->avx = avx; S->xs = xs;
    S->min_iters = c.mutate ? 1 : 2;
    S->wine = t_is_wine();
    S->procmask = (uintptr_t)env->affinity;
    if (!S->procmask) { DWORD_PTR pm = 0, sm = 0; GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm); S->procmask = pm ? pm : 1; }
    { int k = 0; for (uintptr_t m = S->procmask; m; m &= m - 1) k++; S->ncpu = k; }
    S->w = (st_worker *)st_valloc(sizeof(st_worker) * (size_t)(n + ST_NCB));
    S->scratch = (st_blk *)st_valloc(sizeof(st_blk));
    REQUIREF(S->w && S->scratch, "allocation of %d workers failed", n);
    for (int i = 0; i < n; i++) st_init_worker(S, &S->w[i], i, "worker", MK_MAIN);
    if (c.pin && S->ncpu >= 1) {
        for (int i = 0; i < n; i++) {
            unsigned cpu = c.pin == PIN_ONE ? (unsigned)(S->ncpu - 1) : (unsigned)(i / 2) % (unsigned)S->ncpu;
            S->w[i].pin_mask = st_nth_cpu(S->procmask, cpu);
        }
    }
    if (c.hold_ms > 0) {
        /* PAUSE latency differs by an order of magnitude between cores: calibrate the hold loop on this thread
         * (fastest of 6 runs after a warm-up, so a preempted or clock-ramping run does not shrink the count),
         * then size spin for hold_ms. */
        static st_blk cal;
        double best = 1e30;
        const uint32_t probe = 20000;
        /* hybrid parts: PAUSE cost differs between core types, so calibrate on the pinned CPU when there is one */
        DWORD_PTR oldaff = c.pin == PIN_ONE ? SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)S->w[0].pin_mask) : 0;
        for (int k = 0; k < 7; k++) {
            st_fill(&cal, 0xCA11, (uint32_t)k, 0, MK_MAIN, OP_HOLD);
            cal.mx_mode = t_get_mxcsr(); cal.fcw_mode = t_get_x87_cw(); cal.spin = probe;
            double t0c = t_now_ms();
            st_hold_xmm(&cal);
            double d = t_now_ms() - t0c;
            if (k && d < best) best = d;
        }
        if (oldaff) SetThreadAffinityMask(GetCurrentThread(), oldaff);
        if (best < 0.001) best = 0.001;
        double sp = (double)probe * (double)c.hold_ms / best / 1.5;   /* published spin is [spin, 2*spin) */
        if (sp < 1000) sp = 1000;
        if (sp > 40000000.0) sp = 40000000.0;
        c.spin = (unsigned)sp;
        S->c.spin = c.spin;
        LARGE_INTEGER f; QueryPerformanceFrequency(&f);
        /* a hold takes at most ~2/1.5 * hold_ms when it keeps its CPU; beyond 2 * hold_ms it was descheduled */
        S->stretch_qpc = (LONG64)((double)f.QuadPart * (double)c.hold_ms * 2.0 / 1000.0);
        t_info("hold_calibration", "%u PAUSE iterations in %.3f ms -> spin %u for ~%d ms holds", probe, best, c.spin, c.hold_ms);
    }
    S->cbw = &S->w[n];
    for (int k = 0; k < ST_NCB; k++) st_init_worker(S, &S->cbw[k], k, "timer-callback", MK_CB);
    for (int i = 0; i < n; i++) {
        S->w[i].ev = CreateEventA(NULL, FALSE, FALSE, NULL);
        S->w[i].wt = CreateWaitableTimerA(NULL, FALSE, NULL);
        S->w[i].peer = n > 1 ? &S->w[(i ^ 1) < n ? (i ^ 1) : 0] : NULL;
    }
    S->ev_go = CreateEventA(NULL, TRUE, FALSE, NULL);
    S->ev_set = CreateEventA(NULL, TRUE, TRUE, NULL);
    S->sema = CreateSemaphoreA(NULL, n / 4 + 1, n / 4 + 1, NULL);
    InitializeCriticalSection(&S->cs);
    InitializeSRWLock(&S->srw);
    if (xs) {
        if (!st_xc_init(&S->xc, ST_CF_ALL | CONTEXT_XSTATE)) {
            CHECKF(0, "InitializeContext(CONTEXT_XSTATE) failed although the probe succeeded, GLE=%lu", (unsigned long)S->xc.err);
            S->xs = 0;
        } else S->ctx = S->xc.ctx;
    }
    if (!S->ctx) S->ctx = (CONTEXT *)st_valloc(sizeof(CONTEXT) + 64);
    t_info("threads", "%d", n);
    t_info("config", "ops=0x%x yields=0x%x ctl=0x%x tmr=0x%x kind=%d spin=%u ms=%d xstate=%d", c.ops, c.yields, c.ctl,
           c.tmr, c.kind, c.spin, c.ms, S->xs);

    /* baseline new-thread state (before any worker runs) */
    static st_churn base, ch;
    int churn_ok = 0;
    if (c.churn) churn_ok = st_churn_one(&base, avx);

    int created = 0;
    DWORD cerr = 0;
    for (int i = 0; i < n; i++) {
        st_worker *w = &S->w[i];
        w->h = CreateThread(NULL, 256 * 1024, st_worker_main, w, STACK_SIZE_PARAM_IS_A_RESERVATION, &w->tid);
        if (!w->h) { cerr = GetLastError(); break; }
        created++;
    }
    HANDLE hctl = NULL;
    int skip_reason = 0;
    if (created < n) skip_reason = 1;
    else if (c.ctl) {
        hctl = CreateThread(NULL, 0, st_ctl_main, S, CREATE_SUSPENDED, NULL);
        if (!hctl) { cerr = GetLastError(); skip_reason = 2; }
    }
    if (skip_reason) {
        S->abort = 1;
        SetEvent(S->ev_go);
        for (int i = 0; i < created; i++) { WaitForSingleObject(S->w[i].h, 5000); CloseHandle(S->w[i].h); }
        for (int i = 0; i < n; i++) { CloseHandle(S->w[i].ev); if (S->w[i].wt) CloseHandle(S->w[i].wt); }
        CloseHandle(S->ev_go); CloseHandle(S->ev_set); CloseHandle(S->sema); DeleteCriticalSection(&S->cs);
        if (S->xc.buf) st_xc_free(&S->xc); else st_vfree(S->ctx);
        st_vfree(S->w); st_vfree(S->scratch);
        if (skip_reason == 1) SKIP("CreateThread failed for worker %d of %d (GLE=%lu)", created, n, (unsigned long)cerr);
        SKIP("CreateThread failed for the controller (GLE=%lu)", (unsigned long)cerr);
    }
    /* wait until every worker set its mode */
    double ts = t_now_ms();
    for (int i = 0; i < n; i++) while (!S->w[i].started && t_now_ms() - ts < 5000) Sleep(1);

    int oldprio = GetThreadPriority(GetCurrentThread());
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    HANDLE htq = NULL; UINT mmid = 0;
    S->tmr_on = 1;
    if (c.tmr & T_TQ) CHECKF(CreateTimerQueueTimer(&htq, NULL, st_tq_cb, S, 1, 1, WT_EXECUTEDEFAULT), "CreateTimerQueueTimer GLE=%lu", (unsigned long)GetLastError());
    if (c.tmr & T_MM) {
        if (tbp) tbp(1);
        mmid = tse(1, 0, st_mm_cb, (DWORD_PTR)S, ST_TIME_PERIODIC | ST_TIME_CALLBACK_FUNCTION | ST_TIME_KILL_SYNCHRONOUS);
        CHECKF(mmid != 0, "timeSetEvent(1 ms, periodic) failed");
    }
    double t0 = t_now_ms();
    SetEvent(S->ev_go);
    if (hctl) ResumeThread(hctl);

    int churned = 0, churn_fail = 0;
    while (t_now_ms() - t0 < c.ms) {
        if (c.churn && churned < 400) {
            if (st_churn_one(&ch, avx)) {
                churned++;
                int foreign = 0; char ex[96] = "";
                for (int i = 0; i < ST_NV && !foreign; i++)
                    for (int q = 0; q < (avx ? 4 : 2); q++) {
                        uint64_t v; memcpy(&v, ch.regs[i] + 8 * q, 8);
                        unsigned a, b, cc, d;
                        if (st_decode(v, &a, &b, &cc, &d)) {
                            char e2[96];
                            foreign = 1;
                            snprintf(ex, sizeof ex, "%s%d qword %d = %s", avx ? "ymm" : "xmm", i, q, st_explain(v, e2, sizeof e2));
                            break;
                        }
                    }
                CHECKF(!foreign, "new thread #%d starts with a suite pattern in its registers: %s", churned, ex);
                if (churn_ok) CHECKF(ch.mx == base.mx && ch.cw == base.cw,
                                     "new thread #%d initial MXCSR 0x%04x FCW 0x%04x differ from the first new thread's 0x%04x / 0x%04x",
                                     churned, ch.mx, ch.cw, base.mx, base.cw);
            } else churn_fail++;
        } else Sleep(5);
    }
    double run_ms = t_now_ms() - t0;

    /* stop order: controller, timers, restore scheduling, workers */
    S->ctl_stop = 1;
    if (hctl) { DWORD wr = WaitForSingleObject(hctl, 10000); CHECKF(wr == WAIT_OBJECT_0, "controller did not stop"); CloseHandle(hctl); }
    S->tmr_on = 0;
    if (htq) CHECKF(DeleteTimerQueueTimer(NULL, htq, INVALID_HANDLE_VALUE), "DeleteTimerQueueTimer GLE=%lu", (unsigned long)GetLastError());
    if (mmid) { tke(mmid); if (tep) tep(1); }
    { double tw = t_now_ms(); while (S->tmr_inflight && t_now_ms() - tw < 3000) Sleep(1); Sleep(c.tmr ? 20 : 0); }
    for (int i = 0; i < n; i++) {
        SetThreadPriority(S->w[i].h, THREAD_PRIORITY_NORMAL);
        SetThreadAffinityMask(S->w[i].h, (DWORD_PTR)S->procmask);
    }
    S->stop = 1;
    for (int i = 0; i < n; i++) SetEvent(S->w[i].ev);
    double tj = t_now_ms();
    int unfinished = 0;
    for (int i = 0; i < n; i++) {
        double left = 10000 - (t_now_ms() - tj);
        if (WaitForSingleObject(S->w[i].h, left > 0 ? (DWORD)left : 0) != WAIT_OBJECT_0) unfinished++;
    }
    if (unfinished) {
        S->hard_stop = 1;
        for (int i = 0; i < n; i++) SetEvent(S->w[i].ev);
        unfinished = 0;
        for (int i = 0; i < n; i++) if (WaitForSingleObject(S->w[i].h, 3000) != WAIT_OBJECT_0) unfinished++;
    }
    SetThreadPriority(GetCurrentThread(), oldprio);

    /* results */
    LONG64 iters = 0, apcs = 0; LONG bad = 0, modebad = 0;
    uint64_t ops[OP_N] = { 0 }, ys[Y_N] = { 0 };
    for (int i = 0; i < n; i++) {
        st_worker *w = &S->w[i];
        iters += w->iters; apcs += w->apc_iters; bad += w->bad; modebad += w->mode_bad;
        for (int k = 0; k < OP_N; k++) ops[k] += w->ops_done[k];
        for (int k = 0; k < Y_N; k++) ys[k] += w->yields_done[k];
        CHECKF(w->finished, "worker #%d tid %lu did not finish within the stop deadline", i, (unsigned long)w->tid);
        CHECKF(w->iters >= S->min_iters, "worker #%d tid %lu completed only %lld iterations (starved?)", i, (unsigned long)w->tid, (long long)w->iters);
        if (!c.mutate)
            CHECKF(w->bad == 0, "worker #%d tid %lu (MXCSR mode 0x%04x FCW mode 0x%04x): %ld state mismatch(es) in %lld iterations, %ld APC holds",
                   i, (unsigned long)w->tid, w->mx_mode, w->fcw_mode, (long)w->bad, (long long)w->iters, (long)w->apc_iters);
    }
    for (int k = 0; k < ST_NCB; k++)
        CHECKF(S->cbw[k].bad == 0, "timer-callback slot %d: %ld state mismatch(es) in %lld holds", k, (long)S->cbw[k].bad, (long long)S->cbw[k].iters);
    t_info("iterations", "%lld", (long long)iters);
    t_info("apc_holds", "%lld", (long long)apcs);
    t_info("ops", "hold=%llu x87=%llu mmx=%llu compute=%llu call=%llu", (unsigned long long)ops[0], (unsigned long long)ops[1],
           (unsigned long long)ops[2], (unsigned long long)ops[3], (unsigned long long)ops[4]);
    {
        char yb[512]; size_t o = 0; yb[0] = 0;
        for (int k = 0; k < Y_N; k++) if (ys[k] && o < sizeof yb - 40) o += (size_t)snprintf(yb + o, sizeof yb - o, "%s%s=%llu", o ? " " : "", st_y_name[k], (unsigned long long)ys[k]);
        t_info("yields", "%s", yb);
    }
    if (S->stretch_qpc) {
        LONG tm = 0, st = 0;
        for (int i = 0; i < n; i++) { tm += S->w[i].holds_timed; st += S->w[i].holds_stretched; }
        /* how often a hold with live patterns was descheduled and resumed: variable, never a pass criterion */
        t_info("holds_descheduled", "%ld of %ld timed holds took > 2x hold_ms (pinned=%d)", (long)st, (long)tm, c.pin);
    }
    if (c.tmr & TA_POOL)
        t_info("pool_callbacks", "%ld register holds on thread-pool threads (%ld QueueUserWorkItem failures)",
               (long)S->tmr_pool, (long)S->tmr_pool_fail);
    if (c.ctl) {
        t_info("ctl", "suspends=%lld captures=%lld hold_hits=%lld (apc %lld) asm_other=%lld outside=%lld checked_outside=%lld apc_skip=%lld sets=%lld aff=%lld prio=%lld apc=%lld",
               (long long)S->susp_ops, (long long)S->caps, (long long)S->hold_hits, (long long)S->hold_hits_apc,
               (long long)S->asm_other, (long long)S->outside, (long long)S->out_checked, (long long)S->apc_skip,
               (long long)S->sets, (long long)S->aff_ops, (long long)S->prio_ops, (long long)S->apc_ops);
        if (S->upper_sys) t_note("%lld YMM upper halves held non-pattern data during XMM-only holds (system code; not a suite pattern)", (long long)S->upper_sys);
        if (S->x87_nonempty) t_info("x87_nonempty_outside_asm", "%lld", (long long)S->x87_nonempty);
        if (c.ctl & C_CAPTURE) CHECKF(S->susp_ops > 0, "controller performed no SuspendThread");
        if (c.ctl & (C_GET | C_GETX | C_XOPT | C_SET)) CHECKF(S->caps > 0, "controller captured no context");
        CHECKF(S->ctl_bad == 0 || c.mutate, "controller recorded %lld problem(s)", (long long)S->ctl_bad);
        if (c.need_hold) CHECKF(S->hold_hits > 0, "oracle never caught the worker inside a hold loop (%lld captures)", (long long)S->caps);
    }
    if (c.tmr) {
        t_info("timers", "ticks=%ld apc=%ld wake=%ld kick=%ld callback_holds=%ld", (long)S->tmr_ticks, (long)S->tmr_apc,
               (long)S->tmr_wake, (long)S->tmr_kick, (long)S->tmr_regs);
        CHECKF(S->tmr_ticks > 0, "timer callbacks never ran in %.0f ms", run_ms);
        if (c.tmr & TA_REGS) CHECKF(S->tmr_regs > 0, "no timer callback performed its register hold");
        if (c.tmr & TA_POOL) CHECKF(S->tmr_pool > 0, "no thread-pool work item ran in %.0f ms (%ld QueueUserWorkItem failures)", run_ms, (long)S->tmr_pool_fail);
        CHECKF(S->tmr_bad == 0, "%ld timer kick problem(s)", (long)S->tmr_bad);
    }
    if (c.churn) {
        t_info("churn_threads", "%d (failed %d)", churned, churn_fail);
        if (churn_ok) t_info("new_thread_fp_env", "MXCSR 0x%04x FCW 0x%04x", base.mx, base.cw);
        CHECKF(churned > 0, "no churn thread could be created/joined (%d failures)", churn_fail);
    }
    if (c.mutate) {
        st_worker *w = &S->w[0];
        CHECKF(S->mut_done, "controller never applied the mutation (no capture inside a hold loop, %lld captures)", (long long)S->caps);
        if (S->mut_done) {
            CHECKF(w->nmut == 1, "worker recorded %d mismatches, expected exactly the one injected", w->nmut);
            if (w->nmut >= 1) {
                CHECK_EQ(w->mut[0].gen, S->mut_gen);
                if (c.mutate == 2) {
                    CHECK_EQ(w->mut[0].what, W_MX);
                    uint32_t g, wv; memcpy(&g, w->mut[0].got, 4); memcpy(&wv, w->mut[0].want, 4);
                    CHECK_EQ(g, S->mut_mx_new);
                    CHECK_EQ(wv, S->mut_mx_old);
                } else {
                    CHECK_EQ(w->mut[0].what, W_VEC);
                    CHECK_EQ(w->mut[0].reg, S->mut_reg);
                    int off = c.mutate == 3 ? 16 : 0;
                    CHECK_MEM(w->mut[0].got + off, S->mut_new, 16);
                    CHECK_MEM(w->mut[0].want + off, S->mut_old, 16);
                    if (c.mutate == 3) CHECK_MEM(w->mut[0].got, w->mut[0].want, 16);   /* low lane untouched */
                }
            }
        }
    }
    if (c.perf && run_ms > 0) {
        t_perf("worker_iterations_per_s", (double)iters / (run_ms / 1000.0), "iter/s");
        if (S->caps) t_perf(S->xs ? "xstate_capture_cycles_per_s" : "context_capture_cycles_per_s", (double)S->caps / (S->cap_ms / 1000.0), "ops/s");
    }

    /* cleanup (leak deliberately when a thread is still running) */
    for (int i = 0; i < n; i++) CloseHandle(S->w[i].h);
    for (int i = 0; i < n; i++) { CloseHandle(S->w[i].ev); if (S->w[i].wt) CloseHandle(S->w[i].wt); }
    CloseHandle(S->ev_go); CloseHandle(S->ev_set); CloseHandle(S->sema);
    LONG64 wz = S->wine_zero_fp, wh = S->wine_host_ip;
    if (unfinished) { t_note("%d worker(s) still running: run state leaked on purpose", unfinished); return; }
    DeleteCriticalSection(&S->cs);
    if (S->xc.buf) st_xc_free(&S->xc); else st_vfree(S->ctx);
    st_vfree(S->w); st_vfree(S->scratch);
    S->w = NULL;
    if ((wz || wh) && g_st_fails != fails0) {
        t_note("Wine gap also seen (%lld zero-FP + %lld host-IP captures not checked); the failures above stand",
               (long long)wz, (long long)wh);
        return;
    }
    if (wz) XFAIL("Wine: GetThreadContext of a thread inside a system call returned an all-zero legacy FP area (FCW=0, "
                  "MXCSR=0) in %lld captures (+%lld with a host-code IP); those were not checked, hold-loop oracle unaffected",
                  (long long)wz, (long long)wh);
    if (wh) XFAIL("Wine: GetThreadContext of a thread running host-side syscall code returned a non-PE IP with that code's "
                  "FP environment in %lld captures; those were not checked, hold-loop oracle unaffected", (long long)wh);
}

#define ST(name, ...) TEST(name) { static const st_cfg c_ = { __VA_ARGS__ }; st_run_cfg(&c_, #name); }

/* ======================================================================================================
 * 0. Self-tests of the pattern generator, decoder and probe references
 * ====================================================================================================== */
TEST(selftest_pattern_words) {
    /* Every word decodes back to (marker, index, register, word); words of different threads / generations
     * differ; zero and simple values never decode; generation fill is deterministic. */
    static const unsigned mks[5] = { MK_MAIN, MK_APC, MK_CB, MK_MMX, MK_X87 };
    int bad = 0;
    for (unsigned m = 0; m < 5; m++)
        for (unsigned idx = 0; idx < 1024; idx += 37)
            for (unsigned reg = 0; reg < 16; reg += 5)
                for (unsigned w = 0; w < 4; w++) {
                    uint64_t v = st_word(mks[m], idx, reg, w, t_mix64(idx * 131 + reg * 7 + w));
                    unsigned a, b, c, d;
                    if (!st_decode(v, &a, &b, &c, &d) || a != mks[m] || b != idx || c != reg || d != w) bad++;
                    if (!v) bad++;
                }
    CHECKF(bad == 0, "%d words failed to decode to their source", bad);
    unsigned a, b, c, d;
    CHECK(!st_decode(0, &a, &b, &c, &d));
    CHECK(!st_decode(0x3FB999999999999Aull, &a, &b, &c, &d));
    CHECK(!st_decode(~0ull, &a, &b, &c, &d));
    CHECK(!st_decode(0xA500000000000000ull, &a, &b, &c, &d));          /* marker without matching check byte */
    /* x87 values are normals: explicit integer bit set, exponent neither 0 nor 0x7FFF */
    static st_blk b1, b2;
    st_fill(&b1, 0x1234, 7, 3, MK_MAIN, OP_X87);
    st_fill(&b2, 0x1234, 7, 3, MK_MAIN, OP_X87);
    CHECK_MEM(&b1, &b2, sizeof b1);
    for (int i = 0; i < 8; i++) {
        uint64_t m; uint16_t e; memcpy(&m, b1.fin[i], 8); memcpy(&e, b1.fin[i] + 8, 2);
        CHECKF((m >> 63) && (e & 0x7FFF) != 0 && (e & 0x7FFF) != 0x7FFF, "ST value %d: mantissa 0x%llx exp 0x%x", i, (unsigned long long)m, e);
    }
    CHECK_EQ(b1.flags, 1);
    CHECK_EQ(b1.mx_gen & 0x1F80, 0x1F80);                               /* every exception masked */
    CHECK_EQ(b1.fcw_gen & 0x3F, 0x3F);
    CHECK_NE((b1.fcw_gen >> 8) & 3, 1);                                  /* PC=01 is reserved */
    st_fill(&b2, 0x1234, 8, 3, MK_MAIN, OP_HOLD);
    CHECK(memcmp(b1.vin, b2.vin, sizeof b1.vin) != 0);
    st_fill(&b2, 0x1235, 7, 4, MK_MAIN, OP_X87);
    CHECK(memcmp(b1.vin, b2.vin, sizeof b1.vin) != 0);
    /* MXCSR / FCW generations round-trip through the CPU (DAZ only when MXCSR_MASK allows it) */
    int rt = 0;
    for (uint32_t g = 0; g < 64; g++) {
        st_fill(&b2, 0x77, g, g, MK_MAIN, OP_HOLD);
        t_set_mxcsr(b2.mx_gen); t_set_x87_cw(b2.fcw_gen);
        uint32_t mx = t_get_mxcsr(); uint16_t cw = t_get_x87_cw();
        t_set_mxcsr(0x1F80); t_set_x87_cw(0x027F);
        if (mx != b2.mx_gen || cw != b2.fcw_gen) { rt++; CHECKF(0, "gen %u: MXCSR 0x%04x->0x%04x FCW 0x%04x->0x%04x", g, b2.mx_gen, mx, b2.fcw_gen, cw); }
    }
    CHECK_EQ(rt, 0);
    t_info("daz_supported", "%d", st_daz());
    char e[96];
    t_info("explain_example", "%s", st_explain(st_word(MK_APC, 17, 9, 2, 5), e, sizeof e));
}

TEST(selftest_compute_reference) {
    /* The asm compute chains on the case thread match the C reference for every round count. */
    static st_blk b;
    for (int vk = 16; vk <= 32; vk += 16) {
        if (vk == 32 && !cpu_has(F_AVX)) { t_note("AVX not usable: YMM chain not checked"); break; }
        for (uint32_t g = 1; g <= 12; g++) {
            st_fill(&b, 0xC0FFEE, g, 1, MK_MAIN, OP_COMPUTE);
            b.mx_mode = t_get_mxcsr(); b.fcw_mode = t_get_x87_cw(); b.spin = 1;
            if (vk == 32) st_compute_ymm(&b); else st_compute_xmm(&b);
            uint8_t ref[16][32];
            st_ref_compute(&b, vk, ref);
            for (int i = 0; i < ST_NV; i++) {
                char lab[64]; snprintf(lab, sizeof lab, "%s%d gen %u rounds %u", vk == 32 ? "ymm" : "xmm", i, g, b.rounds);
                CHECK_REG(lab, b.cout[i], ref[i], (size_t)vk);
            }
        }
    }
}

TEST(probe_rounding_reference) {
    /* DIVSD 1/10 and FIDIV 1/10 on the case thread for every RC (and x87 PC 24/53/64): IEEE results fixed by the
     * SDM, the reference the workers compare against. */
    static st_blk b;
    static const unsigned pcs[3] = { 0, 2, 3 };
    for (unsigned rc = 0; rc < 4; rc++)
        for (int p = 0; p < 3; p++) {
            st_fill(&b, 1, 1, 0, MK_MAIN, OP_HOLD);
            b.mx_gen = 0x1F80u | (rc << 13);
            b.fcw_gen = (uint16_t)(0x7F | (pcs[p] << 8) | (rc << 10));
            b.mx_mode = t_get_mxcsr(); b.fcw_mode = t_get_x87_cw(); b.spin = 1;
            st_hold_xmm(&b);
            for (int k = 0; k < 2; k++) {
                CHECKF(b.psse[k] == st_exp_sse(rc, k), "DIVSD %s1/10 RC=%u: 0x%016llx want 0x%016llx", k ? "-" : "", rc,
                       (unsigned long long)b.psse[k], (unsigned long long)st_exp_sse(rc, k));
                uint8_t ex[10]; st_exp_x87(pcs[p], rc, k, ex);
                char lab[64]; snprintf(lab, sizeof lab, "FIDIV %s1/10 PC=%u RC=%u", k ? "-" : "", pcs[p], rc);
                CHECK_REG(lab, b.px87[k], ex, 10);
            }
            CHECK_EQ(b.mx_out, b.mx_gen);
            CHECK_EQ(b.fcw_out, b.fcw_gen);
            CHECK_EQ(t_get_mxcsr() & MX_CTRL, b.mx_mode & MX_CTRL);      /* routine restored the caller's mode */
            CHECK_EQ(t_get_x87_cw(), b.fcw_mode);
        }
}

/* Synthetic CONTEXT of a worker sitting in the XMM hold loop with generation `s` loaded. */
static void st_fake_ctx(CONTEXT *c, st_slot *s) {
    memset(c, 0, sizeof *c);
    uint8_t *fx = st_fx(c);
#ifdef T_X64
    c->Rip = (uintptr_t)st_hold_xmm_begin; c->Rcx = (uintptr_t)&s->b; c->Rax = s->b.spin / 2;
    c->MxCsr = s->b.mx_gen;
#else
    c->Eip = (uintptr_t)st_hold_xmm_begin; c->Ecx = (uintptr_t)&s->b; c->Eax = s->b.spin / 2;
#endif
    memcpy(fx, &s->b.fcw_gen, 2);
    fx[4] = s->op == OP_X87 || s->op == OP_MMX ? 0xFF : 0;
    memcpy(fx + 24, &s->b.mx_gen, 4);
    for (int i = 0; i < 8; i++) memcpy(fx + 32 + 16 * i, s->b.fin[i], 10);
    for (int i = 0; i < ST_NV; i++) memcpy(fx + 160 + 16 * i, s->b.vin[i], 16);
}
static int st_has_err(const st_rep *r, int code, int reg) {
    for (int i = 0; i < r->nerr; i++) if (r->e[i].code == code && (reg < 0 || r->e[i].reg == reg)) return 1;
    return 0;
}

TEST(selftest_oracle_analyzer) {
    /* The capture oracle on synthetic contexts: an exact image passes, every single corruption is named. */
    static st_run S;
    memset(&S, 0, sizeof S);
    S.seed = 0x5EED; S.c.kind = K_XMM;
    S.scratch = (st_blk *)st_valloc(sizeof(st_blk));
    st_worker *w = (st_worker *)st_valloc(sizeof(st_worker));
    CONTEXT *c = (CONTEXT *)st_valloc(sizeof(CONTEXT) + 64);
    REQUIRE(S.scratch && w && c);
    st_init_worker(&S, w, 5, "worker", MK_MAIN);
    st_publish(&w->main, 42, OP_X87, 16, w->mx_mode, w->fcw_mode, 100);
    st_rep r;
#define AN() do { memset(&r, 0, sizeof r); st_analyze(&S, w, c, &r); } while (0)
    st_fake_ctx(c, &w->main); AN();
    CHECK_EQ(r.region, R_HOLD); CHECK_EQ(r.nall, 0); CHECK_EQ(r.gen, 42);
    st_fake_ctx(c, &w->main); st_fx(c)[160 + 16 * 2 + 9] ^= 0x40; AN();
    CHECK(st_has_err(&r, E_VLO, 2)); CHECK_EQ(r.nall, 1);
    st_fake_ctx(c, &w->main); st_fx(c)[24] ^= 0x01;
#ifdef T_X64
    c->MxCsr ^= 0x01;
#endif
    AN(); CHECK(st_has_err(&r, E_MX, -1));
    st_fake_ctx(c, &w->main); st_fx(c)[1] ^= 0x0C; AN(); CHECK(st_has_err(&r, E_FCW, -1));
    st_fake_ctx(c, &w->main); st_fx(c)[4] = 0; AN(); CHECK(st_has_err(&r, E_TAG, -1));
    st_fake_ctx(c, &w->main); st_fx(c)[3] |= 0x08; AN(); CHECK(st_has_err(&r, E_TOP, -1));
    st_fake_ctx(c, &w->main); st_fx(c)[32 + 16 * 4 + 8] ^= 0x01; AN(); CHECK(st_has_err(&r, E_ST, 4));
#ifdef T_X64
    st_fake_ctx(c, &w->main); c->Rcx += 64; AN(); CHECK(st_has_err(&r, E_RCX, -1));
    st_fake_ctx(c, &w->main); c->Rax = 1000; AN(); CHECK(st_has_err(&r, E_COUNT, -1));
    st_fake_ctx(c, &w->main); c->MxCsr ^= 0x2000; AN(); CHECK(st_has_err(&r, E_MX2, -1));
    st_fake_ctx(c, &w->main); c->Rip = (uintptr_t)st_hold_ymm_begin; AN(); CHECK(st_has_err(&r, E_KIND, -1));
#else
    st_fake_ctx(c, &w->main); c->Ecx += 64; AN(); CHECK(st_has_err(&r, E_RCX, -1));
    st_fake_ctx(c, &w->main); c->Eax = 1000; AN(); CHECK(st_has_err(&r, E_COUNT, -1));
    st_fake_ctx(c, &w->main); c->Eip = (uintptr_t)st_hold_ymm_begin; AN(); CHECK(st_has_err(&r, E_KIND, -1));
#endif
    /* published block disagreeing with its seed, and a torn (odd) seqlock */
    st_fake_ctx(c, &w->main); w->main.b.vin[0][0] ^= 1; AN(); CHECK(st_has_err(&r, E_REGEN, -1)); w->main.b.vin[0][0] ^= 1;
    st_fake_ctx(c, &w->main); w->main.seq++; AN(); CHECK(st_has_err(&r, E_SEQ, -1)); w->main.seq++;
    /* MMX generation: ST images must carry the 0xFFFF exponent */
    st_publish(&w->main, 43, OP_MMX, 16, w->mx_mode, w->fcw_mode, 100);
    st_fake_ctx(c, &w->main);
    for (int i = 0; i < 8; i++) { memcpy(st_fx(c) + 32 + 16 * i, &w->main.b.mmin[i], 8); st_fx(c)[32 + 16 * i + 8] = 0xFF; st_fx(c)[32 + 16 * i + 9] = 0xFF; }
    AN(); CHECK_EQ(r.nall, 0);
    st_fx(c)[32 + 16 * 6 + 9] = 0x7F; AN(); CHECK(st_has_err(&r, E_MM, 6));
    /* outside the asm: the thread's mode, or the call generation while phase == PH_CALL */
    st_fake_ctx(c, &w->main);
#ifdef T_X64
    c->Rip = (uintptr_t)&st_run_cfg;
#else
    c->Eip = (uintptr_t)&st_run_cfg;
#endif
    memcpy(st_fx(c) + 24, &w->mx_mode, 4); memcpy(st_fx(c), &w->fcw_mode, 2);
    w->phase = PH_C; AN(); CHECK_EQ(r.region, R_OUT); CHECK_EQ(r.nall, 0);
    uint32_t mxs = w->mx_mode | 0x3F;                        /* sticky flags are ignored outside the asm */
    memcpy(st_fx(c) + 24, &mxs, 4); AN(); CHECK_EQ(r.nall, 0);
    memcpy(st_fx(c) + 24, &w->main.b.mx_gen, 4); memcpy(st_fx(c), &w->main.b.fcw_gen, 2);
    if ((w->main.b.mx_gen ^ w->mx_mode) & MX_CTRL) { AN(); CHECK(st_has_err(&r, E_OUT_MX, -1)); }
    if (w->main.b.fcw_gen != w->fcw_mode) { AN(); CHECK(st_has_err(&r, E_OUT_FCW, -1)); }
    w->phase = PH_CALL; AN(); CHECK_EQ(r.nall, 0);
    w->apc_queued = 1; w->phase = PH_C; AN(); CHECK_EQ(r.oracle, 0);   /* APC possibly in delivery: skipped */
    w->apc_queued = 0;
#ifdef T_X64
    c->Rip = (uintptr_t)st_compute_xmm;
#else
    c->Eip = (uintptr_t)st_compute_xmm;
#endif
    AN(); CHECK_EQ(r.region, R_ASM);
#undef AN
    st_vfree(c); st_vfree(w); st_vfree(S.scratch);
}

TEST(xstate_capture_probe) {
    /* Precondition of the XSTATE stress cases: GetThreadContext(CONTEXT_XSTATE) on a suspended thread works when
     * the API, CPUID/XCR0 and GetEnabledXStateFeatures all advertise AVX. */
    int p = st_xstate_probe();
    t_info("xstate_capture", "%s", g_xwhy);
    if (p == 0) SKIP("%s", g_xwhy);
    if (p < 0 && t_is_wine()) t_expect_fail("Wine: XSTATE thread context (%s)", g_xwhy);
    CHECKF(p == 1, "XSTATE thread-context capture advertised but failing: %s", g_xwhy);
}

/* ======================================================================================================
 * 1. Isolation under each yield primitive (no controller): XMM and YMM
 * ====================================================================================================== */
#define ISO(nm, y, k) ST(nm, .threads = 16, .yields = YB(y), .kind = k)
ISO(iso_xmm_spin_preempt_only, Y_SPIN, K_XMM)
ISO(iso_xmm_switchtothread, Y_SWITCH, K_XMM)
ISO(iso_xmm_sleep0, Y_SLEEP0, K_XMM)
ISO(iso_xmm_sleep1, Y_SLEEP1, K_XMM)
ISO(iso_xmm_event_pingpong, Y_PINGPONG, K_XMM)
ISO(iso_xmm_waitable_timer, Y_WTIMER, K_XMM)
ISO(iso_xmm_self_apc, Y_APC, K_XMM)
ISO(iso_xmm_alertable_sleepex, Y_ALERT, K_XMM)
ISO(iso_xmm_critsec_contention, Y_CS, K_XMM)
ISO(iso_xmm_srwlock_contention, Y_SRW, K_XMM)
ISO(iso_xmm_semaphore_contention, Y_SEMA, K_XMM)
ISO(iso_xmm_blocking_syscalls, Y_SYSCALL, K_XMM)
ISO(iso_xmm_self_affinity_hop, Y_HOP, K_XMM)
ISO(iso_xmm_self_priority_change, Y_PRIO, K_XMM)
ISO(iso_ymm_spin_preempt_only, Y_SPIN, K_YMM)
ISO(iso_ymm_switchtothread, Y_SWITCH, K_YMM)
ISO(iso_ymm_sleep0, Y_SLEEP0, K_YMM)
ISO(iso_ymm_sleep1, Y_SLEEP1, K_YMM)
ISO(iso_ymm_event_pingpong, Y_PINGPONG, K_YMM)
ISO(iso_ymm_waitable_timer, Y_WTIMER, K_YMM)
ISO(iso_ymm_self_apc, Y_APC, K_YMM)
ISO(iso_ymm_alertable_sleepex, Y_ALERT, K_YMM)
ISO(iso_ymm_critsec_contention, Y_CS, K_YMM)
ISO(iso_ymm_srwlock_contention, Y_SRW, K_YMM)
ISO(iso_ymm_semaphore_contention, Y_SEMA, K_YMM)
ISO(iso_ymm_blocking_syscalls, Y_SYSCALL, K_YMM)
ISO(iso_ymm_self_affinity_hop, Y_HOP, K_YMM)
ISO(iso_ymm_self_priority_change, Y_PRIO, K_YMM)

/* x87 stack, MMX, compute chains, yielding calls, mixes and scale */
ST(iso_x87_stack_all_yields, .threads = 16, .ops = OPB(OP_X87), .yields = Y_ALL, .kind = K_MIX)
ST(iso_mmx_all_yields, .threads = 16, .ops = OPB(OP_MMX), .yields = Y_ALL, .kind = K_MIX)
ST(iso_compute_xmm_all_yields, .threads = 16, .ops = OPB(OP_COMPUTE), .yields = Y_ALL, .kind = K_XMM)
ST(iso_compute_ymm_all_yields, .threads = 16, .ops = OPB(OP_COMPUTE), .yields = Y_ALL, .kind = K_YMM)
ST(iso_call_nonvolatile_all_yields, .threads = 16, .ops = OPB(OP_CALL), .yields = Y_ALL, .kind = K_XMM)
ST(iso_all_ops_2_threads, .threads = 2, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .ms = 150)
ST(iso_all_ops_4_threads, .threads = 4, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .ms = 150)
ST(scale_64_threads_all_ops, .threads = 64, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .ms = 250)
ST(scale_256_threads_all_ops, .threads = 256, .ops = OPS_ALL, .yields = Y_ALL & ~YB(Y_SPIN), .kind = K_MIX, .ms = 400, .spin = 500)
ST(scale_2x_cpus_spin_preempt, .threads = T_2XCPU, .ops = OPS_HOLDS | OPB(OP_COMPUTE), .yields = YB(Y_SPIN), .kind = K_MIX, .ms = 250)

/* ======================================================================================================
 * 2. Controller: suspend / capture oracle / set unchanged / affinity / priority / APC injection
 * ====================================================================================================== */
ST(ctl_suspend_resume_only, .threads = 16, .ops = OPS_HOLDS | OPB(OP_COMPUTE), .yields = Y_ALL, .kind = K_MIX, .ctl = C_SUSPEND, .ms = 200)
ST(ctl_getctx_xmm_hold, .threads = 8, .ops = OPB(OP_HOLD), .yields = YB(Y_SLEEP0), .ctl = C_GET, .ms = 200)
ST(ctl_getctx_x87_hold, .threads = 8, .ops = OPB(OP_X87), .yields = YB(Y_SWITCH), .ctl = C_GET, .ms = 200)
ST(ctl_getctx_mmx_hold, .threads = 8, .ops = OPB(OP_MMX), .yields = YB(Y_SWITCH), .ctl = C_GET, .ms = 200)
ST(ctl_getctx_mixed_ops_all_yields, .threads = 16, .ops = OPS_ALL, .yields = Y_ALL, .ctl = C_GET, .ms = 250)
ST(ctl_getctx_xstate_ymm_hold, .threads = 8, .ops = OPB(OP_HOLD), .yields = YB(Y_SLEEP0), .kind = K_YMM, .ctl = C_GETX, .ms = 200)
ST(ctl_getctx_xstate_xmm_uppers_clean, .threads = 8, .ops = OPS_HOLDS, .yields = YB(Y_SLEEP0) | YB(Y_SYSCALL), .kind = K_XMM, .ctl = C_GETX, .ms = 200)
ST(ctl_getctx_xstate_mixed_kinds, .threads = 16, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .ctl = C_GETX, .ms = 250)
ST(ctl_setctx_unchanged_xmm, .threads = 8, .ops = OPB(OP_HOLD) | OPB(OP_COMPUTE), .yields = YB(Y_SLEEP0) | YB(Y_SLEEP1), .ctl = C_GET | C_SET, .ms = 200)
ST(ctl_setctx_unchanged_x87_mmx, .threads = 8, .ops = OPB(OP_X87) | OPB(OP_MMX), .yields = YB(Y_SWITCH) | YB(Y_SLEEP1), .ctl = C_GET | C_SET, .ms = 200)
ST(ctl_setctx_unchanged_ymm_xstate, .threads = 8, .ops = OPB(OP_HOLD) | OPB(OP_COMPUTE), .yields = YB(Y_SLEEP0) | YB(Y_ALERT), .kind = K_YMM, .ctl = C_GETX | C_SET, .ms = 200)
ST(ctl_setctx_unchanged_mixed_all, .threads = 16, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .ctl = C_XOPT | C_SET, .ms = 250)
ST(ctl_victim_affinity_hops, .threads = 16, .ops = OPS_HOLDS | OPB(OP_COMPUTE), .yields = YB(Y_SPIN) | YB(Y_SLEEP0), .kind = K_MIX, .ctl = C_XOPT | C_AFF, .ms = 200)
ST(ctl_victim_priority_changes, .threads = 16, .ops = OPS_HOLDS | OPB(OP_COMPUTE), .yields = YB(Y_SPIN) | YB(Y_SWITCH), .kind = K_MIX, .ctl = C_XOPT | C_PRIO, .ms = 200)
ST(ctl_apc_injection, .threads = 8, .ops = OPS_HOLDS, .yields = YB(Y_ALERT) | YB(Y_PINGPONG) | YB(Y_WTIMER), .kind = K_MIX, .ctl = C_XOPT | C_APC, .ms = 250)
ST(ctl_oracle_long_hold_xmm, .threads = 1, .ops = OPB(OP_HOLD), .yields = YB(Y_SPIN), .ctl = C_GET, .spin = 200000, .need_hold = 1, .ms = 250)
ST(ctl_oracle_long_hold_x87, .threads = 1, .ops = OPB(OP_X87), .yields = YB(Y_SPIN), .ctl = C_GET | C_SET, .spin = 200000, .need_hold = 1, .ms = 250)
ST(ctl_oracle_long_hold_mmx, .threads = 1, .ops = OPB(OP_MMX), .yields = YB(Y_SPIN), .ctl = C_GET | C_SET, .spin = 200000, .need_hold = 1, .ms = 250)
ST(ctl_oracle_long_hold_ymm_xstate, .threads = 1, .ops = OPB(OP_HOLD), .yields = YB(Y_SPIN), .kind = K_YMM, .ctl = C_GETX | C_SET, .spin = 200000, .need_hold = 1, .ms = 250)
ST(ctl_oracle_during_yielding_calls, .threads = 8, .ops = OPB(OP_CALL), .yields = YB(Y_SLEEP1) | YB(Y_SLEEP0) | YB(Y_WTIMER) | YB(Y_SYSCALL), .ctl = C_XOPT | C_SET, .ms = 200)
/* Forced time slicing with live patterns: workers pinned to one CPU (or two per CPU) run holds of tens of ms,
 * longer than a scheduling quantum on client Windows, so the OS must switch between pattern-holding threads on
 * the same core mid-hold (counted as holds_descheduled, INFO only: quantum length is a policy). */
ST(preempt_one_cpu_long_holds, .threads = 4, .ops = OPS_HOLDS, .yields = YB(Y_SPIN), .kind = K_MIX, .pin = PIN_ONE,
   .hold_ms = 40, .ms = 500)
ST(preempt_one_cpu_long_holds_captured, .threads = 4, .ops = OPS_HOLDS, .yields = YB(Y_SPIN) | YB(Y_SWITCH), .kind = K_MIX,
   .pin = PIN_ONE, .hold_ms = 30, .ctl = C_XOPT | C_SET, .need_hold = 1, .ms = 500)
ST(preempt_pairs_per_cpu, .threads = T_2XCPU, .ops = OPS_HOLDS | OPB(OP_COMPUTE), .yields = YB(Y_SPIN) | YB(Y_SWITCH),
   .kind = K_MIX, .pin = PIN_PAIRS, .hold_ms = 15, .ms = 500)
/* CPU migration while the patterns are live: the controller moves victims between CPUs during multi-ms holds */
ST(ctl_migrate_during_long_holds, .threads = 8, .ops = OPS_HOLDS, .yields = YB(Y_SPIN), .kind = K_MIX, .hold_ms = 5,
   .ctl = C_XOPT | C_AFF, .need_hold = 1, .ms = 400)
ST(ctl_everything_2x_cpus,.threads = T_2XCPU, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .ctl = C_XOPT | C_SET | C_AFF | C_PRIO | C_APC, .ms = 400)

/* ======================================================================================================
 * 3. Process-wide timer interrupts: timer-queue and multimedia timer callbacks
 * ====================================================================================================== */
ST(tmr_queue_apc_injection, .threads = 16, .ops = OPS_HOLDS, .yields = YB(Y_ALERT) | YB(Y_PINGPONG), .kind = K_MIX, .tmr = T_TQ | TA_APC, .ms = 250)
ST(tmr_queue_kick_suspend, .threads = 16, .ops = OPS_HOLDS | OPB(OP_COMPUTE), .yields = YB(Y_SPIN) | YB(Y_SLEEP0), .kind = K_MIX, .tmr = T_TQ | TA_KICK, .ms = 250)
ST(tmr_queue_wake_events, .threads = 16, .ops = OPS_HOLDS, .yields = YB(Y_PINGPONG), .kind = K_MIX, .tmr = T_TQ | TA_WAKE, .ms = 250)
ST(tmr_queue_callback_regs, .threads = 8, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .tmr = T_TQ | TA_REGS, .ms = 250)
ST(tmr_mm_apc_injection, .threads = 16, .ops = OPS_HOLDS, .yields = YB(Y_ALERT) | YB(Y_WTIMER), .kind = K_MIX, .tmr = T_MM | TA_APC, .ms = 250)
ST(tmr_mm_kick_suspend, .threads = 16, .ops = OPS_HOLDS | OPB(OP_COMPUTE), .yields = YB(Y_SPIN) | YB(Y_SWITCH), .kind = K_MIX, .tmr = T_MM | TA_KICK, .ms = 250)
ST(tmr_mm_callback_regs, .threads = 8, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .tmr = T_MM | TA_REGS | TA_WAKE, .ms = 250)
ST(tmr_threadpool_callback_regs, .threads = 8, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .tmr = T_TQ | TA_POOL | TA_REGS, .ms = 300)
ST(tmr_everything_with_controller,.threads = T_2XCPU, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX,
   .ctl = C_XOPT | C_SET | C_AFF | C_PRIO | C_APC, .tmr = T_TQ | T_MM | TA_APC | TA_WAKE | TA_KICK | TA_REGS | TA_POOL, .ms = 500)

/* ======================================================================================================
 * 4. Oracle self-tests (an injected SetThreadContext change must be seen exactly), churn, rates
 * ====================================================================================================== */
ST(selftest_mutation_xmm_detected, .threads = 1, .ops = OPB(OP_HOLD), .yields = YB(Y_SPIN), .ctl = C_GET, .spin = 200000, .mutate = 1, .ms = 400)
ST(selftest_mutation_mxcsr_detected, .threads = 1, .ops = OPB(OP_HOLD), .yields = YB(Y_SPIN), .ctl = C_GET, .spin = 200000, .mutate = 2, .ms = 400)
ST(selftest_mutation_ymm_upper_detected, .threads = 1, .ops = OPB(OP_HOLD), .yields = YB(Y_SPIN), .kind = K_YMM, .ctl = C_GETX, .spin = 200000, .mutate = 3, .ms = 400)
ST(churn_new_threads_under_stress, .threads = 16, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX, .ctl = C_XOPT | C_SET, .churn = 1, .ms = 400)
ST(perf_stress_rates_plain_capture, .threads = 4, .ops = OPB(OP_HOLD) | OPB(OP_COMPUTE), .yields = YB(Y_SLEEP0), .ctl = C_GET, .perf = 1, .ms = 300)
ST(perf_stress_rates_xstate_capture, .threads = 4, .ops = OPB(OP_HOLD) | OPB(OP_COMPUTE), .yields = YB(Y_SLEEP0), .kind = K_MIX, .ctl = C_GETX, .perf = 1, .ms = 300)

/* opt-in soak (--include-manual): everything for 30 s */
TEST_MANUAL(soak_everything_30s) {
    static const st_cfg c_ = { .threads = T_2XCPU, .ops = OPS_ALL, .yields = Y_ALL, .kind = K_MIX,
                               .ctl = C_XOPT | C_SET | C_AFF | C_PRIO | C_APC, .tmr = T_TQ | T_MM | TA_APC | TA_WAKE | TA_KICK | TA_REGS | TA_POOL,
                               .churn = 1, .ms = 30000 };
    st_run_cfg(&c_, "soak_everything_30s");
}
