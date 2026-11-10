/* context_xstate - GetThreadContext / SetThreadContext, XSTATE-aware CONTEXT buffers and RtlCaptureContext.
 *
 * Contract under test:
 *   - A suspended thread's CONTEXT reports exactly the architectural state the thread holds: GPRs, RIP/RSP,
 *     segment selectors, EFLAGS, x87 control word / stack (FXSAVE layout), MXCSR, XMM, and with
 *     CONTEXT_XSTATE the AVX (YMM_Hi128) and AVX-512 (opmask / ZMM_Hi256 / Hi16_ZMM) components located
 *     with LocateXStateFeature. SetThreadContext changes exactly the requested state, and the resumed
 *     thread observes it. Partial ContextFlags leave the other fields of the caller's buffer untouched.
 *   - Worker threads hold their state inside one asm routine (cx_spin_*): load registers, publish
 *     "ready", spin on a stop flag with GPR-only instructions, store registers. Nothing compiled runs in
 *     between, so every difference is a context-API defect, not compiler noise.
 *   - Buffer negotiation and parameter validation of the XSTATE APIs. Behaviour pinned against the
 *     Windows 10 22H2 (19041) reference decompilation (docker1 corpus explorer), checked 2026-11:
 *       kernelbase!InitializeContext2: Buffer == NULL or *ContextLength < required -> *ContextLength =
 *         required, ERROR_INSUFFICIENT_BUFFER (0x7a), FALSE.  Invalid flags -> RtlGetExtendedContextLength2
 *         -> RtlpValidateContextFlags -> STATUS_INVALID_PARAMETER -> ERROR_INVALID_PARAMETER.
 *       ntdll!RtlpValidateContextFlags: exactly one architecture bit; AMD64 accepts only
 *         (flags & 0x27ffffa0) == 0x100000, i386 (flags & 0x27ffff80) == 0x10000; XSTATE (0x40) needs
 *         enabled features else STATUS_NOT_SUPPORTED.
 *       kernelbase!LocateXStateFeature: feature 0 -> &FltSave (x64, len 0xa0) / ExtendedRegisters (x86),
 *         feature 1 -> &Xmm0 (x64, len 0x100) / ExtendedRegisters+0xa0 (x86, len 0x80); features >= 2 only
 *         with CONTEXT_XSTATE, via RtlLocateExtendedFeature2 (2..63 and enabled, else NULL).
 *       kernelbase!SetXStateFeaturesMask: mask & 3 sets CONTEXT_FLOATING_POINT (x64) /
 *         CONTEXT_EXTENDED_REGISTERS (x86); without an XSTATE chunk, bits above 1 -> ERROR_INVALID_PARAMETER;
 *         with XSTATE, ntdll!RtlSetExtendedFeaturesMask stores mask & enabled & ~3.
 *       kernelbase!GetXStateFeaturesMask: no x86/AMD64 flag -> STATUS_NOT_SUPPORTED (ERROR_NOT_SUPPORTED).
 *       kernelbase!CopyContext -> ntdll!RtlCopyContext: XSTATE source into a destination without an XSTATE
 *         chunk -> STATUS_BUFFER_OVERFLOW (ERROR_MORE_DATA); invalid flags -> ERROR_INVALID_PARAMETER.
 *       ntoskrnl!NtGetContextThread / NtSetContextThread: ObReferenceObjectByHandle with THREAD_GET_CONTEXT
 *         (8) / THREAD_SET_CONTEXT (0x10) -> ERROR_ACCESS_DENIED / ERROR_INVALID_HANDLE; user-mode flags are
 *         sanitized (RtlpSanitizeContextFlags), not rejected, so odd flags are observations (NOTE) only.
 *       ntdll!RtlCaptureContext (x64): ContextFlags = 0x10000F, FXSAVE into FltSave, Rsp = caller RSP after
 *         return, Rip = return address, MxCsr via STMXCSR.
 *   - x86 RtlCaptureContext: Windows takes Eip/Esp/Ebp from the caller's EBP frame, other implementations
 *     from the call itself; both readings are accepted, nothing else.
 *   - WOW64: the x86 build exercises the WOW64 context path when it runs on a 64-bit OS; the x64 build
 *     drives Wow64SuspendThread / Wow64GetThreadContext / Wow64SetThreadContext against the sibling x86
 *     build (build/x86/context_xstate.exe) when it exists, falling back to the documented ntdll implementations
 *     (RtlWow64SuspendThread, NtQuery/SetInformationThread(ThreadWow64Context)) when kernel32 lacks an export;
 *     the WOW64_CONTEXT contract is pinned in section 11.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> t_note/t_info.
 * Deliberately invalid pointers run in CHILD() processes via t_child_run(). See common/API.md. */
#include "t.h"

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
#define CX_ARCHBIT   CONTEXT_AMD64
#define CX_STD       (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT)
#define CX_XMMF      CONTEXT_FLOATING_POINT          /* flags that carry XMM/MXCSR */
#define CX_NGPR      13
#define CX_CTX_ALIGN 16
#define CX_SSE_LEN   0x100
#define CX_FX_DEFINED 416           /* FXSAVE bytes with architectural content in 64-bit mode (XMM0-15 end) */
#define CX_ARCHNAME  "x64"
#else
#define CX_ARCHBIT   CONTEXT_i386
#define CX_STD       (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS)
#define CX_XMMF      CONTEXT_EXTENDED_REGISTERS
#define CX_NGPR      5
#define CX_CTX_ALIGN 4
#define CX_SSE_LEN   0x80
#define CX_FX_DEFINED 288           /* 32-bit mode: XMM0-7 end */
#define CX_ARCHNAME  "x86"
#endif
#define CX_NV        T_NXMM          /* XMM/YMM registers in this mode: 16 (x64) / 8 (x86) */
#define CX_NZ        T_NZMM          /* ZMM registers: 32 / 8 */
#define CX_SENT      0xA5u
#define CX_TMO       5000u

#define CX_XS_AVX     2
#define CX_XS_KMASK   5
#define CX_XS_ZMM_H   6
#define CX_XS_HI16    7

/* ======================================================================================================
 * Worker argument block shared with the asm routines (fixed layout, checked below)
 * ====================================================================================================== */
#define CXO_VIN    0
#define CXO_VOUT   512
#define CXO_GIN    1024
#define CXO_GOUT   1152
#define CXO_FIN    1280
#define CXO_FOUT   1360
#define CXO_READY  1440
#define CXO_STOP   1444
#define CXO_MXIN   1448
#define CXO_MXOUT  1452
#define CXO_FCWIN  1456
#define CXO_FCWOUT 1458
#define CXO_FSWOUT 1460
#define CXO_FLAGS  1464
#define CXO_MAX    1468
#define CXO_REM    1472
#define CXO_EFL    1476
#define CXO_SP     1480
#define CXO_ZIN    1536
#define CXO_ZOUT   3584
#define CXO_KIN    5632
#define CXO_KOUT   5696
/* A check of a pinned Windows contract that Wine is known not to implement: on Wine a failure turns the case
 * into XFAIL (named reason), on everything else it is an ordinary CHECK. Never weakens the check itself. */
#define CX_CHECK_WGAP(gap, c, ...) do { int ok_ = !!(c); \
    if (!ok_ && t_is_wine()) t_expect_fail("wine %s: %s", t_wine_version(), gap); \
    CHECKF(ok_, __VA_ARGS__); } while (0)
/* x87 control word bits that exist architecturally (bit 6 and 13-15 reserved); Windows masks a CONTEXT
 * control word with 0x1F3F (ntoskrnl!KxContextToKframes 19041) and hardware may report bit 6 as 1 */
#define CX_FCW_BITS 0x1F3Fu
#define CX_S_(x) #x
#define CX_S(x) CX_S_(x)

/* CXF_X87: load ST(0..7) from fin; CXF_X87_3: with CXF_X87, load only ST(0..2) (TOP = 5);
 * CXF_VZU: worker executes VZEROUPPER before loading (only set when AVX is usable) */
enum { CXF_X87 = 1, CXF_KBW = 2, CXF_X87_3 = 4, CXF_VZU = 8 };

typedef struct cx_args {
    uint8_t  vin[16][32];        /* XMM/YMM i: low 16 bytes = XMM, bytes 16..31 = upper YMM lane */
    uint8_t  vout[16][32];
    uint64_t gin[16];            /* GPR values, see cx_gpr_name() */
    uint64_t gout[16];
    t_f80    fin[8];             /* fin[0] -> ST(0) */
    t_f80    fout[8];
    volatile LONG ready;
    volatile LONG stop;
    uint32_t mxcsr_in, mxcsr_out;
    uint16_t fcw_in, fcw_out, fsw_out, pad0;
    uint32_t flags, max_loops, remaining, eflags_out;   /* eflags_out: EFLAGS at the loop exit */
    uint64_t sp;                 /* RSP/ESP inside the spin loop */
    uint8_t  pad2[CXO_ZIN - 1488];
    uint8_t  zin[32][64];
    uint8_t  zout[32][64];
    uint64_t kin[8];
    uint64_t kout[8];
} cx_args;
_Static_assert(offsetof(cx_args, vout) == CXO_VOUT, "vout");
_Static_assert(offsetof(cx_args, gin) == CXO_GIN, "gin");
_Static_assert(offsetof(cx_args, gout) == CXO_GOUT, "gout");
_Static_assert(offsetof(cx_args, fin) == CXO_FIN, "fin");
_Static_assert(offsetof(cx_args, fout) == CXO_FOUT, "fout");
_Static_assert(offsetof(cx_args, ready) == CXO_READY, "ready");
_Static_assert(offsetof(cx_args, stop) == CXO_STOP, "stop");
_Static_assert(offsetof(cx_args, mxcsr_in) == CXO_MXIN, "mxin");
_Static_assert(offsetof(cx_args, mxcsr_out) == CXO_MXOUT, "mxout");
_Static_assert(offsetof(cx_args, fcw_in) == CXO_FCWIN, "fcwin");
_Static_assert(offsetof(cx_args, fcw_out) == CXO_FCWOUT, "fcwout");
_Static_assert(offsetof(cx_args, fsw_out) == CXO_FSWOUT, "fswout");
_Static_assert(offsetof(cx_args, flags) == CXO_FLAGS, "flags");
_Static_assert(offsetof(cx_args, max_loops) == CXO_MAX, "max");
_Static_assert(offsetof(cx_args, remaining) == CXO_REM, "rem");
_Static_assert(offsetof(cx_args, eflags_out) == CXO_EFL, "efl");
_Static_assert(offsetof(cx_args, sp) == CXO_SP, "sp");
_Static_assert(offsetof(cx_args, zin) == CXO_ZIN, "zin");
_Static_assert(offsetof(cx_args, zout) == CXO_ZOUT, "zout");
_Static_assert(offsetof(cx_args, kin) == CXO_KIN, "kin");
_Static_assert(offsetof(cx_args, kout) == CXO_KOUT, "kout");

/* RtlCaptureContext probe block */
typedef struct cx_cap {
    uint8_t  xin[16][16];        /* 0   */
    uint64_t gin[8];             /* 256: x64 rbx rbp rsi rdi r12 r13 r14 r15; x86 ebx - esi edi */
    uint64_t fn;                 /* 320 */
    uint64_t ctx;                /* 328 */
    uint64_t sp_after;           /* 336: RSP/ESP right after the call returned */
    uint64_t frame_ip;           /* 344: x86 [ebp+4] */
    uint64_t frame_sp;           /* 352: x86 ebp+8 */
    uint64_t frame_bp;           /* 360: x86 [ebp] */
    uint64_t own_bp;             /* 368: x86 ebp of the probe */
    uint32_t mxcsr;              /* 376 */
    uint32_t pad;
} cx_cap;
_Static_assert(offsetof(cx_cap, fn) == 320 && offsetof(cx_cap, mxcsr) == 376, "cx_cap layout");

/* asm entry points and labels (exact assembler names on both architectures) */
void cx_spin_xmm(cx_args *a) __asm__("cx_spin_xmm");
void cx_spin_ymm(cx_args *a) __asm__("cx_spin_ymm");
void cx_spin_zmm(cx_args *a) __asm__("cx_spin_zmm");
void cx_capture(cx_cap *c) __asm__("cx_capture");
extern const char cx_xmm_loop[] __asm__("cx_xmm_loop");
extern const char cx_xmm_exit[] __asm__("cx_xmm_exit");
extern const char cx_ymm_loop[] __asm__("cx_ymm_loop");
extern const char cx_ymm_exit[] __asm__("cx_ymm_exit");
extern const char cx_zmm_loop[] __asm__("cx_zmm_loop");
extern const char cx_zmm_exit[] __asm__("cx_zmm_exit");
extern const char cx_cap_ret[] __asm__("cx_cap_ret");

#define CXA(o) CX_S(o)
#define CX_X87_LOAD(b) \
    "  testl $1, " CXA(CXO_FLAGS) "(" b ")\n  jz 1f\n  testl $4, " CXA(CXO_FLAGS) "(" b ")\n  jnz 3f\n" \
    "  fldt " CXA(CXO_FIN) "+70(" b ")\n  fldt " CXA(CXO_FIN) "+60(" b ")\n  fldt " CXA(CXO_FIN) "+50(" b ")\n" \
    "  fldt " CXA(CXO_FIN) "+40(" b ")\n  fldt " CXA(CXO_FIN) "+30(" b ")\n3:\n  fldt " CXA(CXO_FIN) "+20(" b ")\n" \
    "  fldt " CXA(CXO_FIN) "+10(" b ")\n  fldt " CXA(CXO_FIN) "+0(" b ")\n1:\n"
#define CX_X87_STORE(b) \
    "  stmxcsr " CXA(CXO_MXOUT) "(" b ")\n  fnstcw " CXA(CXO_FCWOUT) "(" b ")\n  fnstsw " CXA(CXO_FSWOUT) "(" b ")\n" \
    "  testl $1, " CXA(CXO_FLAGS) "(" b ")\n  jz 2f\n" \
    "  fstpt " CXA(CXO_FOUT) "+0(" b ")\n  fstpt " CXA(CXO_FOUT) "+10(" b ")\n  fstpt " CXA(CXO_FOUT) "+20(" b ")\n" \
    "  testl $4, " CXA(CXO_FLAGS) "(" b ")\n  jnz 2f\n" \
    "  fstpt " CXA(CXO_FOUT) "+30(" b ")\n  fstpt " CXA(CXO_FOUT) "+40(" b ")\n  fstpt " CXA(CXO_FOUT) "+50(" b ")\n" \
    "  fstpt " CXA(CXO_FOUT) "+60(" b ")\n  fstpt " CXA(CXO_FOUT) "+70(" b ")\n2:\n"
#define CX_FPENV_LOAD(b) \
    "  ldmxcsr " CXA(CXO_MXIN) "(" b ")\n  fninit\n  fldcw " CXA(CXO_FCWIN) "(" b ")\n"
#define CX_GLOBAL(n) ".globl " n "\n" n ":\n"

#ifdef T_X64
#define CX_R16 "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#define CX_RV  CX_R16
#define CX_RZ  CX_R16 ",16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
#define CX_PRO(name) \
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
    "  .seh_endprologue\n"
#define CX_EPI \
    "  movdqa 48(%rsp), %xmm6\n  movdqa 64(%rsp), %xmm7\n  movdqa 80(%rsp), %xmm8\n  movdqa 96(%rsp), %xmm9\n" \
    "  movdqa 112(%rsp), %xmm10\n  movdqa 128(%rsp), %xmm11\n  movdqa 144(%rsp), %xmm12\n  movdqa 160(%rsp), %xmm13\n" \
    "  movdqa 176(%rsp), %xmm14\n  movdqa 192(%rsp), %xmm15\n" \
    "  addq $216, %rsp\n  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n" \
    "  popq %rdi\n  popq %rsi\n  popq %rbx\n  popq %rbp\n  ret\n.seh_endproc\n"
/* gin/gout index: 0 rbx 1 rcx 2 rdx 3 rsi 4 rdi 5 rbp 6 r8 7 r9 8 r10 9 r12 10 r13 11 r14 12 r15
 * Loop registers: rax = &ready (stop at +4), r11d = remaining count. */
#define CX_SPIN(name, loop, exit, LOADV, STOREV, TAIL) \
    CX_PRO(name) \
    "  movq %rcx, 32(%rsp)\n  movq %rsp, " CXA(CXO_SP) "(%rcx)\n" \
    CX_FPENV_LOAD("%rcx") CX_X87_LOAD("%rcx") LOADV \
    "  leaq " CXA(CXO_READY) "(%rcx), %rax\n  movl " CXA(CXO_MAX) "(%rcx), %r11d\n" \
    "  movq " CXA(CXO_GIN) "+0(%rcx), %rbx\n  movq " CXA(CXO_GIN) "+16(%rcx), %rdx\n" \
    "  movq " CXA(CXO_GIN) "+24(%rcx), %rsi\n  movq " CXA(CXO_GIN) "+32(%rcx), %rdi\n" \
    "  movq " CXA(CXO_GIN) "+40(%rcx), %rbp\n  movq " CXA(CXO_GIN) "+48(%rcx), %r8\n" \
    "  movq " CXA(CXO_GIN) "+56(%rcx), %r9\n  movq " CXA(CXO_GIN) "+64(%rcx), %r10\n" \
    "  movq " CXA(CXO_GIN) "+72(%rcx), %r12\n  movq " CXA(CXO_GIN) "+80(%rcx), %r13\n" \
    "  movq " CXA(CXO_GIN) "+88(%rcx), %r14\n  movq " CXA(CXO_GIN) "+96(%rcx), %r15\n" \
    "  movq " CXA(CXO_GIN) "+8(%rcx), %rcx\n" \
    "  movl $1, (%rax)\n" \
    CX_GLOBAL(loop) \
    "  cmpl $0, 4(%rax)\n  jne " exit "\n  pause\n  decl %r11d\n  jnz " loop "\n" \
    CX_GLOBAL(exit) \
    "  pushfq\n  subq $" CXA(CXO_READY) ", %rax\n  movl %r11d, " CXA(CXO_REM) "(%rax)\n" \
    "  popq %r11\n  movl %r11d, " CXA(CXO_EFL) "(%rax)\n  cld\n" \
    "  movq %rbx, " CXA(CXO_GOUT) "+0(%rax)\n  movq %rcx, " CXA(CXO_GOUT) "+8(%rax)\n" \
    "  movq %rdx, " CXA(CXO_GOUT) "+16(%rax)\n  movq %rsi, " CXA(CXO_GOUT) "+24(%rax)\n" \
    "  movq %rdi, " CXA(CXO_GOUT) "+32(%rax)\n  movq %rbp, " CXA(CXO_GOUT) "+40(%rax)\n" \
    "  movq %r8, " CXA(CXO_GOUT) "+48(%rax)\n  movq %r9, " CXA(CXO_GOUT) "+56(%rax)\n" \
    "  movq %r10, " CXA(CXO_GOUT) "+64(%rax)\n  movq %r12, " CXA(CXO_GOUT) "+72(%rax)\n" \
    "  movq %r13, " CXA(CXO_GOUT) "+80(%rax)\n  movq %r14, " CXA(CXO_GOUT) "+88(%rax)\n" \
    "  movq %r15, " CXA(CXO_GOUT) "+96(%rax)\n" \
    STOREV CX_X87_STORE("%rax") TAIL CX_EPI

__asm__(
    ".text\n"
    CX_SPIN("cx_spin_xmm", "cx_xmm_loop", "cx_xmm_exit",
            "  .irp i," CX_RV "\n  movdqu \\i*32(%rcx), %xmm\\i\n  .endr\n",
            "  .irp i," CX_RV "\n  movdqu %xmm\\i, " CXA(CXO_VOUT) "+\\i*32(%rax)\n  .endr\n",
            "")
    CX_SPIN("cx_spin_ymm", "cx_ymm_loop", "cx_ymm_exit",
            "  .irp i," CX_RV "\n  vmovdqu \\i*32(%rcx), %ymm\\i\n  .endr\n",
            "  .irp i," CX_RV "\n  vmovdqu %ymm\\i, " CXA(CXO_VOUT) "+\\i*32(%rax)\n  .endr\n",
            "  vzeroupper\n")
    CX_SPIN("cx_spin_zmm", "cx_zmm_loop", "cx_zmm_exit",
            "  .irp i," CX_RZ "\n  vmovdqu64 " CXA(CXO_ZIN) "+\\i*64(%rcx), %zmm\\i\n  .endr\n"
            "  testl $2, " CXA(CXO_FLAGS) "(%rcx)\n  jz 5f\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovq " CXA(CXO_KIN) "+\\i*8(%rcx), %k\\i\n  .endr\n  jmp 6f\n5:\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovw " CXA(CXO_KIN) "+\\i*8(%rcx), %k\\i\n  .endr\n6:\n",
            "  .irp i," CX_RZ "\n  vmovdqu64 %zmm\\i, " CXA(CXO_ZOUT) "+\\i*64(%rax)\n  .endr\n"
            "  testl $2, " CXA(CXO_FLAGS) "(%rax)\n  jz 7f\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovq %k\\i, " CXA(CXO_KOUT) "+\\i*8(%rax)\n  .endr\n  jmp 8f\n7:\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovw %k\\i, " CXA(CXO_KOUT) "+\\i*8(%rax)\n  .endr\n8:\n",
            "  vzeroupper\n")

    /* void cx_capture(cx_cap *c): load XMM0-15, MXCSR and the nonvolatile GPRs, call c->fn(c->ctx)
     * (RtlCaptureContext), record RSP after the call. */
    CX_PRO("cx_capture")
    "  movq %rcx, 32(%rsp)\n"
    "  ldmxcsr 376(%rcx)\n"
    "  .irp i," CX_R16 "\n  movdqu \\i*16(%rcx), %xmm\\i\n  .endr\n"
    "  movq 256(%rcx), %rbx\n  movq 264(%rcx), %rbp\n  movq 272(%rcx), %rsi\n  movq 280(%rcx), %rdi\n"
    "  movq 288(%rcx), %r12\n  movq 296(%rcx), %r13\n  movq 304(%rcx), %r14\n  movq 312(%rcx), %r15\n"
    "  movq 320(%rcx), %rax\n  movq 328(%rcx), %rcx\n"
    "  callq *%rax\n"
    CX_GLOBAL("cx_cap_ret")
    "  movq 32(%rsp), %rax\n  movq %rsp, 336(%rax)\n"
    CX_EPI
);
#else /* x86 */
#define CX_RV  "0,1,2,3,4,5,6,7"
#define CX_RZ  CX_RV
/* gin/gout index: 0 ebx 1 edx 2 esi 3 edi 4 ebp.  Loop registers: eax = &ready, ecx = remaining. */
#define CX_SPIN(name, loop, exit, LOADV, STOREV, TAIL) \
    ".globl " name "\n" name ":\n" \
    "  pushl %ebp\n  pushl %ebx\n  pushl %esi\n  pushl %edi\n  movl 20(%esp), %ecx\n" \
    "  movl %esp, " CXA(CXO_SP) "(%ecx)\n  movl $0, " CXA(CXO_SP) "+4(%ecx)\n" \
    CX_FPENV_LOAD("%ecx") CX_X87_LOAD("%ecx") LOADV \
    "  leal " CXA(CXO_READY) "(%ecx), %eax\n" \
    "  movl " CXA(CXO_GIN) "+0(%ecx), %ebx\n  movl " CXA(CXO_GIN) "+8(%ecx), %edx\n" \
    "  movl " CXA(CXO_GIN) "+16(%ecx), %esi\n  movl " CXA(CXO_GIN) "+24(%ecx), %edi\n" \
    "  movl " CXA(CXO_GIN) "+32(%ecx), %ebp\n  movl " CXA(CXO_MAX) "(%ecx), %ecx\n" \
    "  movl $1, (%eax)\n" \
    CX_GLOBAL(loop) \
    "  cmpl $0, 4(%eax)\n  jne " exit "\n  pause\n  decl %ecx\n  jnz " loop "\n" \
    CX_GLOBAL(exit) \
    "  pushfl\n  subl $" CXA(CXO_READY) ", %eax\n  movl %ecx, " CXA(CXO_REM) "(%eax)\n" \
    "  popl %ecx\n  movl %ecx, " CXA(CXO_EFL) "(%eax)\n  cld\n" \
    "  movl %ebx, " CXA(CXO_GOUT) "+0(%eax)\n  movl $0, " CXA(CXO_GOUT) "+4(%eax)\n" \
    "  movl %edx, " CXA(CXO_GOUT) "+8(%eax)\n  movl $0, " CXA(CXO_GOUT) "+12(%eax)\n" \
    "  movl %esi, " CXA(CXO_GOUT) "+16(%eax)\n  movl $0, " CXA(CXO_GOUT) "+20(%eax)\n" \
    "  movl %edi, " CXA(CXO_GOUT) "+24(%eax)\n  movl $0, " CXA(CXO_GOUT) "+28(%eax)\n" \
    "  movl %ebp, " CXA(CXO_GOUT) "+32(%eax)\n  movl $0, " CXA(CXO_GOUT) "+36(%eax)\n" \
    STOREV CX_X87_STORE("%eax") TAIL \
    "  popl %edi\n  popl %esi\n  popl %ebx\n  popl %ebp\n  ret\n"

__asm__(
    ".text\n"
    CX_SPIN("cx_spin_xmm", "cx_xmm_loop", "cx_xmm_exit",
            "  .irp i," CX_RV "\n  movdqu \\i*32(%ecx), %xmm\\i\n  .endr\n",
            "  .irp i," CX_RV "\n  movdqu %xmm\\i, " CXA(CXO_VOUT) "+\\i*32(%eax)\n  .endr\n",
            "")
    CX_SPIN("cx_spin_ymm", "cx_ymm_loop", "cx_ymm_exit",
            "  .irp i," CX_RV "\n  vmovdqu \\i*32(%ecx), %ymm\\i\n  .endr\n",
            "  .irp i," CX_RV "\n  vmovdqu %ymm\\i, " CXA(CXO_VOUT) "+\\i*32(%eax)\n  .endr\n",
            "  vzeroupper\n")
    CX_SPIN("cx_spin_zmm", "cx_zmm_loop", "cx_zmm_exit",
            "  .irp i," CX_RZ "\n  vmovdqu64 " CXA(CXO_ZIN) "+\\i*64(%ecx), %zmm\\i\n  .endr\n"
            "  testl $2, " CXA(CXO_FLAGS) "(%ecx)\n  jz 5f\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovq " CXA(CXO_KIN) "+\\i*8(%ecx), %k\\i\n  .endr\n  jmp 6f\n5:\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovw " CXA(CXO_KIN) "+\\i*8(%ecx), %k\\i\n  .endr\n6:\n",
            "  .irp i," CX_RZ "\n  vmovdqu64 %zmm\\i, " CXA(CXO_ZOUT) "+\\i*64(%eax)\n  .endr\n"
            "  testl $2, " CXA(CXO_FLAGS) "(%eax)\n  jz 7f\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovq %k\\i, " CXA(CXO_KOUT) "+\\i*8(%eax)\n  .endr\n  jmp 8f\n7:\n"
            "  .irp i,0,1,2,3,4,5,6,7\n  kmovw %k\\i, " CXA(CXO_KOUT) "+\\i*8(%eax)\n  .endr\n8:\n",
            "  vzeroupper\n")

    /* void cx_capture(cx_cap *c) (cdecl): standard EBP frame, record both candidate (Eip, Esp, Ebp) triples,
     * load XMM0-7, MXCSR, ebx/esi/edi, call c->fn(c->ctx) (stdcall), record ESP after the call. */
    ".globl cx_capture\ncx_capture:\n"
    "  pushl %ebp\n  movl %esp, %ebp\n  pushl %ebx\n  pushl %esi\n  pushl %edi\n"
    "  movl 8(%ebp), %ecx\n  pushl %ecx\n"
    "  ldmxcsr 376(%ecx)\n"
    "  .irp i," CX_RV "\n  movdqu \\i*16(%ecx), %xmm\\i\n  .endr\n"
    "  movl %ebp, 368(%ecx)\n  movl $0, 372(%ecx)\n"
    "  movl 4(%ebp), %eax\n  movl %eax, 344(%ecx)\n  movl $0, 348(%ecx)\n"
    "  leal 8(%ebp), %eax\n  movl %eax, 352(%ecx)\n  movl $0, 356(%ecx)\n"
    "  movl 0(%ebp), %eax\n  movl %eax, 360(%ecx)\n  movl $0, 364(%ecx)\n"
    "  movl 256(%ecx), %ebx\n  movl 272(%ecx), %esi\n  movl 280(%ecx), %edi\n"
    "  movl 320(%ecx), %eax\n  pushl 328(%ecx)\n  call *%eax\n"
    CX_GLOBAL("cx_cap_ret")
    "  movl -16(%ebp), %eax\n  movl %esp, 336(%eax)\n  movl $0, 340(%eax)\n"
    "  addl $4, %esp\n  popl %edi\n  popl %esi\n  popl %ebx\n  popl %ebp\n  ret\n"
);
#endif

/* ======================================================================================================
 * Windows API (resolved at run time: the exe must load on systems without the XSTATE APIs)
 * ====================================================================================================== */
static struct {
    int resolved;
    __typeof__(&InitializeContext) InitializeContext;
    BOOL (WINAPI *InitializeContext2)(PVOID, DWORD, PCONTEXT *, PDWORD, ULONG64);
    __typeof__(&LocateXStateFeature) LocateXStateFeature;
    __typeof__(&GetXStateFeaturesMask) GetXStateFeaturesMask;
    __typeof__(&SetXStateFeaturesMask) SetXStateFeaturesMask;
    __typeof__(&CopyContext) CopyContext;
    __typeof__(&GetEnabledXStateFeatures) GetEnabledXStateFeatures;
    __typeof__(&Wow64GetThreadContext) Wow64GetThreadContext;
    __typeof__(&Wow64SetThreadContext) Wow64SetThreadContext;
    __typeof__(&Wow64SuspendThread) Wow64SuspendThread;
    __typeof__(&RtlCaptureContext) RtlCaptureContext;
} X;

static void cx_resolve(void) {
    if (X.resolved) return;
    X.InitializeContext = T_PROC("kernel32.dll", InitializeContext);
    X.InitializeContext2 = (BOOL (WINAPI *)(PVOID, DWORD, PCONTEXT *, PDWORD, ULONG64))t_proc("kernel32.dll", "InitializeContext2");
    X.LocateXStateFeature = T_PROC("kernel32.dll", LocateXStateFeature);
    X.GetXStateFeaturesMask = T_PROC("kernel32.dll", GetXStateFeaturesMask);
    X.SetXStateFeaturesMask = T_PROC("kernel32.dll", SetXStateFeaturesMask);
    X.CopyContext = T_PROC("kernel32.dll", CopyContext);
    X.GetEnabledXStateFeatures = T_PROC("kernel32.dll", GetEnabledXStateFeatures);
    X.Wow64GetThreadContext = T_PROC("kernel32.dll", Wow64GetThreadContext);
    X.Wow64SetThreadContext = T_PROC("kernel32.dll", Wow64SetThreadContext);
    X.Wow64SuspendThread = T_PROC("kernel32.dll", Wow64SuspendThread);
    X.RtlCaptureContext = T_PROC("kernel32.dll", RtlCaptureContext);
    if (!X.RtlCaptureContext) X.RtlCaptureContext = T_PROC("ntdll.dll", RtlCaptureContext);
    X.resolved = 1;
}
/* The Windows 7 SP1 XSTATE context API set */
static int cx_xapi(void) {
    cx_resolve();
    return X.InitializeContext && X.LocateXStateFeature && X.GetXStateFeaturesMask && X.SetXStateFeaturesMask &&
           X.CopyContext && X.GetEnabledXStateFeatures;
}
#define CX_NEED_XAPI() do { if (!cx_xapi()) SKIP("XSTATE context API (InitializeContext, LocateXStateFeature, ...) not exported"); } while (0)
static uint64_t cx_enabled(void) { return cx_xapi() ? X.GetEnabledXStateFeatures() : 0; }
/* XSTATE features reported by Windows that are user (XCR0) components */
static uint64_t cx_enabled_user(void) {
    uint64_t e = cx_enabled();
    const t_cpu_info *c = cpu();
    return c->xcr0_valid ? (e & c->xcr0) : e;
}
static int cx_avx_ok(void) { return cpu_has(F_AVX) && (cx_enabled() & T_XS_AVX); }
static int cx_zmm_ok(void) {
    return cpu_has(F_AVX512F) && (cx_enabled() & T_XS_ZMM_STATE) == T_XS_ZMM_STATE;
}
#define CX_NEED_AVX() do { CX_NEED_XAPI(); \
    if (!cpu_has(F_AVX)) SKIP("AVX not usable (CPUID/XCR0)"); \
    if (!(cx_enabled() & T_XS_AVX)) SKIP("GetEnabledXStateFeatures lacks AVX (reported by ctx_api_reports_usable_features)"); } while (0)
#define CX_NEED_ZMM() do { CX_NEED_XAPI(); \
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable (CPUID/XCR0)"); \
    if ((cx_enabled() & T_XS_ZMM_STATE) != T_XS_ZMM_STATE) SKIP("GetEnabledXStateFeatures lacks AVX-512 state"); } while (0)

/* ======================================================================================================
 * Context accessors (x64 CONTEXT / x86 CONTEXT == WOW64_CONTEXT layout)
 * ====================================================================================================== */
static const char *cx_gpr_name(int i) {
#ifdef T_X64
    static const char *n[] = { "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "r8", "r9", "r10", "r12", "r13", "r14", "r15" };
#else
    static const char *n[] = { "ebx", "edx", "esi", "edi", "ebp" };
#endif
    return n[i];
}
static uint64_t *cx_gpr_ptr64(CONTEXT *c, int i) {
#ifdef T_X64
    switch (i) {
    case 0: return &c->Rbx; case 1: return &c->Rcx; case 2: return &c->Rdx; case 3: return &c->Rsi;
    case 4: return &c->Rdi; case 5: return &c->Rbp; case 6: return &c->R8; case 7: return &c->R9;
    case 8: return &c->R10; case 9: return &c->R12; case 10: return &c->R13; case 11: return &c->R14;
    default: return &c->R15;
    }
#else
    (void)c; (void)i; return NULL;
#endif
}
static uint64_t cx_gpr(const CONTEXT *c, int i) {
#ifdef T_X64
    return *cx_gpr_ptr64((CONTEXT *)c, i);
#else
    switch (i) { case 0: return c->Ebx; case 1: return c->Edx; case 2: return c->Esi; case 3: return c->Edi; default: return c->Ebp; }
#endif
}
static void cx_set_gpr(CONTEXT *c, int i, uint64_t v) {
#ifdef T_X64
    *cx_gpr_ptr64(c, i) = v;
#else
    switch (i) { case 0: c->Ebx = (DWORD)v; break; case 1: c->Edx = (DWORD)v; break; case 2: c->Esi = (DWORD)v; break;
                 case 3: c->Edi = (DWORD)v; break; default: c->Ebp = (DWORD)v; break; }
#endif
}
/* x86: EBP belongs to CONTEXT_CONTROL, every other listed GPR to CONTEXT_INTEGER */
static int cx_gpr_is_control(int i) {
#ifdef T_X64
    (void)i; return 0;
#else
    return i == 4;
#endif
}
static uint64_t cx_gpr_expect(const cx_args *a, int i) {
#ifdef T_X64
    return a->gin[i];
#else
    return (uint32_t)a->gin[i];
#endif
}
static uintptr_t cx_ip(const CONTEXT *c) {
#ifdef T_X64
    return (uintptr_t)c->Rip;
#else
    return c->Eip;
#endif
}
static void cx_set_ip(CONTEXT *c, uintptr_t v) {
#ifdef T_X64
    c->Rip = v;
#else
    c->Eip = (DWORD)v;
#endif
}
static uintptr_t cx_sp(const CONTEXT *c) {
#ifdef T_X64
    return (uintptr_t)c->Rsp;
#else
    return c->Esp;
#endif
}
/* FXSAVE-format legacy area: x64 FltSave, x86 ExtendedRegisters */
static uint8_t *cx_fx(CONTEXT *c) {
#ifdef T_X64
    return (uint8_t *)&c->FltSave;
#else
    return c->ExtendedRegisters;
#endif
}
static uint8_t *cx_xmm(CONTEXT *c, int i) { return cx_fx(c) + 160 + 16 * i; }
static uint32_t cx_mxcsr(const CONTEXT *c) {
#ifdef T_X64
    return c->MxCsr;
#else
    uint32_t v; memcpy(&v, c->ExtendedRegisters + 24, 4); return v;
#endif
}
static uint32_t cx_fx_mxcsr(CONTEXT *c) { uint32_t v; memcpy(&v, cx_fx(c) + 24, 4); return v; }
static void cx_set_mxcsr(CONTEXT *c, uint32_t v) {
    memcpy(cx_fx(c) + 24, &v, 4);
#ifdef T_X64
    c->MxCsr = v;
#endif
}
static uint16_t cx_fcw(CONTEXT *c) {
#ifdef T_X64
    return c->FltSave.ControlWord;
#else
    return (uint16_t)c->FloatSave.ControlWord;
#endif
}
static void cx_set_fcw(CONTEXT *c, uint16_t v) {
    memcpy(cx_fx(c), &v, 2);
#ifdef T_X86
    c->FloatSave.ControlWord = (c->FloatSave.ControlWord & 0xFFFF0000u) | v;
#endif
}
/* ST(i) as seen through the CONTEXT_FLOATING_POINT view */
static uint8_t *cx_st(CONTEXT *c, int i) {
#ifdef T_X64
    return (uint8_t *)&c->FltSave.FloatRegisters[i];
#else
    return c->FloatSave.RegisterArea + 10 * i;
#endif
}
static void cx_set_st(CONTEXT *c, int i, const t_f80 *v) {
    memcpy(cx_fx(c) + 32 + 16 * i, v->b, 10);
#ifdef T_X86
    memcpy(c->FloatSave.RegisterArea + 10 * i, v->b, 10);
#endif
}

static uint16_t cx_own_cs(void) { uint16_t v; __asm__ volatile("mov %%cs, %0" : "=r"(v)); return v; }
static uint16_t cx_own_ss(void) { uint16_t v; __asm__ volatile("mov %%ss, %0" : "=r"(v)); return v; }
static uint16_t cx_own_ds(void) { uint16_t v; __asm__ volatile("mov %%ds, %0" : "=r"(v)); return v; }
static uint16_t cx_own_es(void) { uint16_t v; __asm__ volatile("mov %%es, %0" : "=r"(v)); return v; }
static uint16_t cx_own_fs(void) { uint16_t v; __asm__ volatile("mov %%fs, %0" : "=r"(v)); return v; }
static uint16_t cx_own_gs(void) { uint16_t v; __asm__ volatile("mov %%gs, %0" : "=r"(v)); return v; }
/* MXCSR_MASK from FXSAVE (0 means the architectural default 0xFFBF) */
static uint32_t cx_mxcsr_mask(void) {
    T_ALIGN(16) uint8_t fx[512];
    memset(fx, 0, sizeof fx);
    t_fxsave(fx);
    uint32_t m; memcpy(&m, fx + 28, 4);
    return m ? m : 0xFFBFu;
}

/* ======================================================================================================
 * Buffers
 * ====================================================================================================== */
static void *cx_alloc(size_t n) {
    void *p = VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return p;
}
static void cx_free(void *p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }
static CONTEXT *cx_plain(DWORD flags) {
    CONTEXT *c = (CONTEXT *)cx_alloc(sizeof(CONTEXT) + 64);
    if (c) { memset(c, CX_SENT, sizeof *c); c->ContextFlags = flags; }
    return c;
}
static int cx_all_sent(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) if (b[i] != CX_SENT) return 0;
    return 1;
}
static int cx_all_zero(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) if (b[i]) return 0;
    return 1;
}

/* XSTATE-aware CONTEXT from InitializeContext (two-phase size negotiation as documented) */
typedef struct cx_xc { void *buf; CONTEXT *ctx; DWORD len; DWORD flags; DWORD err; } cx_xc;
static int cx_xc_init(cx_xc *x, DWORD flags) {
    memset(x, 0, sizeof *x);
    x->flags = flags;
    DWORD len = 0;
    SetLastError(0);
    BOOL ok = X.InitializeContext(NULL, flags, NULL, &len);
    if (ok || GetLastError() != ERROR_INSUFFICIENT_BUFFER || !len) { x->err = ok ? 0xFFFFFFFFu : GetLastError(); return 0; }
    x->buf = cx_alloc(len);
    if (!x->buf) { x->err = ERROR_OUTOFMEMORY; return 0; }
    memset(x->buf, CX_SENT, len);
    x->len = len;
    if (!X.InitializeContext(x->buf, flags, &x->ctx, &x->len)) { x->err = GetLastError(); cx_free(x->buf); x->buf = NULL; return 0; }
    return 1;
}
static void cx_xc_free(cx_xc *x) { cx_free(x->buf); x->buf = NULL; x->ctx = NULL; }
/* Init + REQUIRE (case thread only, before any worker exists) */
#define CX_XC_REQUIRE(x, flags) REQUIREF(cx_xc_init((x), (flags)), "InitializeContext(0x%lx) failed, GLE=%lu", (unsigned long)(flags), (unsigned long)(x)->err)

/* ======================================================================================================
 * Patterns
 * ====================================================================================================== */
static void cx_f80_make(t_f80 *f, uint32_t tag, int i) {
    uint64_t m = t_pattern_word(tag, 64 + (uint32_t)i, 0) | (1ull << 63);   /* explicit integer bit: normal */
    uint16_t e = (uint16_t)(0x3FF0 + i + (tag & 7));                        /* positive, normal exponent */
    memcpy(f->b, &m, 8); memcpy(f->b + 8, &e, 2);
}
static void cx_args_fill(cx_args *a, uint32_t tag, int x87) {
    memset(a, 0, sizeof *a);
    for (int i = 0; i < 16; i++) t_pattern_reg(a->vin[i], 32, tag, (uint32_t)i);
    for (int i = 0; i < 32; i++) t_pattern_reg(a->zin[i], 64, tag, (uint32_t)i);
    for (int i = 0; i < 16; i++) a->gin[i] = t_pattern_word(tag, 40 + (uint32_t)i, 0);
    for (int i = 0; i < 8; i++) {
        cx_f80_make(&a->fin[i], tag, i);
        a->kin[i] = t_pattern_word(tag, 100 + (uint32_t)i, 0);
        if (!cpu_has(F_AVX512BW)) a->kin[i] &= 0xFFFF;
    }
    a->mxcsr_in = 0x1F80;
    a->fcw_in = 0x027F;
    a->flags = (x87 ? CXF_X87 : 0) | (cpu_has(F_AVX512BW) ? CXF_KBW : 0);
    a->max_loops = 0x7FFFFFFFu;     /* bounded even if the parent never sets stop */
}
static cx_args *cx_args_new(uint32_t tag, int x87) {
    cx_args *a = (cx_args *)cx_alloc(sizeof(cx_args));
    if (a) cx_args_fill(a, tag, x87);
    return a;
}

/* ======================================================================================================
 * Worker threads
 * ====================================================================================================== */
typedef struct cx_worker {
    HANDLE   h;
    DWORD    tid;
    cx_args *a;
    int      kind;       /* T_VK_XMM / T_VK_YMM / T_VK_ZMM */
    int      done;
} cx_worker;

#define CX_MAX_LIVE 8
static cx_args *volatile g_live[CX_MAX_LIVE];      /* workers still spinning, with the case that started them */
static const char *g_live_case[CX_MAX_LIVE];

static __attribute__((target("avx"))) void cx_vzeroupper_avx(void) { __asm__ volatile("vzeroupper" ::: "memory"); }

static DWORD WINAPI cx_worker_main(void *p) {
    cx_worker *w = (cx_worker *)p;
    if ((w->a->flags & CXF_VZU) && cpu_has(F_AVX)) cx_vzeroupper_avx();
    if (w->kind == T_VK_ZMM) cx_spin_zmm(w->a);
    else if (w->kind == T_VK_YMM) cx_spin_ymm(w->a);
    else cx_spin_xmm(w->a);
    return 0;
}
static const char *cx_loop_lo(int kind) { return kind == T_VK_ZMM ? cx_zmm_loop : kind == T_VK_YMM ? cx_ymm_loop : cx_xmm_loop; }
static const char *cx_loop_hi(int kind) { return kind == T_VK_ZMM ? cx_zmm_exit : kind == T_VK_YMM ? cx_ymm_exit : cx_xmm_exit; }

/* Start a worker on a pre-filled cx_args and wait until it holds its registers (spinning).
 * Returns 1 on success. Stale workers left by an earlier, aborted case are stopped first (their cx_args is
 * never freed on the abort path, so the write is safe); workers of the current case are left running. */
static int cx_worker_start(cx_worker *w, cx_args *a, int kind) {
    for (int i = 0; i < CX_MAX_LIVE; i++)
        if (g_live[i] && g_live_case[i] != t_case_name()) { g_live[i]->stop = 1; g_live[i] = NULL; }
    memset(w, 0, sizeof *w);
    w->a = a; w->kind = kind;
    a->ready = 0; a->stop = 0;
    w->h = CreateThread(NULL, 0, cx_worker_main, w, 0, &w->tid);
    if (!w->h) return 0;
    for (int i = 0; i < CX_MAX_LIVE; i++) if (!g_live[i]) { g_live[i] = a; g_live_case[i] = t_case_name(); break; }
    double t0 = t_now_ms();
    while (!a->ready && t_now_ms() - t0 < CX_TMO) Sleep(1);
    return a->ready != 0;
}
/* Stop: set the flag, resume fully, wait. Returns 1 if the thread exited in time. */
static int cx_worker_finish(cx_worker *w) {
    if (!w->h) return 0;
    w->a->stop = 1;
    for (int i = 0; i < 64; i++) { DWORD r = ResumeThread(w->h); if (r == (DWORD)-1 || r <= 1) break; }
    DWORD wr = WaitForSingleObject(w->h, CX_TMO);
    CloseHandle(w->h);
    w->h = NULL;
    for (int i = 0; i < CX_MAX_LIVE; i++) if (g_live[i] == w->a) g_live[i] = NULL;
    w->done = (wr == WAIT_OBJECT_0);
    return w->done;
}
/* Suspend and check the documented previous-count result. */
static int cx_suspend(cx_worker *w) {
    DWORD r = SuspendThread(w->h);
    CHECKF(r == 0, "SuspendThread previous count %lu (GLE=%lu)", (unsigned long)r, (unsigned long)GetLastError());
    return r == 0;
}
/* Resume, set stop and wait, then verify the worker ran to the stop flag (not to max_loops). */
static int cx_release(cx_worker *w) {
    int ok = cx_worker_finish(w);
    CHECKF(ok, "worker did not exit within %u ms after stop", CX_TMO);
    if (ok) CHECKF(w->a->remaining != 0, "worker exhausted its loop budget: stop flag never observed");
    return ok;
}

/* ======================================================================================================
 * Common verification
 * ====================================================================================================== */
static void cx_check_gprs(const CONTEXT *c, const cx_args *a, int with_control, const char *what) {
    for (int i = 0; i < CX_NGPR; i++) {
        if (cx_gpr_is_control(i) && !with_control) continue;
        uint64_t got = cx_gpr(c, i), want = cx_gpr_expect(a, i);
        CHECKF(got == want, "%s: %s = 0x%llx want 0x%llx", what, cx_gpr_name(i),
               (unsigned long long)got, (unsigned long long)want);
    }
}
static int cx_gprs_untouched(const CONTEXT *c, int include_control) {
    for (int i = 0; i < CX_NGPR; i++) {
        if (cx_gpr_is_control(i) && !include_control) continue;
        uint64_t g = cx_gpr(c, i);
        if (!cx_all_sent(&g, sizeof(uintptr_t))) return 0;
    }
    return 1;
}
static void cx_check_xmm(CONTEXT *c, uint8_t v[][32], int n, const char *what) {
    char lab[96];
    for (int i = 0; i < n; i++) {
        snprintf(lab, sizeof lab, "xmm%d %s", i, what);
        CHECK_REG(lab, cx_xmm(c, i), v[i], 16);
    }
}
static void cx_check_control(const CONTEXT *c, const cx_args *a, int kind, const char *what) {
    uintptr_t ip = cx_ip(c), lo = (uintptr_t)cx_loop_lo(kind), hi = (uintptr_t)cx_loop_hi(kind);
    CHECKF(ip >= lo && ip <= hi, "%s: IP %p outside spin loop [%p, %p]", what, (void *)ip, (void *)lo, (void *)hi);
    CHECKF(cx_sp(c) == (uintptr_t)a->sp, "%s: SP %p want %p", what, (void *)cx_sp(c), (void *)(uintptr_t)a->sp);
    CHECKF(c->SegCs == cx_own_cs(), "%s: SegCs 0x%x want 0x%x", what, (unsigned)c->SegCs, (unsigned)cx_own_cs());
    CHECKF(c->SegSs == cx_own_ss(), "%s: SegSs 0x%x want 0x%x", what, (unsigned)c->SegSs, (unsigned)cx_own_ss());
    CHECKF(c->EFlags & 0x2, "%s: EFlags 0x%lx reserved bit 1 clear", what, (unsigned long)c->EFlags);
    CHECKF(c->EFlags & 0x200, "%s: EFlags 0x%lx IF clear in user mode", what, (unsigned long)c->EFlags);
    CHECKF(!(c->EFlags & 0x100), "%s: EFlags 0x%lx TF set without a debugger", what, (unsigned long)c->EFlags);
}
static void cx_check_x87(CONTEXT *c, const cx_args *a, const char *what) {
    char lab[96];
    for (int i = 0; i < 8; i++) {
        snprintf(lab, sizeof lab, "ST(%d) %s", i, what);
        CHECK_REG(lab, cx_st(c, i), a->fin[i].b, 10);
        snprintf(lab, sizeof lab, "FXSAVE ST(%d) %s", i, what);
        CHECK_REG(lab, cx_fx(c) + 32 + 16 * i, a->fin[i].b, 10);
    }
}

/* Get the full standard context of a suspended worker into a fresh plain CONTEXT */
static CONTEXT *cx_get_std(cx_worker *w) {
    CONTEXT *c = cx_plain(CX_STD);
    if (!c) return NULL;
    if (!GetThreadContext(w->h, c)) {
        CHECKF(0, "GetThreadContext(CX_STD) failed, GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
        return NULL;
    }
    return c;
}

/* XSTATE context for a worker: InitializeContext, request the given features, get. */
static int cx_get_x(cx_worker *w, cx_xc *x, DWORD flags, uint64_t req_mask) {
    if (!cx_xc_init(x, flags)) { CHECKF(0, "InitializeContext(0x%lx) GLE=%lu", (unsigned long)flags, (unsigned long)x->err); return 0; }
    if (!X.SetXStateFeaturesMask(x->ctx, req_mask)) {
        CHECKF(0, "SetXStateFeaturesMask(0x%llx) GLE=%lu", (unsigned long long)req_mask, (unsigned long)GetLastError());
        return 0;
    }
    if (!GetThreadContext(w->h, x->ctx)) {
        CHECKF(0, "GetThreadContext(CONTEXT_XSTATE) failed, GLE=%lu", (unsigned long)GetLastError());
        return 0;
    }
    return 1;
}
static uint8_t *cx_locate(CONTEXT *c, DWORD id, DWORD *len) {
    DWORD l = 0xDEADBEEF;
    uint8_t *p = (uint8_t *)X.LocateXStateFeature(c, id, &l);
    if (len) *len = l;
    return p;
}
/* XSAVE header of an InitializeContext context: CONTEXT_EX directly follows the legacy CONTEXT
 * (All, Legacy, XState {LONG Offset; DWORD Length}, offsets relative to the CONTEXT_EX) */
static uint64_t *cx_xs_hdr(CONTEXT *c) {
    LONG *ex = (LONG *)((uint8_t *)c + sizeof(CONTEXT));
    return (uint64_t *)((uint8_t *)ex + ex[4]);
}
static uint64_t cx_mask(CONTEXT *c) {
    DWORD64 m = 0xDEAD;
    if (!X.GetXStateFeaturesMask(c, &m)) { CHECKF(0, "GetXStateFeaturesMask GLE=%lu", (unsigned long)GetLastError()); return 0; }
    return m;
}

/* ======================================================================================================
 * 1. ABI layout and API presence / reporting
 * ====================================================================================================== */
TEST(ctx_layout_abi) {
    /* CONTEXT layouts are fixed by the Windows ABI (winnt.h); the decompiled kernelbase/ntdll code above
     * hard-codes the same offsets (0x30 flags, 0x100 FltSave, 0x1a0 Xmm0, 0x4d0 size; x86 0xcc / 0x2cc). */
#ifdef T_X64
    CHECK_EQ(offsetof(CONTEXT, ContextFlags), 0x30);
    CHECK_EQ(offsetof(CONTEXT, MxCsr), 0x34);
    CHECK_EQ(offsetof(CONTEXT, SegCs), 0x38);
    CHECK_EQ(offsetof(CONTEXT, EFlags), 0x44);
    CHECK_EQ(offsetof(CONTEXT, Dr0), 0x48);
    CHECK_EQ(offsetof(CONTEXT, Dr7), 0x70);
    CHECK_EQ(offsetof(CONTEXT, Rax), 0x78);
    CHECK_EQ(offsetof(CONTEXT, Rsp), 0x98);
    CHECK_EQ(offsetof(CONTEXT, Rip), 0xF8);
    CHECK_EQ(offsetof(CONTEXT, FltSave), 0x100);
    CHECK_EQ(offsetof(CONTEXT, Xmm0), 0x1A0);
    CHECK_EQ(offsetof(CONTEXT, VectorRegister), 0x300);
    CHECK_EQ(offsetof(CONTEXT, DebugControl), 0x4A8);
    CHECK_EQ(sizeof(CONTEXT), 0x4D0);
    CHECK_EQ(sizeof(XMM_SAVE_AREA32), 512);
    CHECK_EQ(offsetof(XMM_SAVE_AREA32, MxCsr), 24);
    CHECK_EQ(offsetof(XMM_SAVE_AREA32, FloatRegisters), 32);
    CHECK_EQ(offsetof(XMM_SAVE_AREA32, XmmRegisters), 160);
    CHECK_EQ(sizeof(WOW64_CONTEXT), 0x2CC);
#else
    CHECK_EQ(offsetof(CONTEXT, ContextFlags), 0);
    CHECK_EQ(offsetof(CONTEXT, FloatSave), 0x1C);
    CHECK_EQ(offsetof(CONTEXT, Edi), 0x9C);
    CHECK_EQ(offsetof(CONTEXT, Ebp), 0xB4);
    CHECK_EQ(offsetof(CONTEXT, Eip), 0xB8);
    CHECK_EQ(offsetof(CONTEXT, Esp), 0xC4);
    CHECK_EQ(offsetof(CONTEXT, ExtendedRegisters), 0xCC);
    CHECK_EQ(sizeof(CONTEXT), 0x2CC);
    CHECK_EQ(sizeof(FLOATING_SAVE_AREA), 112);
#endif
    CHECK_EQ(CONTEXT_XSTATE & 0xFF, 0x40);
    CHECK_EQ(CONTEXT_XSTATE & ~0xFFul, CX_ARCHBIT);
}

TEST(ctx_api_exports) {
    cx_resolve();
    t_info("InitializeContext", "%s", X.InitializeContext ? "yes" : "no");
    t_info("InitializeContext2", "%s", X.InitializeContext2 ? "yes" : "no");
    t_info("LocateXStateFeature", "%s", X.LocateXStateFeature ? "yes" : "no");
    t_info("CopyContext", "%s", X.CopyContext ? "yes" : "no");
    t_info("Wow64GetThreadContext", "%s", X.Wow64GetThreadContext ? "yes" : "no");
    CHECKF(X.RtlCaptureContext != NULL, "RtlCaptureContext not exported (kernel32/ntdll)");
    int n = !!X.InitializeContext + !!X.LocateXStateFeature + !!X.GetXStateFeaturesMask + !!X.SetXStateFeaturesMask +
            !!X.CopyContext + !!X.GetEnabledXStateFeatures;
    if (n == 0) SKIP("no XSTATE context API (pre Windows 7 SP1 API set)");
    /* The Windows 7 SP1 XSTATE API set is shipped together: a partial set breaks documented usage. */
    CHECKF(n == 6, "partial XSTATE API set: %d of 6 exported", n);
    uint64_t e = cx_enabled();
    t_info("enabled_features", "0x%llx", (unsigned long long)e);
    CHECKF((e & 3) == 3, "GetEnabledXStateFeatures 0x%llx lacks x87|SSE (always enabled on x86/x64)", (unsigned long long)e);
}

TEST(ctx_api_reports_usable_features) {
    /* Separate result: instructions work (CPUID+XCR0) but the Windows XSTATE context API does not report the
     * state, so no CONTEXT_XSTATE consumer (debuggers, exception dispatch users) can see it. */
    CX_NEED_XAPI();
    uint64_t e = cx_enabled();
    const t_cpu_info *c = cpu();
    if (!c->xcr0_valid) SKIP("XCR0 not readable (OSXSAVE clear)");
    if (cpu_has(F_AVX)) CHECKF(e & T_XS_AVX, "AVX usable (XCR0 0x%llx) but GetEnabledXStateFeatures 0x%llx lacks bit 2",
                               (unsigned long long)c->xcr0, (unsigned long long)e);
    if (cpu_has(F_AVX512F)) CHECKF((e & T_XS_ZMM_STATE) == T_XS_ZMM_STATE, "AVX-512 usable but API mask 0x%llx lacks 0xE0",
                                   (unsigned long long)e);
    /* Any user component reported by the API must be enabled in XCR0 (else contexts would carry state the
     * thread cannot hold). Supervisor components may appear (GetEnabledXStateFeatures ORs them in). */
    uint64_t user_api = e & c->xcr0_supported;
    CHECKF((user_api & ~c->xcr0) == 0, "API reports user components 0x%llx not enabled in XCR0 0x%llx",
           (unsigned long long)(user_api & ~c->xcr0), (unsigned long long)c->xcr0);
    if (e & ~c->xcr0) t_note("API mask 0x%llx has bits outside XCR0 0x%llx (supervisor/optional: 0x%llx)",
                             (unsigned long long)e, (unsigned long long)c->xcr0, (unsigned long long)(e & ~c->xcr0));
}

/* ======================================================================================================
 * 2. InitializeContext: size negotiation, alignment, validation
 * ====================================================================================================== */
TEST(init_ctx_size_query) {
    CX_NEED_XAPI();
    DWORD len = 0;
    SetLastError(0);
    BOOL ok = X.InitializeContext(NULL, CX_STD, NULL, &len);
    CHECK(!ok);
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECKF(len >= sizeof(CONTEXT), "required length %lu < sizeof(CONTEXT) %u", (unsigned long)len, (unsigned)sizeof(CONTEXT));
    CHECKF(len < sizeof(CONTEXT) + 4096, "required length %lu absurd for a non-XSTATE context", (unsigned long)len);
    t_info("len_std", "%lu", (unsigned long)len);
    /* a too-small non-NULL buffer behaves the same and reports the same size */
    DWORD len2 = 16;
    uint8_t small[64];
    memset(small, CX_SENT, sizeof small);
    CONTEXT *pc = (CONTEXT *)(uintptr_t)0x1234;
    SetLastError(0);
    CHECK(!X.InitializeContext(small, CX_STD, &pc, &len2));
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECK_EQ(len2, len);
    CHECKF(cx_all_sent(small, sizeof small), "InitializeContext wrote into an undersized buffer");
    CHECKF(pc == (CONTEXT *)(uintptr_t)0x1234, "Context out-pointer written on failure");
}

TEST(init_ctx_size_xstate) {
    CX_NEED_XAPI();
    DWORD l_std = 0, l_x = 0;
    X.InitializeContext(NULL, CX_STD, NULL, &l_std);
    SetLastError(0);
    BOOL ok = X.InitializeContext(NULL, CX_STD | CONTEXT_XSTATE, NULL, &l_x);
    CHECK(!ok);
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    t_info("len_xstate", "%lu", (unsigned long)l_x);
    if (!(cx_enabled() & ~3ull)) { t_note("no extended features enabled"); return; }
    CHECKF(l_x > l_std, "XSTATE length %lu not larger than legacy length %lu", (unsigned long)l_x, (unsigned long)l_std);
    if (cx_enabled() & T_XS_AVX)
        CHECKF(l_x >= l_std + 64 + 256, "XSTATE length %lu cannot hold header + AVX (legacy %lu)", (unsigned long)l_x, (unsigned long)l_std);
    if ((cx_enabled() & T_XS_ZMM_STATE) == T_XS_ZMM_STATE)
        CHECKF(l_x >= l_std + 64 + 256 + 64 + 512 + 1024, "XSTATE length %lu cannot hold AVX-512 state", (unsigned long)l_x);
    /* the size must not depend on the non-XSTATE flag bits beyond the legacy part */
    DWORD l_x2 = 0;
    X.InitializeContext(NULL, CONTEXT_CONTROL | CONTEXT_XSTATE, NULL, &l_x2);
    CHECKF(l_x2 >= l_x - 64 && l_x2 <= l_x, "XSTATE length with CONTEXT_CONTROL %lu vs full %lu", (unsigned long)l_x2, (unsigned long)l_x);
}

TEST(init_ctx_undersized_by_one) {
    CX_NEED_XAPI();
    DWORD need = 0;
    X.InitializeContext(NULL, CX_STD | CONTEXT_XSTATE, NULL, &need);
    REQUIRE(need > 1);
    uint8_t *b = (uint8_t *)cx_alloc(need + 64);
    REQUIRE(b);
    memset(b, CX_SENT, need + 64);
    DWORD len = need - 1;
    CONTEXT *pc = NULL;
    SetLastError(0);
    CHECK(!X.InitializeContext(b, CX_STD | CONTEXT_XSTATE, &pc, &len));
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECK_EQ(len, need);
    CHECK(pc == NULL);
    CHECKF(cx_all_sent(b, need + 64), "undersized InitializeContext modified the buffer");
    /* exact size succeeds */
    len = need;
    CHECK(X.InitializeContext(b, CX_STD | CONTEXT_XSTATE, &pc, &len));
    CHECK(pc != NULL);
    cx_free(b);
}

TEST(init_ctx_alignment_sweep_guarded) {
    /* Buffer = exactly the required length, ending at a PAGE_NOACCESS page, at every start alignment 0..63:
     * InitializeContext must align the CONTEXT itself inside the buffer and never touch the guard page. */
    CX_NEED_XAPI();
    DWORD flags[2] = { CX_STD, CX_STD | CONTEXT_XSTATE };
    for (int f = 0; f < 2; f++) {
        DWORD need = 0;
        X.InitializeContext(NULL, flags[f], NULL, &need);
        REQUIRE(need >= sizeof(CONTEXT));
        int bad = 0;
        for (int mis = 0; mis < 64; mis++) {
            void *base = NULL;
            uint8_t *g = (uint8_t *)t_guarded_alloc(need + 64, &base);
            REQUIRE(g);
            /* start alignment varies over all 64 residues; the mis bytes between the buffer end and the
             * guard page are sentinel-checked, mis == 0 ends exactly at the guard page */
            uint8_t *end = g + need + 64;
            uint8_t *b = end - need - mis;
            memset(g, CX_SENT, need + 64);
            DWORD len = need;
            CONTEXT *pc = NULL;
            BOOL ok = X.InitializeContext(b, flags[f], &pc, &len);
            if (!ok) { CHECKF(0, "flags 0x%lx misalign %d: InitializeContext failed GLE=%lu", (unsigned long)flags[f], mis, (unsigned long)GetLastError()); bad++; }
            else {
                CHECKF(((uintptr_t)pc % CX_CTX_ALIGN) == 0, "CONTEXT %p not %d-aligned", (void *)pc, CX_CTX_ALIGN);
                CHECKF((uint8_t *)pc >= b && (uint8_t *)pc + sizeof(CONTEXT) <= b + need, "CONTEXT %p outside buffer [%p,+%lu)",
                       (void *)pc, (void *)b, (unsigned long)need);
                CHECK_EQ(pc->ContextFlags, flags[f]);
                if (flags[f] & 0x40 & ~CX_ARCHBIT) {
                    for (DWORD id = 2; id < 64; id++) {
                        if (!(cx_enabled_user() & (1ull << id))) continue;
                        DWORD fl = 0;
                        uint8_t *p = cx_locate(pc, id, &fl);
                        CHECKF(p != NULL, "misalign %d: LocateXStateFeature(%lu) NULL", mis, (unsigned long)id);
                        if (p) {
                            CHECKF(p >= (uint8_t *)pc + sizeof(CONTEXT) && p + fl <= b + need,
                                   "misalign %d: feature %lu [%p,+%lu) outside buffer end %p", mis, (unsigned long)id, (void *)p,
                                   (unsigned long)fl, (void *)(b + need));
                            CHECKF(((uintptr_t)p & 15) == 0, "feature %lu at %p not 16-aligned", (unsigned long)id, (void *)p);
                        }
                    }
                }
            }
            CHECKF(cx_all_sent(b + need, (size_t)mis), "misalign %d: InitializeContext wrote past the buffer end", mis);
            t_guarded_free(base);
            if (bad > 3) break;
        }
    }
}

TEST(init_ctx_flags_validation) {
    /* RtlpValidateContextFlags (ntdll 19041): one architecture bit, no foreign bits. A NULL buffer
     * distinguishes "valid flags" (ERROR_INSUFFICIENT_BUFFER) from "invalid" (ERROR_INVALID_PARAMETER). */
    CX_NEED_XAPI();
    static const struct { DWORD f; int valid; const char *what; } cases[] = {
        { 0, 0, "zero" },
        { CONTEXT_CONTROL & 0xFF, 0, "control bit without architecture bit" },
        { CX_ARCHBIT, 1, "architecture bit only" },
        { CX_STD, 1, "standard flags" },
        { CX_ARCHBIT | 0x1F, 1, "all legacy parts" },
        { 0x00010000 | 0x00100000 | 0x1, 0, "both i386 and AMD64 bits" },
#ifdef T_X64
        { CONTEXT_AMD64 | 0x20, 0, "AMD64 with i386 CONTEXT_EXTENDED_REGISTERS bit" },
        { CONTEXT_AMD64 | 0x80, 0, "AMD64 with undefined bit 7" },
        { 0x00010000 | 0x1, 1, "i386 flags (valid for the x86 layout)" },
#else
        { CONTEXT_i386 | 0x80, 0, "i386 with undefined bit 7" },
        { CONTEXT_i386 | 0x100, 0, "i386 with undefined bit 8" },
#endif
    };
    int any_wrong = 0;
    for (size_t i = 0; i < T_COUNTOF(cases); i++) {
        DWORD len = 0;
        SetLastError(0);
        BOOL ok = X.InitializeContext(NULL, cases[i].f, NULL, &len);
        DWORD e = GetLastError();
        CHECKF(!ok, "flags 0x%lx (%s): NULL buffer accepted", (unsigned long)cases[i].f, cases[i].what);
        DWORD want = cases[i].valid ? ERROR_INSUFFICIENT_BUFFER : ERROR_INVALID_PARAMETER;
        if (e != want) any_wrong++;
        CHECKF(e == want, "flags 0x%lx (%s): GLE %lu want %lu", (unsigned long)cases[i].f, cases[i].what,
               (unsigned long)e, (unsigned long)want);
    }
    (void)any_wrong;
}

TEST(init_ctx_contents) {
    CX_NEED_XAPI();
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    CHECK_EQ(x.ctx->ContextFlags, CX_STD | CONTEXT_XSTATE);
    CHECKF(((uintptr_t)x.ctx % CX_CTX_ALIGN) == 0, "context %p misaligned", (void *)x.ctx);
    CHECKF((uint8_t *)x.ctx >= (uint8_t *)x.buf && (uint8_t *)x.ctx < (uint8_t *)x.buf + 64, "context %p far from buffer %p",
           (void *)x.ctx, x.buf);
    uint64_t m = cx_mask(x.ctx);
    /* fresh XSTATE header: no extended feature present; legacy bits follow the FP flag */
    CHECKF((m & ~3ull) == 0, "fresh context feature mask 0x%llx has extended features", (unsigned long long)m);
    CHECKF((m & 3) == 3, "mask 0x%llx lacks legacy bits although the FP/extended-registers flag is set", (unsigned long long)m);
    cx_xc_free(&x);
    /* without the FP flag the legacy bits are clear */
    CX_XC_REQUIRE(&x, CONTEXT_CONTROL | CONTEXT_XSTATE);
    m = cx_mask(x.ctx);
    CHECKF(m == 0, "CONTROL|XSTATE fresh mask 0x%llx want 0", (unsigned long long)m);
    cx_xc_free(&x);
}

TEST(init_ctx2_compaction_mask) {
    CX_NEED_XAPI();
    if (!X.InitializeContext2) SKIP("InitializeContext2 not exported (Windows 10 2004+)");
    DWORD l_all = 0, l_avx = 0, l_legacy = 0;
    SetLastError(0);
    CHECK(!X.InitializeContext2(NULL, CX_STD | CONTEXT_XSTATE, NULL, &l_all, ~0ull));
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    X.InitializeContext2(NULL, CX_STD | CONTEXT_XSTATE, NULL, &l_avx, T_XS_YMM_STATE);
    X.InitializeContext2(NULL, CX_STD | CONTEXT_XSTATE, NULL, &l_legacy, 3);
    t_info("len2_all", "%lu", (unsigned long)l_all);
    t_info("len2_avx", "%lu", (unsigned long)l_avx);
    t_info("len2_legacy", "%lu", (unsigned long)l_legacy);
    CHECKF(l_avx <= l_all && l_legacy <= l_avx, "lengths not monotonic in the mask: legacy %lu avx %lu all %lu",
           (unsigned long)l_legacy, (unsigned long)l_avx, (unsigned long)l_all);
    DWORD l1 = 0;
    X.InitializeContext(NULL, CX_STD | CONTEXT_XSTATE, NULL, &l1);
    CHECKF(l1 == l_all, "InitializeContext %lu != InitializeContext2(~0) %lu", (unsigned long)l1, (unsigned long)l_all);
    if (!cx_avx_ok()) return;
    uint8_t *b = (uint8_t *)cx_alloc(l_avx);
    REQUIRE(b);
    CONTEXT *pc = NULL;
    DWORD len = l_avx;
    CHECK(X.InitializeContext2(b, CX_STD | CONTEXT_XSTATE, &pc, &len, T_XS_YMM_STATE));
    if (pc) {
        DWORD fl = 0;
        uint8_t *p = cx_locate(pc, CX_XS_AVX, &fl);
        CHECKF(p != NULL && p + fl <= b + l_avx, "AVX area %p+%lu outside the AVX-only buffer", (void *)p, (unsigned long)fl);
    }
    cx_free(b);
}

/* ======================================================================================================
 * 3. LocateXStateFeature / Get/SetXStateFeaturesMask on initialized buffers
 * ====================================================================================================== */
TEST(locate_legacy_features) {
    CX_NEED_XAPI();
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    DWORD l0 = 0, l1 = 0;
    uint8_t *p0 = cx_locate(x.ctx, 0, &l0), *p1 = cx_locate(x.ctx, 1, &l1);
    CHECKF(p0 == cx_fx(x.ctx), "feature 0 at %p want legacy FXSAVE area %p", (void *)p0, (void *)cx_fx(x.ctx));
    CHECKF(p1 == cx_xmm(x.ctx, 0), "feature 1 at %p want XMM0 %p", (void *)p1, (void *)cx_xmm(x.ctx, 0));
    CHECK_EQ(l0, 0xA0);
    CHECK_EQ(l1, CX_SSE_LEN);
    /* NULL length pointer is allowed */
    CHECK(X.LocateXStateFeature(x.ctx, 1, NULL) == p1);
    cx_xc_free(&x);
    /* legacy features are located without CONTEXT_XSTATE as well */
    CX_XC_REQUIRE(&x, CX_STD);
    CHECK(cx_locate(x.ctx, 0, NULL) == cx_fx(x.ctx));
    CHECK(cx_locate(x.ctx, 1, NULL) == cx_xmm(x.ctx, 0));
    cx_xc_free(&x);
}

TEST(locate_avx_feature) {
    CX_NEED_AVX();
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    DWORD l = 0;
    uint8_t *p = cx_locate(x.ctx, CX_XS_AVX, &l);
    CHECK(p != NULL);
    CHECK_EQ(l, cpu()->comp[2].size ? cpu()->comp[2].size : 256);
    CHECK_EQ(l, 256);
    CHECKF(p >= (uint8_t *)x.ctx + sizeof(CONTEXT) && p + l <= (uint8_t *)x.buf + x.len, "AVX area outside the buffer");
    cx_xc_free(&x);
}

TEST(locate_enabled_and_unsupported_features) {
    CX_NEED_XAPI();
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    uint64_t en = cx_enabled(), user = cx_enabled_user();
    uint8_t *prev_end = (uint8_t *)x.ctx + sizeof(CONTEXT);
    for (DWORD id = 2; id < 64; id++) {
        DWORD l = 0;
        uint8_t *p = cx_locate(x.ctx, id, &l);
        if (!(en & (1ull << id))) {
            CX_CHECK_WGAP("LocateXStateFeature locates features that are not enabled", p == NULL,
                          "feature %lu not enabled (0x%llx) but located at %p", (unsigned long)id, (unsigned long long)en, (void *)p);
        } else if (user & (1ull << id)) {
            CHECKF(p != NULL, "enabled user feature %lu not located", (unsigned long)id);
            if (p) {
                CHECKF(p + l <= (uint8_t *)x.buf + x.len, "feature %lu [%p,+%lu) beyond buffer", (unsigned long)id, (void *)p, (unsigned long)l);
                CHECKF(p >= prev_end || id == 2, "feature %lu at %p overlaps the previous feature (end %p)", (unsigned long)id, (void *)p, (void *)prev_end);
                if (cpu()->comp[id].size) CHECKF(l == cpu()->comp[id].size, "feature %lu length %lu, CPUID.(0Dh,%lu).EAX %u",
                                                (unsigned long)id, (unsigned long)l, (unsigned long)id, cpu()->comp[id].size);
                prev_end = p + l;
            }
        } else {
            t_note("supervisor feature %lu located at %p len %lu", (unsigned long)id, (void *)p, (unsigned long)l);
        }
    }
    CHECK(cx_locate(x.ctx, 64, NULL) == NULL);
    CHECK(cx_locate(x.ctx, 0xFFFFFFFFu, NULL) == NULL);
    CHECK(cx_locate(x.ctx, 0x80000002u, NULL) == NULL);
    cx_xc_free(&x);
}

TEST(locate_without_xstate_flag) {
    CX_NEED_XAPI();
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD);
    for (DWORD id = 2; id < 64; id++)
        CHECKF(cx_locate(x.ctx, id, NULL) == NULL, "feature %lu located in a context without CONTEXT_XSTATE", (unsigned long)id);
    cx_xc_free(&x);
}

TEST(set_mask_basic) {
    CX_NEED_AVX();
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    CHECK(X.SetXStateFeaturesMask(x.ctx, T_XS_AVX));
    CHECK_EQ(cx_mask(x.ctx), 3 | T_XS_AVX);
    CHECK(X.SetXStateFeaturesMask(x.ctx, 0));
    CHECK_EQ(cx_mask(x.ctx), 3);
    CHECK(X.SetXStateFeaturesMask(x.ctx, T_XS_YMM_STATE | T_XS_X87));
    CHECK_EQ(cx_mask(x.ctx), 3 | T_XS_AVX);
    cx_xc_free(&x);
}

TEST(set_mask_filters_unsupported_bits) {
    CX_NEED_XAPI();
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    uint64_t en = cx_enabled();
    SetLastError(0);
    BOOL ok = X.SetXStateFeaturesMask(x.ctx, ~0ull);
    if (!ok) t_note("SetXStateFeaturesMask(~0) rejected, GLE=%lu", (unsigned long)GetLastError());
    else {
        uint64_t m = cx_mask(x.ctx);
        CHECKF((m & ~(en | 3)) == 0, "mask 0x%llx has bits outside enabled features 0x%llx", (unsigned long long)m, (unsigned long long)en);
        t_info("mask_after_all_ones", "0x%llx", (unsigned long long)m);
    }
    uint64_t missing = ~en & ~3ull & 0x7FFFFFFFFFFFFFFFull;
    if (missing) {
        uint64_t bit = missing & (0 - missing);
        ok = X.SetXStateFeaturesMask(x.ctx, bit | T_XS_AVX);
        if (ok) {
            uint64_t m = cx_mask(x.ctx);
            CHECKF(!(m & bit), "unsupported feature bit 0x%llx stored in the header (mask 0x%llx)", (unsigned long long)bit, (unsigned long long)m);
        }
    }
    cx_xc_free(&x);
}

TEST(set_mask_without_xstate_chunk) {
    CX_NEED_XAPI();
    CONTEXT *c = cx_plain(CONTEXT_CONTROL);
    REQUIRE(c);
    SetLastError(0);
    CHECK(!X.SetXStateFeaturesMask(c, T_XS_AVX | 3));
    CX_CHECK_WGAP("SetXStateFeaturesMask fails without a last error", GetLastError() == ERROR_INVALID_PARAMETER,
                  "GLE %lu want ERROR_INVALID_PARAMETER", (unsigned long)GetLastError());
    c->ContextFlags = CONTEXT_CONTROL;
    CHECK(X.SetXStateFeaturesMask(c, 3));
    CHECKF((c->ContextFlags & CX_XMMF) == CX_XMMF, "legacy mask 3 did not request the FP part: flags 0x%lx",
           (unsigned long)c->ContextFlags);
    CHECK(c->ContextFlags & CONTEXT_CONTROL & 0xFF);
    uint64_t m = 0;
    CHECK(X.GetXStateFeaturesMask(c, &m));
    CHECK_EQ(m, 3);
    cx_free(c);
}

TEST(get_mask_plain_contexts) {
    CX_NEED_XAPI();
    CONTEXT *c = cx_plain(CONTEXT_CONTROL);
    REQUIRE(c);
    DWORD64 m = 0xDEAD;
    CHECK(X.GetXStateFeaturesMask(c, &m));
    CHECK_EQ(m, 0);
    c->ContextFlags = CX_STD;
    m = 0xDEAD;
    CHECK(X.GetXStateFeaturesMask(c, &m));
    CHECK_EQ(m, 3);
    /* no architecture bit at all: STATUS_NOT_SUPPORTED */
    c->ContextFlags = 0x1F;
    m = 0xDEAD;
    SetLastError(0);
    CHECK(!X.GetXStateFeaturesMask(c, &m));
    CX_CHECK_WGAP("GetXStateFeaturesMask fails without a last error", GetLastError() == ERROR_NOT_SUPPORTED,
                  "GLE %lu want ERROR_NOT_SUPPORTED", (unsigned long)GetLastError());
    c->ContextFlags = 0x1F;
    SetLastError(0);
    CHECK(!X.SetXStateFeaturesMask(c, 3));
    CX_CHECK_WGAP("SetXStateFeaturesMask fails without a last error", GetLastError() == ERROR_NOT_SUPPORTED,
                  "GLE %lu want ERROR_NOT_SUPPORTED", (unsigned long)GetLastError());
    cx_free(c);
}

/* ======================================================================================================
 * 4. CopyContext
 * ====================================================================================================== */
TEST(copy_context_invalid_flags) {
    CX_NEED_XAPI();
    cx_xc a, b;
    CX_XC_REQUIRE(&a, CX_STD);
    CX_XC_REQUIRE(&b, CX_STD);
    SetLastError(0);
    CHECK(!X.CopyContext(a.ctx, 0, b.ctx));
    CHECK_EQ(GetLastError(), ERROR_INVALID_PARAMETER);
    SetLastError(0);
#ifdef T_X64
    BOOL r = X.CopyContext(a.ctx, CONTEXT_AMD64 | 0x20, b.ctx);
#else
    BOOL r = X.CopyContext(a.ctx, CONTEXT_i386 | 0x80, b.ctx);
#endif
    CX_CHECK_WGAP("CopyContext does not validate ContextFlags", !r && GetLastError() == ERROR_INVALID_PARAMETER,
                  "CopyContext(undefined flag bit) r=%d GLE %lu want FALSE / ERROR_INVALID_PARAMETER", r, (unsigned long)GetLastError());
    cx_xc_free(&a); cx_xc_free(&b);
}

TEST(copy_context_partial_control) {
    CX_NEED_XAPI();
    cx_xc s, d;
    CX_XC_REQUIRE(&s, CX_STD);
    CX_XC_REQUIRE(&d, CX_STD);
    for (int i = 0; i < CX_NGPR; i++) cx_set_gpr(s.ctx, i, 0x1111111111111111ull * (uint64_t)(i + 1));
    cx_set_ip(s.ctx, 0x12345670);
    for (int i = 0; i < CX_NV; i++) memset(cx_xmm(s.ctx, i), 0x30 + i, 16);
    for (int i = 0; i < CX_NGPR; i++) cx_set_gpr(d.ctx, i, 0xA5A5A5A5A5A5A5A5ull);
    for (int i = 0; i < CX_NV; i++) memset(cx_xmm(d.ctx, i), CX_SENT, 16);
    cx_set_ip(d.ctx, 0);
    CHECK(X.CopyContext(d.ctx, CONTEXT_CONTROL, s.ctx));
    CHECK_EQ(cx_ip(d.ctx), 0x12345670);
    for (int i = 0; i < CX_NGPR; i++) {
        if (cx_gpr_is_control(i)) { CHECK_EQ(cx_gpr(d.ctx, i), cx_gpr(s.ctx, i)); continue; }
        uint64_t g = cx_gpr(d.ctx, i);
        CHECKF(cx_all_sent(&g, sizeof(uintptr_t)), "CopyContext(CONTEXT_CONTROL) copied %s", cx_gpr_name(i));
    }
    for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(d.ctx, i), 16), "CopyContext(CONTEXT_CONTROL) copied xmm%d", i);
    CHECK(X.CopyContext(d.ctx, CONTEXT_INTEGER, s.ctx));
    for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(cx_gpr(d.ctx, i), cx_gpr(s.ctx, i));
    for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(d.ctx, i), 16), "CopyContext(CONTEXT_INTEGER) copied xmm%d", i);
    cx_set_mxcsr(s.ctx, 0x3F80);
    cx_set_mxcsr(d.ctx, 0x1F80);
    CHECK(X.CopyContext(d.ctx, CX_XMMF, s.ctx));
    for (int i = 0; i < CX_NV; i++) CHECK_MEM(cx_xmm(d.ctx, i), cx_xmm(s.ctx, i), 16);
    CHECK_EQ(cx_fx_mxcsr(d.ctx), 0x3F80);
#ifdef T_X64
    /* ntdll!RtlpCopyLegacyContextAmd64 copies CONTEXT.MxCsr (+0x34) with the FP part */
    CX_CHECK_WGAP("CopyContext does not copy CONTEXT.MxCsr", d.ctx->MxCsr == 0x3F80, "CONTEXT.MxCsr 0x%lx after CopyContext(FP), want 0x3F80",
                  (unsigned long)d.ctx->MxCsr);
#endif
    cx_xc_free(&s); cx_xc_free(&d);
}

TEST(copy_context_xstate_into_plain) {
    CX_NEED_AVX();
    cx_xc s, d;
    CX_XC_REQUIRE(&s, CX_STD | CONTEXT_XSTATE);
    CX_XC_REQUIRE(&d, CX_STD);
    CHECK(X.SetXStateFeaturesMask(s.ctx, T_XS_AVX));
    SetLastError(0);
    BOOL ok = X.CopyContext(d.ctx, CX_STD | CONTEXT_XSTATE, s.ctx);
    CHECK(!ok);
    CHECK_EQ(GetLastError(), ERROR_MORE_DATA);
    cx_xc_free(&s); cx_xc_free(&d);
}

TEST(copy_context_xstate_contents) {
    CX_NEED_AVX();
    cx_xc s, d;
    CX_XC_REQUIRE(&s, CX_STD | CONTEXT_XSTATE);
    CX_XC_REQUIRE(&d, CX_STD | CONTEXT_XSTATE);
    CHECK(X.SetXStateFeaturesMask(s.ctx, T_XS_AVX));
    uint8_t *sa = cx_locate(s.ctx, CX_XS_AVX, NULL), *da = cx_locate(d.ctx, CX_XS_AVX, NULL);
    REQUIRE(sa && da);
    uint8_t want[256];
    t_pattern_reg(want, 256, 0xC0C0, 1);
    memcpy(sa, want, 256);
    for (int i = 0; i < CX_NV; i++) t_pattern_reg(cx_xmm(s.ctx, i), 16, 0xC0C1, (uint32_t)i);
    memset(da, CX_SENT, 256);
    CHECK(X.CopyContext(d.ctx, CX_STD | CONTEXT_XSTATE, s.ctx));
    CHECK_EQ(cx_mask(d.ctx) & T_XS_AVX, T_XS_AVX);
    da = cx_locate(d.ctx, CX_XS_AVX, NULL);
    REQUIRE(da);
    CHECK_MEM(da, want, 256);
    for (int i = 0; i < CX_NV; i++) CHECK_MEM(cx_xmm(d.ctx, i), cx_xmm(s.ctx, i), 16);
    /* source without AVX in its mask: destination mask follows the source */
    CHECK(X.SetXStateFeaturesMask(s.ctx, 0));
    CHECK(X.CopyContext(d.ctx, CX_STD | CONTEXT_XSTATE, s.ctx));
    CHECK_EQ(cx_mask(d.ctx) & T_XS_AVX, 0);
    cx_xc_free(&s); cx_xc_free(&d);
}

/* ======================================================================================================
 * 5. Suspended worker: GetThreadContext views
 * ====================================================================================================== */
TEST(suspend_resume_counts) {
    cx_args *a = cx_args_new(0x5100, 0);
    REQUIRE(a);
    cx_worker w;
    if (!cx_worker_start(&w, a, T_VK_XMM)) { cx_worker_finish(&w); cx_free(a); REQUIREF(0, "worker did not start"); }
    DWORD r1 = SuspendThread(w.h), r2 = SuspendThread(w.h);
    CHECK_EQ(r1, 0);
    CHECK_EQ(r2, 1);
    DWORD r3 = ResumeThread(w.h), r4 = ResumeThread(w.h), r5 = ResumeThread(w.h);
    CHECK_EQ(r3, 2);
    CHECK_EQ(r4, 1);
    CHECK_EQ(r5, 0);
    /* still running: a fresh suspend returns 0 */
    CHECK_EQ(SuspendThread(w.h), 0);
    cx_release(&w);
    for (int i = 0; i < CX_NV; i++) CHECK_MEM(a->vout[i], a->vin[i], 16);
    cx_free(a);
}

TEST(getctx_full_gprs_control) {
    cx_args *a = cx_args_new(0x5200, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) {
            cx_check_gprs(c, a, 1, "CX_STD");
            cx_check_control(c, a, T_VK_XMM, "CX_STD");
#ifdef T_X86
            CHECKF(c->SegDs == cx_own_ds(), "SegDs 0x%lx want 0x%x", (unsigned long)c->SegDs, cx_own_ds());
            CHECKF(c->SegEs == cx_own_es(), "SegEs 0x%lx want 0x%x", (unsigned long)c->SegEs, cx_own_es());
            CHECKF(c->Ecx <= a->max_loops && c->Ecx > 0, "Ecx (loop counter) 0x%lx out of range", (unsigned long)c->Ecx);
            CHECKF(c->Eax == (DWORD)(uintptr_t)&a->ready, "Eax 0x%lx want &ready %p", (unsigned long)c->Eax, (void *)&a->ready);
#else
            CHECKF(c->Rax == (DWORD64)(uintptr_t)&a->ready, "Rax 0x%llx want &ready %p", (unsigned long long)c->Rax, (void *)&a->ready);
            CHECKF(c->R11 <= a->max_loops && c->R11 > 0, "R11 (loop counter) 0x%llx out of range", (unsigned long long)c->R11);
#endif
            t_info("segcs", "0x%x", (unsigned)c->SegCs);
            t_info("eflags", "0x%lx", (unsigned long)c->EFlags);
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(getctx_control_only_leaves_rest) {
    cx_args *a = cx_args_new(0x5300, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_CONTROL);
        if (c && GetThreadContext(w.h, c)) {
            cx_check_control(c, a, T_VK_XMM, "CONTEXT_CONTROL");
#ifdef T_X86
            /* EBP is part of CONTEXT_CONTROL on x86 (winnt.h) */
            CX_CHECK_WGAP("x86 CONTEXT_CONTROL does not carry Ebp", c->Ebp == (DWORD)a->gin[4], "Ebp 0x%lx want 0x%lx",
                          (unsigned long)c->Ebp, (unsigned long)(DWORD)a->gin[4]);
#endif
            CHECKF(cx_gprs_untouched(c, 0), "CONTEXT_CONTROL filled integer registers");
            for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(c, i), 16), "CONTEXT_CONTROL filled xmm%d", i);
            CHECKF(cx_all_sent(cx_st(c, 0), 10), "CONTEXT_CONTROL filled the x87 stack");
#ifdef T_X64
            CHECKF(cx_all_sent(&c->Dr0, 8) && cx_all_sent(&c->Dr7, 8), "CONTEXT_CONTROL filled debug registers");
            CHECKF(cx_all_sent(&c->Rax, 8), "CONTEXT_CONTROL filled Rax");
            if (!cx_all_sent(&c->MxCsr, 4)) t_note("CONTEXT_CONTROL also filled MxCsr (0x%lx)", (unsigned long)c->MxCsr);
#else
            CHECKF(cx_all_sent(&c->Dr0, 4) && cx_all_sent(&c->Dr7, 4), "CONTEXT_CONTROL filled debug registers");
            CHECKF(cx_all_sent(&c->Eax, 4), "CONTEXT_CONTROL filled Eax");
#endif
            t_info("flags_after", "0x%lx", (unsigned long)c->ContextFlags);
        } else CHECKF(0, "GetThreadContext(CONTEXT_CONTROL) GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(getctx_integer_only_leaves_rest) {
    cx_args *a = cx_args_new(0x5400, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_INTEGER);
        if (c && GetThreadContext(w.h, c)) {
            cx_check_gprs(c, a, 0, "CONTEXT_INTEGER");
            uintptr_t ip = cx_ip(c), sp = cx_sp(c);
            CHECKF(cx_all_sent(&ip, sizeof ip), "CONTEXT_INTEGER filled the instruction pointer");
            CHECKF(cx_all_sent(&sp, sizeof sp), "CONTEXT_INTEGER filled the stack pointer");
            CHECKF(cx_all_sent(&c->SegCs, sizeof c->SegCs) && cx_all_sent(&c->EFlags, sizeof c->EFlags), "CONTEXT_INTEGER filled SegCs/EFlags");
            for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(c, i), 16), "CONTEXT_INTEGER filled xmm%d", i);
#ifdef T_X86
            CHECKF(cx_all_sent(&c->Ebp, 4), "CONTEXT_INTEGER filled Ebp (a CONTEXT_CONTROL register on x86)");
#endif
        } else CHECKF(0, "GetThreadContext(CONTEXT_INTEGER) GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(getctx_fp_only_leaves_rest) {
    cx_args *a = cx_args_new(0x5500, 1);
    REQUIRE(a);
    a->mxcsr_in = 0x5F81;     /* RC=10 (round up), all masked, IE sticky */
    a->fcw_in = 0x0B7F;       /* PC=11, RC=10, all masked */
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CX_XMMF | CONTEXT_FLOATING_POINT);
        if (c && GetThreadContext(w.h, c)) {
            cx_check_xmm(c, a->vin, CX_NV, "FP-only view");
            CHECK_EQ(cx_mxcsr(c), 0x5F81);
            CHECK_EQ(cx_fx_mxcsr(c), 0x5F81);
            CHECK_EQ(cx_fcw(c), 0x0B7F);
            CHECKF(cx_gprs_untouched(c, 1), "CONTEXT_FLOATING_POINT filled general registers");
            uintptr_t ip = cx_ip(c);
            CHECKF(cx_all_sent(&ip, sizeof ip), "CONTEXT_FLOATING_POINT filled the instruction pointer");
#ifdef T_X64
            CHECKF(cx_all_sent(&c->Dr7, 8), "CONTEXT_FLOATING_POINT filled Dr7");
#else
            CHECKF(cx_all_sent(&c->Dr7, 4), "CONTEXT_FLOATING_POINT filled Dr7");
#endif
        } else CHECKF(0, "GetThreadContext(FP) GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    CHECK_EQ(a->mxcsr_out, 0x5F81);
    cx_free(a);
}

TEST(getctx_xmm_all_registers) {
    cx_args *a = cx_args_new(0x5600, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) {
            cx_check_xmm(c, a->vin, CX_NV, "GetThreadContext");
#ifdef T_X64
            /* the named Xmm0..Xmm15 fields alias FltSave.XmmRegisters */
            CHECK_MEM(&c->Xmm7, a->vin[7], 16);
            CHECK_MEM(&c->Xmm15, a->vin[15], 16);
#endif
            CHECK_EQ(cx_mxcsr(c), 0x1F80);
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    for (int i = 0; i < CX_NV; i++) CHECK_MEM(a->vout[i], a->vin[i], 16);
    cx_free(a);
}

TEST(getctx_x87_stack_and_cw) {
    cx_args *a = cx_args_new(0x5700, 1);
    REQUIRE(a);
    a->fcw_in = 0x037F;
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) {
            cx_check_x87(c, a, "eight pushed values");
            CHECK_EQ(cx_fcw(c), 0x037F);
            uint16_t fxcw; memcpy(&fxcw, cx_fx(c), 2);
            CHECK_EQ(fxcw, 0x037F);
            uint16_t fxsw; memcpy(&fxsw, cx_fx(c) + 2, 2);
            CHECKF(((fxsw >> 11) & 7) == 0, "FSW 0x%x: TOP %u after eight pushes, want 0", fxsw, (fxsw >> 11) & 7);
            CHECKF(cx_fx(c)[4] == 0xFF, "abridged tag 0x%02x, want 0xFF (all valid)", cx_fx(c)[4]);
#ifdef T_X86
            CHECKF((c->FloatSave.TagWord & 0xFFFF) == 0, "full tag word 0x%lx, want 0 (all valid normals)", (unsigned long)c->FloatSave.TagWord);
            CHECKF(((c->FloatSave.StatusWord >> 11) & 7) == 0, "FloatSave.StatusWord TOP != 0");
#endif
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    for (int i = 0; i < 8; i++) CHECK_MEM(a->fout[i].b, a->fin[i].b, 10);
    cx_free(a);
}

TEST(getctx_twice_identical) {
    cx_args *a = cx_args_new(0x5800, 1);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c1 = cx_get_std(&w), *c2 = cx_get_std(&w);
        if (c1 && c2) {
            CHECK_EQ(cx_ip(c1), cx_ip(c2));
            CHECK_EQ(cx_sp(c1), cx_sp(c2));
            for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(cx_gpr(c1, i), cx_gpr(c2, i));
            CHECK_MEM(cx_xmm(c1, 0), cx_xmm(c2, 0), 16 * CX_NV);
            CHECK_MEM(cx_fx(c1) + 32, cx_fx(c2) + 32, 128);
        }
        cx_free(c1); cx_free(c2);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(getctx_debug_registers_separate) {
    cx_args *a = cx_args_new(0x5900, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_DEBUG_REGISTERS);
        if (c && GetThreadContext(w.h, c)) {
            if (IsDebuggerPresent()) t_note("debugger present: Dr7 0x%llx not checked", (unsigned long long)c->Dr7);
            else CHECKF((c->Dr7 & 0xFF) == 0, "Dr7 0x%llx has enable bits although no hardware breakpoint was set",
                        (unsigned long long)c->Dr7);
            t_info("dr0_3", "%llx %llx %llx %llx", (unsigned long long)c->Dr0, (unsigned long long)c->Dr1,
                   (unsigned long long)c->Dr2, (unsigned long long)c->Dr3);
            t_info("dr6_dr7", "%llx %llx", (unsigned long long)c->Dr6, (unsigned long long)c->Dr7);
            CHECKF(cx_gprs_untouched(c, 1), "CONTEXT_DEBUG_REGISTERS filled general registers");
            for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(c, i), 16), "CONTEXT_DEBUG_REGISTERS filled xmm%d", i);
            uintptr_t ip = cx_ip(c);
            CHECKF(cx_all_sent(&ip, sizeof ip), "CONTEXT_DEBUG_REGISTERS filled the instruction pointer");
        } else CHECKF(0, "GetThreadContext(CONTEXT_DEBUG_REGISTERS) GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
        /* the FP request does not fill the debug registers */
        c = cx_plain(CX_XMMF);
        if (c && GetThreadContext(w.h, c)) {
#ifdef T_X64
            CHECKF(cx_all_sent(&c->Dr0, 8 * 4) && cx_all_sent(&c->Dr6, 16), "FP request filled Dr0-Dr7");
#else
            CHECKF(cx_all_sent(&c->Dr0, 4 * 6), "FP request filled Dr0-Dr7");
#endif
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(setctx_debug_registers_zero) {
    cx_args *a = cx_args_new(0x5A00, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_DEBUG_REGISTERS);
        if (c && GetThreadContext(w.h, c)) {
            c->Dr0 = c->Dr1 = c->Dr2 = c->Dr3 = 0;
            c->Dr7 = 0;
            c->ContextFlags = CONTEXT_DEBUG_REGISTERS;
            CHECKF(SetThreadContext(w.h, c), "SetThreadContext(CONTEXT_DEBUG_REGISTERS, all disabled) GLE=%lu", (unsigned long)GetLastError());
            memset(c, CX_SENT, sizeof *c);
            c->ContextFlags = CONTEXT_DEBUG_REGISTERS;
            CHECK(GetThreadContext(w.h, c));
            CHECK_EQ(c->Dr7 & 0xFF, 0);
            CHECK_EQ(c->Dr0, 0);
            CHECK_EQ(c->Dr3, 0);
        } else CHECKF(0, "GetThreadContext(CONTEXT_DEBUG_REGISTERS) GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(a->gout[i], cx_gpr_expect(a, i));
    cx_free(a);
}

TEST(getctx_flags_zero_fills_nothing) {
    /* ContextFlags without any part: x64 Windows sanitizes user flags (RtlpSanitizeContextFlags) instead of
     * failing, so the result is an observation; if the call succeeds it must not fill anything. */
    cx_args *a = cx_args_new(0x5B00, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(0);
        SetLastError(0);
        BOOL r = GetThreadContext(w.h, c);
        t_info("flags0_result", "%d gle=%lu flags_after=0x%lx", r, (unsigned long)GetLastError(), (unsigned long)c->ContextFlags);
        if (r) {
            CHECKF(cx_gprs_untouched(c, 1), "ContextFlags=0 filled general registers");
            for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(c, i), 16), "ContextFlags=0 filled xmm%d", i);
            uintptr_t ip = cx_ip(c);
            CHECKF(cx_all_sent(&ip, sizeof ip), "ContextFlags=0 filled the instruction pointer");
        }
        /* architecture bit only: the same */
        memset(c, CX_SENT, sizeof *c);
        c->ContextFlags = CX_ARCHBIT;
        r = GetThreadContext(w.h, c);
        CHECKF(r, "GetThreadContext(architecture bit only) GLE=%lu", (unsigned long)GetLastError());
        if (r) {
            CHECKF(cx_gprs_untouched(c, 1), "arch-bit-only request filled general registers");
            for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(c, i), 16), "arch-bit-only request filled xmm%d", i);
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

/* ======================================================================================================
 * 6. SetThreadContext: the resumed thread observes the change
 * ====================================================================================================== */
TEST(setctx_identity_roundtrip) {
    cx_args *a = cx_args_new(0x6100, 1);
    REQUIRE(a);
    a->mxcsr_in = 0x3F80; a->fcw_in = 0x0A7F;
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) { CHECKF(SetThreadContext(w.h, c), "SetThreadContext(unchanged) GLE=%lu", (unsigned long)GetLastError()); cx_free(c); }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < CX_NV; i++) CHECK_REG("xmm after identity set", a->vout[i], a->vin[i], 16);
        for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(a->gout[i], cx_gpr_expect(a, i));
        for (int i = 0; i < 8; i++) CHECK_MEM(a->fout[i].b, a->fin[i].b, 10);
        CHECK_EQ(a->mxcsr_out, 0x3F80);
        CHECK_EQ(a->fcw_out & CX_FCW_BITS, 0x0A7F & CX_FCW_BITS);
    }
    cx_free(a);
}

TEST(setctx_xmm_observed) {
    cx_args *a = cx_args_new(0x6200, 0);
    REQUIRE(a);
    uint8_t want[16][32];
    for (int i = 0; i < 16; i++) t_pattern_reg(want[i], 32, 0x6201, (uint32_t)i);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CX_XMMF);
        if (c && GetThreadContext(w.h, c)) {
            for (int i = 0; i < CX_NV; i++) memcpy(cx_xmm(c, i), want[i], 16);
            c->ContextFlags = CX_XMMF;
            CHECKF(SetThreadContext(w.h, c), "SetThreadContext(xmm) GLE=%lu", (unsigned long)GetLastError());
        } else CHECKF(0, "GetThreadContext GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < CX_NV; i++) CHECK_REG("xmm set via SetThreadContext", a->vout[i], want[i], 16);
        for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(a->gout[i], cx_gpr_expect(a, i));
    }
    cx_free(a);
}

TEST(setctx_get_after_set_suspended) {
    cx_args *a = cx_args_new(0x6300, 0);
    REQUIRE(a);
    uint8_t want[16];
    t_pattern_reg(want, 16, 0x6301, 5);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) {
            memcpy(cx_xmm(c, 5), want, 16);
            cx_set_gpr(c, 0, 0x0123456789ABCDEFull);
            cx_set_mxcsr(c, 0x7F80);
            CHECK(SetThreadContext(w.h, c));
            CONTEXT *d = cx_get_std(&w);
            if (d) {
                CHECK_MEM(cx_xmm(d, 5), want, 16);
                CHECK_EQ(cx_gpr(d, 0), (uintptr_t)0x0123456789ABCDEFull);
                CHECK_EQ(cx_mxcsr(d), 0x7F80);
                CHECK_EQ(cx_ip(d), cx_ip(c));
                CHECK_EQ(cx_sp(d), cx_sp(c));
                cx_free(d);
            }
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        CHECK_MEM(a->vout[5], want, 16);
        CHECK_EQ(a->gout[0], (uintptr_t)0x0123456789ABCDEFull);
        CHECK_EQ(a->mxcsr_out, 0x7F80);
    }
    cx_free(a);
}

TEST(setctx_mxcsr_observed) {
    cx_args *a = cx_args_new(0x6400, 0);
    REQUIRE(a);
    static const uint32_t vals[] = { 0x7F80, 0x1F80 | 0x8000 | 0x40, 0x1F80 | 0x003F, 0x1F00 };
    uint32_t mask = cx_mxcsr_mask();
    for (size_t k = 0; k < T_COUNTOF(vals); k++) {
        uint32_t v = vals[k] & mask;
        cx_args_fill(a, 0x6400 + (uint32_t)k, 0);
        cx_worker w;
        int ok = cx_worker_start(&w, a, T_VK_XMM);
        if (ok && cx_suspend(&w)) {
            CONTEXT *c = cx_plain(CX_XMMF);
            if (c && GetThreadContext(w.h, c)) {
                CHECK_EQ(cx_mxcsr(c), 0x1F80);
                cx_set_mxcsr(c, v);
                CHECKF(SetThreadContext(w.h, c), "SetThreadContext(MxCsr=0x%x) GLE=%lu", v, (unsigned long)GetLastError());
            }
            cx_free(c);
        } else CHECKF(ok, "worker did not start");
        cx_release(&w);
        if (w.done) CHECKF(a->mxcsr_out == v, "worker MXCSR 0x%x after setting 0x%x", a->mxcsr_out, v);
        if (w.done) for (int i = 0; i < CX_NV; i++) CHECK_MEM(a->vout[i], a->vin[i], 16);
    }
    cx_free(a);
}

TEST(setctx_mxcsr_field_precedence_note) {
#ifdef T_X64
    /* x64 CONTEXT has MxCsr twice (CONTEXT.MxCsr and FltSave.MxCsr). Which one SetThreadContext applies is
     * not documented: observation only. */
    cx_args *a = cx_args_new(0x6500, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CX_XMMF);
        if (c && GetThreadContext(w.h, c)) {
            c->MxCsr = 0x3F80;
            c->FltSave.MxCsr = 0x5F80;
            CHECK(SetThreadContext(w.h, c));
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        CHECKF(a->mxcsr_out == 0x3F80 || a->mxcsr_out == 0x5F80, "MXCSR 0x%x is neither field", a->mxcsr_out);
        t_info("mxcsr_winner", "%s (0x%x)", a->mxcsr_out == 0x3F80 ? "CONTEXT.MxCsr" : "FltSave.MxCsr", a->mxcsr_out);
    }
    cx_free(a);
#else
    SKIP("x86 CONTEXT has one MXCSR (ExtendedRegisters)");
#endif
}

TEST(setctx_x87_cw_observed) {
    cx_args *a = cx_args_new(0x6600, 1);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CX_STD);
        if (c && GetThreadContext(w.h, c)) {
            CHECK_EQ(cx_fcw(c), 0x027F);
            cx_set_fcw(c, 0x0C7F);       /* RC = truncate, PC = 53-bit */
            c->ContextFlags = CONTEXT_FLOATING_POINT | CX_XMMF;
            CHECKF(SetThreadContext(w.h, c), "SetThreadContext(x87 CW) GLE=%lu", (unsigned long)GetLastError());
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        CHECK_EQ(a->fcw_out & CX_FCW_BITS, 0x0C7F & CX_FCW_BITS);
        for (int i = 0; i < 8; i++) CHECK_MEM(a->fout[i].b, a->fin[i].b, 10);
    }
    cx_free(a);
}

TEST(setctx_x87_stack_observed) {
    cx_args *a = cx_args_new(0x6700, 1);
    REQUIRE(a);
    t_f80 nv;
    cx_f80_make(&nv, 0x6777, 3);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CX_STD);
        if (c && GetThreadContext(w.h, c)) {
            cx_set_st(c, 3, &nv);
            c->ContextFlags = CONTEXT_FLOATING_POINT | CX_XMMF;
            CHECKF(SetThreadContext(w.h, c), "SetThreadContext(ST3) GLE=%lu", (unsigned long)GetLastError());
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < 8; i++) {
            const uint8_t *want = i == 3 ? nv.b : a->fin[i].b;
            char lab[32]; snprintf(lab, sizeof lab, "ST(%d)", i);
            CHECK_REG(lab, a->fout[i].b, want, 10);
        }
    }
    cx_free(a);
}

TEST(setctx_integer_observed) {
    cx_args *a = cx_args_new(0x6800, 0);
    REQUIRE(a);
    uint64_t nv[CX_NGPR];
    for (int i = 0; i < CX_NGPR; i++) nv[i] = t_pattern_word(0x6801, 40 + (uint32_t)i, 0);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    uintptr_t ip_before = 0;
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CX_STD);
        if (c && GetThreadContext(w.h, c)) {
            ip_before = cx_ip(c);
            for (int i = 0; i < CX_NGPR; i++) if (!cx_gpr_is_control(i)) cx_set_gpr(c, i, nv[i]);
            cx_set_ip(c, 0x1);    /* must be ignored: CONTEXT_CONTROL not requested */
            c->ContextFlags = CONTEXT_INTEGER;
            CHECKF(SetThreadContext(w.h, c), "SetThreadContext(CONTEXT_INTEGER) GLE=%lu", (unsigned long)GetLastError());
            CONTEXT *d = cx_plain(CONTEXT_CONTROL);
            if (d && GetThreadContext(w.h, d)) CHECKF(cx_ip(d) == ip_before, "CONTEXT_INTEGER set changed IP to %p", (void *)cx_ip(d));
            cx_free(d);
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < CX_NGPR; i++) {
            uint64_t want = cx_gpr_is_control(i) ? cx_gpr_expect(a, i) : nv[i];
#ifdef T_X86
            want = (uint32_t)want;
#endif
            if (cx_gpr_is_control(i))
                CX_CHECK_WGAP("x86 CONTEXT_INTEGER set changes Ebp (a CONTEXT_CONTROL register)", a->gout[i] == want,
                              "%s observed 0x%llx want 0x%llx", cx_gpr_name(i), (unsigned long long)a->gout[i], (unsigned long long)want);
            else CHECKF(a->gout[i] == want, "%s observed 0x%llx want 0x%llx", cx_gpr_name(i), (unsigned long long)a->gout[i], (unsigned long long)want);
        }
        for (int i = 0; i < CX_NV; i++) CHECK_MEM(a->vout[i], a->vin[i], 16);
    }
    cx_free(a);
}

TEST(setctx_control_rip_redirect) {
    /* The worker spins without a stop flag; CONTEXT_CONTROL moves its instruction pointer to the loop exit. */
    cx_args *a = cx_args_new(0x6900, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    int set_ok = 0;
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_CONTROL);
        if (c && GetThreadContext(w.h, c)) {
            cx_set_ip(c, (uintptr_t)cx_xmm_exit);
            set_ok = SetThreadContext(w.h, c);
            CHECKF(set_ok, "SetThreadContext(CONTEXT_CONTROL) GLE=%lu", (unsigned long)GetLastError());
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    if (set_ok) {
        DWORD r = ResumeThread(w.h);
        CHECK_EQ(r, 1);
        DWORD wr = WaitForSingleObject(w.h, CX_TMO);
        CHECKF(wr == WAIT_OBJECT_0, "redirected worker did not exit (wait=%lu)", (unsigned long)wr);
        if (wr == WAIT_OBJECT_0) {
            CHECKF(a->remaining != 0, "worker ran to its loop budget instead of the redirected exit");
            for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(a->gout[i], cx_gpr_expect(a, i));
            for (int i = 0; i < CX_NV; i++) CHECK_MEM(a->vout[i], a->vin[i], 16);
        }
    }
    cx_worker_finish(&w);
    cx_free(a);
}

TEST(setctx_rsp_rip_preserved_by_fp_set) {
    cx_args *a = cx_args_new(0x6A00, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) {
            uintptr_t ip = cx_ip(c), sp = cx_sp(c);
            cx_set_ip(c, 0x10); 
#ifdef T_X64
            c->Rsp = 0x20;
#else
            c->Esp = 0x20;
#endif
            c->ContextFlags = CX_XMMF;
            CHECK(SetThreadContext(w.h, c));
            CONTEXT *d = cx_get_std(&w);
            if (d) { CHECK_EQ(cx_ip(d), ip); CHECK_EQ(cx_sp(d), sp); cx_check_gprs(d, a, 1, "after FP-only set"); cx_free(d); }
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(getctx_segments_only) {
    /* CONTEXT_SEGMENTS = DS/ES/FS/GS. All threads of a process share these selectors, so the worker's values
     * equal the case thread's. CS/SS belong to CONTEXT_CONTROL and stay untouched. */
    cx_args *a = cx_args_new(0x5C00, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_SEGMENTS);
        if (c && GetThreadContext(w.h, c)) {
#define CX_SEGCHK(n, own) CX_CHECK_WGAP("context reports fixed Windows selectors, not the thread's", (uint16_t)c->n == (own), \
                                    #n " 0x%x want 0x%x (the thread's selector)", (unsigned)c->n, (unsigned)(own))
            CX_SEGCHK(SegDs, cx_own_ds());
            CX_SEGCHK(SegEs, cx_own_es());
            CX_SEGCHK(SegFs, cx_own_fs());
            CX_SEGCHK(SegGs, cx_own_gs());
#undef CX_SEGCHK
            CHECKF(cx_gprs_untouched(c, 1), "CONTEXT_SEGMENTS filled general registers");
            uintptr_t ip = cx_ip(c), sp = cx_sp(c);
            CHECKF(cx_all_sent(&ip, sizeof ip) && cx_all_sent(&sp, sizeof sp), "CONTEXT_SEGMENTS filled IP/SP");
            CHECKF(cx_all_sent(&c->SegCs, sizeof c->SegCs) && cx_all_sent(&c->SegSs, sizeof c->SegSs), "CONTEXT_SEGMENTS filled SegCs/SegSs");
            for (int i = 0; i < CX_NV; i++) CHECKF(cx_all_sent(cx_xmm(c, i), 16), "CONTEXT_SEGMENTS filled xmm%d", i);
            t_info("segments", "ds %x es %x fs %x gs %x", (unsigned)c->SegDs, (unsigned)c->SegEs, (unsigned)c->SegFs, (unsigned)c->SegGs);
        } else CHECKF(0, "GetThreadContext(CONTEXT_SEGMENTS) GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(getctx_x87_partial_stack) {
    /* Three values on the x87 stack: TOP = 5, ST(0..2) live in physical R5..R7. The CONTEXT keeps the
     * stack-relative ST(i) order (FXSAVE layout), the abridged tag byte is indexed by physical register.
     * Then ST(1) is replaced through SetThreadContext and the worker pops what it observes. */
    cx_args *a = cx_args_new(0x5D00, 1);
    REQUIRE(a);
    a->flags |= CXF_X87_3;
    t_f80 nv;
    cx_f80_make(&nv, 0x5D77, 1);
    int set_ok = 0;
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) {
            char lab[48];
            for (int i = 0; i < 3; i++) {
                snprintf(lab, sizeof lab, "ST(%d) of 3", i);         CHECK_REG(lab, cx_st(c, i), a->fin[i].b, 10);
                snprintf(lab, sizeof lab, "FXSAVE ST(%d) of 3", i);  CHECK_REG(lab, cx_fx(c) + 32 + 16 * i, a->fin[i].b, 10);
            }
            uint16_t fsw; memcpy(&fsw, cx_fx(c) + 2, 2);
            CHECKF(((fsw >> 11) & 7) == 5, "FSW 0x%x: TOP %u after three pushes, want 5", fsw, (fsw >> 11) & 7);
            CHECKF(cx_fx(c)[4] == 0xE0, "abridged tag 0x%02x, want 0xE0 (R5-R7 valid)", cx_fx(c)[4]);
#ifdef T_X86
            /* full tag word: R0-R4 empty (11), R5-R7 valid normals (00) */
            CHECKF((c->FloatSave.TagWord & 0xFFFF) == 0x03FF, "FloatSave.TagWord 0x%lx, want 0x03FF", (unsigned long)c->FloatSave.TagWord);
            CHECKF(((c->FloatSave.StatusWord >> 11) & 7) == 5, "FloatSave.StatusWord 0x%lx: TOP != 5", (unsigned long)c->FloatSave.StatusWord);
#endif
            cx_set_st(c, 1, &nv);
            c->ContextFlags = CONTEXT_FLOATING_POINT | CX_XMMF;
            set_ok = SetThreadContext(w.h, c);
            CHECKF(set_ok, "SetThreadContext(ST1) GLE=%lu", (unsigned long)GetLastError());
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        CHECK_REG("ST(0) observed", a->fout[0].b, a->fin[0].b, 10);
        CHECK_REG("ST(1) observed", a->fout[1].b, set_ok ? nv.b : a->fin[1].b, 10);
        CHECK_REG("ST(2) observed", a->fout[2].b, a->fin[2].b, 10);
    }
    cx_free(a);
}

TEST(setctx_eflags_sanitized) {
    /* User EFLAGS from a CONTEXT are sanitized, never trusted: IOPL cannot be raised, IF stays set, and on x64
     * NT is dropped (ntoskrnl!KxContextToKframes 19041: EFlags & 0x210DD5 | 0x200). Arithmetic/control flags
     * the user may own (DF here) are applied: the worker captures EFLAGS at its loop exit (CLD follows). */
    cx_args *a = cx_args_new(0x6B00, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    int set_ok = 0;
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_CONTROL);
        if (c && GetThreadContext(w.h, c)) {
            DWORD f = (c->EFlags | 0x400u /* DF */ | 0x3000u /* IOPL 3 */) & ~0x200u /* IF */;
#ifdef T_X64
            f |= 0x4000u;   /* NT */
#endif
            c->EFlags = f;
            set_ok = SetThreadContext(w.h, c);
            CHECKF(set_ok, "SetThreadContext(EFlags 0x%lx) GLE=%lu", (unsigned long)f, (unsigned long)GetLastError());
            CONTEXT *d = cx_plain(CONTEXT_CONTROL);
            if (set_ok && d && GetThreadContext(w.h, d)) {
                t_info("eflags_after_set", "0x%lx (requested 0x%lx)", (unsigned long)d->EFlags, (unsigned long)f);
                /* the stored context must report what the thread will really run with */
                const char *gap = "GetThreadContext reports unsanitized EFlags stored by SetThreadContext";
                CX_CHECK_WGAP(gap, (d->EFlags & 0x3000) == 0, "EFlags 0x%lx: IOPL raised through SetThreadContext", (unsigned long)d->EFlags);
                CX_CHECK_WGAP(gap, d->EFlags & 0x200, "EFlags 0x%lx: IF cleared through SetThreadContext", (unsigned long)d->EFlags);
                CHECKF(d->EFlags & 0x400, "EFlags 0x%lx: DF not applied", (unsigned long)d->EFlags);
#ifdef T_X64
                CX_CHECK_WGAP(gap, !(d->EFlags & 0x4000), "EFlags 0x%lx: NT applied", (unsigned long)d->EFlags);
#endif
            }
            cx_free(d);
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done && set_ok) {
        uint32_t e = a->eflags_out;
        t_info("eflags_observed", "0x%x", e);
        CHECKF(e & 0x400, "worker EFLAGS 0x%x: DF set through SetThreadContext not observed", e);
        CHECKF((e & 0x3000) == 0, "worker EFLAGS 0x%x: running with IOPL %u", e, (e >> 12) & 3);
        CHECKF(e & 0x200, "worker EFLAGS 0x%x: running with IF clear", e);
#ifdef T_X64
        CHECKF(!(e & 0x4000), "worker EFLAGS 0x%x: running with NT set", e);
#endif
        for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(a->gout[i], cx_gpr_expect(a, i));
    }
    cx_free(a);
}

TEST(getctx_eflags_df_clear_baseline) {
    /* Baseline for setctx_eflags_sanitized: without a set, the worker runs with DF clear, IF set, IOPL 0. */
    cx_args *a = cx_args_new(0x6C00, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CONTEXT_CONTROL);
        if (c && GetThreadContext(w.h, c)) {
            CHECKF(!(c->EFlags & 0x400), "EFlags 0x%lx: DF set in a C-callable routine", (unsigned long)c->EFlags);
            CHECKF((c->EFlags & 0x3000) == 0, "EFlags 0x%lx: IOPL != 0", (unsigned long)c->EFlags);
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        CHECKF(!(a->eflags_out & 0x400) && (a->eflags_out & 0x200), "worker EFLAGS 0x%x", a->eflags_out);
    }
    cx_free(a);
}

/* Child: SetThreadContext with reserved MXCSR bits on a suspended worker. Windows masks CONTEXT.MxCsr with
 * MXCSR_MASK (ntoskrnl!KxContextToKframes: & KiMxCsrMask); a raw LDMXCSR of it would #GP. Prints
 * SET=<ok> GLE=<e> then MX=<observed> DONE=<exited>. Runs in a child: a broken implementation kills the
 * worker (an exception on a non-case thread ends the process). */
CHILD(cx_mxres) {
    cx_resolve();
    cx_args *a = cx_args_new(0x6D00, 0);
    cx_worker w;
    if (!a || !cx_worker_start(&w, a, T_VK_XMM)) { printf("SET=-1 GLE=0 start failed\n"); return 3; }
    SuspendThread(w.h);
    CONTEXT *c = cx_plain(CX_XMMF);
    BOOL r = FALSE;
    DWORD e = 0;
    if (c && GetThreadContext(w.h, c)) {
        cx_set_mxcsr(c, 0xFFFF0000u | 0x1F80u);
        SetLastError(0);
        r = SetThreadContext(w.h, c);
        e = GetLastError();
    }
    printf("SET=%d GLE=%lu\n", r ? 1 : 0, (unsigned long)e);
    fflush(stdout);
    int done = cx_worker_finish(&w);
    printf("MX=%x DONE=%d\n", a->mxcsr_out, done);
    fflush(stdout);
    return 0;
}

TEST(setctx_mxcsr_reserved_bits) {
    static t_child_result r;
    REQUIRE(t_child_run("cx_mxres", "", 20000, &r));
    CHECKF(!r.timed_out, "child timed out");
    char val[128];
    int set = -1; unsigned long gle = 0;
    if (t_child_find(&r, "SET=", val, sizeof val)) sscanf(val, "%d GLE=%lu", &set, &gle);
    REQUIREF(set >= 0, "child setup failed: %.300s", r.out);
    unsigned mx = 0; int done = 0;
    int have = t_child_find(&r, "MX=", val, sizeof val) && sscanf(val, "%x DONE=%d", &mx, &done) == 2;
    if (r.exc_code || !have || !done) {
        if (t_is_wine()) XFAIL("wine %s: reserved MXCSR bits from a CONTEXT are not masked (worker died, exception 0x%lx)",
                               t_wine_version(), (unsigned long)r.exc_code);
        CHECKF(0, "worker did not survive SetThreadContext with reserved MXCSR bits (exception 0x%lx, output %.200s)",
               (unsigned long)r.exc_code, r.out);
        return;
    }
    t_info("set_result", "%d gle=%lu observed=0x%x", set, gle, mx);
    uint32_t want = (0xFFFF0000u | 0x1F80u) & cx_mxcsr_mask();
    if (set) CHECKF(mx == want, "observed MXCSR 0x%x, want 0x%x (requested value & MXCSR_MASK)", mx, want);
    else {
        t_note("SetThreadContext rejected reserved MXCSR bits (GLE %lu); Windows 10 masks them instead", gle);
        CHECKF(mx == 0x1F80, "rejected set still changed MXCSR to 0x%x", mx);
    }
    CHECK_EQ(r.exit_code, 0);
}

TEST(setctx_fxsave_reserved_bytes_ignored) {
    /* The FXSAVE image in a CONTEXT has reserved / software-available bytes (x64: 416-511, x86: 288-511).
     * Callers commonly leave them uninitialized (InitializeContext does not clear them, CopyContext does not
     * copy them); FXRSTOR/XRSTOR ignore them, so SetThreadContext must too: the thread keeps exact XMM and,
     * with CONTEXT_XSTATE, YMM upper state. */
    int avx = cx_avx_ok();
    cx_args *a = cx_args_new(0x6E00, 0);
    REQUIRE(a);
    uint8_t lo[16];
    t_pattern_reg(lo, 16, 0x6E01, 1);
    cx_worker w;
    int ok = cx_worker_start(&w, a, avx ? T_VK_YMM : T_VK_XMM);
    int set_ok = 0;
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        CONTEXT *plain = NULL, *c = NULL;
        if (avx) { if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) c = x.ctx; }
        else { plain = cx_get_std(&w); c = plain; }
        if (c) {
            memset(cx_fx(c) + CX_FX_DEFINED, 0xA5, 512 - CX_FX_DEFINED);
            memcpy(cx_xmm(c, 1), lo, 16);
            set_ok = SetThreadContext(w.h, c);
            CHECKF(set_ok, "SetThreadContext with garbage in reserved FXSAVE bytes GLE=%lu", (unsigned long)GetLastError());
        }
        cx_xc_free(&x);
        cx_free(plain);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done && set_ok) {
        CHECK_REG("xmm1 set", a->vout[1], lo, 16);
        for (int i = 0; i < CX_NV; i++) if (i != 1) CHECK_REG("xmm untouched", a->vout[i], a->vin[i], 16);
        if (avx) {
            int up = 1;
            for (int i = 0; i < CX_NV; i++) if (memcmp(a->vout[i] + 16, a->vin[i] + 16, 16)) up = 0;
            CX_CHECK_WGAP("reserved FXSAVE bytes of a CONTEXT reach the kernel's sigreturn frame (YMM state dropped)", up,
                          "YMM upper halves changed by a set whose only difference is reserved FXSAVE bytes");
        }
    }
    cx_free(a);
}

/* ======================================================================================================
 * 7. CONTEXT_XSTATE: YMM
 * ====================================================================================================== */
TEST(xstate_getctx_ymm_upper) {
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7100, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            uint64_t m = cx_mask(x.ctx);
            t_info("mask_after_get", "0x%llx", (unsigned long long)m);
            CHECKF(m & T_XS_AVX, "feature mask 0x%llx lacks AVX although the upper halves are non-zero", (unsigned long long)m);
            DWORD l = 0;
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, &l);
            CHECK(ymmh != NULL);
            CHECK_EQ(l, 256);
            if (ymmh && (m & T_XS_AVX)) {
                char lab[48];
                for (int i = 0; i < CX_NV; i++) {
                    snprintf(lab, sizeof lab, "ymm%d upper (XSTATE)", i);
                    CHECK_REG(lab, ymmh + 16 * i, a->vin[i] + 16, 16);
                }
            }
            cx_check_xmm(x.ctx, a->vin, CX_NV, "low lanes (XSTATE)");
            uint8_t *sse = cx_locate(x.ctx, 1, NULL);
            CHECK(sse == cx_xmm(x.ctx, 0));
            cx_check_gprs(x.ctx, a, 1, "XSTATE get");
            cx_check_control(x.ctx, a, T_VK_YMM, "XSTATE get");
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) for (int i = 0; i < CX_NV; i++) CHECK_REG("ymm after XSTATE get", a->vout[i], a->vin[i], 32);
    cx_free(a);
}

TEST(xstate_getctx_ymm_init_state) {
    /* Worker uses only legacy SSE (fresh thread: YMM upper halves in init state). Either the AVX bit is clear
     * in the returned mask (init state), or the area is present and all zero. */
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7200, 0);
    REQUIRE(a);
    /* the x64 ABI lets earlier code leave YMM upper halves dirty: start from VZEROUPPER, then legacy SSE loads */
    a->flags |= CXF_VZU;
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            uint64_t m = cx_mask(x.ctx);
            t_info("mask_init_ymm", "0x%llx", (unsigned long long)m);
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
            if (m & T_XS_AVX) CHECKF(ymmh && cx_all_zero(ymmh, 16 * CX_NV), "AVX present in mask but upper halves are not zero");
            else if (ymmh) t_info("ymm_area_when_init", "%s", cx_all_zero(ymmh, 16 * CX_NV) ? "zeroed" : "left as is");
            cx_check_xmm(x.ctx, a->vin, CX_NV, "low lanes");
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(xstate_getctx_mask_zero_request) {
    /* Requesting no extended feature: legacy parts still exact; the AVX outcome is recorded. */
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7300, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, 0)) {
            uint64_t m = cx_mask(x.ctx);
            t_info("mask_after_zero_request", "0x%llx", (unsigned long long)m);
            cx_check_xmm(x.ctx, a->vin, CX_NV, "low lanes");
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
            if ((m & T_XS_AVX) && ymmh) {
                char lab[48];
                for (int i = 0; i < CX_NV; i++) { snprintf(lab, sizeof lab, "ymm%d upper", i); CHECK_REG(lab, ymmh + 16 * i, a->vin[i] + 16, 16); }
            }
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(xstate_setctx_ymm_observed) {
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7400, 0);
    REQUIRE(a);
    uint8_t want[16][32];
    for (int i = 0; i < 16; i++) t_pattern_reg(want[i], 32, 0x7401, (uint32_t)i);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
            if (ymmh) {
                for (int i = 0; i < CX_NV; i++) { memcpy(cx_xmm(x.ctx, i), want[i], 16); memcpy(ymmh + 16 * i, want[i] + 16, 16); }
                CHECK(X.SetXStateFeaturesMask(x.ctx, T_XS_AVX | 3));
                x.ctx->ContextFlags = CX_XMMF | CONTEXT_XSTATE;
                CHECKF(SetThreadContext(w.h, x.ctx), "SetThreadContext(XSTATE) GLE=%lu", (unsigned long)GetLastError());
            } else CHECK(ymmh != NULL);
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        char lab[48];
        for (int i = 0; i < CX_NV; i++) { snprintf(lab, sizeof lab, "ymm%d set via XSTATE", i); CHECK_REG(lab, a->vout[i], want[i], 32); }
        for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(a->gout[i], cx_gpr_expect(a, i));
    }
    cx_free(a);
}

TEST(xstate_setctx_upper_only_one_register) {
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7500, 0);
    REQUIRE(a);
    uint8_t nv[16];
    t_pattern_reg(nv, 16, 0x7501, 9);
    int r = CX_NV - 1;     /* last register: ymm15 / ymm7 */
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
            if (ymmh) {
                memcpy(ymmh + 16 * r, nv, 16);
                CHECK(SetThreadContext(w.h, x.ctx));
            }
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < CX_NV; i++) {
            uint8_t want[32];
            memcpy(want, a->vin[i], 32);
            if (i == r) memcpy(want + 16, nv, 16);
            CHECK_REG("ymm after single-upper set", a->vout[i], want, 32);
        }
    }
    cx_free(a);
}

TEST(xstate_setctx_legacy_only_keeps_upper) {
    /* SetThreadContext without CONTEXT_XSTATE replaces XMM (low 128 bits) and must leave the YMM upper halves,
     * which a legacy CONTEXT does not describe, as they were. */
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7600, 0);
    REQUIRE(a);
    uint8_t lo[16][16];
    for (int i = 0; i < 16; i++) t_pattern_reg(lo[i], 16, 0x7601, (uint32_t)i);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_plain(CX_XMMF);
        if (c && GetThreadContext(w.h, c)) {
            for (int i = 0; i < CX_NV; i++) memcpy(cx_xmm(c, i), lo[i], 16);
            CHECK(SetThreadContext(w.h, c));
        }
        cx_free(c);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < CX_NV; i++) {
            uint8_t want[32];
            memcpy(want, lo[i], 16); memcpy(want + 16, a->vin[i] + 16, 16);
            CHECK_REG("ymm after legacy-only set", a->vout[i], want, 32);
        }
    }
    cx_free(a);
}

TEST(xstate_setctx_mask_cleared_note) {
    /* AVX area modified but AVX cleared in the header mask: Windows documents the mask as the set of features
     * present; whether a cleared feature becomes init state or stays is recorded, and only "one of the two"
     * is checked. */
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7700, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
            if (ymmh) memset(ymmh, 0x5A, 16 * CX_NV);
            CHECK(X.SetXStateFeaturesMask(x.ctx, 3));
            CHECK(SetThreadContext(w.h, x.ctx));
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        int kept = 1, zero = 1;
        for (int i = 0; i < CX_NV; i++) {
            if (memcmp(a->vout[i] + 16, a->vin[i] + 16, 16)) kept = 0;
            if (!cx_all_zero(a->vout[i] + 16, 16)) zero = 0;
            CHECK_MEM(a->vout[i], a->vin[i], 16);
        }
        uint8_t five[16]; memset(five, 0x5A, 16);
        CHECKF(memcmp(a->vout[0] + 16, five, 16) != 0, "area of a feature cleared in the mask was applied");
        CHECKF(kept || zero, "upper halves neither preserved nor init state");
        t_info("cleared_mask_effect", "%s", kept ? "preserved" : zero ? "init (zero)" : "other");
    }
    cx_free(a);
}

TEST(xstate_two_workers_isolated) {
    CX_NEED_AVX();
    cx_args *a1 = cx_args_new(0x7810, 0), *a2 = cx_args_new(0x7820, 0);
    REQUIRE(a1 && a2);
    cx_worker w1, w2;
    int ok1 = cx_worker_start(&w1, a1, T_VK_YMM);
    int ok2 = ok1 && cx_worker_start(&w2, a2, T_VK_YMM);
    if (!ok1) memset(&w2, 0, sizeof w2);
    uint8_t nv[16];
    t_pattern_reg(nv, 16, 0x7830, 2);
    int set1 = 0;
    if (ok1 && ok2 && cx_suspend(&w1) && cx_suspend(&w2)) {
        cx_xc x1 = { 0 }, x2 = { 0 };
        if (cx_get_x(&w1, &x1, CX_STD | CONTEXT_XSTATE, T_XS_AVX) && cx_get_x(&w2, &x2, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            uint8_t *h1 = cx_locate(x1.ctx, CX_XS_AVX, NULL), *h2 = cx_locate(x2.ctx, CX_XS_AVX, NULL);
            if (h1 && h2) {
                for (int i = 0; i < CX_NV; i++) {
                    CHECK_REG("worker1 ymm upper", h1 + 16 * i, a1->vin[i] + 16, 16);
                    CHECK_REG("worker2 ymm upper", h2 + 16 * i, a2->vin[i] + 16, 16);
                }
                memcpy(h1 + 16 * 2, nv, 16);
                set1 = SetThreadContext(w1.h, x1.ctx);
                CHECKF(set1, "SetThreadContext(worker1) GLE=%lu", (unsigned long)GetLastError());
            }
        }
        cx_xc_free(&x1); cx_xc_free(&x2);
    } else CHECKF(ok1 && ok2, "workers did not start");
    cx_release(&w1);
    if (ok1) cx_release(&w2);
    if (w1.done) for (int i = 0; i < CX_NV; i++) {
        uint8_t want[32]; memcpy(want, a1->vin[i], 32);
        if (i == 2 && set1) memcpy(want + 16, nv, 16);
        CHECK_REG("worker1 ymm", a1->vout[i], want, 32);
    }
    if (w2.done) for (int i = 0; i < CX_NV; i++) CHECK_REG("worker2 untouched", a2->vout[i], a2->vin[i], 32);
    cx_free(a1); cx_free(a2);
}

TEST(xstate_suspend_get_resume_loop) {
    /* 200 x (suspend, XSTATE get, resume) on a running YMM worker: every snapshot exact, final state intact. */
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7900, 0);
    REQUIRE(a);
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    int bad = 0, n = 0;
    double t0 = t_now_ms();
    for (int it = 0; ok && it < 200 && bad < 3 && t_now_ms() - t0 < 4000; it++, n++) {
        if (SuspendThread(w.h) != 0) { CHECK(0); bad++; break; }
        x.ctx->ContextFlags = CX_STD | CONTEXT_XSTATE;
        X.SetXStateFeaturesMask(x.ctx, T_XS_AVX);
        if (!GetThreadContext(w.h, x.ctx)) { CHECKF(0, "iteration %d: GetThreadContext GLE=%lu", it, (unsigned long)GetLastError()); bad++; }
        else {
            uint8_t *h = cx_locate(x.ctx, CX_XS_AVX, NULL);
            for (int i = 0; i < CX_NV; i++) {
                if (memcmp(cx_xmm(x.ctx, i), a->vin[i], 16) || !h || memcmp(h + 16 * i, a->vin[i] + 16, 16)) {
                    CHECKF(0, "iteration %d: ymm%d snapshot differs", it, i); bad++; break;
                }
            }
            uintptr_t ip = cx_ip(x.ctx);
            if (ip < (uintptr_t)cx_ymm_loop || ip > (uintptr_t)cx_ymm_exit) { CHECKF(0, "iteration %d: IP %p outside loop", it, (void *)ip); bad++; }
        }
        ResumeThread(w.h);
        if (it % 16 == 0) Sleep(0);
    }
    if (!ok) CHECKF(ok, "worker did not start");
    t_info("iterations", "%d", n);
    cx_release(&w);
    if (w.done) for (int i = 0; i < CX_NV; i++) CHECK_REG("ymm after 200 suspend/get/resume", a->vout[i], a->vin[i], 32);
    cx_xc_free(&x);
    cx_free(a);
}

TEST(xstate_copycontext_from_thread) {
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x7A00, 0);
    REQUIRE(a);
    cx_xc d;
    CX_XC_REQUIRE(&d, CX_STD | CONTEXT_XSTATE);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        cx_xc s = { 0 };
        if (cx_get_x(&w, &s, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            CHECK(X.CopyContext(d.ctx, CX_STD | CONTEXT_XSTATE, s.ctx));
            CHECK_EQ(cx_mask(d.ctx), cx_mask(s.ctx));
            /* ntdll!RtlpCopyXStateChunk: XSTATE_BV = src & enabled; XCOMP_BV = src & (enabled|3) | bit 63 when the
             * OS uses the compacted format, else 0 -> for a valid source both header words are copied as is */
            uint64_t *xs = cx_xs_hdr(s.ctx), *xd = cx_xs_hdr(d.ctx);
            CHECKF(xd[0] == xs[0] && xd[1] == xs[1], "XSAVE header after CopyContext %llx/%llx, source %llx/%llx",
                   (unsigned long long)xd[0], (unsigned long long)xd[1], (unsigned long long)xs[0], (unsigned long long)xs[1]);
            uint8_t *hd = cx_locate(d.ctx, CX_XS_AVX, NULL), *hs = cx_locate(s.ctx, CX_XS_AVX, NULL);
            if (hd && hs) CHECK_MEM(hd, hs, 16 * CX_NV);
            CHECK_MEM(cx_xmm(d.ctx, 0), cx_xmm(s.ctx, 0), 16 * CX_NV);
            CHECK_EQ(cx_ip(d.ctx), cx_ip(s.ctx));
            CHECK_EQ(cx_sp(d.ctx), cx_sp(s.ctx));
            for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(cx_gpr(d.ctx, i), cx_gpr(s.ctx, i));
            int mx_ok = 1;
#ifdef T_X64
            mx_ok = d.ctx->MxCsr == s.ctx->MxCsr;
            CX_CHECK_WGAP("CopyContext does not copy CONTEXT.MxCsr", mx_ok, "CONTEXT.MxCsr 0x%lx after CopyContext, source 0x%lx",
                          (unsigned long)d.ctx->MxCsr, (unsigned long)s.ctx->MxCsr);
#endif
            /* the copy is a valid SetThreadContext source (only when it is a faithful copy: a garbage MxCsr
             * would be a different test, see setctx_mxcsr_reserved_bits) */
            /* RtlpCopyLegacyContext{Amd64,X86} copy only the defined FXSAVE bytes (x86: the first 0x120); the
             * rest of the destination keeps its old (here: sentinel) contents. Reserved FXSAVE bytes must not
             * matter to SetThreadContext either, but that is setctx_fxsave_reserved_bytes_ignored (Wine keeps
             * Linux's FP_XSTATE_MAGIC there): take them from the source so this case is only about the copy. */
            memcpy(cx_fx(d.ctx) + CX_FX_DEFINED, cx_fx(s.ctx) + CX_FX_DEFINED, 512 - CX_FX_DEFINED);
            if (hd && mx_ok) { uint8_t nv[16]; t_pattern_reg(nv, 16, 0x7A01, 4); memcpy(hd + 16 * 4, nv, 16);
                      CHECK(SetThreadContext(w.h, d.ctx)); memcpy(a->vin[4] + 16, nv, 16); }
        }
        cx_xc_free(&s);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) for (int i = 0; i < CX_NV; i++) CHECK_REG("ymm after set from CopyContext copy", a->vout[i], a->vin[i], 32);
    cx_xc_free(&d);
    cx_free(a);
}

/* ======================================================================================================
 * 8. CONTEXT_XSTATE: AVX-512
 * ====================================================================================================== */
static int cx_zmm_kbytes(void) { return cpu_has(F_AVX512BW) ? 8 : 2; }

TEST(xstate_getctx_zmm) {
    CX_NEED_ZMM();
    cx_args *a = cx_args_new(0x8100, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_ZMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        uint64_t req = T_XS_AVX | T_XS_OPMASK | T_XS_ZMM_HI256 | T_XS_HI16_ZMM;
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, req)) {
            uint64_t m = cx_mask(x.ctx);
            CHECKF((m & req) == req, "mask 0x%llx lacks AVX-512 components", (unsigned long long)m);
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL), *km = cx_locate(x.ctx, CX_XS_KMASK, NULL);
            uint8_t *zh = cx_locate(x.ctx, CX_XS_ZMM_H, NULL), *hi = cx_locate(x.ctx, CX_XS_HI16, NULL);
            CHECK(ymmh && km && zh);
            char lab[48];
            int nlo = CX_NZ < 16 ? CX_NZ : 16;
            for (int i = 0; i < nlo; i++) {
                snprintf(lab, sizeof lab, "zmm%d bits 0-127", i);   CHECK_REG(lab, cx_xmm(x.ctx, i), a->zin[i], 16);
                if (ymmh) { snprintf(lab, sizeof lab, "zmm%d bits 128-255", i); CHECK_REG(lab, ymmh + 16 * i, a->zin[i] + 16, 16); }
                if (zh) { snprintf(lab, sizeof lab, "zmm%d bits 256-511", i); CHECK_REG(lab, zh + 32 * i, a->zin[i] + 32, 32); }
            }
#ifdef T_X64
            CHECK(hi != NULL);
            if (hi) for (int i = 16; i < 32; i++) { snprintf(lab, sizeof lab, "zmm%d (Hi16_ZMM)", i); CHECK_REG(lab, hi + 64 * (i - 16), a->zin[i], 64); }
#else
            (void)hi;
#endif
            if (km) for (int i = 0; i < 8; i++) { snprintf(lab, sizeof lab, "k%d", i); CHECK_REG(lab, km + 8 * i, &a->kin[i], (size_t)cx_zmm_kbytes()); }
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) for (int i = 0; i < CX_NZ; i++) CHECK_REG("zmm after get", a->zout[i], a->zin[i], 64);
    cx_free(a);
}

TEST(xstate_setctx_zmm_observed) {
    CX_NEED_ZMM();
    cx_args *a = cx_args_new(0x8200, 0);
    REQUIRE(a);
    uint8_t z3[32], z20[64]; uint64_t k2 = t_pattern_word(0x8201, 102, 0);
    if (!cpu_has(F_AVX512BW)) k2 &= 0xFFFF;
    t_pattern_reg(z3, 32, 0x8201, 3);
    t_pattern_reg(z20, 64, 0x8201, 20);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_ZMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        uint64_t req = T_XS_AVX | T_XS_OPMASK | T_XS_ZMM_HI256 | T_XS_HI16_ZMM;
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, req)) {
            uint8_t *km = cx_locate(x.ctx, CX_XS_KMASK, NULL), *zh = cx_locate(x.ctx, CX_XS_ZMM_H, NULL);
            uint8_t *hi = cx_locate(x.ctx, CX_XS_HI16, NULL);
            if (km && zh) {
                memcpy(zh + 32 * 3, z3, 32);
                memcpy(km + 8 * 2, &k2, 8);
#ifdef T_X64
                if (hi) memcpy(hi + 64 * 4, z20, 64);
#else
                (void)hi;
#endif
                CHECK(X.SetXStateFeaturesMask(x.ctx, req | 3));
                CHECKF(SetThreadContext(w.h, x.ctx), "SetThreadContext(AVX-512) GLE=%lu", (unsigned long)GetLastError());
                memcpy(a->zin[3] + 32, z3, 32);
                a->kin[2] = k2;
#ifdef T_X64
                if (hi) memcpy(a->zin[20], z20, 64);
#endif
            } else CHECK(km && zh);
        }
        cx_xc_free(&x);
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < CX_NZ; i++) CHECK_REG("zmm after set", a->zout[i], a->zin[i], 64);
        for (int i = 0; i < 8; i++) CHECK_REG("k after set", &a->kout[i], &a->kin[i], (size_t)cx_zmm_kbytes());
    }
    cx_free(a);
}

/* ======================================================================================================
 * 9. Error paths: handles and access rights
 * ====================================================================================================== */
TEST(err_invalid_handles) {
    CONTEXT *c = cx_plain(CX_STD);
    REQUIRE(c);
    static const struct { uintptr_t h; const char *what; } hs[] = {
        { 0, "NULL" }, { 0x1234, "never-allocated value" }, { (uintptr_t)-1, "current-process pseudo handle" },
        { 0x7FFFFFF0, "large bogus value" },
    };
    for (size_t i = 0; i < T_COUNTOF(hs); i++) {
        memset(c, CX_SENT, sizeof *c); c->ContextFlags = CX_STD;
        SetLastError(0);
        BOOL r = GetThreadContext((HANDLE)hs[i].h, c);
        DWORD e = GetLastError();
        CHECKF(!r, "GetThreadContext(%s) succeeded", hs[i].what);
        CHECKF(e == ERROR_INVALID_HANDLE, "GetThreadContext(%s) GLE %lu want ERROR_INVALID_HANDLE", hs[i].what, (unsigned long)e);
        CHECKF(cx_gprs_untouched(c, 1), "GetThreadContext(%s) wrote registers on failure", hs[i].what);
        SetLastError(0);
        r = SetThreadContext((HANDLE)hs[i].h, c);
        e = GetLastError();
        CHECKF(!r, "SetThreadContext(%s) succeeded", hs[i].what);
        CHECKF(e == ERROR_INVALID_HANDLE, "SetThreadContext(%s) GLE %lu want ERROR_INVALID_HANDLE", hs[i].what, (unsigned long)e);
    }
    /* a closed thread handle */
    HANDLE h = OpenThread(THREAD_ALL_ACCESS, FALSE, GetCurrentThreadId());
    REQUIRE(h);
    CloseHandle(h);
    SetLastError(0);
    CHECK(!GetThreadContext(h, c));
    CHECK_EQ(GetLastError(), ERROR_INVALID_HANDLE);
    /* a valid handle of the wrong object type (event) */
    HANDLE ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    REQUIRE(ev);
    SetLastError(0);
    CHECK(!GetThreadContext(ev, c));
    CHECKF(GetLastError() == ERROR_INVALID_HANDLE, "event handle: GLE %lu want ERROR_INVALID_HANDLE", (unsigned long)GetLastError());
    SetLastError(0);
    CHECK(!SetThreadContext(ev, c));
    CHECKF(GetLastError() == ERROR_INVALID_HANDLE, "event handle (set): GLE %lu", (unsigned long)GetLastError());
    CloseHandle(ev);
    cx_free(c);
}

TEST(err_access_denied_get) {
    cx_args *a = cx_args_new(0x9100, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        HANDLE lim = OpenThread(THREAD_SUSPEND_RESUME | THREAD_QUERY_LIMITED_INFORMATION | THREAD_SET_CONTEXT, FALSE, w.tid);
        CHECKF(lim != NULL, "OpenThread(limited) GLE=%lu", (unsigned long)GetLastError());
        if (lim) {
            CONTEXT *c = cx_plain(CX_STD);
            SetLastError(0);
            BOOL r = GetThreadContext(lim, c);
            CHECKF(!r, "GetThreadContext succeeded without THREAD_GET_CONTEXT");
            CHECK_EQ(GetLastError(), ERROR_ACCESS_DENIED);
            CHECKF(cx_gprs_untouched(c, 1), "denied GetThreadContext wrote registers");
            cx_free(c);
            CloseHandle(lim);
        }
        HANDLE g = OpenThread(THREAD_GET_CONTEXT, FALSE, w.tid);
        CHECKF(g != NULL, "OpenThread(THREAD_GET_CONTEXT) GLE=%lu", (unsigned long)GetLastError());
        if (g) {
            CONTEXT *c = cx_plain(CX_STD);
            CHECKF(GetThreadContext(g, c), "THREAD_GET_CONTEXT alone is sufficient: GLE=%lu", (unsigned long)GetLastError());
            cx_check_gprs(c, a, 1, "minimal-access handle");
            cx_free(c);
            CloseHandle(g);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
}

TEST(err_access_denied_set) {
    cx_args *a = cx_args_new(0x9200, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        HANDLE g = OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, FALSE, w.tid);
        CHECKF(g != NULL, "OpenThread GLE=%lu", (unsigned long)GetLastError());
        if (g) {
            CONTEXT *c = cx_plain(CX_STD);
            if (GetThreadContext(g, c)) {
                for (int i = 0; i < CX_NV; i++) memset(cx_xmm(c, i), 0x77, 16);
                cx_set_gpr(c, 0, 0x7777777777777777ull);
                SetLastError(0);
                BOOL r = SetThreadContext(g, c);
                CHECKF(!r, "SetThreadContext succeeded without THREAD_SET_CONTEXT");
                CHECK_EQ(GetLastError(), ERROR_ACCESS_DENIED);
            } else CHECKF(0, "GetThreadContext(get-only handle) GLE=%lu", (unsigned long)GetLastError());
            cx_free(c);
            CloseHandle(g);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) {
        for (int i = 0; i < CX_NV; i++) CHECK_REG("xmm after denied set", a->vout[i], a->vin[i], 16);
        for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(a->gout[i], cx_gpr_expect(a, i));
    }
    cx_free(a);
}

TEST(err_xstate_chunk_too_short) {
    /* CONTEXT_EX.XState.Length shrunk to the header only while AVX is requested: the call may fail
     * (ERROR_MORE_DATA on Windows 10) but must never write the AVX area beyond the declared length. */
    CX_NEED_AVX();
    cx_args *a = cx_args_new(0x9300, 0);
    REQUIRE(a);
    cx_xc x;
    CX_XC_REQUIRE(&x, CX_STD | CONTEXT_XSTATE);
    CHECK(X.SetXStateFeaturesMask(x.ctx, T_XS_AVX));
    uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
    REQUIRE(ymmh);
    LONG *ex = (LONG *)((uint8_t *)x.ctx + sizeof(CONTEXT));   /* CONTEXT_EX: All, Legacy, XState {Offset, Length} */
    t_info("context_ex", "all %ld/%lu legacy %ld/%lu xstate %ld/%lu", (long)ex[0], (unsigned long)ex[1], (long)ex[2],
           (unsigned long)ex[3], (long)ex[4], (unsigned long)ex[5]);
    CHECKF(ex[4] > 0 && (uint8_t *)ex + ex[4] < ymmh, "CONTEXT_EX.XState offset %ld does not precede the AVX area", (long)ex[4]);
    ex[5] = 64;
    memset(ymmh, CX_SENT, 256);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_YMM);
    if (ok && cx_suspend(&w)) {
        SetLastError(0);
        BOOL r = GetThreadContext(w.h, x.ctx);
        DWORD e = GetLastError();
        t_info("short_chunk_result", "%d gle=%lu", r, (unsigned long)e);
        CX_CHECK_WGAP("GetThreadContext ignores CONTEXT_EX.XState.Length", cx_all_sent(ymmh, 256),
                      "GetThreadContext wrote the AVX area beyond CONTEXT_EX.XState.Length");
        if (!r) {
            CHECKF(e != 0, "short XSTATE chunk: failed without an error code");
            if (e != ERROR_MORE_DATA) t_note("short XSTATE chunk: GLE %lu (Windows 10 path: STATUS_BUFFER_OVERFLOW -> ERROR_MORE_DATA)", (unsigned long)e);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    if (w.done) for (int i = 0; i < CX_NV; i++) CHECK_REG("ymm", a->vout[i], a->vin[i], 32);
    cx_xc_free(&x);
    cx_free(a);
}

/* Child: bad CONTEXT pointers against a suspended worker. Prints R=<ok> GLE=<err>. */
CHILD(cx_badptr) {
    const char *v = argc > 1 ? argv[1] : "";
    cx_resolve();
    cx_args *a = cx_args_new(0x9400, 0);
    cx_worker w;
    if (!a || !cx_worker_start(&w, a, T_VK_XMM)) { printf("R=-1 GLE=0 start failed\n"); return 3; }
    SuspendThread(w.h);
    BOOL r = FALSE;
    DWORD e = 0;
    void *base = NULL;
    if (!strcmp(v, "null_get")) { SetLastError(0); r = GetThreadContext(w.h, NULL); e = GetLastError(); }
    else if (!strcmp(v, "null_set")) { SetLastError(0); r = SetThreadContext(w.h, NULL); e = GetLastError(); }
    else if (!strcmp(v, "guard_plain") || !strcmp(v, "guard_xstate")) {
        CONTEXT *c = (CONTEXT *)t_guarded_alloc(sizeof(CONTEXT), &base);
        if (!c) { printf("R=-1 GLE=0 alloc\n"); return 3; }
        memset(c, CX_SENT, sizeof *c);
        c->ContextFlags = !strcmp(v, "guard_plain") ? CX_STD : (CX_STD | CONTEXT_XSTATE);
        SetLastError(0);
        r = GetThreadContext(w.h, c);
        e = GetLastError();
        if (r && !strcmp(v, "guard_plain")) {
            int good = 1;
            for (int i = 0; i < CX_NGPR; i++) if (cx_gpr(c, i) != cx_gpr_expect(a, i)) good = 0;
            for (int i = 0; i < CX_NV; i++) if (memcmp(cx_xmm(c, i), a->vin[i], 16)) good = 0;
            printf("VALUES=%s\n", good ? "ok" : "bad");
        }
    } else { printf("R=-1 GLE=0 unknown variant\n"); return 3; }
    printf("R=%d GLE=%lu\n", r ? 1 : 0, (unsigned long)e);
    cx_worker_finish(&w);
    t_guarded_free(base);
    return 0;
}

static void cx_run_badptr(const char *variant, int expect_ok) {
    static t_child_result r;
    REQUIRE(t_child_run("cx_badptr", variant, 20000, &r));
    CHECKF(!r.timed_out, "child timed out");
    if (r.exc_code) {
        if (t_is_wine()) XFAIL("wine %s: %s raises exception 0x%lx instead of returning an error", t_wine_version(), variant,
                               (unsigned long)r.exc_code);
        CHECKF(0, "%s: the API raised exception 0x%lx in the caller instead of returning an error", variant, (unsigned long)r.exc_code);
        return;
    }
    char val[128];
    REQUIREF(t_child_find(&r, "R=", val, sizeof val), "no result line; output: %.300s", r.out);
    int ok = -1; unsigned long gle = 0;
    sscanf(val, "%d GLE=%lu", &ok, &gle);
    t_info("result", "R=%d GLE=%lu", ok, gle);
    REQUIREF(ok >= 0, "child setup failed: %s", val);
    if (expect_ok) {
        CHECKF(ok == 1, "%s: call failed, GLE=%lu", variant, gle);
        if (t_child_find(&r, "VALUES=", val, sizeof val)) CHECKF(!strcmp(val, "ok"), "%s: wrong register values", variant);
    } else {
        CHECKF(ok == 0, "%s: call succeeded", variant);
        CHECKF(gle != 0, "%s: failed without an error code", variant);
        if (gle != ERROR_NOACCESS) t_note("%s: GLE %lu (Windows 10: ERROR_NOACCESS 998)", variant, gle);
    }
    CHECK_EQ(r.exit_code, 0);
}
TEST(err_getctx_null_context) { cx_run_badptr("null_get", 0); }
TEST(err_setctx_null_context) { cx_run_badptr("null_set", 0); }
TEST(err_getctx_plain_context_at_guard_page) { cx_run_badptr("guard_plain", 1); }
TEST(err_getctx_xstate_flag_without_chunk) {
    if (!cx_avx_ok()) SKIP("no extended state enabled: CONTEXT_XSTATE is not meaningful");
    cx_run_badptr("guard_xstate", 0);
}

TEST(getctx_current_thread_note) {
    /* Not a valid way to obtain the running thread's context (documented); only must not crash. */
    CONTEXT *c = cx_plain(CX_STD);
    REQUIRE(c);
    SetLastError(0);
    BOOL r = GetThreadContext(GetCurrentThread(), c);
    t_info("self_get", "%d gle=%lu", r, (unsigned long)GetLastError());
    if (r) CHECKF(c->SegCs == cx_own_cs(), "self context SegCs 0x%x", (unsigned)c->SegCs);
    cx_free(c);
}

/* ======================================================================================================
 * 10. RtlCaptureContext
 * ====================================================================================================== */
static void cx_cap_prepare(cx_cap *k, CONTEXT *c, uint32_t tag) {
    memset(k, 0, sizeof *k);
    for (int i = 0; i < 16; i++) t_pattern_reg(k->xin[i], 16, tag, (uint32_t)i);
    for (int i = 0; i < 8; i++) k->gin[i] = t_pattern_word(tag, 50 + (uint32_t)i, 0);
    k->fn = (uint64_t)(uintptr_t)X.RtlCaptureContext;
    k->ctx = (uint64_t)(uintptr_t)c;
    k->mxcsr = 0x3F80;
}
/* noinline: x86 Windows RtlCaptureContext reports the caller frame (cx_capture's return address), so all
 * captures must come from this one call site to be comparable */
static __attribute__((noinline)) void cx_cap_run(cx_cap *k) {
    unsigned mx = t_get_mxcsr();
    cx_capture(k);
    t_set_mxcsr(mx);
}
#define CX_NEED_CAPTURE() do { cx_resolve(); if (!X.RtlCaptureContext) SKIP("RtlCaptureContext not exported"); } while (0)

TEST(rtlcapture_control_integer) {
    CX_NEED_CAPTURE();
    CONTEXT *c = cx_plain(0);
    REQUIRE(c);
    static cx_cap k;
    cx_cap_prepare(&k, c, 0xA100);
    cx_cap_run(&k);
    DWORD need = CONTEXT_CONTROL | CONTEXT_INTEGER;
    CHECKF((c->ContextFlags & need) == need, "ContextFlags 0x%lx lacks CONTROL|INTEGER", (unsigned long)c->ContextFlags);
    t_info("flags", "0x%lx", (unsigned long)c->ContextFlags);
    /* selectors: low 16 bits only - RtlCaptureContext stores them with 16-bit moves, and the x86 CONTEXT
     * fields are DWORDs whose upper half keeps the caller's buffer contents */
    CHECKF((uint16_t)c->SegCs == cx_own_cs(), "SegCs 0x%x want 0x%x", (unsigned)c->SegCs, cx_own_cs());
    CHECKF((uint16_t)c->SegSs == cx_own_ss(), "SegSs 0x%x want 0x%x", (unsigned)c->SegSs, cx_own_ss());
    CHECKF(c->EFlags & 0x2, "EFlags 0x%lx", (unsigned long)c->EFlags);
    CHECKF(c->EFlags & 0x200, "EFlags 0x%lx IF clear", (unsigned long)c->EFlags);
#ifdef T_X64
    CHECKF(c->Rip == (DWORD64)(uintptr_t)cx_cap_ret, "Rip %p want return address %p", (void *)(uintptr_t)c->Rip, (void *)cx_cap_ret);
    CHECKF(c->Rsp == k.sp_after, "Rsp 0x%llx want 0x%llx (RSP after the call)", (unsigned long long)c->Rsp, (unsigned long long)k.sp_after);
    static const char *nm[] = { "rbx", "rbp", "rsi", "rdi", "r12", "r13", "r14", "r15" };
    DWORD64 got[8] = { c->Rbx, c->Rbp, c->Rsi, c->Rdi, c->R12, c->R13, c->R14, c->R15 };
    for (int i = 0; i < 8; i++) CHECKF(got[i] == k.gin[i], "%s 0x%llx want 0x%llx", nm[i], (unsigned long long)got[i], (unsigned long long)k.gin[i]);
    if (c->Rcx != (DWORD64)(uintptr_t)c) t_note("Rcx 0x%llx is not the CONTEXT argument", (unsigned long long)c->Rcx);
    /* ntdll!RtlCaptureContext (x64) stores all six selectors and reports CONTEXT_SEGMENTS (0x10000F) */
    if ((c->ContextFlags & CONTEXT_SEGMENTS) == CONTEXT_SEGMENTS)
        CHECKF(c->SegDs == cx_own_ds() && c->SegEs == cx_own_es() && c->SegFs == cx_own_fs() && c->SegGs == cx_own_gs(),
               "captured DS/ES/FS/GS %x/%x/%x/%x want %x/%x/%x/%x", c->SegDs, c->SegEs, c->SegFs, c->SegGs,
               cx_own_ds(), cx_own_es(), cx_own_fs(), cx_own_gs());
    else t_note("ContextFlags 0x%lx without CONTEXT_SEGMENTS", (unsigned long)c->ContextFlags);
#else
    CHECK_EQ(c->Ebx, (DWORD)k.gin[0]);
    CHECK_EQ(c->Esi, (DWORD)k.gin[2]);
    CHECK_EQ(c->Edi, (DWORD)k.gin[3]);
    int direct = c->Eip == (DWORD)(uintptr_t)cx_cap_ret && c->Esp == (DWORD)k.sp_after && c->Ebp == (DWORD)k.own_bp;
    int frame = c->Eip == (DWORD)k.frame_ip && c->Esp == (DWORD)k.frame_sp && c->Ebp == (DWORD)k.frame_bp;
    CHECKF(direct || frame, "Eip/Esp/Ebp %08lx/%08lx/%08lx match neither the call site (%08lx/%08lx/%08lx) nor the caller frame (%08lx/%08lx/%08lx)",
           (unsigned long)c->Eip, (unsigned long)c->Esp, (unsigned long)c->Ebp,
           (unsigned long)(uintptr_t)cx_cap_ret, (unsigned long)k.sp_after, (unsigned long)k.own_bp,
           (unsigned long)k.frame_ip, (unsigned long)k.frame_sp, (unsigned long)k.frame_bp);
    t_info("x86_capture_style", "%s", direct ? "call site" : frame ? "caller frame" : "other");
    CHECKF((uint16_t)c->SegDs == cx_own_ds() && (uint16_t)c->SegEs == cx_own_es(), "SegDs/SegEs 0x%lx/0x%lx", (unsigned long)c->SegDs, (unsigned long)c->SegEs);
#endif
    cx_free(c);
}

TEST(rtlcapture_fp_state) {
    CX_NEED_CAPTURE();
#ifdef T_X64
    CONTEXT *c = cx_plain(0);
    REQUIRE(c);
    static cx_cap k;
    cx_cap_prepare(&k, c, 0xA200);
    cx_cap_run(&k);
    /* The x64 unwinder restores XMM6-15 and MXCSR from captured contexts: they must be captured. */
    CHECKF((c->ContextFlags & CONTEXT_FLOATING_POINT) == CONTEXT_FLOATING_POINT, "ContextFlags 0x%lx lacks FLOATING_POINT",
           (unsigned long)c->ContextFlags);
    char lab[48];
    for (int i = 6; i < 16; i++) { snprintf(lab, sizeof lab, "xmm%d (nonvolatile)", i); CHECK_REG(lab, cx_xmm(c, i), k.xin[i], 16); }
    int vol_ok = 1;
    for (int i = 0; i < 6; i++) if (memcmp(cx_xmm(c, i), k.xin[i], 16)) vol_ok = 0;
    if (!vol_ok) t_note("volatile XMM0-5 not captured as loaded (allowed by the ABI)");
    CHECK_EQ(c->MxCsr, 0x3F80);
    CHECK_EQ(c->FltSave.MxCsr, 0x3F80);
    t_info("fltsave_fcw", "0x%x", c->FltSave.ControlWord);
    cx_free(c);
#else
    SKIP("x86 RtlCaptureContext captures CONTEXT_CONTROL|INTEGER|SEGMENTS only");
#endif
}

TEST(rtlcapture_repeatable_and_bounded) {
    CX_NEED_CAPTURE();
    /* Two captures at the same site agree; the capture writes nothing beyond sizeof(CONTEXT). */
    void *base = NULL;
    CONTEXT *c = (CONTEXT *)t_guarded_alloc(sizeof(CONTEXT), &base);
    REQUIRE(c);
    CONTEXT *d = cx_plain(0);
    REQUIRE(d);
    static cx_cap k1, k2;
    memset(c, CX_SENT, sizeof *c);
    cx_cap_prepare(&k1, c, 0xA300);
    cx_cap_run(&k1);
    cx_cap_prepare(&k2, d, 0xA300);
    cx_cap_run(&k2);
    CHECK_EQ(cx_ip(c), cx_ip(d));
    CHECK_EQ(cx_sp(c), cx_sp(d));
    CHECK_EQ(c->ContextFlags, d->ContextFlags);
    /* only the registers cx_capture loads (nonvolatile); volatile ones hold whatever the C caller left */
#ifdef T_X64
    static const int nvi[] = { 0, 3, 4, 5, 9, 10, 11, 12 };    /* rbx rsi rdi rbp r12-r15 */
#else
    static const int nvi[] = { 0, 2, 3 };                      /* ebx esi edi */
#endif
    for (size_t j = 0; j < T_COUNTOF(nvi); j++) {
        int i = nvi[j];
        CHECKF(cx_gpr(c, i) == cx_gpr(d, i), "%s 0x%llx vs 0x%llx", cx_gpr_name(i), (unsigned long long)cx_gpr(c, i),
               (unsigned long long)cx_gpr(d, i));
    }
    t_guarded_free(base);
    cx_free(d);
}

/* ======================================================================================================
 * 10b. Target blocked in a kernel wait or system call
 *
 * Every other worker in this file spins in user mode. Here the worker loads its registers in one asm routine
 * (cx_block_xmm / cx_block_ymm), calls a blocking function with nothing compiled in between, and stores its
 * registers right after the call returns. Blocking calls:
 *   wait_nt    ntdll!NtWaitForSingleObject(event)      direct: the registers reach the ntdll stub untouched
 *   delay_nt   ntdll!NtDelayExecution(FALSE, 1.5 s)    direct
 *   read_nt    ntdll!NtReadFile(anonymous pipe)        direct (9 arguments: exercises stack arguments)
 *   wait_k32   kernel32!WaitForSingleObject(event)     through kernelbase (its code may use any register)
 *   read_k32   kernel32!ReadFile(anonymous pipe)
 *   sleepex    kernel32!SleepEx(10 s, TRUE), woken by QueueUserAPC (user APC delivery + NtContinue)
 *
 * Contract pinned against the Windows 10 22H2 (19041.2965) reference (docker1 corpus explorer, 2026-11):
 *   - ntoskrnl!PspGetSetContextSpecialApc / PspGetSetContextInternal: Get/Set of another thread runs as a
 *     kernel APC in the target, also while it waits. FP get = FXSAVE of the live registers (the user state:
 *     the kernel does not use them), XSTATE = XSAVE. Nonvolatile GPRs and XMM6-15 are read/written through
 *     pointers to the user-mode save slots found by unwinding the kernel stack to the system-call trap frame.
 *   - ntoskrnl!KxContextToKframes: Rip/Rsp/EFlags and Rax-R11 into the KTRAP_FRAME, XMM0-5 into the trap
 *     frame, XMM6-15 into the exception frame, MxCsr (& KiMxCsrMask) into TrapFrame.MxCsr, x87 control word
 *     masked with 0x1F3F, then FXRSTOR of the CONTEXT's FltSave (with the kernel's MXCSR) and RtlXRestoreS of
 *     the XSTATE chunk into the live registers.
 *   - ntoskrnl!KiSystemServiceStart exit (RVA 0x40f500-0x40f87b): rbx/rdi/rsi from the trap frame, LDMXCSR
 *     TrapFrame.MxCsr, then the SYSRET path loads Rax = service status, Rcx = TrapFrame.Rip, R11 =
 *     TrapFrame.EFlags, Rsp = TrapFrame.Rsp, Rbp = TrapFrame.Rbp, zeroes edx and PXOR xmm0-xmm5. Before a user
 *     APC (0x40f556) it zeroes the trap frame's Rax-R11 and Xmm0-5 slots.
 *   => After SetThreadContext on a thread blocked in a system call, everything the x64 ABI makes nonvolatile
 *      (rbx rbp rsi rdi r12-r15, xmm6-15 low halves, MXCSR control, x87 CW) and Rip/Rsp take effect when the
 *      call returns. Volatile registers (rax rcx rdx r8-r11, xmm0-5, YMM upper halves, the x87 stack) are not
 *      restored on the fast return on Windows (xmm0-5 come back zero): observations (t_info) only, as the ABI
 *      allows the callee to clobber them anyway.
 *   - WOW64 (x86 build on 64-bit Windows), ntoskrnl FUN_140691cef (PspWow64GetContextThread path):
 *     PspGetContextThreadInternal(0x4010001F) of the live AMD64 state, then RtlpWow64GetContextOnAmd64:
 *     RtlpWow64CtxFromAmd64 converts FltSave into ExtendedRegisters (FXSAVE copy) and FloatSave
 *     (RtlFxToFnFrame); when the AMD64 SegCs != 0x23 (thread inside a system call) only control, integer,
 *     segments and ExtendedRegisters XMM0-5 are overlaid from the saved WOW64 CPU context. So an x86 thread
 *     captured inside a system call reports its own FCW / MXCSR / x87 stack, never an all-zero FP area.
 *   - Rip inside the stub: the x64 ntdll Nt* stubs execute SYSCALL inline (mov r10,rcx / mov eax,n / test /
 *     jne / syscall / ret, 0x18 bytes), so the user Rip of a blocked thread is inside the export it called.
 *     The x86 stubs call through Wow64Transition / KiFastSystemCall: Eip is inside ntdll, not necessarily the
 *     stub (recorded).
 * ====================================================================================================== */
#define CXK_FN     0
#define CXK_RET    8
#define CXK_NARGS  16
#define CXK_LANDED 20
#define CXK_ARGS   24
#define CXK_CALLSP 104
#define CXK_NARG_MAX 10
typedef struct cx_call {
    uint64_t fn;                  /* function called with nothing compiled in between */
    uint64_t ret;                 /* RAX/EAX after the call (or at the redirect landing) */
    uint32_t nargs;
    volatile LONG landed;         /* set by the redirect landing label */
    uint64_t args[CXK_NARG_MAX];  /* x64: args 0-3 in rcx rdx r8 r9, 4-9 on the stack; x86: stdcall, pushed */
    uint64_t call_sp;             /* RSP/ESP at the call instruction (arguments in place) */
} cx_call;
_Static_assert(offsetof(cx_call, ret) == CXK_RET && offsetof(cx_call, nargs) == CXK_NARGS &&
               offsetof(cx_call, landed) == CXK_LANDED && offsetof(cx_call, args) == CXK_ARGS &&
               offsetof(cx_call, call_sp) == CXK_CALLSP, "cx_call layout");

void cx_block_xmm(cx_args *a, cx_call *k) __asm__("cx_block_xmm");
void cx_block_ymm(cx_args *a, cx_call *k) __asm__("cx_block_ymm");
extern const char cx_blk_xmm_ret[] __asm__("cx_blk_xmm_ret");
extern const char cx_blk_xmm_land[] __asm__("cx_blk_xmm_land");
extern const char cx_blk_ymm_ret[] __asm__("cx_blk_ymm_ret");
extern const char cx_blk_ymm_land[] __asm__("cx_blk_ymm_land");

#ifdef T_X64
/* Frame: 8 pushes + 328 bytes. 0-31 shadow space, 32-79 stack arguments 4-9, 96 a, 104 k, 160-319 xmm6-15. */
#define CX_BLK(name, ret, land, LOADV, STOREV, TAIL) \
    ".globl " name "\n.def " name "; .scl 2; .type 32; .endef\n.seh_proc " name "\n" name ":\n" \
    "  pushq %rbp\n  .seh_pushreg %rbp\n  pushq %rbx\n  .seh_pushreg %rbx\n  pushq %rsi\n  .seh_pushreg %rsi\n" \
    "  pushq %rdi\n  .seh_pushreg %rdi\n  pushq %r12\n  .seh_pushreg %r12\n  pushq %r13\n  .seh_pushreg %r13\n" \
    "  pushq %r14\n  .seh_pushreg %r14\n  pushq %r15\n  .seh_pushreg %r15\n" \
    "  subq $328, %rsp\n  .seh_stackalloc 328\n" \
    "  movdqa %xmm6, 160(%rsp)\n  .seh_savexmm %xmm6, 160\n  movdqa %xmm7, 176(%rsp)\n  .seh_savexmm %xmm7, 176\n" \
    "  movdqa %xmm8, 192(%rsp)\n  .seh_savexmm %xmm8, 192\n  movdqa %xmm9, 208(%rsp)\n  .seh_savexmm %xmm9, 208\n" \
    "  movdqa %xmm10, 224(%rsp)\n  .seh_savexmm %xmm10, 224\n  movdqa %xmm11, 240(%rsp)\n  .seh_savexmm %xmm11, 240\n" \
    "  movdqa %xmm12, 256(%rsp)\n  .seh_savexmm %xmm12, 256\n  movdqa %xmm13, 272(%rsp)\n  .seh_savexmm %xmm13, 272\n" \
    "  movdqa %xmm14, 288(%rsp)\n  .seh_savexmm %xmm14, 288\n  movdqa %xmm15, 304(%rsp)\n  .seh_savexmm %xmm15, 304\n" \
    "  .seh_endprologue\n" \
    "  movq %rcx, 96(%rsp)\n  movq %rdx, 104(%rsp)\n" \
    CX_FPENV_LOAD("%rcx") CX_X87_LOAD("%rcx") LOADV \
    "  movq %rdx, %r11\n" \
    "  .irp i,4,5,6,7,8,9\n  movq " CXA(CXK_ARGS) "+\\i*8(%r11), %rax\n  movq %rax, \\i*8(%rsp)\n  .endr\n" \
    "  movq " CXA(CXO_GIN) "+0(%rcx), %rbx\n  movq " CXA(CXO_GIN) "+24(%rcx), %rsi\n" \
    "  movq " CXA(CXO_GIN) "+32(%rcx), %rdi\n  movq " CXA(CXO_GIN) "+40(%rcx), %rbp\n" \
    "  movq " CXA(CXO_GIN) "+72(%rcx), %r12\n  movq " CXA(CXO_GIN) "+80(%rcx), %r13\n" \
    "  movq " CXA(CXO_GIN) "+88(%rcx), %r14\n  movq " CXA(CXO_GIN) "+96(%rcx), %r15\n" \
    "  movq %rsp, " CXA(CXK_CALLSP) "(%r11)\n  movq %rsp, " CXA(CXO_SP) "(%rcx)\n  movl $1, " CXA(CXO_READY) "(%rcx)\n" \
    "  movq " CXA(CXK_ARGS) "+0(%r11), %rcx\n  movq " CXA(CXK_ARGS) "+8(%r11), %rdx\n" \
    "  movq " CXA(CXK_ARGS) "+16(%r11), %r8\n  movq " CXA(CXK_ARGS) "+24(%r11), %r9\n" \
    "  movq " CXA(CXK_FN) "(%r11), %rax\n  callq *%rax\n" \
    CX_GLOBAL(ret) \
    "  movq 104(%rsp), %r11\n  movq %rax, " CXA(CXK_RET) "(%r11)\n  movq 96(%rsp), %rax\n  jmp 9f\n" \
    CX_GLOBAL(land) \
    "  movq 104(%rsp), %r11\n  movl $1, " CXA(CXK_LANDED) "(%r11)\n  movq %rax, " CXA(CXK_RET) "(%r11)\n" \
    "  movq 96(%rsp), %rax\n" \
    "9:\n  cld\n" \
    "  movq %rbx, " CXA(CXO_GOUT) "+0(%rax)\n  movq %rcx, " CXA(CXO_GOUT) "+8(%rax)\n" \
    "  movq %rdx, " CXA(CXO_GOUT) "+16(%rax)\n  movq %rsi, " CXA(CXO_GOUT) "+24(%rax)\n" \
    "  movq %rdi, " CXA(CXO_GOUT) "+32(%rax)\n  movq %rbp, " CXA(CXO_GOUT) "+40(%rax)\n" \
    "  movq %r8, " CXA(CXO_GOUT) "+48(%rax)\n  movq %r9, " CXA(CXO_GOUT) "+56(%rax)\n" \
    "  movq %r10, " CXA(CXO_GOUT) "+64(%rax)\n  movq %r12, " CXA(CXO_GOUT) "+72(%rax)\n" \
    "  movq %r13, " CXA(CXO_GOUT) "+80(%rax)\n  movq %r14, " CXA(CXO_GOUT) "+88(%rax)\n" \
    "  movq %r15, " CXA(CXO_GOUT) "+96(%rax)\n" \
    STOREV CX_X87_STORE("%rax") TAIL \
    "  movdqa 160(%rsp), %xmm6\n  movdqa 176(%rsp), %xmm7\n  movdqa 192(%rsp), %xmm8\n  movdqa 208(%rsp), %xmm9\n" \
    "  movdqa 224(%rsp), %xmm10\n  movdqa 240(%rsp), %xmm11\n  movdqa 256(%rsp), %xmm12\n  movdqa 272(%rsp), %xmm13\n" \
    "  movdqa 288(%rsp), %xmm14\n  movdqa 304(%rsp), %xmm15\n" \
    "  addq $328, %rsp\n  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n" \
    "  popq %rdi\n  popq %rsi\n  popq %rbx\n  popq %rbp\n  ret\n.seh_endproc\n"
#define CX_BLK_A "%rcx"
#define CX_BLK_S "%rax"
#else
/* cdecl cx_block_*(a, k); the callee is stdcall. Stack at the call: args, k, a, saved edi esi ebx ebp. */
#define CX_BLK(name, ret, land, LOADV, STOREV, TAIL) \
    ".globl " name "\n" name ":\n" \
    "  pushl %ebp\n  pushl %ebx\n  pushl %esi\n  pushl %edi\n  movl 20(%esp), %ecx\n  movl 24(%esp), %edx\n" \
    CX_FPENV_LOAD("%ecx") CX_X87_LOAD("%ecx") LOADV \
    "  pushl %ecx\n  pushl %edx\n" \
    "  movl " CXA(CXK_NARGS) "(%edx), %eax\n" \
    "4:\n  testl %eax, %eax\n  jz 5f\n  decl %eax\n  pushl " CXA(CXK_ARGS) "(%edx,%eax,8)\n  jmp 4b\n5:\n" \
    "  movl %esp, " CXA(CXK_CALLSP) "(%edx)\n  movl $0, " CXA(CXK_CALLSP) "+4(%edx)\n" \
    "  movl %esp, " CXA(CXO_SP) "(%ecx)\n  movl $0, " CXA(CXO_SP) "+4(%ecx)\n" \
    "  movl " CXA(CXO_GIN) "+0(%ecx), %ebx\n  movl " CXA(CXO_GIN) "+16(%ecx), %esi\n" \
    "  movl " CXA(CXO_GIN) "+24(%ecx), %edi\n  movl " CXA(CXO_GIN) "+32(%ecx), %ebp\n" \
    "  movl $1, " CXA(CXO_READY) "(%ecx)\n  movl " CXA(CXK_FN) "(%edx), %eax\n  call *%eax\n" \
    CX_GLOBAL(ret) \
    "  pushl %eax\n  movl 4(%esp), %eax\n  popl " CXA(CXK_RET) "(%eax)\n  movl $0, " CXA(CXK_RET) "+4(%eax)\n  jmp 9f\n" \
    CX_GLOBAL(land) \
    "  pushl %eax\n  movl 4(%esp), %eax\n  movl $1, " CXA(CXK_LANDED) "(%eax)\n" \
    "  popl " CXA(CXK_RET) "(%eax)\n  movl $0, " CXA(CXK_RET) "+4(%eax)\n" \
    "9:\n  movl 4(%esp), %eax\n  addl $8, %esp\n  cld\n" \
    "  movl %ebx, " CXA(CXO_GOUT) "+0(%eax)\n  movl $0, " CXA(CXO_GOUT) "+4(%eax)\n" \
    "  movl %edx, " CXA(CXO_GOUT) "+8(%eax)\n  movl $0, " CXA(CXO_GOUT) "+12(%eax)\n" \
    "  movl %esi, " CXA(CXO_GOUT) "+16(%eax)\n  movl $0, " CXA(CXO_GOUT) "+20(%eax)\n" \
    "  movl %edi, " CXA(CXO_GOUT) "+24(%eax)\n  movl $0, " CXA(CXO_GOUT) "+28(%eax)\n" \
    "  movl %ebp, " CXA(CXO_GOUT) "+32(%eax)\n  movl $0, " CXA(CXO_GOUT) "+36(%eax)\n" \
    STOREV CX_X87_STORE("%eax") TAIL \
    "  popl %edi\n  popl %esi\n  popl %ebx\n  popl %ebp\n  ret\n"
#define CX_BLK_A "%ecx"
#define CX_BLK_S "%eax"
#endif

__asm__(
    ".text\n"
    CX_BLK("cx_block_xmm", "cx_blk_xmm_ret", "cx_blk_xmm_land",
           "  .irp i," CX_RV "\n  movdqu \\i*32(" CX_BLK_A "), %xmm\\i\n  .endr\n",
           "  .irp i," CX_RV "\n  movdqu %xmm\\i, " CXA(CXO_VOUT) "+\\i*32(" CX_BLK_S ")\n  .endr\n",
           "")
    CX_BLK("cx_block_ymm", "cx_blk_ymm_ret", "cx_blk_ymm_land",
           "  .irp i," CX_RV "\n  vmovdqu \\i*32(" CX_BLK_A "), %ymm\\i\n  .endr\n",
           "  .irp i," CX_RV "\n  vmovdqu %ymm\\i, " CXA(CXO_VOUT) "+\\i*32(" CX_BLK_S ")\n  .endr\n",
           "  vzeroupper\n")
);

enum { CXB_WAIT_NT, CXB_DELAY_NT, CXB_READ_NT, CXB_WAIT_K32, CXB_READ_K32, CXB_SLEEPEX, CXB_NMODES };
static const struct { const char *name; int direct; } cxb_modes[CXB_NMODES] = {
    { "NtWaitForSingleObject", 1 }, { "NtDelayExecution", 1 }, { "NtReadFile", 1 },
    { "WaitForSingleObject", 0 }, { "ReadFile", 0 }, { "SleepEx(alertable)", 0 },
};
#define CXB_DELAY_MS 1500
typedef struct { union { LONG Status; PVOID Pointer; } u; ULONG_PTR Information; } cx_iosb;

/* Everything the blocked worker references: one VirtualAlloc block, leaked on purpose if the worker never exits */
typedef struct cx_blkst {
    cx_call  k;
    int      mode, kind;
    cx_args *a;
    HANDLE   h, ev, rd, wr;
    DWORD    tid, nread;
    volatile LONG apc_ran;
    LARGE_INTEGER tmo;
    cx_iosb  iosb;
    uint8_t  buf[16], data[16];
} cx_blkst;

static void CALLBACK cx_blk_apc(ULONG_PTR p) { InterlockedExchange(&((cx_blkst *)p)->apc_ran, 1); }
static DWORD WINAPI cx_blk_main(void *p) {
    cx_blkst *s = (cx_blkst *)p;
    if (s->kind == T_VK_YMM) cx_block_ymm(s->a, &s->k); else cx_block_xmm(s->a, &s->k);
    return 0;
}
static const char *cx_blk_entry(int kind) { return kind == T_VK_YMM ? (const char *)(void *)cx_block_ymm : (const char *)(void *)cx_block_xmm; }
static const char *cx_blk_retlab(int kind) { return kind == T_VK_YMM ? cx_blk_ymm_ret : cx_blk_xmm_ret; }
static const char *cx_blk_landlab(int kind) { return kind == T_VK_YMM ? cx_blk_ymm_land : cx_blk_xmm_land; }

static int cx_in_ntdll(uintptr_t ip, uintptr_t *base_out) {
    HMODULE m = GetModuleHandleA("ntdll.dll");
    if (!m) return 0;
    const IMAGE_DOS_HEADER *d = (const IMAGE_DOS_HEADER *)m;
    const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const uint8_t *)m + d->e_lfanew);
    uintptr_t b = (uintptr_t)m;
    if (base_out) *base_out = b;
    return ip >= b && ip < b + nt->OptionalHeader.SizeOfImage;
}
/* SP after the call returned (x86 stdcall callee pops its arguments) */
static uintptr_t cx_blk_ret_sp(const cx_blkst *s) {
#ifdef T_X64
    return (uintptr_t)s->k.call_sp;
#else
    return (uintptr_t)s->k.call_sp + 4u * s->k.nargs;
#endif
}

static CONTEXT *cx_get_std_h(HANDLE h) {
    CONTEXT *c = cx_plain(CX_STD);
    if (!c) return NULL;
    if (!GetThreadContext(h, c)) {
        CHECKF(0, "GetThreadContext(CX_STD) of the blocked worker failed, GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
        return NULL;
    }
    return c;
}

/* Prepare mode + register images (before any thread exists: SKIP/REQUIRE are safe). Returns NULL + reason. */
static cx_blkst *cx_blk_new(int mode, int kind, uint32_t tag, char *why, size_t cap) {
    cx_blkst *s = (cx_blkst *)cx_alloc(sizeof(cx_blkst) + sizeof(cx_args) + 64);
    if (!s) { snprintf(why, cap, "VirtualAlloc failed"); return NULL; }
    memset(s, 0, sizeof *s);
    s->a = (cx_args *)(((uintptr_t)(s + 1) + 63) & ~(uintptr_t)63);
    cx_args_fill(s->a, tag, 1);
    s->a->mxcsr_in = 0x5F80;     /* RC = up: a non-default control value the context must report */
    s->a->fcw_in = 0x0A7F;       /* RC = up, PC = 53-bit */
    s->mode = mode; s->kind = kind;
    for (int i = 0; i < 16; i++) s->data[i] = (uint8_t)(0xC0 + i);
    const char *fname = NULL;
    switch (mode) {
    case CXB_WAIT_NT:  fname = "NtWaitForSingleObject"; break;
    case CXB_DELAY_NT: fname = "NtDelayExecution"; break;
    case CXB_READ_NT:  fname = "NtReadFile"; break;
    default: break;
    }
    void *fn = NULL;
    if (fname) {
        fn = t_proc("ntdll.dll", fname);
        if (!fn) { snprintf(why, cap, "ntdll!%s not exported", fname); cx_free(s); return NULL; }
    }
    if (mode == CXB_WAIT_NT || mode == CXB_WAIT_K32) {
        s->ev = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!s->ev) { snprintf(why, cap, "CreateEvent GLE=%lu", (unsigned long)GetLastError()); cx_free(s); return NULL; }
    }
    if (mode == CXB_READ_NT || mode == CXB_READ_K32) {
        if (!CreatePipe(&s->rd, &s->wr, NULL, 4096)) { snprintf(why, cap, "CreatePipe GLE=%lu", (unsigned long)GetLastError()); cx_free(s); return NULL; }
    }
    cx_call *k = &s->k;
    switch (mode) {
    case CXB_WAIT_NT:     /* NtWaitForSingleObject(h, FALSE, &-10 s): bounded even if the parent never signals */
        s->tmo.QuadPart = -100000000LL;
        k->fn = (uintptr_t)fn; k->nargs = 3;
        k->args[0] = (uintptr_t)s->ev; k->args[1] = 0; k->args[2] = (uintptr_t)&s->tmo; break;
    case CXB_DELAY_NT:
        s->tmo.QuadPart = -(LONGLONG)CXB_DELAY_MS * 10000;
        k->fn = (uintptr_t)fn; k->nargs = 2; k->args[0] = 0; k->args[1] = (uintptr_t)&s->tmo; break;
    case CXB_READ_NT:     /* NtReadFile(h, NULL, NULL, NULL, &iosb, buf, 16, NULL, NULL) on a synchronous pipe */
        k->fn = (uintptr_t)fn; k->nargs = 9;
        k->args[0] = (uintptr_t)s->rd; k->args[4] = (uintptr_t)&s->iosb; k->args[5] = (uintptr_t)s->buf; k->args[6] = 16; break;
    case CXB_WAIT_K32:
        k->fn = (uintptr_t)(void *)WaitForSingleObject; k->nargs = 2; k->args[0] = (uintptr_t)s->ev; k->args[1] = 10000; break;
    case CXB_READ_K32:
        k->fn = (uintptr_t)(void *)ReadFile; k->nargs = 5;
        k->args[0] = (uintptr_t)s->rd; k->args[1] = (uintptr_t)s->buf; k->args[2] = 16; k->args[3] = (uintptr_t)&s->nread; break;
    case CXB_SLEEPEX:
        k->fn = (uintptr_t)(void *)SleepEx; k->nargs = 2; k->args[0] = 10000; k->args[1] = TRUE; break;
    }
    return s;
}

/* Start the worker and wait until it is inside the blocking call; returns 1 with the worker SUSPENDED once.
 * "Inside" = its IP left [entry, return label): it executed the call. A 50 ms grace lets it reach the wait. */
static int cx_blk_start(cx_blkst *s) {
    s->a->ready = 0;
    s->h = CreateThread(NULL, 0, cx_blk_main, s, 0, &s->tid);
    if (!s->h) return 0;
    double t0 = t_now_ms();
    while (!s->a->ready && t_now_ms() - t0 < CX_TMO) Sleep(1);
    if (!s->a->ready) return 0;
    Sleep(50);
    uintptr_t lo = (uintptr_t)cx_blk_entry(s->kind), hi = (uintptr_t)cx_blk_retlab(s->kind);
    CONTEXT *c = cx_plain(CONTEXT_CONTROL);
    if (!c) return 0;
    int ok = 0;
    for (int i = 0; i < 200 && !ok; i++) {
        if (SuspendThread(s->h) == (DWORD)-1) break;
        c->ContextFlags = CONTEXT_CONTROL;
        if (GetThreadContext(s->h, c)) {
            uintptr_t ip = cx_ip(c);
            if (ip < lo || ip >= hi) { ok = 1; break; }
        }
        ResumeThread(s->h);
        Sleep(10);
    }
    cx_free(c);
    return ok;
}
static void cx_blk_wake(cx_blkst *s) {
    DWORD n = 0;
    switch (s->mode) {
    case CXB_WAIT_NT: case CXB_WAIT_K32: SetEvent(s->ev); break;
    case CXB_READ_NT: case CXB_READ_K32:
        CHECKF(WriteFile(s->wr, s->data, 16, &n, NULL) && n == 16, "WriteFile(pipe) GLE=%lu", (unsigned long)GetLastError()); break;
    case CXB_SLEEPEX:
        CHECKF(QueueUserAPC(cx_blk_apc, s->h, (ULONG_PTR)s), "QueueUserAPC GLE=%lu", (unsigned long)GetLastError()); break;
    default: break;
    }
}
/* Resume (once), wake, wait for the worker to finish. Returns 1 if it exited. */
static int cx_blk_finish(cx_blkst *s) {
    if (!s->h) return 0;
    ResumeThread(s->h);
    cx_blk_wake(s);
    DWORD wr = WaitForSingleObject(s->h, CX_TMO + CXB_DELAY_MS);
    if (wr != WAIT_OBJECT_0) {        /* unblock by every means; all waits are bounded (<= 10 s) anyway */
        if (s->ev) SetEvent(s->ev);
        if (s->wr) { CloseHandle(s->wr); s->wr = NULL; }
        QueueUserAPC(cx_blk_apc, s->h, (ULONG_PTR)s);
        wr = WaitForSingleObject(s->h, 12000);
    }
    CHECKF(wr == WAIT_OBJECT_0, "blocked worker did not return after the wake-up (wait=%lu)", (unsigned long)wr);
    return wr == WAIT_OBJECT_0;
}
static void cx_blk_free(cx_blkst *s, int exited) {
    if (!s) return;
    if (s->h && !exited) { t_note("worker still running: its state block is leaked on purpose"); return; }
    if (s->h) CloseHandle(s->h);
    if (s->ev) CloseHandle(s->ev);
    if (s->rd) CloseHandle(s->rd);
    if (s->wr) CloseHandle(s->wr);
    cx_free(s);
}
/* The call's own result when it returned normally */
static void cx_blk_check_result(cx_blkst *s) {
    uint32_t r = (uint32_t)s->k.ret;
    switch (s->mode) {
    case CXB_WAIT_NT: case CXB_DELAY_NT: CHECKF(r == 0, "%s returned 0x%x", cxb_modes[s->mode].name, r); break;
    case CXB_WAIT_K32: CHECKF(r == WAIT_OBJECT_0, "WaitForSingleObject returned 0x%x", r); break;
    case CXB_READ_NT:
        CHECKF(r == 0, "NtReadFile returned 0x%x", r);
        CHECKF(s->iosb.u.Status == 0 && s->iosb.Information == 16, "IO_STATUS_BLOCK {0x%lx, %lu}",
               (unsigned long)s->iosb.u.Status, (unsigned long)s->iosb.Information);
        CHECK_MEM(s->buf, s->data, 16);
        break;
    case CXB_READ_K32:
        CHECKF(r == 1 && s->nread == 16, "ReadFile returned %u, %lu bytes", r, (unsigned long)s->nread);
        CHECK_MEM(s->buf, s->data, 16);
        break;
    case CXB_SLEEPEX:
        CHECKF(r == WAIT_IO_COMPLETION, "SleepEx returned 0x%x, want WAIT_IO_COMPLETION", r);
        CHECK(s->apc_ran);
        break;
    }
    CHECKF(!s->k.landed, "the redirect landing ran without a redirect");
}

/* Nonvolatile GPR indices under the native ABI (x64: rbx rsi rdi rbp r12-r15; x86: ebx esi edi ebp) */
#ifdef T_X64
static const int cxb_nvg[] = { 0, 3, 4, 5, 9, 10, 11, 12 };
#define CXB_NV_XMM_LO 6
#else
static const int cxb_nvg[] = { 0, 2, 3, 4 };
#define CXB_NV_XMM_LO CX_NV          /* x86: every XMM register is volatile */
#endif
static int cxb_is_nvg(int i) { for (size_t j = 0; j < T_COUNTOF(cxb_nvg); j++) if (cxb_nvg[j] == i) return 1; return 0; }

/* Classify a register image: s(et) / o(riginal) / z(ero) / x (anything else); appended to a t_info string */
static char cx_cls(const uint8_t *got, const uint8_t *set, const uint8_t *orig, int n) {
    if (set && !memcmp(got, set, n)) return 's';
    if (orig && !memcmp(got, orig, n)) return 'o';
    if (cx_all_zero(got, n)) return 'z';
    return 'x';
}

/* GetThreadContext of the blocked worker: the user-mode state at the call */
static void cx_blk_check_get(cx_blkst *s, CONTEXT *c, const char *what) {
    const cx_args *a = s->a;
    int direct = cxb_modes[s->mode].direct;
    uintptr_t ip = cx_ip(c), sp = cx_sp(c), csp = (uintptr_t)s->k.call_sp, nb = 0;
    int in_nt = cx_in_ntdll(ip, &nb);
    t_info("blocked_ip", "%p (%s+0x%lx), call fn %p, sp = call_sp - %ld", (void *)ip, in_nt ? "ntdll" : "?",
           (unsigned long)(in_nt ? ip - nb : 0), (void *)(uintptr_t)s->k.fn, (long)((intptr_t)csp - (intptr_t)sp));
    CX_CHECK_WGAP("context of a thread blocked in a system call reports a non-ntdll IP", in_nt,
                  "%s: IP %p is not inside ntdll (the system-call stub)", what, (void *)ip);
#ifdef T_X64
    if (direct) {
        uintptr_t f = (uintptr_t)s->k.fn;
        CX_CHECK_WGAP("context of a thread blocked in a system call reports a non-ntdll IP", ip > f && ip <= f + 0x20,
                      "%s: RIP %p is not inside the %s stub [%p, +0x20]", what, (void *)ip, cxb_modes[s->mode].name, (void *)f);
    }
#endif
    if (direct)
        CX_CHECK_WGAP("context of a thread blocked in a system call reports a non-stub SP", sp < csp && csp - sp <= 64,
                      "%s: SP %p not just below the call frame %p", what, (void *)sp, (void *)csp);
    else CHECKF(sp < csp && csp - sp < 0x20000, "%s: SP %p not below the caller's frame %p", what, (void *)sp, (void *)csp);
    CHECKF(c->SegCs == cx_own_cs(), "%s: SegCs 0x%x want 0x%x", what, (unsigned)c->SegCs, (unsigned)cx_own_cs());
    CHECKF(c->EFlags & 0x200, "%s: EFlags 0x%lx IF clear in user mode", what, (unsigned long)c->EFlags);
    CHECKF(!(c->EFlags & 0x100), "%s: EFlags 0x%lx TF set", what, (unsigned long)c->EFlags);

    /* GPRs / XMM: exact for direct calls (nothing ran between the load and the stub); through kernelbase its own
     * code may hold any register at the system call, so only recorded */
    char cls[64]; int n = 0;
    for (size_t j = 0; j < T_COUNTOF(cxb_nvg); j++) {
        int i = cxb_nvg[j];
        uint64_t got = cx_gpr(c, i), want = cx_gpr_expect(a, i);
        if (direct) CHECKF(got == want, "%s: %s = 0x%llx want 0x%llx", what, cx_gpr_name(i), (unsigned long long)got, (unsigned long long)want);
        cls[n++] = got == want ? 'o' : 'x';
    }
    cls[n] = 0;
    t_info("blocked_nv_gprs", "%s", cls);
    n = 0;
    for (int i = 0; i < CX_NV; i++) {
        if (direct && i >= CXB_NV_XMM_LO) {
            char lab[64]; snprintf(lab, sizeof lab, "xmm%d %s", i, what);
            CHECK_REG(lab, cx_xmm(c, i), a->vin[i], 16);
        }
        cls[n++] = cx_cls(cx_xmm(c, i), NULL, a->vin[i], 16);
    }
    cls[n] = 0;
    t_info("blocked_xmm", "%s (o = loaded value, z = zero, x = other)", cls);

    /* FP control state: no Windows wait path changes it; FCW/MXCSR are ABI-nonvolatile */
    const char *fpgap = "context of a thread blocked in a system call has an all-zero FP area";
    uint32_t mx = cx_mxcsr(c), fmx = cx_fx_mxcsr(c);
    CX_CHECK_WGAP(fpgap, (mx & 0xFFC0) == (a->mxcsr_in & 0xFFC0), "%s: MXCSR 0x%x, loaded 0x%x", what, mx, a->mxcsr_in);
    CX_CHECK_WGAP(fpgap, (fmx & 0xFFC0) == (a->mxcsr_in & 0xFFC0), "%s: FXSAVE MXCSR 0x%x, loaded 0x%x", what, fmx, a->mxcsr_in);
    CX_CHECK_WGAP(fpgap, (cx_fcw(c) & CX_FCW_BITS) == (a->fcw_in & CX_FCW_BITS), "%s: x87 CW 0x%x, loaded 0x%x", what,
                  (unsigned)cx_fcw(c), (unsigned)a->fcw_in);
    n = 0;
    for (int i = 0; i < 8; i++) {
        int eq = !memcmp(cx_fx(c) + 32 + 16 * i, a->fin[i].b, 10);
        if (direct) CX_CHECK_WGAP(fpgap, eq, "%s: ST(%d) is not the loaded value", what, i);
        cls[n++] = eq ? 'o' : cx_all_zero(cx_fx(c) + 32 + 16 * i, 10) ? 'z' : 'x';
    }
    cls[n] = 0;
    t_info("blocked_x87", "%s", cls);
}

/* After the call returned: what the ABI guarantees across a call, with `want_*` the values the caller holds
 * (loaded, or set by SetThreadContext). `alt_*` (may be NULL) is a second acceptable value: a callee that saved
 * the register before the Set restores the old one. Volatile registers are recorded (s/o/z/x). */
static void cx_blk_check_after(cx_blkst *s, const uint64_t *want_g, const uint64_t *alt_g, uint8_t (*want_v)[32],
                               uint8_t (*alt_v)[32], uint32_t want_mx, uint16_t want_cw, const t_f80 *set_st, const char *what) {
    const cx_args *a = s->a;
    char cls[64]; int n = 0;
    for (size_t j = 0; j < T_COUNTOF(cxb_nvg); j++) {
        int i = cxb_nvg[j];
        uint64_t got = a->gout[i];
        uint64_t w = want_g[i], al = alt_g ? alt_g[i] : w;
#ifdef T_X86
        w = (uint32_t)w; al = (uint32_t)al;
#endif
        CHECKF(got == w || got == al, "%s: %s = 0x%llx want 0x%llx%s", what, cx_gpr_name(i), (unsigned long long)got,
               (unsigned long long)w, alt_g ? " (or the pre-Set value)" : "");
        cls[n++] = got == w ? 's' : got == al ? 'o' : 'x';
    }
    cls[n] = 0;
    t_info("after_nv_gprs", "%s", cls);
    n = 0;
    for (int i = 0; i < CX_NV; i++) {
        if (i >= CXB_NV_XMM_LO) {
            int ok = !memcmp(a->vout[i], want_v[i], 16) || (alt_v && !memcmp(a->vout[i], alt_v[i], 16));
            if (!ok) { char lab[64]; snprintf(lab, sizeof lab, "xmm%d %s", i, what); CHECK_REG(lab, a->vout[i], want_v[i], 16); }
        }
        cls[n++] = cx_cls(a->vout[i], want_v[i], alt_v ? alt_v[i] : a->vin[i], 16);
    }
    cls[n] = 0;
    t_info("after_xmm", "%s (s = expected, o = pre-Set/loaded, z = zero, x = other; xmm%d+ checked)", cls, CXB_NV_XMM_LO);
    const char *setgap = "SetThreadContext FP state of a thread blocked in a system call is not applied when the call returns";
    CX_CHECK_WGAP(setgap, (a->mxcsr_out & 0xFFC0) == (want_mx & 0xFFC0), "%s: MXCSR 0x%x want control 0x%x", what, a->mxcsr_out, want_mx);
    CX_CHECK_WGAP(setgap, (a->fcw_out & CX_FCW_BITS) == (want_cw & CX_FCW_BITS), "%s: x87 CW 0x%x want 0x%x", what,
                  (unsigned)a->fcw_out, (unsigned)want_cw);
    if (s->kind == T_VK_YMM) {
        n = 0;
        for (int i = 0; i < CX_NV; i++) cls[n++] = cx_cls(a->vout[i] + 16, want_v[i] + 16, a->vin[i] + 16, 16);
        cls[n] = 0;
        t_info("after_ymm_upper", "%s (volatile: recorded only)", cls);
    }
    n = 0;
    for (int i = 0; i < 8; i++) cls[n++] = cx_cls(a->fout[i].b, set_st ? set_st[i].b : NULL, a->fin[i].b, 10);
    cls[n] = 0;
    t_info("after_x87_stack", "%s (volatile: recorded only)", cls);
}

/* ---- Get while blocked, then the ABI view after the wait returns ---- */
static void cx_blk_get_case(int mode, int kind) {
    char why[128];
    cx_blkst *s = cx_blk_new(mode, kind, 0xD100 + (uint32_t)mode * 16 + (uint32_t)kind, why, sizeof why);
    if (!s) SKIP("%s", why);
    int started = cx_blk_start(s);
    CHECKF(started, "worker did not reach the blocking call");
    if (started) {
        CONTEXT *c = cx_get_std_h(s->h);
        if (c) { cx_blk_check_get(s, c, "blocked"); cx_free(c); }
        if (kind == T_VK_YMM) {
            cx_xc x = { 0 };
            cx_worker w = { 0 }; w.h = s->h;
            if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
                uint64_t m = cx_mask(x.ctx);
                uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
                char cls[24]; int n = 0;
                for (int i = 0; i < CX_NV && ymmh; i++) {
                    const uint8_t *g = (m & T_XS_AVX) ? ymmh + 16 * i : NULL;
                    char ch = g ? cx_cls(g, NULL, s->a->vin[i] + 16, 16) : 'i';
                    cls[n++] = ch;
                    /* XINUSE consistency: a reported upper half is the loaded value or zero, never foreign data */
                    if (g) CX_CHECK_WGAP("context of a thread blocked in a system call has foreign YMM data", ch == 'o' || ch == 'z',
                                         "ymm%d upper half in a blocked thread's XSTATE context is neither the loaded value nor zero", i);
                }
                cls[n] = 0;
                t_info("blocked_xstate", "mask 0x%llx, uppers %s (o = loaded, z = zero, i = init/absent)", (unsigned long long)m, cls);
            }
            cx_xc_free(&x);
        }
    }
    int done = cx_blk_finish(s);
    if (done) {
        cx_blk_check_result(s);
        uint64_t g[16]; for (int i = 0; i < 16; i++) g[i] = s->a->gin[i];
        cx_blk_check_after(s, g, NULL, s->a->vin, NULL, s->a->mxcsr_in, s->a->fcw_in, NULL, "after the wait");
    }
    cx_blk_free(s, done);
}

/* ---- SetThreadContext while blocked: FP (+XSTATE) and nonvolatile GPRs; take effect when the call returns ---- */
static void cx_blk_set_case(int mode, int kind) {
    char why[128];
    if (kind == T_VK_YMM) CX_NEED_AVX();
    cx_blkst *s = cx_blk_new(mode, kind, 0xD800 + (uint32_t)mode * 16 + (uint32_t)kind, why, sizeof why);
    if (!s) SKIP("%s", why);
    static uint8_t nv[16][32];
    static uint64_t ng[16];
    static t_f80 nf[8];
    for (int i = 0; i < 16; i++) { t_pattern_reg(nv[i], 32, 0xD8F0 + (uint32_t)mode, (uint32_t)i); ng[i] = t_pattern_word(0xD8F0 + (uint32_t)mode, 40 + (uint32_t)i, 0); }
    for (int i = 0; i < 8; i++) cx_f80_make(&nf[i], 0xD8F0 + (uint32_t)mode, i);
    const uint32_t nmx = 0x3F80;       /* RC = down */
    const uint16_t ncw = 0x0E7F;       /* RC = truncate, PC = 53-bit */
    int set_ok = 0, direct = cxb_modes[mode].direct;
    int started = cx_blk_start(s);
    CHECKF(started, "worker did not reach the blocking call");
    if (started) {
        cx_xc x = { 0 };
        CONTEXT *c = NULL, *plain = NULL;
        uint8_t *ymmh = NULL;
        cx_worker w = { 0 }; w.h = s->h;
        if (kind == T_VK_YMM) {
            if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) { c = x.ctx; ymmh = cx_locate(c, CX_XS_AVX, NULL); CHECK(ymmh != NULL); }
        } else c = plain = cx_get_std_h(s->h);
        if (c && (kind != T_VK_YMM || ymmh)) {
            for (int i = 0; i < CX_NV; i++) memcpy(cx_xmm(c, i), nv[i], 16);
            if (ymmh) for (int i = 0; i < CX_NV; i++) memcpy(ymmh + 16 * i, nv[i] + 16, 16);
            cx_set_mxcsr(c, nmx);
            cx_set_fcw(c, ncw);
            for (int i = 0; i < 8; i++) cx_set_st(c, i, &nf[i]);
            /* GPRs only for direct calls: inside kernelbase the nonvolatile registers hold kernelbase's own
             * pointers and frame state, so rewriting them would corrupt the callee, not test the OS */
            if (direct) for (size_t j = 0; j < T_COUNTOF(cxb_nvg); j++) cx_set_gpr(c, cxb_nvg[j], ng[cxb_nvg[j]]);
            if (ymmh) { CHECK(X.SetXStateFeaturesMask(c, T_XS_AVX | 3)); c->ContextFlags = CX_STD | CONTEXT_XSTATE; }
            else c->ContextFlags = CX_STD;
            set_ok = SetThreadContext(s->h, c);
            CHECKF(set_ok, "SetThreadContext while blocked in %s, GLE=%lu", cxb_modes[mode].name, (unsigned long)GetLastError());
            /* read back while still blocked: the context reports what was set */
            CONTEXT *d = set_ok ? cx_get_std_h(s->h) : NULL;
            if (d) {
                char lab[64];
                for (int i = 0; i < CX_NV; i++) { snprintf(lab, sizeof lab, "xmm%d read back while blocked", i); CHECK_REG(lab, cx_xmm(d, i), nv[i], 16); }
                CHECKF(cx_mxcsr(d) == nmx, "MXCSR read back 0x%x want 0x%x", cx_mxcsr(d), nmx);
                CHECKF((cx_fcw(d) & CX_FCW_BITS) == (ncw & CX_FCW_BITS), "x87 CW read back 0x%x want 0x%x", (unsigned)cx_fcw(d), (unsigned)ncw);
                for (int i = 0; i < 8; i++) { snprintf(lab, sizeof lab, "ST(%d) read back while blocked", i); CHECK_REG(lab, cx_fx(d) + 32 + 16 * i, nf[i].b, 10); }
                for (size_t j = 0; j < T_COUNTOF(cxb_nvg) && direct; j++) {
                    int i = cxb_nvg[j];
                    uint64_t want = ng[i];
#ifdef T_X86
                    want = (uint32_t)want;
#endif
                    CHECKF(cx_gpr(d, i) == want, "%s read back 0x%llx want 0x%llx", cx_gpr_name(i), (unsigned long long)cx_gpr(d, i), (unsigned long long)want);
                }
                CHECKF(cx_ip(d) == cx_ip(c) && cx_sp(d) == cx_sp(c), "IP/SP changed by an FP/integer set: %p/%p -> %p/%p",
                       (void *)cx_ip(c), (void *)cx_sp(c), (void *)cx_ip(d), (void *)cx_sp(d));
                cx_free(d);
            }
            if (ymmh && set_ok) {
                cx_xc y = { 0 };
                if (cx_get_x(&w, &y, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
                    uint8_t *yh = cx_locate(y.ctx, CX_XS_AVX, NULL);
                    CHECKF(cx_mask(y.ctx) & T_XS_AVX, "AVX missing from the mask after setting non-zero upper halves");
                    char lab[64];
                    for (int i = 0; yh && i < CX_NV; i++) { snprintf(lab, sizeof lab, "ymm%d upper read back while blocked", i); CHECK_REG(lab, yh + 16 * i, nv[i] + 16, 16); }
                }
                cx_xc_free(&y);
            }
        }
        cx_free(plain);
        cx_xc_free(&x);
    }
    int done = cx_blk_finish(s);
    if (done && set_ok) {
        cx_blk_check_result(s);
        /* Through kernelbase an XMM register its code saved before the Set is restored to the pre-Set value on
         * return: either value honours the ABI. MXCSR / x87 CW are never saved and restored by those paths. */
        uint64_t g[16]; for (int i = 0; i < 16; i++) g[i] = direct ? ng[i] : s->a->gin[i];
        cx_blk_check_after(s, g, NULL, nv, direct ? NULL : s->a->vin, nmx, ncw, nf, "after Set while blocked");
    }
    cx_blk_free(s, done);
}

/* ---- CONTEXT_CONTROL redirect: Rip = landing label, Rsp = the frame right after the call ---- */
static void cx_blk_redirect_case(int mode) {
    char why[128];
    cx_blkst *s = cx_blk_new(mode, T_VK_XMM, 0xDE00 + (uint32_t)mode, why, sizeof why);
    if (!s) SKIP("%s", why);
    int set_ok = 0;
    int started = cx_blk_start(s);
    CHECKF(started, "worker did not reach the blocking call");
    if (started) {
        CONTEXT *c = cx_plain(CONTEXT_CONTROL);
        if (c && GetThreadContext(s->h, c)) {
            cx_set_ip(c, (uintptr_t)cx_blk_landlab(T_VK_XMM));
#ifdef T_X64
            c->Rsp = cx_blk_ret_sp(s);
#else
            c->Esp = (DWORD)cx_blk_ret_sp(s);
#endif
            set_ok = SetThreadContext(s->h, c);
            CHECKF(set_ok, "SetThreadContext(CONTEXT_CONTROL) while blocked, GLE=%lu", (unsigned long)GetLastError());
            CONTEXT *d = cx_plain(CONTEXT_CONTROL);
            if (d && set_ok && GetThreadContext(s->h, d))
                CHECKF(cx_ip(d) == (uintptr_t)cx_blk_landlab(T_VK_XMM) && cx_sp(d) == cx_blk_ret_sp(s),
                       "redirect read back IP %p SP %p", (void *)cx_ip(d), (void *)cx_sp(d));
            cx_free(d);
        } else CHECKF(0, "GetThreadContext(CONTEXT_CONTROL) GLE=%lu", (unsigned long)GetLastError());
        cx_free(c);
    }
    int done = cx_blk_finish(s);
    if (done && set_ok) {
        CHECKF(s->k.landed, "the wait returned to its caller instead of the redirected IP");
        t_info("status_at_landing", "0x%llx", (unsigned long long)s->k.ret);
        if (s->k.landed && cxb_modes[mode].direct) {
            /* nothing ran between the load and the stub: the ABI-nonvolatile state arrives at the landing intact */
            uint64_t g[16]; for (int i = 0; i < 16; i++) g[i] = s->a->gin[i];
            cx_blk_check_after(s, g, NULL, s->a->vin, NULL, s->a->mxcsr_in, s->a->fcw_in, NULL, "at the redirect landing");
        }
        if (mode == CXB_READ_NT || mode == CXB_READ_K32) CHECK_MEM(s->buf, s->data, 16);   /* the read itself completed */
    }
    cx_blk_free(s, done);
}

TEST(blk_getctx_wait_nt)      { cx_blk_get_case(CXB_WAIT_NT, T_VK_XMM); }
TEST(blk_getctx_delay_nt)     { cx_blk_get_case(CXB_DELAY_NT, T_VK_XMM); }
TEST(blk_getctx_readfile_nt)  { cx_blk_get_case(CXB_READ_NT, T_VK_XMM); }
TEST(blk_getctx_wait_k32)     { cx_blk_get_case(CXB_WAIT_K32, T_VK_XMM); }
TEST(blk_getctx_readfile_k32) { cx_blk_get_case(CXB_READ_K32, T_VK_XMM); }
TEST(blk_getctx_sleepex_alertable) { cx_blk_get_case(CXB_SLEEPEX, T_VK_XMM); }
TEST(blk_getctx_ymm_wait_nt)  { CX_NEED_AVX(); cx_blk_get_case(CXB_WAIT_NT, T_VK_YMM); }
TEST(blk_getctx_ymm_readfile_nt) { CX_NEED_AVX(); cx_blk_get_case(CXB_READ_NT, T_VK_YMM); }
TEST(blk_setctx_wait_nt)      { cx_blk_set_case(CXB_WAIT_NT, T_VK_XMM); }
TEST(blk_setctx_delay_nt)     { cx_blk_set_case(CXB_DELAY_NT, T_VK_XMM); }
TEST(blk_setctx_readfile_nt)  { cx_blk_set_case(CXB_READ_NT, T_VK_XMM); }
TEST(blk_setctx_wait_k32)     { cx_blk_set_case(CXB_WAIT_K32, T_VK_XMM); }
TEST(blk_setctx_readfile_k32) { cx_blk_set_case(CXB_READ_K32, T_VK_XMM); }
TEST(blk_setctx_sleepex_alertable) { cx_blk_set_case(CXB_SLEEPEX, T_VK_XMM); }
TEST(blk_setctx_xstate_wait_nt) { cx_blk_set_case(CXB_WAIT_NT, T_VK_YMM); }
TEST(blk_setctx_xstate_sleepex_alertable) { cx_blk_set_case(CXB_SLEEPEX, T_VK_YMM); }
TEST(blk_redirect_wait_nt)    { cx_blk_redirect_case(CXB_WAIT_NT); }
TEST(blk_redirect_delay_nt)   { cx_blk_redirect_case(CXB_DELAY_NT); }
TEST(blk_redirect_readfile_nt) { cx_blk_redirect_case(CXB_READ_NT); }
TEST(blk_redirect_wait_k32)   { cx_blk_redirect_case(CXB_WAIT_K32); }

/* ======================================================================================================
 * 11. WOW64
 * ====================================================================================================== */
#define CX_W64_TAG 0xB100u

TEST(wow64_own_threads) {
#ifdef T_X86
    const t_env *e = t_env_get();
    t_info("wow64", "%d", e->wow64);
    if (!e->wow64) SKIP("native 32-bit OS: no WOW64 layer");
    /* GetThreadContext in a WOW64 process returns the x86 view: covered by every other case of this build.
     * Here: the x86 view is internally consistent (32-bit selectors, flags) and Wow64GetThreadContext, when
     * exported to 32-bit code, agrees with it. */
    cx_resolve();
    cx_args *a = cx_args_new(0xB200, 0);
    REQUIRE(a);
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        CONTEXT *c = cx_get_std(&w);
        if (c) {
            CHECKF((c->SegCs & 3) == 3, "SegCs 0x%lx not RPL 3", (unsigned long)c->SegCs);
            CHECKF(c->Eip >= (DWORD)(uintptr_t)cx_xmm_loop && c->Eip <= (DWORD)(uintptr_t)cx_xmm_exit, "Eip outside loop");
            if (X.Wow64GetThreadContext) {
                WOW64_CONTEXT wc;
                memset(&wc, CX_SENT, sizeof wc);
                wc.ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_EXTENDED_REGISTERS;
                SetLastError(0);
                BOOL r = X.Wow64GetThreadContext(w.h, &wc);
                t_info("wow64getthreadcontext_from_x86", "%d gle=%lu", r, (unsigned long)GetLastError());
                if (r) {
                    CHECK_EQ(wc.Ebx, c->Ebx); CHECK_EQ(wc.Esi, c->Esi); CHECK_EQ(wc.Edi, c->Edi);
                    CHECK_EQ(wc.Eip, c->Eip); CHECK_EQ(wc.Esp, c->Esp);
                    CHECK_MEM(wc.ExtendedRegisters + 160, c->ExtendedRegisters + 160, 128);
                }
            } else t_note("Wow64GetThreadContext not exported to 32-bit code");
            /* the worker is suspended once (cx_suspend): Wow64SuspendThread shares the documented count */
            if (X.Wow64SuspendThread) {
                DWORD sc = X.Wow64SuspendThread(w.h);
                CHECKF(sc == 1, "Wow64SuspendThread previous count %lu want 1 (GLE=%lu)", (unsigned long)sc, (unsigned long)GetLastError());
                if (sc != (DWORD)-1) { DWORD rc = ResumeThread(w.h); CHECKF(rc == 2, "ResumeThread previous count %lu want 2", (unsigned long)rc); }
            } else t_note("Wow64SuspendThread not exported to 32-bit code");
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
#else
    SKIP("x86 build only (the x64 build drives WOW64 through wow64_cross_process_context)");
#endif
}

/* WOW64 cross-architecture contract, pinned against the Windows 10 22H2 (19041) reference decompilation
 * (docker1 corpus explorer, checked 2026-11):
 *   kernelbase!Wow64GetThreadContext -> ntdll!RtlWow64GetThreadContext = NtQueryInformationThread(h,
 *     ThreadWow64Context (0x1d), ctx, 0x2cc, NULL); failure -> BaseSetLastNTError, FALSE.
 *   kernelbase!Wow64SetThreadContext -> ntdll!RtlWow64SetThreadContext = NtSetInformationThread(h, 0x1d, ctx, 0x2cc).
 *   kernelbase!Wow64SuspendThread -> ntdll!RtlWow64SuspendThread(h, &prev) (NtSuspendThread for an ordinary
 *     x86 target); failure -> BaseSetLastNTError and (DWORD)-1, else the previous suspend count.
 *   ntoskrnl!PspWow64SetContextThread: length != 0x2cc -> STATUS_INFO_LENGTH_MISMATCH (0xc0000004); target
 *     process without an x86 (0x14c) WOW64 machine -> STATUS_INVALID_PARAMETER (0xc000000d).
 *     PspWow64GetContextThread: same length / machine gates (error status not visible in the decompile).
 *   ntoskrnl!RtlpWow64SanitizeContextFlags: user-mode flags are sanitized (| CONTEXT_i386, & 0xd801007f).
 *   So the WOW64 view is the fixed 0x2cc-byte WOW64_CONTEXT: FNSAVE FloatSave + 512-byte FXSAVE
 *   ExtendedRegisters (XMM0-7, MXCSR); it has no room for XSTATE (YMM) components. YMM uppers of an x86
 *   thread are visible only through GetThreadContext(CONTEXT_XSTATE) inside the x86 process, or through
 *   the native x64 view of a thread suspended in 32-bit code. */
#ifndef WOW64_CONTEXT_XSTATE
#define WOW64_CONTEXT_XSTATE (WOW64_CONTEXT_i386 | 0x40L)
#endif
#define CX_W64_CTX_LEN 0x2cc
_Static_assert(sizeof(WOW64_CONTEXT) == CX_W64_CTX_LEN, "WOW64_CONTEXT is 0x2cc bytes");
#define CX_W64_NEW_MXCSR_DAZ 0x5FC0u     /* RC = up, DAZ, all exceptions masked */
#define CX_W64_NEW_MXCSR     0x5F80u     /* without DAZ when MXCSR_MASK lacks it */
#define CX_W64_NEW_FCW       0x067Fu     /* RC = down, PC = 53 bit, all masked */

static void cx_print_hex(const char *key, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    printf("%s", key);
    for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
    printf("\n");
}

/* x86 child: the "self view" of its own YMM worker through GetThreadContext(CONTEXT_XSTATE). Uses no CHECK
 * (children report KEY=value lines only). */
static void cx_w64_self_view(cx_worker *w) {
    cx_xc x;
    memset(&x, 0, sizeof x);
    if (!cx_xapi()) { printf("W64_SELF_ERR=noapi\n"); return; }
    if (!cx_xc_init(&x, CX_STD | CONTEXT_XSTATE)) { printf("W64_SELF_ERR=init gle=%lu\n", (unsigned long)x.err); return; }
    if (!X.SetXStateFeaturesMask(x.ctx, T_XS_AVX)) { printf("W64_SELF_ERR=setmask gle=%lu\n", (unsigned long)GetLastError()); cx_xc_free(&x); return; }
    DWORD sc = SuspendThread(w->h);
    BOOL ok = GetThreadContext(w->h, x.ctx);
    DWORD gle = GetLastError();
    ResumeThread(w->h);
    if (sc != 0 || !ok) { printf("W64_SELF_ERR=get sc=%lu ok=%d gle=%lu\n", (unsigned long)sc, ok, (unsigned long)gle); cx_xc_free(&x); return; }
    DWORD64 m = 0;
    X.GetXStateFeaturesMask(x.ctx, &m);
    printf("W64_SELF_MASK=%llx\n", (unsigned long long)m);
    cx_print_hex("W64_SELF_X=", cx_xmm(x.ctx, 0), 8 * 16);
    DWORD len = 0;
    uint8_t *yh = cx_locate(x.ctx, CX_XS_AVX, &len);
    if (yh && len >= 8 * 16) cx_print_hex("W64_SELF_YH=", yh, 8 * 16);
    else printf("W64_SELF_ERR=locate p=%d len=%lu\n", yh != NULL, (unsigned long)len);
    cx_xc_free(&x);
}

/* Child (x86 build): worker with a known state; the x64 parent drives it through the WOW64 context API.
 *   --child=wow64_target [xmm|ymm]   (ymm: YMM worker + self view before W64_INFO) */
CHILD(wow64_target) {
    int ymm = argc > 1 && strcmp(argv[1], "ymm") == 0;
    cx_resolve();
    if (ymm && !cx_avx_ok()) { printf("W64_ERR=noavx avx=%d api=0x%llx\n", cpu_has(F_AVX), (unsigned long long)cx_enabled()); return 4; }
    int kind = ymm ? T_VK_YMM : T_VK_XMM;
    cx_args *a = cx_args_new(CX_W64_TAG, 1);
    if (!a) return 3;
    a->fcw_in = 0x0A7F; a->mxcsr_in = 0x3F80;
    cx_worker w;
    if (!cx_worker_start(&w, a, kind)) { printf("W64_ERR=start\n"); return 3; }
    if (ymm) cx_w64_self_view(&w);
    printf("W64_INFO=%llx %llx %llx %lu %llx %d\n", (unsigned long long)(uintptr_t)a, (unsigned long long)(uintptr_t)cx_loop_lo(kind),
           (unsigned long long)(uintptr_t)cx_loop_hi(kind), (unsigned long)w.tid, (unsigned long long)a->sp, (int)sizeof(void *));
    fflush(stdout);
    DWORD wr = WaitForSingleObject(w.h, 20000);
    if (wr != WAIT_OBJECT_0) printf("W64_TIMEOUT=1\n");
    cx_worker_finish(&w);
    printf("W64_GOUT=");
    for (int i = 0; i < CX_NGPR; i++) printf("%llx,", (unsigned long long)a->gout[i]);
    printf("\n");
    uint8_t v[8][16], yh[8][16];
    for (int i = 0; i < 8; i++) { memcpy(v[i], a->vout[i], 16); memcpy(yh[i], a->vout[i] + 16, 16); }
    cx_print_hex("W64_V=", v, sizeof v);
    if (ymm) cx_print_hex("W64_YH=", yh, sizeof yh);
    int fok = 1;
    for (int i = 0; i < 8; i++) if (memcmp(a->fout[i].b, a->fin[i].b, 10)) fok = 0;
    printf("W64_FOK=%d\n", fok);
    printf("W64_MX=%x %x %u\n", a->mxcsr_out, a->fcw_out, a->remaining);
    fflush(stdout);
    return 0;
}

#ifdef T_X64
#define CX_ThreadWow64Context 29
typedef LONG (NTAPI *cx_nqit_t)(HANDLE, ULONG, PVOID, ULONG, PULONG);
typedef LONG (NTAPI *cx_nsit_t)(HANDLE, ULONG, PVOID, ULONG);
typedef LONG (NTAPI *cx_rws_t)(HANDLE, PULONG);
static struct { int resolved; cx_nqit_t nqit; cx_nsit_t nsit; cx_rws_t rws; } N;
static LONG g_w64_status;            /* NTSTATUS of the last ntdll-path call */
static void cx_nt_resolve(void) {
    if (N.resolved) return;
    N.nqit = (cx_nqit_t)t_proc("ntdll.dll", "NtQueryInformationThread");
    N.nsit = (cx_nsit_t)t_proc("ntdll.dll", "NtSetInformationThread");
    N.rws = (cx_rws_t)t_proc("ntdll.dll", "RtlWow64SuspendThread");
    N.resolved = 1;
}
/* Wow64SuspendThread, else its implementation ntdll!RtlWow64SuspendThread, else SuspendThread (which the
 * 19041 implementation reduces to for an x86 target). Previous suspend count or (DWORD)-1. */
static DWORD cx_w64_suspend(HANDLE h, const char **via) {
    if (X.Wow64SuspendThread) { *via = "kernel32!Wow64SuspendThread"; return X.Wow64SuspendThread(h); }
    if (N.rws) {
        *via = "ntdll!RtlWow64SuspendThread";
        ULONG prev = 0xFFFFFFFFu;
        g_w64_status = N.rws(h, &prev);
        return g_w64_status < 0 ? (DWORD)-1 : (DWORD)prev;
    }
    *via = "SuspendThread";
    return SuspendThread(h);
}
static const char *cx_w64_get_via(void) { return X.Wow64GetThreadContext ? "kernel32!Wow64GetThreadContext" : "ntdll!NtQueryInformationThread(ThreadWow64Context)"; }
static const char *cx_w64_set_via(void) { return X.Wow64SetThreadContext ? "kernel32!Wow64SetThreadContext" : "ntdll!NtSetInformationThread(ThreadWow64Context)"; }
static BOOL cx_w64_get(HANDLE h, WOW64_CONTEXT *c) {
    if (X.Wow64GetThreadContext) return X.Wow64GetThreadContext(h, c);
    g_w64_status = N.nqit(h, CX_ThreadWow64Context, c, CX_W64_CTX_LEN, NULL);
    SetLastError((DWORD)g_w64_status);
    return g_w64_status >= 0;
}
static BOOL cx_w64_set(HANDLE h, WOW64_CONTEXT *c) {
    if (X.Wow64SetThreadContext) return X.Wow64SetThreadContext(h, c);
    g_w64_status = N.nsit(h, CX_ThreadWow64Context, c, CX_W64_CTX_LEN);
    SetLastError((DWORD)g_w64_status);
    return g_w64_status >= 0;
}
#define CX_NEED_W64API() do { cx_resolve(); cx_nt_resolve(); \
    if (!X.Wow64GetThreadContext && !N.nqit) SKIP("neither Wow64GetThreadContext nor NtQueryInformationThread exported"); \
    if (!X.Wow64SetThreadContext && !N.nsit) SKIP("neither Wow64SetThreadContext nor NtSetInformationThread exported"); } while (0)

static int cx_hexbytes(const char *s, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) { unsigned v; if (!s || sscanf(s + 2 * i, "%2x", &v) != 1) return 0; out[i] = (uint8_t)v; }
    return 1;
}
static const char *cx_find_line(const char *buf, const char *prefix) {
    const char *p = strstr(buf, prefix);
    return p ? p + strlen(prefix) : NULL;
}

/* The x86 sibling (build/x86/context_xstate.exe --child=wow64_target <mode>) */
typedef struct cx_sib {
    PROCESS_INFORMATION pi;
    HANDLE rd;
    size_t n;
    int exited;
    unsigned long long args, lo, hi, sp;
    unsigned long tid;
    int psz;
    char out[32768];
} cx_sib;
static cx_sib g_sib;
static void cx_sib_pump(cx_sib *s) {
    DWORD avail = 0;
    while (PeekNamedPipe(s->rd, NULL, 0, NULL, &avail, NULL) && avail) {
        DWORD got = 0;
        if (s->n + avail >= sizeof s->out) avail = (DWORD)(sizeof s->out - s->n - 1);
        if (!avail || !ReadFile(s->rd, s->out + s->n, avail, &got, NULL) || !got) break;
        s->n += got; s->out[s->n] = 0;
    }
}
static void cx_sib_close(cx_sib *s) {
    if (s->pi.hProcess) {
        if (WaitForSingleObject(s->pi.hProcess, 0) != WAIT_OBJECT_0) { TerminateProcess(s->pi.hProcess, 1); WaitForSingleObject(s->pi.hProcess, 2000); }
        CloseHandle(s->pi.hProcess); CloseHandle(s->pi.hThread);
    }
    if (s->rd) CloseHandle(s->rd);
    s->pi.hProcess = s->pi.hThread = s->rd = NULL;
}
/* 1: running, W64_INFO parsed. 0: cannot run here (why = SKIP reason). -1: started but failed (why). */
static int cx_sib_start(cx_sib *s, const char *mode, char *why, size_t cap) {
    memset(s, 0, sizeof *s);
    char path[MAX_PATH * 2];
    snprintf(path, sizeof path, "%s", t_self_path());
    char *p = NULL;
    for (char *q = path; (q = strstr(q, "\\x64\\")) != NULL; q++) p = q;
    if (!p) for (char *q = path; (q = strstr(q, "/x64/")) != NULL; q++) p = q;
    if (!p) { snprintf(why, cap, "cannot derive the x86 sibling from %s", path); return 0; }
    p[1] = 'x'; p[2] = '8'; p[3] = '6';
    if (GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES) { snprintf(why, cap, "x86 build not found at %s", path); return 0; }
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE wr = NULL;
    if (!CreatePipe(&s->rd, &wr, &sa, 1 << 16)) { snprintf(why, cap, "CreatePipe GLE=%lu", (unsigned long)GetLastError()); return -1; }
    SetHandleInformation(s->rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si; memset(&si, 0, sizeof si);
    si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr; si.hStdError = wr; si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    char cmd[MAX_PATH * 2 + 64];
    snprintf(cmd, sizeof cmd, "\"%s\" --child=wow64_target %s", path, mode);
    BOOL cp = CreateProcessA(path, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &s->pi);
    DWORD cerr = GetLastError();
    CloseHandle(wr);
    if (!cp) { CloseHandle(s->rd); s->rd = NULL; snprintf(why, cap, "cannot start the x86 sibling (GLE=%lu): WOW64 unavailable", (unsigned long)cerr); return 0; }
    const char *info = NULL;
    double t0 = t_now_ms();
    while (t_now_ms() - t0 < 15000) {
        cx_sib_pump(s);
        const char *q = cx_find_line(s->out, "W64_INFO=");
        if (q && strchr(q, '\n')) { info = q; break; }
        if (cx_find_line(s->out, "W64_ERR=")) break;
        if (WaitForSingleObject(s->pi.hProcess, 5) == WAIT_OBJECT_0) { cx_sib_pump(s); s->exited = 1; break; }
    }
    if (info) sscanf(info, "%llx %llx %llx %lu %llx %d", &s->args, &s->lo, &s->hi, &s->tid, &s->sp, &s->psz);
    if (!info || s->psz != 4) {
        snprintf(why, cap, "x86 sibling did not report (exited=%d psz=%d) output: %.400s", s->exited, s->psz, s->out);
        cx_sib_close(s);
        return -1;
    }
    return 1;
}
static int cx_sib_stop_flag(cx_sib *s) {
    LONG one = 1;
    SIZE_T wn = 0;
    BOOL ok = WriteProcessMemory(s->pi.hProcess, (void *)(uintptr_t)(s->args + CXO_STOP), &one, sizeof one, &wn);
    CHECKF(ok, "WriteProcessMemory(stop) GLE=%lu", (unsigned long)GetLastError());
    return ok;
}
/* Drain until the sibling exits (bounded); 1 if it exited on its own with code 0. */
static int cx_sib_finish(cx_sib *s, DWORD ms) {
    double t0 = t_now_ms();
    int ok = 0;
    for (;;) {
        cx_sib_pump(s);
        if (WaitForSingleObject(s->pi.hProcess, 5) == WAIT_OBJECT_0) {
            cx_sib_pump(s);
            DWORD code = 0;
            GetExitCodeProcess(s->pi.hProcess, &code);
            ok = code == 0;
            CHECKF(ok, "x86 sibling exit code 0x%lx", (unsigned long)code);
            break;
        }
        if (t_now_ms() - t0 > ms) { CHECKF(0, "x86 sibling did not exit within %lu ms", (unsigned long)ms); break; }
    }
    cx_sib_close(s);
    t_log("x86 sibling output:\n%s", s->out);
    CHECKF(!cx_find_line(s->out, "W64_TIMEOUT="), "x86 worker never observed the stop flag");
    return ok;
}
static HANDLE cx_sib_thread(cx_sib *s) {
    HANDLE th = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, FALSE, s->tid);
    CHECKF(th != NULL, "OpenThread(%lu) GLE=%lu", s->tid, (unsigned long)GetLastError());
    return th;
}
#define CX_SIB_START(s, mode) do { char why_[600]; int st_ = cx_sib_start((s), (mode), why_, sizeof why_); \
    if (st_ == 0) { SKIP("%s", why_); } \
    REQUIREF(st_ > 0, "%s", why_); } while (0)

/* The x86 worker's registers through the WOW64 view (x87 stack, FCW, XMM0-7, MXCSR, GPRs, EIP/ESP) */
static void cx_w64_check_state(WOW64_CONTEXT *wc, const cx_args *exp, const cx_sib *s, const char *what) {
    static const char *gn[5] = { "ebx", "edx", "esi", "edi", "ebp" };
    DWORD g[5] = { wc->Ebx, wc->Edx, wc->Esi, wc->Edi, wc->Ebp };
    for (int i = 0; i < 5; i++)
        CHECKF(g[i] == (DWORD)exp->gin[i], "%s %s 0x%lx want 0x%lx", what, gn[i], (unsigned long)g[i], (unsigned long)(DWORD)exp->gin[i]);
    CHECKF(wc->Eip >= s->lo && wc->Eip <= s->hi, "%s Eip 0x%lx outside [0x%llx, 0x%llx]", what, (unsigned long)wc->Eip, s->lo, s->hi);
    CHECKF(wc->Esp == s->sp, "%s Esp 0x%lx want 0x%llx", what, (unsigned long)wc->Esp, s->sp);
    CHECKF(wc->Eax == (DWORD)(s->args + CXO_READY), "%s Eax 0x%lx want &ready", what, (unsigned long)wc->Eax);
    CHECKF((wc->SegCs & 3) == 3 && wc->SegCs != cx_own_cs(), "%s SegCs 0x%lx (64-bit CS is 0x%x)", what, (unsigned long)wc->SegCs, cx_own_cs());
    char lab[64];
    for (int i = 0; i < 8; i++) {
        snprintf(lab, sizeof lab, "%s xmm%d", what, i);
        CHECK_REG(lab, wc->ExtendedRegisters + 160 + 16 * i, exp->vin[i], 16);
        snprintf(lab, sizeof lab, "%s ST(%d) FloatSave", what, i);
        CHECK_REG(lab, wc->FloatSave.RegisterArea + 10 * i, exp->fin[i].b, 10);
        snprintf(lab, sizeof lab, "%s ST(%d) FXSAVE", what, i);
        CHECK_REG(lab, wc->ExtendedRegisters + 32 + 16 * i, exp->fin[i].b, 10);
    }
    uint16_t fx_fcw; memcpy(&fx_fcw, wc->ExtendedRegisters, 2);
    CHECKF((wc->FloatSave.ControlWord & CX_FCW_BITS) == (0x0A7F & CX_FCW_BITS), "%s FloatSave FCW 0x%lx want 0x0A7F", what,
           (unsigned long)wc->FloatSave.ControlWord);
    CHECKF((fx_fcw & CX_FCW_BITS) == (0x0A7F & CX_FCW_BITS), "%s FXSAVE FCW 0x%x want 0x0A7F", what, fx_fcw);
    uint32_t mx; memcpy(&mx, wc->ExtendedRegisters + 24, 4);
    CHECKF(mx == 0x3F80, "%s MXCSR 0x%x want 0x3F80", what, mx);
}
#endif

TEST(wow64_cross_process_context) {
#ifdef T_X64
    CX_NEED_W64API();
    cx_sib *s = &g_sib;
    CX_SIB_START(s, "xmm");
    t_info("api", "get=%s set=%s", cx_w64_get_via(), cx_w64_set_via());
    static cx_args exp;
    cx_args_fill(&exp, CX_W64_TAG, 1);
    uint8_t nx2[16];
    t_pattern_reg(nx2, 16, 0xB1FF, 2);
    const DWORD nebx = 0x13572468;
    const uint32_t nmx = (cx_mxcsr_mask() & 0x40) ? CX_W64_NEW_MXCSR_DAZ : CX_W64_NEW_MXCSR;
    HANDLE th = cx_sib_thread(s);
    int set_done = 0;
    if (th) {
        const char *via = "";
        DWORD sc = cx_w64_suspend(th, &via);
        t_info("suspend_via", "%s", via);
        CHECKF(sc == 0, "%s previous count %lu", via, (unsigned long)sc);
        static WOW64_CONTEXT wc;
        memset(&wc, CX_SENT, sizeof wc);
        wc.ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_FLOATING_POINT | WOW64_CONTEXT_EXTENDED_REGISTERS;
        if (cx_w64_get(th, &wc)) {
            cx_w64_check_state(&wc, &exp, s, "WOW64");
            /* the documented API and its ntdll implementation (ThreadWow64Context, length 0x2cc) agree exactly */
            if (X.Wow64GetThreadContext && N.nqit) {
                static WOW64_CONTEXT w2;
                memset(&w2, CX_SENT, sizeof w2);
                w2.ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_FLOATING_POINT | WOW64_CONTEXT_EXTENDED_REGISTERS;
                ULONG rl = 0xDEAD;
                LONG st = N.nqit(th, CX_ThreadWow64Context, &w2, CX_W64_CTX_LEN, &rl);
                CHECKF(st >= 0, "NtQueryInformationThread(ThreadWow64Context) 0x%lx", (unsigned long)st);
                t_info("nqit_return_length", "0x%lx", (unsigned long)rl);
                if (st >= 0) CHECK_MEM(&w2, &wc, sizeof wc);
            }
            /* ThreadWow64Context length is exactly 0x2cc: shorter and longer buffers fail */
            if (N.nqit) {
                static uint8_t big[CX_W64_CTX_LEN + 64];
                memset(big, CX_SENT, sizeof big);
                ((WOW64_CONTEXT *)big)->ContextFlags = WOW64_CONTEXT_CONTROL;
                LONG s1 = N.nqit(th, CX_ThreadWow64Context, big, CX_W64_CTX_LEN - 4, NULL);
                LONG s2 = N.nqit(th, CX_ThreadWow64Context, big, CX_W64_CTX_LEN + 64, NULL);
                t_info("nqit_bad_length", "0x2c8 -> 0x%lx, 0x30c -> 0x%lx", (unsigned long)s1, (unsigned long)s2);
                CHECKF(s1 < 0, "ThreadWow64Context with length 0x2c8 succeeded (0x%lx)", (unsigned long)s1);
                CHECKF(s2 < 0, "ThreadWow64Context with length 0x30c succeeded (0x%lx)", (unsigned long)s2);
                CHECKF(cx_all_sent(big + CX_W64_CTX_LEN, 64), "ThreadWow64Context wrote past 0x2cc bytes");
            }
            /* WOW64_CONTEXT has no XSTATE area: a WOW64_CONTEXT_XSTATE request must not write past 0x2cc bytes;
             * success or failure is recorded only. */
            {
                static uint8_t xb[CX_W64_CTX_LEN + 256];
                memset(xb, CX_SENT, sizeof xb);
                ((WOW64_CONTEXT *)xb)->ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_EXTENDED_REGISTERS | WOW64_CONTEXT_XSTATE;
                SetLastError(0);
                BOOL r = cx_w64_get(th, (WOW64_CONTEXT *)xb);
                t_info("wow64_xstate_request", "ok=%d gle=0x%lx flags_out=0x%lx", r, (unsigned long)GetLastError(),
                       (unsigned long)((WOW64_CONTEXT *)xb)->ContextFlags);
                CHECKF(cx_all_sent(xb + CX_W64_CTX_LEN, 256), "WOW64_CONTEXT_XSTATE request wrote past the 0x2cc-byte WOW64_CONTEXT");
                if (r) CHECK_MEM(((WOW64_CONTEXT *)xb)->ExtendedRegisters + 160, wc.ExtendedRegisters + 160, 128);
            }
            /* native view of the same thread: an x64 context (observation; see wow64_ymm_views_agree) */
            CONTEXT *nc = cx_plain(CONTEXT_CONTROL);
            if (nc && GetThreadContext(th, nc)) t_info("native_view", "Rip=%p SegCs=0x%x", (void *)(uintptr_t)nc->Rip, (unsigned)nc->SegCs);
            cx_free(nc);
            /* modify Ebx, XMM2, MXCSR and the x87 control word (both the FNSAVE and the FXSAVE copy), read back */
            wc.Ebx = nebx;
            memcpy(wc.ExtendedRegisters + 160 + 32, nx2, 16);
            memcpy(wc.ExtendedRegisters + 24, &nmx, 4);
            uint16_t nfcw = CX_W64_NEW_FCW;
            memcpy(wc.ExtendedRegisters, &nfcw, 2);
            wc.FloatSave.ControlWord = (wc.FloatSave.ControlWord & 0xFFFF0000u) | nfcw;
            /* two sets, one flag group each: a set must change only the groups named in ContextFlags */
            static WOW64_CONTEXT chk;
            wc.ContextFlags = WOW64_CONTEXT_INTEGER;
            int s_int = cx_w64_set(th, &wc);
            CHECKF(s_int, "%s(INTEGER) GLE/status=0x%lx", cx_w64_set_via(), (unsigned long)GetLastError());
            memset(&chk, 0, sizeof chk);
            chk.ContextFlags = WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER;
            if (s_int && cx_w64_get(th, &chk))
                CX_CHECK_WGAP("an INTEGER-only Wow64SetThreadContext clears Ebp (CONTROL)", chk.Ebp == (DWORD)exp.gin[4],
                              "Ebp after an INTEGER-only set 0x%lx want 0x%lx", (unsigned long)chk.Ebp, (unsigned long)(DWORD)exp.gin[4]);
            wc.ContextFlags = WOW64_CONTEXT_FLOATING_POINT | WOW64_CONTEXT_EXTENDED_REGISTERS;
            int s_fp = cx_w64_set(th, &wc);
            CHECKF(s_fp, "%s(FP|EXT) GLE/status=0x%lx", cx_w64_set_via(), (unsigned long)GetLastError());
            memset(&chk, 0, sizeof chk);
            chk.ContextFlags = WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER;
            if (s_fp && cx_w64_get(th, &chk)) {
                CHECKF(chk.Ebx == nebx, "Ebx 0x%lx after an FP|EXT set want 0x%lx", (unsigned long)chk.Ebx, (unsigned long)nebx);
                t_info("ebp_after_fp_set", "0x%lx", (unsigned long)chk.Ebp);
            }
            set_done = s_int && s_fp;
            memset(&wc, CX_SENT, sizeof wc);
            wc.ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_FLOATING_POINT | WOW64_CONTEXT_EXTENDED_REGISTERS;
            if (cx_w64_get(th, &wc)) {
                CHECK_EQ(wc.Ebx, nebx);
                CHECK_REG("WOW64 xmm2 after set", wc.ExtendedRegisters + 160 + 32, nx2, 16);
                for (int i = 0; i < 8; i++) if (i != 2) {
                    char lab[48]; snprintf(lab, sizeof lab, "WOW64 xmm%d after set", i);
                    CHECK_REG(lab, wc.ExtendedRegisters + 160 + 16 * i, exp.vin[i], 16);
                }
                uint32_t mx; memcpy(&mx, wc.ExtendedRegisters + 24, 4);
                CHECKF(mx == nmx, "MXCSR after set 0x%x want 0x%x", mx, nmx);
                CHECKF((wc.FloatSave.ControlWord & CX_FCW_BITS) == (nfcw & CX_FCW_BITS), "FCW after set 0x%lx want 0x%x",
                       (unsigned long)wc.FloatSave.ControlWord, nfcw);
                CHECK_EQ(wc.Esi, (DWORD)exp.gin[2]);
                /* EBP, EIP, ESP belong to WOW64_CONTEXT_CONTROL, which was not part of the set */
                CX_CHECK_WGAP("an INTEGER-only Wow64SetThreadContext clears Ebp (CONTROL)", wc.Ebp == (DWORD)exp.gin[4],
                              "Ebp after INTEGER and FP|EXT sets 0x%lx want 0x%lx", (unsigned long)wc.Ebp, (unsigned long)(DWORD)exp.gin[4]);
                CHECKF(wc.Eip >= s->lo && wc.Eip <= s->hi && wc.Esp == s->sp, "Eip/Esp after set 0x%lx/0x%lx", (unsigned long)wc.Eip,
                       (unsigned long)wc.Esp);
                CHECK_REG("WOW64 ST(0) after set", wc.FloatSave.RegisterArea, exp.fin[0].b, 10);
            } else CHECKF(0, "%s (re-read) GLE/status=0x%lx", cx_w64_get_via(), (unsigned long)GetLastError());
        } else CHECKF(0, "%s GLE/status=0x%lx", cx_w64_get_via(), (unsigned long)GetLastError());
        cx_sib_stop_flag(s);
        DWORD rc = ResumeThread(th);
        CHECKF(rc == 1, "ResumeThread previous count %lu", (unsigned long)rc);
        CloseHandle(th);
    }
    cx_sib_finish(s, 25000);
    const char *go = cx_find_line(s->out, "W64_GOUT="), *vv = cx_find_line(s->out, "W64_V="), *mxl = cx_find_line(s->out, "W64_MX="),
               *fok = cx_find_line(s->out, "W64_FOK=");
    CHECKF(go && vv && mxl && fok, "x86 sibling did not report its registers: %.300s", s->out);
    if (go && vv && mxl && fok && set_done) {
        /* the resumed x86 thread observes exactly the state that was set, and nothing else changed */
        unsigned long long g[5] = { 0 };
        sscanf(go, "%llx,%llx,%llx,%llx,%llx", &g[0], &g[1], &g[2], &g[3], &g[4]);
        CHECK_EQ(g[0], nebx);
        for (int i = 1; i < 4; i++) CHECK_EQ(g[i], (DWORD)exp.gin[i]);
        CX_CHECK_WGAP("an INTEGER-only Wow64SetThreadContext clears Ebp (CONTROL)", g[4] == (DWORD)exp.gin[4],
                      "x86 thread observed ebp 0x%llx want 0x%lx", g[4], (unsigned long)(DWORD)exp.gin[4]);
        uint8_t v[8][16];
        if (cx_hexbytes(vv, &v[0][0], 128)) {
            for (int i = 0; i < 8; i++) {
                char lab[48]; snprintf(lab, sizeof lab, "x86 thread xmm%d observed", i);
                CHECK_REG(lab, v[i], i == 2 ? nx2 : exp.vin[i], 16);
            }
        } else CHECKF(0, "bad W64_V line");
        unsigned mxo = 0, fcwo = 0, rem = 0;
        sscanf(mxl, "%x %x %u", &mxo, &fcwo, &rem);
        CHECKF(mxo == nmx, "x86 thread observed MXCSR 0x%x want 0x%x", mxo, nmx);
        CHECKF((fcwo & CX_FCW_BITS) == (CX_W64_NEW_FCW & CX_FCW_BITS), "x86 thread observed FCW 0x%x want 0x%x", fcwo, CX_W64_NEW_FCW);
        CHECKF(atoi(fok) == 1, "x86 thread's x87 stack changed by a control-word-only set");
        CHECKF(rem != 0, "x86 worker exhausted its loop budget");
    }
#else
    SKIP("x64 build only (an x86 process cannot drive a WOW64 target through Wow64*ThreadContext)");
#endif
}

TEST(wow64_suspend_count) {
#ifdef T_X64
    CX_NEED_W64API();
    cx_sib *s = &g_sib;
    CX_SIB_START(s, "xmm");
    HANDLE th = cx_sib_thread(s);
    if (th) {
        const char *via = "";
        /* Wow64SuspendThread and SuspendThread/ResumeThread share one documented suspend count */
        DWORD c0 = cx_w64_suspend(th, &via);
        DWORD c1 = cx_w64_suspend(th, &via);
        DWORD c2 = SuspendThread(th);
        t_info("suspend_via", "%s", via);
        CHECKF(c0 == 0 && c1 == 1 && c2 == 2, "suspend counts %lu %lu %lu want 0 1 2", (unsigned long)c0, (unsigned long)c1, (unsigned long)c2);
        DWORD r0 = ResumeThread(th), r1 = ResumeThread(th);
        CHECKF(r0 == 3 && r1 == 2, "resume counts %lu %lu want 3 2", (unsigned long)r0, (unsigned long)r1);
        /* still suspended (count 1): two reads 30 ms apart see the same loop counter (ECX) */
        static WOW64_CONTEXT a, b;
        memset(&a, 0, sizeof a); memset(&b, 0, sizeof b);
        a.ContextFlags = b.ContextFlags = WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER;
        BOOL ga = cx_w64_get(th, &a);
        Sleep(30);
        BOOL gb = cx_w64_get(th, &b);
        CHECKF(ga && gb, "%s GLE/status=0x%lx", cx_w64_get_via(), (unsigned long)GetLastError());
        if (ga && gb) {
            CHECKF(a.Ecx == b.Ecx && a.Eip == b.Eip, "thread ran while suspended: ECX 0x%lx -> 0x%lx", (unsigned long)a.Ecx, (unsigned long)b.Ecx);
            CHECKF(a.Eip >= s->lo && a.Eip <= s->hi, "Eip 0x%lx outside the spin loop", (unsigned long)a.Eip);
        }
        DWORD r2 = ResumeThread(th);
        CHECKF(r2 == 1, "ResumeThread previous count %lu want 1", (unsigned long)r2);
        /* running again: the loop counter advances */
        Sleep(50);
        DWORD c3 = cx_w64_suspend(th, &via);
        CHECKF(c3 == 0, "%s after full resume: previous count %lu want 0", via, (unsigned long)c3);
        memset(&b, 0, sizeof b);
        b.ContextFlags = WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER;
        if (ga && cx_w64_get(th, &b)) CHECKF(b.Ecx != a.Ecx, "x86 thread did not run after ResumeThread (ECX 0x%lx)", (unsigned long)b.Ecx);
        /* documented failure value: (DWORD)-1 for a handle without THREAD_SUSPEND_RESUME */
        HANDLE lim = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, s->tid);
        if (lim) {
            SetLastError(0);
            DWORD bad = cx_w64_suspend(lim, &via);
            t_info("suspend_without_access", "%lu gle=%lu", (unsigned long)bad, (unsigned long)GetLastError());
            CHECKF(bad == (DWORD)-1, "%s without THREAD_SUSPEND_RESUME returned %lu", via, (unsigned long)bad);
            if (bad != (DWORD)-1) ResumeThread(th);
            CloseHandle(lim);
        }
        cx_sib_stop_flag(s);
        DWORD r3 = ResumeThread(th);
        CHECKF(r3 == 1, "final ResumeThread previous count %lu want 1", (unsigned long)r3);
        CloseHandle(th);
    }
    cx_sib_finish(s, 25000);
#else
    SKIP("x64 build only");
#endif
}

#ifdef T_X64
static DWORD WINAPI cx_nop_thread(void *p) { (void)p; return 0; }
#endif
TEST(wow64_context_native_thread_rejected) {
#ifdef T_X64
    /* ThreadWow64Context on a thread of a native x64 process: pinned STATUS_INVALID_PARAMETER for set (no x86
     * WOW64 machine), STATUS_INFO_LENGTH_MISMATCH for a wrong length; get fails. */
    CX_NEED_W64API();
    HANDLE h = CreateThread(NULL, 0, cx_nop_thread, NULL, CREATE_SUSPENDED, NULL);
    if (!h) SKIP("CreateThread failed");
    static WOW64_CONTEXT wc;
    memset(&wc, 0, sizeof wc);
    wc.ContextFlags = WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER;
    if (N.nqit) {
        LONG st = N.nqit(h, CX_ThreadWow64Context, &wc, CX_W64_CTX_LEN, NULL);
        t_info("get_status", "0x%lx", (unsigned long)st);
        CHECKF(st < 0, "ThreadWow64Context query on a native x64 thread succeeded (0x%lx)", (unsigned long)st);
    }
    if (X.Wow64GetThreadContext) {
        memset(&wc, 0, sizeof wc);
        wc.ContextFlags = WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER;
        SetLastError(0);
        BOOL r = X.Wow64GetThreadContext(h, &wc);
        t_info("wow64getthreadcontext", "%d gle=%lu", r, (unsigned long)GetLastError());
        CHECKF(!r, "Wow64GetThreadContext on a native x64 thread succeeded");
    }
    if (N.nsit) {
        memset(&wc, 0, sizeof wc);
        wc.ContextFlags = WOW64_CONTEXT_INTEGER;
        LONG s1 = N.nsit(h, CX_ThreadWow64Context, &wc, CX_W64_CTX_LEN - 4);
        LONG s2 = N.nsit(h, CX_ThreadWow64Context, &wc, CX_W64_CTX_LEN);
        t_info("set_status", "len 0x2c8 -> 0x%lx, len 0x2cc -> 0x%lx", (unsigned long)s1, (unsigned long)s2);
        CX_CHECK_WGAP("ThreadWow64Context set status", (unsigned long)s1 == 0xC0000004ul,
                      "wrong length: 0x%lx want STATUS_INFO_LENGTH_MISMATCH", (unsigned long)s1);
        CX_CHECK_WGAP("ThreadWow64Context set status", (unsigned long)s2 == 0xC000000Dul,
                      "native thread: 0x%lx want STATUS_INVALID_PARAMETER", (unsigned long)s2);
    }
    if (X.Wow64SetThreadContext) {
        memset(&wc, 0, sizeof wc);
        wc.ContextFlags = WOW64_CONTEXT_INTEGER;
        SetLastError(0);
        BOOL r = X.Wow64SetThreadContext(h, &wc);
        DWORD e = GetLastError();
        t_info("wow64setthreadcontext", "%d gle=%lu", r, (unsigned long)e);
        CHECKF(!r, "Wow64SetThreadContext on a native x64 thread succeeded");
        CX_CHECK_WGAP("Wow64SetThreadContext error", r || e == ERROR_INVALID_PARAMETER, "GLE %lu want ERROR_INVALID_PARAMETER", (unsigned long)e);
    }
    ResumeThread(h);
    CHECK(WaitForSingleObject(h, CX_TMO) == WAIT_OBJECT_0);
    DWORD code = 1;
    GetExitCodeThread(h, &code);
    CHECK_EQ(code, 0);
    CloseHandle(h);
#else
    SKIP("x64 build only");
#endif
}

/* YMM uppers of an x86 (WOW64) thread: the x86 self view (GetThreadContext with CONTEXT_XSTATE inside the
 * WOW64 process) and the native x64 view of the same suspended thread report the same registers; the
 * WOW64_CONTEXT view carries XMM0-7 only. */
TEST(wow64_ymm_views_agree) {
#ifdef T_X64
    CX_NEED_W64API();
    CX_NEED_AVX();
    cx_sib *s = &g_sib;
    CX_SIB_START(s, "ymm");
    static cx_args exp;
    cx_args_fill(&exp, CX_W64_TAG, 1);
    uint8_t want_x[8][16], want_yh[8][16];
    for (int i = 0; i < 8; i++) { memcpy(want_x[i], exp.vin[i], 16); memcpy(want_yh[i], exp.vin[i] + 16, 16); }
    /* 1. x86 self view (printed by the sibling before W64_INFO) */
    const char *serr = cx_find_line(s->out, "W64_SELF_ERR=");
    const char *smask = cx_find_line(s->out, "W64_SELF_MASK="), *sx = cx_find_line(s->out, "W64_SELF_X="), *syh = cx_find_line(s->out, "W64_SELF_YH=");
    uint8_t self_x[8][16], self_yh[8][16];
    int have_self = 0;
    if (serr) {
        char e[120]; sscanf(serr, "%119[^\n]", e);
        CX_CHECK_WGAP("x86 GetThreadContext(CONTEXT_XSTATE)", 0, "x86 self view failed: %s", e);
    } else {
        unsigned long long m = 0;
        if (smask) sscanf(smask, "%llx", &m);
        t_info("self_mask", "0x%llx", m);
        CHECKF(m & T_XS_AVX, "x86 self view: XSTATE mask 0x%llx lacks AVX while YMM uppers are nonzero", m);
        if (cx_hexbytes(sx, &self_x[0][0], 128) && cx_hexbytes(syh, &self_yh[0][0], 128)) {
            have_self = 1;
            for (int i = 0; i < 8; i++) {
                char lab[48];
                snprintf(lab, sizeof lab, "x86 self view xmm%d", i);
                CHECK_REG(lab, self_x[i], want_x[i], 16);
                snprintf(lab, sizeof lab, "x86 self view ymm%d upper", i);
                CHECK_REG(lab, self_yh[i], want_yh[i], 16);
            }
        } else CHECKF(0, "x86 sibling self view lines missing: %.300s", s->out);
    }
    HANDLE th = cx_sib_thread(s);
    if (th) {
        const char *via = "";
        DWORD sc = cx_w64_suspend(th, &via);
        CHECKF(sc == 0, "%s previous count %lu", via, (unsigned long)sc);
        /* 2. WOW64 view: XMM0-7 (FXSAVE), nothing more */
        static WOW64_CONTEXT wc;
        memset(&wc, CX_SENT, sizeof wc);
        wc.ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_EXTENDED_REGISTERS;
        if (cx_w64_get(th, &wc)) {
            for (int i = 0; i < 8; i++) {
                char lab[48]; snprintf(lab, sizeof lab, "WOW64 view xmm%d", i);
                CHECK_REG(lab, wc.ExtendedRegisters + 160 + 16 * i, want_x[i], 16);
            }
            CHECKF(wc.Eip >= s->lo && wc.Eip <= s->hi, "WOW64 Eip 0x%lx outside the YMM spin loop", (unsigned long)wc.Eip);
        } else CHECKF(0, "%s GLE/status=0x%lx", cx_w64_get_via(), (unsigned long)GetLastError());
        /* 3. native x64 view with CONTEXT_XSTATE. A thread suspended in its 32-bit code reports that code's
         * registers (compatibility-mode CS, RIP in the loop): its XMM0-7 and YMM0-7 uppers must be the x86
         * thread's. Stopped elsewhere (in the WOW64 layer) the view is recorded only. */
        cx_xc x;
        if (cx_xc_init(&x, CX_STD | CONTEXT_XSTATE)) {
            if (X.SetXStateFeaturesMask(x.ctx, T_XS_AVX) && GetThreadContext(th, x.ctx)) {
                CONTEXT *c = x.ctx;
                uint64_t m = cx_mask(c);
                DWORD len = 0;
                uint8_t *yh = cx_locate(c, CX_XS_AVX, &len);
                t_info("x64_view", "Rip=0x%llx SegCs=0x%x mask=0x%llx", (unsigned long long)c->Rip, (unsigned)c->SegCs, (unsigned long long)m);
                int in32 = c->SegCs == wc.SegCs && c->Rip >= s->lo && c->Rip <= s->hi;
                if (in32) {
                    CHECKF(c->Rip == wc.Eip && c->Rsp == wc.Esp, "x64 view Rip/Rsp 0x%llx/0x%llx vs WOW64 Eip/Esp 0x%lx/0x%lx",
                           (unsigned long long)c->Rip, (unsigned long long)c->Rsp, (unsigned long)wc.Eip, (unsigned long)wc.Esp);
                    CHECKF((uint32_t)c->Rbx == wc.Ebx && (uint32_t)c->Rsi == wc.Esi, "x64 view GPRs differ from the WOW64 view");
                    CHECKF(m & T_XS_AVX, "x64 view: XSTATE mask 0x%llx lacks AVX", (unsigned long long)m);
                    for (int i = 0; i < 8; i++) {
                        char lab[48];
                        snprintf(lab, sizeof lab, "x64 view xmm%d", i);
                        CHECK_REG(lab, cx_xmm(c, i), want_x[i], 16);
                        if (yh) {
                            snprintf(lab, sizeof lab, "x64 view ymm%d upper", i);
                            CHECK_REG(lab, yh + 16 * i, want_yh[i], 16);
                            if (have_self) {
                                snprintf(lab, sizeof lab, "x64 vs x86 self view ymm%d upper", i);
                                CHECK_REG(lab, yh + 16 * i, self_yh[i], 16);
                            }
                        }
                    }
                    CHECKF(yh != NULL, "x64 view: LocateXStateFeature(AVX) NULL");
                } else t_note("x64 view of the WOW64 thread is not in its 32-bit code (Rip 0x%llx SegCs 0x%x): YMM comparison not applicable",
                              (unsigned long long)c->Rip, (unsigned)c->SegCs);
            } else t_note("x64 GetThreadContext(CONTEXT_XSTATE) on the WOW64 thread failed, GLE=%lu", (unsigned long)GetLastError());
            cx_xc_free(&x);
        } else t_note("InitializeContext(CONTEXT_XSTATE) failed, GLE=%lu", (unsigned long)x.err);
        cx_sib_stop_flag(s);
        DWORD rc = ResumeThread(th);
        CHECKF(rc == 1, "ResumeThread previous count %lu", (unsigned long)rc);
        CloseHandle(th);
    }
    cx_sib_finish(s, 25000);
    /* 4. the x86 thread itself still holds its YMM uppers after all three views were taken */
    uint8_t yh_out[8][16];
    if (cx_hexbytes(cx_find_line(s->out, "W64_YH="), &yh_out[0][0], 128)) {
        for (int i = 0; i < 8; i++) {
            char lab[48]; snprintf(lab, sizeof lab, "x86 thread ymm%d upper at exit", i);
            CHECK_REG(lab, yh_out[i], want_yh[i], 16);
        }
    } else CHECKF(0, "x86 sibling did not report W64_YH: %.300s", s->out);
#else
    SKIP("x64 build only (the x86 build's own YMM context cases cover the self view)");
#endif
}

/* ======================================================================================================
 * 12. PERF: context capture rates
 * ====================================================================================================== */
typedef struct { HANDLE h; CONTEXT *c; } cx_bench_ctx;
static void cx_bench_get(void *p) {
    cx_bench_ctx *b = (cx_bench_ctx *)p;
    for (int i = 0; i < 16; i++) T_SINK(GetThreadContext(b->h, b->c));
}
static void cx_bench_capture(void *p) {
    cx_bench_ctx *b = (cx_bench_ctx *)p;
    for (int i = 0; i < 64; i++) X.RtlCaptureContext(b->c);
    T_SINK(b->c->ContextFlags);
}
static void cx_perf_get(DWORD flags, int kind, const char *metric) {
    cx_args *a = cx_args_new(0xC100, 0);
    REQUIRE(a);
    cx_xc x = { 0 };
    CONTEXT *plain = NULL;
    if (flags & 0x40 & ~CX_ARCHBIT) { CX_XC_REQUIRE(&x, flags); X.SetXStateFeaturesMask(x.ctx, T_XS_AVX); }
    else { plain = cx_plain(flags); REQUIRE(plain); }
    cx_worker w;
    int ok = cx_worker_start(&w, a, kind);
    if (ok && cx_suspend(&w)) {
        cx_bench_ctx b = { w.h, plain ? plain : x.ctx };
        t_stats s;
        int n = t_bench(cx_bench_get, &b, 3, 15, 5.0, &s);
        CHECK(n > 0);
        /* verify the measured operation still returns correct state */
        CHECK(GetThreadContext(w.h, b.c));
        cx_check_xmm(b.c, a->vin, CX_NV, "after benchmark");
        if (n > 0 && s.median > 0) {
            t_perf(metric, 16.0 / s.median, "ops/s");
            t_info("us_per_call", "%.3f", s.median * 1e6 / 16.0);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_xc_free(&x);
    cx_free(plain);
    cx_free(a);
}
TEST(perf_getctx_std) { cx_perf_get(CX_STD, T_VK_XMM, "getthreadcontext_std"); }
TEST(perf_getctx_xstate) { CX_NEED_AVX(); cx_perf_get(CX_STD | CONTEXT_XSTATE, T_VK_YMM, "getthreadcontext_xstate"); }
TEST(perf_rtlcapturecontext) {
    CX_NEED_CAPTURE();
    CONTEXT *c = cx_plain(0);
    REQUIRE(c);
    cx_bench_ctx b = { NULL, c };
    t_stats s;
    int n = t_bench(cx_bench_capture, &b, 3, 15, 5.0, &s);
    CHECK(n > 0);
    CHECK((c->ContextFlags & (CONTEXT_CONTROL | CONTEXT_INTEGER)) == (CONTEXT_CONTROL | CONTEXT_INTEGER));
    if (n > 0 && s.median > 0) t_perf("rtlcapturecontext", 64.0 / s.median, "ops/s");
    cx_free(c);
}
