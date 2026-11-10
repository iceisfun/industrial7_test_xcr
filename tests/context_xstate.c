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
 *     build (build/x86/context_xstate.exe) when it exists.
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
#define CX_ARCHNAME  "x64"
#else
#define CX_ARCHBIT   CONTEXT_i386
#define CX_STD       (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS)
#define CX_XMMF      CONTEXT_EXTENDED_REGISTERS
#define CX_NGPR      5
#define CX_CTX_ALIGN 4
#define CX_SSE_LEN   0x80
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
#define CXO_SP     1480
#define CXO_ZIN    1536
#define CXO_ZOUT   3584
#define CXO_KIN    5632
#define CXO_KOUT   5696
#define CX_S_(x) #x
#define CX_S(x) CX_S_(x)

enum { CXF_X87 = 1, CXF_KBW = 2 };

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
    uint32_t flags, max_loops, remaining, pad1;
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
    "  testl $1, " CXA(CXO_FLAGS) "(" b ")\n  jz 1f\n" \
    "  fldt " CXA(CXO_FIN) "+70(" b ")\n  fldt " CXA(CXO_FIN) "+60(" b ")\n  fldt " CXA(CXO_FIN) "+50(" b ")\n" \
    "  fldt " CXA(CXO_FIN) "+40(" b ")\n  fldt " CXA(CXO_FIN) "+30(" b ")\n  fldt " CXA(CXO_FIN) "+20(" b ")\n" \
    "  fldt " CXA(CXO_FIN) "+10(" b ")\n  fldt " CXA(CXO_FIN) "+0(" b ")\n1:\n"
#define CX_X87_STORE(b) \
    "  stmxcsr " CXA(CXO_MXOUT) "(" b ")\n  fnstcw " CXA(CXO_FCWOUT) "(" b ")\n  fnstsw " CXA(CXO_FSWOUT) "(" b ")\n" \
    "  testl $1, " CXA(CXO_FLAGS) "(" b ")\n  jz 2f\n" \
    "  fstpt " CXA(CXO_FOUT) "+0(" b ")\n  fstpt " CXA(CXO_FOUT) "+10(" b ")\n  fstpt " CXA(CXO_FOUT) "+20(" b ")\n" \
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
    "  subq $" CXA(CXO_READY) ", %rax\n  movl %r11d, " CXA(CXO_REM) "(%rax)\n" \
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
    "  subl $" CXA(CXO_READY) ", %eax\n  movl %ecx, " CXA(CXO_REM) "(%eax)\n" \
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
static cx_args *volatile g_live[CX_MAX_LIVE];

static DWORD WINAPI cx_worker_main(void *p) {
    cx_worker *w = (cx_worker *)p;
    if (w->kind == T_VK_ZMM) cx_spin_zmm(w->a);
    else if (w->kind == T_VK_YMM) cx_spin_ymm(w->a);
    else cx_spin_xmm(w->a);
    return 0;
}
static const char *cx_loop_lo(int kind) { return kind == T_VK_ZMM ? cx_zmm_loop : kind == T_VK_YMM ? cx_ymm_loop : cx_xmm_loop; }
static const char *cx_loop_hi(int kind) { return kind == T_VK_ZMM ? cx_zmm_exit : kind == T_VK_YMM ? cx_ymm_exit : cx_xmm_exit; }

/* Start a worker on a pre-filled cx_args and wait until it holds its registers (spinning).
 * Returns 1 on success. Stale workers of an aborted case are stopped first. */
static int cx_worker_start(cx_worker *w, cx_args *a, int kind) {
    for (int i = 0; i < CX_MAX_LIVE; i++) if (g_live[i]) { g_live[i]->stop = 1; }
    memset(w, 0, sizeof *w);
    w->a = a; w->kind = kind;
    a->ready = 0; a->stop = 0;
    w->h = CreateThread(NULL, 0, cx_worker_main, w, 0, &w->tid);
    if (!w->h) return 0;
    for (int i = 0; i < CX_MAX_LIVE; i++) if (!g_live[i]) { g_live[i] = a; break; }
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
            CHECKF(p == NULL, "feature %lu not enabled (0x%llx) but located at %p", (unsigned long)id, (unsigned long long)en, (void *)p);
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
    CHECK_EQ(GetLastError(), ERROR_INVALID_PARAMETER);
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
    CHECK_EQ(GetLastError(), ERROR_NOT_SUPPORTED);
    c->ContextFlags = 0x1F;
    SetLastError(0);
    CHECK(!X.SetXStateFeaturesMask(c, 3));
    CHECK_EQ(GetLastError(), ERROR_NOT_SUPPORTED);
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
    CHECK(!X.CopyContext(a.ctx, CONTEXT_AMD64 | 0x20, b.ctx));
#else
    CHECK(!X.CopyContext(a.ctx, CONTEXT_i386 | 0x80, b.ctx));
#endif
    CHECK_EQ(GetLastError(), ERROR_INVALID_PARAMETER);
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
    CHECK(X.CopyContext(d.ctx, CX_XMMF, s.ctx));
    for (int i = 0; i < CX_NV; i++) CHECK_MEM(cx_xmm(d.ctx, i), cx_xmm(s.ctx, i), 16);
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
            CHECK_EQ(c->Ebp, (DWORD)a->gin[4]);   /* EBP is part of CONTEXT_CONTROL on x86 */
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
        CHECK_EQ(a->fcw_out, 0x0A7F);
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
    uint32_t mask = 0xFFBF;
    {
        T_ALIGN(16) uint8_t fx[512];
        memset(fx, 0, sizeof fx);
        t_fxsave(fx);
        uint32_t m; memcpy(&m, fx + 28, 4);
        if (m) mask = m;
    }
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
        CHECK_EQ(a->fcw_out, 0x0C7F);
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
            CHECKF(a->gout[i] == want, "%s observed 0x%llx want 0x%llx", cx_gpr_name(i), (unsigned long long)a->gout[i], (unsigned long long)want);
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
    cx_worker w;
    int ok = cx_worker_start(&w, a, T_VK_XMM);
    if (ok && cx_suspend(&w)) {
        cx_xc x = { 0 };
        if (cx_get_x(&w, &x, CX_STD | CONTEXT_XSTATE, T_XS_AVX)) {
            uint64_t m = cx_mask(x.ctx);
            t_info("mask_init_ymm", "0x%llx", (unsigned long long)m);
            uint8_t *ymmh = cx_locate(x.ctx, CX_XS_AVX, NULL);
            if (m & T_XS_AVX) CHECKF(ymmh && cx_all_zero(ymmh, 16 * CX_NV), "AVX present in mask but upper halves are not zero");
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
                CHECK(SetThreadContext(w1.h, x1.ctx));
            }
        }
        cx_xc_free(&x1); cx_xc_free(&x2);
    } else CHECKF(ok1 && ok2, "workers did not start");
    cx_release(&w1);
    if (ok1) cx_release(&w2);
    if (w1.done) { uint8_t want[32]; memcpy(want, a1->vin[2], 32); memcpy(want + 16, nv, 16); CHECK_REG("worker1 ymm2", a1->vout[2], want, 32); }
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
            t_info("dst_flags", "0x%lx src 0x%lx", (unsigned long)d.ctx->ContextFlags, (unsigned long)s.ctx->ContextFlags);
            t_info("dst_seg", "cs %x ss %x efl %lx dr7 %llx", d.ctx->SegCs, d.ctx->SegSs, (unsigned long)d.ctx->EFlags, (unsigned long long)d.ctx->Dr7);
            CHECK_EQ(cx_mask(d.ctx), cx_mask(s.ctx));
            uint8_t *hd = cx_locate(d.ctx, CX_XS_AVX, NULL), *hs = cx_locate(s.ctx, CX_XS_AVX, NULL);
            if (hd && hs) CHECK_MEM(hd, hs, 16 * CX_NV);
            CHECK_MEM(cx_xmm(d.ctx, 0), cx_xmm(s.ctx, 0), 16 * CX_NV);
            CHECK_EQ(cx_ip(d.ctx), cx_ip(s.ctx));
            CHECK_EQ(cx_sp(d.ctx), cx_sp(s.ctx));
            for (int i = 0; i < CX_NGPR; i++) CHECK_EQ(cx_gpr(d.ctx, i), cx_gpr(s.ctx, i));
            /* the copy is a valid SetThreadContext source */
            if (hd) { uint8_t nv[16]; t_pattern_reg(nv, 16, 0x7A01, 4); memcpy(hd + 16 * 4, nv, 16);
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
        CHECKF(cx_all_sent(ymmh, 256), "GetThreadContext wrote the AVX area beyond CONTEXT_EX.XState.Length");
        if (!r) CHECKF(e == ERROR_MORE_DATA, "short XSTATE chunk: GLE %lu (Windows 10: ERROR_MORE_DATA)", (unsigned long)e);
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
static void cx_cap_run(cx_cap *k) {
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
    CHECKF(c->SegCs == cx_own_cs(), "SegCs 0x%x want 0x%x", (unsigned)c->SegCs, cx_own_cs());
    CHECKF(c->SegSs == cx_own_ss(), "SegSs 0x%x want 0x%x", (unsigned)c->SegSs, cx_own_ss());
    CHECKF(c->EFlags & 0x2, "EFlags 0x%lx", (unsigned long)c->EFlags);
    CHECKF(c->EFlags & 0x200, "EFlags 0x%lx IF clear", (unsigned long)c->EFlags);
#ifdef T_X64
    CHECKF(c->Rip == (DWORD64)(uintptr_t)cx_cap_ret, "Rip %p want return address %p", (void *)(uintptr_t)c->Rip, (void *)cx_cap_ret);
    CHECKF(c->Rsp == k.sp_after, "Rsp 0x%llx want 0x%llx (RSP after the call)", (unsigned long long)c->Rsp, (unsigned long long)k.sp_after);
    static const char *nm[] = { "rbx", "rbp", "rsi", "rdi", "r12", "r13", "r14", "r15" };
    DWORD64 got[8] = { c->Rbx, c->Rbp, c->Rsi, c->Rdi, c->R12, c->R13, c->R14, c->R15 };
    for (int i = 0; i < 8; i++) CHECKF(got[i] == k.gin[i], "%s 0x%llx want 0x%llx", nm[i], (unsigned long long)got[i], (unsigned long long)k.gin[i]);
    if (c->Rcx != (DWORD64)(uintptr_t)c) t_note("Rcx 0x%llx is not the CONTEXT argument", (unsigned long long)c->Rcx);
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
    CHECKF(c->SegDs == cx_own_ds() && c->SegEs == cx_own_es(), "SegDs/SegEs 0x%lx/0x%lx", (unsigned long)c->SegDs, (unsigned long)c->SegEs);
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
    for (int i = 0; i < CX_NGPR; i++) if (i != 1) CHECK_EQ(cx_gpr(c, i), cx_gpr(d, i));
    t_guarded_free(base);
    cx_free(d);
}

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
            cx_free(c);
        }
    } else CHECKF(ok, "worker did not start");
    cx_release(&w);
    cx_free(a);
#else
    SKIP("x86 build only (the x64 build drives WOW64 through wow64_cross_process_context)");
#endif
}

/* Child (x86 build): worker with a known state; parent drives it through the Wow64* API. */
CHILD(wow64_target) {
    cx_args *a = cx_args_new(CX_W64_TAG, 1);
    if (!a) return 3;
    a->fcw_in = 0x0A7F; a->mxcsr_in = 0x3F80;
    cx_worker w;
    if (!cx_worker_start(&w, a, T_VK_XMM)) { printf("W64_ERR=start\n"); return 3; }
    printf("W64_INFO=%llx %llx %llx %lu %llx %d\n", (unsigned long long)(uintptr_t)a, (unsigned long long)(uintptr_t)cx_xmm_loop,
           (unsigned long long)(uintptr_t)cx_xmm_exit, (unsigned long)w.tid, (unsigned long long)a->sp, (int)sizeof(void *));
    fflush(stdout);
    DWORD wr = WaitForSingleObject(w.h, 20000);
    if (wr != WAIT_OBJECT_0) printf("W64_TIMEOUT=1\n");
    cx_worker_finish(&w);
    printf("W64_GOUT=");
    for (int i = 0; i < CX_NGPR; i++) printf("%llx,", (unsigned long long)a->gout[i]);
    printf("\n");
    char hx[64];
    for (int i = 0; i < 16; i++) sprintf(hx + 2 * i, "%02x", a->vout[2][i]);
    printf("W64_X2=%s\n", hx);
    printf("W64_MX=%x %x %u\n", a->mxcsr_out, a->fcw_out, a->remaining);
    fflush(stdout);
    return 0;
}

#ifdef T_X64
static int cx_hexbytes(const char *s, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) { unsigned v; if (sscanf(s + 2 * i, "%2x", &v) != 1) return 0; out[i] = (uint8_t)v; }
    return 1;
}
static const char *cx_find_line(const char *buf, const char *prefix) {
    const char *p = strstr(buf, prefix);
    return p ? p + strlen(prefix) : NULL;
}
#endif

TEST(wow64_cross_process_context) {
#ifdef T_X64
    cx_resolve();
    if (!X.Wow64GetThreadContext || !X.Wow64SetThreadContext || !X.Wow64SuspendThread) SKIP("Wow64*ThreadContext not exported");
    char path[MAX_PATH * 2];
    snprintf(path, sizeof path, "%s", t_self_path());
    char *p = strstr(path, "\\x64\\");
    if (!p) p = strstr(path, "/x64/");
    if (!p) SKIP("cannot derive the x86 sibling from %s", path);
    p[1] = 'x'; p[2] = '8'; p[3] = '6';
    if (GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES) SKIP("x86 build not found at %s", path);
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    REQUIRE(CreatePipe(&rd, &wr, &sa, 1 << 16));
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si; memset(&si, 0, sizeof si);
    si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr; si.hStdError = wr; si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi; memset(&pi, 0, sizeof pi);
    char cmd[MAX_PATH * 2 + 64];
    snprintf(cmd, sizeof cmd, "\"%s\" --child=wow64_target", path);
    BOOL cp = CreateProcessA(path, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(wr);
    if (!cp) { DWORD e = GetLastError(); CloseHandle(rd); SKIP("cannot start the x86 sibling (GLE=%lu): WOW64 unavailable", (unsigned long)e); }
    static char out[16384];
    size_t n = 0;
    out[0] = 0;
    const char *info = NULL;
    double t0 = t_now_ms();
    int exited = 0;
    /* read until the INFO line (or 15 s) */
    while (!info && t_now_ms() - t0 < 15000) {
        DWORD avail = 0;
        if (PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) && avail) {
            DWORD got = 0;
            if (n + avail >= sizeof out) avail = (DWORD)(sizeof out - n - 1);
            if (avail && ReadFile(rd, out + n, avail, &got, NULL)) { n += got; out[n] = 0; }
            const char *q = cx_find_line(out, "W64_INFO=");
            if (q && strchr(q, '\n')) info = q;
        } else {
            if (WaitForSingleObject(pi.hProcess, 0) == WAIT_OBJECT_0) { exited = 1; break; }
            Sleep(5);
        }
    }
    unsigned long long a_args = 0, a_loop = 0, a_exit = 0, a_sp = 0; unsigned long tid = 0; int psz = 0;
    if (info) sscanf(info, "%llx %llx %llx %lu %llx %d", &a_args, &a_loop, &a_exit, &tid, &a_sp, &psz);
    if (!info || psz != 4) {
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(rd);
        REQUIREF(0, "x86 sibling did not report (exited=%d psz=%d) output: %.300s", exited, psz, out);
    }
    static cx_args exp;
    cx_args_fill(&exp, CX_W64_TAG, 1);
    uint8_t nx2[16];
    t_pattern_reg(nx2, 16, 0xB1FF, 2);
    const DWORD nebx = 0x13572468;
    HANDLE th = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, FALSE, tid);
    CHECKF(th != NULL, "OpenThread(%lu) GLE=%lu", tid, (unsigned long)GetLastError());
    int set_done = 0;
    if (th) {
        DWORD sc = X.Wow64SuspendThread(th);
        CHECK_EQ(sc, 0);
        static WOW64_CONTEXT wc;
        memset(&wc, CX_SENT, sizeof wc);
        wc.ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_FLOATING_POINT | WOW64_CONTEXT_EXTENDED_REGISTERS;
        if (X.Wow64GetThreadContext(th, &wc)) {
            DWORD g[5] = { wc.Ebx, wc.Edx, wc.Esi, wc.Edi, wc.Ebp };
            for (int i = 0; i < 5; i++)
                CHECKF(g[i] == (DWORD)exp.gin[i], "WOW64 %s 0x%lx want 0x%lx", i == 0 ? "ebx" : i == 1 ? "edx" : i == 2 ? "esi" : i == 3 ? "edi" : "ebp",
                       (unsigned long)g[i], (unsigned long)(DWORD)exp.gin[i]);
            CHECKF(wc.Eip >= a_loop && wc.Eip <= a_exit, "WOW64 Eip 0x%lx outside [0x%llx, 0x%llx]", (unsigned long)wc.Eip, a_loop, a_exit);
            CHECKF(wc.Esp == a_sp, "WOW64 Esp 0x%lx want 0x%llx", (unsigned long)wc.Esp, a_sp);
            CHECKF(wc.Eax == (DWORD)(a_args + CXO_READY), "WOW64 Eax 0x%lx want &ready", (unsigned long)wc.Eax);
            CHECKF((wc.SegCs & 3) == 3 && wc.SegCs != cx_own_cs(), "WOW64 SegCs 0x%lx (64-bit CS is 0x%x)", (unsigned long)wc.SegCs, cx_own_cs());
            for (int i = 0; i < 8; i++) {
                char lab[40]; snprintf(lab, sizeof lab, "WOW64 xmm%d", i);
                CHECK_REG(lab, wc.ExtendedRegisters + 160 + 16 * i, exp.vin[i], 16);
                snprintf(lab, sizeof lab, "WOW64 ST(%d)", i);
                CHECK_REG(lab, wc.FloatSave.RegisterArea + 10 * i, exp.fin[i].b, 10);
            }
            CHECK_EQ(wc.FloatSave.ControlWord & 0xFFFF, 0x0A7F);
            uint32_t mx; memcpy(&mx, wc.ExtendedRegisters + 24, 4);
            CHECK_EQ(mx, 0x3F80);
            /* native view of the same thread: an x64 context somewhere in the WOW64 layer (observation) */
            CONTEXT *nc = cx_plain(CONTEXT_CONTROL);
            if (nc && GetThreadContext(th, nc)) t_info("native_view", "Rip=%p SegCs=0x%x", (void *)(uintptr_t)nc->Rip, (unsigned)nc->SegCs);
            cx_free(nc);
            /* modify Ebx and XMM2, read back, let the child report */
            wc.Ebx = nebx;
            memcpy(wc.ExtendedRegisters + 160 + 32, nx2, 16);
            wc.ContextFlags = WOW64_CONTEXT_INTEGER | WOW64_CONTEXT_EXTENDED_REGISTERS;
            set_done = X.Wow64SetThreadContext(th, &wc);
            CHECKF(set_done, "Wow64SetThreadContext GLE=%lu", (unsigned long)GetLastError());
            memset(&wc, CX_SENT, sizeof wc);
            wc.ContextFlags = WOW64_CONTEXT_FULL | WOW64_CONTEXT_EXTENDED_REGISTERS;
            if (X.Wow64GetThreadContext(th, &wc)) {
                CHECK_EQ(wc.Ebx, nebx);
                CHECK_MEM(wc.ExtendedRegisters + 160 + 32, nx2, 16);
                CHECK_EQ(wc.Esi, (DWORD)exp.gin[2]);
            }
        } else CHECKF(0, "Wow64GetThreadContext GLE=%lu", (unsigned long)GetLastError());
        LONG one = 1;
        SIZE_T wn = 0;
        CHECKF(WriteProcessMemory(pi.hProcess, (void *)(uintptr_t)(a_args + CXO_STOP), &one, sizeof one, &wn), "WriteProcessMemory GLE=%lu",
               (unsigned long)GetLastError());
        ResumeThread(th);
        CloseHandle(th);
    }
    /* drain until exit */
    t0 = t_now_ms();
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) && avail) {
            DWORD got = 0;
            if (n + avail >= sizeof out) avail = (DWORD)(sizeof out - n - 1);
            if (!avail) break;
            if (ReadFile(rd, out + n, avail, &got, NULL)) { n += got; out[n] = 0; }
        } else if (WaitForSingleObject(pi.hProcess, 0) == WAIT_OBJECT_0) {
            if (!PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) || !avail) break;
        } else if (t_now_ms() - t0 > 25000) { TerminateProcess(pi.hProcess, 1); CHECKF(0, "x86 sibling did not exit"); break; }
        else Sleep(5);
    }
    const char *go = cx_find_line(out, "W64_GOUT="), *x2 = cx_find_line(out, "W64_X2=");
    CHECKF(!cx_find_line(out, "W64_TIMEOUT="), "x86 worker never observed the stop flag");
    CHECKF(go && x2, "x86 sibling did not report its registers: %.300s", out);
    if (go && x2 && set_done) {
        unsigned long long g[5] = { 0 };
        sscanf(go, "%llx,%llx,%llx,%llx,%llx", &g[0], &g[1], &g[2], &g[3], &g[4]);
        CHECK_EQ(g[0], nebx);
        for (int i = 1; i < 5; i++) CHECK_EQ(g[i], (DWORD)exp.gin[i]);
        uint8_t v2[16];
        if (cx_hexbytes(x2, v2, 16)) CHECK_REG("WOW64 xmm2 observed", v2, nx2, 16);
        else CHECK(0);
    }
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(rd);
#else
    SKIP("x64 build only (an x86 process cannot drive a WOW64 target through Wow64*ThreadContext)");
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
