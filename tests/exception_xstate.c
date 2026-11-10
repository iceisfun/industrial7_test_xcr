/* exception_xstate - register preservation and CONTEXT fidelity through exceptions.
 *
 * Every fault scenario runs inside one assembly "kind" routine that loads known patterns into every
 * vector register (XMM / YMM / ZMM + k), the GPRs, MXCSR, optionally the x87 stack/CW and EFLAGS, executes
 * exactly one faulting instruction at a global label, and on resume stores everything back. The handler
 * (vectored, frame-based SEH, or both) compares EXCEPTION_RECORD and CONTEXT with that instruction-level
 * reference, optionally edits the CONTEXT, and resumes; the routine's stores then show what the OS
 * really restored. Faults that could kill the process (#UD, privileged instructions, fast-fail,
 * non-continuable dispatch) run in child processes.
 *
 * Windows behaviour pinned against the Windows 10 19041 reference (corpus explorer decompilation):
 *  - ntoskrnl!KiDispatchException: user-mode exception CONTEXT flags = 0x10005F (CONTROL|INTEGER|SEGMENTS|
 *    FLOATING_POINT|DEBUG|XSTATE) when the kernel uses XSAVE (KeFeatureBits), so handlers see YMM/ZMM state.
 *    For STATUS_BREAKPOINT the context RIP is decremented by one (points at the INT3).
 *  - ntdll!KiUserExceptionDispatcher: CLD; RtlDispatchException(rec, ctx); if handled -> RtlRestoreContext(ctx,
 *    NULL), else NtRaiseException(rec, ctx, FALSE). So CONTEXT edits (incl. XSTATE) are what resumes.
 *  - ntdll!RtlDispatchException: vectored handlers first; frame handlers via RtlVirtualUnwind; a frame handler
 *    returning ExceptionContinueExecution for a non-continuable exception -> RtlRaiseStatus(0xC0000025);
 *    vectored continue handlers (RtlpCallVectoredHandlers(...,1)) run before returning.
 *  - ntoskrnl!KiDivideErrorFault -> 0xC0000094, 0 parameters (KiOpDecode may turn INT_MIN/-1 into 0xC0000095).
 *  - ntoskrnl!KiBreakpointTrap -> 0x80000003, 1 parameter (0), address = RIP-1.
 *  - ntoskrnl!KiPreprocessFault: #GP that decodes to nothing else -> STATUS_ACCESS_VIOLATION with
 *    NumberParameters=2, Information = {0, 0xFFFFFFFFFFFFFFFF} (non-canonical address etc.).
 *  - ntoskrnl!KiXmmException (64-bit code): first unmasked of IE,ZE,DE,OE,UE,PE ->
 *    0xC0000090, 0xC000008E, 0xC0000090(!), 0xC0000091, 0xC0000093, 0xC000008F, 2 parameters;
 *    32-bit code (CS=0x23): 0xC00002B5 for IE/ZE/DE, 0xC00002B4 for OE/UE/PE.
 *  - ntoskrnl!KiFloatingErrorFault: x87 #MF -> IE+SF 0xC0000092, IE 0xC0000090, ZE 0xC000008E, DE 0xC0000090,
 *    OE 0xC0000091, UE 0xC0000093, PE 0xC000008F; 1 parameter; address = the waiting instruction.
 *  - kernelbase!RaiseException: flags &= EXCEPTION_NONCONTINUABLE; NumberParameters = min(n, 15), 0 when
 *    lpArguments == NULL.
 *  - ntdll!RtlCaptureContext (x64): ContextFlags 0x10000F, Rip = return address, Rsp = caller RSP after return,
 *    FXSAVE of all XMM + MXCSR. RtlCaptureContext2 adds XSTATE when the CONTEXT_EX asks for it.
 *  - ntdll!RtlRestoreContext (internal body): a CONTEXT whose flags minus XSTATE are exactly 0x10000F is restored in
 *    user mode (XRSTOR/FXRSTOR/iretq); anything else (e.g. exception contexts carrying CONTEXT_DEBUG_REGISTERS)
 *    goes through NtContinue, so Dr7 edits in a handler's CONTEXT take effect (veh_hw_breakpoint_dr0_rf).
 *  - ntdll!RtlCaptureContext2: ContextFlags &= 0x100040; XSTATE captured when CONTEXT_XSTATE was requested;
 *    then ContextFlags |= 0x10000F.
 *  - ntdll!RtlRestoreContext: STATUS_LONGJUMP loads Rbx,Rsp,Rbp,Rsi,Rdi,R12-R15,Rip,MxCsr,FpCsr,Xmm6-15 from
 *    the _JUMP_BUFFER; plain restore: XRSTOR (when CONTEXT_XSTATE) + FXRSTOR + XMM0-15 + LDMXCSR.
 */
#include "t.h"

/* ---- status codes (own names; independent of which ones the SDK headers define) ---------------- */
#define XS_ACCESS_VIOLATION      0xC0000005u
#define XS_GUARD_PAGE            0x80000001u
#define XS_MISALIGNMENT          0x80000002u
#define XS_BREAKPOINT            0x80000003u
#define XS_SINGLE_STEP           0x80000004u
#define XS_ILLEGAL_INSTRUCTION   0xC000001Du
#define XS_NONCONTINUABLE        0xC0000025u
#define XS_ARRAY_BOUNDS          0xC000008Cu
#define XS_FLT_DENORMAL          0xC000008Du
#define XS_FLT_DIVIDE_BY_ZERO    0xC000008Eu
#define XS_FLT_INEXACT           0xC000008Fu
#define XS_FLT_INVALID           0xC0000090u
#define XS_FLT_OVERFLOW          0xC0000091u
#define XS_FLT_STACK_CHECK       0xC0000092u
#define XS_FLT_UNDERFLOW         0xC0000093u
#define XS_INT_DIVIDE_BY_ZERO    0xC0000094u
#define XS_INT_OVERFLOW          0xC0000095u
#define XS_PRIVILEGED            0xC0000096u
#define XS_FLT_MULTIPLE_FAULTS   0xC00002B4u
#define XS_FLT_MULTIPLE_TRAPS    0xC00002B5u
#define XS_STACK_BUFFER_OVERRUN  0xC0000409u
#define XS_ASSERTION_FAILURE     0xC0000420u

#define XSF_AVX     2
#define XSF_OPMASK  5
#define XSF_ZMM_H   6
#define XSF_HI_ZMM  7
#define XSM_AVX     (1ull << XSF_AVX)

#define X_EXC_NONCONTINUABLE 0x01
#define X_EXC_UNWINDING      0x02
#define X_EXC_EXIT_UNWIND    0x04
#define X_EXC_NESTED_CALL    0x10
#define X_EXC_TARGET_UNWIND  0x20

#ifdef T_X64
#define X_CONTEXT_XSTATE 0x100040u
#define X_CONTEXT_FULLFP 0x10000Bu     /* CONTROL | INTEGER | FLOATING_POINT */
#define PTRSZ 8
#define NGPR 16
#else
#define X_CONTEXT_XSTATE 0x10040u
#define X_CONTEXT_EXT    0x10020u
#define X_CONTEXT_FP     0x10008u
#define X_CONTEXT_FULLFP 0x10003u      /* CONTROL | INTEGER */
#define PTRSZ 4
#define NGPR 8
#endif

/* ========================================================================================== */
/* Reference frame shared with the assembly routines. Offsets are hard-coded in the asm below.  */
/* ========================================================================================== */
enum { XF_YMM = 1, XF_ZMM = 2, XF_X87 = 4, XF_EFL = 8, XF_KBW = 16 };

typedef struct xf {
    T_ALIGN(64) uint8_t vin[32][64];   /*    0  vector images loaded (v[16..31] = "B" images for x64 helpers) */
    uint8_t  vout[32][64];             /* 2048  vector images stored after resume */
    uint64_t kin[8];                   /* 4096 */
    uint64_t kout[8];                  /* 4160 */
    uint64_t gin[16];                  /* 4224  GPRs by hardware number (rax rcx rdx rbx rsp rbp rsi rdi r8..) */
    uint64_t gout[16];                 /* 4352 */
    uint32_t mxcsr_in;                 /* 4480 */
    uint32_t mxcsr_out;                /* 4484 */
    uint32_t mxcsr_restore;            /* 4488  MXCSR re-loaded before returning to C */
    uint32_t flags;                    /* 4492  XF_* */
    uint64_t rsp_at;                   /* 4496  stack pointer at the faulting instruction */
    uint64_t efl_in;                   /* 4504 */
    uint64_t efl_out;                  /* 4512 */
    uint16_t fcw_in, fcw_out, fsw_out, fcw_restore;   /* 4520 4522 4524 4526 */
    uint8_t  x87in[8][16];             /* 4528  ST(i) 80-bit images, stride 16 */
    uint8_t  x87out[8][16];            /* 4656 */
    uint64_t pass;                     /* 4784 */
    uint64_t cb;                       /* 4792  callback (capture probe) */
    uint64_t ctxp;                     /* 4800 */
    uint64_t capfn;                    /* 4808 */
    uint64_t aux;                      /* 4816  setjmp: pass the frame argument */
    uint64_t gin2[16];                 /* 4824  "B" GPR images */
    uint32_t mxcsr2;                   /* 4952 */
    uint16_t fcw2, pad0;               /* 4956 */
    uint8_t  pad1[32];                 /* 4960 */
    T_ALIGN(16) uint8_t jb[256];       /* 4992  jmp_buf */
    /* ---- C-only ---- */
    uint8_t  vexp[32][64];
    uint64_t gexp[16], kexp[8];
    uint8_t  x87exp[8][16];
    uint32_t mxexp, mx_any_out;
    uint16_t fcwexp;
    uint32_t vskip, gskip;             /* registers not compared after resume */
} xf;

_Static_assert(offsetof(xf, vout) == 2048, "vout");
_Static_assert(offsetof(xf, kin) == 4096, "kin");
_Static_assert(offsetof(xf, gin) == 4224, "gin");
_Static_assert(offsetof(xf, gout) == 4352, "gout");
_Static_assert(offsetof(xf, mxcsr_in) == 4480, "mxin");
_Static_assert(offsetof(xf, flags) == 4492, "flags");
_Static_assert(offsetof(xf, rsp_at) == 4496, "rsp_at");
_Static_assert(offsetof(xf, efl_out) == 4512, "efl_out");
_Static_assert(offsetof(xf, fcw_in) == 4520, "fcw");
_Static_assert(offsetof(xf, fcw_restore) == 4526, "fcwr");
_Static_assert(offsetof(xf, x87in) == 4528, "x87in");
_Static_assert(offsetof(xf, x87out) == 4656, "x87out");
_Static_assert(offsetof(xf, pass) == 4784, "pass");
_Static_assert(offsetof(xf, aux) == 4816, "aux");
_Static_assert(offsetof(xf, gin2) == 4824, "gin2");
_Static_assert(offsetof(xf, mxcsr2) == 4952, "mx2");
_Static_assert(offsetof(xf, fcw2) == 4956, "fcw2");
_Static_assert(offsetof(xf, jb) == 4992, "jb");

/* ========================================================================================== */
/* Assembly                                                                                     */
/* ========================================================================================== */
#define R8L  "0,1,2,3,4,5,6,7"
#define R16L R8L ",8,9,10,11,12,13,14,15"
#define R32L R16L ",16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"

#ifdef T_X64
#define GL(n) ".globl " #n "\n" #n ":\n"
#define XPRO(name, extra) \
    ".globl " #name "\n.def " #name "; .scl 2; .type 32; .endef\n.seh_proc " #name "\n" #name ":\n" extra \
    "  pushq %rbp\n  .seh_pushreg %rbp\n  pushq %rbx\n  .seh_pushreg %rbx\n" \
    "  pushq %rsi\n  .seh_pushreg %rsi\n  pushq %rdi\n  .seh_pushreg %rdi\n" \
    "  pushq %r12\n  .seh_pushreg %r12\n  pushq %r13\n  .seh_pushreg %r13\n" \
    "  pushq %r14\n  .seh_pushreg %r14\n  pushq %r15\n  .seh_pushreg %r15\n" \
    "  subq $200, %rsp\n  .seh_stackalloc 200\n" \
    "  movdqa %xmm6, 32(%rsp)\n  .seh_savexmm %xmm6, 32\n  movdqa %xmm7, 48(%rsp)\n  .seh_savexmm %xmm7, 48\n" \
    "  movdqa %xmm8, 64(%rsp)\n  .seh_savexmm %xmm8, 64\n  movdqa %xmm9, 80(%rsp)\n  .seh_savexmm %xmm9, 80\n" \
    "  movdqa %xmm10, 96(%rsp)\n  .seh_savexmm %xmm10, 96\n  movdqa %xmm11, 112(%rsp)\n  .seh_savexmm %xmm11, 112\n" \
    "  movdqa %xmm12, 128(%rsp)\n  .seh_savexmm %xmm12, 128\n  movdqa %xmm13, 144(%rsp)\n  .seh_savexmm %xmm13, 144\n" \
    "  movdqa %xmm14, 160(%rsp)\n  .seh_savexmm %xmm14, 160\n  movdqa %xmm15, 176(%rsp)\n  .seh_savexmm %xmm15, 176\n" \
    "  .seh_endprologue\n"
#define XEPI \
    "  movdqa 32(%rsp), %xmm6\n  movdqa 48(%rsp), %xmm7\n  movdqa 64(%rsp), %xmm8\n  movdqa 80(%rsp), %xmm9\n" \
    "  movdqa 96(%rsp), %xmm10\n  movdqa 112(%rsp), %xmm11\n  movdqa 128(%rsp), %xmm12\n  movdqa 144(%rsp), %xmm13\n" \
    "  movdqa 160(%rsp), %xmm14\n  movdqa 176(%rsp), %xmm15\n" \
    "  addq $200, %rsp\n  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n" \
    "  popq %rdi\n  popq %rsi\n  popq %rbx\n  popq %rbp\n  ret\n  .seh_endproc\n"
#define GLOAD \
    "  movq 4224(%rbx), %rax\n  movq 4232(%rbx), %rcx\n  movq 4240(%rbx), %rdx\n" \
    "  movq 4264(%rbx), %rbp\n  movq 4272(%rbx), %rsi\n  movq 4280(%rbx), %rdi\n" \
    "  .irp i,8,9,10,11,12,13,14,15\n  movq 4224+\\i*8(%rbx), %r\\i\n  .endr\n"
#define GSTORE \
    "  movq %rax, 4352(%rbx)\n  movq %rcx, 4360(%rbx)\n  movq %rdx, 4368(%rbx)\n" \
    "  movq %rbp, 4392(%rbx)\n  movq %rsi, 4400(%rbx)\n  movq %rdi, 4408(%rbx)\n" \
    "  .irp i,8,9,10,11,12,13,14,15\n  movq %r\\i, 4352+\\i*8(%rbx)\n  .endr\n"
#define KHEAD \
    "  movq %rcx, %rbx\n  call xf_vex_load\n  movq %rsp, 4496(%rbx)\n" \
    "  testl $8, 4492(%rbx)\n  jz 77f\n  pushq 4504(%rbx)\n  popfq\n77:\n" GLOAD
#define KTAIL GSTORE "  pushfq\n  popq %rax\n  movq %rax, 4512(%rbx)\n  cld\n  call xf_vex_store\n" XEPI
#define KIND(n, pre, insn) \
    XPRO(xf_##n, "") KHEAD GL(xf_##n##_op) pre GL(xf_##n##_at) "  " insn "\n" GL(xf_##n##_resume) KTAIL
#define KSEH(n, pre, insn) \
    XPRO(xfs_##n, "  .seh_handler xs_seh_tramp, @except, @unwind\n") KHEAD \
    GL(xfs_##n##_op) pre GL(xfs_##n##_at) "  " insn "\n" GL(xfs_##n##_resume) KTAIL

__asm__(
    ".text\n"
    /* xf_vex_load: rbx = frame. MXCSR, optional x87 stack + CW, vector registers per flags. Uses no GPR. */
    ".globl xf_vex_load\nxf_vex_load:\n"
    "  ldmxcsr 4480(%rbx)\n"
    "  testl $4, 4492(%rbx)\n  jz 1f\n  fninit\n"
    "  .irp i,7,6,5,4,3,2,1,0\n  fldt 4528+\\i*16(%rbx)\n  .endr\n  fldcw 4520(%rbx)\n"
    "1:\n  testl $2, 4492(%rbx)\n  jnz 3f\n  testl $1, 4492(%rbx)\n  jnz 2f\n"
    "  .irp i," R16L "\n  movdqu \\i*64(%rbx), %xmm\\i\n  .endr\n  ret\n"
    "2:\n  .irp i," R16L "\n  vmovdqu \\i*64(%rbx), %ymm\\i\n  .endr\n  ret\n"
    "3:\n  .irp i," R32L "\n  vmovdqu64 \\i*64(%rbx), %zmm\\i\n  .endr\n"
    "  testl $16, 4492(%rbx)\n  jz 4f\n"
    "  .irp i," R8L "\n  kmovq 4096+\\i*8(%rbx), %k\\i\n  .endr\n  ret\n"
    "4:\n  .irp i," R8L "\n  kmovw 4096+\\i*8(%rbx), %k\\i\n  .endr\n  ret\n"

    /* xf_vex_store: rbx = frame. Stores MXCSR, vectors, x87 (CW/SW then stack), then restores MXCSR/CW. */
    ".globl xf_vex_store\nxf_vex_store:\n"
    "  stmxcsr 4484(%rbx)\n"
    "  testl $2, 4492(%rbx)\n  jnz 3f\n  testl $1, 4492(%rbx)\n  jnz 2f\n"
    "  .irp i," R16L "\n  movdqu %xmm\\i, 2048+\\i*64(%rbx)\n  .endr\n  jmp 5f\n"
    "2:\n  .irp i," R16L "\n  vmovdqu %ymm\\i, 2048+\\i*64(%rbx)\n  .endr\n  vzeroupper\n  jmp 5f\n"
    "3:\n  .irp i," R32L "\n  vmovdqu64 %zmm\\i, 2048+\\i*64(%rbx)\n  .endr\n"
    "  testl $16, 4492(%rbx)\n  jz 4f\n  .irp i," R8L "\n  kmovq %k\\i, 4160+\\i*8(%rbx)\n  .endr\n  jmp 6f\n"
    "4:\n  .irp i," R8L "\n  kmovw %k\\i, 4160+\\i*8(%rbx)\n  .endr\n"
    "6:\n  vzeroupper\n"
    "5:\n  testl $4, 4492(%rbx)\n  jz 7f\n"
    "  fnstcw 4522(%rbx)\n  fnstsw 4524(%rbx)\n  fnclex\n  fldcw 4526(%rbx)\n"
    "  .irp i,0,1,2,3,4,5,6,7\n  fstpt 4656+\\i*16(%rbx)\n  .endr\n  fnclex\n"
    "7:\n  ldmxcsr 4488(%rbx)\n  ret\n"

    /* xf_clobber_vex(f): from inside a handler, destroy every register the x64 ABI lets a callee destroy:
     * XMM0-5, YMM upper halves, ZMM upper halves / ZMM16-31, k1-7, MXCSR status flags. */
    ".globl xf_clobber_vex\nxf_clobber_vex:\n"
    "  pcmpeqd %xmm0, %xmm0\n  .irp i,1,2,3,4,5\n  movdqa %xmm0, %xmm\\i\n  .endr\n"
    "  testl $3, 4492(%rcx)\n  jz 1f\n"
    "  vpcmpeqd %ymm0, %ymm0, %ymm0\n  .irp i,1,2,3,4,5\n  vmovdqa %ymm0, %ymm\\i\n  .endr\n"
    "  .irp i,6,7,8,9,10,11,12,13,14,15\n  vinsertf128 $1, %xmm0, %ymm\\i, %ymm\\i\n  .endr\n"
    "  testl $2, 4492(%rcx)\n  jz 1f\n"
    "  vpternlogd $0xff, %zmm0, %zmm0, %zmm0\n"
    "  .irp i,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n  vmovdqa64 %zmm0, %zmm\\i\n  .endr\n"
    "  .irp i,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n  vinserti64x4 $1, %ymm0, %zmm\\i, %zmm\\i\n  .endr\n"
    "  .irp i,1,2,3,4,5,6,7\n  kxnorw %k0, %k0, %k\\i\n  .endr\n"
    "1:\n  stmxcsr 8(%rsp)\n  orl $0x3f, 8(%rsp)\n  ldmxcsr 8(%rsp)\n  ret\n"

    KIND(nop, "", "nop")
    KIND(div0, "", "divl %ecx")
    KIND(div64, "", "divq %rcx")
    KIND(idivov, "", "idivl %ecx")
    KIND(int3, "", "int3")
    KIND(int3cd, "", ".byte 0xcd, 0x03")
    KIND(rd, "", "movq (%rcx), %rax")
    KIND(wr, "", "movq %rax, (%rcx)")
    KIND(exec, "", "call *%rcx")
    KIND(ss, "  pushfq\n  orq $0x100, (%rsp)\n  popfq\n  nop\n", "nop")
    KIND(ud2, "", "ud2")
    KIND(hlt, "", "hlt")
    KIND(cli, "", "cli")
    KIND(sti, "", "sti")
    KIND(inb, "", "inb %dx, %al")
    KIND(outb, "", "outb %al, %dx")
    KIND(wbinvd, "", "wbinvd")
    KIND(rdmsr, "", "rdmsr")
    KIND(wrmsr, "", "wrmsr")
    KIND(movcr, "", "movq %cr0, %rax")
    KIND(clts, "", "clts")
    KIND(invd, "", "invd")
    KIND(xsetbv, "", "xsetbv")
    KIND(int2c, "", "int $0x2c")
    KIND(fastfail, "", "int $0x29")
    KIND(movaps, "", "movaps (%rcx), %xmm0")
    KIND(sse_div, "", "divps %xmm1, %xmm0")
    KIND(sse_sqrt, "", "sqrtps %xmm1, %xmm0")
    KIND(sse_mul, "", "mulps %xmm1, %xmm0")
    KIND(sse_add, "", "addps %xmm1, %xmm0")
    KIND(x87_div, "  fdiv %st(1), %st\n", "fwait")
    KIND(x87_ovf, "  fld1\n", "fwait")
    KIND(x87_mul, "  fmul %st(1), %st\n", "fwait")
    KIND(x87_sqrt, "  fsqrt\n", "fwait")
    /* VEX-encoded faulting instructions (reached only after cpu_has(F_AVX); names carry "avx" for check_isa) */
    KIND(avx_div, "", "vdivps %ymm1, %ymm0, %ymm0")
    KIND(avx_movmis, "", "vmovaps (%rcx), %ymm0")
    KIND(avx_ld, "", "vmovdqu (%rcx), %ymm2")
    /* touch one page below the other until the stack reserve is exhausted (bounded: 16384 pages = 64 MB);
     * the handler resumes at the resume label with RSP reset to rsp_at */
    KIND(stkovf, "  movl $16384, %ecx\n81:\n  subq $4096, %rsp\n",
         "movq %rax, (%rsp)\n  decl %ecx\n  jnz 81b\n  movq 4496(%rbx), %rsp")
    KSEH(div0, "", "divl %ecx")
    KSEH(rd, "", "movq (%rcx), %rax")
    KSEH(int3, "", "int3")
    KSEH(sse_div, "", "divps %xmm1, %xmm0")

    /* Outer frame with a handler calls an inner frame that saves/clobbers every nonvolatile register
     * (with unwind info) and faults. Unwinding to xfo_landing must restore the outer's values. */
    XPRO(xfo_outer, "  .seh_handler xs_seh_tramp, @except, @unwind\n")
    "  movq %rcx, %rbx\n  call xf_vex_load\n  movq %rsp, 4496(%rbx)\n" GLOAD
    "  call xfi_inner\n"
    GL(xfo_after_call)
    "  nop\n"
    GL(xfo_landing)
    KTAIL
    XPRO(xfi_inner, "")
    "  .irp i,6,7,8,9,10,11,12,13,14,15\n  movdqu (16+\\i)*64(%rbx), %xmm\\i\n  .endr\n"
    "  movq %rbx, %rcx\n"
    "  movq 4824+40(%rcx), %rbp\n  movq 4824+48(%rcx), %rsi\n  movq 4824+56(%rcx), %rdi\n"
    "  .irp i,12,13,14,15\n  movq 4824+\\i*8(%rcx), %r\\i\n  .endr\n"
    "  movq 4824+24(%rcx), %rbx\n  xorl %eax, %eax\n"
    GL(xfi_fault)
    "  movq (%rax), %rax\n"
    XEPI

    /* setjmp/longjmp probe: load A, _setjmp(jb, frame or 0); first return: load B into the nonvolatile
     * registers, MXCSR and x87 CW, longjmp(jb, 1); second return: store everything. */
    XPRO(xsj_probe, "")
    "  movq %rcx, %rbx\n  call xf_vex_load\n" GLOAD
    "  leaq 4992(%rbx), %rcx\n  xorl %edx, %edx\n  cmpq $0, 4816(%rbx)\n  je 71f\n  movq %rsp, %rdx\n71:\n"
    "  movq %rsp, 4496(%rbx)\n"
    "  call _setjmp\n"
    GL(xsj_ret)
    "  testl %eax, %eax\n  jnz 72f\n"
    "  .irp i,6,7,8,9,10,11,12,13,14,15\n  movdqu (16+\\i)*64(%rbx), %xmm\\i\n  .endr\n"
    "  ldmxcsr 4952(%rbx)\n  fldcw 4956(%rbx)\n"
    "  leaq 4992(%rbx), %rcx\n  movl $1, %edx\n"
    "  movq 4824+40(%rbx), %rbp\n  movq 4824+48(%rbx), %rsi\n  movq 4824+56(%rbx), %rdi\n"
    "  .irp i,12,13,14,15\n  movq 4824+\\i*8(%rbx), %r\\i\n  .endr\n"
    "  movq 4824+24(%rbx), %rbx\n"
    "  call longjmp\n"
    "  int3\n"
    "72:\n" GSTORE
    "  fnstcw 4522(%rbx)\n  fldcw 4526(%rbx)\n"
    "  call xf_vex_store\n"
    XEPI

    /* capture probe: load all; capfn(ctxp); first pass: optional callback (may restore the context);
     * second pass (after a restore lands on xcap_ret): store everything. */
    XPRO(xcap_probe, "")
    "  movq %rcx, %rbx\n  call xf_vex_load\n" GLOAD
    "  movq 4800(%rbx), %rcx\n  movq %rsp, 4496(%rbx)\n"
    "  call *4808(%rbx)\n"
    GL(xcap_ret)
    "  cmpq $0, 4784(%rbx)\n  jne 73f\n"
    "  movq $1, 4784(%rbx)\n"
    "  movq 4792(%rbx), %rax\n  testq %rax, %rax\n  jz 74f\n  movq %rbx, %rcx\n  call *%rax\n"
    "74:\n  movq $2, 4784(%rbx)\n  call xf_vex_store\n  jmp 75f\n"
    "73:\n" GSTORE "  pushfq\n  popq %rax\n  movq %rax, 4512(%rbx)\n  cld\n  call xf_vex_store\n"
    "75:\n"
    XEPI

    /* void *xseh_call(void *(*fn)(void *), void *arg): calls fn inside a frame whose handler is xs_seh_tramp. */
    ".globl xseh_call\n.def xseh_call; .scl 2; .type 32; .endef\n.seh_proc xseh_call\nxseh_call:\n"
    "  .seh_handler xs_seh_tramp, @except, @unwind\n"
    "  pushq %rbp\n  .seh_pushreg %rbp\n  subq $32, %rsp\n  .seh_stackalloc 32\n  .seh_endprologue\n"
    "  movq %rcx, %rax\n  movq %rdx, %rcx\n  callq *%rax\n"
    GL(xseh_call_ret)
    "  nop\n"
    GL(xseh_call_landing)
    "  addq $32, %rsp\n  popq %rbp\n  ret\n  .seh_endproc\n"
);

#else /* ---------------------------------- x86 ---------------------------------------------------- */
#define GL(n) ".globl _" #n "\n_" #n ":\n"
#define XPRO(name) ".globl _" #name "\n.def _" #name "; .scl 2; .type 32; .endef\n_" #name ":\n" \
    "  pushl %ebp\n  pushl %ebx\n  pushl %esi\n  pushl %edi\n  movl 20(%esp), %ebx\n"
#define XEPI "  popl %edi\n  popl %esi\n  popl %ebx\n  popl %ebp\n  ret\n"
#define GLOAD \
    "  movl 4224(%ebx), %eax\n  movl 4232(%ebx), %ecx\n  movl 4240(%ebx), %edx\n" \
    "  movl 4264(%ebx), %ebp\n  movl 4272(%ebx), %esi\n  movl 4280(%ebx), %edi\n"
#define GSTORE \
    "  movl %eax, 4352(%ebx)\n  movl %ecx, 4360(%ebx)\n  movl %edx, 4368(%ebx)\n" \
    "  movl %ebp, 4392(%ebx)\n  movl %esi, 4400(%ebx)\n  movl %edi, 4408(%ebx)\n"
#define KHEAD \
    "  call _xf_vex_load\n  movl %esp, 4496(%ebx)\n" \
    "  testl $8, 4492(%ebx)\n  jz 77f\n  pushl 4504(%ebx)\n  popfl\n77:\n" GLOAD
#define KTAIL GSTORE "  pushfl\n  popl %eax\n  movl %eax, 4512(%ebx)\n  cld\n  call _xf_vex_store\n" XEPI
#define KIND(n, pre, insn) XPRO(xf_##n) KHEAD GL(xf_##n##_op) pre GL(xf_##n##_at) "  " insn "\n" GL(xf_##n##_resume) KTAIL

__asm__(
    ".text\n"
    ".globl _xf_vex_load\n_xf_vex_load:\n"
    "  ldmxcsr 4480(%ebx)\n"
    "  testl $4, 4492(%ebx)\n  jz 1f\n  fninit\n"
    "  .irp i,7,6,5,4,3,2,1,0\n  fldt 4528+\\i*16(%ebx)\n  .endr\n  fldcw 4520(%ebx)\n"
    "1:\n  testl $2, 4492(%ebx)\n  jnz 3f\n  testl $1, 4492(%ebx)\n  jnz 2f\n"
    "  .irp i," R8L "\n  movdqu \\i*64(%ebx), %xmm\\i\n  .endr\n  ret\n"
    "2:\n  .irp i," R8L "\n  vmovdqu \\i*64(%ebx), %ymm\\i\n  .endr\n  ret\n"
    "3:\n  .irp i," R8L "\n  vmovdqu64 \\i*64(%ebx), %zmm\\i\n  .endr\n"
    "  testl $16, 4492(%ebx)\n  jz 4f\n"
    "  .irp i," R8L "\n  kmovq 4096+\\i*8(%ebx), %k\\i\n  .endr\n  ret\n"
    "4:\n  .irp i," R8L "\n  kmovw 4096+\\i*8(%ebx), %k\\i\n  .endr\n  ret\n"

    ".globl _xf_vex_store\n_xf_vex_store:\n"
    "  stmxcsr 4484(%ebx)\n"
    "  testl $2, 4492(%ebx)\n  jnz 3f\n  testl $1, 4492(%ebx)\n  jnz 2f\n"
    "  .irp i," R8L "\n  movdqu %xmm\\i, 2048+\\i*64(%ebx)\n  .endr\n  jmp 5f\n"
    "2:\n  .irp i," R8L "\n  vmovdqu %ymm\\i, 2048+\\i*64(%ebx)\n  .endr\n  vzeroupper\n  jmp 5f\n"
    "3:\n  .irp i," R8L "\n  vmovdqu64 %zmm\\i, 2048+\\i*64(%ebx)\n  .endr\n"
    "  testl $16, 4492(%ebx)\n  jz 4f\n  .irp i," R8L "\n  kmovq %k\\i, 4160+\\i*8(%ebx)\n  .endr\n  jmp 6f\n"
    "4:\n  .irp i," R8L "\n  kmovw %k\\i, 4160+\\i*8(%ebx)\n  .endr\n"
    "6:\n  vzeroupper\n"
    "5:\n  testl $4, 4492(%ebx)\n  jz 7f\n"
    "  fnstcw 4522(%ebx)\n  fnstsw 4524(%ebx)\n  fnclex\n  fldcw 4526(%ebx)\n"
    "  .irp i,0,1,2,3,4,5,6,7\n  fstpt 4656+\\i*16(%ebx)\n  .endr\n  fnclex\n"
    "7:\n  ldmxcsr 4488(%ebx)\n  ret\n"

    /* x86: every XMM/YMM/ZMM register and k1-7 are volatile; MXCSR status flags too. */
    ".globl _xf_clobber_vex\n_xf_clobber_vex:\n  movl 4(%esp), %ecx\n"
    "  pcmpeqd %xmm0, %xmm0\n  .irp i,1,2,3,4,5,6,7\n  movdqa %xmm0, %xmm\\i\n  .endr\n"
    "  testl $3, 4492(%ecx)\n  jz 1f\n"
    "  vpcmpeqd %ymm0, %ymm0, %ymm0\n  .irp i,1,2,3,4,5,6,7\n  vmovdqa %ymm0, %ymm\\i\n  .endr\n"
    "  testl $2, 4492(%ecx)\n  jz 1f\n"
    "  vpternlogd $0xff, %zmm0, %zmm0, %zmm0\n  .irp i,1,2,3,4,5,6,7\n  vmovdqa64 %zmm0, %zmm\\i\n  .endr\n"
    "  .irp i,1,2,3,4,5,6,7\n  kxnorw %k0, %k0, %k\\i\n  .endr\n"
    "1:\n  subl $4, %esp\n  stmxcsr (%esp)\n  orl $0x3f, (%esp)\n  ldmxcsr (%esp)\n  addl $4, %esp\n  ret\n"

    KIND(nop, "", "nop")
    KIND(div0, "", "divl %ecx")
    KIND(idivov, "", "idivl %ecx")
    KIND(int3, "", "int3")
    KIND(int3cd, "", ".byte 0xcd, 0x03")
    KIND(rd, "", "movl (%ecx), %eax")
    KIND(wr, "", "movl %eax, (%ecx)")
    KIND(exec, "", "call *%ecx")
    KIND(ss, "  pushfl\n  orl $0x100, (%esp)\n  popfl\n  nop\n", "nop")
    KIND(ud2, "", "ud2")
    KIND(hlt, "", "hlt")
    KIND(cli, "", "cli")
    KIND(sti, "", "sti")
    KIND(inb, "", "inb %dx, %al")
    KIND(outb, "", "outb %al, %dx")
    KIND(wbinvd, "", "wbinvd")
    KIND(rdmsr, "", "rdmsr")
    KIND(wrmsr, "", "wrmsr")
    KIND(movcr, "", "movl %cr0, %eax")
    KIND(clts, "", "clts")
    KIND(invd, "", "invd")
    KIND(xsetbv, "", "xsetbv")
    KIND(int2c, "", "int $0x2c")
    KIND(fastfail, "", "int $0x29")
    KIND(movaps, "", "movaps (%ecx), %xmm0")
    KIND(sse_div, "", "divps %xmm1, %xmm0")
    KIND(sse_sqrt, "", "sqrtps %xmm1, %xmm0")
    KIND(sse_mul, "", "mulps %xmm1, %xmm0")
    KIND(sse_add, "", "addps %xmm1, %xmm0")
    KIND(x87_div, "  fdiv %st(1), %st\n", "fwait")
    KIND(x87_ovf, "  fld1\n", "fwait")
    KIND(x87_mul, "  fmul %st(1), %st\n", "fwait")
    KIND(x87_sqrt, "  fsqrt\n", "fwait")
    KIND(avx_div, "", "vdivps %ymm1, %ymm0, %ymm0")
    KIND(avx_movmis, "", "vmovaps (%ecx), %ymm0")
    KIND(avx_ld, "", "vmovdqu (%ecx), %ymm2")
    KIND(stkovf, "  movl $16384, %ecx\n81:\n  subl $4096, %esp\n",
         "movl %eax, (%esp)\n  decl %ecx\n  jnz 81b\n  movl 4496(%ebx), %esp")
    KIND(into, "", "into")
    KIND(bound, "", "boundl %eax, (%ecx)")

    /* setjmp probe (msvcrt _setjmp(jmp_buf) / longjmp, cdecl) */
    XPRO(xsj_probe)
    "  call _xf_vex_load\n" GLOAD
    "  movl %esp, 4496(%ebx)\n  leal 4992(%ebx), %eax\n  pushl %eax\n"
    "  call __setjmp\n"
    GL(xsj_ret)
    "  addl $4, %esp\n"
    "  testl %eax, %eax\n  jnz 72f\n"
    "  ldmxcsr 4952(%ebx)\n  fldcw 4956(%ebx)\n"
    "  leal 4992(%ebx), %eax\n  pushl $1\n  pushl %eax\n"
    "  movl 4824+40(%ebx), %ebp\n  movl 4824+48(%ebx), %esi\n  movl 4824+56(%ebx), %edi\n"
    "  movl 4824+24(%ebx), %ebx\n"
    "  call _longjmp\n"
    "  int3\n"
    "72:\n" GSTORE
    "  fnstcw 4522(%ebx)\n  fldcw 4526(%ebx)\n"
    "  call _xf_vex_store\n"
    XEPI

    /* capture probe: RtlCaptureContext is stdcall(1) */
    XPRO(xcap_probe)
    "  call _xf_vex_load\n" GLOAD
    "  movl %esp, 4496(%ebx)\n"
    "  pushl 4800(%ebx)\n  call *4808(%ebx)\n"
    GL(xcap_ret)
    "  movl $2, 4784(%ebx)\n" GSTORE
    "  call _xf_vex_store\n"
    XEPI
);
#endif

/* ---- kind table ---------------------------------------------------------------------------------- */
typedef struct xk {
    const char *name;
    void (*fn)(void *);
    const char *op, *at, *resume;
    void (*sfn)(void *);               /* x64: same instruction in a frame with its own handler */
    const char *sop, *sat, *sresume;
} xk;

#define XKD(n) void xf_##n(void *); extern const char xf_##n##_op[], xf_##n##_at[], xf_##n##_resume[];
#define XKS(n) void xfs_##n(void *); extern const char xfs_##n##_op[], xfs_##n##_at[], xfs_##n##_resume[];
#define XKE(n) { #n, xf_##n, xf_##n##_op, xf_##n##_at, xf_##n##_resume, NULL, NULL, NULL, NULL },
#define XKES(n) { #n, xf_##n, xf_##n##_op, xf_##n##_at, xf_##n##_resume, xfs_##n, xfs_##n##_op, xfs_##n##_at, xfs_##n##_resume },
#define XK_COMMON(X) X(nop) X(idivov) X(int3cd) X(wr) X(exec) X(ss) X(ud2) X(hlt) X(cli) X(sti) X(inb) X(outb) \
    X(wbinvd) X(rdmsr) X(wrmsr) X(movcr) X(clts) X(invd) X(xsetbv) X(int2c) X(fastfail) X(movaps) X(sse_sqrt) \
    X(sse_mul) X(sse_add) X(x87_div) X(x87_ovf) X(x87_mul) X(x87_sqrt) X(avx_div) X(avx_movmis) X(avx_ld) X(stkovf)
XK_COMMON(XKD)
XKD(div0) XKD(rd) XKD(int3) XKD(sse_div)
#ifdef T_X64
XKD(div64)
XKS(div0) XKS(rd) XKS(int3) XKS(sse_div)
void xfo_outer(void *); void xfi_inner(void);
extern const char xfo_after_call[], xfo_landing[], xfi_fault[];
void *xseh_call(void *(*fn)(void *), void *arg);
extern const char xseh_call_ret[], xseh_call_landing[];
#define XKE_SEH XKES
#else
XKD(into) XKD(bound)
#define XKE_SEH XKE
#endif
void xf_clobber_vex(void *f);
void xsj_probe(void *f);  extern const char xsj_ret[];
void xcap_probe(void *f); extern const char xcap_ret[];

static const xk g_kinds[] = {
    XK_COMMON(XKE)
    XKE_SEH(div0) XKE_SEH(rd) XKE_SEH(int3) XKE_SEH(sse_div)
#ifdef T_X64
    XKE(div64)
#else
    XKE(into) XKE(bound)
#endif
};
static const xk *xk_find(const char *n) {
    for (size_t i = 0; i < T_COUNTOF(g_kinds); i++) if (!strcmp(g_kinds[i].name, n)) return &g_kinds[i];
    return NULL;
}

/* ========================================================================================== */
/* Context accessors                                                                            */
/* ========================================================================================== */
static const char *const g_gname[16] = {
#ifdef T_X64
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
#else
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"
#endif
};
/* GPRs loaded with patterns (rbx = frame pointer, rsp = stack) */
static int gpr_patterned(int i) { return i != 3 && i != 4 && i < NGPR; }

static uintptr_t *ctx_gpr(CONTEXT *c, int i) {
#ifdef T_X64
    return (uintptr_t *)&(&c->Rax)[i];
#else
    switch (i) {
    case 0: return (uintptr_t *)&c->Eax; case 1: return (uintptr_t *)&c->Ecx; case 2: return (uintptr_t *)&c->Edx;
    case 3: return (uintptr_t *)&c->Ebx; case 4: return (uintptr_t *)&c->Esp; case 5: return (uintptr_t *)&c->Ebp;
    case 6: return (uintptr_t *)&c->Esi; default: return (uintptr_t *)&c->Edi;
    }
#endif
}
#ifdef T_X64
#define CPC(c) ((uintptr_t)(c)->Rip)
#define CSP(c) ((uintptr_t)(c)->Rsp)
#define CSET_PC(c, v) ((c)->Rip = (DWORD64)(uintptr_t)(v))
#define CSET_SP(c, v) ((c)->Rsp = (DWORD64)(uintptr_t)(v))
#else
#define CPC(c) ((uintptr_t)(c)->Eip)
#define CSP(c) ((uintptr_t)(c)->Esp)
#define CSET_PC(c, v) ((c)->Eip = (DWORD)(uintptr_t)(v))
#define CSET_SP(c, v) ((c)->Esp = (DWORD)(uintptr_t)(v))
#endif

/* XMM i inside the CONTEXT, NULL when the CONTEXT does not carry it */
static uint8_t *ctx_xmm(CONTEXT *c, int i) {
#ifdef T_X64
    return (uint8_t *)&c->FltSave.XmmRegisters[i];
#else
    return (c->ContextFlags & X_CONTEXT_EXT) == X_CONTEXT_EXT ? c->ExtendedRegisters + 160 + 16 * i : NULL;
#endif
}
static int ctx_has_fx(CONTEXT *c) {
#ifdef T_X64
    return (c->ContextFlags & 0x100008u) == 0x100008u;
#else
    return (c->ContextFlags & X_CONTEXT_EXT) == X_CONTEXT_EXT;
#endif
}
/* FXSAVE-format legacy area of the context (x64 FltSave, x86 ExtendedRegisters) */
static uint8_t *ctx_fx(CONTEXT *c) {
#ifdef T_X64
    return (uint8_t *)&c->FltSave;
#else
    return ctx_has_fx(c) ? c->ExtendedRegisters : NULL;
#endif
}
static uint32_t ctx_mxcsr(CONTEXT *c) {
#ifdef T_X64
    return c->MxCsr;
#else
    uint32_t v = 0; if (ctx_has_fx(c)) memcpy(&v, c->ExtendedRegisters + 24, 4); return v;
#endif
}
static void ctx_set_mxcsr(CONTEXT *c, uint32_t v) {
#ifdef T_X64
    c->MxCsr = v; c->FltSave.MxCsr = v;
#else
    if (ctx_has_fx(c)) memcpy(c->ExtendedRegisters + 24, &v, 4);
#endif
}
static uintptr_t read_eflags(void) {
    uintptr_t f;
#ifdef T_X64
    __asm__ volatile("pushfq\n popq %0" : "=r"(f));
#else
    __asm__ volatile("pushfl\n popl %0" : "=r"(f));
#endif
    return f;
}

/* XSTATE API (kernel32, Windows 7 SP1+), resolved once */
typedef PVOID (WINAPI *pfn_locate)(PCONTEXT, DWORD, PDWORD);
typedef BOOL (WINAPI *pfn_getmask)(PCONTEXT, PDWORD64);
typedef BOOL (WINAPI *pfn_setmask)(PCONTEXT, DWORD64);
typedef BOOL (WINAPI *pfn_initctx)(PVOID, DWORD, PCONTEXT *, PDWORD);
static pfn_locate g_locate; static pfn_getmask g_getmask; static pfn_setmask g_setmask; static pfn_initctx g_initctx;

/* 80-bit extended from double (normal numbers and zero) */
static void f80_from_double(uint8_t out[16], double d) {
    uint64_t b; memcpy(&b, &d, 8);
    unsigned sign = (unsigned)(b >> 63), e = (unsigned)((b >> 52) & 0x7FF);
    uint64_t m = b & ((1ull << 52) - 1), mant = 0; uint16_t ex = (uint16_t)(sign << 15);
    if (e) { mant = (1ull << 63) | (m << 11); ex = (uint16_t)((sign << 15) | (e - 1023 + 16383)); }
    memset(out, 0, 16); memcpy(out, &mant, 8); memcpy(out + 8, &ex, 2);
}
static uint32_t f32bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static void fill_ps(uint8_t *v, uint32_t bits) { for (int i = 0; i < 4; i++) memcpy(v + 4 * i, &bits, 4); }

/* ========================================================================================== */
/* Frame setup / result checks                                                                  */
/* ========================================================================================== */
static int vk_of(const xf *f) { return (f->flags & XF_ZMM) ? 64 : (f->flags & XF_YMM) ? 32 : 16; }
static int nv_of(const xf *f) { return (f->flags & XF_ZMM) ? T_NZMM : T_NXMM; }

static xf *xf_alloc(void) {
    void *p = VirtualAlloc(NULL, sizeof(xf), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return (xf *)p;
}
static void xf_free(xf *f) { if (f) VirtualFree(f, 0, MEM_RELEASE); }

static void xf_sync_exp(xf *f) {
    memcpy(f->vexp, f->vin, sizeof f->vexp);
    memcpy(f->gexp, f->gin, sizeof f->gexp);
    memcpy(f->kexp, f->kin, sizeof f->kexp);
    memcpy(f->x87exp, f->x87in, sizeof f->x87exp);
    f->mxexp = f->mxcsr_in; f->fcwexp = f->fcw_in; f->mx_any_out = 0;
}

/* tag identifies the owner of every pattern; vk = 16/32/64 */
static void xf_init(xf *f, uint32_t tag, int vk, int x87, int efl) {
    memset(f, 0, sizeof *f);
    for (int i = 0; i < 32; i++) {
        t_pattern_reg(f->vin[i], 64, tag, (uint32_t)i);
        if (vk < 64) memset(f->vin[i] + vk, 0, (size_t)(64 - vk));
    }
    for (int i = 0; i < 8; i++) f->kin[i] = t_pattern_word(tag, 40u + (uint32_t)i, 9);
    for (int i = 0; i < 16; i++) {
        f->gin[i] = t_pattern_word(tag, (uint32_t)i, 12);
        f->gin2[i] = t_pattern_word(tag ^ 0x5A5Au, (uint32_t)i, 13);
#ifdef T_X86
        f->gin[i] &= 0xFFFFFFFFu; f->gin2[i] &= 0xFFFFFFFFu;
#endif
    }
    f->flags = (vk == 64 ? XF_ZMM : vk == 32 ? XF_YMM : 0) | (x87 ? XF_X87 : 0) | (efl ? XF_EFL : 0);
    if (vk == 64 && cpu_has(F_AVX512BW)) f->flags |= XF_KBW;
    if (vk == 64 && !(f->flags & XF_KBW)) for (int i = 0; i < 8; i++) f->kin[i] &= 0xFFFF;
    f->mxcsr_in = 0xBF80;                /* RC = round down, FTZ, all masked: distinct from the 0x1F80 default */
    f->mxcsr_restore = t_get_mxcsr();
    f->fcw_in = 0x0E7F;                  /* RC = chop, PC = 53 bit, all masked */
    f->fcw_restore = t_get_x87_cw();
    f->efl_in = 0xCD7;                   /* CF PF AF ZF SF DF OF + reserved bit 1 */
    for (int i = 0; i < 8; i++) f80_from_double(f->x87in[i], (double)(tag % 1000u) * 8.0 + i + 1.25);
    f->mxcsr2 = 0x5F80; f->fcw2 = 0x0F7F;
    xf_sync_exp(f);
}

static void xf_check_out(xf *f, const char *label) {
    int vk = vk_of(f), n = nv_of(f);
    char what[128];
    const char *rn = vk == 64 ? "zmm" : vk == 32 ? "ymm" : "xmm";
    for (int i = 0; i < n; i++) {
        if (f->vskip & (1u << i)) continue;
        snprintf(what, sizeof what, "%s%d %s", rn, i, label);
        CHECK_REG(what, f->vout[i], f->vexp[i], (size_t)vk);
    }
    if (f->flags & XF_ZMM) {
        uint64_t m = (f->flags & XF_KBW) ? ~0ull : 0xFFFFull;
        for (int i = 0; i < 8; i++)
            CHECKF((f->kout[i] & m) == (f->kexp[i] & m), "k%d %s: got 0x%llx want 0x%llx", i, label,
                   (unsigned long long)(f->kout[i] & m), (unsigned long long)(f->kexp[i] & m));
    }
    for (int i = 0; i < NGPR; i++) {
        if (!gpr_patterned(i) || (f->gskip & (1u << i))) continue;
        CHECKF((uintptr_t)f->gout[i] == (uintptr_t)f->gexp[i], "%s %s: got 0x%llx want 0x%llx", g_gname[i], label,
               (unsigned long long)(uintptr_t)f->gout[i], (unsigned long long)(uintptr_t)f->gexp[i]);
    }
    CHECKF((f->mxcsr_out & ~f->mx_any_out) == (f->mxexp & ~f->mx_any_out), "MXCSR %s: got 0x%x want 0x%x (ignoring 0x%x)",
           label, f->mxcsr_out, f->mxexp, f->mx_any_out);
    if (f->flags & XF_X87) {
        for (int i = 0; i < 8; i++) {
            char a[64], b[64];
            CHECKF(!memcmp(f->x87out[i], f->x87exp[i], 10), "ST(%d) %s: got %s want %s", i, label,
                   t_hex(a, sizeof a, f->x87out[i], 10), t_hex(b, sizeof b, f->x87exp[i], 10));
        }
        CHECKF(f->fcw_out == f->fcwexp, "x87 CW %s: got 0x%04x want 0x%04x", label, f->fcw_out, f->fcwexp);
    }
    if (f->flags & XF_EFL)
        CHECKF((f->efl_out & 0xCD5) == (f->efl_in & 0xCD5), "EFLAGS %s: got 0x%llx want arithmetic/DF bits 0x%llx",
               label, (unsigned long long)f->efl_out, (unsigned long long)(f->efl_in & 0xCD5));
}

/* ========================================================================================== */
/* Handler state                                                                                */
/* ========================================================================================== */
enum { RS_SKIP, RS_STAY, RS_RET };
enum {
    M_MOD_XMM = 1 << 0, M_MOD_GPR = 1 << 1, M_MOD_MXCSR = 1 << 2, M_MOD_MXCSR_TOP = 1 << 3, M_MOD_YMM = 1 << 4,
    M_MOD_X87 = 1 << 5, M_CLOBBER = 1 << 6, M_FPFIX = 1 << 7, M_CLEAR_TF = 1 << 8, M_NOVERIFY = 1 << 9,
    M_NESTED_VEH = 1 << 10, M_CLEAR_AVX = 1 << 11, M_MOD_ZMM = 1 << 12, M_VEH_SEARCH = 1 << 13,
    M_SET_RF = 1 << 14, M_CLEAR_DR7 = 1 << 15, M_NESTED_HW = 1 << 16, M_RESET_SP = 1 << 17
};
/* V_NONE: no vectored or frame handler of ours; only the top-level (unhandled) exception filter (children only) */
enum { V_VEH = 0, V_SEHFRAME = 1, V_SEHCALL = 2, V_NONE = 3 };
#ifdef T_X64
#define X_CONTEXT_DEBUG 0x100010u
#else
#define X_CONTEXT_DEBUG 0x10010u
#endif
#define XS_STACK_OVERFLOW 0xC00000FDu

typedef struct xexp {
    DWORD code;              /* 0 = no exception expected */
    DWORD alt[3];            /* also accepted (NOTE) */
    int nparams;             /* -1 = not checked */
    int ninfo; ULONG_PTR info[3];
    const void *addr; int addr_soft;   /* ExceptionAddress (NULL = the 'at' label) */
    const void *pc; int pc_soft;       /* context PC (NULL = the 'at' label) */
    intptr_t sp_delta;
    uint32_t mx_set, mx_any;           /* context MXCSR = mxcsr_in | mx_set (mx_any bits ignored) */
    uint16_t x87_sw_set;               /* x87 kinds: bits expected in the context status word */
    uint16_t x87_sw_clear;             /* x87 kinds: bits that must be clear in the context status word */
    int check_fip;
    int resume;
    int sp_soft;                       /* context SP not predictable (stack overflow): NOTE only */
    uint32_t gany;                     /* GPRs (bit = hardware number) not compared in the handler CONTEXT */
    int x87_st0;                       /* context ST(0): 0 = x87in[0], 1 = any (recorded), 2 = x87_st0_val */
    uint8_t x87_st0_val[10];
    int check_dr;                      /* hardware breakpoint: Dr0 = 'at', Dr6.B0, Dr7.L0 in the CONTEXT */
} xexp;

struct xh;
typedef EXCEPTION_DISPOSITION (*xseh_fn)(struct xh *, EXCEPTION_RECORD *, void *, CONTEXT *, void *);
typedef struct xh {
    volatile LONG armed;
    int veh_on;
    xseh_fn sehfn;
    const xk *k; xf *f; xexp e; int mode; uint32_t tag2;
    const char *op, *at, *resume;
    volatile LONG calls, veh_calls, seh_calls, unwind_calls, nested_calls;
    EXCEPTION_RECORD rec;
    uintptr_t ctx_pc, ctx_sp; DWORD ctxflags; int had_xstate; DWORD64 xmask;
    uint32_t live_mxcsr; uint16_t live_fcw; uintptr_t live_eflags;
    LONG veh_calls_at_seh;
    DWORD nested_flags, nested_code, unwind_flags;
    void *est_frame;
    int dc_valid; uintptr_t dc_pc, dc_imagebase, dc_begin, dc_frame;
    uint8_t dc_xmm[16][16]; uint64_t dc_gpr[16];
    uintptr_t retval;
    int seh_action;          /* 0 = continue execution, 1 = search, 2 = unwind to resume label */
    int quiet;
    uint8_t ctx_st0[10];     /* ST(0) as found in the handler CONTEXT */
    int ctx_has_dr; uintptr_t ctx_dr0, ctx_dr6, ctx_dr7;
    int nested_hw_calls;     /* M_NESTED_HW: inner handler invocations */
} xh;

static DWORD g_tls = TLS_OUT_OF_INDEXES;
static PVOID g_veh;
static LONG CALLBACK xh_veh(EXCEPTION_POINTERS *ep);

static void xs_init(void) {
    if (g_tls == TLS_OUT_OF_INDEXES) g_tls = TlsAlloc();
    if (!g_veh) g_veh = AddVectoredExceptionHandler(1, xh_veh);
    if (!g_locate) {
        HMODULE k = GetModuleHandleA("kernel32.dll");
        g_locate = (pfn_locate)(void *)GetProcAddress(k, "LocateXStateFeature");
        g_getmask = (pfn_getmask)(void *)GetProcAddress(k, "GetXStateFeaturesMask");
        g_setmask = (pfn_setmask)(void *)GetProcAddress(k, "SetXStateFeaturesMask");
        g_initctx = (pfn_initctx)(void *)GetProcAddress(k, "InitializeContext");
    }
}
static xh *xh_cur(void) { return g_tls == TLS_OUT_OF_INDEXES ? NULL : (xh *)TlsGetValue(g_tls); }

/* Is this exception the one the armed kind routine produces? */
static int xh_mine(xh *s, EXCEPTION_RECORD *rec, CONTEXT *c) {
    uintptr_t a = (uintptr_t)rec->ExceptionAddress, pc = CPC(c);
    uintptr_t lo = (uintptr_t)s->op - 2, hi = (uintptr_t)s->resume + 2;
    if ((a >= lo && a <= hi) || (pc >= lo && pc <= hi)) return 1;
    if (s->e.addr && a == (uintptr_t)s->e.addr) return 1;
    return 0;
}

static void xh_check_xstate(xh *s, CONTEXT *c, const char *who) {
    xf *f = s->f;
    int has = (c->ContextFlags & X_CONTEXT_XSTATE) == X_CONTEXT_XSTATE;
    s->had_xstate = has;
#ifdef T_X64
    CHECKF(has, "%s: exception CONTEXT lacks CONTEXT_XSTATE (ContextFlags 0x%lx) while YMM/ZMM state is live", who,
           (unsigned long)c->ContextFlags);
#else
    if (!has) { if (!s->quiet) t_note("%s: 32-bit exception CONTEXT carries no XSTATE (flags 0x%lx); upper-half "
                                      "state checked only after resume", who, (unsigned long)c->ContextFlags); return; }
#endif
    if (!has || !g_getmask || !g_locate) return;
    DWORD64 m = 0;
    CHECKF(g_getmask(c, &m), "%s: GetXStateFeaturesMask failed on the exception CONTEXT", who);
    s->xmask = m;
    CHECKF(m & XSM_AVX, "%s: XSTATE mask 0x%llx lacks AVX although the YMM upper halves are non-zero", who,
           (unsigned long long)m);
    DWORD len = 0;
    uint8_t *p = (uint8_t *)g_locate(c, XSF_AVX, &len);
    CHECKF(p != NULL && len >= 16u * T_NXMM, "%s: LocateXStateFeature(AVX) = %p len %lu", who, (void *)p, (unsigned long)len);
    if (p && len >= 16u * T_NXMM && (m & XSM_AVX)) {
        char what[64];
        for (int i = 0; i < T_NXMM; i++) {
            snprintf(what, sizeof what, "%s: ctx ymm%d[255:128]", who, i);
            CHECK_REG(what, p + 16 * i, f->vin[i] + 16, 16);
        }
    }
    if (f->flags & XF_ZMM) {
        uint64_t km = (f->flags & XF_KBW) ? ~0ull : 0xFFFFull;
        len = 0; p = (uint8_t *)g_locate(c, XSF_OPMASK, &len);
        CHECKF(p && len >= 64, "%s: LocateXStateFeature(OPMASK) = %p len %lu", who, (void *)p, (unsigned long)len);
        if (p && len >= 64) for (int i = 0; i < 8; i++) {
            uint64_t v; memcpy(&v, p + 8 * i, 8);
            CHECKF((v & km) == (f->kin[i] & km), "%s: ctx k%d 0x%llx want 0x%llx", who, i, (unsigned long long)v,
                   (unsigned long long)f->kin[i]);
        }
        len = 0; p = (uint8_t *)g_locate(c, XSF_ZMM_H, &len);
        CHECKF(p && len >= 32u * T_NXMM, "%s: LocateXStateFeature(ZMM_Hi256) = %p len %lu", who, (void *)p, (unsigned long)len);
        if (p && len >= 32u * T_NXMM) for (int i = 0; i < T_NXMM; i++) {
            char what[64]; snprintf(what, sizeof what, "%s: ctx zmm%d[511:256]", who, i);
            CHECK_REG(what, p + 32 * i, f->vin[i] + 32, 32);
        }
#ifdef T_X64
        len = 0; p = (uint8_t *)g_locate(c, XSF_HI_ZMM, &len);
        CHECKF(p && len >= 1024, "%s: LocateXStateFeature(Hi16_ZMM) = %p len %lu", who, (void *)p, (unsigned long)len);
        if (p && len >= 1024) for (int i = 16; i < 32; i++) {
            char what[64]; snprintf(what, sizeof what, "%s: ctx zmm%d", who, i);
            CHECK_REG(what, p + 64 * (i - 16), f->vin[i], 64);
        }
#endif
    }
}

/* Compare the handler's EXCEPTION_RECORD and CONTEXT with the reference frame. */
static void xh_verify(xh *s, EXCEPTION_RECORD *rec, CONTEXT *c, const char *who) {
    xf *f = s->f; const xexp *e = &s->e;
    DWORD code = rec->ExceptionCode;
    int alt = 0;
    for (int i = 0; i < 3; i++) if (e->alt[i] && code == e->alt[i]) alt = 1;
    CHECKF(code == e->code || alt, "%s: ExceptionCode 0x%08lx, expected 0x%08lx", who, (unsigned long)code, (unsigned long)e->code);
    if (alt && !s->quiet) t_note("%s: ExceptionCode 0x%08lx (accepted alternative to 0x%08lx)", who, (unsigned long)code,
                                 (unsigned long)e->code);
    CHECKF(rec->ExceptionRecord == NULL, "%s: nested ExceptionRecord pointer %p for a first exception", who, (void *)rec->ExceptionRecord);
    CHECKF((rec->ExceptionFlags & X_EXC_NONCONTINUABLE) == 0, "%s: hardware exception marked non-continuable (flags 0x%lx)",
           who, (unsigned long)rec->ExceptionFlags);
    if (e->nparams >= 0)
        CHECKF((int)rec->NumberParameters == e->nparams, "%s: NumberParameters %lu want %d", who,
               (unsigned long)rec->NumberParameters, e->nparams);
    else if (!s->quiet)
        t_info("nparams", "%s: %lu (info0=0x%llx info1=0x%llx)", who, (unsigned long)rec->NumberParameters,
               (unsigned long long)(rec->NumberParameters > 0 ? rec->ExceptionInformation[0] : 0),
               (unsigned long long)(rec->NumberParameters > 1 ? rec->ExceptionInformation[1] : 0));
    for (int i = 0; i < e->ninfo; i++)
        CHECKF(rec->NumberParameters > (DWORD)i && rec->ExceptionInformation[i] == e->info[i],
               "%s: ExceptionInformation[%d] = 0x%llx want 0x%llx", who, i,
               (unsigned long long)(rec->NumberParameters > (DWORD)i ? rec->ExceptionInformation[i] : 0),
               (unsigned long long)e->info[i]);
    const void *want_addr = e->addr ? e->addr : (const void *)s->at;
    if (e->addr_soft) {
        if (rec->ExceptionAddress != want_addr && !s->quiet)
            t_note("%s: ExceptionAddress = label%+ld", who, (long)((intptr_t)rec->ExceptionAddress - (intptr_t)want_addr));
    } else
        CHECKF(rec->ExceptionAddress == want_addr, "%s: ExceptionAddress %p want %p (label%+ld)", who, rec->ExceptionAddress,
               want_addr, (long)((intptr_t)rec->ExceptionAddress - (intptr_t)want_addr));
    /* ---- CONTEXT ---- */
    CHECKF((c->ContextFlags & X_CONTEXT_FULLFP) == X_CONTEXT_FULLFP, "%s: ContextFlags 0x%lx lack CONTROL|INTEGER%s", who,
           (unsigned long)c->ContextFlags, PTRSZ == 8 ? "|FLOATING_POINT" : "");
    const void *want_pc = e->pc ? e->pc : (const void *)s->at;
    if (e->pc_soft) {
        if (CPC(c) != (uintptr_t)want_pc && !s->quiet)
            t_note("%s: context PC = label%+ld", who, (long)((intptr_t)CPC(c) - (intptr_t)want_pc));
    } else
        CHECKF(CPC(c) == (uintptr_t)want_pc, "%s: context PC %p want %p (label%+ld)", who, (void *)CPC(c), want_pc,
               (long)((intptr_t)CPC(c) - (intptr_t)want_pc));
    if (e->sp_soft) {
        if (!s->quiet) t_info("ctx_sp", "%s: context SP = rsp_at%+lld", who, (long long)((intptr_t)CSP(c) - (intptr_t)f->rsp_at));
    } else
        CHECKF(CSP(c) == (uintptr_t)f->rsp_at + (uintptr_t)e->sp_delta, "%s: context SP 0x%llx want 0x%llx", who,
               (unsigned long long)CSP(c), (unsigned long long)((uintptr_t)f->rsp_at + (uintptr_t)e->sp_delta));
    CHECKF(*ctx_gpr(c, 3) == (uintptr_t)f, "%s: context %s 0x%llx want frame %p", who, g_gname[3],
           (unsigned long long)*ctx_gpr(c, 3), (void *)f);
    for (int i = 0; i < NGPR; i++) {
        if (!gpr_patterned(i) || (e->gany & (1u << i))) continue;
        CHECKF(*ctx_gpr(c, i) == (uintptr_t)f->gin[i], "%s: context %s 0x%llx want 0x%llx", who, g_gname[i],
               (unsigned long long)*ctx_gpr(c, i), (unsigned long long)(uintptr_t)f->gin[i]);
    }
    if (!ctx_has_fx(c)) {
        if (!s->quiet) t_note("%s: CONTEXT carries no FXSAVE/extended registers (flags 0x%lx); XMM/MXCSR checked after resume only",
                              who, (unsigned long)c->ContextFlags);
    } else {
        char what[64];
        for (int i = 0; i < T_NXMM; i++) {
            snprintf(what, sizeof what, "%s: ctx xmm%d", who, i);
            CHECK_REG(what, ctx_xmm(c, i), f->vin[i], 16);
        }
        uint32_t want = (f->mxcsr_in | e->mx_set) & ~e->mx_any, got = ctx_mxcsr(c) & ~e->mx_any;
        CHECKF(got == want, "%s: context MXCSR 0x%x want 0x%x", who, ctx_mxcsr(c), (f->mxcsr_in | e->mx_set));
#ifdef T_X64
        CHECKF(c->FltSave.MxCsr == c->MxCsr, "%s: CONTEXT.MxCsr 0x%lx != FltSave.MxCsr 0x%lx", who,
               (unsigned long)c->MxCsr, (unsigned long)c->FltSave.MxCsr);
#endif
        if (f->flags & XF_X87) {
            uint8_t *fx = ctx_fx(c);
            uint16_t cw, sw; memcpy(&cw, fx + 0, 2); memcpy(&sw, fx + 2, 2);
            CHECKF(cw == f->fcw_in, "%s: context x87 CW 0x%04x want 0x%04x", who, cw, f->fcw_in);
            CHECKF(fx[4] == 0xFF, "%s: context abridged x87 tag 0x%02x want 0xFF (8 valid registers)", who, fx[4]);
            memcpy(s->ctx_st0, fx + 32, 10);
            for (int i = 0; i < 8; i++) {
                char a[48], b[48];
                const uint8_t *want = (i == 0 && e->x87_st0 == 2) ? e->x87_st0_val : f->x87in[i];
                if (i == 0 && e->x87_st0 == 1) continue;   /* destination written with a scaled result: recorded */
                CHECKF(!memcmp(fx + 32 + 16 * i, want, 10), "%s: context ST(%d) %s want %s", who, i,
                       t_hex(a, sizeof a, fx + 32 + 16 * i, 10), t_hex(b, sizeof b, want, 10));
            }
            if (e->x87_sw_set)
                CHECKF((sw & e->x87_sw_set) == e->x87_sw_set, "%s: context x87 SW 0x%04x lacks 0x%04x", who, sw, e->x87_sw_set);
            if (e->x87_sw_clear)
                CHECKF((sw & e->x87_sw_clear) == 0, "%s: context x87 SW 0x%04x has 0x%04x set", who, sw, sw & e->x87_sw_clear);
            if (e->check_fip) {
                uint32_t fip; memcpy(&fip, fx + 8, 4);
                CHECKF(fip == (uint32_t)(uintptr_t)s->op, "%s: context x87 FIP 0x%08x want 0x%08x (the faulting x87 op)", who,
                       fip, (uint32_t)(uintptr_t)s->op);
            }
#ifdef T_X86
            if ((c->ContextFlags & X_CONTEXT_FP) == X_CONTEXT_FP) {
                CHECKF((c->FloatSave.ControlWord & 0xFFFF) == f->fcw_in, "%s: FloatSave.ControlWord 0x%lx want 0x%04x", who,
                       (unsigned long)c->FloatSave.ControlWord, f->fcw_in);
                for (int i = 0; i < 8; i++) {
                    if (i == 0 && e->x87_st0 == 1) continue;
                    const uint8_t *want = (i == 0 && e->x87_st0 == 2) ? e->x87_st0_val : f->x87in[i];
                    CHECKF(!memcmp(c->FloatSave.RegisterArea + 10 * i, want, 10), "%s: FloatSave ST(%d) differs", who, i);
                }
            }
#endif
        }
    }
    if (f->flags & XF_EFL)
        CHECKF((c->EFlags & 0xCD5) == (f->efl_in & 0xCD5), "%s: context EFLAGS 0x%lx want arithmetic/DF bits 0x%llx", who,
               (unsigned long)c->EFlags, (unsigned long long)(f->efl_in & 0xCD5));
    if (f->flags & (XF_YMM | XF_ZMM)) xh_check_xstate(s, c, who);
    if (e->check_dr) {
        s->ctx_has_dr = (c->ContextFlags & X_CONTEXT_DEBUG) == X_CONTEXT_DEBUG;
        if (s->ctx_has_dr) {
            s->ctx_dr0 = (uintptr_t)c->Dr0; s->ctx_dr6 = (uintptr_t)c->Dr6; s->ctx_dr7 = (uintptr_t)c->Dr7;
            CHECKF(s->ctx_dr0 == (uintptr_t)s->at, "%s: context Dr0 %p want the breakpoint address %p", who, (void *)s->ctx_dr0,
                   (const void *)s->at);
            CHECKF(s->ctx_dr6 & 1, "%s: context Dr6 0x%llx lacks B0 (breakpoint 0 hit)", who, (unsigned long long)s->ctx_dr6);
            CHECKF(s->ctx_dr7 & 1, "%s: context Dr7 0x%llx lacks L0", who, (unsigned long long)s->ctx_dr7);
        } else if (!s->quiet)
            t_note("%s: exception CONTEXT has no CONTEXT_DEBUG_REGISTERS (flags 0x%lx)", who, (unsigned long)c->ContextFlags);
    }
}

static void xh_nested_hw(xh *s, EXCEPTION_RECORD *rec, CONTEXT *c, const char *who);

static void xh_modify(xh *s, CONTEXT *c) {
    xf *f = s->f;
    if (s->mode & M_MOD_XMM) {
        for (int i = 0; i < T_NXMM; i++) {
            uint8_t nv[16]; t_pattern_reg(nv, 16, s->tag2, (uint32_t)i);
            uint8_t *p = ctx_xmm(c, i);
            if (!p) break;
            memcpy(p, nv, 16); memcpy(f->vexp[i], nv, 16);
        }
    }
    if (s->mode & M_MOD_GPR) {
        for (int i = 0; i < NGPR; i++) {
            if (!gpr_patterned(i)) continue;
            uintptr_t v = (uintptr_t)t_pattern_word(s->tag2, (uint32_t)i, 14);
            *ctx_gpr(c, i) = v; f->gexp[i] = v;
        }
    }
    if (s->mode & M_MOD_MXCSR) { ctx_set_mxcsr(c, 0x5F80); f->mxexp = 0x5F80; }
#ifdef T_X64
    if (s->mode & M_MOD_MXCSR_TOP) c->MxCsr = 0x5F80;          /* FltSave.MxCsr left as it was */
#endif
    if ((s->mode & (M_MOD_YMM | M_CLEAR_AVX | M_MOD_ZMM)) && (c->ContextFlags & X_CONTEXT_XSTATE) == X_CONTEXT_XSTATE &&
        g_locate && g_getmask && g_setmask) {
        DWORD64 m = 0; g_getmask(c, &m);
        DWORD len = 0;
        if (s->mode & M_MOD_YMM) {
            uint8_t *p = (uint8_t *)g_locate(c, XSF_AVX, &len);
            if (p && len >= 16u * T_NXMM) {
                for (int i = 0; i < T_NXMM; i++) {
                    uint8_t nv[32]; t_pattern_reg(nv, 32, s->tag2 + 1, (uint32_t)i);
                    memcpy(p + 16 * i, nv + 16, 16); memcpy(f->vexp[i] + 16, nv + 16, 16);
                }
                g_setmask(c, m | XSM_AVX);
            }
        }
        if (s->mode & M_MOD_ZMM) {
            uint8_t *p = (uint8_t *)g_locate(c, XSF_ZMM_H, &len);
            if (p && len >= 32u * T_NXMM)
                for (int i = 0; i < T_NXMM; i++) {
                    uint8_t nv[64]; t_pattern_reg(nv, 64, s->tag2 + 2, (uint32_t)i);
                    memcpy(p + 32 * i, nv + 32, 32); memcpy(f->vexp[i] + 32, nv + 32, 32);
                }
            g_setmask(c, m | (1ull << XSF_ZMM_H));
        }
        if (s->mode & M_CLEAR_AVX) g_setmask(c, m & ~XSM_AVX);
    }
    if (s->mode & M_MOD_X87) {
        uint8_t nv[16]; f80_from_double(nv, 12345.5);
        uint8_t *fx = ctx_fx(c);
        if (fx) { memcpy(fx + 32, nv, 10); memcpy(f->x87exp[0], nv, 10); }
#ifdef T_X86
        if ((c->ContextFlags & X_CONTEXT_FP) == X_CONTEXT_FP) memcpy(c->FloatSave.RegisterArea, nv, 10);
#endif
    }
    if (s->mode & M_FPFIX) {
        if (s->k && !strncmp(s->k->name, "x87", 3)) {
            uint8_t *fx = ctx_fx(c);
            if (fx) { uint16_t sw; memcpy(&sw, fx + 2, 2); sw &= 0x7F00; memcpy(fx + 2, &sw, 2); }
#ifdef T_X86
            if ((c->ContextFlags & X_CONTEXT_FP) == X_CONTEXT_FP) c->FloatSave.StatusWord &= 0x7F00;
#endif
        } else {
            ctx_set_mxcsr(c, (ctx_mxcsr(c) | 0x1F80) & ~0x3Fu);
        }
    }
    if (s->mode & M_CLEAR_TF) c->EFlags &= ~0x100u;
    if (s->mode & M_SET_RF) c->EFlags |= 0x10000u;      /* resume flag: re-execute the instruction without re-hitting DR0 */
    if (s->mode & M_CLEAR_DR7) c->Dr7 = 0;
}

static LONG CALLBACK nested_inner_veh(EXCEPTION_POINTERS *ep);
static volatile LONG g_nested_hits;

/* Common handler body. Returns 1 to continue execution. */
static int xh_core(xh *s, EXCEPTION_RECORD *rec, CONTEXT *c, const char *who) {
    LONG n = InterlockedIncrement(&s->calls);
    s->rec = *rec;
    s->ctx_pc = CPC(c); s->ctx_sp = CSP(c); s->ctxflags = c->ContextFlags;
    s->live_mxcsr = t_get_mxcsr(); s->live_fcw = t_get_x87_cw(); s->live_eflags = read_eflags();
    if (n > 3) {
        CHECKF(0, "%s: exception 0x%08lx at %p repeated %ld times; forcing resume", who, (unsigned long)rec->ExceptionCode,
               rec->ExceptionAddress, (long)n);
        CSET_PC(c, s->resume);
        if (s->mode & M_RESET_SP) CSET_SP(c, s->f->rsp_at);
        if (s->k && !strncmp(s->k->name, "x87", 3)) s->mode |= M_FPFIX;
        s->mode |= M_SET_RF | M_CLEAR_DR7;
        xh_modify(s, c);
        return 1;
    }
    if (!(s->mode & M_NOVERIFY)) xh_verify(s, rec, c, who);
    if (s->mode & M_NESTED_HW) xh_nested_hw(s, rec, c, who);
    xh_modify(s, c);
    switch (s->e.resume) {
    case RS_SKIP: CSET_PC(c, s->resume); if (s->mode & M_RESET_SP) CSET_SP(c, s->f->rsp_at); break;
    case RS_RET: {
        uintptr_t ra = *(uintptr_t *)CSP(c);
        CHECKF(ra == (uintptr_t)s->resume, "%s: return address on the faulting stack %p want %p", who, (void *)ra, (const void *)s->resume);
        CSET_PC(c, ra); CSET_SP(c, CSP(c) + PTRSZ);
        break;
    }
    default: break;
    }
    if (s->mode & M_NESTED_VEH) {
        PVOID h = AddVectoredExceptionHandler(1, nested_inner_veh);
        ULONG_PTR arg = 0x1234;
        RaiseException(0xE0000ABCu, 0, 1, &arg);
        if (h) RemoveVectoredExceptionHandler(h);
        InterlockedIncrement(&s->nested_calls);
    }
    if (s->mode & M_CLOBBER) xf_clobber_vex(s->f);
    return 1;
}

static LONG CALLBACK xh_veh(EXCEPTION_POINTERS *ep) {
    xh *s = xh_cur();
    if (!s || !s->armed || !s->veh_on) return EXCEPTION_CONTINUE_SEARCH;
    if (!xh_mine(s, ep->ExceptionRecord, ep->ContextRecord)) return EXCEPTION_CONTINUE_SEARCH;
    InterlockedIncrement(&s->veh_calls);
    if (s->mode & M_VEH_SEARCH) {
        /* verify only; leave the exception to the frame handler (CONTEXT untouched) */
        xh_verify(s, ep->ExceptionRecord, ep->ContextRecord, "VEH");
        return EXCEPTION_CONTINUE_SEARCH;
    }
    return xh_core(s, ep->ExceptionRecord, ep->ContextRecord, "VEH") ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

/* frame-based handler trampoline (x64: named in .seh_handler; x86: put in the FS:0 registration) */
EXCEPTION_DISPOSITION __cdecl xs_seh_tramp(EXCEPTION_RECORD *rec, void *frame, CONTEXT *ctx, void *dc);
__attribute__((used)) EXCEPTION_DISPOSITION __cdecl xs_seh_tramp(EXCEPTION_RECORD *rec, void *frame, CONTEXT *ctx, void *dc) {
    xh *s = xh_cur();
    if (!s || !s->sehfn) return ExceptionContinueSearch;
    return s->sehfn(s, rec, frame, ctx, dc);
}

#ifdef T_X86
typedef struct x86reg { struct x86reg *next; void *handler; } x86reg;
static T_NOINLINE void *xseh_call(void *(*fn)(void *), void *arg) {
    volatile x86reg r;
    void *ret;
    r.handler = (void *)xs_seh_tramp;
    __asm__ volatile("movl %%fs:0, %0" : "=r"(r.next));
    __asm__ volatile("movl %0, %%fs:0" :: "r"(&r) : "memory");
    ret = fn(arg);
    __asm__ volatile("movl %0, %%fs:0" :: "r"(r.next) : "memory");
    return ret;
}
#endif

/* SEH handler used by the kind tests */
static EXCEPTION_DISPOSITION seh_kind(xh *s, EXCEPTION_RECORD *rec, void *frame, CONTEXT *c, void *dcv) {
    if (rec->ExceptionFlags & X_EXC_UNWINDING) {
        InterlockedIncrement(&s->unwind_calls);
        s->unwind_flags = rec->ExceptionFlags;
        return ExceptionContinueSearch;
    }
    if (!s->armed) return ExceptionContinueSearch;
    if (rec->ExceptionFlags & X_EXC_NESTED_CALL) {
        s->nested_flags = rec->ExceptionFlags; s->nested_code = rec->ExceptionCode;
        InterlockedIncrement(&s->nested_calls);
        return ExceptionContinueExecution;     /* the nested RaiseException returns */
    }
    if (!xh_mine(s, rec, c)) return ExceptionContinueSearch;
    InterlockedIncrement(&s->seh_calls);
    s->veh_calls_at_seh = s->veh_calls;
    s->est_frame = frame;
#ifdef T_X64
    DISPATCHER_CONTEXT *dc = (DISPATCHER_CONTEXT *)dcv;
    s->dc_valid = 1;
    s->dc_pc = (uintptr_t)dc->ControlPc; s->dc_imagebase = (uintptr_t)dc->ImageBase;
    s->dc_begin = dc->FunctionEntry ? (uintptr_t)dc->ImageBase + dc->FunctionEntry->BeginAddress : 0;
    s->dc_frame = (uintptr_t)dc->EstablisherFrame;
    if (dc->ContextRecord) {
        for (int i = 6; i < 16; i++) memcpy(s->dc_xmm[i], &dc->ContextRecord->FltSave.XmmRegisters[i], 16);
        for (int i = 0; i < 16; i++) s->dc_gpr[i] = (&dc->ContextRecord->Rax)[i];
    }
#else
    (void)dcv;
#endif
    if (s->seh_action == 3) {          /* raise a nested exception from inside the frame handler */
        ULONG_PTR a = 7;
        RaiseException(0xE0000ABDu, 0, 1, &a);
    }
#ifdef T_X64
    if (s->seh_action == 2) {          /* unwind to the resume label of the establisher frame */
        s->rec = *rec;
        RtlUnwindEx(frame, (PVOID)s->resume, rec, (PVOID)s->retval, dc->ContextRecord, dc->HistoryTable);
        CHECKF(0, "RtlUnwindEx returned");
        return ExceptionContinueSearch;
    }
#endif
    if (s->seh_action == 1) return ExceptionContinueSearch;
    return xh_core(s, rec, c, "SEH") ? ExceptionContinueExecution : ExceptionContinueSearch;
}

/* ========================================================================================== */
/* Kind defaults                                                                               */
/* ========================================================================================== */
static uint8_t *g_nx_page;
static T_ALIGN(64) uint8_t g_mis_buf[128];

static int xk_is_x87(const xk *k) { return !strncmp(k->name, "x87", 3); }
static int xk_is_sse(const xk *k) { return !strncmp(k->name, "sse_", 4); }

/* Operands + expectations for a kind (arch-specific values pinned in the header comment). */
static void xk_prepare(const xk *k, xf *f, xexp *e) {
    const char *n = k->name;
    memset(e, 0, sizeof *e);
    e->nparams = -1; e->resume = RS_SKIP;
    if (!strcmp(n, "nop")) { e->code = 0; }
    else if (!strcmp(n, "div0") || !strcmp(n, "div64")) {
        f->gin[0] = 12345; f->gin[2] = 0; f->gin[1] = 0;
        e->code = XS_INT_DIVIDE_BY_ZERO; e->nparams = 0;
    } else if (!strcmp(n, "idivov")) {
        f->gin[0] = 0x80000000u; f->gin[2] = 0xFFFFFFFFu; f->gin[1] = 0xFFFFFFFFu;
        e->code = XS_INT_OVERFLOW; e->alt[0] = XS_INT_DIVIDE_BY_ZERO;
    } else if (!strcmp(n, "int3")) {
        e->code = XS_BREAKPOINT;
#ifdef T_X64
        e->nparams = 1; e->ninfo = 1; e->info[0] = 0;
#endif
    } else if (!strcmp(n, "int3cd")) {
        e->code = XS_BREAKPOINT; e->addr_soft = 1; e->pc_soft = 1;
    } else if (!strcmp(n, "rd") || !strcmp(n, "wr")) {
        f->gin[1] = 0x10;
        e->code = XS_ACCESS_VIOLATION; e->nparams = 2; e->ninfo = 2; e->info[0] = n[0] == 'w'; e->info[1] = 0x10;
    } else if (!strcmp(n, "exec")) {
        if (!g_nx_page) {
            g_nx_page = (uint8_t *)VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (g_nx_page) g_nx_page[0] = 0xC3;      /* RET: harmless if DEP is not enforced */
        }
        f->gin[1] = (uintptr_t)g_nx_page;
        e->code = XS_ACCESS_VIOLATION; e->nparams = 2; e->ninfo = 2; e->info[0] = 8; e->info[1] = (uintptr_t)g_nx_page;
        e->addr = g_nx_page; e->pc = g_nx_page; e->sp_delta = -PTRSZ; e->resume = RS_RET;
    } else if (!strcmp(n, "ss")) {
        e->code = XS_SINGLE_STEP;
    } else if (!strcmp(n, "ud2")) {
        e->code = XS_ILLEGAL_INSTRUCTION;
    } else if (!strcmp(n, "hlt") || !strcmp(n, "cli") || !strcmp(n, "sti")) {
        e->code = XS_PRIVILEGED;
    } else if (!strcmp(n, "inb") || !strcmp(n, "outb") || !strcmp(n, "wbinvd") || !strcmp(n, "rdmsr") || !strcmp(n, "wrmsr") ||
               !strcmp(n, "movcr") || !strcmp(n, "clts") || !strcmp(n, "invd") || !strcmp(n, "xsetbv")) {
        f->gin[1] = 0; f->gin[2] = 0x80;        /* rdmsr/wrmsr ecx = 0; in/out port 0x80; xsetbv ecx = 0 */
        f->gin[0] = 3;                          /* xsetbv: XCR0 = x87|SSE (never reached at CPL3) */
        if (!strcmp(n, "xsetbv")) f->gin[2] = 0;
        e->code = XS_PRIVILEGED; e->alt[0] = XS_ACCESS_VIOLATION;
    } else if (!strcmp(n, "int2c")) {
        e->code = XS_ASSERTION_FAILURE; e->addr_soft = 1; e->pc_soft = 1;
    } else if (!strcmp(n, "fastfail")) {
        f->gin[1] = 7;                          /* FAST_FAIL_FATAL_APP_EXIT */
        e->code = XS_STACK_BUFFER_OVERRUN;
    } else if (!strcmp(n, "movaps")) {
        f->gin[1] = (uintptr_t)(g_mis_buf + 4);
        e->code = XS_ACCESS_VIOLATION; e->alt[0] = XS_MISALIGNMENT;
    } else if (!strcmp(n, "into")) {
        f->flags |= XF_EFL; f->efl_in = 0x802;  /* OF */
        e->code = XS_INT_OVERFLOW; e->addr_soft = 1; e->pc_soft = 1;
    } else if (!strcmp(n, "bound")) {
        static int32_t bounds[2] = { 0, 10 };
        f->gin[0] = 100; f->gin[1] = (uintptr_t)bounds;
        e->code = XS_ARRAY_BOUNDS;
    } else if (xk_is_sse(k)) {
        /* default per routine; tests refine with sse_setup() */
        e->code = 0;
    } else if (xk_is_x87(k)) {
        e->code = 0;
    }
    xf_sync_exp(f);
}

typedef enum { FPE_IE, FPE_DE, FPE_ZE, FPE_OE, FPE_UE, FPE_PE } fpe_t;
static const char *const g_fpe_name[] = { "IE", "DE", "ZE", "OE", "UE", "PE" };

/* SSE unmasked-exception scenario: operands, the kind to use, expected code / MXCSR / masked result */
static const xk *sse_setup(fpe_t which, xf *f, xexp *e, int unmasked, uint32_t *masked_result) {
    const char *kn = which == FPE_ZE ? "sse_div" : which == FPE_IE ? "sse_sqrt" :
                     (which == FPE_OE || which == FPE_UE) ? "sse_mul" : "sse_add";
    const xk *k = xk_find(kn);
    xk_prepare(k, f, e);
    uint32_t a = 0, b = 0, r = 0, any = 0;
    DWORD code64 = 0, code32 = 0;
    switch (which) {
    case FPE_ZE: a = f32bits(1.0f); b = 0; r = 0x7F800000u; code64 = XS_FLT_DIVIDE_BY_ZERO; code32 = XS_FLT_MULTIPLE_TRAPS; break;
    case FPE_IE: a = f32bits(2.0f); b = f32bits(-1.0f); r = 0xFFC00000u; code64 = XS_FLT_INVALID; code32 = XS_FLT_MULTIPLE_TRAPS; break;
    case FPE_DE: a = f32bits(1.0f); b = 0x00000001u; r = f32bits(1.0f); any = 0x20; code64 = XS_FLT_INVALID; code32 = XS_FLT_MULTIPLE_TRAPS; break;
    case FPE_OE: a = f32bits(1e38f); b = f32bits(1e38f); r = 0x7F800000u; any = 0x20; code64 = XS_FLT_OVERFLOW; code32 = XS_FLT_MULTIPLE_FAULTS; break;
    case FPE_UE: a = f32bits(1e-30f); b = f32bits(1e-30f); r = 0; any = 0x20; code64 = XS_FLT_UNDERFLOW; code32 = XS_FLT_MULTIPLE_FAULTS; break;
    case FPE_PE: a = f32bits(1.0f); b = f32bits(1e-10f); r = f32bits(1.0f); code64 = XS_FLT_INEXACT; code32 = XS_FLT_MULTIPLE_FAULTS; break;
    }
    fill_ps(f->vin[0], a); fill_ps(f->vin[1], b);
    uint32_t bit = 1u << (which == FPE_IE ? 0 : which == FPE_DE ? 1 : which == FPE_ZE ? 2 : which == FPE_OE ? 3 : which == FPE_UE ? 4 : 5);
    f->mxcsr_in = unmasked ? (0x1F80u & ~(bit << 7)) : 0x1F80u;
    xf_sync_exp(f);
    /* after the (re)executed masked operation */
    fill_ps(f->vexp[0], r);
    f->mxexp = 0x1F80u | bit; f->mx_any_out = any;
    if (unmasked) {
#ifdef T_X64
        e->code = code64;
        if (which == FPE_DE) e->alt[0] = XS_FLT_DENORMAL;
        e->nparams = 2; e->ninfo = 1; e->info[0] = 0;
#else
        e->code = code32; e->alt[0] = code64; e->alt[1] = which == FPE_DE ? XS_FLT_DENORMAL : 0;
#endif
        e->mx_set = bit; e->mx_any = any; e->resume = RS_STAY;
        (void)code32; (void)code64;
    }
    if (masked_result) *masked_result = r;
    return k;
}

/* x87 unmasked scenario: "div" (ZE) or "ovf" (stack overflow -> IE|SF) */
static const xk *x87_setup(int ovf, xf *f, xexp *e, int unmasked) {
    const xk *k = xk_find(ovf ? "x87_ovf" : "x87_div");
    xk_prepare(k, f, e);
    f->flags |= XF_X87;
    f80_from_double(f->x87in[0], 1.0);
    f80_from_double(f->x87in[1], ovf ? 3.0 : 0.0);
    f->fcw_in = unmasked ? (uint16_t)(0x027F & ~(ovf ? 0x01 : 0x04)) : 0x027F;
    xf_sync_exp(f);
    if (unmasked) {
        e->code = ovf ? XS_FLT_STACK_CHECK : XS_FLT_DIVIDE_BY_ZERO;
#ifdef T_X64
        e->nparams = 1; e->ninfo = 1; e->info[0] = 0;
#else
        e->alt[0] = XS_FLT_MULTIPLE_TRAPS; e->alt[1] = XS_FLT_MULTIPLE_FAULTS;
        if (ovf) e->alt[2] = XS_FLT_INVALID;
#endif
        e->x87_sw_set = ovf ? 0x00C1 : 0x0084;      /* IE|SF|ES  or  ZE|ES */
#ifdef T_X86
        e->addr_soft = 1;                           /* checked below: the waiting instruction or the x87 op */
#endif
        e->check_fip = 1;
        e->resume = RS_STAY;
    } else if (!ovf) {
        /* masked: ST0 = 1/0 = +inf */
        static const uint8_t inf80[10] = { 0, 0, 0, 0, 0, 0, 0, 0x80, 0xFF, 0x7F };
        memcpy(f->x87exp[0], inf80, 10);
    }
    return k;
}

/* ========================================================================================== */
/* Running a kind                                                                              */
/* ========================================================================================== */
static xh g_main_xh;
static int g_in_child;        /* child processes run scenarios on their main thread */
/* Wine-only known gap: turn this case's failures into XFAIL with a reason (never used on Windows). */
static void wine_gap(const char *why) { if (t_is_wine()) t_expect_fail("Wine gap: %s", why); }

static int xk_arm_and_run(xh *s, const xk *k, xf *f, int variant) {
    s->k = k; s->f = f;
    int seh = variant == V_SEHFRAME && k->sfn;
    s->op = seh ? k->sop : k->op; s->at = seh ? k->sat : k->at; s->resume = seh ? k->sresume : k->resume;
    s->calls = s->veh_calls = s->seh_calls = s->unwind_calls = s->nested_calls = 0;
    s->dc_valid = 0; s->had_xstate = 0; s->xmask = 0; s->unwind_flags = 0; s->nested_flags = 0;
    if (variant != V_VEH && variant != V_NONE && !s->sehfn) s->sehfn = seh_kind;
    TlsSetValue(g_tls, s);
    s->armed = 1;
    if (variant == V_SEHFRAME && k->sfn) k->sfn(f);
    else if (variant == V_VEH || variant == V_NONE) k->fn(f);
    else xseh_call((void *(*)(void *))(void *)k->fn, f);   /* handler in the caller's frame */
    s->armed = 0;
    return (int)s->calls;
}

static int vk_supported(int vk) {
    if (vk == 32) return cpu_has(F_AVX);
    if (vk == 64) return cpu_has(F_AVX512F);
    return 1;
}
static const char *vk_name(int vk) { return vk == 64 ? "zmm" : vk == 32 ? "ymm" : "xmm"; }

/* The standard scenario: prepare, run, check calls / after-resume state. */
typedef void (*xk_tweak)(xf *, xexp *, xh *);
static int xk_test(const char *kname, int vk, int mode, int variant, xk_tweak tweak, const char *label) {
    xs_init();
    const xk *k = xk_find(kname);
    if (!k) { CHECKF(0, "unknown kind %s", kname); return -1; }
    xf *f = xf_alloc();
    if (!f) { CHECKF(0, "VirtualAlloc for the reference frame failed"); return -1; }
    xh *s = (t_on_case_thread() || g_in_child) ? &g_main_xh : (xh *)calloc(1, sizeof(xh));
    if (!s) { xf_free(f); return -1; }
    memset(s, 0, sizeof *s);
    uint64_t hs = 1469598103934665603ull;
    for (const char *q = kname; *q; q++) hs = (hs ^ (uint8_t)*q) * 1099511628211ull;
    uint32_t tag = (uint32_t)(t_mix64(hs ^ ((uint64_t)mode << 20) ^ (uint64_t)vk ^ ((uint64_t)variant << 40)) & 0xFFFFu) + 1u;
    xf_init(f, tag, vk, 0, 0);
    xk_prepare(k, f, &s->e);
    s->mode = mode; s->tag2 = tag + 0x700u; s->veh_on = variant == V_VEH;
    if (tweak) tweak(f, &s->e, s);
    int calls = xk_arm_and_run(s, k, f, variant);
    if (s->e.code) CHECKF(calls == 1, "%s: handler ran %d times, expected exactly 1", label, calls);
    else CHECKF(calls == 0, "%s: unexpected exception (handler ran %d times, code 0x%08lx)", label, calls,
                (unsigned long)s->rec.ExceptionCode);
    xf_check_out(f, label);
    if (s != &g_main_xh) free(s);
    xf_free(f);
    return calls;
}

/* M_NESTED_HW: from inside the outer handler, run a second kind routine (own patterns in every vector register,
 * GPRs, MXCSR) that takes a hardware access violation, handled by the vectored handler with an XMM edit. The nested
 * dispatch builds its own CONTEXT/XSTATE frame below the outer one; afterwards the outer CONTEXT must be unchanged
 * (re-verified) and the outer resume must still restore the outer state. */
static void xh_nested_hw(xh *s, EXCEPTION_RECORD *rec, CONTEXT *c, const char *who) {
    xf *g = xf_alloc();
    xh *t = (xh *)calloc(1, sizeof(xh));
    if (!g || !t) { CHECKF(0, "%s: nested scenario allocation failed", who); xf_free(g); free(t); return; }
    const xk *k = xk_find("rd");
    xf_init(g, (s->tag2 ^ 0x3C3Cu) | 1u, vk_of(s->f), 0, 0);
    xk_prepare(k, g, &t->e);
    t->veh_on = 1; t->quiet = 1; t->mode = M_MOD_XMM; t->tag2 = s->tag2 + 0x99u;
    int calls = xk_arm_and_run(t, k, g, V_VEH);
    TlsSetValue(g_tls, s);
    s->nested_hw_calls = calls;
    CHECKF(calls == 1, "%s: nested hardware fault inside the handler: inner handler ran %d times", who, calls);
    CHECKF(t->rec.ExceptionCode == XS_ACCESS_VIOLATION, "%s: nested fault code 0x%08lx", who, (unsigned long)t->rec.ExceptionCode);
    xf_check_out(g, "nested fault raised inside the handler");
    char w2[64]; snprintf(w2, sizeof w2, "%s after a nested fault", who);
    int q = s->quiet; s->quiet = 1;
    xh_verify(s, rec, c, w2);
    s->quiet = q;
    xf_free(g); free(t);
}

/* ---- child-process scenarios ---------------------------------------------------------------- */
/* Top-level filter for V_NONE: reached through RtlUserThreadStart's filter -> UnhandledExceptionFilter ->
 * SetUnhandledExceptionFilter callback. Returning EXCEPTION_CONTINUE_EXECUTION resumes with the (edited) CONTEXT
 * (documented UnhandledExceptionFilter contract). Anything else terminates like the harness child filter. */
static LONG WINAPI uef_core(EXCEPTION_POINTERS *ep) {
    xh *s = xh_cur();
    if (s && s->armed && xh_mine(s, ep->ExceptionRecord, ep->ContextRecord) && xh_core(s, ep->ExceptionRecord, ep->ContextRecord, "UEF"))
        return EXCEPTION_CONTINUE_EXECUTION;
    DWORD c = ep->ExceptionRecord->ExceptionCode;
    printf("T_CHILD_EXCEPTION code=0x%08lx addr=0x%llx\n", (unsigned long)c, (unsigned long long)(ULONG_PTR)ep->ExceptionRecord->ExceptionAddress);
    fflush(stdout);
    TerminateProcess(GetCurrentProcess(), c);
    return EXCEPTION_CONTINUE_SEARCH;
}

/* argv: xk <kind> <vk> <mode> <variant> */
CHILD(xk) {
    if (argc < 5) return 3;
    g_in_child = 1;
    xs_init();
    const char *kname = argv[1];
    int vk = atoi(argv[2]), mode = atoi(argv[3]), variant = atoi(argv[4]);
    if (!vk_supported(vk)) { printf("XK_SKIP=%s not usable\n", vk_name(vk)); return 0; }
    if (variant == V_NONE) SetUnhandledExceptionFilter(uef_core);
    int calls = xk_test(kname, vk, mode, variant, NULL, kname);
    printf("XK_CALLS=%d CODE=0x%08lx\n", calls, (unsigned long)g_main_xh.rec.ExceptionCode);
    fflush(stdout);
    return 0;
}

static void child_forward(const t_child_result *r, const char *what) {
    const char *p = r->out;
    while (p && *p) {
        const char *e = p; while (*e && *e != '\n' && *e != '\r') e++;
        int len = (int)(e - p);
        if (len > 7 && !strncmp(p, "  FAIL ", 7)) t_check(0, __FILE__, __LINE__, what, "child: %.*s", len - 7, p + 7);
        else if (len > 7 && (!strncmp(p, "  NOTE ", 7) || !strncmp(p, "  INFO ", 7))) t_note("child %s: %.*s", what, len - 7, p + 7);
        p = e; while (*p == '\n' || *p == '\r') p++;
    }
}

static t_child_result g_cr;
static int run_child_kind(const char *kind, int vk, int mode, int variant, DWORD *code_out) {
    char a[160]; snprintf(a, sizeof a, "%s %d %d %d", kind, vk, mode, variant);
    if (!t_child_run("xk", a, 20000, &g_cr) || !g_cr.started) {
        CHECKF(0, "could not start child for %s (error %lu)", kind, (unsigned long)g_cr.spawn_error);
        return -1;
    }
    CHECKF(!g_cr.timed_out, "child %s timed out", kind);
    child_forward(&g_cr, kind);
    char v[96];
    if (t_child_find(&g_cr, "XK_SKIP=", v, sizeof v)) SKIP("%s", v);
    int calls = -1; unsigned long code = 0;
    if (t_child_find(&g_cr, "XK_CALLS=", v, sizeof v)) sscanf(v, "%d CODE=0x%lx", &calls, &code);
    CHECKF(calls >= 0, "child %s did not complete: exit 0x%08lx exception 0x%08lx at 0x%llx", kind,
           (unsigned long)g_cr.exit_code, (unsigned long)g_cr.exc_code, g_cr.exc_addr);
    CHECKF(g_cr.exit_code == 0, "child %s exit code 0x%08lx", kind, (unsigned long)g_cr.exit_code);
    if (code_out) *code_out = (DWORD)code;
    return calls;
}

/* ========================================================================================== */
/* Reference-helper sanity (no exception)                                                      */
/* ========================================================================================== */
static void tw_x87_efl(xf *f, xexp *e, xh *s) { (void)e; (void)s; f->flags |= XF_X87 | XF_EFL; xf_sync_exp(f); }

TEST(ref_nop_roundtrip_xmm) { xk_test("nop", 16, 0, V_VEH, NULL, "after nop"); }
TEST(ref_nop_roundtrip_ymm) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xk_test("nop", 32, 0, V_VEH, NULL, "after nop");
}
TEST(ref_nop_roundtrip_zmm) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    xk_test("nop", 64, 0, V_VEH, NULL, "after nop");
}
TEST(ref_nop_roundtrip_x87_eflags) { xk_test("nop", 16, 0, V_VEH, tw_x87_efl, "after nop"); }

/* ========================================================================================== */
/* Integer / breakpoint / single-step faults with full record + context verification (VEH)     */
/* ========================================================================================== */
TEST(veh_div0_record_context_xmm) { xk_test("div0", 16, 0, V_VEH, NULL, "after #DE"); }
TEST(veh_div0_record_context_ymm) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xk_test("div0", 32, 0, V_VEH, NULL, "after #DE");
}
TEST(veh_div0_record_context_zmm) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    xk_test("div0", 64, 0, V_VEH, NULL, "after #DE");
}
TEST(veh_div64_record_context) {
#ifdef T_X64
    xk_test("div64", 16, 0, V_VEH, NULL, "after #DE (64-bit div)");
#else
    SKIP("64-bit DIV exists only in 64-bit mode");
#endif
}
TEST(veh_idiv_overflow_intmin) {
    xk_test("idivov", 16, 0, V_VEH, NULL, "after #DE (INT_MIN/-1)");
    t_info("code", "0x%08lx", (unsigned long)g_main_xh.rec.ExceptionCode);
}
TEST(veh_int3_record_context) {
    xk_test("int3", 16, 0, V_VEH, NULL, "after INT3");
    if (cpu_has(F_AVX)) xk_test("int3", 32, 0, V_VEH, NULL, "after INT3 (ymm)");
}
TEST(veh_int3_cd03_form) {
    xk_test("int3cd", 16, 0, V_VEH, NULL, "after INT 3 (CD 03)");
    t_info("address_delta", "%ld", (long)((intptr_t)g_main_xh.rec.ExceptionAddress - (intptr_t)xk_find("int3cd")->at));
}
TEST(veh_single_step_tf) {
    xk_test("ss", 16, M_CLEAR_TF, V_VEH, NULL, "after single-step trap");
    if (cpu_has(F_AVX)) xk_test("ss", 32, M_CLEAR_TF | M_MOD_XMM, V_VEH, NULL, "after single-step trap (ymm, xmm edited)");
}
static void tw_efl(xf *f, xexp *e, xh *s) { (void)e; (void)s; f->flags |= XF_EFL; xf_sync_exp(f); }
TEST(veh_ctx_eflags_df_and_arith_flags) {
    /* DF=1 and CF/PF/AF/ZF/SF/OF set at the fault: the CONTEXT must carry them, resume must restore them, and
     * the handler itself must run with DF=0 (ABI; KiUserExceptionDispatcher begins with CLD). */
    xk_test("div0", 16, 0, V_VEH, tw_efl, "after #DE with DF=1");
    CHECKF((g_main_xh.live_eflags & 0x400) == 0, "handler entered with DF set (EFLAGS 0x%llx)", (unsigned long long)g_main_xh.live_eflags);
    xk_test("int3", 16, 0, V_VEH, tw_efl, "after INT3 with DF=1");
    CHECKF((g_main_xh.live_eflags & 0x400) == 0, "handler entered with DF set (EFLAGS 0x%llx)", (unsigned long long)g_main_xh.live_eflags);
}
static void tw_x87(xf *f, xexp *e, xh *s) { (void)e; (void)s; f->flags |= XF_X87; xf_sync_exp(f); }
TEST(veh_ctx_x87_stack_and_cw) {
    xk_test("div0", 16, 0, V_VEH, tw_x87, "after #DE with a full x87 stack");
    xk_test("rd", 16, 0, V_VEH, tw_x87, "after #PF with a full x87 stack");
}

/* ========================================================================================== */
/* Memory faults                                                                               */
/* ========================================================================================== */
TEST(veh_av_read_null) { xk_test("rd", 16, 0, V_VEH, NULL, "after read AV"); }
TEST(veh_av_write_null) { xk_test("wr", 16, 0, V_VEH, NULL, "after write AV"); }
TEST(veh_av_read_null_ymm) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xk_test("rd", 32, 0, V_VEH, NULL, "after read AV");
    xk_test("wr", 32, 0, V_VEH, NULL, "after write AV");
}
static uint8_t *g_tw_addr;
static void tw_addr_rd(xf *f, xexp *e, xh *s) {
    (void)s; f->gin[1] = (uintptr_t)g_tw_addr; xf_sync_exp(f); e->info[1] = (uintptr_t)g_tw_addr;
}
static void tw_addr_wr(xf *f, xexp *e, xh *s) { tw_addr_rd(f, e, s); }
static void tw_straddle(xf *f, xexp *e, xh *s) { tw_addr_rd(f, e, s); e->ninfo = 1; }
TEST(veh_av_read_guard_page_end) {
    void *base = NULL;
    uint8_t *p = (uint8_t *)t_guarded_alloc(64, &base);
    if (!p) SKIP("t_guarded_alloc failed");
    g_tw_addr = p + 64;                         /* first byte of the PAGE_NOACCESS page */
    xk_test("rd", 16, 0, V_VEH, tw_addr_rd, "after AV at the end of a committed range");
    g_tw_addr = p + 64 - PTRSZ / 2;             /* load straddling into the no-access page */
    xk_test("rd", 16, 0, V_VEH, tw_straddle, "after page-straddling AV");
    t_info("straddle_info1", "0x%llx (load at p+60, page at p+64)", (unsigned long long)g_main_xh.rec.ExceptionInformation[1]);
    t_guarded_free(base);
}
TEST(veh_av_write_readonly) {
    uint8_t *p = (uint8_t *)VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY);
    if (!p) SKIP("VirtualAlloc(PAGE_READONLY) failed");
    g_tw_addr = p + 128;
    xk_test("wr", 16, 0, V_VEH, tw_addr_wr, "after write AV on a read-only page");
    CHECK_EQ(p[128], 0);
    VirtualFree(p, 0, MEM_RELEASE);
}
static void tw_noncanon(xf *f, xexp *e, xh *s) {
    (void)s; f->gin[1] = 0x8000000000000000ull; xf_sync_exp(f);
    e->info[1] = (ULONG_PTR)~(ULONG_PTR)0;     /* KiPreprocessFault: #GP -> AV {0, -1} */
}
TEST(veh_av_noncanonical_x64) {
#ifdef T_X64
    xk_test("rd", 16, 0, V_VEH, tw_noncanon, "after #GP (non-canonical address)");
#else
    SKIP("non-canonical addresses exist only in 64-bit mode");
#endif
}
TEST(veh_av_execute_nx) {
    xs_init();
    const xk *k = xk_find("exec");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x3301, 16, 0, 0);
    xk_prepare(k, f, &s->e);
    REQUIREF(g_nx_page, "VirtualAlloc failed");
    s->veh_on = 1;
    int calls = xk_arm_and_run(s, k, f, V_VEH);
    if (calls == 0) {
#ifdef T_X64
        CHECKF(0, "executing a PAGE_READWRITE page did not fault (DEP is always on for 64-bit processes)");
#else
        DWORD fl = 0; BOOL perm = FALSE;
        BOOL (WINAPI *gdp)(HANDLE, LPDWORD, PBOOL) = (BOOL (WINAPI *)(HANDLE, LPDWORD, PBOOL))t_proc("kernel32.dll", "GetProcessDEPPolicy");
        if (gdp) gdp(GetCurrentProcess(), &fl, &perm);
        t_note("DEP not enforced for this 32-bit process (policy flags 0x%lx permanent %d): no execute fault", (unsigned long)fl, perm);
#endif
    } else CHECK_EQ(calls, 1);
    xf_check_out(f, "after execute AV");
    xf_free(f);
}
TEST(veh_movaps_misaligned) {
    xk_test("movaps", 16, 0, V_VEH, NULL, "after misaligned MOVAPS");
    t_info("code", "0x%08lx np=%lu info1=0x%llx", (unsigned long)g_main_xh.rec.ExceptionCode, (unsigned long)g_main_xh.rec.NumberParameters,
           (unsigned long long)g_main_xh.rec.ExceptionInformation[1]);
}
static void tw_guard_rd(xf *f, xexp *e, xh *s) { tw_addr_rd(f, e, s); e->code = XS_GUARD_PAGE; }
static void guard_case(int write) {
    uint8_t *p = (uint8_t *)VirtualAlloc(NULL, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) SKIP("VirtualAlloc failed");
    memset(p, 0x5C, 8192);
    DWORD old;
    REQUIRE(VirtualProtect(p, 4096, PAGE_READWRITE | PAGE_GUARD, &old));
    g_tw_addr = p + 256;
    /* first access: STATUS_GUARD_PAGE_VIOLATION, {r/w, address}; the guard is removed */
    xk_test(write ? "wr" : "rd", 16, 0, V_VEH, tw_guard_rd, "after guard-page fault");
    CHECKF(g_main_xh.rec.ExceptionCode == XS_GUARD_PAGE, "code 0x%08lx want STATUS_GUARD_PAGE_VIOLATION",
           (unsigned long)g_main_xh.rec.ExceptionCode);
    MEMORY_BASIC_INFORMATION mbi;
    CHECK(VirtualQuery(p, &mbi, sizeof mbi) == sizeof mbi);
    CHECKF((mbi.Protect & PAGE_GUARD) == 0, "PAGE_GUARD still set after the one-shot guard exception (Protect 0x%lx)", (unsigned long)mbi.Protect);
    CHECK_EQ(mbi.Protect & 0xFF, PAGE_READWRITE);
    /* second access succeeds */
    xs_init();
    const xk *k = xk_find(write ? "wr" : "rd");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x4401, 16, 0, 0);
    xk_prepare(k, f, &s->e);
    f->gin[1] = (uintptr_t)(p + 256);
    xf_sync_exp(f);
    if (!write) memcpy(&f->gexp[0], p + 256, sizeof(uintptr_t));
    s->veh_on = 1;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_VEH), 0);
    xf_check_out(f, "second access to the former guard page");
    if (write) CHECK(!memcmp(p + 256, &f->gin[0], PTRSZ));
    xf_free(f);
    VirtualFree(p, 0, MEM_RELEASE);
}
TEST(veh_guard_page_read_once) { guard_case(0); }
TEST(veh_guard_page_write_once) { guard_case(1); }

/* ========================================================================================== */
/* CONTEXT edits take effect / unchanged CONTEXT preserves everything                         */
/* ========================================================================================== */
TEST(veh_ctx_modify_xmm) {
    xk_test("div0", 16, M_MOD_XMM, V_VEH, NULL, "after resume with XMM edited in CONTEXT");
    xk_test("rd", 16, M_MOD_XMM, V_VEH, NULL, "after resume with XMM edited in CONTEXT");
}
TEST(veh_ctx_modify_gpr) {
    xk_test("div0", 16, M_MOD_GPR, V_VEH, NULL, "after resume with GPRs edited in CONTEXT");
}
TEST(veh_ctx_modify_mxcsr) {
    xk_test("div0", 16, M_MOD_MXCSR, V_VEH, NULL, "after resume with MXCSR edited in CONTEXT");
}
TEST(veh_ctx_modify_mxcsr_toplevel_only_x64) {
#ifdef T_X64
    /* CONTEXT has two MXCSR copies (MxCsr, FltSave.MxCsr). Record which one resume honours. */
    xs_init();
    const xk *k = xk_find("div0");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x5501, 16, 0, 0); xk_prepare(k, f, &s->e);
    s->mode = M_MOD_MXCSR_TOP; s->veh_on = 1;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_VEH), 1);
    t_info("mxcsr_after", "0x%x (CONTEXT.MxCsr=0x5F80, FltSave.MxCsr=0x%x)", f->mxcsr_out, f->mxcsr_in);
    CHECKF(f->mxcsr_out == 0x5F80 || f->mxcsr_out == f->mxcsr_in, "MXCSR after resume 0x%x is neither copy", f->mxcsr_out);
    xf_free(f);
#else
    SKIP("x64 CONTEXT only");
#endif
}
TEST(veh_ctx_modify_ymm_upper) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xk_test("div0", 32, M_MOD_YMM, V_VEH, NULL, "after resume with YMM uppers edited via LocateXStateFeature");
    if (!g_main_xh.had_xstate) {
#ifdef T_X86
        SKIP("32-bit exception CONTEXT carries no XSTATE: cannot edit YMM through it");
#endif
    }
}
TEST(veh_ctx_modify_ymm_and_xmm) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xk_test("rd", 32, M_MOD_YMM | M_MOD_XMM | M_MOD_GPR, V_VEH, NULL, "after resume with XMM+YMM+GPR edited");
}
TEST(veh_ctx_modify_zmm_upper) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    xk_test("div0", 64, M_MOD_ZMM, V_VEH, NULL, "after resume with ZMM_Hi256 edited");
}
TEST(veh_ctx_modify_x87_st0) {
    xk_test("div0", 16, M_MOD_X87, V_VEH, tw_x87, "after resume with ST(0) edited in CONTEXT");
}
TEST(veh_ctx_clear_avx_mask_note) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xs_init();
    const xk *k = xk_find("div0");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x5601, 32, 0, 0); xk_prepare(k, f, &s->e);
    s->mode = M_CLEAR_AVX; s->veh_on = 1;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_VEH), 1);
    int zero = 1, same = 1;
    for (int i = 0; i < T_NXMM; i++) {
        for (int j = 16; j < 32; j++) { if (f->vout[i][j]) zero = 0; if (f->vout[i][j] != f->vin[i][j]) same = 0; }
        CHECK_REG("xmm low half after clearing the AVX mask bit", f->vout[i], f->vin[i], 16);
    }
    t_info("ymm_upper_after_clear_mask", "%s", zero ? "initialized (zero)" : same ? "unchanged" : "other");
    xf_free(f);
}
static void tw_quiet(xf *f, xexp *e, xh *s) { (void)f; (void)e; s->quiet = 1; }
TEST(veh_ctx_unchanged_all_kinds) {
    static const char *const kinds[] = { "div0", "idivov", "int3", "rd", "wr", "ss", "movaps" };
    for (size_t i = 0; i < T_COUNTOF(kinds); i++) {
        int mode = !strcmp(kinds[i], "ss") ? M_CLEAR_TF : 0;
        char lab[64]; snprintf(lab, sizeof lab, "after %s (unchanged CONTEXT)", kinds[i]);
        xk_test(kinds[i], cpu_has(F_AVX) ? 32 : 16, mode, V_VEH, tw_quiet, lab);
    }
}

/* Handler clobbers the volatile registers in its own code: resume must restore from the CONTEXT. */
TEST(veh_handler_clobbers_volatile_xmm) {
    xk_test("div0", 16, M_CLOBBER, V_VEH, NULL, "after a handler that clobbered XMM0-5 and MXCSR flags");
}
TEST(veh_handler_clobbers_ymm_upper) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xk_test("div0", 32, M_CLOBBER, V_VEH, NULL, "after a handler that clobbered every YMM upper half");
    xk_test("rd", 32, M_CLOBBER | M_MOD_XMM, V_VEH, NULL, "after clobber + XMM edit");
}
TEST(veh_handler_clobbers_zmm) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    xk_test("div0", 64, M_CLOBBER, V_VEH, NULL, "after a handler that clobbered ZMM/k state");
}
TEST(veh_handler_live_fp_environment) {
    xk_test("div0", 16, 0, V_VEH, tw_x87, "after #DE");
    t_info("handler_mxcsr", "0x%x (thread at fault: 0xbf80)", g_main_xh.live_mxcsr);
    t_info("handler_x87_cw", "0x%04x (thread at fault: 0x0e7f)", g_main_xh.live_fcw);
}

/* ========================================================================================== */
/* FP exceptions                                                                               */
/* ========================================================================================== */
static void fp_sse_case(fpe_t which, int vk) {
#ifdef T_X86
    if (which == FPE_OE || which == FPE_UE || which == FPE_PE)
        wine_gap("32-bit SSE #XM for OE/UE/PE is reported as STATUS_FLOAT_MULTIPLE_TRAPS (Windows: _MULTIPLE_FAULTS)");
#endif
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6000u + (uint32_t)which * 16u + (uint32_t)vk, vk, 0, 0);
    uint32_t r = 0;
    const xk *k = sse_setup(which, f, &s->e, 1, &r);
    s->mode = M_FPFIX; s->veh_on = 1;
    char lab[64]; snprintf(lab, sizeof lab, "after unmasked SSE %s, masked in CONTEXT, re-executed", g_fpe_name[which]);
    int calls = xk_arm_and_run(s, k, f, V_VEH);
    CHECKF(calls == 1, "%s: handler ran %d times", lab, calls);
    t_info("code", "0x%08lx np=%lu info0=0x%llx info1=0x%llx", (unsigned long)s->rec.ExceptionCode, (unsigned long)s->rec.NumberParameters,
           (unsigned long long)s->rec.ExceptionInformation[0], (unsigned long long)s->rec.ExceptionInformation[1]);
#ifdef T_X64
    if (s->rec.NumberParameters >= 2)
        CHECKF(((s->rec.ExceptionInformation[1] ^ (f->mxcsr_in | s->e.mx_set)) & 0xFFDFu) == 0,
               "ExceptionInformation[1] 0x%llx is not the faulting MXCSR 0x%x", (unsigned long long)s->rec.ExceptionInformation[1],
               f->mxcsr_in | s->e.mx_set);
#endif
    xf_check_out(f, lab);
    xf_free(f);
}
TEST(fp_sse_zero_divide_unmasked) { fp_sse_case(FPE_ZE, 16); }
TEST(fp_sse_invalid_unmasked) { fp_sse_case(FPE_IE, 16); }
TEST(fp_sse_denormal_unmasked) {
#ifdef T_X64
    wine_gap("SSE #XM with only DE unmasked is reported as STATUS_FLOAT_UNDERFLOW (Linux maps DE to FPE_FLTUND)");
#endif
    fp_sse_case(FPE_DE, 16);
}
TEST(fp_sse_overflow_unmasked) { fp_sse_case(FPE_OE, 16); }
TEST(fp_sse_underflow_unmasked) { fp_sse_case(FPE_UE, 16); }
TEST(fp_sse_inexact_unmasked) { fp_sse_case(FPE_PE, 16); }
TEST(fp_sse_zero_divide_unmasked_ymm) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    fp_sse_case(FPE_ZE, 32);
}
TEST(fp_sse_masked_no_exception) {
    xs_init();
    for (int w = FPE_IE; w <= FPE_PE; w++) {
        xf *f = xf_alloc(); REQUIRE(f);
        xh *s = &g_main_xh; memset(s, 0, sizeof *s);
        xf_init(f, 0x6100u + (uint32_t)w, 16, 0, 0);
        const xk *k = sse_setup((fpe_t)w, f, &s->e, 0, NULL);
        s->veh_on = 1;
        char lab[64]; snprintf(lab, sizeof lab, "after masked SSE %s", g_fpe_name[w]);
        CHECKF(xk_arm_and_run(s, k, f, V_VEH) == 0, "%s raised an exception", lab);
        xf_check_out(f, lab);
        xf_free(f);
    }
}
static void fp_x87_case(int ovf) {
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6200u + (uint32_t)ovf, 16, 1, 0);
    const xk *k = x87_setup(ovf, f, &s->e, 1);
    s->mode = M_FPFIX; s->veh_on = 1;
    const char *lab = ovf ? "after unmasked x87 stack overflow (status cleared in CONTEXT)" :
                            "after unmasked x87 #Z (status cleared in CONTEXT)";
    CHECKF(xk_arm_and_run(s, k, f, V_VEH) == 1, "%s: handler ran %ld times", lab, (long)s->calls);
    t_info("code", "0x%08lx np=%lu", (unsigned long)s->rec.ExceptionCode, (unsigned long)s->rec.NumberParameters);
#ifdef T_X86
    CHECKF(s->rec.ExceptionAddress == (PVOID)k->at || s->rec.ExceptionAddress == (PVOID)k->op,
           "x87 exception address %p is neither the waiting instruction %p nor the x87 op %p", s->rec.ExceptionAddress,
           (const void *)k->at, (const void *)k->op);
    t_info("address", "%s", s->rec.ExceptionAddress == (PVOID)k->at ? "waiting instruction" : "x87 op (FIP)");
#endif
    xf_check_out(f, lab);
    CHECKF((f->fsw_out & 0x80BF) == 0, "x87 SW after resume 0x%04x still has exception/ES bits", f->fsw_out);
    xf_free(f);
}
TEST(fp_x87_zero_divide_unmasked) {
#ifdef T_X64
    wine_gap("x87 #MF exception record has NumberParameters=0 (Windows x64 KiFloatingErrorFault: 1)");
#endif
    fp_x87_case(0);
}
TEST(fp_x87_stack_overflow_unmasked) {
#ifdef T_X64
    wine_gap("x87 #MF exception record has NumberParameters=0 (Windows x64 KiFloatingErrorFault: 1)");
#endif
    fp_x87_case(1);
}
TEST(fp_x87_masked_no_exception) {
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6300, 16, 1, 0);
    const xk *k = x87_setup(0, f, &s->e, 0);
    s->veh_on = 1;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_VEH), 0);
    xf_check_out(f, "after masked x87 1/0");
    CHECKF((f->fsw_out & 0x04) == 0x04, "x87 ZE flag not set after masked 1/0 (SW 0x%04x)", f->fsw_out);
    xf_free(f);
}
TEST(fp_sse_unmasked_seh_frame) {
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6400, cpu_has(F_AVX) ? 32 : 16, 0, 0);
    const xk *k = sse_setup(FPE_ZE, f, &s->e, 1, NULL);
    s->mode = M_FPFIX; s->veh_on = 0; s->sehfn = seh_kind;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_SEHFRAME), 1);
    CHECK_EQ(s->seh_calls, 1);
    xf_check_out(f, "after unmasked SSE #Z handled by a frame handler");
    xf_free(f);
}

/* ========================================================================================== */
/* Frame-based SEH                                                                              */
/* ========================================================================================== */
static void seh_basic(const char *kname, int vk, int mode, const char *lab) {
    xs_init();
    const xk *k = xk_find(kname);
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x7000u + (uint32_t)vk + (uint32_t)mode, vk, 0, 0);
    xk_prepare(k, f, &s->e);
    s->mode = mode; s->tag2 = 0x7777; s->sehfn = seh_kind;
    int calls = xk_arm_and_run(s, k, f, V_SEHFRAME);
    CHECKF(calls == 1 && s->seh_calls == 1 && s->veh_calls == 0, "%s: calls %d seh %ld veh %ld", lab, calls, (long)s->seh_calls, (long)s->veh_calls);
    xf_check_out(f, lab);
#ifdef T_X64
    CHECK(s->dc_valid);
    if (k->sfn) {
        CHECKF(s->dc_pc == (uintptr_t)s->at, "DISPATCHER_CONTEXT.ControlPc %p want the faulting instruction %p", (void *)s->dc_pc, (const void *)s->at);
        CHECKF(s->dc_begin == (uintptr_t)k->sfn, "FunctionEntry begins at %p want %p", (void *)s->dc_begin, (void *)k->sfn);
        CHECKF(s->dc_frame == (uintptr_t)f->rsp_at, "EstablisherFrame %p want the routine's RSP %p", (void *)s->dc_frame, (void *)(uintptr_t)f->rsp_at);
        CHECKF((uintptr_t)s->est_frame == (uintptr_t)f->rsp_at, "handler EstablisherFrame argument %p want %p", s->est_frame,
               (void *)(uintptr_t)f->rsp_at);
        CHECKF(s->dc_imagebase == (uintptr_t)GetModuleHandleA(NULL), "DISPATCHER_CONTEXT.ImageBase %p want %p", (void *)s->dc_imagebase,
               (void *)GetModuleHandleA(NULL));
        /* DispatcherContext->ContextRecord is the context virtually unwound past the establisher frame:
         * RSP = RSP in the body + 200 (stackalloc) + 8 pushes + return address */
        CHECKF(s->dc_gpr[4] == f->rsp_at + 272, "unwound ContextRecord Rsp 0x%llx want 0x%llx (frame popped by unwind codes)",
               (unsigned long long)s->dc_gpr[4], (unsigned long long)(f->rsp_at + 272));
    }
#endif
    xf_free(f);
}
TEST(seh_frame_handler_continue_div0) { seh_basic("div0", 16, 0, "after #DE handled by a frame handler"); }
TEST(seh_frame_handler_continue_av) { seh_basic("rd", 16, 0, "after AV handled by a frame handler"); }
TEST(seh_frame_handler_continue_int3) { seh_basic("int3", 16, 0, "after INT3 handled by a frame handler"); }
TEST(seh_frame_handler_modify_context) {
    seh_basic("div0", 16, M_MOD_XMM | M_MOD_GPR | M_MOD_MXCSR, "after frame handler edited XMM/GPR/MXCSR");
}
TEST(seh_frame_handler_ymm) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    seh_basic("div0", 32, 0, "after #DE (YMM live) handled by a frame handler");
    seh_basic("rd", 32, M_MOD_YMM | M_CLOBBER, "after frame handler edited YMM and clobbered live registers");
}
TEST(seh_frame_handler_via_caller_frame) {
    /* the faulting routine has no handler; the dispatcher must unwind it (RtlVirtualUnwind / FS:0 chain) to
     * reach the caller's handler, and continuing must still restore the faulting routine's registers */
    xs_init();
    const xk *k = xk_find("div0");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x7101, cpu_has(F_AVX) ? 32 : 16, 1, 1);
    xk_prepare(k, f, &s->e);
    s->sehfn = seh_kind;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_SEHCALL), 1);
    CHECK_EQ(s->seh_calls, 1);
#ifdef T_X64
    CHECKF(s->dc_pc == (uintptr_t)xseh_call_ret, "ControlPc for the caller frame %p want the return address %p", (void *)s->dc_pc,
           (const void *)xseh_call_ret);
    CHECKF(s->dc_begin == (uintptr_t)xseh_call, "FunctionEntry %p want xseh_call %p", (void *)s->dc_begin, (void *)xseh_call);
#endif
    xf_check_out(f, "after #DE handled by the caller's frame handler");
    xf_free(f);
}
TEST(seh_veh_runs_before_frame_handler) {
    xs_init();
    const xk *k = xk_find("rd");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x7201, 16, 0, 0); xk_prepare(k, f, &s->e);
    s->veh_on = 1; s->mode = M_VEH_SEARCH; s->sehfn = seh_kind;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_SEHFRAME), 1);
    CHECK_EQ(s->veh_calls, 1);
    CHECK_EQ(s->seh_calls, 1);
    CHECKF(s->veh_calls_at_seh == 1, "frame handler ran before the vectored handler (veh calls at that time %ld)", (long)s->veh_calls_at_seh);
    xf_check_out(f, "after VEH(search) + frame handler(continue)");
    xf_free(f);
}
TEST(seh_nested_call_flag) {
    /* raise a second exception inside the frame handler: when the nested dispatch reaches the same
     * establisher frame again, the handler must see EXCEPTION_NESTED_CALL */
    xs_init();
    const xk *k = xk_find("div0");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x7301, 16, 0, 0); xk_prepare(k, f, &s->e);
    s->sehfn = seh_kind; s->seh_action = 3;
    int calls = xk_arm_and_run(s, k, f, V_SEHCALL);
    CHECKF(calls == 1, "outer handler calls %d", calls);
    CHECKF(s->nested_calls == 1, "nested exception reached the frame handler %ld times with EXCEPTION_NESTED_CALL", (long)s->nested_calls);
    CHECKF(s->nested_code == 0xE0000ABDu, "nested code 0x%08lx", (unsigned long)s->nested_code);
    CHECKF(s->nested_flags & X_EXC_NESTED_CALL, "nested flags 0x%lx", (unsigned long)s->nested_flags);
    xf_check_out(f, "after nested exception inside the frame handler");
    xf_free(f);
}

#ifdef T_X64
static void unwind_same_frame(const char *kname, int vk, const char *lab) {
    xs_init();
    const xk *k = xk_find(kname);
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x7401u + (uint32_t)vk, vk, 0, 0); xk_prepare(k, f, &s->e);
    s->sehfn = seh_kind; s->seh_action = 2; s->retval = 0x5EC0DE5EC0DEull;
    xk_arm_and_run(s, k, f, V_SEHFRAME);
    CHECK_EQ(s->seh_calls, 1);
    CHECKF(s->unwind_calls == 1, "frame handler called %ld times during the unwind", (long)s->unwind_calls);
    CHECKF((s->unwind_flags & (X_EXC_UNWINDING | X_EXC_TARGET_UNWIND)) == (X_EXC_UNWINDING | X_EXC_TARGET_UNWIND),
           "unwind call flags 0x%lx want UNWINDING|TARGET_UNWIND", (unsigned long)s->unwind_flags);
    CHECKF((uintptr_t)f->gout[0] == s->retval, "RAX at the unwind target 0x%llx want ReturnValue 0x%llx",
           (unsigned long long)f->gout[0], (unsigned long long)s->retval);
    /* only nonvolatile state is defined after an unwind */
    f->vskip = 0x3F; f->gskip = 0x0F07u /* rax rcx rdx r8-r11 */;
    if (vk > 16) for (int i = 0; i < 16; i++) memcpy(f->vexp[i] + 16, f->vout[i] + 16, 16);  /* uppers: volatile */
    t_info("mxcsr_after_unwind", "0x%x (at fault 0x%x)", f->mxcsr_out, f->mxcsr_in);
    f->mxexp = f->mxcsr_out;
    xf_check_out(f, lab);
    xf_free(f);
}
#endif
TEST(seh_unwind_same_frame_nonvolatile_x64) {
#ifdef T_X64
    unwind_same_frame("div0", 16, "at the RtlUnwindEx target in the faulting frame");
    if (cpu_has(F_AVX)) unwind_same_frame("rd", 32, "at the RtlUnwindEx target (YMM live)");
#else
    SKIP("table-based unwinding is x64 only");
#endif
}
#ifdef T_X64
static EXCEPTION_DISPOSITION seh_outer(xh *s, EXCEPTION_RECORD *rec, void *frame, CONTEXT *c, void *dcv) {
    DISPATCHER_CONTEXT *dc = (DISPATCHER_CONTEXT *)dcv;
    if (rec->ExceptionFlags & X_EXC_UNWINDING) {
        InterlockedIncrement(&s->unwind_calls); s->unwind_flags = rec->ExceptionFlags;
        return ExceptionContinueSearch;
    }
    if (!s->armed || CPC(c) != (uintptr_t)xfi_fault) return ExceptionContinueSearch;
    InterlockedIncrement(&s->seh_calls);
    s->rec = *rec; s->ctx_pc = CPC(c);
    s->dc_valid = 1; s->dc_pc = (uintptr_t)dc->ControlPc; s->dc_frame = (uintptr_t)dc->EstablisherFrame;
    s->est_frame = frame;
    for (int i = 6; i < 16; i++) memcpy(s->dc_xmm[i], &dc->ContextRecord->FltSave.XmmRegisters[i], 16);
    for (int i = 0; i < 16; i++) s->dc_gpr[i] = (&dc->ContextRecord->Rax)[i];
    /* the original CONTEXT holds the inner frame's (B) values */
    for (int i = 6; i < 16; i++) {
        char what[64]; snprintf(what, sizeof what, "fault CONTEXT xmm%d (inner frame value)", i);
        CHECK_REG(what, &c->FltSave.XmmRegisters[i], s->f->vin[16 + i], 16);
    }
    RtlUnwindEx(frame, (PVOID)xfo_landing, rec, (PVOID)s->retval, dc->ContextRecord, dc->HistoryTable);
    CHECKF(0, "RtlUnwindEx returned");
    return ExceptionContinueSearch;
}
#endif
TEST(seh_unwind_through_inner_frame_x64) {
#ifdef T_X64
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x7501, 16, 0, 0);
    s->sehfn = seh_outer; s->f = f; s->retval = 0xABCDEF12ull;
    s->op = xfi_fault; s->at = xfi_fault; s->resume = xfo_landing;
    TlsSetValue(g_tls, s);
    s->armed = 1;
    xfo_outer(f);
    s->armed = 0;
    CHECK_EQ(s->seh_calls, 1);
    CHECKF(s->dc_pc == (uintptr_t)xfo_after_call, "ControlPc for the outer frame %p want its return address %p", (void *)s->dc_pc,
           (const void *)xfo_after_call);
    CHECKF(s->dc_frame == (uintptr_t)f->rsp_at, "EstablisherFrame %p want %p", (void *)s->dc_frame, (void *)(uintptr_t)f->rsp_at);
    /* the dispatcher's ContextRecord is unwound past the outer frame (inner frame and outer frame popped) */
    CHECKF(s->dc_gpr[4] == f->rsp_at + 272, "unwound ContextRecord Rsp 0x%llx want 0x%llx", (unsigned long long)s->dc_gpr[4],
           (unsigned long long)(f->rsp_at + 272));
    CHECKF(s->unwind_calls == 1 && (s->unwind_flags & X_EXC_TARGET_UNWIND), "unwind calls %ld flags 0x%lx", (long)s->unwind_calls,
           (unsigned long)s->unwind_flags);
    CHECKF(f->gout[0] == s->retval, "RAX at the landing 0x%llx want 0x%llx", (unsigned long long)f->gout[0], (unsigned long long)s->retval);
    f->vskip = 0x3F; f->gskip = 0x0F07u; f->mxexp = f->mxcsr_out;
    xf_check_out(f, "after unwinding the inner frame (nonvolatiles restored from its save slots)");
    xf_free(f);
#else
    SKIP("table-based unwinding is x64 only");
#endif
}

/* ========================================================================================== */
/* Vectored handler chain                                                                       */
/* ========================================================================================== */
#define XC_ORDER 0xE0000101u
static volatile LONG g_ord_n; static int g_ord[16];
static void ord_push(int id) { LONG i = InterlockedIncrement(&g_ord_n) - 1; if (i < 16) g_ord[i] = id; }
#define ORD_VEH(n, ret) static LONG CALLBACK ord_veh##n(EXCEPTION_POINTERS *ep) { \
    if (ep->ExceptionRecord->ExceptionCode != XC_ORDER) { return EXCEPTION_CONTINUE_SEARCH; } \
    ord_push(n); \
    return ret; }
ORD_VEH(1, EXCEPTION_CONTINUE_SEARCH)
ORD_VEH(2, EXCEPTION_CONTINUE_SEARCH)
ORD_VEH(3, EXCEPTION_CONTINUE_EXECUTION)
ORD_VEH(4, EXCEPTION_CONTINUE_SEARCH)
ORD_VEH(5, EXCEPTION_CONTINUE_SEARCH)
static LONG CALLBACK ord_vch(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != XC_ORDER) return EXCEPTION_CONTINUE_SEARCH;
    ord_push(9); return EXCEPTION_CONTINUE_SEARCH;
}
static void ord_reset(void) { g_ord_n = 0; memset(g_ord, 0, sizeof g_ord); }
static void ord_expect(const int *want, int n, const char *what) {
    char got[64] = "", exp[64] = "";
    for (int i = 0; i < (int)g_ord_n && i < 16; i++) snprintf(got + strlen(got), sizeof got - strlen(got), "%d ", g_ord[i]);
    for (int i = 0; i < n; i++) snprintf(exp + strlen(exp), sizeof exp - strlen(exp), "%d ", want[i]);
    int ok = (int)g_ord_n == n;
    for (int i = 0; ok && i < n; i++) ok = g_ord[i] == want[i];
    CHECKF(ok, "%s: handler order [%s] want [%s]", what, got, exp);
}
TEST(veh_first_last_order) {
    /* documented: FirstHandler != 0 -> called first (head of the list); 0 -> called last */
    xs_init();
    PVOID a = AddVectoredExceptionHandler(0, ord_veh1);   /* last */
    PVOID b = AddVectoredExceptionHandler(1, ord_veh2);   /* first */
    PVOID c = AddVectoredExceptionHandler(0, ord_veh3);   /* last (after a) - handles */
    PVOID d = AddVectoredExceptionHandler(1, ord_veh4);   /* first (before b) */
    PVOID e = AddVectoredExceptionHandler(0, ord_veh5);   /* after c: must not run */
    REQUIRE(a && b && c && d && e);
    ord_reset();
    RaiseException(XC_ORDER, 0, 0, NULL);
    static const int want1[] = { 4, 2, 1, 3 };
    ord_expect(want1, 4, "AddVectoredExceptionHandler first/last");
    CHECKF(RemoveVectoredExceptionHandler(b) != 0, "RemoveVectoredExceptionHandler(registered) returned 0");
    CHECKF(RemoveVectoredExceptionHandler(b) == 0, "RemoveVectoredExceptionHandler(already removed) returned nonzero");
    ord_reset();
    RaiseException(XC_ORDER, 0, 0, NULL);
    static const int want2[] = { 4, 1, 3 };
    ord_expect(want2, 3, "after removing one handler");
    RemoveVectoredExceptionHandler(a); RemoveVectoredExceptionHandler(c); RemoveVectoredExceptionHandler(d);
    RemoveVectoredExceptionHandler(e);
}
TEST(vch_called_after_veh_continue) {
    xs_init();
    PVOID (WINAPI *addvch)(ULONG, PVECTORED_EXCEPTION_HANDLER) =
        (PVOID (WINAPI *)(ULONG, PVECTORED_EXCEPTION_HANDLER))t_proc("kernel32.dll", "AddVectoredContinueHandler");
    ULONG (WINAPI *remvch)(PVOID) = (ULONG (WINAPI *)(PVOID))t_proc("kernel32.dll", "RemoveVectoredContinueHandler");
    if (!addvch || !remvch) SKIP("AddVectoredContinueHandler not exported");
    wine_gap("vectored continue handlers are registered but never called by the dispatcher");
    PVOID v = addvch(1, ord_vch);
    PVOID h = AddVectoredExceptionHandler(1, ord_veh3);
    REQUIRE(v && h);
    ord_reset();
    RaiseException(XC_ORDER, 0, 0, NULL);
    static const int want[] = { 3, 9 };
    ord_expect(want, 2, "VEH(continue) then continue handler");
    RemoveVectoredExceptionHandler(h);
    CHECK(remvch(v) != 0);
    CHECK(remvch(v) == 0);
}
/* continue handler edits the CONTEXT: the edit must be what resumes (same CONTEXT object) */
static LONG CALLBACK vch_mod_xmm(EXCEPTION_POINTERS *ep) {
    xh *s = xh_cur();
    if (!s || !s->armed || !xh_mine(s, ep->ExceptionRecord, ep->ContextRecord)) return EXCEPTION_CONTINUE_SEARCH;
    uint8_t *p = ctx_xmm(ep->ContextRecord, 7);
    if (p) { uint8_t nv[16]; t_pattern_reg(nv, 16, 0xC0C0, 7); memcpy(p, nv, 16); memcpy(s->f->vexp[7], nv, 16); }
    InterlockedIncrement(&s->nested_calls);
    return EXCEPTION_CONTINUE_SEARCH;
}
TEST(vch_context_edit_takes_effect) {
    xs_init();
    PVOID (WINAPI *addvch)(ULONG, PVECTORED_EXCEPTION_HANDLER) =
        (PVOID (WINAPI *)(ULONG, PVECTORED_EXCEPTION_HANDLER))t_proc("kernel32.dll", "AddVectoredContinueHandler");
    ULONG (WINAPI *remvch)(PVOID) = (ULONG (WINAPI *)(PVOID))t_proc("kernel32.dll", "RemoveVectoredContinueHandler");
    if (!addvch || !remvch) SKIP("AddVectoredContinueHandler not exported");
    wine_gap("vectored continue handlers are registered but never called by the dispatcher");
    const xk *k = xk_find("div0");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x8101, 16, 0, 0); xk_prepare(k, f, &s->e);
    s->veh_on = 1;
    PVOID v = addvch(1, vch_mod_xmm);
    REQUIRE(v);
    CHECK_EQ(xk_arm_and_run(s, k, f, V_VEH), 1);
    remvch(v);
    CHECKF(s->nested_calls == 1, "continue handler ran %ld times after the VEH continued", (long)s->nested_calls);
    xf_check_out(f, "after a continue handler edited xmm7");
    xf_free(f);
}
/* two vectored handlers share one CONTEXT: the first edits xmm7 and searches, the second sees the edit */
static uint8_t g_share_val[16]; static volatile LONG g_share_seen;
static LONG CALLBACK share_first(EXCEPTION_POINTERS *ep) {
    xh *s = xh_cur();
    if (!s || !s->armed || !xh_mine(s, ep->ExceptionRecord, ep->ContextRecord)) return EXCEPTION_CONTINUE_SEARCH;
    uint8_t *p = ctx_xmm(ep->ContextRecord, 7);
    if (p) memcpy(p, g_share_val, 16);
    return EXCEPTION_CONTINUE_SEARCH;
}
static LONG CALLBACK share_second(EXCEPTION_POINTERS *ep) {
    xh *s = xh_cur();
    if (!s || !s->armed || !xh_mine(s, ep->ExceptionRecord, ep->ContextRecord)) return EXCEPTION_CONTINUE_SEARCH;
    uint8_t *p = ctx_xmm(ep->ContextRecord, 7);
    if (p) { CHECK_REG("second VEH sees xmm7 edited by the first", p, g_share_val, 16); InterlockedIncrement(&g_share_seen); }
    CSET_PC(ep->ContextRecord, s->resume);
    InterlockedIncrement(&s->calls);
    return EXCEPTION_CONTINUE_EXECUTION;
}
TEST(veh_shared_context_between_handlers) {
    xs_init();
    const xk *k = xk_find("rd");
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x8201, 16, 0, 0); xk_prepare(k, f, &s->e);
    t_pattern_reg(g_share_val, 16, 0xBEEF, 7);
    PVOID h2 = AddVectoredExceptionHandler(1, share_second);
    PVOID h1 = AddVectoredExceptionHandler(1, share_first);
    REQUIRE(h1 && h2);
    g_share_seen = 0;
    s->veh_on = 0;
    xk_arm_and_run(s, k, f, V_VEH);
    RemoveVectoredExceptionHandler(h1); RemoveVectoredExceptionHandler(h2);
    CHECK_EQ(s->calls, 1);
#ifdef T_X64
    CHECK_EQ(g_share_seen, 1);
    memcpy(f->vexp[7], g_share_val, 16);
#else
    if (g_share_seen) memcpy(f->vexp[7], g_share_val, 16);
#endif
    xf_check_out(f, "after two VEHs shared the CONTEXT");
    xf_free(f);
}
static LONG CALLBACK nested_inner_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != 0xE0000ABCu) return EXCEPTION_CONTINUE_SEARCH;
    CHECKF(ep->ExceptionRecord->NumberParameters == 1 && ep->ExceptionRecord->ExceptionInformation[0] == 0x1234,
           "nested record params %lu", (unsigned long)ep->ExceptionRecord->NumberParameters);
    InterlockedIncrement(&g_nested_hits);
    return EXCEPTION_CONTINUE_EXECUTION;
}
TEST(veh_nested_exception_in_handler) {
    g_nested_hits = 0;
    xk_test("div0", cpu_has(F_AVX) ? 32 : 16, M_NESTED_VEH, V_VEH, NULL, "after a nested RaiseException inside the VEH");
    CHECK_EQ(g_nested_hits, 1);
    CHECK_EQ(g_main_xh.nested_calls, 1);
}

/* ========================================================================================== */
/* RaiseException                                                                              */
/* ========================================================================================== */
#define XC_RAISE 0xE0001234u
static EXCEPTION_RECORD g_rr; static volatile LONG g_rr_n;
static LONG CALLBACK raise_veh(EXCEPTION_POINTERS *ep) {
    if ((ep->ExceptionRecord->ExceptionCode & 0xFFFFFF00u) != (XC_RAISE & 0xFFFFFF00u)) return EXCEPTION_CONTINUE_SEARCH;
    g_rr = *ep->ExceptionRecord; InterlockedIncrement(&g_rr_n);
    return EXCEPTION_CONTINUE_EXECUTION;
}
TEST(raise_custom_code_args) {
    PVOID h = AddVectoredExceptionHandler(1, raise_veh); REQUIRE(h);
    ULONG_PTR args[4] = { 0x11, (ULONG_PTR)-2, 0x8000, (ULONG_PTR)&g_rr };
    g_rr_n = 0;
    RaiseException(XC_RAISE, 0, 4, args);
    RemoveVectoredExceptionHandler(h);
    CHECK_EQ(g_rr_n, 1);
    CHECK_EQ(g_rr.ExceptionCode, XC_RAISE);
    CHECK_EQ(g_rr.ExceptionFlags, 0);
    CHECK(g_rr.ExceptionRecord == NULL);
    CHECK(g_rr.ExceptionAddress != NULL);
    CHECK_EQ(g_rr.NumberParameters, 4);
    for (int i = 0; i < 4; i++) CHECKF(g_rr.ExceptionInformation[i] == args[i], "Information[%d] 0x%llx want 0x%llx", i,
                                       (unsigned long long)g_rr.ExceptionInformation[i], (unsigned long long)args[i]);
}
TEST(raise_parameter_count_clamped) {
    PVOID h = AddVectoredExceptionHandler(1, raise_veh); REQUIRE(h);
    ULONG_PTR args[20];
    for (int i = 0; i < 20; i++) args[i] = 0x100u + (ULONG_PTR)i;
    g_rr_n = 0;
    RaiseException(XC_RAISE + 1, 0, 20, args);
    CHECK_EQ(g_rr.NumberParameters, EXCEPTION_MAXIMUM_PARAMETERS);
    for (int i = 0; i < EXCEPTION_MAXIMUM_PARAMETERS; i++) CHECK_EQ(g_rr.ExceptionInformation[i], args[i]);
    RaiseException(XC_RAISE + 2, 0, 3, NULL);     /* lpArguments NULL: no parameters */
    CHECK_EQ(g_rr.ExceptionCode, XC_RAISE + 2);
    CHECK_EQ(g_rr.NumberParameters, 0);
    RaiseException(XC_RAISE + 3, 0, 0, args);
    CHECK_EQ(g_rr.NumberParameters, 0);
    RemoveVectoredExceptionHandler(h);
    CHECK_EQ(g_rr_n, 3);
}
TEST(raise_flags_masked_to_noncontinuable) {
    PVOID h = AddVectoredExceptionHandler(1, raise_veh); REQUIRE(h);
    g_rr_n = 0;
    RaiseException(XC_RAISE + 4, 0xFFFFFFFEu, 0, NULL);   /* every bit except NONCONTINUABLE: all dropped */
    RemoveVectoredExceptionHandler(h);
    CHECK_EQ(g_rr_n, 1);
    CHECKF(g_rr.ExceptionFlags == 0, "ExceptionFlags 0x%lx want 0 (RaiseException keeps only EXCEPTION_NONCONTINUABLE)",
           (unsigned long)g_rr.ExceptionFlags);
}
/* RaiseException from code that holds patterns in the nonvolatile registers (call through the x?_call path) */
static void raise_cb(void *arg) { (void)arg; RaiseException(XC_RAISE + 5, 0, 0, NULL); }
TEST(raise_preserves_abi_nonvolatile_registers) {
    PVOID h = AddVectoredExceptionHandler(1, raise_veh); REQUIRE(h);
    t_vregs in, out;
    int kinds[3] = { T_VK_XMM, T_VK_YMM, T_VK_ZMM };
    for (int j = 0; j < 3; j++) {
        if (!t_vregs_supported(kinds[j])) continue;
        t_vregs_pattern(&in, kinds[j], 0x9100u + (uint32_t)j);
        memset(&out, 0, sizeof out);
        g_rr_n = 0;
        REQUIRE(t_vregs_call(kinds[j], &in, &out, raise_cb, NULL) == 0);
        CHECK_EQ(g_rr_n, 1);
#ifdef T_X64
        /* ABI: only XMM6-15 (low 128 bits) are nonvolatile across the call that raised */
        for (int i = 6; i < 16; i++) {
            char what[48]; snprintf(what, sizeof what, "xmm%d after RaiseException (kind %d)", i, kinds[j]);
            CHECK_REG(what, out.v[i], in.v[i], 16);
        }
#endif
    }
    RemoveVectoredExceptionHandler(h);
#ifdef T_X86
    t_note("x86: no vector register is nonvolatile; only completion is checked");
#endif
}
static CONTEXT g_rctx; static DWORD g_rctx_flags;
static LONG CALLBACK raise_ctx_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != XC_RAISE + 6) return EXCEPTION_CONTINUE_SEARCH;
    g_rctx_flags = ep->ContextRecord->ContextFlags;
    memcpy(&g_rctx, ep->ContextRecord, sizeof g_rctx);
    return EXCEPTION_CONTINUE_EXECUTION;
}
TEST(raise_context_flags_info) {
    PVOID h = AddVectoredExceptionHandler(1, raise_ctx_veh); REQUIRE(h);
    RaiseException(XC_RAISE + 6, 0, 0, NULL);
    RemoveVectoredExceptionHandler(h);
    t_info("raise_context_flags", "0x%lx", (unsigned long)g_rctx_flags);
    CHECKF((g_rctx_flags & X_CONTEXT_FULLFP) == X_CONTEXT_FULLFP, "RaiseException CONTEXT flags 0x%lx", (unsigned long)g_rctx_flags);
    CHECKF(CSP(&g_rctx) != 0 && CPC(&g_rctx) != 0, "RaiseException CONTEXT has no PC/SP");
}

/* ========================================================================================== */
/* Worker threads                                                                               */
/* ========================================================================================== */
typedef struct wk { int idx, vk, iters; volatile LONG done, exc; } wk;
static DWORD WINAPI worker_exc(void *p) {
    wk *w = (wk *)p;
    xf *f = xf_alloc();
    xh *s = (xh *)calloc(1, sizeof(xh));
    if (!f || !s) { CHECKF(0, "worker %d: allocation failed", w->idx); return 1; }
    static const char *const kinds[] = { "div0", "rd", "int3", "nop", "wr" };
    for (int it = 0; it < w->iters; it++) {
        const xk *k = xk_find(kinds[it % 5]);
        memset(s, 0, sizeof *s);
        xf_init(f, (uint32_t)(w->idx * 1000 + it + 1), w->vk, 0, (it & 3) == 1);
        xk_prepare(k, f, &s->e);
        s->veh_on = 1; s->quiet = 1; s->mode = (it % 7 == 3) ? M_MOD_XMM : (it % 7 == 5) ? M_CLOBBER : 0;
        s->tag2 = (uint32_t)(w->idx * 1000 + it + 50000);
        int calls = xk_arm_and_run(s, k, f, V_VEH);
        CHECKF(calls == (s->e.code ? 1 : 0), "worker %d iteration %d (%s): handler ran %d times", w->idx, it, k->name, calls);
        char lab[64]; snprintf(lab, sizeof lab, "worker %d it %d %s", w->idx, it, k->name);
        xf_check_out(f, lab);
        InterlockedExchangeAdd(&w->exc, calls);
    }
    InterlockedExchange(&w->done, 1);
    xf_free(f); free(s);
    return 0;
}
static void worker_case(int vk, int nthreads, int iters) {
    xs_init();
    wk w[64]; HANDLE h[64]; int n = 0;
    if (nthreads > 64) nthreads = 64;
    for (int i = 0; i < nthreads; i++) {
        memset(&w[i], 0, sizeof w[i]); w[i].idx = i + 1; w[i].vk = vk; w[i].iters = iters;
        h[n] = CreateThread(NULL, 256 * 1024, worker_exc, &w[i], 0, NULL);
        if (!h[n]) break;
        n++;
    }
    if (n == 0) SKIP("CreateThread failed");
    DWORD wr = WaitForMultipleObjects((DWORD)n, h, TRUE, 30000);
    CHECKF(wr != WAIT_TIMEOUT, "workers did not finish within 30 s");
    long total = 0;
    for (int i = 0; i < n; i++) { CHECKF(w[i].done, "worker %d did not complete", i + 1); total += w[i].exc; CloseHandle(h[i]); }
    t_info("threads", "%d", n);
    t_info("exceptions", "%ld", total);
}
TEST(thr_worker_exceptions_isolated_xmm) { worker_case(16, 8, 60); }
TEST(thr_worker_exceptions_isolated_ymm) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    worker_case(32, 16, 60);
}
TEST(thr_worker_exceptions_isolated_zmm) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    worker_case(64, 8, 40);
}
static DWORD WINAPI worker_fp(void *p) {
    wk *w = (wk *)p;
    xf *f = xf_alloc(); xh *s = (xh *)calloc(1, sizeof(xh));
    if (!f || !s) return 1;
    for (int it = 0; it < w->iters; it++) {
        memset(s, 0, sizeof *s);
        xf_init(f, (uint32_t)(w->idx * 777 + it + 1), w->vk, 0, 0);
        fpe_t which = (fpe_t)(it % 6);
        if (which == FPE_DE && t_is_wine()) which = FPE_ZE;
#ifdef T_X86
        if (t_is_wine() && (which == FPE_OE || which == FPE_UE || which == FPE_PE)) which = FPE_IE;
#endif
        const xk *k = sse_setup(which, f, &s->e, 1, NULL);
        s->veh_on = 1; s->quiet = 1; s->mode = M_FPFIX;
        int calls = xk_arm_and_run(s, k, f, V_VEH);
        CHECKF(calls == 1, "fp worker %d it %d: calls %d", w->idx, it, calls);
        char lab[64]; snprintf(lab, sizeof lab, "fp worker %d it %d %s", w->idx, it, g_fpe_name[it % 6]);
        xf_check_out(f, lab);
        InterlockedExchangeAdd(&w->exc, calls);
    }
    InterlockedExchange(&w->done, 1);
    xf_free(f); free(s);
    return 0;
}
TEST(thr_worker_fp_exceptions) {
    if (t_is_wine()) t_note("Wine: iterations with a known exception-code gap (see fp_sse_*_unmasked) replaced by ZE/IE");
    xs_init();
    wk w[8]; HANDLE h[8]; int n = 0;
    for (int i = 0; i < 8; i++) {
        memset(&w[i], 0, sizeof w[i]); w[i].idx = i + 1; w[i].vk = cpu_has(F_AVX) ? 32 : 16; w[i].iters = 36;
        h[n] = CreateThread(NULL, 256 * 1024, worker_fp, &w[i], 0, NULL);
        if (!h[n]) break;
        n++;
    }
    if (!n) SKIP("CreateThread failed");
    CHECK(WaitForMultipleObjects((DWORD)n, h, TRUE, 30000) != WAIT_TIMEOUT);
    for (int i = 0; i < n; i++) { CHECK(w[i].done); CloseHandle(h[i]); }
}

/* ========================================================================================== */
/* Child-process scenarios (#UD, privileged instructions, fast fail, non-continuable)          */
/* ========================================================================================== */
TEST(child_ud2_illegal_instruction_handled) {
    run_child_kind("ud2", 16, 0, V_VEH, NULL);
    if (cpu_has(F_AVX)) run_child_kind("ud2", 32, M_MOD_XMM, V_VEH, NULL);
}
TEST(child_ud2_frame_handler) { run_child_kind("ud2", 16, 0, V_SEHCALL, NULL); }
CHILD(unhandled) {
    /* argv: unhandled <kind>; print the label address, then fault with no handler */
    if (argc < 2) return 3;
    const xk *k = xk_find(argv[1]);
    if (!k) return 4;
    xf *f = xf_alloc(); if (!f) return 5;
    xexp e;
    xf_init(f, 0xDEAD, 16, 0, 0); xk_prepare(k, f, &e);
    printf("AT=0x%llx\n", (unsigned long long)(uintptr_t)k->at); fflush(stdout);
    k->fn(f);
    printf("RETURNED\n");
    return 0;
}
static void unhandled_case(const char *kind, DWORD code) {
    if (!t_child_run("unhandled", kind, 15000, &g_cr) || !g_cr.started) { CHECKF(0, "child spawn failed"); return; }
    char v[64]; unsigned long long at = 0;
    if (t_child_find(&g_cr, "AT=", v, sizeof v)) at = strtoull(v, NULL, 16);
    CHECKF(!t_child_find(&g_cr, "RETURNED", v, sizeof v), "%s: execution continued after an unhandled fault", kind);
    CHECKF(g_cr.exc_code == code, "%s: child exception 0x%08lx want 0x%08lx (exit 0x%08lx)", kind, (unsigned long)g_cr.exc_code,
           (unsigned long)code, (unsigned long)g_cr.exit_code);
    CHECKF(g_cr.exit_code == code, "%s: exit code 0x%08lx want the exception code", kind, (unsigned long)g_cr.exit_code);
    CHECKF(at && g_cr.exc_addr == at, "%s: unhandled-exception address 0x%llx want 0x%llx", kind, g_cr.exc_addr, at);
}
TEST(child_ud2_unhandled_exit_code) { unhandled_case("ud2", XS_ILLEGAL_INSTRUCTION); }
TEST(child_div0_unhandled_exit_code) { unhandled_case("div0", XS_INT_DIVIDE_BY_ZERO); }
TEST(child_privileged_hlt_cli_sti) {
    run_child_kind("hlt", 16, 0, V_VEH, NULL);
    run_child_kind("cli", 16, 0, V_VEH, NULL);
    run_child_kind("sti", cpu_has(F_AVX) ? 32 : 16, 0, V_VEH, NULL);
}
TEST(child_privileged_misc) {
    static const char *const k[] = { "inb", "outb", "wbinvd", "rdmsr", "wrmsr", "movcr", "clts", "invd" };
    char codes[256] = "";
    for (size_t i = 0; i < T_COUNTOF(k); i++) {
        DWORD c = 0;
        run_child_kind(k[i], 16, 0, V_VEH, &c);
        snprintf(codes + strlen(codes), sizeof codes - strlen(codes), "%s=0x%08lx ", k[i], (unsigned long)c);
    }
    t_info("codes", "%s", codes);
}
TEST(child_xsetbv_privileged) {
    if (!cpu_has(F_OSXSAVE)) SKIP("OSXSAVE not set: XSETBV would be #UD, not a privilege fault");
    DWORD c = 0;
    run_child_kind("xsetbv", cpu_has(F_AVX) ? 32 : 16, 0, V_VEH, &c);
    t_info("code", "0x%08lx", (unsigned long)c);
}
TEST(child_int2c_assertion_failure) {
#ifdef T_X86
    wine_gap("32-bit INT 2Ch raises STATUS_ACCESS_VIOLATION instead of STATUS_ASSERTION_FAILURE");
#endif
    DWORD c = 0;
    run_child_kind("int2c", 16, 0, V_VEH, &c);
    t_info("code", "0x%08lx", (unsigned long)c);
}
static volatile LONG g_ff_veh;
static LONG CALLBACK ff_veh(EXCEPTION_POINTERS *ep) {
    printf("VEH_CALLED code=0x%08lx\n", (unsigned long)ep->ExceptionRecord->ExceptionCode); fflush(stdout);
    InterlockedIncrement(&g_ff_veh);
    return EXCEPTION_CONTINUE_SEARCH;
}
CHILD(fastfail) {
    xs_init();
    AddVectoredExceptionHandler(1, ff_veh);
    const xk *k = xk_find("fastfail");
    xf *f = xf_alloc(); if (!f) return 5;
    xexp e; xf_init(f, 0xFA57, 16, 0, 0); xk_prepare(k, f, &e);
    k->fn(f);
    printf("RETURNED\n");
    return 0;
}
TEST(child_fastfail_bypasses_handlers) {
    if (!t_child_run("fastfail", "", 15000, &g_cr) || !g_cr.started) { CHECKF(0, "child spawn failed"); return; }
    char v[64];
    t_info("exit", "0x%08lx", (unsigned long)g_cr.exit_code);
    CHECKF(!t_child_find(&g_cr, "RETURNED", v, sizeof v), "execution continued after __fastfail");
    CHECKF(g_cr.exit_code == XS_STACK_BUFFER_OVERRUN, "exit code 0x%08lx want STATUS_STACK_BUFFER_OVERRUN", (unsigned long)g_cr.exit_code);
    CHECKF(!t_child_find(&g_cr, "VEH_CALLED", v, sizeof v), "a vectored handler saw the fast-fail (%s)", v);
}
static void *raise_nc(void *a) { (void)a; RaiseException(0xE0000BADu, EXCEPTION_NONCONTINUABLE, 0, NULL); return (void *)1; }
static EXCEPTION_DISPOSITION seh_continue_all(xh *s, EXCEPTION_RECORD *rec, void *frame, CONTEXT *c, void *dc) {
    (void)s; (void)frame; (void)c; (void)dc;
    if (rec->ExceptionCode == 0xE0000BADu && !(rec->ExceptionFlags & X_EXC_UNWINDING)) {
        printf("SEH_SAW flags=0x%lx\n", (unsigned long)rec->ExceptionFlags); fflush(stdout);
        return ExceptionContinueExecution;
    }
    return ExceptionContinueSearch;
}
static LONG CALLBACK nc_veh(EXCEPTION_POINTERS *ep) {
    EXCEPTION_RECORD *r = ep->ExceptionRecord;
    if (r->ExceptionCode == 0xE0000BADu) return EXCEPTION_CONTINUE_SEARCH;
    printf("NC code=0x%08lx flags=0x%lx inner=0x%08lx\n", (unsigned long)r->ExceptionCode, (unsigned long)r->ExceptionFlags,
           (unsigned long)(r->ExceptionRecord ? r->ExceptionRecord->ExceptionCode : 0));
    fflush(stdout);
    TerminateProcess(GetCurrentProcess(), 0x25);
    return EXCEPTION_CONTINUE_SEARCH;
}
CHILD(noncont) {
    xs_init();
    AddVectoredExceptionHandler(1, nc_veh);
    xh *s = (xh *)calloc(1, sizeof(xh));
    s->sehfn = seh_continue_all;
    TlsSetValue(g_tls, s);
    xseh_call(raise_nc, NULL);
    printf("RETURNED\n");
    return 0;
}
TEST(child_noncontinuable_continue_raises_0xC0000025) {
    if (!t_child_run("noncont", "", 15000, &g_cr) || !g_cr.started) { CHECKF(0, "child spawn failed"); return; }
    char v[128];
    int saw = t_child_find(&g_cr, "SEH_SAW ", v, sizeof v);
    CHECKF(saw, "frame handler never saw the non-continuable exception");
    if (saw) {
        unsigned long fl = 0; sscanf(v, "flags=0x%lx", &fl);
        CHECKF(fl & X_EXC_NONCONTINUABLE, "flags 0x%lx lack EXCEPTION_NONCONTINUABLE", fl);
    }
    CHECKF(!t_child_find(&g_cr, "RETURNED", v, sizeof v), "continuing a non-continuable exception returned to the raiser");
    if (t_child_find(&g_cr, "NC ", v, sizeof v)) {
        unsigned long code = 0, fl = 0, inner = 0;
        sscanf(v, "code=0x%lx flags=0x%lx inner=0x%lx", &code, &fl, &inner);
        CHECKF(code == XS_NONCONTINUABLE, "follow-up exception 0x%08lx want STATUS_NONCONTINUABLE_EXCEPTION", code);
        CHECKF(fl & X_EXC_NONCONTINUABLE, "STATUS_NONCONTINUABLE_EXCEPTION flags 0x%lx lack NONCONTINUABLE", fl);
        /* Windows 10 RtlDispatchException raises it with RtlRaiseStatus (fresh record, no chain); older NT chained
         * the original. Either is acceptable. */
        CHECKF(inner == 0 || inner == 0xE0000BADu, "chained ExceptionRecord code 0x%08lx", inner);
        t_info("chained_record", "0x%08lx", inner);
    } else CHECKF(0, "no STATUS_NONCONTINUABLE_EXCEPTION was raised (exit 0x%08lx)", (unsigned long)g_cr.exit_code);
}
static LONG CALLBACK nc_veh_continue(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != 0xE0000BAEu) return EXCEPTION_CONTINUE_SEARCH;
    printf("VEH_FLAGS=0x%lx\n", (unsigned long)ep->ExceptionRecord->ExceptionFlags); fflush(stdout);
    return EXCEPTION_CONTINUE_EXECUTION;
}
CHILD(noncont_veh) {
    AddVectoredExceptionHandler(1, nc_veh_continue);
    RaiseException(0xE0000BAEu, EXCEPTION_NONCONTINUABLE, 0, NULL);
    printf("RETURNED\n");
    return 0;
}
TEST(child_noncontinuable_veh_continue_info) {
    if (!t_child_run("noncont_veh", "", 15000, &g_cr) || !g_cr.started) { CHECKF(0, "child spawn failed"); return; }
    char v[64];
    int saw = t_child_find(&g_cr, "VEH_FLAGS=", v, sizeof v);
    CHECKF(saw, "VEH did not see the exception");
    if (saw) CHECKF(strtoul(v, NULL, 16) & X_EXC_NONCONTINUABLE, "VEH saw flags %s", v);
    t_info("outcome", "%s exit=0x%08lx", t_child_find(&g_cr, "RETURNED", v, sizeof v) ? "RaiseException returned" : "terminated",
           (unsigned long)g_cr.exit_code);
}

/* ========================================================================================== */
/* x86-only traps                                                                               */
/* ========================================================================================== */
TEST(x86_into_overflow_trap) {
#ifdef T_X86
    xk_test("into", 16, 0, V_VEH, NULL, "after INTO");
    t_info("address_delta", "%ld", (long)((intptr_t)g_main_xh.rec.ExceptionAddress - (intptr_t)xk_find("into")->at));
#else
    SKIP("INTO is invalid in 64-bit mode");
#endif
}
TEST(x86_bound_range_exceeded) {
#ifdef T_X86
    xk_test("bound", 16, 0, V_VEH, NULL, "after BOUND");
#else
    SKIP("BOUND is invalid in 64-bit mode");
#endif
}

/* ========================================================================================== */
/* setjmp / longjmp (msvcrt)                                                                    */
/* ========================================================================================== */
static void sj_case(int with_frame) {
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xf_init(f, 0xA100u + (uint32_t)with_frame, 16, 0, 0);
    f->aux = (uint64_t)with_frame;
    f->fcw_in = t_get_x87_cw();
    xsj_probe(f);
    /* only the ABI nonvolatile set is defined after longjmp */
#ifdef T_X64
    f->vskip = 0x3F; f->gskip = 0x0F07u;
#else
    f->vskip = 0xFF; f->gskip = 0x07u;
#endif
    uint32_t mx_after = f->mxcsr_out;
    f->mxexp = f->mxcsr_out;
    char lab[64]; snprintf(lab, sizeof lab, "after longjmp (%s)", with_frame ? "unwinding frame" : "frame 0");
    xf_check_out(f, lab);
    t_info("mxcsr_after_longjmp", "0x%x (setjmp time 0x%x, before longjmp 0x%x)", mx_after, f->mxcsr_in, f->mxcsr2);
    t_info("x87cw_after_longjmp", "0x%04x (setjmp time 0x%04x, before longjmp 0x%04x)", f->fcw_out, f->fcw_in, f->fcw2);
    /* the jump buffer itself (documented _JUMP_BUFFER layout in setjmp.h) */
#ifdef T_X64
    const uint64_t *q = (const uint64_t *)f->jb;
    CHECKF(q[0] == (with_frame ? f->rsp_at : 0), "_JUMP_BUFFER.Frame 0x%llx want 0x%llx", (unsigned long long)q[0],
           (unsigned long long)(with_frame ? f->rsp_at : 0));
    CHECKF(q[1] == (uint64_t)(uintptr_t)f, "_JUMP_BUFFER.Rbx 0x%llx", (unsigned long long)q[1]);
    CHECKF(q[2] == f->rsp_at, "_JUMP_BUFFER.Rsp 0x%llx want 0x%llx", (unsigned long long)q[2], (unsigned long long)f->rsp_at);
    CHECK_EQ(q[3], f->gin[5]); CHECK_EQ(q[4], f->gin[6]); CHECK_EQ(q[5], f->gin[7]);
    for (int i = 12; i < 16; i++) CHECK_EQ(q[6 + i - 12], f->gin[i]);
    CHECKF(q[10] == (uint64_t)(uintptr_t)xsj_ret, "_JUMP_BUFFER.Rip 0x%llx want 0x%llx", (unsigned long long)q[10],
           (unsigned long long)(uintptr_t)xsj_ret);
    uint32_t jmx; uint16_t jcw; memcpy(&jmx, f->jb + 0x58, 4); memcpy(&jcw, f->jb + 0x5C, 2);
    CHECKF(jmx == f->mxcsr_in, "_JUMP_BUFFER.MxCsr 0x%x want 0x%x", jmx, f->mxcsr_in);
    CHECKF(jcw == f->fcw_in, "_JUMP_BUFFER.FpCsr 0x%04x want 0x%04x", jcw, f->fcw_in);
    for (int i = 6; i < 16; i++) {
        char what[48]; snprintf(what, sizeof what, "_JUMP_BUFFER.Xmm%d", i);
        CHECK_REG(what, f->jb + 0x60 + 16 * (i - 6), f->vin[i], 16);
    }
#else
    const uint32_t *q = (const uint32_t *)f->jb;
    CHECK_EQ(q[0], (uint32_t)f->gin[5]);                 /* Ebp */
    CHECK_EQ(q[1], (uint32_t)(uintptr_t)f);              /* Ebx */
    CHECK_EQ(q[2], (uint32_t)f->gin[7]);                 /* Edi */
    CHECK_EQ(q[3], (uint32_t)f->gin[6]);                 /* Esi */
    CHECKF(q[5] == (uint32_t)(uintptr_t)xsj_ret, "_JUMP_BUFFER.Eip 0x%x want 0x%x", q[5], (uint32_t)(uintptr_t)xsj_ret);
    t_info("jb_esp_delta", "%ld", (long)((int32_t)q[4] - (int32_t)f->rsp_at));
#endif
    xf_free(f);
}
TEST(sj_longjmp_unwinding_restores_nonvolatile) {
#ifdef T_X64
    sj_case(1);
#else
    SKIP("x86 _setjmp takes no frame argument; covered by sj_longjmp_frame0_restores_nonvolatile");
#endif
}
TEST(sj_longjmp_frame0_restores_nonvolatile) { sj_case(0); }

/* ========================================================================================== */
/* RtlCaptureContext / RtlCaptureContext2 / RtlRestoreContext / NtContinue                     */
/* ========================================================================================== */
typedef LONG (NTAPI *pfn_ntcontinue)(PCONTEXT, BOOLEAN);
typedef VOID (NTAPI *pfn_capture)(PCONTEXT);
#ifdef T_X64
typedef VOID (WINAPI *pfn_restore)(PCONTEXT, PEXCEPTION_RECORD);
#endif
static struct { int how; uint32_t tag; xf *f; int ymm; } g_cap;
static void cap_cb(void *pf) {
    xf *f = (xf *)pf;
    CONTEXT *c = (CONTEXT *)(uintptr_t)f->ctxp;
    /* edit: every XMM, r12/rsi (nonvolatile), rax (volatile), MXCSR; YMM uppers when XSTATE */
    for (int i = 0; i < T_NXMM; i++) {
        uint8_t nv[16]; t_pattern_reg(nv, 16, g_cap.tag, (uint32_t)i);
        uint8_t *p = ctx_xmm(c, i); if (p) { memcpy(p, nv, 16); memcpy(f->vexp[i], nv, 16); }
    }
    static const int gi[] = { 0, 6, 12 };
    for (size_t j = 0; j < T_COUNTOF(gi); j++) {
        if (gi[j] >= NGPR) continue;
        uintptr_t v = (uintptr_t)t_pattern_word(g_cap.tag, (uint32_t)gi[j], 15);
        *ctx_gpr(c, gi[j]) = v; f->gexp[gi[j]] = v;
    }
    ctx_set_mxcsr(c, 0x7F80); f->mxexp = 0x7F80;
    if (g_cap.ymm && g_locate && g_setmask && g_getmask) {
        DWORD len = 0; uint8_t *p = (uint8_t *)g_locate(c, XSF_AVX, &len);
        if (p && len >= 16u * T_NXMM) {
            for (int i = 0; i < T_NXMM; i++) {
                uint8_t nv[32]; t_pattern_reg(nv, 32, g_cap.tag + 1, (uint32_t)i);
                memcpy(p + 16 * i, nv + 16, 16); memcpy(f->vexp[i] + 16, nv + 16, 16);
            }
            DWORD64 m = 0; g_getmask(c, &m); g_setmask(c, m | XSM_AVX);
        }
    }
    if (g_cap.how == 1) {
#ifdef T_X64
        pfn_restore rr = (pfn_restore)t_proc("kernel32.dll", "RtlRestoreContext");
        if (rr) rr(c, NULL);
#endif
    } else if (g_cap.how == 2) {
        pfn_ntcontinue nc = (pfn_ntcontinue)t_proc("ntdll.dll", "NtContinue");
        if (nc) nc(c, FALSE);
    }
}
static CONTEXT *ctx_alloc_xstate(void **buf, int xstate) {
    *buf = NULL;
    if (!xstate) {
        CONTEXT *c = (CONTEXT *)_aligned_malloc(sizeof(CONTEXT), 64);
        if (c) memset(c, 0, sizeof *c);
        *buf = c;
        return c;
    }
    if (!g_initctx || !g_setmask) return NULL;
    DWORD len = 0;
    g_initctx(NULL, CONTEXT_ALL | X_CONTEXT_XSTATE, NULL, &len);
    if (!len) return NULL;
    void *b = calloc(1, len + 64);
    CONTEXT *c = NULL;
    if (!b || !g_initctx(b, CONTEXT_ALL | X_CONTEXT_XSTATE, &c, &len) || !c) { free(b); return NULL; }
    if (!g_setmask(c, XSM_AVX)) { free(b); return NULL; }
    *buf = b;
    return c;
}
static void ctx_free_xstate(void *buf, int xstate) { if (xstate) free(buf); else _aligned_free(buf); }

static void cap_reference(int use2, int how) {
    xs_init();
    int ymm = use2 && cpu_has(F_AVX);
    pfn_capture cap = (pfn_capture)t_proc("kernel32.dll", "RtlCaptureContext");
    if (use2) cap = (pfn_capture)t_proc("ntdll.dll", "RtlCaptureContext2");
    if (!cap) SKIP("%s not exported", use2 ? "RtlCaptureContext2" : "RtlCaptureContext");
    void *buf;
    CONTEXT *c = ctx_alloc_xstate(&buf, use2);
    if (!c) SKIP("InitializeContext(CONTEXT_XSTATE) / SetXStateFeaturesMask unavailable");
    if (!use2) c->ContextFlags = CONTEXT_ALL;
    xf *f = xf_alloc(); REQUIRE(f);
    xf_init(f, 0xB100u + (uint32_t)use2 * 16u + (uint32_t)how, ymm ? 32 : 16, 0, 0);
    f->ctxp = (uint64_t)(uintptr_t)c; f->capfn = (uint64_t)(uintptr_t)cap;
    g_cap.how = how; g_cap.tag = 0xB800u + (uint32_t)how; g_cap.f = f; g_cap.ymm = ymm;
#ifdef T_X64
    f->cb = how ? (uint64_t)(uintptr_t)cap_cb : 0;
#else
    /* 32-bit RtlCaptureContext walks the EBP chain (Ebp/Eip/Esp of the caller's caller): give it a frame */
    static uintptr_t fake_frame[4];
    fake_frame[0] = (uintptr_t)&fake_frame[2]; fake_frame[1] = (uintptr_t)xcap_ret;
    f->gin[5] = (uintptr_t)fake_frame;
    xf_sync_exp(f);
#endif
#ifdef T_X64
    if (how) f->gexp[1] = (uint64_t)(uintptr_t)c;     /* RCX held the CONTEXT pointer when it was captured */
#endif
    xcap_probe(f);
    /* ---- the captured CONTEXT vs the reference ---- */
    t_info("context_flags", "0x%lx", (unsigned long)c->ContextFlags);
#ifdef T_X64
    CHECKF((c->ContextFlags & 0x10000Fu) == 0x10000Fu, "captured ContextFlags 0x%lx lack CONTROL|INTEGER|SEGMENTS|FLOATING_POINT",
           (unsigned long)c->ContextFlags);
    CHECKF(c->Rip == (DWORD64)(uintptr_t)xcap_ret, "captured Rip %p want the return address %p", (void *)(uintptr_t)c->Rip, (const void *)xcap_ret);
    CHECKF(c->Rsp == f->rsp_at, "captured Rsp 0x%llx want the caller's RSP 0x%llx", (unsigned long long)c->Rsp, (unsigned long long)f->rsp_at);
    CHECKF(c->Rcx == (DWORD64)(uintptr_t)c, "captured Rcx 0x%llx want the CONTEXT pointer", (unsigned long long)c->Rcx);
    CHECKF(c->Rbx == (DWORD64)(uintptr_t)f, "captured Rbx 0x%llx want the frame", (unsigned long long)c->Rbx);
    for (int i = 0; i < 16; i++) {
        if (i == 1 || i == 3 || i == 4) continue;
        if (how && (i == 0 || i == 6 || i == 12)) continue;      /* edited by the callback */
        CHECKF((&c->Rax)[i] == f->gin[i], "captured %s 0x%llx want 0x%llx", g_gname[i], (unsigned long long)(&c->Rax)[i],
               (unsigned long long)f->gin[i]);
    }
    if (!how) {
        CHECKF(c->MxCsr == f->mxcsr_in, "captured MxCsr 0x%lx want 0x%x", (unsigned long)c->MxCsr, f->mxcsr_in);
        CHECKF(c->FltSave.MxCsr == f->mxcsr_in, "captured FltSave.MxCsr 0x%lx want 0x%x", (unsigned long)c->FltSave.MxCsr, f->mxcsr_in);
        for (int i = 0; i < 16; i++) {
            char what[48]; snprintf(what, sizeof what, "captured CONTEXT xmm%d", i);
            CHECK_REG(what, &c->FltSave.XmmRegisters[i], f->vin[i], 16);
        }
        if (ymm) {
            CHECKF((c->ContextFlags & X_CONTEXT_XSTATE) == X_CONTEXT_XSTATE, "RtlCaptureContext2 did not capture XSTATE (flags 0x%lx)",
                   (unsigned long)c->ContextFlags);
            DWORD64 m = 0; DWORD len = 0;
            CHECK(g_getmask(c, &m));
            CHECKF(m & XSM_AVX, "captured XSTATE mask 0x%llx lacks AVX", (unsigned long long)m);
            uint8_t *p = (uint8_t *)g_locate(c, XSF_AVX, &len);
            CHECKF(p && len >= 256, "LocateXStateFeature(AVX) %p len %lu", (void *)p, (unsigned long)len);
            if (p && len >= 256) for (int i = 0; i < 16; i++) {
                char what[48]; snprintf(what, sizeof what, "captured ymm%d[255:128]", i);
                CHECK_REG(what, p + 16 * i, f->vin[i] + 16, 16);
            }
        }
        CHECK_EQ(f->pass, 2);
    } else {
        CHECKF(f->pass == 1, "%s did not resume at the captured return address (pass %llu)", how == 1 ? "RtlRestoreContext" : "NtContinue",
               (unsigned long long)f->pass);
        if (f->pass == 1) {
            if (!ymm && (f->flags & XF_YMM)) for (int i = 0; i < 16; i++) memcpy(f->vexp[i] + 16, f->vout[i] + 16, 16);
            xf_check_out(f, how == 1 ? "after RtlRestoreContext of an edited CONTEXT" : "after NtContinue with an edited CONTEXT");
        }
    }
#else
    CHECKF((c->ContextFlags & 0x10003u) == 0x10003u, "captured ContextFlags 0x%lx lack CONTROL|INTEGER", (unsigned long)c->ContextFlags);
    CHECKF(c->Ebx == (DWORD)(uintptr_t)f, "captured Ebx 0x%lx want the frame", (unsigned long)c->Ebx);
    CHECKF(c->Esi == (DWORD)f->gin[6], "captured Esi 0x%lx want 0x%lx", (unsigned long)c->Esi, (unsigned long)f->gin[6]);
    CHECKF(c->Edi == (DWORD)f->gin[7], "captured Edi 0x%lx want 0x%lx", (unsigned long)c->Edi, (unsigned long)f->gin[7]);
    t_info("x86_capture", "Eip=label%+ld Esp=rsp_at%+ld Ebp=%s", (long)((intptr_t)c->Eip - (intptr_t)xcap_ret),
           (long)((intptr_t)c->Esp - (intptr_t)f->rsp_at), c->Ebp == (DWORD)f->gin[5] ? "pattern" : "other");
    f->gskip = 0x07u; f->vskip = 0;
    xf_check_out(f, "after RtlCaptureContext (x86 registers untouched)");
#endif
    xf_free(f);
    ctx_free_xstate(buf, use2);
}
TEST(cap_rtlcapturecontext_reference) { cap_reference(0, 0); }
TEST(cap_rtlcapturecontext2_xstate_x64) {
#ifdef T_X64
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    cap_reference(1, 0);
#else
    SKIP("RtlCaptureContext2 is x64 only");
#endif
}
TEST(cap_rtlrestorecontext_edited_x64) {
#ifdef T_X64
    cap_reference(0, 1);
#else
    SKIP("RtlRestoreContext is not exported on x86");
#endif
}
TEST(cap_ntcontinue_edited_x64) {
#ifdef T_X64
    cap_reference(0, 2);
#else
    SKIP("x86 NtContinue path is covered by exception continuation");
#endif
}
TEST(cap_rtlrestorecontext_ymm_x64) {
#ifdef T_X64
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    cap_reference(1, 1);
#else
    SKIP("x64 only");
#endif
}
TEST(cap_ntcontinue_ymm_x64) {
#ifdef T_X64
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    cap_reference(1, 2);
#else
    SKIP("x64 only");
#endif
}

/* ========================================================================================== */
/* x87 unmasked exception matrix (IE, DE, OE, UE, PE; ZE and stack fault above)                */
/* ========================================================================================== */
static void f80_raw(uint8_t out[16], int sign, uint16_t exp, uint64_t mant) {
    uint16_t se = (uint16_t)((sign ? 0x8000u : 0u) | exp);
    memset(out, 0, 16); memcpy(out, &mant, 8); memcpy(out + 8, &se, 2);
}
/* Codes per ntoskrnl!KiFloatingErrorFault (19041, checked in the disassembly): unmasked flags = SW & ~CW & 0x3F;
 * IE -> SF ? 0xC0000092 : 0xC0000090, else ZE 0xC000008E, DE 0xC0000090 (not _DENORMAL_OPERAND), OE 0xC0000091,
 * UE 0xC0000093, PE 0xC000008F; NumberParameters = 1, Information[0] = 0, address = RIP of the waiting
 * instruction. KiFloatingErrorFault does not look at CS, so WOW64 code gets the same codes; native 32-bit
 * kernels are not pinned, hence the x86 alternatives. Per the SDM, IE/DE are pre-computation (destination
 * unchanged); OE/UE with a register destination store the exponent-scaled result; PE stores the rounded result. */
static const xk *x87_setup_fpe(fpe_t which, xf *f, xexp *e) {
    const xk *k = xk_find(which == FPE_IE ? "x87_sqrt" : which == FPE_PE ? "x87_div" : "x87_mul");
    xk_prepare(k, f, e);
    f->flags |= XF_X87;
    uint16_t bit = 0; DWORD code = 0;
    switch (which) {
    case FPE_IE: f80_from_double(f->x87in[0], -1.0); bit = 0x01; code = XS_FLT_INVALID; e->x87_sw_clear = 0x40; break;
    case FPE_DE: f80_from_double(f->x87in[0], 1.0); f80_raw(f->x87in[1], 0, 0, 1); bit = 0x02; code = XS_FLT_INVALID;
                 e->alt[0] = XS_FLT_DENORMAL; break;
    case FPE_OE: f80_raw(f->x87in[0], 0, 0x7FFE, ~0ull); f80_from_double(f->x87in[1], 2.0); bit = 0x08; code = XS_FLT_OVERFLOW;
                 e->x87_st0 = 1; break;
    case FPE_UE: f80_raw(f->x87in[0], 0, 0x0001, 1ull << 63); f80_from_double(f->x87in[1], 0.5); bit = 0x10; code = XS_FLT_UNDERFLOW;
                 e->x87_st0 = 1; break;
    case FPE_PE: f80_from_double(f->x87in[0], 1.0); f80_from_double(f->x87in[1], 3.0); bit = 0x20; code = XS_FLT_INEXACT;
                 /* PC = 53 bits, RC = nearest: the stored quotient is exactly the double 1/3 */
                 e->x87_st0 = 2; { uint8_t t[16]; f80_from_double(t, 1.0 / 3.0); memcpy(e->x87_st0_val, t, 10); } break;
    default: break;
    }
    f->fcw_in = (uint16_t)(0x027F & ~bit);
    xf_sync_exp(f);
    if (e->x87_st0 == 2) memcpy(f->x87exp[0], e->x87_st0_val, 10);
    e->code = code;
#ifdef T_X64
    e->nparams = 1; e->ninfo = 1; e->info[0] = 0;
#else
    e->alt[1] = XS_FLT_MULTIPLE_TRAPS; e->alt[2] = XS_FLT_MULTIPLE_FAULTS;
    e->addr_soft = 1;
#endif
    e->x87_sw_set = (uint16_t)(bit | 0x80);
    e->check_fip = 1;
    e->resume = RS_STAY;
    return k;
}
static void fp_x87_fpe_case(fpe_t which) {
#ifdef T_X64
    if (which != FPE_DE) wine_gap("x87 #MF exception record has NumberParameters=0 (Windows x64 KiFloatingErrorFault: 1)");
#endif
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6500u + (uint32_t)which, cpu_has(F_AVX) ? 32 : 16, 1, 0);
    const xk *k = x87_setup_fpe(which, f, &s->e);
    s->mode = M_FPFIX; s->veh_on = 1;
    char lab[80]; snprintf(lab, sizeof lab, "after unmasked x87 %s (status cleared in CONTEXT)", g_fpe_name[which]);
    CHECKF(xk_arm_and_run(s, k, f, V_VEH) == 1, "%s: handler ran %ld times", lab, (long)s->calls);
    t_info("code", "0x%08lx np=%lu", (unsigned long)s->rec.ExceptionCode, (unsigned long)s->rec.NumberParameters);
#ifdef T_X86
    CHECKF(s->rec.ExceptionAddress == (PVOID)k->at || s->rec.ExceptionAddress == (PVOID)k->op,
           "x87 exception address %p is neither the waiting instruction %p nor the x87 op %p", s->rec.ExceptionAddress,
           (const void *)k->at, (const void *)k->op);
#endif
    if (s->e.x87_st0 == 1) {
        /* the scaled result found in the CONTEXT is what must come back after the resume */
        char a[48];
        t_info("st0_scaled_result", "%s", t_hex(a, sizeof a, s->ctx_st0, 10));
        memcpy(f->x87exp[0], s->ctx_st0, 10);
    }
    xf_check_out(f, lab);
    CHECKF((f->fsw_out & 0x80BF) == 0, "x87 SW after resume 0x%04x still has exception/ES bits", f->fsw_out);
    xf_free(f);
}
TEST(fp_x87_invalid_unmasked) { fp_x87_fpe_case(FPE_IE); }
TEST(fp_x87_denormal_unmasked) {
#ifdef T_X64
    wine_gap("x87 #MF exception record has NumberParameters=0 and only-DE-unmasked is reported as STATUS_FLOAT_UNDERFLOW");
#endif
    fp_x87_fpe_case(FPE_DE);
}
TEST(fp_x87_overflow_unmasked) { fp_x87_fpe_case(FPE_OE); }
TEST(fp_x87_underflow_unmasked) { fp_x87_fpe_case(FPE_UE); }
TEST(fp_x87_inexact_unmasked) { fp_x87_fpe_case(FPE_PE); }

/* ========================================================================================== */
/* VEX-encoded (AVX) faulting instructions                                                    */
/* ========================================================================================== */
/* vdivps ymm: unmasked #XM raised by a 256-bit VEX instruction. Same KiXmmException mapping as SSE; the
 * destination is unchanged at the fault; after masking in the CONTEXT the re-executed instruction writes all 8 lanes. */
static void fp_avx_case(int invalid) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6600u + (uint32_t)invalid, 32, 0, 0);
    const xk *k = xk_find("avx_div");
    xexp *e = &s->e;
    xk_prepare(k, f, e);
    uint32_t a = invalid ? 0u : f32bits(1.0f), r = invalid ? 0xFFC00000u : 0x7F800000u, bit = invalid ? 0x01u : 0x04u;
    for (int i = 0; i < 8; i++) { memcpy(f->vin[0] + 4 * i, &a, 4); memset(f->vin[1] + 4 * i, 0, 4); }
    f->mxcsr_in = 0x1F80u & ~(bit << 7);
    xf_sync_exp(f);
    for (int i = 0; i < 8; i++) memcpy(f->vexp[0] + 4 * i, &r, 4);
    f->mxexp = 0x1F80u | bit;
#ifdef T_X64
    e->code = invalid ? XS_FLT_INVALID : XS_FLT_DIVIDE_BY_ZERO;
    e->nparams = 2; e->ninfo = 1; e->info[0] = 0;
#else
    e->code = XS_FLT_MULTIPLE_TRAPS; e->alt[0] = invalid ? XS_FLT_INVALID : XS_FLT_DIVIDE_BY_ZERO;
#endif
    e->mx_set = bit; e->resume = RS_STAY;
    s->mode = M_FPFIX; s->veh_on = 1;
    const char *lab = invalid ? "after unmasked VEX vdivps 0/0, masked in CONTEXT, re-executed" :
                                "after unmasked VEX vdivps 1/0, masked in CONTEXT, re-executed";
    CHECKF(xk_arm_and_run(s, k, f, V_VEH) == 1, "%s: handler ran %ld times", lab, (long)s->calls);
    t_info("code", "0x%08lx np=%lu", (unsigned long)s->rec.ExceptionCode, (unsigned long)s->rec.NumberParameters);
    xf_check_out(f, lab);
    xf_free(f);
}
TEST(fp_avx_zero_divide_unmasked_ymm) { fp_avx_case(0); }
TEST(fp_avx_invalid_unmasked_ymm) { fp_avx_case(1); }
TEST(veh_avx_misaligned_vmovaps_ymm) {
    /* VEX.256 vmovaps needs 32-byte alignment: #GP -> STATUS_ACCESS_VIOLATION (KiPreprocessFault) */
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6701, 32, 0, 1);
    const xk *k = xk_find("avx_movmis");
    xk_prepare(k, f, &s->e);
    f->gin[1] = (uintptr_t)(g_mis_buf + 16); xf_sync_exp(f);
    s->e.code = XS_ACCESS_VIOLATION; s->e.alt[0] = XS_MISALIGNMENT;
    s->veh_on = 1; s->mode = M_MOD_YMM;
    s->tag2 = 0x6777;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_VEH), 1);
    t_info("code", "0x%08lx np=%lu info0=0x%llx info1=0x%llx", (unsigned long)s->rec.ExceptionCode, (unsigned long)s->rec.NumberParameters,
           (unsigned long long)s->rec.ExceptionInformation[0], (unsigned long long)s->rec.ExceptionInformation[1]);
    if (!s->had_xstate) for (int i = 0; i < 16; i++) memcpy(f->vexp[i] + 16, f->vin[i] + 16, 16);
    xf_check_out(f, "after misaligned VEX.256 vmovaps (YMM uppers edited in CONTEXT)");
    xf_free(f);
}
TEST(veh_avx_load_straddle_guard_ymm) {
    /* 32-byte VEX load whose last 16 bytes are in a PAGE_NOACCESS page: read AV; the reported address lies
     * inside the access (the CPU reports the first inaccessible byte); the destination is unchanged. */
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    void *base = NULL;
    uint8_t *p = (uint8_t *)t_guarded_alloc(64, &base);
    if (!p) SKIP("t_guarded_alloc failed");
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh; memset(s, 0, sizeof *s);
    xf_init(f, 0x6801, 32, 0, 0);
    const xk *k = xk_find("avx_ld");
    xk_prepare(k, f, &s->e);
    uint8_t *ld = p + 64 - 16;
    f->gin[1] = (uintptr_t)ld; xf_sync_exp(f);
    s->e.code = XS_ACCESS_VIOLATION; s->e.nparams = 2; s->e.ninfo = 1; s->e.info[0] = 0;
    s->veh_on = 1;
    CHECK_EQ(xk_arm_and_run(s, k, f, V_VEH), 1);
    uintptr_t a = (uintptr_t)s->rec.ExceptionInformation[1];
    CHECKF(a >= (uintptr_t)ld && a < (uintptr_t)ld + 32, "AV address 0x%llx outside the 32-byte access at %p",
           (unsigned long long)a, (void *)ld);
    t_info("av_address", "access%+ld (page boundary at +16)", (long)(a - (uintptr_t)ld));
    xf_check_out(f, "after a page-straddling VEX.256 load fault");
    xf_free(f);
    t_guarded_free(base);
}

/* ========================================================================================== */
/* Nested hardware fault inside a handler                                                      */
/* ========================================================================================== */
TEST(veh_nested_hw_fault_in_handler) {
    for (int vk = 16; vk <= 64; vk *= 2) {
        if (!vk_supported(vk)) continue;
        char lab[80]; snprintf(lab, sizeof lab, "outer state after a nested hardware fault in the VEH (%s)", vk_name(vk));
        xk_test("div0", vk, M_NESTED_HW | M_MOD_XMM, V_VEH, tw_x87_efl, lab);
        CHECK_EQ(g_main_xh.nested_hw_calls, 1);
    }
}
TEST(seh_nested_hw_fault_in_frame_handler) {
    int vk = cpu_has(F_AVX) ? 32 : 16;
    xk_test("div0", vk, M_NESTED_HW, V_SEHCALL, NULL, "outer state after a nested hardware fault in the frame handler");
    CHECK_EQ(g_main_xh.nested_hw_calls, 1);
#ifdef T_X64
    xk_test("rd", vk, M_NESTED_HW | M_MOD_GPR, V_SEHFRAME, NULL, "outer state after a nested fault (handler on the faulting frame)");
    CHECK_EQ(g_main_xh.nested_hw_calls, 1);
#endif
}

/* ========================================================================================== */
/* Top-level (unhandled) exception filter that continues execution (child process)            */
/* ========================================================================================== */
TEST(child_unhandled_filter_continue) {
    DWORD c = 0;
    CHECKF(run_child_kind("div0", 16, M_MOD_XMM | M_MOD_GPR, V_NONE, &c) == 1, "filter did not run exactly once");
    CHECK_EQ(c, XS_INT_DIVIDE_BY_ZERO);
    CHECKF(run_child_kind("rd", 16, M_MOD_MXCSR, V_NONE, &c) == 1, "filter did not run exactly once");
    CHECK_EQ(c, XS_ACCESS_VIOLATION);
    if (cpu_has(F_AVX)) CHECKF(run_child_kind("div0", 32, M_MOD_YMM | M_CLOBBER, V_NONE, &c) == 1, "filter did not run exactly once (ymm)");
}

/* ========================================================================================== */
/* STATUS_STACK_OVERFLOW with live vector state (child process, worker with a 1 MB reserve)    */
/* ========================================================================================== */
static void tw_stkovf(xf *f, xexp *e, xh *s) {
    (void)s;
    e->code = XS_STACK_OVERFLOW; e->nparams = -1; e->sp_soft = 1;
    e->gany = 1u << 1;            /* ecx/rcx = loop counter */
    f->gskip = 1u << 1;
}
static volatile LONG g_so_guarantee_ok;
static DWORD WINAPI stkovf_thread(void *p) {
    int vk = *(int *)p;
    ULONG g = 64 * 1024;
    g_so_guarantee_ok = SetThreadStackGuarantee(&g) ? 1 : 0;
    printf("SO_GUARANTEE ok=%ld err=%lu\n", (long)g_so_guarantee_ok, (unsigned long)GetLastError()); fflush(stdout);
    int calls = xk_test("stkovf", vk, M_RESET_SP | M_MOD_XMM, V_VEH, tw_stkovf, "after STATUS_STACK_OVERFLOW (RSP reset by handler)");
    EXCEPTION_RECORD *r = &g_main_xh.rec;
    printf("XK_CALLS=%d CODE=0x%08lx\n", calls, (unsigned long)r->ExceptionCode);
    printf("SO_INFO np=%lu info0=0x%llx info1=0x%llx guarantee=%ld\n", (unsigned long)r->NumberParameters,
           (unsigned long long)(r->NumberParameters > 0 ? r->ExceptionInformation[0] : 0),
           (unsigned long long)(r->NumberParameters > 1 ? r->ExceptionInformation[1] : 0), (long)g_so_guarantee_ok);
    fflush(stdout);
    return 0;
}
CHILD(stkovf) {
    if (argc < 2) return 3;
    g_in_child = 1;
    xs_init();
    static int vk; vk = atoi(argv[1]);
    HANDLE h = CreateThread(NULL, 1u << 20, stkovf_thread, &vk, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!h) { printf("XK_SKIP=CreateThread failed (%lu)\n", (unsigned long)GetLastError()); return 0; }
    DWORD w = WaitForSingleObject(h, 15000);
    CloseHandle(h);
    if (w != WAIT_OBJECT_0) { printf("SO_HANG\n"); fflush(stdout); TerminateProcess(GetCurrentProcess(), 0x5A); }
    return 0;
}
TEST(child_stack_overflow_context) {
    /* Wine 11: STATUS_STACK_OVERFLOW is raised only in the very last stack page (the SetThreadStackGuarantee
     * reserve is not kept back), the exception frame does not fit, and the thread is killed (x64) or the process
     * hangs (WOW64). Not run there: it would only burn the child timeout. */
    if (t_is_wine()) XFAIL("Wine gap: STATUS_STACK_OVERFLOW delivered without the guaranteed stack; handler cannot run");
    int vk = cpu_has(F_AVX512F) ? 64 : cpu_has(F_AVX) ? 32 : 16;
    char a[16]; snprintf(a, sizeof a, "%d", vk);
    if (!t_child_run("stkovf", a, 20000, &g_cr) || !g_cr.started) { CHECKF(0, "child spawn failed (error %lu)", (unsigned long)g_cr.spawn_error); return; }
    CHECKF(!g_cr.timed_out, "child timed out");
    child_forward(&g_cr, "stkovf");
    char v[160];
    if (t_child_find(&g_cr, "XK_SKIP=", v, sizeof v)) SKIP("%s", v);
    CHECKF(!t_child_find(&g_cr, "SO_HANG", v, sizeof v), "stack-overflow worker did not finish");
    int calls = -1; unsigned long code = 0;
    if (t_child_find(&g_cr, "XK_CALLS=", v, sizeof v)) sscanf(v, "%d CODE=0x%lx", &calls, &code);
    CHECKF(calls == 1, "handler ran %d times (exit 0x%08lx, exception 0x%08lx at 0x%llx)", calls, (unsigned long)g_cr.exit_code,
           (unsigned long)g_cr.exc_code, g_cr.exc_addr);
    CHECKF(code == XS_STACK_OVERFLOW, "exception 0x%08lx want STATUS_STACK_OVERFLOW", code);
    CHECKF(g_cr.exit_code == 0, "child exit code 0x%08lx", (unsigned long)g_cr.exit_code);
    if (t_child_find(&g_cr, "SO_INFO ", v, sizeof v)) t_info("record", "%s", v);
    t_info("vk", "%s", vk_name(vk));
}

/* ========================================================================================== */
/* Hardware execution breakpoint (DR0 set with SetThreadContext) on a worker                   */
/* ========================================================================================== */
typedef struct hwbp { int vk; int calls[3]; } hwbp;
static void tw_hwbp(xf *f, xexp *e, xh *s) {
    (void)f; (void)s;
    e->code = XS_SINGLE_STEP; e->resume = RS_STAY; e->check_dr = 1;
}
static DWORD WINAPI hwbp_worker(void *p) {
    hwbp *h = (hwbp *)p;
    /* run 0: handler sets RF -> the nop executes once, DR0 stays armed; run 1: handler clears Dr7 in the
     * CONTEXT (and sets RF); run 2: no exception expected any more */
    static const int modes[3] = { M_SET_RF, M_SET_RF | M_CLEAR_DR7, 0 };
    for (int r = 0; r < 3; r++) {
        char lab[80]; snprintf(lab, sizeof lab, "after DR0 execute breakpoint run %d", r);
        h->calls[r] = xk_test("nop", h->vk, modes[r] | (r == 0 ? M_MOD_XMM : 0), V_VEH, r < 2 ? tw_hwbp : NULL, lab);
    }
    return 0;
}
TEST(veh_hw_breakpoint_dr0_rf) {
    xs_init();
    hwbp h; memset(&h, 0, sizeof h);
    h.vk = cpu_has(F_AVX) ? 32 : 16;
    HANDLE t = CreateThread(NULL, 256 * 1024, hwbp_worker, &h, CREATE_SUSPENDED, NULL);
    if (!t) SKIP("CreateThread failed");
    CONTEXT ctx; memset(&ctx, 0, sizeof ctx);
    ctx.ContextFlags = X_CONTEXT_DEBUG;
    BOOL ok = GetThreadContext(t, &ctx);
    if (ok) {
        ctx.ContextFlags = X_CONTEXT_DEBUG;
        ctx.Dr0 = (uintptr_t)xk_find("nop")->at;
        ctx.Dr7 = 1;                       /* L0, R/W0 = 00 (execute), LEN0 = 00 */
        ok = SetThreadContext(t, &ctx);
    }
    if (!ok) {
        DWORD err = GetLastError();
        TerminateThread(t, 0); CloseHandle(t);
        CHECKF(0, "Get/SetThreadContext(CONTEXT_DEBUG_REGISTERS) on a suspended thread failed (error %lu)", (unsigned long)err);
        return;
    }
    if (t_is_wine()) {
        CONTEXT chk; memset(&chk, 0, sizeof chk); chk.ContextFlags = X_CONTEXT_DEBUG;
        if (GetThreadContext(t, &chk) && chk.Dr7 == 0) wine_gap("debug registers written with SetThreadContext are not kept");
    }
    ResumeThread(t);
    DWORD w = WaitForSingleObject(t, 15000);
    CHECKF(w == WAIT_OBJECT_0, "breakpoint worker did not finish (wait 0x%lx)", (unsigned long)w);
    if (w != WAIT_OBJECT_0) TerminateThread(t, 0);
    CloseHandle(t);
    CHECKF(h.calls[0] == 1, "run 0: DR0 execute breakpoint raised %d exceptions, want 1", h.calls[0]);
    CHECKF(h.calls[1] == 1, "run 1: DR0 still armed after an RF resume: %d exceptions, want 1", h.calls[1]);
    CHECKF(h.calls[2] == 0, "run 2: DR7 cleared in the CONTEXT did not take effect (%d exceptions)", h.calls[2]);
}

/* ========================================================================================== */
/* Stress / perf                                                                                */
/* ========================================================================================== */
TEST(stress_random_patterns_main_thread) {
    xs_init();
    static const char *const kinds[] = { "div0", "rd", "wr", "int3", "idivov", "nop" };
    t_rng r; t_rng_seed(&r, 0xE5C0DEull);
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh;
    int vmax = cpu_has(F_AVX512F) ? 3 : cpu_has(F_AVX) ? 2 : 1;
    long ex = 0;
    for (int it = 0; it < 300; it++) {
        const xk *k = xk_find(kinds[t_rng_range(&r, T_COUNTOF(kinds))]);
        int vk = 16 << t_rng_range(&r, (uint64_t)vmax);
        memset(s, 0, sizeof *s);
        xf_init(f, t_rng_u32(&r) | 1u, vk, (int)t_rng_range(&r, 2), (int)t_rng_range(&r, 2));
        xk_prepare(k, f, &s->e);
        static const int modes[] = { 0, M_MOD_XMM, M_MOD_GPR, M_CLOBBER, M_MOD_MXCSR, M_MOD_XMM | M_MOD_GPR | M_CLOBBER };
        s->mode = modes[t_rng_range(&r, T_COUNTOF(modes))];
        if (vk >= 32 && t_rng_range(&r, 3) == 0) s->mode |= M_MOD_YMM;
        s->tag2 = t_rng_u32(&r); s->veh_on = 1; s->quiet = 1;
        int variant = V_VEH;
#ifdef T_X64
        if (k->sfn && t_rng_range(&r, 2)) { variant = V_SEHFRAME; s->veh_on = 0; }
#endif
        int calls = xk_arm_and_run(s, k, f, variant);
        CHECKF(calls == (s->e.code ? 1 : 0), "iteration %d %s: calls %d", it, k->name, calls);
        char lab[64]; snprintf(lab, sizeof lab, "stress it %d %s %s mode 0x%x", it, k->name, vk_name(vk), s->mode);
        if (s->mode & M_MOD_YMM && !s->had_xstate) for (int i = 0; i < 16; i++) memcpy(f->vexp[i] + 16, f->vin[i] + 16, 16);
        xf_check_out(f, lab);
        ex += calls;
    }
    t_info("exceptions", "%ld", ex);
    xf_free(f);
}
TEST(perf_exception_roundtrip) {
    xs_init();
    xf *f = xf_alloc(); REQUIRE(f);
    xh *s = &g_main_xh;
    const char *what[2] = { "div0_veh_xmm", "div0_veh_ymm" };
    for (int v = 0; v < 2; v++) {
        if (v == 1 && !cpu_has(F_AVX)) break;
        const xk *k = xk_find("div0");
        double samples[7];
        for (int rep = 0; rep < 7; rep++) {
            memset(s, 0, sizeof *s);
            xf_init(f, 0xC100, v ? 32 : 16, 0, 0); xk_prepare(k, f, &s->e);
            s->mode = M_NOVERIFY; s->veh_on = 1; s->quiet = 1;
            int64_t t0 = t_qpc();
            int n = 0;
            for (int i = 0; i < 200; i++) n += xk_arm_and_run(s, k, f, V_VEH);
            int64_t t1 = t_qpc();
            CHECKF(n == 200, "%d of 200 exceptions handled", n);
            xf_check_out(f, "after the last timed round trip");
            samples[rep] = 200.0 / t_qpc_sec(t0, t1);
        }
        t_stats st; t_stats_compute(samples, 7, &st);
        t_perf_stats(what[v], &st, "exceptions/s");
    }
    xf_free(f);
}
