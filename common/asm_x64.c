/* asm_x64.c - x64 register-file routines (Windows x64 ABI: args RCX, RDX, R8, R9, 5th at [RSP+40]
 * on entry; RBX/RBP/RDI/RSI/R12-R15 and XMM6-15 (low 128 bits) are callee-saved).
 * Every routine has SEH unwind info (.seh_*), so exceptions raised inside a callback unwind
 * correctly through it. Layout of t_vregs: v[i] at i*64, k[i] at 2048 + i*8. */
#include "t.h"
#ifdef T_X64

/* prologue/epilogue shared by all routines: saves rbx, rsi, rdi, rbp, xmm6-15; 32-byte shadow space */
#define PRO(name) \
    ".globl " #name "\n.def " #name "; .scl 2; .type 32; .endef\n.seh_proc " #name "\n" #name ":\n" \
    "  pushq %rbp\n  .seh_pushreg %rbp\n  pushq %rbx\n  .seh_pushreg %rbx\n" \
    "  pushq %rsi\n  .seh_pushreg %rsi\n  pushq %rdi\n  .seh_pushreg %rdi\n" \
    "  subq $200, %rsp\n  .seh_stackalloc 200\n" \
    "  movdqa %xmm6, 32(%rsp)\n  .seh_savexmm %xmm6, 32\n  movdqa %xmm7, 48(%rsp)\n  .seh_savexmm %xmm7, 48\n" \
    "  movdqa %xmm8, 64(%rsp)\n  .seh_savexmm %xmm8, 64\n  movdqa %xmm9, 80(%rsp)\n  .seh_savexmm %xmm9, 80\n" \
    "  movdqa %xmm10, 96(%rsp)\n  .seh_savexmm %xmm10, 96\n  movdqa %xmm11, 112(%rsp)\n  .seh_savexmm %xmm11, 112\n" \
    "  movdqa %xmm12, 128(%rsp)\n  .seh_savexmm %xmm12, 128\n  movdqa %xmm13, 144(%rsp)\n  .seh_savexmm %xmm13, 144\n" \
    "  movdqa %xmm14, 160(%rsp)\n  .seh_savexmm %xmm14, 160\n  movdqa %xmm15, 176(%rsp)\n  .seh_savexmm %xmm15, 176\n" \
    "  .seh_endprologue\n"
#define EPI \
    "  movdqa 32(%rsp), %xmm6\n  movdqa 48(%rsp), %xmm7\n  movdqa 64(%rsp), %xmm8\n  movdqa 80(%rsp), %xmm9\n" \
    "  movdqa 96(%rsp), %xmm10\n  movdqa 112(%rsp), %xmm11\n  movdqa 128(%rsp), %xmm12\n  movdqa 144(%rsp), %xmm13\n" \
    "  movdqa 160(%rsp), %xmm14\n  movdqa 176(%rsp), %xmm15\n" \
    "  addq $200, %rsp\n  popq %rdi\n  popq %rsi\n  popq %rbx\n  popq %rbp\n  ret\n.seh_endproc\n"
/* 5th argument: entry [rsp+40] + 4 pushes (32) + 200 = 272 */
#define ARG5 "272(%rsp)"

#define R16 "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#define R32 R16 ",16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
#define R8  "0,1,2,3,4,5,6,7"

/* spin loop: eax = remaining; r8 = stop flag (32-bit LONG) or NULL. Ends at label 4. */
#define SPIN \
    "  movl %r9d, %eax\n  testl %eax, %eax\n  jz 4f\n" \
    "2: testq %r8, %r8\n  jz 3f\n  cmpl $0, (%r8)\n  jne 4f\n" \
    "3: pause\n  decl %eax\n  jnz 2b\n4:\n"
/* call: rbx = out, rsi = fn, rdi = arg */
#define CALLFN \
    "  testq %rsi, %rsi\n  jz 1f\n  movq %rdi, %rcx\n  callq *%rsi\n1:\n"

#define KLOAD(src) \
    "  testl %ebp, %ebp\n  jz 5f\n" \
    "  .irp i," R8 "\n  kmovq 2048+\\i*8(" src "), %k\\i\n  .endr\n  jmp 6f\n" \
    "5:\n  .irp i," R8 "\n  kmovw 2048+\\i*8(" src "), %k\\i\n  .endr\n6:\n"
#define KSTORE(dst) \
    "  testl %ebp, %ebp\n  jz 7f\n" \
    "  .irp i," R8 "\n  kmovq %k\\i, 2048+\\i*8(" dst ")\n  .endr\n  jmp 8f\n" \
    "7:\n  .irp i," R8 "\n  kmovw %k\\i, 2048+\\i*8(" dst ")\n  .endr\n8:\n"

__asm__(
    ".text\n"
    /* void t__xmm_call(in, out, fn, arg) */
    PRO(t__xmm_call)
    "  movq %rdx, %rbx\n  movq %r8, %rsi\n  movq %r9, %rdi\n"
    "  .irp i," R16 "\n  movdqu \\i*64(%rcx), %xmm\\i\n  .endr\n"
    CALLFN
    "  .irp i," R16 "\n  movdqu %xmm\\i, \\i*64(%rbx)\n  .endr\n"
    EPI

    /* unsigned long t__xmm_spin(in, out, stop, max) */
    PRO(t__xmm_spin)
    "  .irp i," R16 "\n  movdqu \\i*64(%rcx), %xmm\\i\n  .endr\n"
    SPIN
    "  .irp i," R16 "\n  movdqu %xmm\\i, \\i*64(%rdx)\n  .endr\n"
    EPI

    /* void t__ymm_call(in, out, fn, arg)  [AVX] */
    PRO(t__ymm_call)
    "  movq %rdx, %rbx\n  movq %r8, %rsi\n  movq %r9, %rdi\n"
    "  .irp i," R16 "\n  vmovdqu \\i*64(%rcx), %ymm\\i\n  .endr\n"
    CALLFN
    "  .irp i," R16 "\n  vmovdqu %ymm\\i, \\i*64(%rbx)\n  .endr\n"
    "  vzeroupper\n"
    EPI

    /* unsigned long t__ymm_spin(in, out, stop, max)  [AVX] */
    PRO(t__ymm_spin)
    "  .irp i," R16 "\n  vmovdqu \\i*64(%rcx), %ymm\\i\n  .endr\n"
    SPIN
    "  .irp i," R16 "\n  vmovdqu %ymm\\i, \\i*64(%rdx)\n  .endr\n"
    "  vzeroupper\n"
    EPI

    /* void t__zmm_call(in, out, fn, arg, kbw)  [AVX512F; kmovq needs AVX512BW] */
    PRO(t__zmm_call)
    "  movq %rdx, %rbx\n  movq %r8, %rsi\n  movq %r9, %rdi\n  movl " ARG5 ", %ebp\n"
    "  .irp i," R32 "\n  vmovdqu64 \\i*64(%rcx), %zmm\\i\n  .endr\n"
    KLOAD("%rcx")
    CALLFN
    "  .irp i," R32 "\n  vmovdqu64 %zmm\\i, \\i*64(%rbx)\n  .endr\n"
    KSTORE("%rbx")
    "  vzeroupper\n"
    EPI

    /* unsigned long t__zmm_spin(in, out, stop, max, kbw)  [AVX512F] */
    PRO(t__zmm_spin)
    "  movl " ARG5 ", %ebp\n"
    "  .irp i," R32 "\n  vmovdqu64 \\i*64(%rcx), %zmm\\i\n  .endr\n"
    KLOAD("%rcx")
    SPIN
    "  .irp i," R32 "\n  vmovdqu64 %zmm\\i, \\i*64(%rdx)\n  .endr\n"
    KSTORE("%rdx")
    "  vzeroupper\n"
    EPI

    /* unsigned long t__x87_spin(in[8], out[8], stop, max): in[0] -> ST(0) */
    PRO(t__x87_spin)
    "  fnstcw (%rsp)\n  fninit\n  fldcw (%rsp)\n"   /* (%rsp) = our own shadow area: scratch for the CW */
    "  fldt 70(%rcx)\n  fldt 60(%rcx)\n  fldt 50(%rcx)\n  fldt 40(%rcx)\n"
    "  fldt 30(%rcx)\n  fldt 20(%rcx)\n  fldt 10(%rcx)\n  fldt 0(%rcx)\n"
    SPIN
    "  fstpt 0(%rdx)\n  fstpt 10(%rdx)\n  fstpt 20(%rdx)\n  fstpt 30(%rdx)\n"
    "  fstpt 40(%rdx)\n  fstpt 50(%rdx)\n  fstpt 60(%rdx)\n  fstpt 70(%rdx)\n"
    EPI

    /* unsigned long t__mmx_spin(in[8], out[8], stop, max) */
    PRO(t__mmx_spin)
    "  .irp i," R8 "\n  movq \\i*8(%rcx), %mm\\i\n  .endr\n"
    SPIN
    "  .irp i," R8 "\n  movq %mm\\i, \\i*8(%rdx)\n  .endr\n"
    "  emms\n"
    EPI
);
#endif
