/* asm_x86.c - 32-bit register-file routines (cdecl; EBX/ESI/EDI/EBP callee-saved; every XMM register
 * is volatile in the 32-bit ABI). Frame: EBP-based, args at 8(%ebp) ... 24(%ebp).
 * Layout of t_vregs: v[i] at i*64, k[i] at 2048 + i*8. Only XMM0-7 / YMM0-7 / ZMM0-7 exist here. */
#include "t.h"
#ifdef T_X86

#define PRO(name) \
    ".globl _" #name "\n.def _" #name "; .scl 2; .type 32; .endef\n_" #name ":\n" \
    "  pushl %ebp\n  movl %esp, %ebp\n  pushl %ebx\n  pushl %esi\n  pushl %edi\n"
#define EPI "  leal -12(%ebp), %esp\n  popl %edi\n  popl %esi\n  popl %ebx\n  popl %ebp\n  ret\n"
#define R8 "0,1,2,3,4,5,6,7"
#define A1 "8(%ebp)"
#define A2 "12(%ebp)"
#define A3 "16(%ebp)"
#define A4 "20(%ebp)"
#define A5 "24(%ebp)"

/* call fn(arg) with a 16-byte aligned stack; ebx = out afterwards */
#define CALLFN \
    "  movl " A3 ", %esi\n  testl %esi, %esi\n  jz 1f\n" \
    "  andl $-16, %esp\n  subl $12, %esp\n  pushl " A4 "\n  call *%esi\n  leal -12(%ebp), %esp\n1:\n" \
    "  movl " A2 ", %ebx\n"
/* spin: eax = remaining, ecx = stop flag or NULL; ends at label 4 */
#define SPIN \
    "  movl " A3 ", %ecx\n  movl " A4 ", %eax\n  testl %eax, %eax\n  jz 4f\n" \
    "2: testl %ecx, %ecx\n  jz 3f\n  cmpl $0, (%ecx)\n  jne 4f\n" \
    "3: pause\n  decl %eax\n  jnz 2b\n4:\n"
#define KLOAD(src) \
    "  cmpl $0, " A5 "\n  je 5f\n" \
    "  .irp i," R8 "\n  kmovq 2048+\\i*8(" src "), %k\\i\n  .endr\n  jmp 6f\n" \
    "5:\n  .irp i," R8 "\n  kmovw 2048+\\i*8(" src "), %k\\i\n  .endr\n6:\n"
#define KSTORE(dst) \
    "  cmpl $0, " A5 "\n  je 7f\n" \
    "  .irp i," R8 "\n  kmovq %k\\i, 2048+\\i*8(" dst ")\n  .endr\n  jmp 8f\n" \
    "7:\n  .irp i," R8 "\n  kmovw %k\\i, 2048+\\i*8(" dst ")\n  .endr\n8:\n"

__asm__(
    ".text\n"
    PRO(t__xmm_call)
    "  movl " A1 ", %edx\n  .irp i," R8 "\n  movdqu \\i*64(%edx), %xmm\\i\n  .endr\n"
    CALLFN
    "  .irp i," R8 "\n  movdqu %xmm\\i, \\i*64(%ebx)\n  .endr\n"
    EPI

    PRO(t__xmm_spin)
    "  movl " A1 ", %edx\n  .irp i," R8 "\n  movdqu \\i*64(%edx), %xmm\\i\n  .endr\n"
    SPIN
    "  movl " A2 ", %edx\n  .irp i," R8 "\n  movdqu %xmm\\i, \\i*64(%edx)\n  .endr\n"
    EPI

    PRO(t__ymm_call)
    "  movl " A1 ", %edx\n  .irp i," R8 "\n  vmovdqu \\i*64(%edx), %ymm\\i\n  .endr\n"
    CALLFN
    "  .irp i," R8 "\n  vmovdqu %ymm\\i, \\i*64(%ebx)\n  .endr\n  vzeroupper\n"
    EPI

    PRO(t__ymm_spin)
    "  movl " A1 ", %edx\n  .irp i," R8 "\n  vmovdqu \\i*64(%edx), %ymm\\i\n  .endr\n"
    SPIN
    "  movl " A2 ", %edx\n  .irp i," R8 "\n  vmovdqu %ymm\\i, \\i*64(%edx)\n  .endr\n  vzeroupper\n"
    EPI

    PRO(t__zmm_call)
    "  movl " A1 ", %edx\n  .irp i," R8 "\n  vmovdqu64 \\i*64(%edx), %zmm\\i\n  .endr\n"
    KLOAD("%edx")
    CALLFN
    "  .irp i," R8 "\n  vmovdqu64 %zmm\\i, \\i*64(%ebx)\n  .endr\n"
    KSTORE("%ebx")
    "  vzeroupper\n"
    EPI

    PRO(t__zmm_spin)
    "  movl " A1 ", %edx\n  .irp i," R8 "\n  vmovdqu64 \\i*64(%edx), %zmm\\i\n  .endr\n"
    KLOAD("%edx")
    SPIN
    "  movl " A2 ", %edx\n  .irp i," R8 "\n  vmovdqu64 %zmm\\i, \\i*64(%edx)\n  .endr\n"
    KSTORE("%edx")
    "  vzeroupper\n"
    EPI

    PRO(t__x87_spin)
    "  subl $4, %esp\n  fnstcw (%esp)\n  fninit\n  fldcw (%esp)\n"
    "  movl " A1 ", %edx\n"
    "  fldt 70(%edx)\n  fldt 60(%edx)\n  fldt 50(%edx)\n  fldt 40(%edx)\n"
    "  fldt 30(%edx)\n  fldt 20(%edx)\n  fldt 10(%edx)\n  fldt 0(%edx)\n"
    SPIN
    "  movl " A2 ", %edx\n"
    "  fstpt 0(%edx)\n  fstpt 10(%edx)\n  fstpt 20(%edx)\n  fstpt 30(%edx)\n"
    "  fstpt 40(%edx)\n  fstpt 50(%edx)\n  fstpt 60(%edx)\n  fstpt 70(%edx)\n"
    EPI

    PRO(t__mmx_spin)
    "  movl " A1 ", %edx\n  .irp i," R8 "\n  movq \\i*8(%edx), %mm\\i\n  .endr\n"
    SPIN
    "  movl " A2 ", %edx\n  .irp i," R8 "\n  movq %mm\\i, \\i*8(%edx)\n  .endr\n  emms\n"
    EPI
);
#endif
