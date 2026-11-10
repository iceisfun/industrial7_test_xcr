/* capture2_ref.h - hand-built instruction-level reference for ntdll!RtlCaptureContext2 (x64 only).
 *
 * RtlCaptureContext2 needs Windows 10 version 2004 (build 19041) or later, x64. This reference gives the
 * capture checks a self-check path where the export is missing, and is the field-by-field oracle where it
 * exists. Pinned against the Windows 10 19041 ntdll disassembly (corpus explorer, RtlCaptureContext2 at
 * RVA 0xa12c0): push rcx; pushfq; sub rsp,0x28; store RAX RCX RDX R8-R11; FXSAVE (no REX.W) at +0x100; CS DS ES
 * SS FS GS; RBX RBP RSI RDI R12-R15; STMXCSR at +0x34; Rsp = caller RSP after the return; Rip = return address;
 * EFlags = the PUSHFQ image; RtlpSanitizeContextFlags(&Flags, 1); Flags &= 0x100040; when CONTEXT_XSTATE survives,
 * RtlpMergeContextXState(c, RDSSPQ) writes ONLY the CET_U component (XSTATE feature 11) and its mask bit (it does
 * not XSAVE the AVX component); finally Flags |= 0x10000F.
 *
 * xc2_ref_capture_avx(CONTEXT *c) does the same, except: no flag validation (callers pass valid flags), no CET_U
 * merge (comparisons mask bit 11), and when g_xc2_avx_dst is set (LocateXStateFeature(c, XSTATE_AVX) of a
 * CONTEXT_XSTATE record) and CONTEXT_XSTATE survived, it stores YMM0-15[255:128] there and sets XSTATE mask bit 2:
 * what a capture that includes AVX state must produce. VEX code runs only when g_xc2_avx_dst != NULL, which
 * callers set only after cpu_has(F_AVX). Include once per exe (the asm defines global symbols). */
#ifndef CAPTURE2_REF_H
#define CAPTURE2_REF_H
#include "t.h"
#include <stddef.h>

#ifdef T_X64
void *g_xc2_avx_dst;
__asm__(
    ".text\n.globl xc2_ref_capture_avx\nxc2_ref_capture_avx:\n"
    "  pushq %rcx\n  pushfq\n  subq $0x28, %rsp\n"
    "  movq %rax, 0x78(%rcx)\n  movq %rcx, 0x80(%rcx)\n  movq %rdx, 0x88(%rcx)\n"
    "  movq %r8, 0xb8(%rcx)\n  movq %r9, 0xc0(%rcx)\n  movq %r10, 0xc8(%rcx)\n  movq %r11, 0xd0(%rcx)\n"
    "  fxsave 0x100(%rcx)\n"
    "  movw %cs, 0x38(%rcx)\n  movw %ds, 0x3a(%rcx)\n  movw %es, 0x3c(%rcx)\n  movw %ss, 0x42(%rcx)\n"
    "  movw %fs, 0x3e(%rcx)\n  movw %gs, 0x40(%rcx)\n"
    "  movq %rbx, 0x90(%rcx)\n  movq %rbp, 0xa0(%rcx)\n  movq %rsi, 0xa8(%rcx)\n  movq %rdi, 0xb0(%rcx)\n"
    "  movq %r12, 0xd8(%rcx)\n  movq %r13, 0xe0(%rcx)\n  movq %r14, 0xe8(%rcx)\n  movq %r15, 0xf0(%rcx)\n"
    "  stmxcsr 0x34(%rcx)\n"
    "  leaq 0x40(%rsp), %rax\n  movq %rax, 0x98(%rcx)\n"
    "  movq 0x38(%rsp), %rax\n  movq %rax, 0xf8(%rcx)\n"
    "  movl 0x28(%rsp), %eax\n  movl %eax, 0x44(%rcx)\n"
    "  movl 0x30(%rcx), %eax\n  andl $0x100040, %eax\n  cmpl $0x100040, %eax\n  jne 1f\n"
    "  movq g_xc2_avx_dst(%rip), %rdx\n  testq %rdx, %rdx\n  jz 1f\n"
    "  .irp i,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n  vextractf128 $1, %ymm\\i, \\i*16(%rdx)\n  .endr\n"
    "  movslq 0x4e0(%rcx), %rdx\n  orq $4, 0x4d0(%rcx,%rdx)\n"
    "1:\n  orl $0x10000f, %eax\n  movl %eax, 0x30(%rcx)\n"
    "  addq $0x30, %rsp\n  popq %rcx\n  ret\n"
);
void xc2_ref_capture_avx(CONTEXT *c);
_Static_assert(offsetof(CONTEXT, ContextFlags) == 0x30 && offsetof(CONTEXT, MxCsr) == 0x34 && offsetof(CONTEXT, SegCs) == 0x38 &&
               offsetof(CONTEXT, EFlags) == 0x44 && offsetof(CONTEXT, Rax) == 0x78 && offsetof(CONTEXT, Rsp) == 0x98 &&
               offsetof(CONTEXT, R15) == 0xf0 && offsetof(CONTEXT, Rip) == 0xf8 && offsetof(CONTEXT, FltSave) == 0x100 &&
               sizeof(CONTEXT) == 0x4d0, "x64 CONTEXT layout used by xc2_ref_capture_avx");
#endif

#define CAP2_SKIP_REASON "ntdll!RtlCaptureContext2 not exported (needs Windows 10 version 2004, build 19041, or later; x64)"

#endif
