/* alignment - aligned vs unaligned SIMD/GPR memory access, page-boundary straddling, faults on
 * misaligned or inaccessible operands, rep movs across pages, stack / heap / VirtualAlloc alignment.
 *
 * Contract under test (architectural: Intel SDM Vol.1 ch.4.1.1 / 10-15, Vol.2 instruction pages,
 * Vol.3 ch.6 (#GP, #PF, fault precision); AMD APM Vol.1/2/4; Windows: x64 ABI, VirtualAlloc /
 * MapViewOfFile / HeapAlloc / InitializeContext documentation):
 *   - Legacy-SSE 128-bit memory operands must be 16-byte aligned (#GP otherwise) except the explicitly
 *     unaligned forms (MOVUPS/MOVUPD/MOVDQU/LDDQU), scalar/partial (<16 byte) operands and PCMPxSTRx.
 *     MOVAPS/MOVAPD/MOVDQA/MOVNT* need 16, VEX.256 VMOVDQA/VMOVAPS/VMOVNT* need 32, EVEX.512 aligned
 *     moves need 64. VEX/EVEX non-move ALU memory operands have NO alignment requirement.
 *   - FXSAVE/FXRSTOR need 16, XSAVE/XRSTOR/XSAVEOPT/XSAVEC need 64, CMPXCHG16B needs 16: #GP otherwise.
 *   - Faults are precise: the faulting instruction does not update its destination register; the
 *     reported instruction pointer is the faulting instruction (for #GP no data is written).
 *   - A page fault on an access that straddles into an inaccessible page is reported by Windows as
 *     STATUS_ACCESS_VIOLATION with ExceptionInformation[0] = 0 (read) / 1 (write) and [1] = an address
 *     of the access that lies in the inaccessible page (Intel: CR2 may be any byte of the access in the
 *     faulting page, so only the range is hard-checked). PAGE_GUARD -> STATUS_GUARD_PAGE_VIOLATION, one shot.
 *   - Whether a straddling STORE partially wrote the accessible page is implementation-specific: noted only.
 *   - REP MOVSB interrupted by a fault leaves RCX/RSI/RDI describing completed iterations; continuing
 *     after the fault is fixed completes the copy (SDM Vol.1 7.3.9.3, Vol.3 "string instructions").
 *   - Masked-out elements never fault: AVX-512 masking (architectural); VEX VMASKMOV only on Intel
 *     (AMD APM: behaviour for unselected elements is implementation dependent -> INFO on non-Intel).
 *   - Windows x64 ABI: RSP is 16-byte aligned at each call, so RSP % 16 == 8 at function entry, also for
 *     OS-invoked callbacks (thread start, APC, VEH, fibers, thread pool, timers). x86: only 4 guaranteed.
 *   - HeapAlloc returns MEMORY_ALLOCATION_ALIGNMENT-aligned blocks (16 on x64, 8 on x86); CRT malloc 16/8;
 *     _aligned_malloc honours power-of-two alignments and rejects others (EINVAL).
 *   - VirtualAlloc(MEM_RESERVE) bases are multiples of dwAllocationGranularity and an explicit address is
 *     rounded down to it; MEM_COMMIT rounds down to a page. MapViewOfFile offsets must be granularity
 *     multiples (ERROR_MAPPED_ALIGNMENT).
 *   - InitializeContext (checked kernelbase!InitializeContext2 + ntdll!RtlInitializeExtendedContext2
 *     decomp, Win10 19041): a NULL buffer returns FALSE / ERROR_INSUFFICIENT_BUFFER (0x7a) with the
 *     length; the length covers any buffer alignment; the CONTEXT is placed at (buf+15)&~15 on x64
 *     (CONTEXT_AMD64) / (buf+3)&~3 on x86 (CONTEXT_i386), the XSAVE header region at a 64-byte boundary.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> t_note/t_info.
 * Deliberately faulting operations run in CHILD() processes (al_probe & co) with a vectored handler that
 * records the exception and resumes after the faulting instruction (stubs below have known labels). */
#include "t.h"
#include <malloc.h>
#include <errno.h>

#define PG 4096
#define PTRW ((int)sizeof(void *))
#ifndef CONTEXT_XSTATE
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)   /* winnt.h value; missing from mingw's x86 headers */
#endif

/* ======================================================================================
 * Assembly stubs with exact labels:  <n>_entry (call target)  <n> (the probe instruction)
 * <n>_end (resume point).  Signature: void stub(void *p, void *aux)
 *   aux[0..64)   value preloaded into the register (xmm0/ymm0/zmm0/rax) before the probe
 *   aux[64..128) register stored after the probe (or after resuming at _end)
 *   aux[128..)   extra operands (masks)
 * ====================================================================================== */
#ifdef T_X64
#define RP "%rcx"
#define RA "%rdx"
#define RAX "%rax"
#define RSPR "%rsp"
#define ENTRY_ARGS ""
#else
#define RP "%ecx"
#define RA "%edx"
#define RAX "%eax"
#define RSPR "%esp"
#define ENTRY_ARGS "\tmov 4(%esp), %ecx\n\tmov 8(%esp), %edx\n"
#endif
#define AL_STUB(n, pre, ins, post) \
    "\t.p2align 4\n\t.globl " n "_entry\n" n "_entry:\n" ENTRY_ARGS "\t" pre "\n" \
    "\t.globl " n "\n" n ":\n\t" ins "\n\t.globl " n "_end\n" n "_end:\n\t" post "\n\tret\n"
#define MEM    "(" RP ")"
#define PRE_X  "movups (" RA "), %xmm0"
#define POST_X "movups %xmm0, 64(" RA ")"
#define PRE_Y  "vmovdqu (" RA "), %ymm0"
#define POST_Y "vmovdqu %ymm0, 64(" RA ")\n\tvzeroupper"
#define PRE_Z  "vmovdqu64 (" RA "), %zmm0"
#define POST_Z "vmovdqu64 %zmm0, 64(" RA ")\n\tvzeroupper"
#define PRE_XS "xor %edx, %edx\n\tmov $3, %eax"

__asm__(
    ".text\n"
    /* legacy SSE loads */
    AL_STUB("al_movaps_ld",   PRE_X, "movaps "   MEM ", %xmm0", POST_X)
    AL_STUB("al_movapd_ld",   PRE_X, "movapd "   MEM ", %xmm0", POST_X)
    AL_STUB("al_movdqa_ld",   PRE_X, "movdqa "   MEM ", %xmm0", POST_X)
    AL_STUB("al_movups_ld",   PRE_X, "movups "   MEM ", %xmm0", POST_X)
    AL_STUB("al_movdqu_ld",   PRE_X, "movdqu "   MEM ", %xmm0", POST_X)
    AL_STUB("al_lddqu_ld",    PRE_X, "lddqu "    MEM ", %xmm0", POST_X)
    AL_STUB("al_movntdqa_ld", PRE_X, "movntdqa " MEM ", %xmm0", POST_X)
    AL_STUB("al_paddd_ld",    PRE_X, "paddd "    MEM ", %xmm0", POST_X)
    AL_STUB("al_addps_ld",    PRE_X, "addps "    MEM ", %xmm0", POST_X)
    AL_STUB("al_pshufb_ld",   PRE_X, "pshufb "   MEM ", %xmm0", POST_X)
    AL_STUB("al_pcmpeqb_ld",  PRE_X, "pcmpeqb "  MEM ", %xmm0", POST_X)
    AL_STUB("al_pcmpistri_ld", PRE_X, "pcmpistri $0x0c, " MEM ", %xmm0", POST_X)
    AL_STUB("al_movq_ld",     PRE_X, "movq "     MEM ", %xmm0", POST_X)
    AL_STUB("al_movd_ld",     PRE_X, "movd "     MEM ", %xmm0", POST_X)
    AL_STUB("al_gpr_ld", "mov (" RA "), " RAX, "mov " MEM ", " RAX, "mov " RAX ", 64(" RA ")")
    /* legacy SSE / GPR stores */
    AL_STUB("al_movaps_st",   PRE_X, "movaps %xmm0, "  MEM, "")
    AL_STUB("al_movapd_st",   PRE_X, "movapd %xmm0, "  MEM, "")
    AL_STUB("al_movdqa_st",   PRE_X, "movdqa %xmm0, "  MEM, "")
    AL_STUB("al_movntps_st",  PRE_X, "movntps %xmm0, " MEM, "sfence")
    AL_STUB("al_movntpd_st",  PRE_X, "movntpd %xmm0, " MEM, "sfence")
    AL_STUB("al_movntdq_st",  PRE_X, "movntdq %xmm0, " MEM, "sfence")
    AL_STUB("al_movups_st",   PRE_X, "movups %xmm0, "  MEM, "")
    AL_STUB("al_movdqu_st",   PRE_X, "movdqu %xmm0, "  MEM, "")
    AL_STUB("al_movq_st",     PRE_X, "movq %xmm0, "    MEM, "")
    AL_STUB("al_gpr_st", "mov (" RA "), " RAX, "mov " RAX ", " MEM, "")
    /* VEX */
    AL_STUB("al_avx_vmovaps_x_ld",  PRE_Y, "vmovaps "  MEM ", %xmm0", POST_Y)
    AL_STUB("al_avx_vmovdqa_x_st",  PRE_Y, "vmovdqa %xmm0, " MEM, "vzeroupper")
    AL_STUB("al_avx_vmovdqa_y_ld",  PRE_Y, "vmovdqa "  MEM ", %ymm0", POST_Y)
    AL_STUB("al_avx_vmovaps_y_st",  PRE_Y, "vmovaps %ymm0, " MEM, "vzeroupper")
    AL_STUB("al_avx_vmovntdq_y_st", PRE_Y, "vmovntdq %ymm0, " MEM, "sfence\n\tvzeroupper")
    AL_STUB("al_avx_vmovntps_y_st", PRE_Y, "vmovntps %ymm0, " MEM, "sfence\n\tvzeroupper")
    AL_STUB("al_avx2_vmovntdqa_y_ld", PRE_Y, "vmovntdqa " MEM ", %ymm0", POST_Y)
    AL_STUB("al_avx_vmovdqu_y_ld",  PRE_Y, "vmovdqu "  MEM ", %ymm0", POST_Y)
    AL_STUB("al_avx_vmovdqu_y_st",  PRE_Y, "vmovdqu %ymm0, " MEM, "vzeroupper")
    AL_STUB("al_avx_vpxor_x_ld",    PRE_Y, "vpxor "    MEM ", %xmm0, %xmm0", POST_Y)
    AL_STUB("al_avx_vpaddd_x_ld",   PRE_Y, "vpaddd "   MEM ", %xmm0, %xmm0", POST_Y)
    AL_STUB("al_avx_vaddps_y_ld",   PRE_Y, "vaddps "   MEM ", %ymm0, %ymm0", POST_Y)
    AL_STUB("al_avx2_vpaddd_y_ld",  PRE_Y, "vpaddd "   MEM ", %ymm0, %ymm0", POST_Y)
    AL_STUB("al_avx_vmaskmovps_ld", PRE_Y "\n\tvmovdqu 128(" RA "), %ymm1", "vmaskmovps " MEM ", %ymm1, %ymm0", POST_Y)
    AL_STUB("al_avx_vmaskmovps_st", PRE_Y "\n\tvmovdqu 128(" RA "), %ymm1", "vmaskmovps %ymm0, %ymm1, " MEM, "vzeroupper")
    /* EVEX */
    AL_STUB("al_avx512_vmovdqa64_z_ld", PRE_Z, "vmovdqa64 " MEM ", %zmm0", POST_Z)
    AL_STUB("al_avx512_vmovdqa64_z_st", PRE_Z, "vmovdqa64 %zmm0, " MEM, "vzeroupper")
    AL_STUB("al_avx512_vmovntdq_z_st",  PRE_Z, "vmovntdq %zmm0, " MEM, "sfence\n\tvzeroupper")
    AL_STUB("al_avx512_vmovdqu64_z_ld", PRE_Z, "vmovdqu64 " MEM ", %zmm0", POST_Z)
    AL_STUB("al_avx512_vmovdqu64_z_st", PRE_Z, "vmovdqu64 %zmm0, " MEM, "vzeroupper")
    AL_STUB("al_avx512_vpaddd_z_ld",    PRE_Z, "vpaddd " MEM ", %zmm0, %zmm0", POST_Z)
    AL_STUB("al_avx512_vmovdqu32_kz_ld", PRE_Z "\n\tkmovw 128(" RA "), %k1", "vmovdqu32 " MEM ", %zmm0{%k1}{z}", POST_Z)
    AL_STUB("al_avx512_vmovdqu32_k_st",  PRE_Z "\n\tkmovw 128(" RA "), %k1", "vmovdqu32 %zmm0, " MEM "{%k1}", "vzeroupper")
    /* save areas */
    AL_STUB("al_fxsave",   "", "fxsave "   MEM, "")
    AL_STUB("al_fxrstor",  "", "fxrstor "  MEM, "")
    AL_STUB("al_xsave",   PRE_XS, "xsave "    MEM, "")
    AL_STUB("al_xrstor",  PRE_XS, "xrstor "   MEM, "")
    AL_STUB("al_xsaveopt", PRE_XS, "xsaveopt " MEM, "")
    AL_STUB("al_xsavec",  PRE_XS, "xsavec "   MEM, "")
    /* EFLAGS.AC probe (INFO only): set AC, 4-byte load at p, clear AC */
    AL_STUB("al_acprobe", "pushf\n\torl $0x40000, (" RSPR ")\n\tpopf", "mov " MEM ", %eax",
            "pushf\n\tandl $0xfffbffff, (" RSPR ")\n\tpopf")
#ifdef T_X64
    AL_STUB("al_cmpxchg16b", "push %rbx\n\tmov %rcx, %r8\n\txor %eax, %eax\n\txor %edx, %edx\n\txor %ebx, %ebx\n\txor %ecx, %ecx",
            "lock cmpxchg16b (%r8)", "pop %rbx")
    /* void al_repmovsb(void *dst, const void *src, size_t n) */
    "\t.p2align 4\n\t.globl al_repmovsb_entry\nal_repmovsb_entry:\n\tpush %rdi\n\tpush %rsi\n"
    "\tmov %rcx, %rdi\n\tmov %rdx, %rsi\n\tmov %r8, %rcx\n"
    "\t.globl al_repmovsb\nal_repmovsb:\n\trep movsb\n\t.globl al_repmovsb_end\nal_repmovsb_end:\n\tpop %rsi\n\tpop %rdi\n\tret\n"
    /* stack-alignment thunk: record RSP at entry, tail-jump to the real callback */
    "\t.p2align 4\n\t.globl al_sp_thunk\nal_sp_thunk:\n\tmov %rsp, al_sp_slot(%rip)\n\tjmp *al_sp_target(%rip)\n"
    "\t.globl al_get_sp\nal_get_sp:\n\tmov %rsp, %rax\n\tret\n"
#else
    "\t.p2align 4\n\t.globl al_repmovsb_entry\nal_repmovsb_entry:\n\tpush %edi\n\tpush %esi\n"
    "\tmov 12(%esp), %edi\n\tmov 16(%esp), %esi\n\tmov 20(%esp), %ecx\n"
    "\t.globl al_repmovsb\nal_repmovsb:\n\trep movsb\n\t.globl al_repmovsb_end\nal_repmovsb_end:\n\tpop %esi\n\tpop %edi\n\tret\n"
    "\t.p2align 4\n\t.globl al_sp_thunk\nal_sp_thunk:\n\tmov %esp, al_sp_slot\n\tjmp *al_sp_target\n"
    "\t.globl al_get_sp\nal_get_sp:\n\tmov %esp, %eax\n\tret\n"
#endif
);

#define AL_DECL(n) extern char n##_entry[] __asm__(#n "_entry"), n[] __asm__(#n), n##_end[] __asm__(#n "_end");
AL_DECL(al_movaps_ld) AL_DECL(al_movapd_ld) AL_DECL(al_movdqa_ld) AL_DECL(al_movups_ld) AL_DECL(al_movdqu_ld)
AL_DECL(al_lddqu_ld) AL_DECL(al_movntdqa_ld) AL_DECL(al_paddd_ld) AL_DECL(al_addps_ld) AL_DECL(al_pshufb_ld)
AL_DECL(al_pcmpeqb_ld) AL_DECL(al_pcmpistri_ld) AL_DECL(al_movq_ld) AL_DECL(al_movd_ld) AL_DECL(al_gpr_ld)
AL_DECL(al_movaps_st) AL_DECL(al_movapd_st) AL_DECL(al_movdqa_st) AL_DECL(al_movntps_st) AL_DECL(al_movntpd_st)
AL_DECL(al_movntdq_st) AL_DECL(al_movups_st) AL_DECL(al_movdqu_st) AL_DECL(al_movq_st) AL_DECL(al_gpr_st)
AL_DECL(al_avx_vmovaps_x_ld) AL_DECL(al_avx_vmovdqa_x_st) AL_DECL(al_avx_vmovdqa_y_ld) AL_DECL(al_avx_vmovaps_y_st)
AL_DECL(al_avx_vmovntdq_y_st) AL_DECL(al_avx_vmovntps_y_st) AL_DECL(al_avx2_vmovntdqa_y_ld) AL_DECL(al_avx_vmovdqu_y_ld)
AL_DECL(al_avx_vmovdqu_y_st) AL_DECL(al_avx_vpxor_x_ld) AL_DECL(al_avx_vpaddd_x_ld) AL_DECL(al_avx_vaddps_y_ld)
AL_DECL(al_avx2_vpaddd_y_ld) AL_DECL(al_avx_vmaskmovps_ld) AL_DECL(al_avx_vmaskmovps_st)
AL_DECL(al_avx512_vmovdqa64_z_ld) AL_DECL(al_avx512_vmovdqa64_z_st) AL_DECL(al_avx512_vmovntdq_z_st)
AL_DECL(al_avx512_vmovdqu64_z_ld) AL_DECL(al_avx512_vmovdqu64_z_st) AL_DECL(al_avx512_vpaddd_z_ld)
AL_DECL(al_avx512_vmovdqu32_kz_ld) AL_DECL(al_avx512_vmovdqu32_k_st)
AL_DECL(al_fxsave) AL_DECL(al_fxrstor) AL_DECL(al_xsave) AL_DECL(al_xrstor) AL_DECL(al_xsaveopt) AL_DECL(al_xsavec)
AL_DECL(al_acprobe) AL_DECL(al_repmovsb)
#ifdef T_X64
AL_DECL(al_cmpxchg16b)
#endif
extern void al_sp_thunk(void) __asm__("al_sp_thunk");
extern uintptr_t al_get_sp(void) __asm__("al_get_sp");
volatile uintptr_t al_sp_slot __asm__("al_sp_slot");
void *volatile al_sp_target __asm__("al_sp_target");

typedef void (*al_fn2)(void *p, void *aux);
typedef void (*al_fn3)(void *d, const void *s, size_t n);

enum { SK_LOAD, SK_STORE, SK_OTHER, SK_RESTORE };
typedef struct al_stub {
    const char *name; char *entry, *fault, *end;
    short w, kind, regw, pure, gpr, rsz;   /* rsz: valid image bytes to place before SK_RESTORE */
    int feat;
} al_stub;
#define STB(n, w, k, regw, pure, gpr, rsz, feat) { #n, n##_entry, n, n##_end, w, k, regw, pure, gpr, rsz, feat }
static const al_stub g_stubs[] = {
    STB(al_movaps_ld, 16, SK_LOAD, 16, 1, 0, 0, F_SSE),
    STB(al_movapd_ld, 16, SK_LOAD, 16, 1, 0, 0, F_SSE2),
    STB(al_movdqa_ld, 16, SK_LOAD, 16, 1, 0, 0, F_SSE2),
    STB(al_movups_ld, 16, SK_LOAD, 16, 1, 0, 0, F_SSE),
    STB(al_movdqu_ld, 16, SK_LOAD, 16, 1, 0, 0, F_SSE2),
    STB(al_lddqu_ld, 16, SK_LOAD, 16, 1, 0, 0, F_SSE3),
    STB(al_movntdqa_ld, 16, SK_LOAD, 16, 1, 0, 0, F_SSE41),
    STB(al_paddd_ld, 16, SK_LOAD, 16, 0, 0, 0, F_SSE2),
    STB(al_addps_ld, 16, SK_LOAD, 16, 0, 0, 0, F_SSE),
    STB(al_pshufb_ld, 16, SK_LOAD, 16, 0, 0, 0, F_SSSE3),
    STB(al_pcmpeqb_ld, 16, SK_LOAD, 16, 0, 0, 0, F_SSE2),
    STB(al_pcmpistri_ld, 16, SK_LOAD, 16, 0, 0, 0, F_SSE42),
    STB(al_movq_ld, 8, SK_LOAD, 16, 1, 0, 0, F_SSE2),
    STB(al_movd_ld, 4, SK_LOAD, 16, 1, 0, 0, F_SSE2),
    STB(al_gpr_ld, PTRW, SK_LOAD, PTRW, 1, 1, 0, F_SSE2),
    STB(al_movaps_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE),
    STB(al_movapd_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE2),
    STB(al_movdqa_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE2),
    STB(al_movntps_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE),
    STB(al_movntpd_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE2),
    STB(al_movntdq_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE2),
    STB(al_movups_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE),
    STB(al_movdqu_st, 16, SK_STORE, 0, 0, 0, 0, F_SSE2),
    STB(al_movq_st, 8, SK_STORE, 0, 0, 0, 0, F_SSE2),
    STB(al_gpr_st, PTRW, SK_STORE, 0, 0, 1, 0, F_SSE2),
    STB(al_avx_vmovaps_x_ld, 16, SK_LOAD, 32, 1, 0, 0, F_AVX),
    STB(al_avx_vmovdqa_x_st, 16, SK_STORE, 0, 0, 0, 0, F_AVX),
    STB(al_avx_vmovdqa_y_ld, 32, SK_LOAD, 32, 1, 0, 0, F_AVX),
    STB(al_avx_vmovaps_y_st, 32, SK_STORE, 0, 0, 0, 0, F_AVX),
    STB(al_avx_vmovntdq_y_st, 32, SK_STORE, 0, 0, 0, 0, F_AVX),
    STB(al_avx_vmovntps_y_st, 32, SK_STORE, 0, 0, 0, 0, F_AVX),
    STB(al_avx2_vmovntdqa_y_ld, 32, SK_LOAD, 32, 1, 0, 0, F_AVX2),
    STB(al_avx_vmovdqu_y_ld, 32, SK_LOAD, 32, 1, 0, 0, F_AVX),
    STB(al_avx_vmovdqu_y_st, 32, SK_STORE, 0, 0, 0, 0, F_AVX),
    STB(al_avx_vpxor_x_ld, 16, SK_LOAD, 32, 0, 0, 0, F_AVX),
    STB(al_avx_vpaddd_x_ld, 16, SK_LOAD, 32, 0, 0, 0, F_AVX),
    STB(al_avx_vaddps_y_ld, 32, SK_LOAD, 32, 0, 0, 0, F_AVX),
    STB(al_avx2_vpaddd_y_ld, 32, SK_LOAD, 32, 0, 0, 0, F_AVX2),
    STB(al_avx_vmaskmovps_ld, 32, SK_LOAD, 32, 0, 0, 0, F_AVX),
    STB(al_avx_vmaskmovps_st, 32, SK_STORE, 0, 0, 0, 0, F_AVX),
    STB(al_avx512_vmovdqa64_z_ld, 64, SK_LOAD, 64, 1, 0, 0, F_AVX512F),
    STB(al_avx512_vmovdqa64_z_st, 64, SK_STORE, 0, 0, 0, 0, F_AVX512F),
    STB(al_avx512_vmovntdq_z_st, 64, SK_STORE, 0, 0, 0, 0, F_AVX512F),
    STB(al_avx512_vmovdqu64_z_ld, 64, SK_LOAD, 64, 1, 0, 0, F_AVX512F),
    STB(al_avx512_vmovdqu64_z_st, 64, SK_STORE, 0, 0, 0, 0, F_AVX512F),
    STB(al_avx512_vpaddd_z_ld, 64, SK_LOAD, 64, 0, 0, 0, F_AVX512F),
    STB(al_avx512_vmovdqu32_kz_ld, 64, SK_LOAD, 64, 0, 0, 0, F_AVX512F),
    STB(al_avx512_vmovdqu32_k_st, 64, SK_STORE, 0, 0, 0, 0, F_AVX512F),
    STB(al_fxsave, 512, SK_OTHER, 0, 0, 0, 0, F_FXSR),
    STB(al_fxrstor, 512, SK_RESTORE, 0, 0, 0, 512, F_FXSR),
    STB(al_xsave, 576, SK_OTHER, 0, 0, 0, 0, F_XSAVE),
    STB(al_xrstor, 576, SK_RESTORE, 0, 0, 0, 576, F_XSAVE),
    STB(al_xsaveopt, 576, SK_OTHER, 0, 0, 0, 0, F_XSAVEOPT),
    STB(al_xsavec, 576, SK_OTHER, 0, 0, 0, 0, F_XSAVEC),
    STB(al_acprobe, 4, SK_OTHER, 0, 0, 0, 0, F_SSE2),
#ifdef T_X64
    STB(al_cmpxchg16b, 16, SK_OTHER, 0, 0, 0, 0, F_CX16),
#endif
};
static const al_stub *al_find_stub(const char *name) {
    for (size_t i = 0; i < T_COUNTOF(g_stubs); i++) if (!strcmp(g_stubs[i].name, name)) return &g_stubs[i];
    return NULL;
}

/* ======================================================================================
 * Child side: vectored handler + memory layouts
 * ====================================================================================== */
enum { ACT_SKIP = 0, ACT_UNPROTECT = 1 };
static struct {
    volatile int armed, hit;
    uintptr_t fault, end;
    int action;
    void *up; SIZE_T up_len;
    DWORD code, np; ULONG_PTR i0, i1;
    int ipok, xmm_valid; uint8_t xmm0[16];
    uintptr_t ax, cx, di, si;
    const uint8_t *rep_dst, *rep_src; long long rep_lim; int rep_pre;
} g_v;
static T_ALIGN(64) uint8_t g_aux[256];

static LONG CALLBACK al_veh(EXCEPTION_POINTERS *ep) {
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    CONTEXT *c = ep->ContextRecord;
    uintptr_t ip = (uintptr_t)er->ExceptionAddress;
    if (!g_v.armed || ip != g_v.fault) return EXCEPTION_CONTINUE_SEARCH;
    g_v.armed = 0;
    g_v.hit++;
    g_v.code = er->ExceptionCode;
    g_v.np = er->NumberParameters;
    g_v.i0 = er->NumberParameters > 0 ? er->ExceptionInformation[0] : (ULONG_PTR)-1;
    g_v.i1 = er->NumberParameters > 1 ? er->ExceptionInformation[1] : 0;
#ifdef T_X64
    g_v.ipok = c->Rip == ip;
    g_v.ax = c->Rax; g_v.cx = c->Rcx; g_v.di = c->Rdi; g_v.si = c->Rsi;
    g_v.xmm_valid = (c->ContextFlags & CONTEXT_FLOATING_POINT) == CONTEXT_FLOATING_POINT;
    if (g_v.xmm_valid) memcpy(g_v.xmm0, &c->FltSave.XmmRegisters[0], 16);
#else
    g_v.ipok = c->Eip == ip;
    g_v.ax = c->Eax; g_v.cx = c->Ecx; g_v.di = c->Edi; g_v.si = c->Esi;
    g_v.xmm_valid = (c->ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS;
    if (g_v.xmm_valid) memcpy(g_v.xmm0, c->ExtendedRegisters + 160, 16);
#endif
    if (g_v.action == ACT_SKIP) {
#ifdef T_X64
        c->Rip = g_v.end;
#else
        c->Eip = (DWORD)g_v.end;
#endif
    } else {
        if (g_v.rep_dst) {   /* rep movsb: completed prefix must already be copied */
            long long done = (long long)(g_v.di - (uintptr_t)g_v.rep_dst);
            g_v.rep_pre = (done >= 0 && done <= g_v.rep_lim) ? memcmp(g_v.rep_dst, g_v.rep_src, (size_t)done) == 0 : -1;
        }
        DWORD old;
        VirtualProtect(g_v.up, g_v.up_len, PAGE_READWRITE, &old);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

static int al_call(const al_stub *s, void *p) {
    g_v.fault = (uintptr_t)s->fault; g_v.end = (uintptr_t)s->end;
    g_v.action = ACT_SKIP; g_v.hit = 0; g_v.xmm_valid = 0; g_v.ipok = 0; g_v.rep_dst = NULL;
    g_v.armed = 1;
    ((al_fn2)(void *)s->entry)(p, g_aux);
    g_v.armed = 0;
    return g_v.hit;
}

static uint8_t al_pat(size_t i) { return (uint8_t)(0x5Au ^ (i * 29u) ^ (i >> 7) ^ ((i >> 12) * 0x33u)); }

enum { M_HEAP, M_NOACCESS, M_READONLY, M_RESERVE, M_GUARD };
static const char *const g_mode_names[] = { "heap", "noaccess", "readonly", "reserve", "guard" };
typedef struct { uint8_t *base, *ref; int mode; } al_mem;

static int al_mem_setup(al_mem *m, int mode) {
    DWORD old;
    memset(m, 0, sizeof *m);
    m->mode = mode;
    if (mode == M_HEAP) {
        m->base = (uint8_t *)VirtualAlloc(NULL, 3 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    } else if (mode == M_RESERVE) {
        m->base = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE, PAGE_NOACCESS);
        if (m->base && !VirtualAlloc(m->base, PG, MEM_COMMIT, PAGE_READWRITE)) return 0;
    } else {
        m->base = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (m->base && mode == M_NOACCESS && !VirtualProtect(m->base + PG, PG, PAGE_NOACCESS, &old)) return 0;
    }
    if (!m->base) return 0;
    m->ref = m->base + PG;
    return 1;
}
/* bytes that C code may touch right now without faulting */
static int al_readable(const al_mem *m, const uint8_t *a) {
    if (a < m->base) return 0;
    size_t o = (size_t)(a - m->base);
    if (m->mode == M_HEAP) return o < 3 * PG;
    if (m->mode == M_READONLY) return o < 2 * PG;
    return o < PG;
}
static void al_mem_refill(al_mem *m) {
    DWORD old;
    size_t n = m->mode == M_HEAP ? 3 * PG : (m->mode == M_READONLY || m->mode == M_GUARD) ? 2 * PG : PG;
    if (m->mode == M_READONLY || m->mode == M_GUARD) VirtualProtect(m->ref, PG, PAGE_READWRITE, &old);
    for (size_t i = 0; i < n; i++) m->base[i] = al_pat(i);
    if (m->mode == M_READONLY) VirtualProtect(m->ref, PG, PAGE_READONLY, &old);
    if (m->mode == M_GUARD) VirtualProtect(m->ref, PG, PAGE_READWRITE | PAGE_GUARD, &old);
}
static int al_unchanged(const al_mem *m, const uint8_t *lo, const uint8_t *hi) {
    for (const uint8_t *a = lo; a < hi; a++)
        if (al_readable(m, a) && *a != al_pat((size_t)(a - m->base))) return 0;
    return 1;
}
static void al_aux_init(void) {
    for (int i = 0; i < 64; i++) g_aux[i] = (uint8_t)(0xA1 + i * 7);
    memset(g_aux + 64, 0xEE, 64);
}
static int al_parse_mode(const char *s) {
    for (int i = 0; i < (int)T_COUNTOF(g_mode_names); i++) if (!strcmp(s, g_mode_names[i])) return i;
    return -1;
}

/* --child=al_probe <stub> <mode> <lo> <hi> <step>
 * prints per offset: R off f code np i0 i1 i1rel reg post mem data ipok retry */
CHILD(al_probe) {
    if (argc < 6) return 3;
    const al_stub *s = al_find_stub(argv[1]);
    int mode = al_parse_mode(argv[2]);
    int lo = atoi(argv[3]), hi = atoi(argv[4]), step = atoi(argv[5]);
    if (!s || mode < 0 || step <= 0) return 4;
    if (!cpu_has((t_feature)s->feat)) { printf("UNSUPPORTED\n"); return 5; }
    al_mem m;
    if (!al_mem_setup(&m, mode)) { printf("SETUP_FAILED %lu\n", GetLastError()); return 6; }
    static T_ALIGN(64) uint8_t img[1024];
    if (s->kind == SK_RESTORE) {
        memset(img, 0, sizeof img);
        if (s->rsz == 512) t_fxsave(img); else t_xsave(img, 3);
    }
    if (!AddVectoredExceptionHandler(1, al_veh)) return 7;
    for (int off = lo; off <= hi; off += step) {
        uint8_t *p = m.ref + off;
        al_mem_refill(&m);
        if (s->kind == SK_RESTORE) memcpy(p, img, (size_t)s->rsz);
        al_aux_init();
        memset(g_aux + 128, 0xFF, 64);           /* masks: all lanes active */
        int f = al_call(s, p);
        DWORD code = f ? g_v.code : 0, np = f ? g_v.np : 0;
        unsigned long long i0 = f ? g_v.i0 : 0, i1 = f ? g_v.i1 : 0;
        long long i1rel = f ? (long long)(i1 - (unsigned long long)(uintptr_t)m.ref) : 0;
        int ipok = f ? g_v.ipok : -1, reg = -1, post = -1, mem = -1, data = -1, retry = -1;
        if (f && s->regw) {
            if (s->gpr) reg = memcmp(&g_v.ax, g_aux, (size_t)PTRW) == 0;
            else if (g_v.xmm_valid) reg = memcmp(g_v.xmm0, g_aux, 16) == 0;
            post = memcmp(g_aux + 64, g_aux, (size_t)s->regw) == 0;
        }
        if (f && s->kind == SK_STORE) mem = al_unchanged(&m, p - 64, p + s->w + 64);
        int ok_access = !f;
        if (f && mode == M_GUARD) {   /* guard pages are one-shot: the same access must now succeed */
            al_aux_init();
            int f2 = al_call(s, p);
            retry = !f2;
            ok_access = !f2;
        }
        if (ok_access && s->kind == SK_LOAD && s->pure) {
            data = memcmp(g_aux + 64, p, (size_t)s->w) == 0;
            for (int i = s->w; i < s->regw; i++) if (g_aux[64 + i] != 0) data = 0;
        }
        if (ok_access && s->kind == SK_STORE && s->fault != al_avx_vmaskmovps_st) {
            data = memcmp(p, g_aux, (size_t)s->w) == 0;
            if (mode != M_GUARD) data = data && al_unchanged(&m, p - 64, p) && al_unchanged(&m, p + s->w, p + s->w + 64);
        }
        printf("R %d %d %08lx %lu %llx %llx %lld %d %d %d %d %d %d\n", off, f, (unsigned long)code, (unsigned long)np,
               i0, i1, i1rel, reg, post, mem, data, ipok, retry);
    }
    printf("DONE\n");
    fflush(stdout);
    return 0;
}

/* --child=al_maskpage <stub> <lanes>  (dword lanes; p = boundary - 4k, lanes [0,k) active)
 * prints: M k full f code i0 i1rel data */
CHILD(al_maskpage) {
    if (argc < 3) return 3;
    const al_stub *s = al_find_stub(argv[1]);
    int lanes = atoi(argv[2]);
    if (!s || lanes < 1 || lanes > 16) return 4;
    if (!cpu_has((t_feature)s->feat)) { printf("UNSUPPORTED\n"); return 5; }
    al_mem m;
    if (!al_mem_setup(&m, M_NOACCESS)) return 6;
    if (!AddVectoredExceptionHandler(1, al_veh)) return 7;
    for (int k = 0; k <= lanes; k++) {
        for (int full = 0; full <= 1; full++) {
            int active = full ? lanes : k;
            uint8_t *p = m.ref - 4 * k;
            al_mem_refill(&m);
            al_aux_init();
            memset(g_aux + 128, 0, 64);
            if (lanes == 8) { for (int i = 0; i < active; i++) g_aux[128 + 4 * i + 3] = 0x80; }
            else { uint16_t km = (uint16_t)((active >= 16) ? 0xFFFF : ((1u << active) - 1)); memcpy(g_aux + 128, &km, 2); }
            int f = al_call(s, p);
            int data = -1;
            if (!f) {
                data = 1;
                if (s->kind == SK_LOAD) {
                    for (int i = 0; i < lanes; i++) {
                        uint8_t z[4] = { 0, 0, 0, 0 };
                        const uint8_t *want = i < active ? p + 4 * i : z;
                        if (memcmp(g_aux + 64 + 4 * i, want, 4)) data = 0;
                    }
                } else {
                    for (int i = 0; i < active; i++) if (memcmp(p + 4 * i, g_aux + 4 * i, 4)) data = 0;
                    if (!al_unchanged(&m, p - 64, p)) data = 0;
                }
            }
            printf("M %d %d %d %08lx %llx %lld %d\n", k, full, f, (unsigned long)(f ? g_v.code : 0),
                   (unsigned long long)(f ? g_v.i0 : 0), f ? (long long)(g_v.i1 - (uintptr_t)m.ref) : 0LL, data);
        }
    }
    printf("DONE\n");
    fflush(stdout);
    return 0;
}

/* --child=al_repfault <dst|src>
 * rep movsb running into a PAGE_NOACCESS page; the handler records RCX/RSI/RDI, makes the page
 * accessible and restarts the instruction, which must complete the copy.
 * prints: Q k n f code i0 i1rel di si cx pre fin ipok */
CHILD(al_repfault) {
    if (argc < 2) return 3;
    int dstmode = !strcmp(argv[1], "dst");
    al_mem m, h;
    if (!al_mem_setup(&m, M_NOACCESS) || !al_mem_setup(&h, M_HEAP)) return 6;
    if (!AddVectoredExceptionHandler(1, al_veh)) return 7;
    static const int ks[] = { 1, 2, 15, 16, 63, 64, 100, 1000, 4095 };
    for (size_t t = 0; t < T_COUNTOF(ks); t++) {
        int k = ks[t];
        size_t n = (size_t)k + 300;
        DWORD old;
        al_mem_refill(&h);
        for (size_t i = 0; i < PG; i++) m.base[i] = (uint8_t)(al_pat(i) ^ 0x3C);
        uint8_t *d, *s;
        if (dstmode) { d = m.ref - k; s = h.base + 123; memset(d, 0, (size_t)k); }
        else { s = m.ref - k; d = h.base + 77; memset(d, 0, n); }
        g_v.fault = (uintptr_t)al_repmovsb; g_v.end = (uintptr_t)al_repmovsb_end;
        g_v.action = ACT_UNPROTECT; g_v.up = m.ref; g_v.up_len = PG;
        g_v.rep_dst = d; g_v.rep_src = s; g_v.rep_lim = k; g_v.rep_pre = -1;
        g_v.hit = 0; g_v.ipok = 0;
        g_v.armed = 1;
        ((al_fn3)(void *)al_repmovsb_entry)(d, s, n);
        g_v.armed = 0;
        int f = g_v.hit;
        int fin = memcmp(d, s, n) == 0;
        printf("Q %d %u %d %08lx %llx %lld %lld %lld %llu %d %d %d\n", k, (unsigned)n, f, (unsigned long)(f ? g_v.code : 0),
               (unsigned long long)(f ? g_v.i0 : 0), f ? (long long)(g_v.i1 - (uintptr_t)m.ref) : 0LL,
               f ? (long long)(g_v.di - (uintptr_t)d) : 0LL, f ? (long long)(g_v.si - (uintptr_t)s) : 0LL,
               (unsigned long long)(f ? g_v.cx : 0), g_v.rep_pre, fin, f ? g_v.ipok : -1);
        VirtualProtect(m.ref, PG, PAGE_NOACCESS, &old);
    }
    printf("DONE\n");
    fflush(stdout);
    return 0;
}

/* --child=al_unhandled <movaps|noaccess> : the fault is NOT handled; the harness filter reports it. */
CHILD(al_unhandled) {
    if (argc < 2) return 3;
    static T_ALIGN(64) uint8_t buf[128];
    al_aux_init();
    if (!strcmp(argv[1], "movaps")) {
        printf("FAULT=0x%llx\n", (unsigned long long)(uintptr_t)al_movaps_ld);
        fflush(stdout);
        ((al_fn2)(void *)al_movaps_ld_entry)(buf + 8, g_aux);
    } else {
        al_mem m;
        if (!al_mem_setup(&m, M_NOACCESS)) return 6;
        printf("FAULT=0x%llx\nREF=0x%llx\n", (unsigned long long)(uintptr_t)al_movdqu_ld, (unsigned long long)(uintptr_t)m.ref);
        fflush(stdout);
        ((al_fn2)(void *)al_movdqu_ld_entry)(m.ref - 8, g_aux);
    }
    printf("SURVIVED\n");
    return 0;
}

/* --child=al_acflag : EFLAGS.AC=1 + misaligned load (INFO only: whether alignment checking is enabled is OS policy) */
CHILD(al_acflag) {
    static T_ALIGN(64) uint8_t buf[64];
    if (!AddVectoredExceptionHandler(1, al_veh)) return 7;
    memset(buf, 0x11, sizeof buf);
    int f = al_call(al_find_stub("al_acprobe"), buf + 1);
    printf("AC f=%d code=%08lx\n", f, (unsigned long)(f ? g_v.code : 0));
    fflush(stdout);
    return 0;
}

/* --child=al_splitlock : LOCK XADD on a dword that crosses a cache line (INFO on fault: split-lock
 * detection is OS policy; architecturally LOCK works on any alignment). */
CHILD(al_splitlock) {
    static T_ALIGN(64) uint8_t buf[192];
    memset(buf, 0, sizeof buf);
    volatile uint32_t *x = (volatile uint32_t *)(buf + 62);
    for (int i = 0; i < 16; i++) {
        uint32_t one = 1;
        __asm__ volatile("lock xaddl %0, %1" : "+r"(one), "+m"(*x) :: "memory");
    }
    printf("SL val=%u\n", (unsigned)*x);
    fflush(stdout);
    return 0;
}

/* --child=al_alignedbad : _aligned_malloc with a non-power-of-two alignment */
static void __cdecl al_iph(const wchar_t *e, const wchar_t *f, const wchar_t *fi, unsigned l, uintptr_t r) {
    (void)e; (void)f; (void)fi; (void)l; (void)r;
}
CHILD(al_alignedbad) {
    _set_invalid_parameter_handler(al_iph);
    static const size_t bad[] = { 3, 6, 24, 48, 100 };
    for (size_t i = 0; i < T_COUNTOF(bad); i++) {
        errno = 0;
        void *p = _aligned_malloc(64, bad[i]);
        printf("AB %u %d %d\n", (unsigned)bad[i], p == NULL, errno);
        if (p) _aligned_free(p);
    }
    fflush(stdout);
    return 0;
}

/* ======================================================================================
 * Parent side: run probe children and check rows
 * ====================================================================================== */
typedef struct { int off, f; unsigned long code, np; unsigned long long i0, i1; long long i1rel; int reg, post, mem, data, ipok, retry; } al_row;
static al_row g_rows[700];
static t_child_result g_cr;

static int al_child_ok(const char *what) {
    REQUIREF(g_cr.started, "%s: spawn failed (error %lu)", what, g_cr.spawn_error);
    REQUIREF(!g_cr.timed_out, "%s: child timed out", what);
    REQUIREF(g_cr.exc_code == 0, "%s: child died of exception 0x%08lx at 0x%llx (a fault not at the probe instruction, or unhandled)",
             what, g_cr.exc_code, g_cr.exc_addr);
    REQUIREF(g_cr.exit_code == 0, "%s: child exit code 0x%lx; output: %.300s", what, g_cr.exit_code, g_cr.out);
    return 1;
}

static int al_probe(const char *stub, const char *mode, int lo, int hi, int step) {
    char args[256];
    snprintf(args, sizeof args, "%s %s %d %d %d", stub, mode, lo, hi, step);
    t_child_run("al_probe", args, 30000, &g_cr);
    al_child_ok(args);
    int n = 0;
    for (const char *l = g_cr.out; l && *l; l = strchr(l, '\n'), l = l ? l + 1 : NULL) {
        if (l[0] != 'R' || l[1] != ' ' || n >= (int)T_COUNTOF(g_rows)) continue;
        al_row *r = &g_rows[n];
        if (sscanf(l + 2, "%d %d %lx %lu %llx %llx %lld %d %d %d %d %d %d", &r->off, &r->f, &r->code, &r->np, &r->i0, &r->i1,
                   &r->i1rel, &r->reg, &r->post, &r->mem, &r->data, &r->ipok, &r->retry) == 13) n++;
    }
    int want = (hi - lo) / step + 1;
    REQUIREF(n == want && strstr(g_cr.out, "DONE"), "%s: parsed %d rows, want %d", args, n, want);
    return n;
}

static void al_record_first_fault(const al_row *r, const char *stub) {
    char k[96];
    snprintf(k, sizeof k, "%s_exception", stub);
    t_info(k, "code=0x%08lx np=%lu info0=0x%llx info1=0x%llx", r->code, r->np, r->i0, r->i1);
    if (r->code != 0xC0000005ul)
        t_note("%s: misalignment #GP reported as 0x%08lx (Windows normally reports STATUS_ACCESS_VIOLATION)", stub, r->code);
}

/* Offsets 0..hi relative to a page-aligned address; fault expected iff off % align != 0 (align 0 = never). */
static void al_expect_align(const char *stub, int align, int hi) {
    const al_stub *s = al_find_stub(stub);
    REQUIREF(s, "unknown stub %s", stub);
    if (!cpu_has((t_feature)s->feat)) SKIP("%s not usable on this CPU/OS", cpu_feature_name((t_feature)s->feat));
    int n = al_probe(stub, "heap", 0, hi, 1), faults = 0, first = 1;
    for (int i = 0; i < n; i++) {
        const al_row *r = &g_rows[i];
        int want = align > 1 && (r->off % align) != 0;
        CHECKF(r->f == want, "%s off=%d: %s (code 0x%08lx)", stub, r->off, want ? "expected a fault (misaligned operand) but none" : "unexpected fault", r->code);
        if (r->f) {
            faults++;
            if (first) { al_record_first_fault(r, stub); first = 0; }
            CHECKF(r->ipok == 1, "%s off=%d: CONTEXT instruction pointer != ExceptionAddress", stub, r->off);
            if (r->post >= 0) CHECKF(r->post == 1, "%s off=%d: destination register modified by a faulting instruction (or not restored after the handler continued)", stub, r->off);
            if (r->reg >= 0) CHECKF(r->reg == 1, "%s off=%d: exception CONTEXT does not hold the pre-fault register value", stub, r->off);
            if (s->kind == SK_STORE) CHECKF(r->mem == 1, "%s off=%d: memory written by a store that raised #GP", stub, r->off);
        } else if (r->data >= 0) {
            CHECKF(r->data == 1, "%s off=%d: wrong data (or bytes outside the operand modified)", stub, r->off);
        }
    }
    t_info(stub, "faults at %d of %d offsets", faults, n);
}

/* Offsets around a boundary page whose protection is `mode`. Straddling accesses fault (except loads
 * from a PAGE_READONLY page). Checks exception code, read/write flag, address range and precision. */
static void al_expect_boundary(const char *stub, const char *mode) {
    const al_stub *s = al_find_stub(stub);
    REQUIREF(s, "unknown stub %s", stub);
    if (!cpu_has((t_feature)s->feat)) SKIP("%s not usable on this CPU/OS", cpu_feature_name((t_feature)s->feat));
    int w = s->w, is_store = s->kind == SK_STORE;
    int ro = !strcmp(mode, "readonly"), guard = !strcmp(mode, "guard");
    int n = al_probe(stub, mode, -(w + 4), 1, 1), partial = 0, first = 1, faults = 0;
    for (int i = 0; i < n; i++) {
        const al_row *r = &g_rows[i];
        int straddle = r->off + w > 0;
        int want = straddle && !(ro && !is_store);
        CHECKF(r->f == want, "%s/%s off=%d: %s", stub, mode, r->off, want ? "access into the protected page did not fault" : "unexpected fault");
        if (r->f) {
            faults++;
            if (first) {
                char k[96]; snprintf(k, sizeof k, "%s/%s", stub, mode);
                t_info(k, "code=0x%08lx np=%lu info0=0x%llx info1-boundary=%lld (off %d)", r->code, r->np, r->i0, r->i1rel, r->off);
                first = 0;
            }
            CHECKF(r->ipok == 1, "%s/%s off=%d: CONTEXT instruction pointer != ExceptionAddress", stub, mode, r->off);
            if (guard) {
                CHECKF(r->code == 0x80000001ul, "%s/guard off=%d: code 0x%08lx, want STATUS_GUARD_PAGE_VIOLATION", stub, r->off, r->code);
                CHECKF(r->retry == 1, "%s/guard off=%d: retry after the guard exception faulted again (PAGE_GUARD must be one-shot)", stub, r->off);
                if (r->np >= 2 && !(r->i1rel >= (r->off > 0 ? r->off : 0) && r->i1rel < r->off + w))
                    t_note("%s/guard off=%d: guard-page address info1-boundary=%lld outside the access", stub, r->off, r->i1rel);
            } else {
                CHECKF(r->code == 0xC0000005ul, "%s/%s off=%d: code 0x%08lx, want STATUS_ACCESS_VIOLATION", stub, mode, r->off, r->code);
                CHECKF(r->np >= 2, "%s/%s off=%d: NumberParameters %lu < 2", stub, mode, r->off, r->np);
                CHECKF(r->i0 == (unsigned long long)is_store, "%s/%s off=%d: ExceptionInformation[0]=%llu, want %d (%s)", stub, mode, r->off, r->i0, is_store, is_store ? "write" : "read");
                long long lo = r->off > 0 ? r->off : 0;
                CHECKF(r->i1rel >= lo && r->i1rel < r->off + w, "%s/%s off=%d: fault address boundary%+lld not within the inaccessible part of the access [%lld,%d)",
                       stub, mode, r->off, r->i1rel, lo, r->off + w);
            }
            if (r->post >= 0) CHECKF(r->post == 1, "%s/%s off=%d: destination register changed by the faulting load", stub, mode, r->off);
            if (r->reg >= 0) CHECKF(r->reg == 1, "%s/%s off=%d: CONTEXT register differs from the pre-fault value", stub, mode, r->off);
            if (is_store && r->mem == 0) partial++;
        }
        if (r->data >= 0) CHECKF(r->data == 1, "%s/%s off=%d: wrong data after a %s access", stub, mode, r->off, r->f ? "retried" : "non-faulting");
    }
    if (is_store && partial) t_note("%s/%s: %d faulting straddling stores wrote part of the accessible page (allowed, implementation-specific)", stub, mode, partial);
    (void)faults;
}

/* ======================================================================================
 * In-process (non-faulting) access tables
 * ====================================================================================== */
typedef void (*al_io_fn)(void *p, void *o);
typedef struct { const char *name; al_io_fn fn; int w, outw, layout, feat; } al_io;  /* layout: 0 low, 1 high-8 / src off */

#define AL_LD(fn, body) static T_NOINLINE void fn(void *p, void *o) { __asm__ volatile(body :: "r"(p), "r"(o) : "xmm0", "eax", "memory"); }
#define AL_LD_T(fn, tgt, body) static T_TARGET(tgt) T_NOINLINE void fn(void *p, void *o) { __asm__ volatile(body :: "r"(p), "r"(o) : "xmm0", "memory"); }

AL_LD(ld_movups, "movups (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movupd, "movupd (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movdqu, "movdqu (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_lddqu,  "lddqu (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movss,  "pcmpeqb %%xmm0, %%xmm0\n\tmovss (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movsd,  "pcmpeqb %%xmm0, %%xmm0\n\tmovsd (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movd,   "pcmpeqb %%xmm0, %%xmm0\n\tmovd (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movq,   "pcmpeqb %%xmm0, %%xmm0\n\tmovq (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movlps, "pxor %%xmm0, %%xmm0\n\tmovlps (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movlpd, "pxor %%xmm0, %%xmm0\n\tmovlpd (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movhps, "pxor %%xmm0, %%xmm0\n\tmovhps (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_movhpd, "pxor %%xmm0, %%xmm0\n\tmovhpd (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_pinsrw, "pxor %%xmm0, %%xmm0\n\tpinsrw $0, (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD_T(ld_pinsrb, "sse4.1", "pxor %%xmm0, %%xmm0\n\tpinsrb $0, (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD_T(ld_pinsrd, "sse4.1", "pxor %%xmm0, %%xmm0\n\tpinsrd $0, (%0), %%xmm0\n\tmovdqu %%xmm0, (%1)")
AL_LD(ld_gpr8,   "movzbl (%0), %%eax\n\tmovl %%eax, (%1)")
AL_LD(ld_gpr16,  "movzwl (%0), %%eax\n\tmovl %%eax, (%1)")
AL_LD(ld_gpr32,  "movl (%0), %%eax\n\tmovl %%eax, (%1)")
#ifdef T_X64
AL_LD(ld_gpr64,  "movq (%0), %%rax\n\tmovq %%rax, (%1)")
#endif
AL_LD_T(ld_avx_vmovups_y, "avx", "vmovups (%0), %%ymm0\n\tvmovdqu %%ymm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx_vmovdqu_y, "avx", "vmovdqu (%0), %%ymm0\n\tvmovdqu %%ymm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx_vlddqu_y,  "avx", "vlddqu (%0), %%ymm0\n\tvmovdqu %%ymm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx_vmovdqu_x, "avx", "vpcmpeqb %%ymm0, %%ymm0, %%ymm0\n\tvmovdqu (%0), %%xmm0\n\tvmovdqu %%ymm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx_vmovq,     "avx", "vpcmpeqb %%ymm0, %%ymm0, %%ymm0\n\tvmovq (%0), %%xmm0\n\tvmovdqu %%ymm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx_vmovd,     "avx", "vpcmpeqb %%ymm0, %%ymm0, %%ymm0\n\tvmovd (%0), %%xmm0\n\tvmovdqu %%ymm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx512_vmovdqu64_z, "avx512f", "vmovdqu64 (%0), %%zmm0\n\tvmovdqu64 %%zmm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx512_vmovups_z,   "avx512f", "vmovups (%0), %%zmm0\n\tvmovdqu64 %%zmm0, (%1)\n\tvzeroupper")
AL_LD_T(ld_avx512_vmovdqu8_z,  "avx512f,avx512bw", "vmovdqu8 (%0), %%zmm0\n\tvmovdqu64 %%zmm0, (%1)\n\tvzeroupper")

static const al_io g_ld_sse_full[] = {
    { "movups", ld_movups, 16, 16, 0, F_SSE }, { "movupd", ld_movupd, 16, 16, 0, F_SSE2 },
    { "movdqu", ld_movdqu, 16, 16, 0, F_SSE2 }, { "lddqu", ld_lddqu, 16, 16, 0, F_SSE3 },
};
static const al_io g_ld_sse_part[] = {
    { "movss", ld_movss, 4, 16, 0, F_SSE }, { "movsd", ld_movsd, 8, 16, 0, F_SSE2 },
    { "movd", ld_movd, 4, 16, 0, F_SSE2 }, { "movq", ld_movq, 8, 16, 0, F_SSE2 },
    { "movlps", ld_movlps, 8, 16, 0, F_SSE }, { "movlpd", ld_movlpd, 8, 16, 0, F_SSE2 },
    { "movhps", ld_movhps, 8, 16, 1, F_SSE }, { "movhpd", ld_movhpd, 8, 16, 1, F_SSE2 },
    { "pinsrw", ld_pinsrw, 2, 16, 0, F_SSE2 }, { "pinsrb", ld_pinsrb, 1, 16, 0, F_SSE41 },
    { "pinsrd", ld_pinsrd, 4, 16, 0, F_SSE41 },
};
static const al_io g_ld_gpr[] = {
    { "movzbl", ld_gpr8, 1, 4, 0, F_SSE2 }, { "movzwl", ld_gpr16, 2, 4, 0, F_SSE2 }, { "movl", ld_gpr32, 4, 4, 0, F_SSE2 },
#ifdef T_X64
    { "movq_gpr", ld_gpr64, 8, 8, 0, F_SSE2 },
#endif
};
static const al_io g_ld_avx[] = {
    { "vmovups_ymm", ld_avx_vmovups_y, 32, 32, 0, F_AVX }, { "vmovdqu_ymm", ld_avx_vmovdqu_y, 32, 32, 0, F_AVX },
    { "vlddqu_ymm", ld_avx_vlddqu_y, 32, 32, 0, F_AVX }, { "vmovdqu_xmm", ld_avx_vmovdqu_x, 16, 32, 0, F_AVX },
    { "vmovq", ld_avx_vmovq, 8, 32, 0, F_AVX }, { "vmovd", ld_avx_vmovd, 4, 32, 0, F_AVX },
};
static const al_io g_ld_avx512[] = {
    { "vmovdqu64_zmm", ld_avx512_vmovdqu64_z, 64, 64, 0, F_AVX512F }, { "vmovups_zmm", ld_avx512_vmovups_z, 64, 64, 0, F_AVX512F },
    { "vmovdqu8_zmm", ld_avx512_vmovdqu8_z, 64, 64, 0, F_AVX512BW },
};

/* stores: o = 64-byte source; the op stores w bytes taken from source offset `layout` */
#define AL_ST(fn, body) static T_NOINLINE void fn(void *p, void *o) { __asm__ volatile(body :: "r"(p), "r"(o) : "xmm0", "xmm1", "eax", "memory"); }
#define AL_ST_T(fn, tgt, body) static T_TARGET(tgt) T_NOINLINE void fn(void *p, void *o) { __asm__ volatile(body :: "r"(p), "r"(o) : "xmm0", "memory"); }
AL_ST(st_movups, "movdqu (%1), %%xmm0\n\tmovups %%xmm0, (%0)")
AL_ST(st_movupd, "movdqu (%1), %%xmm0\n\tmovupd %%xmm0, (%0)")
AL_ST(st_movdqu, "movdqu (%1), %%xmm0\n\tmovdqu %%xmm0, (%0)")
AL_ST(st_movss,  "movdqu (%1), %%xmm0\n\tmovss %%xmm0, (%0)")
AL_ST(st_movsd,  "movdqu (%1), %%xmm0\n\tmovsd %%xmm0, (%0)")
AL_ST(st_movd,   "movdqu (%1), %%xmm0\n\tmovd %%xmm0, (%0)")
AL_ST(st_movq,   "movdqu (%1), %%xmm0\n\tmovq %%xmm0, (%0)")
AL_ST(st_movlps, "movdqu (%1), %%xmm0\n\tmovlps %%xmm0, (%0)")
AL_ST(st_movlpd, "movdqu (%1), %%xmm0\n\tmovlpd %%xmm0, (%0)")
AL_ST(st_movhps, "movdqu (%1), %%xmm0\n\tmovhps %%xmm0, (%0)")
AL_ST(st_movhpd, "movdqu (%1), %%xmm0\n\tmovhpd %%xmm0, (%0)")
AL_ST(st_movnti, "movl (%1), %%eax\n\tmovnti %%eax, (%0)\n\tsfence")
AL_ST(st_gpr16,  "movl (%1), %%eax\n\tmovw %%ax, (%0)")
AL_ST(st_gpr32,  "movl (%1), %%eax\n\tmovl %%eax, (%0)")
#ifdef T_X64
AL_ST(st_gpr64,  "movq (%1), %%rax\n\tmovq %%rax, (%0)")
AL_ST(st_movnti64, "movq (%1), %%rax\n\tmovnti %%rax, (%0)\n\tsfence")
#endif
AL_ST_T(st_pextrb, "sse4.1", "movdqu (%1), %%xmm0\n\tpextrb $0, %%xmm0, (%0)")
AL_ST_T(st_pextrw, "sse4.1", "movdqu (%1), %%xmm0\n\tpextrw $0, %%xmm0, (%0)")
AL_ST_T(st_pextrd, "sse4.1", "movdqu (%1), %%xmm0\n\tpextrd $0, %%xmm0, (%0)")
AL_ST_T(st_extractps, "sse4.1", "movdqu (%1), %%xmm0\n\textractps $0, %%xmm0, (%0)")
static T_NOINLINE void st_maskmovdqu(void *p, void *o) {   /* implicit [rdi]; all mask bytes set */
    __asm__ volatile("movdqu (%1), %%xmm0\n\tpcmpeqb %%xmm1, %%xmm1\n\tmaskmovdqu %%xmm1, %%xmm0\n\tsfence"
                     : "+D"(p) : "r"(o) : "xmm0", "xmm1", "memory");
}
AL_ST_T(st_avx_vmovups_y, "avx", "vmovdqu (%1), %%ymm0\n\tvmovups %%ymm0, (%0)\n\tvzeroupper")
AL_ST_T(st_avx_vmovdqu_y, "avx", "vmovdqu (%1), %%ymm0\n\tvmovdqu %%ymm0, (%0)\n\tvzeroupper")
AL_ST_T(st_avx_vmovdqu_x, "avx", "vmovdqu (%1), %%ymm0\n\tvmovdqu %%xmm0, (%0)\n\tvzeroupper")
AL_ST_T(st_avx_vextractf128, "avx", "vmovdqu (%1), %%ymm0\n\tvextractf128 $1, %%ymm0, (%0)\n\tvzeroupper")
AL_ST_T(st_avx_vmovq, "avx", "vmovdqu (%1), %%ymm0\n\tvmovq %%xmm0, (%0)\n\tvzeroupper")
AL_ST_T(st_avx512_vmovdqu64_z, "avx512f", "vmovdqu64 (%1), %%zmm0\n\tvmovdqu64 %%zmm0, (%0)\n\tvzeroupper")
AL_ST_T(st_avx512_vextracti64x4, "avx512f", "vmovdqu64 (%1), %%zmm0\n\tvextracti64x4 $1, %%zmm0, (%0)\n\tvzeroupper")

static const al_io g_st_sse[] = {
    { "movups", st_movups, 16, 0, 0, F_SSE }, { "movupd", st_movupd, 16, 0, 0, F_SSE2 }, { "movdqu", st_movdqu, 16, 0, 0, F_SSE2 },
    { "maskmovdqu", st_maskmovdqu, 16, 0, 0, F_SSE2 },
};
static const al_io g_st_sse_part[] = {
    { "movss", st_movss, 4, 0, 0, F_SSE }, { "movsd", st_movsd, 8, 0, 0, F_SSE2 }, { "movd", st_movd, 4, 0, 0, F_SSE2 },
    { "movq", st_movq, 8, 0, 0, F_SSE2 }, { "movlps", st_movlps, 8, 0, 0, F_SSE }, { "movlpd", st_movlpd, 8, 0, 0, F_SSE2 },
    { "movhps", st_movhps, 8, 0, 8, F_SSE }, { "movhpd", st_movhpd, 8, 0, 8, F_SSE2 },
    { "pextrb", st_pextrb, 1, 0, 0, F_SSE41 }, { "pextrw_m16", st_pextrw, 2, 0, 0, F_SSE41 },
    { "pextrd", st_pextrd, 4, 0, 0, F_SSE41 }, { "extractps", st_extractps, 4, 0, 0, F_SSE41 },
};
static const al_io g_st_gpr[] = {
    { "movw", st_gpr16, 2, 0, 0, F_SSE2 }, { "movl", st_gpr32, 4, 0, 0, F_SSE2 }, { "movnti32", st_movnti, 4, 0, 0, F_SSE2 },
#ifdef T_X64
    { "movq_gpr", st_gpr64, 8, 0, 0, F_SSE2 }, { "movnti64", st_movnti64, 8, 0, 0, F_SSE2 },
#endif
};
static const al_io g_st_avx[] = {
    { "vmovups_ymm", st_avx_vmovups_y, 32, 0, 0, F_AVX }, { "vmovdqu_ymm", st_avx_vmovdqu_y, 32, 0, 0, F_AVX },
    { "vmovdqu_xmm", st_avx_vmovdqu_x, 16, 0, 0, F_AVX }, { "vextractf128_m128", st_avx_vextractf128, 16, 0, 16, F_AVX },
    { "vmovq", st_avx_vmovq, 8, 0, 0, F_AVX },
};
static const al_io g_st_avx512[] = {
    { "vmovdqu64_zmm", st_avx512_vmovdqu64_z, 64, 0, 0, F_AVX512F }, { "vextracti64x4_m256", st_avx512_vextracti64x4, 32, 0, 32, F_AVX512F },
};

static T_ALIGN(64) uint8_t g_src[512];
static void al_src_init(void) { for (int i = 0; i < (int)sizeof g_src; i++) g_src[i] = al_pat((size_t)i * 3 + 1); }

/* load e from p; compare with what p holds (zero-extended to outw, rest untouched) */
static int al_ld_check(const al_io *e, const uint8_t *p, const char *ctx, int off) {
    T_ALIGN(64) uint8_t out[64], exp[64];
    char label[160];
    memset(out, 0xEE, sizeof out);
    memset(exp, 0xEE, sizeof exp);
    memset(exp, 0, (size_t)e->outw);
    memcpy(exp + (e->layout ? 8 : 0), p, (size_t)e->w);
    e->fn((void *)p, out);
    snprintf(label, sizeof label, "%s %s off=%d", e->name, ctx, off);
    return t_check_mem(out, exp, sizeof out, __FILE__, __LINE__, label);
}
/* store e at p inside window [win, win+wl); only [p, p+w) may change */
static int al_st_check(const al_io *e, uint8_t *win, size_t wl, uint8_t *p, const char *ctx, int off) {
    static T_ALIGN(64) uint8_t exp[1024];
    T_ALIGN(64) uint8_t in[64];
    char label[160];
    for (int i = 0; i < 64; i++) in[i] = (uint8_t)(0x81 + i * 5);
    memset(win, 0xCC, wl);
    memcpy(exp, win, wl);
    memcpy(exp + (p - win), in + e->layout, (size_t)e->w);
    e->fn(p, in);
    snprintf(label, sizeof label, "%s %s off=%d", e->name, ctx, off);
    return t_check_mem(win, exp, wl, __FILE__, __LINE__, label);
}

static void al_ld_all_offsets(const al_io *t, size_t n) {
    al_src_init();
    int done = 0;
    for (size_t i = 0; i < n; i++) {
        if (!cpu_has((t_feature)t[i].feat)) { t_note("%s skipped: %s not usable", t[i].name, cpu_feature_name((t_feature)t[i].feat)); continue; }
        for (int off = 0; off < 64; off++) al_ld_check(&t[i], g_src + 64 + off, "unaligned", off);
        done++;
    }
    if (!done) SKIP("no instruction of this group is usable");
}
static void al_st_all_offsets(const al_io *t, size_t n) {
    static T_ALIGN(64) uint8_t win[256];
    int done = 0;
    for (size_t i = 0; i < n; i++) {
        if (!cpu_has((t_feature)t[i].feat)) { t_note("%s skipped: %s not usable", t[i].name, cpu_feature_name((t_feature)t[i].feat)); continue; }
        for (int off = 0; off < 64; off++) al_st_check(&t[i], win, sizeof win, win + 64 + off, "unaligned", off);
        done++;
    }
    if (!done) SKIP("no instruction of this group is usable");
}

/* two committed pages: accesses at boundary+off for every off that straddles (and neighbours) */
static uint8_t *al_two_pages(void) {
    uint8_t *b = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (b) for (int i = 0; i < 2 * PG; i++) b[i] = al_pat((size_t)i);
    return b;
}
static void al_ld_straddle(const al_io *t, size_t n, uint8_t *boundary, const char *ctx) {
    for (size_t i = 0; i < n; i++) {
        if (!cpu_has((t_feature)t[i].feat)) continue;
        for (int off = -(t[i].w + 1); off <= 1; off++) al_ld_check(&t[i], boundary + off, ctx, off);
    }
}
static void al_st_straddle(const al_io *t, size_t n, uint8_t *boundary, const char *ctx) {
    for (size_t i = 0; i < n; i++) {
        if (!cpu_has((t_feature)t[i].feat)) continue;
        for (int off = -(t[i].w + 1); off <= 1; off++) al_st_check(&t[i], boundary - 128, 256, boundary + off, ctx, off);
    }
}

/* ======================================================================================
 * Cases: environment
 * ====================================================================================== */
TEST(env_page_size_and_granularity) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    t_info("page_size", "%lu", si.dwPageSize);
    t_info("allocation_granularity", "%lu", si.dwAllocationGranularity);
    t_info("large_page_minimum", "%llu", (unsigned long long)GetLargePageMinimum());
    CHECK_EQ(si.dwPageSize, 4096);   /* the only small-page size on x86/x64 */
    CHECKF(si.dwAllocationGranularity && (si.dwAllocationGranularity & (si.dwAllocationGranularity - 1)) == 0,
           "granularity %lu not a power of two", si.dwAllocationGranularity);
    CHECKF(si.dwAllocationGranularity % si.dwPageSize == 0, "granularity %lu not a page multiple", si.dwAllocationGranularity);
    if (si.dwAllocationGranularity != 65536) t_note("allocation granularity %lu (Windows uses 64 KiB)", si.dwAllocationGranularity);
    t_info("misalignsse_hw", "%d", cpu_hw(F_MISALIGNSSE));
}

TEST(env_mxcsr_misaligned_sse_mode_off) {
    /* AMD MXCSR.MM (bit 17) relaxes legacy-SSE alignment. Windows' initial MXCSR is 0x1F80, so MM is clear. */
    static T_ALIGN(16) uint8_t fx[512];
    t_fxsave(fx);
    uint32_t mask; memcpy(&mask, fx + 28, 4);
    if (!mask) mask = 0xFFBF;
    t_info("mxcsr", "0x%08x", t_get_mxcsr());
    t_info("mxcsr_mask", "0x%08x", mask);
    t_info("mm_supported", "%u", (unsigned)((mask >> 17) & 1));
    CHECKF((t_get_mxcsr() & (1u << 17)) == 0, "MXCSR.MM set: misaligned legacy SSE accesses would not fault");
    CHECK_EQ(t_get_mxcsr() & 0xFFFF0000u, 0);
}

/* ======================================================================================
 * Cases: unaligned access must work at every offset (in process)
 * ====================================================================================== */
TEST(unaligned_sse_full_loads_all_offsets) { al_ld_all_offsets(g_ld_sse_full, T_COUNTOF(g_ld_sse_full)); }
TEST(unaligned_sse_partial_loads_all_offsets) { al_ld_all_offsets(g_ld_sse_part, T_COUNTOF(g_ld_sse_part)); }
TEST(unaligned_gpr_loads_all_offsets) { al_ld_all_offsets(g_ld_gpr, T_COUNTOF(g_ld_gpr)); }
TEST(unaligned_avx_loads_all_offsets) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    al_ld_all_offsets(g_ld_avx, T_COUNTOF(g_ld_avx));
}
TEST(unaligned_avx512_loads_all_offsets) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    al_ld_all_offsets(g_ld_avx512, T_COUNTOF(g_ld_avx512));
}
TEST(unaligned_sse_full_stores_all_offsets) { al_st_all_offsets(g_st_sse, T_COUNTOF(g_st_sse)); }
TEST(unaligned_sse_partial_stores_all_offsets) { al_st_all_offsets(g_st_sse_part, T_COUNTOF(g_st_sse_part)); }
TEST(unaligned_gpr_and_movnti_stores_all_offsets) { al_st_all_offsets(g_st_gpr, T_COUNTOF(g_st_gpr)); }
TEST(unaligned_avx_stores_all_offsets) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    al_st_all_offsets(g_st_avx, T_COUNTOF(g_st_avx));
}
TEST(unaligned_avx512_stores_all_offsets) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    al_st_all_offsets(g_st_avx512, T_COUNTOF(g_st_avx512));
}

/* ---- ALU memory operands without an alignment requirement: unaligned result == aligned result ---- */
typedef void (*al_op_fn)(const void *p, void *o, const void *init);
typedef struct { const char *name; al_op_fn fn; int rd, feat; } al_op;
#define AL_OP(fn, body) static T_NOINLINE void fn(const void *p, void *o, const void *i) { \
    __asm__ volatile(body :: "r"(p), "r"(o), "r"(i) : "xmm0", "xmm1", "cc", "memory"); }
#define AL_OPX(fn, tgt, body) static T_TARGET(tgt) T_NOINLINE void fn(const void *p, void *o, const void *i) { \
    __asm__ volatile(body :: "r"(p), "r"(o), "r"(i) : "xmm0", "xmm1", "cc", "memory"); }
#define AL_OPG(fn, tgt, body) static T_TARGET(tgt) T_NOINLINE void fn(const void *p, void *o, const void *i) { \
    __asm__ volatile(body :: "S"(p), "D"(o), "b"(i) : "xmm0", "xmm1", "eax", "ecx", "edx", "cc", "memory"); }
#define XI "movdqu (%2), %%xmm0\n\t"
#define XO "\n\tmovdqu %%xmm0, (%1)"
AL_OP(op_addss, XI "addss (%0), %%xmm0" XO)
AL_OP(op_addsd, XI "addsd (%0), %%xmm0" XO)
AL_OP(op_mulsd, XI "mulsd (%0), %%xmm0" XO)
AL_OP(op_divss, XI "divss (%0), %%xmm0" XO)
AL_OP(op_sqrtsd, XI "sqrtsd (%0), %%xmm0" XO)
AL_OP(op_minsd, XI "minsd (%0), %%xmm0" XO)
AL_OP(op_maxss, XI "maxss (%0), %%xmm0" XO)
AL_OP(op_cvtss2sd, XI "cvtss2sd (%0), %%xmm0" XO)
AL_OP(op_cvtsd2ss, XI "cvtsd2ss (%0), %%xmm0" XO)
AL_OP(op_cvtps2pd, XI "cvtps2pd (%0), %%xmm0" XO)
AL_OP(op_cvtdq2pd, XI "cvtdq2pd (%0), %%xmm0" XO)
AL_OP(op_cvtsi2sd, XI "cvtsi2sdl (%0), %%xmm0" XO)
AL_OP(op_comisd, XI "comisd (%0), %%xmm0\n\tsetz (%1)\n\tsetc 1(%1)\n\tsetp 2(%1)")
AL_OP(op_ucomiss, XI "ucomiss (%0), %%xmm0\n\tsetz (%1)\n\tsetc 1(%1)\n\tsetp 2(%1)")
AL_OP(op_pinsrw, XI "pinsrw $5, (%0), %%xmm0" XO)
AL_OPX(op_movddup, "sse3", XI "movddup (%0), %%xmm0" XO)
AL_OPX(op_pmovzxbw, "sse4.1", XI "pmovzxbw (%0), %%xmm0" XO)
AL_OPX(op_pmovsxbd, "sse4.1", XI "pmovsxbd (%0), %%xmm0" XO)
AL_OPX(op_pmovzxdq, "sse4.1", XI "pmovzxdq (%0), %%xmm0" XO)
AL_OPX(op_insertps, "sse4.1", XI "insertps $0x20, (%0), %%xmm0" XO)
AL_OPX(op_roundsd, "sse4.1", XI "roundsd $1, (%0), %%xmm0" XO)
#ifdef T_X64
#define GS "%%rsi"
#define GD "%%rdi"
#define GB "%%rbx"
#else
#define GS "%%esi"
#define GD "%%edi"
#define GB "%%ebx"
#endif
AL_OPG(op_pcmpistri, "sse4.2", "movdqu (" GB "), %%xmm0\n\tpcmpistri $0x0c, (" GS "), %%xmm0\n\tmovl %%ecx, (" GD ")\n\tsetc 4(" GD ")\n\tsetz 5(" GD ")\n\tsets 6(" GD ")")
AL_OPG(op_pcmpestri, "sse4.2", "movdqu (" GB "), %%xmm0\n\tmovl $13, %%eax\n\tmovl $16, %%edx\n\tpcmpestri $0x08, (" GS "), %%xmm0\n\tmovl %%ecx, (" GD ")\n\tsetc 4(" GD ")")
AL_OPG(op_pcmpistrm, "sse4.2", "movdqu (" GB "), %%xmm1\n\tpcmpistrm $0x40, (" GS "), %%xmm1\n\tmovdqu %%xmm0, (" GD ")")

static const al_op g_ops_sse[] = {
    { "addss_m32", op_addss, 4, F_SSE }, { "addsd_m64", op_addsd, 8, F_SSE2 }, { "mulsd_m64", op_mulsd, 8, F_SSE2 },
    { "divss_m32", op_divss, 4, F_SSE }, { "sqrtsd_m64", op_sqrtsd, 8, F_SSE2 }, { "minsd_m64", op_minsd, 8, F_SSE2 },
    { "maxss_m32", op_maxss, 4, F_SSE }, { "cvtss2sd_m32", op_cvtss2sd, 4, F_SSE2 }, { "cvtsd2ss_m64", op_cvtsd2ss, 8, F_SSE2 },
    { "cvtps2pd_m64", op_cvtps2pd, 8, F_SSE2 }, { "cvtdq2pd_m64", op_cvtdq2pd, 8, F_SSE2 }, { "cvtsi2sd_m32", op_cvtsi2sd, 4, F_SSE2 },
    { "comisd_m64", op_comisd, 8, F_SSE2 }, { "ucomiss_m32", op_ucomiss, 4, F_SSE }, { "pinsrw_m16", op_pinsrw, 2, F_SSE2 },
    { "movddup_m64", op_movddup, 8, F_SSE3 }, { "pmovzxbw_m64", op_pmovzxbw, 8, F_SSE41 }, { "pmovsxbd_m32", op_pmovsxbd, 4, F_SSE41 },
    { "pmovzxdq_m64", op_pmovzxdq, 8, F_SSE41 }, { "insertps_m32", op_insertps, 4, F_SSE41 }, { "roundsd_m64", op_roundsd, 8, F_SSE41 },
    { "pcmpistri_m128", op_pcmpistri, 16, F_SSE42 }, { "pcmpestri_m128", op_pcmpestri, 16, F_SSE42 },
    { "pcmpistrm_m128", op_pcmpistrm, 16, F_SSE42 },
};

#define YI "vmovdqu (%2), %%ymm0\n\t"
#define YO "\n\tvmovdqu %%ymm0, (%1)\n\tvzeroupper"
AL_OPX(op_avx_vpxor_x, "avx", YI "vpxor (%0), %%xmm0, %%xmm0" YO)
AL_OPX(op_avx_vpaddd_x, "avx", YI "vpaddd (%0), %%xmm0, %%xmm0" YO)
AL_OPX(op_avx_vpshufb_x, "avx", YI "vpshufb (%0), %%xmm0, %%xmm0" YO)
AL_OPX(op_avx_vpcmpeqb_x, "avx", YI "vpcmpeqb (%0), %%xmm0, %%xmm0" YO)
AL_OPX(op_avx_vaddps_y, "avx", YI "vaddps (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx_vandps_y, "avx", YI "vandps (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx_vmulpd_y, "avx", YI "vmulpd (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx_vperm2f128, "avx", YI "vperm2f128 $0x21, (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx_vinsertf128, "avx", YI "vinsertf128 $1, (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx_vbroadcastss, "avx", YI "vbroadcastss (%0), %%ymm0" YO)
AL_OPX(op_avx_vbroadcastf128, "avx", YI "vbroadcastf128 (%0), %%ymm0" YO)
AL_OPX(op_avx_vptest_y, "avx", YI "vptest (%0), %%ymm0\n\tsetz (%1)\n\tsetc 1(%1)\n\tvzeroupper")
AL_OPX(op_avx2_vpaddd_y, "avx2", YI "vpaddd (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx2_vpshufb_y, "avx2", YI "vpshufb (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx2_vpermd_y, "avx2", YI "vpermd (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_avx2_vpmovzxbw_y, "avx2", YI "vpmovzxbw (%0), %%ymm0" YO)
AL_OPX(op_avx2_vpbroadcastq, "avx2", YI "vpbroadcastq (%0), %%ymm0" YO)
AL_OPX(op_avx2_vpsllvd_y, "avx2", YI "vpsllvd (%0), %%ymm0, %%ymm0" YO)
AL_OPX(op_fma_vfmadd231ps_y, "avx,fma", YI "vmovdqu (%2), %%ymm1\n\tvfmadd231ps (%0), %%ymm1, %%ymm0" YO)
AL_OPX(op_f16c_vcvtph2ps_y, "avx,f16c", YI "vcvtph2ps (%0), %%ymm0" YO)

static const al_op g_ops_vex[] = {
    { "vpxor_x_m128", op_avx_vpxor_x, 16, F_AVX }, { "vpaddd_x_m128", op_avx_vpaddd_x, 16, F_AVX },
    { "vpshufb_x_m128", op_avx_vpshufb_x, 16, F_AVX }, { "vpcmpeqb_x_m128", op_avx_vpcmpeqb_x, 16, F_AVX },
    { "vaddps_y_m256", op_avx_vaddps_y, 32, F_AVX }, { "vandps_y_m256", op_avx_vandps_y, 32, F_AVX },
    { "vmulpd_y_m256", op_avx_vmulpd_y, 32, F_AVX }, { "vperm2f128_m256", op_avx_vperm2f128, 32, F_AVX },
    { "vinsertf128_m128", op_avx_vinsertf128, 16, F_AVX }, { "vbroadcastss_m32", op_avx_vbroadcastss, 4, F_AVX },
    { "vbroadcastf128_m128", op_avx_vbroadcastf128, 16, F_AVX }, { "vptest_y_m256", op_avx_vptest_y, 32, F_AVX },
    { "vpaddd_y_m256", op_avx2_vpaddd_y, 32, F_AVX2 }, { "vpshufb_y_m256", op_avx2_vpshufb_y, 32, F_AVX2 },
    { "vpermd_y_m256", op_avx2_vpermd_y, 32, F_AVX2 }, { "vpmovzxbw_y_m128", op_avx2_vpmovzxbw_y, 16, F_AVX2 },
    { "vpbroadcastq_m64", op_avx2_vpbroadcastq, 8, F_AVX2 }, { "vpsllvd_y_m256", op_avx2_vpsllvd_y, 32, F_AVX2 },
    { "vfmadd231ps_y_m256", op_fma_vfmadd231ps_y, 32, F_FMA }, { "vcvtph2ps_y_m128", op_f16c_vcvtph2ps_y, 16, F_F16C },
};

#define ZI "vmovdqu64 (%2), %%zmm0\n\t"
#define ZO "\n\tvmovdqu64 %%zmm0, (%1)\n\tvzeroupper"
AL_OPX(op_avx512_vpaddd_z, "avx512f", ZI "vpaddd (%0), %%zmm0, %%zmm0" ZO)
AL_OPX(op_avx512_vpxord_z, "avx512f", ZI "vpxord (%0), %%zmm0, %%zmm0" ZO)
AL_OPX(op_avx512_vaddps_z, "avx512f", ZI "vaddps (%0), %%zmm0, %%zmm0" ZO)
AL_OPX(op_avx512_vpermd_z, "avx512f", ZI "vpermd (%0), %%zmm0, %%zmm0" ZO)
AL_OPX(op_avx512_vpternlogd_z, "avx512f", ZI "vpternlogd $0x96, (%0), %%zmm0, %%zmm0" ZO)
AL_OPX(op_avx512_vpaddd_bcst, "avx512f", ZI "vpaddd (%0)%{1to16%}, %%zmm0, %%zmm0" ZO)
AL_OPX(op_avx512_vpshufb_z, "avx512f,avx512bw", ZI "vpshufb (%0), %%zmm0, %%zmm0" ZO)

static const al_op g_ops_evex[] = {
    { "vpaddd_z_m512", op_avx512_vpaddd_z, 64, F_AVX512F }, { "vpxord_z_m512", op_avx512_vpxord_z, 64, F_AVX512F },
    { "vaddps_z_m512", op_avx512_vaddps_z, 64, F_AVX512F }, { "vpermd_z_m512", op_avx512_vpermd_z, 64, F_AVX512F },
    { "vpternlogd_z_m512", op_avx512_vpternlogd_z, 64, F_AVX512F }, { "vpaddd_z_m32bcst", op_avx512_vpaddd_bcst, 4, F_AVX512F },
    { "vpshufb_z_m512", op_avx512_vpshufb_z, 64, F_AVX512BW },
};

static void al_ops_all_offsets(const al_op *t, size_t n) {
    static T_ALIGN(64) uint8_t init[64], al[64], o1[64], o2[64];
    al_src_init();
    for (int i = 0; i < 64; i++) init[i] = (uint8_t)(0x40 + i * 11);
    int done = 0;
    for (size_t k = 0; k < n; k++) {
        if (!cpu_has((t_feature)t[k].feat)) { t_note("%s skipped: %s not usable", t[k].name, cpu_feature_name((t_feature)t[k].feat)); continue; }
        for (int off = 0; off < 64; off++) {
            const uint8_t *p = g_src + 64 + off;
            memset(al, 0, sizeof al);
            memcpy(al, p, (size_t)t[k].rd);
            memset(o1, 0xEE, 64); memset(o2, 0xEE, 64);
            t[k].fn(p, o1, init);
            t[k].fn(al, o2, init);
            char label[128];
            snprintf(label, sizeof label, "%s unaligned off=%d vs 64-aligned copy", t[k].name, off);
            t_check_mem(o1, o2, 64, __FILE__, __LINE__, label);
        }
        done++;
    }
    if (!done) SKIP("no instruction of this group is usable");
}
TEST(unaligned_sse_scalar_and_partial_memory_operands) { al_ops_all_offsets(g_ops_sse, T_COUNTOF(g_ops_sse)); }
TEST(unaligned_vex_alu_memory_operands) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    al_ops_all_offsets(g_ops_vex, T_COUNTOF(g_ops_vex));
}
TEST(unaligned_evex_alu_memory_operands) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    al_ops_all_offsets(g_ops_evex, T_COUNTOF(g_ops_evex));
}

TEST(unaligned_known_answers) {
    /* independent reference values (exact arithmetic) for a few unaligned memory operands */
    static T_ALIGN(64) uint8_t buf[128], init[64], out[64];
    double d = 2.5, x = 1.25;
    memset(init, 0, sizeof init); memcpy(init, &x, 8);
    memset(buf, 0, sizeof buf); memcpy(buf + 3, &d, 8);
    op_addsd(buf + 3, out, init);
    double r; memcpy(&r, out, 8);
    CHECKF(r == 3.75, "addsd m64 at +3: %g", r);
    float fs = 6.0f, fi = 1.5f; float rf;
    memset(init, 0, sizeof init); memcpy(init, &fi, 4);
    memcpy(buf + 1, &fs, 4);
    op_divss(buf + 1, out, init); memcpy(&rf, out, 4);
    CHECKF(rf == 0.25f, "divss m32 at +1: %g", (double)rf);
    int32_t iv = -7;
    memcpy(buf + 5, &iv, 4);
    op_cvtsi2sd(buf + 5, out, init); memcpy(&r, out, 8);
    CHECKF(r == -7.0, "cvtsi2sd m32 at +5: %g", r);
    if (cpu_has(F_SSE42)) {
        /* pcmpistri, imm 0x08 (unsigned bytes, equal each, least-significant index): first EQUAL byte */
        static const char a[17] = "xlignment-test!!", b[17] = "alignmenT-test!!";
        memcpy(init, a, 16);
        memcpy(buf + 7, b, 16);
        buf[7 + 16] = 0;
        memset(out, 0, sizeof out);
        op_pcmpestri(buf + 7, out, init);   /* eax=13, edx=16: compares the first 13 bytes */
        uint32_t idx; memcpy(&idx, out, 4);
        CHECKF(idx == 1, "pcmpestri m128 at +7: index %u, want 1", idx);
    }
    if (cpu_has(F_AVX2)) {
        int32_t lanes[8], add[8], got[8];
        T_ALIGN(64) uint8_t ii[64];
        for (int i = 0; i < 8; i++) { lanes[i] = i * 1000 - 3; add[i] = 7 - i * 13; }
        memset(ii, 0, sizeof ii);
        memcpy(ii, lanes, 32);
        memcpy(buf + 9, add, 32);
        op_avx2_vpaddd_y(buf + 9, out, ii);
        memcpy(got, out, 32);
        for (int i = 0; i < 8; i++) CHECKF(got[i] == lanes[i] + add[i], "vpaddd ymm m256 at +9 lane %d: %d", i, got[i]);
    }
}

static T_TARGET("avx2") T_NOINLINE void al_avx2_gather(const void *base, const int32_t *idx, void *out) {
    __asm__ volatile("vmovdqu (%1), %%ymm1\n\tvpcmpeqd %%ymm2, %%ymm2, %%ymm2\n\tvpxor %%ymm0, %%ymm0, %%ymm0\n\t"
                     "vpgatherdd %%ymm2, (%0,%%ymm1,1), %%ymm0\n\tvmovdqu %%ymm0, (%2)\n\tvzeroupper"
                     :: "r"(base), "r"(idx), "r"(out) : "xmm0", "xmm1", "xmm2", "memory");
}
TEST(unaligned_avx2_gather_elements) {
    if (!cpu_has(F_AVX2)) SKIP("AVX2 not usable");
    al_src_init();
    for (int base_off = 0; base_off < 8; base_off++) {
        for (int stride = 1; stride <= 9; stride += 2) {
            int32_t idx[8]; uint32_t out[8], want[8];
            for (int i = 0; i < 8; i++) { idx[i] = i * stride * 5 + (i & 1); memcpy(&want[i], g_src + 64 + base_off + idx[i], 4); }
            al_avx2_gather(g_src + 64 + base_off, idx, out);
            char label[96]; snprintf(label, sizeof label, "vpgatherdd base+%d stride %d", base_off, stride);
            t_check_mem(out, want, sizeof out, __FILE__, __LINE__, label);
        }
    }
}

/* ======================================================================================
 * Cases: alignment-required forms fault on misaligned operands (child processes)
 * ====================================================================================== */
TEST(align_sse_movaps_load_needs_16) { al_expect_align("al_movaps_ld", 16, 48); }
TEST(align_sse_movaps_store_needs_16) { al_expect_align("al_movaps_st", 16, 48); }
TEST(align_sse2_movapd_load_needs_16) { al_expect_align("al_movapd_ld", 16, 32); }
TEST(align_sse2_movapd_store_needs_16) { al_expect_align("al_movapd_st", 16, 32); }
TEST(align_sse2_movdqa_load_needs_16) { al_expect_align("al_movdqa_ld", 16, 48); }
TEST(align_sse2_movdqa_store_needs_16) { al_expect_align("al_movdqa_st", 16, 48); }
TEST(align_sse_movntps_store_needs_16) { al_expect_align("al_movntps_st", 16, 32); }
TEST(align_sse2_movntpd_store_needs_16) { al_expect_align("al_movntpd_st", 16, 32); }
TEST(align_sse2_movntdq_store_needs_16) { al_expect_align("al_movntdq_st", 16, 32); }
TEST(align_sse41_movntdqa_load_needs_16) { al_expect_align("al_movntdqa_ld", 16, 32); }
TEST(align_sse2_paddd_m128_needs_16) { al_expect_align("al_paddd_ld", 16, 32); }
TEST(align_sse_addps_m128_needs_16) { al_expect_align("al_addps_ld", 16, 32); }
TEST(align_ssse3_pshufb_m128_needs_16) { al_expect_align("al_pshufb_ld", 16, 32); }
TEST(align_sse2_pcmpeqb_m128_needs_16) { al_expect_align("al_pcmpeqb_ld", 16, 32); }
TEST(noalign_sse_movups_movdqu_lddqu_never_fault) {
    al_expect_align("al_movups_ld", 0, 64);
    al_expect_align("al_movdqu_ld", 0, 64);
    al_expect_align("al_movups_st", 0, 64);
    al_expect_align("al_movdqu_st", 0, 64);
    if (cpu_has(F_SSE3)) al_expect_align("al_lddqu_ld", 0, 64);
}
TEST(noalign_sse42_pcmpistri_m128_never_faults) { al_expect_align("al_pcmpistri_ld", 0, 32); }
TEST(noalign_sse2_movq_movd_gpr_never_fault) {
    al_expect_align("al_movq_ld", 0, 32);
    al_expect_align("al_movd_ld", 0, 32);
    al_expect_align("al_movq_st", 0, 32);
    al_expect_align("al_gpr_ld", 0, 32);
    al_expect_align("al_gpr_st", 0, 32);
}
TEST(align_avx_vmovaps_xmm_load_needs_16) { al_expect_align("al_avx_vmovaps_x_ld", 16, 48); }
TEST(align_avx_vmovdqa_xmm_store_needs_16) { al_expect_align("al_avx_vmovdqa_x_st", 16, 48); }
TEST(align_avx_vmovdqa_ymm_load_needs_32) { al_expect_align("al_avx_vmovdqa_y_ld", 32, 64); }
TEST(align_avx_vmovaps_ymm_store_needs_32) { al_expect_align("al_avx_vmovaps_y_st", 32, 64); }
TEST(align_avx_vmovntdq_ymm_store_needs_32) { al_expect_align("al_avx_vmovntdq_y_st", 32, 64); }
TEST(align_avx_vmovntps_ymm_store_needs_32) { al_expect_align("al_avx_vmovntps_y_st", 32, 64); }
TEST(align_avx2_vmovntdqa_ymm_load_needs_32) { al_expect_align("al_avx2_vmovntdqa_y_ld", 32, 64); }
TEST(noalign_avx_vmovdqu_ymm_never_faults) {
    al_expect_align("al_avx_vmovdqu_y_ld", 0, 64);
    al_expect_align("al_avx_vmovdqu_y_st", 0, 64);
}
TEST(noalign_vex_alu_m128_m256_never_fault) {
    /* the same paddd that faults in legacy encoding must not fault under VEX */
    al_expect_align("al_avx_vpaddd_x_ld", 0, 32);
    al_expect_align("al_avx_vpxor_x_ld", 0, 32);
    al_expect_align("al_avx_vaddps_y_ld", 0, 64);
    if (cpu_has(F_AVX2)) al_expect_align("al_avx2_vpaddd_y_ld", 0, 64);
}
TEST(align_avx512_vmovdqa64_zmm_load_needs_64) { al_expect_align("al_avx512_vmovdqa64_z_ld", 64, 128); }
TEST(align_avx512_vmovdqa64_zmm_store_needs_64) { al_expect_align("al_avx512_vmovdqa64_z_st", 64, 128); }
TEST(align_avx512_vmovntdq_zmm_store_needs_64) { al_expect_align("al_avx512_vmovntdq_z_st", 64, 128); }
TEST(noalign_avx512_vmovdqu64_and_alu_never_fault) {
    al_expect_align("al_avx512_vmovdqu64_z_ld", 0, 64);
    al_expect_align("al_avx512_vmovdqu64_z_st", 0, 64);
    al_expect_align("al_avx512_vpaddd_z_ld", 0, 64);
}
TEST(align_fxsave_needs_16) { al_expect_align("al_fxsave", 16, 48); }
TEST(align_fxrstor_needs_16) { al_expect_align("al_fxrstor", 16, 48); }
TEST(align_xsave_needs_64) { al_expect_align("al_xsave", 64, 128); }
TEST(align_xrstor_needs_64) { al_expect_align("al_xrstor", 64, 128); }
TEST(align_xsaveopt_needs_64) { al_expect_align("al_xsaveopt", 64, 128); }
TEST(align_xsavec_needs_64) { al_expect_align("al_xsavec", 64, 128); }
TEST(align_cmpxchg16b_needs_16) {
#ifdef T_X64
    al_expect_align("al_cmpxchg16b", 16, 48);
#else
    SKIP("CMPXCHG16B exists only in 64-bit mode");
#endif
}

TEST(align_fault_unhandled_reports_instruction) {
    t_child_run("al_unhandled", "movaps", 15000, &g_cr);
    REQUIREF(g_cr.started && !g_cr.timed_out, "child did not run to completion");
    char v[64]; unsigned long long fa = 0;
    if (t_child_find(&g_cr, "FAULT=", v, sizeof v)) fa = strtoull(v, NULL, 16);
    CHECKF(!strstr(g_cr.out, "SURVIVED"), "misaligned MOVAPS completed without a fault");
    CHECKF(g_cr.exc_code != 0, "child did not die of an exception (exit 0x%lx)", g_cr.exit_code);
    CHECK_EQ(g_cr.exc_addr, fa);
    t_info("exception_code", "0x%08lx", g_cr.exc_code);
    t_info("exit_code", "0x%08lx", g_cr.exit_code);
    if (g_cr.exc_code != 0xC0000005ul) t_note("unhandled misaligned MOVAPS reported as 0x%08lx", g_cr.exc_code);
}
TEST(page_fault_unhandled_reports_access_violation) {
    t_child_run("al_unhandled", "noaccess", 15000, &g_cr);
    REQUIREF(g_cr.started && !g_cr.timed_out, "child did not run to completion");
    char v[64]; unsigned long long fa = 0;
    if (t_child_find(&g_cr, "FAULT=", v, sizeof v)) fa = strtoull(v, NULL, 16);
    CHECKF(!strstr(g_cr.out, "SURVIVED"), "load straddling into PAGE_NOACCESS completed");
    CHECK_EQ(g_cr.exc_code, 0xC0000005u);
    CHECK_EQ(g_cr.exc_addr, fa);
}

/* ======================================================================================
 * Cases: page-boundary straddling (committed pages: data must be exact)
 * ====================================================================================== */
TEST(straddle_committed_gpr_loads_stores) {
    uint8_t *b = al_two_pages(); REQUIRE(b);
    al_ld_straddle(g_ld_gpr, T_COUNTOF(g_ld_gpr), b + PG, "straddle");
    al_st_straddle(g_st_gpr, T_COUNTOF(g_st_gpr), b + PG, "straddle");
    VirtualFree(b, 0, MEM_RELEASE);
}
TEST(straddle_committed_sse_loads_stores) {
    uint8_t *b = al_two_pages(); REQUIRE(b);
    al_ld_straddle(g_ld_sse_full, T_COUNTOF(g_ld_sse_full), b + PG, "straddle");
    al_ld_straddle(g_ld_sse_part, T_COUNTOF(g_ld_sse_part), b + PG, "straddle");
    al_st_straddle(g_st_sse, T_COUNTOF(g_st_sse), b + PG, "straddle");
    al_st_straddle(g_st_sse_part, T_COUNTOF(g_st_sse_part), b + PG, "straddle");
    VirtualFree(b, 0, MEM_RELEASE);
}
TEST(straddle_committed_avx_loads_stores) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    uint8_t *b = al_two_pages(); REQUIRE(b);
    al_ld_straddle(g_ld_avx, T_COUNTOF(g_ld_avx), b + PG, "straddle");
    al_st_straddle(g_st_avx, T_COUNTOF(g_st_avx), b + PG, "straddle");
    VirtualFree(b, 0, MEM_RELEASE);
}
TEST(straddle_committed_avx512_loads_stores) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    uint8_t *b = al_two_pages(); REQUIRE(b);
    al_ld_straddle(g_ld_avx512, T_COUNTOF(g_ld_avx512), b + PG, "straddle");
    al_st_straddle(g_st_avx512, T_COUNTOF(g_st_avx512), b + PG, "straddle");
    VirtualFree(b, 0, MEM_RELEASE);
}
TEST(straddle_separately_committed_pages) {
    /* one reservation, pages committed by two calls, the second page touched first */
    uint8_t *b = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE, PAGE_NOACCESS);
    REQUIRE(b);
    REQUIRE(VirtualAlloc(b + PG, PG, MEM_COMMIT, PAGE_READWRITE) == b + PG);
    memset(b + PG, 0x77, PG);
    REQUIRE(VirtualAlloc(b, PG, MEM_COMMIT, PAGE_READWRITE) == b);
    for (int i = 0; i < 2 * PG; i++) b[i] = al_pat((size_t)i);
    al_ld_straddle(g_ld_sse_full, T_COUNTOF(g_ld_sse_full), b + PG, "sep-commit");
    if (cpu_has(F_AVX)) al_ld_straddle(g_ld_avx, T_COUNTOF(g_ld_avx), b + PG, "sep-commit");
    al_st_straddle(g_st_sse, T_COUNTOF(g_st_sse), b + PG, "sep-commit");
    VirtualFree(b, 0, MEM_RELEASE);
}
TEST(straddle_adjacent_reservations_64k_boundary) {
    /* two independent reservations placed back to back: accesses span two VM regions */
    SYSTEM_INFO si; GetSystemInfo(&si);
    size_t g = si.dwAllocationGranularity;
    uint8_t *x = NULL, *a = NULL, *b2 = NULL;
    for (int attempt = 0; attempt < 8 && !(a && b2); attempt++) {
        x = (uint8_t *)VirtualAlloc(NULL, 2 * g, MEM_RESERVE, PAGE_NOACCESS);
        if (!x) break;
        VirtualFree(x, 0, MEM_RELEASE);
        a = (uint8_t *)VirtualAlloc(x, g, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        b2 = a ? (uint8_t *)VirtualAlloc(x + g, g, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) : NULL;
        if (!(a && b2)) { if (a) VirtualFree(a, 0, MEM_RELEASE); if (b2) VirtualFree(b2, 0, MEM_RELEASE); a = b2 = NULL; }
    }
    if (!(a && b2)) SKIP("could not place two adjacent reservations (address space race)");
    CHECK(b2 == a + g);
    MEMORY_BASIC_INFORMATION mi;
    CHECK(VirtualQuery(b2, &mi, sizeof mi) && mi.AllocationBase == b2);
    for (size_t i = 0; i < PG; i++) { a[g - PG + i] = al_pat(i); b2[i] = al_pat(PG + i); }
    al_ld_straddle(g_ld_sse_full, T_COUNTOF(g_ld_sse_full), b2, "adjacent-regions");
    al_ld_straddle(g_ld_gpr, T_COUNTOF(g_ld_gpr), b2, "adjacent-regions");
    if (cpu_has(F_AVX)) al_ld_straddle(g_ld_avx, T_COUNTOF(g_ld_avx), b2, "adjacent-regions");
    if (cpu_has(F_AVX512F)) al_ld_straddle(g_ld_avx512, T_COUNTOF(g_ld_avx512), b2, "adjacent-regions");
    al_st_straddle(g_st_sse, T_COUNTOF(g_st_sse), b2, "adjacent-regions");
    VirtualFree(a, 0, MEM_RELEASE);
    VirtualFree(b2, 0, MEM_RELEASE);
}

TEST(cacheline_split_loads_stores_all_widths) {
    /* every offset across three 64-byte lines: split and non-split accesses give the same bytes */
    static T_ALIGN(64) uint8_t buf[320];
    for (int i = 0; i < (int)sizeof buf; i++) buf[i] = al_pat((size_t)i * 7);
    const al_io *groups[] = { g_ld_gpr, g_ld_sse_full, g_ld_avx, g_ld_avx512 };
    size_t counts[] = { T_COUNTOF(g_ld_gpr), T_COUNTOF(g_ld_sse_full), T_COUNTOF(g_ld_avx), T_COUNTOF(g_ld_avx512) };
    for (int gi = 0; gi < 4; gi++)
        for (size_t i = 0; i < counts[gi]; i++) {
            const al_io *e = &groups[gi][i];
            if (!cpu_has((t_feature)e->feat)) continue;
            for (int off = 64 - e->w; off <= 128; off += (off < 64 ? 1 : 3)) al_ld_check(e, buf + off, "line-split", off);
        }
    static T_ALIGN(64) uint8_t win[320];
    for (size_t i = 0; i < T_COUNTOF(g_st_sse); i++)
        for (int off = 48; off <= 64; off++) al_st_check(&g_st_sse[i], win, sizeof win, win + off + 64, "line-split", off);
}

/* ---- page end / page start: accesses fully inside the page next to an inaccessible page ---- */
TEST(page_end_exact_accesses_do_not_fault) {
    void *base;
    uint8_t *end = (uint8_t *)t_guarded_alloc(PG, &base);
    REQUIRE(end);
    end += PG;   /* first inaccessible byte */
    for (int i = 0; i < PG; i++) end[i - PG] = al_pat((size_t)i);
    const al_io *lg[] = { g_ld_gpr, g_ld_sse_full, g_ld_sse_part, g_ld_avx, g_ld_avx512 };
    size_t lc[] = { T_COUNTOF(g_ld_gpr), T_COUNTOF(g_ld_sse_full), T_COUNTOF(g_ld_sse_part), T_COUNTOF(g_ld_avx), T_COUNTOF(g_ld_avx512) };
    for (int gi = 0; gi < 5; gi++)
        for (size_t i = 0; i < lc[gi]; i++) {
            const al_io *e = &lg[gi][i];
            if (!cpu_has((t_feature)e->feat)) continue;
            for (int back = e->w; back <= e->w + 3; back++) al_ld_check(e, end - back, "page-end", -back);
        }
    const al_io *sg[] = { g_st_gpr, g_st_sse, g_st_sse_part, g_st_avx, g_st_avx512 };
    size_t sc[] = { T_COUNTOF(g_st_gpr), T_COUNTOF(g_st_sse), T_COUNTOF(g_st_sse_part), T_COUNTOF(g_st_avx), T_COUNTOF(g_st_avx512) };
    for (int gi = 0; gi < 5; gi++)
        for (size_t i = 0; i < sc[gi]; i++) {
            const al_io *e = &sg[gi][i];
            if (!cpu_has((t_feature)e->feat)) continue;
            for (int back = e->w; back <= e->w + 2; back++) al_st_check(e, end - 128, 128, end - back, "page-end", -back);
        }
    t_guarded_free(base);
}
TEST(page_start_after_noaccess_page_accesses_do_not_fault) {
    uint8_t *b = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    REQUIRE(b);
    DWORD old;
    REQUIRE(VirtualProtect(b, PG, PAGE_NOACCESS, &old));
    uint8_t *st = b + PG;
    for (int i = 0; i < PG; i++) st[i] = al_pat((size_t)i + 5);
    const al_io *lg[] = { g_ld_gpr, g_ld_sse_full, g_ld_sse_part, g_ld_avx, g_ld_avx512 };
    size_t lc[] = { T_COUNTOF(g_ld_gpr), T_COUNTOF(g_ld_sse_full), T_COUNTOF(g_ld_sse_part), T_COUNTOF(g_ld_avx), T_COUNTOF(g_ld_avx512) };
    for (int gi = 0; gi < 5; gi++)
        for (size_t i = 0; i < lc[gi]; i++) {
            const al_io *e = &lg[gi][i];
            if (!cpu_has((t_feature)e->feat)) continue;
            for (int fwd = 0; fwd < 3; fwd++) al_ld_check(e, st + fwd, "page-start", fwd);
        }
    for (size_t i = 0; i < T_COUNTOF(g_st_sse); i++) al_st_check(&g_st_sse[i], st, 128, st, "page-start", 0);
    VirtualFree(b, 0, MEM_RELEASE);
}

/* ======================================================================================
 * Cases: straddling into an inaccessible page (child processes)
 * ====================================================================================== */
TEST(boundary_noaccess_sse_load_faults_precisely) { al_expect_boundary("al_movdqu_ld", "noaccess"); }
TEST(boundary_noaccess_sse_store_faults) { al_expect_boundary("al_movdqu_st", "noaccess"); }
TEST(boundary_noaccess_lddqu_load_faults) { al_expect_boundary("al_lddqu_ld", "noaccess"); }
TEST(boundary_noaccess_movq_movd_tiny_loads) {
    al_expect_boundary("al_movq_ld", "noaccess");
    al_expect_boundary("al_movd_ld", "noaccess");
    al_expect_boundary("al_movq_st", "noaccess");
}
TEST(boundary_noaccess_gpr_load_store) {
    al_expect_boundary("al_gpr_ld", "noaccess");
    al_expect_boundary("al_gpr_st", "noaccess");
}
TEST(boundary_noaccess_pcmpistri_load) { al_expect_boundary("al_pcmpistri_ld", "noaccess"); }
TEST(boundary_noaccess_avx_ymm_load_faults) { al_expect_boundary("al_avx_vmovdqu_y_ld", "noaccess"); }
TEST(boundary_noaccess_avx_ymm_store_faults) { al_expect_boundary("al_avx_vmovdqu_y_st", "noaccess"); }
TEST(boundary_noaccess_avx_alu_ymm_load_faults) { al_expect_boundary("al_avx_vaddps_y_ld", "noaccess"); }
TEST(boundary_noaccess_avx512_zmm_load_faults) { al_expect_boundary("al_avx512_vmovdqu64_z_ld", "noaccess"); }
TEST(boundary_noaccess_avx512_zmm_store_faults) { al_expect_boundary("al_avx512_vmovdqu64_z_st", "noaccess"); }
TEST(boundary_readonly_loads_ok_stores_fault) {
    al_expect_boundary("al_movdqu_ld", "readonly");
    al_expect_boundary("al_movdqu_st", "readonly");
    al_expect_boundary("al_gpr_st", "readonly");
    if (cpu_has(F_AVX)) { al_expect_boundary("al_avx_vmovdqu_y_ld", "readonly"); al_expect_boundary("al_avx_vmovdqu_y_st", "readonly"); }
}
TEST(boundary_reserved_uncommitted_page_faults) {
    al_expect_boundary("al_movdqu_ld", "reserve");
    al_expect_boundary("al_movdqu_st", "reserve");
    if (cpu_has(F_AVX)) al_expect_boundary("al_avx_vmovdqu_y_ld", "reserve");
}
TEST(boundary_guard_page_one_shot) {
    al_expect_boundary("al_movdqu_ld", "guard");
    al_expect_boundary("al_movdqu_st", "guard");
    al_expect_boundary("al_gpr_ld", "guard");
    if (cpu_has(F_AVX)) al_expect_boundary("al_avx_vmovdqu_y_ld", "guard");
}

/* ---- masked accesses: masked-out elements in an inaccessible page must not fault ---- */
static void al_expect_maskpage(const char *stub, int lanes, int hard) {
    const al_stub *s = al_find_stub(stub);
    REQUIRE(s);
    char args[96]; snprintf(args, sizeof args, "%s %d", stub, lanes);
    t_child_run("al_maskpage", args, 20000, &g_cr);
    al_child_ok(args);
    int rows = 0, suppressed_bad = 0;
    for (const char *l = g_cr.out; l && *l; l = strchr(l, '\n'), l = l ? l + 1 : NULL) {
        int k, full, f, data; unsigned long code; unsigned long long i0; long long i1rel;
        if (l[0] != 'M' || sscanf(l + 2, "%d %d %d %lx %llx %lld %d", &k, &full, &f, &code, &i0, &i1rel, &data) != 7) continue;
        rows++;
        if (full) {
            int want = k < lanes;
            CHECKF(f == want, "%s k=%d all lanes active: %s", stub, k, want ? "no fault although active lanes are inaccessible" : "unexpected fault");
            if (f) {
                CHECKF(code == 0xC0000005ul, "%s k=%d: code 0x%08lx", stub, k, code);
                CHECKF(i0 == (unsigned long long)(s->kind == SK_STORE), "%s k=%d: info0 %llu", stub, k, i0);
                CHECKF(i1rel >= 0 && i1rel < 4 * (lanes - k), "%s k=%d: fault address boundary%+lld outside active inaccessible lanes", stub, k, i1rel);
            }
        } else {
            if (hard) CHECKF(!f, "%s k=%d: masked-out lanes in an inaccessible page caused fault 0x%08lx", stub, k, code);
            else if (f) suppressed_bad++;
        }
        if (data >= 0) CHECKF(data == 1, "%s k=%d full=%d: wrong data in active lanes / masked lanes not zeroed / bytes before modified", stub, k, full);
    }
    CHECK_EQ(rows, 2 * (lanes + 1));
    if (!hard) {
        t_info("masked_lane_faults", "%d of %d", suppressed_bad, lanes + 1);
        if (suppressed_bad) t_note("%s: masked-out lanes faulted (allowed on non-Intel: AMD APM says implementation dependent)", stub);
    }
}
TEST(masked_avx_vmaskmovps_load_page_end) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    al_expect_maskpage("al_avx_vmaskmovps_ld", 8, cpu()->is_intel);
}
TEST(masked_avx_vmaskmovps_store_page_end) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    al_expect_maskpage("al_avx_vmaskmovps_st", 8, cpu()->is_intel);
}
TEST(masked_avx512_vmovdqu32_load_page_end) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    al_expect_maskpage("al_avx512_vmovdqu32_kz_ld", 16, 1);
}
TEST(masked_avx512_vmovdqu32_store_page_end) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable");
    al_expect_maskpage("al_avx512_vmovdqu32_k_st", 16, 1);
}

/* ======================================================================================
 * Cases: rep movs / stos across page boundaries
 * ====================================================================================== */
static T_NOINLINE void al_rep_movsb(void *d, const void *s, size_t n, uintptr_t r[3]) {
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(n) :: "memory");
    r[0] = (uintptr_t)d; r[1] = (uintptr_t)s; r[2] = n;
}
static T_NOINLINE void al_rep_movsb_back(void *d, const void *s, size_t n, uintptr_t r[3]) {
    __asm__ volatile("std\n\trep movsb\n\tcld" : "+D"(d), "+S"(s), "+c"(n) :: "memory", "cc");
    r[0] = (uintptr_t)d; r[1] = (uintptr_t)s; r[2] = n;
}
static T_NOINLINE void al_rep_movsl(void *d, const void *s, size_t n, uintptr_t r[3]) {
    __asm__ volatile("rep movsl" : "+D"(d), "+S"(s), "+c"(n) :: "memory");
    r[0] = (uintptr_t)d; r[1] = (uintptr_t)s; r[2] = n;
}
static T_NOINLINE void al_rep_stosb(void *d, int v, size_t n, uintptr_t r[3]) {
    __asm__ volatile("rep stosb" : "+D"(d), "+c"(n) : "a"(v) : "memory");
    r[0] = (uintptr_t)d; r[1] = 0; r[2] = n;
}

TEST(rep_movsb_across_page_boundaries) {
    uint8_t *sb = al_two_pages(), *db = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    REQUIRE(sb && db);
    static uint8_t ref[2 * PG];
    static const int lens[] = { 0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 511, 1000, 2049, 4097 };
    static const int soffs[] = { -1, -3, -8, -31, -64, -65, -200, -2049 };
    static const int doffs[] = { -1, -5, -16, -33, -100, -4000 };
    for (size_t li = 0; li < T_COUNTOF(lens); li++)
        for (size_t si = 0; si < T_COUNTOF(soffs); si++)
            for (size_t di = 0; di < T_COUNTOF(doffs); di++) {
                int n = lens[li];
                uint8_t *s = sb + PG + soffs[si], *d = db + PG + doffs[di];
                if (s + n > sb + 2 * PG || d + n > db + 2 * PG) continue;
                memset(db, 0xCC, 2 * PG);
                memcpy(ref, db, 2 * PG);
                memcpy(ref + (d - db), s, (size_t)n);
                uintptr_t r[3];
                al_rep_movsb(d, s, (size_t)n, r);
                char label[96]; snprintf(label, sizeof label, "rep movsb n=%d src%+d dst%+d", n, soffs[si], doffs[di]);
                t_check_mem(db, ref, 2 * PG, __FILE__, __LINE__, label);
                CHECKF(r[0] == (uintptr_t)(d + n) && r[1] == (uintptr_t)(s + n) && r[2] == 0, "%s: final RDI/RSI/RCX wrong", label);
            }
    VirtualFree(sb, 0, MEM_RELEASE); VirtualFree(db, 0, MEM_RELEASE);
}
TEST(rep_movsb_backward_df_across_boundary) {
    uint8_t *sb = al_two_pages(), *db = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    REQUIRE(sb && db);
    static uint8_t ref[2 * PG];
    static const int lens[] = { 1, 2, 16, 33, 64, 100, 513, 3000 };
    for (size_t li = 0; li < T_COUNTOF(lens); li++)
        for (int k = 1; k <= 65; k += 16) {
            int n = lens[li];
            uint8_t *s = sb + PG - k, *d = db + PG - (k * 3 % 61) - 1;
            if (s + n > sb + 2 * PG || d + n > db + 2 * PG) continue;
            memset(db, 0xCC, 2 * PG); memcpy(ref, db, 2 * PG);
            memcpy(ref + (d - db), s, (size_t)n);
            uintptr_t r[3];
            al_rep_movsb_back(d + n - 1, s + n - 1, (size_t)n, r);
            char label[96]; snprintf(label, sizeof label, "std; rep movsb n=%d k=%d", n, k);
            t_check_mem(db, ref, 2 * PG, __FILE__, __LINE__, label);
            CHECKF(r[0] == (uintptr_t)(d - 1) && r[2] == 0, "%s: final RDI/RCX wrong", label);
        }
    VirtualFree(sb, 0, MEM_RELEASE); VirtualFree(db, 0, MEM_RELEASE);
}
TEST(rep_movsd_stosb_unaligned_across_boundary) {
    uint8_t *sb = al_two_pages(), *db = (uint8_t *)VirtualAlloc(NULL, 2 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    REQUIRE(sb && db);
    static uint8_t ref[2 * PG];
    for (int so = 1; so < 8; so += 2)
        for (int dof = 1; dof < 8; dof += 3)
            for (int cnt = 1; cnt <= 300; cnt += 37) {
                uint8_t *s = sb + PG - 4 * (cnt / 2) - so, *d = db + PG - 4 * (cnt / 3) - dof;
                memset(db, 0xCC, 2 * PG); memcpy(ref, db, 2 * PG);
                memcpy(ref + (d - db), s, (size_t)cnt * 4);
                uintptr_t r[3];
                al_rep_movsl(d, s, (size_t)cnt, r);
                char label[96]; snprintf(label, sizeof label, "rep movsd cnt=%d src-%d dst-%d", cnt, so, dof);
                t_check_mem(db, ref, 2 * PG, __FILE__, __LINE__, label);
                CHECK(r[0] == (uintptr_t)(d + 4 * cnt) && r[2] == 0);
            }
    for (int n = 1; n <= 5000; n = n * 3 + 1) {
        uint8_t *d = db + PG - (n / 2) - 3;
        if (d + n > db + 2 * PG) continue;
        memset(db, 0xCC, 2 * PG); memcpy(ref, db, 2 * PG);
        memset(ref + (d - db), 0x5A, (size_t)n);
        uintptr_t r[3];
        al_rep_stosb(d, 0x5A, (size_t)n, r);
        char label[64]; snprintf(label, sizeof label, "rep stosb n=%d", n);
        t_check_mem(db, ref, 2 * PG, __FILE__, __LINE__, label);
        CHECK(r[0] == (uintptr_t)(d + n) && r[2] == 0);
    }
    VirtualFree(sb, 0, MEM_RELEASE); VirtualFree(db, 0, MEM_RELEASE);
}
TEST(rep_movsb_source_ending_at_noaccess_does_not_fault) {
    /* reading exactly up to the last accessible byte must not touch the next page */
    for (int n = 1; n <= 4096; n = n < 64 ? n + 7 : n * 2) {
        void *base;
        uint8_t *s = (uint8_t *)t_guarded_alloc((size_t)n, &base);
        REQUIRE(s);
        for (int i = 0; i < n; i++) s[i] = al_pat((size_t)i);
        uint8_t *d = (uint8_t *)malloc((size_t)n + 64);
        REQUIRE(d);
        uintptr_t r[3];
        al_rep_movsb(d, s, (size_t)n, r);
        char label[64]; snprintf(label, sizeof label, "rep movsb from page end n=%d", n);
        t_check_mem(d, s, (size_t)n, __FILE__, __LINE__, label);
        /* and a destination ending exactly at the page end */
        al_rep_movsb(s, d, (size_t)n, r);
        CHECK(r[2] == 0);
        free(d);
        t_guarded_free(base);
    }
}
static void al_expect_repfault(const char *which) {
    t_child_run("al_repfault", which, 20000, &g_cr);
    al_child_ok(which);
    int rows = 0, dst = !strcmp(which, "dst");
    for (const char *l = g_cr.out; l && *l; l = strchr(l, '\n'), l = l ? l + 1 : NULL) {
        int k, f, pre, fin, ipok; unsigned n; unsigned long code; unsigned long long i0, cx; long long i1rel, di, si;
        if (l[0] != 'Q' || sscanf(l + 2, "%d %u %d %lx %llx %lld %lld %lld %llu %d %d %d", &k, &n, &f, &code, &i0, &i1rel, &di, &si, &cx, &pre, &fin, &ipok) != 12) continue;
        rows++;
        CHECKF(f == 1, "rep movsb %s k=%d: expected exactly one fault, got %d", which, k, f);
        if (f != 1) continue;
        CHECKF(code == 0xC0000005ul, "rep movsb %s k=%d: code 0x%08lx", which, k, code);
        CHECKF(i0 == (unsigned long long)dst, "rep movsb %s k=%d: info0=%llu", which, k, i0);
        CHECKF(i1rel >= 0 && i1rel < (long long)n - k, "rep movsb %s k=%d: fault address boundary%+lld outside the access", which, k, i1rel);
        CHECKF(ipok == 1, "rep movsb %s k=%d: CONTEXT IP != rep instruction", which, k);
        CHECKF(di == si, "rep movsb %s k=%d: RDI advanced %lld but RSI %lld", which, k, di, si);
        CHECKF(di >= 0 && di <= k, "rep movsb %s k=%d: RDI progress %lld beyond the first inaccessible byte", which, k, di);
        CHECKF((unsigned long long)di + cx == n, "rep movsb %s k=%d: progress %lld + RCX %llu != %u", which, k, di, cx, n);
        CHECKF(pre == 1, "rep movsb %s k=%d: bytes before RDI not yet copied at the fault (pre=%d)", which, k, pre);
        CHECKF(fin == 1, "rep movsb %s k=%d: copy incomplete after the handler fixed the page and execution restarted", which, k);
        t_log("rep %s k=%d progress=%lld rcx=%llu addr=boundary%+lld", which, k, di, cx, i1rel);
    }
    CHECK_EQ(rows, 9);
}
TEST(rep_movsb_fault_on_destination_restart) { al_expect_repfault("dst"); }
TEST(rep_movsb_fault_on_source_restart) { al_expect_repfault("src"); }

/* ======================================================================================
 * Cases: atomics on misaligned operands
 * ====================================================================================== */
static volatile uint32_t *g_atom32;
static volatile uint64_t *g_atom64;
static DWORD WINAPI al_atomic_worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 20000; i++) {
        uint32_t one = 1;
        __asm__ volatile("lock xaddl %0, %1" : "+r"(one), "+m"(*g_atom32) :: "memory");
        uint64_t old = *g_atom64, nw;
        do { nw = old + 1; } while (!__atomic_compare_exchange_n(g_atom64, &old, nw, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }
    return 0;
}
TEST(atomic_lock_on_misaligned_within_cacheline) {
    /* LOCK guarantees atomicity regardless of alignment; keep the operands inside one cache line */
    static T_ALIGN(64) uint8_t line[128];
    memset(line, 0, sizeof line);
    g_atom32 = (volatile uint32_t *)(line + 1);
    g_atom64 = (volatile uint64_t *)(line + 13);
    HANDLE th[4]; int nt = 0;
    for (int i = 0; i < 4; i++) { th[nt] = CreateThread(NULL, 0, al_atomic_worker, NULL, 0, NULL); if (th[nt]) nt++; }
    if (!nt) SKIP("CreateThread failed");
    WaitForMultipleObjects((DWORD)nt, th, TRUE, 20000);
    for (int i = 0; i < nt; i++) CloseHandle(th[i]);
    CHECK_EQ(*g_atom32, 20000u * (unsigned)nt);
    CHECK_EQ(*g_atom64, 20000ull * (unsigned)nt);
}
TEST(atomic_split_lock_across_cacheline_info) {
    t_child_run("al_splitlock", NULL, 20000, &g_cr);
    REQUIRE(g_cr.started);
    char v[64];
    if (g_cr.timed_out) { t_note("split-lock child timed out (split-lock throttling by the host OS?)"); return; }
    if (g_cr.exc_code) { t_note("split-locked LOCK XADD raised 0x%08lx (split-lock detection enabled by the OS)", g_cr.exc_code); t_info("split_lock", "exception 0x%08lx", g_cr.exc_code); return; }
    REQUIRE(t_child_find(&g_cr, "SL val=", v, sizeof v));
    CHECK_EQ(strtoul(v, NULL, 10), 16);
    t_info("split_lock", "ok %.0f ms", g_cr.ms);
}
TEST(eflags_ac_alignment_check_info) {
    t_child_run("al_acflag", NULL, 15000, &g_cr);
    REQUIRE(g_cr.started);
    char v[64];
    if (g_cr.timed_out || g_cr.exc_code || !t_child_find(&g_cr, "AC ", v, sizeof v)) {
        t_note("EFLAGS.AC probe child: timeout=%d exc=0x%08lx", g_cr.timed_out, g_cr.exc_code);
        return;
    }
    t_info("ac_probe", "%s", v);   /* f=1 code=0x80000002 when the OS enables CR0.AM */
}

/* ======================================================================================
 * Cases: stack alignment
 * ====================================================================================== */
static volatile uintptr_t g_cb_sp, g_cb_loc16, g_cb_loc32, g_cb_loc64;
static volatile LONG g_cb_hits;
static HANDLE g_cb_event;
static T_NOINLINE void al_cb_body(void) {
    T_ALIGN(16) volatile uint8_t l16[16];
    T_ALIGN(32) volatile uint8_t l32[32];
    T_ALIGN(64) volatile uint8_t l64[64];
    l16[0] = 1; l32[0] = 2; l64[0] = 3;
    g_cb_sp = al_sp_slot;
    g_cb_loc16 = (uintptr_t)l16; g_cb_loc32 = (uintptr_t)l32; g_cb_loc64 = (uintptr_t)l64;
    InterlockedIncrement(&g_cb_hits);
}
static T_NOINLINE DWORD WINAPI al_cb_thread(void *a) { (void)a; al_cb_body(); return 0; }
static T_NOINLINE void CALLBACK al_cb_apc(ULONG_PTR a) { (void)a; al_cb_body(); }
static T_NOINLINE LONG CALLBACK al_cb_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != 0xE0A11600u) return EXCEPTION_CONTINUE_SEARCH;
    al_cb_body();
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void *g_main_fiber;
static T_NOINLINE void CALLBACK al_cb_fiber(void *a) { (void)a; al_cb_body(); SwitchToFiber(g_main_fiber); }
static T_NOINLINE void CALLBACK al_cb_tp(PTP_CALLBACK_INSTANCE i, void *a) { (void)i; (void)a; al_cb_body(); SetEvent(g_cb_event); }
static T_NOINLINE void CALLBACK al_cb_timer(void *a, BOOLEAN t) { (void)a; (void)t; al_cb_body(); SetEvent(g_cb_event); }
static T_NOINLINE void CALLBACK al_cb_wait(void *a, BOOLEAN t) { (void)a; (void)t; al_cb_body(); SetEvent(g_cb_event); }
static T_NOINLINE BOOL CALLBACK al_cb_codepage(LPSTR s) { (void)s; al_cb_body(); return FALSE; }
static T_NOINLINE void WINAPI al_cb_fls(void *a) { (void)a; al_cb_body(); }

static void al_check_cb(const char *what) {
    CHECKF(g_cb_hits > 0, "%s: callback not invoked", what);
    if (!g_cb_hits) return;
    t_info(what, "sp%%16=%u", (unsigned)(g_cb_sp % 16));
#ifdef T_X64
    CHECKF(g_cb_sp % 16 == 8, "%s: RSP at entry = 0x%llx, %%16 = %u (x64 ABI requires 8)", what, (unsigned long long)g_cb_sp, (unsigned)(g_cb_sp % 16));
#else
    CHECKF(g_cb_sp % 4 == 0, "%s: ESP at entry = 0x%lx not 4-byte aligned", what, (unsigned long)g_cb_sp);
#endif
    CHECKF(g_cb_loc16 % 16 == 0 && g_cb_loc32 % 32 == 0 && g_cb_loc64 % 64 == 0,
           "%s: aligned locals misplaced: %p %p %p", what, (void *)g_cb_loc16, (void *)g_cb_loc32, (void *)g_cb_loc64);
}
static void al_arm_cb(void *target) { g_cb_hits = 0; g_cb_sp = 1; al_sp_slot = 1; al_sp_target = target; }

TEST(stack_alignment_direct_call) {
    uintptr_t sp = al_get_sp();
#ifdef T_X64
    CHECKF(sp % 16 == 8, "RSP at entry of a called function %% 16 = %u", (unsigned)(sp % 16));
#else
    CHECK(sp % 4 == 0);
    t_info("esp_mod16", "%u", (unsigned)(sp % 16));
#endif
    T_ALIGN(32) volatile uint8_t l32[32]; T_ALIGN(64) volatile uint8_t l64[64];
    l32[0] = l64[0] = 0;
    CHECK((uintptr_t)l32 % 32 == 0);
    CHECK((uintptr_t)l64 % 64 == 0);
}
TEST(stack_alignment_thread_start) {
    static const SIZE_T sizes[] = { 0, 65536, 65536 + 8, 200000 + 4 };
    for (size_t i = 0; i < T_COUNTOF(sizes); i++) {
        al_arm_cb((void *)al_cb_thread);
        HANDLE h = CreateThread(NULL, sizes[i], (LPTHREAD_START_ROUTINE)(void *)al_sp_thunk, NULL, 0, NULL);
        if (!h) SKIP("CreateThread failed");
        CHECK_EQ(WaitForSingleObject(h, 10000), WAIT_OBJECT_0);
        CloseHandle(h);
        char w[48]; snprintf(w, sizeof w, "thread_start_stack%lu", (unsigned long)sizes[i]);
        al_check_cb(w);
    }
}
TEST(stack_alignment_apc) {
    al_arm_cb((void *)al_cb_apc);
    REQUIRE(QueueUserAPC((PAPCFUNC)(void *)al_sp_thunk, GetCurrentThread(), 0));
    SleepEx(0, TRUE);
    al_check_cb("apc");
}
TEST(stack_alignment_vectored_handler) {
    al_arm_cb((void *)al_cb_veh);
    void *h = AddVectoredExceptionHandler(1, (PVECTORED_EXCEPTION_HANDLER)(void *)al_sp_thunk);
    REQUIRE(h);
    RaiseException(0xE0A11600u, 0, 0, NULL);
    RemoveVectoredExceptionHandler(h);
    al_check_cb("vectored_handler");
}
#pragma GCC diagnostic ignored "-Warray-bounds"   /* mingw's inline GetCurrentFiber() reads TEB via a [0] array */
TEST(stack_alignment_fiber_start) {
    g_main_fiber = ConvertThreadToFiber(NULL);
    int converted = g_main_fiber != NULL;
    if (!g_main_fiber) g_main_fiber = GetCurrentFiber();
    REQUIRE(g_main_fiber);
    al_arm_cb((void *)al_cb_fiber);
    void *f = CreateFiber(0, (LPFIBER_START_ROUTINE)(void *)al_sp_thunk, NULL);
    if (!f) { if (converted) ConvertFiberToThread(); SKIP("CreateFiber failed"); }
    SwitchToFiber(f);
    DeleteFiber(f);
    if (converted) ConvertFiberToThread();
    al_check_cb("fiber_start");
}
TEST(stack_alignment_threadpool_callback) {
    g_cb_event = CreateEventA(NULL, FALSE, FALSE, NULL);
    REQUIRE(g_cb_event);
    al_arm_cb((void *)al_cb_tp);
    REQUIRE(TrySubmitThreadpoolCallback((PTP_SIMPLE_CALLBACK)(void *)al_sp_thunk, NULL, NULL));
    CHECK_EQ(WaitForSingleObject(g_cb_event, 10000), WAIT_OBJECT_0);
    Sleep(10);
    CloseHandle(g_cb_event);
    al_check_cb("threadpool_callback");
}
TEST(stack_alignment_timer_queue_callback) {
    g_cb_event = CreateEventA(NULL, FALSE, FALSE, NULL);
    REQUIRE(g_cb_event);
    al_arm_cb((void *)al_cb_timer);
    HANDLE t = NULL;
    REQUIRE(CreateTimerQueueTimer(&t, NULL, (WAITORTIMERCALLBACK)(void *)al_sp_thunk, NULL, 1, 0, WT_EXECUTEDEFAULT));
    CHECK_EQ(WaitForSingleObject(g_cb_event, 10000), WAIT_OBJECT_0);
    DeleteTimerQueueTimer(NULL, t, INVALID_HANDLE_VALUE);
    CloseHandle(g_cb_event);
    al_check_cb("timer_queue_callback");
}
TEST(stack_alignment_registered_wait_callback) {
    g_cb_event = CreateEventA(NULL, FALSE, FALSE, NULL);
    HANDLE sig = CreateEventA(NULL, TRUE, TRUE, NULL);
    REQUIRE(g_cb_event && sig);
    al_arm_cb((void *)al_cb_wait);
    HANDLE w = NULL;
    REQUIRE(RegisterWaitForSingleObject(&w, sig, (WAITORTIMERCALLBACK)(void *)al_sp_thunk, NULL, 5000, WT_EXECUTEONLYONCE));
    CHECK_EQ(WaitForSingleObject(g_cb_event, 10000), WAIT_OBJECT_0);
    UnregisterWaitEx(w, INVALID_HANDLE_VALUE);
    CloseHandle(g_cb_event); CloseHandle(sig);
    al_check_cb("registered_wait_callback");
}
TEST(stack_alignment_kernel32_enum_callback) {
    al_arm_cb((void *)al_cb_codepage);
    EnumSystemCodePagesA((CODEPAGE_ENUMPROCA)(void *)al_sp_thunk, CP_INSTALLED);
    al_check_cb("enum_codepages_callback");
}
TEST(stack_alignment_fls_callback) {
    al_arm_cb((void *)al_cb_fls);
    DWORD idx = FlsAlloc((PFLS_CALLBACK_FUNCTION)(void *)al_sp_thunk);
    if (idx == FLS_OUT_OF_INDEXES) SKIP("FlsAlloc failed");
    FlsSetValue(idx, (void *)1);
    FlsFree(idx);
    if (!g_cb_hits) { t_note("FlsFree did not invoke the callback for the current fiber"); return; }
    al_check_cb("fls_callback");
}

/* ======================================================================================
 * Cases: static, heap, CRT and virtual memory alignment
 * ====================================================================================== */
static T_ALIGN(4096) uint8_t g_bss_page[4096];
static T_ALIGN(64) uint8_t g_data64[64] = { 1 };
static const T_ALIGN(32) uint8_t g_rdata32[32] = { 2 };
TEST(static_data_alignment_honoured_by_loader) {
    g_bss_page[0] = 1;
    CHECKF((uintptr_t)g_bss_page % 4096 == 0, ".bss page-aligned object at %p", (void *)g_bss_page);
    CHECKF((uintptr_t)g_data64 % 64 == 0, ".data 64-aligned object at %p", (void *)g_data64);
    CHECKF((uintptr_t)g_rdata32 % 32 == 0, ".rdata 32-aligned object at %p", (const void *)g_rdata32);
    CHECKF((uintptr_t)g_aux % 64 == 0 && (uintptr_t)g_src % 64 == 0, "64-aligned statics");
    HMODULE m = GetModuleHandleA(NULL);
    CHECKF((uintptr_t)m % 65536 == 0, "image base %p not 64 KiB aligned", (void *)m);
}

static void al_heap_sweep(HANDLE heap, const char *what) {
    static void *ptrs[1200];
    int bad = 0, n16 = 0, total = 0;
    for (int i = 0; i < 1200; i++) {
        size_t sz = i < 1024 ? (size_t)(i % 512) + 1 : (size_t)(4096u << (i % 8));
        ptrs[i] = HeapAlloc(heap, i & 1 ? HEAP_ZERO_MEMORY : 0, sz);
        if (!ptrs[i]) continue;
        total++;
        if ((uintptr_t)ptrs[i] % MEMORY_ALLOCATION_ALIGNMENT) { if (bad++ < 4) CHECKF(0, "%s: HeapAlloc(%u) = %p not %d-aligned", what, (unsigned)sz, ptrs[i], MEMORY_ALLOCATION_ALIGNMENT); }
        if ((uintptr_t)ptrs[i] % 16 == 0) n16++;
        memset(ptrs[i], 0xA5, sz < 64 ? sz : 64);
    }
    CHECKF(bad == 0, "%s: %d of %d blocks misaligned", what, bad, total);
    CHECKF(total >= 1100, "%s: only %d of 1200 allocations succeeded", what, total);
    t_info(what, "%d blocks, %d 16-aligned", total, n16);
    /* realloc keeps alignment and content */
    for (int i = 0; i < 1200; i += 7) {
        if (!ptrs[i]) continue;
        void *q = HeapReAlloc(heap, 0, ptrs[i], (size_t)(i % 300) * 3 + 100);
        if (!q) continue;
        ptrs[i] = q;
        CHECKF((uintptr_t)q % MEMORY_ALLOCATION_ALIGNMENT == 0, "%s: HeapReAlloc result %p misaligned", what, q);
        CHECKF(((uint8_t *)q)[0] == 0xA5, "%s: HeapReAlloc lost content", what);
    }
    for (int i = 0; i < 1200; i++) if (ptrs[i]) HeapFree(heap, 0, ptrs[i]);
}
TEST(heap_alloc_memory_allocation_alignment) {
    t_info("MEMORY_ALLOCATION_ALIGNMENT", "%d", MEMORY_ALLOCATION_ALIGNMENT);
#ifdef T_X64
    CHECK_EQ(MEMORY_ALLOCATION_ALIGNMENT, 16);
#else
    CHECK_EQ(MEMORY_ALLOCATION_ALIGNMENT, 8);
#endif
    al_heap_sweep(GetProcessHeap(), "process_heap");
    HANDLE h1 = HeapCreate(0, 0, 0), h2 = HeapCreate(HEAP_NO_SERIALIZE, 1 << 20, 0);
    REQUIRE(h1 && h2);
    al_heap_sweep(h1, "private_heap");
    al_heap_sweep(h2, "private_heap_noserialize");
    HeapDestroy(h1); HeapDestroy(h2);
}
TEST(heap_lfh_bucket_alignment) {
    /* many same-size blocks activate the low-fragmentation front end */
    HANDLE h = HeapCreate(0, 0, 0);
    REQUIRE(h);
    static void *p[3000];
    int bad = 0;
    static const size_t sizes[] = { 8, 24, 40, 72, 136 };
    for (size_t s = 0; s < T_COUNTOF(sizes); s++) {
        for (int i = 0; i < 600; i++) {
            p[s * 600 + i] = HeapAlloc(h, 0, sizes[s]);
            if (p[s * 600 + i] && (uintptr_t)p[s * 600 + i] % MEMORY_ALLOCATION_ALIGNMENT) bad++;
        }
    }
    CHECK_EQ(bad, 0);
    for (int i = 0; i < 3000; i++) if (p[i]) HeapFree(h, 0, p[i]);
    HeapDestroy(h);
}
TEST(crt_malloc_fundamental_alignment) {
    const uintptr_t want = 2 * sizeof(void *);   /* MSVC CRT: 16 on x64, 8 on x86 */
    int bad = 0;
    for (size_t sz = 1; sz <= 2048; sz += (sz < 64 ? 1 : 61)) {
        void *a = malloc(sz), *c = calloc(1, sz), *r = malloc(sz);
        if ((a && (uintptr_t)a % want) || (c && (uintptr_t)c % want) || (r && (uintptr_t)r % want)) bad++;
        void *r2 = r ? realloc(r, sz * 3) : NULL;
        if (r2) r = r2;
        if (r2 && (uintptr_t)r2 % want) bad++;
        free(a); free(c); free(r);
    }
    CHECKF(bad == 0, "%d malloc/calloc/realloc results not %u-aligned", bad, (unsigned)want);
}
TEST(crt_aligned_malloc_power_of_two) {
    int n = 0;
    for (size_t a = 1; a <= 65536; a <<= 1) {
        static const size_t sizes[] = { 1, 17, 4099 };
        for (size_t s = 0; s < T_COUNTOF(sizes); s++) {
            uint8_t *p = (uint8_t *)_aligned_malloc(sizes[s], a);
            CHECKF(p != NULL, "_aligned_malloc(%u, %u) failed", (unsigned)sizes[s], (unsigned)a);
            if (!p) continue;
            CHECKF((uintptr_t)p % a == 0, "_aligned_malloc(%u, %u) = %p", (unsigned)sizes[s], (unsigned)a, (void *)p);
            memset(p, 0x3C, sizes[s]);
            uint8_t *q = (uint8_t *)_aligned_realloc(p, sizes[s] * 2 + 5, a);
            CHECKF(q && (uintptr_t)q % a == 0, "_aligned_realloc keeps alignment %u", (unsigned)a);
            if (q) { CHECKF(q[0] == 0x3C && q[sizes[s] - 1] == 0x3C, "_aligned_realloc lost content"); p = q; }
            _aligned_free(p);
            n++;
        }
    }
    t_info("combinations", "%d", n);
}
TEST(crt_aligned_offset_malloc) {
    for (size_t a = 8; a <= 4096; a <<= 1)
        for (size_t off = 0; off < 64 && off < a; off += 7) {
            uint8_t *p = (uint8_t *)_aligned_offset_malloc(256, a, off);
            CHECKF(p != NULL, "_aligned_offset_malloc(256,%u,%u) failed", (unsigned)a, (unsigned)off);
            if (!p) continue;
            CHECKF(((uintptr_t)p + off) % a == 0, "_aligned_offset_malloc(256,%u,%u)=%p", (unsigned)a, (unsigned)off, (void *)p);
            memset(p, 1, 256);
            _aligned_free(p);
        }
}
TEST(crt_aligned_malloc_rejects_non_power_of_two) {
    t_child_run("al_alignedbad", NULL, 15000, &g_cr);
    al_child_ok("al_alignedbad");
    int rows = 0;
    for (const char *l = g_cr.out; l && *l; l = strchr(l, '\n'), l = l ? l + 1 : NULL) {
        unsigned a; int isnull, e;
        if (strncmp(l, "AB ", 3) || sscanf(l + 3, "%u %d %d", &a, &isnull, &e) != 3) continue;
        rows++;
        CHECKF(isnull, "_aligned_malloc(64, %u) returned non-NULL", a);
        CHECKF(e == EINVAL, "_aligned_malloc(64, %u): errno %d, want EINVAL", a, e);
    }
    CHECK_EQ(rows, 5);
}

TEST(virtualalloc_reservation_granularity) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    uintptr_t g = si.dwAllocationGranularity;
    void *v[16];
    for (int i = 0; i < 16; i++) {
        v[i] = VirtualAlloc(NULL, (SIZE_T)(1 + i * 1000), i & 1 ? MEM_RESERVE : MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        CHECKF(v[i] && (uintptr_t)v[i] % g == 0, "VirtualAlloc #%d = %p not a multiple of %lu", i, v[i], (unsigned long)g);
        if (v[i] && !(i & 1)) {
            MEMORY_BASIC_INFORMATION mi;
            CHECK(VirtualQuery(v[i], &mi, sizeof mi) == sizeof mi);
            CHECKF(mi.RegionSize % PG == 0 && mi.RegionSize >= (SIZE_T)(1 + i * 1000), "committed region %lu bytes", (unsigned long)mi.RegionSize);
            ((volatile uint8_t *)v[i])[mi.RegionSize - 1] = 1;   /* the whole rounded page is usable */
        }
    }
    for (int i = 0; i < 16; i++) if (v[i]) VirtualFree(v[i], 0, MEM_RELEASE);
}
TEST(virtualalloc_explicit_address_rounding) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    uintptr_t g = si.dwAllocationGranularity;
    uint8_t *x = (uint8_t *)VirtualAlloc(NULL, 4 * g, MEM_RESERVE, PAGE_NOACCESS);
    REQUIRE(x);
    VirtualFree(x, 0, MEM_RELEASE);
    /* reserve at an unaligned address: rounded down to the allocation granularity */
    uint8_t *r = (uint8_t *)VirtualAlloc(x + g + 0x1234, 0x100, MEM_RESERVE, PAGE_READWRITE);
    if (!r) SKIP("address range taken by another allocation (error %lu)", GetLastError());
    CHECKF(r == x + g, "reserve at base+g+0x1234 returned %p, want %p", (void *)r, (void *)(x + g));
    /* commit at an unaligned address inside: rounded down to the page, covering every touched page */
    uint8_t *c = (uint8_t *)VirtualAlloc(r + 0x0FF0, 0x20, MEM_COMMIT, PAGE_READWRITE);
    CHECKF(c == r, "commit at +0xff0 returned %p, want %p", (void *)c, (void *)r);
    MEMORY_BASIC_INFORMATION mi;
    CHECK(VirtualQuery(r, &mi, sizeof mi) == sizeof mi);
    CHECKF(mi.State == MEM_COMMIT && mi.RegionSize == 2 * PG, "committed run state 0x%lx size 0x%lx, want 2 pages", mi.State, (unsigned long)mi.RegionSize);
    if (mi.State == MEM_COMMIT) { r[0] = 1; r[2 * PG - 1] = 2; }
    VirtualFree(r, 0, MEM_RELEASE);
}
typedef struct { void *lo, *hi; SIZE_T align; } al_mem_addr_req;
typedef struct { DWORD64 type; union { DWORD64 u64; void *ptr; SIZE_T size; HANDLE h; DWORD u32; }; } al_mem_ext_param;
typedef void *(WINAPI *al_virtualalloc2_t)(HANDLE, void *, SIZE_T, ULONG, ULONG, al_mem_ext_param *, ULONG);
TEST(virtualalloc2_alignment_requirement) {
    al_virtualalloc2_t va2 = (al_virtualalloc2_t)t_proc("kernelbase.dll", "VirtualAlloc2");
    if (!va2) SKIP("VirtualAlloc2 not exported (pre-1803)");
    static const SIZE_T aligns[] = { 0x10000, 0x20000, 0x100000, 0x400000 };
    for (size_t i = 0; i < T_COUNTOF(aligns); i++) {
        al_mem_addr_req req = { NULL, NULL, aligns[i] };
        al_mem_ext_param p; memset(&p, 0, sizeof p);
        p.type = 1;   /* MemExtendedParameterAddressRequirements */
        p.ptr = &req;
        uint8_t *v = (uint8_t *)va2(GetCurrentProcess(), NULL, 3 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, &p, 1);
        CHECKF(v != NULL, "VirtualAlloc2 alignment 0x%lx failed: error %lu", (unsigned long)aligns[i], GetLastError());
        if (!v) continue;
        CHECKF((uintptr_t)v % aligns[i] == 0, "VirtualAlloc2 alignment 0x%lx returned %p", (unsigned long)aligns[i], (void *)v);
        v[0] = v[3 * PG - 1] = 1;
        VirtualFree(v, 0, MEM_RELEASE);
    }
    /* a non-power-of-two alignment must be rejected */
    al_mem_addr_req bad = { NULL, NULL, 0x30000 };
    al_mem_ext_param p; memset(&p, 0, sizeof p); p.type = 1; p.ptr = &bad;
    SetLastError(0);
    void *v = va2(GetCurrentProcess(), NULL, PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, &p, 1);
    CHECKF(v == NULL, "VirtualAlloc2 accepted alignment 0x30000");
    t_info("bad_alignment_error", "%lu", GetLastError());
    if (v) VirtualFree(v, 0, MEM_RELEASE);
}
TEST(mapviewoffile_offset_and_base_alignment) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    DWORD g = si.dwAllocationGranularity;
    HANDLE s = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, 4 * g, NULL);
    REQUIRE(s);
    uint8_t *v0 = (uint8_t *)MapViewOfFile(s, FILE_MAP_WRITE, 0, 0, 0);
    REQUIRE(v0);
    CHECKF((uintptr_t)v0 % g == 0, "view %p not granularity aligned", (void *)v0);
    for (DWORD i = 0; i < 4 * g; i += PG) v0[i] = (uint8_t)(i / PG);
    SetLastError(0);
    void *bad = MapViewOfFile(s, FILE_MAP_READ, 0, PG, PG);
    CHECKF(bad == NULL, "offset 0x1000 (not a granularity multiple) accepted");
    if (!bad) CHECK_EQ(GetLastError(), ERROR_MAPPED_ALIGNMENT);
    else UnmapViewOfFile(bad);
    uint8_t *v1 = (uint8_t *)MapViewOfFile(s, FILE_MAP_READ, 0, g, PG);
    CHECKF(v1 != NULL, "offset = granularity rejected: %lu", GetLastError());
    if (v1) {
        CHECKF((uintptr_t)v1 % g == 0, "view %p not granularity aligned", (void *)v1);
        CHECK_EQ(v1[0], (uint8_t)(g / PG));
        UnmapViewOfFile(v1);
    }
    /* explicit base that is not a granularity multiple */
    uint8_t *x = (uint8_t *)VirtualAlloc(NULL, 2 * g, MEM_RESERVE, PAGE_NOACCESS);
    if (x) {
        VirtualFree(x, 0, MEM_RELEASE);
        SetLastError(0);
        void *vb = MapViewOfFileEx(s, FILE_MAP_READ, 0, 0, PG, x + PG);
        CHECKF(vb == NULL, "MapViewOfFileEx at a non-granularity base %p succeeded", (void *)(x + PG));
        t_info("unaligned_base_error", "%lu", GetLastError());
        if (vb) UnmapViewOfFile(vb);
    }
    UnmapViewOfFile(v0);
    CloseHandle(s);
}

TEST(initialize_context_buffer_alignment) {
    __typeof__(&InitializeContext) init = T_PROC("kernel32.dll", InitializeContext);
    __typeof__(&LocateXStateFeature) locate = T_PROC("kernel32.dll", LocateXStateFeature);
    __typeof__(&GetEnabledXStateFeatures) gef = T_PROC("kernel32.dll", GetEnabledXStateFeatures);
    if (!init) SKIP("InitializeContext not exported");
    DWORD64 feats = gef ? gef() : 0;
    int xs = (feats & XSTATE_MASK_AVX) && locate;
    DWORD flags = CONTEXT_ALL | (xs ? CONTEXT_XSTATE : 0);
#ifdef T_X64
    const uintptr_t calign = 16;
#else
    const uintptr_t calign = 4;
#endif
    t_info("flags", "0x%lx", flags);
    t_info("xstate_features", "0x%llx", (unsigned long long)feats);
    DWORD len = 0;
    CONTEXT *ctx = NULL;
    SetLastError(0);
    BOOL ok = init(NULL, flags, &ctx, &len);
    CHECKF(!ok, "InitializeContext(NULL) succeeded");
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    REQUIREF(len >= sizeof(CONTEXT), "length %lu < sizeof(CONTEXT) %u", len, (unsigned)sizeof(CONTEXT));
    t_info("length", "%lu", len);
    uint8_t *raw = (uint8_t *)VirtualAlloc(NULL, len + 4 * PG, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    REQUIRE(raw);
    int xs_aligned64 = 0, nxs = 0;
    for (int off = 0; off < 64; off++) {
        uint8_t *buf = raw + PG + off;
        memset(raw, 0xCD, len + 4 * PG);
        DWORD l2 = len;
        ctx = NULL;
        ok = init(buf, flags, &ctx, &l2);
        CHECKF(ok, "InitializeContext(buf+%d) failed: %lu", off, GetLastError());
        if (!ok || !ctx) continue;
        CHECKF((uintptr_t)ctx % calign == 0, "buf+%d: CONTEXT at %p not %u-aligned", off, (void *)ctx, (unsigned)calign);
        CHECKF((uint8_t *)ctx >= buf && (uint8_t *)ctx + sizeof(CONTEXT) <= buf + len, "buf+%d: CONTEXT [%p,+%u) outside the %lu-byte buffer at %p",
               off, (void *)ctx, (unsigned)sizeof(CONTEXT), len, (void *)buf);
        CHECKF(ctx->ContextFlags == flags, "buf+%d: ContextFlags 0x%lx", off, ctx->ContextFlags);
        CHECKF(raw[PG + off - 1] == 0xCD && buf[len] == 0xCD, "buf+%d: bytes outside the buffer written", off);
        if (xs) {
            DWORD fl = 0;
            uint8_t *ymm = (uint8_t *)locate(ctx, XSTATE_AVX, &fl);
            CHECKF(ymm != NULL, "buf+%d: LocateXStateFeature(AVX) NULL", off);
            if (ymm) {
                CHECKF(ymm >= buf && ymm + fl <= buf + len, "buf+%d: AVX area [%p,+%lu) outside the buffer", off, (void *)ymm, fl);
                nxs++;
                if ((uintptr_t)(ymm - 64) % 64 == 0) xs_aligned64++;   /* YMM_H follows the 64-byte XSAVE header */
            }
        }
    }
    if (xs) t_info("xstate_header_64_aligned", "%d of %d", xs_aligned64, nxs);
    VirtualFree(raw, 0, MEM_RELEASE);
}
