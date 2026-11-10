/* cxx_exceptions - C++ exceptions over the Windows x64 SEH unwinder, checking only ABI-required preservation.
 *
 * Built with x86_64-w64-mingw32-g++ (GCC defines __SEH__): `throw` raises STATUS_GCC_THROW through
 * RaiseException; the frame handler (__gxx_personality_seh0 / _GCC_specific_handler) then calls ntdll!RtlUnwindEx.
 * Every frame between the throw and the catch is undone by RtlVirtualUnwind from the frames' unwind info, and
 * the catch site resumes through RtlRestoreContext. This is the OS path the cases exercise. The x86 build
 * SKIPs every case: GCC's i686 C++ uses DWARF-2 or SJLJ unwinding, which never enters the OS unwinder. Only an
 * SEH-based 32-bit C++ compiler would test the OS path there.
 *
 * Contract (Microsoft x64 calling convention, "x64 ABI conventions" / "x64 exception handling"):
 *  - RBX, RBP, RDI, RSI, R12-R15 and XMM6-XMM15 (low 128 bits) are nonvolatile. They must hold their values
 *    at a catch site and in every frame that unwinding resumes, including frames resumed by RtlUnwindEx.
 *  - The MXCSR control bits (15:6) and the x87 control word are nonvolatile. The frames in between do not
 *    change them, so the catch site must see the values it set before the call.
 *  - The upper halves of YMM registers and XMM0-5 are volatile. They are only observed (t_info), never checked.
 *  - RtlUnwindEx (documented): unwinds to TargetFrame, continues at TargetIp, and places ReturnValue in
 *    the integer return register. ContextRecord is scratch "context during the unwind operation", so whatever
 *    the caller had stored there must not reach the target frame.
 *
 * Windows behaviour pinned against the Windows 10 19041 reference (corpus explorer decompilation):
 *  - ntdll!RtlUnwindEx: RtlCaptureContext2 overwrites ContextRecord first. If the CONTEXT lacks
 *    CONTEXT_XSTATE while XSAVE is enabled, it is replaced by an internal CONTEXT_EX. After an unwind handler
 *    returns ExceptionContinueSearch, the context's MxCsr is set to the live MXCSR, and the x87 state is
 *    re-saved when it is not the default. For the TARGET frame the handler's ContextRecord is not replaced
 *    before RtlRestoreContext, so an edit made by the target frame's unwind handler (EXCEPTION_TARGET_UNWIND)
 *    takes effect. That edit is not documented: rtlunwindex_target_handler_edit records it as info only.
 *  - ntdll!RtlUnwindEx calls the target frame's language handler with EXCEPTION_UNWINDING|EXCEPTION_TARGET_UNWIND
 *    (flags 0x22 for a TargetFrame unwind) and EstablisherFrame == TargetFrame.
 *
 * Deliberate hardware faults (catch(...) of a VEH-translated access violation or #DE) run in child processes.
 */
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "t.h"
#ifdef __cplusplus
}
#endif

#define CX_X86_REASON "x86: GCC's i686 C++ uses DWARF/SJLJ unwinding, not the OS SEH unwinder (no SEH-based 32-bit C++ compiler)"

#if defined(__cplusplus) && defined(__SEH__) && defined(__x86_64__)
#include <emmintrin.h>
#include <exception>

/* ---- shared register image (offsets are hard-coded in the asm below) ----------------------------- */
struct cx_regs {
    uint8_t  xmm[16][16];     /*   0: XMM0-15 (6-15 loaded/stored; 5 stored by the RtlUnwindEx target)  */
    uint8_t  yhi[16][16];     /* 256: YMM upper halves (AVX variant)                                    */
    uint64_t gpr[8];          /* 512: rbx rbp rsi rdi r12 r13 r14 r15                                   */
    uint32_t mxcsr;           /* 576 */
    uint16_t fcw;             /* 580 */
    uint16_t pad0;
    uint64_t rax;             /* 584 */
} __attribute__((aligned(16)));
static_assert(offsetof(cx_regs, yhi) == 256 && offsetof(cx_regs, gpr) == 512 && offsetof(cx_regs, mxcsr) == 576 &&
              offsetof(cx_regs, fcw) == 580 && offsetof(cx_regs, rax) == 584, "cx_regs layout");
static const char *const cx_gpr_names[8] = { "rbx", "rbp", "rsi", "rdi", "r12", "r13", "r14", "r15" };

/* Each asm chain frame loads these (xor its depth) into XMM6-15 and the nonvolatile GPRs before calling on. */
struct cx_chain_ctx {
    uint8_t  clob[10][16];    /*   0: XMM6-15 clobber values */
    uint64_t gclob[8];        /* 160: rbx rbp rsi rdi r12-r15 clobber values */
    void   (*leaf)(void *);   /* 224 */
    void    *leaf_arg;        /* 232 */
} __attribute__((aligned(16)));
static_assert(offsetof(cx_chain_ctx, gclob) == 160 && offsetof(cx_chain_ctx, leaf) == 224 &&
              offsetof(cx_chain_ctx, leaf_arg) == 232, "cx_chain_ctx layout");

extern "C" {
void cx_outer(const cx_regs *in, cx_regs *out, void (*fn)(void *), void *arg);
void cx_outer_avx(const cx_regs *in, cx_regs *out, void (*fn)(void *), void *arg);
void cx_chain(uint64_t depth, cx_chain_ctx *c);
void cx_fault_frame(uint64_t kind, const void *addr, cx_chain_ctx *c);
void cx_unw_target(const cx_regs *in, cx_regs *out, void (*fn)(void *target, void *arg), void *arg);
extern char cx_unw_landing[], cx_fault_av_insn[], cx_fault_de_insn[], cx_fault_landing[];
EXCEPTION_DISPOSITION cx_unw_handler(PEXCEPTION_RECORD rec, PVOID frame, PCONTEXT ctx, PVOID dc);
void cx_translate_throw(uint64_t code, uint64_t addr);
}

/* Frame layout shared by every asm routine: 8 GPR pushes, 216 bytes: [0,32) home area, [32,192) XMM6-15
 * saves (.seh_savexmm), 192/200/208 scratch slots. Entry RSP = 8 mod 16 -> body RSP = 0 mod 16. */
#define CX_BEGIN(name, extra) \
    ".globl " name "\n.def " name "; .scl 2; .type 32; .endef\n.seh_proc " name "\n" name ":\n" extra \
    "  pushq %rbp\n  .seh_pushreg %rbp\n  pushq %rbx\n  .seh_pushreg %rbx\n" \
    "  pushq %rsi\n  .seh_pushreg %rsi\n  pushq %rdi\n  .seh_pushreg %rdi\n" \
    "  pushq %r12\n  .seh_pushreg %r12\n  pushq %r13\n  .seh_pushreg %r13\n" \
    "  pushq %r14\n  .seh_pushreg %r14\n  pushq %r15\n  .seh_pushreg %r15\n" \
    "  subq $216, %rsp\n  .seh_stackalloc 216\n" \
    "  movdqa %xmm6, 32(%rsp)\n  .seh_savexmm %xmm6, 32\n  movdqa %xmm7, 48(%rsp)\n  .seh_savexmm %xmm7, 48\n" \
    "  movdqa %xmm8, 64(%rsp)\n  .seh_savexmm %xmm8, 64\n  movdqa %xmm9, 80(%rsp)\n  .seh_savexmm %xmm9, 80\n" \
    "  movdqa %xmm10, 96(%rsp)\n  .seh_savexmm %xmm10, 96\n  movdqa %xmm11, 112(%rsp)\n  .seh_savexmm %xmm11, 112\n" \
    "  movdqa %xmm12, 128(%rsp)\n  .seh_savexmm %xmm12, 128\n  movdqa %xmm13, 144(%rsp)\n  .seh_savexmm %xmm13, 144\n" \
    "  movdqa %xmm14, 160(%rsp)\n  .seh_savexmm %xmm14, 160\n  movdqa %xmm15, 176(%rsp)\n  .seh_savexmm %xmm15, 176\n" \
    "  .seh_endprologue\n"
#define CX_END \
    "  movdqa 32(%rsp), %xmm6\n  movdqa 48(%rsp), %xmm7\n  movdqa 64(%rsp), %xmm8\n  movdqa 80(%rsp), %xmm9\n" \
    "  movdqa 96(%rsp), %xmm10\n  movdqa 112(%rsp), %xmm11\n  movdqa 128(%rsp), %xmm12\n  movdqa 144(%rsp), %xmm13\n" \
    "  movdqa 160(%rsp), %xmm14\n  movdqa 176(%rsp), %xmm15\n" \
    "  addq $216, %rsp\n  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n" \
    "  popq %rdi\n  popq %rsi\n  popq %rbx\n  popq %rbp\n  ret\n.seh_endproc\n"
/* load XMM6-15 / GPRs from a cx_regs at %rcx; store them to a cx_regs at %rcx */
#define CX_LOAD_RCX \
    "  movdqu 96(%rcx), %xmm6\n  movdqu 112(%rcx), %xmm7\n  movdqu 128(%rcx), %xmm8\n  movdqu 144(%rcx), %xmm9\n" \
    "  movdqu 160(%rcx), %xmm10\n  movdqu 176(%rcx), %xmm11\n  movdqu 192(%rcx), %xmm12\n  movdqu 208(%rcx), %xmm13\n" \
    "  movdqu 224(%rcx), %xmm14\n  movdqu 240(%rcx), %xmm15\n" \
    "  movq 512(%rcx), %rbx\n  movq 520(%rcx), %rbp\n  movq 528(%rcx), %rsi\n  movq 536(%rcx), %rdi\n" \
    "  movq 544(%rcx), %r12\n  movq 552(%rcx), %r13\n  movq 560(%rcx), %r14\n  movq 568(%rcx), %r15\n"
#define CX_STORE_RCX \
    "  movdqu %xmm6, 96(%rcx)\n  movdqu %xmm7, 112(%rcx)\n  movdqu %xmm8, 128(%rcx)\n  movdqu %xmm9, 144(%rcx)\n" \
    "  movdqu %xmm10, 160(%rcx)\n  movdqu %xmm11, 176(%rcx)\n  movdqu %xmm12, 192(%rcx)\n  movdqu %xmm13, 208(%rcx)\n" \
    "  movdqu %xmm14, 224(%rcx)\n  movdqu %xmm15, 240(%rcx)\n" \
    "  movq %rbx, 512(%rcx)\n  movq %rbp, 520(%rcx)\n  movq %rsi, 528(%rcx)\n  movq %rdi, 536(%rcx)\n" \
    "  movq %r12, 544(%rcx)\n  movq %r13, 552(%rcx)\n  movq %r14, 560(%rcx)\n  movq %r15, 568(%rcx)\n" \
    "  stmxcsr 576(%rcx)\n  fnstcw 580(%rcx)\n"
/* clobber XMM6-15 / GPRs from the cx_chain_ctx at %r10, xor depth in %rax */
#define CX_CLOBBER_R10 \
    "  movq %rax, %xmm0\n  punpcklqdq %xmm0, %xmm0\n" \
    "  movdqu 0(%r10), %xmm6\n  pxor %xmm0, %xmm6\n  movdqu 16(%r10), %xmm7\n  pxor %xmm0, %xmm7\n" \
    "  movdqu 32(%r10), %xmm8\n  pxor %xmm0, %xmm8\n  movdqu 48(%r10), %xmm9\n  pxor %xmm0, %xmm9\n" \
    "  movdqu 64(%r10), %xmm10\n  pxor %xmm0, %xmm10\n  movdqu 80(%r10), %xmm11\n  pxor %xmm0, %xmm11\n" \
    "  movdqu 96(%r10), %xmm12\n  pxor %xmm0, %xmm12\n  movdqu 112(%r10), %xmm13\n  pxor %xmm0, %xmm13\n" \
    "  movdqu 128(%r10), %xmm14\n  pxor %xmm0, %xmm14\n  movdqu 144(%r10), %xmm15\n  pxor %xmm0, %xmm15\n" \
    "  movq 160(%r10), %rbx\n  xorq %rax, %rbx\n  movq 168(%r10), %rbp\n  xorq %rax, %rbp\n" \
    "  movq 176(%r10), %rsi\n  xorq %rax, %rsi\n  movq 184(%r10), %rdi\n  xorq %rax, %rdi\n" \
    "  movq 192(%r10), %r12\n  xorq %rax, %r12\n  movq 200(%r10), %r13\n  xorq %rax, %r13\n" \
    "  movq 208(%r10), %r14\n  xorq %rax, %r14\n  movq 216(%r10), %r15\n  xorq %rax, %r15\n"

__asm__(
    ".text\n"
    /* cx_outer(in, out, fn, arg): load XMM6-15, nonvolatile GPRs, MXCSR and x87 CW from *in; fn(arg); store
     * them to *out; restore the caller's MXCSR/CW. Anything between load and store is fn alone, so *out is
     * exactly what the ABI promises a caller after fn returns. */
    CX_BEGIN("cx_outer", "")
    "  movq %rdx, 192(%rsp)\n  stmxcsr 200(%rsp)\n  fnstcw 204(%rsp)\n"
    "  movq %r8, %rax\n  movq %r9, %r10\n"
    "  ldmxcsr 576(%rcx)\n  fldcw 580(%rcx)\n"
    CX_LOAD_RCX
    "  movq %r10, %rcx\n  call *%rax\n"
    "  movq 192(%rsp), %rcx\n"
    CX_STORE_RCX
    "  ldmxcsr 200(%rsp)\n  fldcw 204(%rsp)\n"
    CX_END
    /* AVX variant: also loads YMM6-15 upper halves from in->yhi and stores them to out->yhi. */
    CX_BEGIN("cx_outer_avx", "")
    "  movq %rdx, 192(%rsp)\n  stmxcsr 200(%rsp)\n  fnstcw 204(%rsp)\n"
    "  movq %r8, %rax\n  movq %r9, %r10\n"
    "  ldmxcsr 576(%rcx)\n  fldcw 580(%rcx)\n"
    CX_LOAD_RCX
    "  vinsertf128 $1, 352(%rcx), %ymm6, %ymm6\n  vinsertf128 $1, 368(%rcx), %ymm7, %ymm7\n"
    "  vinsertf128 $1, 384(%rcx), %ymm8, %ymm8\n  vinsertf128 $1, 400(%rcx), %ymm9, %ymm9\n"
    "  vinsertf128 $1, 416(%rcx), %ymm10, %ymm10\n  vinsertf128 $1, 432(%rcx), %ymm11, %ymm11\n"
    "  vinsertf128 $1, 448(%rcx), %ymm12, %ymm12\n  vinsertf128 $1, 464(%rcx), %ymm13, %ymm13\n"
    "  vinsertf128 $1, 480(%rcx), %ymm14, %ymm14\n  vinsertf128 $1, 496(%rcx), %ymm15, %ymm15\n"
    "  movq %r10, %rcx\n  call *%rax\n"
    "  movq 192(%rsp), %rcx\n"
    CX_STORE_RCX
    "  vextractf128 $1, %ymm6, 352(%rcx)\n  vextractf128 $1, %ymm7, 368(%rcx)\n"
    "  vextractf128 $1, %ymm8, 384(%rcx)\n  vextractf128 $1, %ymm9, 400(%rcx)\n"
    "  vextractf128 $1, %ymm10, 416(%rcx)\n  vextractf128 $1, %ymm11, 432(%rcx)\n"
    "  vextractf128 $1, %ymm12, 448(%rcx)\n  vextractf128 $1, %ymm13, 464(%rcx)\n"
    "  vextractf128 $1, %ymm14, 480(%rcx)\n  vextractf128 $1, %ymm15, 496(%rcx)\n"
    "  vzeroupper\n"
    "  ldmxcsr 200(%rsp)\n  fldcw 204(%rsp)\n"
    CX_END
    /* cx_chain(depth, ctx): one frame that saves XMM6-15 + GPRs in its prologue, clobbers them, then calls
     * cx_chain(depth-1) or, at depth 0, ctx->leaf(ctx->leaf_arg). depth+1 frames in total. */
    CX_BEGIN("cx_chain", "")
    "  movq %rcx, %rax\n  movq %rdx, %r10\n"
    CX_CLOBBER_R10
    "  testq %rax, %rax\n  jz 1f\n"
    "  leaq -1(%rax), %rcx\n  movq %r10, %rdx\n  call cx_chain\n  jmp 2f\n"
    "1:\n  movq 232(%r10), %rcx\n  call *224(%r10)\n"
    "2:\n"
    CX_END
    /* cx_fault_frame(kind, addr, ctx): clobbers like cx_chain, then faults: kind 0 = load from addr (#PF),
     * kind 1 = DIV by zero (#DE). The vectored handler redirects RIP to cx_fault_landing (same frame, same
     * RSP) with RCX = exception code, RDX = address, which calls cx_translate_throw like an ordinary call. */
    CX_BEGIN("cx_fault_frame", "")
    "  movq %r8, %r10\n  xorl %eax, %eax\n"
    CX_CLOBBER_R10
    "  testq %rcx, %rcx\n  jnz 1f\n"
    ".globl cx_fault_av_insn\ncx_fault_av_insn:\n  movq (%rdx), %rax\n  jmp 2f\n"
    "1:\n  movl $1, %eax\n  xorl %edx, %edx\n  xorl %ecx, %ecx\n"
    ".globl cx_fault_de_insn\ncx_fault_de_insn:\n  divq %rcx\n  jmp 2f\n"
    ".globl cx_fault_landing\ncx_fault_landing:\n  call cx_translate_throw\n  nop\n"
    "2:\n"
    CX_END
    /* cx_unw_target(in, out, fn, arg): frame with an unwind handler (cx_unw_handler). Loads XMM6-15 and GPRs,
     * calls fn(establisher frame = RSP, arg). fn unwinds back with RtlUnwindEx(TargetFrame = that RSP,
     * TargetIp = cx_unw_landing, ReturnValue), and the landing stores what the target frame was resumed with. */
    CX_BEGIN("cx_unw_target", "  .seh_handler cx_unw_handler, @unwind\n")
    "  movq %rdx, 192(%rsp)\n  movq %r8, %rax\n  movq %r9, %r11\n"
    CX_LOAD_RCX
    "  movq %rsp, %rcx\n  movq %r11, %rdx\n  call *%rax\n"
    ".globl cx_unw_landing\ncx_unw_landing:\n"
    "  movq 192(%rsp), %rcx\n  movdqu %xmm5, 80(%rcx)\n  movq %rax, 584(%rcx)\n"
    CX_STORE_RCX
    CX_END
);

/* ---- exception types with a live-object counter ------------------------------------------------- */
static volatile LONG g_cx_live;
struct CxErr {
    uint64_t v; uint64_t guard;
    explicit CxErr(uint64_t x) : v(x), guard(~x) { InterlockedIncrement(&g_cx_live); }
    CxErr(const CxErr &o) : v(o.v), guard(o.guard) { InterlockedIncrement(&g_cx_live); }
    ~CxErr() { InterlockedDecrement(&g_cx_live); }
    bool ok() const { return guard == ~v; }
};
struct HwFault { uint64_t code, addr; };

extern "C" void cx_translate_throw(uint64_t code, uint64_t addr) { throw HwFault{code, addr}; }

/* ---- scenario ------------------------------------------------------------------------------------ */
enum { M_PLAIN, M_RETHROW, M_NESTED, M_EPTR, M_DTOR, M_FIBER, M_COMPILED };
#define CX_NLIVE 10
struct Scn {
    int mode, depth, depth2;
    uint32_t tag;
    uint64_t payload;
    cx_regs in, out;
    cx_chain_ctx chain, chain2;
    uint8_t live[CX_NLIVE][16];       /* compiled-frame vectors kept live across the throwing call */
    /* results */
    int caught;                      /* 1 = CxErr, 2 = other */
    uint64_t caught_v; int caught_ok;
    unsigned catch_mxcsr, catch_fcw;
    int live_bad_catch;              /* live vectors wrong in the catching compiled frame */
    int live_bad_mid;                /* live vectors wrong in the rethrowing compiled frame (M_DTOR) */
    const void *obj_inner, *obj_outer;
    int nested_inner_ok, nested_outer_intact;
    int dtor_runs, dtor_uncaught;
    unsigned dtor_mxcsr, dtor_fcw;
    int dtor_vreg_bad, dtor_vreg_runs;
    std::exception_ptr eptr;
    void *main_fiber; int fiber_switches;
    int returned_normally;
};

static void cx_fill(Scn *s, uint32_t tag, int mode, int depth) {
    s->mode = mode; s->depth = depth; s->depth2 = 3; s->tag = tag;
    memset(&s->in, 0, sizeof s->in); memset(&s->out, 0, sizeof s->out);
    for (int i = 0; i < 16; i++) { t_pattern_reg(s->in.xmm[i], 16, tag, i); t_pattern_reg(s->in.yhi[i], 16, tag, 32 + i); }
    for (int i = 0; i < 8; i++) s->in.gpr[i] = t_pattern_word(tag, 16 + i, 0);
    s->in.mxcsr = 0xBF80;   /* all masked, RC = round down, FTZ */
    s->in.fcw = 0x077F;     /* all masked, PC = 64-bit, RC = round down */
    for (int c = 0; c < 2; c++) {
        cx_chain_ctx *k = c ? &s->chain2 : &s->chain;
        for (int i = 0; i < 10; i++) t_pattern_reg(k->clob[i], 16, tag + 1000 + c, 6 + i);
        for (int i = 0; i < 8; i++) k->gclob[i] = t_pattern_word(tag + 1000 + c, 16 + i, 1);
        k->leaf = NULL; k->leaf_arg = s;
    }
    for (int i = 0; i < CX_NLIVE; i++) t_pattern_reg(s->live[i], 16, tag + 2000, i);
    s->payload = t_pattern_word(tag, 60, 7);
    s->caught = 0; s->caught_v = 0; s->caught_ok = 0; s->catch_mxcsr = s->catch_fcw = 0;
    s->live_bad_catch = s->live_bad_mid = -1;
    s->obj_inner = s->obj_outer = NULL; s->nested_inner_ok = s->nested_outer_intact = 0;
    s->dtor_runs = 0; s->dtor_uncaught = -1; s->dtor_mxcsr = s->dtor_fcw = 0; s->dtor_vreg_bad = 0; s->dtor_vreg_runs = 0;
    s->eptr = nullptr; s->main_fiber = NULL; s->fiber_switches = 0; s->returned_normally = 0;
}

/* Ten vectors the compiler keeps live across a call (on x64 they go to XMM6-15 or to stack spills; either
 * way the unwinder must give them back). The empty asm makes them opaque so they cannot be rematerialized. */
#define LIVE_LOAD(s) \
    __m128i v0 = _mm_loadu_si128((const __m128i *)(s)->live[0]), v1 = _mm_loadu_si128((const __m128i *)(s)->live[1]); \
    __m128i v2 = _mm_loadu_si128((const __m128i *)(s)->live[2]), v3 = _mm_loadu_si128((const __m128i *)(s)->live[3]); \
    __m128i v4 = _mm_loadu_si128((const __m128i *)(s)->live[4]), v5 = _mm_loadu_si128((const __m128i *)(s)->live[5]); \
    __m128i v6 = _mm_loadu_si128((const __m128i *)(s)->live[6]), v7 = _mm_loadu_si128((const __m128i *)(s)->live[7]); \
    __m128i v8 = _mm_loadu_si128((const __m128i *)(s)->live[8]), v9 = _mm_loadu_si128((const __m128i *)(s)->live[9]); \
    __asm__ volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7), "+x"(v8), "+x"(v9))
#define LIVE_BAD(s, outvar) do { \
    __asm__ volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7), "+x"(v8), "+x"(v9)); \
    __m128i vv_[CX_NLIVE] = { v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 }; int b_ = 0; \
    for (int i_ = 0; i_ < CX_NLIVE; i_++) if (memcmp(&vv_[i_], (s)->live[i_], 16)) b_++; \
    (outvar) = b_; } while (0)

extern "C" void cx_leaf_throw(void *arg) { Scn *s = (Scn *)arg; throw CxErr(s->payload); }
extern "C" void cx_leaf_throw_int(void *arg) { (void)arg; throw 0x5EED; }
extern "C" void cx_leaf_rethrow_eptr(void *arg) { Scn *s = (Scn *)arg; std::rethrow_exception(s->eptr); }
extern "C" void cx_leaf_switch_throw(void *arg) {
    Scn *s = (Scn *)arg;
    s->fiber_switches++; SwitchToFiber(s->main_fiber);     /* main fiber runs, then switches back */
    s->fiber_switches++;
    throw CxErr(s->payload);
}

/* Destructor that runs during unwinding (phase 2, in the landing pad RtlUnwindEx resumed) and uses SIMD. */
struct SimdGuard {
    Scn *s;
    explicit SimdGuard(Scn *x) : s(x) {}
    ~SimdGuard() {
        s->dtor_runs++;
        s->dtor_uncaught = std::uncaught_exceptions();
        s->dtor_mxcsr = t_get_mxcsr(); s->dtor_fcw = t_get_x87_cw();
        /* full register files round-trip in this context, and XMM6-15 survive a call (Sleep(0)) */
        static t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, s->tag + 3000);
        if (t_vregs_call(T_VK_XMM, &in, &out, NULL, NULL) == 0) {
            s->dtor_vreg_runs++;
            for (int i = 0; i < 16; i++) if (memcmp(out.v[i], in.v[i], 16)) s->dtor_vreg_bad++;
        }
        if (t_vregs_call(T_VK_XMM, &in, &out, [](void *) { Sleep(0); }, NULL) == 0) {
            s->dtor_vreg_runs++;
            for (int i = 6; i < 16; i++) if (memcmp(out.v[i], in.v[i], 16)) s->dtor_vreg_bad++;
        }
        if (cpu_has(F_AVX)) {
            t_vregs_pattern(&in, T_VK_YMM, s->tag + 3001);
            if (t_vregs_call(T_VK_YMM, &in, &out, NULL, NULL) == 0) {
                s->dtor_vreg_runs++;
                for (int i = 0; i < 16; i++) if (memcmp(out.v[i], in.v[i], 32)) s->dtor_vreg_bad++;
            }
        }
    }
};

/* Compiled middle frame: a SimdGuard, live vectors across the throwing call, catch(...) that checks them and
 * rethrows. Reached as the leaf of the first asm chain; calls the second chain whose leaf throws. */
extern "C" T_NOINLINE void cx_mid_frame(void *arg) {
    Scn *s = (Scn *)arg;
    SimdGuard g(s);
    LIVE_LOAD(s);
    try {
        cx_chain((uint64_t)s->depth2, &s->chain2);
    } catch (...) {
        LIVE_BAD(s, s->live_bad_mid);
        throw;
    }
    s->returned_normally++;
}

/* Recursive compiled frames, each with ten vectors live across the recursive call. */
static T_NOINLINE uint64_t cx_compiled_rec(Scn *s, int depth) {
    __m128i a = _mm_set1_epi32(depth), b = _mm_add_epi32(a, a), c = _mm_xor_si128(a, b), d = _mm_sub_epi32(c, a);
    __m128i e = _mm_slli_epi32(a, 3), f = _mm_or_si128(e, b), g = _mm_add_epi64(f, c), h = _mm_unpacklo_epi32(g, d);
    __m128i i = _mm_unpackhi_epi64(h, e), j = _mm_xor_si128(i, f);
    __asm__ volatile("" : "+x"(a), "+x"(b), "+x"(c), "+x"(d), "+x"(e), "+x"(f), "+x"(g), "+x"(h), "+x"(i), "+x"(j));
    static void (*volatile thrower)(void *) = cx_leaf_throw;   /* opaque: keeps -Winfinite-recursion quiet */
    uint64_t r = depth == 0 ? (thrower(s), 0) : cx_compiled_rec(s, depth - 1);
    __asm__ volatile("" : "+x"(a), "+x"(b), "+x"(c), "+x"(d), "+x"(e), "+x"(f), "+x"(g), "+x"(h), "+x"(i), "+x"(j));
    __m128i x = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d)), _mm_xor_si128(_mm_xor_si128(e, f), _mm_xor_si128(g, h)));
    x = _mm_xor_si128(x, _mm_xor_si128(i, j));
    return r + (uint64_t)_mm_cvtsi128_si64(x);
}

/* What throws, per mode. Exceptions leave this function as CxErr (or not at all). */
static T_NOINLINE void cx_throwing(Scn *s) {
    switch (s->mode) {
    case M_PLAIN:
        s->chain.leaf = cx_leaf_throw;
        cx_chain((uint64_t)s->depth, &s->chain);
        break;
    case M_COMPILED:
        T_SINK(cx_compiled_rec(s, s->depth));
        break;
    case M_RETHROW:
        s->chain.leaf = cx_leaf_throw;
        try { cx_chain((uint64_t)s->depth, &s->chain); }
        catch (CxErr &e) { s->obj_inner = &e; throw; }
        break;
    case M_NESTED:
        s->chain.leaf = cx_leaf_throw; s->chain2.leaf = cx_leaf_throw_int;
        try { cx_chain((uint64_t)s->depth, &s->chain); }
        catch (CxErr &a) {
            s->obj_inner = &a;
            try { cx_chain((uint64_t)s->depth2, &s->chain2); }
            catch (int x) { s->nested_inner_ok = (x == 0x5EED); }
            s->nested_outer_intact = a.ok() && a.v == s->payload;
            uint64_t replaced = s->payload ^ 0xFFFF;
            s->payload = replaced;
            throw CxErr(replaced);          /* replaces the exception being handled */
        }
        break;
    case M_EPTR:
        s->chain.leaf = cx_leaf_throw; s->chain2.leaf = cx_leaf_rethrow_eptr;
        try { cx_chain((uint64_t)s->depth, &s->chain); }
        catch (...) { s->eptr = std::current_exception(); }
        cx_chain((uint64_t)s->depth2, &s->chain2);  /* rethrows the stored exception from deep down */
        break;
    case M_DTOR:
        s->chain.leaf = cx_mid_frame; s->chain2.leaf = cx_leaf_throw;
        cx_chain((uint64_t)s->depth, &s->chain);
        break;
    case M_FIBER:
        s->chain.leaf = cx_leaf_switch_throw;
        cx_chain((uint64_t)s->depth, &s->chain);
        break;
    }
    s->returned_normally++;
}

/* The catch site: a compiled frame with live vectors across the try, called from cx_outer. */
extern "C" void cx_body(void *arg) {
    Scn *s = (Scn *)arg;
    LIVE_LOAD(s);
    try {
        cx_throwing(s);
    } catch (CxErr &e) {
        s->catch_mxcsr = t_get_mxcsr(); s->catch_fcw = t_get_x87_cw();
        s->caught = 1; s->caught_v = e.v; s->caught_ok = e.ok(); s->obj_outer = &e;
    } catch (...) {
        s->catch_mxcsr = t_get_mxcsr(); s->catch_fcw = t_get_x87_cw();
        s->caught = 2;
    }
    s->eptr = nullptr;
    LIVE_BAD(s, s->live_bad_catch);
}

static void cx_run(Scn *s, int avx) { (avx ? cx_outer_avx : cx_outer)(&s->in, &s->out, cx_body, s); }

/* ABI-required state at the catch site and after the catching function returned to the asm caller. */
static void cx_check_abi(const cx_regs *in, const cx_regs *out, const char *what) {
    char lbl[96];
    for (int i = 6; i < 16; i++) {
        snprintf(lbl, sizeof lbl, "xmm%d %s", i, what);
        CHECK_REG(lbl, out->xmm[i], in->xmm[i], 16);
    }
    for (int i = 0; i < 8; i++) {
        snprintf(lbl, sizeof lbl, "%s %s", cx_gpr_names[i], what);
        CHECK_REG(lbl, &out->gpr[i], &in->gpr[i], 8);
    }
    CHECKF((out->mxcsr & 0xFFC0) == (in->mxcsr & 0xFFC0), "MXCSR control %s: got 0x%04x want 0x%04x", what,
           out->mxcsr & 0xFFC0, in->mxcsr & 0xFFC0);
    CHECKF((out->fcw & 0x1F3F) == (in->fcw & 0x1F3F), "x87 CW %s: got 0x%04x want 0x%04x", what, out->fcw, in->fcw);
}

static void cx_check_scn(Scn *s, const char *what) {
    CHECKF(s->caught == 1, "%s: catch(CxErr&) reached (caught=%d)", what, s->caught);
    CHECKF(s->returned_normally == 0, "%s: a throwing call returned normally", what);
    if (s->caught == 1) {
        CHECKF(s->caught_ok && s->caught_v == s->payload, "%s: exception object 0x%llx intact (want 0x%llx)", what,
               (unsigned long long)s->caught_v, (unsigned long long)s->payload);
        CHECKF((s->catch_mxcsr & 0xFFC0) == (s->in.mxcsr & 0xFFC0), "%s: MXCSR control at the catch site 0x%04x, want 0x%04x",
               what, s->catch_mxcsr & 0xFFC0, s->in.mxcsr & 0xFFC0);
        CHECKF((s->catch_fcw & 0x1F3F) == (s->in.fcw & 0x1F3F), "%s: x87 CW at the catch site 0x%04x, want 0x%04x",
               what, s->catch_fcw, s->in.fcw);
    }
    CHECKF(s->live_bad_catch == 0, "%s: %d of %d live vectors of the catching compiled frame changed", what,
           s->live_bad_catch, CX_NLIVE);
    cx_check_abi(&s->in, &s->out, what);
    CHECKF(g_cx_live == 0, "%s: %ld exception objects still alive after the handler", what, (long)g_cx_live);
}

static Scn g_s;   /* large (~2 KB) and shared with thread-pool and fiber cases: static */

static void cx_plain_case(int depth, uint32_t tag) {
    cx_fill(&g_s, tag, M_PLAIN, depth);
    cx_run(&g_s, 0);
    char w[64]; snprintf(w, sizeof w, "after throw through %d asm frames", depth + 1);
    cx_check_scn(&g_s, w);
    t_info("frames", "%d", depth + 1);
}

TEST(throw_1_asm_frame_xmm_saved)  { cx_plain_case(0, 11); }
TEST(throw_8_asm_frames_xmm_saved) { cx_plain_case(7, 12); }
TEST(throw_64_asm_frames_xmm_saved) { cx_plain_case(63, 13); }

TEST(throw_64_compiled_simd_frames) {
    cx_fill(&g_s, 14, M_COMPILED, 63);
    cx_run(&g_s, 0);
    cx_check_scn(&g_s, "after throw through 64 compiled frames with live vectors");
}

TEST(throw_8_frames_ymm_uppers_volatile) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    cx_fill(&g_s, 15, M_PLAIN, 7);
    cx_run(&g_s, 1);
    cx_check_scn(&g_s, "after throw (YMM uppers loaded)");
    /* Upper halves are volatile under the x64 ABI: record only. */
    int kept = 0, zero = 0;
    static const uint8_t z[16] = {0};
    for (int i = 6; i < 16; i++) {
        if (!memcmp(g_s.out.yhi[i], g_s.in.yhi[i], 16)) kept++;
        else if (!memcmp(g_s.out.yhi[i], z, 16)) zero++;
    }
    t_info("ymm6_15_upper_after_catch", "%s", kept == 10 ? "preserved" : zero == 10 ? "zeroed" : "mixed/other");
    t_note("YMM6-15 upper halves after catch: %d preserved, %d zero, %d other (volatile by ABI, not checked)",
           kept, zero, 10 - kept - zero);
}

TEST(rethrow_same_object) {
    cx_fill(&g_s, 16, M_RETHROW, 7);
    cx_run(&g_s, 0);
    cx_check_scn(&g_s, "after catch + throw; through 8 frames");
    CHECKF(g_s.obj_inner && g_s.obj_inner == g_s.obj_outer, "throw; rethrows the same object (%p vs %p)",
           g_s.obj_inner, g_s.obj_outer);
}

TEST(nested_try_in_catch) {
    cx_fill(&g_s, 17, M_NESTED, 7);
    uint64_t original = g_s.payload;
    cx_run(&g_s, 0);
    cx_check_scn(&g_s, "after nested try in a catch block and a replacing throw");
    CHECKF(g_s.nested_inner_ok, "int thrown through 4 frames inside the catch block was caught");
    CHECKF(g_s.nested_outer_intact, "the exception being handled stayed intact across the nested throw");
    CHECKF(g_s.caught_v != original, "outer catch saw the replacing exception");
}

TEST(exception_ptr_rethrow) {
    cx_fill(&g_s, 18, M_EPTR, 7);
    cx_run(&g_s, 0);
    cx_check_scn(&g_s, "after std::rethrow_exception through 4 more frames");
}

TEST(destructor_with_live_simd) {
    cx_fill(&g_s, 19, M_DTOR, 3);
    cx_run(&g_s, 0);
    cx_check_scn(&g_s, "after unwinding through a frame with a SIMD destructor");
    CHECK_EQ(g_s.dtor_runs, 1);
    CHECKF(g_s.dtor_uncaught == 1, "std::uncaught_exceptions() in the destructor = %d, want 1", g_s.dtor_uncaught);
    CHECKF((g_s.dtor_mxcsr & 0xFFC0) == (g_s.in.mxcsr & 0xFFC0), "MXCSR control in the destructor 0x%04x, want 0x%04x",
           g_s.dtor_mxcsr & 0xFFC0, g_s.in.mxcsr & 0xFFC0);
    CHECKF((g_s.dtor_fcw & 0x1F3F) == (g_s.in.fcw & 0x1F3F), "x87 CW in the destructor 0x%04x, want 0x%04x",
           g_s.dtor_fcw, g_s.in.fcw);
    CHECKF(g_s.dtor_vreg_runs >= 2, "register round trips ran in the destructor (%d)", g_s.dtor_vreg_runs);
    CHECKF(g_s.dtor_vreg_bad == 0, "%d register mismatches in XMM/YMM round trips inside the destructor", g_s.dtor_vreg_bad);
    CHECKF(g_s.live_bad_mid == 0, "%d of %d live vectors wrong in the rethrowing compiled frame", g_s.live_bad_mid, CX_NLIVE);
}

/* ---- thread pool --------------------------------------------------------------------------------- */
struct TpCtx { Scn *s; HANDLE done; DWORD tid; };
static VOID CALLBACK cx_tp_cb(PTP_CALLBACK_INSTANCE, PVOID p) {
    TpCtx *c = (TpCtx *)p;
    c->tid = GetCurrentThreadId();
    cx_run(c->s, 0);
    SetEvent(c->done);
}

TEST(threadpool_callback_throw) {
    static TpCtx c;
    cx_fill(&g_s, 20, M_DTOR, 7);
    c.s = &g_s; c.tid = 0;
    c.done = CreateEventA(NULL, TRUE, FALSE, NULL);
    REQUIRE(c.done != NULL);
    if (!TrySubmitThreadpoolCallback(cx_tp_cb, &c, NULL)) {
        CloseHandle(c.done);
        SKIP("TrySubmitThreadpoolCallback failed (error %lu)", GetLastError());
    }
    DWORD w = WaitForSingleObject(c.done, 5000);
    REQUIREF(w == WAIT_OBJECT_0, "thread-pool callback finished within 5 s (wait=%lu)", w);
    CloseHandle(c.done);
    CHECKF(c.tid != GetCurrentThreadId(), "callback ran on a pool thread");
    cx_check_scn(&g_s, "in a thread-pool callback");
    CHECK_EQ(g_s.dtor_runs, 1);
    CHECKF(g_s.dtor_vreg_bad == 0, "%d register mismatches inside the destructor", g_s.dtor_vreg_bad);
    CHECKF(g_s.live_bad_mid == 0, "%d live vectors wrong in the rethrowing compiled frame", g_s.live_bad_mid);
}

/* ---- fiber: throw on a fiber stack after switching away and back mid-chain ------------------------- */
static volatile int g_fiber_done;
static VOID CALLBACK cx_fiber_proc(PVOID p) {
    Scn *s = (Scn *)p;
    cx_run(s, 0);
    g_fiber_done = 1;
    SwitchToFiber(s->main_fiber);
    for (;;) SwitchToFiber(s->main_fiber);   /* never returns from a fiber proc */
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"   /* GetCurrentFiber() reads gs:[0x20] */
TEST(fiber_throw_across_switch) {
    int converted = 0;
    void *main_fiber;
    if (IsThreadAFiber()) main_fiber = GetCurrentFiber();
    else {
        main_fiber = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
        if (!main_fiber) SKIP("ConvertThreadToFiberEx failed (error %lu)", GetLastError());
        converted = 1;
    }
    cx_fill(&g_s, 21, M_FIBER, 7);
    g_s.main_fiber = main_fiber;
    g_fiber_done = 0;
    void *f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, cx_fiber_proc, &g_s);
    if (!f) { if (converted) ConvertFiberToThread(); SKIP("CreateFiberEx failed (error %lu)", GetLastError()); }
    int rounds = 0;
    while (!g_fiber_done && rounds < 4) {
        SwitchToFiber(f);
        rounds++;
        Sleep(0);   /* the main fiber does unrelated work (and system calls) between the switches */
        static t_vregs a, b; t_vregs_pattern(&a, T_VK_XMM, 99);
        t_vregs_call(T_VK_XMM, &a, &b, NULL, NULL);
    }
    DeleteFiber(f);
    if (converted) ConvertFiberToThread();
    CHECKF(g_fiber_done, "fiber finished (rounds=%d)", rounds);
    CHECK_EQ(g_s.fiber_switches, 2);
    cx_check_scn(&g_s, "on a fiber after SwitchToFiber away and back before the throw");
}
#pragma GCC diagnostic pop

/* ---- VEH translation of hardware faults (child processes) ---------------------------------------- */
static volatile LONG g_veh_calls;
static volatile uint64_t g_veh_addr, g_veh_code, g_veh_excaddr;
static LONG CALLBACK cx_veh(EXCEPTION_POINTERS *ep) {
    CONTEXT *c = ep->ContextRecord;
    EXCEPTION_RECORD *r = ep->ExceptionRecord;
    if (c->Rip != (DWORD64)cx_fault_av_insn && c->Rip != (DWORD64)cx_fault_de_insn) return EXCEPTION_CONTINUE_SEARCH;
    InterlockedIncrement(&g_veh_calls);
    uint64_t addr = (r->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r->NumberParameters >= 2)
                  ? (uint64_t)r->ExceptionInformation[1] : (uint64_t)r->ExceptionAddress;
    g_veh_code = r->ExceptionCode; g_veh_addr = addr; g_veh_excaddr = (uint64_t)r->ExceptionAddress;
    /* resume in the faulting frame (same RSP, prologue complete) at a call to the throwing translator */
    c->Rcx = r->ExceptionCode; c->Rdx = addr; c->Rip = (DWORD64)cx_fault_landing;
    return EXCEPTION_CONTINUE_EXECUTION;
}

struct FaultCtx { Scn *s; uint64_t kind; const void *addr; int caught; uint64_t code, addr_seen; int other; };
extern "C" void cx_fault_leaf(void *arg) {
    FaultCtx *f = (FaultCtx *)arg;
    cx_fault_frame(f->kind, f->addr, &f->s->chain2);
}
extern "C" void cx_fault_body(void *arg) {
    FaultCtx *f = (FaultCtx *)arg;
    Scn *s = f->s;
    LIVE_LOAD(s);
    try {
        cx_chain((uint64_t)s->depth, &s->chain);
    } catch (const HwFault &h) {
        f->caught = 1; f->code = h.code; f->addr_seen = h.addr;
        s->catch_mxcsr = t_get_mxcsr(); s->catch_fcw = t_get_x87_cw();
    } catch (...) {
        f->other = 1;
    }
    LIVE_BAD(s, s->live_bad_catch);
}

static int cx_regs_bad(const cx_regs *in, const cx_regs *out) {
    int bad = 0;
    for (int i = 6; i < 16; i++) if (memcmp(out->xmm[i], in->xmm[i], 16)) { bad++; printf("BAD=xmm%d\n", i); }
    for (int i = 0; i < 8; i++) if (out->gpr[i] != in->gpr[i]) { bad++; printf("BAD=%s got=%llx want=%llx\n", cx_gpr_names[i],
                                    (unsigned long long)out->gpr[i], (unsigned long long)in->gpr[i]); }
    if ((out->mxcsr & 0xFFC0) != (in->mxcsr & 0xFFC0)) { bad++; printf("BAD=mxcsr got=%x\n", out->mxcsr); }
    if ((out->fcw & 0x1F3F) != (in->fcw & 0x1F3F)) { bad++; printf("BAD=fcw got=%x\n", out->fcw); }
    return bad;
}

CHILD(veh_fault) {
    uint64_t kind = (argc > 1 && !strcmp(argv[1], "de")) ? 1 : 0;
    static Scn s;
    static FaultCtx f;
    cx_fill(&s, 22 + (uint32_t)kind, M_PLAIN, 3);
    s.chain.leaf = cx_fault_leaf; s.chain.leaf_arg = &f;
    void *base = NULL, *p = NULL;
    if (kind == 0) { p = t_guarded_alloc(64, &base); if (!p) { printf("SETUP_FAIL=guarded_alloc\n"); return 3; } }
    f.s = &s; f.kind = kind; f.addr = kind == 0 ? (const uint8_t *)p + 64 : NULL;
    if (!AddVectoredExceptionHandler(1, cx_veh)) { printf("SETUP_FAIL=AddVectoredExceptionHandler\n"); return 3; }
    cx_outer(&s.in, &s.out, cx_fault_body, &f);
    printf("VEH_CALLS=%ld\n", (long)g_veh_calls);
    printf("CAUGHT=%d OTHER=%d\n", f.caught, f.other);
    printf("CODE=0x%llx\n", (unsigned long long)f.code);
    printf("EXCADDR_OK=%d\n", g_veh_excaddr == (uint64_t)(kind ? (void *)cx_fault_de_insn : (void *)cx_fault_av_insn));
    printf("ADDR_OK=%d\n", kind ? 1 : (f.addr_seen == (uint64_t)f.addr));
    printf("CATCH_MXCSR=0x%x CATCH_FCW=0x%x\n", s.catch_mxcsr, s.catch_fcw);
    printf("LIVE_BAD=%d\n", s.live_bad_catch);
    printf("REG_BAD=%d\n", cx_regs_bad(&s.in, &s.out));
    fflush(stdout);
    return 0;
}

static void cx_veh_case(const char *arg, uint32_t want_code) {
    static t_child_result r;
    char v[128];
    REQUIRE(t_child_run("veh_fault", arg, 10000, &r));
    if (t_verbose()) t_log("%s", r.out);
    REQUIREF(!r.timed_out, "child timed out");
    REQUIREF(r.exit_code == 0, "child exit 0x%lx (exception 0x%lx)\n%s", r.exit_code, r.exc_code, r.out);
    int calls = -1, caught = -1, other = -1, n = -1; unsigned long long code = 0; unsigned mx = 0, cw = 0;
    if (t_child_find(&r, "VEH_CALLS=", v, sizeof v)) calls = atoi(v);
    if (t_child_find(&r, "CAUGHT=", v, sizeof v)) sscanf(v, "%d OTHER=%d", &caught, &other);
    if (t_child_find(&r, "CODE=", v, sizeof v)) code = strtoull(v, NULL, 16);
    CHECK_EQ(calls, 1);
    CHECKF(caught == 1 && other == 0, "catch(const HwFault&) reached (caught=%d other=%d)", caught, other);
    CHECK_EQ(code, want_code);
    CHECKF(t_child_find(&r, "EXCADDR_OK=", v, sizeof v) && atoi(v) == 1, "ExceptionAddress = the faulting instruction");
    CHECKF(t_child_find(&r, "ADDR_OK=", v, sizeof v) && atoi(v) == 1, "AV ExceptionInformation[1] = the guard-page address");
    if (t_child_find(&r, "CATCH_MXCSR=", v, sizeof v)) sscanf(v, "0x%x CATCH_FCW=0x%x", &mx, &cw);
    CHECKF((mx & 0xFFC0) == 0xBF80, "MXCSR control at the catch site 0x%04x, want 0xbf80", mx & 0xFFC0);
    CHECKF((cw & 0x1F3F) == (0x077F & 0x1F3F), "x87 CW at the catch site 0x%04x, want 0x077f", cw);
    if (t_child_find(&r, "LIVE_BAD=", v, sizeof v)) n = atoi(v);
    CHECKF(n == 0, "%d live vectors of the catching compiled frame changed", n);
    n = -1; if (t_child_find(&r, "REG_BAD=", v, sizeof v)) n = atoi(v);
    CHECKF(n == 0, "%d nonvolatile registers wrong after the catching function returned\n%s", n, r.out);
}

TEST(veh_translate_av_catch_all) { cx_veh_case("av", 0xC0000005u); }
TEST(veh_translate_int_divide)   { cx_veh_case("de", 0xC0000094u); }

/* ---- RtlUnwindEx called directly --------------------------------------------------------------------- */
#define CX_EXC_UNWINDING     0x02
#define CX_EXC_TARGET_UNWIND 0x20
struct UnwCtx {
    int edit;
    void *target;
    cx_regs in, out;
    cx_chain_ctx chain;
    uint64_t retval;
    int calls, target_calls; DWORD flags_or; void *frame_seen; DWORD code_seen;
    uint8_t mark5[16], mark6[16];
    CONTEXT scratch __attribute__((aligned(16)));
};
static UnwCtx g_u;

extern "C" EXCEPTION_DISPOSITION cx_unw_handler(PEXCEPTION_RECORD rec, PVOID frame, PCONTEXT ctx, PVOID dc) {
    (void)dc;
    g_u.calls++; g_u.flags_or |= rec->ExceptionFlags; g_u.frame_seen = frame; g_u.code_seen = rec->ExceptionCode;
    if (rec->ExceptionFlags & CX_EXC_TARGET_UNWIND) {
        g_u.target_calls++;
        if (g_u.edit) { memcpy(&ctx->Xmm5, g_u.mark5, 16); memcpy(&ctx->Xmm6, g_u.mark6, 16); }
    }
    return ExceptionContinueSearch;
}
extern "C" void cx_unw_leaf(void *arg) {
    UnwCtx *u = (UnwCtx *)arg;
    RtlUnwindEx(u->target, (PVOID)cx_unw_landing, NULL, (PVOID)(uintptr_t)u->retval, &u->scratch, NULL);
}
extern "C" void cx_unw_go(void *target, void *arg) {
    UnwCtx *u = (UnwCtx *)arg;
    u->target = target;
    cx_chain(7, &u->chain);
}
extern "C" void cx_unw_body(void *arg) {
    UnwCtx *u = (UnwCtx *)arg;
    cx_unw_target(&u->in, &u->out, cx_unw_go, u);
}

static void cx_unw_case(int edit, uint32_t tag) {
    static Scn s;   /* outer registers (cx_outer around everything) */
    cx_fill(&s, tag, M_PLAIN, 0);
    memset(&g_u, 0, sizeof g_u);
    g_u.edit = edit;
    for (int i = 0; i < 16; i++) t_pattern_reg(g_u.in.xmm[i], 16, tag + 100, i);
    for (int i = 0; i < 8; i++) g_u.in.gpr[i] = t_pattern_word(tag + 100, 16 + i, 0);
    for (int i = 0; i < 10; i++) t_pattern_reg(g_u.chain.clob[i], 16, tag + 200, 6 + i);
    for (int i = 0; i < 8; i++) g_u.chain.gclob[i] = t_pattern_word(tag + 200, 16 + i, 1);
    g_u.chain.leaf = cx_unw_leaf; g_u.chain.leaf_arg = &g_u;
    g_u.retval = t_pattern_word(tag, 61, 3);
    t_pattern_reg(g_u.mark5, 16, tag + 300, 5); t_pattern_reg(g_u.mark6, 16, tag + 300, 6);
    /* "edited" scratch ContextRecord: poison everywhere, a full-looking CONTEXT with foreign XMM6-15 */
    memset(&g_u.scratch, 0x5A, sizeof g_u.scratch);
    g_u.scratch.ContextFlags = CONTEXT_FULL;
    for (int i = 0; i < 10; i++) t_pattern_reg((uint8_t *)&g_u.scratch.Xmm6 + 16 * i, 16, tag + 400, 6 + i);
    g_u.scratch.MxCsr = 0x1F80;

    cx_outer(&s.in, &s.out, cx_unw_body, &g_u);

    CHECKF(g_u.out.rax == g_u.retval, "RAX at TargetIp = ReturnValue: got 0x%llx want 0x%llx",
           (unsigned long long)g_u.out.rax, (unsigned long long)g_u.retval);
    char lbl[64];
    for (int i = edit ? 7 : 6; i < 16; i++) {
        snprintf(lbl, sizeof lbl, "target frame xmm%d after RtlUnwindEx", i);
        CHECK_REG(lbl, g_u.out.xmm[i], g_u.in.xmm[i], 16);
    }
    for (int i = 0; i < 8; i++) {
        snprintf(lbl, sizeof lbl, "target frame %s after RtlUnwindEx", cx_gpr_names[i]);
        CHECK_REG(lbl, &g_u.out.gpr[i], &g_u.in.gpr[i], 8);
    }
    CHECKF((g_u.out.mxcsr & 0xFFC0) == 0xBF80, "MXCSR control at TargetIp 0x%04x, want 0xbf80", g_u.out.mxcsr & 0xFFC0);
    CHECKF((g_u.out.fcw & 0x1F3F) == (0x077F & 0x1F3F), "x87 CW at TargetIp 0x%04x, want 0x077f", g_u.out.fcw);
    CHECKF(g_u.calls >= 1, "the target frame's unwind handler was called (%d)", g_u.calls);
    CHECKF(g_u.flags_or & CX_EXC_UNWINDING, "handler flags include EXCEPTION_UNWINDING (0x%lx)", g_u.flags_or);
    CHECKF(g_u.frame_seen == g_u.target, "handler EstablisherFrame %p = TargetFrame %p", g_u.frame_seen, g_u.target);
    t_info("unwind_handler_flags", "0x%lx", g_u.flags_or);
    t_info("unwind_handler_code", "0x%lx", g_u.code_seen);
    t_info("target_unwind_flag", "%s", g_u.target_calls ? "set" : "not set");
    cx_check_abi(&s.in, &s.out, "after the function that called RtlUnwindEx returned");
    if (edit) {
        int e6 = !memcmp(g_u.out.xmm[6], g_u.mark6, 16), o6 = !memcmp(g_u.out.xmm[6], g_u.in.xmm[6], 16);
        int e5 = !memcmp(g_u.out.xmm[5], g_u.mark5, 16);
        t_info("target_handler_xmm6_edit", "%s", e6 ? "applied" : o6 ? "not applied (frame value)" : "other");
        t_info("target_handler_xmm5_edit", "%s", e5 ? "applied" : "not applied");
        if (!g_u.target_calls) t_note("handler never saw EXCEPTION_TARGET_UNWIND; context edit not attempted");
        else if (!e6) t_note("an XMM6 edit by the target frame's unwind handler did not reach TargetIp "
                             "(Windows 10 19041 RtlUnwindEx restores the handler's ContextRecord for the target frame)");
    }
}

TEST(rtlunwindex_direct_target_frame)  { cx_unw_case(0, 40); }
TEST(rtlunwindex_target_handler_edit)  { cx_unw_case(1, 41); }

#else  /* no SEH-based C++ on this target: every case SKIPs (keep in sync with the x64 TEST names above) */

#define CX_SKIP_CASE(n) TEST(n) { SKIP("%s", CX_X86_REASON); }
/* one per line: cases run in registration-line order */
CX_SKIP_CASE(throw_1_asm_frame_xmm_saved)
CX_SKIP_CASE(throw_8_asm_frames_xmm_saved)
CX_SKIP_CASE(throw_64_asm_frames_xmm_saved)
CX_SKIP_CASE(throw_64_compiled_simd_frames)
CX_SKIP_CASE(throw_8_frames_ymm_uppers_volatile)
CX_SKIP_CASE(rethrow_same_object)
CX_SKIP_CASE(nested_try_in_catch)
CX_SKIP_CASE(exception_ptr_rethrow)
CX_SKIP_CASE(destructor_with_live_simd)
CX_SKIP_CASE(threadpool_callback_throw)
CX_SKIP_CASE(fiber_throw_across_switch)
CX_SKIP_CASE(veh_translate_av_catch_all)
CX_SKIP_CASE(veh_translate_int_divide)
CX_SKIP_CASE(rtlunwindex_direct_target_frame)
CX_SKIP_CASE(rtlunwindex_target_handler_edit)

#endif
