/* apx_xstate - Intel APX extended GPRs (R16-R31, XSAVE component 19): enumeration, XSAVE layout, OS state
 * management and the Windows CONTEXT XSTATE path.
 *
 * Contract under test:
 *   - Enumeration (Intel APX Architecture Specification, 355828; SDM vol.1 ch.13): APX_F = CPUID.(7,1).EDX[21];
 *     the OS enables the state with XCR0[19]. Component 19 is a 128-byte user component (16 x 8 bytes,
 *     R16..R31 in order) at standard-format offset 960, which reuses the MPX (BNDREGS 3 / BNDCSR 4) area, so
 *     APX and MPX are never both enabled in XCR0. CPUID.(0Dh,19).ECX[0] (supervisor) = 0. NDD / NF / CCMP
 *     ("NCI") additionally need CPUID.(29h,0).EBX[0] (APX_NCI_NDD_NF); PUSH2/POP2 and REX2 are APX_F.
 *   - REX2 (0xD5) and the EVEX map-4 promoted forms #UD when APX is not enabled (no APX_F, or XCR0[19] = 0):
 *     in 64-bit mode 0xD5 is AAD, invalid in 64-bit mode, and 0x62 is BOUND (invalid) or an EVEX map that
 *     does not exist without APX. Probed in CHILD processes; when APX is enabled the same instructions must
 *     produce the modelled result.
 *   - XSAVE/XSAVEOPT/XSAVEC/XRSTOR save and restore R16..R31 through component 19; XRSTOR with
 *     XSTATE_BV[19] = 0 loads the init state (all zero).
 *   - OS state management: R16..R31 are ABI-volatile on Windows x64 (like XMM16-31), so across calls
 *     (Sleep, SwitchToThread, waits, fiber switches) only "no value from another thread appears" is
 *     hard-checked. Across preemption, SuspendThread/GetThreadContext from another thread, and hardware
 *     exceptions resumed with EXCEPTION_CONTINUE_EXECUTION the interrupted instruction stream must see its
 *     registers unchanged: hard-checked. Workers hold their values inside one asm routine (load, spin with
 *     GPR-only instructions, store); no compiled code runs in between.
 *   - Windows XSTATE context: GetEnabledXStateFeatures includes bit 19 when XCR0[19] is set (it reports the
 *     enabled XSTATE features). With CONTEXT_XSTATE and SetXStateFeaturesMask(1<<19), GetThreadContext
 *     reports R16..R31 at LocateXStateFeature(ctx, 19) (length 128) and SetThreadContext modifies them.
 *     Pinned against the Windows 10 19041 reference decompilation (docker1 corpus explorer, 2026-11):
 *       ntdll!RtlLocateExtendedFeature2: any feature 2..63 that is enabled (XState.EnabledFeatures |
 *         EnabledSupervisorFeatures) and present in the XSTATE chunk; *Length = XState.Features[i].Size,
 *         standard-format pointer = chunk + Features[i].Offset - 0x200 (no feature-specific code).
 *       ntdll!RtlSetExtendedFeaturesMask: stores mask & enabled & ~3 into the chunk's Mask.
 *     A context API that does not report feature 19 although XCR0[19] is set is a separate FAIL
 *     (apx_api_enabled_features); the context cases then SKIP with that reason.
 * x86 build: APX state exists only in 64-bit mode; only the enumeration case runs (CPUID is mode independent).
 * Instruction encodings are emitted as .byte (generated and cross-checked with llvm-mc 19) so older
 * assemblers build the file; the mnemonic is in the comment next to each.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> t_note/t_info.
 * Deliberately faulting operations run in CHILD() processes via t_child_run(). See common/API.md. */
#include "t.h"

/* mingw-w64 intrin-impl.h (__readgsqword behind GetCurrentFiber) trips -Warray-bounds */
#pragma GCC diagnostic ignored "-Warray-bounds"
#ifndef FIBER_FLAG_FLOAT_SWITCH
#define FIBER_FLAG_FLOAT_SWITCH 0x1
#endif
#ifndef ERROR_ALREADY_FIBER
#define ERROR_ALREADY_FIBER 1280L
#endif

#define APX_COMP     19
#define APX_MASK     (1ull << APX_COMP)
#define APX_STD_OFF  960u
#define APX_SIZE     128u
#define APX_NREG     16
#define APX_TMO      10000u
#define STATUS_ILLEGAL_INSTR 0xC000001Du

#ifndef CONTEXT_XSTATE
#ifdef T_X64
#define CONTEXT_XSTATE (CONTEXT_AMD64 | 0x40L)
#else
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)
#endif
#endif

/* Block shared with the asm routines (fixed offsets). */
typedef struct apx_blk {
    uint64_t in[APX_NREG];      /* 0:   R16..R31 to load */
    uint64_t out[APX_NREG];     /* 128: R16..R31 stored at the end */
    volatile LONG ready;        /* 256: set (locked) after the load */
    volatile LONG stop;         /* 260 */
    uint32_t max;               /* 264: spin iterations (must be > 0) */
    uint32_t rem;               /* 268: iterations left at exit */
} apx_blk;
_Static_assert(offsetof(apx_blk, out) == 128 && offsetof(apx_blk, ready) == 256 && offsetof(apx_blk, stop) == 260 &&
               offsetof(apx_blk, max) == 264 && offsetof(apx_blk, rem) == 268, "apx_blk layout");

/* CPUID.(29h,0).EBX[0]: NCI (CCMP/CTEST/CFCMOV), NDD, NF */
static int apx_nci(void) {
    t_cpuid_regs r;
    if (cpu()->max_basic < 0x29) return 0;
    cpuid_ex(0x29, 0, &r);
    return r.ebx & 1;
}

#ifdef T_X64
/* ======================================================================================================
 * asm (x64). APXLD/APXST: mov disp32(base), R(16+i) / mov R(16+i), disp32(base); base = legacy GPR 0..7
 * (not rsp/rbp). REX2 payload: R4 (0x40) | W (0x08) | R3 (0x04 when i >= 8); ModRM mod=10 reg=i&7 rm=base.
 * ====================================================================================================== */
#define APX_LDALL(off, base) "  .irp i,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n  APXLD \\i, " #off "+\\i*8, " #base "\n  .endr\n"
#define APX_STALL(off, base) "  .irp i,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n  APXST \\i, " #off "+\\i*8, " #base "\n  .endr\n"
#define APX_FN(name) ".globl " #name "\n.def " #name "; .scl 2; .type 32; .endef\n" #name ":\n"

uint32_t t__apx_spin(apx_blk *b);
void     t__apx_call(apx_blk *b, void (*fn)(void *), void *arg);
void     t__apx_xop(const uint64_t *in, void *area, uint64_t rfbm, uint64_t *out, int op);
void     t__apx_fault(apx_blk *b, int kind);
void     t__apx_store(uint64_t *out);
uint64_t t__apx_probe_rex2(uint64_t v);
uint64_t t__apx_probe_evex(uint64_t a, uint64_t b);
void     t__apx_model(uint64_t *io);
extern char t__apx_fault_ud[], t__apx_fault_av[], t__apx_fault_de[], t__apx_probe_rex2_insn[], t__apx_probe_evex_insn[];

__asm__(
    ".macro APXLD i, disp, base\n  .byte 0xd5, 0x48|((\\i&8)>>1), 0x8b, 0x80|((\\i&7)<<3)|\\base\n  .long \\disp\n.endm\n"
    ".macro APXST i, disp, base\n  .byte 0xd5, 0x48|((\\i&8)>>1), 0x89, 0x80|((\\i&7)<<3)|\\base\n  .long \\disp\n.endm\n"
    ".text\n"
    /* uint32_t t__apx_spin(apx_blk *b): leaf, no stack use */
    ".p2align 4\n" APX_FN(t__apx_spin)
    APX_LDALL(0, 1)
    "  lock orl $1, 256(%rcx)\n"
    "  movl 264(%rcx), %eax\n  testl %eax, %eax\n  jz 2f\n"
    "1: cmpl $0, 260(%rcx)\n  jne 2f\n  pause\n  decl %eax\n  jnz 1b\n"
    "2: movl %eax, 268(%rcx)\n"
    APX_STALL(128, 1)
    "  ret\n"
    /* void t__apx_call(apx_blk *b, fn, arg): load, fn(arg), store */
    ".p2align 4\n" ".globl t__apx_call\n.def t__apx_call; .scl 2; .type 32; .endef\n.seh_proc t__apx_call\nt__apx_call:\n"
    "  pushq %rbx\n  .seh_pushreg %rbx\n  subq $32, %rsp\n  .seh_stackalloc 32\n  .seh_endprologue\n"
    "  movq %rcx, %rbx\n"
    APX_LDALL(0, 3)
    "  movq %r8, %rcx\n  callq *%rdx\n"
    APX_STALL(128, 3)
    "  addq $32, %rsp\n  popq %rbx\n  ret\n.seh_endproc\n"
    /* void t__apx_xop(in, area, rfbm, out, op): [load in]; op 0 XSAVE64 / 1 XSAVEC64 / 2 XRSTOR64 /
     * 3 XSAVEOPT64 on area with rfbm; [store out]. Leaf. */
    ".p2align 4\n" APX_FN(t__apx_xop)
    "  movq %rdx, %r10\n  movq %r8, %rax\n  movq %r8, %rdx\n  shrq $32, %rdx\n  movl 40(%rsp), %r11d\n"
    "  testq %rcx, %rcx\n  jz 1f\n"
    APX_LDALL(0, 1)
    "1: cmpl $1, %r11d\n  jne 2f\n  xsavec64 (%r10)\n  jmp 5f\n"
    "2: cmpl $2, %r11d\n  jne 3f\n  xrstor64 (%r10)\n  jmp 5f\n"
    "3: cmpl $3, %r11d\n  jne 4f\n  xsaveopt64 (%r10)\n  jmp 5f\n"
    "4: xsave64 (%r10)\n"
    "5: testq %r9, %r9\n  jz 6f\n  movq %r9, %rcx\n"
    APX_STALL(0, 1)
    "6: ret\n"
    /* void t__apx_fault(apx_blk *b, int kind): load, fault (0 UD2, 1 load from NULL, 2 DIV by 0), store.
     * The test's VEH resumes after the faulting instruction. Leaf. */
    ".p2align 4\n" APX_FN(t__apx_fault)
    APX_LDALL(0, 1)
    "  cmpl $1, %edx\n  je 1f\n  cmpl $2, %edx\n  je 2f\n"
    ".globl t__apx_fault_ud\nt__apx_fault_ud:\n  ud2\n  jmp 9f\n"
    "1: xorl %eax, %eax\n"
    ".globl t__apx_fault_av\nt__apx_fault_av:\n  movq (%rax), %rax\n  jmp 9f\n"       /* 48 8b 00 */
    "2: xorl %r8d, %r8d\n  xorl %edx, %edx\n  movl $1, %eax\n"
    ".globl t__apx_fault_de\nt__apx_fault_de:\n  divq %r8\n"                       /* 49 f7 f0 */
    "9:\n"
    APX_STALL(128, 1)
    "  ret\n"
    /* void t__apx_store(uint64_t *out): store R16..R31 as found */
    ".p2align 4\n" APX_FN(t__apx_store)
    APX_STALL(0, 1)
    "  ret\n"
    /* uint64_t t__apx_probe_rex2(v): v -> R16 -> RAX through REX2 moves */
    ".p2align 4\n" APX_FN(t__apx_probe_rex2)
    "  movq %rcx, %rax\n"
    ".globl t__apx_probe_rex2_insn\nt__apx_probe_rex2_insn:\n"
    "  .byte 0xd5,0x18,0x89,0xc0\n"               /* movq %rax, %r16 */
    "  xorl %eax, %eax\n"
    "  .byte 0xd5,0x48,0x89,0xc0\n"               /* movq %r16, %rax */
    "  ret\n"
    /* uint64_t t__apx_probe_evex(a, b): EVEX map-4 NDD ADD into R18, then R18 -> RAX */
    ".p2align 4\n" APX_FN(t__apx_probe_evex)
    "  xorl %eax, %eax\n"
    ".globl t__apx_probe_evex_insn\nt__apx_probe_evex_insn:\n"
    "  .byte 0x62,0xf4,0xec,0x10,0x01,0xd1\n"     /* addq %rdx, %rcx, %r18 */
    "  .byte 0xd5,0x48,0x89,0xd0\n"               /* movq %r18, %rax */
    "  ret\n"
    /* void t__apx_model(uint64_t *io): see apx_model_expect(). Child only (changes RSP without unwind info). */
    ".p2align 4\n" APX_FN(t__apx_model)
    "  .byte 0xd5,0x48,0x8b,0x01\n"               /* movq 0(%rcx), %r16      (A) */
    "  .byte 0xd5,0x48,0x8b,0x49,0x08\n"          /* movq 8(%rcx), %r17      (B) */
    "  movq %rsp, %r11\n  andq $-16, %rsp\n"     /* PUSH2/POP2 need a 16-byte aligned RSP */
    "  .byte 0x62,0xfc,0x74,0x10,0xff,0xf0\n"     /* push2 %r16, %r17  (Intel PUSH2 r17, r16: r17 first) */
    "  movq (%rsp), %rax\n  movq %rax, 16(%rcx)\n"
    "  movq 8(%rsp), %rax\n  movq %rax, 24(%rcx)\n"
    "  .byte 0x62,0xfc,0x6c,0x10,0x8f,0xc3\n"     /* pop2 %r19, %r18  (Intel POP2 r18, r19: r18 = [rsp]) */
    "  movq %r11, %rsp\n"
    "  .byte 0xd5,0x48,0x89,0x51,0x20\n"          /* movq %r18, 32(%rcx) */
    "  .byte 0xd5,0x48,0x89,0x59,0x28\n"          /* movq %r19, 40(%rcx) */
    "  cmpq $0, 120(%rcx)\n  je 9f\n"             /* io[15] = NCI/NDD/NF enumerated */
    "  movq 0(%rcx), %r8\n  movq 8(%rcx), %r9\n"
    "  .byte 0x62,0x54,0xdc,0x10,0x01,0xc1\n"     /* addq %r8, %r9, %r20   (r20 = r9 + r8, flags) */
    "  setc %al\n  movzbl %al, %eax\n  movq %rax, 56(%rcx)\n"
    "  .byte 0xd5,0x48,0x89,0x61,0x30\n"          /* movq %r20, 48(%rcx) */
    "  stc\n"
    "  .byte 0x62,0xec,0xd4,0x14,0x01,0xc1\n"     /* {nf} addq %r16, %r17, %r21 */
    "  setc %al\n  movzbl %al, %eax\n  movq %rax, 72(%rcx)\n"
    "  .byte 0xd5,0x48,0x89,0x69,0x40\n"          /* movq %r21, 64(%rcx) */
    "  clc\n"
    "  .byte 0x62,0xec,0xcc,0x14,0x01,0xc1\n"     /* {nf} addq %r16, %r17, %r22 */
    "  setc %al\n  movzbl %al, %eax\n  movq %rax, 96(%rcx)\n"
    "  .byte 0xd5,0x48,0x89,0x71,0x68\n"          /* movq %r22, 104(%rcx) */
    "  xorl %eax, %eax\n  cmpq %rax, %rax\n"      /* ZF = 1: condition E true */
    "  .byte 0x62,0xec,0xd4,0x04,0x39,0xc1\n"     /* ccmpeq {dfv=of,zf} %r16, %r17 */
    "  pushfq\n  popq %rax\n  movq %rax, 80(%rcx)\n"
    "  movl $1, %eax\n  cmpq $2, %rax\n"          /* ZF = 0: condition false -> flags = dfv */
    "  .byte 0x62,0xec,0xd4,0x04,0x39,0xc1\n"     /* ccmpeq {dfv=of,zf} %r16, %r17 */
    "  pushfq\n  popq %rax\n  movq %rax, 88(%rcx)\n"
    "9: ret\n"
);
#endif /* T_X64 */

/* ======================================================================================================
 * Helpers
 * ====================================================================================================== */
typedef DWORD64 (WINAPI *pfn_gtexf)(void);               /* GetThreadEnabledXStateFeatures (Win11) */
typedef BOOL    (WINAPI *pfn_epoxf)(DWORD64);            /* EnableProcessOptionalXStateFeatures (Win11) */

static void apx_fill(uint64_t *r, uint32_t tag) {
    for (int i = 0; i < APX_NREG; i++) r[i] = t_pattern_word(tag, 16 + i, 0);
}

/* exact comparison (non-volatile situation) */
static int apx_check_exact(const uint64_t *got, const uint64_t *want, const char *what) {
    int bad = 0;
    char lbl[96];
    for (int i = 0; i < APX_NREG; i++) {
        snprintf(lbl, sizeof lbl, "r%d %s", 16 + i, what);
        if (!CHECK_REG(lbl, &got[i], &want[i], 8)) bad++;
    }
    return bad;
}

/* ABI-volatile situation: a value may be lost, but never replaced by another thread's value.
 * Foreign = a pattern word of any tag in [0, max_tag) other than the own tags [own_lo, own_hi]. */
static int apx_check_no_foreign(const uint64_t *got, const uint64_t *want, uint32_t own_lo, uint32_t own_hi,
                                uint32_t max_tag, const char *what) {
    int lost = 0;
    for (int i = 0; i < APX_NREG; i++) {
        if (got[i] == want[i]) continue;
        uint32_t tag, reg, word;
        int f = got[i] && t_pattern_explain(got[i], max_tag, &tag, &reg, &word) && (tag < own_lo || tag > own_hi);
        CHECKF(!f, "r%d %s holds 0x%016llx = another thread's value (tag %u reg %u)", 16 + i, what,
               (unsigned long long)got[i], f ? tag : 0, f ? reg : 0);
        lost++;
    }
    if (lost) t_note("%s: %d of 16 EGPRs changed (ABI-volatile; allowed)", what, lost);
    return lost;
}

static void apx_require(void) {
#ifndef T_X64
    SKIP("APX extended GPRs (R16-R31) exist only in 64-bit mode (x64 build)");
#else
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_APX_F)) SKIP("CPUID.(7,1).EDX[21] APX_F not set");
    if (!c->osxsave || !(c->xcr0 & APX_MASK))
        SKIP("APX_F advertised but XCR0[19] not enabled by the OS (osxsave=%d xcr0=0x%llx)", c->osxsave,
             (unsigned long long)c->xcr0);
    /* Windows 11: an XFD-capable component may be an optional per-process feature */
    pfn_gtexf gte = (pfn_gtexf)t_proc("kernel32.dll", "GetThreadEnabledXStateFeatures");
    if (gte && !(gte() & APX_MASK)) {
        pfn_epoxf ep = (pfn_epoxf)t_proc("kernel32.dll", "EnableProcessOptionalXStateFeatures");
        BOOL ok = ep ? ep(APX_MASK) : FALSE;
        t_note("GetThreadEnabledXStateFeatures lacked bit 19; EnableProcessOptionalXStateFeatures(1<<19) -> %d", ok);
        if (!(gte() & APX_MASK)) SKIP("XSTATE feature 19 not enabled for this thread (GetThreadEnabledXStateFeatures=0x%llx)",
                                      (unsigned long long)gte());
    }
#endif
}

/* ======================================================================================================
 * Enumeration and XCR0 consistency (both archs)
 * ====================================================================================================== */
TEST(apx_cpuid_layout) {
    const t_cpu_info *c = cpu();
    int hw = cpu_hw(F_APX_F);
    int en = c->osxsave && (c->xcr0 & APX_MASK);
    t_info("apx_f", "%d", hw);
    t_info("xcr0_bit19", "%d", en);
    t_info("xcr0_supported_bit19", "%d", (int)((c->xcr0_supported >> APX_COMP) & 1));
    t_info("apx_nci_ndd_nf", "%d", apx_nci());
    if (c->max_basic >= 0x29) {
        t_cpuid_regs r; cpuid_ex(0x29, 0, &r);
        t_info("cpuid_29_0", "eax=%08x ebx=%08x ecx=%08x edx=%08x", r.eax, r.ebx, r.ecx, r.edx);
    }
    if (en) CHECKF(hw, "XCR0[19] enabled but CPUID.(7,1).EDX[21] APX_F clear");
    if (hw) CHECKF(c->xcr0_supported & APX_MASK, "APX_F advertised but CPUID.(0Dh,0).EAX[19] (component 19) clear");
    if (en) CHECKF(c->xcr0_supported & APX_MASK, "XCR0[19] set although CPUID.(0Dh,0) does not support component 19");
    if (!(c->xcr0_supported & APX_MASK)) SKIP("XSAVE component 19 (APX) not supported (apx_f=%d)", hw);
    t_cpuid_regs d; cpuid_ex(0xD, APX_COMP, &d);
    t_info("cpuid_d_19", "eax=%u ebx=%u ecx=%08x", d.eax, d.ebx, d.ecx);
    CHECK_EQ(d.eax, APX_SIZE);                 /* 16 x 8 bytes */
    CHECK_EQ(d.ebx, APX_STD_OFF);              /* reuses the BNDREGS/BNDCSR area */
    CHECKF(!(d.ecx & 1), "component 19 reported as supervisor (ECX[0]) - it is user state managed by XCR0");
    t_info("comp19_align64", "%u", (d.ecx >> 1) & 1);
    t_info("comp19_xfd", "%u", (d.ecx >> 2) & 1);
    if (en) {
        CHECKF(!(c->xcr0 & 0x18), "XCR0[19] (APX) and XCR0[4:3] (MPX) both set: they share the XSAVE area (xcr0=0x%llx)",
               (unsigned long long)c->xcr0);
        CHECKF(c->xsave_size_enabled >= APX_STD_OFF + APX_SIZE, "CPUID.(0Dh,0).EBX=%u < 1088 with XCR0[19] set",
               c->xsave_size_enabled);
        CHECKF(c->xsave_size_max >= APX_STD_OFF + APX_SIZE, "CPUID.(0Dh,0).ECX=%u < 1088", c->xsave_size_max);
    }
}

#ifdef T_X64
/* ======================================================================================================
 * Windows API reporting
 * ====================================================================================================== */
static int apx_api_reports(void) {
    const t_env *e = t_env_get();
    return e->api_xstate_present && (e->api_xstate_features & APX_MASK);
}

TEST(apx_api_enabled_features) {
    apx_require();
    const t_env *e = t_env_get();
    if (!e->api_xstate_present) SKIP("GetEnabledXStateFeatures not exported");
    t_info("GetEnabledXStateFeatures", "0x%llx", e->api_xstate_features);
    CHECKF(e->api_xstate_features & APX_MASK, "XCR0[19] is set but GetEnabledXStateFeatures()=0x%llx lacks bit 19",
           e->api_xstate_features);
}

/* ======================================================================================================
 * #UD when disabled / model when enabled (children)
 * ====================================================================================================== */
CHILD(apx_probe_rex2) {
    printf("ADDR=%p\n", (void *)t__apx_probe_rex2_insn); fflush(stdout);
    printf("VAL=%016llx\n", (unsigned long long)t__apx_probe_rex2(0x0123456789ABCDEFull)); fflush(stdout);
    return 0;
}
CHILD(apx_probe_evex) {
    printf("ADDR=%p\n", (void *)t__apx_probe_evex_insn); fflush(stdout);
    printf("VAL=%016llx\n", (unsigned long long)t__apx_probe_evex(0x1111111111111111ull, 0x2222222222222223ull)); fflush(stdout);
    return 0;
}

static void apx_probe_case(const char *child, unsigned long long want) {
    static t_child_result r;
    char v[64];
    int enabled = cpu_has(F_APX_F);
    REQUIRE(t_child_run(child, NULL, APX_TMO, &r));
    REQUIRE(!r.timed_out);
    if (!enabled) {
        CHECKF(r.exc_code == STATUS_ILLEGAL_INSTR, "APX not enabled (apx_f=%d xcr0=0x%llx): expected #UD, got exit 0x%lx exc 0x%lx\n%s",
               cpu_hw(F_APX_F), (unsigned long long)cpu()->xcr0, (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
        if (t_child_find(&r, "ADDR=", v, sizeof v)) {
            unsigned long long a = strtoull(v, NULL, 16);
            CHECKF(r.exc_addr == a, "#UD reported at 0x%llx, instruction at 0x%llx", r.exc_addr, a);
        }
    } else {
        CHECKF(r.exit_code == 0 && r.exc_code == 0, "APX enabled: probe faulted (exit 0x%lx exc 0x%lx)\n%s",
               (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
        REQUIRE(t_child_find(&r, "VAL=", v, sizeof v));
        CHECK_EQ(strtoull(v, NULL, 16), want);
    }
}

TEST(apx_rex2_ud_or_model) { apx_probe_case("apx_probe_rex2", 0x0123456789ABCDEFull); }
TEST(apx_evex_egpr_ud_or_model) {
    if (cpu_has(F_APX_F) && !apx_nci()) SKIP("APX enabled but CPUID.(29h).EBX[0] (NDD) clear: NDD ADD not defined");
    apx_probe_case("apx_probe_evex", 0x3333333333333334ull);
}

CHILD(apx_model) {
    uint64_t io[16] = { 0 };
    io[0] = 0xA000000000000456ull; io[1] = 0x9000000000000123ull;
    io[15] = (uint64_t)apx_nci();
    t__apx_model(io);
    for (int i = 0; i < 16; i++) printf("IO%02d=%016llx\n", i, (unsigned long long)io[i]);
    fflush(stdout);
    return 0;
}

TEST(apx_model_push2_pop2_ndd_ccmp) {
    apx_require();
    static t_child_result r;
    REQUIRE(t_child_run("apx_model", NULL, APX_TMO, &r));
    REQUIRE(!r.timed_out);
    REQUIREF(r.exit_code == 0 && r.exc_code == 0, "model child faulted: exit 0x%lx exc 0x%lx\n%s",
             (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    uint64_t io[16];
    for (int i = 0; i < 16; i++) {
        char k[8], v[40];
        snprintf(k, sizeof k, "IO%02d=", i);
        REQUIRE(t_child_find(&r, k, v, sizeof v));
        io[i] = strtoull(v, NULL, 16);
    }
    uint64_t A = io[0], B = io[1];
    /* PUSH2 r17(B), r16(A): B pushed first -> [rsp] = A, [rsp+8] = B; POP2 r18, r19: r18 = [rsp] */
    CHECK_EQ(io[2], A); CHECK_EQ(io[3], B);
    CHECK_EQ(io[4], A); CHECK_EQ(io[5], B);
    if (!io[15]) { t_note("CPUID.(29h).EBX[0] clear: NDD/NF/CCMP not checked"); return; }
    uint64_t sum = A + B;
    CHECK_EQ(io[6], sum);  CHECK_EQ(io[7], sum < A);     /* NDD ADD sets CF */
    CHECK_EQ(io[8], sum);  CHECK_EQ(io[9], 1);           /* {nf}: CF from STC unchanged */
    CHECK_EQ(io[13], sum); CHECK_EQ(io[12], 0);          /* {nf}: CF from CLC unchanged although the add carries */
    /* CCMPE true: flags of r17 - r16 = B - A */
    uint64_t d = B - A;
    unsigned f = (B < A ? 0x1u : 0) | (d == 0 ? 0x40u : 0) | ((d >> 63) ? 0x80u : 0) |
                 ((((B ^ A) & (B ^ d)) >> 63) ? 0x800u : 0);
    CHECK_EQ(io[10] & 0x8C1u, f);
    CHECK_EQ(io[11] & 0x8C1u, 0x840u);                   /* false: OF=1 ZF=1 SF=0 CF=0 from dfv */
}

/* ======================================================================================================
 * XSAVE / XSAVEOPT / XSAVEC / XRSTOR round trip of component 19
 * ====================================================================================================== */
TEST(apx_xsave_xrstor_roundtrip) {
    apx_require();
    REQUIRE(cpu_has(F_XSAVE));
    const t_cpu_info *c = cpu();
    uint64_t xcr0 = c->xcr0;
    size_t sz = xsave_std_size(xcr0) + 4096;
    uint8_t *raw = calloc(1, sz + 64), *area = (uint8_t *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
    REQUIRE(raw);
    uint64_t in[16], in2[16], out[16], zero[16] = { 0 };
    apx_fill(in, 0x5101); apx_fill(in2, 0x5102);

    /* XSAVE with RFBM = 19 only and with RFBM = XCR0 */
    uint64_t rfbms[2] = { APX_MASK, xcr0 };
    for (int k = 0; k < 2; k++) {
        memset(area, 0, sz);
        t__apx_xop(in, area, rfbms[k], NULL, 0);
        uint64_t bv; memcpy(&bv, area + 512, 8);
        CHECKF(bv & APX_MASK, "XSAVE rfbm=0x%llx: XSTATE_BV=0x%llx lacks bit 19 with nonzero R16-R31", (unsigned long long)rfbms[k],
               (unsigned long long)bv);
        apx_check_exact((const uint64_t *)(area + APX_STD_OFF), in, k ? "XSAVE(rfbm=XCR0) image" : "XSAVE(rfbm=1<<19) image");
    }
    /* XRSTOR of a modified image */
    memcpy(area + APX_STD_OFF, in2, APX_SIZE);
    t__apx_xop(in, area, APX_MASK, out, 2);
    apx_check_exact(out, in2, "after XRSTOR of a modified image");
    /* XRSTOR with XSTATE_BV[19] = 0 -> init state (zero) */
    uint64_t bv; memcpy(&bv, area + 512, 8); bv &= ~APX_MASK; memcpy(area + 512, &bv, 8);
    t__apx_xop(in, area, APX_MASK, out, 2);
    apx_check_exact(out, zero, "after XRSTOR with XSTATE_BV[19]=0 (init)");
    /* RFBM without bit 19: XRSTOR leaves R16-R31 alone */
    memcpy(area + APX_STD_OFF, in2, APX_SIZE); bv |= APX_MASK; memcpy(area + 512, &bv, 8);
    /* RFBM = x87 only: restoring the snapshot's SSE/AVX images here would overwrite the caller's
     * callee-saved XMM6-15 with stale values */
    t__apx_xop(in, area, 0x1, out, 2);
    apx_check_exact(out, in, "after XRSTOR with RFBM[19]=0");

    if (cpu_has(F_XSAVEOPT)) {
        memset(area, 0, sz);
        t__apx_xop(in, area, xcr0, NULL, 0);           /* XSAVE first: XSAVEOPT may skip unmodified state */
        t__apx_xop(in2, area, xcr0, NULL, 3);
        apx_check_exact((const uint64_t *)(area + APX_STD_OFF), in2, "XSAVEOPT image");
    } else t_note("XSAVEOPT not usable: skipped");

    if (cpu_has(F_XSAVEC)) {
        uint64_t masks[2] = { APX_MASK, xcr0 };
        for (int k = 0; k < 2; k++) {
            memset(area, 0, sz);
            t__apx_xop(in, area, masks[k], NULL, 1);
            uint64_t xbv, xcomp; memcpy(&xbv, area + 512, 8); memcpy(&xcomp, area + 520, 8);
            CHECKF(xcomp == ((1ull << 63) | (masks[k] & xcr0)), "XSAVEC XCOMP_BV=0x%llx want 0x%llx", (unsigned long long)xcomp,
                   (unsigned long long)((1ull << 63) | (masks[k] & xcr0)));
            CHECKF(xbv & APX_MASK, "XSAVEC XSTATE_BV=0x%llx lacks bit 19", (unsigned long long)xbv);
            uint32_t off = xsave_compacted_offset(masks[k], APX_COMP);
            t_info(k ? "xsavec_off19_xcr0" : "xsavec_off19_only", "%u", off);
            apx_check_exact((const uint64_t *)(area + off), in, k ? "XSAVEC(rfbm=XCR0) image" : "XSAVEC(rfbm=1<<19) image");
            memcpy(area + off, in2, APX_SIZE);
            t__apx_xop(in, area, APX_MASK, out, 2);   /* RFBM subset of XCOMP_BV; only component 19 */
            apx_check_exact(out, in2, "after XRSTOR (compacted) of a modified image");
        }
    } else t_note("XSAVEC not usable: skipped");
    free(raw);
}

/* ======================================================================================================
 * Background spinners: other threads holding their own EGPR patterns (foreign-value sources)
 * ====================================================================================================== */
typedef struct apx_bg { apx_blk b; uint32_t tag; HANDLE th; } apx_bg;

static DWORD WINAPI apx_spin_thread(void *p) {
    apx_bg *g = p;
    t__apx_spin(&g->b);
    return 0;
}

static int apx_bg_start(apx_bg *g, int n, uint32_t tag0, uint32_t max) {
    int k = 0;
    for (; k < n; k++) {
        memset(&g[k].b, 0, sizeof g[k].b);
        g[k].tag = tag0 + k;
        apx_fill(g[k].b.in, g[k].tag);
        g[k].b.max = max;
        g[k].th = CreateThread(NULL, 64 * 1024, apx_spin_thread, &g[k], STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (!g[k].th) break;
    }
    DWORD t0 = GetTickCount();
    for (int i = 0; i < k; i++)
        while (!g[i].b.ready && GetTickCount() - t0 < 5000) Sleep(1);
    return k;
}

/* stop, join, exact-check every spinner (spinning = preemption only) */
static void apx_bg_stop(apx_bg *g, int n, const char *what) {
    for (int i = 0; i < n; i++) InterlockedExchange(&g[i].b.stop, 1);
    for (int i = 0; i < n; i++) {
        DWORD w = WaitForSingleObject(g[i].th, 5000);
        CHECKF(w == WAIT_OBJECT_0, "spinner %d did not finish", i);
        CloseHandle(g[i].th);
        if (w != WAIT_OBJECT_0) continue;
        CHECKF(g[i].b.ready, "spinner %d never ran", i);
        char lbl[96]; snprintf(lbl, sizeof lbl, "%s (spinner %d)", what, i);
        apx_check_exact(g[i].b.out, g[i].b.in, lbl);
    }
}

#define APX_BG_TAG 0x100u
#define APX_MAXTAG 0x600u

/* ======================================================================================================
 * Preemption, thread isolation, SuspendThread / GetThreadContext without XSTATE
 * ====================================================================================================== */
TEST(apx_preemption_thread_isolation) {
    apx_require();
    static const int counts[] = { 2, 16, 64, 256 };
    int ncpu = t_cpu_count();
    for (size_t ci = 0; ci < T_COUNTOF(counts); ci++) {
        int n = counts[ci];
        apx_bg *g = calloc((size_t)n, sizeof *g);
        REQUIRE(g);
        int got = apx_bg_start(g, n, APX_BG_TAG, 0xFFFFFFFFu);
        if (got < n) t_note("only %d of %d threads created", got, n);
        /* oversubscribe: time slicing happens when n > ncpu */
        DWORD t0 = GetTickCount();
        while (GetTickCount() - t0 < 150) SwitchToThread();
        char what[64]; snprintf(what, sizeof what, "after preemption with %d threads (%d CPUs)", got, ncpu);
        apx_bg_stop(g, got, what);
        free(g);
        if (got < 2 && ci == 0) SKIP("could not create worker threads");
        if (got < n) break;
    }
}

TEST(apx_suspend_getcontext_preserves) {
    apx_require();
    apx_bg g[2];
    int got = apx_bg_start(g, 2, APX_BG_TAG, 0xFFFFFFFFu);
    if (got < 2) { apx_bg_stop(g, got, "cleanup"); SKIP("thread creation failed"); }
    CONTEXT ctx;
    for (int i = 0; i < 200; i++) {
        HANDLE th = g[i & 1].th;
        CHECK(SuspendThread(th) != (DWORD)-1);
        memset(&ctx, 0, sizeof ctx);
        ctx.ContextFlags = CONTEXT_FULL;
        CHECK(GetThreadContext(th, &ctx));
        if (i % 3 == 0) CHECK(SetThreadContext(th, &ctx));   /* round trip without XSTATE must not touch R16-R31 */
        CHECK(ResumeThread(th) != (DWORD)-1);
        if (i % 16 == 0) Sleep(1);
    }
    apx_bg_stop(g, 2, "after SuspendThread/GetThreadContext/SetThreadContext(CONTEXT_FULL)");
}

/* ======================================================================================================
 * Calls (ABI-volatile): Sleep, SwitchToThread, waits -> no foreign values
 * ====================================================================================================== */
static HANDLE apx_evt;
static void apx_fn_sleep0(void *a) { (void)a; for (int i = 0; i < 5; i++) Sleep(0); }
static void apx_fn_sleep1(void *a) { (void)a; Sleep(2); }
static void apx_fn_yield(void *a) { (void)a; for (int i = 0; i < 5; i++) SwitchToThread(); }
static void apx_fn_wait(void *a) { (void)a; WaitForSingleObject(apx_evt, 5); }

TEST(apx_blocking_calls_no_foreign) {
    apx_require();
    /* spinners on all but one CPU (at least 2): the calls still block and switch, without starving this thread */
    int n = t_cpu_count() - 1; if (n < 2) n = 2; if (n > 16) n = 16;
    apx_bg *g = calloc((size_t)n, sizeof *g);
    REQUIRE(g);
    int got = apx_bg_start(g, n, APX_BG_TAG, 0xFFFFFFFFu);
    apx_evt = CreateEventA(NULL, TRUE, FALSE, NULL);
    struct { const char *name; void (*fn)(void *); } ops[] = {
        { "Sleep(0)x5", apx_fn_sleep0 }, { "Sleep(2)", apx_fn_sleep1 }, { "SwitchToThread x5", apx_fn_yield },
        { "WaitForSingleObject(5ms timeout)", apx_fn_wait },
    };
    int lost = 0;
    for (int rep = 0; rep < 20; rep++)
        for (size_t k = 0; k < T_COUNTOF(ops); k++) {
            apx_blk b; memset(&b, 0, sizeof b);
            apx_fill(b.in, 0x10 + (uint32_t)k);
            t__apx_call(&b, ops[k].fn, NULL);
            char what[96]; snprintf(what, sizeof what, "across %s", ops[k].name);
            lost += apx_check_no_foreign(b.out, b.in, 0, 0xFF, APX_MAXTAG, what) != 0;
        }
    t_info("calls_with_changed_egprs", "%d of %d", lost, 20 * (int)T_COUNTOF(ops));
    CloseHandle(apx_evt);
    apx_bg_stop(g, got, "spinners during blocking calls");
    free(g);
}

/* ======================================================================================================
 * Fibers (SwitchToFiber is a call: volatile) -> no foreign values
 * ====================================================================================================== */
static void *apx_main_fiber, *apx_other_fiber;
static void apx_fn_to_main(void *a) { (void)a; SwitchToFiber(apx_main_fiber); }
static void apx_fn_to_other(void *a) { (void)a; SwitchToFiber(apx_other_fiber); }
static apx_blk apx_fb_b;
static void WINAPI apx_fiber_proc(void *p) {
    (void)p;
    for (;;) {
        memset(&apx_fb_b, 0, sizeof apx_fb_b);
        apx_fill(apx_fb_b.in, 0x21);
        t__apx_call(&apx_fb_b, apx_fn_to_main, NULL);
    }
}

TEST(apx_fibers_no_foreign) {
    apx_require();
    apx_bg g[4];
    int got = apx_bg_start(g, 4, APX_BG_TAG, 0xFFFFFFFFu);
    int conv = 0;
    apx_main_fiber = ConvertThreadToFiber(NULL);
    if (!apx_main_fiber && GetLastError() == ERROR_ALREADY_FIBER) apx_main_fiber = GetCurrentFiber();
    else conv = 1;
    if (!apx_main_fiber) { apx_bg_stop(g, got, "cleanup"); SKIP("ConvertThreadToFiber failed (%lu)", GetLastError()); }
    static const DWORD flags[2] = { 0, FIBER_FLAG_FLOAT_SWITCH };
    for (int f = 0; f < 2; f++) {
        apx_other_fiber = CreateFiberEx(0, 0, flags[f], apx_fiber_proc, NULL);
        if (!apx_other_fiber) { t_note("CreateFiberEx(flags=%lu) failed", flags[f]); continue; }
        for (int i = 0; i < 50; i++) {
            apx_blk a; memset(&a, 0, sizeof a);
            apx_fill(a.in, 0x20);
            t__apx_call(&a, apx_fn_to_other, NULL);
            char what[64]; snprintf(what, sizeof what, "main fiber across SwitchToFiber (flags=%lu)", flags[f]);
            /* own tags: 0x20 (this fiber) and 0x21 (the other fiber on this thread) */
            if (apx_check_no_foreign(a.out, a.in, 0x20, 0x21, APX_MAXTAG, what) && i == 0)
                t_note("first switch: r16=0x%016llx", (unsigned long long)a.out[0]);
        }
        DeleteFiber(apx_other_fiber);
    }
    if (conv) ConvertFiberToThread();
    apx_bg_stop(g, got, "spinners during fiber switches");
}

/* ======================================================================================================
 * New thread: initial R16-R31 contain no value of another thread
 * ====================================================================================================== */
static uint64_t apx_new_out[16];
static DWORD WINAPI apx_new_thread(void *p) { (void)p; t__apx_store(apx_new_out); return 0; }

TEST(apx_new_thread_initial_state) {
    apx_require();
    apx_bg g[4];
    int got = apx_bg_start(g, 4, APX_BG_TAG, 0xFFFFFFFFu);
    int nonzero = 0;
    for (int i = 0; i < 32; i++) {
        memset(apx_new_out, 0xEE, sizeof apx_new_out);
        HANDLE th = CreateThread(NULL, 0, apx_new_thread, NULL, 0, NULL);
        if (!th) { t_note("CreateThread failed at %d", i); break; }
        CHECK(WaitForSingleObject(th, 5000) == WAIT_OBJECT_0);
        CloseHandle(th);
        for (int r = 0; r < 16; r++) {
            uint32_t tag, reg, word;
            int f = apx_new_out[r] && t_pattern_explain(apx_new_out[r], APX_MAXTAG, &tag, &reg, &word);
            CHECKF(!f, "new thread r%d = 0x%016llx: another thread's value (tag 0x%x reg %u)", 16 + r,
                   (unsigned long long)apx_new_out[r], f ? tag : 0, f ? reg : 0);
            nonzero += apx_new_out[r] != 0;
        }
    }
    t_info("new_thread_nonzero_egprs", "%d", nonzero);
    if (nonzero) t_note("new threads start with %d nonzero EGPR values (not architecturally required to be zero)", nonzero);
    apx_bg_stop(g, got, "spinners during thread creation");
}

/* ======================================================================================================
 * Hardware exceptions resumed by a VEH: the interrupted stream keeps R16-R31; the exception CONTEXT's
 * feature-19 image (when the OS includes it) equals the registers at the fault.
 * ====================================================================================================== */
static volatile LONG apx_veh_hits, apx_veh_located, apx_veh_bad;
static uint64_t apx_veh_want[16];

static LONG CALLBACK apx_veh(EXCEPTION_POINTERS *ep) {
    PCONTEXT c = ep->ContextRecord;
    char *ip = (char *)ep->ExceptionRecord->ExceptionAddress;
    int skip = ip == t__apx_fault_ud ? 2 : (ip == t__apx_fault_av || ip == t__apx_fault_de) ? 3 : 0;
    if (!skip) return EXCEPTION_CONTINUE_SEARCH;
    InterlockedIncrement(&apx_veh_hits);
    __typeof__(&LocateXStateFeature) loc = T_PROC("kernel32.dll", LocateXStateFeature);
    __typeof__(&GetXStateFeaturesMask) gm = T_PROC("kernel32.dll", GetXStateFeaturesMask);
    DWORD64 m = 0;
    if (loc && gm && (c->ContextFlags & CONTEXT_XSTATE) == CONTEXT_XSTATE && gm(c, &m) && (m & APX_MASK)) {
        DWORD len = 0;
        uint64_t *p = loc(c, APX_COMP, &len);
        if (p && len >= APX_SIZE) {
            InterlockedIncrement(&apx_veh_located);
            if (memcmp(p, apx_veh_want, APX_SIZE)) InterlockedIncrement(&apx_veh_bad);
        }
    }
    c->Rip += (DWORD64)skip;
    return EXCEPTION_CONTINUE_EXECUTION;
}

TEST(apx_exception_continue_preserves) {
    apx_require();
    PVOID h = AddVectoredExceptionHandler(1, apx_veh);
    REQUIRE(h);
    static const char *const kn[3] = { "#UD (ud2)", "#PF/AV (NULL load)", "#DE (div by 0)" };
    apx_veh_hits = apx_veh_located = apx_veh_bad = 0;
    for (int rep = 0; rep < 10; rep++)
        for (int k = 0; k < 3; k++) {
            apx_blk b; memset(&b, 0, sizeof b);
            apx_fill(b.in, 0x30 + (uint32_t)k);
            memcpy(apx_veh_want, b.in, sizeof apx_veh_want);
            t__apx_fault(&b, k);
            char what[64]; snprintf(what, sizeof what, "after %s handled by VEH", kn[k]);
            apx_check_exact(b.out, b.in, what);
        }
    RemoveVectoredExceptionHandler(h);
    CHECK_EQ(apx_veh_hits, 30);
    t_info("exception_ctx_with_feature19", "%ld of %ld", apx_veh_located, apx_veh_hits);
    if (!apx_veh_located) t_note("exception CONTEXT did not carry XSTATE feature 19 (not located)");
    CHECKF(apx_veh_bad == 0, "%ld exception contexts reported an R16-R31 image different from the registers at the fault",
           apx_veh_bad);
}

/* ======================================================================================================
 * GetThreadContext / SetThreadContext with CONTEXT_XSTATE feature 19
 * ====================================================================================================== */
TEST(apx_context_xstate_get_set) {
    apx_require();
    __typeof__(&InitializeContext) init = T_PROC("kernel32.dll", InitializeContext);
    __typeof__(&LocateXStateFeature) loc = T_PROC("kernel32.dll", LocateXStateFeature);
    __typeof__(&SetXStateFeaturesMask) setm = T_PROC("kernel32.dll", SetXStateFeaturesMask);
    __typeof__(&GetXStateFeaturesMask) getm = T_PROC("kernel32.dll", GetXStateFeaturesMask);
    if (!init || !loc || !setm || !getm) SKIP("XSTATE context APIs not exported");
    if (!apx_api_reports()) SKIP("GetEnabledXStateFeatures lacks bit 19 (FAIL reported by apx_api_enabled_features)");
    DWORD flags = CONTEXT_ALL | CONTEXT_XSTATE, len = 0;
    init(NULL, flags, NULL, &len);
    REQUIREF(len > sizeof(CONTEXT), "InitializeContext length %lu", len);
    void *buf = calloc(1, len);
    REQUIRE(buf);
    PCONTEXT ctx = NULL;
    REQUIREF(init(buf, flags, &ctx, &len), "InitializeContext failed (%lu)", GetLastError());
    REQUIREF(setm(ctx, APX_MASK), "SetXStateFeaturesMask(1<<19) failed (%lu)", GetLastError());

    apx_bg g[1];
    int got = apx_bg_start(g, 1, APX_BG_TAG, 0xFFFFFFFFu);
    if (got < 1) { free(buf); SKIP("thread creation failed"); }
    uint64_t newv[16]; apx_fill(newv, 0x40);
    REQUIRE(SuspendThread(g[0].th) != (DWORD)-1);
    CHECKF(GetThreadContext(g[0].th, ctx), "GetThreadContext(CONTEXT_XSTATE, mask 1<<19) failed (%lu)", GetLastError());
    DWORD64 m = 0;
    CHECK(getm(ctx, &m));
    t_info("xstate_mask_after_get", "0x%llx", (unsigned long long)m);
    CHECKF(m & APX_MASK, "GetXStateFeaturesMask=0x%llx: bit 19 clear although R16-R31 hold nonzero values", (unsigned long long)m);
    DWORD flen = 0;
    uint64_t *p = loc(ctx, APX_COMP, &flen);
    CHECKF(p != NULL, "LocateXStateFeature(ctx, 19) returned NULL");
    if (p) {
        CHECK_EQ(flen, APX_SIZE);
        apx_check_exact(p, g[0].b.in, "GetThreadContext feature 19 image");
        memcpy(p, newv, APX_SIZE);
        CHECKF(setm(ctx, APX_MASK), "SetXStateFeaturesMask failed (%lu)", GetLastError());
        /* ContextFlags as returned by GetThreadContext: the other state is written back unchanged */
        CHECKF(SetThreadContext(g[0].th, ctx), "SetThreadContext(CONTEXT_XSTATE, mask 1<<19) failed (%lu)", GetLastError());
        /* read back while still suspended */
        memset(p, 0, APX_SIZE);
        ctx->ContextFlags = flags;
        CHECK(setm(ctx, APX_MASK));
        CHECK(GetThreadContext(g[0].th, ctx));
        apx_check_exact(p, newv, "GetThreadContext after SetThreadContext");
    }
    CHECK(ResumeThread(g[0].th) != (DWORD)-1);
    InterlockedExchange(&g[0].b.stop, 1);
    CHECK(WaitForSingleObject(g[0].th, 5000) == WAIT_OBJECT_0);
    CloseHandle(g[0].th);
    if (p) apx_check_exact(g[0].b.out, newv, "worker registers after SetThreadContext feature 19");
    free(buf);
}
#endif /* T_X64 */
