/* feature_faults - unsupported-instruction, privileged-instruction and disabled-state fault behaviour.
 *
 * Every probe that may fault runs in its OWN child process (<exe> --child=probe <name>) under a timeout.
 * A probe is a small assembly routine with three global labels:
 *     fp_<name>        entry: saves callee-saved registers, loads marker values, runs the setup code
 *     fp_<name>_insn   the probed instruction (followed by stores of its results when it does not fault)
 *     fp_<name>_res    resume point: restores RSP/ESP from a saved copy, the x87 control word, MXCSR, the
 *                      callee-saved registers, and returns
 * The child installs a vectored handler that records the exception (code, ExceptionAddress relative to
 * the _insn label, context IP, parameters, marker registers, x87 status word, MXCSR, context flags),
 * redirects the context to _res and continues. The child prints one normalised "FP ..." line that the
 * parent compares with the expectation of the case.
 *
 * Hard CHECKs only for what the SDM/APM or the Windows exception contract guarantees:
 *   - #UD (ud2/ud1/ud0, LOCK on a non-lockable form, invalid VEX prefixes/fields, opcodes invalid in
 *     64-bit mode, an ISA extension that the processor does not implement or whose XCR0 state is
 *     disabled) -> STATUS_ILLEGAL_INSTRUCTION, ExceptionAddress = the instruction (a fault).
 *   - #GP from privileged instructions at CPL3 (always a fault, never executes).
 *   - #DE, #BR, #OF, #BP, #DB(TF), #MF, #XM, page faults: the documented EXCEPTION_* codes.
 * Facts that legitimately vary (UMIP, CR4.PCE for RDPMC, syscall paths, hypervisor CPUID masking) are
 * NOTE/INFO.  A feature that the processor implements and the OS enables is never executed as a
 * "negative" probe: those cases SKIP, so the suite is safe on AVX/AVX-512/AMX-capable hosts.
 *
 * Windows reference (Windows 10 22H2, build 19041.2965 corpus, checked through the decompiler API):
 *  - ntoskrnl!KiPreprocessFault: a user #GP becomes STATUS_ACCESS_VIOLATION with NumberParameters=2,
 *    Information = {0, ~0} (non-canonical style address), then KiOpDecode looks the opcode up in
 *    KiOpOneByteTable / KiOpTwoByteTable.  Entries whose handler is KiOp_Priv change the code to
 *    STATUS_PRIVILEGED_INSTRUCTION (NumberParameters=0):
 *        FA cli, FB sti, E4-E7 / EC-EF in/out, 6C-6F ins/outs, F4 hlt,
 *        0F 00 /2 lldt, /3 ltr, 0F 01 /2,/3 (memory) lgdt/lidt, 0F 01 /6 lmsw, 0F 01 /7 (memory) invlpg,
 *        0F 01 F8 swapgs, 0F 06 clts, 0F 07 sysret, 0F 08 invd, 0F 09 wbinvd, 0F 20-23 mov cr/dr,
 *        0F 30-33 wrmsr/rdtsc/rdmsr/rdpmc, 0F 35 sysexit.          0F AA rsm -> KiOp_Illegal.
 *    XSETBV (0F 01 D1), an invalid XGETBV, misaligned MOVAPS/CMPXCHG16B, LDMXCSR with reserved bits,
 *    XRSTOR with a bad header and INT n through a DPL0 gate are NOT listed: they stay ACCESS_VIOLATION.
 *  - 0F 00 /0,/1 (sldt/str), 0F 01 /0,/1 (memory sgdt/sidt) and 0F 01 /4 (smsw) go to
 *    KiOp_SLDTSTRSMSW / KiOp_SGDTSIDT: with UMIP enabled Windows emulates them, so they never fault.
 *  - 0F 0D /1 (prefetchw) on #UD -> KiOp_PREFETCH3 skips the instruction (emulated as a NOP).
 *  - F6/F7 /6,/7 (div/idiv) on #DE -> KiOp_Div: divisor != 0 -> STATUS_INTEGER_OVERFLOW, else
 *    STATUS_INTEGER_DIVIDE_BY_ZERO.
 *  - KiXmmException (#XM): 64-bit code: IE->FLOAT_INVALID_OPERATION, ZE->FLOAT_DIVIDE_BY_ZERO,
 *    DE->FLOAT_INVALID_OPERATION, OE->FLOAT_OVERFLOW, UE->FLOAT_UNDERFLOW, PE->FLOAT_INEXACT_RESULT
 *    (unmasked flags = MXCSR & ~(MXCSR>>7)).  32-bit code (CS=0x23, WOW64): IE/ZE/DE ->
 *    STATUS_FLOAT_MULTIPLE_TRAPS (0xC00002B5), OE/UE/PE -> STATUS_FLOAT_MULTIPLE_FAULTS (0xC00002B4).
 *  - KiFloatingErrorFault (#MF): SW & ~(CW & 0x3F): IE -> SF ? FLOAT_STACK_CHECK : FLOAT_INVALID_OPERATION;
 *    ZE -> DIVIDE_BY_ZERO, DE -> INVALID_OPERATION, OE, UE, PE -> the matching code.
 *  - KiBreakpointTrap: STATUS_BREAKPOINT, address = RIP-1 (so CD 03 reports insn+1).
 *  - KiOverflowTrap (INTO): STATUS_INTEGER_OVERFLOW at RIP-1.  KiBoundFault: STATUS_ARRAY_BOUNDS_EXCEEDED.
 *  - KiRaiseAssertion (int 2C): RIP -= 2, STATUS_ASSERTION_FAILURE.
 *  - KiRaiseSecurityCheckFailure (int 29): fast fail, STATUS_STACK_BUFFER_OVERRUN, no handlers run.
 *  - KiPreprocessFault (re-checked in review): class 0x10000002 (#UD) sets STATUS_ILLEGAL_INSTRUCTION and decodes
 *    with the #UD table mask; class 0x10000001 (#GP) sets STATUS_ACCESS_VIOLATION and, unless a KiOp handler
 *    changes it, rewrites the parameters to {0, 0xFFFFFFFFFFFFFFFF}.  INT 80h / INT 21h (DPL0 or absent gates)
 *    therefore are ACCESS_VIOLATION: a Linux-hosted layer that runs INT 80h as a Linux system call is wrong.
 * Review additions: ExceptionFlags must be 0 for every hardware exception; worker-thread delivery; the same
 * fault repeated hundreds of times in one process (every repetition identical, callee-saved registers intact);
 * XSAVES/XRSTORS at CPL3; MOV to DR7/CR8, INVD; reserved EVEX.L'L and KMOVW VEX.L=1 encodings; AMD parts that
 * still execute de-enumerated FMA4/XOP/3DNow!/TBM/SSE4a are NOTE, not FAIL; SYSEXIT on AMD may be #UD.
 */
#include "t.h"

#ifndef CONTEXT_XSTATE
#ifdef T_X64
#define CONTEXT_XSTATE 0x00100040
#else
#define CONTEXT_XSTATE 0x00010040
#endif
#endif

/* exception codes */
#define S_AV        0xC0000005u
#define S_INVSVC    0xC000001Cu
#define S_ILL       0xC000001Du
#define S_BOUNDS    0xC000008Cu
#define S_FDEN      0xC000008Du
#define S_FDIV      0xC000008Eu
#define S_FINEX     0xC000008Fu
#define S_FINV      0xC0000090u
#define S_FOVF      0xC0000091u
#define S_FSTK      0xC0000092u
#define S_FUND      0xC0000093u
#define S_IDIV0     0xC0000094u
#define S_IOVF      0xC0000095u
#define S_PRIV      0xC0000096u
#define S_FMF       0xC00002B4u
#define S_FMT       0xC00002B5u
#define S_FASTFAIL  0xC0000409u
#define S_ASSERT    0xC0000420u
#define S_BP        0x80000003u
#define S_SS        0x80000004u

static const char *fp_code_name(unsigned long c) {
    switch (c) {
    case 0: return "none";
    case S_AV: return "STATUS_ACCESS_VIOLATION";
    case S_INVSVC: return "STATUS_INVALID_SYSTEM_SERVICE";
    case S_ILL: return "STATUS_ILLEGAL_INSTRUCTION";
    case S_BOUNDS: return "STATUS_ARRAY_BOUNDS_EXCEEDED";
    case S_FDEN: return "STATUS_FLOAT_DENORMAL_OPERAND";
    case S_FDIV: return "STATUS_FLOAT_DIVIDE_BY_ZERO";
    case S_FINEX: return "STATUS_FLOAT_INEXACT_RESULT";
    case S_FINV: return "STATUS_FLOAT_INVALID_OPERATION";
    case S_FOVF: return "STATUS_FLOAT_OVERFLOW";
    case S_FSTK: return "STATUS_FLOAT_STACK_CHECK";
    case S_FUND: return "STATUS_FLOAT_UNDERFLOW";
    case S_IDIV0: return "STATUS_INTEGER_DIVIDE_BY_ZERO";
    case S_IOVF: return "STATUS_INTEGER_OVERFLOW";
    case S_PRIV: return "STATUS_PRIVILEGED_INSTRUCTION";
    case S_FMF: return "STATUS_FLOAT_MULTIPLE_FAULTS";
    case S_FMT: return "STATUS_FLOAT_MULTIPLE_TRAPS";
    case S_FASTFAIL: return "STATUS_STACK_BUFFER_OVERRUN";
    case S_ASSERT: return "STATUS_ASSERTION_FAILURE";
    case S_BP: return "STATUS_BREAKPOINT";
    case S_SS: return "STATUS_SINGLE_STEP";
    case 0x80000002u: return "STATUS_DATATYPE_MISALIGNMENT";
    case 0xC00000FDu: return "STATUS_STACK_OVERFLOW";
    default: return "?";
    }
}

/* =============================================================================================
 * Shared data used by the probe routines (non-static: referenced by name from assembly)
 * ============================================================================================= */
__attribute__((used, aligned(64))) uint8_t  fp_scratch[8192];
__attribute__((used, aligned(32))) uint8_t  fp_pat[64];          /* register pattern for context probes */
__attribute__((used, aligned(16))) uint8_t  fp_xmm6_save[16];
__attribute__((used, aligned(16))) uint8_t  fp_xmm15_save[16];
__attribute__((used, aligned(16))) uint8_t  fp_dt[16];           /* SGDT/SIDT image for LGDT/LIDT */
__attribute__((used)) uintptr_t fp_saved_sp;
__attribute__((used)) uint64_t  fp_out[4];
__attribute__((used)) uint32_t  fp_arg[4];                       /* xsetbv value lo/hi */
__attribute__((used)) uintptr_t fp_nx_target;
__attribute__((used)) uint16_t  fp_cw_saved;
__attribute__((used)) uint32_t  fp_mx_saved;
__attribute__((used)) uint32_t  fp_bounds[2] = { 0, 10 };
/* MXCSR images */
__attribute__((used)) uint32_t  fp_mx_bad = 0x80001F80u;          /* reserved bit 31 -> #GP */
__attribute__((used)) uint32_t  fp_mx_ctx = 0x3F80u;              /* RC=01 (round down), all masked */
__attribute__((used)) uint32_t  fp_mx_im = 0x1F80u & ~0x0080u;    /* invalid unmasked */
__attribute__((used)) uint32_t  fp_mx_dm = 0x1F80u & ~0x0100u;    /* denormal unmasked */
__attribute__((used)) uint32_t  fp_mx_zm = 0x1F80u & ~0x0200u;    /* zero-divide unmasked */
__attribute__((used)) uint32_t  fp_mx_om = 0x1F80u & ~0x0400u;    /* overflow unmasked */
__attribute__((used)) uint32_t  fp_mx_um = 0x1F80u & ~0x0800u;    /* underflow unmasked */
__attribute__((used)) uint32_t  fp_mx_pm = 0x1F80u & ~0x1000u;    /* precision unmasked */
/* x87 control words (0x037F = all masked, 64-bit precision, nearest) */
__attribute__((used)) uint16_t  fp_cw_ctx = 0x0E7F;               /* RC=11 (chop), PC=10 */
__attribute__((used)) uint16_t  fp_cw_im = 0x037E;
__attribute__((used)) uint16_t  fp_cw_dm = 0x037D;
__attribute__((used)) uint16_t  fp_cw_zm = 0x037B;
__attribute__((used)) uint16_t  fp_cw_om = 0x0377;
__attribute__((used)) uint16_t  fp_cw_um = 0x036F;
__attribute__((used)) uint16_t  fp_cw_pm = 0x035F;
/* operands */
__attribute__((used)) float     fp_f32[8] = { 1.0f, -1.0f, 0.0f /* denormal, set below */, 3.402823466e38f,
                                              1.175494351e-38f, 3.0f, 0.0f, 0.0f };
__attribute__((used)) uint64_t  fp_den64 = 1;                      /* smallest double denormal */
__attribute__((used, aligned(16))) uint8_t fp_big80[16] = { 0,0,0,0,0,0,0,0x80, 0xFE,0x7F };  /* 2^16383 */
__attribute__((used, aligned(16))) uint8_t fp_tiny80[16] = { 0,0,0,0,0,0,0,0x80, 0x01,0x00 }; /* 2^-16382 */

/* =============================================================================================
 * Probe definition machinery
 * ============================================================================================= */
#ifdef T_X64
#define FP_SYM(x)      #x
#define FP_G(x)        #x "(%rip)"
#define FP_GO(x, o)    #x "+" #o "(%rip)"
#define FP_AX "%rax"
#define FP_CX "%rcx"
#define FP_DX "%rdx"
#define FP_SI "%rsi"
#define FP_DI "%rdi"
#define FP_SP "%rsp"
#define FP_M12 0x5A5A0000C0DE0012ull
#define FP_M13 0x5A5A0000C0DE0013ull
#define FP_M14 0x5A5A0000C0DE0014ull
#define FP_M15 0x5A5A0000C0DE0015ull
#define FP_PRO \
    "push %rbx\n push %rbp\n push %rsi\n push %rdi\n push %r12\n push %r13\n push %r14\n push %r15\n" \
    "mov %rsp, fp_saved_sp(%rip)\n fnstcw fp_cw_saved(%rip)\n stmxcsr fp_mx_saved(%rip)\n" \
    "movabs $0x5A5A0000C0DE0012, %r12\n movabs $0x5A5A0000C0DE0013, %r13\n" \
    "movabs $0x5A5A0000C0DE0014, %r14\n movabs $0x5A5A0000C0DE0015, %r15\n"
#define FP_EPI_A "mov fp_saved_sp(%rip), %rsp\n cld\n fninit\n fldcw fp_cw_saved(%rip)\n ldmxcsr fp_mx_saved(%rip)\n"
#define FP_EPI_B "pop %r15\n pop %r14\n pop %r13\n pop %r12\n pop %rdi\n pop %rsi\n pop %rbp\n pop %rbx\n ret\n"
#else
#define FP_SYM(x)      "_" #x
#define FP_G(x)        "_" #x
#define FP_GO(x, o)    "_" #x "+" #o
#define FP_AX "%eax"
#define FP_CX "%ecx"
#define FP_DX "%edx"
#define FP_SI "%esi"
#define FP_DI "%edi"
#define FP_SP "%esp"
#define FP_MBP 0x5A5AEB01u
#define FP_PRO \
    "push %ebx\n push %ebp\n push %esi\n push %edi\n" \
    "mov %esp, _fp_saved_sp\n fnstcw _fp_cw_saved\n stmxcsr _fp_mx_saved\n mov $0x5A5AEB01, %ebp\n"
#define FP_EPI_A "mov _fp_saved_sp, %esp\n cld\n fninit\n fldcw _fp_cw_saved\n ldmxcsr _fp_mx_saved\n"
#define FP_EPI_B "pop %edi\n pop %esi\n pop %ebp\n pop %ebx\n ret\n"
#endif
#define FP_LEA(sym, reg)        "lea " FP_G(sym) ", " reg "\n"
#define FP_LEAO(sym, off, reg)  "lea " FP_GO(sym, off) ", " reg "\n"
#define FP_ST(reg, idx)         "mov " reg ", " FP_GO(fp_out, idx) "\n"   /* idx = byte offset */

typedef struct fp_probe { const char *name; void (*fn)(void); const char *insn, *res; } fp_probe;
static fp_probe g_probes[224];
static int g_nprobes;
static void fp_add(const char *name, void (*fn)(void), const char *insn, const char *res) {
    if (g_nprobes < (int)T_COUNTOF(g_probes)) { fp_probe p = { name, fn, insn, res }; g_probes[g_nprobes++] = p; }
}
static const fp_probe *fp_find(const char *name) {
    for (int i = 0; i < g_nprobes; i++) if (!strcmp(g_probes[i].name, name)) return &g_probes[i];
    return NULL;
}

/* FP_DEF(name, setup, insn, cleanup): see the file header. Strings may contain several instructions. */
#define FP_DEF(name, setup, insn, cleanup) \
    __asm__(".text\n.p2align 4\n" \
            ".globl " FP_SYM(fp_##name) "\n" FP_SYM(fp_##name) ":\n" FP_PRO setup "\n" \
            ".globl " FP_SYM(fp_##name##_insn) "\n" FP_SYM(fp_##name##_insn) ":\n" insn "\n" \
            ".globl " FP_SYM(fp_##name##_res) "\n" FP_SYM(fp_##name##_res) ":\n" FP_EPI_A cleanup "\n" FP_EPI_B \
            ".text\n"); \
    extern void fp_##name(void); \
    extern char fp_##name##_insn[], fp_##name##_res[]; \
    T_CTOR_(fp_reg_##name) { fp_add(#name, fp_##name, fp_##name##_insn, fp_##name##_res); }

/* ---- #UD: architecturally undefined encodings -------------------------------------------- */
FP_DEF(ud_ud2, "", "ud2", "")
FP_DEF(ud_ud1, "", ".byte 0x0f,0xb9,0xc0", "")                 /* UD1 eax,eax */
FP_DEF(ud_ud0, "", ".byte 0x0f,0xff,0xc0", "")                 /* UD0 (ModRM on Intel, none on AMD) */
FP_DEF(ud_rsm, "", ".byte 0x0f,0xaa", "")                      /* RSM outside SMM */
FP_DEF(ud_lock_add_reg, "", ".byte 0xf0,0x01,0xd8", "")        /* lock add eax,ebx: register destination */
FP_DEF(ud_lock_nop, "", ".byte 0xf0,0x90", "")                 /* lock nop */
FP_DEF(ud_lock_cmpxchg_reg, "", ".byte 0xf0,0x0f,0xb1,0xc8", "") /* lock cmpxchg eax,ecx */
FP_DEF(ud_lock_mov_mem, FP_LEA(fp_scratch, FP_AX), ".byte 0xf0,0x89,0x08", "")  /* lock mov [ax],ecx */
FP_DEF(ok_lock_add_mem, FP_LEA(fp_scratch, FP_AX) "mov $1, %ecx\n", ".byte 0xf0,0x01,0x08", "") /* lock add [ax],ecx */
FP_DEF(ud_lea_reg, "", ".byte 0x8d,0xc0", "")                  /* lea eax,eax */
FP_DEF(ud_cmpxchg8b_reg, "", ".byte 0x0f,0xc7,0xc8", "")       /* cmpxchg8b with mod=11 */
FP_DEF(ud_movntdq_reg, "", ".byte 0x66,0x0f,0xe7,0xc0", "")    /* movntdq xmm0,xmm0 (register form) */
/* VEX encodings that #UD by definition (SDM 2.3.2 / instruction pages); on a non-AVX processor they #UD too */
FP_DEF(ud_vex_66_prefix, "", ".byte 0x66,0xc5,0xf8,0x77", "")  /* 66 before VEX */
FP_DEF(ud_vex_f3_prefix, "", ".byte 0xf3,0xc5,0xf8,0x77", "")  /* F3 before VEX */
FP_DEF(ud_vex_lock_prefix, "", ".byte 0xf0,0xc5,0xf8,0x77", "") /* LOCK before VEX */
FP_DEF(ud_vex_l1_vmovd, "", ".byte 0xc5,0xfd,0x6e,0xc0", "")   /* VMOVD with VEX.L=1 */
FP_DEF(ud_vex_vvvv_vmovdqa, "", ".byte 0xc5,0xf1,0x6f,0xc1", "") /* VMOVDQA with VEX.vvvv != 1111b */
/* KMOVW k1,eax with VEX.L=1 (KMOV* are L0-only): #UD with or without AVX-512 (VEX 0F 92 is undefined otherwise) */
FP_DEF(ud_vex_kmovw_l1, "", ".byte 0xc5,0xfc,0x92,0xc8", "")
/* EVEX VADDPS with EVEX.L'L = 11b and EVEX.b = 0 (reserved vector length): #UD with AVX-512; without AVX-512
 * 62 is #UD in 64-bit mode and BOUND with a register operand (#UD) in 32-bit mode. */
FP_DEF(ud_evex_ll11, "", ".byte 0x62,0xf1,0x7c,0x68,0x58,0xc0", "")
FP_DEF(ud_movbe_reg, "", ".byte 0x0f,0x38,0xf0,0xc0", "")       /* 0F 38 F0 with mod=11 and no F2: undefined */
#ifdef T_X64
FP_DEF(ud_vex_rex_prefix, "", ".byte 0x40,0xc5,0xf8,0x77", "") /* REX before VEX */
FP_DEF(ud_x64_daa, "", ".byte 0x27", "")
FP_DEF(ud_x64_aaa, "", ".byte 0x37", "")
FP_DEF(ud_x64_aam, "", ".byte 0xd4,0x0a", "")
FP_DEF(ud_x64_pusha, "", ".byte 0x60", "")
FP_DEF(ud_x64_into, "", ".byte 0xce", "")
FP_DEF(ud_x64_push_es, "", ".byte 0x06", "")
FP_DEF(ud_x64_jmp_far, "", ".byte 0xea,0,0,0,0,0x33,0", "")
FP_DEF(ud_cmpxchg16b_reg, "", ".byte 0x48,0x0f,0xc7,0xc8", "")  /* REX.W cmpxchg16b with mod=11 */
#endif

/* ---- ISA extensions: executed only when the feature is not usable ------------------------- */
FP_DEF(feat_avx_vaddps_ymm, "", ".byte 0xc5,0xfc,0x58,0xc0", "")
FP_DEF(feat_avx_vzeroupper, "", ".byte 0xc5,0xf8,0x77", "")
FP_DEF(feat_avx2_vpaddd_ymm, "", ".byte 0xc5,0xfd,0xfe,0xc0", "")
FP_DEF(feat_fma_vfmadd213ps, "", ".byte 0xc4,0xe2,0x79,0xa8,0xc0", "")
FP_DEF(feat_f16c_vcvtph2ps, "", ".byte 0xc4,0xe2,0x79,0x13,0xc0", "")
FP_DEF(feat_avx512f_vaddps_zmm, "", ".byte 0x62,0xf1,0x7c,0x48,0x58,0xc0", "")
FP_DEF(feat_avx512f_kmovw_kreg, "", ".byte 0xc5,0xf8,0x92,0xc8", "")
FP_DEF(feat_avx512vl_vpternlogd, "", ".byte 0x62,0xf3,0x7d,0x08,0x25,0xc0,0x00", "")
FP_DEF(feat_avx512bw_vpaddb_zmm, "", ".byte 0x62,0xf1,0x7d,0x48,0xfc,0xc0", "")
FP_DEF(feat_avx512dq_vpmullq_zmm, "", ".byte 0x62,0xf2,0xfd,0x48,0x40,0xc0", "")
FP_DEF(feat_avx512cd_vplzcntd_zmm, "", ".byte 0x62,0xf2,0x7d,0x48,0x44,0xc0", "")
FP_DEF(feat_avx512vnni_vpdpbusd, "", ".byte 0x62,0xf2,0x7d,0x48,0x50,0xc0", "")
FP_DEF(feat_avx512fp16_vaddph, "", ".byte 0x62,0xf5,0x7c,0x48,0x58,0xc0", "")
FP_DEF(feat_amx_tilerelease, "", ".byte 0xc4,0xe2,0x78,0x49,0xc0", "")
FP_DEF(feat_amx_tilezero, "", ".byte 0xc4,0xe2,0x7b,0x49,0xc0", "")
FP_DEF(feat_bmi1_andn, "", ".byte 0xc4,0xe2,0x78,0xf2,0xc0", "")
FP_DEF(feat_bmi2_pdep, "", ".byte 0xc4,0xe2,0x7b,0xf5,0xc0", "")
FP_DEF(feat_vaes_ymm, "", ".byte 0xc4,0xe2,0x7d,0xdc,0xc0", "")
FP_DEF(feat_vpclmul_ymm, "", ".byte 0xc4,0xe3,0x7d,0x44,0xc0,0x00", "")
FP_DEF(feat_avx_vnni_vpdpbusd, "", ".byte 0xc4,0xe2,0x79,0x50,0xc0", "")
FP_DEF(feat_fma4_vfmaddps, "", ".byte 0xc4,0xe3,0x79,0x68,0xc0,0x00", "")
FP_DEF(feat_xop_vex_vpcmov, "", ".byte 0x8f,0xe8,0x78,0xa2,0xc0,0x00", "")
FP_DEF(feat_adx_adcx, "", ".byte 0x66,0x0f,0x38,0xf6,0xc0", "")
FP_DEF(feat_aes_aesenc, "", ".byte 0x66,0x0f,0x38,0xdc,0xc0", "")
FP_DEF(feat_pclmulqdq, "", ".byte 0x66,0x0f,0x3a,0x44,0xc0,0x00", "")
FP_DEF(feat_sha1rnds4, "", ".byte 0x0f,0x3a,0xcc,0xc0,0x00", "")
FP_DEF(feat_sha256rnds2, "", ".byte 0x0f,0x38,0xcb,0xc1", "")
FP_DEF(feat_gfni_gf2p8mulb, "", ".byte 0x66,0x0f,0x38,0xcf,0xc0", "")
FP_DEF(feat_sse42_crc32, "", ".byte 0xf2,0x0f,0x38,0xf1,0xc0", "")
FP_DEF(feat_popcnt, "", ".byte 0xf3,0x0f,0xb8,0xc0", "")
FP_DEF(feat_movbe, FP_LEA(fp_scratch, FP_AX), ".byte 0x0f,0x38,0xf0,0x00", "")
FP_DEF(feat_rdrand, "", ".byte 0x0f,0xc7,0xf0", "")
FP_DEF(feat_rdseed, "", ".byte 0x0f,0xc7,0xf8", "")
FP_DEF(feat_rdtscp, "", ".byte 0x0f,0x01,0xf9", "")
FP_DEF(feat_movdiri, FP_LEA(fp_scratch, FP_AX), ".byte 0x0f,0x38,0xf9,0x08", "")
FP_DEF(feat_3dnow_femms, "", ".byte 0x0f,0x0e", "")
FP_DEF(feat_sse4a_extrq, "", ".byte 0x66,0x0f,0x78,0xc0,0x00,0x00", "")
FP_DEF(feat_xgetbv_no_osxsave, "xor %ecx, %ecx\n", ".byte 0x0f,0x01,0xd0", "")
FP_DEF(feat_xsaveopt, FP_LEA(fp_scratch, FP_CX) "mov $3, %eax\n xor %edx, %edx\n", ".byte 0x0f,0xae,0x31", "")
FP_DEF(feat_xsavec, FP_LEA(fp_scratch, FP_CX) "mov $3, %eax\n xor %edx, %edx\n", ".byte 0x0f,0xc7,0x21", "")
FP_DEF(feat_xsave_no_osxsave, FP_LEA(fp_scratch, FP_CX) "mov $3, %eax\n xor %edx, %edx\n", ".byte 0x0f,0xae,0x21", "")

/* ---- defined behaviour without a fault --------------------------------------------------- */
FP_DEF(nf_lzcnt_or_bsr, "mov $0x10, %eax\n", ".byte 0xf3,0x0f,0xbd,0xc0\n" FP_ST("%eax", 0), "")
FP_DEF(nf_tzcnt_or_bsf, "mov $0x12345, %ecx\n xor %eax, %eax\n",
       ".byte 0xf3,0x0f,0xbc,0xc8\n pushf\n pop " FP_AX "\n" FP_ST("%ecx", 0) FP_ST("%eax", 8), "")
FP_DEF(nf_cldemote_hint, FP_LEA(fp_scratch, FP_AX), ".byte 0x0f,0x1c,0x00", "")
FP_DEF(nf_endbr64, "", ".byte 0xf3,0x0f,0x1e,0xfa", "")
FP_DEF(nf_endbr32, "", ".byte 0xf3,0x0f,0x1e,0xfb", "")
FP_DEF(nf_prefetchw, FP_LEA(fp_scratch, FP_AX), ".byte 0x0f,0x0d,0x08", "")
FP_DEF(nf_pause, "", ".byte 0xf3,0x90", "")
FP_DEF(nf_rdtsc, "", "rdtsc\n" FP_ST("%eax", 0) FP_ST("%edx", 4), "")
FP_DEF(nf_rdpid, "xor %eax, %eax\n", ".byte 0xf3,0x0f,0xc7,0xf8\n" FP_ST("%eax", 0), "")
FP_DEF(nf_rdtscp, "", ".byte 0x0f,0x01,0xf9\n" FP_ST("%eax", 0) FP_ST("%edx", 4) FP_ST("%ecx", 8), "")
FP_DEF(nf_xgetbv_ecx1, "mov $1, %ecx\n", ".byte 0x0f,0x01,0xd0\n" FP_ST("%eax", 0) FP_ST("%edx", 4), "")

/* ---- privileged instructions at CPL3 ----------------------------------------------------- */
FP_DEF(priv_hlt, "", ".byte 0xf4", "")
FP_DEF(priv_cli, "", ".byte 0xfa", "")
FP_DEF(priv_sti, "", ".byte 0xfb", "")
FP_DEF(priv_in_dx, "mov $0x80, %edx\n", ".byte 0xec", "")
FP_DEF(priv_out_dx, "mov $0x80, %edx\n xor %eax, %eax\n", ".byte 0xee", "")
FP_DEF(priv_in_imm, "", ".byte 0xe4,0x80", "")
FP_DEF(priv_out_imm, "xor %eax, %eax\n", ".byte 0xe6,0x80", "")
FP_DEF(priv_insb, FP_LEA(fp_scratch, FP_DI) "mov $0x80, %edx\n", ".byte 0x6c", "")
FP_DEF(priv_outsb, FP_LEA(fp_scratch, FP_SI) "mov $0x80, %edx\n", ".byte 0x6e", "")
FP_DEF(priv_mov_from_cr0, "", ".byte 0x0f,0x20,0xc0", "")
FP_DEF(priv_mov_from_cr3, "", ".byte 0x0f,0x20,0xd8", "")
FP_DEF(priv_mov_from_cr4, "", ".byte 0x0f,0x20,0xe0", "")
FP_DEF(priv_mov_from_dr7, "", ".byte 0x0f,0x21,0xf8", "")
FP_DEF(priv_rdmsr, "mov $0x10, %ecx\n", ".byte 0x0f,0x32", "")
FP_DEF(priv_wrmsr, "mov $0xffffffff, %ecx\n xor %eax, %eax\n xor %edx, %edx\n", ".byte 0x0f,0x30", "")
FP_DEF(priv_rdpmc, "xor %ecx, %ecx\n", ".byte 0x0f,0x33\n" FP_ST("%eax", 0) FP_ST("%edx", 4), "")
FP_DEF(priv_clts, "", ".byte 0x0f,0x06", "")
FP_DEF(priv_wbinvd, "", ".byte 0x0f,0x09", "")
FP_DEF(priv_invlpg, FP_LEA(fp_scratch, FP_AX), ".byte 0x0f,0x01,0x38", "")
FP_DEF(priv_lgdt, "sgdt " FP_G(fp_dt) "\n" FP_LEA(fp_dt, FP_AX), ".byte 0x0f,0x01,0x10", "")
FP_DEF(priv_lidt, "sidt " FP_G(fp_dt) "\n" FP_LEA(fp_dt, FP_AX), ".byte 0x0f,0x01,0x18", "")
FP_DEF(priv_lldt, "xor %eax, %eax\n", ".byte 0x0f,0x00,0xd0", "")
FP_DEF(priv_ltr, "xor %eax, %eax\n", ".byte 0x0f,0x00,0xd8", "")
FP_DEF(priv_lmsw, ".byte 0x0f,0x01,0xe0\n", ".byte 0x0f,0x01,0xf0", "")     /* smsw eax; lmsw ax */
FP_DEF(priv_sysexit, "", ".byte 0x0f,0x35", "")
FP_DEF(priv_invd, "", ".byte 0x0f,0x08", "")
FP_DEF(priv_mov_to_dr7, "xor %eax, %eax\n", ".byte 0x0f,0x23,0xf8", "")      /* mov dr7, eax (value 0) */
/* XSAVES/XRSTORS: #GP at CPL>0 (supervisor state), #UD when not implemented */
FP_DEF(gp_xsaves_user, FP_LEA(fp_scratch, FP_CX) "mov $3, %eax\n xor %edx, %edx\n", ".byte 0x0f,0xc7,0x29", "")
FP_DEF(gp_xrstors_user, FP_LEA(fp_scratch, FP_CX) "mov $3, %eax\n xor %edx, %edx\n", ".byte 0x0f,0xc7,0x19", "")
FP_DEF(priv_xsetbv, "xor %ecx, %ecx\n mov " FP_GO(fp_arg, 0) ", %eax\n mov " FP_GO(fp_arg, 4) ", %edx\n",
       ".byte 0x0f,0x01,0xd1", "")
FP_DEF(gp_xgetbv_ecx2, "mov $2, %ecx\n", ".byte 0x0f,0x01,0xd0\n" FP_ST("%eax", 0), "")
FP_DEF(gp_xgetbv_ecx_ffffffff, "mov $0xffffffff, %ecx\n", ".byte 0x0f,0x01,0xd0\n" FP_ST("%eax", 0), "")
#ifdef T_X64
FP_DEF(priv_mov_from_cr8, "", ".byte 0x44,0x0f,0x20,0xc0", "")
FP_DEF(priv_mov_to_cr8, "xor %eax, %eax\n", ".byte 0x44,0x0f,0x22,0xc0", "")  /* mov cr8, rax (value 0) */
FP_DEF(priv_swapgs, "", ".byte 0x0f,0x01,0xf8", "")
FP_DEF(priv_sysret, "lea fp_priv_sysret_res(%rip), %rcx\n pushf\n pop %r11\n", ".byte 0x48,0x0f,0x07", "")
#endif

/* ---- UMIP-class descriptor-table reads ----------------------------------------------------- */
FP_DEF(umip_sgdt, "", "sgdt " FP_G(fp_scratch), "")
FP_DEF(umip_sidt, "", "sidt " FP_G(fp_scratch), "")
FP_DEF(umip_smsw, "xor %eax, %eax\n", ".byte 0x0f,0x01,0xe0\n" FP_ST("%eax", 0), "")
FP_DEF(umip_sldt, "mov $0xffff, %eax\n", ".byte 0x0f,0x00,0xc0\n" FP_ST("%eax", 0), "")
FP_DEF(umip_str, "mov $0xffff, %eax\n", ".byte 0x0f,0x00,0xc8\n" FP_ST("%eax", 0), "")

/* ---- #GP that is not a privileged opcode -------------------------------------------------- */
FP_DEF(gp_movaps_misaligned, FP_LEAO(fp_scratch, 4, FP_AX), ".byte 0x0f,0x28,0x00", "")
FP_DEF(gp_ldmxcsr_reserved, "", "ldmxcsr " FP_G(fp_mx_bad), "")
FP_DEF(gp_fxsave_misaligned, FP_LEAO(fp_scratch, 8, FP_AX), ".byte 0x0f,0xae,0x00", "")
FP_DEF(gp_xsave_misaligned, FP_LEAO(fp_scratch, 8, FP_CX) "mov $3, %eax\n xor %edx, %edx\n", ".byte 0x0f,0xae,0x21", "")
FP_DEF(gp_xrstor_bad_header, FP_LEA(fp_scratch, FP_CX) "mov $0xffffffff, %eax\n mov $0xffffffff, %edx\n",
       ".byte 0x0f,0xae,0x29", "")
FP_DEF(gp_int_0e, "", ".byte 0xcd,0x0e", "")
/* INT 80h / INT 21h: DPL0 (absent) gates on Windows -> #GP. A Linux-hosted layer may run INT 80h as a
 * Linux system call instead (EAX=20 = i386 getpid: harmless if it does). */
FP_DEF(gp_int_80, "mov $20, %eax\n", ".byte 0xcd,0x80\n" FP_ST("%eax", 0), "")
FP_DEF(gp_int_21, "mov $0x3000, %eax\n", ".byte 0xcd,0x21\n" FP_ST("%eax", 0), "")
/* Alignment check: EFLAGS.AC=1 and a misaligned load; #AC only if the OS sets CR0.AM (recorded only).
 * AC is cleared by the probe's cleanup on the resume path. */
FP_DEF(ac_misaligned_load, FP_LEAO(fp_scratch, 1, FP_AX) "pushf\n orl $0x40000, (" FP_SP ")\n popf\n",
       "mov (" FP_AX "), %ecx\n" FP_ST("%ecx", 0), "pushf\n andl $0xfffbffff, (" FP_SP ")\n popf\n")
#ifdef T_X64
FP_DEF(gp_cmpxchg16b_misaligned, "lea fp_scratch+8(%rip), %r8\n xor %eax, %eax\n xor %edx, %edx\n xor %ebx, %ebx\n xor %ecx, %ecx\n",
       ".byte 0x49,0x0f,0xc7,0x08", "")
FP_DEF(ok_cmpxchg16b_aligned, "lea fp_scratch+64(%rip), %r8\n xor %eax, %eax\n xor %edx, %edx\n mov $1, %ebx\n xor %ecx, %ecx\n",
       ".byte 0x49,0x0f,0xc7,0x08\n" FP_ST("%rbx", 0), "")
FP_DEF(gp_noncanonical_read, "movabs $0x8000000000000000, %rax\n", ".byte 0x48,0x8b,0x00", "")
#endif

/* ---- page faults --------------------------------------------------------------------------- */
FP_DEF(av_null_read, "xor %eax, %eax\n", ".byte 0x8b,0x00", "")
FP_DEF(av_null_write, "xor %eax, %eax\n", ".byte 0x89,0x08", "")
FP_DEF(av_nx_exec, "", "jmp *" FP_G(fp_nx_target), "")

/* ---- traps / software interrupts ----------------------------------------------------------- */
FP_DEF(trap_int3, "", ".byte 0xcc", "")
FP_DEF(trap_int3_long, "", ".byte 0xcd,0x03", "")
FP_DEF(trap_int2c, "", ".byte 0xcd,0x2c", "")
FP_DEF(trap_int2d, "xor %eax, %eax\n", ".byte 0xcd,0x2d\n nop\n nop", "")
FP_DEF(trap_fastfail, "mov $7, %ecx\n", ".byte 0xcd,0x29", "")
FP_DEF(trap_icebp, "", ".byte 0xf1\n nop", "")
FP_DEF(trap_single_step, "pushf\n orl $0x100, (" FP_SP ")\n popf\n", "nop\n nop\n nop", "")
FP_DEF(trap_div0, "mov $1, %eax\n xor %edx, %edx\n xor %ecx, %ecx\n", ".byte 0xf7,0xf1", "")     /* div ecx */
FP_DEF(trap_div_overflow, "xor %eax, %eax\n mov $2, %edx\n mov $1, %ecx\n", ".byte 0xf7,0xf1", "")
FP_DEF(trap_idiv_overflow, "mov $0x80000000, %eax\n cltd\n mov $-1, %ecx\n", ".byte 0xf7,0xf9", "") /* idiv ecx */
#ifndef T_X64
FP_DEF(trap_into_x86, "mov $0x7fffffff, %eax\n add $1, %eax\n", ".byte 0xce\n nop", "")
FP_DEF(ok_into_no_of_x86, "xor %eax, %eax\n", ".byte 0xce\n nop", "")
FP_DEF(trap_bound_x86, "mov $20, %eax\n lea _fp_bounds, %ecx\n", ".byte 0x62,0x01", "")
FP_DEF(ok_bound_in_range_x86, "mov $5, %eax\n lea _fp_bounds, %ecx\n", ".byte 0x62,0x01", "")
FP_DEF(sys_sysenter_x86, "push $_fp_sys_sysenter_x86_res\n mov %esp, %edx\n mov $0x0fff, %eax\n", ".byte 0x0f,0x34", "")
#endif

/* ---- system-call entry with an invalid service number (no fault expected on Windows) ------ */
#ifdef T_X64
FP_DEF(sys_syscall_badsvc, "mov $0x0fff, %eax\n mov %rsp, %r10\n", ".byte 0x0f,0x05\n" FP_ST("%rax", 0), "")
FP_DEF(sys_int2e_badsvc, "mov $0x0fff, %eax\n mov %rsp, %r10\n mov %rsp, %rdx\n", ".byte 0xcd,0x2e\n" FP_ST("%rax", 0), "")
#else
FP_DEF(sys_int2e_badsvc, "mov $0x0fff, %eax\n mov %esp, %edx\n", ".byte 0xcd,0x2e\n" FP_ST("%eax", 0), "")
#endif

/* ---- unmasked x87 exceptions (#MF at the next waiting instruction) ------------------------- */
FP_DEF(x87u_stack, "fninit\n fldcw " FP_G(fp_cw_im) "\n fadd %st(1), %st\n", "fwait", "")
FP_DEF(x87u_invalid, "fninit\n fldcw " FP_G(fp_cw_im) "\n fld1\n fchs\n fsqrt\n", "fwait", "")
FP_DEF(x87u_zerodiv, "fninit\n fldcw " FP_G(fp_cw_zm) "\n fldz\n fld1\n fdiv %st(1), %st\n", "fwait", "")
FP_DEF(x87u_overflow, "fninit\n fldcw " FP_G(fp_cw_om) "\n fldt " FP_G(fp_big80) "\n fmul %st(0), %st\n", "fwait", "")
FP_DEF(x87u_underflow, "fninit\n fldcw " FP_G(fp_cw_um) "\n fldt " FP_G(fp_tiny80) "\n fmul %st(0), %st\n", "fwait", "")
FP_DEF(x87u_precision, "fninit\n fldcw " FP_G(fp_cw_pm) "\n fldpi\n fmul %st(0), %st\n", "fwait", "")
FP_DEF(x87u_denormal, "fninit\n fldcw " FP_G(fp_cw_dm) "\n fldl " FP_G(fp_den64) "\n", "fwait", "")

/* ---- unmasked SSE exceptions (#XM, a fault at the SIMD instruction) ------------------------ */
FP_DEF(sseu_invalid, "ldmxcsr " FP_G(fp_mx_im) "\n movss " FP_GO(fp_f32, 4) ", %xmm0\n", "sqrtss %xmm0, %xmm0", "")
FP_DEF(sseu_zerodiv, "ldmxcsr " FP_G(fp_mx_zm) "\n movss " FP_G(fp_f32) ", %xmm0\n xorps %xmm1, %xmm1\n", "divss %xmm1, %xmm0", "")
FP_DEF(sseu_denormal, "ldmxcsr " FP_G(fp_mx_dm) "\n movss " FP_GO(fp_f32, 8) ", %xmm0\n", "addss %xmm0, %xmm0", "")
FP_DEF(sseu_overflow, "ldmxcsr " FP_G(fp_mx_om) "\n movss " FP_GO(fp_f32, 12) ", %xmm0\n", "mulss %xmm0, %xmm0", "")
FP_DEF(sseu_underflow, "ldmxcsr " FP_G(fp_mx_um) "\n movss " FP_GO(fp_f32, 16) ", %xmm0\n", "mulss %xmm0, %xmm0", "")
FP_DEF(sseu_precision, "ldmxcsr " FP_G(fp_mx_pm) "\n movss " FP_G(fp_f32) ", %xmm0\n movss " FP_GO(fp_f32, 20) ", %xmm1\n",
       "divss %xmm1, %xmm0", "")

/* ---- exception-context integrity --------------------------------------------------------- */
#ifdef T_X64
FP_DEF(ctx_gpr_xmm,
       "movdqu %xmm6, fp_xmm6_save(%rip)\n movdqu %xmm15, fp_xmm15_save(%rip)\n"
       "movdqu fp_pat(%rip), %xmm6\n movdqu fp_pat+16(%rip), %xmm15\n"
       "ldmxcsr fp_mx_ctx(%rip)\n fldcw fp_cw_ctx(%rip)\n"
       "movabs $0x5A5A0000C0DE00C1, %rcx\n movabs $0x5A5A0000C0DE00D1, %rdx\n"
       "movabs $0x5A5A0000C0DE0008, %r8\n movabs $0x5A5A0000C0DE0009, %r9\n"
       "movabs $0x5A5A0000C0DE0010, %r10\n movabs $0x5A5A0000C0DE0011, %r11\n",
       "ud2", "movdqu fp_xmm6_save(%rip), %xmm6\n movdqu fp_xmm15_save(%rip), %xmm15\n")   /* XMM6/15 are callee-saved */
FP_DEF(ctx_ymm_ud2,
       "movdqu %xmm6, fp_xmm6_save(%rip)\n vmovdqu fp_pat(%rip), %ymm6\n", "ud2",
       "vzeroupper\n movdqu fp_xmm6_save(%rip), %xmm6\n")
#else
FP_DEF(ctx_gpr_xmm,
       "movdqu _fp_pat, %xmm6\n ldmxcsr _fp_mx_ctx\n fldcw _fp_cw_ctx\n"
       "mov $0x5A5AC1C1, %ecx\n mov $0x5A5AD1D1, %edx\n mov $0x5A5AB1B1, %ebx\n mov $0x5A5A5151, %esi\n mov $0x5A5AD7D7, %edi\n",
       "ud2", "pxor %xmm6, %xmm6\n")
FP_DEF(ctx_ymm_ud2, "vmovdqu _fp_pat, %ymm6\n", "ud2", "vzeroupper\n")
#endif

/* =============================================================================================
 * Child: run one probe under a vectored handler
 * ============================================================================================= */
static const fp_probe *volatile g_cur;
static volatile LONG g_hits;
static DWORD g_code, g_eflags_rec, g_np;
static long long g_ea, g_ip;
static unsigned long long g_p0, g_p1;
static int g_marker = -1, g_setup, g_ymm = -2;
static unsigned g_fsw, g_mxcsr, g_ctxbad;
static unsigned long g_cflags;
static int g_ctxmode;     /* 1 = GPR/XMM/MXCSR/FCW check, 2 = YMM check */
typedef PVOID (WINAPI *pfn_locate)(PCONTEXT, DWORD, PDWORD);
static pfn_locate g_locate;

#ifdef T_X64
#define CTX_IP(c) ((c)->Rip)
#else
#define CTX_IP(c) ((c)->Eip)
#endif

static unsigned fp_ctx_check(CONTEXT *c) {
    unsigned bad = 0;
#ifdef T_X64
    if (c->Rcx != 0x5A5A0000C0DE00C1ull) bad |= 1;
    if (c->Rdx != 0x5A5A0000C0DE00D1ull) bad |= 2;
    if (c->R8 != 0x5A5A0000C0DE0008ull || c->R9 != 0x5A5A0000C0DE0009ull ||
        c->R10 != 0x5A5A0000C0DE0010ull || c->R11 != 0x5A5A0000C0DE0011ull) bad |= 4;
    if (!(c->ContextFlags & 0x8)) bad |= 0x100;   /* CONTEXT_FLOATING_POINT bit */
    else {
        if (memcmp(&c->Xmm6, fp_pat, 16)) bad |= 8;
        if (memcmp(&c->Xmm15, fp_pat + 16, 16)) bad |= 0x10;
        if (c->MxCsr != fp_mx_ctx) bad |= 0x20;
        if (c->FltSave.MxCsr != fp_mx_ctx) bad |= 0x40;
        if (c->FltSave.ControlWord != fp_cw_ctx) bad |= 0x80;
    }
#else
    if (c->Ecx != 0x5A5AC1C1u) bad |= 1;
    if (c->Edx != 0x5A5AD1D1u) bad |= 2;
    if (c->Ebx != 0x5A5AB1B1u || c->Esi != 0x5A5A5151u || c->Edi != 0x5A5AD7D7u) bad |= 4;
    if ((c->ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS) {
        if (memcmp(c->ExtendedRegisters + 160 + 16 * 6, fp_pat, 16)) bad |= 8;
        uint32_t mx; memcpy(&mx, c->ExtendedRegisters + 24, 4);
        if (mx != fp_mx_ctx) bad |= 0x20;
    } else bad |= 0x200;   /* no extended registers in the context: XMM/MXCSR not checkable */
    if ((c->ContextFlags & CONTEXT_FLOATING_POINT) == CONTEXT_FLOATING_POINT) {
        if ((c->FloatSave.ControlWord & 0xFFFF) != fp_cw_ctx) bad |= 0x80;
    } else bad |= 0x100;
#endif
    return bad;
}

static int fp_ymm_check(CONTEXT *c) {
    if ((c->ContextFlags & CONTEXT_XSTATE) != CONTEXT_XSTATE || !g_locate) return -1;
    DWORD len = 0;
    uint8_t *up = (uint8_t *)g_locate(c, 2 /* XSTATE_AVX */, &len);
    if (!up || len < 7 * 16) return -1;
    return memcmp(up + 6 * 16, fp_pat + 16, 16) == 0;
}

static LONG CALLBACK fp_veh(EXCEPTION_POINTERS *ep) {
    const fp_probe *p = g_cur;
    if (!p) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord;
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    uintptr_t ip = (uintptr_t)CTX_IP(c), ea = (uintptr_t)er->ExceptionAddress;
    uintptr_t lo = (uintptr_t)p->fn, hi = (uintptr_t)p->res, insn = (uintptr_t)p->insn;
    uintptr_t nx = fp_nx_target;
    int at_nx = nx && (ip == nx || ea == nx);
    if (!((ip >= lo && ip <= hi) || (ea >= lo && ea <= hi) || at_nx)) return EXCEPTION_CONTINUE_SEARCH;
    LONG n = InterlockedIncrement(&g_hits);
    if (n > 8) return EXCEPTION_CONTINUE_SEARCH;
    if (n == 1) {
        uintptr_t base = at_nx ? nx : insn;
        g_code = er->ExceptionCode;
        g_eflags_rec = er->ExceptionFlags;
        g_np = er->NumberParameters;
        g_p0 = er->NumberParameters > 0 ? (unsigned long long)er->ExceptionInformation[0] : 0;
        g_p1 = er->NumberParameters > 1 ? (unsigned long long)er->ExceptionInformation[1] : 0;
        g_ea = (long long)(intptr_t)(ea - base);
        g_ip = (long long)(intptr_t)(ip - base);
        g_setup = !at_nx && ea < insn && ip < insn;
        g_cflags = c->ContextFlags;
#ifdef T_X64
        g_marker = c->R12 == FP_M12 && c->R13 == FP_M13 && c->R14 == FP_M14 && c->R15 == FP_M15;
        g_fsw = c->FltSave.StatusWord;
        g_mxcsr = c->MxCsr;
#else
        g_marker = c->Ebp == FP_MBP;
        g_fsw = c->FloatSave.StatusWord & 0xFFFF;
        g_mxcsr = 0xFFFFFFFFu;
        if ((c->ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS) memcpy(&g_mxcsr, c->ExtendedRegisters + 24, 4);
#endif
        if (g_ctxmode == 1) g_ctxbad = fp_ctx_check(c);
        if (g_ctxmode == 2) g_ymm = fp_ymm_check(c);
    }
#ifdef T_X64
    c->Rip = hi;
#else
    c->Eip = (DWORD)hi;
#endif
    c->EFlags &= ~0x100u;   /* TF */
    return EXCEPTION_CONTINUE_EXECUTION;
}

static void fp_prepare(const char *name) {
    memset(fp_scratch, 0, sizeof fp_scratch);
    memset(fp_out, 0, sizeof fp_out);
    { uint32_t den = 1; memcpy(&fp_f32[2], &den, 4); }   /* smallest float denormal */
    t_pattern_reg(fp_pat, 32, 0xFAu, 6);
    t_pattern_reg(fp_pat + 32, 32, 0xFAu, 15);
    if (cpu()->osxsave) { uint64_t x = t_xgetbv(0); fp_arg[0] = (uint32_t)x; fp_arg[1] = (uint32_t)(x >> 32); }
    else { fp_arg[0] = 3; fp_arg[1] = 0; }
    if (!strcmp(name, "av_nx_exec")) {
        uint8_t *pg = (uint8_t *)VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (pg) { memset(pg, 0xC3, 4096); fp_nx_target = (uintptr_t)pg; }
    }
    if (!strcmp(name, "gp_xrstor_bad_header")) {
        uint64_t bv = 1ull << 62;                 /* XSTATE_BV bit 62: never a valid XCR0 bit */
        memcpy(fp_scratch + 512, &bv, 8);
    }
    if (!strcmp(name, "ctx_gpr_xmm")) g_ctxmode = 1;
    if (!strcmp(name, "ctx_ymm_ud2")) { g_ctxmode = 2; g_locate = (pfn_locate)t_proc("kernel32.dll", "LocateXStateFeature"); }
}

static DWORD WINAPI fp_thread_main(LPVOID arg) {
    const fp_probe *p = (const fp_probe *)arg;
    g_cur = p;
    p->fn();
    g_cur = NULL;
    return 0x600Du;
}

/* argv[1] = probe; optional argv[2]: "thread" (run it on a new thread) or "repeat=N" (run it N times in
 * a row; every run must give one exception with the same code and address). */
CHILD(probe) {
    if (argc < 2) { printf("FP_USAGE\n"); return 3; }
    const fp_probe *p = fp_find(argv[1]);
    if (!p) { printf("FP_UNKNOWN %s\n", argv[1]); return 3; }
    int on_thread = argc > 2 && !strcmp(argv[2], "thread");
    int repeat = argc > 2 && !strncmp(argv[2], "repeat=", 7) ? atoi(argv[2] + 7) : 0;
    if (repeat > 5000) repeat = 5000;
    fp_prepare(p->name);
    PVOID h = AddVectoredExceptionHandler(1, fp_veh);
    if (!h) { printf("FP_NOVEH\n"); return 4; }
    printf("FP_BEGIN %s\n", p->name);
    fflush(stdout);
    if (repeat > 0) {
        int good = 0, first_bad = -1;
        DWORD code0 = 0; long long ea0 = 0;
        for (int i = 0; i < repeat; i++) {
            g_hits = 0; g_code = 0; g_ea = 0; g_marker = -1;
            g_cur = p;
            p->fn();
            g_cur = NULL;
            if (i == 0) { code0 = g_code; ea0 = g_ea; }
            int ok = g_hits == 1 && g_code == code0 && g_ea == ea0 && g_marker == 1;
            if (ok) good++;
            else if (first_bad < 0) first_bad = i;
        }
        RemoveVectoredExceptionHandler(h);
        printf("FPR probe=%s iters=%d good=%d first_bad=%d code=0x%08lx ea=%lld\n", p->name, repeat, good, first_bad,
               (unsigned long)code0, ea0);
        fflush(stdout);
        return 0;
    }
    if (on_thread) {
        HANDLE th = CreateThread(NULL, 0, fp_thread_main, (LPVOID)p, 0, NULL);
        if (!th) { printf("FP_NOTHREAD %lu\n", GetLastError()); return 5; }
        DWORD w = WaitForSingleObject(th, 5000), tx = 0;
        GetExitCodeThread(th, &tx);
        CloseHandle(th);
        if (w != WAIT_OBJECT_0 || tx != 0x600Du) { printf("FP_THREAD_FAILED wait=%lu exit=0x%lx\n", w, tx); return 6; }
    } else {
        g_cur = p;
        p->fn();
        g_cur = NULL;
    }
    RemoveVectoredExceptionHandler(h);
    printf("FP probe=%s fault=%d code=0x%08lx xflags=0x%lx ea=%lld ip=%lld np=%lu p0=0x%llx p1=0x%llx "
           "marker=%d hits=%ld setup=%d fsw=0x%x mxcsr=0x%x cflags=0x%lx ctxbad=0x%x ymm=%d "
           "out0=0x%llx out1=0x%llx out2=0x%llx\n",
           p->name, g_hits > 0, (unsigned long)g_code, (unsigned long)g_eflags_rec, g_ea, g_ip, (unsigned long)g_np,
           g_p0, g_p1, g_marker, (long)g_hits, g_setup, g_fsw, g_mxcsr, g_cflags, g_ctxbad, g_ymm,
           (unsigned long long)fp_out[0], (unsigned long long)fp_out[1], (unsigned long long)fp_out[2]);
    fflush(stdout);
    return 0;
}

/* =============================================================================================
 * Parent side: run, parse, compare
 * ============================================================================================= */
typedef struct fp_res {
    int line, began, timed_out, fault, marker, setup, ymm;
    DWORD exit_code, exc_code, code;
    unsigned long xflags, np, cflags;
    long hits;
    long long ea, ip;
    unsigned long long p0, p1, out0, out1, out2;
    unsigned fsw, mxcsr, ctxbad;
    char text[600];
} fp_res;
static t_child_result g_cr;

static unsigned long long fp_kv(const char *s, const char *key, int *found) {
    char k[40]; snprintf(k, sizeof k, " %s=", key);
    const char *p = strstr(s, k);
    if (!p) { if (found) *found = 0; return 0; }
    if (found) *found = 1;
    p += strlen(k);
    return (unsigned long long)strtoull(p, NULL, 0);
}
static long long fp_kvs(const char *s, const char *key) {
    char k[40]; snprintf(k, sizeof k, " %s=", key);
    const char *p = strstr(s, k);
    return p ? strtoll(p + strlen(k), NULL, 0) : 0;
}

static void fp_run_mode(const char *probe, const char *mode, DWORD tmo, fp_res *r) {
    memset(r, 0, sizeof *r);
    REQUIREF(fp_find(probe), "probe '%s' is not built into this exe", probe);
    char args[160];
    snprintf(args, sizeof args, "%s%s%s", probe, mode ? " " : "", mode ? mode : "");
    REQUIREF(t_child_run("probe", args, tmo, &g_cr), "child did not start (error %lu)", g_cr.spawn_error);
    r->timed_out = g_cr.timed_out;
    r->exit_code = g_cr.exit_code;
    r->exc_code = g_cr.exc_code;
    char v[600];
    r->began = t_child_find(&g_cr, "FP_BEGIN ", v, sizeof v);
    if (t_child_find(&g_cr, "FP ", v, sizeof v)) {
        char s[620]; snprintf(s, sizeof s, " %s", v);
        r->line = 1;
        snprintf(r->text, sizeof r->text, "%s", v);
        r->fault = (int)fp_kv(s, "fault", NULL);
        r->code = (DWORD)fp_kv(s, "code", NULL);
        r->xflags = (unsigned long)fp_kv(s, "xflags", NULL);
        r->ea = fp_kvs(s, "ea");
        r->ip = fp_kvs(s, "ip");
        r->np = (unsigned long)fp_kv(s, "np", NULL);
        r->p0 = fp_kv(s, "p0", NULL);
        r->p1 = fp_kv(s, "p1", NULL);
        r->marker = (int)fp_kvs(s, "marker");
        r->hits = (long)fp_kvs(s, "hits");
        r->setup = (int)fp_kvs(s, "setup");
        r->fsw = (unsigned)fp_kv(s, "fsw", NULL);
        r->mxcsr = (unsigned)fp_kv(s, "mxcsr", NULL);
        r->cflags = (unsigned long)fp_kv(s, "cflags", NULL);
        r->ctxbad = (unsigned)fp_kv(s, "ctxbad", NULL);
        r->ymm = (int)fp_kvs(s, "ymm");
        r->out0 = fp_kv(s, "out0", NULL);
        r->out1 = fp_kv(s, "out1", NULL);
        r->out2 = fp_kv(s, "out2", NULL);
        t_note("%s", v);
    } else {
        snprintf(r->text, sizeof r->text, "no result line: exit=0x%08lx exception=0x%08lx (%s) timed_out=%d",
                 (unsigned long)r->exit_code, (unsigned long)r->exc_code, fp_code_name(r->exc_code), r->timed_out);
        t_note("%s: %s; output: %.300s", probe, r->text, g_cr.out);
    }
    if (r->fault) {
        t_info("exception", "0x%08lx %s", (unsigned long)r->code, fp_code_name(r->code));
        t_info("ea_offset", "%lld", r->ea);
    } else if (r->line) {
        t_info("exception", "none");
    }
}
static void fp_run(const char *probe, fp_res *r) { fp_run_mode(probe, NULL, 8000, r); }

/* Child started, did not hang, reached the vectored handler (or ran through) and exited normally. */
static int fp_survived(const char *probe, const fp_res *r) {
    CHECKF(!r->timed_out, "%s: child timed out", probe);
    CHECKF(r->began, "%s: child did not reach the probe", probe);
    CHECKF(r->line, "%s: child died before printing its result (%s): the exception was not delivered to the "
           "vectored handler, or the redirected context could not be continued", probe, r->text);
    if (!r->line) return 0;
    CHECKF(r->exit_code == 0, "%s: child exit code 0x%08lx after the probe", probe, (unsigned long)r->exit_code);
    return 1;
}

enum { FX_EA = 1, FX_IPEA = 2, FX_MARK = 4, FX_ONE = 8, FX_FAULT = FX_EA | FX_IPEA | FX_MARK | FX_ONE };
/* Expect an exception with one of codes[0..n). ea_want = expected ExceptionAddress - insn label. */
static int fp_expect_fault(const char *probe, const fp_res *r, const DWORD *codes, int n, long long ea_want, int fx) {
    if (!fp_survived(probe, r)) return 0;
    CHECKF(r->fault, "%s: executed without raising an exception (expected %s)", probe, fp_code_name(codes[0]));
    if (!r->fault) return 0;
    CHECKF(!r->setup, "%s: faulted in the setup code before the probed instruction (code 0x%08lx)", probe, (unsigned long)r->code);
    int ok = 0;
    for (int i = 0; i < n; i++) ok |= r->code == codes[i];
    char want[200] = "";
    for (int i = 0; i < n; i++) {
        size_t l = strlen(want);
        snprintf(want + l, sizeof want - l, "%s%s", i ? " or " : "", fp_code_name(codes[i]));
    }
    CHECKF(ok, "%s: exception 0x%08lx (%s), expected %s", probe, (unsigned long)r->code, fp_code_name(r->code), want);
    if (fx & FX_EA) CHECKF(r->ea == ea_want, "%s: ExceptionAddress = insn%+lld, expected insn%+lld", probe, r->ea, ea_want);
    if (fx & FX_IPEA) CHECKF(r->ip == r->ea, "%s: context IP (insn%+lld) != ExceptionAddress (insn%+lld)", probe, r->ip, r->ea);
    if (fx & FX_MARK) CHECKF(r->marker == 1, "%s: callee-saved marker register(s) wrong in the exception context", probe);
    if (fx & FX_ONE) CHECKF(r->hits == 1, "%s: %ld exceptions for one instruction", probe, r->hits);
    /* A processor exception is dispatched as a continuable exception: ExceptionFlags = 0. */
    CHECKF(r->xflags == 0, "%s: ExceptionFlags 0x%lx, expected 0 (continuable hardware exception)", probe, r->xflags);
    return ok;
}
static void fp_expect_code(const char *probe, DWORD code) {
    static fp_res r; fp_run(probe, &r);
    DWORD c[1] = { code };
    fp_expect_fault(probe, &r, c, 1, 0, FX_FAULT);
}
static void fp_expect_nofault(const char *probe, fp_res *r) {
    fp_run(probe, r);
    if (!fp_survived(probe, r)) return;
    CHECKF(!r->fault, "%s: raised 0x%08lx (%s) at insn%+lld; the instruction is defined to execute here",
           probe, (unsigned long)r->code, fp_code_name(r->code), r->ea);
    CHECKF(r->hits == 0, "%s: vectored handler ran %ld times", probe, r->hits);
}

/* ---- #UD -------------------------------------------------------------------------------- */
static void fp_ud(const char *probe) { fp_expect_code(probe, S_ILL); }

TEST(ud_ud2)              { fp_ud("ud_ud2"); }
TEST(ud_ud1)              { fp_ud("ud_ud1"); }
TEST(ud_ud0)              { fp_ud("ud_ud0"); }
TEST(ud_rsm_outside_smm)  { fp_ud("ud_rsm"); }
TEST(ud_lock_add_reg_dest){ fp_ud("ud_lock_add_reg"); }
TEST(ud_lock_nop)         { fp_ud("ud_lock_nop"); }
TEST(ud_lock_cmpxchg_reg_dest) { fp_ud("ud_lock_cmpxchg_reg"); }
TEST(ud_lock_mov_mem)     { fp_ud("ud_lock_mov_mem"); }
TEST(lock_add_mem_is_legal) {
    static fp_res r; fp_expect_nofault("ok_lock_add_mem", &r);
}
TEST(ud_lea_register_source) { fp_ud("ud_lea_reg"); }
TEST(ud_cmpxchg8b_register)  { fp_ud("ud_cmpxchg8b_reg"); }
TEST(ud_movntdq_register)    { fp_ud("ud_movntdq_reg"); }

static void fp_vex_ud(const char *probe) {
    if (!cpu_has(F_AVX)) t_note("AVX not usable: the encoding #UDs as an absent feature as well");
    fp_ud(probe);
}
TEST(ud_vex_after_66_prefix)   { fp_vex_ud("ud_vex_66_prefix"); }
TEST(ud_vex_after_f3_prefix)   { fp_vex_ud("ud_vex_f3_prefix"); }
TEST(ud_vex_after_lock_prefix) { fp_vex_ud("ud_vex_lock_prefix"); }
TEST(ud_vex_vmovd_l1)          { fp_vex_ud("ud_vex_l1_vmovd"); }
TEST(ud_vex_vmovdqa_vvvv)      { fp_vex_ud("ud_vex_vvvv_vmovdqa"); }
TEST(ud_vex_kmovw_l1)          { fp_vex_ud("ud_vex_kmovw_l1"); }
TEST(ud_evex_reserved_ll11)    { fp_vex_ud("ud_evex_ll11"); }
TEST(ud_movbe_register_form) {
    /* 0F 38 F0 is MOVBE Gy,My (memory-only operand); the register form is not listed in the #UD conditions of
     * the MOVBE page, so a processor that executes it is only recorded. If it faults it must be #UD. */
    static fp_res r; fp_run("ud_movbe_reg", &r);
    if (!fp_survived("ud_movbe_reg", &r)) return;
    if (!r.fault) { t_note("0F 38 F0 C0 (MOVBE register form) executed without a fault"); return; }
    DWORD c[1] = { S_ILL };
    fp_expect_fault("ud_movbe_reg", &r, c, 1, 0, FX_FAULT);
}
#ifdef T_X64
TEST(ud_vex_after_rex_prefix)  { fp_vex_ud("ud_vex_rex_prefix"); }
TEST(ud_x64_daa)      { fp_ud("ud_x64_daa"); }
TEST(ud_x64_aaa)      { fp_ud("ud_x64_aaa"); }
TEST(ud_x64_aam)      { fp_ud("ud_x64_aam"); }
TEST(ud_x64_pusha)    { fp_ud("ud_x64_pusha"); }
TEST(ud_x64_into)     { fp_ud("ud_x64_into"); }
TEST(ud_x64_push_es)  { fp_ud("ud_x64_push_es"); }
TEST(ud_x64_jmp_far)  { fp_ud("ud_x64_jmp_far"); }
TEST(ud_cmpxchg16b_register) { fp_ud("ud_cmpxchg16b_reg"); }
#else
TEST(ud_vex_after_rex_prefix) { SKIP("x64 only (0x40 is INC EAX in 32-bit mode)"); }
TEST(ud_x64_invalid_opcodes)  { SKIP("x64 only: DAA/AAA/AAM/PUSHA/INTO/PUSH ES/JMP FAR are valid in 32-bit mode"); }
#endif

/* ---- ISA extension absent or OS-disabled -------------------------------------------------- */
/* Hard expectation when the processor cannot execute the encoding: the XCR0 state it needs is
 * disabled (hardware-enforced), or there is no hypervisor so CPUID is the hardware truth. Under a
 * hypervisor that only hides the CPUID bit the instruction may still execute: NOTE, not FAIL. */
static void fp_feat(const char *probe, t_feature f, int f2, uint64_t xmask, const char *what) {
    int usable = cpu_has(f) && (f2 < 0 || cpu_has((t_feature)f2));
    if (usable) SKIP("%s is usable here: the instruction is legal", what);
    const t_cpu_info *c = cpu();
    int xoff = xmask && (!c->osxsave || (c->xcr0 & xmask) != xmask);
    int hw = cpu_hw(f) && (f2 < 0 || cpu_hw((t_feature)f2));
    /* AMD family 17h (Zen 1) is known to execute FMA4 (and other AMD-legacy encodings) although CPUID no
     * longer enumerates them: on AMD a "hidden but implemented" result is a NOTE even on bare metal. */
    int amd_legacy = c->is_amd && (f == F_FMA4 || f == F_XOP || f == F_AMD3DNOW || f == F_TBM || f == F_SSE4A);
    int hard = xoff || (!c->hypervisor && !amd_legacy);
    t_info("mode", "%s", hw ? "os-disabled (CPUID set, XCR0 state off)" : xoff ? "absent (and XCR0 state off)" : "absent");
    static fp_res r;
    fp_run(probe, &r);
    if (!fp_survived(probe, &r)) return;
    if (!r.fault && !hard) {
        t_note("%s executed although CPUID reports it absent: %s", what,
               amd_legacy ? "AMD processors may still implement de-enumerated legacy extensions"
                          : "a hypervisor hides the bit but the processor implements it");
        return;
    }
    DWORD c1[1] = { S_ILL };
    fp_expect_fault(probe, &r, c1, 1, 0, FX_FAULT);
}
#define XS_YMM T_XS_YMM_STATE
#define XS_ZMM T_XS_ZMM_STATE
#define XS_AMX T_XS_AMX_STATE
TEST(feat_avx_vaddps_ymm)        { fp_feat("feat_avx_vaddps_ymm", F_AVX, -1, XS_YMM, "AVX"); }
TEST(feat_avx_vzeroupper)        { fp_feat("feat_avx_vzeroupper", F_AVX, -1, XS_YMM, "AVX"); }
TEST(feat_avx2_vpaddd_ymm)       { fp_feat("feat_avx2_vpaddd_ymm", F_AVX2, -1, XS_YMM, "AVX2"); }
TEST(feat_fma_vfmadd213ps)       { fp_feat("feat_fma_vfmadd213ps", F_FMA, -1, XS_YMM, "FMA"); }
TEST(feat_f16c_vcvtph2ps)        { fp_feat("feat_f16c_vcvtph2ps", F_F16C, -1, XS_YMM, "F16C"); }
TEST(feat_avx512f_vaddps_zmm)    { fp_feat("feat_avx512f_vaddps_zmm", F_AVX512F, -1, XS_ZMM, "AVX-512F"); }
TEST(feat_avx512f_kmovw)         { fp_feat("feat_avx512f_kmovw_kreg", F_AVX512F, -1, XS_ZMM, "AVX-512F opmask"); }
TEST(feat_avx512vl_vpternlogd_xmm) { fp_feat("feat_avx512vl_vpternlogd", F_AVX512F, F_AVX512VL, XS_ZMM, "AVX-512VL"); }
TEST(feat_avx512bw_vpaddb_zmm)   { fp_feat("feat_avx512bw_vpaddb_zmm", F_AVX512BW, -1, XS_ZMM, "AVX-512BW"); }
TEST(feat_avx512dq_vpmullq_zmm)  { fp_feat("feat_avx512dq_vpmullq_zmm", F_AVX512DQ, -1, XS_ZMM, "AVX-512DQ"); }
TEST(feat_avx512cd_vplzcntd_zmm) { fp_feat("feat_avx512cd_vplzcntd_zmm", F_AVX512CD, -1, XS_ZMM, "AVX-512CD"); }
TEST(feat_avx512vnni_vpdpbusd)   { fp_feat("feat_avx512vnni_vpdpbusd", F_AVX512VNNI, -1, XS_ZMM, "AVX-512 VNNI"); }
TEST(feat_avx512fp16_vaddph)     { fp_feat("feat_avx512fp16_vaddph", F_AVX512FP16, -1, XS_ZMM, "AVX-512 FP16"); }
TEST(feat_amx_tilerelease)       { fp_feat("feat_amx_tilerelease", F_AMX_TILE, -1, XS_AMX, "AMX-TILE"); }
TEST(feat_amx_tilezero)          { fp_feat("feat_amx_tilezero", F_AMX_TILE, -1, XS_AMX, "AMX-TILE"); }
TEST(feat_bmi1_andn)             { fp_feat("feat_bmi1_andn", F_BMI1, -1, 0, "BMI1"); }
TEST(feat_bmi2_pdep)             { fp_feat("feat_bmi2_pdep", F_BMI2, -1, 0, "BMI2"); }
TEST(feat_vaes_ymm)              { fp_feat("feat_vaes_ymm", F_VAES, F_AVX, XS_YMM, "VAES (VEX.256)"); }
TEST(feat_vpclmulqdq_ymm)        { fp_feat("feat_vpclmul_ymm", F_VPCLMULQDQ, F_AVX, XS_YMM, "VPCLMULQDQ (VEX.256)"); }
TEST(feat_avx_vnni_vpdpbusd)     { fp_feat("feat_avx_vnni_vpdpbusd", F_AVX_VNNI, -1, XS_YMM, "AVX-VNNI"); }
TEST(feat_fma4_vfmaddps)         { fp_feat("feat_fma4_vfmaddps", F_FMA4, -1, XS_YMM, "FMA4"); }
TEST(feat_xop_vpcmov)            { fp_feat("feat_xop_vex_vpcmov", F_XOP, -1, XS_YMM, "XOP"); }
TEST(feat_adx_adcx)              { fp_feat("feat_adx_adcx", F_ADX, -1, 0, "ADX"); }
TEST(feat_aes_aesenc)            { fp_feat("feat_aes_aesenc", F_AES, -1, 0, "AES-NI"); }
TEST(feat_pclmulqdq)             { fp_feat("feat_pclmulqdq", F_PCLMULQDQ, -1, 0, "PCLMULQDQ"); }
TEST(feat_sha1rnds4)             { fp_feat("feat_sha1rnds4", F_SHA, -1, 0, "SHA"); }
TEST(feat_sha256rnds2)           { fp_feat("feat_sha256rnds2", F_SHA, -1, 0, "SHA"); }
TEST(feat_gfni_gf2p8mulb)        { fp_feat("feat_gfni_gf2p8mulb", F_GFNI, -1, 0, "GFNI"); }
TEST(feat_sse42_crc32)           { fp_feat("feat_sse42_crc32", F_SSE42, -1, 0, "SSE4.2"); }
TEST(feat_popcnt)                { fp_feat("feat_popcnt", F_POPCNT, -1, 0, "POPCNT"); }
TEST(feat_movbe)                 { fp_feat("feat_movbe", F_MOVBE, -1, 0, "MOVBE"); }
TEST(feat_rdrand)                { fp_feat("feat_rdrand", F_RDRAND, -1, 0, "RDRAND"); }
TEST(feat_rdseed)                { fp_feat("feat_rdseed", F_RDSEED, -1, 0, "RDSEED"); }
TEST(feat_rdtscp_absent)         { fp_feat("feat_rdtscp", F_RDTSCP, -1, 0, "RDTSCP"); }
TEST(feat_movdiri)               { fp_feat("feat_movdiri", F_MOVDIRI, -1, 0, "MOVDIRI"); }
TEST(feat_3dnow_femms)           { fp_feat("feat_3dnow_femms", F_AMD3DNOW, -1, 0, "3DNow!"); }
TEST(feat_sse4a_extrq)           { fp_feat("feat_sse4a_extrq", F_SSE4A, -1, 0, "SSE4a"); }
/* XCR0 bit 0 is always set when OSXSAVE is: xmask 1 makes "OSXSAVE clear" a hardware-enforced #UD. */
TEST(feat_xsaveopt_absent)       { fp_feat("feat_xsaveopt", F_XSAVEOPT, -1, 1, "XSAVEOPT"); }
TEST(feat_xsavec_absent)         { fp_feat("feat_xsavec", F_XSAVEC, -1, 1, "XSAVEC"); }
TEST(feat_xgetbv_without_osxsave) {
    if (cpu()->osxsave) SKIP("CR4.OSXSAVE is set: XGETBV is legal");
    fp_ud("feat_xgetbv_no_osxsave");
}
TEST(feat_xsave_without_osxsave) {
    if (cpu()->osxsave) SKIP("CR4.OSXSAVE is set: XSAVE is legal");
    if (!cpu_hw(F_XSAVE)) t_note("XSAVE not implemented: #UD as an absent feature");
    fp_ud("feat_xsave_no_osxsave");
}

/* ---- encodings with defined fallback behaviour (no fault) --------------------------------- */
TEST(lzcnt_or_bsr_semantics) {
    static fp_res r; fp_expect_nofault("nf_lzcnt_or_bsr", &r);
    if (!r.line || r.fault) return;
    uint32_t v = (uint32_t)r.out0;
    t_info("result", "%u", v);
    if (cpu_hw(F_LZCNT)) CHECKF(v == 27, "LZCNT(0x10) = %u, expected 27", v);
    else if (!cpu()->hypervisor) CHECKF(v == 4, "F3 0F BD without LZCNT executes as BSR: BSR(0x10) = %u, expected 4", v);
    else CHECKF(v == 4 || v == 27, "F3 0F BD (0x10) = %u: neither BSR (4) nor LZCNT (27)", v);
}
TEST(tzcnt_or_bsf_semantics) {
    static fp_res r; fp_expect_nofault("nf_tzcnt_or_bsf", &r);
    if (!r.line || r.fault) return;
    uint32_t v = (uint32_t)r.out0, fl = (uint32_t)r.out1;
    t_info("result", "%u flags=0x%x", v, fl);
    if (cpu_hw(F_BMI1)) {
        CHECKF(v == 32, "TZCNT(0) = %u, expected 32 (operand size)", v);
        CHECKF(fl & 1, "TZCNT(0) must set CF (flags 0x%x)", fl);
    } else if (!cpu()->hypervisor) {
        CHECKF(fl & 0x40, "F3 0F BC without BMI1 executes as BSF: BSF(0) must set ZF (flags 0x%x)", fl);
    }
}
TEST(cldemote_is_a_hint_nop) {
    static fp_res r; fp_expect_nofault("nf_cldemote_hint", &r);
    if (!cpu_hw(F_CLDEMOTE)) t_note("CLDEMOTE absent: 0F 1C /0 executes as a reserved-NOP hint, as defined");
}
TEST(endbr64_is_a_nop)  { static fp_res r; fp_expect_nofault("nf_endbr64", &r); }
TEST(endbr32_is_a_nop)  { static fp_res r; fp_expect_nofault("nf_endbr32", &r); }
TEST(pause_executes)    { static fp_res r; fp_expect_nofault("nf_pause", &r); }
TEST(prefetchw_never_faults) {
    /* Architectural when CPUID.80000001h:ECX[8]; otherwise Windows emulates 0F 0D /1 as a NOP (KiOp_PREFETCH3). */
    if (!cpu_hw(F_PREFETCHW)) t_note("PREFETCHW not enumerated: relying on the OS to skip it");
    static fp_res r; fp_expect_nofault("nf_prefetchw", &r);
}
TEST(rdtsc_allowed_in_user_mode) {
    if (!cpu_hw(F_TSC)) SKIP("no TSC");
    static fp_res r; fp_expect_nofault("nf_rdtsc", &r);
    if (r.line && !r.fault) { CHECKF(r.out0 != 0, "RDTSC returned 0"); t_info("tsc", "0x%llx", r.out0); }
}
TEST(rdtscp_allowed_in_user_mode) {
    if (!cpu_hw(F_RDTSCP)) SKIP("RDTSCP not enumerated (see feat_rdtscp_absent)");
    static fp_res r; fp_expect_nofault("nf_rdtscp", &r);
    if (r.line && !r.fault) {
        CHECKF(r.out0 != 0, "RDTSCP returned TSC 0");
        t_info("tsc_aux", "0x%llx", r.out1 & 0xFFFFFFFFull);
    }
}
TEST(rdpid_allowed_in_user_mode) {
    static fp_res r;
    if (!cpu_hw(F_RDPID)) {
        fp_run("nf_rdpid", &r);
        if (fp_survived("nf_rdpid", &r)) {
            if (r.fault) t_note("RDPID absent: F3 0F C7 /7 raised 0x%08lx (%s)", (unsigned long)r.code, fp_code_name(r.code));
            else t_note("RDPID absent but F3 0F C7 /7 executed (value 0x%llx)", r.out0);
        }
        return;
    }
    fp_expect_nofault("nf_rdpid", &r);
    if (r.line && !r.fault) t_info("tsc_aux", "0x%llx", r.out0 & 0xFFFFFFFFull);
}
TEST(xgetbv_ecx1_with_xgetbv1) {
    if (!cpu()->osxsave) SKIP("OSXSAVE clear");
    if (!cpu_hw(F_XGETBV1)) SKIP("XGETBV1 not enumerated (see gp_xgetbv_ecx1_without_xgetbv1)");
    static fp_res r; fp_expect_nofault("nf_xgetbv_ecx1", &r);
    if (r.line && !r.fault) {
        uint64_t xin = r.out0;
        t_info("xinuse", "0x%llx", (unsigned long long)xin);
        CHECKF((xin & ~cpu()->xcr0) == 0, "XINUSE 0x%llx has bits outside XCR0 0x%llx", (unsigned long long)xin, (unsigned long long)cpu()->xcr0);
    }
}

/* ---- privileged instructions --------------------------------------------------------------- */
/* Windows-listed privileged opcode (KiOp_Priv): STATUS_PRIVILEGED_INSTRUCTION. Other environments
 * that report the #GP as an access violation are wrong for Windows; under Wine that is a known gap. */
static void fp_priv(const char *probe) {
    static fp_res r; fp_run(probe, &r);
    if (t_is_wine() && r.fault && r.code == S_AV)
        t_expect_fail("Wine reports this #GP as STATUS_ACCESS_VIOLATION (its privileged-opcode list is shorter than Windows')");
    DWORD c[1] = { S_PRIV };
    fp_expect_fault(probe, &r, c, 1, 0, FX_FAULT);
    if (r.fault && r.code == S_PRIV) CHECKF(r.np == 0, "%s: STATUS_PRIVILEGED_INSTRUCTION with %lu parameters (Windows: 0)", probe, r.np);
}
TEST(priv_hlt)           { fp_priv("priv_hlt"); }
TEST(priv_cli)           { fp_priv("priv_cli"); }
TEST(priv_sti)           { fp_priv("priv_sti"); }
TEST(priv_in_al_dx)      { fp_priv("priv_in_dx"); }
TEST(priv_out_dx_al)     { fp_priv("priv_out_dx"); }
TEST(priv_in_al_imm8)    { fp_priv("priv_in_imm"); }
TEST(priv_out_imm8_al)   { fp_priv("priv_out_imm"); }
TEST(priv_insb)          { fp_priv("priv_insb"); }
TEST(priv_outsb)         { fp_priv("priv_outsb"); }
TEST(priv_mov_from_cr0)  { fp_priv("priv_mov_from_cr0"); }
TEST(priv_mov_from_cr3)  { fp_priv("priv_mov_from_cr3"); }
TEST(priv_mov_from_cr4)  { fp_priv("priv_mov_from_cr4"); }
TEST(priv_mov_from_dr7)  { fp_priv("priv_mov_from_dr7"); }
TEST(priv_rdmsr)         { fp_priv("priv_rdmsr"); }
TEST(priv_wrmsr)         { fp_priv("priv_wrmsr"); }
TEST(priv_clts)          { fp_priv("priv_clts"); }
TEST(priv_wbinvd)        { fp_priv("priv_wbinvd"); }
TEST(priv_invlpg)        { fp_priv("priv_invlpg"); }
TEST(priv_lldt)          { fp_priv("priv_lldt"); }
TEST(priv_ltr)           { fp_priv("priv_ltr"); }
TEST(priv_invd)          { fp_priv("priv_invd"); }
TEST(priv_mov_to_dr7)    { fp_priv("priv_mov_to_dr7"); }
TEST(priv_sysexit) {
    /* Intel: #GP at CPL!=0; AMD: #UD in long mode. Windows lists 0F 35 for any fault type -> PRIV. */
    static fp_res r; fp_run("priv_sysexit", &r);
    if (!cpu()->is_intel) {
        /* AMD: SYSEXIT is #UD in long mode (64-bit and compatibility); whether Windows' #UD path rewrites
         * 0F 35 is not pinned down, so either code is accepted there. */
        DWORD c2[2] = { S_PRIV, S_ILL };
        if (fp_expect_fault("priv_sysexit", &r, c2, 2, 0, FX_FAULT) && r.code == S_ILL)
            t_note("non-Intel: SYSEXIT reported as STATUS_ILLEGAL_INSTRUCTION (#UD in long mode)");
        return;
    }
    if (t_is_wine() && r.fault && (r.code == S_AV || r.code == S_ILL))
        t_expect_fail("Wine does not translate SYSEXIT faults to STATUS_PRIVILEGED_INSTRUCTION");
    DWORD c[1] = { S_PRIV };
    fp_expect_fault("priv_sysexit", &r, c, 1, 0, FX_FAULT);
}
static void fp_priv_after_setup(const char *probe, const char *setup_what) {
    static fp_res r; fp_run(probe, &r);
    if (r.line && r.setup) {
        t_note("%s faulted (0x%08lx %s): no safe operand for the privileged load", setup_what, (unsigned long)r.code, fp_code_name(r.code));
        SKIP("%s is not available in user mode here", setup_what);
    }
    if (t_is_wine() && r.fault && r.code == S_AV)
        t_expect_fail("Wine reports this #GP as STATUS_ACCESS_VIOLATION (its privileged-opcode list is shorter than Windows')");
    DWORD c[1] = { S_PRIV };
    fp_expect_fault(probe, &r, c, 1, 0, FX_FAULT);
}
TEST(priv_lgdt) { fp_priv_after_setup("priv_lgdt", "SGDT"); }
TEST(priv_lidt) { fp_priv_after_setup("priv_lidt", "SIDT"); }
TEST(priv_lmsw) { fp_priv_after_setup("priv_lmsw", "SMSW"); }
#ifdef T_X64
TEST(priv_mov_from_cr8) { fp_priv("priv_mov_from_cr8"); }
TEST(priv_mov_to_cr8)   { fp_priv("priv_mov_to_cr8"); }
TEST(priv_swapgs)       { fp_priv("priv_swapgs"); }
TEST(priv_sysret)       { fp_priv("priv_sysret"); }
#else
TEST(priv_x64_only)     { SKIP("MOV from CR8, SWAPGS and SYSRET probes are x64 only"); }
#endif
TEST(priv_rdpmc_recorded) {
    /* CR4.PCE is an OS choice: RDPMC may be allowed. Faults must still be a #GP translation. */
    static fp_res r; fp_run("priv_rdpmc", &r);
    if (!fp_survived("priv_rdpmc", &r)) return;
    if (!r.fault) { t_note("RDPMC(0) executed in user mode (CR4.PCE set): 0x%llx", r.out0); t_info("rdpmc", "allowed"); return; }
    t_info("rdpmc", "faults");
    DWORD c[2] = { S_PRIV, S_AV };
    fp_expect_fault("priv_rdpmc", &r, c, 2, 0, FX_FAULT);
    if (r.code == S_AV) t_note("RDPMC #GP reported as STATUS_ACCESS_VIOLATION (Windows lists 0F 33 as privileged)");
}

/* ---- #GP outside the privileged-opcode list: ACCESS_VIOLATION on Windows (PRIV tolerated) ---- */
static void fp_gp(const char *probe) {
    static fp_res r; fp_run(probe, &r);
    DWORD c[2] = { S_AV, S_PRIV };
    if (fp_expect_fault(probe, &r, c, 2, 0, FX_FAULT)) {
        if (r.code == S_AV) {
            t_info("av_params", "np=%lu p0=0x%llx p1=0x%llx", r.np, r.p0, r.p1);
            CHECKF(r.np >= 2, "%s: STATUS_ACCESS_VIOLATION with %lu parameters (documented: 2)", probe, r.np);
        } else t_note("%s: #GP reported as STATUS_PRIVILEGED_INSTRUCTION (Windows 19041 reports ACCESS_VIOLATION)", probe);
    }
}
TEST(gp_xsetbv_user_mode) {
    if (!cpu()->osxsave) { t_note("OSXSAVE clear: XSETBV is #UD"); fp_ud("priv_xsetbv"); return; }
    fp_gp("priv_xsetbv");
}
TEST(gp_xgetbv_ecx2) {
    if (!cpu()->osxsave) SKIP("OSXSAVE clear: XGETBV itself is #UD");
    fp_gp("gp_xgetbv_ecx2");
}
TEST(gp_xgetbv_ecx_ffffffff) {
    if (!cpu()->osxsave) SKIP("OSXSAVE clear: XGETBV itself is #UD");
    fp_gp("gp_xgetbv_ecx_ffffffff");
}
TEST(gp_xgetbv_ecx1_without_xgetbv1) {
    if (!cpu()->osxsave) SKIP("OSXSAVE clear");
    if (cpu_hw(F_XGETBV1)) SKIP("XGETBV1 enumerated: ECX=1 is legal (see xgetbv_ecx1_with_xgetbv1)");
    fp_gp("nf_xgetbv_ecx1");
}
TEST(gp_movaps_misaligned)  { fp_gp("gp_movaps_misaligned"); }
TEST(gp_ldmxcsr_reserved_bit) { fp_gp("gp_ldmxcsr_reserved"); }
TEST(gp_fxsave_misaligned)  { if (!cpu_hw(F_FXSR)) SKIP("no FXSR"); fp_gp("gp_fxsave_misaligned"); }
TEST(gp_xsave_misaligned)   { if (!cpu_has(F_XSAVE)) SKIP("XSAVE not usable"); fp_gp("gp_xsave_misaligned"); }
TEST(gp_xrstor_bad_header)  { if (!cpu_has(F_XSAVE)) SKIP("XSAVE not usable"); fp_gp("gp_xrstor_bad_header"); }
TEST(gp_int_n_dpl0_gate)    { fp_gp("gp_int_0e"); }
static void fp_gp_int(const char *probe, const char *what) {
    static fp_res r; fp_run(probe, &r);
    if (t_is_wine() && r.line && !r.fault)
        t_expect_fail("%s executed without a fault under Wine (the Linux kernel handled the gate), result 0x%llx", what, r.out0);
    DWORD c[2] = { S_AV, S_PRIV };
    fp_expect_fault(probe, &r, c, 2, 0, FX_FAULT);
}
TEST(gp_int_80_not_a_system_call) { fp_gp_int("gp_int_80", "INT 80h"); }
TEST(gp_int_21_not_a_system_call) { fp_gp_int("gp_int_21", "INT 21h"); }
/* XSAVES/XRSTORS manage supervisor state: #GP at CPL3 when implemented (Windows: not a listed
 * privileged opcode -> ACCESS_VIOLATION; PRIVILEGED_INSTRUCTION tolerated), #UD when absent or OSXSAVE clear. */
static void fp_xsaves(const char *probe, const char *what) {
    const t_cpu_info *c = cpu();
    if (!c->osxsave) { t_note("OSXSAVE clear: %s is #UD", what); fp_ud(probe); return; }
    if (cpu_hw(F_XSAVES)) { fp_gp(probe); return; }
    static fp_res r; fp_run(probe, &r);
    if (c->hypervisor) {
        DWORD cc[3] = { S_ILL, S_AV, S_PRIV };
        if (fp_expect_fault(probe, &r, cc, 3, 0, FX_FAULT) && r.code != S_ILL)
            t_note("%s not enumerated but raised #GP (hidden by a hypervisor, CPL check)", what);
        return;
    }
    DWORD cc[1] = { S_ILL };
    fp_expect_fault(probe, &r, cc, 1, 0, FX_FAULT);
}
TEST(gp_xsaves_user_mode)  { fp_xsaves("gp_xsaves_user", "XSAVES"); }
TEST(gp_xrstors_user_mode) { fp_xsaves("gp_xrstors_user", "XRSTORS"); }
TEST(ac_alignment_check_recorded) {
    /* Whether the OS sets CR0.AM (so user EFLAGS.AC enables #AC) is an OS choice: record only. */
    static fp_res r; fp_run("ac_misaligned_load", &r);
    if (!r.line) { t_note("EFLAGS.AC + misaligned load: %s", r.text); return; }
    if (r.fault) t_note("EFLAGS.AC + misaligned load raised 0x%08lx %s (np=%lu)", (unsigned long)r.code, fp_code_name(r.code), r.np);
    else t_note("EFLAGS.AC + misaligned load executed (CR0.AM clear or alignment checking not delivered)");
    t_info("alignment_check", "%s", r.fault ? fp_code_name(r.code) : "none");
}
#ifdef T_X64
TEST(gp_cmpxchg16b_misaligned) { if (!cpu_hw(F_CX16)) SKIP("no CMPXCHG16B"); fp_gp("gp_cmpxchg16b_misaligned"); }
TEST(cmpxchg16b_aligned_is_legal) {
    if (!cpu_hw(F_CX16)) SKIP("no CMPXCHG16B");
    static fp_res r; fp_expect_nofault("ok_cmpxchg16b_aligned", &r);
}
TEST(gp_noncanonical_read) {
    static fp_res r; fp_run("gp_noncanonical_read", &r);
    DWORD c[2] = { S_AV, S_PRIV };
    if (fp_expect_fault("gp_noncanonical_read", &r, c, 2, 0, FX_FAULT)) {
        CHECKF(r.code == S_AV, "non-canonical load must be STATUS_ACCESS_VIOLATION (got %s)", fp_code_name(r.code));
        t_info("av_params", "np=%lu p0=0x%llx p1=0x%llx", r.np, r.p0, r.p1);
        if (r.p1 != ~0ull) t_note("AV address parameter 0x%llx (Windows reports 0xFFFFFFFFFFFFFFFF for #GP)", r.p1);
    }
}
#else
TEST(gp_x64_only) { SKIP("CMPXCHG16B and non-canonical probes are x64 only"); }
#endif

/* ---- UMIP-class instructions: no fault on Windows (native or emulated) --------------------- */
static void fp_umip(const char *probe, const char *what) {
    static fp_res r; fp_run(probe, &r);
    if (cpu_hw(F_UMIP)) t_note("processor implements UMIP: %s may be emulated by the OS", what);
    if (t_is_wine() && r.fault)
        t_expect_fail("Wine/Linux does not emulate %s under UMIP for this mode", what);
    if (!fp_survived(probe, &r)) return;
    CHECKF(!r.fault, "%s raised 0x%08lx (%s): Windows executes or emulates it in user mode", what, (unsigned long)r.code, fp_code_name(r.code));
    if (!r.fault && r.out0) t_info("value", "0x%llx", r.out0);
}
TEST(umip_sgdt_no_fault) { fp_umip("umip_sgdt", "SGDT"); }
TEST(umip_sidt_no_fault) { fp_umip("umip_sidt", "SIDT"); }
TEST(umip_smsw_no_fault) {
    fp_umip("umip_smsw", "SMSW");
}
TEST(umip_sldt_no_fault) { fp_umip("umip_sldt", "SLDT"); }
TEST(umip_str_no_fault)  { fp_umip("umip_str", "STR"); }

/* ---- page faults ---------------------------------------------------------------------------- */
TEST(av_null_read) {
    static fp_res r; fp_run("av_null_read", &r);
    DWORD c[1] = { S_AV };
    if (fp_expect_fault("av_null_read", &r, c, 1, 0, FX_FAULT)) {
        CHECKF(r.np == 2, "np=%lu, documented 2", r.np);
        CHECKF(r.p0 == 0, "ExceptionInformation[0]=%llu, documented 0 (read)", r.p0);
        CHECKF(r.p1 == 0, "ExceptionInformation[1]=0x%llx, expected the faulting address 0", r.p1);
    }
}
TEST(av_null_write) {
    static fp_res r; fp_run("av_null_write", &r);
    DWORD c[1] = { S_AV };
    if (fp_expect_fault("av_null_write", &r, c, 1, 0, FX_FAULT)) {
        CHECKF(r.np == 2, "np=%lu, documented 2", r.np);
        CHECKF(r.p0 == 1, "ExceptionInformation[0]=%llu, documented 1 (write)", r.p0);
        CHECKF(r.p1 == 0, "ExceptionInformation[1]=0x%llx, expected 0", r.p1);
    }
}
/* Emulator artifacts (README "Emulated validation"), under tools/sde_run.sh only; checks unchanged. */
#define FF_SDE_XFAIL(why) do { if (t_sde_model()) t_expect_fail("Intel SDE (%s) %s", t_sde_model(), why); } while (0)
TEST(av_execute_nx_page) {
    if (!cpu_hw(F_NX)) SKIP("no NX");
    FF_SDE_XFAIL("reports an instruction fetch from a no-execute page as a read (ExceptionInformation[0]=0)");
#ifndef T_X64
    DWORD flags = 0; BOOL perm = FALSE;
    BOOL (WINAPI *gdp)(HANDLE, LPDWORD, PBOOL) = (BOOL (WINAPI *)(HANDLE, LPDWORD, PBOOL))t_proc("kernel32.dll", "GetProcessDEPPolicy");
    if (gdp && gdp(GetCurrentProcess(), &flags, &perm)) {
        t_info("dep_flags", "0x%lx permanent=%d", flags, perm);
        if (!(flags & 1)) SKIP("DEP disabled for this 32-bit process");
    }
#endif
    static fp_res r; fp_run("av_nx_exec", &r);
    DWORD c[1] = { S_AV };
    if (fp_expect_fault("av_nx_exec", &r, c, 1, 0, FX_FAULT)) {
        CHECKF(r.np == 2, "np=%lu, documented 2", r.np);
        CHECKF(r.p0 == 8, "ExceptionInformation[0]=%llu, documented 8 (DEP violation)", r.p0);
    }
}

/* ---- traps, software interrupts, arithmetic faults ------------------------------------------ */
TEST(trap_int3_breakpoint) {
    static fp_res r; fp_run("trap_int3", &r);
    DWORD c[1] = { S_BP };
    if (fp_expect_fault("trap_int3", &r, c, 1, 0, FX_EA | FX_MARK | FX_ONE)) t_info("ctx_ip_offset", "%lld", r.ip);
}
TEST(trap_int3_long_form) {
    static fp_res r; fp_run("trap_int3_long", &r);
    DWORD c[1] = { S_BP };
    if (fp_expect_fault("trap_int3_long", &r, c, 1, 0, FX_MARK | FX_ONE)) {
        if (r.ea != 1) t_note("CD 03: ExceptionAddress insn%+lld (Windows reports RIP-1 = insn+1)", r.ea);
    }
}
TEST(trap_int2c_assertion) {
    static fp_res r; fp_run("trap_int2c", &r);
    if (t_is_wine() && r.fault && r.code != S_ASSERT) t_expect_fail("Wine does not implement the int 2Ch assertion gate");
    DWORD c[1] = { S_ASSERT };
    if (fp_expect_fault("trap_int2c", &r, c, 1, 0, FX_MARK | FX_ONE) && r.ea != 0)
        t_note("int 2Ch: ExceptionAddress insn%+lld (Windows backs RIP up by 2)", r.ea);
}
TEST(trap_int2d_recorded) {
    static fp_res r; fp_run("trap_int2d", &r);
    if (!fp_survived("trap_int2d", &r)) return;
    if (r.fault) t_note("int 2Dh (debug service, no debugger): 0x%08lx %s ea=insn%+lld ip=insn%+lld",
                        (unsigned long)r.code, fp_code_name(r.code), r.ea, r.ip);
    else t_note("int 2Dh returned without an exception");
}
TEST(trap_fastfail_bypasses_handlers) {
    if (!t_env_get()->pf[PF_FASTFAIL_AVAILABLE]) SKIP("PF_FASTFAIL_AVAILABLE not reported");
    static fp_res r; fp_run("trap_fastfail", &r);
    if (t_is_wine() && (r.line || r.exit_code != S_FASTFAIL)) t_expect_fail("Wine's __fastfail differs (handlers called or other exit code)");
    CHECKF(!r.timed_out, "child timed out");
    CHECKF(r.began, "child did not reach the probe");
    CHECKF(!r.line, "int 29h was delivered to a vectored handler (%s): __fastfail must bypass all handlers", r.text);
    CHECKF(r.exit_code == S_FASTFAIL, "process exit code 0x%08lx, expected STATUS_STACK_BUFFER_OVERRUN", (unsigned long)r.exit_code);
}
TEST(trap_icebp_recorded) {
    FF_SDE_XFAIL("executes ICEBP (F1) without raising #DB");
    static fp_res r; fp_run("trap_icebp", &r);
    if (!fp_survived("trap_icebp", &r)) return;
    CHECKF(r.fault, "ICEBP (F1) executed without a #DB");
    if (r.fault) t_note("ICEBP: 0x%08lx %s ea=insn%+lld", (unsigned long)r.code, fp_code_name(r.code), r.ea);
}
TEST(trap_single_step_tf) {
    static fp_res r; fp_run("trap_single_step", &r);
    DWORD c[1] = { S_SS };
    /* POPF sets TF: the trap is taken after the NEXT instruction (the NOP at insn) completes. */
    fp_expect_fault("trap_single_step", &r, c, 1, 1, FX_FAULT);
}
TEST(trap_divide_by_zero) { fp_expect_code("trap_div0", S_IDIV0); }
static void fp_div_overflow(const char *probe) {
    static fp_res r; fp_run(probe, &r);
    if (t_is_wine() && r.fault && r.code == S_IDIV0)
        t_expect_fail("Wine reports every #DE as STATUS_INTEGER_DIVIDE_BY_ZERO");
    DWORD c[1] = { S_IOVF };
    /* Windows KiOp_Div: #DE with a non-zero divisor is quotient overflow -> STATUS_INTEGER_OVERFLOW */
    fp_expect_fault(probe, &r, c, 1, 0, FX_FAULT);
}
TEST(trap_div_quotient_overflow)  { fp_div_overflow("trap_div_overflow"); }
TEST(trap_idiv_int_min_by_minus1) { fp_div_overflow("trap_idiv_overflow"); }
#ifndef T_X64
TEST(trap_into_overflow_x86) {
    static fp_res r; fp_run("trap_into_x86", &r);
    DWORD c[1] = { S_IOVF };
    if (fp_expect_fault("trap_into_x86", &r, c, 1, 0, FX_MARK | FX_ONE) && r.ea != 0)
        t_note("INTO: ExceptionAddress insn%+lld (Windows reports the INTO itself)", r.ea);
}
TEST(into_without_of_x86) { static fp_res r; fp_expect_nofault("ok_into_no_of_x86", &r); }
TEST(trap_bound_out_of_range_x86) { fp_expect_code("trap_bound_x86", S_BOUNDS); }
TEST(bound_in_range_x86) { static fp_res r; fp_expect_nofault("ok_bound_in_range_x86", &r); }
TEST(sys_sysenter_recorded_x86) {
    static fp_res r; fp_run("sys_sysenter_x86", &r);
    if (r.line) t_note("SYSENTER (eax=0xFFF): fault=%d code=0x%08lx %s", r.fault, (unsigned long)r.code, fp_code_name(r.code));
    else t_note("SYSENTER (eax=0xFFF): %s", r.text);
}
#else
TEST(trap_x86_only) { SKIP("INTO/BOUND/SYSENTER probes are x86 only (invalid or different in 64-bit mode)"); }
#endif

/* ---- system-call entry points with an invalid service number ------------------------------- */
#ifdef T_X64
TEST(sys_syscall_invalid_service_recorded) {
    static fp_res r; fp_run("sys_syscall_badsvc", &r);
    if (!r.line) { t_note("SYSCALL (eax=0xFFF): %s", r.text); return; }
    if (r.fault) t_note("SYSCALL (eax=0xFFF) raised 0x%08lx %s", (unsigned long)r.code, fp_code_name(r.code));
    else {
        t_info("status", "0x%08llx %s", r.out0 & 0xFFFFFFFFull, fp_code_name((unsigned long)(r.out0 & 0xFFFFFFFFull)));
        if ((r.out0 & 0xFFFFFFFFull) != S_INVSVC) t_note("SYSCALL (eax=0xFFF) returned 0x%llx (Windows: STATUS_INVALID_SYSTEM_SERVICE)", r.out0);
    }
}
#endif
TEST(sys_int2e_invalid_service_recorded) {
    static fp_res r; fp_run("sys_int2e_badsvc", &r);
    if (!r.line) { t_note("int 2Eh (eax=0xFFF): %s", r.text); return; }
    if (r.fault) t_note("int 2Eh (eax=0xFFF) raised 0x%08lx %s", (unsigned long)r.code, fp_code_name(r.code));
    else t_info("status", "0x%08llx %s", r.out0 & 0xFFFFFFFFull, fp_code_name((unsigned long)(r.out0 & 0xFFFFFFFFull)));
}

/* ---- unmasked x87 exceptions ----------------------------------------------------------------- */
static void fp_x87u(const char *probe, const DWORD *codes, int n, unsigned sw_bits) {
    static fp_res r; fp_run(probe, &r);
    int known = 0;
    for (int i = 0; i < n; i++) known |= r.code == codes[i];
    if (t_is_wine() && r.fault && !known)
        t_expect_fail("Wine derives x87 codes from the Linux si_code (a denormal #MF arrives as FPE_FLTUND)");
#ifdef T_X64
    /* 64-bit: KiFloatingErrorFault reports the waiting instruction (RIP) that took the #MF. */
    int fx = FX_FAULT;
#else
    /* 32-bit: the report may use the x87 FPU instruction pointer (the faulting x87 instruction just
     * before the FWAIT) instead of the FWAIT itself; the context IP must still be the FWAIT. */
    int fx = FX_MARK | FX_ONE;
#endif
    if (fp_expect_fault(probe, &r, codes, n, 0, fx)) {
#ifndef T_X64
        if (t_sde_model() && !(r.ea == 0 || (r.ea < 0 && r.ea >= -8)))   /* emulator artifact, README "Emulated validation" */
            t_expect_fail("Intel SDE (%s): the x87 FPU instruction pointer points into Pin's code cache", t_sde_model());
        CHECKF(r.ip == 0, "%s: context EIP insn%+lld, expected the FWAIT that took the #MF", probe, r.ip);
        CHECKF(r.ea == 0 || (r.ea < 0 && r.ea >= -8), "%s: ExceptionAddress insn%+lld is neither the FWAIT nor the x87 instruction before it", probe, r.ea);
        if (r.ea != 0) t_note("%s: ExceptionAddress = faulting x87 instruction (insn%+lld), not the FWAIT", probe, r.ea);
#endif
        t_info("fsw", "0x%04x", r.fsw);
        CHECKF((r.fsw & sw_bits) == sw_bits, "%s: context x87 status word 0x%04x lacks 0x%04x", probe, r.fsw, sw_bits);
        CHECKF(r.fsw & 0x80, "%s: context x87 status word 0x%04x lacks ES (error summary)", probe, r.fsw);
    }
}
TEST(x87_unmasked_stack_underflow) { DWORD c[1] = { S_FSTK }; fp_x87u("x87u_stack", c, 1, 0x41); }
TEST(x87_unmasked_invalid_sqrt)    { DWORD c[1] = { S_FINV }; fp_x87u("x87u_invalid", c, 1, 0x01); }
TEST(x87_unmasked_zero_divide)     { DWORD c[1] = { S_FDIV }; fp_x87u("x87u_zerodiv", c, 1, 0x04); }
TEST(x87_unmasked_overflow)        { DWORD c[1] = { S_FOVF }; fp_x87u("x87u_overflow", c, 1, 0x08); }
TEST(x87_unmasked_underflow)       { DWORD c[1] = { S_FUND }; fp_x87u("x87u_underflow", c, 1, 0x10); }
TEST(x87_unmasked_precision)       { DWORD c[1] = { S_FINEX }; fp_x87u("x87u_precision", c, 1, 0x20); }
TEST(x87_unmasked_denormal) {
    /* Windows maps x87 DE to STATUS_FLOAT_INVALID_OPERATION (KiFloatingErrorFault); the dedicated
     * STATUS_FLOAT_DENORMAL_OPERAND is also a documented translation. */
    DWORD c[2] = { S_FINV, S_FDEN }; fp_x87u("x87u_denormal", c, 2, 0x02);
}

/* ---- unmasked SSE exceptions ----------------------------------------------------------------- */
static void fp_sseu(const char *probe, DWORD code64, int trap_class, unsigned mx_flag) {
    static fp_res r; fp_run(probe, &r);
    DWORD want = code64;
#ifndef T_X64
    want = trap_class ? S_FMT : S_FMF;     /* 32-bit code: MULTIPLE_TRAPS / MULTIPLE_FAULTS */
    if (!t_env_get()->wow64 && !t_is_wine()) {
        DWORD c2[3] = { want, code64, S_FINV };
        if (fp_expect_fault(probe, &r, c2, 3, 0, FX_FAULT)) CHECKF(r.mxcsr & mx_flag, "MXCSR 0x%x lacks flag 0x%x", r.mxcsr, mx_flag);
        return;
    }
#endif
    if (t_is_wine() && r.fault && r.code != want) t_expect_fail("Wine translates #XM differently from Windows");
    DWORD c[1] = { want };
    if (fp_expect_fault(probe, &r, c, 1, 0, FX_FAULT)) {
        t_info("mxcsr", "0x%x", r.mxcsr);
        if (r.mxcsr != 0xFFFFFFFFu) CHECKF(r.mxcsr & mx_flag, "%s: context MXCSR 0x%x lacks the sticky flag 0x%x", probe, r.mxcsr, mx_flag);
    }
}
TEST(sse_unmasked_invalid)   { fp_sseu("sseu_invalid", S_FINV, 1, 0x01); }
TEST(sse_unmasked_zero_divide) { fp_sseu("sseu_zerodiv", S_FDIV, 1, 0x04); }
TEST(sse_unmasked_denormal)  { fp_sseu("sseu_denormal", S_FINV, 1, 0x02); }
TEST(sse_unmasked_overflow)  { fp_sseu("sseu_overflow", S_FOVF, 0, 0x08); }
TEST(sse_unmasked_underflow) { fp_sseu("sseu_underflow", S_FUND, 0, 0x10); }
TEST(sse_unmasked_precision) { fp_sseu("sseu_precision", S_FINEX, 0, 0x20); }

/* ---- the same fault on a worker thread, and many times in a row --------------------------------- */
static void fp_on_thread(const char *probe, DWORD code) {
    static fp_res r; fp_run_mode(probe, "thread", 8000, &r);
    char v[64];
    if (t_child_find(&g_cr, "FP_NOTHREAD", v, sizeof v)) SKIP("CreateThread failed in the child (%s)", v);
    if (t_child_find(&g_cr, "FP_THREAD_FAILED", v, sizeof v)) CHECKF(0, "%s on a worker thread: thread did not finish (%s)", probe, v);
    DWORD c[1] = { code };
    fp_expect_fault(probe, &r, c, 1, 0, FX_FAULT);
}
/* AVX-512 #UD is certain when the ZMM state is off in XCR0, or on bare metal without the CPUID bit. */
static int fp_avx512_surely_ud(void) {
    const t_cpu_info *c = cpu();
    return !c->osxsave || (c->xcr0 & XS_ZMM) != XS_ZMM || (!c->hypervisor && !cpu_hw(F_AVX512F));
}
TEST(thread_ud2_on_worker_thread) { fp_on_thread("ud_ud2", S_ILL); }
TEST(thread_absent_avx512_on_worker_thread) {
    if (cpu_has(F_AVX512F)) SKIP("AVX-512F is usable here");
    if (!fp_avx512_surely_ud()) SKIP("AVX-512 may be implemented but hidden: covered by feat_avx512f_vaddps_zmm");
    fp_on_thread("feat_avx512f_vaddps_zmm", S_ILL);
}
TEST(thread_av_null_read_on_worker_thread) { fp_on_thread("av_null_read", S_AV); }

static void fp_repeat(const char *probe, int n) {
    static fp_res r;
    char mode[32]; snprintf(mode, sizeof mode, "repeat=%d", n);
    fp_run_mode(probe, mode, 20000, &r);
    CHECKF(!r.timed_out, "%s x%d: child timed out", probe, n);
    CHECKF(r.began, "%s x%d: child did not reach the probe", probe, n);
    char v[300];
    if (!t_child_find(&g_cr, "FPR ", v, sizeof v)) {
        CHECKF(0, "%s x%d: child died before its summary (%s); output: %.200s", probe, n, r.text, g_cr.out);
        return;
    }
    char sv[320]; snprintf(sv, sizeof sv, " %s", v);
    t_note("%s", v);
    long good = (long)fp_kvs(sv, "good"), first_bad = (long)fp_kvs(sv, "first_bad");
    DWORD code = (DWORD)fp_kv(sv, "code", NULL);
    t_info("exception", "0x%08lx %s", (unsigned long)code, fp_code_name(code));
    CHECKF(code != 0, "%s: first iteration raised no exception", probe);
    CHECKF(good == n, "%s: %ld of %d iterations gave exactly one exception with the first iteration's code/address "
           "and intact callee-saved registers (first bad iteration %ld)", probe, good, n, first_bad);
    CHECKF(r.exit_code == 0, "%s x%d: child exit code 0x%08lx", probe, n, (unsigned long)r.exit_code);
}
TEST(repeat_ud2_500)            { fp_repeat("ud_ud2", 500); }
TEST(repeat_null_read_200)      { fp_repeat("av_null_read", 200); }
TEST(repeat_movaps_gp_200)      { fp_repeat("gp_movaps_misaligned", 200); }
TEST(repeat_hlt_200)            { fp_repeat("priv_hlt", 200); }
TEST(repeat_sse_zerodiv_200)    { fp_repeat("sseu_zerodiv", 200); }
#ifndef T_X64
#define FF_SDE_XFAIL_X87IP() FF_SDE_XFAIL("x87 FPU instruction pointer points into Pin's code cache (differs per iteration)")
#else
#define FF_SDE_XFAIL_X87IP() ((void)0)
#endif
TEST(repeat_x87_zerodiv_200)    { FF_SDE_XFAIL_X87IP(); fp_repeat("x87u_zerodiv", 200); }
TEST(repeat_absent_avx512_200) {
    if (cpu_has(F_AVX512F)) SKIP("AVX-512F is usable here");
    if (!fp_avx512_surely_ud()) SKIP("AVX-512 may be implemented but hidden");
    fp_repeat("feat_avx512f_vaddps_zmm", 200);
}

/* ---- exception context integrity ------------------------------------------------------------ */
TEST(ctx_ud2_gpr_xmm_mxcsr_fcw) {
    static fp_res r; fp_run("ctx_gpr_xmm", &r);
    DWORD c[1] = { S_ILL };
    if (!fp_expect_fault("ctx_gpr_xmm", &r, c, 1, 0, FX_FAULT)) return;
    t_info("context_flags", "0x%lx", r.cflags);
    unsigned b = r.ctxbad;
    CHECKF(!(b & 1), "volatile GPR (RCX/ECX) wrong in the #UD context");
    CHECKF(!(b & 2), "volatile GPR (RDX/EDX) wrong in the #UD context");
    CHECKF(!(b & 4), "GPR markers (R8-R11 / EBX,ESI,EDI) wrong in the #UD context");
#ifdef T_X64
    CHECKF(!(b & 0x100), "exception context lacks CONTEXT_FLOATING_POINT (flags 0x%lx)", r.cflags);
    CHECKF(!(b & 0x8), "XMM6 in the exception context differs from the value loaded before ud2");
    CHECKF(!(b & 0x10), "XMM15 in the exception context differs from the value loaded before ud2");
    CHECKF(!(b & 0x20), "CONTEXT.MxCsr differs from the MXCSR loaded before ud2");
    CHECKF(!(b & 0x40), "CONTEXT.FltSave.MxCsr differs from the MXCSR loaded before ud2");
    CHECKF(!(b & 0x80), "CONTEXT.FltSave.ControlWord differs from the x87 CW loaded before ud2");
#else
    if (b & 0x200) t_note("x86 exception context has no CONTEXT_EXTENDED_REGISTERS: XMM/MXCSR not checked");
    else {
        CHECKF(!(b & 0x8), "XMM6 (ExtendedRegisters) differs from the value loaded before ud2");
        CHECKF(!(b & 0x20), "MXCSR (ExtendedRegisters) differs from the value loaded before ud2");
    }
    if (b & 0x100) t_note("x86 exception context has no CONTEXT_FLOATING_POINT");
    else CHECKF(!(b & 0x80), "FloatSave.ControlWord differs from the x87 CW loaded before ud2");
#endif
}
TEST(ctx_ud2_ymm_upper_in_xstate) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    static fp_res r; fp_run("ctx_ymm_ud2", &r);
    DWORD c[1] = { S_ILL };
    if (!fp_expect_fault("ctx_ymm_ud2", &r, c, 1, 0, FX_FAULT)) return;
    t_info("context_flags", "0x%lx", r.cflags);
    if (r.ymm == -1) { t_note("exception context carries no XSTATE (CONTEXT_XSTATE not set / LocateXStateFeature NULL): YMM upper not checkable"); return; }
    CHECKF(r.ymm == 1, "YMM6 upper half in the exception context XSTATE differs from the value loaded before ud2");
}

/* =============================================================================================
 * In-process: masked x87 / MMX semantics (no fault possible), CPUID out-of-range and per-CPU
 * ============================================================================================= */
static const uint8_t k_indef80[10] = { 0,0,0,0,0,0,0,0xC0, 0xFF,0xFF };   /* QNaN floating-point indefinite */

typedef struct { uint16_t sw1, sw2, sw3; uint8_t st[10]; uint8_t pad[6]; } x87_obs;

static T_NOINLINE void x87_overflow_masked(x87_obs *o) {
    __asm__ volatile("fninit\n fld1\n fld1\n fld1\n fld1\n fld1\n fld1\n fld1\n fld1\n fnstsw %0\n"
                     "fld1\n fnstsw %1\n fstpt %3\n fninit\n"
                     : "=m"(o->sw1), "=m"(o->sw2), "=m"(o->sw3), "=m"(o->st) :: "memory");
}
TEST(x87_stack_overflow_masked) {
    x87_obs o; memset(&o, 0, sizeof o);
    x87_overflow_masked(&o);
    t_info("sw", "after8=0x%04x after9=0x%04x", o.sw1, o.sw2);
    CHECKF((o.sw1 & 0x3F) == 0, "8 pushes onto an empty stack raised flags (SW 0x%04x)", o.sw1);
    CHECKF(((o.sw1 >> 11) & 7) == 0, "TOP after 8 pushes = %u, expected 0", (o.sw1 >> 11) & 7);
    CHECKF(o.sw2 & 0x01, "9th push: IE not set (SW 0x%04x)", o.sw2);
    CHECKF(o.sw2 & 0x40, "9th push: SF not set (SW 0x%04x)", o.sw2);
    CHECKF(o.sw2 & 0x200, "9th push: C1 must be 1 for stack overflow (SW 0x%04x)", o.sw2);
    CHECK_MEM(o.st, k_indef80, 10);
}

static T_NOINLINE void x87_underflow_masked(x87_obs *o) {
    __asm__ volatile("fninit\n fadd %%st(1), %%st\n fnstsw %0\n fxam\n fnstsw %1\n fstpt %3\n fninit\n"
                     : "=m"(o->sw1), "=m"(o->sw2), "=m"(o->sw3), "=m"(o->st) :: "memory");
}
TEST(x87_stack_underflow_masked) {
    x87_obs o; memset(&o, 0, sizeof o);
    x87_underflow_masked(&o);
    t_info("sw", "after_fadd=0x%04x after_fxam=0x%04x", o.sw1, o.sw2);
    CHECKF(o.sw1 & 0x01, "FADD on an empty stack: IE not set (SW 0x%04x)", o.sw1);
    CHECKF(o.sw1 & 0x40, "FADD on an empty stack: SF not set (SW 0x%04x)", o.sw1);
    CHECKF(!(o.sw1 & 0x200), "stack underflow must clear C1 (SW 0x%04x)", o.sw1);
    CHECKF((o.sw2 & 0x4500) == 0x0100, "masked response must leave a NaN in ST0: FXAM class C3C2C0 = %u%u%u",
           (o.sw2 >> 14) & 1, (o.sw2 >> 10) & 1, (o.sw2 >> 8) & 1);
    CHECK_MEM(o.st, k_indef80, 10);
}

static T_NOINLINE void x87_fxam_ftst_empty(x87_obs *o) {
    __asm__ volatile("fninit\n fxam\n fnstsw %0\n fninit\n ftst\n fnstsw %1\n fninit\n"
                     : "=m"(o->sw1), "=m"(o->sw2) :: "memory");
}
TEST(x87_fxam_empty_register) {
    x87_obs o; memset(&o, 0, sizeof o);
    x87_fxam_ftst_empty(&o);
    t_info("sw", "0x%04x", o.sw1);
    CHECKF((o.sw1 & 0x4500) == 0x4100, "FXAM of an empty ST0: C3C2C0 = %u%u%u, expected 101 (empty)",
           (o.sw1 >> 14) & 1, (o.sw1 >> 10) & 1, (o.sw1 >> 8) & 1);
    CHECKF((o.sw1 & 0x3F) == 0, "FXAM must not raise exceptions (SW 0x%04x)", o.sw1);
}
TEST(x87_ftst_empty_register) {
    x87_obs o; memset(&o, 0, sizeof o);
    x87_fxam_ftst_empty(&o);
    t_info("sw", "0x%04x", o.sw2);
    CHECKF((o.sw2 & 0x41) == 0x41, "FTST of an empty ST0 must signal stack underflow (IE+SF), SW 0x%04x", o.sw2);
    CHECKF((o.sw2 & 0x4500) == 0x4500, "masked FTST on empty ST0: C3C2C0 = %u%u%u, expected 111 (unordered)",
           (o.sw2 >> 14) & 1, (o.sw2 >> 10) & 1, (o.sw2 >> 8) & 1);
}

static T_NOINLINE void x87_sqrt_neg(x87_obs *o) {
    __asm__ volatile("fninit\n fld1\n fchs\n fsqrt\n fnstsw %0\n fstpt %1\n fninit\n" : "=m"(o->sw1), "=m"(o->st) :: "memory");
}
TEST(x87_fsqrt_negative_masked) {
    x87_obs o; memset(&o, 0, sizeof o);
    x87_sqrt_neg(&o);
    t_info("sw", "0x%04x", o.sw1);
    CHECKF(o.sw1 & 0x01, "FSQRT(-1): IE not set (SW 0x%04x)", o.sw1);
    CHECKF(!(o.sw1 & 0x40), "FSQRT(-1) is not a stack fault: SF set (SW 0x%04x)", o.sw1);
    CHECK_MEM(o.st, k_indef80, 10);
}

static T_NOINLINE void x87_div_zero(x87_obs *o) {
    __asm__ volatile("fninit\n fldz\n fld1\n fdiv %%st(1), %%st\n fnstsw %0\n fstpt %1\n fninit\n" : "=m"(o->sw1), "=m"(o->st) :: "memory");
}
TEST(x87_divide_by_zero_masked) {
    static const uint8_t pinf[10] = { 0,0,0,0,0,0,0,0x80, 0xFF,0x7F };
    x87_obs o; memset(&o, 0, sizeof o);
    x87_div_zero(&o);
    t_info("sw", "0x%04x", o.sw1);
    CHECKF(o.sw1 & 0x04, "1/0: ZE not set (SW 0x%04x)", o.sw1);
    CHECKF(!(o.sw1 & 0x01), "1/0: IE must not be set (SW 0x%04x)", o.sw1);
    CHECK_MEM(o.st, pinf, 10);
}

static T_NOINLINE void x87_fist_ovf(x87_obs *o, int32_t *out) {
    static const double big = 1e20;
    __asm__ volatile("fninit\n fldl %2\n fistpl %1\n fnstsw %0\n fninit\n" : "=m"(o->sw1), "=m"(*out) : "m"(big) : "memory");
}
TEST(x87_fist_overflow_integer_indefinite) {
    x87_obs o; memset(&o, 0, sizeof o);
    int32_t v = 0x1234;
    x87_fist_ovf(&o, &v);
    t_info("result", "0x%08x sw=0x%04x", (unsigned)v, o.sw1);
    CHECKF(o.sw1 & 0x01, "FISTP of 1e20 to int32: IE not set (SW 0x%04x)", o.sw1);
    CHECK_EQ((uint32_t)v, 0x80000000u);
}

typedef struct { uint16_t sw_mmx, tw_mmx, sw_emms, tw_emms, sw_fld; uint8_t pad[6];
                 uint8_t fx1[512] __attribute__((aligned(16))); uint8_t fx2[512] __attribute__((aligned(16))); uint64_t mm7; } mmx_obs;
static const uint64_t k_mmval = 0x0123456789ABCDEFull;

static T_NOINLINE void mmx_tags(mmx_obs *o) {
    uint8_t env[32];
    __asm__ volatile("fninit\n movq %5, %%mm0\n fnstsw %0\n fnstenv %6\n fxsave %3\n emms\n fnstsw %2\n fxsave %4\n fninit\n"
                     : "=m"(o->sw_mmx), "=m"(o->tw_mmx), "=m"(o->sw_emms), "=m"(o->fx1), "=m"(o->fx2)
                     : "m"(k_mmval), "m"(env) : "memory", "mm0");
    memcpy(&o->tw_mmx, env + 8, 2);
}
TEST(mmx_write_sets_tags_valid_and_top0) {
    mmx_obs *o = (mmx_obs *)_aligned_malloc(sizeof *o, 16);
    REQUIRE(o);
    memset(o, 0, sizeof *o);
    mmx_tags(o);
    t_info("state", "sw=0x%04x fnstenv_tag=0x%04x abridged=0x%02x", o->sw_mmx, o->tw_mmx, o->fx1[4]);
    CHECKF(((o->sw_mmx >> 11) & 7) == 0, "after an MMX instruction TOP = %u, expected 0", (o->sw_mmx >> 11) & 7);
    CHECKF(o->fx1[4] == 0xFF, "FXSAVE abridged tag after an MMX instruction 0x%02x, expected 0xFF (all valid)", o->fx1[4]);
    int empty = 0;
    for (int i = 0; i < 8; i++) empty += ((o->tw_mmx >> (2 * i)) & 3) == 3;
    CHECKF(empty == 0, "FNSTENV tag word 0x%04x marks %d register(s) empty after an MMX instruction", o->tw_mmx, empty);
    uint64_t mant; memcpy(&mant, o->fx1 + 32, 8);
    uint16_t ex; memcpy(&ex, o->fx1 + 40, 2);
    CHECK_EQ(mant, k_mmval);
    CHECKF(ex == 0xFFFF, "MMX write must set bits 79:64 of the aliased register to all ones (got 0x%04x)", ex);
    _aligned_free(o);
}
TEST(emms_marks_all_tags_empty) {
    mmx_obs *o = (mmx_obs *)_aligned_malloc(sizeof *o, 16);
    REQUIRE(o);
    memset(o, 0, sizeof *o);
    mmx_tags(o);
    CHECKF(o->fx2[4] == 0x00, "FXSAVE abridged tag after EMMS 0x%02x, expected 0x00 (all empty)", o->fx2[4]);
    CHECKF(((o->sw_emms >> 11) & 7) == 0, "TOP after EMMS = %u, expected 0", (o->sw_emms >> 11) & 7);
    _aligned_free(o);
}

static T_NOINLINE void mmx_then_fld(uint16_t *sw, uint8_t st[10]) {
    __asm__ volatile("fninit\n movq %2, %%mm0\n fld1\n fnstsw %0\n fstpt %1\n fninit\n"
                     : "=m"(*sw), "=m"(*(uint8_t (*)[10])st) : "m"(k_mmval) : "memory", "mm0");
}
TEST(x87_after_mmx_without_emms_overflows) {
    uint16_t sw = 0; uint8_t st[10];
    mmx_then_fld(&sw, st);
    t_info("sw", "0x%04x", sw);
    CHECKF((sw & 0x41) == 0x41, "FLD after MMX without EMMS must overflow the (full) stack: SW 0x%04x", sw);
    CHECKF(sw & 0x200, "stack overflow sets C1 (SW 0x%04x)", sw);
    CHECK_MEM(st, k_indef80, 10);
}

static T_NOINLINE void x87_then_mmx_read(uint64_t *mm7, uint16_t *sw) {
    __asm__ volatile("fninit\n fld1\n movq %%mm7, %0\n fnstsw %1\n emms\n fninit\n" : "=m"(*mm7), "=m"(*sw) :: "memory", "mm7");
}
TEST(mmx_aliases_x87_mantissa) {
    uint64_t v = 0; uint16_t sw = 0;
    x87_then_mmx_read(&v, &sw);
    t_info("mm7", "0x%016llx", (unsigned long long)v);
    /* FLD1 with TOP=0 pushes into physical R7 (= MM7): mantissa of 1.0 is 0x8000000000000000. */
    CHECK_EQ(v, 0x8000000000000000ull);
    CHECKF(((sw >> 11) & 7) == 0, "MMX read must set TOP to 0 (SW 0x%04x)", sw);
}

/* ---- CPUID: out-of-range leaves are not faults ---------------------------------------------- */
/* Leaves compared below may carry per-CPU data (x2APIC ID in 0Bh/1Fh, core type in 1Ah on hybrid parts):
 * keep both CPUID executions on one CPU. */
static uintptr_t g_cpuid_pin;
static void cpuid_pin(void) {
    g_cpuid_pin = t_pin_thread(0);
    if (!g_cpuid_pin) t_note("could not pin to CPU 0: comparisons assume no migration between CPUID executions");
}
static void cpuid_unpin(void) { if (g_cpuid_pin) t_unpin_thread(g_cpuid_pin); g_cpuid_pin = 0; }

TEST(cpuid_above_max_basic_leaf) {
    const t_cpu_info *c = cpu();
    uint32_t leaf = c->max_basic + 1;
    if (leaf >= 0x40000000u) SKIP("max basic leaf 0x%x leaves no room below 0x40000000", c->max_basic);
    t_cpuid_regs hi, top, beyond;
    cpuid_pin();
    cpuid_ex(leaf, 0, &hi);
    cpuid_ex(c->max_basic, 0, &top);
    cpuid_ex(0x3FFFFFF0u, 0, &beyond);
    cpuid_unpin();
    t_info("leaf", "0x%x -> %08x %08x %08x %08x (max 0x%x -> %08x %08x %08x %08x)", leaf, hi.eax, hi.ebx, hi.ecx, hi.edx,
           c->max_basic, top.eax, top.ebx, top.ecx, top.edx);
    if (c->is_intel && !c->hypervisor) {
        /* SDM CPUID: above the maximum -> data of the highest basic leaf. */
        CHECKF(!memcmp(&hi, &top, sizeof hi), "Intel: CPUID(0x%x) must return the data of the highest basic leaf 0x%x", leaf, c->max_basic);
    } else if (c->is_intel) {
        if (memcmp(&hi, &top, sizeof hi)) t_note("hypervisor: CPUID(0x%x) is not the highest basic leaf data (SDM behaviour not emulated)", leaf);
    } else {
        t_note("non-Intel: CPUID(max_basic+1) result is vendor-defined (%s)", (hi.eax | hi.ebx | hi.ecx | hi.edx) ? "non-zero" : "zeros");
    }
    if (c->is_intel && !c->hypervisor) CHECKF(!memcmp(&beyond, &top, sizeof beyond), "Intel: CPUID(0x3FFFFFF0) must return the highest basic leaf data");
}
TEST(cpuid_above_max_extended_leaf) {
    const t_cpu_info *c = cpu();
    if (c->max_ext < 0x80000000u) SKIP("no extended range");
    uint32_t leaf = c->max_ext + 1;
    if (leaf >= 0x8FFFFFFFu) SKIP("extended range full");
    t_cpuid_regs hi, top;
    cpuid_pin();
    cpuid_ex(leaf, 0, &hi);
    cpuid_ex(c->max_basic, 0, &top);
    cpuid_unpin();
    t_info("leaf", "0x%x -> %08x %08x %08x %08x", leaf, hi.eax, hi.ebx, hi.ecx, hi.edx);
    if (c->is_intel && !c->hypervisor)
        CHECKF(!memcmp(&hi, &top, sizeof hi), "Intel: CPUID(0x%x) above the extended maximum must return the highest basic leaf data", leaf);
    else t_note("CPUID(max_ext+1) = %08x %08x %08x %08x", hi.eax, hi.ebx, hi.ecx, hi.edx);
}
TEST(cpuid_leaf7_invalid_subleaf_zero) {
    const t_cpu_info *c = cpu();
    if (c->max_basic < 7) SKIP("no leaf 7");
    uint32_t sub = c->l7_0.eax + 1;
    t_cpuid_regs r; cpuid_ex(7, sub, &r);
    t_info("leaf7", "max subleaf %u, subleaf %u -> %08x %08x %08x %08x", c->l7_0.eax, sub, r.eax, r.ebx, r.ecx, r.edx);
    if (c->is_intel && !c->hypervisor) CHECKF(!(r.eax | r.ebx | r.ecx | r.edx), "Intel: CPUID(7,%u) beyond the max subleaf must return zeros", sub);
    else t_note("non-Intel CPUID(7,%u) = %08x %08x %08x %08x", sub, r.eax, r.ebx, r.ecx, r.edx);
}
TEST(cpuid_leafd_invalid_subleaf_zero) {
    const t_cpu_info *c = cpu();
    if (c->max_basic < 0xD) SKIP("no leaf 0Dh");
    int checked = 0;
    for (uint32_t i = 2; i < 63; i++) {
        uint64_t all = c->xcr0_supported | c->xss_supported;
        if (all & (1ull << i)) continue;
        t_cpuid_regs r; cpuid_ex(0xD, i, &r);
        if (c->is_intel && !c->hypervisor) CHECKF(!(r.eax | r.ebx | r.ecx | r.edx), "Intel: CPUID(0Dh,%u) for an unsupported component must be zero (%08x %08x %08x %08x)",
                                i, r.eax, r.ebx, r.ecx, r.edx);
        else if (r.eax | r.ebx | r.ecx | r.edx) t_note("CPUID(0Dh,%u) non-zero on %s", i, c->vendor);
        checked++;
    }
    t_info("subleaves_checked", "%d", checked);
}
TEST(cpuid_consistent_on_every_cpu) {
    int n = t_cpu_count();
    int cap = (int)(sizeof(DWORD_PTR) * 8);
    if (n > cap) { t_note("checking the first %d of %d CPUs", cap, n); n = cap; }
    if (n < 1) SKIP("no CPU count");
    const t_cpu_info *c = cpu();
    t_cpuid_regs r1_0, r7_0, rd_0;
    uint64_t x0 = 0;
    int have = 0, pinned = 0;
    uint32_t apic_seen[64]; int napic = 0;
    for (int i = 0; i < n; i++) {
        uintptr_t prev = t_pin_thread(i);
        if (!prev) { t_note("could not pin to CPU %d", i); continue; }
        Sleep(0);
        DWORD pn = GetCurrentProcessorNumber();
        t_cpuid_regs r1, r7, rd;
        cpuid_ex(1, 0, &r1);
        cpuid_ex(7, 0, &r7);
        cpuid_ex(0xD, 0, &rd);
        uint64_t x = c->osxsave ? t_xgetbv(0) : 0;
        DWORD pn2 = GetCurrentProcessorNumber();
        t_unpin_thread(prev);
        pinned++;
        if (t_env_get()->active_groups <= 1) {
            if (t_is_wine() && (pn != (DWORD)i || pn2 != (DWORD)i)) t_expect_fail("Wine affinity/processor-number mismatch");
            CHECKF(pn == (DWORD)i && pn2 == (DWORD)i, "thread pinned to CPU %d ran on CPU %lu/%lu", i, pn, pn2);
        }
        uint32_t apic = r1.ebx >> 24;
        int dup = 0; for (int k = 0; k < napic; k++) dup |= apic_seen[k] == apic;
        if (!dup && napic < 64) apic_seen[napic++] = apic;
        if (!have) { r1_0 = r1; r7_0 = r7; rd_0 = rd; x0 = x; have = 1; continue; }
        CHECKF(r1.ecx == r1_0.ecx && r1.edx == r1_0.edx, "CPU %d: CPUID.1 ECX/EDX %08x/%08x differ from first CPU %08x/%08x",
               i, r1.ecx, r1.edx, r1_0.ecx, r1_0.edx);
        CHECKF(r7.ebx == r7_0.ebx && r7.ecx == r7_0.ecx && r7.edx == r7_0.edx,
               "CPU %d: CPUID.7.0 EBX/ECX/EDX %08x/%08x/%08x differ from first CPU %08x/%08x/%08x",
               i, r7.ebx, r7.ecx, r7.edx, r7_0.ebx, r7_0.ecx, r7_0.edx);
        CHECKF(rd.eax == rd_0.eax && rd.edx == rd_0.edx && rd.ebx == rd_0.ebx,
               "CPU %d: CPUID.0Dh.0 differs from first CPU", i);
        CHECKF(x == x0, "CPU %d: XCR0 0x%llx differs from first CPU 0x%llx", i, (unsigned long long)x, (unsigned long long)x0);
    }
    t_info("cpus", "%d pinned of %d, %d distinct initial APIC IDs", pinned, n, napic);
    if (napic != pinned) t_note("%d distinct CPUID.1:EBX[31:24] initial APIC IDs over %d CPUs (VM or >8-bit x2APIC IDs)", napic, pinned);
    REQUIREF(pinned > 0, "could not pin to any CPU");
}
