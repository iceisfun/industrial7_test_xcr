/* fp_environment - floating-point environment conformance and diagnostics.
 *
 * Covers: MXCSR (initial value, MXCSR_MASK/DAZ support, reserved bits, sticky flags, rounding control,
 * FTZ/DAZ), masked SSE results and flags for every exception class (bit-exact, all four rounding modes),
 * unmasked SSE and x87 exceptions delivered as EXCEPTION_FLT_* (child processes, VEH, resume through
 * the CONTEXT), the FP state seen in CONTEXT (handler, RtlCaptureContext, Get/SetThreadContext), x87
 * control/status/tag words, FNSTENV/FNSAVE/FXSAVE layouts, precision and rounding control, fxam/fcom/
 * fcomi condition codes, 80-bit values, MMX/x87 aliasing and EMMS, XRSTOR of MXCSR/x87, per-thread initial
 * FP environment and isolation under preemption, and agreement of the CRT (msvcrt _controlfp_s/_control87/
 * _statusfp/_clearfp/_fpreset, mingw <fenv.h>) with the hardware registers.
 *
 * Contracts hard-checked: the Intel SDM / AMD APM semantics of the instructions, the Windows x64 ABI
 * ("x64 calling convention / floating-point environment": FPCSR 0x027F and MXCSR 0x1F80 at start, control
 * bits nonvolatile), the documented CRT and C99 fenv behaviour, and documented Windows exception delivery.
 * Everything that is implementation-defined is recorded with t_info/t_note.
 *
 * Windows behaviour pinned in the Windows 10 19041 reference corpus (cited where used):
 *  [K1] ntoskrnl!KiXmmException (rva 0x40cf40, continuation at 0x40d224): unmasked = MXCSR flags & ~masks;
 *       64-bit code: IE->C0000090 INVALID_OPERATION, ZE->C000008E DIVIDE_BY_ZERO, DE->C0000090,
 *       OE->C0000091 OVERFLOW, UE->C0000093 UNDERFLOW, PE->C000008F INEXACT_RESULT (that priority order);
 *       32-bit code (CS==0x23, WOW64): IE/ZE/DE->C00002B5 FLOAT_MULTIPLE_TRAPS, OE/UE/PE->C00002B4
 *       FLOAT_MULTIPLE_FAULTS. KiExceptionDispatch is called with 2 parameters, [1] = MXCSR.
 *  [K2] ntoskrnl!KiFloatingErrorFault (rva 0x40bc00): x87 (both bitnesses): unmasked = FSW & ~FCW&0x3F;
 *       IE+SF->C0000092 STACK_CHECK, IE->C0000090, ZE->C000008E, DE->C0000090, OE->C0000091,
 *       UE->C0000093, PE->C000008F; 1 parameter.
 *  [K3] ntoskrnl!KxContextToKframes (rva 0x2d6b70): CONTEXT_FLOATING_POINT on x64 loads the trap-frame
 *       MXCSR from CONTEXT.MxCsr & KiMxCsrMask; for user mode CONTEXT.FltSave.MxCsr is overwritten with the
 *       live MXCSR and FCW &= 0x1F3F before FXRSTOR, i.e. CONTEXT.MxCsr is the authoritative field.
 *  [K4] ntdll!RtlCaptureContext (x64, rva 0xa11b0): fxsave [rcx+0x100]; stmxcsr [rcx+0x34];
 *       ContextFlags = 0x10000F (CONTROL|INTEGER|SEGMENTS|FLOATING_POINT).
 * These are used to choose what to CHECK; where Windows' mapping differs from the plain documented
 * meaning (DE reported as INVALID_OPERATION) both codes are accepted and the observed one is recorded. */
#include "t.h"
#include <float.h>
#include <fenv.h>
#include <process.h>

/* ======================================================================================
 * constants and small helpers
 * ====================================================================================== */
#define MXF_IE 0x01u
#define MXF_DE 0x02u
#define MXF_ZE 0x04u
#define MXF_OE 0x08u
#define MXF_UE 0x10u
#define MXF_PE 0x20u
#define MX_FLAGS 0x3Fu
#define MX_DAZ   0x40u
#define MX_MASKS 0x1F80u
#define MX_FTZ   0x8000u
#define MX_DEF   0x1F80u
#define MX_RC(m) ((unsigned)(m) << 13)
#define CW_RC(m) ((unsigned)(m) << 10)
#define CW_PC24  0x0000u
#define CW_PC53  0x0200u
#define CW_PC64  0x0300u
#define CW_MASKS 0x003Fu
#define CW_CMP   0x1F3Fu      /* defined FCW bits (6 and 13-15 are reserved) */
#define SW_C0 0x0100u
#define SW_C1 0x0200u
#define SW_C2 0x0400u
#define SW_C3 0x4000u
#define SW_CC (SW_C0 | SW_C1 | SW_C2 | SW_C3)
#define SW_SF 0x0040u
#define SW_ES 0x0080u
#define SW_TOP(sw) (((sw) >> 11) & 7u)
enum { RN = 0, RD = 1, RU = 2, RZ = 3 };
static const char *const k_rc[4] = { "RN", "RD", "RU", "RZ" };

#ifndef _SW_INEXACT          /* msvcrt float.h status bits (not always exposed by mingw's float.h) */
#define _SW_INEXACT    0x00000001u
#define _SW_UNDERFLOW  0x00000002u
#define _SW_OVERFLOW   0x00000004u
#define _SW_ZERODIVIDE 0x00000008u
#define _SW_INVALID    0x00000010u
#define _SW_DENORMAL   0x00080000u
#endif
#ifndef STATUS_FLOAT_MULTIPLE_FAULTS
#define STATUS_FLOAT_MULTIPLE_FAULTS 0xC00002B4u
#endif
#ifndef STATUS_FLOAT_MULTIPLE_TRAPS
#define STATUS_FLOAT_MULTIPLE_TRAPS 0xC00002B5u
#endif

static const char *flag_str(unsigned f, char *buf, size_t cap) {
    static const char *const n[6] = { "IE", "DE", "ZE", "OE", "UE", "PE" };
    size_t o = 0; buf[0] = 0;
    for (int i = 0; i < 6; i++)
        if (f & (1u << i)) o += (size_t)snprintf(buf + o, cap - o, "%s%s", o ? "|" : "", n[i]);
    if (!o) snprintf(buf, cap, "none");
    return buf;
}

static unsigned mxcsr_mask_raw(void) {
    static T_ALIGN(16) uint8_t fx[512];
    memset(fx, 0, sizeof fx);
    t_fxsave(fx);
    uint32_t m; memcpy(&m, fx + 28, 4);
    return m;
}
/* SDM 11.6.6: MXCSR_MASK == 0 means the default mask 0xFFBF (no DAZ). */
static unsigned mxcsr_mask(void) { unsigned m = mxcsr_mask_raw(); return m ? m : 0xFFBFu; }
static int daz_supported(void) { return (mxcsr_mask() & MX_DAZ) != 0; }

/* 80-bit extended values */
static t_f80 f80(uint16_t se, uint64_t m) { t_f80 r; memcpy(r.b, &m, 8); memcpy(r.b + 8, &se, 2); return r; }
/* constant initializer form (static tables) */
#define F80I(se, m) { { (uint8_t)((uint64_t)(m) >> 0), (uint8_t)((uint64_t)(m) >> 8), (uint8_t)((uint64_t)(m) >> 16), (uint8_t)((uint64_t)(m) >> 24), (uint8_t)((uint64_t)(m) >> 32), \
    (uint8_t)((uint64_t)(m) >> 40), (uint8_t)((uint64_t)(m) >> 48), (uint8_t)((uint64_t)(m) >> 56), (uint8_t)((se) & 0xFF), (uint8_t)((se) >> 8) } }
static uint16_t f80_se(const t_f80 *x) { uint16_t v; memcpy(&v, x->b + 8, 2); return v; }
static uint64_t f80_m(const t_f80 *x) { uint64_t v; memcpy(&v, x->b, 8); return v; }
static void check_f80(const char *what, const t_f80 *got, uint16_t se, uint64_t m) {
    CHECKF(f80_se(got) == se && f80_m(got) == m, "%s: got 0x%04X:%016llX want 0x%04X:%016llX", what,
           f80_se(got), (unsigned long long)f80_m(got), se, (unsigned long long)m);
}
#define XV(x) ((t_f80)x)
#define X_ONE   F80I(0x3FFF, 0x8000000000000000ull)
#define X_MONE  F80I(0xBFFF, 0x8000000000000000ull)
#define X_TWO   F80I(0x4000, 0x8000000000000000ull)
#define X_THREE F80I(0x4000, 0xC000000000000000ull)
#define X_ZERO  F80I(0x0000, 0)
#define X_HALF  F80I(0x3FFE, 0x8000000000000000ull)
#define X_INF   F80I(0x7FFF, 0x8000000000000000ull)
#define X_IND   F80I(0xFFFF, 0xC000000000000000ull)      /* real indefinite */
#define X_MAX   F80I(0x7FFE, 0xFFFFFFFFFFFFFFFFull)
#define X_MINN  F80I(0x0001, 0x8000000000000000ull)      /* smallest normal 2^-16382 */
#define X_DEN   F80I(0x0000, 0x0000000000000001ull)      /* smallest denormal */
#define X_QNAN  F80I(0x7FFF, 0xC000000000000123ull)
#define X_SNAN  F80I(0x7FFF, 0xA000000000000000ull)

typedef struct fp_snap { unsigned mx; unsigned short cw, sw, tw; } fp_snap;
static void fp_snapshot(fp_snap *s) { s->mx = t_get_mxcsr(); s->cw = t_get_x87_cw(); s->sw = t_get_x87_sw(); s->tw = t_get_x87_tw(); }

/* Default environment per the x64 ABI (also what Windows gives 32-bit threads): FCW 0x027F, MXCSR 0x1F80. */
static void check_default_env(const char *who, const fp_snap *s) {
    CHECKF(s->mx == MX_DEF, "%s: MXCSR 0x%08x want 0x1F80", who, s->mx);
    CHECKF((s->cw & 0x0F3Fu) == 0x023Fu, "%s: FCW 0x%04x want masks=all PC=53 RC=nearest (0x027F)", who, s->cw);
    CHECKF((s->sw & 0xB8FFu) == 0, "%s: FSW 0x%04x want TOP=0, no flags", who, s->sw);
    CHECKF(s->tw == 0xFFFF, "%s: tag word 0x%04x want 0xFFFF (all empty)", who, s->tw);
}

static int run_thread(LPTHREAD_START_ROUTINE fn, void *arg) {
    HANDLE h = CreateThread(NULL, 0, fn, arg, 0, NULL);
    if (!h) return -1;
    DWORD w = WaitForSingleObject(h, 5000);
    CloseHandle(h);
    return w == WAIT_OBJECT_0 ? 1 : 0;
}

static int child_hex(const t_child_result *r, const char *key, unsigned long long *v) {
    char pre[64], buf[128];
    snprintf(pre, sizeof pre, "%s=", key);
    if (!t_child_find(r, pre, buf, sizeof buf)) return 0;
    *v = strtoull(buf, NULL, 0);
    return 1;
}

/* label address for "the faulting instruction" (asm local label 1:) stored into a global */
#ifdef T_X64
#define LAB_TO(lbl, dst) "leaq " lbl "(%%rip), %%rax\n\tmovq %%rax, %[" dst "]\n\t"
#define LAB_CLOBBER "rax"
#else
#define LAB_TO(lbl, dst) "movl $" lbl ", %%eax\n\tmovl %%eax, %[" dst "]\n\t"
#define LAB_CLOBBER "eax"
#endif

/* ======================================================================================
 * SSE scalar execution engine: one instruction under a given MXCSR, returning result + flags.
 * The MXCSR load, the instruction and the MXCSR store are in one asm block; the caller's MXCSR is
 * restored afterwards.
 * ====================================================================================== */
enum { SO_ADDSS, SO_SUBSS, SO_MULSS, SO_DIVSS, SO_SQRTSS, SO_MAXSS, SO_MINSS,
       SO_ADDSD, SO_SUBSD, SO_MULSD, SO_DIVSD, SO_SQRTSD,
       SO_CVTSS2SI, SO_CVTTSS2SI, SO_CVTSD2SI, SO_CVTSI2SS, SO_CVTSD2SS, SO_COMISS, SO_UCOMISS };

#define SSE_SS(insn) __asm__ volatile("stmxcsr %[sv]\n\tmovd %[a], %%xmm0\n\tmovd %[b], %%xmm1\n\tldmxcsr %[mx]\n\t" \
    insn " %%xmm1, %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]\n\tmovd %%xmm0, %[r]" \
    : [sv] "=m"(sv), [mo] "=m"(mo), [r] "=m"(r32) : [mx] "m"(mx), [a] "m"(a32), [b] "m"(b32) : "xmm0", "xmm1")
#define SSE_SD(insn) __asm__ volatile("stmxcsr %[sv]\n\tmovq %[a], %%xmm0\n\tmovq %[b], %%xmm1\n\tldmxcsr %[mx]\n\t" \
    insn " %%xmm1, %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]\n\tmovq %%xmm0, %[r]" \
    : [sv] "=m"(sv), [mo] "=m"(mo), [r] "=m"(r64) : [mx] "m"(mx), [a] "m"(a), [b] "m"(b) : "xmm0", "xmm1")

static T_NOINLINE unsigned sse_exec(int op, uint64_t a, uint64_t b, unsigned mx, uint64_t *res) {
    unsigned sv = 0, mo = 0;
    uint32_t a32 = (uint32_t)a, b32 = (uint32_t)b, r32 = 0;
    uint64_t r64 = 0;
    int32_t ri = 0;
    unsigned char zf = 0, pf = 0, cf = 0;
    switch (op) {
    case SO_ADDSS: SSE_SS("addss"); *res = r32; break;
    case SO_SUBSS: SSE_SS("subss"); *res = r32; break;
    case SO_MULSS: SSE_SS("mulss"); *res = r32; break;
    case SO_DIVSS: SSE_SS("divss"); *res = r32; break;
    case SO_MAXSS: SSE_SS("maxss"); *res = r32; break;
    case SO_MINSS: SSE_SS("minss"); *res = r32; break;
    case SO_SQRTSS: b32 = a32; a32 = 0x12345678u; SSE_SS("sqrtss"); *res = r32; break;
    case SO_ADDSD: SSE_SD("addsd"); *res = r64; break;
    case SO_SUBSD: SSE_SD("subsd"); *res = r64; break;
    case SO_MULSD: SSE_SD("mulsd"); *res = r64; break;
    case SO_DIVSD: SSE_SD("divsd"); *res = r64; break;
    case SO_SQRTSD: b = a; a = 0x123456789ABCDEF0ull; SSE_SD("sqrtsd"); *res = r64; break;
    case SO_CVTSS2SI:
    case SO_CVTTSS2SI:
        if (op == SO_CVTSS2SI)
            __asm__ volatile("stmxcsr %[sv]\n\tmovd %[a], %%xmm0\n\tldmxcsr %[mx]\n\tcvtss2si %%xmm0, %[ri]\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]"
                             : [sv] "=m"(sv), [mo] "=m"(mo), [ri] "=r"(ri) : [mx] "m"(mx), [a] "m"(a32) : "xmm0");
        else
            __asm__ volatile("stmxcsr %[sv]\n\tmovd %[a], %%xmm0\n\tldmxcsr %[mx]\n\tcvttss2si %%xmm0, %[ri]\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]"
                             : [sv] "=m"(sv), [mo] "=m"(mo), [ri] "=r"(ri) : [mx] "m"(mx), [a] "m"(a32) : "xmm0");
        *res = (uint32_t)ri; break;
    case SO_CVTSD2SI:
        __asm__ volatile("stmxcsr %[sv]\n\tmovq %[a], %%xmm0\n\tldmxcsr %[mx]\n\tcvtsd2si %%xmm0, %[ri]\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]"
                         : [sv] "=m"(sv), [mo] "=m"(mo), [ri] "=r"(ri) : [mx] "m"(mx), [a] "m"(a) : "xmm0");
        *res = (uint32_t)ri; break;
    case SO_CVTSI2SS:
        __asm__ volatile("stmxcsr %[sv]\n\tldmxcsr %[mx]\n\tcvtsi2ssl %[a], %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]\n\tmovd %%xmm0, %[r]"
                         : [sv] "=m"(sv), [mo] "=m"(mo), [r] "=m"(r32) : [mx] "m"(mx), [a] "m"(a32) : "xmm0");
        *res = r32; break;
    case SO_CVTSD2SS:
        __asm__ volatile("stmxcsr %[sv]\n\tmovq %[a], %%xmm1\n\tldmxcsr %[mx]\n\tcvtsd2ss %%xmm1, %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]\n\tmovd %%xmm0, %[r]"
                         : [sv] "=m"(sv), [mo] "=m"(mo), [r] "=m"(r32) : [mx] "m"(mx), [a] "m"(a) : "xmm0", "xmm1");
        *res = r32; break;
    case SO_COMISS:
        __asm__ volatile("stmxcsr %[sv]\n\tmovd %[a], %%xmm0\n\tmovd %[b], %%xmm1\n\tldmxcsr %[mx]\n\tcomiss %%xmm1, %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]"
                         : [sv] "=m"(sv), [mo] "=m"(mo), "=@ccz"(zf), "=@ccp"(pf), "=@ccc"(cf) : [mx] "m"(mx), [a] "m"(a32), [b] "m"(b32) : "xmm0", "xmm1");
        *res = (unsigned)zf << 2 | (unsigned)pf << 1 | cf; break;
    case SO_UCOMISS:
        __asm__ volatile("stmxcsr %[sv]\n\tmovd %[a], %%xmm0\n\tmovd %[b], %%xmm1\n\tldmxcsr %[mx]\n\tucomiss %%xmm1, %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]"
                         : [sv] "=m"(sv), [mo] "=m"(mo), "=@ccz"(zf), "=@ccp"(pf), "=@ccc"(cf) : [mx] "m"(mx), [a] "m"(a32), [b] "m"(b32) : "xmm0", "xmm1");
        *res = (unsigned)zf << 2 | (unsigned)pf << 1 | cf; break;
    default: *res = 0; break;
    }
    return mo;
}

/* A test vector: op(a, b) under MXCSR 0x1F80|mx|RC for RC = RN, RD, RU, RZ. r[] = expected results
 * (bit patterns; COMISS: ZF<<2|PF<<1|CF), fl = expected sticky flags (identical in every mode).
 * Expected values were computed with exact rational arithmetic (IEEE 754 rounding, x86 tininess after
 * rounding, SSE NaN rules: first source operand wins, default NaN 0xFFC00000/0xFFF8000000000000). */
typedef struct sse_vec { const char *what; int op; uint64_t a, b; unsigned mx; uint64_t r[4]; unsigned fl; } sse_vec;
#define R1(x) { x, x, x, x }

static void run_sse_vecs(const sse_vec *v, size_t n) {
    char f1[40], f2[40];
    for (size_t i = 0; i < n; i++)
        for (int m = 0; m < 4; m++) {
            uint64_t got = 0;
            unsigned mx = MX_DEF | v[i].mx | MX_RC(m);
            unsigned mo = sse_exec(v[i].op, v[i].a, v[i].b, mx, &got);
            CHECKF(got == v[i].r[m], "%s [%s mxcsr=0x%04x]: result 0x%llx want 0x%llx", v[i].what, k_rc[m], mx,
                   (unsigned long long)got, (unsigned long long)v[i].r[m]);
            CHECKF((mo & MX_FLAGS) == v[i].fl, "%s [%s mxcsr=0x%04x]: flags %s want %s", v[i].what, k_rc[m], mx,
                   flag_str(mo & MX_FLAGS, f1, sizeof f1), flag_str(v[i].fl, f2, sizeof f2));
            CHECKF((mo & ~MX_FLAGS) == mx, "%s: control bits changed by the instruction: 0x%04x -> 0x%04x", v[i].what, mx, mo);
        }
}

/* ======================================================================================
 * x87 execution helpers (operands as exact 80-bit values; fninit first, caller CW restored)
 * ====================================================================================== */
enum { XO_ADD, XO_SUB, XO_MUL, XO_DIV, XO_SQRT, XO_RNDINT };
#define X87_OP(seq) __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldt %[b]\n\tfldt %[a]\n\tfldcw %[cw]\n\t" seq \
    "\n\tfnstsw %[sw]\n\tfstpt %[r]\n\tfstp %%st(0)\n\tfninit\n\tfldcw %[sv]" \
    : [sv] "=m"(sv), [sw] "=m"(sw), [r] "=m"(*r) : [a] "m"(*a), [b] "m"(*b), [cw] "m"(cw))
/* st(0) = a, st(1) = b; result = st(0) after the op. Returns FSW right after the op. */
static T_NOINLINE unsigned short x87_op(int op, const t_f80 *a, const t_f80 *b, unsigned short cw, t_f80 *r) {
    unsigned short sv = 0, sw = 0;
    switch (op) {
    case XO_ADD: X87_OP("fadd %%st(1), %%st"); break;
    case XO_SUB: X87_OP("fsub %%st(1), %%st"); break;
    case XO_MUL: X87_OP("fmul %%st(1), %%st"); break;
    case XO_DIV: X87_OP("fdiv %%st(1), %%st"); break;
    case XO_SQRT: X87_OP("fsqrt"); break;
    case XO_RNDINT: X87_OP("frndint"); break;
    }
    return sw;
}
static T_NOINLINE unsigned short x87_fistp64(const t_f80 *a, unsigned short cw, int64_t *r) {
    unsigned short sv, sw;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldt %[a]\n\tfldcw %[cw]\n\tfistpll %[r]\n\tfnstsw %[sw]\n\tfninit\n\tfldcw %[sv]"
                     : [sv] "=m"(sv), [sw] "=m"(sw), [r] "=m"(*r) : [a] "m"(*a), [cw] "m"(cw));
    return sw;
}
static T_NOINLINE unsigned short x87_fst32(const t_f80 *a, unsigned short cw, uint32_t *r) {
    unsigned short sv, sw;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldt %[a]\n\tfldcw %[cw]\n\tfstps %[r]\n\tfnstsw %[sw]\n\tfninit\n\tfldcw %[sv]"
                     : [sv] "=m"(sv), [sw] "=m"(sw), [r] "=m"(*r) : [a] "m"(*a), [cw] "m"(cw));
    return sw;
}
static T_NOINLINE unsigned short x87_fld32(uint32_t a, unsigned short cw, t_f80 *r) {
    unsigned short sv, sw;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldcw %[cw]\n\tflds %[a]\n\tfnstsw %[sw]\n\tfstpt %[r]\n\tfninit\n\tfldcw %[sv]"
                     : [sv] "=m"(sv), [sw] "=m"(sw), [r] "=m"(*r) : [a] "m"(a), [cw] "m"(cw));
    return sw;
}
/* compare st(0)=a with st(1)=b. kind: 0 fcom, 1 fucom, 2 fcomi, 3 fucomi. eflags: ZF<<2|PF<<1|CF */
static T_NOINLINE unsigned short x87_cmp(int kind, const t_f80 *a, const t_f80 *b, unsigned *eflags) {
    unsigned short sv, sw;
    unsigned char zf = 0, pf = 0, cf = 0;
#define X87_CMP(insn) __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldt %[b]\n\tfldt %[a]\n\t" insn "\n\tfnstsw %[sw]\n\tfstp %%st(0)\n\tfstp %%st(0)\n\tfninit\n\tfldcw %[sv]" \
        : [sv] "=m"(sv), [sw] "=m"(sw), "=@ccz"(zf), "=@ccp"(pf), "=@ccc"(cf) : [a] "m"(*a), [b] "m"(*b))
    switch (kind) {
    case 0: X87_CMP("fcom %%st(1)"); break;
    case 1: X87_CMP("fucom %%st(1)"); break;
    case 2: X87_CMP("fcomi %%st(1), %%st"); break;
    default: X87_CMP("fucomi %%st(1), %%st"); break;
    }
    *eflags = (unsigned)zf << 2 | (unsigned)pf << 1 | cf;
    return sw;
}
static T_NOINLINE unsigned short x87_fxam(const t_f80 *a, int empty) {
    unsigned short sv, sw;
    if (empty)
        __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfxam\n\tfnstsw %[sw]\n\tfninit\n\tfldcw %[sv]" : [sv] "=m"(sv), [sw] "=m"(sw));
    else
        __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldt %[a]\n\tfxam\n\tfnstsw %[sw]\n\tfninit\n\tfldcw %[sv]"
                         : [sv] "=m"(sv), [sw] "=m"(sw) : [a] "m"(*a));
    return sw;
}
static T_NOINLINE void x87_roundtrip(const t_f80 *a, t_f80 *r, unsigned short *sw) {
    unsigned short sv;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldt %[a]\n\tfstpt %[r]\n\tfnstsw %[sw]\n\tfninit\n\tfldcw %[sv]"
                     : [sv] "=m"(sv), [sw] "=m"(*sw), [r] "=m"(*r) : [a] "m"(*a));
}

/* ======================================================================================
 * 1. initial environment and threads
 * ====================================================================================== */
TEST(initial_env_case_thread) {
    /* The harness restores the process-initial MXCSR (flags cleared) before every case; the CRT does not
     * touch MXCSR, so this is the OS-provided start value (x64 ABI: 0x1F80). The x87 CW on the main thread
     * is CRT-defined (mingw's startup _fpreset is FNINIT -> 0x037F; MSVC CRTs use 0x027F): record only. */
    unsigned mx = t_get_mxcsr();
    CHECKF(mx == MX_DEF, "MXCSR at case start 0x%08x want 0x1F80", mx);
    t_info("main_thread_fcw", "0x%04x", t_get_x87_cw());
    t_info("mxcsr_mask", "0x%08x (raw 0x%08x)", mxcsr_mask(), mxcsr_mask_raw());
    t_info("daz_supported", "%d", daz_supported());
}

static DWORD WINAPI th_snap(void *p) { fp_snapshot((fp_snap *)p); return 0; }

TEST(new_thread_default_env) {
    /* x64 ABI: a new thread starts with FPCSR 0x027F and MXCSR 0x1F80 (flags clear); x87 stack empty. */
    for (int i = 0; i < 8; i++) {
        fp_snap s; memset(&s, 0xCC, sizeof s);
        int rc = run_thread(th_snap, &s);
        if (rc < 0) SKIP("CreateThread failed: %lu", GetLastError());
        REQUIREF(rc == 1, "thread %d did not finish", i);
        char who[48]; snprintf(who, sizeof who, "new thread #%d", i);
        check_default_env(who, &s);
        if (i == 0) t_info("new_thread_env", "mxcsr=0x%04x fcw=0x%04x fsw=0x%04x ftw=0x%04x", s.mx, s.cw, s.sw, s.tw);
    }
}

TEST(new_thread_not_inherit_parent_env) {
    /* The creating thread's non-default environment (rounding, FTZ/DAZ, sticky flags, x87 precision,
     * rounding and a non-empty stack with flags) must not be inherited by the new thread. */
    unsigned mx = MX_DEF | MX_RC(RZ) | MX_FTZ | (daz_supported() ? MX_DAZ : 0) | MX_FLAGS;
    t_set_mxcsr(mx);
    t_set_x87_cw((unsigned short)(0x007F | CW_RC(RU) | CW_PC24));
    fp_snap s; memset(&s, 0xCC, sizeof s);
    int rc = run_thread(th_snap, &s);
    unsigned mx_after = t_get_mxcsr();
    unsigned short cw_after = t_get_x87_cw();
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    if (rc < 0) SKIP("CreateThread failed: %lu", GetLastError());
    REQUIRE(rc == 1);
    check_default_env("thread created by a thread with a dirty env", &s);
    CHECKF(mx_after == mx, "creator MXCSR changed across CreateThread/wait: 0x%04x -> 0x%04x", mx, mx_after);
    CHECKF((cw_after & CW_CMP) == ((0x007F | CW_RC(RU) | CW_PC24) & CW_CMP), "creator FCW changed: 0x%04x", cw_after);
}

typedef struct nest_arg { fp_snap inner; int rc; } nest_arg;
static DWORD WINAPI th_dirty_then_spawn(void *p) {
    nest_arg *a = (nest_arg *)p;
    t_set_mxcsr(MX_DEF | MX_RC(RD) | MX_FTZ | MXF_PE | MXF_ZE);
    t_set_x87_cw((unsigned short)(0x007F | CW_RC(RZ) | CW_PC53));
    a->rc = run_thread(th_snap, &a->inner);
    return 0;
}
TEST(new_thread_from_worker_gets_default_env) {
    nest_arg a; memset(&a, 0xCC, sizeof a); a.rc = -2;
    int rc = run_thread(th_dirty_then_spawn, &a);
    if (rc < 0 || a.rc == -1) SKIP("CreateThread failed");
    REQUIRE(rc == 1 && a.rc == 1);
    check_default_env("grandchild thread", &a.inner);
}

static DWORD WINAPI th_dirty_self(void *p) {
    t_set_mxcsr(MX_DEF | MX_RC(RU) | MX_FTZ | MXF_IE);
    t_set_x87_cw((unsigned short)(0x007F | CW_RC(RD) | CW_PC24));
    fp_snapshot((fp_snap *)p);
    return 0;
}
TEST(thread_env_change_not_visible_to_others) {
    fp_snap before, after, t; memset(&t, 0, sizeof t);
    fp_snapshot(&before);
    int rc = run_thread(th_dirty_self, &t);
    if (rc < 0) SKIP("CreateThread failed");
    REQUIRE(rc == 1);
    fp_snapshot(&after);
    CHECKF(t.mx == (MX_DEF | MX_RC(RU) | MX_FTZ | MXF_IE), "worker's own MXCSR 0x%04x", t.mx);
    CHECKF(after.mx == before.mx, "MXCSR leaked from worker: 0x%04x -> 0x%04x", before.mx, after.mx);
    CHECKF(after.cw == before.cw, "FCW leaked from worker: 0x%04x -> 0x%04x", before.cw, after.cw);
    /* and a fresh thread after the dirty one exited still gets defaults */
    fp_snap s; memset(&s, 0xCC, sizeof s);
    REQUIRE(run_thread(th_snap, &s) == 1);
    check_default_env("thread created after a dirty thread exited", &s);
}

static unsigned __stdcall th_snap_crt(void *p) { fp_snapshot((fp_snap *)p); return 0; }
TEST(crt_beginthreadex_env) {
    t_set_mxcsr(MX_DEF | MX_RC(RZ) | MX_FTZ);
    fp_snap s; memset(&s, 0xCC, sizeof s);
    uintptr_t h = _beginthreadex(NULL, 0, th_snap_crt, &s, 0, NULL);
    t_set_mxcsr(MX_DEF);
    if (!h) SKIP("_beginthreadex failed");
    DWORD w = WaitForSingleObject((HANDLE)h, 5000);
    CloseHandle((HANDLE)h);
    REQUIRE(w == WAIT_OBJECT_0);
    CHECKF(s.mx == MX_DEF, "_beginthreadex thread MXCSR 0x%04x want 0x1F80", s.mx);
    CHECKF((s.cw & 0x0C3F) == 0x003F, "_beginthreadex thread FCW 0x%04x: want all masked, round-to-nearest", s.cw);
    t_info("beginthreadex_fcw", "0x%04x", s.cw);
}

/* ---- env spin: load env, put a value and a flag on the x87 stack, spin (GPR only), store env ---- */
typedef struct spin_args {
    unsigned mx_in; unsigned short cw_in;
    volatile long *stop, *ready;
    unsigned long max;
    unsigned mx_out; unsigned short cw_out, sw_out;
} spin_args;
static T_NOINLINE void env_spin(spin_args *s) {
    unsigned sv; unsigned short svcw;
    unsigned long n = s->max;
    __asm__ volatile(
        "stmxcsr %[sv]\n\tfnstcw %[svcw]\n\t"
        "fninit\n\tfldcw %[cw]\n\tldmxcsr %[mx]\n\t"
        "fldz\n\tfld1\n\tfdiv %%st(1), %%st\n\t"        /* masked 1/0: FSW.ZE, TOP=6 */
        "lock incl %[ready]\n"
        "1:\n\tpause\n\tcmpl $0, %[stop]\n\tjne 2f\n\tsub $1, %[n]\n\tjnz 1b\n"
        "2:\n\tstmxcsr %[mo]\n\tfnstcw %[co]\n\tfnstsw %[so]\n\t"
        "fninit\n\tfldcw %[svcw]\n\tldmxcsr %[sv]"
        : [sv] "=&m"(sv), [svcw] "=&m"(svcw), [mo] "=m"(s->mx_out), [co] "=m"(s->cw_out), [so] "=m"(s->sw_out),
          [n] "+r"(n), [ready] "+m"(*s->ready)
        : [cw] "m"(s->cw_in), [mx] "m"(s->mx_in), [stop] "m"(*s->stop)
        : "cc", "memory");
}
static DWORD WINAPI th_env_spin(void *p) { env_spin((spin_args *)p); return 0; }

static int wait_count(volatile long *v, long want, DWORD ms) {
    ULONGLONG t0 = GetTickCount64();
    while (*v < want) { if (GetTickCount64() - t0 > ms) return 0; Sleep(1); }
    return 1;
}

TEST(threads_env_isolated_under_preemption) {
    /* More spinning threads than CPUs, each with its own MXCSR (RC, FTZ, DAZ, sticky flags) and FCW
     * (RC, PC) plus an x87 stack with FSW.ZE: after forced time slicing every thread must see exactly its
     * own state (status flags included: they are thread state, preserved by context switches). */
    int ncpu = t_env_get()->ncpu;
    int n = 2 * ncpu + 2; if (n > 64) n = 64; if (n < 4) n = 4;
    static spin_args a[64]; static HANDLE h[64];
    volatile long stop = 0, ready = 0;
    int daz = daz_supported(), started = 0;
    static const unsigned short pcs[3] = { CW_PC24, CW_PC53, CW_PC64 };
    for (int i = 0; i < n; i++) {
        memset(&a[i], 0, sizeof a[i]);
        a[i].mx_in = MX_DEF | MX_RC(i & 3) | ((i >> 2) & 1 ? MX_FTZ : 0) | (daz && ((i >> 3) & 1) ? MX_DAZ : 0) | ((unsigned)(i * 7 + 1) & MX_FLAGS);
        a[i].cw_in = (unsigned short)(0x007F | CW_RC((i + 1) & 3) | pcs[i % 3]);
        a[i].stop = &stop; a[i].ready = &ready; a[i].max = 0x7FFFFFFFul;
        h[i] = CreateThread(NULL, 64 * 1024, th_env_spin, &a[i], 0, NULL);
        if (!h[i]) break;
        started++;
    }
    if (started < 2) { stop = 1; for (int i = 0; i < started; i++) { WaitForSingleObject(h[i], 5000); CloseHandle(h[i]); } SKIP("could not create threads"); }
    int all_ready = wait_count(&ready, started, 5000);
    for (int k = 0; k < 20; k++) { Sleep(10); SwitchToThread(); }
    stop = 1;
    for (int i = 0; i < started; i++) { CHECKF(WaitForSingleObject(h[i], 5000) == WAIT_OBJECT_0, "thread %d did not stop", i); CloseHandle(h[i]); }
    CHECKF(all_ready, "only %ld of %d threads reached the spin", ready, started);
    for (int i = 0; i < started; i++) {
        CHECKF(a[i].mx_out == a[i].mx_in, "thread %d: MXCSR 0x%04x want 0x%04x", i, a[i].mx_out, a[i].mx_in);
        CHECKF((a[i].cw_out & CW_CMP) == (a[i].cw_in & CW_CMP), "thread %d: FCW 0x%04x want 0x%04x", i, a[i].cw_out, a[i].cw_in);
        CHECKF((a[i].sw_out & 0x38FFu) == 0x3004u, "thread %d: FSW 0x%04x want TOP=6 + ZE", i, a[i].sw_out);
    }
    t_info("threads", "%d (ncpu %d)", started, ncpu);
}

static void wait_ms_event(DWORD ms) { HANDLE e = CreateEventA(NULL, TRUE, FALSE, NULL); if (e) { WaitForSingleObject(e, ms); CloseHandle(e); } }
static DWORD WINAPI th_noop(void *p) { (void)p; t_set_mxcsr(MX_DEF | MX_RC(RD)); return 0; }

TEST(env_control_preserved_across_waits) {
    /* x64 ABI: MXCSR control bits and the x87 control word are nonvolatile, so every API call must
     * return with them unchanged (status bits are volatile and not compared). */
    unsigned mx = MX_DEF | MX_RC(RZ) | MX_FTZ | (daz_supported() ? MX_DAZ : 0);
    unsigned short cw = (unsigned short)(0x007F | CW_RC(RU) | CW_PC24);
    static const char *const what[8] = { "Sleep(0)", "Sleep(1)", "SwitchToThread", "WaitForSingleObject(5ms)",
                                         "CreateThread+join", "VirtualAlloc/Free", "QueryPerformanceCounter", "GetSystemTimeAsFileTime" };
    unsigned bad_mx = 0, bad_cw = 0, got_mx = 0, got_cw = 0;
    int first_bad = -1;
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    for (int rep = 0; rep < 5; rep++)
        for (int k = 0; k < 8; k++) {
            switch (k) {
            case 0: Sleep(0); break;
            case 1: Sleep(1); break;
            case 2: SwitchToThread(); break;
            case 3: wait_ms_event(5); break;
            case 4: { HANDLE th = CreateThread(NULL, 0, th_noop, NULL, 0, NULL); if (th) { WaitForSingleObject(th, 5000); CloseHandle(th); } break; }
            case 5: { void *p = VirtualAlloc(NULL, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); if (p) VirtualFree(p, 0, MEM_RELEASE); break; }
            case 6: { LARGE_INTEGER q; QueryPerformanceCounter(&q); break; }
            default: { FILETIME ft; GetSystemTimeAsFileTime(&ft); break; }
            }
            unsigned m2 = t_get_mxcsr() & ~MX_FLAGS; unsigned short c2 = t_get_x87_cw();
            if (m2 != mx) { bad_mx++; if (first_bad < 0) { first_bad = k; got_mx = m2; } }
            if ((c2 & CW_CMP) != (cw & CW_CMP)) { bad_cw++; if (first_bad < 0) first_bad = k; got_cw = c2; }
        }
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    CHECKF(bad_mx == 0, "MXCSR control bits changed %u times (first after %s: 0x%04x want 0x%04x)", bad_mx, first_bad >= 0 ? what[first_bad] : "?", got_mx, mx);
    CHECKF(bad_cw == 0, "FCW changed %u times (first after %s: 0x%04x want 0x%04x)", bad_cw, first_bad >= 0 ? what[first_bad] : "?", got_cw, cw);
}

/* ======================================================================================
 * 2. MXCSR architecture
 * ====================================================================================== */
TEST(mxcsr_mask_from_fxsave) {
    /* SDM 11.6.6: FXSAVE bytes 28-31 hold MXCSR_MASK (0 => 0xFFBF). Bits 0-15 except DAZ are defined
     * on every SSE2 processor; bits 16-31 are reserved (0 in the mask). FXSAVE byte 24 = MXCSR. */
    static T_ALIGN(16) uint8_t fx[512];
    unsigned raw = mxcsr_mask_raw(), m = mxcsr_mask();
    CHECKF((m & 0xFFBFu) == 0xFFBFu, "MXCSR_MASK 0x%08x lacks architectural bits (want 0xFFBF set)", m);
    /* bit 16 (MXCSR.FZ16-like) stays clear everywhere; AMD/Hygon Zen reports bit 17 (MXCSR.MM, misaligned SSE) writable */
    CHECKF((m & 0x10000u) == 0, "MXCSR_MASK 0x%08x has reserved bit 16 set", m);
    if (cpu()->is_amd) CHECKF((m & 0xFFFD0000u) == 0, "MXCSR_MASK 0x%08x has reserved bits 18-31 set (AMD: only bit 17 MM allowed)", m);
    else CHECKF((m & 0xFFFF0000u) == 0, "MXCSR_MASK 0x%08x has reserved bits 16-31 set", m);
    for (int i = 0; i < 4; i++) {
        unsigned v = MX_DEF | MX_RC(i) | (i & 1 ? MX_FTZ : 0) | (unsigned)i;
        t_set_mxcsr(v);
        memset(fx, 0xA5, sizeof fx);
        t_fxsave(fx);
        uint32_t s; memcpy(&s, fx + 24, 4);
        CHECKF(s == v, "FXSAVE.MXCSR 0x%08x want 0x%08x", s, v);
        uint32_t m2; memcpy(&m2, fx + 28, 4);
        CHECKF(m2 == raw, "MXCSR_MASK not stable: 0x%08x vs 0x%08x", m2, raw);
    }
    t_set_mxcsr(MX_DEF);
    t_info("mxcsr_mask_raw", "0x%08x", raw);
    if (!(m & MX_DAZ)) t_note("processor does not support DAZ (MXCSR_MASK bit 6 clear)");
}

TEST(mxcsr_writable_bits_roundtrip) {
    /* every bit set in MXCSR_MASK must load and read back (flags and masks included); walking ones/zeros */
    /* The default MXCSR is restored before every CHECK: some of these values unmask exceptions, and the
     * harness code that runs between checks must never execute under them. */
    unsigned m = mxcsr_mask() & 0xFFFFu;
    for (int b = 0; b < 16; b++) {
        if (!(m & (1u << b))) continue;
        unsigned v1 = MX_DEF ^ (1u << b);
        t_set_mxcsr(v1);
        unsigned g1 = t_get_mxcsr();
        t_set_mxcsr(MX_DEF);
        CHECKF(g1 == v1, "bit %d: wrote 0x%04x read 0x%04x", b, v1, g1);
    }
    unsigned all = m;
    t_set_mxcsr(all);
    unsigned ga = t_get_mxcsr();
    t_set_mxcsr(0);
    unsigned gz = t_get_mxcsr();
    t_set_mxcsr(MX_DEF);
    CHECKF(ga == all, "all mask bits: wrote 0x%04x read 0x%04x", all, ga);
    CHECKF(gz == 0, "zero: read 0x%04x", gz);
}

TEST(mxcsr_flags_sticky_and_accumulate) {
    /* SDM 11.5.1: flags are sticky - a later non-exceptional operation does not clear them, and later
     * exceptions OR in. Done in one asm block so nothing else touches MXCSR. */
    unsigned m1, m2, m3, m4, sv;
    uint32_t one = 0x3F800000u, zero = 0, three = 0x40400000u, two = 0x40000000u;
    __asm__ volatile(
        "stmxcsr %[sv]\n\tldmxcsr %[def]\n\t"
        "movd %[one], %%xmm0\n\tmovd %[zero], %%xmm1\n\tdivss %%xmm1, %%xmm0\n\tstmxcsr %[m1]\n\t"   /* ZE */
        "movd %[one], %%xmm0\n\tmovd %[two], %%xmm1\n\taddss %%xmm1, %%xmm0\n\tstmxcsr %[m2]\n\t"     /* exact */
        "movd %[one], %%xmm0\n\tmovd %[three], %%xmm1\n\tdivss %%xmm1, %%xmm0\n\tstmxcsr %[m3]\n\t"   /* PE */
        "xorps %%xmm0, %%xmm0\n\txorps %%xmm1, %%xmm1\n\tdivss %%xmm1, %%xmm0\n\tstmxcsr %[m4]\n\t"   /* IE */
        "ldmxcsr %[sv]"
        : [sv] "=&m"(sv), [m1] "=m"(m1), [m2] "=m"(m2), [m3] "=m"(m3), [m4] "=m"(m4)
        : [def] "m"((unsigned){ MX_DEF }), [one] "m"(one), [zero] "m"(zero), [two] "m"(two), [three] "m"(three)
        : "xmm0", "xmm1");
    CHECKF(m1 == (MX_DEF | MXF_ZE), "after 1/0: 0x%04x", m1);
    CHECKF(m2 == (MX_DEF | MXF_ZE), "after exact add: 0x%04x (ZE must stay)", m2);
    CHECKF(m3 == (MX_DEF | MXF_ZE | MXF_PE), "after 1/3: 0x%04x", m3);
    CHECKF(m4 == (MX_DEF | MXF_ZE | MXF_PE | MXF_IE), "after 0/0: 0x%04x", m4);
}

TEST(mxcsr_ldmxcsr_flags_no_exception_until_op) {
    /* Loading MXCSR with a flag set whose exception is unmasked does not by itself raise #XM (only the
     * next SSE arithmetic instruction that detects an exception does): ldmxcsr/stmxcsr only. */
    unsigned v = (MX_DEF & ~0x0200u) | MXF_ZE, got, sv;
    __asm__ volatile("stmxcsr %[sv]\n\tldmxcsr %[v]\n\tstmxcsr %[g]\n\tldmxcsr %[sv]" : [sv] "=&m"(sv), [g] "=m"(got) : [v] "m"(v));
    CHECKF(got == v, "read back 0x%04x want 0x%04x", got, v);
}

TEST(sse_flags_or_across_packed_lanes) {
    /* packed: flags are the OR over all lanes (lane 1 divides by zero, lane 3 is inexact) */
    T_ALIGN(16) uint32_t a[4] = { 0x3F800000u, 0x3F800000u, 0x40000000u, 0x3F800000u };
    T_ALIGN(16) uint32_t b[4] = { 0x3F800000u, 0x00000000u, 0x40000000u, 0x40400000u };
    T_ALIGN(16) uint32_t r[4];
    unsigned mo, sv, mx = MX_DEF;
    __asm__ volatile("stmxcsr %[sv]\n\tldmxcsr %[mx]\n\tmovaps %[a], %%xmm0\n\tdivps %[b], %%xmm0\n\tstmxcsr %[mo]\n\tmovaps %%xmm0, %[r]\n\tldmxcsr %[sv]"
                     : [sv] "=&m"(sv), [mo] "=m"(mo), [r] "=m"(r) : [mx] "m"(mx), [a] "m"(a), [b] "m"(b) : "xmm0");
    CHECKF((mo & MX_FLAGS) == (MXF_ZE | MXF_PE), "divps flags 0x%02x want ZE|PE", mo & MX_FLAGS);
    CHECK_EQ(r[0], 0x3F800000u); CHECK_EQ(r[1], 0x7F800000u); CHECK_EQ(r[2], 0x3F800000u); CHECK_EQ(r[3], 0x3EAAAAABu);
    /* lane-wise SNaN in lane 2 only: IE, quieted in that lane */
    uint32_t sn[4] = { 0x3F800000u, 0x3F800000u, 0x7F800001u, 0x3F800000u };
    memcpy(a, sn, sizeof a);
    uint32_t ones[4] = { 0x3F800000u, 0x3F800000u, 0x3F800000u, 0x3F800000u };
    memcpy(b, ones, sizeof b);
    __asm__ volatile("stmxcsr %[sv]\n\tldmxcsr %[mx]\n\tmovaps %[a], %%xmm0\n\tmulps %[b], %%xmm0\n\tstmxcsr %[mo]\n\tmovaps %%xmm0, %[r]\n\tldmxcsr %[sv]"
                     : [sv] "=&m"(sv), [mo] "=m"(mo), [r] "=m"(r) : [mx] "m"(mx), [a] "m"(a), [b] "m"(b) : "xmm0");
    CHECKF((mo & MX_FLAGS) == MXF_IE, "mulps SNaN lane flags 0x%02x want IE", mo & MX_FLAGS);
    CHECK_EQ(r[2], 0x7FC00001u); CHECK_EQ(r[0], 0x3F800000u); CHECK_EQ(r[3], 0x3F800000u);
}

TEST(sse_and_x87_state_independent) {
    /* MXCSR and the x87 CW/SW are separate registers: an SSE exception does not set FSW flags, an x87
     * exception does not set MXCSR flags, and loading one control register leaves the other alone. */
    uint64_t r;
    t_fnclex();
    unsigned mo = sse_exec(SO_DIVSS, 0x3F800000u, 0, MX_DEF, &r);
    CHECK(mo & MXF_ZE);
    CHECKF((t_get_x87_sw() & 0x3F) == 0, "SSE 1/0 set x87 FSW flags: 0x%04x", t_get_x87_sw());
    t_set_mxcsr(MX_DEF);
    t_f80 one = X_ONE, zero = X_ZERO, res;
    unsigned short sw = x87_op(XO_DIV, &one, &zero, 0x037F, &res);
    CHECK(sw & MXF_ZE);
    CHECKF((t_get_mxcsr() & MX_FLAGS) == 0, "x87 1/0 set MXCSR flags: 0x%04x", t_get_mxcsr());
    unsigned short cw0 = t_get_x87_cw();
    t_set_mxcsr(MX_DEF | MX_RC(RZ) | MX_FTZ);
    CHECKF(t_get_x87_cw() == cw0, "LDMXCSR changed FCW 0x%04x -> 0x%04x", cw0, t_get_x87_cw());
    t_set_x87_cw((unsigned short)(0x007F | CW_RC(RD) | CW_PC24));
    CHECKF(t_get_mxcsr() == (MX_DEF | MX_RC(RZ) | MX_FTZ), "FLDCW changed MXCSR: 0x%04x", t_get_mxcsr());
    /* x87 rounding does not affect SSE arithmetic and vice versa */
    mo = sse_exec(SO_DIVSS, 0x3F800000u, 0x40400000u, MX_DEF, &r);
    CHECKF(r == 0x3EAAAAABu, "SSE 1/3 with x87 RC=down: 0x%llx want 0x3EAAAAAB", (unsigned long long)r);
    t_f80 three = X_THREE;
    t_set_mxcsr(MX_DEF | MX_RC(RD));
    x87_op(XO_DIV, &one, &three, 0x037F, &res);
    check_f80("x87 1/3 (RN, PC64) with MXCSR RC=down", &res, 0x3FFD, 0xAAAAAAAAAAAAAAABull);
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
}

static volatile uintptr_t g_fx_label, g_fx_label2;

CHILD(ldmxcsr_bad) {
    /* argv[1] = value to load. A reserved bit (not in MXCSR_MASK) must raise #GP. */
    unsigned v = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 0) : 0x10000u | MX_DEF;
    t_set_mxcsr(v);
    printf("SURVIVED=0x%08x\n", t_get_mxcsr());
    return 0;
}
TEST(mxcsr_reserved_bits_fault) {
    /* SDM LDMXCSR: #GP(0) if any reserved bit (bits not in MXCSR_MASK) is set. On Windows a user-mode
     * #GP that is not a privileged instruction is delivered as an exception (observed code recorded). */
    unsigned m = mxcsr_mask();
    unsigned cand[3] = { MX_DEF | 0x10000u, MX_DEF | 0x80000000u, MX_DEF | MX_DAZ };
    int tested = 0;
    for (int i = 0; i < 3; i++) {
        unsigned bit = cand[i] & ~MX_DEF;
        if (m & bit) continue;                 /* not reserved on this CPU (e.g. DAZ supported) */
        static t_child_result r;
        char args[32]; snprintf(args, sizeof args, "0x%08x", cand[i]);
        REQUIRE(t_child_run("ldmxcsr_bad", args, 10000, &r));
        CHECKF(!r.timed_out, "child timed out");
        CHECKF(strstr(r.out, "SURVIVED=") == NULL, "LDMXCSR 0x%08x did not fault (%s)", cand[i], r.out);
        CHECKF(r.exc_code == STATUS_ACCESS_VIOLATION || r.exc_code == STATUS_PRIVILEGED_INSTRUCTION,
               "LDMXCSR 0x%08x: exception 0x%08lx (want #GP as ACCESS_VIOLATION or PRIVILEGED_INSTRUCTION)", cand[i], r.exc_code);
        char key[32]; snprintf(key, sizeof key, "gp_code_bit_0x%x", bit);
        t_info(key, "0x%08lx", r.exc_code);
        tested++;
    }
    CHECK(tested >= 2);
}

/* ======================================================================================
 * 3. masked SSE results and flags, every exception class, all rounding modes
 * ====================================================================================== */
#define FL_IE MXF_IE
#define FL_DE MXF_DE
#define FL_ZE MXF_ZE
#define FL_OE MXF_OE
#define FL_UE MXF_UE
#define FL_PE MXF_PE

static const sse_vec v_invalid[] = {
    { "divss 0/0",              SO_DIVSS, 0, 0, 0, R1(0xFFC00000u), FL_IE },
    { "divss inf/inf",          SO_DIVSS, 0x7F800000u, 0x7F800000u, 0, R1(0xFFC00000u), FL_IE },
    { "mulss inf*0",            SO_MULSS, 0x7F800000u, 0, 0, R1(0xFFC00000u), FL_IE },
    { "subss inf-inf",          SO_SUBSS, 0x7F800000u, 0x7F800000u, 0, R1(0xFFC00000u), FL_IE },
    { "sqrtss -1",              SO_SQRTSS, 0xBF800000u, 0, 0, R1(0xFFC00000u), FL_IE },
    { "sqrtss -0 (valid)",      SO_SQRTSS, 0x80000000u, 0, 0, R1(0x80000000u), 0 },
    { "sqrtss +inf (valid)",    SO_SQRTSS, 0x7F800000u, 0, 0, R1(0x7F800000u), 0 },
    { "addss SNaN+1",           SO_ADDSS, 0x7F800001u, 0x3F800000u, 0, R1(0x7FC00001u), FL_IE },
    { "addss 1+(-SNaN)",        SO_ADDSS, 0x3F800000u, 0xFF800003u, 0, R1(0xFFC00003u), FL_IE },
    { "addss QNaN+1 (quiet)",   SO_ADDSS, 0x7FC00005u, 0x3F800000u, 0, R1(0x7FC00005u), 0 },
    { "addss SNaN,QNaN -> src1", SO_ADDSS, 0x7F800002u, 0x7FC00007u, 0, R1(0x7FC00002u), FL_IE },
    { "addss QNaN,SNaN -> src1", SO_ADDSS, 0x7FC00007u, 0x7F800002u, 0, R1(0x7FC00007u), FL_IE },
    { "maxss 1,QNaN -> src2",   SO_MAXSS, 0x3F800000u, 0x7FC00005u, 0, R1(0x7FC00005u), FL_IE },
    { "maxss QNaN,1 -> src2",   SO_MAXSS, 0x7FC00005u, 0x3F800000u, 0, R1(0x3F800000u), FL_IE },
    { "minss +0,-0 -> src2",    SO_MINSS, 0x00000000u, 0x80000000u, 0, R1(0x80000000u), 0 },
    { "maxss -0,+0 -> src2",    SO_MAXSS, 0x80000000u, 0x00000000u, 0, R1(0x00000000u), 0 },
    { "cvtss2si QNaN",          SO_CVTSS2SI, 0x7FC00000u, 0, 0, R1(0x80000000u), FL_IE },
    { "cvttss2si 3e9",          SO_CVTTSS2SI, 0x4F32D05Eu, 0, 0, R1(0x80000000u), FL_IE },
    { "cvtss2si 2^31",          SO_CVTSS2SI, 0x4F000000u, 0, 0, R1(0x80000000u), FL_IE },
    { "cvtss2si -2^31 (exact)", SO_CVTSS2SI, 0xCF000000u, 0, 0, R1(0x80000000u), 0 },
    { "comiss 1,QNaN (signals)", SO_COMISS, 0x3F800000u, 0x7FC00000u, 0, R1(7), FL_IE },
    { "ucomiss 1,QNaN (quiet)", SO_UCOMISS, 0x3F800000u, 0x7FC00000u, 0, R1(7), 0 },
    { "ucomiss 1,SNaN",         SO_UCOMISS, 0x3F800000u, 0x7F800001u, 0, R1(7), FL_IE },
    { "comiss 2,1",             SO_COMISS, 0x40000000u, 0x3F800000u, 0, R1(0), 0 },
    { "comiss 1,2",             SO_COMISS, 0x3F800000u, 0x40000000u, 0, R1(1), 0 },
    { "comiss +0,-0 (equal)",   SO_COMISS, 0x00000000u, 0x80000000u, 0, R1(4), 0 },
    { "addsd SNaN+1",           SO_ADDSD, 0x7FF0000000000001ull, 0x3FF0000000000000ull, 0, R1(0x7FF8000000000001ull), FL_IE },
    { "divsd 0/0",              SO_DIVSD, 0, 0, 0, R1(0xFFF8000000000000ull), FL_IE },
    { "mulsd inf*0",            SO_MULSD, 0x7FF0000000000000ull, 0, 0, R1(0xFFF8000000000000ull), FL_IE },
    { "sqrtsd -1",              SO_SQRTSD, 0xBFF0000000000000ull, 0, 0, R1(0xFFF8000000000000ull), FL_IE },
    { "cvtsd2si QNaN",          SO_CVTSD2SI, 0x7FF8000000000000ull, 0, 0, R1(0x80000000u), FL_IE },
    { "cvtsd2ss SNaN",          SO_CVTSD2SS, 0x7FF4000000000000ull, 0, 0, R1(0x7FE00000u), FL_IE },
};
TEST(sse_masked_invalid_operation) { run_sse_vecs(v_invalid, T_COUNTOF(v_invalid)); }

static const sse_vec v_zerodiv[] = {
    { "divss 1/0",       SO_DIVSS, 0x3F800000u, 0, 0, R1(0x7F800000u), FL_ZE },
    { "divss -1/0",      SO_DIVSS, 0xBF800000u, 0, 0, R1(0xFF800000u), FL_ZE },
    { "divss 1/-0",      SO_DIVSS, 0x3F800000u, 0x80000000u, 0, R1(0xFF800000u), FL_ZE },
    { "divss inf/0 (no ZE)", SO_DIVSS, 0x7F800000u, 0, 0, R1(0x7F800000u), 0 },
    { "divss 0/1",       SO_DIVSS, 0, 0x3F800000u, 0, R1(0x00000000u), 0 },
    { "divsd 1/0",       SO_DIVSD, 0x3FF0000000000000ull, 0, 0, R1(0x7FF0000000000000ull), FL_ZE },
    { "divsd -1/0",      SO_DIVSD, 0xBFF0000000000000ull, 0, 0, R1(0xFFF0000000000000ull), FL_ZE },
};
TEST(sse_masked_divide_by_zero) { run_sse_vecs(v_zerodiv, T_COUNTOF(v_zerodiv)); }

static const sse_vec v_denormal[] = {
    { "addss 1+den",      SO_ADDSS, 0x3F800000u, 0x00000001u, 0, { 0x3F800000u, 0x3F800000u, 0x3F800001u, 0x3F800000u }, FL_DE | FL_PE },
    { "addss -1+-den",    SO_ADDSS, 0xBF800000u, 0x80000001u, 0, { 0xBF800000u, 0xBF800001u, 0xBF800000u, 0xBF800000u }, FL_DE | FL_PE },
    { "addss den+0",      SO_ADDSS, 0x00000001u, 0, 0, R1(0x00000001u), FL_DE },
    { "mulss den*1",      SO_MULSS, 0x00000001u, 0x3F800000u, 0, R1(0x00000001u), FL_DE },
    { "divss den/den",    SO_DIVSS, 0x00000003u, 0x00000003u, 0, R1(0x3F800000u), FL_DE },
    { "comiss den,0",     SO_COMISS, 0x00000001u, 0, 0, R1(0), FL_DE },
    { "addsd 1+den",      SO_ADDSD, 0x3FF0000000000000ull, 1, 0, { 0x3FF0000000000000ull, 0x3FF0000000000000ull, 0x3FF0000000000001ull, 0x3FF0000000000000ull }, FL_DE | FL_PE },
    { "mulsd den*1",      SO_MULSD, 1, 0x3FF0000000000000ull, 0, R1(1), FL_DE },
};
TEST(sse_masked_denormal_operand) { run_sse_vecs(v_denormal, T_COUNTOF(v_denormal)); }

static const sse_vec v_overflow[] = {
    { "mulss max*2",   SO_MULSS, 0x7F7FFFFFu, 0x40000000u, 0, { 0x7F800000u, 0x7F7FFFFFu, 0x7F800000u, 0x7F7FFFFFu }, FL_OE | FL_PE },
    { "mulss -max*2",  SO_MULSS, 0xFF7FFFFFu, 0x40000000u, 0, { 0xFF800000u, 0xFF800000u, 0xFF7FFFFFu, 0xFF7FFFFFu }, FL_OE | FL_PE },
    { "addss max+max", SO_ADDSS, 0x7F7FFFFFu, 0x7F7FFFFFu, 0, { 0x7F800000u, 0x7F7FFFFFu, 0x7F800000u, 0x7F7FFFFFu }, FL_OE | FL_PE },
    { "mulss inf*2 (no OE)", SO_MULSS, 0x7F800000u, 0x40000000u, 0, R1(0x7F800000u), 0 },
    { "mulsd max*2",   SO_MULSD, 0x7FEFFFFFFFFFFFFFull, 0x4000000000000000ull, 0, { 0x7FF0000000000000ull, 0x7FEFFFFFFFFFFFFFull, 0x7FF0000000000000ull, 0x7FEFFFFFFFFFFFFFull }, FL_OE | FL_PE },
    { "cvtsd2ss 2^128", SO_CVTSD2SS, 0x47F0000000000000ull, 0, 0, { 0x7F800000u, 0x7F7FFFFFu, 0x7F800000u, 0x7F7FFFFFu }, FL_OE | FL_PE },
};
TEST(sse_masked_overflow) { run_sse_vecs(v_overflow, T_COUNTOF(v_overflow)); }

static const sse_vec v_underflow[] = {
    { "mulss min*0.5 (exact tiny: no UE)", SO_MULSS, 0x00800000u, 0x3F000000u, 0, R1(0x00400000u), 0 },
    { "mulss (min+ulp)*0.5",  SO_MULSS, 0x00800001u, 0x3F000000u, 0, { 0x00400000u, 0x00400000u, 0x00400001u, 0x00400000u }, FL_UE | FL_PE },
    { "mulss min*min",        SO_MULSS, 0x00800000u, 0x00800000u, 0, { 0, 0, 1, 0 }, FL_UE | FL_PE },
    { "mulss -min*min",       SO_MULSS, 0x80800000u, 0x00800000u, 0, { 0x80000000u, 0x80000001u, 0x80000000u, 0x80000000u }, FL_UE | FL_PE },
    { "mulsd min*0.5 (exact tiny)", SO_MULSD, 0x0010000000000000ull, 0x3FE0000000000000ull, 0, R1(0x0008000000000000ull), 0 },
    { "mulsd (min+ulp)*0.5",  SO_MULSD, 0x0010000000000001ull, 0x3FE0000000000000ull, 0, { 0x0008000000000000ull, 0x0008000000000000ull, 0x0008000000000001ull, 0x0008000000000000ull }, FL_UE | FL_PE },
    { "mulsd min*min",        SO_MULSD, 0x0010000000000000ull, 0x0010000000000000ull, 0, { 0, 0, 1, 0 }, FL_UE | FL_PE },
    { "cvtsd2ss 2^-150",      SO_CVTSD2SS, 0x3690000000000000ull, 0, 0, { 0, 0, 1, 0 }, FL_UE | FL_PE },
};
TEST(sse_masked_underflow) { run_sse_vecs(v_underflow, T_COUNTOF(v_underflow)); }

static const sse_vec v_round_s[] = {
    { "divss 1/3",     SO_DIVSS, 0x3F800000u, 0x40400000u, 0, { 0x3EAAAAABu, 0x3EAAAAAAu, 0x3EAAAAABu, 0x3EAAAAAAu }, FL_PE },
    { "divss -1/3",    SO_DIVSS, 0xBF800000u, 0x40400000u, 0, { 0xBEAAAAABu, 0xBEAAAAABu, 0xBEAAAAAAu, 0xBEAAAAAAu }, FL_PE },
    { "divss 2/3",     SO_DIVSS, 0x40000000u, 0x40400000u, 0, { 0x3F2AAAABu, 0x3F2AAAAAu, 0x3F2AAAABu, 0x3F2AAAAAu }, FL_PE },
    { "divss 10/7",    SO_DIVSS, 0x41200000u, 0x40E00000u, 0, { 0x3FB6DB6Eu, 0x3FB6DB6Du, 0x3FB6DB6Eu, 0x3FB6DB6Du }, FL_PE },
    { "divss -10/7",   SO_DIVSS, 0xC1200000u, 0x40E00000u, 0, { 0xBFB6DB6Eu, 0xBFB6DB6Eu, 0xBFB6DB6Du, 0xBFB6DB6Du }, FL_PE },
    { "subss 1-2^-25", SO_SUBSS, 0x3F800000u, 0x33000000u, 0, { 0x3F800000u, 0x3F7FFFFFu, 0x3F800000u, 0x3F7FFFFFu }, FL_PE },
    { "addss 1+2^-24 (tie->even)", SO_ADDSS, 0x3F800000u, 0x33800000u, 0, { 0x3F800000u, 0x3F800000u, 0x3F800001u, 0x3F800000u }, FL_PE },
    { "addss 1+3*2^-25 (tie->even up)", SO_ADDSS, 0x3F800000u, 0x33C00000u, 0, { 0x3F800001u, 0x3F800000u, 0x3F800001u, 0x3F800000u }, FL_PE },
    { "addss 1e8+1",   SO_ADDSS, 0x4CBEBC20u, 0x3F800000u, 0, { 0x4CBEBC20u, 0x4CBEBC20u, 0x4CBEBC21u, 0x4CBEBC20u }, FL_PE },
    { "sqrtss 2",      SO_SQRTSS, 0x40000000u, 0, 0, { 0x3FB504F3u, 0x3FB504F3u, 0x3FB504F4u, 0x3FB504F3u }, FL_PE },
    { "addss 1+1 (exact)", SO_ADDSS, 0x3F800000u, 0x3F800000u, 0, R1(0x40000000u), 0 },
    { "subss 1-1 (sign of exact zero)", SO_SUBSS, 0x3F800000u, 0x3F800000u, 0, { 0x00000000u, 0x80000000u, 0x00000000u, 0x00000000u }, 0 },
};
TEST(sse_rounding_modes_single) { run_sse_vecs(v_round_s, T_COUNTOF(v_round_s)); }

static const sse_vec v_round_d[] = {
    { "divsd 1/3",   SO_DIVSD, 0x3FF0000000000000ull, 0x4008000000000000ull, 0, { 0x3FD5555555555555ull, 0x3FD5555555555555ull, 0x3FD5555555555556ull, 0x3FD5555555555555ull }, FL_PE },
    { "divsd -1/3",  SO_DIVSD, 0xBFF0000000000000ull, 0x4008000000000000ull, 0, { 0xBFD5555555555555ull, 0xBFD5555555555556ull, 0xBFD5555555555555ull, 0xBFD5555555555555ull }, FL_PE },
    { "divsd 2/3",   SO_DIVSD, 0x4000000000000000ull, 0x4008000000000000ull, 0, { 0x3FE5555555555555ull, 0x3FE5555555555555ull, 0x3FE5555555555556ull, 0x3FE5555555555555ull }, FL_PE },
    { "addsd 0.1+0.2", SO_ADDSD, 0x3FB999999999999Aull, 0x3FC999999999999Aull, 0, { 0x3FD3333333333334ull, 0x3FD3333333333333ull, 0x3FD3333333333334ull, 0x3FD3333333333333ull }, FL_PE },
    { "subsd 1-2^-54... exact 1-eps/2", SO_SUBSD, 0x3FF0000000000000ull, 0x3CA0000000000000ull, 0, R1(0x3FEFFFFFFFFFFFFFull), 0 },
    { "subsd 1-1",   SO_SUBSD, 0x3FF0000000000000ull, 0x3FF0000000000000ull, 0, { 0, 0x8000000000000000ull, 0, 0 }, 0 },
};
TEST(sse_rounding_modes_double) { run_sse_vecs(v_round_d, T_COUNTOF(v_round_d)); }

static const sse_vec v_cvt[] = {
    { "cvtss2si 2.5",  SO_CVTSS2SI, 0x40200000u, 0, 0, { 2, 2, 3, 2 }, FL_PE },
    { "cvtss2si -2.5", SO_CVTSS2SI, 0xC0200000u, 0, 0, { 0xFFFFFFFEu, 0xFFFFFFFDu, 0xFFFFFFFEu, 0xFFFFFFFEu }, FL_PE },
    { "cvtss2si 3.5",  SO_CVTSS2SI, 0x40600000u, 0, 0, { 4, 3, 4, 3 }, FL_PE },
    { "cvtss2si 0.5",  SO_CVTSS2SI, 0x3F000000u, 0, 0, { 0, 0, 1, 0 }, FL_PE },
    { "cvtss2si -0.5", SO_CVTSS2SI, 0xBF000000u, 0, 0, { 0, 0xFFFFFFFFu, 0, 0 }, FL_PE },
    { "cvttss2si 2.7 (always trunc)",  SO_CVTTSS2SI, 0x402CCCCDu, 0, 0, R1(2), FL_PE },
    { "cvttss2si -2.7 (always trunc)", SO_CVTTSS2SI, 0xC02CCCCDu, 0, 0, R1(0xFFFFFFFEu), FL_PE },
    { "cvtss2si 7 (exact)", SO_CVTSS2SI, 0x40E00000u, 0, 0, R1(7), 0 },
    { "cvtsd2si 2.5",  SO_CVTSD2SI, 0x4004000000000000ull, 0, 0, { 2, 2, 3, 2 }, FL_PE },
    { "cvtsd2si -1.5", SO_CVTSD2SI, 0xBFF8000000000000ull, 0, 0, { 0xFFFFFFFEu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu }, FL_PE },
    { "cvtsi2ss 16777217",  SO_CVTSI2SS, 0x01000001u, 0, 0, { 0x4B800000u, 0x4B800000u, 0x4B800001u, 0x4B800000u }, FL_PE },
    { "cvtsi2ss -16777217", SO_CVTSI2SS, 0xFEFFFFFFu, 0, 0, { 0xCB800000u, 0xCB800001u, 0xCB800000u, 0xCB800000u }, FL_PE },
    { "cvtsi2ss 16777219",  SO_CVTSI2SS, 0x01000003u, 0, 0, { 0x4B800002u, 0x4B800001u, 0x4B800002u, 0x4B800001u }, FL_PE },
    { "cvtsi2ss 16777216 (exact)", SO_CVTSI2SS, 0x01000000u, 0, 0, R1(0x4B800000u), 0 },
    { "cvtsd2ss 1+2^-24",    SO_CVTSD2SS, 0x3FF0000001000000ull, 0, 0, { 0x3F800000u, 0x3F800000u, 0x3F800001u, 0x3F800000u }, FL_PE },
    { "cvtsd2ss -(1+2^-24)", SO_CVTSD2SS, 0xBFF0000001000000ull, 0, 0, { 0xBF800000u, 0xBF800001u, 0xBF800000u, 0xBF800000u }, FL_PE },
};
TEST(sse_conversion_rounding) { run_sse_vecs(v_cvt, T_COUNTOF(v_cvt)); }

static const sse_vec v_ftz[] = {
    { "FTZ mulss min*0.5 (exact tiny -> 0, UE|PE)", SO_MULSS, 0x00800000u, 0x3F000000u, MX_FTZ, R1(0), FL_UE | FL_PE },
    { "FTZ mulss (min+ulp)*0.5", SO_MULSS, 0x00800001u, 0x3F000000u, MX_FTZ, R1(0), FL_UE | FL_PE },
    { "FTZ mulss min*min (RU still 0)", SO_MULSS, 0x00800000u, 0x00800000u, MX_FTZ, R1(0), FL_UE | FL_PE },
    { "FTZ mulss -min*min (signed zero)", SO_MULSS, 0x80800000u, 0x00800000u, MX_FTZ, R1(0x80000000u), FL_UE | FL_PE },
    { "FTZ subss 1.5min-min",   SO_SUBSS, 0x00C00000u, 0x00800000u, MX_FTZ, R1(0), FL_UE | FL_PE },
    { "FTZ mulss den*1 (input kept, output flushed)", SO_MULSS, 0x00000001u, 0x3F800000u, MX_FTZ, R1(0), FL_DE | FL_UE | FL_PE },
    { "FTZ addss 1+den (inputs not flushed)", SO_ADDSS, 0x3F800000u, 0x00000001u, MX_FTZ, { 0x3F800000u, 0x3F800000u, 0x3F800001u, 0x3F800000u }, FL_DE | FL_PE },
    { "FTZ divss 1/3 (normal unaffected)", SO_DIVSS, 0x3F800000u, 0x40400000u, MX_FTZ, { 0x3EAAAAABu, 0x3EAAAAAAu, 0x3EAAAAABu, 0x3EAAAAAAu }, FL_PE },
    { "FTZ subss min-min (exact zero)", SO_SUBSS, 0x00800000u, 0x00800000u, MX_FTZ, { 0, 0x80000000u, 0, 0 }, 0 },
    { "FTZ mulsd min*0.5", SO_MULSD, 0x0010000000000000ull, 0x3FE0000000000000ull, MX_FTZ, R1(0), FL_UE | FL_PE },
    { "FTZ cvtsd2ss 2^-150", SO_CVTSD2SS, 0x3690000000000000ull, 0, MX_FTZ, R1(0), FL_UE | FL_PE },
    { "FTZ mulss max*2 (overflow unaffected)", SO_MULSS, 0x7F7FFFFFu, 0x40000000u, MX_FTZ, { 0x7F800000u, 0x7F7FFFFFu, 0x7F800000u, 0x7F7FFFFFu }, FL_OE | FL_PE },
};
TEST(sse_ftz_flush_to_zero) { run_sse_vecs(v_ftz, T_COUNTOF(v_ftz)); }

static const sse_vec v_daz[] = {
    { "DAZ addss 1+den -> 1 exact",   SO_ADDSS, 0x3F800000u, 0x00000001u, MX_DAZ, R1(0x3F800000u), 0 },
    { "DAZ addss den+0 -> +0",        SO_ADDSS, 0x00000001u, 0, MX_DAZ, R1(0), 0 },
    { "DAZ addss -den+0 (RD -> -0)",  SO_ADDSS, 0x80000001u, 0, MX_DAZ, { 0, 0x80000000u, 0, 0 }, 0 },
    { "DAZ mulss den*1 -> 0",         SO_MULSS, 0x00000001u, 0x3F800000u, MX_DAZ, R1(0), 0 },
    { "DAZ mulss -den*1 -> -0",       SO_MULSS, 0x80000001u, 0x3F800000u, MX_DAZ, R1(0x80000000u), 0 },
    { "DAZ divss den/den -> 0/0",     SO_DIVSS, 0x00000003u, 0x00000003u, MX_DAZ, R1(0xFFC00000u), FL_IE },
    { "DAZ divss 1/den -> 1/0",       SO_DIVSS, 0x3F800000u, 0x00000001u, MX_DAZ, R1(0x7F800000u), FL_ZE },
    { "DAZ comiss den,0 -> equal",    SO_COMISS, 0x00000001u, 0, MX_DAZ, R1(4), 0 },
    { "DAZ mulsd den*1 -> 0",         SO_MULSD, 1, 0x3FF0000000000000ull, MX_DAZ, R1(0), 0 },
    { "DAZ mulss min*0.5 (outputs still denormal)", SO_MULSS, 0x00800000u, 0x3F000000u, MX_DAZ, R1(0x00400000u), 0 },
    { "FTZ+DAZ mulss den*1 -> 0 exact", SO_MULSS, 0x00000001u, 0x3F800000u, MX_DAZ | MX_FTZ, R1(0), 0 },
    { "FTZ+DAZ mulss min*0.5",        SO_MULSS, 0x00800000u, 0x3F000000u, MX_DAZ | MX_FTZ, R1(0), FL_UE | FL_PE },
};
TEST(sse_daz_denormals_are_zero) {
    if (!daz_supported()) SKIP("DAZ not supported (MXCSR_MASK bit 6 clear)");
    run_sse_vecs(v_daz, T_COUNTOF(v_daz));
}

TEST(sse_rounding_switch_takes_effect_immediately) {
    /* LDMXCSR affects the very next instruction: RU, RD, RU, RZ alternating in one block */
    uint32_t one = 0x3F800000u, three = 0x40400000u, r[4];
    unsigned m[4] = { MX_DEF | MX_RC(RU), MX_DEF | MX_RC(RD), MX_DEF | MX_RC(RU), MX_DEF | MX_RC(RZ) }, sv;
    __asm__ volatile(
        "stmxcsr %[sv]\n\tmovd %[one], %%xmm2\n\tmovd %[three], %%xmm3\n\t"
        "ldmxcsr %[m0]\n\tmovaps %%xmm2, %%xmm0\n\tdivss %%xmm3, %%xmm0\n\tmovd %%xmm0, %[r0]\n\t"
        "ldmxcsr %[m1]\n\tmovaps %%xmm2, %%xmm0\n\tdivss %%xmm3, %%xmm0\n\tmovd %%xmm0, %[r1]\n\t"
        "ldmxcsr %[m2]\n\tmovaps %%xmm2, %%xmm0\n\tdivss %%xmm3, %%xmm0\n\tmovd %%xmm0, %[r2]\n\t"
        "ldmxcsr %[m3]\n\tmovaps %%xmm2, %%xmm0\n\tdivss %%xmm3, %%xmm0\n\tmovd %%xmm0, %[r3]\n\t"
        "ldmxcsr %[sv]"
        : [sv] "=&m"(sv), [r0] "=m"(r[0]), [r1] "=m"(r[1]), [r2] "=m"(r[2]), [r3] "=m"(r[3])
        : [one] "m"(one), [three] "m"(three), [m0] "m"(m[0]), [m1] "m"(m[1]), [m2] "m"(m[2]), [m3] "m"(m[3])
        : "xmm0", "xmm2", "xmm3");
    CHECK_EQ(r[0], 0x3EAAAAABu); CHECK_EQ(r[1], 0x3EAAAAAAu); CHECK_EQ(r[2], 0x3EAAAAABu); CHECK_EQ(r[3], 0x3EAAAAAAu);
}

TEST(sse41_roundss_mxcsr_vs_immediate) {
    /* ROUNDSS imm[2]=1 uses MXCSR.RC, imm[1:0] otherwise; imm[3]=1 suppresses the precision exception */
    if (!cpu_has(F_SSE41)) SKIP("SSE4.1 not available");
    uint32_t x = 0x40200000u; /* 2.5 */
    for (int m = 0; m < 4; m++) {
        unsigned mx = MX_DEF | MX_RC(m), sv, mo1, mo2, mo3;
        uint32_t r1, r2, r3;
        __asm__ volatile(
            "stmxcsr %[sv]\n\tmovd %[x], %%xmm1\n\t"
            "ldmxcsr %[mx]\n\troundss $4, %%xmm1, %%xmm0\n\tstmxcsr %[mo1]\n\tmovd %%xmm0, %[r1]\n\t"
            "ldmxcsr %[mx]\n\troundss $0xC, %%xmm1, %%xmm0\n\tstmxcsr %[mo2]\n\tmovd %%xmm0, %[r2]\n\t"
            "ldmxcsr %[mx]\n\troundss $2, %%xmm1, %%xmm0\n\tstmxcsr %[mo3]\n\tmovd %%xmm0, %[r3]\n\t"
            "ldmxcsr %[sv]"
            : [sv] "=&m"(sv), [mo1] "=m"(mo1), [mo2] "=m"(mo2), [mo3] "=m"(mo3), [r1] "=m"(r1), [r2] "=m"(r2), [r3] "=m"(r3)
            : [x] "m"(x), [mx] "m"(mx) : "xmm0", "xmm1");
        static const uint32_t want[4] = { 0x40000000u, 0x40000000u, 0x40400000u, 0x40000000u };
        CHECKF(r1 == want[m], "roundss $4 (MXCSR %s) of 2.5: 0x%08x want 0x%08x", k_rc[m], r1, want[m]);
        CHECKF((mo1 & MX_FLAGS) == MXF_PE, "roundss $4: flags 0x%02x want PE", mo1 & MX_FLAGS);
        CHECKF(r2 == want[m], "roundss $0xC (MXCSR %s): 0x%08x", k_rc[m], r2);
        CHECKF((mo2 & MX_FLAGS) == 0, "roundss $0xC: PE not suppressed (0x%02x)", mo2 & MX_FLAGS);
        CHECKF(r3 == 0x40400000u, "roundss $2 (ceil) ignores MXCSR %s: 0x%08x", k_rc[m], r3);
    }
}

/* ======================================================================================
 * 4. unmasked exceptions (child processes): EXCEPTION_FLT_* codes, CONTEXT contents, resume
 * ====================================================================================== */
enum { K_IE, K_DE, K_ZE, K_OE, K_UE, K_PE, K_SF, K_PACKED };
static const char *const k_kind[] = { "IE", "DE", "ZE", "OE", "UE", "PE", "SF", "PACKED" };

static volatile LONG g_fx_hits;
static struct {
    DWORD code, nparams, ctxflags;
    unsigned long long addr, ip, info0, info1, fip;
    unsigned ctx_mx, ctx_fxmx, h_mx;
    unsigned ctx_cw, ctx_sw;
    unsigned short h_cw;
    uint64_t xmm0_lo, xmm0_hi;
} g_fx;

static int is_flt_code(DWORD c) {
    return (c >= 0xC000008Du && c <= 0xC0000093u) || c == STATUS_FLOAT_MULTIPLE_FAULTS || c == STATUS_FLOAT_MULTIPLE_TRAPS;
}

static LONG CALLBACK fx_veh(EXCEPTION_POINTERS *ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (!is_flt_code(code)) return EXCEPTION_CONTINUE_SEARCH;
    LONG n = InterlockedIncrement(&g_fx_hits);
    CONTEXT *c = ep->ContextRecord;
    if (n > 3) { printf("LOOP=%ld\nCODE=0x%08lx\n", n, code); fflush(stdout); ExitProcess(0xBAD1); }
    if (n == 1) {
        g_fx.code = code;
        g_fx.addr = (uintptr_t)ep->ExceptionRecord->ExceptionAddress;
        g_fx.nparams = ep->ExceptionRecord->NumberParameters;
        g_fx.info0 = g_fx.nparams > 0 ? ep->ExceptionRecord->ExceptionInformation[0] : 0;
        g_fx.info1 = g_fx.nparams > 1 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
        g_fx.ctxflags = c->ContextFlags;
        g_fx.h_mx = t_get_mxcsr();
        g_fx.h_cw = t_get_x87_cw();
#ifdef T_X64
        g_fx.ip = c->Rip;
        g_fx.ctx_mx = c->MxCsr;
        g_fx.ctx_fxmx = c->FltSave.MxCsr;
        g_fx.ctx_cw = c->FltSave.ControlWord;
        g_fx.ctx_sw = c->FltSave.StatusWord;
        g_fx.fip = c->FltSave.ErrorOffset;
        g_fx.xmm0_lo = c->Xmm0.Low; g_fx.xmm0_hi = (uint64_t)c->Xmm0.High;
#else
        g_fx.ip = c->Eip;
        if (c->ContextFlags & CONTEXT_EXTENDED_REGISTERS & ~CONTEXT_i386) {
            memcpy(&g_fx.ctx_mx, c->ExtendedRegisters + 24, 4);
            g_fx.ctx_fxmx = g_fx.ctx_mx;
            memcpy(&g_fx.xmm0_lo, c->ExtendedRegisters + 160, 8);
            memcpy(&g_fx.xmm0_hi, c->ExtendedRegisters + 168, 8);
        } else g_fx.ctx_mx = g_fx.ctx_fxmx = 0xFFFFFFFFu;
        if (c->ContextFlags & CONTEXT_FLOATING_POINT & ~CONTEXT_i386) {
            g_fx.ctx_cw = c->FloatSave.ControlWord & 0xFFFF;
            g_fx.ctx_sw = c->FloatSave.StatusWord & 0xFFFF;
            g_fx.fip = c->FloatSave.ErrorOffset;
        } else g_fx.ctx_cw = g_fx.ctx_sw = 0xFFFFFFFFu;
#endif
    }
    /* recover: mask everything, clear the sticky flags (SSE) and the pending x87 exception */
#ifdef T_X64
    c->MxCsr = (c->MxCsr & ~MX_FLAGS) | MX_MASKS;
    c->FltSave.MxCsr = c->MxCsr;
    c->FltSave.ControlWord |= 0x3F;
    c->FltSave.StatusWord &= (WORD)~0x80FFu;
#else
    if (c->ContextFlags & CONTEXT_EXTENDED_REGISTERS & ~CONTEXT_i386) {
        uint8_t *x = c->ExtendedRegisters;
        uint32_t mx; memcpy(&mx, x + 24, 4); mx = (mx & ~MX_FLAGS) | MX_MASKS; memcpy(x + 24, &mx, 4);
        uint16_t w; memcpy(&w, x + 0, 2); w |= 0x3F; memcpy(x + 0, &w, 2);
        memcpy(&w, x + 2, 2); w &= (uint16_t)~0x80FFu; memcpy(x + 2, &w, 2);
    }
    else t_set_mxcsr((t_get_mxcsr() & ~MX_FLAGS) | MX_MASKS);   /* not carried by the CONTEXT: fix live state */
    if (c->ContextFlags & CONTEXT_FLOATING_POINT & ~CONTEXT_i386) {
        c->FloatSave.ControlWord |= 0x3F;
        c->FloatSave.StatusWord &= ~0x80FFu;
    } else { t_fnclex(); t_set_x87_cw((unsigned short)(t_get_x87_cw() | 0x3F)); }
#endif
    return EXCEPTION_CONTINUE_EXECUTION;
}

static void fx_print(void) {
    printf("HITS=%ld\nCODE=0x%08lx\nADDR=0x%llx\nIP=0x%llx\nLABEL=0x%llx\nOPLABEL=0x%llx\nNPARAM=%lu\nINFO0=0x%llx\nINFO1=0x%llx\n"
           "CTXFLAGS=0x%08lx\nCTXMX=0x%08x\nCTXFXMX=0x%08x\nCTXCW=0x%08x\nCTXSW=0x%08x\nFIP=0x%llx\nHMX=0x%08x\nHCW=0x%04x\nXMM0LO=0x%016llx\nXMM0HI=0x%016llx\n",
           g_fx_hits, g_fx.code, g_fx.addr, g_fx.ip, (unsigned long long)g_fx_label, (unsigned long long)g_fx_label2,
           g_fx.nparams, g_fx.info0, g_fx.info1, g_fx.ctxflags, g_fx.ctx_mx, g_fx.ctx_fxmx, g_fx.ctx_cw, g_fx.ctx_sw,
           g_fx.fip, g_fx.h_mx, g_fx.h_cw, (unsigned long long)g_fx.xmm0_lo, (unsigned long long)g_fx.xmm0_hi);
}

/* one SSE instruction with exactly one exception unmasked; label 1: is the faulting instruction */
#define FX_SSE(insn) __asm__ volatile(LAB_TO("1f", "lab") \
    "stmxcsr %[sv]\n\tmovd %[a], %%xmm0\n\tmovd %[b], %%xmm1\n\tldmxcsr %[mx]\n" \
    "1:\n\t" insn " %%xmm1, %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]\n\tmovd %%xmm0, %[r]" \
    : [lab] "=m"(g_fx_label), [sv] "=&m"(sv), [mo] "=m"(*mo), [r] "=m"(*r) : [mx] "m"(mx), [a] "m"(a), [b] "m"(b) : "xmm0", "xmm1", LAB_CLOBBER, "memory")
static T_NOINLINE void fx_sse_run(int k, unsigned mx, uint32_t a, uint32_t b, uint32_t *r, unsigned *mo) {
    unsigned sv;
    switch (k) {
    case K_DE: FX_SSE("addss"); break;
    case K_OE: case K_UE: FX_SSE("mulss"); break;
    default: FX_SSE("divss"); break;
    }
}
static T_NOINLINE void fx_sse_packed(unsigned mx, const uint32_t *a, const uint32_t *b, uint32_t *r, unsigned *mo) {
    unsigned sv;
    __asm__ volatile(LAB_TO("1f", "lab")
        "stmxcsr %[sv]\n\tmovups %[a], %%xmm0\n\tmovups %[b], %%xmm1\n\tldmxcsr %[mx]\n"
        "1:\n\tdivps %%xmm1, %%xmm0\n\tstmxcsr %[mo]\n\tldmxcsr %[sv]\n\tmovups %%xmm0, %[r]"
        : [lab] "=m"(g_fx_label), [sv] "=&m"(sv), [mo] "=m"(*mo), [r] "=m"(*(uint32_t(*)[4])r)
        : [mx] "m"(mx), [a] "m"(*(const uint32_t(*)[4])a), [b] "m"(*(const uint32_t(*)[4])b) : "xmm0", "xmm1", LAB_CLOBBER, "memory");
}

/* x87: op at label 3:, the waiting FWAIT that reports the pending exception at label 2: */
#define FX_X87(opseq) __asm__ volatile(LAB_TO("2f", "lab") LAB_TO("3f", "lab2") \
    "fnstcw %[sv]\n\tfninit\n\tfldt %[b]\n\tfldt %[a]\n\tfldcw %[cw]\n" \
    "3:\n\t" opseq "\n" "2:\n\tfwait\n\tfnstsw %[swo]\n\tfstpt %[r]\n\tfninit\n\tfldcw %[sv]" \
    : [lab] "=m"(g_fx_label), [lab2] "=m"(g_fx_label2), [sv] "=&m"(sv), [swo] "=m"(*swo), [r] "=m"(*r) \
    : [a] "m"(*a), [b] "m"(*b), [cw] "m"(cw) : LAB_CLOBBER, "memory")
static T_NOINLINE void fx_x87_run(int k, unsigned short cw, const t_f80 *a, const t_f80 *b, t_f80 *r, unsigned short *swo) {
    unsigned short sv;
    switch (k) {
    case K_IE: FX_X87("fsqrt"); break;
    case K_DE: FX_X87("fadd %%st(1), %%st"); break;
    case K_OE: case K_UE: FX_X87("fmul %%st(0), %%st"); break;
    case K_SF:
        __asm__ volatile(LAB_TO("2f", "lab") LAB_TO("3f", "lab2")
            "fnstcw %[sv]\n\tfninit\n\tfldcw %[cw]\n"
            "3:\n\tfadd %%st(1), %%st\n" "2:\n\tfwait\n\tfnstsw %[swo]\n\tfninit\n\tfldcw %[sv]"
            : [lab] "=m"(g_fx_label), [lab2] "=m"(g_fx_label2), [sv] "=&m"(sv), [swo] "=m"(*swo) : [cw] "m"(cw) : LAB_CLOBBER, "memory");
        *r = XV(X_ZERO);
        break;
    default: FX_X87("fdiv %%st(1), %%st"); break;   /* ZE, PE */
    }
}

/* <exe> --child=fpx <sse|x87> <kind> [unhandled] */
CHILD(fpx) {
    if (argc < 3) return 2;
    int x87 = !strcmp(argv[1], "x87");
    int k = atoi(argv[2]);
    int unhandled = argc > 3 && !strcmp(argv[3], "unhandled");
    if (!unhandled && !AddVectoredExceptionHandler(1, fx_veh)) { printf("NOVEH=1\n"); return 3; }
    if (!x87) {
        static const uint32_t A[6] = { 0, 0x00000001u, 0x3F800000u, 0x7F7FFFFFu, 0x00800000u, 0x3F800000u };
        static const uint32_t B[6] = { 0, 0x3F800000u, 0, 0x40000000u, 0x3F000000u, 0x40400000u };
        unsigned mo = 0;
        if (k == K_PACKED) {
            uint32_t a[4] = { 0x3F800000u, 0x40000000u, 0x40400000u, 0x40800000u }, b[4] = { 0x3F800000u, 0, 0x3F800000u, 0x3F800000u }, r[4];
            fx_sse_packed(MX_DEF & ~(1u << (7 + K_ZE)), a, b, r, &mo);
            printf("RES0=0x%08x\nRES1=0x%08x\nRES2=0x%08x\nRES3=0x%08x\n", r[0], r[1], r[2], r[3]);
        } else {
            uint32_t r = 0;
            fx_sse_run(k, MX_DEF & ~(1u << (7 + k)), A[k], B[k], &r, &mo);
            printf("RES=0x%08x\n", r);
        }
        printf("MO=0x%08x\n", mo);
    } else {
        t_f80 a, b, r; unsigned short swo = 0;
        unsigned short cw = (unsigned short)(0x037F & ~(1u << (k == K_SF ? 0 : k)));
        switch (k) {
        case K_IE: a = XV(X_MONE); b = XV(X_ZERO); break;
        case K_DE: a = XV(X_ONE); b = XV(X_DEN); break;
        case K_ZE: a = XV(X_ONE); b = XV(X_ZERO); break;
        case K_OE: a = f80(0x7FFE, 0xC000000000000000ull); b = f80(0, 0); break;
        case K_UE: a = XV(X_MINN); b = XV(X_ZERO); break;
        default:   a = XV(X_ONE); b = XV(X_THREE); break;
        }
        fx_x87_run(k, cw, &a, &b, &r, &swo);
        printf("RSE=0x%04x\nRM=0x%016llx\nSWO=0x%04x\n", f80_se(&r), (unsigned long long)f80_m(&r), swo);
    }
    fx_print();
    return 0;
}

static DWORD want_code(int x87, int k) {
    (void)x87;
    switch (k) {
    case K_IE: return 0xC0000090u; case K_DE: return 0xC000008Du; case K_ZE: case K_PACKED: return 0xC000008Eu;
    case K_OE: return 0xC0000091u; case K_UE: return 0xC0000093u; case K_PE: return 0xC000008Fu;
    default: return 0xC0000092u;
    }
}
static int code_acceptable(int x87, int k, DWORD c) {
    if (c == want_code(x87, k)) return 1;
    if (k == K_DE && c == 0xC0000090u) return 1;                      /* [K1]/[K2]: Windows maps DE to INVALID_OPERATION */
#ifdef T_X86
    if (!x87 && (c == STATUS_FLOAT_MULTIPLE_FAULTS || c == STATUS_FLOAT_MULTIPLE_TRAPS)) return 1;  /* [K1] 32-bit SSE */
#endif
    return 0;
}

static unsigned long long cv(const t_child_result *r, const char *key) {
    unsigned long long v = 0;
    CHECKF(child_hex(r, key, &v), "child did not report %s", key);
    return v;
}

static void run_fx_sse(int k) {
    static t_child_result r;
    char args[48]; snprintf(args, sizeof args, "sse %d", k);
    REQUIRE(t_child_run("fpx", args, 15000, &r));
    REQUIREF(!r.timed_out, "child timed out");
    REQUIREF(strstr(r.out, "LOOP=") == NULL, "resume with masked MXCSR from the handler did not take effect (exception re-raised): %s", r.out);
    REQUIREF(r.exit_code == 0, "child exit 0x%08lx exc 0x%08lx: %s", r.exit_code, r.exc_code, r.out);
    int bit = (k == K_PACKED) ? K_ZE : k;
    unsigned long long hits = cv(&r, "HITS"), code = cv(&r, "CODE"), addr = cv(&r, "ADDR"), lab = cv(&r, "LABEL");
    unsigned long long ip = cv(&r, "IP"), cmx = cv(&r, "CTXMX"), mo = cv(&r, "MO");
    CHECKF(hits == 1, "handler ran %llu times, want 1", hits);
    int code_ok = code_acceptable(0, k, (DWORD)code);
    int wine_code_gap = !code_ok && t_is_wine();
    if (!wine_code_gap) CHECKF(code_ok, "%s: exception code 0x%08llx, want 0x%08lx", k_kind[k], code, want_code(0, k));
    CHECKF(addr == lab, "ExceptionAddress 0x%llx != faulting instruction 0x%llx (SIMD exceptions are faults)", addr, lab);
    CHECKF(ip == lab, "CONTEXT instruction pointer 0x%llx != faulting instruction 0x%llx", ip, lab);
    if (cmx != 0xFFFFFFFFull) {
        CHECKF(cmx & (1u << bit), "CONTEXT MXCSR 0x%08llx lacks the %s flag", cmx, k_kind[bit]);
        CHECKF(!(cmx & (1u << (7 + bit))), "CONTEXT MXCSR 0x%08llx has the %s mask set", cmx, k_kind[bit]);
        CHECKF((cmx & 0xFFFF0000ull) == 0, "CONTEXT MXCSR reserved bits: 0x%08llx", cmx);
    } else {
        t_note("CONTEXT has no CONTEXT_EXTENDED_REGISTERS: MXCSR not inspectable");
    }
    /* SDM 11.5.4: an unmasked SIMD exception leaves the destination register unchanged */
    unsigned long long x0 = cv(&r, "XMM0LO");
    if (cmx != 0xFFFFFFFFull) {
        if (k == K_PACKED) CHECKF(x0 == 0x400000003F800000ull, "packed: XMM0 at fault 0x%016llx want unchanged source 0x400000003F800000", x0);
        else {
            static const uint32_t A[6] = { 0, 0x00000001u, 0x3F800000u, 0x7F7FFFFFu, 0x00800000u, 0x3F800000u };
            CHECKF((uint32_t)x0 == A[k], "XMM0[31:0] at fault 0x%08x want unchanged 0x%08x", (uint32_t)x0, A[k]);
        }
    }
    /* after resuming with all exceptions masked the instruction re-executes and gives the masked result */
    static const uint32_t RES[6] = { 0xFFC00000u, 0x3F800000u, 0x7F800000u, 0x7F800000u, 0x00400000u, 0x3EAAAAABu };
    static const unsigned FL[6] = { MXF_IE, MXF_DE | MXF_PE, MXF_ZE, MXF_OE | MXF_PE, 0, MXF_PE };
    if (k == K_PACKED) {
        CHECK_EQ(cv(&r, "RES0"), 0x3F800000u); CHECK_EQ(cv(&r, "RES1"), 0x7F800000u);
        CHECK_EQ(cv(&r, "RES2"), 0x40400000u); CHECK_EQ(cv(&r, "RES3"), 0x40800000u);
        CHECKF((mo & MX_FLAGS) == MXF_ZE, "flags after resume 0x%02llx want ZE", mo & MX_FLAGS);
    } else {
        unsigned long long res = cv(&r, "RES");
        CHECKF(res == RES[k], "masked result after resume 0x%08llx want 0x%08x", res, RES[k]);
        CHECKF((mo & MX_FLAGS) == FL[k], "flags after resume 0x%02llx want 0x%02x", mo & MX_FLAGS, FL[k]);
    }
    CHECKF((mo & MX_MASKS) == MX_MASKS, "MXCSR after resume 0x%08llx: masks from the CONTEXT not applied", mo);
    char key[32];
    snprintf(key, sizeof key, "code_%s", k_kind[k]); t_info(key, "0x%08llx", code);
    t_info("handler_mxcsr", "0x%08llx", cv(&r, "HMX"));
    t_info("exception_info", "n=%llu [0]=0x%llx [1]=0x%llx", cv(&r, "NPARAM"), cv(&r, "INFO0"), cv(&r, "INFO1"));
    t_info("ctx_flags", "0x%08llx", cv(&r, "CTXFLAGS"));
    if (wine_code_gap) XFAIL("Wine %s reports %s as 0x%08llx (Windows: 0x%08lx or INVALID_OPERATION)", t_wine_version(), k_kind[k], code, want_code(0, k));
}

TEST(sse_unmasked_invalid_operation) { run_fx_sse(K_IE); }
TEST(sse_unmasked_denormal_operand) { run_fx_sse(K_DE); }
TEST(sse_unmasked_divide_by_zero)   { run_fx_sse(K_ZE); }
TEST(sse_unmasked_overflow)         { run_fx_sse(K_OE); }
TEST(sse_unmasked_underflow_exact_tiny) {
    /* unmasked UE is signalled for a tiny result even when it is exact (SDM 4.9.1.5); masked it is not */
    run_fx_sse(K_UE);
}
TEST(sse_unmasked_inexact)          { run_fx_sse(K_PE); }
TEST(sse_unmasked_packed_dest_unchanged) { run_fx_sse(K_PACKED); }

TEST(sse_unmasked_unhandled_terminates_with_code) {
    static t_child_result r;
    REQUIRE(t_child_run("fpx", "sse 2 unhandled", 15000, &r));
    REQUIREF(!r.timed_out, "child timed out");
    CHECKF(r.exc_code != 0, "unhandled unmasked SSE divide-by-zero did not terminate the child (exit 0x%08lx): %s", r.exit_code, r.out);
#ifdef T_X64
    if (t_is_wine() && r.exc_code == 0xC00002B5u)   /* known Wine gap (seen with Wine 9; Wine 11 is exact) */
        t_expect_fail("Wine %s reports an x64 #XM as STATUS_FLOAT_MULTIPLE_TRAPS", t_wine_version());
#endif
    CHECKF(code_acceptable(0, K_ZE, r.exc_code), "unhandled code 0x%08lx want 0x%08lx", r.exc_code, want_code(0, K_ZE));
    CHECKF(strstr(r.out, "RES=") == NULL, "execution continued past the faulting instruction");
}

static void run_fx_x87(int k) {
    static t_child_result r;
    char args[48]; snprintf(args, sizeof args, "x87 %d", k);
    REQUIRE(t_child_run("fpx", args, 15000, &r));
    REQUIREF(!r.timed_out, "child timed out");
    REQUIREF(strstr(r.out, "LOOP=") == NULL, "resume with cleared FSW / masked FCW from the handler did not take effect: %s", r.out);
    REQUIREF(r.exit_code == 0, "child exit 0x%08lx exc 0x%08lx: %s", r.exit_code, r.exc_code, r.out);
    unsigned long long hits = cv(&r, "HITS"), code = cv(&r, "CODE"), addr = cv(&r, "ADDR"), lab = cv(&r, "LABEL");
    unsigned long long csw = cv(&r, "CTXSW"), ccw = cv(&r, "CTXCW"), swo = cv(&r, "SWO");
    int bit = k == K_SF ? 0 : k;
    CHECKF(hits == 1, "handler ran %llu times, want 1", hits);
    int code_ok = code_acceptable(1, k, (DWORD)code);
    int wine_code_gap = !code_ok && t_is_wine();
    if (!wine_code_gap) CHECKF(code_ok, "x87 %s: exception code 0x%08llx, want 0x%08lx", k_kind[k], code, want_code(1, k));
    /* x87 exceptions are deferred to the next waiting instruction (SDM 8.6): the FWAIT */
    unsigned long long oplab = cv(&r, "OPLABEL");
#ifdef T_X64
    CHECKF(addr == lab, "ExceptionAddress 0x%llx != the FWAIT that reports the pending exception 0x%llx", addr, lab);
#else
    /* x86: [K2] covers WOW64 (trap RIP = the FWAIT); 32-bit environments may report the x87 instruction
     * that caused the exception (FIP) instead. Either is a correct description of the fault. */
    if (t_sde_model() && addr != lab && addr != oplab)   /* emulator artifact, README "Emulated validation" */
        t_expect_fail("Intel SDE (%s): the x87 FPU instruction pointer points into Pin's code cache", t_sde_model());
    CHECKF(addr == lab || addr == oplab, "ExceptionAddress 0x%llx is neither the FWAIT 0x%llx nor the x87 op 0x%llx", addr, lab, oplab);
    t_info("x87_exception_address", "%s", addr == lab ? "fwait" : addr == oplab ? "faulting x87 instruction (FIP)" : "other");
#endif
    if (csw != 0xFFFFFFFFull) {
        CHECKF(csw & (1u << bit), "CONTEXT FSW 0x%04llx lacks the %s flag", csw, k_kind[bit]);
        CHECKF(csw & SW_ES, "CONTEXT FSW 0x%04llx lacks ES (error summary)", csw);
        if (k == K_SF) CHECKF((csw & (SW_SF | SW_C1)) == SW_SF, "CONTEXT FSW 0x%04llx: want SF=1 C1=0 (stack underflow)", csw);
        CHECKF(!(ccw & (1u << bit)), "CONTEXT FCW 0x%04llx has the %s mask set", ccw, k_kind[bit]);
    } else t_note("CONTEXT has no CONTEXT_FLOATING_POINT");
    CHECKF((swo & 0x80FF) == 0, "FSW after resume 0x%04llx: cleared status from the CONTEXT not applied", swo);
    /* result after resume: SDM 8.5: IE/DE/ZE unmasked leave the destination unchanged; OE/UE unmasked
     * deliver the result with the exponent biased by -/+24576; PE stores the rounded result */
    unsigned long long se = cv(&r, "RSE"), m = cv(&r, "RM");
    switch (k) {
    case K_IE: CHECKF(se == 0xBFFF && m == 0x8000000000000000ull, "IE: st(0) 0x%04llx:%016llx want unchanged -1.0", se, m); break;
    case K_DE: case K_ZE: CHECKF(se == 0x3FFF && m == 0x8000000000000000ull, "%s: st(0) 0x%04llx:%016llx want unchanged 1.0", k_kind[k], se, m); break;
    case K_OE: CHECKF(se == 0x5FFE && m == 0x9000000000000000ull, "OE: st(0) 0x%04llx:%016llx want bias-adjusted 0x5FFE:9000000000000000", se, m); break;
    case K_UE: CHECKF(se == 0x2003 && m == 0x8000000000000000ull, "UE: st(0) 0x%04llx:%016llx want bias-adjusted 0x2003:8000000000000000", se, m); break;
    case K_PE: CHECKF(se == 0x3FFD && m == 0xAAAAAAAAAAAAAAABull, "PE: st(0) 0x%04llx:%016llx want 1/3 (stored)", se, m); break;
    default: break;
    }
    unsigned long long fip = cv(&r, "FIP");
    t_info("fip_matches_op", "%d (FIP 0x%llx, op 0x%llx)", (uint32_t)fip == (uint32_t)oplab, fip, oplab);
    char key[32]; snprintf(key, sizeof key, "code_x87_%s", k_kind[k]); t_info(key, "0x%08llx", code);
    t_info("handler_fcw", "0x%04llx", cv(&r, "HCW"));
    t_info("exception_info", "n=%llu [0]=0x%llx", cv(&r, "NPARAM"), cv(&r, "INFO0"));
    if (wine_code_gap) XFAIL("Wine %s reports x87 %s as 0x%08llx (Windows [K2]: 0x%08lx or INVALID_OPERATION)", t_wine_version(), k_kind[k], code, want_code(1, k));
}
TEST(x87_unmasked_invalid_operation) { run_fx_x87(K_IE); }
TEST(x87_unmasked_stack_fault)       { run_fx_x87(K_SF); }
TEST(x87_unmasked_denormal_operand)  { run_fx_x87(K_DE); }
TEST(x87_unmasked_divide_by_zero)    { run_fx_x87(K_ZE); }
TEST(x87_unmasked_overflow)          { run_fx_x87(K_OE); }
TEST(x87_unmasked_underflow)         { run_fx_x87(K_UE); }
TEST(x87_unmasked_inexact)           { run_fx_x87(K_PE); }

TEST(x87_unmasked_unhandled_terminates_with_code) {
    static t_child_result r;
    REQUIRE(t_child_run("fpx", "x87 2 unhandled", 15000, &r));
    REQUIREF(!r.timed_out, "child timed out");
    CHECKF(r.exc_code != 0, "unhandled unmasked x87 divide-by-zero did not terminate the child: %s", r.out);
    CHECKF(code_acceptable(1, K_ZE, r.exc_code), "unhandled code 0x%08lx want 0x%08lx", r.exc_code, want_code(1, K_ZE));
}

/* ======================================================================================
 * 5. x87 architecture
 * ====================================================================================== */
TEST(x87_fninit_state) {
    /* SDM 8.1.10/FNINIT: FCW=0x037F, FSW=0, FTW=empty; FIP/FDP cleared */
    t_set_x87_cw(0x0C7F);
    t_fninit();
    unsigned short cw = t_get_x87_cw(), sw = t_get_x87_sw(), tw = t_get_x87_tw();
    CHECKF((cw & CW_CMP) == 0x033F, "FCW after FNINIT 0x%04x want 0x037F", cw);
    CHECKF(sw == 0, "FSW after FNINIT 0x%04x", sw);
    CHECKF(tw == 0xFFFF, "FTW after FNINIT 0x%04x", tw);
    t_info("fninit_fcw_raw", "0x%04x", cw);
}

TEST(x87_cw_writable_bits) {
    /* masks, PC (00/10/11) and RC must read back; record what reserved bits read as */
    static const unsigned short pcs[3] = { CW_PC24, CW_PC53, CW_PC64 };
    unsigned short cw0 = t_get_x87_cw();
    for (int m = 0; m < 64; m += 7)
        for (int p = 0; p < 3; p++)
            for (int rc = 0; rc < 4; rc++) {
                unsigned short v = (unsigned short)(0x0040 | m | pcs[p] | CW_RC(rc));
                t_set_x87_cw(v);
                unsigned short g = t_get_x87_cw();
                CHECKF((g & CW_CMP) == (v & CW_CMP), "FLDCW 0x%04x read back 0x%04x", v, g);
            }
    t_set_x87_cw(0xFFFF); unsigned short hi = t_get_x87_cw();
    t_set_x87_cw(0x0000); unsigned short lo = t_get_x87_cw();
    t_set_x87_cw(cw0);
    t_info("fcw_after_fldcw_ffff", "0x%04x", hi);
    t_info("fcw_after_fldcw_0000", "0x%04x", lo);
    CHECKF((hi & CW_CMP) == CW_CMP, "defined bits of 0xFFFF did not stick: 0x%04x", hi);
}

TEST(x87_stack_order_and_top) {
    unsigned short s1, s2, s3, s4, sv;
    t_f80 o1, o2, o3;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfld1\n\tfnstsw %[s1]\n\tfldz\n\tfnstsw %[s2]\n\tfldpi\n\tfnstsw %[s3]\n\t"
                     "fstpt %[o1]\n\tfstpt %[o2]\n\tfstpt %[o3]\n\tfnstsw %[s4]\n\tfldcw %[sv]"
                     : [sv] "=&m"(sv), [s1] "=m"(s1), [s2] "=m"(s2), [s3] "=m"(s3), [s4] "=m"(s4), [o1] "=m"(o1), [o2] "=m"(o2), [o3] "=m"(o3));
    CHECKF(SW_TOP(s1) == 7 && SW_TOP(s2) == 6 && SW_TOP(s3) == 5 && SW_TOP(s4) == 0, "TOP sequence %u %u %u %u want 7 6 5 0",
           SW_TOP(s1), SW_TOP(s2), SW_TOP(s3), SW_TOP(s4));
    check_f80("first pop (pi, RN)", &o1, 0x4000, 0xC90FDAA22168C235ull);
    check_f80("second pop (0)", &o2, 0x0000, 0);
    check_f80("third pop (1)", &o3, 0x3FFF, 0x8000000000000000ull);
    CHECKF((s4 & 0x3F) == 0, "flags after push/pop 0x%04x", s4);
}

typedef struct x87_layout {
    T_ALIGN(16) uint8_t fx[512];
    uint8_t env1[28], env2[28], sv[108];
    t_f80 in[5], out[5];
    unsigned short cwt, svcw, cwb, swb, cwae, cwas, swas, swar;
} x87_layout;
static T_NOINLINE void x87_layout_probe(x87_layout *L) {
    __asm__ volatile(
        "fnstcw %c[o_svcw](%[p])\n\tfninit\n\tfldcw %c[o_cwt](%[p])\n\t"
        "fldt %c[o_in](%[p])\n\tfldt %c[o_in]+10(%[p])\n\tfldt %c[o_in]+20(%[p])\n\tfldt %c[o_in]+30(%[p])\n\tfldt %c[o_in]+40(%[p])\n\t"
        "fnstcw %c[o_cwb](%[p])\n\tfnstsw %c[o_swb](%[p])\n\t"
        "fnstenv %c[o_env1](%[p])\n\tfnstcw %c[o_cwae](%[p])\n\tfldcw %c[o_cwt](%[p])\n\t"
        "fxsave %c[o_fx](%[p])\n\t"
        "fnsave %c[o_sv](%[p])\n\tfnstcw %c[o_cwas](%[p])\n\tfnstsw %c[o_swas](%[p])\n\tfnstenv %c[o_env2](%[p])\n\t"
        "frstor %c[o_sv](%[p])\n\tfnstsw %c[o_swar](%[p])\n\t"
        "fstpt %c[o_out](%[p])\n\tfstpt %c[o_out]+10(%[p])\n\tfstpt %c[o_out]+20(%[p])\n\tfstpt %c[o_out]+30(%[p])\n\tfstpt %c[o_out]+40(%[p])\n\t"
        "fninit\n\tfldcw %c[o_svcw](%[p])"
        :
        : [p] "r"(L), [o_svcw] "i"(offsetof(x87_layout, svcw)), [o_cwt] "i"(offsetof(x87_layout, cwt)),
          [o_in] "i"(offsetof(x87_layout, in)), [o_out] "i"(offsetof(x87_layout, out)),
          [o_cwb] "i"(offsetof(x87_layout, cwb)), [o_swb] "i"(offsetof(x87_layout, swb)),
          [o_env1] "i"(offsetof(x87_layout, env1)), [o_env2] "i"(offsetof(x87_layout, env2)),
          [o_cwae] "i"(offsetof(x87_layout, cwae)), [o_fx] "i"(offsetof(x87_layout, fx)),
          [o_sv] "i"(offsetof(x87_layout, sv)), [o_cwas] "i"(offsetof(x87_layout, cwas)),
          [o_swas] "i"(offsetof(x87_layout, swas)), [o_swar] "i"(offsetof(x87_layout, swar))
        : "memory");
}
static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

TEST(x87_tag_word_and_save_layouts) {
    /* Push 1.0 (valid), +0 (zero), inf, denormal, QNaN (special) -> physical R7..R3.
     * Full tag word (FNSTENV/FNSAVE, 2 bits/register): 00 01 10 10 10 11 11 11 = 0x1ABF.
     * Abridged FXSAVE tag (1 bit/register, 1 = non-empty): R7..R3 = 0xF8. */
    static x87_layout L;
    memset(&L, 0, sizeof L);
    L.cwt = 0x0F7F;      /* RC=chop, PC=64, all masked */
    L.in[0] = XV(X_ONE); L.in[1] = XV(X_ZERO); L.in[2] = XV(X_INF); L.in[3] = XV(X_DEN); L.in[4] = XV(X_QNAN);
    unsigned mx = t_get_mxcsr();
    x87_layout_probe(&L);
    CHECKF((L.cwb & CW_CMP) == (0x0F7F & CW_CMP), "FCW 0x%04x", L.cwb);
    CHECKF(SW_TOP(L.swb) == 3, "TOP after 5 pushes %u want 3", SW_TOP(L.swb));
    CHECKF((L.swb & 0x3F) == 0, "flags after loads 0x%04x", L.swb);
    /* FNSTENV 28-byte protected-mode format: FCW@0 FSW@4 FTW@8 FIP@12 FCS/FOP@16 FDP@20 FDS@24 */
    CHECKF(rd16(L.env1 + 0) == L.cwb, "FNSTENV FCW 0x%04x want 0x%04x", rd16(L.env1), L.cwb);
    CHECKF(rd16(L.env1 + 4) == L.swb, "FNSTENV FSW 0x%04x want 0x%04x", rd16(L.env1 + 4), L.swb);
    CHECKF(rd16(L.env1 + 8) == 0x1ABF, "FNSTENV FTW 0x%04x want 0x1ABF", rd16(L.env1 + 8));
    CHECKF((L.cwae & 0x3F) == 0x3F, "FNSTENV must mask all exceptions afterwards: FCW 0x%04x", L.cwae);
    CHECKF((L.cwae & 0x1F00) == (L.cwb & 0x1F00), "FNSTENV changed PC/RC: 0x%04x", L.cwae);
    /* FXSAVE: FCW@0 FSW@2 abridged FTW@4 MXCSR@24 ST(i)@32+16i */
    CHECKF(rd16(L.fx + 0) == L.cwb, "FXSAVE FCW 0x%04x", rd16(L.fx));
    CHECKF(rd16(L.fx + 2) == L.swb, "FXSAVE FSW 0x%04x want 0x%04x", rd16(L.fx + 2), L.swb);
    CHECKF(L.fx[4] == 0xF8, "FXSAVE abridged FTW 0x%02x want 0xF8", L.fx[4]);
    CHECKF(rd32(L.fx + 24) == mx, "FXSAVE MXCSR 0x%08x want 0x%08x", rd32(L.fx + 24), mx);
    for (int i = 0; i < 5; i++) {
        char w[48]; snprintf(w, sizeof w, "FXSAVE ST(%d)", i);
        t_f80 v; memcpy(v.b, L.fx + 32 + 16 * i, 10);
        check_f80(w, &v, f80_se(&L.in[4 - i]), f80_m(&L.in[4 - i]));
    }
    /* FNSAVE: 28-byte env + ST(0..7) packed 10 bytes each; then the FPU is re-initialized */
    CHECKF(rd16(L.sv + 0) == L.cwb && rd16(L.sv + 4) == L.swb && rd16(L.sv + 8) == 0x1ABF,
           "FNSAVE env FCW 0x%04x FSW 0x%04x FTW 0x%04x", rd16(L.sv), rd16(L.sv + 4), rd16(L.sv + 8));
    for (int i = 0; i < 5; i++) {
        char w[48]; snprintf(w, sizeof w, "FNSAVE ST(%d)", i);
        t_f80 v; memcpy(v.b, L.sv + 28 + 10 * i, 10);
        check_f80(w, &v, f80_se(&L.in[4 - i]), f80_m(&L.in[4 - i]));
    }
    CHECKF((L.cwas & CW_CMP) == 0x033F, "FCW after FNSAVE 0x%04x want 0x037F (FNINIT)", L.cwas);
    CHECKF(L.swas == 0, "FSW after FNSAVE 0x%04x", L.swas);
    CHECKF(rd16(L.env2 + 8) == 0xFFFF, "FTW after FNSAVE 0x%04x", rd16(L.env2 + 8));
    CHECKF(L.swar == L.swb, "FSW after FRSTOR 0x%04x want 0x%04x", L.swar, L.swb);
    for (int i = 0; i < 5; i++) {
        char w[48]; snprintf(w, sizeof w, "pop %d after FRSTOR", i);
        check_f80(w, &L.out[i], f80_se(&L.in[4 - i]), f80_m(&L.in[4 - i]));
    }
    t_info("fnstenv_fip", "0x%08x", rd32(L.env1 + 12));
}

TEST(x87_precision_control) {
    /* SDM 8.1.5.2: PC rounds the significand of add/sub/mul/div/sqrt results to 24/53/64 bits;
     * the exponent range stays extended. C1 = 1 when the result was rounded up (inexact). */
    static const struct { const char *n; t_f80 a, b; uint64_t m[3][4]; uint16_t se; } t[] = {
        { "1/3", X_ONE, X_THREE, { { 0xAAAAAB0000000000ull, 0xAAAAAA0000000000ull, 0xAAAAAB0000000000ull, 0xAAAAAA0000000000ull },
                                   { 0xAAAAAAAAAAAAA800ull, 0xAAAAAAAAAAAAA800ull, 0xAAAAAAAAAAAAB000ull, 0xAAAAAAAAAAAAA800ull },
                                   { 0xAAAAAAAAAAAAAAABull, 0xAAAAAAAAAAAAAAAAull, 0xAAAAAAAAAAAAAAABull, 0xAAAAAAAAAAAAAAAAull } }, 0x3FFD },
        { "10/7", F80I(0x4002, 0xA000000000000000ull), F80I(0x4001, 0xE000000000000000ull),
          { { 0xB6DB6E0000000000ull, 0xB6DB6D0000000000ull, 0xB6DB6E0000000000ull, 0xB6DB6D0000000000ull },
            { 0xB6DB6DB6DB6DB800ull, 0xB6DB6DB6DB6DB000ull, 0xB6DB6DB6DB6DB800ull, 0xB6DB6DB6DB6DB000ull },
            { 0xB6DB6DB6DB6DB6DBull, 0xB6DB6DB6DB6DB6DBull, 0xB6DB6DB6DB6DB6DCull, 0xB6DB6DB6DB6DB6DBull } }, 0x3FFF },
    };
    static const unsigned short pcs[3] = { CW_PC24, CW_PC53, CW_PC64 };
    static const int pcbits[3] = { 24, 53, 64 };
    for (size_t i = 0; i < T_COUNTOF(t); i++)
        for (int p = 0; p < 3; p++)
            for (int rc = 0; rc < 4; rc++) {
                t_f80 r;
                unsigned short cw = (unsigned short)(0x007F | pcs[p] | CW_RC(rc));
                unsigned short sw = x87_op(XO_DIV, &t[i].a, &t[i].b, cw, &r);
                char w[64]; snprintf(w, sizeof w, "%s PC=%d %s", t[i].n, pcbits[p], k_rc[rc]);
                check_f80(w, &r, t[i].se, t[i].m[p][rc]);
                CHECKF((sw & 0x3F) == MXF_PE, "%s: FSW flags 0x%02x want PE", w, sw & 0x3F);
                /* rounded up <=> result significand above the truncated one */
                uint64_t trunc = (rc == RD || rc == RZ) ? t[i].m[p][rc] : t[i].m[p][RZ];
                int up = t[i].m[p][rc] != trunc;
                CHECKF(!!(sw & SW_C1) == up, "%s: C1=%d want %d (round-up indicator)", w, !!(sw & SW_C1), up);
            }
}

TEST(x87_rounding_frndint_fistp) {
    static const struct { const char *n; t_f80 x; uint16_t se[4]; uint64_t m[4]; int64_t i[4]; } t[] = {
        { "2.5",  F80I(0x4000, 0xA000000000000000ull), { 0x4000, 0x4000, 0x4000, 0x4000 }, { 0x8000000000000000ull, 0x8000000000000000ull, 0xC000000000000000ull, 0x8000000000000000ull }, { 2, 2, 3, 2 } },
        { "-2.5", F80I(0xC000, 0xA000000000000000ull), { 0xC000, 0xC000, 0xC000, 0xC000 }, { 0x8000000000000000ull, 0xC000000000000000ull, 0x8000000000000000ull, 0x8000000000000000ull }, { -2, -3, -2, -2 } },
        { "3.5",  F80I(0x4000, 0xE000000000000000ull), { 0x4001, 0x4000, 0x4001, 0x4000 }, { 0x8000000000000000ull, 0xC000000000000000ull, 0x8000000000000000ull, 0xC000000000000000ull }, { 4, 3, 4, 3 } },
        { "0.5",  X_HALF, { 0x0000, 0x0000, 0x3FFF, 0x0000 }, { 0, 0, 0x8000000000000000ull, 0 }, { 0, 0, 1, 0 } },
        { "-0.5", F80I(0xBFFE, 0x8000000000000000ull), { 0x8000, 0xBFFF, 0x8000, 0x8000 }, { 0, 0x8000000000000000ull, 0, 0 }, { 0, -1, 0, 0 } },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++)
        for (int rc = 0; rc < 4; rc++) {
            t_f80 r, z = X_ZERO; int64_t iv = 0x5555;
            unsigned short cw = (unsigned short)(0x037F | CW_RC(rc));
            char w[48]; snprintf(w, sizeof w, "frndint %s %s", t[i].n, k_rc[rc]);
            unsigned short sw = x87_op(XO_RNDINT, &t[i].x, &z, cw, &r);
            check_f80(w, &r, t[i].se[rc], t[i].m[rc]);
            CHECKF((sw & 0x3F) == MXF_PE, "%s: flags 0x%02x want PE", w, sw & 0x3F);
            sw = x87_fistp64(&t[i].x, cw, &iv);
            CHECKF(iv == t[i].i[rc], "fistp %s %s: %lld want %lld", t[i].n, k_rc[rc], (long long)iv, (long long)t[i].i[rc]);
            CHECKF((sw & 0x3F) == MXF_PE, "fistp %s %s: flags 0x%02x", t[i].n, k_rc[rc], sw & 0x3F);
        }
    int64_t iv; t_f80 big = F80I(0x403E, 0x8000000000000000ull), nbig = F80I(0xC03E, 0x8000000000000000ull);
    unsigned short sw = x87_fistp64(&big, 0x037F, &iv);
    CHECKF(iv == INT64_MIN && (sw & 0x3F) == MXF_IE, "fistp 2^63: %lld flags 0x%02x want indefinite + IE", (long long)iv, sw & 0x3F);
    sw = x87_fistp64(&nbig, 0x037F, &iv);
    CHECKF(iv == INT64_MIN && (sw & 0x3F) == 0, "fistp -2^63: %lld flags 0x%02x want exact", (long long)iv, sw & 0x3F);
}

TEST(x87_masked_exception_flags) {
    static const struct { const char *n; int op; t_f80 a, b; uint16_t se; uint64_t m; unsigned fl; } t[] = {
        { "1/0",        XO_DIV, X_ONE, X_ZERO, 0x7FFF, 0x8000000000000000ull, MXF_ZE },
        { "-1/0",       XO_DIV, X_MONE, X_ZERO, 0xFFFF, 0x8000000000000000ull, MXF_ZE },
        { "0/0",        XO_DIV, X_ZERO, X_ZERO, 0xFFFF, 0xC000000000000000ull, MXF_IE },
        { "sqrt(-1)",   XO_SQRT, X_MONE, X_ZERO, 0xFFFF, 0xC000000000000000ull, MXF_IE },
        { "inf-inf",    XO_SUB, X_INF, X_INF, 0xFFFF, 0xC000000000000000ull, MXF_IE },
        { "inf*0",      XO_MUL, X_INF, X_ZERO, 0xFFFF, 0xC000000000000000ull, MXF_IE },
        { "SNaN+1",     XO_ADD, X_SNAN, X_ONE, 0x7FFF, 0xE000000000000000ull, MXF_IE },
        { "QNaN+1",     XO_ADD, X_QNAN, X_ONE, 0x7FFF, 0xC000000000000123ull, 0 },
        { "1+denormal", XO_ADD, X_ONE, X_DEN, 0x3FFF, 0x8000000000000000ull, MXF_DE | MXF_PE },
        { "max*2",      XO_MUL, X_MAX, X_TWO, 0x7FFF, 0x8000000000000000ull, MXF_OE | MXF_PE },
        { "minnormal*0.5 (exact tiny)", XO_MUL, X_MINN, X_HALF, 0x0000, 0x4000000000000000ull, 0 },
        { "minnormal^2", XO_MUL, X_MINN, X_MINN, 0x0000, 0, MXF_UE | MXF_PE },
        { "1/3",        XO_DIV, X_ONE, X_THREE, 0x3FFD, 0xAAAAAAAAAAAAAAABull, MXF_PE },
        { "2*2 exact",  XO_MUL, X_TWO, X_TWO, 0x4001, 0x8000000000000000ull, 0 },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        t_f80 r;
        unsigned short sw = x87_op(t[i].op, &t[i].a, &t[i].b, 0x037F, &r);
        check_f80(t[i].n, &r, t[i].se, t[i].m);
        char f1[40], f2[40];
        CHECKF((sw & 0x3F) == t[i].fl, "%s: FSW flags %s want %s", t[i].n, flag_str(sw & 0x3F, f1, sizeof f1), flag_str(t[i].fl, f2, sizeof f2));
        CHECKF(!(sw & (SW_ES | SW_SF | 0x8000)), "%s: ES/SF/B set with all exceptions masked (FSW 0x%04x)", t[i].n, sw);
    }
    /* memory-format conversions */
    uint32_t r32; t_f80 r80;
    t_f80 big = F80I(0x407F, 0x8000000000000000ull), tiny = F80I(0x3F69, 0x8000000000000000ull), third;
    unsigned short sw = x87_fst32(&big, 0x037F, &r32);
    CHECKF(r32 == 0x7F800000u && (sw & 0x3F) == (MXF_OE | MXF_PE), "fst m32 2^128: 0x%08x flags 0x%02x", r32, sw & 0x3F);
    sw = x87_fst32(&tiny, 0x037F, &r32);
    CHECKF(r32 == 0 && (sw & 0x3F) == (MXF_UE | MXF_PE), "fst m32 2^-150: 0x%08x flags 0x%02x", r32, sw & 0x3F);
    sw = x87_fst32(&tiny, (unsigned short)(0x037F | CW_RC(RU)), &r32);
    CHECKF(r32 == 1 && (sw & 0x3F) == (MXF_UE | MXF_PE), "fst m32 2^-150 RU: 0x%08x flags 0x%02x", r32, sw & 0x3F);
    third = XV(F80I(0x3FFD, 0xAAAAAAAAAAAAAAABull));
    sw = x87_fst32(&third, 0x037F, &r32);
    CHECKF(r32 == 0x3EAAAAABu && (sw & 0x3F) == MXF_PE, "fst m32 1/3: 0x%08x flags 0x%02x", r32, sw & 0x3F);
    sw = x87_fld32(0x00000001u, 0x037F, &r80);
    check_f80("fld m32 denormal (normalized)", &r80, 0x3F6A, 0x8000000000000000ull);
    CHECKF((sw & 0x3F) == MXF_DE, "fld m32 denormal flags 0x%02x want DE", sw & 0x3F);
    sw = x87_fld32(0x7F800001u, 0x037F, &r80);
    check_f80("fld m32 SNaN (quieted)", &r80, 0x7FFF, 0xC000010000000000ull);
    CHECKF((sw & 0x3F) == MXF_IE, "fld m32 SNaN flags 0x%02x want IE", sw & 0x3F);
}

TEST(x87_stack_overflow_underflow_masked) {
    unsigned short s8, s9, s10, sv;
    t_f80 r9, r10;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\t"
                     "fld1\n\tfld1\n\tfld1\n\tfld1\n\tfld1\n\tfld1\n\tfld1\n\tfld1\n\tfnstsw %[s8]\n\t"
                     "fld1\n\tfnstsw %[s9]\n\tfstpt %[r9]\n\t"
                     "fninit\n\tfadd %%st(1), %%st\n\tfnstsw %[s10]\n\tfstpt %[r10]\n\t"
                     "fninit\n\tfldcw %[sv]"
                     : [sv] "=&m"(sv), [s8] "=m"(s8), [s9] "=m"(s9), [s10] "=m"(s10), [r9] "=m"(r9), [r10] "=m"(r10));
    CHECKF((s8 & 0x3F) == 0 && SW_TOP(s8) == 0, "8 pushes: FSW 0x%04x", s8);
    CHECKF((s9 & (0x3F | SW_SF | SW_C1)) == (MXF_IE | SW_SF | SW_C1), "9th push: FSW 0x%04x want IE|SF|C1 (overflow)", s9);
    CHECKF(SW_TOP(s9) == 7, "9th push: TOP %u want 7", SW_TOP(s9));
    check_f80("masked stack overflow loads indefinite", &r9, 0xFFFF, 0xC000000000000000ull);
    CHECKF((s10 & (0x3F | SW_SF | SW_C1)) == (MXF_IE | SW_SF), "fadd on empty: FSW 0x%04x want IE|SF, C1=0 (underflow)", s10);
    check_f80("masked stack underflow result", &r10, 0xFFFF, 0xC000000000000000ull);
}

TEST(x87_fnclex_and_pending_exception) {
    /* FNCLEX clears flags, ES and B without waiting; a pending unmasked exception cleared by FNCLEX
     * must not be reported by a later FWAIT (in-process: a wrong implementation crashes the case). */
    unsigned short a, b, c, d, e, sv, cwz = (unsigned short)(0x037F & ~0x04);
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldz\n\tfld1\n\tfdiv %%st(1), %%st\n\tfnstsw %[a]\n\tfnclex\n\tfnstsw %[b]\n\tfninit\n\tfnstsw %[c]\n\t"
                     "fldcw %[cwz]\n\tfldz\n\tfld1\n\tfdiv %%st(1), %%st\n\tfnstsw %[d]\n\tfnclex\n\tfwait\n\tfnstsw %[e]\n\tfninit\n\tfldcw %[sv]"
                     : [sv] "=&m"(sv), [a] "=m"(a), [b] "=m"(b), [c] "=m"(c), [d] "=m"(d), [e] "=m"(e) : [cwz] "m"(cwz));
    CHECKF((a & 0x3F) == MXF_ZE && SW_TOP(a) == 6, "after masked 1/0: FSW 0x%04x", a);
    CHECKF((b & 0x80FF) == 0 && SW_TOP(b) == 6, "after FNCLEX: FSW 0x%04x want flags clear, TOP kept", b);
    CHECKF(c == 0, "after FNINIT: FSW 0x%04x", c);
    CHECKF((d & (SW_ES | MXF_ZE)) == (SW_ES | MXF_ZE), "unmasked 1/0 pending: FSW 0x%04x want ES|ZE", d);
    CHECKF((e & 0x80FF) == 0, "after FNCLEX+FWAIT: FSW 0x%04x", e);
}

TEST(x87_fxam_classes) {
    static const struct { const char *n; t_f80 x; int empty; unsigned short cc; } t[] = {
        { "+0", X_ZERO, 0, SW_C3 }, { "-0", F80I(0x8000, 0), 0, SW_C3 | SW_C1 },
        { "+1", X_ONE, 0, SW_C2 }, { "-1", X_MONE, 0, SW_C2 | SW_C1 },
        { "+inf", X_INF, 0, SW_C2 | SW_C0 }, { "-inf", F80I(0xFFFF, 0x8000000000000000ull), 0, SW_C2 | SW_C0 | SW_C1 },
        { "QNaN", X_QNAN, 0, SW_C0 }, { "SNaN", X_SNAN, 0, SW_C0 }, { "-QNaN", X_IND, 0, SW_C0 | SW_C1 },
        { "denormal", X_DEN, 0, SW_C3 | SW_C2 }, { "-denormal", F80I(0x8000, 1), 0, SW_C3 | SW_C2 | SW_C1 },
        { "unnormal (unsupported)", F80I(0x3FFF, 0x4000000000000000ull), 0, 0 },
        { "max", X_MAX, 0, SW_C2 }, { "min normal", X_MINN, 0, SW_C2 },
        { "empty", X_ZERO, 1, SW_C3 | SW_C0 },
    };
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        unsigned short sw = x87_fxam(&t[i].x, t[i].empty);
        unsigned short mask = t[i].empty ? (SW_C3 | SW_C2 | SW_C0) : SW_CC;
        CHECKF((sw & mask) == t[i].cc, "fxam %s: C3..C0 0x%04x want 0x%04x", t[i].n, sw & mask, t[i].cc);
        CHECKF((sw & 0x3F) == 0, "fxam %s raised flags 0x%02x", t[i].n, sw & 0x3F);
    }
}

TEST(x87_fcom_fucom_condition_codes) {
    static const struct { const char *n; t_f80 a, b; unsigned short cc; unsigned ie_fcom, ie_fucom; } t[] = {
        { "2>1", X_TWO, X_ONE, 0, 0, 0 }, { "1<2", X_ONE, X_TWO, SW_C0, 0, 0 }, { "1==1", X_ONE, X_ONE, SW_C3, 0, 0 },
        { "+0==-0", X_ZERO, F80I(0x8000, 0), SW_C3, 0, 0 }, { "-inf<1", F80I(0xFFFF, 0x8000000000000000ull), X_ONE, SW_C0, 0, 0 },
        { "1 ? QNaN", X_ONE, X_QNAN, SW_C3 | SW_C2 | SW_C0, MXF_IE, 0 },
        { "1 ? SNaN", X_ONE, X_SNAN, SW_C3 | SW_C2 | SW_C0, MXF_IE, MXF_IE },
        { "den>0", X_DEN, X_ZERO, 0, MXF_DE, MXF_DE },
    };
    /* fcomi/fucomi: ZF PF CF = 000 (>), 001 (<), 100 (=), 111 (unordered) */
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        unsigned ef;
        unsigned want_ef = t[i].cc == 0 ? 0 : t[i].cc == SW_C0 ? 1 : t[i].cc == SW_C3 ? 4 : 7;
        for (int k = 0; k < 4; k++) {
            static const char *const nm[4] = { "fcom", "fucom", "fcomi", "fucomi" };
            unsigned short sw = x87_cmp(k, &t[i].a, &t[i].b, &ef);
            unsigned want_fl = (k & 1) ? t[i].ie_fucom : t[i].ie_fcom;
            if (k < 2) CHECKF((sw & (SW_C3 | SW_C2 | SW_C0)) == t[i].cc, "%s %s: CC 0x%04x want 0x%04x", nm[k], t[i].n, sw & (SW_C3 | SW_C2 | SW_C0), t[i].cc);
            else CHECKF(ef == want_ef, "%s %s: ZF|PF|CF=%u want %u", nm[k], t[i].n, ef, want_ef);
            CHECKF((sw & 0x3F) == want_fl, "%s %s: flags 0x%02x want 0x%02x", nm[k], t[i].n, sw & 0x3F, want_fl);
        }
    }
}

TEST(x87_80bit_load_store_exact) {
    /* FLD m80 / FSTP m80 copy every supported encoding bit-exactly (no #IA even for SNaN: SDM FLD) */
    static const t_f80 v[] = {
        X_ONE, X_MONE, X_ZERO, { { 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x80 } }, X_INF, X_IND, X_MAX, X_MINN, X_DEN,
        X_QNAN, X_SNAN, { { 0x35, 0xC2, 0x68, 0x21, 0xA2, 0xDA, 0x0F, 0xC9, 0x00, 0xC0 } }, { { 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0xC1, 0xFF, 0x7F } },
    };
    for (size_t i = 0; i < T_COUNTOF(v); i++) {
        t_f80 r; unsigned short sw;
        x87_roundtrip(&v[i], &r, &sw);
        char w[48]; snprintf(w, sizeof w, "80-bit roundtrip #%u", (unsigned)i);
        check_f80(w, &r, f80_se(&v[i]), f80_m(&v[i]));
        CHECKF((sw & 0x3F) == 0, "%s: flags 0x%02x", w, sw & 0x3F);
    }
}

TEST(x87_long_double_follows_precision_control) {
    /* mingw long double is the 80-bit format; its arithmetic is x87 and therefore honours FCW.PC:
     * 1 + 2^-63 is representable at PC=64 and rounds to 1 at PC=53. */
    volatile long double one = 1.0L, eps = 0x1p-63L, s;
    t_info("sizeof_long_double", "%u", (unsigned)sizeof(long double));
    t_info("LDBL_MANT_DIG", "%d", LDBL_MANT_DIG);
    if (LDBL_MANT_DIG != 64) SKIP("long double is not the x87 80-bit format here");
    t_set_x87_cw(0x037F);
    s = one + eps;
    CHECKF(s != one, "PC=64: 1 + 2^-63 == 1");
    t_set_x87_cw(0x027F);
    s = one + eps;
    CHECKF(s == one, "PC=53: 1 + 2^-63 != 1");
    t_set_x87_cw(0x007F);
    s = one / 3.0L;
    t_f80 b; memcpy(b.b, (const void *)&s, 10);
    check_f80("PC=24: 1.0L/3", &b, 0x3FFD, 0xAAAAAB0000000000ull);
    t_set_x87_cw(0x037F);
#ifdef T_X64
    /* x64 C double arithmetic is SSE: x87 PC must not affect it */
    volatile double d1 = 1.0, d3 = 3.0, q;
    t_set_x87_cw(0x007F);
    q = d1 / d3;
    t_set_x87_cw(0x037F);
    uint64_t qb; memcpy(&qb, (const void *)&q, 8);
    CHECKF(qb == 0x3FD5555555555555ull, "x64 double 1/3 under x87 PC=24: 0x%016llx", (unsigned long long)qb);
#endif
}

TEST(mmx_tags_valid_and_emms) {
    /* SDM 9.5: any MMX instruction sets TOP=0 and all tags valid; an MMX write sets bits 79:64 of the
     * register to 1s; EMMS empties all tags. Without EMMS the next x87 push overflows the stack. */
    static T_ALIGN(16) uint8_t fx[512];
    uint8_t e0[28], e1[28], e2[28];
    unsigned short s1, s3, sv;
    uint64_t v0 = 0x0123456789ABCDEFull, v3 = 0xFEDCBA9876543210ull;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfnstenv %[e0]\n\t"
                     "movq %[v0], %%mm0\n\tmovq %[v3], %%mm3\n\tfnstenv %[e1]\n\tfnstsw %[s1]\n\tfxsave %[fx]\n\t"
                     "emms\n\tfnstenv %[e2]\n\t"
                     "fninit\n\tmovq %[v0], %%mm0\n\tfld1\n\tfnstsw %[s3]\n\temms\n\tfninit\n\tfldcw %[sv]"
                     : [sv] "=&m"(sv), [e0] "=m"(e0), [e1] "=m"(e1), [e2] "=m"(e2), [s1] "=m"(s1), [s3] "=m"(s3), [fx] "=m"(fx)
                     : [v0] "m"(v0), [v3] "m"(v3) : "mm0", "mm3", "memory");
    CHECKF(rd16(e0 + 8) == 0xFFFF, "FTW before MMX 0x%04x", rd16(e0 + 8));
    /* FNSTENV reconstructs the 2-bit tags from the register contents for every non-empty register:
     * after MMX no register is empty (no 11b field) and MM0/MM3 (exponent 0xFFFF) read as special (10b). */
    unsigned tw1 = rd16(e1 + 8);
    for (int i = 0; i < 8; i++) CHECKF(((tw1 >> (2 * i)) & 3) != 3, "FTW after MMX 0x%04x: R%d still empty", tw1, i);
    CHECKF((tw1 & 3) == 2 && ((tw1 >> 6) & 3) == 2, "FTW after MMX 0x%04x: R0/R3 (exp 0xFFFF) want special (10b)", tw1);
    t_info("ftw_after_mmx", "0x%04x", tw1);
    CHECKF(SW_TOP(s1) == 0, "TOP after MMX %u want 0", SW_TOP(s1));
    CHECKF(fx[4] == 0xFF, "FXSAVE abridged FTW after MMX 0x%02x want 0xFF", fx[4]);
    uint64_t m0, m3; memcpy(&m0, fx + 32, 8); memcpy(&m3, fx + 32 + 3 * 16, 8);
    CHECKF(m0 == v0 && rd16(fx + 32 + 8) == 0xFFFF, "MM0 alias: mantissa 0x%016llx exp 0x%04x", (unsigned long long)m0, rd16(fx + 40));
    CHECKF(m3 == v3 && rd16(fx + 32 + 3 * 16 + 8) == 0xFFFF, "MM3 alias: mantissa 0x%016llx exp 0x%04x", (unsigned long long)m3, rd16(fx + 88));
    CHECKF(rd16(e2 + 8) == 0xFFFF, "FTW after EMMS 0x%04x want 0xFFFF", rd16(e2 + 8));
    CHECKF((s3 & (0x3F | SW_SF | SW_C1)) == (MXF_IE | SW_SF | SW_C1), "x87 push after MMX without EMMS: FSW 0x%04x want stack overflow", s3);
}

TEST(mmx_state_aliases_x87_registers) {
    /* FLD m80 into the stack, read it back through MMX (low 64 bits of the physical register) */
    uint64_t out0 = 0, out1 = 0; unsigned short sv;
    t_f80 a = F80I(0x4000, 0xC90FDAA22168C235ull), b = F80I(0x3FFF, 0x8000000000000001ull);
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfldt %[b]\n\tfldt %[a]\n\tffree %%st(0)\n\tffree %%st(1)\n\tfincstp\n\tfincstp\n\t"
                     "movq %%mm6, %[o0]\n\tmovq %%mm7, %[o1]\n\temms\n\tfninit\n\tfldcw %[sv]"
                     : [sv] "=&m"(sv), [o0] "=m"(out0), [o1] "=m"(out1) : [a] "m"(a), [b] "m"(b) : "mm6", "mm7", "memory");
    /* after fninit TOP=0: b -> R7, a -> R6; MMn aliases physical register Rn */
    CHECKF(out0 == 0xC90FDAA22168C235ull, "MM6 0x%016llx want significand of the value in R6", (unsigned long long)out0);
    CHECKF(out1 == 0x8000000000000001ull, "MM7 0x%016llx want significand of the value in R7", (unsigned long long)out1);
}

/* ======================================================================================
 * 6. save/restore instructions
 * ====================================================================================== */
TEST(fxrstor_restores_fp_env) {
    static T_ALIGN(16) uint8_t fx[512];
    unsigned mxa = MX_DEF | MX_RC(RU) | MX_FTZ | MXF_PE;
    unsigned short cwa = (unsigned short)(0x007F | CW_RC(RZ) | CW_PC24);
    t_set_mxcsr(mxa); t_set_x87_cw(cwa);
    t_fxsave(fx);
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    t_fxrstor(fx);
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    CHECKF(mx == mxa, "MXCSR after FXRSTOR 0x%04x want 0x%04x", mx, mxa);
    CHECKF((cw & CW_CMP) == (cwa & CW_CMP), "FCW after FXRSTOR 0x%04x want 0x%04x", cw, cwa);
}

TEST(xrstor_mxcsr_and_x87_components) {
    /* SDM 13.7/13.8 (standard form): MXCSR is saved/restored when RFBM[1] or RFBM[2] is set (not with
     * x87 alone); x87 state (FCW...) only with RFBM[0]; XSTATE_BV[0]=0 => x87 init (FCW=0x037F). */
    if (!cpu_has(F_XSAVE)) SKIP("XSAVE not usable (no OSXSAVE)");
    size_t sz = xsave_std_size(3);
    REQUIRE(sz > 0 && sz <= 4096);
    static T_ALIGN(64) uint8_t area[4096];
    memset(area, 0, sizeof area);
    unsigned mxa = MX_DEF | MX_RC(RU) | MX_FTZ, mxb = MX_DEF | MX_RC(RD);
    unsigned short cwa = 0x0F7F, cwb = 0x027F;
    t_set_mxcsr(mxa); t_set_x87_cw(cwa);
    t_xsave(area, 3);
    t_set_mxcsr(mxb); t_set_x87_cw(cwb);
    t_xrstor(area, 1);
    unsigned m1 = t_get_mxcsr(); unsigned short c1 = t_get_x87_cw();
    t_set_mxcsr(mxb); t_set_x87_cw(cwb);
    t_xrstor(area, 2);
    unsigned m2 = t_get_mxcsr(); unsigned short c2 = t_get_x87_cw();
    uint64_t bv; memcpy(&bv, area + 512, 8);
    bv &= ~3ull; memcpy(area + 512, &bv, 8);
    t_set_mxcsr(mxb); t_set_x87_cw(cwb);
    t_xrstor(area, 3);
    unsigned m3 = t_get_mxcsr(); unsigned short c3 = t_get_x87_cw();
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    CHECKF((c1 & CW_CMP) == (cwa & CW_CMP), "XRSTOR rfbm=1: FCW 0x%04x want 0x%04x", c1, cwa);
    CHECKF(m1 == mxb, "XRSTOR rfbm=1 changed MXCSR: 0x%04x want 0x%04x", m1, mxb);
    CHECKF(m2 == mxa, "XRSTOR rfbm=2: MXCSR 0x%04x want 0x%04x", m2, mxa);
    CHECKF((c2 & CW_CMP) == (cwb & CW_CMP), "XRSTOR rfbm=2 changed FCW: 0x%04x", c2);
    CHECKF((c3 & CW_CMP) == 0x033F, "XRSTOR init x87: FCW 0x%04x want 0x037F", c3);
    /* SDM 13.8.1 (standard form): MXCSR is loaded from the legacy area whenever RFBM[1] or RFBM[2] is
     * set, independent of XSTATE_BV[1] (only the XMM registers are initialized) */
    CHECKF(m3 == mxa, "XRSTOR rfbm=3 with XSTATE_BV[1]=0: MXCSR 0x%04x want the saved 0x%04x (prior 0x%04x)", m3, mxa, mxb);
}

/* ======================================================================================
 * 7. CRT agreement (msvcrt _controlfp_s/_control87/_statusfp/_clearfp/_fpreset, mingw fenv)
 * ====================================================================================== */
typedef unsigned int (__cdecl *ctl87_2_t)(unsigned int, unsigned int, unsigned int *, unsigned int *);
typedef void (__cdecl *statusfp2_t)(unsigned int *, unsigned int *);
typedef void (__cdecl *fpreset_t)(void);
static void *crt_sym(const char *n) { HMODULE m = GetModuleHandleA("msvcrt.dll"); return m ? (void *)GetProcAddress(m, n) : NULL; }
/* On x86 the documented _control87/_controlfp contract covers both x87 and SSE2; we require the SSE2
 * half only when the CRT exports __control87_2 (the SSE2-aware generation). */
static int crt_sse_on_x86(void) { return crt_sym("__control87_2") != NULL; }

TEST(crt_controlfp_rounding) {
    static const unsigned crc[4] = { _RC_NEAR, _RC_DOWN, _RC_UP, _RC_CHOP };
    unsigned cur = 0;
    for (int m = 0; m < 4; m++) {
        CHECKF(_controlfp_s(&cur, crc[m], _MCW_RC) == 0, "_controlfp_s(%s) failed", k_rc[m]);
        CHECKF((cur & _MCW_RC) == crc[m], "_controlfp_s returned RC 0x%x want 0x%x", cur & _MCW_RC, crc[m]);
        CHECKF((_controlfp(0, 0) & _MCW_RC) == crc[m], "_controlfp(0,0) RC 0x%x", _controlfp(0, 0) & _MCW_RC);
#ifdef T_X64
        CHECKF(((t_get_mxcsr() >> 13) & 3) == (unsigned)m, "%s: MXCSR.RC %u", k_rc[m], (t_get_mxcsr() >> 13) & 3);
#else
        CHECKF(((t_get_x87_cw() >> 10) & 3) == (unsigned)m, "%s: FCW.RC %u", k_rc[m], (t_get_x87_cw() >> 10) & 3);
        if (crt_sse_on_x86()) CHECKF(((t_get_mxcsr() >> 13) & 3) == (unsigned)m, "%s: MXCSR.RC %u (x86 CRT must set SSE2 too)", k_rc[m], (t_get_mxcsr() >> 13) & 3);
#endif
    }
    _controlfp_s(&cur, _RC_NEAR, _MCW_RC);
}

TEST(crt_control87_reads_hardware) {
    /* the CRT reports what the hardware register holds, not a cached copy */
#ifdef T_X64
    t_set_mxcsr(MX_DEF | MX_RC(RU));
    CHECKF((_controlfp(0, 0) & _MCW_RC) == _RC_UP, "_controlfp after LDMXCSR RC=up: 0x%x", _controlfp(0, 0));
    t_set_mxcsr((MX_DEF & ~0x0200u) | MX_RC(RZ));
    unsigned c = _control87(0, 0);
    CHECKF((c & _MCW_RC) == _RC_CHOP, "_control87 RC 0x%x want chop", c & _MCW_RC);
    CHECKF(!(c & _EM_ZERODIVIDE) && (c & _EM_INVALID) && (c & _EM_INEXACT), "_control87 masks 0x%x want ZE unmasked only", c & _MCW_EM);
    t_set_mxcsr(MX_DEF | MX_FTZ | (daz_supported() ? MX_DAZ : 0));
    c = _control87(0, 0);
    t_info("control87_dn_with_ftz_daz", "0x%x", c & _MCW_DN);
    CHECKF((c & _MCW_DN) != 0, "_control87 does not report FTZ/DAZ (0x%x)", c);
#else
    t_set_x87_cw((unsigned short)(0x027F | CW_RC(RD)));
    CHECKF((_controlfp(0, 0) & _MCW_RC) == _RC_DOWN, "_controlfp after FLDCW RC=down: 0x%x", _controlfp(0, 0));
    t_set_x87_cw((unsigned short)(0x007F | CW_RC(RZ) | CW_PC24));
    unsigned c = _control87(0, 0);
    CHECKF((c & _MCW_PC) == _PC_24, "_control87 PC 0x%x want _PC_24", c & _MCW_PC);
#endif
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
}

TEST(crt_controlfp_exception_masks) {
    static const struct { unsigned em; unsigned mxm; unsigned short x87m; int use87; } t[] = {
        { _EM_INVALID, 0x0080, 0x01, 0 }, { _EM_ZERODIVIDE, 0x0200, 0x04, 0 }, { _EM_OVERFLOW, 0x0400, 0x08, 0 },
        { _EM_UNDERFLOW, 0x0800, 0x10, 0 }, { _EM_INEXACT, 0x1000, 0x20, 0 }, { _EM_DENORMAL, 0x0100, 0x02, 1 },
    };
    unsigned cur;
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        if (t[i].use87) _control87(_MCW_EM & ~t[i].em, _MCW_EM);     /* _controlfp ignores the denormal mask */
        else CHECK(_controlfp_s(&cur, _MCW_EM & ~t[i].em, _MCW_EM) == 0);
        unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
        if (t[i].use87) _control87(_MCW_EM, _MCW_EM); else _controlfp_s(&cur, _MCW_EM, _MCW_EM);
#ifdef T_X64
        CHECKF((mx & MX_MASKS) == (MX_MASKS & ~t[i].mxm), "unmask 0x%x: MXCSR masks 0x%04x want 0x%04x", t[i].em, mx & MX_MASKS, MX_MASKS & ~t[i].mxm);
        (void)cw;
#else
        CHECKF((cw & 0x3F) == (0x3F & ~t[i].x87m), "unmask 0x%x: FCW masks 0x%02x want 0x%02x", t[i].em, cw & 0x3F, 0x3F & ~t[i].x87m);
        if (crt_sse_on_x86())
            CHECKF((mx & MX_MASKS) == (MX_MASKS & ~t[i].mxm), "unmask 0x%x: MXCSR masks 0x%04x want 0x%04x", t[i].em, mx & MX_MASKS, MX_MASKS & ~t[i].mxm);
#endif
    }
    CHECKF((t_get_mxcsr() & MX_MASKS) == MX_MASKS, "masks not restored: MXCSR 0x%04x", t_get_mxcsr());
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
}

TEST(crt_controlfp_denormal_control) {
#ifdef T_X86
    if (!crt_sse_on_x86()) SKIP("x86 CRT without SSE2 control (__control87_2 not exported)");
#endif
    unsigned cur = 0;
    errno_t e = _controlfp_s(&cur, _DN_FLUSH, _MCW_DN);
    REQUIREF(e == 0, "_controlfp_s(_DN_FLUSH, _MCW_DN) failed: %d", (int)e);
    unsigned mx = t_get_mxcsr();
    _controlfp_s(&cur, _DN_SAVE, _MCW_DN);
    unsigned mx2 = t_get_mxcsr();
    CHECKF(mx & MX_FTZ, "_DN_FLUSH did not set MXCSR.FTZ (0x%04x)", mx);
    if (daz_supported()) CHECKF(mx & MX_DAZ, "_DN_FLUSH did not set MXCSR.DAZ (0x%04x)", mx);
    CHECKF(!(mx2 & (MX_FTZ | MX_DAZ)), "_DN_SAVE did not clear FTZ/DAZ (0x%04x)", mx2);
    t_info("dn_flush_mxcsr", "0x%04x", mx);
}

TEST(crt_controlfp_precision_x86) {
#ifdef T_X64
    SKIP("precision control is x86-only in the CRT (documented: not supported on x64)");
#else
    static const struct { unsigned pc; unsigned short bits; } t[] = { { _PC_24, 0x000 }, { _PC_53, 0x200 }, { _PC_64, 0x300 } };
    unsigned cur;
    for (size_t i = 0; i < T_COUNTOF(t); i++) {
        CHECK(_controlfp_s(&cur, t[i].pc, _MCW_PC) == 0);
        CHECKF((t_get_x87_cw() & 0x300) == t[i].bits, "_PC 0x%x: FCW 0x%04x", t[i].pc, t_get_x87_cw());
        CHECKF((_controlfp(0, 0) & _MCW_PC) == t[i].pc, "_controlfp reports PC 0x%x", _controlfp(0, 0) & _MCW_PC);
    }
    t_set_x87_cw(0x037F);
#endif
}

static T_NOINLINE void sse_div_keep_flags(uint32_t a, uint32_t b) {
    __asm__ volatile("movd %[a], %%xmm0\n\tmovd %[b], %%xmm1\n\tdivss %%xmm1, %%xmm0" :: [a] "m"(a), [b] "m"(b) : "xmm0", "xmm1");
}
static T_NOINLINE void x87_div_keep_flags(const t_f80 *a, const t_f80 *b) {
    __asm__ volatile("fldt %[b]\n\tfldt %[a]\n\tfdiv %%st(1), %%st\n\tfstp %%st(0)\n\tfstp %%st(0)" :: [a] "m"(*a), [b] "m"(*b));
}

TEST(crt_statusfp_clearfp) {
    t_set_mxcsr(MX_DEF); t_fnclex();
#ifdef T_X64
    sse_div_keep_flags(0x3F800000u, 0);
    CHECKF(t_get_mxcsr() & MXF_ZE, "precondition: MXCSR.ZE");
    unsigned s = _statusfp();
    CHECKF(s & _SW_ZERODIVIDE, "_statusfp 0x%x lacks _SW_ZERODIVIDE", s);
    CHECKF(!(s & _SW_INVALID), "_statusfp 0x%x has _SW_INVALID", s);
    sse_div_keep_flags(0x3F800000u, 0x40400000u);
    s = _clearfp();
    CHECKF((s & (_SW_ZERODIVIDE | _SW_INEXACT)) == (_SW_ZERODIVIDE | _SW_INEXACT), "_clearfp returned 0x%x", s);
    CHECKF((t_get_mxcsr() & MX_FLAGS) == 0, "MXCSR flags after _clearfp 0x%02x", t_get_mxcsr() & MX_FLAGS);
    CHECKF((_statusfp() & (0x1F | _SW_DENORMAL)) == 0, "_statusfp after _clearfp 0x%x", _statusfp());
#else
    t_f80 one = X_ONE, zero = X_ZERO;
    x87_div_keep_flags(&one, &zero);
    unsigned s = _statusfp();
    CHECKF(s & _SW_ZERODIVIDE, "_statusfp 0x%x lacks _SW_ZERODIVIDE (x87)", s);
    s = _clearfp();
    CHECKF(s & _SW_ZERODIVIDE, "_clearfp returned 0x%x", s);
    CHECKF((t_get_x87_sw() & 0x3F) == 0, "FSW flags after _clearfp 0x%04x", t_get_x87_sw());
    statusfp2_t sf2 = (statusfp2_t)crt_sym("_statusfp2");
    if (sf2) {
        unsigned x = 0, sse = 0;
        sse_div_keep_flags(0x3F800000u, 0);
        sf2(&x, &sse);
        CHECKF(sse & _SW_ZERODIVIDE, "_statusfp2 SSE status 0x%x lacks _SW_ZERODIVIDE", sse);
        CHECKF(!(x & _SW_ZERODIVIDE), "_statusfp2 x87 status 0x%x has the SSE flag", x);
        _clearfp();
        t_info("mxcsr_flags_after_clearfp_x86", "0x%02x", t_get_mxcsr() & MX_FLAGS);
    } else t_note("_statusfp2 not exported: SSE status not checked");
#endif
    t_set_mxcsr(MX_DEF); t_fnclex();
}

TEST(crt_fpreset_msvcrt) {
    /* msvcrt _fpreset: control words to the default (round-nearest, all masked), status cleared.
     * (mingw's own _fpreset is FNINIT only - recorded for information.) */
    fpreset_t fr = (fpreset_t)crt_sym("_fpreset");
    if (!fr) SKIP("msvcrt _fpreset not exported");
    t_set_mxcsr(MX_DEF | MX_RC(RU) | MX_FTZ | MXF_PE | MXF_ZE);
    t_set_x87_cw((unsigned short)(0x007F | CW_RC(RZ) | CW_PC24));
    fr();
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw(), sw = t_get_x87_sw();
#ifdef T_X86
    /* the SSE2 half of the x86 CRT environment only exists in the SSE2-aware msvcrt (see crt_sse_on_x86) */
    if (crt_sse_on_x86())
#endif
    CHECKF((mx & (MX_MASKS | 0x6000 | MX_FLAGS)) == MX_MASKS, "MXCSR after _fpreset 0x%04x: want all masked, RN, flags clear", mx);
#ifdef T_X86
    /* x86: msvcrt's floating-point package is the x87 (mingw float.h: msvcrt _fpreset sets FCW 0x27F) */
    CHECKF((cw & 0x0C3F) == 0x003F, "FCW after _fpreset 0x%04x: want all masked, RN", cw);
    CHECKF((sw & 0x3F) == 0, "FSW flags after _fpreset 0x%04x", sw);
#else
    (void)sw;   /* x64: the CRT's FP environment is MXCSR; what _fpreset does to the x87 is recorded only */
#endif
    t_info("msvcrt_fpreset", "mxcsr=0x%04x fcw=0x%04x", mx, cw);
    t_set_mxcsr(MX_DEF | MX_RC(RU));
    t_set_x87_cw((unsigned short)(0x007F | CW_RC(RZ)));
    _fpreset();
    t_info("mingw_fpreset", "mxcsr=0x%04x fcw=0x%04x", t_get_mxcsr(), t_get_x87_cw());
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
}

TEST(crt_fenv_rounding_modes) {
    /* C99 7.6.3: fesetround changes the mode used by arithmetic; fegetround reports it. mingw's fenv
     * programs both FCW.RC (long double / x86 double) and MXCSR.RC (SSE). */
    static const int fe[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
    static const uint64_t third[4] = { 0x3FD5555555555555ull, 0x3FD5555555555555ull, 0x3FD5555555555556ull, 0x3FD5555555555555ull };
    static const uint64_t mthird[4] = { 0xBFD5555555555555ull, 0xBFD5555555555556ull, 0xBFD5555555555555ull, 0xBFD5555555555555ull };
    volatile double one = 1.0, mone = -1.0, three = 3.0, q;
    for (int m = 0; m < 4; m++) {
        CHECKF(fesetround(fe[m]) == 0, "fesetround(%s) failed", k_rc[m]);
        CHECKF(fegetround() == fe[m], "fegetround 0x%x want 0x%x", fegetround(), fe[m]);
        CHECKF(((t_get_mxcsr() >> 13) & 3) == (unsigned)m, "%s: MXCSR.RC %u", k_rc[m], (t_get_mxcsr() >> 13) & 3);
        CHECKF(((t_get_x87_cw() >> 10) & 3) == (unsigned)m, "%s: FCW.RC %u", k_rc[m], (t_get_x87_cw() >> 10) & 3);
        q = one / three;
        uint64_t b; memcpy(&b, (const void *)&q, 8);
        CHECKF(b == third[m], "%s: 1.0/3.0 = 0x%016llx want 0x%016llx", k_rc[m], (unsigned long long)b, (unsigned long long)third[m]);
        q = mone / three;
        memcpy(&b, (const void *)&q, 8);
        CHECKF(b == mthird[m], "%s: -1.0/3.0 = 0x%016llx want 0x%016llx", k_rc[m], (unsigned long long)b, (unsigned long long)mthird[m]);
    }
    /* C99 7.6.3.2: fesetround returns nonzero for a value that is not a rounding mode and leaves the
     * mode unchanged */
    CHECK(fesetround(FE_UPWARD) == 0);
    int bad = fesetround(FE_UPWARD | 0x10000);
    int after = fegetround();
    unsigned mrc = (t_get_mxcsr() >> 13) & 3;
    fesetround(FE_TONEAREST);
    CHECKF(bad != 0, "fesetround(invalid) returned 0");
    CHECKF(after == FE_UPWARD && mrc == RU, "fesetround(invalid) changed the mode: fegetround 0x%x MXCSR.RC %u", after, mrc);
    fesetround(FE_TONEAREST);
}

TEST(crt_fenv_exception_flags) {
    volatile double one = 1.0, zero = 0.0, r;
    volatile long double lone = 1.0L, lzero = 0.0L, lr;
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
    CHECKF(fetestexcept(FE_ALL_EXCEPT) == 0, "flags after feclearexcept: 0x%x", fetestexcept(FE_ALL_EXCEPT));
    CHECKF((t_get_mxcsr() & MX_FLAGS) == 0 && (t_get_x87_sw() & 0x3F) == 0, "hardware flags after feclearexcept: mxcsr 0x%04x fsw 0x%04x", t_get_mxcsr(), t_get_x87_sw());
    r = one / zero;
    CHECKF(fetestexcept(FE_DIVBYZERO), "1.0/0.0 did not raise FE_DIVBYZERO (0x%x)", fetestexcept(FE_ALL_EXCEPT));
    CHECKF(!fetestexcept(FE_INVALID), "1.0/0.0 raised FE_INVALID");
    r = zero / zero;
    CHECKF(fetestexcept(FE_INVALID), "0.0/0.0 did not raise FE_INVALID");
    CHECK(feclearexcept(FE_DIVBYZERO) == 0);
    CHECKF(!fetestexcept(FE_DIVBYZERO) && fetestexcept(FE_INVALID), "feclearexcept(FE_DIVBYZERO) cleared the wrong flags: 0x%x", fetestexcept(FE_ALL_EXCEPT));
    CHECK(feraiseexcept(FE_OVERFLOW) == 0);
    CHECKF(fetestexcept(FE_OVERFLOW), "feraiseexcept(FE_OVERFLOW) not visible");
    feclearexcept(FE_ALL_EXCEPT);
    lr = lone / lzero;                     /* x87 on both architectures */
    CHECKF(fetestexcept(FE_DIVBYZERO), "long double 1/0 not seen by fetestexcept (x87 status)");
    CHECKF(t_get_x87_sw() & MXF_ZE, "long double 1/0 did not set FSW.ZE");
    feclearexcept(FE_ALL_EXCEPT);
    (void)r; (void)lr;
}

TEST(crt_fenv_hold_update) {
    /* C99 7.6.4.2/7.6.4.4 with long double (x87) operations on both architectures. This toolchain's
     * static libmingwex feholdexcept only saves/clears the x87 environment (FNSTENV+FNCLEX), so the
     * MXCSR part is recorded, not checked: it is not an OS property. */
    volatile long double one = 1.0L, zero = 0.0L, r;
    fenv_t env;
    feclearexcept(FE_ALL_EXCEPT);
    r = one / zero;
    unsigned mxf0 = t_get_mxcsr() & MX_FLAGS;
    CHECK(feholdexcept(&env) == 0);
    CHECKF(fetestexcept(FE_ALL_EXCEPT) == 0, "feholdexcept did not clear flags: 0x%x", fetestexcept(FE_ALL_EXCEPT));
    CHECKF((t_get_mxcsr() & MX_MASKS) == MX_MASKS && (t_get_x87_cw() & 0x3F) == 0x3F, "feholdexcept: not non-stop (mxcsr 0x%04x fcw 0x%04x)", t_get_mxcsr(), t_get_x87_cw());
    r = zero / zero;
    CHECK(fetestexcept(FE_INVALID));
    CHECK(feupdateenv(&env) == 0);
    CHECKF(fetestexcept(FE_DIVBYZERO) && fetestexcept(FE_INVALID), "feupdateenv: want saved DIVBYZERO + raised INVALID, got 0x%x", fetestexcept(FE_ALL_EXCEPT));
    feclearexcept(FE_ALL_EXCEPT);
    t_set_mxcsr(MX_DEF | MXF_ZE);
    feholdexcept(&env);
    t_info("toolchain_feholdexcept_clears_mxcsr_flags", "%d", (t_get_mxcsr() & MX_FLAGS) == 0);
    feupdateenv(&env);
    feclearexcept(FE_ALL_EXCEPT);
    t_set_mxcsr(MX_DEF);
    (void)r; (void)mxf0;
}

TEST(crt_fegetenv_fesetenv_roundtrip) {
    fenv_t e;
    unsigned mx = MX_DEF | MX_RC(RU) | MX_FTZ;
    unsigned short cw = (unsigned short)(0x007F | CW_RC(RD) | CW_PC53);
    t_set_mxcsr(mx); t_set_x87_cw(cw);
    CHECK(fegetenv(&e) == 0);
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    CHECK(fesetenv(&e) == 0);
    unsigned mx2 = t_get_mxcsr(); unsigned short cw2 = t_get_x87_cw();
    /* the x87 part is C99-mandated and implemented with FLDENV; this toolchain's static fesetenv
     * re-reads the live MXCSR instead of the saved one, so the MXCSR part is recorded only */
    t_info("toolchain_fesetenv_restores_mxcsr", "%d (0x%04x want 0x%04x)", (mx2 & ~MX_FLAGS) == mx, mx2, mx);
    CHECKF((cw2 & CW_CMP) == (cw & CW_CMP), "FCW after fesetenv 0x%04x want 0x%04x", cw2, cw);
    CHECK(fesetenv(FE_DFL_ENV) == 0);
    t_info("fe_dfl_env", "mxcsr=0x%04x fcw=0x%04x", t_get_mxcsr(), t_get_x87_cw());
    /* the CRT's FE_DFL_ENV leaves MXCSR rounding control bits 14:13 as it likes (Windows 10 msvcrt: 0xDF80): CRT behaviour, not
     * the OS; only the guaranteed part is required: exceptions masked, sticky flags clear */
    t_note("FE_DFL_ENV MXCSR=0x%04x (RC bits 14:13 not required to be 0)", t_get_mxcsr());
    CHECKF((t_get_mxcsr() & MX_MASKS) == MX_MASKS, "FE_DFL_ENV MXCSR 0x%04x exceptions not masked", t_get_mxcsr());
    CHECKF((t_get_mxcsr() & MX_FLAGS) == 0, "FE_DFL_ENV MXCSR 0x%04x sticky flags set", t_get_mxcsr());
    {   /* fegetenv/fesetenv round trip of a value we set ourselves (x87 part is mandated) */
        fenv_t e2;
        t_set_x87_cw((unsigned short)(0x007F | CW_RC(RU) | CW_PC53));
        CHECK(fegetenv(&e2) == 0);
        t_set_x87_cw(0x037F);
        CHECK(fesetenv(&e2) == 0);
        CHECKF((t_get_x87_cw() & CW_CMP) == ((0x007F | CW_RC(RU) | CW_PC53) & CW_CMP), "FCW after own-env round trip 0x%04x", t_get_x87_cw());
    }
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
}

/* ======================================================================================
 * 8. FP state in CONTEXT
 * ====================================================================================== */
TEST(ctx_rtlcapturecontext_fp_env) {
#ifdef T_X64
    /* [K4] x64 RtlCaptureContext stores MXCSR into CONTEXT.MxCsr and FXSAVEs FltSave */
    static T_ALIGN(16) CONTEXT c;
    unsigned mx = MX_DEF | MX_RC(RZ) | MX_FTZ | MXF_OE;
    unsigned short cw = (unsigned short)(0x007F | CW_RC(RU) | CW_PC53);
    memset(&c, 0, sizeof c);
    t_set_x87_cw(cw); t_set_mxcsr(mx);
    RtlCaptureContext(&c);
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    CHECKF(c.ContextFlags & CONTEXT_FLOATING_POINT & ~CONTEXT_AMD64, "ContextFlags 0x%08lx lacks FLOATING_POINT", c.ContextFlags);
    CHECKF(c.MxCsr == mx, "CONTEXT.MxCsr 0x%08lx want 0x%08x", c.MxCsr, mx);
    CHECKF(c.FltSave.MxCsr == mx, "FltSave.MxCsr 0x%08lx want 0x%08x", c.FltSave.MxCsr, mx);
    CHECKF((c.FltSave.ControlWord & CW_CMP) == (cw & CW_CMP), "FltSave.ControlWord 0x%04x want 0x%04x", c.FltSave.ControlWord, cw);
    CHECKF(c.FltSave.MxCsr_Mask == mxcsr_mask_raw(), "FltSave.MxCsr_Mask 0x%08lx want 0x%08x", c.FltSave.MxCsr_Mask, mxcsr_mask_raw());
#else
    SKIP("x86 RtlCaptureContext captures control/integer/segment registers only");
#endif
}

static volatile unsigned g_rx_new_mx, g_rx_seen_mx;
static volatile unsigned short g_rx_new_cw, g_rx_seen_cw;
static volatile DWORD g_rx_flags;
static volatile int g_rx_mode, g_rx_hit;
#define RX_CODE 0xE0F0F001u
static LONG CALLBACK rx_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != RX_CODE) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord;
    g_rx_hit++;
    g_rx_flags = c->ContextFlags;
#ifdef T_X64
    g_rx_seen_mx = c->MxCsr; g_rx_seen_cw = c->FltSave.ControlWord;
    if (g_rx_mode == 0) { c->MxCsr = g_rx_new_mx; c->FltSave.MxCsr = g_rx_new_mx; c->FltSave.ControlWord = g_rx_new_cw; }
    else c->FltSave.MxCsr = g_rx_new_mx;          /* only the legacy copy: [K3] says it is ignored */
#else
    g_rx_seen_mx = 0xFFFFFFFFu; g_rx_seen_cw = 0xFFFF;
    if (c->ContextFlags & CONTEXT_EXTENDED_REGISTERS & ~CONTEXT_i386) {
        uint32_t m; memcpy(&m, c->ExtendedRegisters + 24, 4); g_rx_seen_mx = m;
        m = g_rx_new_mx; memcpy(c->ExtendedRegisters + 24, &m, 4);
        uint16_t w = g_rx_new_cw; memcpy(c->ExtendedRegisters, &w, 2);
    }
    if (c->ContextFlags & CONTEXT_FLOATING_POINT & ~CONTEXT_i386) {
        g_rx_seen_cw = (unsigned short)c->FloatSave.ControlWord;
        c->FloatSave.ControlWord = (c->FloatSave.ControlWord & 0xFFFF0000u) | g_rx_new_cw;
    }
#endif
    return EXCEPTION_CONTINUE_EXECUTION;
}

TEST(ctx_handler_sees_and_sets_fp_env) {
    /* A continuable software exception: the handler's CONTEXT holds the raising thread's MXCSR/FCW, and
     * the environment written into it is what the thread runs with after EXCEPTION_CONTINUE_EXECUTION. */
    PVOID h = AddVectoredExceptionHandler(1, rx_veh);
    REQUIRE(h != NULL);
    unsigned mx = MX_DEF | MX_RC(RD) | MX_FTZ;
    unsigned short cw = (unsigned short)(0x007F | CW_RC(RZ) | CW_PC53);
    g_rx_new_mx = MX_DEF | MX_RC(RU); g_rx_new_cw = (unsigned short)(0x007F | CW_RC(RU) | CW_PC64);
    g_rx_mode = 0; g_rx_hit = 0;
    t_set_x87_cw(cw); t_set_mxcsr(mx);
    RaiseException(RX_CODE, 0, 0, NULL);
    unsigned mx_after = t_get_mxcsr(); unsigned short cw_after = t_get_x87_cw();
    t_set_mxcsr(MX_DEF); t_set_x87_cw(0x037F);
    RemoveVectoredExceptionHandler(h);
    REQUIRE(g_rx_hit == 1);
    t_info("raise_ctx_flags", "0x%08lx", g_rx_flags);
#ifdef T_X64
    CHECKF(g_rx_seen_mx == mx, "handler CONTEXT.MxCsr 0x%08x want 0x%08x", g_rx_seen_mx, mx);
    CHECKF((g_rx_seen_cw & CW_CMP) == (cw & CW_CMP), "handler FltSave.ControlWord 0x%04x want 0x%04x", g_rx_seen_cw, cw);
    CHECKF((mx_after & ~MX_FLAGS) == g_rx_new_mx, "MXCSR after continue 0x%04x want handler's 0x%04x", mx_after, g_rx_new_mx);
    CHECKF((cw_after & CW_CMP) == (g_rx_new_cw & CW_CMP), "FCW after continue 0x%04x want handler's 0x%04x", cw_after, g_rx_new_cw);
#else
    if (!(g_rx_flags & CONTEXT_EXTENDED_REGISTERS & ~CONTEXT_i386) && !(g_rx_flags & CONTEXT_FLOATING_POINT & ~CONTEXT_i386)) {
        t_note("x86 RaiseException CONTEXT has no FP state (flags 0x%08lx): nothing to check", g_rx_flags);
        return;
    }
    if (g_rx_flags & CONTEXT_EXTENDED_REGISTERS & ~CONTEXT_i386) {
        CHECKF(g_rx_seen_mx == mx, "handler MXCSR 0x%08x want 0x%08x", g_rx_seen_mx, mx);
        CHECKF((mx_after & ~MX_FLAGS) == g_rx_new_mx, "MXCSR after continue 0x%04x want 0x%04x", mx_after, g_rx_new_mx);
    }
    if (g_rx_flags & CONTEXT_FLOATING_POINT & ~CONTEXT_i386) {
        CHECKF((g_rx_seen_cw & CW_CMP) == (cw & CW_CMP), "handler FCW 0x%04x want 0x%04x", g_rx_seen_cw, cw);
        CHECKF((cw_after & CW_CMP) == (g_rx_new_cw & CW_CMP), "FCW after continue 0x%04x want 0x%04x", cw_after, g_rx_new_cw);
    }
#endif
}

TEST(ctx_legacy_fltsave_mxcsr_info) {
#ifdef T_X64
    /* Information only: which MXCSR copy in an x64 CONTEXT is honoured on continue ([K3]: CONTEXT.MxCsr). */
    PVOID h = AddVectoredExceptionHandler(1, rx_veh);
    REQUIRE(h != NULL);
    g_rx_new_mx = MX_DEF | MX_RC(RZ); g_rx_mode = 1; g_rx_hit = 0;
    t_set_mxcsr(MX_DEF);
    RaiseException(RX_CODE, 0, 0, NULL);
    unsigned mx_after = t_get_mxcsr();
    t_set_mxcsr(MX_DEF);
    RemoveVectoredExceptionHandler(h);
    REQUIRE(g_rx_hit == 1);
    t_info("fltsave_mxcsr_only_applied", "%d (MXCSR after 0x%04x)", (mx_after & ~MX_FLAGS) == g_rx_new_mx, mx_after);
#else
    SKIP("x64 CONTEXT layout only");
#endif
}

TEST(ctx_get_set_thread_context_fp_env) {
    /* GetThreadContext on a suspended thread reports its MXCSR/FCW/FSW/tags; SetThreadContext with a new
     * MXCSR/FCW is what the thread continues with. */
    volatile long stop = 0, ready = 0;
    static spin_args a;
    memset(&a, 0, sizeof a);
    a.mx_in = MX_DEF | MX_RC(RZ) | MX_FTZ | MXF_PE | MXF_UE;
    a.cw_in = (unsigned short)(0x007F | CW_RC(RD) | CW_PC24);
    a.stop = &stop; a.ready = &ready; a.max = 0x7FFFFFFFul;
    HANDLE th = CreateThread(NULL, 64 * 1024, th_env_spin, &a, 0, NULL);
    if (!th) SKIP("CreateThread failed");
    if (!wait_count(&ready, 1, 5000)) { stop = 1; WaitForSingleObject(th, 5000); CloseHandle(th); REQUIREF(0, "worker never reached the spin"); }
    if (SuspendThread(th) == (DWORD)-1) {
        DWORD e = GetLastError();
        stop = 1; WaitForSingleObject(th, 5000); CloseHandle(th);   /* never leave the spinner behind */
        REQUIREF(0, "SuspendThread failed: %lu", e);
    }
    static T_ALIGN(16) CONTEXT c;
    memset(&c, 0, sizeof c);
    unsigned new_mx = MX_DEF | MX_RC(RU) | MXF_IE;
    unsigned short new_cw = (unsigned short)(0x007F | CW_RC(RU) | CW_PC64);
#ifdef T_X64
    c.ContextFlags = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
    BOOL ok = GetThreadContext(th, &c);
    CHECKF(ok, "GetThreadContext failed: %lu", GetLastError());
    if (ok) {
        CHECKF(c.MxCsr == a.mx_in, "CONTEXT.MxCsr 0x%08lx want 0x%08x", c.MxCsr, a.mx_in);
        CHECKF(c.FltSave.MxCsr == a.mx_in, "FltSave.MxCsr 0x%08lx want 0x%08x", c.FltSave.MxCsr, a.mx_in);
        CHECKF((c.FltSave.ControlWord & CW_CMP) == (a.cw_in & CW_CMP), "FltSave.ControlWord 0x%04x want 0x%04x", c.FltSave.ControlWord, a.cw_in);
        CHECKF((c.FltSave.StatusWord & 0x38FF) == 0x3004, "FltSave.StatusWord 0x%04x want TOP=6 ZE", c.FltSave.StatusWord);
        CHECKF(c.FltSave.TagWord == 0xC0, "FltSave.TagWord (abridged) 0x%02x want 0xC0", c.FltSave.TagWord);
        c.MxCsr = new_mx; c.FltSave.MxCsr = new_mx; c.FltSave.ControlWord = new_cw;
        c.ContextFlags = CONTEXT_FLOATING_POINT;
        CHECKF(SetThreadContext(th, &c), "SetThreadContext failed: %lu", GetLastError());
    }
#else
    c.ContextFlags = CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS;
    BOOL ok = GetThreadContext(th, &c);
    CHECKF(ok, "GetThreadContext failed: %lu", GetLastError());
    if (ok) {
        CHECKF((c.FloatSave.ControlWord & CW_CMP) == (a.cw_in & CW_CMP), "FloatSave.ControlWord 0x%08lx want 0x%04x", c.FloatSave.ControlWord, a.cw_in);
        CHECKF((c.FloatSave.StatusWord & 0x38FF) == 0x3004, "FloatSave.StatusWord 0x%08lx want TOP=6 ZE", c.FloatSave.StatusWord);
        /* FNSAVE-format tags: R7/R6 in use, R5..R0 empty (0x6FFF when reconstructed from contents) */
        unsigned tw = c.FloatSave.TagWord & 0xFFFF;
        CHECKF((tw & 0x0FFF) == 0x0FFF && ((tw >> 12) & 3) != 3 && ((tw >> 14) & 3) != 3, "FloatSave.TagWord 0x%04x: want R7,R6 in use, others empty", tw);
        t_info("wow64_ctx_tagword", "0x%04x", tw);
        uint32_t m; memcpy(&m, c.ExtendedRegisters + 24, 4);
        CHECKF(m == a.mx_in, "ExtendedRegisters MXCSR 0x%08x want 0x%08x", m, a.mx_in);
        m = new_mx; memcpy(c.ExtendedRegisters + 24, &m, 4);
        memcpy(c.ExtendedRegisters, &new_cw, 2);
        c.FloatSave.ControlWord = (c.FloatSave.ControlWord & 0xFFFF0000u) | new_cw;
        CHECKF(SetThreadContext(th, &c), "SetThreadContext failed: %lu", GetLastError());
    }
#endif
    ResumeThread(th);
    stop = 1;
    CHECKF(WaitForSingleObject(th, 5000) == WAIT_OBJECT_0, "worker did not stop");
    CloseHandle(th);
    if (ok) {
        CHECKF(a.mx_out == new_mx, "worker MXCSR after SetThreadContext 0x%08x want 0x%08x", a.mx_out, new_mx);
        CHECKF((a.cw_out & CW_CMP) == (new_cw & CW_CMP), "worker FCW after SetThreadContext 0x%04x want 0x%04x", a.cw_out, new_cw);
        CHECKF((a.sw_out & 0x38FF) == 0x3004, "worker FSW 0x%04x: unrelated x87 state lost", a.sw_out);
    }
}

TEST(ctx_set_context_before_thread_start_info) {
    /* Information: FP state put into a CREATE_SUSPENDED thread before it ever ran. */
    fp_snap s; memset(&s, 0xCC, sizeof s);
    HANDLE th = CreateThread(NULL, 0, th_snap, &s, CREATE_SUSPENDED, NULL);
    if (!th) SKIP("CreateThread failed");
    static T_ALIGN(16) CONTEXT c;
    memset(&c, 0, sizeof c);
    unsigned new_mx = MX_DEF | MX_RC(RD);
#ifdef T_X64
    c.ContextFlags = CONTEXT_FLOATING_POINT;
    BOOL ok = GetThreadContext(th, &c);
    if (ok) {
        t_info("suspended_new_thread_ctx", "mxcsr=0x%08lx fcw=0x%04x", c.MxCsr, c.FltSave.ControlWord);
        c.MxCsr = new_mx; c.FltSave.MxCsr = new_mx;
        ok = SetThreadContext(th, &c);
    }
#else
    c.ContextFlags = CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS;
    BOOL ok = GetThreadContext(th, &c);
    if (ok) {
        uint32_t m; memcpy(&m, c.ExtendedRegisters + 24, 4);
        t_info("suspended_new_thread_ctx", "mxcsr=0x%08x fcw=0x%04lx", m, c.FloatSave.ControlWord & 0xFFFF);
        m = new_mx; memcpy(c.ExtendedRegisters + 24, &m, 4);
        ok = SetThreadContext(th, &c);
    }
#endif
    ResumeThread(th);
    CHECKF(WaitForSingleObject(th, 5000) == WAIT_OBJECT_0, "thread did not finish");
    CloseHandle(th);
    CHECKF(ok, "Get/SetThreadContext on a suspended new thread failed: %lu", GetLastError());
    t_info("pre_start_setcontext_mxcsr_seen", "0x%04x (set 0x%04x)", s.mx, new_mx);
}

/* ======================================================================================
 * 9. exception precedence, FLDENV-loaded exceptions, concurrent exceptions, reserved-bit
 *    handling by the OS and the save/restore instructions, CPU migration
 * ====================================================================================== */

/* <exe> --child=fpx_prec <0|1>: divps with lane 0 = 1/0 (ZE) and lane 1 = 1/3 (PE).
 * mode 0: ZE unmasked, PE masked; mode 1: ZE masked, PE unmasked. */
CHILD(fpx_prec) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    if (!AddVectoredExceptionHandler(1, fx_veh)) { printf("NOVEH=1\n"); return 3; }
    uint32_t a[4] = { 0x3F800000u, 0x3F800000u, 0x40000000u, 0x40800000u };
    uint32_t b[4] = { 0x00000000u, 0x40400000u, 0x3F800000u, 0x3F800000u }, r[4] = { 0 };
    unsigned mo = 0;
    fx_sse_packed(mode == 0 ? (MX_DEF & ~0x0200u) : (MX_DEF & ~0x1000u), a, b, r, &mo);
    printf("RES0=0x%08x\nRES1=0x%08x\nRES2=0x%08x\nRES3=0x%08x\nMO=0x%08x\n", r[0], r[1], r[2], r[3], mo);
    fx_print();
    return 0;
}

static void run_fx_prec(int mode) {
    /* SDM 11.5.2 / 11.5.2.1: for a packed instruction the pre-computation exceptions (IE, DE, ZE) are
     * detected in all lanes first. An unmasked pre-computation exception is reported with only the
     * pre-computation flags set (the post-computation conditions OE/UE/PE are not evaluated). If the
     * pre-computation exceptions are all masked, an unmasked post-computation exception is reported
     * with the masked pre-computation flags set as well. Either way the destination is unchanged. */
    static t_child_result r;
    char args[16]; snprintf(args, sizeof args, "%d", mode);
    REQUIRE(t_child_run("fpx_prec", args, 15000, &r));
    REQUIREF(!r.timed_out, "child timed out");
    REQUIREF(strstr(r.out, "LOOP=") == NULL, "resume with masked MXCSR did not take effect: %s", r.out);
    REQUIREF(r.exit_code == 0, "child exit 0x%08lx exc 0x%08lx: %s", r.exit_code, r.exc_code, r.out);
    int k = mode == 0 ? K_ZE : K_PE;
    unsigned long long hits = cv(&r, "HITS"), code = cv(&r, "CODE"), addr = cv(&r, "ADDR"), lab = cv(&r, "LABEL");
    unsigned long long cmx = cv(&r, "CTXMX"), x0 = cv(&r, "XMM0LO"), mo = cv(&r, "MO");
    CHECKF(hits == 1, "handler ran %llu times, want 1", hits);
    int code_ok = code_acceptable(0, k, (DWORD)code);
    int wine_code_gap = !code_ok && t_is_wine();
    if (!wine_code_gap) CHECKF(code_ok, "exception code 0x%08llx, want 0x%08lx (%s)", code, want_code(0, k), k_kind[k]);
    CHECKF(addr == lab, "ExceptionAddress 0x%llx != faulting divps 0x%llx", addr, lab);
    if (cmx != 0xFFFFFFFFull) {
        unsigned want = mode == 0 ? MXF_ZE : (MXF_ZE | MXF_PE);
        char f1[40], f2[40];
        CHECKF((cmx & MX_FLAGS) == want, "CONTEXT MXCSR flags %s want %s (MXCSR 0x%08llx)",
               flag_str((unsigned)cmx & MX_FLAGS, f1, sizeof f1), flag_str(want, f2, sizeof f2), cmx);
        CHECKF(x0 == 0x3F8000003F800000ull, "XMM0[63:0] at fault 0x%016llx want the unchanged source", x0);
    } else t_note("CONTEXT carries no MXCSR/XMM state");
    CHECK_EQ(cv(&r, "RES0"), 0x7F800000u); CHECK_EQ(cv(&r, "RES1"), 0x3EAAAAABu);
    CHECK_EQ(cv(&r, "RES2"), 0x40000000u); CHECK_EQ(cv(&r, "RES3"), 0x40800000u);
    CHECKF((mo & MX_FLAGS) == (MXF_ZE | MXF_PE), "flags after the masked re-execution 0x%02llx want ZE|PE", mo & MX_FLAGS);
    t_info("code", "0x%08llx", code);
    if (wine_code_gap) XFAIL("Wine %s reports this as 0x%08llx", t_wine_version(), code);
}
TEST(sse_unmasked_packed_precomputation_first) { run_fx_prec(0); }
TEST(sse_unmasked_packed_postcomputation_with_masked_precomputation) { run_fx_prec(1); }

/* <exe> --child=fpx_env: FLDENV loads an x87 environment with a pending unmasked ZE (FSW.ES|ZE, FCW.ZM=0);
 * the next waiting instruction (FWAIT, label 2:) must raise it. */
CHILD(fpx_env) {
    if (!AddVectoredExceptionHandler(1, fx_veh)) { printf("NOVEH=1\n"); return 3; }
    static uint8_t env[28];
    unsigned short sv = 0, swo = 0;
    __asm__ volatile("fnstcw %[sv]\n\tfninit\n\tfnstenv %[e]" : [sv] "=m"(sv), [e] "=m"(env) :: "memory");
    uint16_t cw = (uint16_t)(0x037F & ~0x04), sw = 0x8084;     /* B | ES | ZE, TOP = 0 */
    memcpy(env, &cw, 2); memcpy(env + 4, &sw, 2);
    __asm__ volatile(LAB_TO("2f", "lab")
                     "fldenv %[e]\n" "2:\n\tfwait\n\tfnstsw %[swo]\n\tfninit\n\tfldcw %[sv]"
                     : [lab] "=m"(g_fx_label), [swo] "=m"(swo) : [e] "m"(env), [sv] "m"(sv) : LAB_CLOBBER, "memory");
    printf("SWO=0x%04x\n", swo);
    fx_print();
    return 0;
}
TEST(x87_fldenv_pending_exception_delivered) {
    /* SDM 8.6 / FLDENV: an unmasked exception flag loaded by FLDENV is reported by the next waiting x87
     * instruction. Windows delivers it like any x87 #MF ([K2]: ZE -> FLOAT_DIVIDE_BY_ZERO). */
    static t_child_result r;
    REQUIRE(t_child_run("fpx_env", "", 15000, &r));
    REQUIREF(!r.timed_out, "child timed out");
    REQUIREF(strstr(r.out, "LOOP=") == NULL, "resume with cleared FSW did not take effect: %s", r.out);
    REQUIREF(r.exit_code == 0, "child exit 0x%08lx exc 0x%08lx: %s", r.exit_code, r.exc_code, r.out);
    unsigned long long hits = cv(&r, "HITS"), code = cv(&r, "CODE"), addr = cv(&r, "ADDR"), lab = cv(&r, "LABEL");
    unsigned long long csw = cv(&r, "CTXSW"), ccw = cv(&r, "CTXCW"), swo = cv(&r, "SWO"), fip = cv(&r, "FIP");
    CHECKF(hits == 1, "handler ran %llu times, want 1 (FWAIT must report the loaded exception)", hits);
    CHECKF(code == 0xC000008Eull, "exception code 0x%08llx want 0xC000008E (FLOAT_DIVIDE_BY_ZERO)", code);
#ifdef T_X64
    CHECKF(addr == lab, "ExceptionAddress 0x%llx != the FWAIT 0x%llx", addr, lab);
    (void)fip;
#else
    /* as in run_fx_x87: WOW64 reports the FWAIT; other 32-bit implementations the FIP (here the FIP
     * loaded by FLDENV) */
    CHECKF(addr == lab || (uint32_t)addr == (uint32_t)fip, "ExceptionAddress 0x%llx is neither the FWAIT 0x%llx nor FIP 0x%llx", addr, lab, fip);
    t_info("x87_exception_address", "%s", addr == lab ? "fwait" : "fip");
#endif
    if (csw != 0xFFFFFFFFull) {
        CHECKF((csw & (SW_ES | MXF_ZE)) == (SW_ES | MXF_ZE), "CONTEXT FSW 0x%04llx want ES|ZE", csw);
        CHECKF(!(ccw & 0x04), "CONTEXT FCW 0x%04llx has ZM set", ccw);
    }
    CHECKF((swo & 0x80FF) == 0, "FSW after resume 0x%04llx", swo);
}

/* <exe> --child=fpx_mt <threads> <iters>: every thread repeatedly takes an unmasked SSE ZE with its own
 * MXCSR (distinct RC/FTZ/DAZ); the shared VEH checks that the CONTEXT holds the faulting thread's MXCSR. */
static DWORD g_mt_tls = TLS_OUT_OF_INDEXES;
static volatile LONG g_mt_hits, g_mt_bad, g_mt_nofp, g_mt_limit;
static volatile unsigned g_mt_first_got, g_mt_first_want;
static LONG CALLBACK mt_veh(EXCEPTION_POINTERS *ep) {
    if (!is_flt_code(ep->ExceptionRecord->ExceptionCode)) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord;
    unsigned want = (unsigned)(uintptr_t)TlsGetValue(g_mt_tls), got;
    if (InterlockedIncrement(&g_mt_hits) > g_mt_limit) { printf("LOOP=1\n"); fflush(stdout); ExitProcess(0xBAD1); }
#ifdef T_X64
    got = c->MxCsr;
    c->MxCsr = (got & ~MX_FLAGS) | 0x0200u;      /* mask ZE, clear the flags: the divss re-executes masked */
    c->FltSave.MxCsr = c->MxCsr;
#else
    if (c->ContextFlags & CONTEXT_EXTENDED_REGISTERS & ~CONTEXT_i386) {
        memcpy(&got, c->ExtendedRegisters + 24, 4);
        unsigned n = (got & ~MX_FLAGS) | 0x0200u;
        memcpy(c->ExtendedRegisters + 24, &n, 4);
    } else {
        got = t_get_mxcsr();
        InterlockedIncrement(&g_mt_nofp);
        t_set_mxcsr((got & ~MX_FLAGS) | 0x0200u);
    }
#endif
    if ((got & ~MX_FLAGS) != want || (got & MX_FLAGS) != MXF_ZE)
        if (InterlockedIncrement(&g_mt_bad) == 1) { g_mt_first_got = got; g_mt_first_want = want; }
    return EXCEPTION_CONTINUE_EXECUTION;
}
typedef struct mt_arg { unsigned mx; int iters; int bad_after; unsigned last; } mt_arg;
static DWORD WINAPI mt_thread(void *p) {
    mt_arg *a = (mt_arg *)p;
    TlsSetValue(g_mt_tls, (void *)(uintptr_t)a->mx);
    uint32_t one = 0x3F800000u, zero = 0;
    for (int i = 0; i < a->iters; i++) {
        unsigned mo = 0, mx = a->mx, def = MX_DEF;
        uint32_t r = 0;
        __asm__ volatile("ldmxcsr %[mx]\n\tmovd %[one], %%xmm0\n\tmovd %[zero], %%xmm1\n\tdivss %%xmm1, %%xmm0\n\t"
                         "stmxcsr %[mo]\n\tldmxcsr %[def]\n\tmovd %%xmm0, %[r]"
                         : [mo] "=m"(mo), [r] "=m"(r) : [mx] "m"(mx), [one] "m"(one), [zero] "m"(zero), [def] "m"(def)
                         : "xmm0", "xmm1", "memory");
        if (r != 0x7F800000u || mo != (mx | 0x0200u | MXF_ZE)) { a->bad_after++; a->last = mo; }
        if ((i & 7) == 0) SwitchToThread();
    }
    return 0;
}
CHILD(fpx_mt) {
    int n = argc > 1 ? atoi(argv[1]) : 16, iters = argc > 2 ? atoi(argv[2]) : 64;
    if (n < 1 || n > 64 || iters < 1 || iters > 1000) return 2;
    g_mt_tls = TlsAlloc();
    if (g_mt_tls == TLS_OUT_OF_INDEXES) { printf("NOTLS=1\n"); return 3; }
    g_mt_limit = n * iters * 2 + 16;
    if (!AddVectoredExceptionHandler(1, mt_veh)) { printf("NOVEH=1\n"); return 3; }
    static mt_arg a[64]; static HANDLE h[64];
    int daz = daz_supported(), started = 0;
    for (int i = 0; i < n; i++) {
        a[i].mx = (MX_DEF & ~0x0200u) | MX_RC(i & 3) | ((i >> 2) & 1 ? MX_FTZ : 0) | (daz && ((i >> 3) & 1) ? MX_DAZ : 0);
        a[i].iters = iters;
        h[i] = CreateThread(NULL, 64 * 1024, mt_thread, &a[i], 0, NULL);
        if (!h[i]) break;
        started++;
    }
    int bad_after = 0, timeouts = 0; unsigned last = 0;
    for (int i = 0; i < started; i++) {
        if (WaitForSingleObject(h[i], 20000) != WAIT_OBJECT_0) timeouts++;
        CloseHandle(h[i]);
        if (a[i].bad_after) { bad_after += a[i].bad_after; last = a[i].last; }
    }
    printf("MTTHREADS=%d\nMTITERS=%d\nMTHITS=%ld\nMTBAD=%ld\nMTFIRSTGOT=0x%08x\nMTFIRSTWANT=0x%08x\nMTAFTERBAD=%d\nMTLAST=0x%08x\nMTNOFP=%ld\nMTTIMEOUTS=%d\n",
           started, iters, g_mt_hits, g_mt_bad, g_mt_first_got, g_mt_first_want, bad_after, last, g_mt_nofp, timeouts);
    return 0;
}
TEST(sse_unmasked_exceptions_concurrent_threads) {
    /* Each CONTEXT handed to a handler is the faulting thread's own state, even when many threads
     * fault at the same time with different environments; the environment each handler writes back is
     * applied to that thread only. */
    static t_child_result r;
    REQUIRE(t_child_run("fpx_mt", "16 64", 60000, &r));
    REQUIREF(!r.timed_out, "child timed out");
    REQUIREF(strstr(r.out, "LOOP=") == NULL, "handler re-entered without progress: %s", r.out);
    REQUIREF(r.exit_code == 0 && r.exc_code == 0, "child exit 0x%08lx exc 0x%08lx: %s", r.exit_code, r.exc_code, r.out);
    unsigned long long thr = 0, it = 0, hits = 0, bad = 0, after = 0, to = 0, fg = 0, fw = 0, last = 0, nofp = 0;
    REQUIRE(child_hex(&r, "MTTHREADS", &thr) && child_hex(&r, "MTITERS", &it) && child_hex(&r, "MTHITS", &hits));
    child_hex(&r, "MTBAD", &bad); child_hex(&r, "MTAFTERBAD", &after); child_hex(&r, "MTTIMEOUTS", &to);
    child_hex(&r, "MTFIRSTGOT", &fg); child_hex(&r, "MTFIRSTWANT", &fw); child_hex(&r, "MTLAST", &last); child_hex(&r, "MTNOFP", &nofp);
    if (thr < 2) SKIP("could not create worker threads in the child (%llu)", thr);
    CHECKF(to == 0, "%llu worker threads did not finish", to);
    CHECKF(hits == thr * it, "handler ran %llu times, want %llu (threads %llu x %llu)", hits, thr * it, thr, it);
    CHECKF(bad == 0, "%llu handler CONTEXTs did not hold the faulting thread's MXCSR (first: 0x%08llx want 0x%08llx|ZE)", bad, fg, fw);
    CHECKF(after == 0, "%llu re-executions did not run with the handler's MXCSR (last 0x%08llx)", after, last);
    t_info("threads", "%llu x %llu exceptions", thr, it);
    if (nofp) t_note("%llu exception CONTEXTs had no extended registers (live MXCSR fixed instead)", nofp);
}

/* <exe> --child=ctx_badmx <0|1>: hand the OS an MXCSR with reserved bits through a CONTEXT.
 * mode 0: VEH continue (NtContinue); mode 1: SetThreadContext on a suspended spinning worker. */
#define BADMX_VAL (0xFFFF0000u | MX_DEF | MX_RC(RU))
static LONG CALLBACK badmx_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != RX_CODE) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord;
#ifdef T_X64
    c->MxCsr = BADMX_VAL; c->FltSave.MxCsr = BADMX_VAL;
    printf("CTXSET=1\n");
#else
    if (c->ContextFlags & CONTEXT_EXTENDED_REGISTERS & ~CONTEXT_i386) {
        unsigned v = BADMX_VAL; memcpy(c->ExtendedRegisters + 24, &v, 4);
        printf("CTXSET=1\n");
    } else printf("CTXSET=0\n");
#endif
    return EXCEPTION_CONTINUE_EXECUTION;
}
CHILD(ctx_badmx) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    if (mode == 0) {
        if (!AddVectoredExceptionHandler(1, badmx_veh)) { printf("NOVEH=1\n"); return 3; }
        RaiseException(RX_CODE, 0, 0, NULL);
        unsigned mx = t_get_mxcsr();
        t_set_mxcsr(MX_DEF);
        printf("SURVIVED=1\nMX=0x%08x\n", mx);
        return 0;
    }
    volatile long stop = 0, ready = 0;
    static spin_args a;
    memset(&a, 0, sizeof a);
    a.mx_in = MX_DEF; a.cw_in = 0x027F; a.stop = &stop; a.ready = &ready; a.max = 0x7FFFFFFFul;
    HANDLE th = CreateThread(NULL, 64 * 1024, th_env_spin, &a, 0, NULL);
    if (!th) { printf("NOTHREAD=1\n"); return 4; }
    if (!wait_count(&ready, 1, 5000)) { stop = 1; printf("NOSPIN=1\n"); return 4; }
    int setok = -1; DWORD err = 0;
    if (SuspendThread(th) != (DWORD)-1) {
        static T_ALIGN(16) CONTEXT c;
        memset(&c, 0, sizeof c);
#ifdef T_X64
        c.ContextFlags = CONTEXT_FLOATING_POINT;
        if (GetThreadContext(th, &c)) {
            c.MxCsr = BADMX_VAL; c.FltSave.MxCsr = BADMX_VAL;
            setok = SetThreadContext(th, &c) ? 1 : 0;
        }
#else
        c.ContextFlags = CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS;
        if (GetThreadContext(th, &c)) {
            unsigned v = BADMX_VAL; memcpy(c.ExtendedRegisters + 24, &v, 4);
            setok = SetThreadContext(th, &c) ? 1 : 0;
        }
#endif
        if (setok == 0) err = GetLastError();
        ResumeThread(th);
    }
    stop = 1;
    DWORD w = WaitForSingleObject(th, 5000), ec = 0;
    GetExitCodeThread(th, &ec);
    CloseHandle(th);
    printf("SETOK=%d\nSETERR=%lu\nJOINED=%d\nTHREADEXIT=0x%08lx\nMX=0x%08x\nSURVIVED=1\n", setok, err, w == WAIT_OBJECT_0, ec, a.mx_out);
    return 0;
}
static void run_badmx(int mode) {
    /* An MXCSR value with reserved bits set would #GP on LDMXCSR/FXRSTOR/XRSTOR. A CONTEXT is untrusted
     * input: [K3] the kernel ANDs CONTEXT.MxCsr with the processor's MXCSR mask (re-checked in the 19041
     * corpus: KxContextToKframes "TrapFrame->MxCsr = Context->MxCsr & KiMxCsrMask", used by NtContinue
     * and NtSetContextThread). Contract checked: the
     * process and the thread survive, and the thread never runs with a reserved MXCSR bit set (the API
     * may also refuse the value). What is applied is recorded. */
    static t_child_result r;
    REQUIRE(t_child_run("ctx_badmx", mode ? "1" : "0", 5000, &r));
    if (mode == 0 && !r.timed_out && r.exc_code != 0 && !t_is_wine()) {
        /* documented contract: NtContinue either sanitises the reserved bits (normal exit) or rejects them by faulting the
         * child (Windows 10 22H2 raises STATUS_ACCESS_VIOLATION); both are accepted, the harness process is never involved */
        if (r.exc_code == 0xC0000005u || r.exc_code == 0xC000000Du) {
            t_info("ctx_continue_outcome", "faulted (exc 0x%08lx, exit 0x%08lx)", r.exc_code, r.exit_code);
            t_note("NtContinue with reserved MXCSR bits faults the child on this OS instead of sanitising: %s", r.out);
            return;
        }
    }
    if (mode == 0 && !r.timed_out && r.exc_code == 0) t_info("ctx_continue_outcome", "sanitised or survived (normal exit)");
    int survived = !r.timed_out && strstr(r.out, "SURVIVED=1") != NULL && r.exc_code == 0;
    if (!survived && t_is_wine())
        XFAIL("Wine %s: a CONTEXT MXCSR with reserved bits %s (exit 0x%08lx exc 0x%08lx)", t_wine_version(),
              r.timed_out ? "hangs the process" : "kills the process", r.exit_code, r.exc_code);
    REQUIREF(!r.timed_out, "child hung after a CONTEXT with a reserved MXCSR bit");
    REQUIREF(survived, "process died: exit 0x%08lx exc 0x%08lx: %s", r.exit_code, r.exc_code, r.out);
    unsigned long long mx = 0, setok = 1, joined = 1, ctxset = 1;
    REQUIRE(child_hex(&r, "MX", &mx));
    if (mode == 0) {
        child_hex(&r, "CTXSET", &ctxset);
        if (!ctxset) SKIP("exception CONTEXT carries no MXCSR on this platform");
    } else {
        char buf[32];
        if (t_child_find(&r, "SETOK=", buf, sizeof buf)) setok = (unsigned long long)strtoll(buf, NULL, 10);
        child_hex(&r, "JOINED", &joined);
        CHECKF(joined == 1, "worker did not finish after SetThreadContext with a reserved MXCSR bit");
        REQUIREF(setok != (unsigned long long)-1, "GetThreadContext/SuspendThread failed: %s", r.out);
        t_info("set_thread_context_result", "%llu", setok);
        if (setok == 0) { t_note("SetThreadContext refused the reserved MXCSR bits"); return; }
    }
    CHECKF((mx & ~(unsigned long long)mxcsr_mask()) == 0, "thread runs with reserved MXCSR bits: 0x%08llx (mask 0x%08x)", mx, mxcsr_mask());
    t_info("applied_mxcsr", "0x%08llx (requested 0x%08x)", mx, BADMX_VAL);
}
TEST(ctx_continue_reserved_mxcsr_bits_sanitized) { run_badmx(0); }
TEST(ctx_setthreadcontext_reserved_mxcsr_bits_sanitized) { run_badmx(1); }

/* <exe> --child=rstor_bad <0|1|2>: FXRSTOR (0) / XRSTOR rfbm=3 (1) / XRSTOR rfbm=1 (2) of an image
 * whose MXCSR field has reserved bit 16 set. */
CHILD(rstor_bad) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    static T_ALIGN(64) uint8_t area[4096];
    uint32_t bad = MX_DEF | 0x10000u;
    memset(area, 0, sizeof area);
    if (mode == 0) {
        t_fxsave(area);
        memcpy(area + 24, &bad, 4);
        t_fxrstor(area);
    } else {
        if (!cpu_has(F_XSAVE) || xsave_std_size(3) > sizeof area) { printf("NOXSAVE=1\n"); return 5; }
        t_xsave(area, 3);
        memcpy(area + 24, &bad, 4);
        t_xrstor(area, mode == 1 ? 3 : 1);
    }
    unsigned mx = t_get_mxcsr();
    t_set_mxcsr(MX_DEF);
    printf("SURVIVED=0x%08x\n", mx);
    return 0;
}
TEST(fxrstor_xrstor_reserved_mxcsr_fault) {
    /* SDM FXRSTOR / XRSTOR: #GP(0) when the MXCSR image sets a reserved bit and MXCSR is being loaded
     * (FXRSTOR always; XRSTOR when RFBM[1] or RFBM[2] is set). Windows delivers the user-mode #GP of a
     * non-privileged instruction as STATUS_ACCESS_VIOLATION. XRSTOR with RFBM = x87 only does not load
     * MXCSR: recorded only. */
    static t_child_result r;
    REQUIRE(t_child_run("rstor_bad", "0", 10000, &r));
    CHECKF(!r.timed_out, "child timed out");
    CHECKF(strstr(r.out, "SURVIVED=") == NULL, "FXRSTOR of MXCSR 0x%08x did not fault: %s", MX_DEF | 0x10000u, r.out);
    CHECKF(r.exc_code == STATUS_ACCESS_VIOLATION || r.exc_code == STATUS_PRIVILEGED_INSTRUCTION,
           "FXRSTOR reserved MXCSR: exception 0x%08lx", r.exc_code);
    if (!cpu_has(F_XSAVE)) { t_note("XSAVE not usable: XRSTOR part skipped"); return; }
    if (t_sde_model())   /* emulator artifact, README "Emulated validation" */
        t_expect_fail("Intel SDE (%s): XRSTOR emulation has no #GP for a reserved MXCSR bit", t_sde_model());
    REQUIRE(t_child_run("rstor_bad", "1", 10000, &r));
    CHECKF(!r.timed_out, "child timed out");
    CHECKF(strstr(r.out, "SURVIVED=") == NULL && strstr(r.out, "NOXSAVE") == NULL, "XRSTOR rfbm=3 of a reserved MXCSR did not fault: %s", r.out);
    CHECKF(r.exc_code == STATUS_ACCESS_VIOLATION || r.exc_code == STATUS_PRIVILEGED_INSTRUCTION,
           "XRSTOR reserved MXCSR: exception 0x%08lx", r.exc_code);
    REQUIRE(t_child_run("rstor_bad", "2", 10000, &r));
    char v[32] = "";
    t_info("xrstor_x87_only_reserved_mxcsr", "%s", t_child_find(&r, "SURVIVED=", v, sizeof v) ? "no fault" : "faulted");
}

TEST(env_preserved_across_cpu_migration) {
    /* A spinning worker holds a non-default MXCSR (flags included), FCW and a non-empty x87 stack
     * with FSW.ZE while its affinity is moved to every CPU of the process in turn: the FP state is
     * thread state and must follow the thread to each processor. */
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) || !pm) SKIP("GetProcessAffinityMask failed");
    int ncpu = 0;
    for (unsigned b = 0; b < 8 * sizeof(DWORD_PTR); b++) if (pm & ((DWORD_PTR)1 << b)) ncpu++;
    if (ncpu < 2) SKIP("process can run on one CPU only");
    volatile long stop = 0, ready = 0;
    static spin_args a;
    memset(&a, 0, sizeof a);
    a.mx_in = MX_DEF | MX_RC(RD) | MX_FTZ | (daz_supported() ? MX_DAZ : 0) | MXF_PE | MXF_DE;
    a.cw_in = (unsigned short)(0x007F | CW_RC(RU) | CW_PC53);
    a.stop = &stop; a.ready = &ready; a.max = 0x7FFFFFFFul;
    HANDLE th = CreateThread(NULL, 64 * 1024, th_env_spin, &a, 0, NULL);
    if (!th) SKIP("CreateThread failed");
    int moves = 0, fails = 0, spun = wait_count(&ready, 1, 5000);
    if (spun)
        for (int pass = 0; pass < 2; pass++)
            for (unsigned b = 0; b < 8 * sizeof(DWORD_PTR); b++) {
                if (!(pm & ((DWORD_PTR)1 << b))) continue;
                if (SetThreadAffinityMask(th, (DWORD_PTR)1 << b)) moves++; else fails++;
                Sleep(2);
            }
    stop = 1;
    DWORD w = WaitForSingleObject(th, 5000);
    CloseHandle(th);
    REQUIREF(spun, "worker never reached the spin");
    REQUIREF(w == WAIT_OBJECT_0, "worker did not stop");
    CHECKF(moves > 0, "SetThreadAffinityMask never succeeded (%d failures)", fails);
    CHECKF(a.mx_out == a.mx_in, "MXCSR after migration 0x%04x want 0x%04x", a.mx_out, a.mx_in);
    CHECKF((a.cw_out & CW_CMP) == (a.cw_in & CW_CMP), "FCW after migration 0x%04x want 0x%04x", a.cw_out, a.cw_in);
    CHECKF((a.sw_out & 0x38FFu) == 0x3004u, "FSW after migration 0x%04x want TOP=6 + ZE", a.sw_out);
    t_info("affinity_moves", "%d over %d CPUs (%d refused)", moves, ncpu, fails);
}
