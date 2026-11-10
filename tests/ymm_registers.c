/* ymm_registers - YMM register file conformance: lower/upper 128-bit lanes, VEX.128 vs legacy SSE
 * upper-lane semantics, VZEROUPPER/VZEROALL, lane-crossing and in-lane permutes, XSAVE/XRSTOR of the
 * AVX component, and preservation of the full 256-bit state across OS-mediated events (preemption,
 * suspension, thread-context reads/writes, migration, special APCs, exception continuation), plus the
 * Windows calling-convention contract around calls (only the low 128 bits of XMM6-15 survive on x64).
 *
 * Contract under test:
 *   - Intel SDM Vol.1 14.1.1 / 14.3, AMD APM Vol.4: VEX.128 instructions with an XMM destination zero
 *     bits MAXVL-1:128 of the destination; legacy SSE instructions leave bits MAXVL-1:128 unmodified.
 *     VEX.256 VEXTRACTF128/VEXTRACTI128 with a register destination zero bits MAXVL-1:128 too.
 *   - VZEROUPPER zeroes bits MAXVL-1:128 of YMM0-15 (YMM0-7 outside 64-bit mode), VZEROALL zeroes the
 *     registers; ZMM16-31 are not modified by either.
 *   - VEX-encoded instructions without explicit alignment semantics take unaligned memory operands
 *     (SDM 14.9); VMOVDQA/VMOVAPS/VMOVNTDQ fault (#GP) on misaligned operands.
 *   - A VEX instruction is #UD when XCR0[2:1] != 11b (SDM 13.3 / 14.3). AVX eligibility = CPUID.1:ECX.AVX
 *     + OSXSAVE + XCR0[2:1]=11b; without it every case here SKIPs.
 *   - XINUSE[i]=0 implies component i is in its initial configuration (SDM 13.6), so with a non-zero
 *     upper lane XGETBV(1)[2] must be 1 and XSAVE must set XSTATE_BV[2]; XRSTOR with RFBM[2]=1 and
 *     XSTATE_BV[2]=0 initialises the upper lanes to zero; RFBM[2]=0 leaves them alone.
 *   - Windows: an involuntary context switch, SuspendThread/ResumeThread, GetThreadContext, a special
 *     user APC and exception continuation (VEH returning EXCEPTION_CONTINUE_EXECUTION) all resume the
 *     thread with its full register state. Across a *call* the x64 ABI guarantees only XMM6-15[127:0]
 *     (upper YMM halves are volatile); on x86 every XMM/YMM register is volatile: those cases record
 *     upper-lane preservation only as INFO.
 *   - XSTATE context APIs (checked against the Windows 10 19041 kernelbase decomp):
 *       LocateXStateFeature (kernelbase!0xe8590): x64 CONTEXT (flag 0x100000): feature 0 -> FltSave
 *         (+0x100, len 0xA0), feature 1 -> XmmRegisters (+0x1A0, len 0x100); x86 CONTEXT (0x10000):
 *         feature 0 at +0xCC, feature 1 at +0x16C len 0x80; features >= 2 only when CONTEXT_XSTATE is
 *         set, through RtlLocateExtendedFeature2 (only features enabled in the system XState mask).
 *       SetXStateFeaturesMask (kernelbase!0xe8660) -> RtlSetExtendedFeaturesMask stores
 *         mask & enabled & ~3; GetXStateFeaturesMask (kernelbase!0xe8370) returns 3|extended mask.
 *   - A thread whose AVX component is initial (after VZEROUPPER/VZEROALL) must resume with zero upper
 *     lanes after any context switch / suspension / context get-set; never with another thread's data.
 *   - x86 builds: on 64-bit Windows exceptions and APCs reach 32-bit code through WoW64, whose context
 *     conversion is not documented to carry CONTEXT_XSTATE, so upper-lane survival there is asserted only
 *     when the OS-provided record carries XSTATE (else NOTE); WoW64 XSTATE GetThreadContext gaps -> SKIP.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> NOTE/INFO.
 * Deliberately faulting operations run in CHILD() processes via t_child_run(). See common/API.md.
 *
 * ISA rule: every function that contains a VEX/EVEX instruction has "ymm"/"avx"/"512" in its name and
 * is only reached after ygate()/ygate2(). The YMM register image used here is yfile (32-byte stride). */
#include "t.h"
#include <math.h>

#ifndef CONTEXT_XSTATE
#ifdef T_X64
#define CONTEXT_XSTATE (CONTEXT_AMD64 | 0x40L)
#else
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)
#endif
#endif
#ifndef XSTATE_LEGACY_SSE
#define XSTATE_LEGACY_SSE 1
#endif
#ifndef XSTATE_AVX
#define XSTATE_AVX 2
#endif

/* ======================================================================= */
/* register-file plumbing                                                   */
/* ======================================================================= */
#ifdef T_X64
#define NY 16
#define YR "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#define YCLOB "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7", \
              "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15"
#define GPR_A "rax"
#define GPR_C "rcx"
#define GPR_D "rdx"
#else
#define NY 8
#define YR "0,1,2,3,4,5,6,7"
#define YCLOB "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7"
#define GPR_A "eax"
#define GPR_C "ecx"
#define GPR_D "edx"
#endif

/* 16 registers x 32 bytes; only the first NY are used by the asm. */
typedef struct yfile { T_ALIGN(64) uint8_t v[16][32]; } yfile;

#define YLOAD(p)  ".irp i," YR "\n\tvmovdqu \\i*32(" p "), %%ymm\\i\n.endr\n"
#define YSTORE(p) ".irp i," YR "\n\tvmovdqu %%ymm\\i, \\i*32(" p ")\n.endr\n"

static void ypat(yfile *f, uint32_t tag) {
    memset(f, 0, sizeof *f);
    for (int i = 0; i < 16; i++) t_pattern_reg(f->v[i], 32, tag, (uint32_t)i);
}
static uint32_t rd32(const uint8_t *p, int i) { uint32_t v; memcpy(&v, p + 4 * i, 4); return v; }
static void     wr32(uint8_t *p, int i, uint32_t v) { memcpy(p + 4 * i, &v, 4); }
static uint64_t rd64(const uint8_t *p, int i) { uint64_t v; memcpy(&v, p + 8 * i, 8); return v; }
static void     wr64(uint8_t *p, int i, uint64_t v) { memcpy(p + 8 * i, &v, 8); }
static uint16_t rd16(const uint8_t *p, int i) { uint16_t v; memcpy(&v, p + 2 * i, 2); return v; }
static void     wr16(uint8_t *p, int i, uint16_t v) { memcpy(p + 2 * i, &v, 2); }

static const uint8_t g_zero32[32];
static const yfile g_zfile;

/* CHECK_REG on [off, off+len) of every used register. */
static int ycheck_part(const yfile *got, const yfile *want, int off, int len, int first, int last, const char *label) {
    int bad = 0;
    if (last < 0 || last >= NY) last = NY - 1;
    for (int i = first; i <= last; i++) {
        char what[192];
        snprintf(what, sizeof what, "ymm%d%s %s", i, off == 0 && len == 32 ? "" : off == 0 ? "[127:0]" : "[255:128]", label);
        bad += !CHECK_REG(what, got->v[i] + off, want->v[i] + off, len);
    }
    return bad;
}
static int ycheck(const yfile *got, const yfile *want, const char *label) { return ycheck_part(got, want, 0, 32, 0, -1, label); }
static int ycheck_one(const uint8_t *got, const uint8_t *want, const char *fmt, ...) T_PRINTF(3, 4);
static int ycheck_one(const uint8_t *got, const uint8_t *want, const char *fmt, ...) {
    char what[192]; va_list ap; va_start(ap, fmt); vsnprintf(what, sizeof what, fmt, ap); va_end(ap);
    return CHECK_REG(what, got, want, 32);
}

static void ygate(void) {
    if (!cpu_has(F_AVX)) {
        const t_cpu_info *c = cpu();
        SKIP("AVX not usable: cpuid.avx=%d osxsave=%d xcr0=0x%llx (needs XCR0[2:1]=11b)", cpu_hw(F_AVX), c->osxsave,
             (unsigned long long)(c->xcr0_valid ? c->xcr0 : 0));
    }
}
static void ygate2(void) { ygate(); if (!cpu_has(F_AVX2)) SKIP("AVX2 not usable (cpuid.avx2=%d)", cpu_hw(F_AVX2)); }

/* ---- basic asm primitives ---------------------------------------------- */
static T_NOINLINE void ymm_rt_vmovdqu(const void *in, void *out) {
    __asm__ volatile(YLOAD("%[in]") YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB);
}
static T_NOINLINE void ymm_rt_vmovdqa(const void *in, void *out) {
    __asm__ volatile(".irp i," YR "\n\tvmovdqa \\i*32(%[in]), %%ymm\\i\n.endr\n"
                     ".irp i," YR "\n\tvmovdqa %%ymm\\i, \\i*32(%[out])\n.endr\nvzeroupper\n"
                     :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB);
}
#define DEF_RT(fname, ld, st) \
static T_NOINLINE void fname(const void *in, void *out) { \
    __asm__ volatile(".irp i," YR "\n\t" ld " \\i*32(%[in]), %%ymm\\i\n.endr\n" \
                     ".irp i," YR "\n\t" st " %%ymm\\i, \\i*32(%[out])\n.endr\nvzeroupper\n" \
                     :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB); }
DEF_RT(ymm_rt_ups_aps, "vmovups", "vmovaps")
DEF_RT(ymm_rt_apd_upd, "vmovapd", "vmovupd")
DEF_RT(ymm_rt_lddqu_dqa, "vlddqu", "vmovdqa")
DEF_RT(ymm_rt_aps_ntdq, "vmovaps", "vmovntdq")

/* Store the current registers (no load). */
static T_NOINLINE void ymm_store_all(void *out) {
    __asm__ volatile(YSTORE("%[out]") :: [out]"r"(out) : "memory");
}
/* Load registers and return WITHOUT vzeroupper: clobbers everything the ABI lets a callee clobber
 * (GCC restores XMM6-15[127:0] on x64 because they are in the clobber list). */
static T_NOINLINE void ymm_clobber_all(const void *in) {
    __asm__ volatile(YLOAD("%[in]") :: [in]"r"(in) : "memory", YCLOB);
}

/* rotate: ymm[i] <- ymm[i+1], ymm[n-1] <- in[0] */
#define MV(a, b) "vmovdqa %%ymm" #a ", %%ymm" #b "\n"
static T_NOINLINE void ymm_rotate(const void *in, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        MV(1,0) MV(2,1) MV(3,2) MV(4,3) MV(5,4) MV(6,5) MV(7,6)
#ifdef T_X64
        MV(8,7) MV(9,8) MV(10,9) MV(11,10) MV(12,11) MV(13,12) MV(14,13) MV(15,14)
        "vmovdqu (%[in]), %%ymm15\n"
#else
        "vmovdqu (%[in]), %%ymm7\n"
#endif
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB);
}

/* Per-register op: for every j: load all, apply op to register j, store all to out[j]. */
#define DEF_PERREG(fname, op) \
static T_NOINLINE void fname(const yfile *in, yfile *out) { \
    __asm__ volatile(".irp j," YR "\n" YLOAD("%[in]") op "\n" \
                     ".irp i," YR "\n\tvmovdqu %%ymm\\i, \\i*32+\\j*512(%[out])\n.endr\n.endr\nvzeroupper\n" \
                     :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB); }
DEF_PERREG(ymm_perreg_vxorps256, "vxorps %%ymm\\j, %%ymm\\j, %%ymm\\j")
DEF_PERREG(ymm_perreg_vmovdqa128, "vmovdqa %%xmm\\j, %%xmm\\j")
DEF_PERREG(ymm_perreg_vpxor128, "vpxor %%xmm\\j, %%xmm\\j, %%xmm\\j")
DEF_PERREG(ymm_perreg_vpcmpeqd128, "vpcmpeqd %%xmm\\j, %%xmm\\j, %%xmm\\j")
DEF_PERREG(ymm_perreg_legacy_movdqa, "movdqa %%xmm\\j, %%xmm\\j")
DEF_PERREG(ymm_perreg_legacy_pxor, "pxor %%xmm\\j, %%xmm\\j")
DEF_PERREG(ymm_perreg_legacy_pcmpeqd, "pcmpeqd %%xmm\\j, %%xmm\\j")
DEF_PERREG(ymm_perreg_vextractf128_hi, "vextractf128 $1, %%ymm\\j, %%xmm\\j")

static T_NOINLINE void ymm_vzeroupper_rt(const void *in, void *out) {
    __asm__ volatile(YLOAD("%[in]") "vzeroupper\n" YSTORE("%[out]") :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB);
}
static T_NOINLINE void ymm_vzeroall_rt(const void *in, void *out) {
    __asm__ volatile(YLOAD("%[in]") "vzeroall\n" YSTORE("%[out]") :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB);
}
/* vzeroupper, then legacy SSE writes to x0..x3 and VEX.256 writes to ymm4: upper of 0..3 stays zero */
static T_NOINLINE void ymm_vzu_then_legacy(const void *in, const void *m, void *out) {
    __asm__ volatile(YLOAD("%[in]") "vzeroupper\n"
                     "movdqu (%[m]), %%xmm0\n paddd %%xmm0, %%xmm1\n pxor %%xmm2, %%xmm2\n movss 16(%[m]), %%xmm3\n"
                     "vmovdqu 32(%[m]), %%ymm4\n"
                     YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [m]"r"(m), [out]"r"(out) : "memory", YCLOB);
}

/* ======================================================================= */
/* memory operand used by the op lists (aligned 64)                        */
/* ======================================================================= */
typedef struct ymem { T_ALIGN(64) uint8_t b[256]; } ymem;
static void ymem_init(ymem *m) {
    for (int i = 0; i < 32; i++) wr64(m->b, i, t_pattern_word(0x3E3u, 40, (uint32_t)i));
    wr32(m->b, 0, (uint32_t)-123456);   /* cvtsi2sd source (exact) */
    wr32(m->b, 1, 1000);                /* cvtsi2ss source (exact) */
}

/* ---- op list A: data movement / arithmetic on xmm0-7. Legacy and VEX variants run the same steps
 * in the same order (see sim_listA). ---- */
static T_NOINLINE void ymm_legacy_listA(const void *in, const void *m, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "movdqa %%xmm1, %%xmm0\n"
        "movss %%xmm1, %%xmm7\n"
        "shufps $0x1B, %%xmm1, %%xmm1\n"
        "paddd %%xmm3, %%xmm2\n"
        "psllq $8, %%xmm2\n"
        "movq 32(%[m]), %%xmm3\n"
        "movd 0(%[m]), %%xmm5\n"
        "movhps 48(%[m]), %%xmm5\n"
        "movss 16(%[m]), %%xmm6\n"
        "por %%xmm4, %%xmm6\n"
        "pinsrw $5, 8(%[m]), %%xmm7\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [m]"r"(m), [out]"r"(out) : "memory", YCLOB);
}
static T_NOINLINE void ymm_vex128_listA(const void *in, const void *m, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "vmovdqa %%xmm1, %%xmm0\n"
        "vmovss %%xmm1, %%xmm7, %%xmm7\n"
        "vshufps $0x1B, %%xmm1, %%xmm1, %%xmm1\n"
        "vpaddd %%xmm3, %%xmm2, %%xmm2\n"
        "vpsllq $8, %%xmm2, %%xmm2\n"
        "vmovq 32(%[m]), %%xmm3\n"
        "vmovd 0(%[m]), %%xmm5\n"
        "vmovhps 48(%[m]), %%xmm5, %%xmm5\n"
        "vmovss 16(%[m]), %%xmm6\n"
        "vpor %%xmm4, %%xmm6, %%xmm6\n"
        "vpinsrw $5, 8(%[m]), %%xmm7, %%xmm7\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [m]"r"(m), [out]"r"(out) : "memory", YCLOB);
}
static void hz(yfile *s, int r, int vex) { if (vex) memset(s->v[r] + 16, 0, 16); }
static void sim_listA(yfile *s, const ymem *m, int vex) {
    memcpy(s->v[0], s->v[1], 16); hz(s, 0, vex);
    wr32(s->v[7], 0, rd32(s->v[1], 0)); hz(s, 7, vex);
    { uint32_t t[4]; for (int k = 0; k < 4; k++) t[k] = rd32(s->v[1], 3 - k); for (int k = 0; k < 4; k++) wr32(s->v[1], k, t[k]); hz(s, 1, vex); }
    { for (int k = 0; k < 4; k++) wr32(s->v[2], k, rd32(s->v[2], k) + rd32(s->v[3], k)); } hz(s, 2, vex);
    { for (int k = 0; k < 2; k++) wr64(s->v[2], k, rd64(s->v[2], k) << 8); } hz(s, 2, vex);
    wr64(s->v[3], 0, rd64(m->b, 4)); wr64(s->v[3], 1, 0); hz(s, 3, vex);
    wr32(s->v[5], 0, rd32(m->b, 0)); wr32(s->v[5], 1, 0); wr64(s->v[5], 1, 0); hz(s, 5, vex);
    wr64(s->v[5], 1, rd64(m->b, 6)); hz(s, 5, vex);
    wr32(s->v[6], 0, rd32(m->b, 4)); wr32(s->v[6], 1, 0); wr64(s->v[6], 1, 0); hz(s, 6, vex);
    { for (int k = 0; k < 2; k++) wr64(s->v[6], k, rd64(s->v[6], k) | rd64(s->v[4], k)); } hz(s, 6, vex);
    wr16(s->v[7], 5, rd16(m->b, 4)); hz(s, 7, vex);
}

/* ---- op list B: zero/ones idioms, int->fp conversions (merge), logic with memory, shuffles ---- */
static T_NOINLINE void ymm_legacy_listB(const void *in, const void *m, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "pxor %%xmm0, %%xmm0\n"
        "pcmpeqd %%xmm1, %%xmm1\n"
        "xorps %%xmm2, %%xmm2\n"
        "psubd %%xmm3, %%xmm3\n"
        "cvtsi2sdl 0(%[m]), %%xmm4\n"
        "cvtsi2ssl 4(%[m]), %%xmm5\n"
        "andps 64(%[m]), %%xmm6\n"
        "pshufd $0x4E, %%xmm7, %%xmm7\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [m]"r"(m), [out]"r"(out) : "memory", YCLOB);
}
static T_NOINLINE void ymm_vex128_listB(const void *in, const void *m, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "vpxor %%xmm0, %%xmm0, %%xmm0\n"
        "vpcmpeqd %%xmm1, %%xmm1, %%xmm1\n"
        "vxorps %%xmm2, %%xmm2, %%xmm2\n"
        "vpsubd %%xmm3, %%xmm3, %%xmm3\n"
        "vcvtsi2sdl 0(%[m]), %%xmm4, %%xmm4\n"
        "vcvtsi2ssl 4(%[m]), %%xmm5, %%xmm5\n"
        "vandps 64(%[m]), %%xmm6, %%xmm6\n"
        "vpshufd $0x4E, %%xmm7, %%xmm7\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [m]"r"(m), [out]"r"(out) : "memory", YCLOB);
}
static void sim_listB(yfile *s, const ymem *m, int vex) {
    memset(s->v[0], 0, 16); hz(s, 0, vex);
    memset(s->v[1], 0xFF, 16); hz(s, 1, vex);
    memset(s->v[2], 0, 16); hz(s, 2, vex);
    memset(s->v[3], 0, 16); hz(s, 3, vex);
    { double d = (double)(int32_t)rd32(m->b, 0); uint64_t q; memcpy(&q, &d, 8); wr64(s->v[4], 0, q); hz(s, 4, vex); }
    { float f = (float)(int32_t)rd32(m->b, 1); uint32_t q; memcpy(&q, &f, 4); wr32(s->v[5], 0, q); hz(s, 5, vex); }
    { for (int k = 0; k < 2; k++) wr64(s->v[6], k, rd64(s->v[6], k) & rd64(m->b + 64, k)); } hz(s, 6, vex);
    { uint64_t a = rd64(s->v[7], 0), b = rd64(s->v[7], 1); wr64(s->v[7], 0, b); wr64(s->v[7], 1, a); hz(s, 7, vex); }
}

/* ---- loads into xmm0-7 ---- */
static T_NOINLINE void ymm_legacy_loads(const void *in, const void *m, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "movdqu 0(%[m]), %%xmm0\n movaps 16(%[m]), %%xmm1\n movups 32(%[m]), %%xmm2\n movss 48(%[m]), %%xmm3\n"
        "movsd 56(%[m]), %%xmm4\n movlps 64(%[m]), %%xmm5\n movhpd 72(%[m]), %%xmm6\n movapd 80(%[m]), %%xmm7\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [m]"r"(m), [out]"r"(out) : "memory", YCLOB);
}
static T_NOINLINE void ymm_vex128_loads(const void *in, const void *m, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "vmovdqu 0(%[m]), %%xmm0\n vmovaps 16(%[m]), %%xmm1\n vmovups 32(%[m]), %%xmm2\n vmovss 48(%[m]), %%xmm3\n"
        "vmovsd 56(%[m]), %%xmm4\n vmovlps 64(%[m]), %%xmm5, %%xmm5\n vmovhpd 72(%[m]), %%xmm6, %%xmm6\n vmovapd 80(%[m]), %%xmm7\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [m]"r"(m), [out]"r"(out) : "memory", YCLOB);
}
static void sim_loads(yfile *s, const ymem *m, int vex) {
    memcpy(s->v[0], m->b + 0, 16); hz(s, 0, vex);
    memcpy(s->v[1], m->b + 16, 16); hz(s, 1, vex);
    memcpy(s->v[2], m->b + 32, 16); hz(s, 2, vex);
    memset(s->v[3], 0, 16); memcpy(s->v[3], m->b + 48, 4); hz(s, 3, vex);
    memset(s->v[4], 0, 16); memcpy(s->v[4], m->b + 56, 8); hz(s, 4, vex);
    memcpy(s->v[5], m->b + 64, 8); hz(s, 5, vex);
    memcpy(s->v[6] + 8, m->b + 72, 8); hz(s, 6, vex);
    memcpy(s->v[7], m->b + 80, 16); hz(s, 7, vex);
}

/* ---- GPR -> XMM moves ---- */
static T_NOINLINE void ymm_gpr_moves(const void *in, uint32_t g, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "vmovd %[g], %%xmm1\n"
        "movd %[g], %%xmm2\n"
        "vpinsrd $2, %[g], %%xmm3, %%xmm3\n"
        "pinsrw $3, %[g], %%xmm4\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [g]"r"(g), [out]"r"(out) : "memory", YCLOB);
}
#ifdef T_X64
static T_NOINLINE void ymm_gpr_moves64(const void *in, uint64_t g, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "vmovq %[g], %%xmm9\n movq %[g], %%xmm10\n vpinsrq $1, %[g], %%xmm11, %%xmm11\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [g]"r"(g), [out]"r"(out) : "memory", YCLOB);
}
/* x64-only: VEX/legacy ops on xmm8-15 (VEX.R/X/B + VEX.vvvv high bit, REX for legacy). */
static T_NOINLINE void ymm_x64_ext_list(const void *in, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "vpaddd %%xmm15, %%xmm8, %%xmm12\n"
        "vmovdqa %%xmm9, %%xmm13\n"
        "paddq %%xmm10, %%xmm14\n"
        "pxor %%xmm11, %%xmm11\n"
        "vpxor %%xmm1, %%xmm9, %%xmm8\n"
        "movdqa %%xmm15, %%xmm2\n"
        "vinsertf128 $1, %%xmm10, %%ymm15, %%ymm10\n"
        "vextractf128 $1, %%ymm14, %%xmm15\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [out]"r"(out) : "memory", YCLOB);
}
#endif

/* ---- stores: exact byte footprint, registers unchanged ---- */
static T_NOINLINE void ymm_store_forms(const void *in, void *o, void *out) {
    __asm__ volatile(YLOAD("%[in]")
        "vmovdqu %%xmm1, 0(%[o])\n vmovups %%ymm2, 32(%[o])\n vextractf128 $1, %%ymm3, 64(%[o])\n"
        "vmovss %%xmm4, 80(%[o])\n vmovsd %%xmm5, 88(%[o])\n vmovhps %%xmm6, 96(%[o])\n movdqu %%xmm7, 112(%[o])\n"
        "vextractf128 $0, %%ymm0, 128(%[o])\n"
        YSTORE("%[out]") "vzeroupper\n" :: [in]"r"(in), [o]"r"(o), [out]"r"(out) : "memory", YCLOB);
}

/* ======================================================================= */
/* lane operations (two inputs a -> ymm0, b -> ymm1)                        */
/* ======================================================================= */
#define DEF_IMM(fname, count, insn) \
static T_NOINLINE void fname(const void *a, const void *b, void *o) { \
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n .set .Lyk, 0\n .rept " #count "\n" \
                     insn "\n vmovdqu %%ymm2, (%[o])\n add $32, %[o]\n .set .Lyk, .Lyk+1\n .endr\n vzeroupper\n" \
                     : [o]"+r"(o) : [a]"r"(a), [b]"r"(b) : "memory", "cc", "xmm0", "xmm1", "xmm2"); }
DEF_IMM(ymm_all_vperm2f128, 256, "vperm2f128 $.Lyk, %%ymm1, %%ymm0, %%ymm2")
DEF_IMM(ymm_all_vpermilps, 256, "vpermilps $.Lyk, %%ymm0, %%ymm2")
DEF_IMM(ymm_all_vshufps, 256, "vshufps $.Lyk, %%ymm1, %%ymm0, %%ymm2")
DEF_IMM(ymm_all_vblendps, 256, "vblendps $.Lyk, %%ymm1, %%ymm0, %%ymm2")
DEF_IMM(ymm_all_vpermilpd, 16, "vpermilpd $.Lyk, %%ymm0, %%ymm2")
DEF_IMM(ymm_all_vblendpd, 16, "vblendpd $.Lyk, %%ymm1, %%ymm0, %%ymm2")
DEF_IMM(avx2_ymm_all_vperm2i128, 256, "vperm2i128 $.Lyk, %%ymm1, %%ymm0, %%ymm2")
DEF_IMM(avx2_ymm_all_vpermq, 256, "vpermq $.Lyk, %%ymm0, %%ymm2")
DEF_IMM(avx2_ymm_all_vpshufd, 256, "vpshufd $.Lyk, %%ymm0, %%ymm2")
DEF_IMM(avx2_ymm_all_vpblendd, 256, "vpblendd $.Lyk, %%ymm1, %%ymm0, %%ymm2")
DEF_IMM(avx2_ymm_all_vpalignr, 32, "vpalignr $.Lyk, %%ymm1, %%ymm0, %%ymm2")
DEF_IMM(avx2_ymm_all_vpslldq, 17, "vpslldq $.Lyk, %%ymm0, %%ymm2")
DEF_IMM(avx2_ymm_all_vpsrldq, 17, "vpsrldq $.Lyk, %%ymm0, %%ymm2")

/* insert/extract: out[0..7], ymm7 preloaded dirty with c */
static T_NOINLINE void ymm_insext_f128(const void *a, const void *b, const void *c, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n vmovdqu (%[c]), %%ymm7\n"
        "vinsertf128 $1, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 0(%[o])\n"
        "vinsertf128 $0, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n"
        "vinsertf128 $3, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 64(%[o])\n"
        "vinsertf128 $2, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 96(%[o])\n"
        "vinsertf128 $1, 16(%[b]), %%ymm0, %%ymm2\n vmovdqu %%ymm2, 128(%[o])\n"
        "vextractf128 $1, %%ymm0, %%xmm7\n vmovdqu %%ymm7, 160(%[o])\n"
        "vmovdqu (%[c]), %%ymm7\n vextractf128 $0, %%ymm1, %%xmm7\n vmovdqu %%ymm7, 192(%[o])\n"
        "vextractf128 $3, %%ymm0, 224(%[o])\n"
        "vzeroupper\n" :: [a]"r"(a), [b]"r"(b), [c]"r"(c), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2", "xmm7");
}
static T_NOINLINE void avx2_ymm_insext_i128(const void *a, const void *b, const void *c, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n vmovdqu (%[c]), %%ymm7\n"
        "vinserti128 $1, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 0(%[o])\n"
        "vinserti128 $0, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n"
        "vinserti128 $3, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 64(%[o])\n"
        "vinserti128 $2, %%xmm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 96(%[o])\n"
        "vinserti128 $1, 16(%[b]), %%ymm0, %%ymm2\n vmovdqu %%ymm2, 128(%[o])\n"
        "vextracti128 $1, %%ymm0, %%xmm7\n vmovdqu %%ymm7, 160(%[o])\n"
        "vmovdqu (%[c]), %%ymm7\n vextracti128 $0, %%ymm1, %%xmm7\n vmovdqu %%ymm7, 192(%[o])\n"
        "vextracti128 $3, %%ymm0, 224(%[o])\n"
        "vzeroupper\n" :: [a]"r"(a), [b]"r"(b), [c]"r"(c), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2", "xmm7");
}

/* broadcasts from m (and from xmm4 = m[0..15] for the AVX2 register forms); c = dirty preload */
static T_NOINLINE void ymm_bcast_avx(const void *m, const void *c, void *o) {
    __asm__ volatile("vmovdqu (%[c]), %%ymm0\n vmovdqu (%[c]), %%ymm1\n vmovdqu (%[c]), %%ymm2\n vmovdqu (%[c]), %%ymm3\n"
        "vbroadcastss (%[m]), %%ymm0\n vbroadcastsd 8(%[m]), %%ymm1\n vbroadcastf128 16(%[m]), %%ymm2\n vbroadcastss 4(%[m]), %%xmm3\n"
        "vmovdqu %%ymm0, 0(%[o])\n vmovdqu %%ymm1, 32(%[o])\n vmovdqu %%ymm2, 64(%[o])\n vmovdqu %%ymm3, 96(%[o])\n vzeroupper\n"
        :: [m]"r"(m), [c]"r"(c), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2", "xmm3");
}
static T_NOINLINE void avx2_ymm_bcast(const void *m, const void *c, void *o) {
    __asm__ volatile(".irp i,0,1,2,3,5,6,7\n vmovdqu (%[c]), %%ymm\\i\n .endr\n vmovdqu (%[m]), %%xmm4\n"
        "vbroadcasti128 16(%[m]), %%ymm0\n vpbroadcastb 1(%[m]), %%ymm1\n vpbroadcastw 2(%[m]), %%ymm2\n"
        "vpbroadcastq %%xmm4, %%ymm3\n vbroadcastss %%xmm4, %%ymm5\n vpbroadcastd 12(%[m]), %%xmm6\n vbroadcastsd %%xmm4, %%ymm7\n"
        ".irp i,0,1,2,3,5,6,7\n vmovdqu %%ymm\\i, \\i*32(%[o])\n .endr\n vzeroupper\n"
        :: [m]"r"(m), [c]"r"(c), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");
}

/* bitwise (AVX): o[0]=a^b o[1]=a&b o[2]=a|b o[3]=~a&b */
static T_NOINLINE void ymm_bitops(const void *a, const void *b, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n"
        "vxorps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 0(%[o])\n"
        "vandps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n"
        "vorps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 64(%[o])\n"
        "vandnps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 96(%[o])\n vzeroupper\n"
        :: [a]"r"(a), [b]"r"(b), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2");
}
/* fp (AVX, exact): o0 = vaddps a,b; o1 = vmulpd a,b; o2 = vsqrtpd a; o3 = vdivps a,b; o4 = vcmpps eq; o5 = vminpd */
static T_NOINLINE void ymm_fpops(const void *a, const void *b, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n"
        "vaddps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 0(%[o])\n"
        "vmulpd %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n"
        "vsqrtpd %%ymm0, %%ymm2\n vmovdqu %%ymm2, 64(%[o])\n"
        "vdivps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 96(%[o])\n"
        "vcmpeqps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 128(%[o])\n"
        "vminpd %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 160(%[o])\n vzeroupper\n"
        :: [a]"r"(a), [b]"r"(b), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2");
}
/* integer (AVX2): o0 paddb o1 paddw o2 paddd o3 paddq o4 psubq o5 pcmpeqd o6 paddusb o7 pmaxsd */
static T_NOINLINE void avx2_ymm_intops(const void *a, const void *b, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n"
        "vpaddb %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 0(%[o])\n"
        "vpaddw %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n"
        "vpaddd %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 64(%[o])\n"
        "vpaddq %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 96(%[o])\n"
        "vpsubq %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 128(%[o])\n"
        "vpcmpeqd %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 160(%[o])\n"
        "vpaddusb %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 192(%[o])\n"
        "vpmaxsd %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 224(%[o])\n vzeroupper\n"
        :: [a]"r"(a), [b]"r"(b), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2");
}
/* unpack (AVX: ps, AVX2: qdq): o0 unpcklps o1 unpckhps o2 unpcklpd o3 unpckhpd */
static T_NOINLINE void ymm_unpack(const void *a, const void *b, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n"
        "vunpcklps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 0(%[o])\n"
        "vunpckhps %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n"
        "vunpcklpd %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 64(%[o])\n"
        "vunpckhpd %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, 96(%[o])\n vzeroupper\n"
        :: [a]"r"(a), [b]"r"(b), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2");
}
static T_NOINLINE void avx2_ymm_vpshufb(const void *data, const void *idx, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n vpshufb %%ymm1, %%ymm0, %%ymm2\n vmovdqu %%ymm2, (%[o])\n vzeroupper\n"
        :: [a]"r"(data), [b]"r"(idx), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2");
}
static T_NOINLINE void avx2_ymm_vpermd(const void *data, const void *idx, void *o) {
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n vpermd %%ymm0, %%ymm1, %%ymm2\n vmovdqu %%ymm2, (%[o])\n"
                     "vpermps %%ymm0, %%ymm1, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n vzeroupper\n"
        :: [a]"r"(data), [b]"r"(idx), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2");
}
/* flags: r[0]=ZF(vptest a,b) r[1]=CF(vptest) r[2]=vmovmskps a r[3]=vmovmskpd a r[4]=ZF(vtestps) r[5]=CF(vtestps) */
static T_NOINLINE void ymm_test_mask(const void *a, const void *b, uint32_t *r) {
    uint8_t z, c, z2, c2; uint32_t m1, m2;
    __asm__ volatile("vmovdqu (%[a]), %%ymm0\n vmovdqu (%[b]), %%ymm1\n"
        "vptest %%ymm1, %%ymm0\n setz %[z]\n setc %[c]\n"
        "vtestps %%ymm1, %%ymm0\n setz %[z2]\n setc %[c2]\n"
        "vmovmskps %%ymm0, %[m1]\n vmovmskpd %%ymm0, %[m2]\n vzeroupper\n"
        : [z]"=m"(z), [c]"=m"(c), [z2]"=m"(z2), [c2]"=m"(c2), [m1]"=&r"(m1), [m2]"=&r"(m2)
        : [a]"r"(a), [b]"r"(b) : "memory", "cc", "xmm0", "xmm1");
    r[0] = z; r[1] = c; r[2] = m1; r[3] = m2; r[4] = z2; r[5] = c2;
}

/* VEX.256 -> legacy SSE -> VEX.256 data chain (AVX1 only). in: ymm0=A ymm1=B ymm2=C ymm3=D.
 * ymm4 = A^B; xmm4 ^= C (legacy); ymm5 = ymm4 & D; xmm5 |= A (legacy); ymm6 = ymm5; xmm6 = B (legacy);
 * ymm7 = ymm6 ^ ymm5 (VEX.256 reads merged upper lanes) */
static T_NOINLINE void ymm_mix_chain(const void *in, void *out) {
    __asm__ volatile(".irp i,0,1,2,3\n vmovdqu \\i*32(%[in]), %%ymm\\i\n .endr\n"
        "vxorps %%ymm1, %%ymm0, %%ymm4\n pxor %%xmm2, %%xmm4\n vandps %%ymm3, %%ymm4, %%ymm5\n orps %%xmm0, %%xmm5\n"
        "vmovaps %%ymm5, %%ymm6\n movaps %%xmm1, %%xmm6\n vxorps %%ymm5, %%ymm6, %%ymm7\n"
        ".irp i,4,5,6,7\n vmovdqu %%ymm\\i, \\i*32(%[out])\n .endr\n vzeroupper\n"
        :: [in]"r"(in), [out]"r"(out) : "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");
}

/* ======================================================================= */
/* XSAVE / XRSTOR / XGETBV(1) with live YMM state                            */
/* ======================================================================= */
static T_NOINLINE void ymm_xsave_rt(const void *in, void *area, void *out) {
    __asm__ volatile(YLOAD("%[in]") "xsave (%[a])\n" YSTORE("%[out]") "vzeroupper\n"
        :: [in]"r"(in), [a]"r"(area), [out]"r"(out), "a"(6u), "d"(0u) : "memory", YCLOB);
}
static T_NOINLINE void ymm_xsave_after_vzeroupper(const void *in, void *area) {
    __asm__ volatile(YLOAD("%[in]") "vzeroupper\n xsave (%[a])\n"
        :: [in]"r"(in), [a]"r"(area), "a"(6u), "d"(0u) : "memory", YCLOB);
}
static T_NOINLINE void ymm_xrstor_rt(const void *in, const void *area, uint32_t rfbm, void *out) {
    __asm__ volatile(YLOAD("%[in]") "xrstor (%[a])\n" YSTORE("%[out]") "vzeroupper\n"
        :: [in]"r"(in), [a]"r"(area), [out]"r"(out), "a"(rfbm), "d"(0u) : "memory", YCLOB);
}
static T_NOINLINE void ymm_xinuse_seq(const void *in, uint32_t *r) {
    __asm__ volatile(YLOAD("%[in]")
        "movl $1, %%ecx\n xgetbv\n movl %%eax, 0(%[r])\n"
        "vzeroupper\n movl $1, %%ecx\n xgetbv\n movl %%eax, 4(%[r])\n"
        "vzeroall\n movl $1, %%ecx\n xgetbv\n movl %%eax, 8(%[r])\n"
        :: [in]"r"(in), [r]"r"(r) : "memory", GPR_A, GPR_C, GPR_D, YCLOB);
}

/* ======================================================================= */
/* spin with ready flag (for preemption / suspension / context tests)       */
/* ======================================================================= */
static T_NOINLINE unsigned long ymm_spin_ready(const void *in, void *out, volatile long *ready,
                                               volatile const long *stop, unsigned long max) {
    __asm__ volatile(YLOAD("%[in]")
        "movl $1, (%[rdy])\n"
        "1: pause\n cmpl $0, (%[stop])\n jne 2f\n dec %[n]\n jnz 1b\n2:\n"
        YSTORE("%[out]") "vzeroupper\n"
        : [n]"+r"(max) : [in]"r"(in), [out]"r"(out), [rdy]"r"(ready), [stop]"r"(stop) : "memory", "cc", YCLOB);
    return max;
}
/* Same, but the thread spins with its AVX component in (or near) the initial configuration:
 * zu=1: VZEROUPPER after the load (low lanes = pattern, upper lanes 0, XINUSE[2] may be 0);
 * zu=2: VZEROALL (every register 0). An OS that skips restoring an "init" AVX component (XSTATE_BV[2]=0
 * after XSAVEOPT/XSAVEC) without re-initialising it would leak another thread's upper lanes here. */
static T_NOINLINE unsigned long ymm_spin_ready_clean(const void *in, void *out, volatile long *ready,
                                                     volatile const long *stop, unsigned long max, int zu) {
    if (zu == 2)
        __asm__ volatile(YLOAD("%[in]") "vzeroall\n movl $1, (%[rdy])\n"
            "1: pause\n cmpl $0, (%[stop])\n jne 2f\n dec %[n]\n jnz 1b\n2:\n"
            YSTORE("%[out]") "vzeroupper\n"
            : [n]"+r"(max) : [in]"r"(in), [out]"r"(out), [rdy]"r"(ready), [stop]"r"(stop) : "memory", "cc", YCLOB);
    else
        __asm__ volatile(YLOAD("%[in]") "vzeroupper\n movl $1, (%[rdy])\n"
            "1: pause\n cmpl $0, (%[stop])\n jne 2f\n dec %[n]\n jnz 1b\n2:\n"
            YSTORE("%[out]") "vzeroupper\n"
            : [n]"+r"(max) : [in]"r"(in), [out]"r"(out), [rdy]"r"(ready), [stop]"r"(stop) : "memory", "cc", YCLOB);
    return max;
}

/* ======================================================================= */
/* gate and reporting                                                       */
/* ======================================================================= */
typedef DWORD64 (WINAPI *gexs_t)(void);
TEST(ymm_gate_state) {
    const t_cpu_info *c = cpu();
    t_info("cpuid_avx", "%d", cpu_hw(F_AVX));
    t_info("cpuid_avx2", "%d", cpu_hw(F_AVX2));
    t_info("osxsave", "%d", c->osxsave);
    t_info("xcr0", "0x%llx", (unsigned long long)(c->xcr0_valid ? c->xcr0 : 0));
    if (cpu_hw(F_AVX) && c->osxsave && c->xcr0_valid && (c->xcr0 & 6) != 6)
        t_note("AVX advertised by CPUID but the OS did not enable XCR0[2:1] (AVX state disabled)");
    ygate();
    CHECK(c->osxsave);
    CHECK_EQ(c->xcr0 & 6, 6);
    CHECK(c->xcr0_supported & T_XS_AVX);
    CHECK_EQ(c->comp[2].size, 256);                  /* CPUID.(0Dh,2).EAX: AVX state = 16 x 16 bytes */
    t_info("avx_xsave_offset", "%u", c->comp[2].offset);
    CHECK(c->comp[2].offset >= 576);                 /* after legacy region + header */
    CHECK(c->xsave_size_enabled >= c->comp[2].offset + c->comp[2].size);
    CHECK_EQ(t_vregs_count(T_VK_YMM), NY);
    CHECK(cpu_ymm_state_enabled());
    if (cpu_hw(F_AVX2)) CHECK(cpu_has(F_AVX2));     /* same gate as AVX */
}

TEST(ymm_api_reporting) {
    ygate();
    const t_env *e = t_env_get();
    gexs_t gef = (gexs_t)t_proc("kernel32.dll", "GetEnabledXStateFeatures");
    if (gef) {
        DWORD64 f = gef();
        t_info("GetEnabledXStateFeatures", "0x%llx", (unsigned long long)f);
        CHECKF((f & 7) == 7, "XCR0 enables AVX state but GetEnabledXStateFeatures()=0x%llx lacks bits 0-2", (unsigned long long)f);
    } else t_note("GetEnabledXStateFeatures not exported");
    CHECKF(IsProcessorFeaturePresent(PF_XSAVE_ENABLED), "OSXSAVE set but PF_XSAVE_ENABLED reports FALSE");
    t_info("pf_avx", "%d", (int)e->pf[PF_AVX_INSTRUCTIONS_AVAILABLE]);
    if (!e->pf[PF_AVX_INSTRUCTIONS_AVAILABLE])
        t_note("IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE) = FALSE (constant unknown before Windows 10 2004 / environment-specific)");
    if (cpu_has(F_AVX2)) t_info("pf_avx2", "%d", (int)e->pf[PF_AVX2_INSTRUCTIONS_AVAILABLE]);
}

CHILD(ymm_vex_when_disabled) {
    (void)argc; (void)argv;
    t_vzeroupper();
    printf("EXECUTED=1\n");
    return 0;
}
TEST(ymm_vex_faults_when_avx_disabled) {
    if (cpu_has(F_AVX)) SKIP("AVX state is enabled here; the #UD path is only reachable when XCR0[2:1] != 11b or AVX is absent");
    static t_child_result r;
    REQUIRE(t_child_run("ymm_vex_when_disabled", "", 5000, &r));
    CHECK(!r.timed_out);
    CHECKF(r.exc_code == 0xC000001Du, "VEX instruction with AVX state disabled: exc=0x%lx exit=0x%lx (want STATUS_ILLEGAL_INSTRUCTION)",
           (unsigned long)r.exc_code, (unsigned long)r.exit_code);
}

TEST(ymm_register_count) {
    ygate();
    CHECK_EQ(t_vregs_count(T_VK_YMM), T_NXMM);
#ifdef T_X64
    CHECK_EQ(NY, 16);
    if (cpu_has(F_AVX512VL)) t_note("AVX512VL: YMM16-31 exist (EVEX only)");
#else
    CHECK_EQ(NY, 8);
    t_note("32-bit mode: only YMM0-7 are addressable");
#endif
}

/* ======================================================================= */
/* round trips                                                              */
/* ======================================================================= */
TEST(ymm_roundtrip_vmovdqu) {
    ygate();
    static yfile in, out;
    for (uint32_t tag = 1; tag <= 4; tag++) {
        ypat(&in, tag); memset(&out, 0, sizeof out);
        ymm_rt_vmovdqu(&in, &out);
        char l[32]; snprintf(l, sizeof l, "vmovdqu tag %u", tag);
        ycheck(&out, &in, l);
    }
    /* lower and upper lanes really are distinct values */
    for (int i = 0; i < NY; i++) CHECK(memcmp(in.v[i], in.v[i] + 16, 16) != 0);
}
TEST(ymm_roundtrip_vmovdqa_aligned) {
    ygate();
    static yfile in, out;
    ypat(&in, 11); memset(&out, 0, sizeof out);
    ymm_rt_vmovdqa(&in, &out);
    ycheck(&out, &in, "vmovdqa");
}
TEST(ymm_roundtrip_fp_moves) {
    ygate();
    static yfile in, out;
    ypat(&in, 12);
    memset(&out, 0, sizeof out); ymm_rt_ups_aps(&in, &out); ycheck(&out, &in, "vmovups/vmovaps");
    memset(&out, 0, sizeof out); ymm_rt_apd_upd(&in, &out); ycheck(&out, &in, "vmovapd/vmovupd");
    memset(&out, 0, sizeof out); ymm_rt_lddqu_dqa(&in, &out); ycheck(&out, &in, "vlddqu/vmovdqa");
    memset(&out, 0, sizeof out); ymm_rt_aps_ntdq(&in, &out); __asm__ volatile("mfence" ::: "memory");
    ycheck(&out, &in, "vmovaps/vmovntdq");
}
TEST(ymm_roundtrip_unaligned_offsets) {
    ygate();
    static const int offs[] = { 1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 33, 48, 63 };
    uint8_t *src = (uint8_t *)VirtualAlloc(NULL, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    uint8_t *dst = (uint8_t *)VirtualAlloc(NULL, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(src && dst);
    static yfile in, got;
    for (size_t k = 0; k < T_COUNTOF(offs); k++) {
        ypat(&in, 100 + (uint32_t)k);
        memcpy(src + offs[k], &in, sizeof in);
        memset(dst, 0xCC, 8192);
        ymm_rt_vmovdqu(src + offs[k], dst + 64 - offs[k]);
        memcpy(&got, dst + 64 - offs[k], sizeof got);
        char l[48]; snprintf(l, sizeof l, "src+%d dst+%d", offs[k], 64 - offs[k]);
        ycheck(&got, &in, l);
        /* the stores wrote exactly NY*32 bytes */
        CHECKF(dst[64 - offs[k] - 1] == 0xCC && dst[64 - offs[k] + NY * 32] == 0xCC, "store overran at offset %d", offs[k]);
    }
    /* straddling a page boundary (both pages committed) */
    ypat(&in, 150);
    uint8_t *ps = src + 4096 - 7 * 32 - 5;
    memcpy(ps, &in, sizeof in);
    ymm_rt_vmovdqu(ps, dst + 4096 - 3 * 32 - 9);
    memcpy(&got, dst + 4096 - 3 * 32 - 9, sizeof got);
    ycheck(&got, &in, "page-straddling");
    VirtualFree(src, 0, MEM_RELEASE); VirtualFree(dst, 0, MEM_RELEASE);
}
TEST(ymm_load_ending_at_guard_page) {
    ygate();
    void *base;
    uint8_t *p = (uint8_t *)t_guarded_alloc(NY * 32, &base);
    REQUIRE(p);
    static yfile in, out;
    ypat(&in, 160);
    memcpy(p, &in, NY * 32);
    ymm_rt_vmovdqu(p, &out);   /* last register load ends exactly at the PAGE_NOACCESS page */
    ycheck(&out, &in, "load ending at guard page");
    t_guarded_free(base);
}
TEST(ymm_rotate_register_moves) {
    ygate();
    static yfile in, out, want;
    ypat(&in, 17);
    ymm_rotate(&in, &out);
    for (int i = 0; i < NY; i++) memcpy(want.v[i], in.v[(i + 1) % NY], 32);
    ycheck(&out, &want, "after rotate (vmovdqa reg,reg)");
}

/* ======================================================================= */
/* per-register semantics                                                   */
/* ======================================================================= */
enum { PR_KEEP, PR_ZERO, PR_LO_ZERO_HI0, PR_LO_KEEP_HI0, PR_LO_ZERO_HIKEEP, PR_LO_ONES_HI0, PR_LO_ONES_HIKEEP, PR_HI_TO_LO };
static void perreg_case(void (*fn)(const yfile *, yfile *), int effect, const char *what) {
    static yfile in, out[16], want;
    ypat(&in, 0x200);
    memset(out, 0, sizeof out);
    fn(&in, out);
    for (int j = 0; j < NY; j++) {
        want = in;
        uint8_t *w = want.v[j];
        switch (effect) {
        case PR_KEEP: break;
        case PR_ZERO: memset(w, 0, 32); break;
        case PR_LO_ZERO_HI0: memset(w, 0, 32); break;
        case PR_LO_KEEP_HI0: memset(w + 16, 0, 16); break;
        case PR_LO_ZERO_HIKEEP: memset(w, 0, 16); break;
        case PR_LO_ONES_HI0: memset(w, 0xFF, 16); memset(w + 16, 0, 16); break;
        case PR_LO_ONES_HIKEEP: memset(w, 0xFF, 16); break;
        case PR_HI_TO_LO: memcpy(w, in.v[j] + 16, 16); memset(w + 16, 0, 16); break;
        }
        for (int i = 0; i < NY; i++)
            ycheck_one(out[j].v[i], want.v[i], "ymm%d after %s on reg %d", i, what, j);
    }
}
TEST(ymm_perreg_vxorps256_zeroes_only_target) { ygate(); perreg_case(ymm_perreg_vxorps256, PR_ZERO, "vxorps ymm,ymm,ymm"); }
TEST(ymm_perreg_vex128_move_zeroes_upper) { ygate(); perreg_case(ymm_perreg_vmovdqa128, PR_LO_KEEP_HI0, "vmovdqa xmm,xmm (VEX.128)"); }
TEST(ymm_perreg_vex128_vpxor_zeroes_all) { ygate(); perreg_case(ymm_perreg_vpxor128, PR_LO_ZERO_HI0, "vpxor xmm,xmm,xmm (VEX.128)"); }
TEST(ymm_perreg_vex128_vpcmpeqd_ones_upper0) { ygate(); perreg_case(ymm_perreg_vpcmpeqd128, PR_LO_ONES_HI0, "vpcmpeqd xmm (VEX.128)"); }
TEST(ymm_perreg_legacy_movdqa_keeps_upper) { ygate(); perreg_case(ymm_perreg_legacy_movdqa, PR_KEEP, "movdqa xmm,xmm (legacy SSE)"); }
TEST(ymm_perreg_legacy_pxor_keeps_upper) { ygate(); perreg_case(ymm_perreg_legacy_pxor, PR_LO_ZERO_HIKEEP, "pxor xmm,xmm (legacy SSE)"); }
TEST(ymm_perreg_legacy_pcmpeqd_keeps_upper) { ygate(); perreg_case(ymm_perreg_legacy_pcmpeqd, PR_LO_ONES_HIKEEP, "pcmpeqd xmm,xmm (legacy SSE)"); }
TEST(ymm_perreg_vextractf128_reg_zeroes_upper) { ygate(); perreg_case(ymm_perreg_vextractf128_hi, PR_HI_TO_LO, "vextractf128 $1,ymm,xmm(same)"); }

/* ======================================================================= */
/* op lists: VEX.128 zeroing vs legacy SSE preservation                      */
/* ======================================================================= */
static void oplist_case(void (*fn)(const void *, const void *, void *), void (*sim)(yfile *, const ymem *, int), int vex,
                        uint32_t tag, const char *label) {
    static yfile in, out, want;
    static ymem m;
    ymem_init(&m);
    ypat(&in, tag);
    fn(&in, &m, &out);
    want = in;
    sim(&want, &m, vex);
    ycheck(&out, &want, label);
    /* explicit upper-lane statement for the readable summary */
    int kept = 0, zero = 0;
    for (int i = 0; i < 8; i++) {
        if (!memcmp(out.v[i] + 16, in.v[i] + 16, 16)) kept++;
        if (!memcmp(out.v[i] + 16, g_zero32, 16)) zero++;
    }
    t_info("upper_kept", "%d", kept); t_info("upper_zero", "%d", zero);
}
TEST(ymm_legacy_sse_ops_preserve_upper) { ygate(); oplist_case(ymm_legacy_listA, sim_listA, 0, 0x300, "legacy list A"); }
TEST(ymm_vex128_ops_zero_upper) { ygate(); oplist_case(ymm_vex128_listA, sim_listA, 1, 0x300, "VEX.128 list A"); }
TEST(ymm_legacy_sse_idioms_cvt_preserve_upper) { ygate(); oplist_case(ymm_legacy_listB, sim_listB, 0, 0x301, "legacy list B"); }
TEST(ymm_vex128_idioms_cvt_zero_upper) { ygate(); oplist_case(ymm_vex128_listB, sim_listB, 1, 0x301, "VEX.128 list B"); }
TEST(ymm_legacy_sse_loads_preserve_upper) { ygate(); oplist_case(ymm_legacy_loads, sim_loads, 0, 0x302, "legacy loads"); }
TEST(ymm_vex128_loads_zero_upper) { ygate(); oplist_case(ymm_vex128_loads, sim_loads, 1, 0x302, "VEX.128 loads"); }

TEST(ymm_gpr_to_xmm_moves) {
    ygate();
    static yfile in, out, want;
    ypat(&in, 0x310);
    const uint32_t g = 0xA5B6C7D8u;
    ymm_gpr_moves(&in, g, &out);
    want = in;
    memset(want.v[1], 0, 32); wr32(want.v[1], 0, g);                         /* vmovd: zero 255:32 */
    memset(want.v[2], 0, 16); wr32(want.v[2], 0, g);                         /* movd: zero 127:32, keep upper */
    wr32(want.v[3], 2, g); memset(want.v[3] + 16, 0, 16);                    /* vpinsrd: merge, upper 0 */
    wr16(want.v[4], 3, (uint16_t)g);                                         /* pinsrw: merge, upper kept */
    ycheck(&out, &want, "GPR->XMM moves");
#ifdef T_X64
    const uint64_t g64 = 0x0123456789ABCDEFull;
    ymm_gpr_moves64(&in, g64, &out);
    want = in;
    memset(want.v[9], 0, 32); wr64(want.v[9], 0, g64);
    memset(want.v[10], 0, 16); wr64(want.v[10], 0, g64);
    wr64(want.v[11], 1, g64); memset(want.v[11] + 16, 0, 16);
    ycheck(&out, &want, "GPR64->XMM moves");
#endif
}

TEST(ymm_x64_high_registers_ops) {
    ygate();
#ifndef T_X64
    SKIP("XMM8-15/YMM8-15 exist only in 64-bit mode");
#else
    static yfile in, out, s;
    ypat(&in, 0x320);
    ymm_x64_ext_list(&in, &out);
    s = in;
    { for (int k = 0; k < 4; k++) wr32(s.v[12], k, rd32(s.v[8], k) + rd32(s.v[15], k)); } memset(s.v[12] + 16, 0, 16);
    memcpy(s.v[13], s.v[9], 16); memset(s.v[13] + 16, 0, 16);
    for (int k = 0; k < 2; k++) wr64(s.v[14], k, rd64(s.v[14], k) + rd64(s.v[10], k));
    memset(s.v[11], 0, 16);
    { for (int k = 0; k < 2; k++) wr64(s.v[8], k, rd64(s.v[9], k) ^ rd64(s.v[1], k)); } memset(s.v[8] + 16, 0, 16);
    memcpy(s.v[2], s.v[15], 16);
    { uint8_t t[32]; memcpy(t, s.v[15], 16); memcpy(t + 16, s.v[10], 16); memcpy(s.v[10], t, 32); }
    { uint8_t t[32]; memcpy(t, s.v[14] + 16, 16); memset(t + 16, 0, 16); memcpy(s.v[15], t, 32); }
    ycheck(&out, &s, "x64 high-register op list");
#endif
}

TEST(ymm_store_forms_exact_footprint) {
    ygate();
    static yfile in, out;
    static uint8_t T_ALIGN(64) o[192], want[192];
    ypat(&in, 0x330);
    memset(o, 0xCC, sizeof o); memset(want, 0xCC, sizeof want);
    ymm_store_forms(&in, o, &out);
    memcpy(want + 0, in.v[1], 16);
    memcpy(want + 32, in.v[2], 32);
    memcpy(want + 64, in.v[3] + 16, 16);
    memcpy(want + 80, in.v[4], 4);
    memcpy(want + 88, in.v[5], 8);
    memcpy(want + 96, in.v[6] + 8, 8);
    memcpy(want + 112, in.v[7], 16);
    memcpy(want + 128, in.v[0], 16);
    CHECK_MEM(o, want, sizeof o);
    ycheck(&out, &in, "registers after stores");
}

/* ======================================================================= */
/* vzeroupper / vzeroall                                                    */
/* ======================================================================= */
TEST(ymm_vzeroupper_semantics) {
    ygate();
    static yfile in, out, want;
    for (uint32_t tag = 0x400; tag < 0x403; tag++) {
        ypat(&in, tag);
        ymm_vzeroupper_rt(&in, &out);
        want = in;
        for (int i = 0; i < NY; i++) memset(want.v[i] + 16, 0, 16);
        ycheck(&out, &want, "after vzeroupper");
    }
}
TEST(ymm_vzeroall_semantics) {
    ygate();
    static yfile in, out, want;
    ypat(&in, 0x410);
    ymm_vzeroall_rt(&in, &out);
    memset(&want, 0, sizeof want);
    ycheck(&out, &want, "after vzeroall");
}
TEST(ymm_vzeroupper_then_legacy_and_vex256) {
    ygate();
    static yfile in, out, want;
    static ymem m;
    ymem_init(&m);
    ypat(&in, 0x420);
    ymm_vzu_then_legacy(&in, &m, &out);
    want = in;
    for (int i = 0; i < NY; i++) memset(want.v[i] + 16, 0, 16);
    memcpy(want.v[0], m.b, 16);
    for (int k = 0; k < 4; k++) wr32(want.v[1], k, rd32(want.v[1], k) + rd32(m.b, k));
    memset(want.v[2], 0, 16);
    memset(want.v[3], 0, 16); memcpy(want.v[3], m.b + 16, 4);
    memcpy(want.v[4], m.b + 32, 32);
    ycheck(&out, &want, "vzeroupper; legacy SSE; VEX.256 load");
}
#ifdef T_X64
static T_NOINLINE T_TARGET("avx512f,avx512vl") void avx512_ymm_hi16(const void *in, const void *in2, void *out, void *out2) {
    __asm__ volatile(YLOAD("%[in]")
        ".irp i,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n vmovdqu64 (\\i-16)*32(%[in2]), %%ymm\\i\n .endr\n"
        "vzeroupper\n vmovdqa64 %%xmm17, %%xmm17\n vmovdqa %%xmm0, %%xmm0\n"
        YSTORE("%[out]")
        ".irp i,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n vmovdqu64 %%ymm\\i, (\\i-16)*32(%[out2])\n .endr\n vzeroupper\n"
        :: [in]"r"(in), [in2]"r"(in2), [out]"r"(out), [out2]"r"(out2)
        : "memory", YCLOB, "xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23",
          "xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
}
/* VZEROALL in 64-bit mode zeroes (Z/Y)MM0-15 only; (Z/Y)MM16-31 are not modified (SDM VZEROALL operation). */
static T_NOINLINE T_TARGET("avx512f,avx512vl") void avx512_ymm_hi16_vzeroall(const void *in, const void *in2, void *out, void *out2) {
    __asm__ volatile(YLOAD("%[in]")
        ".irp i,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n vmovdqu64 (\\i-16)*32(%[in2]), %%ymm\\i\n .endr\n"
        "vzeroall\n"
        YSTORE("%[out]")
        ".irp i,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n vmovdqu64 %%ymm\\i, (\\i-16)*32(%[out2])\n .endr\n vzeroupper\n"
        :: [in]"r"(in), [in2]"r"(in2), [out]"r"(out), [out2]"r"(out2)
        : "memory", YCLOB, "xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23",
          "xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
}
#endif
TEST(ymm_avx512vl_regs_16_31_untouched_by_vzeroupper) {
    ygate();
#ifndef T_X64
    SKIP("YMM16-31 exist only in 64-bit mode with AVX512VL");
#else
    if (!cpu_has(F_AVX512VL)) SKIP("AVX512VL not usable: only 16 YMM registers");
    static yfile in, in2, out, out2, want, want2;
    ypat(&in, 0x430); ypat(&in2, 0x431);
    avx512_ymm_hi16(&in, &in2, &out, &out2);
    want = in; want2 = in2;
    for (int i = 0; i < 16; i++) memset(want.v[i] + 16, 0, 16);
    memset(want2.v[1] + 16, 0, 16);   /* EVEX.128 vmovdqa64 xmm17 zeroes 255:128 */
    ycheck(&out, &want, "ymm0-15 after vzeroupper");
    for (int i = 0; i < 16; i++) ycheck_one(out2.v[i], want2.v[i], "ymm%d after vzeroupper (not affected)", i + 16);
#endif
}
TEST(ymm_avx512vl_regs_16_31_untouched_by_vzeroall) {
    ygate();
#ifndef T_X64
    SKIP("YMM16-31 exist only in 64-bit mode with AVX512VL");
#else
    if (!cpu_has(F_AVX512VL)) SKIP("AVX512VL not usable: only 16 YMM registers");
    static yfile in, in2, out, out2;
    ypat(&in, 0x432); ypat(&in2, 0x433);
    memset(&out, 0xEE, sizeof out);
    avx512_ymm_hi16_vzeroall(&in, &in2, &out, &out2);
    ycheck(&out, &g_zfile, "ymm0-15 after vzeroall");
    for (int i = 0; i < 16; i++) ycheck_one(out2.v[i], in2.v[i], "ymm%d after vzeroall (not affected)", i + 16);
#endif
}

/* ======================================================================= */
/* lane operations                                                          */
/* ======================================================================= */
static void lanes2(uint8_t a[32], uint8_t b[32], uint32_t tag) {
    t_pattern_reg(a, 32, tag, 1); t_pattern_reg(b, 32, tag, 2);
}
static void ref_perm2x128(const uint8_t *s1, const uint8_t *s2, unsigned imm, uint8_t *d) {
    const uint8_t *src[4] = { s1, s1 + 16, s2, s2 + 16 };
    if (imm & 0x08) memset(d, 0, 16); else memcpy(d, src[imm & 3], 16);
    if (imm & 0x80) memset(d + 16, 0, 16); else memcpy(d + 16, src[(imm >> 4) & 3], 16);
}
static void perm2x128_case(void (*fn)(const void *, const void *, void *), const char *nm) {
    static uint8_t T_ALIGN(32) a[32], b[32], o[256][32];
    lanes2(a, b, 0x500);
    fn(a, b, o);
    for (unsigned imm = 0; imm < 256; imm++) {
        uint8_t w[32]; ref_perm2x128(a, b, imm, w);
        ycheck_one(o[imm], w, "%s imm=0x%02x", nm, imm);
    }
}
TEST(ymm_vperm2f128_all_imm) { ygate(); perm2x128_case(ymm_all_vperm2f128, "vperm2f128"); }
TEST(ymm_vperm2i128_all_imm) { ygate2(); perm2x128_case(avx2_ymm_all_vperm2i128, "vperm2i128"); }

static void insext_case(void (*fn)(const void *, const void *, const void *, void *), const char *nm) {
    static uint8_t T_ALIGN(32) a[32], b[32], c[32], o[8][32], w[8][32];
    lanes2(a, b, 0x510); t_pattern_reg(c, 32, 0x510, 3);
    memset(o, 0xEE, sizeof o);
    fn(a, b, c, o);
    memcpy(w[0], a, 16); memcpy(w[0] + 16, b, 16);       /* ins $1, xmm1 */
    memcpy(w[1], b, 16); memcpy(w[1] + 16, a + 16, 16);  /* ins $0 */
    memcpy(w[2], w[0], 32);                                /* $3 == $1 (imm8[0] only) */
    memcpy(w[3], w[1], 32);                                /* $2 == $0 */
    memcpy(w[4], a, 16); memcpy(w[4] + 16, b + 16, 16);  /* ins $1, m128 */
    memcpy(w[5], a + 16, 16); memset(w[5] + 16, 0, 16);  /* ext $1 -> xmm: upper zeroed */
    memcpy(w[6], b, 16); memset(w[6] + 16, 0, 16);       /* ext $0 -> xmm */
    memcpy(w[7], a + 16, 16); memset(w[7] + 16, 0xEE, 16); /* ext $3 -> m128: 16 bytes only */
    static const char *what[8] = { "ins $1 xmm", "ins $0 xmm", "ins $3 xmm", "ins $2 xmm", "ins $1 m128",
                                   "ext $1 -> xmm (dirty dst)", "ext $0 -> xmm (dirty dst)", "ext $3 -> m128" };
    for (int k = 0; k < 8; k++) ycheck_one(o[k], w[k], "%s %s", nm, what[k]);
}
TEST(ymm_vinsertf128_vextractf128) { ygate(); insext_case(ymm_insext_f128, "f128"); }
TEST(ymm_vinserti128_vextracti128) { ygate2(); insext_case(avx2_ymm_insext_i128, "i128"); }

/* generic all-imm checker */
typedef void (*ref_imm_fn)(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d);
static void imm_case(void (*fn)(const void *, const void *, void *), ref_imm_fn ref, unsigned count, uint32_t tag, const char *nm) {
    static uint8_t T_ALIGN(32) a[32], b[32], o[256][32];
    lanes2(a, b, tag);
    fn(a, b, o);
    for (unsigned imm = 0; imm < count; imm++) {
        uint8_t w[32]; ref(a, b, imm, w);
        ycheck_one(o[imm], w, "%s imm=%u", nm, imm);
    }
}
static void ref_vpermilps(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    (void)b; for (int l = 0; l < 2; l++) for (int i = 0; i < 4; i++) wr32(d, l * 4 + i, rd32(a, l * 4 + ((imm >> (2 * i)) & 3)));
}
static void ref_vshufps(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    for (int l = 0; l < 2; l++) {
        wr32(d, l * 4 + 0, rd32(a, l * 4 + (imm & 3)));        wr32(d, l * 4 + 1, rd32(a, l * 4 + ((imm >> 2) & 3)));
        wr32(d, l * 4 + 2, rd32(b, l * 4 + ((imm >> 4) & 3))); wr32(d, l * 4 + 3, rd32(b, l * 4 + ((imm >> 6) & 3)));
    }
}
static void ref_vblendps(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    for (int i = 0; i < 8; i++) wr32(d, i, (imm >> i) & 1 ? rd32(b, i) : rd32(a, i));
}
static void ref_vpermilpd(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    (void)b; for (int i = 0; i < 4; i++) wr64(d, i, rd64(a, (i & 2) + ((imm >> i) & 1)));
}
static void ref_vblendpd(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    for (int i = 0; i < 4; i++) wr64(d, i, (imm >> i) & 1 ? rd64(b, i) : rd64(a, i));
}
static void ref_vpermq(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    (void)b; for (int i = 0; i < 4; i++) wr64(d, i, rd64(a, (imm >> (2 * i)) & 3));
}
static void ref_vpalignr(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    for (int l = 0; l < 2; l++) {
        uint8_t t[32]; memcpy(t, b + 16 * l, 16); memcpy(t + 16, a + 16 * l, 16);   /* src1(a):src2(b) */
        for (unsigned k = 0; k < 16; k++) d[16 * l + k] = k + imm < 32 ? t[k + imm] : 0;
    }
}
static void ref_vpslldq(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    (void)b; for (int l = 0; l < 2; l++) for (unsigned k = 0; k < 16; k++) d[16 * l + k] = imm <= 15 && k >= imm ? a[16 * l + k - imm] : 0;
}
static void ref_vpsrldq(const uint8_t *a, const uint8_t *b, unsigned imm, uint8_t *d) {
    (void)b; for (int l = 0; l < 2; l++) for (unsigned k = 0; k < 16; k++) d[16 * l + k] = imm <= 15 && k + imm < 16 ? a[16 * l + k + imm] : 0;
}
TEST(ymm_vpermilps_all_imm_in_lane) { ygate(); imm_case(ymm_all_vpermilps, ref_vpermilps, 256, 0x520, "vpermilps"); }
TEST(ymm_vshufps_all_imm_in_lane) { ygate(); imm_case(ymm_all_vshufps, ref_vshufps, 256, 0x521, "vshufps"); }
TEST(ymm_vblendps_all_imm) { ygate(); imm_case(ymm_all_vblendps, ref_vblendps, 256, 0x522, "vblendps"); }
TEST(ymm_vpermilpd_vblendpd_all_imm) {
    ygate();
    imm_case(ymm_all_vpermilpd, ref_vpermilpd, 16, 0x523, "vpermilpd");
    imm_case(ymm_all_vblendpd, ref_vblendpd, 16, 0x524, "vblendpd");
}
TEST(ymm_vpermq_all_imm_cross_lane) { ygate2(); imm_case(avx2_ymm_all_vpermq, ref_vpermq, 256, 0x525, "vpermq"); }
TEST(ymm_vpshufd_all_imm_in_lane) { ygate2(); imm_case(avx2_ymm_all_vpshufd, ref_vpermilps, 256, 0x526, "vpshufd"); }
TEST(ymm_vpblendd_all_imm) { ygate2(); imm_case(avx2_ymm_all_vpblendd, ref_vblendps, 256, 0x527, "vpblendd"); }
TEST(ymm_vpalignr_all_imm_in_lane) { ygate2(); imm_case(avx2_ymm_all_vpalignr, ref_vpalignr, 32, 0x528, "vpalignr"); }
TEST(ymm_vpslldq_vpsrldq_in_lane) {
    ygate2();
    imm_case(avx2_ymm_all_vpslldq, ref_vpslldq, 17, 0x529, "vpslldq");
    imm_case(avx2_ymm_all_vpsrldq, ref_vpsrldq, 17, 0x52A, "vpsrldq");
}

TEST(ymm_vpshufb_in_lane) {
    ygate2();
    static uint8_t T_ALIGN(32) a[32], idx[32], o[32], w[32];
    t_rng r; t_rng_seed(&r, 0x530);
    for (int t = 0; t < 48; t++) {
        t_rng_fill(&r, a, 32); t_rng_fill(&r, idx, 32);
        if (t == 0) for (int k = 0; k < 32; k++) idx[k] = (uint8_t)(31 - k);   /* indices >15 must wrap within the lane */
        avx2_ymm_vpshufb(a, idx, o);
        for (int l = 0; l < 2; l++) for (int k = 0; k < 16; k++) {
            uint8_t s = idx[16 * l + k];
            w[16 * l + k] = (s & 0x80) ? 0 : a[16 * l + (s & 15)];
        }
        ycheck_one(o, w, "vpshufb trial %d", t);
    }
}
TEST(ymm_vpermd_vpermps_cross_lane) {
    ygate2();
    static uint8_t T_ALIGN(32) a[32], idx[32], o[64], w[32];
    t_rng r; t_rng_seed(&r, 0x531);
    for (int t = 0; t < 48; t++) {
        t_rng_fill(&r, a, 32); t_rng_fill(&r, idx, 32);
        avx2_ymm_vpermd(a, idx, o);
        for (int i = 0; i < 8; i++) wr32(w, i, rd32(a, rd32(idx, i) & 7));
        ycheck_one(o, w, "vpermd trial %d", t);
        ycheck_one(o + 32, w, "vpermps trial %d", t);
    }
}
TEST(ymm_broadcasts_avx) {
    ygate();
    static uint8_t T_ALIGN(32) m[64], c[32], o[4][32], w[4][32];
    t_pattern_reg(m, 64, 0x540, 5); t_pattern_reg(c, 32, 0x540, 6);
    ymm_bcast_avx(m, c, o);
    for (int i = 0; i < 8; i++) memcpy(w[0] + 4 * i, m, 4);
    for (int i = 0; i < 4; i++) memcpy(w[1] + 8 * i, m + 8, 8);
    memcpy(w[2], m + 16, 16); memcpy(w[2] + 16, m + 16, 16);
    memset(w[3], 0, 32); for (int i = 0; i < 4; i++) memcpy(w[3] + 4 * i, m + 4, 4);
    static const char *nm[4] = { "vbroadcastss m32->ymm", "vbroadcastsd m64->ymm", "vbroadcastf128", "vbroadcastss m32->xmm (VEX.128, dirty dst)" };
    for (int k = 0; k < 4; k++) ycheck_one(o[k], w[k], "%s", nm[k]);
}
TEST(ymm_broadcasts_avx2) {
    ygate2();
    static uint8_t T_ALIGN(32) m[64], c[32], o[8][32], w[8][32];
    t_pattern_reg(m, 64, 0x541, 5); t_pattern_reg(c, 32, 0x541, 6);
    memset(o, 0, sizeof o);
    avx2_ymm_bcast(m, c, o);
    memcpy(w[0], m + 16, 16); memcpy(w[0] + 16, m + 16, 16);
    memset(w[1], m[1], 32);
    for (int i = 0; i < 16; i++) memcpy(w[2] + 2 * i, m + 2, 2);
    for (int i = 0; i < 4; i++) memcpy(w[3] + 8 * i, m, 8);
    for (int i = 0; i < 8; i++) memcpy(w[5] + 4 * i, m, 4);
    memset(w[6], 0, 32); for (int i = 0; i < 4; i++) memcpy(w[6] + 4 * i, m + 12, 4);
    for (int i = 0; i < 4; i++) memcpy(w[7] + 8 * i, m, 8);
    static const char *nm[8] = { "vbroadcasti128", "vpbroadcastb m8", "vpbroadcastw m16", "vpbroadcastq xmm", "",
                                 "vbroadcastss xmm->ymm", "vpbroadcastd m32->xmm (VEX.128, dirty dst)", "vbroadcastsd xmm->ymm" };
    for (int k = 0; k < 8; k++) if (k != 4) ycheck_one(o[k], w[k], "%s", nm[k]);
}
TEST(ymm_bitwise_256) {
    ygate();
    static uint8_t T_ALIGN(32) a[32], b[32], o[4][32], w[4][32];
    lanes2(a, b, 0x550);
    ymm_bitops(a, b, o);
    for (int k = 0; k < 32; k++) { w[0][k] = a[k] ^ b[k]; w[1][k] = a[k] & b[k]; w[2][k] = a[k] | b[k]; w[3][k] = (uint8_t)(~a[k] & b[k]); }
    static const char *nm[4] = { "vxorps", "vandps", "vorps", "vandnps" };
    for (int k = 0; k < 4; k++) ycheck_one(o[k], w[k], "%s ymm", nm[k]);
}
TEST(ymm_fp_arith_upper_lane_exact) {
    ygate();
    static float T_ALIGN(32) fa[8], fb[8];
    static double T_ALIGN(32) da[4], db[4];
    static uint8_t T_ALIGN(32) A[32], B[32], o[6][32], w[6][32];
    /* float view for ps ops, double view for pd ops: build separately and run twice */
    for (int i = 0; i < 8; i++) { fa[i] = (float)(i + 1) * 4.0f; fb[i] = (i & 1) ? 4.0f : (float)(i + 1) * 4.0f; }
    memcpy(A, fa, 32); memcpy(B, fb, 32);
    ymm_fpops(A, B, o);
    for (int i = 0; i < 8; i++) {
        float s = (float)(i + 1) * 4.0f + fb[i], q = ((float)(i + 1) * 4.0f) / fb[i];
        memcpy(w[0] + 4 * i, &s, 4); memcpy(w[3] + 4 * i, &q, 4);
        uint32_t eq = fa[i] == fb[i] ? 0xFFFFFFFFu : 0; memcpy(w[4] + 4 * i, &eq, 4);
    }
    ycheck_one(o[0], w[0], "vaddps ymm"); ycheck_one(o[3], w[3], "vdivps ymm"); ycheck_one(o[4], w[4], "vcmpeqps ymm");
    for (int i = 0; i < 4; i++) { da[i] = (double)((i + 2) * (i + 2)); db[i] = (i & 1) ? -3.0 : (double)(i + 5); }
    memcpy(A, da, 32); memcpy(B, db, 32);
    ymm_fpops(A, B, o);
    for (int i = 0; i < 4; i++) {
        double p = da[i] * db[i], r = (double)(i + 2), mn = da[i] < db[i] ? da[i] : db[i];
        memcpy(w[1] + 8 * i, &p, 8); memcpy(w[2] + 8 * i, &r, 8); memcpy(w[5] + 8 * i, &mn, 8);
    }
    ycheck_one(o[1], w[1], "vmulpd ymm"); ycheck_one(o[2], w[2], "vsqrtpd ymm"); ycheck_one(o[5], w[5], "vminpd ymm");
}
TEST(ymm_int_arith_lane_boundaries) {
    ygate2();
    static uint8_t T_ALIGN(32) a[32], b[32], o[8][32], w[8][32];
    t_rng r; t_rng_seed(&r, 0x560);
    for (int t = 0; t < 8; t++) {
        t_rng_fill(&r, a, 32); t_rng_fill(&r, b, 32);
        if (t == 0) { memset(a, 0xFF, 32); memset(b, 0, 32); b[0] = b[8] = b[16] = b[24] = 1; }  /* carries must stop at element bounds */
        if (t == 1) { memset(a, 0, 32); memset(b, 0, 32); b[15] = 1; b[16] = 1; }
        avx2_ymm_intops(a, b, o);
        for (int k = 0; k < 32; k++) { w[0][k] = (uint8_t)(a[k] + b[k]); unsigned s = (unsigned)a[k] + b[k]; w[6][k] = (uint8_t)(s > 255 ? 255 : s); }
        for (int k = 0; k < 16; k++) wr16(w[1], k, (uint16_t)(rd16(a, k) + rd16(b, k)));
        for (int k = 0; k < 8; k++) {
            wr32(w[2], k, rd32(a, k) + rd32(b, k));
            wr32(w[5], k, rd32(a, k) == rd32(b, k) ? 0xFFFFFFFFu : 0);
            wr32(w[7], k, (int32_t)rd32(a, k) > (int32_t)rd32(b, k) ? rd32(a, k) : rd32(b, k));
        }
        for (int k = 0; k < 4; k++) { wr64(w[3], k, rd64(a, k) + rd64(b, k)); wr64(w[4], k, rd64(a, k) - rd64(b, k)); }
        static const char *nm[8] = { "vpaddb", "vpaddw", "vpaddd", "vpaddq", "vpsubq", "vpcmpeqd", "vpaddusb", "vpmaxsd" };
        for (int k = 0; k < 8; k++) ycheck_one(o[k], w[k], "%s trial %d", nm[k], t);
    }
}
TEST(ymm_unpack_in_lane) {
    ygate();
    static uint8_t T_ALIGN(32) a[32], b[32], o[4][32], w[4][32];
    lanes2(a, b, 0x570);
    ymm_unpack(a, b, o);
    for (int l = 0; l < 2; l++) {
        int L = 4 * l;
        wr32(w[0], L + 0, rd32(a, L + 0)); wr32(w[0], L + 1, rd32(b, L + 0)); wr32(w[0], L + 2, rd32(a, L + 1)); wr32(w[0], L + 3, rd32(b, L + 1));
        wr32(w[1], L + 0, rd32(a, L + 2)); wr32(w[1], L + 1, rd32(b, L + 2)); wr32(w[1], L + 2, rd32(a, L + 3)); wr32(w[1], L + 3, rd32(b, L + 3));
        wr64(w[2], 2 * l, rd64(a, 2 * l)); wr64(w[2], 2 * l + 1, rd64(b, 2 * l));
        wr64(w[3], 2 * l, rd64(a, 2 * l + 1)); wr64(w[3], 2 * l + 1, rd64(b, 2 * l + 1));
    }
    static const char *nm[4] = { "vunpcklps", "vunpckhps", "vunpcklpd", "vunpckhpd" };
    for (int k = 0; k < 4; k++) ycheck_one(o[k], w[k], "%s ymm", nm[k]);
}
TEST(ymm_vptest_vmovmsk_see_upper_lane) {
    ygate();
    static uint8_t T_ALIGN(32) a[32], b[32];
    uint32_t r[6];
    /* only the upper lane differs from zero: an implementation that ignored bits 255:128 would set ZF */
    memset(a, 0, 32); memset(b, 0, 32);
    a[25] = 0x10; b[25] = 0x10;
    ymm_test_mask(a, b, r);
    CHECKF(r[0] == 0 && r[1] == 1, "vptest same upper bit: ZF=%u CF=%u (want 0,1)", r[0], r[1]);
    b[25] = 0x20;
    ymm_test_mask(a, b, r);
    CHECKF(r[0] == 1 && r[1] == 0, "vptest disjoint upper bits: ZF=%u CF=%u (want 1,0)", r[0], r[1]);
    /* sign bits in dwords 5 and 7 (upper lane) -> mask 0xA0; qwords 2,3 sign = bits of dwords 5,7 -> 0xC */
    memset(a, 0, 32); a[23] = 0x80; a[31] = 0x80;
    memset(b, 0, 32); b[31] = 0x80;
    ymm_test_mask(a, b, r);
    CHECK_EQ(r[2], 0xA0);
    CHECK_EQ(r[3], 0xC);
    CHECKF(r[4] == 0, "vtestps upper-lane sign overlap: ZF=%u (want 0)", r[4]);
    CHECKF(r[5] == 1, "vtestps: CF=%u (want 1: b's signs subset of a's)", r[5]);
}

/* ======================================================================= */
/* mixing                                                                   */
/* ======================================================================= */
TEST(ymm_mix_vex256_legacy_chain) {
    ygate();
    static yfile in, out;
    ypat(&in, 0x600);
    ymm_mix_chain(&in, &out);
    uint8_t E[32], F[32], G[32], H[32];
    for (int k = 0; k < 32; k++) E[k] = in.v[0][k] ^ in.v[1][k];
    for (int k = 0; k < 16; k++) E[k] ^= in.v[2][k];                 /* legacy pxor: upper kept */
    for (int k = 0; k < 32; k++) F[k] = E[k] & in.v[3][k];
    for (int k = 0; k < 16; k++) F[k] |= in.v[0][k];                 /* legacy orps */
    memcpy(G, F, 32); memcpy(G, in.v[1], 16);                         /* legacy movaps into low lane */
    for (int k = 0; k < 32; k++) H[k] = G[k] ^ F[k];
    ycheck_one(out.v[4], E, "ymm4 (vxorps; pxor)");
    ycheck_one(out.v[5], F, "ymm5 (vandps; orps)");
    ycheck_one(out.v[6], G, "ymm6 (vmovaps; movaps)");
    ycheck_one(out.v[7], H, "ymm7 (vxorps of merged regs)");
    CHECK(memcmp(H + 16, g_zero32, 16) == 0);                         /* upper lanes of G and F agree */
}

/* ======================================================================= */
/* XSAVE / XRSTOR / XINUSE                                                  */
/* ======================================================================= */
static void xgate(void) { ygate(); if (!cpu_has(F_XSAVE)) SKIP("XSAVE not usable"); }
static uint8_t *xarea(void **base) {
    *base = VirtualAlloc(NULL, 16384, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return (uint8_t *)*base;
}
TEST(ymm_xsave_captures_both_lanes) {
    xgate();
    const t_cpu_info *c = cpu();
    void *base; uint8_t *a = xarea(&base);
    REQUIRE(a);
    memset(a, 0xA5, 4096);
    memset(a + 512, 0, 64);            /* header */
    static yfile in, out;
    ypat(&in, 0x700);
    ymm_xsave_rt(&in, a, &out);
    ycheck(&out, &in, "registers after xsave");
    uint64_t bv; memcpy(&bv, a + 512, 8);
    t_info("xstate_bv", "0x%llx", (unsigned long long)bv);
    CHECKF((bv & 6) == 6, "XSTATE_BV=0x%llx: SSE and AVX components hold non-init values", (unsigned long long)bv);
    CHECKF((bv & ~6ull) == 0, "XSTATE_BV=0x%llx has bits outside RFBM=6", (unsigned long long)bv);
    uint32_t off = c->comp[2].offset;
    for (int i = 0; i < NY; i++) {
        char l[64];
        snprintf(l, sizeof l, "xsave legacy XMM%d", i);
        CHECK_REG(l, a + 160 + 16 * i, in.v[i], 16);
        snprintf(l, sizeof l, "xsave AVX YMM%d_H", i);
        CHECK_REG(l, a + off + 16 * i, in.v[i] + 16, 16);
    }
#ifdef T_X86
    int untouched = 1;
    for (int i = 8; i < 16; i++) for (int k = 0; k < 16; k++) if (a[off + 16 * i + k] != 0xA5) untouched = 0;
    t_info("x86_ymm8_15_area_untouched", "%d", untouched);
#endif
    VirtualFree(base, 0, MEM_RELEASE);
}
TEST(ymm_xsave_after_vzeroupper) {
    xgate();
    const t_cpu_info *c = cpu();
    void *base; uint8_t *a = xarea(&base);
    REQUIRE(a);
    memset(a, 0xA5, 4096); memset(a + 512, 0, 64);
    static yfile in;
    ypat(&in, 0x701);
    ymm_xsave_after_vzeroupper(&in, a);
    uint64_t bv; memcpy(&bv, a + 512, 8);
    t_info("xstate_bv_after_vzeroupper", "0x%llx", (unsigned long long)bv);
    CHECK(bv & 2);
    for (int i = 0; i < NY; i++) { char l[48]; snprintf(l, sizeof l, "xsave XMM%d after vzeroupper", i); CHECK_REG(l, a + 160 + 16 * i, in.v[i], 16); }
    if (bv & 4) {
        for (int i = 0; i < NY; i++) { char l[48]; snprintf(l, sizeof l, "YMM%d_H after vzeroupper", i); CHECK_REG(l, a + c->comp[2].offset + 16 * i, g_zero32, 16); }
    } else t_note("XSTATE_BV[2]=0 after VZEROUPPER: processor tracks the AVX component as initial");
    VirtualFree(base, 0, MEM_RELEASE);
}
static void build_xrstor_area(uint8_t *a, const yfile *src, uint64_t bv) {
    memset(a, 0, 4096);
    unsigned mx = t_get_mxcsr(); memcpy(a + 24, &mx, 4);
    unsigned mxmask = 0xFFFF; memcpy(a + 28, &mxmask, 4);
    for (int i = 0; i < 16; i++) { memcpy(a + 160 + 16 * i, src->v[i], 16); memcpy(a + cpu()->comp[2].offset + 16 * i, src->v[i] + 16, 16); }
    memcpy(a + 512, &bv, 8);
}
TEST(ymm_xrstor_component_semantics) {
    xgate();
    void *base; uint8_t *a = xarea(&base);
    REQUIRE(a);
    static yfile in, b, out, want;
    ypat(&in, 0x710); ypat(&b, 0x711);
    static const struct { uint64_t bv; uint32_t rfbm; const char *what; } cs[] = {
        { 6, 6, "XSTATE_BV=6 RFBM=6: both lanes from memory" },
        { 2, 6, "XSTATE_BV=2 RFBM=6: AVX init -> upper zero" },
        { 4, 6, "XSTATE_BV=4 RFBM=6: SSE init -> low zero" },
        { 6, 2, "RFBM=2: upper lanes untouched" },
        { 6, 4, "RFBM=4: low lanes untouched" },
        { 0, 6, "XSTATE_BV=0 RFBM=6: all init" },
    };
    for (size_t k = 0; k < T_COUNTOF(cs); k++) {
        build_xrstor_area(a, &b, cs[k].bv);
        ymm_xrstor_rt(&in, a, cs[k].rfbm, &out);
        want = in;
        for (int i = 0; i < NY; i++) {
            if (cs[k].rfbm & 2) { if (cs[k].bv & 2) memcpy(want.v[i], b.v[i], 16); else memset(want.v[i], 0, 16); }
            if (cs[k].rfbm & 4) { if (cs[k].bv & 4) memcpy(want.v[i] + 16, b.v[i] + 16, 16); else memset(want.v[i] + 16, 0, 16); }
        }
        ycheck(&out, &want, cs[k].what);
    }
    VirtualFree(base, 0, MEM_RELEASE);
}
TEST(ymm_xinuse_tracks_avx_state) {
    ygate();
    if (!cpu_has(F_XGETBV1)) SKIP("XGETBV with ECX=1 not supported");
    static yfile in;
    uint32_t r[3] = { 0, 0, 0 };
    ypat(&in, 0x720);
    ymm_xinuse_seq(&in, r);
    t_info("xinuse_dirty", "0x%x", r[0]); t_info("xinuse_after_vzeroupper", "0x%x", r[1]); t_info("xinuse_after_vzeroall", "0x%x", r[2]);
    CHECKF(r[0] & 4, "XINUSE=0x%x with non-zero upper lanes: bit 2 must be 1", r[0]);
    CHECKF(r[0] & 2, "XINUSE=0x%x with non-zero XMM: bit 1 must be 1", r[0]);
    CHECKF(r[1] & 2, "XINUSE=0x%x after vzeroupper with non-zero XMM: bit 1 must be 1", r[1]);
    if (r[1] & 4) t_note("XINUSE[2] stays 1 after VZEROUPPER (allowed)");
    CHECK_EQ(r[0] & ~cpu()->xcr0, 0);
}

/* ======================================================================= */
/* alignment and faults (children)                                          */
/* ======================================================================= */
static T_NOINLINE void ymm_vmovdqa_load(const void *p) { __asm__ volatile("vmovdqa (%0), %%ymm0\n vzeroupper" :: "r"(p) : "memory", "xmm0"); }
static T_NOINLINE void ymm_vmovntdq_store(void *p) { __asm__ volatile("vpxor %%xmm0, %%xmm0, %%xmm0\n vmovntdq %%ymm0, (%0)\n vzeroupper" :: "r"(p) : "memory", "xmm0"); }
static T_NOINLINE void ymm_vmovaps128_load(const void *p) { __asm__ volatile("vmovaps (%0), %%xmm0\n vzeroupper" :: "r"(p) : "memory", "xmm0"); }
static T_NOINLINE void ymm_vmovdqu_load(const void *p) { __asm__ volatile("vmovdqu (%0), %%ymm0\n vzeroupper" :: "r"(p) : "memory", "xmm0"); }
static uint8_t g_falign[256] T_ALIGN(64);
CHILD(ymm_fault) {
    const char *k = argc > 1 ? argv[1] : "";
    if (!strcmp(k, "vmovdqa16")) ymm_vmovdqa_load(g_falign + 16);
    else if (!strcmp(k, "vmovdqa1")) ymm_vmovdqa_load(g_falign + 1);
    else if (!strcmp(k, "vmovntdq16")) ymm_vmovntdq_store(g_falign + 16);
    else if (!strcmp(k, "vmovaps8")) ymm_vmovaps128_load(g_falign + 8);
    else if (!strcmp(k, "guard")) {
        void *base; uint8_t *p = (uint8_t *)t_guarded_alloc(64, &base);
        if (!p) { printf("ALLOC=0\n"); return 3; }
        ymm_vmovdqu_load(p + 64 - 16);   /* 16 bytes into the no-access page */
    } else { printf("UNKNOWN=%s\n", k); return 4; }
    printf("NOFAULT=1\n");
    return 0;
}
static void fault_case(const char *kind, const char *what) {
    ygate();
    static t_child_result r;
    REQUIRE(t_child_run("ymm_fault", kind, 5000, &r));
    CHECK(!r.timed_out);
    t_info("exc_code", "0x%lx", (unsigned long)r.exc_code);
    CHECKF(r.exc_code != 0, "%s did not fault (exit=0x%lx, out: %.200s)", what, (unsigned long)r.exit_code, r.out);
    if (r.exc_code) CHECKF(r.exc_code == 0xC0000005u || r.exc_code == 0x80000002u,
                           "%s: exception 0x%lx (want STATUS_ACCESS_VIOLATION or STATUS_DATATYPE_MISALIGNMENT)", what, (unsigned long)r.exc_code);
}
TEST(ymm_fault_vmovdqa_misaligned16) { fault_case("vmovdqa16", "vmovdqa ymm at 16-byte (not 32) alignment"); }
TEST(ymm_fault_vmovdqa_misaligned1) { fault_case("vmovdqa1", "vmovdqa ymm at odd address"); }
TEST(ymm_fault_vmovntdq_misaligned) { fault_case("vmovntdq16", "vmovntdq ymm at 16-byte alignment"); }
TEST(ymm_fault_vex128_vmovaps_misaligned) { fault_case("vmovaps8", "VEX.128 vmovaps at 8-byte alignment"); }
TEST(ymm_fault_load_into_guard_page) {
    ygate();
    static t_child_result r;
    REQUIRE(t_child_run("ymm_fault", "guard", 5000, &r));
    CHECK(!r.timed_out);
    CHECKF(r.exc_code == 0xC0000005u, "vmovdqu crossing into PAGE_NOACCESS: exc=0x%lx exit=0x%lx", (unsigned long)r.exc_code, (unsigned long)r.exit_code);
}
/* VEX instructions without explicit alignment semantics accept any address (SDM 14.9) */
static T_NOINLINE void ymm_unaligned_operands(const uint8_t *p, void *o) {
    __asm__ volatile("vpxor %%xmm0, %%xmm0, %%xmm0\n"
        "vxorps 1(%[p]), %%ymm0, %%ymm1\n vmovdqu %%ymm1, 0(%[o])\n"
        "vandps 3(%[p]), %%ymm1, %%ymm2\n vmovdqu %%ymm2, 32(%[o])\n"
        "vaddps 5(%[p]), %%xmm0, %%xmm3\n vmovdqu %%ymm3, 64(%[o])\n"
        "vpor 9(%[p]), %%xmm0, %%xmm4\n vmovdqu %%ymm4, 96(%[o])\n"
        "vperm2f128 $0x21, 17(%[p]), %%ymm1, %%ymm5\n vmovdqu %%ymm5, 128(%[o])\n vzeroupper\n"
        :: [p]"r"(p), [o]"r"(o) : "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
TEST(ymm_vex_unaligned_memory_operands) {
    ygate();
    static uint8_t T_ALIGN(64) m[128], o[5][32], w[5][32];
    /* every byte < 0x40: any 4-byte window is a positive finite float, so 0.0f + x == x exactly */
    for (int k = 0; k < 128; k++) m[k] = (uint8_t)((k * 7 + 3) & 0x3F);
    ymm_unaligned_operands(m, o);
    memcpy(w[0], m + 1, 32);
    for (int k = 0; k < 32; k++) w[1][k] = m[1 + k] & m[3 + k];
    memcpy(w[2], m + 5, 16); memset(w[2] + 16, 0, 16);
    memcpy(w[3], m + 9, 16); memset(w[3] + 16, 0, 16);
    memcpy(w[4], m + 1 + 16, 16); memcpy(w[4] + 16, m + 17, 16);   /* imm 0x21: low = src1.hi, high = src2.lo */
    static const char *nm[5] = { "vxorps m256+1", "vandps m256+3", "vaddps m128+5 (VEX.128)", "vpor m128+9 (VEX.128)", "vperm2f128 m256+17" };
    for (int k = 0; k < 5; k++) ycheck_one(o[k], w[k], "%s", nm[k]);
}

/* ======================================================================= */
/* exception continuation restores the full state (children)                */
/* ======================================================================= */
typedef PVOID (WINAPI *locxs_t)(PCONTEXT, DWORD, PDWORD);
typedef BOOL  (WINAPI *getxsm_t)(PCONTEXT, PDWORD64);
typedef BOOL  (WINAPI *setxsm_t)(PCONTEXT, DWORD64);
typedef BOOL  (WINAPI *initctx_t)(PVOID, DWORD, PCONTEXT *, PDWORD);

static yfile g_cin, g_cout, g_cclob;
static void *volatile g_resume;
static DWORD g_expect_code;
static volatile long g_handled;
static int g_ctx_xstate = -1, g_ctx_hi_match = -1, g_ctx_lo_match = -1;

#ifdef T_X64
#define SET_RESUME "leaq 1f(%%rip), %%rax\n movq %%rax, (%[slot])\n"
#define AXP "%%rax"
#else
#define SET_RESUME "movl $1f, (%[slot])\n"
#define AXP "%%eax"
#endif
#define DEF_FAULT_RT(fname, insn) \
static T_NOINLINE void fname(const void *in, void *out, void *volatile *slot) { \
    __asm__ volatile(SET_RESUME YLOAD("%[in]") insn "\n1:\n" YSTORE("%[out]") "vzeroupper\n" \
        :: [in]"r"(in), [out]"r"(out), [slot]"r"(slot) : "memory", "cc", GPR_A, GPR_C, GPR_D, YCLOB); }
DEF_FAULT_RT(ymm_fault_rt_int3, "int3")
DEF_FAULT_RT(ymm_fault_rt_ud2, "ud2")
DEF_FAULT_RT(ymm_fault_rt_av, "xorl %%eax, %%eax\n movl (" AXP "), %%eax")
DEF_FAULT_RT(ymm_fault_rt_div0, "xorl %%edx, %%edx\n movl $1, %%eax\n xorl %%ecx, %%ecx\n divl %%ecx")

static void inspect_ctx(PCONTEXT ctx) {
    locxs_t loc = (locxs_t)t_proc("kernel32.dll", "LocateXStateFeature");
    getxsm_t gm = (getxsm_t)t_proc("kernel32.dll", "GetXStateFeaturesMask");
    if (!loc || !gm) return;
    DWORD64 mask = 0;
    g_ctx_xstate = (ctx->ContextFlags & CONTEXT_XSTATE) == CONTEXT_XSTATE;
    if (!g_ctx_xstate || !gm(ctx, &mask) || !(mask & 4)) return;
    DWORD hl = 0, ll = 0;
    uint8_t *hi = (uint8_t *)loc(ctx, XSTATE_AVX, &hl), *lo = (uint8_t *)loc(ctx, XSTATE_LEGACY_SSE, &ll);
    if (hi && hl >= NY * 16u) { g_ctx_hi_match = 0; for (int i = 0; i < NY; i++) g_ctx_hi_match += !memcmp(hi + 16 * i, g_cin.v[i] + 16, 16); }
    if (lo && ll >= NY * 16u) { g_ctx_lo_match = 0; for (int i = 0; i < NY; i++) g_ctx_lo_match += !memcmp(lo + 16 * i, g_cin.v[i], 16); }
}
static LONG CALLBACK ymm_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != g_expect_code || !g_resume) return EXCEPTION_CONTINUE_SEARCH;
    inspect_ctx(ep->ContextRecord);
    ymm_clobber_all(&g_cclob);        /* dirty every register the handler may legally touch */
#ifdef T_X64
    ep->ContextRecord->Rip = (DWORD64)(ULONG_PTR)g_resume;
#else
    ep->ContextRecord->Eip = (DWORD)(ULONG_PTR)g_resume;
#endif
    g_resume = NULL;
    InterlockedIncrement(&g_handled);
    return EXCEPTION_CONTINUE_EXECUTION;
}
CHILD(ymm_veh) {
    const char *k = argc > 1 ? argv[1] : "";
    if (!cpu_has(F_AVX)) { printf("NOAVX=1\n"); return 3; }
    ypat(&g_cin, 0x800); ypat(&g_cclob, 0x8FF); memset(&g_cout, 0, sizeof g_cout);
    void (*fn)(const void *, void *, void *volatile *) = NULL;
    if (!strcmp(k, "int3")) { fn = ymm_fault_rt_int3; g_expect_code = STATUS_BREAKPOINT; }
    else if (!strcmp(k, "ud2")) { fn = ymm_fault_rt_ud2; g_expect_code = STATUS_ILLEGAL_INSTRUCTION; }
    else if (!strcmp(k, "av")) { fn = ymm_fault_rt_av; g_expect_code = STATUS_ACCESS_VIOLATION; }
    else if (!strcmp(k, "div0")) { fn = ymm_fault_rt_div0; g_expect_code = STATUS_INTEGER_DIVIDE_BY_ZERO; }
    else { printf("UNKNOWN=%s\n", k); return 4; }
    unsigned mx0 = t_get_mxcsr();
    PVOID h = AddVectoredExceptionHandler(1, ymm_veh);
    fn(&g_cin, &g_cout, &g_resume);
    RemoveVectoredExceptionHandler(h);
    int mism = 0, first = -1, hi_bad = 0, lo_bad = 0;
    for (int i = 0; i < NY; i++) {
        int lb = memcmp(g_cout.v[i], g_cin.v[i], 16) != 0, hb = memcmp(g_cout.v[i] + 16, g_cin.v[i] + 16, 16) != 0;
        lo_bad += lb; hi_bad += hb;
        if (lb || hb) { mism++; if (first < 0) first = i; }
    }
    printf("HANDLED=%ld\nMISMATCH=%d\nLO_BAD=%d\nHI_BAD=%d\nFIRST_BAD=%d\n", g_handled, mism, lo_bad, hi_bad, first);
    if (first >= 0) {
        char a[128], b[128];
        printf("GOT=%s\nWANT=%s\n", t_hex_lanes(a, sizeof a, g_cout.v[first], 32, 8), t_hex_lanes(b, sizeof b, g_cin.v[first], 32, 8));
    }
    printf("CTX_XSTATE=%d\nCTX_HI_MATCH=%d\nCTX_LO_MATCH=%d\nMXCSR_SAME=%d\n", g_ctx_xstate, g_ctx_hi_match, g_ctx_lo_match, t_get_mxcsr() == mx0);
    fflush(stdout);
    return 0;
}
static int child_int(const t_child_result *r, const char *key, int dflt) {
    char v[64]; return t_child_find(r, key, v, sizeof v) ? atoi(v) : dflt;
}
static void veh_case(const char *kind) {
    ygate();
    static t_child_result r;
    REQUIRE(t_child_run("ymm_veh", kind, 10000, &r));
    CHECK(!r.timed_out);
    CHECKF(r.exit_code == 0, "child exit=0x%lx exc=0x%lx out: %.300s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    if (r.exit_code != 0) return;
    CHECK_EQ(child_int(&r, "HANDLED=", -1), 1);
    int lo = child_int(&r, "LO_BAD=", -1), hi = child_int(&r, "HI_BAD=", -1);
    char g[128] = "", w[128] = "";
    t_child_find(&r, "GOT=", g, sizeof g); t_child_find(&r, "WANT=", w, sizeof w);
    CHECKF(lo == 0, "%s: %d XMM low halves differ after EXCEPTION_CONTINUE_EXECUTION (first ymm%d got %s want %s)",
           kind, lo, child_int(&r, "FIRST_BAD=", -1), g, w);
    int cx = child_int(&r, "CTX_XSTATE=", -1), chm = child_int(&r, "CTX_HI_MATCH=", -1), clm = child_int(&r, "CTX_LO_MATCH=", -1);
    t_info("upper_halves_lost", "%d", hi);
#ifdef T_X86
    /* 32-bit: on 64-bit Windows the exception reaches 32-bit code through the WoW64 layer, which converts
     * the native context to a 32-bit CONTEXT; whether that conversion carries CONTEXT_XSTATE is not a
     * documented contract. Hard-check the upper halves only when the ContextRecord the OS handed to the
     * handler carries XSTATE (then NtContinue of that record must restore them). */
    if (cx != 1) {
        if (hi) t_note("%s: %d YMM upper halves not restored and the 32-bit ContextRecord carries no CONTEXT_XSTATE", kind, hi);
        hi = 0;
    }
#endif
    if (hi != 0 && t_is_wine())
        t_expect_fail("Wine: exception continuation does not restore YMM upper halves (%d differ)", hi);
    CHECKF(hi == 0, "%s: %d YMM upper halves differ after EXCEPTION_CONTINUE_EXECUTION (first ymm%d got %s want %s)",
           kind, hi, child_int(&r, "FIRST_BAD=", -1), g, w);
    CHECK_EQ(child_int(&r, "MXCSR_SAME=", 0), 1);
    t_info("ctx_xstate", "%d", cx); t_info("ctx_hi_match", "%d", chm); t_info("ctx_lo_match", "%d", clm);
    if (cx == 1 && chm >= 0) CHECKF(chm == NY, "exception CONTEXT carries XSTATE but only %d/%d upper halves match the faulting state", chm, NY);
    if (cx == 1 && clm >= 0) CHECKF(clm == NY, "exception CONTEXT XMM: only %d/%d low halves match", clm, NY);
    if (cx != 1) t_note("exception ContextRecord has no CONTEXT_XSTATE (flags lack 0x40)");
}
TEST(ymm_exception_int3_continue_restores) { veh_case("int3"); }
TEST(ymm_exception_ud2_continue_restores) { veh_case("ud2"); }
TEST(ymm_exception_av_continue_restores) { veh_case("av"); }
TEST(ymm_exception_div0_continue_restores) { veh_case("div0"); }

/* ======================================================================= */
/* worker groups: preemption, suspension, migration, isolation              */
/* ======================================================================= */
#define SPIN_MAX 400000000UL
typedef struct yw {
    yfile in, out, bad_got;
    volatile long ready;
    int mode;               /* 0 = spin until stop, 1 = churn (load/spin/store/compare repeatedly),
                               2 = spin after VZEROUPPER (uppers 0), 3 = spin after VZEROALL (all 0) */
    unsigned long left;
    long iters, bad;
    HANDLE h;
} yw;
static volatile long g_ystop;
static const long g_zero_flag = 0;
static DWORD WINAPI yw_main(void *p) {
    yw *w = (yw *)p;
    if (w->mode == 0) {
        w->left = ymm_spin_ready(&w->in, &w->out, &w->ready, &g_ystop, SPIN_MAX);
    } else if (w->mode >= 2) {
        w->left = ymm_spin_ready_clean(&w->in, &w->out, &w->ready, &g_ystop, SPIN_MAX, w->mode - 1);
    } else {
        volatile long dummy = 0;
        w->ready = 1;
        while (!g_ystop) {
            ymm_spin_ready(&w->in, &w->out, &dummy, &g_zero_flag, 3000);
            if (memcmp(&w->out, &w->in, NY * 32)) { if (!w->bad) w->bad_got = w->out; w->bad++; }
            w->iters++;
            if (!(w->iters & 3)) SwitchToThread();
        }
    }
    return 0;
}
static yw *yw_alloc(int n) { return (yw *)VirtualAlloc(NULL, (size_t)n * sizeof(yw), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }
static int yw_start(yw *w, int n, int mode_mix, uint32_t tagbase) {
    g_ystop = 0;
    int started = 0;
    for (int i = 0; i < n; i++) {
        ypat(&w[i].in, tagbase + (uint32_t)i);
        /* mode_mix: 0/1 = all that mode; 2 = alternate spin/churn; 3 = cycle dirty / VZEROUPPER / VZEROALL spinners */
        static const int cyc3[3] = { 0, 2, 3 };
        w[i].mode = mode_mix == 2 ? (i & 1) : mode_mix == 3 ? cyc3[i % 3] : mode_mix;
        w[i].h = CreateThread(NULL, 64 * 1024, yw_main, &w[i], STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (!w[i].h) break;
        started++;
    }
    return started;
}
static int yw_wait_ready(yw *w, int n, DWORD ms) {
    double t0 = t_now_ms();
    for (;;) {
        int r = 0;
        for (int i = 0; i < n; i++) r += w[i].ready != 0;
        if (r == n) return r;
        if (t_now_ms() - t0 > ms) return r;
        Sleep(1);
    }
}
static void yw_stop_join(yw *w, int n) {
    InterlockedExchange(&g_ystop, 1);
    for (int i = 0; i < n; i++) {
        if (!w[i].h) continue;
        DWORD r = WaitForSingleObject(w[i].h, 15000);
        CHECKF(r == WAIT_OBJECT_0, "worker %d did not exit (wait=%lu)", i, (unsigned long)r);
        CloseHandle(w[i].h); w[i].h = NULL;
    }
}
static int yw_verify(yw *w, int n, const char *label) {
    int bad = 0; long iters = 0;
    for (int i = 0; i < n; i++) {
        char l[96]; snprintf(l, sizeof l, "%s (thread %d)", label, i);
        if (w[i].mode != 1) {
            CHECKF(w[i].left != 0, "thread %d never observed the stop flag", i);
            static yfile want;
            want = w[i].in;
            if (w[i].mode == 2) for (int r = 0; r < NY; r++) memset(want.v[r] + 16, 0, 16);
            if (w[i].mode == 3) memset(&want, 0, sizeof want);
            if (w[i].mode >= 2) snprintf(l, sizeof l, "%s (thread %d, %s)", label, i, w[i].mode == 2 ? "VZEROUPPER-clean" : "VZEROALL-clean");
            bad += ycheck(&w[i].out, &want, l) != 0;
        } else {
            iters += w[i].iters;
            CHECKF(w[i].bad == 0, "thread %d: %ld of %ld load/spin/store rounds lost register state", i, w[i].bad, w[i].iters);
            if (w[i].bad) ycheck(&w[i].bad_got, &w[i].in, l);
            bad += w[i].bad != 0;
        }
    }
    if (iters) t_info("churn_rounds", "%ld", iters);
    return bad;
}
enum { D_NONE = 0, D_SUSPEND = 1, D_CTX_FULL = 2, D_SETCTX = 4, D_AFFINITY = 8, D_PRIORITY = 16 };
static int g_dist_errors;
static void yw_disturb(yw *w, int n, int ms, unsigned flags) {
    const t_env *e = t_env_get();
    unsigned long long aff = e->affinity ? e->affinity : 1;
    int bits[64], nb = 0;
    for (int b = 0; b < 64; b++) if (aff >> b & 1) bits[nb++] = b;
    double t0 = t_now_ms();
    int k = 0, ops = 0;
    g_dist_errors = 0;
    while (t_now_ms() - t0 < ms) {
        for (int i = 0; i < n; i++) {
            HANDLE h = w[i].h;
            if (!h) continue;
            if (flags & (D_SUSPEND | D_CTX_FULL | D_SETCTX)) {
                if (SuspendThread(h) == (DWORD)-1) { g_dist_errors++; continue; }
                if (flags & (D_CTX_FULL | D_SETCTX)) {
                    CONTEXT c; memset(&c, 0, sizeof c);
                    c.ContextFlags = (flags & D_SETCTX) ? (CONTEXT_CONTROL | CONTEXT_INTEGER) : (CONTEXT_FULL | CONTEXT_FLOATING_POINT);
                    if (!GetThreadContext(h, &c)) g_dist_errors++;
                    else if ((flags & D_SETCTX) && !SetThreadContext(h, &c)) g_dist_errors++;
                }
                ResumeThread(h);
                ops++;
            }
            if (flags & D_AFFINITY) { if (!SetThreadAffinityMask(h, (DWORD_PTR)1 << bits[(i + k) % nb])) g_dist_errors++; ops++; }
            if (flags & D_PRIORITY) {
                static const int pr[] = { THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_LOWEST };
                if (!SetThreadPriority(h, pr[(i + k) & 3])) g_dist_errors++;
                ops++;
            }
        }
        k++;
        Sleep(flags ? 2 : 10);
    }
    t_info("disturb_ops", "%d", ops);
    if (g_dist_errors) t_note("%d disturbance API calls failed", g_dist_errors);
}
static void group_case(int n, int mode_mix, int ms, unsigned flags, const char *label) {
    ygate();
    yw *w = yw_alloc(n);
    if (!w) SKIP("cannot allocate %d worker records", n);
    int s = yw_start(w, n, mode_mix, 0x1000);
    int r = yw_wait_ready(w, s, 10000);
    t_info("threads", "%d", s); t_info("ready", "%d", r);
    if (s > 0) yw_disturb(w, s, ms, flags);
    yw_stop_join(w, s);
    yw_verify(w, s, label);
    VirtualFree(w, 0, MEM_RELEASE);
    if (s < n) SKIP("only %d of %d threads could be created", s, n);
}
TEST(ymm_preempt_spin_2_threads) { group_case(2, 0, 150, D_NONE, "spin, 2 threads"); }
TEST(ymm_preempt_spin_16_threads) { group_case(16, 0, 200, D_NONE, "spin, 16 threads"); }
TEST(ymm_preempt_spin_64_threads) { group_case(64, 0, 250, D_NONE, "spin, 64 threads"); }
TEST(ymm_preempt_spin_256_threads) { group_case(256, 0, 400, D_NONE, "spin, 256 threads"); }
TEST_MANUAL(ymm_preempt_spin_1024_threads) { group_case(1024, 0, 1000, D_NONE, "spin, 1024 threads"); }
TEST(ymm_preempt_oversubscribed) {
    int n = 4 * t_cpu_count(); if (n < 8) n = 8; if (n > 256) n = 256;
    group_case(n, 0, 300, D_NONE, "spin, 4x oversubscribed");
}
TEST(ymm_suspend_resume_spinning) { group_case(4, 0, 300, D_SUSPEND, "spin + Suspend/ResumeThread"); }
TEST(ymm_getcontext_spinning_no_xstate) { group_case(4, 0, 300, D_CTX_FULL, "spin + GetThreadContext(FULL|FP)"); }
TEST(ymm_setcontext_control_spinning) { group_case(4, 0, 300, D_SETCTX, "spin + Get/SetThreadContext(CONTROL|INTEGER)"); }
TEST(ymm_affinity_migration_spinning) {
    if (t_cpu_count() < 2) SKIP("single logical processor: no migration possible");
    group_case(4, 0, 300, D_AFFINITY, "spin + affinity migration");
}
TEST(ymm_priority_changes_spinning) { group_case(8, 0, 300, D_PRIORITY, "spin + priority changes"); }
TEST(ymm_isolation_churn_8_threads) { group_case(8, 1, 300, D_NONE, "churn, 8 threads"); }
TEST(ymm_isolation_churn_64_threads) { group_case(64, 2, 400, D_NONE, "churn+spin, 64 threads"); }
TEST(ymm_isolation_churn_with_suspend) { group_case(8, 2, 300, D_SUSPEND | D_PRIORITY, "churn+spin, suspend+priority"); }
/* Threads whose AVX component is initial (after VZEROUPPER / VZEROALL) interleaved with dirty threads on
 * oversubscribed CPUs: a clean thread must come back with zero upper lanes (zero registers), never with
 * a dirty neighbour's values (init-optimised XSAVEOPT/XSAVEC + XRSTOR must re-initialise, SDM 13.8). */
TEST(ymm_clean_upper_threads_stay_clean) {
    int n = 3 * t_cpu_count(); if (n < 12) n = 12; if (n > 192) n = 192;
    group_case(n, 3, 400, D_NONE, "dirty/VZEROUPPER/VZEROALL spinners, oversubscribed");
}
TEST(ymm_clean_upper_threads_suspend_context) {
    group_case(9, 3, 300, D_SUSPEND | D_CTX_FULL, "dirty/clean spinners + Suspend/GetThreadContext");
}
TEST(ymm_clean_upper_threads_migration) {
    if (t_cpu_count() < 2) SKIP("single logical processor: no migration possible");
    group_case(6, 3, 300, D_AFFINITY | D_PRIORITY, "dirty/clean spinners + affinity migration");
}

/* ---- full-state preservation across a special user APC that interrupts a spinning thread ---- */
typedef BOOL (WINAPI *qapc2_t)(PAPCFUNC, HANDLE, ULONG_PTR, int);
static volatile long g_apc_ran;
static yfile g_apc_clob;
static void CALLBACK ymm_apc_clobber(ULONG_PTR p) { (void)p; ymm_clobber_all(&g_apc_clob); InterlockedIncrement(&g_apc_ran); }
TEST(ymm_special_apc_interrupts_spin) {
    ygate();
    qapc2_t q2 = (qapc2_t)t_proc("kernel32.dll", "QueueUserAPC2");
    if (!q2) SKIP("QueueUserAPC2 not exported (special user APCs need Windows 11 / Server 2022)");
    yw *w = yw_alloc(1);
    REQUIRE(w);
    ypat(&g_apc_clob, 0x9FF);
    g_apc_ran = 0;
    int s = yw_start(w, 1, 0, 0x900);
    int ready = s ? yw_wait_ready(w, 1, 5000) : 0;
    int queued = 0, err = 0;
    if (ready) {
        for (int k = 0; k < 10; k++) {
            if (q2(ymm_apc_clobber, w[0].h, 0, 1 /* QUEUE_USER_APC_FLAGS_SPECIAL_USER_APC */)) queued++;
            else { err = (int)GetLastError(); break; }
            double t0 = t_now_ms();
            while (g_apc_ran < queued && t_now_ms() - t0 < 3000) Sleep(1);
            if (g_apc_ran < queued) break;   /* not delivered: do not keep the spinner waiting */
        }
    }
    yw_stop_join(w, s);
    t_info("apc_queued", "%d", queued); t_info("apc_ran", "%ld", g_apc_ran);
    if (s && ready && queued == 0) {
        VirtualFree(w, 0, MEM_RELEASE);
        SKIP("QueueUserAPC2(SPECIAL_USER_APC) failed with error %d: special user APCs unsupported", err);
    }
    REQUIRE(s == 1 && ready);
    if (t_is_wine() && g_apc_ran < queued) t_expect_fail("Wine: special user APC not delivered to a running thread");
    CHECKF(g_apc_ran == queued, "special user APC ran %ld of %d times within 3 s each", g_apc_ran, queued);
#ifdef T_X64
    yw_verify(w, 1, "spin interrupted by special user APCs (APC clobbers registers)");
#else
    /* 32-bit under WoW64: the APC context is converted by the WoW64 layer; only the legacy XMM state of a
     * 32-bit CONTEXT is documented to round-trip, so the upper lanes are recorded, not asserted. */
    CHECK(w[0].left != 0);
    ycheck_part(&w[0].out, &w[0].in, 0, 16, 0, -1, "spin interrupted by special user APCs");
    int up = 0;
    for (int i = 0; i < NY; i++) up += !memcmp(w[0].out.v[i] + 16, w[0].in.v[i] + 16, 16);
    t_info("upper_halves_preserved", "%d/%d", up, NY);
    if (up != NY) t_note("x86: %d of %d YMM upper halves not preserved across a special user APC", NY - up, NY);
#endif
    VirtualFree(w, 0, MEM_RELEASE);
}

/* ---- XSTATE thread context of a spinning thread ---- */
typedef struct xapi { initctx_t init; locxs_t loc; getxsm_t gm; setxsm_t sm; } xapi;
static int xapi_get(xapi *a) {
    a->init = (initctx_t)t_proc("kernel32.dll", "InitializeContext");
    a->loc = (locxs_t)t_proc("kernel32.dll", "LocateXStateFeature");
    a->gm = (getxsm_t)t_proc("kernel32.dll", "GetXStateFeaturesMask");
    a->sm = (setxsm_t)t_proc("kernel32.dll", "SetXStateFeaturesMask");
    return a->init && a->loc && a->gm && a->sm;
}
#ifdef T_X64
#define XCTX_FLAGS (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT | CONTEXT_XSTATE)
#else
#define XCTX_FLAGS (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS | CONTEXT_XSTATE)
#endif
static void xctx_case(int do_set) {
    ygate();
    xapi a;
    if (!xapi_get(&a)) SKIP("XSTATE context APIs not exported");
    DWORD len = 0;
    BOOL ok = a.init(NULL, XCTX_FLAGS, NULL, &len);
    DWORD le = GetLastError();
    CHECKF(!ok && le == ERROR_INSUFFICIENT_BUFFER && len > sizeof(CONTEXT), "InitializeContext(NULL) ok=%d err=%lu len=%lu", ok, (unsigned long)le, (unsigned long)len);
    REQUIRE(len > 0 && len < (1u << 20));
    void *buf = VirtualAlloc(NULL, len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(buf);
    PCONTEXT ctx = NULL;
    ok = a.init(buf, XCTX_FLAGS, &ctx, &len);
    if (!ok) { VirtualFree(buf, 0, MEM_RELEASE); REQUIREF(ok, "InitializeContext failed: %lu", (unsigned long)GetLastError()); }
    CHECK(a.sm(ctx, XSTATE_MASK_LEGACY | XSTATE_MASK_AVX));
    yw *w = yw_alloc(1);
    REQUIRE(w);
    int s = yw_start(w, 1, 0, 0xA00);
    int ready = s ? yw_wait_ready(w, 1, 5000) : 0;
    int got = 0, set_ok = 0, hi_ok = 0, lo_ok = 0;
    DWORD64 mask = 0; DWORD hl = 0, ll = 0;
    static yfile newhi;
    ypat(&newhi, 0xAFF);
    static uint8_t snap_hi[16][16], snap_lo[16][16];
    if (ready) {
        Sleep(20);
        if (SuspendThread(w[0].h) != (DWORD)-1) {
            got = GetThreadContext(w[0].h, ctx);
            if (got) {
                a.gm(ctx, &mask);
                uint8_t *hi = (uint8_t *)a.loc(ctx, XSTATE_AVX, &hl);
                uint8_t *lo = (uint8_t *)a.loc(ctx, XSTATE_LEGACY_SSE, &ll);
                if (hi && hl >= NY * 16u) { hi_ok = 1; for (int i = 0; i < NY; i++) memcpy(snap_hi[i], hi + 16 * i, 16); }
                if (lo && ll >= NY * 16u) { lo_ok = 1; for (int i = 0; i < NY; i++) memcpy(snap_lo[i], lo + 16 * i, 16); }
                if (do_set && hi_ok) {
                    for (int i = 0; i < NY; i++) memcpy(hi + 16 * i, newhi.v[i] + 16, 16);
                    a.sm(ctx, mask | XSTATE_MASK_AVX);
                    set_ok = SetThreadContext(w[0].h, ctx);
                }
            }
            ResumeThread(w[0].h);
        }
    }
    yw_stop_join(w, s);
    REQUIRE(s == 1 && ready);
    t_info("ctx_len", "%lu", (unsigned long)len); t_info("xstate_mask", "0x%llx", (unsigned long long)mask);
    t_info("avx_len", "%lu", (unsigned long)hl); t_info("sse_len", "%lu", (unsigned long)ll);
    int wine = t_is_wine();
#ifdef T_X86
    if (t_env_get()->wow64 && (!got || !(mask & XSTATE_MASK_AVX))) {
        VirtualFree(w, 0, MEM_RELEASE); VirtualFree(buf, 0, MEM_RELEASE);
        SKIP("WoW64: GetThreadContext(CONTEXT_XSTATE) of a 32-bit thread %s (no documented WoW64 contract)",
             got ? "returned no AVX component" : "failed");
    }
#endif
    if (!got && wine) t_expect_fail("Wine: GetThreadContext with CONTEXT_XSTATE failed");
    CHECKF(got, "GetThreadContext(CONTEXT_XSTATE) failed: %lu", (unsigned long)GetLastError());
    CHECKF(mask & XSTATE_MASK_AVX, "GetXStateFeaturesMask=0x%llx: AVX not reported for a thread with non-zero upper lanes", (unsigned long long)mask);
    CHECK(hi_ok); CHECK(lo_ok);
    for (int i = 0; i < NY && hi_ok; i++) { char l[64]; snprintf(l, sizeof l, "CONTEXT YMM%d_H of spinning thread", i); CHECK_REG(l, snap_hi[i], w[0].in.v[i] + 16, 16); }
    for (int i = 0; i < NY && lo_ok; i++) { char l[64]; snprintf(l, sizeof l, "CONTEXT XMM%d of spinning thread", i); CHECK_REG(l, snap_lo[i], w[0].in.v[i], 16); }
    if (do_set) {
        CHECKF(set_ok, "SetThreadContext(CONTEXT_XSTATE) failed: %lu", (unsigned long)GetLastError());
        static yfile want;
        want = w[0].in;
        if (set_ok) for (int i = 0; i < NY; i++) memcpy(want.v[i] + 16, newhi.v[i] + 16, 16);
        ycheck(&w[0].out, &want, "after SetThreadContext with new upper lanes");
    } else {
        yw_verify(w, 1, "after GetThreadContext(CONTEXT_XSTATE)");
    }
    VirtualFree(w, 0, MEM_RELEASE);
    VirtualFree(buf, 0, MEM_RELEASE);
}
TEST(ymm_context_get_xstate_spinning) { xctx_case(0); }
TEST(ymm_context_set_xstate_upper_spinning) { xctx_case(1); }

/* XSTATE context round trip of a thread whose upper lanes are architecturally zero (VZEROUPPER), and
 * SetThreadContext of a dirty thread with the AVX bit cleared in the XSTATE mask.
 *   clean=1: GetThreadContext; if the mask reports AVX the context's YMM_H must be zero (it describes the
 *            thread); SetThreadContext of the unchanged record must not invent upper-lane data.
 *   clean=0: dirty thread, SetXStateFeaturesMask(mask & ~AVX) then SetThreadContext: the thread's upper
 *            lanes end up either unchanged or initial (zero) - which one is recorded (INFO), anything else
 *            (another thread's or stale buffer data) is a FAIL. Low lanes must be unchanged either way. */
static void xctx_clean_case(int clean) {
    ygate();
    xapi a;
    if (!xapi_get(&a)) SKIP("XSTATE context APIs not exported");
    DWORD len = 0;
    a.init(NULL, XCTX_FLAGS, NULL, &len);
    REQUIRE(len > 0 && len < (1u << 20));
    void *buf = VirtualAlloc(NULL, len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(buf);
    PCONTEXT ctx = NULL;
    if (!a.init(buf, XCTX_FLAGS, &ctx, &len)) { VirtualFree(buf, 0, MEM_RELEASE); REQUIREF(0, "InitializeContext failed: %lu", (unsigned long)GetLastError()); }
    CHECK(a.sm(ctx, XSTATE_MASK_LEGACY | XSTATE_MASK_AVX));
    yw *w = yw_alloc(1);
    REQUIRE(w);
    g_ystop = 0;
    ypat(&w[0].in, clean ? 0xA40u : 0xA50u);
    w[0].mode = clean ? 2 : 0;
    w[0].h = CreateThread(NULL, 64 * 1024, yw_main, &w[0], STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    int s = w[0].h != NULL;
    int ready = s ? yw_wait_ready(w, 1, 5000) : 0;
    int got = 0, set_ok = 0, hi_ok = 0, hi_zero = -1;
    DWORD64 mask = 0, mask_after = 0; DWORD hl = 0;
    if (ready) {
        Sleep(20);
        if (SuspendThread(w[0].h) != (DWORD)-1) {
            got = GetThreadContext(w[0].h, ctx);
            if (got) {
                a.gm(ctx, &mask);
                uint8_t *hi = (uint8_t *)a.loc(ctx, XSTATE_AVX, &hl);
                hi_ok = hi && hl >= NY * 16u;
                if (hi_ok && (mask & XSTATE_MASK_AVX)) {
                    hi_zero = 1;
                    for (int i = 0; i < NY; i++) if (memcmp(hi + 16 * i, g_zero32, 16)) hi_zero = 0;
                }
                if (!clean && hi_ok) {
                    /* poison the (now unflagged) AVX area: if the OS still consumes it, the thread gets 0x5A.. */
                    memset(hi, 0x5A, NY * 16u);
                    a.sm(ctx, mask & ~(DWORD64)XSTATE_MASK_AVX);
                }
                a.gm(ctx, &mask_after);
                set_ok = SetThreadContext(w[0].h, ctx);
            }
            ResumeThread(w[0].h);
        }
    }
    yw_stop_join(w, s);
    REQUIRE(s == 1 && ready);
    t_info("xstate_mask", "0x%llx", (unsigned long long)mask);
    t_info("xstate_mask_set", "0x%llx", (unsigned long long)mask_after);
#ifdef T_X86
    if (t_env_get()->wow64 && (!got || !hi_ok)) {
        VirtualFree(w, 0, MEM_RELEASE); VirtualFree(buf, 0, MEM_RELEASE);
        SKIP("WoW64: GetThreadContext(CONTEXT_XSTATE) of a 32-bit thread %s (no documented WoW64 contract)", got ? "has no AVX area" : "failed");
    }
#endif
    if (!got && t_is_wine()) t_expect_fail("Wine: GetThreadContext with CONTEXT_XSTATE failed");
    REQUIREF(got, "GetThreadContext(CONTEXT_XSTATE) failed: %lu", (unsigned long)GetLastError());
    CHECK(hi_ok);
    CHECKF(set_ok, "SetThreadContext(CONTEXT_XSTATE) failed: %lu", (unsigned long)GetLastError());
    CHECK(w[0].left != 0);
    /* low lanes always intact */
    ycheck_part(&w[0].out, &w[0].in, 0, 16, 0, -1, clean ? "clean thread after Get/SetThreadContext(XSTATE)" : "dirty thread after SetThreadContext(mask without AVX)");
    if (clean) {
        if (mask & XSTATE_MASK_AVX) {
            CHECKF(hi_zero == 1, "XSTATE mask reports AVX for a thread after VZEROUPPER, but the context's YMM_H is not zero");
        } else t_info("avx_bit_clear_for_clean_thread", "1");
        ycheck_part(&w[0].out, &g_zfile, 16, 16, 0, -1, "clean thread uppers after Get/SetThreadContext(XSTATE)");
    } else {
        CHECKF(mask & XSTATE_MASK_AVX, "GetXStateFeaturesMask=0x%llx lacks AVX for a thread with non-zero upper lanes", (unsigned long long)mask);
        int kept = 0, zero = 0, other = 0;
        for (int i = 0; i < NY; i++) {
            if (!memcmp(w[0].out.v[i] + 16, w[0].in.v[i] + 16, 16)) kept++;
            else if (!memcmp(w[0].out.v[i] + 16, g_zero32, 16)) zero++;
            else other++;
        }
        t_info("uppers_kept", "%d", kept); t_info("uppers_initialised", "%d", zero);
        if (zero == NY) t_note("SetThreadContext with AVX cleared in the XSTATE mask initialised the upper lanes");
        else if (kept == NY) t_note("SetThreadContext with AVX cleared in the XSTATE mask left the upper lanes unchanged");
        CHECKF(other == 0 && (kept == NY || zero == NY),
               "upper lanes after SetThreadContext(mask without AVX): kept=%d zeroed=%d other=%d (want all kept or all initial)", kept, zero, other);
        if (other) ycheck_part(&w[0].out, &w[0].in, 16, 16, 0, -1, "dirty thread uppers (mask without AVX)");
    }
    VirtualFree(w, 0, MEM_RELEASE);
    VirtualFree(buf, 0, MEM_RELEASE);
}
TEST(ymm_context_xstate_clean_upper_thread) { xctx_clean_case(1); }
TEST(ymm_context_set_xstate_mask_without_avx) { xctx_clean_case(0); }

/* ---- a new thread never sees another thread's register values ---- */
static yfile g_probe[8];
static DWORD WINAPI probe_thread(void *p) { ymm_store_all(p); return 0; }
static volatile int g_probe_n;
static void cb_create_probes(void *arg) {
    (void)arg;
    for (int i = 0; i < 8; i++) {
        HANDLE h = CreateThread(NULL, 64 * 1024, probe_thread, &g_probe[i], STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (!h) break;
        WaitForSingleObject(h, 5000); CloseHandle(h);
        g_probe_n++;
    }
}
TEST(ymm_new_thread_no_foreign_state) {
    ygate();
    yw *w = yw_alloc(4);
    REQUIRE(w);
    int s = yw_start(w, 4, 0, 0xB00);
    yw_wait_ready(w, s, 5000);
    static t_vregs in, out;
    t_vregs_pattern(&in, T_VK_YMM, 0xB10);
    memset(g_probe, 0, sizeof g_probe); g_probe_n = 0;
    t_vregs_call(T_VK_YMM, &in, &out, cb_create_probes, NULL);
    yw_stop_join(w, s);
    VirtualFree(w, 0, MEM_RELEASE);
    CHECK(g_probe_n > 0);
    int leaks = 0, nz_hi = 0;
    for (int p = 0; p < g_probe_n; p++) {
        for (int r = 0; r < NY; r++) {
            if (memcmp(g_probe[p].v[r] + 16, g_zero32, 16)) nz_hi++;
            for (int q = 0; q < 4; q++) {
                uint64_t v = rd64(g_probe[p].v[r], q);
                if ((v >> 56 & 0x80) == 0 || (v >> 48 & 0xC0) != 0x40) continue;
                uint32_t reg = (uint32_t)(v >> 56) & 0x7F, wd = (uint32_t)(v >> 48) & 0x3F;
                int hit = 0;
                for (uint32_t t = 0; t < 4 && !hit; t++) hit = v == t_pattern_word(0xB00 + t, reg, wd);
                hit = hit || v == t_pattern_word(0xB10, reg, wd);
                if (hit) { leaks++; t_note("new thread %d ymm%d qword %d holds a value of another thread: 0x%016llx", p, r, q, (unsigned long long)v); }
            }
        }
    }
    CHECKF(leaks == 0, "%d register qwords of new threads carry values from other threads", leaks);
    t_info("probes", "%d", g_probe_n); t_info("nonzero_upper_lanes_at_start", "%d", nz_hi);
    if (nz_hi) t_note("new threads start with %d non-zero YMM upper halves (no foreign data)", nz_hi);
}

/* ======================================================================= */
/* calls into the OS: x64 ABI keeps XMM6-15[127:0]; upper halves volatile   */
/* ======================================================================= */
static void ycall_case_ex(void (*fn)(void *), void *arg, const char *label, int expect_full) {
    ygate();
    static t_vregs in, out;
    t_vregs_pattern(&in, T_VK_YMM, 0xC00);
    REQUIRE(t_vregs_call(T_VK_YMM, &in, &out, fn, arg) == 0);
    int up = 0, lo05 = 0, lo615 = 0;
    for (int i = 0; i < NY; i++) {
        up += !memcmp(out.v[i] + 16, in.v[i] + 16, 16);
        if (i < 6) lo05 += !memcmp(out.v[i], in.v[i], 16); else lo615 += !memcmp(out.v[i], in.v[i], 16);
    }
    t_info("upper_halves_preserved", "%d/%d", up, NY);
    t_info("xmm0_5_preserved", "%d/6", lo05);
    if (expect_full) { t_vregs_check(&out, &in, T_VK_YMM, 0, -1, label); return; }
#ifdef T_X64
    t_vregs_check(&out, &in, T_VK_XMM, 6, 15, label);   /* Windows x64 ABI: nonvolatile XMM6-15 low halves */
#else
    t_info("xmm6_7_preserved", "%d/2", lo615);
    t_note("x86 ABI: every XMM/YMM register is volatile across calls; preservation recorded only");
    CHECK(1);
#endif
}
static void ycall_case(void (*fn)(void *), void *arg, const char *label) { ycall_case_ex(fn, arg, label, 0); }

TEST(ymm_call_null_full_preserved) { ycall_case_ex(NULL, NULL, "no call", 1); }
static void cb_sleep0(void *a) { (void)a; for (int i = 0; i < 20; i++) Sleep(0); }
static void cb_sleep1(void *a) { (void)a; for (int i = 0; i < 5; i++) Sleep(1); }
static void cb_switch(void *a) { (void)a; for (int i = 0; i < 50; i++) SwitchToThread(); }
static void cb_wait_signaled(void *a) { HANDLE e = (HANDLE)a; for (int i = 0; i < 50; i++) WaitForSingleObject(e, 0); }
static void cb_wait_timeout(void *a) { HANDLE e = (HANDLE)a; for (int i = 0; i < 4; i++) WaitForSingleObject(e, 5); }
static void cb_wait_multiple(void *a) { HANDLE *h = (HANDLE *)a; for (int i = 0; i < 4; i++) WaitForMultipleObjects(2, h, FALSE, 3); }
TEST(ymm_call_sleep0) { ycall_case(cb_sleep0, NULL, "across Sleep(0)"); }
TEST(ymm_call_sleep1) { ycall_case(cb_sleep1, NULL, "across Sleep(1)"); }
TEST(ymm_call_switchtothread) { ycall_case(cb_switch, NULL, "across SwitchToThread"); }
TEST(ymm_call_wait_signaled) {
    HANDLE e = CreateEventA(NULL, TRUE, TRUE, NULL); REQUIRE(e);
    ycall_case(cb_wait_signaled, e, "across WaitForSingleObject(signaled)"); CloseHandle(e);
}
TEST(ymm_call_wait_timeout) {
    HANDLE e = CreateEventA(NULL, TRUE, FALSE, NULL); REQUIRE(e);
    ycall_case(cb_wait_timeout, e, "across WaitForSingleObject(timeout)"); CloseHandle(e);
}
TEST(ymm_call_wait_multiple_timeout) {
    HANDLE h[2] = { CreateEventA(NULL, TRUE, FALSE, NULL), CreateEventA(NULL, FALSE, FALSE, NULL) };
    REQUIRE(h[0] && h[1]);
    ycall_case(cb_wait_multiple, h, "across WaitForMultipleObjects(timeout)"); CloseHandle(h[0]); CloseHandle(h[1]);
}

/* helper thread that dirties its own YMM state between handshakes */
typedef struct pp { HANDLE go, done, mtx, held; volatile long quit; int mode; yfile pat; } pp;
static DWORD WINAPI pp_thread(void *p) {
    pp *x = (pp *)p;
    if (x->mode == 0) {
        while (WaitForSingleObject(x->go, 5000) == WAIT_OBJECT_0 && !x->quit) { ymm_clobber_all(&x->pat); SetEvent(x->done); }
    } else {
        for (int i = 0; i < 4 && !x->quit; i++) {
            WaitForSingleObject(x->mtx, 5000); SetEvent(x->held); ymm_clobber_all(&x->pat); Sleep(10); ReleaseMutex(x->mtx);
            WaitForSingleObject(x->go, 5000);
        }
    }
    return 0;
}
static void cb_pingpong(void *a) { pp *x = (pp *)a; for (int i = 0; i < 30; i++) { SetEvent(x->go); WaitForSingleObject(x->done, 2000); } }
static void cb_mutex(void *a) {
    pp *x = (pp *)a;
    for (int i = 0; i < 4; i++) {
        WaitForSingleObject(x->held, 2000);
        WaitForSingleObject(x->mtx, 2000); ReleaseMutex(x->mtx);
        SetEvent(x->go);
    }
}
static void pp_run(int mode, void (*cb)(void *), const char *label) {
    ygate();
    static pp x;
    memset(&x, 0, sizeof x); x.mode = mode; ypat(&x.pat, 0xC80);
    x.go = CreateEventA(NULL, FALSE, FALSE, NULL); x.done = CreateEventA(NULL, FALSE, FALSE, NULL);
    x.held = CreateEventA(NULL, FALSE, FALSE, NULL); x.mtx = CreateMutexA(NULL, FALSE, NULL);
    REQUIRE(x.go && x.done && x.held && x.mtx);
    HANDLE h = CreateThread(NULL, 0, pp_thread, &x, 0, NULL);
    if (!h) SKIP("CreateThread failed");
    ycall_case(cb, &x, label);
    x.quit = 1; SetEvent(x.go);
    CHECK(WaitForSingleObject(h, 10000) == WAIT_OBJECT_0);
    CloseHandle(h); CloseHandle(x.go); CloseHandle(x.done); CloseHandle(x.held); CloseHandle(x.mtx);
}
TEST(ymm_call_event_pingpong_dirty_peer) { pp_run(0, cb_pingpong, "across event ping-pong with a YMM-dirty peer"); }
TEST(ymm_call_mutex_contention) { pp_run(1, cb_mutex, "across contended mutex waits"); }

typedef LONG (NTAPI *ntyield_t)(void);
typedef LONG (NTAPI *ntdelay_t)(BOOLEAN, PLARGE_INTEGER);
static void cb_ntyield(void *a) { ntyield_t f = (ntyield_t)a; for (int i = 0; i < 50; i++) f(); }
static void cb_ntdelay(void *a) { ntdelay_t f = (ntdelay_t)a; LARGE_INTEGER d; d.QuadPart = -10000; for (int i = 0; i < 3; i++) f(FALSE, &d); }
TEST(ymm_call_ntyieldexecution) {
    void *f = t_proc("ntdll.dll", "NtYieldExecution"); if (!f) SKIP("NtYieldExecution missing");
    ycall_case(cb_ntyield, f, "across NtYieldExecution");
}
TEST(ymm_call_ntdelayexecution) {
    void *f = t_proc("ntdll.dll", "NtDelayExecution"); if (!f) SKIP("NtDelayExecution missing");
    ycall_case(cb_ntdelay, f, "across NtDelayExecution(1 ms)");
}
static void cb_valloc(void *a) {
    (void)a;
    for (int i = 0; i < 10; i++) {
        void *p = VirtualAlloc(NULL, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (p) { memset(p, i, 4096); VirtualFree(p, 0, MEM_RELEASE); }
    }
}
TEST(ymm_call_virtualalloc) { ycall_case(cb_valloc, NULL, "across VirtualAlloc/VirtualFree"); }
static void cb_writefile(void *a) {
    (void)a;
    HANDLE f = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
    if (f != INVALID_HANDLE_VALUE) { DWORD w; char b[256]; memset(b, 'x', sizeof b); for (int i = 0; i < 10; i++) WriteFile(f, b, sizeof b, &w, NULL); CloseHandle(f); }
}
TEST(ymm_call_file_io) { ycall_case(cb_writefile, NULL, "across CreateFile/WriteFile(NUL)"); }
static void cb_query(void *a) { (void)a; LARGE_INTEGER q; for (int i = 0; i < 100; i++) { QueryPerformanceCounter(&q); GetTickCount(); GetCurrentThreadId(); } }
TEST(ymm_call_no_syscall_apis) { ycall_case(cb_query, NULL, "across QPC/GetTickCount"); }
static void cb_getctx_self(void *a) { (void)a; CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_FULL; GetThreadContext(GetCurrentThread(), &c); RtlCaptureContext(&c); }
TEST(ymm_call_getcontext_self) { ycall_case(cb_getctx_self, NULL, "across GetThreadContext(self)/RtlCaptureContext"); }
static void cb_ods(void *a) { (void)a; for (int i = 0; i < 3; i++) OutputDebugStringA("ymm_registers: OutputDebugString probe\n"); }
TEST(ymm_call_outputdebugstring) { ycall_case(cb_ods, NULL, "across OutputDebugString (internal exception)"); }

static void CALLBACK tp_cb(PTP_CALLBACK_INSTANCE inst, void *ctx) { (void)inst; static yfile pat; ypat(&pat, 0xCA0); ymm_clobber_all(&pat); SetEvent((HANDLE)ctx); }
static void cb_threadpool(void *a) {
    HANDLE e = (HANDLE)a;
    for (int i = 0; i < 4; i++) if (TrySubmitThreadpoolCallback(tp_cb, e, NULL)) WaitForSingleObject(e, 5000);
}
TEST(ymm_call_threadpool_roundtrip) {
    HANDLE e = CreateEventA(NULL, FALSE, FALSE, NULL); REQUIRE(e);
    ycall_case(cb_threadpool, e, "across thread-pool callback round trips"); CloseHandle(e);
}
static volatile long g_apc_n;
static void CALLBACK apc_normal(ULONG_PTR p) { ymm_clobber_all((const void *)p); InterlockedIncrement(&g_apc_n); }
static void cb_apc(void *a) {
    for (int i = 0; i < 3; i++) { QueueUserAPC(apc_normal, GetCurrentThread(), (ULONG_PTR)a); SleepEx(0, TRUE); }
}
TEST(ymm_call_user_apc_alertable) {
    static yfile pat; ypat(&pat, 0xCB0);
    g_apc_n = 0;
    ycall_case(cb_apc, &pat, "across SleepEx(alertable) running user APCs that clobber YMM");
    CHECK_EQ(g_apc_n, 3);
}
static volatile long g_veh_n;
static yfile g_veh_pat;
static LONG CALLBACK veh_soft(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != 0xE0577E01u) return EXCEPTION_CONTINUE_SEARCH;
    ymm_clobber_all(&g_veh_pat); InterlockedIncrement(&g_veh_n);
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void cb_raise(void *a) {
    (void)a;
    PVOID h = AddVectoredExceptionHandler(1, veh_soft);
    for (int i = 0; i < 3; i++) RaiseException(0xE0577E01u, 0, 0, NULL);
    RemoveVectoredExceptionHandler(h);
}
TEST(ymm_call_raiseexception_veh) {
    ypat(&g_veh_pat, 0xCC0); g_veh_n = 0;
    ycall_case(cb_raise, NULL, "across RaiseException handled by a VEH that clobbers YMM");
    CHECK_EQ(g_veh_n, 3);
}
static void *g_main_fiber;
static yfile g_fiber_pat;
static volatile long g_fiber_n;
static void CALLBACK fiber_proc(void *p) {
    (void)p;
    for (;;) { ymm_clobber_all(&g_fiber_pat); InterlockedIncrement(&g_fiber_n); SwitchToFiber(g_main_fiber); }
}
static void cb_fiber(void *a) {
    DWORD flags = (DWORD)(ULONG_PTR)a;
    g_main_fiber = ConvertThreadToFiberEx(NULL, flags);   /* the case thread is never a fiber already */
    if (!g_main_fiber) return;
    void *f = CreateFiberEx(0, 0, flags, fiber_proc, NULL);
    if (f) { for (int i = 0; i < 5; i++) SwitchToFiber(f); DeleteFiber(f); }
    ConvertFiberToThread();
}
TEST(ymm_call_fiber_switch) {
    ypat(&g_fiber_pat, 0xCD0); g_fiber_n = 0;
    ycall_case(cb_fiber, (void *)0, "across SwitchToFiber (fiber clobbers YMM)");
    CHECK_EQ(g_fiber_n, 5);
}
TEST(ymm_call_fiber_switch_float_flag) {
    ypat(&g_fiber_pat, 0xCD1); g_fiber_n = 0;
    ycall_case(cb_fiber, (void *)(ULONG_PTR)FIBER_FLAG_FLOAT_SWITCH, "across SwitchToFiber(FIBER_FLAG_FLOAT_SWITCH)");
    CHECK_EQ(g_fiber_n, 5);
}
static DWORD WINAPI clob_thread(void *p) { ymm_clobber_all(p); return 0; }
static void cb_thread(void *a) {
    for (int i = 0; i < 4; i++) {
        HANDLE h = CreateThread(NULL, 64 * 1024, clob_thread, a, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (h) { WaitForSingleObject(h, 5000); CloseHandle(h); }
    }
}
TEST(ymm_call_create_join_threads) {
    static yfile pat; ypat(&pat, 0xCE0);
    ycall_case(cb_thread, &pat, "across CreateThread/WaitForSingleObject of YMM-dirty threads");
}
static void cb_affinity(void *a) {
    (void)a;
    const t_env *e = t_env_get();
    unsigned long long aff = e->affinity ? e->affinity : 1;
    DWORD_PTR prev = 0;
    for (int b = 0, n = 0; b < 64 && n < 8; b++) {
        if (!(aff >> b & 1)) continue;
        DWORD_PTR p = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << b);
        if (!prev) prev = p;
        Sleep(0); n++;
    }
    if (prev) SetThreadAffinityMask(GetCurrentThread(), prev);
}
TEST(ymm_call_affinity_migration) {
    if (t_cpu_count() < 2) SKIP("single logical processor");
    ycall_case(cb_affinity, NULL, "across self-migration between processors");
}
/* kernel -> user callbacks (KeUserModeCallback): window creation and SendMessage deliver WM_* to the
 * window procedure from win32k; the procedure dirties every YMM register. user32 is loaded dynamically. */
typedef ATOM (WINAPI *regcls_t)(const WNDCLASSEXA *);
typedef HWND (WINAPI *crwin_t)(DWORD, LPCSTR, LPCSTR, DWORD, int, int, int, int, HWND, HMENU, HINSTANCE, LPVOID);
typedef LRESULT (WINAPI *sendmsg_t)(HWND, UINT, WPARAM, LPARAM);
typedef BOOL (WINAPI *dstwin_t)(HWND);
typedef BOOL (WINAPI *unreg_t)(LPCSTR, HINSTANCE);
static struct { regcls_t reg; crwin_t create; sendmsg_t send; dstwin_t destroy; unreg_t unreg; WNDPROC defproc; } g_u32;
static yfile g_wnd_pat;
static volatile long g_wnd_msgs, g_wnd_user;
static LRESULT CALLBACK ymm_wndproc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    ymm_clobber_all(&g_wnd_pat);
    InterlockedIncrement(&g_wnd_msgs);
    if (m == WM_USER + 7) { InterlockedIncrement(&g_wnd_user); return 0x7E57; }
    return g_u32.defproc(h, m, wp, lp);
}
static volatile int g_wnd_created;
static void cb_window(void *a) {
    (void)a;
    HWND w = g_u32.create(0, "ymm_registers_wc", "", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, GetModuleHandleA(NULL), NULL);
    if (!w) return;
    g_wnd_created = 1;
    for (int i = 0; i < 8; i++) g_u32.send(w, WM_USER + 7, (WPARAM)i, 0);
    g_u32.destroy(w);
}
TEST(ymm_call_window_message_callback) {
    ygate();
    g_u32.reg = (regcls_t)t_proc("user32.dll", "RegisterClassExA");
    g_u32.create = (crwin_t)t_proc("user32.dll", "CreateWindowExA");
    g_u32.send = (sendmsg_t)t_proc("user32.dll", "SendMessageA");
    g_u32.destroy = (dstwin_t)t_proc("user32.dll", "DestroyWindow");
    g_u32.unreg = (unreg_t)t_proc("user32.dll", "UnregisterClassA");
    g_u32.defproc = (WNDPROC)t_proc("user32.dll", "DefWindowProcA");
    if (!g_u32.reg || !g_u32.create || !g_u32.send || !g_u32.destroy || !g_u32.unreg || !g_u32.defproc)
        SKIP("user32 window APIs unavailable");
    WNDCLASSEXA wc; memset(&wc, 0, sizeof wc);
    wc.cbSize = sizeof wc; wc.lpfnWndProc = ymm_wndproc; wc.hInstance = GetModuleHandleA(NULL); wc.lpszClassName = "ymm_registers_wc";
    if (!g_u32.reg(&wc)) SKIP("RegisterClassExA failed: %lu", (unsigned long)GetLastError());
    ypat(&g_wnd_pat, 0xCF8); g_wnd_msgs = 0; g_wnd_user = 0; g_wnd_created = 0;
    ycall_case(cb_window, NULL, "across CreateWindowEx/SendMessage (kernel->user window-procedure callbacks)");
    g_u32.unreg("ymm_registers_wc", GetModuleHandleA(NULL));
    if (!g_wnd_created) SKIP("CreateWindowExA(HWND_MESSAGE) failed (no window station / desktop?)");
    CHECK_EQ(g_wnd_user, 8);
    t_info("window_messages", "%ld", g_wnd_msgs);
}
static void cb_clobber(void *a) { ymm_clobber_all(a); }
TEST(ymm_call_abi_legal_clobber) {
    static yfile pat; ypat(&pat, 0xCF0);
    ycall_case(cb_clobber, &pat, "across an ABI-conforming callee that loads every YMM register");
}

/* ======================================================================= */
/* performance (PERF only)                                                  */
/* ======================================================================= */
typedef struct tctx { const void *dirty; uint8_t T_ALIGN(16) x2[16]; uint8_t T_ALIGN(16) x3[16]; unsigned long n; int mode; } tctx;
static T_NOINLINE void ymm_sse_loop(tctx *c) {
    switch (c->mode) {
    case 0:  /* legacy SSE with clean upper state */
        __asm__ volatile("vmovdqu (%[d]), %%ymm7\n vzeroupper\n movdqu (%[a]), %%xmm2\n movdqu (%[b]), %%xmm3\n"
            "1: paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n dec %[n]\n jnz 1b\n"
            "movdqu %%xmm3, (%[b])\n" : [n]"+r"(c->n) : [d]"r"(c->dirty), [a]"r"(c->x2), [b]"r"(c->x3) : "memory", "cc", "xmm2", "xmm3", "xmm7");
        break;
    case 1:  /* legacy SSE with a dirty upper lane (ymm7) */
        __asm__ volatile("vmovdqu (%[d]), %%ymm7\n movdqu (%[a]), %%xmm2\n movdqu (%[b]), %%xmm3\n"
            "1: paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n dec %[n]\n jnz 1b\n"
            "movdqu %%xmm3, (%[b])\n vzeroupper\n" : [n]"+r"(c->n) : [d]"r"(c->dirty), [a]"r"(c->x2), [b]"r"(c->x3) : "memory", "cc", "xmm2", "xmm3", "xmm7");
        break;
    case 2:  /* alternate VEX.256 and legacy SSE every iteration */
        __asm__ volatile("vmovdqu (%[d]), %%ymm7\n movdqu (%[a]), %%xmm2\n movdqu (%[b]), %%xmm3\n"
            "1: vxorps %%ymm7, %%ymm6, %%ymm6\n paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n vxorps %%ymm7, %%ymm6, %%ymm6\n"
            "paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n dec %[n]\n jnz 1b\n"
            "movdqu %%xmm3, (%[b])\n vzeroupper\n" : [n]"+r"(c->n) : [d]"r"(c->dirty), [a]"r"(c->x2), [b]"r"(c->x3) : "memory", "cc", "xmm2", "xmm3", "xmm6", "xmm7");
        break;
    default: /* alternate with vzeroupper before each legacy block */
        __asm__ volatile("vmovdqu (%[d]), %%ymm7\n movdqu (%[a]), %%xmm2\n movdqu (%[b]), %%xmm3\n"
            "1: vxorps %%ymm7, %%ymm6, %%ymm6\n vzeroupper\n paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n vxorps %%ymm7, %%ymm6, %%ymm6\n vzeroupper\n"
            "paddd %%xmm2, %%xmm3\n paddd %%xmm2, %%xmm3\n vmovdqu (%[d]), %%ymm7\n dec %[n]\n jnz 1b\n"
            "movdqu %%xmm3, (%[b])\n vzeroupper\n" : [n]"+r"(c->n) : [d]"r"(c->dirty), [a]"r"(c->x2), [b]"r"(c->x3) : "memory", "cc", "xmm2", "xmm3", "xmm6", "xmm7");
        break;
    }
}
#define TLOOP 4096UL
static tctx g_tc;
static void ymm_bench_loop(void *p) { tctx *c = (tctx *)p; c->n = TLOOP; ymm_sse_loop(c); }
TEST(ymm_sse_avx_transition_perf) {
    ygate();
    static yfile d; ypat(&d, 0xD00);
    static const char *nm[4] = { "sse_clean_upper", "sse_dirty_upper", "alternate_vex256_sse", "alternate_with_vzeroupper" };
    double med[4] = { 0 };
    for (int mode = 0; mode < 4; mode++) {
        g_tc.dirty = d.v[0]; g_tc.mode = mode;
        /* correctness first: one call from known values */
        for (int k = 0; k < 4; k++) { wr32(g_tc.x2, k, (uint32_t)(k + 1)); wr32(g_tc.x3, k, 100u * (uint32_t)k); }
        ymm_bench_loop(&g_tc);
        for (int k = 0; k < 4; k++) CHECK_EQ(rd32(g_tc.x3, k), 100u * (uint32_t)k + 4u * TLOOP * (uint32_t)(k + 1));
        t_stats s;
        int n = t_bench(ymm_bench_loop, &g_tc, 3, 15, 2.0, &s);
        CHECK(n > 0);
        med[mode] = s.median * 1e9 / TLOOP;
        char m[64]; snprintf(m, sizeof m, "%s_ns_per_iter", nm[mode]);
        t_perf(m, med[mode], "ns/iter");
    }
    if (med[0] > 0) t_info("dirty_vs_clean_ratio", "%.2f", med[1] / med[0]);
    if (med[3] > 0) t_info("alternate_vs_vzeroupper_ratio", "%.2f", med[2] / med[3]);
}
typedef struct lctx { uint8_t *p; int off; unsigned long n; } lctx;
static T_NOINLINE void ymm_load_store_loop(lctx *c) {
    uint8_t *p = c->p + c->off; unsigned long n = c->n;
    __asm__ volatile("1: vmovdqu (%[p]), %%ymm0\n vmovdqu 64(%[p]), %%ymm1\n vmovdqu %%ymm0, 128(%[p])\n vmovdqu %%ymm1, 192(%[p])\n dec %[n]\n jnz 1b\n vzeroupper\n"
        : [n]"+r"(n) : [p]"r"(p) : "memory", "cc", "xmm0", "xmm1");
}
static void ymm_ls_bench(void *p) { ((lctx *)p)->n = 4096; ymm_load_store_loop((lctx *)p); }
TEST(ymm_load_store_alignment_perf) {
    ygate();
    uint8_t *buf = (uint8_t *)VirtualAlloc(NULL, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(buf);
    static const int offs[] = { 0, 16, 48 };
    static const char *nm[] = { "aligned32", "aligned16_same_line", "split_cache_line" };
    for (size_t k = 0; k < T_COUNTOF(offs); k++) {
        lctx c = { buf, offs[k], 0 };
        memset(buf, (int)k + 1, 1024);
        ymm_ls_bench(&c);
        CHECK(memcmp(buf + offs[k], buf + offs[k] + 128, 32) == 0);
        t_stats s;
        CHECK(t_bench(ymm_ls_bench, &c, 3, 15, 2.0, &s) > 0);
        char m[64]; snprintf(m, sizeof m, "%s_ns_per_4x32B", nm[k]);
        t_perf(m, s.median * 1e9 / 4096, "ns/iter");
    }
    VirtualFree(buf, 0, MEM_RELEASE);
}
