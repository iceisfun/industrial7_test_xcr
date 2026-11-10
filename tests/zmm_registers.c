/* zmm_registers - AVX-512 register file (ZMM0-31, k0-k7), the EVEX semantics applications depend on
 * (opmask merge/zero masking, embedded broadcast and rounding, VPTERNLOG, conflict detection,
 * compress/expand, gather/scatter, memory fault suppression), preservation of AVX-512 state across
 * OS-mediated operations (preemption, waits, callbacks, exceptions, Get/SetThreadContext), and the
 * CPUID / XCR0 / Windows XSTATE API gating that decides whether AVX-512 may be used at all.
 *
 * Contract sources:
 *  - Intel SDM Vol.1 13.x (XSAVE-managed state: component 5 opmask 64 B, 6 ZMM_Hi256 512 B,
 *    7 Hi16_ZMM 1024 B; XRSTOR initializes RFBM components whose XSTATE_BV bit is 0; XINUSE[i]=0
 *    implies component i is in its init state), Vol.1 15.2-15.4 (AVX-512 detection: CPUID.(7,0):EBX[16]
 *    and OSXSAVE and XCR0[7:5]=111b and XCR0[2:1]=11b), XSETBV (#GP at CPL>0; XCR0[7:5] all-or-none and
 *    only with XCR0[2:1]=11b), Vol.2 instruction pages for every instruction used below, exception
 *    classes E4/E12 (masked-off elements neither fault nor report FP exceptions).
 *  - Windows x64 ABI: only the low 128 bits of XMM6-15 are nonvolatile; ZMM upper bits, ZMM16-31 and
 *    k0-k7 are volatile. On x86 every vector register is volatile.
 *  - Windows XSTATE APIs, read from the Windows 10 19041 reference decompilation:
 *      kernelbase!GetEnabledXStateFeatures  = RtlGetEnabledExtendedFeatures(~0), or 3 when that is 0
 *      ntdll!RtlGetEnabledExtendedFeatures  = (XState.EnabledFeatures | EnabledSupervisorFeatures) & mask
 *      kernelbase!InitializeContext(2)      : Buffer==NULL or *Length < needed -> FALSE,
 *                                              ERROR_INSUFFICIENT_BUFFER (0x7A), *Length = needed;
 *                                              success stores ContextFlags into the CONTEXT
 *      kernelbase!SetXStateFeaturesMask     -> ntdll!RtlSetExtendedFeaturesMask: mask &= enabled & ~3
 *      kernelbase!GetXStateFeaturesMask     -> header mask (| 3 when CONTEXT_FLOATING_POINT is set)
 *      kernelbase!LocateXStateFeature       : no CONTEXT_XSTATE -> NULL; ids 0/1 legacy area; others via
 *                                              ntdll!RtlLocateExtendedFeature2: NULL unless 2 <= id < 64 and
 *                                              id is an enabled feature (and, compacted, in XCOMP_BV);
 *                                              *Length = component size from the shared XState config
 * Result policy: AVX-512 absent or not OS-enabled -> SKIP with the precise reason; advertised but wrong
 * -> FAIL; implementation-defined observations -> t_note / t_info.
 * Deliberate faults (EVEX #UD, XSETBV #GP, unmasked-lane #PF, misaligned VMOVDQA64) and every
 * fault-suppression probe run in CHILD processes. */
#include "t.h"
#include <immintrin.h>
#include <math.h>

#ifndef PF_AVX512F_INSTRUCTIONS_AVAILABLE
#define PF_AVX512F_INSTRUCTIONS_AVAILABLE 41
#endif
#ifndef CONTEXT_XSTATE
#ifdef __x86_64__
#define CONTEXT_XSTATE (CONTEXT_AMD64 | 0x40L)
#else
#define CONTEXT_XSTATE (CONTEXT_i386 | 0x40L)   /* winnt.h value; mingw only defines it for x64 */
#endif
#endif
#define ZX_AVX512   0xE0ull   /* XCR0 bits 7:5 */
#define ZX_ZMMSTATE 0xE6ull   /* + SSE/AVX */

/* =============================================================================================
 * gating
 * ============================================================================================= */
static const char *zx_why_not(void) {
    static char buf[256];
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_AVX512F)) return "AVX512F not advertised (CPUID.(7,0):EBX[16]=0)";
    if (!c->osxsave) return "AVX512F advertised but CPUID.1:ECX.OSXSAVE=0 (OS did not enable XSAVE/XGETBV)";
    if ((c->xcr0 & ZX_ZMMSTATE) != ZX_ZMMSTATE) {
        snprintf(buf, sizeof buf, "AVX512F advertised but XCR0=0x%llx lacks 0xE6 (opmask, ZMM_Hi256, Hi16_ZMM "
                 "and SSE/AVX): the OS did not enable AVX-512 state", (unsigned long long)c->xcr0);
        return buf;
    }
    return "AVX512F usable";
}
#define NEED_F() do { if (!cpu_has(F_AVX512F)) SKIP("%s", zx_why_not()); } while (0)
#define NEED_X(f) do { NEED_F(); if (!cpu_has(f)) SKIP("%s not advertised (AVX512F is usable)", cpu_feature_name(f)); } while (0)
#ifdef T_X64
#define NEED_X64() ((void)0)
#else
#define NEED_X64() SKIP("ZMM16-31 / XMM16-31 are not encodable outside 64-bit mode")
#endif

/* =============================================================================================
 * vector values and generic EVEX operation harness
 * ============================================================================================= */
typedef union T_ALIGN(64) zv {
    uint8_t b[64]; int8_t sb[64]; uint16_t w[32]; int16_t sw[32];
    uint32_t d[16]; int32_t sd[16]; uint64_t q[8]; int64_t sq[8]; float f[16]; double g[8];
} zv;
/* s = zmm0 (destination preload), a = zmm1, b = zmm2, k = k1 source; result zmm0 -> o, k2 -> ko */
typedef struct T_ALIGN(64) zop { zv s, a, b, o; uint64_t k, ko; } zop;
_Static_assert(offsetof(zop, o) == 192 && offsetof(zop, k) == 256 && offsetof(zop, ko) == 264, "ZOP_FN offsets");

#define ZOP_FN(name, isa, kld, body, kst) \
    T_TARGET(isa) T_NOINLINE static void name(zop *p) { \
        __asm__ volatile("vmovdqu64 0(%0), %%zmm0\n\tvmovdqu64 64(%0), %%zmm1\n\tvmovdqu64 128(%0), %%zmm2\n\t" \
                         kld " 256(%0), %%k1\n\tkxorw %%k2, %%k2, %%k2\n\t" body "\n\t" \
                         "vmovdqu64 %%zmm0, 192(%0)\n\t" kst " %%k2, 264(%0)\n\t" \
                         :: "r"(p) : "xmm0", "xmm1", "xmm2", "xmm3", "k1", "k2", "memory", "cc"); }
static void zop_run(void (*fn)(zop *), zop *p) { memset(&p->o, 0xCC, sizeof p->o); p->ko = 0; fn(p); }

static void zx_rand(zv *v, t_rng *r) { t_rng_fill(r, v, sizeof *v); }
static uint32_t zx_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float zx_u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint64_t zx_d2u(double f) { uint64_t u; memcpy(&u, &f, 8); return u; }
static double zx_u2d(uint64_t u) { double f; memcpy(&f, &u, 8); return f; }
static int zx_pop64(uint64_t x) { int n = 0; while (x) { x &= x - 1; n++; } return n; }
static uint64_t zx_lanes_mask(int n) { return n >= 64 ? ~0ull : ((1ull << n) - 1); }

/* compare n lanes of width w; one CHECK per call, details of the first mismatching lane */
static int zx_lanes_eq_at(const char *file, int line, const char *what, const void *got, const void *want, int lanes, int w) {
    const uint8_t *g = got, *x = want;
    for (int i = 0; i < lanes; i++) {
        if (memcmp(g + i * w, x + i * w, (size_t)w)) {
            uint64_t gv = 0, xv = 0; memcpy(&gv, g + i * w, (size_t)(w > 8 ? 8 : w)); memcpy(&xv, x + i * w, (size_t)(w > 8 ? 8 : w));
            t_check(0, file, line, what, "lane %d (of %d x %d bytes): got 0x%llx want 0x%llx", i, lanes, w,
                    (unsigned long long)gv, (unsigned long long)xv);
            return 0;
        }
    }
    t_check(1, file, line, what, NULL);
    return 1;
}
#define CHECK_LANES(what, got, want, lanes, w) zx_lanes_eq_at(__FILE__, __LINE__, (what), (got), (want), (lanes), (w))

/* =============================================================================================
 * whole-register-file routines: load ZMM0-31 (x86: 0-7) + k0-7 from c->in, run MID, store to c->out.
 * Everything happens in one asm statement, so nothing compiled sits between load and store.
 * ============================================================================================= */
typedef struct zctx {
    t_vregs in;
    t_vregs out;
    uint32_t kbw, pad;
    uint64_t aux;
    void *ptr;
    uint64_t rfbm;
} zctx;

#ifdef T_X64
#define ZR_ALL "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
#define ZR_HI16 "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
#define ZCLOB_V "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12", \
    "xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25", \
    "xmm26","xmm27","xmm28","xmm29","xmm30","xmm31"
#define ZP "%%rcx"
#define ZMOVP "movq"
#define ZXSAVE "xsave64"
#define ZXSAVEC "xsavec64"
#define ZXRSTOR "xrstor64"
#else
#define ZR_ALL "0,1,2,3,4,5,6,7"
#define ZCLOB_V "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7"
#define ZP "%%ecx"
#define ZMOVP "movl"
#define ZXSAVE "xsave"
#define ZXSAVEC "xsavec"
#define ZXRSTOR "xrstor"
#endif
#define ZCLOB_K "k0","k1","k2","k3","k4","k5","k6","k7"

#define ZX_LOAD \
    "  .irp i," ZR_ALL "\n  vmovdqu64 \\i*64(%[c]), %%zmm\\i\n  .endr\n" \
    "  cmpl $0, %c[kbw](%[c])\n  je 81f\n" \
    "  .irp i,0,1,2,3,4,5,6,7\n  kmovq %c[kin]+\\i*8(%[c]), %%k\\i\n  .endr\n  jmp 82f\n" \
    "81:\n  .irp i,0,1,2,3,4,5,6,7\n  kmovw %c[kin]+\\i*8(%[c]), %%k\\i\n  .endr\n82:\n"
#define ZX_STORE \
    "  .irp i," ZR_ALL "\n  vmovdqu64 %%zmm\\i, %c[vout]+\\i*64(%[c])\n  .endr\n" \
    "  cmpl $0, %c[kbw](%[c])\n  je 83f\n" \
    "  .irp i,0,1,2,3,4,5,6,7\n  kmovq %%k\\i, %c[kout]+\\i*8(%[c])\n  .endr\n  jmp 84f\n" \
    "83:\n  .irp i,0,1,2,3,4,5,6,7\n  kmovw %%k\\i, %c[kout]+\\i*8(%[c])\n  .endr\n84:\n"
#define ZX_OPS [c] "r"(c), [kbw] "i"(offsetof(zctx, kbw)), [kin] "i"(offsetof(zctx, in.k)), \
    [vout] "i"(offsetof(zctx, out.v)), [kout] "i"(offsetof(zctx, out.k)), [aux] "i"(offsetof(zctx, aux)), \
    [ptr] "i"(offsetof(zctx, ptr)), [rfbm] "i"(offsetof(zctx, rfbm))
#define ZX_FUNC(name, mid) \
    T_TARGET("avx512f") T_NOINLINE __attribute__((noclone)) static void name(zctx *c) { \
        __asm__ volatile(ZX_LOAD mid ZX_STORE "  vzeroupper\n" \
                         :: ZX_OPS : ZCLOB_V, ZCLOB_K, "eax", "ecx", "edx", "memory", "cc"); }

#define ZM_PTR   "  " ZMOVP " %c[ptr](%[c]), " ZP "\n"
#define ZM_RFBM  "  movl %c[rfbm](%[c]), %%eax\n  movl %c[rfbm]+4(%[c]), %%edx\n"
#ifdef T_X64
#define ZM_ZERO_HI16 "  .irp i," ZR_HI16 "\n  vpxord %%zmm\\i, %%zmm\\i, %%zmm\\i\n  .endr\n"
#else
#define ZM_ZERO_HI16 ""
#endif

ZX_FUNC(zmm_rf_roundtrip, "")
ZX_FUNC(zmm_rf_vzeroupper, "  vzeroupper\n")
ZX_FUNC(zmm_rf_vzeroall, "  vzeroall\n")
ZX_FUNC(zmm_rf_xinuse, "  movl $1, %%ecx\n  xgetbv\n  movl %%eax, %c[aux](%[c])\n  movl %%edx, %c[aux]+4(%[c])\n")
ZX_FUNC(zmm_rf_xinuse_after_init,
        "  vzeroall\n  .irp i,0,1,2,3,4,5,6,7\n  kxorw %%k\\i, %%k\\i, %%k\\i\n  .endr\n" ZM_ZERO_HI16
        "  movl $1, %%ecx\n  xgetbv\n  movl %%eax, %c[aux](%[c])\n  movl %%edx, %c[aux]+4(%[c])\n")
ZX_FUNC(zmm_rf_xsave, ZM_PTR ZM_RFBM "  " ZXSAVE " (" ZP ")\n")
ZX_FUNC(zmm_rf_xsavec, ZM_PTR ZM_RFBM "  " ZXSAVEC " (" ZP ")\n")
ZX_FUNC(zmm_rf_xrstor, ZM_PTR ZM_RFBM "  " ZXRSTOR " (" ZP ")\n")
/* faulting instructions; each is exactly the length the VEH skips */
ZX_FUNC(zmm_rf_fault_ud, "  ud2\n")                                                      /* 0F 0B */
ZX_FUNC(zmm_rf_fault_av, ZM_PTR "  movl (" ZP "), %%eax\n")                               /* 8B 01 */
ZX_FUNC(zmm_rf_fault_div, "  xorl %%ecx, %%ecx\n  movl $1, %%eax\n  xorl %%edx, %%edx\n  divl %%ecx\n") /* F7 F1 */
ZX_FUNC(zmm_rf_fault_int3, "  int3\n")                                                   /* CC */

/* store-only: the registers as found (used as the first thing a fresh thread does) */
T_TARGET("avx512f") T_NOINLINE __attribute__((noclone)) static void zmm_rf_store_only(zctx *c) {
    __asm__ volatile(ZX_STORE :: ZX_OPS : "memory", "cc");
}

static void zx_ctx_init(zctx *c, uint32_t tag) {
    memset(c, 0, sizeof *c);
    t_vregs_pattern(&c->in, T_VK_ZMM, tag);
    for (int i = 0; i < 8; i++) c->in.k[i] |= 1;   /* never all-zero (XINUSE checks rely on it) */
    c->kbw = (uint32_t)cpu_has(F_AVX512BW);
}

/* =============================================================================================
 * Windows XSTATE API access (resolved at run time)
 * ============================================================================================= */
typedef struct zx_api {
    __typeof__(&GetEnabledXStateFeatures) en;
    __typeof__(&InitializeContext) init;
    __typeof__(&SetXStateFeaturesMask) setm;
    __typeof__(&GetXStateFeaturesMask) getm;
    __typeof__(&LocateXStateFeature) loc;
} zx_api;
static int zx_api_get(zx_api *a) {
    a->en = T_PROC("kernel32.dll", GetEnabledXStateFeatures);
    a->init = T_PROC("kernel32.dll", InitializeContext);
    a->setm = T_PROC("kernel32.dll", SetXStateFeaturesMask);
    a->getm = T_PROC("kernel32.dll", GetXStateFeaturesMask);
    a->loc = T_PROC("kernel32.dll", LocateXStateFeature);
    return a->en && a->init && a->setm && a->getm && a->loc;
}
#define NEED_API(a) do { if (!zx_api_get(a)) SKIP("XSTATE context APIs (GetEnabledXStateFeatures, InitializeContext, " \
    "Get/SetXStateFeaturesMask, LocateXStateFeature) not all exported"); } while (0)
#define ZX_XFLAGS (CONTEXT_ALL | CONTEXT_XSTATE)
#define ZX_CTXBUF 65536
static uint8_t T_ALIGN(64) g_ctxbuf[2][ZX_CTXBUF];

/* InitializeContext into buf: returns the CONTEXT or NULL (reason in *why). */
static CONTEXT *zx_context_new(zx_api *a, uint8_t *buf, DWORD flags, DWORD *plen, const char **why) {
    DWORD len = 0; CONTEXT *ctx = NULL;
    SetLastError(0);
    if (a->init(NULL, flags, &ctx, &len) || GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) {
        *why = "InitializeContext(NULL) did not report the needed length"; return NULL;
    }
    if (len > ZX_CTXBUF) { *why = "context larger than the static buffer"; return NULL; }
    memset(buf, 0, ZX_CTXBUF);
    if (!a->init(buf, flags, &ctx, &len) || !ctx) { *why = "InitializeContext failed"; return NULL; }
    if (plen) *plen = len;
    return ctx;
}
static const uint8_t *zx_ctx_xmm(const CONTEXT *c, int i) {
#ifdef T_X64
    return (const uint8_t *)&c->FltSave.XmmRegisters[i];
#else
    return (const uint8_t *)c->ExtendedRegisters + 160 + 16 * i;
#endif
}

/* =============================================================================================
 * 1. gating: CPUID / XCR0 / Windows API agreement (runs on every host)
 * ============================================================================================= */
static const t_feature g_subs[] = {
    F_AVX512DQ, F_AVX512IFMA, F_AVX512PF, F_AVX512ER, F_AVX512CD, F_AVX512BW, F_AVX512VL, F_AVX512VBMI,
    F_AVX512VBMI2, F_AVX512VNNI, F_AVX512BITALG, F_AVX512VPOPCNTDQ, F_AVX5124VNNIW, F_AVX5124FMAPS,
    F_AVX512VP2INTERSECT, F_AVX512FP16, F_AVX512BF16,
};

TEST(zmm_gate_cpuid_subfeatures) {
    /* SDM 15.2/15.4: every AVX-512 subset is defined on top of AVX512F; software checks F first.
     * A subset bit without F is an inconsistent CPUID that breaks every dispatcher. */
    char hw[512] = "", us[512] = "";
    size_t nh = 0, nu = 0;
    nh += (size_t)snprintf(hw + nh, sizeof hw - nh, "%s", cpu_hw(F_AVX512F) ? "avx512f" : "");
    nu += (size_t)snprintf(us + nu, sizeof us - nu, "%s", cpu_has(F_AVX512F) ? "avx512f" : "");
    for (size_t i = 0; i < T_COUNTOF(g_subs); i++) {
        t_feature f = g_subs[i];
        if (cpu_hw(f) && nh < sizeof hw - 32) nh += (size_t)snprintf(hw + nh, sizeof hw - nh, " %s", cpu_feature_name(f));
        if (cpu_has(f) && nu < sizeof us - 32) nu += (size_t)snprintf(us + nu, sizeof us - nu, " %s", cpu_feature_name(f));
        if (cpu_hw(f)) CHECKF(cpu_hw(F_AVX512F), "%s advertised without avx512f", cpu_feature_name(f));
        CHECKF(!cpu_has(f) || cpu_has(F_AVX512F), "%s usable while avx512f is not", cpu_feature_name(f));
    }
    t_info("avx512_hw", "%s", hw[0] ? hw : "(none)");
    t_info("avx512_usable", "%s", us[0] ? us : "(none)");
    if (cpu_hw(F_AVX512F) && !cpu_hw(F_AVX2)) t_note("avx512f advertised without avx2 (unusual CPUID)");
    if (cpu_hw(F_AVX10)) t_info("avx10_version", "%d", cpu()->avx10_version);
}

TEST(zmm_gate_eligibility) {
    const t_cpu_info *c = cpu();
    int hwf = cpu_hw(F_AVX512F), osx = c->osxsave;
    int state = osx && (c->xcr0 & ZX_ZMMSTATE) == ZX_ZMMSTATE;
    int expect = hwf && state;
    CHECK_EQ(cpu_has(F_AVX512F), expect);
    CHECK_EQ(cpu_zmm_state_enabled(), state);
    if (state) CHECK(cpu_ymm_state_enabled());
    t_info("avx512_eligible", "%d", expect);
    t_info("reason", "%s", zx_why_not());
    t_info("xcr0", "0x%llx", (unsigned long long)(osx ? c->xcr0 : 0));
    if (hwf && !state) t_note("AVX-512 advertised by CPUID but not enabled by the OS: every AVX-512 case SKIPs");
    if (!hwf && state) t_note("XCR0 enables AVX-512 state although CPUID does not advertise AVX512F");
}

TEST(zmm_gate_xcr0_avx512_bits) {
    const t_cpu_info *c = cpu();
    if (!c->osxsave) SKIP("OSXSAVE=0: XGETBV not available");
    uint64_t x = c->xcr0, now = t_xgetbv(0);
    CHECKF(now == x, "XCR0 changed during the process: 0x%llx -> 0x%llx", (unsigned long long)x, (unsigned long long)now);
    unsigned hi = (unsigned)(x >> 5) & 7;
    CHECKF(hi == 0 || hi == 7, "XCR0[7:5]=%u%u%u: opmask/ZMM_Hi256/Hi16_ZMM must be enabled all together (XSETBV rule)",
           hi >> 2 & 1, hi >> 1 & 1, hi & 1);
    if (hi) CHECKF((x & 6) == 6, "XCR0[7:5] set but XCR0[2:1]=%u (AVX-512 state requires SSE and AVX state)", (unsigned)(x >> 1) & 3);
    CHECKF(x & 1, "XCR0[0] (x87) must always be 1");
    if (x & 4) CHECKF(x & 2, "XCR0[2] (AVX) set without XCR0[1] (SSE)");
    t_info("xcr0_avx512_bits", "0x%llx", (unsigned long long)(x & ZX_AVX512));
}

TEST(zmm_gate_xcr0_within_supported) {
    const t_cpu_info *c = cpu();
    if (!c->osxsave) SKIP("OSXSAVE=0: XGETBV not available");
    CHECKF((c->xcr0 & ~c->xcr0_supported) == 0, "XCR0=0x%llx enables components outside CPUID.(0Dh,0) EDX:EAX=0x%llx",
           (unsigned long long)c->xcr0, (unsigned long long)c->xcr0_supported);
    CHECKF((c->xcr0 & ZX_AVX512 & ~c->xcr0_supported) == 0, "AVX-512 XCR0 bits not enumerated as supported");
}

TEST(zmm_gate_avx512f_has_xsave_components) {
    if (!cpu_hw(F_AVX512F)) SKIP("AVX512F not advertised");
    const t_cpu_info *c = cpu();
    CHECKF(cpu_hw(F_XSAVE), "AVX512F advertised without XSAVE");
    CHECKF((c->xcr0_supported & ZX_AVX512) == ZX_AVX512, "AVX512F advertised but CPUID.(0Dh,0) supported mask 0x%llx lacks 0xE0",
           (unsigned long long)c->xcr0_supported);
    CHECKF((c->xcr0_supported & 7) == 7, "supported mask 0x%llx lacks x87/SSE/AVX", (unsigned long long)c->xcr0_supported);
}

TEST(zmm_gate_leaf_d_avx512_components) {
    const t_cpu_info *c = cpu();
    uint64_t sup = c->xcr0_supported;
    if (!(sup & ZX_AVX512)) SKIP("CPUID.(0Dh,0) does not enumerate AVX-512 components");
    CHECKF((sup & ZX_AVX512) == ZX_AVX512, "supported mask 0x%llx enumerates only part of 0xE0", (unsigned long long)sup);
    static const uint32_t want[8] = { 0, 0, 256, 0, 0, 64, 512, 1024 };
    static const uint32_t usual[8] = { 0, 0, 576, 0, 0, 0x440, 0x480, 0x680 };
    const int comps[] = { 2, 5, 6, 7 };
    for (int n = 0; n < 4; n++) {
        int i = comps[n];
        if (!(sup >> i & 1)) continue;
        const t_xcomp *x = &c->comp[i];
        CHECKF(x->size == want[i], "component %d (%s) size %u, architectural size %u", i, xstate_comp_name(i), x->size, want[i]);
        CHECKF(!x->supervisor, "component %d is user state (XCR0), reported as supervisor", i);
        CHECKF(x->offset >= 576, "component %d offset %u overlaps legacy area/header", i, x->offset);
        CHECKF(x->offset + x->size <= c->xsave_size_max, "component %d [%u,+%u) beyond CPUID.(0Dh,0).ECX=%u", i,
               x->offset, x->size, c->xsave_size_max);
        t_info(xstate_comp_name(i), "offset=%u size=%u align64=%d", x->offset, x->size, x->align64);
        if (x->offset != usual[i]) t_note("component %d at standard offset 0x%x (commonly 0x%x)", i, x->offset, usual[i]);
        for (int m = n + 1; m < 4; m++) {
            int j = comps[m];
            if (!(sup >> j & 1)) continue;
            const t_xcomp *y = &c->comp[j];
            CHECKF(x->offset + x->size <= y->offset || y->offset + y->size <= x->offset,
                   "components %d and %d overlap in the standard format", i, j);
        }
    }
}

TEST(zmm_gate_xsave_size_enabled) {
    const t_cpu_info *c = cpu();
    if (!c->osxsave) SKIP("OSXSAVE=0");
    if ((c->xcr0 & ZX_AVX512) != ZX_AVX512) SKIP("AVX-512 state not enabled in XCR0=0x%llx", (unsigned long long)c->xcr0);
    uint32_t end = 576;
    for (int i = 2; i < 64; i++)
        if ((c->xcr0 >> i & 1) && c->comp[i].valid && !c->comp[i].supervisor && c->comp[i].offset + c->comp[i].size > end)
            end = c->comp[i].offset + c->comp[i].size;
    t_cpuid_regs r; cpuid_ex(0xD, 0, &r);
    CHECK_EQ(r.ebx, c->xsave_size_enabled);
    CHECKF(c->xsave_size_enabled >= end, "CPUID.(0Dh,0).EBX=%u smaller than the end of the enabled components (%u)",
           c->xsave_size_enabled, end);
    CHECKF(c->xsave_size_enabled >= c->comp[7].offset + 1024, "EBX=%u does not cover Hi16_ZMM", c->xsave_size_enabled);
    CHECK(c->xsave_size_max >= c->xsave_size_enabled);
    if (c->xsave_size_enabled != end) t_note("EBX=%u, last enabled component ends at %u", c->xsave_size_enabled, end);
}

TEST(zmm_api_xstate_constants) {
    CHECK_EQ(XSTATE_AVX512_KMASK, 5);
    CHECK_EQ(XSTATE_AVX512_ZMM_H, 6);
    CHECK_EQ(XSTATE_AVX512_ZMM, 7);
    CHECK_EQ(XSTATE_MASK_AVX512, ZX_AVX512);
    CHECK_EQ(XSTATE_MASK_AVX512, T_XS_OPMASK | T_XS_ZMM_HI256 | T_XS_HI16_ZMM);
#ifdef T_X64
    CHECK_EQ(CONTEXT_XSTATE, 0x100040);
#else
    CHECK_EQ(CONTEXT_XSTATE, 0x10040);
#endif
}

TEST(zmm_api_enabled_features_avx512) {
    __typeof__(&GetEnabledXStateFeatures) en = T_PROC("kernel32.dll", GetEnabledXStateFeatures);
    if (!en) SKIP("GetEnabledXStateFeatures not exported");
    const t_cpu_info *c = cpu();
    DWORD64 api = en(), again = en();
    CHECK_EQ(api, again);
    CHECK_EQ(api, t_env_get()->api_xstate_features);
    t_info("api_enabled_features", "0x%llx", (unsigned long long)api);
    CHECKF((api & 3) == 3, "GetEnabledXStateFeatures()=0x%llx lacks XSTATE_MASK_LEGACY", (unsigned long long)api);
    uint64_t a = api & ZX_AVX512, x = c->osxsave ? (c->xcr0 & ZX_AVX512) : 0;
    CHECKF(a == 0 || a == ZX_AVX512, "API reports a partial AVX-512 mask 0x%llx", (unsigned long long)a);
    if (a != x && t_is_wine())
        t_expect_fail("Wine %s: GetEnabledXStateFeatures does not mirror XCR0 AVX-512 bits", t_wine_version());
    /* API claims usable state that XCR0 does not enable -> apps would #UD; the reverse hides a usable feature */
    CHECKF(!(a & ~x), "API reports AVX-512 (0x%llx) but XCR0=0x%llx does not enable it", (unsigned long long)a,
           (unsigned long long)c->xcr0);
    CHECKF(!(x & ~a), "XCR0 enables AVX-512 but GetEnabledXStateFeatures()=0x%llx does not report it", (unsigned long long)api);
}

TEST(zmm_api_processor_feature_avx512) {
    BOOL pf = IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE);
    t_info("pf_avx512f", "%d", pf ? 1 : 0);
    CHECKF(!pf || cpu_has(F_AVX512F), "IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE) is TRUE but %s", zx_why_not());
    if (!pf && cpu_has(F_AVX512F))
        t_note("AVX512F usable but PF_AVX512F_INSTRUCTIONS_AVAILABLE (41) not reported (older Windows builds do not know it)");
}

TEST(zmm_api_initialize_context_length) {
    zx_api a; NEED_API(&a);
    DWORD64 en = a.en();
    DWORD len = 0; CONTEXT *ctx = NULL;
    SetLastError(0);
    BOOL ok = a.init(NULL, ZX_XFLAGS, &ctx, &len);
    CHECK(!ok);
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECKF(len >= sizeof(CONTEXT), "needed length %lu < sizeof(CONTEXT) %u", (unsigned long)len, (unsigned)sizeof(CONTEXT));
    t_info("context_length", "%lu", (unsigned long)len);
    if ((en & ZX_AVX512) == ZX_AVX512) {
        CHECKF(len >= sizeof(CONTEXT) + 64 + 512 + 1024 * (T_NZMM > 8),
               "context length %lu cannot hold the enabled AVX-512 components", (unsigned long)len);
    }
    REQUIRE(len <= ZX_CTXBUF);
    DWORD small = len - 1;
    SetLastError(0);
    ok = a.init(g_ctxbuf[0], ZX_XFLAGS, &ctx, &small);
    CHECK(!ok);
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECK_EQ(small, len);
    DWORD l2 = len;
    ok = a.init(g_ctxbuf[0], ZX_XFLAGS, &ctx, &l2);
    REQUIRE(ok && ctx);
    CHECK((uint8_t *)ctx >= g_ctxbuf[0] && (uint8_t *)ctx + sizeof(CONTEXT) <= g_ctxbuf[0] + len);
    CHECK_EQ(ctx->ContextFlags & ZX_XFLAGS, ZX_XFLAGS);
}

TEST(zmm_api_set_mask_filters_disabled) {
    zx_api a; NEED_API(&a);
    DWORD64 en = a.en();
    const char *why = ""; CONTEXT *ctx = zx_context_new(&a, g_ctxbuf[0], ZX_XFLAGS, NULL, &why);
    REQUIREF(ctx, "%s", why);
    CHECK(a.setm(ctx, XSTATE_MASK_LEGACY | XSTATE_MASK_AVX | XSTATE_MASK_AVX512));
    DWORD64 m = 0;
    CHECK(a.getm(ctx, &m));
    t_info("mask_after_set", "0x%llx", (unsigned long long)m);
    CHECKF((m & ZX_AVX512) == (en & ZX_AVX512), "GetXStateFeaturesMask=0x%llx, enabled=0x%llx: the AVX-512 bits must be "
           "kept exactly when enabled", (unsigned long long)m, (unsigned long long)en);
    CHECKF((m & 4) == (en & 4), "AVX bit 0x%llx vs enabled 0x%llx", (unsigned long long)(m & 4), (unsigned long long)(en & 4));
    CHECK(a.setm(ctx, XSTATE_MASK_AVX512));
    CHECK(a.getm(ctx, &m));
    CHECKF((m & ~3ull) == (en & ZX_AVX512), "mask 0x%llx after SetXStateFeaturesMask(0xE0)", (unsigned long long)m);
}

TEST(zmm_api_locate_avx512_features) {
    zx_api a; NEED_API(&a);
    DWORD64 en = a.en();
    DWORD len = 0; const char *why = "";
    CONTEXT *ctx = zx_context_new(&a, g_ctxbuf[0], ZX_XFLAGS, &len, &why);
    REQUIREF(ctx, "%s", why);
    CHECK(a.setm(ctx, XSTATE_MASK_AVX | XSTATE_MASK_AVX512));
    static const DWORD want[8] = { 0, 0, 256, 0, 0, 64, 512, 1024 };
    uint8_t *lo = g_ctxbuf[0], *hi = g_ctxbuf[0] + len;
    uint8_t *ps[8] = { 0 };
    for (DWORD f = 5; f <= 7; f++) {
        DWORD l = 0xDEAD;
        uint8_t *p = a.loc(ctx, f, &l);
        ps[f] = p;
        if (en >> f & 1) {
            CHECKF(p != NULL, "LocateXStateFeature(%lu) NULL although the feature is enabled", (unsigned long)f);
            CHECKF(l == want[f], "LocateXStateFeature(%lu) length %lu, want %lu", (unsigned long)f, (unsigned long)l,
                   (unsigned long)want[f]);
            if (p) CHECKF(p >= lo && p + want[f] <= hi, "feature %lu [%p,+%lu) outside the context buffer", (unsigned long)f,
                          (void *)p, (unsigned long)want[f]);
        } else {
            CHECKF(p == NULL, "LocateXStateFeature(%lu)=%p for a feature that is not enabled", (unsigned long)f, (void *)p);
        }
    }
    if (ps[5] && ps[6]) CHECK(ps[5] + 64 <= ps[6] || ps[6] + 512 <= ps[5]);
    if (ps[6] && ps[7]) CHECK(ps[6] + 512 <= ps[7] || ps[7] + 1024 <= ps[6]);
    if (!(en & ZX_AVX512)) t_note("AVX-512 not enabled: verified LocateXStateFeature(5..7) == NULL");
}

TEST(zmm_api_locate_invalid_ids) {
    zx_api a; NEED_API(&a);
    const char *why = "";
    CONTEXT *ctx = zx_context_new(&a, g_ctxbuf[0], ZX_XFLAGS, NULL, &why);
    REQUIREF(ctx, "%s", why);
    DWORD l = 0;
    CHECK(a.loc(ctx, 64, &l) == NULL);
    CHECK(a.loc(ctx, 0xFFFFFFFFu, &l) == NULL);
    CHECK(a.loc(ctx, 0x80000005u, &l) == NULL);
    /* a context initialized without CONTEXT_XSTATE has no XSTATE area: AVX-512 cannot be located */
    CONTEXT *plain = zx_context_new(&a, g_ctxbuf[1], CONTEXT_ALL, NULL, &why);
    REQUIREF(plain, "%s", why);
    for (DWORD f = 5; f <= 7; f++) CHECKF(a.loc(plain, f, &l) == NULL, "feature %lu located without CONTEXT_XSTATE", (unsigned long)f);
}

TEST(zmm_api_locate_not_enabled_id) {
    zx_api a; NEED_API(&a);
    DWORD64 en = a.en();
    const char *why = "";
    CONTEXT *ctx = zx_context_new(&a, g_ctxbuf[0], ZX_XFLAGS, NULL, &why);
    REQUIREF(ctx, "%s", why);
    CHECK(a.setm(ctx, XSTATE_MASK_AVX | XSTATE_MASK_AVX512));
    /* RtlLocateExtendedFeature2: NULL unless the id is an enabled feature; bit 63 is never a feature */
    DWORD ids[] = { 63, 62, 5, 6, 7 };
    int n = 0;
    for (size_t i = 0; i < T_COUNTOF(ids); i++) {
        if (en >> ids[i] & 1) continue;
        n++;
        DWORD l = 0;
        void *p = a.loc(ctx, ids[i], &l);
        if (p && t_is_wine()) t_expect_fail("Wine %s: LocateXStateFeature returns a pointer for a feature id that is not enabled",
                                            t_wine_version());
        CHECKF(p == NULL, "LocateXStateFeature(%lu)=%p (length %lu) although feature %lu is not enabled (enabled 0x%llx)",
               (unsigned long)ids[i], p, (unsigned long)l, (unsigned long)ids[i], (unsigned long long)en);
    }
    t_info("ids_checked", "%d", n);
}

/* =============================================================================================
 * 2. fault behaviour that is testable on every host (children)
 * ============================================================================================= */
T_NOINLINE static void zmm_probe_evex_512(void) { __asm__ volatile("vpaddd %%zmm0, %%zmm0, %%zmm0" ::: "xmm0"); }
T_NOINLINE static void zmm_probe_kreg_kmovw(void) { __asm__ volatile("kmovw %%k1, %%eax" ::: "eax"); }
T_NOINLINE static void zx_xsetbv_same(void) {
    uint32_t lo, hi;
    __asm__ volatile("xorl %%ecx, %%ecx\n\txgetbv\n\txorl %%ecx, %%ecx\n\txsetbv" : "=a"(lo), "=d"(hi) :: "ecx", "memory");
    printf("XSETBV_RETURNED=0x%x%08x\n", hi, lo);
}
CHILD(zmm_child_evex) { (void)argc; (void)argv; printf("BEFORE=1\n"); fflush(stdout); zmm_probe_evex_512(); printf("AFTER=1\n"); return 0; }
CHILD(zmm_child_kmov) { (void)argc; (void)argv; printf("BEFORE=1\n"); fflush(stdout); zmm_probe_kreg_kmovw(); printf("AFTER=1\n"); return 0; }
CHILD(zmm_child_xsetbv) {
    (void)argc; (void)argv;
    printf("XCR0=0x%llx\n", (unsigned long long)t_xgetbv(0)); fflush(stdout);
    zx_xsetbv_same();
    return 0;
}

static t_child_result g_cr;

static void zx_expect_ud_child(const char *child, const char *what) {
    if (cpu_has(F_AVX512F)) SKIP("AVX-512 is usable here: the #UD path is not reachable");
    REQUIREF(t_child_run(child, "", 10000, &g_cr), "child did not start (error %lu)", (unsigned long)g_cr.spawn_error);
    CHECK(!g_cr.timed_out);
    char v[32];
    CHECKF(t_child_find(&g_cr, "BEFORE=", v, sizeof v), "child did not reach the probe");
    CHECKF(!t_child_find(&g_cr, "AFTER=", v, sizeof v), "%s executed although %s", what, zx_why_not());
    CHECKF(g_cr.exc_code == STATUS_ILLEGAL_INSTRUCTION, "%s: exception 0x%lx (exit 0x%lx), want STATUS_ILLEGAL_INSTRUCTION "
           "because %s", what, (unsigned long)g_cr.exc_code, (unsigned long)g_cr.exit_code, zx_why_not());
    t_info("ineligible_reason", "%s", zx_why_not());
}

TEST(zmm_fault_evex_ud_when_unusable) { zx_expect_ud_child("zmm_child_evex", "EVEX VPADDD zmm"); }
TEST(zmm_fault_kmov_ud_when_unusable) { zx_expect_ud_child("zmm_child_kmov", "VEX KMOVW k1->eax"); }

TEST(zmm_fault_xsetbv_user_mode) {
    if (!cpu()->osxsave) SKIP("OSXSAVE=0: XGETBV/XSETBV are #UD, nothing to test");
    REQUIREF(t_child_run("zmm_child_xsetbv", "", 10000, &g_cr), "child did not start");
    CHECK(!g_cr.timed_out);
    char v[64];
    CHECKF(!t_child_find(&g_cr, "XSETBV_RETURNED=", v, sizeof v), "XSETBV completed at CPL3 (value %s)", v);
    CHECKF(g_cr.exc_code != 0, "XSETBV at CPL3 must #GP; child exit 0x%lx", (unsigned long)g_cr.exit_code);
    CHECKF(g_cr.exc_code == STATUS_PRIVILEGED_INSTRUCTION || g_cr.exc_code == STATUS_ACCESS_VIOLATION,
           "XSETBV #GP reported as 0x%lx", (unsigned long)g_cr.exc_code);
    t_info("xsetbv_exception", "0x%lx", (unsigned long)g_cr.exc_code);
    CHECK_EQ(t_xgetbv(0), cpu()->xcr0);
}

/* =============================================================================================
 * 3. register file: round trips, zero-upper rules, VZEROUPPER/VZEROALL, XINUSE, XSAVE family
 * ============================================================================================= */
static zctx g_zc, g_zc2;

TEST(zmm_regs_roundtrip_patterns) {
    NEED_F();
    for (uint32_t tag = 1; tag <= 8; tag++) {
        zx_ctx_init(&g_zc, 0x100 + tag);
        zmm_rf_roundtrip(&g_zc);
        char lbl[48]; snprintf(lbl, sizeof lbl, "round trip tag %u", tag);
        t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, lbl);
    }
}

TEST(zmm_regs_roundtrip_special_values) {
    NEED_F();
    static const uint8_t fills[] = { 0x00, 0xFF, 0x55, 0xAA, 0x80, 0x01 };
    for (size_t f = 0; f < sizeof fills; f++) {
        zx_ctx_init(&g_zc, 1);
        memset(g_zc.in.v, fills[f], sizeof g_zc.in.v);
        for (int i = 0; i < 8; i++) g_zc.in.k[i] = g_zc.kbw ? 0x0101010101010101ull * fills[f] : 0x0101u * fills[f];
        zmm_rf_roundtrip(&g_zc);
        char lbl[48]; snprintf(lbl, sizeof lbl, "fill 0x%02x", fills[f]);
        t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, lbl);
    }
    /* walking one: a single set bit per register at a register-specific position */
    zx_ctx_init(&g_zc, 1);
    memset(g_zc.in.v, 0, sizeof g_zc.in.v);
    for (int r = 0; r < T_NZMM; r++) g_zc.in.v[r][(r * 13) % 64] = (uint8_t)(1u << (r % 8));
    for (int i = 0; i < 8; i++) g_zc.in.k[i] = 1ull << (g_zc.kbw ? (i * 9) % 64 : (i * 3) % 16);
    zmm_rf_roundtrip(&g_zc);
    t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, "walking one");
}

TEST(zmm_regs_hi16_independent_x64) {
    NEED_F(); NEED_X64();
    /* zmm16-31 are distinct registers: VZEROALL clears zmm0-15 only, and the high bank keeps its values */
    zx_ctx_init(&g_zc, 0x216);
    zmm_rf_vzeroall(&g_zc);
    for (int i = 16; i < 32; i++) {
        char lbl[32]; snprintf(lbl, sizeof lbl, "zmm%d after VZEROALL", i);
        CHECK_REG(lbl, g_zc.out.v[i], g_zc.in.v[i], 64);
    }
}

TEST(zmm_regs_vzeroupper) {
    NEED_F();
    /* SDM VZEROUPPER: bits MAXVL-1:128 of ZMM0-15 := 0; ZMM16-31 and the opmask registers unmodified */
    zx_ctx_init(&g_zc, 0x300);
    zmm_rf_vzeroupper(&g_zc);
    t_vregs exp = g_zc.in;
    for (int i = 0; i < (T_NZMM > 16 ? 16 : T_NZMM); i++) memset(exp.v[i] + 16, 0, 48);
    t_vregs_check(&g_zc.out, &exp, T_VK_ZMM, 0, -1, "after VZEROUPPER");
}

TEST(zmm_regs_vzeroall) {
    NEED_F();
    zx_ctx_init(&g_zc, 0x301);
    zmm_rf_vzeroall(&g_zc);
    t_vregs exp = g_zc.in;
    for (int i = 0; i < (T_NZMM > 16 ? 16 : T_NZMM); i++) memset(exp.v[i], 0, 64);
    t_vregs_check(&g_zc.out, &exp, T_VK_ZMM, 0, -1, "after VZEROALL");
}

/* single-register zero-upper rules (zmm1); src = 64-byte preload, val = new 16/32-byte value */
#define ZUP_FN(name, isa, insn) \
    T_TARGET(isa) T_NOINLINE static void name(const zv *src, const zv *val, zv *out) { \
        __asm__ volatile("vmovdqu64 (%0), %%zmm1\n\t" insn "\n\tvmovdqu64 %%zmm1, (%2)" \
                         :: "r"(src), "r"(val), "r"(out) : "xmm1", "xmm2", "memory"); }
ZUP_FN(zmm_up_vex128_load, "avx512f", "vmovdqu (%1), %%xmm1")
ZUP_FN(zmm_up_vex128_arith, "avx512f", "vmovdqu (%1), %%xmm2\n\tvpaddd %%xmm2, %%xmm1, %%xmm1")
ZUP_FN(zmm_up_vex256_load, "avx512f", "vmovdqu (%1), %%ymm1")
ZUP_FN(zmm_up_sse_load, "avx512f", "movdqu (%1), %%xmm1")
ZUP_FN(zmm_up_sse_arith, "avx512f", "movdqu (%1), %%xmm2\n\tpaddd %%xmm2, %%xmm1")
ZUP_FN(zmm_up_evex128_load, "avx512f,avx512vl", "vmovdqu32 (%1), %%xmm1")
ZUP_FN(zmm_up_evex256_load, "avx512f,avx512vl", "vmovdqu64 (%1), %%ymm1")

static void zx_upper_case(void (*fn)(const zv *, const zv *, zv *), int keep_bytes_from_val, int zero_upper, const char *what) {
    t_rng r; t_rng_seed(&r, 0x2A2A0000u + (uint32_t)keep_bytes_from_val * 7 + (uint32_t)zero_upper);
    for (int it = 0; it < 8; it++) {
        zv src, val, out, exp;
        zx_rand(&src, &r); zx_rand(&val, &r);
        fn(&src, &val, &out);
        exp = src;
        memcpy(exp.b, val.b, (size_t)keep_bytes_from_val);
        if (zero_upper) memset(exp.b + keep_bytes_from_val, 0, (size_t)(64 - keep_bytes_from_val));
        CHECK_REG(what, out.b, exp.b, 64);
    }
}
TEST(zmm_regs_vex128_zeroes_upper) {
    NEED_F();
    zx_upper_case(zmm_up_vex128_load, 16, 1, "zmm1 after VEX.128 VMOVDQU xmm1 (bits 511:128 := 0)");
}
TEST(zmm_regs_vex128_arith_zeroes_upper) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 77);
    for (int it = 0; it < 8; it++) {
        zv src, val, out, exp;
        zx_rand(&src, &r); zx_rand(&val, &r);
        zmm_up_vex128_arith(&src, &val, &out);
        memset(&exp, 0, sizeof exp);
        for (int i = 0; i < 4; i++) exp.d[i] = src.d[i] + val.d[i];
        CHECK_REG("zmm1 after VEX.128 VPADDD (bits 511:128 := 0)", out.b, exp.b, 64);
    }
}
TEST(zmm_regs_vex256_zeroes_upper) {
    NEED_F();
    zx_upper_case(zmm_up_vex256_load, 32, 1, "zmm1 after VEX.256 VMOVDQU ymm1 (bits 511:256 := 0)");
}
TEST(zmm_regs_legacy_sse_keeps_upper) {
    NEED_F();
    zx_upper_case(zmm_up_sse_load, 16, 0, "zmm1 after legacy MOVDQU xmm1 (bits 511:128 unchanged)");
    t_rng r; t_rng_seed(&r, 78);
    for (int it = 0; it < 4; it++) {
        zv src, val, out, exp;
        zx_rand(&src, &r); zx_rand(&val, &r);
        zmm_up_sse_arith(&src, &val, &out);
        exp = src;
        for (int i = 0; i < 4; i++) exp.d[i] = src.d[i] + val.d[i];
        CHECK_REG("zmm1 after legacy PADDD (bits 511:128 unchanged)", out.b, exp.b, 64);
    }
}
TEST(zmm_regs_evex128_zeroes_upper) {
    NEED_X(F_AVX512VL);
    zx_upper_case(zmm_up_evex128_load, 16, 1, "zmm1 after EVEX.128 VMOVDQU32 xmm1");
}
TEST(zmm_regs_evex256_zeroes_upper) {
    NEED_X(F_AVX512VL);
    zx_upper_case(zmm_up_evex256_load, 32, 1, "zmm1 after EVEX.256 VMOVDQU64 ymm1");
}

TEST(zmm_xinuse_reports_avx512_state) {
    NEED_F();
    if (!cpu_has(F_XGETBV1)) SKIP("XGETBV with ECX=1 not supported (CPUID.(0Dh,1):EAX[2]=0)");
    /* SDM 13.6: XINUSE[i]=0 only if component i is in its init state. Non-zero opmask, upper ZMM halves
     * and (x64) ZMM16-31 are not in init state, so bits 5, 6 (and 7) must read 1 while they are live. */
    zx_ctx_init(&g_zc, 0x400);
    zmm_rf_xinuse(&g_zc);
    uint64_t x = g_zc.aux;
    t_info("xinuse_live", "0x%llx", (unsigned long long)x);
    CHECKF(x & T_XS_OPMASK, "XINUSE=0x%llx: opmask non-zero but bit 5 clear", (unsigned long long)x);
    CHECKF(x & T_XS_ZMM_HI256, "XINUSE=0x%llx: ZMM_Hi256 non-zero but bit 6 clear", (unsigned long long)x);
#ifdef T_X64
    CHECKF(x & T_XS_HI16_ZMM, "XINUSE=0x%llx: ZMM16-31 non-zero but bit 7 clear", (unsigned long long)x);
#endif
    CHECKF(x & T_XS_AVX, "XINUSE=0x%llx: YMM_Hi128 non-zero but bit 2 clear", (unsigned long long)x);
    CHECKF(x & T_XS_SSE, "XINUSE=0x%llx: XMM non-zero but bit 1 clear", (unsigned long long)x);
    CHECKF((x & ~cpu()->xcr0) == 0, "XINUSE=0x%llx has bits outside XCR0", (unsigned long long)x);
    t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, "after XGETBV(1)");
    zx_ctx_init(&g_zc, 0x401);
    zmm_rf_xinuse_after_init(&g_zc);
    t_info("xinuse_after_zeroing", "0x%llx", (unsigned long long)g_zc.aux);
    if (g_zc.aux & (ZX_AVX512 | T_XS_AVX))
        t_note("XINUSE keeps AVX/AVX-512 bits after zeroing (0x%llx): allowed, the processor may report 1 for init state",
               (unsigned long long)g_zc.aux);
}

static uint8_t T_ALIGN(64) g_xarea[16384];
#define NEED_XSAVE() do { NEED_F(); if (!cpu_has(F_XSAVE)) SKIP("XSAVE not usable"); \
    if (cpu()->xsave_size_max > sizeof g_xarea) SKIP("XSAVE area %u > %u", cpu()->xsave_size_max, (unsigned)sizeof g_xarea); } while (0)

/* compare the AVX-512 related parts of a standard-format (or compacted, with offsets given) area to regs */
static void zx_check_area(const uint8_t *area, const uint32_t off[8], const t_vregs *v, const char *label) {
    char what[96];
    for (int i = 0; i < T_NXMM; i++) {
        snprintf(what, sizeof what, "%s legacy xmm%d", label, i);
        CHECK_REG(what, area + 160 + 16 * i, v->v[i], 16);
        snprintf(what, sizeof what, "%s AVX ymm%d[255:128]", label, i);
        CHECK_REG(what, area + off[2] + 16 * i, v->v[i] + 16, 16);
        snprintf(what, sizeof what, "%s ZMM_Hi256 zmm%d[511:256]", label, i);
        CHECK_REG(what, area + off[6] + 32 * i, v->v[i] + 32, 32);
    }
#ifdef T_X64
    for (int j = 0; j < 16; j++) {
        snprintf(what, sizeof what, "%s Hi16_ZMM zmm%d", label, 16 + j);
        CHECK_REG(what, area + off[7] + 64 * j, v->v[16 + j], 64);
    }
#endif
    for (int i = 0; i < 8; i++) {
        uint64_t k; memcpy(&k, area + off[5] + 8 * i, 8);
        CHECKF(k == v->k[i], "%s opmask k%d: area 0x%llx want 0x%llx", label, i, (unsigned long long)k, (unsigned long long)v->k[i]);
    }
}
static void zx_std_offsets(uint32_t off[8]) { for (int i = 0; i < 8; i++) off[i] = cpu()->comp[i].offset; }

TEST(zmm_xsave_avx512_layout) {
    NEED_XSAVE();
    zx_ctx_init(&g_zc, 0x500);
    memset(g_xarea, 0, sizeof g_xarea);
    g_zc.ptr = g_xarea; g_zc.rfbm = 0xE7;
    zmm_rf_xsave(&g_zc);
    uint64_t bv; memcpy(&bv, g_xarea + 512, 8);
    t_info("xstate_bv", "0x%llx", (unsigned long long)bv);
    CHECKF((bv & 0x66) == 0x66, "XSTATE_BV=0x%llx: SSE/AVX/opmask/ZMM_Hi256 hold non-init values and must be marked", (unsigned long long)bv);
#ifdef T_X64
    CHECKF(bv & 0x80, "XSTATE_BV=0x%llx: Hi16_ZMM holds non-init values", (unsigned long long)bv);
#endif
    CHECKF((bv & ~0xE7ull) == 0, "XSTATE_BV=0x%llx has bits outside RFBM", (unsigned long long)bv);
    uint32_t off[8]; zx_std_offsets(off);
    zx_check_area(g_xarea, off, &g_zc.in, "XSAVE");
    t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, "after XSAVE");
}

/* build `want` = in with ZMM_Hi256 / Hi16_ZMM / opmask taken from `src` */
static void zx_merge_avx512(t_vregs *want, const t_vregs *in, const t_vregs *src) {
    *want = *in;
    for (int i = 0; i < T_NXMM; i++) memcpy(want->v[i] + 32, src->v[i] + 32, 32);
#ifdef T_X64
    for (int j = 16; j < 32; j++) memcpy(want->v[j], src->v[j], 64);
#endif
    for (int i = 0; i < 8; i++) want->k[i] = src->k[i];
}
static void zx_area_put_avx512(uint8_t *area, const uint32_t off[8], const t_vregs *src) {
    for (int i = 0; i < 16; i++) memcpy(area + off[6] + 32 * i, src->v[i < T_NXMM ? i : 0] + 32, 32);
#ifdef T_X64
    for (int j = 0; j < 16; j++) memcpy(area + off[7] + 64 * j, src->v[16 + j], 64);
#endif
    for (int i = 0; i < 8; i++) memcpy(area + off[5] + 8 * i, &src->k[i], 8);
}

TEST(zmm_xrstor_avx512_roundtrip) {
    NEED_XSAVE();
    uint32_t off[8]; zx_std_offsets(off);
    zx_ctx_init(&g_zc, 0x510);                 /* A: saved */
    memset(g_xarea, 0, sizeof g_xarea);
    g_zc.ptr = g_xarea; g_zc.rfbm = 0xE7;
    zmm_rf_xsave(&g_zc);
    zx_ctx_init(&g_zc2, 0x511);                /* B: written into the area's AVX-512 components */
    zx_area_put_avx512(g_xarea, off, &g_zc2.in);
    zctx *c = &g_zc2;                          /* C: live before XRSTOR */
    static t_vregs bvals; bvals = g_zc2.in;
    zx_ctx_init(c, 0x512);
    c->ptr = g_xarea; c->rfbm = 0xE7;
    zmm_rf_xrstor(c);
    t_vregs want;
    zx_merge_avx512(&want, &g_zc.in, &bvals);  /* low 256 bits from A, AVX-512 parts from B */
    t_vregs_check(&c->out, &want, T_VK_ZMM, 0, -1, "after XRSTOR of modified AVX-512 components");
}

TEST(zmm_xrstor_init_avx512_components) {
    NEED_XSAVE();
    /* SDM 13.8: RFBM[i]=1 and XSTATE_BV[i]=0 -> component i is set to its init state (all zero) */
    zx_ctx_init(&g_zc, 0x520);
    memset(g_xarea, 0, sizeof g_xarea);
    g_zc.ptr = g_xarea; g_zc.rfbm = 0xE7;
    zmm_rf_xsave(&g_zc);
    uint64_t bv; memcpy(&bv, g_xarea + 512, 8); bv &= ~ZX_AVX512; memcpy(g_xarea + 512, &bv, 8);
    zx_ctx_init(&g_zc2, 0x521);
    g_zc2.ptr = g_xarea; g_zc2.rfbm = 0xE7;
    zmm_rf_xrstor(&g_zc2);
    static t_vregs zero; memset(&zero, 0, sizeof zero);
    t_vregs want;
    zx_merge_avx512(&want, &g_zc.in, &zero);
    t_vregs_check(&g_zc2.out, &want, T_VK_ZMM, 0, -1, "after XRSTOR with XSTATE_BV[7:5]=0");
}

TEST(zmm_xrstor_rfbm_excludes_avx512) {
    NEED_XSAVE();
    /* components outside RFBM are not touched by XRSTOR; the SSE/AVX components do not reach bits 511:256 */
    zx_ctx_init(&g_zc, 0x530);
    memset(g_xarea, 0, sizeof g_xarea);
    g_zc.ptr = g_xarea; g_zc.rfbm = 0xE7;
    zmm_rf_xsave(&g_zc);
    zx_ctx_init(&g_zc2, 0x531);
    g_zc2.ptr = g_xarea; g_zc2.rfbm = 0x07;
    zmm_rf_xrstor(&g_zc2);
    t_vregs want;
    zx_merge_avx512(&want, &g_zc.in, &g_zc2.in);
    t_vregs_check(&g_zc2.out, &want, T_VK_ZMM, 0, -1, "after XRSTOR RFBM=0x7");
}

TEST(zmm_xsavec_avx512_compacted) {
    NEED_XSAVE();
    if (!cpu_has(F_XSAVEC)) SKIP("XSAVEC not supported");
    uint64_t rfbm = 0xE7;
    zx_ctx_init(&g_zc, 0x540);
    memset(g_xarea, 0, sizeof g_xarea);
    g_zc.ptr = g_xarea; g_zc.rfbm = rfbm;
    zmm_rf_xsavec(&g_zc);
    uint64_t bv, xc; memcpy(&bv, g_xarea + 512, 8); memcpy(&xc, g_xarea + 520, 8);
    t_info("xstate_bv", "0x%llx", (unsigned long long)bv);
    t_info("xcomp_bv", "0x%llx", (unsigned long long)xc);
    CHECKF(xc == (rfbm | (1ull << 63)), "XCOMP_BV=0x%llx want 0x%llx", (unsigned long long)xc, (unsigned long long)(rfbm | (1ull << 63)));
    CHECKF((bv & 0x66) == 0x66, "XSTATE_BV=0x%llx", (unsigned long long)bv);
    uint32_t off[8];
    for (int i = 0; i < 8; i++) off[i] = (rfbm >> i & 1) && i >= 2 ? xsave_compacted_offset(rfbm, i) : 0;
    t_info("compacted_offsets", "avx=%u k=%u hi256=%u hi16=%u", off[2], off[5], off[6], off[7]);
    zx_check_area(g_xarea, off, &g_zc.in, "XSAVEC");
    /* XRSTOR understands the compacted form (XCOMP_BV[63]=1) */
    zx_ctx_init(&g_zc2, 0x541);
    g_zc2.ptr = g_xarea; g_zc2.rfbm = rfbm;
    zmm_rf_xrstor(&g_zc2);
    t_vregs_check(&g_zc2.out, &g_zc.in, T_VK_ZMM, 0, -1, "after XRSTOR of an XSAVEC image");
}

/* =============================================================================================
 * 4. opmask registers
 * ============================================================================================= */
typedef struct T_ALIGN(64) zk { uint64_t a, b; uint64_t o[24]; uint8_t fl[8]; } zk;
_Static_assert(offsetof(zk, fl) == 208, "asm below stores flags at 208..211");

T_TARGET("avx512f") T_NOINLINE static void zmm_kreg_ops_w(zk *p) {
    __asm__ volatile(
        "kmovw 0(%0), %%k1\n\tkmovw 8(%0), %%k2\n\t"
        "kandw %%k2, %%k1, %%k3\n\tkmovw %%k3, 16(%0)\n\t"
        "korw %%k2, %%k1, %%k3\n\tkmovw %%k3, 24(%0)\n\t"
        "kxorw %%k2, %%k1, %%k3\n\tkmovw %%k3, 32(%0)\n\t"
        "kxnorw %%k2, %%k1, %%k3\n\tkmovw %%k3, 40(%0)\n\t"
        "kandnw %%k2, %%k1, %%k3\n\tkmovw %%k3, 48(%0)\n\t"   /* k3 = ~k1 & k2 */
        "knotw %%k1, %%k3\n\tkmovw %%k3, 56(%0)\n\t"
        "kunpckbw %%k2, %%k1, %%k3\n\tkmovw %%k3, 64(%0)\n\t" /* k3 = k1[7:0]:k2[7:0] */
        "kshiftlw $3, %%k1, %%k3\n\tkmovw %%k3, 72(%0)\n\t"
        "kshiftrw $5, %%k1, %%k3\n\tkmovw %%k3, 80(%0)\n\t"
        "kshiftlw $15, %%k1, %%k3\n\tkmovw %%k3, 88(%0)\n\t"
        "kshiftrw $16, %%k1, %%k3\n\tkmovw %%k3, 96(%0)\n\t"
        "kshiftlw $200, %%k1, %%k3\n\tkmovw %%k3, 104(%0)\n\t"
        "kmovw %%k1, %%eax\n\tmovl %%eax, 112(%0)\n\t"         /* k -> r32 zero-extends */
        "movl 8(%0), %%eax\n\tkmovw %%eax, %%k4\n\tkmovw %%k4, 120(%0)\n\t"
        "kortestw %%k2, %%k1\n\tsetz 208(%0)\n\tsetc 209(%0)\n\t"
        :: "r"(p) : "eax", "k1", "k2", "k3", "k4", "memory", "cc");
}
T_TARGET("avx512f,avx512bw") T_NOINLINE static void zmm_kreg_ops_q_bw(zk *p) {
    __asm__ volatile(
        "kmovq 0(%0), %%k1\n\tkmovq 8(%0), %%k2\n\t"
        "kandq %%k2, %%k1, %%k3\n\tkmovq %%k3, 16(%0)\n\t"
        "korq %%k2, %%k1, %%k3\n\tkmovq %%k3, 24(%0)\n\t"
        "kxorq %%k2, %%k1, %%k3\n\tkmovq %%k3, 32(%0)\n\t"
        "kxnorq %%k2, %%k1, %%k3\n\tkmovq %%k3, 40(%0)\n\t"
        "kandnq %%k2, %%k1, %%k3\n\tkmovq %%k3, 48(%0)\n\t"
        "knotq %%k1, %%k3\n\tkmovq %%k3, 56(%0)\n\t"
        "kunpckdq %%k2, %%k1, %%k3\n\tkmovq %%k3, 64(%0)\n\t"  /* k1[31:0]:k2[31:0] */
        "kshiftlq $33, %%k1, %%k3\n\tkmovq %%k3, 72(%0)\n\t"
        "kshiftrq $63, %%k1, %%k3\n\tkmovq %%k3, 80(%0)\n\t"
        "kshiftlq $64, %%k1, %%k3\n\tkmovq %%k3, 88(%0)\n\t"
        "kunpckwd %%k2, %%k1, %%k3\n\tkmovq %%k3, 96(%0)\n\t"  /* k1[15:0]:k2[15:0], upper 32 := 0 */
        "kaddq %%k2, %%k1, %%k3\n\tkmovq %%k3, 104(%0)\n\t"
        "kaddd %%k2, %%k1, %%k3\n\tkmovq %%k3, 112(%0)\n\t"
        "kmovq %%k1, %%k4\n\tkmovw 8(%0), %%k4\n\tkmovq %%k4, 120(%0)\n\t"   /* KMOVW load zero-extends to 64 */
        "kmovq %%k1, %%k4\n\tkmovd 8(%0), %%k4\n\tkmovq %%k4, 128(%0)\n\t"   /* KMOVD load zero-extends */
        "kandw %%k2, %%k1, %%k4\n\tkmovq %%k4, 136(%0)\n\t"                 /* 16-bit op clears bits 63:16 */
        "kortestq %%k2, %%k1\n\tsetz 208(%0)\n\tsetc 209(%0)\n\t"
        "kortestd %%k2, %%k1\n\tsetz 210(%0)\n\tsetc 211(%0)\n\t"
        :: "r"(p) : "k1", "k2", "k3", "k4", "memory", "cc");
}
T_TARGET("avx512f,avx512dq") T_NOINLINE static void zmm_kreg_ops_dq(zk *p) {
    __asm__ volatile(
        "kmovw 0(%0), %%k1\n\tkmovw 8(%0), %%k2\n\t"
        "kaddw %%k2, %%k1, %%k3\n\tkmovw %%k3, 16(%0)\n\t"
        "kandb %%k2, %%k1, %%k3\n\tkmovw %%k3, 24(%0)\n\t"     /* 8-bit op: bits 15:8 := 0 */
        "knotb %%k1, %%k3\n\tkmovw %%k3, 32(%0)\n\t"
        "kshiftlb $3, %%k1, %%k3\n\tkmovw %%k3, 40(%0)\n\t"
        "kmovw %%k1, %%k4\n\tkmovb 8(%0), %%k4\n\tkmovw %%k4, 48(%0)\n\t"  /* KMOVB zero-extends */
        "kaddb %%k2, %%k1, %%k3\n\tkmovw %%k3, 56(%0)\n\t"
        "ktestw %%k2, %%k1\n\tsetz 208(%0)\n\tsetc 209(%0)\n\t"
        "ktestb %%k2, %%k1\n\tsetz 210(%0)\n\tsetc 211(%0)\n\t"
        :: "r"(p) : "k1", "k2", "k3", "k4", "memory", "cc");
}

TEST(zmm_kreg_logic_w) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x6B01);
    static const uint16_t fixed[][2] = { { 0, 0 }, { 0xFFFF, 0xFFFF }, { 0xFFFF, 0 }, { 0x00FF, 0xFF00 }, { 0x8001, 0x7FFE } };
    for (int it = 0; it < 40; it++) {
        zk p; memset(&p, 0xA5, sizeof p);
        uint16_t a, b;
        if (it < 5) { a = fixed[it][0]; b = fixed[it][1]; } else { a = (uint16_t)t_rng_u32(&r); b = (uint16_t)t_rng_u32(&r); }
        p.a = a; p.b = b;
        zmm_kreg_ops_w(&p);
        uint16_t w[13]; for (int i = 0; i < 13; i++) w[i] = (uint16_t)p.o[i];
        uint16_t e[13] = { (uint16_t)(a & b), (uint16_t)(a | b), (uint16_t)(a ^ b), (uint16_t)~(a ^ b), (uint16_t)(~a & b),
                           (uint16_t)~a, (uint16_t)((a & 0xFF) << 8 | (b & 0xFF)), (uint16_t)(a << 3), (uint16_t)(a >> 5),
                           (uint16_t)(a << 15), 0, 0, 0 };
        static const char *nm[] = { "kandw", "korw", "kxorw", "kxnorw", "kandnw", "knotw", "kunpckbw", "kshiftlw 3",
                                    "kshiftrw 5", "kshiftlw 15", "kshiftrw 16", "kshiftlw 200" };
        for (int i = 0; i < 12; i++)
            CHECKF(w[i] == e[i], "%s a=0x%04x b=0x%04x: got 0x%04x want 0x%04x", nm[i], a, b, w[i], e[i]);
        uint32_t g32 = (uint32_t)p.o[12];
        CHECKF(g32 == a, "kmovw k->r32 got 0x%08x want 0x%08x (zero-extended)", g32, a);
        CHECKF((uint16_t)p.o[13] == b, "kmovw r32->k->m16 got 0x%04x want 0x%04x", (unsigned)(uint16_t)p.o[13], b);
        uint16_t orv = (uint16_t)(a | b);
        CHECKF(p.fl[0] == (orv == 0), "kortestw ZF a=0x%04x b=0x%04x", a, b);
        CHECKF(p.fl[1] == (orv == 0xFFFF), "kortestw CF a=0x%04x b=0x%04x", a, b);
    }
}

TEST(zmm_kreg_logic_q_bw) {
    NEED_X(F_AVX512BW);
    t_rng r; t_rng_seed(&r, 0x6B02);
    for (int it = 0; it < 40; it++) {
        zk p; memset(&p, 0xA5, sizeof p);
        uint64_t a = it == 0 ? ~0ull : it == 1 ? 0 : t_rng_next(&r), b = it == 0 ? ~0ull : it == 1 ? 0 : t_rng_next(&r);
        if (it == 2) { a = 0xFFFFFFFF00000000ull; b = 0x00000000FFFFFFFFull; }
        p.a = a; p.b = b;
        zmm_kreg_ops_q_bw(&p);
        uint64_t e[16] = { a & b, a | b, a ^ b, ~(a ^ b), ~a & b, ~a, (a & 0xFFFFFFFFull) << 32 | (b & 0xFFFFFFFFull),
                           a << 33, a >> 63, 0, (a & 0xFFFF) << 16 | (b & 0xFFFF), a + b,
                           (uint64_t)(uint32_t)((uint32_t)a + (uint32_t)b), b & 0xFFFF, b & 0xFFFFFFFFull, a & b & 0xFFFF };
        static const char *nm[] = { "kandq", "korq", "kxorq", "kxnorq", "kandnq", "knotq", "kunpckdq", "kshiftlq 33",
                                    "kshiftrq 63", "kshiftlq 64", "kunpckwd", "kaddq", "kaddd", "kmovw m16 zero-extends",
                                    "kmovd m32 zero-extends", "kandw clears 63:16" };
        for (int i = 0; i < 16; i++)
            CHECKF(p.o[i] == e[i], "%s a=0x%llx b=0x%llx: got 0x%llx want 0x%llx", nm[i], (unsigned long long)a,
                   (unsigned long long)b, (unsigned long long)p.o[i], (unsigned long long)e[i]);
        CHECKF(p.fl[0] == ((a | b) == 0), "kortestq ZF");
        CHECKF(p.fl[1] == ((a | b) == ~0ull), "kortestq CF");
        CHECKF(p.fl[2] == ((uint32_t)(a | b) == 0), "kortestd ZF");
        CHECKF(p.fl[3] == ((uint32_t)(a | b) == 0xFFFFFFFFu), "kortestd CF");
    }
}

TEST(zmm_kreg_logic_dq) {
    NEED_X(F_AVX512DQ);
    t_rng r; t_rng_seed(&r, 0x6B03);
    for (int it = 0; it < 40; it++) {
        zk p; memset(&p, 0xA5, sizeof p);
        uint16_t a = (uint16_t)t_rng_u32(&r), b = (uint16_t)t_rng_u32(&r);
        if (it == 0) { a = 0xFFFF; b = 1; }
        if (it == 1) { a = 0x00F0; b = 0x0F0F; }
        p.a = a; p.b = b;
        zmm_kreg_ops_dq(&p);
        uint16_t e[6] = { (uint16_t)(a + b), (uint16_t)(a & b & 0xFF), (uint16_t)(~a & 0xFF), (uint16_t)((a << 3) & 0xFF),
                          (uint16_t)(b & 0xFF), (uint16_t)((a + b) & 0xFF) };
        static const char *nm[] = { "kaddw", "kandb", "knotb", "kshiftlb 3", "kmovb m8 zero-extends", "kaddb" };
        for (int i = 0; i < 6; i++)
            CHECKF((uint16_t)p.o[i] == e[i], "%s a=0x%04x b=0x%04x: got 0x%04x want 0x%04x", nm[i], a, b,
                   (unsigned)(uint16_t)p.o[i], e[i]);
        CHECKF(p.fl[0] == ((a & b) == 0), "ktestw ZF a=0x%04x b=0x%04x", a, b);
        CHECKF(p.fl[1] == ((~a & b & 0xFFFF) == 0), "ktestw CF a=0x%04x b=0x%04x", a, b);
        CHECKF(p.fl[2] == ((a & b & 0xFF) == 0), "ktestb ZF");
        CHECKF(p.fl[3] == ((~a & b & 0xFF) == 0), "ktestb CF");
    }
}

/* k0 as an EVEX writemask encoding means "no masking"; k0 is still a normal opmask destination/source */
T_TARGET("avx512f") T_NOINLINE static void zmm_kreg_k0(zop *p) {
    __asm__ volatile("vmovdqu64 64(%0), %%zmm1\n\tvmovdqu64 128(%0), %%zmm2\n\tkxorw %%k0, %%k0, %%k0\n\t"
                     "vpaddd %%zmm2, %%zmm1, %%zmm0\n\tvmovdqu64 %%zmm0, 192(%0)\n\t"
                     "vpcmpeqd %%zmm1, %%zmm1, %%k0\n\tkmovw %%k0, 264(%0)\n\t"
                     :: "r"(p) : "xmm0", "xmm1", "xmm2", "k0", "memory");
}
TEST(zmm_kreg_k0_means_unmasked) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x6B00);
    static zop p; zx_rand(&p.a, &r); zx_rand(&p.b, &r);
    p.ko = 0;
    zmm_kreg_k0(&p);
    zv e; for (int i = 0; i < 16; i++) e.d[i] = p.a.d[i] + p.b.d[i];
    CHECK_LANES("vpaddd with k0==0 (no writemask) writes every lane", p.o.d, e.d, 16, 4);
    CHECK_EQ(p.ko & 0xFFFF, 0xFFFF);
}

/* =============================================================================================
 * 5. masking: merge vs zero, element widths, FP exception suppression
 * ============================================================================================= */
ZOP_FN(zmm_m_paddd, "avx512f", "kmovw", "vpaddd %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_m_paddd_z, "avx512f", "kmovw", "vpaddd %%zmm2, %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_m_paddq, "avx512f", "kmovw", "vpaddq %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_m_paddq_z, "avx512f", "kmovw", "vpaddq %%zmm2, %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_m_paddb_bw, "avx512f,avx512bw", "kmovq", "vpaddb %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovq")
ZOP_FN(zmm_m_paddb_z_bw, "avx512f,avx512bw", "kmovq", "vpaddb %%zmm2, %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovq")
ZOP_FN(zmm_m_psubw_bw, "avx512f,avx512bw", "kmovd", "vpsubw %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovq")
ZOP_FN(zmm_m_psubw_z_bw, "avx512f,avx512bw", "kmovd", "vpsubw %%zmm2, %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovq")
ZOP_FN(zmm_m_movdqa32, "avx512f", "kmovw", "vmovdqa32 %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_m_movdqa64_z, "avx512f", "kmovw", "vmovdqa64 %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_m_blendmd, "avx512f", "kmovw", "vpblendmd %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_m_blendmpd, "avx512f", "kmovw", "vblendmpd %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_m_divps, "avx512f", "kmovw", "vdivps %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_m_divps_z, "avx512f", "kmovw", "vdivps %%zmm2, %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_m_sqrtpd, "avx512f", "kmovw", "vsqrtpd %%zmm1, %%zmm0%{%%k1%}", "kmovw")

/* generic masked-integer check: op(a,b) per lane of width w (1,2,4,8) */
typedef uint64_t (*zx_lane_op)(uint64_t a, uint64_t b, int w);
static uint64_t zx_lane_get(const zv *v, int i, int w) { uint64_t x = 0; memcpy(&x, v->b + i * w, (size_t)w); return x; }
static void zx_lane_put(zv *v, int i, int w, uint64_t x) { memcpy(v->b + i * w, &x, (size_t)w); }
static uint64_t zx_op_add(uint64_t a, uint64_t b, int w) { return (a + b) & zx_lanes_mask(w * 8); }
static uint64_t zx_op_sub(uint64_t a, uint64_t b, int w) { return (a - b) & zx_lanes_mask(w * 8); }
static uint64_t zx_op_mov_a(uint64_t a, uint64_t b, int w) { (void)b; (void)w; return a; }
static uint64_t zx_op_blend(uint64_t a, uint64_t b, int w) { (void)a; (void)w; return b; }

static void zx_mask_case(void (*fn)(zop *), int w, int zero, zx_lane_op op, uint64_t blend_else_a, const char *what, uint32_t seed) {
    t_rng r; t_rng_seed(&r, seed);
    int lanes = 64 / w;
    static zop p;
    for (int it = 0; it < 24; it++) {
        zx_rand(&p.s, &r); zx_rand(&p.a, &r); zx_rand(&p.b, &r);
        p.k = it == 0 ? 0 : it == 1 ? ~0ull : t_rng_next(&r);
        zop_run(fn, &p);
        zv e;
        for (int i = 0; i < lanes; i++) {
            uint64_t v;
            if (p.k >> i & 1) v = op(zx_lane_get(&p.a, i, w), zx_lane_get(&p.b, i, w), w);
            else if (blend_else_a) v = zx_lane_get(&p.a, i, w);          /* VPBLENDM: k=0 selects src1 */
            else v = zero ? 0 : zx_lane_get(&p.s, i, w);
            zx_lane_put(&e, i, w, v);
        }
        char lbl[128]; snprintf(lbl, sizeof lbl, "%s k=0x%llx", what, (unsigned long long)p.k);
        CHECK_LANES(lbl, p.o.b, e.b, lanes, w);
    }
}
TEST(zmm_mask_merge_dword) { NEED_F(); zx_mask_case(zmm_m_paddd, 4, 0, zx_op_add, 0, "vpaddd {k1}", 0x7301); }
TEST(zmm_mask_zero_dword) { NEED_F(); zx_mask_case(zmm_m_paddd_z, 4, 1, zx_op_add, 0, "vpaddd {k1}{z}", 0x7302); }
TEST(zmm_mask_merge_zero_qword) {
    NEED_F();   /* 8 lanes: k bits 15:8 must be ignored */
    zx_mask_case(zmm_m_paddq, 8, 0, zx_op_add, 0, "vpaddq {k1}", 0x7303);
    zx_mask_case(zmm_m_paddq_z, 8, 1, zx_op_add, 0, "vpaddq {k1}{z}", 0x7304);
}
TEST(zmm_mask_byte_bw) {
    NEED_X(F_AVX512BW);
    zx_mask_case(zmm_m_paddb_bw, 1, 0, zx_op_add, 0, "vpaddb {k1} (64-bit mask)", 0x7305);
    zx_mask_case(zmm_m_paddb_z_bw, 1, 1, zx_op_add, 0, "vpaddb {k1}{z}", 0x7306);
}
TEST(zmm_mask_word_bw) {
    NEED_X(F_AVX512BW);
    zx_mask_case(zmm_m_psubw_bw, 2, 0, zx_op_sub, 0, "vpsubw {k1} (32-bit mask)", 0x7307);
    zx_mask_case(zmm_m_psubw_z_bw, 2, 1, zx_op_sub, 0, "vpsubw {k1}{z}", 0x7308);
}
TEST(zmm_mask_moves) {
    NEED_F();
    zx_mask_case(zmm_m_movdqa32, 4, 0, zx_op_mov_a, 0, "vmovdqa32 reg {k1}", 0x7309);
    zx_mask_case(zmm_m_movdqa64_z, 8, 1, zx_op_mov_a, 0, "vmovdqa64 reg {k1}{z}", 0x730A);
}
TEST(zmm_mask_blendm) {
    NEED_F();
    zx_mask_case(zmm_m_blendmd, 4, 0, zx_op_blend, 1, "vpblendmd", 0x730B);
    zx_mask_case(zmm_m_blendmpd, 8, 0, zx_op_blend, 1, "vblendmpd", 0x730C);
}

#define MX_FLAGS 0x3Fu
/* FP expectations assume RN, DAZ=0, FTZ=0, all exceptions masked: pin MXCSR (the harness restores it per case) */
#define ZX_FP_DEFAULT() t_set_mxcsr(0x1F80)
TEST(zmm_mask_fp_exceptions_suppressed) {
    NEED_F();
    ZX_FP_DEFAULT();
    /* SDM 15.6 / exception class E2: masked-off elements do not report FP exceptions */
    static zop p;
    unsigned mx0 = t_get_mxcsr();
    for (int i = 0; i < 16; i++) {
        p.s.f[i] = -7.0f;
        if (i % 4 == 0) { p.a.f[i] = 0.0f; p.b.f[i] = 0.0f; }          /* invalid */
        else if (i % 4 == 1) { p.a.f[i] = 1.0f; p.b.f[i] = 0.0f; }     /* divide by zero */
        else if (i % 4 == 2) { p.a.f[i] = 1.0f; p.b.f[i] = 3.0f; }     /* inexact */
        else { p.a.f[i] = 1e-30f; p.b.f[i] = 1e30f; }                  /* underflow + inexact */
    }
    /* active lanes get an exact, exception-free division instead */
    p.k = 0x8888 >> 3;   /* lanes 0,4,8,12 active */
    for (int i = 0; i < 16; i++) if (p.k >> i & 1) { p.a.f[i] = 6.0f; p.b.f[i] = 3.0f; }
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    zop_run(zmm_m_divps, &p);
    unsigned mx = t_get_mxcsr();
    t_set_mxcsr(mx0);
    CHECKF((mx & MX_FLAGS) == 0, "MXCSR flags 0x%x after vdivps with all exception-raising lanes masked off", mx & MX_FLAGS);
    for (int i = 0; i < 16; i++) {
        float e = (p.k >> i & 1) ? 2.0f : -7.0f;
        CHECKF(zx_f2u(p.o.f[i]) == zx_f2u(e), "lane %d got 0x%08x want 0x%08x", i, zx_f2u(p.o.f[i]), zx_f2u(e));
    }
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    zop_run(zmm_m_divps_z, &p);
    mx = t_get_mxcsr();
    t_set_mxcsr(mx0);
    CHECKF((mx & MX_FLAGS) == 0, "MXCSR flags 0x%x after zero-masked vdivps", mx & MX_FLAGS);
    /* vsqrtpd of negative numbers in masked-off lanes */
    for (int i = 0; i < 8; i++) { p.a.g[i] = (i & 1) ? -4.0 : 16.0; p.s.g[i] = 5.0; }
    p.k = 0x55;   /* even lanes (16.0) active */
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    zop_run(zmm_m_sqrtpd, &p);
    mx = t_get_mxcsr();
    t_set_mxcsr(mx0);
    CHECKF((mx & MX_FLAGS) == 0, "MXCSR flags 0x%x after masked vsqrtpd(-4) lanes", mx & MX_FLAGS);
    for (int i = 0; i < 8; i++) CHECK(p.o.g[i] == ((i & 1) ? 5.0 : 4.0));
}

TEST(zmm_mask_fp_exceptions_reported) {
    NEED_F();
    ZX_FP_DEFAULT();
    static zop p;
    unsigned mx0 = t_get_mxcsr();
    for (int i = 0; i < 16; i++) { p.s.f[i] = -7.0f; p.a.f[i] = 6.0f; p.b.f[i] = 3.0f; }
    p.a.f[3] = 0.0f; p.b.f[3] = 0.0f;     /* invalid, active */
    p.a.f[9] = -1.0f; p.b.f[9] = 0.0f;    /* divide by zero, active */
    p.k = 0xFFFF;
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    zop_run(zmm_m_divps, &p);
    unsigned mx = t_get_mxcsr();
    t_set_mxcsr(mx0);
    CHECKF(mx & 0x01, "IE not set by an active 0/0 lane (MXCSR 0x%x)", mx);
    CHECKF(mx & 0x04, "ZE not set by an active x/0 lane (MXCSR 0x%x)", mx);
    CHECKF(zx_f2u(p.o.f[3]) == 0xFFC00000u, "0/0 -> QNaN indefinite, got 0x%08x", zx_f2u(p.o.f[3]));
    CHECKF(zx_f2u(p.o.f[9]) == 0xFF800000u, "-1/0 -> -inf, got 0x%08x", zx_f2u(p.o.f[9]));
    CHECKF(p.o.f[0] == 2.0f, "lane 0 %g", (double)p.o.f[0]);
}

/* =============================================================================================
 * 6. embedded broadcast and embedded rounding / SAE
 * ============================================================================================= */
ZOP_FN(zmm_b_paddd, "avx512f", "kmovw", "vpaddd 128(%0)%{1to16%}, %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_b_pandq, "avx512f", "kmovw", "vpandq 128(%0)%{1to8%}, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_b_mulps_z, "avx512f", "kmovw", "vmulps 128(%0)%{1to16%}, %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_b_fmadd231pd, "avx512f", "kmovw", "vfmadd231pd 128(%0)%{1to8%}, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_b_pcmpgtd, "avx512f", "kmovw", "vpcmpgtd 128(%0)%{1to16%}, %%zmm1, %%k2%{%%k1%}", "kmovw")

TEST(zmm_bcst_integer) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0xBC01);
    static zop p;
    for (int it = 0; it < 16; it++) {
        zx_rand(&p.s, &r); zx_rand(&p.a, &r); zx_rand(&p.b, &r);
        p.k = t_rng_next(&r);
        zop_run(zmm_b_paddd, &p);
        zv e;
        for (int i = 0; i < 16; i++) e.d[i] = (p.k >> i & 1) ? p.a.d[i] + p.b.d[0] : p.s.d[i];
        CHECK_LANES("vpaddd m32bcst {k1}", p.o.d, e.d, 16, 4);
        zop_run(zmm_b_pandq, &p);
        for (int i = 0; i < 8; i++) e.q[i] = p.a.q[i] & p.b.q[0];
        CHECK_LANES("vpandq m64bcst", p.o.q, e.q, 8, 8);
        zop_run(zmm_b_pcmpgtd, &p);
        uint64_t ek = 0;
        for (int i = 0; i < 16; i++) if ((p.k >> i & 1) && p.a.sd[i] > p.b.sd[0]) ek |= 1ull << i;
        CHECKF(p.ko == ek, "vpcmpgtd m32bcst {k1}: k=0x%llx want 0x%llx", (unsigned long long)p.ko, (unsigned long long)ek);
    }
}

TEST(zmm_bcst_float) {
    NEED_F();
    ZX_FP_DEFAULT();
    t_rng r; t_rng_seed(&r, 0xBC02);
    static zop p;
    for (int it = 0; it < 16; it++) {
        for (int i = 0; i < 16; i++) { p.a.f[i] = (float)((int)t_rng_range(&r, 2001) - 1000); p.s.f[i] = 99.0f; }
        p.b.f[0] = (float)((int)t_rng_range(&r, 64) - 32);
        p.k = t_rng_next(&r);
        zop_run(zmm_b_mulps_z, &p);
        zv e;
        for (int i = 0; i < 16; i++) e.f[i] = (p.k >> i & 1) ? p.a.f[i] * p.b.f[0] : 0.0f;
        CHECK_LANES("vmulps m32bcst {k1}{z}", p.o.d, e.d, 16, 4);
        for (int i = 0; i < 8; i++) { p.a.g[i] = (double)((int)t_rng_range(&r, 2001) - 1000); p.s.g[i] = (double)((int)t_rng_range(&r, 99) - 49); }
        p.b.g[0] = 3.0;
        zop_run(zmm_b_fmadd231pd, &p);
        for (int i = 0; i < 8; i++) e.g[i] = p.a.g[i] * 3.0 + p.s.g[i];   /* exact small integers */
        CHECK_LANES("vfmadd231pd m64bcst", p.o.q, e.q, 8, 8);
    }
}

ZOP_FN(zmm_r_addps_rn, "avx512f", "kmovw", "vaddps %{rn-sae%}, %%zmm2, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_r_addps_rd, "avx512f", "kmovw", "vaddps %{rd-sae%}, %%zmm2, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_r_addps_ru, "avx512f", "kmovw", "vaddps %{ru-sae%}, %%zmm2, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_r_addps_rz, "avx512f", "kmovw", "vaddps %{rz-sae%}, %%zmm2, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_r_addps, "avx512f", "kmovw", "vaddps %%zmm2, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_r_addpd_rz, "avx512f", "kmovw", "vaddpd %{rz-sae%}, %%zmm2, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_r_addpd_ru, "avx512f", "kmovw", "vaddpd %{ru-sae%}, %%zmm2, %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_r_cmpps_sae, "avx512f", "kmovw", "vcmpps $0, %{sae%}, %%zmm2, %%zmm1, %%k2", "kmovw")
ZOP_FN(zmm_r_cmpps_lt_os, "avx512f", "kmovw", "vcmpps $1, %%zmm2, %%zmm1, %%k2", "kmovw")

/* rounding vectors: a + b, expected results per mode RN, RD, RU, RZ */
static const struct { uint32_t a, b, e[4]; } g_rnd[] = {
    { 0x3F800000, 0x30800000, { 0x3F800000, 0x3F800000, 0x3F800001, 0x3F800000 } },  /*  1 + 2^-30          */
    { 0xBF800000, 0xB0800000, { 0xBF800000, 0xBF800001, 0xBF800000, 0xBF800000 } },  /* -1 - 2^-30          */
    { 0x3F800000, 0x34400000, { 0x3F800002, 0x3F800001, 0x3F800002, 0x3F800001 } },  /*  1 + 1.5 ulp        */
    { 0x40000000, 0x40000000, { 0x40800000, 0x40800000, 0x40800000, 0x40800000 } },  /*  2 + 2 exact        */
    { 0x3F800000, 0x33800000, { 0x3F800000, 0x3F800000, 0x3F800001, 0x3F800000 } },  /*  1 + 0.5 ulp (tie)  */
    { 0xBF800000, 0xB3800000, { 0xBF800000, 0xBF800001, 0xBF800000, 0xBF800000 } },  /* -1 - 0.5 ulp        */
    { 0x3F800000, 0xBF800000, { 0x00000000, 0x80000000, 0x00000000, 0x00000000 } },  /*  1 + -1: -0 in RD   */
    { 0x7F7FFFFF, 0x7F7FFFFF, { 0x7F800000, 0x7F7FFFFF, 0x7F800000, 0x7F7FFFFF } },  /*  overflow           */
    { 0xFF7FFFFF, 0xFF7FFFFF, { 0xFF800000, 0xFF800000, 0xFF7FFFFF, 0xFF7FFFFF } },  /* -overflow           */
    { 0x3F800001, 0x33800000, { 0x3F800002, 0x3F800001, 0x3F800002, 0x3F800001 } },  /* (1+ulp)+0.5ulp tie->even */
};

TEST(zmm_round_embedded_modes_ps) {
    NEED_F();
    ZX_FP_DEFAULT();
    static zop p;
    void (*fns[4])(zop *) = { zmm_r_addps_rn, zmm_r_addps_rd, zmm_r_addps_ru, zmm_r_addps_rz };
    static const char *mn[4] = { "rn-sae", "rd-sae", "ru-sae", "rz-sae" };
    static const unsigned rc_mx[4] = { 0x0000, 0x2000, 0x4000, 0x6000 };
    unsigned mx0 = t_get_mxcsr();
    const int nv = (int)T_COUNTOF(g_rnd);
    for (int i = 0; i < 16; i++) {
        p.a.d[i] = g_rnd[i % nv].a; p.b.d[i] = g_rnd[i % nv].b;
    }
    /* run every embedded mode under every MXCSR.RC: the embedded mode must win, MXCSR stays as is */
    for (int rc = 0; rc < 4; rc++) {
        for (int m = 0; m < 4; m++) {
            unsigned mxin = (mx0 & ~(0x6000u | MX_FLAGS)) | rc_mx[rc];
            t_set_mxcsr(mxin);
            zop_run(fns[m], &p);
            unsigned mx = t_get_mxcsr();
            t_set_mxcsr(mx0);
            CHECKF(mx == mxin, "MXCSR changed by vaddps {%s}: 0x%x -> 0x%x (SAE suppresses flags)", mn[m], mxin, mx);
            for (int i = 0; i < 16; i++) {
                uint32_t e = g_rnd[i % nv].e[m];
                if (i >= nv) continue;
                CHECKF(p.o.d[i] == e, "vaddps {%s} MXCSR.RC=%d: 0x%08x + 0x%08x = 0x%08x want 0x%08x", mn[m], rc,
                       p.a.d[i], p.b.d[i], p.o.d[i], e);
            }
        }
    }
    /* the same inputs without SAE report precision/overflow and use MXCSR.RC (RN) */
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    zop_run(zmm_r_addps, &p);
    unsigned mx = t_get_mxcsr();
    t_set_mxcsr(mx0);
    CHECKF(mx & 0x20, "PE not set by inexact vaddps without SAE (MXCSR 0x%x)", mx);
    CHECKF(mx & 0x08, "OE not set by overflowing vaddps without SAE (MXCSR 0x%x)", mx);
    for (int i = 0; i < nv; i++) CHECKF(p.o.d[i] == g_rnd[i].e[0], "vaddps (MXCSR RN) lane %d 0x%08x want 0x%08x", i, p.o.d[i], g_rnd[i].e[0]);
}

TEST(zmm_round_embedded_pd) {
    NEED_F();
    ZX_FP_DEFAULT();
    static zop p;
    for (int i = 0; i < 8; i++) { p.a.g[i] = 1.0; p.b.q[i] = 0x3C30000000000000ull; }   /* 1 + 2^-60 */
    p.b.g[1] = -ldexp(1.0, -60); p.a.g[1] = -1.0;
    unsigned mx0 = t_get_mxcsr();
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    zop_run(zmm_r_addpd_rz, &p);
    CHECKF(p.o.q[0] == 0x3FF0000000000000ull, "rz: 0x%llx", (unsigned long long)p.o.q[0]);
    CHECKF(p.o.q[1] == 0xBFF0000000000000ull, "rz neg: 0x%llx", (unsigned long long)p.o.q[1]);
    zop_run(zmm_r_addpd_ru, &p);
    unsigned mx = t_get_mxcsr();
    t_set_mxcsr(mx0);
    CHECKF(p.o.q[0] == 0x3FF0000000000001ull, "ru: 0x%llx", (unsigned long long)p.o.q[0]);
    CHECKF(p.o.q[1] == 0xBFF0000000000000ull, "ru neg: 0x%llx", (unsigned long long)p.o.q[1]);
    CHECKF((mx & MX_FLAGS) == 0, "MXCSR flags 0x%x after {sae} ops", mx & MX_FLAGS);
}

TEST(zmm_round_sae_compare) {
    NEED_F();
    ZX_FP_DEFAULT();
    /* VCMPPS with {sae}: a signalling predicate on QNaN must not set IE; without SAE LT_OS sets IE */
    static zop p;
    for (int i = 0; i < 16; i++) { p.a.f[i] = (float)i; p.b.f[i] = 4.0f; }
    p.a.d[2] = 0x7FC00000u;   /* QNaN */
    p.a.d[5] = 0x7F800001u;   /* SNaN */
    unsigned mx0 = t_get_mxcsr();
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    zop_run(zmm_r_cmpps_sae, &p);
    unsigned mx = t_get_mxcsr();
    t_set_mxcsr(mx0 & ~MX_FLAGS);
    CHECKF((mx & MX_FLAGS) == 0, "vcmpps {sae} with SNaN set MXCSR flags 0x%x", mx & MX_FLAGS);
    CHECK_EQ(p.ko, 1u << 4);
    zop_run(zmm_r_cmpps_lt_os, &p);
    mx = t_get_mxcsr();
    t_set_mxcsr(mx0);
    CHECKF(mx & 1, "vcmpps LT_OS with QNaN did not set IE (MXCSR 0x%x)", mx);
    CHECK_EQ(p.ko, 0x0Bu);   /* 0,1,3 < 4 ; lane 2 NaN false */
}

/* =============================================================================================
 * 7. VPTERNLOG: all 256 truth tables
 * ============================================================================================= */
#define ZTL(n) ZOP_FN(zmm_tl_##n, "avx512f", "kmovw", "vpternlogd $" #n ", %%zmm2, %%zmm1, %%zmm0", "kmovw")
#define ZTL16(h) ZTL(0x##h##0) ZTL(0x##h##1) ZTL(0x##h##2) ZTL(0x##h##3) ZTL(0x##h##4) ZTL(0x##h##5) ZTL(0x##h##6) \
    ZTL(0x##h##7) ZTL(0x##h##8) ZTL(0x##h##9) ZTL(0x##h##A) ZTL(0x##h##B) ZTL(0x##h##C) ZTL(0x##h##D) ZTL(0x##h##E) ZTL(0x##h##F)
ZTL16(0) ZTL16(1) ZTL16(2) ZTL16(3) ZTL16(4) ZTL16(5) ZTL16(6) ZTL16(7)
ZTL16(8) ZTL16(9) ZTL16(A) ZTL16(B) ZTL16(C) ZTL16(D) ZTL16(E) ZTL16(F)
#define ZTLP16(h) zmm_tl_0x##h##0, zmm_tl_0x##h##1, zmm_tl_0x##h##2, zmm_tl_0x##h##3, zmm_tl_0x##h##4, zmm_tl_0x##h##5, \
    zmm_tl_0x##h##6, zmm_tl_0x##h##7, zmm_tl_0x##h##8, zmm_tl_0x##h##9, zmm_tl_0x##h##A, zmm_tl_0x##h##B, \
    zmm_tl_0x##h##C, zmm_tl_0x##h##D, zmm_tl_0x##h##E, zmm_tl_0x##h##F,
static void (*const g_tl[256])(zop *) = {
    ZTLP16(0) ZTLP16(1) ZTLP16(2) ZTLP16(3) ZTLP16(4) ZTLP16(5) ZTLP16(6) ZTLP16(7)
    ZTLP16(8) ZTLP16(9) ZTLP16(A) ZTLP16(B) ZTLP16(C) ZTLP16(D) ZTLP16(E) ZTLP16(F)
};
static uint64_t zx_ternlog(uint64_t A, uint64_t B, uint64_t C, unsigned imm) {
    uint64_t r = 0;
    for (unsigned idx = 0; idx < 8; idx++) {
        if (!(imm >> idx & 1)) continue;
        r |= ((idx & 4) ? A : ~A) & ((idx & 2) ? B : ~B) & ((idx & 1) ? C : ~C);
    }
    return r;
}
TEST(zmm_ternlog_all_256_truth_tables) {
    NEED_F();
    /* VPTERNLOGD zmm0(A), zmm1(B), zmm2(C), imm: bit j of result = imm[(A_j<<2)|(B_j<<1)|C_j] */
    t_rng r; t_rng_seed(&r, 0x7E57);
    static zop p;
    zx_rand(&p.s, &r); zx_rand(&p.a, &r); zx_rand(&p.b, &r);
    /* make sure every (A,B,C) combination occurs in qword 0 */
    p.s.q[0] = 0xF0F0F0F0F0F0F0F0ull; p.a.q[0] = 0xCCCCCCCCCCCCCCCCull; p.b.q[0] = 0xAAAAAAAAAAAAAAAAull;
    for (unsigned imm = 0; imm < 256; imm++) {
        zop_run(g_tl[imm], &p);
        zv e;
        for (int i = 0; i < 8; i++) e.q[i] = zx_ternlog(p.s.q[i], p.a.q[i], p.b.q[i], imm);
        char lbl[48]; snprintf(lbl, sizeof lbl, "vpternlogd imm=0x%02x", imm);
        CHECK_LANES(lbl, p.o.q, e.q, 8, 8);
    }
}
ZOP_FN(zmm_tl_q_merge, "avx512f", "kmovw", "vpternlogq $0xCA, %%zmm2, %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_tl_q_zero, "avx512f", "kmovw", "vpternlogq $0x96, %%zmm2, %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_tl_d_bcst, "avx512f", "kmovw", "vpternlogd $0xE8, 128(%0)%{1to16%}, %%zmm1, %%zmm0", "kmovw")
TEST(zmm_ternlog_masked_and_broadcast) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x7E58);
    static zop p;
    for (int it = 0; it < 16; it++) {
        zx_rand(&p.s, &r); zx_rand(&p.a, &r); zx_rand(&p.b, &r); p.k = t_rng_next(&r);
        zv e;
        zop_run(zmm_tl_q_merge, &p);
        for (int i = 0; i < 8; i++) e.q[i] = (p.k >> i & 1) ? zx_ternlog(p.s.q[i], p.a.q[i], p.b.q[i], 0xCA) : p.s.q[i];
        CHECK_LANES("vpternlogq 0xCA {k1} (merge keeps A)", p.o.q, e.q, 8, 8);
        zop_run(zmm_tl_q_zero, &p);
        for (int i = 0; i < 8; i++) e.q[i] = (p.k >> i & 1) ? (p.s.q[i] ^ p.a.q[i] ^ p.b.q[i]) : 0;
        CHECK_LANES("vpternlogq 0x96 {k1}{z}", p.o.q, e.q, 8, 8);
        zop_run(zmm_tl_d_bcst, &p);
        for (int i = 0; i < 16; i++) e.d[i] = (uint32_t)zx_ternlog(p.s.d[i], p.a.d[i], p.b.d[0], 0xE8);
        CHECK_LANES("vpternlogd 0xE8 (majority) m32bcst", p.o.d, e.d, 16, 4);
    }
}

/* =============================================================================================
 * 8. compares into opmask registers
 * ============================================================================================= */
#define ZPC(n) ZOP_FN(zmm_pcmpd_##n, "avx512f", "kmovw", "vpcmpd $" #n ", %%zmm2, %%zmm1, %%k2%{%%k1%}", "kmovw") \
               ZOP_FN(zmm_pcmpud_##n, "avx512f", "kmovw", "vpcmpud $" #n ", %%zmm2, %%zmm1, %%k2%{%%k1%}", "kmovw") \
               ZOP_FN(zmm_pcmpq_##n, "avx512f", "kmovw", "vpcmpq $" #n ", %%zmm2, %%zmm1, %%k2%{%%k1%}", "kmovw")
ZPC(0) ZPC(1) ZPC(2) ZPC(3) ZPC(4) ZPC(5) ZPC(6) ZPC(7)
static void (*const g_pcmpd[8])(zop *) = { zmm_pcmpd_0, zmm_pcmpd_1, zmm_pcmpd_2, zmm_pcmpd_3, zmm_pcmpd_4, zmm_pcmpd_5, zmm_pcmpd_6, zmm_pcmpd_7 };
static void (*const g_pcmpud[8])(zop *) = { zmm_pcmpud_0, zmm_pcmpud_1, zmm_pcmpud_2, zmm_pcmpud_3, zmm_pcmpud_4, zmm_pcmpud_5, zmm_pcmpud_6, zmm_pcmpud_7 };
static void (*const g_pcmpq[8])(zop *) = { zmm_pcmpq_0, zmm_pcmpq_1, zmm_pcmpq_2, zmm_pcmpq_3, zmm_pcmpq_4, zmm_pcmpq_5, zmm_pcmpq_6, zmm_pcmpq_7 };
static int zx_icmp(int pred, int lt, int eq) {
    switch (pred) {
    case 0: return eq; case 1: return lt; case 2: return lt || eq; case 3: return 0;
    case 4: return !eq; case 5: return !lt; case 6: return !lt && !eq; default: return 1;
    }
}
TEST(zmm_cmp_integer_predicates) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0xC0C0);
    static zop p;
    static const char *pn[8] = { "EQ", "LT", "LE", "FALSE", "NE", "NLT", "NLE", "TRUE" };
    for (int it = 0; it < 6; it++) {
        zx_rand(&p.a, &r); zx_rand(&p.b, &r);
        for (int i = 0; i < 16; i++) if (i % 3 == 0) p.b.d[i] = p.a.d[i];          /* equal lanes */
        p.a.d[1] = 0x80000000u; p.b.d[1] = 0x7FFFFFFFu;                            /* signed vs unsigned order */
        p.k = it == 0 ? 0xFFFF : t_rng_next(&r);
        for (int pr = 0; pr < 8; pr++) {
            uint64_t es = 0, eu = 0, eq = 0;
            for (int i = 0; i < 16; i++) {
                if (!(p.k >> i & 1)) continue;
                if (zx_icmp(pr, p.a.sd[i] < p.b.sd[i], p.a.d[i] == p.b.d[i])) es |= 1ull << i;
                if (zx_icmp(pr, p.a.d[i] < p.b.d[i], p.a.d[i] == p.b.d[i])) eu |= 1ull << i;
            }
            for (int i = 0; i < 8; i++) {
                if (!(p.k >> i & 1)) continue;
                if (zx_icmp(pr, p.a.sq[i] < p.b.sq[i], p.a.q[i] == p.b.q[i])) eq |= 1ull << i;
            }
            zop_run(g_pcmpd[pr], &p);
            CHECKF(p.ko == es, "vpcmpd %s {k1=0x%llx}: 0x%llx want 0x%llx", pn[pr], (unsigned long long)p.k,
                   (unsigned long long)p.ko, (unsigned long long)es);
            zop_run(g_pcmpud[pr], &p);
            CHECKF(p.ko == eu, "vpcmpud %s: 0x%llx want 0x%llx", pn[pr], (unsigned long long)p.ko, (unsigned long long)eu);
            zop_run(g_pcmpq[pr], &p);
            CHECKF(p.ko == eq, "vpcmpq %s: 0x%llx want 0x%llx", pn[pr], (unsigned long long)p.ko, (unsigned long long)eq);
        }
    }
}

#define ZFC(n) ZOP_FN(zmm_cmpps_##n, "avx512f", "kmovw", "vcmpps $" #n ", %%zmm2, %%zmm1, %%k2%{%%k1%}", "kmovw")
ZFC(0) ZFC(1) ZFC(2) ZFC(3) ZFC(4) ZFC(5) ZFC(6) ZFC(7) ZFC(8) ZFC(9) ZFC(10) ZFC(11) ZFC(12) ZFC(13) ZFC(14) ZFC(15)
ZFC(16) ZFC(17) ZFC(18) ZFC(19) ZFC(20) ZFC(21) ZFC(22) ZFC(23) ZFC(24) ZFC(25) ZFC(26) ZFC(27) ZFC(28) ZFC(29) ZFC(30) ZFC(31)
static void (*const g_cmpps[32])(zop *) = {
    zmm_cmpps_0, zmm_cmpps_1, zmm_cmpps_2, zmm_cmpps_3, zmm_cmpps_4, zmm_cmpps_5, zmm_cmpps_6, zmm_cmpps_7,
    zmm_cmpps_8, zmm_cmpps_9, zmm_cmpps_10, zmm_cmpps_11, zmm_cmpps_12, zmm_cmpps_13, zmm_cmpps_14, zmm_cmpps_15,
    zmm_cmpps_16, zmm_cmpps_17, zmm_cmpps_18, zmm_cmpps_19, zmm_cmpps_20, zmm_cmpps_21, zmm_cmpps_22, zmm_cmpps_23,
    zmm_cmpps_24, zmm_cmpps_25, zmm_cmpps_26, zmm_cmpps_27, zmm_cmpps_28, zmm_cmpps_29, zmm_cmpps_30, zmm_cmpps_31,
};
/* SDM VCMPPS predicate table: relation of imm[3:0]; imm[4] flips signalling (QNaN raises IE) only */
static int zx_fcmp(int pred, float a, float b) {
    int un = isnan(a) || isnan(b), lt = !un && a < b, eq = !un && a == b, gt = !un && a > b;
    switch (pred & 15) {
    case 0: return eq; case 1: return lt; case 2: return lt || eq; case 3: return un;
    case 4: return un || !eq; case 5: return un || !lt; case 6: return un || gt || (!lt && !eq); case 7: return !un;
    case 8: return un || eq; case 9: return un || lt; case 10: return un || lt || eq; case 11: return 0;
    case 12: return !un && !eq; case 13: return !un && !lt; case 14: return gt; default: return 1;
    }
}
static const uint8_t g_fcmp_sig[16] = { 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0 };
TEST(zmm_cmp_float_32_predicates) {
    NEED_F();
    ZX_FP_DEFAULT();
    static zop p;
    static const float av[16] = { 1, 2, 2, -0.0f, 0, 1, INFINITY, -INFINITY, 5, -5, 3, 1e30f, -1e-30f, 7, 8, 0 };
    static const float bv[16] = { 2, 1, 2, 0.0f, 0, 0, INFINITY, 1, -5, -5, 4, 1e30f, 1e-30f, 8, 7, -0.0f };
    for (int i = 0; i < 16; i++) { p.a.f[i] = av[i]; p.b.f[i] = bv[i]; }
    p.a.d[4] = 0x7FC00000u;   /* QNaN vs 0 */
    p.b.d[5] = 0xFFC00001u;   /* 1 vs QNaN */
    unsigned mx0 = t_get_mxcsr();
    for (int pass = 0; pass < 2; pass++) {
        p.k = pass ? 0xFFCF : 0xFFFF;   /* pass 1: the NaN lanes 4,5 are masked off */
        for (int pr = 0; pr < 32; pr++) {
            uint64_t ek = 0;
            for (int i = 0; i < 16; i++)
                if ((p.k >> i & 1) && zx_fcmp(pr, p.a.f[i], p.b.f[i])) ek |= 1ull << i;
            t_set_mxcsr(mx0 & ~MX_FLAGS);
            zop_run(g_cmpps[pr], &p);
            unsigned mx = t_get_mxcsr();
            t_set_mxcsr(mx0);
            CHECKF(p.ko == ek, "vcmpps imm=%d k1=0x%llx: 0x%llx want 0x%llx", pr, (unsigned long long)p.k,
                   (unsigned long long)p.ko, (unsigned long long)ek);
            int sig = g_fcmp_sig[pr & 15] ^ (pr >> 4);
            int want_ie = pass == 0 && sig;
            CHECKF(!!(mx & 1) == want_ie, "vcmpps imm=%d %s: IE=%d want %d (QNaN %s)", pr, sig ? "signalling" : "quiet",
                   mx & 1, want_ie, pass ? "lanes masked off" : "active");
        }
    }
}

ZOP_FN(zmm_ptestmd, "avx512f", "kmovw", "vptestmd %%zmm2, %%zmm1, %%k2%{%%k1%}", "kmovw")
ZOP_FN(zmm_ptestnmq, "avx512f", "kmovw", "vptestnmq %%zmm2, %%zmm1, %%k2%{%%k1%}", "kmovw")
ZOP_FN(zmm_ptestmb_bw, "avx512f,avx512bw", "kmovq", "vptestmb %%zmm2, %%zmm1, %%k2%{%%k1%}", "kmovq")
TEST(zmm_cmp_vptestm) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x7357);
    static zop p;
    for (int it = 0; it < 16; it++) {
        zx_rand(&p.a, &r); zx_rand(&p.b, &r);
        for (int i = 0; i < 16; i += 3) p.b.d[i] = ~p.a.d[i];
        p.b.q[1] = ~p.a.q[1];
        p.k = t_rng_next(&r);
        uint64_t e = 0;
        for (int i = 0; i < 16; i++) if ((p.k >> i & 1) && (p.a.d[i] & p.b.d[i])) e |= 1ull << i;
        zop_run(zmm_ptestmd, &p);
        CHECKF(p.ko == e, "vptestmd 0x%llx want 0x%llx", (unsigned long long)p.ko, (unsigned long long)e);
        e = 0;
        for (int i = 0; i < 8; i++) if ((p.k >> i & 1) && !(p.a.q[i] & p.b.q[i])) e |= 1ull << i;
        zop_run(zmm_ptestnmq, &p);
        CHECKF(p.ko == e, "vptestnmq 0x%llx want 0x%llx", (unsigned long long)p.ko, (unsigned long long)e);
        if (cpu_has(F_AVX512BW)) {
            e = 0;
            for (int i = 0; i < 64; i++) if ((p.k >> i & 1) && (p.a.b[i] & p.b.b[i])) e |= 1ull << i;
            zop_run(zmm_ptestmb_bw, &p);
            CHECKF(p.ko == e, "vptestmb 0x%llx want 0x%llx", (unsigned long long)p.ko, (unsigned long long)e);
        }
    }
}

/* =============================================================================================
 * 9. conflict detection (AVX512CD)
 * ============================================================================================= */
ZOP_FN(zmm_cd_conflictd, "avx512f,avx512cd", "kmovw", "vpconflictd %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_cd_conflictq_z, "avx512f,avx512cd", "kmovw", "vpconflictq %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_cd_lzcntd, "avx512f,avx512cd", "kmovw", "vplzcntd %%zmm1, %%zmm0", "kmovw")
ZOP_FN(zmm_cd_lzcntq, "avx512f,avx512cd", "kmovw", "vplzcntq %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_cd_bcstmw2d, "avx512f,avx512cd", "kmovw", "vpbroadcastmw2d %%k1, %%zmm0", "kmovw")
ZOP_FN(zmm_cd_bcstmb2q, "avx512f,avx512cd", "kmovw", "vpbroadcastmb2q %%k1, %%zmm0", "kmovw")

TEST(zmm_cd_vpconflict) {
    NEED_X(F_AVX512CD);
    t_rng r; t_rng_seed(&r, 0xCD01);
    static zop p;
    for (int it = 0; it < 24; it++) {
        zx_rand(&p.s, &r);
        for (int i = 0; i < 16; i++) p.a.d[i] = (uint32_t)t_rng_range(&r, it < 4 ? 1 + (uint64_t)it : 6) * 0x01010101u;
        p.k = it == 0 ? 0xFFFF : t_rng_next(&r);
        zop_run(zmm_cd_conflictd, &p);
        zv e;
        for (int i = 0; i < 16; i++) {
            uint32_t m = 0;
            for (int j = 0; j < i; j++) if (p.a.d[j] == p.a.d[i]) m |= 1u << j;
            e.d[i] = (p.k >> i & 1) ? m : p.s.d[i];
        }
        CHECK_LANES("vpconflictd {k1}", p.o.d, e.d, 16, 4);
        for (int i = 0; i < 8; i++) p.a.q[i] = t_rng_range(&r, 3) << 40;
        zop_run(zmm_cd_conflictq_z, &p);
        for (int i = 0; i < 8; i++) {
            uint64_t m = 0;
            for (int j = 0; j < i; j++) if (p.a.q[j] == p.a.q[i]) m |= 1ull << j;
            e.q[i] = (p.k >> i & 1) ? m : 0;
        }
        CHECK_LANES("vpconflictq {k1}{z}", p.o.q, e.q, 8, 8);
    }
}
static int zx_lz32(uint32_t x) { int n = 0; if (!x) return 32; while (!(x & 0x80000000u)) { x <<= 1; n++; } return n; }
static int zx_lz64(uint64_t x) { int n = 0; if (!x) return 64; while (!(x >> 63)) { x <<= 1; n++; } return n; }
TEST(zmm_cd_vplzcnt_and_broadcastm) {
    NEED_X(F_AVX512CD);
    t_rng r; t_rng_seed(&r, 0xCD02);
    static zop p;
    for (int it = 0; it < 16; it++) {
        zx_rand(&p.s, &r);
        for (int i = 0; i < 16; i++) p.a.d[i] = (uint32_t)(t_rng_next(&r) >> t_rng_range(&r, 40));
        p.a.d[0] = 0; p.a.d[1] = 1; p.a.d[2] = 0x80000000u;
        p.k = t_rng_next(&r);
        zv e;
        zop_run(zmm_cd_lzcntd, &p);
        for (int i = 0; i < 16; i++) e.d[i] = (uint32_t)zx_lz32(p.a.d[i]);
        CHECK_LANES("vplzcntd", p.o.d, e.d, 16, 4);
        zop_run(zmm_cd_lzcntq, &p);
        for (int i = 0; i < 8; i++) e.q[i] = (p.k >> i & 1) ? (uint64_t)zx_lz64(p.a.q[i]) : p.s.q[i];
        CHECK_LANES("vplzcntq {k1}", p.o.q, e.q, 8, 8);
        zop_run(zmm_cd_bcstmw2d, &p);
        for (int i = 0; i < 16; i++) e.d[i] = (uint32_t)(p.k & 0xFFFF);
        CHECK_LANES("vpbroadcastmw2d", p.o.d, e.d, 16, 4);
        zop_run(zmm_cd_bcstmb2q, &p);
        for (int i = 0; i < 8; i++) e.q[i] = p.k & 0xFF;
        CHECK_LANES("vpbroadcastmb2q", p.o.q, e.q, 8, 8);
    }
}

/* =============================================================================================
 * 10. compress / expand
 * ============================================================================================= */
ZOP_FN(zmm_x_compressd, "avx512f", "kmovw", "vpcompressd %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_x_compressd_z, "avx512f", "kmovw", "vpcompressd %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_x_compressq, "avx512f", "kmovw", "vpcompressq %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_x_compresspd_z, "avx512f", "kmovw", "vcompresspd %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_x_expandd, "avx512f", "kmovw", "vpexpandd %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_x_expandd_z, "avx512f", "kmovw", "vpexpandd %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_x_expandps, "avx512f", "kmovw", "vexpandps %%zmm1, %%zmm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_x_expandq_z, "avx512f", "kmovw", "vpexpandq %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_x_compressb_vbmi2, "avx512f,avx512vbmi2", "kmovq", "vpcompressb %%zmm1, %%zmm0%{%%k1%}%{z%}", "kmovq")
ZOP_FN(zmm_x_expandw_vbmi2, "avx512f,avx512vbmi2", "kmovd", "vpexpandw %%zmm1, %%zmm0%{%%k1%}", "kmovq")

static void zx_compress_ref(const zv *s, const zv *a, uint64_t k, int w, int zero, zv *e) {
    int lanes = 64 / w, j = 0;
    *e = *s;
    for (int i = 0; i < lanes; i++) if (k >> i & 1) zx_lane_put(e, j++, w, zx_lane_get(a, i, w));
    for (; j < lanes; j++) if (zero) zx_lane_put(e, j, w, 0);
}
static void zx_expand_ref(const zv *s, const zv *a, uint64_t k, int w, int zero, zv *e) {
    int lanes = 64 / w, j = 0;
    for (int i = 0; i < lanes; i++) {
        if (k >> i & 1) zx_lane_put(e, i, w, zx_lane_get(a, j++, w));
        else zx_lane_put(e, i, w, zero ? 0 : zx_lane_get(s, i, w));
    }
}
static void zx_cx_case(void (*fn)(zop *), int w, int zero, int expand, const char *what, uint32_t seed) {
    t_rng r; t_rng_seed(&r, seed);
    static zop p;
    for (int it = 0; it < 20; it++) {
        zx_rand(&p.s, &r); zx_rand(&p.a, &r);
        p.k = it == 0 ? 0 : it == 1 ? ~0ull : it == 2 ? 0x8001 : t_rng_next(&r);
        zop_run(fn, &p);
        zv e;
        if (expand) zx_expand_ref(&p.s, &p.a, p.k, w, zero, &e); else zx_compress_ref(&p.s, &p.a, p.k, w, zero, &e);
        char lbl[96]; snprintf(lbl, sizeof lbl, "%s k=0x%llx", what, (unsigned long long)p.k);
        CHECK_LANES(lbl, p.o.b, e.b, 64 / w, w);
    }
}
TEST(zmm_compress_reg) {
    NEED_F();
    zx_cx_case(zmm_x_compressd, 4, 0, 0, "vpcompressd {k1} (merge keeps the tail)", 0xC501);
    zx_cx_case(zmm_x_compressd_z, 4, 1, 0, "vpcompressd {k1}{z}", 0xC502);
    zx_cx_case(zmm_x_compressq, 8, 0, 0, "vpcompressq {k1}", 0xC503);
    zx_cx_case(zmm_x_compresspd_z, 8, 1, 0, "vcompresspd {k1}{z}", 0xC504);
}
TEST(zmm_expand_reg) {
    NEED_F();
    zx_cx_case(zmm_x_expandd, 4, 0, 1, "vpexpandd {k1}", 0xC505);
    zx_cx_case(zmm_x_expandd_z, 4, 1, 1, "vpexpandd {k1}{z}", 0xC506);
    zx_cx_case(zmm_x_expandps, 4, 0, 1, "vexpandps {k1}", 0xC507);
    zx_cx_case(zmm_x_expandq_z, 8, 1, 1, "vpexpandq {k1}{z}", 0xC508);
}
TEST(zmm_compress_expand_bytes_vbmi2) {
    NEED_X(F_AVX512VBMI2);
    zx_cx_case(zmm_x_compressb_vbmi2, 1, 1, 0, "vpcompressb {k1}{z}", 0xC509);
    zx_cx_case(zmm_x_expandw_vbmi2, 2, 0, 1, "vpexpandw {k1}", 0xC50A);
}

/* memory forms */
T_TARGET("avx512f") T_NOINLINE static void zmm_compressd_store(void *dst, const zv *a, uint32_t k) {
    __asm__ volatile("vmovdqu64 (%1), %%zmm1\n\tkmovw %2, %%k1\n\tvpcompressd %%zmm1, (%0)%{%%k1%}"
                     :: "r"(dst), "r"(a), "r"(k) : "xmm1", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_expandd_load(zv *out, const void *src, const zv *pre, uint32_t k) {
    __asm__ volatile("vmovdqu64 (%2), %%zmm0\n\tkmovw %3, %%k1\n\tvpexpandd (%1), %%zmm0%{%%k1%}\n\tvmovdqu64 %%zmm0, (%0)"
                     :: "r"(out), "r"(src), "r"(pre), "r"(k) : "xmm0", "k1", "memory");
}
TEST(zmm_compress_expand_memory) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0xC50B);
    for (int it = 0; it < 24; it++) {
        zv a, pre, out, e; uint32_t buf[40], ref[40];
        zx_rand(&a, &r); zx_rand(&pre, &r);
        uint32_t k = it == 0 ? 0 : it == 1 ? 0xFFFF : (uint32_t)t_rng_u32(&r) & 0xFFFF;
        for (int i = 0; i < 40; i++) buf[i] = ref[i] = 0xC0DE0000u + (uint32_t)i;
        zmm_compressd_store(buf + 8, &a, k);
        int j = 8;
        for (int i = 0; i < 16; i++) if (k >> i & 1) ref[j++] = a.d[i];
        char lbl[64]; snprintf(lbl, sizeof lbl, "vpcompressd m512 k=0x%04x writes popcnt elements only", k);
        CHECK_LANES(lbl, buf, ref, 40, 4);
        zmm_expandd_load(&out, buf + 8, &pre, k);
        zv src; memcpy(src.d, buf + 8, 64);
        zx_expand_ref(&pre, &src, k, 4, 0, &e);
        snprintf(lbl, sizeof lbl, "vpexpandd m512 k=0x%04x", k);
        CHECK_LANES(lbl, out.d, e.d, 16, 4);
    }
}

/* =============================================================================================
 * 11. gather / scatter
 * ============================================================================================= */
T_TARGET("avx512f") T_NOINLINE static void zmm_gatherdd(zv *out, const zv *pre, const zv *idx, const void *base, uint64_t *k) {
    __asm__ volatile("vmovdqu64 (%1), %%zmm0\n\tvmovdqu64 (%2), %%zmm1\n\tkmovw (%4), %%k1\n\t"
                     "vpgatherdd (%3,%%zmm1,4), %%zmm0%{%%k1%}\n\tvmovdqu64 %%zmm0, (%0)\n\tkmovw %%k1, (%4)"
                     :: "r"(out), "r"(pre), "r"(idx), "r"(base), "r"(k) : "xmm0", "xmm1", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_gatherqq(zv *out, const zv *pre, const zv *idx, const void *base, uint64_t *k) {
    __asm__ volatile("vmovdqu64 (%1), %%zmm0\n\tvmovdqu64 (%2), %%zmm1\n\tkmovw (%4), %%k1\n\t"
                     "vpgatherqq (%3,%%zmm1,8), %%zmm0%{%%k1%}\n\tvmovdqu64 %%zmm0, (%0)\n\tkmovw %%k1, (%4)"
                     :: "r"(out), "r"(pre), "r"(idx), "r"(base), "r"(k) : "xmm0", "xmm1", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_gatherqd(zv *out, const zv *pre, const zv *idx, const void *base, uint64_t *k) {
    __asm__ volatile("vmovdqu64 (%1), %%zmm0\n\tvmovdqu64 (%2), %%zmm1\n\tkmovw (%4), %%k1\n\t"
                     "vpgatherqd (%3,%%zmm1,4), %%ymm0%{%%k1%}\n\tvmovdqu64 %%zmm0, (%0)\n\tkmovw %%k1, (%4)"
                     :: "r"(out), "r"(pre), "r"(idx), "r"(base), "r"(k) : "xmm0", "xmm1", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_scatterdd(const zv *data, const zv *idx, void *base, uint64_t *k) {
    __asm__ volatile("vmovdqu64 (%0), %%zmm0\n\tvmovdqu64 (%1), %%zmm1\n\tkmovw (%3), %%k1\n\t"
                     "vpscatterdd %%zmm0, (%2,%%zmm1,4)%{%%k1%}\n\tkmovw %%k1, (%3)"
                     :: "r"(data), "r"(idx), "r"(base), "r"(k) : "xmm0", "xmm1", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_scatterqq(const zv *data, const zv *idx, void *base, uint64_t *k) {
    __asm__ volatile("vmovdqu64 (%0), %%zmm0\n\tvmovdqu64 (%1), %%zmm1\n\tkmovw (%3), %%k1\n\t"
                     "vpscatterqq %%zmm0, (%2,%%zmm1,8)%{%%k1%}\n\tkmovw %%k1, (%3)"
                     :: "r"(data), "r"(idx), "r"(base), "r"(k) : "xmm0", "xmm1", "k1", "memory");
}

static uint32_t g_tab32[512];
static uint64_t g_tab64[256];
TEST(zmm_gather_dword_qword) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x6A01);
    for (int i = 0; i < 512; i++) g_tab32[i] = 0x6A000000u + (uint32_t)i * 7;
    for (int i = 0; i < 256; i++) g_tab64[i] = 0x6A6A000000000000ull + (uint64_t)i * 13;
    const uint32_t *b32 = g_tab32 + 256;   /* base in the middle: negative indices are valid */
    const uint64_t *b64 = g_tab64 + 128;
    for (int it = 0; it < 24; it++) {
        zv pre, idx, out, e; zx_rand(&pre, &r);
        for (int i = 0; i < 16; i++) idx.sd[i] = (int32_t)t_rng_range(&r, 512) - 256;
        uint64_t k = it == 0 ? 0xFFFF : it == 1 ? 0 : t_rng_next(&r), k0 = k;
        zmm_gatherdd(&out, &pre, &idx, b32, &k);
        for (int i = 0; i < 16; i++) e.d[i] = (k0 >> i & 1) ? b32[idx.sd[i]] : pre.d[i];
        CHECK_LANES("vpgatherdd {k1} (masked lanes keep dest)", out.d, e.d, 16, 4);
        CHECKF((k & 0xFFFF) == 0, "vpgatherdd: mask 0x%llx not cleared on completion", (unsigned long long)(k & 0xFFFF));
        for (int i = 0; i < 8; i++) idx.sq[i] = (int64_t)t_rng_range(&r, 256) - 128;
        k = k0;
        zmm_gatherqq(&out, &pre, &idx, b64, &k);
        for (int i = 0; i < 8; i++) e.q[i] = (k0 >> i & 1) ? b64[idx.sq[i]] : pre.q[i];
        CHECK_LANES("vpgatherqq {k1}", out.q, e.q, 8, 8);
        CHECKF((k & 0xFF) == 0, "vpgatherqq: mask 0x%llx not cleared", (unsigned long long)(k & 0xFF));
        for (int i = 0; i < 8; i++) idx.sq[i] = (int64_t)t_rng_range(&r, 512) - 256;
        k = k0;
        zmm_gatherqd(&out, &pre, &idx, b32, &k);
        memset(&e, 0, sizeof e);
        for (int i = 0; i < 8; i++) e.d[i] = (k0 >> i & 1) ? b32[idx.sq[i]] : pre.d[i];
        CHECK_LANES("vpgatherqd {k1} (ymm dest, bits 511:256 := 0)", out.d, e.d, 16, 4);
        CHECKF((k & 0xFF) == 0, "vpgatherqd: mask 0x%llx not cleared", (unsigned long long)(k & 0xFF));
    }
}
TEST(zmm_scatter_dword_qword) {
    NEED_F();
    /* SDM VPSCATTERDD: writes to overlapping locations are ordered LSB->MSB lane, so the highest active
     * lane wins; the mask is cleared on completion */
    t_rng r; t_rng_seed(&r, 0x6A02);
    for (int it = 0; it < 24; it++) {
        zv data, idx; zx_rand(&data, &r);
        uint32_t ref32[512]; uint64_t ref64[256];
        for (int i = 0; i < 512; i++) g_tab32[i] = ref32[i] = 0x5C000000u + (uint32_t)i;
        for (int i = 0; i < 256; i++) g_tab64[i] = ref64[i] = 0x5C5C000000000000ull + (uint64_t)i;
        int range = it < 8 ? 8 : 512;   /* small range forces duplicate indices */
        for (int i = 0; i < 16; i++) idx.sd[i] = (int32_t)t_rng_range(&r, (uint64_t)range) - range / 2;
        uint64_t k = it == 0 ? 0xFFFF : t_rng_next(&r), k0 = k;
        zmm_scatterdd(&data, &idx, g_tab32 + 256, &k);
        for (int i = 0; i < 16; i++) if (k0 >> i & 1) ref32[256 + idx.sd[i]] = data.d[i];
        CHECK_LANES("vpscatterdd memory (duplicates: highest lane wins)", g_tab32, ref32, 512, 4);
        CHECKF((k & 0xFFFF) == 0, "vpscatterdd: mask 0x%llx not cleared", (unsigned long long)(k & 0xFFFF));
        range = it < 8 ? 4 : 256;
        for (int i = 0; i < 8; i++) idx.sq[i] = (int64_t)t_rng_range(&r, (uint64_t)range) - range / 2;
        k = k0;
        zmm_scatterqq(&data, &idx, g_tab64 + 128, &k);
        for (int i = 0; i < 8; i++) if (k0 >> i & 1) ref64[128 + idx.sq[i]] = data.q[i];
        CHECK_LANES("vpscatterqq memory", g_tab64, ref64, 256, 8);
        CHECKF((k & 0xFF) == 0, "vpscatterqq: mask 0x%llx not cleared", (unsigned long long)(k & 0xFF));
    }
}

/* =============================================================================================
 * 12. permutes, integer / FP / conversion semantics new in AVX-512 (intrinsics, runtime inputs)
 * ============================================================================================= */
T_TARGET("avx512f") T_NOINLINE static void zmm_perm_f(const zv *a, const zv *b, const zv *idx, zv out[5]) {
    __m512i A = _mm512_load_si512(a), B = _mm512_load_si512(b), I = _mm512_load_si512(idx);
    _mm512_store_si512(&out[0], _mm512_permutexvar_epi32(I, A));
    _mm512_store_si512(&out[1], _mm512_permutexvar_epi64(I, A));
    _mm512_store_si512(&out[2], _mm512_permutex2var_epi32(A, I, B));
    _mm512_store_si512(&out[3], _mm512_alignr_epi32(A, B, 3));
    _mm512_store_si512(&out[4], _mm512_alignr_epi64(A, B, 5));
}
TEST(zmm_perm_vpermd_vpermt2d_valign) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x9E01);
    for (int it = 0; it < 16; it++) {
        zv a, b, idx, o[5], e; zx_rand(&a, &r); zx_rand(&b, &r); zx_rand(&idx, &r);
        zmm_perm_f(&a, &b, &idx, o);
        for (int i = 0; i < 16; i++) e.d[i] = a.d[idx.d[i] & 15];
        CHECK_LANES("vpermd (index bits 3:0)", o[0].d, e.d, 16, 4);
        for (int i = 0; i < 8; i++) e.q[i] = a.q[idx.q[i] & 7];
        CHECK_LANES("vpermq (index bits 2:0)", o[1].q, e.q, 8, 8);
        for (int i = 0; i < 16; i++) e.d[i] = (idx.d[i] & 16) ? b.d[idx.d[i] & 15] : a.d[idx.d[i] & 15];
        CHECK_LANES("vpermt2d (bit 4 selects table)", o[2].d, e.d, 16, 4);
        for (int i = 0; i < 16; i++) e.d[i] = i + 3 < 16 ? b.d[i + 3] : a.d[i + 3 - 16];
        CHECK_LANES("valignd 3", o[3].d, e.d, 16, 4);
        for (int i = 0; i < 8; i++) e.q[i] = i + 5 < 8 ? b.q[i + 5] : a.q[i + 5 - 8];
        CHECK_LANES("valignq 5", o[4].q, e.q, 8, 8);
    }
}
T_TARGET("avx512f,avx512vbmi") T_NOINLINE static void zmm_perm_vbmi(const zv *a, const zv *b, const zv *idx, zv out[2]) {
    __m512i A = _mm512_load_si512(a), B = _mm512_load_si512(b), I = _mm512_load_si512(idx);
    _mm512_store_si512(&out[0], _mm512_permutexvar_epi8(I, A));
    _mm512_store_si512(&out[1], _mm512_permutex2var_epi8(A, I, B));
}
TEST(zmm_perm_vpermb_vbmi) {
    NEED_X(F_AVX512VBMI);
    t_rng r; t_rng_seed(&r, 0x9E02);
    for (int it = 0; it < 16; it++) {
        zv a, b, idx, o[2], e; zx_rand(&a, &r); zx_rand(&b, &r); zx_rand(&idx, &r);
        zmm_perm_vbmi(&a, &b, &idx, o);
        for (int i = 0; i < 64; i++) e.b[i] = a.b[idx.b[i] & 63];
        CHECK_LANES("vpermb", o[0].b, e.b, 64, 1);
        for (int i = 0; i < 64; i++) e.b[i] = (idx.b[i] & 64) ? b.b[idx.b[i] & 63] : a.b[idx.b[i] & 63];
        CHECK_LANES("vpermt2b", o[1].b, e.b, 64, 1);
    }
}
T_TARGET("avx512f,avx512bw") T_NOINLINE static void zmm_perm_bw(const zv *a, const zv *b, const zv *idx, zv out[2]) {
    __m512i A = _mm512_load_si512(a), B = _mm512_load_si512(b), I = _mm512_load_si512(idx);
    _mm512_store_si512(&out[0], _mm512_permutexvar_epi16(I, A));
    _mm512_store_si512(&out[1], _mm512_permutex2var_epi16(A, I, B));
}
TEST(zmm_perm_vpermw_bw) {
    NEED_X(F_AVX512BW);
    t_rng r; t_rng_seed(&r, 0x9E03);
    for (int it = 0; it < 16; it++) {
        zv a, b, idx, o[2], e; zx_rand(&a, &r); zx_rand(&b, &r); zx_rand(&idx, &r);
        zmm_perm_bw(&a, &b, &idx, o);
        for (int i = 0; i < 32; i++) e.w[i] = a.w[idx.w[i] & 31];
        CHECK_LANES("vpermw", o[0].w, e.w, 32, 2);
        for (int i = 0; i < 32; i++) e.w[i] = (idx.w[i] & 32) ? b.w[idx.w[i] & 31] : a.w[idx.w[i] & 31];
        CHECK_LANES("vpermt2w", o[1].w, e.w, 32, 2);
    }
}

T_TARGET("avx512f") T_NOINLINE static void zmm_int_f(const zv *a, const zv *b, zv out[8]) {
    __m512i A = _mm512_load_si512(a), B = _mm512_load_si512(b);
    _mm512_store_si512(&out[0], _mm512_abs_epi64(A));
    _mm512_store_si512(&out[1], _mm512_max_epi64(A, B));
    _mm512_store_si512(&out[2], _mm512_min_epu64(A, B));
    _mm512_store_si512(&out[3], _mm512_srav_epi64(A, B));
    _mm512_store_si512(&out[4], _mm512_sllv_epi32(A, B));
    _mm512_store_si512(&out[5], _mm512_rolv_epi32(A, B));
    _mm512_store_si512(&out[6], _mm512_ror_epi64(A, 13));
    _mm512_store_si512(&out[7], _mm512_mul_epu32(A, B));
}
static uint32_t zx_rol32(uint32_t x, unsigned n) { n &= 31; return n ? (x << n) | (x >> (32 - n)) : x; }
static uint64_t zx_ror64(uint64_t x, unsigned n) { n &= 63; return n ? (x >> n) | (x << (64 - n)) : x; }
TEST(zmm_int_avx512f_semantics) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x1A01);
    for (int it = 0; it < 24; it++) {
        zv a, b, o[8], e; zx_rand(&a, &r); zx_rand(&b, &r);
        a.q[0] = 0x8000000000000000ull; a.q[1] = 0; a.q[2] = ~0ull;
        for (int i = 0; i < 8; i++) if (i & 1) b.q[i] = t_rng_range(&r, 80);   /* shift counts incl. >= 64 */
        for (int i = 0; i < 16; i++) if (i % 3 == 0) b.d[i] = (uint32_t)t_rng_range(&r, 40);
        zmm_int_f(&a, &b, o);
        for (int i = 0; i < 8; i++) e.q[i] = a.sq[i] < 0 ? (uint64_t)0 - a.q[i] : a.q[i];
        CHECK_LANES("vpabsq (abs(INT64_MIN) = INT64_MIN)", o[0].q, e.q, 8, 8);
        for (int i = 0; i < 8; i++) e.q[i] = a.sq[i] > b.sq[i] ? a.q[i] : b.q[i];
        CHECK_LANES("vpmaxsq", o[1].q, e.q, 8, 8);
        for (int i = 0; i < 8; i++) e.q[i] = a.q[i] < b.q[i] ? a.q[i] : b.q[i];
        CHECK_LANES("vpminuq", o[2].q, e.q, 8, 8);
        for (int i = 0; i < 8; i++) e.q[i] = b.q[i] >= 64 ? (a.sq[i] < 0 ? ~0ull : 0) : (uint64_t)(a.sq[i] >> b.q[i]);
        CHECK_LANES("vpsravq (count >= 64 fills with sign)", o[3].q, e.q, 8, 8);
        for (int i = 0; i < 16; i++) e.d[i] = b.d[i] >= 32 ? 0 : a.d[i] << b.d[i];
        CHECK_LANES("vpsllvd (count >= 32 -> 0)", o[4].d, e.d, 16, 4);
        for (int i = 0; i < 16; i++) e.d[i] = zx_rol32(a.d[i], b.d[i]);
        CHECK_LANES("vprolvd (count mod 32)", o[5].d, e.d, 16, 4);
        for (int i = 0; i < 8; i++) e.q[i] = zx_ror64(a.q[i], 13);
        CHECK_LANES("vprorq 13", o[6].q, e.q, 8, 8);
        for (int i = 0; i < 8; i++) e.q[i] = (uint64_t)a.d[2 * i] * b.d[2 * i];
        CHECK_LANES("vpmuludq", o[7].q, e.q, 8, 8);
    }
}

T_TARGET("avx512f") T_NOINLINE static void zmm_narrow_f(const zv *a, zv out[4]) {
    __m512i A = _mm512_load_si512(a);
    memset(out, 0, 4 * sizeof(zv));
    _mm256_storeu_si256((__m256i *)&out[0], _mm512_cvtepi64_epi32(A));
    _mm256_storeu_si256((__m256i *)&out[1], _mm512_cvtsepi64_epi32(A));
    _mm256_storeu_si256((__m256i *)&out[2], _mm512_cvtusepi64_epi32(A));
    _mm_storeu_si128((__m128i *)&out[3], _mm512_cvtsepi32_epi8(A));
}
TEST(zmm_int_vpmov_narrowing) {
    NEED_F();
    t_rng r; t_rng_seed(&r, 0x1A02);
    for (int it = 0; it < 16; it++) {
        zv a, o[4], e; zx_rand(&a, &r);
        a.q[0] = 0x100000000ull; a.sq[1] = -((int64_t)1 << 40); a.q[2] = 5; a.sq[3] = -5; a.q[4] = 0x7FFFFFFF;
        a.q[5] = 0xFFFFFFFFFFFFFFFFull;
        zmm_narrow_f(&a, o);
        memset(&e, 0, sizeof e);
        for (int i = 0; i < 8; i++) e.d[i] = (uint32_t)a.q[i];
        CHECK_LANES("vpmovqd (truncate)", o[0].d, e.d, 16, 4);
        memset(&e, 0, sizeof e);
        for (int i = 0; i < 8; i++) e.sd[i] = a.sq[i] > INT32_MAX ? INT32_MAX : a.sq[i] < INT32_MIN ? INT32_MIN : (int32_t)a.sq[i];
        CHECK_LANES("vpmovsqd (signed saturate)", o[1].d, e.d, 16, 4);
        memset(&e, 0, sizeof e);
        for (int i = 0; i < 8; i++) e.d[i] = a.q[i] > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)a.q[i];
        CHECK_LANES("vpmovusqd (unsigned saturate)", o[2].d, e.d, 16, 4);
        memset(&e, 0, sizeof e);
        for (int i = 0; i < 16; i++) e.sb[i] = a.sd[i] > 127 ? 127 : a.sd[i] < -128 ? -128 : (int8_t)a.sd[i];
        CHECK_LANES("vpmovsdb (signed saturate)", o[3].b, e.b, 64, 1);
    }
}

T_TARGET("avx512f,avx512dq") T_NOINLINE static void zmm_dq_f(const zv *a, const zv *b, zv out[4], uint64_t km[2]) {
    __m512i A = _mm512_load_si512(a), B = _mm512_load_si512(b);
    _mm512_store_si512(&out[0], _mm512_mullo_epi64(A, B));
    _mm512_store_pd(out[1].g, _mm512_cvtepi64_pd(A));
    _mm512_store_si512(&out[2], _mm512_cvttpd_epi64(_mm512_load_pd(b->g)));
    km[0] = _mm512_movepi32_mask(A);
    _mm512_store_si512(&out[3], _mm512_movm_epi32((__mmask16)b->w[0]));
    km[1] = _mm512_movepi64_mask(B);
}
TEST(zmm_dq_mullq_cvt_movmask) {
    NEED_X(F_AVX512DQ);
    ZX_FP_DEFAULT();
    t_rng r; t_rng_seed(&r, 0x1A03);
    for (int it = 0; it < 16; it++) {
        zv a, b, o[4], e; uint64_t km[2];
        zx_rand(&a, &r); zx_rand(&b, &r);
        a.q[0] = (1ull << 53) + 1; a.sq[1] = -((int64_t)(1ull << 53) + 3); a.q[2] = 0x8000000000000000ull; a.q[3] = 12345;
        b.g[0] = 1e19; b.g[1] = -1e19; b.q[2] = 0x7FF8000000000000ull; b.g[3] = -3.9; b.g[4] = 4611686018427387904.0;
        b.g[5] = 0.99; b.g[6] = -0.0; b.g[7] = 9007199254740993.0;
        zmm_dq_f(&a, &b, o, km);
        for (int i = 0; i < 8; i++) e.q[i] = a.q[i] * b.q[i];
        CHECK_LANES("vpmullq (low 64 bits)", o[0].q, e.q, 8, 8);
        e.q[0] = zx_d2u(9007199254740992.0);          /* 2^53+1 -> 2^53 (tie to even) */
        e.q[1] = zx_d2u(-9007199254740996.0);         /* -(2^53+3) -> -(2^53+4) */
        e.q[2] = zx_d2u(-9223372036854775808.0);
        e.q[3] = zx_d2u(12345.0);
        CHECK_LANES("vcvtqq2pd rounding", o[1].q, e.q, 4, 8);
        static const uint64_t ec[8] = { 0x8000000000000000ull, 0x8000000000000000ull, 0x8000000000000000ull,
                                        (uint64_t)-3ll, 4611686018427387904ull, 0, 0, 9007199254740992ull };
        CHECK_LANES("vcvttpd2qq (out of range/NaN -> integer indefinite)", o[2].q, ec, 8, 8);
        uint64_t m = 0; for (int i = 0; i < 16; i++) if (a.sd[i] < 0) m |= 1ull << i;
        CHECK_EQ(km[0], m);
        for (int i = 0; i < 16; i++) e.d[i] = (b.w[0] >> i & 1) ? 0xFFFFFFFFu : 0;
        CHECK_LANES("vpmovm2d", o[3].d, e.d, 16, 4);
        m = 0; for (int i = 0; i < 8; i++) if (b.sq[i] < 0) m |= 1ull << i;
        CHECK_EQ(km[1], m);
    }
}

T_TARGET("avx512f") T_NOINLINE static void zmm_fp_f(const zv *a, const zv *b, const zv *c, const zv *x, const zv *y, zv out[8]) {
    __m512d A = _mm512_load_pd(a->g), B = _mm512_load_pd(b->g), C = _mm512_load_pd(c->g);
    __m512 X = _mm512_load_ps(x->f), Y = _mm512_load_ps(y->f);
    _mm512_store_pd(out[0].g, _mm512_fmadd_pd(A, B, C));
    _mm512_store_ps(out[1].f, _mm512_roundscale_ps(X, 0x08));
    _mm512_store_ps(out[2].f, _mm512_roundscale_ps(X, 0x09));
    _mm512_store_ps(out[3].f, _mm512_roundscale_ps(X, 0x0A));
    _mm512_store_ps(out[4].f, _mm512_roundscale_ps(X, 0x0B));
    _mm512_store_ps(out[5].f, _mm512_roundscale_ps(X, 0x18));
    _mm512_store_ps(out[6].f, _mm512_getexp_ps(Y));
    _mm512_store_ps(out[7].f, _mm512_scalef_ps(X, Y));
}
static float zx_rne(float x) {
    float f = floorf(x), d = x - f, r;
    if (d > 0.5f) r = f + 1.0f; else if (d < 0.5f) r = f; else r = fmodf(f, 2.0f) != 0.0f ? f + 1.0f : f;
    return r == 0.0f ? copysignf(0.0f, x) : r;
}
static float zx_rmode(float x, int mode) {
    float r = mode == 0 ? zx_rne(x) : mode == 1 ? floorf(x) : mode == 2 ? ceilf(x) : truncf(x);
    return r == 0.0f ? copysignf(0.0f, x) : r;
}
TEST(zmm_fp_fma_rndscale_getexp_scalef) {
    NEED_F();
    ZX_FP_DEFAULT();
    t_rng r; t_rng_seed(&r, 0xF901);
    for (int it = 0; it < 12; it++) {
        zv a, b, c, x, y, o[8], e;
        for (int i = 0; i < 8; i++) {
            a.g[i] = (double)((int)t_rng_range(&r, 2001) - 1000); b.g[i] = (double)((int)t_rng_range(&r, 201) - 100);
            c.g[i] = (double)((int)t_rng_range(&r, 20001) - 10000);
        }
        a.g[0] = 1.0 + ldexp(1.0, -52); b.g[0] = 1.0 - ldexp(1.0, -52); c.g[0] = -1.0;   /* single rounding */
        for (int i = 0; i < 16; i++) {
            x.f[i] = (float)((int)t_rng_range(&r, 1601) - 800) / 8.0f;
            y.f[i] = (float)((int)t_rng_range(&r, 401) - 200) / 10.0f;
        }
        x.f[0] = 2.5f; x.f[1] = -2.5f; x.f[2] = -0.5f; x.f[3] = 0.5f; x.f[4] = -0.25f; x.f[5] = 1.75f;
        y.d[0] = 0; y.d[1] = 0x80000000u; y.d[2] = 0x7F800000u; y.d[3] = 0xFF800000u; y.f[4] = ldexpf(1.0f, -140);
        y.f[5] = 1.0f; y.f[6] = -3.0e10f; y.f[7] = 0.75f;
        zmm_fp_f(&a, &b, &c, &x, &y, o);
        for (int i = 0; i < 8; i++) e.g[i] = a.g[i] * b.g[i] + c.g[i];
        e.g[0] = -ldexp(1.0, -104);
        CHECK_LANES("vfmadd213pd (single rounding: (1+u)(1-u)-1 = -2^-104)", o[0].q, e.q, 8, 8);
        for (int m = 0; m < 4; m++) {
            for (int i = 0; i < 16; i++) e.f[i] = zx_rmode(x.f[i], m);
            static const char *nm[4] = { "vrndscaleps 0x08 (RNE)", "vrndscaleps 0x09 (floor)", "vrndscaleps 0x0A (ceil)",
                                         "vrndscaleps 0x0B (trunc)" };
            CHECK_LANES(nm[m], o[1 + m].d, e.d, 16, 4);
        }
        for (int i = 0; i < 16; i++) e.f[i] = zx_rne(x.f[i] * 2.0f) / 2.0f;
        for (int i = 0; i < 16; i++) if (e.f[i] == 0.0f) e.f[i] = copysignf(0.0f, x.f[i]);
        CHECK_LANES("vrndscaleps 0x18 (M=1, nearest half)", o[5].d, e.d, 16, 4);
        for (int i = 0; i < 16; i++) {
            float x = y.f[i];
            if (x == 0.0f) e.f[i] = -INFINITY;
            else if (isinf(x)) e.f[i] = INFINITY;
            else { int ex; frexpf(x, &ex); e.f[i] = (float)(ex - 1); }
        }
        CHECK_LANES("vgetexpps (incl. 0 -> -inf, inf -> +inf, denormal)", o[6].d, e.d, 16, 4);
        int ok = 1;
        for (int i = 8; i < 16; i++) {   /* lanes with finite moderate exponents */
            float ex = ldexpf(x.f[i], (int)floorf(y.f[i]));
            if (zx_f2u(o[7].f[i]) != zx_f2u(ex)) { ok = 0; CHECKF(0, "vscalefps lane %d: %g * 2^floor(%g) = %g want %g", i,
                                                                 (double)x.f[i], (double)y.f[i], (double)o[7].f[i], (double)ex); }
        }
        if (ok) CHECK(1);
    }
}

T_TARGET("avx512f") T_NOINLINE static void zmm_approx_f(const zv *a, zv out[2]) {
    __m512 X = _mm512_load_ps(a->f);
    _mm512_store_ps(out[0].f, _mm512_rcp14_ps(X));
    _mm512_store_ps(out[1].f, _mm512_rsqrt14_ps(X));
}
TEST(zmm_fp_rcp14_rsqrt14_accuracy) {
    NEED_F();
    ZX_FP_DEFAULT();
    /* SDM: relative error < 2^-14; implementation-specific values otherwise, so bounds, not bits */
    t_rng r; t_rng_seed(&r, 0xF902);
    double worst_rcp = 0, worst_rsq = 0;
    for (int it = 0; it < 64; it++) {
        zv a, o[2];
        for (int i = 0; i < 16; i++) a.f[i] = ldexpf(1.0f + (float)t_rng_range(&r, 1u << 23) / 8388608.0f, (int)t_rng_range(&r, 61) - 30);
        zmm_approx_f(&a, o);
        for (int i = 0; i < 16; i++) {
            double x = a.f[i], er = fabs((double)o[0].f[i] * x - 1.0), es = fabs((double)o[1].f[i] * sqrt(x) - 1.0);
            if (er > worst_rcp) worst_rcp = er;
            if (es > worst_rsq) worst_rsq = es;
        }
    }
    CHECKF(worst_rcp < ldexp(1.0, -14), "vrcp14ps max relative error %.3g >= 2^-14", worst_rcp);
    CHECKF(worst_rsq < ldexp(1.0, -14), "vrsqrt14ps max relative error %.3g >= 2^-14", worst_rsq);
    t_info("rcp14_max_rel_err", "%.3g", worst_rcp);
    t_info("rsqrt14_max_rel_err", "%.3g", worst_rsq);
}

T_TARGET("avx512f") T_NOINLINE static void zmm_cvtu_f(const zv *a, const zv *b, zv out[2]) {
    _mm512_store_si512(&out[0], _mm512_cvttps_epu32(_mm512_load_ps(a->f)));
    _mm512_store_ps(out[1].f, _mm512_cvtepu32_ps(_mm512_load_si512(b)));
}
TEST(zmm_cvt_unsigned_dword) {
    NEED_F();
    ZX_FP_DEFAULT();
    zv a, b, o[2];
    static const float af[16] = { 0.0f, 1.9f, 3e9f, 4294967040.0f, 4294967296.0f, -1.0f, 0, INFINITY,
                                  -0.0f, 0.99f, 2147483648.0f, 65535.5f, 7.0f, 1e20f, -INFINITY, 123456.0f };
    static const uint32_t ae[16] = { 0, 1, 3000000000u, 4294967040u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                                     0, 0, 2147483648u, 65535, 7, 0xFFFFFFFFu, 0xFFFFFFFFu, 123456 };
    memcpy(a.f, af, 64); a.d[6] = 0x7FC00000u;   /* NaN */
    static const uint32_t bd[16] = { 0xFFFFFFFFu, 0x80000001u, 16777217u, 16777219u, 0, 1, 0x80000000u, 16777216u,
                                     4294967040u, 4294967168u, 100, 3, 0x7FFFFFFFu, 33554435u, 255, 65536 };
    memcpy(b.d, bd, 64);
    zmm_cvtu_f(&a, &b, o);
    CHECK_LANES("vcvttps2udq (unrepresentable -> 0xFFFFFFFF)", o[0].d, ae, 16, 4);
    static const float be[16] = { 4294967296.0f, 2147483648.0f, 16777216.0f, 16777220.0f, 0, 1, 2147483648.0f, 16777216.0f,
                                  4294967040.0f, 4294967296.0f, 100, 3, 2147483648.0f, 33554436.0f, 255, 65536 };
    CHECK_LANES("vcvtudq2ps (round to nearest even)", o[1].d, be, 16, 4);
}

/* VFPCLASSPS classifier per the SDM pseudo-code */
static int zx_fpclass(uint32_t x, unsigned imm) {
    int neg = x >> 31, e1 = ((x >> 23) & 0xFF) == 0xFF, e0 = ((x >> 23) & 0xFF) == 0, m0 = (x & 0x7FFFFF) == 0, q = x >> 22 & 1;
    int r = 0;
    r |= (imm & 0x01) && e1 && q;
    r |= (imm & 0x02) && !neg && e0 && m0;
    r |= (imm & 0x04) && neg && e0 && m0;
    r |= (imm & 0x08) && !neg && e1 && m0;
    r |= (imm & 0x10) && neg && e1 && m0;
    r |= (imm & 0x20) && e0 && !m0;
    r |= (imm & 0x40) && neg && !e1 && !(e0 && m0);
    r |= (imm & 0x80) && e1 && !m0 && !q;
    return r;
}
#define ZFPC(n) T_TARGET("avx512f,avx512dq") T_NOINLINE static uint32_t zmm_fpclass_##n(const zv *a) { \
    return (uint32_t)_mm512_fpclass_ps_mask(_mm512_load_ps(a->f), n); }
ZFPC(0x01) ZFPC(0x02) ZFPC(0x04) ZFPC(0x08) ZFPC(0x10) ZFPC(0x20) ZFPC(0x40) ZFPC(0x80) ZFPC(0x99) ZFPC(0x66) ZFPC(0xFF)
TEST(zmm_dq_vfpclassps) {
    NEED_X(F_AVX512DQ);
    ZX_FP_DEFAULT();
    zv a;
    static const uint32_t v[16] = { 0x7FC00000u, 0xFFC00001u, 0x7F800001u, 0xFFA00000u, 0x00000000u, 0x80000000u,
                                    0x7F800000u, 0xFF800000u, 0x00000001u, 0x807FFFFFu, 0x3F800000u, 0xBF800000u,
                                    0xFF7FFFFFu, 0x00800000u, 0x80800000u, 0x7F7FFFFFu };
    memcpy(a.d, v, 64);
    static const unsigned imms[11] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x99, 0x66, 0xFF };
    uint32_t (*fns[11])(const zv *) = { zmm_fpclass_0x01, zmm_fpclass_0x02, zmm_fpclass_0x04, zmm_fpclass_0x08,
                                        zmm_fpclass_0x10, zmm_fpclass_0x20, zmm_fpclass_0x40, zmm_fpclass_0x80,
                                        zmm_fpclass_0x99, zmm_fpclass_0x66, zmm_fpclass_0xFF };
    for (int t = 0; t < 11; t++) {
        uint32_t e = 0;
        for (int i = 0; i < 16; i++) if (zx_fpclass(v[i], imms[t])) e |= 1u << i;
        uint32_t g = fns[t](&a);
        CHECKF(g == e, "vfpclassps imm=0x%02x: 0x%04x want 0x%04x", imms[t], g, e);
    }
}

T_TARGET("avx512f,avx512bw") T_NOINLINE static void zmm_bw_movm(const zv *a, uint64_t k, zv *o, uint64_t *km) {
    *km = _mm512_movepi8_mask(_mm512_load_si512(a));
    _mm512_store_si512(o, _mm512_movm_epi8(k));
}
TEST(zmm_bw_vpmovb2m_vpmovm2b) {
    NEED_X(F_AVX512BW);
    t_rng r; t_rng_seed(&r, 0xB201);
    for (int it = 0; it < 16; it++) {
        zv a, o, e; uint64_t km; zx_rand(&a, &r);
        uint64_t k = t_rng_next(&r);
        zmm_bw_movm(&a, k, &o, &km);
        uint64_t m = 0; for (int i = 0; i < 64; i++) if (a.sb[i] < 0) m |= 1ull << i;
        CHECK_EQ(km, m);
        for (int i = 0; i < 64; i++) e.b[i] = (k >> i & 1) ? 0xFF : 0;
        CHECK_LANES("vpmovm2b", o.b, e.b, 64, 1);
    }
}

/* ---- gated extensions ---- */
static void zx_mul64(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) {
    uint64_t al = a & 0xFFFFFFFFu, ah = a >> 32, bl = b & 0xFFFFFFFFu, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
    *lo = (ll & 0xFFFFFFFFu) | (mid << 32);
    *hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}
T_TARGET("avx512f,avx512ifma") T_NOINLINE static void zmm_ifma_f(const zv *acc, const zv *b, const zv *c, zv out[2]) {
    __m512i A = _mm512_load_si512(acc), B = _mm512_load_si512(b), C = _mm512_load_si512(c);
    _mm512_store_si512(&out[0], _mm512_madd52lo_epu64(A, B, C));
    _mm512_store_si512(&out[1], _mm512_madd52hi_epu64(A, B, C));
}
TEST(zmm_ext_ifma52) {
    NEED_X(F_AVX512IFMA);
    t_rng r; t_rng_seed(&r, 0xE101);
    const uint64_t m52 = (1ull << 52) - 1;
    for (int it = 0; it < 16; it++) {
        zv acc, b, c, o[2], el, eh; zx_rand(&acc, &r); zx_rand(&b, &r); zx_rand(&c, &r);
        b.q[0] = ~0ull; c.q[0] = ~0ull;   /* bits 63:52 of the sources are ignored */
        zmm_ifma_f(&acc, &b, &c, o);
        for (int i = 0; i < 8; i++) {
            uint64_t hi, lo; zx_mul64(b.q[i] & m52, c.q[i] & m52, &hi, &lo);
            el.q[i] = acc.q[i] + (lo & m52);
            eh.q[i] = acc.q[i] + ((hi << 12) | (lo >> 52));
        }
        CHECK_LANES("vpmadd52luq", o[0].q, el.q, 8, 8);
        CHECK_LANES("vpmadd52huq", o[1].q, eh.q, 8, 8);
    }
}
T_TARGET("avx512f,avx512vnni") T_NOINLINE static void zmm_vnni_f(const zv *s, const zv *a, const zv *b, zv out[3]) {
    __m512i S = _mm512_load_si512(s), A = _mm512_load_si512(a), B = _mm512_load_si512(b);
    _mm512_store_si512(&out[0], _mm512_dpbusd_epi32(S, A, B));
    _mm512_store_si512(&out[1], _mm512_dpbusds_epi32(S, A, B));
    _mm512_store_si512(&out[2], _mm512_dpwssd_epi32(S, A, B));
}
TEST(zmm_ext_vnni) {
    NEED_X(F_AVX512VNNI);
    t_rng r; t_rng_seed(&r, 0xE102);
    for (int it = 0; it < 16; it++) {
        zv s, a, b, o[3], e0, e1, e2; zx_rand(&s, &r); zx_rand(&a, &r); zx_rand(&b, &r);
        s.sd[0] = INT32_MAX - 5; memset(a.b, 0xFF, 4); memset(b.b, 0x7F, 4);       /* saturation case */
        s.sd[1] = INT32_MIN + 5; memset(a.b + 4, 0xFF, 4); memset(b.b + 4, 0x80, 4);
        zmm_vnni_f(&s, &a, &b, o);
        for (int i = 0; i < 16; i++) {
            int64_t sum = 0, sw = 0;
            for (int j = 0; j < 4; j++) sum += (int64_t)a.b[4 * i + j] * b.sb[4 * i + j];
            for (int j = 0; j < 2; j++) sw += (int64_t)a.sw[2 * i + j] * b.sw[2 * i + j];
            e0.d[i] = (uint32_t)((int64_t)s.sd[i] + sum);
            int64_t t = (int64_t)s.sd[i] + sum;
            e1.sd[i] = t > INT32_MAX ? INT32_MAX : t < INT32_MIN ? INT32_MIN : (int32_t)t;
            e2.d[i] = (uint32_t)((int64_t)s.sd[i] + sw);
        }
        CHECK_LANES("vpdpbusd", o[0].d, e0.d, 16, 4);
        CHECK_LANES("vpdpbusds (saturating)", o[1].d, e1.d, 16, 4);
        CHECK_LANES("vpdpwssd", o[2].d, e2.d, 16, 4);
    }
}
T_TARGET("avx512f,avx512vpopcntdq") T_NOINLINE static void zmm_popcnt_dq(const zv *a, zv out[2]) {
    __m512i A = _mm512_load_si512(a);
    _mm512_store_si512(&out[0], _mm512_popcnt_epi32(A));
    _mm512_store_si512(&out[1], _mm512_popcnt_epi64(A));
}
T_TARGET("avx512f,avx512bitalg,avx512bw") T_NOINLINE static void zmm_popcnt_bitalg(const zv *a, zv out[2]) {
    __m512i A = _mm512_load_si512(a);
    _mm512_store_si512(&out[0], _mm512_popcnt_epi8(A));
    _mm512_store_si512(&out[1], _mm512_popcnt_epi16(A));
}
TEST(zmm_ext_vpopcnt) {
    NEED_F();
    if (!cpu_has(F_AVX512VPOPCNTDQ) && !cpu_has(F_AVX512BITALG)) SKIP("neither avx512vpopcntdq nor avx512bitalg advertised");
    t_rng r; t_rng_seed(&r, 0xE103);
    for (int it = 0; it < 16; it++) {
        zv a, o[2], e; zx_rand(&a, &r);
        a.q[0] = 0; a.q[1] = ~0ull;
        if (cpu_has(F_AVX512VPOPCNTDQ)) {
            zmm_popcnt_dq(&a, o);
            for (int i = 0; i < 16; i++) e.d[i] = (uint32_t)zx_pop64(a.d[i]);
            CHECK_LANES("vpopcntd", o[0].d, e.d, 16, 4);
            for (int i = 0; i < 8; i++) e.q[i] = (uint64_t)zx_pop64(a.q[i]);
            CHECK_LANES("vpopcntq", o[1].q, e.q, 8, 8);
        }
        if (cpu_has(F_AVX512BITALG)) {
            zmm_popcnt_bitalg(&a, o);
            for (int i = 0; i < 64; i++) e.b[i] = (uint8_t)zx_pop64(a.b[i]);
            CHECK_LANES("vpopcntb", o[0].b, e.b, 64, 1);
            for (int i = 0; i < 32; i++) e.w[i] = (uint16_t)zx_pop64(a.w[i]);
            CHECK_LANES("vpopcntw", o[1].w, e.w, 32, 2);
        }
    }
}
T_TARGET("avx512f,avx512vbmi2") T_NOINLINE static void zmm_vbmi2_shift(const zv *a, const zv *b, zv out[3]) {
    __m512i A = _mm512_load_si512(a), B = _mm512_load_si512(b);
    _mm512_store_si512(&out[0], _mm512_shldi_epi32(A, B, 7));
    _mm512_store_si512(&out[1], _mm512_shrdi_epi64(A, B, 13));
    _mm512_store_si512(&out[2], _mm512_shldv_epi32(A, B, _mm512_load_si512(a)));
}
TEST(zmm_ext_vbmi2_double_shift) {
    NEED_X(F_AVX512VBMI2);
    t_rng r; t_rng_seed(&r, 0xE104);
    for (int it = 0; it < 16; it++) {
        zv a, b, o[3], e; zx_rand(&a, &r); zx_rand(&b, &r);
        zmm_vbmi2_shift(&a, &b, o);
        for (int i = 0; i < 16; i++) e.d[i] = (a.d[i] << 7) | (b.d[i] >> 25);
        CHECK_LANES("vpshldd 7", o[0].d, e.d, 16, 4);
        for (int i = 0; i < 8; i++) e.q[i] = (a.q[i] >> 13) | (b.q[i] << 51);
        CHECK_LANES("vpshrdq 13", o[1].q, e.q, 8, 8);
        /* VPSHLDVD: dest = upper32((dest:src) << (count & 31)), dest = a, src = b, count = a */
        for (int i = 0; i < 16; i++) {
            unsigned n = a.d[i] & 31;
            e.d[i] = n ? (a.d[i] << n) | (b.d[i] >> (32 - n)) : a.d[i];
        }
        CHECK_LANES("vpshldvd", o[2].d, e.d, 16, 4);
    }
}

/* ---- AVX512VL: EVEX.128/256 masking and zeroing of the upper bits ---- */
ZOP_FN(zmm_vl_paddd_ymm, "avx512f,avx512vl", "kmovw", "vpaddd %%ymm2, %%ymm1, %%ymm0%{%%k1%}", "kmovw")
ZOP_FN(zmm_vl_paddq_xmm_z, "avx512f,avx512vl", "kmovw", "vpaddq %%xmm2, %%xmm1, %%xmm0%{%%k1%}%{z%}", "kmovw")
ZOP_FN(zmm_vl_pcmpd_ymm, "avx512f,avx512vl", "kmovw", "vpcmpd $1, %%ymm2, %%ymm1, %%k2", "kmovw")
TEST(zmm_vl_masked_narrow_ops) {
    NEED_X(F_AVX512VL);
    t_rng r; t_rng_seed(&r, 0x7101);
    static zop p;
    for (int it = 0; it < 16; it++) {
        zx_rand(&p.s, &r); zx_rand(&p.a, &r); zx_rand(&p.b, &r); p.k = t_rng_next(&r);
        zv e; memset(&e, 0, sizeof e);
        zop_run(zmm_vl_paddd_ymm, &p);
        for (int i = 0; i < 8; i++) e.d[i] = (p.k >> i & 1) ? p.a.d[i] + p.b.d[i] : p.s.d[i];
        CHECK_LANES("EVEX.256 vpaddd {k1}: bits 511:256 := 0", p.o.d, e.d, 16, 4);
        memset(&e, 0, sizeof e);
        zop_run(zmm_vl_paddq_xmm_z, &p);
        for (int i = 0; i < 2; i++) e.q[i] = (p.k >> i & 1) ? p.a.q[i] + p.b.q[i] : 0;
        CHECK_LANES("EVEX.128 vpaddq {k1}{z}: bits 511:128 := 0", p.o.q, e.q, 8, 8);
        zop_run(zmm_vl_pcmpd_ymm, &p);
        uint64_t ek = 0; for (int i = 0; i < 8; i++) if (p.a.sd[i] < p.b.sd[i]) ek |= 1ull << i;
        CHECKF(p.ko == ek, "EVEX.256 vpcmpd LT: k=0x%llx want 0x%llx (bits 15:8 := 0)", (unsigned long long)p.ko, (unsigned long long)ek);
    }
}
#ifdef T_X64
T_TARGET("avx512f,avx512vl") T_NOINLINE static void zmm_vl_hi16_ops(zop *p) {
    __asm__ volatile("vmovdqu64 0(%0), %%zmm16\n\tvmovdqu64 64(%0), %%zmm17\n\tvmovdqu64 128(%0), %%zmm18\n\t"
                     "vmovdqu64 0(%0), %%zmm19\n\tvmovdqu64 0(%0), %%zmm20\n\t"
                     "vpaddd %%xmm18, %%xmm17, %%xmm16\n\t"     /* EVEX.128 on xmm16: upper zeroed */
                     "vpxord %%ymm17, %%ymm18, %%ymm19\n\t"     /* EVEX.256 on ymm19 */
                     "vmovdqa64 %%xmm17, %%xmm20\n\t"
                     "vmovdqu64 %%zmm16, 192(%0)\n\tvmovdqu64 %%zmm19, 256(%0)\n\tvmovdqu64 %%zmm20, 320(%0)\n\t"
                     :: "r"(p) : "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "memory");
}
#endif
TEST(zmm_vl_hi16_registers_x64) {
    NEED_X(F_AVX512VL); NEED_X64();
#ifdef T_X64
    static struct T_ALIGN(64) { zv s, a, b, o, o2, o3; } p;
    t_rng r; t_rng_seed(&r, 0x7102);
    for (int it = 0; it < 8; it++) {
        zx_rand(&p.s, &r); zx_rand(&p.a, &r); zx_rand(&p.b, &r);
        zmm_vl_hi16_ops((zop *)&p);
        zv e; memset(&e, 0, sizeof e);
        for (int i = 0; i < 4; i++) e.d[i] = p.a.d[i] + p.b.d[i];
        CHECK_LANES("xmm16 = xmm17 + xmm18 (zmm16 bits 511:128 := 0)", p.o.d, e.d, 16, 4);
        memset(&e, 0, sizeof e);
        for (int i = 0; i < 8; i++) e.d[i] = p.a.d[i] ^ p.b.d[i];
        CHECK_LANES("ymm19 = ymm18 ^ ymm17 (bits 511:256 := 0)", p.o2.d, e.d, 16, 4);
        memset(&e, 0, sizeof e);
        memcpy(e.b, p.a.b, 16);
        CHECK_LANES("vmovdqa64 xmm17 -> xmm20", p.o3.d, e.d, 16, 4);
    }
#endif
}

/* =============================================================================================
 * 13. memory fault suppression and deliberate faults (children)
 * ============================================================================================= */
T_TARGET("avx512f") T_NOINLINE static void zmm_ld32(zv *out, const void *p, uint32_t k, const zv *pre) {
    __asm__ volatile("vmovdqu64 (%3), %%zmm0\n\tkmovw %2, %%k1\n\tvmovdqu32 (%1), %%zmm0%{%%k1%}\n\tvmovdqu64 %%zmm0, (%0)"
                     :: "r"(out), "r"(p), "r"(k), "r"(pre) : "xmm0", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_ld32z(zv *out, const void *p, uint32_t k) {
    __asm__ volatile("kmovw %2, %%k1\n\tvmovdqu32 (%1), %%zmm0%{%%k1%}%{z%}\n\tvmovdqu64 %%zmm0, (%0)"
                     :: "r"(out), "r"(p), "r"(k) : "xmm0", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_st32(void *p, const zv *v, uint32_t k) {
    __asm__ volatile("vmovdqu64 (%1), %%zmm0\n\tkmovw %2, %%k1\n\tvmovdqu32 %%zmm0, (%0)%{%%k1%}"
                     :: "r"(p), "r"(v), "r"(k) : "xmm0", "k1", "memory");
}
T_TARGET("avx512f,avx512bw") T_NOINLINE static void zmm_ld8z_bw(zv *out, const void *p, const uint64_t *k) {
    __asm__ volatile("kmovq (%2), %%k1\n\tvmovdqu8 (%1), %%zmm0%{%%k1%}%{z%}\n\tvmovdqu64 %%zmm0, (%0)"
                     :: "r"(out), "r"(p), "r"(k) : "xmm0", "k1", "memory");
}
T_TARGET("avx512f,avx512bw") T_NOINLINE static void zmm_st8_bw(void *p, const zv *v, const uint64_t *k) {
    __asm__ volatile("vmovdqu64 (%1), %%zmm0\n\tkmovq (%2), %%k1\n\tvmovdqu8 %%zmm0, (%0)%{%%k1%}"
                     :: "r"(p), "r"(v), "r"(k) : "xmm0", "k1", "memory");
}
T_TARGET("avx512f") T_NOINLINE static void zmm_ld_aligned64(zv *out, const void *p) {
    __asm__ volatile("vmovdqa64 (%1), %%zmm0\n\tvmovdqu64 %%zmm0, (%0)" :: "r"(out), "r"(p) : "xmm0", "memory");
}

static int g_bad;
static void zx_cmp_print(const char *what, const void *got, const void *want, size_t n) {
    if (memcmp(got, want, n)) { g_bad++; printf("MISMATCH=%s\n", what); }
    else printf("OK=%s\n", what);
}
static uint8_t *zx_child_tail(size_t n, void **base) {
    uint8_t *p = t_guarded_alloc(n, base);
    if (!p) { printf("ALLOC_FAILED=1\n"); fflush(stdout); exit(4); }
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(0x40 + i);
    return p;
}
#define CHILD_NEED_F() do { if (!cpu_has(F_AVX512F)) { printf("NOTUSABLE=1\n"); return 3; } } while (0)

CHILD(zmm_fs_masked_load) {
    (void)argc; (void)argv; CHILD_NEED_F();
    void *base; uint8_t *p = zx_child_tail(64, &base);
    zv out, e, pre;
    memset(&pre, 0x5A, sizeof pre);
    /* 16 accessible bytes before the guard page: lanes 0-3 active */
    zmm_ld32z(&out, p + 48, 0x000F);
    memset(&e, 0, sizeof e); memcpy(e.b, p + 48, 16);
    zx_cmp_print("vmovdqu32{z} 4 lanes at page end", &out, &e, 64);
    zmm_ld32(&out, p + 48, 0x0005, &pre);
    e = pre; memcpy(e.b, p + 48, 4); memcpy(e.b + 8, p + 56, 4);
    zx_cmp_print("vmovdqu32{k} merge at page end", &out, &e, 64);
    /* 40 accessible bytes: lanes 0-9 */
    zmm_ld32z(&out, p + 24, 0x03FF);
    memset(&e, 0, sizeof e); memcpy(e.b, p + 24, 40);
    zx_cmp_print("vmovdqu32{z} 10 lanes straddling", &out, &e, 64);
    /* mask 0 on the guard page itself */
    zmm_ld32z(&out, p + 64, 0);
    memset(&e, 0, sizeof e);
    zx_cmp_print("vmovdqu32{z} k=0 on PAGE_NOACCESS", &out, &e, 64);
    zmm_ld32(&out, p + 64 + 4096 - 64, 0, &pre);
    zx_cmp_print("vmovdqu32{k} k=0 merge, PAGE_NOACCESS", &out, &pre, 64);
    printf("BAD=%d\n", g_bad); fflush(stdout);
    t_guarded_free(base);
    return g_bad ? 1 : 0;
}
CHILD(zmm_fs_masked_store) {
    (void)argc; (void)argv; CHILD_NEED_F();
    void *base; uint8_t *p = zx_child_tail(64, &base);
    zv v; uint8_t ref[64];
    for (int i = 0; i < 64; i++) v.b[i] = (uint8_t)(0xA0 + i);
    memcpy(ref, p, 64);
    zmm_st32(p + 48, &v, 0x000F);
    memcpy(ref + 48, v.b, 16);
    zx_cmp_print("vmovdqu32{k} store 4 lanes at page end", p, ref, 64);
    zmm_st32(p + 32, &v, 0x0009);
    memcpy(ref + 32, v.b, 4); memcpy(ref + 44, v.b + 12, 4);
    zx_cmp_print("vmovdqu32{k} store lanes 0,3 only", p, ref, 64);
    zmm_st32(p + 64, &v, 0);
    printf("OK=k=0 store on PAGE_NOACCESS\n");
    printf("BAD=%d\n", g_bad); fflush(stdout);
    t_guarded_free(base);
    return g_bad ? 1 : 0;
}
CHILD(zmm_fs_masked_bytes_bw) {
    (void)argc; (void)argv; CHILD_NEED_F();
    if (!cpu_has(F_AVX512BW)) { printf("NOTUSABLE=1\n"); return 3; }
    void *base; uint8_t *p = zx_child_tail(64, &base);
    zv out, e, v; uint64_t k = (1ull << 13) - 1;
    zmm_ld8z_bw(&out, p + 51, &k);
    memset(&e, 0, sizeof e); memcpy(e.b, p + 51, 13);
    zx_cmp_print("vmovdqu8{z} 13 bytes at page end", &out, &e, 64);
    for (int i = 0; i < 64; i++) v.b[i] = (uint8_t)(0xC0 + i);
    uint8_t ref[64]; memcpy(ref, p, 64); memcpy(ref + 51, v.b, 13);
    zmm_st8_bw(p + 51, &v, &k);
    zx_cmp_print("vmovdqu8{k} store 13 bytes at page end", p, ref, 64);
    k = 0x5555555555555555ull & ((1ull << 13) - 1);
    zmm_st8_bw(p + 51, &v, &k);
    printf("BAD=%d\n", g_bad); fflush(stdout);
    t_guarded_free(base);
    return g_bad ? 1 : 0;
}
CHILD(zmm_fs_gather_scatter) {
    (void)argc; (void)argv; CHILD_NEED_F();
    void *base; uint8_t *p = zx_child_tail(64, &base);
    uint32_t *d = (uint32_t *)p;     /* 16 valid dwords, then the guard page */
    zv pre, idx, out, e;
    memset(&pre, 0x77, sizeof pre);
    for (int i = 0; i < 16; i++) idx.sd[i] = (i & 1) ? 16 + 64 * i : 15 - i;   /* odd lanes -> guard page */
    uint64_t k = 0x5555;
    zmm_gatherdd(&out, &pre, &idx, d, &k);
    e = pre; for (int i = 0; i < 16; i += 2) e.d[i] = d[15 - i];
    zx_cmp_print("vpgatherdd masked lanes into PAGE_NOACCESS", &out, &e, 64);
    if (k & 0xFFFF) { g_bad++; printf("MISMATCH=gather mask not cleared 0x%llx\n", (unsigned long long)k); }
    zv data; for (int i = 0; i < 16; i++) data.d[i] = 0xD0D00000u + (uint32_t)i;
    uint32_t ref[16]; memcpy(ref, d, 64);
    for (int i = 0; i < 16; i += 2) ref[15 - i] = data.d[i];
    k = 0x5555;
    zmm_scatterdd(&data, &idx, d, &k);
    zx_cmp_print("vpscatterdd masked lanes into PAGE_NOACCESS", d, ref, 64);
    if (k & 0xFFFF) { g_bad++; printf("MISMATCH=scatter mask not cleared\n"); }
    printf("BAD=%d\n", g_bad); fflush(stdout);
    t_guarded_free(base);
    return g_bad ? 1 : 0;
}
CHILD(zmm_fs_compress_expand) {
    (void)argc; (void)argv; CHILD_NEED_F();
    void *base; uint8_t *p = zx_child_tail(64, &base);
    zv a, out, e, pre;
    for (int i = 0; i < 16; i++) a.d[i] = 0xE0E00000u + (uint32_t)i;
    memset(&pre, 0x33, sizeof pre);
    uint8_t ref[64]; memcpy(ref, p, 64);
    uint32_t k = 0x8421;   /* popcnt 4: exactly the 16 bytes before the guard page */
    zmm_compressd_store(p + 48, &a, k);
    uint32_t w[4] = { a.d[0], a.d[5], a.d[10], a.d[15] };
    memcpy(ref + 48, w, 16);
    zx_cmp_print("vpcompressd store 4 elements ending at the guard page", p, ref, 64);
    zmm_expandd_load(&out, p + 48, &pre, k);
    e = pre; e.d[0] = w[0]; e.d[5] = w[1]; e.d[10] = w[2]; e.d[15] = w[3];
    zx_cmp_print("vpexpandd load 4 elements ending at the guard page", &out, &e, 64);
    zmm_compressd_store(p + 64, &a, 0);
    printf("OK=compress k=0 on PAGE_NOACCESS\n");
    printf("BAD=%d\n", g_bad); fflush(stdout);
    t_guarded_free(base);
    return g_bad ? 1 : 0;
}
CHILD(zmm_ff_active_lane) {
    (void)argc; (void)argv; CHILD_NEED_F();
    void *base; uint8_t *p = zx_child_tail(64, &base);
    zv out;
    printf("BEFORE=1\n"); fflush(stdout);
    zmm_ld32z(&out, p + 48, 0x001F);   /* lane 4 is on the guard page */
    printf("AFTER=1\n");
    return 0;
}
CHILD(zmm_ff_gather_active_lane) {
    (void)argc; (void)argv; CHILD_NEED_F();
    void *base; uint8_t *p = zx_child_tail(64, &base);
    zv pre, idx, out; memset(&pre, 0, sizeof pre);
    for (int i = 0; i < 16; i++) idx.sd[i] = i;
    idx.sd[7] = 16;   /* active lane 7 reads the guard page */
    uint64_t k = 0x00FF;
    printf("BEFORE=1\n"); fflush(stdout);
    zmm_gatherdd(&out, &pre, &idx, p, &k);
    printf("AFTER=1\n");
    return 0;
}
CHILD(zmm_ff_misaligned_vmovdqa64) {
    (void)argc; (void)argv; CHILD_NEED_F();
    static uint8_t T_ALIGN(64) buf[256];
    zv out;
    printf("BEFORE=1\n"); fflush(stdout);
    zmm_ld_aligned64(&out, buf + 16);
    printf("AFTER=1\n");
    return 0;
}

static void zx_child_expect_ok(const char *name) {
    REQUIREF(t_child_run(name, "", 15000, &g_cr), "child did not start (error %lu)", (unsigned long)g_cr.spawn_error);
    CHECK(!g_cr.timed_out);
    char v[64];
    if (t_child_find(&g_cr, "NOTUSABLE=", v, sizeof v)) SKIP("feature not usable in the child");
    CHECKF(g_cr.exc_code == 0, "fault 0x%lx at 0x%llx: masked-off elements must not fault (SDM exception class E4/E12)",
           (unsigned long)g_cr.exc_code, g_cr.exc_addr);
    CHECKF(t_child_find(&g_cr, "BAD=", v, sizeof v) && atoi(v) == 0, "child reported mismatches:\n%s", g_cr.out);
    CHECK_EQ(g_cr.exit_code, 0);
    int oks = 0; for (const char *s = g_cr.out; (s = strstr(s, "OK=")) != NULL; s += 3) oks++;
    t_info("child_ok_lines", "%d", oks);
}
static void zx_child_expect_fault(const char *name, DWORD want, const char *what) {
    REQUIREF(t_child_run(name, "", 15000, &g_cr), "child did not start (error %lu)", (unsigned long)g_cr.spawn_error);
    CHECK(!g_cr.timed_out);
    char v[64];
    if (t_child_find(&g_cr, "NOTUSABLE=", v, sizeof v)) SKIP("feature not usable in the child");
    CHECKF(t_child_find(&g_cr, "BEFORE=", v, sizeof v), "child did not reach the probe");
    CHECKF(!t_child_find(&g_cr, "AFTER=", v, sizeof v), "%s completed without a fault", what);
    CHECKF(g_cr.exc_code == want, "%s: exception 0x%lx, want 0x%lx", what, (unsigned long)g_cr.exc_code, (unsigned long)want);
}

TEST(zmm_fault_suppression_masked_load) { NEED_F(); zx_child_expect_ok("zmm_fs_masked_load"); }
TEST(zmm_fault_suppression_masked_store) { NEED_F(); zx_child_expect_ok("zmm_fs_masked_store"); }
TEST(zmm_fault_suppression_bytes_bw) { NEED_X(F_AVX512BW); zx_child_expect_ok("zmm_fs_masked_bytes_bw"); }
TEST(zmm_fault_suppression_gather_scatter) { NEED_F(); zx_child_expect_ok("zmm_fs_gather_scatter"); }
TEST(zmm_fault_suppression_compress_expand) { NEED_F(); zx_child_expect_ok("zmm_fs_compress_expand"); }
TEST(zmm_fault_active_lane_faults) {
    NEED_F();
    zx_child_expect_fault("zmm_ff_active_lane", STATUS_ACCESS_VIOLATION, "masked load with an active lane on PAGE_NOACCESS");
}
TEST(zmm_fault_gather_active_lane_faults) {
    NEED_F();
    zx_child_expect_fault("zmm_ff_gather_active_lane", STATUS_ACCESS_VIOLATION, "gather with an active lane on PAGE_NOACCESS");
}
TEST(zmm_fault_misaligned_vmovdqa64) {
    NEED_F();
    REQUIREF(t_child_run("zmm_ff_misaligned_vmovdqa64", "", 15000, &g_cr), "child did not start");
    CHECK(!g_cr.timed_out);
    char v[16];
    CHECKF(!t_child_find(&g_cr, "AFTER=", v, sizeof v), "VMOVDQA64 from a 16-byte aligned address did not #GP");
    CHECKF(g_cr.exc_code != 0, "no exception (exit 0x%lx)", (unsigned long)g_cr.exit_code);
    t_info("misaligned_exception", "0x%lx", (unsigned long)g_cr.exc_code);
    if (g_cr.exc_code != STATUS_ACCESS_VIOLATION && g_cr.exc_code != STATUS_DATATYPE_MISALIGNMENT)
        t_note("misaligned VMOVDQA64 #GP reported as 0x%lx", (unsigned long)g_cr.exc_code);
}

/* =============================================================================================
 * 14. preservation: callbacks (ABI), threads (preemption), suspend/context, exceptions
 * ============================================================================================= */
static void zx_cb_case(void (*fn)(void *), void *arg, const char *label) {
    NEED_F();
    static t_vregs in, out;
    t_vregs_pattern(&in, T_VK_ZMM, 0x600);
    REQUIRE(t_vregs_call(T_VK_ZMM, &in, &out, fn, arg) == 0);
#ifdef T_X64
    char what[96];
    for (int i = 6; i < 16; i++) {   /* the only vector state the x64 ABI guarantees across a call */
        snprintf(what, sizeof what, "xmm%d[127:0] across %s (nonvolatile)", i, label);
        CHECK_REG(what, out.v[i], in.v[i], 16);
    }
#endif
    int changed = 0, kchanged = 0;
    for (int i = 0; i < T_NZMM; i++) if (memcmp(out.v[i], in.v[i], 64)) changed++;
    for (int i = 0; i < 8; i++) if (out.k[i] != in.k[i]) kchanged++;
    if (changed || kchanged)
        t_note("%s changed %d volatile ZMM registers and %d opmask registers (allowed by the ABI)", label, changed, kchanged);
    else
        t_note("%s preserved every ZMM and opmask register", label);
    t_info("volatile_changed", "%d zmm, %d k", changed, kchanged);
}
static void zx_cb_sleep0(void *a) { (void)a; Sleep(0); }
static void zx_cb_sleep1(void *a) { (void)a; Sleep(1); }
static void zx_cb_switch(void *a) { (void)a; for (int i = 0; i < 16; i++) SwitchToThread(); }
static void zx_cb_wait_signaled(void *a) {
    (void)a; HANDLE e = CreateEventA(NULL, TRUE, TRUE, NULL);
    if (e) { WaitForSingleObject(e, 1000); CloseHandle(e); }
}
static void zx_cb_wait_timeout(void *a) {
    (void)a; HANDLE e = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (e) { WaitForSingleObject(e, 15); CloseHandle(e); }
}
static DWORD WINAPI zx_mutex_holder(void *p) {
    HANDLE *h = p;
    if (WaitForSingleObject(h[0], 2000) != WAIT_OBJECT_0) return 1;
    SetEvent(h[1]); Sleep(20); ReleaseMutex(h[0]);
    return 0;
}
static void zx_cb_mutex(void *a) {
    (void)a;
    HANDLE h[2] = { CreateMutexA(NULL, FALSE, NULL), CreateEventA(NULL, TRUE, FALSE, NULL) };
    if (!h[0] || !h[1]) return;
    HANDLE t = CreateThread(NULL, 0, zx_mutex_holder, h, 0, NULL);
    if (t) {
        WaitForSingleObject(h[1], 2000);
        if (WaitForSingleObject(h[0], 2000) == WAIT_OBJECT_0) ReleaseMutex(h[0]);
        WaitForSingleObject(t, 3000); CloseHandle(t);
    }
    CloseHandle(h[0]); CloseHandle(h[1]);
}
static void CALLBACK zx_tp_cb(PTP_CALLBACK_INSTANCE inst, void *ctx) { (void)inst; SetEvent((HANDLE)ctx); }
static void zx_cb_threadpool(void *a) {
    (void)a; HANDLE e = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!e) return;
    if (TrySubmitThreadpoolCallback(zx_tp_cb, e, NULL)) WaitForSingleObject(e, 3000);
    CloseHandle(e);
}
static void CALLBACK zx_apc_fn(ULONG_PTR p) { (*(volatile long *)p)++; }
static volatile long g_apc_hits;
static void zx_cb_apc(void *a) {
    (void)a;
    if (QueueUserAPC(zx_apc_fn, GetCurrentThread(), (ULONG_PTR)&g_apc_hits)) SleepEx(200, TRUE);
}
static DWORD WINAPI zx_zmm_user_thread(void *p) {
    static t_vregs i2, o2;   /* used by one helper thread at a time */
    (void)p;
    t_vregs_pattern(&i2, T_VK_ZMM, 0x6F0);
    t_vregs_call(T_VK_ZMM, &i2, &o2, zx_cb_sleep0, NULL);
    return 0;
}
static void zx_cb_thread_zmm(void *a) {
    (void)a; HANDLE t = CreateThread(NULL, 0, zx_zmm_user_thread, NULL, 0, NULL);
    if (t) { WaitForSingleObject(t, 5000); CloseHandle(t); }
}
static void zx_cb_nested(void *a) {
    static t_vregs i2, o2;
    (void)a;
    t_vregs_pattern(&i2, T_VK_ZMM, 0x6F1);
    t_vregs_call(T_VK_ZMM, &i2, &o2, zx_cb_sleep1, NULL);
#ifdef T_X64
    for (int i = 6; i < 16; i++) {
        char what[64]; snprintf(what, sizeof what, "inner nested call xmm%d[127:0]", i);
        CHECK_REG(what, o2.v[i], i2.v[i], 16);
    }
#endif
}
static LONG CALLBACK zx_raise_veh(EXCEPTION_POINTERS *ep) {
    return ep->ExceptionRecord->ExceptionCode == 0xE05A5A01u ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
static void zx_cb_raise(void *a) {
    (void)a; PVOID h = AddVectoredExceptionHandler(1, zx_raise_veh);
    if (!h) return;
    RaiseException(0xE05A5A01u, 0, 0, NULL);
    RemoveVectoredExceptionHandler(h);
}
static void zx_cb_memory(void *a) {
    (void)a;
    for (int i = 0; i < 8; i++) {
        void *p = VirtualAlloc(NULL, 1 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!p) break;
        memset(p, i, 4096); DWORD old; VirtualProtect(p, 4096, PAGE_READONLY, &old);
        VirtualFree(p, 0, MEM_RELEASE);
    }
}
static void zx_cb_file_io(void *a) {
    (void)a; char path[MAX_PATH]; t_tmpfile(path, sizeof path, "zmm_io.bin");
    HANDLE f = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_FLAG_DELETE_ON_CLOSE, NULL);
    if (f == INVALID_HANDLE_VALUE) return;
    static char buf[65536]; DWORD n;
    memset(buf, 0x5A, sizeof buf);
    WriteFile(f, buf, sizeof buf, &n, NULL); FlushFileBuffers(f);
    SetFilePointer(f, 0, NULL, FILE_BEGIN); ReadFile(f, buf, sizeof buf, &n, NULL);
    CloseHandle(f);
}
static void zx_cb_dll(void *a) {
    (void)a; HMODULE m = LoadLibraryA("version.dll");
    if (m) { (void)GetProcAddress(m, "GetFileVersionInfoSizeA"); FreeLibrary(m); }
}
TEST(zmm_preserve_cb_sleep0) { zx_cb_case(zx_cb_sleep0, NULL, "Sleep(0)"); }
TEST(zmm_preserve_cb_sleep1) { zx_cb_case(zx_cb_sleep1, NULL, "Sleep(1)"); }
TEST(zmm_preserve_cb_switch_to_thread) { zx_cb_case(zx_cb_switch, NULL, "SwitchToThread x16"); }
TEST(zmm_preserve_cb_wait_signaled) { zx_cb_case(zx_cb_wait_signaled, NULL, "WaitForSingleObject(signaled)"); }
TEST(zmm_preserve_cb_wait_timeout) { zx_cb_case(zx_cb_wait_timeout, NULL, "WaitForSingleObject(15 ms timeout)"); }
TEST(zmm_preserve_cb_mutex_contention) { zx_cb_case(zx_cb_mutex, NULL, "contended mutex wait"); }
TEST(zmm_preserve_cb_threadpool) { zx_cb_case(zx_cb_threadpool, NULL, "thread-pool callback + wait"); }
TEST(zmm_preserve_cb_apc) {
    g_apc_hits = 0;
    zx_cb_case(zx_cb_apc, NULL, "user APC in alertable SleepEx");
    CHECK_EQ(g_apc_hits, 1);
}
TEST(zmm_preserve_cb_other_thread_uses_zmm) { zx_cb_case(zx_cb_thread_zmm, NULL, "join of a thread that used ZMM"); }
TEST(zmm_preserve_cb_nested) { zx_cb_case(zx_cb_nested, NULL, "nested ZMM call with Sleep(1)"); }
TEST(zmm_preserve_cb_raise_exception) { zx_cb_case(zx_cb_raise, NULL, "RaiseException continued by a VEH"); }
TEST(zmm_preserve_cb_virtual_memory) { zx_cb_case(zx_cb_memory, NULL, "VirtualAlloc/Protect/Free"); }
TEST(zmm_preserve_cb_file_io) { zx_cb_case(zx_cb_file_io, NULL, "file write/flush/read"); }
TEST(zmm_preserve_cb_load_library) { zx_cb_case(zx_cb_dll, NULL, "LoadLibrary/FreeLibrary"); }

/* ---- spinning workers: nothing but the OS can touch the registers between load and store ---- */
#define ZX_SPIN_MAX 400000000ul
#define ZX_MAX_SP 1024
typedef struct zx_spinner {
    t_vregs in, out;
    HANDLE h;
    volatile long stop, ready;
    long remaining;
} zx_spinner;
static zx_spinner g_sp[ZX_MAX_SP];
static DWORD WINAPI zx_spin_thread(void *p) {
    zx_spinner *s = p;
    InterlockedExchange(&s->ready, 1);
    s->remaining = t_vregs_spin(T_VK_ZMM, &s->in, &s->out, &s->stop, ZX_SPIN_MAX);
    return 0;
}
static int zx_spin_start(zx_spinner *s, uint32_t tag, DWORD_PTR aff, SIZE_T stack) {
    t_vregs_pattern(&s->in, T_VK_ZMM, tag);
    memset(&s->out, 0, sizeof s->out);
    s->stop = 0; s->ready = 0; s->remaining = -1;
    s->h = CreateThread(NULL, stack, zx_spin_thread, s, CREATE_SUSPENDED | (stack ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0), NULL);
    if (!s->h) return 0;
    if (aff && !SetThreadAffinityMask(s->h, aff)) { TerminateThread(s->h, 0); CloseHandle(s->h); s->h = NULL; return 0; }
    ResumeThread(s->h);
    return 1;
}
static void zx_spin_wait_ready(zx_spinner *s, int n) {
    double t0 = t_now_ms();
    for (int i = 0; i < n; i++) while (!s[i].ready && t_now_ms() - t0 < 5000) Sleep(1);
}
static int zx_spin_stop(zx_spinner *s, int n) {
    int stuck = 0;
    for (int i = 0; i < n; i++) InterlockedExchange(&s[i].stop, 1);
    double t0 = t_now_ms();
    for (int i = 0; i < n; i++) {
        double left = 15000 - (t_now_ms() - t0);
        if (WaitForSingleObject(s[i].h, left > 0 ? (DWORD)left : 0) != WAIT_OBJECT_0) { stuck++; TerminateThread(s[i].h, 1); }
        CloseHandle(s[i].h); s[i].h = NULL;
    }
    return stuck;
}
static void zx_spin_group(int n, DWORD_PTR aff, DWORD run_ms, uint32_t tagbase, SIZE_T stack, const char *what) {
    int started = 0;
    for (; started < n; started++) if (!zx_spin_start(&g_sp[started], tagbase + (uint32_t)started, aff, stack)) break;
    if (started < 2) {
        zx_spin_stop(g_sp, started);
        SKIP("could create only %d worker threads", started);
    }
    if (started < n) t_note("created %d of %d threads", started, n);
    zx_spin_wait_ready(g_sp, started);
    Sleep(run_ms);
    int stuck = zx_spin_stop(g_sp, started);
    CHECKF(stuck == 0, "%d spinning threads did not stop within the bound", stuck);
    int bad = 0;
    for (int i = 0; i < started; i++) {
        char lbl[96]; snprintf(lbl, sizeof lbl, "%s, thread %d", what, i);
        bad += t_vregs_check(&g_sp[i].out, &g_sp[i].in, T_VK_ZMM, 0, -1, lbl) != 0;
    }
    t_info("threads", "%d", started);
    t_info("threads_with_corruption", "%d", bad);
}

TEST(zmm_preserve_oversubscribed_threads) {
    NEED_F();
    int n = t_env_get()->ncpu * 2;
    if (n < 4) n = 4;
    if (n > 64) n = 64;
    zx_spin_group(n, 0, 400, 0x7000, 0, "preempted (2x CPUs)");
}
TEST(zmm_preserve_same_cpu_threads) {
    NEED_F();
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) || !pm) SKIP("no process affinity mask");
    DWORD_PTR one = pm & (~pm + 1);
    zx_spin_group(6, one, 400, 0x7100, 0, "time-sliced on one CPU");
}
TEST(zmm_preserve_256_threads) {
    NEED_F();
    zx_spin_group(256, 0, 600, 0x7200, 65536, "256 threads");
}
TEST_MANUAL(zmm_preserve_1024_threads) {
    NEED_F();
    zx_spin_group(1024, 0, 1500, 0x7600, 65536, "1024 threads");
}

TEST(zmm_preserve_suspend_resume) {
    NEED_F();
    zx_spinner *s = &g_sp[0];
    REQUIRE(zx_spin_start(s, 0x7300, 0, 0));
    zx_spin_wait_ready(s, 1);
    Sleep(20);
    int ok = 0;
    for (int i = 0; i < 200; i++) {
        if (SuspendThread(s->h) == (DWORD)-1) break;
        CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
        ok += GetThreadContext(s->h, &c) ? 1 : 0;
        ResumeThread(s->h);
        if (i & 1) Sleep(0); else SwitchToThread();
    }
    CHECK_EQ(ok, 200);
    CHECK_EQ(zx_spin_stop(s, 1), 0);
    t_vregs_check(&s->out, &s->in, T_VK_ZMM, 0, -1, "after 200x Suspend/GetThreadContext/Resume");
}

TEST(zmm_preserve_set_context_without_xstate) {
    NEED_F();
    /* SetThreadContext with CONTROL|INTEGER (no FP, no XSTATE) must not touch vector state */
    zx_spinner *s = &g_sp[0];
    REQUIRE(zx_spin_start(s, 0x7310, 0, 0));
    zx_spin_wait_ready(s, 1);
    Sleep(20);
    int ok = 0;
    for (int i = 0; i < 50; i++) {
        if (SuspendThread(s->h) == (DWORD)-1) break;
        CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
        if (GetThreadContext(s->h, &c) && SetThreadContext(s->h, &c)) ok++;
        ResumeThread(s->h);
        Sleep(0);
    }
    CHECK_EQ(ok, 50);
    CHECK_EQ(zx_spin_stop(s, 1), 0);
    t_vregs_check(&s->out, &s->in, T_VK_ZMM, 0, -1, "after SetThreadContext(CONTROL|INTEGER)");
}

TEST(zmm_preserve_set_context_legacy_fp) {
    NEED_F();
    zx_spinner *s = &g_sp[0];
    REQUIRE(zx_spin_start(s, 0x7320, 0, 0));
    zx_spin_wait_ready(s, 1);
    Sleep(20);
    REQUIRE(SuspendThread(s->h) != (DWORD)-1);
    static CONTEXT c; memset(&c, 0, sizeof c);
#ifdef T_X64
    c.ContextFlags = CONTEXT_FLOATING_POINT;
#else
    c.ContextFlags = CONTEXT_FLOATING_POINT | CONTEXT_EXTENDED_REGISTERS;
#endif
    BOOL g = GetThreadContext(s->h, &c), st = g && SetThreadContext(s->h, &c);
    ResumeThread(s->h);
    CHECK(g); CHECK(st);
    CHECK_EQ(zx_spin_stop(s, 1), 0);
    for (int i = 0; i < T_NXMM; i++) {
        char lbl[64]; snprintf(lbl, sizeof lbl, "xmm%d[127:0] after SetThreadContext(FP)", i);
        CHECK_REG(lbl, s->out.v[i], s->in.v[i], 16);
    }
    int up = 0;
    for (int i = 0; i < T_NZMM; i++) if (memcmp(s->out.v[i] + 16, s->in.v[i] + 16, 48)) up++;
    for (int i = 0; i < 8; i++) if (s->out.k[i] != s->in.k[i]) up++;
    if (up) t_note("legacy-FP-only SetThreadContext changed %d upper-ZMM/opmask registers (not covered by a documented contract)", up);
    else t_note("legacy-FP-only SetThreadContext left all AVX-512 state intact");
}

/* ---- fresh threads must not see another thread's AVX-512 state ---- */
static zctx g_fresh;
static DWORD WINAPI zx_fresh_thread(void *p) { zmm_rf_store_only((zctx *)p); return 0; }
static int zx_find_pattern(uint64_t v, uint32_t tag0, int ntags) {
    for (int t = 0; t < ntags; t++)
        for (uint32_t r = 0; r < 32; r++)
            for (uint32_t w = 0; w < 8; w++) if (t_pattern_word(tag0 + (uint32_t)t, r, w) == v) return 1;
    return 0;
}
TEST(zmm_isolation_fresh_thread_no_leak) {
    NEED_F();
    /* polluters keep live AVX-512 patterns on other threads while fresh threads are created */
    int np = 4;
    for (int i = 0; i < np; i++)
        if (!zx_spin_start(&g_sp[i], 0x7400 + (uint32_t)i, 0, 0)) { zx_spin_stop(g_sp, i); SKIP("could not start polluter threads"); }
    zx_spin_wait_ready(g_sp, np);
    int leaks = 0, zero_threads = 0;
    for (int it = 0; it < 16; it++) {
        memset(&g_fresh, 0, sizeof g_fresh);
        g_fresh.kbw = (uint32_t)cpu_has(F_AVX512BW);
        HANDLE h = CreateThread(NULL, 0, zx_fresh_thread, &g_fresh, 0, NULL);
        if (!h) { CHECKF(0, "CreateThread failed: %lu", (unsigned long)GetLastError()); break; }
        WaitForSingleObject(h, 5000); CloseHandle(h);
        int nz = 0;
        for (int r = 0; r < T_NZMM; r++)
            for (int w = 0; w < 8; w++) {
                uint64_t v; memcpy(&v, g_fresh.out.v[r] + 8 * w, 8);
                if (v) nz++;
                if (v && zx_find_pattern(v, 0x7400, np)) {
                    leaks++;
                    if (leaks < 4) t_note("fresh thread %d: zmm%d word %d = 0x%016llx belongs to a polluter thread", it, r, w,
                                          (unsigned long long)v);
                }
            }
        for (int k = 0; k < 8; k++) if (g_fresh.out.k[k]) nz++;
        if (!nz) zero_threads++;
    }
    CHECK_EQ(zx_spin_stop(g_sp, np), 0);
    CHECKF(leaks == 0, "%d register words of other threads visible in freshly created threads", leaks);
    t_info("fresh_threads_all_zero", "%d/16", zero_threads);
    for (int i = 0; i < np; i++) t_vregs_check(&g_sp[i].out, &g_sp[i].in, T_VK_ZMM, 0, -1, "polluter");
}

/* ---- exceptions: fault with live AVX-512 state, VEH continues, all state restored ---- */
static struct {
    volatile LONG armed;
    DWORD tid, code;
    int len, mode;           /* mode 0 continue, 1 inspect, 2 modify */
    volatile LONG hits;
    DWORD flags;
    int located[8];
    DWORD lens[8];
    uint8_t k[64], hi256[512], hi16[1024];
    uint8_t new_hi256[32], new_hi16[64];
    uint64_t new_k3;
    zx_api api;
} g_veh;
static LONG CALLBACK zx_veh(EXCEPTION_POINTERS *ep) {
    if (!g_veh.armed || GetCurrentThreadId() != g_veh.tid) return EXCEPTION_CONTINUE_SEARCH;
    if (ep->ExceptionRecord->ExceptionCode != g_veh.code) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord;
    g_veh.hits++;
    g_veh.flags = c->ContextFlags;
    if (g_veh.mode && (c->ContextFlags & CONTEXT_XSTATE) == CONTEXT_XSTATE && g_veh.api.loc) {
        uint8_t *dst[8] = { 0, 0, 0, 0, 0, g_veh.k, g_veh.hi256, g_veh.hi16 };
        for (DWORD f = 5; f <= 7; f++) {
            DWORD l = 0;
            uint8_t *p = g_veh.api.loc(c, f, &l);
            g_veh.lens[f] = l;
            g_veh.located[f] = p != NULL;
            if (!p) continue;
            DWORD n = f == 5 ? 64 : f == 6 ? 512 : 1024;
            if (l < n) n = l;
            for (DWORD i = 0; i < n; i++) dst[f][i] = p[i];
            if (g_veh.mode == 2) {
                if (f == 6) for (int i = 0; i < 32; i++) p[32 + i] = g_veh.new_hi256[i];            /* zmm1[511:256] */
                if (f == 7) for (int i = 0; i < 64; i++) p[64 + i] = g_veh.new_hi16[i];             /* zmm17 */
                if (f == 5) for (int i = 0; i < 8; i++) p[24 + i] = (uint8_t)(g_veh.new_k3 >> (8 * i)); /* k3 */
            }
        }
        if (g_veh.mode == 2 && g_veh.api.getm && g_veh.api.setm) {
            DWORD64 m = 0;
            if (g_veh.api.getm(c, &m)) g_veh.api.setm(c, m | XSTATE_MASK_AVX512);
        }
    }
#ifdef T_X64
    c->Rip = (DWORD64)(ULONG_PTR)ep->ExceptionRecord->ExceptionAddress + (DWORD64)g_veh.len;
#else
    c->Eip = (DWORD)(ULONG_PTR)ep->ExceptionRecord->ExceptionAddress + (DWORD)g_veh.len;
#endif
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void *g_noaccess;
static void zx_exc_run(void (*fn)(zctx *), DWORD code, int len, int mode, uint32_t tag) {
    zx_ctx_init(&g_zc, tag);
    if (!g_noaccess) g_noaccess = VirtualAlloc(NULL, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    REQUIRE(g_noaccess);
    g_zc.ptr = g_noaccess;
    memset(&g_veh, 0, sizeof g_veh);
    zx_api_get(&g_veh.api);
    g_veh.tid = GetCurrentThreadId(); g_veh.code = code; g_veh.len = len; g_veh.mode = mode;
    if (mode == 2) {
        t_pattern_reg(g_veh.new_hi256, 32, 0x7E7E, 1);
        t_pattern_reg(g_veh.new_hi16, 64, 0x7E7E, 17);
        g_veh.new_k3 = g_zc.kbw ? 0x5AA5C33C0FF01234ull : 0x5AA5;
    }
    PVOID h = AddVectoredExceptionHandler(1, zx_veh);
    REQUIRE(h);
    InterlockedExchange(&g_veh.armed, 1);
    fn(&g_zc);
    InterlockedExchange(&g_veh.armed, 0);
    RemoveVectoredExceptionHandler(h);
    CHECK_EQ(g_veh.hits, 1);
}
static void zx_exc_case(void (*fn)(zctx *), DWORD code, int len, const char *what) {
    NEED_F();
    zx_exc_run(fn, code, len, 0, 0x7500 + code % 97);
    t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, what);
    t_info("exception_context_flags", "0x%lx", (unsigned long)g_veh.flags);
}
TEST(zmm_exception_ud_continue) { zx_exc_case(zmm_rf_fault_ud, STATUS_ILLEGAL_INSTRUCTION, 2, "after #UD + VEH continue"); }
TEST(zmm_exception_av_continue) { zx_exc_case(zmm_rf_fault_av, STATUS_ACCESS_VIOLATION, 2, "after #PF (AV) + VEH continue"); }
TEST(zmm_exception_div0_continue) { zx_exc_case(zmm_rf_fault_div, STATUS_INTEGER_DIVIDE_BY_ZERO, 2, "after #DE + VEH continue"); }
TEST(zmm_exception_int3_continue) { zx_exc_case(zmm_rf_fault_int3, STATUS_BREAKPOINT, 1, "after INT3 + VEH continue"); }

TEST(zmm_exception_context_reports_avx512) {
    NEED_F();
    zx_exc_run(zmm_rf_fault_ud, STATUS_ILLEGAL_INSTRUCTION, 2, 1, 0x7580);
    t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, "after #UD (inspecting VEH)");
    int has_x = (g_veh.flags & CONTEXT_XSTATE) == CONTEXT_XSTATE;
    t_info("context_flags", "0x%lx", (unsigned long)g_veh.flags);
    t_info("located", "k=%d hi256=%d hi16=%d", g_veh.located[5], g_veh.located[6], g_veh.located[7]);
    if (!has_x || !g_veh.located[6]) {
        t_note("the exception ContextRecord does not expose AVX-512 state (flags 0x%lx); contents not checked",
               (unsigned long)g_veh.flags);
        return;
    }
    CHECK_EQ(g_veh.lens[6], 512);
    for (int i = 0; i < T_NXMM; i++) {
        char lbl[64]; snprintf(lbl, sizeof lbl, "ContextRecord ZMM_Hi256 zmm%d", i);
        CHECK_REG(lbl, g_veh.hi256 + 32 * i, g_zc.in.v[i] + 32, 32);
    }
    if (g_veh.located[5]) {
        CHECK_EQ(g_veh.lens[5], 64);
        for (int i = 0; i < 8; i++) {
            uint64_t k; memcpy(&k, g_veh.k + 8 * i, 8);
            CHECKF(k == g_zc.in.k[i], "ContextRecord k%d 0x%llx want 0x%llx", i, (unsigned long long)k, (unsigned long long)g_zc.in.k[i]);
        }
    }
#ifdef T_X64
    if (g_veh.located[7]) {
        CHECK_EQ(g_veh.lens[7], 1024);
        for (int j = 0; j < 16; j++) {
            char lbl[64]; snprintf(lbl, sizeof lbl, "ContextRecord Hi16_ZMM zmm%d", 16 + j);
            CHECK_REG(lbl, g_veh.hi16 + 64 * j, g_zc.in.v[16 + j], 64);
        }
    }
#endif
}

TEST(zmm_exception_context_modify_avx512) {
    NEED_F();
    zx_exc_run(zmm_rf_fault_ud, STATUS_ILLEGAL_INSTRUCTION, 2, 2, 0x7590);
    if ((g_veh.flags & CONTEXT_XSTATE) != CONTEXT_XSTATE || !g_veh.located[6]) {
        t_vregs_check(&g_zc.out, &g_zc.in, T_VK_ZMM, 0, -1, "after #UD");
        SKIP("the exception ContextRecord does not expose AVX-512 state (flags 0x%lx)", (unsigned long)g_veh.flags);
    }
    t_vregs want = g_zc.in;
    memcpy(want.v[1] + 32, g_veh.new_hi256, 32);
#ifdef T_X64
    if (g_veh.located[7]) memcpy(want.v[17], g_veh.new_hi16, 64);
#endif
    if (g_veh.located[5]) want.k[3] = g_veh.new_k3;
    t_vregs_check(&g_zc.out, &want, T_VK_ZMM, 0, -1, "after continuing with a modified ContextRecord");
}

/* ---- Get/SetThreadContext with AVX-512 XSTATE on a spinning thread ---- */
static int zx_ctx_matches_spin(zx_api *a, CONTEXT *ctx, const t_vregs *in) {
    if (memcmp(zx_ctx_xmm(ctx, 0), in->v[0], 16)) return 0;
    DWORD l = 0; uint8_t *k = a->loc(ctx, 5, &l);
    if (k && memcmp(k + 56, &in->k[7], 8)) return 0;
    return 1;
}
static CONTEXT *zx_capture_spinner(zx_api *a, zx_spinner *s, uint8_t *buf, int *suspended) {
    const char *why = "";
    *suspended = 0;
    for (int attempt = 0; attempt < 25; attempt++) {
        CONTEXT *ctx = zx_context_new(a, buf, ZX_XFLAGS, NULL, &why);
        if (!ctx) return NULL;
        if (!a->setm(ctx, XSTATE_MASK_AVX | XSTATE_MASK_AVX512)) return NULL;
        if (SuspendThread(s->h) == (DWORD)-1) return NULL;
        if (!GetThreadContext(s->h, ctx)) { ResumeThread(s->h); return NULL; }
        if (zx_ctx_matches_spin(a, ctx, &s->in)) { *suspended = 1; return ctx; }
        ResumeThread(s->h);
        Sleep(20);
    }
    return NULL;
}
TEST(zmm_context_get_xstate_avx512) {
    NEED_F();
    zx_api a; NEED_API(&a);
    if ((a.en() & ZX_AVX512) != ZX_AVX512) SKIP("GetEnabledXStateFeatures does not report AVX-512 (0x%llx)", (unsigned long long)a.en());
    zx_spinner *s = &g_sp[0];
    REQUIRE(zx_spin_start(s, 0x7700, 0, 0));
    zx_spin_wait_ready(s, 1);
    Sleep(20);
    int susp = 0;
    CONTEXT *ctx = zx_capture_spinner(&a, s, g_ctxbuf[0], &susp);
    if (!ctx) { if (susp) ResumeThread(s->h); zx_spin_stop(s, 1); REQUIREF(0, "could not capture the spinning thread's XSTATE context"); }
    DWORD64 m = 0;
    CHECK(a.getm(ctx, &m));
    t_info("captured_mask", "0x%llx", (unsigned long long)m);
    CHECKF((m & 0x64) == 0x64, "GetXStateFeaturesMask=0x%llx: AVX/opmask/ZMM_Hi256 hold non-init values", (unsigned long long)m);
#ifdef T_X64
    CHECKF(m & 0x80, "GetXStateFeaturesMask=0x%llx lacks Hi16_ZMM", (unsigned long long)m);
#endif
    DWORD l5 = 0, l6 = 0, l7 = 0, l2 = 0;
    uint8_t *pk = a.loc(ctx, 5, &l5), *ph = a.loc(ctx, 6, &l6), *p16 = a.loc(ctx, 7, &l7), *py = a.loc(ctx, 2, &l2);
    int x86_missing = 0;
#ifdef T_X86
    if (!pk || !ph) { x86_missing = 1; t_note("WOW64/x86 CONTEXT does not expose AVX-512 components (opmask %p, ZMM_Hi256 %p)", (void *)pk, (void *)ph); }
#endif
    if (!x86_missing) {
        CHECK(pk != NULL); CHECK(ph != NULL);
        CHECK_EQ(l5, 64); CHECK_EQ(l6, 512);
    }
    char lbl[80];
    for (int i = 0; i < T_NXMM; i++) {
        snprintf(lbl, sizeof lbl, "CONTEXT legacy xmm%d", i);
        CHECK_REG(lbl, zx_ctx_xmm(ctx, i), s->in.v[i], 16);
        if (py) { snprintf(lbl, sizeof lbl, "CONTEXT AVX ymm%d[255:128]", i); CHECK_REG(lbl, py + 16 * i, s->in.v[i] + 16, 16); }
        if (ph) { snprintf(lbl, sizeof lbl, "CONTEXT ZMM_Hi256 zmm%d[511:256]", i); CHECK_REG(lbl, ph + 32 * i, s->in.v[i] + 32, 32); }
    }
    if (pk) for (int i = 0; i < 8; i++) {
        uint64_t k; memcpy(&k, pk + 8 * i, 8);
        CHECKF(k == s->in.k[i], "CONTEXT opmask k%d 0x%llx want 0x%llx", i, (unsigned long long)k, (unsigned long long)s->in.k[i]);
    }
#ifdef T_X64
    CHECK(p16 != NULL); CHECK_EQ(l7, 1024);
    if (p16) for (int j = 0; j < 16; j++) {
        snprintf(lbl, sizeof lbl, "CONTEXT Hi16_ZMM zmm%d", 16 + j);
        CHECK_REG(lbl, p16 + 64 * j, s->in.v[16 + j], 64);
    }
#else
    (void)p16; (void)l7;
#endif
    ResumeThread(s->h);
    CHECK_EQ(zx_spin_stop(s, 1), 0);
    t_vregs_check(&s->out, &s->in, T_VK_ZMM, 0, -1, "after GetThreadContext(XSTATE)");
}

TEST(zmm_context_set_xstate_avx512) {
    NEED_F();
    zx_api a; NEED_API(&a);
    if ((a.en() & ZX_AVX512) != ZX_AVX512) SKIP("GetEnabledXStateFeatures does not report AVX-512 (0x%llx)", (unsigned long long)a.en());
    zx_spinner *s = &g_sp[0];
    REQUIRE(zx_spin_start(s, 0x7710, 0, 0));
    zx_spin_wait_ready(s, 1);
    Sleep(20);
    int susp = 0;
    CONTEXT *ctx = zx_capture_spinner(&a, s, g_ctxbuf[0], &susp);
    if (!ctx) { if (susp) ResumeThread(s->h); zx_spin_stop(s, 1); REQUIREF(0, "could not capture the spinning thread's XSTATE context"); }
    DWORD l = 0;
    uint8_t *pk = a.loc(ctx, 5, &l), *ph = a.loc(ctx, 6, &l), *p16 = a.loc(ctx, 7, &l);
    if (!pk || !ph) {
        ResumeThread(s->h); zx_spin_stop(s, 1);
#ifdef T_X86
        SKIP("x86 CONTEXT does not expose AVX-512 components");
#else
        REQUIREF(0, "LocateXStateFeature failed for enabled AVX-512 features");
#endif
    }
    static t_vregs nv; t_vregs_pattern(&nv, T_VK_ZMM, 0x7711);
    for (int i = 0; i < 8; i++) nv.k[i] |= 1;
    for (int i = 0; i < T_NXMM; i++) memcpy(ph + 32 * i, nv.v[i] + 32, 32);
#ifdef T_X64
    if (p16) for (int j = 0; j < 16; j++) memcpy(p16 + 64 * j, nv.v[16 + j], 64);
#else
    (void)p16;
#endif
    for (int i = 0; i < 8; i++) memcpy(pk + 8 * i, &nv.k[i], 8);
    CHECK(a.setm(ctx, XSTATE_MASK_AVX | XSTATE_MASK_AVX512));
    BOOL ok = SetThreadContext(s->h, ctx);
    ResumeThread(s->h);
    CHECKF(ok, "SetThreadContext(XSTATE) failed: %lu", (unsigned long)GetLastError());
    Sleep(20);
    CHECK_EQ(zx_spin_stop(s, 1), 0);
    t_vregs want;
    zx_merge_avx512(&want, &s->in, &nv);
    t_vregs_check(&s->out, &want, T_VK_ZMM, 0, -1, "after SetThreadContext with modified AVX-512 state");
}
