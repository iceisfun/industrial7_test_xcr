/* xcr0_probe - XGETBV legality, XCR0 architectural invariants, XINUSE (XGETBV ECX=1), CPUID leaf 0Dh
 * consistency, IA32_XSS vs XCR0 separation, XSETBV privilege, stability of XCR0 across CPUs, threads,
 * time, processes and the 32/64-bit (WOW64) view, the AVX/AVX2/AVX-512/AMX eligibility rules, and the
 * Windows reporting APIs (GetEnabledXStateFeatures, RtlGetEnabledExtendedFeatures,
 * GetThreadEnabledXStateFeatures, IsProcessorFeaturePresent).
 *
 * Principle: hard CHECKs only for what the Intel SDM / AMD APM guarantee (XSETBV #GP conditions imply
 * the XCR0 invariants; leaf 0Dh enumeration rules; XGETBV/XINUSE semantics; fault semantics) or what a
 * documented Windows contract guarantees. Everything implementation-dependent is NOTE/INFO.
 *
 * Windows reference (Windows 10 22H2 build 19041 corpus, checked through the decompiler API):
 *  - kernelbase!GetEnabledXStateFeatures = RtlGetEnabledExtendedFeatures(~0), or 3 when that returns 0.
 *  - ntdll!RtlGetEnabledExtendedFeatures(m) = (KUSER_SHARED_DATA.XState.EnabledFeatures |
 *    XState.EnabledUserVisibleSupervisorFeatures) & m   (reads 0x7ffe03d8 and 0x7ffe0708; the
 *    _XSTATE_CONFIGURATION PDB type puts EnabledUserVisibleSupervisorFeatures at +0x330).
 *    So the API may legitimately report supervisor bits (e.g. CET_U) that are NOT in XCR0.
 *  - ntoskrnl!KiPreprocessFault turns a user #GP into STATUS_ACCESS_VIOLATION and then consults
 *    KiOpDecode / KiOpTwoByteTable: only listed opcodes become STATUS_PRIVILEGED_INSTRUCTION
 *    (KiOp_Priv). The table has 0F 01 /2 only in its MEMORY form (LGDT); XSETBV (0F 01 D1) and an invalid
 *    XGETBV (0F 01 D0) are register forms and are not listed, so on that build a user-mode XSETBV is
 *    reported as STATUS_ACCESS_VIOLATION (0xC0000005), not 0xC0000096. Both codes are accepted; the
 *    contract checked is "it faults, at the instruction, and changes nothing".
 */
#include "t.h"
#include <immintrin.h>

/* ---------------------------------------------------------------------------------------------
 * Component classes (SDM Vol.1 13.1, Table 13-1; AMD APM Vol.2 11.5.2)
 * ------------------------------------------------------------------------------------------- */
#define XP_USER_KNOWN  (0xFFull | T_XS_PKRU | T_XS_TILECFG | T_XS_TILEDATA | T_XS_APX)       /* 0-7,9,17,18,19 */
#define XP_SUP_KNOWN   (T_XS_PT | T_XS_PASID | T_XS_CET_U | T_XS_CET_S | T_XS_HDC | T_XS_UINTR | T_XS_LBR | T_XS_HWP) /* 8,10-16 */
#define XP_AVX512_BITS (T_XS_OPMASK | T_XS_ZMM_HI256 | T_XS_HI16_ZMM)                        /* 0xE0 */
#define XP_MPX_BITS    (T_XS_BNDREGS | T_XS_BNDCSR)                                         /* 0x18 */

#define XP_STATUS_PRIV  0xC0000096u
#define XP_STATUS_AV    0xC0000005u
#define XP_STATUS_ILL   0xC000001Du

/* Windows XSTATE API pointers (resolved at run time: they may be absent) */
typedef DWORD64 (WINAPI *pfn_GetEnabledXStateFeatures)(void);
typedef DWORD64 (WINAPI *pfn_GetThreadEnabledXStateFeatures)(void);
typedef ULONG64 (NTAPI *pfn_RtlGetEnabledExtendedFeatures)(ULONG64);

static pfn_GetEnabledXStateFeatures api_gef(void) {
    return (pfn_GetEnabledXStateFeatures)t_proc("kernel32.dll", "GetEnabledXStateFeatures");
}
static pfn_RtlGetEnabledExtendedFeatures api_rtl(void) {
    return (pfn_RtlGetEnabledExtendedFeatures)t_proc("ntdll.dll", "RtlGetEnabledExtendedFeatures");
}
static pfn_GetThreadEnabledXStateFeatures api_gtef(void) {
    pfn_GetThreadEnabledXStateFeatures p = (pfn_GetThreadEnabledXStateFeatures)t_proc("kernel32.dll", "GetThreadEnabledXStateFeatures");
    if (!p) p = (pfn_GetThreadEnabledXStateFeatures)t_proc("kernelbase.dll", "GetThreadEnabledXStateFeatures");
    return p;
}

/* ---------------------------------------------------------------------------------------------
 * Fresh raw reads (independent of the cached cpu() view)
 * ------------------------------------------------------------------------------------------- */
typedef struct xp_raw {
    uint32_t maxleaf;
    int xsave_hw;           /* CPUID.1:ECX[26] */
    int osxsave;            /* CPUID.1:ECX[27] */
    int avx_hw;             /* CPUID.1:ECX[28] */
    int has_d;              /* maxleaf >= 0xD */
    t_cpuid_regs l1, d0, d1;
    uint64_t sup;           /* (0Dh,0) EDX:EAX */
    uint64_t xss;           /* (0Dh,1) EDX:ECX */
    uint64_t xcr0;          /* XGETBV(0) when osxsave, else 0 */
    int xg1;                /* (0Dh,1):EAX[2] */
} xp_raw;

static void xp_read(xp_raw *x) {
    memset(x, 0, sizeof *x);
    t_cpuid_regs r;
    cpuid_ex(0, 0, &r);
    x->maxleaf = r.eax;
    cpuid_ex(1, 0, &x->l1);
    x->xsave_hw = (x->l1.ecx >> 26) & 1;
    x->osxsave  = (x->l1.ecx >> 27) & 1;
    x->avx_hw   = (x->l1.ecx >> 28) & 1;
    if (x->maxleaf >= 0xD) {
        x->has_d = 1;
        cpuid_ex(0xD, 0, &x->d0);
        cpuid_ex(0xD, 1, &x->d1);
        x->sup = (uint64_t)x->d0.edx << 32 | x->d0.eax;
        x->xss = (uint64_t)x->d1.edx << 32 | x->d1.ecx;
        x->xg1 = (x->d1.eax >> 2) & 1;
    }
    if (x->osxsave) x->xcr0 = t_xgetbv(0);
}

/* Read and SKIP when XGETBV is not legal (OSXSAVE=0): XGETBV would #UD. */
static void xp_need_xcr0(xp_raw *x) {
    xp_read(x);
    if (!x->osxsave) SKIP("CPUID.1:ECX.OSXSAVE=0: the OS has not enabled XSAVE, XGETBV is not executable");
}
static void xp_need_leaf_d(xp_raw *x) {
    xp_read(x);
    if (!x->xsave_hw) SKIP("CPUID.1:ECX.XSAVE=0: no XSAVE feature set");
    if (!x->has_d) SKIP("CPUID max basic leaf 0x%x < 0xD", x->maxleaf);
}
static int bit(uint64_t v, int i) { return (int)((v >> i) & 1); }
static int ospke(void) { t_cpuid_regs r; if (cpuid_max_basic() < 7) return 0; cpuid_ex(7, 0, &r); return (r.ecx >> 4) & 1; }
static int pku_hw(void) { t_cpuid_regs r; if (cpuid_max_basic() < 7) return 0; cpuid_ex(7, 0, &r); return (r.ecx >> 3) & 1; }
static int leaf7_bit(int sub, int reg, int b) {
    t_cpuid_regs r;
    if (cpuid_max_basic() < 7) return 0;
    cpuid_ex(7, 0, &r);
    if (sub > (int)r.eax) return 0;
    cpuid_ex(7, (uint32_t)sub, &r);
    return (int)((cpuid_reg(&r, reg) >> b) & 1);
}
static int hex_bits(char *buf, size_t cap, uint64_t m) {
    size_t n = 0; buf[0] = 0;
    for (int i = 0; i < 64 && n + 16 < cap; i++)
        if (bit(m, i)) n += (size_t)snprintf(buf + n, cap - n, "%s%s", n ? "," : "", xstate_comp_name(i));
    if (!n) snprintf(buf, cap, "-");
    return (int)n;
}

/* Architectural component sizes fixed by the register content (CPUID.(0Dh,i).EAX). */
static int arch_size(int i) {
    switch (i) {
    case 2: return 256;    /* YMM_Hi128: 16 x 16 bytes */
    case 3: return 64;     /* BNDREGS: 4 x 16 */
    case 4: return 16;     /* BNDCFGU + BNDSTATUS */
    case 5: return 64;     /* k0-k7 x 8 */
    case 6: return 512;    /* ZMM_Hi256: 16 x 32 */
    case 7: return 1024;   /* Hi16_ZMM: 16 x 64 */
    case 9: return 8;      /* PKRU (4 bytes + 4 padding) */
    case 11: return 16;    /* CET_U: IA32_U_CET + IA32_PL3_SSP */
    case 12: return 24;    /* CET_S: IA32_PL0..2_SSP */
    case 17: return 64;    /* TILECFG */
    case 18: return 8192;  /* TILEDATA: 8 tiles x 1 KB */
    case 19: return 128;   /* APX: R16-R31 */
    default: return 0;
    }
}

/* ---------------------------------------------------------------------------------------------
 * Faulting-instruction helpers (only ever executed in child processes)
 * ------------------------------------------------------------------------------------------- */
static volatile LONG     g_armed;
static volatile LONG     g_hits;
static volatile DWORD    g_code, g_nparams;
static volatile ULONG_PTR g_faddr, g_info0, g_info1;

/* Skips exactly one faulting XGETBV/XSETBV (0F 01 D0 / 0F 01 D1, 3 bytes) after recording the record. */
static LONG CALLBACK xp_skip_veh(EXCEPTION_POINTERS *ep) {
    if (!g_armed) return EXCEPTION_CONTINUE_SEARCH;
    const unsigned char *ip = (const unsigned char *)ep->ExceptionRecord->ExceptionAddress;
    if (!ip || ip[0] != 0x0F || ip[1] != 0x01 || (ip[2] != 0xD0 && ip[2] != 0xD1)) return EXCEPTION_CONTINUE_SEARCH;
    g_armed = 0;
    g_hits++;
    g_code = ep->ExceptionRecord->ExceptionCode;
    g_faddr = (ULONG_PTR)ep->ExceptionRecord->ExceptionAddress;
    g_nparams = ep->ExceptionRecord->NumberParameters;
    g_info0 = g_nparams > 0 ? ep->ExceptionRecord->ExceptionInformation[0] : 0;
    g_info1 = g_nparams > 1 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
#ifdef T_X64
    ep->ContextRecord->Rip += 3;
#else
    ep->ContextRecord->Eip += 3;
#endif
    return EXCEPTION_CONTINUE_EXECUTION;
}

/* XSETBV with VEH skip. *insn = address of the XSETBV instruction. */
static T_NOINLINE void xp_xsetbv_probe(uint32_t idx, uint64_t val, ULONG_PTR *insn) {
    uint32_t lo = (uint32_t)val, hi = (uint32_t)(val >> 32);
    ULONG_PTR a;
#ifdef T_X64
    __asm__ volatile("lea 1f(%%rip), %[a]\n\t1: xsetbv" : [a] "=&r"(a), "+a"(lo), "+d"(hi) : "c"(idx) : "memory");
#else
    __asm__ volatile("movl $1f, %[a]\n\t1: xsetbv" : [a] "=&r"(a), "+a"(lo), "+d"(hi) : "c"(idx) : "memory");
#endif
    *insn = a;
}
/* XGETBV that is expected to fault: EDX:EAX preloaded with a sentinel that a fault must leave intact. */
static T_NOINLINE uint64_t xp_xgetbv_probe(uint32_t idx, ULONG_PTR *insn) {
    uint32_t lo = 0x5A5A1234u, hi = 0xA5A5C0DEu;
    ULONG_PTR a;
#ifdef T_X64
    __asm__ volatile("lea 1f(%%rip), %[a]\n\t1: xgetbv" : [a] "=&r"(a), "+a"(lo), "+d"(hi) : "c"(idx) : "memory");
#else
    __asm__ volatile("movl $1f, %[a]\n\t1: xgetbv" : [a] "=&r"(a), "+a"(lo), "+d"(hi) : "c"(idx) : "memory");
#endif
    *insn = a;
    return (uint64_t)hi << 32 | lo;
}
#define XP_SENTINEL 0xA5A5C0DE5A5A1234ull

static void child_print_fault(ULONG_PTR insn) {
    printf("FAULTED=%ld\n", (long)g_hits);
    printf("CODE=0x%08lx\n", (unsigned long)g_code);
    printf("ADDR_OK=%d\n", g_hits && g_faddr == insn);
    printf("NPARAMS=%lu\n", (unsigned long)g_nparams);
    printf("INFO0=0x%llx\nINFO1=0x%llx\n", (unsigned long long)g_info0, (unsigned long long)g_info1);
}

/* XSETBV(0, current XCR0): harmless even if it were (wrongly) executed. */
CHILD(xsetbv_same) {
    if (!(cpu()->osxsave)) { printf("NO_OSXSAVE=1\n"); return 0; }
    uint64_t before = t_xgetbv(0);
    PVOID h = AddVectoredExceptionHandler(1, xp_skip_veh);
    ULONG_PTR insn = 0;
    g_armed = 1;
    xp_xsetbv_probe(0, before, &insn);
    g_armed = 0;
    if (h) RemoveVectoredExceptionHandler(h);
    child_print_fault(insn);
    printf("XCR0_BEFORE=0x%llx\nXCR0_AFTER=0x%llx\n", (unsigned long long)before, (unsigned long long)t_xgetbv(0));
    return 0;
}
/* XSETBV(0, 0): clears XCR0[0], which #GPs even at CPL0 - can never succeed anywhere. */
CHILD(xsetbv_invalid) {
    if (!(cpu()->osxsave)) { printf("NO_OSXSAVE=1\n"); return 0; }
    uint64_t before = t_xgetbv(0);
    PVOID h = AddVectoredExceptionHandler(1, xp_skip_veh);
    ULONG_PTR insn = 0;
    g_armed = 1;
    xp_xsetbv_probe(0, 0, &insn);
    g_armed = 0;
    if (h) RemoveVectoredExceptionHandler(h);
    child_print_fault(insn);
    printf("XCR0_BEFORE=0x%llx\nXCR0_AFTER=0x%llx\n", (unsigned long long)before, (unsigned long long)t_xgetbv(0));
    return 0;
}
/* XSETBV to XCR1 (read-only XINUSE) with the current XINUSE value. */
CHILD(xsetbv_xcr1) {
    if (!(cpu()->osxsave)) { printf("NO_OSXSAVE=1\n"); return 0; }
    uint64_t before = t_xgetbv(0);
    PVOID h = AddVectoredExceptionHandler(1, xp_skip_veh);
    ULONG_PTR insn = 0;
    g_armed = 1;
    xp_xsetbv_probe(1, cpu()->xinuse_valid ? t_xgetbv(1) : 3, &insn);
    g_armed = 0;
    if (h) RemoveVectoredExceptionHandler(h);
    child_print_fault(insn);
    printf("XCR0_BEFORE=0x%llx\nXCR0_AFTER=0x%llx\n", (unsigned long long)before, (unsigned long long)t_xgetbv(0));
    return 0;
}
/* XGETBV with an explicit index (argv[1], hex). Expected to fault for invalid XCRs. */
CHILD(xgetbv_index) {
    uint32_t idx = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 16) : 0xFFFFFFFFu;
    if (!(cpu()->osxsave)) { printf("NO_OSXSAVE=1\n"); return 0; }
    PVOID h = AddVectoredExceptionHandler(1, xp_skip_veh);
    ULONG_PTR insn = 0;
    g_armed = 1;
    uint64_t v = xp_xgetbv_probe(idx, &insn);
    g_armed = 0;
    if (h) RemoveVectoredExceptionHandler(h);
    child_print_fault(insn);
    printf("VALUE=0x%llx\nUNCHANGED=%d\n", (unsigned long long)v, v == XP_SENTINEL);
    return 0;
}
/* Plain XGETBV(0) with no handler: must run without any exception when OSXSAVE=1. */
CHILD(xgetbv0_plain) {
    t_cpuid_regs r;
    cpuid_ex(1, 0, &r);
    printf("OSXSAVE=%u\n", (r.ecx >> 27) & 1);
    if ((r.ecx >> 27) & 1) printf("XCR0=0x%llx\n", (unsigned long long)t_xgetbv(0));
    return 0;
}
/* XGETBV(0) regardless of OSXSAVE (used only when OSXSAVE=0: must #UD). */
CHILD(xgetbv0_forced) {
    uint64_t v = t_xgetbv(0);
    printf("SURVIVED=1\nXCR0=0x%llx\n", (unsigned long long)v);
    return 0;
}
/* Process-level XCR0 report (also used for the other-architecture exe). */
CHILD(report) {
    xp_raw x;
    xp_read(&x);
    pfn_GetEnabledXStateFeatures gef = api_gef();
    printf("ARCH=%s\n", T_ARCH);
    printf("OSXSAVE=%d\n", x.osxsave);
    printf("XCR0=0x%llx\n", (unsigned long long)x.xcr0);
    printf("XCR0SUP=0x%llx\n", (unsigned long long)x.sup);
    printf("XSS=0x%llx\n", (unsigned long long)x.xss);
    printf("D0EBX=0x%x\n", x.d0.ebx);
    printf("API=0x%llx\n", gef ? (unsigned long long)gef() : 0ull);
    printf("API_PRESENT=%d\n", gef != NULL);
    printf("PF17=%d\n", IsProcessorFeaturePresent(PF_XSAVE_ENABLED) ? 1 : 0);
    if (x.osxsave && x.xg1) printf("XINUSE=0x%llx\n", (unsigned long long)t_xgetbv(1));
    return 0;
}

/* VEX / EVEX instructions used when the state is NOT enabled (children only: must #UD). */
T_TARGET("avx") static T_NOINLINE void ud_probe_avx_ymm(void) {
    __asm__ volatile("vxorps %%ymm0, %%ymm0, %%ymm0\n\tvzeroupper" ::: "xmm0", "memory");
}
T_TARGET("avx512f") static T_NOINLINE void ud_probe_avx512_zmm(void) {
    __asm__ volatile("vpxord %%zmm0, %%zmm0, %%zmm0\n\tvzeroupper" ::: "xmm0", "memory");
}
CHILD(avx_ymm_disabled) { ud_probe_avx_ymm(); printf("SURVIVED=1\n"); return 0; }
CHILD(avx512_zmm_disabled) { ud_probe_avx512_zmm(); printf("SURVIVED=1\n"); return 0; }

static int child_u64(const t_child_result *r, const char *key, uint64_t *v) {
    char buf[128];
    if (!t_child_find(r, key, buf, sizeof buf)) return 0;
    *v = strtoull(buf, NULL, 0);
    return 1;
}
static t_child_result g_cr;   /* large: static */

/* Common evaluation for an expected-fault child */
static void expect_gp_fault(const char *child, const char *args, int check_xcr0) {
    REQUIREF(t_child_run(child, args, 5000, &g_cr), "child did not start (error %lu)", g_cr.spawn_error);
    REQUIREF(!g_cr.timed_out, "child timed out");
    uint64_t faulted = 0, code = 0, addr_ok = 0, np = 0, i0 = 0, i1 = 0;
    CHECKF(g_cr.exc_code == 0, "child died of exception 0x%08lx instead of delivering it to its VEH", g_cr.exc_code);
    CHECK_EQ(g_cr.exit_code, 0);
    REQUIREF(child_u64(&g_cr, "FAULTED=", &faulted), "no FAULTED line; output: %.400s", g_cr.out);
    CHECKF(faulted == 1, "the instruction must raise exactly one exception (got %llu): it executed at CPL3",
           (unsigned long long)faulted);
    if (faulted != 1) return;
    child_u64(&g_cr, "CODE=", &code);
    child_u64(&g_cr, "ADDR_OK=", &addr_ok);
    child_u64(&g_cr, "NPARAMS=", &np);
    child_u64(&g_cr, "INFO0=", &i0);
    child_u64(&g_cr, "INFO1=", &i1);
    t_info("exception_code", "0x%08llx", (unsigned long long)code);
    t_info("exception_params", "n=%llu [0x%llx, 0x%llx]", (unsigned long long)np, (unsigned long long)i0, (unsigned long long)i1);
    CHECKF(code == XP_STATUS_PRIV || code == XP_STATUS_AV,
           "#GP(0) at CPL3 must surface as STATUS_PRIVILEGED_INSTRUCTION or STATUS_ACCESS_VIOLATION, got 0x%08llx",
           (unsigned long long)code);
    if (code == XP_STATUS_AV)
        t_note("reported as STATUS_ACCESS_VIOLATION (what Windows 10 19041 KiOpDecode does for register-form 0F 01 opcodes)");
    else
        t_note("reported as STATUS_PRIVILEGED_INSTRUCTION");
    CHECKF(addr_ok == 1, "#GP is a fault: ExceptionAddress must be the faulting instruction");
    if (check_xcr0) {
        uint64_t b = 0, a = 0;
        CHECK(child_u64(&g_cr, "XCR0_BEFORE=", &b));
        CHECK(child_u64(&g_cr, "XCR0_AFTER=", &a));
        CHECKF(a == b, "XCR0 changed by a faulting XSETBV: 0x%llx -> 0x%llx", (unsigned long long)b, (unsigned long long)a);
    }
}

/* =============================================================================================
 * 1. CPUID gating of XGETBV
 * =========================================================================================== */
TEST(cpuid_osxsave_implies_xsave) {
    xp_raw x; xp_read(&x);
    t_info("cpuid1_ecx", "0x%08x", x.l1.ecx);
    t_info("xsave_osxsave_avx", "%d/%d/%d", x.xsave_hw, x.osxsave, x.avx_hw);
    /* CR4.OSXSAVE can only be set when CPUID.1:ECX.XSAVE=1 (SDM 2.5, MOV CR4 #GP) */
    CHECKF(!x.osxsave || x.xsave_hw, "OSXSAVE=1 but XSAVE=0");
    /* XSAVE support implies leaf 0Dh exists */
    CHECKF(!x.xsave_hw || x.maxleaf >= 0xD, "XSAVE=1 but max basic leaf 0x%x < 0xD", x.maxleaf);
    if (!x.osxsave) t_note("OSXSAVE=0: no XSAVE-managed state is usable (AVX and later are unusable)");
}

TEST(xgetbv0_executes_when_osxsave) {
    xp_raw x; xp_need_xcr0(&x);
    REQUIREF(t_child_run("xgetbv0_plain", "", 5000, &g_cr), "child did not start");
    REQUIRE(!g_cr.timed_out);
    CHECKF(g_cr.exc_code == 0, "XGETBV(0) faulted with 0x%08lx although OSXSAVE=1", g_cr.exc_code);
    CHECK_EQ(g_cr.exit_code, 0);
    uint64_t osx = 0, v = 0;
    CHECK(child_u64(&g_cr, "OSXSAVE=", &osx) && osx == 1);
    CHECK(child_u64(&g_cr, "XCR0=", &v));
    CHECK_EQ(v, x.xcr0);
}

TEST(xgetbv0_ud_without_osxsave) {
    xp_raw x; xp_read(&x);
    if (x.osxsave) SKIP("OSXSAVE=1: XGETBV is legal here (the #UD rule cannot be exercised)");
    REQUIRE(t_child_run("xgetbv0_forced", "", 5000, &g_cr));
    REQUIRE(!g_cr.timed_out);
    CHECKF(!strstr(g_cr.out, "SURVIVED=1"), "XGETBV executed with CR4.OSXSAVE=0 (must be #UD)");
    CHECKF(g_cr.exc_code == XP_STATUS_ILL, "expected STATUS_ILLEGAL_INSTRUCTION, got 0x%08lx", g_cr.exc_code);
}

TEST(xgetbv0_repeatable) {
    xp_raw x; xp_need_xcr0(&x);
    int bad = 0; uint64_t first_bad = 0;
    for (int i = 0; i < 100000; i++) {
        uint64_t v = t_xgetbv(0);
        if (v != x.xcr0) { if (!bad) first_bad = v; bad++; }
    }
    CHECKF(bad == 0, "%d of 100000 XGETBV(0) reads differ (first 0x%llx vs 0x%llx)", bad,
           (unsigned long long)first_bad, (unsigned long long)x.xcr0);
}

TEST(harness_cpu_info_matches_raw) {
    xp_raw x; xp_read(&x);
    const t_cpu_info *c = cpu();
    CHECK_EQ(c->osxsave, x.osxsave);
    CHECK_EQ(c->max_basic, x.maxleaf);
    if (x.has_d) {
        CHECK_EQ(c->xcr0_supported, x.sup);
        CHECK_EQ(c->xss_supported, x.xss);
        CHECK_EQ(c->xsave_size_max, x.d0.ecx);
    }
    if (x.osxsave) {
        CHECK(c->xcr0_valid);
        CHECK_EQ(c->xcr0, x.xcr0);
        CHECK_EQ(c->xsave_size_enabled, x.d0.ebx);
        CHECK_EQ(c->xinuse_valid, x.xg1);
    }
}

/* =============================================================================================
 * 2. XCR0 architectural invariants (each is an XSETBV #GP condition, so no valid XCR0 violates it)
 * =========================================================================================== */
TEST(xcr0_x87_bit0_set) {
    xp_raw x; xp_need_xcr0(&x);
    char b[256]; hex_bits(b, sizeof b, x.xcr0);
    t_info("xcr0", "0x%llx (%s)", (unsigned long long)x.xcr0, b);
    CHECKF(bit(x.xcr0, 0), "XCR0[0] (x87) must always be 1");
}

TEST(xcr0_sse_bit1_set) {
    xp_raw x; xp_need_xcr0(&x);
    /* Not an XSETBV rule, but every x64 Windows enables SSE state; record if it is missing. */
    if (!bit(x.xcr0, 1)) t_note("XCR0[1] (SSE) is 0: XMM state is not XSAVE-managed (unusual; AVX impossible)");
    CHECKF(bit(x.xcr0, 1) || !bit(x.xcr0, 2), "XCR0[2] (AVX) set without XCR0[1] (SSE)");
}

TEST(xcr0_avx_requires_sse) {
    xp_raw x; xp_need_xcr0(&x);
    CHECKF(!bit(x.xcr0, 2) || bit(x.xcr0, 1), "XCR0[2]=1 requires XCR0[1]=1 (XSETBV #GP rule)");
}

TEST(xcr0_mpx_bits_pair) {
    xp_raw x; xp_need_xcr0(&x);
    CHECKF(bit(x.xcr0, 3) == bit(x.xcr0, 4), "XCR0[3] (BNDREGS) != XCR0[4] (BNDCSR) (XSETBV #GP rule)");
    if (x.xcr0 & XP_MPX_BITS) t_note("MPX state enabled in XCR0");
}

TEST(xcr0_avx512_bits_all_or_none) {
    xp_raw x; xp_need_xcr0(&x);
    uint64_t z = x.xcr0 & XP_AVX512_BITS;
    CHECKF(z == 0 || z == XP_AVX512_BITS, "XCR0[7:5] must be 000b or 111b, got %llx", (unsigned long long)(z >> 5));
}

TEST(xcr0_avx512_requires_avx) {
    xp_raw x; xp_need_xcr0(&x);
    CHECKF(!(x.xcr0 & XP_AVX512_BITS) || bit(x.xcr0, 2), "XCR0[7:5] set requires XCR0[2] (AVX)");
    CHECKF(!(x.xcr0 & XP_AVX512_BITS) || bit(x.xcr0, 1), "XCR0[7:5] set requires XCR0[1] (SSE)");
}

TEST(xcr0_amx_bits_pair) {
    xp_raw x; xp_need_xcr0(&x);
    CHECKF(bit(x.xcr0, 17) == bit(x.xcr0, 18), "XCR0[17] (TILECFG) != XCR0[18] (TILEDATA) (XSETBV #GP rule)");
}

TEST(xcr0_pt_bit8_clear) {
    xp_raw x; xp_need_xcr0(&x);
    CHECKF(!bit(x.xcr0, 8), "XCR0[8] (PT) is a supervisor component and reserved in XCR0");
}

TEST(xcr0_bit10_clear) {
    xp_raw x; xp_need_xcr0(&x);
    CHECKF(!bit(x.xcr0, 10), "XCR0[10] (PASID) is a supervisor component and reserved in XCR0");
}

TEST(xcr0_no_supervisor_components) {
    xp_raw x; xp_need_xcr0(&x);
    for (int i = 0; i < 64; i++)
        if (bit(XP_SUP_KNOWN, i)) CHECKF(!bit(x.xcr0, i), "XCR0[%d] (%s) is a supervisor (IA32_XSS) component", i, xstate_comp_name(i));
    if (x.has_d) CHECKF((x.xcr0 & x.xss) == 0, "XCR0 0x%llx shares bits with the IA32_XSS-supported mask 0x%llx",
                        (unsigned long long)x.xcr0, (unsigned long long)x.xss);
}

TEST(xcr0_pkru_bit9) {
    xp_raw x; xp_need_xcr0(&x);
    int pke = ospke(), pku = pku_hw();
    t_info("pkru", "xcr0[9]=%d pku=%d ospke=%d", bit(x.xcr0, 9), pku, pke);
    if (bit(x.xcr0, 9)) CHECKF(bit(x.sup, 9), "XCR0[9] set but PKRU not in CPUID.(0Dh,0) supported mask");
    if (bit(x.xcr0, 9) && !pke) t_note("PKRU state enabled in XCR0 but CR4.PKE (OSPKE) is 0");
}

TEST(xcr0_bit63_and_unknown_bits) {
    xp_raw x; xp_need_xcr0(&x);
    CHECKF(!bit(x.xcr0, 63), "XCR0[63] is reserved (extension bit) and must be 0");
    uint64_t unk = x.xcr0 & ~(XP_USER_KNOWN | XP_SUP_KNOWN);
    if (unk) t_note("XCR0 contains components unknown to this suite: 0x%llx", (unsigned long long)unk);
}

TEST(xcr0_subset_of_cpuid_supported) {
    xp_raw x; xp_need_xcr0(&x);
    REQUIRE(x.has_d);
    t_info("xcr0_supported", "0x%llx", (unsigned long long)x.sup);
    CHECKF((x.xcr0 & ~x.sup) == 0, "XCR0 0x%llx has bits not in CPUID.(0Dh,0) EDX:EAX 0x%llx (XSETBV #GP rule)",
           (unsigned long long)x.xcr0, (unsigned long long)x.sup);
}

TEST(xcr0_avx_enabled_iff_supported_note) {
    xp_raw x; xp_need_xcr0(&x);
    /* An OS may legitimately leave supported components disabled; record what is left out. */
    uint64_t off = x.sup & ~x.xcr0;
    char b[256]; hex_bits(b, sizeof b, off);
    t_info("supported_not_enabled", "0x%llx (%s)", (unsigned long long)off, b);
    if (bit(off, 2)) t_note("AVX state supported by the CPU but not enabled by the OS");
    if (off & XP_AVX512_BITS) t_note("AVX-512 state supported by the CPU but not enabled by the OS");
    if (off & T_XS_AMX_STATE) t_note("AMX state supported by the CPU but not enabled in XCR0");
    CHECK(1);
}

/* =============================================================================================
 * 3. CPUID leaf 0Dh: supported masks, user/supervisor separation, sizes
 * =========================================================================================== */
TEST(cpuid_d0_supported_mask_shape) {
    xp_raw x; xp_need_leaf_d(&x);
    CHECKF(bit(x.sup, 0), "CPUID.(0Dh,0):EAX[0] (x87) must be 1");
    CHECKF(bit(x.sup, 1), "CPUID.(0Dh,0):EAX[1] (SSE) must be 1");
    CHECKF(!bit(x.sup, 2) || bit(x.sup, 1), "AVX supported without SSE");
    CHECKF((x.sup & XP_AVX512_BITS) == 0 || (x.sup & XP_AVX512_BITS) == XP_AVX512_BITS, "AVX-512 components partially supported");
    CHECKF(bit(x.sup, 3) == bit(x.sup, 4), "MPX components partially supported");
    CHECKF(bit(x.sup, 17) == bit(x.sup, 18), "AMX components partially supported");
    for (int i = 0; i < 64; i++)
        if (bit(XP_SUP_KNOWN, i)) CHECKF(!bit(x.sup, i), "supervisor component %d (%s) listed in the XCR0-supported mask", i, xstate_comp_name(i));
    CHECKF(!bit(x.sup, 63), "bit 63 of the XCR0-supported mask is reserved");
}

TEST(cpuid_d1_xss_mask_shape) {
    xp_raw x; xp_need_leaf_d(&x);
    char b[256]; hex_bits(b, sizeof b, x.xss);
    t_info("xss_supported", "0x%llx (%s)", (unsigned long long)x.xss, b);
    for (int i = 0; i < 64; i++)
        if (bit(XP_USER_KNOWN, i)) CHECKF(!bit(x.xss, i), "user component %d (%s) listed in the IA32_XSS-supported mask", i, xstate_comp_name(i));
    if (!bit(x.d1.eax, 3) && x.xss) t_note("IA32_XSS components enumerated but XSAVES is not");
}

TEST(xcr0_xss_masks_disjoint) {
    xp_raw x; xp_need_leaf_d(&x);
    CHECKF((x.sup & x.xss) == 0, "a component is both user (0x%llx) and supervisor (0x%llx)",
           (unsigned long long)x.sup, (unsigned long long)x.xss);
}

TEST(cpuid_d_user_component_flags) {
    xp_raw x; xp_need_leaf_d(&x);
    int n = 0;
    for (int i = 2; i < 63; i++) {
        if (!bit(x.sup, i)) continue;
        t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
        n++;
        CHECKF((r.ecx & 1) == 0, "user component %d (%s): ECX[0] (supervisor) set", i, xstate_comp_name(i));
        CHECKF(r.eax != 0, "user component %d (%s): size 0", i, xstate_comp_name(i));
        CHECKF(r.ebx >= 576, "user component %d (%s): standard offset %u overlaps the legacy area/header", i, xstate_comp_name(i), r.ebx);
        CHECKF(r.ebx + r.eax <= x.d0.ecx, "user component %d (%s): end %u beyond max size %u", i, xstate_comp_name(i), r.ebx + r.eax, x.d0.ecx);
        CHECKF((r.ecx & ~7u) == 0, "user component %d: reserved ECX bits 0x%x", i, r.ecx);
        CHECKF(r.edx == 0, "user component %d: EDX reserved, got 0x%x", i, r.edx);
        t_log("comp %d %s size %u off %u ecx 0x%x", i, xstate_comp_name(i), r.eax, r.ebx, r.ecx);
    }
    t_info("user_components", "%d", n);
}

TEST(cpuid_d_supervisor_component_flags) {
    xp_raw x; xp_need_leaf_d(&x);
    if (!x.xss) SKIP("no IA32_XSS components enumerated");
    for (int i = 0; i < 63; i++) {
        if (!bit(x.xss, i)) continue;
        t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
        CHECKF(r.ecx & 1, "supervisor component %d (%s): ECX[0] must be 1", i, xstate_comp_name(i));
        CHECKF(r.ebx == 0, "supervisor component %d (%s): EBX (standard offset) must be 0, got %u", i, xstate_comp_name(i), r.ebx);
        CHECKF(r.eax != 0, "supervisor component %d (%s): size 0", i, xstate_comp_name(i));
    }
}

TEST(cpuid_d_component_sizes_architectural) {
    xp_raw x; xp_need_leaf_d(&x);
    int n = 0;
    for (int i = 2; i < 63; i++) {
        if (!bit(x.sup | x.xss, i) || !arch_size(i)) continue;
        t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
        n++;
        CHECKF((int)r.eax == arch_size(i), "component %d (%s): size %u, architectural %d", i, xstate_comp_name(i), r.eax, arch_size(i));
        if (i == 2) t_info("avx_offset", "%u", r.ebx);
        if (i == 5) t_info("opmask_offset", "%u", r.ebx);
    }
    if (!n) SKIP("no extended components with an architectural size are supported");
}

TEST(cpuid_d_invalid_subleaves_zero) {
    xp_raw x; xp_need_leaf_d(&x);
    /* SDM CPUID leaf 0Dh: sub-leaf n>=2 is invalid if bit n is clear in both (0,EDX:EAX) and (1,EDX:ECX);
     * an invalid sub-leaf returns 0 in EAX, EBX, ECX and EDX. */
    int n = 0;
    for (int i = 2; i < 64; i++) {
        if (bit(x.sup | x.xss, i)) continue;
        t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
        n++;
        CHECKF(r.eax == 0 && r.ebx == 0 && r.ecx == 0 && r.edx == 0,
               "invalid sub-leaf %d returned %08x %08x %08x %08x", i, r.eax, r.ebx, r.ecx, r.edx);
    }
    t_info("invalid_subleaves", "%d", n);
}

TEST(cpuid_d_enabled_offsets_nonoverlapping) {
    xp_raw x; xp_need_xcr0(&x);
    REQUIRE(x.has_d);
    uint32_t off[64] = { 0 }, sz[64] = { 0 };
    for (int i = 2; i < 63; i++)
        if (bit(x.xcr0, i)) { t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r); off[i] = r.ebx; sz[i] = r.eax; }
    for (int i = 2; i < 63; i++)
        for (int j = i + 1; j < 63; j++)
            if (bit(x.xcr0, i) && bit(x.xcr0, j))
                CHECKF(off[i] + sz[i] <= off[j] || off[j] + sz[j] <= off[i], "components %d [%u,+%u) and %d [%u,+%u) overlap",
                       i, off[i], sz[i], j, off[j], sz[j]);
}

TEST(xsave_size_ebx_matches_xcr0) {
    xp_raw x; xp_need_xcr0(&x);
    REQUIRE(x.has_d);
    uint32_t end = 576;
    for (int i = 2; i < 63; i++)
        if (bit(x.xcr0, i)) { t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r); if (r.ebx + r.eax > end) end = r.ebx + r.eax; }
    t_info("xsave_size_xcr0", "%u", x.d0.ebx);
    CHECKF(x.d0.ebx == end, "CPUID.(0Dh,0):EBX=%u, but the components enabled in XCR0 end at %u", x.d0.ebx, end);
}

TEST(xsave_size_ecx_matches_supported) {
    xp_raw x; xp_need_leaf_d(&x);
    uint32_t end = 576;
    for (int i = 2; i < 63; i++)
        if (bit(x.sup, i)) { t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r); if (r.ebx + r.eax > end) end = r.ebx + r.eax; }
    t_info("xsave_size_max", "%u", x.d0.ecx);
    CHECKF(x.d0.ecx == end, "CPUID.(0Dh,0):ECX=%u, but all supported user components end at %u", x.d0.ecx, end);
    CHECKF(x.d0.ebx <= x.d0.ecx, "EBX (size for XCR0) %u > ECX (max) %u", x.d0.ebx, x.d0.ecx);
    CHECKF(x.d0.ebx >= 576 || !x.osxsave, "EBX %u below the legacy+header size 576", x.d0.ebx);
}

TEST(xsaves_size_covers_xcr0) {
    xp_raw x; xp_need_xcr0(&x);
    REQUIRE(x.has_d);
    if (!bit(x.d1.eax, 3) && !bit(x.d1.eax, 1)) SKIP("neither XSAVEC nor XSAVES: (0Dh,1):EBX not defined");
    uint32_t want = xsave_compacted_size(x.xcr0);
    t_info("xsave_size_compacted_xcr0_xss", "%u (user-only compacted %u)", x.d1.ebx, want);
    if (bit(x.d1.eax, 3))
        CHECKF(x.d1.ebx >= want, "(0Dh,1):EBX=%u < compacted size %u of XCR0 alone (it covers XCR0|IA32_XSS)", x.d1.ebx, want);
    else
        t_note("XSAVEC without XSAVES: (0Dh,1):EBX=%u", x.d1.ebx);
}

TEST(cpuid_d1_capability_bits) {
    xp_raw x; xp_need_leaf_d(&x);
    t_info("d1_eax", "0x%08x xsaveopt=%d xsavec=%d xgetbv1=%d xsaves=%d xfd=%d", x.d1.eax, bit(x.d1.eax, 0), bit(x.d1.eax, 1),
           bit(x.d1.eax, 2), bit(x.d1.eax, 3), bit(x.d1.eax, 4));
    if (bit(x.d1.eax, 3) && !bit(x.d1.eax, 1)) t_note("XSAVES without XSAVEC (unusual)");
    /* XFD: per-component ECX[2] only for components that exist */
    for (int i = 2; i < 63; i++) {
        if (!bit(x.sup, i)) continue;
        t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
        if (r.ecx & 4) {
            t_info("xfd_component", "%d (%s)", i, xstate_comp_name(i));
            CHECKF(bit(x.d1.eax, 4), "component %d advertises XFD support but (0Dh,1):EAX[4] (XFD) is 0", i);
        }
    }
}

/* =============================================================================================
 * 4. Effective XCR0 as seen by XSAVE/XSAVEC (RFBM = XCR0 AND EDX:EAX)
 * =========================================================================================== */
static uint64_t xp_safe_rfbm(void) {
    /* exclude components that support XFD (AMX TILEDATA): an armed XFD makes them fault-prone */
    uint64_t m = ~0ull & ~T_XS_AMX_STATE;
    const t_cpu_info *c = cpu();
    for (int i = 0; i < 64; i++) if (c->comp[i].xfd) m &= ~(1ull << i);
    return m;
}

TEST(xsavec_xcomp_bv_equals_xcr0) {
    xp_raw x; xp_need_xcr0(&x);
    if (!cpu_has(F_XSAVEC)) SKIP("XSAVEC not available");
    uint8_t *area = (uint8_t *)VirtualAlloc(NULL, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(area);
    memset(area, 0xCC, 65536);
    uint64_t rfbm = xp_safe_rfbm();
    t_xsavec(area, rfbm);
    uint64_t xstate_bv, xcomp_bv;
    memcpy(&xstate_bv, area + 512, 8);
    memcpy(&xcomp_bv, area + 520, 8);
    uint64_t want = (1ull << 63) | (x.xcr0 & rfbm);
    t_info("xcomp_bv", "0x%llx xstate_bv 0x%llx", (unsigned long long)xcomp_bv, (unsigned long long)xstate_bv);
    CHECKF(xcomp_bv == want, "XSAVEC XCOMP_BV 0x%llx, want bit63|(XCR0&RFBM) 0x%llx: hardware XCR0 differs from XGETBV",
           (unsigned long long)xcomp_bv, (unsigned long long)want);
    CHECKF((xstate_bv & ~(x.xcr0 & rfbm)) == 0, "XSTATE_BV 0x%llx has bits outside XCR0&RFBM", (unsigned long long)xstate_bv);
    uint32_t sz = xsave_compacted_size(x.xcr0 & rfbm);
    int canary_ok = 1;
    for (uint32_t i = sz; i < sz + 256 && i < 65536; i++) if (area[i] != 0xCC) { canary_ok = 0; break; }
    CHECKF(canary_ok, "XSAVEC wrote beyond the compacted size %u", sz);
    VirtualFree(area, 0, MEM_RELEASE);
}

TEST(xsave_header_within_xcr0) {
    xp_raw x; xp_need_xcr0(&x);
    if (!cpu_has(F_XSAVE)) SKIP("XSAVE not usable");
    uint8_t *area = (uint8_t *)VirtualAlloc(NULL, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(area);
    memset(area, 0, 65536);
    for (uint32_t i = x.d0.ebx; i < x.d0.ebx + 512 && i < 65536; i++) area[i] = 0xCC;
    uint64_t rfbm = xp_safe_rfbm();
    t_xsave(area, rfbm);
    uint64_t xstate_bv; memcpy(&xstate_bv, area + 512, 8);
    t_info("xstate_bv", "0x%llx", (unsigned long long)xstate_bv);
    CHECKF((xstate_bv & ~(x.xcr0 & rfbm)) == 0, "XSAVE XSTATE_BV 0x%llx outside XCR0&RFBM 0x%llx",
           (unsigned long long)xstate_bv, (unsigned long long)(x.xcr0 & rfbm));
    int canary_ok = 1;
    for (uint32_t i = x.d0.ebx; i < x.d0.ebx + 512 && i < 65536; i++) if (area[i] != 0xCC) { canary_ok = 0; break; }
    CHECKF(canary_ok, "XSAVE wrote beyond CPUID.(0Dh,0):EBX=%u (size for the enabled XCR0)", x.d0.ebx);
    VirtualFree(area, 0, MEM_RELEASE);
}

/* =============================================================================================
 * 5. XINUSE (XGETBV with ECX=1): result = XCR0 AND XINUSE; a component not in its initial
 *    configuration has XINUSE[i]=1 (the converse need not hold).
 * =========================================================================================== */
static void xp_need_xg1(xp_raw *x) {
    xp_need_xcr0(x);
    if (!x->xg1) SKIP("CPUID.(0Dh,1):EAX[2] (XGETBV ECX=1) not supported");
}

static uint64_t xinuse_after_xmm_write(void) {
    uint32_t lo, hi;
    __asm__ volatile("pcmpeqd %%xmm0, %%xmm0\n\txgetbv\n\tpxor %%xmm0, %%xmm0" : "=a"(lo), "=d"(hi) : "c"(1) : "xmm0");
    return (uint64_t)hi << 32 | lo;
}
static uint64_t xinuse_after_fld1(void) {
    uint32_t lo, hi;
    __asm__ volatile("fld1\n\txgetbv\n\tfstp %%st(0)" : "=a"(lo), "=d"(hi) : "c"(1) : "st");
    return (uint64_t)hi << 32 | lo;
}
T_TARGET("avx") static T_NOINLINE void xinuse_avx_ymm(uint64_t *with, uint64_t *after) {
    uint32_t lo, hi, lo2, hi2;
    __asm__ volatile("vcmpps $15, %%ymm1, %%ymm1, %%ymm1\n\txgetbv\n\tvzeroupper"
                     : "=a"(lo), "=d"(hi) : "c"(1) : "xmm1");
    __asm__ volatile("xgetbv" : "=a"(lo2), "=d"(hi2) : "c"(1));
    *with = (uint64_t)hi << 32 | lo;
    *after = (uint64_t)hi2 << 32 | lo2;
}
T_TARGET("avx512f") static T_NOINLINE uint64_t xinuse_avx512_opmask(void) {
    uint32_t lo, hi;
    __asm__ volatile("kxnorw %%k1, %%k1, %%k1\n\txgetbv\n\tkxorw %%k1, %%k1, %%k1" : "=a"(lo), "=d"(hi) : "c"(1) : "k1");
    return (uint64_t)hi << 32 | lo;
}
T_TARGET("avx512f") static T_NOINLINE uint64_t xinuse_avx512_zmm_hi256(void) {
    uint32_t lo, hi;
    __asm__ volatile("vpternlogd $0xff, %%zmm1, %%zmm1, %%zmm1\n\txgetbv\n\tvzeroupper" : "=a"(lo), "=d"(hi) : "c"(1) : "xmm1");
    return (uint64_t)hi << 32 | lo;
}
#ifdef T_X64
T_TARGET("avx512f") static T_NOINLINE uint64_t xinuse_avx512_zmm17(void) {
    uint32_t lo, hi;
    __asm__ volatile("vpternlogd $0xff, %%zmm17, %%zmm17, %%zmm17\n\txgetbv\n\tvpxord %%zmm17, %%zmm17, %%zmm17"
                     : "=a"(lo), "=d"(hi) : "c"(1) : "xmm17");
    return (uint64_t)hi << 32 | lo;
}
#endif

TEST(xgetbv1_subset_of_xcr0) {
    xp_raw x; xp_need_xg1(&x);
    uint64_t or_all = 0;
    for (int i = 0; i < 10000; i++) {
        uint64_t v = t_xgetbv(1);
        or_all |= v;
        if (v & ~x.xcr0) { CHECKF(0, "XGETBV(1)=0x%llx has bits outside XCR0 0x%llx", (unsigned long long)v, (unsigned long long)x.xcr0); break; }
    }
    t_info("xinuse_union", "0x%llx", (unsigned long long)or_all);
}

TEST(xgetbv1_sse_inuse_after_xmm_write) {
    xp_raw x; xp_need_xg1(&x);
    if (!bit(x.xcr0, 1)) SKIP("SSE state not in XCR0");
    uint64_t v = xinuse_after_xmm_write();
    CHECKF(bit(v, 1), "XINUSE[1] is 0 while XMM0 holds all-ones (XGETBV(1)=0x%llx)", (unsigned long long)v);
    CHECKF((v & ~x.xcr0) == 0, "XINUSE 0x%llx outside XCR0", (unsigned long long)v);
}

TEST(xgetbv1_x87_inuse_after_fld) {
    xp_raw x; xp_need_xg1(&x);
    uint64_t v = xinuse_after_fld1();
    CHECKF(bit(v, 0), "XINUSE[0] is 0 while ST(0) is valid (XGETBV(1)=0x%llx)", (unsigned long long)v);
}

TEST(xgetbv1_avx_inuse_after_ymm_write) {
    xp_raw x; xp_need_xg1(&x);
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    uint64_t with = 0, after = 0;
    xinuse_avx_ymm(&with, &after);
    t_info("xinuse_avx", "with upper=0x%llx after vzeroupper=0x%llx", (unsigned long long)with, (unsigned long long)after);
    CHECKF(bit(with, 2), "XINUSE[2] is 0 while YMM1[255:128] is all-ones (0x%llx)", (unsigned long long)with);
    CHECKF((with & ~x.xcr0) == 0 && (after & ~x.xcr0) == 0, "XINUSE outside XCR0");
    if (!bit(after, 2)) t_note("XINUSE[2] tracks VZEROUPPER (init optimization observed)");
    else t_note("XINUSE[2] stays 1 after VZEROUPPER (allowed)");
}

TEST(xgetbv1_opmask_inuse) {
    xp_raw x; xp_need_xg1(&x);
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512 not usable");
    uint64_t v = xinuse_avx512_opmask();
    CHECKF(bit(v, 5), "XINUSE[5] is 0 while k1=0xFFFF (0x%llx)", (unsigned long long)v);
}

TEST(xgetbv1_zmm_hi256_inuse) {
    xp_raw x; xp_need_xg1(&x);
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512 not usable");
    uint64_t v = xinuse_avx512_zmm_hi256();
    CHECKF(bit(v, 6), "XINUSE[6] is 0 while ZMM1[511:256] is all-ones (0x%llx)", (unsigned long long)v);
}

TEST(xgetbv1_hi16_zmm_inuse) {
#ifdef T_X64
    xp_raw x; xp_need_xg1(&x);
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512 not usable");
    uint64_t v = xinuse_avx512_zmm17();
    CHECKF(bit(v, 7), "XINUSE[7] is 0 while ZMM17 is all-ones (0x%llx)", (unsigned long long)v);
#else
    SKIP("ZMM16-31 are not addressable in 32-bit mode");
#endif
}

TEST(xgetbv1_pkru_info) {
    xp_raw x; xp_need_xg1(&x);
    if (!ospke()) SKIP("CR4.PKE (OSPKE) not set: RDPKRU not usable");
    uint32_t pkru, d;
    __asm__ volatile(".byte 0x0f,0x01,0xee" : "=a"(pkru), "=d"(d) : "c"(0));   /* RDPKRU */
    uint64_t v = t_xgetbv(1);
    t_info("pkru", "0x%08x xinuse[9]=%d", pkru, bit(v, 9));
    if (pkru != 0 && bit(x.xcr0, 9)) CHECKF(bit(v, 9), "XINUSE[9] is 0 while PKRU=0x%x is not its initial value 0", pkru);
}

TEST(xgetbv1_gp_when_unsupported) {
    xp_raw x; xp_need_xcr0(&x);
    if (x.xg1) SKIP("XGETBV ECX=1 is supported (the #GP rule cannot be exercised)");
    expect_gp_fault("xgetbv_index", "1", 0);
    uint64_t un = 0;
    CHECK(child_u64(&g_cr, "UNCHANGED=", &un) && un == 1);
}

TEST(xgetbv_invalid_index_faults) {
    xp_raw x; xp_need_xcr0(&x);
    expect_gp_fault("xgetbv_index", "ffffffff", 0);
    uint64_t un = 0;
    CHECKF(child_u64(&g_cr, "UNCHANGED=", &un) && un == 1, "a faulting XGETBV must leave EDX:EAX unchanged");
}

TEST(xgetbv_index2_info) {
    xp_raw x; xp_need_xcr0(&x);
    /* No XCR2 is defined today, so XGETBV(2) should #GP; recorded only (future XCRs may appear). */
    REQUIRE(t_child_run("xgetbv_index", "2", 5000, &g_cr));
    REQUIRE(!g_cr.timed_out);
    uint64_t f = 0, code = 0, v = 0;
    child_u64(&g_cr, "FAULTED=", &f); child_u64(&g_cr, "CODE=", &code); child_u64(&g_cr, "VALUE=", &v);
    t_info("xgetbv2", "faulted=%llu code=0x%08llx value=0x%llx", (unsigned long long)f, (unsigned long long)code, (unsigned long long)v);
    if (!f) t_note("XGETBV(2) did not fault (returned 0x%llx)", (unsigned long long)v);
    CHECK(g_cr.exc_code == 0);
}

/* =============================================================================================
 * 6. XSETBV is privileged (#GP(0) at CPL>0)
 * =========================================================================================== */
TEST(xsetbv_same_value_faults) {
    xp_raw x; xp_need_xcr0(&x);
    expect_gp_fault("xsetbv_same", "", 1);
}

TEST(xsetbv_invalid_value_faults) {
    xp_raw x; xp_need_xcr0(&x);
    expect_gp_fault("xsetbv_invalid", "", 1);
}

TEST(xsetbv_xcr1_faults) {
    xp_raw x; xp_need_xcr0(&x);
    expect_gp_fault("xsetbv_xcr1", "", 1);
}

TEST(xcr0_unchanged_after_xsetbv_attempts) {
    xp_raw x; xp_need_xcr0(&x);
    /* the children above ran on arbitrary CPUs; XCR0 is per logical processor */
    REQUIRE(t_child_run("xsetbv_same", "", 5000, &g_cr));
    xp_raw y; xp_read(&y);
    CHECK_EQ(y.xcr0, x.xcr0);
    CHECK_EQ(y.d0.ebx, x.d0.ebx);
    REQUIRE(t_child_run("report", "", 5000, &g_cr));
    uint64_t v = 0;
    CHECK(child_u64(&g_cr, "XCR0=", &v));
    CHECK_EQ(v, x.xcr0);
}

/* =============================================================================================
 * 7. XCR0 stability: CPUs, threads, time, processes, architectures
 * =========================================================================================== */
typedef struct xp_tres {
    int cpu, ran;
    uint32_t l1ecx, d0ebx, d0eax, d0edx;
    uint64_t xcr0, xinuse;
    int xinuse_ok;
    DWORD procnum;
    long mismatches;
    uint64_t ref;
} xp_tres;

static DWORD WINAPI xp_cpu_worker(void *p) {
    xp_tres *t = (xp_tres *)p;
    t_cpuid_regs r;
    cpuid_ex(1, 0, &r);
    t->l1ecx = r.ecx;
    if (cpuid_max_basic() >= 0xD) { cpuid_ex(0xD, 0, &r); t->d0ebx = r.ebx; t->d0eax = r.eax; t->d0edx = r.edx; }
    if ((t->l1ecx >> 27) & 1) {
        t->xcr0 = t_xgetbv(0);
        if (cpu()->xinuse_valid) { t->xinuse = t_xgetbv(1); t->xinuse_ok = 1; }
    }
    t->procnum = GetCurrentProcessorNumber();
    t->ran = 1;
    return 0;
}

TEST(xcr0_identical_on_every_cpu) {
    xp_raw x; xp_need_xcr0(&x);
    DWORD_PTR pmask = 0, smask = 0;
    REQUIRE(GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask));
    static xp_tres res[64];
    HANDLE h[64]; int n = 0;
    memset(res, 0, sizeof res);
    for (int c = 0; c < (int)(8 * sizeof(DWORD_PTR)) && n < 64; c++) {
        if (!((pmask >> c) & 1)) continue;
        res[n].cpu = c;
        h[n] = CreateThread(NULL, 64 * 1024, xp_cpu_worker, &res[n], CREATE_SUSPENDED, NULL);
        if (!h[n]) break;
        if (!SetThreadAffinityMask(h[n], (DWORD_PTR)1 << c)) t_note("SetThreadAffinityMask(cpu %d) failed: %lu", c, GetLastError());
        ResumeThread(h[n]);
        n++;
    }
    if (!n) SKIP("could not create any thread");
    for (int i = 0; i < n; i++) { CHECK(WaitForSingleObject(h[i], 5000) == WAIT_OBJECT_0); CloseHandle(h[i]); }
    int off_cpu = 0;
    for (int i = 0; i < n; i++) {
        xp_tres *t = &res[i];
        if (!t->ran) { CHECKF(0, "worker for cpu %d did not run", t->cpu); continue; }
        CHECKF(((t->l1ecx >> 27) & 1) == 1, "cpu %d: OSXSAVE=0 while the main thread sees 1", t->cpu);
        CHECKF(t->xcr0 == x.xcr0, "cpu %d: XCR0 0x%llx differs from 0x%llx", t->cpu, (unsigned long long)t->xcr0, (unsigned long long)x.xcr0);
        CHECKF(t->d0ebx == x.d0.ebx, "cpu %d: CPUID.(0Dh,0):EBX %u differs from %u", t->cpu, t->d0ebx, x.d0.ebx);
        CHECKF(t->d0eax == x.d0.eax && t->d0edx == x.d0.edx, "cpu %d: XCR0-supported mask differs", t->cpu);
        if (t->xinuse_ok) CHECKF((t->xinuse & ~t->xcr0) == 0, "cpu %d: XINUSE outside XCR0", t->cpu);
        if ((int)t->procnum != t->cpu) off_cpu++;
    }
    t_info("cpus_checked", "%d", n);
    if (off_cpu) t_note("%d of %d pinned workers reported another processor number", off_cpu, n);
}

static DWORD WINAPI xp_churn_worker(void *p) {
    xp_tres *t = (xp_tres *)p;
    for (int i = 0; i < 2000; i++) {
        uint64_t v = t_xgetbv(0);
        if (v != t->ref) { if (!t->mismatches) t->xcr0 = v; t->mismatches++; }
        if ((i & 63) == 0) SwitchToThread();
    }
    t->ran = 1;
    return 0;
}

TEST(xcr0_constant_across_many_threads) {
    xp_raw x; xp_need_xcr0(&x);
    enum { N = 128 };
    static xp_tres res[N];
    static HANDLE h[N];
    memset(res, 0, sizeof res);
    int n = 0;
    for (; n < N; n++) {
        res[n].ref = x.xcr0;
        h[n] = CreateThread(NULL, 64 * 1024, xp_churn_worker, &res[n], 0, NULL);
        if (!h[n]) break;
    }
    if (n < 2) { for (int i = 0; i < n; i++) { WaitForSingleObject(h[i], 5000); CloseHandle(h[i]); } SKIP("could not create threads"); }
    for (int i = 0; i < n; i++) { CHECK(WaitForSingleObject(h[i], 10000) == WAIT_OBJECT_0); CloseHandle(h[i]); }
    long bad = 0;
    for (int i = 0; i < n; i++) {
        CHECKF(res[i].ran, "thread %d did not finish", i);
        if (res[i].mismatches) { bad++; CHECKF(0, "thread %d saw XCR0 0x%llx (%ld times)", i, (unsigned long long)res[i].xcr0, res[i].mismatches); }
    }
    t_info("threads", "%d", n);
    CHECK_EQ(bad, 0);
}

TEST(xcr0_constant_over_time) {
    xp_raw x; xp_need_xcr0(&x);
    int bad = 0;
    double t0 = t_now_ms();
    for (int i = 0; i < 150; i++) {
        if (i % 3 == 0) Sleep(1); else if (i % 3 == 1) Sleep(0); else SwitchToThread();
        if (t_xgetbv(0) != x.xcr0) bad++;
        t_cpuid_regs r; cpuid_ex(1, 0, &r);
        if (!((r.ecx >> 27) & 1)) bad++;
    }
    if (cpu_has(F_AVX)) {   /* using AVX state does not change XCR0 */
        uint64_t a, b; xinuse_avx_ymm(&a, &b);
        Sleep(5);
        if (t_xgetbv(0) != x.xcr0) bad++;
    }
    t_info("window_ms", "%.0f", t_now_ms() - t0);
    CHECKF(bad == 0, "%d reads disagreed with XCR0 0x%llx", bad, (unsigned long long)x.xcr0);
    xp_raw y; xp_read(&y);
    CHECK_EQ(y.d0.ebx, x.d0.ebx);
    CHECK_EQ(y.sup, x.sup);
}

static volatile LONG g_ex_seen;
static LONG CALLBACK xp_sw_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != 0xE0C0FFEEu) return EXCEPTION_CONTINUE_SEARCH;
    g_ex_seen++;
    return EXCEPTION_CONTINUE_EXECUTION;
}
TEST(xcr0_unchanged_after_exception_dispatch) {
    xp_raw x; xp_need_xcr0(&x);
    PVOID h = AddVectoredExceptionHandler(1, xp_sw_veh);
    REQUIRE(h);
    g_ex_seen = 0;
    for (int i = 0; i < 20; i++) RaiseException(0xE0C0FFEEu, 0, 0, NULL);
    RemoveVectoredExceptionHandler(h);
    CHECK_EQ(g_ex_seen, 20);
    CHECK_EQ(t_xgetbv(0), x.xcr0);
    if (x.xg1) CHECK((t_xgetbv(1) & ~x.xcr0) == 0);
}

TEST(xcr0_same_in_child_process) {
    xp_raw x; xp_read(&x);
    REQUIRE(t_child_run("report", "", 5000, &g_cr));
    REQUIRE(!g_cr.timed_out);
    CHECK_EQ(g_cr.exc_code, 0);
    uint64_t osx = 0, v = 0, sup = 0, ebx = 0, api = 0;
    CHECK(child_u64(&g_cr, "OSXSAVE=", &osx)); CHECK_EQ(osx, x.osxsave);
    CHECK(child_u64(&g_cr, "XCR0=", &v));      CHECK_EQ(v, x.xcr0);
    CHECK(child_u64(&g_cr, "XCR0SUP=", &sup)); CHECK_EQ(sup, x.sup);
    CHECK(child_u64(&g_cr, "D0EBX=", &ebx));   CHECK_EQ(ebx, x.d0.ebx);
    pfn_GetEnabledXStateFeatures gef = api_gef();
    if (gef) { CHECK(child_u64(&g_cr, "API=", &api)); CHECK_EQ(api, gef()); }
}

/* Run another exe with a timeout and capture stdout into r->out (minimal, for the other-arch build). */
static int run_other_exe(const char *path, const char *args, DWORD tmo, t_child_result *r) {
    memset(r, 0, sizeof *r);
    char tmp[MAX_PATH]; t_tmpfile(tmp, sizeof tmp, "xcr0_other.txt");
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE f = CreateFileA(tmp, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, NULL);
    if (f == INVALID_HANDLE_VALUE) return 0;
    char cmd[1024]; snprintf(cmd, sizeof cmd, "\"%s\" %s", path, args);
    STARTUPINFOA si; PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si); si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = f; si.hStdError = f; si.hStdInput = NULL;
    memset(&pi, 0, sizeof pi);
    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    CloseHandle(f);
    if (!ok) { r->spawn_error = GetLastError(); DeleteFileA(tmp); return 0; }
    r->started = 1;
    if (WaitForSingleObject(pi.hProcess, tmo) != WAIT_OBJECT_0) { r->timed_out = 1; TerminateProcess(pi.hProcess, 1); WaitForSingleObject(pi.hProcess, 2000); }
    GetExitCodeProcess(pi.hProcess, &r->exit_code);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    FILE *fp = fopen(tmp, "rb");
    if (fp) { r->out_len = fread(r->out, 1, sizeof r->out - 1, fp); r->out[r->out_len] = 0; fclose(fp); }
    DeleteFileA(tmp);
    return 1;
}

TEST(xcr0_same_in_other_architecture_process) {
    xp_raw x; xp_read(&x);
    const t_env *e = t_env_get();
    t_info("process", "arch=%s wow64=%d native_machine=0x%x", e->process_arch, e->wow64, e->native_machine);
    char other[MAX_PATH];
    snprintf(other, sizeof other, "%s", t_self_path());
#ifdef T_X64
    const char *mine = "\\x64\\", *theirs = "\\x86\\";
#else
    const char *mine = "\\x86\\", *theirs = "\\x64\\";
#endif
    char *p = NULL;
    for (char *q = other; (q = strstr(q, mine)) != NULL; q++) p = q;
    if (!p) SKIP("the other-architecture build is not next to this exe (no %s directory in the path)", mine);
    memcpy(p, theirs, strlen(theirs));
    if (GetFileAttributesA(other) == INVALID_FILE_ATTRIBUTES) SKIP("other-architecture build not found: %s", other);
#ifndef T_X64
    if (!e->wow64) SKIP("32-bit OS: a 64-bit process cannot run here");
#endif
    static t_child_result r;
    if (!run_other_exe(other, "--child=report", 10000, &r)) SKIP("could not start %s (error %lu)", other, r.spawn_error);
    REQUIRE(!r.timed_out);
    CHECK_EQ(r.exit_code, 0);
    char arch[32] = "";
    t_child_find(&r, "ARCH=", arch, sizeof arch);
    t_info("other_arch", "%s", arch);
    uint64_t osx = 0, v = 0, sup = 0, ebx = 0, api = 0, pres = 0;
    CHECK(child_u64(&r, "OSXSAVE=", &osx)); CHECK_EQ(osx, x.osxsave);
    CHECK(child_u64(&r, "XCR0=", &v));
    CHECKF(v == x.xcr0, "XCR0 0x%llx in the %s process, 0x%llx in this %s process", (unsigned long long)v, arch,
           (unsigned long long)x.xcr0, T_ARCH);
    CHECK(child_u64(&r, "XCR0SUP=", &sup)); CHECK_EQ(sup, x.sup);
    CHECK(child_u64(&r, "D0EBX=", &ebx));   CHECK_EQ(ebx, x.d0.ebx);
    pfn_GetEnabledXStateFeatures gef = api_gef();
    if (gef && child_u64(&r, "API_PRESENT=", &pres) && pres) {
        CHECK(child_u64(&r, "API=", &api));
        if (api != gef()) t_note("GetEnabledXStateFeatures differs: 0x%llx (%s) vs 0x%llx (%s)", (unsigned long long)api, arch,
                                 (unsigned long long)gef(), T_ARCH);
        /* the user-state part must agree: both processes run on the same XCR0 */
        CHECK_EQ(api & x.sup, gef() & x.sup);
    }
}

/* =============================================================================================
 * 8. Eligibility rules (CPUID feature bit AND OSXSAVE AND XCR0 component bits)
 * =========================================================================================== */
static int ymm_ok(const xp_raw *x) { return x->osxsave && (x->xcr0 & 6) == 6; }
static int zmm_ok(const xp_raw *x) { return x->osxsave && (x->xcr0 & 0xE6) == 0xE6; }
static int amx_ok(const xp_raw *x) { return x->osxsave && (x->xcr0 & T_XS_AMX_STATE) == T_XS_AMX_STATE; }

TEST(avx_eligibility_rule) {
    xp_raw x; xp_read(&x);
    int want = x.avx_hw && ymm_ok(&x);
    t_info("avx", "hw=%d osxsave=%d xcr0[2:1]=%d%d -> %d", x.avx_hw, x.osxsave, bit(x.xcr0, 2), bit(x.xcr0, 1), want);
    CHECK_EQ(cpu_has(F_AVX), want);
    CHECK_EQ(cpu_ymm_state_enabled(), ymm_ok(&x));
    if (x.avx_hw && !want) t_note("AVX advertised by CPUID but the OS has not enabled YMM state");
}

TEST(avx2_eligibility_rule) {
    xp_raw x; xp_read(&x);
    int hw = leaf7_bit(0, CPUID_EBX, 5);
    CHECK_EQ(cpu_hw(F_AVX2), hw);
    CHECK_EQ(cpu_has(F_AVX2), hw && ymm_ok(&x));
    if (hw && !x.avx_hw) t_note("AVX2 advertised without AVX (inconsistent CPUID, e.g. masked by a hypervisor)");
}

TEST(fma_f16c_eligibility_rule) {
    xp_raw x; xp_read(&x);
    int fma = (x.l1.ecx >> 12) & 1, f16c = (x.l1.ecx >> 29) & 1;
    CHECK_EQ(cpu_has(F_FMA), fma && ymm_ok(&x));
    CHECK_EQ(cpu_has(F_F16C), f16c && ymm_ok(&x));
    if ((fma || f16c) && !x.avx_hw) t_note("FMA/F16C advertised without AVX");
}

TEST(avx512_eligibility_rule) {
    xp_raw x; xp_read(&x);
    int f = leaf7_bit(0, CPUID_EBX, 16);
    int want = f && zmm_ok(&x);
    t_info("avx512f", "hw=%d xcr0[7:5]=%d%d%d xcr0[2:1]=%d%d -> %d", f, bit(x.xcr0, 7), bit(x.xcr0, 6), bit(x.xcr0, 5),
           bit(x.xcr0, 2), bit(x.xcr0, 1), want);
    CHECK_EQ(cpu_has(F_AVX512F), want);
    CHECK_EQ(cpu_zmm_state_enabled(), zmm_ok(&x));
    static const t_feature zf[] = { F_AVX512DQ, F_AVX512IFMA, F_AVX512CD, F_AVX512BW, F_AVX512VL, F_AVX512VBMI,
                                    F_AVX512VBMI2, F_AVX512VNNI, F_AVX512BITALG, F_AVX512VPOPCNTDQ, F_AVX512FP16, F_AVX512BF16 };
    for (size_t i = 0; i < T_COUNTOF(zf); i++)
        CHECKF(cpu_has(zf[i]) == (cpu_hw(zf[i]) && zmm_ok(&x)), "%s: usable=%d hw=%d zmm_state=%d", cpu_feature_name(zf[i]),
               cpu_has(zf[i]), cpu_hw(zf[i]), zmm_ok(&x));
    if (f && !zmm_ok(&x)) t_note("AVX-512F advertised but ZMM state not enabled in XCR0");
    if (f && !leaf7_bit(0, CPUID_EBX, 5)) t_note("AVX-512F advertised without AVX2");
}

TEST(amx_eligibility_rule) {
    xp_raw x; xp_read(&x);
    int tile = leaf7_bit(0, CPUID_EDX, 24);
    CHECK_EQ(cpu_has(F_AMX_TILE), tile && amx_ok(&x));
    CHECK_EQ(cpu_has(F_AMX_INT8), cpu_hw(F_AMX_INT8) && amx_ok(&x));
    CHECK_EQ(cpu_has(F_AMX_BF16), cpu_hw(F_AMX_BF16) && amx_ok(&x));
    if (!tile && !(x.sup & T_XS_AMX_STATE)) SKIP("no AMX on this CPU (eligibility rule checked, nothing to execute)");
    t_info("amx", "hw=%d xcr0[18:17]=%d%d", tile, bit(x.xcr0, 18), bit(x.xcr0, 17));
    if (tile && amx_ok(&x))
        t_note("AMX enabled in XCR0; Windows additionally requires EnableProcessOptionalXStateFeatures (XFD) before use");
}

TEST(feature_gate_table_consistency) {
    xp_raw x; xp_read(&x);
    int pke = ospke(), n = 0;
    for (int f = 0; f < F_COUNT; f++) {
        const char *g = cpu_feature_gate_name((t_feature)f);
        int hw = cpu_hw((t_feature)f), want;
        /* only the XSTATE-dependent gates are this exe's business ("none" features may carry
         * process-mode rules, e.g. LM is not usable in a 32-bit process) */
        if (!strcmp(g, "none")) continue;
        else if (!strcmp(g, "osxsave")) want = hw && x.osxsave;
        else if (!strcmp(g, "ymm")) want = hw && ymm_ok(&x);
        else if (!strcmp(g, "zmm")) want = hw && zmm_ok(&x);
        else if (!strcmp(g, "amx")) want = hw && amx_ok(&x);
        else if (!strcmp(g, "apx")) want = hw && x.osxsave && bit(x.xcr0, 19);
        else if (!strcmp(g, "pku")) want = hw && pke;
        else want = 0;   /* priv */
        CHECKF(cpu_has((t_feature)f) == want, "%s (gate %s): usable=%d, rule says %d", cpu_feature_name((t_feature)f), g,
               cpu_has((t_feature)f), want);
        n++;
    }
    t_info("features", "%d", n);
}

TEST(pku_ospke_consistency) {
    int pku = pku_hw(), pke = ospke();
    t_info("pku_ospke", "%d/%d", pku, pke);
    CHECKF(!pke || pku, "CR4.PKE (OSPKE) set but PKU not supported");
}

/* --- execution when eligible ---------------------------------------------------------------- */
T_TARGET("avx") static T_NOINLINE int exec_avx_ymm(void) {
    float a[8] = { 1, 2, 3, 4, 5, 6, 7, 8 }, b[8] = { .5f, .5f, .5f, .5f, .5f, .5f, .5f, .5f }, o[8];
    __asm__ volatile("vmovups %1, %%ymm0\n\tvaddps %2, %%ymm0, %%ymm0\n\tvmovups %%ymm0, %0\n\tvzeroupper"
                     : "=m"(o) : "m"(a), "m"(b) : "xmm0");
    for (int i = 0; i < 8; i++) if (o[i] != (float)i + 1.5f) return 0;
    return 1;
}
T_TARGET("avx2") static T_NOINLINE int exec_avx2_ymm(void) {
    int32_t a[8] = { 1, 2, 3, 4, 5, 6, 7, 8 }, b[8] = { 10, 20, 30, 40, 50, 60, 70, 80 }, o[8];
    __asm__ volatile("vmovdqu %1, %%ymm0\n\tvpaddd %2, %%ymm0, %%ymm0\n\tvmovdqu %%ymm0, %0\n\tvzeroupper"
                     : "=m"(o) : "m"(a), "m"(b) : "xmm0");
    for (int i = 0; i < 8; i++) if (o[i] != 11 * (i + 1)) return 0;
    return 1;
}
T_TARGET("avx,fma") static T_NOINLINE int exec_fma_ymm(void) {
    float a[8] = { 1, 2, 3, 4, 5, 6, 7, 8 }, b[8] = { 2, 2, 2, 2, 2, 2, 2, 2 }, c[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    __asm__ volatile("vmovups %0, %%ymm0\n\tvmovups %1, %%ymm1\n\tvfmadd231ps %2, %%ymm1, %%ymm0\n\tvmovups %%ymm0, %0\n\tvzeroupper"
                     : "+m"(c) : "m"(a), "m"(b) : "xmm0", "xmm1");
    for (int i = 0; i < 8; i++) if (c[i] != 1.0f + 2.0f * (float)(i + 1)) return 0;
    return 1;
}
T_TARGET("avx512f") static T_NOINLINE int exec_avx512_zmm(void) {
    int32_t a[16], b[16], o[16];
    for (int i = 0; i < 16; i++) { a[i] = i; b[i] = 100 * i; }
    __asm__ volatile("vmovdqu32 %1, %%zmm0\n\tvpaddd %2, %%zmm0, %%zmm0\n\tvmovdqu32 %%zmm0, %0\n\tvzeroupper"
                     : "=m"(o) : "m"(a), "m"(b) : "xmm0");
    for (int i = 0; i < 16; i++) if (o[i] != 101 * i) return 0;
    return 1;
}

TEST(avx_executes_when_eligible) {
    if (!cpu_has(F_AVX)) SKIP("AVX not eligible");
    CHECKF(exec_avx_ymm(), "VADDPS ymm produced a wrong result");
}
TEST(avx2_executes_when_eligible) {
    if (!cpu_has(F_AVX2)) SKIP("AVX2 not eligible");
    CHECKF(exec_avx2_ymm(), "VPADDD ymm produced a wrong result");
}
TEST(fma_executes_when_eligible) {
    if (!cpu_has(F_FMA) || !cpu_has(F_AVX)) SKIP("FMA not eligible");
    CHECKF(exec_fma_ymm(), "VFMADD231PS ymm produced a wrong result");
}
TEST(avx512_executes_when_eligible) {
    if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not eligible");
    CHECKF(exec_avx512_zmm(), "VPADDD zmm produced a wrong result");
}

TEST(avx_ud_when_ymm_state_disabled) {
    xp_raw x; xp_read(&x);
    if (!x.avx_hw) SKIP("AVX not advertised");
    if (ymm_ok(&x)) SKIP("YMM state enabled: the disabled-state #UD cannot be exercised");
    REQUIRE(t_child_run("avx_ymm_disabled", "", 5000, &g_cr));
    REQUIRE(!g_cr.timed_out);
    CHECKF(!strstr(g_cr.out, "SURVIVED=1"), "VEX ymm instruction executed although XCR0[2:1] != 11b");
    CHECKF(g_cr.exc_code == XP_STATUS_ILL, "expected STATUS_ILLEGAL_INSTRUCTION, got 0x%08lx", g_cr.exc_code);
}

TEST(avx512_ud_when_zmm_state_disabled) {
    xp_raw x; xp_read(&x);
    if (!leaf7_bit(0, CPUID_EBX, 16)) SKIP("AVX-512F not advertised");
    if (zmm_ok(&x)) SKIP("ZMM state enabled: the disabled-state #UD cannot be exercised");
    REQUIRE(t_child_run("avx512_zmm_disabled", "", 5000, &g_cr));
    REQUIRE(!g_cr.timed_out);
    CHECKF(!strstr(g_cr.out, "SURVIVED=1"), "EVEX instruction executed although XCR0[7:5,2:1] are not all set");
    CHECKF(g_cr.exc_code == XP_STATUS_ILL, "expected STATUS_ILLEGAL_INSTRUCTION, got 0x%08lx", g_cr.exc_code);
}

/* =============================================================================================
 * 9. Windows reporting APIs vs XCR0
 * =========================================================================================== */
TEST(api_get_enabled_xstate_features_present) {
    pfn_GetEnabledXStateFeatures gef = api_gef();
    if (!gef) SKIP("GetEnabledXStateFeatures not exported (pre-Windows 7 SP1)");
    uint64_t v = gef();
    char b[256]; hex_bits(b, sizeof b, v);
    t_info("api_enabled", "0x%llx (%s)", (unsigned long long)v, b);
    CHECKF((v & 3) == 3, "x87 and SSE (bits 0,1) are always reported enabled (got 0x%llx)", (unsigned long long)v);
    CHECK_EQ(v, t_env_get()->api_xstate_features);
    CHECK_EQ(gef(), v);   /* stable */
}

TEST(api_user_bits_subset_of_xcr0) {
    xp_raw x; xp_read(&x);
    pfn_GetEnabledXStateFeatures gef = api_gef();
    if (!gef) SKIP("GetEnabledXStateFeatures not exported");
    uint64_t api = gef();
    uint64_t eff = x.osxsave ? x.xcr0 : 3;               /* without OSXSAVE only legacy x87/SSE exist */
    uint64_t userish = x.has_d ? (x.sup | XP_USER_KNOWN) : XP_USER_KNOWN;
    uint64_t claim = api & userish & ~x.xss;
    CHECKF((claim & ~eff) == 0, "API reports user state 0x%llx that XCR0 0x%llx does not enable: apps would #UD",
           (unsigned long long)(claim & ~eff), (unsigned long long)eff);
}

TEST(api_reports_xcr0_documented_bits) {
    xp_raw x; xp_need_xcr0(&x);
    pfn_GetEnabledXStateFeatures gef = api_gef();
    if (!gef) SKIP("GetEnabledXStateFeatures not exported");
    uint64_t api = gef();
    /* XSTATE_LEGACY_FLOATING_POINT(0), XSTATE_LEGACY_SSE(1), XSTATE_AVX(2), XSTATE_AVX512_KMASK(5),
     * XSTATE_AVX512_ZMM_H(6), XSTATE_AVX512_ZMM(7): when XCR0 enables them the API must report them. */
    static const int doc[] = { 0, 1, 2, 5, 6, 7 };
    for (size_t i = 0; i < T_COUNTOF(doc); i++)
        if (bit(x.xcr0, doc[i]))
            CHECKF(bit(api, doc[i]), "XCR0[%d] (%s) enabled but GetEnabledXStateFeatures 0x%llx omits it", doc[i],
                   xstate_comp_name(doc[i]), (unsigned long long)api);
    uint64_t other = x.xcr0 & ~api & ~0xE7ull;
    if (other) { char b[256]; hex_bits(b, sizeof b, other); t_note("XCR0 components not reported by the API: %s", b); }
    uint64_t sup = api & ~x.xcr0;
    if (sup) { char b[256]; hex_bits(b, sizeof b, sup); t_note("API reports components outside XCR0 (user-visible supervisor?): %s", b); }
}

TEST(api_extra_bits_are_supervisor) {
    xp_raw x; xp_need_xcr0(&x);
    pfn_GetEnabledXStateFeatures gef = api_gef();
    if (!gef) SKIP("GetEnabledXStateFeatures not exported");
    uint64_t extra = gef() & ~x.xcr0;
    if (!extra) { CHECK(1); return; }
    /* RtlGetEnabledExtendedFeatures ORs EnabledUserVisibleSupervisorFeatures: such bits must be real
     * IA32_XSS components (e.g. CET_U), never invented user components. */
    CHECKF((extra & ~x.xss) == 0, "API bits 0x%llx are neither in XCR0 nor IA32_XSS-supported 0x%llx",
           (unsigned long long)(extra & ~x.xss), (unsigned long long)x.xss);
}

TEST(api_rtl_enabled_extended_features_mask_semantics) {
    pfn_RtlGetEnabledExtendedFeatures rtl = api_rtl();
    if (!rtl) SKIP("ntdll!RtlGetEnabledExtendedFeatures not exported");
    uint64_t all = rtl(~0ull);
    t_info("rtl_all", "0x%llx", (unsigned long long)all);
    CHECK_EQ(rtl(0), 0);
    for (int i = 0; i < 64; i++) {
        uint64_t m = 1ull << i, v = rtl(m);
        CHECKF(v == (all & m), "RtlGetEnabledExtendedFeatures(1<<%d)=0x%llx, want 0x%llx", i, (unsigned long long)v,
               (unsigned long long)(all & m));
    }
    uint64_t m2 = 0x5555555555555555ull;
    CHECK_EQ(rtl(m2), all & m2);
    CHECK_EQ(rtl(~m2), all & ~m2);
}

TEST(api_kernelbase_matches_rtl) {
    pfn_RtlGetEnabledExtendedFeatures rtl = api_rtl();
    pfn_GetEnabledXStateFeatures gef = api_gef();
    if (!rtl || !gef) SKIP("GetEnabledXStateFeatures or RtlGetEnabledExtendedFeatures missing");
    uint64_t r = rtl(~0ull), g = gef();
    uint64_t expect = r ? r : 3;
    if (g != expect) t_note("GetEnabledXStateFeatures 0x%llx != RtlGetEnabledExtendedFeatures(~0) 0x%llx", (unsigned long long)g,
                            (unsigned long long)r);
    /* user-state part must agree whatever the implementation */
    xp_raw x; xp_read(&x);
    uint64_t user = x.has_d ? x.sup : 3;
    CHECK_EQ(g & user, expect & user);
}

TEST(api_thread_enabled_xstate_features) {
    pfn_GetThreadEnabledXStateFeatures gt = api_gtef();
    if (!gt) SKIP("GetThreadEnabledXStateFeatures not exported (Windows 11 API)");
    pfn_GetEnabledXStateFeatures gef = api_gef();
    uint64_t t = gt();
    t_info("thread_enabled", "0x%llx", (unsigned long long)t);
    CHECKF((t & 3) == 3, "x87/SSE must be enabled for every thread (0x%llx)", (unsigned long long)t);
    if (gef) CHECKF((t & ~gef()) == 0, "thread-enabled 0x%llx not a subset of GetEnabledXStateFeatures 0x%llx",
                    (unsigned long long)t, (unsigned long long)gef());
    xp_raw x; xp_read(&x);
    if (x.osxsave) CHECKF((t & ~x.xss & ~x.xcr0) == 0, "thread-enabled user bits 0x%llx outside XCR0 0x%llx",
                          (unsigned long long)(t & ~x.xss), (unsigned long long)x.xcr0);
    if (x.osxsave && (x.xcr0 & T_XS_AMX_STATE) && !(t & T_XS_AMX_STATE))
        t_note("AMX in XCR0 but not enabled for this thread (optional feature not requested)");
}

TEST(pf_xsave_enabled_matches_osxsave) {
    xp_raw x; xp_read(&x);
    int pf = IsProcessorFeaturePresent(PF_XSAVE_ENABLED) ? 1 : 0;
    t_info("pf_xsave_enabled", "%d osxsave=%d", pf, x.osxsave);
    CHECKF(!pf || x.osxsave, "PF_XSAVE_ENABLED reported but CPUID.1:ECX.OSXSAVE=0");
    CHECKF(pf || !x.osxsave, "OSXSAVE=1 but IsProcessorFeaturePresent(PF_XSAVE_ENABLED) is FALSE");
}

TEST(pf_avx_family_vs_xcr0) {
    xp_raw x; xp_read(&x);
    int p39 = IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE) ? 1 : 0;
    int p40 = IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) ? 1 : 0;
    int p41 = IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE) ? 1 : 0;
    t_info("pf_avx_avx2_avx512f", "%d/%d/%d", p39, p40, p41);
    /* claiming a feature whose state the OS has not enabled is a contract violation (apps would #UD) */
    CHECKF(!p39 || cpu_has(F_AVX), "PF_AVX_INSTRUCTIONS_AVAILABLE but AVX not usable (XCR0 0x%llx)", (unsigned long long)x.xcr0);
    CHECKF(!p40 || cpu_has(F_AVX2), "PF_AVX2_INSTRUCTIONS_AVAILABLE but AVX2 not usable");
    CHECKF(!p41 || cpu_has(F_AVX512F), "PF_AVX512F_INSTRUCTIONS_AVAILABLE but AVX-512F not usable");
    /* older Windows builds do not know these constants: absence is only recorded */
    if (cpu_has(F_AVX) && !p39) t_note("AVX usable but PF_AVX_INSTRUCTIONS_AVAILABLE not reported");
    if (cpu_has(F_AVX2) && !p40) t_note("AVX2 usable but PF_AVX2_INSTRUCTIONS_AVAILABLE not reported");
    if (cpu_has(F_AVX512F) && !p41) t_note("AVX-512F usable but PF_AVX512F_INSTRUCTIONS_AVAILABLE not reported");
}

TEST(kuser_shared_xstate_info) {
    /* KUSER_SHARED_DATA (0x7ffe0000) XState: EnabledFeatures at +0x3d8, EnabledVolatileFeatures +0x3e0,
     * Size +0x3e8, EnabledUserVisibleSupervisorFeatures +0x708 (Windows 10 19041 PDB layout).
     * Internal structure: INFO only, never a check. */
    MEMORY_BASIC_INFORMATION mbi;
    const uint8_t *k = (const uint8_t *)(ULONG_PTR)0x7ffe0000u;
    if (!VirtualQuery(k, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT || !(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE)) ||
        mbi.RegionSize < 0x710)
        SKIP("KUSER_SHARED_DATA not mapped readable at 0x7ffe0000");
    uint64_t en, vol, sup; uint32_t size;
    memcpy(&en, k + 0x3d8, 8); memcpy(&vol, k + 0x3e0, 8); memcpy(&size, k + 0x3e8, 4); memcpy(&sup, k + 0x708, 8);
    t_info("kusd_xstate", "enabled=0x%llx volatile=0x%llx size=%u uv_supervisor=0x%llx", (unsigned long long)en,
           (unsigned long long)vol, size, (unsigned long long)sup);
    xp_raw x; xp_read(&x);
    if (x.osxsave && en != x.xcr0) t_note("KUSER_SHARED_DATA XState.EnabledFeatures 0x%llx != XCR0 0x%llx", (unsigned long long)en,
                                          (unsigned long long)x.xcr0);
    CHECK(1);
}

TEST(xcr0_summary_info) {
    xp_raw x; xp_read(&x);
    const t_cpu_info *c = cpu();
    const t_env *e = t_env_get();
    char b[256];
    t_info("cpu", "%s %s fam %u model %u", c->vendor, c->brand, c->family, c->model);
    t_info("os", "%lu.%lu.%lu wine=%s arch=%s wow64=%d", e->os_major, e->os_minor, e->os_build, e->wine[0] ? e->wine : "-",
           e->process_arch, e->wow64);
    hex_bits(b, sizeof b, x.xcr0); t_info("xcr0", "0x%llx (%s)", (unsigned long long)x.xcr0, b);
    hex_bits(b, sizeof b, x.sup);  t_info("xcr0_supported", "0x%llx (%s)", (unsigned long long)x.sup, b);
    hex_bits(b, sizeof b, x.xss);  t_info("xss_supported", "0x%llx (%s)", (unsigned long long)x.xss, b);
    t_info("sizes", "ebx=%u ecx=%u d1.ebx=%u", x.d0.ebx, x.d0.ecx, x.d1.ebx);
    t_info("eligible", "avx=%d avx2=%d fma=%d avx512f=%d amx=%d", cpu_has(F_AVX), cpu_has(F_AVX2), cpu_has(F_FMA),
           cpu_has(F_AVX512F), cpu_has(F_AMX_TILE));
    if (x.osxsave && x.xg1) t_info("xinuse", "0x%llx", (unsigned long long)t_xgetbv(1));
    CHECK(1);
}
