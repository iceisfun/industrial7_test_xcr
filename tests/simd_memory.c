/* simd_memory - SIMD memory-access contracts: non-temporal stores/loads and fences, cache control,
 * prefetch, masked memory access fault suppression, gathers/scatters, 8/16-byte compare-exchange,
 * PAUSE / TSC ordering, RDRAND/RDSEED, MOVDIRI/MOVDIR64B, and how the OS handles faults raised by
 * these instructions (exception codes, access-violation parameters, restart of partially completed
 * gathers/scatters through guard pages and demand-committed pages).
 *
 * Contract under test:
 *   - Intel SDM Vol 1 10.4.6 / 11.4.4 / Vol 3 8.2 (memory ordering): non-temporal (NT) stores are weakly
 *     ordered, SFENCE/MFENCE order them before later stores; a processor always observes its own stores in
 *     program order (self-consistency, incl. NT/WC stores); x86 forbids load->load, store->store and
 *     load->store reordering (litmus MP, LB) but allows store->load reordering (SB) unless MFENCE or a
 *     locked instruction separates them.
 *   - MOVNTDQ/MOVNTPS/MOVNTPD/MOVNTDQA and VEX/EVEX VMOVNTDQ(A) require natural alignment (16/32/64 B),
 *     else #GP; MOVNTI has no alignment requirement.  Windows reports #GP as STATUS_ACCESS_VIOLATION.
 *   - CLFLUSH/CLFLUSHOPT/CLWB: permission checks and faults of a byte load (no fault on read-only or
 *     execute pages; #PF on PAGE_NOACCESS); memory contents are never changed by them.
 *   - PREFETCHh / PREFETCHW / PREFETCHWT1: hints, never raise memory exceptions (only #UD for LOCK).
 *   - VMASKMOVPS/PD, VPMASKMOVD/Q (SDM Vol 2 VMASKMOV): "Faults will not occur due to referencing any
 *     memory location if the corresponding mask bit for that memory location is 0"; loads zero inactive
 *     elements, stores leave them untouched; only the element MSB is the mask.
 *   - AVX-512 masked moves, compress/expand, gathers/scatters: memory fault suppression for masked-off
 *     elements (SDM Vol 1 ch. 15 "memory fault suppression", Vol 2 ch. 2 EVEX exception classes);
 *     merge vs zero masking; compress/expand access only popcnt(k) contiguous elements.
 *   - AVX2/AVX-512 gathers and scatters: inactive lanes are not accessed; on completion the mask register
 *     is all zero; when an element faults, completed elements have their mask bit cleared so the
 *     instruction can be restarted after the OS resolves the fault - which requires the OS to restore the
 *     full vector/opmask state from the exception CONTEXT (EXCEPTION_CONTINUE_EXECUTION).
 *   - MASKMOVDQU with a zero mask "may" still signal exceptions (implementation dependent) -> NOTE only.
 *   - CMPXCHG always performs a write cycle (writes the old value back on failure) -> on a read-only page
 *     even a failing LOCK CMPXCHG raises a write access violation.  CMPXCHG16B needs a 16-byte aligned
 *     operand (#GP).
 *   - RDRAND/RDSEED: CF=1 valid; CF=0 -> destination zeroed; OF/SF/ZF/AF/PF cleared.  RDPID returns
 *     IA32_TSC_AUX, the same value as RDTSCP's ECX.
 *   - Windows EXCEPTION_ACCESS_VIOLATION / STATUS_GUARD_PAGE_VIOLATION parameters (MSDN EXCEPTION_RECORD):
 *     ExceptionInformation[0] = 0 read / 1 write / 8 DEP, [1] = inaccessible virtual address.
 *     A guard page raises STATUS_GUARD_PAGE_VIOLATION once and loses PAGE_GUARD (MSDN "Creating Guard
 *     Pages"); a prefetch or a masked-off element must therefore leave PAGE_GUARD in place.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; implementation-defined facts
 * (write-combining speed, TSC_AUX numbering, split-lock behaviour, MASKMOVDQU zero-mask faults) -> NOTE/INFO.
 * Deliberately faulting operations run in the CHILD sm_fault. Non-faulting expectations run in-process
 * under a vectored "probe" handler so an unexpected exception becomes a FAIL with full details.
 *
 * Helpers not in common/ (integrator: candidates for common/): sm_region (multi-page regions with mixed
 * protections, reserved-only pages, guard pages), the probe VEH (resume after the faulting instruction
 * via an asm label, commit-on-demand / guard-page continue mode), litmus runner. */
#include "t.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>

#ifndef STATUS_GUARD_PAGE_VIOLATION
#define STATUS_GUARD_PAGE_VIOLATION ((DWORD)0x80000001L)
#endif
#define SM_AV   0xC0000005u
#define SM_GUARD 0x80000001u

#define NEED(f) do { if (!cpu_has(f)) SKIP("%s not usable (cpuid bit %d)", cpu_feature_name(f), cpu_hw(f)); } while (0)
#ifdef T_X64
#define NEED_X64() ((void)0)
#define SM_SP "rsp"
#else
#define NEED_X64() SKIP("x64-only instruction form")
#define SM_SP "esp"
#endif

/* ======================================================================== */
/* utilities                                                                */
/* ======================================================================== */
static size_t sm_pg(void) {
    static size_t pg;
    if (!pg) { SYSTEM_INFO si; GetSystemInfo(&si); pg = si.dwPageSize ? si.dwPageSize : 4096; }
    return pg;
}
static inline uint32_t sm_pat32(uint64_t off, uint32_t salt) {
    return (uint32_t)t_mix64(off * 0x9E3779B97F4A7C15ull ^ ((uint64_t)salt << 40) ^ 0x51ED270B);
}
/* deterministic page contents: dword j = sm_pat32(region offset + 4j) */
static void sm_fill_page_pat(uint8_t *page, size_t pg, size_t off) {
    uint32_t *w = (uint32_t *)page;
    for (size_t j = 0; j < pg / 4; j++) w[j] = sm_pat32(off + 4 * j, 0x5EED);
}
static inline void sm_sfence(void) { __asm__ volatile("sfence" ::: "memory"); }
static inline void sm_mfence(void) { __asm__ volatile("mfence" ::: "memory"); }
static inline void sm_pause(void)  { __asm__ volatile("pause" ::: "memory"); }
static inline void sm_cbar(void)   { __asm__ volatile("" ::: "memory"); }

static int sm_cpu_list(int *out, int max) {
    DWORD_PTR pm = 0, sysm = 0; int n = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sysm)) return 0;
    for (int i = 0; i < (int)(sizeof(DWORD_PTR) * 8) && n < max; i++) if ((pm >> i) & 1) out[n++] = i;
    return n;
}
static uint8_t *sm_valloc(size_t n) { return (uint8_t *)VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }
static void sm_vfree(void *p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }
static int sm_is_guard(const void *p) {
    MEMORY_BASIC_INFORMATION m;
    return VirtualQuery(p, &m, sizeof m) && m.State == MEM_COMMIT && (m.Protect & PAGE_GUARD);
}

/* ---- regions: n pages, page i committed with prot[i] (0 = reserved only), PAGE_GUARD allowed ---- */
typedef struct sm_region { uint8_t *base; size_t pg; int n; } sm_region;
static int sm_region_alloc(sm_region *r, int n, const DWORD *prot) {
    r->pg = sm_pg(); r->n = n;
    r->base = (uint8_t *)VirtualAlloc(NULL, (size_t)n * r->pg, MEM_RESERVE, PAGE_NOACCESS);
    if (!r->base) return 0;
    for (int i = 0; i < n; i++) {
        if (!prot[i]) continue;
        uint8_t *p = r->base + (size_t)i * r->pg;
        if (!VirtualAlloc(p, r->pg, MEM_COMMIT, PAGE_READWRITE)) goto fail;
        sm_fill_page_pat(p, r->pg, (size_t)i * r->pg);
        DWORD old;
        if (prot[i] != PAGE_READWRITE && !VirtualProtect(p, r->pg, prot[i], &old)) goto fail;
    }
    return 1;
fail:
    VirtualFree(r->base, 0, MEM_RELEASE); r->base = NULL; return 0;
}
static void sm_region_free(sm_region *r) { if (r->base) VirtualFree(r->base, 0, MEM_RELEASE); r->base = NULL; }
static inline uint8_t *sm_page(const sm_region *r, int i) { return r->base + (size_t)i * r->pg; }
/* standard layout: [0 RW][1 NOACCESS][2 RW][3 READONLY] */
static const DWORD g_lay4[4] = { PAGE_READWRITE, PAGE_NOACCESS, PAGE_READWRITE, PAGE_READONLY };
#define REGION4(r) REQUIREF(sm_region_alloc(&(r), 4, g_lay4), "region alloc failed (%lu)", GetLastError())

/* ======================================================================== */
/* probe: in-process VEH that turns an exception at a marked instruction    */
/* into a recorded event and resumes after it                               */
/* ======================================================================== */
typedef struct sm_probe {
    volatile ULONG_PTR resume;     /* continue address after the probed instruction (0 = not armed) */
    volatile DWORD tid;
    volatile LONG nexc;            /* exceptions that hit the armed instruction */
    DWORD code; ULONG_PTR rw, addr, ip; DWORD nparams;
    /* region mode: AV on a reserved page -> commit+fill+continue; guard page -> continue */
    uint8_t *rlo, *rhi; LONG maxfix;
    volatile LONG nfix, nguard, ncommit;
} sm_probe;
static sm_probe g_pr;
static volatile LONG g_veh_installed;

static LONG CALLBACK sm_veh(PEXCEPTION_POINTERS ep) {
    if (!g_pr.tid || GetCurrentThreadId() != g_pr.tid) return EXCEPTION_CONTINUE_SEARCH;
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    DWORD code = er->ExceptionCode;
    if (code != SM_AV && code != SM_GUARD && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_PRIV_INSTRUCTION && code != EXCEPTION_DATATYPE_MISALIGNMENT &&
        code != EXCEPTION_IN_PAGE_ERROR)
        return EXCEPTION_CONTINUE_SEARCH;
    ULONG_PTR rw = er->NumberParameters >= 1 ? er->ExceptionInformation[0] : 0;
    ULONG_PTR a = er->NumberParameters >= 2 ? er->ExceptionInformation[1] : 0;
    if (g_pr.rlo && (code == SM_AV || code == SM_GUARD) && er->NumberParameters >= 2 &&
        a >= (ULONG_PTR)g_pr.rlo && a < (ULONG_PTR)g_pr.rhi && g_pr.nfix < g_pr.maxfix) {
        size_t pg = sm_pg();
        if (code == SM_GUARD) { g_pr.nguard++; g_pr.nfix++; return EXCEPTION_CONTINUE_EXECUTION; }
        uint8_t *page = g_pr.rlo + ((a - (ULONG_PTR)g_pr.rlo) / pg) * pg;
        MEMORY_BASIC_INFORMATION m;
        if (VirtualQuery(page, &m, sizeof m) && m.State == MEM_RESERVE &&
            VirtualAlloc(page, pg, MEM_COMMIT, PAGE_READWRITE)) {
            sm_fill_page_pat(page, pg, (size_t)(page - g_pr.rlo));
            g_pr.ncommit++; g_pr.nfix++;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if (!g_pr.resume) return EXCEPTION_CONTINUE_SEARCH;
    if (InterlockedIncrement(&g_pr.nexc) == 1) {
        g_pr.code = code; g_pr.rw = rw; g_pr.addr = a; g_pr.nparams = er->NumberParameters;
        g_pr.ip = (ULONG_PTR)er->ExceptionAddress;
    }
#ifdef T_X64
    ep->ContextRecord->Rip = g_pr.resume;
#else
    ep->ContextRecord->Eip = g_pr.resume;
#endif
    g_pr.resume = 0;
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void pr_begin(void) {
    if (InterlockedCompareExchange(&g_veh_installed, 1, 0) == 0) AddVectoredExceptionHandler(1, sm_veh);
    g_pr.tid = 0;
    g_pr.resume = 0; g_pr.nexc = 0; g_pr.code = 0; g_pr.rw = g_pr.addr = g_pr.ip = 0;
    g_pr.rlo = g_pr.rhi = NULL; g_pr.maxfix = 0; g_pr.nfix = g_pr.nguard = g_pr.ncommit = 0;
    g_pr.tid = GetCurrentThreadId();
}
static void pr_region(uint8_t *lo, size_t n, LONG maxfix) {
    g_pr.rlo = lo; g_pr.rhi = lo + n; g_pr.maxfix = maxfix; g_pr.nfix = g_pr.nguard = g_pr.ncommit = 0;
}
static void pr_end(void) { g_pr.tid = 0; g_pr.resume = 0; g_pr.rlo = g_pr.rhi = NULL; }
static void pr_clear(void) { g_pr.nexc = 0; g_pr.code = 0; g_pr.rw = g_pr.addr = g_pr.ip = 0; }
static int pr_desc(char *b, size_t cap) {
    return snprintf(b, cap, "exception 0x%08lx rw=%lu addr=%p ip=%p", (unsigned long)g_pr.code,
                    (unsigned long)g_pr.rw, (void *)g_pr.addr, (void *)g_pr.ip);
}
/* expect that the armed instruction(s) since the last clear raised nothing */
#define PR_EXPECT_NONE(...) do { char d_[160]; d_[0] = 0; if (g_pr.nexc) pr_desc(d_, sizeof d_); \
    char l_[200]; snprintf(l_, sizeof l_, __VA_ARGS__); \
    CHECKF(g_pr.nexc == 0, "%s: unexpected %s", l_, d_); pr_clear(); } while (0)

/* asm fragments: PR_ARM stores the address of the local label 9 into g_pr.resume; PR_DONE defines it and
 * disarms. Exactly one instruction between them should be the one that may fault. */
#ifdef T_X64
#define PR_ARM   "lea 9f(%%rip), %%r11\n\tmov %%r11, %[pr_res]\n\t"
#define PR_DONE  "\n9:\n\tmovq $0, %[pr_res]\n\t"
#define PR_CLOB_C , "r11"
#else
#define PR_ARM   "movl $9f, %[pr_res]\n\t"
#define PR_DONE  "\n9:\n\tmovl $0, %[pr_res]\n\t"
#define PR_CLOB_C
#endif
#define PR_OUT   [pr_res] "=m"(g_pr.resume)

/* ======================================================================== */
/* fault children: one generic child; the VEH reports the first exception   */
/* ======================================================================== */
static volatile ULONG_PTR g_child_bad;     /* page the op is expected to fault on */
static LONG CALLBACK child_veh(PEXCEPTION_POINTERS ep) {
    static volatile LONG once;
    if (InterlockedIncrement(&once) != 1) return EXCEPTION_CONTINUE_SEARCH;
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    ULONG_PTR rw = er->NumberParameters >= 1 ? er->ExceptionInformation[0] : 0;
    ULONG_PTR a = er->NumberParameters >= 2 ? er->ExceptionInformation[1] : 0;
#ifdef T_X64
    ULONG_PTR cip = (ULONG_PTR)ep->ContextRecord->Rip;
#else
    ULONG_PTR cip = (ULONG_PTR)ep->ContextRecord->Eip;
#endif
    /* hardware faults: ExceptionAddress and the context IP both name the faulting instruction */
    printf("SM_EXC code=0x%08lx rw=%lu np=%lu off=%lld addr=0x%llx ipeq=%d\n", (unsigned long)er->ExceptionCode,
           (unsigned long)rw, (unsigned long)er->NumberParameters, (long long)(a - g_child_bad), (unsigned long long)a,
           (ULONG_PTR)er->ExceptionAddress == cip);
    fflush(stdout);
    return EXCEPTION_CONTINUE_SEARCH;
}

/* ---- non-temporal stores/loads with misalignment / bad pages ---- */
static void fo_movntdq_mis(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 8; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("pxor %%xmm0, %%xmm0\n\tmovntdq %%xmm0, (%0)" :: "r"(p) : "xmm0", "memory");
}
static void fo_movntps_mis(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 4; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("xorps %%xmm0, %%xmm0\n\tmovntps %%xmm0, (%0)" :: "r"(p) : "xmm0", "memory");
}
static void fo_movntpd_mis(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 8; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("xorpd %%xmm0, %%xmm0\n\tmovntpd %%xmm0, (%0)" :: "r"(p) : "xmm0", "memory");
}
static void fo_movntdqa_mis(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 8; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("movntdqa (%0), %%xmm0" :: "r"(p) : "xmm0", "memory");
}
static void fo_vmovntdq_ymm_mis_avx(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 16; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("vxorps %%xmm0, %%xmm0, %%xmm0\n\tvmovntdq %%ymm0, (%0)\n\tvzeroupper" :: "r"(p) : "xmm0", "memory");
}
static void fo_vmovntdq_xmm_mis_avx(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 8; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("vxorps %%xmm0, %%xmm0, %%xmm0\n\tvmovntdq %%xmm0, (%0)" :: "r"(p) : "xmm0", "memory");
}
static void fo_vmovntdqa_ymm_mis_avx2(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 16; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("vmovntdqa (%0), %%ymm0\n\tvzeroupper" :: "r"(p) : "xmm0", "memory");
}
static void fo_vmovntdq_zmm_mis_512(sm_region *r) {
    uint8_t *p = sm_page(r, 0) + 32; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("vpxord %%zmm0, %%zmm0, %%zmm0\n\tvmovntdq %%zmm0, (%0)\n\tvzeroupper" :: "r"(p) : "xmm0", "memory");
}
static void fo_movnti_noacc(sm_region *r) {
    uint8_t *p = sm_page(r, 1) + 64; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("movnti %1, (%0)" :: "r"(p), "r"(0x1234) : "memory");
}
static void fo_movntdq_ro(sm_region *r) {
    uint8_t *p = sm_page(r, 3) + 64; g_child_bad = (ULONG_PTR)sm_page(r, 3);
    __asm__ volatile("pxor %%xmm0, %%xmm0\n\tmovntdq %%xmm0, (%0)" :: "r"(p) : "xmm0", "memory");
}
static void fo_movntdqa_noacc(sm_region *r) {
    uint8_t *p = sm_page(r, 1) + 128; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("movntdqa (%0), %%xmm0" :: "r"(p) : "xmm0", "memory");
}
/* ---- cache control on PAGE_NOACCESS ---- */
static void fo_clflush_noacc(sm_region *r) {
    uint8_t *p = sm_page(r, 1) + 192; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("clflush (%0)" :: "r"(p) : "memory");
}
static void fo_clflushopt_noacc(sm_region *r) {
    uint8_t *p = sm_page(r, 1) + 192; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("clflushopt (%0)" :: "r"(p) : "memory");
}
static void fo_clwb_noacc(sm_region *r) {
    uint8_t *p = sm_page(r, 1) + 192; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("clwb (%0)" :: "r"(p) : "memory");
}
/* ---- MASKMOVDQU straddling into the bad page ---- */
static void fo_maskmov_common(sm_region *r, const uint8_t *mask, uint8_t *p) {
    static __attribute__((aligned(16))) uint8_t data[16], m[16];
    memcpy(m, mask, 16);
    for (int i = 0; i < 16; i++) data[i] = (uint8_t)(0xC0 + i);
    g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("movdqu (%1), %%xmm0\n\tmovdqu (%2), %%xmm1\n\tmaskmovdqu %%xmm1, %%xmm0"
                     : : "D"(p), "r"(data), "r"(m) : "xmm0", "xmm1", "memory");
    printf("SM_BYTES %02x%02x%02x%02x\n", p[0], p[1], p[2], p[3]);
}
static void fo_maskmovdqu_active(sm_region *r) {
    uint8_t m[16] = { 0 }; m[12] = 0x80; m[0] = 0x80;
    fo_maskmov_common(r, m, sm_page(r, 1) - 8);
}
static void fo_maskmovdqu_inactive_info(sm_region *r) {
    uint8_t m[16] = { 0 }; for (int i = 0; i < 8; i++) m[i] = 0x80;
    fo_maskmov_common(r, m, sm_page(r, 1) - 8);
}
static void fo_maskmovdqu_zero_info(sm_region *r) {
    uint8_t m[16] = { 0 };
    uint8_t *p = sm_page(r, 1) + 32;
    static __attribute__((aligned(16))) uint8_t data[16];
    g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("movdqu (%1), %%xmm0\n\tmovdqu (%2), %%xmm1\n\tmaskmovdqu %%xmm1, %%xmm0"
                     : : "D"(p), "r"(data), "r"(m) : "xmm0", "xmm1", "memory");
}
/* ---- VEX masked moves with an ACTIVE element in the bad page ---- */
static void fo_vmaskmovps_ld_active_avx(sm_region *r) {
    static __attribute__((aligned(32))) uint32_t m[8] = { 0, 0, 0, 0, 0, 0x80000000u, 0, 0 };
    uint8_t *p = sm_page(r, 1) - 16; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("vmovdqu (%1), %%ymm1\n\tvmaskmovps (%0), %%ymm1, %%ymm0\n\tvzeroupper" :: "r"(p), "r"(m) : "xmm0", "xmm1", "memory");
}
static void fo_vmaskmovps_st_active_avx(sm_region *r) {
    static __attribute__((aligned(32))) uint32_t m[8] = { 0x80000000u, 0, 0, 0, 0, 0x80000000u, 0, 0 };
    uint8_t *p = sm_page(r, 1) - 16; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("vmovdqu (%1), %%ymm1\n\tvpcmpeqd %%xmm0, %%xmm0, %%xmm0\n\tvmaskmovps %%ymm0, %%ymm1, (%0)\n\tvzeroupper"
                     :: "r"(p), "r"(m) : "xmm0", "xmm1", "memory");
}
static void fo_vpmaskmovd_ld_active_avx2(sm_region *r) {
    static __attribute__((aligned(32))) uint32_t m[8] = { 0, 0, 0, 0, 0, 0, 0, 0x80000000u };
    uint8_t *p = sm_page(r, 1) - 16; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("vmovdqu (%1), %%ymm1\n\tvpmaskmovd (%0), %%ymm1, %%ymm0\n\tvzeroupper" :: "r"(p), "r"(m) : "xmm0", "xmm1", "memory");
}
static void fo_vpmaskmovq_st_active_avx2(sm_region *r) {
    static __attribute__((aligned(32))) uint64_t m[4] = { 0, 0, 0x8000000000000000ull, 0 };
    uint8_t *p = sm_page(r, 1) - 16; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("vmovdqu (%1), %%ymm1\n\tvpcmpeqd %%xmm0, %%xmm0, %%xmm0\n\tvpmaskmovq %%ymm0, %%ymm1, (%0)\n\tvzeroupper"
                     :: "r"(p), "r"(m) : "xmm0", "xmm1", "memory");
}
/* ---- AVX-512 masked moves ---- */
T_TARGET("avx512f") static void fo_zmm_ld_active_512(sm_region *r) {
    static uint16_t k = 0x0101;  /* lane 0 good, lane 8 in bad page */
    uint8_t *p = sm_page(r, 1) - 32; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("kmovw (%1), %%k1\n\tvmovdqu32 (%0), %%zmm0%{%%k1%}%{z%}\n\tvzeroupper" :: "r"(p), "r"(&k) : "xmm0", "k1", "memory");
}
T_TARGET("avx512f") static void fo_zmm_st_active_512(sm_region *r) {
    static uint16_t k = 0x0201;
    uint8_t *p = sm_page(r, 1) - 32; g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("kmovw (%1), %%k1\n\tvpternlogd $0xff, %%zmm0, %%zmm0, %%zmm0\n\tvmovdqu32 %%zmm0, (%0)%{%%k1%}\n\tvzeroupper"
                     :: "r"(p), "r"(&k) : "xmm0", "k1", "memory");
}
T_TARGET("avx512f") static void fo_vmovdqa32_k0_mis_info_512(sm_region *r) {
    static uint16_t k = 0;
    uint8_t *p = sm_page(r, 0) + 16; g_child_bad = (ULONG_PTR)p;
    __asm__ volatile("kmovw (%1), %%k1\n\tvmovdqa32 (%0), %%zmm0%{%%k1%}%{z%}\n\tvzeroupper" :: "r"(p), "r"(&k) : "xmm0", "k1", "memory");
}
/* ---- gathers / scatters with an ACTIVE lane in the bad page ---- */
static void fo_gather_avx2_active(sm_region *r) {
    static __attribute__((aligned(32))) int32_t idx[8];
    static __attribute__((aligned(32))) uint32_t m[8], d[8];
    size_t pg = sm_pg();
    for (int i = 0; i < 8; i++) { idx[i] = i * 4; m[i] = 0x80000000u; }
    idx[3] = (int32_t)(pg / 4 + 5);   /* page 1 */
    uint8_t *b = sm_page(r, 0); g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("vmovdqu (%1), %%ymm2\n\tvmovdqu (%2), %%ymm3\n\tvpxor %%xmm1, %%xmm1, %%xmm1\n\t"
                     "vpgatherdd %%ymm3, (%0,%%ymm2,4), %%ymm1\n\tvmovdqu %%ymm1, (%3)\n\tvzeroupper"
                     :: "r"(b), "r"(idx), "r"(m), "r"(d) : "xmm1", "xmm2", "xmm3", "memory");
}
T_TARGET("avx512f") static void fo_gather_512_active(sm_region *r) {
    static __attribute__((aligned(64))) int32_t idx[16];
    static uint16_t k = 0xFFFF;
    size_t pg = sm_pg();
    for (int i = 0; i < 16; i++) idx[i] = i * 3;
    idx[9] = (int32_t)(pg / 4 + 7);
    uint8_t *b = sm_page(r, 0); g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("kmovw (%2), %%k1\n\tvmovdqu64 (%1), %%zmm2\n\tvpxord %%zmm1, %%zmm1, %%zmm1\n\t"
                     "vpgatherdd (%0,%%zmm2,4), %%zmm1%{%%k1%}\n\tvzeroupper"
                     :: "r"(b), "r"(idx), "r"(&k) : "xmm1", "xmm2", "k1", "memory");
}
T_TARGET("avx512f") static void fo_scatter_512_active(sm_region *r) {
    static __attribute__((aligned(64))) int32_t idx[16];
    static uint16_t k = 0xFFFF;
    size_t pg = sm_pg();
    for (int i = 0; i < 16; i++) idx[i] = i * 3;
    idx[12] = (int32_t)(pg / 4 + 9);
    uint8_t *b = sm_page(r, 0); g_child_bad = (ULONG_PTR)sm_page(r, 1);
    __asm__ volatile("kmovw (%2), %%k1\n\tvmovdqu64 (%1), %%zmm2\n\tvpternlogd $0xff, %%zmm1, %%zmm1, %%zmm1\n\t"
                     "vpscatterdd %%zmm1, (%0,%%zmm2,4)%{%%k1%}\n\tvzeroupper"
                     :: "r"(b), "r"(idx), "r"(&k) : "xmm1", "xmm2", "k1", "memory");
}
/* ---- locked ops ---- */
static void fo_cmpxchg16b_mis(sm_region *r) {
#ifdef T_X64
    uint8_t *p = sm_page(r, 0) + 8; g_child_bad = (ULONG_PTR)p;
    uint64_t lo = 0, hi = 0;
    __asm__ volatile("lock cmpxchg16b (%2)" : "+a"(lo), "+d"(hi) : "r"(p), "b"((uint64_t)1), "c"((uint64_t)2) : "memory", "cc");
#else
    (void)r;
#endif
}
static void fo_lock_cmpxchg_ro(sm_region *r) {
    uint32_t *p = (uint32_t *)(sm_page(r, 3) + 256); g_child_bad = (ULONG_PTR)sm_page(r, 3);
    uint32_t expect = ~*p;   /* compare fails: the old value is still written back */
    __asm__ volatile("lock cmpxchgl %2, (%1)" : "+a"(expect) : "r"(p), "r"(0x55u) : "memory", "cc");
    printf("SM_CMPXCHG_RO eax=0x%08x\n", expect);
}
static void fo_split_lock_info(sm_region *r) {
    uint8_t *line = sm_page(r, 0) + 128;
    uint32_t *p = (uint32_t *)(line + 62);   /* straddles a 64-byte line boundary */
    *p = 41;
    double t0 = t_now_ms();
    __asm__ volatile("lock addl $1, (%0)" :: "r"(p) : "memory", "cc");
    double t1 = t_now_ms();
    printf("SM_SPLIT value=%u ms=%.3f\n", *p, t1 - t0);
}
static void fo_movdir64b_mis(sm_region *r) {
    uint8_t *dst = sm_page(r, 0) + 8, *src = sm_page(r, 0) + 1024; g_child_bad = (ULONG_PTR)dst;
    __asm__ volatile("movdir64b (%1), %0" :: "r"(dst), "r"(src) : "memory");
}
static void fo_waitpkg_info(sm_region *r) {
    (void)r;
    uint32_t lo, hi; unsigned char cf1, cf2;
    uint64_t t0 = t_rdtsc(), dl = t0 + 100000;
    lo = (uint32_t)dl; hi = (uint32_t)(dl >> 32);
    __asm__ volatile("tpause %%ecx\n\tsetc %0" : "=qm"(cf1) : "a"(lo), "d"(hi), "c"(0) : "cc", "memory");
    uint64_t t1 = t_rdtsc();
    static volatile uint32_t mon;
    dl = t_rdtsc() + 100000; lo = (uint32_t)dl; hi = (uint32_t)(dl >> 32);
    __asm__ volatile("umonitor %1\n\tumwait %%ecx\n\tsetc %0" : "=qm"(cf2) : "r"(&mon), "a"(lo), "d"(hi), "c"(1) : "cc", "memory");
    uint64_t t2 = t_rdtsc();
    printf("SM_WAITPKG tpause_cf=%d tpause_ticks=%llu umwait_cf=%d umwait_ticks=%llu\n", cf1,
           (unsigned long long)(t1 - t0), cf2, (unsigned long long)(t2 - t1));
}

typedef struct fault_op { const char *name; int f1, f2, x64only; void (*fn)(sm_region *); } fault_op;
#define NOF (-1)
static const fault_op g_fops[] = {
    { "movntdq_mis", F_SSE2, NOF, 0, fo_movntdq_mis },
    { "movntps_mis", F_SSE, NOF, 0, fo_movntps_mis },
    { "movntpd_mis", F_SSE2, NOF, 0, fo_movntpd_mis },
    { "movntdqa_mis", F_SSE41, NOF, 0, fo_movntdqa_mis },
    { "vmovntdq_xmm_mis", F_AVX, NOF, 0, fo_vmovntdq_xmm_mis_avx },
    { "vmovntdq_ymm_mis", F_AVX, NOF, 0, fo_vmovntdq_ymm_mis_avx },
    { "vmovntdqa_ymm_mis", F_AVX2, NOF, 0, fo_vmovntdqa_ymm_mis_avx2 },
    { "vmovntdq_zmm_mis", F_AVX512F, NOF, 0, fo_vmovntdq_zmm_mis_512 },
    { "movnti_noacc", F_SSE2, NOF, 0, fo_movnti_noacc },
    { "movntdq_ro", F_SSE2, NOF, 0, fo_movntdq_ro },
    { "movntdqa_noacc", F_SSE41, NOF, 0, fo_movntdqa_noacc },
    { "clflush_noacc", F_CLFSH, NOF, 0, fo_clflush_noacc },
    { "clflushopt_noacc", F_CLFLUSHOPT, NOF, 0, fo_clflushopt_noacc },
    { "clwb_noacc", F_CLWB, NOF, 0, fo_clwb_noacc },
    { "maskmovdqu_active", F_SSE2, NOF, 0, fo_maskmovdqu_active },
    { "maskmovdqu_inactive_info", F_SSE2, NOF, 0, fo_maskmovdqu_inactive_info },
    { "maskmovdqu_zero_info", F_SSE2, NOF, 0, fo_maskmovdqu_zero_info },
    { "vmaskmovps_ld_active", F_AVX, NOF, 0, fo_vmaskmovps_ld_active_avx },
    { "vmaskmovps_st_active", F_AVX, NOF, 0, fo_vmaskmovps_st_active_avx },
    { "vpmaskmovd_ld_active", F_AVX2, NOF, 0, fo_vpmaskmovd_ld_active_avx2 },
    { "vpmaskmovq_st_active", F_AVX2, NOF, 0, fo_vpmaskmovq_st_active_avx2 },
    { "zmm_ld_active", F_AVX512F, NOF, 0, fo_zmm_ld_active_512 },
    { "zmm_st_active", F_AVX512F, NOF, 0, fo_zmm_st_active_512 },
    { "vmovdqa32_k0_mis_info", F_AVX512F, NOF, 0, fo_vmovdqa32_k0_mis_info_512 },
    { "gather_avx2_active", F_AVX2, NOF, 0, fo_gather_avx2_active },
    { "gather_512_active", F_AVX512F, NOF, 0, fo_gather_512_active },
    { "scatter_512_active", F_AVX512F, NOF, 0, fo_scatter_512_active },
    { "cmpxchg16b_mis", F_CX16, NOF, 1, fo_cmpxchg16b_mis },
    { "lock_cmpxchg_ro", NOF, NOF, 0, fo_lock_cmpxchg_ro },
    { "split_lock_info", NOF, NOF, 0, fo_split_lock_info },
    { "movdir64b_mis", F_MOVDIR64B, NOF, 0, fo_movdir64b_mis },
    { "waitpkg_info", F_WAITPKG, F_TSC, 0, fo_waitpkg_info },
};
static const fault_op *fop_find(const char *n) {
    for (size_t i = 0; i < T_COUNTOF(g_fops); i++) if (!strcmp(g_fops[i].name, n)) return &g_fops[i];
    return NULL;
}
static int fop_avail(const fault_op *o) {
#ifndef T_X64
    if (o->x64only) return 0;
#endif
    return (o->f1 == NOF || cpu_has((t_feature)o->f1)) && (o->f2 == NOF || cpu_has((t_feature)o->f2));
}

CHILD(sm_fault) {
    if (argc < 2) return 2;
    const fault_op *o = fop_find(argv[1]);
    if (!o) { printf("SM_UNKNOWN\n"); return 2; }
    if (!fop_avail(o)) { printf("SM_UNSUPPORTED\n"); return 4; }
    sm_region r;
    if (!sm_region_alloc(&r, 4, g_lay4)) { printf("SM_ALLOC_FAILED\n"); return 5; }
    AddVectoredExceptionHandler(1, child_veh);
    fflush(stdout);
    o->fn(&r);
    printf("SM_NOFAULT\n"); fflush(stdout);
    return 3;
}

typedef struct sm_fres { int ok, faulted, nofault, ipeq; DWORD code; unsigned long rw; long long off; unsigned long long addr; } sm_fres;
static t_child_result g_cr;
static void run_fop(const char *op, sm_fres *f) {
    memset(f, 0, sizeof *f);
    if (!t_child_run("sm_fault", op, 15000, &g_cr)) { CHECKF(0, "%s: child did not start (err %lu)", op, g_cr.spawn_error); return; }
    CHECKF(!g_cr.timed_out, "%s: child timed out", op);
    char v[256];
    if (t_child_find(&g_cr, "SM_UNKNOWN", v, sizeof v)) { CHECKF(0, "%s: unknown op", op); return; }
    if (t_child_find(&g_cr, "SM_ALLOC_FAILED", v, sizeof v)) { CHECKF(0, "%s: child region alloc failed", op); return; }
    f->ok = 1;
    if (t_child_find(&g_cr, "SM_EXC ", v, sizeof v)) {
        unsigned long c = 0, rw = 0, np = 0; long long off = 0; unsigned long long a = 0; int ipeq = -1;
        sscanf(v, "code=0x%lx rw=%lu np=%lu off=%lld addr=0x%llx ipeq=%d", &c, &rw, &np, &off, &a, &ipeq);
        f->faulted = 1; f->code = (DWORD)c; f->rw = rw; f->off = off; f->addr = a; f->ipeq = ipeq;
    }
    if (t_child_find(&g_cr, "SM_NOFAULT", v, sizeof v)) f->nofault = 1;
}
/* expect a hardware fault reported as STATUS_ACCESS_VIOLATION.
 * want_rw: 0/1 = check ExceptionInformation[0]; -1 = record only.  page_check: address must lie in the
 * expected page (page faults); 0 for #GP (the address is not architecturally defined -> NOTE). */
static void expect_av(const char *op, int want_rw, int page_check) {
    sm_fres f;
    const fault_op *o = fop_find(op);
    if (!o || !fop_avail(o)) { t_note("%s: not available here", op); return; }
    run_fop(op, &f);
    if (!f.ok) return;
    CHECKF(f.faulted && !f.nofault, "%s: expected an access violation, child reported none (exit 0x%lx)", op, (unsigned long)g_cr.exit_code);
    if (!f.faulted) return;
    CHECKF(f.code == SM_AV, "%s: exception 0x%08lx, want STATUS_ACCESS_VIOLATION", op, (unsigned long)f.code);
    CHECKF(g_cr.exc_code == f.code, "%s: process exit/exception code 0x%08lx != first exception 0x%08lx", op,
           (unsigned long)g_cr.exc_code, (unsigned long)f.code);
    /* a fault is reported at the faulting instruction: EXCEPTION_RECORD.ExceptionAddress == CONTEXT IP */
    CHECKF(f.ipeq == 1, "%s: ExceptionAddress differs from the context instruction pointer (ipeq=%d)", op, f.ipeq);
    if (want_rw >= 0) CHECKF(f.rw == (unsigned long)want_rw, "%s: ExceptionInformation[0]=%lu want %d", op, f.rw, want_rw);
    else t_info(op, "rw=%lu addr=0x%llx", f.rw, f.addr);
    if (page_check) CHECKF(f.off >= 0 && f.off < (long long)sm_pg(), "%s: fault address offset %lld outside the inaccessible page", op, f.off);
    else t_note("%s: #GP reported as code=0x%08lx rw=%lu addr=0x%llx", op, (unsigned long)f.code, f.rw, f.addr);
}

/* ======================================================================== */
/* non-temporal copy kernels: n % 64 == 0, n > 0                             */
/* ======================================================================== */
typedef void (*sm_copy_fn)(void *d, const void *s, size_t n);
static void k_movnti32(void *d, const void *s, size_t n) {
    int *dp = (int *)d; const uint8_t *sp = (const uint8_t *)s;
    for (size_t i = 0; i < n / 4; i++) { int v; memcpy(&v, sp + 4 * i, 4); _mm_stream_si32(dp + i, v); }
}
#ifdef T_X64
static void k_movnti64(void *d, const void *s, size_t n) {
    long long *dp = (long long *)d; const uint8_t *sp = (const uint8_t *)s;
    for (size_t i = 0; i < n / 8; i++) { long long v; memcpy(&v, sp + 8 * i, 8); _mm_stream_si64(dp + i, v); }
}
#endif
static void k_movntdq(void *d, const void *s, size_t n) {
    uint8_t *dp = (uint8_t *)d; const uint8_t *sp = (const uint8_t *)s;
    for (size_t i = 0; i < n; i += 64) {
        __m128i a = _mm_loadu_si128((const __m128i *)(sp + i)), b = _mm_loadu_si128((const __m128i *)(sp + i + 16));
        __m128i c = _mm_loadu_si128((const __m128i *)(sp + i + 32)), e = _mm_loadu_si128((const __m128i *)(sp + i + 48));
        _mm_stream_si128((__m128i *)(dp + i), a); _mm_stream_si128((__m128i *)(dp + i + 16), b);
        _mm_stream_si128((__m128i *)(dp + i + 32), c); _mm_stream_si128((__m128i *)(dp + i + 48), e);
    }
}
static void k_movntps(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 16) _mm_stream_ps((float *)((uint8_t *)d + i), _mm_loadu_ps((const float *)((const uint8_t *)s + i)));
}
static void k_movntpd(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 16) _mm_stream_pd((double *)((uint8_t *)d + i), _mm_loadu_pd((const double *)((const uint8_t *)s + i)));
}
static void k_movntq_mmx(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 8)
        __asm__ volatile("movq (%1), %%mm0\n\tmovntq %%mm0, (%0)" :: "r"((uint8_t *)d + i), "r"((const uint8_t *)s + i) : "mm0", "memory");
    __asm__ volatile("emms");
}
static void k_maskmovdqu_all(void *d, const void *s, size_t n) {
    __m128i all = _mm_set1_epi8((char)0x80);
    for (size_t i = 0; i < n; i += 16) _mm_maskmoveu_si128(_mm_loadu_si128((const __m128i *)((const uint8_t *)s + i)), all, (char *)d + i);
}
T_TARGET("sse4.1") static void k_movntdqa_sse41(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 16) _mm_storeu_si128((__m128i *)((uint8_t *)d + i), _mm_stream_load_si128((__m128i *)((uint8_t *)s + i)));
}
T_TARGET("avx") static void k_vmovntdq_xmm_avx(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 16) _mm_stream_si128((__m128i *)((uint8_t *)d + i), _mm_loadu_si128((const __m128i *)((const uint8_t *)s + i)));
}
T_TARGET("avx") static void k_vmovntdq_ymm_avx(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 32) _mm256_stream_si256((__m256i *)((uint8_t *)d + i), _mm256_loadu_si256((const __m256i *)((const uint8_t *)s + i)));
}
T_TARGET("avx") static void k_vmovntps_ymm_avx(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 32) _mm256_stream_ps((float *)((uint8_t *)d + i), _mm256_loadu_ps((const float *)((const uint8_t *)s + i)));
}
T_TARGET("avx") static void k_vmovntpd_ymm_avx(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 32) _mm256_stream_pd((double *)((uint8_t *)d + i), _mm256_loadu_pd((const double *)((const uint8_t *)s + i)));
}
T_TARGET("avx2") static void k_vmovntdqa_ymm_avx2(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 32) _mm256_storeu_si256((__m256i *)((uint8_t *)d + i), _mm256_stream_load_si256((const __m256i *)((const uint8_t *)s + i)));
}
T_TARGET("avx512f") static void k_vmovntdq_zmm_512(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 64) _mm512_stream_si512((void *)((uint8_t *)d + i), _mm512_loadu_si512((const void *)((const uint8_t *)s + i)));
}
T_TARGET("avx512f") static void k_vmovntdqa_zmm_512(void *d, const void *s, size_t n) {
    for (size_t i = 0; i < n; i += 64) _mm512_storeu_si512((void *)((uint8_t *)d + i), _mm512_stream_load_si512((void *)((uint8_t *)s + i)));
}

typedef struct sm_kernel { const char *name; sm_copy_fn fn; int align; int feat; int src_aligned; } sm_kernel;
static const sm_kernel g_k[] = {
    { "movnti32", k_movnti32, 4, F_SSE2, 0 },
#ifdef T_X64
    { "movnti64", k_movnti64, 8, F_SSE2, 0 },
#else
    { "movnti64", NULL, 8, F_SSE2, 0 },
#endif
    { "movntdq", k_movntdq, 16, F_SSE2, 0 },
    { "movntps", k_movntps, 16, F_SSE, 0 },
    { "movntpd", k_movntpd, 16, F_SSE2, 0 },
    { "movntq_mmx", k_movntq_mmx, 8, F_SSE, 0 },
    { "maskmovdqu", k_maskmovdqu_all, 1, F_SSE2, 0 },
    { "movntdqa", k_movntdqa_sse41, 16, F_SSE41, 1 },
    { "vmovntdq_xmm", k_vmovntdq_xmm_avx, 16, F_AVX, 0 },
    { "vmovntdq_ymm", k_vmovntdq_ymm_avx, 32, F_AVX, 0 },
    { "vmovntps_ymm", k_vmovntps_ymm_avx, 32, F_AVX, 0 },
    { "vmovntpd_ymm", k_vmovntpd_ymm_avx, 32, F_AVX, 0 },
    { "vmovntdqa_ymm", k_vmovntdqa_ymm_avx2, 32, F_AVX2, 1 },
    { "vmovntdq_zmm", k_vmovntdq_zmm_512, 64, F_AVX512F, 0 },
    { "vmovntdqa_zmm", k_vmovntdqa_zmm_512, 64, F_AVX512F, 1 },
};
enum { K_MOVNTI32, K_MOVNTI64, K_MOVNTDQ, K_MOVNTPS, K_MOVNTPD, K_MOVNTQ, K_MASKMOVDQU, K_MOVNTDQA,
       K_VMOVNTDQ_X, K_VMOVNTDQ_Y, K_VMOVNTPS_Y, K_VMOVNTPD_Y, K_VMOVNTDQA_Y, K_VMOVNTDQ_Z, K_VMOVNTDQA_Z };
static int k_avail(const sm_kernel *k) { return k->fn && cpu_has((t_feature)k->feat); }
#define NEED_K(k) do { if (!(k)->fn) SKIP("%s: x64-only instruction form", (k)->name); \
    if (!cpu_has((t_feature)(k)->feat)) SKIP("%s needs %s (cpuid bit %d)", (k)->name, cpu_feature_name((t_feature)(k)->feat), cpu_hw((t_feature)(k)->feat)); } while (0)

/* arbitrary size/alignment copy built on a kernel; ends with SFENCE */
static void nt_memcpy(const sm_kernel *k, void *d, const void *s, size_t n) {
    uint8_t *dp = (uint8_t *)d; const uint8_t *sp = (const uint8_t *)s;
    uintptr_t a = k->src_aligned ? (uintptr_t)sp : (uintptr_t)dp;
    size_t head = (size_t)((k->align - (a & (uintptr_t)(k->align - 1))) & (uintptr_t)(k->align - 1));
    if (head > n) head = n;
    memcpy(dp, sp, head); dp += head; sp += head; n -= head;
    size_t body = n & ~(size_t)63;
    if (body) k->fn(dp, sp, body);
    memcpy(dp + body, sp + body, n - body);
    sm_sfence();
}

/* per-kernel store/load correctness: sizes, offsets, no overrun, self-visibility before SFENCE */
static void nt_kernel_case(int ki) {
    const sm_kernel *k = &g_k[ki];
    NEED_K(k);
    size_t cap = 256 * 1024;
    uint8_t *src = sm_valloc(cap + 4096), *dst = sm_valloc(cap + 4096);
    REQUIRE(src && dst);
    t_rng r; t_rng_seed(&r, 0xA11CE000u + (unsigned)ki);
    static const size_t sizes[] = { 64, 128, 192, 4096, 4160, 65536, 262144 };
    for (size_t si = 0; si < T_COUNTOF(sizes); si++) {
        for (int oi = 0; oi < 3; oi++) {
            size_t n = sizes[si];
            size_t off = (size_t)(oi * (k->align < 64 ? k->align : 64)) % 1024;   /* aligned for the kernel */
            if (off + n + 64 > cap + 4096) off = 0;
            t_rng_fill(&r, src, n + off + 64);
            memset(dst, 0xCC, n + off + 128);
            uint8_t *d = dst + off; const uint8_t *s = src + off;
            k->fn(d, s, n);
            /* same processor: own NT stores are visible to its own later loads without a fence */
            int self = memcmp(d, s, n) == 0;
            sm_sfence();
            CHECKF(self, "%s n=%llu off=%llu: own NT stores not visible to own loads before SFENCE", k->name, (unsigned long long)n, (unsigned long long)off);
            CHECK_MEM(d, s, n);
            int under = 0, over = 0;
            for (size_t i = 0; i < off; i++) under |= dst[i] != 0xCC;
            for (size_t i = 0; i < 64; i++) over |= d[n + i] != 0xCC;
            CHECKF(!under && !over, "%s n=%llu off=%llu: bytes outside the destination modified (before=%d after=%d)",
                   k->name, (unsigned long long)n, (unsigned long long)off, under, over);
        }
    }
    /* special values: all-zero, all-ones, NaN/denormal bit patterns pass bit-exactly through ps/pd stores */
    static const uint32_t specials[] = { 0, 0xFFFFFFFFu, 0x7FC00001u, 0xFF800000u, 0x00000001u, 0x80000000u, 0x7F800001u, 0x007FFFFFu };
    uint32_t *s32 = (uint32_t *)src;
    for (int i = 0; i < 1024; i++) s32[i] = specials[(i + i / 8) % T_COUNTOF(specials)];
    memset(dst, 0, 4096);
    k->fn(dst, src, 4096); sm_sfence();
    CHECK_MEM(dst, src, 4096);
    sm_vfree(src); sm_vfree(dst);
}
TEST(nt_copy_movnti32)      { nt_kernel_case(K_MOVNTI32); }
TEST(nt_copy_movnti64)      { nt_kernel_case(K_MOVNTI64); }
TEST(nt_copy_movntdq)       { nt_kernel_case(K_MOVNTDQ); }
TEST(nt_copy_movntps)       { nt_kernel_case(K_MOVNTPS); }
TEST(nt_copy_movntpd)       { nt_kernel_case(K_MOVNTPD); }
TEST(nt_copy_movntq_mmx)    { nt_kernel_case(K_MOVNTQ); }
TEST(nt_copy_maskmovdqu)    { nt_kernel_case(K_MASKMOVDQU); }
TEST(nt_copy_movntdqa)      { nt_kernel_case(K_MOVNTDQA); }
TEST(nt_copy_vmovntdq_xmm)  { nt_kernel_case(K_VMOVNTDQ_X); }
TEST(nt_copy_vmovntdq_ymm)  { nt_kernel_case(K_VMOVNTDQ_Y); }
TEST(nt_copy_vmovntps_ymm)  { nt_kernel_case(K_VMOVNTPS_Y); }
TEST(nt_copy_vmovntpd_ymm)  { nt_kernel_case(K_VMOVNTPD_Y); }
TEST(nt_copy_vmovntdqa_ymm) { nt_kernel_case(K_VMOVNTDQA_Y); }
TEST(nt_copy_vmovntdq_zmm)  { nt_kernel_case(K_VMOVNTDQ_Z); }
TEST(nt_copy_vmovntdqa_zmm) { nt_kernel_case(K_VMOVNTDQA_Z); }

TEST(nt_memcpy_edge_sizes_offsets) {
    NEED(F_SSE2);
    static const size_t sizes[] = { 0, 1, 3, 15, 16, 17, 31, 33, 63, 64, 65, 127, 128, 129, 255, 1000, 4095, 4096, 4097, 65537 };
    static const int doffs[] = { 0, 1, 7, 15, 31, 63 }, soffs[] = { 0, 3, 16 };
    uint8_t *src = sm_valloc(80 * 1024), *dst = sm_valloc(80 * 1024);
    REQUIRE(src && dst);
    t_rng r; t_rng_seed(&r, 77);
    t_rng_fill(&r, src, 80 * 1024);
    int nk = 0;
    for (size_t ki = 0; ki < T_COUNTOF(g_k); ki++) {
        const sm_kernel *k = &g_k[ki];
        if (!k_avail(k)) continue;
        nk++;
        int bad = 0; char first[160] = "";
        for (size_t si = 0; si < T_COUNTOF(sizes); si++)
            for (size_t di = 0; di < T_COUNTOF(doffs); di++)
                for (size_t oi = 0; oi < T_COUNTOF(soffs); oi++) {
                    size_t n = sizes[si]; uint8_t *d = dst + 64 + doffs[di]; const uint8_t *s = src + 128 + soffs[oi];
                    memset(dst, 0x5A, n + 256);
                    nt_memcpy(k, d, s, n);
                    int ok = memcmp(d, s, n) == 0;
                    for (int i = 0; i < 64 + doffs[di]; i++) ok &= dst[i] == 0x5A;
                    for (int i = 0; i < 64; i++) ok &= d[n + i] == 0x5A;
                    if (!ok && !bad++) snprintf(first, sizeof first, "n=%llu doff=%d soff=%d", (unsigned long long)n, doffs[di], soffs[oi]);
                }
        CHECKF(bad == 0, "%s: %d size/offset combinations wrong; first %s", k->name, bad, first);
    }
    /* destination ending exactly at a PAGE_NOACCESS page: no overrun (would fault) */
    pr_begin();
    void *gb = NULL;
    uint8_t *g = (uint8_t *)t_guarded_alloc(8192, &gb);
    REQUIRE(g);
    for (size_t ki = 0; ki < T_COUNTOF(g_k); ki++) {
        const sm_kernel *k = &g_k[ki];
        if (!k_avail(k) || k->src_aligned) continue;
        static const size_t ends[] = { 64, 4096, 8192, 100, 8191 };
        for (size_t e = 0; e < T_COUNTOF(ends); e++) {
            uint8_t *d = g + 8192 - ends[e];
            g_pr.resume = 0; g_pr.nexc = 0;
            nt_memcpy(k, d, src, ends[e]);
            CHECKF(memcmp(d, src, ends[e]) == 0, "%s: copy ending at guard page n=%llu wrong", k->name, (unsigned long long)ends[e]);
        }
    }
    t_guarded_free(gb);
    pr_end();
    t_info("kernels", "%d", nk);
    sm_vfree(src); sm_vfree(dst);
}

/* 32 MB copies: every available NT kernel equals memcpy */
TEST(nt_copy_32mb_equals_memcpy) {
    NEED(F_SSE2);
    size_t n = 32u << 20;
    uint8_t *src = sm_valloc(n), *dst = sm_valloc(n), *ref = sm_valloc(n);
    if (!src || !dst || !ref) { sm_vfree(src); sm_vfree(dst); sm_vfree(ref); SKIP("cannot allocate 3 x 32 MB"); }
    t_rng r; t_rng_seed(&r, 0x32);
    t_rng_fill(&r, src, n);
    memcpy(ref, src, n);
    int done = 0;
    for (size_t ki = 0; ki < T_COUNTOF(g_k); ki++) {
        const sm_kernel *k = &g_k[ki];
        if (!k_avail(k) || ki == K_MOVNTQ || ki == K_MASKMOVDQU) continue;
        memset(dst, 0, n);
        k->fn(dst, src, n);
        sm_sfence();
        int eq = memcmp(dst, ref, n) == 0;
        CHECKF(eq, "%s: 32 MB NT copy differs from memcpy", k->name);
        if (!eq) CHECK_MEM(dst, ref, n);
        done++;
    }
    t_info("kernels", "%d", done);
    sm_vfree(src); sm_vfree(dst); sm_vfree(ref);
}

typedef struct bench_ctx { const sm_kernel *k; uint8_t *d, *s; size_t n; } bench_ctx;
static void bench_nt(void *c) { bench_ctx *b = (bench_ctx *)c; b->k->fn(b->d, b->s, b->n); sm_sfence(); }
static void bench_memcpy(void *c) { bench_ctx *b = (bench_ctx *)c; memcpy(b->d, b->s, b->n); t_sink_ptr(b->d); }
TEST(nt_copy_bandwidth_perf) {
    NEED(F_SSE2);
    size_t n = 16u << 20;
    uint8_t *src = sm_valloc(n), *dst = sm_valloc(n);
    if (!src || !dst) { sm_vfree(src); sm_vfree(dst); SKIP("cannot allocate 2 x 16 MB"); }
    t_rng r; t_rng_seed(&r, 5); t_rng_fill(&r, src, n);
    bench_ctx c = { NULL, dst, src, n };
    t_stats s;
    t_bench(bench_memcpy, &c, 1, 5, 0.0, &s);
    CHECK(memcmp(dst, src, n) == 0);
    t_perf("memcpy_16mb", (double)n / s.median / 1e9, "GB/s");
    static const int ks[] = { K_MOVNTDQ, K_VMOVNTDQ_Y, K_VMOVNTDQ_Z, K_MOVNTI32, K_MOVNTDQA };
    for (size_t i = 0; i < T_COUNTOF(ks); i++) {
        c.k = &g_k[ks[i]];
        if (!k_avail(c.k)) continue;
        memset(dst, 0, n);
        t_bench(bench_nt, &c, 1, 5, 0.0, &s);
        int ok = memcmp(dst, src, n) == 0;
        CHECKF(ok, "%s: benchmark copy wrong", c.k->name);
        char m[64]; snprintf(m, sizeof m, "%s_16mb", c.k->name);
        if (ok) t_perf(m, (double)n / s.median / 1e9, "GB/s");
    }
    sm_vfree(src); sm_vfree(dst);
}

/* program order of normal and NT stores from one thread (self-consistency) */
TEST(nt_program_order_same_thread) {
    NEED(F_SSE2);
    uint8_t *buf = sm_valloc(4096), shadow[256];
    REQUIRE(buf);
    uint32_t *w = (uint32_t *)buf;
    /* 1. normal store, NT store, normal load */
    memset(buf, 0x11, 64);
    _mm_stream_si32((int *)w, 0x22222222);
    CHECK_EQ(*(volatile uint32_t *)w, 0x22222222u);
    /* 2. NT 16 bytes, then a normal byte store into the middle, then SFENCE */
    _mm_stream_si128((__m128i *)(buf + 16), _mm_set1_epi8(0x44));
    ((volatile uint8_t *)buf)[20] = 0x33;
    sm_sfence();
    for (int i = 16; i < 32; i++) CHECKF(buf[i] == (i == 20 ? 0x33 : 0x44), "byte %d = 0x%02x", i, buf[i]);
    /* 3. random interleaving of normal and NT stores of all widths against a shadow copy */
    t_rng r; t_rng_seed(&r, 1234);
    memset(buf, 0, 256); memset(shadow, 0, 256);
    int bad = -1;
    for (int it = 0; it < 4000 && bad < 0; it++) {
        int kind = (int)t_rng_range(&r, 4);
        uint32_t v = t_rng_u32(&r);
        if (kind == 0) { size_t o = t_rng_range(&r, 256 - 4); memcpy(buf + o, &v, 4); memcpy(shadow + o, &v, 4); }
        else if (kind == 1) { size_t o = t_rng_range(&r, 64) * 4; _mm_stream_si32((int *)(buf + o), (int)v); memcpy(shadow + o, &v, 4); }
        else if (kind == 2) {
            size_t o = t_rng_range(&r, 16) * 16; __m128i x = _mm_set1_epi32((int)v);
            _mm_stream_si128((__m128i *)(buf + o), x); _mm_storeu_si128((__m128i *)(shadow + o), x);
        } else { size_t o = t_rng_range(&r, 256); buf[o] = (uint8_t)v; shadow[o] = (uint8_t)v; }
        sm_cbar();
        if (memcmp(buf, shadow, 256)) bad = it;
    }
    sm_sfence();
    CHECKF(bad < 0, "own loads disagreed with program order after op %d", bad);
    /* 4. NT store, CLFLUSH, read back */
    if (cpu_has(F_CLFSH)) {
        _mm_stream_si128((__m128i *)(buf + 512), _mm_set1_epi32(0x0BADF00D));
        _mm_clflush(buf + 512);
        sm_mfence();
        CHECK_EQ(((volatile uint32_t *)buf)[128 + 3], 0x0BADF00Du);
    }
    sm_vfree(buf);
}

/* NT stores not yet fenced, then a context switch / migration: the thread must see its own data on the
 * new processor (the OS context switch must make them globally visible). */
TEST(nt_stores_survive_context_switch_and_migration) {
    NEED(F_SSE2);
    int cpus[64]; int nc = sm_cpu_list(cpus, 64);
    uint8_t *buf = sm_valloc(65536), *src = sm_valloc(65536);
    REQUIRE(buf && src);
    t_rng r; t_rng_seed(&r, 99);
    int bad = 0, migr = 0;
    for (int it = 0; it < 200; it++) {
        t_rng_fill(&r, src, 65536);
        k_movntdq(buf, src, 32768);
        if (nc > 1) { if (SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpus[it % nc])) migr++; }
        if (it % 3 == 0) Sleep(0); else if (it % 3 == 1) SwitchToThread(); else Sleep(1 * (it % 7 == 2));
        k_movntdq(buf + 32768, src + 32768, 32768);
        if (it % 2 && nc > 1) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpus[(it + 1) % nc]);
        Sleep(0);
        if (memcmp(buf, src, 65536)) bad++;    /* deliberately no SFENCE before the check */
        sm_sfence();
    }
    {   DWORD_PTR pm, sm; if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) SetThreadAffinityMask(GetCurrentThread(), pm); }
    CHECKF(bad == 0, "%d/200 iterations saw stale data after a context switch/migration", bad);
    t_info("migrations", "%d", migr);
    sm_vfree(buf); sm_vfree(src);
}

TEST(nt_movnti_unaligned_and_page_crossing) {
    NEED(F_SSE2);
    static const DWORD prot[2] = { PAGE_READWRITE, PAGE_READWRITE };
    sm_region rg; REQUIRE(sm_region_alloc(&rg, 2, prot));
    pr_begin();
    uint8_t *b = sm_page(&rg, 1);
    for (int off = -7; off <= 3; off++) {
        uint8_t *p = b + off;
        uint32_t v = 0xA1B2C3D4u ^ (uint32_t)off;
        __asm__ volatile(PR_ARM "movnti %[v], (%[p])" PR_DONE : PR_OUT : [p] "r"(p), [v] "r"(v) : "memory" PR_CLOB_C);
        PR_EXPECT_NONE("movnti32 at page offset %d", off);
        sm_sfence();
        uint32_t got; memcpy(&got, p, 4);
        CHECKF(got == v, "movnti32 at %d: 0x%08x want 0x%08x", off, got, v);
    }
#ifdef T_X64
    for (int off = -15; off <= 1; off += 2) {
        uint8_t *p = b + off;
        uint64_t v = 0x0123456789ABCDEFull ^ (uint64_t)(int64_t)off;
        __asm__ volatile(PR_ARM "movnti %[v], (%[p])" PR_DONE : PR_OUT : [p] "r"(p), [v] "r"(v) : "memory" PR_CLOB_C);
        PR_EXPECT_NONE("movnti64 at page offset %d", off);
        sm_sfence();
        uint64_t got; memcpy(&got, p, 8);
        CHECKF(got == v, "movnti64 at %d: 0x%016llx", off, (unsigned long long)got);
    }
#endif
    /* cache-line crossing within a page */
    for (int off = 57; off < 64; off++) {
        uint32_t v = 0x600DD00Du + (uint32_t)off;
        _mm_stream_si32((int *)(b + 1024 + off), (int)v);
        sm_sfence();
        uint32_t got; memcpy(&got, b + 1024 + off, 4);
        CHECKF(got == v, "movnti32 line-crossing at %d", off);
    }
    pr_end();
    sm_region_free(&rg);
}

/* AMD SSE4A scalar NT stores: MOVNTSS m32 / MOVNTSD m64 store only the low element (APM Vol 4). */
TEST(nt_movntss_movntsd_sse4a) {
    NEED(F_SSE4A);
    uint8_t *b = sm_valloc(4096); REQUIRE(b);
    pr_begin();
    int unaligned_faults = 0;
    for (int off = 0; off < 16; off++) {
        uint8_t *p = b + 256 + off;
        memset(b, 0xA5, 1024);
        __attribute__((aligned(16))) uint32_t v4[4] = { 0x11223344u + (uint32_t)off, 0xEEEEEEEEu, 0xDDDDDDDDu, 0xCCCCCCCCu };
        __asm__ volatile("movdqa %[v], %%xmm0\n\t" PR_ARM "movntss %%xmm0, (%[p])" PR_DONE
                         : PR_OUT : [p] "r"(p), [v] "m"(v4) : "xmm0", "memory" PR_CLOB_C);
        int fault_ss = g_pr.nexc != 0; pr_clear();
        sm_sfence();
        uint32_t got; memcpy(&got, p, 4);
        int ok_ss = !fault_ss && got == v4[0] && p[-1] == 0xA5 && p[4] == 0xA5;
        uint8_t *q = b + 512 + off;
        __attribute__((aligned(16))) uint64_t v2[2] = { 0x0102030405060708ull ^ (uint64_t)off, 0xFFFFFFFFFFFFFFFFull };
        __asm__ volatile("movdqa %[v], %%xmm0\n\t" PR_ARM "movntsd %%xmm0, (%[p])" PR_DONE
                         : PR_OUT : [p] "r"(q), [v] "m"(v2) : "xmm0", "memory" PR_CLOB_C);
        int fault_sd = g_pr.nexc != 0; pr_clear();
        sm_sfence();
        uint64_t g2; memcpy(&g2, q, 8);
        int ok_sd = !fault_sd && g2 == v2[0] && q[-1] == 0xA5 && q[8] == 0xA5;
        if (off % 8 == 0) {
            CHECKF(ok_ss, "movntss at +%d: fault=%d value 0x%08x want 0x%08x (or neighbours modified)", off, fault_ss, got, v4[0]);
            CHECKF(ok_sd, "movntsd at +%d: fault=%d value 0x%016llx (or neighbours modified)", off, fault_sd, (unsigned long long)g2);
        } else {
            unaligned_faults += fault_ss + fault_sd;
            if (!fault_ss) CHECKF(ok_ss, "movntss at +%d (no fault) stored 0x%08x want 0x%08x", off, got, v4[0]);
            if (!fault_sd) CHECKF(ok_sd, "movntsd at +%d (no fault) stored 0x%016llx", off, (unsigned long long)g2);
        }
    }
    pr_end();
    t_note("movntss/movntsd at unaligned addresses: %d exceptions", unaligned_faults);
    sm_vfree(b);
}

TEST(nt_misaligned_store_load_faults) {
    NEED(F_SSE2);
    /* MOVNTDQ/MOVNTPS/MOVNTPD m128: #GP(0) if not 16-byte aligned (SDM Vol 2, exceptions type 1) */
    expect_av("movntdq_mis", -1, 0);
    expect_av("movntps_mis", -1, 0);
    expect_av("movntpd_mis", -1, 0);
    expect_av("movntdqa_mis", -1, 0);
}
TEST(nt_vex_evex_misaligned_faults) {
    NEED(F_AVX);
    /* VMOVNTDQ/VMOVNTDQA keep the alignment requirement of the vector length even in VEX/EVEX form */
    expect_av("vmovntdq_xmm_mis", -1, 0);
    expect_av("vmovntdq_ymm_mis", -1, 0);
    expect_av("vmovntdqa_ymm_mis", -1, 0);
    if (t_sde_model() && cpu_has(F_AVX512F))   /* emulator artifact, README "Emulated validation" */
        t_expect_fail("Intel SDE (%s) does not raise #GP for misaligned emulated EVEX VMOVNTDQ", t_sde_model());
    expect_av("vmovntdq_zmm_mis", -1, 0);
}
TEST(nt_store_to_noaccess_and_readonly_faults) {
    NEED(F_SSE2);
    expect_av("movnti_noacc", 1, 1);
    expect_av("movntdq_ro", 1, 1);
}
TEST(nt_load_from_noaccess_faults) {
    NEED(F_SSE41);
    expect_av("movntdqa_noacc", 0, 1);
}

/* ======================================================================== */
/* cross-thread message passing with NT stores + SFENCE                       */
/* ======================================================================== */
typedef struct mp_ctx {
    const sm_kernel *k; uint8_t *shared, *src; size_t n; int iters; double deadline;
    __attribute__((aligned(64))) volatile LONG seq;
    __attribute__((aligned(64))) volatile LONG ack;
    volatile LONG abort; LONG bad, first_bad_it, done;
} mp_ctx;
static inline uint32_t mp_pat(uint32_t it, uint32_t j) { return (it * 0x9E3779B1u) ^ (j * 0x85EBCA6Bu) ^ 0xA5A5A5A5u; }
static int mp_wait(volatile LONG *v, LONG want, mp_ctx *c) {
    unsigned spins = 0;
    while (*v != want) {
        if (c->abort) return 0;
        sm_pause();
        if ((++spins & 1023) == 0) { if (t_now_ms() > c->deadline) { c->abort = 1; return 0; } SwitchToThread(); }
    }
    sm_cbar();
    return 1;
}
static DWORD WINAPI mp_producer(void *arg) {
    mp_ctx *c = (mp_ctx *)arg;
    for (LONG it = 1; it <= c->iters; it++) {
        if (!mp_wait(&c->ack, it - 1, c)) return 0;
        uint32_t *s = (uint32_t *)c->src;
        for (size_t j = 0; j < c->n / 4; j++) s[j] = mp_pat((uint32_t)it, (uint32_t)j);
        c->k->fn(c->shared, c->src, c->n);
        sm_sfence();                         /* orders the NT stores before the flag store */
        *(volatile LONG *)&c->seq = it;      /* plain store (no locked op that would also drain) */
    }
    return 0;
}
static DWORD WINAPI mp_consumer(void *arg) {
    mp_ctx *c = (mp_ctx *)arg;
    for (LONG it = 1; it <= c->iters; it++) {
        if (!mp_wait(&c->seq, it, c)) return 0;
        const volatile uint32_t *d = (const volatile uint32_t *)c->shared;
        int bad = 0;
        for (size_t j = 0; j < c->n / 4; j++) bad |= d[j] != mp_pat((uint32_t)it, (uint32_t)j);
        if (bad && !c->bad++) c->first_bad_it = it;
        c->done = it;
        sm_cbar();
        *(volatile LONG *)&c->ack = it;
    }
    return 0;
}
static void mp_case(int ki) {
    const sm_kernel *k = &g_k[ki];
    NEED_K(k);
    static mp_ctx c;
    memset((void *)&c, 0, sizeof c);
    c.k = k; c.n = 4096; c.iters = 400;
    c.shared = sm_valloc(c.n); c.src = sm_valloc(c.n);
    REQUIRE(c.shared && c.src);
    c.deadline = t_now_ms() + 4000;
    HANDLE h[2];
    h[0] = CreateThread(NULL, 65536, mp_consumer, &c, 0, NULL);
    h[1] = h[0] ? CreateThread(NULL, 65536, mp_producer, &c, 0, NULL) : NULL;
    if (!h[0] || !h[1]) {
        c.abort = 1;
        if (h[0]) { WaitForSingleObject(h[0], 5000); CloseHandle(h[0]); }
        sm_vfree(c.shared); sm_vfree(c.src);
        SKIP("cannot create threads (%lu)", GetLastError());
    }
    DWORD w = WaitForMultipleObjects(2, h, TRUE, 10000);
    if (w == WAIT_TIMEOUT) {
        c.abort = 1;
        /* a thread that ignores the abort flag is hung (not slow): it still uses c and the buffers */
        if (WaitForMultipleObjects(2, h, TRUE, 5000) == WAIT_TIMEOUT) t_retire_process("simd_memory: message-passing thread hung after abort");
    }
    CloseHandle(h[0]); CloseHandle(h[1]);
    CHECKF(c.bad == 0, "%s: consumer saw stale/partial data in %ld of %ld messages (first at %ld)", k->name, c.bad, c.done, c.first_bad_it);
    if (c.done < 20) t_note("%s: only %ld messages exchanged before the deadline (slow host; correctness is judged on data only)", k->name, c.done);
    if (c.done < c.iters) t_note("%s: %ld/%d messages before the deadline", k->name, c.done, c.iters);
    t_info("messages", "%ld", c.done);
    sm_vfree(c.shared); sm_vfree(c.src);
}
TEST(mp_movnti32_sfence_flag)     { mp_case(K_MOVNTI32); }
TEST(mp_movntdq_sfence_flag)      { mp_case(K_MOVNTDQ); }
TEST(mp_movntps_sfence_flag)      { mp_case(K_MOVNTPS); }
TEST(mp_maskmovdqu_sfence_flag)   { mp_case(K_MASKMOVDQU); }
TEST(mp_vmovntdq_ymm_sfence_flag) { mp_case(K_VMOVNTDQ_Y); }
TEST(mp_vmovntdq_zmm_sfence_flag) { mp_case(K_VMOVNTDQ_Z); }

/* NT stores WITHOUT SFENCE, then SetEvent; another thread (on another CPU when possible) wakes from
 * WaitForSingleObject and must see the data. Contract: Windows synchronization functions (SetEvent,
 * WaitForSingleObject, ...) are full memory barriers (MSDN "Synchronization and Multiprocessor Issues"), and
 * the processor drains write-combining buffers on a LOCKed instruction, a serializing instruction or an
 * interrupt (SDM Vol 3 11.3.1), so a signalling path that goes through the OS cannot overtake the NT data. */
typedef struct ev_ctx { HANDLE data, ack; uint8_t *buf; size_t n; int iters; volatile LONG bad, first_bad, seen, quit; } ev_ctx;
static DWORD WINAPI ev_consumer(void *arg) {
    ev_ctx *c = (ev_ctx *)arg;
    for (int it = 1; it <= c->iters; it++) {
        if (WaitForSingleObject(c->data, 3000) != WAIT_OBJECT_0 || c->quit) return 0;
        const volatile uint32_t *d = (const volatile uint32_t *)c->buf;
        int bad = 0;
        for (size_t j = 0; j < c->n / 4; j++) bad |= d[j] != mp_pat((uint32_t)it, (uint32_t)j);
        if (bad && !c->bad++) c->first_bad = it;
        c->seen = it;
        SetEvent(c->ack);
    }
    return 0;
}
static void ev_case(int ki) {
    const sm_kernel *k = &g_k[ki];
    NEED_K(k);
    static ev_ctx c;
    memset((void *)&c, 0, sizeof c);
    c.n = 16384; c.iters = 300;
    c.buf = sm_valloc(c.n); uint8_t *src = sm_valloc(c.n);
    c.data = CreateEventA(NULL, FALSE, FALSE, NULL); c.ack = CreateEventA(NULL, FALSE, FALSE, NULL);
    REQUIRE(c.buf && src && c.data && c.ack);
    int cpus[64]; int nc = sm_cpu_list(cpus, 64);
    uintptr_t prev = nc >= 2 ? t_pin_thread(cpus[0]) : 0;
    HANDLE h = CreateThread(NULL, 65536, ev_consumer, &c, CREATE_SUSPENDED, NULL);
    if (!h) { t_unpin_thread(prev); CloseHandle(c.data); CloseHandle(c.ack); sm_vfree(c.buf); sm_vfree(src); SKIP("cannot create a thread (%lu)", GetLastError()); }
    if (nc >= 2) SetThreadAffinityMask(h, (DWORD_PTR)1 << cpus[1]);
    ResumeThread(h);
    int sent = 0;
    for (int it = 1; it <= c.iters; it++) {
        uint32_t *s = (uint32_t *)src;
        for (size_t j = 0; j < c.n / 4; j++) s[j] = mp_pat((uint32_t)it, (uint32_t)j);
        k->fn(c.buf, src, c.n);          /* deliberately no SFENCE */
        SetEvent(c.data);
        sent = it;
        if (WaitForSingleObject(c.ack, 3000) != WAIT_OBJECT_0) break;
    }
    c.quit = 1; SetEvent(c.data);
    WaitForSingleObject(h, 5000); CloseHandle(h);
    t_unpin_thread(prev);
    CHECKF(c.seen == c.iters, "%s: consumer acknowledged %ld of %d messages (sent %d)", k->name, c.seen, c.iters, sent);
    CHECKF(c.bad == 0, "%s: consumer saw stale NT data after SetEvent/WaitForSingleObject in %ld messages (first %ld)",
           k->name, c.bad, c.first_bad);
    t_info("cross_cpu", "%d", nc >= 2);
    CloseHandle(c.data); CloseHandle(c.ack); sm_vfree(c.buf); sm_vfree(src);
}
TEST(ev_movntdq_nofence_setevent)      { ev_case(K_MOVNTDQ); }
TEST(ev_movnti32_nofence_setevent)     { ev_case(K_MOVNTI32); }
TEST(ev_vmovntdq_ymm_nofence_setevent) { ev_case(K_VMOVNTDQ_Y); }

/* ======================================================================== */
/* litmus tests (two worker threads, main thread sequences iterations)        */
/* ======================================================================== */
enum { LIT_SB_PLAIN, LIT_SB_MFENCE, LIT_SB_XCHG, LIT_SB_LOCKOR, LIT_SB_SFENCE, LIT_SB_LFENCE,
       LIT_MP_PLAIN, LIT_MP_NT_SFENCE, LIT_MP_NT_NOFENCE, LIT_LB_PLAIN,
       LIT_MP_NT_MFENCE, LIT_MP_NT_LOCK, LIT_MP_NTDQ_SFENCE, LIT_MP_NT_XCHGFLAG };
typedef struct lit_ctx {
    int variant, iters; volatile LONG stop;
    __attribute__((aligned(64))) volatile LONG gen;
    __attribute__((aligned(64))) volatile LONG done0;
    __attribute__((aligned(64))) volatile LONG done1;
    __attribute__((aligned(64))) volatile LONG x;
    __attribute__((aligned(64))) volatile LONG y;
    __attribute__((aligned(64))) volatile LONG r0;
    __attribute__((aligned(64))) volatile LONG r1;
} lit_ctx;
static lit_ctx g_lit;
static inline void l_st1(volatile LONG *p) { __asm__ volatile("movl $1, %0" : "=m"(*p) :: "memory"); }
static inline LONG l_ld(volatile LONG *p) { LONG r; __asm__ volatile("movl %1, %0" : "=r"(r) : "m"(*p) : "memory"); return r; }
static LONG lit_op(lit_ctx *c, int me) {
    volatile LONG *mine = me ? &c->y : &c->x, *other = me ? &c->x : &c->y;
    LONG r = 0, one = 1;
    switch (c->variant) {
    case LIT_SB_PLAIN:  l_st1(mine); r = l_ld(other); break;
    case LIT_SB_MFENCE: l_st1(mine); __asm__ volatile("mfence" ::: "memory"); r = l_ld(other); break;
    case LIT_SB_SFENCE: l_st1(mine); __asm__ volatile("sfence" ::: "memory"); r = l_ld(other); break;
    case LIT_SB_LFENCE: l_st1(mine); __asm__ volatile("lfence" ::: "memory"); r = l_ld(other); break;
    case LIT_SB_XCHG:   __asm__ volatile("xchgl %0, %1" : "+r"(one), "+m"(*mine) :: "memory"); r = l_ld(other); break;
    case LIT_SB_LOCKOR: l_st1(mine); __asm__ volatile("lock orl $0, (%%" SM_SP ")" ::: "memory", "cc"); r = l_ld(other); break;
    case LIT_MP_PLAIN: case LIT_MP_NT_SFENCE: case LIT_MP_NT_NOFENCE:
    case LIT_MP_NT_MFENCE: case LIT_MP_NT_LOCK: case LIT_MP_NTDQ_SFENCE: case LIT_MP_NT_XCHGFLAG:
        if (me == 0) {   /* writer: data = x, flag = y */
            if (c->variant == LIT_MP_PLAIN) l_st1(&c->x);
            else if (c->variant == LIT_MP_NTDQ_SFENCE)   /* x is 64-byte aligned with 60 spare bytes after it */
                __asm__ volatile("movd %1, %%xmm0\n\tmovntdq %%xmm0, %0" : "=m"(*(volatile __m128i *)&c->x) : "r"(one) : "xmm0", "memory");
            else __asm__ volatile("movnti %1, %0" : "=m"(c->x) : "r"(one) : "memory");
            if (c->variant == LIT_MP_NT_SFENCE || c->variant == LIT_MP_NTDQ_SFENCE) __asm__ volatile("sfence" ::: "memory");
            if (c->variant == LIT_MP_NT_MFENCE) __asm__ volatile("mfence" ::: "memory");
            if (c->variant == LIT_MP_NT_LOCK) __asm__ volatile("lock orl $0, (%%" SM_SP ")" ::: "memory", "cc");
            if (c->variant == LIT_MP_NT_XCHGFLAG) __asm__ volatile("xchgl %0, %1" : "+r"(one), "+m"(c->y) :: "memory");
            else l_st1(&c->y);
        } else {         /* reader: flag then data */
            LONG f = l_ld(&c->y), d = l_ld(&c->x);
            r = f | (d << 1);
        }
        break;
    case LIT_LB_PLAIN: r = l_ld(other); l_st1(mine); break;
    }
    return r;
}
static DWORD WINAPI lit_thread(void *arg) {
    int me = (int)(intptr_t)arg; lit_ctx *c = &g_lit;
    for (LONG i = 1; i <= c->iters; i++) {
        unsigned spins = 0;
        while (c->gen < i) { if (c->stop) return 0; sm_pause(); if ((++spins & 4095) == 0) SwitchToThread(); }
        LONG r = lit_op(c, me);
        if (me) c->r1 = r; else c->r0 = r;
        sm_cbar();
        if (me) c->done1 = i; else c->done0 = i;
    }
    return 0;
}
/* returns iterations done; *forbidden = outcomes the model forbids for the variant (or relaxed for _info) */
static int lit_run(int variant, int iters, long *forbidden, long *relaxed_any) {
    lit_ctx *c = &g_lit;
    memset((void *)c, 0, sizeof *c);
    c->variant = variant; c->iters = iters;
    HANDLE h[2]; int cpus[64]; int nc = sm_cpu_list(cpus, 64);
    h[0] = CreateThread(NULL, 65536, lit_thread, (void *)(intptr_t)0, CREATE_SUSPENDED, NULL);
    h[1] = h[0] ? CreateThread(NULL, 65536, lit_thread, (void *)(intptr_t)1, CREATE_SUSPENDED, NULL) : NULL;
    if (!h[0] || !h[1]) {
        if (h[0]) { TerminateThread(h[0], 0); CloseHandle(h[0]); }
        return -1;
    }
    if (nc >= 3) { SetThreadAffinityMask(h[0], (DWORD_PTR)1 << cpus[1]); SetThreadAffinityMask(h[1], (DWORD_PTR)1 << cpus[2]); }
    ResumeThread(h[0]); ResumeThread(h[1]);
    double deadline = t_now_ms() + 2000;
    int done = 0; *forbidden = 0; *relaxed_any = 0;
    for (LONG i = 1; i <= iters; i++) {
        c->x = 0; c->y = 0; c->r0 = -1; c->r1 = -1;
        MemoryBarrier();
        c->gen = i;
        unsigned spins = 0; int to = 0;
        while (c->done0 < i || c->done1 < i) {
            sm_pause();
            if ((++spins & 4095) == 0) { if (t_now_ms() > deadline) { to = 1; break; } SwitchToThread(); }
        }
        if (to) break;
        sm_cbar();
        LONG r0 = c->r0, r1 = c->r1;
        int bad = 0;
        switch (variant) {
        case LIT_SB_PLAIN: case LIT_SB_MFENCE: case LIT_SB_SFENCE: case LIT_SB_LFENCE: case LIT_SB_XCHG: case LIT_SB_LOCKOR:
            bad = r0 == 0 && r1 == 0; break;
        case LIT_MP_PLAIN: case LIT_MP_NT_SFENCE: case LIT_MP_NT_NOFENCE:
        case LIT_MP_NT_MFENCE: case LIT_MP_NT_LOCK: case LIT_MP_NTDQ_SFENCE: case LIT_MP_NT_XCHGFLAG:
            bad = (r1 & 1) && !(r1 & 2); break;
        case LIT_LB_PLAIN: bad = r0 == 1 && r1 == 1; break;
        }
        if (bad) (*forbidden)++;
        if (r0 != r1) (*relaxed_any)++;
        done = (int)i;
    }
    c->stop = 1;
    WaitForMultipleObjects(2, h, TRUE, 5000);
    CloseHandle(h[0]); CloseHandle(h[1]);
    return done;
}
static void lit_case(int variant, const char *name, int must_hold) {
    NEED(F_SSE2);
    long forb = 0, rel = 0;
    int done = lit_run(variant, 20000, &forb, &rel);
    if (done < 0) SKIP("cannot create threads");
    if (done < 100) t_note("%s: only %d iterations before the deadline (slow host; correctness is judged on the outcomes seen)", name, done);
    t_info("iterations", "%d", done);
    t_info("forbidden_or_relaxed", "%ld", forb);
    if (must_hold) CHECKF(forb == 0, "%s: %ld of %d iterations show an outcome the x86 memory model forbids", name, forb, done);
    else t_note("%s: relaxed outcome observed %ld/%d times (allowed by the x86 model)", name, forb, done);
    if (t_cpu_count() < 2) t_note("single logical CPU: litmus outcomes are trivially ordered");
}
TEST(litmus_sb_plain_info)            { lit_case(LIT_SB_PLAIN, "SB plain", 0); }
TEST(litmus_sb_mfence)                { lit_case(LIT_SB_MFENCE, "SB+mfence", 1); }
TEST(litmus_sb_xchg)                  { lit_case(LIT_SB_XCHG, "SB+xchg", 1); }
TEST(litmus_sb_lock_or)               { lit_case(LIT_SB_LOCKOR, "SB+lock or", 1); }
TEST(litmus_sb_sfence_info)           { lit_case(LIT_SB_SFENCE, "SB+sfence", 0); }
TEST(litmus_sb_lfence_info)           { lit_case(LIT_SB_LFENCE, "SB+lfence", 0); }
TEST(litmus_mp_plain)                 { lit_case(LIT_MP_PLAIN, "MP plain", 1); }
TEST(litmus_mp_movnti_sfence)         { lit_case(LIT_MP_NT_SFENCE, "MP movnti+sfence", 1); }
TEST(litmus_mp_movnti_nofence_info)   { lit_case(LIT_MP_NT_NOFENCE, "MP movnti no fence", 0); }
TEST(litmus_lb_plain)                 { lit_case(LIT_LB_PLAIN, "LB plain", 1); }
/* NT (weakly ordered) stores are ordered before later stores by MFENCE (SDM Vol 2 MFENCE: "serializing
 * operation on all load-from-memory and store-to-memory instructions issued prior"), and by any LOCKed
 * instruction: "Reads or writes cannot be reordered with I/O instructions, locked instructions, or
 * serializing instructions" (SDM Vol 3 8.2.2) and WC buffers are drained by "an execution of a LOCK
 * instruction" (SDM Vol 3 11.3.1). XCHG with memory is implicitly locked. */
TEST(litmus_mp_movnti_mfence)         { lit_case(LIT_MP_NT_MFENCE, "MP movnti+mfence", cpu()->is_intel); }  /* ordering documented for Intel; info-only elsewhere */
TEST(litmus_mp_movnti_lock_or)        { lit_case(LIT_MP_NT_LOCK, "MP movnti+lock or", cpu()->is_intel); }  /* ordering documented for Intel; info-only elsewhere */
TEST(litmus_mp_movnti_xchg_flag)      { lit_case(LIT_MP_NT_XCHGFLAG, "MP movnti+xchg flag", cpu()->is_intel); }  /* ordering documented for Intel; info-only elsewhere */
TEST(litmus_mp_movntdq_sfence)        { lit_case(LIT_MP_NTDQ_SFENCE, "MP movntdq+sfence", 1); }

/* ======================================================================== */
/* fences, serialize, pause                                                   */
/* ======================================================================== */
#define SM_FL 0x8D5u   /* OF SF ZF AF PF CF */
#define FLAGS_PROBE(fname, insn) \
    static uintptr_t fname(uintptr_t pat) { uintptr_t t = 0; \
        __asm__ volatile("pushf\n\tpop %0\n\tand %2, %0\n\tor %1, %0\n\tpush %0\n\tpopf\n\t" insn "\n\tpushf\n\tpop %0" \
                         : "=&r"(t) : "r"(pat), "r"(~(uintptr_t)SM_FL) : "cc", "memory"); return t & SM_FL; }
FLAGS_PROBE(fl_sfence, "sfence")
FLAGS_PROBE(fl_lfence, "lfence")
FLAGS_PROBE(fl_mfence, "mfence")
FLAGS_PROBE(fl_pause, "pause")
FLAGS_PROBE(fl_serialize, "serialize")
FLAGS_PROBE(fl_nop, "nop")

TEST(fences_preserve_flags_and_registers) {
    NEED(F_SSE2);
    static const uintptr_t pats[] = { 0, SM_FL, 0x801, 0x0C4, 0x011 };
    for (size_t i = 0; i < T_COUNTOF(pats); i++) {
        CHECKF(fl_nop(pats[i]) == pats[i], "flag round trip via popf/pushf broken for 0x%lx", (unsigned long)pats[i]);
        CHECKF(fl_sfence(pats[i]) == pats[i], "sfence changed flags 0x%lx", (unsigned long)pats[i]);
        CHECKF(fl_lfence(pats[i]) == pats[i], "lfence changed flags 0x%lx", (unsigned long)pats[i]);
        CHECKF(fl_mfence(pats[i]) == pats[i], "mfence changed flags 0x%lx", (unsigned long)pats[i]);
        CHECKF(fl_pause(pats[i]) == pats[i], "pause changed flags 0x%lx", (unsigned long)pats[i]);
    }
    /* XMM and MXCSR unchanged across fences */
    unsigned mx = t_get_mxcsr();
    __m128i v = _mm_set_epi32(0x01234567, 0x89ABCDEF, 0x0F1E2D3C, 0x4B5A6978), o;
    __asm__ volatile("movdqa %1, %%xmm0\n\tsfence\n\tlfence\n\tmfence\n\tpause\n\tmovdqa %%xmm0, %0" : "=m"(o) : "m"(v) : "xmm0", "memory");
    CHECK_MEM(&o, &v, 16);
    CHECK_EQ(t_get_mxcsr(), mx);
}
TEST(serialize_preserves_flags) {
    NEED(F_SERIALIZE);
    static const uintptr_t pats[] = { 0, SM_FL, 0x801, 0x0C4 };
    for (size_t i = 0; i < T_COUNTOF(pats); i++)
        CHECKF(fl_serialize(pats[i]) == pats[i], "serialize changed flags 0x%lx", (unsigned long)pats[i]);
}
TEST(pause_latency_info_and_spin_handoff) {
    double t0 = t_now_ms();
    for (int i = 0; i < 200000; i++) sm_pause();
    double ms = t_now_ms() - t0;
    t_info("ns_per_pause", "%.2f", ms * 1e6 / 200000);
    CHECKF(ms < 5000, "200000 PAUSE took %.0f ms", ms);
}
static volatile LONG g_pp[32];
static DWORD WINAPI pp_thread(void *arg) {
    int me = (int)(intptr_t)arg;
    for (LONG i = 0; i < 2000; i++) {
        unsigned spins = 0;
        while ((g_pp[0] & 1) != me) { sm_pause(); if ((++spins & 1023) == 0) { if (g_pp[16]) return 0; SwitchToThread(); } }
        g_pp[8] += 1;              /* only the owner touches it: plain increment */
        sm_cbar();
        g_pp[0] = g_pp[0] + 1;     /* hand off: plain store (TSO release) */
    }
    return 0;
}
TEST(pause_spin_pingpong_threads) {
    memset((void *)g_pp, 0, sizeof g_pp);
    HANDLE a = CreateThread(NULL, 65536, pp_thread, (void *)0, 0, NULL);
    HANDLE b = a ? CreateThread(NULL, 65536, pp_thread, (void *)1, 0, NULL) : NULL;
    if (!a || !b) { g_pp[16] = 1; if (a) { WaitForSingleObject(a, 5000); CloseHandle(a); } SKIP("cannot create threads"); }
    HANDLE h[2] = { a, b };
    DWORD w = WaitForMultipleObjects(2, h, TRUE, 10000);
    if (w == WAIT_TIMEOUT) { g_pp[16] = 1; WaitForMultipleObjects(2, h, TRUE, 5000); }
    CloseHandle(a); CloseHandle(b);
    CHECKF(w != WAIT_TIMEOUT, "ping-pong did not finish in 10 s");
    CHECK_EQ(g_pp[0], 4000);
    CHECK_EQ(g_pp[8], 4000);
}

/* ======================================================================== */
/* TSC                                                                        */
/* ======================================================================== */
static inline uint64_t rdtsc_lf(void) {
    uint32_t lo, hi; __asm__ volatile("lfence\n\trdtsc\n\tlfence" : "=a"(lo), "=d"(hi) :: "memory");
    return (uint64_t)hi << 32 | lo;
}
TEST(rdtsc_lfence_monotonic_pinned) {
    NEED(F_TSC); NEED(F_SSE2);
    int cpus[64]; int nc = sm_cpu_list(cpus, 64);
    uintptr_t prev = nc ? t_pin_thread(cpus[0]) : 0;
    enum { N = 20000 };
    static uint64_t v[N];
    for (int i = 0; i < N; i++) v[i] = rdtsc_lf();
    int back = 0; uint64_t mind = ~0ull;
    for (int i = 1; i < N; i++) { if (v[i] < v[i - 1]) back++; else if (v[i] - v[i - 1] < mind) mind = v[i] - v[i - 1]; }
    if (prev) CHECKF(back == 0, "pinned: lfence;rdtsc went backwards %d times", back);
    else t_note("could not pin; TSC went backwards %d times (not a contract unpinned)", back);
    uint64_t a = rdtsc_lf(); Sleep(5); uint64_t b = rdtsc_lf();
    CHECKF(b > a, "TSC did not advance across Sleep(5)");
    t_info("min_fenced_delta_ticks", "%llu", (unsigned long long)mind);
    /* unfenced vs fenced read cost */
    double d[64];
    for (int s = 0; s < 64; s++) { uint64_t t0 = t_rdtsc(), t1 = t_rdtsc(); d[s] = (double)(t1 - t0); }
    t_stats st; t_stats_compute(d, 64, &st);
    t_info("rdtsc_back_to_back_ticks", "%.0f", st.median);
    t_unpin_thread(prev);
}
TEST(rdtscp_aux_stable_per_cpu) {
    NEED(F_RDTSCP);
    int cpus[64]; int nc = sm_cpu_list(cpus, 64);
    if (nc > 16) nc = 16;
    int unstable = 0, mism = 0, tested = 0;
    uint32_t auxv[64];
    for (int i = 0; i < nc; i++) {
        uintptr_t prev = t_pin_thread(cpus[i]);
        if (!prev) continue;
        Sleep(0);
        uint32_t a0 = 0, a = 0; t_rdtscp(&a0);
        for (int j = 0; j < 16; j++) { uint64_t t1, t2; t1 = t_rdtscp(&a); if (a != a0) unstable++; t2 = t_rdtscp(&a); CHECKF(t2 >= t1, "rdtscp went backwards on cpu %d", cpus[i]); }
        if ((a0 & 0xFFF) != GetCurrentProcessorNumber()) mism++;
        auxv[tested++] = a0;
        t_unpin_thread(prev);
    }
    if (!tested) SKIP("cannot pin the thread to any CPU");
    CHECKF(unstable == 0, "TSC_AUX changed %d times while pinned to one CPU", unstable);
    int distinct = 1;
    for (int i = 1; i < tested; i++) { int seen = 0; for (int j = 0; j < i; j++) seen |= auxv[j] == auxv[i]; distinct += !seen; }
    t_info("cpus_tested", "%d", tested);
    t_info("distinct_tsc_aux", "%d", distinct);
    if (mism) t_note("TSC_AUX low bits differ from GetCurrentProcessorNumber on %d/%d CPUs (OS-defined numbering)", mism, tested);
}
TEST(rdpid_equals_rdtscp_aux) {
    NEED(F_RDPID); NEED(F_RDTSCP);
    if (t_sde_model())   /* emulator artifact, README "Emulated validation" */
        t_expect_fail("Intel SDE (%s) emulates RDPID with its own TSC_AUX value", t_sde_model());
    int cpus[64]; int nc = sm_cpu_list(cpus, 64);
    if (nc > 8) nc = 8;
    pr_begin();
    int tested = 0;
    for (int i = 0; i < nc; i++) {
        uintptr_t prev = t_pin_thread(cpus[i]);
        if (!prev) continue;
        Sleep(0);
        uintptr_t id = 0; uint32_t aux = 0;
        t_rdtscp(&aux);
        __asm__ volatile(PR_ARM "rdpid %[id]" PR_DONE : PR_OUT, [id] "=r"(id) :: "memory" PR_CLOB_C);
        PR_EXPECT_NONE("rdpid (advertised by CPUID)");
        CHECKF((uint32_t)id == aux, "cpu %d: rdpid 0x%lx != rdtscp aux 0x%x", cpus[i], (unsigned long)id, aux);
        tested++;
        t_unpin_thread(prev);
    }
    pr_end();
    if (!tested) SKIP("cannot pin the thread");
}
TEST(tsc_rate_vs_qpc_info) {
    NEED(F_TSC);
    uint64_t q0 = t_qpc(), c0 = t_rdtsc();
    Sleep(50);
    uint64_t q1 = t_qpc(), c1 = t_rdtsc();
    double sec = t_qpc_sec(q0, q1);
    CHECK(c1 > c0);
    CHECK(sec > 0.0);
    t_info("tsc_mhz_vs_qpc", "%.1f", (double)(c1 - c0) / sec / 1e6);
    t_info("invariant_tsc", "%d", cpu_hw(F_INVARIANT_TSC));
}

/* ======================================================================== */
/* cache control                                                              */
/* ======================================================================== */
enum { FL_CLFLUSH, FL_CLFLUSHOPT, FL_CLWB, FL_CLDEMOTE };
static void pr_flush(int which, const void *p) {
    switch (which) {
    case FL_CLFLUSH:    __asm__ volatile(PR_ARM "clflush (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case FL_CLFLUSHOPT: __asm__ volatile(PR_ARM "clflushopt (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case FL_CLWB:       __asm__ volatile(PR_ARM "clwb (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case FL_CLDEMOTE:   __asm__ volatile(PR_ARM "cldemote (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    }
}
static const char *g_flname[] = { "clflush", "clflushopt", "clwb", "cldemote" };
static T_NOINLINE int flush_target_fn(int x) { return x * 3 + 7; }

static void flush_data_case(int which, t_feature f) {
    NEED(f); NEED(F_SSE2);
    const char *nm = g_flname[which];
    uint32_t line = ((cpu()->l1.ebx >> 8) & 0xFF) * 8;
    if (which == FL_CLFLUSH) { CHECKF(line != 0, "CLFSH set but CPUID.1:EBX[15:8] (CLFLUSH line size) is 0"); t_info("clflush_line", "%u", line); }
    if (!line) line = 64;
    size_t n = 256 * 1024;
    uint8_t *b = sm_valloc(n), *ref = (uint8_t *)malloc(n);
    REQUIRE(b && ref);
    t_rng r; t_rng_seed(&r, 0xF1u + (unsigned)which);
    t_rng_fill(&r, b, n); memcpy(ref, b, n);
    pr_begin();
    for (size_t i = 0; i < n; i += line) pr_flush(which, b + i);
    PR_EXPECT_NONE("%s over 256 KB", nm);
    sm_mfence();
    CHECKF(memcmp(b, ref, n) == 0, "%s changed memory contents", nm);
    /* odd addresses, flush of a dirty line, rewrite after flush */
    for (int it = 0; it < 2000; it++) {
        size_t o = t_rng_range(&r, n - 8);
        uint32_t v = t_rng_u32(&r);
        memcpy(b + o, &v, 4); memcpy(ref + o, &v, 4);
        pr_flush(which, b + o + (it & 3));
        if (it & 1) sm_sfence();
    }
    PR_EXPECT_NONE("%s at odd addresses", nm);
    sm_mfence();
    CHECKF(memcmp(b, ref, n) == 0, "%s: dirty-line flush lost data", nm);
    /* NT store then flush */
    _mm_stream_si128((__m128i *)b, _mm_set1_epi32(0x12345678)); pr_flush(which, b); sm_mfence();
    CHECK_EQ(((volatile uint32_t *)b)[2], 0x12345678u);
    PR_EXPECT_NONE("%s after NT store", nm);
    pr_end();
    free(ref); sm_vfree(b);
}
static void flush_perm_case(int which, t_feature f) {
    NEED(f);
    const char *nm = g_flname[which];
    static const DWORD prot[3] = { PAGE_READONLY, PAGE_EXECUTE_READ, PAGE_READWRITE };
    sm_region rg; REQUIRE(sm_region_alloc(&rg, 3, prot));
    pr_begin();
    pr_flush(which, sm_page(&rg, 0) + 100);
    PR_EXPECT_NONE("%s on PAGE_READONLY (byte-load permission semantics)", nm);
    pr_flush(which, sm_page(&rg, 1) + 300);
    PR_EXPECT_NONE("%s on PAGE_EXECUTE_READ", nm);
    CHECK_EQ(*(volatile uint32_t *)(sm_page(&rg, 0) + 96), sm_pat32(96, 0x5EED));   /* contents unchanged */
    /* flush our own code, then run it */
    const uint8_t *code = (const uint8_t *)(uintptr_t)flush_target_fn;
    for (int i = 0; i < 128; i += 16) pr_flush(which, code + i);
    PR_EXPECT_NONE("%s on own code", nm);
    sm_mfence();
    int (*volatile fp)(int) = flush_target_fn;
    CHECK_EQ(fp(5), 22);
    pr_flush(which, sm_page(&rg, 2) + 8);
    PR_EXPECT_NONE("%s on PAGE_READWRITE", nm);
    pr_end();
    sm_region_free(&rg);
}
TEST(cache_clflush_data_intact)        { flush_data_case(FL_CLFLUSH, F_CLFSH); }
TEST(cache_clflush_readonly_exec_ok)   { flush_perm_case(FL_CLFLUSH, F_CLFSH); }
TEST(cache_clflush_noaccess_faults)    { NEED(F_CLFSH); expect_av("clflush_noacc", -1, 1); }
TEST(cache_clflushopt_data_intact)     { flush_data_case(FL_CLFLUSHOPT, F_CLFLUSHOPT); }
TEST(cache_clflushopt_readonly_exec_ok){ flush_perm_case(FL_CLFLUSHOPT, F_CLFLUSHOPT); }
/* Emulator artifacts (README "Emulated validation"): under tools/sde_run.sh failures after this point are
 * expected; the checks themselves are unchanged, and a pass is reported as XPASS. */
#define SM_SDE_XFAIL(why) do { if (t_sde_model()) t_expect_fail("Intel SDE (%s) " why, t_sde_model()); } while (0)
/* SDE does not fault CLFLUSHOPT/CLWB on a no-access page */
#define SM_SDE_FLUSH_XFAIL(n) do { if (t_sde_model()) t_expect_fail("Intel SDE (%s) emulates " n " without the page-permission check", t_sde_model()); } while (0)
TEST(cache_clflushopt_noaccess_faults) { NEED(F_CLFLUSHOPT); SM_SDE_FLUSH_XFAIL("CLFLUSHOPT"); expect_av("clflushopt_noacc", -1, 1); }
TEST(cache_clwb_data_intact)           { flush_data_case(FL_CLWB, F_CLWB); }
TEST(cache_clwb_readonly_exec_ok)      { flush_perm_case(FL_CLWB, F_CLWB); }
TEST(cache_clwb_noaccess_faults)       { NEED(F_CLWB); SM_SDE_FLUSH_XFAIL("CLWB"); expect_av("clwb_noacc", -1, 1); }

TEST(cache_cldemote_hint) {
    /* CLDEMOTE (0F 1C /0) lies in the reserved-NOP hint space: a NOP where unsupported. */
    uint8_t *b = sm_valloc(4096); REQUIRE(b);
    memset(b, 0x3C, 4096);
    pr_begin();
    for (int i = 0; i < 4096; i += 64) pr_flush(FL_CLDEMOTE, b + i);
    int faults = (int)g_pr.nexc; char d[160] = ""; if (faults) pr_desc(d, sizeof d);
    pr_clear();
    if (cpu_hw(F_CLDEMOTE)) CHECKF(faults == 0, "cldemote advertised but raised %s", d);
    else if (faults) t_note("cldemote (hint NOP space) raised %s on a CPU without CLDEMOTE", d);
    else t_note("cldemote executed as a hint NOP (CPUID bit clear)");
    int same = 1; for (int i = 0; i < 4096; i++) same &= b[i] == 0x3C;
    CHECK(same);
    pr_end();
    sm_vfree(b);
}
TEST(cache_clzero_zeroes_one_line) {
    NEED(F_CLZERO);
    uint32_t line = ((cpu()->l1.ebx >> 8) & 0xFF) * 8;
    if (!line) line = 64;
    uint8_t *b = sm_valloc(4096); REQUIRE(b);
    memset(b, 0xEE, 4096);
    uint8_t *p = b + 1024 + 37;
    pr_begin();
#ifdef T_X64
    __asm__ volatile(PR_ARM "clzero %%rax" PR_DONE : PR_OUT : "a"(p) : "memory" PR_CLOB_C);
#else
    __asm__ volatile(PR_ARM "clzero %%eax" PR_DONE : PR_OUT : "a"(p) : "memory" PR_CLOB_C);
#endif
    PR_EXPECT_NONE("clzero");
    pr_end();
    sm_mfence();
    uintptr_t lo = (uintptr_t)p & ~(uintptr_t)(line - 1);
    int ok = 1;
    for (int i = 0; i < 4096; i++) { uintptr_t a = (uintptr_t)(b + i); ok &= b[i] == ((a >= lo && a < lo + line) ? 0 : 0xEE); }
    CHECKF(ok, "clzero did not zero exactly the %u-byte line containing the address", line);
    sm_vfree(b);
}

/* ======================================================================== */
/* prefetch                                                                   */
/* ======================================================================== */
enum { PF_NTA, PF_T0, PF_T1, PF_T2, PF_W, PF_WT1, PF_3DNOW, PF_N };
static const char *g_pfname[] = { "prefetchnta", "prefetcht0", "prefetcht1", "prefetcht2", "prefetchw", "prefetchwt1", "prefetch" };
static void pr_prefetch(int h, const void *p) {
    switch (h) {
    case PF_NTA:   __asm__ volatile(PR_ARM "prefetchnta (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case PF_T0:    __asm__ volatile(PR_ARM "prefetcht0 (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case PF_T1:    __asm__ volatile(PR_ARM "prefetcht1 (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case PF_T2:    __asm__ volatile(PR_ARM "prefetcht2 (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case PF_W:     __asm__ volatile(PR_ARM "prefetchw (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case PF_WT1:   __asm__ volatile(PR_ARM "prefetchwt1 (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    case PF_3DNOW: __asm__ volatile(PR_ARM "prefetch (%[p])" PR_DONE : PR_OUT : [p] "r"(p) : "memory" PR_CLOB_C); break;
    }
}
static int pf_avail(int h) {
    switch (h) {
    case PF_W: return cpu_hw(F_PREFETCHW) || cpu_hw(F_AMD3DNOW);
    case PF_WT1: return cpu_hw(F_PREFETCHWT1);
    /* 0F 0D /0 (PREFETCH): AMD defines it under 3DNow! or 3DNowPrefetch (CPUID 80000001h:ECX[8]); Intel's
     * same CPUID bit documents only PREFETCHW (0F 0D /1), so /0 is not part of Intel's contract. */
    case PF_3DNOW: return cpu_hw(F_AMD3DNOW) || (cpu_hw(F_PREFETCHW) && !strcmp(cpu()->vendor, "AuthenticAMD"));
    default: return cpu_has(F_SSE);
    }
}
/* run every available hint over a list of addresses: no exception allowed */
static int pf_sweep(const char *what, const void *const *addrs, int n) {
    int hints = 0;
    for (int h = 0; h < PF_N; h++) {
        if (!pf_avail(h)) continue;
        hints++;
        for (int i = 0; i < n; i++) {
            pr_prefetch(h, addrs[i]);
            PR_EXPECT_NONE("%s on %s %p", g_pfname[h], what, addrs[i]);
        }
    }
    return hints;
}
TEST(prefetch_valid_memory_all_hints) {
    NEED(F_SSE);
    uint8_t *b = sm_valloc(65536); REQUIRE(b);
    t_rng r; t_rng_seed(&r, 3); t_rng_fill(&r, b, 65536);
    uint32_t h0 = 0; for (int i = 0; i < 65536; i++) h0 = h0 * 31 + b[i];
    pr_begin();
    const void *a[64];
    for (int i = 0; i < 64; i++) a[i] = b + i * 1021;
    int hints = pf_sweep("valid memory", a, 64);
    pr_end();
    uint32_t h1 = 0; for (int i = 0; i < 65536; i++) h1 = h1 * 31 + b[i];
    CHECKF(h0 == h1, "prefetch changed memory");
    t_info("hints", "%d", hints);
    sm_vfree(b);
}
TEST(prefetch_noaccess_and_readonly_no_fault) {
    NEED(F_SSE);
    sm_region rg; REGION4(rg);
    pr_begin();
    const void *a[] = { sm_page(&rg, 1), sm_page(&rg, 1) + 1, sm_page(&rg, 1) + rg.pg - 1, sm_page(&rg, 1) - 1,
                        sm_page(&rg, 3), sm_page(&rg, 3) + 2047 };
    pf_sweep("PAGE_NOACCESS/PAGE_READONLY", a, (int)T_COUNTOF(a));
    pr_end();
    CHECK_EQ(*(volatile uint32_t *)sm_page(&rg, 3), sm_pat32(3 * rg.pg, 0x5EED));
    sm_region_free(&rg);
}
TEST(prefetch_reserved_and_freed_no_fault) {
    NEED(F_SSE);
    uint8_t *res = (uint8_t *)VirtualAlloc(NULL, 1 << 20, MEM_RESERVE, PAGE_NOACCESS);
    uint8_t *fr = sm_valloc(65536);
    REQUIRE(res && fr);
    VirtualFree(fr, 0, MEM_RELEASE);
    pr_begin();
    const void *a[] = { res, res + 4096 * 7 + 5, res + (1 << 20) - 64, fr, fr + 32768 };
    pf_sweep("reserved/freed", a, (int)T_COUNTOF(a));
    pr_end();
    VirtualFree(res, 0, MEM_RELEASE);
}
TEST(prefetch_null_and_kernel_addresses_no_fault) {
    NEED(F_SSE);
    pr_begin();
#ifdef T_X64
    const void *a[] = { (void *)0, (void *)8, (void *)0xFFF, (void *)0x10000, (void *)0x00007FFFFFFFFFC0ull,
                        (void *)0xFFFF800000000000ull, (void *)0xFFFFF78000000000ull, (void *)0xFFFFFFFFFFFFFFC0ull };
#else
    const void *a[] = { (void *)0, (void *)8, (void *)0xFFF, (void *)0x10000, (void *)0x7FFEFFC0u,
                        (void *)0x80000000u, (void *)0xC0000000u, (void *)0xFFFFFFC0u };
#endif
    pf_sweep("null/kernel", a, (int)T_COUNTOF(a));
    pr_end();
}
TEST(prefetch_noncanonical_no_fault) {
    NEED(F_SSE);
#ifdef T_X64
    pr_begin();
    const void *a[] = { (void *)0x0000800000000000ull, (void *)0x8000000000000000ull, (void *)0x7FFFFFFFFFFFFFC0ull,
                        (void *)0xFFFF7FFFFFFFFFC0ull, (void *)0x1234567890ABCDE0ull };
    pf_sweep("non-canonical", a, (int)T_COUNTOF(a));
    pr_end();
#else
    SKIP("no canonical-address rule in 32-bit mode");
#endif
}
TEST(prefetch_guard_page_stays_armed) {
    NEED(F_SSE);
    static const DWORD prot[2] = { PAGE_READWRITE, PAGE_READWRITE | PAGE_GUARD };
    sm_region rg; REQUIRE(sm_region_alloc(&rg, 2, prot));
    uint8_t *g = sm_page(&rg, 1);
    REQUIRE(sm_is_guard(g));
    pr_begin();
    const void *a[] = { g, g + 64, g + rg.pg - 64, g + 1000 };
    pf_sweep("PAGE_GUARD", a, (int)T_COUNTOF(a));
    CHECKF(sm_is_guard(g), "a prefetch consumed the PAGE_GUARD one-shot");
    /* a real access raises STATUS_GUARD_PAGE_VIOLATION exactly once, then the page is ordinary */
    uint32_t v = 0;
    __asm__ volatile(PR_ARM "movl (%[p]), %[v]" PR_DONE : PR_OUT, [v] "+r"(v) : [p] "r"(g + 64) : "memory" PR_CLOB_C);
    CHECKF(g_pr.nexc == 1 && g_pr.code == SM_GUARD, "first touch: nexc=%ld code=0x%08lx (want one STATUS_GUARD_PAGE_VIOLATION)",
           g_pr.nexc, (unsigned long)g_pr.code);
    if (g_pr.nexc) CHECKF(g_pr.addr == (ULONG_PTR)(g + 64), "guard violation address %p want %p", (void *)g_pr.addr, (void *)(g + 64));
    pr_clear();
    CHECKF(!sm_is_guard(g), "PAGE_GUARD still set after the guard exception");
    __asm__ volatile(PR_ARM "movl (%[p]), %[v]" PR_DONE : PR_OUT, [v] "+r"(v) : [p] "r"(g + 64) : "memory" PR_CLOB_C);
    PR_EXPECT_NONE("second touch of former guard page");
    CHECK_EQ(v, sm_pat32(rg.pg + 64, 0x5EED));
    pr_end();
    sm_region_free(&rg);
}
TEST(prefetchw_readonly_page_no_write) {
    if (!pf_avail(PF_W)) SKIP("PREFETCHW not advertised (CPUID 80000001h:ECX[8] / 3DNow)");
    sm_region rg; REGION4(rg);
    pr_begin();
    for (int i = 0; i < 64; i++) { pr_prefetch(PF_W, sm_page(&rg, 3) + i * 64); }
    PR_EXPECT_NONE("prefetchw on PAGE_READONLY");
    pr_prefetch(PF_W, sm_page(&rg, 1) + 128);
    PR_EXPECT_NONE("prefetchw on PAGE_NOACCESS");
    pr_end();
    int same = 1; uint32_t *w = (uint32_t *)sm_page(&rg, 3);
    for (size_t j = 0; j < rg.pg / 4; j++) same &= w[j] == sm_pat32(3 * rg.pg + 4 * j, 0x5EED);
    CHECK(same);
    sm_region_free(&rg);
}
TEST(prefetchit_code_hints) {
#ifdef T_X64
    NEED(F_PREFETCHI);
    pr_begin();
    __asm__ volatile(PR_ARM "prefetchit0 0(%%rip)" PR_DONE : PR_OUT :: "memory" PR_CLOB_C);
    PR_EXPECT_NONE("prefetchit0");
    __asm__ volatile(PR_ARM "prefetchit1 0(%%rip)" PR_DONE : PR_OUT :: "memory" PR_CLOB_C);
    PR_EXPECT_NONE("prefetchit1");
    pr_end();
#else
    SKIP("PREFETCHIT0/1 are 64-bit mode only");
#endif
}

/* ======================================================================== */
/* MASKMOVDQU / MASKMOVQ                                                      */
/* ======================================================================== */
TEST(maskmovdqu_byte_select) {
    NEED(F_SSE2);
    uint8_t *b = sm_valloc(4096), ref[64];
    REQUIRE(b);
    t_rng r; t_rng_seed(&r, 41);
    for (int it = 0; it < 500; it++) {
        size_t o = 16 + t_rng_range(&r, 4096 - 64);
        t_rng_fill(&r, b + o - 16, 48); memcpy(ref, b + o - 16, 48);
        __attribute__((aligned(16))) uint8_t data[16], m[16];
        t_rng_fill(&r, data, 16); t_rng_fill(&r, m, 16);
        if (it == 0) memset(m, 0, 16);
        if (it == 1) memset(m, 0xFF, 16);
        _mm_maskmoveu_si128(_mm_load_si128((const __m128i *)data), _mm_load_si128((const __m128i *)m), (char *)(b + o));
        for (int i = 0; i < 16; i++) if (m[i] & 0x80) ref[16 + i] = data[i];
        int ok = memcmp(b + o - 16, ref, 48) == 0;
        CHECKF(ok, "maskmovdqu iteration %d offset %llu mismatch", it, (unsigned long long)o);
        if (!ok) { CHECK_MEM(b + o - 16, ref, 48); break; }
    }
    sm_sfence();
    sm_vfree(b);
}
TEST(maskmovq_mmx_byte_select) {
    NEED(F_SSE);
    uint8_t *b = sm_valloc(4096), ref[24];
    REQUIRE(b);
    t_rng r; t_rng_seed(&r, 42);
    int bad = 0;
    for (int it = 0; it < 500 && !bad; it++) {
        size_t o = 8 + t_rng_range(&r, 4096 - 32);
        t_rng_fill(&r, b + o - 8, 24); memcpy(ref, b + o - 8, 24);
        uint64_t data = t_rng_next(&r), m = t_rng_next(&r);
        __asm__ volatile("movq %1, %%mm0\n\tmovq %2, %%mm1\n\tmaskmovq %%mm1, %%mm0\n\temms"
                         :: "D"(b + o), "m"(data), "m"(m) : "mm0", "mm1", "memory");
        for (int i = 0; i < 8; i++) if ((m >> (8 * i + 7)) & 1) ref[8 + i] = (uint8_t)(data >> (8 * i));
        if (memcmp(b + o - 8, ref, 24)) { bad = 1; CHECK_MEM(b + o - 8, ref, 24); }
    }
    CHECK(!bad);
    CHECK_EQ(t_get_x87_tw(), 0xFFFF);   /* EMMS emptied the tags */
    sm_sfence();
    sm_vfree(b);
}
TEST(maskmovdqu_active_byte_in_bad_page_faults) {
    NEED(F_SSE2);
    expect_av("maskmovdqu_active", 1, 1);
}
TEST(maskmovdqu_inactive_bytes_bad_page_info) {
    NEED(F_SSE2);
    /* SDM: with masked-off bytes (or an all-zero mask) "exceptions associated with addressing memory and
     * page faults may still be signaled (implementation dependent)" -> record only. */
    static const char *ops[] = { "maskmovdqu_inactive_info", "maskmovdqu_zero_info" };
    for (int i = 0; i < 2; i++) {
        sm_fres f; run_fop(ops[i], &f);
        if (!f.ok) continue;
        CHECKF(f.faulted || f.nofault, "%s: child neither faulted nor completed (exit 0x%lx)", ops[i], (unsigned long)g_cr.exit_code);
        if (f.faulted) {
            CHECKF(f.code == SM_AV, "%s: unexpected exception 0x%08lx", ops[i], (unsigned long)f.code);
            t_note("%s: faulted (implementation-dependent) rw=%lu off=%lld", ops[i], f.rw, f.off);
        } else t_note("%s: no fault (masked bytes suppressed)", ops[i]);
    }
}

/* ======================================================================== */
/* masked loads/stores: VMASKMOV / VPMASKMOV / AVX-512 k-masked moves         */
/* ======================================================================== */
enum { MO_LDZ, MO_LDM, MO_ST };
typedef void (*mfn)(void *p, const void *mask, void *io);

#define VM_LD(fname, insn, R) \
    T_NOINLINE static void fname(void *p, const void *m, void *io) { \
        __asm__ volatile("vmovdqu (%[io]), %%" R "0\n\tvmovdqu (%[m]), %%" R "1\n\t" PR_ARM \
                         insn " (%[p]), %%" R "1, %%" R "0\n\t" PR_DONE \
                         "vmovdqu %%" R "0, (%[io])\n\tvzeroupper\n\t" \
                         : PR_OUT : [p] "r"(p), [m] "r"(m), [io] "r"(io) : "xmm0", "xmm1", "memory", "cc" PR_CLOB_C); }
#define VM_ST(fname, insn, R) \
    T_NOINLINE static void fname(void *p, const void *m, void *io) { \
        __asm__ volatile("vmovdqu (%[io]), %%" R "0\n\tvmovdqu (%[m]), %%" R "1\n\t" PR_ARM \
                         insn " %%" R "0, %%" R "1, (%[p])\n\t" PR_DONE "vzeroupper\n\t" \
                         : PR_OUT : [p] "r"(p), [m] "r"(m), [io] "r"(io) : "xmm0", "xmm1", "memory", "cc" PR_CLOB_C); }
VM_LD(vm_ps_x_ld_avx, "vmaskmovps", "xmm") VM_ST(vm_ps_x_st_avx, "vmaskmovps", "xmm")
VM_LD(vm_ps_y_ld_avx, "vmaskmovps", "ymm") VM_ST(vm_ps_y_st_avx, "vmaskmovps", "ymm")
VM_LD(vm_pd_x_ld_avx, "vmaskmovpd", "xmm") VM_ST(vm_pd_x_st_avx, "vmaskmovpd", "xmm")
VM_LD(vm_pd_y_ld_avx, "vmaskmovpd", "ymm") VM_ST(vm_pd_y_st_avx, "vmaskmovpd", "ymm")
VM_LD(vm_d_x_ld_avx2, "vpmaskmovd", "xmm") VM_ST(vm_d_x_st_avx2, "vpmaskmovd", "xmm")
VM_LD(vm_d_y_ld_avx2, "vpmaskmovd", "ymm") VM_ST(vm_d_y_st_avx2, "vpmaskmovd", "ymm")
VM_LD(vm_q_x_ld_avx2, "vpmaskmovq", "xmm") VM_ST(vm_q_x_st_avx2, "vpmaskmovq", "xmm")
VM_LD(vm_q_y_ld_avx2, "vpmaskmovq", "ymm") VM_ST(vm_q_y_st_avx2, "vpmaskmovq", "ymm")

#define ZM_TGT "avx512f,avx512bw,avx512vl"
#define ZM_LD(fname, kmov, mov, insn, R, Z) \
    T_TARGET(ZM_TGT) T_NOINLINE static void fname(void *p, const void *k, void *io) { \
        __asm__ volatile(kmov " (%[k]), %%k1\n\t" mov " (%[io]), %%" R "0\n\t" PR_ARM \
                         insn " (%[p]), %%" R "0%{%%k1%}" Z "\n\t" PR_DONE \
                         mov " %%" R "0, (%[io])\n\tvzeroupper\n\t" \
                         : PR_OUT : [p] "r"(p), [k] "r"(k), [io] "r"(io) : "xmm0", "k1", "memory", "cc" PR_CLOB_C); }
#define ZM_ST(fname, kmov, mov, insn, R) \
    T_TARGET(ZM_TGT) T_NOINLINE static void fname(void *p, const void *k, void *io) { \
        __asm__ volatile(kmov " (%[k]), %%k1\n\t" mov " (%[io]), %%" R "0\n\t" PR_ARM \
                         insn " %%" R "0, (%[p])%{%%k1%}\n\t" PR_DONE "vzeroupper\n\t" \
                         : PR_OUT : [p] "r"(p), [k] "r"(k), [io] "r"(io) : "xmm0", "k1", "memory", "cc" PR_CLOB_C); }
#define ZM_SET(sz, kmov, insn) \
    ZM_LD(zm_##sz##_z_ldm_512, kmov, "vmovdqu64", insn, "zmm", "") \
    ZM_LD(zm_##sz##_z_ldz_512, kmov, "vmovdqu64", insn, "zmm", "%{z%}") \
    ZM_ST(zm_##sz##_z_st_512, kmov, "vmovdqu64", insn, "zmm") \
    ZM_LD(zm_##sz##_y_ldm_512, kmov, "vmovdqu", insn, "ymm", "") \
    ZM_LD(zm_##sz##_y_ldz_512, kmov, "vmovdqu", insn, "ymm", "%{z%}") \
    ZM_ST(zm_##sz##_y_st_512, kmov, "vmovdqu", insn, "ymm") \
    ZM_LD(zm_##sz##_x_ldm_512, kmov, "vmovdqu", insn, "xmm", "") \
    ZM_LD(zm_##sz##_x_ldz_512, kmov, "vmovdqu", insn, "xmm", "%{z%}") \
    ZM_ST(zm_##sz##_x_st_512, kmov, "vmovdqu", insn, "xmm")
ZM_SET(b8, "kmovq", "vmovdqu8")
ZM_SET(w16, "kmovq", "vmovdqu16")
ZM_SET(d32, "kmovw", "vmovdqu32")
ZM_SET(q64, "kmovw", "vmovdqu64")
/* also the FP-typed forms */
ZM_LD(zm_ps_z_ldz_512, "kmovw", "vmovdqu64", "vmovups", "zmm", "%{z%}") ZM_ST(zm_ps_z_st_512, "kmovw", "vmovdqu64", "vmovups", "zmm")
ZM_LD(zm_pd_z_ldm_512, "kmovw", "vmovdqu64", "vmovupd", "zmm", "")     ZM_ST(zm_pd_z_st_512, "kmovw", "vmovdqu64", "vmovupd", "zmm")

typedef struct mop { const char *group, *form; mfn fn; int op, es, vb, kmask, feat; } mop;
#define ZMOPS(grp, sz, es) \
    { grp, "zmm.ldm", zm_##sz##_z_ldm_512, MO_LDM, es, 64, 1, F_AVX512F }, { grp, "zmm.ldz", zm_##sz##_z_ldz_512, MO_LDZ, es, 64, 1, F_AVX512F }, \
    { grp, "zmm.st", zm_##sz##_z_st_512, MO_ST, es, 64, 1, F_AVX512F }, \
    { grp, "ymm.ldm", zm_##sz##_y_ldm_512, MO_LDM, es, 32, 1, F_AVX512F }, { grp, "ymm.ldz", zm_##sz##_y_ldz_512, MO_LDZ, es, 32, 1, F_AVX512F }, \
    { grp, "ymm.st", zm_##sz##_y_st_512, MO_ST, es, 32, 1, F_AVX512F }, \
    { grp, "xmm.ldm", zm_##sz##_x_ldm_512, MO_LDM, es, 16, 1, F_AVX512F }, { grp, "xmm.ldz", zm_##sz##_x_ldz_512, MO_LDZ, es, 16, 1, F_AVX512F }, \
    { grp, "xmm.st", zm_##sz##_x_st_512, MO_ST, es, 16, 1, F_AVX512F }
static const mop g_mops[] = {
    { "vmaskmovps_xmm", "ld", vm_ps_x_ld_avx, MO_LDZ, 4, 16, 0, F_AVX }, { "vmaskmovps_xmm", "st", vm_ps_x_st_avx, MO_ST, 4, 16, 0, F_AVX },
    { "vmaskmovps_ymm", "ld", vm_ps_y_ld_avx, MO_LDZ, 4, 32, 0, F_AVX }, { "vmaskmovps_ymm", "st", vm_ps_y_st_avx, MO_ST, 4, 32, 0, F_AVX },
    { "vmaskmovpd_xmm", "ld", vm_pd_x_ld_avx, MO_LDZ, 8, 16, 0, F_AVX }, { "vmaskmovpd_xmm", "st", vm_pd_x_st_avx, MO_ST, 8, 16, 0, F_AVX },
    { "vmaskmovpd_ymm", "ld", vm_pd_y_ld_avx, MO_LDZ, 8, 32, 0, F_AVX }, { "vmaskmovpd_ymm", "st", vm_pd_y_st_avx, MO_ST, 8, 32, 0, F_AVX },
    { "vpmaskmovd_xmm", "ld", vm_d_x_ld_avx2, MO_LDZ, 4, 16, 0, F_AVX2 }, { "vpmaskmovd_xmm", "st", vm_d_x_st_avx2, MO_ST, 4, 16, 0, F_AVX2 },
    { "vpmaskmovd_ymm", "ld", vm_d_y_ld_avx2, MO_LDZ, 4, 32, 0, F_AVX2 }, { "vpmaskmovd_ymm", "st", vm_d_y_st_avx2, MO_ST, 4, 32, 0, F_AVX2 },
    { "vpmaskmovq_xmm", "ld", vm_q_x_ld_avx2, MO_LDZ, 8, 16, 0, F_AVX2 }, { "vpmaskmovq_xmm", "st", vm_q_x_st_avx2, MO_ST, 8, 16, 0, F_AVX2 },
    { "vpmaskmovq_ymm", "ld", vm_q_y_ld_avx2, MO_LDZ, 8, 32, 0, F_AVX2 }, { "vpmaskmovq_ymm", "st", vm_q_y_st_avx2, MO_ST, 8, 32, 0, F_AVX2 },
    ZMOPS("avx512_vmovdqu8", b8, 1), ZMOPS("avx512_vmovdqu16", w16, 2),
    ZMOPS("avx512_vmovdqu32", d32, 4), ZMOPS("avx512_vmovdqu64", q64, 8),
    { "avx512_vmovups_vmovupd", "ps.zmm.ldz", zm_ps_z_ldz_512, MO_LDZ, 4, 64, 1, F_AVX512F },
    { "avx512_vmovups_vmovupd", "ps.zmm.st", zm_ps_z_st_512, MO_ST, 4, 64, 1, F_AVX512F },
    { "avx512_vmovups_vmovupd", "pd.zmm.ldm", zm_pd_z_ldm_512, MO_LDM, 8, 64, 1, F_AVX512F },
    { "avx512_vmovups_vmovupd", "pd.zmm.st", zm_pd_z_st_512, MO_ST, 8, 64, 1, F_AVX512F },
};
static int mop_avail(const mop *m) {
    if (!cpu_has((t_feature)m->feat)) return 0;
    if (m->kmask) {
        if (m->es < 4 && !cpu_has(F_AVX512BW)) return 0;
        if (m->vb < 64 && !cpu_has(F_AVX512VL)) return 0;
    }
    return 1;
}
static inline uint64_t lowbits(int n) { return n >= 64 ? ~0ull : ((1ull << n) - 1); }
static void mask_build(const mop *m, uint64_t bits, uint8_t *mb, t_rng *r) {
    memset(mb, 0, 64);
    if (m->kmask) { memcpy(mb, &bits, 8); return; }
    int n = m->vb / m->es;
    for (int i = 0; i < n; i++) {
        for (int b = 0; b < m->es; b++) mb[i * m->es + b] = (uint8_t)t_rng_next(r);   /* only the MSB counts */
        uint8_t *top = &mb[i * m->es + m->es - 1];
        if ((bits >> i) & 1) *top |= 0x80; else *top &= 0x7F;
    }
}
/* run one masked op at p; verify register (loads) or memory window [wlo, whi) (stores); 1 = ok */
static int mop_check(const mop *m, uint8_t *p, uint64_t bits, uint8_t *wlo, uint8_t *whi, t_rng *r, char *why, size_t cap) {
    __attribute__((aligned(64))) uint8_t mb[64], io[64], io0[64];
    uint8_t snap[512], exp[512];
    int n = m->vb / m->es;
    size_t wn = (size_t)(whi - wlo);
    if (wn > sizeof snap) wn = sizeof snap;
    mask_build(m, bits, mb, r);
    t_rng_fill(r, io, 64); memcpy(io0, io, 64);
    if (wn) memcpy(snap, wlo, wn);
    pr_clear();
    m->fn(p, mb, io);
    if (g_pr.nexc) {
        char d[160]; pr_desc(d, sizeof d);
        snprintf(why, cap, "%s (p=%p bits=0x%llx)", d, (void *)p, (unsigned long long)bits);
        pr_clear(); return -1;
    }
    if (m->op == MO_ST) {
        memcpy(exp, snap, wn);
        for (int i = 0; i < n; i++) if ((bits >> i) & 1)
            for (int b = 0; b < m->es; b++) {
                uint8_t *a = p + i * m->es + b;
                if (a >= wlo && a < wlo + wn) exp[a - wlo] = io0[i * m->es + b];
            }
        for (size_t j = 0; j < wn; j++) if (wlo[j] != exp[j]) {
            snprintf(why, cap, "memory byte at p%+lld = 0x%02x want 0x%02x (bits=0x%llx)", (long long)(wlo + j - p), wlo[j], exp[j], (unsigned long long)bits);
            return 0;
        }
    } else {
        for (int i = 0; i < n; i++)
            for (int b = 0; b < m->es; b++) {
                int k = i * m->es + b;
                uint8_t want = ((bits >> i) & 1) ? p[k] : (m->op == MO_LDZ ? 0 : io0[k]);
                if (io[k] != want) {
                    snprintf(why, cap, "element %d byte %d = 0x%02x want 0x%02x (%s, bits=0x%llx)", i, b, io[k], want,
                             ((bits >> i) & 1) ? "active" : "inactive", (unsigned long long)bits);
                    return 0;
                }
            }
    }
    return 1;
}
/* Fault suppression for masked-off elements of the VEX forms (VMASKMOVPS/PD, VPMASKMOVD/Q) is an Intel SDM
 * guarantee ("Faults will not occur due to referencing any memory location if the corresponding mask bit
 * for that memory location is 0"). The AMD APM entries for the same instructions say instead: "Exception
 * and trap behavior for elements not selected for loading or storing from/to memory is implementation
 * dependent", e.g. a page fault for a zero-masked element. On non-Intel processors an exception on a
 * masked-off VEX element is therefore recorded as NOTE (processor behaviour, not an OS contract).
 * EVEX (AVX-512) fault suppression is architectural for both vendors and stays a hard check. */
static int sm_vex_mask_fault_impl_dep(const mop *m) { return !m->kmask && strcmp(cpu()->vendor, "GenuineIntel") != 0; }
static int g_mo_notes;
/* res from mop_check: 1 ok, 0 wrong data, -1 exception. allow_dep: the exception hit masked-off elements only */
static int mo_ok(const mop *m, int res, int allow_dep, const char *why) {
    if (res == 1) return 1;
    if (res == -1 && allow_dep && sm_vex_mask_fault_impl_dep(m)) {
        if (g_mo_notes++ < 4) t_note("%s %s: masked-off element raised %s (implementation dependent on %s per AMD APM)",
                                     m->group, m->form, why, cpu()->vendor);
        return 1;
    }
    return 0;
}
static void masked_group(const char *group) {
    sm_region rg; int any = 0, skipped = 0;
    g_mo_notes = 0;
    for (size_t mi = 0; mi < T_COUNTOF(g_mops); mi++) if (!strcmp(g_mops[mi].group, group)) { if (mop_avail(&g_mops[mi])) any++; else skipped++; }
    if (!any) SKIP("%s: no form usable on this CPU/OS", group);
    REGION4(rg);
    pr_begin();
    t_rng r; t_rng_seed(&r, t_mix64((uint64_t)(uintptr_t)group[0] * 131 + strlen(group)));
    uint8_t *P0 = sm_page(&rg, 0), *P1 = sm_page(&rg, 1), *P2 = sm_page(&rg, 2), *P3 = sm_page(&rg, 3);
    size_t pg = rg.pg;
    char why[256];
    for (size_t mi = 0; mi < T_COUNTOF(g_mops); mi++) {
        const mop *m = &g_mops[mi];
        if (strcmp(m->group, group) || !mop_avail(m)) continue;
        int n = m->vb / m->es; uint64_t full = lowbits(n);
        int ok = 1;
        /* A: valid memory, random masks, unaligned addresses */
        for (int it = 0; it < 64 && ok; it++) {
            uint8_t *p = P0 + 16 + t_rng_range(&r, pg - m->vb - 32);
            uint64_t bits = t_rng_next(&r) & full;
            if (it == 0) bits = 0;
            if (it == 1) bits = full;
            ok = mop_check(m, p, bits, p - 16, p + m->vb + 16, &r, why, sizeof why) == 1;
            CHECKF(ok, "%s %s valid memory: %s", group, m->form, why);
        }
        /* B: trailing elements in the PAGE_NOACCESS page, masked off */
        for (int k = 0; k <= n && ok; k++)
            for (int d = 0; d < (m->es > 1 ? 2 : 1) && ok; d++) {
                int dd = d ? m->es / 2 : 0;
                uint8_t *p = P1 - (size_t)k * m->es - dd;
                uint64_t bits = t_rng_next(&r) & lowbits(k);
                if (k > 0 && (t_rng_next(&r) & 1)) bits |= 1ull << (k - 1);
                uint8_t *wlo = p - 16 < P0 ? P0 : p - 16;
                ok = mo_ok(m, mop_check(m, p, bits, wlo, P1, &r, why, sizeof why), 1, why);
                CHECKF(ok, "%s %s: %d elements before PAGE_NOACCESS (+%d straddle), inactive beyond: %s", group, m->form, k, dd, why);
            }
        /* C: leading elements in the PAGE_NOACCESS page, masked off */
        for (int j = 0; j <= n && ok; j++)
            for (int d = 0; d < (m->es > 1 ? 2 : 1) && ok; d++) {
                int dd = d ? m->es / 2 : 0;
                uint8_t *p = P2 - (size_t)j * m->es + dd;
                uint64_t bits = t_rng_next(&r) & full & ~lowbits(j);
                ok = mo_ok(m, mop_check(m, p, bits, P2, p + m->vb + 16, &r, why, sizeof why), 1, why);
                CHECKF(ok, "%s %s: %d leading elements in PAGE_NOACCESS (+%d straddle), inactive: %s", group, m->form, j, dd, why);
            }
        /* D: entire vector inside PAGE_NOACCESS with an all-zero mask */
        if (ok) {
            ok = mo_ok(m, mop_check(m, P1 + 64, 0, P0, P0, &r, why, sizeof why), 1, why);
            CHECKF(ok, "%s %s: all-zero mask entirely in PAGE_NOACCESS: %s", group, m->form, why);
        }
        if (m->op == MO_ST && ok) {
            /* E: stores into PAGE_READONLY with the mask clear there */
            ok = mo_ok(m, mop_check(m, P3 + 40, 0, P3 + 32, P3 + 32 + m->vb + 16, &r, why, sizeof why), 1, why);
            CHECKF(ok, "%s %s: zero-mask store onto PAGE_READONLY: %s", group, m->form, why);
            for (int k = 1; k < n && ok; k++) {
                uint8_t *p = P3 - (size_t)k * m->es;
                uint64_t bits = (t_rng_next(&r) & lowbits(k)) | (1ull << (k - 1));
                ok = mo_ok(m, mop_check(m, p, bits, p - 8, P3 + 16, &r, why, sizeof why), 1, why);
                CHECKF(ok, "%s %s: store RW->READONLY, active only in RW (%d): %s", group, m->form, k, why);
            }
        } else if (ok) {
            /* E: full-mask load from PAGE_READONLY */
            ok = mop_check(m, P3 + 40, full, P3, P3, &r, why, sizeof why) == 1;
            CHECKF(ok, "%s %s: load from PAGE_READONLY: %s", group, m->form, why);
        }
    }
    pr_end();
    if (skipped) t_note("%s: %d form(s) not usable (AVX512BW/VL)", group, skipped);
    sm_region_free(&rg);
}
TEST(masked_vmaskmovps_xmm)  { masked_group("vmaskmovps_xmm"); }
TEST(masked_vmaskmovps_ymm)  { masked_group("vmaskmovps_ymm"); }
TEST(masked_vmaskmovpd_xmm)  { masked_group("vmaskmovpd_xmm"); }
TEST(masked_vmaskmovpd_ymm)  { masked_group("vmaskmovpd_ymm"); }
TEST(masked_vpmaskmovd_xmm)  { masked_group("vpmaskmovd_xmm"); }
TEST(masked_vpmaskmovd_ymm)  { masked_group("vpmaskmovd_ymm"); }
TEST(masked_vpmaskmovq_xmm)  { masked_group("vpmaskmovq_xmm"); }
TEST(masked_vpmaskmovq_ymm)  { masked_group("vpmaskmovq_ymm"); }
TEST(masked_avx512_vmovdqu8)  { masked_group("avx512_vmovdqu8"); }
TEST(masked_avx512_vmovdqu16) { masked_group("avx512_vmovdqu16"); }
TEST(masked_avx512_vmovdqu32) { masked_group("avx512_vmovdqu32"); }
TEST(masked_avx512_vmovdqu64) { masked_group("avx512_vmovdqu64"); }
TEST(masked_avx512_vmovups_vmovupd) { masked_group("avx512_vmovups_vmovupd"); }

TEST(masked_vex_active_lane_in_bad_page_faults) {
    NEED(F_AVX);
    expect_av("vmaskmovps_ld_active", 0, 1);
    expect_av("vmaskmovps_st_active", 1, 1);
    expect_av("vpmaskmovd_ld_active", 0, 1);
    expect_av("vpmaskmovq_st_active", 1, 1);
}
TEST(masked_avx512_active_lane_in_bad_page_faults) {
    NEED(F_AVX512F);
    expect_av("zmm_ld_active", 0, 1);
    SM_SDE_XFAIL("reports a fault of an emulated EVEX store as a read (ExceptionInformation[0]=0)");
    expect_av("zmm_st_active", 1, 1);
}
TEST(masked_avx512_aligned_move_k0_misaligned_info) {
    NEED(F_AVX512F);
    sm_fres f; run_fop("vmovdqa32_k0_mis_info", &f);
    if (!f.ok) return;
    if (f.faulted) t_note("vmovdqa32 zmm{k=0} misaligned: exception 0x%08lx (alignment checked regardless of mask)", (unsigned long)f.code);
    else t_note("vmovdqa32 zmm{k=0} misaligned: no exception");
}

/* ---- compress store / expand load bounded by popcnt(k) ---- */
T_TARGET("avx512f") T_NOINLINE static void zc_compress_d_512(void *p, const void *k, void *io) {
    __asm__ volatile("kmovw (%[k]), %%k1\n\tvmovdqu64 (%[io]), %%zmm0\n\t" PR_ARM
                     "vpcompressd %%zmm0, (%[p])%{%%k1%}\n\t" PR_DONE "vzeroupper\n\t"
                     : PR_OUT : [p] "r"(p), [k] "r"(k), [io] "r"(io) : "xmm0", "k1", "memory", "cc" PR_CLOB_C);
}
T_TARGET("avx512f") T_NOINLINE static void zc_expand_d_512(void *p, const void *k, void *io) {
    __asm__ volatile("kmovw (%[k]), %%k1\n\tvmovdqu64 (%[io]), %%zmm0\n\t" PR_ARM
                     "vpexpandd (%[p]), %%zmm0%{%%k1%}\n\t" PR_DONE "vmovdqu64 %%zmm0, (%[io])\n\tvzeroupper\n\t"
                     : PR_OUT : [p] "r"(p), [k] "r"(k), [io] "r"(io) : "xmm0", "k1", "memory", "cc" PR_CLOB_C);
}
TEST(masked_avx512_compress_store_bounded) {
    NEED(F_AVX512F);
    SM_SDE_XFAIL("probes the whole 64-byte operand of VPCOMPRESSD instead of popcnt(k) elements");
    sm_region rg; REGION4(rg);
    pr_begin();
    t_rng r; t_rng_seed(&r, 0xC0);
    uint8_t *B = sm_page(&rg, 1);
    for (int it = 0; it < 200; it++) {
        uint64_t k = it < 17 ? lowbits(it) << (16 - it) : (t_rng_next(&r) & 0xFFFF);
        int c = __builtin_popcountll(k);
        __attribute__((aligned(64))) uint32_t io[16]; t_rng_fill(&r, io, 64);
        uint8_t *p = B - 4 * c;
        uint8_t before[64]; memcpy(before, B - 128, 64);
        zc_compress_d_512(p, &k, io);
        PR_EXPECT_NONE("vpcompressd popcnt=%d ending at PAGE_NOACCESS", c);
        uint32_t want[16]; int j = 0;
        for (int i = 0; i < 16; i++) if ((k >> i) & 1) want[j++] = io[i];
        int ok = memcmp(p, want, 4 * c) == 0;
        if (c <= 16) ok &= memcmp(B - 128, before, 64) == 0;
        CHECKF(ok, "vpcompressd k=0x%04llx: memory mismatch", (unsigned long long)k);
        if (!ok) break;
    }
    pr_end();
    sm_region_free(&rg);
}
TEST(masked_avx512_expand_load_bounded) {
    NEED(F_AVX512F);
    SM_SDE_XFAIL("probes the whole 64-byte operand of VPEXPANDD instead of popcnt(k) elements");
    sm_region rg; REGION4(rg);
    pr_begin();
    t_rng r; t_rng_seed(&r, 0xE0);
    uint8_t *B = sm_page(&rg, 1);
    for (int it = 0; it < 200; it++) {
        uint64_t k = it < 17 ? lowbits(it) : (t_rng_next(&r) & 0xFFFF);
        int c = __builtin_popcountll(k);
        __attribute__((aligned(64))) uint32_t io[16], io0[16]; t_rng_fill(&r, io, 64); memcpy(io0, io, 64);
        const uint32_t *p = (const uint32_t *)(B - 4 * c);
        zc_expand_d_512((void *)p, &k, io);
        PR_EXPECT_NONE("vpexpandd popcnt=%d ending at PAGE_NOACCESS", c);
        int j = 0, ok = 1;
        for (int i = 0; i < 16; i++) ok &= io[i] == (((k >> i) & 1) ? p[j++] : io0[i]);
        CHECKF(ok, "vpexpandd k=0x%04llx: register mismatch", (unsigned long long)k);
        if (!ok) break;
    }
    pr_end();
    sm_region_free(&rg);
}

/* ======================================================================== */
/* gathers / scatters                                                          */
/* ======================================================================== */
typedef void (*gfn)(const void *b, const void *idx, void *m, void *d);
#define G2(fname, insn, RD, RI, SC) \
    T_NOINLINE static void fname(const void *b, const void *idx, void *m, void *d) { \
        __asm__ volatile("vmovdqu (%[d]), %%" RD "1\n\tvmovdqu (%[i]), %%" RI "2\n\tvmovdqu (%[m]), %%" RD "3\n\t" PR_ARM \
                         insn " %%" RD "3, (%[b],%%" RI "2," SC "), %%" RD "1\n\t" PR_DONE \
                         "vmovdqu %%" RD "1, (%[d])\n\tvmovdqu %%" RD "3, (%[m])\n\tvzeroupper\n\t" \
                         : PR_OUT : [b] "r"(b), [i] "r"(idx), [m] "r"(m), [d] "r"(d) : "xmm1", "xmm2", "xmm3", "memory", "cc" PR_CLOB_C); }
G2(g2_dd_y_avx2, "vpgatherdd", "ymm", "ymm", "4")
G2(g2_dq_y_avx2, "vpgatherdq", "ymm", "xmm", "8")
G2(g2_qd_x_avx2, "vpgatherqd", "xmm", "ymm", "4")
G2(g2_qq_y_avx2, "vpgatherqq", "ymm", "ymm", "8")
G2(g2_dps_y_avx2, "vgatherdps", "ymm", "ymm", "4")
G2(g2_qpd_y_avx2, "vgatherqpd", "ymm", "ymm", "8")
G2(g2_dd_x_avx2, "vpgatherdd", "xmm", "xmm", "4")
T_TARGET("avx512f") T_NOINLINE static void zg_dd_512(const void *b, const void *idx, void *k, void *d) {
    __asm__ volatile("kmovw (%[k]), %%k1\n\tvmovdqu64 (%[d]), %%zmm1\n\tvmovdqu64 (%[i]), %%zmm2\n\t" PR_ARM
                     "vpgatherdd (%[b],%%zmm2,4), %%zmm1%{%%k1%}\n\t" PR_DONE
                     "vmovdqu64 %%zmm1, (%[d])\n\tkmovw %%k1, (%[k])\n\tvzeroupper\n\t"
                     : PR_OUT : [b] "r"(b), [i] "r"(idx), [k] "r"(k), [d] "r"(d) : "xmm1", "xmm2", "k1", "memory", "cc" PR_CLOB_C);
}
T_TARGET("avx512f") T_NOINLINE static void zg_qq_512(const void *b, const void *idx, void *k, void *d) {
    __asm__ volatile("kmovw (%[k]), %%k1\n\tvmovdqu64 (%[d]), %%zmm1\n\tvmovdqu64 (%[i]), %%zmm2\n\t" PR_ARM
                     "vpgatherqq (%[b],%%zmm2,8), %%zmm1%{%%k1%}\n\t" PR_DONE
                     "vmovdqu64 %%zmm1, (%[d])\n\tkmovw %%k1, (%[k])\n\tvzeroupper\n\t"
                     : PR_OUT : [b] "r"(b), [i] "r"(idx), [k] "r"(k), [d] "r"(d) : "xmm1", "xmm2", "k1", "memory", "cc" PR_CLOB_C);
}
T_TARGET("avx512f") T_NOINLINE static void zs_dd_512(const void *b, const void *idx, void *k, void *d) {
    __asm__ volatile("kmovw (%[k]), %%k1\n\tvmovdqu64 (%[d]), %%zmm1\n\tvmovdqu64 (%[i]), %%zmm2\n\t" PR_ARM
                     "vpscatterdd %%zmm1, (%[b],%%zmm2,4)%{%%k1%}\n\t" PR_DONE
                     "kmovw %%k1, (%[k])\n\tvzeroupper\n\t"
                     : PR_OUT : [b] "r"(b), [i] "r"(idx), [k] "r"(k), [d] "r"(d) : "xmm1", "xmm2", "k1", "memory", "cc" PR_CLOB_C);
}
T_TARGET("avx512f") T_NOINLINE static void zs_qq_512(const void *b, const void *idx, void *k, void *d) {
    __asm__ volatile("kmovw (%[k]), %%k1\n\tvmovdqu64 (%[d]), %%zmm1\n\tvmovdqu64 (%[i]), %%zmm2\n\t" PR_ARM
                     "vpscatterqq %%zmm1, (%[b],%%zmm2,8)%{%%k1%}\n\t" PR_DONE
                     "kmovw %%k1, (%[k])\n\tvzeroupper\n\t"
                     : PR_OUT : [b] "r"(b), [i] "r"(idx), [k] "r"(k), [d] "r"(d) : "xmm1", "xmm2", "k1", "memory", "cc" PR_CLOB_C);
}
typedef struct gop { const char *name; gfn fn; int nl, isz, dsz, kmask, scatter, feat; } gop;
static const gop g_gops[] = {
    { "vpgatherdd_ymm", g2_dd_y_avx2, 8, 4, 4, 0, 0, F_AVX2 },
    { "vpgatherdd_xmm", g2_dd_x_avx2, 4, 4, 4, 0, 0, F_AVX2 },
    { "vpgatherdq_ymm", g2_dq_y_avx2, 4, 4, 8, 0, 0, F_AVX2 },
    { "vpgatherqd_xmm", g2_qd_x_avx2, 4, 8, 4, 0, 0, F_AVX2 },
    { "vpgatherqq_ymm", g2_qq_y_avx2, 4, 8, 8, 0, 0, F_AVX2 },
    { "vgatherdps_ymm", g2_dps_y_avx2, 8, 4, 4, 0, 0, F_AVX2 },
    { "vgatherqpd_ymm", g2_qpd_y_avx2, 4, 8, 8, 0, 0, F_AVX2 },
    { "vpgatherdd_zmm", zg_dd_512, 16, 4, 4, 1, 0, F_AVX512F },
    { "vpgatherqq_zmm", zg_qq_512, 8, 8, 8, 1, 0, F_AVX512F },
    { "vpscatterdd_zmm", zs_dd_512, 16, 4, 4, 1, 1, F_AVX512F },
    { "vpscatterqq_zmm", zs_qq_512, 8, 8, 8, 1, 1, F_AVX512F },
};
static const gop *gop_find(const char *n) { for (size_t i = 0; i < T_COUNTOF(g_gops); i++) if (!strcmp(g_gops[i].name, n)) return &g_gops[i]; return NULL; }

/* one gather/scatter at base with idx[]/bits; *exc gets exceptions; 1 = ok */
static int gop_check(const gop *g, uint8_t *base, const int64_t *idx, uint64_t bits, size_t chk_bytes, t_rng *r, char *why, size_t cap) {
    __attribute__((aligned(64))) uint8_t ib[64], mb[64], db[64], d0[64];
    static uint8_t snap[8192], exp[8192];
    memset(ib, 0, 64); memset(mb, 0, 64);
    for (int l = 0; l < g->nl; l++) { int64_t v = idx[l]; memcpy(ib + l * g->isz, &v, g->isz); }
    if (g->kmask) memcpy(mb, &bits, 8);
    else for (int l = 0; l < g->nl; l++) {
        for (int b = 0; b < g->dsz; b++) mb[l * g->dsz + b] = (uint8_t)t_rng_next(r);
        uint8_t *top = &mb[l * g->dsz + g->dsz - 1];
        if ((bits >> l) & 1) *top |= 0x80; else *top &= 0x7F;
    }
    t_rng_fill(r, db, 64); memcpy(d0, db, 64);
    if (chk_bytes > sizeof snap) chk_bytes = sizeof snap;
    if (g->scatter) memcpy(snap, base, chk_bytes);
    pr_clear();
    g->fn(base, ib, mb, db);
    if (g_pr.nexc) { char d[160]; pr_desc(d, sizeof d); snprintf(why, cap, "%s (bits=0x%llx)", d, (unsigned long long)bits); pr_clear(); return 0; }
    /* mask fully cleared on completion */
    int mbytes = g->kmask ? 2 : g->nl * g->dsz;
    for (int j = 0; j < mbytes; j++) if (mb[j]) { snprintf(why, cap, "mask not cleared on completion (byte %d = 0x%02x, bits=0x%llx)", j, mb[j], (unsigned long long)bits); return 0; }
    if (g->scatter) {
        memcpy(exp, snap, chk_bytes);
        for (int l = 0; l < g->nl; l++) if ((bits >> l) & 1) {
            size_t o = (size_t)(idx[l] * g->dsz);
            if (o + g->dsz <= chk_bytes) memcpy(exp + o, d0 + l * g->dsz, g->dsz);
        }
        for (size_t j = 0; j < chk_bytes; j++) if (base[j] != exp[j]) { snprintf(why, cap, "memory +%llu = 0x%02x want 0x%02x (bits=0x%llx)", (unsigned long long)j, base[j], exp[j], (unsigned long long)bits); return 0; }
    } else {
        for (int l = 0; l < g->nl; l++) {
            const uint8_t *want = ((bits >> l) & 1) ? base + idx[l] * g->dsz : d0 + l * g->dsz;
            if (memcmp(db + l * g->dsz, want, g->dsz)) {
                snprintf(why, cap, "lane %d (%s, idx %lld) wrong (bits=0x%llx)", l, ((bits >> l) & 1) ? "active" : "inactive",
                         (long long)idx[l], (unsigned long long)bits);
                return 0;
            }
        }
    }
    return 1;
}
static void gather_case(const char *name) {
    const gop *g = gop_find(name);
    REQUIRE(g);
    NEED((t_feature)g->feat);
    sm_region rg; REGION4(rg);
    pr_begin();
    t_rng r; t_rng_seed(&r, t_mix64(strlen(name) * 977 + (uint64_t)name[3]));
    uint8_t *base = sm_page(&rg, 0);
    size_t pg = rg.pg; int64_t nel = (int64_t)(pg / g->dsz);
    uint64_t full = lowbits(g->nl);
    char why[256]; int ok = 1;
    int64_t idx[16];
    /* far/bad indices for inactive lanes */
    int64_t bad[6]; int nb = 0;
    bad[nb++] = nel + 3;                              /* PAGE_NOACCESS page */
    bad[nb++] = nel + nel - 1;
    if (g->isz == 4) { bad[nb++] = 0x08000000; bad[nb++] = -0x08000000; bad[nb++] = 0x7FFFFFFF; bad[nb++] = (int64_t)(int32_t)0x80000000u; }
    else { bad[nb++] = (int64_t)((1ull << 40) / g->dsz); bad[nb++] = -(int64_t)((1ull << 40) / g->dsz); bad[nb++] = 0x7FFFFFFF; bad[nb++] = -0x10000000LL; }
    /* A: valid indices */
    for (int it = 0; it < 64 && ok; it++) {
        uint64_t bits = it == 0 ? full : it == 1 ? 0 : (t_rng_next(&r) & full);
        for (int l = 0; l < g->nl; l++) idx[l] = g->scatter ? (int64_t)(l * 7 + (it % 5)) * 3 % nel : (int64_t)t_rng_range(&r, (uint64_t)nel);
        ok = gop_check(g, base, idx, bits, pg, &r, why, sizeof why);
        CHECKF(ok, "%s valid indices: %s", name, why);
    }
    /* B: masked-off lanes point to PAGE_NOACCESS / unmapped / far addresses */
    for (int it = 0; it < 96 && ok; it++) {
        uint64_t bits = t_rng_next(&r) & full;
        for (int l = 0; l < g->nl; l++) idx[l] = ((bits >> l) & 1) ? (int64_t)((l * 13 + it) % nel) : bad[(l + it) % nb];
        ok = gop_check(g, base, idx, bits, pg, &r, why, sizeof why);
        CHECKF(ok, "%s inactive lanes at bad addresses: %s", name, why);
    }
    /* C: every lane bad, mask all clear */
    if (ok) {
        for (int l = 0; l < g->nl; l++) idx[l] = bad[l % nb];
        ok = gop_check(g, base, idx, 0, pg, &r, why, sizeof why);
        CHECKF(ok, "%s all lanes masked off at bad addresses: %s", name, why);
    }
    /* D: duplicate indices (gather only) */
    if (ok && !g->scatter) {
        for (int l = 0; l < g->nl; l++) idx[l] = 5;
        ok = gop_check(g, base, idx, full, pg, &r, why, sizeof why);
        CHECKF(ok, "%s duplicate indices: %s", name, why);
    }
#ifdef T_X64
    /* E: 64-bit indices giving non-canonical addresses in masked-off lanes: record */
    if (ok && g->isz == 8) {
        for (int l = 0; l < g->nl; l++) idx[l] = (l & 1) ? (int64_t)(0x4000000000000000ull / g->dsz) : l;
        uint64_t bits = 0x5555555555555555ull & full;
        int nc = gop_check(g, base, idx, bits, pg, &r, why, sizeof why);
        t_note("%s: masked-off non-canonical lanes -> %s", name, nc ? "no fault" : why);
    }
#endif
    pr_end();
    sm_region_free(&rg);
}
TEST(gather_avx2_vpgatherdd_ymm) { gather_case("vpgatherdd_ymm"); }
TEST(gather_avx2_vpgatherdd_xmm) { gather_case("vpgatherdd_xmm"); }
TEST(gather_avx2_vpgatherdq_ymm) { gather_case("vpgatherdq_ymm"); }
TEST(gather_avx2_vpgatherqd_xmm) { gather_case("vpgatherqd_xmm"); }
TEST(gather_avx2_vpgatherqq_ymm) { gather_case("vpgatherqq_ymm"); }
TEST(gather_avx2_vgatherdps_ymm) { gather_case("vgatherdps_ymm"); }
TEST(gather_avx2_vgatherqpd_ymm) { gather_case("vgatherqpd_ymm"); }
TEST(gather_avx512_vpgatherdd_zmm) { gather_case("vpgatherdd_zmm"); }
TEST(gather_avx512_vpgatherqq_zmm) { gather_case("vpgatherqq_zmm"); }
TEST(scatter_avx512_vpscatterdd_zmm) { gather_case("vpscatterdd_zmm"); }
TEST(scatter_avx512_vpscatterqq_zmm) { gather_case("vpscatterqq_zmm"); }
TEST(gather_avx2_active_lane_in_bad_page_faults) { NEED(F_AVX2); expect_av("gather_avx2_active", 0, 1); }
TEST(gather_avx512_active_lane_in_bad_page_faults) {
    NEED(F_AVX512F);
    expect_av("gather_512_active", 0, 1);
    SM_SDE_XFAIL("reports a fault of an emulated scatter as a read (ExceptionInformation[0]=0)");
    expect_av("scatter_512_active", 1, 1);
}

/* ======================================================================== */
/* restartable faults: the OS resolves a fault in the middle of a gather /     */
/* scatter / masked op and EXCEPTION_CONTINUE_EXECUTION must resume it with    */
/* the partially-updated vector + mask state intact                            */
/* ======================================================================== */
static void guard_pages(sm_region *rg, int first, int n) {
    DWORD old;
    for (int i = first; i < first + n; i++) VirtualProtect(sm_page(rg, i), rg->pg, PAGE_READWRITE | PAGE_GUARD, &old);
}
static int resume_gather_check(const gop *g, uint8_t *base, const int64_t *idx, uint64_t bits, const uint8_t *ref, char *why, size_t cap) {
    __attribute__((aligned(64))) uint8_t ib[64], mb[64], db[64], d0[64];
    memset(ib, 0, 64); memset(mb, 0, 64);
    for (int l = 0; l < g->nl; l++) memcpy(ib + l * g->isz, &idx[l], g->isz);
    if (g->kmask) memcpy(mb, &bits, 8);
    else for (int l = 0; l < g->nl; l++) if ((bits >> l) & 1) mb[l * g->dsz + g->dsz - 1] = 0x80;
    for (int j = 0; j < 64; j++) db[j] = d0[j] = (uint8_t)(0xA0 + j);
    pr_clear();
    g->fn(base, ib, mb, db);
    if (g_pr.nexc) { char d[160]; pr_desc(d, sizeof d); snprintf(why, cap, "not resumed: %s", d); pr_clear(); return 0; }
    int mbytes = g->kmask ? 2 : g->nl * g->dsz;
    for (int j = 0; j < mbytes; j++) if (mb[j]) { snprintf(why, cap, "mask byte %d = 0x%02x after completion", j, mb[j]); return 0; }
    if (!g->scatter)
        for (int l = 0; l < g->nl; l++) {
            const uint8_t *want = ((bits >> l) & 1) ? ref + idx[l] * g->dsz : d0 + l * g->dsz;
            if (memcmp(db + l * g->dsz, want, g->dsz)) { snprintf(why, cap, "lane %d wrong after restart (idx %lld)", l, (long long)idx[l]); return 0; }
        }
    return 1;
}
static void resume_gather_case(const char *name, int commit_mode) {
    const gop *g = gop_find(name);
    REQUIRE(g);
    NEED((t_feature)g->feat);
    int np = 4;
    size_t pg = sm_pg();
    DWORD prot[4];
    for (int i = 0; i < np; i++) prot[i] = commit_mode ? 0 : PAGE_READWRITE;
    sm_region rg; REQUIRE(sm_region_alloc(&rg, np, prot));
    uint8_t *ref = (uint8_t *)malloc(np * pg); REQUIRE(ref);
    for (int i = 0; i < np; i++) sm_fill_page_pat(ref + i * pg, pg, i * pg);
    if (!commit_mode) guard_pages(&rg, 0, np);
    pr_begin();
    pr_region(rg.base, np * pg, 64);
    int64_t epp = (int64_t)(pg / g->dsz), idx[16];
    uint64_t full = lowbits(g->nl);
    char why[256];
    /* phase 1: active lanes in pages 0 and 2 only, masked-off lanes in pages 1 and 3 */
    uint64_t bits = 0;
    for (int l = 0; l < g->nl; l++) {
        int active = (l % 2) == 0;
        int page = active ? ((l / 2) % 2) * 2 : 1 + ((l / 2) % 2) * 2;
        idx[l] = page * epp + (l * 5 + 3) % epp;
        if (active) bits |= 1ull << l;
    }
    int ok = resume_gather_check(g, rg.base, idx, bits, ref, why, sizeof why);
    CHECKF(ok, "%s phase 1: %s", name, why);
    LONG fixes1 = commit_mode ? g_pr.ncommit : g_pr.nguard;
    CHECKF(fixes1 == 2, "%s phase 1: %ld %s, want 2 (one per page touched by active lanes)", name, fixes1, commit_mode ? "demand commits" : "guard violations");
    if (!commit_mode) {
        CHECKF(sm_is_guard(sm_page(&rg, 1)) && sm_is_guard(sm_page(&rg, 3)), "%s: a masked-off lane consumed a PAGE_GUARD", name);
        CHECKF(!sm_is_guard(sm_page(&rg, 0)) && !sm_is_guard(sm_page(&rg, 2)), "%s: touched pages still PAGE_GUARD", name);
    } else {
        MEMORY_BASIC_INFORMATION m;
        CHECKF(VirtualQuery(sm_page(&rg, 1), &m, sizeof m) && m.State == MEM_RESERVE, "%s: a masked-off lane caused page 1 to be committed", name);
    }
    /* phase 2: every lane active, spread over all pages (pages 1 and 3 still need fixing) */
    for (int l = 0; l < g->nl; l++) idx[l] = (l % np) * epp + (l * 11 + 1) % epp;
    ok = resume_gather_check(g, rg.base, idx, full, ref, why, sizeof why);
    CHECKF(ok, "%s phase 2: %s", name, why);
    LONG fixes2 = (commit_mode ? g_pr.ncommit : g_pr.nguard) - fixes1;
    CHECKF(fixes2 == 2, "%s phase 2: %ld more fixes, want 2", name, fixes2);
    if (g->scatter) {
        /* scatter wrote the d0 pattern lanes: check memory */
        int bad = 0;
        for (int l = 0; l < g->nl; l++) { uint8_t want[8]; for (int b = 0; b < g->dsz; b++) want[b] = (uint8_t)(0xA0 + l * g->dsz + b);
            bad |= memcmp(rg.base + idx[l] * g->dsz, want, g->dsz) != 0; }
        CHECKF(!bad, "%s: scattered memory wrong after restarts", name);
    }
    pr_end();
    free(ref);
    sm_region_free(&rg);
}
TEST(resume_gather_avx2_guard_pages)        { resume_gather_case("vpgatherdd_ymm", 0); }
TEST(resume_gather_avx2_qq_guard_pages)     { resume_gather_case("vpgatherqq_ymm", 0); }
TEST(resume_gather_avx2_commit_on_demand)   { resume_gather_case("vpgatherdd_ymm", 1); }
TEST(resume_gather_avx512_guard_pages)      { resume_gather_case("vpgatherdd_zmm", 0); }
TEST(resume_gather_avx512_commit_on_demand) {
    SM_SDE_XFAIL("does not complete the gather elements below the faulting one before the fault");
    resume_gather_case("vpgatherdd_zmm", 1);
}
TEST(resume_scatter_avx512_guard_pages)     { resume_gather_case("vpscatterdd_zmm", 0); }

static void resume_masked_case(mfn fn, int es, int vb, int kmask, int is_store, const char *name) {
    size_t pg = sm_pg();
    static const DWORD prot[3] = { PAGE_READWRITE, PAGE_READWRITE, PAGE_READWRITE };
    sm_region rg; REQUIRE(sm_region_alloc(&rg, 3, prot));
    guard_pages(&rg, 0, 2);
    pr_begin();
    pr_region(rg.base, 3 * pg, 16);
    int n = vb / es, half = n / 2;
    uint8_t *p = sm_page(&rg, 1) - half * es;        /* first half in page 0, second half in page 1 */
    __attribute__((aligned(64))) uint8_t mb[64], io[64], io0[64];
    mop m = { name, "", fn, is_store ? MO_ST : MO_LDZ, es, vb, kmask, 0 };
    t_rng r; t_rng_seed(&r, 17);
    for (int phase = 0; phase < 2; phase++) {
        uint64_t bits = phase == 0 ? lowbits(half) : lowbits(n);
        mask_build(&m, bits, mb, &r);
        for (int j = 0; j < 64; j++) io[j] = io0[j] = (uint8_t)(0x30 + j + phase);
        pr_clear();
        fn(p, mb, io);
        CHECKF(g_pr.nexc == 0, "%s phase %d: not resumed (exception 0x%08lx)", name, phase, (unsigned long)g_pr.code);
        pr_clear();
        if (sm_vex_mask_fault_impl_dep(&m) && (g_pr.nguard != phase + 1 || (phase == 0 && !sm_is_guard(sm_page(&rg, 1))))) {
            /* masked-off elements touched the guard page: allowed on this vendor (see sm_vex_mask_fault_impl_dep) */
            t_note("%s phase %d: %ld guard violations (masked-off elements accessed; implementation dependent on %s)",
                   name, phase, g_pr.nguard, cpu()->vendor);
            break;
        }
        CHECKF(g_pr.nguard == phase + 1, "%s phase %d: %ld guard violations, want %d", name, phase, g_pr.nguard, phase + 1);
        if (phase == 0) CHECKF(sm_is_guard(sm_page(&rg, 1)), "%s: masked-off half consumed PAGE_GUARD of the next page", name);
        for (int i = 0; i < n; i++) {
            int act = (bits >> i) & 1;
            for (int b = 0; b < es; b++) {
                int k = i * es + b;
                if (is_store) { if (act) CHECKF(p[k] == io0[k], "%s phase %d: stored byte %d wrong", name, phase, k); }
                else {
                    uint8_t want = act ? p[k] : 0;
                    if (io[k] != want) { CHECKF(0, "%s phase %d: loaded element %d byte %d = 0x%02x want 0x%02x", name, phase, i, b, io[k], want); i = n; break; }
                }
            }
        }
    }
    pr_end();
    sm_region_free(&rg);
}
TEST(resume_vmaskmovps_store_guard_pages) { NEED(F_AVX); resume_masked_case(vm_ps_y_st_avx, 4, 32, 0, 1, "vmaskmovps ymm store"); }
TEST(resume_vpmaskmovq_load_guard_pages)  { NEED(F_AVX2); resume_masked_case(vm_q_y_ld_avx2, 8, 32, 0, 0, "vpmaskmovq ymm load"); }
TEST(resume_avx512_masked_load_guard_pages)  { NEED(F_AVX512F); resume_masked_case(zm_d32_z_ldz_512, 4, 64, 1, 0, "vmovdqu32 zmm{z} load"); }
TEST(resume_avx512_masked_store_guard_pages) { NEED(F_AVX512F); resume_masked_case(zm_d32_z_st_512, 4, 64, 1, 1, "vmovdqu32 zmm store"); }

static void resume_nt_case(int ki) {
    const sm_kernel *k = &g_k[ki];
    NEED_K(k);
    size_t pg = sm_pg(); int np = 4;
    static const DWORD prot[4] = { 0, 0, 0, 0 };
    sm_region rg; REQUIRE(sm_region_alloc(&rg, np, prot));
    uint8_t *src = sm_valloc(np * pg); REQUIRE(src);
    t_rng r; t_rng_seed(&r, 8); t_rng_fill(&r, src, np * pg);
    pr_begin();
    pr_region(rg.base, np * pg, 16);
    pr_clear();
    __asm__ volatile("" ::: "memory");
    k->fn(rg.base, src, np * pg);
    sm_sfence();
    CHECKF(g_pr.ncommit == np, "%s: %ld demand commits, want %d", k->name, g_pr.ncommit, np);
    CHECKF(memcmp(rg.base, src, np * pg) == 0, "%s: data wrong after demand-commit restarts", k->name);
    pr_end();
    sm_vfree(src);
    sm_region_free(&rg);
}
TEST(resume_nt_movntdq_commit_on_demand)      { resume_nt_case(K_MOVNTDQ); }
TEST(resume_nt_movnti32_commit_on_demand)     { resume_nt_case(K_MOVNTI32); }
TEST(resume_nt_vmovntdq_ymm_commit_on_demand) { resume_nt_case(K_VMOVNTDQ_Y); }
TEST(resume_nt_vmovntdq_zmm_commit_on_demand) { resume_nt_case(K_VMOVNTDQ_Z); }

/* ---- the exception CONTEXT of a gather that faults part-way shows the architectural partial progress ----
 * SDM Vol 2 VPGATHERDD / VPGATHERDD(EVEX): "If a fault is triggered by an element and delivered, all elements
 * closer to the LSB of the destination will be completed (and non-faulting). Individual elements closer to
 * the MSB may or may not be completed." Completed elements are written to the destination and their mask
 * element (AVX2) / mask bit (AVX-512) is cleared; the faulting element keeps its mask. The fault is a fault:
 * the context IP is the gather itself. Windows hands this state to the handler in the exception CONTEXT
 * (XMM part in the legacy FXSAVE area; YMM upper halves and opmasks only when the context carries
 * CONTEXT_XSTATE, located with LocateXStateFeature - kernelbase decomp: NULL unless ContextFlags has
 * CONTEXT_XSTATE, features 0/1 map to the legacy area). The gather runs on a PAGE_GUARD page, so the
 * handler simply continues and the OS-restored state lets the instruction finish. */
typedef struct gc_cap {
    volatile LONG hits; volatile DWORD tid; uint8_t *lo, *hi;
    DWORD code, flags; ULONG_PTR ip, addr;
    int have_xmm, have_ymmh, have_k;
    uint8_t x1[16], x3[16], yh1[16], yh3[16]; uint64_t k1;
} gc_cap;
static gc_cap g_gc;
static volatile ULONG_PTR g_gc_insn;
static LONG CALLBACK gc_veh(PEXCEPTION_POINTERS ep) {
    if (!g_gc.tid || GetCurrentThreadId() != g_gc.tid) return EXCEPTION_CONTINUE_SEARCH;
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    if (er->ExceptionCode != SM_GUARD || er->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    uint8_t *a = (uint8_t *)er->ExceptionInformation[1];
    if (a < g_gc.lo || a >= g_gc.hi) return EXCEPTION_CONTINUE_SEARCH;
    if (InterlockedIncrement(&g_gc.hits) == 1) {
        CONTEXT *c = ep->ContextRecord;
        g_gc.code = er->ExceptionCode; g_gc.addr = (ULONG_PTR)a; g_gc.flags = c->ContextFlags;
#ifdef T_X64
        g_gc.ip = (ULONG_PTR)c->Rip;
        if ((c->ContextFlags & CONTEXT_FLOATING_POINT) == CONTEXT_FLOATING_POINT) {
            memcpy(g_gc.x1, &c->FltSave.XmmRegisters[1], 16); memcpy(g_gc.x3, &c->FltSave.XmmRegisters[3], 16); g_gc.have_xmm = 1;
        }
#else
        g_gc.ip = (ULONG_PTR)c->Eip;
        if ((c->ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS) {   /* FXSAVE image: XMMn at 160+16n */
            memcpy(g_gc.x1, c->ExtendedRegisters + 160 + 16, 16); memcpy(g_gc.x3, c->ExtendedRegisters + 160 + 48, 16); g_gc.have_xmm = 1;
        }
#endif
        __typeof__(&LocateXStateFeature) loc = T_PROC("kernel32.dll", LocateXStateFeature);
        if (loc) {
            DWORD len = 0;
            uint8_t *yh = (uint8_t *)loc(c, XSTATE_AVX, &len);
            if (yh && len >= 64) { memcpy(g_gc.yh1, yh + 16, 16); memcpy(g_gc.yh3, yh + 48, 16); g_gc.have_ymmh = 1; }
            len = 0;
            uint8_t *km = (uint8_t *)loc(c, 5 /* XSTATE_AVX512_KMASK */, &len);
            if (km && len >= 16) { memcpy(&g_gc.k1, km + 8, 8); g_gc.have_k = 1; }
        }
    }
    return EXCEPTION_CONTINUE_EXECUTION;   /* the OS already removed PAGE_GUARD: the gather re-executes and completes */
}
#ifdef T_X64
#define GC_ARM "lea 8f(%%rip), %%r10\n\tmov %%r10, %[ia]\n\t"
#define GC_CLOB , "r10"
#else
#define GC_ARM "movl $8f, %[ia]\n\t"
#define GC_CLOB
#endif
T_NOINLINE static void gcf_dd_y_avx2(const void *b, const void *idx, void *m, void *d) {
    __asm__ volatile("vmovdqu (%[d]), %%ymm1\n\tvmovdqu (%[i]), %%ymm2\n\tvmovdqu (%[m]), %%ymm3\n\t" GC_ARM
                     "8:\n\tvpgatherdd %%ymm3, (%[b],%%ymm2,4), %%ymm1\n\t"
                     "vmovdqu %%ymm1, (%[d])\n\tvmovdqu %%ymm3, (%[m])\n\tvzeroupper\n\t"
                     : [ia] "=m"(g_gc_insn) : [b] "r"(b), [i] "r"(idx), [m] "r"(m), [d] "r"(d)
                     : "xmm1", "xmm2", "xmm3", "memory" GC_CLOB);
}
T_TARGET("avx512f") T_NOINLINE static void gcf_dd_z_512(const void *b, const void *idx, void *k, void *d) {
    __asm__ volatile("kmovw (%[k]), %%k1\n\tvmovdqu64 (%[d]), %%zmm1\n\tvmovdqu64 (%[i]), %%zmm2\n\t" GC_ARM
                     "8:\n\tvpgatherdd (%[b],%%zmm2,4), %%zmm1%{%%k1%}\n\t"
                     "vmovdqu64 %%zmm1, (%[d])\n\tkmovw %%k1, (%[k])\n\tvzeroupper\n\t"
                     : [ia] "=m"(g_gc_insn) : [b] "r"(b), [i] "r"(idx), [k] "r"(k), [d] "r"(d)
                     : "xmm1", "xmm2", "k1", "memory" GC_CLOB);
}
static void gather_fault_context_case(int evex) {
    if (evex) NEED(F_AVX512F); else NEED(F_AVX2);
    static const DWORD prot[2] = { PAGE_READWRITE, PAGE_READWRITE | PAGE_GUARD };
    sm_region rg; REQUIRE(sm_region_alloc(&rg, 2, prot));
    REQUIRE(sm_is_guard(sm_page(&rg, 1)));
    size_t pg = rg.pg; int32_t epp = (int32_t)(pg / 4);
    int nl = evex ? 16 : 8;
    /* lanes 3 and 5 (and 9, 14 for EVEX) in the guard page, everything else in page 0 */
    __attribute__((aligned(64))) int32_t idx[16];
    __attribute__((aligned(64))) uint32_t m[16], d[16], d0[16];
    uint16_t k = (uint16_t)((1u << nl) - 1);
    int guard_lane[16] = { 0 };
    guard_lane[3] = guard_lane[5] = 1; if (evex) guard_lane[9] = guard_lane[14] = 1;
    for (int l = 0; l < 16; l++) {
        idx[l] = guard_lane[l] ? epp + 40 + 7 * l : 100 + 13 * l;
        m[l] = l < nl ? 0xFFFFFFFFu : 0;
        d[l] = d0[l] = 0xD0D00000u + (uint32_t)l;
    }
    const uint32_t *base = (const uint32_t *)rg.base;
    uint32_t want[16];
    for (int l = 0; l < nl; l++) want[l] = sm_pat32((uint64_t)idx[l] * 4, 0x5EED);
    pr_end();
    memset((void *)&g_gc, 0, sizeof g_gc);
    g_gc.lo = rg.base; g_gc.hi = rg.base + 2 * pg;
    PVOID vh = AddVectoredExceptionHandler(1, gc_veh);
    REQUIRE(vh);
    g_gc.tid = GetCurrentThreadId();
    if (evex) gcf_dd_z_512(base, idx, &k, d); else gcf_dd_y_avx2(base, idx, m, d);
    g_gc.tid = 0;
    RemoveVectoredExceptionHandler(vh);
    const char *nm = evex ? "vpgatherdd zmm" : "vpgatherdd ymm";
    /* after the restart: complete */
    for (int l = 0; l < nl; l++) CHECKF(d[l] == want[l], "%s: lane %d = 0x%08x want 0x%08x after the restart", nm, l, d[l], want[l]);
    if (evex) CHECKF(k == 0, "%s: k1 = 0x%04x after completion", nm, k);
    else for (int l = 0; l < nl; l++) CHECKF(m[l] == 0, "%s: mask lane %d = 0x%08x after completion", nm, l, m[l]);
    CHECKF(!sm_is_guard(sm_page(&rg, 1)), "%s: PAGE_GUARD still set", nm);
    REQUIREF(g_gc.hits == 1, "%s: %ld guard-page exceptions, want exactly 1", nm, g_gc.hits);
    CHECKF(g_gc.ip == g_gc_insn, "%s: context IP %p is not the gather at %p (a fault must report the faulting instruction)",
           nm, (void *)g_gc.ip, (void *)g_gc_insn);
    /* The right-to-left delivery order is stated by the Intel SDM; on other vendors only per-lane consistency
     * (each lane either complete with its mask cleared, or pending and untouched) is checked. */
    int strict = !strcmp(cpu()->vendor, "GenuineIntel");
    ULONG_PTR a3 = (ULONG_PTR)(base + idx[3]);
    if (strict) CHECKF(g_gc.addr == a3, "%s: fault address %p, want lane 3 element %p (lowest faulting lane is delivered first)",
                       nm, (void *)g_gc.addr, (void *)a3);
    else {
        CHECKF(g_gc.addr >= (ULONG_PTR)sm_page(&rg, 1), "%s: fault address %p outside the guard page", nm, (void *)g_gc.addr);
        if (g_gc.addr != a3) t_note("%s: fault reported at %p, not the lowest faulting lane %p", nm, (void *)g_gc.addr, (void *)a3);
    }
    t_info("context_flags", "0x%lx", (unsigned long)g_gc.flags);
    if (!g_gc.have_xmm) {
#ifdef T_X64
        /* x64: CONTEXT_FULL includes CONTEXT_FLOATING_POINT; an exception context always carries XMM0-15 */
        CHECKF(0, "%s: x64 exception CONTEXT without CONTEXT_FLOATING_POINT (flags 0x%lx)", nm, (unsigned long)g_gc.flags);
#else
        t_note("%s: x86 exception CONTEXT without CONTEXT_EXTENDED_REGISTERS (flags 0x%lx): XMM not inspected", nm, (unsigned long)g_gc.flags);
#endif
        goto out;
    }
    {
        uint32_t x1[4], x3[4]; memcpy(x1, g_gc.x1, 16); memcpy(x3, g_gc.x3, 16);
        if (strict) {
            for (int l = 0; l < 3; l++) CHECKF(x1[l] == want[l], "%s: in the fault context, completed lane %d of the destination = 0x%08x want 0x%08x", nm, l, x1[l], want[l]);
            CHECKF(x1[3] == d0[3], "%s: in the fault context, faulting lane 3 destination = 0x%08x want unchanged 0x%08x", nm, x1[3], d0[3]);
            if (!evex) {
                for (int l = 0; l < 3; l++) CHECKF(x3[l] == 0, "%s: in the fault context, mask lane %d = 0x%08x want 0 (completed)", nm, l, x3[l]);
                CHECKF(x3[3] == 0xFFFFFFFFu, "%s: in the fault context, mask lane 3 = 0x%08x want 0xFFFFFFFF (faulting element)", nm, x3[3]);
            }
        } else if (!evex) {
            for (int l = 0; l < 4; l++) {
                CHECKF(x3[l] == 0 || x3[l] == 0xFFFFFFFFu, "%s: fault context mask lane %d = 0x%08x", nm, l, x3[l]);
                CHECKF(x1[l] == (x3[l] == 0 ? want[l] : d0[l]), "%s: fault context lane %d = 0x%08x inconsistent with its mask", nm, l, x1[l]);
            }
            CHECKF(x3[3] != 0, "%s: lane 3 in the not-present page reported complete", nm);
        }
    }
    /* upper lanes: each is either complete (mask clear + new value) or untouched; guard-page lanes untouched */
    if (g_gc.have_ymmh) {
        uint32_t yd[4], ym[4]; memcpy(yd, g_gc.yh1, 16); memcpy(ym, g_gc.yh3, 16);
        for (int j = 0; j < 4; j++) {
            int l = 4 + j;
            int done = evex ? !(g_gc.have_k && ((g_gc.k1 >> l) & 1)) : ym[j] == 0;
            if (!evex) CHECKF(ym[j] == 0 || ym[j] == 0xFFFFFFFFu, "%s: fault context mask lane %d = 0x%08x", nm, l, ym[j]);
            if (evex && !g_gc.have_k) break;
            CHECKF(yd[j] == (done ? want[l] : d0[l]), "%s: fault context lane %d = 0x%08x inconsistent with its mask (%s)", nm, l, yd[j], done ? "complete" : "pending");
            if (guard_lane[l]) CHECKF(!done, "%s: lane %d in the not-present page reported complete", nm, l);
        }
    } else t_note("%s: exception CONTEXT has no XSTATE (AVX) area (flags 0x%lx): upper lanes not inspected", nm, (unsigned long)g_gc.flags);
    if (evex) {
        if (g_gc.have_k) {
            if (strict) CHECKF((g_gc.k1 & 0xF) == 0x8, "%s: fault context k1 = 0x%04llx, want bits 0-2 clear and bit 3 set", nm, (unsigned long long)g_gc.k1);
            for (int l = 0; l < nl; l++) if (guard_lane[l]) CHECKF((g_gc.k1 >> l) & 1, "%s: k1 bit %d (guard-page lane) cleared in the fault context", nm, l);
        } else t_note("%s: exception CONTEXT has no AVX-512 opmask state", nm);
    }
out:
    sm_region_free(&rg);
}
TEST(gather_fault_context_partial_avx2)   { gather_fault_context_case(0); }
TEST(gather_fault_context_partial_avx512) {
    SM_SDE_XFAIL("does not complete the gather elements below the faulting one before the fault");
    gather_fault_context_case(1);
}

/* ======================================================================== */
/* locked operations                                                          */
/* ======================================================================== */
#ifdef T_X64
static int cx16(volatile void *p, uint64_t *lo, uint64_t *hi, uint64_t nlo, uint64_t nhi) {
    unsigned char z;
    __asm__ volatile("lock cmpxchg16b %1\n\tsetz %0" : "=q"(z), "+m"(*(volatile uint64_t (*)[2])p), "+a"(*lo), "+d"(*hi)
                     : "b"(nlo), "c"(nhi) : "memory", "cc");
    return z;
}
#endif
static int cx8(volatile void *p, uint32_t *lo, uint32_t *hi, uint32_t nlo, uint32_t nhi) {
    unsigned char z;
    __asm__ volatile("lock cmpxchg8b %1\n\tsetz %0" : "=q"(z), "+m"(*(volatile uint32_t (*)[2])p), "+a"(*lo), "+d"(*hi)
                     : "b"(nlo), "c"(nhi) : "memory", "cc");
    return z;
}
TEST(lock_cmpxchg16b_semantics) {
#ifdef T_X64
    NEED(F_CX16);
    __attribute__((aligned(16))) volatile uint64_t m[4] = { 0x1111111111111111ull, 0x2222222222222222ull, 0x5A5A, 0xA5A5 };
    uint64_t lo = 0x1111111111111111ull, hi = 0x2222222222222222ull;
    CHECK(cx16(m, &lo, &hi, 0x3333, 0x4444));
    CHECK_EQ(m[0], 0x3333); CHECK_EQ(m[1], 0x4444);
    CHECK_EQ(lo, 0x1111111111111111ull); CHECK_EQ(hi, 0x2222222222222222ull);
    /* mismatch in the high qword only */
    lo = 0x3333; hi = 0x4445;
    CHECK(!cx16(m, &lo, &hi, 1, 2));
    CHECK_EQ(lo, 0x3333); CHECK_EQ(hi, 0x4444);
    CHECK_EQ(m[0], 0x3333); CHECK_EQ(m[1], 0x4444);
    /* mismatch in the low qword only */
    lo = 0x3334; hi = 0x4444;
    CHECK(!cx16(m, &lo, &hi, 1, 2));
    CHECK_EQ(lo, 0x3333); CHECK_EQ(hi, 0x4444);
    CHECK_EQ(m[2], 0x5A5A); CHECK_EQ(m[3], 0xA5A5);   /* neighbours untouched */
    /* at the end of a page followed by PAGE_NOACCESS */
    void *gb = NULL; uint64_t *g = (uint64_t *)t_guarded_alloc(16, &gb); REQUIRE(g);
    g[0] = 7; g[1] = 8; lo = 7; hi = 8;
    CHECK(cx16(g, &lo, &hi, 9, 10));
    CHECK_EQ(g[0], 9); CHECK_EQ(g[1], 10);
    t_guarded_free(gb);
#else
    SKIP("CMPXCHG16B is a 64-bit mode instruction");
#endif
}
TEST(lock_cmpxchg16b_misaligned_faults) {
#ifdef T_X64
    NEED(F_CX16);
    expect_av("cmpxchg16b_mis", -1, 0);
#else
    SKIP("CMPXCHG16B is a 64-bit mode instruction");
#endif
}
TEST(lock_cmpxchg8b_semantics) {
    NEED(F_CX8);
    __attribute__((aligned(8))) volatile uint32_t m[4] = { 0x11, 0x22, 0x77, 0x88 };
    uint32_t lo = 0x11, hi = 0x22;
    CHECK(cx8(m, &lo, &hi, 0x33, 0x44));
    CHECK_EQ(m[0], 0x33); CHECK_EQ(m[1], 0x44);
    lo = 0x33; hi = 0x45;
    CHECK(!cx8(m, &lo, &hi, 1, 2));
    CHECK_EQ(lo, 0x33); CHECK_EQ(hi, 0x44);
    lo = 0x34; hi = 0x44;
    CHECK(!cx8(m, &lo, &hi, 1, 2));
    CHECK_EQ(lo, 0x33); CHECK_EQ(hi, 0x44);
    CHECK_EQ(m[2], 0x77); CHECK_EQ(m[3], 0x88);
    /* CMPXCHG8B has no alignment requirement */
    __attribute__((aligned(16))) uint8_t buf[32]; memset(buf, 0, 32);
    uint32_t z0 = 0, z1 = 0;
    CHECK(cx8(buf + 3, &z0, &z1, 0xDEADBEEF, 0xCAFEF00D));
    uint32_t a, b; memcpy(&a, buf + 3, 4); memcpy(&b, buf + 7, 4);
    CHECK_EQ(a, 0xDEADBEEFu); CHECK_EQ(b, 0xCAFEF00Du);
}
typedef struct cnt_ctx { volatile void *p; int iters; volatile LONG *go; int kind; } cnt_ctx;
static DWORD WINAPI cnt_thread(void *arg) {
    cnt_ctx *c = (cnt_ctx *)arg;
    unsigned spins = 0;
    while (!*c->go) { sm_pause(); if ((++spins & 1023) == 0) SwitchToThread(); }
    for (int i = 0; i < c->iters; i++) {
        if (c->kind == 0) {
#ifdef T_X64
            volatile uint64_t *q = (volatile uint64_t *)c->p;
            uint64_t lo = q[0], hi = q[1];
            for (;;) { uint64_t nlo = lo + 1, nhi = hi + (nlo == 0); if (cx16(q, &lo, &hi, nlo, nhi)) break; }
#endif
        } else if (c->kind == 1) {
            volatile uint32_t *q = (volatile uint32_t *)c->p;
            uint32_t lo = q[0], hi = q[1];
            for (;;) { uint32_t nlo = lo + 1, nhi = hi + (nlo == 0); if (cx8(q, &lo, &hi, nlo, nhi)) break; }
        } else {
            uint32_t one = 1;
            __asm__ volatile("lock xaddl %0, %1" : "+r"(one), "+m"(*(volatile uint32_t *)c->p) :: "memory", "cc");
        }
    }
    return 0;
}
static int counter_case(int kind, int nthreads, int iters, volatile void *p) {
    HANDLE h[16]; cnt_ctx c[16]; volatile LONG go = 0; int n = 0;
    for (int i = 0; i < nthreads; i++) {
        c[i].p = p; c[i].iters = iters; c[i].go = &go; c[i].kind = kind;
        h[i] = CreateThread(NULL, 65536, cnt_thread, &c[i], 0, NULL);
        if (!h[i]) break;
        n++;
    }
    if (n < 2) { for (int i = 0; i < n; i++) c[i].iters = 0; go = 1; for (int i = 0; i < n; i++) { WaitForSingleObject(h[i], 5000); CloseHandle(h[i]); } SKIP("cannot create threads"); }
    go = 1;
    DWORD w = WaitForMultipleObjects((DWORD)n, h, TRUE, 20000);
    CHECKF(w != WAIT_TIMEOUT, "counter threads did not finish in 20 s");
    for (int i = 0; i < n; i++) CloseHandle(h[i]);
    t_info("threads", "%d", n);
    return n;
}
TEST(lock_cmpxchg16b_threads_counter) {
#ifdef T_X64
    NEED(F_CX16);
    static __attribute__((aligned(16))) volatile uint64_t q[2];
    q[0] = 0xFFFFFFFFFFFFC000ull; q[1] = 5;
    int nt = t_cpu_count() < 4 ? 2 : 4, it = 20000;
    nt = counter_case(0, nt, it, q);
    unsigned long long total = (unsigned long long)nt * it;
    uint64_t wlo = 0xFFFFFFFFFFFFC000ull + total, whi = 5 + (wlo < 0xFFFFFFFFFFFFC000ull);
    CHECK_EQ(q[0], wlo); CHECK_EQ(q[1], whi);
#else
    SKIP("CMPXCHG16B is a 64-bit mode instruction");
#endif
}
TEST(lock_cmpxchg8b_threads_counter) {
    NEED(F_CX8);
    static __attribute__((aligned(8))) volatile uint32_t q[2];
    q[0] = 0xFFFFC000u; q[1] = 9;
    int nt = t_cpu_count() < 4 ? 2 : 4, it = 20000;
    nt = counter_case(1, nt, it, q);
    uint32_t wlo = 0xFFFFC000u + (uint32_t)(nt * it), whi = 9 + (wlo < 0xFFFFC000u);
    CHECK_EQ(q[0], wlo); CHECK_EQ(q[1], whi);
}
TEST(lock_xadd_threads_counter) {
    static __attribute__((aligned(64))) volatile uint32_t v[16];
    v[0] = 0; v[1] = 0x77;
    int nt = t_cpu_count(), it = 50000;
    if (nt < 2) nt = 2;
    if (nt > 8) nt = 8;
    nt = counter_case(2, nt, it, v);
    CHECK_EQ(v[0], (uint32_t)(nt * it));
    CHECK_EQ(v[1], 0x77);
}
TEST(lock_cmpxchg_readonly_page_write_faults) {
    /* CMPXCHG always performs a write cycle: a failing compare still faults on a read-only page */
    expect_av("lock_cmpxchg_ro", 1, 1);
}
TEST(lock_split_cacheline_info) {
    sm_fres f; run_fop("split_lock_info", &f);
    if (!f.ok) return;
    char v[128];
    if (f.faulted) t_note("split-locked add raised 0x%08lx (OS split-lock detection)", (unsigned long)f.code);
    else if (t_child_find(&g_cr, "SM_SPLIT ", v, sizeof v)) {
        unsigned val = 0; double ms = 0; sscanf(v, "value=%u ms=%lf", &val, &ms);
        CHECKF(val == 42, "split-locked add result %u want 42", val);
        t_info("split_lock_ms", "%.3f", ms);
    } else CHECKF(0, "split lock child produced no result (exit 0x%lx)", (unsigned long)g_cr.exit_code);
}

/* ======================================================================== */
/* RDRAND / RDSEED                                                            */
/* ======================================================================== */
#define RR_FN(fname, insn, T, REGM) \
    static int fname(T *v, uintptr_t *fl) { T x; uintptr_t f, pat = SM_FL; \
        __asm__ volatile("pushf\n\tpop %1\n\tor %3, %1\n\tpush %1\n\tpopf\n\t" insn " %" REGM "0\n\tpushf\n\tpop %1" \
                         : "=&r"(x), "=&r"(f) : "0"((T)0x5A5A), "r"(pat) : "cc", "memory"); \
        *v = x; if (fl) *fl = f; return (int)(f & 1); }
RR_FN(rdrand16, "rdrand", uint16_t, "w")
RR_FN(rdrand32, "rdrand", uint32_t, "k")
RR_FN(rdseed16, "rdseed", uint16_t, "w")
RR_FN(rdseed32, "rdseed", uint32_t, "k")
#ifdef T_X64
RR_FN(rdrand64, "rdrand", uint64_t, "q")
RR_FN(rdseed64, "rdseed", uint64_t, "q")
#endif
/* collect n values (retry up to `retries` per value); checks flags and CF=0 => dest=0 */
typedef int (*rr16)(uint16_t *, uintptr_t *);
typedef int (*rr32)(uint32_t *, uintptr_t *);
typedef struct rr_stat { int got, failed_values, cf0_seen, cf0_nonzero, flag_bad; uintptr_t bad_fl; } rr_stat;
static void rr_collect(int width, int seed, int n, int retries, uint64_t *out, rr_stat *s) {
    memset(s, 0, sizeof *s);
    for (int i = 0; i < n; i++) {
        int ok = 0;
        for (int t = 0; t < retries && !ok; t++) {
            uintptr_t fl = 0; uint64_t v = 0; int cf = 0;
            if (width == 16) { uint16_t x; cf = seed ? rdseed16(&x, &fl) : rdrand16(&x, &fl); v = x; }
            else if (width == 32) { uint32_t x; cf = seed ? rdseed32(&x, &fl) : rdrand32(&x, &fl); v = x; }
#ifdef T_X64
            else { uint64_t x; cf = seed ? rdseed64(&x, &fl) : rdrand64(&x, &fl); v = x; }
#endif
            if (fl & (SM_FL & ~1u)) { s->flag_bad++; s->bad_fl = fl; }
            if (cf) { out[s->got++] = v; ok = 1; }
            else { s->cf0_seen++; if (v) s->cf0_nonzero++; if (seed) sm_pause(); }
        }
        if (!ok) s->failed_values++;
    }
}
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void rr_case(int width, int seed) {
    NEED(seed ? F_RDSEED : F_RDRAND);
#ifndef T_X64
    if (width == 64) SKIP("64-bit operand size needs 64-bit mode");
#endif
    const char *nm = seed ? "rdseed" : "rdrand";
    enum { N = 1024 };
    static uint64_t v[N];
    rr_stat s;
    rr_collect(width, seed, N, seed ? 2000 : 10, v, &s);
    CHECKF(s.flag_bad == 0, "%s%d: OF/SF/ZF/AF/PF not cleared (%d times, last eflags 0x%lx)", nm, width, s.flag_bad, (unsigned long)s.bad_fl);
    CHECKF(s.cf0_nonzero == 0, "%s%d: CF=0 with a non-zero destination %d times", nm, width, s.cf0_nonzero);
    if (seed) {
        CHECKF(s.got > 0, "%s%d: no value obtained in %d x 2000 attempts", nm, width, N);
        if (s.failed_values) t_note("%s%d: %d/%d values not obtained within 2000 retries", nm, width, s.failed_values, N);
    } else CHECKF(s.failed_values == 0, "%s%d: %d/%d values failed after 10 retries (Intel DRNG guidance)", nm, width, s.failed_values, N);
    t_info("cf0_retries", "%d", s.cf0_seen);
    if (s.got < 16) { t_note("%s%d: only %d values obtained: distribution checks skipped", nm, width, s.got); if (seed) SKIP("%s%d: fewer than 16 values obtained", nm, width); return; }
    qsort(v, (size_t)s.got, sizeof v[0], cmp_u64);
    int distinct = 1, dups = 0;
    for (int i = 1; i < s.got; i++) { if (v[i] != v[i - 1]) distinct++; else dups++; }
    uint64_t allones = width == 64 ? ~0ull : ((1ull << width) - 1);
    int ones = 0; for (int i = 0; i < s.got; i++) ones += v[i] == allones;
    CHECKF(distinct > 1, "%s%d: all %d values identical (0x%llx)", nm, width, s.got, (unsigned long long)v[0]);
    CHECKF(ones < 3, "%s%d: %d all-ones values (known broken-RNG signature)", nm, width, ones);
    if (width == 64) CHECKF(dups == 0, "%s64: %d duplicate values among %d", nm, dups, s.got);
    else if (width == 32) { if (dups) t_note("%s32: %d duplicates among %d (possible by chance)", nm, dups, s.got); }
    else CHECKF(distinct > s.got * 85 / 100, "%s16: only %d distinct of %d", nm, distinct, s.got);
    t_info("distinct", "%d/%d", distinct, s.got);
}
TEST(rdrand_16)  { rr_case(16, 0); }
TEST(rdrand_32)  { rr_case(32, 0); }
TEST(rdrand_64)  { rr_case(64, 0); }
TEST(rdseed_16)  { rr_case(16, 1); }
TEST(rdseed_32)  { rr_case(32, 1); }
TEST(rdseed_64)  { rr_case(64, 1); }
TEST(rdrand_bit_balance) {
    NEED(F_RDRAND);
    enum { N = 4096 };
    static uint64_t v[N]; rr_stat s;
    rr_collect(32, 0, N, 10, v, &s);
    REQUIRE(s.got > N / 2);
    long ones = 0; for (int i = 0; i < s.got; i++) ones += __builtin_popcountll(v[i]);
    double bits = 32.0 * s.got, mean = bits / 2, sigma = sqrt(bits) / 2;
    CHECKF(fabs((double)ones - mean) < 8 * sigma, "rdrand32: %ld ones in %.0f bits (mean %.0f, sigma %.1f)", ones, bits, mean, sigma);
    /* per-bit-position balance */
    int worst = 0;
    for (int b = 0; b < 32; b++) { int c = 0; for (int i = 0; i < s.got; i++) c += (int)((v[i] >> b) & 1);
        int dev = abs(c - s.got / 2); if (dev > worst) worst = dev; }
    double bs = sqrt((double)s.got) / 2;
    CHECKF(worst < 8 * bs, "rdrand32: bit position imbalance %d (sigma %.1f)", worst, bs);
    t_info("ones_fraction", "%.5f", (double)ones / bits);
}
typedef struct rr_thr { uint64_t v[256]; int got; } rr_thr;
static DWORD WINAPI rr_thread(void *arg) {
    rr_thr *t = (rr_thr *)arg; rr_stat s;
#ifdef T_X64
    rr_collect(64, 0, 256, 10, t->v, &s);
#else
    uint64_t tmp[512];
    rr_collect(32, 0, 512, 10, tmp, &s);
    s.got /= 2;
    for (int i = 0; i < s.got; i++) t->v[i] = tmp[2 * i] << 32 | tmp[2 * i + 1];
#endif
    t->got = s.got;
    return 0;
}
TEST(rdrand_threads_unique) {
    NEED(F_RDRAND);
    enum { NT = 8 };
    static rr_thr t[NT]; HANDLE h[NT]; int n = 0;
    memset(t, 0, sizeof t);
    for (int i = 0; i < NT; i++) { h[i] = CreateThread(NULL, 65536, rr_thread, &t[i], 0, NULL); if (!h[i]) break; n++; }
    if (!n) SKIP("cannot create threads");
    WaitForMultipleObjects((DWORD)n, h, TRUE, 10000);
    for (int i = 0; i < n; i++) CloseHandle(h[i]);
    static uint64_t all[NT * 256]; int m = 0;
    for (int i = 0; i < n; i++) { CHECKF(t[i].got == 256, "thread %d obtained %d/256 values", i, t[i].got); for (int j = 0; j < t[i].got; j++) all[m++] = t[i].v[j]; }
    qsort(all, (size_t)m, sizeof all[0], cmp_u64);
    int dups = 0; for (int i = 1; i < m; i++) dups += all[i] == all[i - 1];
    CHECKF(dups == 0, "%d duplicate 64-bit values across %d threads (%d values)", dups, n, m);
    t_info("values", "%d", m);
}

/* ======================================================================== */
/* MOVDIRI / MOVDIR64B, WAITPKG                                               */
/* ======================================================================== */
TEST(movdiri_direct_store) {
    NEED(F_MOVDIRI);
    uint8_t *b = sm_valloc(4096); REQUIRE(b);
    pr_begin();
    for (int i = 0; i < 64; i++) {
        uint32_t *p = (uint32_t *)(b + 4 * i); uint32_t v = 0xD1D1D100u + (uint32_t)i;
        __asm__ volatile(PR_ARM "movdiri %[v], (%[p])" PR_DONE : PR_OUT : [p] "r"(p), [v] "r"(v) : "memory" PR_CLOB_C);
        PR_EXPECT_NONE("movdiri32");
        sm_sfence();
        CHECKF(*p == v, "movdiri32 [%d] = 0x%08x", i, *p);
    }
#ifdef T_X64
    uint64_t *q = (uint64_t *)(b + 1024), v = 0x0102030405060708ull;
    __asm__ volatile(PR_ARM "movdiri %[v], (%[p])" PR_DONE : PR_OUT : [p] "r"(q), [v] "r"(v) : "memory" PR_CLOB_C);
    PR_EXPECT_NONE("movdiri64");
    sm_sfence();
    CHECK_EQ(*q, v);
#endif
    pr_end();
    sm_vfree(b);
}
TEST(movdir64b_direct_store) {
    NEED(F_MOVDIR64B);
    uint8_t *b = sm_valloc(8192); REQUIRE(b);
    t_rng r; t_rng_seed(&r, 64);
    pr_begin();
    for (int it = 0; it < 32; it++) {
        uint8_t *src = b + 4096 + it * 3, *dst = b + 64 * it;   /* source any alignment, destination 64-aligned */
        t_rng_fill(&r, src, 64);
        __asm__ volatile(PR_ARM "movdir64b (%[s]), %[d]" PR_DONE : PR_OUT : [d] "r"(dst), [s] "r"(src) : "memory" PR_CLOB_C);
        PR_EXPECT_NONE("movdir64b");
        sm_sfence();
        CHECK_MEM(dst, src, 64);
    }
    pr_end();
    sm_vfree(b);
}
TEST(movdir64b_misaligned_destination_faults) {
    NEED(F_MOVDIR64B);
    SM_SDE_XFAIL("does not raise #GP for a misaligned MOVDIR64B destination");
    expect_av("movdir64b_mis", -1, 0);
}
TEST(waitpkg_tpause_umwait_bounded) {
    NEED(F_WAITPKG); NEED(F_TSC);
    sm_fres f; run_fop("waitpkg_info", &f);
    if (!f.ok) return;
    CHECKF(!f.faulted, "tpause/umwait advertised but raised 0x%08lx", (unsigned long)f.code);
    char v[160];
    if (t_child_find(&g_cr, "SM_WAITPKG ", v, sizeof v)) t_note("waitpkg: %s", v);
}

/* ======================================================================== */
/* write-combining / uncached memory (behaviour recorded as NOTE)             */
/* ======================================================================== */
static void special_mem_case(DWORD flag, const char *nm) {
    NEED(F_SSE2);
    size_t n = 1 << 20;
    uint8_t *m = (uint8_t *)VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | flag);
    if (!m) SKIP("VirtualAlloc(PAGE_READWRITE|%s) failed (%lu)", nm, GetLastError());
    uint8_t *src = sm_valloc(n), *back = sm_valloc(n);
    REQUIRE(src && back);
    t_rng r; t_rng_seed(&r, flag); t_rng_fill(&r, src, n);
    MEMORY_BASIC_INFORMATION mb;
    if (VirtualQuery(m, &mb, sizeof mb)) t_info("protect", "0x%lx", (unsigned long)mb.Protect);
    double t0 = t_now_ms();
    k_movntdq(m, src, n); sm_sfence();
    double t1 = t_now_ms();
    if (cpu_has(F_SSE41)) k_movntdqa_sse41(back, m, n); else memcpy(back, m, n);
    double t2 = t_now_ms();
    CHECKF(memcmp(back, src, n) == 0, "%s: NT store + SFENCE + (NT) load round trip differs", nm);
    /* normal stores and loads too */
    for (size_t i = 0; i < n; i += 4096) m[i] = (uint8_t)(i >> 12);
    sm_sfence();
    int ok = 1; for (size_t i = 0; i < n; i += 4096) ok &= m[i] == (uint8_t)(i >> 12);
    CHECKF(ok, "%s: byte stores not read back", nm);
    t_note("%s: 1 MB NT store %.2f ms, read back %.2f ms", nm, t1 - t0, t2 - t1);
    VirtualFree(m, 0, MEM_RELEASE); sm_vfree(src); sm_vfree(back);
}
TEST(wc_memory_roundtrip_note) { special_mem_case(PAGE_WRITECOMBINE, "PAGE_WRITECOMBINE"); }
TEST(nocache_memory_roundtrip_note) { special_mem_case(PAGE_NOCACHE, "PAGE_NOCACHE"); }
