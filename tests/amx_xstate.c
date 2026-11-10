/* amx_xstate - Intel AMX tile state as an OS-managed, opt-in (XFD) extended state component: actually
 * executes LDTILECFG / TILELOADD / TILESTORED / TDP* / TILERELEASE and checks what Windows does with
 * TILECFG (XSTATE component 17, 64 bytes) and TILEDATA (component 18, 8192 bytes).
 *
 * x64 only (AMX instructions are #UD outside 64-bit mode, SDM Vol.2 "Intel AMX instruction exception
 * classes"); the x86 build has one case that SKIPs. Every case SKIPs unless the CPU advertises AMX-TILE
 * (CPUID.(7,0):EDX[24]), the OS enabled XCR0[18:17], CPUID leaf 1Dh describes the 8 x 16 x 64-byte
 * palette 1 used here, and the Windows 11 optional-feature APIs (GetThreadEnabledXStateFeatures,
 * EnableProcessOptionalXStateFeatures) are exported. Context cases additionally need GetEnabledXStateFeatures
 * to report 0x60000 and InitializeContext2 / LocateXStateFeature / Get/SetXStateFeaturesMask / CopyContext.
 *
 * Contracts:
 *  - SDM Vol.1 ch.3/13 + Vol.2 (LDTILECFG, STTILECFG, TILELOADD, TILESTORED, TILEZERO, TILERELEASE,
 *    TDPBSSD/TDPBSUD/TDPBUSD/TDPBUUD, TDPBF16PS):
 *      palette 1 = 8 tiles x 16 rows x 64 bytes (CPUID.(1Dh,1)); 64-byte TILECFG {palette, start_row,
 *      reserved[14], colsb[16] (u16), rows[16] (u8)}; LDTILECFG zeroes all tile data; STTILECFG stores 64
 *      zero bytes while TILECFG is in its INIT state (palette 0); TILESTORED writes only rows x colsb;
 *      TILERELEASE returns TILECFG and TILEDATA to INIT, so XINUSE[18:17] (XGETBV(1)) read 0.
 *      TDPB[SU][SU]D: C[m][n] += sum_k sum_i A.b[m][4k+i] * B.b[k][4n+i] with the byte signedness of the
 *      mnemonic (first letter = tmm2/A, second = tmm3/B), 32-bit wrap-around. TDPBF16PS: C[m][n] +=
 *      A[m][2k]*B[k][2n] + A[m][2k+1]*B[k][2n+1] in fp32 (inputs here are small integers, so every partial
 *      result is exact and the comparison is bit-exact regardless of fusing/ordering).
 *      TILEDATA in the XSAVE image: tile i at byte 1024*i, row r at +64*r (CPUID.(0Dh,18) size 8192).
 *  - XFD (SDM Vol.1 13.14): an AMX instruction touching TILEDATA while IA32_XFD[18]=1 raises #NM.
 *    Windows 11 arms XFD for TILEDATA until the process calls EnableProcessOptionalXStateFeatures and
 *    reports the unpermitted use as STATUS_ILLEGAL_INSTRUCTION (0xC000001D). The permission is per process:
 *    threads created later report it in GetThreadEnabledXStateFeatures; another process does not get it.
 *  - Windows 11 XSTATE context: XSTATE_MASK_AMX_TILE_DATA is in XSTATE_MASK_LARGE_FEATURES (winnt.h), so
 *    AMX state is in a CONTEXT only when it is requested through InitializeContext2(..., mask) and
 *    SetXStateFeaturesMask; LocateXStateFeature(17/18) then return 64/8192 bytes. GetThreadContext /
 *    SetThreadContext of a suspended thread report / install its tiles. These Win11 APIs are NOT in the
 *    Windows 10 19041 reference corpus (checked: kernelbase/kernel32 have no EnableProcessOptionalXStateFeatures
 *    or GetThreadEnabledXStateFeatures), so their results beyond the MS docs are recorded, not hard-checked:
 *    InitializeContext default length, CopyContext into a destination without AMX space, the feature mask of a
 *    thread that existed before the enable call.
 *    From the 19041 decomp (shared with the non-AMX paths): InitializeContext2(NULL or short buffer) ->
 *    FALSE + ERROR_INSUFFICIENT_BUFFER + required length; LocateXStateFeature on a compacted context whose
 *    XCOMP_BV lacks the feature -> NULL; RtlCopyContext dest XSTATE_BV = src & enabled & ~3.
 *  - Preservation: an involuntary context switch (16 threads pinned to one CPU), SuspendThread/ResumeThread,
 *    Get/SetThreadContext of another thread and exception continuation (VEH -> EXCEPTION_CONTINUE_EXECUTION)
 *    resume the thread with its tiles intact: hard checks. Across a CALL (Sleep, SwitchToThread, waits,
 *    SwitchToFiber) tile registers are volatile under the ABI, so preservation there is INFO; what is
 *    hard-checked is that a thread never observes another thread's tile data.
 *  Tile registers are never used by compiler-generated code (nothing here is built with -mamx-*), so tile
 *  state survives between separate asm statements; only the explicit operations above can disturb it.
 *
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> NOTE/INFO.
 * Deliberately faulting operations (the XFD probe) run in CHILD() processes. ISA rule: every function that
 * contains an AMX instruction has "amx" in its name (tools/check_isa.py). */
#include "t.h"

#ifndef T_X64
TEST(amx_requires_x64) { SKIP("AMX instructions exist only in 64-bit mode (this is the x86 build)"); }
#else

#define AMX_STATUS_ILLEGAL_INSTRUCTION 0xC000001Du
#define AMX_CFG_BIT   (1ull << 17)
#define AMX_DATA_BIT  (1ull << 18)
#define AMX_BITS      (AMX_CFG_BIT | AMX_DATA_BIT)
#define AMX_CTX_XSTATE 0x00100040u          /* CONTEXT_AMD64 | 0x40 */
#define AMX_CTX_GET    ((DWORD)CONTEXT_ALL | AMX_CTX_XSTATE)
#define AMX_NTHREADS   16

typedef BOOL    (WINAPI *pfnInitializeContext)(PVOID, DWORD, PCONTEXT *, PDWORD);
typedef BOOL    (WINAPI *pfnInitializeContext2)(PVOID, DWORD, PCONTEXT *, PDWORD, ULONG64);
typedef DWORD64 (WINAPI *pfnGetEnabledXStateFeatures)(void);
typedef BOOL    (WINAPI *pfnGetXStateFeaturesMask)(PCONTEXT, PDWORD64);
typedef BOOL    (WINAPI *pfnSetXStateFeaturesMask)(PCONTEXT, DWORD64);
typedef PVOID   (WINAPI *pfnLocateXStateFeature)(PCONTEXT, DWORD, PDWORD);
typedef BOOL    (WINAPI *pfnCopyContext)(PCONTEXT, DWORD, PCONTEXT);
typedef DWORD64 (WINAPI *pfnGetThreadEnabledXStateFeatures)(void);
typedef BOOL    (WINAPI *pfnEnableProcessOptionalXStateFeatures)(DWORD64);

static struct {
    int done;
    pfnInitializeContext InitializeContext;
    pfnInitializeContext2 InitializeContext2;
    pfnGetEnabledXStateFeatures GetEnabledXStateFeatures;
    pfnGetXStateFeaturesMask GetXStateFeaturesMask;
    pfnSetXStateFeaturesMask SetXStateFeaturesMask;
    pfnLocateXStateFeature LocateXStateFeature;
    pfnCopyContext CopyContext;
    pfnGetThreadEnabledXStateFeatures GetThreadEnabledXStateFeatures;
    pfnEnableProcessOptionalXStateFeatures EnableProcessOptionalXStateFeatures;
} A;

static void amx_api(void) {
    if (A.done) return;
#define K32(n) A.n = (pfn##n)t_proc("kernel32.dll", #n)
    K32(InitializeContext); K32(InitializeContext2); K32(GetEnabledXStateFeatures); K32(GetXStateFeaturesMask);
    K32(SetXStateFeaturesMask); K32(LocateXStateFeature); K32(CopyContext); K32(GetThreadEnabledXStateFeatures);
    K32(EnableProcessOptionalXStateFeatures);
#undef K32
    A.done = 1;
}

/* ======================= tile images ===================================== */
typedef struct amx_cfg { T_ALIGN(64) uint8_t b[64]; } amx_cfg;
typedef struct amx_tiles { T_ALIGN(64) uint8_t t[8][16][64]; } amx_tiles;   /* = XSAVE TILEDATA layout */

static void cfg_set(amx_cfg *c, int tile, int rows, int colsb) {
    c->b[0] = 1;
    c->b[16 + 2 * tile] = (uint8_t)colsb; c->b[17 + 2 * tile] = (uint8_t)(colsb >> 8);
    c->b[48 + tile] = (uint8_t)rows;
}
static void cfg_full(amx_cfg *c) {
    memset(c, 0, sizeof *c);
    for (int i = 0; i < 8; i++) cfg_set(c, i, 16, 64);
}
static void tiles_pattern(amx_tiles *t, uint32_t tag) {
    for (int i = 0; i < 8; i++) t_pattern_reg(t->t[i], 1024, tag, (uint32_t)i);
}
static int all_zero(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) if (b[i]) return 0;
    return 1;
}

/* ======================= AMX instruction helpers (names contain "amx") === */
#define AMX_S 64LL   /* row stride */
static T_NOINLINE void amx_ldtilecfg(const amx_cfg *c) { __asm__ volatile("ldtilecfg (%0)" :: "r"(c) : "memory"); }
static T_NOINLINE void amx_sttilecfg(amx_cfg *c) { __asm__ volatile("sttilecfg (%0)" :: "r"(c) : "memory"); }
static T_NOINLINE void amx_tilerelease(void) { __asm__ volatile("tilerelease" ::: "memory"); }
static T_NOINLINE void amx_load8(const amx_tiles *t) {
    __asm__ volatile(
        "tileloadd 0(%[p],%[s],1), %%tmm0\n\t"    "tileloadd 1024(%[p],%[s],1), %%tmm1\n\t"
        "tileloadd 2048(%[p],%[s],1), %%tmm2\n\t" "tileloadd 3072(%[p],%[s],1), %%tmm3\n\t"
        "tileloadd 4096(%[p],%[s],1), %%tmm4\n\t" "tileloadd 5120(%[p],%[s],1), %%tmm5\n\t"
        "tileloadd 6144(%[p],%[s],1), %%tmm6\n\t" "tileloadd 7168(%[p],%[s],1), %%tmm7\n\t"
        :: [p]"r"(t), [s]"r"(AMX_S) : "memory");
}
static T_NOINLINE void amx_store8(amx_tiles *t) {
    __asm__ volatile(
        "tilestored %%tmm0, 0(%[p],%[s],1)\n\t"    "tilestored %%tmm1, 1024(%[p],%[s],1)\n\t"
        "tilestored %%tmm2, 2048(%[p],%[s],1)\n\t" "tilestored %%tmm3, 3072(%[p],%[s],1)\n\t"
        "tilestored %%tmm4, 4096(%[p],%[s],1)\n\t" "tilestored %%tmm5, 5120(%[p],%[s],1)\n\t"
        "tilestored %%tmm6, 6144(%[p],%[s],1)\n\t" "tilestored %%tmm7, 7168(%[p],%[s],1)\n\t"
        :: [p]"r"(t), [s]"r"(AMX_S) : "memory");
}
static T_NOINLINE void amx_load_tile0(const void *p) { __asm__ volatile("tileloadd (%[p],%[s],1), %%tmm0" :: [p]"r"(p), [s]"r"(AMX_S) : "memory"); }
static T_NOINLINE void amx_store_tile0(void *p) { __asm__ volatile("tilestored %%tmm0, (%[p],%[s],1)" :: [p]"r"(p), [s]"r"(AMX_S) : "memory"); }
static T_NOINLINE void amx_zero_tile5(void) { __asm__ volatile("tilezero %%tmm5" ::: "memory"); }
/* TILERELEASE and XGETBV(1) back to back (needs F_XGETBV1). */
static T_NOINLINE uint64_t amx_release_xinuse(void) {
    uint32_t lo, hi;
    __asm__ volatile("tilerelease\n\txgetbv" : "=a"(lo), "=d"(hi) : "c"(1) : "memory");
    return ((uint64_t)hi << 32) | lo;
}
enum { DP_SSD, DP_SUD, DP_USD, DP_UUD, DP_BF16 };
/* tmm0 = C (dst), tmm1 = A, tmm2 = B. */
static T_NOINLINE void amx_tdp_run(int op, const amx_cfg *cfg, const void *c_in, const void *a, const void *b, void *c_out) {
    amx_ldtilecfg(cfg);
    __asm__ volatile("tileloadd (%[c],%[s],1), %%tmm0\n\ttileloadd (%[a],%[s],1), %%tmm1\n\ttileloadd (%[b],%[s],1), %%tmm2"
                     :: [c]"r"(c_in), [a]"r"(a), [b]"r"(b), [s]"r"(AMX_S) : "memory");
    switch (op) {
    case DP_SSD:  __asm__ volatile("tdpbssd %%tmm2, %%tmm1, %%tmm0" ::: "memory"); break;
    case DP_SUD:  __asm__ volatile("tdpbsud %%tmm2, %%tmm1, %%tmm0" ::: "memory"); break;
    case DP_USD:  __asm__ volatile("tdpbusd %%tmm2, %%tmm1, %%tmm0" ::: "memory"); break;
    case DP_UUD:  __asm__ volatile("tdpbuud %%tmm2, %%tmm1, %%tmm0" ::: "memory"); break;
    default:      __asm__ volatile("tdpbf16ps %%tmm2, %%tmm1, %%tmm0" ::: "memory"); break;
    }
    __asm__ volatile("tilestored %%tmm0, (%[c],%[s],1)" :: [c]"r"(c_out), [s]"r"(AMX_S) : "memory");
    amx_tilerelease();
}

/* ======================= gating ============================================ */
static amx_cfg g_full;

static void amx_need_cpu(void) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_AMX_TILE)) SKIP("AMX-TILE not advertised by CPUID");
    if (!cpu_has(F_AMX_TILE)) SKIP("AMX-TILE advertised but XCR0[18:17]=%d%d (OS has not enabled AMX state)",
                                   (int)((c->xcr0 >> 18) & 1), (int)((c->xcr0 >> 17) & 1));
    if (c->max_basic < 0x1D) SKIP("CPUID leaf 1Dh unreachable (max basic 0x%x)", c->max_basic);
    t_cpuid_regs r0, r1;
    cpuid_ex(0x1D, 0, &r0); cpuid_ex(0x1D, 1, &r1);
    if (r0.eax < 1 || (r1.eax & 0xFFFF) != 8192 || (r1.eax >> 16) != 1024 || (r1.ebx & 0xFFFF) != 64 ||
        (r1.ebx >> 16) < 8 || (r1.ecx & 0xFFFF) < 16)
        SKIP("palette 1 geometry differs from 8x16x64 (1Dh.1 eax=0x%x ebx=0x%x ecx=0x%x): not modelled here",
             r1.eax, r1.ebx, r1.ecx);
    if (c->comp[17].size != 64 || c->comp[18].size != 8192)
        SKIP("CPUID.0Dh sizes TILECFG=%u TILEDATA=%u differ from 64/8192", c->comp[17].size, c->comp[18].size);
    cfg_full(&g_full);
}
static void amx_need_api(void) {
    amx_need_cpu();
    amx_api();
    if (!A.GetThreadEnabledXStateFeatures || !A.EnableProcessOptionalXStateFeatures)
        SKIP("GetThreadEnabledXStateFeatures/EnableProcessOptionalXStateFeatures not exported (Windows 11+ API)");
}
static DWORD64 amx_os_enabled(void) { return A.GetEnabledXStateFeatures ? A.GetEnabledXStateFeatures() : 0; }

static int g_enabled;   /* 1 = this process enabled TILEDATA */
static void amx_enable(void) {
    amx_need_api();
    if (g_enabled) return;
    DWORD64 en = amx_os_enabled();
    if ((en & AMX_BITS) != AMX_BITS) {
        t_note("XCR0 enables AMX but GetEnabledXStateFeatures()=0x%llx lacks 0x60000", (unsigned long long)en);
        SKIP("OS does not offer AMX as an XSTATE feature (GetEnabledXStateFeatures=0x%llx)", (unsigned long long)en);
    }
    SetLastError(0);
    BOOL ok = A.EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA);
    DWORD err = GetLastError();
    REQUIREF(ok, "EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA) failed, error %lu", err);
    DWORD64 th = A.GetThreadEnabledXStateFeatures();
    CHECKF(th & AMX_DATA_BIT, "enabled TILEDATA but GetThreadEnabledXStateFeatures()=0x%llx lacks bit 18", (unsigned long long)th);
    g_enabled = 1;
}
static void amx_need_ctx(void) {
    amx_enable();
    if (!A.InitializeContext2 || !A.LocateXStateFeature || !A.GetXStateFeaturesMask || !A.SetXStateFeaturesMask)
        SKIP("InitializeContext2/LocateXStateFeature/Get/SetXStateFeaturesMask not exported");
}

/* ======================= XSTATE context helpers =========================== */
typedef struct actx { BYTE *buf; void *base; DWORD len; CONTEXT *ctx; } actx;

static DWORD actx_len(DWORD flags, DWORD64 mask) {
    DWORD len = 0; PCONTEXT c = NULL;
    SetLastError(0xDEADBEEFu);
    BOOL ok = A.InitializeContext2(NULL, flags, &c, &len, mask);
    if (ok || GetLastError() != ERROR_INSUFFICIENT_BUFFER) return 0;
    return len;
}
/* Buffer of exactly the required length ending at a PAGE_NOACCESS page; features requested = req. */
static int actx_new(actx *x, DWORD flags, DWORD64 mask, DWORD64 req) {
    memset(x, 0, sizeof *x);
    DWORD len = actx_len(flags, mask);
    if (!len) return 0;
    x->buf = (BYTE *)t_guarded_alloc(len, &x->base);
    if (!x->buf) return 0;
    memset(x->buf, 0xA5, len);
    x->len = len;
    DWORD l2 = len;
    if (!A.InitializeContext2(x->buf, flags, &x->ctx, &l2, mask) || !x->ctx) { t_guarded_free(x->base); memset(x, 0, sizeof *x); return 0; }
    if ((flags & AMX_CTX_XSTATE) == AMX_CTX_XSTATE && !A.SetXStateFeaturesMask(x->ctx, req)) { t_guarded_free(x->base); memset(x, 0, sizeof *x); return 0; }
    return 1;
}
static void actx_free(actx *x) { if (x->base) t_guarded_free(x->base); memset(x, 0, sizeof *x); }
static int actx_inside(const actx *x, const void *p, size_t n) { return (const BYTE *)p >= x->buf && (const BYTE *)p + n <= x->buf + x->len; }
static DWORD64 actx_mask(const actx *x) { DWORD64 m = 0; if (!A.GetXStateFeaturesMask(x->ctx, &m)) return ~0ull; return m; }
static void *actx_loc(const actx *x, DWORD id, DWORD *len) { DWORD l = 0; void *p = A.LocateXStateFeature(x->ctx, id, &l); if (len) *len = l; return p; }
#define ACTX_AMX(x) REQUIREF(actx_new((x), AMX_CTX_GET, AMX_BITS, AMX_BITS), \
    "InitializeContext2(CONTEXT_ALL|CONTEXT_XSTATE, mask 0x60000) + SetXStateFeaturesMask failed, error %lu", GetLastError())

/* ======================= worker threads ===================================== */
typedef struct amx_w {
    T_ALIGN(64) amx_tiles in, out;
    amx_cfg cfg_out;
    uint32_t tag;
    int cpu, pin_ok, release_first;
    HANDLE h;
    volatile long ready, stop;
    unsigned long long spins;
} amx_w;

/* Load tiles, then a GPR-only spin (no calls) until stop: preemption, SuspendThread and Get/SetThreadContext
 * from other threads are the only things that can touch the tiles. */
static DWORD WINAPI amx_spin_worker(void *p) {
    amx_w *w = (amx_w *)p;
    if (w->cpu >= 0) w->pin_ok = t_pin_thread(w->cpu) != 0;
    amx_ldtilecfg(&g_full);
    amx_load8(&w->in);
    if (w->release_first) amx_tilerelease();
    InterlockedExchange(&w->ready, 1);
    unsigned long long n = 0;
    while (!w->stop && n < 4000000000ull) { __asm__ volatile("pause"); n++; }
    w->spins = n;
    amx_sttilecfg(&w->cfg_out);
    if (!w->release_first) amx_store8(&w->out);
    amx_tilerelease();
    return 0;
}

static amx_w *workers_alloc(int n) {
    return (amx_w *)VirtualAlloc(NULL, sizeof(amx_w) * (size_t)n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
static void workers_free(amx_w *w) { if (w) VirtualFree(w, 0, MEM_RELEASE); }
/* Start n spinning workers; returns how many started. */
static int workers_start(amx_w *w, int n, uint32_t tag0, int cpu) {
    int i;
    for (i = 0; i < n; i++) {
        w[i].tag = tag0 + (uint32_t)i; w[i].cpu = cpu; w[i].ready = 0; w[i].stop = 0; w[i].release_first = 0;
        tiles_pattern(&w[i].in, w[i].tag);
        memset(&w[i].out, 0xEE, sizeof w[i].out);
        w[i].h = CreateThread(NULL, 64 * 1024, amx_spin_worker, &w[i], 0, NULL);
        if (!w[i].h) break;
    }
    return i;
}
static int workers_wait_ready(amx_w *w, int n, DWORD ms) {
    double t0 = t_now_ms();
    for (;;) {
        int r = 0;
        for (int i = 0; i < n; i++) r += w[i].ready != 0;
        if (r == n) return n;
        if (t_now_ms() - t0 > ms) return r;
        Sleep(1);
    }
}
static int workers_join(amx_w *w, int n) {
    int ok = 1;
    for (int i = 0; i < n; i++) InterlockedExchange(&w[i].stop, 1);
    for (int i = 0; i < n; i++) {
        if (WaitForSingleObject(w[i].h, 20000) != WAIT_OBJECT_0) ok = 0;
        CloseHandle(w[i].h); w[i].h = NULL;
    }
    return ok;
}
static void workers_verify(amx_w *w, int n, const char *what) {
    char lbl[96];
    for (int i = 0; i < n; i++) {
        snprintf(lbl, sizeof lbl, "thread %d TILECFG %s", i, what);
        CHECK_REG(lbl, w[i].cfg_out.b, g_full.b, 64);
        for (int t = 0; t < 8; t++) {
            snprintf(lbl, sizeof lbl, "thread %d tmm%d %s", i, t, what);
            if (!CHECK_REG(lbl, w[i].out.t[t], w[i].in.t[t], 1024)) break;
        }
    }
}
static int first_cpu(void) {
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) || !pm) return -1;
    for (int i = 0; i < (int)(sizeof pm * 8); i++) if (pm & ((DWORD_PTR)1 << i)) return i;
    return -1;
}

/* ======================= A. XFD / optional-feature enabling (children) ==== */
static void out_flush(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    fflush(stdout);
}
/* Executes the first tile accesses: LDTILECFG, TILELOADD x8, TILESTORED x8. Prints STEP= before each. */
static int amx_first_use_steps(const char *who) {
    static amx_tiles in, out;
    cfg_full(&g_full);
    tiles_pattern(&in, 0x51);
    out_flush("STEP=%s_ldtilecfg\n", who);
    amx_ldtilecfg(&g_full);
    out_flush("STEP=%s_tileloadd\n", who);
    amx_load8(&in);
    out_flush("STEP=%s_tilestored\n", who);
    amx_store8(&out);
    amx_tilerelease();
    int ok = memcmp(&in, &out, sizeof in) == 0;
    out_flush("%s_OK=%d\n", who, ok);
    return ok;
}

CHILD(amx_xfd) {
    int enable = argc > 1 && !strcmp(argv[1], "enable");
    amx_api();
    if (!A.GetThreadEnabledXStateFeatures || !A.EnableProcessOptionalXStateFeatures) { out_flush("RESULT=noapi\n"); return 0; }
    out_flush("BEFORE=0x%llx\n", (unsigned long long)A.GetThreadEnabledXStateFeatures());
    if (enable) {
        SetLastError(0);
        BOOL ok = A.EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA);
        out_flush("ENABLE_RET=%d\nENABLE_ERR=%lu\n", ok, GetLastError());
        out_flush("AFTER=0x%llx\n", (unsigned long long)A.GetThreadEnabledXStateFeatures());
    }
    amx_first_use_steps("main");
    out_flush("STEP=done\n");
    return 0;
}

static HANDLE g_go;
static DWORD WINAPI inherit_thread(void *p) {
    const char *who = (const char *)p;
    WaitForSingleObject(g_go, 10000);
    out_flush("%s_FEAT=0x%llx\n", who, (unsigned long long)A.GetThreadEnabledXStateFeatures());
    amx_first_use_steps(who);
    return 0;
}
/* Thread "pre" exists before the enable call, "post" is created after it; both then use AMX. */
CHILD(amx_inherit) {
    (void)argc; (void)argv;
    amx_api();
    if (!A.GetThreadEnabledXStateFeatures || !A.EnableProcessOptionalXStateFeatures) { out_flush("RESULT=noapi\n"); return 0; }
    g_go = CreateEventA(NULL, TRUE, FALSE, NULL);
    HANDLE pre = CreateThread(NULL, 0, inherit_thread, (void *)"pre", 0, NULL);
    if (!g_go || !pre) { out_flush("RESULT=nothread\n"); return 0; }
    Sleep(20);
    out_flush("MAIN_BEFORE=0x%llx\n", (unsigned long long)A.GetThreadEnabledXStateFeatures());
    SetLastError(0);
    BOOL ok = A.EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA);
    out_flush("ENABLE_RET=%d\nENABLE_ERR=%lu\n", ok, GetLastError());
    out_flush("MAIN_AFTER=0x%llx\n", (unsigned long long)A.GetThreadEnabledXStateFeatures());
    HANDLE post = CreateThread(NULL, 0, inherit_thread, (void *)"post", CREATE_SUSPENDED, NULL);
    if (!post) { out_flush("RESULT=nothread\n"); return 0; }
    SetEvent(g_go);
    ResumeThread(post);
    WaitForSingleObject(post, 10000);     /* post first, then pre: output never interleaves */
    WaitForSingleObject(pre, 10000);
    amx_first_use_steps("main");
    out_flush("STEP=done\n");
    return 0;
}

static unsigned long long find_hex(const t_child_result *r, const char *key, int *found) {
    char v[64];
    *found = t_child_find(r, key, v, sizeof v);
    return *found ? strtoull(v, NULL, 16) : 0;
}
static void last_step(const t_child_result *r, char *out, size_t cap) {
    out[0] = 0;
    const char *p = r->out, *hit = NULL;
    while ((p = strstr(p, "STEP=")) != NULL) { if (p == r->out || p[-1] == '\n') hit = p + 5; p += 5; }
    if (!hit) return;
    size_t i = 0;
    while (hit[i] && hit[i] != '\n' && hit[i] != '\r' && i + 1 < cap) { out[i] = hit[i]; i++; }
    out[i] = 0;
}

/* Run the XFD probe child without enabling; shared by the fault case and the per-process case. */
static void xfd_probe_expect_fault(const char *ctx) {
    static t_child_result r;
    REQUIREF(t_child_run("amx_xfd", NULL, 15000, &r), "spawn failed: error %lu", r.spawn_error);
    REQUIREF(!r.timed_out, "XFD probe child timed out");
    int f;
    unsigned long long before = find_hex(&r, "BEFORE=", &f);
    REQUIREF(f, "child printed no BEFORE line; output:\n%s", r.out);
    t_info("child_thread_features_before", "0x%llx", before);
    char step[64]; last_step(&r, step, sizeof step);
    t_info("child_last_step", "%s", step);
    if (before & AMX_DATA_BIT) {
        t_note("%s: a fresh process already reports TILEDATA enabled (0x%llx): XFD is not armed", ctx, before);
        CHECKF(r.exc_code == 0 && strcmp(step, "done") == 0, "TILEDATA reported enabled but first use died: exc 0x%lx at %s",
               (unsigned long)r.exc_code, step);
        SKIP("TILEDATA already enabled for a fresh process: no XFD fault to test");
    }
    CHECKF(r.exc_code == AMX_STATUS_ILLEGAL_INSTRUCTION,
           "%s: first tile access without EnableProcessOptionalXStateFeatures: exc 0x%lx exit 0x%lx (want STATUS_ILLEGAL_INSTRUCTION), last step %s",
           ctx, (unsigned long)r.exc_code, (unsigned long)r.exit_code, step);
    CHECKF(!strcmp(step, "main_ldtilecfg") || !strcmp(step, "main_tileloadd"),
           "the XFD fault must hit the first TILECFG/TILEDATA access, last step was '%s'", step);
}

TEST(amx_xfd_first_use_faults_without_enable) {
    amx_need_api();
    xfd_probe_expect_fault("child");
}

TEST(amx_enable_then_first_use_works) {
    amx_need_api();
    DWORD64 en = amx_os_enabled();
    if ((en & AMX_BITS) != AMX_BITS) SKIP("GetEnabledXStateFeatures()=0x%llx lacks AMX", (unsigned long long)en);
    static t_child_result r;
    REQUIREF(t_child_run("amx_xfd", "enable", 15000, &r), "spawn failed: error %lu", r.spawn_error);
    REQUIRE(!r.timed_out);
    char v[64]; int f;
    CHECKF(t_child_find(&r, "ENABLE_RET=", v, sizeof v) && !strcmp(v, "1"), "EnableProcessOptionalXStateFeatures(TILEDATA) returned %s; output:\n%s", v, r.out);
    unsigned long long before = find_hex(&r, "BEFORE=", &f), after = find_hex(&r, "AFTER=", &f);
    t_info("thread_features_before", "0x%llx", before); t_info("thread_features_after", "0x%llx", after);
    CHECKF(f && (after & AMX_DATA_BIT), "GetThreadEnabledXStateFeatures after enabling = 0x%llx lacks bit 18", after);
    CHECKF((after & before) == before, "enabling removed features: before 0x%llx after 0x%llx", before, after);
    CHECKF((after & ~en) == 0, "thread features 0x%llx exceed GetEnabledXStateFeatures 0x%llx", after, (unsigned long long)en);
    if (!(after & AMX_CFG_BIT)) t_note("thread features after enabling lack TILECFG (bit 17): 0x%llx", after);
    char step[64]; last_step(&r, step, sizeof step);
    CHECKF(r.exc_code == 0 && !strcmp(step, "done"), "first AMX use after enabling died: exc 0x%lx at step %s", (unsigned long)r.exc_code, step);
    CHECKF(t_child_find(&r, "main_OK=", v, sizeof v) && !strcmp(v, "1"), "tile round trip after enabling wrong (%s)", v);
}

TEST(amx_enable_inherited_by_threads) {
    amx_need_api();
    DWORD64 en = amx_os_enabled();
    if ((en & AMX_BITS) != AMX_BITS) SKIP("GetEnabledXStateFeatures()=0x%llx lacks AMX", (unsigned long long)en);
    static t_child_result r;
    REQUIREF(t_child_run("amx_inherit", NULL, 20000, &r), "spawn failed: error %lu", r.spawn_error);
    REQUIRE(!r.timed_out);
    char v[64]; int f;
    REQUIREF(!t_child_find(&r, "RESULT=", v, sizeof v), "child could not run: %s", v);
    CHECKF(t_child_find(&r, "ENABLE_RET=", v, sizeof v) && !strcmp(v, "1"), "EnableProcessOptionalXStateFeatures returned %s", v);
    unsigned long long mb = find_hex(&r, "MAIN_BEFORE=", &f), ma = find_hex(&r, "MAIN_AFTER=", &f);
    int fpost; unsigned long long post = find_hex(&r, "post_FEAT=", &fpost);
    int fpre;  unsigned long long pre = find_hex(&r, "pre_FEAT=", &fpre);
    t_info("main_before", "0x%llx", mb); t_info("main_after", "0x%llx", ma);
    t_info("thread_created_after", "0x%llx", post); t_info("thread_created_before", "0x%llx", pre);
    CHECKF(ma & AMX_DATA_BIT, "enabling thread reports 0x%llx (no TILEDATA)", ma);
    CHECKF(fpost && (post & AMX_DATA_BIT), "thread created after enabling reports 0x%llx (no TILEDATA)", post);
    if (fpre && !(pre & AMX_DATA_BIT)) t_note("thread that existed before enabling reports 0x%llx (TILEDATA not listed)", pre);
    char step[64]; last_step(&r, step, sizeof step);
    CHECKF(r.exc_code == 0 && !strcmp(step, "done"), "a thread died using AMX after the process enabled it: exc 0x%lx at step %s\n%s",
           (unsigned long)r.exc_code, step, r.out);
    static const char *who[] = { "post_OK=", "pre_OK=", "main_OK=" };
    for (int i = 0; i < 3; i++)
        CHECKF(t_child_find(&r, who[i], v, sizeof v) && !strcmp(v, "1"), "%s%s (tile round trip)", who[i], v);
}

TEST(amx_enable_is_per_process) {
    amx_enable();
    DWORD64 th = A.GetThreadEnabledXStateFeatures();
    t_info("parent_thread_features", "0x%llx", (unsigned long long)th);
    CHECKF(th & AMX_DATA_BIT, "parent enabled TILEDATA but reports 0x%llx", (unsigned long long)th);
    xfd_probe_expect_fault("child of an AMX-enabled parent");   /* permission is not inherited by processes */
}

TEST(amx_enable_api_parameters) {
    amx_enable();
    DWORD64 en = amx_os_enabled();
    SetLastError(0);
    BOOL again = A.EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA);
    t_info("enable_again", "%d err=%lu", again, GetLastError());   /* return value not documented: recorded */
    SetLastError(0);
    BOOL z = A.EnableProcessOptionalXStateFeatures(0);
    t_info("enable_zero", "%d err=%lu", z, GetLastError());
    uint64_t bogus = (1ull << 62) & ~en;
    if (bogus) {
        SetLastError(0);
        BOOL b = A.EnableProcessOptionalXStateFeatures(bogus);
        t_info("enable_unsupported_bit62", "%d err=%lu", b, GetLastError());
    }
    DWORD64 th = A.GetThreadEnabledXStateFeatures();
    CHECKF(th & AMX_DATA_BIT, "AMX permission lost after further enable calls: 0x%llx", (unsigned long long)th);
    CHECKF((th & ~en) == 0, "thread features 0x%llx exceed GetEnabledXStateFeatures 0x%llx", (unsigned long long)th, (unsigned long long)en);
}

/* ======================= B. known answers ================================== */
typedef struct dp_buf { T_ALIGN(64) uint8_t b[16][64]; } dp_buf;

static void dp_model_int(int op, int M, int N, int K, const dp_buf *a, const dp_buf *b, dp_buf *c) {
    int as = op == DP_SSD || op == DP_SUD, bs = op == DP_SSD || op == DP_USD;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            uint32_t acc; memcpy(&acc, &c->b[m][4 * n], 4);
            for (int k = 0; k < K; k++)
                for (int i = 0; i < 4; i++) {
                    int32_t x = as ? (int8_t)a->b[m][4 * k + i] : a->b[m][4 * k + i];
                    int32_t y = bs ? (int8_t)b->b[k][4 * n + i] : b->b[k][4 * n + i];
                    acc += (uint32_t)(x * y);
                }
            memcpy(&c->b[m][4 * n], &acc, 4);
        }
}
static uint16_t bf16_of_int(int v) { float f = (float)v; uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float f_of_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static void dp_model_bf16(int M, int N, int K, const dp_buf *a, const dp_buf *b, dp_buf *c) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc; memcpy(&acc, &c->b[m][4 * n], 4);
            double s = acc;
            for (int k = 0; k < K; k++)
                for (int i = 0; i < 2; i++) {
                    uint16_t x, y;
                    memcpy(&x, &a->b[m][4 * k + 2 * i], 2); memcpy(&y, &b->b[k][4 * n + 2 * i], 2);
                    s += (double)f_of_bf16(x) * (double)f_of_bf16(y);
                }
            float r = (float)s;   /* exact: integer data, |s| < 2^24 */
            memcpy(&c->b[m][4 * n], &r, 4);
        }
}

static void dp_case(int op, const char *name, t_feature feat) {
    amx_enable();
    if (!cpu_has(feat)) SKIP("%s not advertised", cpu_feature_name(feat));
    static const struct { int M, N, K; } shapes[] = { { 16, 16, 16 }, { 5, 6, 3 }, { 1, 1, 1 }, { 16, 3, 7 } };
    t_rng r; t_rng_seed(&r, 0xA3A3u + (unsigned)op);
    for (size_t s = 0; s < T_COUNTOF(shapes); s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        amx_cfg cfg; memset(&cfg, 0, sizeof cfg);
        cfg_set(&cfg, 0, M, 4 * N); cfg_set(&cfg, 1, M, 4 * K); cfg_set(&cfg, 2, K, 4 * N);
        static dp_buf a, b, c, want, got;
        if (op == DP_BF16) {
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 32; j++) {
                    uint16_t x = bf16_of_int((int)t_rng_range(&r, 17) - 8), y = bf16_of_int((int)t_rng_range(&r, 17) - 8);
                    memcpy(&a.b[i][2 * j], &x, 2); memcpy(&b.b[i][2 * j], &y, 2);
                }
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++) { float f = (float)((int)t_rng_range(&r, 2001) - 1000); memcpy(&c.b[i][4 * j], &f, 4); }
        } else {
            t_rng_fill(&r, &a, sizeof a); t_rng_fill(&r, &b, sizeof b); t_rng_fill(&r, &c, sizeof c);
            if (s == 0) {   /* extremes in row 0: -128*-128, 255*-128, 255*255 sums and 32-bit wrap */
                memset(a.b[0], 0x80, 64); memset(b.b[0], 0x80, 64); memset(b.b[1], 0xFF, 64);
                memset(c.b[0], 0xFF, 64);
            }
        }
        want = c;
        if (op == DP_BF16) dp_model_bf16(M, N, K, &a, &b, &want); else dp_model_int(op, M, N, K, &a, &b, &want);
        memset(&got, 0xCC, sizeof got);
        amx_tdp_run(op, &cfg, &c, &a, &b, &got);
        /* bytes outside rows x colsb of tmm0 must not be written by TILESTORED */
        for (int m = 0; m < 16; m++)
            for (int x = 0; x < 64; x++) if (m >= M || x >= 4 * N) want.b[m][x] = 0xCC;
        if (memcmp(&got, &want, sizeof got)) {
            t_note("%s M=%d N=%d K=%d: result differs from the C model (rows of tmm0 at a 64-byte stride)", name, M, N, K);
            CHECK_MEM(&got, &want, sizeof got);
        }
    }
}
TEST(amx_ka_tdpbssd)   { dp_case(DP_SSD, "TDPBSSD", F_AMX_INT8); }
TEST(amx_ka_tdpbsud)   { dp_case(DP_SUD, "TDPBSUD", F_AMX_INT8); }
TEST(amx_ka_tdpbusd)   { dp_case(DP_USD, "TDPBUSD", F_AMX_INT8); }
TEST(amx_ka_tdpbuud)   { dp_case(DP_UUD, "TDPBUUD", F_AMX_INT8); }
TEST(amx_ka_tdpbf16ps) { dp_case(DP_BF16, "TDPBF16PS", F_AMX_BF16); }

TEST(amx_ka_tilecfg_and_tiledata) {
    amx_enable();
    static amx_tiles in, out;
    amx_cfg c;
    tiles_pattern(&in, 0x61);
    /* STTILECFG round trip of a full and of a partial configuration */
    amx_ldtilecfg(&g_full); amx_sttilecfg(&c);
    CHECK_REG("STTILECFG after LDTILECFG(full)", c.b, g_full.b, 64);
    amx_cfg part; memset(&part, 0, sizeof part);
    cfg_set(&part, 0, 5, 24); cfg_set(&part, 3, 16, 4); cfg_set(&part, 7, 1, 64);
    amx_ldtilecfg(&part); amx_sttilecfg(&c);
    CHECK_REG("STTILECFG after LDTILECFG(partial)", c.b, part.b, 64);
    /* partial tile: TILELOADD / TILESTORED move only rows x colsb */
    static dp_buf src, dst, want;
    memcpy(&src, in.t[2], sizeof src);
    memset(&dst, 0xCC, sizeof dst);
    amx_load_tile0(&src); amx_store_tile0(&dst);
    want = dst;
    for (int m = 0; m < 5; m++) memcpy(want.b[m], src.b[m], 24);
    CHECK_MEM(&dst, &want, sizeof dst);
    /* full tiles round trip, TILEZERO, LDTILECFG zeroes tile data */
    amx_ldtilecfg(&g_full);
    amx_load8(&in);
    amx_zero_tile5();
    memset(&out, 0xEE, sizeof out);
    amx_store8(&out);
    amx_tiles want8 = in; memset(want8.t[5], 0, 1024);
    CHECK_MEM(&out, &want8, sizeof out);
    amx_ldtilecfg(&g_full);
    memset(&out, 0xEE, sizeof out);
    amx_store8(&out);
    CHECKF(all_zero(&out, sizeof out), "LDTILECFG must zero all tile data");
    /* TILERELEASE: TILECFG back to INIT (STTILECFG stores zeros) */
    amx_tilerelease();
    memset(&c, 0xEE, sizeof c);
    amx_sttilecfg(&c);
    CHECKF(all_zero(&c, 64), "STTILECFG after TILERELEASE must store 64 zero bytes");
}

TEST(amx_tilerelease_clears_xinuse) {
    amx_enable();
    if (!cpu_has(F_XGETBV1)) SKIP("XGETBV with ECX=1 not supported");
    static amx_tiles in;
    tiles_pattern(&in, 0x62);
    amx_ldtilecfg(&g_full);
    amx_load8(&in);
    uint64_t live = t_xgetbv(1);
    t_info("xinuse_tiles_loaded", "0x%llx", (unsigned long long)live);
    CHECKF((live & AMX_BITS) == AMX_BITS, "XINUSE 0x%llx: non-INIT tile state must have XINUSE[18:17]=11b", (unsigned long long)live);
    uint64_t rel = amx_release_xinuse();
    t_info("xinuse_after_tilerelease", "0x%llx", (unsigned long long)rel);
    CHECKF((rel & AMX_BITS) == 0, "XINUSE 0x%llx after TILERELEASE: bits 18:17 must be 0", (unsigned long long)rel);
}

/* ======================= C. preservation ==================================== */
TEST(amx_preempt_pinned_16_threads) {
    amx_enable();
    int cpu0 = first_cpu();
    if (cpu0 < 0) SKIP("no process affinity");
    amx_w *w = workers_alloc(AMX_NTHREADS);
    REQUIRE(w);
    int n = workers_start(w, AMX_NTHREADS, 0x100, cpu0);
    if (n < AMX_NTHREADS) { workers_join(w, n); workers_free(w); SKIP("could only create %d of %d threads", n, AMX_NTHREADS); }
    int ready = workers_wait_ready(w, n, 10000);
    Sleep(200);
    int joined = workers_join(w, n);
    REQUIREF(joined, "workers did not finish");
    int pinned = 0;
    for (int i = 0; i < n; i++) pinned += w[i].pin_ok;
    t_info("ready_while_spinning", "%d/%d", ready, n);
    t_info("pinned", "%d/%d on cpu %d", pinned, n, cpu0);
    if (ready == n && pinned == n) t_note("all %d tile-holding threads spun on one CPU: each was preempted while its tiles were live", n);
    workers_verify(w, n, "after preemption on a shared CPU");
    workers_free(w);
}

TEST(amx_suspend_getcontext_16_threads) {
    amx_enable();
    amx_w *w = workers_alloc(AMX_NTHREADS);
    REQUIRE(w);
    int n = workers_start(w, AMX_NTHREADS, 0x200, -1);
    if (n < AMX_NTHREADS) { workers_join(w, n); workers_free(w); SKIP("could only create %d of %d threads", n, AMX_NTHREADS); }
    int ready = workers_wait_ready(w, n, 10000);
    int have_ctx = A.InitializeContext2 && A.LocateXStateFeature && A.SetXStateFeaturesMask && A.GetXStateFeaturesMask &&
                   (amx_os_enabled() & AMX_BITS) == AMX_BITS;
    actx x; int xok = have_ctx && actx_new(&x, AMX_CTX_GET, AMX_BITS, AMX_BITS);
    if (have_ctx && !xok) t_note("AMX context could not be initialized (error %lu): only CONTEXT_FULL reads done", GetLastError());
    int bad_ctx = 0, ctx_reads = 0;
    for (int round = 0; round < 20; round++)
        for (int i = 0; i < n; i++) {
            if (SuspendThread(w[i].h) == (DWORD)-1) { CHECKF(0, "SuspendThread error %lu", GetLastError()); continue; }
            CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_FULL;
            CHECK(GetThreadContext(w[i].h, &c));
            if (xok && w[i].ready && round % 5 == 0) {
                A.SetXStateFeaturesMask(x.ctx, AMX_BITS);
                if (GetThreadContext(w[i].h, x.ctx)) {
                    ctx_reads++;
                    DWORD l18 = 0; const BYTE *d = (const BYTE *)actx_loc(&x, 18, &l18);
                    if (!d || l18 != 8192 || memcmp(d, &w[i].in, 8192)) bad_ctx++;
                } else bad_ctx++;
            }
            ResumeThread(w[i].h);
        }
    if (xok) {
        t_info("amx_context_reads", "%d", ctx_reads);
        CHECKF(bad_ctx == 0, "%d of %d AMX GetThreadContext reads did not return the suspended thread's own tiles", bad_ctx, ctx_reads);
        actx_free(&x);
    }
    REQUIRE(workers_join(w, n));
    t_info("ready", "%d/%d", ready, n);
    workers_verify(w, n, "after 20 rounds of SuspendThread/GetThreadContext/ResumeThread");
    workers_free(w);
}

/* --- exception continuation on the case thread --- */
static volatile DWORD g_exc_tid;
static volatile long g_exc_hits, g_exc_ctx_amx, g_exc_ctx_xstate;
static LONG CALLBACK amx_veh(EXCEPTION_POINTERS *ep) {
    if (GetCurrentThreadId() != g_exc_tid) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord;
    const uint8_t *ip = (const uint8_t *)c->Rip;
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    int len = 0;
    if (code == STATUS_ILLEGAL_INSTRUCTION && ip[0] == 0x0F && ip[1] == 0x0B) len = 2;                /* ud2 */
    else if (code == STATUS_ACCESS_VIOLATION && ip[0] == 0x8B && ip[1] == 0x00) len = 2;              /* mov (%rax),%eax */
    else if (code == STATUS_INTEGER_DIVIDE_BY_ZERO && ip[0] == 0x48 && ip[1] == 0xF7 && ip[2] == 0xF1) {   /* div %rcx */
        InterlockedIncrement(&g_exc_hits); c->Rcx = 1; return EXCEPTION_CONTINUE_EXECUTION;   /* re-executes the div */
    }
    if (!len) return EXCEPTION_CONTINUE_SEARCH;
    InterlockedIncrement(&g_exc_hits);
    if ((c->ContextFlags & AMX_CTX_XSTATE) == AMX_CTX_XSTATE) {
        InterlockedIncrement(&g_exc_ctx_xstate);
        DWORD l = 0;
        if (A.LocateXStateFeature && A.LocateXStateFeature(c, 18, &l)) InterlockedIncrement(&g_exc_ctx_amx);
    }
    c->Rip += (DWORD64)len;
    return EXCEPTION_CONTINUE_EXECUTION;
}
TEST(amx_exception_continue_preserves_tiles) {
    amx_enable();
    void *gbase = NULL;
    uint8_t *g = (uint8_t *)t_guarded_alloc(64, &gbase);
    REQUIRE(g);
    void *h = AddVectoredExceptionHandler(1, amx_veh);
    REQUIRE(h);
    g_exc_hits = g_exc_ctx_amx = g_exc_ctx_xstate = 0;
    g_exc_tid = GetCurrentThreadId();
    static amx_tiles in, out;
    amx_cfg c;
    int bad = 0;
    for (int it = 0; it < 12 && !bad; it++) {
        tiles_pattern(&in, 0x300u + (uint32_t)it);
        amx_ldtilecfg(&g_full);
        amx_load8(&in);
        switch (it % 3) {
        case 0: __asm__ volatile("ud2" ::: "memory"); break;
        case 1: { void *p = g + 64; __asm__ volatile("movl (%%rax), %%eax" : "+a"(p) :: "memory"); break; }
        default: { unsigned long long a = 7, d = 0, cx = 0; __asm__ volatile("divq %%rcx" : "+a"(a), "+d"(d), "+c"(cx) :: "cc"); break; }
        }
        memset(&out, 0xEE, sizeof out);
        amx_sttilecfg(&c);
        if (memcmp(&c, &g_full, 64)) { CHECK_REG("TILECFG after exception continuation", c.b, g_full.b, 64); bad = 1; break; }
        amx_store8(&out);
        for (int t = 0; t < 8 && !bad; t++) {
            char lbl[80]; snprintf(lbl, sizeof lbl, "tmm%d after %s + EXCEPTION_CONTINUE_EXECUTION", t,
                                   it % 3 == 0 ? "#UD" : it % 3 == 1 ? "access violation" : "#DE");
            if (!CHECK_REG(lbl, out.t[t], in.t[t], 1024)) bad = 1;
        }
    }
    amx_tilerelease();
    g_exc_tid = 0;
    RemoveVectoredExceptionHandler(h);
    t_guarded_free(gbase);
    t_info("exceptions_handled", "%ld", g_exc_hits);
    t_info("exception_ctx_with_xstate", "%ld", g_exc_ctx_xstate);
    t_info("exception_ctx_with_tiledata", "%ld", g_exc_ctx_amx);
    CHECKF(g_exc_hits >= 12, "VEH saw only %ld of the 12 #UD/AV/#DE exceptions", g_exc_hits);
}

/* --- calls: Sleep / SwitchToThread / waits (ABI-volatile: INFO), never foreign data (hard) --- */
typedef struct call_w {
    T_ALIGN(64) amx_tiles in, out;
    int idx, cpu;
    long preserved, cfg_lost, changed, foreign, iters;
    char foreign_msg[160];
} call_w;
static call_w *g_cw;
static HANDLE g_cw_event;
static int foreign_owner(const call_w *all, int n, int self, const uint8_t row[64]) {
    for (int j = 0; j < n; j++) {
        if (j == self) continue;
        for (int t = 0; t < 8; t++)
            for (int r = 0; r < 16; r++) if (!memcmp(all[j].in.t[t][r], row, 64)) return j;
    }
    return -1;
}
static DWORD WINAPI call_worker(void *p) {
    call_w *w = (call_w *)p;
    if (w->cpu >= 0) t_pin_thread(w->cpu);
    amx_cfg c;
    for (int it = 0; it < 40; it++) {
        amx_ldtilecfg(&g_full);
        amx_load8(&w->in);
        switch (it % 4) {
        case 0: Sleep(0); break;
        case 1: SwitchToThread(); break;
        case 2: Sleep(1); break;
        default: WaitForSingleObject(g_cw_event, 1); break;
        }
        w->iters++;
        amx_sttilecfg(&c);
        if (memcmp(&c, &g_full, 64)) { w->cfg_lost++; continue; }
        amx_store8(&w->out);
        if (!memcmp(&w->out, &w->in, sizeof w->in)) { w->preserved++; continue; }
        w->changed++;
        for (int t = 0; t < 8; t++)
            for (int r = 0; r < 16; r++) {
                int o = foreign_owner(g_cw, AMX_NTHREADS, w->idx, w->out.t[t][r]);
                if (o >= 0 && !w->foreign) {
                    snprintf(w->foreign_msg, sizeof w->foreign_msg, "tmm%d row %d holds thread %d's data", t, r, o);
                    w->foreign++;
                }
            }
    }
    amx_tilerelease();
    return 0;
}
TEST(amx_calls_sleep_switch_wait_16_threads) {
    amx_enable();
    int cpu0 = first_cpu();
    g_cw = (call_w *)VirtualAlloc(NULL, sizeof(call_w) * AMX_NTHREADS, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(g_cw);
    g_cw_event = CreateEventA(NULL, TRUE, FALSE, NULL);
    HANDLE h[AMX_NTHREADS]; int n;
    for (n = 0; n < AMX_NTHREADS; n++) {
        g_cw[n].idx = n; g_cw[n].cpu = cpu0;
        tiles_pattern(&g_cw[n].in, 0x400u + (uint32_t)n);
    }
    for (n = 0; n < AMX_NTHREADS; n++) {
        h[n] = CreateThread(NULL, 64 * 1024, call_worker, &g_cw[n], CREATE_SUSPENDED, NULL);
        if (!h[n]) break;
    }
    for (int i = 0; i < n; i++) ResumeThread(h[i]);
    int all = 1;
    for (int i = 0; i < n; i++) { if (WaitForSingleObject(h[i], 30000) != WAIT_OBJECT_0) all = 0; CloseHandle(h[i]); }
    CloseHandle(g_cw_event);
    REQUIREF(all, "call workers did not finish");
    if (n < AMX_NTHREADS) { VirtualFree(g_cw, 0, MEM_RELEASE); SKIP("could only create %d of %d threads", n, AMX_NTHREADS); }
    long pres = 0, lost = 0, chg = 0, it = 0;
    for (int i = 0; i < n; i++) {
        pres += g_cw[i].preserved; lost += g_cw[i].cfg_lost; chg += g_cw[i].changed; it += g_cw[i].iters;
        CHECKF(g_cw[i].foreign == 0, "thread %d observed foreign tile data across a call: %s", i, g_cw[i].foreign_msg);
    }
    t_info("calls", "%ld", it);
    t_info("tiles_preserved", "%ld", pres);
    t_info("tilecfg_reset", "%ld", lost);
    t_info("tiles_changed", "%ld", chg);
    if (pres != it) t_note("tile state not preserved across %ld of %ld Sleep/SwitchToThread/wait calls (volatile across calls under the ABI)", it - pres, it);
    VirtualFree(g_cw, 0, MEM_RELEASE);
}

/* --- fibers --- */
static LPVOID g_fib_main;
static amx_cfg g_fib_seen;
static VOID CALLBACK fib_proc(LPVOID p) {
    (void)p;
    for (;;) {
        amx_sttilecfg(&g_fib_seen);
        SwitchToFiber(g_fib_main);
    }
}
TEST(amx_fiber_switch) {
    amx_enable();
    static const DWORD flags[2] = { 0, FIBER_FLAG_FLOAT_SWITCH };
    for (int f = 0; f < 2; f++) {
        g_fib_main = ConvertThreadToFiberEx(NULL, flags[f]);
        REQUIREF(g_fib_main, "ConvertThreadToFiberEx failed: %lu", GetLastError());
        LPVOID fb = CreateFiberEx(0, 0, flags[f], fib_proc, NULL);
        if (!fb) { ConvertFiberToThread(); SKIP("CreateFiberEx failed: %lu", GetLastError()); }
        static amx_tiles in, out;
        tiles_pattern(&in, 0x500u + (uint32_t)f);
        int preserved = 0, reps = 8;
        for (int r = 0; r < reps; r++) {
            amx_ldtilecfg(&g_full);
            amx_load8(&in);
            memset(&g_fib_seen, 0xEE, sizeof g_fib_seen);
            SwitchToFiber(fb);
            CHECKF(!memcmp(&g_fib_seen, &g_full, 64) || all_zero(&g_fib_seen, 64),
                   "fiber saw a TILECFG that is neither the thread's nor INIT");
            amx_cfg c; amx_sttilecfg(&c);
            if (memcmp(&c, &g_full, 64)) continue;
            amx_store8(&out);
            if (!memcmp(&out, &in, sizeof in)) preserved++;
        }
        amx_tilerelease();
        DeleteFiber(fb);
        ConvertFiberToThread();
        t_info(f ? "preserved_float_switch" : "preserved_no_float_switch", "%d/%d", preserved, reps);
    }
}

/* ======================= D. XSTATE contexts ================================== */
TEST(amx_ctx_buffer_negotiation) {
    amx_need_api();
    amx_api();
    if (!A.InitializeContext2) SKIP("InitializeContext2 not exported");
    DWORD64 en = amx_os_enabled();
    if ((en & AMX_BITS) != AMX_BITS) SKIP("GetEnabledXStateFeatures()=0x%llx lacks AMX", (unsigned long long)en);
    DWORD l_no = actx_len(AMX_CTX_GET, en & ~AMX_BITS), l_amx = actx_len(AMX_CTX_GET, en), l_only = actx_len(AMX_CTX_GET, AMX_BITS);
    DWORD l_cfg = actx_len(AMX_CTX_GET, en & ~AMX_DATA_BIT);
    t_info("len_without_amx", "%lu", l_no); t_info("len_with_amx", "%lu", l_amx);
    t_info("len_amx_only", "%lu", l_only); t_info("len_tilecfg_no_tiledata", "%lu", l_cfg);
    REQUIREF(l_no && l_amx && l_only, "InitializeContext2(NULL) did not report a length (no=%lu amx=%lu only=%lu)", l_no, l_amx, l_only);
    CHECKF(l_amx >= l_no + 8192, "length with AMX %lu < length without %lu + 8192", l_amx, l_no);
    CHECKF(l_only >= 8192 + 64, "AMX-only length %lu too small", l_only);
    CHECKF(l_cfg < l_no + 8192, "TILECFG-only mask length %lu includes TILEDATA (no-AMX %lu)", l_cfg, l_no);
    if (A.InitializeContext) {
        DWORD ld = 0; PCONTEXT pc = NULL;
        SetLastError(0);
        A.InitializeContext(NULL, AMX_CTX_GET, &pc, &ld);
        t_info("len_InitializeContext_default", "%lu (%s TILEDATA)", ld, ld >= l_amx ? "includes" : "excludes");
    }
    /* short buffer: FALSE + ERROR_INSUFFICIENT_BUFFER + required length, buffer untouched beyond len */
    void *base = NULL;
    BYTE *buf = (BYTE *)t_guarded_alloc(l_amx - 1, &base);
    REQUIRE(buf);
    memset(buf, 0x5A, l_amx - 1);
    DWORD l = l_amx - 1; PCONTEXT pc = NULL;
    SetLastError(0);
    BOOL ok = A.InitializeContext2(buf, AMX_CTX_GET, &pc, &l, en);
    DWORD err = GetLastError();
    CHECKF(!ok && err == ERROR_INSUFFICIENT_BUFFER && l == l_amx, "short buffer: ret %d err %lu len %lu (want 0, 122, %lu)", ok, err, l, l_amx);
    t_guarded_free(base);
    /* exact buffers: features located inside, absent ones NULL */
    actx x;
    REQUIRE(actx_new(&x, AMX_CTX_GET, en & ~AMX_BITS, en & ~AMX_BITS));
    uint64_t comp = *(uint64_t *)((BYTE *)x.ctx + sizeof(CONTEXT) + ((LONG *)((BYTE *)x.ctx + sizeof(CONTEXT)))[4] + 8);
    DWORD ll = 0x999;
    void *p18 = A.LocateXStateFeature(x.ctx, 18, &ll);
    t_info("no_amx_ctx_locate18", "%p len %lu xcomp_bv 0x%llx", p18, ll, (unsigned long long)comp);
    if (comp & (1ull << 63)) CHECKF(p18 == NULL, "compacted context without TILEDATA located feature 18 at %p", p18);
    else if (p18) CHECKF(!actx_inside(&x, p18, 8192), "standard-format context: feature 18 inside a buffer sized without it?");
    actx_free(&x);
}

TEST(amx_ctx_locate_sizes) {
    amx_need_ctx();
    actx x; ACTX_AMX(&x);
    DWORD l17 = 0, l18 = 0;
    BYTE *p17 = (BYTE *)actx_loc(&x, 17, &l17), *p18 = (BYTE *)actx_loc(&x, 18, &l18);
    CHECKF(p17 && l17 == 64 && actx_inside(&x, p17, 64), "TILECFG located at %p len %lu", p17, l17);
    CHECKF(p18 && l18 == 8192 && actx_inside(&x, p18, 8192), "TILEDATA located at %p len %lu", p18, l18);
    if (p18 && cpu()->comp[18].align64) CHECKF(((uintptr_t)p18 & 63) == 0, "TILEDATA at %p not 64-byte aligned (CPUID align64)", p18);
    if (p17 && p18) CHECKF(p17 + 64 <= p18 || p18 + 8192 <= p17, "TILECFG and TILEDATA overlap");
    actx_free(&x);
}

/* One spinning worker holding tag `tag`; *x gets its AMX context. */
static void ctx_capture(amx_w *w, uint32_t tag, int release_first, actx *x) {
    memset(w, 0, sizeof *w);
    w->cpu = -1; w->tag = tag; w->release_first = release_first;
    tiles_pattern(&w->in, tag);
    ACTX_AMX(x);                      /* before the worker exists: a REQUIRE here leaves nothing running */
    w->h = CreateThread(NULL, 64 * 1024, amx_spin_worker, w, 0, NULL);
    REQUIREF(w->h, "CreateThread failed");
    if (workers_wait_ready(w, 1, 10000) != 1) { workers_join(w, 1); REQUIREF(0, "worker never loaded its tiles"); }
    REQUIRE(SuspendThread(w->h) != (DWORD)-1);
    BOOL ok = GetThreadContext(w->h, x->ctx);
    DWORD err = GetLastError();
    if (!ok) { ResumeThread(w->h); workers_join(w, 1); REQUIREF(0, "GetThreadContext(AMX mask) failed, error %lu", err); }
}

TEST(amx_ctx_get_reports_tiles) {
    amx_need_ctx();
    amx_w *w = workers_alloc(1); REQUIRE(w);
    actx x;
    ctx_capture(w, 0x600, 0, &x);
    DWORD64 m = actx_mask(&x);
    t_info("context_mask", "0x%llx", (unsigned long long)m);
    CHECKF((m & AMX_BITS) == AMX_BITS, "GetXStateFeaturesMask 0x%llx: live tiles must be reported (bits 17,18)", (unsigned long long)m);
    DWORD l17 = 0, l18 = 0;
    BYTE *p17 = (BYTE *)actx_loc(&x, 17, &l17), *p18 = (BYTE *)actx_loc(&x, 18, &l18);
    if (p17 && l17 == 64) CHECK_REG("context TILECFG", p17, g_full.b, 64); else CHECKF(0, "TILECFG not located (%p, %lu)", p17, l17);
    if (p18 && l18 == 8192) {
        for (int t = 0; t < 8; t++) { char lbl[48]; snprintf(lbl, sizeof lbl, "context tmm%d", t); CHECK_REG(lbl, p18 + 1024 * t, w->in.t[t], 1024); }
    } else CHECKF(0, "TILEDATA not located (%p, %lu)", p18, l18);
    ResumeThread(w->h);
    REQUIRE(workers_join(w, 1));
    workers_verify(w, 1, "after GetThreadContext(AMX)");
    actx_free(&x); workers_free(w);
}

TEST(amx_ctx_set_modifies_tiles) {
    amx_need_ctx();
    amx_w *w = workers_alloc(1); REQUIRE(w);
    actx x;
    ctx_capture(w, 0x610, 0, &x);
    DWORD l18 = 0;
    BYTE *p18 = (BYTE *)actx_loc(&x, 18, &l18);
    if (!p18 || l18 != 8192 || (actx_mask(&x) & AMX_BITS) != AMX_BITS) {
        ResumeThread(w->h); workers_join(w, 1); actx_free(&x); workers_free(w);
        REQUIREF(0, "AMX context not reported (located %p len %lu)", p18, l18);
    }
    amx_tiles want = w->in;
    t_pattern_reg(want.t[3], 1024, 0x6FF, 3);
    memset(want.t[6][15], 0x7E, 64);   /* one row of another tile */
    memcpy(p18, &want, 8192);
    BOOL ok = SetThreadContext(w->h, x.ctx);
    DWORD err = GetLastError();
    ResumeThread(w->h);
    REQUIRE(workers_join(w, 1));
    CHECKF(ok, "SetThreadContext(AMX) failed, error %lu", err);
    CHECK_REG("TILECFG after SetThreadContext", w->cfg_out.b, g_full.b, 64);
    for (int t = 0; t < 8; t++) { char lbl[64]; snprintf(lbl, sizeof lbl, "tmm%d after SetThreadContext", t); CHECK_REG(lbl, w->out.t[t], want.t[t], 1024); }
    actx_free(&x); workers_free(w);
}

TEST(amx_ctx_tilerelease_reports_init) {
    amx_need_ctx();
    amx_w *w = workers_alloc(1); REQUIRE(w);
    actx x;
    ctx_capture(w, 0x620, 1, &x);     /* worker loaded tiles, then TILERELEASE */
    DWORD64 m = actx_mask(&x);
    DWORD l17 = 0, l18 = 0;
    BYTE *p17 = (BYTE *)actx_loc(&x, 17, &l17), *p18 = (BYTE *)actx_loc(&x, 18, &l18);
    t_info("context_mask_after_tilerelease", "0x%llx", (unsigned long long)m);
    int cfg_init = !(m & AMX_CFG_BIT) || (p17 && all_zero(p17, 64));
    int data_init = !(m & AMX_DATA_BIT) || (p18 && all_zero(p18, 8192));
    CHECKF(cfg_init, "after TILERELEASE the context reports TILECFG in use with non-INIT content");
    CHECKF(data_init, "after TILERELEASE the context reports TILEDATA in use with non-zero content");
    if (m & AMX_BITS) t_note("context mask still lists AMX after TILERELEASE (content is INIT)");
    ResumeThread(w->h);
    REQUIRE(workers_join(w, 1));
    CHECKF(all_zero(&w->cfg_out, 64), "worker TILECFG not INIT after TILERELEASE + suspension");
    actx_free(&x); workers_free(w);
}

TEST(amx_ctx_copycontext) {
    amx_need_ctx();
    if (!A.CopyContext) SKIP("CopyContext not exported");
    amx_w *w = workers_alloc(1); REQUIRE(w);
    actx src;
    ctx_capture(w, 0x630, 0, &src);
    ResumeThread(w->h);
    REQUIRE(workers_join(w, 1));
    DWORD l = 0;
    BYTE *s18 = (BYTE *)actx_loc(&src, 18, &l), *s17 = (BYTE *)actx_loc(&src, 17, NULL);
    REQUIREF(s18 && s17 && (actx_mask(&src) & AMX_BITS) == AMX_BITS, "source AMX context incomplete");
    actx dst; ACTX_AMX(&dst);
    SetLastError(0);
    BOOL ok = A.CopyContext(dst.ctx, AMX_CTX_GET, src.ctx);
    CHECKF(ok, "CopyContext(AMX -> AMX) failed, error %lu", GetLastError());
    if (ok) {
        DWORD64 dm = actx_mask(&dst);
        CHECKF((dm & AMX_BITS) == AMX_BITS, "destination mask 0x%llx lacks AMX", (unsigned long long)dm);
        BYTE *d18 = (BYTE *)actx_loc(&dst, 18, NULL), *d17 = (BYTE *)actx_loc(&dst, 17, NULL);
        if (d18 && d17) { CHECK_REG("copied TILECFG", d17, s17, 64); CHECK_MEM(d18, s18, 8192); }
        else CHECKF(0, "copied context: AMX features not located");
        CHECK_MEM(d18, &w->in, 8192);
    }
    /* destination without room for AMX: recorded (Win11 behaviour not in the reference corpus); a guard page
     * after the exact-size buffer turns any overflow into a crash */
    DWORD64 en = amx_os_enabled();
    actx small;
    if (actx_new(&small, AMX_CTX_GET, en & ~AMX_BITS, en & ~AMX_BITS)) {
        SetLastError(0);
        BOOL ok2 = A.CopyContext(small.ctx, AMX_CTX_GET, src.ctx);
        DWORD e2 = GetLastError();
        void *p = actx_loc(&small, 18, NULL);
        t_info("copy_into_no_amx_dest", "ret=%d err=%lu mask=0x%llx tiledata_located=%d", ok2, e2,
               (unsigned long long)actx_mask(&small), p != NULL);
        actx_free(&small);
    }
    actx_free(&dst); actx_free(&src); workers_free(w);
}

/* ======================= E. isolation ========================================= */
typedef struct fresh_r { T_ALIGN(64) amx_tiles out; amx_cfg cfg0; uint64_t xinuse0; int xinuse_ok; int cpu; uint32_t tag; } fresh_r;
static DWORD WINAPI fresh_thread(void *p) {
    fresh_r *r = (fresh_r *)p;
    if (r->cpu >= 0) t_pin_thread(r->cpu);
    if (cpu_has(F_XGETBV1)) { r->xinuse0 = t_xgetbv(1); r->xinuse_ok = 1; }
    amx_sttilecfg(&r->cfg0);
    amx_ldtilecfg(&g_full);
    amx_store8(&r->out);
    /* leave own data behind for the next fresh thread to (not) see */
    static amx_tiles mine[8];
    tiles_pattern(&mine[r->tag & 7], r->tag);
    amx_load8(&mine[r->tag & 7]);
    return 0;
}
TEST(amx_thread_isolation_fresh_threads) {
    amx_enable();
    int cpu0 = first_cpu();
    amx_w *w = workers_alloc(1); REQUIRE(w);
    int n = workers_start(w, 1, 0x700, cpu0);
    REQUIRE(n == 1);
    workers_wait_ready(w, 1, 10000);
    fresh_r *r = (fresh_r *)VirtualAlloc(NULL, sizeof(fresh_r), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(r);
    int bad = 0;
    for (int i = 0; i < 8 && !bad; i++) {
        memset(r, 0xEE, sizeof *r);
        r->cpu = cpu0; r->tag = 0x710u + (uint32_t)i; r->xinuse_ok = 0;
        HANDLE h = CreateThread(NULL, 64 * 1024, fresh_thread, r, 0, NULL);
        if (!h) break;
        DWORD wr = WaitForSingleObject(h, 10000);
        DWORD code = 0; GetExitCodeThread(h, &code);
        CloseHandle(h);
        CHECKF(wr == WAIT_OBJECT_0, "fresh thread %d hung", i);
        if (i == 0 && r->xinuse_ok) t_info("fresh_thread_xinuse", "0x%llx", (unsigned long long)r->xinuse0);
        if (r->xinuse_ok && (r->xinuse0 & AMX_BITS))
            t_note("fresh thread %d starts with XINUSE[18:17] set (0x%llx)", i, (unsigned long long)r->xinuse0);
        if (!all_zero(&r->cfg0, 64)) { CHECK_REG("fresh thread initial TILECFG (must be INIT)", r->cfg0.b, (uint8_t[64]){ 0 }, 64); bad = 1; }
        if (!all_zero(&r->out, sizeof r->out)) {
            const uint8_t *b = (const uint8_t *)&r->out; size_t k = 0;
            while (k < sizeof r->out && !b[k]) k++;
            uint64_t v; memcpy(&v, b + (k & ~(size_t)7), 8);
            uint32_t tag = 0, reg = 0, word = 0;
            int ex = t_pattern_explain(v, 0x800, &tag, &reg, &word);
            CHECKF(0, "fresh thread %d: tiles after its first LDTILECFG are not zero (byte %zu = 0x%02x%s)", i, k, b[k],
                   ex ? (tag == 0x700 ? ", data of the live holder thread" : ", data of an exited thread") : "");
            bad = 1;
        }
    }
    VirtualFree(r, 0, MEM_RELEASE);
    REQUIRE(workers_join(w, 1));
    workers_verify(w, 1, "while fresh threads ran on the same CPU");
    workers_free(w);
}

#endif /* T_X64 */
