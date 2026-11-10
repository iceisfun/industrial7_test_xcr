/* xstate_windows - Windows XSTATE API contracts: GetEnabledXStateFeatures, InitializeContext(2),
 * Get/SetXStateFeaturesMask, LocateXStateFeature, CopyContext, the ntdll extended-context layer
 * (CONTEXT_EX, RtlGetExtendedContextLength, RtlInitializeExtendedContext, RtlLocateExtendedFeature,
 * RtlGet/SetExtendedFeaturesMask, RtlCopyContext, RtlCopyExtendedContext) and Windows 11 optional
 * features (GetThreadEnabledXStateFeatures, EnableProcessOptionalXStateFeatures).
 *
 * API reporting and actual execution are separate cases (api_* / init_* / locate_* ... versus exec_*
 * and thread_*), so an OS whose AVX executes but whose XSTATE context APIs are incomplete shows both facts.
 *
 * Contracts are pinned from the Microsoft docs plus the Windows 10 22H2 (19041) reference binaries
 * (kernelbase/ntdll decompilations checked through the corpus explorer):
 *  - kernelbase!InitializeContext = InitializeContext2(.., RtlGetEnabledExtendedFeatures(~0)).
 *    InitializeContext2: if RtlGetEnabledExtendedFeatures(~0)==0 the XSTATE bit (0x40) is stripped;
 *    RtlGetExtendedContextLength2 validates the flags FIRST (bad flags -> NTSTATUS -> Win32 error, length
 *    untouched); then Buffer==NULL or *ContextLength < required -> *ContextLength = required,
 *    SetLastError(ERROR_INSUFFICIENT_BUFFER=122), FALSE. On success *Context = RtlLocateLegacyContext().
 *  - ntdll!RtlpValidateContextFlags: valid iff
 *      (f & 0x27FFFF80) == 0x10000 (i386)  or (f & 0x27FFFFA0) == 0x100000 (AMD64)  or ARM/ARM64 forms;
 *    else STATUS_INVALID_PARAMETER (0xC000000D -> ERROR_INVALID_PARAMETER 87). So: no arch bit, both arch
 *    bits, AMD64|0x20, i386|0x80, bit 29 are invalid; bits 27, 28, 30, 31 (CONTEXT_EXCEPTION_*,
 *    CONTEXT_SERVICE_ACTIVE) are accepted. XSTATE with XState.EnabledFeatures==0 -> STATUS_NOT_SUPPORTED.
 *  - ntdll!RtlInitializeExtendedContext2: CONTEXT at align16(buf) (AMD64) / align4(buf) (i386);
 *    CONTEXT_EX {All, Legacy, XState} (3 x {LONG Offset; DWORD Length}, 0x18 bytes) directly after the
 *    CONTEXT (sizeof(CONTEXT) = 0x4D0 / 0x2CC). Legacy.Offset = All.Offset = -sizeof(CONTEXT);
 *    Legacy.Length = sizeof(CONTEXT) (i386 without CONTEXT_EXTENDED_REGISTERS: 0xCC).
 *    With XSTATE: XState chunk (= XSAVE header, i.e. byte 512 of an XSAVE image) at 64-byte alignment,
 *    zeroed, XState.Length = area length - 512, XCOMP_BV = mask|bit63 when the OS uses the compacted
 *    format; All.Length = XState.Offset - Legacy.Offset + XState.Length.
 *    Without XSTATE: XState = {0x19, 0}.
 *  - kernelbase!LocateXStateFeature: AMD64: id 0 -> &FltSave (len 0xA0), id 1 -> &FltSave.XmmRegisters
 *    (+0x1A0, len 0x100); i386: id 0 -> ExtendedRegisters (+0xCC, len 0xA0), id 1 -> +0x16C (len 0x80).
 *    Legacy ids work without CONTEXT_XSTATE. id >= 2 needs CONTEXT_XSTATE and goes to
 *    RtlLocateExtendedFeature2: id in [2,63], enabled (user|supervisor), XCOMP_BV bit set when compacted;
 *    else NULL with *Length untouched. Context without either arch bit -> NULL.
 *  - kernelbase!GetXStateFeaturesMask: no arch bit -> STATUS_NOT_SUPPORTED (ERROR_NOT_SUPPORTED 50);
 *    bits 0|1 reported iff CONTEXT_FLOATING_POINT (AMD64) / CONTEXT_EXTENDED_REGISTERS (i386);
 *    XSTATE contexts add XSTATE_BV & ~3.
 *  - kernelbase!SetXStateFeaturesMask: (mask & 3) sets CONTEXT_FLOATING_POINT / CONTEXT_EXTENDED_REGISTERS;
 *    XSTATE contexts get XSTATE_BV = mask & enabled & ~3 (unsupported bits silently dropped, TRUE);
 *    non-XSTATE context with mask & ~3 -> ERROR_INVALID_PARAMETER; no arch bit -> ERROR_NOT_SUPPORTED.
 *  - ntdll!RtlCopyContext: dest flags |= (src flags & requested); copying XSTATE into a destination that
 *    is not XSTATE-initialized -> STATUS_BUFFER_OVERFLOW (0x80000005 -> ERROR_MORE_DATA 234);
 *    destination XSTATE_BV = src XSTATE_BV & enabled & ~3; destination chunk too small -> 0x80000005.
 *    RtlpCopyLegacyContext* copy only the parts named by (src & requested); RtlpCopyXStateChunk zeroes the
 *    64-byte destination header and rebuilds XSTATE_BV/XCOMP_BV. i386: copying CONTEXT_EXTENDED_REGISTERS
 *    into an XSTATE destination (and SetXStateFeaturesMask(mask & 3)) raises Legacy.Length 0xCC -> 0x2CC;
 *    RtlpCopyExtendedContext rejects dest Legacy.Offset != src or dest Legacy.Length < src (0xC000000D).
 *  - ntoskrnl!PspGetContextThreadInternal (user caller): captures into a full kernel copy, then
 *    RtlpWriteExtendedContext -> RtlpCopyXStateChunk (too-small user chunk -> 0x80000005, ERROR_MORE_DATA).
 *    RtlpReadExtendedContextLayout rejects a CONTEXT_EX whose Legacy.Offset != -sizeof(CONTEXT) etc. with
 *    0xC000000D before writing; a chunk not 64-byte aligned raises STATUS_DATATYPE_MISALIGNMENT (998).
 *    The i386 (WOW64) conversion lives in wow64.dll, not in the reference corpus: those results are recorded.
 *  - kernelbase!GetEnabledXStateFeatures = RtlGetEnabledExtendedFeatures(~0), or 3 when that is 0;
 *    RtlGetEnabledExtendedFeatures(m) = (EnabledFeatures | EnabledSupervisorFeatures) & m.
 */
#include "t.h"
#include <immintrin.h>

/* ======================= per-arch constants ============================== */
#ifdef T_X64
#define XA_ARCH        0x00100000u
#define XA_FPFLAG      (XA_ARCH | 0x08u)   /* flag that carries XSTATE bits 0|1 (CONTEXT_FLOATING_POINT) */
#define XA_FP_ALL      (XA_ARCH | 0x08u)
#define XA_BADLOW      0x20u               /* low flag bit RtlpValidateContextFlags rejects for AMD64 */
#define XA_CTX_ALIGN   16u
#define XA_CTX_SIZE    0x4D0u
#define XA_LEG0_OFF    0x100u
#define XA_LEG0_LEN    0xA0u
#define XA_LEG1_OFF    0x1A0u
#define XA_LEG1_LEN    0x100u
#define XA_OTHER_ARCH  0x00010000u
#define XA_NXMM        16
#else
#define XA_ARCH        0x00010000u
#define XA_FPFLAG      (XA_ARCH | 0x20u)   /* CONTEXT_EXTENDED_REGISTERS */
#define XA_FP_ALL      (XA_ARCH | 0x28u)   /* FLOATING_POINT | EXTENDED_REGISTERS */
#define XA_BADLOW      0x80u
#define XA_CTX_ALIGN   4u
#define XA_CTX_SIZE    0x2CCu
#define XA_LEG0_OFF    0xCCu
#define XA_LEG0_LEN    0xA0u
#define XA_LEG1_OFF    0x16Cu
#define XA_LEG1_LEN    0x80u
#define XA_OTHER_ARCH  0x00100000u
#define XA_NXMM        8
#endif
#define XA_CONTROL     (XA_ARCH | 0x01u)
#define XA_INTEGER     (XA_ARCH | 0x02u)
#define XA_XSTATE      (XA_ARCH | 0x40u)
#define XA_ALL         ((DWORD)CONTEXT_ALL)

#define X_ERR_SENTINEL 0xDEADBEEFu
#define X_STATUS_SUCCESS            0x00000000L
#define X_STATUS_INVALID_PARAMETER  ((LONG)0xC000000Du)
#define X_STATUS_NOT_SUPPORTED      ((LONG)0xC00000BBu)
#define X_STATUS_BUFFER_OVERFLOW    ((LONG)0x80000005u)
#define X_BIT63 (1ull << 63)

typedef struct { LONG Offset; DWORD Length; } X_CHUNK;
typedef struct { X_CHUNK All, Legacy, XState; } X_CTX_EX;     /* ntdll CONTEXT_EX (0x18 bytes) */

/* ======================= dynamically resolved APIs ======================= */
typedef BOOL    (WINAPI *pfnInitializeContext)(PVOID, DWORD, PCONTEXT *, PDWORD);
typedef BOOL    (WINAPI *pfnInitializeContext2)(PVOID, DWORD, PCONTEXT *, PDWORD, ULONG64);
typedef DWORD64 (WINAPI *pfnGetEnabledXStateFeatures)(void);
typedef BOOL    (WINAPI *pfnGetXStateFeaturesMask)(PCONTEXT, PDWORD64);
typedef BOOL    (WINAPI *pfnSetXStateFeaturesMask)(PCONTEXT, DWORD64);
typedef PVOID   (WINAPI *pfnLocateXStateFeature)(PCONTEXT, DWORD, PDWORD);
typedef BOOL    (WINAPI *pfnCopyContext)(PCONTEXT, DWORD, PCONTEXT);
typedef DWORD64 (WINAPI *pfnGetThreadEnabledXStateFeatures)(void);
typedef BOOL    (WINAPI *pfnEnableProcessOptionalXStateFeatures)(DWORD64);
typedef ULONG64 (NTAPI *pfnRtlGetEnabledExtendedFeatures)(ULONG64);
typedef LONG    (NTAPI *pfnRtlGetExtendedContextLength)(ULONG, PULONG);
typedef LONG    (NTAPI *pfnRtlGetExtendedContextLength2)(ULONG, PULONG, ULONG64);
typedef LONG    (NTAPI *pfnRtlInitializeExtendedContext)(PVOID, ULONG, X_CTX_EX **);
typedef LONG    (NTAPI *pfnRtlInitializeExtendedContext2)(PVOID, ULONG, X_CTX_EX **, ULONG64);
typedef PVOID   (NTAPI *pfnRtlLocateLegacyContext)(X_CTX_EX *, PULONG);
typedef PVOID   (NTAPI *pfnRtlLocateExtendedFeature)(X_CTX_EX *, ULONG, PULONG);
typedef ULONG64 (NTAPI *pfnRtlGetExtendedFeaturesMask)(X_CTX_EX *);
typedef void    (NTAPI *pfnRtlSetExtendedFeaturesMask)(X_CTX_EX *, ULONG64);
typedef LONG    (NTAPI *pfnRtlCopyContext)(PCONTEXT, ULONG, PCONTEXT);
typedef LONG    (NTAPI *pfnRtlCopyExtendedContext)(X_CTX_EX *, ULONG, X_CTX_EX *);

static struct {
    int resolved;
    pfnInitializeContext InitializeContext;
    pfnInitializeContext2 InitializeContext2;
    pfnGetEnabledXStateFeatures GetEnabledXStateFeatures;
    pfnGetXStateFeaturesMask GetXStateFeaturesMask;
    pfnSetXStateFeaturesMask SetXStateFeaturesMask;
    pfnLocateXStateFeature LocateXStateFeature;
    pfnCopyContext CopyContext;
    pfnGetThreadEnabledXStateFeatures GetThreadEnabledXStateFeatures;
    pfnEnableProcessOptionalXStateFeatures EnableProcessOptionalXStateFeatures;
    pfnRtlGetEnabledExtendedFeatures RtlGetEnabledExtendedFeatures;
    pfnRtlGetExtendedContextLength RtlGetExtendedContextLength;
    pfnRtlGetExtendedContextLength2 RtlGetExtendedContextLength2;
    pfnRtlInitializeExtendedContext RtlInitializeExtendedContext;
    pfnRtlInitializeExtendedContext2 RtlInitializeExtendedContext2;
    pfnRtlLocateLegacyContext RtlLocateLegacyContext;
    pfnRtlLocateExtendedFeature RtlLocateExtendedFeature;
    pfnRtlGetExtendedFeaturesMask RtlGetExtendedFeaturesMask;
    pfnRtlSetExtendedFeaturesMask RtlSetExtendedFeaturesMask;
    pfnRtlCopyContext RtlCopyContext;
    pfnRtlCopyExtendedContext RtlCopyExtendedContext;
} X;

static void xa_resolve(void) {
    if (X.resolved) return;
#define K32(n) X.n = (pfn##n)t_proc("kernel32.dll", #n)
#define NTD(n) X.n = (pfn##n)t_proc("ntdll.dll", #n)
    K32(InitializeContext); K32(InitializeContext2); K32(GetEnabledXStateFeatures); K32(GetXStateFeaturesMask);
    K32(SetXStateFeaturesMask); K32(LocateXStateFeature); K32(CopyContext); K32(GetThreadEnabledXStateFeatures);
    K32(EnableProcessOptionalXStateFeatures);
    NTD(RtlGetEnabledExtendedFeatures); NTD(RtlGetExtendedContextLength); NTD(RtlGetExtendedContextLength2);
    NTD(RtlInitializeExtendedContext); NTD(RtlInitializeExtendedContext2); NTD(RtlLocateLegacyContext);
    NTD(RtlLocateExtendedFeature); NTD(RtlGetExtendedFeaturesMask); NTD(RtlSetExtendedFeaturesMask);
    NTD(RtlCopyContext); NTD(RtlCopyExtendedContext);
#undef K32
#undef NTD
    X.resolved = 1;
}

/* Every case that uses the Windows 7 SP1 XSTATE API set calls this first. */
static void need_core_api(void) {
    xa_resolve();
    if (!X.InitializeContext || !X.GetEnabledXStateFeatures || !X.LocateXStateFeature ||
        !X.GetXStateFeaturesMask || !X.SetXStateFeaturesMask || !X.CopyContext)
        SKIP("XSTATE context API (Windows 7 SP1 set) not exported by kernel32");
}
/* Wine 9 returns RtlGetEnabledExtendedFeatures(~0) unchanged, i.e. 0 on a CPU without XSAVE (Windows 10
 * kernelbase!GetEnabledXStateFeatures returns 3 then): documented Wine gap, checks unchanged. */
static DWORD64 enabled(void) {
    need_core_api();
    DWORD64 en = X.GetEnabledXStateFeatures();
    if (en == 0 && t_is_wine()) t_expect_fail("Wine %s: GetEnabledXStateFeatures returns 0 without XSAVE", t_wine_version());
    return en;
}
static void need_avx_api(void) {
    if (!(enabled() & T_XS_AVX)) SKIP("GetEnabledXStateFeatures() does not report AVX (0x%llx)", (unsigned long long)enabled());
}
static void need_rtl(void *p, const char *name) {
    xa_resolve();
    if (!p) SKIP("ntdll!%s not exported", name);
}
/* Does the OS manage any state with XSAVE at all? kernelbase!GetEnabledXStateFeatures reports 3 both for
 * "x87+SSE via XSAVE" and for "no XSAVE" (Rtl(~0)==0), so ask ntdll. Without it InitializeContext(2) strips
 * CONTEXT_XSTATE (0x40) and RtlpValidateContextFlags returns STATUS_NOT_SUPPORTED for XSTATE flags. */
static int os_xstate(void) {
    xa_resolve();
    if (X.RtlGetEnabledExtendedFeatures) return X.RtlGetEnabledExtendedFeatures(~0ull) != 0;
    return cpu()->osxsave;
}
static void need_xstate_ctx(void) {
    need_core_api();
    if (!os_xstate()) SKIP("OS enables no XSAVE-managed state (RtlGetEnabledExtendedFeatures(~0)=0): CONTEXT_XSTATE unsupported");
}

/* Error-code-only cases call this first: on Wine a known gap turns their failures into XFAIL.
 * Never used for return-value or memory-safety checks. */
static void wine_known_gap(const char *what) { if (t_is_wine()) t_expect_fail("known Wine gap: %s", what); }

/* LocateXStateFeature: Windows 19041 leaves *Length untouched when it returns NULL. The docs do not
 * promise it, so this is recorded, not checked. */
static int g_len_written;
static void len_untouched_note(DWORD l, DWORD f) {
    if (l != 0x999 && g_len_written++ < 3)
        t_note("feature %lu: NULL returned but *Length written (%lu) - Windows 19041 leaves it untouched", f, l);
}

/* ======================= context buffer helpers ========================== */
typedef struct xctx {
    BYTE    *buf;      /* allocation passed to InitializeContext */
    void    *base;     /* guarded allocation base (t_guarded_free) */
    DWORD    len;      /* length used */
    CONTEXT *ctx;
} xctx;

static X_CTX_EX *xex(const CONTEXT *ctx) { return (X_CTX_EX *)((BYTE *)ctx + sizeof(CONTEXT)); }
static BYTE *xchunk(const CONTEXT *ctx) { X_CTX_EX *e = xex(ctx); return (BYTE *)e + e->XState.Offset; }
static uint64_t xs_bv(const CONTEXT *ctx) { return *(uint64_t *)xchunk(ctx); }
static uint64_t xs_comp(const CONTEXT *ctx) { return ((uint64_t *)xchunk(ctx))[1]; }

/* Query the length for flags (InitializeContext with NULL buffer). 0 on unexpected result. */
static DWORD init_len(DWORD flags) {
    DWORD len = 0; PCONTEXT c = NULL;
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.InitializeContext(NULL, flags, &c, &len);
    if (ok || GetLastError() != ERROR_INSUFFICIENT_BUFFER) return 0;
    return len;
}
static DWORD init2_len(DWORD flags, DWORD64 mask) {
    DWORD len = 0; PCONTEXT c = NULL;
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.InitializeContext2(NULL, flags, &c, &len, mask);
    if (ok || GetLastError() != ERROR_INSUFFICIENT_BUFFER) return 0;
    return len;
}
/* Allocate a guarded buffer (ends at a PAGE_NOACCESS page) of exactly the required length and
 * initialize it. mask = ~0 -> InitializeContext, else InitializeContext2(mask). fill = initial byte. */
static int xctx_new2(xctx *x, DWORD flags, DWORD64 mask, int use2, BYTE fill) {
    memset(x, 0, sizeof *x);
    DWORD len = use2 ? init2_len(flags, mask) : init_len(flags);
    if (!len) return 0;
    x->buf = (BYTE *)t_guarded_alloc(len, &x->base);
    if (!x->buf) return 0;
    memset(x->buf, fill, len);
    x->len = len;
    DWORD l2 = len;
    BOOL ok = use2 ? X.InitializeContext2(x->buf, flags, &x->ctx, &l2, mask) : X.InitializeContext(x->buf, flags, &x->ctx, &l2);
    if (!ok || !x->ctx) { t_guarded_free(x->base); memset(x, 0, sizeof *x); return 0; }
    return 1;
}
static int xctx_new(xctx *x, DWORD flags) { return xctx_new2(x, flags, 0, 0, 0xA5); }
static void xctx_free(xctx *x) { if (x->base) t_guarded_free(x->base); memset(x, 0, sizeof *x); }
#define XCTX_REQ(x, flags) REQUIREF(xctx_new((x), (flags)), "InitializeContext(flags=0x%lx) failed (error %lu)", (unsigned long)(flags), GetLastError())

static int in_buf(const xctx *x, const void *p, size_t n) {
    return (const BYTE *)p >= x->buf && (const BYTE *)p + n <= x->buf + x->len;
}

/* Expected XSAVE-image offset of component f (>= 2) for a context with the given XCOMP_BV. */
static uint32_t xs_expected_offset(uint64_t xcomp, int f) {
    const t_cpu_info *c = cpu();
    if (!(xcomp & X_BIT63)) return c->comp[f].offset;
    uint32_t off = 576;
    for (int i = 2; i < f; i++)
        if (xcomp >> i & 1) {
            if (c->comp[i].align64) off = (off + 63) & ~63u;
            off += c->comp[i].size;
        }
    if (c->comp[f].align64) off = (off + 63) & ~63u;
    return off;
}

/* ======================= A. constants and exports ======================== */
TEST(const_context_flag_values) {
#ifdef T_X64
    CHECK_EQ(CONTEXT_AMD64, 0x00100000);
    CHECK_EQ(CONTEXT_XSTATE, 0x00100040);
    CHECK_EQ(CONTEXT_FLOATING_POINT, 0x00100008);
    CHECK_EQ(CONTEXT_ALL, 0x0010001F);
    CHECK_EQ(sizeof(CONTEXT), 0x4D0);
    CHECK_EQ(offsetof(CONTEXT, ContextFlags), 0x30);
    CHECK_EQ(offsetof(CONTEXT, MxCsr), 0x34);
    CHECK_EQ(offsetof(CONTEXT, FltSave), 0x100);
    CHECK_EQ(offsetof(CONTEXT, Xmm0), 0x1A0);
    CHECK_EQ(offsetof(CONTEXT, VectorRegister), 0x300);
    CHECK_EQ(offsetof(CONTEXT, Rip), 0xF8);
    CHECK_EQ(sizeof(XSAVE_FORMAT), 512);
#else
    CHECK_EQ(CONTEXT_i386, 0x00010000);
    CHECK_EQ(XA_XSTATE, 0x00010040);
    CHECK_EQ(CONTEXT_EXTENDED_REGISTERS, 0x00010020);
    CHECK_EQ(CONTEXT_ALL, 0x0001003F);
    CHECK_EQ(sizeof(CONTEXT), 0x2CC);
    CHECK_EQ(offsetof(CONTEXT, ContextFlags), 0);
    CHECK_EQ(offsetof(CONTEXT, FloatSave), 0x1C);
    CHECK_EQ(offsetof(CONTEXT, Eip), 0xB8);
    CHECK_EQ(offsetof(CONTEXT, ExtendedRegisters), 0xCC);
#endif
    CHECK_EQ(sizeof(X_CTX_EX), 0x18);
    CHECK_EQ(XSTATE_LEGACY_FLOATING_POINT, 0);
    CHECK_EQ(XSTATE_LEGACY_SSE, 1);
    CHECK_EQ(XSTATE_AVX, 2);
    CHECK_EQ(XSTATE_AVX512_KMASK, 5);
    CHECK_EQ(XSTATE_AMX_TILE_DATA, 18);
    CHECK_EQ(MAXIMUM_XSTATE_FEATURES, 64);
    CHECK_EQ(XSTATE_MASK_LEGACY, 3);
    CHECK_EQ(XSTATE_MASK_AVX512, 0xE0);
}

TEST(api_exports_present) {
    xa_resolve();
    const t_env *e = t_env_get();
    struct { const char *n; void *p; int since_win10_2004; } a[] = {
        { "kernel32!InitializeContext", (void *)X.InitializeContext, 0 },
        { "kernel32!CopyContext", (void *)X.CopyContext, 0 },
        { "kernel32!GetEnabledXStateFeatures", (void *)X.GetEnabledXStateFeatures, 0 },
        { "kernel32!GetXStateFeaturesMask", (void *)X.GetXStateFeaturesMask, 0 },
        { "kernel32!SetXStateFeaturesMask", (void *)X.SetXStateFeaturesMask, 0 },
        { "kernel32!LocateXStateFeature", (void *)X.LocateXStateFeature, 0 },
        { "ntdll!RtlGetEnabledExtendedFeatures", (void *)X.RtlGetEnabledExtendedFeatures, 0 },
        { "ntdll!RtlGetExtendedContextLength", (void *)X.RtlGetExtendedContextLength, 0 },
        { "ntdll!RtlInitializeExtendedContext", (void *)X.RtlInitializeExtendedContext, 0 },
        { "ntdll!RtlLocateLegacyContext", (void *)X.RtlLocateLegacyContext, 0 },
        { "ntdll!RtlLocateExtendedFeature", (void *)X.RtlLocateExtendedFeature, 0 },
        { "ntdll!RtlGetExtendedFeaturesMask", (void *)X.RtlGetExtendedFeaturesMask, 0 },
        { "ntdll!RtlSetExtendedFeaturesMask", (void *)X.RtlSetExtendedFeaturesMask, 0 },
        { "ntdll!RtlCopyExtendedContext", (void *)X.RtlCopyExtendedContext, 0 },
        { "ntdll!RtlCopyContext", (void *)X.RtlCopyContext, 0 },
        { "kernel32!InitializeContext2", (void *)X.InitializeContext2, 1 },
        { "ntdll!RtlGetExtendedContextLength2", (void *)X.RtlGetExtendedContextLength2, 1 },
        { "ntdll!RtlInitializeExtendedContext2", (void *)X.RtlInitializeExtendedContext2, 1 },
    };
    char buf[64];
    for (size_t i = 0; i < T_COUNTOF(a); i++) {
        snprintf(buf, sizeof buf, "export.%s", a[i].n);
        t_info(buf, "%s", a[i].p ? "present" : "missing");
        /* Windows 10 ships the Windows 7 SP1 XSTATE set; the *2 variants arrived in 19041 (2004). */
        if (e->os_major >= 10 && !a[i].since_win10_2004)
            CHECKF(a[i].p != NULL, "%s missing on a Windows %lu.%lu build %lu system", a[i].n, e->os_major, e->os_minor, e->os_build);
        if (e->os_major >= 10 && e->os_build >= 19041 && a[i].since_win10_2004)
            CHECKF(a[i].p != NULL, "%s missing on build %lu (present since 19041)", a[i].n, e->os_build);
    }
    t_info("export.GetThreadEnabledXStateFeatures", "%s", X.GetThreadEnabledXStateFeatures ? "present" : "missing");
    t_info("export.EnableProcessOptionalXStateFeatures", "%s", X.EnableProcessOptionalXStateFeatures ? "present" : "missing");
}

/* ======================= B. GetEnabledXStateFeatures ===================== */
TEST(api_enabled_features_legacy_bits) {
    DWORD64 en = enabled();
    t_info("enabled_features", "0x%llx", (unsigned long long)en);
    CHECKF((en & 3) == 3, "x87|SSE must always be reported, got 0x%llx", (unsigned long long)en);
    CHECKF(!(en & X_BIT63), "bit 63 (compaction) must not be reported as a feature: 0x%llx", (unsigned long long)en);
}

TEST(api_enabled_features_vs_xcr0) {
    DWORD64 en = enabled();
    const t_cpu_info *c = cpu();
    if (!c->osxsave) {
        CHECKF(en == 3, "OSXSAVE clear (no XSAVE-managed state) but API reports 0x%llx (want 3)", (unsigned long long)en);
        return;
    }
    REQUIRE(c->xcr0_valid);
    uint64_t xcr0 = c->xcr0;
    t_info("xcr0", "0x%llx", (unsigned long long)xcr0);
    uint64_t user = 0;
    for (int i = 0; i < 64; i++) if ((en >> i & 1) && !(c->comp[i].valid && c->comp[i].supervisor)) user |= 1ull << i;
    /* A user-state feature the OS reports must be enabled in XCR0: otherwise XSAVE/XRSTOR of a context
     * containing it faults (#GP) or silently loses it. */
    CHECKF((user & ~xcr0) == 0, "API reports user features not in XCR0: api=0x%llx xcr0=0x%llx extra=0x%llx",
           (unsigned long long)en, (unsigned long long)xcr0, (unsigned long long)(user & ~xcr0));
    /* Vector state enabled in XCR0 must be reported, or no XSTATE-aware software can locate it. */
    static const int must[] = { 2, 5, 6, 7 };
    if ((xcr0 & 0xE0) && !(en & 0xE0) && t_is_wine())   /* known Wine gap (seen with Wine 9 on an AVX-512 CPU model): AVX-512 runs but the API does not report it */
        t_expect_fail("Wine %s: GetEnabledXStateFeatures does not report AVX-512 state", t_wine_version());
    for (size_t k = 0; k < T_COUNTOF(must); k++) {
        int b = must[k];
        if (xcr0 >> b & 1)
            CHECKF(en >> b & 1, "XCR0 bit %d (%s) enabled but GetEnabledXStateFeatures lacks it (api=0x%llx xcr0=0x%llx)",
                   b, xstate_comp_name(b), (unsigned long long)en, (unsigned long long)xcr0);
    }
    uint64_t diff = (xcr0 & ~en) & ~0x3ull;
    if (diff) t_note("XCR0 bits not reported by the API (OS choice): 0x%llx", (unsigned long long)diff);
}

TEST(api_enabled_features_vs_cpuid) {
    DWORD64 en = enabled();
    const t_cpu_info *c = cpu();
    if (!c->osxsave) SKIP("OSXSAVE clear");
    uint64_t sup = c->xcr0_supported | c->xss_supported | 3;
    CHECKF((en & ~sup) == 0, "API reports features the CPU does not enumerate in CPUID.(0Dh,0/1): api=0x%llx cpuid=0x%llx",
           (unsigned long long)en, (unsigned long long)sup);
    for (int i = 2; i < 63; i++) {
        if (!(en >> i & 1)) continue;
        CHECKF(c->comp[i].valid && c->comp[i].size > 0, "reported feature %d (%s) has no CPUID.(0Dh,%d) size", i, xstate_comp_name(i), i);
    }
}

TEST(api_enabled_features_supervisor_info) {
    DWORD64 en = enabled();
    const t_cpu_info *c = cpu();
    uint64_t sv = 0;
    for (int i = 0; i < 64; i++) if ((en >> i & 1) && c->comp[i].valid && c->comp[i].supervisor) sv |= 1ull << i;
    t_info("enabled_supervisor", "0x%llx", (unsigned long long)sv);
    if (sv) t_note("API includes supervisor components 0x%llx (EnabledSupervisorFeatures, e.g. CET_U): not in XCR0 by design", (unsigned long long)sv);
    /* Supervisor components are managed through IA32_XSS: they can only appear if CPUID enumerates them there. */
    CHECKF((sv & ~c->xss_supported) == 0, "supervisor bits 0x%llx not in CPUID.(0Dh,1) ECX:EDX 0x%llx",
           (unsigned long long)sv, (unsigned long long)c->xss_supported);
}

static volatile DWORD64 g_en_thread;
static DWORD WINAPI en_thread(void *p) { (void)p; g_en_thread = X.GetEnabledXStateFeatures(); return 0; }
TEST(api_enabled_features_stable) {
    DWORD64 en = enabled();
    for (int i = 0; i < 200; i++) {
        DWORD64 v = X.GetEnabledXStateFeatures();
        if (v != en) { CHECK_EQ(v, en); break; }
    }
    g_en_thread = 0;
    HANDLE h = CreateThread(NULL, 0, en_thread, NULL, 0, NULL);
    if (!h) SKIP("CreateThread failed (%lu)", GetLastError());
    CHECK_EQ(WaitForSingleObject(h, 5000), WAIT_OBJECT_0);
    CloseHandle(h);
    CHECK_EQ(g_en_thread, en);
}

TEST(api_enabled_features_env_record) {
    DWORD64 en = enabled();
    const t_env *e = t_env_get();
    CHECK(e->api_xstate_present);
    CHECK_EQ(e->api_xstate_features, en);
}

TEST(rtl_enabled_features_masking) {
    DWORD64 en = enabled();
    need_rtl((void *)X.RtlGetEnabledExtendedFeatures, "RtlGetEnabledExtendedFeatures");
    ULONG64 full = X.RtlGetEnabledExtendedFeatures(~0ull);
    t_info("rtl_enabled_features", "0x%llx", (unsigned long long)full);
    /* kernelbase!GetEnabledXStateFeatures = Rtl(~0) or 3 when that is 0 */
    CHECK_EQ(full ? full : 3, en);
    CHECK_EQ(X.RtlGetEnabledExtendedFeatures(0), 0);
    static const ULONG64 masks[] = { 1, 2, 3, 4, 7, 0xE0, 0xE7, 0x200, 0x60000, 0x80000, 1ull << 62, X_BIT63, 0xFFFFFFFFull, 0xFFFFFFFF00000000ull };
    for (size_t i = 0; i < T_COUNTOF(masks); i++) {
        ULONG64 got = X.RtlGetEnabledExtendedFeatures(masks[i]);
        CHECKF(got == (full & masks[i]), "Rtl(0x%llx)=0x%llx want 0x%llx", (unsigned long long)masks[i],
               (unsigned long long)got, (unsigned long long)(full & masks[i]));
    }
}

/* ======================= C. InitializeContext size negotiation ============ */
TEST(init_size_query_null_buffer) {
    need_core_api();
    DWORD flags[] = { XA_CONTROL, XA_ALL, XA_XSTATE, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(flags); i++) {
        DWORD len = 0; PCONTEXT c = (PCONTEXT)(uintptr_t)0x1234;
        SetLastError(X_ERR_SENTINEL);
        BOOL ok = X.InitializeContext(NULL, flags[i], &c, &len);
        DWORD err = GetLastError();
        CHECKF(!ok, "flags 0x%lx: NULL buffer must fail", (unsigned long)flags[i]);
        CHECKF(err == ERROR_INSUFFICIENT_BUFFER, "flags 0x%lx: error %lu want 122", (unsigned long)flags[i], err);
        CHECKF(len >= sizeof(CONTEXT), "flags 0x%lx: len %lu < sizeof(CONTEXT)", (unsigned long)flags[i], len);
        CHECKF(len < 0x100000, "flags 0x%lx: absurd length %lu", (unsigned long)flags[i], len);
        CHECKF(c == (PCONTEXT)(uintptr_t)0x1234, "flags 0x%lx: *Context written on failure (%p)", (unsigned long)flags[i], (void *)c);
    }
}

TEST(init_size_query_null_buffer_nonzero_len) {
    need_core_api();
    DWORD want = init_len(XA_ALL);
    REQUIRE(want);
    DWORD len = 0x100000; PCONTEXT c = NULL;
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.InitializeContext(NULL, XA_ALL, &c, &len);
    CHECK(!ok);
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECK_EQ(len, want);
}

TEST(init_size_query_xstate) {
    need_xstate_ctx();
    DWORD64 en = enabled();
    const t_cpu_info *c = cpu();
    DWORD plain = init_len(XA_ALL), xs = init_len(XA_ALL | XA_XSTATE);
    t_info("len_plain", "%lu", plain); t_info("len_xstate", "%lu", xs);
    REQUIRE(plain && xs);
    CHECKF(plain >= sizeof(CONTEXT) + sizeof(X_CTX_EX), "plain length %lu must hold CONTEXT + CONTEXT_EX", plain);
    CHECKF(xs > plain, "XSTATE length %lu must exceed plain %lu", xs, plain);
    /* lower bound: CONTEXT + CONTEXT_EX + XSAVE header + every enabled component >= 2 */
    uint32_t need = (uint32_t)sizeof(CONTEXT) + (uint32_t)sizeof(X_CTX_EX) + 64;
    for (int i = 2; i < 63; i++) if ((en >> i & 1) && c->comp[i].valid) need += c->comp[i].size;
    CHECKF(xs >= need, "XSTATE length %lu < minimum %u for enabled 0x%llx", xs, need, (unsigned long long)en);
    if (en & T_XS_AVX) CHECKF(xs - plain >= 64 + 256, "AVX enabled but XSTATE adds only %lu bytes", xs - plain);
}

TEST(init_size_short_by_one) {
    need_core_api();
    DWORD fl[] = { XA_ALL, XA_ALL | XA_XSTATE };
    for (size_t k = 0; k < T_COUNTOF(fl); k++) {
        DWORD want = init_len(fl[k]);
        REQUIRE(want);
        BYTE *b = (BYTE *)malloc(want + 64);
        REQUIRE(b);
        memset(b, 0x5A, want + 64);
        DWORD len = want - 1; PCONTEXT c = NULL;
        SetLastError(X_ERR_SENTINEL);
        BOOL ok = X.InitializeContext(b, fl[k], &c, &len);
        CHECKF(!ok, "flags 0x%lx: buffer one byte short must fail", (unsigned long)fl[k]);
        CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
        CHECK_EQ(len, want);
        int touched = 0;
        for (DWORD i = 0; i < want + 64; i++) if (b[i] != 0x5A) { touched = 1; break; }
        CHECKF(!touched, "flags 0x%lx: buffer written although the call failed", (unsigned long)fl[k]);
        free(b);
    }
}

TEST(init_size_zero_len_with_buffer) {
    need_core_api();
    BYTE b[64];
    memset(b, 0x11, sizeof b);
    DWORD len = 0; PCONTEXT c = NULL;
    SetLastError(X_ERR_SENTINEL);
    CHECK(!X.InitializeContext(b, XA_CONTROL, &c, &len));
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECK_EQ(len, init_len(XA_CONTROL));
    CHECK(c == NULL);
}

/* Exact-size buffer ending at a guard page, at every alignment mod 64: an overrun crashes the case. */
static void init_exact_guarded(DWORD flags) {
    DWORD want = init_len(flags);
    REQUIRE(want);
    for (unsigned off = 0; off < 64; off++) {
        void *base; BYTE *p = (BYTE *)t_guarded_alloc(want + off, &base);
        REQUIRE(p);
        BYTE *b = p + off;
        memset(p, 0xC3, want + off);
        DWORD len = want; PCONTEXT c = NULL;
        SetLastError(X_ERR_SENTINEL);
        BOOL ok = X.InitializeContext(b, flags, &c, &len);
        CHECKF(ok, "offset %u: InitializeContext failed (error %lu)", off, GetLastError());
        if (ok) {
            CHECKF(len == want, "offset %u: length changed %lu -> %lu", off, want, len);
            CHECKF((BYTE *)c >= b && (BYTE *)c + sizeof(CONTEXT) + sizeof(X_CTX_EX) <= b + want,
                   "offset %u: context %p outside buffer [%p,+%lu)", off, (void *)c, (void *)b, want);
            CHECKF(((uintptr_t)c % XA_CTX_ALIGN) == 0, "offset %u: context %p not %u-aligned", off, (void *)c, XA_CTX_ALIGN);
            CHECKF((BYTE *)c - b < (ptrdiff_t)XA_CTX_ALIGN, "offset %u: context placed %d bytes into buffer (more than alignment)",
                   off, (int)((BYTE *)c - b));
            int pre = 0;
            for (BYTE *q = p; q < (BYTE *)c; q++) if (*q != 0xC3) pre = 1;
            CHECKF(!pre, "offset %u: bytes before the aligned context were written", off);
            CHECKF(c->ContextFlags == flags, "offset %u: ContextFlags 0x%lx want 0x%lx", off,
                   (unsigned long)c->ContextFlags, (unsigned long)flags);
            if ((flags & XA_XSTATE) == XA_XSTATE) {
                X_CTX_EX *e = xex(c);
                BYTE *ch = (BYTE *)e + e->XState.Offset;
                CHECKF(((uintptr_t)ch & 63) == 0, "offset %u: XSTATE chunk %p not 64-byte aligned", off, (void *)ch);
                CHECKF(ch >= (BYTE *)(e + 1) && ch + e->XState.Length <= b + want,
                       "offset %u: XSTATE chunk [%p,+%lu) outside buffer end %p", off, (void *)ch, e->XState.Length, (void *)(b + want));
            }
        }
        t_guarded_free(base);
    }
}
TEST(init_exact_size_guarded_plain) { need_core_api(); init_exact_guarded(XA_ALL); }
TEST(init_exact_size_guarded_xstate) { need_xstate_ctx(); init_exact_guarded(XA_ALL | XA_XSTATE); }
TEST(init_exact_size_guarded_xstate_only) { need_xstate_ctx(); init_exact_guarded(XA_XSTATE); }

TEST(init_larger_buffer_len_reported) {
    need_core_api();
    DWORD want = init_len(XA_XSTATE);
    REQUIRE(want);
    BYTE *b = (BYTE *)malloc(want + 4096);
    REQUIRE(b);
    DWORD len = want + 4096; PCONTEXT c = NULL;
    BOOL ok = X.InitializeContext(b, XA_XSTATE, &c, &len);
    CHECK(ok);
    CHECKF(len == want, "on success ContextLength = %lu, want the required length %lu", len, want);
    free(b);
}

TEST(init_length_independent_of_low_flags) {
    need_core_api();
    DWORD base_plain = init_len(XA_ARCH | 1), base_xs = init_len(XA_XSTATE | 1);
    REQUIRE(base_plain && base_xs);
#ifdef T_X64
    const DWORD lowmask = 0x1F;
#else
    const DWORD lowmask = 0x3F;
#endif
    int n = 0;
    for (DWORD low = 0; low <= lowmask; low++) {
        if (low & ~lowmask) continue;
        DWORD a = init_len(XA_ARCH | low), b = init_len(XA_XSTATE | low);
        CHECKF(a == base_plain, "flags 0x%lx: length %lu != %lu", (unsigned long)(XA_ARCH | low), a, base_plain);
        CHECKF(b == base_xs, "flags 0x%lx: length %lu != %lu", (unsigned long)(XA_XSTATE | low), b, base_xs);
        n++;
    }
    t_log("%d flag combinations", n);
}

static void check_invalid_flags(DWORD flags, DWORD want_err) {
    DWORD len = 0x777; PCONTEXT c = NULL;
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.InitializeContext(NULL, flags, &c, &len);
    DWORD err = GetLastError();
    CHECKF(!ok, "flags 0x%lx must be rejected", (unsigned long)flags);
    CHECKF(err == want_err, "flags 0x%lx: error %lu want %lu", (unsigned long)flags, err, want_err);
    CHECKF(len == 0x777, "flags 0x%lx: length written (%lu) although the flags are invalid", (unsigned long)flags, len);
    /* also with a real buffer: must not write it */
    BYTE b[0x800]; memset(b, 0x3C, sizeof b);
    len = sizeof b;
    SetLastError(X_ERR_SENTINEL);
    ok = X.InitializeContext(b, flags, &c, &len);
    CHECKF(!ok, "flags 0x%lx (buffer given) must be rejected", (unsigned long)flags);
    CHECKF(GetLastError() == want_err, "flags 0x%lx (buffer given): error %lu want %lu", (unsigned long)flags, GetLastError(), want_err);
    int touched = 0;
    for (size_t i = 0; i < sizeof b; i++) if (b[i] != 0x3C) touched = 1;
    CHECKF(!touched, "flags 0x%lx: buffer written although the flags are invalid", (unsigned long)flags);
}
TEST(init_invalid_flags_zero) { need_core_api(); check_invalid_flags(0, ERROR_INVALID_PARAMETER); }
TEST(init_invalid_flags_no_arch_bit) {
    need_core_api();
    check_invalid_flags(0x1F, ERROR_INVALID_PARAMETER);
    check_invalid_flags(0x40, ERROR_INVALID_PARAMETER);
}
TEST(init_invalid_flags_both_arch_bits) { need_core_api(); check_invalid_flags(0x00110001u, ERROR_INVALID_PARAMETER); }
TEST(init_invalid_flags_reserved_low_bit) {
    need_core_api();
    check_invalid_flags(XA_ARCH | XA_BADLOW, ERROR_INVALID_PARAMETER);
    check_invalid_flags(XA_ALL | XA_BADLOW, ERROR_INVALID_PARAMETER);
    check_invalid_flags(XA_ARCH | 0x100, ERROR_INVALID_PARAMETER);
    check_invalid_flags(XA_ARCH | 0x8000, ERROR_INVALID_PARAMETER);
}
TEST(init_invalid_flags_reserved_high_bit) {
    need_core_api();
    check_invalid_flags(XA_ALL | 0x20000000u, ERROR_INVALID_PARAMETER);
    check_invalid_flags(XA_ALL | 0x01000000u, ERROR_INVALID_PARAMETER);
    check_invalid_flags(XA_ALL | 0x00400000u, ERROR_INVALID_PARAMETER);
}
TEST(init_valid_flags_exception_bits) {
    need_core_api();
    /* CONTEXT_EXCEPTION_ACTIVE / SERVICE_ACTIVE / EXCEPTION_REQUEST / EXCEPTION_REPORTING are accepted */
    DWORD hi[] = { 0x08000000u, 0x10000000u, 0x40000000u, 0x80000000u, 0xD8000000u };
    DWORD base = init_len(XA_ALL);
    REQUIRE(base);
    for (size_t i = 0; i < T_COUNTOF(hi); i++) {
        DWORD f = XA_ALL | hi[i];
        DWORD l = init_len(f);
        CHECKF(l == base, "flags 0x%lx: length %lu want %lu", (unsigned long)f, l, base);
        xctx x;
        int ok = xctx_new(&x, f);
        CHECKF(ok, "InitializeContext(0x%lx) failed (%lu)", (unsigned long)f, GetLastError());
        if (ok) { CHECKF(x.ctx->ContextFlags == f, "ContextFlags 0x%lx want 0x%lx", (unsigned long)x.ctx->ContextFlags, (unsigned long)f); xctx_free(&x); }
    }
}

TEST(init_invalid_flags_validated_before_size) {
    need_core_api();
    /* RtlGetExtendedContextLength2 validates before the size check: invalid flags never yield 122. */
    DWORD bad[] = { 0, XA_ARCH | XA_BADLOW, 0x00110001u, XA_XSTATE | 0x20000000u };
    for (size_t i = 0; i < T_COUNTOF(bad); i++) {
        DWORD len = 0; PCONTEXT c = NULL;
        SetLastError(X_ERR_SENTINEL);
        X.InitializeContext(NULL, bad[i], &c, &len);
        CHECKF(GetLastError() != ERROR_INSUFFICIENT_BUFFER, "invalid flags 0x%lx reported as a size query", (unsigned long)bad[i]);
        CHECKF(len == 0, "invalid flags 0x%lx produced a length (%lu)", (unsigned long)bad[i], len);
    }
}

TEST(init_cross_arch_flags_info) {
    need_core_api();
    DWORD len = 0; PCONTEXT c = NULL;
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.InitializeContext(NULL, XA_OTHER_ARCH | 1, &c, &len);
    t_info("cross_arch_query", "ok=%d err=%lu len=%lu", ok, GetLastError(), len);
    t_note("other-architecture context flags 0x%lx: error %lu, length %lu (informational: not a documented contract)",
           (unsigned long)(XA_OTHER_ARCH | 1), GetLastError(), len);
}

TEST(init_context_all_and_full) {
    need_core_api();
    DWORD fl[] = { XA_ALL, XA_ALL | XA_XSTATE, (DWORD)CONTEXT_FULL, (DWORD)CONTEXT_FULL | XA_XSTATE, XA_CONTROL, XA_INTEGER, XA_FP_ALL };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        xctx x;
        int ok = xctx_new(&x, fl[i]);
        CHECKF(ok, "InitializeContext(0x%lx) failed (%lu)", (unsigned long)fl[i], GetLastError());
        if (!ok) continue;
        /* kernelbase!InitializeContext2 strips CONTEXT_XSTATE (0x40) when no XSTATE is enabled at all */
        DWORD want = os_xstate() ? fl[i] : (fl[i] & ~0x40u);
        CHECKF(x.ctx->ContextFlags == want, "ContextFlags 0x%lx want 0x%lx", (unsigned long)x.ctx->ContextFlags, (unsigned long)want);
        xctx_free(&x);
    }
}

/* kernelbase!InitializeContext2: RtlGetEnabledExtendedFeatures(~0)==0 -> flags &= ~0x40 before sizing.
 * ntdll!RtlpValidateContextFlags: XSTATE flags with XState.EnabledFeatures==0 -> STATUS_NOT_SUPPORTED. */
TEST(init_xstate_stripped_without_os_xstate) {
    need_core_api();
    if (os_xstate()) SKIP("OS enables XSAVE-managed state: the CONTEXT_XSTATE stripping path does not apply");
    CHECK_EQ(enabled(), 3);
    DWORD plain = init_len(XA_ALL), xs = init_len(XA_ALL | XA_XSTATE);
    CHECKF(plain && xs == plain, "XSTATE length %lu != plain %lu although XSTATE is unsupported", xs, plain);
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    CHECKF(x.ctx->ContextFlags == XA_ALL, "ContextFlags 0x%lx want 0x%lx (XSTATE bit stripped)", (unsigned long)x.ctx->ContextFlags, (unsigned long)XA_ALL);
    CHECK_EQ(xex(x.ctx)->XState.Length, 0);
    CHECK(X.LocateXStateFeature(x.ctx, XSTATE_AVX, NULL) == NULL);
    if (X.RtlGetExtendedContextLength) {
        ULONG l = 0x777;
        CHECK_EQ((ULONG)X.RtlGetExtendedContextLength(XA_ALL | XA_XSTATE, &l), (ULONG)X_STATUS_NOT_SUPPORTED);
        CHECK_EQ(l, 0x777);
    }
    xctx s; XCTX_REQ(&s, XA_ALL);
    SetLastError(X_ERR_SENTINEL);
    CHECK(!X.CopyContext(x.ctx, XA_ALL | XA_XSTATE, s.ctx));
    CHECK_EQ(GetLastError(), ERROR_NOT_SUPPORTED);
    xctx_free(&s); xctx_free(&x);
}

TEST(init_reinit_clears_xstate_header) {
    need_xstate_ctx();
    xctx x;
    XCTX_REQ(&x, XA_XSTATE);
    BYTE *ch = xchunk(x.ctx);
    uint64_t comp0 = xs_comp(x.ctx);
    memset(ch, 0xEE, 64);
    PCONTEXT c2 = NULL; DWORD len = x.len;
    REQUIRE(X.InitializeContext(x.buf, XA_XSTATE, &c2, &len));
    CHECK(c2 == x.ctx);
    CHECK(xchunk(c2) == ch);
    CHECK_EQ(xs_bv(c2), 0);
    CHECK_EQ(xs_comp(c2), comp0);
    BYTE zero[48] = { 0 };
    CHECK_MEM(ch + 16, zero, 48);
    xctx_free(&x);
}

TEST(init_rtl_length_matches) {
    need_xstate_ctx();
    need_rtl((void *)X.RtlGetExtendedContextLength, "RtlGetExtendedContextLength");
    DWORD fl[] = { XA_CONTROL, XA_ALL, XA_XSTATE, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        ULONG rl = 0;
        LONG st = X.RtlGetExtendedContextLength(fl[i], &rl);
        CHECKF(st == X_STATUS_SUCCESS, "RtlGetExtendedContextLength(0x%lx) = 0x%lx", (unsigned long)fl[i], (unsigned long)st);
        CHECKF(rl == init_len(fl[i]), "flags 0x%lx: Rtl length %lu, InitializeContext %lu", (unsigned long)fl[i], rl, init_len(fl[i]));
    }
}

TEST(rtl_length_invalid_flags_status) {
    need_rtl((void *)X.RtlGetExtendedContextLength, "RtlGetExtendedContextLength");
    DWORD bad[] = { 0, 0x00110001u, XA_ARCH | XA_BADLOW, XA_ALL | 0x20000000u };
    for (size_t i = 0; i < T_COUNTOF(bad); i++) {
        ULONG rl = 0x777;
        LONG st = X.RtlGetExtendedContextLength(bad[i], &rl);
        CHECKF(st == X_STATUS_INVALID_PARAMETER, "flags 0x%lx: status 0x%lx want 0xC000000D", (unsigned long)bad[i], (unsigned long)st);
        CHECKF(rl == 0x777, "flags 0x%lx: length written on failure", (unsigned long)bad[i]);
    }
}

/* ======================= D. CONTEXT_EX layout ============================= */
static void check_ctxex(const xctx *x, DWORD flags) {
    X_CTX_EX *e = xex(x->ctx);
    CHECKF(in_buf(x, e, sizeof *e), "CONTEXT_EX %p outside buffer", (void *)e);
    CHECKF(e->Legacy.Offset == -(LONG)sizeof(CONTEXT), "Legacy.Offset %ld want %ld", (long)e->Legacy.Offset, -(long)sizeof(CONTEXT));
    CHECKF(e->All.Offset == e->Legacy.Offset, "All.Offset %ld != Legacy.Offset %ld", (long)e->All.Offset, (long)e->Legacy.Offset);
#ifdef T_X86
    DWORD wantleg = (flags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS ? 0x2CCu : 0xCCu;
#else
    DWORD wantleg = XA_CTX_SIZE;
#endif
    CHECKF(e->Legacy.Length == wantleg, "Legacy.Length 0x%lx want 0x%lx (flags 0x%lx)", e->Legacy.Length, wantleg, (unsigned long)flags);
    if ((flags & XA_XSTATE) == XA_XSTATE) {
        CHECKF(e->XState.Offset >= (LONG)sizeof(X_CTX_EX), "XState.Offset %ld overlaps CONTEXT_EX", (long)e->XState.Offset);
        CHECKF(e->XState.Length >= 64, "XState.Length %lu < XSAVE header", e->XState.Length);
        CHECKF((((uintptr_t)e + e->XState.Offset) & 63) == 0, "XState chunk not 64-aligned");
        CHECKF(e->All.Length == (DWORD)(e->XState.Offset - e->All.Offset) + e->XState.Length,
               "All.Length 0x%lx != XState.Offset-All.Offset+XState.Length 0x%lx", e->All.Length,
               (unsigned long)((e->XState.Offset - e->All.Offset) + e->XState.Length));
        CHECKF(in_buf(x, (BYTE *)e + e->XState.Offset, e->XState.Length), "XState chunk outside buffer");
        CHECKF((BYTE *)e + e->All.Offset + e->All.Length <= x->buf + x->len, "All chunk ends past buffer");
    } else {
        CHECKF(e->XState.Length == 0, "XState.Length %lu for a non-XSTATE context", e->XState.Length);
        CHECKF(e->All.Length == (DWORD)(-e->All.Offset) + (DWORD)sizeof(X_CTX_EX),
               "All.Length 0x%lx want 0x%lx", e->All.Length, (unsigned long)(-e->All.Offset + sizeof(X_CTX_EX)));
    }
}
TEST(ctxex_layout_plain) {
    need_core_api();
    DWORD fl[] = { XA_CONTROL, XA_ALL, XA_FPFLAG };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) { xctx x; XCTX_REQ(&x, fl[i]); check_ctxex(&x, fl[i]); xctx_free(&x); }
}
TEST(ctxex_layout_xstate) {
    need_xstate_ctx();
    DWORD fl[] = { XA_XSTATE, XA_ALL | XA_XSTATE, XA_CONTROL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) { xctx x; XCTX_REQ(&x, fl[i]); check_ctxex(&x, fl[i]); xctx_free(&x); }
}

TEST(ctxex_xstate_header_initial) {
    need_xstate_ctx();
    DWORD64 en = enabled();
    xctx x;
    XCTX_REQ(&x, XA_XSTATE);
    uint64_t bv = xs_bv(x.ctx), comp = xs_comp(x.ctx);
    t_info("xcomp_bv", "0x%llx", (unsigned long long)comp);
    CHECK_EQ(bv, 0);
    if (comp) {
        CHECKF(comp & X_BIT63, "nonzero XCOMP_BV 0x%llx without bit 63", (unsigned long long)comp);
        /* the OS may include XCR0 user components the API does not report (PKRU/CET are never exercised on the oracle):
         * XCOMP_BV must cover every reported enabled component and stay within enabled | XCR0 user bits */
        uint64_t xu = cpu()->osxsave ? cpu()->xcr0 : 3, cb = comp & ~X_BIT63 & ~3ull;
        CHECKF((en & ~3ull & ~cb) == 0, "XCOMP_BV 0x%llx does not cover enabled 0x%llx",
               (unsigned long long)comp, (unsigned long long)en);
        CHECKF((cb & ~(en | xu)) == 0, "XCOMP_BV 0x%llx has components outside enabled 0x%llx | XCR0 0x%llx",
               (unsigned long long)comp, (unsigned long long)en, (unsigned long long)xu);
        if (cb != (en & ~3ull)) t_note("XCOMP_BV 0x%llx differs from the API mask 0x%llx", (unsigned long long)comp, (unsigned long long)en);
        t_note("context uses the compacted XSAVE format (XCOMP_BV=0x%llx)", (unsigned long long)comp);
    } else {
        t_note("context uses the standard XSAVE format");
    }
    BYTE zero[48] = { 0 };
    CHECK_MEM(xchunk(x.ctx) + 16, zero, 48);
    xctx_free(&x);
}

TEST(ctxex_xstate_length_matches_cpuid) {
    need_xstate_ctx();
    DWORD64 en = enabled();
    const t_cpu_info *c = cpu();
    if (!c->osxsave) SKIP("OSXSAVE clear");
    xctx x;
    XCTX_REQ(&x, XA_XSTATE);
    X_CTX_EX *e = xex(x.ctx);
    uint64_t comp = xs_comp(x.ctx);
    uint32_t want;
    t_info("xstate_chunk_length", "%lu", e->XState.Length);
    if (comp & X_BIT63) {
        /* compacted: RtlpGetEntireXStateAreaLength sums the XCOMP_BV components exactly (SDM 13.4.3) */
        want = 576;
        for (int i = 2; i < 63; i++) if (comp >> i & 1) { if (c->comp[i].align64) want = (want + 63) & ~63u; want += c->comp[i].size; }
        CHECKF(e->XState.Length == want - 512, "XState.Length %lu, want %u - 512 for compacted format", e->XState.Length, want);
    } else {
        /* standard: ntdll uses XSTATE_CONFIGURATION.Size, which the kernel may take from CPUID.(0Dh,0).EBX
         * (covers every XCR0 component, e.g. PKRU that the API does not report). Contract: it holds every
         * enabled component and is no larger than CPUID.(0Dh,0).ECX. */
        want = xsave_std_size(en & ~3ull);
        CHECKF(e->XState.Length + 512 >= want, "XState.Length %lu cannot hold the enabled components (need %u - 512)", e->XState.Length, want);
        if (c->xsave_size_max)
            CHECKF(e->XState.Length + 512 <= c->xsave_size_max, "XState.Length %lu exceeds CPUID.(0Dh,0).ECX %u - 512",
                   e->XState.Length, c->xsave_size_max);
        if (e->XState.Length + 512 != want)
            t_note("standard-format area %lu+512 != end of last enabled component %u (CPUID EBX %u)", e->XState.Length, want, c->xsave_size_enabled);
    }
    xctx_free(&x);
}

#ifdef T_X86
TEST(ctxex_x86_legacy_length_extended_regs) {
    need_xstate_ctx();
    xctx a, b;
    XCTX_REQ(&a, XA_CONTROL | XA_XSTATE);
    XCTX_REQ(&b, XA_CONTROL | CONTEXT_EXTENDED_REGISTERS | XA_XSTATE);
    CHECK_EQ(xex(a.ctx)->Legacy.Length, 0xCC);
    CHECK_EQ(xex(b.ctx)->Legacy.Length, 0x2CC);
    CHECK_EQ(xex(a.ctx)->Legacy.Offset, -0x2CC);
    xctx_free(&a); xctx_free(&b);
}
#else
TEST(ctxex_x86_legacy_length_extended_regs) { SKIP("i386 CONTEXT layout: x86 build only"); }
#endif

TEST(rtl_locate_legacy_context) {
    need_xstate_ctx();
    need_rtl((void *)X.RtlLocateLegacyContext, "RtlLocateLegacyContext");
    DWORD fl[] = { XA_ALL, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        xctx x; XCTX_REQ(&x, fl[i]);
        ULONG l = 0;
        void *p = X.RtlLocateLegacyContext(xex(x.ctx), &l);
        CHECKF(p == (void *)x.ctx, "flags 0x%lx: RtlLocateLegacyContext %p want %p", (unsigned long)fl[i], p, (void *)x.ctx);
        CHECK_EQ(l, xex(x.ctx)->Legacy.Length);
        CHECK(X.RtlLocateLegacyContext(xex(x.ctx), NULL) == (void *)x.ctx);
        xctx_free(&x);
    }
}

TEST(rtl_initialize_extended_context) {
    need_xstate_ctx();
    need_rtl((void *)X.RtlInitializeExtendedContext, "RtlInitializeExtendedContext");
    need_rtl((void *)X.RtlLocateLegacyContext, "RtlLocateLegacyContext");
    DWORD fl[] = { XA_ALL, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        DWORD len = init_len(fl[i]);
        REQUIRE(len);
        for (unsigned off = 0; off < 64; off += 7) {
            void *base; BYTE *p = (BYTE *)t_guarded_alloc(len + off, &base);
            REQUIRE(p);
            BYTE *b = p + off;
            X_CTX_EX *ex = NULL;
            LONG st = X.RtlInitializeExtendedContext(b, fl[i], &ex);
            CHECKF(st == X_STATUS_SUCCESS, "flags 0x%lx off %u: status 0x%lx", (unsigned long)fl[i], off, (unsigned long)st);
            if (st == X_STATUS_SUCCESS && ex) {
                CONTEXT *c = (CONTEXT *)X.RtlLocateLegacyContext(ex, NULL);
                CHECKF(c && ((uintptr_t)c % XA_CTX_ALIGN) == 0 && (BYTE *)c - b < (ptrdiff_t)XA_CTX_ALIGN, "legacy context %p for buffer %p", (void *)c, (void *)b);
                CHECKF((BYTE *)ex == (BYTE *)c + sizeof(CONTEXT), "CONTEXT_EX %p not right after CONTEXT %p", (void *)ex, (void *)c);
                if (c) CHECK_EQ(c->ContextFlags, fl[i]);
                /* same placement as kernel32!InitializeContext */
                PCONTEXT k = NULL; DWORD kl = len;
                CHECK(X.InitializeContext(b, fl[i], &k, &kl));
                CHECK(k == c);
            }
            t_guarded_free(base);
        }
    }
}

TEST(rtl_initialize_invalid_flags_status) {
    need_rtl((void *)X.RtlInitializeExtendedContext, "RtlInitializeExtendedContext");
    BYTE b[0x1000]; memset(b, 0x6B, sizeof b);
    DWORD bad[] = { 0, 0x00110001u, XA_ARCH | XA_BADLOW };
    for (size_t i = 0; i < T_COUNTOF(bad); i++) {
        X_CTX_EX *ex = (X_CTX_EX *)(uintptr_t)0x4321;
        LONG st = X.RtlInitializeExtendedContext(b, bad[i], &ex);
        CHECKF(st == X_STATUS_INVALID_PARAMETER, "flags 0x%lx: status 0x%lx", (unsigned long)bad[i], (unsigned long)st);
        CHECKF(ex == (X_CTX_EX *)(uintptr_t)0x4321, "flags 0x%lx: CONTEXT_EX pointer written on failure", (unsigned long)bad[i]);
    }
    int touched = 0;
    for (size_t i = 0; i < sizeof b; i++) if (b[i] != 0x6B) touched = 1;
    CHECK(!touched);
}

/* ======================= E. InitializeContext2 ============================ */
static void need_init2(void) {
    need_xstate_ctx();
    if (!X.InitializeContext2) SKIP("kernel32!InitializeContext2 not exported (Windows 10 2004+)");
}

TEST(init2_full_mask_matches_init) {
    need_init2();
    DWORD64 en = enabled();
    DWORD fl[] = { XA_ALL, XA_XSTATE, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        DWORD a = init_len(fl[i]), b = init2_len(fl[i], en);
        CHECKF(a == b, "flags 0x%lx: InitializeContext %lu, InitializeContext2(enabled) %lu", (unsigned long)fl[i], a, b);
        xctx x1, x2;
        REQUIRE(xctx_new2(&x1, fl[i], 0, 0, 0));
        REQUIRE(xctx_new2(&x2, fl[i], en, 1, 0));
        CHECK_EQ(x1.ctx->ContextFlags, x2.ctx->ContextFlags);
        CHECK_EQ(xex(x1.ctx)->XState.Length, xex(x2.ctx)->XState.Length);
        if ((fl[i] & XA_XSTATE) == XA_XSTATE) CHECK_EQ(xs_comp(x1.ctx), xs_comp(x2.ctx));
        xctx_free(&x1); xctx_free(&x2);
    }
}

TEST(init2_size_query_null_buffer) {
    need_init2();
    DWORD len = 0; PCONTEXT c = NULL;
    SetLastError(X_ERR_SENTINEL);
    CHECK(!X.InitializeContext2(NULL, XA_ALL | XA_XSTATE, &c, &len, 3 | T_XS_AVX));
    CHECK_EQ(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
    CHECK(len > sizeof(CONTEXT));
    SetLastError(X_ERR_SENTINEL);
    len = 0x55;
    CHECK(!X.InitializeContext2(NULL, 0, &c, &len, 3));
    CHECK_EQ(GetLastError(), ERROR_INVALID_PARAMETER);
    CHECK_EQ(len, 0x55);
}

TEST(init2_small_mask) {
    need_init2();
    need_avx_api();
    DWORD full = init2_len(XA_XSTATE, enabled()), leg = init2_len(XA_XSTATE, 3), avx = init2_len(XA_XSTATE, 3 | T_XS_AVX);
    t_info("len2_legacy", "%lu", leg); t_info("len2_avx", "%lu", avx); t_info("len2_full", "%lu", full);
    REQUIRE(full && leg && avx);
    CHECKF(leg <= avx && avx <= full, "lengths not monotonic in the mask: legacy %lu avx %lu full %lu", leg, avx, full);
    xctx x;
    REQUIRE(xctx_new2(&x, XA_XSTATE, 3, 1, 0));
    uint64_t comp = xs_comp(x.ctx);
    if (comp & X_BIT63) {
        CHECKF(avx - leg >= 256, "compacted: AVX adds only %lu bytes", avx - leg);
        CHECKF((comp & ~X_BIT63 & ~3ull) == 0, "XCOMP_BV 0x%llx for mask 3", (unsigned long long)comp);
        CHECKF(xex(x.ctx)->XState.Length == 64, "compacted legacy-only chunk length %lu want 64", xex(x.ctx)->XState.Length);
    } else {
        t_note("standard format: compaction mask does not change the length (legacy %lu full %lu)", leg, full);
        CHECK_EQ(leg, full);
    }
    xctx_free(&x);
}

TEST(init2_locate_feature_outside_mask) {
    need_init2();
    need_avx_api();
    xctx x;
    REQUIRE(xctx_new2(&x, XA_XSTATE, 3, 1, 0));
    DWORD l = 0x999;
    void *p = X.LocateXStateFeature(x.ctx, XSTATE_AVX, &l);
    if (xs_comp(x.ctx) & X_BIT63) {
        CHECKF(p == NULL, "compacted context without AVX in XCOMP_BV: LocateXStateFeature(AVX) = %p", p);
        len_untouched_note(l, XSTATE_AVX);
    } else {
        CHECKF(p != NULL && in_buf(&x, p, 256), "standard-format context: AVX location %p not inside buffer", p);
    }
    /* legacy features never depend on the compaction mask */
    CHECK(X.LocateXStateFeature(x.ctx, 0, NULL) == (BYTE *)x.ctx + XA_LEG0_OFF);
    CHECK(X.LocateXStateFeature(x.ctx, 1, NULL) == (BYTE *)x.ctx + XA_LEG1_OFF);
    xctx_free(&x);
}

TEST(init2_mask_unsupported_bits) {
    need_init2();
    DWORD64 en = enabled();
    uint64_t junk = (1ull << 62) | (1ull << 40) | (1ull << 8);
    junk &= ~en;
    DWORD a = init2_len(XA_XSTATE, en), b = init2_len(XA_XSTATE, en | junk);
    CHECKF(a == b, "unsupported compaction-mask bits changed the length: %lu -> %lu", a, b);
    xctx x;
    REQUIRE(xctx_new2(&x, XA_XSTATE, en | junk, 1, 0));
    uint64_t comp = xs_comp(x.ctx);
    CHECKF((comp & junk) == 0, "XCOMP_BV 0x%llx contains unsupported bits 0x%llx", (unsigned long long)comp, (unsigned long long)junk);
    xctx_free(&x);
}

TEST(init2_rtl_length2_matches) {
    need_init2();
    need_rtl((void *)X.RtlGetExtendedContextLength2, "RtlGetExtendedContextLength2");
    DWORD64 en = enabled();
    DWORD64 masks[] = { 3, 3 | T_XS_AVX, en };
    for (size_t i = 0; i < T_COUNTOF(masks); i++) {
        ULONG rl = 0;
        LONG st = X.RtlGetExtendedContextLength2(XA_ALL | XA_XSTATE, &rl, masks[i]);
        CHECK_EQ(st, 0);
        CHECKF(rl == init2_len(XA_ALL | XA_XSTATE, masks[i]), "mask 0x%llx: Rtl %lu kernel32 %lu",
               (unsigned long long)masks[i], rl, init2_len(XA_ALL | XA_XSTATE, masks[i]));
    }
}

TEST(init2_amx_tiledata) {
    need_init2();
    DWORD64 en = enabled();
    if ((en & T_XS_AMX_STATE) != T_XS_AMX_STATE) SKIP("AMX state not enabled (0x%llx)", (unsigned long long)en);
    const t_cpu_info *c = cpu();
    DWORD a = init2_len(XA_XSTATE, en & ~(uint64_t)T_XS_TILEDATA), b = init2_len(XA_XSTATE, en);
    t_info("len_without_tiledata", "%lu", a); t_info("len_with_tiledata", "%lu", b);
    CHECKF(b >= a + c->comp[18].size || b == a, "tile data length delta %lu unexpected (comp size %u)", b - a, c->comp[18].size);
    xctx x;
    REQUIRE(xctx_new2(&x, XA_XSTATE, en, 1, 0));
    DWORD l = 0;
    void *p = X.LocateXStateFeature(x.ctx, XSTATE_AMX_TILE_DATA, &l);
    CHECKF(p && l == c->comp[18].size && in_buf(&x, p, l), "AMX tile data location %p len %lu", p, l);
    xctx_free(&x);
}

/* ======================= F. LocateXStateFeature =========================== */
TEST(locate_legacy_fp) {
    need_core_api();
    DWORD fl[] = { XA_ALL, XA_CONTROL, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        xctx x; XCTX_REQ(&x, fl[i]);
        DWORD l = 0;
        void *p = X.LocateXStateFeature(x.ctx, XSTATE_LEGACY_FLOATING_POINT, &l);
        CHECKF(p == (BYTE *)x.ctx + XA_LEG0_OFF, "flags 0x%lx: feature 0 at %p want ctx+0x%x", (unsigned long)fl[i], p, XA_LEG0_OFF);
        CHECKF(l == XA_LEG0_LEN, "flags 0x%lx: feature 0 length 0x%lx want 0x%x", (unsigned long)fl[i], l, XA_LEG0_LEN);
        xctx_free(&x);
    }
}

TEST(locate_legacy_sse) {
    need_core_api();
    DWORD fl[] = { XA_ALL, XA_CONTROL, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        xctx x; XCTX_REQ(&x, fl[i]);
        DWORD l = 0;
        void *p = X.LocateXStateFeature(x.ctx, XSTATE_LEGACY_SSE, &l);
        CHECKF(p == (BYTE *)x.ctx + XA_LEG1_OFF, "flags 0x%lx: feature 1 at %p want ctx+0x%x", (unsigned long)fl[i], p, XA_LEG1_OFF);
        CHECKF(l == XA_LEG1_LEN, "flags 0x%lx: feature 1 length 0x%lx want 0x%x (%d XMM registers)", (unsigned long)fl[i], l, XA_LEG1_LEN, XA_NXMM);
#ifdef T_X64
        CHECK(p == (void *)&x.ctx->Xmm0);
        CHECK(p == (void *)&x.ctx->FltSave.XmmRegisters[0]);
#else
        CHECK(p == (void *)&x.ctx->ExtendedRegisters[160]);
#endif
        xctx_free(&x);
    }
}

TEST(locate_legacy_on_plain_context_struct) {
    need_core_api();
    /* A plain CONTEXT (not from InitializeContext) works for the legacy features. */
    CONTEXT *c = (CONTEXT *)_aligned_malloc(sizeof(CONTEXT) + 64, 64);
    REQUIRE(c);
    memset(c, 0, sizeof(CONTEXT));
    c->ContextFlags = XA_ALL;
    CHECK(X.LocateXStateFeature(c, 0, NULL) == (BYTE *)c + XA_LEG0_OFF);
    CHECK(X.LocateXStateFeature(c, 1, NULL) == (BYTE *)c + XA_LEG1_OFF);
    DWORD l = 0x999;
    CHECK(X.LocateXStateFeature(c, XSTATE_AVX, &l) == NULL);
    len_untouched_note(l, XSTATE_AVX);
    _aligned_free(c);
}

TEST(locate_avx_xstate) {
    need_avx_api();
    const t_cpu_info *c = cpu();
    DWORD fl[] = { XA_XSTATE, XA_ALL | XA_XSTATE };
    for (size_t i = 0; i < T_COUNTOF(fl); i++) {
        xctx x; XCTX_REQ(&x, fl[i]);
        DWORD l = 0;
        BYTE *p = (BYTE *)X.LocateXStateFeature(x.ctx, XSTATE_AVX, &l);
        REQUIREF(p != NULL, "flags 0x%lx: AVX is enabled but LocateXStateFeature(AVX) = NULL", (unsigned long)fl[i]);
        CHECKF(l == 256, "AVX length %lu want 256 (16 x 16-byte YMM upper halves)", l);
        CHECKF(c->comp[2].size == 0 || l == c->comp[2].size, "AVX length %lu != CPUID.(0Dh,2).EAX %u", l, c->comp[2].size);
        CHECKF(in_buf(&x, p, l), "AVX area [%p,+%lu) outside buffer [%p,+%lu)", (void *)p, l, (void *)x.buf, x.len);
        CHECKF(((uintptr_t)p & 63) == 0, "AVX area %p not 64-byte aligned (offset 576 of an aligned XSAVE image)", (void *)p);
        CHECKF(p == xchunk(x.ctx) + 64, "AVX area %p != XSTATE chunk %p + 64", (void *)p, (void *)xchunk(x.ctx));
        CHECKF(p >= (BYTE *)x.ctx + sizeof(CONTEXT) + sizeof(X_CTX_EX), "AVX area overlaps CONTEXT/CONTEXT_EX");
        xctx_free(&x);
    }
}

TEST(locate_avx_without_xstate_flag) {
    need_avx_api();
    xctx x; XCTX_REQ(&x, XA_ALL);
    DWORD l = 0x999;
    CHECK(X.LocateXStateFeature(x.ctx, XSTATE_AVX, &l) == NULL);
    len_untouched_note(l, XSTATE_AVX);
    xctx_free(&x);
    /* XSTATE-initialized but the XSTATE flag removed afterwards */
    XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    x.ctx->ContextFlags &= ~0x40u;
    CHECK(X.LocateXStateFeature(x.ctx, XSTATE_AVX, NULL) == NULL);
    xctx_free(&x);
}

TEST(locate_unsupported_features_null) {
    DWORD64 en = enabled();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    int n = 0;
    for (DWORD f = 2; f < 63; f++) {   /* 63 = XCOMP_BV compaction bit: own case */
        if (en >> f & 1) continue;
        DWORD l = 0x999;
        void *p = X.LocateXStateFeature(x.ctx, f, &l);
        CHECKF(p == NULL, "feature %lu (%s) not enabled but located at %p", f, xstate_comp_name((int)f), p);
        len_untouched_note(l, f);
        n++;
    }
    DWORD ids[] = { 64, 65, 127, 0x100, 0x7FFFFFFF, 0xFFFFFFFF };
    for (size_t i = 0; i < T_COUNTOF(ids); i++) {
        DWORD l = 0x999;
        CHECKF(X.LocateXStateFeature(x.ctx, ids[i], &l) == NULL, "feature id 0x%lx must be NULL", ids[i]);
        len_untouched_note(l, ids[i]);
    }
    t_log("%d absent features checked", n);
    xctx_free(&x);
}

TEST(locate_feature63_compaction_bit) {
    DWORD64 en = enabled();
    REQUIRE(!(en & X_BIT63));
    /* id 63 is the XCOMP_BV compaction-format bit, never a state component (SDM 13.4.3) */
    wine_known_gap("LocateXStateFeature(63) treats the XCOMP_BV compaction bit as a feature");
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    DWORD l = 0x999;
    void *p = X.LocateXStateFeature(x.ctx, 63, &l);
    CHECKF(p == NULL, "feature 63 located at %p (buffer [%p,+%lu))", p, (void *)x.buf, x.len);
    xctx_free(&x);
}

TEST(locate_enabled_features_offsets) {
    DWORD64 en = enabled();
    const t_cpu_info *c = cpu();
    if (!c->osxsave) SKIP("OSXSAVE clear");
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    BYTE *img = xchunk(x.ctx) - 512;   /* virtual XSAVE image base: chunk = header at byte 512 */
    uint64_t comp = xs_comp(x.ctx);
    int n = 0;
    for (int f = 2; f < 63; f++) {
        if (!(en >> f & 1)) continue;
        DWORD l = 0;
        BYTE *p = (BYTE *)X.LocateXStateFeature(x.ctx, (DWORD)f, &l);
        if (!(comp & X_BIT63) && c->comp[f].supervisor) {
            /* supervisor state has no standard-format offset */
            t_note("feature %d (%s) supervisor in standard format: %p", f, xstate_comp_name(f), (void *)p);
            continue;
        }
        if ((comp & X_BIT63) && !(comp >> f & 1)) {
            CHECKF(p == NULL, "feature %d not in XCOMP_BV 0x%llx but located", f, (unsigned long long)comp);
            continue;
        }
        REQUIREF(p != NULL, "enabled feature %d (%s) not located", f, xstate_comp_name(f));
        CHECKF(l == c->comp[f].size, "feature %d (%s): length %lu want CPUID size %u", f, xstate_comp_name(f), l, c->comp[f].size);
        uint32_t want = xs_expected_offset(comp, f);
        CHECKF((uint32_t)(p - img) == want, "feature %d (%s): XSAVE offset %u want %u (%s)", f, xstate_comp_name(f),
               (unsigned)(p - img), want, (comp & X_BIT63) ? "compacted" : "standard");
        CHECKF(in_buf(&x, p, l), "feature %d area outside buffer", f);
        if ((comp & X_BIT63) && c->comp[f].align64) CHECKF(((uintptr_t)p & 63) == 0, "feature %d (align64) not 64-aligned: %p", f, (void *)p);
        n++;
    }
    t_info("located_features", "%d", n);
}

TEST(locate_null_length_ok) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    for (DWORD f = 0; f < 3; f++) {
        DWORD l = 0;
        void *a = X.LocateXStateFeature(x.ctx, f, &l);
        void *b = X.LocateXStateFeature(x.ctx, f, NULL);
        CHECKF(a == b, "feature %lu: pointer differs with NULL Length (%p vs %p)", f, a, b);
    }
    xctx_free(&x);
}

TEST(locate_no_arch_flags) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    x.ctx->ContextFlags = 0;
    for (DWORD f = 0; f < 4; f++) {
        DWORD l = 0x999;
        CHECKF(X.LocateXStateFeature(x.ctx, f, &l) == NULL, "feature %lu located in a context with ContextFlags=0", f);
        len_untouched_note(l, f);
    }
    xctx_free(&x);
}

TEST(locate_does_not_modify_context) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    BYTE *snap = (BYTE *)malloc(x.len);
    REQUIRE(snap);
    memcpy(snap, x.buf, x.len);
    for (DWORD f = 0; f < 70; f++) { DWORD l; X.LocateXStateFeature(x.ctx, f, &l); }
    CHECK_MEM(x.buf, snap, x.len);
    free(snap);
    xctx_free(&x);
}

TEST(rtl_locate_extended_feature_matches) {
    DWORD64 en = enabled();
    need_rtl((void *)X.RtlLocateExtendedFeature, "RtlLocateExtendedFeature");
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    for (DWORD f = 2; f < 64; f++) {
        ULONG rl = 0x999; DWORD kl = 0x999;
        void *r = X.RtlLocateExtendedFeature(xex(x.ctx), f, &rl);
        void *k = X.LocateXStateFeature(x.ctx, f, &kl);
        CHECKF(r == k, "feature %lu: Rtl %p kernel32 %p", f, r, k);
        if (r || k) CHECKF(rl == kl, "feature %lu: Rtl length %lu kernel32 %lu", f, rl, kl);
    }
    /* legacy ids 0/1 are not extended features */
    CHECK(X.RtlLocateExtendedFeature(xex(x.ctx), 0, NULL) == NULL);
    CHECK(X.RtlLocateExtendedFeature(xex(x.ctx), 1, NULL) == NULL);
    (void)en;
    xctx_free(&x);
}

/* ======================= G. Get/SetXStateFeaturesMask ===================== */
static DWORD64 getmask(CONTEXT *c, BOOL *ok, DWORD *err) {
    DWORD64 m = 0xBADBADBADull;
    SetLastError(X_ERR_SENTINEL);
    *ok = X.GetXStateFeaturesMask(c, &m);
    *err = GetLastError();
    return m;
}

TEST(getmask_fresh_xstate_context) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_XSTATE);
    BOOL ok; DWORD err;
    DWORD64 m = getmask(x.ctx, &ok, &err);
    CHECKF(ok, "GetXStateFeaturesMask failed (%lu)", err);
    CHECKF(m == 0, "fresh XSTATE context without the FP flag: mask 0x%llx want 0", (unsigned long long)m);
    xctx_free(&x);
}

TEST(getmask_fp_flag_reports_legacy) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_FPFLAG | XA_XSTATE);
    BOOL ok; DWORD err;
    DWORD64 m = getmask(x.ctx, &ok, &err);
    CHECK(ok);
    CHECKF(m == 3, "context with flag 0x%lx: mask 0x%llx want 3", (unsigned long)XA_FPFLAG, (unsigned long long)m);
    xctx_free(&x);
    XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    m = getmask(x.ctx, &ok, &err);
    CHECK(ok);
    CHECKF(m == 3, "CONTEXT_ALL|XSTATE: mask 0x%llx want 3", (unsigned long long)m);
    xctx_free(&x);
}

TEST(getmask_plain_context) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_CONTROL);
    BOOL ok; DWORD err;
    DWORD64 m = getmask(x.ctx, &ok, &err);
    CHECK(ok);
    CHECK_EQ(m, 0);
    xctx_free(&x);
    XCTX_REQ(&x, XA_ALL);
    m = getmask(x.ctx, &ok, &err);
    CHECK(ok);
    CHECK_EQ(m, 3);
    xctx_free(&x);
#ifdef T_X86
    /* i386: only CONTEXT_EXTENDED_REGISTERS (FXSAVE image) carries the XSTATE legacy bits */
    XCTX_REQ(&x, XA_ARCH | 0x08);
    m = getmask(x.ctx, &ok, &err);
    CHECK(ok);
    CHECKF(m == 0, "i386 CONTEXT_FLOATING_POINT alone: mask 0x%llx want 0", (unsigned long long)m);
    xctx_free(&x);
#endif
}

TEST(getmask_no_arch_flags) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    x.ctx->ContextFlags = 0;
    BOOL ok; DWORD err;
    DWORD64 m = getmask(x.ctx, &ok, &err);
    CHECK(!ok);
    CHECKF(m == 0xBADBADBADull, "mask written on failure (0x%llx)", (unsigned long long)m);
    xctx_free(&x);
}
TEST(getmask_no_arch_flags_error_code) {
    need_core_api();
    wine_known_gap("GetXStateFeaturesMask fails without SetLastError");
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    x.ctx->ContextFlags = 0;
    BOOL ok; DWORD err;
    getmask(x.ctx, &ok, &err);
    CHECK(!ok);
    CHECKF(err == ERROR_NOT_SUPPORTED, "error %lu want ERROR_NOT_SUPPORTED (50, STATUS_NOT_SUPPORTED)", err);
    xctx_free(&x);
}

TEST(setmask_avx_roundtrip) {
    need_avx_api();
    xctx x; XCTX_REQ(&x, XA_XSTATE);
    DWORD flags0 = x.ctx->ContextFlags;
    SetLastError(X_ERR_SENTINEL);
    CHECK(X.SetXStateFeaturesMask(x.ctx, T_XS_AVX));
    BOOL ok; DWORD err;
    DWORD64 m = getmask(x.ctx, &ok, &err);
    CHECK(ok);
    CHECK_EQ(m, T_XS_AVX);
    CHECK_EQ(xs_bv(x.ctx), T_XS_AVX);
    CHECKF(x.ctx->ContextFlags == flags0, "ContextFlags changed 0x%lx -> 0x%lx by a mask without bits 0|1",
           (unsigned long)flags0, (unsigned long)x.ctx->ContextFlags);
    CHECK(X.SetXStateFeaturesMask(x.ctx, 0));
    m = getmask(x.ctx, &ok, &err);
    CHECK_EQ(m, 0);
    CHECK_EQ(xs_bv(x.ctx), 0);
    xctx_free(&x);
}

TEST(setmask_legacy_sets_fp_flag) {
    need_xstate_ctx();
    xctx x; XCTX_REQ(&x, XA_XSTATE);
    CHECK(X.SetXStateFeaturesMask(x.ctx, 1));
    CHECKF((x.ctx->ContextFlags & XA_FPFLAG) == XA_FPFLAG, "mask bit 0 did not add flag 0x%lx (flags 0x%lx)",
           (unsigned long)XA_FPFLAG, (unsigned long)x.ctx->ContextFlags);
    BOOL ok; DWORD err;
    CHECK_EQ(getmask(x.ctx, &ok, &err), 3);
    CHECK_EQ(xs_bv(x.ctx), 0);   /* bits 0|1 live in the legacy area, never in the chunk's XSTATE_BV */
    xctx_free(&x);
    /* plain (non-XSTATE) context: legacy-only mask succeeds and sets the flag */
    XCTX_REQ(&x, XA_CONTROL);
    CHECK(X.SetXStateFeaturesMask(x.ctx, 2));
    CHECK((x.ctx->ContextFlags & XA_FPFLAG) == XA_FPFLAG);
    xctx_free(&x);
}

/* kernelbase!SetXStateFeaturesMask, i386 branch: (mask & 3) sets CONTEXT_EXTENDED_REGISTERS and, for an
 * XSTATE context, raises CONTEXT_EX.Legacy.Length to 0x2CC (the FXSAVE image is now part of the context). */
#ifdef T_X86
TEST(setmask_x86_legacy_length_grows) {
    need_xstate_ctx();
    wine_known_gap("SetXStateFeaturesMask does not grow the i386 CONTEXT_EX.Legacy.Length");
    xctx x; XCTX_REQ(&x, XA_CONTROL | XA_XSTATE);
    CHECK_EQ(xex(x.ctx)->Legacy.Length, 0xCC);
    CHECK(X.SetXStateFeaturesMask(x.ctx, 0));
    CHECKF(xex(x.ctx)->Legacy.Length == 0xCC, "mask 0 changed Legacy.Length to 0x%lx", xex(x.ctx)->Legacy.Length);
    CHECK(X.SetXStateFeaturesMask(x.ctx, 3));
    CHECKF((x.ctx->ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS, "flags 0x%lx", (unsigned long)x.ctx->ContextFlags);
    CHECKF(xex(x.ctx)->Legacy.Length == 0x2CC, "Legacy.Length 0x%lx want 0x2CC after legacy bits were set", xex(x.ctx)->Legacy.Length);
    CHECK_EQ(xex(x.ctx)->Legacy.Offset, -0x2CC);
    xctx_free(&x);
}
#else
TEST(setmask_x86_legacy_length_grows) { SKIP("i386 CONTEXT_EX.Legacy.Length rule: x86 build only"); }
#endif

TEST(setmask_zero_keeps_fp_flag) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    CHECK(X.SetXStateFeaturesMask(x.ctx, 0));
    CHECKF((x.ctx->ContextFlags & XA_FPFLAG) == XA_FPFLAG, "mask 0 cleared the FP flag (documented only to set it)");
    xctx_free(&x);
}

TEST(setmask_unsupported_bits_dropped) {
    need_xstate_ctx();
    DWORD64 en = enabled();
    xctx x; XCTX_REQ(&x, XA_XSTATE);
    uint64_t junk = ~en & ~3ull;   /* every bit the OS does not enable, incl. bit 63 */
    SetLastError(X_ERR_SENTINEL);
    BOOL r = X.SetXStateFeaturesMask(x.ctx, en | junk);
    CHECKF(r, "SetXStateFeaturesMask with extra bits failed (%lu)", GetLastError());
    BOOL ok; DWORD err;
    DWORD64 m = getmask(x.ctx, &ok, &err);
    CHECKF((m & junk) == 0, "unsupported bits stored in XSTATE_BV: mask 0x%llx junk 0x%llx", (unsigned long long)m, (unsigned long long)(m & junk));
    CHECKF((m & ~3ull) == (en & ~3ull), "enabled bits lost: mask 0x%llx enabled 0x%llx", (unsigned long long)m, (unsigned long long)en);
    CHECKF((xs_bv(x.ctx) & junk) == 0, "raw XSTATE_BV 0x%llx carries unsupported bits", (unsigned long long)xs_bv(x.ctx));
    xctx_free(&x);
}

TEST(setmask_extended_on_plain_context) {
    need_core_api();
    xctx x; XCTX_REQ(&x, XA_ALL);
    CHECK(!X.SetXStateFeaturesMask(x.ctx, 3 | T_XS_AVX));
    CHECK(!X.SetXStateFeaturesMask(x.ctx, 1ull << 62));
    CHECK(!X.SetXStateFeaturesMask(x.ctx, X_BIT63));
    CHECK(X.SetXStateFeaturesMask(x.ctx, 3));
    CHECKF((x.ctx->ContextFlags & 0x40) == 0, "flags gained XSTATE: 0x%lx", (unsigned long)x.ctx->ContextFlags);
    xctx_free(&x);
}
TEST(setmask_extended_on_plain_context_error_code) {
    need_core_api();
    wine_known_gap("SetXStateFeaturesMask fails without SetLastError");
    xctx x; XCTX_REQ(&x, XA_ALL);
    SetLastError(X_ERR_SENTINEL);
    CHECK(!X.SetXStateFeaturesMask(x.ctx, 3 | T_XS_AVX));
    CHECKF(GetLastError() == ERROR_INVALID_PARAMETER, "error %lu want 87 (STATUS_INVALID_PARAMETER)", GetLastError());
    xctx_free(&x);
}

TEST(setmask_no_arch_flags) {
    need_xstate_ctx();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    x.ctx->ContextFlags = 0;
    uint64_t bv0 = xs_bv(x.ctx);
    CHECK(!X.SetXStateFeaturesMask(x.ctx, 3 | T_XS_AVX));
    CHECK_EQ(x.ctx->ContextFlags, 0);
    CHECK_EQ(xs_bv(x.ctx), bv0);
    xctx_free(&x);
}
TEST(setmask_no_arch_flags_error_code) {
    need_core_api();
    wine_known_gap("SetXStateFeaturesMask fails without SetLastError");
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    x.ctx->ContextFlags = 0;
    SetLastError(X_ERR_SENTINEL);
    CHECK(!X.SetXStateFeaturesMask(x.ctx, 3 | T_XS_AVX));
    CHECKF(GetLastError() == ERROR_NOT_SUPPORTED, "error %lu want 50 (STATUS_NOT_SUPPORTED)", GetLastError());
    xctx_free(&x);
}

TEST(setmask_preserves_header_and_data) {
    need_avx_api();
    xctx x; XCTX_REQ(&x, XA_XSTATE);
    uint64_t comp = xs_comp(x.ctx);
    BYTE *avx = (BYTE *)X.LocateXStateFeature(x.ctx, XSTATE_AVX, NULL);
    REQUIRE(avx);
    for (int i = 0; i < 256; i++) avx[i] = (BYTE)(i * 7 + 1);
    BYTE want[256]; memcpy(want, avx, 256);
    CHECK(X.SetXStateFeaturesMask(x.ctx, T_XS_AVX));
    CHECK(X.SetXStateFeaturesMask(x.ctx, 0));
    CHECK(X.SetXStateFeaturesMask(x.ctx, T_XS_AVX));
    CHECK_EQ(xs_comp(x.ctx), comp);
    CHECK_MEM(avx, want, 256);
    xctx_free(&x);
}

TEST(rtl_get_set_extended_features_mask) {
    need_xstate_ctx();
    DWORD64 en = enabled();
    need_rtl((void *)X.RtlGetExtendedFeaturesMask, "RtlGetExtendedFeaturesMask");
    need_rtl((void *)X.RtlSetExtendedFeaturesMask, "RtlSetExtendedFeaturesMask");
    xctx x; XCTX_REQ(&x, XA_XSTATE);
    X_CTX_EX *e = xex(x.ctx);
    CHECK_EQ(X.RtlGetExtendedFeaturesMask(e), 0);
    X.RtlSetExtendedFeaturesMask(e, ~0ull);
    CHECKF(xs_bv(x.ctx) == (en & ~3ull), "Rtl set(~0): XSTATE_BV 0x%llx want 0x%llx", (unsigned long long)xs_bv(x.ctx), (unsigned long long)(en & ~3ull));
    CHECK_EQ(X.RtlGetExtendedFeaturesMask(e), en & ~3ull);
    X.RtlSetExtendedFeaturesMask(e, 3);
    CHECK_EQ(xs_bv(x.ctx), 0);
    CHECK_EQ(X.RtlGetExtendedFeaturesMask(e), 0);
    /* Rtl never touches ContextFlags (that is the kernel32 wrapper's job) */
    CHECK_EQ(x.ctx->ContextFlags, XA_XSTATE);
    xctx_free(&x);
}

/* ======================= H. CopyContext =================================== */
static void fill_legacy(CONTEXT *c, BYTE seed) {
    BYTE *p = (BYTE *)c;
    size_t skip0 = offsetof(CONTEXT, ContextFlags), skip1 = skip0 + sizeof(DWORD);
    for (size_t i = 0; i < sizeof(CONTEXT); i++) if (i < skip0 || i >= skip1) p[i] = (BYTE)(seed + i * 13);
#ifdef T_X64
    c->MxCsr = 0x1F80; c->FltSave.MxCsr = 0x1F80; c->FltSave.MxCsr_Mask = 0xFFFF; c->FltSave.ControlWord = 0x27F;
    c->EFlags = 0x202; c->SegCs = 0x33; c->SegSs = 0x2B;
#else
    c->EFlags = 0x202; c->SegCs = 0x23; c->SegSs = 0x2B;
#endif
}

TEST(copy_plain_full) {
    need_core_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL); XCTX_REQ(&d, XA_ALL);
    fill_legacy(s.ctx, 0x10);
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.CopyContext(d.ctx, XA_ALL, s.ctx);
    CHECKF(ok, "CopyContext(ALL) failed (%lu)", GetLastError());
#ifdef T_X64
    CHECK_EQ(d.ctx->Rip, s.ctx->Rip); CHECK_EQ(d.ctx->Rsp, s.ctx->Rsp); CHECK_EQ(d.ctx->Rax, s.ctx->Rax);
    CHECK_EQ(d.ctx->R15, s.ctx->R15); CHECK_EQ(d.ctx->EFlags, s.ctx->EFlags); CHECK_EQ(d.ctx->SegDs, s.ctx->SegDs);
    CHECK_EQ(d.ctx->Dr7, s.ctx->Dr7);
    CHECK_MEM(&d.ctx->FltSave, &s.ctx->FltSave, offsetof(XSAVE_FORMAT, Reserved4));   /* bytes 416..511 are reserved */
#else
    CHECK_EQ(d.ctx->Eip, s.ctx->Eip); CHECK_EQ(d.ctx->Esp, s.ctx->Esp); CHECK_EQ(d.ctx->Eax, s.ctx->Eax);
    CHECK_EQ(d.ctx->EFlags, s.ctx->EFlags); CHECK_EQ(d.ctx->SegDs, s.ctx->SegDs); CHECK_EQ(d.ctx->Dr7, s.ctx->Dr7);
    CHECK_MEM(d.ctx->ExtendedRegisters, s.ctx->ExtendedRegisters, 288);   /* FXSAVE through XMM7; rest reserved */
#endif
    CHECKF((d.ctx->ContextFlags & XA_ALL) == XA_ALL, "dest flags 0x%lx", (unsigned long)d.ctx->ContextFlags);
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_control_only_partial) {
    need_core_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL); XCTX_REQ(&d, XA_CONTROL);
    fill_legacy(s.ctx, 0x20);
    fill_legacy(d.ctx, 0x90);
    d.ctx->ContextFlags = XA_CONTROL;
#ifdef T_X64
    DWORD64 rax0 = d.ctx->Rax; BYTE flt0[512]; memcpy(flt0, &d.ctx->FltSave, 512);
#else
    DWORD eax0 = d.ctx->Eax; BYTE flt0[512]; memcpy(flt0, d.ctx->ExtendedRegisters, 512);
#endif
    CHECK(X.CopyContext(d.ctx, XA_CONTROL, s.ctx));
#ifdef T_X64
    CHECK_EQ(d.ctx->Rip, s.ctx->Rip); CHECK_EQ(d.ctx->Rsp, s.ctx->Rsp); CHECK_EQ(d.ctx->EFlags, s.ctx->EFlags);
    CHECK_EQ(d.ctx->Rax, rax0);
    CHECK_MEM(&d.ctx->FltSave, flt0, 512);
#else
    CHECK_EQ(d.ctx->Eip, s.ctx->Eip); CHECK_EQ(d.ctx->Esp, s.ctx->Esp); CHECK_EQ(d.ctx->EFlags, s.ctx->EFlags);
    CHECK_EQ(d.ctx->Eax, eax0);
    CHECK_MEM(d.ctx->ExtendedRegisters, flt0, 512);
#endif
    CHECKF(d.ctx->ContextFlags == XA_CONTROL, "dest flags 0x%lx want 0x%lx", (unsigned long)d.ctx->ContextFlags, (unsigned long)XA_CONTROL);
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_dest_flags_union) {
    need_core_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL); XCTX_REQ(&d, XA_CONTROL);
    CHECK(X.CopyContext(d.ctx, XA_INTEGER, s.ctx));
    CHECKF(d.ctx->ContextFlags == (XA_CONTROL | XA_INTEGER), "dest flags 0x%lx want old|copied 0x%lx",
           (unsigned long)d.ctx->ContextFlags, (unsigned long)(XA_CONTROL | XA_INTEGER));
    /* requested parts the source lacks are not claimed */
    xctx s2, d2;
    XCTX_REQ(&s2, XA_CONTROL); XCTX_REQ(&d2, XA_CONTROL);
    CHECK(X.CopyContext(d2.ctx, XA_ALL, s2.ctx));
    CHECKF(d2.ctx->ContextFlags == XA_CONTROL, "dest flags 0x%lx want 0x%lx (source had only CONTROL)",
           (unsigned long)d2.ctx->ContextFlags, (unsigned long)XA_CONTROL);
    xctx_free(&s); xctx_free(&d); xctx_free(&s2); xctx_free(&d2);
}

TEST(copy_xstate_to_xstate) {
    need_avx_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL | XA_XSTATE);
    fill_legacy(s.ctx, 0x31);
    BYTE *sa = (BYTE *)X.LocateXStateFeature(s.ctx, XSTATE_AVX, NULL), *da = (BYTE *)X.LocateXStateFeature(d.ctx, XSTATE_AVX, NULL);
    REQUIRE(sa && da);
    for (int i = 0; i < 256; i++) { sa[i] = (BYTE)(0x80 ^ i); da[i] = 0x11; }
    REQUIRE(X.SetXStateFeaturesMask(s.ctx, 3 | T_XS_AVX));
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx);
    CHECKF(ok, "CopyContext(ALL|XSTATE) failed (%lu)", GetLastError());
    BOOL gok; DWORD err;
    CHECK_EQ(getmask(d.ctx, &gok, &err), 3 | T_XS_AVX);
    CHECK_MEM(da, sa, 256);
    CHECK_EQ(xs_comp(d.ctx), xs_comp(s.ctx));
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_xstate_mask_zero_skips_data) {
    need_avx_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL | XA_XSTATE);
    BYTE *sa = (BYTE *)X.LocateXStateFeature(s.ctx, XSTATE_AVX, NULL), *da = (BYTE *)X.LocateXStateFeature(d.ctx, XSTATE_AVX, NULL);
    REQUIRE(sa && da);
    memset(sa, 0x77, 256); memset(da, 0x22, 256);
    REQUIRE(X.SetXStateFeaturesMask(d.ctx, T_XS_AVX));
    REQUIRE(X.SetXStateFeaturesMask(s.ctx, 0));
    CHECK(X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx));
    BOOL ok; DWORD err;
    CHECKF((getmask(d.ctx, &ok, &err) & T_XS_AVX) == 0, "dest still claims AVX after copying a source whose mask is 0");
    BYTE want[256]; memset(want, 0x22, 256);
    CHECK_MEM(da, want, 256);   /* data of features absent from the source mask is not copied */
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_xstate_into_plain_dest) {
    need_avx_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL);
    BYTE *snap = (BYTE *)malloc(d.len);
    REQUIRE(snap);
    memcpy(snap, d.buf, d.len);
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx);
    CHECK(!ok);
    CHECKF(GetLastError() == ERROR_MORE_DATA, "error %lu want ERROR_MORE_DATA (234, STATUS_BUFFER_OVERFLOW)", GetLastError());
    CHECK_MEM(d.buf, snap, d.len);
    free(snap);
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_plain_src_xstate_flag_ok) {
    need_xstate_ctx();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL); XCTX_REQ(&d, XA_ALL);
    fill_legacy(s.ctx, 0x41);
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx);
    CHECKF(ok, "requesting XSTATE from a source without it must succeed (%lu)", GetLastError());
    CHECKF((d.ctx->ContextFlags & 0x40) == 0, "dest gained the XSTATE flag 0x%lx", (unsigned long)d.ctx->ContextFlags);
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_invalid_flags) {
    need_core_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL); XCTX_REQ(&d, XA_ALL);
    DWORD bad[] = { 0, 0x00110001u, 0x1Fu };
    for (size_t i = 0; i < T_COUNTOF(bad); i++) {
        SetLastError(X_ERR_SENTINEL);
        BOOL ok = X.CopyContext(d.ctx, bad[i], s.ctx);
        CHECKF(!ok, "flags 0x%lx accepted", (unsigned long)bad[i]);
        CHECKF(GetLastError() == ERROR_INVALID_PARAMETER, "flags 0x%lx: error %lu want 87", (unsigned long)bad[i], GetLastError());
    }
    xctx_free(&s); xctx_free(&d);
}
TEST(copy_invalid_flags_reserved_bits) {
    need_core_api();
    /* RtlCopyContext runs RtlpValidateContextFlags on the requested flags and on both contexts' flags */
    wine_known_gap("CopyContext does not validate reserved ContextFlags bits");
    xctx s, d;
    XCTX_REQ(&s, XA_ALL); XCTX_REQ(&d, XA_ALL);
    DWORD bad[] = { XA_ARCH | XA_BADLOW, XA_ALL | 0x20000000u };
    for (size_t i = 0; i < T_COUNTOF(bad); i++) {
        SetLastError(X_ERR_SENTINEL);
        BOOL ok = X.CopyContext(d.ctx, bad[i], s.ctx);
        CHECKF(!ok, "flags 0x%lx accepted", (unsigned long)bad[i]);
        CHECKF(GetLastError() == ERROR_INVALID_PARAMETER, "flags 0x%lx: error %lu want 87", (unsigned long)bad[i], GetLastError());
    }
    s.ctx->ContextFlags = XA_ARCH | XA_BADLOW;   /* source with corrupt flags */
    SetLastError(X_ERR_SENTINEL);
    CHECK(!X.CopyContext(d.ctx, XA_ALL, s.ctx));
    CHECK_EQ(GetLastError(), ERROR_INVALID_PARAMETER);
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_xstate_mask_filtered) {
    DWORD64 en = enabled();
    need_avx_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL | XA_XSTATE);
    /* corrupt the source XSTATE_BV directly with bits the OS does not enable */
    uint64_t junk = ~en & ~3ull & ~X_BIT63 & 0x7FFFFFFFFFFFFFF8ull;
    *(uint64_t *)xchunk(s.ctx) = T_XS_AVX | junk;
    BOOL ok = X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx);
    CHECKF(ok, "CopyContext failed (%lu)", GetLastError());
    if (ok) CHECKF(xs_bv(d.ctx) == T_XS_AVX, "dest XSTATE_BV 0x%llx want 0x%llx (source & enabled)",
                   (unsigned long long)xs_bv(d.ctx), (unsigned long long)T_XS_AVX);
    xctx_free(&s); xctx_free(&d);
}

TEST(copy_into_smaller_compacted_dest) {
    need_init2();
    need_avx_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE);
    REQUIRE(xctx_new2(&d, XA_ALL | XA_XSTATE, 3, 1, 0));
    if (!(xs_comp(d.ctx) & X_BIT63)) { xctx_free(&s); xctx_free(&d); SKIP("standard XSAVE format: every XSTATE context has full size"); }
    REQUIRE(X.SetXStateFeaturesMask(s.ctx, 3 | T_XS_AVX));
    wine_known_gap("CopyContext into a compacted destination without room for AVX succeeds");
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx);
    CHECKF(!ok, "copying AVX state into a compacted context whose chunk has no AVX room must fail");
    CHECKF(GetLastError() == ERROR_MORE_DATA, "error %lu want 234", GetLastError());
    xctx_free(&s); xctx_free(&d);
}

/* CopyContext(XSTATE only): ntdll!RtlpCopyLegacyContext copies only the parts named by (src & requested)
 * flags, so the legacy CONTEXT (incl. FltSave/XMM lows) is untouched while the XSTATE chunk is copied. */
TEST(copy_xstate_only_request) {
    DWORD64 en = enabled();
    need_avx_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL | XA_XSTATE);
    fill_legacy(s.ctx, 0x61); fill_legacy(d.ctx, 0x19);
    BYTE *sa = (BYTE *)X.LocateXStateFeature(s.ctx, XSTATE_AVX, NULL), *da = (BYTE *)X.LocateXStateFeature(d.ctx, XSTATE_AVX, NULL);
    REQUIRE(sa && da);
    for (int i = 0; i < 256; i++) { sa[i] = (BYTE)(i * 3 + 7); da[i] = 0x33; }
    REQUIRE(X.SetXStateFeaturesMask(s.ctx, 3 | T_XS_AVX));
    CONTEXT before; memcpy(&before, d.ctx, sizeof before);
    SetLastError(X_ERR_SENTINEL);
    BOOL ok = X.CopyContext(d.ctx, XA_XSTATE, s.ctx);
    CHECKF(ok, "CopyContext(CONTEXT_XSTATE only) failed (%lu)", GetLastError());
    CHECKF(d.ctx->ContextFlags == (XA_ALL | XA_XSTATE), "dest flags 0x%lx want 0x%lx", (unsigned long)d.ctx->ContextFlags,
           (unsigned long)(XA_ALL | XA_XSTATE));
    before.ContextFlags = d.ctx->ContextFlags;
    CHECK_MEM(d.ctx, &before, sizeof(CONTEXT));   /* no legacy part requested -> legacy CONTEXT unchanged */
    CHECK_MEM(da, sa, 256);
    CHECKF(xs_bv(d.ctx) == (T_XS_AVX & en), "dest XSTATE_BV 0x%llx want 0x%llx", (unsigned long long)xs_bv(d.ctx), (unsigned long long)T_XS_AVX);
    xctx_free(&s); xctx_free(&d);
}

/* ntdll!RtlpCopyXStateChunk rebuilds the destination header: 64 bytes zeroed, XSTATE_BV = src & enabled & ~3,
 * XCOMP_BV = ((enabled|3) & src XCOMP_BV) | bit63 when compacted, else 0. Reserved header bytes left nonzero
 * would make XRSTOR(S) of the context #GP (SDM 13.8). */
TEST(copy_xstate_dest_header_rebuilt) {
    need_avx_api();
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL | XA_XSTATE);
    REQUIRE(X.SetXStateFeaturesMask(s.ctx, 3 | T_XS_AVX));
    BYTE *dh = xchunk(d.ctx);
    memset(dh + 16, 0xEE, 48);
    CHECK(X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx));
    BYTE zero[48] = { 0 };
    CHECK_MEM(dh + 16, zero, 48);
    CHECK_EQ(xs_bv(d.ctx), T_XS_AVX);
    CHECK_EQ(xs_comp(d.ctx), xs_comp(s.ctx));
    xctx_free(&s); xctx_free(&d);
}

#ifdef T_X86
/* ntdll!RtlCopyContext, i386: copying CONTEXT_EXTENDED_REGISTERS into an XSTATE destination that lacked
 * them raises the destination's CONTEXT_EX.Legacy.Length from 0xCC to 0x2CC. */
TEST(copy_x86_legacy_length_grows) {
    need_xstate_ctx();
    wine_known_gap("CopyContext does not grow the i386 destination CONTEXT_EX.Legacy.Length");
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_CONTROL | XA_XSTATE);
    fill_legacy(s.ctx, 0x72);
    REQUIRE(xex(d.ctx)->Legacy.Length == 0xCC);
    CHECK(X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx));
    CHECKF((d.ctx->ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS, "flags 0x%lx", (unsigned long)d.ctx->ContextFlags);
    CHECKF(xex(d.ctx)->Legacy.Length == 0x2CC, "dest Legacy.Length 0x%lx want 0x2CC", xex(d.ctx)->Legacy.Length);
    CHECK_MEM(d.ctx->ExtendedRegisters, s.ctx->ExtendedRegisters, 288);
    xctx_free(&s); xctx_free(&d);
}
/* ntdll!RtlpCopyExtendedContext: dest Legacy.Offset must equal the source's and dest Legacy.Length must be
 * >= the source's, else STATUS_INVALID_PARAMETER (nothing copied). */
TEST(rtl_copy_extended_context_legacy_too_small) {
    need_xstate_ctx();
    need_rtl((void *)X.RtlCopyExtendedContext, "RtlCopyExtendedContext");
    wine_known_gap("RtlCopyExtendedContext does not compare the Legacy chunks");
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_CONTROL | XA_XSTATE);
    CHECK_EQ((ULONG)X.RtlCopyExtendedContext(xex(d.ctx), XA_ALL | XA_XSTATE, xex(s.ctx)), (ULONG)X_STATUS_INVALID_PARAMETER);
    CHECK_EQ(xex(d.ctx)->Legacy.Length, 0xCC);
    /* the other direction (dest has the larger legacy area) is fine */
    CHECK_EQ((ULONG)X.RtlCopyExtendedContext(xex(s.ctx), XA_CONTROL | XA_XSTATE, xex(d.ctx)), 0u);
    xctx_free(&s); xctx_free(&d);
}
#else
TEST(copy_x86_legacy_length_grows) { SKIP("i386 CONTEXT_EX.Legacy.Length rule: x86 build only"); }
TEST(rtl_copy_extended_context_legacy_too_small) { SKIP("needs differing i386 Legacy.Length: x86 build only"); }
#endif

TEST(rtl_copy_context_status) {
    need_avx_api();
    need_rtl((void *)X.RtlCopyContext, "RtlCopyContext");
    xctx s, d, p;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL | XA_XSTATE); XCTX_REQ(&p, XA_ALL);
    REQUIRE(X.SetXStateFeaturesMask(s.ctx, 3 | T_XS_AVX));
    CHECK_EQ((ULONG)X.RtlCopyContext(d.ctx, XA_ALL | XA_XSTATE, s.ctx), 0u);
    CHECK_EQ((ULONG)X.RtlCopyContext(p.ctx, XA_ALL | XA_XSTATE, s.ctx), 0x80000005u);
    CHECK_EQ((ULONG)X.RtlCopyContext(d.ctx, 0, s.ctx), 0xC000000Du);
    xctx_free(&s); xctx_free(&d); xctx_free(&p);
}

TEST(rtl_copy_extended_context) {
    need_avx_api();
    need_rtl((void *)X.RtlCopyExtendedContext, "RtlCopyExtendedContext");
    xctx s, d;
    XCTX_REQ(&s, XA_ALL | XA_XSTATE); XCTX_REQ(&d, XA_ALL | XA_XSTATE);
    fill_legacy(s.ctx, 0x52);
    BYTE *sa = (BYTE *)X.LocateXStateFeature(s.ctx, XSTATE_AVX, NULL), *da = (BYTE *)X.LocateXStateFeature(d.ctx, XSTATE_AVX, NULL);
    REQUIRE(sa && da);
    for (int i = 0; i < 256; i++) sa[i] = (BYTE)(i ^ 0x5C);
    REQUIRE(X.SetXStateFeaturesMask(s.ctx, 3 | T_XS_AVX));
    LONG st = X.RtlCopyExtendedContext(xex(d.ctx), XA_ALL | XA_XSTATE, xex(s.ctx));
    CHECKF(st == X_STATUS_SUCCESS, "status 0x%lx", (unsigned long)st);
    CHECK_MEM(da, sa, 256);
    CHECK_EQ(xs_bv(d.ctx), T_XS_AVX);
#ifdef T_X64
    CHECK_EQ(d.ctx->Rip, s.ctx->Rip);
#else
    CHECK_EQ(d.ctx->Eip, s.ctx->Eip);
#endif
    CHECK_EQ((ULONG)X.RtlCopyExtendedContext(xex(d.ctx), 0, xex(s.ctx)), 0xC000000Du);
    xctx_free(&s); xctx_free(&d);
}

/* ======================= I. real thread contexts ========================== */
typedef struct xw {
    t_vregs in, out;
    volatile long stop, ready;
    unsigned mxcsr; unsigned short fcw;
    long rem; int kind;
    HANDLE h;
} xw;

static DWORD WINAPI xw_proc(void *p) {
    xw *w = (xw *)p;
    w->mxcsr = t_get_mxcsr();
    w->fcw = t_get_x87_cw();
    InterlockedExchange(&w->ready, 1);
    w->rem = t_vregs_spin(w->kind, &w->in, &w->out, &w->stop, 0xFFFFFFFFul);
    return 0;
}
static xw *xw_start(int kind, uint32_t tag) {
    xw *w = (xw *)VirtualAlloc(NULL, sizeof(xw), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!w) return NULL;
    w->kind = kind;
    t_vregs_pattern(&w->in, kind, tag);
    w->h = CreateThread(NULL, 0, xw_proc, w, 0, NULL);
    if (!w->h) { VirtualFree(w, 0, MEM_RELEASE); return NULL; }
    for (int i = 0; i < 2000 && !w->ready; i++) Sleep(1);
    Sleep(20);
    return w;
}
static int xw_finish(xw *w) {
    InterlockedExchange(&w->stop, 1);
    while ((int)ResumeThread(w->h) > 1) {}
    DWORD r = WaitForSingleObject(w->h, 5000);
    CloseHandle(w->h);
    return r == WAIT_OBJECT_0;
}
static void xw_free(xw *w) { if (w) VirtualFree(w, 0, MEM_RELEASE); }

/* Suspend and capture until the spin routine has its registers loaded (checked on XMM0 low). */
static int xw_capture(xw *w, xctx *x, DWORD64 reqmask, DWORD *err) {
    for (int tries = 0; tries < 25; tries++) {
        if (SuspendThread(w->h) == (DWORD)-1) { *err = GetLastError(); return 0; }
        DWORD flags = x->ctx->ContextFlags;
        if ((flags & XA_XSTATE) == XA_XSTATE && reqmask != ~0ull) X.SetXStateFeaturesMask(x->ctx, reqmask);
        if (!GetThreadContext(w->h, x->ctx)) { *err = GetLastError(); ResumeThread(w->h); return 0; }
        BYTE *xmm = (BYTE *)x->ctx + XA_LEG1_OFF;
        if (memcmp(xmm, w->in.v[0], 16) == 0) return 1;
        ResumeThread(w->h);
        x->ctx->ContextFlags = flags;
        Sleep(10);
    }
    *err = 0;
    return 2;   /* thread never observed with loaded registers */
}

static void need_avx_exec(void) {
    need_avx_api();
    if (!cpu_has(F_AVX)) SKIP("AVX not usable (cpu_has)");
}

TEST(thread_get_xstate_avx) {
    need_avx_exec();
    xw *w = xw_start(T_VK_YMM, 0x5101);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx x;
    if (!xctx_new(&x, XA_FP_ALL | XA_CONTROL | XA_XSTATE)) { xw_finish(w); xw_free(w); REQUIREF(0, "InitializeContext failed"); }
    DWORD err = 0;
    int r = xw_capture(w, &x, 3 | T_XS_AVX, &err);
    CHECKF(r == 1, "GetThreadContext(XSTATE) on suspended worker: result %d error %lu", r, err);
    if (r == 1) {
        int n = t_vregs_count(T_VK_YMM);
        CHECKF((x.ctx->ContextFlags & XA_XSTATE) == XA_XSTATE, "XSTATE flag lost after GetThreadContext (0x%lx)", (unsigned long)x.ctx->ContextFlags);
        BOOL ok; DWORD e2;
        DWORD64 m = getmask(x.ctx, &ok, &e2);
        CHECK(ok);
        CHECKF(m & T_XS_AVX, "YMM uppers hold a non-init pattern but the mask 0x%llx lacks AVX", (unsigned long long)m);
        CHECKF((m & ~enabled()) == 0, "mask 0x%llx has bits outside enabled 0x%llx", (unsigned long long)m, (unsigned long long)enabled());
        BYTE *lo = (BYTE *)X.LocateXStateFeature(x.ctx, XSTATE_LEGACY_SSE, NULL);
        BYTE *hi = (BYTE *)X.LocateXStateFeature(x.ctx, XSTATE_AVX, NULL);
        CHECKF(lo && hi, "LocateXStateFeature: SSE %p AVX %p", (void *)lo, (void *)hi);   /* no REQUIRE: worker suspended */
        char lbl[48];
        for (int i = 0; lo && hi && i < n; i++) {
            snprintf(lbl, sizeof lbl, "ctx xmm%d low", i);
            CHECK_REG(lbl, lo + 16 * i, w->in.v[i], 16);
            snprintf(lbl, sizeof lbl, "ctx ymm%d upper", i);
            CHECK_REG(lbl, hi + 16 * i, w->in.v[i] + 16, 16);
        }
    }
    CHECK(xw_finish(w));
    t_vregs_check(&w->out, &w->in, T_VK_YMM, 0, -1, "worker after GetThreadContext");
    xctx_free(&x); xw_free(w);
}

TEST(thread_get_mxcsr_x87cw) {
    need_core_api();
    xw *w = xw_start(T_VK_XMM, 0x5102);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx x;
    DWORD fl = XA_FP_ALL | XA_CONTROL | ((enabled() & T_XS_AVX) ? XA_XSTATE : 0);
    if (!xctx_new(&x, fl)) { xw_finish(w); xw_free(w); REQUIREF(0, "InitializeContext failed"); }
    DWORD err = 0;
    int r = xw_capture(w, &x, ~0ull, &err);
    CHECKF(r == 1, "GetThreadContext: result %d error %lu", r, err);
    XSAVE_FORMAT *fx = r == 1 ? (XSAVE_FORMAT *)X.LocateXStateFeature(x.ctx, 0, NULL) : NULL;
    if (r == 1) CHECK(fx != NULL);   /* no REQUIRE: worker suspended */
    if (fx) {
        CHECKF(fx->MxCsr == w->mxcsr, "context MXCSR 0x%lx, thread 0x%x", (unsigned long)fx->MxCsr, w->mxcsr);
        CHECKF(fx->ControlWord == w->fcw, "context x87 CW 0x%x, thread 0x%x", fx->ControlWord, w->fcw);
#ifdef T_X64
        CHECKF(x.ctx->MxCsr == w->mxcsr, "CONTEXT.MxCsr 0x%lx, thread 0x%x", (unsigned long)x.ctx->MxCsr, w->mxcsr);
        CHECKF(w->mxcsr == 0x1F80, "new thread MXCSR 0x%x want 0x1F80", w->mxcsr);
        CHECKF(w->fcw == 0x27F, "new thread x87 CW 0x%x want 0x27F", w->fcw);
#else
        CHECKF((x.ctx->FloatSave.ControlWord & 0xFFFF) == w->fcw, "FloatSave.ControlWord 0x%lx, thread 0x%x",
               (unsigned long)x.ctx->FloatSave.ControlWord, w->fcw);
        t_info("x86_new_thread_mxcsr", "0x%x", w->mxcsr);
        t_info("x86_new_thread_fcw", "0x%x", w->fcw);
#endif
        int n = t_vregs_count(T_VK_XMM);
        char lbl[32];
        for (int i = 0; i < n; i++) { snprintf(lbl, sizeof lbl, "ctx xmm%d", i); CHECK_REG(lbl, &fx->XmmRegisters[i], w->in.v[i], 16); }
    }
    CHECK(xw_finish(w));
    xctx_free(&x); xw_free(w);
}

TEST(thread_set_xstate_avx) {
    need_avx_exec();
    t_vregs *want = (t_vregs *)_aligned_malloc(sizeof(t_vregs), 64);
    REQUIRE(want);
    xw *w = xw_start(T_VK_YMM, 0x5103);
    if (!w) { _aligned_free(want); SKIP("CreateThread failed (%lu)", GetLastError()); }
    xctx x;
    if (!xctx_new(&x, XA_FP_ALL | XA_XSTATE)) { xw_finish(w); xw_free(w); _aligned_free(want); REQUIREF(0, "InitializeContext failed"); }
    DWORD err = 0;
    int r = xw_capture(w, &x, 3 | T_XS_AVX, &err);
    memcpy(want, &w->in, sizeof *want);
    int n = t_vregs_count(T_VK_YMM);
    CHECKF(r == 1, "GetThreadContext: result %d error %lu", r, err);
    if (r == 1) {
        t_vregs alt;
        t_vregs_pattern(&alt, T_VK_YMM, 0x5104);
        BYTE *hi = (BYTE *)X.LocateXStateFeature(x.ctx, XSTATE_AVX, NULL);
        CHECKF(hi != NULL, "LocateXStateFeature(AVX) = NULL");   /* no REQUIRE: worker suspended */
        for (int i = 0; hi && i < n; i++) { memcpy(hi + 16 * i, alt.v[i] + 16, 16); memcpy(want->v[i] + 16, alt.v[i] + 16, 16); }
        CHECK(X.SetXStateFeaturesMask(x.ctx, 3 | T_XS_AVX));
        SetLastError(X_ERR_SENTINEL);
        BOOL ok = hi && SetThreadContext(w->h, x.ctx);
        CHECKF(ok, "SetThreadContext(XSTATE) failed (%lu)", GetLastError());
        if (!ok) memcpy(want, &w->in, sizeof *want);
    }
    CHECK(xw_finish(w));
    if (r == 1) t_vregs_check(&w->out, want, T_VK_YMM, 0, -1, "after SetThreadContext(new YMM uppers)");
    _aligned_free(want);
    xctx_free(&x); xw_free(w);
}

TEST(thread_set_xstate_avx_mask_cleared) {
    need_avx_exec();
    /* SetThreadContext with XSTATE_BV[AVX]=0: the AVX component is in its initial configuration
     * (YMM uppers = 0) per the XRSTOR contract; the low halves keep the legacy-area values. */
    xw *w = xw_start(T_VK_YMM, 0x5105);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx x;
    if (!xctx_new(&x, XA_FP_ALL | XA_XSTATE)) { xw_finish(w); xw_free(w); REQUIREF(0, "InitializeContext failed"); }
    DWORD err = 0;
    int r = xw_capture(w, &x, 3 | T_XS_AVX, &err);
    CHECKF(r == 1, "GetThreadContext: result %d error %lu", r, err);
    BOOL ok = FALSE;
    if (r == 1) {
        CHECK(X.SetXStateFeaturesMask(x.ctx, 3));
        ok = SetThreadContext(w->h, x.ctx);
        CHECKF(ok, "SetThreadContext failed (%lu)", GetLastError());
    }
    CHECK(xw_finish(w));
    if (r == 1 && ok) {
        int n = t_vregs_count(T_VK_YMM);
        BYTE zero[16] = { 0 };
        int nonzero = 0, lowbad = 0;
        for (int i = 0; i < n; i++) {
            if (memcmp(w->out.v[i] + 16, zero, 16)) nonzero++;
            if (memcmp(w->out.v[i], w->in.v[i], 16)) lowbad++;
        }
        CHECKF(lowbad == 0, "%d XMM low halves changed", lowbad);
        if (nonzero) t_note("%d YMM uppers kept their value after SetThreadContext with AVX absent from the mask "
                            "(the OS ignored absent components instead of initializing them)", nonzero);
        else t_note("YMM uppers were initialized to 0 (AVX absent from the mask)");
    }
    xctx_free(&x); xw_free(w);
}

TEST(thread_get_request_mask_zero_info) {
    need_avx_exec();
    xw *w = xw_start(T_VK_YMM, 0x5106);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx x;
    if (!xctx_new(&x, XA_FP_ALL | XA_XSTATE)) { xw_finish(w); xw_free(w); REQUIREF(0, "InitializeContext failed"); }
    DWORD err = 0;
    int r = xw_capture(w, &x, 0, &err);
    CHECKF(r == 1, "GetThreadContext with request mask 0 failed: %d / %lu", r, err);
    if (r == 1) {
        BOOL ok; DWORD e2;
        DWORD64 m = getmask(x.ctx, &ok, &e2);
        CHECKF((m & ~enabled()) == 0, "mask 0x%llx outside enabled", (unsigned long long)m);
        t_info("mask_after_get_request0", "0x%llx", (unsigned long long)m);
        t_note("GetThreadContext with XSTATE request mask 0 returned mask 0x%llx", (unsigned long long)m);
    }
    CHECK(xw_finish(w));
    xctx_free(&x); xw_free(w);
}

TEST(thread_plain_vs_xstate_legacy_equal) {
    need_core_api();
    xw *w = xw_start(T_VK_XMM, 0x5107);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx a, b;
    DWORD fx = (enabled() & T_XS_AVX) ? XA_XSTATE : 0;
    if (!xctx_new(&a, XA_FP_ALL | XA_CONTROL) || !xctx_new(&b, XA_FP_ALL | XA_CONTROL | fx)) {
        xw_finish(w); xw_free(w); REQUIREF(0, "InitializeContext failed");
    }
    DWORD err = 0;
    int r = xw_capture(w, &a, ~0ull, &err);
    CHECKF(r == 1, "plain GetThreadContext: %d / %lu", r, err);
    if (r == 1) {
        BOOL ok = GetThreadContext(w->h, b.ctx);
        CHECKF(ok, "XSTATE GetThreadContext failed (%lu)", GetLastError());
        if (ok) {
            /* legacy FXSAVE image (x87 + MXCSR + XMM) is identical between both requests */
            CHECK_MEM((BYTE *)b.ctx + XA_LEG1_OFF, (BYTE *)a.ctx + XA_LEG1_OFF, 16 * t_vregs_count(T_VK_XMM));
            XSAVE_FORMAT *fa = (XSAVE_FORMAT *)((BYTE *)a.ctx + XA_LEG0_OFF), *fb = (XSAVE_FORMAT *)((BYTE *)b.ctx + XA_LEG0_OFF);
            CHECK_EQ(fa->ControlWord, fb->ControlWord);
            CHECK_EQ(fa->MxCsr, fb->MxCsr);
#ifdef T_X64
            CHECK_EQ(a.ctx->Rip, b.ctx->Rip);
            CHECK_EQ(a.ctx->Rsp, b.ctx->Rsp);
#else
            CHECK_EQ(a.ctx->Eip, b.ctx->Eip);
            CHECK_EQ(a.ctx->Esp, b.ctx->Esp);
#endif
        }
    }
    CHECK(xw_finish(w));
    xctx_free(&a); xctx_free(&b); xw_free(w);
}

TEST(thread_copycontext_from_thread) {
    need_avx_exec();
    xw *w = xw_start(T_VK_YMM, 0x5108);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx x, d;
    if (!xctx_new(&x, XA_FP_ALL | XA_CONTROL | XA_XSTATE) || !xctx_new(&d, XA_ALL | XA_XSTATE)) {
        xw_finish(w); xw_free(w); REQUIREF(0, "InitializeContext failed");
    }
    DWORD err = 0;
    int r = xw_capture(w, &x, 3 | T_XS_AVX, &err);
    CHECKF(r == 1, "GetThreadContext: %d / %lu", r, err);
    CHECK(xw_finish(w));
    if (r == 1) {
        BOOL ok = X.CopyContext(d.ctx, XA_ALL | XA_XSTATE, x.ctx);
        CHECKF(ok, "CopyContext failed (%lu)", GetLastError());
        if (ok) {
            BOOL g; DWORD e2;
            CHECK_EQ(getmask(d.ctx, &g, &e2), getmask(x.ctx, &g, &e2));
            CHECK_MEM(X.LocateXStateFeature(d.ctx, XSTATE_AVX, NULL), X.LocateXStateFeature(x.ctx, XSTATE_AVX, NULL), 16 * t_vregs_count(T_VK_YMM));
            CHECK_MEM(X.LocateXStateFeature(d.ctx, 1, NULL), X.LocateXStateFeature(x.ctx, 1, NULL), 16 * t_vregs_count(T_VK_XMM));
        }
    }
    xctx_free(&x); xctx_free(&d); xw_free(w);
}

/* A plain CONTEXT (no CONTEXT_EX after it) ending exactly at a PAGE_NOACCESS page: a non-XSTATE request must
 * never write past sizeof(CONTEXT). Then a CONTEXT_DEBUG_REGISTERS-only request on an XSTATE-initialized
 * buffer must leave the XSTATE chunk alone (debug requests are separate from FP/vector requests). */
TEST(thread_get_plain_and_debug_only) {
    need_core_api();
    xw *w = xw_start(T_VK_XMM, 0x510A);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx p = { 0 }, y = { 0 };
    p.buf = (BYTE *)t_guarded_alloc(sizeof(CONTEXT), &p.base);
    int have_y = os_xstate() && xctx_new(&y, XA_ALL | XA_XSTATE);
    if (!p.buf) { xw_finish(w); xw_free(w); xctx_free(&y); REQUIREF(0, "guarded alloc failed"); }
    p.len = sizeof(CONTEXT); p.ctx = (CONTEXT *)p.buf;
    memset(p.buf, 0, sizeof(CONTEXT));
    p.ctx->ContextFlags = XA_ALL;
    DWORD err = 0;
    int r = xw_capture(w, &p, ~0ull, &err);
    CHECKF(r == 1, "GetThreadContext(CONTEXT_ALL) into an exact-size CONTEXT: result %d error %lu", r, err);
    if (r == 1) {
        CHECK_EQ(p.ctx->ContextFlags, XA_ALL);
        char lbl[32];
        for (int i = 0; i < t_vregs_count(T_VK_XMM); i++) {
            snprintf(lbl, sizeof lbl, "plain ctx xmm%d", i);
            CHECK_REG(lbl, p.buf + XA_LEG1_OFF + 16 * i, w->in.v[i], 16);
        }
        if (have_y) {
            DWORD tail = y.len - (DWORD)((BYTE *)y.ctx + sizeof(CONTEXT) - y.buf);
            BYTE *snap = (BYTE *)malloc(tail);
            if (snap) {
                BYTE *ch = xchunk(y.ctx);                  /* header stays as initialized (XSTATE_BV 0) */
                memset(ch + 64, 0x5A, (size_t)(y.buf + y.len - (ch + 64)));
                memcpy(snap, (BYTE *)y.ctx + sizeof(CONTEXT), tail);
                BYTE *fp = (BYTE *)y.ctx + XA_LEG0_OFF;
                memset(fp, 0x6B, 32);
                y.ctx->ContextFlags = XA_ARCH | 0x10;    /* CONTEXT_DEBUG_REGISTERS only */
                SetLastError(X_ERR_SENTINEL);
                BOOL ok = GetThreadContext(w->h, y.ctx);
                CHECKF(ok, "GetThreadContext(CONTEXT_DEBUG_REGISTERS) failed (%lu)", GetLastError());
                if (ok) {
                    CHECK_MEM((BYTE *)y.ctx + sizeof(CONTEXT), snap, tail);   /* CONTEXT_EX + XSTATE chunk untouched */
                    CHECK_EQ(y.ctx->Dr7, p.ctx->Dr7);
                    CHECK_EQ(y.ctx->Dr0, p.ctx->Dr0);
                    BYTE b6[32]; memset(b6, 0x6B, sizeof b6);
                    if (memcmp(fp, b6, 32))
                        t_note("debug-only GetThreadContext also wrote the FP area (unrequested part; not a documented contract)");
                }
                free(snap);
            }
        }
    }
    CHECK(xw_finish(w));
    t_guarded_free(p.base);
    xctx_free(&y); xw_free(w);
}

/* InitializeContext2(mask 3|AVX) gives a compacted context sized for x87/SSE/AVX only; the kernel copies the
 * thread's AVX state into it (ntoskrnl!RtlpWriteExtendedContext -> RtlpCopyXStateChunk). */
TEST(thread_get_into_init2_compact_context) {
    need_init2();
    need_avx_exec();
    xw *w = xw_start(T_VK_YMM, 0x510B);
    if (!w) SKIP("CreateThread failed (%lu)", GetLastError());
    xctx x;
    if (!xctx_new2(&x, XA_FP_ALL | XA_CONTROL | XA_XSTATE, 3 | T_XS_AVX, 1, 0)) { xw_finish(w); xw_free(w); REQUIREF(0, "InitializeContext2 failed"); }
    t_info("init2_avx_xcomp_bv", "0x%llx", (unsigned long long)xs_comp(x.ctx));
    DWORD err = 0;
    int r = xw_capture(w, &x, 3 | T_XS_AVX, &err);
    CHECKF(r == 1, "GetThreadContext into InitializeContext2(3|AVX) context: result %d error %lu", r, err);
    if (r == 1) {
        BOOL ok; DWORD e2;
        CHECKF(getmask(x.ctx, &ok, &e2) & T_XS_AVX, "mask lacks AVX although YMM uppers hold a pattern");
        DWORD l = 0;
        BYTE *hi = (BYTE *)X.LocateXStateFeature(x.ctx, XSTATE_AVX, &l);
        CHECKF(hi && l == 256 && in_buf(&x, hi, 256), "AVX area %p len %lu", (void *)hi, l);
        char lbl[40];
        for (int i = 0; hi && in_buf(&x, hi, 256) && i < t_vregs_count(T_VK_YMM); i++) {
            snprintf(lbl, sizeof lbl, "compact ctx ymm%d upper", i);
            CHECK_REG(lbl, hi + 16 * i, w->in.v[i] + 16, 16);
        }
    }
    CHECK(xw_finish(w));
    xctx_free(&x); xw_free(w);
}

/* Compacted InitializeContext2(mask 3) context (64-byte chunk) + request 3|AVX while the thread's YMM uppers
 * are in use: ntoskrnl!RtlpCopyXStateChunk finds no room -> STATUS_BUFFER_OVERFLOW -> FALSE/ERROR_MORE_DATA.
 * Hard requirement either way: nothing written past the guarded buffer and no AVX claimed without room. */
TEST(thread_get_compact_no_room) {
    need_init2();
    need_avx_exec();
    xctx small;
    REQUIRE(xctx_new2(&small, XA_FP_ALL | XA_CONTROL | XA_XSTATE, 3, 1, 0));
    if (!(xs_comp(small.ctx) & X_BIT63)) { xctx_free(&small); SKIP("standard XSAVE format: every XSTATE context has full size"); }
    xw *w = xw_start(T_VK_YMM, 0x510C);
    if (!w) { xctx_free(&small); SKIP("CreateThread failed (%lu)", GetLastError()); }
    xctx full;
    if (!xctx_new(&full, XA_FP_ALL | XA_CONTROL | XA_XSTATE)) { xw_finish(w); xw_free(w); xctx_free(&small); REQUIREF(0, "InitializeContext failed"); }
    DWORD err = 0;
    int r = xw_capture(w, &full, 3 | T_XS_AVX, &err);   /* suspends with the YMM pattern loaded */
    CHECKF(r == 1, "sync capture: result %d error %lu", r, err);
    if (r == 1) {
        X_CTX_EX ex0 = *xex(small.ctx);
        CHECK(X.SetXStateFeaturesMask(small.ctx, 3 | T_XS_AVX));
        SetLastError(X_ERR_SENTINEL);
        BOOL ok = GetThreadContext(w->h, small.ctx);
        DWORD e = GetLastError();
        t_info("compact_no_room", "ok=%d err=%lu bv=0x%llx", ok, e, (unsigned long long)xs_bv(small.ctx));
        CHECKF(memcmp(xex(small.ctx), &ex0, sizeof ex0) == 0, "CONTEXT_EX rewritten by GetThreadContext");
        if (ok) {
            CHECKF(!(xs_bv(small.ctx) & T_XS_AVX), "GetThreadContext claims AVX (XSTATE_BV 0x%llx) in a chunk of %lu bytes",
                   (unsigned long long)xs_bv(small.ctx), xex(small.ctx)->XState.Length);
            t_note("GetThreadContext succeeded without AVX data (Windows 19041 returns ERROR_MORE_DATA)");
        } else {
#ifdef T_X64
            CHECKF(e == ERROR_MORE_DATA, "error %lu want ERROR_MORE_DATA (234, STATUS_BUFFER_OVERFLOW)", e);
#else
            t_note("WOW64 GetThreadContext without room for AVX: error %lu", e);
#endif
        }
    }
    CHECK(xw_finish(w));
    xctx_free(&full); xctx_free(&small); xw_free(w);
}

/* Child: XSTATE-flagged Get/SetThreadContext with a bogus CONTEXT_EX. ntoskrnl!RtlpReadExtendedContextLayout
 * requires Legacy.Offset == -sizeof(CONTEXT), Legacy.Length >= the legacy size and the XState chunk inside All,
 * else STATUS_INVALID_PARAMETER before anything is written; RtlpReadExtendedContext raises
 * STATUS_DATATYPE_MISALIGNMENT (ERROR_NOACCESS 998) for a chunk that is not 64-byte aligned. */
CHILD(bogus_ctxex) {
    (void)argc; (void)argv;
    xa_resolve();
    if (!X.InitializeContext || !os_xstate()) { printf("RESULT=skip\n"); return 0; }
    xw *w = xw_start(T_VK_XMM, 0x510D);
    if (!w) { printf("RESULT=nothread\n"); return 0; }
    const DWORD fl = XA_FP_ALL | XA_CONTROL | XA_XSTATE;
    xctx s, a, b;
    if (!xctx_new(&s, fl) || !xctx_new(&a, fl) || !xctx_new(&b, fl)) { printf("RESULT=initfail\n"); xw_finish(w); return 0; }
    DWORD err = 0;
    int r = xw_capture(w, &s, ~0ull, &err);
    printf("GET=%d\nGET_ERR=%lu\n", r, (unsigned long)err);
    if (r == 1) {
        memcpy(a.ctx, s.ctx, sizeof(CONTEXT));                 /* valid legacy part: an accepted Set cannot derail the thread */
        memcpy(b.ctx, s.ctx, sizeof(CONTEXT));
        memset(xex(a.ctx), 0, sizeof(X_CTX_EX));              /* V1: zeroed CONTEXT_EX */
        BYTE *snap = (BYTE *)malloc(a.len);
        if (snap) memcpy(snap, a.buf, a.len);
        SetLastError(0);
        BOOL ok = GetThreadContext(w->h, a.ctx);
        printf("V1_RET=%d\nV1_ERR=%lu\nV1_UNCHANGED=%d\n", ok, GetLastError(), snap ? memcmp(snap, a.buf, a.len) == 0 : -1);
        free(snap);
        X_CTX_EX *e = xex(b.ctx);                              /* V2: chunk moved off 64-byte alignment */
        e->XState.Offset += 8; e->XState.Length -= 8;
        SetLastError(0);
        ok = GetThreadContext(w->h, b.ctx);
        printf("V2_RET=%d\nV2_ERR=%lu\n", ok, GetLastError());
        a.ctx->ContextFlags = fl;                              /* V3: SetThreadContext with the zeroed CONTEXT_EX */
        SetLastError(0);
        ok = SetThreadContext(w->h, a.ctx);
        printf("V3_RET=%d\nV3_ERR=%lu\n", ok, GetLastError());
    }
    fflush(stdout);
    int fin = xw_finish(w);
    printf("THREAD_FINISHED=%d\n", fin);
    printf("REGS_OK=%d\n", fin && t_vregs_check(&w->out, &w->in, T_VK_XMM, 0, -1, "worker after bogus contexts") == 0);
    printf("RESULT=done\n");
    return 0;
}

static t_child_result g_bogus;
static int g_bogus_ran;
static void bogus_run(void) {
    need_xstate_ctx();
    if (!g_bogus_ran) {
        REQUIREF(t_child_run("bogus_ctxex", NULL, 15000, &g_bogus), "spawn failed: error %lu", g_bogus.spawn_error);
        g_bogus_ran = 1;
    }
    char v[64];
    if (t_child_find(&g_bogus, "RESULT=", v, sizeof v) && strcmp(v, "skip") == 0) SKIP("child: XSTATE unavailable");
}
/* Robustness half (hard everywhere): the process and the target thread survive and keep their registers. */
TEST(thread_bogus_context_ex_child) {
    bogus_run();
    const t_child_result *r = &g_bogus;
    CHECKF(!r->timed_out, "child timed out");
    CHECKF(r->exc_code == 0 && r->exit_code == 0, "child exit 0x%lx exception 0x%lx at 0x%llx; output:\n%s",
           (unsigned long)r->exit_code, (unsigned long)r->exc_code, r->exc_addr, r->out);
    char v[64];
    REQUIREF(t_child_find(r, "GET=", v, sizeof v) && strcmp(v, "1") == 0, "worker never captured; output:\n%s", r->out);
    CHECKF(t_child_find(r, "THREAD_FINISHED=", v, sizeof v) && strcmp(v, "1") == 0, "target thread did not finish");
    CHECKF(t_child_find(r, "REGS_OK=", v, sizeof v) && strcmp(v, "1") == 0, "worker registers changed by bogus context calls");
}
/* Contract half: the bogus layouts are rejected with the pinned status. */
TEST(thread_bogus_context_ex_rejected) {
    bogus_run();
    const t_child_result *r = &g_bogus;
    char v[64];
    REQUIREF(t_child_find(r, "GET=", v, sizeof v) && strcmp(v, "1") == 0, "worker never captured; output:\n%s", r->out);
#ifdef T_X64   /* pinned from ntoskrnl 19041 (RtlpReadExtendedContextLayout / RtlpReadExtendedContext) */
    wine_known_gap("Get/SetThreadContext do not validate the CONTEXT_EX layout or XSTATE chunk alignment");
    CHECKF(t_child_find(r, "V1_RET=", v, sizeof v) && strcmp(v, "0") == 0, "GetThreadContext with zeroed CONTEXT_EX succeeded");
    CHECKF(t_child_find(r, "V1_UNCHANGED=", v, sizeof v) && strcmp(v, "1") == 0, "rejected GetThreadContext still wrote the buffer");
    CHECKF(t_child_find(r, "V2_RET=", v, sizeof v) && strcmp(v, "0") == 0, "GetThreadContext with a misaligned XSTATE chunk succeeded");
    CHECKF(t_child_find(r, "V3_RET=", v, sizeof v) && strcmp(v, "0") == 0, "SetThreadContext with zeroed CONTEXT_EX succeeded");
    CHECKF(t_child_find(r, "V1_ERR=", v, sizeof v) && strcmp(v, "87") == 0, "zeroed CONTEXT_EX: error %s want 87", v);
    CHECKF(t_child_find(r, "V2_ERR=", v, sizeof v) && strcmp(v, "998") == 0, "misaligned chunk: error %s want 998 (STATUS_DATATYPE_MISALIGNMENT)", v);
    CHECKF(t_child_find(r, "V3_ERR=", v, sizeof v) && strcmp(v, "87") == 0, "SetThreadContext zeroed CONTEXT_EX: error %s want 87", v);
#else
    /* WOW64 converts i386 contexts in wow64.dll (not in the reference corpus): record, do not pin. */
    static const char *keys[] = { "V1_RET", "V1_ERR", "V1_UNCHANGED", "V2_RET", "V2_ERR", "V3_RET", "V3_ERR" };
    char pre[32];
    for (size_t k = 0; k < T_COUNTOF(keys); k++) {
        snprintf(pre, sizeof pre, "%s=", keys[k]);
        if (t_child_find(r, pre, v, sizeof v)) t_info(keys[k], "%s", v);
    }
    t_note("i386 (WOW64) bogus CONTEXT_EX results recorded only; the native x64 build pins them");
#endif
}

/* Child: SetThreadContext with a corrupted XSAVE header (reserved bytes, unsupported XSTATE_BV bits).
 * The OS must sanitize or reject it; the process and the target thread must survive. */
CHILD(set_garbage_header) {
    (void)argc; (void)argv;
    xa_resolve();
    if (!X.InitializeContext || !X.LocateXStateFeature) { printf("RESULT=skip\n"); return 0; }
    xw *w = xw_start(T_VK_YMM, 0x5109);
    if (!w) { printf("RESULT=nothread\n"); return 0; }
    DWORD len = 0; PCONTEXT c = NULL;
    X.InitializeContext(NULL, XA_FP_ALL | XA_XSTATE, &c, &len);
    BYTE *buf = (BYTE *)malloc(len);
    if (!buf || !X.InitializeContext(buf, XA_FP_ALL | XA_XSTATE, &c, &len)) { printf("RESULT=initfail\n"); xw_finish(w); return 0; }
    xctx x = { buf, NULL, len, c };
    DWORD err = 0;
    int r = xw_capture(w, &x, 3 | T_XS_AVX, &err);
    printf("GET=%d\n", r);
    if (r == 1) {
        BYTE *ch = xchunk(c);
        uint64_t en = X.GetEnabledXStateFeatures();
        *(uint64_t *)ch |= (~en & 0x7FFFFFFFFFFFFFF8ull);
        memset(ch + 16, 0xFF, 48);
        SetLastError(0);
        BOOL ok = SetThreadContext(w->h, c);
        printf("SET=%d ERR=%lu\n", ok, GetLastError());
    }
    fflush(stdout);
    int fin = xw_finish(w);
    printf("THREAD_FINISHED=%d\n", fin);
    printf("RESULT=done\n");
    return 0;
}

TEST(thread_set_garbage_header_child) {
    need_avx_exec();
    static t_child_result r;
    REQUIREF(t_child_run("set_garbage_header", NULL, 15000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECKF(!r.timed_out, "child timed out");
    CHECKF(r.exc_code == 0, "child died of exception 0x%lx at 0x%llx", (unsigned long)r.exc_code, r.exc_addr);
    CHECKF(r.exit_code == 0, "child exit code 0x%lx", (unsigned long)r.exit_code);
    char v[64];
    if (t_child_find(&r, "RESULT=", v, sizeof v) && strcmp(v, "skip") == 0) SKIP("child: API unavailable");
    CHECKF(t_child_find(&r, "THREAD_FINISHED=", v, sizeof v) && strcmp(v, "1") == 0, "target thread did not finish normally; output:\n%s", r.out);
    if (t_child_find(&r, "SET=", v, sizeof v)) t_note("SetThreadContext with corrupted XSAVE header: %s", v);
}

/* ======================= J. API reporting vs execution ===================== */
T_TARGET("avx") T_NOINLINE static int exec_avx_add_ymm(void) {
    __m256 a = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8), b = _mm256_set1_ps(0.5f);
    float out[8];
    _mm256_storeu_ps(out, _mm256_add_ps(a, b));
    _mm256_zeroupper();
    for (int i = 0; i < 8; i++) if (out[i] != (float)(i + 1) + 0.5f) return 0;
    return 1;
}
T_TARGET("avx512f") T_NOINLINE static int exec_avx512_add_zmm(void) {
    __m512i a = _mm512_set1_epi32(40), b = _mm512_set1_epi32(2);
    int out[16];
    _mm512_storeu_si512(out, _mm512_add_epi32(a, b));
    _mm256_zeroupper();
    for (int i = 0; i < 16; i++) if (out[i] != 42) return 0;
    return 1;
}

TEST(exec_avx_matches_api) {
    DWORD64 en = enabled();
    if (!cpu_hw(F_AVX)) SKIP("CPU does not advertise AVX");
    int api = (en & T_XS_AVX) != 0, usable = cpu_has(F_AVX);
    t_info("avx_api", "%d", api); t_info("avx_usable", "%d", usable);
    if (!usable) {
        CHECKF(!api, "API reports AVX state but XCR0 does not enable it");
        SKIP("AVX advertised but not enabled by the OS (XCR0[2:1] != 11b)");
    }
    CHECKF(exec_avx_add_ymm(), "AVX instruction produced a wrong result");
    CHECKF(api, "AVX executes but GetEnabledXStateFeatures lacks the AVX bit (context APIs cannot reach YMM)");
}

TEST(exec_avx512_matches_api) {
    DWORD64 en = enabled();
    if (!cpu_hw(F_AVX512F)) SKIP("CPU does not advertise AVX-512F");
    int api = (en & 0xE0) == 0xE0, usable = cpu_has(F_AVX512F);
    t_info("avx512_api", "%d", api); t_info("avx512_usable", "%d", usable);
    if (!usable) {
        CHECKF((en & 0xE0) == 0, "API reports AVX-512 state 0x%llx but XCR0 does not enable it", (unsigned long long)(en & 0xE0));
        SKIP("AVX-512 advertised but not enabled by the OS");
    }
    int exec_ok = exec_avx512_add_zmm();
    CHECKF(exec_ok, "AVX-512 instruction produced a wrong result");
    if (exec_ok && !api && t_is_wine())   /* known Wine gap (seen with Wine 9 on an AVX-512 CPU model): AVX-512 runs but the API does not report it */
        t_expect_fail("Wine %s: GetEnabledXStateFeatures does not report AVX-512 state", t_wine_version());
    CHECKF(api, "AVX-512 executes but GetEnabledXStateFeatures lacks 0xE0 (got 0x%llx)", (unsigned long long)en);
}

TEST(locate_avx512_components) {
    DWORD64 en = enabled();
    if ((en & 0xE0) != 0xE0) SKIP("AVX-512 state not reported by the API");
    const t_cpu_info *c = cpu();
    xctx x; XCTX_REQ(&x, XA_ALL | XA_XSTATE);
    static const struct { int id; uint32_t size; } want[] = { { 5, 64 }, { 6, 512 }, { 7, 1024 } };
    for (size_t i = 0; i < T_COUNTOF(want); i++) {
        DWORD l = 0;
        void *p = X.LocateXStateFeature(x.ctx, (DWORD)want[i].id, &l);
        CHECKF(p != NULL, "feature %d not located", want[i].id);
        CHECKF(l == want[i].size, "feature %d length %lu want %u", want[i].id, l, want[i].size);
        CHECKF(l == c->comp[want[i].id].size, "feature %d length %lu != CPUID %u", want[i].id, l, c->comp[want[i].id].size);
        if (p) CHECK(in_buf(&x, p, l));
    }
    xctx_free(&x);
}

TEST(exec_xinuse_vs_context_mask) {
    need_avx_exec();
    if (!cpu_has(F_XGETBV1)) SKIP("XGETBV(1) not supported");
    /* After VZEROUPPER, XINUSE[2] may be 0 (init optimization): informational comparison only. */
    t_vzeroupper();
    uint64_t xin = t_xgetbv(1);
    t_info("xinuse_after_vzeroupper", "0x%llx", (unsigned long long)xin);
    CHECKF((xin & ~cpu()->xcr0) == 0, "XINUSE 0x%llx has bits outside XCR0 0x%llx", (unsigned long long)xin, (unsigned long long)cpu()->xcr0);
}

/* ======================= K. Windows 11 optional features ================== */
#define WIN11_API_REASON "(needs Windows 11, build 22000, or later)"
TEST(thread_enabled_features_win11) {
    DWORD64 en = enabled();
    if (!X.GetThreadEnabledXStateFeatures) SKIP("GetThreadEnabledXStateFeatures not exported " WIN11_API_REASON "; thread_enabled_features_oracle has the self-check");
    DWORD64 th = X.GetThreadEnabledXStateFeatures();
    t_info("thread_enabled_features", "0x%llx", (unsigned long long)th);
    CHECKF((th & 3) == 3, "thread features 0x%llx lack x87|SSE", (unsigned long long)th);
    CHECKF((th & ~en) == 0, "thread features 0x%llx exceed GetEnabledXStateFeatures 0x%llx", (unsigned long long)th, (unsigned long long)en);
    if (en & T_XS_AVX) CHECKF(th & T_XS_AVX, "AVX enabled system-wide but not for this thread (0x%llx)", (unsigned long long)th);
    if ((en & 0xE0) == 0xE0) CHECKF((th & 0xE0) == 0xE0, "AVX-512 enabled system-wide but not for this thread (0x%llx)", (unsigned long long)th);
    uint64_t opt = en & ~th;
    if (opt) t_note("features enabled system-wide but not yet for this thread (optional, e.g. AMX): 0x%llx", (unsigned long long)opt);
}

/* Oracle for GetThreadEnabledXStateFeatures that needs no Windows 11: the user components the OS enabled in
 * XCR0, minus those that support extended feature disable (CPUID.(0Dh,i):ECX[2], SDM vol. 1 13.2), which Windows
 * treats as optional and enables per process only through EnableProcessOptionalXStateFeatures (AMX TILEDATA
 * today). Before that call a thread must have every non-optional XCR0 component (it may already use them);
 * whether optional ones are already on is recorded (a process may have enabled them early). */
static uint64_t thread_enabled_oracle(uint64_t *optional) {
    const t_cpu_info *c = cpu();
    uint64_t user = c->osxsave ? c->xcr0 : 3, opt = 0;
    for (int i = 2; i < 64; i++) if (((user >> i) & 1) && c->comp[i].xfd) opt |= 1ull << i;
    /* AMX TILEDATA (18) is opt-in: TILECFG (17) then goes with it (TILECFG alone is not enabled per thread either) */
    if (opt & (1ull << 18)) opt |= T_XS_AMX_STATE & user;
    *optional = opt;
    return user & ~opt;
}
TEST(thread_enabled_features_oracle) {
    DWORD64 en = enabled();
    uint64_t opt = 0, orc = thread_enabled_oracle(&opt);
    /* XCR0 components GetEnabledXStateFeatures does not report (e.g. PKRU under a compatibility layer) are a
     * reporting discrepancy, recorded here and judged by the api_* cases; the oracle keeps only reported ones */
    if ((orc | opt) & ~en) t_note("XCR0 components 0x%llx not reported by GetEnabledXStateFeatures 0x%llx: excluded from the oracle",
                                  (unsigned long long)((orc | opt) & ~en), (unsigned long long)en);
    orc &= en | 3; opt &= en;
    t_info("oracle", "thread-enabled 0x%llx optional (XFD-capable, in XCR0) 0x%llx", (unsigned long long)orc, (unsigned long long)opt);
    CHECKF((orc & 3) == 3, "oracle 0x%llx lacks x87|SSE", (unsigned long long)orc);
    if (!X.GetThreadEnabledXStateFeatures) SKIP("GetThreadEnabledXStateFeatures not exported " WIN11_API_REASON "; oracle recorded");
    DWORD64 th = X.GetThreadEnabledXStateFeatures();
    t_info("thread_enabled_features", "0x%llx", (unsigned long long)th);
    CHECKF((orc & ~th) == 0, "GetThreadEnabledXStateFeatures 0x%llx lacks non-optional XCR0 components 0x%llx", (unsigned long long)th,
           (unsigned long long)(orc & ~th));
    uint64_t user_extra = th & ~(orc | opt) & cpu()->xcr0 & en;
    CHECKF(user_extra == 0, "thread features 0x%llx include user components 0x%llx outside the oracle", (unsigned long long)th,
           (unsigned long long)user_extra);
    if (th & opt) t_note("optional components 0x%llx already enabled for this thread before EnableProcessOptionalXStateFeatures",
                         (unsigned long long)(th & opt));
    if (th & ~cpu()->xcr0) t_note("thread features include non-XCR0 bits 0x%llx (supervisor components)", (unsigned long long)(th & ~cpu()->xcr0));
}
static DWORD WINAPI optional_newthread(void *p) { *(DWORD64 *)p = X.GetThreadEnabledXStateFeatures(); return 0; }

CHILD(enable_optional) {
    (void)argc; (void)argv;
    xa_resolve();
    if (!X.EnableProcessOptionalXStateFeatures || !X.GetThreadEnabledXStateFeatures || !X.GetEnabledXStateFeatures) {
        printf("RESULT=skip\n"); return 0;
    }
    DWORD64 en = X.GetEnabledXStateFeatures();
    printf("BEFORE=0x%llx\n", (unsigned long long)X.GetThreadEnabledXStateFeatures());
    SetLastError(0);
    BOOL z = X.EnableProcessOptionalXStateFeatures(0);
    printf("ZERO_RET=%d ZERO_ERR=%lu\n", z, GetLastError());
    DWORD64 req = en & T_XS_AMX_STATE;
    if (req) {
        SetLastError(0);
        BOOL ok = X.EnableProcessOptionalXStateFeatures(req);
        printf("AMX_RET=%d\nAMX_ERR=%lu\n", ok, GetLastError());
        printf("AFTER=0x%llx\n", (unsigned long long)X.GetThreadEnabledXStateFeatures());
        /* process-wide by name: a thread created afterwards must have the features too */
        DWORD64 nt = 0;
        HANDLE h = CreateThread(NULL, 64 * 1024, optional_newthread, &nt, 0, NULL);
        if (h) { if (WaitForSingleObject(h, 5000) == WAIT_OBJECT_0) printf("NEWTHREAD=0x%llx\n", (unsigned long long)nt); CloseHandle(h); }
    }
    SetLastError(0);
    BOOL bad = X.EnableProcessOptionalXStateFeatures(1ull << 62);
    printf("BOGUS_RET=%d BOGUS_ERR=%lu\n", bad, GetLastError());
    printf("RESULT=done\n");
    return 0;
}

TEST(enable_optional_features_child) {
    DWORD64 en = enabled();
    uint64_t opt = 0; (void)thread_enabled_oracle(&opt);
    t_info("optional_oracle", "XFD-capable XCR0 components 0x%llx", (unsigned long long)opt);
    if (!X.EnableProcessOptionalXStateFeatures)
        SKIP("EnableProcessOptionalXStateFeatures not exported " WIN11_API_REASON "%s", opt ? "; this CPU HAS optional components" : "");
    static t_child_result r;
    REQUIREF(t_child_run("enable_optional", NULL, 15000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(!r.timed_out);
    CHECKF(r.exc_code == 0 && r.exit_code == 0, "child exit 0x%lx exc 0x%lx", (unsigned long)r.exit_code, (unsigned long)r.exc_code);
    char v[128];
    if (t_child_find(&r, "BEFORE=", v, sizeof v)) t_info("optional_before", "%s", v);
    if (t_child_find(&r, "ZERO_RET=", v, sizeof v)) t_note("EnableProcessOptionalXStateFeatures(0): %s", v);
    if (t_child_find(&r, "BOGUS_RET=", v, sizeof v)) t_note("EnableProcessOptionalXStateFeatures(bit62): %s", v);
    if (en & T_XS_AMX_STATE) {
        CHECKF(t_child_find(&r, "AMX_RET=", v, sizeof v), "no AMX result; output:\n%s", r.out);
        if (strcmp(v, "1") == 0) {
            CHECKF(t_child_find(&r, "AFTER=", v, sizeof v), "no AFTER line");
            unsigned long long after = strtoull(v, NULL, 16);
            CHECKF((after & (en & T_XS_AMX_STATE)) == (en & T_XS_AMX_STATE), "AMX enabled (TRUE) but thread features 0x%llx lack it", after);
            if (t_child_find(&r, "NEWTHREAD=", v, sizeof v)) {
                unsigned long long nt = strtoull(v, NULL, 16);
                CHECKF((nt & (en & T_XS_AMX_STATE)) == (en & T_XS_AMX_STATE), "thread created after enabling AMX lacks it (0x%llx)", nt);
            } else CHECKF(0, "no NEWTHREAD line; output:\n%s", r.out);
        } else {
            t_note("EnableProcessOptionalXStateFeatures(AMX) returned %s", v);
        }
    } else {
        t_note("no optional (AMX) state enabled on this system: only parameter behaviour recorded");
    }
}
