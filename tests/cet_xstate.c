/* cet_xstate - CET user-mode shadow stacks (CET_U, IA32_XSS component 11) as seen through the Windows contract
 *
 * The exe is linked CET shadow-stack compatible (IMAGE_DLLCHARACTERISTICS_EX_CET_COMPAT in an
 * IMAGE_DEBUG_TYPE_EX_DLLCHARACTERISTICS debug-directory entry, what link /CETCOMPAT emits; see
 * T_CETCOMPAT_DEBUG_DIR below and tools/pe_cetcompat.py). Whether the OS then enforces shadow stacks is policy
 * (hardware, OS version, system/image mitigation settings): every enforcement case SKIPs unless RDSSP returns a
 * nonzero SSP in this process. CET user shadow stacks exist only for x64 processes on Windows: the x86 build
 * records the policy and SKIPs the rest.
 *
 * Contracts checked (Windows 10 22H2 19041 reference binaries, decompiled):
 *   - kernelbase!GetProcessMitigationPolicy: ProcessUserShadowStackPolicy (15) needs Length == 4, then
 *     NtQueryInformationProcess(ProcessMitigationPolicy); any other length -> FALSE, STATUS_INVALID_PARAMETER
 *     (ERROR_INVALID_PARAMETER).
 *   - kernelbase!InitializeContext2 / ntdll!RtlInitializeExtendedContext2: the XSTATE compaction mask is
 *     ANDed with (XState.EnabledFeatures | XState.EnabledSupervisorFeatures) from KUSER_SHARED_DATA, so CET_U
 *     (a user-visible supervisor feature, XSTATE_MASK_USER_VISIBLE_SUPERVISOR) can be requested.
 *     kernelbase!SetXStateFeaturesMask -> ntdll!RtlSetExtendedFeaturesMask uses the same mask.
 *     ntdll!RtlLocateExtendedFeature2 (behind LocateXStateFeature): compacted format locates a feature only
 *     when it is in the context's compaction mask. CET_U layout (SDM vol.1 13.5.6): IA32_U_CET, IA32_PL3_SSP.
 *   - ntoskrnl!PspGetSetContextInternal -> KeVerifyContextXStateCetU / KiVerifyContextXStateCetUEnabled
 *     (set path, CET_U present in the XSTATE mask): thread without shadow stack -> CET_U must be all zero;
 *     thread with shadow stack -> U_CET must be exactly 1 (SH_STK_EN), SSP 8-byte aligned, SSP >= the
 *     thread's current SSP (pop only) and <= its shadow-stack region end - 0x1000. Otherwise
 *     STATUS_SET_CONTEXT_DENIED (0xC000060A). CET_U absent from the mask -> SSP left as is.
 *   - ntoskrnl!KeVerifyContextIpForUserCet (SetThreadContext with CONTEXT_CONTROL, and NtContinue via
 *     KiVerifyContextRecord): RIP is validated only when the SetContextIpValidation mitigation is on. With it
 *     off, a RIP redirect is allowed. NtContinue (mode 0) also accepts any target whose image has no EH
 *     continuation table (RtlVerifyUserUnwindTarget: no load config / no IMAGE_GUARD_EH_CONTINUATION_TABLE_PRESENT)
 *     - this exe has none.
 *   - ntdll!RtlRestoreContext: with shadow stacks it restores through NtContinue and, if that returns
 *     STATUS_SET_CONTEXT_DENIED, __fastfail(FAST_FAIL_SET_CONTEXT_DENIED = 0x30) -> process exit 0xC0000409.
 *   - ntoskrnl!KiControlProtectionFault: a #CP from user mode (RET whose address differs from the shadow stack)
 *     ends in KiFastFailDispatch(STATUS_STACK_BUFFER_OVERRUN, FAST_FAIL_CONTROL_INVALID_RETURN_ADDRESS = 0x39)
 *     unless audit mode lets it continue. Handlers never see a fast fail.
 * Architecture (SDM vol.1 ch.17, vol.2 RDSSP/INCSSP): RDSSP is a NOP (destination unchanged) when shadow stacks
 * are not enabled; CALL pushes 8 bytes on the 64-bit shadow stack, RET pops and compares; XCR0[12:11] are
 * reserved (CET state is IA32_XSS only).
 *
 * Anything that could corrupt or terminate the process on a non-conforming system (context transfers, fiber
 * switches, return-address smash, CET_U edits) runs in a CHILD process. The harness itself is CET-safe: common/t.c
 * pops the shadow stack before its __builtin_longjmp (SKIP/REQUIRE/crash recovery). */
#include "t.h"

#ifndef XSTATE_CET_U
#define XSTATE_CET_U 11
#endif
#define CET_XS_MASK_CET_U (1ull << 11)
#define CET_XS_MASK_CET_S (1ull << 12)
#define CET_POLICY_CLASS ((PROCESS_MITIGATION_POLICY)15)   /* ProcessUserShadowStackPolicy */
#define CET_STATUS_SET_CONTEXT_DENIED 0xC000060Au
#define CET_STATUS_STACK_BUFFER_OVERRUN 0xC0000409u
#define CET_DEBUG_TYPE_EX_DLLCHARACTERISTICS 20
#define CET_DLLCHARACTERISTICS_EX_CET_COMPAT 0x01
#define CET_LAND_EXIT_THREAD 0x1A4Du

_Static_assert(ProcessUserShadowStackPolicy == 15, "ProcessUserShadowStackPolicy enum value");

/* IMAGE_DEBUG_DIRECTORY {Characteristics, TimeDateStamp, Major, Minor, Type=20, SizeOfData=4, AddressOfRawData,
 * PointerToRawData (filled by tools/pe_cetcompat.py)} + payload IMAGE_DLLCHARACTERISTICS_EX_CET_COMPAT. */
__asm__(".section .rdata,\"dr\"\n"
        ".balign 4\n"
        ".ascii \"XCR_CETCOMPAT_DD\"\n"
        "cet_xcr_debug_dir:\n"
        ".long 0, 0\n"
        ".short 0, 0\n"
        ".long 20, 4\n"
        ".rva cet_xcr_dllchar_ex\n"
        ".long 0\n"
        "cet_xcr_dllchar_ex:\n"
        ".long 1\n"
        ".text\n");

/* ---------------------------------------------------------------------------------------------------------- */
static inline uintptr_t cet_rdssp(void) {
    uintptr_t s = 0;
#ifdef T_X64
    __asm__ volatile("rdsspq %0" : "+r"(s));
#else
    __asm__ volatile("rdsspd %0" : "+r"(s));
#endif
    return s;
}
/* SSP as seen by the caller's frame (the call into this function pushed one entry). */
static T_NOINLINE uintptr_t cet_rdssp_callee(void) { uintptr_t s = cet_rdssp(); __asm__ volatile(""); return s; }

typedef struct cet_policy {
    int ok; DWORD gle; DWORD flags;
    int enable, audit, ipval, ipval_audit, strict, relaxed;
} cet_policy;

static void cet_get_policy(cet_policy *p) {
    memset(p, 0, sizeof *p);
    BOOL (WINAPI *gp)(HANDLE, PROCESS_MITIGATION_POLICY, PVOID, SIZE_T) =
        (BOOL (WINAPI *)(HANDLE, PROCESS_MITIGATION_POLICY, PVOID, SIZE_T))t_proc("kernel32.dll", "GetProcessMitigationPolicy");
    if (!gp) { p->gle = ERROR_PROC_NOT_FOUND; return; }
    PROCESS_MITIGATION_USER_SHADOW_STACK_POLICY pol;
    memset(&pol, 0, sizeof pol);
    SetLastError(0);
    p->ok = gp(GetCurrentProcess(), CET_POLICY_CLASS, &pol, sizeof pol) ? 1 : 0;
    p->gle = p->ok ? 0 : GetLastError();
    if (!p->ok) return;
    p->flags = pol.Flags;
    p->enable = pol.EnableUserShadowStack;
    p->audit = pol.AuditUserShadowStack;
    p->ipval = pol.SetContextIpValidation;
    p->ipval_audit = pol.AuditSetContextIpValidation;
    p->strict = pol.EnableUserShadowStackStrictMode;
    p->relaxed = pol.SetContextIpValidationRelaxedMode;
}

/* Shadow stacks active for this process (RDSSP nonzero). x86: never (Windows has no 32-bit user CET). */
static int cet_active(void) { return cet_rdssp() != 0; }

#define CET_NEED_ACTIVE() do { \
    if (!T_X64_ONLY) SKIP("CET user shadow stacks are x64-only on Windows"); \
    if (!cet_active()) { cet_policy p_; cet_get_policy(&p_); \
        SKIP("user shadow stack not active in this process (RDSSP=0; policy %s flags=0x%lx)", \
             p_.ok ? "ok" : "query failed", (unsigned long)p_.flags); } } while (0)
#ifdef T_X64
#define T_X64_ONLY 1
#else
#define T_X64_ONLY 0
#endif

static DWORD cet_status_to_win32(DWORD st) {
    ULONG (NTAPI *f)(LONG) = (ULONG (NTAPI *)(LONG))t_proc("ntdll.dll", "RtlNtStatusToDosError");
    return f ? f((LONG)st) : 0xFFFFFFFFu;
}

/* Own image: DllCharacteristicsEx from the debug directory, and whether a load config with an EH continuation
 * table exists (it changes which RIP targets NtContinue accepts). */
static int cet_image_flags(DWORD *dllchar_ex, int *ehcont) {
    *dllchar_ex = 0; *ehcont = 0;
    uint8_t *base = (uint8_t *)GetModuleHandleA(NULL);
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(base + ((IMAGE_DOS_HEADER *)base)->e_lfanew);
    IMAGE_DATA_DIRECTORY *dd = nt->OptionalHeader.DataDirectory;
    int found = 0;
    if (dd[IMAGE_DIRECTORY_ENTRY_DEBUG].VirtualAddress) {
        IMAGE_DEBUG_DIRECTORY *d = (IMAGE_DEBUG_DIRECTORY *)(base + dd[IMAGE_DIRECTORY_ENTRY_DEBUG].VirtualAddress);
        DWORD n = dd[IMAGE_DIRECTORY_ENTRY_DEBUG].Size / sizeof *d;
        for (DWORD i = 0; i < n; i++)
            if (d[i].Type == CET_DEBUG_TYPE_EX_DLLCHARACTERISTICS && d[i].SizeOfData >= 4 && d[i].AddressOfRawData) {
                *dllchar_ex = *(DWORD *)(base + d[i].AddressOfRawData);
                found = 1;
            }
    }
    if (dd[IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG].VirtualAddress) {
        /* GuardFlags: offset 0x90 (PE32+) / 0x58 (PE32); 0x00400000 = IMAGE_GUARD_EH_CONTINUATION_TABLE_PRESENT */
        const uint8_t *lc = base + dd[IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG].VirtualAddress;
        DWORD size = *(const DWORD *)lc, off = sizeof(void *) == 8 ? 0x90 : 0x58;
        if (size >= off + 4 && (*(const DWORD *)(lc + off) & 0x00400000u)) *ehcont = 1;
    }
    return found;
}

/* ============================================================================================================
 * x64 assembly helpers (leaf routines without stack adjustment: no unwind info needed)
 * ============================================================================================================ */
#ifdef T_X64
typedef struct cet_blk {
    T_ALIGN(64) uint8_t in[16][32];   /*    0 */
    uint8_t out[16][32];              /*  512 */
    uint8_t save[10][16];             /* 1024: caller's XMM6-15 */
    volatile long *stop;              /* 1184 */
    uint64_t ssp_before;              /* 1192 */
    uint64_t ssp_after;               /* 1200 */
    volatile uint64_t loops;          /* 1208 */
    volatile long ready;              /* 1216 */
} cet_blk;
_Static_assert(offsetof(cet_blk, out) == 512 && offsetof(cet_blk, save) == 1024 && offsetof(cet_blk, stop) == 1184 &&
               offsetof(cet_blk, ssp_before) == 1192 && offsetof(cet_blk, ssp_after) == 1200 &&
               offsetof(cet_blk, loops) == 1208 && offsetof(cet_blk, ready) == 1216, "cet_blk layout");

void cet_spin_xmm(cet_blk *b);      /* load in -> XMM0-15, RDSSP, spin until *stop, RDSSP, store XMM0-15 -> out */
void cet_spin_ymm(cet_blk *b);      /* same with YMM0-15 (AVX) */
void cet_fault_xmm(cet_blk *b);     /* load XMM0-15, RDSSP, UD2 (2 bytes), RDSSP, store */
void cet_landing(void);             /* context-transfer target: XMM0-15/MXCSR/SSP -> cet_land_*, cet_landing_finish() */
void cet_smash(void);               /* overwrite own return address with cet_smash_target, RET */
extern char cet_fault_ud2[];

T_ALIGN(16) uint8_t cet_land_xmm[16][16];
uint32_t cet_land_mxcsr;
uint64_t cet_land_ssp;
volatile long cet_land_mode;        /* 0: ExitThread(CET_LAND_EXIT_THREAD); 1: print + ExitProcess(0) */
void cet_landing_finish(void);
void cet_smash_survived(void);

#define CET_SAVE_6_15 \
    "movdqu %xmm6, 1024(%rcx)\n movdqu %xmm7, 1040(%rcx)\n movdqu %xmm8, 1056(%rcx)\n movdqu %xmm9, 1072(%rcx)\n" \
    "movdqu %xmm10, 1088(%rcx)\n movdqu %xmm11, 1104(%rcx)\n movdqu %xmm12, 1120(%rcx)\n movdqu %xmm13, 1136(%rcx)\n" \
    "movdqu %xmm14, 1152(%rcx)\n movdqu %xmm15, 1168(%rcx)\n"
#define CET_RESTORE_6_15 \
    "movdqu 1024(%rcx), %xmm6\n movdqu 1040(%rcx), %xmm7\n movdqu 1056(%rcx), %xmm8\n movdqu 1072(%rcx), %xmm9\n" \
    "movdqu 1088(%rcx), %xmm10\n movdqu 1104(%rcx), %xmm11\n movdqu 1120(%rcx), %xmm12\n movdqu 1136(%rcx), %xmm13\n" \
    "movdqu 1152(%rcx), %xmm14\n movdqu 1168(%rcx), %xmm15\n"
#define CET_XMM_IO(op, base) \
    op " " base "+0(%rcx), %xmm0\n " op " " base "+32(%rcx), %xmm1\n " op " " base "+64(%rcx), %xmm2\n " op " " base "+96(%rcx), %xmm3\n" \
    op " " base "+128(%rcx), %xmm4\n " op " " base "+160(%rcx), %xmm5\n " op " " base "+192(%rcx), %xmm6\n " op " " base "+224(%rcx), %xmm7\n" \
    op " " base "+256(%rcx), %xmm8\n " op " " base "+288(%rcx), %xmm9\n " op " " base "+320(%rcx), %xmm10\n " op " " base "+352(%rcx), %xmm11\n" \
    op " " base "+384(%rcx), %xmm12\n " op " " base "+416(%rcx), %xmm13\n " op " " base "+448(%rcx), %xmm14\n " op " " base "+480(%rcx), %xmm15\n"
#define CET_XMM_ST(op, base) \
    op " %xmm0, " base "+0(%rcx)\n " op " %xmm1, " base "+32(%rcx)\n " op " %xmm2, " base "+64(%rcx)\n " op " %xmm3, " base "+96(%rcx)\n" \
    op " %xmm4, " base "+128(%rcx)\n " op " %xmm5, " base "+160(%rcx)\n " op " %xmm6, " base "+192(%rcx)\n " op " %xmm7, " base "+224(%rcx)\n" \
    op " %xmm8, " base "+256(%rcx)\n " op " %xmm9, " base "+288(%rcx)\n " op " %xmm10, " base "+320(%rcx)\n " op " %xmm11, " base "+352(%rcx)\n" \
    op " %xmm12, " base "+384(%rcx)\n " op " %xmm13, " base "+416(%rcx)\n " op " %xmm14, " base "+448(%rcx)\n " op " %xmm15, " base "+480(%rcx)\n"
#define CET_YMM_IO(base) \
    "vmovdqu " base "+0(%rcx), %ymm0\n vmovdqu " base "+32(%rcx), %ymm1\n vmovdqu " base "+64(%rcx), %ymm2\n vmovdqu " base "+96(%rcx), %ymm3\n" \
    "vmovdqu " base "+128(%rcx), %ymm4\n vmovdqu " base "+160(%rcx), %ymm5\n vmovdqu " base "+192(%rcx), %ymm6\n vmovdqu " base "+224(%rcx), %ymm7\n" \
    "vmovdqu " base "+256(%rcx), %ymm8\n vmovdqu " base "+288(%rcx), %ymm9\n vmovdqu " base "+320(%rcx), %ymm10\n vmovdqu " base "+352(%rcx), %ymm11\n" \
    "vmovdqu " base "+384(%rcx), %ymm12\n vmovdqu " base "+416(%rcx), %ymm13\n vmovdqu " base "+448(%rcx), %ymm14\n vmovdqu " base "+480(%rcx), %ymm15\n"
#define CET_YMM_ST(base) \
    "vmovdqu %ymm0, " base "+0(%rcx)\n vmovdqu %ymm1, " base "+32(%rcx)\n vmovdqu %ymm2, " base "+64(%rcx)\n vmovdqu %ymm3, " base "+96(%rcx)\n" \
    "vmovdqu %ymm4, " base "+128(%rcx)\n vmovdqu %ymm5, " base "+160(%rcx)\n vmovdqu %ymm6, " base "+192(%rcx)\n vmovdqu %ymm7, " base "+224(%rcx)\n" \
    "vmovdqu %ymm8, " base "+256(%rcx)\n vmovdqu %ymm9, " base "+288(%rcx)\n vmovdqu %ymm10, " base "+320(%rcx)\n vmovdqu %ymm11, " base "+352(%rcx)\n" \
    "vmovdqu %ymm12, " base "+384(%rcx)\n vmovdqu %ymm13, " base "+416(%rcx)\n vmovdqu %ymm14, " base "+448(%rcx)\n vmovdqu %ymm15, " base "+480(%rcx)\n"
#define CET_SPIN_BODY \
    "xor %eax, %eax\n rdsspq %rax\n mov %rax, 1192(%rcx)\n" \
    "mov 1184(%rcx), %rdx\n movl $1, 1216(%rcx)\n" \
    "1: pause\n incq 1208(%rcx)\n cmpl $0, (%rdx)\n je 1b\n" \
    "xor %eax, %eax\n rdsspq %rax\n mov %rax, 1200(%rcx)\n"

__asm__(".text\n"
        ".globl cet_spin_xmm\n .p2align 4\ncet_spin_xmm:\n"
        CET_SAVE_6_15 CET_XMM_IO("movdqu", "0") CET_SPIN_BODY CET_XMM_ST("movdqu", "512") CET_RESTORE_6_15
        "ret\n"
        ".globl cet_spin_ymm\n .p2align 4\ncet_spin_ymm:\n"
        CET_SAVE_6_15 CET_YMM_IO("0") CET_SPIN_BODY CET_YMM_ST("512") "vzeroupper\n" CET_RESTORE_6_15
        "ret\n"
        ".globl cet_fault_xmm\n .p2align 4\ncet_fault_xmm:\n"
        CET_SAVE_6_15 CET_XMM_IO("movdqu", "0")
        "xor %eax, %eax\n rdsspq %rax\n mov %rax, 1192(%rcx)\n"
        ".globl cet_fault_ud2\ncet_fault_ud2:\n ud2\n"
        "xor %eax, %eax\n rdsspq %rax\n mov %rax, 1200(%rcx)\n"
        CET_XMM_ST("movdqu", "512") CET_RESTORE_6_15
        "ret\n"
        ".globl cet_landing\n .p2align 4\ncet_landing:\n"
        "movdqu %xmm0, cet_land_xmm+0(%rip)\n movdqu %xmm1, cet_land_xmm+16(%rip)\n movdqu %xmm2, cet_land_xmm+32(%rip)\n"
        "movdqu %xmm3, cet_land_xmm+48(%rip)\n movdqu %xmm4, cet_land_xmm+64(%rip)\n movdqu %xmm5, cet_land_xmm+80(%rip)\n"
        "movdqu %xmm6, cet_land_xmm+96(%rip)\n movdqu %xmm7, cet_land_xmm+112(%rip)\n movdqu %xmm8, cet_land_xmm+128(%rip)\n"
        "movdqu %xmm9, cet_land_xmm+144(%rip)\n movdqu %xmm10, cet_land_xmm+160(%rip)\n movdqu %xmm11, cet_land_xmm+176(%rip)\n"
        "movdqu %xmm12, cet_land_xmm+192(%rip)\n movdqu %xmm13, cet_land_xmm+208(%rip)\n movdqu %xmm14, cet_land_xmm+224(%rip)\n"
        "movdqu %xmm15, cet_land_xmm+240(%rip)\n"
        "stmxcsr cet_land_mxcsr(%rip)\n"
        "xor %eax, %eax\n rdsspq %rax\n mov %rax, cet_land_ssp(%rip)\n"
        "and $-16, %rsp\n sub $32, %rsp\n call cet_landing_finish\n ud2\n"
        ".globl cet_smash\n .p2align 4\ncet_smash:\n"
        "lea cet_smash_target(%rip), %rax\n mov %rax, (%rsp)\n ret\n"
        "cet_smash_target:\n and $-16, %rsp\n sub $32, %rsp\n call cet_smash_survived\n ud2\n");

static void cet_pattern(cet_blk *b, uint32_t tag, int ymm) {
    memset(b, 0, sizeof *b);
    for (int i = 0; i < 16; i++) t_pattern_reg(b->in[i], ymm ? 32 : 16, tag, (unsigned)i);
}
static int cet_blk_check(const cet_blk *b, int ymm, const char *label) {
    int bad = 0;
    for (int i = 0; i < 16; i++) {
        char l[96]; snprintf(l, sizeof l, "%s %s%d", label, ymm ? "ymm" : "xmm", i);
        if (memcmp(b->out[i], b->in[i], ymm ? 32 : 16)) { CHECK_REG(l, b->out[i], b->in[i], ymm ? 32 : 16); bad++; }
    }
    return bad;
}
static int cet_use_ymm(void) {
    const t_env *e = t_env_get();
    return cpu_has(F_AVX) && e->api_xstate_present && (e->api_xstate_features & T_XS_AVX);
}

typedef struct cet_worker { cet_blk b; volatile long stop; int ymm; HANDLE h; } cet_worker;
static DWORD WINAPI cet_worker_proc(void *arg) {
    cet_worker *w = (cet_worker *)arg;
    w->b.stop = &w->stop;
    if (w->ymm) cet_spin_ymm(&w->b); else cet_spin_xmm(&w->b);
    return 0;
}
static int cet_worker_start(cet_worker *w, uint32_t tag, int ymm) {
    cet_pattern(&w->b, tag, ymm);
    w->stop = 0; w->ymm = ymm;
    w->h = CreateThread(NULL, 0, cet_worker_proc, w, 0, NULL);
    if (!w->h) return 0;
    for (int i = 0; i < 2000 && !w->b.ready; i++) Sleep(1);
    return w->b.ready != 0;
}
static DWORD cet_worker_join(cet_worker *w) {
    w->stop = 1;
    DWORD ec = 0xFFFFFFFFu;
    if (WaitForSingleObject(w->h, 5000) == WAIT_OBJECT_0) GetExitCodeThread(w->h, &ec);
    CloseHandle(w->h);
    return ec;
}

/* XSTATE context with CET_U (and AVX when usable) in its compaction mask. */
typedef struct cet_xctx { void *buf; DWORD len; CONTEXT *c; uint64_t mask; } cet_xctx;
typedef BOOL (WINAPI *cet_initctx2_t)(PVOID, DWORD, PCONTEXT *, PDWORD, ULONG64);
static const char *cet_xctx_init(cet_xctx *x, uint64_t mask) {
    memset(x, 0, sizeof *x);
    cet_initctx2_t ic2 = (cet_initctx2_t)t_proc("kernel32.dll", "InitializeContext2");
    BOOL (WINAPI *sxm)(PCONTEXT, DWORD64) = (BOOL (WINAPI *)(PCONTEXT, DWORD64))t_proc("kernel32.dll", "SetXStateFeaturesMask");
    if (!ic2 || !sxm) return "InitializeContext2/SetXStateFeaturesMask not exported";
    DWORD flags = CONTEXT_ALL | CONTEXT_XSTATE, len = 0;
    SetLastError(0);
    if (ic2(NULL, flags, NULL, &len, mask) || GetLastError() != ERROR_INSUFFICIENT_BUFFER || !len)
        return "InitializeContext2 size query did not report ERROR_INSUFFICIENT_BUFFER";
    x->buf = calloc(1, len + 64);
    if (!x->buf) return "out of memory";
    x->len = len;
    if (!ic2(x->buf, flags, &x->c, &len, mask)) return "InitializeContext2 failed";
    if (!sxm(x->c, mask)) return "SetXStateFeaturesMask failed";
    x->mask = mask;
    return NULL;
}
static void cet_xctx_request(cet_xctx *x) {
    BOOL (WINAPI *sxm)(PCONTEXT, DWORD64) = (BOOL (WINAPI *)(PCONTEXT, DWORD64))t_proc("kernel32.dll", "SetXStateFeaturesMask");
    if (sxm) sxm(x->c, x->mask);
}
static void cet_xctx_free(cet_xctx *x) { free(x->buf); x->buf = NULL; }
static uint8_t *cet_locate(cet_xctx *x, DWORD id, DWORD *len) {
    PVOID (WINAPI *loc)(PCONTEXT, DWORD, PDWORD) = (PVOID (WINAPI *)(PCONTEXT, DWORD, PDWORD))t_proc("kernel32.dll", "LocateXStateFeature");
    DWORD l = 0;
    uint8_t *p = loc ? (uint8_t *)loc(x->c, id, &l) : NULL;
    if (len) *len = l;
    if (p && (p < (uint8_t *)x->buf || p + l > (uint8_t *)x->buf + x->len + 64)) {
        CHECKF(0, "LocateXStateFeature(%lu) = %p len %lu lies outside the %lu-byte context buffer at %p",
               (unsigned long)id, (void *)p, (unsigned long)l, (unsigned long)x->len, x->buf);
        return NULL;
    }
    return p;
}
static uint64_t cet_xs_mask(cet_xctx *x) {
    DWORD64 m = 0;
    BOOL (WINAPI *gxm)(PCONTEXT, PDWORD64) = (BOOL (WINAPI *)(PCONTEXT, PDWORD64))t_proc("kernel32.dll", "GetXStateFeaturesMask");
    if (gxm && !gxm(x->c, &m)) m = 0;
    return m;
}
static uint64_t cet_api_features(void) {
    DWORD64 (WINAPI *g)(void) = (DWORD64 (WINAPI *)(void))t_proc("kernel32.dll", "GetEnabledXStateFeatures");
    return g ? g() : 0;
}
typedef LONG (NTAPI *cet_ntsetctx_t)(HANDLE, PCONTEXT);
#endif /* T_X64 */

/* ============================================================================================================
 * Cases that run everywhere
 * ============================================================================================================ */
TEST(cet_feature_reporting) {
    const t_cpu_info *c = cpu();
    uint64_t api = t_env_get()->api_xstate_features;
    t_info("cpuid_cet_ss", "%d", cpu_hw(F_CET_SS));
    t_info("cpuid_cet_ibt", "%d", cpu_hw(F_CET_IBT));
    t_info("xss_supported", "0x%llx", (unsigned long long)c->xss_supported);
    t_info("xss_cet_u_supported", "%d", (int)((c->xss_supported >> 11) & 1));
    t_info("xss_cet_s_supported", "%d", (int)((c->xss_supported >> 12) & 1));
    t_info("api_xstate_present", "%d", t_env_get()->api_xstate_present);
    t_info("api_cet_u", "%d", (int)((api >> 11) & 1));
    t_info("api_cet_s", "%d", (int)((api >> 12) & 1));
    ULONG64 (NTAPI *rge)(ULONG64) = (ULONG64 (NTAPI *)(ULONG64))t_proc("ntdll.dll", "RtlGetEnabledExtendedFeatures");
    if (rge) t_info("RtlGetEnabledExtendedFeatures_CET_U", "0x%llx", (unsigned long long)rge(CET_XS_MASK_CET_U));
    /* SDM vol.1 13.3: XCR0[12:11] (CET_U/CET_S) are reserved, CET state is managed only through IA32_XSS. */
    if (c->xcr0_valid) {
        CHECKF(!(c->xcr0 & (CET_XS_MASK_CET_U | CET_XS_MASK_CET_S)), "XCR0=0x%llx has CET bits (IA32_XSS-only components)",
               (unsigned long long)c->xcr0);
        CHECKF(!(c->xcr0_supported & (CET_XS_MASK_CET_U | CET_XS_MASK_CET_S)),
               "CPUID(0Dh,0) EDX:EAX=0x%llx reports CET as XCR0-supported", (unsigned long long)c->xcr0_supported);
    }
    uintptr_t ssp = cet_rdssp();
    t_info("rdssp_main", "0x%llx", (unsigned long long)ssp);
    if (ssp) {
        /* RDSSP only returns a value with CR4.CET and SH_STK_EN set: the processor must enumerate CET_SS. */
        CHECKF(cpu_hw(F_CET_SS), "RDSSP returned 0x%llx but CPUID.(7,0):ECX[7] (CET_SS) is clear", (unsigned long long)ssp);
        if (!(api & CET_XS_MASK_CET_U))
            t_note("shadow stack active but GetEnabledXStateFeatures lacks CET_U: thread contexts cannot carry the SSP");
    }
    if ((api & CET_XS_MASK_CET_U) && !cpu_hw(F_CET_SS))
        t_note("GetEnabledXStateFeatures reports CET_U but CPUID does not enumerate CET_SS");
    DWORD ex; int ehcont;
    int found = cet_image_flags(&ex, &ehcont);
    t_info("image_dllcharacteristics_ex", "%s0x%lx", found ? "" : "(none) ", (unsigned long)ex);
    t_info("image_ehcont_table", "%d", ehcont);
    /* Build contract of this exe (Makefile CETCOMPAT + tools/pe_cetcompat.py), not an OS check. */
    CHECKF(found && (ex & CET_DLLCHARACTERISTICS_EX_CET_COMPAT), "exe not marked IMAGE_DLLCHARACTERISTICS_EX_CET_COMPAT");
}

TEST(cet_mitigation_policy) {
    cet_policy p; cet_get_policy(&p);
    t_info("policy_query", "%s", p.ok ? "ok" : "failed");
    if (!p.ok) {
        t_info("policy_gle", "%lu", (unsigned long)p.gle);
        if (p.gle == ERROR_PROC_NOT_FOUND) SKIP("GetProcessMitigationPolicy not exported");
        if (cet_active()) CHECKF(0, "shadow stack active but GetProcessMitigationPolicy(ProcessUserShadowStackPolicy) failed, GLE=%lu",
                                 (unsigned long)p.gle);
        if (p.gle == ERROR_INVALID_PARAMETER || p.gle == ERROR_NOT_SUPPORTED || p.gle == ERROR_CALL_NOT_IMPLEMENTED)
            SKIP("ProcessUserShadowStackPolicy not supported by this OS (GLE=%lu)", (unsigned long)p.gle);
        CHECKF(0, "GetProcessMitigationPolicy(ProcessUserShadowStackPolicy) failed with unexpected GLE=%lu", (unsigned long)p.gle);
        return;
    }
    t_info("policy_flags", "0x%lx", (unsigned long)p.flags);
    t_info("EnableUserShadowStack", "%d", p.enable);
    t_info("AuditUserShadowStack", "%d", p.audit);
    t_info("SetContextIpValidation", "%d", p.ipval);
    t_info("AuditSetContextIpValidation", "%d", p.ipval_audit);
    t_info("EnableUserShadowStackStrictMode", "%d", p.strict);
    t_info("SetContextIpValidationRelaxedMode", "%d", p.relaxed);
    int active = cet_active();
    t_info("rdssp_active", "%d", active);
#ifdef T_X64
    /* Policy and hardware state must agree: an enabled policy gives the main thread a shadow stack, and RDSSP
     * cannot return an SSP in a process whose policy says shadow stacks are off. */
    CHECKF(!!p.enable == !!active, "EnableUserShadowStack=%d but RDSSP %s", p.enable, active ? "returns an SSP" : "returns 0");
#else
    CHECKF(!active, "RDSSPD returned an SSP in a 32-bit process");
#endif
    /* kernelbase!GetProcessMitigationPolicy: this class requires Length == 4. */
    BOOL (WINAPI *gp)(HANDLE, PROCESS_MITIGATION_POLICY, PVOID, SIZE_T) =
        (BOOL (WINAPI *)(HANDLE, PROCESS_MITIGATION_POLICY, PVOID, SIZE_T))t_proc("kernel32.dll", "GetProcessMitigationPolicy");
    uint64_t big = 0xA5A5A5A5A5A5A5A5ull;
    SetLastError(0);
    BOOL r = gp(GetCurrentProcess(), CET_POLICY_CLASS, &big, sizeof big);
    DWORD gle = GetLastError();
    if (t_is_wine() && r) t_expect_fail("Wine does not validate the policy buffer length (%s)", t_wine_version());
    CHECKF(!r && gle == ERROR_INVALID_PARAMETER, "Length 8: ret=%d GLE=%lu, want FALSE/ERROR_INVALID_PARAMETER", r, (unsigned long)gle);
}

/* ============================================================================================================
 * Enforcement cases (x64, shadow stack active)
 * ============================================================================================================ */
#ifdef T_X64
typedef struct cet_tinfo { uintptr_t ssp[6]; uintptr_t depth_outer, depth_inner; void *alloc; } cet_tinfo;
static T_NOINLINE void cet_depth_probe(cet_tinfo *t) {
    t->depth_outer = cet_rdssp();
    t->depth_inner = cet_rdssp_callee();
}
static DWORD WINAPI cet_thread_ssp_proc(void *arg) {
    cet_tinfo *t = (cet_tinfo *)arg;
    HANDLE ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    t->ssp[0] = cet_rdssp();
    Sleep(0);
    t->ssp[1] = cet_rdssp();
    SwitchToThread();
    t->ssp[2] = cet_rdssp();
    if (ev) WaitForSingleObject(ev, 2);
    t->ssp[3] = cet_rdssp();
    Sleep(3);
    t->ssp[4] = cet_rdssp();
    cet_depth_probe(t);
    t->ssp[5] = cet_rdssp();
    MEMORY_BASIC_INFORMATION mbi;
    if (t->ssp[0] && VirtualQuery((void *)t->ssp[0], &mbi, sizeof mbi)) t->alloc = mbi.AllocationBase;
    if (ev) CloseHandle(ev);
    return 0;
}

TEST(cet_rdssp_per_thread) {
    CET_NEED_ACTIVE();
    enum { N = 8 };
    static cet_tinfo ti[N + 1];
    memset(ti, 0, sizeof ti);
    HANDLE h[N]; int n = 0;
    for (int i = 0; i < N; i++) {
        h[i] = CreateThread(NULL, 0, cet_thread_ssp_proc, &ti[i], 0, NULL);
        if (!h[i]) break;
        n++;
    }
    if (n < 2) { for (int i = 0; i < n; i++) { WaitForSingleObject(h[i], 5000); CloseHandle(h[i]); } SKIP("could not create threads"); }
    cet_thread_ssp_proc(&ti[N]);     /* main (case) thread too */
    for (int i = 0; i < n; i++) { CHECK_EQ(WaitForSingleObject(h[i], 5000), WAIT_OBJECT_0); CloseHandle(h[i]); }
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery((void *)ti[N].ssp[0], &mbi, sizeof mbi)) {
        t_info("ssp_region_protect", "0x%lx", (unsigned long)mbi.Protect);
        t_info("ssp_region_type", "0x%lx", (unsigned long)mbi.Type);
    }
    for (int i = 0; i <= N; i++) {
        if (i < N && i >= n) continue;
        cet_tinfo *t = &ti[i];
        CHECKF(t->ssp[0] != 0, "thread %d: RDSSP returned 0 (new threads of a shadow-stack process get a shadow stack)", i);
        CHECKF((t->ssp[0] & 7) == 0, "thread %d: SSP 0x%llx not 8-byte aligned", i, (unsigned long long)t->ssp[0]);
        static const char *what[6] = {"start", "Sleep(0)", "SwitchToThread", "WaitForSingleObject timeout", "Sleep(3)", "nested calls"};
        for (int k = 1; k < 6; k++)
            CHECKF(t->ssp[k] == t->ssp[0], "thread %d: SSP after %s 0x%llx != 0x%llx", i, what[k],
                   (unsigned long long)t->ssp[k], (unsigned long long)t->ssp[0]);
        /* SDM: a near CALL in 64-bit mode pushes 8 bytes onto the shadow stack. */
        CHECKF(t->depth_inner == t->depth_outer - 8, "thread %d: callee SSP 0x%llx, caller 0x%llx (want caller-8)", i,
               (unsigned long long)t->depth_inner, (unsigned long long)t->depth_outer);
        for (int j = 0; j < i; j++) {
            if (j < N && j >= n) continue;
            CHECKF(ti[j].ssp[0] != t->ssp[0], "threads %d and %d share SSP 0x%llx", j, i, (unsigned long long)t->ssp[0]);
            CHECKF(ti[j].alloc != t->alloc || !t->alloc, "threads %d and %d share the shadow-stack allocation %p", j, i, t->alloc);
        }
    }
}

/* Suspended spinning threads: GetThreadContext with CET_U returns the SSP each thread read with RDSSP. */
TEST(cet_getcontext_ssp_matches_rdssp) {
    CET_NEED_ACTIVE();
    uint64_t api = cet_api_features();
    REQUIREF(api & CET_XS_MASK_CET_U, "shadow stack active but GetEnabledXStateFeatures()=0x%llx lacks CET_U (bit 11)",
             (unsigned long long)api);
    int ymm = cet_use_ymm();
    uint64_t mask = 3 | CET_XS_MASK_CET_U | (ymm ? T_XS_AVX : 0);
    enum { N = 4 };
    static cet_worker w[N];
    int n = 0;
    for (int i = 0; i < N; i++) { if (!cet_worker_start(&w[i], 0xCE70 + i, ymm)) { if (w[i].h) cet_worker_join(&w[i]); break; } n++; }
    if (!n) SKIP("could not start worker threads");
    for (int i = 0; i < n; i++) {
        cet_xctx x;
        const char *err = cet_xctx_init(&x, mask);
        if (err) { CHECKF(0, "%s (GLE=%lu)", err, (unsigned long)GetLastError()); cet_xctx_free(&x); break; }
        CHECK_NE(SuspendThread(w[i].h), (DWORD)-1);
        BOOL ok = GetThreadContext(w[i].h, x.c);
        CHECKF(ok, "worker %d: GetThreadContext(CONTEXT_ALL|CONTEXT_XSTATE, CET_U) GLE=%lu", i, (unsigned long)GetLastError());
        if (ok) {
            uint64_t m = cet_xs_mask(&x);
            CHECKF(m & CET_XS_MASK_CET_U, "worker %d: XSTATE mask 0x%llx after GetThreadContext lacks CET_U", i, (unsigned long long)m);
            DWORD l = 0;
            uint8_t *p = cet_locate(&x, XSTATE_CET_U, &l);
            CHECKF(p != NULL, "worker %d: LocateXStateFeature(XSTATE_CET_U) returned NULL", i);
            if (p) {
                uint64_t ucet, ssp;
                memcpy(&ucet, p, 8); memcpy(&ssp, p + 8, 8);
                if (i == 0) { t_info("cet_u_len", "%lu", (unsigned long)l); t_info("u_cet", "0x%llx", (unsigned long long)ucet); }
                CHECKF(l >= 16, "worker %d: CET_U length %lu < 16", i, (unsigned long)l);
                CHECKF(ucet & 1, "worker %d: IA32_U_CET=0x%llx lacks SH_STK_EN", i, (unsigned long long)ucet);
                CHECKF(ssp == w[i].b.ssp_before, "worker %d: context SSP 0x%llx != RDSSP 0x%llx", i,
                       (unsigned long long)ssp, (unsigned long long)w[i].b.ssp_before);
                for (int j = 0; j < i; j++)
                    CHECKF(ssp != w[j].b.ssp_before, "workers %d and %d report the same SSP", j, i);
            }
            /* SSE/AVX state of the same context */
            for (int r = 0; r < 16; r++) {
                char lb[64]; snprintf(lb, sizeof lb, "worker %d context xmm%d", i, r);
                CHECK_REG(lb, &x.c->FltSave.XmmRegisters[r], w[i].b.in[r], 16);
            }
            if (ymm) {
                uint8_t *hi = cet_locate(&x, XSTATE_AVX, NULL);
                CHECKF(hi != NULL, "worker %d: LocateXStateFeature(XSTATE_AVX) NULL", i);
                for (int r = 0; hi && r < 16; r++) {
                    char lb[64]; snprintf(lb, sizeof lb, "worker %d context ymm%d[255:128]", i, r);
                    CHECK_REG(lb, hi + 16 * r, w[i].b.in[r] + 16, 16);
                }
            }
            /* Unchanged round trip through SetThreadContext (CET_U included) must be accepted. */
            if (i == 0) {
                BOOL s = SetThreadContext(w[i].h, x.c);
                CHECKF(s, "SetThreadContext with the unchanged CET_U context failed GLE=%lu", (unsigned long)GetLastError());
            }
        }
        CHECK_NE(ResumeThread(w[i].h), (DWORD)-1);
        cet_xctx_free(&x);
    }
    for (int i = 0; i < n; i++) {
        CHECK_EQ(cet_worker_join(&w[i]), 0);
        cet_blk_check(&w[i].b, ymm, "after suspend/GetThreadContext/resume");
        CHECKF(w[i].b.ssp_after == w[i].b.ssp_before, "worker %d: SSP changed while suspended 0x%llx -> 0x%llx", i,
               (unsigned long long)w[i].b.ssp_before, (unsigned long long)w[i].b.ssp_after);
    }
}

/* ---- children ----------------------------------------------------------------------------------------- */
/* Children run only with an active shadow stack, except with the "noss" argument (cet_transfers_without_shadow_stack
 * runs them that way to check the transfer plumbing itself on any system). */
static int cet_child_arg(int argc, char **argv, const char *a) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], a)) return 1;
    return 0;
}
#define CET_CHILD_GATE() do { if (!cet_active() && !cet_child_arg(argc, argv, "noss")) { printf("NOT_ACTIVE=1\n"); return 3; } } while (0)
static cet_blk g_land_ref;              /* what cet_landing should see in XMM0-15 */
void cet_landing_finish(void) {
    if (cet_land_mode == 0) ExitThread(CET_LAND_EXIT_THREAD);
    int bad = 0;
    for (int i = 0; i < 16; i++) if (memcmp(cet_land_xmm[i], g_land_ref.in[i], 16)) bad++;
    printf("LANDED=1\nLAND_XMM_BAD=%d\nLAND_MXCSR=0x%x\nLAND_SSP=0x%llx\n", bad, cet_land_mxcsr, (unsigned long long)cet_land_ssp);
    fflush(stdout);
    ExitProcess(0);
}
void cet_smash_survived(void) {
    printf("SMASH_SURVIVED=1\n"); fflush(stdout);
    ExitProcess(0x5A5A);
}

/* SetThreadContext edits of CET_U on a suspended spinning thread. */
CHILD(cet_child_set_ssp) {
    (void)argc; (void)argv;
    (void)argc; (void)argv;
    cet_ntsetctx_t ntset = (cet_ntsetctx_t)t_proc("ntdll.dll", "NtSetContextThread");
    if (!cet_active() || !ntset) { printf("NOT_ACTIVE=1\n"); return 3; }
    int ymm = cet_use_ymm();
    static cet_worker w;
    if (!cet_worker_start(&w, 0x55E7, ymm)) { printf("NO_WORKER=1\n"); return 4; }
    cet_xctx x;
    const char *err = cet_xctx_init(&x, 3 | CET_XS_MASK_CET_U | (ymm ? T_XS_AVX : 0));
    if (err) { printf("CTX_ERR=%s\n", err); return 5; }
    SuspendThread(w.h);
    if (!GetThreadContext(w.h, x.c)) { printf("GET_FAILED=%lu\n", GetLastError()); return 6; }
    uint8_t *p = cet_locate(&x, XSTATE_CET_U, NULL);
    if (!p) { printf("NO_CET_U=1\n"); return 7; }
    uint64_t ucet0, ssp0; memcpy(&ucet0, p, 8); memcpy(&ssp0, p + 8, 8);
    DWORD64 rip0 = x.c->Rip;
    printf("SSP0=0x%llx\nUCET0=0x%llx\n", (unsigned long long)ssp0, (unsigned long long)ucet0);
    static const struct { const char *name; int64_t dssp; int ucet_zero; } ed[] = {
        {"UNCHANGED", 0, 0}, {"MISALIGNED", 4, 0}, {"PUSH", -8, 0}, {"ABOVE", 0x40000000, 0}, {"DISABLE", 0, 1},
    };
    int stable = 1;
    for (unsigned i = 0; i < T_COUNTOF(ed); i++) {
        if (!GetThreadContext(w.h, x.c)) { printf("GET_FAILED=%lu\n", GetLastError()); return 6; }
        uint64_t ucet = ed[i].ucet_zero ? 0 : ucet0, ssp = ssp0 + (uint64_t)ed[i].dssp;
        memcpy(p, &ucet, 8); memcpy(p + 8, &ssp, 8);
        cet_xctx_request(&x);         /* CET_U in XSTATE_BV even if the Get reported it in its init state */
        LONG st = ntset(w.h, x.c);
        printf("%s=0x%08lx\n", ed[i].name, (unsigned long)st);
        if (!GetThreadContext(w.h, x.c)) { printf("GET_FAILED=%lu\n", GetLastError()); return 6; }
        uint64_t s2; memcpy(&s2, p + 8, 8);
        if (s2 != ssp0 || x.c->Rip != rip0) {
            stable = 0;
            printf("UNSTABLE_AFTER_%s=ssp 0x%llx rip 0x%llx\n", ed[i].name, (unsigned long long)s2, (unsigned long long)x.c->Rip);
        }
    }
    /* SetThreadContext (the documented API) must agree with the NTSTATUS. */
    if (!GetThreadContext(w.h, x.c)) { printf("GET_FAILED=%lu\n", GetLastError()); return 6; }
    uint64_t bad = ssp0 + 4; memcpy(p + 8, &bad, 8);
    cet_xctx_request(&x);
    SetLastError(0);
    BOOL r = SetThreadContext(w.h, x.c);
    printf("API_MISALIGNED=%d,%lu\n", r, GetLastError());
    printf("SSP_STABLE=%d\n", stable);
    ResumeThread(w.h);
    DWORD ec = cet_worker_join(&w);
    int bad_regs = 0;
    for (int i = 0; i < 16; i++) if (memcmp(w.b.out[i], w.b.in[i], ymm ? 32 : 16)) bad_regs++;
    printf("WORKER_EXIT=%lu\nWORKER_REGS_BAD=%d\nWORKER_SSP_SAME=%d\n", (unsigned long)ec, bad_regs, w.b.ssp_after == w.b.ssp_before);
    fflush(stdout);
    return 0;
}

static unsigned long long cet_child_val(const t_child_result *r, const char *key, int *found) {
    char v[128]; char k[64]; snprintf(k, sizeof k, "%s=", key);
    if (!t_child_find(r, k, v, sizeof v)) { if (found) *found = 0; return 0; }
    if (found) *found = 1;
    return strtoull(v, NULL, 0);
}
#define CET_RUN_CHILD(name, args, r) do { \
    REQUIRE(t_child_run(name, args, 15000, r)); \
    REQUIREF(!(r)->timed_out, "child %s timed out", name); \
    if (strstr((r)->out, "NOT_ACTIVE=1")) SKIP("shadow stack not active in the child"); } while (0)

TEST(cet_setcontext_ssp_validation) {
    CET_NEED_ACTIVE();
    REQUIREF(cet_api_features() & CET_XS_MASK_CET_U, "GetEnabledXStateFeatures lacks CET_U");
    static t_child_result r;
    CET_RUN_CHILD("cet_child_set_ssp", NULL, &r);
    t_log("%s", r.out);
    CHECKF(r.exit_code == 0, "child exit 0x%lx (exc 0x%lx); output:\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    if (r.exit_code != 0) return;
    int f;
    CHECK_EQ(cet_child_val(&r, "UNCHANGED", &f), 0);
    static const char *deny[] = {"MISALIGNED", "PUSH", "ABOVE", "DISABLE"};
    for (unsigned i = 0; i < T_COUNTOF(deny); i++) {
        unsigned long long st = cet_child_val(&r, deny[i], &f);
        CHECKF(f && (st & 0xC0000000u) == 0xC0000000u, "%s CET_U edit accepted (NTSTATUS 0x%llx): SSP may only move up (pop) "
               "within the shadow stack, 8-aligned, with SH_STK_EN kept", deny[i], st);
        if (f && st != CET_STATUS_SET_CONTEXT_DENIED)
            t_note("%s rejected with 0x%llx (19041 returns STATUS_SET_CONTEXT_DENIED)", deny[i], st);
    }
    char v[64] = "";
    t_child_find(&r, "API_MISALIGNED=", v, sizeof v);
    unsigned rr = 1; unsigned long gle = 0;
    sscanf(v, "%u,%lu", &rr, &gle);
    CHECKF(rr == 0, "SetThreadContext accepted a misaligned SSP");
    if (rr == 0 && gle != cet_status_to_win32(CET_STATUS_SET_CONTEXT_DENIED))
        t_note("SetThreadContext GLE=%lu (ERROR for STATUS_SET_CONTEXT_DENIED is %lu)", gle,
               (unsigned long)cet_status_to_win32(CET_STATUS_SET_CONTEXT_DENIED));
    CHECK_EQ(cet_child_val(&r, "SSP_STABLE", &f), 1);
    CHECK_EQ(cet_child_val(&r, "WORKER_EXIT", &f), 0);
    CHECK_EQ(cet_child_val(&r, "WORKER_REGS_BAD", &f), 0);
    CHECK_EQ(cet_child_val(&r, "WORKER_SSP_SAME", &f), 1);
}

/* SetThreadContext RIP redirect of a suspended spinning thread to cet_landing (which never returns). */
CHILD(cet_child_redirect) {
    CET_CHILD_GATE();
    static cet_worker w;
    if (!cet_worker_start(&w, 0x7ED1, 0)) { printf("NO_WORKER=1\n"); return 4; }
    g_land_ref = w.b;
    cet_land_mode = 0;
    SuspendThread(w.h);
    static CONTEXT c T_ALIGN(16);
    memset(&c, 0, sizeof c);
    c.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
    if (!GetThreadContext(w.h, &c)) { printf("GET_FAILED=%lu\n", GetLastError()); return 6; }
    c.Rip = (DWORD64)(ULONG_PTR)cet_landing;
    c.Rsp = ((c.Rsp - 512) & ~(DWORD64)15) - 8;
    SetLastError(0);
    BOOL ok = SetThreadContext(w.h, &c);
    printf("SET_OK=%d\nSET_GLE=%lu\n", ok, GetLastError());
    ResumeThread(w.h);
    if (!ok) {
        DWORD ec = cet_worker_join(&w);
        printf("WORKER_EXIT=%lu\n", (unsigned long)ec);
        return 0;
    }
    DWORD wr = WaitForSingleObject(w.h, 5000), ec = 0;
    GetExitCodeThread(w.h, &ec);
    int bad = 0;
    for (int i = 0; i < 16; i++) if (memcmp(cet_land_xmm[i], w.b.in[i], 16)) bad++;
    printf("WAIT=%lu\nWORKER_EXIT=%lu\nLAND_XMM_BAD=%d\nLAND_SSP=0x%llx\nSPIN_SSP=0x%llx\n", (unsigned long)wr, (unsigned long)ec, bad,
           (unsigned long long)cet_land_ssp, (unsigned long long)w.b.ssp_before);
    if (wr != WAIT_OBJECT_0) { w.stop = 1; TerminateThread(w.h, 1); }
    CloseHandle(w.h);
    fflush(stdout);
    return 0;
}

TEST(cet_setcontext_rip_redirect) {
    CET_NEED_ACTIVE();
    cet_policy p; cet_get_policy(&p);
    static t_child_result r;
    CET_RUN_CHILD("cet_child_redirect", NULL, &r);
    t_log("%s", r.out);
    CHECKF(r.exit_code == 0, "child exit 0x%lx (exc 0x%lx); output:\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    if (r.exit_code != 0) return;
    int f;
    int ok = (int)cet_child_val(&r, "SET_OK", &f);
    unsigned long gle = (unsigned long)cet_child_val(&r, "SET_GLE", &f);
    t_info("set_ok", "%d", ok);
    t_info("SetContextIpValidation", "%d", p.ipval);
    if (!ok) {
        /* KeVerifyContextIpForUserCet only validates RIP when SetContextIpValidation is on. */
        CHECKF(p.ok && p.ipval, "SetThreadContext RIP redirect failed (GLE=%lu) although SetContextIpValidation is off", gle);
        t_note("RIP redirect rejected under SetContextIpValidation (relaxed=%d): GLE=%lu", p.relaxed, gle);
        CHECK_EQ(cet_child_val(&r, "WORKER_EXIT", &f), 0);
        return;
    }
    CHECK_EQ(cet_child_val(&r, "WAIT", &f), WAIT_OBJECT_0);
    CHECK_EQ(cet_child_val(&r, "WORKER_EXIT", &f), CET_LAND_EXIT_THREAD);
    CHECK_EQ(cet_child_val(&r, "LAND_XMM_BAD", &f), 0);
    unsigned long long ls = cet_child_val(&r, "LAND_SSP", &f), ss = cet_child_val(&r, "SPIN_SSP", &f);
    t_info("land_ssp_minus_spin_ssp", "%lld", (long long)(ls - ss));
    CHECKF(ls != 0, "redirected thread runs without a shadow stack");
    if (!p.ipval) CHECKF(ls == ss, "SSP changed by a RIP-only SetThreadContext: 0x%llx -> 0x%llx", ss, ls);
}

/* longjmp (msvcrt, unwinding through RtlUnwindEx) out of nested frames, then normal returns. */
#include <setjmp.h>
static jmp_buf g_cet_jb;
static volatile int g_cet_lj_depth;
#pragma GCC diagnostic ignored "-Winfinite-recursion"   /* longjmp at n == 0 ends the recursion */
static T_NOINLINE void cet_lj_deep(int n) {
    if (n == 0) longjmp(g_cet_jb, 7);
    cet_lj_deep(n - 1);
    g_cet_lj_depth++;
}
static T_NOINLINE int cet_lj_once(int depth, uintptr_t *s0, uintptr_t *s1) {
    *s0 = cet_rdssp();
    int v = setjmp(g_cet_jb);
    if (v == 0) { cet_lj_deep(depth); return -1; }
    *s1 = cet_rdssp();
    return v;           /* RET checked against the shadow stack */
}
CHILD(cet_child_longjmp) {
    CET_CHILD_GATE();
    int okc = 0;
    for (int i = 0; i < 4; i++) {
        uintptr_t s0 = 0, s1 = 0;
        int v = cet_lj_once(1 + 7 * i, &s0, &s1);
        printf("LJ%d=%d ssp_same=%d\n", i, v, s0 == s1);
        if (v == 7 && s0 == s1) okc++;
    }
    printf("LJ_OK=%d\n", okc);
    fflush(stdout);
    return 0;
}

TEST(cet_longjmp) {
    CET_NEED_ACTIVE();
    static t_child_result r;
    CET_RUN_CHILD("cet_child_longjmp", NULL, &r);
    t_log("%s", r.out);
    if (r.exit_code == CET_STATUS_STACK_BUFFER_OVERRUN)
        XFAIL("longjmp under shadow stacks ended in a fast fail (STATUS_STACK_BUFFER_OVERRUN): the CRT/unwinder did not "
              "unwind the shadow stack");
    CHECKF(r.exit_code == 0, "child exit 0x%lx (exc 0x%lx); output:\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    int f;
    CHECK_EQ(cet_child_val(&r, "LJ_OK", &f), 4);
}

/* RtlRestoreContext: (a) back to the capture point in the same frame, MXCSR edited; (b) to cet_landing with
 * XMM0-15 set in the context. Neither path returns (stale shadow-stack entries remain above the target). */
static CONTEXT g_rc_ctx T_ALIGN(16);
static volatile int g_rc_n;
static T_NOINLINE void cet_rc_restore(void) { RtlRestoreContext(&g_rc_ctx, NULL); }
CHILD(cet_child_restorecontext) {
    CET_CHILD_GATE();
    int to_landing = cet_child_arg(argc, argv, "landing");
    t_set_mxcsr(0x1F80);
    g_rc_n = 0;
    memset(&g_rc_ctx, 0, sizeof g_rc_ctx);
    RtlCaptureContext(&g_rc_ctx);
    if (g_rc_n++ == 0) {
        if (to_landing) {
            cet_pattern(&g_land_ref, 0x2C70, 0);
            for (int i = 0; i < 16; i++) memcpy(&g_rc_ctx.FltSave.XmmRegisters[i], g_land_ref.in[i], 16);
            g_rc_ctx.Rip = (DWORD64)(ULONG_PTR)cet_landing;
            g_rc_ctx.Rsp = ((g_rc_ctx.Rsp - 1024) & ~(DWORD64)15) - 8;
            cet_land_mode = 1;
        }
        g_rc_ctx.MxCsr = 0x3F80;                 /* RC = round down */
        g_rc_ctx.FltSave.MxCsr = 0x3F80;
        cet_rc_restore();
        printf("RESTORE_RETURNED=1\n");
        fflush(stdout);
        ExitProcess(9);
    }
    printf("RESTORED=1\nN=%d\nMXCSR=0x%x\n", g_rc_n, t_get_mxcsr());
    fflush(stdout);
    ExitProcess(0);
}

static void cet_restore_case(const char *arg) {
    cet_policy p; cet_get_policy(&p);
    static t_child_result r;
    CET_RUN_CHILD("cet_child_restorecontext", arg, &r);
    t_log("%s", r.out);
    if (r.exit_code == CET_STATUS_STACK_BUFFER_OVERRUN && p.ok && p.ipval) {
        t_note("RtlRestoreContext fast-failed (FAST_FAIL_SET_CONTEXT_DENIED) under SetContextIpValidation");
        return;
    }
    CHECKF(r.exit_code == 0, "child exit 0x%lx (exc 0x%lx); output:\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    if (r.exit_code != 0) return;
    int f;
    if (arg && !strcmp(arg, "landing")) {
        CHECK_EQ(cet_child_val(&r, "LANDED", &f), 1);
        CHECK_EQ(cet_child_val(&r, "LAND_XMM_BAD", &f), 0);
        CHECK_EQ(cet_child_val(&r, "LAND_MXCSR", &f) & ~0x3Full, 0x3F80);
        CHECKF(cet_child_val(&r, "LAND_SSP", &f) != 0, "restored thread runs without a shadow stack");
    } else {
        CHECK_EQ(cet_child_val(&r, "RESTORED", &f), 1);
        CHECK_EQ(cet_child_val(&r, "N", &f), 2);
        CHECK_EQ(cet_child_val(&r, "MXCSR", &f) & ~0x3Full, 0x3F80);
    }
}
TEST(cet_restorecontext_same_frame) { CET_NEED_ACTIVE(); cet_restore_case(NULL); }
TEST(cet_restorecontext_new_rip) { CET_NEED_ACTIVE(); cet_restore_case("landing"); }

/* Exception continue: (a) skip the UD2 (RIP+2) and return normally; (b) continue at cet_landing with XMM5 edited. */
static volatile int g_ex_mode, g_ex_seen, g_ex_ctx_bad;
static cet_blk g_ex_blk;
static LONG CALLBACK cet_ex_veh(EXCEPTION_POINTERS *ep) {
    CONTEXT *c = ep->ContextRecord;
    if (ep->ExceptionRecord->ExceptionCode != STATUS_ILLEGAL_INSTRUCTION || c->Rip != (DWORD64)(ULONG_PTR)cet_fault_ud2)
        return EXCEPTION_CONTINUE_SEARCH;
    g_ex_seen++;
    for (int i = 0; i < 16; i++) if (memcmp(&c->FltSave.XmmRegisters[i], g_ex_blk.in[i], 16)) g_ex_ctx_bad++;
    if (g_ex_mode == 0) {
        c->Rip += 2;
    } else {
        g_land_ref = g_ex_blk;
        t_pattern_reg(g_land_ref.in[5], 16, 0xED17, 5);
        memcpy(&c->FltSave.XmmRegisters[5], g_land_ref.in[5], 16);
        c->Rip = (DWORD64)(ULONG_PTR)cet_landing;
        c->Rsp = ((c->Rsp - 256) & ~(DWORD64)15) - 8;
        cet_land_mode = 1;
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}
CHILD(cet_child_continue) {
    CET_CHILD_GATE();
    g_ex_mode = cet_child_arg(argc, argv, "landing");
    cet_pattern(&g_ex_blk, 0xE7C0, 0);
    if (!AddVectoredExceptionHandler(1, cet_ex_veh)) { printf("NO_VEH=1\n"); return 4; }
    cet_fault_xmm(&g_ex_blk);         /* mode 1 never comes back */
    int bad = 0;
    for (int i = 0; i < 16; i++) if (memcmp(g_ex_blk.out[i], g_ex_blk.in[i], 16)) bad++;
    printf("RETURNED=1\nSEEN=%d\nCTX_XMM_BAD=%d\nXMM_BAD=%d\nSSP_SAME=%d\n", g_ex_seen, g_ex_ctx_bad, bad,
           g_ex_blk.ssp_before == g_ex_blk.ssp_after && g_ex_blk.ssp_before != 0);
    fflush(stdout);
    return 0;
}
static void cet_continue_case(const char *arg) {
    cet_policy p; cet_get_policy(&p);
    static t_child_result r;
    CET_RUN_CHILD("cet_child_continue", arg, &r);
    t_log("%s", r.out);
    CHECKF(r.exit_code == 0, "child exit 0x%lx (exc 0x%lx); output:\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    if (r.exit_code != 0) return;
    int f;
    if (arg) {
        CHECK_EQ(cet_child_val(&r, "LANDED", &f), 1);
        CHECK_EQ(cet_child_val(&r, "LAND_XMM_BAD", &f), 0);
        CHECKF(cet_child_val(&r, "LAND_SSP", &f) != 0, "continued thread runs without a shadow stack");
    } else {
        CHECK_EQ(cet_child_val(&r, "RETURNED", &f), 1);
        CHECK_EQ(cet_child_val(&r, "SEEN", &f), 1);
        CHECK_EQ(cet_child_val(&r, "CTX_XMM_BAD", &f), 0);
        CHECK_EQ(cet_child_val(&r, "XMM_BAD", &f), 0);
        CHECK_EQ(cet_child_val(&r, "SSP_SAME", &f), 1);
    }
}
TEST(cet_exception_continue_skip) { CET_NEED_ACTIVE(); cet_continue_case(NULL); }
TEST(cet_exception_continue_new_rip) { CET_NEED_ACTIVE(); cet_continue_case("landing"); }

/* Fibers: each fiber has its own shadow stack; switches preserve both SSPs and (FIBER_FLAG_FLOAT_SWITCH) MXCSR. */
static void *g_fb_main, *g_fb_fiber;
static uintptr_t g_fb_ssp[4], g_fb_deep[4];
static unsigned g_fb_mxcsr[4];
static T_NOINLINE void cet_fb_switch_deep(int i) {
    g_fb_deep[i] = cet_rdssp();
    SwitchToFiber(g_fb_main);
}
static VOID CALLBACK cet_fb_proc(void *arg) {
    (void)arg;
    t_set_mxcsr(0x3F80);
    for (int i = 0; i < 4; i++) {
        g_fb_ssp[i] = cet_rdssp();
        g_fb_mxcsr[i] = t_get_mxcsr();
        if (i & 1) cet_fb_switch_deep(i); else SwitchToFiber(g_fb_main);
    }
    for (;;) SwitchToFiber(g_fb_main);
}
CHILD(cet_child_fiber) {
    CET_CHILD_GATE();
    g_fb_main = ConvertThreadToFiber(NULL);
    if (!g_fb_main) { printf("NO_FIBER=%lu\n", GetLastError()); return 4; }
    g_fb_fiber = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, cet_fb_proc, NULL);
    if (!g_fb_fiber) { printf("NO_FIBER=%lu\n", GetLastError()); return 4; }
    t_set_mxcsr(0x1F80);
    uintptr_t m0 = cet_rdssp();
    int main_ok = 1, mx_ok = 1;
    for (int i = 0; i < 4; i++) {
        SwitchToFiber(g_fb_fiber);
        if (cet_rdssp() != m0) main_ok = 0;
        if (t_get_mxcsr() != 0x1F80) mx_ok = 0;
    }
    int fb_ok = 1;
    for (int i = 0; i < 4; i++) {
        if (!g_fb_ssp[i] || g_fb_ssp[i] != g_fb_ssp[0]) fb_ok = 0;
        if (g_fb_mxcsr[i] != 0x3F80) mx_ok = 0;
    }
    int deep_ok = g_fb_deep[1] == g_fb_ssp[0] - 8 && g_fb_deep[3] == g_fb_ssp[0] - 8;
    MEMORY_BASIC_INFORMATION a, b;
    int sep = VirtualQuery((void *)m0, &a, sizeof a) && g_fb_ssp[0] && VirtualQuery((void *)g_fb_ssp[0], &b, sizeof b) &&
              a.AllocationBase != b.AllocationBase;
    DeleteFiber(g_fb_fiber);
    ConvertFiberToThread();
    printf("MAIN_SSP=0x%llx\nFIBER_SSP=0x%llx\nMAIN_SSP_STABLE=%d\nFIBER_SSP_STABLE=%d\nDEEP_OK=%d\nSEPARATE=%d\nMXCSR_OK=%d\nEND_SSP_SAME=%d\n",
           (unsigned long long)m0, (unsigned long long)g_fb_ssp[0], main_ok, fb_ok, deep_ok, sep, mx_ok, cet_rdssp() == m0);
    fflush(stdout);
    return 0;
}
TEST(cet_fiber_switch) {
    CET_NEED_ACTIVE();
    static t_child_result r;
    CET_RUN_CHILD("cet_child_fiber", NULL, &r);
    t_log("%s", r.out);
    CHECKF(r.exit_code == 0, "child exit 0x%lx (exc 0x%lx); output:\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    if (r.exit_code != 0) return;
    int f;
    CHECK_EQ(cet_child_val(&r, "MAIN_SSP_STABLE", &f), 1);
    CHECK_EQ(cet_child_val(&r, "FIBER_SSP_STABLE", &f), 1);
    CHECK_EQ(cet_child_val(&r, "DEEP_OK", &f), 1);
    CHECKF(cet_child_val(&r, "MAIN_SSP", &f) != cet_child_val(&r, "FIBER_SSP", &f), "fiber shares the thread's SSP");
    CHECKF(cet_child_val(&r, "SEPARATE", &f) == 1, "fiber shadow stack not in its own allocation");
    CHECK_EQ(cet_child_val(&r, "MXCSR_OK", &f), 1);
    CHECK_EQ(cet_child_val(&r, "END_SSP_SAME", &f), 1);
}

/* A RET to an address that differs from the shadow stack copy: #CP -> fast fail, no handler runs. */
static LONG CALLBACK cet_smash_veh(EXCEPTION_POINTERS *ep) {
    printf("VEH_SAW=0x%08lx\n", ep->ExceptionRecord->ExceptionCode);
    fflush(stdout);
    return EXCEPTION_CONTINUE_SEARCH;
}
CHILD(cet_child_smash) {
    CET_CHILD_GATE();
    AddVectoredExceptionHandler(1, cet_smash_veh);
    printf("SMASHING=1\n"); fflush(stdout);
    cet_smash();
    printf("SMASH_RETURNED=1\n"); fflush(stdout);
    return 0;
}
TEST(cet_return_address_smash) {
    CET_NEED_ACTIVE();
    cet_policy p; cet_get_policy(&p);
    static t_child_result r;
    CET_RUN_CHILD("cet_child_smash", NULL, &r);
    t_log("%s", r.out);
    t_info("child_exit", "0x%lx", (unsigned long)r.exit_code);
    CHECKF(strstr(r.out, "SMASHING=1") != NULL, "child did not reach the smash");
    CHECKF(!strstr(r.out, "SMASH_RETURNED=1"), "RET through a smashed return address returned to the original caller");
    if (p.ok && p.audit) {
        t_note("audit mode: survived=%d", strstr(r.out, "SMASH_SURVIVED=1") != NULL);
        return;
    }
    CHECKF(r.exit_code == CET_STATUS_STACK_BUFFER_OVERRUN,
           "child exit 0x%lx, want 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN, FAST_FAIL_CONTROL_INVALID_RETURN_ADDRESS); output:\n%s",
           (unsigned long)r.exit_code, r.out);
    CHECKF(!strstr(r.out, "SMASH_SURVIVED=1"), "mismatching RET executed: shadow stack not enforced");
    if (strstr(r.out, "VEH_SAW=")) t_note("a vectored handler saw the #CP before termination (a fast fail bypasses handlers)");
}
/* Manual: the same children with "noss" on a process WITHOUT shadow stacks, checking only what does not depend on
 * CET (the transfers themselves, register images, MXCSR, and that the smash really returns elsewhere). Validates
 * this exe's plumbing on systems (or emulators) that cannot enforce shadow stacks. */
static void cet_expect(const char *child, const char *args, DWORD exit_want, const char *const *kv) {
    static t_child_result r;
    if (!t_child_run(child, args, 15000, &r)) { CHECKF(0, "%s: could not start", child); return; }
    t_log("%s %s:\n%s", child, args, r.out);
    CHECKF(!r.timed_out && r.exit_code == exit_want, "%s %s: exit 0x%lx (want 0x%lx) timeout=%d\n%s", child, args,
           (unsigned long)r.exit_code, (unsigned long)exit_want, r.timed_out, r.out);
    for (; kv && *kv; kv += 2) {
        int f; unsigned long long v = cet_child_val(&r, kv[0], &f);
        CHECKF(f && v == strtoull(kv[1], NULL, 0), "%s %s: %s=%s%llx, want %s", child, args, kv[0], f ? "0x" : "(missing) ", v, kv[1]);
    }
}
TEST_MANUAL(cet_transfers_without_shadow_stack) {
    if (cet_active()) SKIP("shadow stack active: the enforcement cases cover this");
    static const char *const redirect[] = {"SET_OK", "1", "WAIT", "0", "WORKER_EXIT", "0x1A4D", "LAND_XMM_BAD", "0", NULL};
    cet_expect("cet_child_redirect", "noss", 0, redirect);
    static const char *const lj[] = {"LJ_OK", "4", NULL};
    cet_expect("cet_child_longjmp", "noss", 0, lj);
    static const char *const rc[] = {"RESTORED", "1", "N", "2", "MXCSR", "0x3F80", NULL};
    cet_expect("cet_child_restorecontext", "noss", 0, rc);
    static const char *const rcl[] = {"LANDED", "1", "LAND_XMM_BAD", "0", "LAND_MXCSR", "0x3F80", NULL};
    cet_expect("cet_child_restorecontext", "landing noss", 0, rcl);
    static const char *const cs[] = {"RETURNED", "1", "SEEN", "1", "CTX_XMM_BAD", "0", "XMM_BAD", "0", NULL};
    cet_expect("cet_child_continue", "noss", 0, cs);
    static const char *const cl[] = {"LANDED", "1", "LAND_XMM_BAD", "0", NULL};
    cet_expect("cet_child_continue", "landing noss", 0, cl);
    /* Known Wine gap (see fiber_xstate): SwitchToFiber does not switch MXCSR per fiber -> not checked there. */
    static const char *const fb[] = {"END_SSP_SAME", "1", "MXCSR_OK", "1", NULL};
    static const char *const fb_wine[] = {"END_SSP_SAME", "1", NULL};
    cet_expect("cet_child_fiber", "noss", 0, t_is_wine() ? fb_wine : fb);
    static const char *const sm[] = {"SMASHING", "1", "SMASH_SURVIVED", "1", NULL};
    cet_expect("cet_child_smash", "noss", 0x5A5A, sm);
}
#else /* T_X86 */
TEST(cet_enforcement_x86) {
    CHECKF(cet_rdssp() == 0, "RDSSPD returned an SSP in a 32-bit process");
    SKIP("CET user shadow stacks are x64-only on Windows; 32-bit and WOW64 processes have none");
}
#endif
