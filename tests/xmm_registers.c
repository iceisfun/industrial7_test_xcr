/* xmm_registers - XMM0-15 (x64) / XMM0-7 (x86): data movement semantics, register-file integrity and
 * preservation of XMM state across OS-mediated events (preemption, page faults, exceptions, context APIs)
 * and across calls into Windows (ABI).
 *
 * Contracts under test:
 *   - Intel SDM Vol.2 / AMD APM Vol.4 instruction semantics for the SSE..SSE4.1 data movement forms:
 *       MOVD/MOVQ (load or GPR source) zero bits 127:32 / 127:64;  MOVQ xmm,xmm (F3 0F 7E) zeroes 127:64;
 *       MOVSS/MOVSD from memory zero the upper bits, register-to-register forms merge (preserve) them;
 *       MOVHPS/MOVLPS/MOVHPD/MOVLPD loads merge, stores write exactly 8 bytes; MOVLHPS/MOVHLPS merge;
 *       scalar arithmetic and CVTSI2SS/SD, CVTSS2SD, CVTSD2SS preserve the destination's upper bits;
 *       legacy-SSE writes leave YMM[255:128] unchanged, VEX.128 writes zero it; VZEROUPPER keeps 127:0.
 *       MOVAPS/MOVDQA on a non-16-byte-aligned address raise #GP (Windows: STATUS_ACCESS_VIOLATION).
 *       Every memory form accesses exactly its operand width (checked against a PAGE_NOACCESS page);
 *       MASKMOVDQU stores only bytes whose mask byte has bit 7 set; PEXTRB/PEXTRD/PEXTRQ and EXTRACTPS zero-extend.
 *       FXSAVE/XSAVE store XMMi at byte 160+16*i; XSAVE sets XSTATE_BV[1] when SSE state is not in its
 *       initial configuration; XRSTOR with RFBM[1]=1, XSTATE_BV[1]=0 initialises XMM to zero (SDM 13.8).
 *   - OS contract (both architectures): the complete XMM register file and MXCSR of a thread survive
 *     preemption, other threads using XMM, demand-zero page faults (also split 16-byte accesses), migration
 *     between CPUs (SetThreadAffinityMask), VM-exiting instructions, Suspend/Resume and Get/SetThreadContext;
 *     an exception continued from a handler resumes with the CONTEXT's XMM state, MXCSR and x87 CW
 *     (unchanged CONTEXT -> unchanged registers; modified CONTEXT -> modified registers).
 *     GetThreadContext (kernelbase) is a thin wrapper of NtGetContextThread (checked in the Win10 19041
 *     reference decompilation), so the XMM values reported for a suspended thread are the kernel's.
 *   - Microsoft x64 calling convention ("x64 software conventions: register volatility"): XMM6-XMM15
 *     (low 128 bits) are nonvolatile, XMM0-5 volatile; MXCSR bits 6-15 and the x87 control word are
 *     nonvolatile. Checked on x64 only. 32-bit (cdecl/stdcall): no XMM register is preserved, so x86
 *     call-based cases check only the functional result and record the clobber count as INFO.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> t_note/t_info.
 * Deliberately faulting operations run in CHILD() processes via t_child_run(). See common/API.md. */
#include "t.h"

#define NX T_NXMM

/* ======================================================================================
 * inline-asm building blocks (operate on t_vregs.v: register i at byte i*64)
 * ====================================================================================== */
#ifdef T_X64
#define XR_RL   "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#define XR_EVEN "0,2,4,6,8,10,12,14"
#define XR_ODD  "1,3,5,7,9,11,13,15"
#define XR_CLOB "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7", \
                "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15"
#define XS(n) #n
#else
#define XR_RL   "0,1,2,3,4,5,6,7"
#define XR_EVEN "0,2,4,6"
#define XR_ODD  "1,3,5,7"
#define XR_CLOB "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7"
#define XS(n) "_" #n
#endif
#define XR_LOADALL(p)  ".irp i," XR_RL "\n\tmovdqu \\i*64(" p "), %%xmm\\i\n\t.endr\n\t"
#define XR_STOREALL(p) ".irp i," XR_RL "\n\tmovdqu %%xmm\\i, \\i*64(" p ")\n\t.endr\n\t"

/* ======================================================================================
 * asm stubs used as t_vregs_call callbacks. They touch NO XMM register, so every XMM register must
 * survive them (whatever the OS does underneath: VM exits, page faults, exceptions).
 * ====================================================================================== */
#define XFN(n) ".globl " XS(n) "\n" XS(n) ":\n"
void xr_stub_nop(void *);
void xr_stub_ud2(void *);
void xr_stub_int3(void *);
void xr_stub_div0(void *);
void xr_stub_av(void *);
void xr_stub_hlt(void *);
void xr_stub_misalign(void *);
void xr_stub_sstep(void *);
void xr_stub_cpuid(void *);
void xr_stub_rdtsc(void *);
void xr_stub_touch(void *);    /* arg = base of XR_TOUCH_PAGES fresh pages: writes one byte per page */
void xr_stub_x87(void *);      /* 1000 x87 ops, stack balanced, CW untouched */
void xr_stub_mmx(void *);      /* MMX ops on mm0-7, ends with EMMS */
extern char xr_f_ud2[], xr_r_ud2[], xr_f_int3[], xr_r_int3[], xr_f_div[], xr_r_div[], xr_f_av[], xr_r_av[],
            xr_f_hlt[], xr_r_hlt[], xr_f_mis[], xr_r_mis[];
#define XR_TOUCH_PAGES 64
__attribute__((aligned(64))) unsigned char xr_buf[128];

__asm__(
    ".text\n"
    XFN(xr_stub_nop) "  ret\n"
    XFN(xr_stub_ud2) XFN(xr_f_ud2) "  ud2\n" XFN(xr_r_ud2) "  ret\n"
    XFN(xr_stub_int3) XFN(xr_f_int3) "  int3\n" XFN(xr_r_int3) "  nop\n  ret\n"
    XFN(xr_stub_hlt) XFN(xr_f_hlt) "  hlt\n" XFN(xr_r_hlt) "  ret\n"
#ifdef T_X64
    XFN(xr_stub_div0) "  xorl %edx, %edx\n  movl $7, %eax\n  xorl %r10d, %r10d\n"
    XFN(xr_f_div) "  divl %r10d\n" XFN(xr_r_div) "  ret\n"
    XFN(xr_stub_av) "  xorl %eax, %eax\n" XFN(xr_f_av) "  movl (%rax), %eax\n" XFN(xr_r_av) "  ret\n"
    XFN(xr_stub_misalign) "  leaq xr_buf+1(%rip), %rax\n" XFN(xr_f_mis) "  movaps %xmm0, (%rax)\n" XFN(xr_r_mis) "  ret\n"
    XFN(xr_stub_sstep) "  pushfq\n  orl $0x100, (%rsp)\n  popfq\n  nop\n  nop\n  nop\n  ret\n"
    XFN(xr_stub_cpuid) "  movq %rbx, %r11\n  movl $200, %r10d\n"
    "1: xorl %eax, %eax\n  xorl %ecx, %ecx\n  cpuid\n  movl $1, %eax\n  xorl %ecx, %ecx\n  cpuid\n"
    "  movl $7, %eax\n  xorl %ecx, %ecx\n  cpuid\n  decl %r10d\n  jnz 1b\n  movq %r11, %rbx\n  ret\n"
    XFN(xr_stub_rdtsc) "  movl $1000, %r10d\n1: rdtsc\n  decl %r10d\n  jnz 1b\n  ret\n"
    XFN(xr_stub_touch) "  movl $64, %eax\n1: movb $1, (%rcx)\n  addq $4096, %rcx\n  decl %eax\n  jnz 1b\n  ret\n"
    XFN(xr_stub_x87) "  movl $1000, %r10d\n1: fld1\n  fldpi\n  fmulp\n  fsqrt\n  fstp %st(0)\n  decl %r10d\n  jnz 1b\n  ret\n"
    XFN(xr_stub_mmx) "  leaq xr_buf(%rip), %rax\n  movl $1000, %r10d\n"
    "  .irp i,0,1,2,3,4,5,6,7\n  movq \\i*8(%rax), %mm\\i\n  .endr\n"
    "1: paddq %mm1, %mm0\n  pxor %mm3, %mm2\n  psubd %mm5, %mm4\n  paddw %mm7, %mm6\n  decl %r10d\n  jnz 1b\n"
    "  emms\n  ret\n"
#else
    XFN(xr_stub_div0) "  xorl %edx, %edx\n  movl $7, %eax\n  xorl %ecx, %ecx\n"
    XFN(xr_f_div) "  divl %ecx\n" XFN(xr_r_div) "  ret\n"
    XFN(xr_stub_av) "  xorl %eax, %eax\n" XFN(xr_f_av) "  movl (%eax), %eax\n" XFN(xr_r_av) "  ret\n"
    XFN(xr_stub_misalign) "  movl $_xr_buf+1, %eax\n" XFN(xr_f_mis) "  movaps %xmm0, (%eax)\n" XFN(xr_r_mis) "  ret\n"
    XFN(xr_stub_sstep) "  pushfl\n  orl $0x100, (%esp)\n  popfl\n  nop\n  nop\n  nop\n  ret\n"
    XFN(xr_stub_cpuid) "  pushl %ebx\n  pushl %esi\n  movl $200, %esi\n"
    "1: xorl %eax, %eax\n  xorl %ecx, %ecx\n  cpuid\n  movl $1, %eax\n  xorl %ecx, %ecx\n  cpuid\n"
    "  movl $7, %eax\n  xorl %ecx, %ecx\n  cpuid\n  decl %esi\n  jnz 1b\n  popl %esi\n  popl %ebx\n  ret\n"
    XFN(xr_stub_rdtsc) "  movl $1000, %ecx\n1: rdtsc\n  decl %ecx\n  jnz 1b\n  ret\n"
    XFN(xr_stub_touch) "  movl 4(%esp), %ecx\n  movl $64, %eax\n1: movb $1, (%ecx)\n  addl $4096, %ecx\n  decl %eax\n  jnz 1b\n  ret\n"
    XFN(xr_stub_x87) "  movl $1000, %ecx\n1: fld1\n  fldpi\n  fmulp\n  fsqrt\n  fstp %st(0)\n  decl %ecx\n  jnz 1b\n  ret\n"
    XFN(xr_stub_mmx) "  movl $_xr_buf, %eax\n  movl $1000, %ecx\n"
    "  .irp i,0,1,2,3,4,5,6,7\n  movq \\i*8(%eax), %mm\\i\n  .endr\n"
    "1: paddq %mm1, %mm0\n  pxor %mm3, %mm2\n  psubd %mm5, %mm4\n  paddw %mm7, %mm6\n  decl %ecx\n  jnz 1b\n"
    "  emms\n  ret\n"
#endif
);

/* ======================================================================================
 * common helpers
 * ====================================================================================== */
static int diff_regs(const t_vregs *a, const t_vregs *b, int first, int last) {
    int n = 0;
    if (last >= NX) last = NX - 1;
    for (int i = first; i <= last; i++) n += memcmp(a->v[i], b->v[i], 16) != 0;
    return n;
}
static void check_all(const t_vregs *out, const t_vregs *in, const char *label) {
    t_vregs_check(out, in, T_VK_XMM, 0, -1, label);
}
/* Registers the calling convention preserves: x64 XMM6-15 checked; volatile ones recorded only. */
static void check_abi(const t_vregs *out, const t_vregs *in, const char *label) {
#ifdef T_X64
    t_vregs_check(out, in, T_VK_XMM, 6, 15, label);
    t_info("volatile_xmm0_5_changed", "%d of 6 (%s)", diff_regs(out, in, 0, 5), label);
#else
    t_info("x86_xmm_changed", "%d of 8 (%s; all volatile in the 32-bit ABI, not checked)", diff_regs(out, in, 0, 7), label);
#endif
}
static void run_abi(void (*fn)(void *), void *arg, uint32_t tag, const char *label) {
    t_vregs in, out;
    t_vregs_pattern(&in, T_VK_XMM, tag);
    REQUIRE(t_vregs_call(T_VK_XMM, &in, &out, fn, arg) == 0);
    check_abi(&out, &in, label);
}
/* fn touches no XMM register (asm stub / NULL): every register must survive. */
static void run_all(void (*fn)(void *), void *arg, uint32_t tag, const char *label) {
    t_vregs in, out;
    t_vregs_pattern(&in, T_VK_XMM, tag);
    REQUIRE(t_vregs_call(T_VK_XMM, &in, &out, fn, arg) == 0);
    check_all(&out, &in, label);
}
/* ABI-correct clobber of every XMM register (the compiler saves/restores XMM6-15 on x64). */
static T_NOINLINE void clobber_xmm(uint32_t salt) {
    t_vregs junk;
    t_pattern_regs(junk.v, NX, 64, 16, 3000 + salt % 1000);
    __asm__ volatile(XR_LOADALL("%0") :: "r"(junk.v) : XR_CLOB, "memory");
}
static void fill_bytes(void *p, size_t n, uint8_t v) { memset(p, v, n); }
static void x87_prior_use(void) {
    /* x87 work before the XMM test: sets C-flags / PE in the status word, leaves the stack empty. */
    double r = 0;
    __asm__ volatile("fld1\n\tfldpi\n\tfmulp\n\tfsqrt\n\tfstpl %0" : "=m"(r) :: "st", "memory");
    CHECKF(r > 1.77 && r < 1.78, "x87 sqrt(pi) = %f", r);
    CHECK_EQ(t_get_x87_tw(), 0xFFFF);
}

/* ======================================================================================
 * 1. register-file round trips (no call: every register checked on both architectures)
 * ====================================================================================== */
TEST(xmm_roundtrip_plain) {
    run_all(NULL, NULL, __LINE__, "load/store, no call");
    t_info("xmm_registers", "%d", NX);
}

TEST(xmm_roundtrip_nop_stub) {
    run_all((void (*)(void *))xr_stub_nop, NULL, __LINE__, "call+ret to an asm stub");
}

TEST(xmm_patterns_all_distinct) {
    /* sanity of the test data itself: 2*NX distinct, nonzero qwords, so a swapped/aliased register shows */
    t_vregs in;
    t_vregs_pattern(&in, T_VK_XMM, __LINE__);
    int dup = 0, zero = 0;
    for (int i = 0; i < NX * 2; i++) {
        uint64_t a; memcpy(&a, in.v[i / 2] + 8 * (i % 2), 8);
        zero += a == 0;
        for (int j = i + 1; j < NX * 2; j++) { uint64_t b; memcpy(&b, in.v[j / 2] + 8 * (j % 2), 8); dup += a == b; }
    }
    CHECK_EQ(dup, 0);
    CHECK_EQ(zero, 0);
    t_vregs out;
    REQUIRE(t_vregs_call(T_VK_XMM, &in, &out, NULL, NULL) == 0);
    for (int i = 0; i < NX; i++)
        for (int j = i + 1; j < NX; j++)
            CHECKF(memcmp(out.v[i], out.v[j], 16) != 0, "xmm%d and xmm%d read back identical (aliasing)", i, j);
}

static void special_image(t_vregs *r, int kind) {
    memset(r, 0, sizeof *r);
    for (int i = 0; i < NX; i++) {
        uint8_t *p = r->v[i];
        switch (kind) {
        case 0: memset(p, 0, 16); break;
        case 1: memset(p, 0xFF, 16); break;
        case 2: memset(p, (i & 1) ? 0x55 : 0xAA, 16); break;
        case 3: for (int b = 0; b < 16; b++) p[b] = (uint8_t)(i * 16 + b); break;
        case 4: { uint32_t f[4] = { 0x7F800001u, 0xFFBFFFFFu, 0x00000001u, 0x80000000u };   /* SNaN, -SNaN, denormal, -0 */
                  f[i & 3] ^= (uint32_t)i << 8; memcpy(p, f, 16); break; }
        case 5: { uint64_t d[2] = { 0x7FF0000000000001ull + i, 0x000FFFFFFFFFFFFFull - i }; memcpy(p, d, 16); break; }
        default: { uint32_t f[4] = { 0x80000000u >> (i & 31), 0x7F800000u, 0xFF800000u, 0x7FC00000u | (uint32_t)i }; memcpy(p, f, 16); }
        }
    }
}

TEST(xmm_roundtrip_special_values) {
    static const char *names[] = { "zeros", "ones", "0x55/0xAA", "byte-index", "float SNaN/denorm/-0", "double SNaN/denorm", "inf/qnan" };
    for (int k = 0; k < 7; k++) {
        t_vregs in, out;
        special_image(&in, k);
        REQUIRE(t_vregs_call(T_VK_XMM, &in, &out, NULL, NULL) == 0);
        char lbl[64]; snprintf(lbl, sizeof lbl, "pattern %s", names[k]);
        check_all(&out, &in, lbl);
    }
}

TEST(xmm_walking_ones_and_zeros) {
    int bad = 0, first = 1;
    for (int inv = 0; inv < 2; inv++)
        for (int bit = 0; bit < 128; bit++) {
            t_vregs in, out;
            memset(&in, 0, sizeof in);
            for (int i = 0; i < NX; i++) {
                int b = (bit + i * 7) & 127;
                memset(in.v[i], inv ? 0xFF : 0, 16);
                in.v[i][b / 8] ^= (uint8_t)(1u << (b % 8));
            }
            REQUIRE(t_vregs_call(T_VK_XMM, &in, &out, NULL, NULL) == 0);
            int d = diff_regs(&out, &in, 0, NX - 1);
            bad += d;
            if (d && first) { first = 0; check_all(&out, &in, inv ? "walking zero" : "walking one"); }
        }
    CHECKF(bad == 0, "%d register images differ over 256 walking-bit iterations", bad);
}

TEST(xmm_roundtrip_10000_iterations) {
    int bad = 0, first = 1;
    for (int it = 0; it < 10000; it++) {
        t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(it % 4000));
        t_vregs_call(T_VK_XMM, &in, &out, NULL, NULL);
        int d = diff_regs(&out, &in, 0, NX - 1);
        bad += d;
        if (d && first) { first = 0; check_all(&out, &in, "iteration"); }
    }
    CHECKF(bad == 0, "%d mismatching registers over 10000 iterations", bad);
}

TEST(xmm_roundtrip_10000_after_x87_use) {
    x87_prior_use();
    int bad = 0, first = 1;
    for (int it = 0; it < 10000; it++) {
        t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(1000 + it % 3000));
        t_vregs_call(T_VK_XMM, &in, &out, (it & 1) ? (void (*)(void *))xr_stub_x87 : NULL, NULL);
        int d = diff_regs(&out, &in, 0, NX - 1);
        bad += d;
        if (d && first) { first = 0; check_all(&out, &in, "after x87 use"); }
        if (it % 1000 == 0) x87_prior_use();
    }
    CHECKF(bad == 0, "%d mismatching registers over 10000 iterations with x87 use", bad);
}

/* load/store instruction variants (aligned operands: t_vregs.v is 64-byte aligned) */
#define XR_LS(fname, ld, st) \
    static T_NOINLINE void fname(const t_vregs *in, t_vregs *out) { \
        __asm__ volatile(".irp i," XR_RL "\n\t" ld " \\i*64(%0), %%xmm\\i\n\t.endr\n\t" \
                         ".irp i," XR_RL "\n\t" st " %%xmm\\i, \\i*64(%1)\n\t.endr\n\tsfence\n\t" \
                         :: "r"(in->v), "r"(out->v) : XR_CLOB, "memory"); }
XR_LS(ls_movdqa, "movdqa", "movdqa")
XR_LS(ls_movaps, "movaps", "movaps")
XR_LS(ls_movapd, "movapd", "movapd")
XR_LS(ls_movups, "movups", "movups")
XR_LS(ls_movupd, "movupd", "movupd")
XR_LS(ls_movntdq, "movdqa", "movntdq")
XR_LS(ls_movntps, "movaps", "movntps")
XR_LS(ls_movntpd, "movapd", "movntpd")
XR_LS(ls_lddqu, "lddqu", "movdqu")

TEST(xmm_load_store_forms_aligned) {
    static const struct { const char *n; void (*f)(const t_vregs *, t_vregs *); int sse3; } v[] = {
        { "movdqa", ls_movdqa, 0 }, { "movaps", ls_movaps, 0 }, { "movapd", ls_movapd, 0 },
        { "movups", ls_movups, 0 }, { "movupd", ls_movupd, 0 }, { "movntdq", ls_movntdq, 0 },
        { "movntps", ls_movntps, 0 }, { "movntpd", ls_movntpd, 0 }, { "lddqu", ls_lddqu, 1 } };
    for (size_t k = 0; k < T_COUNTOF(v); k++) {
        if (v[k].sse3 && !cpu_has(F_SSE3)) { t_note("%s skipped: no SSE3", v[k].n); continue; }
        t_vregs in, out;
        special_image(&in, (int)k % 7);
        if (k % 2) t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(__LINE__ + k));
        memset(&out, 0, sizeof out);
        v[k].f(&in, &out);
        char lbl[64]; snprintf(lbl, sizeof lbl, "via %s", v[k].n);
        check_all(&out, &in, lbl);
    }
}

TEST(xmm_load_store_unaligned_offsets) {
    __attribute__((aligned(64))) uint8_t src[NX * 16 + 64], dst[NX * 16 + 64];
    int bad = 0;
    for (int off = 0; off < 16; off++) {
        int doff = (off * 7) & 15;
        for (size_t i = 0; i < sizeof src; i++) src[i] = (uint8_t)(i * 31 + off);
        memset(dst, 0xCC, sizeof dst);
        __asm__ volatile(".irp i," XR_RL "\n\tmovdqu \\i*16(%0), %%xmm\\i\n\t.endr\n\t"
                         ".irp i," XR_RL "\n\tmovups %%xmm\\i, \\i*16(%1)\n\t.endr\n\t"
                         :: "r"(src + off), "r"(dst + doff) : XR_CLOB, "memory");
        if (memcmp(dst + doff, src + off, NX * 16)) { bad++; CHECK_MEM(dst + doff, src + off, NX * 16); }
        for (int i = 0; i < doff; i++) bad += dst[i] != 0xCC;
        for (size_t i = doff + NX * 16; i < sizeof dst; i++) bad += dst[i] != 0xCC;
    }
    CHECKF(bad == 0, "%d unaligned load/store mismatches (incl. writes outside the target)", bad);
}

/* write exactly one register; the others must keep their values (register-file isolation) */
#define XR_W1(n) \
    static T_NOINLINE void w1_##n(const t_vregs *in, t_vregs *out, const void *val) { \
        __asm__ volatile(XR_LOADALL("%0") "movdqu (%2), %%xmm" #n "\n\t" XR_STOREALL("%1") \
                         :: "r"(in->v), "r"(out->v), "r"(val) : XR_CLOB, "memory"); }
XR_W1(0) XR_W1(1) XR_W1(2) XR_W1(3) XR_W1(4) XR_W1(5) XR_W1(6) XR_W1(7)
#ifdef T_X64
XR_W1(8) XR_W1(9) XR_W1(10) XR_W1(11) XR_W1(12) XR_W1(13) XR_W1(14) XR_W1(15)
#endif
typedef void (*w1_fn)(const t_vregs *, t_vregs *, const void *);
static const w1_fn g_w1[] = { w1_0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7,
#ifdef T_X64
    w1_8, w1_9, w1_10, w1_11, w1_12, w1_13, w1_14, w1_15
#endif
};

TEST(xmm_write_one_register_isolation) {
    for (int n = 0; n < NX; n++) {
        t_vregs in, out, want;
        uint8_t val[16];
        t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(__LINE__ + n));
        t_pattern_reg(val, 16, 4000 + n, 31);
        want = in;
        memcpy(want.v[n], val, 16);
        g_w1[n](&in, &out, val);
        char lbl[48]; snprintf(lbl, sizeof lbl, "after writing only xmm%d", n);
        check_all(&out, &want, lbl);
    }
}

#define MV(d, s) "movdqa %%xmm" #s ", %%xmm" #d "\n\t"
TEST(xmm_register_rotate_moves) {
    /* xmm(i) <- xmm(i+1), last <- old xmm0 : every register-to-register path once */
    t_vregs in, out, want;
    __attribute__((aligned(16))) uint8_t tmp[16];
    t_vregs_pattern(&in, T_VK_XMM, __LINE__);
    memset(&out, 0, sizeof out);
    __asm__ volatile(XR_LOADALL("%0") "movdqu %%xmm0, (%2)\n\t"
                     MV(0,1) MV(1,2) MV(2,3) MV(3,4) MV(4,5) MV(5,6) MV(6,7)
#ifdef T_X64
                     MV(7,8) MV(8,9) MV(9,10) MV(10,11) MV(11,12) MV(12,13) MV(13,14) MV(14,15)
                     "movdqu (%2), %%xmm15\n\t"
#else
                     "movdqu (%2), %%xmm7\n\t"
#endif
                     XR_STOREALL("%1") :: "r"(in.v), "r"(out.v), "r"(tmp) : XR_CLOB, "memory");
    want = in;
    for (int i = 0; i < NX; i++) memcpy(want.v[i], in.v[(i + 1) % NX], 16);
    check_all(&out, &want, "after rotate");
}

/* ======================================================================================
 * 2. data movement semantics: partial writes, zeroing vs merging (SDM/APM)
 * ====================================================================================== */
#define XR_OP2(fname, insn) \
    static T_NOINLINE void fname(const void *a, const void *b, void *o) { \
        __asm__ volatile("movdqu (%0), %%xmm1\n\tmovdqu (%1), %%xmm2\n\t" insn "\n\tmovdqu %%xmm1, (%2)\n\t" \
                         :: "r"(a), "r"(b), "r"(o) : "xmm1", "xmm2", "eax", "memory"); }
typedef void (*op2_fn)(const void *, const void *, void *);

XR_OP2(op_movd_m,   "movd (%1), %%xmm1")
XR_OP2(op_movd_r,   "movl (%1), %%eax\n\tmovd %%eax, %%xmm1")
XR_OP2(op_movq_m,   "movq (%1), %%xmm1")
XR_OP2(op_movq_x,   "movq %%xmm2, %%xmm1")
#ifdef T_X64
XR_OP2(op_movq_r,   "movq (%1), %%rax\n\tmovq %%rax, %%xmm1")
#endif
XR_OP2(op_movss_m,  "movss (%1), %%xmm1")
XR_OP2(op_movss_x,  "movss %%xmm2, %%xmm1")
XR_OP2(op_movsd_m,  "movsd (%1), %%xmm1")
XR_OP2(op_movsd_x,  "movsd %%xmm2, %%xmm1")
XR_OP2(op_movhps_m, "movhps (%1), %%xmm1")
XR_OP2(op_movlps_m, "movlps (%1), %%xmm1")
XR_OP2(op_movhpd_m, "movhpd (%1), %%xmm1")
XR_OP2(op_movlpd_m, "movlpd (%1), %%xmm1")
XR_OP2(op_movlhps,  "movlhps %%xmm2, %%xmm1")
XR_OP2(op_movhlps,  "movhlps %%xmm2, %%xmm1")
XR_OP2(op_movaps_x, "movaps %%xmm2, %%xmm1")
XR_OP2(op_movapd_x, "movapd %%xmm2, %%xmm1")
XR_OP2(op_movdqa_x, "movdqa %%xmm2, %%xmm1")
XR_OP2(op_movups_x, "movups %%xmm2, %%xmm1")
XR_OP2(op_movupd_x, "movupd %%xmm2, %%xmm1")
XR_OP2(op_movdqu_x, "movdqu %%xmm2, %%xmm1")
XR_OP2(op_movdqu_m, "movdqu (%1), %%xmm1")
/* scalar arithmetic / conversions (destination upper bits preserved) */
XR_OP2(op_addss,    "addss %%xmm2, %%xmm1")
XR_OP2(op_subss,    "subss (%1), %%xmm1")
XR_OP2(op_mulss,    "mulss %%xmm2, %%xmm1")
XR_OP2(op_divss,    "divss %%xmm1, %%xmm2\n\tmovss %%xmm2, %%xmm1")
XR_OP2(op_sqrtss,   "sqrtss %%xmm2, %%xmm1")
XR_OP2(op_minss,    "minss %%xmm2, %%xmm1")
XR_OP2(op_maxss,    "maxss (%1), %%xmm1")
XR_OP2(op_cmpltss,  "cmpltss %%xmm2, %%xmm1")
XR_OP2(op_rcpss,    "rcpss %%xmm2, %%xmm1")
XR_OP2(op_rsqrtss,  "rsqrtss %%xmm2, %%xmm1")
XR_OP2(op_cvtsi2ss, "movl $7, %%eax\n\tcvtsi2ssl %%eax, %%xmm1")
XR_OP2(op_cvtsi2sd, "movl $7, %%eax\n\tcvtsi2sdl %%eax, %%xmm1")
XR_OP2(op_cvtss2sd, "cvtss2sd %%xmm2, %%xmm1")
XR_OP2(op_cvtsd2ss, "cvtsd2ss %%xmm2, %%xmm1")
XR_OP2(op_addsd,    "addsd %%xmm2, %%xmm1")
XR_OP2(op_mulsd,    "mulsd (%1), %%xmm1")
XR_OP2(op_sqrtsd,   "sqrtsd %%xmm2, %%xmm1")
XR_OP2(op_minsd,    "minsd %%xmm2, %%xmm1")
XR_OP2(op_maxsd,    "maxsd %%xmm2, %%xmm1")
XR_OP2(op_cmpltsd,  "cmpltsd %%xmm2, %%xmm1")
/* lanes / shuffles / packed integer */
XR_OP2(op_pshufd_1b, "pshufd $0x1B, %%xmm2, %%xmm1")
XR_OP2(op_pshufd_00, "pshufd $0x00, %%xmm2, %%xmm1")
XR_OP2(op_pshufd_4e, "pshufd $0x4E, %%xmm2, %%xmm1")
XR_OP2(op_pshufd_e4, "pshufd $0xE4, %%xmm2, %%xmm1")
XR_OP2(op_shufps_b1, "shufps $0xB1, %%xmm2, %%xmm1")
XR_OP2(op_pshuflw,   "pshuflw $0x1B, %%xmm2, %%xmm1")
XR_OP2(op_pshufhw,   "pshufhw $0x1B, %%xmm2, %%xmm1")
XR_OP2(op_punpcklqdq, "punpcklqdq %%xmm2, %%xmm1")
XR_OP2(op_punpckhqdq, "punpckhqdq %%xmm2, %%xmm1")
XR_OP2(op_punpcklbw, "punpcklbw %%xmm2, %%xmm1")
XR_OP2(op_punpckhdq, "punpckhdq %%xmm2, %%xmm1")
XR_OP2(op_unpcklps,  "unpcklps %%xmm2, %%xmm1")
XR_OP2(op_paddd,     "paddd %%xmm2, %%xmm1")
XR_OP2(op_paddq,     "paddq %%xmm2, %%xmm1")
XR_OP2(op_psubb,     "psubb %%xmm2, %%xmm1")
XR_OP2(op_psllq8,    "psllq $8, %%xmm1")
XR_OP2(op_pslldq3,   "pslldq $3, %%xmm1")
XR_OP2(op_psrldq5,   "psrldq $5, %%xmm1")
XR_OP2(op_pand,      "pand %%xmm2, %%xmm1")
XR_OP2(op_por,       "por %%xmm2, %%xmm1")
XR_OP2(op_pxor,      "pxor %%xmm2, %%xmm1")
XR_OP2(op_pandn,     "pandn %%xmm2, %%xmm1")
XR_OP2(op_pinsrw_all, "movl $0xA000, %%eax\n\t"
        ".irp k,0,1,2,3,4,5,6,7\n\tpinsrw $\\k, %%eax, %%xmm1\n\tincl %%eax\n\t.endr")
XR_OP2(op_pinsrw_3,  "movl $0xBEEF, %%eax\n\tpinsrw $3, %%eax, %%xmm1")
/* SSE3 / SSSE3 / SSE4.1 */
XR_OP2(op_movddup,   "movddup (%1), %%xmm1")
XR_OP2(op_movsldup,  "movsldup %%xmm2, %%xmm1")
XR_OP2(op_movshdup,  "movshdup %%xmm2, %%xmm1")
XR_OP2(op_lddqu,     "lddqu (%1), %%xmm1")
XR_OP2(op_pshufb,    "pshufb %%xmm2, %%xmm1")
XR_OP2(op_palignr5,  "palignr $5, %%xmm2, %%xmm1")
XR_OP2(op_pinsrd2,   "movl $0xC0FFEE11, %%eax\n\tpinsrd $2, %%eax, %%xmm1")
#ifdef T_X64
XR_OP2(op_pinsrq1,   "movq (%1), %%rax\n\tpinsrq $1, %%rax, %%xmm1")
#endif
XR_OP2(op_insertps,  "insertps $0x61, %%xmm2, %%xmm1")
XR_OP2(op_pblendw,   "pblendw $0xA5, %%xmm2, %%xmm1")
XR_OP2(op_blendps,   "blendps $0x5, %%xmm2, %%xmm1")

typedef struct { __attribute__((aligned(16))) uint8_t a[16]; __attribute__((aligned(16))) uint8_t b[16];
                 __attribute__((aligned(16))) uint8_t o[16]; __attribute__((aligned(16))) uint8_t e[16]; } v3;

static void v3_init(v3 *t, uint32_t tag) {
    t_pattern_reg(t->a, 16, tag, 1);
    t_pattern_reg(t->b, 16, tag, 2);
    memset(t->o, 0xCC, 16);
}
/* run op, compare with t->e */
static void v3_run(v3 *t, op2_fn f, const char *what) {
    memset(t->o, 0xCC, 16);
    f(t->a, t->b, t->o);
    CHECK_REG(what, t->o, t->e, 16);
}
static void set32(uint8_t *p, int lane, uint32_t v) { memcpy(p + 4 * lane, &v, 4); }
static void set64(uint8_t *p, int lane, uint64_t v) { memcpy(p + 8 * lane, &v, 8); }
static uint32_t get32(const uint8_t *p, int lane) { uint32_t v; memcpy(&v, p + 4 * lane, 4); return v; }
static uint64_t get64(const uint8_t *p, int lane) { uint64_t v; memcpy(&v, p + 8 * lane, 8); return v; }
static uint16_t get16(const uint8_t *p, int lane) { uint16_t v; memcpy(&v, p + 2 * lane, 2); return v; }

TEST(xmm_movd_zero_extends) {
    v3 t; v3_init(&t, __LINE__);
    memset(t.e, 0, 16); memcpy(t.e, t.b, 4);
    v3_run(&t, op_movd_m, "movd m32 -> xmm (bits 127:32 zeroed)");
    v3_run(&t, op_movd_r, "movd r32 -> xmm (bits 127:32 zeroed)");
}

TEST(xmm_movq_zero_extends) {
    v3 t; v3_init(&t, __LINE__);
    memset(t.e, 0, 16); memcpy(t.e, t.b, 8);
    v3_run(&t, op_movq_m, "movq m64 -> xmm (bits 127:64 zeroed)");
    v3_run(&t, op_movq_x, "movq xmm -> xmm (bits 127:64 zeroed)");
#ifdef T_X64
    v3_run(&t, op_movq_r, "movq r64 -> xmm (bits 127:64 zeroed)");
#else
    t_note("movq r64 -> xmm needs 64-bit mode: not tested on x86");
#endif
}

TEST(xmm_movss_mem_zeroes_reg_merges) {
    v3 t; v3_init(&t, __LINE__);
    memset(t.e, 0, 16); memcpy(t.e, t.b, 4);
    v3_run(&t, op_movss_m, "movss m32 -> xmm (upper zeroed)");
    memcpy(t.e, t.a, 16); memcpy(t.e, t.b, 4);
    v3_run(&t, op_movss_x, "movss xmm, xmm (upper 96 bits of destination preserved)");
}

TEST(xmm_movsd_mem_zeroes_reg_merges) {
    v3 t; v3_init(&t, __LINE__);
    memset(t.e, 0, 16); memcpy(t.e, t.b, 8);
    v3_run(&t, op_movsd_m, "movsd m64 -> xmm (upper zeroed)");
    memcpy(t.e, t.a, 16); memcpy(t.e, t.b, 8);
    v3_run(&t, op_movsd_x, "movsd xmm, xmm (upper 64 bits of destination preserved)");
}

TEST(xmm_movhps_movlps_movhpd_movlpd_merge) {
    v3 t; v3_init(&t, __LINE__);
    memcpy(t.e, t.a, 8); memcpy(t.e + 8, t.b, 8);
    v3_run(&t, op_movhps_m, "movhps m64 (low half preserved)");
    v3_run(&t, op_movhpd_m, "movhpd m64 (low half preserved)");
    memcpy(t.e, t.b, 8); memcpy(t.e + 8, t.a + 8, 8);
    v3_run(&t, op_movlps_m, "movlps m64 (high half preserved)");
    v3_run(&t, op_movlpd_m, "movlpd m64 (high half preserved)");
}

TEST(xmm_movlhps_movhlps) {
    v3 t; v3_init(&t, __LINE__);
    memcpy(t.e, t.a, 8); memcpy(t.e + 8, t.b, 8);
    v3_run(&t, op_movlhps, "movlhps (dst[127:64] = src[63:0], dst low preserved)");
    memcpy(t.e, t.b + 8, 8); memcpy(t.e + 8, t.a + 8, 8);
    v3_run(&t, op_movhlps, "movhlps (dst[63:0] = src[127:64], dst high preserved)");
}

TEST(xmm_full_register_moves) {
    v3 t; v3_init(&t, __LINE__);
    memcpy(t.e, t.b, 16);
    v3_run(&t, op_movaps_x, "movaps xmm,xmm");
    v3_run(&t, op_movapd_x, "movapd xmm,xmm");
    v3_run(&t, op_movdqa_x, "movdqa xmm,xmm");
    v3_run(&t, op_movups_x, "movups xmm,xmm");
    v3_run(&t, op_movupd_x, "movupd xmm,xmm");
    v3_run(&t, op_movdqu_x, "movdqu xmm,xmm");
    v3_run(&t, op_movdqu_m, "movdqu m128");
    /* data moves never quiet an SNaN or flush a denormal */
    uint32_t sn[4] = { 0x7F800001u, 0x00000001u, 0xFFA00000u, 0x80000003u };
    memcpy(t.b, sn, 16); memcpy(t.e, sn, 16);
    unsigned m0 = t_get_mxcsr();
    t_set_mxcsr(m0 | 0x8000);   /* FTZ must not affect data moves (DAZ not set: MXCSR_MASK may lack it) */
    v3_run(&t, op_movaps_x, "movaps with SNaN/denormal lanes under FTZ");
    memcpy(t.e, t.a, 16); memcpy(t.e, sn, 4);
    v3_run(&t, op_movss_x, "movss reg with SNaN lane under FTZ");
    CHECKF((t_get_mxcsr() & 0x3F) == 0, "data moves raised MXCSR flags 0x%x", t_get_mxcsr() & 0x3F);
    t_set_mxcsr(m0);
}

static uint32_t fbits(float f) { uint32_t v; memcpy(&v, &f, 4); return v; }
static uint64_t dbits(double d) { uint64_t v; memcpy(&v, &d, 8); return v; }

TEST(xmm_scalar_single_ops_preserve_upper) {
    v3 t; v3_init(&t, __LINE__);
    set32(t.a, 0, fbits(1.5f)); set32(t.b, 0, fbits(2.25f));
    struct { op2_fn f; uint32_t lo; const char *n; } c[] = {
        { op_addss, fbits(3.75f), "addss" }, { op_subss, fbits(-0.75f), "subss m32" }, { op_mulss, fbits(3.375f), "mulss" },
        { op_divss, fbits(1.5f), "divss" }, { op_sqrtss, fbits(1.5f), "sqrtss" }, { op_minss, fbits(1.5f), "minss" },
        { op_maxss, fbits(2.25f), "maxss m32" }, { op_cmpltss, 0xFFFFFFFFu, "cmpltss" }, { op_cvtsi2ss, fbits(7.0f), "cvtsi2ss r32" },
        { op_cvtsd2ss, 0, "cvtsd2ss" } };
    for (size_t k = 0; k < T_COUNTOF(c); k++) {
        if (c[k].f == op_cvtsd2ss) { set64(t.b, 0, dbits(2.25)); c[k].lo = fbits(2.25f); }
        memcpy(t.e, t.a, 16); set32(t.e, 0, c[k].lo);
        char lbl[80]; snprintf(lbl, sizeof lbl, "%s (bits 127:32 of destination preserved)", c[k].n);
        v3_run(&t, c[k].f, lbl);
    }
    /* approximations: low lane in a tolerance, upper exact. Intel SDM bounds |rel err| <= 1.5*2^-12; the
     * check uses 2^-11 so a vendor-specific approximation table cannot turn this lane-merge test into a FAIL. */
    set32(t.b, 0, fbits(2.0f));
    op_rcpss(t.a, t.b, t.o);
    CHECK_MEM(t.o + 4, t.a + 4, 12);
    float r; memcpy(&r, t.o, 4);
    CHECKF(r > 0.5f * (1 - 1.0f / 2048) && r < 0.5f * (1 + 1.0f / 2048), "rcpss(2) = %.9g", r);
    set32(t.b, 0, fbits(4.0f));
    op_rsqrtss(t.a, t.b, t.o);
    CHECK_MEM(t.o + 4, t.a + 4, 12);
    memcpy(&r, t.o, 4);
    CHECKF(r > 0.5f * (1 - 1.0f / 2048) && r < 0.5f * (1 + 1.0f / 2048), "rsqrtss(4) = %.9g", r);
}

TEST(xmm_scalar_double_ops_preserve_upper) {
    v3 t; v3_init(&t, __LINE__);
    set64(t.a, 0, dbits(1.5)); set64(t.b, 0, dbits(2.25));
    struct { op2_fn f; uint64_t lo; const char *n; } c[] = {
        { op_addsd, dbits(3.75), "addsd" }, { op_mulsd, dbits(3.375), "mulsd m64" }, { op_sqrtsd, dbits(1.5), "sqrtsd" },
        { op_minsd, dbits(1.5), "minsd" }, { op_maxsd, dbits(2.25), "maxsd" }, { op_cmpltsd, ~0ull, "cmpltsd" },
        { op_cvtsi2sd, dbits(7.0), "cvtsi2sd r32" } };
    for (size_t k = 0; k < T_COUNTOF(c); k++) {
        memcpy(t.e, t.a, 16); set64(t.e, 0, c[k].lo);
        char lbl[80]; snprintf(lbl, sizeof lbl, "%s (bits 127:64 of destination preserved)", c[k].n);
        v3_run(&t, c[k].f, lbl);
    }
    set32(t.b, 0, fbits(2.25f));
    memcpy(t.e, t.a, 16); set64(t.e, 0, dbits(2.25));
    v3_run(&t, op_cvtss2sd, "cvtss2sd (bits 127:64 preserved)");
}

TEST(xmm_scalar_stores_write_exact_width) {
    __attribute__((aligned(16))) uint8_t b[16], o[32], e[32];
    t_pattern_reg(b, 16, __LINE__, 3);
    struct { const char *n; int w; int src_off; } c[] = {
        { "movd xmm->m32", 4, 0 }, { "movq xmm->m64", 8, 0 }, { "movss xmm->m32", 4, 0 }, { "movsd xmm->m64", 8, 0 },
        { "movlps xmm->m64", 8, 0 }, { "movhps xmm->m64", 8, 8 }, { "movlpd xmm->m64", 8, 0 }, { "movhpd xmm->m64", 8, 8 },
        { "movd xmm->r32->m32", 4, 0 } };
    for (size_t k = 0; k < T_COUNTOF(c); k++) {
        memset(o, 0xCC, sizeof o);
        uint8_t *d = o + 8;
        switch (k) {
        case 0: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovd %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        case 1: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovq %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        case 2: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovss %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        case 3: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovsd %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        case 4: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovlps %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        case 5: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovhps %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        case 6: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovlpd %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        case 7: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovhpd %%xmm2, (%1)" :: "r"(b), "r"(d) : "xmm2", "memory"); break;
        default: __asm__ volatile("movdqu (%0), %%xmm2\n\tmovd %%xmm2, %%eax\n\tmovl %%eax, (%1)" :: "r"(b), "r"(d) : "xmm2", "eax", "memory"); break;
        }
        memset(e, 0xCC, sizeof e);
        memcpy(e + 8, b + c[k].src_off, (size_t)c[k].w);
        if (!CHECK_MEM(o, e, sizeof o)) t_note("store form: %s", c[k].n);
    }
}

TEST(xmm_extract_insert_word_lanes) {
    v3 t; v3_init(&t, __LINE__);
    for (int k = 0; k < 8; k++) {
        unsigned got = 0;
        switch (k) {
#define PX(K) case K: __asm__ volatile("movdqu (%1), %%xmm2\n\tpextrw $" #K ", %%xmm2, %0" : "=r"(got) : "r"(t.b) : "xmm2"); break;
        PX(0) PX(1) PX(2) PX(3) PX(4) PX(5) PX(6) PX(7)
#undef PX
        }
        CHECKF(got == get16(t.b, k), "pextrw lane %d: got 0x%x want 0x%x", k, got, get16(t.b, k));
    }
    memcpy(t.e, t.a, 16);
    for (int k = 0; k < 8; k++) { uint16_t w = (uint16_t)(0xA000 + k); memcpy(t.e + 2 * k, &w, 2); }
    v3_run(&t, op_pinsrw_all, "pinsrw into each word lane");
    memcpy(t.e, t.a, 16); { uint16_t w = 0xBEEF; memcpy(t.e + 6, &w, 2); }
    v3_run(&t, op_pinsrw_3, "pinsrw lane 3 only (other lanes preserved)");
}

TEST(xmm_shuffle_lane_routing) {
    v3 t; v3_init(&t, __LINE__);
    for (int i = 0; i < 4; i++) set32(t.e, i, get32(t.b, 3 - i));
    v3_run(&t, op_pshufd_1b, "pshufd 0x1B (reverse dwords)");
    for (int i = 0; i < 4; i++) set32(t.e, i, get32(t.b, 0));
    v3_run(&t, op_pshufd_00, "pshufd 0x00 (broadcast dword0)");
    set64(t.e, 0, get64(t.b, 1)); set64(t.e, 1, get64(t.b, 0));
    v3_run(&t, op_pshufd_4e, "pshufd 0x4E (swap qwords)");
    memcpy(t.e, t.b, 16);
    v3_run(&t, op_pshufd_e4, "pshufd 0xE4 (identity)");
    /* shufps 0xB1: d0 = a[1], d1 = a[0], d2 = b[3], d3 = b[2] */
    set32(t.e, 0, get32(t.a, 1)); set32(t.e, 1, get32(t.a, 0)); set32(t.e, 2, get32(t.b, 3)); set32(t.e, 3, get32(t.b, 2));
    v3_run(&t, op_shufps_b1, "shufps 0xB1");
    memcpy(t.e, t.b, 16);
    for (int i = 0; i < 4; i++) { uint16_t w = get16(t.b, 3 - i); memcpy(t.e + 2 * i, &w, 2); }
    v3_run(&t, op_pshuflw, "pshuflw 0x1B (high qword copied)");
    memcpy(t.e, t.b, 16);
    for (int i = 0; i < 4; i++) { uint16_t w = get16(t.b, 7 - i); memcpy(t.e + 8 + 2 * i, &w, 2); }
    v3_run(&t, op_pshufhw, "pshufhw 0x1B (low qword copied)");
}

TEST(xmm_unpack_lane_routing) {
    v3 t; v3_init(&t, __LINE__);
    set64(t.e, 0, get64(t.a, 0)); set64(t.e, 1, get64(t.b, 0));
    v3_run(&t, op_punpcklqdq, "punpcklqdq");
    set64(t.e, 0, get64(t.a, 1)); set64(t.e, 1, get64(t.b, 1));
    v3_run(&t, op_punpckhqdq, "punpckhqdq");
    for (int i = 0; i < 8; i++) { t.e[2 * i] = t.a[i]; t.e[2 * i + 1] = t.b[i]; }
    v3_run(&t, op_punpcklbw, "punpcklbw");
    set32(t.e, 0, get32(t.a, 2)); set32(t.e, 1, get32(t.b, 2)); set32(t.e, 2, get32(t.a, 3)); set32(t.e, 3, get32(t.b, 3));
    v3_run(&t, op_punpckhdq, "punpckhdq");
    set32(t.e, 0, get32(t.a, 0)); set32(t.e, 1, get32(t.b, 0)); set32(t.e, 2, get32(t.a, 1)); set32(t.e, 3, get32(t.b, 1));
    v3_run(&t, op_unpcklps, "unpcklps");
}

TEST(xmm_packed_int_lanes_independent) {
    v3 t; v3_init(&t, __LINE__);
    /* carries must not cross lane boundaries */
    for (int i = 0; i < 4; i++) { set32(t.a, i, 0xFFFFFFFFu - (uint32_t)i); set32(t.b, i, 1u + (uint32_t)i * 2); }
    for (int i = 0; i < 4; i++) set32(t.e, i, get32(t.a, i) + get32(t.b, i));
    v3_run(&t, op_paddd, "paddd (wrap per dword)");
    set64(t.a, 0, ~0ull); set64(t.a, 1, 0x8000000000000000ull); set64(t.b, 0, 1); set64(t.b, 1, 0x8000000000000000ull);
    set64(t.e, 0, 0); set64(t.e, 1, 0);
    v3_run(&t, op_paddq, "paddq (wrap per qword)");
    v3_init(&t, __LINE__);
    for (int i = 0; i < 16; i++) t.e[i] = (uint8_t)(t.a[i] - t.b[i]);
    v3_run(&t, op_psubb, "psubb");
    set64(t.e, 0, get64(t.a, 0) << 8); set64(t.e, 1, get64(t.a, 1) << 8);
    v3_run(&t, op_psllq8, "psllq 8 (no carry across qwords)");
    memset(t.e, 0, 16); memcpy(t.e + 3, t.a, 13);
    v3_run(&t, op_pslldq3, "pslldq 3 (whole-register byte shift)");
    memset(t.e, 0, 16); memcpy(t.e, t.a + 5, 11);
    v3_run(&t, op_psrldq5, "psrldq 5");
    for (int i = 0; i < 16; i++) t.e[i] = t.a[i] & t.b[i];
    v3_run(&t, op_pand, "pand");
    for (int i = 0; i < 16; i++) t.e[i] = t.a[i] | t.b[i];
    v3_run(&t, op_por, "por");
    for (int i = 0; i < 16; i++) t.e[i] = t.a[i] ^ t.b[i];
    v3_run(&t, op_pxor, "pxor");
    for (int i = 0; i < 16; i++) t.e[i] = (uint8_t)(~t.a[i] & t.b[i]);
    v3_run(&t, op_pandn, "pandn");
}

TEST(xmm_movmsk_extraction) {
    __attribute__((aligned(16))) uint8_t b[16];
    t_pattern_reg(b, 16, __LINE__, 5);
    b[3] |= 0x80; b[7] &= 0x7F; b[11] |= 0x80; b[15] &= 0x7F;
    unsigned ps = 0, pb = 0, pd = 0;
    __asm__ volatile("movdqu (%3), %%xmm3\n\tmovmskps %%xmm3, %0\n\tpmovmskb %%xmm3, %1\n\tmovmskpd %%xmm3, %2"
                     : "=r"(ps), "=r"(pb), "=r"(pd) : "r"(b) : "xmm3");
    unsigned eps = 0, epb = 0, epd = 0;
    for (int i = 0; i < 4; i++) eps |= (unsigned)(b[4 * i + 3] >> 7) << i;
    for (int i = 0; i < 16; i++) epb |= (unsigned)(b[i] >> 7) << i;
    for (int i = 0; i < 2; i++) epd |= (unsigned)(b[8 * i + 7] >> 7) << i;
    CHECK_EQ(ps, eps); CHECK_EQ(pb, epb); CHECK_EQ(pd, epd);
}

/* zero / all-ones idioms on every register */
#define XR_IDIOM(fname, insn, regs) \
    static T_NOINLINE void fname(const t_vregs *in, t_vregs *out) { \
        __asm__ volatile(XR_LOADALL("%0") ".irp i," regs "\n\t" insn " %%xmm\\i, %%xmm\\i\n\t.endr\n\t" XR_STOREALL("%1") \
                         :: "r"(in->v), "r"(out->v) : XR_CLOB, "memory"); }
XR_IDIOM(id_pxor, "pxor", XR_RL)     XR_IDIOM(id_xorps, "xorps", XR_RL)   XR_IDIOM(id_xorpd, "xorpd", XR_RL)
XR_IDIOM(id_psubb, "psubb", XR_RL)   XR_IDIOM(id_psubd, "psubd", XR_RL)   XR_IDIOM(id_psubq, "psubq", XR_RL)
XR_IDIOM(id_pcmpgtd, "pcmpgtd", XR_RL) XR_IDIOM(id_andnps, "andnps", XR_RL) XR_IDIOM(id_pandn, "pandn", XR_RL)
XR_IDIOM(id_pcmpeqb, "pcmpeqb", XR_RL) XR_IDIOM(id_pcmpeqw, "pcmpeqw", XR_RL) XR_IDIOM(id_pcmpeqd, "pcmpeqd", XR_RL)
XR_IDIOM(id_pxor_even, "pxor", XR_EVEN) XR_IDIOM(id_pcmpeqd_odd, "pcmpeqd", XR_ODD)
XR_IDIOM(id_pcmpeqq, "pcmpeqq", XR_RL)

static void idiom_case(void (*f)(const t_vregs *, t_vregs *), const char *name, int ones, int which /*0 all,1 even,2 odd*/) {
    t_vregs in, out, want;
    t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(name[0] * 7 + name[3]));
    want = in;
    for (int i = 0; i < NX; i++)
        if (which == 0 || (which == 1 && !(i & 1)) || (which == 2 && (i & 1))) memset(want.v[i], ones ? 0xFF : 0, 16);
    memset(&out, 0x5A, sizeof out);
    f(&in, &out);
    char lbl[64]; snprintf(lbl, sizeof lbl, "after %s idiom", name);
    check_all(&out, &want, lbl);
}

TEST(xmm_zero_idioms_all_registers) {
    idiom_case(id_pxor, "pxor x,x", 0, 0);
    idiom_case(id_xorps, "xorps x,x", 0, 0);
    idiom_case(id_xorpd, "xorpd x,x", 0, 0);
    idiom_case(id_psubb, "psubb x,x", 0, 0);
    idiom_case(id_psubd, "psubd x,x", 0, 0);
    idiom_case(id_psubq, "psubq x,x", 0, 0);
    idiom_case(id_pcmpgtd, "pcmpgtd x,x", 0, 0);
    idiom_case(id_andnps, "andnps x,x", 0, 0);
    idiom_case(id_pandn, "pandn x,x", 0, 0);
}

TEST(xmm_ones_idioms_all_registers) {
    idiom_case(id_pcmpeqb, "pcmpeqb x,x", 1, 0);
    idiom_case(id_pcmpeqw, "pcmpeqw x,x", 1, 0);
    idiom_case(id_pcmpeqd, "pcmpeqd x,x", 1, 0);
    if (cpu_has(F_SSE41)) idiom_case(id_pcmpeqq, "pcmpeqq x,x", 1, 0);
    else t_note("pcmpeqq skipped: no SSE4.1");
}

TEST(xmm_idioms_touch_only_their_register) {
    idiom_case(id_pxor_even, "pxor even regs", 0, 1);
    idiom_case(id_pcmpeqd_odd, "pcmpeqd odd regs", 1, 2);
}

TEST(xmm_sse3_moves) {
    if (!cpu_has(F_SSE3)) SKIP("SSE3 not present");
    v3 t; v3_init(&t, __LINE__);
    set64(t.e, 0, get64(t.b, 0)); set64(t.e, 1, get64(t.b, 0));
    v3_run(&t, op_movddup, "movddup m64");
    set32(t.e, 0, get32(t.b, 0)); set32(t.e, 1, get32(t.b, 0)); set32(t.e, 2, get32(t.b, 2)); set32(t.e, 3, get32(t.b, 2));
    v3_run(&t, op_movsldup, "movsldup");
    set32(t.e, 0, get32(t.b, 1)); set32(t.e, 1, get32(t.b, 1)); set32(t.e, 2, get32(t.b, 3)); set32(t.e, 3, get32(t.b, 3));
    v3_run(&t, op_movshdup, "movshdup");
    memcpy(t.e, t.b, 16);
    v3_run(&t, op_lddqu, "lddqu");
}

TEST(xmm_ssse3_pshufb_palignr) {
    if (!cpu_has(F_SSSE3)) SKIP("SSSE3 not present");
    v3 t; v3_init(&t, __LINE__);
    for (int i = 0; i < 16; i++) t.b[i] = (uint8_t)((i % 5 == 4) ? 0x80 | i : (15 - i));
    for (int i = 0; i < 16; i++) t.e[i] = (t.b[i] & 0x80) ? 0 : t.a[t.b[i] & 15];
    v3_run(&t, op_pshufb, "pshufb (bit7 lanes zeroed)");
    v3_init(&t, __LINE__);
    memcpy(t.e, t.b + 5, 11); memcpy(t.e + 11, t.a, 5);
    v3_run(&t, op_palignr5, "palignr 5");
}

TEST(xmm_sse41_insert_extract_blend) {
    if (!cpu_has(F_SSE41)) SKIP("SSE4.1 not present");
    v3 t; v3_init(&t, __LINE__);
    memcpy(t.e, t.a, 16); set32(t.e, 2, 0xC0FFEE11u);
    v3_run(&t, op_pinsrd2, "pinsrd lane 2");
#ifdef T_X64
    memcpy(t.e, t.a, 16); set64(t.e, 1, get64(t.b, 0));
    v3_run(&t, op_pinsrq1, "pinsrq lane 1");
#endif
    /* insertps 0x61: src lane 1 -> dst lane 2, zero lane 0 */
    memcpy(t.e, t.a, 16); set32(t.e, 2, get32(t.b, 1)); set32(t.e, 0, 0);
    v3_run(&t, op_insertps, "insertps 0x61");
    for (int i = 0; i < 8; i++) { uint16_t w = (0xA5 >> i & 1) ? get16(t.b, i) : get16(t.a, i); memcpy(t.e + 2 * i, &w, 2); }
    v3_run(&t, op_pblendw, "pblendw 0xA5");
    for (int i = 0; i < 4; i++) set32(t.e, i, (0x5 >> i & 1) ? get32(t.b, i) : get32(t.a, i));
    v3_run(&t, op_blendps, "blendps 0x5");
    for (int k = 0; k < 4; k++) {
        unsigned got = 0;
        switch (k) {
#define PD(K) case K: __asm__ volatile("movdqu (%1), %%xmm2\n\tpextrd $" #K ", %%xmm2, %0" : "=r"(got) : "r"(t.b) : "xmm2"); break;
        PD(0) PD(1) PD(2) PD(3)
#undef PD
        }
        CHECKF(got == get32(t.b, k), "pextrd lane %d: 0x%x want 0x%x", k, got, get32(t.b, k));
    }
}

TEST(xmm_movq2dq_movdq2q) {
    if (!cpu_has(F_MMX)) SKIP("MMX not present");
    v3 t; v3_init(&t, __LINE__);
    __asm__ volatile("movdqu (%0), %%xmm1\n\tmovq (%1), %%mm0\n\tmovq2dq %%mm0, %%xmm1\n\tmovdqu %%xmm1, (%2)\n\temms"
                     :: "r"(t.a), "r"(t.b), "r"(t.o) : "xmm1", "mm0", "memory");
    memset(t.e, 0, 16); memcpy(t.e, t.b, 8);
    CHECK_REG("movq2dq (bits 127:64 zeroed)", t.o, t.e, 16);
    uint64_t q = 0;
    __asm__ volatile("movdqu (%1), %%xmm2\n\tmovdq2q %%xmm2, %%mm1\n\tmovq %%mm1, %0\n\temms"
                     : "=m"(q) : "r"(t.b) : "xmm2", "mm1", "memory");
    CHECK_EQ(q, get64(t.b, 0));
    CHECK_EQ(t_get_x87_tw(), 0xFFFF);
}

TEST(xmm_gpr_extract_forms) {
    /* MOVD r32,xmm / MOVQ r64,xmm take bits 31:0 / 63:0; PEXTRB/PEXTRD/PEXTRQ/EXTRACTPS pick one lane and
     * zero-extend into the GPR (SDM Vol.2). The source register must be unchanged afterwards. */
    v3 t; v3_init(&t, __LINE__);
    unsigned d32 = 0;
    __asm__ volatile("movdqu (%2), %%xmm2\n\tmovd %%xmm2, %0\n\tmovdqu %%xmm2, (%1)"
                     : "=&r"(d32) : "r"(t.o), "r"(t.b) : "xmm2", "memory");
    CHECK_EQ(d32, get32(t.b, 0));
    CHECK_REG("movd r32,xmm leaves the source register intact", t.o, t.b, 16);
#ifdef T_X64
    uint64_t q64 = 0;
    __asm__ volatile("movdqu (%1), %%xmm2\n\tmovq %%xmm2, %0" : "=r"(q64) : "r"(t.b) : "xmm2");
    CHECK_EQ(q64, get64(t.b, 0));
#endif
    if (!cpu_has(F_SSE41)) { t_note("SSE4.1 extract forms skipped: no SSE4.1"); return; }
    for (int k = 0; k < 4; k++) {
        unsigned b = 0xFFFFFFFFu, ps = 0;
        switch (k) {
#define PB(K, L) case K: __asm__ volatile("movdqu (%2), %%xmm2\n\tpextrb $" #L ", %%xmm2, %0\n\textractps $" #K ", %%xmm2, %1" \
                                          : "=&r"(b), "=&r"(ps) : "r"(t.b) : "xmm2"); \
                         CHECKF(b == t.b[L], "pextrb lane %d: 0x%x want 0x%x (zero-extended)", L, b, t.b[L]); break;
        PB(0, 0) PB(1, 7) PB(2, 9) PB(3, 15)
#undef PB
        }
        CHECKF(ps == get32(t.b, k), "extractps lane %d: 0x%x want 0x%x", k, ps, get32(t.b, k));
    }
#ifdef T_X64
    uint64_t q1 = 0;
    __asm__ volatile("movdqu (%1), %%xmm2\n\tpextrq $1, %%xmm2, %0" : "=r"(q1) : "r"(t.b) : "xmm2");
    CHECK_EQ(q1, get64(t.b, 1));
#endif
}

static T_NOINLINE void maskmov(const void *data, const void *mask, void *dst) {
    /* Intel syntax MASKMOVDQU xmm1(data), xmm2(mask); destination implicit DS:[E/RDI] */
    __asm__ volatile("movdqu (%0), %%xmm1\n\tmovdqu (%1), %%xmm2\n\tmaskmovdqu %%xmm2, %%xmm1\n\tsfence"
                     :: "r"(data), "r"(mask), "D"(dst) : "xmm1", "xmm2", "memory");
}
TEST(xmm_maskmovdqu_byte_select) {
    /* SDM: byte i is stored iff bit 7 of mask byte i is set; unselected bytes are not written. */
    __attribute__((aligned(16))) uint8_t data[16], mask[16], buf[48], want[48];
    t_pattern_reg(data, 16, __LINE__, 1);
    static const char *names[] = { "all selected", "none selected", "alternating", "only bit7 counts", "single byte 15" };
    for (int m = 0; m < 5; m++) {
        for (int i = 0; i < 16; i++) {
            switch (m) {
            case 0: mask[i] = 0x80; break;
            case 1: mask[i] = 0x00; break;
            case 2: mask[i] = (i & 1) ? 0xFF : 0x00; break;
            case 3: mask[i] = (i % 3 == 0) ? 0x80 : 0x7F; break;
            default: mask[i] = i == 15 ? 0x80 : 0x7F; break;
            }
        }
        memset(buf, 0xCC, sizeof buf);
        memcpy(want, buf, sizeof want);
        for (int i = 0; i < 16; i++) if (mask[i] & 0x80) want[16 + i] = data[i];
        maskmov(data, mask, buf + 16);
        if (!CHECK_MEM(buf, want, sizeof buf)) t_note("maskmovdqu mask: %s", names[m]);
    }
    /* unaligned destination */
    for (int i = 0; i < 16; i++) mask[i] = (i * 5 % 7 < 3) ? 0x80 : 0;
    memset(buf, 0xCC, sizeof buf); memcpy(want, buf, sizeof want);
    for (int i = 0; i < 16; i++) if (mask[i] & 0x80) want[19 + i] = data[i];
    maskmov(data, mask, buf + 19);
    CHECK_MEM(buf, want, sizeof buf);
}

/* --- page edges: every memory form touches exactly its operand width ---------------------------------
 * A form that reads or writes even one byte more than its architectural width faults on the PAGE_NOACCESS
 * page that follows the buffer (t_guarded_alloc). No fault is expected, so these run in-process: an
 * unexpected fault becomes a CRASH result. LDDQU is deliberately absent: the SDM allows it to fetch an
 * aligned 32-byte block. */
#define XR_EDGE_LD(fname, insn) \
    static T_NOINLINE void fname(const void *pre, const void *m, void *o) { \
        __asm__ volatile("movdqu (%0), %%xmm1\n\t" insn "\n\tmovdqu %%xmm1, (%2)\n\t" \
                         :: "r"(pre), "r"(m), "r"(o) : "xmm1", "memory"); }
XR_EDGE_LD(edge_movd,   "movd (%1), %%xmm1")
XR_EDGE_LD(edge_movss,  "movss (%1), %%xmm1")
XR_EDGE_LD(edge_movq,   "movq (%1), %%xmm1")
XR_EDGE_LD(edge_movsd,  "movsd (%1), %%xmm1")
XR_EDGE_LD(edge_movhps, "movhps (%1), %%xmm1")
XR_EDGE_LD(edge_movlps, "movlps (%1), %%xmm1")
XR_EDGE_LD(edge_movhpd, "movhpd (%1), %%xmm1")
XR_EDGE_LD(edge_movlpd, "movlpd (%1), %%xmm1")
XR_EDGE_LD(edge_movdqu, "movdqu (%1), %%xmm1")
XR_EDGE_LD(edge_movups, "movups (%1), %%xmm1")
XR_EDGE_LD(edge_pinsrw, "pinsrw $5, (%1), %%xmm1")
XR_EDGE_LD(edge_addss,  "addss (%1), %%xmm1")
XR_EDGE_LD(edge_addsd,  "addsd (%1), %%xmm1")
XR_EDGE_LD(edge_movddup, "movddup (%1), %%xmm1")
XR_EDGE_LD(edge_pinsrb, "pinsrb $15, (%1), %%xmm1")
XR_EDGE_LD(edge_insertps, "insertps $0x20, (%1), %%xmm1")
XR_EDGE_LD(edge_pmovzxbw, "pmovzxbw (%1), %%xmm1")
#define XR_EDGE_ST(fname, insn) \
    static T_NOINLINE void fname(const void *src, void *m) { \
        __asm__ volatile("movdqu (%0), %%xmm1\n\t" insn "\n\t" :: "r"(src), "r"(m) : "xmm1", "memory"); }
XR_EDGE_ST(edst_movd,   "movd %%xmm1, (%1)")
XR_EDGE_ST(edst_movss,  "movss %%xmm1, (%1)")
XR_EDGE_ST(edst_movq,   "movq %%xmm1, (%1)")
XR_EDGE_ST(edst_movsd,  "movsd %%xmm1, (%1)")
XR_EDGE_ST(edst_movlps, "movlps %%xmm1, (%1)")
XR_EDGE_ST(edst_movhps, "movhps %%xmm1, (%1)")
XR_EDGE_ST(edst_movlpd, "movlpd %%xmm1, (%1)")
XR_EDGE_ST(edst_movhpd, "movhpd %%xmm1, (%1)")
XR_EDGE_ST(edst_movdqu, "movdqu %%xmm1, (%1)")
XR_EDGE_ST(edst_movups, "movups %%xmm1, (%1)")
XR_EDGE_ST(edst_pextrw, "pextrw $3, %%xmm1, (%1)")
XR_EDGE_ST(edst_pextrb, "pextrb $9, %%xmm1, (%1)")
XR_EDGE_ST(edst_extractps, "extractps $2, %%xmm1, (%1)")

TEST(xmm_page_edge_exact_width_access) {
    void *base = NULL;
    uint8_t *p = (uint8_t *)t_guarded_alloc(8192, &base);
    if (!p) SKIP("guarded allocation failed: %lu", GetLastError());
    uint8_t *end = p + 8192;
    __attribute__((aligned(16))) uint8_t pre[16], o[16], e[16];
    t_pattern_reg(pre, 16, __LINE__, 1);
    set32(pre, 0, fbits(1.5f));
    for (int i = 0; i < 8192; i++) p[i] = (uint8_t)(i * 13 + 7);
    set32(end - 4, 0, fbits(2.25f));     /* scalar single operand at the very end */
    static const struct { const char *n; void (*f)(const void *, const void *, void *); int w, kind, feat; } ld[] = {
        /* kind: 0 zero-extend into low w bytes, 1 replace high 8, 2 replace low 8, 3 full 16,
         *       4 pinsrw word 5, 5 addss, 6 addsd, 7 movddup, 8 pinsrb byte 15, 9 insertps lane 2, 10 pmovzxbw */
        { "movd m32", edge_movd, 4, 0, 0 }, { "movss m32", edge_movss, 4, 0, 0 },
        { "movq m64", edge_movq, 8, 0, 0 }, { "movsd m64", edge_movsd, 8, 0, 0 },
        { "movhps m64", edge_movhps, 8, 1, 0 }, { "movhpd m64", edge_movhpd, 8, 1, 0 },
        { "movlps m64", edge_movlps, 8, 2, 0 }, { "movlpd m64", edge_movlpd, 8, 2, 0 },
        { "movdqu m128", edge_movdqu, 16, 3, 0 }, { "movups m128", edge_movups, 16, 3, 0 },
        { "pinsrw m16", edge_pinsrw, 2, 4, 0 }, { "addss m32", edge_addss, 4, 5, 0 }, { "addsd m64", edge_addsd, 8, 6, 0 },
        { "movddup m64", edge_movddup, 8, 7, F_SSE3 }, { "pinsrb m8", edge_pinsrb, 1, 8, F_SSE41 },
        { "insertps m32", edge_insertps, 4, 9, F_SSE41 }, { "pmovzxbw m64", edge_pmovzxbw, 8, 10, F_SSE41 },
    };
    for (size_t k = 0; k < T_COUNTOF(ld); k++) {
        if (ld[k].feat && !cpu_has((t_feature)ld[k].feat)) { t_note("%s skipped: feature absent", ld[k].n); continue; }
        const uint8_t *m = end - ld[k].w;
        if (ld[k].kind == 6) set64((uint8_t *)end - 8, 0, dbits(2.25));
        memcpy(e, pre, 16);
        switch (ld[k].kind) {
        case 0: memset(e, 0, 16); memcpy(e, m, (size_t)ld[k].w); break;
        case 1: memcpy(e + 8, m, 8); break;
        case 2: memcpy(e, m, 8); break;
        case 3: memcpy(e, m, 16); break;
        case 4: memcpy(e + 10, m, 2); break;
        case 5: set32(e, 0, fbits(1.5f + 2.25f)); break;
        case 6: { __attribute__((aligned(16))) uint8_t pd[16]; memcpy(pd, pre, 16); set64(pd, 0, dbits(1.25));
                  memcpy(e, pd, 16); set64(e, 0, dbits(1.25 + 2.25));
                  memset(o, 0xCC, 16); ld[k].f(pd, m, o);
                  char lbl[80]; snprintf(lbl, sizeof lbl, "%s ending at a guard page", ld[k].n);
                  CHECK_REG(lbl, o, e, 16);
                  set32(end - 4, 0, fbits(2.25f));
                  continue; }
        case 7: memcpy(e, m, 8); memcpy(e + 8, m, 8); break;
        case 8: e[15] = m[0]; break;
        case 9: memcpy(e + 8, m, 4); break;
        default: for (int i = 0; i < 8; i++) { e[2 * i] = m[i]; e[2 * i + 1] = 0; } break;
        }
        memset(o, 0xCC, 16);
        ld[k].f(pre, m, o);
        char lbl[80]; snprintf(lbl, sizeof lbl, "%s ending at a guard page", ld[k].n);
        CHECK_REG(lbl, o, e, 16);
    }
    static const struct { const char *n; void (*f)(const void *, void *); int w, src_off, feat; } st[] = {
        { "movd m32", edst_movd, 4, 0, 0 }, { "movss m32", edst_movss, 4, 0, 0 },
        { "movq m64", edst_movq, 8, 0, 0 }, { "movsd m64", edst_movsd, 8, 0, 0 },
        { "movlps m64", edst_movlps, 8, 0, 0 }, { "movhps m64", edst_movhps, 8, 8, 0 },
        { "movlpd m64", edst_movlpd, 8, 0, 0 }, { "movhpd m64", edst_movhpd, 8, 8, 0 },
        { "movdqu m128", edst_movdqu, 16, 0, 0 }, { "movups m128", edst_movups, 16, 0, 0 },
        { "pextrw m16", edst_pextrw, 2, 6, F_SSE41 }, { "pextrb m8", edst_pextrb, 1, 9, F_SSE41 },
        { "extractps m32", edst_extractps, 4, 8, F_SSE41 },
    };
    __attribute__((aligned(16))) uint8_t src[16], before[32];
    t_pattern_reg(src, 16, __LINE__, 2);
    for (size_t k = 0; k < T_COUNTOF(st); k++) {
        if (st[k].feat && !cpu_has((t_feature)st[k].feat)) { t_note("%s skipped: feature absent", st[k].n); continue; }
        memset(end - 32, 0xA5, 32);
        memcpy(before, end - 32, 32);
        memcpy(before + 32 - st[k].w, src + st[k].src_off, (size_t)st[k].w);
        st[k].f(src, end - st[k].w);
        char lbl[80]; snprintf(lbl, sizeof lbl, "%s store ending at a guard page (and nothing below it)", st[k].n);
        CHECK_REG(lbl, end - 32, before, 32);
    }
    t_guarded_free(base);
}

TEST(xmm_split_page_access_fresh_pages) {
    /* 16-byte unaligned accesses straddling the boundary of two committed pages, the first touch of each
     * page being the split access itself (demand-zero faults restarted in the middle of an XMM access). */
    __attribute__((aligned(16))) uint8_t src[16], o[16], z[16];
    memset(z, 0, 16);
    int bad = 0;
    for (int off = 1; off < 16; off++) {
        uint8_t *mem = (uint8_t *)VirtualAlloc(NULL, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!mem) SKIP("VirtualAlloc failed: %lu", GetLastError());
        uint8_t *a = mem + 4096 - off;
        /* load first (both pages untouched: reads must see zeros), then store, then reload */
        memset(o, 0xCC, 16);
        __asm__ volatile("pcmpeqd %%xmm3, %%xmm3\n\tmovdqu (%1), %%xmm3\n\tmovdqu %%xmm3, (%0)" :: "r"(o), "r"(a) : "xmm3", "memory");
        bad += memcmp(o, z, 16) != 0;
        t_pattern_reg(src, 16, (uint32_t)(__LINE__ + off), 3);
        __asm__ volatile("movdqu (%0), %%xmm4\n\tmovups %%xmm4, (%1)" :: "r"(src), "r"(a) : "xmm4", "memory");
        memset(o, 0xCC, 16);
        __asm__ volatile("movups (%1), %%xmm5\n\tmovdqu %%xmm5, (%0)" :: "r"(o), "r"(a) : "xmm5", "memory");
        if (memcmp(o, src, 16)) { bad++; CHECK_MEM(o, src, 16); }
        bad += mem[4096 - off - 1] != 0 || mem[4096 - off + 16] != 0;
        VirtualFree(mem, 0, MEM_RELEASE);
        /* store first onto fresh pages */
        mem = (uint8_t *)VirtualAlloc(NULL, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!mem) SKIP("VirtualAlloc failed: %lu", GetLastError());
        a = mem + 4096 - off;
        __asm__ volatile("movdqu (%0), %%xmm6\n\tmovdqu %%xmm6, (%1)" :: "r"(src), "r"(a) : "xmm6", "memory");
        if (memcmp(a, src, 16)) { bad++; CHECK_MEM(a, src, 16); }
        bad += mem[4096 - off - 1] != 0 || mem[4096 - off + 16] != 0;
        VirtualFree(mem, 0, MEM_RELEASE);
    }
    CHECKF(bad == 0, "%d split-page XMM access mismatches", bad);
}

/* ======================================================================================
 * 3. XMM vs x87 / MMX / FXSAVE / XSAVE / AVX
 * ====================================================================================== */
TEST(xmm_independent_of_full_x87_stack) {
    t_vregs in, out, want;
    double x[8], y[8];
    for (int i = 0; i < 8; i++) x[i] = 1.25 + i * 3.5;
    t_vregs_pattern(&in, T_VK_XMM, __LINE__);
    memset(&out, 0, sizeof out);
    __asm__ volatile(".irp i,0,1,2,3,4,5,6,7\n\tfldl \\i*8(%2)\n\t.endr\n\t"
                     XR_LOADALL("%0")
                     ".irp i," XR_RL "\n\tpaddq %%xmm\\i, %%xmm\\i\n\t.endr\n\t"
                     XR_STOREALL("%1")
                     ".irp i,0,1,2,3,4,5,6,7\n\tfstpl \\i*8(%3)\n\t.endr\n\t"
                     :: "r"(in.v), "r"(out.v), "r"(x), "r"(y)
                     : XR_CLOB, "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory");
    want = in;
    for (int i = 0; i < NX; i++) { set64(want.v[i], 0, get64(in.v[i], 0) * 2); set64(want.v[i], 1, get64(in.v[i], 1) * 2); }
    check_all(&out, &want, "paddq with a full x87 stack");
    for (int i = 0; i < 8; i++) CHECKF(y[i] == x[7 - i], "x87 st value %d: %g want %g", i, y[i], x[7 - i]);
    CHECK_EQ(t_get_x87_tw(), 0xFFFF);
    CHECKF((t_get_x87_sw() & 0x41) == 0, "x87 stack fault/invalid flagged: sw=0x%x", t_get_x87_sw());
}

TEST(xmm_independent_of_mmx_state) {
    if (!cpu_has(F_MMX)) SKIP("MMX not present");
    t_vregs in, out;
    uint64_t m[8], mo[8];
    for (int i = 0; i < 8; i++) m[i] = t_pattern_word(__LINE__, 20 + i, 0);
    t_vregs_pattern(&in, T_VK_XMM, __LINE__);
    memset(&out, 0, sizeof out);
    __asm__ volatile(".irp i,0,1,2,3,4,5,6,7\n\tmovq \\i*8(%2), %%mm\\i\n\t.endr\n\t"
                     XR_LOADALL("%0") XR_STOREALL("%1")
                     ".irp i,0,1,2,3,4,5,6,7\n\tmovq %%mm\\i, \\i*8(%3)\n\t.endr\n\temms\n\t"
                     :: "r"(in.v), "r"(out.v), "r"(m), "r"(mo)
                     : XR_CLOB, "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7", "memory");
    check_all(&out, &in, "while MMX registers live");
    CHECK_MEM(mo, m, sizeof m);
    run_all((void (*)(void *))xr_stub_mmx, NULL, __LINE__, "around MMX use (asm stub with EMMS)");
    run_all((void (*)(void *))xr_stub_x87, NULL, __LINE__, "around x87 use (asm stub)");
    CHECK_EQ(t_get_x87_tw(), 0xFFFF);
}

TEST(xmm_fxsave_image_layout) {
    if (!cpu_has(F_FXSR)) SKIP("FXSR not present");
    t_vregs in;
    __attribute__((aligned(16))) uint8_t area[512];
    memset(area, 0xEE, sizeof area);
    t_vregs_pattern(&in, T_VK_XMM, __LINE__);
    __asm__ volatile(XR_LOADALL("%0") "fxsave (%1)\n\t" :: "r"(in.v), "r"(area) : XR_CLOB, "memory");
    for (int i = 0; i < NX; i++) {
        char lbl[48]; snprintf(lbl, sizeof lbl, "FXSAVE image xmm%d (offset %d)", i, 160 + 16 * i);
        CHECK_REG(lbl, area + 160 + 16 * i, in.v[i], 16);
    }
    uint32_t mx; memcpy(&mx, area + 24, 4);
    CHECK_EQ(mx, t_get_mxcsr());
    uint32_t mask; memcpy(&mask, area + 28, 4);
    t_info("mxcsr_mask", "0x%08x", mask);
}

TEST(xmm_fxrstor_loads_registers) {
    if (!cpu_has(F_FXSR)) SKIP("FXSR not present");
    t_vregs nv, out;
    __attribute__((aligned(16))) uint8_t area[512];
    t_fxsave(area);
    t_vregs_pattern(&nv, T_VK_XMM, __LINE__);
    for (int i = 0; i < NX; i++) memcpy(area + 160 + 16 * i, nv.v[i], 16);
    memset(&out, 0, sizeof out);
    t_vregs junk; t_vregs_pattern(&junk, T_VK_XMM, __LINE__ + 1);
    __asm__ volatile(XR_LOADALL("%0") "fxrstor (%1)\n\t" XR_STOREALL("%2")
                     :: "r"(junk.v), "r"(area), "r"(out.v) : XR_CLOB, "memory");
    check_all(&out, &nv, "after FXRSTOR of a patched image");
}

static int xsave_sse_ok(void) {
    if (!cpu_has(F_XSAVE)) return 0;
    return (cpu()->xcr0 & T_XS_SSE) != 0;
}

TEST(xmm_xsave_sse_component) {
    if (!xsave_sse_ok()) SKIP("XSAVE not usable (no OSXSAVE / XCR0[1] clear)");
    t_vregs in;
    static __attribute__((aligned(64))) uint8_t area[1024];
    memset(area, 0, sizeof area);
    memset(area + 160, 0xEE, 256);
    t_vregs_pattern(&in, T_VK_XMM, __LINE__);
    __asm__ volatile(XR_LOADALL("%0") "xsave (%1)\n\t" :: "r"(in.v), "r"(area), "a"((uint32_t)T_XS_SSE), "d"(0u) : XR_CLOB, "memory");
    for (int i = 0; i < NX; i++) {
        char lbl[48]; snprintf(lbl, sizeof lbl, "XSAVE image xmm%d", i);
        CHECK_REG(lbl, area + 160 + 16 * i, in.v[i], 16);
    }
    uint64_t bv; memcpy(&bv, area + 512, 8);
    /* SDM 13.6: XINUSE[1]=0 only when XMM regs are all 0 (and MXCSR=1F80h): nonzero patterns -> bit set */
    CHECKF(bv & T_XS_SSE, "XSTATE_BV=0x%llx: SSE bit clear although XMM registers hold nonzero data", (unsigned long long)bv);
    CHECKF((bv & ~T_XS_SSE) == 0, "XSTATE_BV=0x%llx: bits outside RFBM written", (unsigned long long)bv);
    uint32_t mx; memcpy(&mx, area + 24, 4);
    CHECK_EQ(mx, t_get_mxcsr());
}

TEST(xmm_xrstor_init_and_load) {
    if (!xsave_sse_ok()) SKIP("XSAVE not usable (no OSXSAVE / XCR0[1] clear)");
    static __attribute__((aligned(64))) uint8_t area[1024];
    t_vregs junk, out, nv, zero;
    t_vregs_pattern(&junk, T_VK_XMM, __LINE__);
    /* (a) XSTATE_BV[1] = 0 with RFBM[1] = 1: SSE state initialised -> XMM = 0; MXCSR still loaded */
    memset(area, 0, sizeof area);
    memset(area + 160, 0xEE, 256);          /* must be ignored */
    uint32_t mx = 0x1F80; memcpy(area + 24, &mx, 4);
    memset(&out, 0x77, sizeof out);
    __asm__ volatile(XR_LOADALL("%0") "xrstor (%1)\n\t" XR_STOREALL("%2")
                     :: "r"(junk.v), "r"(area), "r"(out.v), "a"((uint32_t)T_XS_SSE), "d"(0u) : XR_CLOB, "memory");
    memset(&zero, 0, sizeof zero);
    check_all(&out, &zero, "after XRSTOR with XSTATE_BV[1]=0 (init)");
    /* (b) XSTATE_BV[1] = 1: registers loaded from the image */
    t_vregs_pattern(&nv, T_VK_XMM, __LINE__);
    for (int i = 0; i < NX; i++) memcpy(area + 160 + 16 * i, nv.v[i], 16);
    uint64_t bv = T_XS_SSE; memcpy(area + 512, &bv, 8);
    __asm__ volatile(XR_LOADALL("%0") "xrstor (%1)\n\t" XR_STOREALL("%2")
                     :: "r"(junk.v), "r"(area), "r"(out.v), "a"((uint32_t)T_XS_SSE), "d"(0u) : XR_CLOB, "memory");
    check_all(&out, &nv, "after XRSTOR with XSTATE_BV[1]=1");
    CHECK_EQ(t_get_mxcsr(), 0x1F80);
}

/* --- AVX interplay (functions carry the ISA hint "avx" in their names) --- */
T_TARGET("avx") static T_NOINLINE void legacy_vs_vex128_avx(const void *y32, const void *x16, void *o_legacy, void *o_vex) {
    __asm__ volatile("vmovdqu (%0), %%ymm1\n\tmovdqu (%1), %%xmm1\n\tvmovdqu %%ymm1, (%2)\n\t"
                     "vmovdqu (%0), %%ymm1\n\tvmovdqu (%1), %%xmm1\n\tvmovdqu %%ymm1, (%3)\n\t"
                     "vmovdqu (%0), %%ymm1\n\tvpxor %%xmm1, %%xmm1, %%xmm1\n\tvmovdqu %%ymm1, 32(%3)\n\t"
                     "vmovdqu (%0), %%ymm1\n\tpxor %%xmm1, %%xmm1\n\tvmovdqu %%ymm1, 32(%2)\n\tvzeroupper\n\t"
                     :: "r"(y32), "r"(x16), "r"(o_legacy), "r"(o_vex) : "xmm1", "memory");
}
T_TARGET("avx") static T_NOINLINE void legacy_all_regs_avx(const t_vregs *in, t_vregs *out) {
    __asm__ volatile(".irp i," XR_RL "\n\tvmovdqu \\i*64(%0), %%ymm\\i\n\t.endr\n\t"
                     ".irp i," XR_RL "\n\tpcmpeqd %%xmm\\i, %%xmm\\i\n\t.endr\n\t"
                     ".irp i," XR_RL "\n\tvmovdqu %%ymm\\i, \\i*64(%1)\n\t.endr\n\tvzeroupper\n\t"
                     :: "r"(in->v), "r"(out->v) : XR_CLOB, "memory");
}
T_TARGET("avx") static T_NOINLINE void vzeroupper_keeps_xmm_avx(const t_vregs *in, t_vregs *out, t_vregs *outy) {
    __asm__ volatile(".irp i," XR_RL "\n\tvmovdqu \\i*64(%0), %%ymm\\i\n\t.endr\n\tvzeroupper\n\t"
                     XR_STOREALL("%1")
                     ".irp i," XR_RL "\n\tvmovdqu %%ymm\\i, \\i*64(%2)\n\t.endr\n\tvzeroupper\n\t"
                     :: "r"(in->v), "r"(out->v), "r"(outy->v) : XR_CLOB, "memory");
}

TEST(xmm_avx_legacy_sse_preserves_upper_vex_zeroes) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    __attribute__((aligned(32))) uint8_t y[32], x[16], ol[64], ov[64], e[32];
    t_pattern_reg(y, 32, __LINE__, 1); t_pattern_reg(x, 16, __LINE__, 2);
    legacy_vs_vex128_avx(y, x, ol, ov);
    memcpy(e, x, 16); memcpy(e + 16, y + 16, 16);
    CHECK_REG("legacy movdqu xmm1: ymm1[255:128] preserved", ol, e, 32);
    memset(e + 16, 0, 16);
    CHECK_REG("VEX vmovdqu xmm1: ymm1[255:128] zeroed", ov, e, 32);
    memset(e, 0, 32);
    CHECK_REG("vpxor xmm1 zero idiom: whole ymm1 zero", ov + 32, e, 32);
    memcpy(e + 16, y + 16, 16);
    CHECK_REG("legacy pxor xmm1 zero idiom: upper preserved", ol + 32, e, 32);
}

TEST(xmm_avx_legacy_writes_all_registers_keep_upper) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    t_vregs in, out, want;
    t_vregs_pattern(&in, T_VK_YMM, __LINE__);
    memset(&out, 0, sizeof out);
    legacy_all_regs_avx(&in, &out);
    want = in;
    for (int i = 0; i < NX; i++) memset(want.v[i], 0xFF, 16);
    t_vregs_check(&out, &want, T_VK_YMM, 0, -1, "after legacy pcmpeqd on xmm (upper must survive)");
}

TEST(xmm_avx_vzeroupper_keeps_low_128) {
    if (!cpu_has(F_AVX)) SKIP("AVX not usable");
    t_vregs in, out, outy, want, wanty;
    t_vregs_pattern(&in, T_VK_YMM, __LINE__);
    memset(&out, 0, sizeof out); memset(&outy, 0x33, sizeof outy);
    vzeroupper_keeps_xmm_avx(&in, &out, &outy);
    memset(&want, 0, sizeof want);
    for (int i = 0; i < NX; i++) memcpy(want.v[i], in.v[i], 16);
    check_all(&out, &want, "xmm after vzeroupper");
    wanty = want;
    t_vregs_check(&outy, &wanty, T_VK_YMM, 0, -1, "ymm after vzeroupper (upper zero)");
}

/* ======================================================================================
 * 4. OS-mediated events without a call: every register on both architectures
 * ====================================================================================== */
TEST(xmm_demand_zero_page_faults) {
    void *mem = VirtualAlloc(NULL, XR_TOUCH_PAGES * 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    REQUIRE(mem != NULL);
    run_all((void (*)(void *))xr_stub_touch, mem, __LINE__, "after 64 demand-zero page faults");
    int ok = 1;
    for (int i = 0; i < XR_TOUCH_PAGES; i++) ok &= ((volatile uint8_t *)mem)[i * 4096] == 1 && ((uint8_t *)mem)[i * 4096 + 1] == 0;
    CHECK(ok);
    /* decommitted and recommitted pages fault again */
    CHECK(VirtualFree(mem, XR_TOUCH_PAGES * 4096, MEM_DECOMMIT));
    CHECK(VirtualAlloc(mem, XR_TOUCH_PAGES * 4096, MEM_COMMIT, PAGE_READWRITE) == mem);
    run_all((void (*)(void *))xr_stub_touch, mem, __LINE__, "after faults on recommitted pages");
    VirtualFree(mem, 0, MEM_RELEASE);
}

TEST(xmm_across_cpuid_loop) {
    run_all((void (*)(void *))xr_stub_cpuid, NULL, __LINE__, "after 600 CPUID (may VM-exit)");
}

TEST(xmm_across_rdtsc_loop) {
    if (!cpu_has(F_TSC)) SKIP("TSC not present");
    run_all((void (*)(void *))xr_stub_rdtsc, NULL, __LINE__, "after 1000 RDTSC");
}

TEST(xmm_across_x87_and_mmx_stubs_10000) {
    int bad = 0, first = 1;
    for (int it = 0; it < 2000; it++) {
        t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(2000 + it % 1000));
        void (*f)(void *) = (it & 1) ? (void (*)(void *))xr_stub_x87 : cpu_has(F_MMX) ? (void (*)(void *))xr_stub_mmx : NULL;
        t_vregs_call(T_VK_XMM, &in, &out, f, NULL);
        int d = diff_regs(&out, &in, 0, NX - 1);
        bad += d;
        if (d && first) { first = 0; check_all(&out, &in, "x87/MMX stub"); }
    }
    CHECKF(bad == 0, "%d mismatches (x87 x1000 / MMX x1000 per iteration, 2000 iterations)", bad);
    CHECK_EQ(t_get_x87_tw(), 0xFFFF);
}

/* --- spinning threads (preemption) --- */
typedef struct spinner {
    t_vregs in, out;
    volatile long *stop;
    long rem;
    int pin;
    unsigned mxcsr_in, mxcsr_out, mxcsr_initial;
    unsigned short cw_initial;
    volatile long started;
    HANDLE h;
} spinner;

static DWORD WINAPI spinner_main(void *p) {
    spinner *s = (spinner *)p;
    s->mxcsr_initial = t_get_mxcsr();
    s->cw_initial = t_get_x87_cw();
    if (s->pin >= 0) t_pin_thread(s->pin);
    t_set_mxcsr(s->mxcsr_in);
    InterlockedExchange(&s->started, 1);
    s->rem = t_vregs_spin(T_VK_XMM, &s->in, &s->out, s->stop, 0x7FFFFFFF);
    s->mxcsr_out = t_get_mxcsr();
    t_set_mxcsr(0x1F80);
    return 0;
}

static int first_cpu(void) {
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) || !pm) return -1;
    for (int i = 0; i < (int)(sizeof pm * 8); i++) if (pm >> i & 1) return i;
    return -1;
}

static spinner *spinners_alloc(int n) {
    return (spinner *)VirtualAlloc(NULL, sizeof(spinner) * (size_t)n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}

/* start n spinners; returns the number started. pin = -1: no pinning */
static int spinners_start(spinner *s, int n, volatile long *stop, int pin, uint32_t tagbase) {
    int started = 0;
    for (int i = 0; i < n; i++) {
        t_vregs_pattern(&s[i].in, T_VK_XMM, tagbase + (uint32_t)i);
        s[i].stop = stop; s[i].pin = pin; s[i].started = 0;
        s[i].mxcsr_in = 0x1F80 | ((unsigned)(i & 3) << 13) | ((i & 4) ? 0x8000u : 0);
        s[i].h = CreateThread(NULL, 64 * 1024, spinner_main, &s[i], STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (!s[i].h) break;
        started++;
    }
    DWORD t0 = GetTickCount();
    for (int i = 0; i < started; i++)
        while (!s[i].started && GetTickCount() - t0 < 5000) Sleep(1);
    return started;
}

static int spinners_join(spinner *s, int n, volatile long *stop, const char *label) {
    InterlockedExchange((volatile LONG *)stop, 1);
    int hung = 0;
    DWORD t0 = GetTickCount();   /* one 10 s budget for all threads, not 10 s each */
    for (int i = 0; i < n; i++) {
        DWORD el = GetTickCount() - t0, left = el < 10000 ? 10000 - el : 0;
        if (WaitForSingleObject(s[i].h, left) != WAIT_OBJECT_0) hung++;
        CloseHandle(s[i].h);
    }
    CHECKF(hung == 0, "%d spinner threads did not finish", hung);
    if (hung) return -1;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        char lbl[96]; snprintf(lbl, sizeof lbl, "%s (thread %d)", label, i);
        bad += t_vregs_check(&s[i].out, &s[i].in, T_VK_XMM, 0, -1, lbl);
        CHECKF(s[i].mxcsr_out == s[i].mxcsr_in, "thread %d MXCSR 0x%x want 0x%x", i, s[i].mxcsr_out, s[i].mxcsr_in);
        CHECKF(s[i].started, "thread %d never started", i);
    }
    return bad;
}

static void spin_group_case(int n, int pinned, DWORD run_ms, uint32_t tagbase, const char *label) {
    int cpu = pinned ? first_cpu() : -1;
    if (pinned && cpu < 0) SKIP("no CPU in the process affinity mask");
    spinner *s = spinners_alloc(n);
    REQUIRE(s != NULL);
    volatile long stop = 0;
    int started = spinners_start(s, n, &stop, cpu, tagbase);
    if (started < n) t_note("only %d of %d threads could be created", started, n);
    if (started < 1) { VirtualFree(s, 0, MEM_RELEASE); SKIP("could not create threads"); }
    Sleep(run_ms);
    spinners_join(s, started, &stop, label);
    t_info("threads", "%d%s", started, pinned ? " (pinned to one CPU)" : "");
    t_info("new_thread_mxcsr", "0x%x", s[0].mxcsr_initial);
    t_info("new_thread_x87_cw", "0x%x", s[0].cw_initial);
    VirtualFree(s, 0, MEM_RELEASE);
}

TEST(xmm_preempt_single_spinner) {
    spin_group_case(1, 0, 200, 100, "after 200 ms spin");
}

TEST(xmm_preempt_2_threads_one_cpu) {
    spin_group_case(2, 1, 300, 200, "after time-slicing with 2 threads on one CPU");
}

TEST(xmm_preempt_16_threads_one_cpu) {
    spin_group_case(16, 1, 400, 300, "after time-slicing with 16 threads on one CPU");
}

TEST(xmm_preempt_64_threads_unpinned) {
    spin_group_case(64, 0, 400, 400, "after 64 threads oversubscribed");
}

TEST(xmm_preempt_256_threads_unpinned) {
    spin_group_case(256, 0, 300, 1200, "after 256 threads oversubscribed");
}

/* Migration: the spinner's affinity is moved across every CPU of the process while it holds XMM state
 * (no call in between). The kernel must carry the register file and MXCSR to the new processor. */
static int process_cpus(int *cpus, int cap) {
    DWORD_PTR pm = 0, sm = 0;
    int n = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return 0;
    for (int i = 0; i < (int)(sizeof pm * 8) && n < cap; i++) if (pm >> i & 1) cpus[n++] = i;
    return n;
}
TEST(xmm_migration_across_cpus_spinner) {
    int cpus[64];
    int n = process_cpus(cpus, 64);
    if (n < 2) SKIP("fewer than 2 CPUs in the process affinity mask (%d)", n);
    spinner *s = spinners_alloc(2);
    REQUIRE(s != NULL);
    volatile long stop = 0;
    int started = spinners_start(s, 2, &stop, -1, 1300);
    if (started < 1) { VirtualFree(s, 0, MEM_RELEASE); SKIP("could not create threads"); }
    int hops = 0, refused = 0;
    for (int i = 0; i < 64; i++) {
        for (int k = 0; k < started; k++) {
            DWORD_PTR m = (DWORD_PTR)1 << cpus[(i + k) % n];   /* the two spinners chase each other */
            if (SetThreadAffinityMask(s[k].h, m)) hops++; else refused++;
        }
        Sleep(2);
    }
    spinners_join(s, started, &stop, "after hopping across all CPUs of the process");
    CHECKF(hops > 0, "SetThreadAffinityMask refused every hop (%d refused, last error %lu)", refused, GetLastError());
    t_info("migration", "%d affinity hops over %d CPUs (%d refused)", hops, n, refused);
    VirtualFree(s, 0, MEM_RELEASE);
}

TEST(xmm_preempt_spin_while_main_yields) {
    /* main thread does XMM work + yields on the same CPU as the spinner */
    int cpu = first_cpu();
    if (cpu < 0) SKIP("no CPU in the process affinity mask");
    spinner *s = spinners_alloc(1);
    REQUIRE(s != NULL);
    volatile long stop = 0;
    REQUIRE(spinners_start(s, 1, &stop, cpu, 500) == 1);
    uintptr_t prev = t_pin_thread(cpu);
    int bad = 0;
    for (int it = 0; it < 200; it++) {
        t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(600 + it));
        t_vregs_call(T_VK_XMM, &in, &out, (void (*)(void *))(void *)SwitchToThread, NULL);
        bad += diff_regs(&out, &in, NX > 8 ? 6 : NX, NX - 1);
    }
    t_unpin_thread(prev);
    spinners_join(s, 1, &stop, "spinner sharing a CPU with a yielding XMM thread");
#ifdef T_X64
    CHECKF(bad == 0, "main thread: %d nonvolatile XMM mismatches across SwitchToThread", bad);
#endif
    VirtualFree(s, 0, MEM_RELEASE);
}

/* --- suspend / context APIs on a spinning thread --- */
static void *ctx_xmm(CONTEXT *c, int i) {
#ifdef T_X64
    return &c->FltSave.XmmRegisters[i];
#else
    return c->ExtendedRegisters + 160 + 16 * i;
#endif
}
static unsigned ctx_mxcsr(CONTEXT *c) {
#ifdef T_X64
    return c->FltSave.MxCsr;
#else
    uint32_t v; memcpy(&v, c->ExtendedRegisters + 24, 4); return v;
#endif
}
static uintptr_t ctx_pc(const CONTEXT *c) {
#ifdef T_X64
    return (uintptr_t)c->Rip;
#else
    return (uintptr_t)c->Eip;
#endif
}
static int ctx_has_xmm(const CONTEXT *c) {
#ifdef T_X64
    return (c->ContextFlags & CONTEXT_FLOATING_POINT) == CONTEXT_FLOATING_POINT;
#else
    return (c->ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS;
#endif
}
#ifdef T_X64
#define XR_CTX_FP CONTEXT_FLOATING_POINT
#else
#define XR_CTX_FP CONTEXT_EXTENDED_REGISTERS
#endif

static const uint8_t *spin_pause(void) {
    const uint8_t *p = (const uint8_t *)(void *)t__xmm_spin;
    for (int i = 0; i < 0x800; i++) if (p[i] == 0xF3 && p[i + 1] == 0x90) return p + i;
    return NULL;
}
/* Suspend h at a point inside the spin loop (registers loaded). Leaves the thread suspended on success. */
static int suspend_in_loop(HANDLE h, CONTEXT *c, DWORD flags) {
    const uint8_t *pz = spin_pause();
    if (!pz) return 0;
    for (int tries = 0; tries < 200; tries++) {
        if (SuspendThread(h) == (DWORD)-1) return 0;
        memset(c, 0, sizeof *c);
        c->ContextFlags = flags | CONTEXT_CONTROL;
        if (GetThreadContext(h, c)) {
            uintptr_t pc = ctx_pc(c);
            if (pc >= (uintptr_t)pz - 24 && pc <= (uintptr_t)pz + 16) return 1;
        }
        ResumeThread(h);
        Sleep(1);
    }
    return 0;
}

TEST(xmm_suspend_resume_spinner_200x) {
    spinner *s = spinners_alloc(1);
    REQUIRE(s != NULL);
    volatile long stop = 0;
    REQUIRE(spinners_start(s, 1, &stop, -1, 700) == 1);
    int fails = 0;
    for (int i = 0; i < 200; i++) {
        if (SuspendThread(s->h) == (DWORD)-1) { fails++; continue; }
        CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_CONTROL;
        if (!GetThreadContext(s->h, &c)) fails++;
        ResumeThread(s->h);
        if (i % 20 == 0) Sleep(1);
    }
    CHECK_EQ(fails, 0);
    spinners_join(s, 1, &stop, "after 200 Suspend/GetThreadContext(CONTROL)/Resume");
    VirtualFree(s, 0, MEM_RELEASE);
}

TEST(xmm_getthreadcontext_reports_spinner_xmm) {
    spinner *s = spinners_alloc(1);
    REQUIRE(s != NULL);
    volatile long stop = 0;
    REQUIRE(spinners_start(s, 1, &stop, -1, 800) == 1);
    CONTEXT c;
    int in_loop = suspend_in_loop(s->h, &c, XR_CTX_FP);
    if (in_loop) {
        CHECKF(ctx_has_xmm(&c), "ContextFlags 0x%lx lost the FP/XMM flag", (unsigned long)c.ContextFlags);
        for (int i = 0; i < NX; i++) {
            char lbl[64]; snprintf(lbl, sizeof lbl, "GetThreadContext xmm%d of suspended thread", i);
            CHECK_REG(lbl, ctx_xmm(&c, i), s->in.v[i], 16);
        }
        CHECKF(ctx_mxcsr(&c) == s->mxcsr_in, "context MXCSR 0x%x want 0x%x", ctx_mxcsr(&c), s->mxcsr_in);
#ifdef T_X64
        t_info("ctx_MxCsr_field", "0x%lx (FltSave.MxCsr 0x%lx)", (unsigned long)c.MxCsr, (unsigned long)c.FltSave.MxCsr);
#endif
        ResumeThread(s->h);
    }
    spinners_join(s, 1, &stop, "after GetThreadContext(FP)");
    VirtualFree(s, 0, MEM_RELEASE);
    if (!in_loop) SKIP("could not catch the spinner inside its loop");
}

TEST(xmm_setthreadcontext_changes_spinner_xmm) {
    spinner *s = spinners_alloc(1);
    REQUIRE(s != NULL);
    volatile long stop = 0;
    REQUIRE(spinners_start(s, 1, &stop, -1, 900) == 1);
    CONTEXT c;
    int in_loop = suspend_in_loop(s->h, &c, XR_CTX_FP);
    t_vregs nv;
    t_vregs_pattern(&nv, T_VK_XMM, 950);
    int set_ok = 0;
    if (in_loop) {
        for (int i = 0; i < NX; i++) memcpy(ctx_xmm(&c, i), nv.v[i], 16);
        c.ContextFlags = XR_CTX_FP;
        set_ok = SetThreadContext(s->h, &c);
        CHECKF(set_ok, "SetThreadContext(FP) failed: %lu", GetLastError());
        ResumeThread(s->h);
    }
    InterlockedExchange((volatile LONG *)&stop, 1);
    CHECK(WaitForSingleObject(s->h, 10000) == WAIT_OBJECT_0);
    CloseHandle(s->h);
    if (in_loop && set_ok) {
        check_all(&s->out, &nv, "spinner after SetThreadContext replaced XMM");
        CHECKF(s->mxcsr_out == s->mxcsr_in, "MXCSR 0x%x want 0x%x", s->mxcsr_out, s->mxcsr_in);
    }
    VirtualFree(s, 0, MEM_RELEASE);
    if (!in_loop) SKIP("could not catch the spinner inside its loop");
}

TEST(xmm_get_set_context_identity_100x) {
    spinner *s = spinners_alloc(1);
    REQUIRE(s != NULL);
    volatile long stop = 0;
    REQUIRE(spinners_start(s, 1, &stop, -1, 1000) == 1);
    int fails = 0;
    for (int i = 0; i < 100; i++) {
        if (SuspendThread(s->h) == (DWORD)-1) { fails++; continue; }
        CONTEXT c; memset(&c, 0, sizeof c);
        c.ContextFlags = CONTEXT_FULL | XR_CTX_FP;
        if (!GetThreadContext(s->h, &c) || !SetThreadContext(s->h, &c)) fails++;
        ResumeThread(s->h);
    }
    CHECK_EQ(fails, 0);
    spinners_join(s, 1, &stop, "after 100 Get+Set of an unmodified CONTEXT");
    VirtualFree(s, 0, MEM_RELEASE);
}

/* ======================================================================================
 * 5. exceptions continued by a vectored handler (children)
 * ====================================================================================== */
typedef struct { const char *name; void (*stub)(void *); char *f, *r; DWORD code; } exc_kind;
static const exc_kind g_exc_kinds[] = {
    { "ud2",   xr_stub_ud2,   xr_f_ud2,  xr_r_ud2,  STATUS_ILLEGAL_INSTRUCTION },
    { "int3",  xr_stub_int3,  xr_f_int3, xr_r_int3, STATUS_BREAKPOINT },
    { "div0",  xr_stub_div0,  xr_f_div,  xr_r_div,  STATUS_INTEGER_DIVIDE_BY_ZERO },
    { "av",    xr_stub_av,    xr_f_av,   xr_r_av,   STATUS_ACCESS_VIOLATION },
    { "hlt",   xr_stub_hlt,   xr_f_hlt,  xr_r_hlt,  STATUS_PRIVILEGED_INSTRUCTION },
    { "misalign", xr_stub_misalign, xr_f_mis, xr_r_mis, STATUS_ACCESS_VIOLATION },
    { "sstep", xr_stub_sstep, NULL, NULL, STATUS_SINGLE_STEP },
};
static const exc_kind *g_ek;
static int g_exc_modify;
static volatile LONG g_exc_hits;
static DWORD g_exc_code, g_exc_flags;
static int g_exc_ctxbad = -1;
static unsigned g_exc_ctxmx = 0xFFFFFFFFu, g_exc_ctxcw = 0xFFFFFFFFu;
/* non-default FP environment held across the fault: MXCSR RC=toward zero + FTZ, all exceptions masked;
 * x87 CW RC=truncate, PC=53 bit, all masked. Both must come back unchanged after the handler continues. */
#define XR_EXC_MXCSR 0xFF80u
#define XR_EXC_X87CW 0x0E7Fu
static t_vregs g_exc_in, g_exc_mod;

static void ctx_set_pc(CONTEXT *c, uintptr_t pc) {
#ifdef T_X64
    c->Rip = pc;
#else
    c->Eip = (DWORD)pc;
#endif
}

static LONG CALLBACK exc_veh(EXCEPTION_POINTERS *ep) {
    CONTEXT *c = ep->ContextRecord;
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    uintptr_t pc = ctx_pc(c);
    if (!g_ek) return EXCEPTION_CONTINUE_SEARCH;
    if (g_ek->f == NULL) {
        if (code != STATUS_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
        c->EFlags &= ~0x100u;
    } else if (pc == (uintptr_t)g_ek->f || (code == STATUS_BREAKPOINT && pc == (uintptr_t)g_ek->f + 1)) {
        ctx_set_pc(c, (uintptr_t)g_ek->r);
    } else {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (InterlockedIncrement(&g_exc_hits) == 1) {
        g_exc_code = code;
        g_exc_flags = c->ContextFlags;
        if (ctx_has_xmm(c)) {
            g_exc_ctxmx = ctx_mxcsr(c);
#ifdef T_X64
            g_exc_ctxcw = c->FltSave.ControlWord;
#else
            { uint16_t fcw; memcpy(&fcw, c->ExtendedRegisters, 2); g_exc_ctxcw = fcw; }   /* FXSAVE image FCW */
#endif
            g_exc_ctxbad = 0;
            for (int i = 0; i < NX; i++) g_exc_ctxbad += memcmp(ctx_xmm(c, i), g_exc_in.v[i], 16) != 0;
        }
        if (g_exc_modify && ctx_has_xmm(c))
            for (int i = 0; i < NX; i++) memcpy(ctx_xmm(c, i), g_exc_mod.v[i], 16);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

CHILD(xmm_exc) {
    if (argc < 2) return 3;
    for (size_t k = 0; k < T_COUNTOF(g_exc_kinds); k++) if (!strcmp(argv[1], g_exc_kinds[k].name)) g_ek = &g_exc_kinds[k];
    if (!g_ek) return 3;
    g_exc_modify = argc > 2 && atoi(argv[2]);
    t_vregs_pattern(&g_exc_in, T_VK_XMM, 1777);
    t_vregs_pattern(&g_exc_mod, T_VK_XMM, 1778);
    PVOID h = AddVectoredExceptionHandler(1, exc_veh);
    if (!h) return 4;
    t_vregs out;
    t_set_x87_cw(XR_EXC_X87CW);
    t_set_mxcsr(XR_EXC_MXCSR);
    t_vregs_call(T_VK_XMM, &g_exc_in, &out, g_ek->stub, NULL);
    unsigned mx_after = t_get_mxcsr(), cw_after = t_get_x87_cw();
    t_set_mxcsr(0x1F80);
    t_set_x87_cw(0x027F);
    RemoveVectoredExceptionHandler(h);
    const t_vregs *want = g_exc_modify ? &g_exc_mod : &g_exc_in;
    int bad = 0;
    for (int i = 0; i < NX; i++) {
        if (!memcmp(out.v[i], want->v[i], 16)) continue;
        bad++;
        char a[80], b[80];
        printf("BAD xmm%d got %s want %s\n", i, t_hex_lanes(a, sizeof a, out.v[i], 16, 8), t_hex_lanes(b, sizeof b, want->v[i], 16, 8));
    }
    printf("XR_RESULT hits=%ld code=0x%08lx ctxflags=0x%08lx ctxbad=%d outbad=%d mx=0x%x cw=0x%x ctxmx=0x%x ctxcw=0x%x\n",
           (long)g_exc_hits, (unsigned long)g_exc_code, (unsigned long)g_exc_flags, g_exc_ctxbad, bad,
           mx_after, cw_after, g_exc_ctxmx, g_exc_ctxcw);
    fflush(stdout);
    return 0;
}

static void exc_case(const char *kind, int modify) {
    const exc_kind *ek = NULL;
    for (size_t k = 0; k < T_COUNTOF(g_exc_kinds); k++) if (!strcmp(kind, g_exc_kinds[k].name)) ek = &g_exc_kinds[k];
    REQUIRE(ek != NULL);
    static t_child_result r;
    char args[64]; snprintf(args, sizeof args, "%s %d", kind, modify);
    REQUIRE(t_child_run("xmm_exc", args, 15000, &r));
    CHECKF(!r.timed_out, "child timed out");
    char line[256];
    if (!t_child_find(&r, "XR_RESULT ", line, sizeof line)) {
        CHECKF(0, "child produced no result: exit 0x%lx exc 0x%lx\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
        return;
    }
    long hits = 0; unsigned long code = 0, flags = 0; int ctxbad = -1, outbad = -1;
    unsigned mx = 0, cw = 0, ctxmx = 0, ctxcw = 0;
    REQUIRE(sscanf(line, "hits=%ld code=0x%lx ctxflags=0x%lx ctxbad=%d outbad=%d mx=0x%x cw=0x%x ctxmx=0x%x ctxcw=0x%x",
                   &hits, &code, &flags, &ctxbad, &outbad, &mx, &cw, &ctxmx, &ctxcw) == 9);
    CHECK_EQ(r.exit_code, 0);
    CHECK_EQ(hits, 1);
    CHECKF(code == ek->code, "exception code 0x%08lx want 0x%08lx", code, (unsigned long)ek->code);
    t_info("context_flags", "0x%08lx", flags);
    if (ctxbad < 0) {
        if (modify) SKIP("exception CONTEXT carries no XMM state (ContextFlags 0x%lx)", flags);
        t_note("exception CONTEXT has no XMM state (ContextFlags 0x%lx): context comparison not possible", flags);
    } else {
        CHECKF(ctxbad == 0, "%d XMM registers in the exception CONTEXT differ from the live values", ctxbad);
        CHECKF((ctxmx & ~0x3Fu) == XR_EXC_MXCSR, "exception CONTEXT MXCSR 0x%x want 0x%x (status flags ignored)", ctxmx, XR_EXC_MXCSR);
        CHECKF((ctxcw & 0x1F3F) == (XR_EXC_X87CW & 0x1F3F), "exception CONTEXT x87 CW 0x%x want 0x%x", ctxcw, XR_EXC_X87CW);
    }
    /* the thread's FP environment is part of the CONTEXT that NtContinue restores */
    CHECKF((mx & ~0x3Fu) == XR_EXC_MXCSR, "MXCSR after continuing 0x%x want 0x%x (status flags ignored)", mx, XR_EXC_MXCSR);
    CHECKF((cw & 0x1F3F) == (XR_EXC_X87CW & 0x1F3F), "x87 control word after continuing 0x%x want 0x%x", cw, XR_EXC_X87CW);
    CHECKF(outbad == 0, "%d XMM registers wrong after continuing (%s)\n%s", outbad,
           modify ? "want the values the handler wrote into the CONTEXT" : "want the values at the fault", r.out);
}

TEST(xmm_exc_ud2_continue) { exc_case("ud2", 0); }
TEST(xmm_exc_int3_continue) { exc_case("int3", 0); }
TEST(xmm_exc_div0_continue) { exc_case("div0", 0); }
TEST(xmm_exc_access_violation_continue) { exc_case("av", 0); }
TEST(xmm_exc_privileged_hlt_continue) { exc_case("hlt", 0); }
TEST(xmm_exc_misaligned_movaps_gp) { exc_case("misalign", 0); }
TEST(xmm_exc_single_step_continue) {
    if (t_sde_model())   /* emulator artifact (README "Emulated validation") */
        t_expect_fail("Intel SDE (%s): Pin cannot single-step the application (a TF trap kills the child)", t_sde_model());
    exc_case("sstep", 0);
}
TEST(xmm_exc_handler_modifies_xmm_ud2) { exc_case("ud2", 1); }
TEST(xmm_exc_handler_modifies_xmm_av) { exc_case("av", 1); }

/* ======================================================================================
 * 6. ABI: calls into C / kernel32 / ntdll (x64: XMM6-15 checked; x86: functional checks only)
 * ====================================================================================== */
static void fn_plain_c(void *arg) { volatile int *p = (volatile int *)arg; *p += 1; }
TEST(xmm_abi_plain_c_call) {
    volatile int n = 0;
    run_abi(fn_plain_c, (void *)&n, __LINE__, "after plain C call");
    CHECK_EQ(n, 1);
}

static void fn_clobber_all(void *arg) { clobber_xmm((uint32_t)(uintptr_t)arg); }
TEST(xmm_abi_callee_uses_all_xmm) {
    run_abi(fn_clobber_all, (void *)7, __LINE__, "after callee overwrote all XMM (ABI-correctly)");
}

static T_NOINLINE double nest(int depth, double x) {
    volatile double v = x * 1.0000001 + depth;
    if (depth == 0) { clobber_xmm(9); return v; }
    return nest(depth - 1, v) + v * 0.5;
}
static T_NOINLINE void fn_nested(void *arg) { *(double *)arg = nest(64, 1.0); }
TEST(xmm_abi_nested_calls_depth64) {
    volatile double r = 0, r2 = 0;   /* volatile: compare stored doubles, not x87 extended temporaries */
    run_abi(fn_nested, (void *)&r, __LINE__, "after 64-deep nested calls");
    fn_nested((void *)&r2);
    CHECKF(r == r2 && r > 64, "r=%.17g r2=%.17g", r, r2);
}

static void fn_sleep0(void *a) { (void)a; Sleep(0); }
static void fn_sleep1(void *a) { (void)a; Sleep(1); }
static void fn_switch(void *a) { (void)a; SwitchToThread(); }
TEST(xmm_abi_sleep0) { run_abi(fn_sleep0, NULL, __LINE__, "after Sleep(0)"); }
TEST(xmm_abi_sleep1) { run_abi(fn_sleep1, NULL, __LINE__, "after Sleep(1)"); }
TEST(xmm_abi_switchtothread) { run_abi(fn_switch, NULL, __LINE__, "after SwitchToThread"); }

static void fn_sleep0_x10000(void *a) { (void)a; for (int i = 0; i < 10000; i++) Sleep(0); }
TEST(xmm_abi_sleep0_10000_in_one_call) { run_abi(fn_sleep0_x10000, NULL, __LINE__, "after 10000 x Sleep(0)"); }

TEST(xmm_abi_switchtothread_10000_iterations) {
    int bad = 0, first = 1, changed = 0;
    for (int it = 0; it < 10000; it++) {
        t_vregs in, out;
        t_vregs_pattern(&in, T_VK_XMM, (uint32_t)(it % 4000));
        t_vregs_call(T_VK_XMM, &in, &out, fn_switch, NULL);
#ifdef T_X64
        int d = diff_regs(&out, &in, 6, 15);
        bad += d;
        if (d && first) { first = 0; t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "SwitchToThread iteration"); }
#else
        changed += diff_regs(&out, &in, 0, NX - 1) != 0;
#endif
    }
    CHECKF(bad == 0, "%d nonvolatile mismatches over 10000 SwitchToThread calls", bad);
    (void)first; (void)changed;
}

TEST(xmm_abi_sleep0_after_x87_use) {
    x87_prior_use();
    run_abi(fn_sleep0, NULL, __LINE__, "after x87 use + Sleep(0)");
    x87_prior_use();
    run_abi(fn_switch, NULL, __LINE__, "after x87 use + SwitchToThread");
}

typedef struct { HANDLE ev; DWORD r; } wait_ctx;
static void fn_wait(void *a) { wait_ctx *w = (wait_ctx *)a; w->r = WaitForSingleObject(w->ev, 2); }
TEST(xmm_abi_wait_signaled) {
    wait_ctx w = { CreateEventW(NULL, TRUE, TRUE, NULL), 0 };
    REQUIRE(w.ev != NULL);
    run_abi(fn_wait, &w, __LINE__, "after WaitForSingleObject (signaled)");
    CHECK_EQ(w.r, WAIT_OBJECT_0);
    CloseHandle(w.ev);
}
TEST(xmm_abi_wait_timeout) {
    wait_ctx w = { CreateEventW(NULL, TRUE, FALSE, NULL), 0 };
    REQUIRE(w.ev != NULL);
    run_abi(fn_wait, &w, __LINE__, "after WaitForSingleObject (timeout)");
    CHECK_EQ(w.r, WAIT_TIMEOUT);
    CloseHandle(w.ev);
}

typedef struct { HANDLE a, b; int n; volatile LONG done; } pingpong;
static DWORD WINAPI pong_main(void *p) {
    pingpong *pp = (pingpong *)p;
    for (int i = 0; i < pp->n; i++) {
        if (WaitForSingleObject(pp->a, 5000) != WAIT_OBJECT_0) break;
        clobber_xmm((uint32_t)i);
        SetEvent(pp->b);
    }
    return 0;
}
static void fn_ping(void *p) {
    pingpong *pp = (pingpong *)p;
    for (int i = 0; i < pp->n; i++) {
        SetEvent(pp->a);
        if (WaitForSingleObject(pp->b, 5000) != WAIT_OBJECT_0) return;
        pp->done++;
    }
}
TEST(xmm_abi_event_pingpong_1000) {
    pingpong pp = { CreateEventW(NULL, FALSE, FALSE, NULL), CreateEventW(NULL, FALSE, FALSE, NULL), 1000, 0 };
    REQUIRE(pp.a && pp.b);
    HANDLE t = CreateThread(NULL, 0, pong_main, &pp, 0, NULL);
    if (!t) SKIP("CreateThread failed: %lu", GetLastError());
    run_abi(fn_ping, &pp, __LINE__, "after 1000 event ping-pongs with an XMM-clobbering thread");
    CHECK_EQ(pp.done, 1000);
    CHECK(WaitForSingleObject(t, 5000) == WAIT_OBJECT_0);
    CloseHandle(t); CloseHandle(pp.a); CloseHandle(pp.b);
}

typedef struct { HANDLE mtx, held; CRITICAL_SECTION cs; int use_cs; DWORD r; } contend;
static DWORD WINAPI holder_main(void *p) {
    contend *c = (contend *)p;
    if (c->use_cs) EnterCriticalSection(&c->cs); else WaitForSingleObject(c->mtx, 5000);
    SetEvent(c->held);
    clobber_xmm(77);
    Sleep(50);
    if (c->use_cs) LeaveCriticalSection(&c->cs); else ReleaseMutex(c->mtx);
    return 0;
}
static void fn_contend(void *p) {
    contend *c = (contend *)p;
    WaitForSingleObject(c->held, 5000);
    if (c->use_cs) { EnterCriticalSection(&c->cs); c->r = WAIT_OBJECT_0; LeaveCriticalSection(&c->cs); }
    else { c->r = WaitForSingleObject(c->mtx, 5000); if (c->r == WAIT_OBJECT_0) ReleaseMutex(c->mtx); }
}
static void contend_case(int use_cs, const char *label) {
    contend c; memset(&c, 0, sizeof c);
    c.use_cs = use_cs;
    c.mtx = CreateMutexW(NULL, FALSE, NULL);
    c.held = CreateEventW(NULL, TRUE, FALSE, NULL);
    InitializeCriticalSection(&c.cs);
    REQUIRE(c.mtx && c.held);
    HANDLE t = CreateThread(NULL, 0, holder_main, &c, 0, NULL);
    if (!t) SKIP("CreateThread failed: %lu", GetLastError());
    run_abi(fn_contend, &c, __LINE__, label);
    CHECK_EQ(c.r, WAIT_OBJECT_0);
    CHECK(WaitForSingleObject(t, 5000) == WAIT_OBJECT_0);
    CloseHandle(t); CloseHandle(c.mtx); CloseHandle(c.held); DeleteCriticalSection(&c.cs);
}
TEST(xmm_abi_mutex_contention) { contend_case(0, "after blocking on a contended mutex"); }
TEST(xmm_abi_critical_section_contention) { contend_case(1, "after blocking on a contended critical section"); }

static void fn_ticks(void *a) {
    ULONGLONG t0 = GetTickCount64(), t1 = t0;
    LONG64 n = 0;
    uint64_t q0 = t_qpc();
    while (t1 == t0 && t_qpc_ms(q0, t_qpc()) < 500) { t1 = GetTickCount64(); n++; }
    *(ULONGLONG *)a = t1 - t0;
}
TEST(xmm_abi_gettickcount64_loop) {
    ULONGLONG d = 0;
    run_abi(fn_ticks, &d, __LINE__, "after GetTickCount64 loop until tick");
    CHECK(d > 0);
}

typedef LONG (NTAPI *pNtYield)(void);
static void fn_ntyield(void *a) {
    pNtYield f = (pNtYield)a;
    for (int i = 0; i < 1000; i++) f();
}
TEST(xmm_abi_ntyieldexecution_1000) {
    pNtYield f = (pNtYield)t_proc("ntdll.dll", "NtYieldExecution");
    if (!f) SKIP("ntdll!NtYieldExecution not exported");
    run_abi(fn_ntyield, (void *)f, __LINE__, "after 1000 NtYieldExecution system calls");
    LONG st = f();
    CHECKF(st == 0 || st == 0x40000024 /* STATUS_NO_YIELD_PERFORMED */, "NtYieldExecution status 0x%lx", (unsigned long)st);
}

typedef struct { int ok; } mix_ctx;
static void fn_syscall_mix(void *a) {
    mix_ctx *m = (mix_ctx *)a;
    int ok = 1;
    for (int i = 0; i < 50; i++) {
        HANDLE e = CreateEventW(NULL, TRUE, FALSE, NULL), d = NULL;
        ok &= e != NULL;
        ok &= SetEvent(e) && ResetEvent(e);
        ok &= DuplicateHandle(GetCurrentProcess(), e, GetCurrentProcess(), &d, 0, FALSE, DUPLICATE_SAME_ACCESS);
        if (d) CloseHandle(d);
        if (e) CloseHandle(e);
        void *p = VirtualAlloc(NULL, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        ok &= p != NULL;
        if (p) { DWORD old; ((volatile char *)p)[4096] = 1; ok &= VirtualProtect(p, 4096, PAGE_READONLY, &old); VirtualFree(p, 0, MEM_RELEASE); }
        FILETIME a1, b1, c1, d1; ok &= GetThreadTimes(GetCurrentThread(), &a1, &b1, &c1, &d1);
        LARGE_INTEGER q; ok &= QueryPerformanceCounter(&q);
        MEMORY_BASIC_INFORMATION mbi; ok &= VirtualQuery(&mbi, &mbi, sizeof mbi) == sizeof mbi;
    }
    m->ok = ok;
}
TEST(xmm_abi_syscall_mix) {
    mix_ctx m = { 0 };
    run_abi(fn_syscall_mix, &m, __LINE__, "after events/handles/VM/thread-times system calls");
    CHECK(m.ok);
}

/* direct calls: fn pointer = the DLL export itself (no C wrapper; x86 helper restores ESP after stdcall) */
TEST(xmm_abi_direct_dll_exports) {
    static const struct { const char *dll, *name; uintptr_t arg; } c[] = {
        { "kernel32.dll", "SwitchToThread", 0 }, { "kernel32.dll", "Sleep", 0 }, { "kernel32.dll", "GetTickCount64", 0 },
        { "kernel32.dll", "GetCurrentThreadId", 0 }, { "kernel32.dll", "IsProcessorFeaturePresent", PF_XMMI64_INSTRUCTIONS_AVAILABLE },
        { "kernel32.dll", "SetLastError", 0x1234 }, { "kernel32.dll", "GetLastError", 0 }, { "ntdll.dll", "NtYieldExecution", 0 },
        { "ntdll.dll", "RtlGetCurrentProcessorNumber", 0 }, { "kernel32.dll", "FlushProcessWriteBuffers", 0 },
    };
    for (size_t k = 0; k < T_COUNTOF(c); k++) {
        void *f = t_proc(c[k].dll, c[k].name);
        if (!f) { t_note("%s!%s not exported", c[k].dll, c[k].name); continue; }
        char lbl[96]; snprintf(lbl, sizeof lbl, "after direct call to %s!%s", c[k].dll, c[k].name);
        run_abi((void (*)(void *))f, (void *)c[k].arg, (uint32_t)(__LINE__ + k), lbl);
    }
}

/* APCs */
static volatile LONG g_apc_count;
static void CALLBACK apc_fn(ULONG_PTR p) { clobber_xmm((uint32_t)p); InterlockedIncrement(&g_apc_count); }
typedef struct { DWORD r; HANDLE self; HANDLE ready; } apc_ctx;
static void fn_apc_self(void *a) {
    apc_ctx *c = (apc_ctx *)a;
    QueueUserAPC(apc_fn, GetCurrentThread(), 1);
    QueueUserAPC(apc_fn, GetCurrentThread(), 2);
    c->r = SleepEx(0, TRUE);
}
TEST(xmm_abi_user_apc_sleepex_self) {
    apc_ctx c = { 0, NULL, NULL };
    g_apc_count = 0;
    run_abi(fn_apc_self, &c, __LINE__, "after SleepEx delivered 2 user APCs");
    CHECK_EQ(c.r, WAIT_IO_COMPLETION);
    CHECK_EQ(g_apc_count, 2);
}
static DWORD WINAPI apc_queuer(void *p) {
    apc_ctx *c = (apc_ctx *)p;
    if (WaitForSingleObject(c->ready, 5000) != WAIT_OBJECT_0) return 1;
    Sleep(20);
    return QueueUserAPC(apc_fn, c->self, 3) ? 0 : 2;
}
static void fn_apc_wait(void *a) { apc_ctx *c = (apc_ctx *)a; SetEvent(c->ready); c->r = SleepEx(5000, TRUE); }
TEST(xmm_abi_user_apc_from_other_thread) {
    apc_ctx c = { 0, OpenThread(THREAD_SET_CONTEXT | SYNCHRONIZE, FALSE, GetCurrentThreadId()), CreateEventW(NULL, TRUE, FALSE, NULL) };
    REQUIRE(c.self && c.ready);
    g_apc_count = 0;
    HANDLE t = CreateThread(NULL, 0, apc_queuer, &c, 0, NULL);
    if (!t) SKIP("CreateThread failed");
    run_abi(fn_apc_wait, &c, __LINE__, "after alertable SleepEx woken by a cross-thread APC");
    CHECK_EQ(c.r, WAIT_IO_COMPLETION);
    CHECK_EQ(g_apc_count, 1);
    DWORD ec = 99;
    CHECK(WaitForSingleObject(t, 5000) == WAIT_OBJECT_0);
    GetExitCodeThread(t, &ec);
    CHECK_EQ(ec, 0);
    CloseHandle(t); CloseHandle(c.self); CloseHandle(c.ready);
}

/* the calling thread moves itself across CPUs inside the call */
typedef struct { int n, hops, cur_ok, cur_bad; int cpus[64]; } mig_ctx;
static void fn_self_migrate(void *a) {
    mig_ctx *m = (mig_ctx *)a;
    DWORD_PTR prev = 0;
    for (int i = 0; i < 4 * m->n && i < 256; i++) {
        int cpu = m->cpus[i % m->n];
        DWORD_PTR old = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
        if (!old) continue;
        if (!prev) prev = old;
        m->hops++;
        /* documented: a thread whose affinity excludes its current CPU is moved before it runs again */
        if (GetCurrentProcessorNumber() == (DWORD)cpu) m->cur_ok++; else m->cur_bad++;
        clobber_xmm((uint32_t)i);
    }
    if (prev) SetThreadAffinityMask(GetCurrentThread(), prev);
}
TEST(xmm_abi_self_migration_across_cpus) {
    static mig_ctx m;
    memset(&m, 0, sizeof m);
    m.n = process_cpus(m.cpus, 64);
    if (m.n < 2) SKIP("fewer than 2 CPUs in the process affinity mask (%d)", m.n);
    run_abi(fn_self_migrate, &m, __LINE__, "after the thread migrated itself across all CPUs");
    CHECK(m.hops > 0);
    t_info("self_migration", "%d hops, on the requested CPU afterwards %d, elsewhere %d", m.hops, m.cur_ok, m.cur_bad);
}

/* thread pool */
typedef struct { HANDLE done; volatile LONG bad; volatile LONG ran; BOOL submitted; DWORD w; } tp_ctx;
static void CALLBACK tp_cb(PTP_CALLBACK_INSTANCE inst, void *p) {
    (void)inst;
    tp_ctx *c = (tp_ctx *)p;
    t_vregs in, out;
    t_vregs_pattern(&in, T_VK_XMM, 1901);
    t_vregs_call(T_VK_XMM, &in, &out, (void (*)(void *))xr_stub_nop, NULL);
    c->bad = diff_regs(&out, &in, 0, NX - 1);
    clobber_xmm(19);
    InterlockedExchange(&c->ran, 1);
    SetEvent(c->done);
}
static void fn_tp(void *a) {
    tp_ctx *c = (tp_ctx *)a;
    c->submitted = TrySubmitThreadpoolCallback(tp_cb, c, NULL);
    c->w = c->submitted ? WaitForSingleObject(c->done, 5000) : WAIT_FAILED;
}
TEST(xmm_abi_threadpool_callback) {
    tp_ctx c; memset(&c, 0, sizeof c);
    c.done = CreateEventW(NULL, TRUE, FALSE, NULL);
    REQUIRE(c.done != NULL);
    run_abi(fn_tp, &c, __LINE__, "after waiting for a thread-pool callback");
    REQUIRE(c.submitted);
    CHECK_EQ(c.w, WAIT_OBJECT_0);
    CHECK_EQ(c.ran, 1);
    CHECKF(c.bad == 0, "%ld XMM registers corrupted inside the pool thread's own round trip", (long)c.bad);
    CloseHandle(c.done);
}

/* qsort callback */
static volatile LONG g_cmp_calls;
static int cmp_dbl(const void *a, const void *b) {
    g_cmp_calls++;
    if ((g_cmp_calls & 63) == 0) clobber_xmm((uint32_t)g_cmp_calls);
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static void fn_qsort(void *a) { qsort(a, 2000, sizeof(double), cmp_dbl); }
TEST(xmm_abi_qsort_callback) {
    static double arr[2000];
    t_rng r; t_rng_seed(&r, 42);
    for (int i = 0; i < 2000; i++) arr[i] = (double)(t_rng_next(&r) >> 11) / 1024.0;
    g_cmp_calls = 0;
    run_abi(fn_qsort, arr, __LINE__, "after qsort with XMM-using comparator");
    int sorted = 1;
    for (int i = 1; i < 2000; i++) sorted &= arr[i - 1] <= arr[i];
    CHECK(sorted);
    CHECK(g_cmp_calls > 2000);
}

/* RaiseException continued by a VEH (software exception, in-process) */
#define XR_SWEXC 0xE0584D4Du
static volatile LONG g_swexc_hits;
static LONG CALLBACK veh_sw(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != XR_SWEXC) return EXCEPTION_CONTINUE_SEARCH;
    clobber_xmm(31);
    InterlockedIncrement(&g_swexc_hits);
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void fn_raise(void *a) { (void)a; for (int i = 0; i < 10; i++) RaiseException(XR_SWEXC, 0, 0, NULL); }
TEST(xmm_abi_raiseexception_continued) {
    PVOID h = AddVectoredExceptionHandler(1, veh_sw);
    REQUIRE(h != NULL);
    g_swexc_hits = 0;
    run_abi(fn_raise, NULL, __LINE__, "after 10 RaiseException continued by a VEH");
    RemoveVectoredExceptionHandler(h);
    CHECK_EQ(g_swexc_hits, 10);
}

/* thread creation / termination inside the call */
static DWORD WINAPI short_thread(void *p) { clobber_xmm((uint32_t)(uintptr_t)p); return (DWORD)(uintptr_t)p; }
typedef struct { int ok; } ct_ctx;
static void fn_threads(void *a) {
    ct_ctx *c = (ct_ctx *)a;
    int ok = 1;
    for (int i = 0; i < 32; i++) {
        HANDLE t = CreateThread(NULL, 64 * 1024, short_thread, (void *)(uintptr_t)(i + 1), STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        if (!t) { ok = 0; break; }
        DWORD ec = 0;
        ok &= WaitForSingleObject(t, 5000) == WAIT_OBJECT_0 && GetExitCodeThread(t, &ec) && ec == (DWORD)(i + 1);
        CloseHandle(t);
    }
    c->ok = ok;
}
TEST(xmm_abi_create_join_32_threads) {
    ct_ctx c = { 0 };
    run_abi(fn_threads, &c, __LINE__, "after creating and joining 32 threads");
    CHECK(c.ok);
}

static void fn_vm(void *a) {
    int *ok = (int *)a;
    size_t n = 8u << 20;
    uint8_t *p = (uint8_t *)VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) { *ok = 0; return; }
    for (size_t i = 0; i < n; i += 4096) p[i] = (uint8_t)i;
    *ok = p[4096 * 5] == 0 && VirtualFree(p, 0, MEM_RELEASE);
}
TEST(xmm_abi_virtualalloc_touch_8mb) {
    int ok = 0;
    run_abi(fn_vm, &ok, __LINE__, "after allocating/touching/freeing 8 MB");
    CHECK(ok);
}

static void fn_x87_callee(void *a) {
    t_f80 *io = (t_f80 *)a;
    t_x87_spin(io, io + 8, NULL, 1000);
}
TEST(xmm_abi_callee_uses_x87_stack) {
    t_f80 io[16];
    memset(io, 0, sizeof io);
    for (int i = 0; i < 8; i++) {
        uint64_t m = 0x8000000000000000ull | t_pattern_word(__LINE__, i, 0);
        uint16_t e = (uint16_t)(0x3FFF + i);
        memcpy(io[i].b, &m, 8); memcpy(io[i].b + 8, &e, 2);
    }
    run_abi(fn_x87_callee, io, __LINE__, "after callee used all 8 x87 registers");
    CHECK_MEM(io + 8, io, sizeof(t_f80) * 8);
    CHECK_EQ(t_get_x87_tw(), 0xFFFF);
}

/* MXCSR control bits and the x87 control word are nonvolatile in the x64 ABI */
static void fn_fpenv_calls(void *a) { (void)a; Sleep(0); SwitchToThread(); Sleep(1); GetTickCount64(); }
TEST(xmm_abi_mxcsr_and_x87cw_preserved) {
    unsigned m = 0x1F80 | 0x2000 | 0x8000;    /* RC = round down, FTZ; all exceptions stay masked */
    unsigned short cw = 0x067F;                /* RC = round down, PC = 53-bit, all masked */
    t_set_mxcsr(m);
    t_set_x87_cw(cw);
    run_abi(fn_fpenv_calls, NULL, __LINE__, "after Sleep/SwitchToThread with non-default MXCSR");
    unsigned m2 = t_get_mxcsr();
    unsigned short cw2 = t_get_x87_cw();
    t_set_mxcsr(0x1F80);
#ifdef T_X64
    CHECKF((m2 & 0xFFC0) == (m & 0xFFC0), "MXCSR control bits 0x%x want 0x%x", m2 & 0xFFC0, m & 0xFFC0);
    CHECKF(cw2 == cw, "x87 control word 0x%x want 0x%x", cw2, cw);
#else
    t_info("x86_mxcsr_after_calls", "0x%x (set 0x%x)", m2, m);
    t_info("x86_x87cw_after_calls", "0x%x (set 0x%x)", cw2, cw);
    CHECK(1);
#endif
}

/* informational: what a new thread sees in its XMM registers while the creator holds patterns */
static DWORD WINAPI capture_thread(void *p) {
    t_vregs *o = (t_vregs *)p;
    __asm__ volatile(XR_STOREALL("%0") :: "r"(o->v) : "memory");
    return 0;
}
typedef struct { t_vregs seen; int ok; } nt_ctx;
static void fn_newthread(void *a) {
    nt_ctx *c = (nt_ctx *)a;
    HANDLE t = CreateThread(NULL, 0, capture_thread, &c->seen, 0, NULL);
    c->ok = t && WaitForSingleObject(t, 5000) == WAIT_OBJECT_0;
    if (t) CloseHandle(t);
}
TEST(xmm_new_thread_initial_xmm_info) {
    static nt_ctx c;
    memset(&c, 0, sizeof c);
    t_vregs in, out;
    t_vregs_pattern(&in, T_VK_XMM, __LINE__);
    REQUIRE(t_vregs_call(T_VK_XMM, &in, &out, fn_newthread, &c) == 0);
    check_abi(&out, &in, "after CreateThread+join");
    CHECK(c.ok);
    int zero = 0, creator = 0;
    for (int i = 0; i < NX; i++) {
        static const uint8_t z[16];
        zero += !memcmp(c.seen.v[i], z, 16);
        for (int j = 0; j < NX; j++) creator += !memcmp(c.seen.v[i], in.v[j], 16);
    }
    t_info("new_thread_xmm_zero", "%d of %d registers zero at thread procedure entry", zero, NX);
    if (creator) t_note("%d XMM registers of the new thread hold the creator's values (not a contract violation)", creator);
}
