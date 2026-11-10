/* xsave_roundtrip - XSAVE / XSAVEOPT / XSAVEC / XRSTOR and FXSAVE / FXRSTOR conformance from user mode.
 *
 * Contract under test (Intel SDM Vol.1 ch.13 "Managing State Using the XSAVE Feature Set", Vol.1 10.5
 * "FXSAVE/FXRSTOR", Vol.2 instruction pages XSAVE/XSAVEOPT/XSAVEC/XRSTOR/FXSAVE/FXRSTOR; AMD APM Vol.2 11.5):
 *   - legacy region: FCW@0 FSW@2 abridged FTW@4 FOP@6 FIP@8 FDP@16 MXCSR@24 MXCSR_MASK@28 ST/MM@32+16i
 *     XMM@160+16i; bytes 464..511 are never written by the processor; XRSTOR ignores 416..511 and
 *     MXCSR_MASK; MXCSR_MASK=0 means the default 0xFFBF.
 *   - XSAVE header @512: XSTATE_BV, XCOMP_BV, bytes 16..63 reserved.
 *   - XSAVE/XSAVEOPT pseudocode: XSTATE_BV <- (OLD_BV & ~RFBM) | (XINUSE & RFBM); nothing else in the
 *     header is written; MXCSR+MXCSR_MASK are written iff RFBM[1] | RFBM[2]; RFBM = EDX:EAX & XCR0.
 *   - XSAVEC: XSTATE_BV <- XINUSE & RFBM, XCOMP_BV <- 1<<63 | RFBM; compacted offsets from CPUID.(0Dh,i)
 *     sizes and ECX[1] alignment; components with XINUSE[i]=0 are not written.
 *   - XINUSE[i]=0 only if component i is in its initial configuration (a non-init component always has
 *     XSTATE_BV[i]=1 after XSAVE*).
 *   - XRSTOR: RFBM[i]&XSTATE_BV[i] -> load, RFBM[i]&~XSTATE_BV[i] -> init, ~RFBM[i] -> unchanged; #GP for
 *     XSTATE_BV bits outside XCR0, nonzero header bytes 8..23 (standard), XCOMP_BV outside XCR0 /
 *     XSTATE_BV outside XCOMP_BV / nonzero bytes 16..63 (compacted), reserved MXCSR bits when MXCSR is
 *     loaded, misaligned (64-byte) area. FXSAVE/FXRSTOR need 16-byte alignment.
 *   - x87 init: FCW=037Fh FSW=0 FTW=empty FOP/FIP/FDP=0 ST0-7=0; SSE init: XMM=0; AVX init: upper YMM=0.
 *   - MMX writes set TOP=0, all tags valid and exponent bits 79:64 to 1s; EMMS empties all tags.
 *   - XSETBV / XSAVES / XRSTORS at CPL3 -> #GP.
 *   - FXRSTOR/XRSTOR load FOP/FIP/FDP and (F)XSAVE store them (AMD only with ES=1 unless
 *     CPUID.8000_0008h:EBX[2]); x87 init (XRSTOR XSTATE_BV[0]=0, FNINIT) zeroes them; the OS must keep them
 *     across context switches.
 *   - OS: XSAVE-family accesses to demand-zero / page-straddling areas fault transparently; a too-small
 *     area or a read-only page gives STATUS_ACCESS_VIOLATION with the right access type and address.
 *   Windows: #GP in user mode is delivered as STATUS_ACCESS_VIOLATION or STATUS_PRIVILEGED_INSTRUCTION
 *   (the kernel decodes privileged opcodes); both are accepted, the exact code is recorded as INFO.
 *   kernelbase!GetEnabledXStateFeatures (decomp checked): returns RtlGetEnabledExtendedFeatures(~0), or 3
 *   when that is 0 -> legacy x87|SSE bits are always reported.
 * Vendor-ambiguous details (MXCSR handling for RFBM = AVX only, FIP/FDP/FOP saving policy, header writes
 * by XSAVEC beyond XCOMP_BV) are hard-checked only where the SDM pseudocode is explicit and the CPU is
 * Intel, otherwise recorded as INFO.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> t_note/t_info.
 * Deliberately faulting operations run in CHILD() processes via t_child_run(). See common/API.md.
 *
 * Every "registers then save" step runs in ONE asm routine (xs_seq_*), driven by a request block, so no
 * compiled code runs between loading registers, saving/restoring and reading back. */
#include "t.h"

#define XS_MAX     16384
#define XS_CARE    (T_XS_X87 | T_XS_SSE | T_XS_AVX | T_XS_OPMASK | T_XS_ZMM_HI256 | T_XS_HI16_ZMM)   /* 0xE7 */
#define XS_COMPACT (1ull << 63)
#define XW_MX      0x100u
#define SENT       0x5A
#define SENT64     0x5A5A5A5A5A5A5A5Aull
#ifdef T_X64
#define XW_ARCH    0x1E7u
#else
#define XW_ARCH    0x167u           /* Hi16_ZMM is not accessible outside 64-bit mode */
#endif
#define EXC_AV   0xC0000005u
#define EXC_PRIV 0xC0000096u
#define EXC_ILL  0xC000001Du

typedef struct { T_ALIGN(64) uint8_t b[XS_MAX]; } xs_buf;
static xs_buf g_a, g_b, g_c, g_z, g_fx;
static T_ALIGN(16) uint8_t g_env[32];
static uint64_t g_m;        /* XCR0 & XS_CARE: the components these tests drive */
static int g_kind;          /* widest usable register kind */
static int g_kbw;           /* AVX512BW: 64-bit opmask moves */

/* ---- byte helpers --------------------------------------------------------- */
static uint16_t g16(const void *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t g32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t g64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static void p16(void *p, uint16_t v) { memcpy(p, &v, 2); }
static void p32(void *p, uint32_t v) { memcpy(p, &v, 4); }
static void p64(void *p, uint64_t v) { memcpy(p, &v, 8); }
static int all_eq(const void *p, size_t n, uint8_t v) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) if (b[i] != v) return 0;
    return 1;
}
static long first_ne(const void *p, size_t n, uint8_t v) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) if (b[i] != v) return (long)i;
    return -1;
}
#define CHECK_UNTOUCHED(p, n, what) do { long fn_ = first_ne((p), (n), SENT); \
    CHECKF(fn_ < 0, "%s: byte +%ld written (0x%02x), expected untouched sentinel 0x%02x", (what), fn_, \
           fn_ >= 0 ? ((const uint8_t *)(p))[fn_] : 0, SENT); } while (0)

static int is_intel(void) { return cpu()->is_intel; }

/* ======================= one-routine register/save sequencer ============== */
/* Slots (uintptr_t each). Order of execution inside the asm:
 *  R1, R2 restores -> x87 loads -> MMX loads -> vector loads -> LDMXCSR -> XGETBV(1)#1 -> spin ->
 *  S1, S2 saves -> EMMS -> XGETBV(1)#2 -> vector stores -> FXSAVE readout -> FNSTENV -> STMXCSR */
enum {
    Q_R1 = 0, Q_R2 = 4, Q_X87 = 8, Q_X87N = 9, Q_X87CW = 10, Q_MMX = 11, Q_VIN = 12, Q_MXCSR = 13,
    Q_XIN1 = 14, Q_SPIN = 15, Q_S1 = 16, Q_S2 = 20, Q_XIN2 = 24, Q_VOUT = 25, Q_FXOUT = 26, Q_ENV = 27,
    Q_KBW = 28, Q_STMX = 29, Q_EMMS = 30, Q_STOP = 31, Q_N = 32
};
enum { XR_XRSTOR = 1, XR_FXRSTOR, XR_XRSTOR32, XR_FXRSTOR32 };
enum { XO_XSAVE = 1, XO_XSAVEOPT, XO_XSAVEC, XO_FXSAVE, XO_XSAVE32, XO_XSAVEOPT32, XO_XSAVEC32, XO_FXSAVE32 };
typedef struct xs_req { uintptr_t s[Q_N]; } xs_req;

#ifdef T_X64
#define PS "8"
#define RC "%%rcx"
#define RS "%%rsi"
#define W64 "64"
#define CLOB_GPR "rax", "rcx", "rdx", "rsi"
#define CLOB_X "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", \
               "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
#define CLOB_Z CLOB_X, "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22", "xmm23", \
               "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31"
#else
#define PS "4"
#define RC "%%ecx"
#define RS "%%esi"
#define W64 ""
#define CLOB_GPR "eax", "ecx", "edx", "esi"
#define CLOB_X "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
#define CLOB_Z CLOB_X
#endif
#define CLOB_ST "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)"
#define CLOB_MM "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"
#define CLOB_K "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"

#define SL(n) PS "*" #n "(%[q])"
#define TESTJZ(r) "test " r ", " r "\n\tjz 9f\n\t"
#define STEP_RST(a, b, c, d) \
    "mov " SL(a) ", " RC "\n\t" TESTJZ(RC) \
    "mov " SL(b) ", %%eax\n\tmov " SL(c) ", %%edx\n\tmov " SL(d) ", %%esi\n\t" \
    "cmp $1, %%esi\n\tjne 1f\n\txrstor" W64 " (" RC ")\n\tjmp 9f\n" \
    "1:\tcmp $2, %%esi\n\tjne 2f\n\tfxrstor" W64 " (" RC ")\n\tjmp 9f\n" \
    "2:\tcmp $3, %%esi\n\tjne 3f\n\txrstor (" RC ")\n\tjmp 9f\n" \
    "3:\tcmp $4, %%esi\n\tjne 9f\n\tfxrstor (" RC ")\n" \
    "9:\n\t"
#define STEP_SAV(a, b, c, d) \
    "mov " SL(a) ", " RC "\n\t" TESTJZ(RC) \
    "mov " SL(b) ", %%eax\n\tmov " SL(c) ", %%edx\n\tmov " SL(d) ", %%esi\n\t" \
    "cmp $1, %%esi\n\tjne 1f\n\txsave" W64 " (" RC ")\n\tjmp 9f\n" \
    "1:\tcmp $2, %%esi\n\tjne 2f\n\txsaveopt" W64 " (" RC ")\n\tjmp 9f\n" \
    "2:\tcmp $3, %%esi\n\tjne 3f\n\txsavec" W64 " (" RC ")\n\tjmp 9f\n" \
    "3:\tcmp $4, %%esi\n\tjne 4f\n\tfxsave" W64 " (" RC ")\n\tjmp 9f\n" \
    "4:\tcmp $5, %%esi\n\tjne 5f\n\txsave (" RC ")\n\tjmp 9f\n" \
    "5:\tcmp $6, %%esi\n\tjne 6f\n\txsaveopt (" RC ")\n\tjmp 9f\n" \
    "6:\tcmp $7, %%esi\n\tjne 7f\n\txsavec (" RC ")\n\tjmp 9f\n" \
    "7:\tcmp $8, %%esi\n\tjne 9f\n\tfxsave (" RC ")\n" \
    "9:\n\t"
#define STEP_X87 \
    "mov " SL(8) ", " RC "\n\t" TESTJZ(RC) "fninit\n\t" \
    "mov " SL(10) ", " RS "\n\ttest " RS ", " RS "\n\tjz 1f\n\tfldcw (" RS ")\n" \
    "1:\tmov " SL(9) ", %%esi\n" \
    "2:\ttest %%esi, %%esi\n\tjz 9f\n\tfldt (" RC ")\n\tadd $10, " RC "\n\tdec %%esi\n\tjmp 2b\n" \
    "9:\n\t"
#define STEP_MMX \
    "mov " SL(11) ", " RC "\n\t" TESTJZ(RC) \
    "movq 0(" RC "), %%mm0\n\tmovq 8(" RC "), %%mm1\n\tmovq 16(" RC "), %%mm2\n\tmovq 24(" RC "), %%mm3\n\t" \
    "movq 32(" RC "), %%mm4\n\tmovq 40(" RC "), %%mm5\n\tmovq 48(" RC "), %%mm6\n\tmovq 56(" RC "), %%mm7\n" \
    "9:\n\t"
#define STEP_MXCSR "mov " SL(13) ", " RC "\n\t" TESTJZ(RC) "ldmxcsr (" RC ")\n9:\n\t"
#define STEP_XIN(n) "mov " SL(n) ", " RS "\n\t" TESTJZ(RS) \
    "mov $1, %%ecx\n\txgetbv\n\tmov %%eax, (" RS ")\n\tmov %%edx, 4(" RS ")\n9:\n\t"
#define STEP_SPIN \
    "mov " SL(31) ", " RC "\n\tmov " SL(15) ", %%esi\n" \
    "1:\ttest %%esi, %%esi\n\tjz 9f\n\ttest " RC ", " RC "\n\tjz 2f\n\tcmpl $0, (" RC ")\n\tjne 9f\n" \
    "2:\tpause\n\tdec %%esi\n\tjmp 1b\n9:\n\t"
#define STEP_EMMS "mov " SL(30) ", %%esi\n\ttest %%esi, %%esi\n\tjz 9f\n\temms\n9:\n\t"
#define STEP_FXOUT "mov " SL(26) ", " RC "\n\t" TESTJZ(RC) "fxsave" W64 " (" RC ")\n9:\n\t"
#define STEP_ENV "mov " SL(27) ", " RC "\n\t" TESTJZ(RC) "fnstenv (" RC ")\n9:\n\t"
#define STEP_STMX "mov " SL(29) ", " RC "\n\t" TESTJZ(RC) "stmxcsr (" RC ")\n9:\n\t"

#define R8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define R16(M) R8(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#define R32(M) R16(M) M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) \
               M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31)
#ifdef T_X64
#define RX(M) R16(M)
#define RZ(M) R32(M)
#else
#define RX(M) R8(M)
#define RZ(M) R8(M)
#endif
#define LX(i) "movdqu " #i "*64(" RC "), %%xmm" #i "\n\t"
#define SX(i) "movdqu %%xmm" #i ", " #i "*64(" RC ")\n\t"
#define LY(i) "vmovdqu " #i "*64(" RC "), %%ymm" #i "\n\t"
#define SY(i) "vmovdqu %%ymm" #i ", " #i "*64(" RC ")\n\t"
#define LZ(i) "vmovdqu64 " #i "*64(" RC "), %%zmm" #i "\n\t"
#define SZ(i) "vmovdqu64 %%zmm" #i ", " #i "*64(" RC ")\n\t"
#define LKQ(i) "kmovq 2048+8*" #i "(" RC "), %%k" #i "\n\t"
#define LKW(i) "kmovw 2048+8*" #i "(" RC "), %%k" #i "\n\t"
#define SKQ(i) "kmovq %%k" #i ", 2048+8*" #i "(" RC ")\n\t"
#define SKW(i) "kmovw %%k" #i ", 2048+8*" #i "(" RC ")\n\t"

#define VIN_X  "mov " SL(12) ", " RC "\n\t" TESTJZ(RC) RX(LX) "9:\n\t"
#define VOUT_X "mov " SL(25) ", " RC "\n\t" TESTJZ(RC) RX(SX) "9:\n\t"
#define VIN_Y  "mov " SL(12) ", " RC "\n\t" TESTJZ(RC) RX(LY) "9:\n\t"
#define VOUT_Y "mov " SL(25) ", " RC "\n\t" TESTJZ(RC) RX(SY) "9:\n\t"
#define VIN_Z  "mov " SL(12) ", " RC "\n\t" TESTJZ(RC) RZ(LZ) \
    "mov " SL(28) ", %%esi\n\ttest %%esi, %%esi\n\tjz 1f\n\t" R8(LKQ) "jmp 9f\n1:\t" R8(LKW) "9:\n\t"
#define VOUT_Z "mov " SL(25) ", " RC "\n\t" TESTJZ(RC) RZ(SZ) \
    "mov " SL(28) ", %%esi\n\ttest %%esi, %%esi\n\tjz 1f\n\t" R8(SKQ) "jmp 9f\n1:\t" R8(SKW) "9:\n\t"

#define XS_SEQ(VIN, VOUT) \
    STEP_RST(0, 1, 2, 3) STEP_RST(4, 5, 6, 7) STEP_X87 STEP_MMX VIN STEP_MXCSR STEP_XIN(14) STEP_SPIN \
    STEP_SAV(16, 17, 18, 19) STEP_SAV(20, 21, 22, 23) STEP_EMMS STEP_XIN(24) VOUT STEP_FXOUT STEP_ENV STEP_STMX

static T_NOINLINE void xs_seq_x(xs_req *q) {
    __asm__ volatile(XS_SEQ(VIN_X, VOUT_X) :: [q] "r"(q) : "memory", "cc", CLOB_GPR, CLOB_X, CLOB_ST, CLOB_MM);
}
static T_NOINLINE T_TARGET("avx") void xs_seq_ymm_avx(xs_req *q) {
    __asm__ volatile(XS_SEQ(VIN_Y, VOUT_Y) :: [q] "r"(q) : "memory", "cc", CLOB_GPR, CLOB_X, CLOB_ST, CLOB_MM);
}
static T_NOINLINE T_TARGET("avx512f") void xs_seq_zmm512(xs_req *q) {
    __asm__ volatile(XS_SEQ(VIN_Z, VOUT_Z) :: [q] "r"(q) : "memory", "cc", CLOB_GPR, CLOB_Z, CLOB_K, CLOB_ST, CLOB_MM);
}

static void q_init(xs_req *q) { memset(q, 0, sizeof *q); }
static void q_rst(xs_req *q, int slot, const void *a, uint64_t m, int op) {
    q->s[slot] = (uintptr_t)a; q->s[slot + 1] = (uint32_t)m; q->s[slot + 2] = (uint32_t)(m >> 32); q->s[slot + 3] = (uintptr_t)op;
}
#define q_sav q_rst

/* Run the sequence; afterwards restore a clean FP environment for the C code that follows. */
static void xs_run(xs_req *q, int kind) {
    unsigned short cw = t_get_x87_cw();
    unsigned mx = t_get_mxcsr();
    q->s[Q_KBW] = (uintptr_t)g_kbw;
    if (kind == T_VK_ZMM) xs_seq_zmm512(q);
    else if (kind == T_VK_YMM) xs_seq_ymm_avx(q);
    else xs_seq_x(q);
    t_fninit(); t_set_x87_cw(cw); t_set_mxcsr(mx);
}

/* ======================= state model ======================================= */
typedef struct xs_state {
    uint16_t fcw, fsw, fsw_mask, fop; uint8_t ftw; int st_n;
    uint64_t fip, fdp;
    uint32_t mxcsr;
    t_f80 st[8];
    uint8_t xmm[16][16], ymmh[16][16], zmmh[16][32], hi16[16][64];
    uint64_t k[8];
} xs_state;

static uint32_t g_mxmask_raw = 0xFFFFFFFFu;
static uint32_t mx_mask_raw(void) {
    if (g_mxmask_raw == 0xFFFFFFFFu) { memset(g_fx.b, 0, 512); t_fxsave(g_fx.b); g_mxmask_raw = g32(g_fx.b + 28); }
    return g_mxmask_raw;
}
static uint32_t mx_mask(void) { uint32_t m = mx_mask_raw(); return m ? m : 0xFFBFu; }

static void f80_set(t_f80 *f, uint64_t mant, uint16_t se) { memcpy(f->b, &mant, 8); memcpy(f->b + 8, &se, 2); }
static void f80_normal(t_f80 *f, uint32_t tag, uint32_t i) {
    uint64_t w = t_pattern_word(tag, 0x20 + i, 7);
    f80_set(f, w | (1ull << 63), (uint16_t)(((w >> 8) & 0x8000u) | (0x3FF0u + (w & 0x1Fu))));
}
static const uint16_t k_cws[] = { 0x037F, 0x027F, 0x0F7F, 0x0C7F, 0x007F, 0x0B7F, 0x047F, 0x087F };

static void st_init(xs_state *s) {
    memset(s, 0, sizeof *s);
    s->fcw = 0x037F; s->fsw_mask = 0xFFFF; s->st_n = 8; s->mxcsr = 0x1F80;
}
/* Distinct, legal, non-init state for every component (x87 with masked sticky flags, ES=0). */
static void st_pattern(xs_state *s, uint32_t tag) {
    st_init(s);
    s->fcw = k_cws[tag & 7];
    s->fsw = (uint16_t)((((tag >> 1) + 3u) & 7u) << 11 | ((tag & 1) ? 0x0020u : 0) | ((tag & 4) ? 0x0004u : 0) |
                        ((tag & 8) ? 0x0200u : 0) | ((tag & 16) ? 0x4100u : 0));
    s->ftw = (uint8_t)(t_pattern_word(tag, 0x30, 0) >> 8);
    if (!s->ftw) s->ftw = 0x81;
    for (int i = 0; i < 8; i++) f80_normal(&s->st[i], tag, (uint32_t)i);
    uint32_t mx = 0x1F80u | ((tag & 3u) << 13) | ((tag & 4u) ? 0x8000u : 0) | ((tag >> 3) & 0x3Fu);
    if ((tag & 2) && (mx_mask() & 0x40)) mx |= 0x40;
    s->mxcsr = mx & mx_mask();
    for (int i = 0; i < 16; i++) {
        uint8_t r[64];
        t_pattern_reg(r, 64, tag, (uint32_t)i);
        memcpy(s->xmm[i], r, 16); memcpy(s->ymmh[i], r + 16, 16); memcpy(s->zmmh[i], r + 32, 32);
        t_pattern_reg(s->hi16[i], 64, tag, 16u + (uint32_t)i);
    }
    for (int i = 0; i < 8; i++) s->k[i] = t_pattern_word(tag, 0x40u + (uint32_t)i, 0);
}
/* d takes the parts of s selected by `what` (component bits + XW_MX). */
static void st_take(xs_state *d, const xs_state *s, unsigned what) {
    if (what & 1) { d->fcw = s->fcw; d->fsw = s->fsw; d->fsw_mask = s->fsw_mask; d->ftw = s->ftw; d->st_n = s->st_n;
                    memcpy(d->st, s->st, sizeof d->st); d->fop = s->fop; d->fip = s->fip; d->fdp = s->fdp; }
    if (what & 2) memcpy(d->xmm, s->xmm, sizeof d->xmm);
    if (what & 4) memcpy(d->ymmh, s->ymmh, sizeof d->ymmh);
    if (what & 0x20) memcpy(d->k, s->k, sizeof d->k);
    if (what & 0x40) memcpy(d->zmmh, s->zmmh, sizeof d->zmmh);
    if (what & 0x80) memcpy(d->hi16, s->hi16, sizeof d->hi16);
    if (what & XW_MX) d->mxcsr = s->mxcsr;
}
static void st_reset(xs_state *d, unsigned what) { xs_state z; st_init(&z); st_take(d, &z, what); }
static void st_to_vregs(const xs_state *s, t_vregs *v) {
    memset(v, 0, sizeof *v);
    for (int i = 0; i < 16; i++) {
        memcpy(v->v[i], s->xmm[i], 16); memcpy(v->v[i] + 16, s->ymmh[i], 16); memcpy(v->v[i] + 32, s->zmmh[i], 32);
        memcpy(v->v[16 + i], s->hi16[i], 64);
    }
    for (int i = 0; i < 8; i++) v->k[i] = g_kbw ? s->k[i] : (s->k[i] & 0xFFFF);
}
/* Expected state after: everything init, then n x87 pushes of v[] with cw, then vector loads of p of the
 * given kind (legacy MOVDQU leaves the upper bits alone, VEX.256 zeroes bits 511:256), optional LDMXCSR. */
static void st_loaded(xs_state *e, const xs_state *p, int kind, int n, const t_f80 *v, uint16_t cw, int mx) {
    st_init(e);
    if (n) {
        e->fcw = cw; e->fsw = (uint16_t)(((8 - n) & 7) << 11); e->fsw_mask = 0x3AFF;   /* C0/C2/C3 undefined after FLD */
        e->ftw = 0;
        for (int j = 0; j < n && j < 8; j++) e->ftw |= (uint8_t)(1u << (7 - j));
        for (int i = 0; i < n && i < 8; i++) e->st[i] = v[n - 1 - i];
    }
    if (p) {
        memcpy(e->xmm, p->xmm, sizeof e->xmm);
        if (kind >= T_VK_YMM) memcpy(e->ymmh, p->ymmh, sizeof e->ymmh);
        if (kind == T_VK_ZMM) {
            memcpy(e->zmmh, p->zmmh, sizeof e->zmmh); memcpy(e->hi16, p->hi16, sizeof e->hi16);
            for (int i = 0; i < 8; i++) e->k[i] = g_kbw ? p->k[i] : (p->k[i] & 0xFFFF);
        }
        if (mx) e->mxcsr = p->mxcsr;
    }
}
static int st_is_init(const xs_state *e, int comp) {
    int nx = T_NXMM;
    switch (comp) {
    case 0:
        if (e->fcw != 0x037F || (e->fsw & e->fsw_mask) != 0 || e->ftw != 0) return 0;
        return all_eq(e->st, sizeof e->st, 0);
    case 1: return all_eq(e->xmm, (size_t)nx * 16, 0);
    case 2: return all_eq(e->ymmh, (size_t)nx * 16, 0);
    case 5: return all_eq(e->k, sizeof e->k, 0);
    case 6: return all_eq(e->zmmh, (size_t)nx * 32, 0);
    case 7: return all_eq(e->hi16, sizeof e->hi16, 0);
    }
    return 1;
}
static uint16_t f80_tag(const t_f80 *f) {
    uint64_t m; uint16_t se; memcpy(&m, f->b, 8); memcpy(&se, f->b + 8, 2);
    uint16_t ex = se & 0x7FFF;
    if (ex == 0x7FFF) return 2;
    if (ex == 0) return m ? 2 : 1;
    return (m >> 63) ? 0 : 2;          /* unnormal (J bit clear) is special */
}
/* Full tag word from abridged FTW + register contents (SDM 10.5.1.1). */
static uint16_t expect_tw(const xs_state *e) {
    int top = (e->fsw >> 11) & 7; uint16_t tw = 0;
    for (int j = 0; j < 8; j++) {
        int i = (j - top) & 7;
        uint16_t t = (e->ftw >> j & 1) ? f80_tag(&e->st[i]) : 3;
        tw = (uint16_t)(tw | t << (2 * j));
    }
    return tw;
}

/* ---- layout ------------------------------------------------------------------- */
static uint32_t comp_off(uint64_t xcomp, int i) {
    if (!(xcomp & XS_COMPACT)) return cpu()->comp[i].offset;
    return xsave_compacted_offset(xcomp & ~XS_COMPACT, i);
}
static void put_legacy(uint8_t *a, const xs_state *s) {
    p16(a + 0, s->fcw); p16(a + 2, s->fsw); a[4] = s->ftw; a[5] = 0; p16(a + 6, s->fop);
    p64(a + 8, s->fip); p64(a + 16, s->fdp); p32(a + 24, s->mxcsr); p32(a + 28, 0);
    for (int i = 0; i < 8; i++) { memcpy(a + 32 + 16 * i, s->st[i].b, 10); memset(a + 42 + 16 * i, 0, 6); }
    for (int i = 0; i < 16; i++) memcpy(a + 160 + 16 * i, s->xmm[i], 16);
    memset(a + 416, 0, 96);
}
static void put_comps(uint8_t *a, uint64_t xcomp, uint64_t comps, const xs_state *s) {
    if (comps & T_XS_AVX) { uint8_t *p = a + comp_off(xcomp, 2); for (int i = 0; i < 16; i++) memcpy(p + 16 * i, s->ymmh[i], 16); }
    if (comps & T_XS_OPMASK) { uint8_t *p = a + comp_off(xcomp, 5); for (int i = 0; i < 8; i++) p64(p + 8 * i, s->k[i]); }
    if (comps & T_XS_ZMM_HI256) { uint8_t *p = a + comp_off(xcomp, 6); for (int i = 0; i < 16; i++) memcpy(p + 32 * i, s->zmmh[i], 32); }
    if (comps & T_XS_HI16_ZMM) { uint8_t *p = a + comp_off(xcomp, 7); for (int i = 0; i < 16; i++) memcpy(p + 64 * i, s->hi16[i], 64); }
}
/* Standard-format area for g_m with the given XSTATE_BV. */
static void xs_make_std(uint8_t *a, const xs_state *s, uint64_t bv) {
    memset(a, 0, xsave_std_size(g_m));
    put_legacy(a, s); p64(a + 512, bv);
    put_comps(a, 0, g_m, s);
}
/* Compacted-format area: XCOMP_BV = 1<<63 | rfbm. */
static void xs_make_cmp(uint8_t *a, const xs_state *s, uint64_t rfbm, uint64_t bv) {
    memset(a, 0, xsave_compacted_size(rfbm));
    put_legacy(a, s); p64(a + 512, bv); p64(a + 520, XS_COMPACT | rfbm);
    put_comps(a, XS_COMPACT | rfbm, rfbm, s);
}

/* ---- comparisons ------------------------------------------------------------------ */
enum { CF_QUIET = 1, CF_BV = 2 };
#define REP(cond, bit, ...) do { int ok_ = !!(cond); if (!ok_) bad |= (bit); \
    if (!quiet) t_check(ok_, __FILE__, __LINE__, #cond, __VA_ARGS__); } while (0)
static int cmp_reg(int quiet, const char *nm, const void *got, const void *want, size_t n) {
    if (quiet) return !memcmp(got, want, n);
    return CHECK_REG(nm, got, want, n);
}
/* Compare an XSAVE/FXSAVE image (a, standard when xcomp has no bit 63) with expected state e for the
 * parts in `what`. CF_BV: components whose XSTATE_BV bit is 0 are only required to be init in e.
 * Returns the mask of mismatching parts. */
static unsigned xs_cmp(const uint8_t *a, uint64_t xcomp, const xs_state *e, unsigned what, unsigned fl, const char *lbl) {
    unsigned bad = 0; int quiet = fl & CF_QUIET; char nm[160];
    uint64_t bv = g64(a + 512);
    what &= XW_ARCH;
    static const int comps[] = { 0, 1, 2, 5, 6, 7 };
    for (int ci = 0; ci < 6; ci++) {
        int c = comps[ci];
        if (!(what >> c & 1)) continue;
        if ((fl & CF_BV) && !(bv >> c & 1)) {
            REP(st_is_init(e, c), 1u << c, "%s: XSTATE_BV[%d]=0 (init) but %s state is not in its initial configuration "
                "(XINUSE may only be 0 for init state)", lbl, c, xstate_comp_name(c));
            continue;
        }
        switch (c) {
        case 0:
            REP(g16(a) == e->fcw, 1, "%s: FCW got 0x%04x want 0x%04x", lbl, g16(a), e->fcw);
            REP((g16(a + 2) & e->fsw_mask) == (e->fsw & e->fsw_mask), 1, "%s: FSW got 0x%04x want 0x%04x (mask 0x%04x)",
                lbl, g16(a + 2), e->fsw, e->fsw_mask);
            REP(a[4] == e->ftw, 1, "%s: abridged FTW got 0x%02x want 0x%02x", lbl, a[4], e->ftw);
            for (int i = 0; i < e->st_n; i++) {
                snprintf(nm, sizeof nm, "%s: ST(%d) (10 bytes at +%d)", lbl, i, 32 + 16 * i);
                if (!cmp_reg(quiet, nm, a + 32 + 16 * i, e->st[i].b, 10)) bad |= 1;
            }
            break;
        case 1:
            for (int i = 0; i < T_NXMM; i++) {
                snprintf(nm, sizeof nm, "%s: XMM%d (+%d)", lbl, i, 160 + 16 * i);
                if (!cmp_reg(quiet, nm, a + 160 + 16 * i, e->xmm[i], 16)) bad |= 2;
            }
            break;
        case 2: {
            const uint8_t *p = a + comp_off(xcomp, 2);
            for (int i = 0; i < T_NXMM; i++) {
                snprintf(nm, sizeof nm, "%s: YMM%d[255:128] (+%u)", lbl, i, (unsigned)(p - a) + 16u * (unsigned)i);
                if (!cmp_reg(quiet, nm, p + 16 * i, e->ymmh[i], 16)) bad |= 4;
            }
            break; }
        case 5: {
            const uint8_t *p = a + comp_off(xcomp, 5);
            for (int i = 0; i < 8; i++)
                REP(g64(p + 8 * i) == e->k[i], 0x20, "%s: k%d got 0x%016llx want 0x%016llx", lbl, i,
                    (unsigned long long)g64(p + 8 * i), (unsigned long long)e->k[i]);
            break; }
        case 6: {
            const uint8_t *p = a + comp_off(xcomp, 6);
            for (int i = 0; i < T_NXMM; i++) {
                snprintf(nm, sizeof nm, "%s: ZMM%d[511:256]", lbl, i);
                if (!cmp_reg(quiet, nm, p + 32 * i, e->zmmh[i], 32)) bad |= 0x40;
            }
            break; }
        case 7: {
            const uint8_t *p = a + comp_off(xcomp, 7);
            for (int i = 0; i < 16; i++) {
                snprintf(nm, sizeof nm, "%s: ZMM%d", lbl, 16 + i);
                if (!cmp_reg(quiet, nm, p + 64 * i, e->hi16[i], 64)) bad |= 0x80;
            }
            break; }
        }
    }
    if (what & XW_MX) REP(g32(a + 24) == e->mxcsr, XW_MX, "%s: MXCSR got 0x%08x want 0x%08x", lbl, g32(a + 24), e->mxcsr);
    return bad;
}
static unsigned xs_cmp_vregs(const t_vregs *v, const xs_state *e, int kind, unsigned what, int quiet, const char *lbl) {
    unsigned bad = 0; char nm[160];
    what &= XW_ARCH;
    for (int i = 0; i < T_NXMM; i++) {
        if (what & 2) { snprintf(nm, sizeof nm, "%s: xmm%d", lbl, i); if (!cmp_reg(quiet, nm, v->v[i], e->xmm[i], 16)) bad |= 2; }
        if ((what & 4) && kind >= T_VK_YMM) { snprintf(nm, sizeof nm, "%s: ymm%d[255:128]", lbl, i); if (!cmp_reg(quiet, nm, v->v[i] + 16, e->ymmh[i], 16)) bad |= 4; }
        if ((what & 0x40) && kind == T_VK_ZMM) { snprintf(nm, sizeof nm, "%s: zmm%d[511:256]", lbl, i); if (!cmp_reg(quiet, nm, v->v[i] + 32, e->zmmh[i], 32)) bad |= 0x40; }
    }
    if (kind == T_VK_ZMM) {
        if (what & 0x80) for (int i = 0; i < 16; i++) { snprintf(nm, sizeof nm, "%s: zmm%d", lbl, 16 + i); if (!cmp_reg(quiet, nm, v->v[16 + i], e->hi16[i], 64)) bad |= 0x80; }
        if (what & 0x20) for (int i = 0; i < 8; i++) {
            uint64_t want = g_kbw ? e->k[i] : (e->k[i] & 0xFFFF);     /* KMOVW stores only 16 bits */
            uint64_t got = g_kbw ? v->k[i] : (v->k[i] & 0xFFFF);
            REP(got == want, 0x20, "%s: k%d got 0x%llx want 0x%llx", lbl, i, (unsigned long long)got, (unsigned long long)want);
        }
    }
    return bad;
}
static void check_env(const uint8_t *env, const xs_state *e, const char *lbl) {
    CHECKF(g16(env) == e->fcw, "%s: FNSTENV CW 0x%04x want 0x%04x", lbl, g16(env), e->fcw);
    CHECKF((g16(env + 4) & e->fsw_mask) == (e->fsw & e->fsw_mask), "%s: FNSTENV SW 0x%04x want 0x%04x", lbl, g16(env + 4), e->fsw);
    CHECKF(g16(env + 8) == expect_tw(e), "%s: FNSTENV full tag word 0x%04x want 0x%04x (abridged 0x%02x TOP %d)",
           lbl, g16(env + 8), expect_tw(e), e->ftw, (e->fsw >> 11) & 7);
}
static unsigned comps_of(uint64_t m) { return (unsigned)(m & 0xE7) | ((m & 6) ? XW_MX : 0); }
static unsigned kind_what(int kind) {     /* register parts a vector readout of this kind covers */
    return kind == T_VK_ZMM ? 0xE6u : kind == T_VK_YMM ? 6u : 2u;
}

/* ---- gates -------------------------------------------------------------------- */
static void need_fxsr(void) { if (!cpu_hw(F_FXSR)) SKIP("FXSAVE/FXRSTOR not advertised (CPUID.1:EDX.FXSR=0)"); }
static void need_xsave(void) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_XSAVE)) SKIP("XSAVE not advertised (CPUID.1:ECX.XSAVE=0)");
    if (!c->osxsave) SKIP("XSAVE advertised but CPUID.1:ECX.OSXSAVE=0: the OS did not enable XSAVE");
    g_m = c->xcr0 & XS_CARE;
    g_kbw = cpu_has(F_AVX512BW);
    g_kind = (cpu_has(F_AVX512F) && (g_m & 0xE0) == 0xE0) ? T_VK_ZMM : cpu_has(F_AVX) ? T_VK_YMM : T_VK_XMM;
    REQUIREF(xsave_std_size(g_m) + 256 <= XS_MAX, "standard size %u exceeds test buffer", xsave_std_size(g_m));
    REQUIREF(xsave_compacted_size(g_m) + 256 <= XS_MAX, "compacted size exceeds test buffer");
    xs_state z; st_init(&z);
    xs_make_std(g_z.b, &z, 0);              /* init-everything area, also used to retarget XSAVEOPT tracking */
}
static void need_avx(void) { need_xsave(); if (!cpu_has(F_AVX)) SKIP("AVX not usable (CPUID AVX=%d, XCR0=0x%llx)", cpu_hw(F_AVX), (unsigned long long)cpu()->xcr0); }
static void need_avx512(void) {
    need_xsave();
    if (!cpu_has(F_AVX512F) || (g_m & 0xE0) != 0xE0) SKIP("AVX-512F not usable (CPUID=%d, XCR0=0x%llx)", cpu_hw(F_AVX512F), (unsigned long long)cpu()->xcr0);
}
static void need_xsaveopt(void) { need_xsave(); if (!cpu_has(F_XSAVEOPT)) SKIP("XSAVEOPT not advertised (CPUID.(0Dh,1):EAX[0]=0)"); }
static void need_xsavec(void) { need_xsave(); if (!cpu_has(F_XSAVEC)) SKIP("XSAVEC not advertised (CPUID.(0Dh,1):EAX[1]=0)"); }
static void need_xgetbv1(void) { need_xsave(); if (!cpu_has(F_XGETBV1)) SKIP("XGETBV(ECX=1) not advertised (CPUID.(0Dh,1):EAX[2]=0)"); }

/* Normal x87 test values */
static t_f80 g_vals[9];
static void make_vals(uint32_t tag) { for (int i = 0; i < 9; i++) f80_normal(&g_vals[i], tag, 0x10u + (uint32_t)i); }
static uint16_t g_cw;

static void q_x87(xs_req *q, int n, uint16_t cw) {
    g_cw = cw;
    q->s[Q_X87] = (uintptr_t)g_vals; q->s[Q_X87N] = (uintptr_t)n; q->s[Q_X87CW] = (uintptr_t)&g_cw;
}
static size_t std_size(void) { return xsave_std_size(g_m); }

/* ======================= CPUID / XCR0 enumeration ============================ */
TEST(cpuid_xcr0_rules) {
    need_xsave();
    const t_cpu_info *c = cpu();
    uint64_t x = c->xcr0;
    t_info("xcr0", "0x%llx", (unsigned long long)x);
    t_info("xcr0_supported", "0x%llx", (unsigned long long)c->xcr0_supported);
    t_info("xss_supported", "0x%llx", (unsigned long long)c->xss_supported);
    CHECK_EQ(t_xgetbv(0), x);                                /* XCR0 is stable */
    CHECK(x & T_XS_X87);                                     /* XCR0[0] is always 1 */
    CHECKF((x & ~c->xcr0_supported) == 0, "XCR0 0x%llx has bits not in CPUID.(0Dh,0) EDX:EAX 0x%llx", (unsigned long long)x, (unsigned long long)c->xcr0_supported);
    CHECK((c->xcr0_supported & 3) == 3);                     /* x87 and SSE are always supported */
    if (x & T_XS_AVX) CHECK(x & T_XS_SSE);                    /* XSETBV #GP rules (SDM 13.3) */
    if (x & 0xE0) { CHECK_EQ(x & 0xE0, 0xE0); CHECK(x & T_XS_AVX); }
    CHECK_EQ((x >> 3) & 1, (x >> 4) & 1);                    /* MPX bits are set together */
    CHECK_EQ((x >> 17) & 1, (x >> 18) & 1);                  /* AMX bits are set together */
    CHECKF((x & c->xss_supported) == 0, "supervisor component in XCR0 0x%llx", (unsigned long long)x);
    CHECK(c->xsave_size_enabled >= 576);
    CHECK(c->xsave_size_max >= c->xsave_size_enabled);
    CHECK_EQ(c->xsave_size_enabled, xsave_std_size(x));      /* (0Dh,0).EBX = size for current XCR0 */
    CHECK_EQ(c->xsave_size_max, xsave_std_size(c->xcr0_supported));
    t_info("xsave_size_enabled", "%u", c->xsave_size_enabled);
    t_info("xsave_size_max", "%u", c->xsave_size_max);
    t_info("xsave_subfeatures", "xsaveopt=%d xsavec=%d xgetbv1=%d xsaves=%d xfd=%d", cpu_hw(F_XSAVEOPT), cpu_hw(F_XSAVEC),
           cpu_hw(F_XGETBV1), cpu_hw(F_XSAVES), cpu_hw(F_XFD));
    if ((x & 3) != 3) t_note("XCR0[1] (SSE) is clear: unusual configuration");
}
TEST(cpuid_component_sizes_offsets) {
    need_xsave();
    const t_cpu_info *c = cpu();
    static const struct { int i; uint32_t size; } known[] = { { 2, 256 }, { 3, 64 }, { 4, 64 }, { 5, 64 }, { 6, 512 },
        { 7, 1024 }, { 9, 8 }, { 17, 64 }, { 18, 8192 } };
    int n = 0;
    for (int i = 2; i < 63; i++) {
        if (!(c->xcr0_supported >> i & 1)) continue;
        n++;
        const t_xcomp *x = &c->comp[i];
        CHECKF(x->valid && x->size > 0, "comp %d (%s) supported but CPUID.(0Dh,%d).EAX=0", i, xstate_comp_name(i), i);
        CHECKF(!x->supervisor, "user component %d reports supervisor (ECX[0]=1)", i);
        CHECKF(x->offset >= 576, "comp %d offset %u overlaps legacy region/header", i, x->offset);
        CHECKF(x->offset + x->size <= c->xsave_size_max, "comp %d [%u,+%u) beyond max size %u", i, x->offset, x->size, c->xsave_size_max);
        if (c->xcr0 >> i & 1)
            CHECKF(x->offset + x->size <= c->xsave_size_enabled, "comp %d beyond enabled size %u", i, c->xsave_size_enabled);
        for (size_t k = 0; k < T_COUNTOF(known); k++)
            if (known[k].i == i) CHECKF(x->size == known[k].size, "comp %d (%s) size %u, architectural %u", i, xstate_comp_name(i), x->size, known[k].size);
        for (int j = 2; j < i; j++) {
            if (!(c->xcr0_supported >> j & 1)) continue;
            const t_xcomp *y = &c->comp[j];
            if (i == 19 && (j == 3 || j == 4)) continue;   /* APX reuses the MPX area (mutually exclusive) */
            CHECKF(x->offset >= y->offset + y->size || y->offset >= x->offset + x->size,
                   "standard-format components %d [%u,+%u) and %d [%u,+%u) overlap", i, x->offset, x->size, j, y->offset, y->size);
        }
        t_log("comp %2d %-10s off %5u size %5u align64 %d xfd %d", i, xstate_comp_name(i), x->offset, x->size, x->align64, x->xfd);
    }
    for (int i = 0; i < 63; i++) {
        if (!(c->xss_supported >> i & 1)) continue;
        t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
        CHECKF(r.ecx & 1, "supervisor comp %d without ECX[0]", i);
        CHECKF(r.ebx == 0, "supervisor comp %d has standard offset %u (must be 0)", i, r.ebx);
    }
    t_info("user_components", "%d", n);
}
TEST(cpuid_unsupported_subleaves_zero) {
    need_xsave();
    const t_cpu_info *c = cpu();
    int n = 0;
    for (int i = 2; i < 63; i++) {
        if ((c->xcr0_supported | c->xss_supported) >> i & 1) continue;
        t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
        CHECKF(r.eax == 0 && r.ebx == 0 && r.ecx == 0 && r.edx == 0, "CPUID.(0Dh,%d) for unsupported component = %08x %08x %08x %08x",
               i, r.eax, r.ebx, r.ecx, r.edx);
        n++;
    }
    t_info("unsupported_subleaves", "%d", n);
}
TEST(cpuid_compacted_layout) {
    need_xsavec();
    const t_cpu_info *c = cpu();
    uint64_t x = c->xcr0;
    uint32_t prev_end = 576;
    for (int i = 2; i < 63; i++) {
        if (!(x >> i & 1)) continue;
        uint32_t off = xsave_compacted_offset(x, i);
        CHECKF(off >= prev_end, "compacted comp %d at %u before previous end %u", i, off, prev_end);
        if (c->comp[i].align64) CHECKF(off % 64 == 0, "comp %d needs 64-byte alignment, offset %u", i, off);
        CHECKF(off - prev_end < 64, "comp %d gap %u", i, off - prev_end);
        prev_end = off + c->comp[i].size;
    }
    CHECK_EQ(xsave_compacted_size(x), prev_end);
    CHECKF(xsave_compacted_size(x) <= c->xsave_size_compacted, "compacted size for XCR0 %u > CPUID.(0Dh,1).EBX %u (XCR0|XSS)",
           xsave_compacted_size(x), c->xsave_size_compacted);
    CHECKF(c->xsave_size_compacted >= 576, "CPUID.(0Dh,1).EBX = %u", c->xsave_size_compacted);
    if (x & T_XS_AVX) CHECK_EQ(xsave_compacted_offset(x, 2), 576);
    CHECK_EQ(xsave_compacted_size(1), 576);
    CHECK_EQ(xsave_compacted_size(3), 576);
    t_info("compacted_size_xcr0", "%u", xsave_compacted_size(x));
    t_info("compacted_size_xcr0_xss", "%u", c->xsave_size_compacted);
}
TEST(cpuid_windows_api_mask) {
    need_xsave();
    const t_env *e = t_env_get();
    if (!e->api_xstate_present) SKIP("GetEnabledXStateFeatures not exported (pre-Windows 7 SP1 class system)");
    uint64_t api = e->api_xstate_features, x = cpu()->xcr0;
    t_info("GetEnabledXStateFeatures", "0x%llx", (unsigned long long)api);
    CHECK((api & 3) == 3);     /* decomp: returns RtlGetEnabledExtendedFeatures(~0) or 3 -> legacy bits always present */
    if ((x & XS_CARE) & ~api)
        t_note("XCR0 user bits 0x%llx not reported by GetEnabledXStateFeatures 0x%llx (API/hardware discrepancy)",
               (unsigned long long)((x & XS_CARE) & ~api), (unsigned long long)api);
    if ((api & ~(x | cpu()->xss_supported)) != 0)
        t_note("API reports bits 0x%llx that are neither in XCR0 nor supervisor-supported", (unsigned long long)(api & ~(x | cpu()->xss_supported)));
}

/* ======================= FXSAVE / FXRSTOR ==================================== */
TEST(fxsave_mxcsr_and_mask) {
    need_fxsr();
    xs_req q; q_init(&q);
    uint32_t mx = 0x1F80 | 0x6000 | 0x0001;   /* RC=11, IE flag */
    memset(g_a.b, SENT, 512);
    q.s[Q_MXCSR] = (uintptr_t)&mx;
    q_sav(&q, Q_S1, g_a.b, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    CHECK_EQ(g32(g_a.b + 24), mx);
    uint32_t m = g32(g_a.b + 28);
    t_info("mxcsr_mask", "0x%08x", m);
    if (m == 0) t_note("MXCSR_MASK=0: processor default 0xFFBF applies (no DAZ)");
    uint32_t em = m ? m : 0xFFBF;
    CHECKF((em & 0xFFBF) == 0xFFBF, "MXCSR_MASK 0x%08x lacks architectural bits 0xFFBF", em);
    CHECKF(!(em & 0x80000000u), "MXCSR_MASK bit 31 set: 0x%08x", em);
    t_info("daz_supported", "%d", (em >> 6) & 1);
    if (em & ~0xFFFFu) t_note("MXCSR_MASK has bits above 15: 0x%08x (e.g. AMD MM bit 17)", em);
    CHECK_UNTOUCHED(g_a.b + 464, 48, "FXSAVE bytes 464..511");
}
TEST(fxsave_daz_follows_mask) {
    need_fxsr();
    if (!(mx_mask() & 0x40)) SKIP("DAZ not supported (MXCSR_MASK bit 6 = 0)");
    xs_req q; q_init(&q);
    uint32_t mx = 0x1FC0, got = 0;
    q.s[Q_MXCSR] = (uintptr_t)&mx; q.s[Q_STMX] = (uintptr_t)&got;
    q_sav(&q, Q_S1, g_a.b, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    CHECK_EQ(got, 0x1FC0);
    CHECK_EQ(g32(g_a.b + 24), 0x1FC0);
}
static void fx_x87_push(int n, uint16_t cw, const char *lbl) {
    make_vals(0x77);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, 512);
    q_x87(&q, n, cw);
    q_sav(&q, Q_S1, g_a.b, 0, XO_FXSAVE);
    q.s[Q_ENV] = (uintptr_t)g_env;
    xs_run(&q, T_VK_XMM);
    xs_state e; st_loaded(&e, NULL, 0, n, g_vals, cw, 0);
    e.st_n = n;
    xs_cmp(g_a.b, 0, &e, 1, 0, lbl);
    CHECKF(((g16(g_a.b + 2) >> 11) & 7) == ((8 - n) & 7), "%s: TOP %d want %d", lbl, (g16(g_a.b + 2) >> 11) & 7, (8 - n) & 7);
    CHECKF((g16(g_a.b + 2) & 0x3F) == 0, "%s: exception flags 0x%x after loading normal values", lbl, g16(g_a.b + 2) & 0x3F);
    CHECK_EQ(g16(g_env), cw);
    uint16_t tw = g16(g_env + 8), want = 0;
    for (int j = 0; j < 8; j++) want = (uint16_t)(want | ((j >= 8 - n) ? 0 : 3) << (2 * j));
    CHECKF(tw == want, "%s: full tag word 0x%04x want 0x%04x", lbl, tw, want);
    CHECK_UNTOUCHED(g_a.b + 464, 48, "FXSAVE bytes 464..511");
}
TEST(fxsave_x87_three_values) { need_fxsr(); fx_x87_push(3, 0x037F, "3 pushes"); CHECK_EQ(g_a.b[4], 0xE0); }
TEST(fxsave_x87_full_stack)   { need_fxsr(); fx_x87_push(8, 0x0C7F, "8 pushes"); CHECK_EQ(g_a.b[4], 0xFF); }
TEST(fxsave_x87_overflow_indefinite) {
    need_fxsr();
    make_vals(0x78);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, 512);
    q_x87(&q, 9, 0x037F);                       /* 9th push overflows: masked IE+SF, C1=1, QNaN indefinite */
    q_sav(&q, Q_S1, g_a.b, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    uint16_t sw = g16(g_a.b + 2);
    CHECKF((sw & 0x3AFF) == 0x3A41, "FSW 0x%04x want IE|SF|C1 TOP=7 (0x3A41 under mask 0x3AFF)", sw);
    CHECK_EQ(g_a.b[4], 0xFF);
    t_f80 ind; f80_set(&ind, 0xC000000000000000ull, 0xFFFF);
    CHECK_REG("ST(0) = QNaN indefinite", g_a.b + 32, ind.b, 10);
    for (int i = 1; i < 8; i++) {
        char nm[64]; snprintf(nm, sizeof nm, "ST(%d) after overflow", i);
        CHECK_REG(nm, g_a.b + 32 + 16 * i, g_vals[8 - i].b, 10);
    }
}
TEST(fxsave_mmx_aliasing_and_emms) {
    need_fxsr();
    if (!cpu_has(F_MMX)) SKIP("MMX not advertised");
    static uint64_t mm[8];
    for (int i = 0; i < 8; i++) mm[i] = t_pattern_word(0x55, (uint32_t)i, 0);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, 512); memset(g_b.b, SENT, 512);
    q.s[Q_MMX] = (uintptr_t)mm;
    q_sav(&q, Q_S1, g_a.b, 0, XO_FXSAVE);
    q.s[Q_EMMS] = 1;
    q.s[Q_FXOUT] = (uintptr_t)g_b.b;
    xs_run(&q, T_VK_XMM);
    CHECKF(((g16(g_a.b + 2) >> 11) & 7) == 0, "MMX write must set TOP=0 (FSW 0x%04x)", g16(g_a.b + 2));
    CHECKF(g_a.b[4] == 0xFF, "MMX write must tag all registers valid (abridged FTW 0x%02x)", g_a.b[4]);
    for (int i = 0; i < 8; i++) {
        char nm[64]; snprintf(nm, sizeof nm, "MM%d bits 63:0", i);
        CHECK_REG(nm, g_a.b + 32 + 16 * i, &mm[i], 8);
        CHECKF(g16(g_a.b + 40 + 16 * i) == 0xFFFF, "MM%d exponent bits 79:64 = 0x%04x, MMX writes set them to 1s", i, g16(g_a.b + 40 + 16 * i));
    }
    CHECKF(g_b.b[4] == 0x00, "after EMMS abridged FTW 0x%02x want 0 (all empty)", g_b.b[4]);
    int kept = !memcmp(g_b.b + 32, g_a.b + 32, 128);
    t_info("register_contents_after_emms_retained", "%d", kept);
}
TEST(fxsave_xmm_from_registers) {
    need_fxsr();
    xs_state p; st_pattern(&p, 0x101);
    static t_vregs vin; st_to_vregs(&p, &vin);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, 512);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    xs_state e; st_loaded(&e, &p, T_VK_XMM, 0, NULL, 0, 1);
    xs_cmp(g_a.b, 0, &e, 2 | XW_MX, 0, "FXSAVE");
#ifndef T_X64
    t_info("xmm8_15_area_written_in_32bit_mode", "%d", !all_eq(g_a.b + 288, 128, SENT));
#endif
}
TEST(fxrstor_restores_legacy_state) {
    need_fxsr();
    xs_state p; st_pattern(&p, 0x102);
    put_legacy(g_a.b, &p);
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    xs_req q; q_init(&q);
    memset(g_b.b, SENT, 512);
    q_rst(&q, Q_R1, g_a.b, 0, XR_FXRSTOR);
    q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_FXOUT] = (uintptr_t)g_b.b; q.s[Q_ENV] = (uintptr_t)g_env;
    uint32_t mx = 0; q.s[Q_STMX] = (uintptr_t)&mx;
    xs_run(&q, T_VK_XMM);
    xs_cmp_vregs(&vout, &p, T_VK_XMM, 2, 0, "FXRSTOR -> movdqu");
    xs_cmp(g_b.b, 0, &p, 3 | XW_MX, 0, "FXRSTOR -> FXSAVE");
    CHECK_EQ(mx, p.mxcsr);
    check_env(g_env, &p, "FXRSTOR -> FNSTENV");
}
TEST(fxrstor_full_tag_word_classes) {
    need_fxsr();
    xs_state s; st_init(&s);
    s.fcw = 0x037F; s.fsw = 3 << 11; s.ftw = 0xDB;      /* physical regs 2 and 5 empty */
    t_f80 phys[8];
    f80_set(&phys[0], 0x8000000000000123ull, 0x3FFF);    /* valid */
    f80_set(&phys[1], 0, 0x8000);                        /* -0 : zero */
    f80_set(&phys[2], 0x9999999999999999ull, 0x4001);    /* empty */
    f80_set(&phys[3], 0x8000000000000000ull, 0x7FFF);    /* +inf : special */
    f80_set(&phys[4], 0x0000000000001234ull, 0x0000);    /* denormal : special */
    f80_set(&phys[5], 0x8000000000000001ull, 0x3FFE);    /* empty */
    f80_set(&phys[6], 0xC000000000000001ull, 0xFFFF);    /* QNaN : special */
    f80_set(&phys[7], 0x0000000000001234ull, 0x4000);    /* unnormal : special */
    for (int j = 0; j < 8; j++) s.st[(j - 3) & 7] = phys[j];
    put_legacy(g_a.b, &s);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, 0, XR_FXRSTOR);
    q.s[Q_FXOUT] = (uintptr_t)g_b.b; q.s[Q_ENV] = (uintptr_t)g_env;
    xs_run(&q, T_VK_XMM);
    CHECK_EQ(expect_tw(&s), 0xAEB4);   /* model self-check: R7..R0 = 10 10 11 10 10 11 01 00 */
    check_env(g_env, &s, "FXRSTOR tag classes");
    xs_cmp(g_b.b, 0, &s, 1, 0, "FXRSTOR -> FXSAVE");
}
TEST(fxsave_fxrstor_many_patterns) {
    need_fxsr();
    int bad = 0;
    for (uint32_t t = 0; t < 48; t++) {
        xs_state p; st_pattern(&p, 0x200 + t);
        put_legacy(g_a.b, &p);
        memset(g_b.b, SENT, 512);
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_a.b, 0, XR_FXRSTOR);
        q_sav(&q, Q_S1, g_b.b, 0, XO_FXSAVE);
        xs_run(&q, T_VK_XMM);
        unsigned d = xs_cmp(g_b.b, 0, &p, 3 | XW_MX, CF_QUIET, "pattern");
        if (d && bad++ == 0) xs_cmp(g_b.b, 0, &p, 3 | XW_MX, 0, "first mismatching FXRSTOR/FXSAVE pattern");
        CHECKF(d == 0, "tag 0x%x: parts 0x%x differ", 0x200 + t, d);
    }
}
TEST(fxrstor_ignores_mask_and_software_bytes) {
    need_fxsr();
    xs_state p; st_pattern(&p, 0x103);
    put_legacy(g_a.b, &p);
    p32(g_a.b + 28, 0xDEADBEEF);              /* MXCSR_MASK field is ignored by FXRSTOR */
    memset(g_a.b + 464, 0xA5, 48);            /* software-available bytes */
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, 0, XR_FXRSTOR);
    q.s[Q_FXOUT] = (uintptr_t)g_b.b;
    xs_run(&q, T_VK_XMM);
    xs_cmp(g_b.b, 0, &p, 3 | XW_MX, 0, "FXRSTOR with junk MXCSR_MASK/464..511");
    CHECK_EQ(g32(g_b.b + 28), mx_mask_raw());
}
TEST(fxsave_fdp_and_forms) {
    need_fxsr();
    make_vals(0x79);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, 512); memset(g_b.b, SENT, 512);
    q_x87(&q, 3, 0x037F);
    q_sav(&q, Q_S1, g_a.b, 0, XO_FXSAVE);
    q_sav(&q, Q_S2, g_b.b, 0, XO_FXSAVE32);
    xs_run(&q, T_VK_XMM);
    uintptr_t want_fdp = (uintptr_t)&g_vals[2];          /* operand of the last FLD */
#ifdef T_X64
    /* FDP is updated only for unmasked exceptions when CPUID.(7,0):EBX[6] (FDP_EXCPTN_ONLY) is set, and some
     * implementations do not store the pointers unless FSW.ES=1: 0 or "not written" are acceptable, a
     * value other than the operand address is not. */
    uint64_t fdp = g64(g_a.b + 16), fip = g64(g_a.b + 8);
    CHECKF(fdp == 0 || fdp == SENT64 || fdp == want_fdp, "FXSAVE64 FDP 0x%llx is neither 0 nor the last x87 operand 0x%llx",
           (unsigned long long)fdp, (unsigned long long)want_fdp);
    t_info("fxsave64_fip", "0x%llx", (unsigned long long)fip);
    t_info("fxsave64_fdp", "0x%llx", (unsigned long long)fdp);
    uint32_t fdp32 = g32(g_b.b + 16);
    CHECKF(fdp32 == 0 || fdp32 == (uint32_t)SENT64 || fdp32 == (uint32_t)want_fdp, "FXSAVE (32-bit form) FDP 0x%x", fdp32);
    if (fdp == want_fdp && fdp32 != 0 && fdp32 != (uint32_t)SENT64) CHECK_EQ(fdp32, (uint32_t)fdp);
    t_info("fxsave32_fcs_fds", "fcs=0x%04x fds=0x%04x", g16(g_b.b + 12), g16(g_b.b + 20));
    /* everything except the pointer fields is identical between the two forms */
    CHECK_MEM(g_b.b, g_a.b, 6);
    CHECK_MEM(g_b.b + 24, g_a.b + 24, 464 - 24);
#else
    uint32_t fdp = g32(g_a.b + 16);
    CHECKF(fdp == 0 || fdp == (uint32_t)SENT64 || fdp == (uint32_t)want_fdp, "FXSAVE FDP 0x%x is neither 0 nor the last x87 operand 0x%x", fdp, (unsigned)want_fdp);
    t_info("fxsave_fip_fcs", "fip=0x%08x fcs=0x%04x", g32(g_a.b + 8), g16(g_a.b + 12));
    t_info("fxsave_fdp_fds", "fdp=0x%08x fds=0x%04x", fdp, g16(g_a.b + 20));
    CHECK_MEM(g_b.b, g_a.b, 512);
#endif
    t_info("fop", "0x%04x", g16(g_a.b + 6));
    t_info("fdp_excptn_only", "%u", (cpu()->l7_0.ebx >> 6) & 1);
    t_info("fcs_fds_deprecated", "%u", (cpu()->l7_0.ebx >> 13) & 1);
}

/* ---- x87 pointer registers FOP/FIP/FDP through (F)XRSTOR -> (F)XSAVE -------------------------------
 * Intel SDM: FXRSTOR/XRSTOR load FOP, FIP, FDP and FXSAVE/XSAVE store them (FDP_EXCPTN_ONLY / FOP
 * deprecation only change when the processor UPDATES them). AMD stores them only when FSW.ES=1 unless
 * CPUID.8000_0008h:EBX[2] (XSaveErPtr, "always save error pointers") is set -> strict only for Intel or
 * XSaveErPtr; elsewhere loaded / zero / not-written are accepted and recorded.
 * Layout: FOP@6; 64-bit forms FIP@8 (8 B) FDP@16 (8 B); 32-bit forms FIP@8 FCS@12 FDP@16 FDS@20.
 * FCS/FDS are not compared: deprecated (CPUID.(7,0):EBX[13]) and not preserved by a 64-bit kernel's
 * context switch of a 32-bit process. */
#define PTR_FOP 0x05A3u
#ifdef T_X64
#define PTR_FIP 0x00007FF612345670ull
#define PTR_FDP 0x00007FF6ABCDEF08ull
#else
#define PTR_FIP 0x12345670ull
#define PTR_FDP 0x2BCDEF08ull
#endif
/* exact FOP/FIP/FDP round trip is hard only on GenuineIntel: Windows 10 on AMD Zen 2/Zen 4 reads them back 0 after FXRSTOR->FXSAVE */
static int ptrs_hard(void) { return is_intel(); }
static int ptrs_strict(void) { return is_intel() || ((cpu()->e8.ebx >> 2) & 1); }
static void put_ptrs(uint8_t *a) {
    p16(a + 6, PTR_FOP);
#ifdef T_X64
    p64(a + 8, PTR_FIP); p64(a + 16, PTR_FDP);
#else
    p32(a + 8, (uint32_t)PTR_FIP); p16(a + 12, 0x0023); p16(a + 14, 0);
    p32(a + 16, (uint32_t)PTR_FDP); p16(a + 20, 0x002B); p16(a + 22, 0);
#endif
}
typedef struct { uint32_t fop; uint64_t fip, fdp; } x87_ptrs;
static x87_ptrs get_ptrs(const uint8_t *a) {
    x87_ptrs r; r.fop = g16(a + 6);
#ifdef T_X64
    r.fip = g64(a + 8); r.fdp = g64(a + 16);
#else
    r.fip = g32(a + 8); r.fdp = g32(a + 16);
#endif
    return r;
}
static int ptrs_eq(x87_ptrs a, x87_ptrs b) { return (a.fop & 0x7FF) == (b.fop & 0x7FF) && a.fip == b.fip && a.fdp == b.fdp; }
static int ptrs_unwritten(x87_ptrs a) {
    return (a.fop == 0 && a.fip == 0 && a.fdp == 0) ||
           (a.fop == 0x5A5A && a.fip == (SENT64 & (sizeof(void *) == 8 ? ~0ull : 0xFFFFFFFFull)) &&
            a.fdp == (SENT64 & (sizeof(void *) == 8 ? ~0ull : 0xFFFFFFFFull)));
}
static int ptrs_exact(x87_ptrs got) {
    x87_ptrs want = { PTR_FOP, PTR_FIP, PTR_FDP };
    return ptrs_eq(got, want);
}
static void check_ptrs(x87_ptrs got, const char *lbl) {
    x87_ptrs want = { PTR_FOP, PTR_FIP, PTR_FDP };
    if (!ptrs_hard()) {
        t_note("%s: FOP/FIP/FDP observed 0x%x/0x%llx/0x%llx (loaded 0x%x/0x%llx/0x%llx; not required exact on non-Intel)", lbl, got.fop,
               (unsigned long long)got.fip, (unsigned long long)got.fdp, PTR_FOP, (unsigned long long)PTR_FIP, (unsigned long long)PTR_FDP);
        return;
    }
    if (ptrs_strict())
        CHECKF(ptrs_eq(got, want), "%s: FOP/FIP/FDP 0x%03x/0x%llx/0x%llx, loaded 0x%03x/0x%llx/0x%llx", lbl, got.fop,
               (unsigned long long)got.fip, (unsigned long long)got.fdp, PTR_FOP, (unsigned long long)PTR_FIP, (unsigned long long)PTR_FDP);
    else
        CHECKF(ptrs_eq(got, want) || ptrs_unwritten(got), "%s: FOP/FIP/FDP 0x%x/0x%llx/0x%llx are neither the loaded values nor 0/unwritten",
               lbl, got.fop, (unsigned long long)got.fip, (unsigned long long)got.fdp);
    CHECKF((got.fop & ~0x7FFu) == 0 || ptrs_unwritten(got), "%s: FOP 0x%04x has bits above 10 (FOP is 11 bits)", lbl, got.fop);
}
TEST(x87_pointers_fxrstor_fxsave) {
    need_fxsr();
    xs_state p; st_pattern(&p, 0x1B0);
    put_legacy(g_a.b, &p); put_ptrs(g_a.b);
    xs_req q;
    /* an interrupt/context switch between the load and the save can legitimately rewrite the pointers on some
     * parts (the pointer fields are not guaranteed across them): retry up to 20 times, one exact round trip suffices */
    int tries = 0;
    do {
        memset(g_b.b, SENT, 512);
        q_init(&q);
        q_rst(&q, Q_R1, g_a.b, 0, XR_FXRSTOR);
        q_sav(&q, Q_S1, g_b.b, 0, XO_FXSAVE);
        xs_run(&q, T_VK_XMM);
        tries++;
    } while (!ptrs_exact(get_ptrs(g_b.b)) && tries < 20);
    t_info("ptr_attempts", "%d", tries);
    t_info("cpuid_8000_0008_ebx2_xsaveerptr", "%u", (cpu()->e8.ebx >> 2) & 1);
    check_ptrs(get_ptrs(g_b.b), "FXRSTOR -> FXSAVE");
    xs_cmp(g_b.b, 0, &p, 3 | XW_MX, 0, "FXRSTOR -> FXSAVE (with pointer fields)");
    t_info("strict", "%d (intel %d, XSaveErPtr %u)", ptrs_strict(), is_intel(), (cpu()->e8.ebx >> 2) & 1);
}
TEST(x87_pointers_xrstor_xsave) {
    need_xsave();
    xs_state p; st_pattern(&p, 0x1B1);
    xs_make_std(g_a.b, &p, 1); put_ptrs(g_a.b);
    xs_req q;
    int tries = 0;      /* retry up to 20 times (see x87_pointers_fxrstor_fxsave): one exact round trip suffices */
    do {
        memset(g_b.b, SENT, std_size()); memset(g_c.b, SENT, 512);
        q_init(&q);
        q_rst(&q, Q_R1, g_a.b, 1, XR_XRSTOR);
        q_sav(&q, Q_S1, g_b.b, 1, XO_XSAVE);
        q_sav(&q, Q_S2, g_c.b, 0, XO_FXSAVE);
        xs_run(&q, T_VK_XMM);
        tries++;
    } while (!(ptrs_exact(get_ptrs(g_b.b)) && ptrs_exact(get_ptrs(g_c.b))) && tries < 20);
    t_info("ptr_attempts", "%d", tries);
    t_info("cpuid_8000_0008_ebx2_xsaveerptr", "%u", (cpu()->e8.ebx >> 2) & 1);
    check_ptrs(get_ptrs(g_b.b), "XRSTOR(x87) -> XSAVE(x87)");
    check_ptrs(get_ptrs(g_c.b), "XRSTOR(x87) -> FXSAVE");
    xs_cmp(g_b.b, 0, &p, 1, 0, "XRSTOR(x87) -> XSAVE(x87)");
    CHECK_EQ(g64(g_b.b + 512) & 1, 1);
#ifdef T_X64
    /* 32-bit form in 64-bit mode stores FIP[31:0] / FDP[31:0] (+ FCS/FDS) */
    memset(g_c.b, SENT, 512);
    q_init(&q);
    q_rst(&q, Q_R1, g_a.b, 1, XR_XRSTOR);
    q_sav(&q, Q_S1, g_c.b, 1, XO_XSAVE32);
    xs_run(&q, T_VK_XMM);
    if (ptrs_hard()) {
        CHECK_EQ(g32(g_c.b + 8), (uint32_t)PTR_FIP);
        CHECK_EQ(g32(g_c.b + 16), (uint32_t)PTR_FDP);
    }
    t_info("xsave32_form_fcs_fds", "fcs=0x%04x fds=0x%04x", g16(g_c.b + 12), g16(g_c.b + 20));
#endif
}
/* x87 init (XRSTOR with XSTATE_BV[0]=0, or FNINIT) sets FOP/FIP/FDP (and FCS/FDS) to 0 (SDM 13.6 x87
 * initial configuration; FINIT/FNINIT operation). */
TEST(x87_pointers_cleared_by_init) {
    need_xsave();
    xs_state p; st_pattern(&p, 0x1B2);
    put_legacy(g_c.b, &p); put_ptrs(g_c.b);
    for (int v = 0; v < 2; v++) {
        memset(g_b.b, SENT, 512);
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_c.b, 0, XR_FXRSTOR);
        if (v == 0) q_rst(&q, Q_R2, g_z.b, 1, XR_XRSTOR);      /* XSTATE_BV[0]=0 -> x87 init */
        else q_x87(&q, 0, 0x037F);                           /* FNINIT (+FLDCW, a control instruction) */
        q_sav(&q, Q_S1, g_b.b, 0, XO_FXSAVE);
        xs_run(&q, T_VK_XMM);
        const char *lbl = v ? "FNINIT" : "XRSTOR x87 init";
        x87_ptrs got = get_ptrs(g_b.b);
        if (ptrs_strict()) CHECKF(got.fop == 0 && got.fip == 0 && got.fdp == 0, "%s: FOP/FIP/FDP 0x%x/0x%llx/0x%llx want 0", lbl,
                                  got.fop, (unsigned long long)got.fip, (unsigned long long)got.fdp);
        else CHECKF(ptrs_unwritten(got), "%s: FOP/FIP/FDP 0x%x/0x%llx/0x%llx neither 0 nor unwritten", lbl, got.fop,
                    (unsigned long long)got.fip, (unsigned long long)got.fdp);
#ifndef T_X64
        CHECKF(g16(g_b.b + 12) == 0 || g16(g_b.b + 12) == 0x5A5A, "%s: FCS 0x%04x", lbl, g16(g_b.b + 12));
        CHECKF(g16(g_b.b + 20) == 0 || g16(g_b.b + 20) == 0x5A5A, "%s: FDS 0x%04x", lbl, g16(g_b.b + 20));
#endif
        CHECK_EQ(g16(g_b.b), 0x037F);
        CHECK_EQ(g16(g_b.b + 2), 0);
        CHECK_EQ(g_b.b[4], 0);
    }
}

/* ======================= XSAVE (standard format) ============================== */
TEST(xsave_legacy_matches_fxsave) {
    need_xsave();
    make_vals(0x80);
    xs_state p; st_pattern(&p, 0x110);
    static t_vregs vin; st_to_vregs(&p, &vin);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, std_size()); memset(g_b.b, SENT, 512);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 3, 0x0C7F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, T_XS_X87 | T_XS_SSE, XO_XSAVE);
    q_sav(&q, Q_S2, g_b.b, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    xs_state e; st_loaded(&e, &p, T_VK_XMM, 3, g_vals, 0x0C7F, 1);
    xs_cmp(g_a.b, 0, &e, 3 | XW_MX, 0, "XSAVE(3)");
    CHECK_MEM(g_a.b, g_b.b, 6);
    CHECK_MEM(g_a.b + 24, g_b.b + 24, 8);              /* MXCSR + MXCSR_MASK */
    for (int i = 0; i < 8; i++) CHECK_MEM(g_a.b + 32 + 16 * i, g_b.b + 32 + 16 * i, 10);
    CHECK_MEM(g_a.b + 160, g_b.b + 160, 16 * T_NXMM);
    CHECK_EQ(g64(g_a.b + 512) & 3, 3);
    t_info("pointer_fields_equal", "%d", !memcmp(g_a.b + 6, g_b.b + 6, 18));
}
TEST(xsave_xmm_and_header) {
    need_xsave();
    xs_state p; st_pattern(&p, 0x111);
    static t_vregs vin; st_to_vregs(&p, &vin);
    xs_req q; q_init(&q);
    memset(g_a.b, 0, std_size());
    memset(g_a.b + 520, SENT, 56);                       /* XCOMP_BV + reserved header bytes: must stay as they are */
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVE);
    xs_run(&q, T_VK_XMM);
    xs_state e; st_loaded(&e, &p, T_VK_XMM, 0, NULL, 0, 1);
    xs_cmp(g_a.b, 0, &e, 2 | XW_MX, 0, "XSAVE(xcr0) xmm");
    CHECK_EQ(g32(g_a.b + 28), mx_mask_raw());          /* same MXCSR_MASK as FXSAVE */
    uint64_t bv = g64(g_a.b + 512);
    CHECKF(bv & 2, "XSTATE_BV[1]=0 although XMM registers hold non-zero data (0x%llx)", (unsigned long long)bv);
    CHECKF((bv & ~g_m) == 0, "XSTATE_BV 0x%llx has bits outside RFBM 0x%llx (header was zero)", (unsigned long long)bv, (unsigned long long)g_m);
    CHECK_UNTOUCHED(g_a.b + 520, 56, "XCOMP_BV and header bytes 16..63 (XSAVE writes only XSTATE_BV)");
}
TEST(xsave_ymm_upper_halves) {
    need_avx();
    xs_state p; st_pattern(&p, 0x112);
    static t_vregs vin; st_to_vregs(&p, &vin);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, std_size());
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q.s[Q_VIN] = (uintptr_t)&vin;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVE);
    xs_run(&q, T_VK_YMM);
    xs_state e; st_loaded(&e, &p, T_VK_YMM, 0, NULL, 0, 0);
    xs_cmp(g_a.b, 0, &e, 6 | (unsigned)(g_m & 0xE0), 0, "XSAVE after VEX ymm loads");   /* VEX.256 zeroes bits 511:256 */
    uint64_t bv = g64(g_a.b + 512);
    CHECKF(bv & 4, "XSTATE_BV[2]=0 although upper YMM halves are non-zero (0x%llx)", (unsigned long long)bv);
    CHECK_EQ(cpu()->comp[2].size, 256);
}
TEST(xsave_x87_from_fld) {
    need_xsave();
    make_vals(0x81);
    xs_req q; q_init(&q);
    memset(g_a.b, SENT, std_size());
    q_x87(&q, 8, 0x0C7F);
    q_sav(&q, Q_S1, g_a.b, T_XS_X87, XO_XSAVE);
    xs_run(&q, T_VK_XMM);
    xs_state e; st_loaded(&e, NULL, 0, 8, g_vals, 0x0C7F, 0);
    xs_cmp(g_a.b, 0, &e, 1, 0, "XSAVE(x87) after 8 FLD");
    CHECK_EQ(g64(g_a.b + 512) & 1, 1);
    CHECK_UNTOUCHED(g_a.b + 24, 8, "MXCSR/MXCSR_MASK with RFBM=x87");
    CHECK_UNTOUCHED(g_a.b + 160, 256, "XMM area with RFBM=x87");
}
/* XSAVE with a partial RFBM: only the selected regions, MXCSR iff RFBM[1]|RFBM[2], header bits outside RFBM kept. */
static void xsave_rfbm_case(uint64_t rfbm, int op) {
    make_vals(0x82);
    xs_state p; st_pattern(&p, 0x113);
    static t_vregs vin; st_to_vregs(&p, &vin);
    int kind = g_kind == T_VK_ZMM ? T_VK_YMM : g_kind;
    size_t sz = std_size();
    memset(g_a.b, SENT, sz + 64);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 2, 0x037F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, rfbm, op);
    xs_run(&q, kind);
    xs_state e; st_loaded(&e, &p, kind, 2, g_vals, 0x037F, 1);
    char lbl[64]; snprintf(lbl, sizeof lbl, "%s RFBM=0x%llx", op == XO_XSAVEOPT ? "XSAVEOPT" : "XSAVE", (unsigned long long)rfbm);
    unsigned what = (unsigned)(rfbm & 0x7);
    xs_cmp(g_a.b, 0, &e, what, op == XO_XSAVEOPT ? CF_BV : 0, lbl);
    uint64_t bv = g64(g_a.b + 512);
    uint64_t inuse = 1 | 2 | (kind >= T_VK_YMM ? 4 : 0);
    CHECKF((bv & ~rfbm) == (SENT64 & ~rfbm), "%s: XSTATE_BV bits outside RFBM modified: 0x%016llx", lbl, (unsigned long long)bv);
    CHECKF((bv & rfbm & inuse) == (rfbm & inuse), "%s: XSTATE_BV 0x%llx misses in-use components 0x%llx", lbl,
           (unsigned long long)bv, (unsigned long long)(rfbm & inuse));
    CHECK_UNTOUCHED(g_a.b + 520, 56, "header bytes 8..63");
    CHECK_UNTOUCHED(g_a.b + 464, 48, "legacy bytes 464..511");
    if (rfbm & 6) {
        if ((rfbm & 6) == 4 && !is_intel()) {
            t_info("mxcsr_written_rfbm_avx_only", "%d", g32(g_a.b + 24) == p.mxcsr);
        } else {
            CHECKF(g32(g_a.b + 24) == p.mxcsr, "%s: MXCSR 0x%08x want 0x%08x (written iff RFBM[1]|RFBM[2])", lbl, g32(g_a.b + 24), p.mxcsr);
            CHECKF(g32(g_a.b + 28) == mx_mask_raw(), "%s: MXCSR_MASK 0x%08x", lbl, g32(g_a.b + 28));
        }
    } else CHECK_UNTOUCHED(g_a.b + 24, 8, "MXCSR/MXCSR_MASK (RFBM has neither SSE nor AVX)");
    if (!(rfbm & 1)) { CHECK_UNTOUCHED(g_a.b, 24, "x87 header fields"); CHECK_UNTOUCHED(g_a.b + 32, 128, "ST0-7"); }
    if (!(rfbm & 2)) CHECK_UNTOUCHED(g_a.b + 160, 256, "XMM area");
    if ((g_m & 4) && !(rfbm & 4)) CHECK_UNTOUCHED(g_a.b + cpu()->comp[2].offset, 256, "YMM_Hi128 area");
    CHECK_UNTOUCHED(g_a.b + sz, 64, "bytes past the standard size");
}
TEST(xsave_rfbm_x87_only)  { need_xsave(); xsave_rfbm_case(1, XO_XSAVE); }
TEST(xsave_rfbm_sse_only)  { need_xsave(); xsave_rfbm_case(2, XO_XSAVE); }
TEST(xsave_rfbm_avx_only)  { need_avx(); xsave_rfbm_case(4, XO_XSAVE); }
TEST(xsave_rfbm_x87_avx)   { need_avx(); xsave_rfbm_case(5, XO_XSAVE); }
TEST(xsave_rfbm_all_user)  { need_xsave(); xsave_rfbm_case(g_m & 7, XO_XSAVE); }
TEST(xsave_rfbm_zero_writes_nothing) {
    need_xsave();
    size_t sz = std_size();
    memset(g_a.b, SENT, sz + 64);
    xs_req q; q_init(&q);
    q_sav(&q, Q_S1, g_a.b, 0, XO_XSAVE);
    xs_run(&q, T_VK_XMM);
    CHECK_UNTOUCHED(g_a.b, sz + 64, "XSAVE with EDX:EAX=0");
}
TEST(xsave_rfbm_beyond_xcr0) {
    need_xsave();
    const t_cpu_info *c = cpu();
    size_t sz = c->xsave_size_enabled;
    if (sz + 256 > XS_MAX) SKIP("enabled XSAVE size %zu exceeds the test buffer", sz);
    xs_state p; st_pattern(&p, 0x114);
    static t_vregs vin; st_to_vregs(&p, &vin);
    memset(g_a.b, SENT, sz + 256);
    xs_req q; q_init(&q);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, ~0ull, XO_XSAVE);            /* RFBM = XCR0 & EDX:EAX: legal */
    xs_run(&q, T_VK_XMM);
    xs_state e; st_loaded(&e, &p, T_VK_XMM, 0, NULL, 0, 1);
    xs_cmp(g_a.b, 0, &e, 2 | XW_MX, 0, "XSAVE(EDX:EAX=~0)");
    uint64_t bv = g64(g_a.b + 512);
    CHECKF((bv & ~c->xcr0) == (SENT64 & ~c->xcr0), "XSTATE_BV bits outside XCR0 changed: 0x%016llx", (unsigned long long)bv);
    CHECK(bv & 2);
    CHECK_UNTOUCHED(g_a.b + sz, 256, "bytes past CPUID.(0Dh,0).EBX");
    CHECK_UNTOUCHED(g_a.b + 520, 56, "header bytes 8..63");
}
TEST(xsave_exact_size_guard_page) {
    need_xsave();
    const t_cpu_info *c = cpu();
    struct { const char *what; uint32_t size; uint64_t m; int op; } v[4];
    int n = 0;
    v[n].what = "XSAVE(xcr0)"; v[n].size = c->xsave_size_enabled; v[n].m = ~0ull; v[n].op = XO_XSAVE; n++;
    v[n].what = "XSAVE(user)"; v[n].size = xsave_std_size(g_m); v[n].m = g_m; v[n].op = XO_XSAVE; n++;
    if (cpu_has(F_XSAVEC)) { v[n].what = "XSAVEC(user)"; v[n].size = xsave_compacted_size(g_m); v[n].m = g_m; v[n].op = XO_XSAVEC; n++; }
    if (cpu_has(F_XSAVEOPT)) { v[n].what = "XSAVEOPT(user)"; v[n].size = xsave_std_size(g_m); v[n].m = g_m; v[n].op = XO_XSAVEOPT; n++; }
    for (int i = 0; i < n; i++) {
        uint32_t sz = (v[i].size + 63) & ~63u;
        if (sz != v[i].size) t_note("%s size %u is not a multiple of 64; area rounded to %u", v[i].what, v[i].size, sz);
        void *base = NULL;
        uint8_t *a = (uint8_t *)t_guarded_alloc(sz, &base);
        REQUIRE(a != NULL);
        CHECK(((uintptr_t)a & 63) == 0);
        xs_req q; q_init(&q);
        q_sav(&q, Q_S1, a, v[i].m, v[i].op);
        xs_run(&q, T_VK_XMM);                            /* a write past the size would crash the case */
        CHECKF(g32(a + 24) == t_get_mxcsr(), "%s: MXCSR in area 0x%x", v[i].what, g32(a + 24));
        xs_req r; q_init(&r);
        q_rst(&r, Q_R1, a, v[i].m == ~0ull ? g_m : v[i].m, XR_XRSTOR);   /* reading it back must not fault either */
        xs_run(&r, T_VK_XMM);
        t_guarded_free(base);
    }
    void *base = NULL;
    uint8_t *a = (uint8_t *)t_guarded_alloc(512, &base);
    REQUIRE(a != NULL);
    xs_req q; q_init(&q);
    q_sav(&q, Q_S1, a, 0, XO_FXSAVE);
    q_rst(&q, Q_R2, a, 0, XR_FXRSTOR);
    xs_run(&q, T_VK_XMM);
    t_guarded_free(base);
}
/* OS contract: an XSAVE-family access to committed-but-never-touched (demand-zero) pages, with the area
 * straddling a page boundary, takes page faults that the OS resolves transparently and restarts the
 * instruction. Each operation gets brand-new pages. */
static uint8_t *fresh_area(size_t sz, size_t before, void **base) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    size_t pg = si.dwPageSize ? si.dwPageSize : 4096;
    size_t tot = (before + sz + 2 * pg) / pg * pg;
    uint8_t *m = (uint8_t *)VirtualAlloc(NULL, tot, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    *base = m;
    return m ? m + pg - before : NULL;                    /* `before` bytes in the first page */
}
/* Emulator artifacts (README "Emulated validation"): Intel SDE emulates the XSAVE family when its CPU model
 * has state the host lacks, and that emulation deviates from the SDM in the places marked XR_SDE_XFAIL
 * (32-bit XCOMP_BV[7], XINUSE tracking, XCOMP_BV for RFBM=0 / init components / PKRU, partial-RFBM XRSTOR,
 * and no #GP/#PF checks). Under tools/sde_run.sh only; the checks themselves are unchanged. */
#define XR_SDE_XFAIL(why) do { if (t_sde_model()) t_expect_fail("Intel SDE (%s) XSAVE emulation: %s", t_sde_model(), why); } while (0)
#ifdef T_X64
#define XR_SDE_XFAIL32(why) ((void)0)
#else
#define XR_SDE_XFAIL32(why) XR_SDE_XFAIL(why)
#endif

TEST(xsave_fresh_pages_straddle) {
    need_xsave();
    XR_SDE_XFAIL32("clears XCOMP_BV[7] (Hi16_ZMM) outside 64-bit mode");
    make_vals(0x8A);
    xs_state p; st_pattern(&p, 0x118);
    static t_vregs vin; st_to_vregs(&p, &vin);
    size_t sz = std_size();
    xs_state e; st_loaded(&e, &p, g_kind, 3, g_vals, 0x037F, 1);
    /* 1. XSAVE: legacy region in page 0, header + extended components in page 1 */
    void *b1 = NULL; uint8_t *A = fresh_area(sz, 512, &b1);
    REQUIRE(A != NULL);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 3, 0x037F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, A, g_m, XO_XSAVE);
    xs_run(&q, g_kind);
    xs_cmp(A, 0, &e, comps_of(g_m), 0, "XSAVE to fresh straddling pages");
    CHECK_EQ(g64(A + 512) & ~g_m, 0);
    CHECK(all_eq(A + 520, 56, 0));                        /* header bytes 8..63 never written: still demand-zero */
    CHECK(all_eq((uint8_t *)b1, (size_t)(A - (uint8_t *)b1), 0));
    /* 2. XRSTOR from it into a fresh XSAVE target (page boundary inside the XMM area) */
    void *b2 = NULL; uint8_t *B = fresh_area(sz, 256, &b2);
    REQUIRE(B != NULL);
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    q_init(&q);
    q_rst(&q, Q_R1, A, g_m, XR_XRSTOR);
    q_sav(&q, Q_S1, B, g_m, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout;
    xs_run(&q, g_kind);
    xs_cmp(B, 0, &e, comps_of(g_m), 0, "XRSTOR straddling -> XSAVE fresh (boundary in XMM area)");
    xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, "registers after straddling XRSTOR");
    /* 3. XRSTOR from never-touched zero pages: XSTATE_BV=0 -> every RFBM component init; MXCSR field 0 is
     * loaded (standard form) or MXCSR initialised */
    void *b3 = NULL; uint8_t *Z = fresh_area(sz, 64, &b3);
    REQUIRE(Z != NULL);
    uint32_t mx = 0xFFFFFFFFu;
    memset(B, SENT, sz);
    q_init(&q);
    q_rst(&q, Q_R1, A, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, Z, g_m, XR_XRSTOR);
    q_sav(&q, Q_S1, B, g_m, XO_XSAVE);
    q.s[Q_STMX] = (uintptr_t)&mx;
    xs_run(&q, g_kind);
    xs_state z; st_init(&z);
    xs_cmp(B, 0, &z, comps_of(g_m) & ~XW_MX, 0, "XRSTOR from untouched zero pages");
    CHECKF(mx == 0 || mx == 0x1F80, "MXCSR 0x%08x after XRSTOR of an all-zero area (want 0 from memory or 0x1F80)", mx);
    t_info("zero_area_mxcsr", "0x%08x", mx);
    /* 4. FXSAVE straddling fresh pages, 5. XSAVEC straddling fresh pages + compacted XRSTOR back */
    void *b4 = NULL; uint8_t *F = fresh_area(512, 256, &b4);
    REQUIRE(F != NULL);
    q_init(&q);
    q_rst(&q, Q_R1, A, g_m, XR_XRSTOR);
    q_sav(&q, Q_S1, F, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    xs_cmp(F, 0, &e, 3 | XW_MX, 0, "FXSAVE to fresh straddling pages");
    void *b5 = NULL;
    if (cpu_has(F_XSAVEC)) {
        uint8_t *C = fresh_area(xsave_compacted_size(g_m), 512, &b5);
        REQUIRE(C != NULL);
        memset(B, SENT, sz);
        q_init(&q);
        q_rst(&q, Q_R1, A, g_m, XR_XRSTOR);
        q_sav(&q, Q_S1, C, g_m, XO_XSAVEC);
        xs_run(&q, g_kind);
        CHECK_EQ(g64(C + 520), XS_COMPACT | g_m);
        xs_cmp(C, XS_COMPACT | g_m, &e, (unsigned)(g_m & 0xE7), CF_BV, "XSAVEC to fresh straddling pages");
        q_init(&q);
        q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
        q_rst(&q, Q_R2, C, g_m, XR_XRSTOR);
        q_sav(&q, Q_S1, B, g_m, XO_XSAVE);
        xs_run(&q, g_kind);
        xs_cmp(B, 0, &e, comps_of(g_m), 0, "compacted XRSTOR from straddling pages");
    }
    VirtualFree(b1, 0, MEM_RELEASE); VirtualFree(b2, 0, MEM_RELEASE); VirtualFree(b3, 0, MEM_RELEASE);
    VirtualFree(b4, 0, MEM_RELEASE); if (b5) VirtualFree(b5, 0, MEM_RELEASE);
}
TEST(xsave_software_bytes_untouched_all_forms) {
    need_xsave();
    int ops[] = { XO_XSAVE, XO_XSAVEOPT, XO_XSAVEC, XO_FXSAVE, XO_XSAVE32, XO_FXSAVE32 };
    const char *nm[] = { "XSAVE", "XSAVEOPT", "XSAVEC", "FXSAVE", "XSAVE(32-bit form)", "FXSAVE(32-bit form)" };
    for (size_t i = 0; i < T_COUNTOF(ops); i++) {
        if (ops[i] == XO_XSAVEOPT && !cpu_has(F_XSAVEOPT)) continue;
        if (ops[i] == XO_XSAVEC && !cpu_has(F_XSAVEC)) continue;
        memset(g_a.b, SENT, std_size());
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
        q_sav(&q, Q_S1, g_a.b, g_m, ops[i]);
        xs_run(&q, T_VK_XMM);
        char w[64]; snprintf(w, sizeof w, "%s bytes 464..511", nm[i]);
        CHECK_UNTOUCHED(g_a.b + 464, 48, w);
        t_info(nm[i], "bytes 416..463 written=%d", !all_eq(g_a.b + 416, 48, SENT));
    }
}
TEST(xsave32_vs_xsave64_forms) {
    need_xsave();
#ifndef T_X64
    SKIP("XSAVE64 (REX.W) exists only in 64-bit mode; x86 build uses the 32-bit form everywhere");
#else
    make_vals(0x83);
    xs_state p; st_pattern(&p, 0x115);
    static t_vregs vin; st_to_vregs(&p, &vin);
    int kind = g_kind;
    size_t sz = std_size();
    memset(g_a.b, SENT, sz); memset(g_b.b, SENT, sz);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 4, 0x037F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVE);
    q_sav(&q, Q_S2, g_b.b, g_m, XO_XSAVE32);
    xs_run(&q, kind);
    xs_state e; st_loaded(&e, &p, kind, 4, g_vals, 0x037F, 1);
    xs_cmp(g_a.b, 0, &e, comps_of(g_m), 0, "XSAVE64");
    xs_cmp(g_b.b, 0, &e, comps_of(g_m), 0, "XSAVE (32-bit form)");
    CHECK_MEM(g_b.b, g_a.b, 6);
    CHECK_MEM(g_b.b + 24, g_a.b + 24, 416 - 24);
    CHECK_EQ(g64(g_b.b + 512), g64(g_a.b + 512));
    CHECK_MEM(g_b.b + 576, g_a.b + 576, sz - 576);
    t_info("xsave64_fip_fdp", "fip=0x%llx fdp=0x%llx", (unsigned long long)g64(g_a.b + 8), (unsigned long long)g64(g_a.b + 16));
    t_info("xsave32_fip_fcs_fdp_fds", "fip=0x%08x fcs=0x%04x fdp=0x%08x fds=0x%04x", g32(g_b.b + 8), g16(g_b.b + 12), g32(g_b.b + 16), g16(g_b.b + 20));
#endif
}
TEST(xinuse_set_by_nonzero_state) {
    need_xgetbv1();
    make_vals(0x84);
    xs_state p; st_pattern(&p, 0x116);
    static t_vregs vin; st_to_vregs(&p, &vin);
    uint64_t xin = 0, xin_init = 0;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 1, 0x037F);
    q.s[Q_VIN] = (uintptr_t)&vin;
    q.s[Q_XIN1] = (uintptr_t)&xin;
    xs_run(&q, g_kind);
    uint64_t want = 3 | (g_kind >= T_VK_YMM ? 4 : 0);
#ifdef T_X64
    if (g_kind == T_VK_ZMM) want |= 0xE0;
#else
    if (g_kind == T_VK_ZMM) want |= 0x60;
#endif
    CHECKF((xin & want) == want, "XINUSE 0x%llx lacks 0x%llx although those components hold non-init values",
           (unsigned long long)xin, (unsigned long long)want);
    CHECKF((xin & ~cpu()->xcr0) == 0, "XINUSE 0x%llx outside XCR0", (unsigned long long)xin);
    t_info("xinuse_loaded", "0x%llx", (unsigned long long)xin);
    q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q.s[Q_XIN1] = (uintptr_t)&xin_init;
    xs_run(&q, T_VK_XMM);
    t_info("xinuse_after_xrstor_init", "0x%llx", (unsigned long long)xin_init);
    CHECKF((xin_init & ~cpu()->xcr0) == 0, "XINUSE 0x%llx outside XCR0", (unsigned long long)xin_init);
}
TEST(xsave_bv_equals_xinuse) {
    need_xgetbv1();
    make_vals(0x85);
    xs_state p; st_pattern(&p, 0x117);
    static t_vregs vin; st_to_vregs(&p, &vin);
    for (int scen = 0; scen < 2; scen++) {
        int ok = 0, tries = 0;
        uint64_t x1 = 0, x2 = 0, bv = 0;
        for (tries = 0; tries < 5 && !ok; tries++) {
            memset(g_a.b, 0, std_size());
            xs_req q; q_init(&q);
            q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
            if (scen == 0) { q_x87(&q, 2, 0x037F); q.s[Q_VIN] = (uintptr_t)&vin; }
            q.s[Q_XIN1] = (uintptr_t)&x1; q.s[Q_XIN2] = (uintptr_t)&x2;
            q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVE);
            xs_run(&q, scen == 0 ? g_kind : T_VK_XMM);
            bv = g64(g_a.b + 512);
            if (x1 != x2) continue;                    /* XINUSE changed (preemption): retry */
            ok = 1;
        }
        CHECKF(ok, "XINUSE unstable over 5 tries (x1 0x%llx x2 0x%llx)", (unsigned long long)x1, (unsigned long long)x2);
        CHECKF((bv & g_m) == (x1 & g_m), "%s: XSTATE_BV 0x%llx != XINUSE 0x%llx & RFBM 0x%llx", scen ? "after init" : "loaded",
               (unsigned long long)bv, (unsigned long long)x1, (unsigned long long)g_m);
        t_info(scen ? "bv_after_init" : "bv_loaded", "0x%llx", (unsigned long long)bv);
    }
}

/* ======================= XRSTOR ============================================== */
/* Craft P, XRSTOR(mask) it over an init state, read back through instructions and XSAVE. */
static unsigned do_roundtrip(uint64_t mask, uint32_t tag, int quiet, int restore_op, int kind) {
    xs_state p; st_pattern(&p, tag);
    xs_make_std(g_a.b, &p, mask);
    size_t sz = std_size();
    memset(g_b.b, SENT, sz);
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    memset(g_fx.b, SENT, 512);
    uint32_t mx = 0;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_a.b, mask, restore_op);
    q_sav(&q, Q_S1, g_b.b, mask, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_FXOUT] = (uintptr_t)g_fx.b; q.s[Q_ENV] = (uintptr_t)g_env; q.s[Q_STMX] = (uintptr_t)&mx;
    xs_run(&q, kind);
    xs_state e; st_init(&e); st_take(&e, &p, comps_of(mask));
    char lbl[96]; snprintf(lbl, sizeof lbl, "XRSTOR(0x%llx) tag 0x%x", (unsigned long long)mask, tag);
    unsigned bad = 0;
    bad |= xs_cmp(g_b.b, 0, &e, comps_of(mask), quiet ? CF_QUIET : 0, lbl);
    bad |= xs_cmp_vregs(&vout, &e, kind, (unsigned)mask & kind_what(kind), quiet, lbl);
    bad |= xs_cmp(g_fx.b, 0, &e, (unsigned)(mask & 3) | XW_MX, quiet ? CF_QUIET : 0, lbl);
    if (mx != e.mxcsr) bad |= XW_MX;
    if (!quiet) {
        CHECKF(mx == e.mxcsr, "%s: STMXCSR 0x%08x want 0x%08x", lbl, mx, e.mxcsr);
        if (mask & 1) check_env(g_env, &e, lbl);
        uint64_t bv = g64(g_b.b + 512);
        CHECKF((bv & mask & XW_ARCH) == (mask & XW_ARCH), "%s: XSTATE_BV after XSAVE 0x%llx: loaded non-init components must be in use",
               lbl, (unsigned long long)bv);
    }
    return bad;
}
TEST(xrstor_roundtrip_x87) { need_xsave(); do_roundtrip(T_XS_X87, 0x120, 0, XR_XRSTOR, T_VK_XMM); }
TEST(xrstor_roundtrip_sse) { need_xsave(); do_roundtrip(T_XS_SSE, 0x121, 0, XR_XRSTOR, T_VK_XMM); }
TEST(xrstor_roundtrip_x87_sse) { need_xsave(); do_roundtrip(3, 0x122, 0, XR_XRSTOR, T_VK_XMM); }
TEST(xrstor_roundtrip_ymm) { need_avx(); do_roundtrip(7, 0x123, 0, XR_XRSTOR, T_VK_YMM); }
TEST(xrstor_roundtrip_all_user) { need_xsave(); do_roundtrip(g_m, 0x124, 0, XR_XRSTOR, g_kind); }
TEST(xrstor_roundtrip_many_patterns) {
    need_xsave();
    int bad = 0;
    for (uint32_t t = 0; t < 64; t++) {
        unsigned d = do_roundtrip(g_m, 0x300 + t, 1, XR_XRSTOR, g_kind);
        if (d && bad++ == 0) do_roundtrip(g_m, 0x300 + t, 0, XR_XRSTOR, g_kind);
        CHECKF(d == 0, "pattern tag 0x%x: parts 0x%x differ", 0x300 + t, d);
    }
}
TEST(xrstor32_form_roundtrip) {
    need_xsave();
#ifndef T_X64
    SKIP("XRSTOR64 exists only in 64-bit mode; x86 build uses the 32-bit form everywhere");
#else
    do_roundtrip(g_m, 0x125, 0, XR_XRSTOR32, g_kind);
#endif
}
/* R1 = P1 (everything), R2 = P2 area with bv2 under rfbm2 -> expected state. */
static void xrstor_mix(uint64_t bv2, uint64_t rfbm2, const char *lbl, int mx_from_p2) {
    xs_state p1, p2; st_pattern(&p1, 0x130); st_pattern(&p2, 0x131);
    xs_make_std(g_a.b, &p1, g_m);
    xs_make_std(g_c.b, &p2, bv2);
    size_t sz = std_size();
    memset(g_b.b, SENT, sz);
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    uint32_t mx = 0;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_c.b, rfbm2, XR_XRSTOR);
    q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_FXOUT] = (uintptr_t)g_fx.b; q.s[Q_ENV] = (uintptr_t)g_env; q.s[Q_STMX] = (uintptr_t)&mx;
    xs_run(&q, g_kind);
    xs_state e = p1;
    unsigned load = (unsigned)(rfbm2 & bv2 & 0xE7), init = (unsigned)(rfbm2 & ~bv2 & 0xE7);
    st_take(&e, &p2, load);
    st_reset(&e, init);
    int mx_known = 1;
    if (mx_from_p2 > 0) e.mxcsr = p2.mxcsr;
    else if (mx_from_p2 < 0) mx_known = 0;
    xs_cmp(g_b.b, 0, &e, comps_of(g_m) & ~(mx_known ? 0 : XW_MX), 0, lbl);
    xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, lbl);
    check_env(g_env, &e, lbl);
    if (mx_known) CHECKF(mx == e.mxcsr, "%s: MXCSR 0x%08x want 0x%08x", lbl, mx, e.mxcsr);
    else {
        t_info("mxcsr_loaded", "%d", mx == p2.mxcsr);
        CHECKF(mx == p1.mxcsr || mx == p2.mxcsr, "%s: MXCSR 0x%08x is neither old 0x%08x nor area 0x%08x", lbl, mx, p1.mxcsr, p2.mxcsr);
    }
}
TEST(xrstor_init_x87_only)   { need_xsave(); xrstor_mix(g_m & ~1ull, g_m, "XSTATE_BV[0]=0 -> x87 init", 1); }
TEST(xrstor_init_sse_only)   { need_xsave(); xrstor_mix(g_m & ~2ull, g_m, "XSTATE_BV[1]=0 -> XMM init", (g_m & 4) ? 1 : -1); }
TEST(xrstor_init_avx_only)   { need_avx(); xrstor_mix(g_m & ~4ull, g_m, "XSTATE_BV[2]=0 -> upper YMM init", 1); }
TEST(xrstor_rfbm_sse_only)   { need_xsave(); xrstor_mix(g_m, 2, "RFBM=SSE leaves x87/AVX", 1); }
TEST(xrstor_rfbm_avx_only)   { need_avx(); xrstor_mix(g_m, 4, "RFBM=AVX leaves XMM/x87", is_intel() ? 1 : -1); }
TEST(xrstor_rfbm_x87_only)   { need_xsave(); XR_SDE_XFAIL("XRSTOR modifies components outside RFBM"); xrstor_mix(g_m, 1, "RFBM=x87 leaves XMM/AVX/MXCSR", 0); }
TEST(xrstor_rfbm_x87_avx)    { need_avx(); XR_SDE_XFAIL("XRSTOR modifies components outside RFBM"); xrstor_mix(g_m, 5, "RFBM=x87|AVX leaves XMM", is_intel() ? 1 : -1); }
TEST(xrstor_rfbm_zero_noop)  { need_xsave(); xrstor_mix(g_m, 0, "RFBM=0 changes nothing", 0); }
TEST(xrstor_bv_subset_mixed) { need_avx(); xrstor_mix(5, 7, "XSTATE_BV=x87|AVX, RFBM=7", 1); }
TEST(xrstor_init_all) {
    need_xsave();
    xs_state p; st_pattern(&p, 0x132);
    xs_make_std(g_a.b, &p, g_m);
    size_t sz = std_size();
    memset(g_b.b, SENT, sz);
    static t_vregs vout; memset(&vout, 0xEE, sizeof vout);
    uint32_t mx = 0;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_z.b, g_m, XR_XRSTOR);              /* XSTATE_BV = 0, MXCSR field 0x1F80 */
    q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_FXOUT] = (uintptr_t)g_fx.b; q.s[Q_ENV] = (uintptr_t)g_env; q.s[Q_STMX] = (uintptr_t)&mx;
    xs_run(&q, g_kind);
    xs_state e; st_init(&e);
    xs_cmp(g_b.b, 0, &e, comps_of(g_m), 0, "XSAVE after XRSTOR init");
    xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, "registers after XRSTOR init");
    xs_cmp(g_fx.b, 0, &e, 3 | XW_MX, 0, "FXSAVE after XRSTOR init");
    CHECK_EQ(g16(g_env), 0x037F);
    CHECK_EQ(g16(g_env + 4), 0);
    CHECK_EQ(g16(g_env + 8), 0xFFFF);
    CHECK_EQ(mx, 0x1F80);
    t_info("xstate_bv_after_init", "0x%llx", (unsigned long long)g64(g_b.b + 512));
}
TEST(xrstor_init_mxcsr_source) {
    need_xsave();
    xs_state p; st_pattern(&p, 0x133);
    p.mxcsr = 0x1F80 | 0x2000;
    xs_state z; st_init(&z); z.mxcsr = 0x1F80 | 0x4000 | 0x8000;     /* RC=10, FZ in the area */
    xs_make_std(g_a.b, &p, g_m);
    xs_make_std(g_c.b, &z, 0);
    uint32_t mx = 0;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_c.b, g_m, XR_XRSTOR);
    q.s[Q_STMX] = (uintptr_t)&mx;
    xs_run(&q, T_VK_XMM);
    /* Standard form with XSTATE_BV[2:1]=0: SDM loads MXCSR from memory when RFBM[1]|RFBM[2]; some
     * implementations initialise it. Both are recorded; anything else is a failure. */
    CHECKF(mx == z.mxcsr || mx == 0x1F80, "MXCSR 0x%08x is neither the area value 0x%08x nor 0x1F80", mx, z.mxcsr);
    t_info("std_init_mxcsr", "%s (0x%08x)", mx == z.mxcsr ? "loaded-from-area" : "initialised", mx);
}
TEST(xrstor_ignores_legacy_416_511_and_mask) {
    need_xsave();
    xs_state p; st_pattern(&p, 0x134);
    xs_make_std(g_a.b, &p, g_m);
    memset(g_a.b + 416, 0xA5, 96);
    p32(g_a.b + 28, 0xDEADBEEF);
    unsigned d = 0;
    memset(g_b.b, SENT, std_size());
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout;
    xs_run(&q, g_kind);
    d |= xs_cmp(g_b.b, 0, &p, comps_of(g_m), 0, "XRSTOR with junk 416..511 + MXCSR_MASK");
    d |= xs_cmp_vregs(&vout, &p, g_kind, (unsigned)g_m & kind_what(g_kind), 0, "registers");
    CHECK_EQ(g32(g_b.b + 28), mx_mask_raw());
    CHECK_EQ(d, 0);
}
TEST(xrstor_then_fxsave_consistent) {
    need_xsave();
    xs_state p; st_pattern(&p, 0x135);
    xs_make_std(g_a.b, &p, g_m);
    memset(g_b.b, SENT, 512);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, 3, XR_XRSTOR);
    q_sav(&q, Q_S1, g_b.b, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    xs_cmp(g_b.b, 0, &p, 3 | XW_MX, 0, "XRSTOR(3) -> FXSAVE");
}
TEST(fxrstor_then_xsave_consistent) {
    need_xsave();
    XR_SDE_XFAIL("FXRSTOR of non-init x87 state does not set XINUSE[0]");
    xs_state p; st_pattern(&p, 0x136);
    put_legacy(g_c.b, &p);
    memset(g_b.b, SENT, std_size());
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_c.b, 0, XR_FXRSTOR);
    q_sav(&q, Q_S1, g_b.b, 3, XO_XSAVE);
    xs_run(&q, T_VK_XMM);
    xs_cmp(g_b.b, 0, &p, 3 | XW_MX, 0, "FXRSTOR -> XSAVE(3)");
    CHECK_EQ(g64(g_b.b + 512) & 3, 3);
}
TEST(xrstor_ftw_top_patterns) {
    need_xsave();
    static const uint8_t ftws[] = { 0x00, 0x01, 0x80, 0xFF, 0xA5, 0x5A, 0x0F, 0xF0, 0x3C, 0x7E, 0x81, 0x18 };
    for (size_t i = 0; i < T_COUNTOF(ftws); i++) {
        xs_state p; st_pattern(&p, 0x140 + (uint32_t)i);
        p.ftw = ftws[i];
        p.fsw = (uint16_t)((p.fsw & ~0x3800u) | ((unsigned)(i * 3) & 7u) << 11);
        xs_make_std(g_a.b, &p, 1);
        memset(g_b.b, SENT, std_size());
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_a.b, 1, XR_XRSTOR);
        q_sav(&q, Q_S1, g_b.b, 1, XO_XSAVE);
        q.s[Q_ENV] = (uintptr_t)g_env;
        xs_run(&q, T_VK_XMM);
        char lbl[64]; snprintf(lbl, sizeof lbl, "FTW 0x%02x TOP %u", ftws[i], (p.fsw >> 11) & 7u);
        xs_cmp(g_b.b, 0, &p, 1, 0, lbl);
        check_env(g_env, &p, lbl);
        CHECK_EQ(g64(g_b.b + 512) & 1, 1);
    }
}
TEST(xrstor_mxcsr_values) {
    need_xsave();
    uint32_t vals[] = { 0x1F80, 0x9F80, 0x3F80, 0x5F80, 0x7F80, 0x1FBF, 0x1D80, 0x0000, 0xFFBF, 0x1FC0, 0xFFFF };
    for (size_t i = 0; i < T_COUNTOF(vals); i++) {
        uint32_t v = vals[i];
        if (v & ~mx_mask()) continue;                    /* DAZ only where MXCSR_MASK allows it */
        xs_state p; st_init(&p); p.mxcsr = v;
        p.xmm[0][0] = 1;
        xs_make_std(g_a.b, &p, 2);
        memset(g_b.b, SENT, std_size());
        uint32_t got = 0;
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_a.b, 2, XR_XRSTOR);
        q_sav(&q, Q_S1, g_b.b, 2, XO_XSAVE);
        q.s[Q_STMX] = (uintptr_t)&got;
        xs_run(&q, T_VK_XMM);
        CHECKF(got == v, "XRSTOR MXCSR 0x%08x -> STMXCSR 0x%08x", v, got);
        CHECKF(g32(g_b.b + 24) == v, "XRSTOR MXCSR 0x%08x -> XSAVE 0x%08x", v, g32(g_b.b + 24));
        put_legacy(g_c.b, &p);
        q_init(&q);
        q_rst(&q, Q_R1, g_c.b, 0, XR_FXRSTOR);
        q.s[Q_STMX] = (uintptr_t)&got;
        xs_run(&q, T_VK_XMM);
        CHECKF(got == v, "FXRSTOR MXCSR 0x%08x -> STMXCSR 0x%08x", v, got);
    }
}

/* ======================= XSAVEOPT ============================================ */
TEST(xsaveopt_fresh_area_matches_xsave) {
    need_xsaveopt();
    make_vals(0x86);
    xs_state p; st_pattern(&p, 0x150);
    static t_vregs vin; st_to_vregs(&p, &vin);
    size_t sz = std_size();
    memset(g_a.b, SENT, sz); memset(g_b.b, 0, sz);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);              /* also retargets the XRSTOR tracking away from g_a */
    q_x87(&q, 3, 0x037F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVEOPT);
    q_sav(&q, Q_S2, g_b.b, g_m, XO_XSAVE);
    xs_run(&q, g_kind);
    xs_state e; st_loaded(&e, &p, g_kind, 3, g_vals, 0x037F, 1);
    xs_cmp(g_a.b, 0, &e, comps_of(g_m), CF_BV, "XSAVEOPT fresh");
    xs_cmp(g_b.b, 0, &e, comps_of(g_m), 0, "XSAVE");
    uint64_t bva = g64(g_a.b + 512), bvb = g64(g_b.b + 512);
    CHECKF((bva & g_m) == (bvb & g_m), "XSTATE_BV XSAVEOPT 0x%llx vs XSAVE 0x%llx", (unsigned long long)bva, (unsigned long long)bvb);
    CHECKF((bva & ~g_m) == (SENT64 & ~g_m), "XSAVEOPT changed XSTATE_BV bits outside RFBM: 0x%016llx", (unsigned long long)bva);
    CHECK_UNTOUCHED(g_a.b + 520, 56, "XSAVEOPT header bytes 8..63");
    CHECK_EQ(g32(g_a.b + 28), mx_mask_raw());
}
TEST(xsaveopt_after_xrstor_new_values) {
    need_xsaveopt();
    xs_state p1, p2; st_pattern(&p1, 0x151); st_pattern(&p2, 0x152);
    static t_vregs vin; st_to_vregs(&p2, &vin);
    xs_make_std(g_a.b, &p1, g_m);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q.s[Q_VIN] = (uintptr_t)&vin;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVEOPT);            /* modified optimisation must notice the new values */
    xs_run(&q, g_kind);
    xs_state e = p1;
    st_take(&e, &p2, kind_what(g_kind));
    if (g_kind == T_VK_YMM) st_reset(&e, 0x40);           /* VEX.256 loads zero bits 511:256 */
    if (g_kind == T_VK_ZMM && !g_kbw) for (int i = 0; i < 8; i++) e.k[i] &= 0xFFFF;
    xs_cmp(g_a.b, 0, &e, comps_of(g_m), CF_BV, "XRSTOR A; new regs; XSAVEOPT A");
}
TEST(xsaveopt_after_xrstor_unchanged) {
    need_xsaveopt();
    xs_state p; st_pattern(&p, 0x153);
    xs_make_std(g_a.b, &p, g_m);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVEOPT);
    xs_run(&q, T_VK_XMM);
    xs_cmp(g_a.b, 0, &p, comps_of(g_m), CF_BV, "XRSTOR A; XSAVEOPT A");
    CHECK_EQ(g64(g_a.b + 512) & g_m & XW_ARCH, g_m & XW_ARCH);
}
TEST(xsaveopt_other_area_after_xrstor) {
    need_xsaveopt();
    xs_state p; st_pattern(&p, 0x154);
    xs_make_std(g_a.b, &p, g_m);
    memset(g_b.b, SENT, std_size());
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVEOPT);
    xs_run(&q, T_VK_XMM);
    xs_cmp(g_b.b, 0, &p, comps_of(g_m), 0, "XRSTOR A; XSAVEOPT B");
    CHECK_EQ(g64(g_b.b + 512) & g_m & XW_ARCH, g_m & XW_ARCH);
}
TEST(xsaveopt_init_component) {
    need_xsaveopt();
    if (!(g_m & 4)) SKIP("AVX state not enabled in XCR0");
    xs_state p; st_pattern(&p, 0x155);
    static t_vregs vin; st_to_vregs(&p, &vin);
    size_t sz = std_size();
    memset(g_a.b, SENT, sz);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q.s[Q_VIN] = (uintptr_t)&vin;                        /* legacy MOVDQU: upper halves stay init */
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVEOPT);
    xs_run(&q, T_VK_XMM);
    xs_state e; st_loaded(&e, &p, T_VK_XMM, 0, NULL, 0, 0);
    xs_cmp(g_a.b, 0, &e, comps_of(g_m) & ~XW_MX, CF_BV, "XSAVEOPT with init upper YMM");
    uint64_t bv = g64(g_a.b + 512);
    CHECK(bv & 2);
    t_info("init_optimisation_avx", "XSTATE_BV[2]=%d region_untouched=%d", (int)(bv >> 2 & 1),
           all_eq(g_a.b + cpu()->comp[2].offset, 256, SENT));
}
TEST(xsaveopt_rfbm_sse_only) { need_xsaveopt(); xsave_rfbm_case(2, XO_XSAVEOPT); }
TEST(xsaveopt_rfbm_x87_only) { need_xsaveopt(); xsave_rfbm_case(1, XO_XSAVEOPT); }

/* ======================= XSAVEC (compacted) =================================== */
static void xsavec_mask_case(uint64_t m, int kind) {
    make_vals(0x87);
    xs_state p; st_pattern(&p, 0x160 + (uint32_t)m);
    static t_vregs vin; st_to_vregs(&p, &vin);
    uint32_t csz = xsave_compacted_size(m);
    memset(g_a.b, SENT, csz + 128);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 3, 0x037F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, m, XO_XSAVEC);
    xs_run(&q, kind);
    xs_state e; st_loaded(&e, &p, kind, 3, g_vals, 0x037F, 1);
    char lbl[64]; snprintf(lbl, sizeof lbl, "XSAVEC RFBM=0x%llx", (unsigned long long)m);
    uint64_t xc = g64(g_a.b + 520), bv = g64(g_a.b + 512);
    CHECKF(xc == (XS_COMPACT | m), "%s: XCOMP_BV 0x%016llx want 0x%016llx", lbl, (unsigned long long)xc, (unsigned long long)(XS_COMPACT | m));
    CHECKF((bv & ~m) == 0, "%s: XSTATE_BV 0x%llx has bits outside RFBM", lbl, (unsigned long long)bv);
    uint64_t inuse = 3 | (kind >= T_VK_YMM ? 4 : 0) | (kind == T_VK_ZMM ? 0xE0 : 0);
#ifndef T_X64
    inuse &= ~(uint64_t)0x80;
#endif
    CHECKF((bv & m & inuse) == (m & inuse), "%s: XSTATE_BV 0x%llx misses non-init components 0x%llx", lbl,
           (unsigned long long)bv, (unsigned long long)(m & inuse));
    xs_cmp(g_a.b, XS_COMPACT | m, &e, (unsigned)(m & 0xE7), CF_BV, lbl);
    /* Compacted form: MXCSR travels with SSE state. Written when XMM state is written (RFBM[1] and
     * XSTATE_BV[1]); RFBM[2]-only behaviour differs from the standard form (observed: not written) -> INFO. */
    if ((m & 2) && (bv & 2)) CHECKF(g32(g_a.b + 24) == p.mxcsr, "%s: MXCSR 0x%08x want 0x%08x", lbl, g32(g_a.b + 24), p.mxcsr);
    else if (m & 6) {
        uint32_t got = g32(g_a.b + 24);
        CHECKF(got == p.mxcsr || got == SENT64 >> 32, "%s: MXCSR field 0x%08x is neither untouched nor MXCSR 0x%08x", lbl, got, p.mxcsr);
        t_info("xsavec_mxcsr_written_without_sse_data", "RFBM=0x%llx: %d", (unsigned long long)m, got == p.mxcsr);
    } else CHECK_UNTOUCHED(g_a.b + 24, 8, "MXCSR with RFBM[2:1]=0");
    if (!(m & 1)) { CHECK_UNTOUCHED(g_a.b, 24, "x87 fields (not in RFBM)"); CHECK_UNTOUCHED(g_a.b + 32, 128, "ST area (not in RFBM)"); }
    if (!(m & 2)) CHECK_UNTOUCHED(g_a.b + 160, 256, "XMM area (not in RFBM)");
    CHECK_UNTOUCHED(g_a.b + 464, 48, "legacy bytes 464..511");
    CHECK_UNTOUCHED(g_a.b + csz, 128, "bytes past the compacted size");
    t_info(lbl, "header 16..63 %s", all_eq(g_a.b + 528, 48, SENT) ? "untouched" : all_eq(g_a.b + 528, 48, 0) ? "zeroed" : "other");
}
TEST(xsavec_rfbm_x87_only) { need_xsavec(); xsavec_mask_case(1, T_VK_XMM); }
TEST(xsavec_rfbm_sse_only) { need_xsavec(); xsavec_mask_case(2, T_VK_XMM); }
TEST(xsavec_rfbm_x87_sse)  { need_xsavec(); xsavec_mask_case(3, T_VK_XMM); }
TEST(xsavec_rfbm_avx_only) { need_xsavec(); if (!cpu_has(F_AVX)) SKIP("AVX not usable"); xsavec_mask_case(4, T_VK_YMM); }
TEST(xsavec_rfbm_x87_avx)  { need_xsavec(); if (!cpu_has(F_AVX)) SKIP("AVX not usable"); xsavec_mask_case(5, T_VK_YMM); }
TEST(xsavec_rfbm_all_user) { need_xsavec(); XR_SDE_XFAIL32("clears XCOMP_BV[7] (Hi16_ZMM) outside 64-bit mode"); xsavec_mask_case(g_m, g_kind); }
TEST(xsavec_rfbm_beyond_xcr0) {
    need_xsavec();
    XR_SDE_XFAIL("XCOMP_BV omits PKRU (and Hi16_ZMM outside 64-bit mode)");
    const t_cpu_info *c = cpu();
    uint32_t csz = xsave_compacted_size(c->xcr0);
    if (csz + 256 > XS_MAX) SKIP("compacted size %u exceeds the test buffer", csz);
    memset(g_a.b, SENT, csz + 256);
    xs_req q; q_init(&q);
    q_sav(&q, Q_S1, g_a.b, ~0ull, XO_XSAVEC);
    xs_run(&q, T_VK_XMM);
    CHECK_EQ(g64(g_a.b + 520), XS_COMPACT | c->xcr0);
    CHECK_EQ(g64(g_a.b + 512) & ~c->xcr0, 0);
    CHECK_UNTOUCHED(g_a.b + csz, 256, "bytes past the compacted size for XCR0");
}
TEST(xsavec_matches_xsave) {
    need_xsavec();
    make_vals(0x88);
    xs_state p; st_pattern(&p, 0x170);
    static t_vregs vin; st_to_vregs(&p, &vin);
    memset(g_a.b, SENT, std_size()); memset(g_b.b, SENT, xsave_compacted_size(g_m));
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 5, 0x037F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVE);
    q_sav(&q, Q_S2, g_b.b, g_m, XO_XSAVEC);
    xs_run(&q, g_kind);
    uint64_t bvc = g64(g_b.b + 512);
    CHECK_EQ(bvc & g_m, g64(g_a.b + 512) & g_m);
    CHECK_MEM(g_b.b, g_a.b, 6);
    CHECK_MEM(g_b.b + 24, g_a.b + 24, 8);
    for (int i = 0; i < 8; i++) CHECK_MEM(g_b.b + 32 + 16 * i, g_a.b + 32 + 16 * i, 10);
    CHECK_MEM(g_b.b + 160, g_a.b + 160, 16 * T_NXMM);
    static const int cs[] = { 2, 5, 6, 7 };
    for (int k = 0; k < 4; k++) {
        int i = cs[k];
        if (!(g_m >> i & 1) || !(bvc >> i & 1) || !((XW_ARCH >> i) & 1)) continue;
        uint32_t n = cpu()->comp[i].size;
#ifndef T_X64
        if (i == 2 || i == 6) n /= 2;                     /* only regs 0-7 exist in 32-bit mode */
#endif
        char w[64]; snprintf(w, sizeof w, "component %d compacted vs standard", i);
        t_check(!memcmp(g_b.b + xsave_compacted_offset(g_m, i), g_a.b + cpu()->comp[i].offset, n), __FILE__, __LINE__, w, NULL);
    }
}
TEST(xsavec_xrstor_roundtrip) {
    need_xsavec();
    make_vals(0x89);
    xs_state p1, p2; st_pattern(&p1, 0x171); st_pattern(&p2, 0x172);
    static t_vregs vin; st_to_vregs(&p1, &vin);
    /* XSAVEC does not have to write header bytes 16..63 (observed: left untouched), yet XRSTOR of a
     * compacted image requires them to be zero: software zeroes the area first. */
    memset(g_a.b, 0, xsave_compacted_size(g_m));
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q_x87(&q, 4, 0x0C7F);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p1.mxcsr;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVEC);
    xs_run(&q, g_kind);
    xs_make_std(g_c.b, &p2, g_m);
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    memset(g_b.b, SENT, std_size());
    q_init(&q);
    q_rst(&q, Q_R1, g_c.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_a.b, g_m, XR_XRSTOR);               /* compacted form (XCOMP_BV[63]=1) */
    q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_ENV] = (uintptr_t)g_env;
    xs_run(&q, g_kind);
    xs_state e; st_loaded(&e, &p1, g_kind, 4, g_vals, 0x0C7F, 1);
    xs_cmp(g_b.b, 0, &e, comps_of(g_m), 0, "XSAVEC -> XRSTOR(compacted) -> XSAVE");
    xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, "registers after compacted XRSTOR");
    e.fsw_mask = 0x3AFF;
    check_env(g_env, &e, "compacted XRSTOR x87");
}
static void xsavec_crafted(uint64_t m) {
    xs_state p1, p2; st_pattern(&p1, 0x180); st_pattern(&p2, 0x181 + (uint32_t)m);
    xs_make_std(g_a.b, &p1, g_m);
    xs_make_cmp(g_c.b, &p2, m, m);
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    memset(g_b.b, SENT, std_size());
    uint32_t mx = 0;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_c.b, m, XR_XRSTOR);
    q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_STMX] = (uintptr_t)&mx;
    xs_run(&q, g_kind);
    xs_state e = p1;
    st_take(&e, &p2, (unsigned)(m & 0xE7));
    /* compacted XRSTOR: MXCSR is loaded with SSE state (RFBM[1] & XSTATE_BV[1]); RFBM[2] alone is
     * implementation-specific (observed on Intel: MXCSR unchanged) -> accept old or area value. */
    int mx_known = (m & 2) || !(m & 4);
    if (m & 2) e.mxcsr = p2.mxcsr;
    char lbl[64]; snprintf(lbl, sizeof lbl, "crafted compacted RFBM=XCOMP=0x%llx", (unsigned long long)m);
    xs_cmp(g_b.b, 0, &e, comps_of(g_m) & ~(mx_known ? 0 : XW_MX), 0, lbl);
    xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, lbl);
    if (mx_known) CHECKF(mx == e.mxcsr, "%s: MXCSR 0x%08x want 0x%08x", lbl, mx, e.mxcsr);
    else {
        CHECKF(mx == p1.mxcsr || mx == p2.mxcsr, "%s: MXCSR 0x%08x neither old 0x%08x nor area 0x%08x", lbl, mx, p1.mxcsr, p2.mxcsr);
        t_info("compacted_rfbm_avx_only_mxcsr", "RFBM=0x%llx: %s", (unsigned long long)m, mx == p2.mxcsr ? "loaded" : "unchanged");
    }
}
TEST(xsavec_crafted_compacted_xrstor) {
    need_xsavec();
    XR_SDE_XFAIL("XRSTOR modifies components outside RFBM");
    uint64_t ms[] = { 1, 2, 3, 4, 5, 6, 7, 0x21, 0x41, 0x81, 0x61, 0xE7, 0xA3 };
    int n = 0;
    for (size_t i = 0; i < T_COUNTOF(ms); i++) {
        if ((ms[i] & g_m) != ms[i]) continue;
        xsavec_crafted(ms[i]);
        n++;
    }
    t_info("masks_tested", "%d", n);
}
TEST(xsavec_compacted_init_all) {
    need_xsavec();
    xs_state p; st_pattern(&p, 0x190);
    xs_make_std(g_a.b, &p, g_m);
    for (int variant = 0; variant < 2; variant++) {
        xs_state z; st_init(&z);
        if (variant) z.mxcsr = 0x1F80 | 0x6000;
        xs_make_cmp(g_c.b, &z, g_m, 0);
        static t_vregs vout; memset(&vout, 0xEE, sizeof vout);
        uint32_t mx = 0;
        memset(g_b.b, SENT, std_size());
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
        q_rst(&q, Q_R2, g_c.b, g_m, XR_XRSTOR);
        q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
        q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_STMX] = (uintptr_t)&mx; q.s[Q_ENV] = (uintptr_t)g_env;
        xs_run(&q, g_kind);
        xs_state e; st_init(&e);
        xs_cmp(g_b.b, 0, &e, comps_of(g_m) & ~XW_MX, 0, "compacted XSTATE_BV=0");
        xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, "registers after compacted init");
        CHECK_EQ(g16(g_env + 8), 0xFFFF);
        if (!variant) CHECK_EQ(mx, 0x1F80);
        else {
            CHECKF(mx == z.mxcsr || mx == 0x1F80, "MXCSR 0x%08x neither area 0x%08x nor 0x1F80", mx, z.mxcsr);
            t_info("compacted_init_mxcsr", "%s (0x%08x)", mx == z.mxcsr ? "loaded-from-area" : "initialised", mx);
        }
    }
}
TEST(xsavec_compacted_mxcsr_load) {
    need_xsavec();
    xs_state p1, p2; st_pattern(&p1, 0x191); st_pattern(&p2, 0x192);
    p2.mxcsr = 0x3F80;
    xs_make_std(g_a.b, &p1, g_m);
    xs_make_cmp(g_c.b, &p2, g_m, 2);                      /* only SSE in use */
    uint32_t mx = 0;
    static t_vregs vout; memset(&vout, 0, sizeof vout);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_c.b, g_m, XR_XRSTOR);
    q.s[Q_STMX] = (uintptr_t)&mx; q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_ENV] = (uintptr_t)g_env;
    xs_run(&q, g_kind);
    CHECK_EQ(mx, 0x3F80);
    xs_state e; st_init(&e); st_take(&e, &p2, 2 | XW_MX);
    xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, "compacted XSTATE_BV=SSE");
    CHECK_EQ(g16(g_env + 8), 0xFFFF);                     /* x87 initialised */
    CHECK_EQ(g16(g_env), 0x037F);
}
/* XSAVEC pseudocode writes XSTATE_BV (= XINUSE & RFBM = 0) and XCOMP_BV (= 1<<63 | RFBM) unconditionally;
 * with RFBM = 0 no component and no MXCSR (RFBM[1]=0) is written. */
TEST(xsavec_rfbm_zero_header_only) {
    need_xsavec();
    XR_SDE_XFAIL("XSAVEC with RFBM=0 leaves XCOMP_BV[63] clear");
    memset(g_a.b, SENT, 1024);
    xs_req q; q_init(&q);
    q_sav(&q, Q_S1, g_a.b, 0, XO_XSAVEC);
    xs_run(&q, T_VK_XMM);
    CHECK_EQ(g64(g_a.b + 512), 0);
    CHECK_EQ(g64(g_a.b + 520), XS_COMPACT);
    CHECK_UNTOUCHED(g_a.b, 512, "legacy region with RFBM=0");
    CHECK_UNTOUCHED(g_a.b + 576, 1024 - 576, "bytes past the 576-byte compacted size");
    t_info("header_16_63", "%s", all_eq(g_a.b + 528, 48, SENT) ? "untouched" : all_eq(g_a.b + 528, 48, 0) ? "zeroed" : "other");
}
/* Compacted XRSTOR: RFBM[i]=1 with XSTATE_BV[i]=0 initialises component i, and XSTATE_BV may only name
 * components of XCOMP_BV - so RFBM bits outside XCOMP_BV mean "initialise". XCOMP_BV = 1<<63 alone (no
 * components, 576-byte area) is a valid compacted image that initialises every RFBM component. */
static void xrstor_cmp_partial_xcomp(uint64_t xcomp, uint64_t bv, const char *lbl) {
    xs_state p1, p2; st_pattern(&p1, 0x194); st_pattern(&p2, 0x195);
    xs_make_std(g_a.b, &p1, g_m);
    xs_make_cmp(g_c.b, &p2, xcomp, bv);
    memset(g_b.b, SENT, std_size());
    static t_vregs vout; memset(&vout, 0xEE, sizeof vout);
    uint32_t mx = 0;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
    q_rst(&q, Q_R2, g_c.b, g_m, XR_XRSTOR);
    q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
    q.s[Q_VOUT] = (uintptr_t)&vout; q.s[Q_STMX] = (uintptr_t)&mx; q.s[Q_ENV] = (uintptr_t)g_env;
    xs_run(&q, g_kind);
    xs_state e; st_init(&e);
    st_take(&e, &p2, (unsigned)(bv & 0xE7));
    int mx_known = (bv & 2) != 0;                         /* SSE loaded -> MXCSR from the area */
    if (mx_known) e.mxcsr = p2.mxcsr;
    xs_cmp(g_b.b, 0, &e, comps_of(g_m) & ~(mx_known ? 0 : XW_MX), 0, lbl);
    xs_cmp_vregs(&vout, &e, g_kind, (unsigned)g_m & kind_what(g_kind), 0, lbl);
    if (bv & 1) check_env(g_env, &e, lbl);
    else CHECK_EQ(g16(g_env + 8), 0xFFFF);
    if (mx_known) CHECKF(mx == e.mxcsr, "%s: MXCSR 0x%08x want 0x%08x", lbl, mx, e.mxcsr);
    else {
        CHECKF(mx == 0x1F80 || mx == p2.mxcsr, "%s: MXCSR 0x%08x neither 0x1F80 nor the area value 0x%08x", lbl, mx, p2.mxcsr);
        t_info("mxcsr", "%s: %s", lbl, mx == 0x1F80 ? "initialised" : "loaded-from-area");
    }
}
TEST(xrstor_cmp_empty_xcomp_inits_all) { need_xsavec(); xrstor_cmp_partial_xcomp(0, 0, "XCOMP_BV=1<<63, XSTATE_BV=0, RFBM=all"); }
TEST(xrstor_cmp_rfbm_outside_xcomp_inits) {
    need_xsavec();
    if (!(g_m & ~3ull)) SKIP("no component beyond x87/SSE enabled in XCR0 (0x%llx)", (unsigned long long)cpu()->xcr0);
    xrstor_cmp_partial_xcomp(3, 3, "XCOMP_BV=XSTATE_BV=x87|SSE, RFBM=all");
}
TEST(xsavec_init_component_not_saved) {
    need_xsavec();
    XR_SDE_XFAIL("XCOMP_BV drops RFBM components that are in their initial state");
    uint32_t csz = xsave_compacted_size(g_m);
    memset(g_a.b, SENT, csz);
    uint32_t mx = 0x1F80 | 0x2000;
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q.s[Q_MXCSR] = (uintptr_t)&mx;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVEC);
    xs_run(&q, T_VK_XMM);
    xs_state e; st_init(&e); e.mxcsr = mx;
    uint64_t bv = g64(g_a.b + 512);
    CHECK_EQ(g64(g_a.b + 520), XS_COMPACT | g_m);
    CHECK_EQ(g32(g_a.b + 24), mx);                        /* MXCSR written whenever RFBM[1]|RFBM[2] */
    xs_cmp(g_a.b, XS_COMPACT | g_m, &e, comps_of(g_m), CF_BV, "XSAVEC of init state");
    t_info("xsavec_bv_init_state", "0x%llx", (unsigned long long)bv);
    if (!(bv & 1)) t_info("x87_region_untouched_when_init", "%d", all_eq(g_a.b, 24, SENT) && all_eq(g_a.b + 32, 128, SENT));
}
TEST(xsavec32_vs_xsavec64_forms) {
    need_xsavec();
#ifndef T_X64
    SKIP("XSAVEC64 exists only in 64-bit mode");
#else
    xs_state p; st_pattern(&p, 0x193);
    static t_vregs vin; st_to_vregs(&p, &vin);
    uint32_t csz = xsave_compacted_size(g_m);
    memset(g_a.b, SENT, csz); memset(g_b.b, SENT, csz);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q.s[Q_VIN] = (uintptr_t)&vin; q.s[Q_MXCSR] = (uintptr_t)&p.mxcsr;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVEC);
    q_sav(&q, Q_S2, g_b.b, g_m, XO_XSAVEC32);
    xs_run(&q, g_kind);
    CHECK_MEM(g_b.b + 24, g_a.b + 24, 416 - 24);
    CHECK_MEM(g_b.b + 512, g_a.b + 512, 16);
    CHECK_MEM(g_b.b + 576, g_a.b + 576, csz - 576);
#endif
}

/* ======================= AVX-512 components ======================================= */
TEST(zmm_xsave_from_registers) {
    need_avx512();
    xs_state p; st_pattern(&p, 0x1A0);
    static t_vregs vin; st_to_vregs(&p, &vin);
    memset(g_a.b, SENT, std_size());
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);
    q.s[Q_VIN] = (uintptr_t)&vin;
    q_sav(&q, Q_S1, g_a.b, g_m, XO_XSAVE);
    xs_run(&q, T_VK_ZMM);
    xs_state e; st_loaded(&e, &p, T_VK_ZMM, 0, NULL, 0, 0);
    xs_cmp(g_a.b, 0, &e, (unsigned)(g_m & 0xE6), 0, "XSAVE after EVEX loads");
    uint64_t bv = g64(g_a.b + 512);
#ifdef T_X64
    CHECK_EQ(bv & 0xE6, 0xE6);
#else
    CHECK_EQ(bv & 0x66, 0x66);
#endif
}
TEST(zmm_xrstor_roundtrip) { need_avx512(); do_roundtrip(g_m, 0x1A1, 0, XR_XRSTOR, T_VK_ZMM); }
TEST(zmm_rfbm_opmask_only)  { need_avx512(); xrstor_mix(g_m, 0x20, "RFBM=opmask", 0); }
TEST(zmm_rfbm_hi256_only)   { need_avx512(); xrstor_mix(g_m, 0x40, "RFBM=ZMM_Hi256", 0); }
TEST(zmm_rfbm_hi16_only)    {
    need_avx512();
#ifndef T_X64
    SKIP("Hi16_ZMM (zmm16-31) is not accessible outside 64-bit mode");
#endif
    xrstor_mix(g_m, 0x80, "RFBM=Hi16_ZMM", 0);
}
TEST(zmm_init_components)   { need_avx512(); xrstor_mix(g_m & ~(uint64_t)0xE0, g_m, "XSTATE_BV without AVX-512 components", 1); }
TEST(zmm_ymm_restore_keeps_hi256) { need_avx512(); xrstor_mix(g_m, 6, "RFBM=SSE|AVX keeps ZMM_Hi256/opmask", 1); }
TEST(zmm_xsavec_layouts) {
    need_avx512();
    if (!cpu_has(F_XSAVEC)) SKIP("XSAVEC not advertised");
    XR_SDE_XFAIL32("clears XCOMP_BV[7] (Hi16_ZMM) outside 64-bit mode");
    uint64_t ms[] = { 0x21, 0x41, 0x81, 0xE3, 0xE7, 0x60, 0xA0 };
    for (size_t i = 0; i < T_COUNTOF(ms); i++) xsavec_mask_case(ms[i] & g_m, T_VK_ZMM);
}

/* ======================= thread isolation / OS preservation ======================= */
typedef struct thr_ctx {
    int idx, iters, mode; unsigned long spin;
    volatile long *stop;
    int bad; unsigned first_diff; uint32_t first_tag;
} thr_ctx;
static x87_ptrs g_ptr_ref;   /* mode 3: pointer fields read back without a preemption window */
static size_t thr_area_size(void) { return ((size_t)xsave_std_size(g_m) + 63) & ~(size_t)63; }
static DWORD WINAPI thr_iso(void *arg) {
    thr_ctx *t = (thr_ctx *)arg;
    size_t sz = thr_area_size();
    uint8_t *mem = (uint8_t *)VirtualAlloc(NULL, 2 * sz + 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!mem) { t->bad = -1; return 0; }
    uint8_t *A = mem, *B = mem + sz;
    for (int it = 0; it < t->iters; it++) {
        uint32_t tag = 0x10000u + (uint32_t)t->idx * 256u + (uint32_t)it;
        xs_state s; st_pattern(&s, tag);
        xs_req q; q_init(&q);
        q.s[Q_SPIN] = t->spin; q.s[Q_STOP] = (uintptr_t)t->stop;
        unsigned d = 0;
        if (t->mode == 0) {                       /* XRSTOR -> preemption window -> XSAVE */
            xs_make_std(A, &s, g_m);
            memset(B, SENT, sz);
            q_rst(&q, Q_R1, A, g_m, XR_XRSTOR);
            q_sav(&q, Q_S1, B, g_m, XO_XSAVE);
            xs_run(&q, T_VK_XMM);
            d = xs_cmp(B, 0, &s, comps_of(g_m), CF_QUIET, "thread");
        } else if (t->mode == 1) {                /* FXRSTOR -> window -> FXSAVE */
            put_legacy(A, &s);
            memset(B, SENT, 512);
            q_rst(&q, Q_R1, A, 0, XR_FXRSTOR);
            q_sav(&q, Q_S1, B, 0, XO_FXSAVE);
            xs_run(&q, T_VK_XMM);
            d = xs_cmp(B, 0, &s, 3 | XW_MX, CF_QUIET, "thread");
        } else if (t->mode == 3) {                /* FXRSTOR with FOP/FIP/FDP -> window -> FXSAVE */
            put_legacy(A, &s); put_ptrs(A);
            memset(B, SENT, 512);
            q_rst(&q, Q_R1, A, 0, XR_FXRSTOR);
            q_sav(&q, Q_S1, B, 0, XO_FXSAVE);
            xs_run(&q, T_VK_XMM);
            d = xs_cmp(B, 0, &s, 3 | XW_MX, CF_QUIET, "thread");
            if (!ptrs_eq(get_ptrs(B), g_ptr_ref)) d |= 0x200;
        } else {                                  /* XRSTOR A -> window -> XSAVEOPT A */
            xs_make_std(A, &s, g_m);
            q_rst(&q, Q_R1, A, g_m, XR_XRSTOR);
            q_sav(&q, Q_S1, A, g_m, XO_XSAVEOPT);
            xs_run(&q, T_VK_XMM);
            d = xs_cmp(A, 0, &s, comps_of(g_m), CF_QUIET | CF_BV, "thread");
        }
        if (d) { if (!t->bad) { t->first_diff = d; t->first_tag = tag; } t->bad++; }
    }
    VirtualFree(mem, 0, MEM_RELEASE);
    return 0;
}
static void run_iso(int mode, int iters, unsigned long spin) {
    int ncpu = t_cpu_count();
    int n = 2 * ncpu; if (n < 4) n = 4; if (n > 32) n = 32;
    static thr_ctx ctx[32]; HANDLE h[32]; int started = 0;
    static volatile long stop;
    stop = 0;
    (void)mx_mask();                  /* initialise the lazily computed MXCSR_MASK before workers use it */
    for (int i = 0; i < n; i++) {
        memset(&ctx[i], 0, sizeof ctx[i]);
        ctx[i].idx = i; ctx[i].iters = iters; ctx[i].mode = mode; ctx[i].spin = spin; ctx[i].stop = &stop;
        h[started] = CreateThread(NULL, 0, thr_iso, &ctx[i], 0, NULL);
        if (h[started]) started++;
    }
    if (!started) SKIP("CreateThread failed: %lu", GetLastError());
    DWORD w = WaitForMultipleObjects((DWORD)started, h, TRUE, 30000);
    CHECKF(w != WAIT_TIMEOUT, "worker threads did not finish within 30 s");
    if (w == WAIT_TIMEOUT) {          /* end the spin windows; workers must be gone before the case returns */
        stop = 1;
        w = WaitForMultipleObjects((DWORD)started, h, TRUE, 10000);
        REQUIREF(w != WAIT_TIMEOUT, "worker threads still running 10 s after the stop request");
    }
    for (int i = 0; i < started; i++) CloseHandle(h[i]);
    int total = 0;
    for (int i = 0; i < started; i++) {
        CHECKF(ctx[i].bad >= 0, "thread %d: VirtualAlloc failed", i);
        CHECKF(ctx[i].bad == 0, "thread %d: %d/%d iterations lost state (first tag 0x%x parts 0x%x: bit0 x87 1 sse 2 avx 5 k 6 zmmh 7 hi16 8 mxcsr 9 FOP/FIP/FDP)",
               i, ctx[i].bad, iters, ctx[i].first_tag, ctx[i].first_diff);
        if (ctx[i].bad > 0) total += ctx[i].bad;
    }
    t_info("threads", "%d (ncpu %d), iterations %d, spin %lu, lost %d", started, ncpu, iters, spin, total);
    if (started < n) t_note("only %d of %d threads could be created", started, n);
}
TEST(thread_xrstor_xsave_isolation) { need_xsave(); run_iso(0, 12, 300000); }
TEST(thread_fxrstor_fxsave_isolation) { need_fxsr(); run_iso(1, 12, 300000); }
TEST(thread_xsaveopt_tracking) { need_xsaveopt(); run_iso(2, 12, 300000); }
/* The OS must preserve the x87 last-instruction/operand pointers across context switches (they are part
 * of the x87 XSAVE component). Reference = the same FXRSTOR->FXSAVE without a window. */
TEST(thread_x87_pointer_preservation) {
    need_fxsr();
    if (!ptrs_strict()) SKIP("processor stores FOP/FIP/FDP only when FSW.ES=1 (no XSaveErPtr): nothing to preserve");
    xs_state p; st_pattern(&p, 0x1B3);
    put_legacy(g_a.b, &p); put_ptrs(g_a.b);
    memset(g_b.b, SENT, 512);
    xs_req q; q_init(&q);
    q_rst(&q, Q_R1, g_a.b, 0, XR_FXRSTOR);
    q_sav(&q, Q_S1, g_b.b, 0, XO_FXSAVE);
    xs_run(&q, T_VK_XMM);
    g_ptr_ref = get_ptrs(g_b.b);
    check_ptrs(g_ptr_ref, "reference without window");
    run_iso(3, 12, 300000);
}

static xs_state g_hold_s;
static volatile long g_hold_stop, g_hold_rounds, g_hold_bad;
static unsigned g_hold_first;
/* Holds its pattern in the registers (XRSTOR -> spin -> XSAVE) round after round until told to stop, so
 * it occupies CPU 0 for the whole observation window however fast PAUSE is; bounded to 20 s. */
static DWORD WINAPI thr_holder(void *arg) {
    uint8_t *B = (uint8_t *)arg;
    double t0 = t_now_ms();
    while (!g_hold_stop && t_now_ms() - t0 < 20000.0) {
        memset(B, SENT, std_size());
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_a.b, g_m, XR_XRSTOR);
        q.s[Q_SPIN] = 2000000ul; q.s[Q_STOP] = (uintptr_t)&g_hold_stop;
        q_sav(&q, Q_S1, B, g_m, XO_XSAVE);
        xs_run(&q, T_VK_XMM);
        unsigned d = xs_cmp(B, 0, &g_hold_s, comps_of(g_m), CF_QUIET, "holder");
        if (d && !g_hold_bad) g_hold_first = d;
        if (d) { g_hold_bad++; break; }                   /* keep the bad image in B for diagnostics */
        g_hold_rounds++;
    }
    return 0;
}
TEST(thread_no_pattern_leak_same_cpu) {
    need_xsave();
    st_pattern(&g_hold_s, 0x7777);
    xs_make_std(g_a.b, &g_hold_s, g_m);
    memset(g_c.b, SENT, std_size());
    /* words that identify the holder's state */
    static uint64_t words[600]; int nw = 0;
    for (int i = 0; i < 16; i++) for (int w = 0; w < 2; w++) { words[nw++] = g64(g_hold_s.xmm[i] + 8 * w); words[nw++] = g64(g_hold_s.ymmh[i] + 8 * w); }
    for (int i = 0; i < 16; i++) for (int w = 0; w < 4; w++) words[nw++] = g64(g_hold_s.zmmh[i] + 8 * w);
    for (int i = 0; i < 16; i++) for (int w = 0; w < 8; w++) words[nw++] = g64(g_hold_s.hi16[i] + 8 * w);
    for (int i = 0; i < 8; i++) { words[nw++] = g_hold_s.k[i]; words[nw++] = g64(g_hold_s.st[i].b); }
    uintptr_t prev = t_pin_thread(0);
    if (!prev) SKIP("cannot pin to CPU 0");
    g_hold_stop = 0; g_hold_rounds = 0; g_hold_bad = 0; g_hold_first = 0;
    HANDLE h = CreateThread(NULL, 0, thr_holder, g_c.b, CREATE_SUSPENDED, NULL);
    if (!h) { t_unpin_thread(prev); SKIP("CreateThread failed: %lu", GetLastError()); }
    SetThreadAffinityMask(h, 1);
    ResumeThread(h);
    Sleep(20);
    int leaks = 0, iters = 0;
    size_t sz = std_size();
    for (iters = 0; iters < 60; iters++) {
        SwitchToThread();
        memset(g_b.b, 0, sz);
        xs_req q; q_init(&q);
        q_rst(&q, Q_R1, g_z.b, g_m, XR_XRSTOR);        /* own state: init; a leak can only enter during the spin */
        q.s[Q_SPIN] = 20000;
        q_sav(&q, Q_S1, g_b.b, g_m, XO_XSAVE);
        xs_run(&q, T_VK_XMM);
        for (size_t off = 0; off + 8 <= sz; off += 8) {
            if (off >= 512 && off < 576) continue;
            uint64_t v = g64(g_b.b + off);
            for (int k = 0; k < nw; k++) if (v == words[k]) { if (!leaks) t_note("holder word 0x%016llx found at +%zu", (unsigned long long)v, off); leaks++; break; }
        }
    }
    g_hold_stop = 1;
    DWORD w = WaitForSingleObject(h, 10000);
    CloseHandle(h);
    t_unpin_thread(prev);
    CHECKF(w == WAIT_OBJECT_0, "holder thread did not stop");
    CHECKF(leaks == 0, "%d register words of another thread appeared in this thread's XSAVE image", leaks);
    CHECKF(g_hold_bad == 0, "holder thread lost its state while sharing CPU 0 (parts 0x%x)", g_hold_first);
    if (g_hold_bad) xs_cmp(g_c.b, 0, &g_hold_s, comps_of(g_m), 0, "holder state after sharing CPU 0");
    CHECKF(g_hold_rounds + g_hold_bad > 0, "holder thread completed no round");
    t_info("iterations", "%d, holder rounds %ld", iters, (long)g_hold_rounds);
}

/* ======================= deliberate faults (child processes) ======================= */
static void *volatile g_site, *volatile g_resume;
static volatile LONG g_fx_hit, g_fx_site_ok;
static volatile DWORD g_fx_code;
static volatile ULONG_PTR g_fx_i0, g_fx_i1;
static LONG CALLBACK xs_veh(EXCEPTION_POINTERS *ep) {
    void *site = g_site;
    if (!site || ep->ExceptionRecord->ExceptionAddress != site) return EXCEPTION_CONTINUE_SEARCH;
    g_fx_hit++; g_fx_site_ok = 1;
    g_fx_code = ep->ExceptionRecord->ExceptionCode;
    g_fx_i0 = ep->ExceptionRecord->NumberParameters > 0 ? ep->ExceptionRecord->ExceptionInformation[0] : 0;
    g_fx_i1 = ep->ExceptionRecord->NumberParameters > 1 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
#ifdef T_X64
    ep->ContextRecord->Rip = (DWORD64)(uintptr_t)g_resume;
#else
    ep->ContextRecord->Eip = (DWORD)(uintptr_t)g_resume;
#endif
    g_site = NULL;
    return EXCEPTION_CONTINUE_EXECUTION;
}
#ifdef T_X64
#define SITE_PRE "lea 1f(%%rip), %%r11\n\tmov %%r11, %[site]\n\tlea 2f(%%rip), %%r11\n\tmov %%r11, %[res]\n\t"
#define SITE_CLOB "r11"
#else
#define SITE_PRE "movl $1f, %[site]\n\tmovl $2f, %[res]\n\t"
#define SITE_CLOB "cc"
#endif
#define FAULT_MEM(insn, ptr, m) __asm__ volatile(SITE_PRE "1:\t" insn " (%[p])\n2:\n" \
    : [site] "=m"(g_site), [res] "=m"(g_resume) : [p] "r"(ptr), "a"((uint32_t)(m)), "d"((uint32_t)((uint64_t)(m) >> 32)) \
    : "memory", "cc", SITE_CLOB, CLOB_X, CLOB_ST)

static T_NOINLINE void f_xsave(void *p, uint64_t m)    { FAULT_MEM("xsave" W64, p, m); }
static T_NOINLINE void f_xsaveopt(void *p, uint64_t m) { FAULT_MEM("xsaveopt" W64, p, m); }
static T_NOINLINE void f_xsavec(void *p, uint64_t m)   { FAULT_MEM("xsavec" W64, p, m); }
static T_NOINLINE void f_xrstor(void *p, uint64_t m)   { FAULT_MEM("xrstor" W64, p, m); }
static T_NOINLINE void f_fxsave(void *p)               { FAULT_MEM("fxsave" W64, p, 0); }
static T_NOINLINE void f_fxrstor(void *p)              { FAULT_MEM("fxrstor" W64, p, 0); }
static T_NOINLINE void f_xsaves(void *p, uint64_t m)   { FAULT_MEM("xsaves" W64, p, m); }
static T_NOINLINE void f_xrstors(void *p, uint64_t m)  { FAULT_MEM("xrstors" W64, p, m); }
static T_NOINLINE void f_xsetbv(uint64_t v) {
    __asm__ volatile(SITE_PRE "1:\txsetbv\n2:\n" : [site] "=m"(g_site), [res] "=m"(g_resume)
                     : "a"((uint32_t)v), "d"((uint32_t)(v >> 32)), "c"(0) : "memory", "cc", SITE_CLOB);
}

/* Usable size for the guard-page overrun children: the guard must start inside bytes the instruction is
 * certain to access. 64-bit mode: the last 64 bytes (the last component is fully saved/loaded). Outside
 * 64-bit mode XMM8-15 / YMM8-15 halves / ZMM16-31 are not accessible, so the tail of the area may be
 * skipped: the guard starts at the first extended component (YMM0-7 upper halves) or at the header. */
static size_t guard_cut(size_t sz) {
#ifdef T_X64
    return ((sz + 63) & ~(size_t)63) - 64;
#else
    return sz > 576 ? 576 : 512;
#endif
}
CHILD(xsf) {
    const char *n = argc > 1 ? argv[1] : "";
    cpu_detect();
    const t_cpu_info *c = cpu();
    g_m = c->xcr0 & XS_CARE;
    g_kbw = cpu_has(F_AVX512BW);
    AddVectoredExceptionHandler(1, xs_veh);
    xs_state p; st_pattern(&p, 0x1F0);
    size_t sz = c->osxsave ? xsave_std_size(g_m) : 512;
    int intact = -1;
    unsigned long long lo = 0, hi = 0;
    uint32_t mx0 = t_get_mxcsr();
    uint8_t *A = g_a.b;
    if (!strcmp(n, "xsave_unaligned") || !strcmp(n, "xsaveopt_unaligned") || !strcmp(n, "xsavec_unaligned")) {
        memset(A, SENT, sz + 128);
        uint8_t *u = A + (n[5] == 'c' ? 32 : 16);
        if (n[5] == '_') f_xsave(u, g_m); else if (n[5] == 'o') f_xsaveopt(u, g_m); else f_xsavec(u, g_m);
        intact = all_eq(A, sz + 128, SENT);
    } else if (!strcmp(n, "fxsave_unaligned")) {
        memset(A, SENT, 1024);
        f_fxsave(A + 8);
        intact = all_eq(A, 1024, SENT);
    } else if (!strcmp(n, "fxrstor_unaligned")) {
        put_legacy(g_b.b, &p); memcpy(A + 8, g_b.b, 512);
        f_fxrstor(A + 8);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_unaligned")) {
        xs_make_std(g_b.b, &p, g_m); memcpy(A + 16, g_b.b, sz);
        f_xrstor(A + 16, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_bv_not_in_xcr0")) {
        int bit = 62;
        for (int i = 8; i < 63; i++) if (!(c->xcr0 >> i & 1) && !(c->xss_supported >> i & 1)) { bit = i; break; }
        xs_make_std(A, &p, g_m | (1ull << bit));
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_std_xcomp_nonzero")) {
        xs_make_std(A, &p, g_m); p64(A + 520, 1);          /* XCOMP_BV[63]=0 but nonzero */
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_std_hdr_bytes_16_23")) {
        xs_make_std(A, &p, g_m); A[528] = 1;
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_cmp_xcomp_not_in_xcr0")) {
        xs_make_cmp(A, &p, g_m, g_m); p64(A + 520, XS_COMPACT | g_m | (1ull << 62));
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_cmp_bv_not_in_xcomp")) {
        xs_make_cmp(A, &p, 1, 3);                             /* XSTATE_BV has SSE, XCOMP_BV only x87 */
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_cmp_reserved_hdr")) {
        xs_make_cmp(A, &p, g_m, g_m); A[512 + 40] = 1;
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_mxcsr_reserved")) {
        p.mxcsr = 0x1F80u | 0x80000000u;
        xs_make_std(A, &p, g_m);
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_cmp_mxcsr_reserved")) {
        p.mxcsr = 0x1F80u | 0x80000000u;
        xs_make_cmp(A, &p, g_m, g_m);
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "fxrstor_mxcsr_reserved")) {
        p.mxcsr = 0x1F80u | 0x80000000u;
        put_legacy(A, &p);
        f_fxrstor(A);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_mxcsr_not_loaded")) {
        p.mxcsr = 0x1F80u | 0x80000000u;
        xs_make_std(A, &p, g_m);
        f_xrstor(A, T_XS_X87);                                /* RFBM[2:1]=0: MXCSR not loaded, no #GP */
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xsetbv")) {
        f_xsetbv(c->xcr0);
    } else if (!strcmp(n, "xsaves")) {
        memset(A, 0, 4096);
        f_xsaves(A, g_m);
    } else if (!strcmp(n, "xrstors")) {
        xs_make_cmp(A, &p, g_m, 0);
        f_xrstors(A, g_m);
    } else if (!strcmp(n, "xrstor_mxcsr_reserved_low")) {
        uint32_t m = mx_mask(), bit = 0;
        while (bit < 32 && (m >> bit & 1)) bit++;              /* lowest MXCSR bit the processor does not implement */
        if (bit == 32) { printf("RESULT fault=9\n"); return 6; }
        lo = bit;
        p.mxcsr = 0x1F80u | (1u << bit);
        xs_make_std(A, &p, g_m);
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xrstor_cmp_without_xsavec")) {
        xs_make_cmp(A, &p, g_m, g_m);                         /* XCOMP_BV[63]=1 on a CPU without compaction */
        f_xrstor(A, g_m);
        intact = t_get_mxcsr() == mx0;
    } else if (!strcmp(n, "xsave_readonly")) {
        uint8_t *g = (uint8_t *)VirtualAlloc(NULL, 2 * sz + 8192, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        DWORD old;
        if (!g || !VirtualProtect(g, 2 * sz + 8192, PAGE_READONLY, &old)) { printf("RESULT fault=8\n"); return 4; }
        lo = (unsigned long long)(uintptr_t)g; hi = (unsigned long long)(uintptr_t)(g + sz);
        f_xsave(g, g_m);
        intact = all_eq(g, sz, 0);
    } else if (!strcmp(n, "xrstor_guard")) {
        size_t asz = guard_cut(sz);
        void *base = NULL;
        uint8_t *g = (uint8_t *)t_guarded_alloc(asz, &base);
        if (!g) { printf("RESULT fault=8\n"); return 4; }
        xs_make_std(g_b.b, &p, g_m);                          /* every component in use: all are read */
        memcpy(g, g_b.b, asz);
        lo = (unsigned long long)(uintptr_t)(g + asz); hi = (unsigned long long)(uintptr_t)(g + sz);
        f_xrstor(g, g_m);
    } else if (!strcmp(n, "xsave_guard")) {
        size_t asz = guard_cut(sz);
        void *base = NULL;
        uint8_t *g = (uint8_t *)t_guarded_alloc(asz, &base);
        if (!g) { printf("RESULT fault=8\n"); return 4; }
        lo = (unsigned long long)(uintptr_t)(g + asz); hi = (unsigned long long)(uintptr_t)(g + sz);
        f_xsave(g, g_m);
    } else {
        printf("RESULT fault=7\n");
        return 5;
    }
    t_fninit(); t_set_mxcsr(0x1F80);
    printf("RESULT fault=%d code=0x%08lx site=%d info0=0x%llx info1=0x%llx intact=%d lo=0x%llx hi=0x%llx\n",
           (int)g_fx_hit, (unsigned long)g_fx_code, (int)g_fx_site_ok, (unsigned long long)g_fx_i0, (unsigned long long)g_fx_i1,
           intact, lo, hi);
    fflush(stdout);
    return 0;
}

enum { EXP_GP, EXP_GP_OR_UD, EXP_PF, EXP_NONE, EXP_PFR };
static void fault_case(const char *scen, int exp, int intact_hard) {
    static t_child_result r;
    /* SDE runs FXSAVE/FXRSTOR and XSAVES/XRSTORS natively (those cases pass); the emulated XSAVE family
     * and XSETBV have no #GP/#PF checks */
    if (strncmp(scen, "fx", 2) && strncmp(scen, "xsaves", 6) && strncmp(scen, "xrstors", 7))
        XR_SDE_XFAIL("no #GP/#PF checks in the emulated XSAVE family / XSETBV");
    REQUIREF(t_child_run("xsf", scen, 15000, &r), "spawn failed (error %lu)", r.spawn_error);
    CHECK(!r.timed_out);
    char line[512] = "";
    int found = t_child_find(&r, "RESULT ", line, sizeof line);
    CHECKF(found, "child printed no RESULT line (exit 0x%lx exc 0x%lx); output: %.400s", r.exit_code, r.exc_code, r.out);
    if (!found) return;
    int fault = -1, site = 0, intact = -1; unsigned long code = 0; unsigned long long i0 = 0, i1 = 0, lo = 0, hi = 0;
    sscanf(line, "fault=%d code=0x%lx site=%d info0=0x%llx info1=0x%llx intact=%d lo=0x%llx hi=0x%llx",
           &fault, &code, &site, &i0, &i1, &intact, &lo, &hi);
    CHECK_EQ(r.exit_code, 0);
    t_info("child", "%s", line);
    if (exp == EXP_NONE) {
        CHECKF(fault == 0, "%s: unexpected exception 0x%08lx", scen, code);
    } else {
        CHECKF(fault == 1, "%s: expected exactly one exception at the instruction, got %d", scen, fault);
        CHECKF(site == 1, "%s: exception address is not the faulting instruction", scen);
        if (exp == EXP_PF || exp == EXP_PFR) {
            CHECKF(code == EXC_AV, "%s: code 0x%08lx want STATUS_ACCESS_VIOLATION", scen, code);
            CHECKF(i0 == (exp == EXP_PF ? 1u : 0u), "%s: access type %llu want %d (%s)", scen, i0, exp == EXP_PF, exp == EXP_PF ? "write" : "read");
            CHECKF(i1 >= lo && i1 < hi, "%s: fault address 0x%llx outside [0x%llx, 0x%llx) (the guard page part of the area)", scen, i1, lo, hi);
        } else {
            int gp = code == EXC_AV || code == EXC_PRIV;
            CHECKF(gp || (exp == EXP_GP_OR_UD && code == EXC_ILL), "%s: code 0x%08lx is not a #GP translation (ACCESS_VIOLATION/PRIVILEGED_INSTRUCTION)%s",
                   scen, code, exp == EXP_GP_OR_UD ? " or #UD" : "");
            t_info("exception", "%s", code == EXC_AV ? "STATUS_ACCESS_VIOLATION" : code == EXC_PRIV ? "STATUS_PRIVILEGED_INSTRUCTION" :
                   code == EXC_ILL ? "STATUS_ILLEGAL_INSTRUCTION" : "other");
        }
    }
    if (intact >= 0) {
        if (intact_hard) CHECKF(intact == 1, "%s: state/memory modified by the faulting instruction", scen);
        else t_info("state_intact", "%d", intact);
    }
}
TEST(fault_xsave_unaligned)    { need_xsave(); fault_case("xsave_unaligned", EXP_GP, 1); }
TEST(fault_xsaveopt_unaligned) { need_xsaveopt(); fault_case("xsaveopt_unaligned", EXP_GP, 1); }
TEST(fault_xsavec_unaligned)   { need_xsavec(); fault_case("xsavec_unaligned", EXP_GP, 1); }
TEST(fault_xrstor_unaligned)   { need_xsave(); fault_case("xrstor_unaligned", EXP_GP, 0); }
TEST(fault_fxsave_unaligned)   { need_fxsr(); fault_case("fxsave_unaligned", EXP_GP, 1); }
TEST(fault_fxrstor_unaligned)  { need_fxsr(); fault_case("fxrstor_unaligned", EXP_GP, 0); }
TEST(fault_xrstor_bv_not_in_xcr0)       { need_xsave(); fault_case("xrstor_bv_not_in_xcr0", EXP_GP, 0); }
TEST(fault_xrstor_std_xcomp_nonzero)    { need_xsave(); fault_case("xrstor_std_xcomp_nonzero", EXP_GP, 0); }
TEST(fault_xrstor_std_hdr_bytes_16_23)  { need_xsave(); fault_case("xrstor_std_hdr_bytes_16_23", EXP_GP, 0); }
TEST(fault_xrstor_cmp_xcomp_not_in_xcr0) { need_xsavec(); fault_case("xrstor_cmp_xcomp_not_in_xcr0", EXP_GP, 0); }
TEST(fault_xrstor_cmp_bv_not_in_xcomp)  { need_xsavec(); fault_case("xrstor_cmp_bv_not_in_xcomp", EXP_GP, 0); }
TEST(fault_xrstor_cmp_reserved_hdr)     { need_xsavec(); fault_case("xrstor_cmp_reserved_hdr", EXP_GP, 0); }
TEST(fault_xrstor_mxcsr_reserved)       { need_xsave(); fault_case("xrstor_mxcsr_reserved", EXP_GP, 0); }
TEST(fault_xrstor_cmp_mxcsr_reserved)   { need_xsavec(); fault_case("xrstor_cmp_mxcsr_reserved", EXP_GP, 0); }
TEST(fault_fxrstor_mxcsr_reserved)      { need_fxsr(); fault_case("fxrstor_mxcsr_reserved", EXP_GP, 0); }
TEST(fault_xrstor_mxcsr_not_loaded_no_fault) { need_xsave(); fault_case("xrstor_mxcsr_not_loaded", EXP_NONE, 1); }
TEST(fault_xsetbv_cpl3)   { need_xsave(); fault_case("xsetbv", EXP_GP, 0); }
TEST(fault_xsaves_cpl3)   { need_xsave(); fault_case("xsaves", cpu_hw(F_XSAVES) ? EXP_GP : EXP_GP_OR_UD, 0); }
TEST(fault_xrstors_cpl3)  { need_xsave(); fault_case("xrstors", cpu_hw(F_XSAVES) ? EXP_GP : EXP_GP_OR_UD, 0); }
TEST(fault_xsave_guard_overrun) { need_xsave(); fault_case("xsave_guard", EXP_PF, 0); }
TEST(fault_xrstor_guard_overrun) { need_xsave(); fault_case("xrstor_guard", EXP_PFR, 0); }
TEST(fault_xsave_readonly_page)  { need_xsave(); fault_case("xsave_readonly", EXP_PF, 1); }
/* any MXCSR bit outside MXCSR_MASK (bit 16 on most CPUs, DAZ on the oldest) -> #GP, not only bit 31 */
TEST(fault_xrstor_mxcsr_lowest_reserved_bit) {
    need_xsave();
    uint32_t m = mx_mask(), bit = 0;
    while (bit < 32 && (m >> bit & 1)) bit++;
    if (bit == 32) SKIP("MXCSR_MASK 0x%08x implements every bit", m);
    t_info("bit", "%u", bit);
    fault_case("xrstor_mxcsr_reserved_low", EXP_GP, 0);
}
TEST(fault_xrstor_compacted_without_xsavec) {
    need_xsave();
    if (cpu_hw(F_XSAVEC)) SKIP("XSAVEC/compaction supported: XCOMP_BV[63]=1 is legal here (covered by the compacted cases)");
    fault_case("xrstor_cmp_without_xsavec", EXP_GP, 0);
}
