/* avx_arithmetic - AVX / AVX2 / FMA3 / F16C / BMI1 / BMI2 / ADX reference-vector conformance.
 *
 * Contract under test (Intel SDM vol.1 ch.4, 10-11, 14; vol.2 instruction pages; AMD APM vol.4/5):
 *   - 256-bit float/double arithmetic equals the per-lane scalar operation under the SAME MXCSR (rounding
 *     mode, DAZ, FTZ), exception flags are the OR of the per-lane flags; MXCSR rounding is honoured.
 *   - SSE/AVX NaN rules (SDM vol.1 4.8.3.5, Table 4-7): one NaN -> that NaN quieted; two NaNs -> src1 quieted;
 *     invalid operation with no NaN input -> default QNaN (sign set, "real indefinite"); MIN/MAX return src2
 *     whenever a NaN is involved or both operands are zero; CMPPS signalling predicates raise #I on QNaN.
 *   - VEX encoding: three-operand non-destructive form; VEX.128 zeroes bits 255:128 of the destination;
 *     legacy SSE leaves them unchanged; VZEROUPPER/VZEROALL.
 *   - in-lane (vpermilps, vshufps, vunpck*, vpshufb, vpalignr, vpslldq, vpack*, vphadd*) vs cross-lane
 *     (vpermd/ps, vpermq/pd, vperm2f128/i128) shuffle semantics; blends; broadcasts; insert/extract.
 *   - VMASKMOVPS/PD and VPMASKMOVD/Q: masked-off lanes are zero on load, untouched on store and never fault.
 *   - VGATHER*: masked elements keep the destination, the mask register is cleared on completion; a page
 *     fault is restartable: elements before the faulting one are complete (mask bits cleared) and the
 *     instruction resumes after the OS continues the thread (the OS must restore full YMM state).
 *   - FMA3: all 132/213/231 operand orders, a single rounding (checked against an exact integer model;
 *     the mingw CRT fma()/fmaf() are not correctly rounded, see avx_arithmetic_core.h), special operands.
 *   - F16C: vcvtph2ps exhaustively (exact), vcvtps2ph in all imm rounding modes and with MXCSR.RC.
 *   - BMI1/BMI2/ADX/LZCNT scalar semantics and their architecturally defined flags.
 *   - Windows: an unmasked SIMD FP exception raised by a VEX instruction is delivered as
 *     STATUS_FLOAT_MULTIPLE_TRAPS (0xC00002B5, the documented #XM mapping), the CONTEXT carries MXCSR and
 *     execution resumes with a handler-modified MXCSR; page faults inside VGATHER / VMASKMOV are delivered
 *     as STATUS_ACCESS_VIOLATION and the instruction restarts correctly after EXCEPTION_CONTINUE_EXECUTION.
 *   - Also: unmasked #O/#U/#P/#D (trigger in an upper YMM lane) and FMA #I deliver one STATUS_FLOAT_* exception
 *     and restart; DAZ applies to compare/min/max/sqrt/convert sources; F16C DAZ/FTZ treatment; VEX memory
 *     operands need no alignment (unaligned and page-split); VLDMXCSR/VSTMXCSR (+ reserved-bit #GP);
 *     vpextr/vpinsr/vinsertps immediates; VEX scalar merge forms.
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL. Deliberate faults run in CHILD()
 * processes. Every instruction under test executes inside one asm block (avx_arithmetic_core.h). */
#include "avx_arithmetic_core.h"
#include <math.h>

#define NEED(f) do { if (!cpu_has(F_##f)) SKIP(#f " not usable (CPUID bit %d, OS state %s)", cpu_hw(F_##f), \
                                                cpu_ymm_state_enabled() ? "enabled" : "not enabled"); } while (0)

/* run f; ymm1..ymm3 whose bit is set in keep must be unchanged (VEX non-destructive sources) */
static void vexec(agg_t *a, vop_fn f, vctx *c, unsigned keep) {
    vctx in = *c;
    f(c);
    for (int i = 1; i < 4; i++)
        if ((keep >> i) & 1) {
            a->n++;
            if (memcmp(&c->r[i], &in.r[i], 32)) agg_fail(a, "source ymm%d modified by a non-destructive VEX op", i);
        }
}
#define KEEP123 0xEu
#define KEEP12  0x6u

/* =========================================================================== feature report */
TEST(avx_feature_report) {
    static const t_feature fs[] = { F_AVX, F_AVX2, F_FMA, F_F16C, F_BMI1, F_BMI2, F_ADX, F_LZCNT };
    char buf[256]; size_t o = 0;
    for (size_t i = 0; i < T_COUNTOF(fs); i++)
        o += (size_t)snprintf(buf + o, sizeof buf - o, "%s%s=%d/%d", i ? " " : "", cpu_feature_name(fs[i]), cpu_hw(fs[i]), cpu_has(fs[i]));
    t_info("features_hw_usable", "%s", buf);
    t_info("ymm_state_enabled", "%d", cpu_ymm_state_enabled());
    /* architectural dependencies: a usable AVX2/FMA/F16C implies usable AVX (same YMM gate) */
    if (cpu_has(F_AVX2) || cpu_has(F_FMA) || cpu_has(F_F16C)) CHECKF(cpu_has(F_AVX), "AVX2/FMA/F16C usable but AVX not");
    if (cpu_hw(F_AVX) && !cpu_has(F_AVX)) t_note("AVX advertised by CPUID but YMM state not enabled by the OS: AVX cases SKIP");
    CHECK(1);
}

/* =========================================================================== packed FP vs scalar SSE */
VOP(vaddps_y_avx, "vaddps %%ymm2, %%ymm1, %%ymm0")  VOP(vaddpd_y_avx, "vaddpd %%ymm2, %%ymm1, %%ymm0")
VOP(vsubps_y_avx, "vsubps %%ymm2, %%ymm1, %%ymm0")  VOP(vsubpd_y_avx, "vsubpd %%ymm2, %%ymm1, %%ymm0")
VOP(vmulps_y_avx, "vmulps %%ymm2, %%ymm1, %%ymm0")  VOP(vmulpd_y_avx, "vmulpd %%ymm2, %%ymm1, %%ymm0")
VOP(vdivps_y_avx, "vdivps %%ymm2, %%ymm1, %%ymm0")  VOP(vdivpd_y_avx, "vdivpd %%ymm2, %%ymm1, %%ymm0")
VOP(vminps_y_avx, "vminps %%ymm2, %%ymm1, %%ymm0")  VOP(vminpd_y_avx, "vminpd %%ymm2, %%ymm1, %%ymm0")
VOP(vmaxps_y_avx, "vmaxps %%ymm2, %%ymm1, %%ymm0")  VOP(vmaxpd_y_avx, "vmaxpd %%ymm2, %%ymm1, %%ymm0")
VOP(vsqrtps_y_avx, "vsqrtps %%ymm1, %%ymm0")        VOP(vsqrtpd_y_avx, "vsqrtpd %%ymm1, %%ymm0")
VOP(vaddps_x_avx, "vaddps %%xmm2, %%xmm1, %%xmm0")  VOP(vaddpd_x_avx, "vaddpd %%xmm2, %%xmm1, %%xmm0")
VOP(vmulps_x_avx, "vmulps %%xmm2, %%xmm1, %%xmm0")  VOP(vdivpd_x_avx, "vdivpd %%xmm2, %%xmm1, %%xmm0")
VOP(vminps_x_avx, "vminps %%xmm2, %%xmm1, %%xmm0")  VOP(vmaxpd_x_avx, "vmaxpd %%xmm2, %%xmm1, %%xmm0")
VOP(vsqrtps_x_avx, "vsqrtps %%xmm1, %%xmm0")        VOP(vsubpd_x_avx, "vsubpd %%xmm2, %%xmm1, %%xmm0")
/* memory source operand form (src2 from memory) */
VOP(vaddps_ym_avx, "vaddps (" RC "), %%ymm1, %%ymm0") VOP(vmulpd_ym_avx, "vmulpd (" RC "), %%ymm1, %%ymm0")

typedef struct fpop { const char *name; int op, dbl, bytes; vop_fn f; int mem; } fpop;
static const fpop fp_ops[] = {
    { "vaddps ymm", OP_ADD, 0, 32, vaddps_y_avx, 0 }, { "vaddpd ymm", OP_ADD, 1, 32, vaddpd_y_avx, 0 },
    { "vsubps ymm", OP_SUB, 0, 32, vsubps_y_avx, 0 }, { "vsubpd ymm", OP_SUB, 1, 32, vsubpd_y_avx, 0 },
    { "vmulps ymm", OP_MUL, 0, 32, vmulps_y_avx, 0 }, { "vmulpd ymm", OP_MUL, 1, 32, vmulpd_y_avx, 0 },
    { "vdivps ymm", OP_DIV, 0, 32, vdivps_y_avx, 0 }, { "vdivpd ymm", OP_DIV, 1, 32, vdivpd_y_avx, 0 },
    { "vminps ymm", OP_MIN, 0, 32, vminps_y_avx, 0 }, { "vminpd ymm", OP_MIN, 1, 32, vminpd_y_avx, 0 },
    { "vmaxps ymm", OP_MAX, 0, 32, vmaxps_y_avx, 0 }, { "vmaxpd ymm", OP_MAX, 1, 32, vmaxpd_y_avx, 0 },
    { "vsqrtps ymm", OP_SQRT, 0, 32, vsqrtps_y_avx, 0 }, { "vsqrtpd ymm", OP_SQRT, 1, 32, vsqrtpd_y_avx, 0 },
    { "vaddps xmm", OP_ADD, 0, 16, vaddps_x_avx, 0 }, { "vaddpd xmm", OP_ADD, 1, 16, vaddpd_x_avx, 0 },
    { "vmulps xmm", OP_MUL, 0, 16, vmulps_x_avx, 0 }, { "vdivpd xmm", OP_DIV, 1, 16, vdivpd_x_avx, 0 },
    { "vminps xmm", OP_MIN, 0, 16, vminps_x_avx, 0 }, { "vmaxpd xmm", OP_MAX, 1, 16, vmaxpd_x_avx, 0 },
    { "vsqrtps xmm", OP_SQRT, 0, 16, vsqrtps_x_avx, 0 }, { "vsubpd xmm", OP_SUB, 1, 16, vsubpd_x_avx, 0 },
    { "vaddps ymm,m256", OP_ADD, 0, 32, vaddps_ym_avx, 1 }, { "vmulpd ymm,m256", OP_MUL, 1, 32, vmulpd_ym_avx, 1 },
};
static const uint32_t fp_modes[] = { MX_RC(RN), MX_RC(RD), MX_RC(RU), MX_RC(RZ), MX_DEFAULT | MX_FTZ | MX_DAZ,
                                     MX_RC(RU) | MX_FTZ, MX_RC(RD) | MX_DAZ };

/* one op, one MXCSR, nvec random vectors: lanes and flags vs legacy scalar SSE under the same MXCSR */
static void fp_vs_scalar(const fpop *o, uint32_t mx, int nvec, uint64_t seed) {
    agg_t a; agg_init(&a, "%s mxcsr=0x%04x", o->name, mx);
    t_rng r; t_rng_seed(&r, seed);
    static v256 mem;
    for (int it = 0; it < nvec; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        if (o->dbl) { gen_f64v(&r, &c.r[1], 0); gen_f64v(&r, &c.r[2], 0); }
        else { gen_f32v(&r, &c.r[1], 0); gen_f32v(&r, &c.r[2], 0); }
        if (o->mem) { mem = c.r[2]; c.mem = (uintptr_t)&mem; }
        c.mx_in = mx;
        vexec(&a, o->f, &c, o->mem ? 0x2u : KEEP12);
        v256 want = V_ZERO; uint32_t fl = 0;
        int n = o->bytes / (o->dbl ? 8 : 4);
        for (int i = 0; i < n; i++) {
            if (o->dbl) want.q[i] = sse_f64(o->op, c.r[1].q[i], o->mem ? mem.q[i] : c.r[2].q[i], mx, &fl);
            else want.d[i] = sse_f32(o->op, c.r[1].d[i], o->mem ? mem.d[i] : c.r[2].d[i], mx, &fl);
        }
        char ctx[400], ha[200], hb[200];
        snprintf(ctx, sizeof ctx, "vec %d a=[%s] b=[%s]", it, hexv(ha, sizeof ha, &c.r[1], (size_t)o->bytes, o->dbl ? 8 : 4),
                 hexv(hb, sizeof hb, o->mem ? &mem : &c.r[2], (size_t)o->bytes, o->dbl ? 8 : 4));
        agg_mem(&a, &c.r[0], &want, 32, o->dbl ? 8 : 4, ctx);   /* VEX.128: bits 255:128 must be zero */
        a.n++;
        if ((c.mx_out & MX_FLAGS) != fl) agg_fail(&a, "%s: MXCSR flags 0x%02x want 0x%02x (OR of per-lane scalar flags)", ctx, c.mx_out & MX_FLAGS, fl);
        if ((c.mx_out & ~MX_FLAGS) != (mx & ~MX_FLAGS)) agg_fail(&a, "MXCSR control bits changed: 0x%x -> 0x%x", mx, c.mx_out);
    }
    AGG_END(&a);
}
TEST(avx_fp_packed_vs_scalar_sse) {
    NEED(AVX);
    for (size_t i = 0; i < T_COUNTOF(fp_ops); i++)
        for (size_t m = 0; m < T_COUNTOF(fp_modes); m++)
            fp_vs_scalar(&fp_ops[i], fp_modes[m], 120, 0xA11 + i * 31 + m);
}

/* =========================================================================== known-answer rounding */
typedef struct kv32 { uint32_t a, b; uint32_t want[4]; } kv32;   /* want per RN, RD, RU, RZ */
TEST(avx_mxcsr_rounding_known_answers) {
    NEED(AVX);
    /* hand-derived: 1 + 2^-24 is a tie (RN -> even), 0.75 ulp, ties at odd, 1/3 (0x3eaaaaab RN) etc. */
    static const kv32 add[] = {
        { 0x3f800000, 0x33800000, { 0x3f800000, 0x3f800000, 0x3f800001, 0x3f800000 } },  /* 1 + 2^-24 */
        { 0xbf800000, 0xb3800000, { 0xbf800000, 0xbf800001, 0xbf800000, 0xbf800000 } },  /* -1 - 2^-24 */
        { 0x3f800000, 0x33c00000, { 0x3f800001, 0x3f800000, 0x3f800001, 0x3f800000 } },  /* 1 + 0.75ulp */
        { 0x3f800001, 0x33800000, { 0x3f800002, 0x3f800001, 0x3f800002, 0x3f800001 } },  /* odd + half ulp */
        { 0x7f7fffff, 0x72800000, { 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0x7f7fffff } },  /* MAX + quarter ulp: only RU overflows */
        { 0x3f800000, 0xbf800000, { 0x00000000, 0x80000000, 0x00000000, 0x00000000 } },  /* x - x: -0 only in RD */
    };
    static const kv32 dv[] = {
        { 0x3f800000, 0x40400000, { 0x3eaaaaab, 0x3eaaaaaa, 0x3eaaaaab, 0x3eaaaaaa } },  /* 1/3 */
        { 0x40000000, 0x40400000, { 0x3f2aaaab, 0x3f2aaaaa, 0x3f2aaaab, 0x3f2aaaaa } },  /* 2/3 */
        { 0xbf800000, 0x40400000, { 0xbeaaaaab, 0xbeaaaaab, 0xbeaaaaaa, 0xbeaaaaaa } },  /* -1/3 */
        { 0x7f7fffff, 0x3f000000, { 0x7f800000, 0x7f7fffff, 0x7f800000, 0x7f7fffff } },  /* MAX/0.5 overflow */
    };
    static const kv32 sq[] = {
        { 0x40000000, 0, { 0x3fb504f3, 0x3fb504f3, 0x3fb504f4, 0x3fb504f3 } },          /* sqrt(2) */
        { 0x40400000, 0, { 0x3fddb3d7, 0x3fddb3d7, 0x3fddb3d8, 0x3fddb3d7 } },          /* sqrt(3) */
    };
    for (int rc = 0; rc < 4; rc++) {
        agg_t a; agg_init(&a, "known answers %s", rc_name[rc]);
        for (int k = 0; k < 3; k++) {
            const kv32 *tab = k == 0 ? add : k == 1 ? dv : sq;
            int n = k == 0 ? (int)T_COUNTOF(add) : k == 1 ? (int)T_COUNTOF(dv) : (int)T_COUNTOF(sq);
            vop_fn f = k == 0 ? vaddps_y_avx : k == 1 ? vdivps_y_avx : vsqrtps_y_avx;
            for (int i = 0; i < n; i++) {
                vctx c; vctx_pattern(&c, 1);
                for (int l = 0; l < 8; l++) { c.r[1].d[l] = tab[i].a; c.r[2].d[l] = tab[i].b; }
                c.mx_in = MX_RC(rc);
                f(&c);
                char ctx[96]; snprintf(ctx, sizeof ctx, "%s 0x%08x,0x%08x", k == 0 ? "add" : k == 1 ? "div" : "sqrt", tab[i].a, tab[i].b);
                for (int l = 0; l < 8; l++) if (!agg_eq(&a, c.r[0].d[l], tab[i].want[rc], ctx)) break;
            }
        }
        /* double: 1/3 and 2/3 */
        static const uint64_t d13[4] = { 0x3fd5555555555555ull, 0x3fd5555555555555ull, 0x3fd5555555555556ull, 0x3fd5555555555555ull };
        static const uint64_t d23[4] = { 0x3fe5555555555555ull, 0x3fe5555555555555ull, 0x3fe5555555555556ull, 0x3fe5555555555555ull };
        vctx c; vctx_pattern(&c, 2);
        c.r[1].q[0] = c.r[1].q[2] = 0x3ff0000000000000ull; c.r[1].q[1] = c.r[1].q[3] = 0x4000000000000000ull;
        for (int l = 0; l < 4; l++) c.r[2].q[l] = 0x4008000000000000ull;
        c.mx_in = MX_RC(rc);
        vdivpd_y_avx(&c);
        agg_eq(&a, c.r[0].q[0], d13[rc], "vdivpd 1/3"); agg_eq(&a, c.r[0].q[1], d23[rc], "vdivpd 2/3");
        agg_eq(&a, c.r[0].q[2], d13[rc], "vdivpd 1/3 lane2"); agg_eq(&a, c.r[0].q[3], d23[rc], "vdivpd 2/3 lane3");
        AGG_END(&a);
    }
}

/* =========================================================================== MXCSR exception flags */
typedef struct kflag { const char *what; vop_fn f; uint32_t a, b, mx, want, wfl; } kflag;
TEST(avx_mxcsr_exception_flags_daz_ftz) {
    NEED(AVX);
    static const kflag k[] = {
        { "1/3 inexact",           vdivps_y_avx, 0x3f800000, 0x40400000, MX_DEFAULT, 0x3eaaaaab, MX_PE },
        { "MAX*2 overflow",        vmulps_y_avx, 0x7f7fffff, 0x40000000, MX_DEFAULT, 0x7f800000, MX_OE | MX_PE },
        { "MAX*2 overflow RZ",     vmulps_y_avx, 0x7f7fffff, 0x40000000, MX_RC(RZ),  0x7f7fffff, MX_OE | MX_PE },
        { "1/0 divide-by-zero",    vdivps_y_avx, 0x3f800000, 0x00000000, MX_DEFAULT, 0x7f800000, MX_ZE },
        { "-1/+0",                 vdivps_y_avx, 0xbf800000, 0x00000000, MX_DEFAULT, 0xff800000, MX_ZE },
        { "0/0 invalid",           vdivps_y_avx, 0x00000000, 0x00000000, MX_DEFAULT, 0xffc00000, MX_IE },
        { "inf-inf invalid",       vsubps_y_avx, 0x7f800000, 0x7f800000, MX_DEFAULT, 0xffc00000, MX_IE },
        { "0*inf invalid",         vmulps_y_avx, 0x00000000, 0xff800000, MX_DEFAULT, 0xffc00000, MX_IE },
        { "sqrt(-1) invalid",      vsqrtps_y_avx, 0xbf800000, 0,         MX_DEFAULT, 0xffc00000, MX_IE },
        { "sqrt(-0) = -0",         vsqrtps_y_avx, 0x80000000, 0,         MX_DEFAULT, 0x80000000, 0 },
        { "denorm+denorm exact",   vaddps_y_avx, 0x00000001, 0x00000001, MX_DEFAULT, 0x00000002, MX_DE },
        { "tiny exact: no UE",     vmulps_y_avx, 0x00800000, 0x3f000000, MX_DEFAULT, 0x00400000, 0 },
        { "tiny inexact tie",      vmulps_y_avx, 0x00800001, 0x3f000000, MX_DEFAULT, 0x00400000, MX_UE | MX_PE },
        { "FTZ tiny -> +0",        vmulps_y_avx, 0x00800000, 0x3f000000, MX_DEFAULT | MX_FTZ, 0x00000000, MX_UE | MX_PE },
        { "FTZ tiny -> -0",        vmulps_y_avx, 0x80800000, 0x3f000000, MX_DEFAULT | MX_FTZ, 0x80000000, MX_UE | MX_PE },
        { "DAZ denorm+0 -> 0",     vaddps_y_avx, 0x00000001, 0x00000000, MX_DEFAULT | MX_DAZ, 0x00000000, 0 },
        { "DAZ -denorm + -0 = -0", vaddps_y_avx, 0x80000001, 0x80000000, MX_DEFAULT | MX_DAZ, 0x80000000, 0 },
        { "denorm*2^24 no DAZ",    vmulps_y_avx, 0x00400000, 0x4b800000, MX_DEFAULT, 0x0c000000, MX_DE },
        { "denorm*2^24 DAZ",       vmulps_y_avx, 0x00400000, 0x4b800000, MX_DEFAULT | MX_DAZ, 0x00000000, 0 },
        { "max(QNaN,1) = 1 (#I)",  vmaxps_y_avx, 0x7fc00001, 0x3f800000, MX_DEFAULT, 0x3f800000, MX_IE },
        { "min(1,SNaN) = SNaN",    vminps_y_avx, 0x3f800000, 0x7fa00001, MX_DEFAULT, 0x7fa00001, MX_IE },
    };
    agg_t a; agg_init(&a, "vector exception flags / DAZ / FTZ");
    for (size_t i = 0; i < T_COUNTOF(k); i++) {
        vctx c; vctx_pattern(&c, 3);
        for (int l = 0; l < 8; l++) { c.r[1].d[l] = k[i].a; c.r[2].d[l] = k[i].b; }
        c.mx_in = k[i].mx;
        k[i].f(&c);
        int ok = 1;
        for (int l = 0; l < 8; l++) ok &= c.r[0].d[l] == k[i].want;
        a.n++;
        if (!ok || (c.mx_out & MX_FLAGS) != k[i].wfl)
            agg_fail(&a, "%s: lane0 0x%08x flags 0x%02x, want 0x%08x flags 0x%02x", k[i].what, c.r[0].d[0], c.mx_out & MX_FLAGS, k[i].want, k[i].wfl);
    }
    /* flags are sticky: a second op does not clear flags set in MXCSR before it */
    vctx c; vctx_pattern(&c, 4);
    for (int l = 0; l < 8; l++) { c.r[1].d[l] = 0x3f800000; c.r[2].d[l] = 0x40000000; }
    c.mx_in = MX_DEFAULT | MX_ZE | MX_OE;
    vaddps_y_avx(&c);
    agg_eq(&a, c.mx_out & MX_FLAGS, MX_ZE | MX_OE, "sticky flags after exact 1+2");
    /* double DAZ/FTZ */
    vctx_pattern(&c, 5);
    for (int l = 0; l < 4; l++) { c.r[1].q[l] = 0x0010000000000000ull; c.r[2].q[l] = 0x3fe0000000000000ull; }
    c.mx_in = MX_DEFAULT | MX_FTZ;
    vmulpd_y_avx(&c);
    agg_eq(&a, c.r[0].q[0], 0, "vmulpd FTZ tiny"); agg_eq(&a, c.mx_out & MX_FLAGS, MX_UE | MX_PE, "vmulpd FTZ flags");
    vctx_pattern(&c, 6);
    for (int l = 0; l < 4; l++) { c.r[1].q[l] = 0x0000000000000001ull; c.r[2].q[l] = 0x4330000000000000ull; }
    c.mx_in = MX_DEFAULT | MX_DAZ;
    vmulpd_y_avx(&c);
    agg_eq(&a, c.r[0].q[3], 0, "vmulpd DAZ denorm"); agg_eq(&a, c.mx_out & MX_FLAGS, 0, "vmulpd DAZ flags (no DE)");
    AGG_END(&a);
}

/* =========================================================================== NaN propagation */
TEST(avx_nan_propagation_rules) {
    NEED(AVX);
    const uint32_t Q1 = 0x7fc11111, Q2 = 0xffc22222, S1 = 0x7f811111, S2 = 0xff822222, ONE = 0x3f800000;
    /* lanes: (Q1,Q2) (S1,Q2) (Q1,S2) (1,S2) (S1,1) (1,Q2) (invalid) (S1,S2) */
    const uint32_t A[8] = { Q1, S1, Q1, ONE, S1, ONE, 0, S1 }, B[8] = { Q2, Q2, S2, S2, ONE, Q2, 0, S2 };
    const uint32_t W[8] = { Q1, Q1 | 0x400000, Q1, S2 | 0x400000, S1 | 0x400000, Q2, 0xffc00000, S1 | 0x400000 };
    struct { const char *n; vop_fn f; uint32_t ia, ib; } ops[] = {
        { "vaddps", vaddps_y_avx, 0x7f800000, 0xff800000 }, { "vsubps", vsubps_y_avx, 0xff800000, 0xff800000 },
        { "vmulps", vmulps_y_avx, 0x00000000, 0x7f800000 }, { "vdivps", vdivps_y_avx, 0x7f800000, 0xff800000 } };
    agg_t a; agg_init(&a, "NaN propagation (src1 first, quieted; default NaN 0xFFC00000)");
    for (size_t o = 0; o < T_COUNTOF(ops); o++) {
        vctx c; vctx_pattern(&c, 7);
        for (int l = 0; l < 8; l++) { c.r[1].d[l] = A[l]; c.r[2].d[l] = B[l]; }
        c.r[1].d[6] = ops[o].ia; c.r[2].d[6] = ops[o].ib;
        ops[o].f(&c);
        v256 w = V_ZERO; memcpy(w.d, W, sizeof W);
        agg_mem(&a, &c.r[0], &w, 32, 4, ops[o].n);
        agg_eq(&a, c.mx_out & MX_IE, MX_IE, ops[o].n);
        /* QNaN-only operands: no #I */
        vctx_pattern(&c, 8);
        for (int l = 0; l < 8; l++) { c.r[1].d[l] = (l & 1) ? Q1 : ONE; c.r[2].d[l] = (l & 2) ? Q2 : ONE; }
        ops[o].f(&c);
        int ok = 1;
        for (int l = 0; l < 8; l++) { uint32_t e = (l & 1) ? Q1 : (l & 2) ? Q2 : 0; if (e && c.r[0].d[l] != e) ok = 0; }
        a.n++;
        if (!ok || (c.mx_out & MX_IE)) agg_fail(&a, "%s QNaN-only: lanes or #I wrong (mxcsr 0x%x)", ops[o].n, c.mx_out);
    }
    /* double */
    const uint64_t DQ1 = 0x7ff8000011111111ull, DS2 = 0xfff0000022222222ull, DS1 = 0x7ff0000011111111ull;
    vctx c; vctx_pattern(&c, 9);
    c.r[1].q[0] = DQ1; c.r[2].q[0] = DS2;                                      /* Q,S -> src1 */
    c.r[1].q[1] = 0x3ff0000000000000ull; c.r[2].q[1] = DS2;                     /* 1,S -> quiet(S) */
    c.r[1].q[2] = DS1; c.r[2].q[2] = DS2;                                       /* S,S -> quiet(src1) */
    c.r[1].q[3] = 0x7ff0000000000000ull; c.r[2].q[3] = 0xfff0000000000000ull;  /* inf-inf */
    vaddpd_y_avx(&c);
    agg_eq(&a, c.r[0].q[0], DQ1, "vaddpd Q,S"); agg_eq(&a, c.r[0].q[1], DS2 | 0x0008000000000000ull, "vaddpd 1,S");
    agg_eq(&a, c.r[0].q[2], DS1 | 0x0008000000000000ull, "vaddpd S,S"); agg_eq(&a, c.r[0].q[3], 0xfff8000000000000ull, "vaddpd inf-inf");
    /* sqrt: SNaN quieted, negative -> default NaN, -0 -> -0, +inf -> +inf, -inf -> default */
    vctx_pattern(&c, 10);
    const uint32_t SA[8] = { S1, 0xbf800000, 0x80000000, 0x7f800000, 0xff800000, Q2, 0x40800000, 0x80000001 };
    const uint32_t SW[8] = { S1 | 0x400000, 0xffc00000, 0x80000000, 0x7f800000, 0xffc00000, Q2, 0x40000000, 0xffc00000 };
    memcpy(c.r[1].d, SA, sizeof SA);
    vsqrtps_y_avx(&c);
    v256 w = V_ZERO; memcpy(w.d, SW, sizeof SW);
    agg_mem(&a, &c.r[0], &w, 32, 4, "vsqrtps specials");
    agg_eq(&a, c.mx_out & MX_IE, MX_IE, "vsqrtps #I");
    /* min/max: src2 on NaN or on +-0 pair */
    vctx_pattern(&c, 11);
    const uint32_t MA[8] = { 0x3f800000, 0x40000000, 0x00000000, 0x80000000, Q1, ONE, S1, ONE };
    const uint32_t MB[8] = { 0x40000000, 0x3f800000, 0x80000000, 0x00000000, ONE, Q2, ONE, S2 };
    const uint32_t MXW[8] = { 0x40000000, 0x40000000, 0x80000000, 0x00000000, ONE, Q2, ONE, S2 };
    const uint32_t MNW[8] = { 0x3f800000, 0x3f800000, 0x80000000, 0x00000000, ONE, Q2, ONE, S2 };
    memcpy(c.r[1].d, MA, sizeof MA); memcpy(c.r[2].d, MB, sizeof MB);
    vctx c2 = c;
    vmaxps_y_avx(&c); vminps_y_avx(&c2);
    memcpy(w.d, MXW, sizeof MXW); agg_mem(&a, &c.r[0], &w, 32, 4, "vmaxps src2 rules");
    memcpy(w.d, MNW, sizeof MNW); agg_mem(&a, &c2.r[0], &w, 32, 4, "vminps src2 rules");
    agg_eq(&a, c.mx_out & MX_IE, MX_IE, "vmaxps #I on QNaN"); agg_eq(&a, c2.mx_out & MX_IE, MX_IE, "vminps #I on QNaN");
    AGG_END(&a);
}

/* =========================================================================== addsub / hadd / hsub / dp */
VOP(vaddsubps_avx, "vaddsubps %%ymm2, %%ymm1, %%ymm0") VOP(vaddsubpd_avx, "vaddsubpd %%ymm2, %%ymm1, %%ymm0")
VOP(vhaddps_avx, "vhaddps %%ymm2, %%ymm1, %%ymm0")     VOP(vhaddpd_avx, "vhaddpd %%ymm2, %%ymm1, %%ymm0")
VOP(vhsubps_avx, "vhsubps %%ymm2, %%ymm1, %%ymm0")     VOP(vhsubpd_avx, "vhsubpd %%ymm2, %%ymm1, %%ymm0")
TEST(avx_addsub_hadd_hsub_in_lane) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0xADD5);
    agg_t a; agg_init(&a, "vaddsub/vhadd/vhsub ps/pd (lane-local)");
    for (int it = 0; it < 300; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        uint32_t mx = MX_RC(it & 3);
        c.mx_in = mx;
        gen_f32v(&r, &c.r[1], 1); gen_f32v(&r, &c.r[2], 1);
        v256 w; uint32_t fl = 0; vctx o;
        o = c; vexec(&a, vaddsubps_avx, &o, KEEP12);
        for (int i = 0; i < 8; i++) w.d[i] = sse_f32((i & 1) ? OP_ADD : OP_SUB, c.r[1].d[i], c.r[2].d[i], mx, &fl);
        agg_mem(&a, &o.r[0], &w, 32, 4, "vaddsubps");
        o = c; vexec(&a, vhaddps_avx, &o, KEEP12);
        for (int L = 0; L < 2; L++) {
            const uint32_t *x = &c.r[1].d[4 * L], *y = &c.r[2].d[4 * L];
            w.d[4*L+0] = sse_f32(OP_ADD, x[0], x[1], mx, &fl); w.d[4*L+1] = sse_f32(OP_ADD, x[2], x[3], mx, &fl);
            w.d[4*L+2] = sse_f32(OP_ADD, y[0], y[1], mx, &fl); w.d[4*L+3] = sse_f32(OP_ADD, y[2], y[3], mx, &fl);
        }
        agg_mem(&a, &o.r[0], &w, 32, 4, "vhaddps");
        o = c; vexec(&a, vhsubps_avx, &o, KEEP12);
        for (int L = 0; L < 2; L++) {
            const uint32_t *x = &c.r[1].d[4 * L], *y = &c.r[2].d[4 * L];
            w.d[4*L+0] = sse_f32(OP_SUB, x[0], x[1], mx, &fl); w.d[4*L+1] = sse_f32(OP_SUB, x[2], x[3], mx, &fl);
            w.d[4*L+2] = sse_f32(OP_SUB, y[0], y[1], mx, &fl); w.d[4*L+3] = sse_f32(OP_SUB, y[2], y[3], mx, &fl);
        }
        agg_mem(&a, &o.r[0], &w, 32, 4, "vhsubps");
        gen_f64v(&r, &c.r[1], 1); gen_f64v(&r, &c.r[2], 1);
        o = c; vexec(&a, vaddsubpd_avx, &o, KEEP12);
        for (int i = 0; i < 4; i++) w.q[i] = sse_f64((i & 1) ? OP_ADD : OP_SUB, c.r[1].q[i], c.r[2].q[i], mx, &fl);
        agg_mem(&a, &o.r[0], &w, 32, 8, "vaddsubpd");
        o = c; vexec(&a, vhaddpd_avx, &o, KEEP12);
        for (int L = 0; L < 2; L++) {
            w.q[2*L] = sse_f64(OP_ADD, c.r[1].q[2*L], c.r[1].q[2*L+1], mx, &fl);
            w.q[2*L+1] = sse_f64(OP_ADD, c.r[2].q[2*L], c.r[2].q[2*L+1], mx, &fl);
        }
        agg_mem(&a, &o.r[0], &w, 32, 8, "vhaddpd");
        o = c; vexec(&a, vhsubpd_avx, &o, KEEP12);
        for (int L = 0; L < 2; L++) {
            w.q[2*L] = sse_f64(OP_SUB, c.r[1].q[2*L], c.r[1].q[2*L+1], mx, &fl);
            w.q[2*L+1] = sse_f64(OP_SUB, c.r[2].q[2*L], c.r[2].q[2*L+1], mx, &fl);
        }
        agg_mem(&a, &o.r[0], &w, 32, 8, "vhsubpd");
    }
    AGG_END(&a);
}

#define vdpps_INSN "vdpps %1, %%ymm2, %%ymm1, %%ymm0"
#define vdppd_INSN "vdppd %1, %%xmm2, %%xmm1, %%xmm0"
DEF_IMM256(vdpps) DEF_IMM256(vdppd)
static uint32_t i2f(int v) { /* small integer -> float bits, exact */
    if (!v) return 0;
    int s; uint64_t m = (uint64_t)(v < 0 ? -(int64_t)v : v); s = v < 0;
    return (uint32_t)fround(FMT32, s, u128_from(m), 0, RN, NULL);
}
static uint64_t i2d(int64_t v) {
    if (!v) return 0;
    uint64_t m = (uint64_t)(v < 0 ? -v : v);
    return fround(FMT64, v < 0, u128_from(m), 0, RN, NULL);
}
TEST(avx_dpps_dppd_all_imm) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0xD0D0);
    agg_t a; agg_init(&a, "vdpps ymm (256 imm, per-lane) / vdppd xmm (256 imm)");
    for (int imm = 0; imm < 256; imm++) {
        for (int it = 0; it < 4; it++) {
            vctx c; vctx_pattern(&c, (uint32_t)imm);
            int A[8], B[8];
            for (int i = 0; i < 8; i++) {
                A[i] = (int)t_rng_range(&r, 2001) - 1000; B[i] = (int)t_rng_range(&r, 2001) - 1000;
                c.r[1].d[i] = i2f(A[i]); c.r[2].d[i] = i2f(B[i]);
            }
            vexec(&a, vdpps_tab[imm], &c, KEEP12);
            v256 w = V_ZERO;
            for (int L = 0; L < 2; L++) {
                int64_t sum = 0;
                for (int j = 0; j < 4; j++) if ((imm >> (4 + j)) & 1) sum += (int64_t)A[4*L+j] * B[4*L+j];
                for (int j = 0; j < 4; j++) w.d[4*L+j] = ((imm >> j) & 1) ? i2f((int)sum) : 0;
            }
            char ctx[48]; snprintf(ctx, sizeof ctx, "vdpps imm=0x%02x", imm);
            agg_mem(&a, &c.r[0], &w, 32, 4, ctx);
            /* dppd */
            vctx_pattern(&c, (uint32_t)imm + 1000);
            for (int i = 0; i < 2; i++) { c.r[1].q[i] = i2d(A[i]); c.r[2].q[i] = i2d(B[i]); }
            vexec(&a, vdppd_tab[imm], &c, KEEP12);
            w = V_ZERO;
            int64_t sum = 0;
            for (int j = 0; j < 2; j++) if ((imm >> (4 + j)) & 1) sum += (int64_t)A[j] * B[j];
            for (int j = 0; j < 2; j++) w.q[j] = ((imm >> j) & 1) ? i2d(sum) : 0;
            snprintf(ctx, sizeof ctx, "vdppd imm=0x%02x", imm);
            agg_mem(&a, &c.r[0], &w, 32, 8, ctx);
        }
    }
    AGG_END(&a);
}

/* =========================================================================== round / rcp / rsqrt */
#define vroundps_INSN "vroundps %1, %%ymm1, %%ymm0"
#define vroundpd_INSN "vroundpd %1, %%ymm1, %%ymm0"
#define vroundss_INSN "vroundss %1, %%xmm2, %%xmm1, %%xmm0"
DEF_IMM32(vroundps) DEF_IMM32(vroundpd) DEF_IMM32(vroundss)
TEST(avx_vround_all_imm_modes) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x50D);
    for (int imm = 0; imm < 16; imm++) {
        agg_t a; agg_init(&a, "vroundps/pd/ss imm=%d", imm);
        for (int mrc = 0; mrc < 4; mrc++) {
            for (int it = 0; it < 40; it++) {
                vctx c; vctx_pattern(&c, (uint32_t)it);
                c.mx_in = MX_RC(mrc);
                for (int i = 0; i < 8; i++) {   /* bias towards values with a fractional part */
                    c.r[1].d[i] = (i & 1) ? gen_f32(&r, 0) : ((t_rng_u32(&r) & 0x80000000u) | ((uint32_t)(120 + t_rng_range(&r, 35)) << 23) | (t_rng_u32(&r) & 0x7fffff));
                    if (!t_rng_range(&r, 6)) c.r[1].d[i] = (t_rng_u32(&r) & 0x80000000u) | (0x3f000000u + ((uint32_t)t_rng_range(&r, 4) << 23));  /* .5 ties */
                }
                int mode = (imm & 4) ? mrc : (imm & 3), sup = (imm >> 3) & 1;
                vctx o = c; vexec(&a, vroundps_tab[imm], &o, KEEP12);
                v256 w; unsigned fl = 0;
                for (int i = 0; i < 8; i++) w.d[i] = (uint32_t)f_round_int(FMT32, c.r[1].d[i], mode, sup, &fl);
                char ctx[64]; snprintf(ctx, sizeof ctx, "vroundps mxcsr.rc=%s", rc_name[mrc]);
                agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
                agg_eq(&a, o.mx_out & (MX_IE | MX_PE), fl, "vroundps flags (IE, PE unless imm[3])");
                /* vroundss: low from src2, 127:32 from src1, 255:128 zero */
                o = c; c.r[2] = c.r[1]; o.r[2] = c.r[1]; t_pattern_reg(&o.r[1], 32, 77, 1); vctx o2 = o;
                vexec(&a, vroundss_tab[imm], &o, KEEP12);
                w = V_ZERO; memcpy(&w, &o2.r[1], 16); w.d[0] = (uint32_t)f_round_int(FMT32, o2.r[2].d[0], mode, sup, NULL);
                agg_mem(&a, &o.r[0], &w, 32, 4, "vroundss merge");
                gen_f64v(&r, &c.r[1], 0);
                for (int i = 0; i < 4; i += 2) c.r[1].q[i] = (t_rng_next(&r) & 0x8000000000000000ull) | ((uint64_t)(1018 + t_rng_range(&r, 60)) << 52) | (t_rng_next(&r) & 0xfffffffffffffull);
                o = c; vexec(&a, vroundpd_tab[imm], &o, KEEP12);
                fl = 0;
                for (int i = 0; i < 4; i++) w.q[i] = f_round_int(FMT64, c.r[1].q[i], mode, sup, &fl);
                agg_mem(&a, &o.r[0], &w, 32, 8, "vroundpd");
                agg_eq(&a, o.mx_out & (MX_IE | MX_PE), fl, "vroundpd flags");
            }
        }
        AGG_END(&a);
    }
}

VOP(vrcpps_avx, "vrcpps %%ymm1, %%ymm0") VOP(vrsqrtps_avx, "vrsqrtps %%ymm1, %%ymm0")
static double f32_to_double(uint32_t x) { float f; memcpy(&f, &x, 4); return (double)f; }
TEST(avx_rcp_rsqrt_error_bound) {
    NEED(AVX);
    /* SDM RCPPS/RSQRTPS: |relative error| <= 1.5 * 2^-12 for normal inputs; 0 -> inf of the same sign;
     * inf -> 0; RSQRT of a negative non-zero value -> QNaN indefinite. RCPPS treats denormal inputs as 0. */
    t_rng r; t_rng_seed(&r, 0x3C9);
    agg_t a; agg_init(&a, "vrcpps/vrsqrtps relative error <= 1.5*2^-12");
    const double bound = 1.5 / 4096.0;
    double worst_rcp = 0, worst_rsq = 0;
    for (int it = 0; it < 2000; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        for (int i = 0; i < 8; i++) c.r[1].d[i] = ((uint32_t)(127 - 60 + t_rng_range(&r, 121)) << 23) | (t_rng_u32(&r) & 0x7fffff) | ((i & 1) ? 0x80000000u : 0);
        vctx o = c; vexec(&a, vrcpps_avx, &o, KEEP12);
        for (int i = 0; i < 8; i++) {
            double x = f32_to_double(c.r[1].d[i]), y = f32_to_double(o.r[0].d[i]), e = (y - 1.0 / x) * x;
            if (e < 0) e = -e;
            if (e > worst_rcp) worst_rcp = e;
            a.n++;
            if (!(e <= bound)) agg_fail(&a, "vrcpps(0x%08x) = 0x%08x rel.err %.3g", c.r[1].d[i], o.r[0].d[i], e);
        }
        for (int i = 0; i < 8; i++) c.r[1].d[i] &= 0x7fffffffu;
        o = c; vexec(&a, vrsqrtps_avx, &o, KEEP12);
        for (int i = 0; i < 8; i++) {
            double x = f32_to_double(c.r[1].d[i]), y = f32_to_double(o.r[0].d[i]), ex = 1.0 / sqrt(x), e = (y - ex) / ex;
            if (e < 0) e = -e;
            if (e > worst_rsq) worst_rsq = e;
            a.n++;
            if (!(e <= bound)) agg_fail(&a, "vrsqrtps(0x%08x) = 0x%08x rel.err %.3g", c.r[1].d[i], o.r[0].d[i], e);
        }
    }
    t_info("rcpps_worst_rel_err_x2^12", "%.4f", worst_rcp * 4096.0);
    t_info("rsqrtps_worst_rel_err_x2^12", "%.4f", worst_rsq * 4096.0);
    vctx c; vctx_pattern(&c, 1);
    const uint32_t in[8] = { 0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0x80400000, 0x3f800000, 0x40800000 };
    const uint32_t rcpw[6] = { 0x7f800000, 0xff800000, 0x00000000, 0x80000000, 0x7f800000, 0xff800000 };
    memcpy(c.r[1].d, in, sizeof in);
    vctx o = c; vrcpps_avx(&o);
    /* lanes 4,5: denormal input treated as 0 -> inf (Intel SDM RCPPS); recorded only as a note on other vendors */
    for (int i = 0; i < 6; i++) {
        char ctx[48]; snprintf(ctx, sizeof ctx, "vrcpps(0x%08x)", in[i]);
        if (i >= 4 && !cpu()->is_intel) { if (o.r[0].d[i] != rcpw[i]) t_note("%s = 0x%08x (Intel: 0x%08x)", ctx, o.r[0].d[i], rcpw[i]); continue; }
        agg_eq(&a, o.r[0].d[i], rcpw[i], ctx);
    }
    const uint32_t rsin[4] = { 0x00000000, 0x80000000, 0x7f800000, 0xbf800000 }, rsw[4] = { 0x7f800000, 0xff800000, 0x00000000, 0xffc00000 };
    memcpy(c.r[1].d, rsin, sizeof rsin);
    o = c; vrsqrtps_avx(&o);
    for (int i = 0; i < 4; i++) { char ctx[48]; snprintf(ctx, sizeof ctx, "vrsqrtps(0x%08x)", rsin[i]); agg_eq(&a, o.r[0].d[i], rsw[i], ctx); }
    AGG_END(&a);
}

/* =========================================================================== conversions */
VOP(vcvtps2pd_avx, "vcvtps2pd %%xmm1, %%ymm0")    VOP(vcvtpd2ps_avx, "vcvtpd2ps %%ymm1, %%xmm0")
VOP(vcvtdq2ps_avx, "vcvtdq2ps %%ymm1, %%ymm0")    VOP(vcvtps2dq_avx, "vcvtps2dq %%ymm1, %%ymm0")
VOP(vcvttps2dq_avx, "vcvttps2dq %%ymm1, %%ymm0")  VOP(vcvtdq2pd_avx, "vcvtdq2pd %%xmm1, %%ymm0")
VOP(vcvtpd2dq_avx, "vcvtpd2dq %%ymm1, %%xmm0")    VOP(vcvttpd2dq_avx, "vcvttpd2dq %%ymm1, %%xmm0")
VOP(vcvtpd2psx_avx, "vcvtpd2ps %%xmm1, %%xmm0")   VOP(vcvtsi2ss_avx, "vcvtsi2ss %%edx, %%xmm1, %%xmm0")
VOP(vcvtss2si_avx, "vcvtss2si %%xmm1, %%edx")     VOP(vcvttsd2si_avx, "vcvttsd2si %%xmm1, %%edx")
static uint32_t gen_f32_intish(t_rng *r) {
    if (t_rng_range(r, 3) == 0) return gen_f32(r, 0);
    return (t_rng_u32(r) & 0x80000000u) | ((uint32_t)(126 + t_rng_range(r, 34)) << 23) | (t_rng_u32(r) & 0x7fffff);
}
TEST(avx_conversions_all_rounding_modes) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0xC0C0);
    for (int rc = 0; rc < 4; rc++) {
        agg_t a; agg_init(&a, "vcvt* %s", rc_name[rc]);
        uint32_t mx = MX_RC(rc);
        for (int it = 0; it < 400; it++) {
            vctx c; vctx_pattern(&c, (uint32_t)it); c.mx_in = mx;
            v256 w; unsigned fl;
            /* ps -> pd (exact) */
            gen_f32v(&r, &c.r[1], 0);
            vctx o = c; vexec(&a, vcvtps2pd_avx, &o, KEEP12); fl = 0;
            for (int i = 0; i < 4; i++) w.q[i] = f_convert(FMT32, FMT64, c.r[1].d[i], rc, &fl);
            agg_mem(&a, &o.r[0], &w, 32, 8, "vcvtps2pd");
            agg_eq(&a, o.mx_out & (MX_IE | MX_DE), fl & (MX_IE | MX_DE), "vcvtps2pd flags");
            /* ps -> dq */
            for (int i = 0; i < 8; i++) c.r[1].d[i] = gen_f32_intish(&r);
            o = c; vexec(&a, vcvtps2dq_avx, &o, KEEP12); fl = 0;
            for (int i = 0; i < 8; i++) w.d[i] = f_to_i32(FMT32, c.r[1].d[i], rc, &fl);
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtps2dq");
            agg_eq(&a, o.mx_out & (MX_IE | MX_PE), fl & (MX_IE | MX_PE), "vcvtps2dq flags");
            o = c; vexec(&a, vcvttps2dq_avx, &o, KEEP12);
            for (int i = 0; i < 8; i++) w.d[i] = f_to_i32(FMT32, c.r[1].d[i], RZ, NULL);
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvttps2dq (ignores MXCSR.RC)");
            /* dq -> ps */
            gen_int(&r, &c.r[1], 4);
            o = c; vexec(&a, vcvtdq2ps_avx, &o, KEEP12); fl = 0;
            for (int i = 0; i < 8; i++) {
                int32_t v = c.r[1].sd[i];
                w.d[i] = v ? (uint32_t)fround(FMT32, v < 0, u128_from(v < 0 ? (uint64_t)(-(int64_t)v) : (uint64_t)v), 0, rc, &fl) : 0;
            }
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtdq2ps");
            agg_eq(&a, o.mx_out & MX_PE, fl & MX_PE, "vcvtdq2ps PE");
            o = c; vexec(&a, vcvtdq2pd_avx, &o, KEEP12);
            for (int i = 0; i < 4; i++) w.q[i] = i2d(c.r[1].sd[i]);
            agg_mem(&a, &o.r[0], &w, 32, 8, "vcvtdq2pd");
            /* pd -> ps (rounded, overflow/underflow) */
            for (int i = 0; i < 4; i++) {
                c.r[1].q[i] = gen_f64(&r, 0);
                if (t_rng_range(&r, 3) == 0)   /* around the float range limits */
                    c.r[1].q[i] = (t_rng_next(&r) & 0x8000000000000000ull) | ((uint64_t)(t_rng_range(&r, 2) ? 1023 + 120 + t_rng_range(&r, 12) : 1023 - 155 + t_rng_range(&r, 35)) << 52) | (t_rng_next(&r) & 0xfffffffffffffull);
            }
            o = c; vexec(&a, vcvtpd2ps_avx, &o, KEEP12); fl = 0;
            w = V_ZERO;
            for (int i = 0; i < 4; i++) w.d[i] = (uint32_t)f_convert(FMT64, FMT32, c.r[1].q[i], rc, &fl);
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtpd2ps ymm->xmm (upper zero)");
            agg_eq(&a, o.mx_out & (MX_IE | MX_OE | MX_UE | MX_PE | MX_DE), fl, "vcvtpd2ps flags");
            o = c; vexec(&a, vcvtpd2psx_avx, &o, KEEP12);
            w = V_ZERO; for (int i = 0; i < 2; i++) w.d[i] = (uint32_t)f_convert(FMT64, FMT32, c.r[1].q[i], rc, NULL);
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtpd2ps xmm->xmm");
            /* pd -> dq */
            for (int i = 0; i < 4; i++) {
                c.r[1].q[i] = (t_rng_next(&r) & 0x8000000000000000ull) | ((uint64_t)(1022 + t_rng_range(&r, 34)) << 52) | (t_rng_next(&r) & 0xfffffffffffffull);
                if (!t_rng_range(&r, 5)) c.r[1].q[i] = gen_f64(&r, 0);
            }
            o = c; vexec(&a, vcvtpd2dq_avx, &o, KEEP12); fl = 0;
            w = V_ZERO; for (int i = 0; i < 4; i++) w.d[i] = f_to_i32(FMT64, c.r[1].q[i], rc, &fl);
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtpd2dq");
            agg_eq(&a, o.mx_out & (MX_IE | MX_PE), fl & (MX_IE | MX_PE), "vcvtpd2dq flags");
            o = c; vexec(&a, vcvttpd2dq_avx, &o, KEEP12);
            w = V_ZERO; for (int i = 0; i < 4; i++) w.d[i] = f_to_i32(FMT64, c.r[1].q[i], RZ, NULL);
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvttpd2dq");
            /* scalar GPR forms */
            o = c; o.g_in = c.r[1].d[5]; vexec(&a, vcvtsi2ss_avx, &o, KEEP12);
            w = V_ZERO; memcpy(&w, &c.r[1], 16);
            { int32_t v = (int32_t)o.g_in; w.d[0] = v ? (uint32_t)fround(FMT32, v < 0, u128_from(v < 0 ? (uint64_t)(-(int64_t)v) : (uint64_t)v), 0, rc, NULL) : 0; }
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtsi2ss (127:32 from src1, upper zero)");
            c.r[1].d[0] = gen_f32_intish(&r);
            o = c; vexec(&a, vcvtss2si_avx, &o, KEEP12);
            agg_eq(&a, o.g_out, f_to_i32(FMT32, c.r[1].d[0], rc, NULL), "vcvtss2si");
            o = c; vexec(&a, vcvttsd2si_avx, &o, KEEP12);
            agg_eq(&a, o.g_out, f_to_i32(FMT64, c.r[1].q[0], RZ, NULL), "vcvttsd2si");
        }
        AGG_END(&a);
    }
}

/* =========================================================================== compares */
#define vcmpps_INSN "vcmpps %1, %%ymm2, %%ymm1, %%ymm0"
#define vcmppd_INSN "vcmppd %1, %%ymm2, %%ymm1, %%ymm0"
#define vcmpss_INSN "vcmpss %1, %%xmm2, %%xmm1, %%xmm0"
#define vcmpsd_INSN "vcmpsd %1, %%xmm2, %%xmm1, %%xmm0"
DEF_IMM32(vcmpps) DEF_IMM32(vcmppd) DEF_IMM32(vcmpss) DEF_IMM32(vcmpsd)
enum { REL_LT = 1, REL_EQ = 2, REL_GT = 4, REL_UN = 8 };
static const char *const cmp_pred_name[32] = {
    "EQ_OQ", "LT_OS", "LE_OS", "UNORD_Q", "NEQ_UQ", "NLT_US", "NLE_US", "ORD_Q", "EQ_UQ", "NGE_US", "NGT_US", "FALSE_OQ",
    "NEQ_OQ", "GE_OS", "GT_OS", "TRUE_UQ", "EQ_OS", "LT_OQ", "LE_OQ", "UNORD_S", "NEQ_US", "NLT_UQ", "NLE_UQ", "ORD_S",
    "EQ_US", "NGE_UQ", "NGT_UQ", "FALSE_OS", "NEQ_OS", "GE_OQ", "GT_OQ", "TRUE_US" };
/* SDM vol.2 CMPPS Table 3-1: set of relations for which the predicate is true, and whether it signals on QNaN */
static unsigned cmp_true_set(int imm) {
    static const unsigned base[8] = { REL_EQ, REL_LT, REL_LT | REL_EQ, REL_UN, REL_LT | REL_GT | REL_UN,
                                      REL_EQ | REL_GT | REL_UN, REL_GT | REL_UN, REL_LT | REL_EQ | REL_GT };
    unsigned s = base[imm & 7];
    if (imm & 8) s ^= REL_UN;
    return s;
}
static int cmp_signals_qnan(int imm) {
    static const int sig[16] = { 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0 };
    return sig[imm & 15] ^ ((imm >> 4) & 1);
}
static unsigned rel_f(ffmt f, uint64_t a, uint64_t b) {
    if (f_isnan(f, a) || f_isnan(f, b)) return REL_UN;
    uint64_t sm = fmt_sign(f) - 1;
    int64_t ka = (a & fmt_sign(f)) ? -(int64_t)(a & sm) : (int64_t)(a & sm);
    int64_t kb = (b & fmt_sign(f)) ? -(int64_t)(b & sm) : (int64_t)(b & sm);
    return ka < kb ? REL_LT : ka > kb ? REL_GT : REL_EQ;
}
TEST(avx_cmp_32_predicates_and_invalid_flag) {
    NEED(AVX);
    /* three operand sets: ordered only; with QNaN (no SNaN); with SNaN */
    static const uint32_t A0[8] = { 0x3f800000, 0x40000000, 0x40400000, 0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0xc0000000 };
    static const uint32_t B0[8] = { 0x40000000, 0x40000000, 0x40000000, 0x00000000, 0x7f800000, 0x3f800000, 0x00000000, 0xc0400000 };
    static const uint32_t A1[8] = { 0x7fc00000, 0x3f800000, 0xffc12345, 0x40000000, 0x3f800000, 0x7fc00001, 0x00000000, 0x40000000 };
    static const uint32_t B1[8] = { 0x3f800000, 0x7fc00000, 0x7fc00000, 0x40000000, 0x40000000, 0x00000000, 0x80000000, 0x3f800000 };
    static const uint32_t A2[8] = { 0x7f800001, 0x3f800000, 0x7fa00000, 0x40000000, 0x3f800000, 0x7fc00001, 0x00000000, 0x40000000 };
    static const uint32_t B2[8] = { 0x3f800000, 0xffa00000, 0x7fc00000, 0x40000000, 0x40000000, 0x00000000, 0x80000000, 0x3f800000 };
    const uint32_t *As[3] = { A0, A1, A2 }, *Bs[3] = { B0, B1, B2 };
    for (int imm = 0; imm < 32; imm++) {
        agg_t a; agg_init(&a, "vcmp ps/pd/ss/sd imm=%d (%s)", imm, cmp_pred_name[imm]);
        unsigned ts = cmp_true_set(imm);
        for (int set = 0; set < 3; set++) {
            vctx c; vctx_pattern(&c, (uint32_t)(imm * 3 + set));
            memcpy(c.r[1].d, As[set], 32); memcpy(c.r[2].d, Bs[set], 32);
            vctx o = c; vexec(&a, vcmpps_tab[imm], &o, KEEP12);
            v256 w; int anyq = 0, anys = 0;
            for (int i = 0; i < 8; i++) {
                w.d[i] = (rel_f(FMT32, As[set][i], Bs[set][i]) & ts) ? 0xffffffffu : 0;
                anyq |= f_isnan(FMT32, As[set][i]) || f_isnan(FMT32, Bs[set][i]);
                anys |= f_issnan(FMT32, As[set][i]) || f_issnan(FMT32, Bs[set][i]);
            }
            char ctx[64]; snprintf(ctx, sizeof ctx, "vcmpps set%d", set);
            agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
            uint32_t wie = (anys || (anyq && cmp_signals_qnan(imm))) ? MX_IE : 0;
            snprintf(ctx, sizeof ctx, "vcmpps set%d #I (signalling=%d)", set, cmp_signals_qnan(imm));
            agg_eq(&a, o.mx_out & MX_IE, wie, ctx);
            /* scalar: low lane result, 127:32 from src1, 255:128 zero */
            o = c; vexec(&a, vcmpss_tab[imm], &o, KEEP12);
            w = V_ZERO; memcpy(&w, &c.r[1], 16);
            w.d[0] = (rel_f(FMT32, As[set][0], Bs[set][0]) & ts) ? 0xffffffffu : 0;
            agg_mem(&a, &o.r[0], &w, 32, 4, "vcmpss");
            /* double: widen the same relations */
            for (int i = 0; i < 4; i++) {
                c.r[1].q[i] = f_convert(FMT32, FMT64, As[set][2 * i + (set ? 0 : 1)], RN, NULL);
                c.r[2].q[i] = f_convert(FMT32, FMT64, Bs[set][2 * i + (set ? 0 : 1)], RN, NULL);
                if (set == 2 && i == 0) c.r[1].q[0] = 0x7ff0000000000001ull;          /* keep an SNaN */
            }
            o = c; vexec(&a, vcmppd_tab[imm], &o, KEEP12);
            anyq = anys = 0;
            for (int i = 0; i < 4; i++) {
                w.q[i] = (rel_f(FMT64, c.r[1].q[i], c.r[2].q[i]) & ts) ? ~0ull : 0;
                anyq |= f_isnan(FMT64, c.r[1].q[i]) || f_isnan(FMT64, c.r[2].q[i]);
                anys |= f_issnan(FMT64, c.r[1].q[i]) || f_issnan(FMT64, c.r[2].q[i]);
            }
            agg_mem(&a, &o.r[0], &w, 32, 8, "vcmppd");
            agg_eq(&a, o.mx_out & MX_IE, (anys || (anyq && cmp_signals_qnan(imm))) ? MX_IE : 0, "vcmppd #I");
            o = c; vexec(&a, vcmpsd_tab[imm], &o, KEEP12);
            w = V_ZERO; w.q[1] = c.r[1].q[1];
            w.q[0] = (rel_f(FMT64, c.r[1].q[0], c.r[2].q[0]) & ts) ? ~0ull : 0;
            agg_mem(&a, &o.r[0], &w, 32, 8, "vcmpsd");
        }
        AGG_END(&a);
    }
}

VOP(vcomiss_avx, "vcomiss %%xmm2, %%xmm1")   VOP(vucomiss_avx, "vucomiss %%xmm2, %%xmm1")
VOP(vcomisd_avx, "vcomisd %%xmm2, %%xmm1")   VOP(vucomisd_avx, "vucomisd %%xmm2, %%xmm1")
/* float -> double keeping NaN signalling-ness (unlike cvtps2pd) */
static uint64_t f32_widen_raw(uint32_t x) {
    if (f_isnan(FMT32, x)) return ((uint64_t)(x >> 31) << 63) | 0x7ff0000000000000ull | ((uint64_t)(x & 0x7fffff) << 29);
    return f_convert(FMT32, FMT64, x, RN, NULL);
}
TEST(avx_comis_ucomis_flags) {
    NEED(AVX);
    static const struct { uint32_t a, b; unsigned rel; } v[] = {
        { 0x3f800000, 0x40000000, REL_LT }, { 0x40000000, 0x3f800000, REL_GT }, { 0x40000000, 0x40000000, REL_EQ },
        { 0x80000000, 0x00000000, REL_EQ }, { 0x7fc00000, 0x3f800000, REL_UN }, { 0x3f800000, 0x7f800001, REL_UN },
        { 0xff800000, 0x7f800000, REL_LT }, { 0x00000001, 0x00000000, REL_GT } };
    agg_t a; agg_init(&a, "vcomiss/vucomiss/vcomisd/vucomisd EFLAGS and #I");
    for (size_t i = 0; i < T_COUNTOF(v); i++) {
        unsigned want = v[i].rel == REL_UN ? (EF_ZF | EF_PF | EF_CF) : v[i].rel == REL_LT ? EF_CF : v[i].rel == REL_EQ ? EF_ZF : 0;
        int q = f_isnan(FMT32, v[i].a) || f_isnan(FMT32, v[i].b), s = f_issnan(FMT32, v[i].a) || f_issnan(FMT32, v[i].b);
        for (int k = 0; k < 4; k++) {
            vctx c; vctx_pattern(&c, (uint32_t)i);
            if (k < 2) { c.r[1].d[0] = v[i].a; c.r[2].d[0] = v[i].b; }
            else { c.r[1].q[0] = f32_widen_raw(v[i].a); c.r[2].q[0] = f32_widen_raw(v[i].b); }
            vop_fn f = k == 0 ? vcomiss_avx : k == 1 ? vucomiss_avx : k == 2 ? vcomisd_avx : vucomisd_avx;
            vexec(&a, f, &c, KEEP12);
            char ctx[80]; snprintf(ctx, sizeof ctx, "%s 0x%08x,0x%08x", k == 0 ? "vcomiss" : k == 1 ? "vucomiss" : k == 2 ? "vcomisd" : "vucomisd", v[i].a, v[i].b);
            agg_eq(&a, c.fl_out & (EF_ZF | EF_PF | EF_CF | EF_OF | EF_SF | EF_AF), want, ctx);
            int wie = (k & 1) ? s : q;      /* COMIS signals on any NaN, UCOMIS only on SNaN */
            agg_eq(&a, (c.mx_out & MX_IE) != 0, wie, ctx);
        }
    }
    AGG_END(&a);
}

/* =========================================================================== logic, VEX semantics */
VOP(vandps_avx, "vandps %%ymm2, %%ymm1, %%ymm0")  VOP(vandnps_avx, "vandnps %%ymm2, %%ymm1, %%ymm0")
VOP(vorpd_avx, "vorpd %%ymm2, %%ymm1, %%ymm0")    VOP(vxorps_avx, "vxorps %%ymm2, %%ymm1, %%ymm0")
VOP(vandnpd_avx, "vandnpd %%ymm2, %%ymm1, %%ymm0")
TEST(avx_fp_logic_bitwise) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x10C);
    agg_t a; agg_init(&a, "vandps/vandnps/vandnpd/vorpd/vxorps (bitwise, no FP exceptions)");
    for (int it = 0; it < 300; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        gen_f32v(&r, &c.r[1], 0); gen_f32v(&r, &c.r[2], 0);    /* includes SNaNs: logic ops never signal */
        v256 w; vctx o;
        o = c; vexec(&a, vandps_avx, &o, KEEP12);  for (int i = 0; i < 4; i++) w.q[i] = c.r[1].q[i] & c.r[2].q[i];  agg_mem(&a, &o.r[0], &w, 32, 8, "vandps");
        agg_eq(&a, o.mx_out & MX_FLAGS, 0, "vandps flags");
        o = c; vexec(&a, vandnps_avx, &o, KEEP12); for (int i = 0; i < 4; i++) w.q[i] = ~c.r[1].q[i] & c.r[2].q[i]; agg_mem(&a, &o.r[0], &w, 32, 8, "vandnps (~src1 & src2)");
        o = c; vexec(&a, vandnpd_avx, &o, KEEP12); agg_mem(&a, &o.r[0], &w, 32, 8, "vandnpd");
        o = c; vexec(&a, vorpd_avx, &o, KEEP12);   for (int i = 0; i < 4; i++) w.q[i] = c.r[1].q[i] | c.r[2].q[i];  agg_mem(&a, &o.r[0], &w, 32, 8, "vorpd");
        o = c; vexec(&a, vxorps_avx, &o, KEEP12);  for (int i = 0; i < 4; i++) w.q[i] = c.r[1].q[i] ^ c.r[2].q[i];  agg_mem(&a, &o.r[0], &w, 32, 8, "vxorps");
    }
    AGG_END(&a);
}

/* dest aliasing: dest == src1, dest == src2, all three equal */
VOP(vaddps_d1_avx, "vaddps %%ymm2, %%ymm1, %%ymm1")  VOP(vaddps_d2_avx, "vaddps %%ymm2, %%ymm1, %%ymm2")
VOP(vsubps_d2_avx, "vsubps %%ymm2, %%ymm1, %%ymm2")  VOP(vmulps_all_avx, "vmulps %%ymm1, %%ymm1, %%ymm1")
VOP(vsubps_rev_avx, "vsubps %%ymm1, %%ymm2, %%ymm0")
TEST(avx_vex_nondestructive_three_operand) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x3);
    agg_t a; agg_init(&a, "VEX 3-operand: dest independent of sources; aliasing; operand order");
    for (int it = 0; it < 200; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        gen_f32v(&r, &c.r[1], 1); gen_f32v(&r, &c.r[2], 1);
        v256 wadd, wsub, wrsub, wsq; uint32_t fl = 0;
        for (int i = 0; i < 8; i++) {
            wadd.d[i] = sse_f32(OP_ADD, c.r[1].d[i], c.r[2].d[i], MX_DEFAULT, &fl);
            wsub.d[i] = sse_f32(OP_SUB, c.r[1].d[i], c.r[2].d[i], MX_DEFAULT, &fl);
            wrsub.d[i] = sse_f32(OP_SUB, c.r[2].d[i], c.r[1].d[i], MX_DEFAULT, &fl);
            wsq.d[i] = sse_f32(OP_MUL, c.r[1].d[i], c.r[1].d[i], MX_DEFAULT, &fl);
        }
        vctx o = c; vaddps_y_avx(&o);
        agg_mem(&a, &o.r[0], &wadd, 32, 4, "ymm0 = ymm1+ymm2");
        agg_mem(&a, &o.r[1], &c.r[1], 32, 4, "ymm1 untouched"); agg_mem(&a, &o.r[2], &c.r[2], 32, 4, "ymm2 untouched");
        agg_mem(&a, &o.r[3], &c.r[3], 32, 4, "ymm3 untouched");
        o = c; vaddps_d1_avx(&o);
        agg_mem(&a, &o.r[1], &wadd, 32, 4, "ymm1 = ymm1+ymm2"); agg_mem(&a, &o.r[0], &c.r[0], 32, 4, "ymm0 untouched");
        o = c; vaddps_d2_avx(&o);
        agg_mem(&a, &o.r[2], &wadd, 32, 4, "ymm2 = ymm1+ymm2"); agg_mem(&a, &o.r[1], &c.r[1], 32, 4, "ymm1 untouched (d2)");
        o = c; vsubps_d2_avx(&o);
        agg_mem(&a, &o.r[2], &wsub, 32, 4, "ymm2 = ymm1-ymm2 (src1 is the minuend)");
        o = c; vsubps_rev_avx(&o);
        agg_mem(&a, &o.r[0], &wrsub, 32, 4, "ymm0 = ymm2-ymm1");
        o = c; vmulps_all_avx(&o);
        agg_mem(&a, &o.r[1], &wsq, 32, 4, "ymm1 = ymm1*ymm1");
    }
    AGG_END(&a);
}

/* VEX.128 zeroes 255:128 of the destination; legacy SSE preserves it (SDM vol.1 14.1.2) */
VOP(z_vaddps_avx, "vaddps %%xmm2, %%xmm1, %%xmm0")       VOP(z_vmovaps_avx, "vmovaps %%xmm1, %%xmm0")
VOP(z_vpxor_avx, "vpxor %%xmm2, %%xmm1, %%xmm0")         VOP(z_vmovd_avx, "vmovd %%edx, %%xmm0")
VOP(z_vcvtps2dq_avx, "vcvtps2dq %%xmm1, %%xmm0")         VOP(z_vaddss_avx, "vaddss %%xmm2, %%xmm1, %%xmm0")
VOP(z_vmovssm_avx, "vmovss (" RC "), %%xmm0")            VOP(z_vmovssr_avx, "vmovss %%xmm2, %%xmm1, %%xmm0")
VOP(z_vbcst_avx, "vbroadcastss (" RC "), %%xmm0")        VOP(z_vpinsrd_avx, "vpinsrd $1, %%edx, %%xmm1, %%xmm0")
VOP(z_vmovq_avx, "vmovq (" RC "), %%xmm0")               VOP(z_vshufps_avx, "vshufps $0x1b, %%xmm2, %%xmm1, %%xmm0")
VOP(z_vpshufb_avx, "vpshufb %%xmm2, %%xmm1, %%xmm0")     VOP(z_vsqrtsd_avx, "vsqrtsd %%xmm2, %%xmm1, %%xmm0")
VOP(z_vinsertps_avx, "vinsertps $0x10, %%xmm2, %%xmm1, %%xmm0") VOP(z_vmovhlps_avx, "vmovhlps %%xmm2, %%xmm1, %%xmm0")
VOP(z_vpmovzxbw_avx, "vpmovzxbw %%xmm1, %%xmm0")         VOP(z_vcvtsi2sd_avx, "vcvtsi2sd %%edx, %%xmm1, %%xmm0")
VOP(z_vmovdqu_avx, "vmovdqu (" RC "), %%xmm0")           VOP(z_vpalignr_avx, "vpalignr $4, %%xmm2, %%xmm1, %%xmm0")
VOP(l_addps_avx, "addps %%xmm2, %%xmm0")                 VOP(l_movaps_avx, "movaps %%xmm1, %%xmm0")
VOP(l_pxor_avx, "pxor %%xmm2, %%xmm0")                   VOP(l_movd_avx, "movd %%edx, %%xmm0")
VOP(l_cvtps2dq_avx, "cvtps2dq %%xmm1, %%xmm0")           VOP(l_addss_avx, "addss %%xmm2, %%xmm0")
VOP(l_movssm_avx, "movss (" RC "), %%xmm0")              VOP(l_shufps_avx, "shufps $0x1b, %%xmm2, %%xmm0")
VOP(l_movq_avx, "movq (" RC "), %%xmm0")                 VOP(l_sqrtsd_avx, "sqrtsd %%xmm2, %%xmm0")
VOP(l_movdqu_avx, "movdqu (" RC "), %%xmm0")             VOP(l_cvtsi2sd_avx, "cvtsi2sd %%edx, %%xmm0")
TEST(avx_vex128_upper_zeroing_vs_legacy_sse) {
    NEED(AVX);
    static const struct { const char *n; vop_fn f; int vex; } ops[] = {
        { "vaddps xmm", z_vaddps_avx, 1 }, { "vmovaps xmm", z_vmovaps_avx, 1 }, { "vpxor xmm", z_vpxor_avx, 1 },
        { "vmovd xmm,r32", z_vmovd_avx, 1 }, { "vcvtps2dq xmm", z_vcvtps2dq_avx, 1 }, { "vaddss", z_vaddss_avx, 1 },
        { "vmovss xmm,m32", z_vmovssm_avx, 1 }, { "vmovss xmm,xmm,xmm", z_vmovssr_avx, 1 }, { "vbroadcastss xmm,m32", z_vbcst_avx, 1 },
        { "vpinsrd", z_vpinsrd_avx, 1 }, { "vmovq xmm,m64", z_vmovq_avx, 1 }, { "vshufps xmm", z_vshufps_avx, 1 },
        { "vpshufb xmm", z_vpshufb_avx, 1 }, { "vsqrtsd", z_vsqrtsd_avx, 1 }, { "vinsertps", z_vinsertps_avx, 1 },
        { "vmovhlps", z_vmovhlps_avx, 1 }, { "vpmovzxbw xmm", z_vpmovzxbw_avx, 1 }, { "vcvtsi2sd", z_vcvtsi2sd_avx, 1 },
        { "vmovdqu xmm,m128", z_vmovdqu_avx, 1 }, { "vpalignr xmm", z_vpalignr_avx, 1 },
        { "addps (legacy)", l_addps_avx, 0 }, { "movaps (legacy)", l_movaps_avx, 0 }, { "pxor (legacy)", l_pxor_avx, 0 },
        { "movd (legacy)", l_movd_avx, 0 }, { "cvtps2dq (legacy)", l_cvtps2dq_avx, 0 }, { "addss (legacy)", l_addss_avx, 0 },
        { "movss m32 (legacy)", l_movssm_avx, 0 }, { "shufps (legacy)", l_shufps_avx, 0 }, { "movq m64 (legacy)", l_movq_avx, 0 },
        { "sqrtsd (legacy)", l_sqrtsd_avx, 0 }, { "movdqu m128 (legacy)", l_movdqu_avx, 0 }, { "cvtsi2sd (legacy)", l_cvtsi2sd_avx, 0 },
    };
    static uint8_t mem[64];
    for (int i = 0; i < 64; i++) mem[i] = (uint8_t)(0x40 + i);
    agg_t a; agg_init(&a, "bits 255:128 of the destination: VEX.128 zero, legacy SSE preserved");
    for (size_t k = 0; k < T_COUNTOF(ops); k++) {
        vctx c; vctx_pattern(&c, 0x5EED);
        for (int i = 0; i < 8; i++) { c.r[1].d[i] = 0x3f800000u + (uint32_t)i; c.r[2].d[i] = 0x40000000u + (uint32_t)i; }
        c.mem = (uintptr_t)mem; c.g_in = 0x12345678;
        vctx o = c; ops[k].f(&o);
        char ctx[80]; snprintf(ctx, sizeof ctx, "%s upper 128", ops[k].n);
        agg_mem(&a, &o.r[0].b[16], ops[k].vex ? (const void *)&V_ZERO : (const void *)&c.r[0].b[16], 16, 8, ctx);
    }
    AGG_END(&a);
}

/* VZEROUPPER / VZEROALL on the whole register file, via the harness helper (fn runs between load/store) */
static void vzu_avx(void *p) { (void)p; __asm__ volatile("vzeroupper" ::: "memory"); }
static void vza_avx(void *p) { (void)p; __asm__ volatile("vzeroall" ::: "memory"); }
TEST(avx_vzeroupper_vzeroall_all_registers) {
    NEED(AVX);
    static t_vregs in, out;
    int n = t_vregs_count(T_VK_YMM);
    t_vregs_pattern(&in, T_VK_YMM, 0xA5);
    REQUIRE(t_vregs_call(T_VK_YMM, &in, &out, vzu_avx, NULL) == 0);
    for (int i = 0; i < n; i++) {
        char what[48];
        snprintf(what, sizeof what, "xmm%d low after vzeroupper", i); CHECK_REG(what, out.v[i], in.v[i], 16);
        snprintf(what, sizeof what, "ymm%d high after vzeroupper", i); CHECK_REG(what, out.v[i] + 16, V_ZERO.b, 16);
    }
    REQUIRE(t_vregs_call(T_VK_YMM, &in, &out, vza_avx, NULL) == 0);
    for (int i = 0; i < n; i++) { char what[48]; snprintf(what, sizeof what, "ymm%d after vzeroall", i); CHECK_REG(what, out.v[i], V_ZERO.b, 32); }
}

/* =========================================================================== moves / broadcast / insert / extract */
VOP(mv_vmovups_avx, "vmovups (" RC "), %%ymm0\n\tvmovups %%ymm0, 64(" RC ")")
VOP(mv_vmovaps_avx, "vmovaps (" RC "), %%ymm0\n\tvmovaps %%ymm0, 64(" RC ")")
VOP(mv_vmovdqu_avx, "vmovdqu 1(" RC "), %%ymm0\n\tvmovdqu %%ymm0, 65(" RC ")")
VOP(mv_vlddqu_avx, "vlddqu 3(" RC "), %%ymm0\n\tvmovupd %%ymm0, 67(" RC ")")
VOP(mv_vmovntdq_avx, "vmovdqa (" RC "), %%ymm0\n\tvmovntdq %%ymm0, 64(" RC ")\n\tsfence")
VOP(mv_vmovntps_avx, "vmovapd (" RC "), %%ymm0\n\tvmovntps %%ymm0, 64(" RC ")\n\tsfence")
VOP(mv_vmovsd_st_avx, "vmovsd %%xmm1, 64(" RC ")")       VOP(mv_vmovhps_avx, "vmovhps 8(" RC "), %%xmm1, %%xmm0")
VOP(mv_vmovlpd_avx, "vmovlpd 8(" RC "), %%xmm1, %%xmm0") VOP(mv_vmovddup_avx, "vmovddup %%ymm1, %%ymm0")
VOP(mv_vmovsldup_avx, "vmovsldup %%ymm1, %%ymm0")        VOP(mv_vmovshdup_avx, "vmovshdup %%ymm1, %%ymm0")
TEST(avx_load_store_move_forms) {
    NEED(AVX);
    static T_ALIGN(64) uint8_t buf[160], ref[160];
    agg_t a; agg_init(&a, "vmovups/aps/dqu/lddqu/ntdq/ntps/sd/hps/lpd/ddup/sldup/shdup");
    struct { vop_fn f; int src, dst; const char *n; } rt[] = {
        { mv_vmovups_avx, 0, 64, "vmovups" }, { mv_vmovaps_avx, 0, 64, "vmovaps" }, { mv_vmovdqu_avx, 1, 65, "vmovdqu unaligned" },
        { mv_vlddqu_avx, 3, 67, "vlddqu" }, { mv_vmovntdq_avx, 0, 64, "vmovntdq" }, { mv_vmovntps_avx, 0, 64, "vmovntps" } };
    for (size_t k = 0; k < T_COUNTOF(rt); k++) {
        for (int i = 0; i < 160; i++) buf[i] = (uint8_t)(i * 7 + k);
        memcpy(ref, buf, sizeof ref); memcpy(ref + rt[k].dst, buf + rt[k].src, 32);
        vctx c; vctx_pattern(&c, 1); c.mem = (uintptr_t)buf;
        rt[k].f(&c);
        agg_mem(&a, buf, ref, 160, 8, rt[k].n);           /* exactly 32 bytes written, nothing else */
        agg_mem(&a, &c.r[0], ref + rt[k].dst, 32, 8, rt[k].n);
    }
    for (int i = 0; i < 160; i++) buf[i] = (uint8_t)i;
    vctx c; vctx_pattern(&c, 2); c.mem = (uintptr_t)buf;
    memcpy(ref, buf, sizeof ref); memcpy(ref + 64, &c.r[1], 8);
    vctx o = c; mv_vmovsd_st_avx(&o); agg_mem(&a, buf, ref, 160, 8, "vmovsd m64 store (8 bytes only)");
    v256 w = V_ZERO;
    o = c; mv_vmovhps_avx(&o); memcpy(&w, &c.r[1], 8); memcpy(w.b + 8, buf + 8, 8); agg_mem(&a, &o.r[0], &w, 32, 8, "vmovhps load");
    o = c; mv_vmovlpd_avx(&o); w = V_ZERO; memcpy(&w, buf + 8, 8); memcpy(w.b + 8, c.r[1].b + 8, 8); agg_mem(&a, &o.r[0], &w, 32, 8, "vmovlpd load");
    o = c; mv_vmovddup_avx(&o); w.q[0] = w.q[1] = c.r[1].q[0]; w.q[2] = w.q[3] = c.r[1].q[2]; agg_mem(&a, &o.r[0], &w, 32, 8, "vmovddup ymm (per lane)");
    o = c; mv_vmovsldup_avx(&o); for (int i = 0; i < 8; i++) w.d[i] = c.r[1].d[i & ~1]; agg_mem(&a, &o.r[0], &w, 32, 4, "vmovsldup");
    o = c; mv_vmovshdup_avx(&o); for (int i = 0; i < 8; i++) w.d[i] = c.r[1].d[i | 1]; agg_mem(&a, &o.r[0], &w, 32, 4, "vmovshdup");
    AGG_END(&a);
}

VOP(bc_ss_y_avx, "vbroadcastss (" RC "), %%ymm0")   VOP(bc_ss_x_avx, "vbroadcastss (" RC "), %%xmm0")
VOP(bc_sd_y_avx, "vbroadcastsd (" RC "), %%ymm0")   VOP(bc_f128_avx, "vbroadcastf128 (" RC "), %%ymm0")
TEST(avx_broadcast_memory_sources) {
    NEED(AVX);
    static T_ALIGN(32) uint8_t m[32];
    for (int i = 0; i < 32; i++) m[i] = (uint8_t)(0xA0 + i);
    agg_t a; agg_init(&a, "AVX vbroadcastss/sd/f128 from memory");
    v256 w; vctx c, o;
    vctx_pattern(&c, 3); c.mem = (uintptr_t)m;
    o = c; bc_ss_y_avx(&o); for (int i = 0; i < 8; i++) memcpy(&w.d[i], m, 4); agg_mem(&a, &o.r[0], &w, 32, 4, "vbroadcastss ymm");
    o = c; bc_ss_x_avx(&o); for (int i = 4; i < 8; i++) w.d[i] = 0; agg_mem(&a, &o.r[0], &w, 32, 4, "vbroadcastss xmm (upper zero)");
    o = c; bc_sd_y_avx(&o); for (int i = 0; i < 4; i++) memcpy(&w.q[i], m, 8); agg_mem(&a, &o.r[0], &w, 32, 8, "vbroadcastsd ymm");
    o = c; bc_f128_avx(&o); memcpy(w.b, m, 16); memcpy(w.b + 16, m, 16); agg_mem(&a, &o.r[0], &w, 32, 8, "vbroadcastf128");
    AGG_END(&a);
}

VOP(bc2_ss_r_avx, "vbroadcastss %%xmm1, %%ymm0")     VOP(bc2_sd_r_avx, "vbroadcastsd %%xmm1, %%ymm0")
VOP(bc2_ssx_r_avx, "vbroadcastss %%xmm1, %%xmm0")    VOP(bc2_i128_avx, "vbroadcasti128 (" RC "), %%ymm0")
VOP(bc2_b_y_avx, "vpbroadcastb %%xmm1, %%ymm0")      VOP(bc2_w_y_avx, "vpbroadcastw %%xmm1, %%ymm0")
VOP(bc2_d_y_avx, "vpbroadcastd %%xmm1, %%ymm0")      VOP(bc2_q_y_avx, "vpbroadcastq %%xmm1, %%ymm0")
VOP(bc2_b_x_avx, "vpbroadcastb %%xmm1, %%xmm0")      VOP(bc2_w_x_avx, "vpbroadcastw %%xmm1, %%xmm0")
VOP(bc2_d_x_avx, "vpbroadcastd %%xmm1, %%xmm0")      VOP(bc2_q_x_avx, "vpbroadcastq %%xmm1, %%xmm0")
VOP(bc2_b_m_avx, "vpbroadcastb 5(" RC "), %%ymm0")   VOP(bc2_w_m_avx, "vpbroadcastw 6(" RC "), %%ymm0")
VOP(bc2_d_m_avx, "vpbroadcastd 4(" RC "), %%ymm0")   VOP(bc2_q_m_avx, "vpbroadcastq 8(" RC "), %%ymm0")
TEST(avx2_broadcast_all_sources) {
    NEED(AVX2);
    static T_ALIGN(32) uint8_t m[32];
    for (int i = 0; i < 32; i++) m[i] = (uint8_t)(0x31 * i + 7);
    agg_t a; agg_init(&a, "AVX2 vbroadcastss/sd reg, vbroadcasti128, vpbroadcastb/w/d/q reg+mem, xmm+ymm");
    vctx c; vctx_pattern(&c, 0xBC); c.mem = (uintptr_t)m;
    struct { vop_fn f; int esz, bytes, mem_off; const char *n; } t[] = {
        { bc2_ss_r_avx, 4, 32, -1, "vbroadcastss ymm,xmm" }, { bc2_sd_r_avx, 8, 32, -1, "vbroadcastsd ymm,xmm" },
        { bc2_ssx_r_avx, 4, 16, -1, "vbroadcastss xmm,xmm" }, { bc2_b_y_avx, 1, 32, -1, "vpbroadcastb ymm" },
        { bc2_w_y_avx, 2, 32, -1, "vpbroadcastw ymm" }, { bc2_d_y_avx, 4, 32, -1, "vpbroadcastd ymm" },
        { bc2_q_y_avx, 8, 32, -1, "vpbroadcastq ymm" }, { bc2_b_x_avx, 1, 16, -1, "vpbroadcastb xmm" },
        { bc2_w_x_avx, 2, 16, -1, "vpbroadcastw xmm" }, { bc2_d_x_avx, 4, 16, -1, "vpbroadcastd xmm" },
        { bc2_q_x_avx, 8, 16, -1, "vpbroadcastq xmm" }, { bc2_b_m_avx, 1, 32, 5, "vpbroadcastb m8" },
        { bc2_w_m_avx, 2, 32, 6, "vpbroadcastw m16" }, { bc2_d_m_avx, 4, 32, 4, "vpbroadcastd m32" },
        { bc2_q_m_avx, 8, 32, 8, "vpbroadcastq m64" } };
    for (size_t k = 0; k < T_COUNTOF(t); k++) {
        vctx o = c; vexec(&a, t[k].f, &o, KEEP12);
        v256 w = V_ZERO;
        const uint8_t *src = t[k].mem_off >= 0 ? m + t[k].mem_off : c.r[1].b;
        for (int i = 0; i < t[k].bytes; i += t[k].esz) memcpy(w.b + i, src, (size_t)t[k].esz);
        agg_mem(&a, &o.r[0], &w, 32, t[k].esz, t[k].n);
    }
    vctx o = c; bc2_i128_avx(&o);
    v256 w; memcpy(w.b, m, 16); memcpy(w.b + 16, m, 16);
    agg_mem(&a, &o.r[0], &w, 32, 8, "vbroadcasti128");
    AGG_END(&a);
}

#define vinsf_INSN "vinsertf128 %1, %%xmm2, %%ymm1, %%ymm0"
#define vextf_INSN "vextractf128 %1, %%ymm1, %%xmm0"
#define vinsi_INSN "vinserti128 %1, %%xmm2, %%ymm1, %%ymm0"
#define vexti_INSN "vextracti128 %1, %%ymm1, %%xmm0"
#define vinsfm_INSN "vinsertf128 %1, (" RC "), %%ymm1, %%ymm0"
#define vextfm_INSN "vextractf128 %1, %%ymm1, (" RC ")"
DEF_IMM32(vinsf) DEF_IMM32(vextf) DEF_IMM32(vinsi) DEF_IMM32(vexti) DEF_IMM32(vinsfm) DEF_IMM32(vextfm)
static void ins_ext_check(agg_t *a, int avx2) {
    static T_ALIGN(32) uint8_t m[64];
    for (int imm = 0; imm < 32; imm++) {
        int h = imm & 1;      /* only imm[0] selects the half; other bits ignored */
        vctx c; vctx_pattern(&c, (uint32_t)imm);
        for (int i = 0; i < 64; i++) m[i] = (uint8_t)(imm * 3 + i);
        c.mem = (uintptr_t)m;
        v256 w; vctx o; char ctx[48];
        o = c; vexec(a, (avx2 ? vinsi_tab : vinsf_tab)[imm], &o, KEEP12);
        w = c.r[1]; memcpy(w.b + 16 * h, c.r[2].b, 16);
        snprintf(ctx, sizeof ctx, "vinsert%s128 imm=%d", avx2 ? "i" : "f", imm); agg_mem(a, &o.r[0], &w, 32, 8, ctx);
        o = c; vexec(a, (avx2 ? vexti_tab : vextf_tab)[imm], &o, KEEP12);
        w = V_ZERO; memcpy(w.b, c.r[1].b + 16 * h, 16);
        snprintf(ctx, sizeof ctx, "vextract%s128 imm=%d", avx2 ? "i" : "f", imm); agg_mem(a, &o.r[0], &w, 32, 8, ctx);
        if (!avx2) {
            o = c; vexec(a, vinsfm_tab[imm], &o, KEEP12);
            w = c.r[1]; memcpy(w.b + 16 * h, m, 16);
            agg_mem(a, &o.r[0], &w, 32, 8, "vinsertf128 m128");
            uint8_t ref[64]; memcpy(ref, m, 64); memcpy(ref, c.r[1].b + 16 * h, 16);
            o = c; vexec(a, vextfm_tab[imm], &o, KEEP12);
            agg_mem(a, m, ref, 64, 8, "vextractf128 m128 (16 bytes written)");
        }
    }
}
TEST(avx_insert_extract_f128) { NEED(AVX); agg_t a; agg_init(&a, "vinsertf128/vextractf128 all imm"); ins_ext_check(&a, 0); AGG_END(&a); }
TEST(avx2_insert_extract_i128) { NEED(AVX2); agg_t a; agg_init(&a, "vinserti128/vextracti128 all imm"); ins_ext_check(&a, 1); AGG_END(&a); }


/* =========================================================================== shuffles / permutes */
#define vperm2f128_INSN "vperm2f128 %1, %%ymm2, %%ymm1, %%ymm0"
#define vperm2i128_INSN "vperm2i128 %1, %%ymm2, %%ymm1, %%ymm0"
#define vpermilps_INSN  "vpermilps %1, %%ymm1, %%ymm0"
#define vpermilpd_INSN  "vpermilpd %1, %%ymm1, %%ymm0"
#define vpermq_INSN     "vpermq %1, %%ymm1, %%ymm0"
#define vpermpd_INSN    "vpermpd %1, %%ymm1, %%ymm0"
#define vshufps_INSN    "vshufps %1, %%ymm2, %%ymm1, %%ymm0"
#define vshufpd_INSN    "vshufpd %1, %%ymm2, %%ymm1, %%ymm0"
#define vblendps_INSN   "vblendps %1, %%ymm2, %%ymm1, %%ymm0"
#define vblendpd_INSN   "vblendpd %1, %%ymm2, %%ymm1, %%ymm0"
#define vpblendd_INSN   "vpblendd %1, %%ymm2, %%ymm1, %%ymm0"
#define vpblendw_INSN   "vpblendw %1, %%ymm2, %%ymm1, %%ymm0"
#define vpshufd_INSN    "vpshufd %1, %%ymm1, %%ymm0"
#define vpshufhw_INSN   "vpshufhw %1, %%ymm1, %%ymm0"
#define vpshuflw_INSN   "vpshuflw %1, %%ymm1, %%ymm0"
#define vmpsadbw_INSN   "vmpsadbw %1, %%ymm2, %%ymm1, %%ymm0"
#define vpalignr_INSN   "vpalignr %1, %%ymm2, %%ymm1, %%ymm0"
#define vpslldq_INSN    "vpslldq %1, %%ymm1, %%ymm0"
#define vpsrldq_INSN    "vpsrldq %1, %%ymm1, %%ymm0"
#define vpsllwi_INSN    "vpsllw %1, %%ymm1, %%ymm0"
#define vpsrlwi_INSN    "vpsrlw %1, %%ymm1, %%ymm0"
#define vpsrawi_INSN    "vpsraw %1, %%ymm1, %%ymm0"
#define vpslldi_INSN    "vpslld %1, %%ymm1, %%ymm0"
#define vpsrldi_INSN    "vpsrld %1, %%ymm1, %%ymm0"
#define vpsradi_INSN    "vpsrad %1, %%ymm1, %%ymm0"
#define vpsllqi_INSN    "vpsllq %1, %%ymm1, %%ymm0"
#define vpsrlqi_INSN    "vpsrlq %1, %%ymm1, %%ymm0"
DEF_IMM256(vperm2f128) DEF_IMM256(vperm2i128) DEF_IMM256(vpermilps) DEF_IMM32(vpermilpd) DEF_IMM256(vpermq)
DEF_IMM256(vpermpd) DEF_IMM256(vshufps) DEF_IMM32(vshufpd) DEF_IMM256(vblendps) DEF_IMM32(vblendpd)
DEF_IMM256(vpblendd) DEF_IMM256(vpblendw) DEF_IMM256(vpshufd) DEF_IMM256(vpshufhw) DEF_IMM256(vpshuflw)
DEF_IMM256(vmpsadbw) DEF_IMM48(vpalignr) DEF_IMM32(vpslldq) DEF_IMM32(vpsrldq)
DEF_IMM48(vpsllwi) DEF_IMM48(vpsrlwi) DEF_IMM48(vpsrawi) DEF_IMM48(vpslldi) DEF_IMM48(vpsrldi) DEF_IMM48(vpsradi)
DEF_IMM48(vpsllqi) DEF_IMM48(vpsrlqi)

typedef void (*model_imm)(v256 *d, const v256 *a, const v256 *b, int imm);
static void mi_perm2(v256 *d, const v256 *a, const v256 *b, int imm) {
    for (int h = 0; h < 2; h++) {
        int ctl = (imm >> (4 * h)) & 0xf, sel = ctl & 3;
        const uint8_t *src = (sel < 2 ? a->b : b->b) + 16 * (sel & 1);
        if (ctl & 8) memset(d->b + 16 * h, 0, 16); else memcpy(d->b + 16 * h, src, 16);
    }
}
static void mi_permilps(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; for (int i = 0; i < 8; i++) d->d[i] = a->d[(i & 4) + ((imm >> (2 * (i & 3))) & 3)]; }
static void mi_permilpd(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; for (int i = 0; i < 4; i++) d->q[i] = a->q[(i & 2) + ((imm >> i) & 1)]; }
static void mi_permq(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; for (int i = 0; i < 4; i++) d->q[i] = a->q[(imm >> (2 * i)) & 3]; }
static void mi_shufps(v256 *d, const v256 *a, const v256 *b, int imm) {
    for (int L = 0; L < 2; L++) {
        d->d[4*L+0] = a->d[4*L + (imm & 3)];        d->d[4*L+1] = a->d[4*L + ((imm >> 2) & 3)];
        d->d[4*L+2] = b->d[4*L + ((imm >> 4) & 3)]; d->d[4*L+3] = b->d[4*L + ((imm >> 6) & 3)];
    }
}
static void mi_shufpd(v256 *d, const v256 *a, const v256 *b, int imm) {
    d->q[0] = a->q[imm & 1]; d->q[1] = b->q[(imm >> 1) & 1]; d->q[2] = a->q[2 + ((imm >> 2) & 1)]; d->q[3] = b->q[2 + ((imm >> 3) & 1)];
}
static void mi_blendps(v256 *d, const v256 *a, const v256 *b, int imm) { for (int i = 0; i < 8; i++) d->d[i] = ((imm >> i) & 1) ? b->d[i] : a->d[i]; }
static void mi_blendpd(v256 *d, const v256 *a, const v256 *b, int imm) { for (int i = 0; i < 4; i++) d->q[i] = ((imm >> i) & 1) ? b->q[i] : a->q[i]; }
static void mi_pblendw(v256 *d, const v256 *a, const v256 *b, int imm) { for (int i = 0; i < 16; i++) d->w[i] = ((imm >> (i & 7)) & 1) ? b->w[i] : a->w[i]; }
static void mi_pshufd(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; for (int i = 0; i < 8; i++) d->d[i] = a->d[(i & 4) + ((imm >> (2 * (i & 3))) & 3)]; }
static void mi_pshufhw(v256 *d, const v256 *a, const v256 *b, int imm) {
    (void)b; *d = *a;
    for (int L = 0; L < 2; L++) for (int i = 0; i < 4; i++) d->w[8*L+4+i] = a->w[8*L + 4 + ((imm >> (2 * i)) & 3)];
}
static void mi_pshuflw(v256 *d, const v256 *a, const v256 *b, int imm) {
    (void)b; *d = *a;
    for (int L = 0; L < 2; L++) for (int i = 0; i < 4; i++) d->w[8*L+i] = a->w[8*L + ((imm >> (2 * i)) & 3)];
}
static void mi_mpsadbw(v256 *d, const v256 *a, const v256 *b, int imm) {
    for (int L = 0; L < 2; L++) {
        int ctl = (imm >> (3 * L)) & 7, blk2 = ctl & 3, blk1 = (ctl >> 2) & 1;
        const uint8_t *s1 = a->b + 16 * L + 4 * blk1, *s2 = b->b + 16 * L + 4 * blk2;
        for (int j = 0; j < 8; j++) {
            int sum = 0;
            for (int k = 0; k < 4; k++) sum += abs((int)s1[j + k] - (int)s2[k]);
            d->w[8 * L + j] = (uint16_t)sum;
        }
    }
}
static void mi_palignr(v256 *d, const v256 *a, const v256 *b, int imm) {
    for (int L = 0; L < 2; L++) {
        uint8_t t[32]; memcpy(t, b->b + 16 * L, 16); memcpy(t + 16, a->b + 16 * L, 16);   /* src1 high : src2 low */
        for (int i = 0; i < 16; i++) d->b[16 * L + i] = imm + i < 32 ? t[imm + i] : 0;
    }
}
static void mi_pslldq(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; for (int L = 0; L < 2; L++) for (int i = 0; i < 16; i++) d->b[16*L+i] = i >= imm ? a->b[16*L + i - imm] : 0; }
static void mi_psrldq(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; for (int L = 0; L < 2; L++) for (int i = 0; i < 16; i++) d->b[16*L+i] = i + imm < 16 ? a->b[16*L + i + imm] : 0; }
static void shift_model(v256 *d, const v256 *a, int esz, int kind /*0 sll 1 srl 2 sra*/, uint64_t cnt) {
    int bits = esz * 8;
    for (int i = 0; i < 32 / esz; i++) {
        uint64_t u = lane_u(a, esz, i); int64_t s = lane_s(a, esz, i), r;
        if (kind == 0) r = cnt >= (uint64_t)bits ? 0 : (int64_t)(u << cnt);
        else if (kind == 1) r = cnt >= (uint64_t)bits ? 0 : (int64_t)(u >> cnt);
        else r = s >> (cnt >= (uint64_t)bits ? bits - 1 : (int)cnt);
        lane_set(d, esz, i, (uint64_t)r);
    }
}
static void mi_sllw(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 2, 0, (uint64_t)imm); }
static void mi_srlw(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 2, 1, (uint64_t)imm); }
static void mi_sraw(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 2, 2, (uint64_t)imm); }
static void mi_slld(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 4, 0, (uint64_t)imm); }
static void mi_srld(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 4, 1, (uint64_t)imm); }
static void mi_srad(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 4, 2, (uint64_t)imm); }
static void mi_sllq(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 8, 0, (uint64_t)imm); }
static void mi_srlq(v256 *d, const v256 *a, const v256 *b, int imm) { (void)b; shift_model(d, a, 8, 1, (uint64_t)imm); }

typedef struct immop { const char *name; const vop_fn *tab; int nimm; model_imm m; int lane; } immop;
static void run_immops(const immop *ops, size_t n, int vecs, uint64_t seed) {
    t_rng r; t_rng_seed(&r, seed);
    for (size_t k = 0; k < n; k++) {
        agg_t a; agg_init(&a, "%s (imm 0..%d)", ops[k].name, ops[k].nimm - 1);
        for (int imm = 0; imm < ops[k].nimm; imm++)
            for (int it = 0; it < vecs; it++) {
                vctx c; vctx_pattern(&c, (uint32_t)(imm * 16 + it));
                gen_int(&r, &c.r[1], ops[k].lane); gen_int(&r, &c.r[2], ops[k].lane);
                vctx o = c; vexec(&a, ops[k].tab[imm], &o, KEEP123);
                v256 w; ops[k].m(&w, &c.r[1], &c.r[2], imm);
                char ctx[400], ha[150], hb[150];
                snprintf(ctx, sizeof ctx, "imm=0x%02x a=[%s] b=[%s]", imm, hexv(ha, sizeof ha, &c.r[1], 32, ops[k].lane), hexv(hb, sizeof hb, &c.r[2], 32, ops[k].lane));
                agg_mem(&a, &o.r[0], &w, 32, ops[k].lane, ctx);
            }
        AGG_END(&a);
    }
}
TEST(avx_vperm2f128_vpermil_imm_all) {
    NEED(AVX);
    static const immop ops[] = {
        { "vperm2f128 (cross-lane select/zero)", vperm2f128_tab, 256, mi_perm2, 8 },
        { "vpermilps imm (in-lane)", vpermilps_tab, 256, mi_permilps, 4 },
        { "vpermilpd imm (in-lane, imm[3:0])", vpermilpd_tab, 32, mi_permilpd, 8 } };
    run_immops(ops, T_COUNTOF(ops), 3, 0x9E2);
}
TEST(avx_vshufps_vshufpd_vblend_imm_all) {
    NEED(AVX);
    static const immop ops[] = {
        { "vshufps ymm (in-lane)", vshufps_tab, 256, mi_shufps, 4 }, { "vshufpd ymm (in-lane)", vshufpd_tab, 32, mi_shufpd, 8 },
        { "vblendps ymm", vblendps_tab, 256, mi_blendps, 4 }, { "vblendpd ymm", vblendpd_tab, 32, mi_blendpd, 8 } };
    run_immops(ops, T_COUNTOF(ops), 3, 0x5F5);
}
TEST(avx2_vpermq_vperm2i128_imm_all) {
    NEED(AVX2);
    static const immop ops[] = {
        { "vperm2i128", vperm2i128_tab, 256, mi_perm2, 8 }, { "vpermq (cross-lane)", vpermq_tab, 256, mi_permq, 8 },
        { "vpermpd (cross-lane)", vpermpd_tab, 256, mi_permq, 8 } };
    run_immops(ops, T_COUNTOF(ops), 3, 0x9E3);
}
TEST(avx2_pshuf_blend_imm_all) {
    NEED(AVX2);
    static const immop ops[] = {
        { "vpshufd ymm (in-lane)", vpshufd_tab, 256, mi_pshufd, 4 }, { "vpshufhw ymm", vpshufhw_tab, 256, mi_pshufhw, 2 },
        { "vpshuflw ymm", vpshuflw_tab, 256, mi_pshuflw, 2 }, { "vpblendd ymm", vpblendd_tab, 256, mi_blendps, 4 },
        { "vpblendw ymm (imm8 reused per lane)", vpblendw_tab, 256, mi_pblendw, 2 } };
    run_immops(ops, T_COUNTOF(ops), 3, 0x5F6);
}
TEST(avx2_vpalignr_vpslldq_lane_semantics) {
    NEED(AVX2);
    static const immop ops[] = {
        { "vpalignr ymm (per 128-bit lane, src1:src2)", vpalignr_tab, 48, mi_palignr, 1 },
        { "vpslldq ymm (per lane)", vpslldq_tab, 32, mi_pslldq, 1 }, { "vpsrldq ymm (per lane)", vpsrldq_tab, 32, mi_psrldq, 1 } };
    run_immops(ops, T_COUNTOF(ops), 6, 0xA11);
}
TEST(avx2_vmpsadbw_all_imm) {
    NEED(AVX2);
    static const immop ops[] = { { "vmpsadbw ymm (imm[2:0] lane0, imm[5:3] lane1)", vmpsadbw_tab, 256, mi_mpsadbw, 2 } };
    run_immops(ops, T_COUNTOF(ops), 3, 0x3AD);
}
TEST(avx2_shift_immediate_counts) {
    NEED(AVX2);
    static const immop ops[] = {
        { "vpsllw imm", vpsllwi_tab, 48, mi_sllw, 2 }, { "vpsrlw imm", vpsrlwi_tab, 48, mi_srlw, 2 },
        { "vpsraw imm", vpsrawi_tab, 48, mi_sraw, 2 }, { "vpslld imm", vpslldi_tab, 48, mi_slld, 4 },
        { "vpsrld imm", vpsrldi_tab, 48, mi_srld, 4 }, { "vpsrad imm", vpsradi_tab, 48, mi_srad, 4 },
        { "vpsllq imm", vpsllqi_tab, 48, mi_sllq, 8 }, { "vpsrlq imm", vpsrlqi_tab, 48, mi_srlq, 8 } };
    run_immops(ops, T_COUNTOF(ops), 4, 0x5417);
}

/* ---- variable (register-controlled) permutes */
VOP(vpermilps_v_avx, "vpermilps %%ymm2, %%ymm1, %%ymm0")  VOP(vpermilpd_v_avx, "vpermilpd %%ymm2, %%ymm1, %%ymm0")
VOP(vpermd_avx, "vpermd %%ymm2, %%ymm1, %%ymm0")          VOP(vpermps_avx, "vpermps %%ymm2, %%ymm1, %%ymm0")
VOP(vpshufb_y_avx, "vpshufb %%ymm2, %%ymm1, %%ymm0")
TEST(avx_vpermil_variable_control) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x7E);
    agg_t a; agg_init(&a, "vpermilps/pd variable control (in-lane; pd uses control bit 1)");
    for (int it = 0; it < 500; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it); gen_int(&r, &c.r[1], 4); t_rng_fill(&r, &c.r[2], 32);
        vctx o = c; vexec(&a, vpermilps_v_avx, &o, KEEP12);
        v256 w; for (int i = 0; i < 8; i++) w.d[i] = c.r[1].d[(i & 4) + (c.r[2].d[i] & 3)];
        agg_mem(&a, &o.r[0], &w, 32, 4, "vpermilps");
        o = c; vexec(&a, vpermilpd_v_avx, &o, KEEP12);
        for (int i = 0; i < 4; i++) w.q[i] = c.r[1].q[(i & 2) + ((c.r[2].q[i] >> 1) & 1)];
        agg_mem(&a, &o.r[0], &w, 32, 8, "vpermilpd");
    }
    AGG_END(&a);
}
TEST(avx2_vpermd_vpermps_vpshufb_cross_vs_in_lane) {
    NEED(AVX2);
    t_rng r; t_rng_seed(&r, 0x7F);
    agg_t a; agg_init(&a, "vpermd/vpermps (cross-lane, idx[2:0] from src1) vs vpshufb (in-lane, bit7 zeroes)");
    for (int it = 0; it < 500; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it); t_rng_fill(&r, &c.r[1], 32); gen_int(&r, &c.r[2], 4);
        vctx o = c; vexec(&a, vpermd_avx, &o, KEEP12);
        v256 w; for (int i = 0; i < 8; i++) w.d[i] = c.r[2].d[c.r[1].d[i] & 7];
        agg_mem(&a, &o.r[0], &w, 32, 4, "vpermd");
        o = c; vexec(&a, vpermps_avx, &o, KEEP12);
        agg_mem(&a, &o.r[0], &w, 32, 4, "vpermps");
        o = c; vexec(&a, vpshufb_y_avx, &o, KEEP12);
        for (int i = 0; i < 32; i++) { uint8_t k = c.r[2].b[i]; w.b[i] = (k & 0x80) ? 0 : c.r[1].b[(i & 16) + (k & 15)]; }
        agg_mem(&a, &o.r[0], &w, 32, 1, "vpshufb");
    }
    /* explicit lane-crossing contrast */
    vctx c; vctx_pattern(&c, 1);
    for (int i = 0; i < 8; i++) { c.r[1].d[i] = (uint32_t)(7 - i); c.r[2].d[i] = 0x100u + (uint32_t)i; }
    vctx o = c; vpermd_avx(&o);
    v256 w; for (int i = 0; i < 8; i++) w.d[i] = 0x107u - (uint32_t)i;
    agg_mem(&a, &o.r[0], &w, 32, 4, "vpermd full reverse crosses lanes");
    o = c; o.r[1] = c.r[2]; vpshufd_0_0_avx(&o);
    vctx o2 = c; o2.r[1] = c.r[2]; vpshufd_1_11_avx(&o2);   /* imm 0x00, 0x1b */
    for (int i = 0; i < 8; i++) w.d[i] = 0x100u + (uint32_t)(i & 4);
    agg_mem(&a, &o.r[0], &w, 32, 4, "vpshufd imm=0 broadcasts element 0 of EACH lane");
    for (int i = 0; i < 8; i++) w.d[i] = 0x100u + (uint32_t)((i & 4) + 3 - (i & 3));
    agg_mem(&a, &o2.r[0], &w, 32, 4, "vpshufd imm=0x1b reverses within each lane only");
    AGG_END(&a);
}

VOP(vunpcklps_avx, "vunpcklps %%ymm2, %%ymm1, %%ymm0")    VOP(vunpckhps_avx, "vunpckhps %%ymm2, %%ymm1, %%ymm0")
VOP(vunpcklpd_avx, "vunpcklpd %%ymm2, %%ymm1, %%ymm0")    VOP(vunpckhpd_avx, "vunpckhpd %%ymm2, %%ymm1, %%ymm0")
VOP(vpunpcklbw_avx, "vpunpcklbw %%ymm2, %%ymm1, %%ymm0")  VOP(vpunpckhbw_avx, "vpunpckhbw %%ymm2, %%ymm1, %%ymm0")
VOP(vpunpcklwd_avx, "vpunpcklwd %%ymm2, %%ymm1, %%ymm0")  VOP(vpunpckhwd_avx, "vpunpckhwd %%ymm2, %%ymm1, %%ymm0")
VOP(vpunpckldq_avx, "vpunpckldq %%ymm2, %%ymm1, %%ymm0")  VOP(vpunpckhdq_avx, "vpunpckhdq %%ymm2, %%ymm1, %%ymm0")
VOP(vpunpcklqdq_avx, "vpunpcklqdq %%ymm2, %%ymm1, %%ymm0") VOP(vpunpckhqdq_avx, "vpunpckhqdq %%ymm2, %%ymm1, %%ymm0")
static void m_unpack(v256 *d, const v256 *a, const v256 *b, int esz, int hi) {
    int n = 16 / esz;     /* elements per lane */
    for (int L = 0; L < 2; L++)
        for (int i = 0; i < n / 2; i++) {
            int src = L * n + (hi ? n / 2 : 0) + i;
            memcpy(d->b + 16 * L + (2 * i) * esz, a->b + src * esz, (size_t)esz);
            memcpy(d->b + 16 * L + (2 * i + 1) * esz, b->b + src * esz, (size_t)esz);
        }
}
static void unpack_cases(int avx2) {
    t_rng r; t_rng_seed(&r, 0x0B + (uint64_t)avx2);
    static const struct { vop_fn f; int esz, hi; const char *n; } t[] = {
        { vunpcklps_avx, 4, 0, "vunpcklps" }, { vunpckhps_avx, 4, 1, "vunpckhps" }, { vunpcklpd_avx, 8, 0, "vunpcklpd" },
        { vunpckhpd_avx, 8, 1, "vunpckhpd" }, { vpunpcklbw_avx, 1, 0, "vpunpcklbw" }, { vpunpckhbw_avx, 1, 1, "vpunpckhbw" },
        { vpunpcklwd_avx, 2, 0, "vpunpcklwd" }, { vpunpckhwd_avx, 2, 1, "vpunpckhwd" }, { vpunpckldq_avx, 4, 0, "vpunpckldq" },
        { vpunpckhdq_avx, 4, 1, "vpunpckhdq" }, { vpunpcklqdq_avx, 8, 0, "vpunpcklqdq" }, { vpunpckhqdq_avx, 8, 1, "vpunpckhqdq" } };
    for (size_t k = avx2 ? 4 : 0; k < (avx2 ? T_COUNTOF(t) : 4); k++) {
        agg_t a; agg_init(&a, "%s ymm (in-lane interleave)", t[k].n);
        for (int it = 0; it < 100; it++) {
            vctx c; vctx_pattern(&c, (uint32_t)it); t_rng_fill(&r, &c.r[1], 32); t_rng_fill(&r, &c.r[2], 32);
            vctx o = c; vexec(&a, t[k].f, &o, KEEP12);
            v256 w; m_unpack(&w, &c.r[1], &c.r[2], t[k].esz, t[k].hi);
            agg_mem(&a, &o.r[0], &w, 32, t[k].esz, t[k].n);
        }
        AGG_END(&a);
    }
}
TEST(avx_unpck_ps_pd_in_lane) { NEED(AVX); unpack_cases(0); }
TEST(avx2_punpck_in_lane) { NEED(AVX2); unpack_cases(1); }

VOP(vblendvps_avx, "vblendvps %%ymm3, %%ymm2, %%ymm1, %%ymm0")  VOP(vblendvpd_avx, "vblendvpd %%ymm3, %%ymm2, %%ymm1, %%ymm0")
VOP(vpblendvb_avx, "vpblendvb %%ymm3, %%ymm2, %%ymm1, %%ymm0")
TEST(avx_variable_blends_sign_bit) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0xB1E);
    agg_t a; agg_init(&a, "vblendvps/vblendvpd (mask sign bit)%s", cpu_has(F_AVX2) ? " + vpblendvb" : "");
    for (int it = 0; it < 400; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        t_rng_fill(&r, &c.r[1], 32); t_rng_fill(&r, &c.r[2], 32); t_rng_fill(&r, &c.r[3], 32);
        vctx o = c; vexec(&a, vblendvps_avx, &o, KEEP123);
        v256 w; for (int i = 0; i < 8; i++) w.d[i] = (c.r[3].d[i] >> 31) ? c.r[2].d[i] : c.r[1].d[i];
        agg_mem(&a, &o.r[0], &w, 32, 4, "vblendvps");
        o = c; vexec(&a, vblendvpd_avx, &o, KEEP123);
        for (int i = 0; i < 4; i++) w.q[i] = (c.r[3].q[i] >> 63) ? c.r[2].q[i] : c.r[1].q[i];
        agg_mem(&a, &o.r[0], &w, 32, 8, "vblendvpd");
        if (cpu_has(F_AVX2)) {
            o = c; vexec(&a, vpblendvb_avx, &o, KEEP123);
            for (int i = 0; i < 32; i++) w.b[i] = (c.r[3].b[i] & 0x80) ? c.r[2].b[i] : c.r[1].b[i];
            agg_mem(&a, &o.r[0], &w, 32, 1, "vpblendvb");
        }
    }
    AGG_END(&a);
}

VOP(vtestps_avx, "vtestps %%ymm2, %%ymm1")  VOP(vtestpd_avx, "vtestpd %%ymm2, %%ymm1")  VOP(vptest_avx, "vptest %%ymm2, %%ymm1")
VOP(vtestps_x_avx, "vtestps %%xmm2, %%xmm1")
VOP(vmovmskps_avx, "vmovmskps %%ymm1, %%edx")  VOP(vmovmskpd_avx, "vmovmskpd %%ymm1, %%edx")  VOP(vpmovmskb_avx, "vpmovmskb %%ymm1, %%edx")
TEST(avx_vtest_vptest_flags_and_movmsk) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x7E57);
    agg_t a; agg_init(&a, "vtestps/pd/vptest ZF,CF (others cleared); vmovmskps/pd%s", cpu_has(F_AVX2) ? ", vpmovmskb" : "");
    const unsigned FM = EF_ZF | EF_CF | EF_OF | EF_SF | EF_AF | EF_PF;
    for (int it = 0; it < 600; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        t_rng_fill(&r, &c.r[1], 32); t_rng_fill(&r, &c.r[2], 32);
        int mode = it % 4;      /* force ZF / CF cases */
        if (mode == 1) for (int i = 0; i < 4; i++) c.r[2].q[i] &= ~c.r[1].q[i];
        if (mode == 2) for (int i = 0; i < 4; i++) c.r[2].q[i] &= c.r[1].q[i];
        if (mode == 3) c.r[2] = V_ZERO;
        unsigned zps = EF_ZF, cps = EF_CF, zpd = EF_ZF, cpd = EF_CF, zpt = EF_ZF, cpt = EF_CF, zx = EF_ZF, cx = EF_CF;
        for (int i = 0; i < 8; i++) {
            if ((c.r[1].d[i] & c.r[2].d[i]) >> 31) { zps = 0; if (i < 4) zx = 0; }
            if ((~c.r[1].d[i] & c.r[2].d[i]) >> 31) { cps = 0; if (i < 4) cx = 0; }
        }
        for (int i = 0; i < 4; i++) {
            if ((c.r[1].q[i] & c.r[2].q[i]) >> 63) zpd = 0;
            if ((~c.r[1].q[i] & c.r[2].q[i]) >> 63) cpd = 0;
            if (c.r[1].q[i] & c.r[2].q[i]) zpt = 0;
            if (~c.r[1].q[i] & c.r[2].q[i]) cpt = 0;
        }
        vctx o = c; vexec(&a, vtestps_avx, &o, KEEP12);   agg_eq(&a, o.fl_out & FM, zps | cps, "vtestps ymm");
        o = c; vexec(&a, vtestpd_avx, &o, KEEP12);        agg_eq(&a, o.fl_out & FM, zpd | cpd, "vtestpd ymm");
        o = c; vexec(&a, vptest_avx, &o, KEEP12);         agg_eq(&a, o.fl_out & FM, zpt | cpt, "vptest ymm");
        o = c; vexec(&a, vtestps_x_avx, &o, KEEP12);      agg_eq(&a, o.fl_out & FM, zx | cx, "vtestps xmm");
        agg_mem(&a, &o.r[0], &c.r[0], 32, 8, "vtest does not write a register");
        unsigned mps = 0, mpd = 0, mb = 0;
        for (int i = 0; i < 8; i++) mps |= (c.r[1].d[i] >> 31) << i;
        for (int i = 0; i < 4; i++) mpd |= (unsigned)(c.r[1].q[i] >> 63) << i;
        for (int i = 0; i < 32; i++) mb |= (unsigned)(c.r[1].b[i] >> 7) << i;
        o = c; vexec(&a, vmovmskps_avx, &o, KEEP12); agg_eq(&a, o.g_out, mps, "vmovmskps ymm");
        o = c; vexec(&a, vmovmskpd_avx, &o, KEEP12); agg_eq(&a, o.g_out, mpd, "vmovmskpd ymm");
        if (cpu_has(F_AVX2)) { o = c; vexec(&a, vpmovmskb_avx, &o, KEEP12); agg_eq(&a, o.g_out, mb, "vpmovmskb ymm"); }
    }
    AGG_END(&a);
}

/* =========================================================================== masked moves */
VOP(mm_ldps_avx, "vmaskmovps (" RC "), %%ymm3, %%ymm0")    VOP(mm_ldpd_avx, "vmaskmovpd (" RC "), %%ymm3, %%ymm0")
VOP(mm_stps_avx, "vmaskmovps %%ymm1, %%ymm3, (" RC ")")    VOP(mm_stpd_avx, "vmaskmovpd %%ymm1, %%ymm3, (" RC ")")
VOP(mm_ldd_avx, "vpmaskmovd (" RC "), %%ymm3, %%ymm0")     VOP(mm_ldq_avx, "vpmaskmovq (" RC "), %%ymm3, %%ymm0")
VOP(mm_std_avx, "vpmaskmovd %%ymm1, %%ymm3, (" RC ")")     VOP(mm_stq_avx, "vpmaskmovq %%ymm1, %%ymm3, (" RC ")")
VOP(mm_ldps_x_avx, "vmaskmovps (" RC "), %%xmm3, %%xmm0")  VOP(mm_stps_x_avx, "vmaskmovps %%xmm1, %%xmm3, (" RC ")")
typedef struct mmop { vop_fn f; int esz, store, bytes, avx2; const char *n; } mmop;
static const mmop mm_ops[] = {
    { mm_ldps_avx, 4, 0, 32, 0, "vmaskmovps load" }, { mm_ldpd_avx, 8, 0, 32, 0, "vmaskmovpd load" },
    { mm_stps_avx, 4, 1, 32, 0, "vmaskmovps store" }, { mm_stpd_avx, 8, 1, 32, 0, "vmaskmovpd store" },
    { mm_ldps_x_avx, 4, 0, 16, 0, "vmaskmovps xmm load" }, { mm_stps_x_avx, 4, 1, 16, 0, "vmaskmovps xmm store" },
    { mm_ldd_avx, 4, 0, 32, 1, "vpmaskmovd load" }, { mm_ldq_avx, 8, 0, 32, 1, "vpmaskmovq load" },
    { mm_std_avx, 4, 1, 32, 1, "vpmaskmovd store" }, { mm_stq_avx, 8, 1, 32, 1, "vpmaskmovq store" } };
TEST(avx_maskmov_semantics) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x3A5C);
    static uint8_t mem[96], ref[96];
    for (size_t k = 0; k < T_COUNTOF(mm_ops); k++) {
        if (mm_ops[k].avx2 && !cpu_has(F_AVX2)) { t_note("%s: AVX2 not usable, skipped", mm_ops[k].n); continue; }
        agg_t a; agg_init(&a, "%s (masked lanes: zero on load, untouched on store)", mm_ops[k].n);
        for (int it = 0; it < 200; it++) {
            int off = (int)t_rng_range(&r, 32);
            t_rng_fill(&r, mem, sizeof mem); memcpy(ref, mem, sizeof ref);
            vctx c; vctx_pattern(&c, (uint32_t)it); c.mem = (uintptr_t)(mem + off);
            t_rng_fill(&r, &c.r[1], 32); t_rng_fill(&r, &c.r[3], 32);
            if (it == 0) c.r[3] = V_ZERO;
            vctx o = c; vexec(&a, mm_ops[k].f, &o, KEEP123);
            int n = mm_ops[k].bytes / mm_ops[k].esz, es = mm_ops[k].esz;
            if (mm_ops[k].store) {
                for (int i = 0; i < n; i++) if (c.r[3].b[i * es + es - 1] & 0x80) memcpy(ref + off + i * es, c.r[1].b + i * es, (size_t)es);
                agg_mem(&a, mem, ref, sizeof mem, 8, "memory after masked store");
                agg_mem(&a, &o.r[0], &c.r[0], 32, 8, "store leaves ymm0");
            } else {
                v256 w = V_ZERO;
                for (int i = 0; i < n; i++) if (c.r[3].b[i * es + es - 1] & 0x80) memcpy(w.b + i * es, mem + off + i * es, (size_t)es);
                agg_mem(&a, &o.r[0], &w, 32, es, "masked load");
                agg_mem(&a, mem, ref, sizeof mem, 8, "load does not write memory");
            }
        }
        AGG_END(&a);
    }
}

/* =========================================================================== gathers */
VOP(g_dps1_avx, "vgatherdps %%ymm3, (" RC ",%%ymm2,1), %%ymm0")    VOP(g_dps2_avx, "vgatherdps %%ymm3, (" RC ",%%ymm2,2), %%ymm0")
VOP(g_dps4_avx, "vgatherdps %%ymm3, (" RC ",%%ymm2,4), %%ymm0")    VOP(g_dps8_avx, "vgatherdps %%ymm3, (" RC ",%%ymm2,8), %%ymm0")
VOP(g_dps4d_avx, "vgatherdps %%ymm3, 12(" RC ",%%ymm2,4), %%ymm0") VOP(g_dps_x_avx, "vgatherdps %%xmm3, (" RC ",%%xmm2,4), %%xmm0")
VOP(g_qps_avx, "vgatherqps %%xmm3, (" RC ",%%ymm2,4), %%xmm0")     VOP(g_dpd_avx, "vgatherdpd %%ymm3, (" RC ",%%xmm2,8), %%ymm0")
VOP(g_qpd_avx, "vgatherqpd %%ymm3, (" RC ",%%ymm2,8), %%ymm0")     VOP(g_dd_avx, "vpgatherdd %%ymm3, (" RC ",%%ymm2,4), %%ymm0")
VOP(g_dq_avx, "vpgatherdq %%ymm3, (" RC ",%%xmm2,8), %%ymm0")      VOP(g_qd_avx, "vpgatherqd %%xmm3, (" RC ",%%ymm2,4), %%xmm0")
VOP(g_qq_avx, "vpgatherqq %%ymm3, (" RC ",%%ymm2,8), %%ymm0")      VOP(g_qq_x_avx, "vpgatherqq %%xmm3, (" RC ",%%xmm2,8), %%xmm0")
VOP(g_qd_dispneg_avx, "vpgatherqd %%xmm3, -8(" RC ",%%ymm2,2), %%xmm0")
static T_ALIGN(32) uint8_t g_gather_tab[4096];
typedef struct gop { vop_fn f; int es, is, n, scale, disp, dbytes; const char *name; } gop;
static const gop g_ops[] = {
    { g_dps1_avx, 4, 4, 8, 1, 0, 32, "vgatherdps ymm scale1" }, { g_dps2_avx, 4, 4, 8, 2, 0, 32, "vgatherdps ymm scale2" },
    { g_dps4_avx, 4, 4, 8, 4, 0, 32, "vgatherdps ymm scale4" }, { g_dps8_avx, 4, 4, 8, 8, 0, 32, "vgatherdps ymm scale8" },
    { g_dps4d_avx, 4, 4, 8, 4, 12, 32, "vgatherdps ymm disp12" }, { g_dps_x_avx, 4, 4, 4, 4, 0, 16, "vgatherdps xmm" },
    { g_qps_avx, 4, 8, 4, 4, 0, 16, "vgatherqps xmm,ymm-index" }, { g_dpd_avx, 8, 4, 4, 8, 0, 32, "vgatherdpd ymm,xmm-index" },
    { g_qpd_avx, 8, 8, 4, 8, 0, 32, "vgatherqpd ymm" }, { g_dd_avx, 4, 4, 8, 4, 0, 32, "vpgatherdd ymm" },
    { g_dq_avx, 8, 4, 4, 8, 0, 32, "vpgatherdq ymm" }, { g_qd_avx, 4, 8, 4, 4, 0, 16, "vpgatherqd xmm" },
    { g_qq_avx, 8, 8, 4, 8, 0, 32, "vpgatherqq ymm" }, { g_qq_x_avx, 8, 8, 2, 8, 0, 16, "vpgatherqq xmm" },
    { g_qd_dispneg_avx, 4, 8, 4, 2, -8, 16, "vpgatherqd xmm scale2 disp-8" } };
TEST(avx2_gather_semantics_mask_cleared) {
    NEED(AVX2);
    /* SDM VGATHERDPS etc: element i loaded from base+disp+sext(index[i])*scale iff the sign bit of mask element i
     * is set, otherwise the destination element is kept; on completion the whole mask register is zero;
     * VEX.128 forms (and the xmm-destination qword-index forms) zero bits 255:128 of dest and mask. */
    t_rng r; t_rng_seed(&r, 0x6A7);
    for (int i = 0; i < 4096; i++) g_gather_tab[i] = (uint8_t)t_rng_u32(&r);
    uint8_t *base = g_gather_tab + 2048;
    for (size_t k = 0; k < T_COUNTOF(g_ops); k++) {
        const gop *g = &g_ops[k];
        agg_t a; agg_init(&a, "%s", g->name);
        for (int it = 0; it < 150; it++) {
            vctx c; vctx_pattern(&c, (uint32_t)it); c.mem = (uintptr_t)base;
            c.r[2] = V_ZERO;
            for (int i = 0; i < g->n; i++) {
                int64_t off = (int64_t)t_rng_range(&r, 4096 - 16) - 2048 - g->disp;
                int64_t idx = off / g->scale;
                if (g->is == 4) c.r[2].sd[i] = (int32_t)idx; else c.r[2].sq[i] = idx;
            }
            if (g->is == 4 && g->n == 4) t_rng_fill(&r, &c.r[2].b[16], 16);     /* xmm index: upper ignored */
            t_rng_fill(&r, &c.r[3], 32);
            if (it == 0) for (int i = 0; i < 32; i++) c.r[3].b[i] |= 0x80;    /* all set */
            if (it == 1) c.r[3] = V_ZERO;                                     /* none set */
            vctx o = c; vexec(&a, g->f, &o, 0x6u);    /* ymm1 untouched, index untouched */
            v256 w = V_ZERO; memcpy(&w, &c.r[0], (size_t)g->dbytes);
            for (int i = 0; i < g->n; i++) {
                if (!(c.r[3].b[i * g->es + g->es - 1] & 0x80)) continue;
                int64_t idx = g->is == 4 ? (int64_t)c.r[2].sd[i] : c.r[2].sq[i];
                memcpy(w.b + i * g->es, base + g->disp + idx * g->scale, (size_t)g->es);
            }
            char ctx[48]; snprintf(ctx, sizeof ctx, "vec %d dest", it);
            agg_mem(&a, &o.r[0], &w, 32, g->es, ctx);
            snprintf(ctx, sizeof ctx, "vec %d mask register zero", it);
            agg_mem(&a, &o.r[3], &V_ZERO, 32, g->es, ctx);
        }
        AGG_END(&a);
    }
}

/* ---- deliberate / possible faults: children */
static volatile LONG g_veh_count;
static ULONG_PTR g_veh_addr;
static DWORD g_veh_code;
static uint8_t g_veh_xmm0[16], g_veh_xmm3[16];
static int g_veh_ctx_ok;
static uint32_t g_veh_mxcsr;
static void *g_veh_page;
static size_t g_veh_page_size;
static int ctx_get_vec(CONTEXT *cx, int reg, uint8_t out[16]) {
#ifdef T_X64
    memcpy(out, (const uint8_t *)&cx->FltSave.XmmRegisters[reg], 16); return 1;
#else
    if ((cx->ContextFlags & CONTEXT_EXTENDED_REGISTERS) != CONTEXT_EXTENDED_REGISTERS) return 0;
    memcpy(out, cx->ExtendedRegisters + 160 + 16 * reg, 16); return 1;
#endif
}
static int ctx_get_mxcsr(CONTEXT *cx, uint32_t *v) {
#ifdef T_X64
    *v = cx->MxCsr; return 1;
#else
    if ((cx->ContextFlags & CONTEXT_EXTENDED_REGISTERS) != CONTEXT_EXTENDED_REGISTERS) return 0;
    memcpy(v, cx->ExtendedRegisters + 24, 4); return 1;
#endif
}
static void ctx_set_mxcsr(CONTEXT *cx, uint32_t v) {
#ifdef T_X64
    cx->MxCsr = v; cx->FltSave.MxCsr = v;     /* both copies: the kernel restores MXCSR from the CONTEXT */
#else
    memcpy(cx->ExtendedRegisters + 24, &v, 4);
#endif
}
static LONG CALLBACK veh_page_fault(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != STATUS_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    if (InterlockedIncrement(&g_veh_count) > 3) return EXCEPTION_CONTINUE_SEARCH;
    g_veh_code = ep->ExceptionRecord->ExceptionCode;
    g_veh_addr = ep->ExceptionRecord->NumberParameters >= 2 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
    g_veh_ctx_ok = ctx_get_vec(ep->ContextRecord, 3, g_veh_xmm3) && ctx_get_vec(ep->ContextRecord, 0, g_veh_xmm0);
    DWORD old;
    if (!VirtualProtect(g_veh_page, g_veh_page_size, PAGE_READWRITE, &old)) return EXCEPTION_CONTINUE_SEARCH;
    return EXCEPTION_CONTINUE_EXECUTION;
}
static size_t page_size(void) { SYSTEM_INFO si; GetSystemInfo(&si); return si.dwPageSize ? si.dwPageSize : 4096; }

/* masked-off elements on a PAGE_NOACCESS page must not fault (Intel SDM; AMD: maskmov implementation dependent) */
CHILD(avx_fault_suppress) {
    if (argc < 2) return 3;
    const char *w = argv[1];
    void *base = NULL;
    uint8_t *p = (uint8_t *)t_guarded_alloc(64, &base);     /* p[0..63] readable, p[64..] PAGE_NOACCESS */
    if (!p) { printf("NOALLOC=1\n"); return 4; }
    for (int i = 0; i < 64; i++) p[i] = (uint8_t)(i + 1);
    vctx c; vctx_pattern(&c, 9);
    int ok = 1;
    printf("STEP=%s\n", w); fflush(stdout);
    if (!strcmp(w, "maskload") || !strcmp(w, "pmaskload") || !strcmp(w, "maskstore") || !strcmp(w, "pmaskstore")) {
        int store = strstr(w, "store") != NULL, pi = w[0] == 'p';
        c.mem = (uintptr_t)(p + 48);                       /* 16 bytes valid, 16 bytes in the guard page */
        c.r[3] = V_ZERO; for (int i = 0; i < 4; i++) c.r[3].d[i] = 0x80000000u;
        for (int i = 0; i < 8; i++) c.r[1].d[i] = 0xC0DE0000u + (uint32_t)i;
        if (!store) (pi ? mm_ldd_avx : mm_ldps_avx)(&c); else (pi ? mm_std_avx : mm_stps_avx)(&c);
        if (!store) {
            for (int i = 0; i < 4; i++) { uint32_t v; memcpy(&v, p + 48 + 4 * i, 4); ok &= c.r[0].d[i] == v; }
            for (int i = 4; i < 8; i++) ok &= c.r[0].d[i] == 0;
        } else for (int i = 0; i < 4; i++) { uint32_t v; memcpy(&v, p + 48 + 4 * i, 4); ok &= v == 0xC0DE0000u + (uint32_t)i; }
    } else if (!strcmp(w, "maskzero")) {
        c.mem = (uintptr_t)(p + 64);                       /* entirely inside the guard page, mask all zero */
        c.r[3] = V_ZERO;
        mm_ldps_avx(&c); ok &= !memcmp(&c.r[0], &V_ZERO, 32);
        mm_stps_avx(&c);
    } else if (!strcmp(w, "gather")) {
        c.mem = (uintptr_t)p;
        vctx in;
        for (int i = 0; i < 8; i++) { c.r[2].d[i] = (uint32_t)i; c.r[3].d[i] = 0x80000000u; }
        c.r[2].d[5] = 16 + 2; c.r[3].d[5] = 0;                 /* element 5 -> guard page (p+72), masked off */
        c.r[2].d[7] = 0x10000000; c.r[3].d[7] = 0x7fffffffu;   /* 1 GiB away, sign bit clear -> not accessed */
        in = c;
        g_dps4_avx(&c);
        for (int i = 0; i < 8; i++) {
            uint32_t want; if (i == 5 || i == 7) want = in.r[0].d[i]; else memcpy(&want, p + 4 * i, 4);
            ok &= c.r[0].d[i] == want;
        }
        ok &= !memcmp(&c.r[3], &V_ZERO, 32);
    } else return 5;
    printf("DATA_OK=%d\n", ok);
    t_guarded_free(base);
    return 0;
}
TEST(avx_masked_lanes_never_fault) {
    NEED(AVX);
    static const char *const steps[] = { "maskload", "maskstore", "maskzero", "pmaskload", "pmaskstore", "gather" };
    int intel = cpu()->is_intel;
    for (size_t i = 0; i < T_COUNTOF(steps); i++) {
        int need2 = steps[i][0] == 'p' || steps[i][0] == 'g';
        if (need2 && !cpu_has(F_AVX2)) { t_note("%s: AVX2 not usable", steps[i]); continue; }
        static t_child_result r; char v[32] = "";
        REQUIRE(t_child_run("avx_fault_suppress", steps[i], 10000, &r));
        CHECKF(!r.timed_out, "%s: child timed out", steps[i]);
        int gather = steps[i][0] == 'g';
        if (r.exc_code && !gather && !intel) {
            t_note("%s: masked-off element faulted (exception 0x%08lx); AMD APM allows implementation-dependent faults for masked-off VMASKMOV elements", steps[i], (unsigned long)r.exc_code);
            continue;
        }
        CHECKF(r.exc_code == 0 && r.exit_code == 0, "%s: masked-off lanes on a PAGE_NOACCESS page faulted: exit 0x%lx exc 0x%lx", steps[i], (unsigned long)r.exit_code, (unsigned long)r.exc_code);
        CHECKF(t_child_find(&r, "DATA_OK=", v, sizeof v) && !strcmp(v, "1"), "%s: data wrong (DATA_OK=%s)", steps[i], v);
    }
}

/* a page fault on an ACTIVE element: delivered as STATUS_ACCESS_VIOLATION at the instruction, restartable.
 * gather: elements to the right of the faulting one are complete (mask bits cleared) at fault time. */
CHILD(avx_fault_restart) {
    if (argc < 2) return 3;
    const char *w = argv[1];
    void *base = NULL;
    uint8_t *p = (uint8_t *)t_guarded_alloc(64, &base);
    if (!p) return 4;
    for (int i = 0; i < 64; i++) p[i] = (uint8_t)(0x11 * (i + 1));
    g_veh_page = p + 64; g_veh_page_size = page_size();
    PVOID h = AddVectoredExceptionHandler(1, veh_page_fault);
    if (!h) return 6;
    vctx c, in; vctx_pattern(&c, 0x6F);
    int ok = 1;
    if (!strcmp(w, "gather")) {
        c.mem = (uintptr_t)p;
        for (int i = 0; i < 8; i++) { c.r[2].d[i] = (uint32_t)i; c.r[3].d[i] = 0xFFFFFFFFu; }
        c.r[2].d[2] = 18;                                      /* element 2 -> p+72: guard page */
        in = c;
        g_dps4_avx(&c);
        for (int i = 0; i < 8; i++) { uint32_t want = 0; if (i != 2) memcpy(&want, p + 4 * i, 4); ok &= c.r[0].d[i] == want; }
        printf("MASK_ZERO=%d\n", !memcmp(&c.r[3], &V_ZERO, 32));
        printf("FAULT_OFF=%ld\n", (long)((intptr_t)g_veh_addr - (intptr_t)p));
        if (g_veh_ctx_ok) {
            uint32_t m[4]; memcpy(m, g_veh_xmm3, 16);
            printf("CTX_MASK=%08x,%08x,%08x,%08x\n", m[0], m[1], m[2], m[3]);
        } else printf("CTX_MASK=unavailable\n");
    } else if (!strcmp(w, "maskload")) {
        c.mem = (uintptr_t)(p + 48);
        for (int i = 0; i < 8; i++) c.r[3].d[i] = i >= 4 ? 0x80000000u : 0;   /* only the guard-page half active */
        in = c;
        mm_ldps_avx(&c);
        for (int i = 0; i < 8; i++) ok &= c.r[0].d[i] == 0;                  /* active lanes read the fresh zero page */
        printf("FAULT_OFF=%ld\n", (long)((intptr_t)g_veh_addr - (intptr_t)p));
    } else if (!strcmp(w, "maskstore")) {
        c.mem = (uintptr_t)(p + 48);
        for (int i = 0; i < 8; i++) { c.r[3].d[i] = (i == 1 || i == 6) ? 0x80000000u : 0; c.r[1].d[i] = 0xAB000000u + (uint32_t)i; }
        in = c;
        mm_stps_avx(&c);
        uint32_t v1, v6; memcpy(&v1, p + 52, 4); memcpy(&v6, p + 72, 4);
        ok &= v1 == 0xAB000001u && v6 == 0xAB000006u && p[48] == 0x11 * 49 % 256;
        printf("FAULT_OFF=%ld\n", (long)((intptr_t)g_veh_addr - (intptr_t)p));
    } else return 5;
    (void)in;
    RemoveVectoredExceptionHandler(h);
    printf("FAULTS=%ld\nCODE=0x%08lx\nDATA_OK=%d\n", (long)g_veh_count, (unsigned long)g_veh_code, ok);
    return 0;
}
TEST(avx_fault_on_active_element_restarts) {
    NEED(AVX);
    static const char *const steps[] = { "maskload", "maskstore", "gather" };
    for (size_t i = 0; i < T_COUNTOF(steps); i++) {
        if (steps[i][0] == 'g' && !cpu_has(F_AVX2)) { t_note("gather: AVX2 not usable"); continue; }
        static t_child_result r; char v[96] = "";
        REQUIRE(t_child_run("avx_fault_restart", steps[i], 10000, &r));
        CHECKF(!r.timed_out && r.exit_code == 0, "%s: child exit 0x%lx exc 0x%lx timed_out %d", steps[i], (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.timed_out);
        CHECKF(t_child_find(&r, "FAULTS=", v, sizeof v) && !strcmp(v, "1"), "%s: exactly one access violation expected, got %s", steps[i], v);
        CHECKF(t_child_find(&r, "CODE=", v, sizeof v) && !strcmp(v, "0xc0000005"), "%s: exception code %s, want STATUS_ACCESS_VIOLATION", steps[i], v);
        long want_off = steps[i][0] == 'g' ? 72 : 64 + (steps[i][5] == 's' ? 8 : 0);
        if (t_child_find(&r, "FAULT_OFF=", v, sizeof v)) {
            long off = strtol(v, NULL, 10);
            if (steps[i][0] == 'g') CHECKF(off == 72, "gather: fault address offset %ld, want 72 (element 2)", off);
            else CHECKF(off >= 64 && off < 80, "%s: fault address offset %ld outside the active guard-page lanes (want ~%ld)", steps[i], off, want_off);
        } else CHECKF(0, "%s: no FAULT_OFF line", steps[i]);
        CHECKF(t_child_find(&r, "DATA_OK=", v, sizeof v) && !strcmp(v, "1"), "%s: result after restart wrong (state not restored by the OS?)", steps[i]);
        if (steps[i][0] == 'g') {
            CHECKF(t_child_find(&r, "MASK_ZERO=", v, sizeof v) && !strcmp(v, "1"), "gather: mask register not zero after completion");
            if (t_child_find(&r, "CTX_MASK=", v, sizeof v) && strcmp(v, "unavailable")) {
                unsigned m0, m1, m2, m3;
                if (sscanf(v, "%x,%x,%x,%x", &m0, &m1, &m2, &m3) == 4) {
                    CHECKF(m0 == 0 && m1 == 0, "gather: elements 0,1 (right of the faulting one) not complete at fault: mask %s", v);
                    CHECKF(m2 == 0xFFFFFFFFu, "gather: faulting element 2 mask must still be set at fault: mask %s", v);
                    t_info("gather_mask_at_fault", "%s", v);
                } else CHECKF(0, "gather: bad CTX_MASK line %s", v);
            } else t_note("gather: CONTEXT did not carry XMM registers (CONTEXT_EXTENDED_REGISTERS absent)");
        }
    }
}

/* #UD when destination, index and mask registers are not distinct (SDM VGATHERDPS) */
/* vgatherdps ymm0, [RC + ymm0*4], ymm3 (index == destination): raw bytes, gas warns on the intentional misuse */
VOP(g_same_regs_avx, ".byte 0xc4, 0xe2, 0x65, 0x92, 0x04, 0x81")
CHILD(avx_gather_same_regs) {
    static T_ALIGN(32) uint32_t tab[64];
    vctx c; vctx_pattern(&c, 1); c.mem = (uintptr_t)tab; c.r[0] = V_ZERO; c.r[3] = V_ZERO;
    printf("STEP=ud\n"); fflush(stdout);
    g_same_regs_avx(&c);
    printf("NO_FAULT=1\n");
    return 0;
}
TEST(avx2_gather_nondistinct_registers_ud) {
    NEED(AVX2);
    static t_child_result r;
    REQUIRE(t_child_run("avx_gather_same_regs", NULL, 10000, &r));
    CHECK(!r.timed_out);
    CHECKF(r.exc_code == 0xC000001Du, "vgatherdps with index == destination: exception 0x%08lx (exit 0x%lx), want STATUS_ILLEGAL_INSTRUCTION",
           (unsigned long)r.exc_code, (unsigned long)r.exit_code);
}

/* =========================================================================== unmasked SIMD FP exception
 * Windows reports #XM (unmasked SSE/AVX exception) as follows (pinned by Wine's conformance test
 * dlls/ntdll/tests/exception.c test_simd_exceptions, which is written against Windows):
 *   x64:  the specific code (EXCEPTION_FLT_INVALID_OPERATION 0xC0000090 for #I, EXCEPTION_FLT_DIVIDE_BY_ZERO
 *         0xC000008E for #Z), NumberParameters == 2, ExceptionInformation[0] == 0, [1] == CONTEXT.MxCsr;
 *   x86:  STATUS_FLOAT_MULTIPLE_TRAPS (0xC00002B5), ExceptionInformation[0] == 0.
 * The CONTEXT carries MXCSR at the faulting instruction; the destination is not written (SDM vol.1 11.5.1),
 * and a handler that masks the exception in CONTEXT and continues re-executes the instruction. */
static ULONG_PTR g_veh_info[3];
static DWORD g_veh_nparams;
static int is_float_status(DWORD code) {
    return (code >= 0xC000008Du && code <= 0xC0000093u) || code == STATUS_FLOAT_MULTIPLE_FAULTS || code == STATUS_FLOAT_MULTIPLE_TRAPS;
}
static LONG CALLBACK veh_simd_fp(EXCEPTION_POINTERS *ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (!is_float_status(code)) return EXCEPTION_CONTINUE_SEARCH;
    if (InterlockedIncrement(&g_veh_count) > 3) return EXCEPTION_CONTINUE_SEARCH;
    g_veh_code = code;
    g_veh_nparams = ep->ExceptionRecord->NumberParameters;
    for (DWORD i = 0; i < 3 && i < g_veh_nparams; i++) g_veh_info[i] = ep->ExceptionRecord->ExceptionInformation[i];
    uint32_t mx = 0;
    g_veh_ctx_ok = ctx_get_mxcsr(ep->ContextRecord, &mx) && ctx_get_vec(ep->ContextRecord, 0, g_veh_xmm0);
    g_veh_mxcsr = mx;
    if (!g_veh_ctx_ok) return EXCEPTION_CONTINUE_SEARCH;
    ctx_set_mxcsr(ep->ContextRecord, (mx | 0x1F80u) & ~MX_FLAGS);      /* mask everything, clear flags, retry */
    return EXCEPTION_CONTINUE_EXECUTION;
}
/* One unmasked SIMD FP exception per kind. Lanes not named below compute 1 op 2 (exact, no flags).
 *   inv: vaddps, lane 3 = SNaN + 2            -> #I   (masked retry: QNaN 0x7fe00001)
 *   div: vdivps, lane 3 = 1 / 0               -> #Z   (+inf)
 *   ovf: vmulps, lane 6 = FLT_MAX * 2         -> #O   (+inf, SDM 11.5.2.3: unmasked -> no result stored)
 *   unf: vmulps, lane 6 = 0x00800001 * 0.5    -> #U   (0x00400000, tie to even)
 *   pe:  vdivps, lane 6 = 1 / 3               -> #P   (0x3eaaaaab)
 *   den: vaddps, lane 6 = 2^-149 + 2          -> #D   (2.0)
 *   fma: vfmadd231ps, lane 5 = inf * 0 + 1    -> #I   (default NaN); other lanes 1*2+1 = 3
 * SSE/AVX exceptions are precise: with any unmasked exception pending the destination is not written
 * (SDM vol.1 11.5.1 / 11.8), so the CONTEXT shows the old destination and a handler that masks the
 * exception in CONTEXT.MxCsr and continues re-executes the instruction to completion. */
static void f_fmadd231ps_y_avx(vctx *c_);   /* defined with the FMA cases below */
typedef struct simd_exc_kind { const char *name; vop_fn f; int lane; uint32_t a, b, z, mask, flag, want_lane, want_other; int fma; } simd_exc_kind;
static const simd_exc_kind simd_exc_kinds[] = {
    { "inv", vaddps_y_avx,       3, 0x7fa00001u, 0x40000000u, 0,           0x080u, MX_IE, 0x7fe00001u, 0x40400000u, 0 },
    { "div", vdivps_y_avx,       3, 0x3f800000u, 0x00000000u, 0,           0x200u, MX_ZE, 0x7f800000u, 0x3f000000u, 0 },
    { "ovf", vmulps_y_avx,       6, 0x7f7fffffu, 0x40000000u, 0,           0x400u, MX_OE, 0x7f800000u, 0x40000000u, 0 },
    { "unf", vmulps_y_avx,       6, 0x00800001u, 0x3f000000u, 0,           0x800u, MX_UE, 0x00400000u, 0x40000000u, 0 },
    { "pe",  vdivps_y_avx,       6, 0x3f800000u, 0x40400000u, 0,           0x1000u, MX_PE, 0x3eaaaaabu, 0x3f000000u, 0 },
    { "den", vaddps_y_avx,       6, 0x00000001u, 0x40000000u, 0,           0x100u, MX_DE, 0x40000000u, 0x40400000u, 0 },
    { "fma", f_fmadd231ps_y_avx, 5, 0x7f800000u, 0x00000000u, 0x3f800000u, 0x080u, MX_IE, 0xffc00000u, 0x40400000u, 1 },
};
static const simd_exc_kind *simd_exc_find(const char *n) {
    for (size_t i = 0; i < T_COUNTOF(simd_exc_kinds); i++) if (!strcmp(simd_exc_kinds[i].name, n)) return &simd_exc_kinds[i];
    return NULL;
}
CHILD(avx_simd_fp_unmasked) {
    const simd_exc_kind *k = simd_exc_find(argc > 1 ? argv[1] : "");
    if (!k) return 5;
    PVOID h = AddVectoredExceptionHandler(1, veh_simd_fp);
    if (!h) return 6;
    vctx c; vctx_pattern(&c, 0xFE);
    for (int i = 0; i < 8; i++) { c.r[1].d[i] = 0x3f800000u; c.r[2].d[i] = 0x40000000u; if (k->fma) c.r[0].d[i] = 0x3f800000u; }
    c.r[1].d[k->lane] = k->a; c.r[2].d[k->lane] = k->b; if (k->fma) c.r[0].d[k->lane] = k->z;
    c.mx_in = MX_DEFAULT & ~k->mask;
    vctx in = c;
    k->f(&c);
    RemoveVectoredExceptionHandler(h);
    int ok = 1;
    for (int i = 0; i < 8; i++) ok &= c.r[0].d[i] == (i == k->lane ? k->want_lane : k->want_other);
    printf("FAULTS=%ld\nCODE=0x%08lx\nNPARAMS=%lu\nINFO0=0x%llx\nINFO1=0x%llx\nCTX_OK=%d\nCTX_MXCSR=0x%08x\nMX_OUT=0x%08x\nDATA_OK=%d\n",
           (long)g_veh_count, (unsigned long)g_veh_code, (unsigned long)g_veh_nparams, (unsigned long long)g_veh_info[0],
           (unsigned long long)g_veh_info[1], g_veh_ctx_ok, g_veh_mxcsr, c.mx_out, ok);
    printf("DEST_UNCHANGED_AT_FAULT=%d\n", g_veh_ctx_ok && !memcmp(g_veh_xmm0, &in.r[0], 16));
    return 0;
}
static void simd_exc_case(const char *kind) {
    static t_child_result r; char v[64] = "";
    const simd_exc_kind *k = simd_exc_find(kind);
    REQUIRE(k != NULL);
    REQUIRE(t_child_run("avx_simd_fp_unmasked", kind, 10000, &r));
    REQUIREF(!r.timed_out && r.exit_code == 0, "%s: child exit 0x%lx exc 0x%lx (unhandled: the exception was not a STATUS_FLOAT_* code or the CONTEXT lacked MXCSR)",
             kind, (unsigned long)r.exit_code, (unsigned long)r.exc_code);
    CHECKF(t_child_find(&r, "FAULTS=", v, sizeof v) && !strcmp(v, "1"), "%s: exactly one #XM expected, got %s", kind, v);
    t_child_find(&r, "CODE=", v, sizeof v);
    char key[48]; snprintf(key, sizeof key, "simd_exception_code_%s", kind);
    t_info(key, "%s", v);
    /* Pinned codes (Wine dlls/ntdll/tests/exception.c test_simd_exceptions, written against Windows): x64 #Z and
     * #I -> the specific EXCEPTION_FLT_* code; x86 (native and WOW64) -> STATUS_FLOAT_MULTIPLE_TRAPS. Other kinds:
     * only the STATUS_FLOAT_* family is required (veh_simd_fp handles nothing else). */
    const char *want = NULL;
#ifdef T_X64
    if (k->flag == MX_ZE) want = "0xc000008e"; else if (k->flag == MX_IE) want = "0xc0000090";
#else
    if (k->flag == MX_ZE || k->flag == MX_IE) want = "0xc00002b5";
#endif
    if (want) CHECKF(!strcmp(v, want), "%s: exception code %s, want %s", kind, v, want);
    char mxs[32] = "";
    t_child_find(&r, "CTX_MXCSR=", mxs, sizeof mxs);
#ifdef T_X64
    CHECKF(t_child_find(&r, "NPARAMS=", v, sizeof v) && !strcmp(v, "2"), "%s: NumberParameters %s, want 2", kind, v);
    CHECKF(t_child_find(&r, "INFO0=", v, sizeof v) && !strcmp(v, "0x0"), "%s: ExceptionInformation[0] %s, want 0", kind, v);
    CHECKF(t_child_find(&r, "INFO1=", v, sizeof v) && strtoull(v, NULL, 16) == strtoull(mxs, NULL, 16),
           "%s: ExceptionInformation[1] %s, want CONTEXT.MxCsr %s", kind, v, mxs);
#endif
    CHECKF(t_child_find(&r, "CTX_OK=", v, sizeof v) && !strcmp(v, "1"), "%s: CONTEXT did not carry MXCSR/XMM state", kind);
    unsigned mx = (unsigned)strtoul(mxs, NULL, 16), fl = k->flag, msk = k->mask;
    CHECKF((mx & fl) && !(mx & msk), "%s: CONTEXT MXCSR 0x%x: want the flag set and its mask clear (state at the faulting instruction)", kind, mx);
    CHECKF(t_child_find(&r, "DEST_UNCHANGED_AT_FAULT=", v, sizeof v) && !strcmp(v, "1"), "%s: destination modified by a faulting instruction", kind);
    CHECKF(t_child_find(&r, "DATA_OK=", v, sizeof v) && !strcmp(v, "1"), "%s: result after the handler masked the exception is wrong", kind);
    if (t_child_find(&r, "MX_OUT=", v, sizeof v)) {
        unsigned mo = (unsigned)strtoul(v, NULL, 16);
        CHECKF((mo & msk) && (mo & fl), "%s: MXCSR after resume 0x%x: handler's CONTEXT change not applied", kind, mo);
    }
}
TEST(avx_unmasked_simd_exception_invalid) { NEED(AVX); simd_exc_case("inv"); }
TEST(avx_unmasked_simd_exception_divzero) { NEED(AVX); simd_exc_case("div"); }
TEST(avx_unmasked_simd_exception_overflow_upper_lane) { NEED(AVX); simd_exc_case("ovf"); }
TEST(avx_unmasked_simd_exception_underflow_upper_lane) { NEED(AVX); simd_exc_case("unf"); }
TEST(avx_unmasked_simd_exception_precision_upper_lane) { NEED(AVX); simd_exc_case("pe"); }
TEST(avx_unmasked_simd_exception_denormal_upper_lane) { NEED(AVX); simd_exc_case("den"); }
TEST(fma_unmasked_simd_exception_invalid) { NEED(FMA); simd_exc_case("fma"); }

/* =========================================================================== AVX2 integer arithmetic */
typedef void (*model2)(v256 *d, const v256 *a, const v256 *b);
#define EWM(name, esz, expr) static void name(v256 *d, const v256 *A, const v256 *B) { \
    for (int i = 0; i < 32 / (esz); i++) { \
        int64_t sa = lane_s(A, esz, i), sb = lane_s(B, esz, i); uint64_t ua = lane_u(A, esz, i), ub = lane_u(B, esz, i); \
        (void)sa; (void)sb; (void)ua; (void)ub; lane_set(d, esz, i, (uint64_t)(expr)); } }
#define IOP(mn) VOP(i_##mn##_avx, "v" #mn " %%ymm2, %%ymm1, %%ymm0")
EWM(m_paddb, 1, ua + ub) EWM(m_paddw, 2, ua + ub) EWM(m_paddd, 4, ua + ub) EWM(m_paddq, 8, ua + ub)
EWM(m_psubb, 1, ua - ub) EWM(m_psubw, 2, ua - ub) EWM(m_psubd, 4, ua - ub) EWM(m_psubq, 8, ua - ub)
EWM(m_paddsb, 1, sat_s(sa + sb, 8)) EWM(m_paddsw, 2, sat_s(sa + sb, 16)) EWM(m_psubsb, 1, sat_s(sa - sb, 8)) EWM(m_psubsw, 2, sat_s(sa - sb, 16))
EWM(m_paddusb, 1, sat_u((int64_t)(ua + ub), 8)) EWM(m_paddusw, 2, sat_u((int64_t)(ua + ub), 16))
EWM(m_psubusb, 1, sat_u((int64_t)ua - (int64_t)ub, 8)) EWM(m_psubusw, 2, sat_u((int64_t)ua - (int64_t)ub, 16))
EWM(m_pmullw, 2, ua * ub) EWM(m_pmulhw, 2, (sa * sb) >> 16) EWM(m_pmulhuw, 2, (ua * ub) >> 16) EWM(m_pmulld, 4, ua * ub)
EWM(m_pmulhrsw, 2, (((sa * sb) >> 14) + 1) >> 1)
EWM(m_pmuludq, 8, (ua & 0xffffffffull) * (ub & 0xffffffffull)) EWM(m_pmuldq, 8, (uint64_t)((int64_t)(int32_t)ua * (int64_t)(int32_t)ub))
EWM(m_pavgb, 1, (ua + ub + 1) >> 1) EWM(m_pavgw, 2, (ua + ub + 1) >> 1)
EWM(m_pminsb, 1, sa < sb ? sa : sb) EWM(m_pminsw, 2, sa < sb ? sa : sb) EWM(m_pminsd, 4, sa < sb ? sa : sb)
EWM(m_pminub, 1, ua < ub ? ua : ub) EWM(m_pminuw, 2, ua < ub ? ua : ub) EWM(m_pminud, 4, ua < ub ? ua : ub)
EWM(m_pmaxsb, 1, sa > sb ? sa : sb) EWM(m_pmaxsw, 2, sa > sb ? sa : sb) EWM(m_pmaxsd, 4, sa > sb ? sa : sb)
EWM(m_pmaxub, 1, ua > ub ? ua : ub) EWM(m_pmaxuw, 2, ua > ub ? ua : ub) EWM(m_pmaxud, 4, ua > ub ? ua : ub)
EWM(m_pabsb, 1, sa < 0 ? -sa : sa) EWM(m_pabsw, 2, sa < 0 ? -sa : sa) EWM(m_pabsd, 4, sa < 0 ? -sa : sa)
EWM(m_psignb, 1, sb < 0 ? -sa : sb == 0 ? 0 : sa) EWM(m_psignw, 2, sb < 0 ? -sa : sb == 0 ? 0 : sa) EWM(m_psignd, 4, sb < 0 ? -sa : sb == 0 ? 0 : sa)
EWM(m_pcmpeqb, 1, ua == ub ? ~0ull : 0) EWM(m_pcmpeqw, 2, ua == ub ? ~0ull : 0) EWM(m_pcmpeqd, 4, ua == ub ? ~0ull : 0) EWM(m_pcmpeqq, 8, ua == ub ? ~0ull : 0)
EWM(m_pcmpgtb, 1, sa > sb ? ~0ull : 0) EWM(m_pcmpgtw, 2, sa > sb ? ~0ull : 0) EWM(m_pcmpgtd, 4, sa > sb ? ~0ull : 0) EWM(m_pcmpgtq, 8, sa > sb ? ~0ull : 0)
EWM(m_pand, 8, ua & ub) EWM(m_pandn, 8, ~ua & ub) EWM(m_por, 8, ua | ub) EWM(m_pxor, 8, ua ^ ub)
EWM(m_psllvd, 4, ub > 31 ? 0 : ua << ub) EWM(m_psrlvd, 4, ub > 31 ? 0 : ua >> ub) EWM(m_psravd, 4, sa >> (ub > 31 ? 31 : ub))
EWM(m_psllvq, 8, ub > 63 ? 0 : ua << ub) EWM(m_psrlvq, 8, ub > 63 ? 0 : ua >> ub)
IOP(paddb) IOP(paddw) IOP(paddd) IOP(paddq) IOP(psubb) IOP(psubw) IOP(psubd) IOP(psubq)
IOP(paddsb) IOP(paddsw) IOP(psubsb) IOP(psubsw) IOP(paddusb) IOP(paddusw) IOP(psubusb) IOP(psubusw)
IOP(pmullw) IOP(pmulhw) IOP(pmulhuw) IOP(pmulld) IOP(pmulhrsw) IOP(pmuludq) IOP(pmuldq) IOP(pavgb) IOP(pavgw)
IOP(pminsb) IOP(pminsw) IOP(pminsd) IOP(pminub) IOP(pminuw) IOP(pminud)
IOP(pmaxsb) IOP(pmaxsw) IOP(pmaxsd) IOP(pmaxub) IOP(pmaxuw) IOP(pmaxud)
IOP(psignb) IOP(psignw) IOP(psignd)
IOP(pcmpeqb) IOP(pcmpeqw) IOP(pcmpeqd) IOP(pcmpeqq) IOP(pcmpgtb) IOP(pcmpgtw) IOP(pcmpgtd) IOP(pcmpgtq)
IOP(pand) IOP(pandn) IOP(por) IOP(pxor) IOP(psllvd) IOP(psrlvd) IOP(psravd) IOP(psllvq) IOP(psrlvq)
VOP(i_pabsb_avx, "vpabsb %%ymm1, %%ymm0") VOP(i_pabsw_avx, "vpabsw %%ymm1, %%ymm0") VOP(i_pabsd_avx, "vpabsd %%ymm1, %%ymm0")
/* x-forms of the variable shifts (VEX.128 zero upper) */
VOP(i_psllvd_x_avx, "vpsllvd %%xmm2, %%xmm1, %%xmm0")  VOP(i_psravd_x_avx, "vpsravd %%xmm2, %%xmm1, %%xmm0")
/* pairwise / horizontal / packing / sad */
static void m_pmaddwd(v256 *d, const v256 *a, const v256 *b) { for (int i = 0; i < 8; i++) d->d[i] = (uint32_t)((int64_t)a->sw[2*i] * b->sw[2*i] + (int64_t)a->sw[2*i+1] * b->sw[2*i+1]); }
static void m_pmaddubsw(v256 *d, const v256 *a, const v256 *b) { for (int i = 0; i < 16; i++) d->w[i] = (uint16_t)sat_s((int64_t)a->b[2*i] * b->sb[2*i] + (int64_t)a->b[2*i+1] * b->sb[2*i+1], 16); }
static void m_psadbw(v256 *d, const v256 *a, const v256 *b) {
    for (int q = 0; q < 4; q++) { uint64_t s = 0; for (int k = 0; k < 8; k++) s += (uint64_t)abs((int)a->b[8*q+k] - (int)b->b[8*q+k]); d->q[q] = s; }
}
static void m_pack(v256 *d, const v256 *a, const v256 *b, int from, int sgn) {   /* in-lane: [a lane | b lane] */
    int n = 16 / from;
    for (int L = 0; L < 2; L++)
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < n; i++) {
                int64_t v = lane_s(k ? b : a, from, L * n + i);
                uint64_t r = sgn ? (uint64_t)sat_s(v, from * 4) : sat_u(v, from * 4);
                lane_set(d, from / 2, L * 2 * n + k * n + i, r);
            }
}
static void m_packsswb(v256 *d, const v256 *a, const v256 *b) { m_pack(d, a, b, 2, 1); }
static void m_packuswb(v256 *d, const v256 *a, const v256 *b) { m_pack(d, a, b, 2, 0); }
static void m_packssdw(v256 *d, const v256 *a, const v256 *b) { m_pack(d, a, b, 4, 1); }
static void m_packusdw(v256 *d, const v256 *a, const v256 *b) { m_pack(d, a, b, 4, 0); }
static void m_hop(v256 *d, const v256 *a, const v256 *b, int esz, int sub, int sat) {
    int n = 16 / esz;
    for (int L = 0; L < 2; L++)
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < n / 2; i++) {
                const v256 *s = k ? b : a;
                int64_t x = lane_s(s, esz, L * n + 2 * i), y = lane_s(s, esz, L * n + 2 * i + 1), r = sub ? x - y : x + y;
                if (sat) r = sat_s(r, 16);
                lane_set(d, esz, L * n + k * n / 2 + i, (uint64_t)r);
            }
}
static void m_phaddw(v256 *d, const v256 *a, const v256 *b) { m_hop(d, a, b, 2, 0, 0); }
static void m_phaddd(v256 *d, const v256 *a, const v256 *b) { m_hop(d, a, b, 4, 0, 0); }
static void m_phaddsw(v256 *d, const v256 *a, const v256 *b) { m_hop(d, a, b, 2, 0, 1); }
static void m_phsubw(v256 *d, const v256 *a, const v256 *b) { m_hop(d, a, b, 2, 1, 0); }
static void m_phsubd(v256 *d, const v256 *a, const v256 *b) { m_hop(d, a, b, 4, 1, 0); }
static void m_phsubsw(v256 *d, const v256 *a, const v256 *b) { m_hop(d, a, b, 2, 1, 1); }
IOP(pmaddwd) IOP(pmaddubsw) IOP(psadbw) IOP(packsswb) IOP(packuswb) IOP(packssdw) IOP(packusdw)
IOP(phaddw) IOP(phaddd) IOP(phaddsw) IOP(phsubw) IOP(phsubd) IOP(phsubsw)
/* shifts with the count in the low 64 bits of an xmm register */
static void m_sllw_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 2, 0, b->q[0]); }
static void m_srlw_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 2, 1, b->q[0]); }
static void m_sraw_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 2, 2, b->q[0]); }
static void m_slld_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 4, 0, b->q[0]); }
static void m_srld_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 4, 1, b->q[0]); }
static void m_srad_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 4, 2, b->q[0]); }
static void m_sllq_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 8, 0, b->q[0]); }
static void m_srlq_x(v256 *d, const v256 *a, const v256 *b) { shift_model(d, a, 8, 1, b->q[0]); }
VOP(i_psllw_x_avx, "vpsllw %%xmm2, %%ymm1, %%ymm0") VOP(i_psrlw_x_avx, "vpsrlw %%xmm2, %%ymm1, %%ymm0") VOP(i_psraw_x_avx, "vpsraw %%xmm2, %%ymm1, %%ymm0")
VOP(i_pslld_x_avx, "vpslld %%xmm2, %%ymm1, %%ymm0") VOP(i_psrld_x_avx, "vpsrld %%xmm2, %%ymm1, %%ymm0") VOP(i_psrad_x_avx, "vpsrad %%xmm2, %%ymm1, %%ymm0")
VOP(i_psllq_x_avx, "vpsllq %%xmm2, %%ymm1, %%ymm0") VOP(i_psrlq_x_avx, "vpsrlq %%xmm2, %%ymm1, %%ymm0")
/* sign / zero extension */
static void m_ext(v256 *d, const v256 *a, int from, int to, int sgn) {
    for (int i = 0; i < 32 / to; i++) lane_set(d, to, i, sgn ? (uint64_t)lane_s(a, from, i) : lane_u(a, from, i));
}
#define EXTM(name, f, t, s) static void m_##name(v256 *d, const v256 *a, const v256 *b) { (void)b; m_ext(d, a, f, t, s); } \
    VOP(i_##name##_avx, "v" #name " %%xmm1, %%ymm0")
EXTM(pmovsxbw, 1, 2, 1) EXTM(pmovsxbd, 1, 4, 1) EXTM(pmovsxbq, 1, 8, 1) EXTM(pmovsxwd, 2, 4, 1) EXTM(pmovsxwq, 2, 8, 1) EXTM(pmovsxdq, 4, 8, 1)
EXTM(pmovzxbw, 1, 2, 0) EXTM(pmovzxbd, 1, 4, 0) EXTM(pmovzxbq, 1, 8, 0) EXTM(pmovzxwd, 2, 4, 0) EXTM(pmovzxwq, 2, 8, 0) EXTM(pmovzxdq, 4, 8, 0)

typedef struct bop { const char *n; vop_fn f; model2 m; int esz; int cnt; int xform; } bop;
#define B(mn, esz) { "v" #mn, i_##mn##_avx, m_##mn, esz, 0, 0 }
#define BC(mn, esz, md) { "v" #mn " ymm, ymm, xmm-count", i_##mn##_x_avx, md, esz, 1, 0 }
static void run_bops(const bop *ops, size_t n, int vecs, uint64_t seed) {
    t_rng r; t_rng_seed(&r, seed);
    for (size_t k = 0; k < n; k++) {
        agg_t a; agg_init(&a, "%s", ops[k].n);
        for (int it = 0; it < vecs; it++) {
            vctx c; vctx_pattern(&c, (uint32_t)it);
            gen_int(&r, &c.r[1], ops[k].esz); gen_int(&r, &c.r[2], ops[k].esz);
            if (ops[k].cnt && (it & 3)) c.r[2].q[0] = t_rng_range(&r, (uint64_t)ops[k].esz * 8 + 4);
            if (ops[k].cnt && !(it & 7)) c.r[2].q[0] = 0x100000000ull + (uint64_t)it;   /* high count bits matter */
            vctx o = c; vexec(&a, ops[k].f, &o, KEEP123);
            v256 w; ops[k].m(&w, &c.r[1], &c.r[2]);
            if (ops[k].xform) memset(w.b + 16, 0, 16);
            char ctx[400], ha[150], hb[150];
            snprintf(ctx, sizeof ctx, "a=[%s] b=[%s]", hexv(ha, sizeof ha, &c.r[1], 32, ops[k].esz), hexv(hb, sizeof hb, &c.r[2], 32, ops[k].esz));
            agg_mem(&a, &o.r[0], &w, 32, ops[k].esz, ctx);
        }
        AGG_END(&a);
    }
}
TEST(avx2_int_add_sub_saturating) {
    NEED(AVX2);
    static const bop ops[] = { B(paddb,1), B(paddw,2), B(paddd,4), B(paddq,8), B(psubb,1), B(psubw,2), B(psubd,4), B(psubq,8),
        B(paddsb,1), B(paddsw,2), B(psubsb,1), B(psubsw,2), B(paddusb,1), B(paddusw,2), B(psubusb,1), B(psubusw,2) };
    run_bops(ops, T_COUNTOF(ops), 300, 0x1A);
}
TEST(avx2_int_multiply) {
    NEED(AVX2);
    static const bop ops[] = { B(pmullw,2), B(pmulhw,2), B(pmulhuw,2), B(pmulld,4), B(pmulhrsw,2), B(pmuludq,8), B(pmuldq,8) };
    run_bops(ops, T_COUNTOF(ops), 300, 0x1B);
}
TEST(avx2_int_minmax_avg_abs_sign) {
    NEED(AVX2);
    static const bop ops[] = { B(pminsb,1), B(pminsw,2), B(pminsd,4), B(pminub,1), B(pminuw,2), B(pminud,4),
        B(pmaxsb,1), B(pmaxsw,2), B(pmaxsd,4), B(pmaxub,1), B(pmaxuw,2), B(pmaxud,4), B(pavgb,1), B(pavgw,2),
        B(pabsb,1), B(pabsw,2), B(pabsd,4), B(psignb,1), B(psignw,2), B(psignd,4) };
    run_bops(ops, T_COUNTOF(ops), 300, 0x1C);
}
TEST(avx2_int_compare_logic) {
    NEED(AVX2);
    static const bop ops[] = { B(pcmpeqb,1), B(pcmpeqw,2), B(pcmpeqd,4), B(pcmpeqq,8), B(pcmpgtb,1), B(pcmpgtw,2), B(pcmpgtd,4),
        B(pcmpgtq,8), B(pand,8), B(pandn,8), B(por,8), B(pxor,8) };
    run_bops(ops, T_COUNTOF(ops), 300, 0x1D);
}
TEST(avx2_pmadd_psadbw) {
    NEED(AVX2);
    static const bop ops[] = { B(pmaddwd,2), B(pmaddubsw,1), B(psadbw,1) };
    run_bops(ops, T_COUNTOF(ops), 500, 0x1E);
}
TEST(avx2_pack_saturate_in_lane) {
    NEED(AVX2);
    static const bop ops[] = { B(packsswb,2), B(packuswb,2), B(packssdw,4), B(packusdw,4) };
    run_bops(ops, T_COUNTOF(ops), 400, 0x1F);
}
TEST(avx2_phadd_phsub_in_lane) {
    NEED(AVX2);
    static const bop ops[] = { B(phaddw,2), B(phaddd,4), B(phaddsw,2), B(phsubw,2), B(phsubd,4), B(phsubsw,2) };
    run_bops(ops, T_COUNTOF(ops), 400, 0x20);
}
TEST(avx2_pmovsx_pmovzx) {
    NEED(AVX2);
    static const bop ops[] = { B(pmovsxbw,1), B(pmovsxbd,1), B(pmovsxbq,1), B(pmovsxwd,2), B(pmovsxwq,2), B(pmovsxdq,4),
        B(pmovzxbw,1), B(pmovzxbd,1), B(pmovzxbq,1), B(pmovzxwd,2), B(pmovzxwq,2), B(pmovzxdq,4) };
    run_bops(ops, T_COUNTOF(ops), 300, 0x21);
}
TEST(avx2_variable_shifts_vpsllv_vpsrlv_vpsrav) {
    NEED(AVX2);
    static const bop ops[] = { B(psllvd,4), B(psrlvd,4), B(psravd,4), B(psllvq,8), B(psrlvq,8),
        { "vpsllvd xmm (upper zero)", i_psllvd_x_avx, m_psllvd, 4, 0, 1 }, { "vpsravd xmm (upper zero)", i_psravd_x_avx, m_psravd, 4, 0, 1 } };
    run_bops(ops, T_COUNTOF(ops), 400, 0x22);
    /* explicit per-element count edges: 0, 1, 31, 32, 33, 0x7fffffff, 0x80000000, 0xffffffff */
    agg_t a; agg_init(&a, "vpsllvd/vpsrlvd/vpsravd per-element count edges");
    vctx c; vctx_pattern(&c, 1);
    static const uint32_t cnt[8] = { 0, 1, 31, 32, 33, 0x7fffffffu, 0x80000000u, 0xffffffffu };
    for (int i = 0; i < 8; i++) { c.r[1].d[i] = 0x80000001u; c.r[2].d[i] = cnt[i]; }
    static const uint32_t wl[8] = { 0x80000001u, 2, 0x80000000u, 0, 0, 0, 0, 0 };
    static const uint32_t wr[8] = { 0x80000001u, 0x40000000u, 1, 0, 0, 0, 0, 0 };
    static const uint32_t wa[8] = { 0x80000001u, 0xC0000000u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu };
    v256 w; vctx o;
    o = c; i_psllvd_avx(&o); memcpy(w.d, wl, 32); agg_mem(&a, &o.r[0], &w, 32, 4, "vpsllvd");
    o = c; i_psrlvd_avx(&o); memcpy(w.d, wr, 32); agg_mem(&a, &o.r[0], &w, 32, 4, "vpsrlvd");
    o = c; i_psravd_avx(&o); memcpy(w.d, wa, 32); agg_mem(&a, &o.r[0], &w, 32, 4, "vpsravd");
    AGG_END(&a);
}
TEST(avx2_shift_by_xmm_count) {
    NEED(AVX2);
    static const bop ops[] = { BC(psllw, 2, m_sllw_x), BC(psrlw, 2, m_srlw_x), BC(psraw, 2, m_sraw_x), BC(pslld, 4, m_slld_x),
        BC(psrld, 4, m_srld_x), BC(psrad, 4, m_srad_x), BC(psllq, 8, m_sllq_x), BC(psrlq, 8, m_srlq_x) };
    run_bops(ops, T_COUNTOF(ops), 300, 0x23);
}

/* =========================================================================== FMA3 */
/* AT&T "vfmaddNNNps %ymm2, %ymm1, %ymm0" = Intel "vfmaddNNNps ymm0, ymm1, ymm2":
 *   132: ymm0 = ymm0*ymm2 + ymm1   213: ymm0 = ymm1*ymm0 + ymm2   231: ymm0 = ymm1*ymm2 + ymm0 */
#define FMA_P(op, ord) \
    VOP(f_##op##ord##ps_y_avx, "v" #op #ord "ps %%ymm2, %%ymm1, %%ymm0") VOP(f_##op##ord##pd_y_avx, "v" #op #ord "pd %%ymm2, %%ymm1, %%ymm0") \
    VOP(f_##op##ord##ps_x_avx, "v" #op #ord "ps %%xmm2, %%xmm1, %%xmm0") VOP(f_##op##ord##pd_x_avx, "v" #op #ord "pd %%xmm2, %%xmm1, %%xmm0")
#define FMA_S(op, ord) \
    VOP(f_##op##ord##ss_avx, "v" #op #ord "ss %%xmm2, %%xmm1, %%xmm0") VOP(f_##op##ord##sd_avx, "v" #op #ord "sd %%xmm2, %%xmm1, %%xmm0")
#define FMA_ORDS(M, op) M(op, 132) M(op, 213) M(op, 231)
FMA_ORDS(FMA_P, fmadd) FMA_ORDS(FMA_P, fmsub) FMA_ORDS(FMA_P, fnmadd) FMA_ORDS(FMA_P, fnmsub)
FMA_ORDS(FMA_P, fmaddsub) FMA_ORDS(FMA_P, fmsubadd)
FMA_ORDS(FMA_S, fmadd) FMA_ORDS(FMA_S, fmsub) FMA_ORDS(FMA_S, fnmadd) FMA_ORDS(FMA_S, fnmsub)
enum { FM_MADD, FM_MSUB, FM_NMADD, FM_NMSUB, FM_MADDSUB, FM_MSUBADD };
enum { FK_YMM, FK_XMM, FK_SCALAR };
typedef struct fop { vop_fn f; int op, ord, dbl, kind; const char *n; } fop;
#define FE_P(op, OPC, ord, oi) { f_##op##ord##ps_y_avx, OPC, oi, 0, FK_YMM, "v" #op #ord "ps ymm" }, \
    { f_##op##ord##pd_y_avx, OPC, oi, 1, FK_YMM, "v" #op #ord "pd ymm" }, { f_##op##ord##ps_x_avx, OPC, oi, 0, FK_XMM, "v" #op #ord "ps xmm" }, \
    { f_##op##ord##pd_x_avx, OPC, oi, 1, FK_XMM, "v" #op #ord "pd xmm" },
#define FE_S(op, OPC, ord, oi) { f_##op##ord##ss_avx, OPC, oi, 0, FK_SCALAR, "v" #op #ord "ss" }, { f_##op##ord##sd_avx, OPC, oi, 1, FK_SCALAR, "v" #op #ord "sd" },
#define FE_ALL(M, op, OPC) M(op, OPC, 132, 0) M(op, OPC, 213, 1) M(op, OPC, 231, 2)
static const fop fma_ops[] = {
    FE_ALL(FE_P, fmadd, FM_MADD) FE_ALL(FE_P, fmsub, FM_MSUB) FE_ALL(FE_P, fnmadd, FM_NMADD) FE_ALL(FE_P, fnmsub, FM_NMSUB)
    FE_ALL(FE_P, fmaddsub, FM_MADDSUB) FE_ALL(FE_P, fmsubadd, FM_MSUBADD)
    FE_ALL(FE_S, fmadd, FM_MADD) FE_ALL(FE_S, fmsub, FM_MSUB) FE_ALL(FE_S, fnmadd, FM_NMADD) FE_ALL(FE_S, fnmsub, FM_NMSUB) };
/* expected element i of an FMA form (x*y+z with the form's negations), soft_fma model */
static uint64_t fma_expect(const fop *o, const vctx *c, int i, int rc, unsigned *fl) {
    ffmt f = o->dbl ? FMT64 : FMT32;
    int es = o->dbl ? 8 : 4;
    uint64_t r0 = 0, r1 = 0, r2 = 0;
    memcpy(&r0, c->r[0].b + i * es, (size_t)es); memcpy(&r1, c->r[1].b + i * es, (size_t)es); memcpy(&r2, c->r[2].b + i * es, (size_t)es);
    uint64_t x, y, z;
    if (o->ord == 0) { x = r0; y = r2; z = r1; } else if (o->ord == 1) { x = r1; y = r0; z = r2; } else { x = r1; y = r2; z = r0; }
    int negp = o->op == FM_NMADD || o->op == FM_NMSUB, subz = o->op == FM_MSUB || o->op == FM_NMSUB;
    if (o->op == FM_MADDSUB) subz = !(i & 1);
    if (o->op == FM_MSUBADD) subz = i & 1;
    if (negp) x ^= fmt_sign(f);
    if (subz) z ^= fmt_sign(f);
    return soft_fma(f, x, y, z, rc, fl);
}
static void fma_check_one(agg_t *a, const fop *o, vctx *c, int rc, uint32_t mxextra, const char *what) {
    c->mx_in = MX_RC(rc) | mxextra;
    vctx o_ = *c; vexec(a, o->f, &o_, KEEP12);
    int es = o->dbl ? 8 : 4, n = o->kind == FK_YMM ? 32 / es : o->kind == FK_XMM ? 16 / es : 1;
    v256 w = V_ZERO; unsigned fl = 0;
    if (o->kind == FK_SCALAR) memcpy(&w, &c->r[0], 16);                     /* 127:es from dest (= src1) */
    for (int i = 0; i < n; i++) { uint64_t v = fma_expect(o, c, i, rc, &fl); memcpy(w.b + i * es, &v, (size_t)es); }
    char ctx[300], h0[120], h1[120], h2[120];
    snprintf(ctx, sizeof ctx, "%s %s r0=[%s] r1=[%s] r2=[%s]", what, rc_name[rc], hexv(h0, sizeof h0, &c->r[0], 16, es),
             hexv(h1, sizeof h1, &c->r[1], 16, es), hexv(h2, sizeof h2, &c->r[2], 16, es));
    agg_mem(a, &o_.r[0], &w, 32, es, ctx);
    a->n++;
    if ((o_.mx_out & (MX_IE | MX_ZE | MX_OE | MX_UE | MX_PE)) != fl)
        agg_fail(a, "%s: MXCSR flags 0x%02x, model 0x%02x", ctx, o_.mx_out & MX_FLAGS, fl);
}
static void gen_fma_vec(t_rng *r, vctx *c, int dbl, int hard) {
    for (int k = 0; k < 3; k++) {
        if (dbl) gen_f64v(r, &c->r[k], 1); else gen_f32v(r, &c->r[k], 1);
        if (hard) {   /* bias towards cancellation: addend close to -(x*y) magnitude */
            for (int i = 0; i < (dbl ? 4 : 8); i++) {
                if (dbl) c->r[k].q[i] = (t_rng_next(r) & 0x8000000000000000ull) | ((uint64_t)(1023 - 8 + t_rng_range(r, 16)) << 52) | (t_rng_next(r) & 0xfffffffffffffull);
                else c->r[k].d[i] = (t_rng_u32(r) & 0x80000000u) | ((uint32_t)(127 - 8 + t_rng_range(r, 16)) << 23) | (t_rng_u32(r) & 0x7fffff);
            }
        }
    }
}
TEST(fma_all_forms_vs_exact_model) {
    NEED(FMA);
    t_rng r; t_rng_seed(&r, 0xF3A);
    for (size_t k = 0; k < T_COUNTOF(fma_ops); k++) {
        agg_t a; agg_init(&a, "%s (operand order, single rounding, flags)", fma_ops[k].n);
        for (int it = 0; it < 80; it++) {
            vctx c; vctx_pattern(&c, (uint32_t)it);
            gen_fma_vec(&r, &c, fma_ops[k].dbl, it & 1);
            fma_check_one(&a, &fma_ops[k], &c, RN, 0, "rand");
        }
        /* distinct small integers make every operand order and sign visible: 2,3,5 */
        vctx c; vctx_pattern(&c, 0x235);
        for (int i = 0; i < 8; i++) {
            if (fma_ops[k].dbl) { if (i < 4) { c.r[0].q[i] = i2d(2 + i); c.r[1].q[i] = i2d(3 + 7 * i); c.r[2].q[i] = i2d(5 + 11 * i); } }
            else { c.r[0].d[i] = i2f(2 + i); c.r[1].d[i] = i2f(3 + 7 * i); c.r[2].d[i] = i2f(5 + 11 * i); }
        }
        fma_check_one(&a, &fma_ops[k], &c, RN, 0, "ints");
        AGG_END(&a);
    }
}
TEST(fma_rounding_modes_and_ftz_daz) {
    NEED(FMA);
    t_rng r; t_rng_seed(&r, 0xF3B);
    static const int pick[] = { 2, 3, 6, 7, 10, 11 };    /* vfmadd231ps/pd ymm, vfmsub231 ymm, vfnmadd132 ymm/x ... */
    for (int rc = 0; rc < 4; rc++) {
        agg_t a; agg_init(&a, "FMA under MXCSR.RC=%s", rc_name[rc]);
        for (size_t p = 0; p < T_COUNTOF(pick); p++)
            for (int it = 0; it < 150; it++) {
                const fop *o = &fma_ops[pick[p] % T_COUNTOF(fma_ops)];
                vctx c; vctx_pattern(&c, (uint32_t)it);
                gen_fma_vec(&r, &c, o->dbl, it % 3 == 0);
                fma_check_one(&a, o, &c, rc, 0, o->n);
            }
        AGG_END(&a);
    }
    /* FTZ / DAZ (hand vectors): vfmadd231ps ymm0 = ymm1*ymm2 + ymm0 */
    agg_t a; agg_init(&a, "FMA with FTZ/DAZ");
    struct { uint32_t x, y, z, mx, want, wfl; const char *n; } v[] = {
        { 0x00800000, 0x3f000000, 0x00000000, MX_DEFAULT, 0x00400000, 0, "tiny exact, no FTZ" },
        { 0x00800000, 0x3f000000, 0x00000000, MX_DEFAULT | MX_FTZ, 0x00000000, MX_UE | MX_PE, "tiny, FTZ -> +0" },
        { 0x00400000, 0x4b800000, 0x00000000, MX_DEFAULT | MX_DAZ, 0x00000000, 0, "denormal multiplicand, DAZ -> 0" },
        { 0x3f800000, 0x3f800000, 0x00000001, MX_DEFAULT | MX_DAZ, 0x3f800000, 0, "denormal addend, DAZ: exact 1" },
        { 0x3f800000, 0x3f800000, 0x00000001, MX_DEFAULT, 0x3f800000, MX_PE | MX_DE, "denormal addend: inexact, #D" },
    };
    for (size_t i = 0; i < T_COUNTOF(v); i++) {
        vctx c; vctx_pattern(&c, 7);
        for (int l = 0; l < 8; l++) { c.r[1].d[l] = v[i].x; c.r[2].d[l] = v[i].y; c.r[0].d[l] = v[i].z; }
        c.mx_in = v[i].mx;
        f_fmadd231ps_y_avx(&c);
        a.n++;
        if (c.r[0].d[5] != v[i].want || (c.mx_out & MX_FLAGS) != v[i].wfl)
            agg_fail(&a, "%s: got 0x%08x flags 0x%02x want 0x%08x flags 0x%02x", v[i].n, c.r[0].d[5], c.mx_out & MX_FLAGS, v[i].want, v[i].wfl);
    }
    AGG_END(&a);
}
TEST(fma_single_rounding_and_special_operands) {
    NEED(FMA);
    agg_t a; agg_init(&a, "FMA known answers (single rounding, invalid, signed zero, overflow avoidance)");
    /* vfmadd231ps: r0 = r1*r2 + r0 ; columns: x(r1) y(r2) z(r0) rc want flags */
    struct { uint32_t x, y, z; int rc; uint32_t want, wfl; const char *n; } v[] = {
        { 0x3f800001, 0x3f7ffffe, 0xbf800000, RN, 0xa8800000, 0, "(1+u)(1-2u)-1 = -2^-46 exactly (mul+add would give 0)" },
        { 0x3f800001, 0x3f800001, 0xbf800002, RN, 0x28800000, 0, "(1+2^-23)^2-(1+2^-22) = 2^-46" },
        { 0x7f800000, 0x00000000, 0x3f800000, RN, 0xffc00000, MX_IE, "inf*0+1 -> default NaN, #I" },
        { 0x7f800000, 0x40000000, 0xff800000, RN, 0xffc00000, MX_IE, "inf*2-inf -> default NaN, #I" },
        { 0x7f800000, 0x40000000, 0x7f800000, RN, 0x7f800000, 0, "inf*2+inf = inf" },
        { 0x00000000, 0x40a00000, 0x80000000, RN, 0x00000000, 0, "+0*5 + -0 = +0 (RN)" },
        { 0x00000000, 0x40a00000, 0x80000000, RD, 0x80000000, 0, "+0*5 + -0 = -0 (RD)" },
        { 0x80000000, 0x40a00000, 0x80000000, RN, 0x80000000, 0, "-0*5 + -0 = -0" },
        { 0x40000000, 0x40400000, 0xc0c00000, RN, 0x00000000, 0, "2*3-6 = +0 (RN)" },
        { 0x40000000, 0x40400000, 0xc0c00000, RD, 0x80000000, 0, "2*3-6 = -0 (RD)" },
        { 0x7f7fffff, 0x40000000, 0xff7fffff, RN, 0x7f7fffff, 0, "MAX*2-MAX = MAX: no intermediate overflow" },
        { 0x1a000000, 0x1a000000, 0x00000000, RN, 0x00000000, MX_UE | MX_PE, "2^-75*2^-75 = 2^-150: tie -> +0" },
        { 0x1a000000, 0x1a000000, 0x00000000, RU, 0x00000001, MX_UE | MX_PE, "2^-150 RU -> min denormal" },
        { 0x0d800000, 0x27000000, 0x00000000, RN, 0x00000001, 0, "2^-100*2^-49 = 2^-149 exact denormal" },
        { 0x3f800000, 0x3f800000, 0x7fa00001, RN, 0x7fe00001, MX_IE, "1*1+SNaN -> quiet SNaN, #I" },
        { 0x7fc00005, 0x3f800000, 0x3f800000, RN, 0x7fc00005, 0, "QNaN*1+1 -> QNaN, no #I" },
    };
    for (size_t i = 0; i < T_COUNTOF(v); i++) {
        vctx c; vctx_pattern(&c, 8);
        for (int l = 0; l < 8; l++) { c.r[1].d[l] = v[i].x; c.r[2].d[l] = v[i].y; c.r[0].d[l] = v[i].z; }
        c.mx_in = MX_RC(v[i].rc);
        f_fmadd231ps_y_avx(&c);
        a.n++;
        if (c.r[0].d[2] != v[i].want || (c.mx_out & MX_FLAGS) != v[i].wfl)
            agg_fail(&a, "%s: got 0x%08x flags 0x%02x want 0x%08x flags 0x%02x", v[i].n, c.r[0].d[2], c.mx_out & MX_FLAGS, v[i].want, v[i].wfl);
    }
    vctx c; vctx_pattern(&c, 9);
    for (int l = 0; l < 4; l++) { c.r[1].q[l] = 0x3ff0000000000001ull; c.r[2].q[l] = 0x3fefffffffffffffull; c.r[0].q[l] = 0xbff0000000000000ull; }
    /* (1+2^-52)(1-2^-53) = 1 + 2^-53 - 2^-105 ; minus 1 = 2^-53 - 2^-105 (exact in double: 0x3c9ffffffffffffe) */
    f_fmadd231pd_y_avx(&c);
    agg_eq(&a, c.r[0].q[1], 0x3c9ffffffffffffeull, "(1+2^-52)(1-2^-53)-1 = 2^-53-2^-105 exactly");
    AGG_END(&a);
    /* informational: which NaN an FMA returns when several operands are NaN (not architecturally pinned across vendors) */
    vctx_pattern(&c, 10);
    for (int l = 0; l < 8; l++) { c.r[1].d[l] = 0x7fc00001; c.r[2].d[l] = 0x7fc00002; c.r[0].d[l] = 0x7fc00003; }
    vctx o = c; f_fmadd231ps_y_avx(&o);
    vctx o2 = c; f_fmadd132ps_y_avx(&o2);
    t_info("fma_nan_choice", "231(x=1,y=2,z=3)->%08x 132(r0=3,r2=2,r1=1)->%08x", o.r[0].d[0], o2.r[0].d[0]);
    vctx_pattern(&c, 11);
    for (int l = 0; l < 8; l++) { c.r[1].d[l] = 0x00000000; c.r[2].d[l] = 0x7f800000; c.r[0].d[l] = 0x7fc00003; }
    o = c; f_fmadd231ps_y_avx(&o);
    t_info("fma_0xinf_plus_qnan", "result %08x mxcsr.IE=%u (IEEE 754-2008: implementation defined)", o.r[0].d[0], o.mx_out & MX_IE);
}
/* the mingw CRT fma()/fmaf() are not usable as references: record how they compare (INFO only) */
TEST(fma_crt_reference_quality_info) {
    NEED(FMA);
    t_rng r; t_rng_seed(&r, 0xC47);
    int bad_d = 0, bad_f = 0, model_bad = 0, n = 20000;
    for (int i = 0; i < n; i++) {
        vctx c; vctx_pattern(&c, 1);
        gen_fma_vec(&r, &c, 1, i & 1);
        double x, y, z; memcpy(&x, &c.r[1].q[0], 8); memcpy(&y, &c.r[2].q[0], 8); memcpy(&z, &c.r[0].q[0], 8);
        volatile double crt = fma(x, y, z);
        vctx o = c; f_fmadd231sd_avx(&o);
        uint64_t cb; double cd = crt; memcpy(&cb, &cd, 8);
        if (cb != o.r[0].q[0]) bad_d++;
        if (soft_fma(FMT64, c.r[1].q[0], c.r[2].q[0], c.r[0].q[0], RN, NULL) != o.r[0].q[0]) model_bad++;
        float xf, yf, zf; uint32_t a1 = (uint32_t)(c.r[1].q[1] >> 32) & 0x3fffffffu, a2 = (uint32_t)(c.r[2].q[1] >> 32) & 0x3fffffffu, a3 = (uint32_t)(c.r[0].q[1] >> 32) & 0xbfffffffu;
        memcpy(&xf, &a1, 4); memcpy(&yf, &a2, 4); memcpy(&zf, &a3, 4);
        volatile float crtf = fmaf(xf, yf, zf);
        vctx of; vctx_pattern(&of, 2); of.r[1].d[0] = a1; of.r[2].d[0] = a2; of.r[0].d[0] = a3;
        f_fmadd231ss_avx(&of);
        uint32_t fb; float ff = crtf; memcpy(&fb, &ff, 4);
        if (fb != of.r[0].d[0]) bad_f++;
    }
    t_info("crt_fma_mismatch_vs_vfmadd", "%d/%d", bad_d, n);
    t_info("crt_fmaf_mismatch_vs_vfmadd", "%d/%d", bad_f, n);
    if (bad_d || bad_f) t_note("CRT fma/fmaf disagree with VFMADD on %d/%d double and %d/%d float inputs: the CRT is not a correctly-rounded reference (exact integer model used instead)", bad_d, n, bad_f, n);
    CHECKF(model_bad == 0, "exact model disagrees with VFMADD231SD on %d/%d inputs", model_bad, n);
}

/* =========================================================================== F16C */
VOP(h_ph2ps_avx, "vcvtph2ps %%xmm1, %%ymm0")  VOP(h_ph2ps_x_avx, "vcvtph2ps %%xmm1, %%xmm0")  VOP(h_ph2ps_m_avx, "vcvtph2ps (" RC "), %%ymm0")
#define vcvtps2ph_INSN "vcvtps2ph %1, %%ymm1, %%xmm0"
#define vcvtps2phm_INSN "vcvtps2ph %1, %%ymm1, (" RC ")"
DEF_IMM32(vcvtps2ph) DEF_IMM32(vcvtps2phm)
TEST(f16c_vcvtph2ps_exhaustive) {
    NEED(F16C);
    agg_t a; agg_init(&a, "vcvtph2ps all 65536 halves (exact; SNaN quieted with #I)");
    for (uint32_t h0 = 0; h0 < 65536; h0 += 8) {
        vctx c; vctx_pattern(&c, h0);
        for (int i = 0; i < 8; i++) c.r[1].w[i] = (uint16_t)(h0 + (uint32_t)i);
        vctx o = c; h_ph2ps_avx(&o);
        v256 w; int sn = 0;
        for (int i = 0; i < 8; i++) {
            uint16_t h = (uint16_t)(h0 + (uint32_t)i);
            w.d[i] = half_to_f32(h);
            sn |= (h & 0x7c00) == 0x7c00 && (h & 0x3ff) && !(h & 0x200);
        }
        char ctx[40]; snprintf(ctx, sizeof ctx, "halves 0x%04x..", h0);
        agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        agg_eq(&a, o.mx_out & MX_IE, sn ? MX_IE : 0, ctx);
    }
    static uint16_t hm[8] = { 0x3c00, 0xc000, 0x0001, 0x7bff, 0x8400, 0x7c00, 0xfe01, 0x0000 };
    vctx c; vctx_pattern(&c, 1); c.mem = (uintptr_t)hm;
    vctx o = c; h_ph2ps_m_avx(&o);
    v256 w; for (int i = 0; i < 8; i++) w.d[i] = half_to_f32(hm[i]);
    agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtph2ps m128");
    memcpy(c.r[1].w, hm, 16);
    o = c; h_ph2ps_x_avx(&o);
    w = V_ZERO; for (int i = 0; i < 4; i++) w.d[i] = half_to_f32(hm[i]);
    agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtph2ps xmm (upper zero)");
    AGG_END(&a);
}
static uint32_t gen_f32_halfrange(t_rng *r) {
    switch (t_rng_range(r, 6)) {
    case 0: return gen_f32(r, 0);
    case 1: return (t_rng_u32(r) & 0x80000000u) | ((uint32_t)(127 - 26 + t_rng_range(r, 12)) << 23) | (t_rng_u32(r) & 0x7fffff);  /* half denormal range */
    case 2: return (t_rng_u32(r) & 0x80000000u) | ((uint32_t)(127 + 14 + t_rng_range(r, 3)) << 23) | (t_rng_u32(r) & 0x7fffff);   /* near 65504 */
    case 3: return (t_rng_u32(r) & 0x80000000u) | ((uint32_t)(127 - 14) << 23) | (t_rng_u32(r) & 0x7fffff);                      /* min normal */
    case 4: return (t_rng_u32(r) & 0x80000000u) | ((uint32_t)(127 - 20 + t_rng_range(r, 40)) << 23) | ((t_rng_u32(r) & 0x7ff) << 12) | 0x1000u; /* ties */
    default: return (t_rng_u32(r) & 0x80000000u) | ((uint32_t)(127 - 20 + t_rng_range(r, 40)) << 23) | (t_rng_u32(r) & 0x7fffff);
    }
}
TEST(f16c_vcvtps2ph_all_rounding_imm) {
    NEED(F16C);
    t_rng r; t_rng_seed(&r, 0xF16);
    static uint8_t mem[48];
    for (int imm = 0; imm < 32; imm++) {
        agg_t a; agg_init(&a, "vcvtps2ph imm=%d (%s)", imm, (imm & 4) ? "MXCSR.RC" : rc_name[imm & 3]);
        for (int mrc = 0; mrc < 4; mrc++)
            for (int it = 0; it < 60; it++) {
                vctx c; vctx_pattern(&c, (uint32_t)it); c.mx_in = MX_RC(mrc);
                for (int i = 0; i < 8; i++) c.r[1].d[i] = gen_f32_halfrange(&r);
                int mode = (imm & 4) ? mrc : (imm & 3);
                vctx o = c; vexec(&a, vcvtps2ph_tab[imm], &o, KEEP12);
                v256 w = V_ZERO; unsigned fl = 0;
                for (int i = 0; i < 8; i++) w.w[i] = f32_to_half(c.r[1].d[i], mode, &fl);
                char ctx[300], h1[200];
                snprintf(ctx, sizeof ctx, "mxcsr.rc=%s in=[%s]", rc_name[mrc], hexv(h1, sizeof h1, &c.r[1], 32, 4));
                agg_mem(&a, &o.r[0], &w, 32, 2, ctx);
                a.n++;
                if ((o.mx_out & (MX_IE | MX_OE | MX_UE | MX_PE)) != fl) agg_fail(&a, "%s: flags 0x%02x model 0x%02x", ctx, o.mx_out & MX_FLAGS, fl);
                if (it == 0) {   /* memory form writes exactly 16 bytes */
                    memset(mem, 0xEE, sizeof mem); c.mem = (uintptr_t)(mem + 8);
                    o = c; vexec(&a, vcvtps2phm_tab[imm], &o, KEEP12);
                    uint8_t ref[48]; memset(ref, 0xEE, sizeof ref); memcpy(ref + 8, w.b, 16);
                    agg_mem(&a, mem, ref, sizeof mem, 2, "vcvtps2ph m128");
                }
            }
        AGG_END(&a);
    }
    /* known answers */
    agg_t a; agg_init(&a, "vcvtps2ph known answers");
    struct { uint32_t f; int rc; uint16_t h; } k[] = {
        { 0x3f800000, RN, 0x3c00 }, { 0x477fe000, RN, 0x7bff }, { 0x477ff000, RN, 0x7c00 }, { 0x477ff000, RZ, 0x7bff },
        { 0xc77ff000, RU, 0xfbff }, { 0xc77ff000, RD, 0xfc00 }, { 0x33000000, RN, 0x0000 }, { 0x33000001, RN, 0x0001 },
        { 0x33000000, RU, 0x0001 }, { 0x387fe000, RN, 0x0400 }, { 0x7f800000, RN, 0x7c00 }, { 0x7fa00001, RN, 0x7f00 },
        { 0xffc12345, RN, 0xfe09 }, { 0x80000000, RN, 0x8000 } };
    for (size_t i = 0; i < T_COUNTOF(k); i++) {
        vctx c; vctx_pattern(&c, 3);
        for (int l = 0; l < 8; l++) c.r[1].d[l] = k[i].f;
        vctx o = c; vcvtps2ph_tab[k[i].rc](&o);
        char ctx[64]; snprintf(ctx, sizeof ctx, "0x%08x %s", k[i].f, rc_name[k[i].rc]);
        agg_eq(&a, o.r[0].w[6], k[i].h, ctx);
        agg_eq(&a, f32_to_half(k[i].f, k[i].rc, NULL), k[i].h, "model self-check");
    }
    AGG_END(&a);
}

/* =========================================================================== AVX <-> legacy SSE transition */
VOP(trans_avx, "vaddps %%ymm1, %%ymm1, %%ymm0\n\taddps %%xmm2, %%xmm1\n\tvaddps %%ymm1, %%ymm1, %%ymm3")
TEST(avx_legacy_sse_transition_preserves_upper) {
    NEED(AVX);
    /* AVX op, then a legacy-SSE op on xmm1 (upper 128 of ymm1 must be preserved), then AVX reads all of ymm1 */
    t_rng r; t_rng_seed(&r, 0x7A);
    agg_t a; agg_init(&a, "AVX -> SSE -> AVX without VZEROUPPER: architectural state intact");
    for (int it = 0; it < 200; it++) {
        vctx c; vctx_pattern(&c, (uint32_t)it);
        gen_f32v(&r, &c.r[1], 1); gen_f32v(&r, &c.r[2], 1);
        vctx o = c; trans_avx(&o);
        v256 w0, w1, w3; uint32_t fl = 0;
        for (int i = 0; i < 8; i++) w0.d[i] = sse_f32(OP_ADD, c.r[1].d[i], c.r[1].d[i], MX_DEFAULT, &fl);
        w1 = c.r[1];
        for (int i = 0; i < 4; i++) w1.d[i] = sse_f32(OP_ADD, c.r[1].d[i], c.r[2].d[i], MX_DEFAULT, &fl);
        for (int i = 0; i < 8; i++) w3.d[i] = sse_f32(OP_ADD, w1.d[i], w1.d[i], MX_DEFAULT, &fl);
        agg_mem(&a, &o.r[0], &w0, 32, 4, "ymm0 (first AVX op)");
        agg_mem(&a, &o.r[1], &w1, 32, 4, "ymm1 after legacy addps (upper preserved)");
        agg_mem(&a, &o.r[3], &w3, 32, 4, "ymm3 (AVX op after SSE)");
    }
    AGG_END(&a);
}

/* =========================================================================== BMI1 / BMI2 / LZCNT / ADX */
typedef struct gctx { uint64_t a, b, c, r, r2, fl_in, fl_out; } gctx;
#ifdef T_X64
#define GPUSHF "pushq 40(%0)\n\tpopfq\n\t"
#define GPOPF  "pushfq\n\tpopq 48(%0)\n\t"
#else
#define GPUSHF "pushl 40(%0)\n\tpopfl\n\t"
#define GPOPF  "pushfl\n\tpopl 48(%0)\n\t"
#endif
#define GOP(fname, INSN) static T_NOINLINE void fname(gctx *g_) { __asm__ volatile( \
    "mov 0(%0), %%eax\n\tmov 8(%0), %%ecx\n\tmov 16(%0), %%edx\n\t" GPUSHF INSN "\n\t" GPOPF \
    "mov %%eax, 24(%0)\n\tmov %%edx, 32(%0)" :: "r"(g_) : "eax", "ecx", "edx", "cc", "memory"); }
#define GOP64(fname, INSN) static T_NOINLINE void fname(gctx *g_) { __asm__ volatile( \
    "mov 0(%0), %%rax\n\tmov 8(%0), %%rcx\n\tmov 16(%0), %%rdx\n\t" GPUSHF INSN "\n\t" GPOPF \
    "mov %%rax, 24(%0)\n\tmov %%rdx, 32(%0)" :: "r"(g_) : "rax", "rcx", "rdx", "cc", "memory"); }
GOP(b_andn32_bmi, "andn %%ecx, %%eax, %%eax")     GOP(b_bextr32_bmi, "bextr %%ecx, %%eax, %%eax")
GOP(b_blsi32_bmi, "blsi %%eax, %%eax")            GOP(b_blsmsk32_bmi, "blsmsk %%eax, %%eax")
GOP(b_blsr32_bmi, "blsr %%eax, %%eax")            GOP(b_tzcnt32_bmi, "tzcnt %%eax, %%eax")
GOP(b_lzcnt32_bmi, "lzcnt %%eax, %%eax")          GOP(b_bzhi32_bmi, "bzhi %%ecx, %%eax, %%eax")
GOP(b_mulx32_bmi, "mulx %%ecx, %%eax, %%edx")     GOP(b_mulx32s_bmi, "mulx %%ecx, %%eax, %%eax")
GOP(b_pdep32_bmi, "pdep %%ecx, %%eax, %%eax")     GOP(b_pext32_bmi, "pext %%ecx, %%eax, %%eax")
GOP(b_sarx32_bmi, "sarx %%ecx, %%eax, %%eax")     GOP(b_shlx32_bmi, "shlx %%ecx, %%eax, %%eax")
GOP(b_shrx32_bmi, "shrx %%ecx, %%eax, %%eax")
GOP(b_rorx32_0_bmi, "rorx $0, %%eax, %%eax")      GOP(b_rorx32_1_bmi, "rorx $1, %%eax, %%eax")
GOP(b_rorx32_13_bmi, "rorx $13, %%eax, %%eax")    GOP(b_rorx32_31_bmi, "rorx $31, %%eax, %%eax")
GOP(b_rorx32_33_bmi, "rorx $33, %%eax, %%eax")
GOP(b_adcx32_adx, "adcx %%ecx, %%eax")            GOP(b_adox32_adx, "adox %%ecx, %%eax")
#ifdef T_X64
GOP64(b_andn64_bmi, "andn %%rcx, %%rax, %%rax")   GOP64(b_bextr64_bmi, "bextr %%rcx, %%rax, %%rax")
GOP64(b_blsi64_bmi, "blsi %%rax, %%rax")          GOP64(b_blsmsk64_bmi, "blsmsk %%rax, %%rax")
GOP64(b_blsr64_bmi, "blsr %%rax, %%rax")          GOP64(b_tzcnt64_bmi, "tzcnt %%rax, %%rax")
GOP64(b_lzcnt64_bmi, "lzcnt %%rax, %%rax")        GOP64(b_bzhi64_bmi, "bzhi %%rcx, %%rax, %%rax")
GOP64(b_mulx64_bmi, "mulx %%rcx, %%rax, %%rdx")   GOP64(b_mulx64s_bmi, "mulx %%rcx, %%rax, %%rax")
GOP64(b_pdep64_bmi, "pdep %%rcx, %%rax, %%rax")   GOP64(b_pext64_bmi, "pext %%rcx, %%rax, %%rax")
GOP64(b_sarx64_bmi, "sarx %%rcx, %%rax, %%rax")   GOP64(b_shlx64_bmi, "shlx %%rcx, %%rax, %%rax")
GOP64(b_shrx64_bmi, "shrx %%rcx, %%rax, %%rax")
GOP64(b_rorx64_0_bmi, "rorx $0, %%rax, %%rax")    GOP64(b_rorx64_1_bmi, "rorx $1, %%rax, %%rax")
GOP64(b_rorx64_13_bmi, "rorx $13, %%rax, %%rax")  GOP64(b_rorx64_63_bmi, "rorx $63, %%rax, %%rax")
GOP64(b_rorx64_65_bmi, "rorx $65, %%rax, %%rax")
GOP64(b_adcx64_adx, "adcx %%rcx, %%rax")          GOP64(b_adox64_adx, "adox %%rcx, %%rax")
#endif
#define ST_FLAGS (EF_CF | EF_PF | EF_AF | EF_ZF | EF_SF | EF_OF)
enum { G_ANDN, G_BEXTR, G_BLSI, G_BLSMSK, G_BLSR, G_TZCNT, G_LZCNT, G_BZHI, G_MULX, G_MULXS, G_PDEP, G_PEXT, G_SARX, G_SHLX,
       G_SHRX, G_RORX, G_ADCX, G_ADOX };
typedef struct gpr_op { void (*f)(gctx *); int op, w, imm; const char *n; } gop_t;
static uint64_t wmask(int w) { return w == 64 ? ~0ull : (1ull << w) - 1; }
/* model: result r, r2 (mulx high / untouched edx), defined-flag mask + values, unaffected-flag mask */
static void gmodel(const gop_t *o, const gctx *g, uint64_t *r, uint64_t *r2, unsigned *defm, unsigned *defv, unsigned *unaff) {
    int w = o->w; uint64_t M = wmask(w), a = g->a & M, b = g->b & M, s = 1ull << (w - 1);
    *r = 0; *r2 = g->c & M; *defm = 0; *defv = 0; *unaff = 0;
    unsigned fin = (unsigned)g->fl_in;
    switch (o->op) {
    case G_ANDN: *r = ~a & b & M; *defm = EF_SF | EF_ZF | EF_CF | EF_OF; break;
    case G_BEXTR: {
        unsigned st = (unsigned)(b & 0xff), len = (unsigned)((b >> 8) & 0xff);
        uint64_t t = st >= (unsigned)w ? 0 : a >> st;
        if (len < (unsigned)w) t &= (1ull << len) - 1;
        *r = t; *defm = EF_ZF | EF_CF | EF_OF; break; }
    case G_BLSI: *r = a & (0 - a) & M; *defm = EF_SF | EF_ZF | EF_CF | EF_OF; if (a) *defv |= EF_CF; break;
    case G_BLSMSK: *r = (a ^ (a - 1)) & M; *defm = EF_SF | EF_ZF | EF_CF | EF_OF; if (!a) *defv |= EF_CF; break;
    case G_BLSR: *r = a & (a - 1) & M; *defm = EF_SF | EF_ZF | EF_CF | EF_OF; if (!a) *defv |= EF_CF; break;
    case G_TZCNT: { uint64_t n = 0; if (!a) n = (uint64_t)w; else while (!((a >> n) & 1)) n++; *r = n; *defm = EF_ZF | EF_CF; if (!a) *defv |= EF_CF; if (!n) *defv |= EF_ZF; return; }
    case G_LZCNT: { uint64_t n = 0; if (!a) n = (uint64_t)w; else while (!((a << n) & s)) n++; *r = n; *defm = EF_ZF | EF_CF; if (!a) *defv |= EF_CF; if (!n) *defv |= EF_ZF; return; }
    case G_BZHI: { unsigned n = (unsigned)(b & 0xff); *r = n < (unsigned)w ? a & ((1ull << n) - 1) : a; *defm = EF_SF | EF_ZF | EF_CF | EF_OF; if (n > (unsigned)w - 1) *defv |= EF_CF; break; }
    case G_MULX: case G_MULXS: {
        uint64_t c = g->c & M, lo, hi;
        if (w == 32) { uint64_t p = c * b; lo = p & M; hi = p >> 32; } else { u128 p = u128_mul64(c, b); lo = p.lo; hi = p.hi; }
        if (o->op == G_MULX) { *r = lo; *r2 = hi; } else *r = hi;
        *unaff = ST_FLAGS; return; }
    case G_PDEP: { uint64_t t = 0; int k = 0; for (int i = 0; i < w; i++) if ((b >> i) & 1) { if ((a >> k) & 1) t |= 1ull << i; k++; } *r = t; *unaff = ST_FLAGS; return; }
    case G_PEXT: { uint64_t t = 0; int k = 0; for (int i = 0; i < w; i++) if ((b >> i) & 1) { if ((a >> i) & 1) t |= 1ull << k; k++; } *r = t; *unaff = ST_FLAGS; return; }
    case G_SARX: { unsigned n = (unsigned)(b & (unsigned)(w - 1)); int64_t sv = w == 32 ? (int64_t)(int32_t)a : (int64_t)a; *r = (uint64_t)(sv >> n) & M; *unaff = ST_FLAGS; return; }
    case G_SHLX: { unsigned n = (unsigned)(b & (unsigned)(w - 1)); *r = (a << n) & M; *unaff = ST_FLAGS; return; }
    case G_SHRX: { unsigned n = (unsigned)(b & (unsigned)(w - 1)); *r = a >> n; *unaff = ST_FLAGS; return; }
    case G_RORX: { unsigned n = (unsigned)o->imm & (unsigned)(w - 1); *r = n ? ((a >> n) | (a << (w - n))) & M : a; *unaff = ST_FLAGS; return; }
    case G_ADCX: case G_ADOX: {
        unsigned cin = o->op == G_ADCX ? (fin & EF_CF) != 0 : (fin & EF_OF) != 0;
        uint64_t t = (a + b + cin) & M; int cout = w == 64 ? (t < a || (cin && t == a)) : ((a + b + cin) >> 32) != 0;
        *r = t; *defm = o->op == G_ADCX ? EF_CF : EF_OF; if (cout) *defv = *defm; *unaff = ST_FLAGS & ~*defm; return; }
    }
    if (*r == 0 && o->op != G_BLSMSK) *defv |= EF_ZF;
    if (*r & s) *defv |= EF_SF;
}
static void run_gops(const gop_t *ops, size_t n, uint64_t seed) {
    t_rng r; t_rng_seed(&r, seed);
    static const uint64_t edge[] = { 0, 1, 2, 0x80000000ull, 0x7fffffffull, 0xffffffffull, 0x8000000000000000ull, ~0ull, 0x00ff00ff00ff00f0ull };
    for (size_t k = 0; k < n; k++) {
        agg_t a; agg_init(&a, "%s (result, defined flags, unaffected flags)", ops[k].n);
        for (int it = 0; it < 600; it++) {
            gctx g; memset(&g, 0, sizeof g);
            g.a = (it % 5 == 0) ? edge[t_rng_range(&r, T_COUNTOF(edge))] : t_rng_next(&r) >> t_rng_range(&r, 64);
            g.b = (it % 7 == 0) ? edge[t_rng_range(&r, T_COUNTOF(edge))] : t_rng_next(&r);
            g.c = t_rng_next(&r);
            if (ops[k].op == G_BEXTR && (it & 1)) g.b = (t_rng_next(&r) & ~0xffffull) | (t_rng_range(&r, 80) << 8) | t_rng_range(&r, 80);
            if (ops[k].op == G_BZHI && (it & 1)) g.b = (t_rng_next(&r) & ~0xffull) | t_rng_range(&r, 80);
            g.fl_in = 0x2 | ((it & 1) ? ST_FLAGS : 0) | ((it & 2) ? EF_CF : 0) | ((it & 4) ? EF_OF : 0);
            if (ops[k].w == 32) { g.a &= 0xffffffffull; g.b &= 0xffffffffull; g.c &= 0xffffffffull; }
            gctx o = g; ops[k].f(&o);
            uint64_t wr, wr2; unsigned dm, dv, un;
            gmodel(&ops[k], &g, &wr, &wr2, &dm, &dv, &un);
            uint64_t M = wmask(ops[k].w);
            a.n++;
            unsigned fo = (unsigned)o.fl_out, fi = (unsigned)g.fl_in;
            if ((o.r & M) != wr || ((o.r2 & M) != wr2) || (fo & dm) != dv || (fo & un) != (fi & un))
                agg_fail(&a, "a=0x%llx b=0x%llx c=0x%llx fl_in=0x%x: r=0x%llx r2=0x%llx fl=0x%x; want r=0x%llx r2=0x%llx flags[0x%x]=0x%x unaffected[0x%x]",
                         (unsigned long long)g.a, (unsigned long long)g.b, (unsigned long long)g.c, fi, (unsigned long long)(o.r & M),
                         (unsigned long long)(o.r2 & M), fo & ST_FLAGS, (unsigned long long)wr, (unsigned long long)wr2, dm, dv, un);
        }
        AGG_END(&a);
    }
}
TEST(bmi1_andn_bextr_blsi_blsmsk_blsr_tzcnt) {
    NEED(BMI1);
    static const gop_t ops[] = {
        { b_andn32_bmi, G_ANDN, 32, 0, "andn r32" }, { b_bextr32_bmi, G_BEXTR, 32, 0, "bextr r32" }, { b_blsi32_bmi, G_BLSI, 32, 0, "blsi r32" },
        { b_blsmsk32_bmi, G_BLSMSK, 32, 0, "blsmsk r32" }, { b_blsr32_bmi, G_BLSR, 32, 0, "blsr r32" }, { b_tzcnt32_bmi, G_TZCNT, 32, 0, "tzcnt r32" },
#ifdef T_X64
        { b_andn64_bmi, G_ANDN, 64, 0, "andn r64" }, { b_bextr64_bmi, G_BEXTR, 64, 0, "bextr r64" }, { b_blsi64_bmi, G_BLSI, 64, 0, "blsi r64" },
        { b_blsmsk64_bmi, G_BLSMSK, 64, 0, "blsmsk r64" }, { b_blsr64_bmi, G_BLSR, 64, 0, "blsr r64" }, { b_tzcnt64_bmi, G_TZCNT, 64, 0, "tzcnt r64" },
#endif
    };
    run_gops(ops, T_COUNTOF(ops), 0xB1);
#ifndef T_X64
    t_note("64-bit operand forms exist only in 64-bit mode: covered by the x64 build");
#endif
}
TEST(bmi2_bzhi_mulx_pdep_pext_rorx_shiftx) {
    NEED(BMI2);
    static const gop_t ops[] = {
        { b_bzhi32_bmi, G_BZHI, 32, 0, "bzhi r32" }, { b_mulx32_bmi, G_MULX, 32, 0, "mulx r32 (lo, hi)" },
        { b_mulx32s_bmi, G_MULXS, 32, 0, "mulx r32 same dest -> high half" }, { b_pdep32_bmi, G_PDEP, 32, 0, "pdep r32" },
        { b_pext32_bmi, G_PEXT, 32, 0, "pext r32" }, { b_sarx32_bmi, G_SARX, 32, 0, "sarx r32" }, { b_shlx32_bmi, G_SHLX, 32, 0, "shlx r32" },
        { b_shrx32_bmi, G_SHRX, 32, 0, "shrx r32" }, { b_rorx32_0_bmi, G_RORX, 32, 0, "rorx r32 $0" }, { b_rorx32_1_bmi, G_RORX, 32, 1, "rorx r32 $1" },
        { b_rorx32_13_bmi, G_RORX, 32, 13, "rorx r32 $13" }, { b_rorx32_31_bmi, G_RORX, 32, 31, "rorx r32 $31" },
        { b_rorx32_33_bmi, G_RORX, 32, 33, "rorx r32 $33 (count masked to 5 bits)" },
#ifdef T_X64
        { b_bzhi64_bmi, G_BZHI, 64, 0, "bzhi r64" }, { b_mulx64_bmi, G_MULX, 64, 0, "mulx r64 (lo, hi)" },
        { b_mulx64s_bmi, G_MULXS, 64, 0, "mulx r64 same dest -> high half" }, { b_pdep64_bmi, G_PDEP, 64, 0, "pdep r64" },
        { b_pext64_bmi, G_PEXT, 64, 0, "pext r64" }, { b_sarx64_bmi, G_SARX, 64, 0, "sarx r64" }, { b_shlx64_bmi, G_SHLX, 64, 0, "shlx r64" },
        { b_shrx64_bmi, G_SHRX, 64, 0, "shrx r64" }, { b_rorx64_0_bmi, G_RORX, 64, 0, "rorx r64 $0" }, { b_rorx64_1_bmi, G_RORX, 64, 1, "rorx r64 $1" },
        { b_rorx64_13_bmi, G_RORX, 64, 13, "rorx r64 $13" }, { b_rorx64_63_bmi, G_RORX, 64, 63, "rorx r64 $63" },
        { b_rorx64_65_bmi, G_RORX, 64, 65, "rorx r64 $65 (count masked to 6 bits)" },
#endif
    };
    run_gops(ops, T_COUNTOF(ops), 0xB2);
}
TEST(lzcnt_semantics) {
    NEED(LZCNT);
    static const gop_t ops[] = { { b_lzcnt32_bmi, G_LZCNT, 32, 0, "lzcnt r32" },
#ifdef T_X64
        { b_lzcnt64_bmi, G_LZCNT, 64, 0, "lzcnt r64" },
#endif
    };
    run_gops(ops, T_COUNTOF(ops), 0xB3);
}
TEST(adx_adcx_adox_flags_and_chains) {
    NEED(ADX);
    static const gop_t ops[] = { { b_adcx32_adx, G_ADCX, 32, 0, "adcx r32 (CF only)" }, { b_adox32_adx, G_ADOX, 32, 0, "adox r32 (OF only)" },
#ifdef T_X64
        { b_adcx64_adx, G_ADCX, 64, 0, "adcx r64" }, { b_adox64_adx, G_ADOX, 64, 0, "adox r64" },
#endif
    };
    run_gops(ops, T_COUNTOF(ops), 0xAD);
    /* two interleaved carry chains (the reason ADX exists): sum of 4 x 128-bit (32-bit limbs) built by
     * alternating adcx/adox must equal the schoolbook result */
    agg_t a; agg_init(&a, "adcx/adox interleaved independent carry chains");
    t_rng r; t_rng_seed(&r, 0xC4A1);
    for (int it = 0; it < 300; it++) {
        uint32_t x[4], y[4], u[4], v[4], s1[4], s2[4];
        for (int i = 0; i < 4; i++) { x[i] = t_rng_u32(&r); y[i] = t_rng_u32(&r); u[i] = t_rng_u32(&r); v[i] = t_rng_u32(&r); }
        if (it & 1) for (int i = 0; i < 4; i++) { x[i] = 0xffffffffu; u[i] = 0xffffffffu; }
        unsigned cf = 0, of = 0;
        for (int i = 0; i < 4; i++) {   /* chain 1 via CF (adcx), chain 2 via OF (adox), alternating */
            gctx g = { x[i], y[i], 0, 0, 0, 0x2u | (cf ? EF_CF : 0) | (of ? EF_OF : 0), 0 };
            b_adcx32_adx(&g); s1[i] = (uint32_t)g.r; cf = (g.fl_out & EF_CF) != 0;
            a.n++; if (((g.fl_out & EF_OF) != 0) != (of != 0)) agg_fail(&a, "adcx changed OF");
            gctx h = { u[i], v[i], 0, 0, 0, 0x2u | (cf ? EF_CF : 0) | (of ? EF_OF : 0), 0 };
            b_adox32_adx(&h); s2[i] = (uint32_t)h.r; of = (h.fl_out & EF_OF) != 0;
            a.n++; if (((h.fl_out & EF_CF) != 0) != (cf != 0)) agg_fail(&a, "adox changed CF");
        }
        uint64_t c1 = 0, c2 = 0; int ok = 1;
        for (int i = 0; i < 4; i++) {
            c1 += (uint64_t)x[i] + y[i]; ok &= (uint32_t)c1 == s1[i]; c1 >>= 32;
            c2 += (uint64_t)u[i] + v[i]; ok &= (uint32_t)c2 == s2[i]; c2 >>= 32;
        }
        ok &= (unsigned)c1 == cf && (unsigned)c2 == of;
        a.n++; if (!ok) agg_fail(&a, "multi-precision chain %d mismatch", it);
    }
    AGG_END(&a);
}

/* =========================================================================== DAZ beyond arithmetic
 * SDM vol.1 10.2.3.4: with MXCSR.DAZ=1 every denormal SOURCE operand of an SSE/AVX FP instruction is converted
 * to a zero of the same sign before the operation and #D is not reported. That includes compares, (U)COMIS,
 * MIN/MAX (which then see two zeros and return src2 - the converted value), SQRT (sqrt(-denormal) = -0, no #I)
 * and conversions. Hand-worked vectors; the DAZ=0 rows double as a control (DE must be set). */
TEST(avx_daz_compare_minmax_sqrt_convert) {
    NEED(AVX);
    static const uint32_t A[8] = { 0x00000001, 0x80000001, 0x00400000, 0x3f800000, 0x00000000, 0x80000001, 0x00000001, 0x00000000 };
    static const uint32_t B[8] = { 0x00000000, 0x00000000, 0x00400000, 0x00000001, 0x3f800000, 0x3f800000, 0x80000001, 0x00000001 };
    static const uint32_t EQ[2][8] = { { 0, 0, ~0u, 0, 0, 0, 0, 0 }, { ~0u, ~0u, ~0u, 0, 0, 0, ~0u, ~0u } };
    static const uint32_t MN[2][8] = { { 0x00000000, 0x80000001, 0x00400000, 0x00000001, 0x00000000, 0x80000001, 0x80000001, 0x00000000 },
                                       { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000 } };
    static const uint32_t MXv[2][8] = { { 0x00000001, 0x00000000, 0x00400000, 0x3f800000, 0x3f800000, 0x3f800000, 0x00000001, 0x00000001 },
                                        { 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x80000000, 0x00000000 } };
    agg_t a; agg_init(&a, "DAZ applied to vcmpps/vminps/vmaxps/vsqrtps/vcvtps2pd/vucomiss sources");
    for (int daz = 0; daz < 2; daz++) {
        uint32_t mx = MX_DEFAULT | (daz ? MX_DAZ : 0), wde = daz ? 0 : MX_DE;
        char ctx[64]; v256 w;
        vctx c; vctx_pattern(&c, 0xDA2u + (uint32_t)daz); c.mx_in = mx;
        memcpy(c.r[1].d, A, 32); memcpy(c.r[2].d, B, 32);
        vctx o = c; vexec(&a, vcmpps_tab[0], &o, KEEP12);
        memcpy(w.d, EQ[daz], 32); snprintf(ctx, sizeof ctx, "vcmpeqps DAZ=%d", daz); agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        agg_eq(&a, o.mx_out & MX_FLAGS, wde, ctx);
        o = c; vexec(&a, vminps_y_avx, &o, KEEP12);
        memcpy(w.d, MN[daz], 32); snprintf(ctx, sizeof ctx, "vminps DAZ=%d", daz); agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        agg_eq(&a, o.mx_out & MX_FLAGS, wde, ctx);
        o = c; vexec(&a, vmaxps_y_avx, &o, KEEP12);
        memcpy(w.d, MXv[daz], 32); snprintf(ctx, sizeof ctx, "vmaxps DAZ=%d", daz); agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        agg_eq(&a, o.mx_out & MX_FLAGS, wde, ctx);
        /* sqrt: -denormal -> -0 (DAZ) or default NaN + #I (no DAZ); +denormal -> +0 or an exact-ish normal */
        o = c; for (int i = 0; i < 8; i++) o.r[1].d[i] = (i & 1) ? 0x80000001u : 0x3f800000u;
        vexec(&a, vsqrtps_y_avx, &o, KEEP12);
        for (int i = 0; i < 8; i++) w.d[i] = (i & 1) ? (daz ? 0x80000000u : 0xffc00000u) : 0x3f800000u;
        snprintf(ctx, sizeof ctx, "vsqrtps(-denormal) DAZ=%d", daz); agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        /* without DAZ: #I (DE alongside is not pinned: the invalid operand is reported, hardware may stop there) */
        agg_eq(&a, o.mx_out & (MX_FLAGS & ~MX_DE), daz ? 0 : MX_IE, ctx);
        if (daz) agg_eq(&a, o.mx_out & MX_DE, 0, "vsqrtps DAZ: no #D");
        /* vcvtps2pd: denormal -> signed zero under DAZ, exact widening otherwise */
        o = c; vexec(&a, vcvtps2pd_avx, &o, KEEP12);
        for (int i = 0; i < 4; i++) w.q[i] = (daz && f_isdenorm(FMT32, A[i])) ? ((uint64_t)(A[i] >> 31) << 63) : f_convert(FMT32, FMT64, A[i], RN, NULL);
        snprintf(ctx, sizeof ctx, "vcvtps2pd DAZ=%d", daz); agg_mem(&a, &o.r[0], &w, 32, 8, ctx);
        agg_eq(&a, o.mx_out & MX_FLAGS, wde, ctx);
        /* vucomiss(denormal, +0): GT without DAZ, EQ with DAZ */
        o = c; o.r[1].d[0] = 0x00000001u; o.r[2].d[0] = 0;
        vexec(&a, vucomiss_avx, &o, KEEP12);
        snprintf(ctx, sizeof ctx, "vucomiss(2^-149, 0) DAZ=%d", daz);
        agg_eq(&a, o.fl_out & (EF_ZF | EF_PF | EF_CF), daz ? EF_ZF : 0, ctx);
        agg_eq(&a, o.mx_out & MX_FLAGS, wde, ctx);
    }
    AGG_END(&a);
}

/* =========================================================================== F16C denormal controls
 * SDM vol.1 14.4.1 (half-precision conversions): VCVTPS2PH honours MXCSR.DAZ for its single-precision source
 * and ignores MXCSR.FTZ (half denormal results are delivered); VCVTPH2PS is exact - half denormals become
 * normal singles regardless of DAZ. */
TEST(f16c_denormal_daz_ftz_treatment) {
    NEED(F16C);
    agg_t a; agg_init(&a, "vcvtps2ph DAZ honoured / FTZ ignored; vcvtph2ps ignores DAZ");
    static const uint32_t din[8] = { 0x00000001, 0x80000001, 0x00400000, 0x807fffff, 0x3f800000, 0x00000000, 0x80000000, 0x00000001 };
    for (int daz = 0; daz < 2; daz++) {
        vctx c; vctx_pattern(&c, 0xF0u + (uint32_t)daz); c.mx_in = MX_DEFAULT | (daz ? MX_DAZ : 0);
        memcpy(c.r[1].d, din, 32);
        vctx o = c; vexec(&a, vcvtps2ph_tab[2], &o, KEEP12);      /* imm 2: round up */
        v256 w = V_ZERO; unsigned fl = 0;
        for (int i = 0; i < 8; i++) {
            uint32_t x = din[i];
            if (f_isdenorm(FMT32, x)) { if (daz) x &= 0x80000000u; else fl |= MX_DE; }
            w.w[i] = f32_to_half(x, RU, &fl);
        }
        char ctx[48]; snprintf(ctx, sizeof ctx, "vcvtps2ph RU single denormals DAZ=%d", daz);
        agg_mem(&a, &o.r[0], &w, 32, 2, ctx);
        agg_eq(&a, o.r[0].w[0], daz ? 0x0000 : 0x0001, "vcvtps2ph RU(2^-149): 0x0001, or 0 when DAZ flushes the source");
        agg_eq(&a, o.mx_out & (MX_IE | MX_DE | MX_OE | MX_UE | MX_PE), fl, ctx);
    }
    for (int ftz = 0; ftz < 2; ftz++) {
        static const uint32_t fin[8] = { 0x33800000, 0x38000000, 0x387fc000, 0x34000001, 0x80000000, 0xb8000000, 0x3f800000, 0x38800000 };
        static const uint16_t fw[8] = { 0x0001, 0x0200, 0x03ff, 0x0002, 0x8000, 0x8200, 0x3c00, 0x0400 };
        vctx c; vctx_pattern(&c, 0xF2u + (uint32_t)ftz); c.mx_in = MX_DEFAULT | (ftz ? MX_FTZ : 0);
        memcpy(c.r[1].d, fin, 32);
        vctx o = c; vexec(&a, vcvtps2ph_tab[0], &o, KEEP12);
        v256 w = V_ZERO; memcpy(w.w, fw, 16);
        char ctx[48]; snprintf(ctx, sizeof ctx, "vcvtps2ph RN half-denormal results FTZ=%d", ftz);
        agg_mem(&a, &o.r[0], &w, 32, 2, ctx);
        agg_eq(&a, o.mx_out & MX_FLAGS, MX_UE | MX_PE, ctx);       /* only lane 3 is inexact (and tiny) */
    }
    static const uint16_t hin[8] = { 0x0001, 0x8200, 0x03ff, 0x0400, 0x83ff, 0x0000, 0x7c00, 0x3c00 };
    for (int daz = 0; daz < 2; daz++) {
        vctx c; vctx_pattern(&c, 0xF4u + (uint32_t)daz); c.mx_in = MX_DEFAULT | (daz ? MX_DAZ : 0);
        memcpy(c.r[1].w, hin, 16);
        vctx o = c; vexec(&a, h_ph2ps_avx, &o, KEEP12);
        v256 w; for (int i = 0; i < 8; i++) w.d[i] = half_to_f32(hin[i]);
        char ctx[48]; snprintf(ctx, sizeof ctx, "vcvtph2ps half denormals DAZ=%d", daz);
        agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        if (o.mx_out & MX_FLAGS) t_note("vcvtph2ps of half denormals set MXCSR flags 0x%02x (DAZ=%d)", o.mx_out & MX_FLAGS, daz);
    }
    AGG_END(&a);
}

/* =========================================================================== VEX memory operands: no alignment
 * SDM vol.1 14.9: VEX-encoded instructions with a memory operand (other than explicitly aligned moves such as
 * VMOVAPS/VMOVDQA/VMOVNT*) do not require alignment; legacy SSE arithmetic requires 16-byte alignment.
 * Runs in a child: a fault here would otherwise take down the whole exe. Operands are placed at +1, +3 and
 * straddling a page boundary (both pages committed); results must equal the register forms. */
VOP(um_vaddps_avx, "vaddps (" RC "), %%ymm1, %%ymm0")            VOP(um_vmulpd_avx, "vmulpd (" RC "), %%ymm1, %%ymm0")
VOP(um_vpermilps_avx, "vpermilps (" RC "), %%ymm1, %%ymm0")      VOP(um_vcmpltps_avx, "vcmpps $1, (" RC "), %%ymm1, %%ymm0")
VOP(um_vptest_avx, "vptest (" RC "), %%ymm1")                    VOP(um_vbcstss_avx, "vbroadcastss (" RC "), %%ymm0")
VOP(um_vandnpd_avx, "vandnpd (" RC "), %%ymm1, %%ymm0")          VOP(um_vdivss_avx, "vdivss (" RC "), %%xmm1, %%xmm0")
VOP(um_fmadd231ps_fma, "vfmadd231ps (" RC "), %%ymm1, %%ymm0")   VOP(um_fmadd132pd_fma, "vfmadd132pd (" RC "), %%ymm1, %%ymm0")
VOP(um_fnmsub213ss_fma, "vfnmsub213ss (" RC "), %%xmm1, %%xmm0")
VOP(um_vpaddd_avx, "vpaddd (" RC "), %%ymm1, %%ymm0")            VOP(um_vpshufb_avx, "vpshufb (" RC "), %%ymm1, %%ymm0")
VOP(um_vpmaddwd_avx, "vpmaddwd (" RC "), %%ymm1, %%ymm0")        VOP(um_vpermd_avx, "vpermd (" RC "), %%ymm1, %%ymm0")
VOP(um_vph2ps_avx, "vcvtph2ps (" RC "), %%ymm0")
VOP(um_vdivss_ref_avx, "vdivss %%xmm2, %%xmm1, %%xmm0")
typedef struct umop { const char *n; vop_fn mem, reg; int bytes, src_reg, need; } umop;   /* need: 0 AVX, 1 FMA, 2 AVX2, 3 F16C */
static const umop um_ops[] = {
    { "vaddps m256", um_vaddps_avx, vaddps_y_avx, 32, 2, 0 }, { "vmulpd m256", um_vmulpd_avx, vmulpd_y_avx, 32, 2, 0 },
    { "vpermilps m256 control", um_vpermilps_avx, vpermilps_v_avx, 32, 2, 0 }, { "vcmpltps m256", um_vcmpltps_avx, vcmpps_0_1_avx, 32, 2, 0 },
    { "vptest m256", um_vptest_avx, vptest_avx, 32, 2, 0 }, { "vandnpd m256", um_vandnpd_avx, vandnpd_avx, 32, 2, 0 },
    { "vdivss m32", um_vdivss_avx, um_vdivss_ref_avx, 4, 2, 0 },
    { "vfmadd231ps m256", um_fmadd231ps_fma, f_fmadd231ps_y_avx, 32, 2, 1 }, { "vfmadd132pd m256", um_fmadd132pd_fma, f_fmadd132pd_y_avx, 32, 2, 1 },
    { "vfnmsub213ss m32", um_fnmsub213ss_fma, f_fnmsub213ss_avx, 4, 2, 1 },
    { "vpaddd m256", um_vpaddd_avx, i_paddd_avx, 32, 2, 2 }, { "vpshufb m256", um_vpshufb_avx, vpshufb_y_avx, 32, 2, 2 },
    { "vpmaddwd m256", um_vpmaddwd_avx, i_pmaddwd_avx, 32, 2, 2 }, { "vpermd m256", um_vpermd_avx, vpermd_avx, 32, 2, 2 },
    { "vcvtph2ps m128", um_vph2ps_avx, h_ph2ps_avx, 16, 1, 3 },
};
static int um_usable(int need) {
    return need == 0 ? cpu_has(F_AVX) : need == 1 ? cpu_has(F_FMA) : need == 2 ? cpu_has(F_AVX2) : cpu_has(F_F16C);
}
CHILD(avx_vex_unaligned_mem) {
    size_t pg = page_size();
    uint8_t *b = (uint8_t *)VirtualAlloc(NULL, 2 * pg, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!b) { printf("NOALLOC=1\n"); return 4; }
    t_rng r; t_rng_seed(&r, 0x0A1);
    static const char *const where[3] = { "+1", "+3", "page-split" };
    int bad = 0, n = 0;
    for (size_t k = 0; k < T_COUNTOF(um_ops); k++) {
        if (!um_usable(um_ops[k].need)) continue;
        for (int wi = 0; wi < 3; wi++) {
            size_t off = wi == 0 ? 64 + 1 : wi == 1 ? 128 + 3 : pg - 13;
            for (int it = 0; it < 8; it++) {
                vctx c; vctx_pattern(&c, (uint32_t)(k * 31 + (size_t)it));
                gen_f32v(&r, &c.r[0], 1); gen_f32v(&r, &c.r[1], 1);
                v256 m; gen_f32v(&r, &m, 1);
                if (um_ops[k].need == 2) { gen_int(&r, &c.r[1], 4); gen_int(&r, &m, 4); }
                memcpy(b + off, &m, (size_t)um_ops[k].bytes);
                c.mem = (uintptr_t)(b + off);
                vctx om = c, orf = c;
                memset(&orf.r[um_ops[k].src_reg], 0, 32); memcpy(&orf.r[um_ops[k].src_reg], &m, (size_t)um_ops[k].bytes);
                um_ops[k].mem(&om);
                um_ops[k].reg(&orf);
                n++;
                if (memcmp(&om.r[0], &orf.r[0], 32) || om.fl_out != orf.fl_out || om.mx_out != orf.mx_out) {
                    bad++;
                    printf("BAD=%s at %s (iteration %d)\n", um_ops[k].n, where[wi], it);
                }
            }
        }
    }
    VirtualFree(b, 0, MEM_RELEASE);
    printf("COMPARED=%d\nMISMATCHES=%d\n", n, bad);
    return 0;
}
TEST(avx_vex_memory_operands_unaligned_and_page_split) {
    NEED(AVX);
    static t_child_result r; char v[96] = "";
    REQUIRE(t_child_run("avx_vex_unaligned_mem", NULL, 10000, &r));
    CHECKF(!r.timed_out && r.exc_code == 0 && r.exit_code == 0,
           "VEX op with an unaligned / page-split memory operand faulted: exit 0x%lx exc 0x%lx timed_out %d",
           (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.timed_out);
    if (t_child_find(&r, "BAD=", v, sizeof v)) CHECKF(0, "memory form differs from the register form: %s", v);
    CHECKF(t_child_find(&r, "MISMATCHES=", v, sizeof v) && !strcmp(v, "0"), "mismatches: %s", v);
    if (t_child_find(&r, "COMPARED=", v, sizeof v)) t_info("vex_unaligned_compared", "%s", v);
}

/* =========================================================================== VLDMXCSR / VSTMXCSR */
static T_NOINLINE void vmx_roundtrip_avx(const uint32_t *in, uint32_t *legacy, uint32_t *vex) {
    uint32_t save;
    __asm__ volatile("stmxcsr %0\n\tvldmxcsr %1\n\tstmxcsr %2\n\tvstmxcsr %3\n\tldmxcsr %0"
                     : "=m"(save) : "m"(*in), "m"(*legacy), "m"(*vex) : "memory");
}
/* vldmxcsr <rc>, then vaddps 1 + 2^-24 in all lanes: RU must round up */
static T_NOINLINE void vmx_rc_add_avx(const uint32_t *mx, v256 *out) {
    static const uint32_t one = 0x3f800000u, tiny = 0x33800000u;
    uint32_t save;
    __asm__ volatile("stmxcsr %0\n\tvldmxcsr %1\n\tvbroadcastss %2, %%ymm1\n\tvbroadcastss %3, %%ymm2\n\t"
                     "vaddps %%ymm2, %%ymm1, %%ymm0\n\tvmovdqu %%ymm0, %4\n\tldmxcsr %0\n\tvzeroupper"
                     : "=m"(save) : "m"(*mx), "m"(one), "m"(tiny), "m"(*out) : "xmm0", "xmm1", "xmm2", "memory");
}
CHILD(avx_vldmxcsr_reserved) {
    static const uint32_t bad = 0x00011F80u;    /* bit 16: reserved on every processor (MXCSR_MASK bits 16-31 are 0) */
    uint32_t l = 0, v = 0;
    printf("STEP=vldmxcsr\n"); fflush(stdout);
    vmx_roundtrip_avx(&bad, &l, &v);
    printf("NO_FAULT=1 READ=0x%08x\n", l);
    return 0;
}
TEST(avx_vldmxcsr_vstmxcsr) {
    NEED(AVX);
    agg_t a; agg_init(&a, "VLDMXCSR/VSTMXCSR equal LDMXCSR/STMXCSR; RC honoured");
    for (unsigned rc = 0; rc < 4; rc++)
        for (unsigned ctl = 0; ctl < 8; ctl++) {
            /* ctl: bit0 DAZ, bit1 FTZ, bit2 all masks clear (no flags set, so nothing is raised) */
            uint32_t in = ((ctl & 4) ? 0 : 0x1F80u) | (rc << 13) | ((ctl & 1) ? MX_DAZ : 0) | ((ctl & 2) ? MX_FTZ : 0);
            if (!(ctl & 4)) in |= (rc * 7 + ctl) & MX_FLAGS;       /* sticky flags only with everything masked */
            uint32_t l = 0, v = 0;
            vmx_roundtrip_avx(&in, &l, &v);
            char ctx[48]; snprintf(ctx, sizeof ctx, "load 0x%08x", in);
            agg_eq(&a, l, in, ctx); agg_eq(&a, v, l, "vstmxcsr == stmxcsr");
        }
    static const uint32_t want[4] = { 0x3f800000u, 0x3f800000u, 0x3f800001u, 0x3f800000u };
    for (int rc = 0; rc < 4; rc++) {
        uint32_t mx = MX_RC(rc); v256 o;
        vmx_rc_add_avx(&mx, &o);
        char ctx[48]; snprintf(ctx, sizeof ctx, "vaddps 1+2^-24 after vldmxcsr %s", rc_name[rc]);
        for (int i = 0; i < 8; i++) if (!agg_eq(&a, o.d[i], want[rc], ctx)) break;
    }
    AGG_END(&a);
    static t_child_result r;
    REQUIRE(t_child_run("avx_vldmxcsr_reserved", NULL, 10000, &r));
    CHECKF(!r.timed_out && (r.exc_code == 0xC0000005u || r.exc_code == 0xC0000096u),
           "VLDMXCSR with reserved bit 16 set: exception 0x%08lx exit 0x%lx (want #GP as ACCESS_VIOLATION or PRIVILEGED_INSTRUCTION)",
           (unsigned long)r.exc_code, (unsigned long)r.exit_code);
}

/* =========================================================================== VEX.128 element insert / extract */
#define vpextrb_INSN "vpextrb %1, %%xmm1, %%edx"
#define vpextrw_INSN "vpextrw %1, %%xmm1, %%edx"
#define vpextrd_INSN "vpextrd %1, %%xmm1, %%edx"
#define vextractps_INSN "vextractps %1, %%xmm1, %%edx"
#define vpinsrb_INSN "vpinsrb %1, %%edx, %%xmm1, %%xmm0"
#define vpinsrw_INSN "vpinsrw %1, %%edx, %%xmm1, %%xmm0"
#define vpinsrd_INSN "vpinsrd %1, %%edx, %%xmm1, %%xmm0"
#define vpextrbm_INSN "vpextrb %1, %%xmm1, (" RC ")"
#define vpinsrwm_INSN "vpinsrw %1, (" RC "), %%xmm1, %%xmm0"
#define vinsertps_INSN "vinsertps %1, %%xmm2, %%xmm1, %%xmm0"
#define vinsertpsm_INSN "vinsertps %1, (" RC "), %%xmm1, %%xmm0"
DEF_IMM32(vpextrb) DEF_IMM32(vpextrw) DEF_IMM32(vpextrd) DEF_IMM32(vextractps) DEF_IMM32(vpinsrb) DEF_IMM32(vpinsrw)
DEF_IMM32(vpinsrd) DEF_IMM32(vpextrbm) DEF_IMM32(vpinsrwm) DEF_IMM256(vinsertps) DEF_IMM256(vinsertpsm)
TEST(avx_pextr_pinsr_insertps_all_imm) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x1E5);
    static uint8_t mem[16];
    agg_t a; agg_init(&a, "vpextrb/w/d, vextractps, vpinsrb/w/d, vinsertps (imm, ignored imm bits, upper zero)");
    for (int imm = 0; imm < 32; imm++) {
        vctx c; vctx_pattern(&c, (uint32_t)imm); t_rng_fill(&r, &c.r[1], 32); c.g_in = t_rng_u32(&r);
        char ctx[48]; vctx o; v256 w;
        snprintf(ctx, sizeof ctx, "vpextrb imm=%d", imm);    o = c; vexec(&a, vpextrb_tab[imm], &o, KEEP12);    agg_eq(&a, o.g_out, c.r[1].b[imm & 15], ctx);
        snprintf(ctx, sizeof ctx, "vpextrw imm=%d", imm);    o = c; vexec(&a, vpextrw_tab[imm], &o, KEEP12);    agg_eq(&a, o.g_out, c.r[1].w[imm & 7], ctx);
        snprintf(ctx, sizeof ctx, "vpextrd imm=%d", imm);    o = c; vexec(&a, vpextrd_tab[imm], &o, KEEP12);    agg_eq(&a, o.g_out, c.r[1].d[imm & 3], ctx);
        snprintf(ctx, sizeof ctx, "vextractps imm=%d", imm); o = c; vexec(&a, vextractps_tab[imm], &o, KEEP12); agg_eq(&a, o.g_out, c.r[1].d[imm & 3], ctx);
        w = V_ZERO; memcpy(&w, &c.r[1], 16); w.b[imm & 15] = (uint8_t)c.g_in;
        snprintf(ctx, sizeof ctx, "vpinsrb imm=%d", imm); o = c; vexec(&a, vpinsrb_tab[imm], &o, KEEP12); agg_mem(&a, &o.r[0], &w, 32, 1, ctx);
        w = V_ZERO; memcpy(&w, &c.r[1], 16); w.w[imm & 7] = (uint16_t)c.g_in;
        snprintf(ctx, sizeof ctx, "vpinsrw imm=%d", imm); o = c; vexec(&a, vpinsrw_tab[imm], &o, KEEP12); agg_mem(&a, &o.r[0], &w, 32, 2, ctx);
        w = V_ZERO; memcpy(&w, &c.r[1], 16); w.d[imm & 3] = c.g_in;
        snprintf(ctx, sizeof ctx, "vpinsrd imm=%d", imm); o = c; vexec(&a, vpinsrd_tab[imm], &o, KEEP12); agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        /* memory forms: vpextrb writes exactly one byte, vpinsrw reads exactly a word */
        t_rng_fill(&r, mem, sizeof mem); uint8_t ref[16]; memcpy(ref, mem, 16); ref[5] = c.r[1].b[imm & 15];
        o = c; o.mem = (uintptr_t)(mem + 5); vexec(&a, vpextrbm_tab[imm], &o, KEEP12);
        snprintf(ctx, sizeof ctx, "vpextrb m8 imm=%d", imm); agg_mem(&a, mem, ref, 16, 1, ctx);
        w = V_ZERO; memcpy(&w, &c.r[1], 16); memcpy(&w.w[imm & 7], mem + 7, 2);
        o = c; o.mem = (uintptr_t)(mem + 7); vexec(&a, vpinsrwm_tab[imm], &o, KEEP12);
        snprintf(ctx, sizeof ctx, "vpinsrw m16 imm=%d", imm); agg_mem(&a, &o.r[0], &w, 32, 2, ctx);
    }
    for (int imm = 0; imm < 256; imm++) {
        vctx c; vctx_pattern(&c, (uint32_t)imm + 77); t_rng_fill(&r, &c.r[1], 32); t_rng_fill(&r, &c.r[2], 32);
        t_rng_fill(&r, mem, sizeof mem); c.mem = (uintptr_t)(mem + 3);
        for (int m = 0; m < 2; m++) {
            v256 w = V_ZERO; memcpy(&w, &c.r[1], 16);
            uint32_t src; if (m) memcpy(&src, mem + 3, 4); else src = c.r[2].d[(imm >> 6) & 3];   /* m32: imm[7:6] ignored */
            w.d[(imm >> 4) & 3] = src;
            for (int j = 0; j < 4; j++) if ((imm >> j) & 1) w.d[j] = 0;
            vctx o = c; vexec(&a, (m ? vinsertpsm_tab : vinsertps_tab)[imm], &o, KEEP12);
            char ctx[48]; snprintf(ctx, sizeof ctx, "vinsertps %s imm=0x%02x", m ? "m32" : "xmm", imm);
            agg_mem(&a, &o.r[0], &w, 32, 4, ctx);
        }
    }
    AGG_END(&a);
}

/* =========================================================================== VEX scalar merge forms
 * VEX scalar ops: bits 31:0 (63:0) = op(src1, src2), 127:32 (127:64) copied from src1 (not from the destination),
 * 255:128 zeroed. Reference: legacy scalar SSE under the same MXCSR, flags included. */
VOP(s_vaddss_avx, "vaddss %%xmm2, %%xmm1, %%xmm0")     VOP(s_vsubsd_avx, "vsubsd %%xmm2, %%xmm1, %%xmm0")
VOP(s_vmulss_avx, "vmulss %%xmm2, %%xmm1, %%xmm0")     VOP(s_vdivsd_avx, "vdivsd %%xmm2, %%xmm1, %%xmm0")
VOP(s_vminss_avx, "vminss %%xmm2, %%xmm1, %%xmm0")     VOP(s_vmaxsd_avx, "vmaxsd %%xmm2, %%xmm1, %%xmm0")
VOP(s_vsqrtss_avx, "vsqrtss %%xmm2, %%xmm1, %%xmm0")   VOP(s_vsqrtsd_avx, "vsqrtsd %%xmm2, %%xmm1, %%xmm0")
VOP(s_vcvtss2sd_avx, "vcvtss2sd %%xmm2, %%xmm1, %%xmm0") VOP(s_vcvtsd2ss_avx, "vcvtsd2ss %%xmm2, %%xmm1, %%xmm0")
VOP(s_vrcpss_avx, "vrcpss %%xmm2, %%xmm1, %%xmm0")     VOP(s_vrsqrtss_avx, "vrsqrtss %%xmm2, %%xmm1, %%xmm0")
VOP(s_vroundsd_avx, "vroundsd $4, %%xmm2, %%xmm1, %%xmm0") VOP(s_vmovsd_avx, "vmovsd %%xmm2, %%xmm1, %%xmm0")
VOP(s_vmovlhps_avx, "vmovlhps %%xmm2, %%xmm1, %%xmm0")
TEST(avx_scalar_merge_forms) {
    NEED(AVX);
    t_rng r; t_rng_seed(&r, 0x5CA);
    static const struct { const char *n; vop_fn f; int op, dbl; } ar[] = {
        { "vaddss", s_vaddss_avx, OP_ADD, 0 }, { "vsubsd", s_vsubsd_avx, OP_SUB, 1 }, { "vmulss", s_vmulss_avx, OP_MUL, 0 },
        { "vdivsd", s_vdivsd_avx, OP_DIV, 1 }, { "vminss", s_vminss_avx, OP_MIN, 0 }, { "vmaxsd", s_vmaxsd_avx, OP_MAX, 1 } };
    agg_t a; agg_init(&a, "VEX scalar: low = op(src1,src2), rest of 127:0 from src1, 255:128 zero");
    for (int it = 0; it < 300; it++) {
        uint32_t mx = fp_modes[it % T_COUNTOF(fp_modes)];
        vctx c; vctx_pattern(&c, (uint32_t)it); c.mx_in = mx;
        t_rng_fill(&r, &c.r[0], 32);       /* destination garbage must not leak into the result */
        gen_f32v(&r, &c.r[1], 0); gen_f32v(&r, &c.r[2], 0);
        if (it & 1) { gen_f64v(&r, &c.r[1], 0); gen_f64v(&r, &c.r[2], 0); }
        for (size_t k = 0; k < T_COUNTOF(ar); k++) {
            vctx o = c; vexec(&a, ar[k].f, &o, KEEP12);
            v256 w = V_ZERO; memcpy(&w, &c.r[1], 16); uint32_t fl = 0;
            if (ar[k].dbl) w.q[0] = sse_f64(ar[k].op, c.r[1].q[0], c.r[2].q[0], mx, &fl);
            else w.d[0] = sse_f32(ar[k].op, c.r[1].d[0], c.r[2].d[0], mx, &fl);
            agg_mem(&a, &o.r[0], &w, 32, ar[k].dbl ? 8 : 4, ar[k].n);
            agg_eq(&a, o.mx_out & MX_FLAGS, fl, ar[k].n);
        }
        /* sqrt takes its operand from src2 */
        vctx o = c; vexec(&a, s_vsqrtss_avx, &o, KEEP12);
        v256 w = V_ZERO; memcpy(&w, &c.r[1], 16); w.d[0] = sse_f32(OP_SQRT, c.r[2].d[0], 0, mx, NULL);
        agg_mem(&a, &o.r[0], &w, 32, 4, "vsqrtss (operand = src2 low)");
        o = c; vexec(&a, s_vsqrtsd_avx, &o, KEEP12);
        w = V_ZERO; memcpy(&w, &c.r[1], 16); w.q[0] = sse_f64(OP_SQRT, c.r[2].q[0], 0, mx, NULL);
        agg_mem(&a, &o.r[0], &w, 32, 8, "vsqrtsd (operand = src2 low)");
        /* conversions (DAZ: a denormal source converts as zero) */
        int rc = (int)((mx >> 13) & 3), daz = (mx & MX_DAZ) != 0, ftz = (mx & MX_FTZ) != 0;
        o = c; vexec(&a, s_vcvtss2sd_avx, &o, KEEP12);
        uint32_t s32 = c.r[2].d[0]; if (daz && f_isdenorm(FMT32, s32)) s32 &= 0x80000000u;
        w = V_ZERO; memcpy(&w, &c.r[1], 16); w.q[0] = f_convert(FMT32, FMT64, s32, rc, NULL);
        agg_mem(&a, &o.r[0], &w, 32, 8, "vcvtss2sd");
        o = c; vexec(&a, s_vcvtsd2ss_avx, &o, KEEP12);
        uint64_t s64 = c.r[2].q[0]; if (daz && f_isdenorm(FMT64, s64)) s64 &= 0x8000000000000000ull;
        w = V_ZERO; memcpy(&w, &c.r[1], 16);
        if (f_isnan(FMT64, s64) || f_isinf(FMT64, s64) || f_iszero(FMT64, s64)) w.d[0] = (uint32_t)f_convert(FMT64, FMT32, s64, rc, NULL);
        else { int sg, e; uint64_t m; f_unpack(FMT64, s64, &sg, &m, &e); w.d[0] = (uint32_t)fround_ex(FMT32, sg, u128_from(m), e, rc, ftz, NULL); }
        agg_mem(&a, &o.r[0], &w, 32, 4, "vcvtsd2ss");
        /* round (imm 4: MXCSR.RC) */
        o = c; vexec(&a, s_vroundsd_avx, &o, KEEP12);
        w = V_ZERO; memcpy(&w, &c.r[1], 16); w.q[0] = f_round_int(FMT64, (daz && f_isdenorm(FMT64, c.r[2].q[0])) ? c.r[2].q[0] & 0x8000000000000000ull : c.r[2].q[0], rc, 0, NULL);
        agg_mem(&a, &o.r[0], &w, 32, 8, "vroundsd $4");
        /* moves */
        o = c; vexec(&a, s_vmovsd_avx, &o, KEEP12);
        w = V_ZERO; w.q[0] = c.r[2].q[0]; w.q[1] = c.r[1].q[1]; agg_mem(&a, &o.r[0], &w, 32, 8, "vmovsd xmm,xmm,xmm");
        o = c; vexec(&a, s_vmovlhps_avx, &o, KEEP12);
        w = V_ZERO; w.q[0] = c.r[1].q[0]; w.q[1] = c.r[2].q[0]; agg_mem(&a, &o.r[0], &w, 32, 8, "vmovlhps");
        /* approximations: merge bits exact, low element within the SDM bound */
        for (int k = 0; k < 2; k++) {
            vctx q = c; q.mx_in = MX_DEFAULT;
            q.r[2].d[0] = ((uint32_t)(127 - 40 + t_rng_range(&r, 81)) << 23) | (t_rng_u32(&r) & 0x7fffff);
            vctx oq = q; vexec(&a, k ? s_vrsqrtss_avx : s_vrcpss_avx, &oq, KEEP12);
            v256 g = oq.r[0], ww = V_ZERO; memcpy(&ww, &q.r[1], 16);
            uint32_t low = g.d[0]; g.d[0] = 0; ww.d[0] = 0;
            agg_mem(&a, &g, &ww, 32, 4, k ? "vrsqrtss merge bits" : "vrcpss merge bits");
            double x = f32_to_double(q.r[2].d[0]), y = f32_to_double(low), ex = k ? 1.0 / sqrt(x) : 1.0 / x, e = (y - ex) / ex;
            if (e < 0) e = -e;
            a.n++;
            if (!(e <= 1.5 / 4096.0)) agg_fail(&a, "%s(0x%08x) = 0x%08x rel.err %.3g", k ? "vrsqrtss" : "vrcpss", q.r[2].d[0], low, e);
        }
    }
    AGG_END(&a);
}
