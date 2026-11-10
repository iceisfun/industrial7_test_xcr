/* simd_bench_asm.h - register-pinning asm for simd_bench: spinning workers that hold a known XMM/YMM
 * image while the context APIs and the scheduler act on them, and the AVX <-> legacy-SSE transition
 * kernels. Everything that must stay in hardware registers happens inside ONE asm statement. */
#ifndef SIMD_BENCH_ASM_H
#define SIMD_BENCH_ASM_H
#include "simd_bench_util.h"

/* ============================================================================ spinning register holders
 * load vin -> XMM/YMM 0..N-1, ready = 1, PAUSE-spin until stop != 0 (GPR only), store -> vout. */
typedef struct bx_spin {
    T_ALIGN(64) uint8_t vin[16][32];    /* offset 0    */
    uint8_t vout[16][32];               /* offset 512  */
    volatile LONG ready;                /* offset 1024 */
    volatile LONG stop;                 /* offset 1028 */
} bx_spin;
_Static_assert(offsetof(bx_spin, vout) == 512 && offsetof(bx_spin, ready) == 1024 && offsetof(bx_spin, stop) == 1028, "bx_spin layout");

#define BXL(i)  "movdqu " #i "*32(%0), %%xmm" #i "\n\t"
#define BXS(i)  "movdqu %%xmm" #i ", 512+" #i "*32(%0)\n\t"
#define BYL(i)  "vmovdqu " #i "*32(%0), %%ymm" #i "\n\t"
#define BYS(i)  "vmovdqu %%ymm" #i ", 512+" #i "*32(%0)\n\t"
#define BX_SPIN_LOOP "movl $1, 1024(%0)\n\t" "1:\n\tpause\n\tcmpl $0, 1028(%0)\n\tje 1b\n\t"
#ifdef T_X64
#define BX_CLOB "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
#define BX_HI(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#else
#define BX_CLOB "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
#define BX_HI(M)
#endif
#define BX_ALL(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) BX_HI(M)

static T_NOINLINE void bx_spin_xmm(bx_spin *a) {
    __asm__ volatile(BX_ALL(BXL) BX_SPIN_LOOP BX_ALL(BXS) :: "r"(a) : "memory", "cc", BX_CLOB);
}
static T_NOINLINE T_TARGET("avx") void bx_spin_ymm(bx_spin *a) {
    __asm__ volatile(BX_ALL(BYL) BX_SPIN_LOOP BX_ALL(BYS) "vzeroupper\n\t" :: "r"(a) : "memory", "cc", BX_CLOB);
}

/* ============================================================================ AVX <-> SSE transitions */
typedef struct tr_ctx {
    T_ALIGN(32) float ones[8];
    T_ALIGN(32) float out[4][8];
    long K;
} tr_ctx;

/* A: VEX.128 vaddps only (no transition) */
static T_NOINLINE T_TARGET("avx") void trans_avx_vex128(void *p) {
    tr_ctx *x = (tr_ctx *)p; long k = x->K;
    __asm__ volatile(
        "vmovups (%[o1]), %%xmm4\n\t"
        "vxorps %%xmm0, %%xmm0, %%xmm0\n\t" "vxorps %%xmm1, %%xmm1, %%xmm1\n\t"
        "vxorps %%xmm2, %%xmm2, %%xmm2\n\t" "vxorps %%xmm3, %%xmm3, %%xmm3\n\t"
        "1:\n\t"
        "vaddps %%xmm4, %%xmm0, %%xmm0\n\t" "vaddps %%xmm4, %%xmm1, %%xmm1\n\t"
        "vaddps %%xmm4, %%xmm2, %%xmm2\n\t" "vaddps %%xmm4, %%xmm3, %%xmm3\n\t"
        "dec %[k]\n\t" "jnz 1b\n\t"
        "vmovups %%xmm0, 0(%[out])\n\t" "vmovups %%xmm1, 32(%[out])\n\t" "vmovups %%xmm2, 64(%[out])\n\t" "vmovups %%xmm3, 96(%[out])\n\t"
        "vzeroupper\n\t"
        : [k] "+r"(k) : [o1] "r"(x->ones), [out] "r"(x->out) : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
/* B: legacy SSE addps after VZEROUPPER (clean upper state) */
static T_NOINLINE T_TARGET("avx") void trans_avx_legacy_clean(void *p) {
    tr_ctx *x = (tr_ctx *)p; long k = x->K;
    __asm__ volatile(
        "vzeroupper\n\t"
        "movups (%[o1]), %%xmm4\n\t"
        "xorps %%xmm0, %%xmm0\n\t" "xorps %%xmm1, %%xmm1\n\t" "xorps %%xmm2, %%xmm2\n\t" "xorps %%xmm3, %%xmm3\n\t"
        "1:\n\t"
        "addps %%xmm4, %%xmm0\n\t" "addps %%xmm4, %%xmm1\n\t" "addps %%xmm4, %%xmm2\n\t" "addps %%xmm4, %%xmm3\n\t"
        "dec %[k]\n\t" "jnz 1b\n\t"
        "movups %%xmm0, 0(%[out])\n\t" "movups %%xmm1, 32(%[out])\n\t" "movups %%xmm2, 64(%[out])\n\t" "movups %%xmm3, 96(%[out])\n\t"
        : [k] "+r"(k) : [o1] "r"(x->ones), [out] "r"(x->out) : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
/* C: the same legacy SSE loop while YMM5 holds a non-zero upper half (dirty upper state, no VZEROUPPER) */
static T_NOINLINE T_TARGET("avx") void trans_avx_legacy_dirty(void *p) {
    tr_ctx *x = (tr_ctx *)p; long k = x->K;
    __asm__ volatile(
        "vmovups (%[o1]), %%ymm5\n\t"
        "movups (%[o1]), %%xmm4\n\t"
        "xorps %%xmm0, %%xmm0\n\t" "xorps %%xmm1, %%xmm1\n\t" "xorps %%xmm2, %%xmm2\n\t" "xorps %%xmm3, %%xmm3\n\t"
        "1:\n\t"
        "addps %%xmm4, %%xmm0\n\t" "addps %%xmm4, %%xmm1\n\t" "addps %%xmm4, %%xmm2\n\t" "addps %%xmm4, %%xmm3\n\t"
        "dec %[k]\n\t" "jnz 1b\n\t"
        "movups %%xmm0, 0(%[out])\n\t" "movups %%xmm1, 32(%[out])\n\t" "movups %%xmm2, 64(%[out])\n\t" "movups %%xmm3, 96(%[out])\n\t"
        "vzeroupper\n\t"
        : [k] "+r"(k) : [o1] "r"(x->ones), [out] "r"(x->out) : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
/* D: alternate a 256-bit VEX op and a legacy SSE op every iteration, no VZEROUPPER */
static T_NOINLINE T_TARGET("avx") void trans_avx_alternate_novzu(void *p) {
    tr_ctx *x = (tr_ctx *)p; long k = x->K;
    __asm__ volatile(
        "vmovups (%[o1]), %%ymm7\n\t" "vxorps %%ymm6, %%ymm6, %%ymm6\n\t"
        "movups (%[o1]), %%xmm4\n\t" "xorps %%xmm0, %%xmm0\n\t"
        "1:\n\t"
        "vaddps %%ymm7, %%ymm6, %%ymm6\n\t"
        "addps %%xmm4, %%xmm0\n\t"
        "dec %[k]\n\t" "jnz 1b\n\t"
        "vmovups %%ymm6, 0(%[out])\n\t" "movups %%xmm0, 32(%[out])\n\t"
        "vzeroupper\n\t"
        : [k] "+r"(k) : [o1] "r"(x->ones), [out] "r"(x->out) : "cc", "memory", "xmm0", "xmm4", "xmm6", "xmm7");
}
/* E: the same with VZEROUPPER between the 256-bit op and the legacy op */
static T_NOINLINE T_TARGET("avx") void trans_avx_alternate_vzu(void *p) {
    tr_ctx *x = (tr_ctx *)p; long k = x->K;
    __asm__ volatile(
        "vmovups (%[o1]), %%ymm7\n\t" "vxorps %%ymm6, %%ymm6, %%ymm6\n\t"
        "movups (%[o1]), %%xmm4\n\t" "xorps %%xmm0, %%xmm0\n\t"
        "1:\n\t"
        "vaddps %%ymm7, %%ymm6, %%ymm6\n\t"
        "vzeroupper\n\t"
        "addps %%xmm4, %%xmm0\n\t"
        "dec %[k]\n\t" "jnz 1b\n\t"
        "vmovups %%ymm6, 0(%[out])\n\t" "movups %%xmm0, 32(%[out])\n\t"
        "vzeroupper\n\t"
        : [k] "+r"(k) : [o1] "r"(x->ones), [out] "r"(x->out) : "cc", "memory", "xmm0", "xmm4", "xmm6", "xmm7");
}
/* Architectural semantics around the transition (SDM vol.1 14.1.3 / 14.3, vol.2 VZEROUPPER, VZEROALL):
 *   out[0]: legacy SSE (POR) on XMM0 of a full YMM0 image  -> bits 255:128 unchanged      = P0
 *   out[1]: VEX.128 VPOR on YMM2                            -> bits 255:128 zeroed         = P0.lo | 0
 *   out[2]: legacy MOVDQU load into XMM3 of YMM3 = P1       -> low replaced, upper kept    = P0.lo | P1.hi
 *   out[3]: VZEROUPPER with YMM4 = P0                       -> low kept, upper zeroed      = P0.lo | 0
 *   out[4]: VZEROALL with YMM5 = P0                         -> all zero
 *   out[5]: YMM6 = P1 loaded after VZEROALL, legacy MOVAPS xmm6 <- xmm5 (= 0) -> low 0, upper kept = 0 | P1.hi */
static T_NOINLINE T_TARGET("avx") void trans_avx_semantics(const uint8_t pat[2][32], uint8_t out[6][32]) {
    __asm__ volatile(
        "vmovdqu 0(%[p]), %%ymm0\n\t"
        "pxor %%xmm1, %%xmm1\n\t"
        "por %%xmm1, %%xmm0\n\t"
        "vmovdqu %%ymm0, 0(%[o])\n\t"
        "vmovdqu 0(%[p]), %%ymm2\n\t"
        "vpor %%xmm1, %%xmm2, %%xmm2\n\t"
        "vmovdqu %%ymm2, 32(%[o])\n\t"
        "vmovdqu 32(%[p]), %%ymm3\n\t"
        "movdqu 0(%[p]), %%xmm3\n\t"
        "vmovdqu %%ymm3, 64(%[o])\n\t"
        "vmovdqu 0(%[p]), %%ymm4\n\t"
        "vzeroupper\n\t"
        "vmovdqu %%ymm4, 96(%[o])\n\t"
        "vmovdqu 0(%[p]), %%ymm5\n\t"
        "vzeroall\n\t"
        "vmovdqu %%ymm5, 128(%[o])\n\t"
        "vmovdqu 32(%[p]), %%ymm6\n\t"
        "movaps %%xmm5, %%xmm6\n\t"
        "vmovdqu %%ymm6, 160(%[o])\n\t"
        "vzeroupper\n\t"
        :: [p] "r"(pat), [o] "r"(out) : "memory", BX_CLOB);   /* VZEROALL touches every vector register */
}
#endif
