/* asm_common.c - single-instruction wrappers and checked register-file wrappers (both arches).
 * Each AVX-class instruction lives in its own small function; nothing here runs unless the caller
 * checked the feature (see asm.h). */
#include "t.h"

unsigned t_get_mxcsr(void) { unsigned v; __asm__ volatile("stmxcsr %0" : "=m"(v)); return v; }
void t_set_mxcsr(unsigned v) { __asm__ volatile("ldmxcsr %0" :: "m"(v)); }
unsigned short t_get_x87_cw(void) { unsigned short v; __asm__ volatile("fnstcw %0" : "=m"(v)); return v; }
void t_set_x87_cw(unsigned short v) { __asm__ volatile("fldcw %0" :: "m"(v)); }
unsigned short t_get_x87_sw(void) { unsigned short v; __asm__ volatile("fnstsw %0" : "=m"(v)); return v; }
unsigned short t_get_x87_tw(void) {
    unsigned char env[28];   /* 32-bit protected-mode FNSTENV layout: tag word at offset 8 */
    unsigned short cw;
    __asm__ volatile("fnstcw %0" : "=m"(cw));
    __asm__ volatile("fnstenv %0" : "=m"(env));
    __asm__ volatile("fldcw %0" :: "m"(cw));   /* fnstenv masks all exceptions: restore CW */
    unsigned short tw; memcpy(&tw, env + 8, 2);
    return tw;
}
void t_fninit(void) { __asm__ volatile("fninit"); }
void t_fnclex(void) { __asm__ volatile("fnclex"); }
void t_emms(void) { __asm__ volatile("emms"); }
void t_vzeroupper(void) { __asm__ volatile("vzeroupper" ::: "memory"); }
void t_vzeroall(void) { __asm__ volatile("vzeroall" ::: "memory"); }
uint64_t t_xgetbv(uint32_t idx) { uint32_t lo, hi; __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(idx)); return (uint64_t)hi << 32 | lo; }

#ifdef T_X64
#define XS(op) op "64"
#else
#define XS(op) op
#endif
void t_fxsave(void *p)        { __asm__ volatile(XS("fxsave") " (%0)" :: "r"(p) : "memory"); }
void t_fxrstor(const void *p) { __asm__ volatile(XS("fxrstor") " (%0)" :: "r"(p) : "memory"); }
void t_xsave(void *p, uint64_t m)    { __asm__ volatile(XS("xsave") " (%0)" :: "r"(p), "a"((uint32_t)m), "d"((uint32_t)(m >> 32)) : "memory"); }
void t_xsaveopt(void *p, uint64_t m) { __asm__ volatile(XS("xsaveopt") " (%0)" :: "r"(p), "a"((uint32_t)m), "d"((uint32_t)(m >> 32)) : "memory"); }
void t_xsavec(void *p, uint64_t m)   { __asm__ volatile(XS("xsavec") " (%0)" :: "r"(p), "a"((uint32_t)m), "d"((uint32_t)(m >> 32)) : "memory"); }
void t_xrstor(const void *p, uint64_t m) { __asm__ volatile(XS("xrstor") " (%0)" :: "r"(p), "a"((uint32_t)m), "d"((uint32_t)(m >> 32)) : "memory"); }

/* ---- checked register-file wrappers ------------------------------------- */
int t_vregs_supported(int kind) {
    switch (kind) {
    case T_VK_XMM: return cpu_has(F_SSE2);
    case T_VK_YMM: return cpu_has(F_AVX);
    case T_VK_ZMM: return cpu_has(F_AVX512F);
    default: return 0;
    }
}
int t_vregs_count(int kind) {
    if (!t_vregs_supported(kind)) return 0;
    return kind == T_VK_ZMM ? T_NZMM : T_NXMM;
}
void t_vregs_pattern(t_vregs *r, int kind, uint32_t tag) {
    int n = kind == T_VK_ZMM ? T_NZMM : T_NXMM;
    memset(r, 0, sizeof *r);
    t_pattern_regs(r->v, n, 64, (size_t)kind, tag);
    if (kind == T_VK_ZMM) {
        for (int i = 0; i < 8; i++) {
            uint64_t k = t_pattern_word(tag, 0x40u + (uint32_t)i, 0);
            r->k[i] = cpu_has(F_AVX512BW) ? k : (k & 0xFFFF);
        }
    }
}
int t_vregs_check(const t_vregs *got, const t_vregs *want, int kind, int first, int last, const char *label) {
    int n = kind == T_VK_ZMM ? T_NZMM : T_NXMM, bad = 0;
    const char *nm = kind == T_VK_ZMM ? "zmm" : kind == T_VK_YMM ? "ymm" : "xmm";
    if (last < 0 || last >= n) last = n - 1;
    for (int i = first < 0 ? 0 : first; i <= last; i++) {
        char what[160]; snprintf(what, sizeof what, "%s%d %s", nm, i, label ? label : "");
        bad += !t_check_reg_at(__FILE__, __LINE__, what, got->v[i], want->v[i], (size_t)kind);
    }
    if (kind == T_VK_ZMM && first <= 0) {
        uint64_t m = cpu_has(F_AVX512BW) ? ~0ull : 0xFFFFull;
        for (int i = 0; i < 8; i++) {
            if ((got->k[i] & m) == (want->k[i] & m)) { t_check(1, __FILE__, __LINE__, "k", NULL); continue; }
            t_check(0, __FILE__, __LINE__, "k reg", "k%d %s: got 0x%llx want 0x%llx", i, label ? label : "",
                    (unsigned long long)(got->k[i] & m), (unsigned long long)(want->k[i] & m));
            bad++;
        }
    }
    return bad;
}
int t_vregs_call(int kind, const t_vregs *in, t_vregs *out, void (*fn)(void *), void *arg) {
    if (!t_vregs_supported(kind)) return -1;
    memset(out, 0, sizeof *out);
    if (kind == T_VK_XMM) t__xmm_call(in, out, fn, arg);
    else if (kind == T_VK_YMM) t__ymm_call(in, out, fn, arg);
    else t__zmm_call(in, out, fn, arg, cpu_has(F_AVX512BW));
    return 0;
}
long t_vregs_spin(int kind, const t_vregs *in, t_vregs *out, volatile const long *stop, unsigned long max) {
    if (!t_vregs_supported(kind)) return -1;
    memset(out, 0, sizeof *out);
    unsigned long r;
    if (kind == T_VK_XMM) r = t__xmm_spin(in, out, stop, max);
    else if (kind == T_VK_YMM) r = t__ymm_spin(in, out, stop, max);
    else r = t__zmm_spin(in, out, stop, max, cpu_has(F_AVX512BW));
    return (long)(r & 0x7FFFFFFF);
}
long t_x87_spin(const t_f80 in[8], t_f80 out[8], volatile const long *stop, unsigned long max) {
    memset(out, 0, sizeof(t_f80) * 8);
    return (long)(t__x87_spin(in, out, stop, max) & 0x7FFFFFFF);
}
long t_mmx_spin(const uint64_t in[8], uint64_t out[8], volatile const long *stop, unsigned long max) {
    if (!cpu_has(F_MMX)) return -1;
    memset(out, 0, sizeof(uint64_t) * 8);
    return (long)(t__mmx_spin(in, out, stop, max) & 0x7FFFFFFF);
}
