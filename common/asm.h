/* asm.h - instruction-level helpers. Ordinary C variables do not pin hardware registers, so every
 * "registers hold a pattern across X" test must use the *_call / *_spin helpers below: they load the
 * registers, perform X (a callback or a GPR-only spin loop) and store the registers in ONE assembly
 * routine, with nothing compiled in between.
 *
 * ISA safety: AVX/AVX-512/x87/MMX instructions exist only inside the routines that need them
 * (asm_x64.c / asm_x86.c / asm_common.c). Callers MUST check availability first:
 *   t_vregs_*(T_VK_YMM, ...) requires cpu_has(F_AVX); T_VK_ZMM requires cpu_has(F_AVX512F)
 *   (the t_vregs_* wrappers check and return -1 instead of executing an unsupported instruction).
 *   t_xsave* need cpu_has(F_XSAVE) (+F_XSAVEOPT / F_XSAVEC); t_vzero* need cpu_has(F_AVX).
 * On x64 the FXSAVE/XSAVE wrappers use the 64-bit forms (FXSAVE64, XSAVE64, ...), on x86 the plain forms. */
#ifndef T_ASM_H
#define T_ASM_H
#include <stdint.h>

/* ---- control/status registers ---------------------------------------- */
unsigned       t_get_mxcsr(void);
void           t_set_mxcsr(unsigned v);
unsigned short t_get_x87_cw(void);
void           t_set_x87_cw(unsigned short v);
unsigned short t_get_x87_sw(void);
unsigned short t_get_x87_tw(void);          /* full tag word from FNSTENV (0xFFFF = all empty) */
void           t_fninit(void);
void           t_fnclex(void);
void           t_emms(void);
void           t_vzeroupper(void);          /* AVX */
void           t_vzeroall(void);            /* AVX */
uint64_t       t_xgetbv(uint32_t index);    /* needs OSXSAVE; index 1 needs F_XGETBV1 */

/* ---- save areas ----------------------------------------------------------- */
void t_fxsave(void *area512);               /* 16-byte aligned, 512 bytes */
void t_fxrstor(const void *area512);
void t_xsave(void *area, uint64_t rfbm);    /* 64-byte aligned, xsave_std_size(rfbm) bytes, header must be valid for xrstor */
void t_xsaveopt(void *area, uint64_t rfbm);
void t_xsavec(void *area, uint64_t rfbm);   /* compacted format: size xsave_compacted_size(rfbm) */
void t_xrstor(const void *area, uint64_t rfbm);

/* ---- vector register files --------------------------------------------- */
enum { T_VK_XMM = 16, T_VK_YMM = 32, T_VK_ZMM = 64 };   /* kind == register width in bytes */
#ifdef __x86_64__
#define T_NXMM 16      /* XMM0-15 / YMM0-15 */
#define T_NZMM 32      /* ZMM0-31 */
#else
#define T_NXMM 8
#define T_NZMM 8
#endif
/* Register image. v[i] holds register i in its low `kind` bytes (rest ignored/zeroed).
 * k[] = opmask k0..k7 (ZMM kind only; 64-bit with AVX512BW, else low 16 bits). */
typedef struct t_vregs {
    __attribute__((aligned(64))) uint8_t v[32][64];
    uint64_t k[8];
} t_vregs;

int  t_vregs_count(int kind);               /* registers handled for kind (0 if kind not usable here) */
int  t_vregs_supported(int kind);           /* cpu_has gate for kind */
/* Fill `r` with t_pattern_reg(tag, i) for every register of `kind` (+ k regs for ZMM). */
void t_vregs_pattern(t_vregs *r, int kind, uint32_t tag);
/* Compare the registers of `kind` (and k regs for ZMM) with CHECK_REG; label e.g. "after Sleep(0)".
 * first/last restrict the register range (use 0, -1 for all). Returns number of mismatching registers. */
int  t_vregs_check(const t_vregs *got, const t_vregs *want, int kind, int first, int last, const char *label);
/* load all; call fn(arg) if fn != NULL; store all. Callee-saved registers of the caller are preserved.
 * NOTE: fn is an ordinary function, so per the ABI it may clobber volatile registers (x64: XMM0-5,
 * upper YMM/ZMM halves, XMM16-31, k regs; x86: all). Returns 0, or -1 if kind is not usable. */
int  t_vregs_call(int kind, const t_vregs *in, t_vregs *out, void (*fn)(void *), void *arg);
/* load all; spin (GPR-only loop with PAUSE) until *stop != 0 or max_loops iterations; store all.
 * stop may be NULL. Returns the remaining loop count (0 = ran to max), or -1 if not usable. */
long t_vregs_spin(int kind, const t_vregs *in, t_vregs *out, volatile const long *stop, unsigned long max_loops);

/* x87 stack / MMX. in[0] ends up in ST(0). Caller's x87 CW is preserved; stack is empty on return. */
typedef struct t_f80 { uint8_t b[10]; } t_f80;
long t_x87_spin(const t_f80 in[8], t_f80 out[8], volatile const long *stop, unsigned long max_loops);
/* MM0-7 (aliases of the x87 registers); ends with EMMS. */
long t_mmx_spin(const uint64_t in[8], uint64_t out[8], volatile const long *stop, unsigned long max_loops);

/* raw entry points (asm, implemented per arch). Prefer the checked wrappers above. */
void          t__xmm_call(const t_vregs *in, t_vregs *out, void (*fn)(void *), void *arg);
void          t__ymm_call(const t_vregs *in, t_vregs *out, void (*fn)(void *), void *arg);
void          t__zmm_call(const t_vregs *in, t_vregs *out, void (*fn)(void *), void *arg, int kbw);
unsigned long t__xmm_spin(const t_vregs *in, t_vregs *out, volatile const long *stop, unsigned long max);
unsigned long t__ymm_spin(const t_vregs *in, t_vregs *out, volatile const long *stop, unsigned long max);
unsigned long t__zmm_spin(const t_vregs *in, t_vregs *out, volatile const long *stop, unsigned long max, int kbw);
unsigned long t__x87_spin(const t_f80 *in, t_f80 *out, volatile const long *stop, unsigned long max);
unsigned long t__mmx_spin(const uint64_t *in, uint64_t *out, volatile const long *stop, unsigned long max);
#endif
