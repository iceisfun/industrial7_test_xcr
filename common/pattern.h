/* pattern.h - deterministic PRNG, register/buffer pattern generators, hexdump diagnostics. */
#ifndef T_PATTERN_H
#define T_PATTERN_H
#include <stdint.h>
#include <stddef.h>

/* xoshiro256** seeded through splitmix64: identical sequence on every platform/arch. */
typedef struct t_rng { uint64_t s[4]; } t_rng;
void     t_rng_seed(t_rng *r, uint64_t seed);
uint64_t t_rng_next(t_rng *r);
uint32_t t_rng_u32(t_rng *r);
uint64_t t_rng_range(t_rng *r, uint64_t n);          /* uniform in [0, n), n > 0 */
void     t_rng_fill(t_rng *r, void *buf, size_t n);
uint64_t t_mix64(uint64_t x);                         /* splitmix64 finalizer (stateless hash) */

/* Register pattern: deterministic, distinctive, never all-zero per 8-byte word.
 * Layout of each 64-bit word w of register `reg` for owner `tag` (thread id, iteration, ...):
 *   byte 7 = 0xA0 | (reg & 0x1F) >> 1 ... (hash-based), see pattern.c; t_pattern_explain decodes.
 * Every (tag, reg, word) triple gives a different value, so leaked state identifies its source. */
uint64_t t_pattern_word(uint32_t tag, uint32_t reg, uint32_t word);
void     t_pattern_reg(void *out, size_t bytes, uint32_t tag, uint32_t reg);
/* Fill nregs registers of `stride` bytes each (e.g. a t_vregs.v array: stride 64, width 16/32/64). */
void     t_pattern_regs(void *base, int nregs, size_t stride, size_t width, uint32_t tag);
/* Try to find which (tag, reg, word) produced v among tags [0, max_tag) and regs [0, 32).
 * Returns 1 and fills tag/reg/word when found. Diagnostic only. */
int      t_pattern_explain(uint64_t v, uint32_t max_tag, uint32_t *tag, uint32_t *reg, uint32_t *word);

/* Hex formatting. t_hex: bytes in memory order, space every 4 bytes (as printed by CHECK_MEM).
 * t_hex_lanes: little-endian lane values, highest lane first (like a debugger shows a register),
 * lane_bytes = 1/2/4/8. Both return buf (always NUL-terminated). */
char *t_hex(char *buf, size_t cap, const void *p, size_t n);
char *t_hex_lanes(char *buf, size_t cap, const void *p, size_t n, int lane_bytes);
/* Print "label: <lanes>" for a 16/32/64-byte value through t_note (or t_log if verbose_only). */
void  t_dump_vec(const char *label, const void *p, size_t n, int verbose_only);
/* Compare two register images; on mismatch records a FAIL naming the register and the first differing
 * qword lane and prints both values. Returns 1 when equal. `what` e.g. "xmm7 after Sleep(0)". */
int   t_check_reg_at(const char *file, int line, const char *what, const void *got, const void *want, size_t n);
#define CHECK_REG(what, got, want, n) t_check_reg_at(__FILE__, __LINE__, (what), (got), (want), (size_t)(n))
#endif
