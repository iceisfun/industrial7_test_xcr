/* cpuid.h - CPUID wrappers (x86 and x64) and leaf enumeration. Raw hardware view: no OS gating here
 * (see feature_detect.h for "usable" features). */
#ifndef T_CPUID_H
#define T_CPUID_H
#include <stdint.h>

typedef struct t_cpuid_regs { uint32_t eax, ebx, ecx, edx; } t_cpuid_regs;

int  cpuid_available(void);                                   /* EFLAGS.ID toggles (always 1 on x64) */
void cpuid_ex(uint32_t leaf, uint32_t subleaf, t_cpuid_regs *r); /* CPUID with EAX=leaf, ECX=subleaf */
static inline void cpuid_leaf(uint32_t leaf, t_cpuid_regs *r) { cpuid_ex(leaf, 0, r); }
uint32_t cpuid_max_basic(void);     /* CPUID.0:EAX */
uint32_t cpuid_max_ext(void);       /* CPUID.80000000h:EAX (0 if the extended range is absent) */
uint32_t cpuid_max_hv(void);        /* CPUID.40000000h:EAX when CPUID.1:ECX[31] (hypervisor) is set, else 0 */

/* Enumerate every implemented leaf/subleaf (basic, hypervisor, extended), following the documented
 * subleaf rules for leaves 4, 7, 0Bh, 0Dh, 0Fh, 10h, 12h, 14h, 17h, 18h, 1Ah, 1Bh, 1Dh, 1Eh, 1Fh, 20h,
 * 23h, 24h, 8000001Dh, 80000020h, 80000026h. Leaves above 0x100 entries or 64 subleaves are capped.
 * Returns the number of entries written (<= max). */
typedef struct t_cpuid_entry { uint32_t leaf, subleaf; t_cpuid_regs r; } t_cpuid_entry;
int cpuid_enumerate(t_cpuid_entry *out, int max);

/* Register selector used by feature tables */
enum { CPUID_EAX = 0, CPUID_EBX = 1, CPUID_ECX = 2, CPUID_EDX = 3 };
static inline uint32_t cpuid_reg(const t_cpuid_regs *r, int reg) {
    return reg == CPUID_EAX ? r->eax : reg == CPUID_EBX ? r->ebx : reg == CPUID_ECX ? r->ecx : r->edx;
}
#endif
