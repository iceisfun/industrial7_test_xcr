/* cpuid.c - CPUID access and leaf enumeration (Intel SDM Vol.2A CPUID; AMD APM Vol.3 Appendix E). */
#include "t.h"

int cpuid_available(void) {
#ifdef T_X64
    return 1;
#else
    uint32_t a, b;
    __asm__ volatile("pushfl\n\tpopl %0\n\tmovl %0, %1\n\txorl $0x200000, %0\n\tpushl %0\n\tpopfl\n\t"
                     "pushfl\n\tpopl %0\n\tpushl %1\n\tpopfl" : "=&r"(a), "=&r"(b) :: "cc");
    return ((a ^ b) & 0x200000) != 0;
#endif
}

void cpuid_ex(uint32_t leaf, uint32_t subleaf, t_cpuid_regs *r) {
    uint32_t a, b, c, d;
#ifdef T_X86
    /* keep EBX handling explicit (works with or without PIC) */
    __asm__ volatile("xchgl %%ebx, %1\n\tcpuid\n\txchgl %%ebx, %1" : "=a"(a), "=&r"(b), "=c"(c), "=d"(d) : "0"(leaf), "2"(subleaf));
#else
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "0"(leaf), "2"(subleaf));
#endif
    r->eax = a; r->ebx = b; r->ecx = c; r->edx = d;
}

uint32_t cpuid_max_basic(void) { t_cpuid_regs r; if (!cpuid_available()) return 0; cpuid_ex(0, 0, &r); return r.eax; }
uint32_t cpuid_max_ext(void) {
    t_cpuid_regs r; if (!cpuid_available()) return 0;
    cpuid_ex(0x80000000u, 0, &r);
    return (r.eax >= 0x80000000u && r.eax < 0x80010000u) ? r.eax : 0;
}
uint32_t cpuid_max_hv(void) {
    t_cpuid_regs r;
    if (cpuid_max_basic() < 1) return 0;
    cpuid_ex(1, 0, &r);
    if (!(r.ecx & (1u << 31))) return 0;
    cpuid_ex(0x40000000u, 0, &r);
    if (r.eax < 0x40000000u || r.eax > 0x400000FFu) return 0x40000000u;  /* some hypervisors report 0 */
    return r.eax;
}

static int add(t_cpuid_entry *out, int n, int max, uint32_t leaf, uint32_t sub) {
    if (n >= max) return n;
    out[n].leaf = leaf; out[n].subleaf = sub; cpuid_ex(leaf, sub, &out[n].r);
    return n + 1;
}

/* subleaf rule for one leaf; returns new count */
static int enum_leaf(t_cpuid_entry *out, int n, int max, uint32_t leaf) {
    t_cpuid_regs r0; cpuid_ex(leaf, 0, &r0);
    uint32_t s, cap = 64;
    switch (leaf) {
    case 0x4: case 0x8000001Du:            /* deterministic cache parameters: until type == 0 */
        for (s = 0; s < cap; s++) { t_cpuid_regs r; cpuid_ex(leaf, s, &r); if ((r.eax & 0x1F) == 0 && s) break; n = add(out, n, max, leaf, s); if ((r.eax & 0x1F) == 0) break; }
        return n;
    case 0x7: case 0x14: case 0x17: case 0x18: case 0x1D: case 0x1E: case 0x20: case 0x24:
        /* subleaf 0 EAX = max valid subleaf */
        for (s = 0; s <= r0.eax && s < cap; s++) n = add(out, n, max, leaf, s);
        return n;
    case 0xB: case 0x1F: case 0x80000026u: /* topology: until level type ECX[15:8] == 0 */
        for (s = 0; s < cap; s++) { t_cpuid_regs r; cpuid_ex(leaf, s, &r); n = add(out, n, max, leaf, s); if (((r.ecx >> 8) & 0xFF) == 0) break; }
        return n;
    case 0xD: {                            /* XSAVE: 0, 1, then every supported component */
        t_cpuid_regs r1; cpuid_ex(0xD, 1, &r1);
        uint64_t m = ((uint64_t)r0.edx << 32 | r0.eax) | ((uint64_t)r1.edx << 32 | r1.ecx);
        n = add(out, n, max, leaf, 0); n = add(out, n, max, leaf, 1);
        for (s = 2; s < 64; s++) if (m >> s & 1) n = add(out, n, max, leaf, s);
        return n;
    }
    case 0xF: case 0x10:
        for (s = 0; s < 4; s++) n = add(out, n, max, leaf, s);
        return n;
    case 0x12:                             /* SGX: 0, 1, then EPC sections until type == 0 */
        n = add(out, n, max, leaf, 0); n = add(out, n, max, leaf, 1);
        for (s = 2; s < 18; s++) { t_cpuid_regs r; cpuid_ex(leaf, s, &r); if ((r.eax & 0xF) == 0) break; n = add(out, n, max, leaf, s); }
        return n;
    case 0x1B:
        for (s = 0; s < 8; s++) { t_cpuid_regs r; cpuid_ex(leaf, s, &r); n = add(out, n, max, leaf, s); if ((r.eax & 0xFFF) == 0) break; }
        return n;
    case 0x23:                             /* subleaf 0 EBX/EAX bitmap of valid subleaves */
        n = add(out, n, max, leaf, 0);
        for (s = 1; s < 32; s++) if (r0.eax >> s & 1) n = add(out, n, max, leaf, s);
        return n;
    case 0x80000020u:
        for (s = 0; s < 4; s++) n = add(out, n, max, leaf, s);
        return n;
    default:
        return add(out, n, max, leaf, 0);
    }
}

int cpuid_enumerate(t_cpuid_entry *out, int max) {
    int n = 0;
    if (!cpuid_available()) return 0;
    uint32_t mb = cpuid_max_basic(); if (mb > 0xFF) mb = 0xFF;
    for (uint32_t l = 0; l <= mb; l++) n = enum_leaf(out, n, max, l);
    uint32_t mh = cpuid_max_hv(); if (mh > 0x400000FFu) mh = 0x400000FFu;
    if (mh) for (uint32_t l = 0x40000000u; l <= mh; l++) n = add(out, n, max, l, 0);
    uint32_t me = cpuid_max_ext(); if (me > 0x800000FFu) me = 0x800000FFu;
    if (me) for (uint32_t l = 0x80000000u; l <= me; l++) n = enum_leaf(out, n, max, l);
    return n;
}
