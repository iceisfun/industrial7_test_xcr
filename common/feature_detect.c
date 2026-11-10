/* feature_detect.c - CPUID + XCR0 based feature detection (Intel SDM Vol.1 13.2/13.3, 14.3, 15.2;
 * Vol.2 CPUID leaves 1, 7, 0Dh, 24h; AMD APM Vol.3 CPUID Fn8000_0001). */
#include "t.h"

typedef struct { uint32_t leaf, sub; int reg, bit, gate; const char *name; } feat_def;
#define T_FEAT_DEF_(id, l, s, r, b, g, n) { l, s, r, b, g, n },
static const feat_def g_defs[F_COUNT] = { T_FEATURE_LIST(T_FEAT_DEF_) };
#undef T_FEAT_DEF_

static t_cpu_info g_ci;
static volatile LONG g_done;

static void read_leaf(uint32_t leaf, uint32_t sub, t_cpuid_regs *r, uint32_t max) {
    memset(r, 0, sizeof *r);
    if (leaf <= max) cpuid_ex(leaf, sub, r);
}

void cpu_detect(void) {
    if (g_done) return;
    t_cpu_info *c = &g_ci;
    memset(c, 0, sizeof *c);
    c->cpuid_ok = cpuid_available();
    if (c->cpuid_ok) {
        t_cpuid_regs r;
        cpuid_ex(0, 0, &r);
        c->max_basic = r.eax;
        memcpy(c->vendor + 0, &r.ebx, 4); memcpy(c->vendor + 4, &r.edx, 4); memcpy(c->vendor + 8, &r.ecx, 4);
        c->is_intel = !strcmp(c->vendor, "GenuineIntel");
        c->is_amd = !strcmp(c->vendor, "AuthenticAMD") || !strcmp(c->vendor, "HygonGenuine");
        c->max_ext = cpuid_max_ext();
        read_leaf(1, 0, &c->l1, c->max_basic);
        read_leaf(7, 0, &c->l7_0, c->max_basic);
        if (c->max_basic >= 7 && c->l7_0.eax >= 1) cpuid_ex(7, 1, &c->l7_1);
        read_leaf(0xD, 0, &c->ld_0, c->max_basic);
        read_leaf(0xD, 1, &c->ld_1, c->max_basic);
        if (c->max_ext >= 0x80000001u) cpuid_ex(0x80000001u, 0, &c->e1);
        if (c->max_ext >= 0x80000007u) cpuid_ex(0x80000007u, 0, &c->e7);
        if (c->max_ext >= 0x80000008u) cpuid_ex(0x80000008u, 0, &c->e8);
        if (c->max_ext >= 0x80000004u) {
            uint32_t b[12];
            for (int i = 0; i < 3; i++) { cpuid_ex(0x80000002u + (uint32_t)i, 0, &r); b[i*4] = r.eax; b[i*4+1] = r.ebx; b[i*4+2] = r.ecx; b[i*4+3] = r.edx; }
            memcpy(c->brand, b, 48); c->brand[48] = 0;
            char *s = c->brand; while (*s == ' ') s++;
            memmove(c->brand, s, strlen(s) + 1);
            for (size_t n = strlen(c->brand); n && c->brand[n-1] == ' '; n--) c->brand[n-1] = 0;
        }
        uint32_t sig = c->l1.eax, fam = (sig >> 8) & 0xF, mod = (sig >> 4) & 0xF;
        c->signature = sig; c->stepping = sig & 0xF;
        c->family = fam == 0xF ? fam + ((sig >> 20) & 0xFF) : fam;
        c->model = (fam == 0x6 || fam == 0xF) ? mod | (((sig >> 16) & 0xF) << 4) : mod;
        c->hypervisor = (c->l1.ecx >> 31) & 1;
        c->max_hv = cpuid_max_hv();
        if (c->max_hv) { cpuid_ex(0x40000000u, 0, &r); memcpy(c->hv_vendor, &r.ebx, 4); memcpy(c->hv_vendor + 4, &r.ecx, 4); memcpy(c->hv_vendor + 8, &r.edx, 4); }
    }
    /* XSAVE enumeration */
    c->osxsave = (c->l1.ecx >> 27) & 1;
    if ((c->l1.ecx >> 26) & 1 && c->max_basic >= 0xD) {
        c->xcr0_supported = (uint64_t)c->ld_0.edx << 32 | c->ld_0.eax;
        c->xss_supported = (uint64_t)c->ld_1.edx << 32 | c->ld_1.ecx;
        c->xsave_size_enabled = c->ld_0.ebx;
        c->xsave_size_max = c->ld_0.ecx;
        c->xsave_size_compacted = c->ld_1.ebx;
        uint64_t all = c->xcr0_supported | c->xss_supported;
        c->comp[0].valid = (all & 1) != 0; c->comp[0].size = 160; c->comp[0].offset = 0;
        c->comp[1].valid = (all & 2) != 0; c->comp[1].size = 256; c->comp[1].offset = 160;
        for (int i = 2; i < 63; i++) {
            if (!(all >> i & 1)) continue;
            t_cpuid_regs r; cpuid_ex(0xD, (uint32_t)i, &r);
            c->comp[i].valid = 1; c->comp[i].size = r.eax; c->comp[i].offset = r.ebx;
            c->comp[i].supervisor = r.ecx & 1; c->comp[i].align64 = (r.ecx >> 1) & 1; c->comp[i].xfd = (r.ecx >> 2) & 1;
        }
    }
    if (c->osxsave) {
        c->xcr0 = t_xgetbv(0); c->xcr0_valid = 1;
        if (c->ld_1.eax & 4) { c->xinuse = t_xgetbv(1); c->xinuse_valid = 1; }
    }
    if (((c->l7_1.edx >> 19) & 1) && c->max_basic >= 0x24) { t_cpuid_regs r; cpuid_ex(0x24, 0, &r); c->avx10_version = r.ebx & 0xFF; }

    /* raw and usable */
    int ymm = c->osxsave && (c->xcr0 & T_XS_YMM_STATE) == T_XS_YMM_STATE;
    int zmm = c->osxsave && (c->xcr0 & T_XS_ZMM_STATE) == T_XS_ZMM_STATE;
    int amx = c->osxsave && (c->xcr0 & T_XS_AMX_STATE) == T_XS_AMX_STATE;
    int apx = c->osxsave && (c->xcr0 & T_XS_APX);
    int pku = (c->l7_0.ecx >> 4) & 1;   /* RDPKRU/WRPKRU need CR4.PKE (reported as OSPKE), not XCR0 */
    for (int f = 0; f < F_COUNT; f++) {
        const feat_def *d = &g_defs[f];
        const t_cpuid_regs *r = NULL;
        if (d->leaf == 1) r = &c->l1;
        else if (d->leaf == 7 && d->sub == 0) r = &c->l7_0;
        else if (d->leaf == 7 && d->sub == 1) r = &c->l7_1;
        else if (d->leaf == 0xD && d->sub == 1) r = &c->ld_1;
        else if (d->leaf == 0x80000001u) r = &c->e1;
        else if (d->leaf == 0x80000007u) r = &c->e7;
        else if (d->leaf == 0x80000008u) r = &c->e8;
        int hw = r ? (int)((cpuid_reg(r, d->reg) >> d->bit) & 1) : 0;
        int ok;
        switch (d->gate) {
        case G_OSXSAVE: ok = c->osxsave; break;
        case G_YMM: ok = ymm; break;
        case G_ZMM: ok = zmm; break;
        case G_AMX: ok = amx; break;
        case G_APX: ok = apx; break;
        case G_PKU: ok = pku; break;
        case G_PRIV: ok = 0; break;
        default: ok = 1;
        }
        c->hw[f] = (uint8_t)hw;
        c->usable[f] = (uint8_t)(hw && ok);
    }
#ifdef T_X86
    c->usable[F_LM] = 0;   /* long mode exists but this is a 32-bit process */
#endif
    InterlockedExchange(&g_done, 1);
}

const t_cpu_info *cpu(void) { if (!g_done) cpu_detect(); return &g_ci; }
int cpu_has(t_feature f) { return (unsigned)f < F_COUNT ? cpu()->usable[f] : 0; }
int cpu_hw(t_feature f) { return (unsigned)f < F_COUNT ? cpu()->hw[f] : 0; }
const char *cpu_feature_name(t_feature f) { return (unsigned)f < F_COUNT ? g_defs[f].name : "?"; }
const char *cpu_feature_gate_name(t_feature f) {
    static const char *const n[] = { "none", "osxsave", "ymm", "zmm", "amx", "apx", "pku", "priv" };
    return (unsigned)f < F_COUNT ? n[g_defs[f].gate] : "?";
}
int cpu_feature_lookup(const char *name) {
    for (int f = 0; f < F_COUNT; f++) if (!_stricmp(g_defs[f].name, name)) return f;
    return -1;
}
size_t cpu_features_string(char *buf, size_t cap, int usable_only) {
    size_t n = 0;
    if (cap) buf[0] = 0;
    for (int f = 0; f < F_COUNT; f++) {
        int on = usable_only ? cpu_has((t_feature)f) : cpu_hw((t_feature)f);
        if (!on) continue;
        int w = snprintf(buf + n, n < cap ? cap - n : 0, "%s%s", n ? " " : "", g_defs[f].name);
        if (w > 0) n += (size_t)w;
        if (n >= cap) { n = cap ? cap - 1 : 0; break; }
    }
    return n;
}
int cpu_ymm_state_enabled(void) { const t_cpu_info *c = cpu(); return c->osxsave && (c->xcr0 & T_XS_YMM_STATE) == T_XS_YMM_STATE; }
int cpu_zmm_state_enabled(void) { const t_cpu_info *c = cpu(); return c->osxsave && (c->xcr0 & T_XS_ZMM_STATE) == T_XS_ZMM_STATE; }

const char *xstate_comp_name(int i) {
    static const char *const n[20] = { "x87", "sse", "avx", "bndregs", "bndcsr", "opmask", "zmm_hi256", "hi16_zmm",
        "pt", "pkru", "pasid", "cet_u", "cet_s", "hdc", "uintr", "lbr", "hwp", "tilecfg", "tiledata", "apx" };
    static char buf[64][12];
    if (i >= 0 && i < 20) return n[i];
    if (i < 0 || i > 63) return "?";
    snprintf(buf[i], sizeof buf[i], "comp%d", i);
    return buf[i];
}

uint32_t xsave_std_size(uint64_t mask) {
    const t_cpu_info *c = cpu();
    uint32_t end = 576;  /* legacy region 512 + header 64 */
    for (int i = 2; i < 63; i++)
        if ((mask >> i & 1) && c->comp[i].valid && !c->comp[i].supervisor && c->comp[i].offset + c->comp[i].size > end)
            end = c->comp[i].offset + c->comp[i].size;
    return end;
}
/* SDM Vol.1 13.4.3: compacted format places components i >= 2 present in RFBM in ascending order,
 * each at the previous end, rounded up to 64 if CPUID.(0Dh,i):ECX[1]. */
uint32_t xsave_compacted_offset(uint64_t mask, int comp) {
    const t_cpu_info *c = cpu();
    if (comp == 0) return 0;
    if (comp == 1) return 160;
    uint32_t off = 576;
    for (int i = 2; i < 63; i++) {
        if (!(mask >> i & 1) || !c->comp[i].valid) continue;
        if (c->comp[i].align64) off = (off + 63) & ~63u;
        if (i == comp) return off;
        off += c->comp[i].size;
    }
    return 0;
}
uint32_t xsave_compacted_size(uint64_t mask) {
    const t_cpu_info *c = cpu();
    uint32_t off = 576;
    for (int i = 2; i < 63; i++) {
        if (!(mask >> i & 1) || !c->comp[i].valid) continue;
        if (c->comp[i].align64) off = (off + 63) & ~63u;
        off += c->comp[i].size;
    }
    return off;
}
