/* env.c - environment recorder (see env.h).
 * GetEnabledXStateFeatures (kernelbase, Win10 19041 decomp) = RtlGetEnabledExtendedFeatures(~0), or 3
 * when that returns 0; RtlGetEnabledExtendedFeatures(m) = (KUSER_SHARED_DATA.XState.EnabledFeatures |
 * .EnabledSupervisorFeatures) & m. So the API may include supervisor bits (e.g. CET_U) that are not in
 * XCR0: compare against XCR0 accounting for that (recorded here, judged by the tests). */
#include "t.h"

static t_env g_env;
static int g_env_done;

static void reg_str(HKEY root, const char *path, const char *name, char *out, size_t cap) {
    HKEY k; out[0] = 0;
    if (RegOpenKeyExA(root, path, 0, KEY_READ | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return;
    DWORD type = 0, sz = (DWORD)cap - 1;
    if (RegQueryValueExA(k, name, NULL, &type, (BYTE *)out, &sz) == ERROR_SUCCESS && type == REG_SZ) out[sz < cap ? sz : cap - 1] = 0;
    else out[0] = 0;
    RegCloseKey(k);
}
static DWORD reg_dword(HKEY root, const char *path, const char *name) {
    HKEY k; DWORD v = 0, type = 0, sz = sizeof v;
    if (RegOpenKeyExA(root, path, 0, KEY_READ | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return 0;
    if (RegQueryValueExA(k, name, NULL, &type, (BYTE *)&v, &sz) != ERROR_SUCCESS || type != REG_DWORD) v = 0;
    RegCloseKey(k);
    return v;
}

const t_env *t_env_get(void) {
    if (g_env_done) return &g_env;
    t_env *e = &g_env;
    memset(e, 0, sizeof *e);
    typedef LONG (WINAPI *rgv_t)(OSVERSIONINFOW *);
    HMODULE nt = GetModuleHandleA("ntdll.dll"), k32 = GetModuleHandleA("kernel32.dll");
    rgv_t rgv = nt ? (rgv_t)(void *)GetProcAddress(nt, "RtlGetVersion") : NULL;
    OSVERSIONINFOW vi; memset(&vi, 0, sizeof vi); vi.dwOSVersionInfoSize = sizeof vi;
    if (rgv && rgv(&vi) == 0) { e->os_major = vi.dwMajorVersion; e->os_minor = vi.dwMinorVersion; e->os_build = vi.dwBuildNumber; }
    const char *cv = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    e->os_ubr = reg_dword(HKEY_LOCAL_MACHINE, cv, "UBR");
    char prod[96], disp[32];
    reg_str(HKEY_LOCAL_MACHINE, cv, "ProductName", prod, sizeof prod);
    reg_str(HKEY_LOCAL_MACHINE, cv, "DisplayVersion", disp, sizeof disp);
    snprintf(e->os_product, sizeof e->os_product, "%s%s%s", prod, disp[0] ? " " : "", disp);
    e->process_arch = T_ARCH;
    typedef BOOL (WINAPI *w2_t)(HANDLE, USHORT *, USHORT *);
    w2_t w2 = k32 ? (w2_t)(void *)GetProcAddress(k32, "IsWow64Process2") : NULL;
    USHORT pm = 0, nm = 0;
    if (w2 && w2(GetCurrentProcess(), &pm, &nm)) { e->wow64 = pm != 0; e->native_machine = nm; }
    else { BOOL w = FALSE; if (IsWow64Process(GetCurrentProcess(), &w)) e->wow64 = w ? 1 : 0; }
    snprintf(e->wine, sizeof e->wine, "%s", t_wine_version());
    e->ncpu = t_cpu_count();
    DWORD_PTR am = 0, sm = 0; if (GetProcessAffinityMask(GetCurrentProcess(), &am, &sm)) e->affinity = am;
    typedef WORD (WINAPI *gapg_t)(void);
    gapg_t gapg = k32 ? (gapg_t)(void *)GetProcAddress(k32, "GetActiveProcessorGroupCount") : NULL;
    e->active_groups = gapg ? gapg() : 1;
    e->qpc_freq = t_qpc_freq();
    typedef DWORD64 (WINAPI *gef_t)(void);
    gef_t gef = k32 ? (gef_t)(void *)GetProcAddress(k32, "GetEnabledXStateFeatures") : NULL;
    if (gef) { e->api_xstate_present = 1; e->api_xstate_features = gef(); }
    for (int i = 0; i < 64; i++) e->pf[i] = IsProcessorFeaturePresent((DWORD)i) ? 1 : 0;
    g_env_done = 1;
    return e;
}

static void jstr(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"' || *p == '\\') fprintf(f, "\\%c", *p);
        else if (*p < 0x20 || *p >= 0x7f) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}

void t_env_print(FILE *f) {
    const t_env *e = t_env_get(); const t_cpu_info *c = cpu();
    static char feats[4096];
    fprintf(f, "ENV   exe=%s arch=%s wow64=%d os=%lu.%lu.%lu.%lu (%s)%s%s\n", t_exe_name(), T_ARCH, e->wow64,
            e->os_major, e->os_minor, e->os_build, e->os_ubr, e->os_product, e->wine[0] ? " wine=" : "", e->wine);
    fprintf(f, "ENV   cpu=\"%s\" vendor=%s family=0x%x model=0x%x stepping=%u hypervisor=%d%s%s ncpu=%d\n",
            c->brand, c->vendor, c->family, c->model, c->stepping, c->hypervisor, c->hv_vendor[0] ? " hv=" : "", c->hv_vendor, e->ncpu);
    fprintf(f, "ENV   osxsave=%d xcr0=0x%llx xcr0_supported=0x%llx xss_supported=0x%llx xsave_size=%u/%u api_xstate=0x%llx%s\n",
            c->osxsave, (unsigned long long)c->xcr0, (unsigned long long)c->xcr0_supported, (unsigned long long)c->xss_supported,
            c->xsave_size_enabled, c->xsave_size_max, e->api_xstate_features, e->api_xstate_present ? "" : " (API missing)");
    cpu_features_string(feats, sizeof feats, 1);
    fprintf(f, "ENV   usable: %s\n", feats);
    fflush(f);
}

void t_env_write_json(FILE *f, const char *exe, const char *version) {
    const t_env *e = t_env_get(); const t_cpu_info *c = cpu();
    static char feats[4096];
    SYSTEMTIME st; GetSystemTime(&st);
    char ts[40]; snprintf(ts, sizeof ts, "%04u-%02u-%02uT%02u:%02u:%02uZ", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    fprintf(f, "{\"type\":\"env\",\"exe\":"); jstr(f, exe);
    fprintf(f, ",\"version\":"); jstr(f, version);
    fprintf(f, ",\"compiler\":"); jstr(f, "gcc " __VERSION__);
    fprintf(f, ",\"arch\":\"%s\",\"wow64\":%s,\"native_machine\":%u,\"timestamp\":\"%s\"", T_ARCH, e->wow64 ? "true" : "false", e->native_machine, ts);
    fprintf(f, ",\"os\":{\"major\":%lu,\"minor\":%lu,\"build\":%lu,\"ubr\":%lu,\"product\":", e->os_major, e->os_minor, e->os_build, e->os_ubr);
    jstr(f, e->os_product); fprintf(f, "},\"wine\":"); jstr(f, e->wine);
    fprintf(f, ",\"cpu\":{\"vendor\":"); jstr(f, c->vendor);
    fprintf(f, ",\"brand\":"); jstr(f, c->brand);
    fprintf(f, ",\"signature\":\"0x%08x\",\"family\":%u,\"model\":%u,\"stepping\":%u,\"max_basic\":\"0x%x\",\"max_ext\":\"0x%x\",\"hypervisor\":%s,\"hv_vendor\":",
            c->signature, c->family, c->model, c->stepping, c->max_basic, c->max_ext, c->hypervisor ? "true" : "false");
    jstr(f, c->hv_vendor);
    fprintf(f, ",\"avx10_version\":%d}", c->avx10_version);
    cpu_features_string(feats, sizeof feats, 0);
    fprintf(f, ",\"features_hw\":"); jstr(f, feats);
    cpu_features_string(feats, sizeof feats, 1);
    fprintf(f, ",\"features_usable\":"); jstr(f, feats);
    fprintf(f, ",\"osxsave\":%s,\"xcr0\":", c->osxsave ? "true" : "false");
    if (c->xcr0_valid) fprintf(f, "\"0x%llx\"", (unsigned long long)c->xcr0); else fprintf(f, "null");
    fprintf(f, ",\"xinuse\":");
    if (c->xinuse_valid) fprintf(f, "\"0x%llx\"", (unsigned long long)c->xinuse); else fprintf(f, "null");
    fprintf(f, ",\"xcr0_supported\":\"0x%llx\",\"xss_supported\":\"0x%llx\"",
            (unsigned long long)c->xcr0_supported, (unsigned long long)c->xss_supported);
    fprintf(f, ",\"xsave_size_enabled\":%u,\"xsave_size_max\":%u,\"xsave_size_compacted\":%u",
            c->xsave_size_enabled, c->xsave_size_max, c->xsave_size_compacted);
    fprintf(f, ",\"api_xstate_present\":%s,\"api_xstate_features\":\"0x%llx\"", e->api_xstate_present ? "true" : "false", e->api_xstate_features);
    fprintf(f, ",\"pf_present\":[");
    for (int i = 0, n = 0; i < 64; i++) if (e->pf[i]) fprintf(f, "%s%d", n++ ? "," : "", i);
    fprintf(f, "],\"ncpu\":%d,\"affinity\":\"0x%llx\",\"groups\":%u,\"qpc_freq\":%llu}\n", e->ncpu, e->affinity, e->active_groups, e->qpc_freq);
}
