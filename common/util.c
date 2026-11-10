/* util.c - run-time export lookup, guarded buffers, temp file names. */
#include "t.h"

void *t_proc(const char *dll, const char *name) {
    HMODULE m = GetModuleHandleA(dll);
    if (!m) m = LoadLibraryA(dll);
    return m ? (void *)GetProcAddress(m, name) : NULL;
}

void *t_guarded_alloc(size_t n, void **base) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    size_t pg = si.dwPageSize ? si.dwPageSize : 4096;
    size_t data = (n + pg - 1) / pg * pg;
    if (base) *base = NULL;
    unsigned char *b = (unsigned char *)VirtualAlloc(NULL, data + pg, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!b) return NULL;
    DWORD old;
    if (!VirtualProtect(b + data, pg, PAGE_NOACCESS, &old)) { VirtualFree(b, 0, MEM_RELEASE); return NULL; }
    if (base) *base = b;
    return b + data - n;
}
void t_guarded_free(void *base) { if (base) VirtualFree(base, 0, MEM_RELEASE); }

void t_tmpfile(char *out, size_t cap, const char *leaf) {
    static volatile LONG seq;
    char tmp[MAX_PATH];
    if (!GetTempPathA(sizeof tmp, tmp)) strcpy(tmp, ".\\");
    snprintf(out, cap, "%sxcr_%s_%lu_%ld_%s", tmp, t_exe_name(), GetCurrentProcessId(), (long)InterlockedIncrement(&seq), leaf ? leaf : "tmp");
}
