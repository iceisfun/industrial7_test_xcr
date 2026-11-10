/* child.c - re-exec self as a child process with timeout, output capture and exception reporting. */
#include "t.h"

static HANDLE make_job(void) {
    HANDLE j = CreateJobObjectA(NULL, NULL);
    if (!j) return NULL;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li;
    memset(&li, 0, sizeof li);
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
    if (!SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof li)) { CloseHandle(j); return NULL; }
    return j;
}

static void drain(HANDLE rd, t_child_result *r) {
    for (;;) {
        DWORD avail = 0, got = 0;
        if (!PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) || !avail) return;
        char buf[4096];
        if (!ReadFile(rd, buf, avail < sizeof buf ? avail : (DWORD)sizeof buf, &got, NULL) || !got) return;
        size_t room = sizeof r->out - 1 - r->out_len;
        size_t n = got < room ? got : room;
        memcpy(r->out + r->out_len, buf, n);
        r->out_len += n; r->out[r->out_len] = 0;
    }
}

/* Under tools/sde_run.sh, SDE writes its error reports (chip-check stops, failed emulated accesses) to
 * files in XCR_SDE_LOGDIR_WIN (one per Pin process: sde-chip-check[.<ppid>-<pid>].txt, sde-error*.txt), not
 * to the child's stdout/stderr pipe. Snapshot the file sizes around a child run; error text from files
 * that changed is appended to r->out so that t_sde_killed_child() can see it. Inactive outside SDE runs. */
typedef struct sde_snap { int n; struct { char name[96]; DWORD size; FILETIME t; } f[256]; } sde_snap;
static void sde_snap_take(sde_snap *sn, char *dir, size_t cap) {
    sn->n = 0; dir[0] = 0;
    if (!t_sde_model()) return;
    DWORD n = GetEnvironmentVariableA("XCR_SDE_LOGDIR_WIN", dir, (DWORD)cap);
    if (n == 0 || n >= cap - 16) { dir[0] = 0; return; }
    char pat[MAX_PATH + 16]; snprintf(pat, sizeof pat, "%s\\sde-*.txt", dir);
    WIN32_FIND_DATAA fd; HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (sn->n >= (int)T_COUNTOF(sn->f) || strstr(fd.cFileName, "align-checker") || strlen(fd.cFileName) >= sizeof sn->f[0].name) continue;
        snprintf(sn->f[sn->n].name, sizeof sn->f[0].name, "%s", fd.cFileName);
        sn->f[sn->n].size = fd.nFileSizeLow; sn->f[sn->n].t = fd.ftLastWriteTime; sn->n++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}
static void sde_snap_append(t_child_result *r, const sde_snap *before, char *dir) {
    if (!dir[0]) return;
    sde_snap *ap = (sde_snap *)malloc(sizeof *ap);
    if (!ap) return;
    char d2[MAX_PATH]; sde_snap_take(ap, d2, sizeof d2);
#define after (*ap)
    for (int i = 0; i < after.n; i++) {
        int changed = 1;
        for (int j = 0; j < before->n; j++)
            if (!strcmp(after.f[i].name, before->f[j].name))
                changed = after.f[i].size != before->f[j].size || CompareFileTime(&after.f[i].t, &before->f[j].t) != 0;
        if (!changed) continue;
        char path[MAX_PATH + 100]; snprintf(path, sizeof path, "%s\\%s", dir, after.f[i].name);
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, 0, NULL);
        if (h == INVALID_HANDLE_VALUE) continue;
        char buf[2048]; DWORD got = 0;
        if (!ReadFile(h, buf, sizeof buf - 1, &got, NULL)) got = 0;
        CloseHandle(h);
        buf[got] = 0;
        const char *m = strstr(buf, "SDE-ERROR");
        if (!m) m = strstr(buf, "SDE ERROR");
        if (!m || r->out_len + 64 >= sizeof r->out) continue;
        snprintf(r->out + r->out_len, sizeof r->out - r->out_len, "\n[%s] %.400s", after.f[i].name, m);
        r->out_len += strlen(r->out + r->out_len);
    }
#undef after
    free(ap);
}

int t_run_self(const char *args, DWORD timeout_ms, t_child_result *r) {
    memset(r, 0, sizeof *r);
    char *cmd = (char *)malloc(8192);
    if (!cmd) return 0;
    snprintf(cmd, 8192, "\"%s\" %s", t_self_path(), args ? args : "");

    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 16)) { r->spawn_error = GetLastError(); free(cmd); return 0; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);

    STARTUPINFOA si; PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si); si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul != INVALID_HANDLE_VALUE ? nul : NULL;
    si.hStdOutput = wr; si.hStdError = wr;
    memset(&pi, 0, sizeof pi);
    HANDLE job = make_job();
    LARGE_INTEGER t0, t1, f; QueryPerformanceCounter(&t0); QueryPerformanceFrequency(&f);
    BOOL ok = FALSE;
    for (int attempt = 0; attempt < 3 && !ok; attempt++) {   /* tolerate transient failures on loaded hosts */
        if (attempt) Sleep(200);
        ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL, &si, &pi);
        if (!ok) { r->spawn_error = GetLastError(); t_log("CreateProcess failed (attempt %d): error %lu", attempt + 1, r->spawn_error); }
    }
    free(cmd);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) { CloseHandle(rd); if (job) CloseHandle(job); return 0; }
    r->spawn_error = 0;
    /* the child is still suspended: everything SDE reports from here on belongs to this run */
    sde_snap *sde_before = t_sde_model() ? (sde_snap *)malloc(sizeof *sde_before) : NULL;
    char sde_dir[MAX_PATH]; sde_dir[0] = 0;
    if (sde_before) sde_snap_take(sde_before, sde_dir, sizeof sde_dir);
    if (job && !AssignProcessToJobObject(job, pi.hProcess)) { CloseHandle(job); job = NULL; }  /* nested jobs unsupported: fine */
    ResumeThread(pi.hThread);
    r->started = 1;

    for (;;) {
        DWORD w = WaitForSingleObject(pi.hProcess, 10);
        drain(rd, r);
        if (w == WAIT_OBJECT_0) break;
        QueryPerformanceCounter(&t1);
        if ((double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart > (double)timeout_ms) {
            r->timed_out = 1;
            if (job) TerminateJobObject(job, 0xDEAD0001); else TerminateProcess(pi.hProcess, 0xDEAD0001);
            WaitForSingleObject(pi.hProcess, 5000);
            break;
        }
    }
    drain(rd, r);
    QueryPerformanceCounter(&t1);
    r->ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart;
    GetExitCodeProcess(pi.hProcess, &r->exit_code);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess); CloseHandle(rd);
    if (job) CloseHandle(job);    /* kill-on-close: grandchildren die here */
    if (sde_before) { sde_snap_append(r, sde_before, sde_dir); free(sde_before); }

    char v[128];
    if (t_child_find(r, "T_CHILD_EXCEPTION ", v, sizeof v)) {
        unsigned long c = 0; unsigned long long a = 0;
        if (sscanf(v, "code=0x%lx addr=0x%llx", &c, &a) >= 1) { r->exc_code = (DWORD)c; r->exc_addr = a; }
    } else if (!r->timed_out && (r->exit_code & 0xF0000000u) == 0xC0000000u) {
        r->exc_code = r->exit_code;   /* died of an exception without reaching our filter */
    }
    t_log("child '%s' exit=0x%lx exc=0x%lx timeout=%d %.1f ms", args ? args : "", r->exit_code, r->exc_code, r->timed_out, r->ms);
    if (t_sde_killed_child(r)) {
        /* Under tools/sde_run.sh only, and only when SDE itself reported that it stopped the child (its
         * chip-check terminates a process that executes an instruction the emulated CPU or XCR0 lacks,
         * where hardware raises #UD): the case's failures become XFAIL. Checks are unchanged. */
        const char *m = strstr(r->out, "SDE-ERROR");
        if (!m) m = strstr(r->out, "SDE ERROR");
        if (!m) m = strstr(r->out, "Tool (or Pin) caused signal");
        char line[160]; size_t n = 0;
        while (m && m[n] && m[n] != '\n' && m[n] != '\r' && n < sizeof line - 1) { line[n] = m[n]; n++; }
        line[n] = 0;
        t_expect_fail("Intel SDE (%s) terminated the child instead of raising the architectural exception: %s", t_sde_model(), line);
    }
    return 1;
}

int t_child_run(const char *name, const char *args, DWORD timeout_ms, t_child_result *r) {
    char a[2048];
    snprintf(a, sizeof a, "--child=%s %s", name, args ? args : "");
    return t_run_self(a, timeout_ms, r);
}

int t_child_find(const t_child_result *r, const char *prefix, char *val, size_t cap) {
    size_t pl = strlen(prefix);
    const char *p = r->out;
    while (p && *p) {
        if (!strncmp(p, prefix, pl)) {
            const char *s = p + pl, *e = s;
            while (*e && *e != '\r' && *e != '\n') e++;
            size_t n = (size_t)(e - s); if (n >= cap) n = cap - 1;
            if (cap) { memcpy(val, s, n); val[n] = 0; }
            return 1;
        }
        p = strchr(p, '\n'); if (p) p++;
    }
    return 0;
}
