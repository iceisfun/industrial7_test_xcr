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

/* Under tools/sde_run.sh, SDE writes its error reports to XCR_SDE_ERRFILE (a Windows path), not to the
 * child's stdout/stderr pipe. Snapshot it around a child run; new text is appended to r->out so that
 * t_sde_killed_child() can see it. Never active outside SDE runs. */
static size_t sde_errfile_read(char *buf, size_t cap) {
    char path[MAX_PATH];
    buf[0] = 0;
    if (!t_sde_model()) return 0;
    DWORD n = GetEnvironmentVariableA("XCR_SDE_ERRFILE", path, sizeof path);
    if (n == 0 || n >= sizeof path) return 0;
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)cap - 1, &got, NULL)) got = 0;
    CloseHandle(h);
    buf[got] = 0;
    return got;
}
static void sde_errfile_append(t_child_result *r, const char *before, size_t nb) {
    char after[4096];
    size_t na = sde_errfile_read(after, sizeof after);
    if (!na || (na == nb && !memcmp(after, before, na))) return;
    const char *add = (na > nb && !memcmp(after, before, nb)) ? after + nb : after;
    snprintf(r->out + r->out_len, sizeof r->out - r->out_len, "\n[sde errfile] %s", add);
    r->out_len += strlen(r->out + r->out_len);
}

int t_run_self(const char *args, DWORD timeout_ms, t_child_result *r) {
    char sde_before[4096];
    memset(r, 0, sizeof *r);
    size_t sde_nb = sde_errfile_read(sde_before, sizeof sde_before);
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
    sde_errfile_append(r, sde_before, sde_nb);

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
