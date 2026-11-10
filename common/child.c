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

int t_run_self(const char *args, DWORD timeout_ms, t_child_result *r) {
    memset(r, 0, sizeof *r);
    char *cmd = (char *)malloc(8192);
    if (!cmd) return 0;
    snprintf(cmd, 8192, "\"%s\" %s", t_self_path(), args ? args : "");

    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 16)) { free(cmd); return 0; }
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
    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL, &si, &pi);
    free(cmd);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) { CloseHandle(rd); if (job) CloseHandle(job); return 0; }
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

    char v[128];
    if (t_child_find(r, "T_CHILD_EXCEPTION ", v, sizeof v)) {
        unsigned long c = 0; unsigned long long a = 0;
        if (sscanf(v, "code=0x%lx addr=0x%llx", &c, &a) >= 1) { r->exc_code = (DWORD)c; r->exc_addr = a; }
    } else if (!r->timed_out && (r->exit_code & 0xF0000000u) == 0xC0000000u) {
        r->exc_code = r->exit_code;   /* died of an exception without reaching our filter */
    }
    t_log("child '%s' exit=0x%lx exc=0x%lx timeout=%d %.1f ms", args ? args : "", r->exit_code, r->exc_code, r->timed_out, r->ms);
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
