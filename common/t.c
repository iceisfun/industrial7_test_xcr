/* t.c - harness core: registration, per-case execution with crash recovery, result recording,
 * human + JSON-lines output, CLI.  See t.h / API.md for the contract. */
#include "t.h"
#include <math.h>

#ifndef T_VERSION
#define T_VERSION "unknown"
#endif

/* ---- small growable string buffer (internal) --------------------------- */
typedef struct { char *p; size_t n, cap; } sb_t;
static void sb_reserve(sb_t *b, size_t add) {
    if (b->n + add + 1 <= b->cap) return;
    size_t nc = b->cap ? b->cap * 2 : 256;
    while (nc < b->n + add + 1) nc *= 2;
    char *np = (char *)realloc(b->p, nc);
    if (!np) return;
    b->p = np; b->cap = nc;
}
static void sb_vaddf(sb_t *b, const char *fmt, va_list ap) {
    va_list a2; va_copy(a2, ap);
    int need = vsnprintf(NULL, 0, fmt, a2); va_end(a2);
    if (need < 0) return;
    sb_reserve(b, (size_t)need);
    if (b->n + (size_t)need + 1 > b->cap) return;
    vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    b->n += (size_t)need;
}
static void sb_addf(sb_t *b, const char *fmt, ...) { va_list a; va_start(a, fmt); sb_vaddf(b, fmt, a); va_end(a); }
static void sb_json_str(sb_t *b, const char *s) {
    sb_addf(b, "\"");
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"' || *p == '\\') sb_addf(b, "\\%c", *p);
        else if (*p == '\n') sb_addf(b, "\\n");
        else if (*p == '\r') sb_addf(b, "\\r");
        else if (*p == '\t') sb_addf(b, "\\t");
        else if (*p < 0x20 || *p >= 0x7f) sb_addf(b, "\\u%04x", *p);  /* keep the file pure ASCII */
        else sb_addf(b, "%c", *p);
    }
    sb_addf(b, "\"");
}
static void sb_reset(sb_t *b) { b->n = 0; if (b->p) b->p[0] = 0; }
static const char *sb_str(sb_t *b) { return b->p ? b->p : ""; }

/* JSON number: never emit nan/inf (invalid JSON) */
static void sb_num(sb_t *b, double v) {
    if (v != v || v > 1.7e308 || v < -1.7e308) sb_addf(b, "null"); else sb_addf(b, "%.6g", v);
}

/* ---- registry ---------------------------------------------------------- */
#define MAX_CASES 2048
#define MAX_CHILDREN 512
static struct { const char *n; t_fn f; int flags, line, seq; } g_t[MAX_CASES];
static int g_nt;
static struct { const char *n; t_child_fn f; } g_ch[MAX_CHILDREN];
static int g_nch;
void t_register(const char *n, t_fn f, int flags, int line) {
    if (g_nt < MAX_CASES) { g_t[g_nt].n = n; g_t[g_nt].f = f; g_t[g_nt].flags = flags; g_t[g_nt].line = line; g_t[g_nt].seq = g_nt; g_nt++; }
}
void t_register_child(const char *n, t_child_fn f) { if (g_nch < MAX_CHILDREN) { g_ch[g_nch].n = n; g_ch[g_nch].f = f; g_nch++; } }

/* ---- global run state ---------------------------------------------------- */
static CRITICAL_SECTION g_lock;
static FILE *g_json;
static int g_verbose, g_child_mode, g_include_manual;
static char g_exe[128];
static DWORD g_case_timeout_ms = 120000;

enum { R_PASS, R_FAIL, R_SKIP, R_XFAIL, R_CRASH, R_PERF, R_N };
static const char *const R_NAME[R_N] = { "PASS", "FAIL", "SKIP", "XFAIL", "CRASH", "PERF" };
static int g_count[R_N];
static long g_checks_total, g_checks_failed;

static struct {
    const char *name;
    volatile LONG checks, fails;
    volatile LONG active;           /* 1 while the case body runs */
    DWORD tid;
    int skip, xfail_now, expect_fail, crashed, nperf, nfail_detail, finished;
    DWORD exc_code; unsigned long long exc_addr; int exc_other_thread;
    char reason[600];
    sb_t fails_j, notes_j, info_j, perf_j;
    LARGE_INTEGER t0;
    volatile LONG seq;
} g_c;

static void *g_jb[5];              /* __builtin_setjmp buffer for the case thread */
#ifdef T_X86
static void *g_seh_head;           /* FS:[0] at case start (restored after an abort/crash) */
#endif

static double qpc_ms_since(LARGE_INTEGER t0) {
    LARGE_INTEGER t1, f; QueryPerformanceCounter(&t1); QueryPerformanceFrequency(&f);
    return (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart;
}

static void out_line(const char *fmt, ...) T_PRINTF(1, 2);
static void out_line(const char *fmt, ...) {
    va_list a; va_start(a, fmt); vprintf(fmt, a); va_end(a); fputc('\n', stdout); fflush(stdout);
}

/* ---- queries -------------------------------------------------------------- */
int t_verbose(void) { return g_verbose; }
const char *t_case_name(void) { return g_c.name ? g_c.name : ""; }
const char *t_exe_name(void) { return g_exe; }
int t_on_case_thread(void) { return g_c.active && GetCurrentThreadId() == g_c.tid; }
const char *t_self_path(void) {
    static char p[MAX_PATH * 2];
    if (!p[0]) GetModuleFileNameA(NULL, p, sizeof p);
    return p;
}
int t_is_wine(void) {
    HMODULE n = GetModuleHandleA("ntdll.dll");
    return n && GetProcAddress(n, "wine_get_version") != NULL;
}
const char *t_wine_version(void) {
    typedef const char *(CDECL *pfn)(void);
    HMODULE n = GetModuleHandleA("ntdll.dll");
    pfn f = n ? (pfn)(void *)GetProcAddress(n, "wine_get_version") : NULL;
    return f ? f() : "";
}

/* ---- recording ------------------------------------------------------------ */
static void set_reason(const char *s) { if (!g_c.reason[0]) snprintf(g_c.reason, sizeof g_c.reason, "%s", s); }

void t_check(int ok, const char *file, int line, const char *expr, const char *fmt, ...) {
    InterlockedIncrement(&g_c.checks);
    if (ok) return;
    InterlockedIncrement(&g_c.fails);
    char msg[1024] = "";
    if (fmt) { va_list a; va_start(a, fmt); vsnprintf(msg, sizeof msg, fmt, a); va_end(a); }
    const char *base = file; for (const char *p = file; *p; p++) if (*p == '/' || *p == '\\') base = p + 1;
    EnterCriticalSection(&g_lock);
    if (g_c.nfail_detail < 50) {
        out_line("  FAIL %s:%d: %s%s%s%s", base, line, expr, msg[0] ? " (" : "", msg, msg[0] ? ")" : "");
        if (g_c.fails_j.n) sb_addf(&g_c.fails_j, ",");
        sb_addf(&g_c.fails_j, "{\"file\":"); sb_json_str(&g_c.fails_j, base);
        sb_addf(&g_c.fails_j, ",\"line\":%d,\"expr\":", line); sb_json_str(&g_c.fails_j, expr);
        sb_addf(&g_c.fails_j, ",\"msg\":"); sb_json_str(&g_c.fails_j, msg); sb_addf(&g_c.fails_j, "}");
        if (!g_c.reason[0]) snprintf(g_c.reason, sizeof g_c.reason, "%s:%d: %s%s%s", base, line, expr, msg[0] ? ": " : "", msg);
    } else if (g_c.nfail_detail == 50) out_line("  FAIL ... further failures not printed");
    g_c.nfail_detail++;
    LeaveCriticalSection(&g_lock);
}

int t_check_mem(const void *got, const void *want, size_t n, const char *file, int line, const char *expr) {
    const unsigned char *g = (const unsigned char *)got, *w = (const unsigned char *)want;
    size_t i = 0;
    while (i < n && g[i] == w[i]) i++;
    if (i == n) { t_check(1, file, line, expr, NULL); return 1; }
    size_t bad = 0; for (size_t j = i; j < n; j++) bad += g[j] != w[j];
    t_check(0, file, line, expr, "%zu of %zu bytes differ, first at offset %zu", bad, n, i);
    size_t win = i & ~(size_t)15, len = n - win < 64 ? n - win : 64;
    char a[512], b[512], m[512];
    t_hex(a, sizeof a, g + win, len); t_hex(b, sizeof b, w + win, len);
    size_t k = 0;
    for (size_t j = 0; j < len && k + 4 < sizeof m; j++) {      /* marker line aligned with t_hex output */
        int d = g[win + j] != w[win + j];
        m[k++] = d ? '^' : ' '; m[k++] = d ? '^' : ' ';
        if (j + 1 < len && ((j + 1) % 4) == 0) m[k++] = ' ';
    }
    m[k] = 0;
    EnterCriticalSection(&g_lock);
    out_line("       got  @%04zx: %s", win, a);
    out_line("       want @%04zx: %s", win, b);
    out_line("                   %s", m);
    LeaveCriticalSection(&g_lock);
    return 0;
}

void t_note(const char *fmt, ...) {
    char msg[1024]; va_list a; va_start(a, fmt); vsnprintf(msg, sizeof msg, fmt, a); va_end(a);
    EnterCriticalSection(&g_lock);
    out_line("  NOTE %s", msg);
    if (g_c.notes_j.n) sb_addf(&g_c.notes_j, ",");
    sb_json_str(&g_c.notes_j, msg);
    LeaveCriticalSection(&g_lock);
}
void t_info(const char *key, const char *fmt, ...) {
    char msg[1024]; va_list a; va_start(a, fmt); vsnprintf(msg, sizeof msg, fmt, a); va_end(a);
    EnterCriticalSection(&g_lock);
    out_line("  INFO %s = %s", key, msg);
    if (g_c.info_j.n) sb_addf(&g_c.info_j, ",");
    sb_json_str(&g_c.info_j, key); sb_addf(&g_c.info_j, ":"); sb_json_str(&g_c.info_j, msg);
    LeaveCriticalSection(&g_lock);
}
void t_log(const char *fmt, ...) {
    if (!g_verbose) return;
    char msg[2048]; va_list a; va_start(a, fmt); vsnprintf(msg, sizeof msg, fmt, a); va_end(a);
    EnterCriticalSection(&g_lock); out_line("  .. %s", msg); LeaveCriticalSection(&g_lock);
}
static void perf_add(const char *metric, double v, const char *unit, const t_stats *s) {
    EnterCriticalSection(&g_lock);
    if (s) out_line("  PERF %s = %.6g %s (n=%d min=%.6g max=%.6g sd=%.3g)", metric, v, unit, s->n, s->min, s->max, s->stddev);
    else   out_line("  PERF %s = %.6g %s", metric, v, unit);
    if (g_c.perf_j.n) sb_addf(&g_c.perf_j, ",");
    sb_addf(&g_c.perf_j, "{\"metric\":"); sb_json_str(&g_c.perf_j, metric);
    sb_addf(&g_c.perf_j, ",\"value\":"); sb_num(&g_c.perf_j, v);
    sb_addf(&g_c.perf_j, ",\"unit\":"); sb_json_str(&g_c.perf_j, unit);
    if (s) {
        sb_addf(&g_c.perf_j, ",\"n\":%d,\"min\":", s->n); sb_num(&g_c.perf_j, s->min);
        sb_addf(&g_c.perf_j, ",\"max\":"); sb_num(&g_c.perf_j, s->max);
        sb_addf(&g_c.perf_j, ",\"mean\":"); sb_num(&g_c.perf_j, s->mean);
        sb_addf(&g_c.perf_j, ",\"stddev\":"); sb_num(&g_c.perf_j, s->stddev);
    }
    sb_addf(&g_c.perf_j, "}");
    g_c.nperf++;
    LeaveCriticalSection(&g_lock);
}
void t_perf(const char *metric, double value, const char *unit) { perf_add(metric, value, unit, NULL); }
void t_perf_stats(const char *metric, const struct t_stats *s, const char *unit) { perf_add(metric, s->median, unit, s); }

/* ---- leaving a case early -------------------------------------------------- */
T_NORETURN void t_abort_case(void) {
    if (t_on_case_thread()) __builtin_longjmp(g_jb, 1);
    ExitThread(0xAB0127);          /* worker thread: result already recorded; end only this thread */
}
void t_skip(const char *fmt, ...) {
    char msg[600]; va_list a; va_start(a, fmt); vsnprintf(msg, sizeof msg, fmt, a); va_end(a);
    EnterCriticalSection(&g_lock); out_line("  SKIP %s", msg); g_c.skip = 1; set_reason(msg); LeaveCriticalSection(&g_lock);
    t_abort_case();
}
void t_xfail(const char *fmt, ...) {
    char msg[600]; va_list a; va_start(a, fmt); vsnprintf(msg, sizeof msg, fmt, a); va_end(a);
    EnterCriticalSection(&g_lock); out_line("  XFAIL %s", msg); g_c.xfail_now = 1; set_reason(msg); LeaveCriticalSection(&g_lock);
    t_abort_case();
}
void t_expect_fail(const char *fmt, ...) {
    char msg[600]; va_list a; va_start(a, fmt); vsnprintf(msg, sizeof msg, fmt, a); va_end(a);
    EnterCriticalSection(&g_lock);
    out_line("  EXPECT-FAIL %s", msg);
    if (!g_c.expect_fail) { g_c.expect_fail = 1; snprintf(g_c.reason, sizeof g_c.reason, "%s", msg); }
    LeaveCriticalSection(&g_lock);
}

/* ---- result emission --------------------------------------------------------- */
static int classify(void) {
    if (g_c.crashed) return R_CRASH;
    if (g_c.xfail_now) return R_XFAIL;
    if (g_c.fails) return g_c.expect_fail ? R_XFAIL : R_FAIL;
    if (g_c.skip) return R_SKIP;
    if (g_c.nperf) return R_PERF;
    return R_PASS;
}

/* Emits the record for the current case (once). Caller holds g_lock. */
static int emit_case(double ms) {
    if (g_c.finished) return -1;
    g_c.finished = 1;
    int r = classify();
    if (r == R_PASS && g_c.expect_fail) {
        out_line("  NOTE XPASS: expected failure did not happen (%s)", g_c.reason);
        if (g_c.notes_j.n) sb_addf(&g_c.notes_j, ",");
        sb_json_str(&g_c.notes_j, "XPASS: expected failure did not happen");
        g_c.reason[0] = 0;
    }
    g_count[r]++;
    g_checks_total += g_c.checks; g_checks_failed += g_c.fails;
    if (r == R_PASS || r == R_PERF) out_line("%-5s %s (%.1f ms)", R_NAME[r], g_c.name, ms);
    else out_line("%-5s %s (%.1f ms): %s", R_NAME[r], g_c.name, ms, g_c.reason);
    if (g_json) {
        sb_t b = {0};
        sb_addf(&b, "{\"type\":\"case\",\"exe\":"); sb_json_str(&b, g_exe);
        sb_addf(&b, ",\"arch\":\"%s\",\"case\":", T_ARCH); sb_json_str(&b, g_c.name);
        sb_addf(&b, ",\"result\":\"%s\",\"ms\":", R_NAME[r]); sb_num(&b, ms);
        sb_addf(&b, ",\"checks\":%ld,\"failed\":%ld,\"reason\":", (long)g_c.checks, (long)g_c.fails); sb_json_str(&b, g_c.reason);
        if (g_c.crashed) sb_addf(&b, ",\"exception\":{\"code\":\"0x%08lx\",\"addr\":\"0x%llx\",\"other_thread\":%s}",
                                 g_c.exc_code, g_c.exc_addr, g_c.exc_other_thread ? "true" : "false");
        sb_addf(&b, ",\"failures\":[%s],\"notes\":[%s],\"info\":{%s},\"perf\":[%s]}\n",
                sb_str(&g_c.fails_j), sb_str(&g_c.notes_j), sb_str(&g_c.info_j), sb_str(&g_c.perf_j));
        fputs(sb_str(&b), g_json); fflush(g_json);
        free(b.p);
    }
    return r;
}

static int exit_code_now(void) { return (g_count[R_FAIL] || g_count[R_CRASH]) ? 1 : 0; }
static LARGE_INTEGER g_run_t0;

static void emit_summary(void) {
    int total = 0; for (int i = 0; i < R_N; i++) total += g_count[i];
    int ec = exit_code_now();
    out_line("SUMMARY exe=%s arch=%s total=%d pass=%d fail=%d crash=%d skip=%d xfail=%d perf=%d checks=%ld checks_failed=%ld exit=%d",
             g_exe, T_ARCH, total, g_count[R_PASS], g_count[R_FAIL], g_count[R_CRASH], g_count[R_SKIP], g_count[R_XFAIL],
             g_count[R_PERF], g_checks_total, g_checks_failed, ec);
    if (g_json) {
        fprintf(g_json, "{\"type\":\"summary\",\"exe\":\"%s\",\"arch\":\"%s\",\"total\":%d,\"pass\":%d,\"fail\":%d,\"crash\":%d,"
                "\"skip\":%d,\"xfail\":%d,\"perf\":%d,\"checks\":%ld,\"checks_failed\":%ld,\"ms\":%.1f,\"exit_code\":%d}\n",
                g_exe, T_ARCH, total, g_count[R_PASS], g_count[R_FAIL], g_count[R_CRASH], g_count[R_SKIP], g_count[R_XFAIL],
                g_count[R_PERF], g_checks_total, g_checks_failed, qpc_ms_since(g_run_t0), ec);
        fflush(g_json);
    }
}

/* Fatal path (exception on another thread, watchdog): record CRASH for the running case, summary, exit 1. */
static T_NORETURN void fatal_finish(const char *why, DWORD code, unsigned long long addr, int other_thread) {
    /* bounded wait for the lock: a hung case thread may own it; proceed anyway after 2 s */
    for (int i = 0; i < 200 && !TryEnterCriticalSection(&g_lock); i++) Sleep(10);
    if (g_c.active && !g_c.finished) {
        g_c.crashed = 1; g_c.exc_code = code; g_c.exc_addr = addr; g_c.exc_other_thread = other_thread;
        snprintf(g_c.reason, sizeof g_c.reason, "%s", why);
        out_line("  CRASH %s", why);
        emit_case(qpc_ms_since(g_c.t0));
    } else {
        out_line("CRASH outside of a case: %s", why);
        g_count[R_CRASH]++;
    }
    emit_summary();
    if (g_json) fclose(g_json);
    fflush(stdout);
    TerminateProcess(GetCurrentProcess(), 1);
    for (;;) Sleep(1000);
}

/* ---- crash recovery ------------------------------------------------------ */
static int is_fatal_code(DWORD c) { return (c & 0xF0000000u) == 0xC0000000u || (c & 0xF0000000u) == 0x80000000u || c == 0xE06D7363u; }

static T_NOINLINE T_NORETURN void crash_landing(void) {
    __builtin_longjmp(g_jb, 1);
}

static LONG WINAPI child_uef(EXCEPTION_POINTERS *ep) {
    DWORD c = ep->ExceptionRecord->ExceptionCode;
    printf("T_CHILD_EXCEPTION code=0x%08lx addr=0x%llx\n", c, (unsigned long long)(ULONG_PTR)ep->ExceptionRecord->ExceptionAddress);
    fflush(stdout); fflush(stderr);
    TerminateProcess(GetCurrentProcess(), c);
    return EXCEPTION_CONTINUE_SEARCH;
}

/* Top-level (unhandled) exception filter: only reached when no VEH / SEH frame of the test handled it,
 * so tests remain free to use their own handlers for deliberate exceptions. */
static LONG WINAPI case_uef(EXCEPTION_POINTERS *ep) {
    DWORD c = ep->ExceptionRecord->ExceptionCode;
    unsigned long long addr = (unsigned long long)(ULONG_PTR)ep->ExceptionRecord->ExceptionAddress;
    if (!is_fatal_code(c)) return EXCEPTION_CONTINUE_SEARCH;
    if (!(g_c.active && GetCurrentThreadId() == g_c.tid)) {
        char why[200];
        snprintf(why, sizeof why, "unhandled exception 0x%08lx at 0x%llx on thread %lu (not the case thread)", c, addr, GetCurrentThreadId());
        fatal_finish(why, c, addr, 1);
    }
    g_c.crashed = 1; g_c.exc_code = c; g_c.exc_addr = addr;
    snprintf(g_c.reason, sizeof g_c.reason, "unhandled exception 0x%08lx at 0x%llx", c, addr);
    if (ep->ExceptionRecord->ExceptionFlags & EXCEPTION_NONCONTINUABLE) __builtin_longjmp(g_jb, 1);
    /* Resume at crash_landing on a fresh, aligned part of the same stack. */
    CONTEXT *ctx = ep->ContextRecord;
#ifdef T_X64
    ctx->Rsp = ((ctx->Rsp - 512) & ~(DWORD64)15) - 8;
    ctx->Rip = (DWORD64)(ULONG_PTR)crash_landing;
#else
    ctx->Esp = ((ctx->Esp - 512) & ~(DWORD)15) - 4;
    ctx->Eip = (DWORD)(ULONG_PTR)crash_landing;
#endif
    ctx->EFlags &= ~0x100u;    /* clear TF */
    return EXCEPTION_CONTINUE_EXECUTION;
}

/* ---- watchdog ------------------------------------------------------------- */
static DWORD WINAPI watchdog(void *arg) {
    (void)arg;
    for (;;) {
        Sleep(250);
        if (!g_case_timeout_ms || !g_c.active) continue;
        LONG seq = g_c.seq;
        if (qpc_ms_since(g_c.t0) > (double)g_case_timeout_ms && g_c.active && seq == g_c.seq) {
            char why[160];
            snprintf(why, sizeof why, "watchdog: case still running after %lu s (--case-timeout)", g_case_timeout_ms / 1000);
            fatal_finish(why, 0, 0, 0);
        }
    }
    return 0;
}

/* ---- per-case FP environment hygiene ----------------------------------------
 * Every case starts from the process-initial FP environment captured at the top of main():
 * x87 FNINIT + initial control word, MXCSR = initial value with the sticky status flags (bits 5:0)
 * cleared, and VZEROUPPER when AVX is usable. A case that changes MXCSR/CW cannot leak into the next. */
static unsigned g_mx0; static unsigned short g_cw0;
static void fp_env_baseline(void) { g_mx0 = t_get_mxcsr() & ~0x3Fu; g_cw0 = t_get_x87_cw(); }
static void fp_env_reset(void) {
    t_fninit(); t_set_x87_cw(g_cw0); t_set_mxcsr(g_mx0);
    if (cpu_has(F_AVX)) t_vzeroupper();
}

static T_NOINLINE void run_case_body(t_fn f) { f(); }

static void run_case(int i) {
    EnterCriticalSection(&g_lock);
    sb_reset(&g_c.fails_j); sb_reset(&g_c.notes_j); sb_reset(&g_c.info_j); sb_reset(&g_c.perf_j);
    g_c.name = g_t[i].n; g_c.checks = g_c.fails = 0; g_c.skip = g_c.xfail_now = g_c.expect_fail = g_c.crashed = 0;
    g_c.nperf = g_c.nfail_detail = g_c.finished = 0; g_c.exc_code = 0; g_c.exc_addr = 0; g_c.exc_other_thread = 0;
    g_c.reason[0] = 0; g_c.tid = GetCurrentThreadId();
    out_line("RUN   %s", g_c.name);
    LeaveCriticalSection(&g_lock);
#ifdef T_X86
    __asm__ volatile("movl %%fs:0, %0" : "=r"(g_seh_head));
#endif
    QueryPerformanceCounter(&g_c.t0);
    InterlockedIncrement(&g_c.seq);
    SetLastError(0);
    fp_env_reset();
    if (__builtin_setjmp(g_jb) == 0) {
        g_c.active = 1;
        run_case_body(g_t[i].f);
    }
    g_c.active = 0;
#ifdef T_X86
    __asm__ volatile("movl %0, %%fs:0" :: "r"(g_seh_head) : "memory");
#endif
    if (g_c.crashed) {
        if (g_c.exc_code == STATUS_STACK_OVERFLOW) {
            typedef int (CDECL *rs_t)(void);
            HMODULE m = GetModuleHandleA("msvcrt.dll");
            rs_t rs = m ? (rs_t)(void *)GetProcAddress(m, "_resetstkoflw") : NULL;
            if (rs) rs();
        }
        out_line("  CRASH %s", g_c.reason);
    }
    fp_env_reset();
    double ms = qpc_ms_since(g_c.t0);
    EnterCriticalSection(&g_lock);
    emit_case(ms);
    LeaveCriticalSection(&g_lock);
}

static int cmp_case(const void *a, const void *b) {
    const int *x = (const int *)a, *y = (const int *)b;
    if (g_t[*x].line != g_t[*y].line) return g_t[*x].line - g_t[*y].line;
    return g_t[*x].seq - g_t[*y].seq;
}

static int filter_match(const char *filter, const char *name) {
    if (!filter || !*filter) return 1;
    char buf[512]; snprintf(buf, sizeof buf, "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ","))
        if (*tok && strstr(name, tok)) return 1;
    return 0;
}

static void usage(void) {
    printf("usage: %s.exe [--json=<file>] [--filter=<substr>[,<substr>...]] [--list] [--verbose]\n"
           "              [--case-timeout=<sec>] [--include-manual] [--child=<name> [args...]]\n"
           "exit: 0 = no FAIL/CRASH, 1 = at least one FAIL/CRASH, 2 = harness error\n", g_exe);
}

static void init_exe_name(void) {
    const char *p = t_self_path(), *b = p;
    for (const char *q = p; *q; q++) if (*q == '\\' || *q == '/') b = q + 1;
    snprintf(g_exe, sizeof g_exe, "%s", b);
    char *dot = strrchr(g_exe, '.'); if (dot && !_stricmp(dot, ".exe")) *dot = 0;
}

int main(int argc, char **argv) {
    const char *json_path = NULL, *filter = NULL;
    int list = 0;
    fp_env_baseline();
    setvbuf(stdout, NULL, _IONBF, 0);
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    InitializeCriticalSection(&g_lock);
    init_exe_name();
    cpu_detect();

    if (argc > 1 && !strncmp(argv[1], "--child=", 8)) {
        const char *name = argv[1] + 8;
        g_child_mode = 1;
        SetUnhandledExceptionFilter(child_uef);
        for (int i = 0; i < g_nch; i++)
            if (!strcmp(g_ch[i].n, name)) {
                argv[1] = (char *)name;
                int rc = g_ch[i].f(argc - 1, argv + 1);
                fflush(stdout); fflush(stderr);
                return rc;
            }
        printf("harness error: unknown child '%s'\n", name);
        return (int)T_CHILD_UNKNOWN;
    }

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strncmp(a, "--json=", 7)) json_path = a + 7;
        else if (!strncmp(a, "--filter=", 9)) filter = a + 9;
        else if (!strcmp(a, "--list")) list = 1;
        else if (!strcmp(a, "--verbose") || !strcmp(a, "-v")) g_verbose = 1;
        else if (!strcmp(a, "--include-manual")) g_include_manual = 1;
        else if (!strncmp(a, "--case-timeout=", 15)) g_case_timeout_ms = (DWORD)strtoul(a + 15, NULL, 10) * 1000u;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h") || !strcmp(a, "/?")) { usage(); return 0; }
        else { printf("harness error: unknown argument '%s'\n", a); usage(); return 2; }
    }

    int order[MAX_CASES];
    for (int i = 0; i < g_nt; i++) order[i] = i;
    qsort(order, (size_t)g_nt, sizeof order[0], cmp_case);

    if (list) {
        for (int k = 0; k < g_nt; k++) {
            int i = order[k];
            if (filter_match(filter, g_t[i].n)) printf("%s%s\n", g_t[i].n, (g_t[i].flags & T_CASE_MANUAL) ? " [manual]" : "");
        }
        return 0;
    }

    if (json_path) {
        g_json = fopen(json_path, "wb");
        if (!g_json) { printf("harness error: cannot open --json file '%s'\n", json_path); return 2; }
    }
    QueryPerformanceCounter(&g_run_t0);
    t_env_print(stdout);
    if (g_json) { t_env_write_json(g_json, g_exe, T_VERSION); fflush(g_json); }

    SetUnhandledExceptionFilter(case_uef);
    HANDLE wd = CreateThread(NULL, 64 * 1024, watchdog, NULL, 0, NULL);
    if (wd) CloseHandle(wd);

    int ran = 0;
    for (int k = 0; k < g_nt; k++) {
        int i = order[k];
        if (!filter_match(filter, g_t[i].n)) continue;
        if ((g_t[i].flags & T_CASE_MANUAL) && !g_include_manual) continue;
        run_case(i);
        ran++;
    }
    EnterCriticalSection(&g_lock);
    if (!ran) {
        out_line("harness error: no case matched (filter=%s)", filter ? filter : "");
        if (g_json) fclose(g_json);
        return 2;
    }
    emit_summary();
    int ec = exit_code_now();
    if (g_json) { fclose(g_json); g_json = NULL; }
    LeaveCriticalSection(&g_lock);
    fflush(stdout);
    /* Do not run DLL detach / CRT teardown while stray test threads may still be running. */
    TerminateProcess(GetCurrentProcess(), (UINT)ec);
    return ec;
}
