/* t.h - test harness for the CPU SIMD / extended-processor-state suite.
 *
 * One exe = tests/<name>.c + every common/<x>.c.  Include only "t.h"; it pulls in every common module.
 * Full API reference: common/API.md.  Keep this header small and stable: ~22 exes depend on it.
 *
 * Result of a case (precedence high -> low):
 *   CRASH  unrecoverable/unexpected exception inside the case (recovered by the harness, run continues)
 *   FAIL   at least one CHECK failed (or REQUIRE aborted the case)
 *   XFAIL  a failure happened after t_expect_fail(), or XFAIL() was called
 *   SKIP   SKIP() was called (capability absent) and no check failed
 *   PERF   t_perf*() recorded a measurement and no check failed
 *   PASS   otherwise
 * TIMEOUT is assigned only by the external runner (scripts/run_all.*).
 * Exit code: 0 = no FAIL/CRASH, 1 = at least one FAIL/CRASH, 2 = harness error.
 */
#ifndef T_H
#define T_H
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef WINVER
#define WINVER 0x0A00
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#if defined(_MSC_VER) && !defined(__clang__)
#error "use gcc (mingw-w64) or clang/clang-cl: the suite relies on GNU inline asm and target attributes"
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define T_X64 1
#define T_ARCH "x64"
#else
#define T_X86 1
#define T_ARCH "x86"
#endif

#define T_ALIGN(n)    __attribute__((aligned(n)))
#define T_NOINLINE    __attribute__((noinline))
#define T_NORETURN    __attribute__((noreturn))
#define T_PRINTF(a,b) __attribute__((format(gnu_printf, a, b)))
#define T_TARGET(s)   __attribute__((target(s)))   /* per-function ISA: T_TARGET("avx2,fma") */
#define T_COUNTOF(a)  (sizeof(a) / sizeof((a)[0]))

/* ---- registration ------------------------------------------------------ */
typedef void (*t_fn)(void);
typedef int  (*t_child_fn)(int argc, char **argv);   /* argv[0] = child name, argv[1..] = extra args */
enum { T_CASE_DEFAULT = 0, T_CASE_MANUAL = 1 };      /* MANUAL: only run with --include-manual */
void t_register(const char *name, t_fn fn, int flags, int line);
void t_register_child(const char *name, t_child_fn fn);

#define T_CTOR_(f) static void __attribute__((constructor)) f(void)
/* Cases run in source order. Name = C identifier, shown and filtered as written. */
#define TEST_EX(n, flags) \
    static void test_##n(void); \
    T_CTOR_(t_reg_##n) { t_register(#n, test_##n, flags, __LINE__); } \
    static void test_##n(void)
#define TEST(n)        TEST_EX(n, T_CASE_DEFAULT)
#define TEST_MANUAL(n) TEST_EX(n, T_CASE_MANUAL)
/* Child entry points: run as  <exe> --child=<n> [extra args]. Return value = process exit code.
 * Unhandled exceptions in a child terminate it with the exception code as exit code (and print
 * "T_CHILD_EXCEPTION code=0x... addr=0x..." first). Children never write JSON. */
#define CHILD(n) \
    static int child_##n(int argc, char **argv); \
    T_CTOR_(t_regc_##n) { t_register_child(#n, child_##n); } \
    static int child_##n(int argc, char **argv)

/* ---- checks (thread-safe; CHECK may be used on worker threads) --------- */
void t_check(int ok, const char *file, int line, const char *expr, const char *fmt, ...) T_PRINTF(5, 6);
T_NORETURN void t_abort_case(void);   /* end the current case now (case thread only) */

#define CHECK(c)          t_check(!!(c), __FILE__, __LINE__, #c, NULL)
#define CHECKF(c, ...)    t_check(!!(c), __FILE__, __LINE__, #c, __VA_ARGS__)
/* Integer equality, printed in hex and decimal. */
#define CHECK_EQ(a, b) do { unsigned long long a_ = (unsigned long long)(a), b_ = (unsigned long long)(b); \
    t_check(a_ == b_, __FILE__, __LINE__, #a " == " #b, "got 0x%llx (%llu) want 0x%llx (%llu)", a_, a_, b_, b_); } while (0)
#define CHECK_NE(a, b) do { unsigned long long a_ = (unsigned long long)(a), b_ = (unsigned long long)(b); \
    t_check(a_ != b_, __FILE__, __LINE__, #a " != " #b, "both 0x%llx", a_); } while (0)
/* Byte-exact memory equality; on mismatch prints labelled hexdumps of both sides (first differing
 * 64-byte window) and the offset of the first difference. */
#define CHECK_MEM(got, want, n) t_check_mem((got), (want), (size_t)(n), __FILE__, __LINE__, #got " == " #want)
int  t_check_mem(const void *got, const void *want, size_t n, const char *file, int line, const char *expr);
/* REQUIRE: like CHECK but ends the case on failure (case thread only; on other threads it records only). */
#define REQUIRE(c)        do { if (!(c)) { t_check(0, __FILE__, __LINE__, #c, NULL); t_abort_case(); } } while (0)
#define REQUIREF(c, ...)  do { if (!(c)) { t_check(0, __FILE__, __LINE__, #c, __VA_ARGS__); t_abort_case(); } } while (0)

/* ---- non-check outcomes -------------------------------------------------- */
/* SKIP: capability absent. Ends the case (case thread). Never use SKIP for "advertised but broken". */
T_NORETURN void t_skip(const char *fmt, ...) T_PRINTF(1, 2);
#define SKIP(...)  t_skip(__VA_ARGS__)
/* XFAIL now: documented expected failure (e.g. known environment gap), ends the case. */
T_NORETURN void t_xfail(const char *fmt, ...) T_PRINTF(1, 2);
#define XFAIL(...) t_xfail(__VA_ARGS__)
/* Mark the case as expected to fail: any failed CHECK (before or after) turns the case into XFAIL.
 * If nothing fails the case is PASS and an "XPASS" note is recorded. */
void t_expect_fail(const char *fmt, ...) T_PRINTF(1, 2);
/* Observations: never affect the result. NOTE = free text, INFO = key/value (JSON "info" object). */
void t_note(const char *fmt, ...) T_PRINTF(1, 2);
void t_info(const char *key, const char *fmt, ...) T_PRINTF(2, 3);
/* Measurements (PERF): value in unit; optional distribution from timing.h stats. */
struct t_stats;
void t_perf(const char *metric, double value, const char *unit);
void t_perf_stats(const char *metric, const struct t_stats *s, const char *unit); /* value = median */
/* Diagnostics printed only with --verbose (not in JSON). */
void t_log(const char *fmt, ...) T_PRINTF(1, 2);

/* ---- environment / state queries -------------------------------------- */
int         t_verbose(void);
int         t_is_wine(void);          /* ntdll exports wine_get_version: use ONLY to pick XFAIL/SKIP reasons */
const char *t_wine_version(void);     /* "" when not wine */
const char *t_sde_model(void);        /* XCR_SDE env (tools/sde_run.sh: Intel SDE model) or NULL: use ONLY to pick XFAIL/NOTE reasons */
const char *t_case_name(void);        /* current case or "" */
const char *t_exe_name(void);         /* e.g. "xmm_registers" */
const char *t_self_path(void);        /* full path of this exe */
int         t_on_case_thread(void);   /* 1 on the thread running the case */

/* ---- child processes ---------------------------------------------------- */
#define T_CHILD_UNKNOWN 0x7E57C0DEu   /* exit code when --child=<name> is not registered */
typedef struct t_child_result {
    int      started;       /* CreateProcess succeeded */
    DWORD    spawn_error;   /* GetLastError of the last failed CreatePipe/CreateProcess, else 0 */
    int      timed_out;     /* exceeded timeout and was killed */
    DWORD    exit_code;     /* raw process exit code (exception code when it crashed) */
    DWORD    exc_code;      /* exception code if the child died of an exception, else 0 */
    unsigned long long exc_addr; /* faulting address reported by the child, 0 if unknown */
    double   ms;            /* wall time */
    char     out[16384];    /* captured stdout+stderr (truncated, NUL-terminated) */
    size_t   out_len;
} t_child_result;
/* Run  <self> --child=<name> [args]  and wait up to timeout_ms (killed after). Returns 1 if started.
 * The child runs in a kill-on-close job so grandchildren die with it. Output is captured. */
int t_child_run(const char *name, const char *args, DWORD timeout_ms, t_child_result *r);
/* Run <self> with an arbitrary command-line tail (e.g. "--filter=x --json=f"). Same semantics. */
int t_run_self(const char *args, DWORD timeout_ms, t_child_result *r);
/* Look for a line starting with prefix in r->out; copies the rest of the line to val. */
int t_child_find(const t_child_result *r, const char *prefix, char *val, size_t cap);
/* 1 when running under Intel SDE (t_sde_model()) and the child's output carries an SDE error report
 * (chip-check "instruction not valid for specified chip/xcr0", Pin abort): SDE terminated the child
 * instead of raising the architectural exception. Use ONLY to pick an XFAIL reason. */
int t_sde_killed_child(const t_child_result *r);

/* ---- small utilities (util.c) ------------------------------------------------ */
/* Resolve an export at run time (LoadLibraryA on first use). Use for every API that may be missing on
 * some Windows versions (XSTATE APIs, Win11-only functions) so the exe still loads everywhere.
 *   T_PROC("kernel32.dll", InitializeContext2) -> typed pointer (from the SDK declaration) or NULL */
void *t_proc(const char *dll, const char *name);
#define T_PROC(dll, name) ((__typeof__(&name))t_proc(dll, #name))
/* n usable bytes ending exactly at a PAGE_NOACCESS guard page (for end-of-page / fault-suppression
 * tests). *base receives the allocation to pass to t_guarded_free. NULL on failure. */
void *t_guarded_alloc(size_t n, void **base);
void  t_guarded_free(void *base);
/* Unique scratch file path in %TEMP% (not created). */
void  t_tmpfile(char *out, size_t cap, const char *leaf);

/* Everything else (in one include) */
#include "cpuid.h"
#include "feature_detect.h"
#include "timing.h"
#include "pattern.h"
#include "asm.h"
#include "env.h"
#endif
