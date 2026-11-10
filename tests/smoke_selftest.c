/* smoke_selftest - validates the harness itself: every result kind, crash recovery, FP-environment
 * hygiene between cases, worker-thread checks, child-process pass/fail/crash/timeout handling, JSON
 * validity, CLI exit codes, and the common helpers (PRNG, patterns, stats, cpuid gating, register-file
 * round trips, FXSAVE/XSAVE wrappers).
 *
 * demo_* cases are TEST_MANUAL: they deliberately FAIL/CRASH and only run inside the child started by
 * harness_result_kinds (--include-manual --filter=demo_), whose stdout, exit code and JSON are checked. */
#include "t.h"
#include <ctype.h>

/* ======================= demo cases (manual) ============================ */
static unsigned g_mxcsr_initial;
static volatile ULONG_PTR g_bad_addr = 0x10;   /* volatile: keeps the compiler from reasoning about the fault */

TEST_MANUAL(demo_pass) {
    g_mxcsr_initial = t_get_mxcsr();
    CHECK(1 + 1 == 2);
    CHECK_EQ(0x1234, 0x1234);
    CHECK_NE(1, 2);
    unsigned char a[20] = { 1, 2, 3 }, b[20] = { 1, 2, 3 };
    CHECK_MEM(a, b, sizeof a);
}
TEST_MANUAL(demo_fail) {
    unsigned char a[40], b[40];
    for (int i = 0; i < 40; i++) a[i] = b[i] = (unsigned char)i;
    a[37] ^= 0x80;
    CHECK(1 == 2);
    CHECK_EQ(1, 2);
    CHECK_MEM(a, b, sizeof a);
}
TEST_MANUAL(demo_skip) { SKIP("capability absent (demo)"); }
TEST_MANUAL(demo_xfail) { t_expect_fail("documented gap (demo)"); CHECK(0); }
TEST_MANUAL(demo_xfail_now) { XFAIL("known environment gap (demo)"); }
TEST_MANUAL(demo_xpass) { t_expect_fail("expected to fail but does not"); CHECK(1); }
static void spin_work(void *ctx) { volatile int x = 0; for (int i = 0; i < 1000; i++) x += i; (void)ctx; }
TEST_MANUAL(demo_perf) {
    t_stats s;
    CHECK(t_bench(spin_work, NULL, 2, 5, 1.0, &s) == 5);
    t_perf_stats("loop1000", &s, "s/call");
    t_perf("constant", 42.0, "units");
}
TEST_MANUAL(demo_require) { REQUIRE(0 == 1); CHECK(!"not reached"); }
TEST_MANUAL(demo_crash_av) { volatile int *p = (volatile int *)g_bad_addr; *p = 1; }
TEST_MANUAL(demo_after_crash) { CHECK(1); }
TEST_MANUAL(demo_crash_ud) { __asm__ volatile("ud2"); }
TEST_MANUAL(demo_crash_noncont) { RaiseException(0xC0000420u, EXCEPTION_NONCONTINUABLE, 0, NULL); }
static DWORD WINAPI worker_fail(void *p) { (void)p; CHECK(0 && "worker failure"); return 0; }
TEST_MANUAL(demo_worker_fail) {
    HANDLE h = CreateThread(NULL, 0, worker_fail, NULL, 0, NULL);
    REQUIRE(h != NULL);
    WaitForSingleObject(h, 5000); CloseHandle(h);
}
TEST_MANUAL(demo_mxcsr_dirty) { t_set_mxcsr(0x9FC0); t_set_x87_cw(0x0C7F); CHECK(t_get_mxcsr() == 0x9FC0); }
TEST_MANUAL(demo_mxcsr_restored) {
    CHECKF(t_get_mxcsr() == g_mxcsr_initial, "mxcsr 0x%x want 0x%x", t_get_mxcsr(), g_mxcsr_initial);
    CHECKF(t_get_x87_cw() != 0x0C7F, "x87 cw leaked from previous case");
}
TEST_MANUAL(demo_note_info) { t_note("a note with \"quotes\" and \\ backslash"); t_info("key", "value %d", 7); CHECK(1); }
TEST_MANUAL(demo_fail_then_skip) { CHECK(0); SKIP("too late: already failed"); }

static const struct { const char *name, *result; } g_expect[] = {
    { "demo_pass", "PASS" }, { "demo_fail", "FAIL" }, { "demo_skip", "SKIP" }, { "demo_xfail", "XFAIL" },
    { "demo_xfail_now", "XFAIL" }, { "demo_xpass", "PASS" }, { "demo_perf", "PERF" }, { "demo_require", "FAIL" },
    { "demo_crash_av", "CRASH" }, { "demo_after_crash", "PASS" }, { "demo_crash_ud", "CRASH" },
    { "demo_crash_noncont", "CRASH" }, { "demo_worker_fail", "FAIL" }, { "demo_mxcsr_dirty", "PASS" },
    { "demo_mxcsr_restored", "PASS" }, { "demo_note_info", "PASS" }, { "demo_fail_then_skip", "FAIL" },
};

/* ======================= minimal JSON validator =========================== */
static const char *js_ws(const char *p) { while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++; return p; }
static const char *js_value(const char *p, int depth);
static const char *js_string(const char *p) {
    if (*p != '"') return NULL;
    for (p++; *p && *p != '"'; p++) {
        if ((unsigned char)*p < 0x20) return NULL;
        if (*p == '\\') {
            p++;
            if (*p == 'u') { for (int i = 1; i <= 4; i++) if (!isxdigit((unsigned char)p[i])) return NULL; p += 4; }
            else if (!strchr("\"\\/bfnrt", *p) || !*p) return NULL;
        }
    }
    return *p == '"' ? p + 1 : NULL;
}
static const char *js_number(const char *p) {
    const char *s = p;
    if (*p == '-') p++;
    if (!isdigit((unsigned char)*p)) return NULL;
    if (*p == '0') p++; else while (isdigit((unsigned char)*p)) p++;
    if (*p == '.') { p++; if (!isdigit((unsigned char)*p)) return NULL; while (isdigit((unsigned char)*p)) p++; }
    if (*p == 'e' || *p == 'E') { p++; if (*p == '+' || *p == '-') p++; if (!isdigit((unsigned char)*p)) return NULL; while (isdigit((unsigned char)*p)) p++; }
    return p > s ? p : NULL;
}
static const char *js_value(const char *p, int depth) {
    if (depth > 32) return NULL;
    p = js_ws(p);
    if (*p == '{') {
        p = js_ws(p + 1);
        if (*p == '}') return p + 1;
        for (;;) {
            p = js_string(js_ws(p)); if (!p) return NULL;
            p = js_ws(p); if (*p != ':') return NULL;
            p = js_value(p + 1, depth + 1); if (!p) return NULL;
            p = js_ws(p);
            if (*p == ',') { p++; continue; }
            return *p == '}' ? p + 1 : NULL;
        }
    }
    if (*p == '[') {
        p = js_ws(p + 1);
        if (*p == ']') return p + 1;
        for (;;) {
            p = js_value(p, depth + 1); if (!p) return NULL;
            p = js_ws(p);
            if (*p == ',') { p++; continue; }
            return *p == ']' ? p + 1 : NULL;
        }
    }
    if (*p == '"') return js_string(p);
    if (!strncmp(p, "true", 4)) return p + 4;
    if (!strncmp(p, "false", 5)) return p + 5;
    if (!strncmp(p, "null", 4)) return p + 4;
    return js_number(p);
}
static int js_valid_line(const char *line) { const char *e = js_value(line, 0); return e && *js_ws(e) == 0; }
/* value of a top-level "key":"string" (first occurrence); "" if absent */
static void js_get_str(const char *line, const char *key, char *out, size_t cap) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\":\"", key);
    const char *p = strstr(line, pat); out[0] = 0;
    if (!p) return;
    p += strlen(pat);
    size_t n = 0; while (p[n] && p[n] != '"' && n + 1 < cap) { out[n] = p[n]; n++; }
    out[n] = 0;
}
static long js_get_int(const char *line, const char *key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(line, pat);
    return p ? strtol(p + strlen(pat), NULL, 10) : -1;
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1); if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)n, f); fclose(f); b[got] = 0; if (len) *len = got;
    return b;
}

/* ======================= harness behaviour ================================== */
TEST(harness_result_kinds) {
    char json[MAX_PATH], args[MAX_PATH + 128];
    t_tmpfile(json, sizeof json, "demo.json");
    snprintf(args, sizeof args, "--include-manual --filter=demo_ --case-timeout=30 \"--json=%s\"", json);
    static t_child_result r;
    REQUIREF(t_run_self(args, 60000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(!r.timed_out);
    CHECK_EQ(r.exit_code, 1);                          /* FAIL and CRASH present */
    char sum[512] = "";
    CHECK(t_child_find(&r, "SUMMARY ", sum, sizeof sum));
    t_log("child summary: %s", sum);
    CHECK(strstr(sum, "exit=1") != NULL);
    CHECK(strstr(r.out, "  CRASH unhandled exception 0xc0000005") != NULL);
    CHECK(strstr(r.out, "       want @0020: ") != NULL);   /* CHECK_MEM hexdump window of demo_fail */

    size_t len = 0;
    char *buf = read_file(json, &len);
    REQUIREF(buf != NULL, "cannot read %s", json);
    DeleteFileA(json);
    int lines = 0, ok_lines = 0, nenv = 0, nsum = 0, ncase = 0, first_env = 0, last_sum = 0;
    char seen[T_COUNTOF(g_expect)][16]; memset(seen, 0, sizeof seen);
    for (char *ln = buf, *nx; ln && *ln; ln = nx) {
        nx = strchr(ln, '\n'); if (nx) *nx++ = 0;
        lines++;
        int valid = js_valid_line(ln);
        ok_lines += valid;
        CHECKF(valid, "invalid JSON line %d: %.200s", lines, ln);
        char type[32], name[64], res[16];
        js_get_str(ln, "type", type, sizeof type);
        last_sum = 0;
        if (!strcmp(type, "env")) { nenv++; if (lines == 1) first_env = 1; }
        else if (!strcmp(type, "summary")) {
            nsum++; last_sum = 1;
            CHECK_EQ(js_get_int(ln, "exit_code"), 1);
            CHECK_EQ(js_get_int(ln, "crash"), 3);
            CHECK_EQ(js_get_int(ln, "total"), T_COUNTOF(g_expect));
        } else if (!strcmp(type, "case")) {
            ncase++;
            js_get_str(ln, "case", name, sizeof name);
            js_get_str(ln, "result", res, sizeof res);
            for (size_t i = 0; i < T_COUNTOF(g_expect); i++)
                if (!strcmp(name, g_expect[i].name)) snprintf(seen[i], sizeof seen[i], "%s", res);
            if (!strcmp(name, "demo_crash_av")) CHECK(strstr(ln, "\"code\":\"0xc0000005\"") != NULL);
            if (!strcmp(name, "demo_crash_ud")) CHECK(strstr(ln, "\"code\":\"0xc000001d\"") != NULL);
            if (!strcmp(name, "demo_crash_noncont")) CHECK(strstr(ln, "\"code\":\"0xc0000420\"") != NULL);
            if (!strcmp(name, "demo_perf")) CHECK(strstr(ln, "\"metric\":\"loop1000\"") && strstr(ln, "\"stddev\":"));
            if (!strcmp(name, "demo_note_info")) CHECK(strstr(ln, "\"key\":\"value 7\"") != NULL);
            if (!strcmp(name, "demo_require")) CHECK_EQ(js_get_int(ln, "checks"), 1);
            if (!strcmp(name, "demo_xpass")) CHECK(strstr(ln, "XPASS") != NULL);
        }
    }
    CHECK_EQ(ok_lines, lines);
    CHECK(first_env);
    CHECK_EQ(nenv, 1);
    CHECK_EQ(nsum, 1);
    CHECK(last_sum);
    CHECK_EQ(ncase, T_COUNTOF(g_expect));
    for (size_t i = 0; i < T_COUNTOF(g_expect); i++)
        CHECKF(!strcmp(seen[i], g_expect[i].result), "%s: got '%s' want '%s'", g_expect[i].name, seen[i], g_expect[i].result);
    free(buf);
}

TEST(harness_list_and_cli) {
    static t_child_result r;
    REQUIREF(t_run_self("--list", 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK_EQ(r.exit_code, 0);
    CHECK(strstr(r.out, "harness_result_kinds") != NULL);
    CHECK(strstr(r.out, "demo_crash_av [manual]") != NULL);
    REQUIREF(t_run_self("--no-such-option", 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK_EQ(r.exit_code, 2);
    REQUIREF(t_run_self("--filter=no_case_matches_this", 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK_EQ(r.exit_code, 2);
    REQUIREF(t_run_self("--filter=demo_pass,demo_skip --include-manual", 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK_EQ(r.exit_code, 0);                          /* SKIP does not fail the run */
    CHECK(strstr(r.out, "total=2 pass=1 fail=0 crash=0 skip=1") != NULL);
}

/* ======================= child processes ===================================== */
CHILD(exit7) { (void)argc; (void)argv; return 7; }
CHILD(echo) { for (int i = 0; i < argc; i++) printf("ARG%d=%s\n", i, argv[i]); fprintf(stderr, "ERRLINE\n"); return 0; }
CHILD(av) { (void)argc; (void)argv; volatile int *p = (volatile int *)g_bad_addr; return *p; }
CHILD(ud) { (void)argc; (void)argv; __asm__ volatile("ud2"); return 0; }
CHILD(hang) { (void)argc; (void)argv; printf("HANGING\n"); fflush(stdout); Sleep(INFINITE); return 0; }

TEST(child_exit_code) {
    static t_child_result r;
    REQUIREF(t_child_run("exit7", NULL, 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(r.started);
    CHECK(!r.timed_out);
    CHECK_EQ(r.exit_code, 7);
    CHECK_EQ(r.exc_code, 0);
}
TEST(child_output_capture) {
    static t_child_result r;
    char v[64];
    REQUIREF(t_child_run("echo", "alpha \"b c\"", 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK_EQ(r.exit_code, 0);
    CHECK(t_child_find(&r, "ARG0=", v, sizeof v) && !strcmp(v, "echo"));
    CHECK(t_child_find(&r, "ARG1=", v, sizeof v) && !strcmp(v, "alpha"));
    CHECK(t_child_find(&r, "ARG2=", v, sizeof v) && !strcmp(v, "b c"));
    CHECK(t_child_find(&r, "ERRLINE", v, sizeof v));
}
TEST(child_access_violation) {
    static t_child_result r;
    REQUIREF(t_child_run("av", NULL, 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(!r.timed_out);
    CHECK_EQ(r.exc_code, 0xC0000005u);
    CHECK_EQ(r.exit_code, 0xC0000005u);
    CHECK(strstr(r.out, "T_CHILD_EXCEPTION code=0xc0000005") != NULL);
}
TEST(child_illegal_instruction) {
    static t_child_result r;
    REQUIREF(t_child_run("ud", NULL, 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK_EQ(r.exc_code, 0xC000001Du);
    CHECK(r.exc_addr != 0);
}
TEST(child_timeout_kill) {
    static t_child_result r;
    REQUIREF(t_child_run("hang", NULL, 1500, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(r.timed_out);
    CHECK(r.ms >= 1400 && r.ms < 15000);
    CHECK(strstr(r.out, "HANGING") != NULL);
    CHECK_EQ(r.exc_code, 0);
}
TEST(child_unknown_name) {
    static t_child_result r;
    REQUIREF(t_child_run("no_such_child", NULL, 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK_EQ(r.exit_code, T_CHILD_UNKNOWN);
}

/* ======================= common helpers ======================================== */
TEST(prng_and_pattern_reference) {
    t_rng r; t_rng_seed(&r, 12345);
    CHECK_EQ(t_rng_next(&r), 0x90b6115441078f2cull);
    CHECK_EQ(t_rng_next(&r), 0x68b3f159529919abull);
    CHECK_EQ(t_rng_next(&r), 0xbdc514bab540895bull);
    CHECK_EQ(t_pattern_word(0, 0, 0), 0x8040143d34585f45ull);
    CHECK_EQ(t_pattern_word(7, 31, 7), 0x9f47859bc5ebf917ull);
    uint32_t tag, reg, word;
    CHECK(t_pattern_explain(t_pattern_word(123, 9, 5), 1024, &tag, &reg, &word) && tag == 123 && reg == 9 && word == 5);
    for (uint64_t n = 1; n < 50; n++) CHECK(t_rng_range(&r, n) < n);
    char h[128];
    unsigned char b[6] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab };
    t_hex(h, sizeof h, b, 6);             CHECK(!strcmp(h, "01234567 89ab"));
    uint64_t q[2] = { 0x1122334455667788ull, 0x99aabbccddeeff00ull };
    t_hex_lanes(h, sizeof h, q, 16, 8);   CHECK(!strcmp(h, "99aabbccddeeff00_1122334455667788"));
}
TEST(stats_and_timing) {
    double v[] = { 5, 1, 3, 2, 4 };
    t_stats s; t_stats_compute(v, 5, &s);
    CHECK(s.n == 5 && s.min == 1 && s.max == 5 && s.median == 3 && s.mean == 3);
    CHECK(s.stddev > 1.58 && s.stddev < 1.59);
    double w[] = { 4, 1, 3, 2 };
    t_stats_compute(w, 4, &s); CHECK(s.median == 2.5);
    uint64_t a = t_qpc(); Sleep(20); uint64_t b = t_qpc();
    CHECK(t_qpc_ms(a, b) >= 10.0);
    CHECK(t_qpc_freq() > 0);
    uintptr_t prev = t_pin_thread(0);
    if (prev) { CHECK(GetCurrentProcessorNumber() == 0); t_unpin_thread(prev); }
    else t_note("pinning to cpu 0 not possible here");
    CHECK(t_cpu_count() >= 1);
}
TEST(cpuid_gating_consistency) {
    const t_cpu_info *c = cpu();
    REQUIRE(c->cpuid_ok);
    CHECK(c->max_basic >= 1);
    CHECK(c->vendor[0] != 0);
    CHECK(cpu_hw(F_SSE2));                       /* Windows 8+ requires SSE2 */
    for (int f = 0; f < F_COUNT; f++) CHECKF(!cpu_has((t_feature)f) || cpu_hw((t_feature)f), "%s usable but not advertised", cpu_feature_name((t_feature)f));
    if (cpu_has(F_AVX)) CHECK(c->osxsave && (c->xcr0 & 6) == 6);
    if (cpu_has(F_AVX512F)) CHECK((c->xcr0 & 0xE6) == 0xE6);
    CHECK(!cpu_has(F_XSAVES));                  /* CPL0 only */
    CHECK_EQ(cpu_feature_lookup("avx2"), F_AVX2);
    if (c->osxsave) {
        CHECK(c->xcr0 & 1);                       /* XCR0[0] must be 1 */
        CHECK((c->xcr0 & ~c->xcr0_supported) == 0);
        /* SDM: CPUID.(0Dh,0):EBX = size for the features currently enabled in XCR0 */
        CHECK_EQ(xsave_std_size(c->xcr0), c->xsave_size_enabled);
        CHECK(xsave_compacted_size(c->xcr0) <= xsave_std_size(c->xcr0) || (c->xcr0 >> 2) == 0);
        if (c->xcr0 & T_XS_AVX) CHECK_EQ(xsave_compacted_offset(c->xcr0, 2), 576);
    } else t_note("OSXSAVE clear: XCR0 not read");
    static t_cpuid_entry e[512];
    int n = cpuid_enumerate(e, 512);
    CHECK(n > 3);
    t_info("cpuid_entries", "%d", n);
}
static T_NOINLINE void empty_fn(void *p) { __asm__ volatile("" ::: "memory"); (void)p; }
static void vregs_roundtrip(int kind) {
    static t_vregs in, out;
    t_vregs_pattern(&in, kind, 0x51);
    CHECK_EQ(t_vregs_call(kind, &in, &out, NULL, NULL), 0);
    CHECK_EQ(t_vregs_check(&out, &in, kind, 0, -1, "load/store"), 0);
    static volatile long stop = 0;
    CHECK(t_vregs_spin(kind, &in, &out, &stop, 1000) == 0);
    CHECK_EQ(t_vregs_check(&out, &in, kind, 0, -1, "after spin"), 0);
    stop = 1;
    CHECK(t_vregs_spin(kind, &in, &out, &stop, 1000) == 1000);  /* stop flag honoured at once */
    stop = 0;
    CHECK_EQ(t_vregs_call(kind, &in, &out, empty_fn, NULL), 0);
#ifdef T_X64
    t_vregs_check(&out, &in, T_VK_XMM, 6, 15, "after call (ABI nonvolatile)");  /* low 128 bits only */
    if (kind != T_VK_XMM) for (int i = 6; i < 16; i++) CHECK_REG("xmm low half", out.v[i], in.v[i], 16);
#endif
}
TEST(vregs_xmm_roundtrip) { vregs_roundtrip(T_VK_XMM); }
TEST(vregs_ymm_roundtrip) { if (!cpu_has(F_AVX)) SKIP("AVX not usable"); vregs_roundtrip(T_VK_YMM); }
TEST(vregs_zmm_roundtrip) { if (!cpu_has(F_AVX512F)) SKIP("AVX-512F not usable"); vregs_roundtrip(T_VK_ZMM); }
TEST(x87_mmx_roundtrip) {
    t_f80 in[8], out[8];
    for (int i = 0; i < 8; i++) {  /* normal 80-bit values: mantissa with explicit integer bit */
        uint64_t m = t_pattern_word(1, (uint32_t)i, 0) | 0x8000000000000000ull;
        uint16_t se = (uint16_t)(0x3FFF + i);
        memcpy(in[i].b, &m, 8); memcpy(in[i].b + 8, &se, 2);
    }
    unsigned short cw = t_get_x87_cw();
    CHECK(t_x87_spin(in, out, NULL, 100) == 0);
    CHECK_MEM(out, in, sizeof in);
    CHECK_EQ(t_get_x87_cw(), cw);
    CHECK_EQ(t_get_x87_tw() & 0xFFFF, 0xFFFF);       /* stack empty again */
    uint64_t mi[8], mo[8];
    for (int i = 0; i < 8; i++) mi[i] = t_pattern_word(2, (uint32_t)i, 0);
    if (!cpu_has(F_MMX)) SKIP("no MMX");
    CHECK(t_mmx_spin(mi, mo, NULL, 100) == 0);
    CHECK_MEM(mo, mi, sizeof mi);
    CHECK_EQ(t_get_x87_tw() & 0xFFFF, 0xFFFF);       /* EMMS executed */
}
TEST(fxsave_xsave_wrappers) {
    static T_ALIGN(64) unsigned char area[16384];
    unsigned mx = t_get_mxcsr();
    memset(area, 0xCC, 512);
    t_fxsave(area);
    unsigned got; memcpy(&got, area + 24, 4);
    CHECK_EQ(got, mx);                                  /* FXSAVE: MXCSR at byte 24 */
    unsigned short fcw; memcpy(&fcw, area, 2);
    CHECK_EQ(fcw, t_get_x87_cw());
    if (!cpu_has(F_XSAVE)) SKIP("XSAVE not usable");
    const t_cpu_info *c = cpu();
    REQUIRE(c->xsave_size_enabled <= sizeof area);
    memset(area, 0, sizeof area);
    t_xsave(area, c->xcr0);
    uint64_t xstate_bv; memcpy(&xstate_bv, area + 512, 8);
    CHECKF((xstate_bv & ~c->xcr0) == 0, "XSTATE_BV 0x%llx outside XCR0 0x%llx", (unsigned long long)xstate_bv, (unsigned long long)c->xcr0);
    memcpy(&got, area + 24, 4);
    CHECK_EQ(got, mx);
    t_xrstor(area, c->xcr0);
    CHECK_EQ(t_get_mxcsr(), mx);
    if (cpu_has(F_XSAVEC)) {
        memset(area, 0, sizeof area);
        t_xsavec(area, c->xcr0);
        uint64_t xcomp_bv; memcpy(&xcomp_bv, area + 520, 8);
        CHECK(xcomp_bv >> 63);                          /* compacted format bit */
        t_xrstor(area, c->xcr0);
    }
}
TEST(env_recorder) {
    const t_env *e = t_env_get();
    CHECK(e->os_major >= 6);
    CHECK(!strcmp(e->process_arch, T_ARCH));
    CHECK(e->ncpu >= 1);
    if (e->api_xstate_present) {
        if (!e->api_xstate_features && t_is_wine())   /* known Wine gap (seen with Wine 9 on a CPU model without XSAVE): the API returns 0, Windows returns 3 */
            t_expect_fail("Wine %s: GetEnabledXStateFeatures returns 0 without XSAVE", t_wine_version());
        CHECK(e->api_xstate_features & 1);              /* x87 legacy state always reported */
        t_info("GetEnabledXStateFeatures", "0x%llx", e->api_xstate_features);
    }
    if (t_is_wine()) t_info("wine", "%s", t_wine_version());
}
CHILD(guard_overrun) {
    (void)argc; (void)argv;
    void *base; volatile unsigned char *p = (volatile unsigned char *)t_guarded_alloc(100, &base);
    if (!p) return 3;
    for (int i = 0; i < 100; i++) p[i] = (unsigned char)i;   /* all usable */
    printf("WRITES_OK\n"); fflush(stdout);
    return p[100];                                            /* first byte of the guard page */
}
TEST(util_helpers) {
    CHECK(T_PROC("kernel32.dll", GetTickCount) != NULL);
    CHECK(t_proc("kernel32.dll", "NoSuchExport_xyz") == NULL);
    CHECK(t_proc("no_such_dll_xyz.dll", "x") == NULL);
    static t_child_result r;
    REQUIREF(t_child_run("guard_overrun", NULL, 20000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(strstr(r.out, "WRITES_OK") != NULL);
    CHECK_EQ(r.exc_code, 0xC0000005u);
    char a[MAX_PATH], b[MAX_PATH];
    t_tmpfile(a, sizeof a, "x.json"); t_tmpfile(b, sizeof b, "x.json");
    CHECK(strcmp(a, b) != 0);
}

/* watchdog and worker-thread crash: each ends the whole run, so they run in their own child runs */
TEST_MANUAL(wd_hang) { Sleep(20000); }
static DWORD WINAPI worker_crash(void *p) { (void)p; volatile int *q = (volatile int *)g_bad_addr; return (DWORD)*q; }
TEST_MANUAL(wt_crash) {
    HANDLE h = CreateThread(NULL, 0, worker_crash, NULL, 0, NULL);
    REQUIRE(h != NULL);
    WaitForSingleObject(h, 10000); CloseHandle(h);
}
TEST(harness_fatal_paths) {
    static t_child_result r;
    char json[MAX_PATH], args[MAX_PATH + 128];
    t_tmpfile(json, sizeof json, "wd.json");
    snprintf(args, sizeof args, "--include-manual --filter=wd_hang --case-timeout=1 \"--json=%s\"", json);
    REQUIREF(t_run_self(args, 15000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(!r.timed_out);
    CHECK_EQ(r.exit_code, 1);
    CHECK(r.ms < 10000);
    CHECK(strstr(r.out, "CRASH wd_hang") && strstr(r.out, "watchdog"));
    CHECK(strstr(r.out, "SUMMARY ") && strstr(r.out, "crash=1"));
    char *buf = read_file(json, NULL);
    CHECK(buf && strstr(buf, "\"result\":\"CRASH\"") && strstr(buf, "\"type\":\"summary\""));
    free(buf); DeleteFileA(json);

    REQUIREF(t_run_self("--include-manual --filter=wt_crash", 15000, &r), "spawn failed: error %lu", r.spawn_error);
    CHECK(!r.timed_out);
    CHECK_EQ(r.exit_code, 1);
    CHECK(strstr(r.out, "CRASH wt_crash") && strstr(r.out, "not the case thread"));
    CHECK(strstr(r.out, "SUMMARY ") != NULL);
}
