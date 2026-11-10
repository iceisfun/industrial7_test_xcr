/* jpeg_bench - JPEG decode throughput matrix: image size x kernel tier x decoding threads
 * (docs/GUIDANCE.md section 5 "JPEG decode MPix/s", section 6 "repeated multithreaded decode").
 *
 * Decoder: the self-contained baseline/progressive decoder of tests/simd_jpeg_dec.h with its scalar,
 * SSE2 and AVX2 kernel sets (islow integer IDCT, fancy upsampling, YCbCr->RGB). Inputs: the synthetic
 * images of tests/simd_jpeg_corpus.h, re-encoded at run time by tests/simd_jpeg_enc.h (deterministic),
 * whose decoded pixels have golden hashes from libjpeg-turbo djpeg (`-dct int`, fancy upsampling):
 *   synth_vga_gray 640x480 grayscale, synth_mt_420 512x384 4:2:0 with restarts,
 *   synth_1k_444_rst 1024x768 4:4:4 with restarts, synth_4k_420 3840x2160 4:2:0,
 *   synth_8k_420 7680x4320 4:2:0 (manual case, --include-manual: 100 MB per decoded image).
 * Threads: 1, 2, 4, 8, 16, limited by the logical processors in the affinity mask and by a 320 MB
 * budget for decoder buffers (about 3x the output image per thread). Every thread decodes the same image
 * `reps` times per sample; the sample time runs from releasing the threads to the last one finishing.
 * Row metric: jpeg_decode_<name>_<tier>_<N>t in aggregate MPix/s (pixels decoded by all threads / time).
 *
 * Pass/fail: every decode of every sample is hashed and compared with djpeg's hash; a wrong pixel is a
 * FAIL (and that tier's numbers are not reported). The encoder output must match the host's bytes
 * (length + hash recorded in the corpus) before any decode is trusted. Slowness never fails.
 * Feature absent (SSE2/AVX2 kernel set not usable) -> that tier is skipped with a NOTE; a case whose
 * every tier is unusable cannot happen (scalar always runs). Threads that cannot be created -> NOTE and
 * the row is not reported (resource, not capability). */
#include "simd_bench_util.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#include "simd_jpeg_dec.h"
#include "simd_jpeg_enc.h"
#include "simd_jpeg_corpus.h"
#pragma GCC diagnostic pop

#define JB_BUDGET ((size_t)320 << 20)
#define JB_MAXT 16

typedef struct jb_tier { const char *name; const jd_kernels *k; int usable; } jb_tier;
static int jb_tiers(jb_tier *t) {
    t[0].name = "scalar"; t[0].k = &jd_kernels_scalar; t[0].usable = 1;
    t[1].name = "sse2";   t[1].k = &jd_kernels_sse2;   t[1].usable = cpu_has(F_SSE2);
    t[2].name = "avx2";   t[2].k = &jd_kernels_avx2;   t[2].usable = cpu_has(F_AVX2);
    return 3;
}

static const sj_synth *jb_find(const char *name) {
    for (size_t i = 0; i < T_COUNTOF(sj_synth_list); i++) if (!strcmp(sj_synth_list[i].name, name)) return &sj_synth_list[i];
    return NULL;
}

/* encoded images, cached for the run */
typedef struct jb_img { const sj_synth *s; uint8_t *jpg; size_t len; int state; } jb_img;
static jb_img g_jb[8];
static jb_img *jb_get(const char *name) {
    const sj_synth *s = jb_find(name);
    REQUIREF(s != NULL, "synthetic image %s missing from the corpus table", name);
    jb_img *g = &g_jb[s - sj_synth_list];
    if (g->state) return g;
    uint8_t *rgb = (uint8_t *)malloc((size_t)s->w * s->h * 3);
    if (!rgb) SKIP("%s: no memory for the %dx%d source image", name, s->w, s->h);
    double t0 = t_now_ms();
    je_synth_rgb(rgb, s->w, s->h, s->seed);
    g->jpg = je_encode(rgb, s->w, s->h, s->sub, s->q, s->ri, &g->len);
    free(rgb);
    if (!g->jpg) SKIP("%s: encoder out of memory", name);
    t_info("encode_ms", "%.0f", t_now_ms() - t0);
    REQUIREF(g->len == s->jpeg_len && jd_fnv64(g->jpg, g->len) == s->jpeg_hash,
             "%s: encoder produced %u bytes (hash %016llx), host produced %u (%016llx)", name, (unsigned)g->len,
             (unsigned long long)jd_fnv64(g->jpg, g->len), s->jpeg_len, (unsigned long long)s->jpeg_hash);
    g->s = s; g->state = 1;
    return g;
}

/* one decode, verified */
static int jb_decode_ok(const jb_img *g, const jd_kernels *k) {
    jd_options o;
    memset(&o, 0, sizeof o);
    o.k = k; o.dct = JD_DCT_ISLOW; o.upsample = JD_UP_FANCY; o.out_fmt = JD_OUT_NATIVE;
    o.max_pixels = 0xFFFFFFFFu;               /* the 8K image exceeds the default 16M-pixel limit */
    jd_image im;
    int rc = jd_decode(g->jpg, g->len, &o, &im);
    int ok = rc == JD_OK && im.pix && im.warnings == 0 && jd_fnv64(im.pix, im.stride * (size_t)im.h) == g->s->h_int;
    jd_free(&im);
    return ok;
}

typedef struct jb_worker { const jb_img *g; const jd_kernels *k; int reps, pin; HANDLE go; volatile LONG bad, done; } jb_worker;
static DWORD WINAPI jb_worker_proc(LPVOID p) {
    jb_worker *w = (jb_worker *)p;
    if (w->pin >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << w->pin);
    if (WaitForSingleObject(w->go, 30000) != WAIT_OBJECT_0) { w->bad = -1; return 1; }
    for (int i = 0; i < w->reps; i++) { if (!jb_decode_ok(w->g, w->k)) w->bad++; w->done++; }
    return 0;
}

/* one sample with n threads; returns seconds or < 0 (thread creation failed / timeout); *bad gets wrong decodes */
static double jb_sample(const jb_img *g, const jd_kernels *k, int n, int reps, int pin, long *bad) {
    static jb_worker w[JB_MAXT];
    HANDLE th[JB_MAXT];
    HANDLE go = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!go) return -1;
    int created = 0;
    for (int i = 0; i < n; i++) {
        memset(&w[i], 0, sizeof w[i]);
        w[i].g = g; w[i].k = k; w[i].reps = reps; w[i].go = go; w[i].pin = n == 1 ? pin : -1;
        th[i] = CreateThread(NULL, 256 * 1024, jb_worker_proc, &w[i], 0, NULL);
        if (!th[i]) break;
        created++;
    }
    uint64_t t0 = t_qpc();
    SetEvent(go);
    int all = created ? WaitForMultipleObjects((DWORD)created, th, TRUE, 120000) == WAIT_OBJECT_0 : 1;
    uint64_t t1 = t_qpc();
    for (int i = 0; i < created; i++) {
        if (!all) TerminateThread(th[i], 1);
        CloseHandle(th[i]);
        *bad += w[i].bad;
    }
    CloseHandle(go);
    if (created < n) { t_note("created %d of %d decode threads (error %lu)", created, n, GetLastError()); return -1; }
    if (!all) { CHECKF(0, "%d decode threads did not finish within 120 s", n); return -1; }
    return t_qpc_sec(t0, t1);
}

static void jb_case(const char *name) {
    jb_img *g = jb_get(name);
    const sj_synth *s = g->s;
    jb_tier tier[3]; int nt = jb_tiers(tier);
    size_t per_thread = (size_t)s->w * s->h * (s->sub == 0 ? 1 : 3) * 3;
    int ncpu = t_cpu_count(), maxt = ncpu < JB_MAXT ? ncpu : JB_MAXT;
    if ((size_t)maxt * per_thread > JB_BUDGET) maxt = (int)(JB_BUDGET / per_thread);
    if (maxt < 1) maxt = 1;
    int threads[8], nth = 0;
    for (int t = 1; t <= maxt && nth < 8; t *= 2) threads[nth++] = t;
    t_info("threads", "up to %d (logical processors %d, buffer budget %u MB)", threads[nth - 1], ncpu, (unsigned)(JB_BUDGET >> 20));
    double pix = (double)s->w * s->h;
    double target = bm_full() ? 8e6 : 2e6;                 /* pixels per thread per sample */
    int reps = (int)ceil(target / pix); if (reps < 1) reps = 1;
    int samples = bm_full() ? 7 : 3;
    int pin = bm_choose_cpu();
    for (int ti = 0; ti < nt; ti++) {
        if (!tier[ti].usable) { t_note("%s kernel set not usable on this machine: tier skipped", tier[ti].name); continue; }
        int ok = jb_decode_ok(g, tier[ti].k);
        CHECKF(ok, "%s %s: decoded pixels differ from djpeg (hash) - no numbers reported for this tier", name, tier[ti].name);
        if (!ok) continue;
        double one = 0;
        for (int hi = 0; hi < nth; hi++) {
            int n = threads[hi];
            long bad = 0;
            if (jb_sample(g, tier[ti].k, n, 1, pin, &bad) < 0) break;          /* warm-up */
            double v[16]; int got = 0;
            for (int r = 0; r < samples; r++) {
                double sec = jb_sample(g, tier[ti].k, n, reps, pin, &bad);
                if (sec <= 0) break;
                v[got++] = pix * reps * n / sec / 1e6;
            }
            CHECKF(bad == 0, "%s %s %d threads: %ld decodes produced wrong pixels", name, tier[ti].name, n, bad);
            if (bad || got < samples) break;
            t_stats st; t_stats_compute(v, got, &st);
            char m[96];
            snprintf(m, sizeof m, "jpeg_decode_%s_%s_%dt", name + 6 /* drop "synth_" */, tier[ti].name, n);
            t_perf_stats(m, &st, "MPix/s");
            if (n == 1) one = st.median;
            else if (one > 0) { char key[128]; snprintf(key, sizeof key, "%s.scaling", m); t_info(key, "%.2fx of 1 thread", st.median / one); }
        }
    }
}

TEST_MANUAL(bench_mode_full) {
    g_bm_full = 1;
    t_note("full mode selected: 7 samples of >= 8 MPix per thread, 8K image included");
    CHECK(bm_full());
}
TEST(jpeg_env_report) {
    jb_tier t[3]; jb_tiers(t);
    for (int i = 0; i < 3; i++) { char k[32]; snprintf(k, sizeof k, "tier_%s", t[i].name); t_info(k, "%s", t[i].usable ? "usable" : "not usable"); }
    t_info("ncpu", "%d", t_cpu_count());
    t_info("cpu", "%s", cpu()->brand);
    t_info("bench_cpu", "%d (single-thread rows are pinned to it; multi-thread rows are not pinned)", bm_choose_cpu());
    t_info("mode", "%s", bm_full() ? "full" : "quick");
    if (t_is_wine()) t_note("running under Wine %s: numbers describe the host + Wine, not Windows", t_wine_version());
}
TEST(jpeg_decode_vga_gray)    { jb_case("synth_vga_gray"); }
TEST(jpeg_decode_512_420_rst) { jb_case("synth_mt_420"); }
TEST(jpeg_decode_1k_444_rst)  { jb_case("synth_1k_444_rst"); }
TEST(jpeg_decode_4k_420)      { jb_case("synth_4k_420"); }
/* manual: 100 MB per decoded image and a ~7 MB encode; runs with --include-manual (which also selects
 * full mode through bench_mode_full, the first case in source order) */
TEST_MANUAL(jpeg_decode_8k_420) { jb_case("synth_8k_420"); }
