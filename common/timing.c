/* timing.c - see timing.h */
#include "t.h"
#include <math.h>

volatile uint64_t t_sink;
void t_sink_ptr(const void *p) { __asm__ volatile("" :: "r"(p) : "memory"); }

uint64_t t_qpc(void) { LARGE_INTEGER t; QueryPerformanceCounter(&t); return (uint64_t)t.QuadPart; }
uint64_t t_qpc_freq(void) {
    static uint64_t f;
    if (!f) { LARGE_INTEGER x; QueryPerformanceFrequency(&x); f = x.QuadPart > 0 ? (uint64_t)x.QuadPart : 1; }
    return f;
}
double t_qpc_sec(uint64_t t0, uint64_t t1) { return (double)(int64_t)(t1 - t0) / (double)t_qpc_freq(); }
double t_qpc_ms(uint64_t t0, uint64_t t1) { return t_qpc_sec(t0, t1) * 1000.0; }
double t_now_ms(void) { static uint64_t base; if (!base) base = t_qpc(); return t_qpc_ms(base, t_qpc()); }

uint64_t t_rdtsc(void) { uint32_t lo, hi; __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi)); return (uint64_t)hi << 32 | lo; }
uint64_t t_rdtscp(uint32_t *aux) {
    uint32_t lo, hi, a;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(a));
    if (aux) *aux = a;
    return (uint64_t)hi << 32 | lo;
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
void t_stats_compute(const double *v, int n, t_stats *s) {
    memset(s, 0, sizeof *s);
    if (n <= 0) return;
    double *c = (double *)malloc(sizeof(double) * (size_t)n);
    if (!c) return;
    memcpy(c, v, sizeof(double) * (size_t)n);
    qsort(c, (size_t)n, sizeof(double), cmpd);
    double sum = 0; for (int i = 0; i < n; i++) sum += c[i];
    s->n = n; s->min = c[0]; s->max = c[n - 1]; s->mean = sum / n;
    s->median = (n & 1) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    for (int k = 0; k < 2; k++) {                  /* quartiles: position p*(n-1), linear interpolation */
        double pos = (k ? 0.75 : 0.25) * (n - 1); int i = (int)pos; double fr = pos - i;
        double q = i + 1 < n ? c[i] + fr * (c[i + 1] - c[i]) : c[i];
        if (k) s->q3 = q; else s->q1 = q;
    }
    double var = 0; for (int i = 0; i < n; i++) var += (c[i] - s->mean) * (c[i] - s->mean);
    s->stddev = n > 1 ? sqrt(var / (n - 1)) : 0;
    free(c);
}

uintptr_t t_pin_thread(int cpu) {
    DWORD_PTR pm = 0, sm = 0;
    if (cpu < 0 || cpu >= (int)(sizeof(DWORD_PTR) * 8)) return 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return 0;
    DWORD_PTR want = (DWORD_PTR)1 << cpu;
    if (!(pm & want)) return 0;
    return (uintptr_t)SetThreadAffinityMask(GetCurrentThread(), want);
}
void t_unpin_thread(uintptr_t prev) { if (prev) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)prev); }
int t_cpu_count(void) {
    DWORD_PTR pm = 0, sm = 0; int n = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) for (; pm; pm &= pm - 1) n++;
    if (!n) { SYSTEM_INFO si; GetSystemInfo(&si); n = (int)si.dwNumberOfProcessors; }
    return n > 0 ? n : 1;
}

int t_bench(t_bench_fn fn, void *ctx, int warmup, int reps, double min_sample_ms, t_stats *out) {
    double samples[256];
    if (reps > 256) reps = 256;
    if (reps < 1) reps = 1;
    for (int i = 0; i < warmup; i++) fn(ctx);
    uint64_t f = t_qpc_freq(), need = (uint64_t)(min_sample_ms * (double)f / 1000.0);
    for (int r = 0; r < reps; r++) {
        uint64_t calls = 0, t0 = t_qpc(), t1;
        do { fn(ctx); calls++; t1 = t_qpc(); } while (t1 - t0 < need);
        samples[r] = t_qpc_sec(t0, t1) / (double)calls;
    }
    t_stats_compute(samples, reps, out);
    return reps;
}
