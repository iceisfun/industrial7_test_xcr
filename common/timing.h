/* timing.h - QPC time, TSC diagnostics, statistics, affinity pinning, benchmark loop.
 * QPC is the time base for every PERF number. RDTSC/RDTSCP are diagnostics only: never assume
 * TSC ticks == core cycles. */
#ifndef T_TIMING_H
#define T_TIMING_H
#include <stdint.h>
#include <stddef.h>

uint64_t t_qpc(void);                      /* QueryPerformanceCounter ticks */
uint64_t t_qpc_freq(void);                 /* ticks per second (cached) */
double   t_qpc_sec(uint64_t t0, uint64_t t1);
double   t_qpc_ms(uint64_t t0, uint64_t t1);
double   t_now_ms(void);                   /* monotonic ms since first call */

uint64_t t_rdtsc(void);                    /* requires cpu_has(F_TSC) */
uint64_t t_rdtscp(uint32_t *aux);          /* requires cpu_has(F_RDTSCP); aux may be NULL */

typedef struct t_stats { int n; double min, max, mean, median, stddev; } t_stats;
/* Statistics over v[0..n-1] (v is not modified). n == 0 gives all zeros. */
void t_stats_compute(const double *v, int n, t_stats *s);

/* Affinity: pin the calling thread to logical processor `cpu` of its current group.
 * Returns the previous mask (pass to t_unpin) or 0 on failure. */
uintptr_t t_pin_thread(int cpu);
void      t_unpin_thread(uintptr_t prev_mask);
int       t_cpu_count(void);               /* logical processors in the process affinity mask */

/* Anti dead-code-elimination sink. T_SINK(x) consumes a value; t_sink_ptr() forces memory. */
extern volatile uint64_t t_sink;
#define T_SINK(x) (t_sink += (uint64_t)(x))
void t_sink_ptr(const void *p);

/* Benchmark loop: fn(ctx) is one iteration (it may do many internal operations).
 * Runs `warmup` untimed iterations, then `reps` timed samples; each sample repeats fn until at least
 * min_sample_ms elapsed (>= 1 call). Stores seconds-per-call statistics. Bounded by reps*min_sample_ms
 * plus one call per sample. Returns the number of samples (<= 256). */
typedef void (*t_bench_fn)(void *ctx);
int t_bench(t_bench_fn fn, void *ctx, int warmup, int reps, double min_sample_ms, t_stats *sec_per_call);
#endif
