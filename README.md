# CPU SIMD and extended processor state: Windows conformance suite

A set of standalone Windows executables. Each one tests one part of the contract between an
operating system and the programs that use CPU SIMD and extended processor state:

- CPUID feature reporting, XCR0 and XSAVE layout
- the Windows XSTATE/CONTEXT APIs
- preservation of XMM/YMM/ZMM/opmask/x87/MXCSR state across threads, waits, exceptions, fibers and
  context operations
- the ABI rules for which registers a function must preserve
- SIMD arithmetic correctness and fault behaviour
- simple performance comparisons

The same exe runs on ordinary Windows 10/11 and on any other environment that runs Windows programs:
compatibility layers, hypervisors, emulators and other operating systems. Comparing the results from
two environments shows where they differ. Guiding principle: **test the OS contract, not the processor
implementation.** The owner's full guidance is in [docs/GUIDANCE.md](docs/GUIDANCE.md).

## Result model

Every test case ends in exactly one of these results:

| result | meaning |
|---|---|
| PASS    | the contract held |
| FAIL    | a contract violation: the feature is advertised or documented but behaves incorrectly |
| SKIP    | the capability is absent on this machine or OS (not a failure) |
| XFAIL   | a documented, expected failure (for example, a known gap in an environment), with a reason |
| CRASH   | unexpected termination inside the case. It is recovered when possible, and the run continues. |
| TIMEOUT | assigned only by the runner scripts, when an exe exceeds its time limit |
| PERF    | a measurement only. It never fails the run. |

Facts that vary between machines or APIs are recorded as NOTE/INFO, never as failures.

Exit codes: **0** means no FAIL and no CRASH (SKIP, XFAIL and PERF are fine). **1** means at least one
FAIL or CRASH. **2** means a harness error.

## Layout

```
common/    harness (t.h/t.c), child processes, cpuid, feature_detect, timing, pattern, asm helpers,
           env recorder, util; API reference in common/API.md
tests/     one <name>.c per exe  -> build/x64/<name>.exe, build/x86/<name>.exe
benchmarks/ one <name>.c per benchmark exe, same harness and output (see "Benchmarks" below)
scripts/   run_all.ps1 (Windows), run_all.sh (Linux + Wine), compare_results.py, bench_table.py
tools/     wine_run.sh, make_base_prefix.sh, reap_wine.py (headless Wine sandbox), new_exe.sh, check_isa.py
docs/      GUIDANCE.md
```

## Build (Linux, mingw-w64)

```
make            # build/x64/*.exe and build/x86/*.exe (static, no AVX in baseline code)
make x64 | x86  # one architecture
make one T=smoke_selftest
make check-isa  # report VEX/EVEX instructions outside runtime-gated functions
```

The exes are built without `-march` and without `-mavx*`, so they start on CPUs that have no AVX.
Wider instruction sets live only in functions marked with `T_TARGET(...)` and in assembly helpers,
and run only after runtime detection. The x86 build assumes SSE2, which Windows 8 and later require.

A Windows-native build with clang-cl is possible through `CMakeLists.txt` (best effort). Plain MSVC
`cl.exe` is not supported, because the helpers use GNU inline asm.

## Run

Windows: copy `build\` (or just the exes) and `scripts\`, then run

```
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-ExeDir build\x64] [-OutDir results] [-TimeoutSec 600]
```

The script writes `results\<arch>\<name>.json` (JSON lines), `results\<arch>\<name>.txt` (console
output) and `results\<arch>\run.json` (exit codes, status, duration and environment).

Linux and Wine (plumbing validation only; Windows is the authority):

```
scripts/run_all.sh [--arch x64|x86|all] [--out results] [--timeout 600]
tools/wine_run.sh build/x64/<name>.exe [--filter=...]       # headless, throw-away prefix, hard timeout
make wine-run                                               # all exes, sequentially, summary lines
```

Compare two environments:

```
python3 scripts/compare_results.py results_win11 results_other [--perf-threshold=0.10] [--json=diff.json]
```

The comparison reports environment differences, exe status changes, functional mismatches,
capability (SKIP vs run) differences, missing cases and PERF deltas. Its exit status is 1 when there
are functional mismatches.

## Benchmarks

Benchmarks are ordinary suite exes: same CLI, same JSON, same result model. PERF rows never fail a
run; a case FAILs only when the computation it timed produced a wrong result or a contract the
measurement relies on is broken. Every PERF row carries the median, min, max, mean, stddev and the
quartiles q1/q3 of at least 9 samples (3 to 7 for jpeg_bench, whose samples are whole decodes), and
every case is pinned to one processor (a performance core on hybrid CPUs) unless it measures threads.
`SIMD_BENCH_MODE=full` or `--include-manual` selects longer runs.

The guidance (docs/GUIDANCE.md section 8) names `benchmarks/{simd_bench,memory_bench,context_bench,jpeg_bench}.c`.
They map to these files:

| guidance | file | what it measures |
|---|---|---|
| simd_bench    | `tests/simd_bench.c` (kept in tests/: its kernels and helpers `tests/simd_bench_*.h` are shared) | array add, integer multiply, FP throughput/latency, dot product, matmul, popcount, conversions, gather, memcpy/memset/strlen at 3 sizes, CRC32C, AES, CLMUL, RGB/YCbCr, AVX/SSE transitions, XSAVE variants, context capture, scheduler round trips, thread state stress |
| memory_bench  | `benchmarks/memory_bench.c`  | memcpy / memset: scalar, CRT, SSE2, SSE2 non-temporal, AVX, AVX-512, REP MOVSB/STOSB over 4 KB .. 256 MB (copies to 128 MB), aligned and misaligned, each row labelled L1/L2/L3/DRAM from the OS-reported cache sizes |
| context_bench | `benchmarks/context_bench.c` | RtlCaptureContext(2), Get/SetThreadContext for CONTROL, full FP and XSTATE (AVX, AVX-512), RaiseException and hardware #UD round trips through a vectored handler, SwitchToFiber with and without FIBER_FLAG_FLOAT_SWITCH, thread create/join (single and batches of 64) |
| jpeg_bench    | `benchmarks/jpeg_bench.c`    | JPEG decode MPix/s: image size (VGA gray .. 4K, 8K manual) x kernel tier (scalar, SSE2, AVX2) x threads (1 .. 16), every decode checked against libjpeg-turbo's pixel hash |

The Makefile builds `benchmarks/<name>.c` exactly like `tests/<name>.c` (with `-Itests`, so the
benchmarks reuse the tests' kernels), into `build/<arch>/<name>.exe`; the runner scripts pick them up
with the other exes.

Render the guidance section 5 table (metric x scalar / SSE2 / AVX2 / AVX-512, median and IQR, with the
CPU, OS, affinity and frequency of the run) from the JSON, optionally against a second run:

```
python3 scripts/bench_table.py results_win11 [--compare results_other] [--format text|md|csv] [--exe memory_bench] [--metric memcpy]
```

The tier column is taken from tokens in the metric name (`sse2`, `avx`, `avx512`, `scalar`, ...); rows
without a tier token (CRT, REP MOVSB, API calls) appear in an "other" column. The ratio column is
oriented so that a value above 1 means the second run is faster, for rates and for times alike.

The env record's `version` is `git describe --always --dirty` at build time. `build/version.h` is
regenerated whenever that string changes, so a new HEAD or a dirty tree re-stamps every exe.

## Exe command line

```
<name>.exe [--json=<file>] [--filter=<substr>[,<substr>...]] [--list] [--verbose]
           [--case-timeout=<sec>] [--include-manual]
```

`--child=<case>` is internal: the exe uses it to re-run itself as a child process for cases that fault
on purpose (illegal instruction, access violation, privileged instruction) or that might hang.

## Writing a test exe

`tools/new_exe.sh <name> "<purpose>"`, then follow [common/API.md](common/API.md):

- Gate every ISA use on `cpu_has(F_...)`.
- Hold register patterns with the `t_vregs_*` assembly helpers.
- Put deliberate faults in `CHILD()` processes.
- Resolve optional Windows APIs with `T_PROC`.
- Cite the SDM/APM or Microsoft contract that each hard check relies on.
