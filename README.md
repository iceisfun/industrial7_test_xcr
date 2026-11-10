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
scripts/   run_all.ps1 (Windows), run_all.sh (Linux + Wine), compare_results.py
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
