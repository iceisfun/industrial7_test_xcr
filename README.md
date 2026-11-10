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
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-Arch all|x64|x86] [-ExeDir build\x64] [-OutDir results]
           [-TimeoutSec 600] [-Only <substr>] [-Baseline auto|<name>] [-Compare auto|<dir>] [-Note "..."]
```

The script writes `results\<arch>\<name>.json` (JSON lines), `results\<arch>\<name>.txt` (console
output), `results\<arch>\run.json` (exit codes, status, duration, exe sha256 and environment) and
`results\host.json` (OS build/UBR/DisplayVersion, CPU, hypervisor, git revision). The x86 exes run as
WOW64 processes on 64-bit Windows. `-Baseline auto` writes to `results\baselines\<os>-<cpu>\`, the
committed reference runs (see [results/baselines/README.md](results/baselines/README.md)).

Linux and Wine (plumbing validation only; Windows is the authority):

```
scripts/run_all.sh [--arch x64|x86|all] [--out results | --baseline auto|<name>] [--timeout 600] [--only <substr>]
tools/wine_run.sh build/x64/<name>.exe [--filter=...]       # headless, throw-away prefix, hard timeout
make wine-run                                               # all exes, sequentially, summary lines
```

Compare and triage:

```
python3 scripts/compare_results.py results_win11 results_other [--perf-threshold=0.10] [--json=diff.json]
python3 scripts/compare_results.py --baseline results          # against the closest committed baseline
python3 scripts/compare_results.py --triage results            # FAIL/CRASH/XFAIL list + expected-on-Windows rows
python3 scripts/compare_results.py --list-baselines
```

The comparison reports environment differences, exe status changes, functional mismatches (tagged
REGRESSION / FIXED / XFAIL-FLIP / CHANGED), capability (SKIP vs run) differences, missing cases and
PERF deltas. Its exit status is 1 when there are functional mismatches other than PASS<->XFAIL flips
(`--strict` counts those too) or an exe that was OK is no longer OK.

## Emulated validation (Intel SDE)

A host without AVX-512 (or without AMX, APX, ...) only SKIPs those cases. `tools/sde_run.sh` runs an exe
under the [Intel Software Development Emulator](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html)
inside the same headless, throw-away Wine sandbox as `tools/wine_run.sh` (one wine tree at a time: runs
are serialised with a lock file; hard timeout, default 1800 s because SDE is 10-100x slower). SDE emulates
CPUID, XGETBV and every instruction of the chosen CPU model that the host lacks, so the AVX-512 code
(EVEX encodings, opmasks, fault suppression, compress/expand, gather/scatter, ZMM XSAVE layouts) really
executes. Like Wine itself, this is plumbing validation: Windows remains the authority.

```
tools/sde_run.sh --fetch                                 # once: download SDE into ~/.cache/xcr_suite/sde
sudo apt install wine64 wine32                           # once: the distro Wine (see below)
tools/sde_run.sh --cpu=spr build/x64/zmm_registers.exe   # Sapphire Rapids (default); also icx, skx, gnr, future ...
tools/sde_run.sh --cpu=spr build/x86/zmm_registers.exe   # 32-bit exes work the same way
tools/sde_run.sh --cpu=nhm build/x64/cpu_features.exe    # Nehalem: no AVX, no XSAVE/OSXSAVE
XCR_SDE_ARGS="-xsave_default 0x3" tools/sde_run.sh --cpu=hsw build/x64/xcr0_probe.exe   # AVX in CPUID, XCR0 without YMM
for a in x64 x86; do mkdir -p sde_spr/$a; for e in build/$a/*.exe; do   # everything, one model
  tools/sde_run.sh --cpu=spr "$e" --json=Z:$PWD/sde_spr/$a/$(basename "$e" .exe).json > sde_spr/$a/$(basename "$e" .exe).txt 2>&1
done; done
```

What each model proves:

- `spr` / `icx` / `future`: the AVX-512 (and AMX/AVX-512 FP16/...) cases run instead of SKIPping, and are
  checked against the same architectural rules as on hardware.
- `nhm` / `wsm`: SDE's chip check stops any process that executes an instruction the model lacks, so an
  exe that reaches its SUMMARY line proves that the baseline code needs no AVX (or SSE4.2/AES on nhm).
  The "feature absent" paths run: `cpu_has()` gating, simd_dispatch tier fallback, the SKIP paths, and
  the #UD probes (feature_faults `feat_*`, `feat_xgetbv_without_osxsave`, xcr0_probe and ymm/zmm
  `*_ud_*` cases) up to the faulting instruction.
- `hsw` with `-xsave_default 0x3`: AVX is advertised but the OS-enabled XCR0 lacks YMM state, which
  exercises `avx_ud_when_ymm_state_disabled` / `ymm_vex_faults_when_avx_disabled`.

### Wine for SDE

Pin cannot run Proton/Wine 11 builds here: they install a seccomp syscall filter and their signal path
crashes under Pin, so every hardware exception kills the process. `sde_run.sh` therefore uses the distro
Wine (`/usr/lib/wine/wine64`, Wine 9; `XCR_SDE_WINE` overrides) with its own base prefix
(`~/.cache/xcr_suite/prefix_sde`, created on first use) and sets `WINELOADERNOEXEC=1` (Pin cannot follow
Wine's exec of the static preloader). Wine 9 differs from Wine 11 in ways that are real Wine
discrepancies, not SDE artifacts, and are reported as such:
`GetEnabledXStateFeatures` does not report AVX-512 (XFAIL "Wine ...: ... does not report AVX-512", so the
exception/context ZMM cases SKIP), it returns 0 on a CPU without XSAVE (XFAIL), and on x64 it groups
Rbp with CONTEXT_CONTROL (context_xstate `getctx_control_only_leaves_rest`,
`getctx_integer_only_leaves_rest`, `setctx_integer_observed`) and returns ERROR_NOACCESS instead of
ERROR_MORE_DATA for a compact context without room (xstate_windows `thread_get_compact_no_room`): FAIL.
Wine 9 also reports every x64 #XM as `STATUS_FLOAT_MULTIPLE_TRAPS` (Windows x64: the one `STATUS_FLOAT_*`
code of the unmasked flag; Wine 11 is exact): XFAIL "Wine 9.0 reports an x64 #XM as ...". Because a CONTEXT
carries only the components `GetEnabledXStateFeatures` reports, exception_xstate runs its ZMM frames only
where that API reports AVX-512 and otherwise SKIPs them (or uses YMM frames) with that reason.

### Emulator artifacts

`sde_run.sh` exports `XCR_SDE=<model>`; tests read it with `t_sde_model()` and use it **only** to mark
known SDE behaviour as an expected failure (`t_expect_fail`, so the check still runs and a pass shows as
XPASS) or, where the case would kill the whole process, as XFAIL without running it. Without `XCR_SDE`
nothing changes. The ENV header shows `sde=<model>` and the JSON env record has `"sde"`. SDE is run with
`-ptr_raise 1` (an emulated access to an inaccessible page raises #PF like hardware), `-raise_int3 1` and
`-chip_check_emit_file 1`; the harness compares SDE's per-process report files in `XCR_SDE_LOGDIR_WIN`
around every child run, and when SDE stopped the child (chip check instead of #UD) the case becomes
XFAIL with SDE's message (`t_sde_killed_child`). Known artifacts of SDE 10.13 on an AVX2 host:

| area | SDE behaviour (hardware/SDM behaviour in brackets) |
|---|---|
| chip check | stops the process on an instruction the model or XCR0 lacks [#UD] |
| XSAVE/XSAVEOPT/XSAVEC with CPUID.OSXSAVE clear (`nhm`, `wsm`) | stops the process with "OSXAVE BIT IS NOT SET IN CPUID" on Pin's stderr, no report file [#UD] |
| emulated EVEX aligned moves, MOVDIR64B, XSAVE family, legacy SHA m128 | no #GP for a misaligned operand |
| XSETBV / XGETBV | XSETBV works at CPL3, XGETBV accepts any index [#GP] |
| XSAVE emulation | XCOMP_BV omits PKRU, initial components and (32-bit) Hi16_ZMM, is 0 for RFBM=0; XRSTOR with partial RFBM touches other components; FXRSTOR does not set XINUSE[0]; no XRSTOR header #GP, no #PF on the area |
| VPCOMPRESSD / VPEXPANDD | fault on popcnt(k) elements that end at a guard page [only the elements accessed] |
| emulated stores, scatters, NX fetch | page fault reported as a read (ExceptionInformation[0]=0) |
| gathers | elements below the faulting one are not completed before the fault |
| unmasked #XM in an emulated EVEX instruction | Pin aborts the process |
| RDPID, CLFLUSHOPT/CLWB on no-access pages, ICEBP | own TSC_AUX value; no fault; no #DB |
| TF single-step, RSM/SYSEXIT/SYSRET, INT 80h | Pin aborts the process |
| DR0-DR3 execute breakpoints set with SetThreadContext | never fire: Pin runs the code from its code cache [#DB] |
| x87 #MF in 32-bit code | FPU instruction pointer points into Pin's code cache |
| CPUID topology (leaves 0Bh/1Fh, APIC IDs) | one emulated APIC ID on every CPU |
| `-nhm`: F3 0F BD / F3 0F BC | run as the host's LZCNT/TZCNT instead of BSR/BSF |
| reserved FXSAVE bytes in SetThreadContext | Pin asserts (case not run) |
| a thread that keeps suspending/capturing other threads (xstate_stress `ctl_*`, rarely simd_bench `ctx_getthreadcontext_control`) | Pin deadlocks at random and the process hangs (xstate_stress `ctl_*` not run under SDE) |
| more than 256 threads in a 32-bit process | Pin asserts in CreateThreadData (thread_xstate `*_n1024`, simd_jpeg `mt_decode_1024_*` not run) |

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
