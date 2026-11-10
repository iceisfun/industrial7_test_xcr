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
implementation.** The owner's full guidance is in [docs/GUIDANCE.md](docs/GUIDANCE.md), the map from
guidance to cases is [docs/COVERAGE.md](docs/COVERAGE.md), and [HANDOFF.md](HANDOFF.md) is the procedure
for running the suite on real Windows 10/11 and on other environments and reporting the results.

## Exes

31 exes, each built for x64 and x86 (`build/x64/<name>.exe`, `build/x86/<name>.exe`). Every exe is
independent: run any one alone.

| exe | what it proves |
|---|---|
| cpu_features | CPUID leaves and subleaves (0, 1, 7, 0Dh, 4, 0Bh/1Fh, extended) are well formed and stable across calls, threads and every CPU of the affinity mask; IsProcessorFeaturePresent claims are checked by executing the instruction |
| xcr0_probe | XGETBV legality, XCR0 invariants, XINUSE, CPUID 0Dh consistency, XCR0 vs IA32_XSS separation, XSETBV is privileged, AVX/AVX-512/AMX eligibility rules, XCR0 identical across CPUs, processes and the WOW64 view; native XCR0 vs GetEnabledXStateFeatures |
| xstate_windows | GetEnabledXStateFeatures, InitializeContext(2), Get/SetXStateFeaturesMask, LocateXStateFeature, CopyContext and the ntdll CONTEXT_EX layer against their documented (decomp-pinned) contracts, including invalid flags, unsupported bits and undersized buffers |
| xsave_roundtrip | XSAVE / XSAVEOPT / XSAVEC / XRSTOR / FXSAVE / FXRSTOR from user mode with legal masks: layouts, init/modified optimisation, header rules, #GP cases (in children) |
| xmm_registers | XMM data movement and preservation across preemption, page faults, exceptions, context APIs and calls into Windows (x64 ABI: XMM6-15 nonvolatile) |
| ymm_registers | YMM lanes, VEX.128 vs legacy SSE upper-lane rules, VZEROUPPER/VZEROALL, AVX XSAVE component, 256-bit state across OS events, #UD when YMM state is disabled |
| zmm_registers | AVX-512 ZMM0-31 and k0-k7, EVEX masking/broadcast/rounding, fault suppression, gather/scatter, FP16/BF16/VBMI2/BITALG reference models, AVX-512 state across OS events and context APIs |
| sse_arithmetic | SSE..SSE4.2 arithmetic, integer, conversion, shuffle and string instructions bit-exact against scalar C models |
| avx_arithmetic | AVX / AVX2 / FMA3 / F16C / BMI1 / BMI2 / ADX reference vectors, rounding modes, DAZ/FTZ, NaN rules, SIMD FP exceptions |
| fp_environment | MXCSR and x87 control/status: initial values (new threads, 0x1F80 / 0x27F), rounding, DAZ/FTZ, sticky flags, masked/unmasked exceptions, CRT float APIs, context round trips |
| thread_xstate | per-thread SIMD/FP isolation through every wait primitive, APCs, thread pools, suspension and migration with 2..1024 threads |
| context_xstate | Get/SetThreadContext with XSTATE-aware CONTEXT buffers on running, suspended and kernel-blocked threads, RtlCaptureContext, WOW64 contexts, parameter validation |
| exception_xstate | register preservation and CONTEXT fidelity through SEH, VEH, VCH, RaiseException, hardware faults (#DE, #UD, #AV, guard page, #MF, #XM), longjmp and RtlRestoreContext, compared with instruction-level reference captures |
| fiber_xstate | fibers with and without FIBER_FLAG_FLOAT_SWITCH: MXCSR/x87 CW per fiber, nonvolatile XMM/GPR, FLS, migration, parameter validation |
| cxx_exceptions | C++ throw/catch through the x64 OS unwinder (RtlUnwindEx, UWOP_SAVE_XMM128) preserves exactly the ABI nonvolatile registers (x86: SKIP by design) |
| alignment | aligned/unaligned SIMD and GPR access, page-boundary straddling, faults on misaligned or inaccessible operands, stack/heap/VirtualAlloc alignment, misaligned CONTEXT buffers |
| simd_memory | non-temporal stores and fences, cache control, prefetch, masked-access fault suppression, gathers/scatters, CMPXCHG8B/16B, RDRAND/RDSEED, MOVDIRI/MOVDIR64B, OS handling of their faults |
| simd_dispatch | runtime detection and implementation selection the way applications do it (including synthetic CPUID/XCR0 matrices); every usable tier bit-identical to scalar |
| simd_crypto | AES-NI, PCLMULQDQ, SHA, GFNI, VAES/VPCLMULQDQ, SHA512 against bit-exact models and published known answers; state survives OS events |
| simd_jpeg | in-tree baseline/progressive JPEG decoder (scalar/SSE2/AVX2) and stb_image vs libjpeg-turbo golden hashes; corrupt/truncated input; multithreaded decode |
| simd_png | in-tree inflate + PNG decoder (scalar/SSE2/SSSE3/AVX2) vs Python zlib vectors; corrupt/truncated input; multithreaded decode |
| feature_faults | unsupported (absent ISA), privileged and disabled-state instructions raise the right exception and status code; every probe in a child process |
| xstate_stress | many-thread contamination and context-switch stress with a capture oracle; churn, timers, suspend/get/set controllers |
| amx_xstate | AMX TILECFG/TILEDATA as an opt-in (XFD) component: EnableProcessOptionalXStateFeatures, first-use, preservation, context capture (x64, Windows 11, AMX hardware) |
| apx_xstate | APX R16-R31 (XSAVE component 19): enumeration, layout, #UD when disabled, CONTEXT XSTATE path |
| cet_xstate | CET user shadow stacks: policy, RDSSP per thread, CET_U context SSP, RIP redirect/longjmp/fiber transfers under enforcement (x64) |
| simd_bench | PERF: scalar/SSE/AVX/AVX-512 throughput and latency, memcpy/memset, CRC32C, AES, RGB/YUV, transitions, XSAVE variants, context capture |
| memory_bench | PERF: memcpy/memset bandwidth 4 KB .. 256 MB per kernel, labelled by cache level |
| context_bench | PERF: RtlCaptureContext, Get/SetThreadContext (CONTROL/FP/XSTATE), exception round trips, fiber switches, thread create |
| jpeg_bench | PERF: JPEG decode MPix/s by image size x tier x threads |
| smoke_selftest | the harness itself: result kinds, crash recovery, child pass/fail/crash/timeout, JSON validity, exit codes |

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
docs/      GUIDANCE.md (owner guidance), COVERAGE.md (guidance -> cases map, validation status)
corpus/    JPEG/PNG inputs and reference vectors
results/baselines/  committed reference runs (everything else under results/ is ignored)
HANDOFF.md how to run on Windows / other environments and what to report back
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

Prerequisites on Debian/Ubuntu: `gcc-mingw-w64-x86-64 gcc-mingw-w64-i686 g++-mingw-w64-x86-64
g++-mingw-w64-i686` (the exes are linked statically and need no runtime DLLs). `make check-isa` must
report 0 functions; the integration audit also disassembles every exe and confirms that VEX/EVEX/AMX
instructions appear only in the runtime-gated `T_TARGET` functions and the `t__ymm_*`/`t__zmm_*` asm helpers.

Windows-native builds (best effort, through `CMakeLists.txt`; the mingw Makefile is the reference):

```
cmake -S . -B build-cl   -G Ninja -DCMAKE_C_COMPILER=clang-cl                       (x64)
cmake -S . -B build-cl32 -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_C_FLAGS=-m32  (x86)
cmake --build build-cl
```

clang (MSVC or GNU target) and mingw gcc also work through CMake. Plain MSVC `cl.exe` is **not**
supported: the helpers need GNU inline asm, top-level asm and per-function target attributes, so CMake
stops with an error. Never add `/arch:AVX*`, `-mavx*` or `-march`.

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
| `-xsave_default V` (XCR_SDE_XCR0=V) | only the emulated XGETBV returns V; CPUID.(0Dh,0).EBX, GetEnabledXStateFeatures, IsProcessorFeaturePresent and the other-architecture process keep the host XCR0 (xcr0_probe expected failures) |
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

| option | effect |
|---|---|
| `--json=<file>` | write JSON lines (below) |
| `--filter=a,b` | run only cases whose name contains one of the substrings |
| `--list` | print the case names and exit |
| `--verbose` / `-v` | print `t_log` diagnostics |
| `--case-timeout=<sec>` | per-case watchdog (default 120, 0 = off); a case that exceeds it is a CRASH |
| `--include-manual` | also run long/manual cases (8K decodes, full benchmark sweeps) |

Exit codes: 0 = no FAIL and no CRASH, 1 = at least one FAIL or CRASH, 2 = harness error (bad argument,
unwritable `--json` file, no case matches the filter).

## JSON output

`--json=<file>` writes one JSON object per line ([common/API.md](common/API.md) section 9):

```
{"type":"env", "exe", "version", "compiler", "arch", "wow64", "native_machine", "timestamp",
 "os":{major,minor,build,ubr,product}, "wine"?, "sde"?, "cpu":{vendor,brand,signature,family,model,...},
 "features_hw", "features_usable", "osxsave", "xcr0", "xinuse", "xsave_size_*", "api_xstate_features",
 "pf_present", "ncpu", "affinity", ...}
{"type":"case", "exe", "arch", "case", "result":"PASS|FAIL|SKIP|XFAIL|CRASH|PERF", "ms", "checks", "failed",
 "reason", "failures":[{"file","line","expr","msg"}...], "notes":[...], "info":{key:value},
 "perf":[{"metric","value","unit","n","min","max","mean","stddev","q1","q3"}], "exception"?}
{"type":"summary", "exe", "arch", "total", "pass", "fail", "crash", "skip", "xfail", "perf", "checks",
 "checks_failed", "ms", "exit_code"}
```

The runners add `<arch>/run.json` (per exe: status OK/FAIL/CRASH/TIMEOUT/HARNESS_ERROR, exit code,
duration, sha256, env) and `host.json`. `scripts/compare_results.py` reads exactly these files.

## Known SKIPs and Wine caveats

SKIP means the capability is absent; it is never a failure. On a typical host expect:

- **no AVX-512** (most client CPUs): about 225 cases per arch SKIP (zmm_registers ~120, simd_memory,
  alignment, simd_bench, memory_bench, exception_xstate ZMM frames). Use an AVX-512 host or Intel SDE.
- **no AMX** or no Windows 11 optional-XSTATE APIs: every amx_xstate case SKIPs.
- **no APX**: apx_xstate runs only the disabled-state #UD probes and SKIPs the rest.
- **CET shadow stacks not enforced** for the process: cet_xstate records the policy and SKIPs.
- **x86 build**: cxx_exceptions SKIPs (GCC i686 C++ does not unwind through SEH); AMX/APX/CET are x64-only.
- Windows 10: cases that need Windows 11 APIs (GetThreadEnabledXStateFeatures,
  EnableProcessOptionalXStateFeatures, QueueUserAPC2 special APCs) SKIP and name the build they need.
- Single processor group: `xcr0_identical_across_processor_groups` SKIPs after its group-0 checks.

Wine (11.0 here) is plumbing validation only. Wine gaps are guarded by `t_is_wine()` and reported as
XFAIL with a reason starting "Wine", never by weakening a check: context API last-error handling,
CopyContext flag validation, CONTEXT_EX validation, #MF/#XM/#DE status mapping, privileged opcodes
reported as access violations, vectored continue handlers never called, SwitchToFiber not switching
MXCSR/x87 CW, special user APCs, stack overflow handling, x86 GetThreadContext inside system calls, and
missing exports (RtlCaptureContext2, Wow64Get/SetThreadContext, the Windows 11 XSTATE APIs) which SKIP.
An XFAIL with a "Wine" reason must never appear on real Windows.

## Related corpora

Sibling conformance suites for the same differential method (Windows as the reference, any other
environment as the candidate):

- [industrial7_test_ntdll](https://github.com/iceisfun/industrial7_test_ntdll),
  [industrial7_test_kernel32_kernelbase](https://github.com/iceisfun/industrial7_test_kernel32_kernelbase),
  [industrial7_test_advapi32](https://github.com/iceisfun/industrial7_test_advapi32),
  [industrial7_test_msvcrt](https://github.com/iceisfun/industrial7_test_msvcrt)
- [industrial7_test_user32](https://github.com/iceisfun/industrial7_test_user32),
  [industrial7_test_win32k](https://github.com/iceisfun/industrial7_test_win32k),
  [industrial7_test_gdi](https://github.com/iceisfun/industrial7_test_gdi),
  [industrial7_test_gdi32_gdifull](https://github.com/iceisfun/industrial7_test_gdi32_gdifull),
  [industrial7_test_theme](https://github.com/iceisfun/industrial7_test_theme),
  [industrial7_test_dwrite](https://github.com/iceisfun/industrial7_test_dwrite),
  [industrial7_test_d3d](https://github.com/iceisfun/industrial7_test_d3d),
  [industrial7_test_shell32](https://github.com/iceisfun/industrial7_test_shell32),
  [industrial7_test_ole](https://github.com/iceisfun/industrial7_test_ole)
- [industrial7_test_ws2](https://github.com/iceisfun/industrial7_test_ws2),
  [industrial7_test_wininet](https://github.com/iceisfun/industrial7_test_wininet),
  [industrial7_test_curl](https://github.com/iceisfun/industrial7_test_curl),
  [industrial7_test_tls](https://github.com/iceisfun/industrial7_test_tls),
  [industrial7_test_crypt](https://github.com/iceisfun/industrial7_test_crypt),
  [industrial7_test_webauthn](https://github.com/iceisfun/industrial7_test_webauthn),
  [industrial7_test_winmm](https://github.com/iceisfun/industrial7_test_winmm)

Industrial7 (a from-scratch OS that runs Windows exes) is one of the environments these suites are run
on; nothing in the test logic depends on it.

## Writing a test exe

`tools/new_exe.sh <name> "<purpose>"`, then follow [common/API.md](common/API.md):

- Gate every ISA use on `cpu_has(F_...)`.
- Hold register patterns with the `t_vregs_*` assembly helpers.
- Put deliberate faults in `CHILD()` processes.
- Resolve optional Windows APIs with `T_PROC`.
- Cite the SDM/APM or Microsoft contract that each hard check relies on.
