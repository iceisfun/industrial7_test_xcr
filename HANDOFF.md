# Handoff: running the suite on Windows 10/11 and on other environments

This file is for the lane that runs the exes on genuine Windows 10/11 machines and on other environments
that run Windows programs (compatibility layers, hypervisors, emulators, other operating systems). The
suite has only been validated under headless Wine on Linux and under Intel SDE + Wine (see README.md).
**Nothing has run on real Windows yet.** Real Windows is the authority for every documented contract, so
the first Windows runs are the most valuable output of this lane.

## What to run, in priority order

1. Windows 10 22H2 (build 19045), x64 build and x86 build (WOW64), bare metal or a VM with AVX2.
2. Windows 11 23H2/24H2 (build 22631 / 26100), x64 + x86. Several cases need Windows 11
   (GetThreadEnabledXStateFeatures, EnableProcessOptionalXStateFeatures, special user APCs).
3. Any Windows 10/11 host with **AVX-512** enabled (Ice Lake / Sapphire Rapids / Zen 4 / Zen 5). About 225
   cases per arch SKIP everywhere else.
4. A Windows 11 host with **AMX** (Sapphire Rapids or later) for amx_xstate, and a CET-capable host (Tiger
   Lake+ / Zen 3+) with hardware-enforced stack protection for cet_xstate.
5. The other environments under test, on the same hardware as one of the Windows runs where possible, so
   that the comparison shows OS differences rather than CPU differences.

## Prerequisites

- Windows 10 1809+ or Windows 11, x64. PowerShell 5.1 (built in) or pwsh 7.
- No admin rights needed. Nothing is installed, no registry or system setting is changed. Every exe is
  bounded: at most 1024 threads (only when creation succeeds), under 512 MB of memory, child processes
  for every deliberate fault, a per-case timeout (120 s default) and a per-exe timeout in the runner.
- Antivirus may flag self-re-executing exes that fault on purpose. If an exe is quarantined, exclude
  the folder and say so in the run note.
- Python 3 (any host, not necessarily the Windows one) for `scripts/compare_results.py` triage.
- For comparable PERF numbers: AC power, a "High performance" or "Balanced" plan, no other load.

## Build

On Linux (the reference build):

```
sudo apt install gcc-mingw-w64-x86-64 gcc-mingw-w64-i686 g++-mingw-w64-x86-64 g++-mingw-w64-i686 zip
make clean && make all          # build/x64/*.exe, build/x86/*.exe (31 exes each)
make check-isa                  # must print 0 functions: no AVX outside runtime-gated code
scripts/pack_windows.sh         # build/xcr-windows-<version>.zip: exes + scripts + corpus
```

On Windows without Linux: clang-cl or mingw gcc through CMake (best effort, see CMakeLists.txt). Plain
MSVC `cl.exe` cannot build the suite (GNU inline asm and target attributes). Prefer the mingw build
from the zip: results are only directly comparable between identical exes (the runner records each
exe's sha256).

## Run

Copy the zip to the Windows machine, unzip it anywhere (a local disk, not a network share), then:

```
scripts\run_baseline.cmd "bare metal, VBS on, HVCI off"
```

or, with more control:

```
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Baseline auto -Note "..."
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Arch x64 -Only context -OutDir C:\xcr\r
```

A full run of both arches takes roughly 5-10 minutes on a desktop CPU. Single exes can be run by hand:

```
build\x64\context_xstate.exe --list
build\x64\context_xstate.exe --filter=setctx_ --verbose --json=ctx.json
```

The note should say: bare metal or VM (which hypervisor), VBS/HVCI/Memory Integrity state, any
mitigation policy changes, and anything unusual (AV exclusions, power plan).

## Collect results

`run_baseline.cmd` writes `results\baselines\<os>-<cpu>\` (host.json, x64\ and x86\ with run.json,
per-exe .json and .txt) and zips it to `results\baseline-<os>-<cpu>.zip`. Bring that zip back. For other
environments use the same script (the slug then names that environment, for example `wine-11.0-...`), or
`scripts/run_all.sh --baseline auto` on Linux + Wine.

## Triage

```
python3 scripts/compare_results.py --triage results/baselines/<slug>
python3 scripts/compare_results.py --baseline results/baselines/<slug>       # vs the closest committed baseline
python3 scripts/compare_results.py results/baselines/<win-slug> results/baselines/<other-slug>
```

- `--triage` lists every FAIL / CRASH / TIMEOUT / non-OK exe with its failure lines (file:line, the
  expression and the values), every XFAIL, and applies the "expected on Windows" rows of
  `results/baselines/expect_windows.json` (cases that have never run anywhere but Windows).
- The two-run diff tags functional differences REGRESSION / FIXED / XFAIL-FLIP / CHANGED, lists
  CAPABILITY differences (SKIP on one side, run on the other: a feature exposure difference, not a defect
  by itself), MISSING cases and PERF deltas (never a failure).

How to judge a result on **real Windows**:

| result | meaning on Windows | action |
|---|---|---|
| FAIL / CRASH | either the test asserts something Windows does not guarantee, or a contract was pinned wrongly from the decompilation | real Windows wins: report it with the failure lines; the test or its expectation gets fixed and cited |
| XFAIL | only legitimate from `t_expect_fail` for a documented Windows behaviour; an XFAIL whose reason starts with "Wine" must never appear on Windows | report it (it means a Wine guard misfired) |
| SKIP | capability absent (CPU feature, XCR0 component, API export, OS build) | check the reason matches the machine (for example no AVX-512 SKIPs on an AVX-512 host) |
| TIMEOUT | the runner killed the exe | report the exe, the last RUN line in its .txt and the machine's load |
| PERF | measurement | never judged; keep for comparison |

How to judge a result on **another environment**: diff it against a Windows run on the same CPU. A
REGRESSION there is a candidate defect of that environment; read the case's source comment for the
contract it checks (SDM/APM section or the Windows API decomp it cites). Separate OS-API reporting
problems (xstate_windows, context_xstate) from execution problems (ymm/zmm_registers, *_arithmetic):
the guidance asks for them as separate results.

## What to report back

1. The baseline zip(s), or the `results\baselines\<slug>\` directory, committed to this repository with
   `git add results/baselines/<slug>` (the rest of `results/` is ignored).
2. The `--triage` output and, for each FAIL / CRASH / TIMEOUT / unexpected XFAIL: exe, arch, case, the
   failure lines, and whether it reproduces on a second run (`-Only <exe>`).
3. The machine: CPU model, BIOS AVX-512 / hybrid settings, hypervisor, OS build + UBR, VBS/HVCI state.
4. For other environments: the diff against the Windows run on the same CPU, with REGRESSION rows first.

## Known expectations

- Hosts without AVX-512 SKIP about 225 cases per arch (zmm_registers, simd_memory, alignment,
  simd_bench, memory_bench, exception_xstate ZMM frames). Hosts without AMX SKIP all of amx_xstate;
  without APX, apx_xstate runs only its #UD probes; without CET shadow stacks enforced, cet_xstate
  records the policy and SKIPs.
- x86 exes run under WOW64: cxx_exceptions SKIPs on x86 by design (GCC i686 C++ does not use SEH), and
  amx/apx/cet are x64-only contracts.
- The committed `wine-*` baseline is a plumbing reference only; its XFAIL list is the list of Windows
  contracts that have never met real Windows.
