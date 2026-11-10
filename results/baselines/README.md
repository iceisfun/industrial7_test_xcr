# Committed result baselines

Each directory here is one raw run of the whole suite, both arches, as written by
`scripts/run_all.ps1 -Baseline auto` (Windows) or `scripts/run_all.sh --baseline auto` (Wine):

```
results/baselines/<os>-<cpu>/
  host.json           OS build/UBR/DisplayVersion/edition, CPU, hypervisor, git revision, runner, note
  x64/run.json        per-exe status, exit code, duration, exe sha256, env record
  x64/<exe>.json      JSON lines: env, one record per case, summary   (the data compare_results.py reads)
  x64/<exe>.txt       console output (stdout + stderr)
  x86/...             the 32-bit exes (WOW64 on 64-bit Windows)
```

The name is `<os>-<cpu>`: for example `win10-22h2-19045-intel-core-i7-8700`,
`win11-24h2-26100-amd-ryzen-9-7950x` or `wine-11.0-...`. `compare_results.py --slug <run_dir>`
prints the name for any run.

Only a run on real Windows is a reference for the documented contracts. A `wine-*` baseline is a
plumbing reference: it shows what the suite does under Wine and catches regressions in the suite
itself, and its XFAIL list is the list of contracts that have never met real Windows.

## Status

| baseline | OS | CPU | notes |
|---|---|---|---|
| `wine-*` | Wine 11.0 (reports 10.0.19045) | host CPU, AVX2/FMA/AES/PCLMUL/SHA, no AVX-512 | plumbing reference only |
| Windows 10 22H2 x64 + WOW64 | pending | | needs a Windows host (see below) |
| Windows 11 24H2 x64 + WOW64 | pending | | needs a Windows host |
| AVX-512 machine | pending | | any Windows 10/11 host with AVX-512 enabled |

No Windows baseline has been committed yet: every decomp-derived contract (InitializeContext /
LocateXStateFeature / CopyContext errors, #MF NumberParameters, privileged-opcode status codes, fiber
MXCSR switching, VCH, special user APCs, RtlCaptureContext2, GetThreadEnabledXStateFeatures,
Wow64Get/SetThreadContext) has only met Wine, where it is XFAIL or SKIP.

## Recording a Windows baseline

On Linux, `scripts/pack_windows.sh` builds and packs `build/x64`, `build/x86`, `scripts/` and
`corpus/` into `build/xcr-windows-<version>.zip`. On the Windows machine, unzip it and run

```
scripts\run_baseline.cmd "bare metal, VBS off"
rem or: powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Baseline auto [-Note "..."]
```

That writes `results\baselines\<slug>\` (and `results\baseline-<slug>.zip` to carry back). Before committing it, triage it on any host with Python 3:

```
python3 scripts/compare_results.py --triage results/baselines/<slug>
```

- Every `FAIL` / `CRASH` / `TIMEOUT` is either a test bug (the check asserts something Windows does
  not guarantee: fix the test, re-run, re-record) or a contract the test pinned wrongly from the
  decompilation (fix the expectation and cite the evidence). Real Windows is the authority.
- Every `XFAIL` outside Wine is flagged: an XFAIL is only legitimate on Windows when it comes from
  `t_expect_fail` for a documented Windows behaviour, never from a `t_is_wine()` guard.
- `PERF` values are recorded, never judged.

Commit the directory with `git add results/baselines/<slug>` (the rest of `results/` is ignored).

## Comparing a run with a baseline

```
python3 scripts/compare_results.py --list-baselines
python3 scripts/compare_results.py --baseline <run_dir>               # closest baseline, chosen automatically
python3 scripts/compare_results.py --baseline <run_dir> --use=<slug>  # a named one
python3 scripts/compare_results.py <baseline_dir> <run_dir>           # any two runs
```

The closest baseline is chosen by Wine-or-not, OS family and build, CPU brand, usable features,
XCR0 and the arches present. Functional mismatches are tagged `REGRESSION` (was PASS/XFAIL, now
FAIL/CRASH/TIMEOUT), `FIXED` or `CHANGED`; SKIP-vs-run differences are listed separately as
capability differences. The exit status is 1 on any functional mismatch.
