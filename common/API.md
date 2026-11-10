# Suite API (common/)

Every exe is one file `tests/<name>.c` that does `#include "t.h"` and nothing else from common/.
The Makefile links it with every `common/*.c` for x64 and x86. Start a new exe with
`tools/new_exe.sh <name> "<purpose>"`. This API is shared by about 22 exes: extend it only by
adding things. Do not change existing signatures or semantics.

## 1. Cases, results, exit codes

```c
TEST(name)        { ... }   // runs by default, in source order
TEST_MANUAL(name) { ... }   // only with --include-manual (opt-in stress / demo cases)
CHILD(name)       { return code; }  // entry point for  <exe> --child=name [args]; argv[0] = name
```

Each case gets one result. If more than one applies, the highest in this list wins:

| result | when |
|---|---|
| CRASH | an unhandled exception in the case. The harness recovers it and later cases still run. A crash on another thread, or a hang past `--case-timeout`, ends the run instead, after it writes this case and the summary. |
| FAIL  | a `CHECK*` failed, or `REQUIRE*` aborted the case |
| XFAIL | `XFAIL(...)` was called, or a check failed after `t_expect_fail(...)` |
| SKIP  | `SKIP(...)` was called and nothing failed. Use it only when the capability is absent. |
| PERF  | `t_perf*()` recorded a value and nothing failed |
| PASS  | none of the above |

Only the external runner assigns TIMEOUT. The exe exits with 0 when there is no FAIL or CRASH (SKIP,
XFAIL and PERF are fine), 1 when there is at least one FAIL or CRASH, and 2 on a harness error: a bad
argument, a `--json` file it cannot open, or no case matching the filter.

**Policy**: when a feature is absent, the case ends in SKIP. When it is advertised but broken, the case
ends in FAIL. Hard-check only what the SDM/APM or a documented Windows contract guarantees. Record facts
that vary between machines with `t_note` / `t_info`: frequencies, optional features, and API
discrepancies that are not contract violations. Never reference a specific OS implementation in test
logic. `t_is_wine()` exists only to choose an XFAIL/SKIP reason for a known Wine gap. It must never
weaken a check on real Windows.

## 2. Checks and outcomes

```c
CHECK(c); CHECKF(c, fmt, ...);            // non-fatal; thread-safe (worker threads OK)
CHECK_EQ(a, b); CHECK_NE(a, b);           // integers, printed hex+dec
CHECK_MEM(got, want, n);                  // bytes; prints hexdump window + ^^ markers
CHECK_REG("xmm7 after Sleep", got, want, n); // register image; names first bad qword, decodes pattern source
REQUIRE(c); REQUIREF(c, fmt, ...);        // fail + end case (case thread)
SKIP(fmt, ...);  XFAIL(fmt, ...);         // end case (noreturn on the case thread)
t_expect_fail(fmt, ...);                  // later failures => XFAIL; none => PASS + "XPASS" note
t_note(fmt, ...);  t_info(key, fmt, ...); // observations (JSON "notes" / "info"), never affect result
t_perf(metric, value, unit);  t_perf_stats(metric, &stats, unit);   // PERF (value = median)
t_log(fmt, ...);                          // only with --verbose, not in JSON
```

`SKIP`, `XFAIL` and `REQUIRE` jump straight to the end of the case, so a pending `free` or
`CloseHandle` is skipped. On a worker thread they record the result and then call `ExitThread`. Join
your threads before the case returns. A worker that is still running when the next case starts will
attribute its checks to that case.

Every case starts from the same FP environment: x87 FNINIT plus the initial control word, MXCSR at
its initial value with the sticky flags cleared, and VZEROUPPER when AVX is usable. Changes made by
one case never leak into the next.

## 3. Child processes (deliberate faults, timeouts)

```c
static t_child_result r;                       // large (16 KB output buffer): make it static
t_child_run("name", "extra args", 5000, &r);   // runs <self> --child=name extra args
t_run_self("--filter=x --json=f", 5000, &r);   // arbitrary command tail
r.started, r.timed_out (killed), r.exit_code, r.exc_code (0 if no exception), r.exc_addr, r.ms, r.out
t_child_find(&r, "PREFIX=", val, cap);         // read a line the child printed
```

- An unhandled exception in a child prints `T_CHILD_EXCEPTION code=0x... addr=0x...` and the child
  exits with the exception code. The harness's top-level filter is reached only after the child's own
  VEH/SEH handlers decline the exception.
- Children run in a kill-on-close job object, so grandchildren die with them. Output is captured from
  stdout and stderr.
- Children never write JSON. Have them print `KEY=value` lines and parse those in the parent.
- An unknown child name exits with `T_CHILD_UNKNOWN`.
- Run in a child: anything that executes an illegal or privileged instruction (`XSETBV`, a
  `#UD` probe, `XRSTOR` with a bad header), anything that faults on purpose, and anything that
  could hang.
- Tests may install their own VEH/SEH for expected exceptions inside the case. The harness only
  catches what nobody else handled.

## 4. CPU features (`cpuid.h`, `feature_detect.h`)

```c
cpu_has(F_AVX2)          // usable: CPUID bit AND OS state (OSXSAVE + XCR0 bits) -> gate every ISA use
cpu_hw(F_AVX2)           // raw CPUID bit only (report "advertised but not enabled" separately)
const t_cpu_info *c = cpu();  // vendor, brand, family/model, osxsave, xcr0, xinuse, xcr0_supported,
                              // xss_supported, xsave sizes, comp[i] {size, offset, supervisor, align64, xfd},
                              // raw leaves l1, l7_0, l7_1, ld_0, ld_1, e1, e7, e8, avx10_version
cpuid_ex(leaf, sub, &regs);  cpuid_enumerate(entries, max);   // every leaf/subleaf by the documented rules
cpu_feature_name(f), cpu_feature_gate_name(f), cpu_feature_lookup("avx2"), cpu_features_string(buf, cap, usable_only)
cpu_ymm_state_enabled(), cpu_zmm_state_enabled()
xstate_comp_name(i), xsave_std_size(mask), xsave_compacted_offset(mask, i), xsave_compacted_size(mask)
T_XS_X87 ... T_XS_APX, T_XS_YMM_STATE (0x6), T_XS_ZMM_STATE (0xE6), T_XS_AMX_STATE (0x60000)
```

Gates:

| gate | features | requires |
|---|---|---|
| YMM | AVX, AVX2, FMA, F16C, VAES, VPCLMULQDQ, AVX-VNNI, ... | OSXSAVE and `XCR0[2:1]=11b` |
| ZMM | AVX-512* | OSXSAVE and `XCR0 & 0xE6 == 0xE6` |
| AMX | AMX* | OSXSAVE and `XCR0[18:17]` |
| OSXSAVE | XSAVE, XSAVEOPT, XSAVEC, XGETBV1 | OSXSAVE |
| PRIV | XSAVES and other privileged features | never usable from user mode (`cpu_has` returns 0) |

Two special cases:
- PKU depends on OSPKE, not on XCR0.
- BMI1/BMI2 (VEX-encoded GPR instructions) and the legacy SSE forms of AES, PCLMULQDQ, SHA and GFNI
  need no XCR0 bits.

XGETBV runs only when OSXSAVE is set, and XGETBV(1) only when `F_XGETBV1` is present.

## 5. Register files and instruction helpers (`asm.h`)

C variables do not pin hardware registers. To test that values survive an operation, the load, the
operation and the store must all happen in one assembly routine:

```c
t_vregs in, out;                          // v[32][64] (64-byte aligned) + k[8]
t_vregs_pattern(&in, T_VK_YMM, tag);      // kinds: T_VK_XMM (16 B), T_VK_YMM (32 B), T_VK_ZMM (64 B + k0-7)
t_vregs_call(kind, &in, &out, fn, arg);   // load all, fn(arg), store all  (-1 if kind not usable)
t_vregs_spin(kind, &in, &out, &stop, n);  // load all, GPR-only PAUSE loop until *stop or n iterations, store
t_vregs_check(&out, &in, kind, first, last, "after X");   // CHECK_REG per register (+k for ZMM)
t_vregs_count(kind)   // x64: 16 XMM/YMM, 32 ZMM; x86: 8; 0 if not usable
t_x87_spin(in[8] t_f80, out, &stop, n);  t_mmx_spin(in[8] u64, out, &stop, n)   // ST(0)=in[0]; EMMS at end
```

- On x64, a callback `fn` may clobber volatile state under the ABI: XMM0-5, the upper YMM/ZMM bits,
  XMM16-31 and k0-7. Only the low 128 bits of XMM6-15 are guaranteed to survive. On x86 every XMM
  register is volatile.
- `*_spin` runs no code of its own between load and store, so preemption, `SuspendThread` and
  `Get/SetThreadContext` from another thread are what put the registers at risk.
  `stop` is a `volatile long`.
- The helpers preserve the caller's callee-saved registers, and on x64 they carry SEH unwind info.

Scalar helpers:

```c
t_get_mxcsr/t_set_mxcsr, t_get_x87_cw/t_set_x87_cw, t_get_x87_sw, t_get_x87_tw, t_fninit, t_fnclex, t_emms,
t_vzeroupper/t_vzeroall (AVX), t_xgetbv(i),
t_fxsave/t_fxrstor (512 B, 16-aligned), t_xsave/t_xsaveopt/t_xsavec/t_xrstor(area 64-aligned, rfbm)
```

On x64 these use FXSAVE64/XSAVE64/...; on x86, the plain forms.

**Baseline-ISA rule:** never compile with `-mavx*` or `-march`. Use `T_TARGET("avx2,fma")` on functions
that use intrinsics, and give those functions an ISA hint in their name: avx, fma, 512, ymm, zmm,
vaes, ... Call them only after a `cpu_has()` check. `make check-isa` lists any VEX/EVEX instruction
found outside such functions.

## 6. Timing (`timing.h`)

QPC is the time base for every PERF number. Never assume TSC ticks equal core cycles.

```c
t_qpc(), t_qpc_freq(), t_qpc_sec(t0,t1), t_qpc_ms(t0,t1), t_now_ms()
t_rdtsc() (F_TSC), t_rdtscp(&aux) (F_RDTSCP)
t_stats_compute(v, n, &s)            // n, min, max, mean, median, stddev, q1, q3 (quartiles; IQR = q3 - q1)
t_bench(fn, ctx, warmup, reps, min_sample_ms, &s)  // s = seconds per call; reps <= 256
t_pin_thread(cpu) -> prev mask (0 = failed); t_unpin_thread(prev); t_cpu_count()
T_SINK(x); t_sink_ptr(p)             // defeat dead-code elimination
```

Before you report a number with `t_perf_stats`, check the result of the computation you timed.

## 7. Patterns and diagnostics (`pattern.h`)

```c
t_rng r; t_rng_seed(&r, seed); t_rng_next(&r); t_rng_u32; t_rng_range(&r, n); t_rng_fill(&r, buf, n); t_mix64(x)
t_pattern_word(tag, reg, word); t_pattern_reg(out, bytes, tag, reg); t_pattern_regs(base, n, stride, width, tag)
t_pattern_explain(v, max_tag, &tag, &reg, &word)   // which thread/register did a leaked value come from
t_hex(buf, cap, p, n); t_hex_lanes(buf, cap, p, n, lane_bytes); t_dump_vec(label, p, n, verbose_only)
```

`t_pattern_word` sets byte 7 to `0x80|reg` and byte 6 to `0x40|word`. Bytes 0-5 hold
`hash(tag, reg, word)`, so a word is never zero. Every value is the same on every platform.

## 8. Environment and utilities

```c
const t_env *e = t_env_get();   // os_major/minor/build/ubr, os_product, process_arch, wow64, native_machine,
                                // wine, ncpu, affinity, active_groups, qpc_freq,
                                // api_xstate_present/api_xstate_features (GetEnabledXStateFeatures), pf[64]
t_is_wine(), t_wine_version(), t_exe_name(), t_case_name(), t_self_path(), t_verbose(), t_on_case_thread()
T_PROC("kernel32.dll", InitializeContext2)   // typed run-time lookup; NULL if missing -> SKIP
t_guarded_alloc(n, &base) / t_guarded_free(base)   // n bytes ending at a PAGE_NOACCESS page
t_tmpfile(buf, cap, "leaf")
```

Resolve every XSTATE or Windows 11-only API through `T_PROC`, so that the exe still loads on systems
that lack it.

`GetEnabledXStateFeatures` (kernelbase, Windows 10 19041) returns `RtlGetEnabledExtendedFeatures(~0)`,
or 3 when that call returns 0. That value is `(XState.EnabledFeatures | XState.EnabledSupervisorFeatures)`,
read from KUSER_SHARED_DATA, so it may contain supervisor bits that are not in XCR0.

## 9. CLI and output

```
<exe> [--json=<file>] [--filter=a,b] [--list] [--verbose|-v] [--case-timeout=<sec>(default 120, 0=off)]
      [--include-manual] [--child=<name> args...]
```

Human-readable output on stdout:

```
ENV   ...
RUN   case
  FAIL file:line: expr (msg)
  NOTE / INFO / PERF / SKIP / XFAIL / CRASH ...
PASS  case (1.2 ms)
...
SUMMARY exe=.. arch=.. total= pass= fail= crash= skip= xfail= perf= checks= checks_failed= exit=
```

The `--json` file is in JSON-lines format:

1. `{"type":"env", ...}`: compiler, version, OS, wine, CPU, features_hw / features_usable, xcr0,
   xinuse, xsave sizes, api_xstate_features, pf_present, ncpu, affinity, ...
2. One `{"type":"case", "exe", "arch", "case", "result", "ms", "checks", "failed", "reason", "failures":[...],
   "notes":[...], "info":{...}, "perf":[{metric, value, unit, n, min, max, mean, stddev, q1, q3}], "exception"?}`
   line per case.
3. `{"type":"summary", total, pass, fail, crash, skip, xfail, perf, checks, checks_failed, ms, exit_code}`.

Filters match case names by substring. Make case names specific enough that a filter does not
accidentally select other cases.
