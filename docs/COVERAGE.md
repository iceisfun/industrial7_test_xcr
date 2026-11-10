# Coverage map: docs/GUIDANCE.md -> exes and cases

Produced by a completeness review at commit 7080d1e (2026-11-09). Every bullet of GUIDANCE sections 2, 3, 4,
5, 6, 7 and 10 is mapped to the exe and cases that implement it. Gaps are listed at the end of each section
and summarized in "Open work". Case names are the `--list` names. `x64` and `x86` mean the two builds.

Status legend: **covered** = hard checks exist and ran on the validation host; **covered (unvalidated)** =
checks exist but only SKIP on the validation host, so the code path has never executed; **INFO** = recorded
but not hard-checked; **thin** = partly covered; **MISSING** = no test.

## Validation run (headless Wine, plumbing only)

Host: Intel i7-14700KF (Raptor Lake, hybrid). AVX2/FMA/F16C/AES/PCLMUL/SHA/GFNI/VAES/VPCLMULQDQ (VEX)/AVX-VNNI
are present. AVX-512, AMX, APX, SHA512, SSE4A/XOP/FMA4/TBM/CLZERO and PREFETCHI are absent. XCR0 = 0x207.
Wine 11.0 (proton-ge build) reports Windows 10 build 19045. The run used `scripts/run_all.sh --arch all
--timeout 600`, one wine tree at a time.

All 23 exes x 2 archs exited 0. There were no FAIL, CRASH, TIMEOUT or HARNESS_ERROR results, and no hangs.

| arch | cases | PASS | PERF | SKIP | XFAIL | AVX-512-only SKIPs |
|---|---|---|---|---|---|---|
| x64 | 2515 | 1904 | 243 | 290 | 78 | 214 |
| x86 | 2506 | 1855 | 241 | 318 | 92 | 213 |

Longest exes: thread_xstate takes 40 s, xstate_stress 19 s, simd_jpeg 15-18 s and ymm_registers 9-12 s. The
others take less than 8 s.

Every XFAIL is a Wine gap, and each one records its reason. The main groups are:
- **context API validation.** Locate/Get/SetXStateFeaturesMask last-error handling, CopyContext not
  validating flags or copying MxCsr, CONTEXT_EX not validated, and reserved MXCSR bits in a CONTEXT
  killing the thread.
- **#MF/#XM/#DE status mapping.** NumberParameters=0, DE reported as underflow, and every #DE reported as
  divide-by-zero.
- **privileged opcodes.** These are reported as AV instead of PRIVILEGED_INSTRUCTION.
- **vectored continue handlers.** They are never called.
- **fibers.** SwitchToFiber does not switch MXCSR or the x87 CW. CreateFiberEx and ConvertThreadToFiberEx
  do not validate flags.
- **special user APCs.** They are not delivered.
- **stack overflow handling.** STATUS_STACK_OVERFLOW arrives without the guaranteed stack.
- **x86 capture during system calls.** On x86 only, xstate_stress has 21 XFAILs because GetThreadContext
  of a thread inside a system call returns an all-zero FP area.

Exe version stamp defect: the env JSON reported `version=8b988f1-dirty` while HEAD was 7080d1e. T_VERSION
is compiled only into `build/obj/*/t.o`, and the Makefile does not rebuild that object when HEAD changes.
Results from different builds therefore cannot be told apart (§9 asks for the exe version).

## 2. Processor feature consistency

| Guidance bullet | Implemented by | Status |
|---|---|---|
| CPUID leaves 0, 1, 7 and extended leaves, enumerating subleaves | cpu_features: leaf0_vendor_and_max, leaf1_*, leaf7_subleaves, leaf4_cache_parameters, leaf0b_1f_topology, leaf2_descriptors, leaf6_thermal_power_info, leaf14_15_16_1a_info, ext_*, cpuid_flat_leaves_ignore_ecx, leaf_beyond_max_basic_info, hypervisor_leaves_info; feature_faults cpuid_above_max_*, cpuid_leaf7_invalid_subleaf_zero | covered. Leaves 0x10/0x12/0x17/0x18/0x1B subleaves are not walked (only relevant as INFO). |
| Leaf 0xD: components, sizes, offsets, XSAVEOPT/XSAVEC/XGETBV1/XSAVES, compacted format | cpu_features leafd_*; xcr0_probe cpuid_d_* , xsave_size_*, cpuid_d1_capability_bits, xsavec_xcomp_bv_equals_xcr0; xsave_roundtrip cpuid_component_sizes_offsets, cpuid_compacted_layout, xsavec_* | covered (AVX-512/AMX/APX components unvalidated: absent on host) |
| AVX hardware separate from OSXSAVE/XCR0; AVX needs XCR0[2:1]; AVX-512 needs XCR0 1,2,5,6,7 + CPUID | xcr0_probe avx_eligibility_rule, avx512_eligibility_rule, fma_f16c_eligibility_rule, avx*_executes_when_eligible; cpu_features avx_state_gating, feature_table_and_gates; simd_dispatch detect_*_gate_rule, synth_* (synthetic CPUID/XCR0 matrices), misdetect_*; zmm_registers zmm_gate_* | covered (logic); the AVX-512 execute side is unvalidated |
| GetEnabledXStateFeatures vs native XCR0 (API semantics); GetXStateFeaturesMask on an initialized CONTEXT; IsProcessorFeaturePresent vs execution | xstate_windows api_enabled_features_*, rtl_enabled_features_masking, getmask_*; xcr0_probe api_*; cpu_features pf* (each PF claim is checked by executing the instruction), pf_out_of_range_and_stable, pf_non_x86_constants_zero; simd_dispatch detect_vs_* | covered |
| CPUs/VMs with AVX/AVX-512 disabled or hidden; XGETBV only when OSXSAVE; XCR0 vs IA32_XSS | xcr0_probe xgetbv0_ud_without_osxsave, avx_ud_when_ymm_state_disabled, avx512_ud_when_zmm_state_disabled, cpuid_d1_xss_mask_shape, xcr0_xss_masks_disjoint, xcr0_no_supervisor_components; ymm_registers ymm_vex_faults_when_avx_disabled; feature_faults feat_* (absent ISA -> #UD in a child), feat_xgetbv_without_osxsave; simd_dispatch synth_no_osxsave_never_executes_xgetbv | **covered (unvalidated)**: every disabled or hidden path SKIPs on the host. Only the synthetic simd_dispatch matrices run. |
| GetEnabledXStateFeatures, InitializeContext, GetXStateFeaturesMask, LocateXStateFeature, SetXStateFeaturesMask against documented contracts | xstate_windows (115 cases: init_*, init2_*, ctxex_*, locate_*, getmask_*, setmask_*, copy_*, rtl_*); context_xstate init_ctx_*, locate_*, set_mask_*, copy_context_*; zmm_registers zmm_api_* | covered, with decomp-pinned contracts cited in the sources |
| Working AVX2 execution with incomplete context APIs reported as SEPARATE results | xstate_windows exec_avx_matches_api, exec_xinuse_vs_context_mask; context_xstate ctx_api_reports_usable_features; simd_dispatch detect_vs_context_xstate_api | covered |

Section 2 gaps: Windows 11 GetThreadEnabledXStateFeatures and EnableProcessOptionalXStateFeatures are
unvalidated (Wine does not export them). AMX/APX (leaf 7.1 / XCR0 bits 17-19) are only reported, never
executed.

## 3. Userland tests

### Register state and scheduling

| Bullet | Implemented by | Status |
|---|---|---|
| XMM/YMM/ZMM patterns verified after Sleep(0), SwitchToThread, waits, mutex, events, migration | xmm_registers xmm_abi_*, xmm_preempt_*, xmm_migration_*; ymm_registers ymm_call_*, ymm_preempt_*; zmm_registers zmm_preserve_*; thread_xstate iso_* (sleep0/switch/event_ring/mutex/semaphore/apc/condvar/srw/critsec/waitaddr/pipe/waitable_timer/wfmo/signal_and_wait/migrate/suspend); xstate_stress iso_*, ctl_*, preempt_*, tmr_* | covered (ZMM unvalidated) |
| 2/16/64/256/1024 threads where resources permit | thread_xstate iso_*_n2 .. _n1024 (SKIP if CreateThread fails); ymm_preempt_spin_1024_threads; zmm_preserve_1024_threads; simd_jpeg mt_decode_1024_threads_17x13 | covered |
| Per-thread pattern to detect leakage | t_pattern_word tags in thread_xstate and xstate_stress (leak decoder: selftest_leak_decoder, selftest_mutation_*_detected) | covered |
| State survives blocking syscalls | ymm_call_file_io, ymm_call_virtualalloc, xmm_abi_syscall_mix, zmm_preserve_cb_file_io, xstate_stress iso_*_blocking_syscalls | covered |
| After creating/terminating many threads | thread_xstate churn_create_join_1000_*, init_new_thread_after_terminate_spinning; ymm_isolation_churn_*; zmm_preserve_thread_churn; xstate_stress churn_new_threads_under_stress | covered |
| Nested calls, callbacks, DLL calls, thread-pool callbacks | xmm_abi_nested_calls_depth64, xmm_abi_qsort_callback, xmm_abi_direct_dll_exports, xmm_abi_threadpool_callback; ymm_call_window_message_callback; zmm_preserve_cb_load_library/nested/threadpool; thread_xstate tp_* | covered |
| Fibers with and without FIBER_FLAG_FLOAT_SWITCH vs native | fiber_xstate (81 cases: *_float_switch vs *_noflag, xmm6_15, ymm/zmm low128, FLS, migration, 1024 ring) | covered. 26 x64 / 22 x86 XFAILs under Wine. |
| x64 ABI: XMM6-15 low 128 nonvolatile, upper YMM volatile | xmm_abi_* (hard-check XMM6-15), ymm_call_abi_legal_clobber, exception_xstate raise_preserves_abi_nonvolatile_registers, simd_dispatch impls_preserve_nonvolatile_xmm | covered |
| x86 and x64 separately; WOW64 context tests | Both builds run every exe. context_xstate wow64_own_threads (x86) and wow64_cross_process_context (x64 drives a WOW64 child through Wow64*ThreadContext); xcr0_probe xcr0_same_in_other_architecture_process | **thin**: wow64_cross_process_context SKIPs under Wine ("Wow64*ThreadContext not exported") and has never run. The WOW64 XSTATE limits (YMM of a WOW64 thread seen from x64) are not tested. |
| Assembly helpers, not C variables | common/asm_*.c t_vregs_*; every register case uses asm | covered |

### Windows XSTATE context

| Bullet | Implemented by | Status |
|---|---|---|
| Initialize XSTATE-aware CONTEXT via supported APIs; size negotiation and alignment | xstate_windows init_size_*, init_exact_size_guarded_*, init2_*; context_xstate init_ctx_*; alignment initialize_context_buffer_alignment | covered |
| GetThreadContext on suspended worker | context_xstate getctx_*, xstate_getctx_*; xmm_getthreadcontext_reports_spinner_xmm; thread_xstate iso_suspend_getcontext_*; xstate_stress ctl_getctx_* (oracle) | covered |
| Modify with SetThreadContext, resume, inspect | context_xstate setctx_*, xstate_setctx_*; xmm_setthreadcontext_changes_spinner_xmm; ymm_context_set_xstate_upper_spinning; zmm_context_set_xstate_avx512 | covered for **spinning** (user-mode) targets. **MISSING** for a target blocked inside a kernel wait or syscall (see gaps). |
| CopyContext and compare XSTATE | xstate_windows copy_*; context_xstate copy_context_*, xstate_copycontext_from_thread | covered |
| LocateXStateFeature for enabled and unsupported features | xstate_windows locate_*; context_xstate locate_*; zmm_api_locate_* | covered |
| Active masks, initialized vs modified components | xstate_windows exec_xinuse_vs_context_mask, getmask_fresh_xstate_context; context_xstate xstate_getctx_ymm_init_state; xcr0_probe xgetbv1_* | covered |
| Invalid masks and undersized buffers return documented errors, no corruption | xstate_windows init_size_short_by_one, init_invalid_flags_*, setmask_unsupported_bits_dropped, thread_bogus_context_ex_*; context_xstate err_* | covered |
| x87 CW and MXCSR survive context operations | context_xstate setctx_mxcsr_observed, setctx_x87_cw_observed, getctx_x87_stack_and_cw; fp_environment ctx_* | covered |
| Debug-register requests separate from FP/vector | context_xstate getctx_debug_registers_separate, setctx_debug_registers_zero; xstate_windows thread_get_plain_and_debug_only | covered |
| Negative parameter validation tests | xstate_windows *_invalid_*, *_error_code; context_xstate err_* | covered |

### Exceptions

| Bullet | Implemented by | Status |
|---|---|---|
| Integer exceptions with SIMD patterns live | exception_xstate veh_div0_record_context_{xmm,ymm,zmm}, veh_idiv_overflow_intmin; xmm_exc_div0_continue; ymm_exception_div0_continue_restores | covered |
| SEH and vectored handlers | exception_xstate veh_*, seh_*, vch_* | covered (VCH is XFAIL under Wine) |
| Guard-page and access violations | exception_xstate veh_guard_page_*, veh_av_*; alignment boundary_guard_page_one_shot | covered |
| Unsupported instructions under controlled handling | feature_faults ud_*, feat_* (child processes); exception_xstate child_ud2_* | covered |
| FP control/status after exceptions; masked and unmasked recovery | exception_xstate fp_sse_*_unmasked, fp_x87_*_unmasked, veh_handler_live_fp_environment; fp_environment sse_unmasked_*, x87_unmasked_* | covered |
| RaiseException, RtlCaptureContext | exception_xstate raise_*, cap_*; context_xstate rtlcapture_*; simd_bench ctx_rtlcapturecontext* | covered (RtlCaptureContext2 unvalidated: not exported by Wine) |
| setjmp/longjmp and **C++ exceptions** (ABI-required preservation only) | exception_xstate sj_longjmp_*; fiber_xstate fiber_seh_frame_handler_and_longjmp_child | setjmp covered. **C++ exceptions MISSING**: the suite is C only, with no throw/catch through the OS unwinder (RtlUnwindEx with UWOP_SAVE_XMM128). |
| Compare reported context against an instruction-level reference helper | exception_xstate ref_nop_roundtrip_*, cap_rtlcapturecontext_reference; xstate_stress capture oracle | covered |
| Faulting tests in CHILD processes | t_child_run is used by feature_faults, exception_xstate child_*, alignment, zmm/ymm fault children | covered |

## 4. SIMD math correctness

| Category | Implemented by | Status |
|---|---|---|
| Integer add/sub/mul/saturating | sse_arithmetic sse2_int_*; avx_arithmetic avx2_int_*; zmm_int_avx512f_semantics | covered |
| FP add/sub/mul/div/sqrt | sse_arithmetic sse_fp32_*, sse2_fp64_*; avx_arithmetic avx_fp_packed_vs_scalar_sse; ymm_fp_arith_upper_lane_exact | covered |
| Permutations | sse2_pshufd_*, ssse3_pshufb, sse41_blend/insertps/extract sweeps; avx_vperm*, avx2_vperm*; ymm_vperm*; zmm_perm_* | covered |
| Comparisons (ordered/unordered) | sse_fp32_cmpps_cmpss_predicates, sse_comiss_ucomiss_flags, avx_cmp_32_predicates_and_invalid_flag, zmm_cmp_* | covered |
| Bit ops, shifts, popcnt | sse2_int_logic, sse2_shift_*, gpr_popcnt/lzcnt/tzcnt, avx2_variable_shifts_*, bmi1/bmi2/lzcnt, zmm_ext_vpopcnt | covered |
| FMA rounding/exceptional | avx_arithmetic fma_all_forms_vs_exact_model, fma_rounding_modes_and_ftz_daz, fma_single_rounding_and_special_operands, fma_unmasked_simd_exception_invalid | covered |
| Conversions incl. packing | sse_arithmetic sse2_cvt*, sse_cvt*, sse2_pack_saturate; avx_conversions_all_rounding_modes, f16c_*; zmm_cvt_unsigned_dword, zmm_int_vpmov_narrowing | covered |
| SSE4.2 string compares, CRC32C | sse42_pcmp{e,i}str{i,m}_imm_sweep, sse42_crc32_vs_software; simd_crypto crc*; simd_dispatch vectors_crc32c_known_values | covered |
| Memory aligned/unaligned/masked/streaming | alignment (128 cases), simd_memory (149 cases), xmm_load_store_*, ymm_roundtrip_* | covered |
| Special inputs (zeros, -0, inf, NaN, denormal, min/max int, rounding boundaries, saturation) | special-value tables in sse_arithmetic_softfp.h / avx_arithmetic_core.h; avx_nan_propagation_rules; avx_daz_*; sse_fp32_known_answers | covered |
| Exact bits vs tolerance (rcp/rsqrt), original bits in diagnostics | sse_rcpps_rsqrtps_accuracy, avx_rcp_rsqrt_error_bound, zmm_fp_rcp14_rsqrt14_accuracy | covered |
| AVX -> SSE routine -> AVX transition | avx_legacy_sse_transition_preserves_upper; simd_bench transition_*; simd_dispatch sse_avx_transitions_between_tiers; ymm_mix_vex256_legacy_chain | covered |

Section 4 gaps (lower priority by the guidance's own ordering): AVX-512 math is entirely unvalidated on
this host. AVX-512 FP16, BF16, AVX-IFMA, AVX-VNNI-INT8, AVX-NE-CONVERT and CMPccXADD are only
exec-probed (cpu_features exec_*), with no reference vectors. AVX-VNNI is present on the host but has only
the exec probe plus zmm_ext_vnni (EVEX).

## 5. Benchmarks (PERF)

| Table row | simd_bench (unless noted) | Status |
|---|---|---|
| Array add GB/s scalar/SSE2/AVX2/AVX-512 | arradd_{f32,i32}_{scalar,sse,sse2,avx,avx2,avx512} | covered (AVX-512 rows unvalidated) |
| Int multiply Mops/s | imul_* | covered |
| FP ops/s scalar/SSE/AVX/FMA | fpt_* (throughput), fplat_* (latency), fp_denormal_penalty | covered |
| Dot product | dot_{scalar,sse,avx,fma,avx512}_f32 | covered |
| Matmul GFLOPS | matmul_{scalar,sse2,avx2_fma,avx512} | covered |
| memcpy/memset CRT/SSE2/AVX2 GB/s | memcpy_*, memset_*, *_kernels_correctness; simd_memory nt_copy_bandwidth_perf | covered |
| CRC32C software/SSE4.2 | crc32c_sw_*, crc32c_sse42_*; simd_crypto perf_crc32c_table_vs_pclmul_vs_sse42 | covered |
| AES software/AES-NI | aes_sw_*, aesni_*, vaes_*; simd_crypto perf_aes128_ecb_software_vs_aesni | covered |
| JPEG decode MPix/s | simd_jpeg perf_decode_4k_*, perf_decode_mt_scaling, perf_kernel_* | covered |
| RGB/YUV conversion | yuv_* | covered |
| Thread state stress iterations/s | thread_xstate iso_* (PERF rows), xstate_stress perf_stress_rates_* | covered |
| Context capture normal/XSTATE ops/s | ctx_*; context_xstate perf_getctx_std/xstate, perf_rtlcapturecontext; exception_xstate perf_exception_roundtrip | covered |
| QPC, optional RDTSC/RDTSCP, warm-up, median + variability, DCE defeat, cache-resident vs bandwidth, pinning, env/affinity/frequency record, verify before accepting | bench_qpc_contract, bench_tsc_estimate, bench_affinity_contract, bench_stats_selftest, bench_env_report, known-answer cases before each family | covered |

Section 5 gaps: §8 lists `benchmarks/{simd_bench,memory_bench,context_bench,jpeg_bench}.c`, but no
`benchmarks/` directory exists. Everything lives in tests/simd_bench.c and the PERF cases of other exes,
and that mapping is undocumented. No script turns PERF records into the §5 table (compare_results.py only
diffs two runs).

## 6. Real-world decoder tests

| Bullet | Implemented by | Status |
|---|---|---|
| libjpeg-turbo-style scalar/SSE2/AVX2 | simd_jpeg (own baseline+progressive decoder; islow/float IDCT, fancy upsampling, YCC kernels in scalar/SSE2/AVX2) | covered (in-tree decoder, not libjpeg-turbo itself) |
| **libpng / zlib** | none (simd_dispatch has only adler32/crc32 kernels) | **MISSING** |
| **stb_image as independent reference** | none. References are djpeg checksums in corpus/reference_vectors/simd_jpeg_refs.txt. | **MISSING** (djpeg is an independent reference, but there is no second in-process decoder) |
| YCbCr->RGBA scalar + SIMD | simd_jpeg ycc_rgba_exhaustive_*; simd_bench yuv_* | covered |
| Corpus: 8x8/16x16, 4K/8K, gray/RGB/subsampled, progressive/baseline, odd dims/strides, corrupt/truncated, repeated MT decode | corpus/jpeg (94 files incl. err_*); synth_4k_*, synth_8k_420_vs_djpeg, trunc_*, fuzz_*, mt_decode_*, idct_unaligned_and_stride | covered |
| Wrong pixels = FAIL; compare full-buffer checksums | corpus_* and synth_* vs djpeg hashes | covered |

## 7. Negative tests

| Bullet | Implemented by | Status |
|---|---|---|
| Unsupported instruction isolated and exception-handled; expected #UD | feature_faults ud_*, feat_*, repeat_*; zmm_fault_evex_ud_when_unusable | covered (absent-ISA #UD validated for AVX-512/AMX/FMA4/XOP etc.; for present ISAs it SKIPs) |
| Unsupported XSTATE feature bits to Windows APIs | xstate_windows setmask_unsupported_bits_dropped, locate_unsupported_features_null, init2_mask_unsupported_bits; zmm_api_locate_invalid_ids | covered |
| Invalid/inconsistent context flags | xstate_windows init_invalid_flags_*, copy_invalid_flags*, getmask_no_arch_flags*; context_xstate err_getctx_xstate_flag_without_chunk | covered |
| Unaligned buffers where alignment is required | alignment align_*_needs_*, init_ctx_alignment_sweep_guarded, context_misaligned_getset_thread_context, context_misaligned_xstate_chunk_getset, context_misaligned_rtlcapturecontext, context_misaligned_ntcontinue, context_initialize_misaligned_buffer_xstate_chunk, context_copycontext_misaligned_dest_xstate | covered. Pinned against the 19041 decomp (PspGet/SetContextThreadInternal -> RtlpReadExtendedContext, KiContinueEx, RtlCopyContext, RtlInitializeExtendedContext2): misaligned CONTEXT (16 x64 / 4 x86) or XState chunk (64, x64) -> FALSE / ERROR_NOACCESS (998), NtContinue -> STATUS_DATATYPE_MISALIGNMENT, CopyContext has no alignment check. Wine accepts misaligned buffers: XFAIL. |
| SIMD access at the end of a committed page | alignment boundary_*, page_end_exact_accesses_do_not_fault; xmm_page_edge_exact_width_access; simd_crypto crypto_m128_operand_ending_at_guard_page_no_fault | covered |
| Masked instructions do not access inactive lanes | alignment masked_*_page_end; simd_memory masked_*, resume_*; avx_masked_lanes_never_fault; zmm_fault_suppression_* | covered (AVX-512 unvalidated) |
| SIMD state of one thread never in another | thread_xstate, xstate_stress, ymm_isolation_*, zmm_isolation_fresh_thread_no_leak, exception_xstate thr_worker_exceptions_isolated_* | covered |
| New thread initial FP env (MXCSR 0x1F80, CW 0x27F) | fp_environment new_thread_*; thread_xstate init_new_thread_*; crt_beginthreadex_env | covered on both archs |
| MMX/x87 interactions, EMMS | feature_faults mmx_*, emms_*, x87_after_mmx_without_emms_overflows; fp_environment mmx_*; xsave_roundtrip fxsave_mmx_aliasing_and_emms | covered |

## 10. Further areas

| Bullet | Implemented by | Status |
|---|---|---|
| (1) Preservation across callbacks/transitions | See §3. This is the strongest area of the suite. | covered |
| (2) Windows 11 GetThreadEnabledXStateFeatures / EnableProcessOptionalXStateFeatures (AMX) | xstate_windows thread_enabled_features_win11, enable_optional_features_child, init2_amx_tiledata; xcr0_probe api_thread_enabled_xstate_features, amx_eligibility_rule; feature_faults feat_amx_tilerelease/tilezero | **thin and unvalidated**: there is no LDTILECFG/TILELOADD anywhere. AMX tile state is never loaded, preserved, captured by GetThreadContext or checked across threads. The XFD first-use path is untested. |
| (3) Hardware vs OS reporting disagreement, recorded and evaluated for impact | simd_dispatch detect_vs_*, misdetect_*; xstate_windows exec_*_matches_api; xcr0_probe api_*, kuser_shared_xstate_info | covered |

## Repository layout (§8) and scripts

| Guidance item | State |
|---|---|
| common/{cpuid,feature_detect,timing,report,test_helpers}.c | present under other names (t.c = report, util.c/pattern.c/child.c = helpers) |
| tests/*.c one per exe | present (22 + smoke_selftest) |
| **benchmarks/{simd_bench,memory_bench,context_bench,jpeg_bench}.c** | **MISSING** (simd_bench is in tests/) |
| corpus/jpeg, corpus/reference_vectors | present (reference_vectors holds only simd_jpeg_refs.txt; other vectors are embedded in sources) |
| scripts/run_all.ps1, compare_results.py | present (+ run_all.sh) |
| CMakeLists.txt | present (best effort, not built here) |

## Open work (most valuable first)

1. **Nothing has been validated on real Windows.** Every hard check derived from a decomp or the
   documentation has only ever run under Wine. Run scripts/run_all.ps1 on Windows 10 22H2 and Windows 11
   24H2 (x64 + WOW64), keep the baselines, and triage every check that differs.
2. **214 AVX-512 cases (x64) have never executed.** zmm_registers has 111 of them, simd_memory 24,
   simd_bench 16 and alignment 13. The disabled/hidden-feature #UD paths also never ran. Use an emulator
   (Intel SDE) or a lab host with AVX-512.
3. **AMX tile state** (TILECFG/TILEDATA, XFD, EnableProcessOptionalXStateFeatures) is not exercised.
4. **Context of a thread blocked in a syscall** is untested. Get/SetThreadContext have only been tested
   against user-mode spinners.
5. **C++ exceptions** through the OS unwinder (UWOP_SAVE_XMM128) are MISSING.
6. **WOW64 cross-architecture context** (Wow64Get/SetThreadContext) has never run.
7. **libpng/zlib decoder and an independent decoder reference** (§6) are MISSING.
8. **benchmarks/ split and §5 table script** are MISSING. The stale T_VERSION stamp also needs fixing.
9. **APX** (XCR0 bit 19, EGPR R16-R31) and **CET_U** (supervisor XSTATE, user shadow stack) are only
   reported, never tested.
10. **AVX-512 FP16/BF16 and the VEX-encoded VNNI/IFMA arithmetic** have no reference vectors.
