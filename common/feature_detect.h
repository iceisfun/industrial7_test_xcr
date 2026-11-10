/* feature_detect.h - complete CPU feature detection with OS-state gating.
 *
 *   cpu_hw(F_X)   raw CPUID bit (what the processor/hypervisor advertises)
 *   cpu_has(F_X)  usable from user mode right now = CPUID bit AND the OS enabled the needed state:
 *                 - XSAVE family / XGETBV: CPUID.1:ECX.OSXSAVE[27] (CR4.OSXSAVE)
 *                 - AVX-class (VEX/YMM):  OSXSAVE and XCR0[2:1] == 11b
 *                 - AVX-512 (EVEX):       OSXSAVE and XCR0[7:5,2:1] == 111b,11b
 *                 - AMX:                  OSXSAVE and XCR0[18:17] == 11b (Windows may still need
 *                                         EnableProcessOptionalXStateFeatures; tests handle that)
 *                 - APX:                  XCR0[19];  PKU: CPUID.7:ECX.OSPKE[4] (CR4.PKE)
 *                 - privileged (XSAVES etc): never usable at CPL3 -> cpu_has() = 0
 *                 VEX-encoded GPR instructions (BMI1/BMI2) and legacy SSE forms do not depend on XCR0.
 * XGETBV is only executed when OSXSAVE is set; XGETBV(1) only when CPUID.(0Dh,1):EAX[2].
 * Detection runs once (cpu_detect(), called by the harness at startup; safe to call again). */
#ifndef T_FEATURE_DETECT_H
#define T_FEATURE_DETECT_H
#include <stdint.h>
#include <stddef.h>
#include "cpuid.h"

/* XCR0 / XSTATE component bits */
#define T_XS_X87       (1ull << 0)
#define T_XS_SSE       (1ull << 1)
#define T_XS_AVX       (1ull << 2)
#define T_XS_BNDREGS   (1ull << 3)
#define T_XS_BNDCSR    (1ull << 4)
#define T_XS_OPMASK    (1ull << 5)
#define T_XS_ZMM_HI256 (1ull << 6)
#define T_XS_HI16_ZMM  (1ull << 7)
#define T_XS_PT        (1ull << 8)    /* supervisor */
#define T_XS_PKRU      (1ull << 9)
#define T_XS_PASID     (1ull << 10)   /* supervisor */
#define T_XS_CET_U     (1ull << 11)   /* supervisor */
#define T_XS_CET_S     (1ull << 12)   /* supervisor */
#define T_XS_HDC       (1ull << 13)   /* supervisor */
#define T_XS_UINTR     (1ull << 14)   /* supervisor */
#define T_XS_LBR       (1ull << 15)   /* supervisor */
#define T_XS_HWP       (1ull << 16)   /* supervisor */
#define T_XS_TILECFG   (1ull << 17)
#define T_XS_TILEDATA  (1ull << 18)
#define T_XS_APX       (1ull << 19)
#define T_XS_YMM_STATE (T_XS_SSE | T_XS_AVX)                                            /* 0x6  */
#define T_XS_ZMM_STATE (T_XS_YMM_STATE | T_XS_OPMASK | T_XS_ZMM_HI256 | T_XS_HI16_ZMM) /* 0xE6 */
#define T_XS_AMX_STATE (T_XS_TILECFG | T_XS_TILEDATA)                                   /* 0x60000 */

/* Feature table: X(id, leaf, subleaf, reg, bit, gate, "name") */
enum { G_NONE, G_OSXSAVE, G_YMM, G_ZMM, G_AMX, G_APX, G_PKU, G_PRIV };
#define T_FEATURE_LIST(X) \
    X(FPU, 1,0,CPUID_EDX,0, G_NONE, "fpu") X(TSC, 1,0,CPUID_EDX,4, G_NONE, "tsc") \
    X(CX8, 1,0,CPUID_EDX,8, G_NONE, "cx8") X(CMOV, 1,0,CPUID_EDX,15, G_NONE, "cmov") \
    X(CLFSH, 1,0,CPUID_EDX,19, G_NONE, "clflush") X(MMX, 1,0,CPUID_EDX,23, G_NONE, "mmx") \
    X(FXSR, 1,0,CPUID_EDX,24, G_NONE, "fxsr") X(SSE, 1,0,CPUID_EDX,25, G_NONE, "sse") \
    X(SSE2, 1,0,CPUID_EDX,26, G_NONE, "sse2") X(HTT, 1,0,CPUID_EDX,28, G_NONE, "htt") \
    X(SSE3, 1,0,CPUID_ECX,0, G_NONE, "sse3") X(PCLMULQDQ, 1,0,CPUID_ECX,1, G_NONE, "pclmulqdq") \
    X(MONITOR, 1,0,CPUID_ECX,3, G_NONE, "monitor") X(SSSE3, 1,0,CPUID_ECX,9, G_NONE, "ssse3") \
    X(FMA, 1,0,CPUID_ECX,12, G_YMM, "fma") X(CX16, 1,0,CPUID_ECX,13, G_NONE, "cx16") \
    X(SSE41, 1,0,CPUID_ECX,19, G_NONE, "sse4.1") X(SSE42, 1,0,CPUID_ECX,20, G_NONE, "sse4.2") \
    X(MOVBE, 1,0,CPUID_ECX,22, G_NONE, "movbe") X(POPCNT, 1,0,CPUID_ECX,23, G_NONE, "popcnt") \
    X(AES, 1,0,CPUID_ECX,25, G_NONE, "aes") X(XSAVE, 1,0,CPUID_ECX,26, G_OSXSAVE, "xsave") \
    X(OSXSAVE, 1,0,CPUID_ECX,27, G_NONE, "osxsave") X(AVX, 1,0,CPUID_ECX,28, G_YMM, "avx") \
    X(F16C, 1,0,CPUID_ECX,29, G_YMM, "f16c") X(RDRAND, 1,0,CPUID_ECX,30, G_NONE, "rdrand") \
    X(HYPERVISOR, 1,0,CPUID_ECX,31, G_NONE, "hypervisor") \
    X(FSGSBASE, 7,0,CPUID_EBX,0, G_PRIV, "fsgsbase") X(BMI1, 7,0,CPUID_EBX,3, G_NONE, "bmi1") \
    X(HLE, 7,0,CPUID_EBX,4, G_NONE, "hle") X(AVX2, 7,0,CPUID_EBX,5, G_YMM, "avx2") \
    X(SMEP, 7,0,CPUID_EBX,7, G_PRIV, "smep") X(BMI2, 7,0,CPUID_EBX,8, G_NONE, "bmi2") \
    X(ERMS, 7,0,CPUID_EBX,9, G_NONE, "erms") X(INVPCID, 7,0,CPUID_EBX,10, G_PRIV, "invpcid") \
    X(RTM, 7,0,CPUID_EBX,11, G_NONE, "rtm") X(AVX512F, 7,0,CPUID_EBX,16, G_ZMM, "avx512f") \
    X(AVX512DQ, 7,0,CPUID_EBX,17, G_ZMM, "avx512dq") X(RDSEED, 7,0,CPUID_EBX,18, G_NONE, "rdseed") \
    X(ADX, 7,0,CPUID_EBX,19, G_NONE, "adx") X(SMAP, 7,0,CPUID_EBX,20, G_PRIV, "smap") \
    X(AVX512IFMA, 7,0,CPUID_EBX,21, G_ZMM, "avx512ifma") X(CLFLUSHOPT, 7,0,CPUID_EBX,23, G_NONE, "clflushopt") \
    X(CLWB, 7,0,CPUID_EBX,24, G_NONE, "clwb") X(AVX512PF, 7,0,CPUID_EBX,26, G_ZMM, "avx512pf") \
    X(AVX512ER, 7,0,CPUID_EBX,27, G_ZMM, "avx512er") X(AVX512CD, 7,0,CPUID_EBX,28, G_ZMM, "avx512cd") \
    X(SHA, 7,0,CPUID_EBX,29, G_NONE, "sha") X(AVX512BW, 7,0,CPUID_EBX,30, G_ZMM, "avx512bw") \
    X(AVX512VL, 7,0,CPUID_EBX,31, G_ZMM, "avx512vl") \
    X(PREFETCHWT1, 7,0,CPUID_ECX,0, G_NONE, "prefetchwt1") X(AVX512VBMI, 7,0,CPUID_ECX,1, G_ZMM, "avx512vbmi") \
    X(UMIP, 7,0,CPUID_ECX,2, G_PRIV, "umip") X(PKU, 7,0,CPUID_ECX,3, G_PKU, "pku") \
    X(OSPKE, 7,0,CPUID_ECX,4, G_NONE, "ospke") X(WAITPKG, 7,0,CPUID_ECX,5, G_NONE, "waitpkg") \
    X(AVX512VBMI2, 7,0,CPUID_ECX,6, G_ZMM, "avx512vbmi2") X(CET_SS, 7,0,CPUID_ECX,7, G_PRIV, "cet_ss") \
    X(GFNI, 7,0,CPUID_ECX,8, G_NONE, "gfni") X(VAES, 7,0,CPUID_ECX,9, G_YMM, "vaes") \
    X(VPCLMULQDQ, 7,0,CPUID_ECX,10, G_YMM, "vpclmulqdq") X(AVX512VNNI, 7,0,CPUID_ECX,11, G_ZMM, "avx512vnni") \
    X(AVX512BITALG, 7,0,CPUID_ECX,12, G_ZMM, "avx512bitalg") X(AVX512VPOPCNTDQ, 7,0,CPUID_ECX,14, G_ZMM, "avx512vpopcntdq") \
    X(LA57, 7,0,CPUID_ECX,16, G_PRIV, "la57") X(RDPID, 7,0,CPUID_ECX,22, G_NONE, "rdpid") \
    X(KL, 7,0,CPUID_ECX,23, G_NONE, "keylocker") X(CLDEMOTE, 7,0,CPUID_ECX,25, G_NONE, "cldemote") \
    X(MOVDIRI, 7,0,CPUID_ECX,27, G_NONE, "movdiri") X(MOVDIR64B, 7,0,CPUID_ECX,28, G_NONE, "movdir64b") \
    X(ENQCMD, 7,0,CPUID_ECX,29, G_PRIV, "enqcmd") \
    X(AVX5124VNNIW, 7,0,CPUID_EDX,2, G_ZMM, "avx512_4vnniw") X(AVX5124FMAPS, 7,0,CPUID_EDX,3, G_ZMM, "avx512_4fmaps") \
    X(FSRM, 7,0,CPUID_EDX,4, G_NONE, "fsrm") X(UINTR, 7,0,CPUID_EDX,5, G_PRIV, "uintr") \
    X(AVX512VP2INTERSECT, 7,0,CPUID_EDX,8, G_ZMM, "avx512vp2intersect") X(MD_CLEAR, 7,0,CPUID_EDX,10, G_NONE, "md_clear") \
    X(SERIALIZE, 7,0,CPUID_EDX,14, G_NONE, "serialize") X(HYBRID, 7,0,CPUID_EDX,15, G_NONE, "hybrid") \
    X(TSXLDTRK, 7,0,CPUID_EDX,16, G_NONE, "tsxldtrk") X(PCONFIG, 7,0,CPUID_EDX,18, G_PRIV, "pconfig") \
    X(CET_IBT, 7,0,CPUID_EDX,20, G_PRIV, "cet_ibt") X(AMX_BF16, 7,0,CPUID_EDX,22, G_AMX, "amx_bf16") \
    X(AVX512FP16, 7,0,CPUID_EDX,23, G_ZMM, "avx512fp16") X(AMX_TILE, 7,0,CPUID_EDX,24, G_AMX, "amx_tile") \
    X(AMX_INT8, 7,0,CPUID_EDX,25, G_AMX, "amx_int8") \
    X(SHA512, 7,1,CPUID_EAX,0, G_YMM, "sha512") X(SM3, 7,1,CPUID_EAX,1, G_YMM, "sm3") \
    X(SM4, 7,1,CPUID_EAX,2, G_YMM, "sm4") X(RAO_INT, 7,1,CPUID_EAX,3, G_NONE, "rao_int") \
    X(AVX_VNNI, 7,1,CPUID_EAX,4, G_YMM, "avx_vnni") X(AVX512BF16, 7,1,CPUID_EAX,5, G_ZMM, "avx512bf16") \
    X(CMPCCXADD, 7,1,CPUID_EAX,7, G_NONE, "cmpccxadd") X(AMX_FP16, 7,1,CPUID_EAX,21, G_AMX, "amx_fp16") \
    X(HRESET, 7,1,CPUID_EAX,22, G_PRIV, "hreset") X(AVX_IFMA, 7,1,CPUID_EAX,23, G_YMM, "avx_ifma") \
    X(LAM, 7,1,CPUID_EAX,26, G_PRIV, "lam") \
    X(AVX_VNNI_INT8, 7,1,CPUID_EDX,4, G_YMM, "avx_vnni_int8") X(AVX_NE_CONVERT, 7,1,CPUID_EDX,5, G_YMM, "avx_ne_convert") \
    X(AMX_COMPLEX, 7,1,CPUID_EDX,8, G_AMX, "amx_complex") X(AVX_VNNI_INT16, 7,1,CPUID_EDX,10, G_YMM, "avx_vnni_int16") \
    X(PREFETCHI, 7,1,CPUID_EDX,14, G_NONE, "prefetchi") X(USER_MSR, 7,1,CPUID_EDX,15, G_PRIV, "user_msr") \
    X(AVX10, 7,1,CPUID_EDX,19, G_YMM, "avx10") X(APX_F, 7,1,CPUID_EDX,21, G_APX, "apx_f") \
    X(XSAVEOPT, 0xD,1,CPUID_EAX,0, G_OSXSAVE, "xsaveopt") X(XSAVEC, 0xD,1,CPUID_EAX,1, G_OSXSAVE, "xsavec") \
    X(XGETBV1, 0xD,1,CPUID_EAX,2, G_OSXSAVE, "xgetbv1") X(XSAVES, 0xD,1,CPUID_EAX,3, G_PRIV, "xsaves") \
    X(XFD, 0xD,1,CPUID_EAX,4, G_OSXSAVE, "xfd") \
    X(LAHF_LM, 0x80000001u,0,CPUID_ECX,0, G_NONE, "lahf_lm") X(LZCNT, 0x80000001u,0,CPUID_ECX,5, G_NONE, "lzcnt") \
    X(SSE4A, 0x80000001u,0,CPUID_ECX,6, G_NONE, "sse4a") X(MISALIGNSSE, 0x80000001u,0,CPUID_ECX,7, G_NONE, "misalignsse") \
    X(PREFETCHW, 0x80000001u,0,CPUID_ECX,8, G_NONE, "prefetchw") X(XOP, 0x80000001u,0,CPUID_ECX,11, G_YMM, "xop") \
    X(FMA4, 0x80000001u,0,CPUID_ECX,16, G_YMM, "fma4") X(TBM, 0x80000001u,0,CPUID_ECX,21, G_NONE, "tbm") \
    X(SYSCALL, 0x80000001u,0,CPUID_EDX,11, G_NONE, "syscall") X(NX, 0x80000001u,0,CPUID_EDX,20, G_NONE, "nx") \
    X(MMXEXT, 0x80000001u,0,CPUID_EDX,22, G_NONE, "mmxext") X(FFXSR, 0x80000001u,0,CPUID_EDX,25, G_NONE, "ffxsr") \
    X(PAGE1GB, 0x80000001u,0,CPUID_EDX,26, G_NONE, "pdpe1gb") X(RDTSCP, 0x80000001u,0,CPUID_EDX,27, G_NONE, "rdtscp") \
    X(LM, 0x80000001u,0,CPUID_EDX,29, G_NONE, "lm") X(AMD3DNOWEXT, 0x80000001u,0,CPUID_EDX,30, G_NONE, "3dnowext") \
    X(AMD3DNOW, 0x80000001u,0,CPUID_EDX,31, G_NONE, "3dnow") \
    X(INVARIANT_TSC, 0x80000007u,0,CPUID_EDX,8, G_NONE, "invariant_tsc") \
    X(CLZERO, 0x80000008u,0,CPUID_EBX,0, G_NONE, "clzero") X(WBNOINVD, 0x80000008u,0,CPUID_EBX,9, G_PRIV, "wbnoinvd")

#define T_FEAT_ENUM_(id, l, s, r, b, g, n) F_##id,
typedef enum t_feature { T_FEATURE_LIST(T_FEAT_ENUM_) F_COUNT } t_feature;
#undef T_FEAT_ENUM_

typedef struct t_xcomp {
    uint8_t  valid;       /* supported by the CPU (in XCR0-supported or XSS-supported mask) */
    uint8_t  supervisor;  /* CPUID.(0Dh,i):ECX[0] - managed via IA32_XSS, not XCR0 */
    uint8_t  align64;     /* CPUID.(0Dh,i):ECX[1] - 64-byte aligned in the compacted format */
    uint8_t  xfd;         /* CPUID.(0Dh,i):ECX[2] - supports extended feature disable */
    uint32_t size;        /* CPUID.(0Dh,i):EAX */
    uint32_t offset;      /* CPUID.(0Dh,i):EBX - standard-format offset (0 for supervisor) */
} t_xcomp;

typedef struct t_cpu_info {
    int      cpuid_ok;
    char     vendor[13], brand[49], hv_vendor[13];
    uint32_t max_basic, max_ext, max_hv;
    uint32_t signature;                   /* CPUID.1:EAX */
    uint32_t family, model, stepping;     /* display family/model (extended fields folded in) */
    int      is_intel, is_amd, hypervisor;
    t_cpuid_regs l1, l7_0, l7_1, ld_0, ld_1, e1, e7, e8;
    /* OS state */
    int      osxsave;                     /* CPUID.1:ECX[27] */
    int      xcr0_valid;  uint64_t xcr0;  /* XGETBV(0), only read when osxsave */
    int      xinuse_valid; uint64_t xinuse; /* XGETBV(1) at detection time */
    /* XSAVE enumeration (CPUID leaf 0Dh) */
    uint64_t xcr0_supported;              /* (0Dh,0) EDX:EAX */
    uint64_t xss_supported;               /* (0Dh,1) EDX:ECX */
    uint32_t xsave_size_enabled;          /* (0Dh,0) EBX: standard size for current XCR0 */
    uint32_t xsave_size_max;              /* (0Dh,0) ECX: standard size for all supported components */
    uint32_t xsave_size_compacted;        /* (0Dh,1) EBX: compacted size for XCR0|IA32_XSS */
    t_xcomp  comp[64];
    int      avx10_version;               /* CPUID.24h.EBX[7:0] when AVX10, else 0 */
    uint8_t  hw[F_COUNT], usable[F_COUNT];
} t_cpu_info;

void               cpu_detect(void);           /* idempotent */
const t_cpu_info  *cpu(void);                   /* detects on first use */
int                cpu_has(t_feature f);        /* usable now (gated) */
int                cpu_hw(t_feature f);         /* raw CPUID bit */
const char        *cpu_feature_name(t_feature f);
const char        *cpu_feature_gate_name(t_feature f);  /* "none","osxsave","ymm","zmm","amx","apx","pku","priv" */
int                cpu_feature_lookup(const char *name); /* name -> t_feature, -1 if unknown */
/* Space-separated lowercase names (usable_only: cpu_has, else cpu_hw). Returns length. */
size_t             cpu_features_string(char *buf, size_t cap, int usable_only);
int                cpu_ymm_state_enabled(void); /* osxsave && (XCR0 & 0x6) == 0x6  */
int                cpu_zmm_state_enabled(void); /* osxsave && (XCR0 & 0xE6) == 0xE6 */

/* XSAVE layout helpers (from CPUID leaf 0Dh) */
const char *xstate_comp_name(int i);            /* "x87","sse","avx",...; "comp<i>" if unnamed */
uint32_t    xsave_std_size(uint64_t mask);      /* standard-format bytes needed for user components in mask (>= 576) */
uint32_t    xsave_compacted_offset(uint64_t mask, int comp); /* offset of comp in compacted format for RFBM=mask */
uint32_t    xsave_compacted_size(uint64_t mask); /* compacted-format size for mask (>= 576) */
#endif
