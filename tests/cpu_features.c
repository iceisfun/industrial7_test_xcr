/* cpu_features - CPUID enumeration, vendor/family decode, feature tables, XSAVE (leaf 0Dh) layout,
 * OSXSAVE/XGETBV usability, IsProcessorFeaturePresent vs real instruction execution, and stability of
 * all of it across repeated calls, threads and every CPU of the process affinity mask.
 *
 * Contracts under test (sources):
 *   - Intel SDM Vol.2A "CPUID" (leaves 0,1,2,4,6,7,0Bh,0Dh,14h,15h,16h,1Ah,1Fh,24h, 8000000xh) and
 *     AMD APM Vol.3 Appendix E (Fn8000_0001 mirrors, Fn8000_0005/6/8, Fn8000_001D).
 *       * leaf 7 / leaf 0Dh: "If ECX contains an invalid sub-leaf index, EAX/EBX/ECX/EDX return 0"
 *         (0Dh sub-leaf n invalid when neither (0Dh,0) EDX:EAX[n] nor (0Dh,1) EDX:ECX[n] is set).
 *       * leaf 0Dh: EAX[1:0] (x87, SSE) always 1; EBX = size for components enabled in XCR0
 *         (max offset+size); ECX = size for all supported user components; supervisor components
 *         report EBX(offset) = 0 and ECX[0] = 1; (0Dh,1) EBX = compacted size for XCR0 | IA32_XSS.
 *       * XSAVE state component sizes fixed by their register content (AVX 256 @ 576, opmask 64,
 *         ZMM_Hi256 512, Hi16_ZMM 1024, PKRU 8, BNDREGS 64, BNDCSR 64, TILECFG 64, TILEDATA 8192, APX 128).
 *       * CR4.OSXSAVE (CPUID.1:ECX[27]) can only be set if XSAVE is supported; XGETBV #UD when it is 0;
 *         XGETBV #GP for an unsupported ECX; XSETBV #GP at CPL > 0.
 *       * VEX/EVEX instructions #UD when XCR0 does not enable the needed state (SDM 14.3, 15.2).
 *       * Leaf 0Bh EDX = x2APIC ID of the current logical processor for every sub-leaf; its low 8
 *         bits equal leaf 1 EBX[31:24] (initial APIC ID).
 *       * RDPID and RDTSCP both return IA32_TSC_AUX.
 *   - Microsoft docs IsProcessorFeaturePresent (sdk-api processthreadsapi, checked in the doc corpus):
 *     PF_* meanings, "All x64 processors always return a nonzero value" for PF_PAE_ENABLED.
 *     Win10 19041 decomp: kernelbase!IsProcessorFeaturePresent -> ntdll!RtlIsProcessorFeaturePresent,
 *     which returns KUSER_SHARED_DATA.ProcessorFeatures[f] for f < 64 and FALSE otherwise.
 *   - SYSTEM_INFO docs: wProcessorLevel is vendor-defined, wProcessorRevision = model<<8 | stepping
 *     ("display purposes": recorded as NOTE only).
 *
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; vendor/VM-variable facts
 * (frequencies, cache sizes, optional features, PF reverse direction for newer constants) -> NOTE/INFO.
 * Instructions that may fault run in CHILD processes. No AVX code runs before cpu_has() gates it. */
#include "t.h"
#include <x86intrin.h>

#define STATUS_ILLEGAL_INSN   0xC000001Du
#define STATUS_PRIV_INSN      0xC0000096u
#define STATUS_ACCESS_VIOL    0xC0000005u
#define STATUS_STACK_OVERRUN  0xC0000409u

#define OPQ(v)  __asm__ volatile("" : "+r"(v))        /* opaque integer: defeat constant folding */
#define OPQX(v) __asm__ volatile("" : "+x"(v))        /* opaque XMM/YMM value */
#define OPQM(p) __asm__ volatile("" : : "r"(p) : "memory")

static int is_intel(void) { return cpu()->is_intel; }
static int is_amd(void) { return cpu()->is_amd; }
static int bit(uint32_t v, int b) { return (int)((v >> b) & 1u); }
static t_cpuid_regs cpuidr(uint32_t leaf, uint32_t sub) { t_cpuid_regs r; cpuid_ex(leaf, sub, &r); return r; }
static int has_leaf(uint32_t leaf) {
    const t_cpu_info *c = cpu();
    if (leaf >= 0x80000000u) return c->max_ext && leaf <= c->max_ext;
    return leaf <= c->max_basic;
}
static int popcount64(uint64_t v) { int n = 0; for (; v; v &= v - 1) n++; return n; }
static int is_pow2(uint32_t v) { return v && !(v & (v - 1)); }
static int native_is_arm(void) { unsigned short m = t_env_get()->native_machine; return m == 0xAA64 || m == 0x01C4 || m == 0x01C0; }
static int mxcsr_daz_supported(void) {
    static uint8_t T_ALIGN(16) fx[512];
    memset(fx, 0, sizeof fx);
    t_fxsave(fx);
    uint32_t mask; memcpy(&mask, fx + 28, 4);
    if (!mask) mask = 0xFFBF;                       /* SDM 11.6.6: 0 means default mask (no DAZ) */
    return (mask >> 6) & 1;
}

/* ===================================================================================== */
/*  Instruction probes: each executes the instruction with opaque inputs and checks a    */
/*  known answer. Return 1 = correct. They run in-process only after cpu_has(), else in  */
/*  a child (--child=probe <name>).                                                       */
/* ===================================================================================== */
typedef int (*probe_fn)(char *m, size_t cap);
#define PFAIL(...) do { snprintf(m, cap, __VA_ARGS__); return 0; } while (0)

static int probe_fpu(char *m, size_t cap) {
    double a = 1.5, b = 2.25, four = 4.0, r1, r2;
    OPQM(&a);
    __asm__ volatile("fldl %2\n\tfldl %3\n\tfmulp\n\tfstpl %0\n\tfldl %4\n\tfsqrt\n\tfstpl %1"
                     : "=m"(r1), "=m"(r2) : "m"(a), "m"(b), "m"(four));
    if (r1 != 3.375 || r2 != 2.0) PFAIL("x87 1.5*2.25=%g sqrt(4)=%g", r1, r2);
    return 1;
}
static int probe_tsc(char *m, size_t cap) {
    uint64_t a = t_rdtsc(), b = t_rdtsc();
    if (a == 0 && b == 0) PFAIL("rdtsc returned 0 twice");
    (void)cap; (void)m; return 1;
}
static int probe_cx8(char *m, size_t cap) {
    static volatile uint64_t T_ALIGN(8) mem;
    mem = 0x1122334455667788ull;
    uint32_t lo = 0x55667788u, hi = 0x11223344u, nlo = 0xDDCCBBAAu, nhi = 0x01020304u;
    uint8_t ok1, ok2;
    __asm__ volatile("lock cmpxchg8b %0\n\tsetz %1" : "+m"(mem), "=q"(ok1), "+a"(lo), "+d"(hi) : "b"(nlo), "c"(nhi) : "cc");
    if (!ok1 || mem != 0x01020304DDCCBBAAull) PFAIL("cmpxchg8b success path ZF=%u mem=%llx", ok1, (unsigned long long)mem);
    lo = 0; hi = 0;
    __asm__ volatile("lock cmpxchg8b %0\n\tsetz %1" : "+m"(mem), "=q"(ok2), "+a"(lo), "+d"(hi) : "b"(nlo), "c"(nhi) : "cc");
    if (ok2 || lo != 0xDDCCBBAAu || hi != 0x01020304u) PFAIL("cmpxchg8b fail path ZF=%u edx:eax=%x:%x", ok2, hi, lo);
    return 1;
}
static int probe_cmov(char *m, size_t cap) {
    uint32_t a = 5, b = 9, z = 0; OPQ(z);
    __asm__("testl %2, %2\n\tcmovzl %1, %0" : "+r"(a) : "r"(b), "r"(z) : "cc");
    if (a != 9) PFAIL("cmovz gave %u", a);
    return 1;
}
static int probe_clflush(char *m, size_t cap) {
    static volatile uint8_t T_ALIGN(64) buf[128];
    for (int i = 0; i < 128; i++) buf[i] = (uint8_t)(i * 7);
    __asm__ volatile("clflush %0\n\tmfence" : "+m"(buf[0]) :: "memory");
    for (int i = 0; i < 128; i++) if (buf[i] != (uint8_t)(i * 7)) PFAIL("data changed after clflush at %d", i);
    return 1;
}
static int probe_mmx(char *m, size_t cap) {
    uint64_t a = 0x0001000200037FFFull, b = 0x0001000100010001ull, r, rs;
    OPQ(a);
    __asm__ volatile("movq %2, %%mm0\n\tmovq %3, %%mm1\n\tmovq %%mm0, %%mm2\n\tpaddw %%mm1, %%mm0\n\tpaddsw %%mm1, %%mm2\n\t"
                     "movq %%mm0, %0\n\tmovq %%mm2, %1\n\temms" : "=m"(r), "=m"(rs) : "m"(a), "m"(b) : "mm0", "mm1", "mm2");
    if (r != 0x0002000300048000ull || rs != 0x0002000300047FFFull) PFAIL("paddw=%016llx paddsw=%016llx", (unsigned long long)r, (unsigned long long)rs);
    unsigned short tw = t_get_x87_tw();
    if (tw != 0xFFFF) PFAIL("x87 tag word 0x%04x after EMMS (want 0xFFFF)", tw);
    return 1;
}
static int probe_fxsr(char *m, size_t cap) {
    static uint8_t T_ALIGN(16) fx[512];
    unsigned mx = t_get_mxcsr(); unsigned short cw = t_get_x87_cw();
    memset(fx, 0xA5, sizeof fx);
    t_fxsave(fx);
    uint32_t smx; uint16_t scw; memcpy(&smx, fx + 24, 4); memcpy(&scw, fx + 0, 2);
    if (smx != mx || scw != cw) PFAIL("fxsave mxcsr=%x (cur %x) fcw=%x (cur %x)", smx, mx, scw, cw);
    t_fxrstor(fx);
    if (t_get_mxcsr() != mx) PFAIL("mxcsr changed by fxrstor");
    return 1;
}
static int probe_sse1(char *m, size_t cap) {
    __m128 a = _mm_set_ps(4, 3, 2, 1), b = _mm_set_ps(0.5f, 0.25f, -2, 8);
    OPQX(a); OPQX(b);
    float r[4], s; _mm_storeu_ps(r, _mm_add_ps(a, b));
    __m128 q = _mm_set_ss(16.0f); OPQX(q); s = _mm_cvtss_f32(_mm_sqrt_ss(q));
    if (r[0] != 9 || r[1] != 0 || r[2] != 3.25f || r[3] != 4.5f || s != 4.0f) PFAIL("addps %g %g %g %g sqrtss %g", r[0], r[1], r[2], r[3], s);
    return 1;
}
static int probe_sse2(char *m, size_t cap) {
    __m128i a = _mm_set_epi64x(-1, 0x7FFFFFFFFFFFFFFFll), b = _mm_set_epi64x(1, 1);
    OPQX(a); OPQX(b);
    uint64_t r[2]; _mm_storeu_si128((__m128i *)r, _mm_add_epi64(a, b));
    __m128d d = _mm_set_sd(-2.75); OPQX(d);
    int t = _mm_cvttsd_si32(d);
    if (r[0] != 0x8000000000000000ull || r[1] != 0 || t != -2) PFAIL("paddq %llx %llx cvttsd2si %d", (unsigned long long)r[0], (unsigned long long)r[1], t);
    return 1;
}
static T_TARGET("sse3") int probe_sse3(char *m, size_t cap) {
    __m128 a = _mm_set_ps(4, 3, 2, 1), b = _mm_set_ps(8, 7, 6, 5); OPQX(a); OPQX(b);
    float r[4]; _mm_storeu_ps(r, _mm_hadd_ps(a, b));
    __m128d c = _mm_set_pd(2, 1), d = _mm_set_pd(0.5, 0.5); OPQX(c); OPQX(d);
    double s[2]; _mm_storeu_pd(s, _mm_addsub_pd(c, d));
    if (r[0] != 3 || r[1] != 7 || r[2] != 11 || r[3] != 15 || s[0] != 0.5 || s[1] != 2.5) PFAIL("haddps %g %g %g %g addsubpd %g %g", r[0], r[1], r[2], r[3], s[0], s[1]);
    return 1;
}
static T_TARGET("ssse3") int probe_ssse3(char *m, size_t cap) {
    __m128i v = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i rev = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, (char)0x80);
    OPQX(v); OPQX(rev);
    uint8_t r[16]; _mm_storeu_si128((__m128i *)r, _mm_shuffle_epi8(v, rev));
    __m128i n = _mm_set1_epi32(-5); OPQX(n);
    int32_t a[4]; _mm_storeu_si128((__m128i *)a, _mm_abs_epi32(n));
    if (r[0] != 15 || r[14] != 1 || r[15] != 0 || a[0] != 5 || a[3] != 5) PFAIL("pshufb r0=%u r14=%u r15=%u pabsd=%d", r[0], r[14], r[15], a[0]);
    return 1;
}
static T_TARGET("sse4.1") int probe_sse41(char *m, size_t cap) {
    __m128i a = _mm_setr_epi32(0x10000, -3, 7, 1), b = _mm_setr_epi32(0x10000, 7, -1, 1); OPQX(a); OPQX(b);
    int32_t r[4]; _mm_storeu_si128((__m128i *)r, _mm_mullo_epi32(a, b));
    __m128 f = _mm_setr_ps(-1.5f, 2.5f, -0.0f, 3.0f); OPQX(f);
    float fl[4]; _mm_storeu_ps(fl, _mm_floor_ps(f));
    __m128 one = _mm_set1_ps(1.0f); OPQX(one);
    float dp = _mm_cvtss_f32(_mm_dp_ps(_mm_setr_ps(1, 2, 3, 4), one, 0xF1));
    int tz = _mm_testz_si128(a, _mm_setzero_si128());
    if (r[0] != 0 || r[1] != -21 || r[2] != -7 || fl[0] != -2 || fl[1] != 2 || dp != 10 || tz != 1)
        PFAIL("pmulld %d %d %d floor %g %g dpps %g ptest %d", r[0], r[1], r[2], fl[0], fl[1], dp, tz);
    return 1;
}
static T_TARGET("sse4.2") int probe_sse42(char *m, size_t cap) {
    static const char msg[] = "123456789";
    uint32_t crc = 0xFFFFFFFFu; OPQ(crc);
    for (int i = 0; i < 9; i++) crc = _mm_crc32_u8(crc, (unsigned char)msg[i]);
    crc ^= 0xFFFFFFFFu;
    __m128i hay = _mm_loadu_si128((const __m128i *)"hello world!!!!!"), nd = _mm_loadu_si128((const __m128i *)"wor\0............");
    OPQX(hay);
    int idx = _mm_cmpistri(nd, hay, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED);
    __m128i g1 = _mm_set_epi64x(-1, 5), g2 = _mm_set_epi64x(-2, 6); OPQX(g1);
    int64_t gt[2]; _mm_storeu_si128((__m128i *)gt, _mm_cmpgt_epi64(g1, g2));
    if (crc != 0xE3069283u || idx != 6 || gt[0] != 0 || gt[1] != -1) PFAIL("crc32c=%08x (want e3069283) pcmpistri=%d pcmpgtq=%lld,%lld", crc, idx, (long long)gt[0], (long long)gt[1]);
    return 1;
}
static T_TARGET("popcnt") int probe_popcnt(char *m, size_t cap) {
    uint32_t a = 0xF0F00001u, z = 0; OPQ(a); OPQ(z);
    uint32_t r, rz; __asm__("popcntl %1, %0" : "=r"(r) : "r"(a) : "cc"); __asm__("popcntl %1, %0" : "=r"(rz) : "r"(z) : "cc");
    if (r != 9 || rz != 0) PFAIL("popcnt %u %u", r, rz);
    return 1;
}
static int probe_lzcnt(char *m, size_t cap) {
    uint32_t one = 1, zero = 0, r1 = 0x55, r0 = 0x55; OPQ(one); OPQ(zero);
    __asm__("lzcntl %1, %0" : "+r"(r1) : "r"(one) : "cc");
    __asm__("lzcntl %1, %0" : "+r"(r0) : "r"(zero) : "cc");
    if (r1 != 31 || r0 != 32) PFAIL("lzcnt(1)=%u lzcnt(0)=%u (BSR semantics would give 0/unchanged)", r1, r0);
    return 1;
}
/* FIPS-197 Appendix C.1 AES-128 known answer, full key schedule (AESKEYGENASSIST), encrypt and decrypt (AESIMC). */
static T_TARGET("aes,sse2") __m128i aes_assist(__m128i k, __m128i t) {
    t = _mm_shuffle_epi32(t, 0xFF);
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4)); k = _mm_xor_si128(k, _mm_slli_si128(k, 4)); k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    return _mm_xor_si128(k, t);
}
static T_TARGET("aes,sse2") int probe_aes(char *m, size_t cap) {
    static const uint8_t key[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    static const uint8_t pt[16] = { 0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff };
    static const uint8_t ct[16] = { 0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a };
    __m128i ks[11];
    ks[0] = _mm_loadu_si128((const __m128i *)key); OPQX(ks[0]);
#define KX(i, rc) ks[i] = aes_assist(ks[i - 1], _mm_aeskeygenassist_si128(ks[i - 1], rc))
    KX(1, 0x01); KX(2, 0x02); KX(3, 0x04); KX(4, 0x08); KX(5, 0x10); KX(6, 0x20); KX(7, 0x40); KX(8, 0x80); KX(9, 0x1b); KX(10, 0x36);
#undef KX
    __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i *)pt), ks[0]);
    for (int i = 1; i < 10; i++) x = _mm_aesenc_si128(x, ks[i]);
    x = _mm_aesenclast_si128(x, ks[10]);
    uint8_t out[16]; _mm_storeu_si128((__m128i *)out, x);
    if (memcmp(out, ct, 16)) PFAIL("AES-128 FIPS-197 C.1 ciphertext mismatch: %02x%02x%02x%02x...", out[0], out[1], out[2], out[3]);
    x = _mm_xor_si128(x, ks[10]);
    for (int i = 9; i >= 1; i--) x = _mm_aesdec_si128(x, _mm_aesimc_si128(ks[i]));
    x = _mm_aesdeclast_si128(x, ks[0]);
    _mm_storeu_si128((__m128i *)out, x);
    if (memcmp(out, pt, 16)) PFAIL("AES-128 decrypt did not round-trip");
    return 1;
}
static T_TARGET("pclmul,sse2") int probe_pclmul(char *m, size_t cap) {
    __m128i a = _mm_set_epi64x(3, -1), b = _mm_set_epi64x(3, 2); OPQX(a); OPQX(b);
    uint64_t lo[2], hi[2];
    _mm_storeu_si128((__m128i *)lo, _mm_clmulepi64_si128(a, b, 0x00));
    _mm_storeu_si128((__m128i *)hi, _mm_clmulepi64_si128(a, b, 0x11));
    if (lo[0] != 0xFFFFFFFFFFFFFFFEull || lo[1] != 1 || hi[0] != 5 || hi[1] != 0)
        PFAIL("pclmulqdq lo=%llx:%llx hi=%llx:%llx", (unsigned long long)lo[1], (unsigned long long)lo[0], (unsigned long long)hi[1], (unsigned long long)hi[0]);
    return 1;
}
/* SHA-256("abc") with SHA256MSG1/MSG2 (schedule cross-checked against software) + SHA256RNDS2, and SHA1NEXTE. */
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
static uint32_t ror32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static T_TARGET("sha,sse4.1") int probe_sha(char *m, size_t cap) {
    uint32_t W[64] = { 0x61626380u }, SW[64];
    W[15] = 24;
    memcpy(SW, W, sizeof W);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ror32(SW[i-15], 7) ^ ror32(SW[i-15], 18) ^ (SW[i-15] >> 3);
        uint32_t s1 = ror32(SW[i-2], 17) ^ ror32(SW[i-2], 19) ^ (SW[i-2] >> 10);
        SW[i] = SW[i-16] + s0 + SW[i-7] + s1;
    }
    __m128i X[16];
    for (int i = 0; i < 4; i++) { X[i] = _mm_loadu_si128((const __m128i *)&W[4 * i]); OPQX(X[i]); }
    for (int i = 4; i < 16; i++) {
        __m128i t = _mm_sha256msg1_epu32(X[i - 4], X[i - 3]);
        t = _mm_add_epi32(t, _mm_alignr_epi8(X[i - 1], X[i - 2], 4));
        X[i] = _mm_sha256msg2_epu32(t, X[i - 1]);
    }
    for (int i = 0; i < 16; i++) _mm_storeu_si128((__m128i *)&W[4 * i], X[i]);
    for (int i = 16; i < 64; i++) if (W[i] != SW[i]) PFAIL("SHA256MSG1/MSG2 schedule W[%d]=%08x want %08x", i, W[i], SW[i]);
    uint32_t H[8] = { 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19 };
    __m128i tmp = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&H[0]), 0xB1);   /* CDAB */
    __m128i s1 = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&H[4]), 0x1B);    /* EFGH */
    __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);                                          /* ABEF */
    s1 = _mm_blend_epi16(s1, tmp, 0xF0);                                               /* CDGH */
    __m128i a0 = s0, c0 = s1;
    for (int i = 0; i < 16; i++) {
        __m128i wk = _mm_add_epi32(X[i], _mm_loadu_si128((const __m128i *)&K256[4 * i]));
        s1 = _mm_sha256rnds2_epu32(s1, s0, wk);
        wk = _mm_shuffle_epi32(wk, 0x0E);
        s0 = _mm_sha256rnds2_epu32(s0, s1, wk);
    }
    s0 = _mm_add_epi32(s0, a0); s1 = _mm_add_epi32(s1, c0);
    tmp = _mm_shuffle_epi32(s0, 0x1B);          /* FEBA */
    s1 = _mm_shuffle_epi32(s1, 0xB1);           /* DCHG */
    s0 = _mm_blend_epi16(tmp, s1, 0xF0);        /* DCBA */
    s1 = _mm_alignr_epi8(s1, tmp, 8);           /* HGFE */
    uint32_t out[8]; _mm_storeu_si128((__m128i *)&out[0], s0); _mm_storeu_si128((__m128i *)&out[4], s1);
    static const uint32_t want[8] = { 0xba7816bf,0x8f01cfea,0x414140de,0x5dae2223,0xb00361a3,0x96177a9c,0xb410ff61,0xf20015ad };
    if (memcmp(out, want, sizeof want)) PFAIL("SHA-256(\"abc\") = %08x%08x... want ba7816bf8f01cfea...", out[0], out[1]);
    /* SHA1NEXTE: dst[127:96] = ROL30(dst[127:96]) + src[127:96]; lower dwords = src */
    __m128i d = _mm_setr_epi32(0x11, 0x22, 0x33, 4), s = _mm_setr_epi32(7, 8, 9, 10); OPQX(d);
    uint32_t n[4]; _mm_storeu_si128((__m128i *)n, _mm_sha1nexte_epu32(d, s));
    if (n[3] != 11 || n[0] != 7 || n[1] != 8 || n[2] != 9) PFAIL("sha1nexte %x %x %x %x", n[0], n[1], n[2], n[3]);
    return 1;
}
static int probe_xsave(char *m, size_t cap) {
    const t_cpu_info *c = cpu();
    uint64_t x = t_xgetbv(0);
    if (x != c->xcr0) PFAIL("xgetbv(0)=%llx differs from startup %llx", (unsigned long long)x, (unsigned long long)c->xcr0);
    if ((x & 3) != 3) PFAIL("XCR0[1:0] != 11b: %llx", (unsigned long long)x);
    uint32_t sz = c->xsave_size_enabled;
    if (sz < 576 || sz > 64 * 1024) PFAIL("(0Dh,0) EBX size %u out of range", sz);
    uint8_t *raw = (uint8_t *)malloc(sz + 64 + 4096);
    if (!raw) PFAIL("malloc");
    uint8_t *a = (uint8_t *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
    memset(a, 0, sz + 4096);
    memset(a + sz, 0xEE, 4096);
    t_xsave(a, x);
    uint64_t bv; memcpy(&bv, a + 512, 8);
    uint32_t mx; memcpy(&mx, a + 24, 4);
    int over = 0; for (int i = 0; i < 4096; i++) if (a[sz + i] != 0xEE) { over = 1; break; }
    int ok = !(bv & ~x) && mx == t_get_mxcsr() && !over;
    if (ok) t_xrstor(a, x);
    free(raw);
    if (!ok) PFAIL("xsave: XSTATE_BV=%llx (XCR0 %llx) mxcsr=%x wrote past EBX size=%d", (unsigned long long)bv, (unsigned long long)x, mx, over);
    return 1;
}
static int probe_xsaveopt(char *m, size_t cap) {
    const t_cpu_info *c = cpu();
    uint32_t sz = c->xsave_size_enabled;
    uint8_t *raw = (uint8_t *)calloc(1, sz + 64);
    if (!raw) PFAIL("calloc");
    uint8_t *a = (uint8_t *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
    t_xsaveopt(a, c->xcr0);
    uint64_t bv; memcpy(&bv, a + 512, 8);
    uint32_t mx; memcpy(&mx, a + 24, 4);
    free(raw);
    if (bv & ~c->xcr0) PFAIL("xsaveopt XSTATE_BV %llx not subset of XCR0", (unsigned long long)bv);
    if ((bv & 2) && mx != t_get_mxcsr()) PFAIL("xsaveopt mxcsr %x", mx);
    return 1;
}
static int probe_xsavec(char *m, size_t cap) {
    const t_cpu_info *c = cpu();
    uint32_t sz = xsave_compacted_size(c->xcr0);
    uint8_t *raw = (uint8_t *)calloc(1, sz + 64 + 4096);
    if (!raw) PFAIL("calloc");
    uint8_t *a = (uint8_t *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
    memset(a + sz, 0xEE, 4096);
    t_xsavec(a, c->xcr0);
    uint64_t bv, comp; memcpy(&bv, a + 512, 8); memcpy(&comp, a + 520, 8);
    int over = 0; for (int i = 0; i < 4096; i++) if (a[sz + i] != 0xEE) { over = 1; break; }
    if (comp == ((1ull << 63) | c->xcr0) && !(bv & ~c->xcr0) && !over) t_xrstor(a, c->xcr0);
    free(raw);
    /* SDM XSAVEC: XCOMP_BV = 8000_0000_0000_0000H | RFBM (RFBM = XCR0 & EDX:EAX) */
    if (comp != ((1ull << 63) | c->xcr0)) PFAIL("XCOMP_BV=%llx want %llx", (unsigned long long)comp, (unsigned long long)((1ull << 63) | c->xcr0));
    if (bv & ~c->xcr0) PFAIL("XSTATE_BV %llx not subset", (unsigned long long)bv);
    if (over) PFAIL("xsavec wrote past computed compacted size %u", sz);
    return 1;
}
static int probe_xgetbv1(char *m, size_t cap) {
    uint64_t in = t_xgetbv(1), x = t_xgetbv(0);
    if (in & ~x) PFAIL("XINUSE %llx has bits outside XCR0 %llx", (unsigned long long)in, (unsigned long long)x);
    return 1;
}
static T_TARGET("avx") int probe_avx1(char *m, size_t cap) {
    __m256 a = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8), b = _mm256_set1_ps(0.5f); OPQX(a); OPQX(b);
    float r[8], p[8];
    _mm256_storeu_ps(r, _mm256_add_ps(a, b));
    _mm256_storeu_ps(p, _mm256_permute2f128_ps(a, a, 0x01));
    for (int i = 0; i < 8; i++) if (r[i] != (float)(i + 1) + 0.5f) PFAIL("vaddps ymm lane %d = %g", i, r[i]);
    if (p[0] != 5 || p[4] != 1) PFAIL("vperm2f128 %g %g", p[0], p[4]);
    return 1;
}
static T_TARGET("avx2") int probe_avx2(char *m, size_t cap) {
    static const int32_t tab[16] = { 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115 };
    __m256i a = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0x7FFFFFFF), b = _mm256_set1_epi32(1); OPQX(a); OPQX(b);
    int32_t r[8], g[8], pv[8];
    _mm256_storeu_si256((__m256i *)r, _mm256_add_epi32(a, b));
    __m256i idx = _mm256_setr_epi32(15, 0, 3, 7, 8, 1, 2, 9); OPQX(idx);
    _mm256_storeu_si256((__m256i *)g, _mm256_i32gather_epi32(tab, idx, 4));
    _mm256_storeu_si256((__m256i *)pv, _mm256_permutevar8x32_epi32(a, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)));
    if (r[0] != 2 || r[7] != (int32_t)0x80000000) PFAIL("vpaddd %d %d", r[0], r[7]);
    for (int i = 0; i < 8; i++) {
        static const int gi[8] = { 15, 0, 3, 7, 8, 1, 2, 9 };
        if (g[i] != 100 + gi[i]) PFAIL("vpgatherdd lane %d = %d", i, g[i]);
    }
    if (pv[0] != 0x7FFFFFFF || pv[7] != 1) PFAIL("vpermd %d %d", pv[0], pv[7]);
    return 1;
}
static T_TARGET("avx,fma") int probe_fma(char *m, size_t cap) {
    /* x = 1 + 2^-27: x*x = 1 + 2^-26 + 2^-54; fma(x, x, -(1+2^-26)) = 2^-54 exactly (unfused gives 0) */
    __m128d x = _mm_set_sd(1.0 + 0x1p-27), c = _mm_set_sd(-(1.0 + 0x1p-26)); OPQX(x); OPQX(c);
    double r = _mm_cvtsd_f64(_mm_fmadd_sd(x, x, c));
    __m256 a = _mm256_set1_ps(3), b = _mm256_set1_ps(4), d = _mm256_set1_ps(-2); OPQX(a);
    float f[8]; _mm256_storeu_ps(f, _mm256_fmadd_ps(a, b, d));
    if (r != 0x1p-54) PFAIL("vfmadd sd fused residual %a want 0x1p-54", r);
    if (f[0] != 10 || f[7] != 10) PFAIL("vfmadd ps %g", f[0]);
    return 1;
}
static T_TARGET("avx,f16c") int probe_f16c(char *m, size_t cap) {
    __m128i h = _mm_setr_epi16(0x3C00, (short)0xC000, 0x7C00, 0x0001, 0x7BFF, 0, (short)0x8000, 0x3555); OPQX(h);
    float f[8]; _mm256_storeu_ps(f, _mm256_cvtph_ps(h));
    __m128 v = _mm_setr_ps(65504.0f, 1.0f, -0.0f, 70000.0f); OPQX(v);
    uint16_t o[8]; _mm_storeu_si128((__m128i *)o, _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    if (f[0] != 1.0f || f[1] != -2.0f || f[2] != __builtin_inff() || f[3] != 0x1p-24f || f[4] != 65504.0f) PFAIL("vcvtph2ps %g %g %g %a %g", f[0], f[1], f[2], f[3], f[4]);
    if (o[0] != 0x7BFF || o[1] != 0x3C00 || o[2] != 0x8000 || o[3] != 0x7C00) PFAIL("vcvtps2ph %04x %04x %04x %04x", o[0], o[1], o[2], o[3]);
    return 1;
}
static T_TARGET("bmi") int probe_bmi1(char *m, size_t cap) {
    uint32_t a = 0xF0F0F0F0u, b = 0xFFFF0000u, z = 0, x = 0x00001234u; OPQ(a); OPQ(z); OPQ(x);
    uint32_t andn = _andn_u32(a, b), bx = _bextr_u32(x, 4, 8), bi = _blsi_u32(0x30u + z), br = _blsr_u32(0x30u + z);
    uint32_t tz = 0x55; __asm__("tzcntl %1, %0" : "+r"(tz) : "r"(z) : "cc");
    if (andn != 0x0F0F0000u || bx != 0x23 || bi != 0x10 || br != 0x20 || tz != 32) PFAIL("andn %x bextr %x blsi %x blsr %x tzcnt(0)=%u", andn, bx, bi, br, tz);
    return 1;
}
static T_TARGET("bmi2") int probe_bmi2(char *m, size_t cap) {
    uint32_t a = 0xFF, mask = 0xF0F0, x = 0x12345678u, f = 0xFFFFFFFFu; OPQ(a); OPQ(x); OPQ(f);
    uint32_t pd = _pdep_u32(a, mask), pe = _pext_u32(x, 0xFF00FF00u), bz = _bzhi_u32(f, 7), lo, hi, rx, sh;
    __asm__("mulxl %3, %0, %1" : "=r"(lo), "=r"(hi) : "d"(f), "rm"(f));
    __asm__("rorxl $8, %1, %0" : "=r"(rx) : "r"(x));
    __asm__("shlxl %2, %1, %0" : "=r"(sh) : "r"(x), "r"(36u));
    if (pd != 0xF0F0 || pe != 0x1256 || bz != 0x7F || lo != 1 || hi != 0xFFFFFFFEu || rx != 0x78123456u || sh != 0x23456780u)
        PFAIL("pdep %x pext %x bzhi %x mulx %x:%x rorx %x shlx %x", pd, pe, bz, hi, lo, rx, sh);
    return 1;
}
static int probe_adx(char *m, size_t cap) {
    uint32_t a = 0xFFFFFFFFu, b = 1, c = 0x7FFFFFFFu, d = 1; uint8_t cf, of; OPQ(a); OPQ(c);
    __asm__("clc\n\tadcxl %2, %0\n\tsetc %1" : "+r"(a), "=q"(cf) : "r"(b) : "cc");
    __asm__("xorl %%eax, %%eax\n\tadoxl %2, %0\n\tseto %1" : "+r"(c), "=q"(of) : "r"(d) : "cc", "eax");
    if (a != 0 || cf != 1 || c != 0x80000000u || of != 0) PFAIL("adcx %x cf=%u adox %x of=%u", a, cf, c, of);
    return 1;
}
static int probe_movbe(char *m, size_t cap) {
    static volatile uint32_t src = 0x11223344u, dst;
    uint32_t v; __asm__ volatile("movbel %1, %0" : "=r"(v) : "m"(src));
    __asm__ volatile("movbel %1, %0" : "=m"(dst) : "r"(v));
    if (v != 0x44332211u || dst != 0x11223344u) PFAIL("movbe %x %x", v, dst);
    return 1;
}
static T_TARGET("rdrnd") int probe_rdrand(char *m, size_t cap) {
    unsigned int v[4] = { 0 }; int got = 0;
    for (int k = 0; k < 4; k++) for (int i = 0; i < 10; i++) if (_rdrand32_step(&v[k])) { got++; break; }
    if (got < 4) PFAIL("RDRAND CF=0 after 10 retries (%d/4 values)", got);
    if (v[0] == v[1] && v[1] == v[2] && v[2] == v[3]) PFAIL("RDRAND returned the same value 4 times: %08x", v[0]);
    return 1;
}
static T_TARGET("rdseed") int probe_rdseed(char *m, size_t cap) {
    unsigned int v = 0; int ok = 0;
    for (int i = 0; i < 1000 && !ok; i++) { ok = _rdseed32_step(&v); if (!ok) _mm_pause(); }
    if (!ok) PFAIL("RDSEED CF=0 for 1000 attempts");
    return 1;
}
static int probe_rdtscp(char *m, size_t cap) {
    uint32_t a1 = 0, a2 = 0; uint64_t t1 = t_rdtscp(&a1), t2 = t_rdtscp(&a2);
    if (!t1 && !t2) PFAIL("rdtscp returned 0");
    (void)cap; (void)m; return 1;
}
static T_TARGET("rdpid") int probe_rdpid(char *m, size_t cap) {
    uintptr_t prev = t_pin_thread(t_env_get()->ncpu > 0 ? (int)__builtin_ctzll(t_env_get()->affinity ? t_env_get()->affinity : 1) : 0);
    SwitchToThread();
    uint32_t aux = 0, p1, p2;
    p1 = _rdpid_u32(); t_rdtscp(&aux); p2 = _rdpid_u32();
    if (prev) t_unpin_thread(prev);
    if (prev && (p1 != aux || p2 != aux)) PFAIL("RDPID %x/%x != RDTSCP aux %x (both read IA32_TSC_AUX)", p1, p2, aux);
    return 1;
}
#ifdef T_X64
static int probe_cx16(char *m, size_t cap) {
    static volatile uint64_t T_ALIGN(16) mem[2];
    mem[0] = 1; mem[1] = 2;
    uint64_t lo = 1, hi = 2; uint8_t ok;
    __asm__ volatile("lock cmpxchg16b %0\n\tsetz %1" : "+m"(*(volatile uint64_t (*)[2])mem), "=q"(ok), "+a"(lo), "+d"(hi) : "b"(0xAAAAull), "c"(0xBBBBull) : "cc", "memory");
    if (!ok || mem[0] != 0xAAAA || mem[1] != 0xBBBB) PFAIL("cmpxchg16b ZF=%u mem=%llx:%llx", ok, (unsigned long long)mem[1], (unsigned long long)mem[0]);
    return 1;
}
static int probe_lahf(char *m, size_t cap) {
    uint32_t r1, r2;
    __asm__ volatile("stc\n\tlahf\n\tshrl $8, %%eax\n\tandl $0xff, %%eax" : "=a"(r1) :: "cc");
    __asm__ volatile("movl $0x100, %%eax\n\tsahf\n\tsetc %%al\n\tmovzbl %%al, %%eax" : "=a"(r2) :: "cc");
    if ((r1 & 3) != 3 || r2 != 1) PFAIL("lahf AH=%02x sahf->CF=%u", r1, r2);
    return 1;
}
#endif
static T_TARGET("gfni,sse2") int probe_gfni(char *m, size_t cap) {
    __m128i a = _mm_setr_epi8(2, (char)0x80, 0x53, 1, 0, 3, 7, 9, 2, 2, 2, 2, 2, 2, 2, 2);
    __m128i b = _mm_setr_epi8(3, 2, (char)0xCA, 0x77, 0x55, 3, 7, 9, 3, 3, 3, 3, 3, 3, 3, 3); OPQX(a); OPQX(b);
    uint8_t r[16]; _mm_storeu_si128((__m128i *)r, _mm_gf2p8mul_epi8(a, b));
    if (r[0] != 6 || r[1] != 0x1B || r[2] != 0x01 || r[3] != 0x77 || r[4] != 0 || r[5] != 5 || r[6] != 0x15) PFAIL("gf2p8mulb %02x %02x %02x %02x %02x %02x %02x", r[0], r[1], r[2], r[3], r[4], r[5], r[6]);
    return 1;
}
static T_TARGET("avx,vaes") int probe_vaes(char *m, size_t cap) {
    /* AESENC(state=0, key=k): SubBytes(0)=63h everywhere, ShiftRows/MixColumns keep 63h -> 63h ^ k */
    __m256i z = _mm256_setzero_si256(), k = _mm256_setr_epi32(1, 2, 3, 4, 0x10, 0x20, 0x30, 0x40); OPQX(z); OPQX(k);
    uint32_t r[8]; _mm256_storeu_si256((__m256i *)r, _mm256_aesenc_epi128(z, k));
    static const uint32_t kk[8] = { 1, 2, 3, 4, 0x10, 0x20, 0x30, 0x40 };
    for (int i = 0; i < 8; i++) if (r[i] != (0x63636363u ^ kk[i])) PFAIL("vaesenc ymm dword %d = %08x", i, r[i]);
    return 1;
}
static T_TARGET("avx,vpclmulqdq") int probe_vpclmul(char *m, size_t cap) {
    __m256i a = _mm256_setr_epi64x(3, 0, 0x87, 0), b = _mm256_setr_epi64x(3, 0, 2, 0); OPQX(a); OPQX(b);
    uint64_t r[4]; _mm256_storeu_si256((__m256i *)r, _mm256_clmulepi64_epi128(a, b, 0x00));
    if (r[0] != 5 || r[1] != 0 || r[2] != 0x10E || r[3] != 0) PFAIL("vpclmulqdq %llx %llx", (unsigned long long)r[0], (unsigned long long)r[2]);
    return 1;
}
static T_TARGET("avxvnni") int probe_avxvnni(char *m, size_t cap) {
    __m256i acc = _mm256_set1_epi32(10), u = _mm256_set1_epi8(2), s = _mm256_set1_epi8(-3); OPQX(acc);
    int32_t r[8]; _mm256_storeu_si256((__m256i *)r, _mm256_dpbusd_avx_epi32(acc, u, s));
    if (r[0] != 10 - 24 || r[7] != -14) PFAIL("vpdpbusd (VEX) %d", r[0]);
    return 1;
}
static T_TARGET("avx512f") int probe_avx512f(char *m, size_t cap) {
    __m512i a = _mm512_set1_epi32(7), b = _mm512_set1_epi32(-2), src = _mm512_set1_epi32(99);
    __asm__ volatile("" : "+v"(a), "+v"(b));
    int32_t r[16]; _mm512_storeu_si512(r, _mm512_mask_add_epi32(src, (__mmask16)0x5555, a, b));
    for (int i = 0; i < 16; i++) if (r[i] != ((i & 1) ? 99 : 5)) PFAIL("vpaddd zmm{k} lane %d = %d", i, r[i]);
    __mmask16 k = _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(7));
    if (k != 0xFFFF) PFAIL("vpcmpeqd k=%x", (unsigned)k);
    double d[8]; _mm512_storeu_pd(d, _mm512_sqrt_pd(_mm512_set1_pd(2.25)));
    if (d[7] != 1.5) PFAIL("vsqrtpd zmm %g", d[7]);
    return 1;
}
static T_TARGET("avx512f,avx512bw") int probe_avx512bw(char *m, size_t cap) {
    __m512i a = _mm512_set1_epi8(100), b = _mm512_set1_epi8(100);
    __asm__ volatile("" : "+v"(a));
    int8_t r[64]; _mm512_storeu_si512(r, _mm512_adds_epi8(a, b));
    __mmask64 k = _mm512_cmpeq_epi8_mask(a, b);
    if (r[0] != 127 || r[63] != 127 || k != ~0ull) PFAIL("vpaddsb zmm %d k=%llx", r[0], (unsigned long long)k);
    return 1;
}
static T_TARGET("avx512f,avx512dq") int probe_avx512dq(char *m, size_t cap) {
    __m512i a = _mm512_set1_epi64(0x100000001ll), b = _mm512_set1_epi64(3);
    __asm__ volatile("" : "+v"(a));
    int64_t r[8]; _mm512_storeu_si512(r, _mm512_mullo_epi64(a, b));
    if (r[0] != 0x300000003ll) PFAIL("vpmullq %llx", (unsigned long long)r[0]);
    return 1;
}
static T_TARGET("avx512f,avx512vl") int probe_avx512vl(char *m, size_t cap) {
    __m256i a = _mm256_set1_epi32(1), b = _mm256_set1_epi32(2), s = _mm256_set1_epi32(-1);
    __asm__ volatile("" : "+v"(a));
    int32_t r[8]; _mm256_storeu_si256((__m256i *)r, _mm256_mask_add_epi32(s, (__mmask8)0x0F, a, b));
    if (r[0] != 3 || r[3] != 3 || r[4] != -1 || r[7] != -1) PFAIL("EVEX.256 vpaddd{k} %d %d", r[0], r[4]);
    return 1;
}
static T_TARGET("avx512f,avx512cd") int probe_avx512cd(char *m, size_t cap) {
    __m512i a = _mm512_set1_epi32(1);
    __asm__ volatile("" : "+v"(a));
    int32_t r[16]; _mm512_storeu_si512(r, _mm512_lzcnt_epi32(a));
    if (r[0] != 31 || r[15] != 31) PFAIL("vplzcntd %d", r[0]);
    return 1;
}
static T_TARGET("avx512f,avx512vpopcntdq") int probe_avx512vpopcntdq(char *m, size_t cap) {
    __m512i a = _mm512_set1_epi32(0x0F0F0F0F);
    __asm__ volatile("" : "+v"(a));
    int32_t r[16]; _mm512_storeu_si512(r, _mm512_popcnt_epi32(a));
    if (r[0] != 16) PFAIL("vpopcntd %d", r[0]);
    return 1;
}
static T_TARGET("avx512f,avx512vnni") int probe_avx512vnni(char *m, size_t cap) {
    __m512i acc = _mm512_set1_epi32(10), u = _mm512_set1_epi8(2), s = _mm512_set1_epi8(-3);
    __asm__ volatile("" : "+v"(acc));
    int32_t r[16]; _mm512_storeu_si512(r, _mm512_dpbusd_epi32(acc, u, s));
    if (r[0] != -14) PFAIL("vpdpbusd zmm %d", r[0]);
    return 1;
}
static T_TARGET("avx512f,avx512vbmi") int probe_avx512vbmi(char *m, size_t cap) {
    uint8_t in[64], idx[64], out[64];
    for (int i = 0; i < 64; i++) { in[i] = (uint8_t)(i * 3); idx[i] = (uint8_t)(63 - i); }
    __m512i v = _mm512_loadu_si512(in), x = _mm512_loadu_si512(idx);
    __asm__ volatile("" : "+v"(v));
    _mm512_storeu_si512(out, _mm512_permutexvar_epi8(x, v));
    for (int i = 0; i < 64; i++) if (out[i] != (uint8_t)((63 - i) * 3)) PFAIL("vpermb lane %d", i);
    return 1;
}
static T_TARGET("clflushopt") int probe_clflushopt(char *m, size_t cap) {
    static volatile uint8_t T_ALIGN(64) buf[64]; buf[5] = 0x5A;
    _mm_clflushopt((void *)buf); _mm_sfence();
    if (buf[5] != 0x5A) PFAIL("data lost");
    return 1;
}
static T_TARGET("clwb") int probe_clwb(char *m, size_t cap) {
    static volatile uint8_t T_ALIGN(64) buf[64]; buf[9] = 0xA5;
    _mm_clwb((void *)buf); _mm_sfence();
    if (buf[9] != 0xA5) PFAIL("data lost");
    return 1;
}
static int probe_serialize(char *m, size_t cap) { __asm__ volatile(".byte 0x0f,0x01,0xe8" ::: "memory"); (void)m; (void)cap; return 1; }
static int probe_movdiri(char *m, size_t cap) {
    static volatile uint32_t T_ALIGN(64) dst; uint32_t v = 0xC0FFEEu; OPQ(v);
    __asm__ volatile("movdiri %1, %0" : "=m"(dst) : "r"(v) : "memory");
    if (dst != 0xC0FFEEu) PFAIL("movdiri stored %x", dst);
    return 1;
}
static int probe_pku(char *m, size_t cap) {
    uint32_t pkru, edx;
    __asm__ volatile(".byte 0x0f,0x01,0xee" : "=a"(pkru), "=d"(edx) : "c"(0));
    (void)cap; (void)m; t_log("PKRU=%08x", pkru); return 1;
}
static int probe_prefetchw(char *m, size_t cap) {
    static volatile uint8_t T_ALIGN(64) buf[64]; buf[0] = 1;
    __asm__ volatile("prefetchw %0" :: "m"(buf[0]));
    (void)m; (void)cap; return buf[0] == 1;
}
static int probe_erms(char *m, size_t cap) {
    static uint8_t s[300], d[300];
    for (int i = 0; i < 300; i++) { s[i] = (uint8_t)(i * 13 + 1); d[i] = 0; }
    void *dp = d + 3; const void *sp = s + 1; size_t n = 290;
    __asm__ volatile("cld\n\trep movsb" : "+D"(dp), "+S"(sp), "+c"(n) :: "memory");
    if (memcmp(d + 3, s + 1, 290) || d[2] || d[293]) PFAIL("rep movsb copy mismatch");
    return 1;
}
static int probe_clzero(char *m, size_t cap) {
    static volatile uint8_t T_ALIGN(64) buf[128];
    for (int i = 0; i < 128; i++) buf[i] = 0xAA;
    void *p = (void *)(buf + 8);
    __asm__ volatile(".byte 0x0f,0x01,0xfc\n\tsfence" :: "a"(p) : "memory");
    for (int i = 0; i < 64; i++) if (buf[i]) PFAIL("clzero left byte %d = %02x", i, buf[i]);
    if (buf[64] != 0xAA) PFAIL("clzero zeroed beyond its 64-byte line");
    return 1;
}
static int probe_3dnow(char *m, size_t cap) {
    float a[2] = { 1.5f, 2.0f }, b[2] = { 0.5f, -4.0f }, r[2];
    __asm__ volatile("movq %1, %%mm0\n\tpfadd %2, %%mm0\n\tmovq %%mm0, %0\n\tfemms" : "=m"(r) : "m"(a), "m"(b) : "mm0");
    if (r[0] != 2.0f || r[1] != -2.0f) PFAIL("pfadd %g %g", r[0], r[1]);
    return 1;
}

typedef struct { const char *name; t_feature f; probe_fn fn; int x64only; } probe_t;
static const probe_t g_probes[] = {
    { "fpu", F_FPU, probe_fpu, 0 }, { "tsc", F_TSC, probe_tsc, 0 }, { "cx8", F_CX8, probe_cx8, 0 },
    { "cmov", F_CMOV, probe_cmov, 0 }, { "clflush", F_CLFSH, probe_clflush, 0 }, { "mmx", F_MMX, probe_mmx, 0 },
    { "fxsr", F_FXSR, probe_fxsr, 0 }, { "sse1", F_SSE, probe_sse1, 0 }, { "sse2", F_SSE2, probe_sse2, 0 },
    { "sse3", F_SSE3, probe_sse3, 0 }, { "ssse3", F_SSSE3, probe_ssse3, 0 }, { "sse41", F_SSE41, probe_sse41, 0 },
    { "sse42", F_SSE42, probe_sse42, 0 }, { "popcnt", F_POPCNT, probe_popcnt, 0 }, { "lzcnt", F_LZCNT, probe_lzcnt, 0 },
    { "aes", F_AES, probe_aes, 0 }, { "pclmulqdq", F_PCLMULQDQ, probe_pclmul, 0 }, { "sha", F_SHA, probe_sha, 0 },
    { "xsave", F_XSAVE, probe_xsave, 0 }, { "xsaveopt", F_XSAVEOPT, probe_xsaveopt, 0 }, { "xsavec", F_XSAVEC, probe_xsavec, 0 },
    { "xgetbv1", F_XGETBV1, probe_xgetbv1, 0 }, { "avx1", F_AVX, probe_avx1, 0 }, { "avx2", F_AVX2, probe_avx2, 0 },
    { "fma", F_FMA, probe_fma, 0 }, { "f16c", F_F16C, probe_f16c, 0 }, { "bmi1", F_BMI1, probe_bmi1, 0 },
    { "bmi2", F_BMI2, probe_bmi2, 0 }, { "adx", F_ADX, probe_adx, 0 }, { "movbe", F_MOVBE, probe_movbe, 0 },
    { "rdrand", F_RDRAND, probe_rdrand, 0 }, { "rdseed", F_RDSEED, probe_rdseed, 0 }, { "rdtscp", F_RDTSCP, probe_rdtscp, 0 },
    { "rdpid", F_RDPID, probe_rdpid, 0 },
#ifdef T_X64
    { "cx16", F_CX16, probe_cx16, 1 }, { "lahf_lm", F_LAHF_LM, probe_lahf, 1 },
#else
    { "cx16", F_CX16, NULL, 1 }, { "lahf_lm", F_LAHF_LM, NULL, 1 },
#endif
    { "gfni", F_GFNI, probe_gfni, 0 }, { "vaes", F_VAES, probe_vaes, 0 }, { "vpclmulqdq", F_VPCLMULQDQ, probe_vpclmul, 0 },
    { "avx_vnni", F_AVX_VNNI, probe_avxvnni, 0 },
    { "avx512f", F_AVX512F, probe_avx512f, 0 }, { "avx512bw", F_AVX512BW, probe_avx512bw, 0 }, { "avx512dq", F_AVX512DQ, probe_avx512dq, 0 },
    { "avx512vl", F_AVX512VL, probe_avx512vl, 0 }, { "avx512cd", F_AVX512CD, probe_avx512cd, 0 },
    { "avx512vpopcntdq", F_AVX512VPOPCNTDQ, probe_avx512vpopcntdq, 0 }, { "avx512vnni", F_AVX512VNNI, probe_avx512vnni, 0 },
    { "avx512vbmi", F_AVX512VBMI, probe_avx512vbmi, 0 },
    { "clflushopt", F_CLFLUSHOPT, probe_clflushopt, 0 }, { "clwb", F_CLWB, probe_clwb, 0 }, { "serialize", F_SERIALIZE, probe_serialize, 0 },
    { "movdiri", F_MOVDIRI, probe_movdiri, 0 }, { "pku", F_PKU, probe_pku, 0 }, { "prefetchw", F_PREFETCHW, probe_prefetchw, 0 },
    { "erms", F_ERMS, probe_erms, 0 }, { "clzero", F_CLZERO, probe_clzero, 0 }, { "3dnow", F_AMD3DNOW, probe_3dnow, 0 },
};
static const probe_t *find_probe(const char *n) {
    for (size_t i = 0; i < T_COUNTOF(g_probes); i++) if (!strcmp(g_probes[i].name, n)) return &g_probes[i];
    return NULL;
}
/* Extra requirements beyond the probe's own feature (other instructions the known-answer uses). */
static int probe_prereq(const probe_t *p, char *why, size_t cap) {
    struct { const char *n; t_feature f; } req[] = {
        { "sha", F_SSE41 }, { "aes", F_SSE2 }, { "xsaveopt", F_XSAVE }, { "xsavec", F_XSAVE }, { "xgetbv1", F_XSAVE },
        { "fma", F_AVX }, { "f16c", F_AVX }, { "vaes", F_AVX }, { "vpclmulqdq", F_AVX }, { "avx_vnni", F_AVX2 },
        { "avx512bw", F_AVX512F }, { "avx512dq", F_AVX512F }, { "avx512vl", F_AVX512F }, { "avx512cd", F_AVX512F },
        { "avx512vpopcntdq", F_AVX512F }, { "avx512vnni", F_AVX512F }, { "avx512vbmi", F_AVX512F }, { "rdpid", F_RDTSCP },
        { "3dnow", F_MMX },
    };
    for (size_t i = 0; i < T_COUNTOF(req); i++)
        if (!strcmp(req[i].n, p->name) && !cpu_has(req[i].f)) { snprintf(why, cap, "%s also needs %s", p->name, cpu_feature_name(req[i].f)); return 0; }
    return 1;
}

CHILD(probe) {
    if (argc < 2) { printf("PROBE_ERR=noarg\n"); return 3; }
    const probe_t *p = find_probe(argv[1]);
    if (!p || !p->fn) { printf("PROBE_ERR=unknown\n"); return 3; }
    char msg[256] = "";
    printf("PROBE_START=%s\n", p->name); fflush(stdout);
    int ok = p->fn(msg, sizeof msg);
    printf("PROBE_OK=%d\nPROBE_MSG=%s\n", ok, msg);
    return ok ? 0 : 1;
}

/* exec_<name>: usable -> execute + known answer (FAIL if wrong). Advertised but OS state not enabled
 * (VEX/EVEX/XSAVE/PKU gates) -> child must take #UD, then SKIP. Not advertised -> SKIP. */
static void run_probe_case(const char *name) {
    const probe_t *p = find_probe(name);
    REQUIREF(p != NULL, "no probe '%s'", name);
#ifdef T_X86
    if (p->x64only) SKIP("%s is a 64-bit-mode instruction form (x86 process)", name);
#endif
    if (!cpu_hw(p->f)) SKIP("CPUID does not advertise %s", cpu_feature_name(p->f));
    if (!cpu_has(p->f)) {
        const char *g = cpu_feature_gate_name(p->f);
        if (!strcmp(g, "priv")) SKIP("%s is not usable from user mode", name);
        /* advertised but the OS has not enabled the state (XCR0 / CR4.OSXSAVE / CR4.PKE): SDM -> #UD */
        static t_child_result r;
        REQUIRE(t_child_run("probe", name, 10000, &r));
        CHECKF(!r.timed_out, "probe child timed out");
        CHECKF(r.exc_code == STATUS_ILLEGAL_INSN, "%s advertised by CPUID but gate '%s' closed: expected #UD (0xC000001D), got exit 0x%lx exc 0x%lx",
               name, g, (unsigned long)r.exit_code, (unsigned long)r.exc_code);
        SKIP("%s advertised but OS state not enabled (gate %s); #UD verified", name, g);
    }
    char why[128];
    if (!probe_prereq(p, why, sizeof why)) SKIP("%s", why);
    char msg[256] = "";
    int ok = p->fn(msg, sizeof msg);
    CHECKF(ok, "%s known-answer: %s", name, msg);
}

/* ===================================================================================== */
/*  CPUID leaf 0 / 1 / vendor / signature                                                 */
/* ===================================================================================== */
TEST(cpuid_eflags_id) {
    CHECK(cpuid_available());
    CHECK(cpu()->cpuid_ok);
}

TEST(leaf0_vendor_and_max) {
    const t_cpu_info *c = cpu();
    t_cpuid_regs r = cpuidr(0, 0);
    char v[13]; memcpy(v, &r.ebx, 4); memcpy(v + 4, &r.edx, 4); memcpy(v + 8, &r.ecx, 4); v[12] = 0;
    CHECK(!strcmp(v, c->vendor));
    for (int i = 0; i < 12; i++) CHECKF(v[i] >= 0x20 && v[i] < 0x7F, "vendor byte %d = 0x%02x not printable", i, (unsigned char)v[i]);
    CHECKF(r.eax >= 1, "max basic leaf %u: leaf 1 must exist on any SSE2 CPU", r.eax);
    CHECK_EQ(c->max_basic, r.eax);
    static const char *const known[] = { "GenuineIntel", "AuthenticAMD", "HygonGenuine", "CentaurHauls", "  Shanghai  ",
                                         "VIA VIA VIA ", "GenuineIotel", "Genuine  RDC", "TransmetaCPU", "GenuineTMx86", "Vortex86 SoC", "Microsoft Hv" };
    int k = 0; for (size_t i = 0; i < T_COUNTOF(known); i++) if (!strcmp(v, known[i])) k = 1;
    if (!k) t_note("unrecognised vendor string '%s'", v);
    t_info("vendor", "%s", v);
    t_info("max_basic", "0x%x", r.eax);
    if (r.eax > 0x100) t_note("max basic leaf 0x%x is unusually large", r.eax);
    if (cpu_hw(F_XSAVE)) CHECKF(r.eax >= 0xD, "XSAVE advertised but leaf 0Dh not reachable (max 0x%x)", r.eax);
    if (cpu_hw(F_AVX10)) CHECKF(r.eax >= 0x24, "AVX10 advertised but leaf 24h not reachable (max 0x%x)", r.eax);
}

TEST(leaf_beyond_max_basic_info) {
    const t_cpu_info *c = cpu();
    t_cpuid_regs hi = cpuidr(c->max_basic, 0), over = cpuidr(c->max_basic + 1, 0), big = cpuidr(0x3FFFFFFFu, 0);
    t_info("beyond_max", "%08x %08x %08x %08x", over.eax, over.ebx, over.ecx, over.edx);
    t_info("leaf_3fffffff", "%08x %08x %08x %08x", big.eax, big.ebx, big.ecx, big.edx);
    if (is_intel() && memcmp(&over, &hi, sizeof hi))
        t_note("Intel: leaf max+1 does not return the highest basic leaf data (SDM describes that behaviour); hypervisors differ");
    CHECK(1);
}

/* Independent signature decode per SDM (Intel: ext model for family 6 and 0Fh) / APM (AMD: ext model only for family 0Fh). */
TEST(leaf1_signature_decode) {
    const t_cpu_info *c = cpu();
    uint32_t s = cpuidr(1, 0).eax;
    CHECK_EQ(s, c->signature);
    uint32_t base_f = (s >> 8) & 0xF, base_m = (s >> 4) & 0xF, ext_f = (s >> 20) & 0xFF, ext_m = (s >> 16) & 0xF;
    uint32_t fam = base_f == 0xF ? base_f + ext_f : base_f;
    uint32_t mod = is_amd() ? (base_f == 0xF ? (ext_m << 4) | base_m : base_m)
                            : ((base_f == 6 || base_f == 0xF) ? (ext_m << 4) | base_m : base_m);
    CHECK_EQ(c->family, fam);
    CHECK_EQ(c->stepping, s & 0xF);
    if (is_amd() && base_f != 0xF && ext_m) t_note("AMD with base family %u has nonzero extended model %u", base_f, ext_m);
    if (!(is_amd() && base_f != 0xF)) CHECK_EQ(c->model, mod);
    CHECKF(base_f != 0, "base family 0 is not a valid x86 family");
    if (base_f != 0xF && ext_f) t_note("extended family %u set although base family is %u (ignored per SDM)", ext_f, base_f);
    if (s >> 28) t_note("CPUID.1:EAX reserved bits [31:28] = %x", s >> 28);
    if (fam < 6) t_note("display family %u < 6 on an SSE2-capable processor", fam);
    t_info("signature", "0x%08x family=0x%x model=0x%x stepping=%u type=%u", s, fam, mod, s & 0xF, (s >> 12) & 3);
}

TEST(leaf1_ebx_fields) {
    t_cpuid_regs r = cpuidr(1, 0);
    uint32_t clf = ((r.ebx >> 8) & 0xFF) * 8, nlog = (r.ebx >> 16) & 0xFF, apic = r.ebx >> 24;
    if (bit(r.edx, 19)) {
        CHECKF(is_pow2(clf) && clf >= 16 && clf <= 512, "CLFLUSH line size %u bytes (CPUID.1:EBX[15:8]*8) must be a power of two", clf);
    }
    if (bit(r.edx, 28)) CHECKF(nlog >= 1, "HTT set but max addressable logical IDs = %u", nlog);
    t_info("clflush_line", "%u", clf);
    t_info("max_logical_ids", "%u (HTT=%d)", nlog, bit(r.edx, 28));
    t_info("initial_apic_id", "%u", apic);
    t_info("brand_index", "%u", r.ebx & 0xFF);
}

TEST(leaf1_mandatory_baseline) {
    /* This exe runs SSE2 code (x86: -msse2; x64: SSE2 is architectural), so these must be advertised. */
    const t_cpu_info *c = cpu();
    CHECKF(cpu_hw(F_SSE2), "SSE2 not advertised but this exe executes SSE2");
    CHECKF(cpu_hw(F_SSE), "SSE not advertised");
    CHECKF(cpu_hw(F_FXSR), "FXSR not advertised: SSE state requires CR4.OSFXSR / FXSAVE");
    CHECKF(cpu_hw(F_FPU), "x87 FPU not advertised");
#ifdef T_X64
    /* Intel 64 / AMD64 architectural baseline */
    CHECKF(cpu_hw(F_TSC), "TSC"); CHECKF(cpu_hw(F_CX8), "CMPXCHG8B"); CHECKF(cpu_hw(F_CMOV), "CMOV"); CHECKF(cpu_hw(F_MMX), "MMX");
    CHECKF(c->max_ext >= 0x80000001u, "extended leaf 80000001h missing on a 64-bit processor (max_ext 0x%x)", c->max_ext);
    CHECKF(cpu_hw(F_LM), "CPUID.80000001h:EDX.LM[29] = 0 while executing 64-bit code");
    if (!cpu_hw(F_CX16)) t_note("CMPXCHG16B not advertised (required by Windows 8.1+ x64)");
    if (!cpu_hw(F_LAHF_LM)) t_note("LAHF/SAHF in 64-bit mode not advertised (required by Windows 8.1+ x64)");
    if (!cpu_hw(F_NX)) t_note("NX not advertised (required by Windows 8+ x64)");
    if (!cpu_hw(F_PREFETCHW)) t_note("PREFETCHW not advertised (required by Windows 8.1+ x64)");
    if (c->max_ext < 0x80000008u) t_note("leaf 80000008h (address sizes) missing on a 64-bit processor");
#else
    (void)c;
    if (!cpu_hw(F_CX8)) t_note("CMPXCHG8B not advertised (required by Windows)");
#endif
}

/* Architectural / state-management implications between feature bits. */
TEST(feature_implications_architectural) {
    const t_cpu_info *c = cpu();
    CHECKF(!c->osxsave || cpu_hw(F_XSAVE), "OSXSAVE=1 but XSAVE=0: CR4.OSXSAVE cannot be set without XSAVE");
    CHECKF(!cpu_hw(F_OSPKE) || cpu_hw(F_PKU), "OSPKE=1 but PKU=0: CR4.PKE cannot be set without PKU");
    static const t_feature xs_dep[] = { F_AVX, F_AVX2, F_FMA, F_F16C, F_AVX512F, F_AMX_TILE, F_XSAVEOPT, F_XSAVEC, F_XGETBV1, F_XSAVES, F_APX_F };
    for (size_t i = 0; i < T_COUNTOF(xs_dep); i++)
        if (cpu_hw(xs_dep[i])) CHECKF(cpu_hw(F_XSAVE), "%s advertised without XSAVE (its state is XSAVE-managed)", cpu_feature_name(xs_dep[i]));
    static const t_feature z_dep[] = { F_AVX512DQ, F_AVX512IFMA, F_AVX512PF, F_AVX512ER, F_AVX512CD, F_AVX512BW, F_AVX512VL,
        F_AVX512VBMI, F_AVX512VBMI2, F_AVX512VNNI, F_AVX512BITALG, F_AVX512VPOPCNTDQ, F_AVX5124VNNIW, F_AVX5124FMAPS,
        F_AVX512VP2INTERSECT, F_AVX512FP16, F_AVX512BF16 };
    for (size_t i = 0; i < T_COUNTOF(z_dep); i++)
        if (cpu_hw(z_dep[i])) CHECKF(cpu_hw(F_AVX512F), "%s advertised without AVX512F", cpu_feature_name(z_dep[i]));
    if (cpu_hw(F_AVX)) CHECKF((c->xcr0_supported & T_XS_YMM_STATE) == T_XS_YMM_STATE, "AVX advertised but XCR0-supported mask 0x%llx lacks AVX state", (unsigned long long)c->xcr0_supported);
    if (cpu_hw(F_AVX512F)) CHECKF((c->xcr0_supported & T_XS_ZMM_STATE) == T_XS_ZMM_STATE, "AVX512F advertised but XCR0-supported 0x%llx lacks opmask/ZMM", (unsigned long long)c->xcr0_supported);
    if (cpu_hw(F_AMX_TILE)) CHECKF((c->xcr0_supported & T_XS_AMX_STATE) == T_XS_AMX_STATE, "AMX-TILE advertised but TILECFG/TILEDATA not XSAVE-supported");
    if (cpu_hw(F_PKU)) CHECKF(c->xcr0_supported & T_XS_PKRU, "PKU advertised but PKRU not an XSAVE component");
    if (cpu_hw(F_APX_F)) CHECKF(c->xcr0_supported & T_XS_APX, "APX_F advertised but component 19 not supported");
    static const t_feature amx_dep[] = { F_AMX_BF16, F_AMX_INT8, F_AMX_FP16, F_AMX_COMPLEX };
    for (size_t i = 0; i < T_COUNTOF(amx_dep); i++)
        if (cpu_hw(amx_dep[i])) CHECKF(cpu_hw(F_AMX_TILE), "%s advertised without AMX-TILE", cpu_feature_name(amx_dep[i]));
    if (cpu_hw(F_AVX10)) CHECKF(c->avx10_version >= 1, "AVX10 advertised but leaf 24h version %d", c->avx10_version);
}

/* Implications every shipping processor satisfies and software relies on; not SDM-mandated -> NOTE. */
TEST(feature_implications_conventional) {
    struct { t_feature a, b; } imp[] = {
        { F_SSE3, F_SSE2 }, { F_SSSE3, F_SSE3 }, { F_SSE41, F_SSSE3 }, { F_SSE42, F_SSE41 }, { F_AVX, F_SSE42 },
        { F_AVX2, F_AVX }, { F_FMA, F_AVX }, { F_F16C, F_AVX }, { F_AVX512F, F_AVX2 }, { F_VAES, F_AES },
        { F_VPCLMULQDQ, F_PCLMULQDQ }, { F_AES, F_SSE2 }, { F_SHA, F_SSSE3 }, { F_POPCNT, F_SSE2 }, { F_AVX512F, F_FMA },
        { F_XSAVEC, F_XSAVEOPT }, { F_RDTSCP, F_TSC }, { F_SSE2, F_SSE }, { F_SSE, F_FXSR }, { F_MMX, F_FPU }
    };
    int n = 0;
    for (size_t i = 0; i < T_COUNTOF(imp); i++)
        if (cpu_hw(imp[i].a) && !cpu_hw(imp[i].b)) { t_note("unusual: %s advertised without %s", cpu_feature_name(imp[i].a), cpu_feature_name(imp[i].b)); n++; }
    /* SSE requires FXSR (state saved by FXSAVE under CR4.OSFXSR) and SSE2 requires SSE's register file: hard. */
    CHECKF(!cpu_hw(F_SSE) || cpu_hw(F_FXSR), "SSE without FXSR");
    CHECKF(!cpu_hw(F_SSE2) || cpu_hw(F_SSE), "SSE2 without SSE");
    t_info("unusual_combinations", "%d", n);
}

/* Print the whole table and re-derive every gate independently from raw registers + XCR0. */
TEST(feature_table_and_gates) {
    const t_cpu_info *c = cpu();
    uint64_t x = c->xcr0_valid ? c->xcr0 : 0;
    int ymm = c->osxsave && (x & 6) == 6, zmm = c->osxsave && (x & 0xE6) == 0xE6, amx = c->osxsave && (x & 0x60000) == 0x60000;
    int apx = c->osxsave && (x & T_XS_APX) != 0, pku = bit(cpuidr(7, 0).ecx, 4) && c->max_basic >= 7;
    static char hw[4096], us[4096];
    cpu_features_string(hw, sizeof hw, 0); cpu_features_string(us, sizeof us, 1);
    t_info("features_hw", "%s", hw);
    t_info("features_usable", "%s", us);
    int nhw = 0, nus = 0;
    for (int f = 0; f < F_COUNT; f++) {
        const char *g = cpu_feature_gate_name((t_feature)f);
        int want = cpu_hw((t_feature)f);
        if (!strcmp(g, "osxsave")) want = want && c->osxsave;
        else if (!strcmp(g, "ymm")) want = want && ymm;
        else if (!strcmp(g, "zmm")) want = want && zmm;
        else if (!strcmp(g, "amx")) want = want && amx;
        else if (!strcmp(g, "apx")) want = want && apx;
        else if (!strcmp(g, "pku")) want = want && pku;
        else if (!strcmp(g, "priv")) want = 0;
#ifdef T_X86
        if (f == F_LM) want = 0;
#endif
        CHECKF(cpu_has((t_feature)f) == want, "%s: usable=%d but gate %s recomputes %d", cpu_feature_name((t_feature)f), cpu_has((t_feature)f), g, want);
        CHECKF(!cpu_has((t_feature)f) || cpu_hw((t_feature)f), "%s usable but not advertised", cpu_feature_name((t_feature)f));
        nhw += cpu_hw((t_feature)f); nus += cpu_has((t_feature)f);
        if (cpu_hw((t_feature)f) && !cpu_has((t_feature)f) && strcmp(g, "priv") && f != F_LM)
            t_note("%s advertised but not usable (gate %s closed)", cpu_feature_name((t_feature)f), g);
    }
    t_info("counts", "hw=%d usable=%d", nhw, nus);
    /* raw re-read of feature words matches the snapshot */
    t_cpuid_regs l1 = cpuidr(1, 0);
    CHECK_EQ(l1.ecx, c->l1.ecx); CHECK_EQ(l1.edx, c->l1.edx);
    if (c->max_basic >= 7) { t_cpuid_regs l7 = cpuidr(7, 0); CHECK_EQ(l7.ebx, c->l7_0.ebx); CHECK_EQ(l7.ecx, c->l7_0.ecx); CHECK_EQ(l7.edx, c->l7_0.edx); }
    if (c->max_ext >= 0x80000001u) { t_cpuid_regs e = cpuidr(0x80000001u, 0); CHECK_EQ(e.ecx, c->e1.ecx); CHECK_EQ(e.edx, c->e1.edx); }
}

/* ===================================================================================== */
/*  Leaf 2 / 4 / 6 / 7 / 0Bh / 14h / 15h / 16h / 1Ah                                      */
/* ===================================================================================== */
TEST(leaf2_descriptors) {
    if (!has_leaf(2)) SKIP("leaf 2 not implemented (max basic 0x%x)", cpu()->max_basic);
    t_cpuid_regs r = cpuidr(2, 0);
    if (!is_intel()) { t_info("leaf2", "%08x %08x %08x %08x", r.eax, r.ebx, r.ecx, r.edx); SKIP("leaf 2 descriptors are defined by Intel only"); }
    CHECKF((r.eax & 0xFF) == 1, "CPUID.2:AL = %u (SDM: always 01h)", r.eax & 0xFF);
    uint32_t regs[4] = { r.eax & 0xFFFFFF00u, r.ebx, r.ecx, r.edx };
    char buf[256]; size_t n = 0; int ff = 0;
    for (int i = 0; i < 4; i++) {
        if (regs[i] >> 31) continue;   /* bit 31 set: register holds no valid descriptors */
        for (int b = 0; b < 4; b++) { uint8_t d = (uint8_t)(regs[i] >> (8 * b)); if (!d) continue; if (d == 0xFF) ff = 1; n += (size_t)snprintf(buf + n, n < sizeof buf ? sizeof buf - n : 0, "%02x ", d); if (n >= sizeof buf) n = sizeof buf - 1; }
    }
    t_info("descriptors", "%s", n ? buf : "(none)");
    if (ff) CHECKF(has_leaf(4), "descriptor FFh says 'use leaf 4' but leaf 4 is not implemented");
}

typedef struct { int level, type; uint32_t line, ways, parts, sets, size, sharing; } cache_t;
static int enum_caches(cache_t *out, int max, const char **src) {
    int n = 0;
    uint32_t leaf = 0;
    if (is_amd() && has_leaf(0x8000001Du) && bit(cpu()->e1.ecx, 22)) leaf = 0x8000001Du;
    else if (!is_amd() && has_leaf(4)) leaf = 4;
    *src = leaf == 4 ? "leaf4" : leaf ? "leaf8000001D" : "none";
    if (!leaf) return 0;
    for (uint32_t s = 0; s < 32 && n < max; s++) {
        t_cpuid_regs r = cpuidr(leaf, s);
        int type = r.eax & 0x1F;
        if (!type) break;
        cache_t *c = &out[n++];
        c->type = type; c->level = (r.eax >> 5) & 7;
        c->line = (r.ebx & 0xFFF) + 1; c->parts = ((r.ebx >> 12) & 0x3FF) + 1; c->ways = (r.ebx >> 22) + 1; c->sets = r.ecx + 1;
        c->size = c->line * c->parts * c->ways * c->sets;
        c->sharing = ((r.eax >> 14) & 0xFFF) + 1;
    }
    return n;
}

TEST(leaf4_cache_parameters) {
    cache_t c[32]; const char *src;
    int n = enum_caches(c, 32, &src);
    if (!n) SKIP("no deterministic cache leaf (leaf 4 / 8000001Dh) on this vendor/CPU");
    int l1d = 0;
    for (int i = 0; i < n; i++) {
        CHECKF(c[i].type >= 1 && c[i].type <= 3, "cache %d type %d (1=data 2=instr 3=unified)", i, c[i].type);
        CHECKF(c[i].level >= 1 && c[i].level <= 4, "cache %d level %d", i, c[i].level);
        CHECKF(is_pow2(c[i].line) && c[i].line >= 16 && c[i].line <= 512, "cache %d line size %u", i, c[i].line);
        CHECKF(i == 0 || c[i].level >= c[i - 1].level, "cache %d level %d listed after level %d (subleaves ascend by level)", i, c[i].level, c[i - 1].level);
        if (c[i].level == 1 && c[i].type == 1) l1d = 1;
        t_info(c[i].type == 1 ? "data" : c[i].type == 2 ? "instr" : "unified", "L%d %u KB line %u ways %u sets %u shared_by<=%u",
               c[i].level, c[i].size / 1024, c[i].line, c[i].ways, c[i].sets, c[i].sharing);
    }
    CHECKF(l1d, "no L1 data cache enumerated by %s", src);
    t_info("source", "%s", src);
}

TEST(cache_line_consistency) {
    const t_cpu_info *c = cpu();
    uint32_t clf = ((c->l1.ebx >> 8) & 0xFF) * 8;
    cache_t ca[32]; const char *src; int n = enum_caches(ca, 32, &src);
    uint32_t l1d_line = 0;
    for (int i = 0; i < n; i++) if (ca[i].level == 1 && ca[i].type == 1) l1d_line = ca[i].line;
    if (!l1d_line && is_amd() && has_leaf(0x80000005u)) l1d_line = cpuidr(0x80000005u, 0).ecx & 0xFF;
    if (has_leaf(0x80000006u)) {
        uint32_t l2line = cpuidr(0x80000006u, 0).ecx & 0xFF;
        t_info("l2_line_80000006", "%u", l2line);
        if (l2line && l1d_line && l2line != l1d_line) t_note("L2 line (80000006h) %u != L1D line %u", l2line, l1d_line);
    }
    if (!l1d_line) SKIP("no cache enumeration leaf available");
    if (cpu_hw(F_CLFSH) && clf != l1d_line) t_note("CLFLUSH line size %u differs from L1D line size %u", clf, l1d_line);
    /* Windows GetLogicalProcessorInformation must report the processor's L1D line size */
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    REQUIREF(len > 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER, "GLPI size query: len=%lu err=%lu", len, GetLastError());
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *b = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    REQUIRE(b != NULL);
    BOOL ok = GetLogicalProcessorInformation(b, &len);
    CHECKF(ok, "GLPI err=%lu", GetLastError());
    int seen = 0; unsigned noted = 0;
    for (DWORD i = 0; ok && i < len / sizeof *b; i++) {
        if (b[i].Relationship != RelationCache) continue;
        CACHE_DESCRIPTOR *d = &b[i].Cache;
        if (d->Level == 1 && d->Type == CacheData) {
            seen++;
            CHECKF(d->LineSize == l1d_line, "GLPI L1D LineSize %u != CPUID %s line %u", d->LineSize, src, l1d_line);
        }
        CHECKF(d->Level >= 1 && d->Level <= 4, "GLPI cache level %u", d->Level);
        CHECKF(is_pow2(d->LineSize), "GLPI L%u line size %u", d->Level, d->LineSize);
        for (int k = 0; k < n; k++)
            if (ca[k].level == d->Level && ((d->Type == CacheData && ca[k].type == 1) || (d->Type == CacheInstruction && ca[k].type == 2) || (d->Type == CacheUnified && ca[k].type == 3))) {
                unsigned key = 1u << ((d->Level & 7) * 4 + ((unsigned)d->Type & 3));
                if (noted & key) break;
                if (ca[k].line != d->LineSize) { t_note("GLPI L%u line %u vs CPUID %u", d->Level, d->LineSize, ca[k].line); noted |= key; }
                /* per-core-type caches differ on hybrid parts: CPUID describes only the current core */
                else if (ca[k].size != d->Size && !cpu_hw(F_HYBRID)) { t_note("GLPI L%u type %d size %lu vs CPUID %u", d->Level, (int)d->Type, (unsigned long)d->Size, ca[k].size); noted |= key; }
                break;
            }
    }
    free(b);
    if (!seen) t_note("GLPI reports no L1 data cache");
    t_info("l1d_line", "%u", l1d_line);
}

TEST(logical_processor_counts) {
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    REQUIRE(len > 0);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *b = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    REQUIRE(b != NULL);
    REQUIREF(GetLogicalProcessorInformation(b, &len), "GLPI err=%lu", GetLastError());
    ULONG_PTR cores_mask = 0; int cores = 0, pkgs = 0, smt = 0, max_per_pkg = 0;
    for (DWORD i = 0; i < len / sizeof *b; i++) {
        if (b[i].Relationship == RelationProcessorCore) {
            CHECKF(!(cores_mask & b[i].ProcessorMask), "logical processors listed in two cores (mask 0x%llx)", (unsigned long long)b[i].ProcessorMask);
            cores_mask |= b[i].ProcessorMask; cores++;
            if (b[i].ProcessorCore.Flags & 1) smt = 1;
        } else if (b[i].Relationship == RelationProcessorPackage) {
            pkgs++;
            int c = popcount64(b[i].ProcessorMask); if (c > max_per_pkg) max_per_pkg = c;
        }
    }
    free(b);
    SYSTEM_INFO si; GetSystemInfo(&si);
    int lps = popcount64(cores_mask);
    CHECKF(cores >= 1 && pkgs >= 1, "GLPI cores=%d packages=%d", cores, pkgs);
    if (t_env_get()->active_groups <= 1)
        CHECKF((DWORD)lps == si.dwNumberOfProcessors, "GLPI logical processors %d != GetSystemInfo %lu", lps, si.dwNumberOfProcessors);
    CHECKF((si.dwActiveProcessorMask & ~cores_mask) == 0, "active mask 0x%llx has CPUs absent from GLPI 0x%llx",
           (unsigned long long)si.dwActiveProcessorMask, (unsigned long long)cores_mask);
    CHECKF((t_env_get()->affinity & ~(unsigned long long)si.dwActiveProcessorMask) == 0, "process affinity outside active mask");
    const t_cpu_info *c = cpu();
    uint32_t ids = (c->l1.ebx >> 16) & 0xFF;
    if (bit(c->l1.edx, 28) && ids) {
        uint32_t p2 = 1; while (p2 < ids) p2 <<= 1;
        if ((uint32_t)max_per_pkg > p2 && ids != 0xFF) t_note("package has %d logical CPUs but CPUID.1:EBX[23:16] addressable IDs = %u", max_per_pkg, ids);
    }
    if (smt && !bit(c->l1.edx, 28)) t_note("GLPI reports SMT cores but CPUID.1:EDX.HTT = 0");
    t_info("topology", "packages=%d cores=%d logical=%d smt=%d max_per_package=%d", pkgs, cores, lps, smt, max_per_pkg);
}

TEST(leaf6_thermal_power_info) {
    if (!has_leaf(6)) SKIP("leaf 6 not implemented");
    t_cpuid_regs r = cpuidr(6, 0);
    t_info("leaf6", "eax=%08x ebx=%08x ecx=%08x edx=%08x", r.eax, r.ebx, r.ecx, r.edx);
    t_info("turbo", "%d", bit(r.eax, 1)); t_info("arat", "%d", bit(r.eax, 2)); t_info("hwp", "%d", bit(r.eax, 7));
    t_info("hw_feedback", "%d", bit(r.eax, 19)); t_info("thread_director", "%d", bit(r.eax, 23));
    CHECK(1);
}

TEST(leaf7_subleaves) {
    const t_cpu_info *c = cpu();
    if (!has_leaf(7)) SKIP("leaf 7 not implemented (max basic 0x%x)", c->max_basic);
    t_cpuid_regs r0 = cpuidr(7, 0);
    uint32_t max = r0.eax;
    t_info("max_subleaf", "%u", max);
    CHECKF(max < 64, "leaf 7 max sub-leaf %u implausible", max);
    for (uint32_t s = 0; s <= max && s < 64; s++) {
        t_cpuid_regs r = cpuidr(7, s);
        char k[16]; snprintf(k, sizeof k, "sub%u", s);
        t_info(k, "%08x %08x %08x %08x", r.eax, r.ebx, r.ecx, r.edx);
    }
    /* SDM: invalid sub-leaf -> EAX/EBX/ECX/EDX = 0 */
    for (uint32_t s = max + 1; s < max + 4; s++) {
        t_cpuid_regs r = cpuidr(7, s);
        int z = !r.eax && !r.ebx && !r.ecx && !r.edx;
        if (is_intel()) CHECKF(z, "leaf 7 invalid sub-leaf %u returned %08x %08x %08x %08x (SDM: all zero)", s, r.eax, r.ebx, r.ecx, r.edx);
        else if (!z) t_note("leaf 7 invalid sub-leaf %u nonzero on vendor %s", s, c->vendor);
    }
    CHECK_EQ(cpuidr(7, 0).ebx, c->l7_0.ebx);
    if (max >= 1) { t_cpuid_regs r1 = cpuidr(7, 1); CHECK_EQ(r1.eax, c->l7_1.eax); CHECK_EQ(r1.edx, c->l7_1.edx); }
    else CHECKF(!c->l7_1.eax && !c->l7_1.edx, "sub-leaf 1 features recorded although max sub-leaf is 0");
}

TEST(leaf0b_1f_topology) {
    const t_cpu_info *c = cpu();
    uint32_t leaves[2] = { 0xB, 0x1F }; int any = 0;
    uintptr_t prev = t_pin_thread(__builtin_ctzll(t_env_get()->affinity ? t_env_get()->affinity : 1));
    SwitchToThread();
    for (int li = 0; li < 2; li++) {
        uint32_t L = leaves[li];
        if (!has_leaf(L)) continue;
        t_cpuid_regs r0 = cpuidr(L, 0);
        if (!(r0.ebx & 0xFFFF)) continue;   /* SDM: EBX[15:0] = 0 -> leaf not supported */
        any = 1;
        uint32_t x2 = r0.edx, last_shift = 0, prev_type = 0;
        for (uint32_t s = 0; s < 16; s++) {
            t_cpuid_regs r = cpuidr(L, s);
            uint32_t type = (r.ecx >> 8) & 0xFF;
            CHECKF(r.edx == x2, "leaf %xh sub-leaf %u EDX %u != sub-leaf 0 x2APIC ID %u", L, s, r.edx, x2);
            CHECKF((r.ecx & 0xFF) == s, "leaf %xh sub-leaf %u ECX[7:0]=%u (must echo the sub-leaf)", L, s, r.ecx & 0xFF);
            if (!type) break;
            uint32_t shift = r.eax & 0x1F;
            CHECKF(shift >= last_shift, "leaf %xh level %u shift %u < previous %u", L, s, shift, last_shift);
            CHECKF(type > prev_type, "leaf %xh level types not ascending (%u after %u)", L, type, prev_type);
            last_shift = shift; prev_type = type;
            char k[32]; snprintf(k, sizeof k, "leaf%x_level%u", L, s);
            t_info(k, "type=%u shift=%u logical=%u", type, shift, r.ebx & 0xFFFF);
        }
        if (L == 0xB && c->max_basic >= 1) {
            uint32_t apic8 = cpuidr(1, 0).ebx >> 24;
            CHECKF((x2 & 0xFF) == apic8, "x2APIC ID %u low byte != initial APIC ID %u (CPUID.1:EBX[31:24])", x2, apic8);
        }
    }
    t_unpin_thread(prev);
    if (!any) SKIP("no x2APIC topology leaf (0Bh/1Fh)");
}

TEST(leaf14_15_16_1a_info) {
    const t_cpu_info *c = cpu();
    int any = 0;
    if (cpu_hw(F_HYBRID)) CHECKF(has_leaf(0x1A), "hybrid bit set but leaf 1Ah not implemented");
    if (bit(c->l7_0.ebx, 25)) {  /* Intel PT */
        CHECKF(has_leaf(0x14), "Intel PT advertised but leaf 14h unreachable");
        if (has_leaf(0x14)) { t_cpuid_regs r = cpuidr(0x14, 0); t_info("pt_leaf14", "%08x %08x %08x %08x", r.eax, r.ebx, r.ecx, r.edx); }
        any = 1;
    }
    if (has_leaf(0x15)) {
        t_cpuid_regs r = cpuidr(0x15, 0);
        t_info("tsc_ratio", "den=%u num=%u crystal_hz=%u", r.eax, r.ebx, r.ecx);
        if (r.eax && r.ebx && r.ecx) t_info("tsc_hz", "%llu", (unsigned long long)r.ecx * r.ebx / r.eax);
        any = 1;
    }
    if (has_leaf(0x16)) { t_cpuid_regs r = cpuidr(0x16, 0); t_info("freq_mhz", "base=%u max=%u bus=%u", r.eax & 0xFFFF, r.ebx & 0xFFFF, r.ecx & 0xFFFF); any = 1; }
    if (has_leaf(0x1A)) { t_cpuid_regs r = cpuidr(0x1A, 0); t_info("core_type", "0x%02x native_model=0x%x", r.eax >> 24, r.eax & 0xFFFFFF); any = 1; }
    if (!any) t_note("none of leaves 14h/15h/16h/1Ah present");
    CHECK(1);
}

/* ===================================================================================== */
/*  Leaf 0Dh (XSAVE)                                                                      */
/* ===================================================================================== */
TEST(leafd_main) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_XSAVE)) SKIP("XSAVE not advertised");
    REQUIRE(has_leaf(0xD));
    t_cpuid_regs r0 = cpuidr(0xD, 0), r1 = cpuidr(0xD, 1);
    uint64_t user = (uint64_t)r0.edx << 32 | r0.eax, sup = (uint64_t)r1.edx << 32 | r1.ecx;
    CHECKF((user & 3) == 3, "(0Dh,0) EAX[1:0] = %llx: x87 and SSE must be supported", (unsigned long long)(user & 3));
    CHECKF(!(user & sup), "component(s) 0x%llx listed as both user (XCR0) and supervisor (XSS)", (unsigned long long)(user & sup));
    CHECKF(!(sup & 3), "x87/SSE listed as supervisor components");
    static const int sup_only[] = { 8, 10, 11, 12, 13, 14, 15, 16 };
    static const int user_only[] = { 2, 3, 4, 5, 6, 7, 9, 17, 18, 19 };
    for (size_t i = 0; i < T_COUNTOF(sup_only); i++) CHECKF(!(user >> sup_only[i] & 1), "supervisor component %d (%s) in XCR0-supported mask", sup_only[i], xstate_comp_name(sup_only[i]));
    for (size_t i = 0; i < T_COUNTOF(user_only); i++) CHECKF(!(sup >> user_only[i] & 1), "user component %d (%s) in XSS-supported mask", user_only[i], xstate_comp_name(user_only[i]));
    CHECKF(!(user >> 63 & 1) && !(sup >> 63 & 1), "bit 63 is reserved (XCOMP_BV compaction flag)");
    CHECKF(r0.ecx >= 576 && r0.ebx >= 576, "sizes EBX=%u ECX=%u below legacy+header 576", r0.ebx, r0.ecx);
    CHECKF(r0.ebx <= r0.ecx, "size for enabled (EBX %u) > size for all supported (ECX %u)", r0.ebx, r0.ecx);
    CHECKF(r0.ecx == xsave_std_size(user), "(0Dh,0) ECX %u != max(offset+size) over supported user components %u", r0.ecx, xsave_std_size(user));
    if (c->osxsave) CHECKF(r0.ebx == xsave_std_size(c->xcr0), "(0Dh,0) EBX %u != standard size computed for XCR0 0x%llx: %u", r0.ebx, (unsigned long long)c->xcr0, xsave_std_size(c->xcr0));
    CHECK_EQ(user, c->xcr0_supported); CHECK_EQ(sup, c->xss_supported);
    t_info("xcr0_supported", "0x%llx", (unsigned long long)user);
    t_info("xss_supported", "0x%llx", (unsigned long long)sup);
    t_info("sizes", "enabled=%u max=%u", r0.ebx, r0.ecx);
}

TEST(leafd_subleaf1) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_XSAVE)) SKIP("XSAVE not advertised");
    t_cpuid_regs r1 = cpuidr(0xD, 1);
    t_info("sub1_eax", "0x%08x xsaveopt=%d xsavec=%d xgetbv1=%d xsaves=%d xfd=%d", r1.eax, bit(r1.eax, 0), bit(r1.eax, 1), bit(r1.eax, 2), bit(r1.eax, 3), bit(r1.eax, 4));
    if (r1.eax >> 5) t_note("(0Dh,1) EAX reserved bits set: 0x%x", r1.eax >> 5);
    if (bit(r1.eax, 3) && !bit(r1.eax, 1)) t_note("XSAVES without XSAVEC");
    if (bit(r1.eax, 4)) {
        int any = 0; for (int i = 2; i < 63; i++) if (c->comp[i].valid && c->comp[i].xfd) any = 1;
        if (!any) t_note("XFD advertised but no component reports ECX[2] (XFD-capable)");
    }
    if (!bit(r1.eax, 1) && !bit(r1.eax, 3)) { CHECK(1); return; }
    if (!c->osxsave) { t_note("OSXSAVE=0: (0Dh,1) EBX cannot be related to XCR0"); return; }
    /* EBX = compacted size for XCR0 | IA32_XSS; XSS is unknown to user mode: EBX must equal the compacted
     * size for XCR0 | S for some S within the supervisor-supported mask. */
    uint64_t sup = c->xss_supported; int nbits = popcount64(sup), match = 0;
    uint64_t found = 0;
    if (nbits <= 12) {
        int pos[12], k = 0; for (int i = 0; i < 63; i++) if (sup >> i & 1) pos[k++] = i;
        for (uint32_t m = 0; m < (1u << nbits); m++) {
            uint64_t s = 0; for (int j = 0; j < nbits; j++) if (m >> j & 1) s |= 1ull << pos[j];
            if (xsave_compacted_size(c->xcr0 | s) == r1.ebx) { match = 1; found = s; break; }
        }
        CHECKF(match, "(0Dh,1) EBX %u matches no compacted size of XCR0 0x%llx | subset of XSS-supported 0x%llx (XCR0 alone: %u)",
               r1.ebx, (unsigned long long)c->xcr0, (unsigned long long)sup, xsave_compacted_size(c->xcr0));
        if (match) t_info("inferred_xss", "0x%llx (first matching subset)", (unsigned long long)found);
    } else CHECKF(r1.ebx >= xsave_compacted_size(c->xcr0), "(0Dh,1) EBX %u < compacted size for XCR0 %u", r1.ebx, xsave_compacted_size(c->xcr0));
    t_info("compacted_size", "ebx=%u xcr0_only=%u", r1.ebx, xsave_compacted_size(c->xcr0));
}

TEST(leafd_components) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_XSAVE)) SKIP("XSAVE not advertised");
    t_cpuid_regs r0 = cpuidr(0xD, 0);
    static const struct { int i; uint32_t size; } fixed[] = { {2,256},{3,64},{4,64},{5,64},{6,512},{7,1024},{9,8},{17,64},{18,8192},{19,128} };
    static const uint32_t intel_off[20] = { [2]=576, [3]=960, [4]=1024, [5]=1088, [6]=1152, [7]=1664, [9]=2688, [17]=2816, [18]=2880, [19]=960 };
    uint64_t all = c->xcr0_supported | c->xss_supported;
    for (int i = 2; i < 63; i++) {
        t_cpuid_regs r = cpuidr(0xD, (uint32_t)i);
        int valid = (int)(all >> i & 1);
        if (!valid) {
            int z = !r.eax && !r.ebx && !r.ecx && !r.edx;
            if (is_intel()) CHECKF(z, "(0Dh,%d) unsupported component returned %08x %08x %08x %08x (SDM: zeros)", i, r.eax, r.ebx, r.ecx, r.edx);
            else if (!z) t_note("(0Dh,%d) unsupported component nonzero: %08x %08x %08x %08x", i, r.eax, r.ebx, r.ecx, r.edx);
            continue;
        }
        int sup = (int)(c->xss_supported >> i & 1);
        CHECKF(r.eax > 0, "component %d (%s) size 0", i, xstate_comp_name(i));
        CHECKF(bit(r.ecx, 0) == sup, "component %d (%s) ECX[0]=%d but it is in the %s mask", i, xstate_comp_name(i), bit(r.ecx, 0), sup ? "XSS" : "XCR0");
        if (sup) CHECKF(r.ebx == 0, "supervisor component %d (%s) EBX offset %u (SDM: 0)", i, xstate_comp_name(i), r.ebx);
        else {
            CHECKF(r.ebx >= 576, "user component %d (%s) offset %u inside legacy area/header", i, xstate_comp_name(i), r.ebx);
            CHECKF(r.ebx + r.eax <= r0.ecx, "user component %d (%s) [%u,+%u) beyond max size %u", i, xstate_comp_name(i), r.ebx, r.eax, r0.ecx);
            if (i < 20 && intel_off[i] && r.ebx != intel_off[i]) t_note("component %d (%s) offset %u (Intel layout uses %u)", i, xstate_comp_name(i), r.ebx, intel_off[i]);
        }
        if (r.ecx >> 3) t_note("component %d ECX reserved bits 0x%x", i, r.ecx >> 3);
        for (size_t k = 0; k < T_COUNTOF(fixed); k++)
            if (fixed[k].i == i) CHECKF(r.eax == fixed[k].size, "component %d (%s) size %u, architectural size %u", i, xstate_comp_name(i), r.eax, fixed[k].size);
        CHECK_EQ(r.eax, c->comp[i].size);
        char k[32]; snprintf(k, sizeof k, "comp%d_%s", i, xstate_comp_name(i));
        t_info(k, "size=%u offset=%u %s%s%s", r.eax, r.ebx, sup ? "supervisor " : "user ", bit(r.ecx, 1) ? "align64 " : "", bit(r.ecx, 2) ? "xfd" : "");
    }
    if (all >> 2 & 1) CHECKF(cpuidr(0xD, 2).ebx == 576, "AVX state offset %u (always 576: directly after legacy area + header)", cpuidr(0xD, 2).ebx);
    /* user components supported at the same time must not overlap in the standard format
     * (APX (19) reuses the MPX (3,4) area by design and is mutually exclusive with MPX) */
    for (int i = 2; i < 63; i++) for (int j = i + 1; j < 63; j++) {
        if (!(c->xcr0_supported >> i & 1) || !(c->xcr0_supported >> j & 1)) continue;
        if (j == 19 && (i == 3 || i == 4)) continue;
        uint32_t a0 = c->comp[i].offset, a1 = a0 + c->comp[i].size, b0 = c->comp[j].offset, b1 = b0 + c->comp[j].size;
        CHECKF(a1 <= b0 || b1 <= a0, "components %d [%u,%u) and %d [%u,%u) overlap", i, a0, a1, j, b0, b1);
    }
}

TEST(leafd_invalid_subleaves_32_63) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_XSAVE)) SKIP("XSAVE not advertised");
    int nz = 0;
    for (int i = 32; i < 64; i++) {
        if ((c->xcr0_supported | c->xss_supported) >> i & 1) continue;
        t_cpuid_regs r = cpuidr(0xD, (uint32_t)i);
        int z = !r.eax && !r.ebx && !r.ecx && !r.edx;
        if (!z) nz++;
        if (is_intel()) CHECKF(z, "(0Dh,%d) invalid sub-leaf returned %08x %08x %08x %08x", i, r.eax, r.ebx, r.ecx, r.edx);
    }
    if (nz && !is_intel()) t_note("%d invalid 0Dh sub-leaves above 31 are nonzero (vendor %s)", nz, c->vendor);
}

TEST(xcr0_architectural_rules) {
    const t_cpu_info *c = cpu();
    if (!c->osxsave) SKIP("OSXSAVE = 0: XCR0 not accessible");
    uint64_t x = c->xcr0;
    t_info("xcr0", "0x%llx", (unsigned long long)x);
    CHECKF(x & 1, "XCR0[0] (x87) must be 1");
    CHECKF(!(x & ~c->xcr0_supported), "XCR0 0x%llx has bits outside the supported mask 0x%llx", (unsigned long long)x, (unsigned long long)c->xcr0_supported);
    CHECKF(!(x & 4) || (x & 2), "XCR0.AVX without SSE");
    CHECKF(((x >> 3) & 3) == 0 || ((x >> 3) & 3) == 3, "XCR0 BNDREGS/BNDCSR must be set together: 0x%llx", (unsigned long long)x);
    uint64_t z = x & 0xE0;
    CHECKF(z == 0 || z == 0xE0, "XCR0 AVX-512 bits 7:5 must be all or none: 0x%llx", (unsigned long long)x);
    CHECKF(!z || (x & 4), "XCR0 AVX-512 state without AVX state");
    CHECKF(((x >> 17) & 3) == 0 || ((x >> 17) & 3) == 3, "XCR0 TILECFG/TILEDATA must be set together");
    CHECKF(!(x & c->xss_supported), "XCR0 contains supervisor component bits 0x%llx", (unsigned long long)(x & c->xss_supported));
    if (cpu_hw(F_AVX) && !(x & 4)) t_note("AVX advertised but OS has not enabled YMM state in XCR0");
    if (cpu_hw(F_AVX512F) && z != 0xE0) t_note("AVX-512 advertised but OS has not enabled ZMM state");
    const t_env *e = t_env_get();
    if (e->api_xstate_present) {
        t_info("GetEnabledXStateFeatures", "0x%llx", (unsigned long long)e->api_xstate_features);
        if ((e->api_xstate_features & x) != x) t_note("GetEnabledXStateFeatures 0x%llx does not cover XCR0 0x%llx (judged in xstate_windows)", (unsigned long long)e->api_xstate_features, (unsigned long long)x);
    }
}

/* ===================================================================================== */
/*  OSXSAVE / XGETBV / XSETBV usability (faulting forms in children)                     */
/* ===================================================================================== */
CHILD(xgetbv_idx) {
    uint32_t idx = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 0) : 0;
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(idx));
    printf("XGETBV=%08x%08x\n", hi, lo);
    return 0;
}
CHILD(xsetbv_cpl3) {
    uint32_t lo = (uint32_t)cpu()->xcr0, hi = (uint32_t)(cpu()->xcr0 >> 32);
    __asm__ volatile("xsetbv" :: "a"(lo), "d"(hi), "c"(0) : "memory");
    printf("XSETBV_RETURNED=1\n");
    return 0;
}

TEST(osxsave_xgetbv_usable) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_XSAVE)) SKIP("XSAVE not advertised");
    if (!c->osxsave) {
        static t_child_result r;
        REQUIRE(t_child_run("xgetbv_idx", "0", 10000, &r));
        CHECKF(r.exc_code == STATUS_ILLEGAL_INSN, "OSXSAVE=0: XGETBV must #UD, got exit 0x%lx exc 0x%lx", (unsigned long)r.exit_code, (unsigned long)r.exc_code);
        SKIP("OS has not enabled XSAVE (OSXSAVE=0); #UD of XGETBV verified");
    }
    uint64_t first = t_xgetbv(0);
    CHECK_EQ(first, c->xcr0);
    int diff = 0;
    for (int i = 0; i < 1000; i++) if (t_xgetbv(0) != first) diff++;
    CHECKF(diff == 0, "XGETBV(0) changed %d times in 1000 reads", diff);
    if (cpu_hw(F_XGETBV1)) {
        uint64_t in = t_xgetbv(1);
        CHECKF(!(in & ~first), "XINUSE 0x%llx not a subset of XCR0 0x%llx", (unsigned long long)in, (unsigned long long)first);
        t_info("xinuse", "0x%llx", (unsigned long long)in);
    }
    static t_child_result r;
    REQUIRE(t_child_run("xgetbv_idx", "0", 10000, &r));
    char v[64] = "";
    CHECKF(r.exit_code == 0 && t_child_find(&r, "XGETBV=", v, sizeof v), "child XGETBV(0) failed: exit 0x%lx", (unsigned long)r.exit_code);
    CHECKF(strtoull(v, NULL, 16) == first, "child process XCR0 %s != parent 0x%llx", v, (unsigned long long)first);
}

TEST(xgetbv_invalid_index_faults) {
    if (!cpu()->osxsave) SKIP("OSXSAVE = 0");
    static t_child_result r;
    const char *idx = cpu_hw(F_XGETBV1) ? "0x7fffffff" : "1";
    REQUIRE(t_child_run("xgetbv_idx", idx, 10000, &r));
    CHECK(!r.timed_out);
    t_info("exception", "XGETBV(ECX=%s) exc=0x%lx exit=0x%lx", idx, (unsigned long)r.exc_code, (unsigned long)r.exit_code);
    /* SDM: #GP(0) for an unsupported XCR index. Windows reports a user-mode #GP as an access violation or
     * a privileged-instruction exception; either is a fault, a normal return is not. */
    CHECKF(r.exc_code == STATUS_PRIV_INSN || r.exc_code == STATUS_ACCESS_VIOL,
           "XGETBV with invalid index %s must fault (#GP), got exit 0x%lx exc 0x%lx", idx, (unsigned long)r.exit_code, (unsigned long)r.exc_code);
}

TEST(xsetbv_cpl3_faults) {
    if (!cpu()->osxsave) SKIP("OSXSAVE = 0 (XSETBV would #UD)");
    static t_child_result r;
    REQUIRE(t_child_run("xsetbv_cpl3", NULL, 10000, &r));
    CHECK(!r.timed_out);
    char v[16];
    CHECKF(!t_child_find(&r, "XSETBV_RETURNED=", v, sizeof v), "XSETBV executed at CPL3 without faulting");
    CHECKF(r.exc_code == STATUS_PRIV_INSN || r.exc_code == STATUS_ACCESS_VIOL,
           "XSETBV at CPL3 must #GP: exit 0x%lx exc 0x%lx", (unsigned long)r.exit_code, (unsigned long)r.exc_code);
    t_info("exception", "0x%lx", (unsigned long)r.exc_code);
    if (r.exc_code == STATUS_ACCESS_VIOL) t_note("XSETBV #GP reported as STATUS_ACCESS_VIOLATION (Windows reports STATUS_PRIVILEGED_INSTRUCTION for privileged opcodes)");
}

TEST(avx_state_gating) {
    const t_cpu_info *c = cpu();
    if (!cpu_hw(F_AVX)) SKIP("AVX not advertised");
    int expect = c->osxsave && (c->xcr0 & 6) == 6;
    CHECK_EQ(cpu_has(F_AVX), expect);
    CHECK_EQ(cpu_ymm_state_enabled(), expect);
    if (cpu_hw(F_AVX512F)) {
        int ez = c->osxsave && (c->xcr0 & 0xE6) == 0xE6;
        CHECK_EQ(cpu_has(F_AVX512F), ez);
        CHECK_EQ(cpu_zmm_state_enabled(), ez);
    }
    if (!expect) {
        static t_child_result r;
        REQUIRE(t_child_run("probe", "avx1", 10000, &r));
        CHECKF(r.exc_code == STATUS_ILLEGAL_INSN, "AVX with YMM state disabled must #UD, got 0x%lx", (unsigned long)r.exc_code);
        SKIP("YMM state not enabled by the OS; #UD verified");
    }
    char m[128] = "";
    CHECKF(probe_avx1(m, sizeof m), "%s", m);
    t_info("ymm", "enabled");
}

/* ===================================================================================== */
/*  Extended leaves                                                                       */
/* ===================================================================================== */
TEST(ext_max_leaf) {
    const t_cpu_info *c = cpu();
    t_cpuid_regs r = cpuidr(0x80000000u, 0);
    t_info("max_ext", "0x%08x", r.eax);
    if (!(r.eax & 0x80000000u)) {
#ifdef T_X64
        CHECKF(0, "extended leaf range absent (EAX=0x%x) on a 64-bit processor", r.eax);
#endif
        SKIP("no extended CPUID leaves");
    }
    CHECKF(r.eax < 0x80000100u, "max extended leaf 0x%x implausible", r.eax);
    CHECK_EQ(c->max_ext, r.eax);
    if (is_amd()) {
        char v[13]; memcpy(v, &r.ebx, 4); memcpy(v + 4, &r.edx, 4); memcpy(v + 8, &r.ecx, 4); v[12] = 0;
        CHECKF(!strcmp(v, c->vendor), "AMD 80000000h vendor '%s' != leaf 0 '%s'", v, c->vendor);
    }
}

TEST(ext_80000001_features) {
    if (!has_leaf(0x80000001u)) SKIP("leaf 80000001h not implemented");
    t_cpuid_regs e = cpuidr(0x80000001u, 0), l1 = cpuidr(1, 0);
    t_info("e1", "eax=%08x ebx=%08x ecx=%08x edx=%08x", e.eax, e.ebx, e.ecx, e.edx);
    if (is_amd()) {
        /* APM: Fn8000_0001_EDX bits 0-9, 12-17, 23, 24 mirror Fn0000_0001_EDX */
        uint32_t mir = 0x3FFu | (0x3Fu << 12) | (1u << 23) | (1u << 24);
        CHECKF((e.edx & mir) == (l1.edx & mir), "AMD mirrored EDX bits differ: 80000001h %08x vs 1 %08x (mask %08x)", e.edx & mir, l1.edx & mir, mir);
        CHECKF(e.eax == l1.eax || e.eax == 0, "AMD 80000001h EAX %08x should mirror signature %08x", e.eax, l1.eax);
        if (bit(e.ecx, 22) && has_leaf(0x8000001Du)) t_info("topoext", "1");
    } else if (is_intel()) {
        uint32_t intel_defined = (1u << 11) | (1u << 20) | (1u << 26) | (1u << 27) | (1u << 29);
        if (e.edx & ~intel_defined) t_note("Intel 80000001h EDX has bits outside SYSCALL/XD/1GB/RDTSCP/LM: %08x", e.edx);
    }
#ifdef T_X64
    CHECKF(bit(e.edx, 29), "LM bit clear while running 64-bit code");
    CHECKF(bit(e.edx, 11) || !is_intel(), "Intel reports SYSCALL in 64-bit mode (EDX[11])");
#endif
    if (bit(e.edx, 27)) CHECKF(cpu_hw(F_TSC), "RDTSCP advertised without TSC");
    t_info("syscall_bit", "%d (Intel reports it only when CPUID executes in 64-bit mode)", bit(e.edx, 11));
}

TEST(ext_brand_string) {
    const t_cpu_info *c = cpu();
    if (!has_leaf(0x80000004u)) SKIP("brand string leaves absent");
    char raw[49]; uint32_t w[12];
    for (int i = 0; i < 3; i++) { t_cpuid_regs r = cpuidr(0x80000002u + (uint32_t)i, 0); w[i*4] = r.eax; w[i*4+1] = r.ebx; w[i*4+2] = r.ecx; w[i*4+3] = r.edx; }
    memcpy(raw, w, 48); raw[48] = 0;
    size_t len = strnlen(raw, 48);
    CHECKF(len < 48, "brand string not NUL-terminated within 48 bytes");
    CHECKF(len > 0, "empty brand string");
    for (size_t i = 0; i < len; i++) CHECKF(raw[i] >= 0x20 && raw[i] < 0x7F, "brand byte %zu = 0x%02x not printable ASCII", i, (unsigned char)raw[i]);
    for (size_t i = len; i < 48; i++) if (raw[i]) { t_note("bytes after the brand-string terminator are nonzero"); break; }
    CHECKF(strstr(raw, c->brand) != NULL, "trimmed brand '%s' not found in raw '%s'", c->brand, raw);
    t_info("brand", "%s", c->brand);
    /* registry: Windows populates HARDWARE\DESCRIPTION\System\CentralProcessor\0 from CPUID */
    HKEY k;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &k) == ERROR_SUCCESS) {
        char name[128] = "", vend[64] = "", ident[128] = ""; DWORD sz, mhz = 0, t;
        sz = sizeof name - 1; RegQueryValueExA(k, "ProcessorNameString", NULL, &t, (BYTE *)name, &sz);
        sz = sizeof vend - 1; RegQueryValueExA(k, "VendorIdentifier", NULL, &t, (BYTE *)vend, &sz);
        sz = sizeof ident - 1; RegQueryValueExA(k, "Identifier", NULL, &t, (BYTE *)ident, &sz);
        sz = sizeof mhz; RegQueryValueExA(k, "~MHz", NULL, &t, (BYTE *)&mhz, &sz);
        RegCloseKey(k);
        t_info("reg_name", "%s", name); t_info("reg_vendor", "%s", vend); t_info("reg_identifier", "%s", ident); t_info("reg_mhz", "%lu", mhz);
        char tn[128]; snprintf(tn, sizeof tn, "%s", name);
        for (size_t n = strlen(tn); n && tn[n-1] == ' '; n--) tn[n-1] = 0;
        const char *pn = tn; while (*pn == ' ') pn++;
        if (*pn && strcmp(pn, c->brand)) t_note("registry ProcessorNameString '%s' != CPUID brand '%s'", pn, c->brand);
        if (vend[0] && strcmp(vend, c->vendor)) t_note("registry VendorIdentifier '%s' != CPUID vendor '%s'", vend, c->vendor);
        unsigned fam = 0, mod = 0, step = 0;
        const char *f = strstr(ident, "Family ");
        if (f && sscanf(f, "Family %u Model %u Stepping %u", &fam, &mod, &step) == 3 && (fam != c->family || mod != c->model || step != c->stepping))
            t_note("registry Identifier '%s' differs from CPUID family %u model %u stepping %u", ident, c->family, c->model, c->stepping);
    } else t_note("CentralProcessor\\0 registry key not present");
}

TEST(ext_80000008_address_sizes) {
    if (!has_leaf(0x80000008u)) SKIP("leaf 80000008h not implemented");
    t_cpuid_regs r = cpuidr(0x80000008u, 0);
    uint32_t phys = r.eax & 0xFF, lin = (r.eax >> 8) & 0xFF;
    t_info("address_bits", "physical=%u linear=%u guest_phys=%u", phys, lin, (r.eax >> 16) & 0xFF);
    CHECKF(phys >= 32 && phys <= 52, "physical address width %u", phys);
#ifdef T_X64
    CHECKF(lin == 48 || lin == 57, "linear address width %u (48 or 57 on 64-bit processors)", lin);
    SYSTEM_INFO si; GetSystemInfo(&si);
    unsigned long long maxa = (unsigned long long)(ULONG_PTR)si.lpMaximumApplicationAddress;
    if (lin >= 32 && lin < 64) CHECKF(maxa < (1ull << (lin - 1)), "lpMaximumApplicationAddress 0x%llx is not canonical for %u-bit linear addresses", maxa, lin);
    t_info("max_app_address", "0x%llx", maxa);
#else
    CHECKF(lin >= 32, "linear address width %u", lin);
#endif
    if (is_amd()) t_info("amd_ncores", "%u", (r.ecx & 0xFF) + 1);
}

TEST(ext_cache_tsc_leaves_info) {
    int any = 0;
    if (has_leaf(0x80000005u)) { t_cpuid_regs r = cpuidr(0x80000005u, 0); t_info("e5", "%08x %08x %08x %08x", r.eax, r.ebx, r.ecx, r.edx); any = 1;
        if (is_amd()) t_info("amd_l1d", "%u KB line %u", r.ecx >> 24, r.ecx & 0xFF); }
    if (has_leaf(0x80000006u)) { t_cpuid_regs r = cpuidr(0x80000006u, 0); t_info("l2", "%u KB line %u", r.ecx >> 16, r.ecx & 0xFF); any = 1;
        CHECKF(!(r.ecx & 0xFF) || is_pow2(r.ecx & 0xFF), "80000006h L2 line size %u", r.ecx & 0xFF);
        if (is_amd() && (r.edx >> 18)) t_info("amd_l3", "%u KB", (r.edx >> 18) * 512); }
    if (has_leaf(0x80000007u)) { t_cpuid_regs r = cpuidr(0x80000007u, 0); t_info("invariant_tsc", "%d", bit(r.edx, 8)); any = 1; }
    if (!any) SKIP("extended cache/power leaves absent");
}

TEST(hypervisor_leaves_info) {
    const t_cpu_info *c = cpu();
    t_info("hypervisor_bit", "%d", c->hypervisor);
    if (!c->hypervisor) {
        t_cpuid_regs r = cpuidr(0x40000000u, 0);
        t_info("leaf40000000_without_hv", "%08x %08x %08x %08x", r.eax, r.ebx, r.ecx, r.edx);
        CHECK(1);
        return;
    }
    t_cpuid_regs r = cpuidr(0x40000000u, 0);
    char v[13]; memcpy(v, &r.ebx, 4); memcpy(v + 4, &r.ecx, 4); memcpy(v + 8, &r.edx, 4); v[12] = 0;
    for (int i = 0; i < 12; i++) if (v[i] && (v[i] < 0x20 || v[i] >= 0x7F)) v[i] = '?';
    t_info("hv_vendor", "%s", v);
    t_info("hv_max_leaf", "0x%08x", r.eax);
    if (r.eax >= 0x40000001u && r.eax < 0x40000100u) {
        t_cpuid_regs i1 = cpuidr(0x40000001u, 0);
        char s[5]; memcpy(s, &i1.eax, 4); s[4] = 0;
        for (int i = 0; i < 4; i++) if (s[i] && (s[i] < 0x20 || s[i] >= 0x7F)) s[i] = '?';
        t_info("hv_interface", "%s (0x%08x)", s, i1.eax);
    }
    CHECK(1);
}

/* ===================================================================================== */
/*  Stability: repeated, threads, every CPU                                              */
/* ===================================================================================== */
static int volatile_field(uint32_t leaf, int reg) {
    /* fields that legitimately differ per CPU / per call */
    if (leaf == 1 && reg == CPUID_EBX) return 1;                         /* initial APIC ID */
    if ((leaf == 0xB || leaf == 0x1F || leaf == 0x80000026u) && reg == CPUID_EDX) return 1;  /* x2APIC ID */
    if (leaf == 0x8000001Eu) return 1;                                   /* AMD APIC/core/node IDs */
    if (leaf == 0x1A || leaf == 4 || leaf == 0x8000001Du) return 1;      /* hybrid core type / per-core-type caches */
    if (leaf == 6) return 1;                                             /* thermal/HW-feedback may differ by core type */
    if (leaf >= 0x40000000u && leaf < 0x50000000u) return 1;             /* hypervisor-defined */
    return 0;
}

TEST(cpuid_stable_repeated) {
    static t_cpuid_entry a[1024], b[1024];
    uintptr_t prev = t_pin_thread(__builtin_ctzll(t_env_get()->affinity ? t_env_get()->affinity : 1));
    int n = cpuid_enumerate(a, 1024);
    CHECKF(n > 3, "cpuid_enumerate returned %d entries", n);
    t_info("entries", "%d", n);
    int bad = 0, hv = 0;
    for (int it = 0; it < 100; it++) {
        int m = cpuid_enumerate(b, 1024);
        CHECKF(m == n, "enumeration %d returned %d entries (first %d)", it, m, n);
        if (m != n) break;
        for (int i = 0; i < n; i++) {
            if (a[i].leaf != b[i].leaf || a[i].subleaf != b[i].subleaf) { bad++; continue; }
            for (int r = 0; r < 4; r++) {
                if (cpuid_reg(&a[i].r, r) == cpuid_reg(&b[i].r, r)) continue;
                if (a[i].leaf >= 0x40000000u && a[i].leaf < 0x50000000u) { hv++; continue; }
                if (prev && !(a[i].leaf == 6 && r == CPUID_EAX)) {   /* pinned: everything else must be identical */
                    if (bad < 10) CHECKF(0, "leaf %x.%u reg %d changed: %08x -> %08x (pinned)", a[i].leaf, a[i].subleaf, r, cpuid_reg(&a[i].r, r), cpuid_reg(&b[i].r, r));
                    bad++;
                } else if (!volatile_field(a[i].leaf, r)) { if (bad < 10) CHECKF(0, "leaf %x.%u reg %d changed", a[i].leaf, a[i].subleaf, r); bad++; }
            }
        }
    }
    t_unpin_thread(prev);
    if (hv) t_note("%d hypervisor-leaf values changed between calls", hv);
    CHECKF(bad == 0, "%d CPUID values changed over 100 enumerations", bad);
}

typedef struct { uint32_t l1c, l1d, l7b, l7c, l7d, l71a, ld0a, ld0b, ld0c, ld1a, e1c, e1d; uint64_t xcr0; } snap_t;
static void take_snap(snap_t *s) {
    const t_cpu_info *c = cpu();
    t_cpuid_regs r = cpuidr(1, 0); s->l1c = r.ecx; s->l1d = r.edx;
    memset(&r, 0, sizeof r); if (c->max_basic >= 7) r = cpuidr(7, 0); s->l7b = r.ebx; s->l7c = r.ecx; s->l7d = r.edx;
    memset(&r, 0, sizeof r); if (c->max_basic >= 7 && c->l7_0.eax >= 1) r = cpuidr(7, 1); s->l71a = r.eax;
    memset(&r, 0, sizeof r); if (c->max_basic >= 0xD) r = cpuidr(0xD, 0); s->ld0a = r.eax; s->ld0b = r.ebx; s->ld0c = r.ecx;
    memset(&r, 0, sizeof r); if (c->max_basic >= 0xD) r = cpuidr(0xD, 1); s->ld1a = r.eax;
    memset(&r, 0, sizeof r); if (c->max_ext >= 0x80000001u) r = cpuidr(0x80000001u, 0); s->e1c = r.ecx; s->e1d = r.edx;
    s->xcr0 = c->osxsave ? t_xgetbv(0) : 0;
}
static void snap_diff(const snap_t *a, const snap_t *b, const char *where) {
    CHECKF(a->l1c == b->l1c && a->l1d == b->l1d, "%s: leaf 1 ECX/EDX %08x/%08x vs %08x/%08x", where, b->l1c, b->l1d, a->l1c, a->l1d);
    CHECKF(a->l7b == b->l7b && a->l7c == b->l7c && a->l7d == b->l7d, "%s: leaf 7.0 %08x %08x %08x vs %08x %08x %08x", where, b->l7b, b->l7c, b->l7d, a->l7b, a->l7c, a->l7d);
    CHECKF(a->l71a == b->l71a, "%s: leaf 7.1 EAX %08x vs %08x", where, b->l71a, a->l71a);
    CHECKF(a->ld0a == b->ld0a && a->ld0b == b->ld0b && a->ld0c == b->ld0c && a->ld1a == b->ld1a, "%s: leaf 0Dh differs", where);
    CHECKF(a->e1c == b->e1c && a->e1d == b->e1d, "%s: leaf 80000001h %08x/%08x vs %08x/%08x", where, b->e1c, b->e1d, a->e1c, a->e1d);
    CHECKF(a->xcr0 == b->xcr0, "%s: XCR0 0x%llx vs 0x%llx", where, (unsigned long long)b->xcr0, (unsigned long long)a->xcr0);
}

static snap_t g_ref;
static volatile LONG g_thr_bad;
static DWORD WINAPI snap_worker(void *p) {
    (void)p;
    for (int i = 0; i < 300; i++) {
        snap_t s; take_snap(&s);
        if (memcmp(&s, &g_ref, sizeof s)) { if (InterlockedIncrement(&g_thr_bad) <= 3) snap_diff(&g_ref, &s, "worker thread"); }
        if (!(i & 31)) SwitchToThread();
    }
    return 0;
}
TEST(cpuid_consistent_across_threads) {
    memset(&g_ref, 0, sizeof g_ref); take_snap(&g_ref); g_thr_bad = 0;
    HANDLE h[16]; int n = 0;
    for (int i = 0; i < 16; i++) { h[n] = CreateThread(NULL, 0, snap_worker, NULL, 0, NULL); if (h[n]) n++; }
    if (!n) SKIP("CreateThread failed: %lu", GetLastError());
    DWORD w = WaitForMultipleObjects((DWORD)n, h, TRUE, 20000);
    CHECKF(w != WAIT_TIMEOUT, "workers did not finish in 20 s");
    if (w == WAIT_TIMEOUT) for (int i = 0; i < n; i++) TerminateThread(h[i], 1);
    for (int i = 0; i < n; i++) CloseHandle(h[i]);
    CHECKF(g_thr_bad == 0, "%ld snapshots differed on worker threads", g_thr_bad);
    t_info("threads", "%d", n);
}

TEST(cpuid_consistent_across_cpus) {
    const t_env *e = t_env_get();
    unsigned long long aff = e->affinity;
    if (!aff) SKIP("process affinity mask unavailable");
    snap_t ref; take_snap(&ref);
    uint32_t apic[64]; int ncpu = 0, pinned = 0, hybrid = cpu_hw(F_HYBRID), multigroup = e->active_groups > 1;
    uint32_t x2[64]; int have_x2 = has_leaf(0xB) && (cpuidr(0xB, 0).ebx & 0xFFFF);
    char types[256] = ""; size_t tl = 0;
    for (int cpu_i = 0; cpu_i < (int)(sizeof(DWORD_PTR) * 8) && cpu_i < 64; cpu_i++) {
        if (!(aff >> cpu_i & 1)) continue;
        uintptr_t prev = t_pin_thread(cpu_i);
        if (!prev) { t_note("could not pin to CPU %d", cpu_i); continue; }
        SwitchToThread();
        DWORD cur = GetCurrentProcessorNumber();
        if (!multigroup) CHECKF(cur == (DWORD)cpu_i, "pinned to CPU %d but GetCurrentProcessorNumber() = %lu", cpu_i, cur);
        snap_t s; take_snap(&s);
        char where[32]; snprintf(where, sizeof where, "CPU %d", cpu_i);
        snap_diff(&ref, &s, where);
        apic[ncpu] = cpuidr(1, 0).ebx >> 24;
        x2[ncpu] = have_x2 ? cpuidr(0xB, 0).edx : 0;
        if (hybrid && has_leaf(0x1A) && tl < sizeof types - 8) tl += (size_t)snprintf(types + tl, sizeof types - tl, "%02x ", cpuidr(0x1A, 0).eax >> 24);
        if (cpu_has(F_RDTSCP)) {
            uint32_t aux = 0; t_rdtscp(&aux);
            if (ncpu < 4) { char k[24]; snprintf(k, sizeof k, "tsc_aux_cpu%d", cpu_i); t_info(k, "0x%x", aux); }
        }
        ncpu++; pinned++;
        t_unpin_thread(prev);
    }
    if (!pinned) SKIP("thread affinity cannot be changed");
    /* distinct logical processors have distinct APIC IDs */
    int dup = 0;
    for (int i = 0; i < ncpu; i++) for (int j = i + 1; j < ncpu; j++) {
        if (have_x2 && x2[i] == x2[j]) dup++;
        else if (!have_x2 && ncpu <= 255 && apic[i] == apic[j]) dup++;
    }
    CHECKF(dup == 0, "%d pairs of pinned CPUs reported the same APIC ID", dup);
    t_info("cpus_checked", "%d", ncpu);
    if (tl) t_info("core_types", "%s", types);
}

TEST(exec_probes_on_every_cpu) {
    unsigned long long aff = t_env_get()->affinity;
    if (!aff) SKIP("process affinity mask unavailable");
    int cpus = 0, runs = 0;
    for (int cpu_i = 0; cpu_i < 64; cpu_i++) {
        if (!(aff >> cpu_i & 1)) continue;
        uintptr_t prev = t_pin_thread(cpu_i);
        if (!prev) continue;
        SwitchToThread();
        for (size_t i = 0; i < T_COUNTOF(g_probes); i++) {
            const probe_t *p = &g_probes[i];
            char why[96];
            if (!p->fn || !cpu_has(p->f) || !probe_prereq(p, why, sizeof why)) continue;
            if (!strcmp(p->name, "rdrand") || !strcmp(p->name, "rdseed") || !strcmp(p->name, "rdpid")) continue;
            char m[256] = "";
            CHECKF(p->fn(m, sizeof m), "CPU %d %s: %s", cpu_i, p->name, m);
            runs++;
        }
        t_unpin_thread(prev);
        cpus++;
    }
    if (!cpus) SKIP("thread affinity cannot be changed");
    t_info("runs", "%d probes on %d CPUs", runs, cpus);
}

/* ===================================================================================== */
/*  IsProcessorFeaturePresent                                                            */
/* ===================================================================================== */
static int pf(DWORD f) { return IsProcessorFeaturePresent(f) ? 1 : 0; }

TEST(pf_out_of_range_and_stable) {
    /* RtlIsProcessorFeaturePresent: f >= 64 -> FALSE (decomp); docs: unsupported -> zero */
    CHECK_EQ(pf(64), 0); CHECK_EQ(pf(100), 0); CHECK_EQ(pf(1000), 0); CHECK_EQ(pf(0x7FFFFFFFu), 0); CHECK_EQ(pf(0xFFFFFFFFu), 0);
    const t_env *e = t_env_get();
    for (DWORD i = 0; i < 64; i++) CHECKF(pf(i) == e->pf[i], "PF %lu changed since startup (%d -> %d)", i, e->pf[i], pf(i));
    char buf[256]; size_t n = 0;
    for (int i = 0; i < 64; i++) if (e->pf[i]) n += (size_t)snprintf(buf + n, sizeof buf - n, "%d ", i);
    t_info("pf_present", "%s", n ? buf : "(none)");
    for (int i = 44; i < 64; i++) if (e->pf[i]) t_note("PF %d set (constant beyond PF_ARM_V82_DP; meaning depends on SDK version)", i);
}

TEST(pf_non_x86_constants_zero) {
    if (native_is_arm()) SKIP("x86 code emulated on an ARM host: ARM PF_* constants describe the native processor");
    static const int arm[] = { 4, 5, 15, 18, 19, 24, 25, 26, 27, 29, 30, 31, 34, 43 };
    for (size_t i = 0; i < T_COUNTOF(arm); i++)
        CHECKF(pf((DWORD)arm[i]) == 0, "PF %d (PowerPC/Alpha/Itanium/ARM-only constant) is set on an x86 processor", arm[i]);
}

/* PF claims an ISA -> that ISA must execute correctly. */
static void pf_claim_exec(DWORD f, const char *probe) {
    const probe_t *p = find_probe(probe);
    if (!p || !p->fn || !pf(f)) return;
    if (cpu_has(p->f)) {
        char why[96], m[256] = "";
        if (!probe_prereq(p, why, sizeof why)) return;
        CHECKF(p->fn(m, sizeof m), "PF %lu claims %s: known-answer failed: %s", f, probe, m);
    } else {
        static t_child_result r;
        if (t_child_run("probe", probe, 10000, &r))
            t_info("claimed_but_unusable_probe", "%s exit=0x%lx exc=0x%lx", probe, (unsigned long)r.exit_code, (unsigned long)r.exc_code);
    }
}
/* mode: 0 = equality (documented, long-standing), 1 = PF implies feature (reverse is NOTE) */
static void pf_vs(DWORD f, const char *pfname, int want, int mode, const char *probe) {
    int got = pf(f);
    t_info(pfname, "pf=%d cpu=%d", got, want);
    if (mode == 0) CHECKF(got == want, "IsProcessorFeaturePresent(%s=%lu) = %d but the processor/OS state says %d", pfname, f, got, want);
    else {
        CHECKF(!got || want, "IsProcessorFeaturePresent(%s=%lu) = 1 but the feature is not available (CPUID/OS state)", pfname, f);
        if (!got && want) t_note("%s not reported although available (older OS may not publish this constant)", pfname);
    }
    if (probe) pf_claim_exec(f, probe);
}
static int x64_proc(void) {
#ifdef T_X64
    return 1;
#else
    return t_env_get()->wow64;
#endif
}

TEST(pf00_precision_errata) { t_info("pf0", "%d", pf(0)); if (pf(0)) t_note("PF_FLOATING_POINT_PRECISION_ERRATA set (Pentium FDIV)"); CHECK(cpu()->family != 5 || 1); }
TEST(pf01_fp_emulated)      { if (native_is_arm()) SKIP("emulated x86 on ARM host"); CHECKF(!cpu_hw(F_FPU) || pf(1) == 0, "PF_FLOATING_POINT_EMULATED=1 with a hardware x87 FPU"); }
TEST(pf02_cmpxchg8b)        { pf_vs(2, "PF_COMPARE_EXCHANGE_DOUBLE", cpu_hw(F_CX8), 0, "cx8"); }
TEST(pf03_mmx)              { pf_vs(3, "PF_MMX_INSTRUCTIONS_AVAILABLE", cpu_hw(F_MMX), 0, "mmx"); }
TEST(pf06_xmmi_sse)         { pf_vs(6, "PF_XMMI_INSTRUCTIONS_AVAILABLE", cpu_has(F_SSE), 0, "sse1"); }
TEST(pf07_3dnow)            { pf_vs(7, "PF_3DNOW_INSTRUCTIONS_AVAILABLE", cpu_hw(F_AMD3DNOW), 1, "3dnow"); }
TEST(pf08_rdtsc)            { pf_vs(8, "PF_RDTSC_INSTRUCTION_AVAILABLE", cpu_hw(F_TSC), 0, "tsc"); }
TEST(pf09_pae) {
    int hw = bit(cpu()->l1.edx, 6);
    if (x64_proc() && !native_is_arm()) CHECKF(pf(9) == 1, "PF_PAE_ENABLED = 0 on x64 (docs: all x64 processors return nonzero)");
    pf_vs(9, "PF_PAE_ENABLED", hw, 1, NULL);
}
TEST(pf10_xmmi64_sse2)      { pf_vs(10, "PF_XMMI64_INSTRUCTIONS_AVAILABLE", cpu_has(F_SSE2), 0, "sse2"); }
TEST(pf11_sse_daz)          { pf_vs(11, "PF_SSE_DAZ_MODE_AVAILABLE", cpu_has(F_SSE) && mxcsr_daz_supported(), 0, NULL);
    if (pf(11)) { unsigned old = t_get_mxcsr(); t_set_mxcsr(old | 0x40); CHECKF(t_get_mxcsr() & 0x40, "DAZ bit did not stick"); t_set_mxcsr(old); } }
TEST(pf12_nx)               { pf_vs(12, "PF_NX_ENABLED", cpu_hw(F_NX), 1, NULL); }
TEST(pf13_sse3)             { pf_vs(13, "PF_SSE3_INSTRUCTIONS_AVAILABLE", cpu_has(F_SSE3), 0, "sse3"); }
TEST(pf14_cmpxchg16b)       { pf_vs(14, "PF_COMPARE_EXCHANGE128", cpu_hw(F_CX16), x64_proc() ? 0 : 1, "cx16"); }
TEST(pf17_xsave) {
    const t_cpu_info *c = cpu();
    CHECKF(!pf(17) || cpu_hw(F_XSAVE), "PF_XSAVE_ENABLED=1 but CPUID.1:ECX.XSAVE=0");
    CHECKF(!c->osxsave || pf(17), "OS enabled XSAVE (OSXSAVE=1) but PF_XSAVE_ENABLED=0");
    if (pf(17) && !c->osxsave) t_note("PF_XSAVE_ENABLED=1 but OSXSAVE=0");
    t_info("pf17", "%d osxsave=%d", pf(17), c->osxsave);
    if (pf(17) && c->osxsave) { char m[256] = ""; CHECKF(probe_xsave(m, sizeof m), "%s", m); }
}
TEST(pf20_21_virt_info)     { t_info("PF_SECOND_LEVEL_ADDRESS_TRANSLATION", "%d", pf(20)); t_info("PF_VIRT_FIRMWARE_ENABLED", "%d", pf(21)); t_info("PF_CHANNELS_ENABLED", "%d", pf(16)); CHECK(1); }

CHILD(fsgsbase) {
#ifdef T_X64
    uint64_t gs = 0, fs = 0, teb;
    __asm__ volatile("movq %%gs:0x30, %0" : "=r"(teb));     /* NT_TIB.Self */
    __asm__ volatile("rdgsbase %0\n\trdfsbase %1" : "=r"(gs), "=r"(fs));
    printf("GSBASE=%llx\nTEB=%llx\nFSBASE=%llx\n", (unsigned long long)gs, (unsigned long long)teb, (unsigned long long)fs);
    return gs == teb ? 0 : 1;
#else
    (void)argc; (void)argv; return 2;
#endif
}
TEST(pf22_rdwrfsgsbase) {
    pf_vs(22, "PF_RDWRFSGSBASE_AVAILABLE", cpu_hw(F_FSGSBASE), 1, NULL);
#ifdef T_X64
    if (!cpu_hw(F_FSGSBASE)) SKIP("FSGSBASE not advertised");
    static t_child_result r;
    REQUIRE(t_child_run("fsgsbase", NULL, 10000, &r));
    t_info("rdgsbase_child", "exit=0x%lx exc=0x%lx", (unsigned long)r.exit_code, (unsigned long)r.exc_code);
    if (pf(22)) CHECKF(r.exit_code == 0, "PF_RDWRFSGSBASE_AVAILABLE=1 but RDGSBASE failed or GS base != TEB (exit 0x%lx exc 0x%lx)\n%s", (unsigned long)r.exit_code, (unsigned long)r.exc_code, r.out);
    else if (r.exc_code == 0) t_note("RDGSBASE executes although PF_RDWRFSGSBASE_AVAILABLE=0");
#else
    SKIP("RDFSBASE/RDGSBASE are 64-bit-mode only");
#endif
}

CHILD(fastfail) {
    printf("FASTFAIL_GO=1\n"); fflush(stdout);
    __asm__ volatile("int $0x29" :: "c"(7) : "memory");    /* __fastfail(FAST_FAIL_FATAL_APP_EXIT) */
    printf("FASTFAIL_RETURNED=1\n");
    return 0;
}
TEST(pf23_fastfail) {
    t_info("PF_FASTFAIL_AVAILABLE", "%d", pf(23));
    if (!pf(23)) SKIP("PF_FASTFAIL_AVAILABLE = 0");
    static t_child_result r;
    REQUIRE(t_child_run("fastfail", NULL, 10000, &r));
    char v[8];
    CHECKF(!t_child_find(&r, "FASTFAIL_RETURNED=", v, sizeof v), "__fastfail returned");
    /* docs (__fastfail): the process terminates with STATUS_STACK_BUFFER_OVERRUN / FAST_FAIL code */
    CHECKF(r.exit_code == STATUS_STACK_OVERRUN, "__fastfail exit code 0x%lx (want 0xC0000409)", (unsigned long)r.exit_code);
    if (t_child_find(&r, "T_CHILD_EXCEPTION ", v, sizeof v)) t_note("__fastfail reached the unhandled-exception filter (Windows terminates without running handlers)");
}
TEST(pf28_rdrand)           { pf_vs(28, "PF_RDRAND_INSTRUCTION_AVAILABLE", cpu_hw(F_RDRAND), 1, "rdrand"); }
TEST(pf32_rdtscp)           { pf_vs(32, "PF_RDTSCP_INSTRUCTION_AVAILABLE", cpu_hw(F_RDTSCP), 1, "rdtscp"); }
TEST(pf33_rdpid)            { pf_vs(33, "PF_RDPID_INSTRUCTION_AVAILABLE", cpu_hw(F_RDPID), 1, "rdpid"); }
TEST(pf35_monitorx)         { pf_vs(35, "PF_MONITORX_INSTRUCTION_AVAILABLE", has_leaf(0x80000001u) && bit(cpu()->e1.ecx, 29), 1, NULL); }
TEST(pf36_ssse3)            { pf_vs(36, "PF_SSSE3_INSTRUCTIONS_AVAILABLE", cpu_has(F_SSSE3), 1, "ssse3"); }
TEST(pf37_sse41)            { pf_vs(37, "PF_SSE4_1_INSTRUCTIONS_AVAILABLE", cpu_has(F_SSE41), 1, "sse41"); }
TEST(pf38_sse42)            { pf_vs(38, "PF_SSE4_2_INSTRUCTIONS_AVAILABLE", cpu_has(F_SSE42), 1, "sse42"); }
TEST(pf39_avx)              { pf_vs(39, "PF_AVX_INSTRUCTIONS_AVAILABLE", cpu_has(F_AVX), 1, "avx1"); }
TEST(pf40_avx2)             { pf_vs(40, "PF_AVX2_INSTRUCTIONS_AVAILABLE", cpu_has(F_AVX2), 1, "avx2"); }
TEST(pf41_avx512f)          { pf_vs(41, "PF_AVX512F_INSTRUCTIONS_AVAILABLE", cpu_has(F_AVX512F), 1, "avx512f"); }
TEST(pf42_erms)             { pf_vs(42, "PF_ERMS_AVAILABLE", cpu_hw(F_ERMS), 1, "erms"); }

TEST(system_info_processor) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    const t_cpu_info *c = cpu();
#ifdef T_X64
    CHECK_EQ(si.wProcessorArchitecture, PROCESSOR_ARCHITECTURE_AMD64);
#else
    CHECK_EQ(si.wProcessorArchitecture, PROCESSOR_ARCHITECTURE_INTEL);
#endif
    CHECKF(si.dwNumberOfProcessors >= 1 && si.dwNumberOfProcessors <= 64, "dwNumberOfProcessors %lu", si.dwNumberOfProcessors);
    t_info("level_revision", "level=%u revision=0x%04x type=%lu", si.wProcessorLevel, si.wProcessorRevision, si.dwProcessorType);
    if (!native_is_arm()) {
        if (si.wProcessorLevel != c->family) t_note("wProcessorLevel %u != CPUID family %u", si.wProcessorLevel, c->family);
        if (si.wProcessorRevision != (WORD)(((c->model & 0xFF) << 8) | c->stepping))
            t_note("wProcessorRevision 0x%04x != model<<8|stepping 0x%04x", si.wProcessorRevision, ((c->model & 0xFF) << 8) | c->stepping);
    }
    SYSTEM_INFO ni; GetNativeSystemInfo(&ni);
    t_info("native_arch", "%u", ni.wProcessorArchitecture);
#ifdef T_X86
    if (t_env_get()->wow64 && !native_is_arm()) CHECK_EQ(ni.wProcessorArchitecture, PROCESSOR_ARCHITECTURE_AMD64);
#endif
}

/* ===================================================================================== */
/*  Per-feature execution (exec_<probe>)                                                  */
/* ===================================================================================== */
TEST(exec_fpu)        { run_probe_case("fpu"); }
TEST(exec_tsc)        { run_probe_case("tsc"); }
TEST(exec_cx8)        { run_probe_case("cx8"); }
TEST(exec_cmov)       { run_probe_case("cmov"); }
TEST(exec_clflush)    { run_probe_case("clflush"); }
TEST(exec_mmx)        { run_probe_case("mmx"); }
TEST(exec_fxsr)       { run_probe_case("fxsr"); }
TEST(exec_sse1)       { run_probe_case("sse1"); }
TEST(exec_sse2)       { run_probe_case("sse2"); }
TEST(exec_sse3)       { run_probe_case("sse3"); }
TEST(exec_ssse3)      { run_probe_case("ssse3"); }
TEST(exec_sse41)      { run_probe_case("sse41"); }
TEST(exec_sse42)      { run_probe_case("sse42"); }
TEST(exec_popcnt)     { run_probe_case("popcnt"); }
TEST(exec_lzcnt)      { run_probe_case("lzcnt"); }
TEST(exec_aes)        { run_probe_case("aes"); }
TEST(exec_pclmulqdq)  { run_probe_case("pclmulqdq"); }
TEST(exec_sha)        { run_probe_case("sha"); }
TEST(exec_xsave)      { run_probe_case("xsave"); }
TEST(exec_xsaveopt)   { run_probe_case("xsaveopt"); }
TEST(exec_xsavec)     { run_probe_case("xsavec"); }
TEST(exec_xgetbv1)    { run_probe_case("xgetbv1"); }
TEST(exec_avx1)       { run_probe_case("avx1"); }
TEST(exec_avx2)       { run_probe_case("avx2"); }
TEST(exec_fma)        { run_probe_case("fma"); }
TEST(exec_f16c)       { run_probe_case("f16c"); }
TEST(exec_bmi1)       { run_probe_case("bmi1"); }
TEST(exec_bmi2)       { run_probe_case("bmi2"); }
TEST(exec_adx)        { run_probe_case("adx"); }
TEST(exec_movbe)      { run_probe_case("movbe"); }
TEST(exec_rdrand)     { run_probe_case("rdrand"); }
TEST(exec_rdseed)     { run_probe_case("rdseed"); }
TEST(exec_rdtscp)     { run_probe_case("rdtscp"); }
TEST(exec_rdpid)      { run_probe_case("rdpid"); }
TEST(exec_cx16)       { run_probe_case("cx16"); }
TEST(exec_lahf_lm)    { run_probe_case("lahf_lm"); }
TEST(exec_gfni)       { run_probe_case("gfni"); }
TEST(exec_vaes)       { run_probe_case("vaes"); }
TEST(exec_vpclmulqdq) { run_probe_case("vpclmulqdq"); }
TEST(exec_avx_vnni)   { run_probe_case("avx_vnni"); }
TEST(exec_avx512f)    { run_probe_case("avx512f"); }
TEST(exec_avx512bw)   { run_probe_case("avx512bw"); }
TEST(exec_avx512dq)   { run_probe_case("avx512dq"); }
TEST(exec_avx512vl)   { run_probe_case("avx512vl"); }
TEST(exec_avx512cd)   { run_probe_case("avx512cd"); }
TEST(exec_avx512vpopcntdq) { run_probe_case("avx512vpopcntdq"); }
TEST(exec_avx512vnni) { run_probe_case("avx512vnni"); }
TEST(exec_avx512vbmi) { run_probe_case("avx512vbmi"); }
TEST(exec_clflushopt) { run_probe_case("clflushopt"); }
TEST(exec_clwb)       { run_probe_case("clwb"); }
TEST(exec_serialize)  { run_probe_case("serialize"); }
TEST(exec_movdiri)    { run_probe_case("movdiri"); }
TEST(exec_pku)        { run_probe_case("pku"); }
TEST(exec_prefetchw)  { run_probe_case("prefetchw"); }
TEST(exec_erms)       { run_probe_case("erms"); }
TEST(exec_clzero)     { run_probe_case("clzero"); }
TEST(exec_3dnow)      { run_probe_case("3dnow"); }

/* Diagnostic: what happens when unadvertised instructions are executed (hidden-but-working features
 * under VMs/emulators). Observation only: executing an unadvertised instruction is not a contract. */
TEST(unadvertised_probe_survey) {
    int n = 0, ud = 0, ran = 0;
    static t_child_result r;
    for (size_t i = 0; i < T_COUNTOF(g_probes); i++) {
        const probe_t *p = &g_probes[i];
        if (!p->fn || cpu_hw(p->f)) continue;
        if (!strcmp(p->name, "lzcnt") || !strcmp(p->name, "prefetchw")) continue;   /* decode as BSR / NOP-like without the feature */
        if (n >= 40) break;
        if (!t_child_run("probe", p->name, 5000, &r)) continue;
        n++;
        char k[48]; snprintf(k, sizeof k, "unadvertised_%s", p->name);
        if (r.exc_code == STATUS_ILLEGAL_INSN) { ud++; t_info(k, "#UD"); }
        else if (r.timed_out) t_info(k, "timeout");
        else if (r.exc_code) t_info(k, "exception 0x%lx", (unsigned long)r.exc_code);
        else { ran++; t_info(k, "executed (exit %lu)", (unsigned long)r.exit_code); t_note("%s executes although CPUID does not advertise it", p->name); }
    }
    t_info("survey", "%d probes: %d #UD, %d executed", n, ud, ran);
    if (!n) SKIP("every probed feature is advertised");
}
