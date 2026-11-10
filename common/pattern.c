/* pattern.c - see pattern.h */
#include "t.h"

uint64_t t_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
void t_rng_seed(t_rng *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) { seed += 0x9E3779B97F4A7C15ull; r->s[i] = t_mix64(seed); }
    if (!(r->s[0] | r->s[1] | r->s[2] | r->s[3])) r->s[0] = 1;
}
uint64_t t_rng_next(t_rng *r) {
    uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
uint32_t t_rng_u32(t_rng *r) { return (uint32_t)(t_rng_next(r) >> 32); }
uint64_t t_rng_range(t_rng *r, uint64_t n) {
    if (n <= 1) return 0;
    uint64_t lim = UINT64_MAX - UINT64_MAX % n, v;
    do v = t_rng_next(r); while (v >= lim);
    return v % n;
}
void t_rng_fill(t_rng *r, void *buf, size_t n) {
    unsigned char *p = (unsigned char *)buf;
    while (n >= 8) { uint64_t v = t_rng_next(r); memcpy(p, &v, 8); p += 8; n -= 8; }
    if (n) { uint64_t v = t_rng_next(r); memcpy(p, &v, n); }
}

/* byte7 = 0x80|reg, byte6 = 0x40|word, bytes0..5 = hash(tag, reg, word) */
uint64_t t_pattern_word(uint32_t tag, uint32_t reg, uint32_t word) {
    uint64_t h = t_mix64(((uint64_t)tag << 32) ^ ((uint64_t)reg << 12) ^ word ^ 0x5EED5EEDull) & 0x0000FFFFFFFFFFFFull;
    if (!h) h = 1;
    return ((uint64_t)(0x80 | (reg & 0x7F)) << 56) | ((uint64_t)(0x40 | (word & 0x3F)) << 48) | h;
}
void t_pattern_reg(void *out, size_t bytes, uint32_t tag, uint32_t reg) {
    unsigned char *p = (unsigned char *)out;
    for (uint32_t w = 0; (size_t)w * 8 < bytes; w++) {
        uint64_t v = t_pattern_word(tag, reg, w);
        size_t n = bytes - (size_t)w * 8 < 8 ? bytes - (size_t)w * 8 : 8;
        memcpy(p + (size_t)w * 8, &v, n);
    }
}
void t_pattern_regs(void *base, int nregs, size_t stride, size_t width, uint32_t tag) {
    for (int i = 0; i < nregs; i++) {
        unsigned char *p = (unsigned char *)base + (size_t)i * stride;
        memset(p, 0, stride);
        t_pattern_reg(p, width, tag, (uint32_t)i);
    }
}
int t_pattern_explain(uint64_t v, uint32_t max_tag, uint32_t *tag, uint32_t *reg, uint32_t *word) {
    if (((v >> 56) & 0x80) == 0 || ((v >> 48) & 0xC0) != 0x40) return 0;
    uint32_t r = (uint32_t)(v >> 56) & 0x7F, w = (uint32_t)(v >> 48) & 0x3F;
    for (uint32_t t = 0; t < max_tag; t++)
        if (t_pattern_word(t, r, w) == v) { *tag = t; *reg = r; *word = w; return 1; }
    return 0;
}

char *t_hex(char *buf, size_t cap, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    size_t k = 0;
    if (!cap) return buf;
    buf[0] = 0;
    for (size_t i = 0; i < n && k + 4 < cap; i++) {
        k += (size_t)snprintf(buf + k, cap - k, "%02x", b[i]);
        if (i + 1 < n && ((i + 1) % 4) == 0 && k + 1 < cap) buf[k++] = ' ', buf[k] = 0;
    }
    return buf;
}
char *t_hex_lanes(char *buf, size_t cap, const void *p, size_t n, int lb) {
    const unsigned char *b = (const unsigned char *)p;
    size_t k = 0;
    if (!cap) return buf;
    buf[0] = 0;
    if (lb != 1 && lb != 2 && lb != 4 && lb != 8) lb = 8;
    for (size_t lane = n / (size_t)lb; lane-- > 0 && k + 2 * (size_t)lb + 2 < cap;) {
        for (int j = lb - 1; j >= 0; j--) k += (size_t)snprintf(buf + k, cap - k, "%02x", b[lane * (size_t)lb + (size_t)j]);
        if (lane) buf[k++] = '_', buf[k] = 0;
    }
    return buf;
}
void t_dump_vec(const char *label, const void *p, size_t n, int verbose_only) {
    char h[400];
    t_hex_lanes(h, sizeof h, p, n, 8);
    if (verbose_only) t_log("%s: %s", label, h); else t_note("%s: %s", label, h);
}
int t_check_reg_at(const char *file, int line, const char *what, const void *got, const void *want, size_t n) {
    if (!memcmp(got, want, n)) { t_check(1, file, line, what, NULL); return 1; }
    const unsigned char *g = (const unsigned char *)got, *w = (const unsigned char *)want;
    size_t i = 0; while (g[i] == w[i]) i++;
    char a[400], b[400];
    t_hex_lanes(a, sizeof a, got, n, 8); t_hex_lanes(b, sizeof b, want, n, 8);
    uint64_t gq = 0; memcpy(&gq, g + (i & ~(size_t)7), (n - (i & ~(size_t)7)) < 8 ? n - (i & ~(size_t)7) : 8);
    uint32_t tg, rg, wd; char src[96] = "";
    if (t_pattern_explain(gq, 4096, &tg, &rg, &wd)) snprintf(src, sizeof src, "; got qword looks like pattern tag=%u reg=%u word=%u", tg, rg, wd);
    t_check(0, file, line, what, "first difference in qword %zu (byte %zu)%s\n         got:  %s\n         want: %s",
            i / 8, i, src, a, b);
    return 0;
}
