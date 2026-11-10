/* gen_synth.c - HOST tool (native Linux gcc) used by gen_corpus.py. It compiles the exact decoder /
 * encoder headers the simd_jpeg exe uses, so:
 *   gen_synth rgb <w> <h> <seed>                 synthetic source image as binary PPM on stdout
 *   gen_synth enc <w> <h> <sub> <q> <ri> <seed>  deterministic baseline JPEG on stdout (sub 0 gray,1 444,2 420)
 *   gen_synth golden                             KEY=value golden hashes of the scalar kernels
 * build: gcc -O2 -I tests -o gen_synth corpus/jpeg/gen_synth.c */
#include <stdio.h>
#include "simd_jpeg_dec.h"
#include "simd_jpeg_enc.h"

static uint64_t fnv_upd(uint64_t h, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}
int main(int argc, char **argv) {
    if (argc >= 5 && !strcmp(argv[1], "rgb")) {
        int w = atoi(argv[2]), h = atoi(argv[3]);
        uint8_t *rgb = malloc((size_t)w * h * 3);
        je_synth_rgb(rgb, w, h, (uint32_t)strtoul(argv[4], 0, 0));
        printf("P6\n%d %d\n255\n", w, h);
        fwrite(rgb, 1, (size_t)w * h * 3, stdout);
        return 0;
    }
    if (argc >= 8 && !strcmp(argv[1], "enc")) {
        int w = atoi(argv[2]), h = atoi(argv[3]);
        uint8_t *rgb = malloc((size_t)w * h * 3);
        je_synth_rgb(rgb, w, h, (uint32_t)strtoul(argv[7], 0, 0));
        size_t n = 0;
        uint8_t *j = je_encode(rgb, w, h, atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), &n);
        if (!j) return 1;
        fwrite(j, 1, n, stdout);
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "golden")) {
        int16_t coef[64] __attribute__((aligned(16)));
        uint16_t q[64]; float fq[64]; uint8_t out[64];
        for (int kind = 0; kind < SJV_KINDS; kind++) {
            uint64_t s = 0x5EED0000u + (uint64_t)kind, hi = 0xcbf29ce484222325ull, hf = hi;
            for (int i = 0; i < 4096; i++) {
                sjv_block(&s, kind, coef, q);
                jd_idct_islow_c(coef, q, out, 8); hi = fnv_upd(hi, out, 64);
                jd_make_fq(q, fq);
                jd_idct_float_c(coef, fq, out, 8); hf = fnv_upd(hf, out, 64);
            }
            printf("ISLOW%d=0x%016llx\nFLOAT%d=0x%016llx\n", kind, (unsigned long long)hi, kind, (unsigned long long)hf);
        }
        return 0;
    }
    fprintf(stderr, "usage: see header\n");
    return 2;
}
