#!/usr/bin/env python3
"""gen_corpus.py - regenerate the simd_jpeg reference corpus (HOST side, Linux).

Needs: gcc, cjpeg/djpeg (libjpeg-turbo; reference values were produced with 2.1.5), ImageMagick `convert`
(only for the CMYK negative sample). Run from anywhere:  python3 corpus/jpeg/gen_corpus.py

Produces (all committed):
  corpus/jpeg/*.jpg                         the small reference JPEGs (also embedded in the exe)
  corpus/reference_vectors/simd_jpeg_refs.txt  human-readable expected values
  tests/simd_jpeg_corpus.h                  the same data as C arrays for tests/simd_jpeg.c

Reference = libjpeg-turbo djpeg, an independent decoder:
  h_int   FNV-1a64 of `djpeg -dct int` output pixels (PPM RGB / PGM gray payload)   -> must match exactly
  h_box   same with -nosmooth (merged/box upsampling)                              -> must match exactly
  h_float `djpeg -dct float` (informational only: libjpeg-turbo's float IDCT result depends on its SIMD path)
Synthetic large images are not stored: gen_synth (built from the exe's own headers) encodes them
deterministically; their JPEG byte hash and djpeg pixel hashes are recorded and the exe re-creates them.
"""
import os, subprocess, sys, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JDIR = os.path.join(ROOT, "corpus", "jpeg")
RDIR = os.path.join(ROOT, "corpus", "reference_vectors")
HDR = os.path.join(ROOT, "tests", "simd_jpeg_corpus.h")
TMP = tempfile.mkdtemp(prefix="sjcorpus.")
GEN = os.path.join(TMP, "gen_synth")


def run(cmd, inp=None, check=True):
    r = subprocess.run(cmd, input=inp, capture_output=True)
    if check and r.returncode != 0:
        sys.exit("command failed: %s\n%s" % (" ".join(cmd), r.stderr.decode(errors="replace")))
    return r


def fnv64(b):
    h = 0xcbf29ce484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def pnm_payload(b):
    # P5/P6 "Pn\nW H\n255\n" + data (djpeg writes exactly this form)
    parts, i = [], 0
    while len(parts) < 4:
        while b[i:i + 1].isspace():
            i += 1
        j = i
        while not b[j:j + 1].isspace():
            j += 1
        parts.append(b[i:j]); i = j
    return int(parts[1]), int(parts[2]), b[i + 1:]


def djpeg(data, *args):
    p = os.path.join(TMP, "in.jpg")
    open(p, "wb").write(data)
    r = run(["djpeg", *args, p], check=False)
    if not r.stdout:
        return r.returncode, None
    w, h, pix = pnm_payload(r.stdout)
    return r.returncode, pix


def src(w, h, seed, gray=False):
    p = os.path.join(TMP, "src_%d_%d_%d.ppm" % (w, h, seed))
    if not os.path.exists(p):
        open(p, "wb").write(run([GEN, "rgb", str(w), str(h), str(seed)]).stdout)
    return p


def cjpeg(srcp, *args):
    return run(["cjpeg", *args, srcp]).stdout


def scans_file(name, text):
    p = os.path.join(TMP, name)
    open(p, "w").write(text)
    return p


# ---- byte-level helpers for fabricated variants ----
def segments(d):
    """list of (marker, offset, total_len_incl_marker) up to and including the first SOS header"""
    out, i = [], 2
    while i < len(d):
        assert d[i] == 0xFF
        m = d[i + 1]
        L = d[i + 2] << 8 | d[i + 3]
        out.append((m, i, L + 2))
        if m == 0xDA:
            break
        i += 2 + L
    return out


def find(d, m):
    return [s for s in segments(d) if s[0] == m]


def patch(d, off, vals):
    b = bytearray(d)
    b[off:off + len(vals)] = bytes(vals)
    return bytes(b)


def drop(d, m):
    b = d
    for (mm, off, ln) in reversed(find(d, m)):
        b = b[:off] + b[off + ln:]
    return b


def sof_off(d):
    return [s for s in segments(d) if s[0] in (0xC0, 0xC1, 0xC2)][0][1]


def fill_markers(d):
    """insert COM + APP1 junk segments, 0xFF fill bytes before markers and garbage bytes before SOS"""
    s = segments(d)
    sos = [x for x in s if x[0] == 0xDA][0][1]
    com = b"\xFF\xFE\x00\x0Esimd_jpeg :)"
    app = b"\xFF\xE1\x00\x08junk!!"
    dqt = find(d, 0xDB)[0][1]
    out = d[:dqt] + com + b"\xFF\xFF\xFF" + d[dqt:sos] + app + b"\x00\x13\x37" + b"\xFF\xFF" + d[sos:]
    return out


GROUPS = []   # (name, description, bytes, expect)


def add(name, desc, data, expect="JD_OK"):
    GROUPS.append((name, desc, data, expect))


def build():
    run(["gcc", "-O2", "-I", os.path.join(ROOT, "tests"), "-o", GEN, os.path.join(JDIR, "gen_synth.c")])
    sp = lambda w, h, s=7: src(w, h, s)
    # tiny / odd geometry
    for (w, h) in ((8, 8), (16, 16), (17, 13), (1, 1), (3, 77), (77, 3)):
        tag = "%dx%d" % (w, h)
        add("gray_" + tag, "grayscale %s" % tag, cjpeg(sp(w, h), "-grayscale"))
        add("c444_" + tag, "4:4:4 %s" % tag, cjpeg(sp(w, h), "-sample", "1x1"))
        add("c420_" + tag, "4:2:0 %s" % tag, cjpeg(sp(w, h), "-sample", "2x2"))
        if (w, h) in ((16, 16), (17, 13), (3, 77)):
            add("c422_" + tag, "4:2:2 %s" % tag, cjpeg(sp(w, h), "-sample", "2x1"))
            add("c440_" + tag, "4:4:0 %s" % tag, cjpeg(sp(w, h), "-sample", "1x2"))
    m = sp(96, 72, 11)
    add("c420_96x72", "4:2:0 q75 baseline", cjpeg(m, "-sample", "2x2"))
    add("c422_96x72", "4:2:2", cjpeg(m, "-sample", "2x1"))
    add("c444_96x72", "4:4:4", cjpeg(m, "-sample", "1x1"))
    add("c440_96x72", "4:4:0 (h1v2 fancy)", cjpeg(m, "-sample", "1x2"))
    add("gray_96x72", "grayscale", cjpeg(m, "-grayscale"))
    add("c411_96x72", "4:1:1 (h4v1 box upsampling)", cjpeg(m, "-sample", "4x1"))
    add("cinv_96x72", "luma 1x1, chroma 2x2 (luma is the upsampled component)", cjpeg(m, "-sample", "1x1,2x2,2x2"))
    add("rgb_96x72", "RGB colour space (Adobe transform 0 / 'RGB' component ids)", cjpeg(m, "-rgb", "-sample", "1x1"))
    add("c420_q100_96x72", "quality 100", cjpeg(m, "-quality", "100"))
    add("c420_q5_96x72", "quality 5", cjpeg(m, "-quality", "5"))
    add("c420_q1_16bit_96x72", "quality 1, 16-bit quant tables (SOF1 extended)", cjpeg(m, "-quality", "1"))
    add("c420_opt_96x72", "optimised Huffman tables", cjpeg(m, "-optimize"))
    add("c420_rst1row_96x72", "restart every MCU row", cjpeg(m, "-restart", "1"))
    add("c420_rst1b_96x72", "restart every MCU", cjpeg(m, "-restart", "1B"))
    add("c444_rst5b_96x72", "4:4:4 restart every 5 MCUs", cjpeg(m, "-sample", "1x1", "-restart", "5B"))
    add("gray_rst2b_96x72", "grayscale restart every 2 blocks", cjpeg(m, "-grayscale", "-restart", "2B"))
    seq = scans_file("seq.txt", "0: 0-63, 0, 0;\n1: 0-63, 0, 0;\n2: 0-63, 0, 0;\n")
    add("seq_scans_96x72", "sequential, 3 non-interleaved scans", cjpeg(m, "-scans", seq))
    add("prog_420_96x72", "progressive 4:2:0 (default script: spectral selection + successive approx.)", cjpeg(m, "-progressive"))
    add("prog_444_96x72", "progressive 4:4:4", cjpeg(m, "-progressive", "-sample", "1x1"))
    add("prog_422_96x72", "progressive 4:2:2", cjpeg(m, "-progressive", "-sample", "2x1"))
    add("prog_440_96x72", "progressive 4:4:0", cjpeg(m, "-progressive", "-sample", "1x2"))
    add("prog_gray_96x72", "progressive grayscale", cjpeg(m, "-progressive", "-grayscale"))
    cust = scans_file("prog.txt",
                      "0,1,2: 0-0, 0, 2;\n0: 1-5, 0, 2;\n2: 1-63, 0, 1;\n1: 1-63, 0, 1;\n0: 6-63, 0, 2;\n"
                      "0: 1-63, 2, 1;\n0,1,2: 0-0, 2, 1;\n0,1,2: 0-0, 1, 0;\n2: 1-63, 1, 0;\n1: 1-63, 1, 0;\n0: 1-63, 1, 0;\n")
    add("prog_custom_96x72", "progressive, custom scans: Al=2 DC/AC, two refinement passes", cjpeg(m, "-scans", cust))
    add("prog_rst_96x72", "progressive with restart every 2 MCUs", cjpeg(m, "-progressive", "-restart", "2B"))
    o = sp(61, 45, 5)
    add("c420_61x45", "4:2:0 odd size", cjpeg(o, "-sample", "2x2"))
    add("prog_420_61x45", "progressive 4:2:0 odd size", cjpeg(o, "-progressive"))
    add("prog_420_17x13", "progressive 4:2:0 17x13", cjpeg(sp(17, 13), "-progressive"))
    add("prog_420_1x1", "progressive 4:2:0 1x1", cjpeg(sp(1, 1), "-progressive"))
    base = cjpeg(m, "-sample", "2x2")
    add("markers_fill_96x72", "COM/APP1 segments, FF fill bytes, garbage before SOS", fill_markers(base))
    add("c420_256x192", "4:2:0 256x192 (multi-thread / perf)", cjpeg(sp(256, 192, 3), "-sample", "2x2"))
    # ---- negative samples ----
    small = cjpeg(sp(16, 16), "-sample", "2x2")
    add("err_arith", "arithmetic coding (SOF9)", cjpeg(sp(16, 16), "-arithmetic"), "JD_ERR_UNSUPPORTED")
    add("err_sof3", "lossless SOF3 marker", patch(small, sof_off(small) + 1, [0xC3]), "JD_ERR_UNSUPPORTED")
    add("err_12bit", "12-bit precision", patch(small, sof_off(small) + 4, [12]), "JD_ERR_UNSUPPORTED")
    add("err_dnl", "height 0 (DNL)", patch(small, sof_off(small) + 5, [0, 0]), "JD_ERR_UNSUPPORTED")
    add("err_too_big", "65535x65535", patch(small, sof_off(small) + 5, [0xFF, 0xFF, 0xFF, 0xFF]), "JD_ERR_TOO_BIG")
    add("err_zero_width", "width 0", patch(small, sof_off(small) + 7, [0, 0]), "JD_ERR_BAD_SOF")
    add("err_mcu_blocks", "luma 4x4 + 2 chroma = 18 blocks per MCU (> 10)", patch(small, sof_off(small) + 11, [0x44]), "JD_ERR_BAD_SOS")
    add("mjpeg_no_dht", "DHT removed: Annex K default tables (Motion-JPEG convention, libjpeg std_huff_tables)", drop(small, 0xC4))
    add("err_no_dqt", "DQT segments removed", drop(small, 0xDB), "JD_ERR_BAD_DQT")
    add("err_no_soi", "SOI missing", small[2:], "JD_ERR_NOT_JPEG")
    sos = find(small, 0xDA)[0][1]
    add("err_sos_badcomp", "SOS names a component id absent from the frame", patch(small, sos + 5, [9]), "JD_ERR_BAD_SOS")
    add("err_dht_slot2", "SOS selects Huffman tables 2/2 that were never defined", patch(small, sos + 6, [0x22]), "JD_ERR_BAD_DHT")
    dht = find(small, 0xC4)[0][1]
    add("err_dht_oversub", "over-subscribed Huffman table", patch(small, dht + 5, [3]), "JD_ERR_BAD_DHT")
    pr = cjpeg(sp(16, 16), "-progressive")
    s2 = [s for s in segments_all(pr) if s[0] == 0xDA][1][1]   # 2nd scan = first AC scan
    ns = pr[s2 + 4]
    add("err_prog_badss", "progressive scan with Se < Ss", patch(pr, s2 + 5 + 2 * ns, [5, 2]), "JD_ERR_BAD_SOS")
    add("err_soi_eoi", "SOI + EOI only", b"\xFF\xD8\xFF\xD9", "JD_ERR_TRUNC_HEADER")
    cm = os.path.join(TMP, "cmyk.jpg")
    run(["convert", sp(16, 16), "-colorspace", "CMYK", "-quality", "80", cm])
    add("err_cmyk", "4-component CMYK (Adobe)", open(cm, "rb").read(), "JD_ERR_UNSUPPORTED")


def segments_all(d):
    """every marker segment incl. later SOS (skips entropy data)"""
    out, i = [], 2
    while i + 3 < len(d):
        if d[i] != 0xFF:
            i += 1; continue
        m = d[i + 1]
        if m in (0x00, 0xFF) or 0xD0 <= m <= 0xD7:
            i += 1 if m == 0xFF else 2; continue
        if m == 0xD9:
            break
        L = d[i + 2] << 8 | d[i + 3]
        out.append((m, i, L + 2))
        i += 2 + L
    return out


TRUNC = [  # (image, fractions)  -1 = len-2 (EOI removed), -2 = len-1
    ("c420_96x72", (0.05, 0.45, 0.6, 0.75, 0.9, -1, -2)),
    ("c420_rst1b_96x72", (0.5, 0.8)),
    ("c444_rst5b_96x72", (0.7,)),
    ("gray_96x72", (0.5, -1)),
    ("seq_scans_96x72", (0.5, 0.85)),
    ("c420_61x45", (0.66,)),
    ("prog_420_96x72", (0.4, 0.7, -1)),
    ("prog_rst_96x72", (0.55,)),
]
SYNTH = [  # name, w, h, sub(0 gray,1 444,2 420), quality, restart MCUs, seed
    ("synth_4k_420", 3840, 2160, 2, 85, 0, 1),
    ("synth_1k_444_rst", 1024, 768, 1, 90, 64, 2),
    ("synth_vga_gray", 640, 480, 0, 75, 0, 3),
    ("synth_mt_420", 512, 384, 2, 80, 8, 4),
]


def c_bytes(name, b):
    lines = []
    for i in range(0, len(b), 24):
        lines.append("    " + ",".join("%d" % x for x in b[i:i + 24]) + ",")
    return "static const uint8_t %s[%d] = {\n%s\n};\n" % (name, max(len(b), 1), "\n".join(lines) if b else "    0")


def main():
    for f in os.listdir(JDIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(JDIR, f))
    build()
    ver = run(["djpeg", "-version"], check=False).stderr.decode().strip().splitlines()[0]
    rows, out = [], []
    out.append("/* GENERATED by corpus/jpeg/gen_corpus.py - do not edit. Reference decoder: %s */\n" % ver)
    out.append("#ifndef SIMD_JPEG_CORPUS_H\n#define SIMD_JPEG_CORPUS_H\n")
    out.append("#define SJ_REF_DECODER \"%s\"\n" % ver)
    ent = []
    for (name, desc, data, expect) in GROUPS:
        open(os.path.join(JDIR, name + ".jpg"), "wb").write(data)
        rc, pi = djpeg(data, "-dct", "int")
        _, pb = djpeg(data, "-dct", "int", "-nosmooth")
        _, pf = djpeg(data, "-dct", "float")
        info = {}
        if expect == "JD_OK":
            if pi is None or pb is None:
                sys.exit("djpeg failed on %s" % name)
            r = run(["djpeg", "-dct", "int", os.path.join(JDIR, name + ".jpg")], check=False).stdout
            w, h, _ = pnm_payload(r)
            nc = 1 if r[:2] == b"P5" else 3
        else:
            w = h = nc = 0
        hi = fnv64(pi) if pi is not None else 0
        hb = fnv64(pb) if pb is not None else 0
        hf = fnv64(pf) if pf is not None else 0
        pix = pi if (pi is not None and expect == "JD_OK" and w * h <= 256) else None
        out.append(c_bytes("sj_d_" + name, data))
        if pix is not None:
            out.append(c_bytes("sj_p_" + name, pix))
        ent.append((name, desc, len(data), w, h, nc, expect, hi, hb, hf, pix is not None, rc))
        rows.append("%-24s %6d B  %4dx%-4d nc=%d expect=%-20s djpeg_rc=%d h_int=%016x h_box=%016x h_float=%016x  %s"
                    % (name, len(data), w, h, nc, expect, rc, hi, hb, hf, desc))
    out.append("typedef struct sj_corpus { const char *name, *desc; const uint8_t *data; uint32_t len; int w, h, nc, expect;\n"
               "    uint64_t h_int, h_box, h_float; const uint8_t *ref_pix; int djpeg_rc; } sj_corpus;\n")
    out.append("static const sj_corpus sj_corpus_list[] = {\n")
    for (name, desc, ln, w, h, nc, expect, hi, hb, hf, haspix, rc) in ent:
        out.append('    { "%s", "%s", sj_d_%s, %d, %d, %d, %d, %s, 0x%016xull, 0x%016xull, 0x%016xull, %s, %d },\n'
                   % (name, desc, name, ln, w, h, nc, expect, hi, hb, hf, ("sj_p_" + name) if haspix else "NULL", rc))
    out.append("};\n#define SJ_CORPUS_LIST(X) \\\n" + "".join("    X(%s) \\\n" % e[0] for e in ent) + "\n")
    # truncations
    rows.append("\n# truncated inputs: image len djpeg_rc h_int (compare=0: progressive, djpeg applies block smoothing)")
    out.append("typedef struct sj_trunc { const char *img; uint32_t len; int djpeg_ok, compare; uint64_t h_int; } sj_trunc;\n")
    out.append("static const sj_trunc sj_trunc_list[] = {\n")
    gmap = {g[0]: g[2] for g in GROUPS}
    for (img, fr) in TRUNC:
        d = gmap[img]
        for f in fr:
            L = len(d) - 2 if f == -1 else len(d) - 1 if f == -2 else int(len(d) * f)
            rc, pi = djpeg(d[:L], "-dct", "int")
            ok = pi is not None
            cmp_ = 0 if img.startswith("prog") else 1
            h = fnv64(pi) if ok else 0
            out.append('    { "%s", %d, %d, %d, 0x%016xull },\n' % (img, L, int(ok), cmp_, h))
            rows.append("%-24s len=%6d djpeg_rc=%d out=%d compare=%d h_int=%016x" % (img, L, rc, int(ok), cmp_, h))
    out.append("};\n")
    # synthetic
    rows.append("\n# synthetic (generated at run time by je_encode): name w h sub q ri seed jpeg_len jpeg_fnv h_int h_box")
    out.append("typedef struct sj_synth { const char *name; int w, h, sub, q, ri; uint32_t seed; uint32_t jpeg_len;\n"
               "    uint64_t jpeg_hash, h_int, h_box; } sj_synth;\n")
    out.append("static const sj_synth sj_synth_list[] = {\n")
    for (name, w, h, sub, q, ri, seed) in SYNTH:
        j = run([GEN, "enc", str(w), str(h), str(sub), str(q), str(ri), str(seed)]).stdout
        _, pi = djpeg(j, "-dct", "int")
        _, pb = djpeg(j, "-dct", "int", "-nosmooth")
        out.append('    { "%s", %d, %d, %d, %d, %d, %d, %d, 0x%016xull, 0x%016xull, 0x%016xull },\n'
                   % (name, w, h, sub, q, ri, seed, len(j), fnv64(j), fnv64(pi), fnv64(pb)))
        rows.append("%-20s %dx%d sub=%d q=%d ri=%d seed=%d len=%d jpeg=%016x h_int=%016x h_box=%016x"
                    % (name, w, h, sub, q, ri, seed, len(j), fnv64(j), fnv64(pi), fnv64(pb)))
    out.append("};\n")
    # golden kernel hashes
    rows.append("\n# golden scalar-kernel hashes over 4096 sjv_block() vectors per kind (seed 0x5EED0000+kind)")
    for line in run([GEN, "golden"]).stdout.decode().split():
        k, v = line.split("=")
        out.append("#define SJ_GOLD_%s %sull\n" % (k, v))
        rows.append(line)
    out.append("#endif\n")
    open(HDR, "w").write("".join(out))
    open(os.path.join(RDIR, "simd_jpeg_refs.txt"), "w").write(
        "# simd_jpeg reference vectors - generated by corpus/jpeg/gen_corpus.py with %s\n"
        "# FNV-1a64 of decoded pixels (RGB interleaved or gray, rows packed); djpeg -dct int (h_int), -nosmooth (h_box)\n"
        % ver + "\n".join(rows) + "\n")
    shutil.rmtree(TMP)
    print("corpus: %d images, %d truncations, %d synthetic" % (len(GROUPS), sum(len(f) for _, f in TRUNC), len(SYNTH)))


if __name__ == "__main__":
    main()
