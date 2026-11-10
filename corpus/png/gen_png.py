#!/usr/bin/env python3
"""Generate the simd_png corpus.

Writes
  corpus/png/*.png                            every corpus file (valid and corrupt)
  tests/simd_png_corpus.h                     the same files embedded as C arrays + expected results
  corpus/reference_vectors/simd_png_refs.txt  human-readable expected results

Encoding uses Python's zlib (every zlib level / strategy / window size the corpus needs) and a filter
encoder written here (every PNG filter type, per-row choice). Expected results come from an independent
Python PNG decoder (this file: zlib.decompress + unfilter + RGBA expansion written from the PNG spec),
which is itself checked against the source samples (pre-filter scanlines and RGBA computed straight from
the samples) and, for 8-bit images, against Pillow when it is installed.

Expected-result contract (mirrors tests/simd_png_dec.h): status codes PD_*; for truncated input the
number of inflated bytes equals what zlib's decompressobj produces from the available IDAT bytes; the
raw hash covers the unfiltered stream (filter bytes included) up to the last complete row.

Deterministic: no randomness besides a fixed-seed xorshift; rerunning gives byte-identical output.
usage: python3 corpus/png/gen_png.py   (from the repo root or anywhere)
"""
import os, struct, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_H = os.path.join(ROOT, "tests", "simd_png_corpus.h")
OUT_REF = os.path.join(ROOT, "corpus", "reference_vectors", "simd_png_refs.txt")

# ---------------------------------------------------------------------------------------------------
# status codes (tests/simd_png_dec.h)
PD_OK = 0
ZE_IN, ZE_HDR, ZE_BTYPE, ZE_STORED, ZE_CODES, ZE_SYM, ZE_DIST, ZE_OUT, ZE_ADLER = -1, -2, -3, -4, -5, -6, -7, -8, -9
E_ARG, E_SIG, E_CHUNK, E_CRC, E_IHDR, E_PLTE, E_LIMIT, E_NOMEM, E_TRUNC, E_ORDER, E_FILTER = -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30
E_ZSHORT, E_ZLONG, E_CRITICAL, E_NODATA = -31, -32, -33, -34
E_ZLIB = -100
F_IGNORE_CRC, F_IGNORE_ADLER = 1, 2
UNCHECKED = 0xFFFFFFFF


class Rng:
    def __init__(self, seed):
        self.s = (seed * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF or 1

    def next(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = x
        return (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def below(self, n):
        return (self.next() >> 33) % n


def fnv64(b):
    h = 0xcbf29ce484222325
    for c in b:
        h ^= c
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------------------------------------------
# encoder side
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
A7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]  # x0 y0 dx dy


def chunk(t, data):
    return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)


def ihdr(w, h, depth, ctype, interlace=0, comp=0, filt=0):
    return chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, comp, filt, interlace))


SIG = bytes([137, 80, 78, 71, 13, 10, 26, 10])


def make_samples(w, h, ctype, depth, rng, content, npal):
    ch = CHANNELS[ctype]
    maxv = (1 << depth) - 1
    px = []
    for y in range(h):
        row = []
        for x in range(w):
            s = []
            for c in range(ch):
                if content == "noise":
                    v = rng.next() & maxv
                elif content == "flat":
                    v = (c * 37 + (x // 13) * 11 + (y // 7) * 5) & maxv
                else:   # mixed gradient + small noise + occasional jumps
                    g = x * (3 + 2 * c) + y * (5 + c) + ((x * y) >> 6)
                    nz = rng.below(6) == 0
                    if depth == 16:
                        v = (g * 257 + ((rng.next() & 0x3F) - 32 if nz else 0)) & 0xFFFF
                    elif depth == 8:
                        v = (g + ((rng.next() & 7) - 4 if nz else 0)) & 0xFF
                    else:
                        v = ((g >> 2) + (1 if nz else 0)) & maxv
                    if rng.below(256) == 0:
                        v = rng.next() & maxv
                if ctype == 3:
                    v = v % npal
                s.append(v)
            row.append(s)
        px.append(row)
    return px


def pack_row(row, ctype, depth):
    out = bytearray()
    if depth >= 8:
        for s in row:
            for v in s:
                out += struct.pack(">H", v) if depth == 16 else bytes([v])
        return bytes(out)
    acc, nb = 0, 0
    for s in row:
        acc = (acc << depth) | s[0]
        nb += depth
        if nb == 8:
            out.append(acc); acc, nb = 0, 0
    if nb:
        out.append(acc << (8 - nb))
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_row(ft, cur, prev, bpp):
    n = len(cur)
    o = bytearray(n)
    for i in range(n):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ft == 0: p = 0
        elif ft == 1: p = a
        elif ft == 2: p = b
        elif ft == 3: p = (a + b) >> 1
        else: p = paeth(a, b, c)
        o[i] = (cur[i] - p) & 0xFF
    return bytes(o)


def passes(w, h, interlace):
    if not interlace:
        return [(0, 0, 1, 1, w, h)]
    res = []
    for x0, y0, dx, dy in A7:
        pw = (w - x0 + dx - 1) // dx if w > x0 else 0
        ph = (h - y0 + dy - 1) // dy if h > y0 else 0
        res.append((x0, y0, dx, dy, pw, ph))
    return res


def build_raw(px, w, h, ctype, depth, interlace, filt, rng):
    """returns (filtered stream, unfiltered stream with filter bytes) - the second is what the decoder's
    raw buffer holds after unfiltering"""
    bitspp = CHANNELS[ctype] * depth
    bpp = max(1, bitspp // 8)
    filtered, plain = bytearray(), bytearray()
    rowno = 0
    for x0, y0, dx, dy, pw, ph in passes(w, h, interlace):
        if pw == 0 or ph == 0:
            continue
        rb = (pw * bitspp + 7) // 8
        prev = bytes(rb)
        for yy in range(ph):
            y = y0 + yy * dy
            cur = pack_row([px[y][x0 + i * dx] for i in range(pw)], ctype, depth)
            assert len(cur) == rb
            if isinstance(filt, int): ft = filt
            elif filt == "cycle": ft = rowno % 5
            elif filt == "random": ft = rng.below(5)
            elif filt == "best":   # minimum sum of absolute differences, like libpng's heuristic
                best = None
                for t in range(5):
                    f = filter_row(t, cur, prev, bpp)
                    sc = sum(v if v < 128 else 256 - v for v in f)
                    if best is None or sc < best[0]: best = (sc, t)
                ft = best[1]
            else: raise ValueError(filt)
            filtered.append(ft); filtered += filter_row(ft, cur, prev, bpp)
            plain.append(ft); plain += cur
            prev = cur
            rowno += 1
    return bytes(filtered), bytes(plain)


def expected_rgba(px, w, h, ctype, depth, trns, pal):
    """RGBA straight from the samples (independent of the byte-level reference decoder)"""
    out = bytearray()
    for y in range(h):
        for x in range(w):
            s = px[y][x]
            if depth == 16:
                if ctype == 0: r = g = b = s[0]; a = 0 if trns is not None and s[0] == trns[0] else 65535
                elif ctype == 2: r, g, b = s; a = 0 if trns is not None and tuple(s) == tuple(trns) else 65535
                elif ctype == 4: r = g = b = s[0]; a = s[1]
                else: r, g, b, a = s
                out += struct.pack("<HHHH", r, g, b, a)
            else:
                if ctype == 0:
                    v = s[0] * (255 // ((1 << depth) - 1))
                    out += bytes([v, v, v, 0 if trns is not None and s[0] == trns[0] else 255])
                elif ctype == 3:
                    out += bytes(pal[s[0]]) if s[0] < len(pal) else bytes([0, 0, 0, 255])
                elif ctype == 2:
                    out += bytes(s) + bytes([0 if trns is not None and tuple(s) == tuple(trns) else 255])
                elif ctype == 4:
                    out += bytes([s[0], s[0], s[0], s[1]])
                else:
                    out += bytes(s)
    return bytes(out)


def zcompress(data, level=9, strategy=None, wbits=15, memlevel=8):
    co = zlib.compressobj(level, zlib.DEFLATED, wbits, memlevel, zlib.Z_DEFAULT_STRATEGY if strategy is None else strategy)
    return co.compress(data) + co.flush()


def split_idat(z, split):
    if split is None:
        return [z]
    if split == "zero":
        parts = [b""]
        for i in range(0, len(z), 64):
            parts += [z[i:i + 64], b""]
        return parts
    return [z[i:i + split] for i in range(0, len(z), split)] or [b""]


# ---------------------------------------------------------------------------------------------------
# reference decoder (independent of the encoder above: works from the file bytes only)
def ref_decode(data, flags=0, max_pixels=1 << 24):
    """returns dict(rc, w, h, raw_produced, raw_complete, h_raw, h_pix, raw, pix)"""
    r = dict(rc=PD_OK, raw_produced=0, raw_complete=0, raw=b"", pix=None, w=0, h=0)
    if len(data) < 8 or data[:8] != SIG:
        r["rc"] = E_SIG; return r
    pos, idat, rc = 8, bytearray(), PD_OK
    hdr = None; pal = None; npal = 0; trns_key = None; pal_alpha = None
    seen = set()
    while True:
        if len(data) - pos < 8:
            rc = E_TRUNC; break
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        t = data[pos + 4:pos + 8]
        if ln > 0x7FFFFFFF:
            rc = E_CHUNK; break
        if not all(65 <= c <= 90 or 97 <= c <= 122 for c in t):
            rc = E_CHUNK; break
        avail = len(data) - pos - 8
        d = data[pos + 8:pos + 8 + ln]
        if avail < ln + 4:
            if t == b"IDAT" and hdr is not None:
                idat += data[pos + 8:pos + 8 + min(avail, ln)]
            rc = E_TRUNC; break
        crc = struct.unpack(">I", data[pos + 8 + ln:pos + 12 + ln])[0]
        if not (flags & F_IGNORE_CRC) and zlib.crc32(t + d) & 0xFFFFFFFF != crc:
            rc = E_CRC; break
        pos += ln + 12
        if hdr is None and t != b"IHDR":
            rc = E_ORDER; break
        if t == b"IHDR":
            if hdr is not None: rc = E_ORDER; break
            if ln != 13: rc = E_IHDR; break
            w, h, depth, ctype, comp, filt, il = struct.unpack(">IIBBBBB", d)
            hdr = (w, h, depth, ctype, il)
            okd = {0: (1, 2, 4, 8, 16), 3: (1, 2, 4, 8), 2: (8, 16), 4: (8, 16), 6: (8, 16)}
            if not (0 < w <= 0x7FFFFFFF and 0 < h <= 0x7FFFFFFF) or ctype not in okd or depth not in okd[ctype] or comp or filt or il > 1:
                rc = E_IHDR; break
        elif t == b"PLTE":
            if "PLTE" in seen or "IDAT" in seen or "tRNS" in seen: rc = E_ORDER; break
            if hdr[3] in (0, 4): rc = E_PLTE; break
            if ln == 0 or ln % 3 or ln // 3 > 256 or (hdr[3] == 3 and ln // 3 > (1 << hdr[2])): rc = E_PLTE; break
            npal = ln // 3
            pal = [[d[3 * i], d[3 * i + 1], d[3 * i + 2], 255] for i in range(npal)]
        elif t == b"tRNS":
            if "tRNS" in seen or "IDAT" in seen: rc = E_ORDER; break
            if hdr[3] == 0:
                if ln != 2: rc = E_CHUNK; break
                trns_key = (struct.unpack(">H", d)[0],)
            elif hdr[3] == 2:
                if ln != 6: rc = E_CHUNK; break
                trns_key = struct.unpack(">HHH", d)
            elif hdr[3] == 3:
                if "PLTE" not in seen: rc = E_ORDER; break
                if ln > npal: rc = E_CHUNK; break
                pal_alpha = list(d)
        elif t == b"IDAT":
            if hdr[3] == 3 and "PLTE" not in seen: rc = E_PLTE; break
            idat += d
        elif t == b"IEND":
            break
        elif not (t[0] & 0x20):
            rc = E_CRITICAL; break
        seen.add(t.decode())
    if rc == PD_OK and "IDAT" not in seen:
        rc = E_NODATA
    r["rc"] = rc
    if (rc != PD_OK and rc != E_TRUNC) or hdr is None:
        return r
    w, h, depth, ctype, il = hdr
    r["w"], r["h"] = w, h
    if w * h > max_pixels:
        r["rc"] = E_LIMIT; return r
    bitspp = CHANNELS[ctype] * depth
    bpp = max(1, bitspp // 8)
    ps = [p for p in passes(w, h, il) if p[4] and p[5]]
    raw_len = sum(((p[4] * bitspp + 7) // 8 + 1) * p[5] for p in ps)
    # inflate. zlib.decompressobj gives the bytes decodable from a prefix; errors are classified by the
    # caller for crafted cases (zstatus) - here only: complete / truncated / error.
    zerr = None
    try:
        if flags & F_IGNORE_ADLER:   # raw deflate after the 2-byte header: the trailer is not checked
            do = zlib.decompressobj(-15)
            out = do.decompress(bytes(idat[2:]))
            complete = do.eof and len(do.unused_data) >= 4
        else:
            do = zlib.decompressobj()
            out = do.decompress(bytes(idat))
            complete = do.eof
    except zlib.error as e:
        out, complete, zerr = b"", False, str(e)
    produced = min(len(out), raw_len)
    if rc == PD_OK:
        if zerr is not None:
            r["rc"] = "zerr:" + zerr     # resolved by the caller's expectation
        elif len(out) > raw_len:
            r["rc"] = E_ZLONG
        elif not complete:
            r["rc"] = E_ZLIB + ZE_IN
        elif len(out) < raw_len:
            r["rc"] = E_ZSHORT
    raw = bytearray(out[:produced])
    r["raw_produced"] = produced if zerr is None else UNCHECKED
    # unfilter
    off, stop = 0, False
    for x0, y0, dx, dy, pw, ph in ps:
        if stop: break
        rb = (pw * bitspp + 7) // 8
        prev = bytes(rb)
        for yy in range(ph):
            if off + rb + 1 > produced: stop = True; break
            ft = raw[off]
            if ft > 4:
                if r["rc"] == PD_OK: r["rc"] = E_FILTER
                stop = True; break
            cur = raw[off + 1:off + 1 + rb]
            for i in range(rb):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]; c = prev[i - bpp] if i >= bpp else 0
                p = (0, a, b, (a + b) >> 1, paeth(a, b, c) if ft == 4 else 0)[ft]
                cur[i] = (cur[i] + p) & 0xFF
            raw[off + 1:off + 1 + rb] = cur
            prev = bytes(cur)
            off += rb + 1
    r["raw_complete"] = off if zerr is None else UNCHECKED
    r["raw"] = bytes(raw[:off])
    if r["rc"] != PD_OK:
        return r
    # expand
    pal4 = [[0, 0, 0, 255] for _ in range(256)]
    if pal:
        for i, p in enumerate(pal): pal4[i] = list(p)
        if pal_alpha:
            for i, a in enumerate(pal_alpha): pal4[i][3] = a
    bps = 2 if depth == 16 else 1
    img = bytearray(w * h * 4 * bps)
    off = 0
    ch = CHANNELS[ctype]
    for x0, y0, dx, dy, pw, ph in ps:
        rb = (pw * bitspp + 7) // 8
        for yy in range(ph):
            row = raw[off + 1:off + 1 + rb]; off += rb + 1
            y = y0 + yy * dy
            for i in range(pw):
                x = x0 + i * dx
                if depth == 16:
                    s = struct.unpack(">" + "H" * ch, row[i * 2 * ch:(i + 1) * 2 * ch])
                    if ctype == 0: v = (s[0], s[0], s[0], 0 if trns_key and s[0] == trns_key[0] else 65535)
                    elif ctype == 2: v = (s[0], s[1], s[2], 0 if trns_key and tuple(s) == tuple(trns_key) else 65535)
                    elif ctype == 4: v = (s[0], s[0], s[0], s[1])
                    else: v = s
                    q = struct.pack("<HHHH", *v)
                elif depth < 8:
                    bit = i * depth
                    sv = (row[bit >> 3] >> (8 - depth - (bit & 7))) & ((1 << depth) - 1)
                    if ctype == 3: q = bytes(pal4[sv])
                    else:
                        g = sv * (255 // ((1 << depth) - 1))
                        q = bytes([g, g, g, 0 if trns_key and sv == trns_key[0] else 255])
                else:
                    s = row[i * ch:(i + 1) * ch]
                    if ctype == 0: q = bytes([s[0], s[0], s[0], 0 if trns_key and s[0] == trns_key[0] else 255])
                    elif ctype == 3: q = bytes(pal4[s[0]])
                    elif ctype == 2: q = bytes(s) + bytes([0 if trns_key and tuple(s) == tuple(trns_key) else 255])
                    elif ctype == 4: q = bytes([s[0], s[0], s[0], s[1]])
                    else: q = bytes(s)
                p = (y * w + x) * 4 * bps
                img[p:p + 4 * bps] = q
    r["pix"] = bytes(img)
    return r


# ---------------------------------------------------------------------------------------------------
# corpus
IMAGES = []     # dict(name, data, flags, exp)
BASES = {}      # truncation family -> base image (truncated files are prefixes of it, not stored separately)
PIL_CHECKED = [0, 0]


def pil_check(name, data, ctype, depth, pix):
    try:
        from PIL import Image
        import io
    except Exception:
        return
    if depth == 16:
        return
    im = Image.open(io.BytesIO(data))
    im.load()
    got = im.convert("RGBA").tobytes()
    PIL_CHECKED[0] += 1
    if got != pix:
        # Pillow resolves grey tRNS keys for sub-byte depths and palette out-of-range indices in its
        # own way; colour channels must still agree
        rgb_ok = all(got[i] == pix[i] for i in range(len(pix)) if i % 4 != 3)
        if not rgb_ok:
            sys.exit(f"{name}: Pillow decodes different colour values")
        PIL_CHECKED[1] += 1


def add_image(name, data, flags=0, exp_rc=None, check_src=None, max_pixels=1 << 24):
    r = ref_decode(data, flags, max_pixels)
    rc = r["rc"]
    if isinstance(rc, str):
        if exp_rc is None:
            sys.exit(f"{name}: zlib error {rc} without an expectation")
        rc = exp_rc
    elif exp_rc is not None and rc != exp_rc:
        sys.exit(f"{name}: reference decoder rc {rc}, expected {exp_rc}")
    if check_src is not None:
        plain, rgba, ctype, depth = check_src
        if r["raw"] != plain: sys.exit(f"{name}: reference unfilter differs from the source scanlines")
        if r["pix"] != rgba: sys.exit(f"{name}: reference RGBA differs from the samples")
        pil_check(name, data, ctype, depth, r["pix"])
    if rc == PD_OK and r["pix"] is None: sys.exit(f"{name}: no pixels")
    IMAGES.append(dict(name=name, data=data, flags=flags, rc=rc, w=r["w"], h=r["h"],
                       raw_produced=r["raw_produced"], raw_complete=r["raw_complete"],
                       h_raw=fnv64(r["raw"]), h_pix=fnv64(r["pix"]) if rc == PD_OK else 0, max_pixels=max_pixels))
    return r


def png(w, h, ctype, depth, interlace=0, filt="cycle", level=9, strategy=None, wbits=15, memlevel=8, split=None,
        trns=None, npal=None, pal_alpha=None, content="mixed", seed=1, pre=(), mid=(), post=(), suggested_plte=False,
        oob=False):
    rng = Rng(seed)
    if ctype == 3 and npal is None:
        npal = 1 << depth
    sample_pal = npal if not oob else min(1 << depth, 256)
    px = make_samples(w, h, ctype, depth, rng, content, sample_pal if ctype == 3 else 0)
    filtered, plain = build_raw(px, w, h, ctype, depth, interlace, filt, rng)
    z = zcompress(filtered, level, strategy, wbits, memlevel)
    out = SIG + ihdr(w, h, depth, ctype, interlace)
    for c in pre: out += c
    pal = None
    if ctype == 3 or suggested_plte:
        n = npal if ctype == 3 else 17
        pal = [[(i * 53 + 7) & 255, (i * 97 + 31) & 255, (i * 29 + 200) & 255, 255] for i in range(n)]
        out += chunk(b"PLTE", bytes(v for p in pal for v in p[:3]))
    if ctype == 3 and pal_alpha is not None:
        out += chunk(b"tRNS", bytes(pal_alpha))
        for i, a in enumerate(pal_alpha): pal[i][3] = a
    if trns is not None and ctype in (0, 2):
        out += chunk(b"tRNS", b"".join(struct.pack(">H", v) for v in trns))
    if trns is not None and ctype in (4, 6):
        out += chunk(b"tRNS", b"\x00\x01" * (1 if ctype == 4 else 3))   # not allowed with alpha: ignored
    for c in mid: out += c
    for part in split_idat(z, split):
        out += chunk(b"IDAT", part)
    for c in post: out += c
    out += chunk(b"IEND", b"")
    rgba = expected_rgba(px, w, h, ctype, depth, trns if ctype in (0, 2) else None, [p for p in pal] if ctype == 3 else None)
    return out, (plain, rgba, ctype, depth), filtered, z


def add_png(name, *a, **k):
    data, src, _, _ = png(*a, **k)
    add_image(name, data, check_src=src)
    return data


FORMATS = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (2, 8), (2, 16), (3, 1), (3, 2), (3, 4), (3, 8), (4, 8), (4, 16), (6, 8), (6, 16)]
FNAME = {0: "g", 2: "rgb", 3: "p", 4: "ga", 6: "rgba"}


def build_valid():
    seed = 100
    # 1. every colour type / depth, odd sizes, plain and Adam7
    for ct, d in FORMATS:
        for (w, h) in [(1, 1), (3, 2), (7, 5), (17, 9), (33, 4)]:
            seed += 1
            add_png(f"fmt_{FNAME[ct]}{d}_{w}x{h}", w, h, ct, d, 0, "cycle", seed=seed)
        for (w, h) in [(1, 1), (2, 2), (5, 7), (9, 9), (17, 13)]:
            seed += 1
            add_png(f"fmt_{FNAME[ct]}{d}_{w}x{h}_i", w, h, ct, d, 1, "random", seed=seed)
    # 2. every filter type at every bytes-per-pixel class
    for ct, d, tag in [(0, 1, "bpp1sub"), (0, 4, "bpp1n4"), (0, 8, "bpp1"), (4, 8, "bpp2"), (0, 16, "bpp2g16"), (2, 8, "bpp3"),
                       (6, 8, "bpp4"), (4, 16, "bpp4ga16"), (2, 16, "bpp6"), (6, 16, "bpp8")]:
        for ft in range(5):
            seed += 1
            add_png(f"filt{ft}_{tag}_41x6", 41, 6, ct, d, 0, ft, content="noise" if ft == 4 else "mixed", seed=seed)
        seed += 1
        add_png(f"filtbest_{tag}_57x9_i", 57, 9, ct, d, 1, "best", seed=seed)
    # 3. odd and extreme sizes
    add_png("wide_rgba8_4097x2", 4097, 2, 6, 8, 0, "cycle", seed=201)
    add_png("wide_rgb8_4097x2_i", 4097, 2, 2, 8, 1, "random", seed=202)
    add_png("wide_g1_4097x2", 4097, 2, 0, 1, 0, "cycle", seed=203)
    add_png("wide_rgba16_4097x1", 4097, 1, 6, 16, 0, 4, seed=204)
    add_png("wide_p4_4095x2", 4095, 2, 3, 4, 0, "cycle", seed=205)
    add_png("wide_p8_4097x2_i", 4097, 2, 3, 8, 1, "cycle", seed=206)
    add_png("wide_ga8_4096x2", 4096, 2, 4, 8, 0, 3, seed=207)
    add_png("wide_rgb16_4097x1", 4097, 1, 2, 16, 0, 1, seed=208)
    add_png("tall_g8_1x4097", 1, 4097, 0, 8, 0, "random", seed=209)
    add_png("tall_rgba8_2x2049_i", 2, 2049, 6, 8, 1, "random", seed=210)
    add_png("tall_g2_3x1031", 3, 1031, 0, 2, 0, "cycle", seed=211)
    add_png("sq_rgb8_255x40", 255, 40, 2, 8, 0, "random", seed=212)
    add_png("sq_rgba8_129x63_i", 129, 63, 6, 8, 1, "best", seed=213)
    # 4. zlib levels, strategies, window sizes, memory levels
    for lvl in (0, 1, 6, 9):
        add_png(f"zlvl{lvl}_rgb8_64x40", 64, 40, 2, 8, 0, "random", level=lvl, seed=301)
    for nm, st in [("filtered", zlib.Z_FILTERED), ("huffonly", zlib.Z_HUFFMAN_ONLY), ("rle", zlib.Z_RLE), ("fixed", zlib.Z_FIXED)]:
        add_png(f"zstrat_{nm}_rgb8_64x40", 64, 40, 2, 8, 0, "random", strategy=st, seed=301)
    for wb in (9, 10, 12):
        add_png(f"zwbits{wb}_rgba8_64x30", 64, 30, 6, 8, 0, "cycle", wbits=wb, seed=302)
    add_png("zmem1_rgba8_64x30", 64, 30, 6, 8, 0, "cycle", memlevel=1, seed=302)
    add_png("zlvl0_mem1_multistored_g8_96x40", 96, 40, 0, 8, 0, "cycle", level=0, memlevel=1, seed=303)
    add_png("zlvl1_rgba16_70x40_i", 70, 40, 6, 16, 1, "random", level=1, seed=304)
    add_png("zlvl0_g1_33x7_i", 33, 7, 0, 1, 1, "cycle", level=0, seed=305)
    # 5. IDAT splitting and ancillary chunks
    add_png("split1_rgb8_23x11", 23, 11, 2, 8, 0, "cycle", split=1, seed=401)
    add_png("split7_ga16_19x5_i", 19, 5, 4, 16, 1, "random", split=7, seed=402)
    add_png("splitzero_rgba8_64x16", 64, 16, 6, 8, 0, "cycle", split="zero", seed=403)
    add_png("anc_chunks_rgb8_31x9", 31, 9, 2, 8, 0, "cycle", seed=404,
            pre=[chunk(b"gAMA", struct.pack(">I", 45455)), chunk(b"tEXt", b"Comment\x00simd_png corpus")],
            mid=[chunk(b"prVt", b"\x01\x02\x03"), chunk(b"zzZz", b"")],
            post=[chunk(b"tIME", b"\x07\xe6\x01\x02\x03\x04\x05"), chunk(b"teXt", b"x" * 300)])
    # 6. transparency
    add_png("trns_g8_key_40x8", 40, 8, 0, 8, 0, "cycle", trns=(9,), content="flat", seed=501)
    add_png("trns_g1_key_37x5", 37, 5, 0, 1, 0, "cycle", trns=(1,), seed=502)
    add_png("trns_g2_key_37x5_i", 37, 5, 0, 2, 1, "cycle", trns=(2,), seed=503)
    add_png("trns_g4_key_37x5", 37, 5, 0, 4, 0, "cycle", trns=(7,), seed=504)
    add_png("trns_g4_bigkey_17x3", 17, 3, 0, 4, 0, "cycle", trns=(0x107,), seed=505)
    add_png("trns_g16_key_40x8", 40, 8, 0, 16, 0, "cycle", trns=(0x0507,), content="flat", seed=506)
    add_png("trns_rgb8_key_40x8", 40, 8, 2, 8, 0, "cycle", trns=(37, 11, 5), content="flat", seed=507)
    add_png("trns_rgb16_key_40x8_i", 40, 8, 2, 16, 1, "cycle", trns=(37, 11, 5), content="flat", seed=508)
    add_png("trns_rgb8_nomatch_19x3", 19, 3, 2, 8, 0, "cycle", trns=(256, 1, 2), seed=509)
    add_png("trns_pal8_partial_50x10", 50, 10, 3, 8, 0, "cycle", npal=200, pal_alpha=[(i * 37) & 255 for i in range(77)], seed=510)
    add_png("trns_pal2_full_23x7", 23, 7, 3, 2, 0, "cycle", pal_alpha=[0, 85, 170, 255], seed=511)
    add_png("trns_ga8_ignored_9x3", 9, 3, 4, 8, 0, "cycle", trns=(1,), seed=512)
    add_png("pal8_oob_index_33x9", 33, 9, 3, 8, 0, "cycle", npal=100, oob=True, seed=513)
    add_png("pal4_oob_index_33x9_i", 33, 9, 3, 4, 1, "cycle", npal=5, oob=True, seed=514)
    add_png("rgb8_suggested_plte_21x6", 21, 6, 2, 8, 0, "cycle", suggested_plte=True, seed=515)
    # 7. workloads used by the MT / suspend / perf cases
    add_png("mt_rgba8_96x64_i", 96, 64, 6, 8, 1, "best", seed=601)
    add_png("mt_rgb8_160x120", 160, 120, 2, 8, 0, "best", seed=602)
    add_png("mt_p8_200x100", 200, 100, 3, 8, 0, "best", npal=181, pal_alpha=[255 - i for i in range(64)], seed=603)
    add_png("mt_rgba16_64x48", 64, 48, 6, 16, 0, "cycle", seed=604)
    add_png("mt_g8_96x96_lvl1", 96, 96, 0, 8, 0, "random", level=1, seed=605)


def mutate_chunk_crc_fix(data):
    """recompute every chunk CRC (used after editing chunk payloads)"""
    out, pos = bytearray(data[:8]), 8
    while pos + 8 <= len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        t = data[pos + 4:pos + 8]
        d = data[pos + 8:pos + 8 + ln]
        out += chunk(t, d)
        pos += ln + 12
    return bytes(out)


def raw_png(w, h, ctype, depth, z, interlace=0, pre=(), extra_ihdr=None):
    out = SIG + (extra_ihdr if extra_ihdr is not None else ihdr(w, h, depth, ctype, interlace))
    for c in pre: out += c
    return out + chunk(b"IDAT", z) + chunk(b"IEND", b"")


def build_errors():
    base, src, filtered, z = png(29, 13, 2, 8, 0, "cycle", seed=701)
    add_image("base_rgb8_29x13", base, check_src=src)
    basei, srci, filteredi, zi = png(31, 17, 6, 8, 1, "random", seed=702)
    add_image("base_rgba8_31x17_i", basei, check_src=srci)
    basep, srcp, _, _ = png(45, 7, 3, 4, 0, "cycle", pal_alpha=[0, 128], seed=703)
    add_image("base_p4_45x7", basep, check_src=srcp)
    BASES.update({"rgb8": "base_rgb8_29x13", "rgba8_i": "base_rgba8_31x17_i", "p4": "base_p4_45x7"})
    # signature
    add_image("err_sig_byte0", bytes([0x89 ^ 1]) + base[1:], exp_rc=E_SIG)
    add_image("err_sig_crlf", base[:4] + b"\n" + base[5:], exp_rc=E_SIG)
    add_image("err_sig_empty", b"", exp_rc=E_SIG)
    add_image("err_sig_7bytes", base[:7], exp_rc=E_SIG)
    # CRC
    def flip(data, off):
        b = bytearray(data); b[off] ^= 0x10; return bytes(b)
    idat_off = base.index(b"IDAT")
    add_image("err_crc_ihdr", flip(base, 8 + 8 + 13), exp_rc=E_CRC)
    add_image("err_crc_ihdr_data", flip(base, 8 + 8 + 2), exp_rc=E_CRC)
    add_image("err_crc_idat_data", flip(base, idat_off + 30), exp_rc=E_CRC)
    add_image("err_crc_iend", flip(base, len(base) - 1), exp_rc=E_CRC)
    add_image("ok_crc_idat_crcfield_ignored", flip(base, idat_off + 4 + struct.unpack(">I", base[idat_off - 4:idat_off])[0] + 2), flags=F_IGNORE_CRC)
    # Adler-32
    zb = bytearray(z); zb[-1] ^= 0x01
    bad_adler = raw_png(29, 13, 2, 8, bytes(zb))
    add_image("err_adler", bad_adler, exp_rc=E_ZLIB + ZE_ADLER)
    r = ref_decode(bad_adler, F_IGNORE_ADLER)
    IMAGES.append(dict(name="ok_adler_ignored", data=bad_adler, flags=F_IGNORE_ADLER, rc=PD_OK, w=29, h=13,
                       raw_produced=r["raw_produced"], raw_complete=r["raw_complete"], h_raw=fnv64(r["raw"]), h_pix=fnv64(src[1]), max_pixels=1 << 24))
    if r["pix"] != src[1]: sys.exit("ok_adler_ignored")
    # truncations of the three bases (every interesting boundary and a spread inside IDAT)
    for bn, bd in [("rgb8", base), ("rgba8_i", basei), ("p4", basep)]:
        io = bd.index(b"IDAT") - 4
        il = struct.unpack(">I", bd[io:io + 4])[0]
        cuts = sorted(set([8, 9, 15, 16, 20, 32, 33, io, io + 4, io + 8, io + 9, io + 10, io + 11, io + 8 + il - 4, io + 8 + il - 1,
                           io + 8 + il, io + 8 + il + 3, len(bd) - 12, len(bd) - 8, len(bd) - 4, len(bd) - 1] +
                          [io + 8 + (il * f) // 16 for f in range(1, 16)]))
        for c in cuts:
            if 8 <= c < len(bd):
                add_image(f"trunc_{bn}_{c}", bd[:c], exp_rc=E_TRUNC)
                IMAGES[-1]["ref"] = bn
    # zlib-level corruption inside a structurally valid PNG
    def zpng(zz, w=29, h=13, ct=2, d=8): return raw_png(w, h, ct, d, zz)
    add_image("err_z_hdr_cm7", zpng(bytes([0x77]) + z[1:]), exp_rc=E_ZLIB + ZE_HDR)
    add_image("err_z_hdr_fcheck", zpng(z[:1] + bytes([z[1] ^ 1]) + z[2:]), exp_rc=E_ZLIB + ZE_HDR)
    sync = zlib.compressobj(9); zs = sync.compress(filtered) + sync.flush(zlib.Z_SYNC_FLUSH)
    add_image("err_z_no_final_block", zpng(zs), exp_rc=E_ZLIB + ZE_IN)
    add_image("err_z_short_stream", zpng(zcompress(filtered[:-20])), exp_rc=E_ZSHORT)
    add_image("err_z_long_stream", zpng(zcompress(filtered + b"\x00" * 17)), exp_rc=E_ZLONG)
    add_image("err_z_btype3", zpng(bytes([0x78, 0x9c, 0x07]) + z[3:]), exp_rc=E_ZLIB + ZE_BTYPE)
    stored = zcompress(filtered, 0)
    sb = bytearray(stored); sb[5] ^= 0xFF   # NLEN byte
    add_image("err_z_stored_nlen", zpng(bytes(sb)), exp_rc=E_ZLIB + ZE_STORED)
    # distance too far back: literal 0 (filter byte) then a match at distance 2 (fixed Huffman)
    bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(1, 2); fixed_lit(bw, 0); fixed_match(bw, 10, 2); fixed_lit(bw, 256); bw.align(); bw.adler(b"")
    add_image("err_z_dist_too_far", zpng(bw.get()), exp_rc=E_ZLIB + ZE_DIST)
    # bad filter byte in row 5 (stream otherwise valid)
    fb = bytearray(filtered); rb = 29 * 3 + 1; fb[5 * rb] = 5
    add_image("err_filter_type5_row5", zpng(zcompress(bytes(fb))), exp_rc=E_FILTER)
    fb = bytearray(filtered); fb[0] = 255
    add_image("err_filter_type255_row0", zpng(zcompress(bytes(fb))), exp_rc=E_FILTER)
    # PNG structure
    def ih(w, h, d, ct, il=0, comp=0, flt=0): return chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, d, ct, comp, flt, il))
    add_image("err_ihdr_rgb_depth4", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(29, 13, 4, 2)), exp_rc=E_IHDR)
    add_image("err_ihdr_ctype5", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(29, 13, 8, 5)), exp_rc=E_IHDR)
    add_image("err_ihdr_pal16", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(29, 13, 16, 3)), exp_rc=E_IHDR)
    add_image("err_ihdr_interlace2", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(29, 13, 8, 2, 2)), exp_rc=E_IHDR)
    add_image("err_ihdr_comp1", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(29, 13, 8, 2, 0, 1)), exp_rc=E_IHDR)
    add_image("err_ihdr_filter1", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(29, 13, 8, 2, 0, 0, 1)), exp_rc=E_IHDR)
    add_image("err_ihdr_width0", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(0, 13, 8, 2)), exp_rc=E_IHDR)
    add_image("err_ihdr_width2g", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(0x80000000, 1, 8, 2)), exp_rc=E_IHDR)
    add_image("err_ihdr_len12", SIG + chunk(b"IHDR", struct.pack(">IIBBBB", 29, 13, 8, 2, 0, 0)) + chunk(b"IDAT", z) + chunk(b"IEND", b""), exp_rc=E_IHDR)
    add_image("err_limit_huge", raw_png(0, 0, 0, 0, z, extra_ihdr=ih(0x7FFFFFFF, 0x7FFFFFFF, 8, 6)), exp_rc=E_LIMIT)
    add_image("err_limit_small_max", base, exp_rc=E_LIMIT, max_pixels=29 * 13 - 1)
    add_image("err_order_text_first", SIG + chunk(b"tEXt", b"a\x00b") + base[8:], exp_rc=E_ORDER)
    add_image("err_order_dup_ihdr", base[:33] + base[8:], exp_rc=E_ORDER)
    add_image("err_plte_in_grey", raw_png(0, 0, 0, 0, zcompress(bytes(13 * 30)), pre=[chunk(b"PLTE", b"\x00" * 6)], extra_ihdr=ih(29, 13, 8, 0)), exp_rc=E_PLTE)
    add_image("err_plte_len_not3", raw_png(0, 0, 0, 0, zcompress(bytes(13 * 16)), pre=[chunk(b"PLTE", b"\x00" * 7)], extra_ihdr=ih(29, 13, 4, 3)), exp_rc=E_PLTE)
    add_image("err_plte_too_many_for_depth", raw_png(0, 0, 0, 0, zcompress(bytes(13 * 16)), pre=[chunk(b"PLTE", b"\x00" * 51)], extra_ihdr=ih(29, 13, 4, 3)), exp_rc=E_PLTE)
    add_image("err_plte_missing", raw_png(0, 0, 0, 0, zcompress(bytes(13 * 16)), extra_ihdr=ih(29, 13, 4, 3)), exp_rc=E_PLTE)
    add_image("err_plte_after_idat", SIG + ih(29, 13, 8, 2) + chunk(b"IDAT", z) + chunk(b"PLTE", b"\x00" * 6) + chunk(b"IEND", b""), exp_rc=E_ORDER)
    add_image("err_trns_grey_len3", raw_png(0, 0, 0, 0, zcompress(bytes(13 * 30)), pre=[chunk(b"tRNS", b"\x00\x01\x02")], extra_ihdr=ih(29, 13, 8, 0)), exp_rc=E_CHUNK)
    add_image("err_trns_before_plte", raw_png(0, 0, 0, 0, zcompress(bytes(13 * 16)), pre=[chunk(b"tRNS", b"\x00"), chunk(b"PLTE", b"\x00" * 6)], extra_ihdr=ih(29, 13, 4, 3)), exp_rc=E_ORDER)
    add_image("err_trns_longer_than_plte", raw_png(0, 0, 0, 0, zcompress(bytes(13 * 16)), pre=[chunk(b"PLTE", b"\x00" * 6), chunk(b"tRNS", b"\x00" * 3)], extra_ihdr=ih(29, 13, 4, 3)), exp_rc=E_CHUNK)
    add_image("err_critical_unknown", base[:33] + chunk(b"CRIT", b"x") + base[33:], exp_rc=E_CRITICAL)
    add_image("err_chunk_type_digit", base[:33] + chunk(b"ab1d", b"x") + base[33:], exp_rc=E_CHUNK)
    add_image("err_chunk_len_2g", base[:33] + struct.pack(">I", 0x80000000) + b"abcd" + base[33:], exp_rc=E_CHUNK)
    add_image("err_no_idat", SIG + ih(29, 13, 8, 2) + chunk(b"IEND", b""), exp_rc=E_NODATA)
    add_image("err_no_iend", base[:-12], exp_rc=E_TRUNC)
    add_image("ok_data_after_iend", base + b"garbage after IEND")


# ---------------------------------------------------------------------------------------------------
# hand-assembled deflate streams for the inflate API cases
class BitW:
    def __init__(self):
        self.out = bytearray(); self.acc = 0; self.n = 0; self.data = bytearray()

    def bits(self, v, n):
        self.acc |= (v & ((1 << n) - 1)) << self.n; self.n += n
        while self.n >= 8:
            self.out.append(self.acc & 255); self.acc >>= 8; self.n -= 8

    def code(self, code, ln):   # Huffman codes are sent MSB first
        for i in range(ln - 1, -1, -1): self.bits((code >> i) & 1, 1)

    def align(self):
        if self.n: self.bits(0, 8 - self.n)

    def zhdr(self, cmf=0x78, flg=None):
        if flg is None:
            flg = 0x01
            while ((cmf << 8) | flg) % 31: flg += 1
        self.out += bytes([cmf, flg])

    def adler(self, data):
        self.align(); self.out += struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF)

    def get(self):
        self.align()
        return bytes(self.out)


def canon(lens):
    """canonical Huffman codes for a length list"""
    mx = max(lens) if lens else 0
    bl = [0] * (mx + 2)
    for l in lens:
        if l: bl[l] += 1
    code, nxt = 0, [0] * (mx + 2)
    for b in range(1, mx + 1):
        code = (code + bl[b - 1]) << 1
        nxt[b] = code
    codes = [0] * len(lens)
    for i, l in enumerate(lens):
        if l: codes[i] = nxt[l]; nxt[l] += 1
    return codes


FIXED_LIT = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
FIXED_LIT_C = canon(FIXED_LIT)
LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
LEXT = [0] * 8 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4 + [0]
DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577]
DEXT = [0, 0, 0, 0] + [i // 2 for i in range(2, 28)]


def fixed_lit(bw, s): bw.code(FIXED_LIT_C[s], FIXED_LIT[s])


def len_code(ln):
    for i in range(28, -1, -1):
        if ln >= LBASE[i] and (i == 28 and ln == 258 or i < 28 and ln < LBASE[i] + (1 << LEXT[i])):
            return i, ln - LBASE[i]
    raise ValueError(ln)


def dist_code(d):
    for i in range(29, -1, -1):
        if d >= DBASE[i]: return i, d - DBASE[i]


def fixed_match(bw, ln, d, force_lsym=None):
    if force_lsym is None:
        li, le = len_code(ln)
    else:
        li, le = force_lsym, ln - LBASE[force_lsym]
    fixed_lit(bw, 257 + li)
    if LEXT[li]: bw.bits(le, LEXT[li])
    di, de = dist_code(d)
    bw.code(di, 5)
    if DEXT[di]: bw.bits(de, DEXT[di])


CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]


def dyn_header(bw, lit_lens, dist_lens, cl_lens=None, seq=None, hlit=None, hdist=None):
    """write a dynamic block header; seq = explicit code-length symbol sequence [(sym, extra)] or None
    (lengths written literally with symbols 0..15)"""
    if seq is None:
        seq = [(l, None) for l in list(lit_lens) + list(dist_lens)]
    if cl_lens is None:
        cl_lens = [0] * 19
        used = sorted(set(s for s, _ in seq))
        # complete code over the used symbols (pad with an extra symbol when only one is used)
        if len(used) == 1: used = used + [(used[0] + 1) % 19]
        n = len(used); k = (n - 1).bit_length()
        # lengths: k-bit for all, then shorten some to k-1 so that the code is complete
        short = (1 << k) - n
        for i, s in enumerate(used): cl_lens[s] = k - 1 if i < short else k
    hlit = len(lit_lens) if hlit is None else hlit
    hdist = len(dist_lens) if hdist is None else hdist
    hclen = 19
    while hclen > 4 and cl_lens[CL_ORDER[hclen - 1]] == 0: hclen -= 1
    bw.bits(hlit - 257, 5); bw.bits(hdist - 1, 5); bw.bits(hclen - 4, 4)
    for i in range(hclen): bw.bits(cl_lens[CL_ORDER[i]], 3)
    cc = canon(cl_lens)
    for s, ex in seq:
        bw.code(cc[s], cl_lens[s])
        if s == 16: bw.bits(ex, 2)
        elif s == 17: bw.bits(ex, 3)
        elif s == 18: bw.bits(ex, 7)


ZVECS = []


def add_z(name, data, cap=None, exp_st=None, check_partial=True, ref=None):
    """expected: zlib decides validity; exp_st gives our status for errors"""
    try:
        out = zlib.decompress(data)
        ok = True
    except zlib.error:
        ok = False
    if ok:
        if exp_st is not None and exp_st != ZE_OUT:
            sys.exit(f"{name}: zlib accepts a stream expected to fail ({exp_st})")
        if cap is None: cap = len(out)
        st = PD_OK if cap >= len(out) else ZE_OUT
        if exp_st is not None and st != exp_st: sys.exit(name)
        ZVECS.append(dict(name=name, data=data, cap=cap, st=st, out_len=len(out) if st == PD_OK else cap,
                          h=fnv64(out if st == PD_OK else out[:cap])))
        return
    if exp_st is None:
        sys.exit(f"{name}: zlib rejects it")
    pl = UNCHECKED
    if exp_st == ZE_IN and check_partial:
        do = zlib.decompressobj()
        try:
            part = do.decompress(data)
            pl = len(part)
        except zlib.error:
            sys.exit(f"{name}: zlib reports an error, not a truncation")
    ZVECS.append(dict(name=name, data=data, cap=cap if cap is not None else 1 << 16, st=exp_st, out_len=pl,
                      h=fnv64(part) if pl != UNCHECKED else 0))
    if ref: ZVECS[-1]["ref"] = ref


def build_zvecs():
    text = b"".join(b"line %d: the quick brown fox jumps over the lazy dog %d\n" % (i, i * i % 97) for i in range(400))
    rng = Rng(9)
    noise = bytes(rng.next() & 255 for _ in range(6000))
    add_z("ok_empty_level0", zlib.compress(b"", 0))
    add_z("ok_empty_level9", zlib.compress(b"", 9))
    add_z("ok_text_level1", zlib.compress(text, 1))
    add_z("ok_text_level9", zlib.compress(text, 9))
    add_z("ok_text_fixed", zcompress(text, 9, zlib.Z_FIXED))
    add_z("ok_text_huffonly", zcompress(text[:5000], 9, zlib.Z_HUFFMAN_ONLY))
    add_z("ok_text_rle", zcompress(text[:5000], 9, zlib.Z_RLE))
    add_z("ok_noise_level6", zlib.compress(noise, 6))
    add_z("ok_noise_level0", zlib.compress(noise, 0))
    add_z("ok_run_a_100000", zlib.compress(b"a" * 100000, 9))
    add_z("ok_run_pattern3", zlib.compress(b"abc" * 30000, 9))
    add_z("ok_wbits9", zcompress(text, 9, None, 9))
    # mixed blocks: stored + fixed + dynamic in one stream (zlib full flush points switch block types)
    # (raw deflate segments ending in a sync flush are byte aligned and concatenate into one stream)
    def seg(data, level, strategy, final):
        co = zlib.compressobj(level, zlib.DEFLATED, -15, 8, strategy)
        return co.compress(data) + co.flush(zlib.Z_FINISH if final else zlib.Z_SYNC_FLUSH)
    m = (bytes([0x78, 0x9c]) + seg(text[:3000], 0, zlib.Z_DEFAULT_STRATEGY, 0) + seg(text[3000:6000], 9, zlib.Z_FIXED, 0) +
         seg(text[6000:], 9, zlib.Z_DEFAULT_STRATEGY, 1) + struct.pack(">I", zlib.adler32(text) & 0xFFFFFFFF))
    add_z("ok_mixed_block_types", m)
    # distance 32768 and length 258 via code 285 and code 284+31
    # first 32768 bytes: 251 literals (a prime period) then period-251 matches, so that an off-by-one
    # distance 32768 copy would produce different bytes (32768 mod 251 = 138)
    pat = bytes(rng.next() & 255 for _ in range(251))
    first = bytes(pat[i % 251] for i in range(32768))
    bw = BitW(); bw.zhdr()
    bw.bits(0, 1); bw.bits(1, 2)
    for v in pat: fixed_lit(bw, v)
    left = 32768 - 251
    while left:
        ln = min(258, left)
        if left - ln in (1, 2): ln -= 3
        fixed_match(bw, ln, 251); left -= ln
    fixed_lit(bw, 256)
    bw.bits(1, 1); bw.bits(1, 2); fixed_match(bw, 258, 32768); fixed_match(bw, 258, 1, force_lsym=27); fixed_match(bw, 3, 32768); fixed_lit(bw, 256)
    ref = bytearray(first)
    for ln, d in [(258, 32768), (258, 1), (3, 32768)]:
        for _ in range(ln): ref.append(ref[-d])
    bw.adler(bytes(ref))
    add_z("ok_dist32768_len258_code284", bw.get())
    # dynamic block with a single distance code of length 1, and one with no distance codes at all
    def dyn_stream(lit_lens, dist_lens, body, data_out):
        bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(2, 2)
        dyn_header(bw, lit_lens, dist_lens)
        lc, dc = canon(lit_lens), canon(dist_lens)
        for item in body:
            if item[0] == "lit": bw.code(lc[item[1]], lit_lens[item[1]])
            else:
                _, ln, d = item
                li, le = len_code(ln); bw.code(lc[257 + li], lit_lens[257 + li])
                if LEXT[li]: bw.bits(le, LEXT[li])
                di, de = dist_code(d); bw.code(dc[di], dist_lens[di])
                if DEXT[di]: bw.bits(de, DEXT[di])
        bw.code(lc[256], lit_lens[256]); bw.adler(data_out)
        return bw.get()
    ll = [0] * 258; ll[ord("x")] = 2; ll[ord("y")] = 2; ll[256] = 2; ll[257] = 2   # 4 codes of length 2: complete
    add_z("ok_dyn_single_dist_code", dyn_stream(ll, [1], [("lit", 120), ("lit", 121), ("m", 3, 1), ("lit", 120)], b"xyyyyx"))
    ll2 = [0] * 257; ll2[ord("p")] = 1; ll2[256] = 1
    add_z("ok_dyn_no_dist_codes", dyn_stream(ll2, [0], [("lit", 112)] * 5, b"ppppp"))
    ll3 = [0] * 257; ll3[256] = 1
    add_z("ok_dyn_only_eob_single_code", dyn_stream(ll3, [0], [], b""))
    # ---- errors
    add_z("err_hdr_cm7", bytes([0x77, 0x01]) + zlib.compress(text)[2:], exp_st=ZE_HDR)
    hz = zlib.compress(text)
    add_z("err_hdr_fcheck", hz[:1] + bytes([hz[1] ^ 1]) + hz[2:], exp_st=ZE_HDR)
    bw = BitW(); bw.zhdr(cmf=0x88); bw.bits(1, 1); bw.bits(1, 2); fixed_lit(bw, 256); bw.adler(b"")
    add_z("err_hdr_cinfo8", bw.get(), exp_st=ZE_HDR)
    f = 0x20
    while ((0x78 << 8) | f) % 31: f += 1
    add_z("err_hdr_fdict", bytes([0x78, f]) + b"\x00\x00\x00\x01" + zlib.compress(text)[2:], exp_st=ZE_HDR)
    add_z("err_btype3", bytes([0x78, 0x9c, 0x07, 0x00]), exp_st=ZE_BTYPE)
    st = bytearray(zlib.compress(b"hello world", 0)); st[5] ^= 1
    add_z("err_stored_nlen", bytes(st), exp_st=ZE_STORED)
    def dyn_bad(name, lit_lens, dist_lens, exp, cl_lens=None, seq=None, hlit=None, hdist=None):
        bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(2, 2)
        dyn_header(bw, lit_lens, dist_lens, cl_lens, seq, hlit, hdist)
        bw.bits(0, 32); bw.bits(0, 32)   # padding so that the error is not a truncation
        add_z(name, bw.get(), exp_st=exp)
    dyn_bad("err_dyn_hlit_287", [8] * 287, [5] * 30, ZE_CODES, hlit=287)
    dyn_bad("err_dyn_hdist_31", [8] * 257, [5] * 31, ZE_CODES, hdist=31)
    cl_over = [0] * 19; cl_over[0] = 1; cl_over[8] = 1; cl_over[9] = 1
    dyn_bad("err_dyn_cl_oversubscribed", [8] * 257, [1, 1], ZE_CODES, cl_lens=cl_over, seq=[(8, None)])
    cl_inc = [0] * 19; cl_inc[8] = 2; cl_inc[1] = 2; cl_inc[0] = 2
    dyn_bad("err_dyn_cl_incomplete", [8] * 257, [1, 1], ZE_CODES, cl_lens=cl_inc, seq=[(8, None)])
    dyn_bad("err_dyn_repeat16_first", [8] * 257, [1, 1], ZE_CODES, seq=[(16, 0)] + [(8, None)] * 300)
    dyn_bad("err_dyn_repeat_overrun", [8] * 257, [1, 1], ZE_CODES, seq=[(8, None)] * 250 + [(18, 127)])
    nolen = [8] * 256 + [0] + [8]
    dyn_bad("err_dyn_no_eob_code", nolen, [1, 1], ZE_CODES)
    dyn_bad("err_dyn_lit_oversubscribed", [7] * 257, [1, 1], ZE_CODES)
    inc = [0] * 257; inc[0] = 2; inc[256] = 2
    dyn_bad("err_dyn_lit_incomplete", inc, [1, 1], ZE_CODES)
    lit_ok = [0] * 257; lit_ok[0] = 1; lit_ok[256] = 1
    dyn_bad("err_dyn_dist_incomplete", lit_ok, [2, 2], ZE_CODES)
    dyn_bad("err_dyn_dist_oversubscribed", lit_ok, [1, 1, 1], ZE_CODES)
    # invalid symbols in fixed blocks
    bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(1, 2); fixed_lit(bw, 65); fixed_lit(bw, 286); bw.bits(0, 32); bw.bits(0, 32)
    add_z("err_fixed_lit286", bw.get(), exp_st=ZE_SYM)
    bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(1, 2); fixed_lit(bw, 65); fixed_lit(bw, 287); bw.bits(0, 32); bw.bits(0, 32)
    add_z("err_fixed_lit287", bw.get(), exp_st=ZE_SYM)
    bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(1, 2); fixed_lit(bw, 65); fixed_lit(bw, 257); bw.code(30, 5); bw.bits(0, 32); bw.bits(0, 32)
    add_z("err_fixed_dist30", bw.get(), exp_st=ZE_SYM)
    bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(1, 2); fixed_lit(bw, 65); fixed_match(bw, 3, 2); fixed_lit(bw, 256); bw.adler(b"")
    add_z("err_dist_too_far", bw.get(), exp_st=ZE_DIST)
    # no distance codes, then a length symbol: the distance cannot be decoded
    bw = BitW(); bw.zhdr(); bw.bits(1, 1); bw.bits(2, 2)
    l4 = [0] * 258; l4[ord("a")] = 1; l4[256] = 2; l4[257] = 2
    dyn_header(bw, l4, [0]); lc = canon(l4)
    bw.code(lc[ord("a")], 1); bw.code(lc[257], 2); bw.bits(0, 32); bw.bits(0, 32)
    add_z("err_dyn_dist_without_codes", bw.get(), exp_st=ZE_SYM)
    ad = bytearray(zlib.compress(text, 9)); ad[-2] ^= 4
    add_z("err_adler", bytes(ad), exp_st=ZE_ADLER)
    # output buffer one byte too small
    add_z("err_out_cap_minus1", zlib.compress(text, 9), cap=len(text) - 1, exp_st=ZE_OUT)
    add_z("err_out_cap_minus1_stored", zlib.compress(text[:3000], 0), cap=2999, exp_st=ZE_OUT)
    add_z("ok_cap_plus100", zlib.compress(text, 9), cap=len(text) + 100)
    # truncations: every byte for a short dynamic stream, a spread for stored / long ones
    small = zlib.compress(text[:700], 9)
    add_z("base_dyn", small)
    for c in range(0, len(small)):
        add_z(f"trunc_dyn_{c}", small[:c], exp_st=ZE_IN, ref="base_dyn")
    fx = zcompress(text[:400], 9, zlib.Z_FIXED)
    add_z("base_fixed", fx)
    for c in range(0, len(fx), 7):
        add_z(f"trunc_fixed_{c}", fx[:c], exp_st=ZE_IN, ref="base_fixed")
    sto = zlib.compress(noise[:2000], 0)
    add_z("base_stored", sto)
    for c in [1, 2, 3, 4, 5, 6, 7, 8, 100, 1000, 2006, len(sto) - 4, len(sto) - 1]:
        add_z(f"trunc_stored_{c}", sto[:c], exp_st=ZE_IN, ref="base_stored")


# ---------------------------------------------------------------------------------------------------
# checksum vectors
def ck_data(i):
    return (((i * 2654435761) & 0xFFFFFFFF) >> 13 ^ (i >> 8)) & 0xFF


CK_LENS = [0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 1000, 5551, 5552, 5553, 11104,
           11105, 65535, 65536, 65537, 1000003]
CK_FF_LENS = [1, 5552, 5553, 100000, 1 << 20, 3 << 20]


def build_ck():
    mx = max(CK_LENS)
    buf = bytes(ck_data(i) for i in range(mx))
    vec = []
    for n in CK_LENS:
        vec.append((n, 0, zlib.adler32(buf[:n]) & 0xFFFFFFFF, zlib.crc32(buf[:n]) & 0xFFFFFFFF))
    for n in CK_FF_LENS:
        b = b"\xff" * n
        vec.append((n, 1, zlib.adler32(b) & 0xFFFFFFFF, zlib.crc32(b) & 0xFFFFFFFF))
    return vec


# ---------------------------------------------------------------------------------------------------
def c_array(name, data):
    lines = [f"static const uint8_t {name}[{max(1, len(data))}] = {{"]
    for i in range(0, len(data), 24):
        lines.append("    " + ",".join(str(b) for b in data[i:i + 24]) + ",")
    if not data: lines.append("    0,")
    lines.append("};")
    return "\n".join(lines)


def cid(s):
    return "".join(c if c.isalnum() else "_" for c in s)


def main():
    build_valid()
    build_errors()
    build_zvecs()
    ck = build_ck()
    for names in ([im["name"] for im in IMAGES], [z["name"] for z in ZVECS]):
        if len(set(names)) != len(names): sys.exit("duplicate names")
    for f in os.listdir(HERE):
        if f.endswith(".png"): os.remove(os.path.join(HERE, f))
    for im in IMAGES:
        if "ref" in im: continue
        with open(os.path.join(HERE, im["name"] + ".png"), "wb") as fh: fh.write(im["data"])
    h = ["/* GENERATED by corpus/png/gen_png.py - do not edit. Expected results from the generator's own",
         " * Python PNG decoder (zlib + spec unfilter/expansion), checked against the source samples and Pillow. */",
         "#ifndef SIMD_PNG_CORPUS_H", "#define SIMD_PNG_CORPUS_H",
         "typedef struct sp_img { const char *name; const uint8_t *data; uint32_t len; int flags; int rc; uint32_t w, h;",
         "                        uint32_t raw_produced, raw_complete; uint64_t h_raw, h_pix; uint32_t max_pixels; } sp_img;",
         "typedef struct sp_zvec { const char *name; const uint8_t *data; uint32_t len, cap; int st; uint32_t out_len; uint64_t h_out; } sp_zvec;",
         "typedef struct sp_ck { uint32_t len; int ff; uint32_t adler, crc; } sp_ck;",
         f"#define SP_PIL_CHECKED {PIL_CHECKED[0]}", f"#define SP_PIL_ALPHA_DIFF {PIL_CHECKED[1]}"]
    for im in IMAGES:
        if "ref" not in im: h.append(c_array("sp_d_" + cid(im["name"]), im["data"]))
    for z in ZVECS:
        if "ref" not in z: h.append(c_array("sp_z_" + cid(z["name"]), z["data"]))
    h.append("static const sp_img sp_images[] = {")
    for im in IMAGES:
        arr = "sp_d_" + cid(BASES[im["ref"]] if "ref" in im else im["name"])
        h.append(f'    {{ "{im["name"]}", {arr}, {len(im["data"])}u, {im["flags"]}, {im["rc"]}, {im["w"]}u, {im["h"]}u, '
                 f'{im["raw_produced"]}u, {im["raw_complete"]}u, 0x{im["h_raw"]:016x}ull, 0x{im["h_pix"]:016x}ull, {im["max_pixels"]}u }},')
    h.append("};")
    h.append("static const sp_zvec sp_zvecs[] = {")
    for z in ZVECS:
        h.append(f'    {{ "{z["name"]}", sp_z_{cid(z.get("ref", z["name"]))}, {len(z["data"])}u, {z["cap"]}u, {z["st"]}, {z["out_len"]}u, 0x{z["h"]:016x}ull }},')
    h.append("};")
    h.append("static const sp_ck sp_cks[] = {")
    for n, ff, a, c in ck: h.append(f"    {{ {n}u, {ff}, 0x{a:08x}u, 0x{c:08x}u }},")
    h.append("};")
    h.append("#endif")
    with open(OUT_H, "w") as fh: fh.write("\n".join(h) + "\n")
    with open(OUT_REF, "w") as fh:
        fh.write("# simd_png reference results (corpus/png/gen_png.py). name rc w h raw_produced raw_complete h_raw h_pix flags\n")
        for im in IMAGES:
            fh.write(f'{im["name"]} {im["rc"]} {im["w"]} {im["h"]} {im["raw_produced"]} {im["raw_complete"]} {im["h_raw"]:016x} {im["h_pix"]:016x} {im["flags"]}\n')
        fh.write("# inflate vectors: name status cap out_len h_out\n")
        for z in ZVECS: fh.write(f'z:{z["name"]} {z["st"]} {z["cap"]} {z["out_len"]} {z["h"]:016x}\n')
        fh.write("# checksums over b[i] = ((i*2654435761 mod 2^32) >> 13 ^ (i >> 8)) & 255 (ff=1: all 0xFF): len ff adler32 crc32\n")
        for n, ff, a, c in ck: fh.write(f"ck {n} {ff} {a:08x} {c:08x}\n")
    tot = sum(len(im["data"]) for im in IMAGES)
    print(f"{len(IMAGES)} images ({tot} bytes), {len(ZVECS)} inflate vectors, {len(ck)} checksum vectors; "
          f"Pillow cross-checked {PIL_CHECKED[0]} (alpha-only differences {PIL_CHECKED[1]})")


if __name__ == "__main__":
    main()
