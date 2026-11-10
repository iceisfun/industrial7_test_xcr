#!/usr/bin/env python3
"""Mark a linked PE image CET shadow-stack compatible (what link.exe /CETCOMPAT and ld --cetcompat do).

The flag lives in an IMAGE_DEBUG_DIRECTORY entry of Type 20 (IMAGE_DEBUG_TYPE_EX_DLLCHARACTERISTICS) whose
4-byte payload holds IMAGE_DLLCHARACTERISTICS_EX_CET_COMPAT (0x01). The binutils ld this suite is built with
has no --cetcompat, so the test source emits the entry itself (see T_CETCOMPAT_DEBUG_DIR in tests/cet_xstate.c):

    16-byte marker "XCR_CETCOMPAT_DD", then a 28-byte IMAGE_DEBUG_DIRECTORY {0, 0, 0, 0, 20, 4, .rva data, 0},
    then the DWORD payload.

This tool finds the marker, fills the entry's PointerToRawData and points DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG]
at the entry. Idempotent. Usage: pe_cetcompat.py <image.exe> [...]
"""
import struct
import sys

MARKER = b"XCR_CETCOMPAT_DD"
DEBUG_TYPE_EX_DLLCHARACTERISTICS = 20
DIRENT = 28


def patch(path):
    with open(path, "rb") as f:
        img = bytearray(f.read())
    if img[:2] != b"MZ":
        raise SystemExit(f"{path}: not an MZ image")
    pe = struct.unpack_from("<I", img, 0x3C)[0]
    if img[pe:pe + 4] != b"PE\0\0":
        raise SystemExit(f"{path}: no PE signature")
    nsec, = struct.unpack_from("<H", img, pe + 6)
    opt_size, = struct.unpack_from("<H", img, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", img, opt)
    if magic == 0x20B:
        dd = opt + 112
    elif magic == 0x10B:
        dd = opt + 96
    else:
        raise SystemExit(f"{path}: unknown optional header magic 0x{magic:x}")
    sections = []
    for i in range(nsec):
        s = opt + opt_size + 40 * i
        vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", img, s + 8)
        sections.append((va, max(vsize, rawsize), rawptr, rawsize))

    def off_to_rva(off):
        for va, _, rawptr, rawsize in sections:
            if rawptr <= off < rawptr + rawsize:
                return va + off - rawptr
        raise SystemExit(f"{path}: file offset 0x{off:x} is not inside a section")

    def rva_to_off(rva):
        for va, size, rawptr, rawsize in sections:
            if va <= rva < va + size and rva - va < rawsize:
                return rawptr + rva - va
        raise SystemExit(f"{path}: RVA 0x{rva:x} is not backed by file data")

    m = img.find(MARKER)
    if m < 0:
        raise SystemExit(f"{path}: marker {MARKER.decode()} not found (does the source emit T_CETCOMPAT_DEBUG_DIR?)")
    if img.find(MARKER, m + 1) >= 0:
        raise SystemExit(f"{path}: marker found more than once")
    ent = m + len(MARKER)
    typ, size, data_rva = struct.unpack_from("<III", img, ent + 12)
    if typ != DEBUG_TYPE_EX_DLLCHARACTERISTICS or size != 4:
        raise SystemExit(f"{path}: entry after marker has type {typ} size {size}")
    struct.pack_into("<I", img, ent + 24, rva_to_off(data_rva))
    ent_rva = off_to_rva(ent)
    cur_rva, cur_size = struct.unpack_from("<II", img, dd + 6 * 8)
    if (cur_rva, cur_size) not in ((0, 0), (ent_rva, DIRENT)):
        raise SystemExit(f"{path}: image already has a debug directory (0x{cur_rva:x}, {cur_size}); merge not supported")
    struct.pack_into("<II", img, dd + 6 * 8, ent_rva, DIRENT)
    flags, = struct.unpack_from("<I", img, rva_to_off(data_rva))
    with open(path, "wb") as f:
        f.write(img)
    print(f"{path}: debug directory -> RVA 0x{ent_rva:x}, DllCharacteristicsEx 0x{flags:x}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        patch(p)
