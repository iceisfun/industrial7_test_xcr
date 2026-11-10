#!/usr/bin/env python3
"""Baseline-ISA guard: list VEX/EVEX/AMX instructions that live outside functions allowed to contain them.
Baseline exes must start and run their dispatch code on CPUs without AVX, so wide instructions may only
appear in functions that are reached after runtime detection. Convention: such functions carry an ISA
hint in their name (avx, fma, f16c, 512, ymm, zmm, vex, evex, amx, vaes, vpclmul, gfni, vnni, vzero, kreg,
bmi) or are the internal asm helpers (t__*). Use T_TARGET("...") for intrinsics in those functions.
usage: check_isa.py <exe>...   exit 1 if any violation."""
import re, subprocess, sys, shutil
ALLOW = re.compile(r"(avx|fma|f16c|512|ymm|zmm|vex|evex|amx|vaes|vpclmul|gfni|vnni|vzero|kreg|bmi|^_?t__)", re.I)
WIDE = re.compile(r"^(v(?!err|erw)[a-z0-9]+|k(mov|and|andn|or|xor|xnor|not|test|ortest|shift[lr]|unpck|add)[bwdq]?\b.*|ldtilecfg|sttilecfg|tilezero|tileload\w*|tilestore\w*|tdp\w+|tilerelease)")
def objdump_for(exe):
    for t in ("x86_64-w64-mingw32-objdump", "i686-w64-mingw32-objdump", "objdump"):
        if shutil.which(t): return t
    sys.exit("objdump not found")
bad = 0
for exe in sys.argv[1:]:
    out = subprocess.run([objdump_for(exe), "-d", "--no-show-raw-insn", exe], capture_output=True, text=True).stdout
    sym, hits = "?", {}
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m: sym = m.group(1); continue
        parts = line.split("\t")
        if len(parts) < 2: continue
        mnem = parts[-1].strip().split(" ")[0]
        if WIDE.match(mnem) and not ALLOW.search(sym):
            hits.setdefault(sym, set()).add(mnem)
    for s, ms in sorted(hits.items()):
        print(f"{exe}: {s}: {' '.join(sorted(ms)[:8])}")
        bad += 1
print(f"check_isa: {bad} function(s) with wide instructions outside allowed functions")
sys.exit(1 if bad else 0)
