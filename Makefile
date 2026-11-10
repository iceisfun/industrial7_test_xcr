# Every tests/<name>.c and benchmarks/<name>.c builds to build/x64/<name>.exe and build/x86/<name>.exe,
# linked with common/*.c.
# NO -march / -mavx*: baseline exes must start on CPUs without AVX. Wider ISAs only through
# __attribute__((target("..."))) functions or the asm helpers, behind runtime detection.
CC64    ?= x86_64-w64-mingw32-gcc
CC32    ?= i686-w64-mingw32-gcc
# tests/<name>.cpp: C++ exes (x64 g++ uses Windows SEH). GCC's i686 C++ does not (DWARF/SJLJ), so the x86 exe
# is built from the same file as C (or with CXX32 when installed); the file then SKIPs every case.
CXX64   ?= x86_64-w64-mingw32-g++
CXX32   ?= i686-w64-mingw32-g++
VERSION := $(shell git describe --always --dirty --tags 2>/dev/null || echo unknown)
CFLAGS  ?= -O2 -g0
WARN    := -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers -Wno-cast-function-type
BASE    := -std=gnu11 -fno-strict-aliasing -fno-omit-frame-pointer -Icommon
# 32-bit: SSE2 is the floor for this suite (XMM tests); keep x87 math for C code (no -mfpmath=sse needed).
# -mstackrealign: callbacks from Windows (thread procs, VEH, fibers) may arrive with 4-byte alignment.
ARCH32  := -msse2 -mstackrealign
LIBS    := -static -lkernel32 -ladvapi32 -lntdll -lm

TESTS   := $(basename $(notdir $(wildcard tests/*.c))) $(basename $(notdir $(wildcard tests/*.cpp)))
CXXBASE := $(filter-out -std=gnu11,$(BASE)) -std=gnu++17
# Exes marked CET shadow-stack compatible (IMAGE_DLLCHARACTERISTICS_EX_CET_COMPAT, = link /CETCOMPAT). This ld
# has no --cetcompat: the source emits the debug-directory entry, tools/pe_cetcompat.py hooks it up.
CETCOMPAT := cet_xstate
BENCHES := $(basename $(notdir $(wildcard benchmarks/*.c)))
CSRC    := $(wildcard common/*.c)
CHDR    := $(wildcard common/*.h)
# headers private to tests/ and benchmarks/ (shared kernels, corpora): every exe depends on all of them
XHDR    := $(wildcard tests/*.h benchmarks/*.h)
OBJ64   := $(patsubst common/%.c,build/obj/x64/%.o,$(CSRC))
OBJ32   := $(patsubst common/%.c,build/obj/x86/%.o,$(CSRC))
EXE64   := $(addprefix build/x64/,$(addsuffix .exe,$(TESTS) $(BENCHES)))
EXE32   := $(addprefix build/x86/,$(addsuffix .exe,$(TESTS) $(BENCHES)))

# Version stamp (env JSON "version"): build/version.h is rewritten only when `git describe` output
# changes, and t.o (the only user of T_VERSION) depends on it, so a new HEAD or a dirty tree re-stamps
# every exe instead of keeping the version of the first build.
VERSION_H := build/version.h
$(VERSION_H): FORCE
	@mkdir -p $(@D); v='#define T_VERSION "$(VERSION)"'; \
	  [ "$$(cat $@ 2>/dev/null)" = "$$v" ] || echo "$$v" > $@
build/obj/x64/t.o build/obj/x86/t.o: $(VERSION_H)
build/obj/x64/t.o build/obj/x86/t.o: XVER = -include $(VERSION_H)

all: x64 x86
x64: $(EXE64)
x86: $(EXE32)

build/obj/x64/%.o: common/%.c $(CHDR)
	@mkdir -p $(@D)
	$(CC64) $(CFLAGS) $(WARN) $(BASE) $(XVER) -c -o $@ $<
build/obj/x86/%.o: common/%.c $(CHDR)
	@mkdir -p $(@D)
	$(CC32) $(CFLAGS) $(WARN) $(BASE) $(ARCH32) $(XVER) -c -o $@ $<

build/x64/%.exe: tests/%.c $(OBJ64) $(CHDR) $(XHDR)
	@mkdir -p $(@D)
	$(CC64) $(CFLAGS) $(WARN) $(BASE) -o $@ $< $(OBJ64) $(LIBS)
	$(if $(filter $*,$(CETCOMPAT)),python3 tools/pe_cetcompat.py $@)
build/x86/%.exe: tests/%.c $(OBJ32) $(CHDR) $(XHDR)
	@mkdir -p $(@D)
	$(CC32) $(CFLAGS) $(WARN) $(BASE) $(ARCH32) -o $@ $< $(OBJ32) $(LIBS)
	$(if $(filter $*,$(CETCOMPAT)),python3 tools/pe_cetcompat.py $@)
# benchmarks/<name>.c: same harness and flags; -Itests lets them reuse the tests/simd_*.h kernels
build/x64/%.exe: benchmarks/%.c $(OBJ64) $(CHDR) $(XHDR)
	@mkdir -p $(@D)
	$(CC64) $(CFLAGS) $(WARN) $(BASE) -Itests -o $@ $< $(OBJ64) $(LIBS)
build/x86/%.exe: benchmarks/%.c $(OBJ32) $(CHDR) $(XHDR)
	@mkdir -p $(@D)
	$(CC32) $(CFLAGS) $(WARN) $(BASE) $(ARCH32) -Itests -o $@ $< $(OBJ32) $(LIBS)

build/x64/%.exe: tests/%.cpp $(OBJ64) $(CHDR)
	@mkdir -p $(@D)
	$(CXX64) $(CFLAGS) $(WARN) $(CXXBASE) -o $@ $< $(OBJ64) $(LIBS)
build/x86/%.exe: tests/%.cpp $(OBJ32) $(CHDR)
	@mkdir -p $(@D)
	if command -v $(CXX32) >/dev/null 2>&1; then \
	  $(CXX32) $(CFLAGS) $(WARN) $(CXXBASE) $(ARCH32) -o $@ $< $(OBJ32) $(LIBS); \
	else \
	  $(CC32) $(CFLAGS) $(WARN) $(BASE) $(ARCH32) -o $@ -x c $< -x none $(OBJ32) $(LIBS); \
	fi

# one exe:  make one T=xmm_registers
one: build/x64/$(T).exe build/x86/$(T).exe

# Report VEX/EVEX/AMX instructions outside functions allowed to contain them (see tools/check_isa.py).
check-isa: all
	python3 tools/check_isa.py $(EXE64) $(EXE32)

clean:
	rm -rf build

# Sequential headless Wine validation (plumbing only; Windows is the authority). One wine tree at a time.
WINE_ARCHS ?= x64 x86
wine-run: all
	@for a in $(WINE_ARCHS); do for t in $(TESTS) $(BENCHES); do echo "== $$a/$$t"; \
	  tools/wine_run.sh build/$$a/$$t.exe | grep -E '^(FAIL|CRASH|XFAIL|SKIP|SUMMARY|harness)|^  (FAIL|CRASH)'; done; done

.PHONY: all x64 x86 one check-isa clean wine-run FORCE
FORCE:
.SECONDARY: $(OBJ64) $(OBJ32)
