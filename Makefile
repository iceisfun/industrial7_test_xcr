# Every tests/<name>.c builds to build/x64/<name>.exe and build/x86/<name>.exe, linked with common/*.c.
# NO -march / -mavx*: baseline exes must start on CPUs without AVX. Wider ISAs only through
# __attribute__((target("..."))) functions or the asm helpers, behind runtime detection.
CC64    ?= x86_64-w64-mingw32-gcc
CC32    ?= i686-w64-mingw32-gcc
VERSION := $(shell git describe --always --dirty --tags 2>/dev/null || echo unknown)
CFLAGS  ?= -O2 -g0
WARN    := -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers -Wno-cast-function-type
BASE    := -std=gnu11 -fno-strict-aliasing -fno-omit-frame-pointer -Icommon -DT_VERSION='"$(VERSION)"'
# 32-bit: SSE2 is the floor for this suite (XMM tests); keep x87 math for C code (no -mfpmath=sse needed).
# -mstackrealign: callbacks from Windows (thread procs, VEH, fibers) may arrive with 4-byte alignment.
ARCH32  := -msse2 -mstackrealign
LIBS    := -static -lkernel32 -ladvapi32 -lntdll -lm

TESTS   := $(basename $(notdir $(wildcard tests/*.c)))
CSRC    := $(wildcard common/*.c)
CHDR    := $(wildcard common/*.h)
OBJ64   := $(patsubst common/%.c,build/obj/x64/%.o,$(CSRC))
OBJ32   := $(patsubst common/%.c,build/obj/x86/%.o,$(CSRC))
EXE64   := $(addprefix build/x64/,$(addsuffix .exe,$(TESTS)))
EXE32   := $(addprefix build/x86/,$(addsuffix .exe,$(TESTS)))

all: x64 x86
x64: $(EXE64)
x86: $(EXE32)

build/obj/x64/%.o: common/%.c $(CHDR)
	@mkdir -p $(@D)
	$(CC64) $(CFLAGS) $(WARN) $(BASE) -c -o $@ $<
build/obj/x86/%.o: common/%.c $(CHDR)
	@mkdir -p $(@D)
	$(CC32) $(CFLAGS) $(WARN) $(BASE) $(ARCH32) -c -o $@ $<

build/x64/%.exe: tests/%.c $(OBJ64) $(CHDR)
	@mkdir -p $(@D)
	$(CC64) $(CFLAGS) $(WARN) $(BASE) -o $@ $< $(OBJ64) $(LIBS)
build/x86/%.exe: tests/%.c $(OBJ32) $(CHDR)
	@mkdir -p $(@D)
	$(CC32) $(CFLAGS) $(WARN) $(BASE) $(ARCH32) -o $@ $< $(OBJ32) $(LIBS)

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
	@for a in $(WINE_ARCHS); do for t in $(TESTS); do echo "== $$a/$$t"; \
	  tools/wine_run.sh build/$$a/$$t.exe | grep -E '^(FAIL|CRASH|XFAIL|SKIP|SUMMARY|harness)|^  (FAIL|CRASH)'; done; done

.PHONY: all x64 x86 one check-isa clean wine-run
.SECONDARY: $(OBJ64) $(OBJ32)
