#!/usr/bin/env bash
# Scaffold a new test exe:  tools/new_exe.sh <name> "<one-line purpose>"
# Creates tests/<name>.c (builds to build/x64/<name>.exe and build/x86/<name>.exe via the Makefile).
set -euo pipefail
NAME="${1:?usage: $0 <name> \"<purpose>\"}"; PURPOSE="${2:-TODO purpose}"
[[ "$NAME" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "name must be lower_snake_case" >&2; exit 2; }
cd "$(dirname "$0")/.."
F="tests/$NAME.c"
[ -e "$F" ] && { echo "$F exists" >&2; exit 1; }
cat > "$F" <<EOT
/* $NAME - $PURPOSE
 *
 * Contract under test (cite Intel SDM / AMD APM / Microsoft docs / checked decomp here):
 *   - ...
 * Result policy: feature absent -> SKIP; advertised but wrong -> FAIL; observed-but-variable -> t_note/t_info.
 * Deliberately faulting operations run in CHILD() processes via t_child_run(). See common/API.md. */
#include "t.h"

TEST(${NAME}_example) {
    if (!cpu_has(F_SSE2)) SKIP("SSE2 not usable");
    t_vregs in, out;
    t_vregs_pattern(&in, T_VK_XMM, 1);
    CHECK_EQ(t_vregs_call(T_VK_XMM, &in, &out, NULL, NULL), 0);
    t_vregs_check(&out, &in, T_VK_XMM, 0, -1, "round trip");
}

/* Example child: runs as  $NAME.exe --child=${NAME}_fault ; crash -> exit code = exception code. */
CHILD(${NAME}_fault) {
    (void)argc; (void)argv;
    __asm__ volatile("ud2");
    return 0;
}

TEST(${NAME}_fault_in_child) {
    static t_child_result r;
    REQUIRE(t_child_run("${NAME}_fault", NULL, 10000, &r));
    CHECK(!r.timed_out);
    CHECK_EQ(r.exc_code, 0xC000001Du);   /* STATUS_ILLEGAL_INSTRUCTION */
}
EOT
echo "created $F  (build: make one T=$NAME; run: tools/wine_run.sh build/x64/$NAME.exe)"
