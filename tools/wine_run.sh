#!/usr/bin/env bash
# Run one test exe (x64 or x86) under Wine in a throw-away, HEADLESS sandbox.
#   tools/wine_run.sh build/x64/<name>.exe [exe args...]      e.g. --filter=xmm --json=Z:/abs/path.json
# - fresh copy of the base prefix per run (tests cannot pollute each other or your real prefix);
#   the base prefix is win64 (WoW64), so build/x86 exes run in it too
# - no X: DISPLAY/WAYLAND_DISPLAY are unset (Wine's null display driver). Never touches the live X session.
# - hard timeout (XCR_TIMEOUT seconds, default 600); the prefix's wineserver and children are killed and
#   the prefix removed afterwards. Exit status = the exe's exit status (124 = timeout).
# Unix paths given to the exe must be translated: use Z:/abs/path (see scripts/run_all.sh).
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <exe> [args...]" >&2; exit 2; }
EXE="$1"; shift
WINE="${WINE:-$HOME/work/wine/proton-ge-custom/build/dist/files/bin/wine}"
[ -x "$WINE" ] || WINE="$(command -v wine)"
BASE="${XCR_BASE_PREFIX:-$HOME/.cache/xcr_suite/prefix}"
TMO="${XCR_TIMEOUT:-600}"
[ -f "$EXE" ] || { echo "no such exe: $EXE" >&2; exit 2; }
[ -d "$BASE/drive_c" ] || WINE="$WINE" "$(dirname "$0")/make_base_prefix.sh" >&2
RUN="$(mktemp -d "${TMPDIR:-/tmp}/xcr_prefix.XXXXXX")"
cp -a --reflink=auto "$BASE/." "$RUN/"
cleanup() {
  WINEPREFIX="$RUN" "$(dirname "$WINE")/wineserver" -k 2>/dev/null
  sleep 0.3
  rm -rf "$RUN"
}
trap cleanup EXIT
env -u DISPLAY -u WAYLAND_DISPLAY WINEPREFIX="$RUN" WINEARCH=win64 WINEDEBUG="${WINEDEBUG:--all}" \
    timeout -k 5 "$TMO" "$WINE" "$EXE" "$@"
