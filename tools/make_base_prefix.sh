#!/usr/bin/env bash
# Create (once) the base Wine prefix used by wine_run.sh. Headless: no X connection is made.
set -euo pipefail
WINE="${WINE:-$HOME/work/wine/proton-ge-custom/build/dist/files/bin/wine}"
[ -x "$WINE" ] || WINE="$(command -v wine)"
BASE="${XCR_BASE_PREFIX:-$HOME/.cache/xcr_suite/prefix}"
if [ -d "$BASE/drive_c" ]; then echo "base prefix exists: $BASE"; exit 0; fi
mkdir -p "$(dirname "$BASE")"
env -u DISPLAY -u WAYLAND_DISPLAY WINEPREFIX="$BASE" WINEARCH=win64 WINEDEBUG=-all timeout 300 "$WINE" wineboot -i >/dev/null 2>&1 || true
env -u DISPLAY -u WAYLAND_DISPLAY WINEPREFIX="$BASE" WINEDEBUG=-all timeout 120 "$WINE" wineboot -s >/dev/null 2>&1 || true
env -u DISPLAY -u WAYLAND_DISPLAY WINEPREFIX="$BASE" "$(dirname "$WINE")/wineserver" -k 2>/dev/null || true
echo "created $BASE"
