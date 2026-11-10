#!/usr/bin/env bash
# Pack the built suite for a Windows machine:  scripts/pack_windows.sh [out.zip]
# -> build/xcr-windows-<version>.zip with build/x64, build/x86, scripts/ and corpus/ (simd_jpeg compares
# its embedded corpus with corpus/jpeg when present). On Windows: unzip, then scripts\run_baseline.cmd
# (or scripts\run_all.ps1 -Baseline auto). Bring results\baselines\<os>-<cpu>\ back and commit it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make -s all
VER="$(sed -n 's/.*T_VERSION "\(.*\)".*/\1/p' build/version.h 2>/dev/null || true)"
VER="${VER:-$(git describe --always --dirty --tags 2>/dev/null || echo unknown)}"
OUTZIP="${1:-$ROOT/build/xcr-windows-$VER.zip}"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
D="$STAGE/xcr-$VER"; mkdir -p "$D/build" "$D/scripts"
cp -a build/x64 build/x86 "$D/build/"
cp scripts/run_all.ps1 scripts/run_baseline.cmd scripts/compare_results.py "$D/scripts/"
[ -d corpus ] && cp -a corpus "$D/"
git rev-parse HEAD > "$D/GIT_REV" 2>/dev/null || true
rm -f "$OUTZIP"; (cd "$STAGE" && zip -qr "$OUTZIP" "xcr-$VER")
echo "$OUTZIP ($(du -h "$OUTZIP" | cut -f1)), $(ls "$D/build/x64" | wc -l) x64 + $(ls "$D/build/x86" | wc -l) x86 exes"
