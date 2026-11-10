#!/usr/bin/env bash
# Run one test exe (x64 or x86) under Intel SDE (Software Development Emulator) inside the headless
# throw-away Wine sandbox of tools/wine_run.sh. SDE emulates CPUID, XGETBV and every instruction the
# chosen CPU model has (AVX-512, AMX, ...), so the AVX-512 cases that only SKIP on a host without
# AVX-512 actually execute, and old models (-nhm, -wsm) prove the baseline exes start without AVX.
#
#   tools/sde_run.sh [--cpu=MODEL] build/x64/<name>.exe [exe args...]
#   tools/sde_run.sh --fetch                 download + unpack SDE into $XCR_SDE_DIR (needs network)
#   tools/sde_run.sh --where                 print the sde64 that would be used
#
#   MODEL: any SDE chip option without the dash: spr (default), icx, skx, gnr, future, hsw, snb, wsm, nhm ...
#
# Environment:
#   XCR_SDE_BIN    path to the SDE launcher (default: newest $XCR_SDE_DIR/sde-external-*/sde, the
#                  launcher that can start both 32- and 64-bit processes; sde64 cannot follow Wine's
#                  exec of the 32-bit loader)
#   XCR_SDE_WINE   wine loader to run under SDE (default /usr/lib/wine/wine64, the distro Wine). Wine
#                  builds that install a seccomp syscall filter (Proton) do not work under Pin: every
#                  signal (hardware exception) kills the process. Upstream Wine 9+ works.
#   XCR_SDE_PREFIX base prefix for that Wine (default ~/.cache/xcr_suite/prefix_sde; created on first use)
#   XCR_SDE_LOGDIR where SDE/Pin write pin-log.txt, sde-*.txt (default: a temp dir, removed unless kept)
#   XCR_SDE_DIR    where --fetch unpacks SDE (default ~/.cache/xcr_suite/sde)
#   XCR_SDE_ARGS   extra SDE knobs (put before "--"). Always passed: -ptr_raise 1 (an emulated
#                  instruction that touches an inaccessible page raises #PF -> STATUS_ACCESS_VIOLATION
#                  like hardware, instead of SDE aborting), -raise_int3 1 (INT3 raises
#                  STATUS_BREAKPOINT instead of SDE stopping) and -align_checker_action warn.
#   XCR_TIMEOUT    hard timeout in seconds (default 1800 here: SDE is 10-100x slower than native)
#   XCR_SDE_LOCK   lock file that serialises SDE+Wine runs (default ~/.cache/xcr_suite/sde.lock):
#                  only one wine tree started by this script runs at a time.
#
# What it sets for the exe: XCR_SDE=<MODEL> in the environment (Wine copies Unix variables into the
# Windows environment). Tests use it ONLY to classify known SDE+Wine artifacts as XFAIL/NOTE (Wine
# builds signal/exception CONTEXTs from the host XSAVE image, which SDE does not extend, and Wine's
# helper processes may die under Pin). It never weakens a check: without XCR_SDE every check is as on
# Windows.
# Also sets WINELOADERNOEXEC=1: Pin cannot follow Wine's exec of the static wine-preloader.
#
# Exit status = the exe's exit status (124 = timeout), as wine_run.sh. "Tool (or Pin) caused signal"
# reports printed on stderr come from Wine's background services (services.exe, plugplay, ...) dying
# under Pin; they are harmless when the exe prints its SUMMARY line.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SDE_DIR="${XCR_SDE_DIR:-$HOME/.cache/xcr_suite/sde}"
SDE_URL_PAGE="https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html"

find_sde() {
  if [ -n "${XCR_SDE_BIN:-}" ]; then echo "$XCR_SDE_BIN"; return; fi
  command -v sde 2>/dev/null && return
  ls -d "$SDE_DIR"/sde-external-*-lin/sde 2>/dev/null | sort -V | tail -1
}

fetch_sde() {
  mkdir -p "$SDE_DIR" || exit 2
  local url
  url="$(timeout 60 curl -sSfL "$SDE_URL_PAGE" | grep -oE 'https://downloadmirror\.intel\.com/[^"]*-lin\.tar\.xz' | sort -u | tail -1)"
  [ -n "$url" ] || { echo "could not find the SDE Linux download on $SDE_URL_PAGE" >&2; exit 2; }
  echo "fetching $url (Intel Simplified Software License)" >&2
  local dl; dl="$(mktemp -d "$SDE_DIR/dl.XXXXXX")"
  timeout 600 curl -sSfL -o "$dl/sde.tar.xz" "$url" && tar -C "$SDE_DIR" -xf "$dl/sde.tar.xz"
  local rc=$?; rm -rf "$dl"; [ $rc -eq 0 ] || exit 2
  find_sde
}

CPU=spr
while [ $# -gt 0 ]; do
  case "$1" in
    --fetch) fetch_sde; exit $?;;
    --where) s="$(find_sde)"; [ -n "$s" ] && echo "$s" || { echo "SDE not found (tools/sde_run.sh --fetch)" >&2; exit 2; }; exit 0;;
    --cpu=*) CPU="${1#--cpu=}"; shift;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;;
    *) break;;
  esac
done
[ $# -ge 1 ] || { echo "usage: $0 [--cpu=MODEL] <exe> [args...]" >&2; exit 2; }
SDE="$(find_sde)"
[ -n "$SDE" ] && [ -x "$SDE" ] || { echo "SDE not found: set XCR_SDE_BIN or run $0 --fetch" >&2; exit 2; }
case "$CPU" in *[!a-z0-9_-]*|"") echo "bad --cpu=$CPU" >&2; exit 2;; esac

LOCK="${XCR_SDE_LOCK:-$HOME/.cache/xcr_suite/sde.lock}"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -w 3600 9 || { echo "could not take $LOCK" >&2; exit 2; }

EXE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
[ -f "$EXE" ] || { echo "no such exe: $EXE" >&2; exit 2; }
export WINE="${XCR_SDE_WINE:-/usr/lib/wine/wine64}"
[ -x "$WINE" ] || { echo "no wine loader $WINE (set XCR_SDE_WINE)" >&2; exit 2; }
export XCR_BASE_PREFIX="${XCR_SDE_PREFIX:-$HOME/.cache/xcr_suite/prefix_sde}"
[ -d "$XCR_BASE_PREFIX/drive_c" ] || "$HERE/make_base_prefix.sh" >&2 9>&-
export XCR_SDE="$CPU" WINELOADERNOEXEC=1
export XCR_TIMEOUT="${XCR_TIMEOUT:-1800}"
# The Windows exe sees XCR_SDE (model); SDE gets the chip option and any extra knobs.
LOGDIR="${XCR_SDE_LOGDIR:-}"
if [ -z "$LOGDIR" ]; then LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/xcr_sde_log.XXXXXX")"; trap 'rm -rf "$LOGDIR"' EXIT; fi
mkdir -p "$LOGDIR"; LOGDIR="$(cd "$LOGDIR" && pwd)"
# SDE reports (chip-check stops, failed emulated accesses) go to these files for every process of the
# tree (each Pin process rewrites them at start); the harness reads them around each child run
# (XCR_SDE_ERRFILE, ';'-separated Windows paths) to attribute a child's death to SDE.
export XCR_SDE_ERRFILE="Z:$LOGDIR/sde-error.txt;Z:$LOGDIR/sde-chip-check.txt"
export XCR_WINE_WRAP="$SDE -$CPU -ptr_raise 1 -raise_int3 1 -align_checker_action warn -err_filename $LOGDIR/sde-error.txt -chip_check_emit_file 1 -chip_check_file $LOGDIR/sde-chip-check.txt ${XCR_SDE_ARGS:-} --"
# SDE writes its logs into the current directory: run from LOGDIR. The lock stays with this shell
# (fd 9 closed for the children), not with a lingering wineserver.
(cd "$LOGDIR" && "$HERE/wine_run.sh" "$EXE" "$@") 9>&-
