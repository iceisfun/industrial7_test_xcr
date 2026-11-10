#!/usr/bin/env bash
# Run every built exe under headless Wine (sequentially: one wine tree at a time) and collect results.
#   scripts/run_all.sh [--arch x64|x86|all] [--out DIR] [--timeout SEC] [--only SUBSTR] [-- extra exe args]
# Writes DIR/<arch>/<name>.json (JSON lines from --json), <name>.txt (stdout+stderr), and DIR/<arch>/run.json.
# Status per exe: OK (exit 0), FAIL (1), HARNESS_ERROR (2), TIMEOUT (killed by the runner),
# CRASH (any other exit, or no SUMMARY record). Wine is plumbing validation only; Windows is the authority.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH=all; OUT="$ROOT/results"; TMO=600; ONLY=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2;; --out) OUT="$2"; shift 2;; --timeout) TMO="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;; --) shift; EXTRA=("$@"); break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ "$ARCH" = all ] && ARCHS="x64 x86" || ARCHS="$ARCH"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
worst=0
for a in $ARCHS; do
  D="$OUT/$a"; mkdir -p "$D"; LIST="$D/.runlist"; : > "$LIST"
  for exe in "$ROOT"/build/$a/*.exe; do
    [ -f "$exe" ] || continue
    n="$(basename "$exe" .exe)"
    [ -n "$ONLY" ] && [[ "$n" != *"$ONLY"* ]] && continue
    rm -f "$D/$n.json"
    t0=$(date +%s%N)
    XCR_TIMEOUT="$TMO" "$ROOT/tools/wine_run.sh" "$exe" "--json=Z:$D/$n.json" "${EXTRA[@]}" > "$D/$n.txt" 2>&1
    rc=$?
    t1=$(date +%s%N)
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then st=TIMEOUT
    elif ! grep -q '"type":"summary"' "$D/$n.json" 2>/dev/null; then st=CRASH
    elif [ $rc -eq 0 ]; then st=OK
    elif [ $rc -eq 1 ]; then st=FAIL
    elif [ $rc -eq 2 ]; then st=HARNESS_ERROR
    else st=CRASH; fi
    printf '%s\t%s\t%s\t%s\n' "$n" "$rc" "$st" "$(( (t1 - t0) / 1000000 ))" >> "$LIST"
    printf '%-4s %-28s %-13s rc=%s  %s\n' "$a" "$n" "$st" "$rc" "$(grep -m1 '^SUMMARY' "$D/$n.txt" | cut -c9-)"
    [ "$st" != OK ] && worst=1
  done
  python3 - "$D" "$a" "$TMO" <<'PY'
import json, os, sys, datetime
d, arch, tmo = sys.argv[1], sys.argv[2], int(sys.argv[3])
exes, env = [], None
for line in open(os.path.join(d, ".runlist")):
    n, rc, st, dur = line.rstrip("\n").split("\t")
    summ = None
    p = os.path.join(d, n + ".json")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8", errors="replace"):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("type") == "env" and env is None: env = r
            if r.get("type") == "summary": summ = r
    exes.append({"name": n, "arch": arch, "exit_code": int(rc), "status": st, "duration_s": round(int(dur or 0) / 1000.0, 3), "summary": summ})
json.dump({"runner": "run_all.sh (wine)", "arch": arch, "timeout_s": tmo,
           "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "env": env, "exes": exes},
          open(os.path.join(d, "run.json"), "w"), indent=1)
os.remove(os.path.join(d, ".runlist"))
PY
done
exit $worst
