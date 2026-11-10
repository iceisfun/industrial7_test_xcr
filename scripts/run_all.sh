#!/usr/bin/env bash
# Run every built exe under headless Wine (sequentially: one wine tree at a time) and collect results.
#   scripts/run_all.sh [--arch x64|x86|all] [--out DIR | --baseline NAME|auto] [--timeout SEC] [--only SUBSTR] [-- extra exe args]
# Writes DIR/<arch>/<name>.json (JSON lines from --json), <name>.txt (stdout+stderr), DIR/<arch>/run.json
# (with --only, merged with an earlier run.json in DIR so the other exes are kept) and DIR/host.json.
# --baseline NAME writes to results/baselines/NAME; auto = compare_results.py --slug (wine-<ver>-<cpu>).
# Status per exe: OK (exit 0), FAIL (1), HARNESS_ERROR (2), TIMEOUT (killed by the runner),
# CRASH (any other exit, or no SUMMARY record). Wine is plumbing validation only; Windows is the authority.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH=all; OUT="$ROOT/results"; TMO=600; ONLY=""; EXTRA=(); BASELINE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2;; --out) OUT="$2"; shift 2;; --timeout) TMO="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;; --baseline) BASELINE="$2"; shift 2;; --) shift; EXTRA=("$@"); break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ "$ARCH" = all ] && ARCHS="x64 x86" || ARCHS="$ARCH"
if [ -n "$BASELINE" ] && [ "$BASELINE" != auto ]; then
  [[ "$BASELINE" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad baseline name $BASELINE" >&2; exit 2; }
  OUT="$ROOT/results/baselines/$BASELINE"
elif [ "$BASELINE" = auto ]; then
  [ -z "$ONLY" ] || { echo "--baseline auto records whole arches: use --baseline <name> with --only" >&2; exit 2; }
  mkdir -p "$ROOT/results"; OUT="$(mktemp -d "$ROOT/results/.baseline.XXXXXX")"   # renamed once the env is known
fi
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
  python3 - "$D" "$a" "$TMO" "$ROOT" "$ONLY" <<'PY'
import json, os, sys, datetime, hashlib
d, arch, tmo, root, only = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
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
    try:
        with open(os.path.join(root, "build", arch, n + ".exe"), "rb") as f: sha = hashlib.sha256(f.read()).hexdigest()
    except OSError: sha = None
    exes.append({"name": n, "arch": arch, "exit_code": int(rc), "status": st, "duration_s": round(int(dur or 0) / 1000.0, 3), "sha256": sha, "summary": summ})
try:   # keep exes of an earlier run that this one (--only) did not re-run
    if not only: raise FileNotFoundError
    with open(os.path.join(d, "run.json"), encoding="utf-8-sig") as f: old = json.load(f)
    done = {e["name"] for e in exes}
    exes = sorted(exes + [e for e in old.get("exes", []) if e.get("name") not in done], key=lambda e: e["name"])
    env = env or old.get("env")
except Exception: pass
json.dump({"runner": "run_all.sh (wine)", "arch": arch, "timeout_s": tmo,
           "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "env": env, "exes": exes},
          open(os.path.join(d, "run.json"), "w"), indent=1)
os.remove(os.path.join(d, ".runlist"))
PY
done
if [ "$BASELINE" = auto ]; then
  SLUG="$(python3 "$ROOT/scripts/compare_results.py" --slug "$OUT")" || { echo "cannot derive a baseline name" >&2; exit 2; }
  DEST="$ROOT/results/baselines/$SLUG"; mkdir -p "$DEST"
  for a in $ARCHS; do rm -rf "$DEST/$a"; mv "$OUT/$a" "$DEST/$a"; done
  rm -rf "$OUT"; OUT="$DEST"; echo "baseline: $OUT"
fi
python3 - "$OUT" "$ROOT" "$TMO" "$ONLY" <<'PY'
import json, os, sys, subprocess, platform, datetime
out, root, tmo, only = sys.argv[1:5]
def run(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception: return ""
cpu = ""
try:
    for l in open("/proc/cpuinfo"):
        if l.startswith("model name"): cpu = l.split(":", 1)[1].strip(); break
except OSError: pass
h = {"runner": "run_all.sh (wine)", "slug": run(sys.executable, os.path.join(root, "scripts", "compare_results.py"), "--slug", out),
     "host_os": platform.platform(), "cpu": cpu, "wine": run(os.environ.get("WINE") or next((w for w in (os.path.expanduser("~/work/wine/proton-ge-custom/build/dist/files/bin/wine"),) if os.access(w, os.X_OK)), "wine"), "--version"),
     "git_rev": run("git", "-C", root, "rev-parse", "--short=12", "HEAD"),
     "git_dirty": bool(run("git", "-C", root, "status", "--porcelain", "--untracked-files=no")),
     "timeout_s": int(tmo), "only": only, "arches": sorted(a for a in ("x64", "x86") if os.path.isdir(os.path.join(out, a))),
     "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "note": "Wine run: plumbing reference only, not a Windows contract reference"}
json.dump(h, open(os.path.join(out, "host.json"), "w"), indent=1)
PY
exit $worst
