#!/usr/bin/env python3
"""Compare two result directories (e.g. Windows 11 vs another environment) produced by run_all.ps1 /
run_all.sh (or any tree of <exe>.json JSON-lines files plus optional run.json).

  compare_results.py <baseline_dir> <candidate_dir> [--perf-threshold=0.10] [--json=out.json] [--all-perf]

Sections:
  ENV          OS build, CPU, XCR0, usable-feature differences, GetEnabledXStateFeatures
  EXE STATUS   per-exe status differences (OK / FAIL / CRASH / TIMEOUT / HARNESS_ERROR / missing)
  FUNCTIONAL   case results that differ in a functional way (PASS vs FAIL/CRASH/XFAIL/TIMEOUT ...)
  CAPABILITY   SKIP on one side, run on the other (feature exposure difference, not a defect by itself)
  MISSING      cases present on only one side
  PERF         metric deltas (candidate/baseline); |delta| above threshold is flagged
Exit status: 0 = no functional mismatch, 1 = functional mismatches, 2 = usage/input error. stdlib only."""
import json, os, sys, math

def load(root):
    cases, envs, exes = {}, {}, {}
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if fn == "run.json":
                try: run = json.load(open(p, encoding="utf-8-sig"))  # PS 5.1 writes a BOM
                except Exception: continue
                for e in run.get("exes", []):
                    exes[(e.get("arch", "?"), e["name"])] = e.get("status", "?")
                continue
            if not fn.endswith(".json"): continue
            for line in open(p, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except Exception: continue
                t, arch, exe = r.get("type"), r.get("arch", "?"), r.get("exe", fn[:-5])
                if t == "env": envs[(arch, exe)] = r
                elif t == "case": cases[(arch, exe, r.get("case"))] = r
                elif t == "summary" and (arch, exe) not in exes:
                    exes[(arch, exe)] = "OK" if r.get("exit_code") == 0 else "FAIL"
    return cases, envs, exes

FUNC = {"PASS": "ok", "PERF": "ok", "FAIL": "fail", "CRASH": "crash", "XFAIL": "xfail", "TIMEOUT": "timeout", "SKIP": "skip"}

def env_summary(envs):
    out = {}
    for (arch, exe), e in sorted(envs.items()):
        if arch in out: continue
        os_ = e.get("os", {})
        out[arch] = {
            "os": f"{os_.get('major')}.{os_.get('minor')}.{os_.get('build')}.{os_.get('ubr')} {os_.get('product', '')}".strip(),
            "wine": e.get("wine", ""), "cpu": e.get("cpu", {}).get("brand", ""), "xcr0": e.get("xcr0"),
            "api_xstate": e.get("api_xstate_features"), "wow64": e.get("wow64"),
            "usable": set((e.get("features_usable") or "").split()), "version": e.get("version"),
            "compiler": e.get("compiler"), "hypervisor": e.get("cpu", {}).get("hv_vendor", ""),
        }
    return out

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if len(args) != 2:
        print(__doc__); return 2
    thr = float(opts.get("perf-threshold", "0.10"))
    A, B = load(args[0]), load(args[1])
    report = {"baseline": args[0], "candidate": args[1], "env": {}, "exe_status": [], "functional": [],
              "capability": [], "missing": [], "perf": []}
    print(f"baseline : {args[0]}\ncandidate: {args[1]}\n")

    print("== ENV")
    ea, eb = env_summary(A[1]), env_summary(B[1])
    for arch in sorted(set(ea) | set(eb)):
        a, b = ea.get(arch), eb.get(arch)
        if not a or not b:
            print(f"  [{arch}] only in {'baseline' if a else 'candidate'}"); continue
        for k in ("os", "wine", "cpu", "hypervisor", "xcr0", "api_xstate", "wow64", "version", "compiler"):
            mark = "  " if a[k] == b[k] else "!="
            print(f"  [{arch}] {mark} {k:10s} {a[k]!s:40.40s} | {b[k]!s}")
        only_a, only_b = sorted(a["usable"] - b["usable"]), sorted(b["usable"] - a["usable"])
        if only_a: print(f"  [{arch}] usable only in baseline : {' '.join(only_a)}")
        if only_b: print(f"  [{arch}] usable only in candidate: {' '.join(only_b)}")
        report["env"][arch] = {"baseline": {k: (sorted(v) if isinstance(v, set) else v) for k, v in a.items()},
                               "candidate": {k: (sorted(v) if isinstance(v, set) else v) for k, v in b.items()}}

    print("\n== EXE STATUS")
    n = 0
    for key in sorted(set(A[2]) | set(B[2])):
        sa, sb = A[2].get(key, "missing"), B[2].get(key, "missing")
        if sa != sb:
            print(f"  {key[0]}/{key[1]}: {sa} -> {sb}"); n += 1
            report["exe_status"].append({"arch": key[0], "exe": key[1], "baseline": sa, "candidate": sb})
    if not n: print("  (no differences)")

    func, cap, miss = [], [], []
    for key in sorted(set(A[0]) | set(B[0]), key=lambda k: tuple(str(x) for x in k)):
        ra, rb = A[0].get(key), B[0].get(key)
        if not ra or not rb:
            miss.append((key, "baseline" if ra else "candidate")); continue
        fa, fb = FUNC.get(ra["result"], ra["result"]), FUNC.get(rb["result"], rb["result"])
        if fa == fb: continue
        item = {"arch": key[0], "exe": key[1], "case": key[2], "baseline": ra["result"], "candidate": rb["result"],
                "baseline_reason": ra.get("reason", ""), "candidate_reason": rb.get("reason", "")}
        (cap if "skip" in (fa, fb) else func).append(item)
    print(f"\n== FUNCTIONAL mismatches: {len(func)}")
    for i in func:
        print(f"  {i['arch']}/{i['exe']}/{i['case']}: {i['baseline']} -> {i['candidate']}")
        for side in ("baseline", "candidate"):
            if i[side + "_reason"]: print(f"      {side}: {i[side + '_reason'][:200]}")
    print(f"\n== CAPABILITY (SKIP vs run): {len(cap)}")
    for i in cap:
        print(f"  {i['arch']}/{i['exe']}/{i['case']}: {i['baseline']} -> {i['candidate']}  ({(i['baseline_reason'] or i['candidate_reason'])[:120]})")
    print(f"\n== MISSING cases: {len(miss)}")
    for key, side in miss[:200]:
        print(f"  {'/'.join(str(x) for x in key)}: only in {side}")
    report["functional"], report["capability"] = func, cap
    report["missing"] = [{"key": list(k), "only_in": s} for k, s in miss]

    print(f"\n== PERF (candidate/baseline, flag |delta| > {thr:.0%})")
    rows = []
    for key in sorted(set(A[0]) & set(B[0]), key=lambda k: tuple(str(x) for x in k)):
        pa = {p["metric"]: p for p in A[0][key].get("perf", [])}
        pb = {p["metric"]: p for p in B[0][key].get("perf", [])}
        for m in sorted(set(pa) & set(pb)):
            va, vb = pa[m].get("value"), pb[m].get("value")
            if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)) or va == 0: continue
            ratio = vb / va
            rows.append({"arch": key[0], "exe": key[1], "case": key[2], "metric": m, "unit": pa[m].get("unit", ""),
                         "baseline": va, "candidate": vb, "ratio": ratio, "flag": abs(ratio - 1) > thr})
    rows.sort(key=lambda r: -abs(math.log(r["ratio"])) if r["ratio"] > 0 else 0)
    shown = 0
    for r in rows:
        if not r["flag"] and "all-perf" not in opts: continue
        print(f"  {'!' if r['flag'] else ' '} {r['arch']}/{r['exe']}/{r['case']} {r['metric']}: "
              f"{r['baseline']:.4g} -> {r['candidate']:.4g} {r['unit']} (x{r['ratio']:.3f})")
        shown += 1
    print(f"  {len(rows)} metrics compared, {sum(r['flag'] for r in rows)} beyond threshold" + ("" if shown else ""))
    report["perf"] = rows
    if "json" in opts:
        json.dump(report, open(opts["json"], "w"), indent=1, default=str)
    print(f"\nRESULT functional_mismatches={len(func)} capability_differences={len(cap)} missing={len(miss)} "
          f"perf_flagged={sum(r['flag'] for r in rows)}")
    return 1 if func else 0

if __name__ == "__main__":
    sys.exit(main())
