#!/usr/bin/env python3
"""Compare suite results produced by run_all.ps1 / run_all.sh (or any tree of <exe>.json JSON-lines files
plus optional run.json / host.json). stdlib only.

  compare_results.py <baseline_dir> <candidate_dir> [options]     diff two runs
  compare_results.py --baseline <run_dir> [--use=<name>] [--baselines=<dir>] [options]
                     diff a run against a committed baseline under results/baselines/ (the closest one by
                     wine-or-not, OS family/build, CPU, usable features, XCR0 and arch, unless --use names it)
  compare_results.py --triage <run_dir> [--json=out.json] [--expect=<file>]
                     every FAIL / CRASH / TIMEOUT / non-OK exe of one run with its failure lines, plus XFAILs
                     (an XFAIL outside Wine means a t_is_wine-style guard or a t_expect_fail met real Windows:
                     look at it), grouped by exe, and the expected-on-Windows rows of
                     results/baselines/expect_windows.json (cases that never ran anywhere else) applied to a
                     non-Wine run. Use it on a fresh Windows baseline before committing it.
  compare_results.py --list-baselines [--baselines=<dir>]
  compare_results.py --slug <run_dir>      print the baseline name for a run (<os>-<cpu>, as run_all -Baseline auto)

Options: --perf-threshold=0.10  --json=out.json  --all-perf  --arch=x64|x86  --exe=<substr>  --strict

Diff sections:
  ENV          OS build, CPU, XCR0, usable-feature differences, GetEnabledXStateFeatures
  EXE STATUS   per-exe status differences (OK / FAIL / CRASH / TIMEOUT / HARNESS_ERROR / missing)
  FUNCTIONAL   case results that differ in a functional way (PASS vs FAIL/CRASH/XFAIL/TIMEOUT ...), each
               tagged REGRESSION (was ok, now fail/crash/timeout), FIXED (the reverse), XFAIL-FLIP
               (PASS <-> XFAIL: an expected-failure guard fired on one side only) or CHANGED
  CAPABILITY   SKIP on one side, run on the other (feature exposure difference, not a defect by itself)
  MISSING      cases present on only one side
  PERF         metric deltas (candidate/baseline); |delta| above threshold is flagged (PERF never fails)
Exit status: 0 = no functional mismatch other than XFAIL-FLIPs (--strict counts those too; triage: no
FAIL/CRASH/TIMEOUT), 1 = functional mismatches or an exe that was OK and now is not (triage: failures),
2 = usage/input error."""
import json, os, re, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASELINES = os.path.join(os.path.dirname(HERE), "results", "baselines")


def read_json_file(p):
    try:
        with open(p, encoding="utf-8-sig") as f:   # PS 5.1 writes a BOM
            return json.load(f)
    except Exception:
        return None


def load(root):
    """-> (cases{(arch,exe,case):rec}, envs{(arch,exe):env}, exes{(arch,exe):status}, host dict or None)"""
    cases, envs, exes = {}, {}, {}
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")
    host = read_json_file(os.path.join(root, "host.json"))
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if fn == "host.json":
                continue
            if fn == "run.json":
                run = read_json_file(p)
                if not isinstance(run, dict): continue
                for e in run.get("exes") or []:
                    exes[(e.get("arch", "?"), e["name"])] = e.get("status", "?")
                continue
            if not fn.endswith(".json"): continue
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip().lstrip("﻿")
                    if not line.startswith("{"): continue
                    try: r = json.loads(line)
                    except Exception: continue
                    if not isinstance(r, dict): continue
                    t, arch, exe = r.get("type"), r.get("arch", "?"), r.get("exe", fn[:-5])
                    if t == "env": envs[(arch, exe)] = r
                    elif t == "case": cases[(arch, exe, r.get("case"))] = r
                    elif t == "summary" and (arch, exe) not in exes:
                        exes[(arch, exe)] = "OK" if r.get("exit_code") == 0 else "FAIL"
    return cases, envs, exes, host


FUNC = {"PASS": "ok", "PERF": "ok", "FAIL": "fail", "CRASH": "crash", "XFAIL": "xfail", "TIMEOUT": "timeout", "SKIP": "skip"}
BAD = ("fail", "crash", "timeout")


def env_summary(envs):
    out = {}
    for (arch, exe), e in sorted(envs.items()):
        if arch in out: continue
        os_ = e.get("os", {}) or {}
        out[arch] = {
            "os": f"{os_.get('major')}.{os_.get('minor')}.{os_.get('build')}.{os_.get('ubr')} {os_.get('product', '')}".strip(),
            "build": os_.get("build") or 0,
            "wine": e.get("wine", ""), "cpu": (e.get("cpu", {}) or {}).get("brand", ""), "xcr0": e.get("xcr0"),
            "vendor": (e.get("cpu", {}) or {}).get("vendor", ""),
            "api_xstate": e.get("api_xstate_features"), "wow64": e.get("wow64"),
            "usable": set((e.get("features_usable") or "").split()), "version": e.get("version"),
            "compiler": e.get("compiler"), "hypervisor": (e.get("cpu", {}) or {}).get("hv_vendor", ""),
        }
    return out


def os_family(build, wine):
    if wine: return "wine"
    try: b = int(build)
    except Exception: return "unknown"
    if b >= 22000: return "win11"
    if b >= 10240: return "win10"
    return "windows"


def slug_for(root):
    """Baseline name for a run: host.json slug (run_all.ps1) if present, else derived from the env record."""
    data = load(root)
    host = data[3]
    if isinstance(host, dict) and host.get("slug"):
        return host["slug"]
    es = env_summary(data[1])
    if not es:
        return None
    e = es.get("x64") or next(iter(es.values()))
    fam = os_family(e["build"], e["wine"])
    if fam == "wine":
        m = re.search(r"(\d+(?:\.\d+)?)", e["wine"] or "")
        osp = "wine" + ("-" + m.group(1) if m else "")
    else:
        osp = f"{fam}-{e['build']}"
    c = re.sub(r"\((R|TM|tm|r)\)", " ", e["cpu"] or "")
    c = re.sub(r"@.*$", "", c)
    c = re.sub(r"\b(CPU|Processor|\d+-Core)\b", " ", c)
    c = re.sub(r"\bwith Radeon.*$", "", c)
    c = re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")[:40].strip("-") or "cpu"
    return f"{osp}-{c}"


def find_baselines(root):
    out = []
    if not os.path.isdir(root):
        return out
    for n in sorted(os.listdir(root)):
        d = os.path.join(root, n)
        if not os.path.isdir(d): continue
        if any(os.path.isdir(os.path.join(d, a)) for a in ("x64", "x86", "native")) or os.path.exists(os.path.join(d, "host.json")):
            out.append(d)
    return out


def match_score(base_env, cand_env):
    """Higher is closer. Returns (score, reasons)."""
    score, why = 0, []
    common = set(base_env) & set(cand_env)
    score += 5 * len(common)
    if not common:
        return -1, ["no common arch"]
    a, b = base_env[sorted(common)[0]], cand_env[sorted(common)[0]]
    fa, fb = os_family(a["build"], a["wine"]), os_family(b["build"], b["wine"])
    if bool(a["wine"]) == bool(b["wine"]): score += 100; why.append("same wine-or-not")
    if fa == fb: score += 40; why.append(f"os {fa}")
    if a["build"] == b["build"]: score += 20; why.append(f"build {a['build']}")
    if a["cpu"] and a["cpu"] == b["cpu"]: score += 30; why.append("same cpu")
    elif a["vendor"] == b["vendor"]: score += 5; why.append(f"vendor {a['vendor']}")
    if a["usable"] == b["usable"]: score += 20; why.append("same usable features")
    else: score += max(0, 15 - len(a["usable"] ^ b["usable"]))
    if a["xcr0"] == b["xcr0"]: score += 10; why.append(f"xcr0 {a['xcr0']}")
    return score, why


def filt(data, opts):
    arch, exe = opts.get("arch"), opts.get("exe")
    if not arch and not exe:
        return data
    ok = lambda a, e: (not arch or a == arch) and (not exe or exe in e)
    cases = {k: v for k, v in data[0].items() if ok(k[0], k[1])}
    envs = {k: v for k, v in data[1].items() if ok(k[0], k[1])}
    exes = {k: v for k, v in data[2].items() if ok(k[0], k[1])}
    return cases, envs, exes, data[3]


def kind_of(fa, fb):
    if fa in ("ok", "xfail") and fb in BAD: return "REGRESSION"
    if fa in BAD and fb in ("ok", "xfail"): return "FIXED"
    if {fa, fb} == {"ok", "xfail"}: return "XFAIL-FLIP"
    return "CHANGED"


def diff(base_dir, cand_dir, opts, A=None, B=None):
    thr = float(opts.get("perf-threshold", "0.10"))
    A = filt(A or load(base_dir), opts)
    B = filt(B or load(cand_dir), opts)
    report = {"baseline": base_dir, "candidate": cand_dir, "env": {}, "exe_status": [], "functional": [],
              "capability": [], "missing": [], "perf": []}
    print(f"baseline : {base_dir}\ncandidate: {cand_dir}\n")

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
    n, exe_reg = 0, 0
    for key in sorted(set(A[2]) | set(B[2])):
        sa, sb = A[2].get(key, "missing"), B[2].get(key, "missing")
        if sa != sb:
            worse = sa == "OK" and sb in ("FAIL", "CRASH", "TIMEOUT", "HARNESS_ERROR")
            exe_reg += worse
            print(f"  {'REGRESSION ' if worse else ''}{key[0]}/{key[1]}: {sa} -> {sb}"); n += 1
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
                "kind": kind_of(fa, fb),
                "baseline_reason": ra.get("reason", ""), "candidate_reason": rb.get("reason", ""),
                "candidate_failures": [str(x)[:300] for x in (rb.get("failures") or [])[:5]]}
        (cap if "skip" in (fa, fb) else func).append(item)
    order = {"REGRESSION": 0, "CHANGED": 1, "FIXED": 2, "XFAIL-FLIP": 3}
    func.sort(key=lambda i: order[i["kind"]])
    nreg = sum(i["kind"] == "REGRESSION" for i in func)
    nflip = sum(i["kind"] == "XFAIL-FLIP" for i in func)
    print(f"\n== FUNCTIONAL mismatches: {len(func)} (regressions {nreg}, xfail flips {nflip})")
    for i in func:
        print(f"  {i['kind']:10s} {i['arch']}/{i['exe']}/{i['case']}: {i['baseline']} -> {i['candidate']}")
        for side in ("baseline", "candidate"):
            if i[side + "_reason"]: print(f"      {side}: {i[side + '_reason'][:200]}")
        for fl in i["candidate_failures"][:3]:
            print(f"      | {fl[:200]}")
    print(f"\n== CAPABILITY (SKIP vs run): {len(cap)}")
    for i in cap:
        print(f"  {i['arch']}/{i['exe']}/{i['case']}: {i['baseline']} -> {i['candidate']}  ({(i['baseline_reason'] or i['candidate_reason'])[:120]})")
    print(f"\n== MISSING cases: {len(miss)}")
    for key, side in miss[:200]:
        print(f"  {'/'.join(str(x) for x in key)}: only in {side}")
    if len(miss) > 200: print(f"  ... {len(miss) - 200} more")
    report["functional"], report["capability"] = func, cap
    report["missing"] = [{"key": list(k), "only_in": s} for k, s in miss]

    print(f"\n== PERF (candidate/baseline, flag |delta| > {thr:.0%})")
    rows = []
    for key in sorted(set(A[0]) & set(B[0]), key=lambda k: tuple(str(x) for x in k)):
        pa = {p["metric"]: p for p in A[0][key].get("perf", []) or []}
        pb = {p["metric"]: p for p in B[0][key].get("perf", []) or []}
        for m in sorted(set(pa) & set(pb)):
            va, vb = pa[m].get("value"), pb[m].get("value")
            if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)) or va == 0: continue
            ratio = vb / va
            rows.append({"arch": key[0], "exe": key[1], "case": key[2], "metric": m, "unit": pa[m].get("unit", ""),
                         "baseline": va, "candidate": vb, "ratio": ratio, "flag": abs(ratio - 1) > thr})
    rows.sort(key=lambda r: -abs(math.log(r["ratio"])) if r["ratio"] > 0 else 0)
    for r in rows:
        if not r["flag"] and "all-perf" not in opts: continue
        print(f"  {'!' if r['flag'] else ' '} {r['arch']}/{r['exe']}/{r['case']} {r['metric']}: "
              f"{r['baseline']:.4g} -> {r['candidate']:.4g} {r['unit']} (x{r['ratio']:.3f})")
    print(f"  {len(rows)} metrics compared, {sum(r['flag'] for r in rows)} beyond threshold")
    report["perf"] = rows
    if "json" in opts:
        with open(opts["json"], "w") as f:
            json.dump(report, f, indent=1, default=str)
    print(f"\nRESULT functional_mismatches={len(func)} regressions={nreg} exe_regressions={exe_reg} xfail_flips={nflip} "
          f"capability_differences={len(cap)} "
          f"missing={len(miss)} perf_flagged={sum(r['flag'] for r in rows)}")
    counted = len(func) if "strict" in opts else len(func) - nflip
    return 1 if (counted or exe_reg) else 0


def baseline_mode(run_dir, opts):
    root = opts.get("baselines", DEFAULT_BASELINES)
    cand = load(run_dir)
    if opts.get("use"):
        d = opts["use"] if os.path.isdir(opts["use"]) else os.path.join(root, opts["use"])
        if not os.path.isdir(d):
            print(f"no such baseline: {opts['use']} (under {root})"); return 2
        print(f"baseline selected by --use: {d}")
        return diff(d, run_dir, opts, B=cand)
    ce = env_summary(cand[1])
    if not ce:
        print(f"{run_dir}: no env records (not a result directory?)"); return 2
    best = None
    print(f"== BASELINE selection (from {root})")
    for d in find_baselines(root):
        if os.path.abspath(d) == os.path.abspath(run_dir): continue
        data = load(d)
        s, why = match_score(env_summary(data[1]), ce)
        print(f"  {s:4d}  {os.path.basename(d)}  ({', '.join(why)})")
        if best is None or s > best[0]:
            best = (s, d, data, why)
    if best is None or best[0] < 0:
        print("  no usable baseline found"); return 2
    print(f"  -> {os.path.basename(best[1])}")
    if "same wine-or-not" not in best[3]:
        print("  WARNING: no baseline from the same kind of environment (Wine vs Windows): expect Wine-gap XFAIL/SKIP"
              " differences; record a baseline for this environment (run_all -Baseline auto)")
    print()
    return diff(best[1], run_dir, opts, A=best[2], B=cand)


def triage(run_dir, opts):
    cases, envs, exes, host = filt(load(run_dir), opts)
    es = env_summary(envs)
    out = {"run": run_dir, "env": {}, "exe_status": [], "cases": []}
    print(f"run: {run_dir}")
    if isinstance(host, dict):
        osd = (f"build {host.get('build')}.{host.get('ubr')} {host.get('display_version') or ''}" if host.get("build")
               else f"{host.get('host_os', '')} {host.get('wine', '')}")
        print(f"host: {host.get('slug')}  {osd}  cpu {host.get('cpu')}  git {host.get('git_rev')}  note {host.get('note') or ''}")
    for arch, e in sorted(es.items()):
        print(f"  [{arch}] os {e['os']}  wine {e['wine'] or '-'}  wow64 {e['wow64']}  cpu {e['cpu']}  xcr0 {e['xcr0']}  "
              f"api_xstate {e['api_xstate']}  version {e['version']}")
        out["env"][arch] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in e.items()}
    counts = {}
    for k, r in cases.items():
        counts[r.get("result")] = counts.get(r.get("result"), 0) + 1
    print("  results: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    bad_exe = [(k, s) for k, s in sorted(exes.items()) if s != "OK"]
    print(f"\n== EXE STATUS not OK: {len(bad_exe)}")
    for (a, e), s in bad_exe:
        print(f"  {a}/{e}: {s}")
        out["exe_status"].append({"arch": a, "exe": e, "status": s})
    groups = {"FAIL": [], "CRASH": [], "TIMEOUT": [], "XFAIL": []}
    for key in sorted(cases, key=lambda k: tuple(str(x) for x in k)):
        r = cases[key]
        if r.get("result") in groups: groups[r["result"]].append((key, r))
    nfail = 0
    for res in ("CRASH", "TIMEOUT", "FAIL", "XFAIL"):
        lst = groups[res]
        print(f"\n== {res}: {len(lst)}")
        for (a, exe, c), r in lst:
            wine = bool((es.get(a) or {}).get("wine"))
            flag = ""
            if res == "XFAIL" and not wine:
                flag = "  [XFAIL outside Wine: verify the guard / expectation]"
            if res != "XFAIL": nfail += 1
            print(f"  {a}/{exe}/{c}{flag}")
            if r.get("reason"): print(f"      reason: {str(r['reason'])[:240]}")
            if r.get("exception"): print(f"      exception: {json.dumps(r['exception'])[:240]}")
            for fl in (r.get("failures") or [])[:6]:
                print(f"      | {str(fl)[:240]}")
            out["cases"].append({"arch": a, "exe": exe, "case": c, "result": res, "reason": r.get("reason", ""),
                                 "failures": [str(x)[:500] for x in (r.get("failures") or [])[:10]],
                                 "xfail_outside_wine": bool(flag)})
    viol, nchk, epath = check_expectations(cases, es, opts)
    print(f"\n== EXPECTED-ON-WINDOWS rows ({os.path.basename(epath)}): {nchk} matched, {len(viol)} violated")
    if all((e or {}).get("wine") for e in es.values()):
        print("  (Wine run: the rows apply to real Windows only)")
    for v in viol:
        print(f"  {v['arch']}/{v['exe']}/{v['case']}: {v['result']}, expected {'/'.join(v['expect'])} on build {v['build']}")
        if v["reason"]: print(f"      reason: {str(v['reason'])[:240]}")
    out["expectation_violations"] = viol
    if "json" in opts:
        with open(opts["json"], "w") as f:
            json.dump(out, f, indent=1, default=str)
    nx = sum(1 for i in out["cases"] if i["xfail_outside_wine"])
    print(f"\nRESULT failing_cases={nfail} non_ok_exes={len(bad_exe)} xfail={len(groups['XFAIL'])} xfail_outside_wine={nx} "
          f"expectation_violations={len(viol)}")
    return 1 if (nfail or bad_exe or viol) else 0


def check_expectations(cases, es, opts):
    """Rows of expect_windows.json applied to a non-Wine run -> list of violations."""
    path = opts.get("expect") or os.path.join(opts.get("baselines", DEFAULT_BASELINES), "expect_windows.json")
    rules = (read_json_file(path) or {}).get("rules", [])
    out, checked = [], 0
    for (arch, exe, case), r in sorted(cases.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        e = es.get(arch) or {}
        if e.get("wine"): continue
        try: build = int(e.get("build") or 0)
        except Exception: build = 0
        for rule in rules:
            if rule.get("exe") != exe or rule.get("case") != case: continue
            if rule.get("arch") and rule["arch"] != arch: continue
            if "min_build" in rule and build < rule["min_build"]: continue
            if "max_build" in rule and build > rule["max_build"]: continue
            checked += 1
            if r.get("result") not in rule["expect"]:
                out.append({"arch": arch, "exe": exe, "case": case, "result": r.get("result"), "expect": rule["expect"],
                            "build": build, "reason": r.get("reason", "")})
            break
    return out, checked, path


def list_baselines(opts):
    root = opts.get("baselines", DEFAULT_BASELINES)
    bl = find_baselines(root)
    if not bl:
        print(f"no baselines under {root}"); return 0
    for d in bl:
        cases, envs, exes, host = load(d)
        es = env_summary(envs)
        counts = {}
        for r in cases.values(): counts[r.get("result")] = counts.get(r.get("result"), 0) + 1
        archs = ",".join(sorted(es))
        e = es.get("x64") or (next(iter(es.values())) if es else {})
        print(f"{os.path.basename(d):50s} arch={archs:8s} os={e.get('os', '?')} wine={e.get('wine') or '-'} "
              f"cpu={e.get('cpu', '?')} xcr0={e.get('xcr0')}  " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if _ else "1"
    for flag in ("baseline", "triage", "slug"):
        if opts.get(flag) not in (None, "1") and not args:   # --triage=<dir> form
            args = [opts[flag]]; opts[flag] = "1"
    try:
        if "list-baselines" in opts:
            return list_baselines(opts)
        if "slug" in opts:
            if len(args) != 1: print(__doc__); return 2
            s = slug_for(args[0])
            if not s: print("no env records", file=sys.stderr); return 2
            print(s); return 0
        if "triage" in opts:
            if len(args) != 1: print(__doc__); return 2
            return triage(args[0], opts)
        if "baseline" in opts:
            if len(args) != 1: print(__doc__); return 2
            return baseline_mode(args[0], opts)
        if len(args) != 2:
            print(__doc__); return 2
        return diff(args[0], args[1], opts)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
