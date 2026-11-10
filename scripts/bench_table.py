#!/usr/bin/env python3
"""Render benchmark PERF rows as the docs/GUIDANCE.md section 5 table: metric x scalar / SSE2 / AVX2 /
AVX-512, each cell = median (IQR), with an environment header (CPU, OS, affinity, frequency) and,
when a second run is given, a ratio column.

    scripts/bench_table.py RUN [--compare RUN2] [--format text|md|csv] [--exe SUBSTR] [--metric SUBSTR]

RUN / RUN2: a --json file written by an exe (JSON lines), or a directory searched recursively for them
(e.g. results/ from scripts/run_all.sh or run_all.ps1). Files that are not suite JSON lines are ignored.

Columns. The tier of a PERF metric comes from tokens in its name (split on "_"); the widest tier token
wins and all tier tokens are removed to form the row name, so memcpy_sse2_4KB_aligned and
memcpy_avx_4KB_aligned land in one row "memcpy_4KB_aligned":
    AVX-512: avx512 avx512f avx512bw avx512dq zmm 512        AVX2: avx avx2 fma ymm vaes avxvnni
    SSE2:    sse sse2 sse3 ssse3 sse4 sse41 sse42 ni aesni pclmulqdq popcnt pcmpistri psadbw movntdq
    scalar:  scalar sw
Metrics with no tier token (crt, rep_movsb, API calls, ...) go to the "other" column.
Cells: median (IQR = q3 - q1) from the perf record; "-" when the run had no such row. Records from
older exes without q1/q3 show the median only.

Ratio column (--compare): RUN2 relative to RUN, per tier, oriented so that > 1 means RUN2 is faster:
RUN2/RUN for rates (GB/s, MPix/s, ...), RUN/RUN2 for times (units starting with s/ ms/ us/ ns/ or
cycles). Rows are matched by (arch, exe, metric). Exit status: 0, or 2 for unusable arguments.
"""
import argparse
import json
import os
import sys

TIERS = ["scalar", "SSE2", "AVX2", "AVX-512", "other"]
TOK = {
    "AVX-512": {"avx512", "avx512f", "avx512bw", "avx512dq", "zmm", "512"},
    "AVX2": {"avx", "avx2", "fma", "ymm", "vaes", "avxvnni"},
    "SSE2": {"sse", "sse2", "sse3", "ssse3", "sse4", "sse41", "sse42", "ni", "aesni", "pclmulqdq", "popcnt",
             "pcmpistri", "psadbw", "movntdq"},
    "scalar": {"scalar", "sw"},
}
ORDER = ["AVX-512", "AVX2", "SSE2", "scalar"]   # widest first
TIME_PREFIX = ("s/", "ms/", "us/", "ns/", "cycles")
ENV_INFO = ("tsc_ghz_estimate", "pinned_cpu", "bench_cpu", "process_affinity", "mode", "l1d_bytes", "l2_bytes", "l3_bytes")


def classify(metric):
    toks = metric.split("_")
    tier = None
    for t in ORDER:
        if any(x.lower() in TOK[t] for x in toks):
            tier = t
            break
    if tier is None:
        return "other", metric
    alltok = set().union(*TOK.values())
    rest = [x for x in toks if x.lower() not in alltok]
    return tier, "_".join(rest) if rest else metric


def json_files(path):
    if os.path.isfile(path):
        yield path
        return
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".json") or f.endswith(".jsonl"):
                yield os.path.join(root, f)


def load(path):
    """-> (envs: {(arch, exe): env}, rows: {(arch, exe, metric): perf+case}, infos: {(arch, exe): {k: v}})"""
    envs, rows, infos = {}, {}, {}
    if not os.path.exists(path):
        sys.exit("bench_table: no such file or directory: %s" % path)
    for fn in json_files(path):
        try:
            fh = open(fn, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                t = r.get("type")
                if t == "env":
                    envs[(r.get("arch", "?"), r.get("exe", "?"))] = r
                elif t == "case":
                    key = (r.get("arch", "?"), r.get("exe", "?"))
                    inf = infos.setdefault(key, {})
                    for k, v in (r.get("info") or {}).items():
                        if k in ENV_INFO and k not in inf:
                            inf[k] = v
                    for p in r.get("perf") or []:
                        m = p.get("metric")
                        if m is None or not isinstance(p.get("value"), (int, float)):
                            continue
                        rows[key + (m,)] = dict(p, case=r.get("case"), result=r.get("result"))
    return envs, rows, infos


def is_time(unit):
    u = (unit or "").lower()
    return u.startswith(TIME_PREFIX) or u == "s"


def fmt_num(v):
    if v is None:
        return "-"
    a = abs(v)
    if a >= 1000:
        return "%.0f" % v
    if a >= 100:
        return "%.1f" % v
    if a >= 1:
        return "%.2f" % v
    return "%.3g" % v


def cell(p):
    if p is None:
        return "-"
    s = fmt_num(p["value"])
    if isinstance(p.get("q1"), (int, float)) and isinstance(p.get("q3"), (int, float)):
        s += " (%s)" % fmt_num(p["q3"] - p["q1"])
    return s


def ratio(a, b):
    """speed of run2 relative to run1 (> 1 = run2 faster)"""
    if a is None or b is None:
        return None
    va, vb = a["value"], b["value"]
    if not va or not vb:
        return None
    return va / vb if is_time(a.get("unit")) else vb / va


def env_lines(envs, infos, label):
    out = []
    for key in sorted(envs):
        e = envs[key]
        cpu = e.get("cpu") or {}
        os_ = e.get("os") or {}
        inf = infos.get(key, {})
        freq = []
        if "tsc_ghz_estimate" in inf:
            freq.append("TSC ~%s GHz" % str(inf["tsc_ghz_estimate"]).split(" ")[0])
        if e.get("qpc_freq"):
            freq.append("QPC %s Hz" % e["qpc_freq"])
        pin = inf.get("pinned_cpu") or inf.get("bench_cpu")
        out.append("%s %s/%s: %s | %s build %s.%s%s | ncpu %s affinity %s%s | %s | xcr0 %s | version %s" % (
            label, key[0], key[1], cpu.get("brand", "?"), os_.get("product", "?"), os_.get("build", "?"), os_.get("ubr", "?"),
            (" wine " + e["wine"]) if e.get("wine") else "", e.get("ncpu", "?"), e.get("affinity", "?"),
            (" pinned cpu " + str(pin)) if pin is not None else "", ", ".join(freq) or "frequency n/a",
            e.get("xcr0"), e.get("version", "?")))
    return out


def build(rows, rows2, exe_f, metric_f):
    """-> {(arch, exe): {(base, unit): {tier: (p1, p2)}}} preserving first-seen order"""
    groups = {}
    keys = list(rows.keys()) + [k for k in (rows2 or {}) if k not in rows]
    for k in keys:
        arch, exe, m = k
        if exe_f and exe_f not in exe:
            continue
        if metric_f and metric_f not in m:
            continue
        p1 = rows.get(k)
        p2 = (rows2 or {}).get(k)
        unit = (p1 or p2).get("unit", "")
        tier, base = classify(m)
        g = groups.setdefault((arch, exe), {})
        slot = g.setdefault((base, unit), {})
        if tier in slot:   # two metrics collapse onto one cell: keep them apart under their full name
            slot = g.setdefault((m, unit), {})
        slot[tier] = (p1, p2)
    return groups


def render(groups, compare, fmt):
    head = ["metric", "unit"] + [t + " median (IQR)" for t in TIERS]
    if compare:
        head.append("ratio run2/run1 (>1 = run2 faster)")
    tables = []
    for (arch, exe), g in groups.items():
        used = [t for t in TIERS if any(t in slot for slot in g.values())]
        cols = [0, 1] + [2 + TIERS.index(t) for t in used] + ([len(head) - 1] if compare else [])
        body = []
        for (base, unit), slot in g.items():
            r = [base, unit]
            for t in TIERS:
                r.append(cell(slot.get(t, (None, None))[0]))
            if compare:
                parts = []
                for t in TIERS:
                    if t in slot:
                        q = ratio(*slot[t])
                        parts.append("%s %s" % (t, "%.3f" % q if q is not None else "n/a"))
                r.append(", ".join(parts))
            body.append(r)
        tables.append(("%s / %s" % (arch, exe), [head[i] for i in cols], [[r[i] for i in cols] for r in body]))
    out = []
    for title, h, body in tables:
        if fmt == "csv":
            import csv
            import io
            buf = io.StringIO()
            w = csv.writer(buf)
            for r in [[title] + h] + [[title] + r for r in body]:
                w.writerow(r)
            out.append(buf.getvalue().rstrip("\n"))
        elif fmt == "md":
            out.append("### %s\n" % title)
            out.append("| " + " | ".join(h) + " |")
            out.append("|" + "|".join("---" if i < 2 else "---:" for i in range(len(h))) + "|")
            for r in body:
                out.append("| " + " | ".join(x.replace("|", "\\|") for x in r) + " |")
            out.append("")
        else:
            wd = [max(len(str(x)) for x in col) for col in zip(h, *body)]
            out.append("== " + title)
            out.append("  ".join(str(x).ljust(wd[i]) if i < 2 else str(x).rjust(wd[i]) for i, x in enumerate(h)))
            for r in body:
                out.append("  ".join(str(x).ljust(wd[i]) if i < 2 else str(x).rjust(wd[i]) for i, x in enumerate(r)))
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="JSON-lines file or results directory")
    ap.add_argument("--compare", metavar="RUN2", help="second run: adds the ratio column")
    ap.add_argument("--format", choices=["text", "md", "csv"], default="text")
    ap.add_argument("--exe", help="only exes whose name contains this")
    ap.add_argument("--metric", help="only metrics whose name contains this")
    a = ap.parse_args()
    envs, rows, infos = load(a.run)
    envs2, rows2, infos2 = load(a.compare) if a.compare else ({}, None, {})
    if not rows and not rows2:
        print("bench_table: no PERF rows found", file=sys.stderr)
        return 0
    groups = build(rows, rows2, a.exe, a.metric)
    pre = "> " if a.format == "md" else ""
    if a.format != "csv":
        for l in env_lines(envs, infos, "run1" if a.compare else "env"):
            if a.exe and a.exe not in l.split(":")[0]:
                continue
            print(pre + l + ("  " if pre else ""))
        for l in env_lines(envs2, infos2, "run2"):
            if a.exe and a.exe not in l.split(":")[0]:
                continue
            print(pre + l + ("  " if pre else ""))
        print()
    print(render(groups, bool(a.compare), a.format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
