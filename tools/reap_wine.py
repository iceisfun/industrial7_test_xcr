#!/usr/bin/env python3
"""Emergency cleanup: kill wine process groups started from xcr_prefix.* sandboxes (or any prefix whose
path contains a given substring) that have no test exe running. Safe against your real prefixes.
usage: reap_wine.py [prefix-substring=xcr_prefix] [--all]   (--all: kill even if a test exe still runs)"""
import os, sys, signal, collections
args = [a for a in sys.argv[1:] if not a.startswith("--")]
sub = args[0] if args else "xcr_prefix"
kill_all = "--all" in sys.argv
HELP = {"winedevice.exe","winhlp32.exe","svchost.exe","services.exe","plugplay.exe","rpcss.exe","explorer.exe","tabtip.exe",
        "wineserver","conhost.exe","start.exe","rundll32.exe","wineboot.exe","winemenubuilder.exe","msiexec.exe"}
def rd(p, mode="rb"):
    try:
        with open(p, mode) as f: return f.read()
    except Exception: return b"" if mode == "rb" else ""
g = collections.defaultdict(list)
for p in filter(str.isdigit, os.listdir("/proc")):
    pre = next((kv[11:].decode(errors="replace") for kv in rd(f"/proc/{p}/environ").split(b"\0") if kv.startswith(b"WINEPREFIX=")), None)
    if pre and sub in pre: g[pre].append(int(p))
killed = 0
for pre, pids in g.items():
    names = [rd(f"/proc/{x}/comm", "r").strip() for x in pids]
    active = any(n not in HELP and not n.startswith("wine") for n in names)
    if kill_all or not active:
        for x in pids:
            try: os.kill(x, signal.SIGKILL); killed += 1
            except Exception: pass
print(f"prefixes={len(g)} killed={killed}")
