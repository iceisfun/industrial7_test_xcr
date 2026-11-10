@echo off
rem Record a baseline of the whole suite on this Windows machine (x64 exes + x86 exes under WOW64).
rem Layout: <dir>\build\x64\*.exe, <dir>\build\x86\*.exe, <dir>\scripts\ (this file), optional <dir>\corpus\.
rem Writes <dir>\results\baselines\<os>-<cpu>\ and zips it to <dir>\results\baseline-<os>-<cpu>.zip.
rem Usage: run_baseline.cmd ["note, e.g. bare metal, VBS off"]
setlocal
set "HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%run_all.ps1" -Baseline auto -Note "%~1"
set RC=%ERRORLEVEL%
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d = Get-ChildItem -Directory '%HERE%..\results\baselines' | Sort-Object LastWriteTime | Select-Object -Last 1;" ^
  "if ($d) { $z = Join-Path (Split-Path $d.Parent.FullName) ('baseline-' + $d.Name + '.zip');" ^
  "Compress-Archive -Force -Path $d.FullName -DestinationPath $z; Write-Host ('zip: ' + $z) }"
echo run_all exit code %RC% (0 = no FAIL/CRASH; 1 = something to triage)
exit /b %RC%
