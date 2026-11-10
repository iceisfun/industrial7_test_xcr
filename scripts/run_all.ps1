<#
.SYNOPSIS
  Run every suite exe with --json, a per-exe timeout, and collect results (Windows PowerShell 5.1+ / pwsh).

.DESCRIPTION
  Looks for exes in -ExeDir; by default in <repo>\build\x64 and <repo>\build\x86 (each arch separately),
  or in the script's own folder / the current folder when the exes were copied without the build tree.
  For each exe writes  <OutDir>\<arch>\<name>.json  (JSON lines: env, case..., summary)
                       <OutDir>\<arch>\<name>.txt   (stdout + stderr)
  and <OutDir>\<arch>\run.json with exit codes, status, duration and the env record.
  Status: OK (exit 0), FAIL (1), HARNESS_ERROR (2), TIMEOUT (killed by this runner),
          CRASH (any other exit code, or no summary record in the JSON).
  Compare two result folders with scripts\compare_results.py.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -ExeDir C:\xcr\x64 -OutDir C:\xcr\results -TimeoutSec 300
#>
param(
    [string]$ExeDir = "",
    [string]$OutDir = "",
    [int]$TimeoutSec = 600,
    [string]$Only = "",
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path (Get-Location) "results" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

$dirs = @()
if ($ExeDir) { $dirs += (Resolve-Path $ExeDir).Path }
else {
    foreach ($a in "x64", "x86") { $d = Join-Path $repo "build\$a"; if (Test-Path $d) { $dirs += $d } }
    if (-not $dirs) {
        foreach ($d in $PSScriptRoot, (Get-Location).Path) { if (Get-ChildItem -Path $d -Filter *.exe -ErrorAction SilentlyContinue) { $dirs += $d; break } }
    }
}
if (-not $dirs) { Write-Error "no exe directory found (use -ExeDir)"; exit 2 }

function Stop-Tree([int]$procId) {
    # kill the exe and any children it spawned (its own job object also kills them when it dies)
    & taskkill.exe /T /F /PID $procId 2>&1 | Out-Null
}

$worst = 0
foreach ($dir in $dirs) {
    $leaf = Split-Path -Leaf $dir
    $arch = if ($leaf -in @("x64", "x86")) { $leaf } else { "native" }
    $od = Join-Path $OutDir $arch
    New-Item -ItemType Directory -Force -Path $od | Out-Null
    $exes = @()
    $envRec = $null
    foreach ($exe in Get-ChildItem -Path $dir -Filter *.exe | Sort-Object Name) {
        $name = [IO.Path]::GetFileNameWithoutExtension($exe.Name)
        if ($Only -and $name -notlike "*$Only*") { continue }
        $json = Join-Path $od "$name.json"
        $txt = Join-Path $od "$name.txt"
        $err = Join-Path $od "$name.stderr.txt"
        Remove-Item -Force -ErrorAction SilentlyContinue $json, $txt, $err
        $argList = @("`"--json=$json`"") + $ExtraArgs
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath $exe.FullName -ArgumentList $argList -NoNewWindow -PassThru `
                           -RedirectStandardOutput $txt -RedirectStandardError $err -WorkingDirectory $od
        $null = $p.Handle   # keep the handle so ExitCode stays available
        $status = $null
        if (-not $p.WaitForExit($TimeoutSec * 1000)) {
            Stop-Tree $p.Id
            $p.WaitForExit(10000) | Out-Null
            $status = "TIMEOUT"
        }
        $sw.Stop()
        $code = if ($status -eq "TIMEOUT") { $null } else { [int]$p.ExitCode }   # Int32: crash codes are negative
        if (Test-Path $err) {
            $e = Get-Content -Raw $err
            if ($e) { Add-Content -Path $txt -Value "`n--- stderr ---`n$e" }
            Remove-Item -Force $err
        }
        $summary = $null
        if (Test-Path $json) {
            foreach ($line in Get-Content $json) {
                try { $r = $line | ConvertFrom-Json } catch { continue }
                if ($r.type -eq "summary") { $summary = $r }
                elseif ($r.type -eq "env" -and -not $envRec) { $envRec = $r }
            }
        }
        if (-not $status) {
            if (-not $summary) { $status = "CRASH" }
            elseif ($code -eq 0) { $status = "OK" }
            elseif ($code -eq 1) { $status = "FAIL" }
            elseif ($code -eq 2) { $status = "HARNESS_ERROR" }
            else { $status = "CRASH" }
        }
        if ($status -ne "OK") { $worst = 1 }
        $codeHex = if ($null -ne $code) { "0x{0:X8}" -f $code } else { "n/a" }
        $sumLine = (Select-String -Path $txt -Pattern '^SUMMARY' -ErrorAction SilentlyContinue | Select-Object -First 1)
        Write-Host ("{0,-6} {1,-28} {2,-13} exit={3} {4}" -f $arch, $name, $status, $codeHex, $(if ($sumLine) { $sumLine.Line.Substring(8) } else { "" }))
        $exes += [ordered]@{ name = $name; arch = $arch; exit_code = $code; exit_code_hex = $codeHex; status = $status;
                             duration_s = [math]::Round($sw.Elapsed.TotalSeconds, 3); summary = $summary }
    }
    $run = [ordered]@{
        runner = "run_all.ps1"; arch = $arch; exe_dir = $dir; timeout_s = $TimeoutSec
        finished = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        host = [ordered]@{ os = [Environment]::OSVersion.VersionString; is64 = [Environment]::Is64BitOperatingSystem
                           ps = $PSVersionTable.PSVersion.ToString() }
        env = $envRec; exes = $exes
    }
    $run | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path (Join-Path $od "run.json")
}
exit $worst
