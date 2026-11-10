<#
.SYNOPSIS
  Run every suite exe with --json, a per-exe timeout, and collect results (Windows PowerShell 5.1+ / pwsh).

.DESCRIPTION
  Looks for exes in -ExeDir; by default in <repo>\build\x64 and <repo>\build\x86 (each arch separately),
  or in <dir>\x64 and <dir>\x86 next to the script / in the current folder, or in the script's own
  folder / the current folder when the exes were copied flat.
  For each exe writes  <OutDir>\<arch>\<name>.json  (JSON lines: env, case..., summary)
                       <OutDir>\<arch>\<name>.txt   (stdout + stderr)
  and <OutDir>\<arch>\run.json with exit codes, status, duration, exe sha256 and the env record,
  and <OutDir>\host.json with the OS build / edition / CPU / hypervisor / git revision of this run.
  Status: OK (exit 0), FAIL (1), HARNESS_ERROR (2), TIMEOUT (killed by this runner),
          CRASH (any other exit code, or no summary record in the JSON).

  Baselines (results\baselines\<os>-<cpu>\, committed to the repository):
    -Baseline auto       writes to <repo>\results\baselines\<slug>, slug = e.g. win10-22h2-19045-intel-core-i7-8700
    -Baseline <name>     writes to <repo>\results\baselines\<name>
  The x86 directory is run as a WOW64 process on 64-bit Windows; both arches go to the same baseline.
  Compare a run with a committed baseline (on any host with Python 3):
    python3 scripts/compare_results.py --baseline <run_dir>            (picks the closest committed baseline)
    python3 scripts/compare_results.py <baseline_dir> <run_dir>
  -Compare <dir> runs that comparison right after the run when python is on PATH.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Baseline auto
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Arch x86 -Only context -OutDir C:\xcr\r
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -ExeDir C:\xcr\x64 -OutDir C:\xcr\results -TimeoutSec 300
#>
param(
    [string]$ExeDir = "",
    [string]$OutDir = "",
    [int]$TimeoutSec = 600,
    [string]$Only = "",
    [ValidateSet("all", "x64", "x86")][string]$Arch = "all",
    [string]$Baseline = "",
    [string]$Compare = "",
    [string]$Note = "",
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

function Get-RegValue([string]$path, [string]$name) {
    try { return (Get-ItemProperty -Path $path -Name $name -ErrorAction Stop).$name } catch { return $null }
}

function Get-HostInfo {
    $cv = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    $build = [int](Get-RegValue $cv "CurrentBuildNumber")
    $h = [ordered]@{
        product        = Get-RegValue $cv "ProductName"      # says "Windows 10" on Windows 11 too: use build
        edition        = Get-RegValue $cv "EditionID"
        installation   = Get-RegValue $cv "InstallationType"
        display_version = Get-RegValue $cv "DisplayVersion"  # 22H2, 24H2 (20H2 and later)
        release_id     = Get-RegValue $cv "ReleaseId"
        build          = $build
        ubr            = Get-RegValue $cv "UBR"
        version_string = [Environment]::OSVersion.VersionString
        os64           = [Environment]::Is64BitOperatingSystem
        cpu            = $null; cpu_vendor = $null; cpu_identifier = $null
        cores          = $null; logical = [Environment]::ProcessorCount
        hypervisor_present = $null; manufacturer = $null; model = $null; memory_gb = $null
        ps             = $PSVersionTable.PSVersion.ToString()
        git_rev        = $null; note = $Note
    }
    $cpuKey = "HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    $h.cpu = Get-RegValue $cpuKey "ProcessorNameString"
    $h.cpu_vendor = Get-RegValue $cpuKey "VendorIdentifier"
    $h.cpu_identifier = Get-RegValue $cpuKey "Identifier"
    try {
        $p = @(Get-CimInstance Win32_Processor -ErrorAction Stop)
        $h.cores = ($p | Measure-Object -Property NumberOfCores -Sum).Sum
        if (-not $h.cpu) { $h.cpu = $p[0].Name }
        $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
        $h.hypervisor_present = [bool]$cs.HypervisorPresent   # also true on a host that runs VBS / Hyper-V
        $h.manufacturer = $cs.Manufacturer; $h.model = $cs.Model
        $h.memory_gb = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    } catch { }
    if ($h.cpu) { $h.cpu = ($h.cpu -replace '\s+', ' ').Trim() }
    try {
        $g = & git -C $repo rev-parse --short=12 HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $g) { $h.git_rev = "$g".Trim() }
    } catch { }   # no git (exes copied without the tree): the env record's "version" names the build
    return $h
}

function Get-BaselineSlug($h) {
    $os = "windows"
    if ($h.installation -eq "Server") { $os = "server$($h.build)" }
    elseif ($h.build -ge 22000) { $os = "win11" }
    elseif ($h.build -ge 10240) { $os = "win10" }
    $rel = if ($h.display_version) { $h.display_version } elseif ($h.release_id) { $h.release_id } else { "" }
    $parts = @($os)
    if ($rel) { $parts += $rel.ToLower() }
    $parts += "$($h.build)"
    $c = "$($h.cpu)" -replace '\((R|TM|tm|r)\)', ' ' -replace '@.*$', '' -replace '\b(CPU|Processor|\d+-Core)\b', ' ' `
                      -replace '\bwith Radeon.*$', ''
    $c = ($c.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
    if ($c.Length -gt 40) { $c = $c.Substring(0, 40).Trim('-') }
    if (-not $c) { $c = "cpu" }
    return (($parts -join "-") + "-" + $c)
}

$hostInfo = Get-HostInfo
if ($Baseline) {
    $name = if ($Baseline -eq "auto") { Get-BaselineSlug $hostInfo } else { $Baseline }
    if ($name -notmatch '^[A-Za-z0-9._-]+$') { Write-Error "bad baseline name '$name'"; exit 2 }
    if ($OutDir) { Write-Error "-Baseline and -OutDir are exclusive"; exit 2 }
    $OutDir = Join-Path $repo "results\baselines\$name"
    Write-Host "baseline: $OutDir"
}
if (-not $OutDir) { $OutDir = Join-Path (Get-Location) "results" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

$dirs = @()
$archs = if ($Arch -eq "all") { @("x64", "x86") } else { @($Arch) }
if ($ExeDir) { $dirs += (Resolve-Path $ExeDir).Path }
else {
    foreach ($root in (Join-Path $repo "build"), $PSScriptRoot, (Get-Location).Path) {
        foreach ($a in $archs) { $d = Join-Path $root $a; if ((Test-Path $d) -and ($dirs -notcontains $d)) { $dirs += $d } }
        if ($dirs) { break }
    }
    if (-not $dirs) {
        foreach ($d in $PSScriptRoot, (Get-Location).Path) { if (Get-ChildItem -Path $d -Filter *.exe -ErrorAction SilentlyContinue) { $dirs += $d; break } }
    }
}
if (-not $dirs) { Write-Error "no exe directory found (use -ExeDir)"; exit 2 }

function Stop-Tree([int]$procId) {
    # kill the exe and any children it spawned (its own job object also kills them when it dies)
    try { & taskkill.exe /T /F /PID $procId 2>&1 | Out-Null } catch { }
    try { Stop-Process -Id $procId -Force -ErrorAction Stop } catch { }   # already gone after taskkill: fine
}

$hostInfo["started"] = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$hostInfo["runner"] = "run_all.ps1"; $hostInfo["timeout_s"] = $TimeoutSec
$hostInfo["only"] = $Only; $hostInfo["extra_args"] = $ExtraArgs
$hostInfo["slug"] = Get-BaselineSlug $hostInfo
$hostInfo["arches"] = @()

$worst = 0
foreach ($dir in $dirs) {
    $leaf = Split-Path -Leaf $dir
    $arch = if ($leaf -in @("x64", "x86")) { $leaf } else { "native" }
    $hostInfo["arches"] += $arch
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
        $sha = $null
        try { $sha = (Get-FileHash -Algorithm SHA256 -Path $exe.FullName).Hash.ToLower() } catch { }
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
                             duration_s = [math]::Round($sw.Elapsed.TotalSeconds, 3); sha256 = $sha; summary = $summary }
    }
    $runPath = Join-Path $od "run.json"
    if ($Only -and (Test-Path $runPath)) {
        # a filtered re-run keeps the other exes of the earlier run in run.json
        try {
            $old = Get-Content -Raw $runPath | ConvertFrom-Json
            $done = @($exes | ForEach-Object { $_.name })
            foreach ($e in @($old.exes)) { if ($e -and ($done -notcontains $e.name)) { $exes += $e } }
            $exes = @($exes | Sort-Object { $_.name })
            if (-not $envRec) { $envRec = $old.env }
        } catch { }
    }
    $run = [ordered]@{
        runner = "run_all.ps1"; arch = $arch; exe_dir = $dir; timeout_s = $TimeoutSec
        finished = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        host = [ordered]@{ os = [Environment]::OSVersion.VersionString; is64 = [Environment]::Is64BitOperatingSystem
                           ps = $PSVersionTable.PSVersion.ToString(); build = $hostInfo.build; ubr = $hostInfo.ubr
                           display_version = $hostInfo.display_version; cpu = $hostInfo.cpu; git_rev = $hostInfo.git_rev }
        env = $envRec; exes = $exes
    }
    $run | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $runPath
}
$hostInfo["finished"] = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$hostInfo["worst"] = $worst
$hostInfo | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path (Join-Path $OutDir "host.json")
Write-Host "results: $OutDir  (slug $($hostInfo.slug))"

if ($Compare) {
    $py = $null
    foreach ($c in "python3", "python", "py") { if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break } }
    if (-not $py) { Write-Warning "no python on PATH: run scripts/compare_results.py elsewhere" }
    else {
        $cmp = Join-Path $PSScriptRoot "compare_results.py"
        if ($Compare -eq "auto") { & $py $cmp --baseline $OutDir } else { & $py $cmp $Compare $OutDir }
    }
}
exit $worst
