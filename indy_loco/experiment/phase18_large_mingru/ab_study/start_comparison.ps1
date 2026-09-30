$ErrorActionPreference = 'Stop'

$phaseRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $phaseRoot '..\..\..\..')).Path
$outputRoot = Join-Path $phaseRoot '..\results\ab_comparison_v1'
$interpreter = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runner = 'indy_loco\experiment\phase18_large_mingru\ab_study\train.py'
$diagnosticReceipt = Join-Path $phaseRoot '..\results\ab_diagnostics_v1\summary.json'

if (-not (Test-Path -LiteralPath $diagnosticReceipt -PathType Leaf)) {
    throw 'A/B diagnostics must finish before training. Missing ab_diagnostics_v1/summary.json.'
}

if (-not (Test-Path -LiteralPath $interpreter -PathType Leaf)) {
    throw "Repository Python interpreter is missing: $interpreter"
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $runner) -PathType Leaf)) {
    throw "Training runner is missing: $runner"
}

# Fail closed if process enumeration is unavailable. The Python runner also
# holds OS locks, so a concurrent launch cannot overlap the checked training.
$active = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -match 'phase17_architecture_comparison|phase18_large_mingru'
})
if ($active.Count -gt 0) {
    $active | Select-Object ProcessId, ParentProcessId, CommandLine | Format-List
    throw 'A Phase17 or Phase18 Python process is already active. No new process was launched.'
}
if (Test-Path -LiteralPath (Join-Path $outputRoot 'STOP_AFTER_FOLD')) {
    throw 'Phase18 A/B STOP_AFTER_FOLD exists. Remove it only when ready to resume Phase18 A/B comparison.'
}

$logRoot = Join-Path $outputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$stdout = Join-Path $logRoot ($stamp + '.stdout.log')
$stderr = Join-Path $logRoot ($stamp + '.stderr.log')
$arguments = @('-X', 'utf8', '-u', $runner, '--device', 'cuda')
if (Test-Path -LiteralPath (Join-Path $outputRoot 'config.json')) {
    $arguments += '--resume'
}

$process = Start-Process -FilePath $interpreter -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record = [ordered]@{
    launcher_pid = $process.Id
    started_at = (Get-Date).ToString('o')
    working_directory = $repoRoot
    command = $interpreter + ' ' + ($arguments -join ' ')
    stdout = [System.IO.Path]::GetFullPath($stdout)
    stderr = [System.IO.Path]::GetFullPath($stderr)
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot ($stamp + '.launch.json')) -Encoding utf8
$record | ConvertTo-Json
