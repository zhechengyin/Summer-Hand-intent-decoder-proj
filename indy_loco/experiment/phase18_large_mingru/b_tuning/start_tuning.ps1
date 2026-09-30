$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\results\b_tuning_v1'))
$interpreter = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runner = 'indy_loco\experiment\phase18_large_mingru\b_tuning\train.py'

foreach ($required in @($interpreter, (Join-Path $repoRoot $runner), (Join-Path $outputRoot 'preflight.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required training file is missing: $required. Complete the training preflight first."
    }
}
if (Test-Path -LiteralPath (Join-Path $outputRoot 'STOP_AFTER_FOLD')) {
    throw 'B tuning STOP_AFTER_FOLD exists. Inspect the stopped run before explicitly resuming it.'
}

# Enumeration is fail-closed; the Python runner additionally holds OS locks.
$active = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -match 'phase17_architecture_comparison|phase18_large_mingru'
})
if ($active.Count -gt 0) {
    $active | Select-Object ProcessId, ParentProcessId, CommandLine | Format-List
    throw 'A Phase17/18 Python process is active. No training was launched.'
}

$logRoot = Join-Path $outputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$stdout = Join-Path $logRoot ($stamp + '.stdout.log')
$stderr = Join-Path $logRoot ($stamp + '.stderr.log')
$arguments = @('-X', 'utf8', '-u', $runner, '--device', 'cuda', '--threads', '4')
if (Test-Path -LiteralPath (Join-Path $outputRoot 'config.json')) {
    $arguments += '--resume'
}

$process = Start-Process -FilePath $interpreter -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record = [ordered]@{
    launcher_pid = $process.Id
    started_at = (Get-Date).ToString('o')
    working_directory = $repoRoot
    command = $interpreter + ' ' + ($arguments -join ' ')
    stdout = $stdout
    stderr = $stderr
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot ($stamp + '.launch.json')) -Encoding utf8
$record | ConvertTo-Json
