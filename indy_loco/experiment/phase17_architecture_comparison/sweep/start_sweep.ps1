$ErrorActionPreference = 'Stop'
$sweepRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $sweepRoot '..\..\..\..')).Path
$outputRoot = Join-Path $sweepRoot '..\results\sweep_v1'
$interpreter = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runner = 'indy_loco\experiment\phase17_architecture_comparison\sweep\run_sweep.py'
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'phase17_architecture_comparison'
})
if ($active.Count -gt 0) {
    $active | Select-Object ProcessId, ParentProcessId, CommandLine | Format-List
    throw 'A Phase17 Python process is already active. No new process was launched.'
}
if (Test-Path -LiteralPath (Join-Path $outputRoot 'STOP_AFTER_FOLD')) {
    throw 'STOP_AFTER_FOLD exists. Remove it only when ready to resume.'
}
$logRoot = Join-Path $outputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
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
