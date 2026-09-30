param([Parameter(Mandatory=$true)][string]$RunTag, [switch]$NoOnnxOptimizer)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\results\b_export_v1'))
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runner = Join-Path $PSScriptRoot 'cubeai.py'
$progressPath = Join-Path $outputRoot ($RunTag + '.progress.json')
Set-Location -LiteralPath $repoRoot
try {
    foreach ($stage in @('analyze', 'generate', 'validate')) {
        @{status='running'; stage=$stage; pid=$PID; updated_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $progressPath -Encoding utf8
        $stageArgs = @('-X', 'utf8', '-u', $runner, $stage, '--timeout', '600', '--tag', "_$RunTag")
        if ($NoOnnxOptimizer) { $stageArgs += '--no-onnx-optimizer' }
        $stageProcess = Start-Process -FilePath $python -ArgumentList $stageArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputRoot "${RunTag}_${stage}.runner.stdout.log") -RedirectStandardError (Join-Path $outputRoot "${RunTag}_${stage}.runner.stderr.log") -PassThru
        while (-not $stageProcess.HasExited) {
            Start-Sleep -Seconds 2
            $converters = @(Get-CimInstance Win32_Process -Filter "Name = 'stedgeai.exe'" | Where-Object { $_.CommandLine -like "*$RunTag*" })
            foreach ($converter in $converters) {
                $memoryProcess = Get-Process -Id $converter.ProcessId -ErrorAction SilentlyContinue
                if ($memoryProcess -and $memoryProcess.PrivateMemorySize64 -gt 8GB) {
                    Stop-Process -Id $converter.ProcessId -Force
                    "Stopped analyzer at 8 GiB memory limit to preserve training." | Set-Content -LiteralPath (Join-Path $outputRoot "${RunTag}_${stage}.memory_limit.txt")
                }
            }
            $stageProcess.Refresh()
        }
        $stageProcess.WaitForExit()
        $stageReceipt = Join-Path $outputRoot "cubeai_${stage}_$RunTag.json"
        if (-not (Test-Path -LiteralPath $stageReceipt) -or (Get-Content -LiteralPath $stageReceipt -Raw | ConvertFrom-Json).status -ne 'complete') { throw "Cube.AI $stage failed; inspect cubeai_${stage}_$RunTag.json and its log." }
    }
    @{status='complete'; stage='validate'; pid=$PID; updated_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $progressPath -Encoding utf8
} catch {
    @{status='failed'; stage=$stage; error=$_.Exception.Message; pid=$PID; updated_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $progressPath -Encoding utf8
    throw
}
