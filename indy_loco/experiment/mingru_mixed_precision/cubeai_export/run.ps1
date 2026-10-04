param([string]$Tag = ("runtime_" + (Get-Date -Format 'yyyyMMdd_HHmmss')))
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$results = Join-Path $PSScriptRoot '..\results\b2_cubeai_export_v1'
Push-Location $repo
try {
    foreach ($script in @('export.py', 'compact.py', 'runtime_weights.py')) {
        & $python -X utf8 (Join-Path $PSScriptRoot $script)
        if ($LASTEXITCODE -ne 0) { throw "$script failed" }
    }
    & $python -X utf8 (Join-Path $PSScriptRoot 'convert.py') (Join-Path $results 'b2_620kb_runtime_weights.onnx') --tag $Tag --external-inputs --dll --timeout 240
    if ($LASTEXITCODE -ne 0) { throw 'Cube.AI conversion wrapper failed' }
    & $python -X utf8 (Join-Path $PSScriptRoot 'package_validate.py') --tag $Tag
    if ($LASTEXITCODE -ne 0) { throw 'Generated C validation failed; do not deploy this package' }
} finally {
    Pop-Location
}
