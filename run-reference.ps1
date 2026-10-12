# Explicit reference implementation; the default Windows entry is native.
$ErrorActionPreference = 'Stop'
$pipelineArgs = $args
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Prepare the reference environment first: .\setup-windows.ps1' }
Push-Location $PSScriptRoot
try {
    $ErrorActionPreference = 'Continue'
    & $python -X utf8 -m geosunpre run @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
