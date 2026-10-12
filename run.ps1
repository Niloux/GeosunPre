# Run or resume the production pipeline with the local Windows environment.
$ErrorActionPreference = 'Stop'
$pipelineArgs = $args
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Prepare the environment first: .\setup-windows.ps1'
}
Push-Location $PSScriptRoot
try {
    & $python -X utf8 -m geosunpre run @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
