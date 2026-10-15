$ErrorActionPreference = 'Stop'
$program = Join-Path $PSScriptRoot 'build\ninja\bin\geosunpre.exe'
if (-not (Test-Path -LiteralPath $program)) { throw 'Run build.ps1 first.' }
$previousModels = $env:GEOSUNPRE_MODELS
try {
    $env:GEOSUNPRE_MODELS = Join-Path $PSScriptRoot 'models'
    Write-Host "Development executable: $program"
    & $program @args
    $code = $LASTEXITCODE
} finally { $env:GEOSUNPRE_MODELS = $previousModels }
exit $code
