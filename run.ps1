$ErrorActionPreference = 'Stop'
$tools = Join-Path $PSScriptRoot 'tools\windows'
if ($args[0] -eq 'prepare-models') {
    if ($args.Count -gt 2) { throw 'Usage: .\run.ps1 prepare-models [path-to-sam3.pt]' }
    $options = @{}
    if ($args.Count -eq 2) { $options.Sam3 = (Resolve-Path -LiteralPath $args[1]).Path }
    & (Join-Path $tools 'prepare-models.ps1') @options
    return
}
& (Join-Path $tools 'build.ps1') | Out-Host
$program = Join-Path $PSScriptRoot 'build\ninja\bin\geosunpre.exe'
$previousModels = $env:GEOSUNPRE_MODELS
try {
    $env:GEOSUNPRE_MODELS = Join-Path $PSScriptRoot 'models'
    Write-Host "Development executable: $program"
    & $program @args
    $code = $LASTEXITCODE
} finally { $env:GEOSUNPRE_MODELS = $previousModels }
exit $code
