# Run or resume the standalone C++/CUDA production pipeline.
$ErrorActionPreference = 'Stop'
$pipelineArgs = $args
$program = Join-Path $PSScriptRoot '.cache\native\runtime\geosunpre-native.exe'
if (-not (Test-Path -LiteralPath $program)) {
    $program = Join-Path $PSScriptRoot '.cache\native\build\bin\geosunpre-native.exe'
}
if (-not (Test-Path -LiteralPath $program)) {
    throw 'Build the native runtime first: .\setup-native.ps1'
}
Push-Location $PSScriptRoot
try {
    # Native stderr also carries progress. Its exit code determines failure.
    $ErrorActionPreference = 'Continue'
    & $program run @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
