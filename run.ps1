$ErrorActionPreference = 'Stop'
$program = Join-Path $PSScriptRoot 'dist\geosunpre.exe'
if (-not (Test-Path -LiteralPath $program)) { $program = Join-Path $PSScriptRoot 'build\bin\Release\geosunpre.exe' }
if (-not (Test-Path -LiteralPath $program)) { throw 'Run build.ps1 first.' }
& $program @args
exit $LASTEXITCODE
