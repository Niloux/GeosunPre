[CmdletBinding()]
param([string]$Models = (Join-Path $PSScriptRoot 'models'))
$ErrorActionPreference = 'Stop'
$destination = Join-Path $PSScriptRoot 'dist'
if (Test-Path -LiteralPath $destination) { throw 'dist already exists. Move it aside before creating a new package.' }
New-Item -ItemType Directory -Path $destination | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build\bin\Release\geosunpre.exe') -Destination $destination
Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'build\bin\Release') -Filter '*.dll' | Copy-Item -Destination $destination
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination $destination
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'docs') -Destination $destination -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'dependencies\licenses') -Destination $destination -Recurse
$modelOutput = Join-Path $destination 'models'
New-Item -ItemType Directory -Path $modelOutput | Out-Null
Copy-Item -LiteralPath (Join-Path $Models 'manifest.json') -Destination $modelOutput
Copy-Item -LiteralPath (Join-Path $Models 'engines'),(Join-Path $Models 'runtime') -Destination $modelOutput -Recurse
& (Join-Path $destination 'geosunpre.exe') info
if ($LASTEXITCODE) { throw 'Packaged CUDA runtime did not start.' }
Write-Host "Package: $destination"
