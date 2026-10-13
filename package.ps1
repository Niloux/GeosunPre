[CmdletBinding()]
param(
    [string]$Models = (Join-Path $PSScriptRoot 'models'),
    [string]$Output = (Join-Path $PSScriptRoot 'dist'),
    [string]$Image
)
$ErrorActionPreference = 'Stop'
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output).TrimEnd([char[]]'\/')
if (Test-Path -LiteralPath $destination) { throw 'Output already exists. Choose a new -Output directory.' }
if (-not $Image) {
    $Image = Join-Path $PSScriptRoot '.cache\model-prepare\source\LoMa-8fb59c458f2a2ef44314f462467b229c27786bf0\assets\0015_A.jpg'
}
if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) { throw 'Specify -Image with a real photograph for model verification.' }
$pending = $destination + '.pending'
if (Test-Path -LiteralPath $pending) { throw 'Unfinished package exists. Choose a new -Output directory or move it aside.' }
New-Item -ItemType Directory -Path $pending | Out-Null
$published = $destination
$destination = $pending
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build\bin\Release\geosunpre.exe') -Destination $destination
Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'build\bin\Release') -Filter '*.dll' | Copy-Item -Destination $destination
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'docs\deployment.md') -Destination (Join-Path $destination 'README.md')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'docs') -Destination $destination -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'dependencies\licenses') -Destination $destination -Recurse
$modelOutput = Join-Path $destination 'models'
New-Item -ItemType Directory -Path $modelOutput | Out-Null
Copy-Item -LiteralPath (Join-Path $Models 'manifest.json') -Destination $modelOutput
Copy-Item -LiteralPath (Join-Path $Models 'engines'),(Join-Path $Models 'runtime') -Destination $modelOutput -Recurse
& (Join-Path $destination 'geosunpre.exe') check-models --models $modelOutput --image $Image
if ($LASTEXITCODE) { throw "Packaged inference failed. Files remain in $pending" }
[IO.Directory]::Move($pending, $published)
Write-Host "Package: $published"
