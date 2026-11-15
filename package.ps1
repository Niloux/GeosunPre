[CmdletBinding()]
param(
    [string]$Models,
    [string]$Output,
    [string]$Image,
    [string]$Preprocess,
    [string]$Trainer
)
$ErrorActionPreference = 'Stop'
$tools = Join-Path $PSScriptRoot 'tools\windows'
if ($Preprocess -or $Trainer) {
    if (-not ($Preprocess -and $Trainer) -or $Models) {
        throw 'Bundle packaging requires both -Preprocess and -Trainer, without -Models.'
    }
    & (Join-Path $tools 'build.ps1') | Out-Host
    & (Join-Path $tools 'package-bundle.ps1') -Preprocess $Preprocess -Trainer $Trainer -Output $Output -Image $Image
    return
}
& git -C $PSScriptRoot diff --quiet HEAD --
if ($LASTEXITCODE -eq 1) { throw 'Commit your changes and rerun package.ps1.' }
if ($LASTEXITCODE) { throw 'Cannot verify the source revision for delivery.' }
& (Join-Path $tools 'build.ps1') | Out-Host
$program = Join-Path $PSScriptRoot 'build\ninja\bin\geosunpre.exe'
$infoJson = & $program info
if ($LASTEXITCODE) { throw 'geosunpre.exe info failed.' }
$info = ($infoJson -join "`n") | ConvertFrom-Json
# A delivery must be reproducible from a commit.
if ($info.revision -like '*-dirty' -or $info.revision -eq 'unknown') {
    throw "Executable was built from uncommitted sources ($($info.revision)). Commit your changes and rerun package.ps1."
}
if (-not $Models) {
    $localModels = Join-Path $PSScriptRoot 'models'
    $localManifest = Join-Path $localModels 'manifest.json'
    if (-not (Test-Path -LiteralPath $localManifest -PathType Leaf)) {
        throw 'Run .\run.ps1 prepare-models [path-to-sam3.pt] first so delivery can reuse the ONNX graphs.'
    }
    # Keep a separate delivery model directory for each set of local models.
    $key = (Get-FileHash -LiteralPath $localManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    $Models = Join-Path $PSScriptRoot ".cache\delivery-models\$key"
    if (-not (Test-Path -LiteralPath (Join-Path $Models 'manifest.json') -PathType Leaf)) {
        & (Join-Path $tools 'prepare-models.ps1') -Models $Models -OnnxFrom $localModels -HardwareCompatibility ampere_plus
    }
}
$Models = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Models).TrimEnd([char[]]'\/')
$manifestPath = Join-Path $Models 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Model package has no manifest: $Models" }
$manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifestPath | ConvertFrom-Json
$hardware = $manifest.hardware_compatibility
if ($hardware -ne 'ampere_plus') { throw "Delivery requires ampere_plus models: $Models" }
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot "packages\GeosunPre-$($info.version)-$($info.revision)-$hardware"
}
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output).TrimEnd([char[]]'\/')
if (Test-Path -LiteralPath $destination) { throw 'Output already exists. Choose a new -Output directory.' }
if (-not $Image) {
    $Image = @((Join-Path $PSScriptRoot 'packages\onnx-bundle-v1\images\A.jpg'),
        (Join-Path $PSScriptRoot '.cache\model-prepare\source\LoMa-8fb59c458f2a2ef44314f462467b229c27786bf0\assets\0015_A.jpg')) |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $Image -or -not (Test-Path -LiteralPath $Image -PathType Leaf)) { throw 'Specify -Image with a real photograph for model verification.' }
$pending = $destination + '.pending'
if (Test-Path -LiteralPath $pending) { throw 'Unfinished package exists. Choose a new -Output directory or move it aside.' }
New-Item -ItemType Directory -Path $pending | Out-Null

Copy-Item -LiteralPath $program -Destination $pending
Get-ChildItem -LiteralPath (Split-Path $program) -Filter '*.dll' | Copy-Item -Destination $pending
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'docs\deployment.md') -Destination (Join-Path $pending 'README.md')
# Only user-facing contracts; model preparation is a development task.
New-Item -ItemType Directory -Path (Join-Path $pending 'docs') | Out-Null
foreach ($name in @('input.md', 'output.md', 'cache.md')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "docs\$name") -Destination (Join-Path $pending 'docs')
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'dependencies\licenses') -Destination $pending -Recurse
$modelOutput = Join-Path $pending 'models'
New-Item -ItemType Directory -Path $modelOutput | Out-Null
Copy-Item -LiteralPath $manifestPath -Destination $modelOutput
Copy-Item -LiteralPath (Join-Path $Models 'engines'),(Join-Path $Models 'runtime') -Destination $modelOutput -Recurse

function Get-PackageFiles([string]$Root) {
    $files = [ordered]@{}
    Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object Name -ne 'package.json' |
        Sort-Object FullName | ForEach-Object {
            $files[$_.FullName.Substring($Root.Length + 1).Replace('\', '/')] =
                (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    $files
}
$files = Get-PackageFiles $pending
$record = [ordered]@{
    version = $info.version; revision = $info.revision; built_utc = $info.built_utc
    models = [ordered]@{ hardware_compatibility = $hardware; build_gpu = $manifest.gpu; tensorrt = $manifest.tensorrt }
    files = $files
}
[IO.File]::WriteAllText((Join-Path $pending 'package.json'), ($record | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))

& (Join-Path $pending 'geosunpre.exe') check-models --models $modelOutput --image $Image
if ($LASTEXITCODE) { throw "Packaged inference failed. Files remain in $pending" }
# Read back: the check must not have changed any delivered file.
$after = Get-PackageFiles $pending
if (($after.Keys -join "`n") -ne ($files.Keys -join "`n") -or
    @($files.Keys | Where-Object { $files[$_] -ne $after[$_] }).Count) {
    throw "Package files changed during verification. Files remain in $pending"
}
[IO.Directory]::Move($pending, $destination)

Write-Host "Package: $destination"
