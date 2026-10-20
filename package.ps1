[CmdletBinding()]
param(
    [string]$Models = (Join-Path $PSScriptRoot 'models-ampere-plus'),
    [string]$Output,
    [string]$Image
)
$ErrorActionPreference = 'Stop'
$program = Join-Path $PSScriptRoot 'build\ninja\bin\geosunpre.exe'
if (-not (Test-Path -LiteralPath $program)) { throw 'Run build.ps1 first.' }
$Models = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Models).TrimEnd([char[]]'\/')
$manifestPath = Join-Path $Models 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Model package has no manifest: $Models" }
$manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifestPath | ConvertFrom-Json
$hardware = if ($manifest.hardware_compatibility) { $manifest.hardware_compatibility } else { 'none' }
$info = (& $program info) -join "`n" | ConvertFrom-Json
if ($LASTEXITCODE) { throw 'geosunpre.exe info failed.' }
# A delivery must be reproducible from a commit.
if ($info.revision -like '*-dirty' -or $info.revision -eq 'unknown') {
    throw "Executable was built from uncommitted sources ($($info.revision)). Commit, run build.ps1, then package."
}
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
    pipeline_version = $info.pipeline_version
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

$archive = $destination + '.zip'
if (Test-Path -LiteralPath $archive) { throw "Archive already exists: $archive" }
& "$env:SystemRoot\System32\tar.exe" --format zip -c -f ($archive + '.pending') -C (Split-Path $destination) (Split-Path $destination -Leaf)
if ($LASTEXITCODE) { throw "Cannot create archive for $destination" }
Move-Item -LiteralPath ($archive + '.pending') -Destination $archive
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($archive + '.sha256', "$digest  $(Split-Path $archive -Leaf)`n", [Text.UTF8Encoding]::new($false))
Write-Host "Package: $destination"
Write-Host "Archive: $archive"
