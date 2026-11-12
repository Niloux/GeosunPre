[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Preprocess,
    [Parameter(Mandatory)] [string]$Trainer,
    [string]$Output,
    [string]$Image
)
# Combine a GeosunPre package and its paired trainer package into one delivery.
# Both packages are copied unchanged; the entry script and README come from the
# GeosunPre commit the preprocess package was built from.
$ErrorActionPreference = 'Stop'

function Resolve-Dir([string]$Path) {
    $full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path).TrimEnd([char[]]'\/')
    if (-not (Test-Path -LiteralPath $full -PathType Container)) { throw "Not a package directory: $full" }
    $full
}
function Get-Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Get-Files([string]$Root) {
    $files = [ordered]@{}
    Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName | ForEach-Object {
        $files[$_.FullName.Substring($Root.Length + 1).Replace('\', '/')] = Get-Sha256 $_.FullName
    }
    $files
}
# Every file listed with its recorded hash, and nothing besides package.json.
function Test-Package([string]$Root, $Record, [string]$Name) {
    $expected = @($Record.files.PSObject.Properties.Name | Sort-Object)
    $actual = Get-Files $Root
    $present = @($actual.Keys | Where-Object { $_ -ne 'package.json' } | Sort-Object)
    if (($expected -join "`n") -ne ($present -join "`n")) { throw "$Name file list differs from its package.json: $Root" }
    $bad = @($expected | Where-Object { $actual[$_] -ne $Record.files.$_ })
    if ($bad.Count) { throw "$Name files differ from package.json: $($bad -join ', ')" }
}
function Read-Record([string]$Root) {
    $path = Join-Path $Root 'package.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "No package.json in $Root" }
    Get-Content -Raw -Encoding UTF8 -LiteralPath $path | ConvertFrom-Json
}

$Preprocess = Resolve-Dir $Preprocess
$Trainer = Resolve-Dir $Trainer
$pre = Read-Record $Preprocess
$train = Read-Record $Trainer
if (-not (Test-Path -LiteralPath (Join-Path $Preprocess 'geosunpre.exe') -PathType Leaf)) { throw "Not a GeosunPre package: $Preprocess" }
if (-not (Test-Path -LiteralPath (Join-Path $Trainer 'bin/geosun_3dgs_train.exe') -PathType Leaf)) { throw "Not a trainer package: $Trainer" }

# Pairing: both from commits, the trainer packaged against this GeosunPre revision.
if ($pre.revision -notmatch '^[0-9a-f]{7,40}$') { throw "GeosunPre package is not from a commit: $($pre.revision)" }
if ($train.revision -notmatch '^[0-9a-f]{7,40}$') { throw "Trainer package is not from a commit: $($train.revision)" }
if ($null -ne $train.dirty) { throw 'Trainer package was made by a packager that allowed uncommitted sources. Repackage it.' }
$pairRev = [string]$train.geosunpre_revision
$n = [Math]::Min($pairRev.Length, $pre.revision.Length)
if ($n -lt 7 -or $pairRev.Substring(0, $n) -ne $pre.revision.Substring(0, $n)) {
    throw "Trainer is paired with GeosunPre $pairRev, not $($pre.revision)."
}
Test-Package $Preprocess $pre 'GeosunPre'
Test-Package $Trainer $train 'Trainer'

# The training config, entry script and README are versioned with GeosunPre.
# cmd redirection keeps the blob bytes; PowerShell would re-decode them.
function Save-Committed([string]$Path, [string]$Destination) {
    & cmd /c "git -C `"$PSScriptRoot`" show `"$($pre.revision):$Path`" > `"$Destination`""
    if ($LASTEXITCODE) { throw "GeosunPre commit $($pre.revision) has no $Path." }
}
$scratch = Join-Path ([IO.Path]::GetTempPath()) "geosun-bundle-$PID.json"
Save-Committed 'configs/trainer.json' $scratch
$committedConfig = Get-Content -Raw -Encoding UTF8 -LiteralPath $scratch | ConvertFrom-Json | ConvertTo-Json -Depth 32 -Compress
Remove-Item -LiteralPath $scratch
$packagedConfig = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $Trainer 'config/trainer.json') |
    ConvertFrom-Json | ConvertTo-Json -Depth 32 -Compress
if ($committedConfig -ne $packagedConfig) {
    throw "Trainer config/trainer.json differs from configs/trainer.json at GeosunPre $($pre.revision)."
}

if (-not $Output) {
    $Output = Join-Path $PSScriptRoot "packages\Geosun-$($pre.version)-$($pre.revision)-$($pre.models.hardware_compatibility)-trainer-$($train.revision)"
}
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output).TrimEnd([char[]]'\/')
$pending = "$destination.pending"
$archive = "$destination.zip"
foreach ($path in @($destination, $pending, $archive, "$archive.pending", "$archive.sha256")) {
    if (Test-Path -LiteralPath $path) { throw "Choose a new -Output. Already exists: $path" }
}
if (-not $Image) {
    $Image = @((Join-Path $PSScriptRoot 'packages\onnx-bundle-v1\images\A.jpg'),
        (Join-Path $PSScriptRoot '.cache\model-prepare\source\LoMa-8fb59c458f2a2ef44314f462467b229c27786bf0\assets\0015_A.jpg')) |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $Image -or -not (Test-Path -LiteralPath $Image -PathType Leaf)) { throw 'Specify -Image with a real photograph for model verification.' }

New-Item -ItemType Directory -Path $pending | Out-Null
Copy-Item -LiteralPath $Preprocess -Destination (Join-Path $pending 'preprocess') -Recurse
Copy-Item -LiteralPath $Trainer -Destination (Join-Path $pending 'trainer') -Recurse
$utf8 = [Text.UTF8Encoding]::new($false)
# PowerShell 5.1 reads BOM-less scripts as ANSI; the entry script stays ASCII.
Save-Committed 'bundle/geosun.ps1' (Join-Path $pending 'geosun.ps1')
Save-Committed 'bundle/README.md' (Join-Path $pending 'README.md')
$record = [ordered]@{
    preprocess = [ordered]@{
        version = $pre.version; revision = $pre.revision
        hardware_compatibility = $pre.models.hardware_compatibility
        package_json_sha256 = Get-Sha256 (Join-Path $pending 'preprocess/package.json')
    }
    trainer = [ordered]@{
        name = $train.name; version = $train.version; revision = $train.revision
        cuda_architectures = $train.cuda_architectures
        package_json_sha256 = Get-Sha256 (Join-Path $pending 'trainer/package.json')
    }
    files = [ordered]@{
        'geosun.ps1' = Get-Sha256 (Join-Path $pending 'geosun.ps1')
        'README.md' = Get-Sha256 (Join-Path $pending 'README.md')
    }
}
[IO.File]::WriteAllText((Join-Path $pending 'bundle.json'), ($record | ConvertTo-Json -Depth 4), $utf8)

# Read back the copies and run both programs from the bundle.
Test-Package (Join-Path $pending 'preprocess') $pre 'Bundled GeosunPre'
Test-Package (Join-Path $pending 'trainer') $train 'Bundled trainer'
$before = Get-Files $pending
& (Join-Path $pending 'preprocess\geosunpre.exe') check-models --image $Image
if ($LASTEXITCODE) { throw "Bundled inference failed. Files remain in $pending" }
$version = & (Join-Path $pending 'trainer\bin\geosun_3dgs_train.exe') --version
if ($LASTEXITCODE) { throw "Bundled trainer does not start. Files remain in $pending" }
Write-Host ($version -join "`n")
$after = Get-Files $pending
if (($after.Keys -join "`n") -ne ($before.Keys -join "`n") -or
    @($before.Keys | Where-Object { $before[$_] -ne $after[$_] }).Count) {
    throw "Bundle files changed during verification. Files remain in $pending"
}
[IO.Directory]::Move($pending, $destination)

& "$env:SystemRoot\System32\tar.exe" --format zip -c -f "$archive.pending" -C (Split-Path $destination) (Split-Path $destination -Leaf)
if ($LASTEXITCODE) { throw "Cannot create archive for $destination" }
Move-Item -LiteralPath "$archive.pending" -Destination $archive
[IO.File]::WriteAllText("$archive.sha256", "$(Get-Sha256 $archive)  $(Split-Path $archive -Leaf)`n", $utf8)
Write-Host "Bundle: $destination"
Write-Host "Archive: $archive"
