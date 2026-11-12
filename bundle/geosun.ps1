[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Output,
    [string]$Scene,
    [ValidateSet('all', 'preprocess', 'train', 'export')] [string]$Step = 'all',
    [string]$BundleRoot = $PSScriptRoot,
    [ValidateScript({ $_ -gt 0 })] [double]$TexelSize = 0.02,
    [string[]]$PreprocessArgs = @(),
    [string[]]$TrainArgs = @()
)
# Preprocess a scene with GeosunPre, then train it with the Geosun 3DGS Trainer.
# OUT\delivery is the GeosunPre delivery, OUT\train the training result.
$ErrorActionPreference = 'Stop'
$root = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BundleRoot)
$geosunpre = Join-Path $root 'preprocess\geosunpre.exe'
$trainer = Join-Path $root 'trainer\bin\geosun_3dgs_train.exe'
$config = Join-Path $root 'trainer\config\trainer.json'
$out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output).TrimEnd([char[]]'\/')
$delivery = Join-Path $out 'delivery'
$train = Join-Path $out 'train'
$export = Join-Path $out 'export'
$exportPending = "$export.pending"

if ($Step -in @('all', 'export')) {
    foreach ($path in @($export, $exportPending)) {
        if (Test-Path -LiteralPath $path) { throw "Export path already exists: $path" }
    }
}

function Invoke-Checked([string]$Program, [string[]]$Arguments, [string]$What) {
    Write-Host "> $What"
    & $Program @Arguments
    if ($LASTEXITCODE) { throw "$What failed (exit code $LASTEXITCODE)." }
}

if ($Step -in @('all', 'preprocess')) {
    # GeosunPre resumes an interrupted run from OUT\delivery.work with the same command;
    # a finished delivery without a work directory is only read back.
    if ((Test-Path -LiteralPath $delivery) -and -not (Test-Path -LiteralPath "$delivery.work")) {
        Write-Host "Delivery already exists: $delivery"
    } else {
        if (-not $Scene) { throw 'Specify -Scene with the scene directory.' }
        Invoke-Checked $geosunpre (@('run', '--input', $Scene, '--output', $delivery) + $PreprocessArgs) 'Preprocess'
    }
    Invoke-Checked $geosunpre @('check', '--output', $delivery) 'Delivery check'
}

if ($Step -in @('all', 'train')) {
    if (-not (Test-Path -LiteralPath (Join-Path $delivery 'result.json') -PathType Leaf)) {
        throw "No GeosunPre delivery in $delivery. Run the preprocess step first."
    }
    if (Test-Path -LiteralPath $train) { throw "Training output already exists: $train. Choose a new -Output." }
    $pending = "$train.pending"
    if (Test-Path -LiteralPath $pending) {
        throw "Unfinished training exists: $pending. Move it aside, then rerun."
    }
    if ($root -match '[^\x00-\x7F]') {
        Write-Warning "Bundle path is not ASCII ($root); the trainer falls back to CPU JPEG decoding."
    }
    New-Item -ItemType Directory -Path $pending | Out-Null
    Invoke-Checked $trainer (@('--config', $config, '-d', $delivery, '-o', $pending) + $TrainArgs) 'Training'
    [IO.Directory]::Move($pending, $train)
    Write-Host "Training: $train"
}

if ($Step -in @('all', 'export')) {
    $project = Join-Path $train 'project.licht'
    if (-not (Test-Path -LiteralPath $project -PathType Leaf)) { throw "Missing training project: $project" }
    New-Item -ItemType Directory -Path $exportPending | Out-Null
    $splats = Join-Path $exportPending 'splats.ply'
    $mesh = Join-Path $exportPending 'mesh.ply'
    $texel = $TexelSize.ToString([Globalization.CultureInfo]::InvariantCulture)
    Invoke-Checked $trainer @('mesh', $project, '-d', $delivery, '-o', $mesh, '--texel-size', $texel, '--export', 'ply', '--output-name', 'splats') 'Export 3DGS PLY, mesh and OSGB'
    foreach ($path in @($splats, $mesh)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing export: $path" }
    }
    if (-not (Get-ChildItem -LiteralPath (Join-Path $exportPending 'mesh_osgb') -Filter '*.osgb' -Recurse -File | Select-Object -First 1)) {
        throw "No OSGB files in $exportPending"
    }
    [IO.Directory]::Move($exportPending, $export)
    Write-Host "Export: $export"
}
