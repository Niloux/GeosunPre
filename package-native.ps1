# Package the native runtime and converted model weights without Python/WSL.
[CmdletBinding()]
param([string]$Destination = (Join-Path $PSScriptRoot '.cache\native\runtime'))
$ErrorActionPreference = 'Stop'
$build = Join-Path $PSScriptRoot '.cache\native\build\bin'
$models = Join-Path $PSScriptRoot '.cache\native\models'
if (-not (Test-Path -LiteralPath (Join-Path $build 'geosunpre-native.exe'))) { throw 'Build first: .\setup-native.ps1' }
if (-not (Test-Path -LiteralPath (Join-Path $models 'manifest.json'))) { throw 'Convert the fixed weights first: scripts\export_native_models.py' }
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination,(Join-Path $Destination 'inference'),(Join-Path $Destination 'models') | Out-Null
foreach ($folder in @('', 'inference')) {
    $source = Join-Path $build $folder
    $target = Join-Path $Destination $folder
    Get-ChildItem -LiteralPath $source -File | Where-Object { $_.Extension -in @('.exe','.dll') -and $_.Name -ne 'torch_python.dll' } | Copy-Item -Destination $target -Force
}
foreach ($name in @('manifest.json','detector.pt','descriptor.pt','matcher.pt','segmentation.pt')) {
    Copy-Item -LiteralPath (Join-Path $models $name) -Destination (Join-Path $Destination 'models') -Force
}
# Share the CUDA DLLs shipped with the matching LibTorch SDK with the numerical CLI.
Get-ChildItem -LiteralPath (Join-Path $Destination 'inference') -File | Where-Object { $_.Name -match '^(cublas|cusolver|cusparse|cudart|nvJitLink).*\.dll$' } | Copy-Item -Destination $Destination -Force
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$crt = Get-ChildItem -LiteralPath (Join-Path $visualStudio 'VC\Redist\MSVC') -Directory | Where-Object { $_.Name -match '^\d+\.' } | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
$crtFolder = Join-Path $crt.FullName 'x64\Microsoft.VC143.CRT'
foreach ($target in @($Destination,(Join-Path $Destination 'inference'))) {
    Get-ChildItem -LiteralPath $crtFolder -Filter '*.dll' -File | Copy-Item -Destination $target -Force
}
& (Join-Path $Destination 'geosunpre-native.exe') info
if ($LASTEXITCODE -ne 0) { throw 'Packaged numerical runtime failed to start.' }
& (Join-Path $Destination 'inference\geosunpre-inference.exe') info
if ($LASTEXITCODE -ne 0) { throw 'Packaged CUDA runtime failed to start.' }
Write-Host "Native runtime: $Destination"
