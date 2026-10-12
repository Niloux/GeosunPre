[CmdletBinding()]
param([int]$Jobs = 8)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $headers = Join-Path $PSScriptRoot 'build\_deps\tensorrt-headers-10.13\include'
    New-Item -ItemType Directory -Force -Path $headers | Out-Null
    foreach ($header in @('NvInferRuntime.h','NvInferRuntimeBase.h','NvInferRuntimeCommon.h','NvInferImpl.h','NvInferPluginBase.h','NvInferVersion.h','NvInferLegacyDims.h','NvInferRuntimePlugin.h')) {
        $path = Join-Path $headers $header
        if (-not (Test-Path -LiteralPath $path)) {
            $temporary = $path + '.pending'
            Invoke-WebRequest ('https://raw.githubusercontent.com/NVIDIA/TensorRT/b8db91e15be2cae4465ac17fab19e0f969e45407/include/' + $header) -OutFile $temporary -TimeoutSec 60
            Move-Item -LiteralPath $temporary -Destination $path
        }
    }
    & cmake --preset windows
    if ($LASTEXITCODE) { throw 'CMake configuration failed.' }
    & cmake --build --preset release --target geosunpre --parallel $Jobs
    if ($LASTEXITCODE) { throw 'Native compilation failed.' }
    if (-not $env:CUDA_PATH) { throw 'CUDA_PATH must point to the CUDA Toolkit.' }
    $runtime = Join-Path $env:CUDA_PATH 'bin\cudart64_12.dll'
    Copy-Item -LiteralPath $runtime -Destination (Join-Path $PSScriptRoot 'build\bin\Release')
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $crt = Get-ChildItem -LiteralPath (Join-Path $vs 'VC\Redist\MSVC') -Directory | Where-Object Name -Match '^\d' | Sort-Object Name -Descending | Select-Object -First 1
    Get-ChildItem -LiteralPath (Join-Path $crt.FullName 'x64\Microsoft.VC143.CRT') -Filter '*.dll' | Copy-Item -Destination (Join-Path $PSScriptRoot 'build\bin\Release')
    Write-Host 'Built: build\bin\Release\geosunpre.exe'
} finally { Pop-Location }
