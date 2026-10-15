[CmdletBinding()]
param([ValidateRange(1, 2147483647)][int]$Jobs = 8, [switch]$Configure)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install the Visual Studio 2022 C++ desktop tools.' }
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -or -not $vs) { throw 'Cannot find the Visual Studio C++ tools.' }
    Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
    # CMake's /showIncludes probe and Ninja must use the same output encoding.
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    & "$env:SystemRoot\System32\chcp.com" 65001 | Out-Null
    if ($LASTEXITCODE) { throw 'Cannot select UTF-8 console output for the build.' }
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
        $ninjaRoot = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
        if (-not (Test-Path -LiteralPath (Join-Path $ninjaRoot 'ninja.exe'))) {
            throw 'Install Ninja or the Visual Studio C++ CMake tools.'
        }
        $env:PATH = $ninjaRoot + ';' + $env:PATH
    }
    $buildRoot = Join-Path $PSScriptRoot 'build\ninja'
    $bin = Join-Path $buildRoot 'bin'
    $headers = Join-Path $buildRoot '_deps\tensorrt-headers-10.13\include'
    New-Item -ItemType Directory -Force -Path $headers | Out-Null
    foreach ($header in @('NvInferRuntime.h','NvInferRuntimeBase.h','NvInferRuntimeCommon.h','NvInferImpl.h','NvInferPluginBase.h','NvInferVersion.h','NvInferLegacyDims.h','NvInferRuntimePlugin.h')) {
        $path = Join-Path $headers $header
        if (-not (Test-Path -LiteralPath $path)) {
            $temporary = $path + '.pending'
            Invoke-WebRequest ('https://raw.githubusercontent.com/NVIDIA/TensorRT/b8db91e15be2cae4465ac17fab19e0f969e45407/include/' + $header) -OutFile $temporary -TimeoutSec 60
            Move-Item -LiteralPath $temporary -Destination $path
        }
    }
    $cache = Join-Path $buildRoot 'CMakeCache.txt'
    $needsConfigure = $Configure -or -not (Test-Path -LiteralPath $cache) -or
        -not (Test-Path -LiteralPath (Join-Path $buildRoot 'build.ninja'))
    if (-not $needsConfigure) {
        $needsConfigure = (Get-Item -LiteralPath (Join-Path $PSScriptRoot 'CMakePresets.json')).LastWriteTimeUtc -gt
            (Get-Item -LiteralPath $cache).LastWriteTimeUtc
    }
    if ($needsConfigure) {
        & cmake --preset windows
        if ($LASTEXITCODE) { throw 'CMake configuration failed.' }
    }
    & cmake --build --preset release --target geosunpre --parallel $Jobs
    if ($LASTEXITCODE) { throw 'Native compilation failed.' }
    if (-not $env:CUDA_PATH) { throw 'CUDA_PATH must point to the CUDA Toolkit.' }
    $runtime = Join-Path $env:CUDA_PATH 'bin\cudart64_12.dll'
    Copy-Item -LiteralPath $runtime -Destination $bin
    $crt = Get-ChildItem -LiteralPath (Join-Path $vs 'VC\Redist\MSVC') -Directory | Where-Object Name -Match '^\d' | Sort-Object Name -Descending | Select-Object -First 1
    Get-ChildItem -LiteralPath (Join-Path $crt.FullName 'x64\Microsoft.VC143.CRT') -Filter '*.dll' | Copy-Item -Destination $bin
    Write-Host "Built: $bin\geosunpre.exe"
} finally { Pop-Location }
