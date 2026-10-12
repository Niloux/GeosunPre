# Build the standalone C++ solver using project-local dependency/output caches.
[CmdletBinding()]
param(
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [int]$Jobs = 8,
    [switch]$SkipDependencies
)
$ErrorActionPreference = 'Stop'
function Invoke-NativeBuild([scriptblock]$Command, [string]$Failure) {
    # Windows PowerShell treats native stderr warnings as error records when
    # redirected. Keep the diagnostic output and decide from the exit code.
    $ErrorActionPreference = 'Continue'
    & $Command
    if ($LASTEXITCODE -ne 0) { throw $Failure }
}
Push-Location $PSScriptRoot
try {
    $cache = Join-Path $PSScriptRoot '.cache\native'
    New-Item -ItemType Directory -Force -Path $cache,(Join-Path $cache 'downloads'),(Join-Path $cache 'src') | Out-Null
    if (-not $VcpkgRoot) {
        $candidate = Join-Path (Split-Path $PSScriptRoot -Parent) 'vcpkg'
        if (Test-Path (Join-Path $candidate 'vcpkg.exe')) { $VcpkgRoot = $candidate }
        else { throw 'Pass -VcpkgRoot or set VCPKG_ROOT to an existing vcpkg checkout.' }
    }
    $VcpkgRoot = (Resolve-Path -LiteralPath $VcpkgRoot).Path
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $visualStudio) { throw 'Install the Visual Studio C++ desktop workload and Windows SDK.' }
    & (Join-Path $visualStudio 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) { throw 'CUDA Toolkit 12.8 is required for the Ceres CUDA build.' }
    $archive = Join-Path $cache 'downloads\colmap-4.2.0.tar.gz'
    $source = Join-Path $cache 'src\colmap-4.2.0'
    if (-not (Test-Path (Join-Path $source 'CMakeLists.txt'))) {
        if (-not (Test-Path $archive)) {
            Invoke-WebRequest 'https://codeload.github.com/colmap/colmap/tar.gz/refs/tags/4.2.0' -OutFile $archive
        }
        $expected = 'FD332A93A225B2244656BE786D83A0EA9B9F55275EEE8600A29F9696A233B7A12612E10489E8DD3DBD250EEF95CBB21D93B8555DDD25A88238141CC4595987C4'
        if ((Get-FileHash $archive -Algorithm SHA512).Hash -ne $expected) { throw 'COLMAP source archive differs from the pinned 4.2.0 source.' }
        Invoke-NativeBuild { & tar -xf $archive -C (Join-Path $cache 'src') } 'COLMAP source extraction failed.'
    }
    $env:VCPKG_MAX_CONCURRENCY = "$Jobs"
    $manifest = Join-Path $PSScriptRoot 'native\cpp'
    $installed = Join-Path $cache 'installed'
    if (-not $SkipDependencies) {
        Invoke-NativeBuild { & (Join-Path $VcpkgRoot 'vcpkg.exe') install --triplet=x64-windows "--x-manifest-root=$manifest" "--x-install-root=$installed" "--x-buildtrees-root=$(Join-Path $cache 'buildtrees')" "--x-packages-root=$(Join-Path $cache 'packages')" "--downloads-root=$(Join-Path $cache 'downloads')" "--overlay-ports=$(Join-Path $manifest 'ports')" "--binarysource=clear;files,$(Join-Path $cache 'binary-cache'),readwrite" } 'Native dependency build failed.'
    }
    $build = Join-Path $cache 'build'
    Invoke-NativeBuild { & cmake -S $manifest -B $build -G 'Visual Studio 17 2022' -A x64 "-DCMAKE_TOOLCHAIN_FILE=$(Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake')" '-DVCPKG_MANIFEST_INSTALL=OFF' "-DVCPKG_INSTALLED_DIR=$installed" "-DGEOSUNPRE_COLMAP_SOURCE=$source" '-DCMAKE_BUILD_TYPE=Release' } 'Native CMake configuration failed.'
    Invoke-NativeBuild { & cmake --build $build --config Release --target geosunpre-inference geosunpre-native --parallel $Jobs } 'Native compilation failed.'
    $program = Join-Path $build 'bin\geosunpre-native.exe'
    Invoke-NativeBuild { & $program info } 'Native runtime dependency check failed.'
    Write-Host "Native solver: $program"
} finally {
    Pop-Location
}
