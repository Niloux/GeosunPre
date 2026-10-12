# Prepare the locked Windows environment without changing the system Python.
[CmdletBinding()]
param(
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot
try {
    if (-not $CheckOnly) {
        $env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.cache\python'
        $env:PYTHONUTF8 = '1'
        $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
        $uvPath = if ($uvCommand) { $uvCommand.Source } else { $null }
        # An app launched before uv was installed can have an old process PATH.
        if (-not $uvPath) {
            foreach ($directory in ([Environment]::GetEnvironmentVariable('Path', 'User') -split ';')) {
                if (-not $directory) { continue }
                $candidate = Join-Path $directory 'uv.exe'
                if (Test-Path -LiteralPath $candidate) { $uvPath = $candidate; break }
            }
        }
        if (-not $uvPath) { throw 'uv was not found. Install it with: winget install --id astral-sh.uv -e' }

        if (-not (Get-Command cl -CommandType Application -ErrorAction SilentlyContinue)) {
            $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
            if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio C++ Build Tools with the Windows SDK.' }
            $visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if (-not $visualStudio) { throw 'Visual Studio is missing the Desktop development with C++ workload.' }
            & (Join-Path $visualStudio 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
        }
        & $uvPath sync --locked --python 3.13
        if ($LASTEXITCODE -ne 0) { throw 'uv sync failed; see the dependency or compiler error above.' }
        & (Join-Path $PSScriptRoot 'setup-native.ps1')
    }
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'The Windows virtual environment is missing. Run .\setup-windows.ps1 first.' }
    & $python -X utf8 scripts/check_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed; see the error above.' }
} finally {
    Pop-Location
}
