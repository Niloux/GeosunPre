[CmdletBinding()]
param(
    [string]$Models = (Join-Path $PSScriptRoot 'models'),
    [ValidateRange(1, 2147483647)][int]$ImageWidth = 3000,
    [ValidateRange(1, 2147483647)][int]$ImageHeight = 4000,
    [string]$OnnxFrom,
    [string[]]$Images
)
$ErrorActionPreference = 'Stop'
if ($Images -and $Images.Count -ne 2) { throw '-Images requires two real image paths.' }

function Invoke-Tool([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE) { throw "Model preparation failed: $Program (exit $LASTEXITCODE)" }
}

Push-Location $PSScriptRoot
$environmentNames = @('UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'UV_PYTHON_BIN_DIR', 'PYTHONUTF8')
$previousEnvironment = @{}
foreach ($name in $environmentNames) { $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
    $cache = Join-Path $PSScriptRoot '.cache\model-prepare'
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    $env:UV_CACHE_DIR = Join-Path $cache 'uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $cache 'python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $cache 'bin'
    $env:PYTHONUTF8 = '1'

    $uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if ($uv) { $uvPath = $uv.Source }
    else {
        $uvPath = Join-Path $cache 'uv\uv.exe'
        if (-not (Test-Path -LiteralPath $uvPath)) {
            $archive = Join-Path $cache 'uv.zip'
            Write-Host 'Downloading the project-local model preparation tool...'
            Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/download/0.12.9/uv-x86_64-pc-windows-msvc.zip' -OutFile ($archive + '.pending')
            Move-Item -LiteralPath ($archive + '.pending') -Destination $archive -Force
            Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $cache 'uv') -Force
        }
    }
    $python = Join-Path $cache 'venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        Invoke-Tool $uvPath @('python', 'install', '3.11.13', '--no-bin')
        Invoke-Tool $uvPath @('venv', (Join-Path $cache 'venv'), '--python', '3.11.13', '--managed-python')
    }
    $requirements = Join-Path $PSScriptRoot 'tools\models\requirements.txt'
    $stamp = Join-Path $cache 'environment.sha256'
    $identity = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
    if (-not (Test-Path -LiteralPath $stamp) -or (Get-Content -Raw -LiteralPath $stamp).Trim() -ne $identity) {
        Write-Host 'Installing the isolated model export/build environment (first run only)...'
        Invoke-Tool $uvPath @('pip', 'install', '--python', $python, 'torch==2.8.0', 'torchvision==0.23.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
        Invoke-Tool $uvPath @('pip', 'install', '--python', $python, '-r', $requirements, '--extra-index-url', 'https://pypi.nvidia.com')
        Set-Content -LiteralPath $stamp -Value $identity -Encoding ASCII
    }
    $modelRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Models).TrimEnd([char[]]'\/')
    $arguments = @('-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'tools\models\prepare.py'), '--models', $modelRoot,
        '--image-width', "$ImageWidth", '--image-height', "$ImageHeight")
    if ($OnnxFrom) { $arguments += @('--onnx-from', $OnnxFrom) }
    if ($Images) {
        $arguments += @('--images') + $Images
    }
    Invoke-Tool $python $arguments
    if (-not (Test-Path -LiteralPath $modelRoot)) {
        $pending = $modelRoot + '.pending'
        try {
            [IO.Directory]::Move($pending, $modelRoot)
        } catch [IO.IOException], [UnauthorizedAccessException] {
            throw "Cannot publish models: $($_.Exception.Message) Verified files remain in '$pending'. Close programs using this directory and rerun prepare-models.ps1."
        }
        Write-Host "Models ready: $modelRoot"
    }
} finally {
    foreach ($name in $environmentNames) { [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process') }
    Pop-Location
}
