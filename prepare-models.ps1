[CmdletBinding()]
param(
    [string]$Models = (Join-Path $PSScriptRoot 'models'),
    [ValidateSet('fp32', 'tf32', 'fp16')][string]$MatcherPrecision = 'fp16',
    [ValidateSet('none', 'ampere_plus')][string]$HardwareCompatibility = 'none',
    [string]$OnnxFrom,
    [string]$OnnxBundle,
    [string[]]$Images,
    [string]$Sam3,
    [switch]$NoGrounding
)
$ErrorActionPreference = 'Stop'
if ($Images -and $Images.Count -ne 2) { throw '-Images requires two real image paths.' }
if ($Sam3 -and $NoGrounding) { throw '-Sam3 adds the grounding model; it cannot be combined with -NoGrounding.' }
if ($OnnxBundle -and ($Sam3 -or $NoGrounding -or $OnnxFrom -or $PSBoundParameters.ContainsKey('Models') -or $PSBoundParameters.ContainsKey('MatcherPrecision') -or $PSBoundParameters.ContainsKey('HardwareCompatibility'))) {
    throw '-OnnxBundle exports GPU-independent ONNX only; it cannot be combined with -Models, -OnnxFrom, -Sam3, -NoGrounding, -MatcherPrecision or -HardwareCompatibility.'
}

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
    $withoutDependencies = Join-Path $PSScriptRoot 'tools\models\requirements-nodeps.txt'
    $identity = (Get-FileHash -LiteralPath $requirements, $withoutDependencies -Algorithm SHA256).Hash -join ''
    if (-not (Test-Path -LiteralPath $stamp) -or (Get-Content -Raw -LiteralPath $stamp).Trim() -ne $identity) {
        Write-Host 'Installing the isolated model export/build environment (first run only)...'
        Invoke-Tool $uvPath @('pip', 'install', '--python', $python, 'torch==2.8.0', 'torchvision==0.23.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
        Invoke-Tool $uvPath @('pip', 'install', '--python', $python, '-r', $requirements, '--extra-index-url', 'https://pypi.nvidia.com')
        Invoke-Tool $uvPath @('pip', 'install', '--python', $python, '--no-deps', '-r', $withoutDependencies)
        Set-Content -LiteralPath $stamp -Value $identity -Encoding ASCII
    }
    $target = if ($OnnxBundle) { $OnnxBundle } else { $Models }
    $modelRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($target).TrimEnd([char[]]'\/')
    $arguments = @('-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'tools\models\prepare.py'))
    if ($OnnxBundle) { $arguments += @('--bundle', $modelRoot) }
    else { $arguments += @('--models', $modelRoot, '--matcher-precision', $MatcherPrecision, '--hardware-compatibility', $HardwareCompatibility) }
    if ($OnnxFrom) { $arguments += @('--onnx-from', $OnnxFrom) }
    if ($Sam3) { $arguments += @('--sam3', $Sam3) }
    if ($NoGrounding) { $arguments += '--no-grounding' }
    if ($Images) {
        $arguments += @('--images') + $Images
    }
    Invoke-Tool $python $arguments
    if (-not (Test-Path -LiteralPath $modelRoot)) {
        $pending = $modelRoot + '.pending'
        try {
            [IO.Directory]::Move($pending, $modelRoot)
        } catch [IO.IOException], [UnauthorizedAccessException] {
            throw "Cannot publish: $($_.Exception.Message) Verified files remain in '$pending'. Close programs using this directory and rerun prepare-models.ps1."
        }
        Write-Host $(if ($OnnxBundle) { "ONNX bundle ready: $modelRoot" } else { "Models ready: $modelRoot" })
    }
} finally {
    foreach ($name in $environmentNames) { [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process') }
    Pop-Location
}
