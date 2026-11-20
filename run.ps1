$ErrorActionPreference = 'Stop'
$tools = Join-Path $PSScriptRoot 'tools\windows'
if ($args[0] -eq 'prepare-models') {
    if ($args.Count -gt 2) { throw 'Usage: .\run.ps1 prepare-models [path-to-sam3.pt]' }
    $options = @{}
    if ($args.Count -eq 2) { $options.Sam3 = (Resolve-Path -LiteralPath $args[1]).Path }
    & (Join-Path $tools 'prepare-models.ps1') @options
    return
}

# Development shortcut: `--input SCENE [--project NAME.pro]` expands to the explicit inputs the
# program requires, using the usual PCC scene layout. The program itself never guesses paths.
function Expand-SceneInput([string[]]$arguments) {
    $index = [Array]::IndexOf($arguments, '--input')
    if ($index -lt 0) { return , $arguments }
    if ($index + 1 -ge $arguments.Count) { throw 'Expected --input SCENE' }
    $scene = (Resolve-Path -LiteralPath $arguments[$index + 1]).Path
    $rest = [System.Collections.Generic.List[string]]::new()
    $project = $null
    for ($i = 0; $i -lt $arguments.Count; ++$i) {
        if ($i -eq $index) { ++$i; continue }
        if ($arguments[$i] -eq '--project' -and $i + 1 -lt $arguments.Count) {
            $project = $arguments[++$i]; continue
        }
        $rest.Add($arguments[$i])
    }
    if ($project) {
        if ([IO.Path]::GetFileName($project) -ne $project) { $project = (Resolve-Path -LiteralPath $project).Path }
        else { $project = Join-Path $scene $project }
    } else {
        $candidates = @(Get-ChildItem -LiteralPath $scene -File -Filter *.pro)
        if (-not $candidates) { throw "No .pro project in $scene" }
        $latest = ($candidates | Measure-Object LastWriteTimeUtc -Maximum).Maximum
        $newest = @($candidates | Where-Object LastWriteTimeUtc -eq $latest)
        if ($newest.Count -gt 1) {
            throw "PCC projects share the latest modification time. Use --project to select one:`n  " +
                  (($newest | ForEach-Object Name) -join "`n  ")
        }
        $project = $newest[0].FullName
    }
    $stem = [IO.Path]::GetFileNameWithoutExtension($project)
    $folder = Split-Path -Parent $project
    $images = Join-Path $scene 'Image'
    $lenses = @('L', 'R') | ForEach-Object { Join-Path $images $_ } | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
    if (-not $lenses) { $lenses = @($images) }
    $las = Join-Path $scene "ResultOut\$stem.las"
    if (-not (Test-Path -LiteralPath $las) -and (Test-Path -LiteralPath (Join-Path $scene "$stem.las"))) {
        $las = Join-Path $scene "$stem.las"
    }
    $expanded = [System.Collections.Generic.List[string]]::new()
    $expanded.Add($rest[0])
    $expanded.AddRange([string[]]@('--project', $project))
    foreach ($lens in $lenses) { $expanded.AddRange([string[]]@('--images', $lens)) }
    $expanded.AddRange([string[]]@(
        '--camera-pos', (Join-Path $folder "${stem}_CameraPos_C2E.txt"),
        '--pos', (Join-Path $folder "${stem}_pos.txt"),
        '--pos-enh', (Join-Path $folder "${stem}_pos_ENH.txt"),
        '--las', $las))
    if (-not $rest.Contains('--output') -and -not $rest.Contains('--work')) {
        $expanded.AddRange([string[]]@('--output', (Join-Path $PSScriptRoot ('output\' + (Split-Path -Leaf $scene)))))
    }
    for ($i = 1; $i -lt $rest.Count; ++$i) { $expanded.Add($rest[$i]) }
    Write-Host ('Expanded --input: ' + ($expanded -join ' '))
    return , $expanded.ToArray()
}

$arguments = Expand-SceneInput ([string[]]$args)
& (Join-Path $tools 'build.ps1') | Out-Host
$program = Join-Path $PSScriptRoot 'build\ninja\bin\geosunpre.exe'
$previousModels = $env:GEOSUNPRE_MODELS
try {
    $env:GEOSUNPRE_MODELS = Join-Path $PSScriptRoot 'models'
    Write-Host "Development executable: $program"
    & $program @arguments
    $code = $LASTEXITCODE
} finally { $env:GEOSUNPRE_MODELS = $previousModels }
exit $code
