param(
    [Parameter(Mandatory = $true)]
    [string]$Scenario,

    [double]$PruneSlowerThanPct = 10
)

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$matrix = Join-Path $PSScriptRoot "phase25_ollama_matrix.py"

if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python"
    exit 1
}

if (-not (Test-Path $matrix)) {
    Write-Error "Matrix runner not found at $matrix"
    exit 1
}

& $python $matrix --prune-slower-than-pct $PruneSlowerThanPct --scenario $Scenario
exit $LASTEXITCODE
