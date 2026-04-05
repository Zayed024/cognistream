param(
    [double]$PruneSlowerThanPct = 10
)

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$matrix = Join-Path $PSScriptRoot "phase25_ollama_matrix.py"

& $python $matrix --prune-slower-than-pct $PruneSlowerThanPct --scenario single_np3_w4
exit $LASTEXITCODE
