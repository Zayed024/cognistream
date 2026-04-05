# Scripts Guide

This folder contains benchmark and utility scripts used for local performance work.

## Core scripts

- `benchmark_local.py`
  - Query-quality and timing summary for a processed video.

- `phase0_baseline.py`
  - Repeatable baseline runner with per-run reports.

- `phase1_worker_saturation.py`
  - Worker fanout comparison on a fixed keyframe subset.

- `phase2_reuse_benchmark.py`
  - Semantic reuse benchmark wrapper and summary extraction.

- `phase25_ollama_matrix.py`
  - Scenario matrix for Ollama topology testing.

- `phase25_run.ps1`
  - Generic PowerShell helper to run one matrix scenario.
  - Example:
    - `pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\phase25_run.ps1 -Scenario single_np4_w1 -PruneSlowerThanPct 10`

- `pull_models.sh`
  - Pull required local models.

## Notes

- Benchmark outputs are intentionally not committed (`reports/` is git-ignored).
- Keep one-off/manual experiment wrappers out of source control.
