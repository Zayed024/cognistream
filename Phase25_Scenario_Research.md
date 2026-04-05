# Phase 2.5 Scenario Research (Ollama Topology)

Date: 2026-04-05
Target machine: Windows, CPU-only local runtime
Video under test: ae3008ed225a4a63ba47e56008c6f779

## Goal
Identify the fastest stable local strategy by tuning:
- Ollama request parallelism (single server)
- Backend VLM worker count

## Completed Scenario Results

| Scenario | Ollama Parallel | VLM Workers | Elapsed (s) | VLM (s) | Semantic Reuse Hits | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|
| single_np4_w1 | 4 | 1 | 295.8 | 269.266 | 15 | 772.5 |
| single_np5_w1 | 5 | 1 | 312.4 | 279.453 | 15 | 865.07 |
| single_np4_w2 | 4 | 2 | 389.7 | 368.234 | 9 | 867.46 |
| single_np3_w2 | 3 | 2 | 410.6 | 387.969 | 10 | 865.47 |
| single_np2_w2 | 2 | 2 | 435.0 | 411.5 | 9 | 866.45 |
| single_np3_w4 | 3 | 4 | 591.0 | 569.453 | 1 | 752.88 |

## Recommendation
Use `np4_w1` as the default local backend strategy:
- `OLLAMA_NUM_PARALLEL=4` (set where `ollama serve` runs)
- `VLM_WORKERS=1` in backend

Rationale:
- Best total latency among tested stable runs.
- Better semantic reuse than high-worker alternatives.
- Lower peak RSS than other high-performing options.

## Operational Notes
- `OLLAMA_NUM_PARALLEL` is applied by `ollama serve`, not by backend process itself.
- Increasing `VLM_WORKERS` did not improve throughput on this hardware; in several runs it regressed.
- Manual artifact generation was used for some runs when wrapper finalization lagged behind pipeline completion.

## Main Backend Implementation Applied
- Backend default VLM workers set to `1` for local strategy alignment.
- Startup logs now include:
  - `local_strategy_profile: np4_w1`
  - `ollama_num_parallel_recommended: 4`
