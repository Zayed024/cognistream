# CogniStream Performance Plan

Last updated: 2026-04-06

## Objective

Improve local CPU-first throughput and stability while preserving retrieval quality and event coverage.

## Current Status

### Completed

- Phase 0 baseline harness implemented with repeatable per-run reports.
- Phase 1 conservative gating and worker tests implemented.
- Phase 2 semantic caption reuse implemented and wired into quality metrics.
- Phase 2.5 topology benchmarking completed; best local profile identified as `np4_w1`.

### Current recommended local profile

- `VLM_WORKERS=1` in backend.
- `OLLAMA_NUM_PARALLEL=4` on the `ollama serve` process.
- Keep novelty filtering and semantic reuse enabled by default.

### Observed outcome (latest benchmark set)

- Best elapsed (tested set): about `295.8s` (`single_np4_w1`).
- Higher backend worker fanout generally regressed on this machine.
- Raising Ollama parallelism helped up to a point, then plateau/regressed.

## Constraints

- Host: Windows, CPU-only, about 16 cores and about 16 GB RAM.
- Avoid memory thrash and scheduler contention.
- Keep behavior deterministic enough for repeated benchmarking.

## Plan From Here

## Phase 3: Runtime and model benchmark

Goal: determine if runtime/model changes can beat `np4_w1` without quality drop.

Scope:

- Benchmark current Ollama path against one candidate at a time.
- Candidate tracks:
  - smaller/faster VLM variant on Ollama.
  - quantized model variant.
  - alternative local runtime if integration cost is acceptable.

Acceptance:

- At least 15% end-to-end speedup vs current local best profile.
- No material retrieval regression (max 5% drop in aggregate score bands).
- Peak RSS remains within safe operating range for this host.

Rollback:

- Any candidate with unstable runs or clear quality degradation is rejected.

## Phase 4: Selective escalation (optional)

Goal: reduce expensive reasoning on low-information frames.

Scope:

- Add a lightweight pre-gate before VLM calls.
- Escalate only uncertain/novel frames.

Acceptance:

- Additional 10%+ reduction in VLM stage time over Phase 3 winner.
- No missed high-salience events in validation set.

## Phase 5: Hardening and operationalization

Goal: make tuned settings easy to run and validate repeatedly.

Scope:

- Keep benchmark scripts as first-class tooling.
- Ensure all scenario runs produce normalized summary JSON.
- Keep startup logs explicit about active tuning profile.

Acceptance:

- One-command benchmark workflow for baseline and topology checks.
- Stable results variance within 10% on repeated runs.

## Benchmark protocol

Track per run:

- `elapsed_sec`
- `stage_timings.vlm_sec`
- keyframe kept/dropped stats
- semantic reuse counters
- retrieval summary metrics
- process RSS start/peak/delta

Required run shape:

- At least 3 repeated runs per profile for any keep/reject decision.
- Compare against latest accepted baseline, not historical stale baselines.

## Near-term TODO (next 1-2 weeks)

1. Run the remaining pending scenario (`single_np5_w2`) and append result to research report.
2. Evaluate one model/runtime candidate in a controlled A/B benchmark.
3. Promote a locked "local default profile" section in main docs and sample env.
