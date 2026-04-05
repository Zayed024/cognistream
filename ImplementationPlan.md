# CogniStream VLM Performance Implementation Plan

## Goal

Improve local/offline video analysis throughput without materially reducing retrieval relevance, event fidelity, or stability.

## Current Baseline

- Hardware: Windows, 16 CPU cores, 15.73 GB RAM, no CUDA
- Current local path: Ollama-based VLM, CPU-only
- Observed VLM throughput: about 2.4 to 2.5 frames/min on 127 keyframes
- Current auto tuning resolves VLM workers to 1 in local CPU mode
- Existing protections already in place:
  - novelty prefilter before VLM
  - adaptive worker selection
  - benchmark endpoints and local benchmark script
  - per-run debug logs

## Principles

1. Reduce the number of frames that reach the VLM before changing the model backend.
2. Benchmark every change against the current baseline.
3. Preserve quality by using conservative gates and clear rollback thresholds.
4. Prefer incremental changes over a full runtime swap in one step.
5. Treat OpenVINO or any model swap as a benchmarked option, not a default assumption.
6. Move obvious low-risk throughput gains earlier than runtime migrations.

## Not in Scope For The First Pass

- Full three-stage cascade with multiple heavy detectors
- Immediate migration to a new runtime for every model
- Increasing VLM workers beyond what CPU-only hardware can sustain
- Any change that cannot be benchmarked against the current baseline

## Phase 0: Baseline Measurement

Purpose: establish a stable reference point before changing logic.

Tasks:

- Run the current pipeline on a small, repeatable video set.
- Capture:
  - VLM frames/min
  - total elapsed time
  - stage timings
  - benchmark trend data
  - retrieval diversity and result quality
- Record memory peak and any instability.

Exit criteria:

- Baseline metrics are documented and repeatable.

Rollback trigger:

- Not applicable; this is measurement only.

## Phase 1: Frame Reduction and Safer Gating

Purpose: lower the total number of VLM calls with minimal quality risk.

Tasks:

- Tighten novelty filtering conservatively.
- Reduce max keyframes only if quality stays stable.
- Keep local VLM workers at 1 unless benchmark evidence shows a gain.
- Validate that fewer frames still preserve important scenes and events.
- Run a small worker-saturation test in parallel with this phase: compare 1, 2, and 4 workers on the same short benchmark set.
- Keep the higher worker count only if throughput improves without instability or quality loss.

Suggested starting changes:

- Lower max keyframes modestly
- Increase novelty threshold slightly
- Keep a minimum retained frame floor

Success metrics:

- Fewer VLM calls per video
- Lower total elapsed time
- Throughput at or above 10 frames/min on the benchmark set
- Retrieval quality drop no worse than 5% versus the current baseline
- No more than 10% variance under repeated runs on the same input

Rollback trigger:

- Missed salient events or clear retrieval degradation.
- Worker saturation causes contention, crashes, or slower end-to-end runs.

## Phase 2: Lightweight Semantic Cache

Purpose: skip redundant VLM work on near-identical frames or repeated scenes.

Tasks:

- Add a conservative similarity-based reuse layer.
- Reuse captions for frames that are effectively unchanged.
- Keep cache invalidation simple at first.
- Track cache hit ratio and false reuse risk.

Preferred design:

- Start with embedding-based semantic reuse
- Keep exact-frame or near-duplicate reuse separate from broader semantic reuse

Success metrics:

- Cache hit ratio improves on static or slow-moving scenes
- Total VLM calls drop further
- Quality remains stable on change-heavy scenes

Rollback trigger:

- Wrong caption reuse across scene changes or lighting transitions.

## Phase 3: Runtime and Model Benchmarking

Purpose: compare a faster local model/runtime before migrating the whole pipeline.

Tasks:

- Benchmark the current model/runtime against candidate faster local options.
- Evaluate only one change at a time:
  - model family
  - quantization level
  - runtime backend
- Compare on the same benchmark set.
- Treat INT8 quantization as a required benchmark path, not an optional tweak.
- Compare INT8 against the current baseline before considering more aggressive compression.

Candidate directions to test:

- smaller or faster vision model variants
- quantized builds
- OpenVINO or llama.cpp style runtimes if supported and stable
- INT8 quantization for the chosen CPU-local VLM as the first model-level optimization

Success metrics:

- Throughput at or above 25 frames/min on the benchmark set for the winning configuration
- Retrieval quality drop no worse than 5% versus the baseline
- Peak memory stays below 15.5 GB

Rollback trigger:

- Quality regression beyond the acceptable threshold.
- Memory pressure, swap activity, or unstable runtime behavior.

## Phase 4: Selective Escalation

Purpose: reserve expensive reasoning for hard frames only.

Tasks:

- Route easy frames through the fast path.
- Escalate only uncertain or complex frames.
- Use confidence or novelty thresholds to decide when to escalate.
- Avoid making the cascade too deep too early.
- Use YOLOv8 or an equivalent lightweight CPU detector as Stage I to reject frames with no salient motion or object presence before VLM analysis.
- Keep Stage II as a semantic similarity gate for near-duplicate frames.

Success metrics:

- Hard frames still get detailed analysis
- Easy frames avoid unnecessary expensive inference
- Average throughput improves while retrieval quality stays within the 5% tolerance band
- Stage I removes a measurable share of frames before they reach the VLM

Rollback trigger:

- Increased false negatives or poor event coverage.
- Stage I gate misses salient events or suppresses too many valid frames.

## Phase 5: Limited Parallelism Tuning

Purpose: use hardware more effectively without overcommitting memory or CPU.

Tasks:

- Re-test limited parallelism only after frame reduction and caching are in place.
- Keep concurrency small and benchmark each step.
- Watch memory pressure and Windows stability carefully.
- If the worker-saturation test in Phase 1 shows a win, promote the best safe worker count into the main path earlier instead of waiting for this phase.

Recommendation:

- Do not assume more workers are better on CPU-only local inference.
- Treat any increase in concurrency as an empirical question.

Success metrics:

- Better throughput without instability
- No increase in memory thrash or process contention
- No more than 10% variance in throughput or inter-token latency under repeated runs

Rollback trigger:

- Thermal throttling, swap usage, crashes, or slower end-to-end runs.

## Benchmark Protocol

Use the existing benchmark endpoints and script to compare runs.

Track:

- frames/min
- elapsed time per stage
- cache hit ratio
- retrieval diversity
- result score distribution
- benchmark trend over multiple runs

Minimum acceptance thresholds:

- Throughput must be at least 10 frames/min for a pass
- Target throughput is 25 frames/min or better for the optimized configuration
- Retrieval quality must not drop by more than 5% versus the baseline
- Stability must remain within 10% variance under repeated runs and concurrent load

## Recommended Order Of Work

1. Baseline measurement
2. Frame reduction and conservative gating
3. Small parallelism test in parallel with frame reduction
4. Semantic cache
5. Runtime/model benchmark with INT8 quantization
6. Selective escalation with YOLOv8 Stage I
7. Limited parallelism tuning only if still needed

## Review Questions Before Implementation

- What is the minimum quality regression you are willing to accept?
- Which matters more for the first release: speed or model quality?
- Do you want to prioritize static-scene speedups or complex-scene reasoning?
- Should the first implementation stay entirely CPU-local, or may it optionally use cloud inference later?

## Immediate Next Step

Review this plan and adjust the phase order, acceptance thresholds, or scope before any implementation changes are made.
