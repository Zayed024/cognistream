#!/usr/bin/env python3
"""Finalize Phase 1 by comparing Phase 0 baseline against latest successful run."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASE0 = ROOT / "reports" / "phase0" / "ae3008ed" / "phase0_summary.json"
PHASE1_SAT = ROOT / "reports" / "phase1" / "worker_saturation_ae3008ed225a4a63ba47e56008c6f779.json"
HISTORY = ROOT / "reports" / "phase1" / "ae3008ed" / "bench_phase1.json"
OUT_JSON = ROOT / "reports" / "phase1" / "phase1_completion.json"
OUT_MD = ROOT / "reports" / "phase1" / "phase1_completion.md"

with PHASE0.open("r", encoding="utf-8") as f:
    phase0 = json.load(f)
with PHASE1_SAT.open("r", encoding="utf-8") as f:
    sat = json.load(f)
with HISTORY.open("r", encoding="utf-8") as f:
    bench = json.load(f)

base_elapsed = float(phase0["metrics"]["elapsed_sec"]["mean"])
base_fpm = float(phase0["metrics"]["vlm_frames_per_min"]["mean"])

pipeline = bench.get("pipeline_benchmark") or {}
stage = pipeline.get("stage_timings") or {}
quality = pipeline.get("quality_metrics") or {}

p1_elapsed = float(pipeline.get("elapsed_sec", 0.0) or 0.0)
keyframes_kept = float(quality.get("keyframes_kept", 0.0) or 0.0)
vlm_sec = float(stage.get("vlm_sec", 0.0) or 0.0)
p1_fpm = (keyframes_kept / vlm_sec * 60.0) if (keyframes_kept > 0 and vlm_sec > 0) else 0.0

elapsed_delta_pct = ((p1_elapsed - base_elapsed) / base_elapsed * 100.0) if base_elapsed > 0 else 0.0
fpm_delta_pct = ((p1_fpm - base_fpm) / base_fpm * 100.0) if base_fpm > 0 else 0.0

summary = {
    "phase": "phase1",
    "status": "completed",
    "video_id": bench.get("video_id"),
    "phase0_reference": {
        "elapsed_sec_mean": round(base_elapsed, 3),
        "vlm_frames_per_min_mean": round(base_fpm, 3),
    },
    "phase1_latest": {
        "elapsed_sec": round(p1_elapsed, 3),
        "vlm_frames_per_min": round(p1_fpm, 3),
        "keyframes_kept": round(keyframes_kept, 1),
        "keyframes_input": float(quality.get("keyframes_input", 0.0) or 0.0),
        "keyframes_dropped": float(quality.get("keyframes_dropped", 0.0) or 0.0),
        "process_rss_peak_mb": float(quality.get("process_rss_peak_mb", 0.0) or 0.0),
        "captions_fallback_ratio": float(quality.get("captions_fallback_ratio", 0.0) or 0.0),
    },
    "delta_vs_phase0_pct": {
        "elapsed_sec": round(elapsed_delta_pct, 3),
        "vlm_frames_per_min": round(fpm_delta_pct, 3),
    },
    "worker_saturation": {
        "tested_workers": [r["workers"] for r in sat.get("runs", [])],
        "best_workers": sat.get("best_workers"),
        "best_frames_per_min": sat.get("best_frames_per_min"),
        "gain_vs_workers1_pct": sat.get("gain_vs_baseline_pct"),
    },
    "retrieval_summary": bench.get("retrieval_summary") or {},
    "recommendation": {
        "keep_vlm_workers": int(sat.get("best_workers") or 1),
        "keep_phase1_gating_defaults": True,
    },
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

md = []
md.append("# Phase 1 Completion")
md.append("")
md.append(f"Video: {summary['video_id']}")
md.append("")
md.append("## Outcome")
md.append(f"- Status: {summary['status']}")
md.append(f"- Elapsed vs Phase 0: {summary['delta_vs_phase0_pct']['elapsed_sec']}%")
md.append(f"- VLM frames/min vs Phase 0: {summary['delta_vs_phase0_pct']['vlm_frames_per_min']}%")
md.append("")
md.append("## Worker Saturation")
md.append(f"- Tested workers: {summary['worker_saturation']['tested_workers']}")
md.append(f"- Best workers: {summary['worker_saturation']['best_workers']}")
md.append(f"- Gain vs workers=1: {summary['worker_saturation']['gain_vs_workers1_pct']}%")
md.append("")
md.append("## Retrieval")
retr = summary.get("retrieval_summary") or {}
md.append(f"- Diversity index: {retr.get('diversity_index')}")
md.append(f"- Avg result score: {retr.get('avg_result_score')}")
md.append("")
md.append("## Memory")
md.append(f"- Peak RSS MB: {summary['phase1_latest']['process_rss_peak_mb']}")

OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"Wrote: {OUT_JSON}")
print(f"Wrote: {OUT_MD}")
