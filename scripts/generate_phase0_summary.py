#!/usr/bin/env python3
"""Generate Phase 0 baseline summary from completed benchmark data."""
import json
import urllib.request
import statistics
from pathlib import Path

api_base = "http://127.0.0.1:8000"
video_id = "ae3008ed225a4a63ba47e56008c6f779"

# Fetch both benchmark runs
response = urllib.request.urlopen(f"{api_base}/video/{video_id}/benchmark/history?limit=10")
data = json.loads(response.read().decode())
runs = data.get("runs", [])

print(f"Found {len(runs)} runs")

# Reverse to process oldest first (they're newest-first from API)
runs = list(reversed(runs))

# Extract metrics from both runs
elapsed_times = []
vlm_times = []
keyframes_list = []
caption_counts = []
fallback_ratios = []

for run in runs:
    stage = run.get("stage_timings", {})
    quality = run.get("quality_metrics", {})
    
    elapsed = run.get("elapsed_sec", 0)
    vlm = stage.get("vlm_sec", 0)
    keyframes = quality.get("keyframes_kept", 0)
    captions = quality.get("captions_count", 0)
    fallback = quality.get("captions_fallback_ratio", 0)
    
    elapsed_times.append(elapsed)
    vlm_times.append(vlm)
    keyframes_list.append(keyframes)
    caption_counts.append(captions)
    fallback_ratios.append(fallback)
    
    # Calculate FPM
    if vlm > 0 and keyframes > 0:
        fpm = (keyframes / vlm) * 60
        print(f"Run {run['id'][:8]}: elapsed={elapsed:.1f}s, vlm={vlm:.1f}s, frames/min={fpm:.2f}, fallback_ratio={fallback:.4f}")

# Compute aggregates
elapsed_mean = statistics.mean(elapsed_times)
elapsed_stdev = statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0
vlm_mean = statistics.mean(vlm_times)
vlm_stdev = statistics.stdev(vlm_times) if len(vlm_times) > 1 else 0
fpm_values = [(kf / vlm) * 60 for kf, vlm in zip(keyframes_list, vlm_times) if vlm > 0]
fpm_mean = statistics.mean(fpm_values) if fpm_values else 0

print(f"\nAggregates:")
print(f"  elapsed_sec_mean: {elapsed_mean:.1f} ± {elapsed_stdev:.1f}")
print(f"  vlm_fpm_mean: {fpm_mean:.2f}")

# Create summary
summary = {
    "phase": "phase0_baseline",
    "video_id": video_id,
    "run_count": len(runs),
    "metrics": {
        "elapsed_sec": {
            "mean": round(elapsed_mean, 2),
            "stdev": round(elapsed_stdev, 2),
            "values": [round(x, 2) for x in elapsed_times]
        },
        "vlm_frames_per_min": {
            "mean": round(fpm_mean, 3),
            "values": [round(x, 3) for x in fpm_values]
        },
        "keyframes_kept": {
            "mean": round(statistics.mean(keyframes_list), 1),
            "values": [round(x, 1) for x in keyframes_list]
        },
        "captions_fallback_ratio": {
            "mean": round(statistics.mean(fallback_ratios), 4),
            "values": [round(x, 4) for x in fallback_ratios]
        }
    },
    "raw_runs": [
        {
            "id": run["id"],
            "captured_at": run["captured_at"],
            "elapsed_sec": run["elapsed_sec"],
            "stage_timings": run["stage_timings"],
            "quality_metrics": run["quality_metrics"]
        }
        for run in runs
    ]
}

# Write summary JSON
output_dir = Path("reports/phase0/ae3008ed")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "phase0_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nWrote: {output_dir / 'phase0_summary.json'}")

# Write markdown report
md_report = f"""# Phase 0 Baseline Report
Generated for video: `{video_id}`

## Summary
- **Runs**: {len(runs)}
- **Average Elapsed Time**: {elapsed_mean:.2f}s ± {elapsed_stdev:.2f}s
- **Average VLM Throughput**: {fpm_mean:.3f} frames/minute

## Detailed Metrics

### Elapsed Time (seconds)
- **Mean**: {elapsed_mean:.2f}
- **Stdev**: {elapsed_stdev:.2f}
- **Values**: {elapsed_times}

### VLM Frames Per Minute
- **Mean**: {fpm_mean:.3f} fpm
- **Values**: {[round(x, 3) for x in fpm_values]}

### Keyframes Processed
- **Mean**: {statistics.mean(keyframes_list):.1f}
- **Values**: {keyframes_list}

### Captions Fallback Ratio
- **Mean**: {statistics.mean(fallback_ratios):.4f}
- **Values**: {[round(x, 4) for x in fallback_ratios]}

## Per-Run Breakdown
"""

for i, run in enumerate(runs, 1):
    stage = run.get("stage_timings", {})
    quality = run.get("quality_metrics", {})
    fpm = (quality.get("keyframes_kept", 0) / stage.get("vlm_sec", 1)) * 60
    
    md_report += f"""
### Run {i}
- **ID**: {run['id'][:16]}...
- **Captured At**: {run['captured_at']}
- **Elapsed**: {run['elapsed_sec']:.1f}s
- **VLM Time**: {stage.get('vlm_sec', 0):.2f}s
- **Keyframes**: {quality.get('keyframes_kept', 0):.0f} / {quality.get('keyframes_input', 0):.0f}
- **Throughput**: {fpm:.3f} frames/min
- **Captions Fallback Ratio**: {quality.get('captions_fallback_ratio', 0):.4f}

**Stage Timings:**
"""
    for stage_name, stage_sec in stage.items():
        md_report += f"- {stage_name}: {stage_sec:.3f}s\n"

with open(output_dir / "phase0_summary.md", "w") as f:
    f.write(md_report)

print(f"Wrote: {output_dir / 'phase0_summary.md'}")
print("\n✓ Phase 0 baseline summary complete")
