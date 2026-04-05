#!/usr/bin/env python3
"""Phase 2 benchmark helper focused on semantic reuse metrics.

Runs the existing phase0 baseline harness (single run) and extracts
reuse-hit metrics from the generated benchmark payload.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 semantic reuse benchmark")
    parser.add_argument("--video-id", required=True, help="Video ID to benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--timeout-sec", type=int, default=12000, help="Timeout for one run")
    parser.add_argument("--output-dir", default="reports/phase2", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "phase0_baseline.py"),
        "--base-url",
        args.base_url,
        "--video-id",
        args.video_id,
        "--runs",
        "1",
        "--timeout-sec",
        str(args.timeout_sec),
        "--output-dir",
        str(output_dir),
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"Baseline command failed with exit code {completed.returncode}")
        return completed.returncode

    summary_path = output_dir / "phase0_summary.json"
    if not summary_path.is_file():
        print(f"Missing summary output: {summary_path}")
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    videos = summary.get("videos") or []
    if not videos:
        print("No video results found in summary")
        return 3

    runs = videos[0].get("runs") or []
    if not runs:
        print("No runs found in summary")
        return 4

    bench = runs[0].get("benchmark") or {}
    quality = bench.get("quality_metrics") or {}

    hits_total = float(quality.get("reuse_hits_total", 0.0) or 0.0)
    hits_exact = float(quality.get("reuse_hits_exact", 0.0) or 0.0)
    hits_semantic = float(quality.get("reuse_hits_semantic", 0.0) or 0.0)
    misses = float(quality.get("reuse_misses", 0.0) or 0.0)
    denom = hits_total + misses
    hit_ratio = (hits_total / denom) if denom > 0 else 0.0

    phase2 = {
        "video_id": args.video_id,
        "reuse_hits_total": hits_total,
        "reuse_hits_exact": hits_exact,
        "reuse_hits_semantic": hits_semantic,
        "reuse_misses": misses,
        "reuse_hit_ratio": round(hit_ratio, 4),
        "quality_metrics": quality,
    }

    out = output_dir / "phase2_summary.json"
    out.write_text(json.dumps(phase2, indent=2), encoding="utf-8")

    print("Phase 2 summary:")
    print(json.dumps(phase2, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
