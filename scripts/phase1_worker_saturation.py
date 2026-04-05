#!/usr/bin/env python3
"""Phase 1 worker saturation benchmark for local VLM analysis.

Runs VLM analysis over a fixed keyframe subset with worker counts 1, 2, and 4,
then reports elapsed time, throughput, and novelty-filter stats.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when running as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.models import Keyframe  # noqa: E402
from backend.visual.vlm_runner import OllamaClient, VLMRunner  # noqa: E402


def _http_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _load_keyframes(video_id: str, sample_size: int) -> list[Keyframe]:
    frame_dir = PROJECT_ROOT / "data" / "frames" / video_id
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found for video {video_id}: {frame_dir}")

    frame_paths = sorted([p for p in frame_dir.glob("*.jpg") if p.is_file()])
    if not frame_paths:
        raise RuntimeError(f"No keyframes found under {frame_dir}")

    if sample_size > 0 and len(frame_paths) > sample_size:
        # Uniform subsample to keep coverage while controlling benchmark runtime.
        idxs = [round(i * (len(frame_paths) - 1) / (sample_size - 1)) for i in range(sample_size)]
        selected = [frame_paths[i] for i in idxs]
    else:
        selected = frame_paths

    keyframes: list[Keyframe] = []
    for idx, path in enumerate(selected):
        try:
            frame_number = int(path.stem)
        except ValueError:
            frame_number = idx
        keyframes.append(
            Keyframe(
                video_id=video_id,
                segment_index=0,
                frame_number=frame_number,
                timestamp=float(idx),
                file_path=str(path),
            )
        )

    return keyframes


def _run_once(video_id: str, keyframes: list[Keyframe], workers: int, mode: str) -> dict[str, Any]:
    # VLMRunner reads worker count from config/env, so set env before construction.
    os.environ["VLM_WORKERS"] = str(workers)

    client = OllamaClient()
    runner = VLMRunner(client=client, fast_mode=(mode == "fast"))

    started = time.monotonic()
    try:
        captions = runner.analyse_keyframes(keyframes)
    finally:
        client.close()
    elapsed = time.monotonic() - started

    fps_min = (len(captions) / elapsed) * 60.0 if elapsed > 0 else 0.0

    return {
        "workers": workers,
        "mode": mode,
        "input_keyframes": len(keyframes),
        "captions": len(captions),
        "elapsed_sec": round(elapsed, 3),
        "frames_per_min": round(fps_min, 3),
        "novelty_stats": runner.last_novelty_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 worker saturation benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--video-id", required=True, help="Video ID to benchmark")
    parser.add_argument("--sample-size", type=int, default=24, help="Number of keyframes to benchmark")
    parser.add_argument("--mode", choices=["fast", "quality"], default="fast", help="VLM analysis mode")
    parser.add_argument("--workers", default="1,2,4", help="Comma-separated worker counts")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Default: reports/phase1/worker_saturation_<video_id>.json",
    )
    args = parser.parse_args()

    # Validate backend availability early so the benchmark fails fast when API is down.
    _http_json(f"{args.base_url}/health")

    worker_counts = []
    for token in args.workers.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("Worker count must be >= 1")
        worker_counts.append(value)
    if not worker_counts:
        raise ValueError("No valid worker counts provided")

    keyframes = _load_keyframes(args.video_id, args.sample_size)
    print(f"Loaded {len(keyframes)} keyframes for video {args.video_id}")

    runs: list[dict[str, Any]] = []
    for workers in worker_counts:
        print(f"Running workers={workers} ...")
        run = _run_once(args.video_id, keyframes, workers, args.mode)
        runs.append(run)
        print(
            f"  workers={workers} elapsed={run['elapsed_sec']:.1f}s "
            f"fpm={run['frames_per_min']:.2f} captions={run['captions']}"
        )

    best = max(runs, key=lambda r: r.get("frames_per_min", 0.0))
    baseline = next((r for r in runs if r["workers"] == 1), runs[0])
    gain = 0.0
    if baseline["frames_per_min"] > 0:
        gain = ((best["frames_per_min"] - baseline["frames_per_min"]) / baseline["frames_per_min"]) * 100.0

    summary = {
        "video_id": args.video_id,
        "mode": args.mode,
        "sample_size": len(keyframes),
        "runs": runs,
        "best_workers": best["workers"],
        "best_frames_per_min": best["frames_per_min"],
        "baseline_workers": baseline["workers"],
        "baseline_frames_per_min": baseline["frames_per_min"],
        "gain_vs_baseline_pct": round(gain, 2),
        "frames_per_min_mean": round(statistics.fmean([r["frames_per_min"] for r in runs]), 3),
    }

    output_path = Path(args.output) if args.output else (PROJECT_ROOT / "reports" / "phase1" / f"worker_saturation_{args.video_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(
        f"Best workers={summary['best_workers']} fpm={summary['best_frames_per_min']:.3f} "
        f"gain_vs_workers1={summary['gain_vs_baseline_pct']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
