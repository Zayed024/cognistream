"""
CogniStream — Benchmark on Standard Test Videos

Processes test videos through the pipeline and records quality metrics.
Run BEFORE and AFTER fine-tuning to measure improvement.

Usage:
    python scripts/benchmark_test_videos.py
    python scripts/benchmark_test_videos.py --tag baseline    # saves as baseline
    python scripts/benchmark_test_videos.py --tag finetuned   # saves as finetuned

Output:
    reports/benchmark_<tag>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_VIDEOS_DIR = PROJECT_ROOT / "data" / "test_videos"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def process_video(video_path: Path) -> dict:
    """Process a single video through the full pipeline and collect metrics."""
    from backend.config import OLLAMA_MODEL, WHISPER_MODEL_SIZE, EMBEDDING_MODEL, PIPELINE_MODE
    from backend.db.sqlite import SQLiteDB
    from backend.db.chroma_store import ChromaStore
    from backend.ingestion.loader import VideoLoader
    from backend.pipeline.orchestrator import PipelineOrchestrator

    db = SQLiteDB()
    store = ChromaStore()
    loader = VideoLoader()
    orchestrator = PipelineOrchestrator(db=db, store=store)

    logger.info("Processing: %s", video_path.name)
    t_start = time.monotonic()

    # Ingest
    meta = loader.load(video_path)
    db.save_video(meta)

    # Process
    result = orchestrator.process(meta)

    elapsed = time.monotonic() - t_start

    # Collect metrics
    segments = store.get_by_video(meta.id)
    source_counts = {}
    for seg in segments:
        st = seg.get("source_type", "unknown")
        source_counts[st] = source_counts.get(st, 0) + 1

    # Sample captions for quality assessment
    caption_samples = []
    for seg in segments[:10]:
        text = seg.get("text", "")
        caption_samples.append({
            "start_time": seg.get("start_time", 0),
            "source_type": seg.get("source_type", ""),
            "text": text[:200],
            "length": len(text),
            "is_empty": len(text.strip()) < 10,
        })

    empty_count = sum(1 for s in caption_samples if s["is_empty"])

    metrics = {
        "video": video_path.name,
        "video_id": meta.id,
        "duration_sec": meta.duration_sec,
        "fps": meta.fps,
        "total_frames": meta.total_frames,
        "success": result.success,
        "elapsed_sec": round(elapsed, 1),
        "segments_stored": result.segments_stored,
        "events_detected": result.events_detected,
        "segments_by_source": source_counts,
        "stage_timings": result.stage_timings,
        "quality_metrics": result.quality_metrics,
        "warnings": result.warnings,
        "errors": result.errors,
        "caption_samples": caption_samples,
        "empty_caption_ratio": round(empty_count / max(1, len(caption_samples)), 3),
    }

    logger.info(
        "  %s: %s | %d segments | %d events | %.1fs | empty=%.0f%%",
        video_path.name,
        "OK" if result.success else "FAIL",
        result.segments_stored,
        result.events_detected,
        elapsed,
        metrics["empty_caption_ratio"] * 100,
    )

    # Cleanup — delete processed data so next video starts fresh
    store.purge_video(meta.id)
    db.delete_video(meta.id)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark CogniStream on test videos")
    parser.add_argument("--tag", default="baseline", help="Tag for this run (e.g. baseline, finetuned)")
    parser.add_argument("--videos-dir", default=str(TEST_VIDEOS_DIR))
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    videos = sorted(videos_dir.glob("*.mp4"))
    # Skip corrupted files
    videos = [v for v in videos if v.stat().st_size > 10000]

    if not videos:
        logger.error("No test videos found in %s", videos_dir)
        logger.error("Run: python scripts/download_test_videos.py")
        sys.exit(1)

    logger.info("Benchmarking %d test videos (tag=%s)", len(videos), args.tag)
    logger.info("")

    # Record config
    from backend.config import OLLAMA_MODEL, WHISPER_MODEL_SIZE, EMBEDDING_MODEL, PIPELINE_MODE
    from backend.providers.nvidia import nvidia

    config = {
        "vlm_model": OLLAMA_MODEL,
        "whisper_model": WHISPER_MODEL_SIZE,
        "embedding_model": EMBEDDING_MODEL,
        "pipeline_mode": PIPELINE_MODE,
        "nvidia_cloud": nvidia.available,
    }

    results = []
    for video in videos:
        try:
            metrics = process_video(video)
            results.append(metrics)
        except Exception as exc:
            logger.error("  %s: CRASHED — %s", video.name, exc)
            results.append({
                "video": video.name,
                "success": False,
                "error": str(exc),
            })

    # Summary
    successful = [r for r in results if r.get("success")]
    total_segments = sum(r.get("segments_stored", 0) for r in successful)
    total_events = sum(r.get("events_detected", 0) for r in successful)
    total_time = sum(r.get("elapsed_sec", 0) for r in successful)
    avg_empty = (
        sum(r.get("empty_caption_ratio", 0) for r in successful) / max(1, len(successful))
    )

    report = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "summary": {
            "videos_processed": len(successful),
            "videos_failed": len(results) - len(successful),
            "total_segments": total_segments,
            "total_events": total_events,
            "total_time_sec": round(total_time, 1),
            "avg_empty_caption_ratio": round(avg_empty, 3),
        },
        "videos": results,
    }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"benchmark_{args.tag}_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS (tag=%s)", args.tag)
    logger.info("=" * 60)
    logger.info("  Videos processed: %d/%d", len(successful), len(results))
    logger.info("  Total segments:   %d", total_segments)
    logger.info("  Total events:     %d", total_events)
    logger.info("  Total time:       %.1fs", total_time)
    logger.info("  Avg empty ratio:  %.1f%%", avg_empty * 100)
    logger.info("")
    logger.info("  Report saved: %s", report_path)


if __name__ == "__main__":
    main()
