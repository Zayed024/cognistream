"""
CogniStream — Knowledge Distillation Script

Generates high-quality training data by running NVIDIA cloud VLM
(Llama-3.2-11B-Vision) on keyframes, then saving the results as
a training dataset for fine-tuning moondream.

Usage:
    # Distill from all processed videos
    python scripts/finetune/distill.py

    # Distill from specific video
    python scripts/finetune/distill.py --video-id abc123

    # Limit number of frames
    python scripts/finetune/distill.py --max-frames 500

Output:
    data/finetune/distillation_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import FRAME_DIR, DATA_DIR
from backend.providers.nvidia import nvidia
from backend.visual.caption_processor import PromptLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FINETUNE_DIR = DATA_DIR / "finetune"
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = FINETUNE_DIR / "distillation_dataset.jsonl"


def collect_keyframes(video_id: str | None, max_frames: int) -> list[Path]:
    """Collect keyframe image paths from processed videos."""
    frames = []

    if video_id:
        video_dirs = [FRAME_DIR / video_id]
    else:
        video_dirs = [d for d in FRAME_DIR.iterdir() if d.is_dir() and not d.name.startswith("live-")]

    for vdir in video_dirs:
        jpgs = sorted(vdir.glob("**/*.jpg"))
        frames.extend(jpgs)

    # Deduplicate and limit
    frames = frames[:max_frames]
    logger.info("Collected %d keyframes from %d video(s)", len(frames), len(video_dirs))
    return frames


def distill_frame(image_path: Path, prompt: str) -> dict | None:
    """Generate a teacher caption for one frame via NVIDIA cloud."""
    result = nvidia.caption_image(str(image_path), prompt)
    if not result or len(result.strip()) < 10:
        return None

    return {
        "image_path": str(image_path),
        "prompt": prompt,
        "response": result.strip(),
        "video_id": image_path.parent.name,
    }


def main():
    parser = argparse.ArgumentParser(description="Distill training data from NVIDIA cloud VLM")
    parser.add_argument("--video-id", help="Specific video to distill from")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing dataset")
    args = parser.parse_args()

    if not nvidia.available:
        logger.error("NVIDIA API key not set. Set NVIDIA_API_KEY in .env first.")
        sys.exit(1)

    # Collect frames
    frames = collect_keyframes(args.video_id, args.max_frames)
    if not frames:
        logger.error("No keyframes found. Process some videos first.")
        sys.exit(1)

    # Load existing dataset if resuming
    existing = set()
    if args.resume and DATASET_PATH.exists():
        with open(DATASET_PATH) as f:
            for line in f:
                item = json.loads(line)
                existing.add(item["image_path"])
        logger.info("Resuming: %d existing entries, skipping those", len(existing))

    # Use the combined prompt (same one moondream will be trained on)
    prompt = PromptLibrary.combined_prompt()

    # Distill
    mode = "a" if args.resume else "w"
    success = 0
    failed = 0
    t_start = time.monotonic()

    with open(DATASET_PATH, mode) as f:
        for i, frame_path in enumerate(frames):
            if str(frame_path) in existing:
                continue

            item = distill_frame(frame_path, prompt)
            if item:
                f.write(json.dumps(item) + "\n")
                f.flush()
                success += 1
            else:
                failed += 1

            # Progress
            elapsed = time.monotonic() - t_start
            rate = (success + failed) / elapsed if elapsed > 0 else 0
            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                logger.info(
                    "Progress: %d/%d (success=%d, failed=%d, %.1f frames/s)",
                    i + 1, len(frames), success, failed, rate,
                )

    elapsed = time.monotonic() - t_start
    logger.info(
        "Distillation complete: %d successful, %d failed, %.1f min",
        success, failed, elapsed / 60,
    )
    logger.info("Dataset saved to: %s", DATASET_PATH)
    logger.info("Dataset size: %d entries", success + len(existing))


if __name__ == "__main__":
    main()
