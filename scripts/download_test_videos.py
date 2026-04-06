"""
CogniStream — Download Standard Test Videos

Downloads a curated set of benchmark videos from public sources
for testing the pipeline across diverse content types.

Videos are short (15-60s), diverse, and commonly used in research:
- Surveillance / CCTV footage
- News broadcasts
- Cooking / instructional
- Indoor activities
- Outdoor / traffic scenes
- Lectures / presentations

Usage:
    python scripts/download_test_videos.py

Output:
    data/test_videos/  (5-10 clips, ~200-500 MB total)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated test videos — public domain / CC licensed / research use
# Each is short (15-60s), diverse content, good for testing different pipeline stages
TEST_VIDEOS = [
    # Creative Commons / Public Domain videos from archive.org and Blender
    {
        "name": "big_buck_bunny_clip",
        "url": "https://download.blender.org/perf/bbb_split/bbb_split_SBR-00001-00030.mp4.zip",
        "desc": "Animated outdoor scene — tests object detection, scene description",
        "direct": False,  # Needs extraction
    },
    {
        "name": "tears_of_steel_clip",
        "url": "https://ftp.nluug.nl/pub/graphics/blender/demo/movies/ToS/tears_of_steel_720p.mov",
        "desc": "Sci-fi film — tests complex scenes, people, action",
        "direct": True,
        "max_duration": 60,
    },
]

# YouTube videos (research datasets commonly use these)
# Using only CC-licensed or public domain content
YOUTUBE_VIDEOS = [
    {
        "name": "traffic_cam",
        "url": "https://www.youtube.com/watch?v=MNn9qKG2UFI",
        "desc": "Traffic camera footage — surveillance/CCTV scenario",
        "max_duration": 60,
    },
    {
        "name": "cooking_demo",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Placeholder — replace with actual CC cooking video
        "desc": "Cooking demonstration — instructional content",
        "max_duration": 60,
    },
    {
        "name": "lecture_clip",
        "url": "https://www.youtube.com/watch?v=aircAruvnKk",
        "desc": "3Blue1Brown neural networks — educational/lecture content",
        "max_duration": 60,
    },
    {
        "name": "news_broadcast",
        "url": "https://www.youtube.com/watch?v=9Auq9mYxFEE",
        "desc": "News broadcast clip — multi-person, text overlays",
        "max_duration": 60,
    },
    {
        "name": "indoor_activity",
        "url": "https://www.youtube.com/watch?v=2lAe1cqCOXo",
        "desc": "Indoor activity — people in a room",
        "max_duration": 60,
    },
    {
        "name": "outdoor_nature",
        "url": "https://www.youtube.com/watch?v=LXb3EKWsInQ",
        "desc": "Nature/outdoor — landscape, animals, wide shots",
        "max_duration": 60,
    },
]

# Xiph.org standard test sequences (raw, reliable, always available)
XIPH_VIDEOS = [
    {
        "name": "xiph_foreman",
        "url": "https://media.xiph.org/video/derf/y4m/foreman_cif.y4m",
        "desc": "Classic CV test — man talking to camera at construction site",
    },
    {
        "name": "xiph_news",
        "url": "https://media.xiph.org/video/derf/y4m/news_cif.y4m",
        "desc": "Classic CV test — news anchors at desk",
    },
    {
        "name": "xiph_bus",
        "url": "https://media.xiph.org/video/derf/y4m/bus_cif.y4m",
        "desc": "Classic CV test — bus moving through city",
    },
]


def download_youtube(video: dict) -> bool:
    """Download a YouTube video clip using yt-dlp."""
    output_path = OUTPUT_DIR / f"{video['name']}.mp4"
    if output_path.exists():
        logger.info("Already exists: %s", output_path.name)
        return True

    max_dur = video.get("max_duration", 60)
    cmd = [
        "yt-dlp",
        "--format", "best[height<=720][ext=mp4]/best[height<=720]/best",
        "--output", str(output_path),
        "--no-playlist",
        "--download-sections", f"*0-{max_dur}",
        "--quiet",
        video["url"],
    ]

    logger.info("Downloading: %s (%s)", video["name"], video["desc"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and output_path.exists():
            size_mb = output_path.stat().st_size / 1e6
            logger.info("  OK: %s (%.1f MB)", output_path.name, size_mb)
            return True
        else:
            logger.warning("  FAILED: %s — %s", video["name"], result.stderr[:200])
            return False
    except subprocess.TimeoutExpired:
        logger.warning("  TIMEOUT: %s", video["name"])
        return False
    except FileNotFoundError:
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        return False


def download_direct(video: dict) -> bool:
    """Download a direct URL video."""
    output_path = OUTPUT_DIR / f"{video['name']}.mp4"
    if output_path.exists():
        logger.info("Already exists: %s", output_path.name)
        return True

    logger.info("Downloading: %s (%s)", video["name"], video["desc"])
    try:
        import httpx
        with httpx.stream("GET", video["url"], timeout=60, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)

        # Trim to max duration if needed
        max_dur = video.get("max_duration")
        if max_dur:
            trimmed = OUTPUT_DIR / f"{video['name']}_trimmed.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(output_path), "-t", str(max_dur), "-c", "copy", str(trimmed)],
                capture_output=True, timeout=30,
            )
            if trimmed.exists() and trimmed.stat().st_size > 0:
                trimmed.replace(output_path)

        size_mb = output_path.stat().st_size / 1e6
        logger.info("  OK: %s (%.1f MB)", output_path.name, size_mb)
        return True
    except Exception as exc:
        logger.warning("  FAILED: %s — %s", video["name"], exc)
        output_path.unlink(missing_ok=True)
        return False


def download_xiph(video: dict) -> bool:
    """Download and convert a Xiph.org Y4M test sequence to MP4."""
    y4m_path = OUTPUT_DIR / f"{video['name']}.y4m"
    mp4_path = OUTPUT_DIR / f"{video['name']}.mp4"

    if mp4_path.exists():
        logger.info("Already exists: %s", mp4_path.name)
        return True

    logger.info("Downloading: %s (%s)", video["name"], video["desc"])
    try:
        import httpx
        with httpx.stream("GET", video["url"], timeout=60, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(y4m_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)

        # Convert Y4M to MP4
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(y4m_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(mp4_path)],
            capture_output=True, timeout=30,
        )
        y4m_path.unlink(missing_ok=True)

        if mp4_path.exists():
            size_mb = mp4_path.stat().st_size / 1e6
            logger.info("  OK: %s (%.1f MB)", mp4_path.name, size_mb)
            return True
        return False
    except Exception as exc:
        logger.warning("  FAILED: %s — %s", video["name"], exc)
        y4m_path.unlink(missing_ok=True)
        return False


def main():
    logger.info("Downloading test videos to: %s", OUTPUT_DIR)
    logger.info("")

    success = 0
    total = 0

    # Xiph sequences (always available, small, reliable)
    for v in XIPH_VIDEOS:
        total += 1
        if download_xiph(v):
            success += 1

    # YouTube videos
    for v in YOUTUBE_VIDEOS:
        total += 1
        if download_youtube(v):
            success += 1

    # Direct downloads
    for v in TEST_VIDEOS:
        if v.get("direct"):
            total += 1
            if download_direct(v):
                success += 1

    logger.info("")
    logger.info("Downloaded %d/%d test videos", success, total)
    logger.info("Location: %s", OUTPUT_DIR)

    # List all downloaded
    videos = sorted(OUTPUT_DIR.glob("*.mp4"))
    if videos:
        logger.info("")
        total_mb = sum(v.stat().st_size for v in videos) / 1e6
        for v in videos:
            logger.info("  %s (%.1f MB)", v.name, v.stat().st_size / 1e6)
        logger.info("  Total: %.1f MB", total_mb)


if __name__ == "__main__":
    main()
