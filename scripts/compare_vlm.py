"""
3-way VLM comparison: base moondream (Ollama) vs NVIDIA cloud.

Runs the combined prompt on benchmark keyframes and saves the outputs
side-by-side for manual review.

Usage:
    python scripts/compare_vlm.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.visual.vlm_runner import OllamaClient
from backend.visual.caption_processor import PromptLibrary
from backend.providers.nvidia import nvidia

logging.basicConfig(level=logging.WARNING)

FRAMES_DIR = PROJECT_ROOT / "data" / "frames" / "benchmark_samples"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "vlm_comparison.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def is_empty(response: str) -> bool:
    """Check if a caption response is effectively empty (failed generation)."""
    if not response:
        return True
    normalized = response.strip()
    return len(normalized) < 20


def is_structured(response: str) -> bool:
    """Check if a response follows the SCENE/OBJECTS/ACTIVITY/ANOMALY format."""
    if not response:
        return False
    upper = response.upper()
    # Must have at least 3 of the 4 fields
    fields = sum(1 for f in ["SCENE:", "OBJECTS:", "ACTIVITY:", "ANOMALY:"] if f in upper)
    return fields >= 3


def main():
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        print(f"No frames in {FRAMES_DIR}")
        sys.exit(1)

    print(f"Comparing VLMs on {len(frames)} frames")
    print()

    ollama = OllamaClient()
    if not ollama.is_available():
        print("Ollama not running. Start with: ollama serve")
        sys.exit(1)
    print(f"Base VLM: Ollama {ollama.model}")
    print(f"Cloud VLM: NVIDIA ({'available' if nvidia.available else 'DISABLED'})")
    print()

    prompt = PromptLibrary.combined_prompt()
    results = []
    base_empty = 0
    base_structured = 0
    nvidia_empty = 0
    nvidia_structured = 0

    for i, frame in enumerate(frames):
        print(f"[{i+1}/{len(frames)}] {frame.name}")

        # Base moondream via Ollama
        t0 = time.monotonic()
        try:
            base_resp = ollama.generate(prompt, str(frame))
        except Exception as exc:
            base_resp = f"ERROR: {exc}"
        base_time = time.monotonic() - t0
        base_is_empty = is_empty(base_resp)
        base_is_structured = is_structured(base_resp)
        if base_is_empty:
            base_empty += 1
        if base_is_structured:
            base_structured += 1

        # NVIDIA cloud
        nv_resp = "N/A (NVIDIA disabled)"
        nv_time = 0
        nv_is_structured = False
        if nvidia.available:
            t0 = time.monotonic()
            nv_resp = nvidia.caption_image(str(frame), prompt) or ""
            nv_time = time.monotonic() - t0
            if is_empty(nv_resp):
                nvidia_empty += 1
            nv_is_structured = is_structured(nv_resp)
            if nv_is_structured:
                nvidia_structured += 1

        print(f"  Base ({base_time:.1f}s): {'[EMPTY]' if base_is_empty else base_resp[:120]}")
        if nvidia.available:
            print(f"  NVIDIA ({nv_time:.1f}s): {nv_resp[:120]}")
        print()

        results.append({
            "frame": frame.name,
            "base_moondream": {
                "time_sec": round(base_time, 2),
                "empty": base_is_empty,
                "structured": base_is_structured,
                "response": base_resp,
            },
            "nvidia_llama_11b": {
                "time_sec": round(nv_time, 2),
                "empty": is_empty(nv_resp) if nvidia.available else None,
                "structured": nv_is_structured if nvidia.available else None,
                "response": nv_resp,
            },
        })

    summary = {
        "total_frames": len(frames),
        "base_empty": base_empty,
        "base_empty_pct": round(100 * base_empty / len(frames), 1),
        "base_structured": base_structured,
        "base_structured_pct": round(100 * base_structured / len(frames), 1),
        "nvidia_empty": nvidia_empty if nvidia.available else None,
        "nvidia_empty_pct": round(100 * nvidia_empty / len(frames), 1) if nvidia.available else None,
        "nvidia_structured": nvidia_structured if nvidia.available else None,
        "nvidia_structured_pct": round(100 * nvidia_structured / len(frames), 1) if nvidia.available else None,
    }

    output = {"summary": summary, "results": results}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Frames tested:       {len(frames)}")
    print()
    print(f"  Base moondream (Ollama):")
    print(f"    Empty:             {base_empty}/{len(frames)} ({summary['base_empty_pct']}%)")
    print(f"    Structured format: {base_structured}/{len(frames)} ({summary['base_structured_pct']}%)")
    if nvidia.available:
        print()
        print(f"  NVIDIA Llama-3.2-11B (cloud):")
        print(f"    Empty:             {nvidia_empty}/{len(frames)} ({summary['nvidia_empty_pct']}%)")
        print(f"    Structured format: {nvidia_structured}/{len(frames)} ({summary['nvidia_structured_pct']}%)")
    print()
    print(f"  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
