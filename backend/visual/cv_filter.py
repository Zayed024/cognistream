"""
CogniStream — Fast CV Pre-Filter

Lightweight object detection that runs BEFORE the slow VLM. Frames with
no interesting objects skip VLM inference entirely. Frames with objects
get tagged + passed through with bounding boxes for SoM prompting.

Borrowed from NVIDIA Metropolis VSS 3's two-stage architecture:
    Fast CV (15 FPS) → Slow VLM (only on triggers)

This is the single biggest optimization for live video — most frames
in a video are uninteresting, and we waste VLM cycles on them.

Three backends, in order of preference:
    1. NVIDIA Grounding DINO (cloud, best quality, needs API key)
    2. Ultralytics YOLO (local, fast, optional pip install)
    3. None (pre-filter disabled, every frame goes to VLM)

Usage:
    from backend.visual.cv_filter import cv_filter

    detections = cv_filter.detect("path/to/frame.jpg")
    if detections:
        # Has interesting objects → run VLM
        run_vlm(frame)
    else:
        # Skip VLM
        pass
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result."""
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    track_id: int = -1  # -1 = untracked


# Default labels considered "interesting" — frames with these go to VLM.
# Empty list = all detections trigger VLM.
INTERESTING_LABELS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "horse", "cow", "sheep",
    "backpack", "handbag", "suitcase",
    "knife", "scissors", "bottle",
    "chair", "couch", "bed", "dining table",
    "tv", "laptop", "cell phone",
}


class CVFilter:
    """Fast pre-filter using NVIDIA Grounding DINO or local YOLO."""

    # If the cloud Grounding DINO call fails this many times in a row, stop
    # trying for the rest of the process and use the local YOLO backend instead.
    # Avoids burning quota on a broken/throttled endpoint.
    _NVIDIA_FAILURE_THRESHOLD = 3

    def __init__(self):
        self._yolo_model = None
        self._yolo_tried = False
        self._nvidia_failures = 0
        self._nvidia_disabled = False

    @property
    def available(self) -> bool:
        """True if at least one backend is available."""
        if self._try_load_yolo() is not None:
            return True
        from backend.providers.nvidia import nvidia
        return nvidia.available and not self._nvidia_disabled

    def detect(
        self,
        image_path: str,
        labels: Optional[list[str]] = None,
        threshold: float = 0.3,
    ) -> list[Detection]:
        """Detect objects in an image.

        Args:
            image_path: Path to the image file.
            labels: Optional list of labels to detect (used by Grounding DINO).
                    If None, uses INTERESTING_LABELS.
            threshold: Minimum confidence (0-1).

        Returns:
            List of Detection objects.
        """
        if labels is None:
            labels = list(INTERESTING_LABELS)

        # Prefer local YOLO when available — no network round-trip, no
        # quota, ~30 ms per frame on CPU. Cloud Grounding DINO is only
        # used as a fallback when ultralytics isn't installed.
        if self._try_load_yolo() is not None:
            return self._detect_yolo(image_path, threshold)

        # Cloud fallback. Disabled after a streak of failures so we don't
        # waste HTTP calls on a broken/throttled endpoint.
        from backend.providers.nvidia import nvidia
        if nvidia.available and not self._nvidia_disabled:
            results = self._detect_nvidia(image_path, labels, threshold)
            if results is not None:
                self._nvidia_failures = 0
                return results
            self._nvidia_failures += 1
            if self._nvidia_failures >= self._NVIDIA_FAILURE_THRESHOLD:
                logger.warning(
                    "NV-Grounding-DINO failed %d times in a row — disabling "
                    "cloud CV pre-filter for the rest of this session.",
                    self._nvidia_failures,
                )
                self._nvidia_disabled = True

        return []

    def is_interesting(self, image_path: str, threshold: float = 0.3) -> tuple[bool, list[Detection]]:
        """Quick check: does this frame contain interesting objects?

        Returns:
            (interesting, detections) — interesting=True means VLM should run.
        """
        detections = self.detect(image_path, threshold=threshold)
        if not detections:
            return False, []

        # Filter to interesting labels
        interesting = [
            d for d in detections
            if d.label.lower() in INTERESTING_LABELS
        ]
        return len(interesting) > 0, detections

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NVIDIA Grounding DINO backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _detect_nvidia(
        self, image_path: str, labels: list[str], threshold: float
    ) -> list[Detection] | None:
        """Returns None on failure (so the caller can fall back to YOLO),
        empty list when the call succeeded but found nothing."""
        from backend.providers.nvidia import nvidia
        try:
            results = nvidia.detect_objects(image_path, labels, threshold=threshold)
            if results is None:
                return None
            return [
                Detection(
                    label=str(r.get("label", "object")),
                    confidence=float(r.get("confidence", 0)),
                    bbox=tuple(r.get("bbox", [0, 0, 0, 0])),
                )
                for r in results
            ]
        except Exception as exc:
            logger.debug("NVIDIA Grounding DINO failed: %s", exc)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Local YOLO backend (via opencv DNN or ultralytics if installed)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _try_load_yolo(self):
        """Lazy-load ultralytics YOLO if installed."""
        if self._yolo_tried:
            return self._yolo_model
        self._yolo_tried = True
        try:
            from ultralytics import YOLO
            # YOLOv8n is ~6 MB, runs fast on CPU
            self._yolo_model = YOLO("yolov8n.pt")
            logger.info("Loaded ultralytics YOLOv8n for CV pre-filter")
        except ImportError:
            logger.debug(
                "ultralytics not installed — CV pre-filter will use NVIDIA cloud only. "
                "Install with: pip install ultralytics"
            )
            self._yolo_model = None
        except Exception as exc:
            logger.warning("YOLO load failed: %s", exc)
            self._yolo_model = None
        return self._yolo_model

    def _detect_yolo(self, image_path: str, threshold: float) -> list[Detection]:
        model = self._try_load_yolo()
        if model is None:
            return []

        try:
            results = model(image_path, conf=threshold, verbose=False)
            if not results:
                return []

            detections = []
            r = results[0]
            names = r.names if hasattr(r, "names") else {}

            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=tuple(xyxy),
                ))
            return detections
        except Exception as exc:
            logger.debug("YOLO detection failed: %s", exc)
            return []


# Module-level singleton
cv_filter = CVFilter()
