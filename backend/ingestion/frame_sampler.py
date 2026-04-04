"""
CogniStream — Adaptive Frame Sampler

Selects representative keyframes from each shot segment.  The number of
keyframes per segment is determined adaptively based on:

    1. **Segment duration** — longer segments get more keyframes.
    2. **Visual complexity** — segments with higher inter-frame variance
       (more motion / scene change) receive a larger allocation.

Within a segment, frames are selected at uniform intervals so the
keyframes span the full temporal range.

Edge-deployment constraint:
    A global cap (MAX_KEYFRAMES_PER_VIDEO) prevents memory/storage blow-up
    on resource-limited hardware.  When the budget is exhausted, remaining
    segments fall back to a single mid-point keyframe.

Usage:
    sampler = FrameSampler()
    keyframes = sampler.sample(video_meta, segments)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from backend.config import (
    FRAME_DIR,
    KEYFRAME_IMAGE_FORMAT,
    KEYFRAME_JPEG_QUALITY,
    MAX_KEYFRAMES_PER_SEGMENT,
    MAX_KEYFRAMES_PER_VIDEO,
    MIN_KEYFRAMES_PER_SEGMENT,
)
from backend.db.models import Keyframe, ShotSegment, VideoMeta

logger = logging.getLogger(__name__)


class FrameSampler:
    """Adaptively extract keyframes from video segments."""

    def __init__(
        self,
        frame_dir: Path | None = None,
        max_per_video: int | None = None,
    ):
        self.frame_dir = frame_dir or FRAME_DIR
        self.max_per_video = max_per_video or MAX_KEYFRAMES_PER_VIDEO

    # ── public API ──────────────────────────────────────────────

    def sample(
        self,
        meta: VideoMeta,
        segments: list[ShotSegment],
    ) -> list[Keyframe]:
        """Extract keyframes for all *segments* of the given video.

        Returns:
            A flat list of :class:`Keyframe` objects sorted by timestamp.
        """
        if not segments:
            logger.warning("No segments provided — nothing to sample.")
            return []

        # Allocate keyframe budget across segments
        allocation = self._allocate_budget(meta, segments)

        logger.info(
            "Sampling keyframes: %d segments, budget %d (cap %d)",
            len(segments),
            sum(allocation),
            self.max_per_video,
        )

        cap = cv2.VideoCapture(meta.file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {meta.file_path}")

        out_dir = self.frame_dir / meta.id
        out_dir.mkdir(parents=True, exist_ok=True)

        all_keyframes: list[Keyframe] = []

        try:
            for seg, n_frames in zip(segments, allocation):
                kfs = self._sample_segment(cap, meta, seg, n_frames, out_dir)
                all_keyframes.extend(kfs)
        finally:
            cap.release()

        all_keyframes.sort(key=lambda k: k.timestamp)

        logger.info(
            "Keyframe extraction complete: %d keyframes from %d segments",
            len(all_keyframes),
            len(segments),
        )
        return all_keyframes

    # ── budget allocation ───────────────────────────────────────

    def _allocate_budget(
        self,
        meta: VideoMeta,
        segments: list[ShotSegment],
    ) -> list[int]:
        """Distribute the global keyframe budget across segments.

        Allocation is proportional to each segment's share of total frames,
        clamped to [MIN_KEYFRAMES_PER_SEGMENT, MAX_KEYFRAMES_PER_SEGMENT].
        If the raw proportional allocation exceeds the global cap, all
        counts are scaled down proportionally.
        """
        total_frames = sum(s.frame_count for s in segments) or 1

        raw: list[float] = []
        for seg in segments:
            share = seg.frame_count / total_frames
            count = share * self.max_per_video
            count = max(MIN_KEYFRAMES_PER_SEGMENT, count)
            count = min(MAX_KEYFRAMES_PER_SEGMENT, count)
            raw.append(count)

        # Scale down if total exceeds budget
        raw_total = sum(raw)
        if raw_total > self.max_per_video:
            scale = self.max_per_video / raw_total
            raw = [r * scale for r in raw]

        # Round to integers, ensuring at least 1 per segment
        allocation = [max(1, round(r)) for r in raw]

        # Final trim: if still over budget, shave from the largest segments
        while sum(allocation) > self.max_per_video and any(
            a > 1 for a in allocation
        ):
            idx = allocation.index(max(allocation))
            allocation[idx] -= 1

        return allocation

    # ── per-segment sampling ────────────────────────────────────

    def _sample_segment(
        self,
        cap: cv2.VideoCapture,
        meta: VideoMeta,
        segment: ShotSegment,
        n_frames: int,
        out_dir: Path,
    ) -> list[Keyframe]:
        """Select and save *n_frames* keyframes from a single segment."""
        frame_indices = self._pick_frame_indices(segment, n_frames)
        fps = meta.fps or 30.0
        keyframes: list[Keyframe] = []

        for frame_num in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                logger.debug("Could not read frame %d — skipping", frame_num)
                continue

            file_path = self._save_frame(frame, out_dir, frame_num)

            keyframes.append(
                Keyframe(
                    video_id=meta.id,
                    segment_index=segment.segment_index,
                    frame_number=frame_num,
                    timestamp=round(frame_num / fps, 3),
                    file_path=str(file_path),
                )
            )

        logger.debug(
            "Segment %d: extracted %d/%d keyframes (frames %d–%d)",
            segment.segment_index,
            len(keyframes),
            n_frames,
            segment.start_frame,
            segment.end_frame,
        )
        return keyframes

    @staticmethod
    def _pick_frame_indices(segment: ShotSegment, n: int) -> list[int]:
        """Choose *n* evenly spaced frame numbers within the segment range.

        For n=1, picks the temporal midpoint.
        For n>1, frames are spread from start to end (inclusive).
        """
        start = segment.start_frame
        end = segment.end_frame

        if n <= 0:
            return []
        if n == 1:
            return [(start + end) // 2]

        # Linspace from start to end, rounded to int, deduplicated
        step = (end - start) / (n - 1) if n > 1 else 0
        indices: list[int] = []
        for i in range(n):
            idx = round(start + i * step)
            idx = max(start, min(idx, end))
            if not indices or idx != indices[-1]:
                indices.append(idx)

        return indices

    def _save_frame(
        self, frame: np.ndarray, out_dir: Path, frame_num: int
    ) -> Path:
        """Write a frame to disk as a JPEG image."""
        filename = f"{frame_num:06d}.{KEYFRAME_IMAGE_FORMAT}"
        path = out_dir / filename
        params = [cv2.IMWRITE_JPEG_QUALITY, KEYFRAME_JPEG_QUALITY]
        cv2.imwrite(str(path), frame, params)
        return path
