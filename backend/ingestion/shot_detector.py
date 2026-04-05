"""
CogniStream — Shot Boundary Detector

Segments a video into shots (contiguous scenes) by measuring HSV histogram
correlation between consecutive frames.  A sharp drop in correlation signals
a scene change.

Algorithm:
    1. Sample every Nth frame (stride) to keep computation manageable.
    2. Convert each frame to HSV and compute a normalised colour histogram.
    3. Compare consecutive histograms with cv2.HISTCMP_CORREL.
    4. Mark a shot boundary wherever correlation < SHOT_THRESHOLD.
    5. Merge micro-segments (< MIN_SEGMENT_FRAMES) into neighbours.

Usage:
    detector = ShotDetector()
    segments = detector.detect(video_meta)
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from backend.config import MIN_SEGMENT_FRAMES, SHOT_THRESHOLD, resolve_shot_detection_workers
from backend.db.models import ShotSegment, VideoMeta

logger = logging.getLogger(__name__)

# Histogram parameters: 50 hue bins × 60 saturation bins
_H_BINS = 50
_S_BINS = 60
_HIST_SIZE = [_H_BINS, _S_BINS]
_H_RANGE = [0, 180]
_S_RANGE = [0, 256]
_HIST_RANGES = _H_RANGE + _S_RANGE
_HIST_CHANNELS = [0, 1]  # H and S channels of HSV


class ShotDetector:
    """Detect shot boundaries via histogram correlation."""

    def __init__(
        self,
        threshold: float | None = None,
        min_segment_frames: int | None = None,
        stride: int = 3,
    ):
        """
        Args:
            threshold: Correlation below this value triggers a boundary.
                       Lower = fewer cuts detected.  Default from config.
            min_segment_frames: Segments shorter than this are merged.
            stride: Analyse every *stride*-th frame to reduce I/O.
                    Boundaries snap to actual frame numbers.
        """
        self.threshold = threshold if threshold is not None else SHOT_THRESHOLD
        self.min_segment_frames = (
            min_segment_frames
            if min_segment_frames is not None
            else MIN_SEGMENT_FRAMES
        )
        self.stride = max(1, stride)

    # ── public API ──────────────────────────────────────────────

    def detect(self, meta: VideoMeta) -> list[ShotSegment]:
        """Detect shot boundaries for the video described by *meta*.

        Uses parallel chunk processing when SHOT_DETECTION_WORKERS > 1
        and the video has enough frames to justify splitting.

        Returns:
            Sorted list of :class:`ShotSegment` covering the entire video.
        """
        workers = resolve_shot_detection_workers()
        logger.info(
            "Starting shot detection: %s (%d frames, stride=%d, workers=%d)",
            meta.filename,
            meta.total_frames,
            self.stride,
            workers,
        )

        if workers > 1 and meta.total_frames > 500:
            boundaries = self._compute_boundaries_parallel(meta, workers)
        else:
            boundaries = self._compute_boundaries(meta)

        segments = self._boundaries_to_segments(boundaries, meta)
        segments = self._merge_short_segments(segments)

        logger.info(
            "Shot detection complete: %d segments from %d frames",
            len(segments),
            meta.total_frames,
        )
        return segments

    # ── internals ───────────────────────────────────────────────

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute a normalised 2-D HS histogram for a single frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], _HIST_CHANNELS, None, _HIST_SIZE, _HIST_RANGES)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _compute_boundaries_parallel(
        self, meta: VideoMeta, workers: int
    ) -> list[int]:
        """Split the video into chunks and detect boundaries concurrently.

        Each worker opens its own cv2.VideoCapture (thread-safe since they
        are separate file handles).  Chunk boundaries are stitched together
        and de-duplicated.
        """
        from concurrent.futures import ThreadPoolExecutor

        total = meta.total_frames
        chunk_size = max(100, total // workers)
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(start + chunk_size, total)
            chunks.append((start, end))
            start = end

        logger.debug("Parallel shot detection: %d chunks across %d workers", len(chunks), workers)

        def _detect_chunk(frame_range: tuple[int, int]) -> list[int]:
            """Detect boundaries within a frame range using a private VideoCapture."""
            start_frame, end_frame = frame_range
            cap = cv2.VideoCapture(meta.file_path)
            if not cap.isOpened():
                return []

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            chunk_boundaries: list[int] = []
            prev_hist: np.ndarray | None = None
            frame_idx = start_frame

            try:
                while frame_idx < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % self.stride == 0:
                        hist = self._compute_histogram(frame)
                        if prev_hist is not None:
                            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                            if corr < self.threshold:
                                chunk_boundaries.append(frame_idx)
                        prev_hist = hist
                    frame_idx += 1
            finally:
                cap.release()

            return chunk_boundaries

        all_boundaries = [0]  # video always starts with a boundary
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shot") as pool:
            results = pool.map(_detect_chunk, chunks)
            for chunk_result in results:
                all_boundaries.extend(chunk_result)

        # De-duplicate and sort
        return sorted(set(all_boundaries))

    def _compute_boundaries(self, meta: VideoMeta) -> list[int]:
        """Walk through the video, returning frame numbers where shots change."""
        cap = cv2.VideoCapture(meta.file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {meta.file_path}")

        boundaries: list[int] = [0]  # video always starts with a boundary
        prev_hist: np.ndarray | None = None
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.stride == 0:
                    hist = self._compute_histogram(frame)

                    if prev_hist is not None:
                        corr = cv2.compareHist(
                            prev_hist, hist, cv2.HISTCMP_CORREL
                        )
                        if corr < self.threshold:
                            boundaries.append(frame_idx)
                            logger.debug(
                                "Shot boundary at frame %d (corr=%.3f)",
                                frame_idx,
                                corr,
                            )

                    prev_hist = hist

                frame_idx += 1
        finally:
            cap.release()

        return boundaries

    def _boundaries_to_segments(
        self, boundaries: list[int], meta: VideoMeta
    ) -> list[ShotSegment]:
        """Convert a list of boundary frame numbers into ShotSegment objects."""
        fps = meta.fps or 30.0
        total = meta.total_frames
        segments: list[ShotSegment] = []

        for i, start_frame in enumerate(boundaries):
            end_frame = (
                boundaries[i + 1] - 1 if i + 1 < len(boundaries) else total - 1
            )
            if end_frame < start_frame:
                continue

            segments.append(
                ShotSegment(
                    segment_index=i,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=round(start_frame / fps, 3),
                    end_time=round(end_frame / fps, 3),
                    frame_count=end_frame - start_frame + 1,
                )
            )

        return segments

    def _merge_short_segments(
        self, segments: list[ShotSegment]
    ) -> list[ShotSegment]:
        """Merge segments shorter than *min_segment_frames* into the previous one.

        This suppresses spurious boundaries caused by camera flashes,
        compression artefacts, or brief overlays.
        """
        if not segments:
            return segments

        merged: list[ShotSegment] = [segments[0]]

        for seg in segments[1:]:
            if seg.frame_count < self.min_segment_frames:
                # Absorb into the previous segment
                prev = merged[-1]
                prev.end_frame = seg.end_frame
                prev.end_time = seg.end_time
                prev.frame_count = prev.end_frame - prev.start_frame + 1
                logger.debug(
                    "Merged micro-segment %d (%d frames) into segment %d",
                    seg.segment_index,
                    seg.frame_count,
                    prev.segment_index,
                )
            else:
                merged.append(seg)

        # Re-index after merging
        for i, seg in enumerate(merged):
            seg.segment_index = i

        if len(merged) < len(segments):
            logger.info(
                "Merged %d micro-segments → %d final segments",
                len(segments),
                len(merged),
            )

        return merged
