"""Tests for backend.ingestion.shot_detector — shot boundary detection."""

import pytest
import numpy as np
import cv2

from backend.db.models import ShotSegment, VideoMeta, VideoStatus
from backend.ingestion.shot_detector import ShotDetector


class TestShotDetectorBoundaries:
    def test_detect_returns_segments(self, sample_video_meta):
        """The synthetic video has a colour change at frame 45 — should detect it."""
        detector = ShotDetector(threshold=0.3, stride=1)
        segments = detector.detect(sample_video_meta)

        assert len(segments) >= 1
        # All segments should cover the full video
        assert segments[0].start_frame == 0
        assert segments[-1].end_frame == sample_video_meta.total_frames - 1

    def test_segment_fields_populated(self, sample_video_meta):
        detector = ShotDetector(threshold=0.3, stride=1)
        segments = detector.detect(sample_video_meta)

        for seg in segments:
            assert seg.start_frame >= 0
            assert seg.end_frame >= seg.start_frame
            assert seg.start_time >= 0.0
            assert seg.end_time >= seg.start_time
            assert seg.frame_count > 0

    def test_single_scene_video(self, tmp_path):
        """A uniform video should produce exactly 1 segment."""
        video_path = tmp_path / "uniform.mp4"
        fps = 30.0
        w, h = 160, 120
        total = 60

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        for _ in range(total):
            writer.write(np.full((h, w, 3), (128, 128, 128), dtype=np.uint8))
        writer.release()

        meta = VideoMeta(id="uni", filename="uniform.mp4", file_path=str(video_path),
                         duration_sec=total / fps, fps=fps, width=w, height=h,
                         total_frames=total, status=VideoStatus.UPLOADED, created_at="")

        detector = ShotDetector(threshold=0.45, stride=1)
        segments = detector.detect(meta)
        assert len(segments) == 1


class TestMergeShortSegments:
    def test_micro_segments_merged(self):
        detector = ShotDetector(min_segment_frames=20)
        segments = [
            ShotSegment(0, 0, 49, 0.0, 1.63, 50),
            ShotSegment(1, 50, 55, 1.67, 1.83, 6),   # micro-segment
            ShotSegment(2, 56, 99, 1.87, 3.3, 44),
        ]
        merged = detector._merge_short_segments(segments)
        assert len(merged) == 2
        # First segment should absorb the micro-segment
        assert merged[0].end_frame == 55

    def test_no_merge_needed(self):
        detector = ShotDetector(min_segment_frames=5)
        segments = [
            ShotSegment(0, 0, 49, 0.0, 1.63, 50),
            ShotSegment(1, 50, 99, 1.67, 3.3, 50),
        ]
        merged = detector._merge_short_segments(segments)
        assert len(merged) == 2

    def test_empty_segments(self):
        detector = ShotDetector()
        assert detector._merge_short_segments([]) == []

    def test_reindexes_after_merge(self):
        detector = ShotDetector(min_segment_frames=20)
        segments = [
            ShotSegment(0, 0, 49, 0.0, 1.63, 50),
            ShotSegment(1, 50, 55, 1.67, 1.83, 6),
            ShotSegment(2, 56, 99, 1.87, 3.3, 44),
        ]
        merged = detector._merge_short_segments(segments)
        for i, seg in enumerate(merged):
            assert seg.segment_index == i
