"""Tests for backend.ingestion.frame_sampler — adaptive keyframe extraction."""

import pytest

from backend.db.models import ShotSegment
from backend.ingestion.frame_sampler import FrameSampler


class TestPickFrameIndices:
    def test_single_frame_picks_midpoint(self):
        seg = ShotSegment(0, 0, 100, 0.0, 3.33, 101)
        indices = FrameSampler._pick_frame_indices(seg, 1)
        assert indices == [50]

    def test_two_frames_picks_endpoints(self):
        seg = ShotSegment(0, 0, 100, 0.0, 3.33, 101)
        indices = FrameSampler._pick_frame_indices(seg, 2)
        assert indices == [0, 100]

    def test_multiple_frames_evenly_spaced(self):
        seg = ShotSegment(0, 0, 100, 0.0, 3.33, 101)
        indices = FrameSampler._pick_frame_indices(seg, 5)
        assert len(indices) == 5
        assert indices[0] == 0
        assert indices[-1] == 100
        # All should be monotonically increasing
        assert indices == sorted(indices)

    def test_zero_frames(self):
        seg = ShotSegment(0, 0, 100, 0.0, 3.33, 101)
        assert FrameSampler._pick_frame_indices(seg, 0) == []

    def test_no_duplicates(self):
        seg = ShotSegment(0, 10, 12, 0.33, 0.4, 3)
        indices = FrameSampler._pick_frame_indices(seg, 5)
        assert len(indices) == len(set(indices))


class TestAllocateBudget:
    def test_proportional_allocation(self):
        sampler = FrameSampler(max_per_video=100)
        segs = [
            ShotSegment(0, 0, 299, 0.0, 10.0, 300),
            ShotSegment(1, 300, 599, 10.0, 20.0, 300),
        ]
        alloc = sampler._allocate_budget(
            type("M", (), {"total_frames": 600, "fps": 30.0})(), segs
        )
        # Equal segments should get roughly equal allocation
        assert abs(alloc[0] - alloc[1]) <= 1

    def test_respects_global_cap(self):
        """When cap < segments, each segment gets at least 1 — total = num segments."""
        sampler = FrameSampler(max_per_video=50)
        segs = [
            ShotSegment(i, i * 1000, (i + 1) * 1000 - 1, 0, 0, 1000)
            for i in range(5)
        ]
        alloc = sampler._allocate_budget(
            type("M", (), {"total_frames": 5000, "fps": 30.0})(), segs
        )
        assert sum(alloc) <= 50

    def test_at_least_one_per_segment(self):
        sampler = FrameSampler(max_per_video=5)
        segs = [
            ShotSegment(0, 0, 99, 0, 3.3, 100),
            ShotSegment(1, 100, 199, 3.3, 6.6, 100),
            ShotSegment(2, 200, 299, 6.6, 10.0, 100),
        ]
        alloc = sampler._allocate_budget(
            type("M", (), {"total_frames": 300, "fps": 30.0})(), segs
        )
        assert all(a >= 1 for a in alloc)


class TestSampleIntegration:
    def test_sample_produces_keyframes(self, sample_video_meta, sample_shot_segments, tmp_path):
        sampler = FrameSampler(frame_dir=tmp_path, max_per_video=10)
        keyframes = sampler.sample(sample_video_meta, sample_shot_segments)

        assert len(keyframes) > 0
        assert len(keyframes) <= 10
        for kf in keyframes:
            assert kf.video_id == "test123"
            assert kf.timestamp >= 0

    def test_sample_empty_segments(self, sample_video_meta, tmp_path):
        sampler = FrameSampler(frame_dir=tmp_path)
        assert sampler.sample(sample_video_meta, []) == []
