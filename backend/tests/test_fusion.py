"""Tests for backend.fusion.multimodal_embedder — fusion logic (no model loading)."""

import pytest

from backend.db.models import FusedSegment, Keyframe, TranscriptSegment, VisualCaption
from backend.fusion.multimodal_embedder import MultimodalEmbedder


@pytest.fixture
def embedder():
    """Embedder instance — only tests fusion logic, not embed()."""
    return MultimodalEmbedder()


def _make_caption(timestamp, scene="A scene", objects=None, activity="activity"):
    kf = Keyframe(video_id="v1", segment_index=0, frame_number=int(timestamp * 30), timestamp=timestamp, file_path=f"/tmp/{timestamp}.jpg")
    return VisualCaption(keyframe=kf, scene_description=scene, objects=objects or [], activity=activity)


def _make_transcript(start, end, text):
    return TranscriptSegment(start_time=start, end_time=end, text=text)


class TestFusionEmptyInputs:
    def test_empty_both(self, embedder):
        assert embedder.fuse("v1", [], []) == []

    def test_captions_only(self, embedder):
        cap = _make_caption(5.0, scene="A street", objects=["car"])
        result = embedder.fuse("v1", [cap], [])
        assert len(result) == 1
        assert result[0].source_type == "visual"
        assert "car" in result[0].text

    def test_transcripts_only(self, embedder):
        tr = _make_transcript(2.0, 4.0, "Hello world")
        result = embedder.fuse("v1", [], [tr])
        assert len(result) == 1
        assert result[0].source_type == "audio"
        assert result[0].text == "Hello world"


class TestFusionMerging:
    def test_overlapping_caption_and_transcript(self, embedder):
        """Caption at t=5 with window ±2 should merge with transcript [4, 6]."""
        cap = _make_caption(5.0, scene="A car")
        tr = _make_transcript(4.0, 6.0, "The car is stopping")
        result = embedder.fuse("v1", [cap], [tr])

        fused = [r for r in result if r.source_type == "fused"]
        assert len(fused) == 1
        assert "[Speech:" in fused[0].text
        assert "The car is stopping" in fused[0].text

    def test_non_overlapping_produces_separate_segments(self, embedder):
        """Caption at t=5 should NOT merge with transcript [50, 52]."""
        cap = _make_caption(5.0, scene="A car")
        tr = _make_transcript(50.0, 52.0, "Later speech")
        result = embedder.fuse("v1", [cap], [tr])

        types = {r.source_type for r in result}
        assert "visual" in types
        assert "audio" in types
        assert "fused" not in types

    def test_transcript_assigned_to_nearest_keyframe(self, embedder):
        """A transcript overlapping two captions goes to the nearest one only."""
        cap1 = _make_caption(5.0, scene="Scene A")
        cap2 = _make_caption(6.5, scene="Scene B")
        tr = _make_transcript(4.5, 7.5, "Shared speech")

        result = embedder.fuse("v1", [cap1, cap2], [tr])

        # The transcript should appear in exactly one fused segment, not both
        fused = [r for r in result if r.source_type == "fused"]
        speech_segments = [r for r in result if "Shared speech" in r.text]
        assert len(speech_segments) == 1  # no duplication

    def test_all_segments_have_ids(self, embedder):
        cap = _make_caption(5.0)
        tr = _make_transcript(100.0, 102.0, "Audio only")
        result = embedder.fuse("v1", [cap], [tr])
        for seg in result:
            assert seg.id  # non-empty
            assert seg.video_id == "v1"

    def test_sorted_by_start_time(self, embedder):
        cap = _make_caption(10.0)
        tr = _make_transcript(1.0, 2.0, "Early speech")
        result = embedder.fuse("v1", [cap], [tr])
        times = [r.start_time for r in result]
        assert times == sorted(times)


class TestFusionTextContent:
    def test_visual_text_includes_objects(self, embedder):
        cap = _make_caption(5.0, scene="A park", objects=["dog", "frisbee"], activity="playing")
        result = embedder.fuse("v1", [cap], [])
        text = result[0].text
        assert "Objects: dog, frisbee" in text
        assert "Activity: playing" in text

    def test_anomaly_included_when_present(self, embedder):
        kf = Keyframe(video_id="v1", segment_index=0, frame_number=150, timestamp=5.0, file_path="/tmp/f.jpg")
        cap = VisualCaption(keyframe=kf, scene_description="Scene", anomaly="Fire detected")
        result = embedder.fuse("v1", [cap], [])
        assert "Anomaly: Fire detected" in result[0].text

    def test_frame_path_preserved(self, embedder):
        cap = _make_caption(5.0)
        result = embedder.fuse("v1", [cap], [])
        assert result[0].frame_path == "/tmp/5.0.jpg"


class TestOverlapsHelper:
    def test_overlapping(self):
        assert MultimodalEmbedder._overlaps(1, 5, 3, 7) is True

    def test_adjacent_not_overlapping(self):
        assert MultimodalEmbedder._overlaps(1, 3, 3, 5) is False

    def test_contained(self):
        assert MultimodalEmbedder._overlaps(1, 10, 3, 5) is True

    def test_no_overlap(self):
        assert MultimodalEmbedder._overlaps(1, 3, 5, 7) is False

    def test_same_interval(self):
        assert MultimodalEmbedder._overlaps(1, 5, 1, 5) is True
