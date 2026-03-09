"""Tests for backend.db.models — data classes and enums."""

from backend.db.models import (
    Event,
    FusedSegment,
    Keyframe,
    SearchResult,
    ShotSegment,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)


class TestVideoStatus:
    def test_enum_values(self):
        assert VideoStatus.UPLOADED.value == "UPLOADED"
        assert VideoStatus.PROCESSING.value == "PROCESSING"
        assert VideoStatus.PROCESSED.value == "PROCESSED"
        assert VideoStatus.FAILED.value == "FAILED"

    def test_from_string(self):
        assert VideoStatus("UPLOADED") == VideoStatus.UPLOADED


class TestVideoMeta:
    def test_defaults(self):
        m = VideoMeta(id="abc", filename="test.mp4", file_path="/tmp/test.mp4")
        assert m.status == VideoStatus.UPLOADED
        assert m.duration_sec == 0.0
        assert m.processed_at is None
        assert m.error_message is None


class TestFusedSegment:
    def test_default_embedding_is_none(self):
        seg = FusedSegment(id="s1", video_id="v1", start_time=0, end_time=1, text="hello")
        assert seg.embedding is None
        assert seg.source_type == "fused"
        assert seg.frame_path is None


class TestShotSegment:
    def test_creation(self):
        s = ShotSegment(segment_index=0, start_frame=0, end_frame=100, start_time=0.0, end_time=3.33, frame_count=101)
        assert s.frame_count == 101


class TestTranscriptSegment:
    def test_default_keywords(self):
        t = TranscriptSegment(start_time=0, end_time=1, text="hello world")
        assert t.keywords == []


class TestEvent:
    def test_creation(self):
        e = Event(id="e1", video_id="v1", event_type="car_arrival", start_time=0, end_time=5)
        assert e.entities == []
        assert e.description == ""


class TestSearchResult:
    def test_creation(self):
        r = SearchResult(video_id="v1", segment_id="s1", start_time=0, end_time=1, text="hello", source_type="fused", score=0.95)
        assert r.event_type is None
        assert r.frame_url is None
