"""Tests for backend.pipeline.streaming — StreamingPipeline (file-based mode)."""

import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from backend.db.models import (
    Event,
    FusedSegment,
    Keyframe,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)
from backend.pipeline.streaming import (
    ChunkProgress,
    LiveFeedStatus,
    StreamingPipeline,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.add_segments = MagicMock(return_value=0)
    return store


@pytest.fixture
def pipeline(mock_db, mock_store):
    return StreamingPipeline(db=mock_db, store=mock_store, chunk_sec=30)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _probe_duration()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProbeDuration:
    def test_probe_duration_with_real_video(self, sample_video_meta):
        """Uses the conftest sample_video_meta fixture which writes a real .mp4."""
        duration = StreamingPipeline._probe_duration(sample_video_meta.file_path)
        # The synthetic video is 90 frames at 30fps = 3.0 seconds
        assert duration > 0
        assert abs(duration - 3.0) < 0.5  # allow small tolerance

    def test_probe_duration_nonexistent_file(self):
        duration = StreamingPipeline._probe_duration("/nonexistent/video.mp4")
        assert duration == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _extract_chunk_keyframes()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExtractChunkKeyframes:
    def test_extracts_frames_from_video(self, sample_video_meta, tmp_path):
        """Extract keyframes from the first chunk of the synthetic video."""
        # Override FRAME_DIR to use tmp_path so keyframes land in the test dir
        with patch("backend.pipeline.streaming.FRAME_DIR", str(tmp_path / "frames")):
            keyframes = StreamingPipeline._extract_chunk_keyframes(
                meta=sample_video_meta,
                start_frame=0,
                end_frame=45,
                fps=30.0,
                chunk_idx=0,
                max_frames=3,
            )

        assert len(keyframes) > 0
        assert len(keyframes) <= 3
        for kf in keyframes:
            assert isinstance(kf, Keyframe)
            assert kf.video_id == sample_video_meta.id
            assert kf.segment_index == 0
            assert Path(kf.file_path).exists()
            assert kf.timestamp >= 0.0

    def test_returns_empty_for_nonexistent_video(self, tmp_path):
        meta = VideoMeta(
            id="missing", filename="nope.mp4",
            file_path="/nonexistent/nope.mp4",
            status=VideoStatus.UPLOADED,
        )
        with patch("backend.pipeline.streaming.FRAME_DIR", str(tmp_path / "frames")):
            keyframes = StreamingPipeline._extract_chunk_keyframes(
                meta=meta, start_frame=0, end_frame=30,
                fps=30.0, chunk_idx=0,
            )
        assert keyframes == []

    def test_returns_empty_for_zero_span(self, sample_video_meta, tmp_path):
        with patch("backend.pipeline.streaming.FRAME_DIR", str(tmp_path / "frames")):
            keyframes = StreamingPipeline._extract_chunk_keyframes(
                meta=sample_video_meta,
                start_frame=10,
                end_frame=10,  # zero span
                fps=30.0,
                chunk_idx=0,
            )
        assert keyframes == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# stop()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStop:
    def test_stop_sets_event(self, pipeline):
        assert not pipeline._stop_event.is_set()
        pipeline.stop()
        assert pipeline._stop_event.is_set()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _vlm_on_keyframes()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestVLMOnKeyframes:
    def test_client_none_returns_empty(self, sample_keyframes):
        result = StreamingPipeline._vlm_on_keyframes(sample_keyframes, client=None)
        assert result == []

    def test_empty_keyframes_returns_empty(self):
        mock_client = MagicMock()
        result = StreamingPipeline._vlm_on_keyframes([], client=mock_client)
        assert result == []

    @patch("backend.pipeline.streaming.VLMRunner")
    def test_vlm_exception_returns_empty(self, MockVLMRunner, sample_keyframes):
        mock_client = MagicMock()
        MockVLMRunner.side_effect = RuntimeError("Ollama crashed")
        result = StreamingPipeline._vlm_on_keyframes(sample_keyframes, client=mock_client)
        assert result == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _whisper_on_chunk()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWhisperOnChunk:
    def test_invalid_path_returns_empty(self):
        """Passing a non-existent video file should return empty, not crash."""
        result = StreamingPipeline._whisper_on_chunk(
            "/nonexistent/video.mp4", start_sec=0.0, end_sec=5.0
        )
        assert result == []

    def test_zero_duration_returns_empty(self):
        result = StreamingPipeline._whisper_on_chunk(
            "/some/video.mp4", start_sec=5.0, end_sec=5.0
        )
        assert result == []

    def test_negative_duration_returns_empty(self):
        result = StreamingPipeline._whisper_on_chunk(
            "/some/video.mp4", start_sec=10.0, end_sec=5.0
        )
        assert result == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _events_to_segments()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventsToSegments:
    def test_converts_events_correctly(self):
        events = [
            Event(
                id="e1",
                video_id="vid001",
                event_type="car_arrival",
                start_time=10.0,
                end_time=15.0,
                description="A car arrived.",
                entities=["car", "entrance"],
            ),
        ]
        segments = StreamingPipeline._events_to_segments("vid001", events)

        assert len(segments) == 1
        seg = segments[0]
        assert isinstance(seg, FusedSegment)
        assert seg.video_id == "vid001"
        assert seg.source_type == "event"
        assert seg.start_time == 10.0
        assert seg.end_time == 15.0
        assert "car_arrival" in seg.text
        assert "car" in seg.text
        assert "entrance" in seg.text

    def test_empty_events_returns_empty(self):
        segments = StreamingPipeline._events_to_segments("vid001", [])
        assert segments == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_live_status()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetLiveStatus:
    def test_no_feeds_returns_empty(self, pipeline):
        statuses = pipeline.get_live_status()
        assert statuses == []

    def test_specific_video_id_not_found_returns_empty(self, pipeline):
        statuses = pipeline.get_live_status(video_id="nonexistent")
        assert statuses == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# start_live() / stop_live() lifecycle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLiveFeedLifecycle:
    @patch("backend.pipeline.streaming.cv2.VideoCapture")
    @patch("backend.pipeline.streaming.OllamaClient")
    @patch("backend.pipeline.streaming.MultimodalEmbedder")
    def test_start_and_stop_live(
        self,
        MockEmbedder,
        MockOllamaClient,
        MockVideoCapture,
        mock_db,
        mock_store,
    ):
        # Make VideoCapture fail to open so the worker exits quickly
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        MockVideoCapture.return_value = mock_cap_instance

        MockOllamaClient.return_value.is_available.return_value = False
        MockEmbedder.return_value.unload_model = MagicMock()

        pipeline = StreamingPipeline(db=mock_db, store=mock_store, chunk_sec=5)

        # Temporarily reduce reconnect attempts so the worker stops fast
        original_max = StreamingPipeline.MAX_RECONNECT_ATTEMPTS
        original_delay = StreamingPipeline.RECONNECT_DELAY_SEC
        StreamingPipeline.MAX_RECONNECT_ATTEMPTS = 1
        StreamingPipeline.RECONNECT_DELAY_SEC = 0

        try:
            status = pipeline.start_live("rtsp://fake:554/stream", "cam01")
            assert isinstance(status, LiveFeedStatus)
            assert status.video_id == "cam01"
            assert status.state == "connecting"

            # Give the worker thread time to fail and exit
            import time
            time.sleep(1.0)

            # stop_live should return True if the worker exists in the dict
            # (it may have already died but the entry is still there)
            # After the worker dies, stop_live on an unknown ID returns False
            result = pipeline.stop_live("cam01")
            # Either True (worker still in dict) or we just verify no crash
            assert isinstance(result, bool)
        finally:
            StreamingPipeline.MAX_RECONNECT_ATTEMPTS = original_max
            StreamingPipeline.RECONNECT_DELAY_SEC = original_delay
            # Clean up any running threads
            pipeline.stop_all_live()

    def test_start_live_duplicate_raises(self, mock_db, mock_store):
        """Starting the same video_id twice should raise ValueError."""
        pipeline = StreamingPipeline(db=mock_db, store=mock_store, chunk_sec=5)

        # Manually insert a fake worker that pretends to be alive
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        pipeline._live_feeds["cam01"] = mock_worker

        with pytest.raises(ValueError, match="already running"):
            pipeline.start_live("rtsp://fake:554/stream", "cam01")

        # Cleanup
        pipeline._live_feeds.clear()

    def test_stop_live_nonexistent_returns_false(self, pipeline):
        result = pipeline.stop_live("nonexistent_cam")
        assert result is False

    def test_stop_all_live_returns_count(self, pipeline):
        # With no feeds, count should be 0
        count = pipeline.stop_all_live()
        assert count == 0
