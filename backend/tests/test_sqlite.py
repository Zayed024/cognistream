"""Tests for backend.db.sqlite — SQLite persistence layer."""

import pytest

from backend.db.models import VideoMeta, VideoStatus
from backend.db.sqlite import SQLiteDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory-like SQLite DB in a temp directory."""
    db_path = tmp_path / "test.db"
    return SQLiteDB(db_path=db_path)


@pytest.fixture
def video_meta():
    return VideoMeta(
        id="vid001",
        filename="lecture.mp4",
        file_path="/tmp/videos/vid001.mp4",
        duration_sec=120.5,
        fps=30.0,
        width=1920,
        height=1080,
        total_frames=3615,
        status=VideoStatus.UPLOADED,
        created_at="2026-01-01T00:00:00Z",
    )


class TestSQLiteDB:
    def test_save_and_get(self, db, video_meta):
        db.save_video(video_meta)
        retrieved = db.get_video("vid001")
        assert retrieved is not None
        assert retrieved.id == "vid001"
        assert retrieved.filename == "lecture.mp4"
        assert retrieved.duration_sec == 120.5
        assert retrieved.status == VideoStatus.UPLOADED

    def test_get_nonexistent(self, db):
        assert db.get_video("nonexistent") is None

    def test_list_videos(self, db, video_meta):
        db.save_video(video_meta)
        videos = db.list_videos()
        assert len(videos) == 1
        assert videos[0].id == "vid001"

    def test_list_videos_empty(self, db):
        assert db.list_videos() == []

    def test_update_status(self, db, video_meta):
        db.save_video(video_meta)
        db.update_status("vid001", VideoStatus.PROCESSING)
        updated = db.get_video("vid001")
        assert updated.status == VideoStatus.PROCESSING

    def test_update_status_with_error(self, db, video_meta):
        db.save_video(video_meta)
        db.update_status("vid001", VideoStatus.FAILED, error_message="Something broke")
        updated = db.get_video("vid001")
        assert updated.status == VideoStatus.FAILED
        assert updated.error_message == "Something broke"

    def test_delete_video(self, db, video_meta):
        db.save_video(video_meta)
        db.delete_video("vid001")
        assert db.get_video("vid001") is None

    def test_segment_count_zero(self, db, video_meta):
        db.save_video(video_meta)
        assert db.segment_count("vid001") == 0

    def test_event_count_zero(self, db, video_meta):
        db.save_video(video_meta)
        assert db.event_count("vid001") == 0

    def test_save_video_upsert(self, db, video_meta):
        """Saving the same video twice should update, not duplicate."""
        db.save_video(video_meta)
        video_meta.status = VideoStatus.PROCESSED
        db.save_video(video_meta)
        videos = db.list_videos()
        assert len(videos) == 1
        assert videos[0].status == VideoStatus.PROCESSED

    def test_save_and_get_latest_benchmark(self, db, video_meta):
        db.save_video(video_meta)

        db.save_benchmark_run(
            {
                "id": "run1",
                "video_id": "vid001",
                "captured_at": "2026-01-01T00:00:00Z",
                "success": True,
                "elapsed_sec": 12.4,
                "segments_stored": 11,
                "events_detected": 2,
                "stage_timings": {"vlm_sec": 4.1},
                "quality_metrics": {"keyframes_kept": 24.0},
                "warnings": [],
                "errors": [],
            }
        )

        latest = db.get_latest_benchmark("vid001")
        assert latest is not None
        assert latest["id"] == "run1"
        assert latest["success"] is True
        assert latest["stage_timings"]["vlm_sec"] == 4.1

    def test_list_benchmark_runs_ordered_desc(self, db, video_meta):
        db.save_video(video_meta)

        db.save_benchmark_run(
            {
                "id": "run_old",
                "video_id": "vid001",
                "captured_at": "2026-01-01T00:00:00Z",
                "success": True,
                "elapsed_sec": 14.0,
                "segments_stored": 10,
                "events_detected": 1,
                "stage_timings": {},
                "quality_metrics": {},
                "warnings": [],
                "errors": [],
            }
        )
        db.save_benchmark_run(
            {
                "id": "run_new",
                "video_id": "vid001",
                "captured_at": "2026-01-01T01:00:00Z",
                "success": True,
                "elapsed_sec": 9.0,
                "segments_stored": 13,
                "events_detected": 3,
                "stage_timings": {},
                "quality_metrics": {},
                "warnings": [],
                "errors": [],
            }
        )

        runs = db.list_benchmark_runs("vid001", limit=5)
        assert len(runs) == 2
        assert runs[0]["id"] == "run_new"
        assert runs[1]["id"] == "run_old"

    def test_benchmark_deleted_with_video(self, db, video_meta):
        db.save_video(video_meta)
        db.save_benchmark_run(
            {
                "id": "run1",
                "video_id": "vid001",
                "captured_at": "2026-01-01T00:00:00Z",
                "success": True,
                "elapsed_sec": 12.0,
                "segments_stored": 1,
                "events_detected": 0,
                "stage_timings": {},
                "quality_metrics": {},
                "warnings": [],
                "errors": [],
            }
        )

        db.delete_video("vid001")
        assert db.get_latest_benchmark("vid001") is None
