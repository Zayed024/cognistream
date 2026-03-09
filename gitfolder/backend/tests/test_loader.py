"""Tests for backend.ingestion.loader — video loading and validation."""

import pytest
import numpy as np
import cv2

from backend.db.models import VideoStatus
from backend.ingestion.loader import VideoLoader, VideoLoadError


@pytest.fixture
def loader(tmp_path):
    return VideoLoader(video_dir=tmp_path / "videos")


def _create_video(path, fps=30.0, width=160, height=120, frames=30):
    """Write a tiny synthetic video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


class TestVideoLoader:
    def test_load_valid_video(self, loader, tmp_path):
        video_path = tmp_path / "test.mp4"
        _create_video(video_path)

        meta = loader.load(video_path)
        assert meta.status == VideoStatus.UPLOADED
        assert meta.filename == "test.mp4"
        assert meta.fps > 0
        assert meta.width == 160
        assert meta.height == 120
        assert meta.total_frames == 30

    def test_load_nonexistent_file(self, loader):
        with pytest.raises(Exception, match="not found|File not found"):
            loader.load("/nonexistent/video.mp4")

    def test_load_unsupported_format(self, loader, tmp_path):
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not a video")
        with pytest.raises(Exception, match="Unsupported format"):
            loader.load(bad_file)

    def test_load_assigns_uuid(self, loader, tmp_path):
        video_path = tmp_path / "test.mp4"
        _create_video(video_path)
        meta = loader.load(video_path)
        assert len(meta.id) == 32  # hex UUID

    def test_load_copies_to_storage(self, loader, tmp_path):
        video_path = tmp_path / "test.mp4"
        _create_video(video_path)
        meta = loader.load(video_path)
        # The file should exist at the new path
        from pathlib import Path
        assert Path(meta.file_path).exists()
        assert meta.file_path != str(video_path)

    def test_load_from_bytes(self, loader, tmp_path):
        video_path = tmp_path / "src.mp4"
        _create_video(video_path)
        data = video_path.read_bytes()
        meta = loader.load_from_bytes(data, "upload.mp4")
        assert meta.status == VideoStatus.UPLOADED

    def test_load_from_bytes_bad_extension(self, loader):
        with pytest.raises(Exception, match="Unsupported format"):
            loader.load_from_bytes(b"data", "upload.xyz")
