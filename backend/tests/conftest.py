"""
Shared fixtures for CogniStream tests.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
import cv2

# Ensure the project root is on sys.path so `backend.*` imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.models import (
    FusedSegment,
    Keyframe,
    ShotSegment,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory."""
    return tmp_path


@pytest.fixture
def sample_video_meta(tmp_path):
    """Create a minimal synthetic video file and return its VideoMeta."""
    video_path = tmp_path / "test_video.mp4"
    fps = 30.0
    width, height = 320, 240
    total_frames = 90  # 3 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(total_frames):
        # Alternate between two distinct colours to create a shot boundary
        if i < 45:
            frame = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
        else:
            frame = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return VideoMeta(
        id="test123",
        filename="test_video.mp4",
        file_path=str(video_path),
        duration_sec=total_frames / fps,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        status=VideoStatus.UPLOADED,
        created_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_keyframes():
    """Return a list of Keyframe objects at known timestamps."""
    return [
        Keyframe(video_id="v1", segment_index=0, frame_number=30, timestamp=1.0, file_path="/tmp/f1.jpg"),
        Keyframe(video_id="v1", segment_index=0, frame_number=90, timestamp=3.0, file_path="/tmp/f2.jpg"),
        Keyframe(video_id="v1", segment_index=1, frame_number=150, timestamp=5.0, file_path="/tmp/f3.jpg"),
    ]


@pytest.fixture
def sample_captions(sample_keyframes):
    """Return VisualCaption objects for the sample keyframes."""
    return [
        VisualCaption(
            keyframe=sample_keyframes[0],
            scene_description="A red car parked on the street.",
            objects=["car", "street", "traffic_light"],
            activity="Car is stopped at the light.",
            anomaly=None,
        ),
        VisualCaption(
            keyframe=sample_keyframes[1],
            scene_description="A person crossing the road.",
            objects=["person", "crosswalk"],
            activity="Person is walking across the crosswalk.",
            anomaly=None,
        ),
        VisualCaption(
            keyframe=sample_keyframes[2],
            scene_description="An empty parking lot.",
            objects=["parking_lot"],
            activity="static scene",
            anomaly=None,
        ),
    ]


@pytest.fixture
def sample_transcripts():
    """Return TranscriptSegment objects spanning known time ranges."""
    return [
        TranscriptSegment(start_time=0.5, end_time=2.0, text="The red car stopped at the traffic light.", keywords=["car", "traffic", "light"]),
        TranscriptSegment(start_time=2.5, end_time=4.0, text="A pedestrian began crossing the street.", keywords=["pedestrian", "crossing", "street"]),
        TranscriptSegment(start_time=10.0, end_time=12.0, text="Nothing much happening now.", keywords=["nothing"]),
    ]


@pytest.fixture
def sample_shot_segments():
    """Return ShotSegment objects for a 90-frame video."""
    return [
        ShotSegment(segment_index=0, start_frame=0, end_frame=44, start_time=0.0, end_time=1.467, frame_count=45),
        ShotSegment(segment_index=1, start_frame=45, end_frame=89, start_time=1.5, end_time=2.967, frame_count=45),
    ]
