"""Tests for the audio extractor (backend/audio/audio_extractor.py).

Tests cover the three-phase pipeline: probe -> extract -> validate.
Uses a synthetic video (no audio track) created with cv2 to test
graceful handling of missing audio.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import cv2
import pytest

from backend.audio.audio_extractor import (
    AudioExtractError,
    AudioExtractor,
    AudioProbe,
    AudioResult,
    _SILENCE_RMS_THRESHOLD,
)
from backend.db.models import VideoMeta, VideoStatus


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def extractor(tmp_path):
    """AudioExtractor using a temp directory for output."""
    return AudioExtractor(audio_dir=tmp_path / "audio_out")


@pytest.fixture
def silent_wav(tmp_path):
    """Create a silent 16 kHz mono WAV file (all zeros)."""
    path = tmp_path / "silent.wav"
    n_frames = 16000  # 1 second
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * n_frames)
    return path


@pytest.fixture
def noisy_wav(tmp_path):
    """Create a WAV file with clearly audible noise (high RMS)."""
    path = tmp_path / "noisy.wav"
    n_frames = 16000  # 1 second at 16 kHz
    rng = np.random.default_rng(42)
    # Generate 16-bit samples with large amplitude
    samples = rng.integers(-10000, 10000, size=n_frames, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return path


@pytest.fixture
def video_no_audio(tmp_path):
    """Create a video file using cv2 (no audio track) and return VideoMeta."""
    video_path = tmp_path / "no_audio.mp4"
    fps = 30.0
    width, height = 160, 120
    total_frames = 30  # 1 second

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for _ in range(total_frames):
        frame = np.full((height, width, 3), (100, 150, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return VideoMeta(
        id="noaudio_test",
        filename="no_audio.mp4",
        file_path=str(video_path),
        duration_sec=total_frames / fps,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        status=VideoStatus.UPLOADED,
        created_at="2026-01-01T00:00:00Z",
    )


# ──────────────────────────────────────────────
# Phase 1: Probe
# ──────────────────────────────────────────────


class TestProbe:
    def test_probe_returns_none_when_no_audio_stream(self, extractor, video_no_audio):
        """ffprobe on a cv2-generated video (no audio) should return None
        or a fallback probe (if ffprobe is not found)."""
        result = extractor.probe(video_no_audio.file_path)
        # Depending on whether ffprobe is installed:
        #   - With ffprobe: returns None (no audio stream)
        #   - Without ffprobe: returns fallback AudioProbe
        if result is not None:
            # Fallback probe: codec is "unknown"
            assert result.codec == "unknown"

    def test_probe_with_mocked_ffprobe_success(self, extractor, tmp_path):
        """Mock subprocess.run to simulate ffprobe finding an audio stream."""
        fake_output = json.dumps({
            "streams": [{
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": "2",
                "duration": "120.5",
            }]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output.encode()
        mock_result.stderr = b""

        with patch("backend.audio.audio_extractor.subprocess.run", return_value=mock_result):
            probe = extractor.probe(tmp_path / "fake.mp4")

        assert probe is not None
        assert probe.codec == "aac"
        assert probe.sample_rate == 44100
        assert probe.channels == 2
        assert probe.duration_sec == 120.5

    def test_probe_returns_none_on_nonzero_exit(self, extractor, tmp_path):
        """ffprobe exiting non-zero means no audio stream."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"error"

        with patch("backend.audio.audio_extractor.subprocess.run", return_value=mock_result):
            probe = extractor.probe(tmp_path / "bad.mp4")

        assert probe is None

    def test_probe_returns_none_on_empty_streams(self, extractor, tmp_path):
        """ffprobe succeeds but reports no streams."""
        fake_output = json.dumps({"streams": []})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output.encode()

        with patch("backend.audio.audio_extractor.subprocess.run", return_value=mock_result):
            probe = extractor.probe(tmp_path / "empty.mp4")

        assert probe is None

    def test_probe_fallback_when_ffprobe_not_found(self, extractor, tmp_path):
        """When ffprobe binary is missing, returns fallback probe."""
        with patch(
            "backend.audio.audio_extractor.subprocess.run",
            side_effect=FileNotFoundError("ffprobe not found"),
        ):
            probe = extractor.probe(tmp_path / "any.mp4")

        assert probe is not None
        assert probe.codec == "unknown"
        assert probe.sample_rate == 0

    def test_probe_fallback_on_timeout(self, extractor, tmp_path):
        """When ffprobe times out, returns fallback probe."""
        with patch(
            "backend.audio.audio_extractor.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
        ):
            probe = extractor.probe(tmp_path / "slow.mp4")

        assert probe is not None
        assert probe.codec == "unknown"

    def test_probe_returns_none_on_json_decode_error(self, extractor, tmp_path):
        """Corrupt ffprobe output should be handled gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"not valid json"

        with patch("backend.audio.audio_extractor.subprocess.run", return_value=mock_result):
            probe = extractor.probe(tmp_path / "corrupt.mp4")

        assert probe is None


# ──────────────────────────────────────────────
# Phase 3: Validation & silence detection
# ──────────────────────────────────────────────


class TestValidateWav:
    def test_silent_wav_detected(self, extractor, silent_wav):
        """A WAV full of zeros should be flagged as silent."""
        duration, is_silent = extractor._validate_wav(silent_wav)
        assert is_silent is True
        assert duration > 0  # has frames but is silent

    def test_noisy_wav_not_silent(self, extractor, noisy_wav):
        """A WAV with significant amplitude should NOT be flagged as silent."""
        duration, is_silent = extractor._validate_wav(noisy_wav)
        assert is_silent is False
        assert duration > 0

    def test_invalid_wav_returns_silent(self, extractor, tmp_path):
        """A file that is not a valid WAV should return (0, True)."""
        bad_file = tmp_path / "bad.wav"
        bad_file.write_bytes(b"not a wav file at all")
        duration, is_silent = extractor._validate_wav(bad_file)
        assert duration == 0.0
        assert is_silent is True


class TestCheckSilence:
    def test_empty_bytes_is_silent(self):
        assert AudioExtractor._check_silence(b"", 2) is True

    def test_zero_samples_is_silent(self):
        # 10 zero samples (16-bit)
        raw = struct.pack("<10h", *([0] * 10))
        assert AudioExtractor._check_silence(raw, 2) is True

    def test_loud_samples_not_silent(self):
        # 10 samples with large amplitude
        raw = struct.pack("<10h", *([5000] * 10))
        assert AudioExtractor._check_silence(raw, 2) is False

    def test_threshold_boundary(self):
        """Samples right at the threshold boundary."""
        # RMS = sqrt(mean(s^2)) needs to be >= _SILENCE_RMS_THRESHOLD
        # If all samples are value v, RMS = |v|.
        # So value = threshold should NOT be silent (rms == threshold, not < threshold)
        val = int(_SILENCE_RMS_THRESHOLD)
        raw = struct.pack("<10h", *([val] * 10))
        assert AudioExtractor._check_silence(raw, 2) is False

        # Value just below threshold should be silent
        val_below = int(_SILENCE_RMS_THRESHOLD) - 1
        raw = struct.pack("<10h", *([val_below] * 10))
        assert AudioExtractor._check_silence(raw, 2) is True


# ──────────────────────────────────────────────
# Full extract pipeline (mocked FFmpeg)
# ──────────────────────────────────────────────


class TestExtractPipeline:
    def test_extract_returns_none_when_no_audio(self, extractor, video_no_audio):
        """When probe returns None (no audio), extract() returns None."""
        with patch.object(extractor, "probe", return_value=None):
            result = extractor.extract(video_no_audio)
        assert result is None

    def test_extract_with_mocked_ffmpeg(self, extractor, video_no_audio, noisy_wav, tmp_path):
        """Full pipeline with mocked probe and ffmpeg — returns AudioResult."""
        fake_probe = AudioProbe(
            codec="aac", sample_rate=44100, channels=2, duration_sec=10.0
        )

        # Mock _run_ffmpeg to "produce" the noisy WAV as output
        with patch.object(extractor, "probe", return_value=fake_probe), \
             patch.object(extractor, "_run_ffmpeg", return_value=noisy_wav):
            result = extractor.extract(video_no_audio)

        assert result is not None
        assert isinstance(result, AudioResult)
        assert result.video_id == "noaudio_test"
        assert result.is_silent is False
        assert result.sample_rate == 16000
        assert result.duration_sec > 0
        assert result.probe.codec == "aac"

    def test_extract_flags_silent_audio(self, extractor, video_no_audio, silent_wav):
        """When extracted audio is silent, result.is_silent should be True."""
        fake_probe = AudioProbe(
            codec="aac", sample_rate=44100, channels=2, duration_sec=5.0
        )

        with patch.object(extractor, "probe", return_value=fake_probe), \
             patch.object(extractor, "_run_ffmpeg", return_value=silent_wav):
            result = extractor.extract(video_no_audio)

        assert result is not None
        assert result.is_silent is True

    def test_extract_raises_on_ffmpeg_failure(self, extractor, video_no_audio):
        """When FFmpeg fails, AudioExtractError should propagate."""
        fake_probe = AudioProbe(
            codec="aac", sample_rate=44100, channels=2, duration_sec=5.0
        )

        with patch.object(extractor, "probe", return_value=fake_probe), \
             patch.object(
                 extractor, "_run_ffmpeg",
                 side_effect=AudioExtractError("FFmpeg crashed"),
             ):
            with pytest.raises(AudioExtractError, match="FFmpeg crashed"):
                extractor.extract(video_no_audio)


# ──────────────────────────────────────────────
# WAV sample rate check
# ──────────────────────────────────────────────


class TestWavSampleRate:
    def test_wav_at_16khz(self, tmp_path):
        """Verify that a constructed 16 kHz WAV is recognized correctly."""
        path = tmp_path / "check_rate.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)

        with wave.open(str(path), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────


class TestUtilities:
    def test_ffprobe_path_from_ffmpeg_on_path(self):
        """When FFMPEG_PATH is just 'ffmpeg', ffprobe should be 'ffprobe'."""
        with patch("backend.audio.audio_extractor.FFMPEG_PATH", "ffmpeg"):
            result = AudioExtractor._ffprobe_path()
        assert result == "ffprobe"

    def test_ffprobe_path_from_absolute_ffmpeg(self):
        """When FFMPEG_PATH is absolute, ffprobe is in the same directory."""
        with patch("backend.audio.audio_extractor.FFMPEG_PATH", "/usr/local/bin/ffmpeg"):
            result = AudioExtractor._ffprobe_path()
        assert result.endswith("ffprobe")
        # On Windows, Path converts slashes: /usr/local/bin -> \usr\local\bin
        assert "usr" in result and "local" in result and "bin" in result

    def test_fallback_probe(self):
        """Fallback probe should have 'unknown' codec and zero values."""
        probe = AudioExtractor._fallback_probe()
        assert probe.codec == "unknown"
        assert probe.sample_rate == 0
        assert probe.channels == 0
        assert probe.duration_sec == 0.0

    def test_audio_dir_created_on_init(self, tmp_path):
        """AudioExtractor.__init__ should create the audio directory."""
        audio_dir = tmp_path / "new_audio_dir"
        assert not audio_dir.exists()
        AudioExtractor(audio_dir=audio_dir)
        assert audio_dir.exists()
