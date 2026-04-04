"""Tests for WhisperRunner NVIDIA fallback logic (backend/audio/whisper_runner.py).

Focuses on the transcribe() method's decision tree:
  1. Try NVIDIA Parakeet ASR when available
  2. Fall back to local Faster-Whisper when NVIDIA fails
  3. Use local directly when NVIDIA is not available

All tests mock external dependencies — no actual whisper model loading or
NVIDIA API calls.

Note: The whisper_runner imports ``nvidia`` and ``WhisperModel`` lazily
inside methods, so we patch at the source:
  - ``backend.providers.nvidia.nvidia``  (the module-level singleton)
  - ``faster_whisper.WhisperModel``      (imported inside _get_model)
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from backend.audio.whisper_runner import WhisperRunner, KeywordExtractor
from backend.db.models import TranscriptSegment


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def runner():
    """WhisperRunner instance — model will be mocked, never loaded."""
    return WhisperRunner(model_size="tiny", device="cpu", compute_type="int8")


@pytest.fixture
def audio_file(tmp_path):
    """Create a minimal valid WAV file for path-existence checks."""
    path = tmp_path / "test_audio.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 0.5 seconds of silence
        wf.writeframes(b"\x00\x00" * 8000)
    return path


@pytest.fixture
def nvidia_segments():
    """Sample NVIDIA ASR result (list of dicts)."""
    return [
        {"start_time": 0.0, "end_time": 2.5, "text": "Hello, this is a test."},
        {"start_time": 2.5, "end_time": 5.0, "text": "The weather is nice today."},
        {"start_time": 5.0, "end_time": 7.0, "text": "Goodbye."},
    ]


def _make_mock_whisper_segment(start: float, end: float, text: str):
    """Create a mock Faster-Whisper segment object."""
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def _make_mock_whisper_info(language: str = "en", probability: float = 0.95):
    """Create a mock Faster-Whisper TranscriptionInfo object."""
    info = MagicMock()
    info.language = language
    info.language_probability = probability
    return info


# The nvidia singleton is imported lazily inside transcribe() via:
#   from backend.providers.nvidia import nvidia
# So we patch the singleton at its source module.
_NVIDIA_PATCH_TARGET = "backend.providers.nvidia.nvidia"


# ──────────────────────────────────────────────
# NVIDIA-first path
# ──────────────────────────────────────────────


class TestNvidiaFirst:
    """When NVIDIA is available and succeeds, WhisperRunner should use it."""

    def test_uses_nvidia_when_available(self, runner, audio_file, nvidia_segments):
        """transcribe() should return NVIDIA results when available=True and
        nvidia.transcribe returns segments."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = True
        mock_nvidia.transcribe.return_value = nvidia_segments

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia):
            result = runner.transcribe(audio_file, extract_keywords=False)

        # Should have used NVIDIA, NOT loaded local model
        mock_nvidia.transcribe.assert_called_once_with(str(audio_file))
        assert len(result) == 3
        assert all(isinstance(seg, TranscriptSegment) for seg in result)
        assert result[0].text == "Hello, this is a test."
        assert result[0].start_time == 0.0
        assert result[0].end_time == 2.5
        assert result[2].text == "Goodbye."

    def test_nvidia_result_excludes_blank_text(self, runner, audio_file):
        """Segments with empty/whitespace text should be filtered out."""
        nvidia_result = [
            {"start_time": 0.0, "end_time": 1.0, "text": "Real content."},
            {"start_time": 1.0, "end_time": 2.0, "text": "   "},
            {"start_time": 2.0, "end_time": 3.0, "text": ""},
            {"start_time": 3.0, "end_time": 4.0, "text": "More content."},
        ]
        mock_nvidia = MagicMock()
        mock_nvidia.available = True
        mock_nvidia.transcribe.return_value = nvidia_result

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia):
            result = runner.transcribe(audio_file, extract_keywords=False)

        assert len(result) == 2
        assert result[0].text == "Real content."
        assert result[1].text == "More content."

    def test_nvidia_with_keyword_extraction(self, runner, audio_file, nvidia_segments):
        """When extract_keywords=True, keywords should be populated."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = True
        mock_nvidia.transcribe.return_value = nvidia_segments

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia):
            result = runner.transcribe(audio_file, extract_keywords=True)

        # Keywords may or may not be populated depending on text length,
        # but the extraction should not raise
        assert len(result) == 3


# ──────────────────────────────────────────────
# Fallback to local Whisper
# ──────────────────────────────────────────────


class TestFallbackToLocal:
    """When NVIDIA fails, WhisperRunner should fall back to local Whisper."""

    def test_falls_back_when_nvidia_returns_none(self, runner, audio_file):
        """If nvidia.transcribe() returns None, local Whisper should be used."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = True
        mock_nvidia.transcribe.return_value = None  # NVIDIA failed

        # Mock the local Whisper model
        local_segments = [
            _make_mock_whisper_segment(0.0, 3.0, "Local transcription result."),
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter(local_segments),
            _make_mock_whisper_info(),
        )

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia), \
             patch.object(runner, "_get_model", return_value=mock_model):
            result = runner.transcribe(audio_file, extract_keywords=False)

        # Verify NVIDIA was tried first
        mock_nvidia.transcribe.assert_called_once()
        # Verify local model was used as fallback
        mock_model.transcribe.assert_called_once()
        assert len(result) == 1
        assert result[0].text == "Local transcription result."

    def test_falls_back_when_nvidia_raises(self, runner, audio_file):
        """If nvidia.transcribe() raises an exception, it should be caught
        upstream (in the nvidia provider), returning None, triggering fallback."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = True
        mock_nvidia.transcribe.return_value = None  # Provider caught the error

        local_segments = [
            _make_mock_whisper_segment(0.0, 2.0, "Fallback works."),
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter(local_segments),
            _make_mock_whisper_info(),
        )

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia), \
             patch.object(runner, "_get_model", return_value=mock_model):
            result = runner.transcribe(audio_file, extract_keywords=False)

        assert len(result) == 1
        assert result[0].text == "Fallback works."


# ──────────────────────────────────────────────
# Local-only path (NVIDIA not available)
# ──────────────────────────────────────────────


class TestLocalOnly:
    """When NVIDIA is not available, go directly to local Whisper."""

    def test_uses_local_directly(self, runner, audio_file):
        """With nvidia.available=False, should skip NVIDIA entirely."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = False

        local_segments = [
            _make_mock_whisper_segment(0.0, 1.5, "Direct local result."),
            _make_mock_whisper_segment(1.5, 3.0, "Second segment."),
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter(local_segments),
            _make_mock_whisper_info(),
        )

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia), \
             patch.object(runner, "_get_model", return_value=mock_model):
            result = runner.transcribe(audio_file, extract_keywords=False)

        # NVIDIA.transcribe should NOT have been called
        mock_nvidia.transcribe.assert_not_called()
        # Local model should have been used
        mock_model.transcribe.assert_called_once()
        assert len(result) == 2
        assert result[0].text == "Direct local result."
        assert result[1].text == "Second segment."

    def test_local_filters_empty_text(self, runner, audio_file):
        """Empty segments from local Whisper should be filtered out."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = False

        local_segments = [
            _make_mock_whisper_segment(0.0, 1.0, "Content."),
            _make_mock_whisper_segment(1.0, 2.0, "  "),
            _make_mock_whisper_segment(2.0, 3.0, ""),
            _make_mock_whisper_segment(3.0, 4.0, "More content."),
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter(local_segments),
            _make_mock_whisper_info(),
        )

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia), \
             patch.object(runner, "_get_model", return_value=mock_model):
            result = runner.transcribe(audio_file, extract_keywords=False)

        assert len(result) == 2
        assert result[0].text == "Content."
        assert result[1].text == "More content."

    def test_local_with_language_param(self, runner, audio_file):
        """The language parameter should be forwarded to the local model."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = False

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter([]),
            _make_mock_whisper_info(language="fr", probability=0.88),
        )

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia), \
             patch.object(runner, "_get_model", return_value=mock_model):
            runner.transcribe(audio_file, language="fr", extract_keywords=False)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "fr"


# ──────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────


class TestErrorHandling:
    def test_raises_file_not_found_for_missing_audio(self, runner):
        """transcribe() should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            runner.transcribe("/nonexistent/path/audio.wav")

    def test_raises_file_not_found_for_directory(self, runner, tmp_path):
        """A directory path should also raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            runner.transcribe(tmp_path)


# ──────────────────────────────────────────────
# Model management
# ──────────────────────────────────────────────


class TestModelManagement:
    def test_unload_model_clears_model(self, runner):
        """unload_model() should set _model to None."""
        runner._model = MagicMock()  # Simulate loaded model
        assert runner._model is not None

        runner.unload_model()
        assert runner._model is None

    def test_unload_model_is_idempotent(self, runner):
        """Calling unload_model() when no model is loaded should not raise."""
        assert runner._model is None
        runner.unload_model()
        assert runner._model is None

    def test_get_model_lazy_loads(self, runner):
        """_get_model() should lazy-load the Faster-Whisper model."""
        assert runner._model is None

        mock_whisper_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_whisper_model_cls.return_value = mock_instance

        # WhisperModel is imported lazily inside _get_model via:
        #   from faster_whisper import WhisperModel
        # We install a fake faster_whisper module so the import succeeds.
        fake_faster_whisper = MagicMock()
        fake_faster_whisper.WhisperModel = mock_whisper_model_cls

        with patch.dict(sys.modules, {"faster_whisper": fake_faster_whisper}):
            model = runner._get_model()

        assert model is mock_instance
        assert runner._model is mock_instance
        mock_whisper_model_cls.assert_called_once_with(
            "tiny", device="cpu", compute_type="int8"
        )

    def test_get_model_returns_cached(self, runner):
        """Second call to _get_model() should return cached instance."""
        mock_model = MagicMock()
        runner._model = mock_model

        result = runner._get_model()
        assert result is mock_model

    def test_unload_then_reload(self, runner):
        """After unloading, _get_model() should load a fresh model."""
        first_model = MagicMock()
        runner._model = first_model

        runner.unload_model()
        assert runner._model is None

        second_model = MagicMock()
        mock_whisper_model_cls = MagicMock(return_value=second_model)
        fake_faster_whisper = MagicMock()
        fake_faster_whisper.WhisperModel = mock_whisper_model_cls

        with patch.dict(sys.modules, {"faster_whisper": fake_faster_whisper}):
            result = runner._get_model()

        assert result is second_model
        assert result is not first_model


# ──────────────────────────────────────────────
# Timestamp rounding
# ──────────────────────────────────────────────


class TestTimestampRounding:
    """Verify that timestamps are rounded to 3 decimal places."""

    def test_nvidia_timestamps_rounded(self, runner, audio_file):
        """NVIDIA segment timestamps should be rounded to 3 decimals."""
        nvidia_result = [
            {"start_time": 1.23456789, "end_time": 3.98765432, "text": "Test."},
        ]
        mock_nvidia = MagicMock()
        mock_nvidia.available = True
        mock_nvidia.transcribe.return_value = nvidia_result

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia):
            result = runner.transcribe(audio_file, extract_keywords=False)

        assert result[0].start_time == 1.235
        assert result[0].end_time == 3.988

    def test_local_timestamps_rounded(self, runner, audio_file):
        """Local Whisper segment timestamps should be rounded to 3 decimals."""
        mock_nvidia = MagicMock()
        mock_nvidia.available = False

        local_segments = [
            _make_mock_whisper_segment(0.12345, 1.67891, "Rounded."),
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter(local_segments),
            _make_mock_whisper_info(),
        )

        with patch(_NVIDIA_PATCH_TARGET, mock_nvidia), \
             patch.object(runner, "_get_model", return_value=mock_model):
            result = runner.transcribe(audio_file, extract_keywords=False)

        assert result[0].start_time == 0.123
        assert result[0].end_time == 1.679


# ──────────────────────────────────────────────
# Init parameters
# ──────────────────────────────────────────────


class TestInitParameters:
    def test_defaults_from_config(self):
        """WhisperRunner defaults should come from backend.config."""
        with patch("backend.audio.whisper_runner.WHISPER_MODEL_SIZE", "base"), \
             patch("backend.audio.whisper_runner.WHISPER_DEVICE", "cuda"), \
             patch("backend.audio.whisper_runner.WHISPER_COMPUTE_TYPE", "float16"):
            runner = WhisperRunner()

        assert runner.model_size == "base"
        assert runner.device == "cuda"
        assert runner.compute_type == "float16"

    def test_explicit_parameters_override_config(self):
        """Explicit constructor args should override config defaults."""
        runner = WhisperRunner(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
        )
        assert runner.model_size == "large-v2"
        assert runner.device == "cpu"
        assert runner.compute_type == "int8"
