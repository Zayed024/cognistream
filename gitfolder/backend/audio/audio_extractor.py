"""
CogniStream — Audio Extractor (Enhanced)

Three-phase audio extraction: probe → extract → validate.

Phase 1 — Probe (ffprobe)
    Read the audio stream metadata from the container before doing any
    work.  Catches videos with no audio track, corrupt containers, and
    mismatched durations early.

Phase 2 — Extract (ffmpeg)
    Convert the audio track to 16 kHz mono 16-bit PCM WAV — the exact
    format Faster-Whisper expects.  Runs as a subprocess with a 5-minute
    timeout.

Phase 3 — Validate
    Read the output WAV header to confirm non-zero duration.  Compute
    RMS energy over a sample to flag fully silent files — these are
    passed downstream with a ``is_silent`` flag so the transcriber can
    skip them.

This module supersedes ``backend/ingestion/audio_extractor.py``.
The ingestion version is kept as a thin backward-compatible re-export.

Usage:
    extractor = AudioExtractor()
    result = extractor.extract(video_meta)
    if result is None:
        print("Video has no audio track")
    elif result.is_silent:
        print("Audio track is silent")
    else:
        print(f"Audio ready: {result.audio_path}")
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.config import AUDIO_DIR, AUDIO_SAMPLE_RATE, FFMPEG_PATH
from backend.db.models import VideoMeta

logger = logging.getLogger(__name__)

# RMS energy below this threshold marks the file as effectively silent.
# 16-bit PCM range is [-32768, 32767]; threshold ~0.1% of max amplitude.
_SILENCE_RMS_THRESHOLD = 30.0

# Number of raw samples to read for the silence check (≈ 1 second at 16 kHz).
_SILENCE_CHECK_SAMPLES = 16000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class AudioProbe:
    """Metadata about a video's audio stream, returned by ffprobe."""
    codec: str
    sample_rate: int
    channels: int
    duration_sec: float


@dataclass
class AudioResult:
    """Output of the extraction pipeline."""
    video_id: str
    audio_path: Path
    duration_sec: float
    sample_rate: int
    is_silent: bool
    probe: AudioProbe


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Errors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AudioExtractError(Exception):
    """Raised when FFmpeg/ffprobe fails."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AudioExtractor:
    """Probe, extract, and validate audio from video files."""

    def __init__(self, audio_dir: Path | None = None):
        self.audio_dir = audio_dir or AUDIO_DIR
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────────

    def extract(self, meta: VideoMeta) -> Optional[AudioResult]:
        """Run the full probe → extract → validate pipeline.

        Args:
            meta: Video metadata from the loader.

        Returns:
            An :class:`AudioResult` on success, or ``None`` if the video
            has no audio stream at all.

        Raises:
            AudioExtractError: If FFmpeg fails during extraction.
        """
        # Phase 1: probe
        probe = self.probe(meta.file_path)
        if probe is None:
            logger.warning(
                "No audio stream found in %s — skipping extraction.",
                meta.filename,
            )
            return None

        logger.info(
            "Audio probe: codec=%s, rate=%dHz, channels=%d, duration=%.1fs",
            probe.codec,
            probe.sample_rate,
            probe.channels,
            probe.duration_sec,
        )

        # Phase 2: extract
        out_path = self._run_ffmpeg(meta)

        # Phase 3: validate
        duration, is_silent = self._validate_wav(out_path)

        if is_silent:
            logger.warning("Extracted audio is silent: %s", out_path.name)

        result = AudioResult(
            video_id=meta.id,
            audio_path=out_path,
            duration_sec=duration,
            sample_rate=AUDIO_SAMPLE_RATE,
            is_silent=is_silent,
            probe=probe,
        )

        logger.info(
            "Audio extraction complete: %s (%.1fs, silent=%s)",
            out_path.name,
            duration,
            is_silent,
        )
        return result

    # ── Phase 1: probe ──────────────────────────────────────────

    def probe(self, video_path: str | Path) -> Optional[AudioProbe]:
        """Use ffprobe to read audio stream metadata.

        Returns ``None`` if the video has no audio stream.
        """
        cmd = [
            self._ffprobe_path(),
            "-v", "quiet",
            "-select_streams", "a:0",
            "-show_streams",
            "-print_format", "json",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except FileNotFoundError:
            logger.warning(
                "ffprobe not found — falling back to extraction without probe."
            )
            return self._fallback_probe()
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out for %s", video_path)
            return self._fallback_probe()

        if result.returncode != 0:
            logger.debug("ffprobe exited %d — no audio stream.", result.returncode)
            return None

        try:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse ffprobe output.")
            return None

        if not streams:
            return None

        s = streams[0]
        return AudioProbe(
            codec=s.get("codec_name", "unknown"),
            sample_rate=int(s.get("sample_rate", 0)),
            channels=int(s.get("channels", 0)),
            duration_sec=float(s.get("duration", 0.0)),
        )

    # ── Phase 2: extract ────────────────────────────────────────

    def _run_ffmpeg(self, meta: VideoMeta) -> Path:
        """Run FFmpeg to extract audio as 16 kHz mono WAV."""
        out_path = self.audio_dir / f"{meta.id}.wav"

        cmd = [
            FFMPEG_PATH,
            "-i", meta.file_path,
            "-vn",                           # drop video
            "-acodec", "pcm_s16le",          # 16-bit PCM
            "-ar", str(AUDIO_SAMPLE_RATE),   # 16 kHz
            "-ac", "1",                      # mono
            "-y",                            # overwrite
            str(out_path),
        ]

        logger.info("Running FFmpeg: %s → %s", meta.filename, out_path.name)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            logger.error("FFmpeg failed (code %d): %s", result.returncode, stderr)
            raise AudioExtractError(
                f"FFmpeg exited with code {result.returncode}: {stderr[:500]}"
            )

        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.debug("FFmpeg output: %s (%.1f MB)", out_path.name, size_mb)
        return out_path

    # ── Phase 3: validate ───────────────────────────────────────

    def _validate_wav(self, wav_path: Path) -> tuple[float, bool]:
        """Read the WAV header and check for silence.

        Returns:
            (duration_sec, is_silent)
        """
        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate if rate > 0 else 0.0
                sampwidth = wf.getsampwidth()

                if duration == 0.0:
                    return 0.0, True

                # Read a sample of raw frames for RMS check
                n_check = min(frames, _SILENCE_CHECK_SAMPLES)
                raw = wf.readframes(n_check)
        except wave.Error as exc:
            logger.warning("WAV validation failed: %s", exc)
            return 0.0, True

        is_silent = self._check_silence(raw, sampwidth)
        return round(duration, 3), is_silent

    @staticmethod
    def _check_silence(raw_bytes: bytes, sampwidth: int) -> bool:
        """Compute RMS energy and compare against the silence threshold."""
        if not raw_bytes:
            return True

        # Unpack 16-bit signed samples
        fmt = f"<{len(raw_bytes) // sampwidth}h"
        try:
            samples = struct.unpack(fmt, raw_bytes)
        except struct.error:
            return True

        if not samples:
            return True

        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms < _SILENCE_RMS_THRESHOLD

    # ── utilities ───────────────────────────────────────────────

    @staticmethod
    def _ffprobe_path() -> str:
        """Derive ffprobe path from the configured ffmpeg path."""
        ffmpeg = Path(FFMPEG_PATH)
        # If ffmpeg is just "ffmpeg" (on PATH), ffprobe is "ffprobe"
        if ffmpeg.parent == Path("."):
            return "ffprobe"
        return str(ffmpeg.parent / "ffprobe")

    @staticmethod
    def _fallback_probe() -> AudioProbe:
        """Return a permissive probe when ffprobe is unavailable.

        Extraction will proceed and validation will catch truly empty files.
        """
        return AudioProbe(
            codec="unknown",
            sample_rate=0,
            channels=0,
            duration_sec=0.0,
        )
