"""
CogniStream — Audio Extractor (ingestion re-export)

The canonical implementation lives in ``backend.audio.audio_extractor``.
This module re-exports the classes for backward compatibility with code
that imports from the ingestion package.
"""

from backend.audio.audio_extractor import (  # noqa: F401
    AudioExtractError,
    AudioExtractor,
    AudioProbe,
    AudioResult,
)

__all__ = ["AudioExtractError", "AudioExtractor", "AudioProbe", "AudioResult"]
