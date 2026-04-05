"""
CogniStream — Whisper Runner

Speech-to-text engine using Faster-Whisper (CTranslate2 backend).
Produces timestamped transcript segments with extracted keywords.

Architecture
────────────
WhisperRunner
    Loads the Faster-Whisper model once (lazy singleton pattern) and
    exposes a single ``transcribe()`` method that returns a list of
    ``TranscriptSegment`` objects.

KeywordExtractor
    Lightweight TF-IDF keyword extractor.  No external NLP library —
    uses Python stdlib only (Counter + math.log).  Runs after
    transcription to tag each segment with its top-K keywords.

Why lazy singleton?
    On edge hardware (4–6 GB RAM), loading the Whisper model (~150 MB
    for base/int8) should happen once and persist.  The singleton is
    created on first call to ``transcribe()`` and reused across all
    subsequent videos.

Why not spaCy for keywords?
    spaCy is reserved for entity extraction in the knowledge graph
    module.  Loading ``en_core_web_sm`` (~50 MB) here would add memory
    pressure for a task that TF-IDF handles adequately.

Usage:
    runner = WhisperRunner()
    segments = runner.transcribe("data/audio/abc123.wav")

    # With keyword extraction disabled:
    segments = runner.transcribe(path, extract_keywords=False)
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

# ── CUDA DLL fix for Windows ──────────────────────────────────
# pip-installed nvidia-cublas-cu12 places DLLs in site-packages,
# which Python/CTranslate2 can't find without this.
_CUDA_PACKAGES = ["nvidia.cublas", "nvidia.cuda_nvrtc", "nvidia.cudnn"]
for _pkg_name in _CUDA_PACKAGES:
    try:
        _pkg = __import__(_pkg_name, fromlist=[""])
        _bin = Path(_pkg.__path__[0]) / "bin"
        if _bin.is_dir():
            os.add_dll_directory(str(_bin))
    except (ImportError, AttributeError, OSError):
        pass

from backend.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
from backend.db.models import TranscriptSegment

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stop words (compact set — covers English function words)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "when",
    "where", "why", "how", "not", "no", "nor", "if", "then", "than",
    "so", "as", "just", "also", "very", "too", "here", "there", "all",
    "each", "every", "both", "few", "more", "most", "some", "any",
    "such", "only", "own", "same", "about", "up", "out", "into",
    "over", "after", "before", "between", "under", "again", "once",
    "during", "while", "through", "above", "below", "because",
    "until", "against", "going", "gonna", "like", "well", "okay",
    "oh", "uh", "um", "ah", "yeah", "yes", "right", "know",
    "think", "thing", "things", "really", "actually", "basically",
    "got", "get", "go", "come", "see", "look", "make", "take",
    "want", "say", "said", "one", "two",
}

# Minimum word length to be considered a keyword candidate
_MIN_WORD_LENGTH = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Keyword extractor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class KeywordExtractor:
    """TF-IDF keyword extraction over a corpus of transcript segments.

    Each segment is treated as a "document".  Term frequency is computed
    per segment, inverse document frequency across the full transcript.
    Top-K scoring terms per segment become that segment's keywords.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def extract(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Compute keywords for each segment in-place and return the list.

        Segments with very little text (< 3 candidate words) get no keywords.
        """
        if not segments:
            return segments

        corpus = [self._tokenise(seg.text) for seg in segments]
        idf = self._compute_idf(corpus)

        for seg, tokens in zip(segments, corpus):
            if len(tokens) < _MIN_WORD_LENGTH:
                continue
            seg.keywords = self._tfidf_top_k(tokens, idf)

        total_kw = sum(len(s.keywords) for s in segments)
        logger.debug(
            "Keyword extraction: %d keywords across %d segments",
            total_kw,
            len(segments),
        )
        return segments

    # ── internals ───────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lowercase, strip punctuation, remove stop words and short tokens."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return [
            w for w in words
            if len(w) >= _MIN_WORD_LENGTH and w not in _STOP_WORDS
        ]

    @staticmethod
    def _compute_idf(corpus: list[list[str]]) -> dict[str, float]:
        """Inverse document frequency: log(N / df) for each term."""
        n = len(corpus)
        if n == 0:
            return {}

        df: Counter[str] = Counter()
        for tokens in corpus:
            df.update(set(tokens))

        return {
            term: math.log(n / count) if count < n else 0.1
            for term, count in df.items()
        }

    def _tfidf_top_k(
        self, tokens: list[str], idf: dict[str, float]
    ) -> list[str]:
        """Return the top-K terms by TF-IDF score."""
        tf = Counter(tokens)
        total = len(tokens)

        scores: dict[str, float] = {}
        for term, count in tf.items():
            tf_score = count / total
            idf_score = idf.get(term, 0.1)
            scores[term] = tf_score * idf_score

        ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return ranked[: self.top_k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Whisper runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class WhisperRunner:
    """Transcribe audio files using Faster-Whisper.

    The model is loaded lazily on first use and cached for subsequent
    calls.  This avoids loading ~150 MB into RAM until actually needed,
    while ensuring the model stays resident once loaded.
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        keywords_top_k: int = 5,
    ):
        self.model_size = model_size or WHISPER_MODEL_SIZE
        configured_device = (device or WHISPER_DEVICE).strip().lower()
        # Backward-compatible alias used in older env samples.
        self.device = "cuda" if configured_device == "gpu" else configured_device
        self.compute_type = compute_type or WHISPER_COMPUTE_TYPE
        self._model = None  # lazy loaded
        self._keyword_extractor = KeywordExtractor(top_k=keywords_top_k)

    # ── public API ──────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        extract_keywords: bool = True,
    ) -> list[TranscriptSegment]:
        """Transcribe an audio file into timestamped segments.

        Uses NVIDIA Parakeet ASR when available (cloud, higher quality),
        otherwise falls back to local Faster-Whisper.

        Args:
            audio_path:        Path to a 16 kHz mono WAV file.
            language:          ISO 639-1 code (e.g. "en").  ``None`` for
                               auto-detection.
            extract_keywords:  If True, run TF-IDF keyword extraction
                               on the resulting segments.

        Returns:
            List of :class:`TranscriptSegment` sorted by start time.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Try NVIDIA cloud ASR first
        from backend.providers.nvidia import nvidia
        if nvidia.available:
            logger.info("Transcribing via NVIDIA ASR: %s", audio_path.name)
            result = nvidia.transcribe(str(audio_path))
            if result is not None:
                transcript = [
                    TranscriptSegment(
                        start_time=round(seg["start_time"], 3),
                        end_time=round(seg["end_time"], 3),
                        text=seg["text"],
                    )
                    for seg in result
                    if seg["text"].strip()
                ]
                logger.info("NVIDIA ASR: %d segments", len(transcript))
                if extract_keywords and transcript:
                    self._keyword_extractor.extract(transcript)
                return transcript
            logger.warning("NVIDIA ASR failed, falling back to local Whisper")

        model = self._get_model()

        logger.info(
            "Transcribing: %s (model=%s, device=%s)",
            audio_path.name,
            self.model_size,
            self.device,
        )
        t_start = time.monotonic()

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,               # skip silence regions
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        logger.info(
            "Detected language: %s (probability %.2f)",
            info.language,
            info.language_probability,
        )

        # Materialise the segment iterator into dataclass objects
        transcript: list[TranscriptSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            transcript.append(
                TranscriptSegment(
                    start_time=round(seg.start, 3),
                    end_time=round(seg.end, 3),
                    text=text,
                )
            )

        elapsed = time.monotonic() - t_start
        logger.info(
            "Transcription complete: %d segments in %.1fs",
            len(transcript),
            elapsed,
        )

        # Keyword extraction
        if extract_keywords and transcript:
            logger.debug("Running keyword extraction on %d segments", len(transcript))
            self._keyword_extractor.extract(transcript)

        return transcript

    # ── model management ────────────────────────────────────────

    def _get_model(self):
        """Lazy-load the Faster-Whisper model (singleton per instance)."""
        if self._model is not None:
            return self._model

        logger.info(
            "Loading Faster-Whisper model: size=%s, device=%s, compute=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        t_start = time.monotonic()

        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            if self.device != "cpu":
                logger.warning(
                    "Whisper device '%s' unavailable (%s). Falling back to CPU.",
                    self.device,
                    exc,
                )
                self.device = "cpu"
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="int8",
                )
            else:
                raise

        logger.info(
            "Whisper model loaded in %.1fs",
            time.monotonic() - t_start,
        )
        return self._model

    def unload_model(self) -> None:
        """Release the model from memory.

        Useful in edge environments when switching to a memory-intensive
        stage of the pipeline (e.g. VLM inference).
        """
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Whisper model unloaded.")
