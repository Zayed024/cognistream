"""
CogniStream — Pipeline Orchestrator

Wires all processing modules into a sequential pipeline with:
    - Status tracking (persisted to SQLite)
    - Graceful degradation (VLM failure → audio-only mode)
    - Model lifecycle management (unload between stages)
    - Progress callbacks for real-time frontend updates
    - Single-job concurrency guard for edge hardware

Pipeline stages:
    1. Load & validate video         (VideoLoader)
    2. Detect shot boundaries        (ShotDetector)
    3. Extract keyframes             (FrameSampler)
    4. Extract audio                 (AudioExtractor)
    5. Visual analysis via VLM       (VLMRunner)       — skippable
    6. Audio transcription           (WhisperRunner)   — skippable
    7. Multimodal fusion & embedding (MultimodalEmbedder)
    8. Knowledge graph construction  (KnowledgeGraph)
    9. Event detection               (EventDetector)
   10. Store to ChromaDB             (ChromaStore)

Usage:
    orchestrator = PipelineOrchestrator()
    result = orchestrator.process(video_meta)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.audio.audio_extractor import AudioExtractor
from backend.audio.whisper_runner import WhisperRunner
from backend.db.chroma_store import ChromaStore
from backend.db.models import (
    FusedSegment,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)
from backend.db.sqlite import SQLiteDB
from backend.fusion.multimodal_embedder import MultimodalEmbedder
from backend.ingestion.frame_sampler import FrameSampler
from backend.ingestion.shot_detector import ShotDetector
from backend.knowledge.event_detector import EventDetector
from backend.knowledge.graph import KnowledgeGraph
from backend.visual.vlm_runner import OllamaClient, VLMRunner

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    """Tracks progress through the pipeline stages."""
    video_id: str
    stage: str = ""
    stage_number: int = 0
    total_stages: int = 10
    detail: str = ""
    started_at: float = 0.0
    elapsed_sec: float = 0.0


@dataclass
class PipelineResult:
    """Final result of a pipeline run."""
    video_id: str
    success: bool
    segments_stored: int = 0
    events_detected: int = 0
    elapsed_sec: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Type alias for progress callback
ProgressCallback = Callable[[PipelineProgress], None]


class PipelineOrchestrator:
    """End-to-end video processing pipeline with fault tolerance."""

    def __init__(
        self,
        db: SQLiteDB | None = None,
        store: ChromaStore | None = None,
        on_progress: ProgressCallback | None = None,
    ):
        self.db = db or SQLiteDB()
        self.store = store or ChromaStore()
        self._on_progress = on_progress
        self._lock = threading.Lock()
        self._active_video: Optional[str] = None

    @property
    def is_busy(self) -> bool:
        """True if a video is currently being processed."""
        return self._active_video is not None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Main entry point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def process(self, meta: VideoMeta) -> PipelineResult:
        """Run the full processing pipeline for a video.

        Thread-safe: only one video can process at a time.
        Additional calls while busy raise RuntimeError.
        """
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError(
                f"Pipeline busy processing {self._active_video}. "
                "Only one video at a time on edge hardware."
            )

        self._active_video = meta.id
        t_start = time.monotonic()
        result = PipelineResult(video_id=meta.id, success=False)
        progress = PipelineProgress(
            video_id=meta.id,
            started_at=t_start,
        )
        embedder: Optional[MultimodalEmbedder] = None

        try:
            # Mark as processing
            self.db.update_status(meta.id, VideoStatus.PROCESSING)

            # ── Stage 1–4: Ingestion ───────────────────────────
            self._emit(progress, 1, "Shot detection")
            detector = ShotDetector()
            segments = detector.detect(meta)

            self._emit(progress, 2, "Frame sampling")
            sampler = FrameSampler()
            keyframes = sampler.sample(meta, segments)

            self._emit(progress, 3, "Audio extraction")
            extractor = AudioExtractor()
            audio_result = extractor.extract(meta)

            # ── Stage 5: Visual analysis (skippable) ───────────
            self._emit(progress, 4, "Visual analysis (VLM)")
            captions: list[VisualCaption] = []
            try:
                client = OllamaClient()
                if client.is_available():
                    runner = VLMRunner(client)
                    captions = runner.analyse_keyframes(keyframes)
                else:
                    msg = "Ollama not available — skipping VLM analysis."
                    logger.warning(msg)
                    result.warnings.append(msg)
            except Exception as exc:
                msg = f"VLM analysis failed: {exc}"
                logger.error(msg)
                result.warnings.append(msg)

            # ── Stage 6: Transcription (skippable) ─────────────
            self._emit(progress, 5, "Audio transcription")
            transcripts: list[TranscriptSegment] = []
            if audio_result and not audio_result.is_silent:
                try:
                    whisper = WhisperRunner()
                    transcripts = whisper.transcribe(audio_result.audio_path)
                    # Free Whisper memory before embedding
                    whisper.unload_model()
                except Exception as exc:
                    msg = f"Transcription failed: {exc}"
                    logger.error(msg)
                    result.warnings.append(msg)
            elif audio_result and audio_result.is_silent:
                result.warnings.append("Audio track is silent — skipping transcription.")
            else:
                result.warnings.append("No audio stream — skipping transcription.")

            # ── Stage 7: Fusion & embedding ────────────────────
            self._emit(progress, 6, "Multimodal fusion")
            if not captions and not transcripts:
                msg = "No captions or transcripts — nothing to fuse."
                logger.error(msg)
                result.errors.append(msg)
                self._mark_failed(meta.id, msg)
                return result

            embedder = MultimodalEmbedder()
            fused = embedder.fuse_and_embed(meta.id, captions, transcripts)

            # ── Stage 8: Knowledge graph ───────────────────────
            self._emit(progress, 7, "Knowledge graph")
            kg = KnowledgeGraph(meta.id)
            kg.build_from_captions(captions, transcripts)
            kg.save()

            # ── Stage 9: Event detection ───────────────────────
            self._emit(progress, 8, "Event detection")
            event_detector = EventDetector()
            events = event_detector.detect(kg)
            result.events_detected = len(events)

            # Add events as searchable segments
            event_segments = self._events_to_segments(meta.id, events)
            if event_segments:
                embedder.embed(event_segments)
                fused.extend(event_segments)

            # Free embedding model memory
            embedder.unload_model()

            # ── Stage 10: Store to ChromaDB ────────────────────
            self._emit(progress, 9, "Storing embeddings")
            # Purge old data for this video (idempotent reprocessing)
            self.store.purge_video(meta.id)
            result.segments_stored = self.store.add_segments(fused)

            # ── Finalise ───────────────────────────────────────
            self._emit(progress, 10, "Complete")
            elapsed = time.monotonic() - t_start
            result.elapsed_sec = round(elapsed, 1)
            result.success = True

            self.db.update_status(
                meta.id,
                VideoStatus.PROCESSED,
                processed_at=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                "Pipeline complete: video=%s, segments=%d, events=%d, time=%.1fs",
                meta.id, result.segments_stored, result.events_detected, elapsed,
            )

        except Exception as exc:
            msg = f"Pipeline failed: {exc}"
            logger.exception(msg)
            result.errors.append(msg)
            self._mark_failed(meta.id, str(exc))

        finally:
            # Release models that may still be loaded on exception path
            if embedder is not None:
                try:
                    embedder.unload_model()
                except Exception:
                    pass
            self._active_video = None
            self._lock.release()

        return result

    # ── helpers ─────────────────────────────────────────────────

    def _emit(self, progress: PipelineProgress, stage: int, name: str) -> None:
        """Update and emit progress."""
        progress.stage_number = stage
        progress.stage = name
        progress.elapsed_sec = round(time.monotonic() - progress.started_at, 1)
        progress.detail = f"Stage {stage}/{progress.total_stages}: {name}"
        logger.info(progress.detail)
        if self._on_progress:
            self._on_progress(progress)

    def _mark_failed(self, video_id: str, error: str) -> None:
        self.db.update_status(video_id, VideoStatus.FAILED, error_message=error[:500])

    @staticmethod
    def _events_to_segments(
        video_id: str, events: list
    ) -> list[FusedSegment]:
        """Convert detected events into embeddable FusedSegments."""
        import uuid as _uuid

        segments = []
        for ev in events:
            segments.append(FusedSegment(
                id=_uuid.uuid4().hex,
                video_id=video_id,
                start_time=ev.start_time,
                end_time=ev.end_time,
                text=f"Event: {ev.event_type}. {ev.description}. Entities: {', '.join(ev.entities)}",
                source_type="event",
            ))
        return segments
