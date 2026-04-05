"""
Integration test — runs the pipeline stages that work without external services.

Stages tested (no Ollama/Whisper model needed):
  1. Video loading
  2. Shot detection
  3. Frame sampling
  4. Multimodal fusion (with synthetic data)
  5. Knowledge graph construction
  6. Event detection
  7. ChromaDB storage + retrieval

This validates the full data flow from ingested video to searchable results.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.models import (
    FusedSegment,
    Keyframe,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)
from backend.db.chroma_store import ChromaStore
from backend.db.sqlite import SQLiteDB
from backend.fusion.multimodal_embedder import MultimodalEmbedder
from backend.ingestion.frame_sampler import FrameSampler
from backend.ingestion.loader import VideoLoader
from backend.ingestion.shot_detector import ShotDetector
from backend.knowledge.event_detector import EventDetector
from backend.knowledge.graph import KnowledgeGraph
from backend.retrieval.query_engine import QueryEngine


def _create_test_video(path: Path, fps=30.0, seconds=6):
    """Create a synthetic video with 2 distinct scenes:
    - Scene 1 (0-3s): Red frames
    - Scene 2 (3-6s): Blue frames with a white rectangle (simulating object)
    """
    w, h = 320, 240
    total_frames = int(fps * seconds)
    mid = total_frames // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    for i in range(total_frames):
        if i < mid:
            frame = np.full((h, w, 3), (0, 0, 200), dtype=np.uint8)  # Red scene
        else:
            frame = np.full((h, w, 3), (200, 0, 0), dtype=np.uint8)  # Blue scene
            # Add a white rectangle (simulated object)
            cv2.rectangle(frame, (100, 80), (220, 160), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    return total_frames, fps, w, h


class TestIntegrationPipeline:
    """End-to-end integration test through all non-ML stages."""

    def test_full_pipeline_flow(self, tmp_path):
        """Run stages 1-3 with real video, stages 4-7 with synthetic captions."""

        # ── Stage 1: Load video ─────────────────────────────
        video_path = tmp_path / "integration_test.mp4"
        total_frames, fps, w, h = _create_test_video(video_path)

        loader = VideoLoader(video_dir=tmp_path / "videos")
        meta = loader.load(video_path)

        assert meta.status == VideoStatus.UPLOADED
        assert meta.total_frames == total_frames
        assert meta.fps == fps
        assert meta.width == w
        assert meta.height == h

        # Save to SQLite
        db = SQLiteDB(db_path=tmp_path / "test.db")
        db.save_video(meta)
        assert db.get_video(meta.id) is not None

        # ── Stage 2: Shot detection ─────────────────────────
        detector = ShotDetector(threshold=0.3, stride=1)
        segments = detector.detect(meta)

        assert len(segments) >= 1
        # With a sharp colour change, we should detect 2 segments
        assert len(segments) >= 2, f"Expected at least 2 segments, got {len(segments)}"

        # Segments should cover the full video
        assert segments[0].start_frame == 0
        assert segments[-1].end_frame == meta.total_frames - 1

        # ── Stage 3: Frame sampling ─────────────────────────
        sampler = FrameSampler(frame_dir=tmp_path / "frames", max_per_video=20)
        keyframes = sampler.sample(meta, segments)

        assert len(keyframes) > 0
        assert len(keyframes) <= 20
        # Verify keyframe images were written to disk
        for kf in keyframes:
            assert Path(kf.file_path).exists()

        # ── Stage 4: Simulate VLM captions ──────────────────
        # (We can't run Ollama in tests, so create realistic captions)
        captions = []
        for kf in keyframes:
            if kf.timestamp < 3.0:
                cap = VisualCaption(
                    keyframe=kf,
                    scene_description="A red-tinted scene, possibly indoor lighting.",
                    objects=["wall", "red_surface"],
                    activity="static scene",
                )
            else:
                cap = VisualCaption(
                    keyframe=kf,
                    scene_description="A blue scene with a white rectangular object.",
                    objects=["white_box", "blue_surface"],
                    activity="Object appearing in frame.",
                )
            captions.append(cap)

        # ── Simulate transcripts ────────────────────────────
        transcripts = [
            TranscriptSegment(0.5, 2.0, "The lighting in here is very red.", keywords=["lighting", "red"]),
            TranscriptSegment(3.5, 5.0, "Now I see a white box on a blue background.", keywords=["white", "box", "blue"]),
        ]

        # ── Stage 5: Fusion (no embedding yet) ─────────────
        embedder = MultimodalEmbedder()
        fused = embedder.fuse(meta.id, captions, transcripts)

        assert len(fused) > 0
        # Check we have fused, visual, and/or audio segments
        types = {s.source_type for s in fused}
        assert "visual" in types or "fused" in types

        # Verify transcript deduplication: each transcript text should appear at most once
        all_texts = " ".join(s.text for s in fused)
        assert all_texts.count("The lighting in here") <= 1
        assert all_texts.count("Now I see a white box") <= 1

        # ── Stage 6: Knowledge graph ───────────────────────
        kg = KnowledgeGraph(meta.id, graph_dir=tmp_path / "graphs")
        kg.build_from_captions(captions, transcripts)

        assert kg.G.number_of_nodes() > 0
        graph_path = kg.save()
        assert graph_path.exists()

        # ── Stage 7: Event detection ───────────────────────
        event_detector = EventDetector()
        events = event_detector.detect(kg)
        # May or may not find events with synthetic data — just verify it doesn't crash
        assert isinstance(events, list)

        # ── Stage 8: Embedding + storage ───────────────────
        # Use SentenceTransformer for real embeddings
        embedded = embedder.embed(fused)
        assert all(s.embedding is not None for s in embedded)
        # Dimension depends on provider: 384 (local MiniLM) or 1024 (NVIDIA NV-Embed)
        assert len(embedded[0].embedding) in (384, 1024)

        # Store in ChromaDB
        store = ChromaStore(
            persist_dir=tmp_path / "chroma",
            collection_name="test_integration",
            host="",
        )
        stored = store.add_segments(embedded)
        assert stored == len(embedded)

        # ── Stage 9: Query / retrieval ─────────────────────
        query_engine = QueryEngine(embedder=embedder, store=store)
        results = query_engine.search("red lighting scene", top_k=5)

        assert len(results) > 0
        # The top result should relate to the red scene
        assert results[0].score > 0.0
        assert results[0].video_id == meta.id

        # Search for the blue scene
        results2 = query_engine.search("white box blue background", top_k=5)
        assert len(results2) > 0

        # ── Verify ChromaDB persistence ────────────────────
        assert store.count(video_id=meta.id) == len(embedded)

        # Purge and verify
        store.purge_video(meta.id)
        assert store.count(video_id=meta.id) == 0

        # ── Cleanup model memory ───────────────────────────
        embedder.unload_model()

        print(f"\n  Integration test passed:")
        print(f"    Video: {meta.total_frames} frames, {meta.duration_sec:.1f}s")
        print(f"    Segments: {len(segments)} shots")
        print(f"    Keyframes: {len(keyframes)} extracted")
        print(f"    Fused: {len(fused)} segments ({dict((t, sum(1 for s in fused if s.source_type == t)) for t in types)})")
        print(f"    Graph: {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")
        print(f"    Events: {len(events)} detected")
        print(f"    Stored: {stored} vectors in ChromaDB")
        print(f"    Search 'red lighting': {len(results)} results (top score={results[0].score:.3f})")
        print(f"    Search 'white box': {len(results2)} results (top score={results2[0].score:.3f})")
