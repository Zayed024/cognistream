"""
CogniStream — API Routes

All FastAPI endpoints in a single router.  Kept in one file because
edge deployments have a small API surface — splitting into five files
would add complexity without benefit.

Endpoints:
    POST /ingest-video     Upload a video file
    POST /process-video    Trigger processing pipeline
    POST /search           Natural language search
    GET  /video/{id}       Video metadata & status
    GET  /video/{id}/stream   Stream video file (range requests)
    GET  /video/{id}/frame/{name}  Serve a keyframe image
    GET  /video/{id}/progress      Polling progress
    GET  /video/{id}/progress/stream  SSE progress stream
    GET  /videos           List all videos
    DELETE /video/{id}     Delete a video and all its data
    GET  /health           Liveness check
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    FFMPEG_PATH,
    FRAME_DIR,
    GRAPH_DIR,
    MAX_VIDEO_SIZE_MB,
    VIDEO_DIR,
)
from backend.db.chroma_store import ChromaStore
from backend.db.models import VideoStatus
from backend.db.sqlite import SQLiteDB
from backend.fusion.multimodal_embedder import MultimodalEmbedder
from backend.ingestion.loader import VideoLoadError, VideoLoader
from backend.pipeline.orchestrator import PipelineOrchestrator, PipelineProgress
from backend.pipeline.streaming import LiveEvent, StreamingPipeline
from backend.retrieval.query_engine import QueryEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Progress tracking (in-memory, single-instance edge deployment) ──
_progress_store: dict[str, PipelineProgress] = {}
_progress_events: dict[str, asyncio.Event] = {}
_progress_lock = threading.Lock()
# Captured at startup by init_event_loop() so background threads can
# safely schedule asyncio.Event.set() via call_soon_threadsafe.
_event_loop: asyncio.AbstractEventLoop | None = None


def init_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the main event loop reference (called from lifespan handler)."""
    global _event_loop
    _event_loop = loop


def _on_progress(progress: PipelineProgress) -> None:
    """Callback to store progress updates from the orchestrator."""
    with _progress_lock:
        _progress_store[progress.video_id] = progress
        # Signal any waiting SSE clients
        if progress.video_id in _progress_events and _event_loop is not None:
            _event_loop.call_soon_threadsafe(
                _progress_events[progress.video_id].set
            )


def cleanup_progress(video_id: str) -> None:
    """Remove progress data for a completed/failed video."""
    with _progress_lock:
        _progress_store.pop(video_id, None)
        _progress_events.pop(video_id, None)


# ── Batch processing queue ──
_process_queue: deque[str] = deque()
_queue_lock = threading.Lock()


def _queue_worker() -> None:
    """Background thread that processes videos from the queue one at a time."""
    while True:
        with _queue_lock:
            if not _process_queue:
                return
            video_id = _process_queue[0]

        meta = _db.get_video(video_id)
        if meta is not None:
            try:
                _orchestrator.process(meta)
            except Exception:
                logger.exception("Queue processing failed for video %s", video_id)
            finally:
                cleanup_progress(video_id)

        with _queue_lock:
            if _process_queue and _process_queue[0] == video_id:
                _process_queue.popleft()


# ── Shared singletons (created once, reused across requests) ──
_db = SQLiteDB()
_store = ChromaStore()
_embedder = MultimodalEmbedder()
_query_engine = QueryEngine(embedder=_embedder, store=_store)
_orchestrator = PipelineOrchestrator(db=_db, store=_store, on_progress=_on_progress)

# ── Live feed WebSocket management ──
_ws_clients: dict[str, set[WebSocket]] = {}  # video_id → connected clients
_ws_lock = threading.Lock()


def _on_live_event(event: LiveEvent) -> None:
    """Broadcast a live event to all WebSocket clients subscribed to that feed."""
    with _ws_lock:
        clients = _ws_clients.get(event.video_id, set()).copy()
    if not clients or _event_loop is None:
        return

    payload = json.dumps({
        "video_id": event.video_id,
        "event_type": event.event_type,
        "data": event.data,
        "timestamp": event.timestamp,
    })
    for ws in clients:
        try:
            _event_loop.call_soon_threadsafe(
                asyncio.ensure_future,
                ws.send_text(payload),
            )
        except Exception:
            pass  # Client may have disconnected


_streaming_pipeline = StreamingPipeline(
    db=_db, store=_store, on_live_event=_on_live_event,
)
_loader = VideoLoader()

# Maximum allowed top_k to prevent ChromaDB abuse
_MAX_TOP_K = 100

# Chunk size for streaming uploads (1 MB)
_UPLOAD_CHUNK = 1024 * 1024


# ── Request / Response models ──────────────────────────────────

class ProcessRequest(BaseModel):
    video_id: str
    mode: str = Field(default="standard", description="'standard' or 'streaming' (chunked, near-RT)")
    chunk_sec: int = Field(default=30, ge=5, le=120)


class BatchProcessRequest(BaseModel):
    video_ids: list[str]


class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=_MAX_TOP_K)
    source_filter: Optional[str] = None


class SimilarRequest(BaseModel):
    segment_id: str
    top_k: int = Field(default=10, ge=1, le=_MAX_TOP_K)
    video_id: Optional[str] = None


class ClipRequest(BaseModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)


class AnnotationRequest(BaseModel):
    video_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    label: str
    note: str = ""
    color: str = "#3b82f6"


class LiveStartRequest(BaseModel):
    url: str = Field(description="RTSP/RTMP/HTTP stream URL or webcam index ('0', '1')")
    video_id: str = Field(description="Unique identifier for this live feed")
    chunk_sec: int = Field(default=15, ge=5, le=120)


class LiveStopRequest(BaseModel):
    video_id: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/health")
async def health():
    """Liveness check for Docker health probes."""
    return {"status": "ok", "service": "cognistream-backend"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/ingest-video", status_code=201)
async def ingest_video(
    file: UploadFile = File(...),
    name: Optional[str] = Query(None),
):
    """Upload a video file for processing.

    Streams the upload to disk in chunks to avoid loading the entire
    file into memory (critical on edge hardware with 3 GB RAM limit).
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    # Validate extension before reading any bytes
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}",
        )

    # Stream upload to a temp file, enforcing size limit
    max_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    tmp_path = VIDEO_DIR / f"_upload_{uuid.uuid4().hex}{ext}"

    try:
        bytes_written = 0
        with open(tmp_path, "wb") as f:
            while chunk := await file.read(_UPLOAD_CHUNK):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"File too large. Max allowed: {MAX_VIDEO_SIZE_MB} MB",
                    )
                f.write(chunk)

        meta = _loader.load(tmp_path)
    except VideoLoadError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    finally:
        # Loader copies the file; remove the temp upload
        tmp_path.unlink(missing_ok=True)

    _db.save_video(meta)

    return {
        "video_id": meta.id,
        "filename": meta.filename,
        "status": meta.status.value,
        "duration_sec": meta.duration_sec,
        "message": "Video uploaded. Call POST /process-video to begin processing.",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_process(meta):
    """Wrapper that catches and logs exceptions from the background thread."""
    try:
        _orchestrator.process(meta)
    except Exception:
        logger.exception("Background processing failed for video %s", meta.id)
    finally:
        cleanup_progress(meta.id)


def _safe_stream_process(meta, chunk_sec: int):
    """Streaming pipeline wrapper for background thread."""
    try:
        _db.update_status(meta.id, VideoStatus.PROCESSING)
        sp = StreamingPipeline(db=_db, store=_store, chunk_sec=chunk_sec)
        for progress in sp.process_streaming(meta):
            # Re-use the progress system so SSE works
            pp = PipelineProgress(
                video_id=meta.id,
                stage=f"Chunk {progress.chunk_index + 1}/{progress.total_chunks}",
                stage_number=progress.chunk_index + 1,
                total_stages=progress.total_chunks,
                detail=f"{progress.segments_stored} segments stored",
                started_at=0,
                elapsed_sec=progress.elapsed_sec,
            )
            _on_progress(pp)
        _db.update_status(meta.id, VideoStatus.PROCESSED)
    except Exception:
        logger.exception("Streaming pipeline failed for video %s", meta.id)
        _db.update_status(meta.id, VideoStatus.FAILED, error_message="Streaming pipeline failed")
    finally:
        cleanup_progress(meta.id)


@router.post("/process-video", status_code=202)
async def process_video(req: ProcessRequest):
    """Trigger the processing pipeline (runs in background thread).

    Modes:
        - ``standard``: Full sequential pipeline (parallel VLM+Whisper).
        - ``streaming``: Chunked pipeline — segments become searchable as
          each chunk finishes.  Better for long videos and near-RT use.
    """
    meta = _db.get_video(req.video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {req.video_id}")

    if meta.status == VideoStatus.PROCESSING:
        raise HTTPException(409, "Video is already being processed.")

    if _orchestrator.is_busy:
        raise HTTPException(
            429,
            "Pipeline is busy processing another video. "
            "Edge hardware supports one job at a time.",
        )

    if req.mode == "streaming":
        thread = threading.Thread(
            target=_safe_stream_process,
            args=(meta, req.chunk_sec),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=_safe_process,
            args=(meta,),
            daemon=True,
        )
    thread.start()

    return {
        "video_id": meta.id,
        "status": "PROCESSING",
        "mode": req.mode,
        "message": f"Processing started ({req.mode} mode).",
    }


@router.get("/video/{video_id}/progress")
async def get_progress(video_id: str):
    """Get processing progress for a video."""
    progress = _progress_store.get(video_id)
    if progress is None:
        meta = _db.get_video(video_id)
        if meta is None:
            raise HTTPException(404, f"Video not found: {video_id}")
        # Not processing or already done
        return {
            "video_id": video_id,
            "stage": meta.status.value,
            "stage_number": 10 if meta.status == VideoStatus.PROCESSED else 0,
            "total_stages": 10,
            "percent": 100 if meta.status == VideoStatus.PROCESSED else 0,
        }

    return {
        "video_id": progress.video_id,
        "stage": progress.stage,
        "stage_number": progress.stage_number,
        "total_stages": progress.total_stages,
        "percent": round((progress.stage_number / progress.total_stages) * 100),
        "elapsed_sec": progress.elapsed_sec,
    }


@router.get("/video/{video_id}/progress/stream")
async def stream_progress(video_id: str):
    """Stream processing progress via Server-Sent Events (SSE).

    Much more efficient than polling - client receives updates in real-time.
    """
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    async def event_generator():
        # Create an event for this video
        event = asyncio.Event()
        _progress_events[video_id] = event
        last_stage = -1

        try:
            while True:
                # Check current status
                progress = _progress_store.get(video_id)
                current_meta = _db.get_video(video_id)

                if current_meta is None:
                    # Video was deleted
                    yield f"data: {json.dumps({'done': True, 'deleted': True})}\n\n"
                    break

                if current_meta.status == VideoStatus.PROCESSED:
                    yield f"data: {json.dumps({'video_id': video_id, 'stage': 'Complete', 'stage_number': 10, 'total_stages': 10, 'percent': 100, 'done': True})}\n\n"
                    break

                if current_meta.status == VideoStatus.FAILED:
                    yield f"data: {json.dumps({'video_id': video_id, 'stage': 'Failed', 'percent': 0, 'done': True, 'error': True})}\n\n"
                    break

                if progress and progress.stage_number != last_stage:
                    last_stage = progress.stage_number
                    yield f"data: {json.dumps({'video_id': progress.video_id, 'stage': progress.stage, 'stage_number': progress.stage_number, 'total_stages': progress.total_stages, 'percent': round((progress.stage_number / progress.total_stages) * 100), 'elapsed_sec': progress.elapsed_sec})}\n\n"

                # Wait for next update or timeout after 2 seconds
                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            # Server shutting down or client disconnected
            return
        finally:
            # Cleanup
            _progress_events.pop(video_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/search")
async def search(req: SearchRequest):
    """Natural language search across processed videos."""
    results = _query_engine.search(
        query=req.query,
        top_k=req.top_k,
        video_id=req.video_id,
        source_filter=req.source_filter,
    )

    return {
        "query": req.query,
        "results": [
            {
                "video_id": r.video_id,
                "segment_id": r.segment_id,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "text": r.text,
                "source_type": r.source_type,
                "score": r.score,
                "event_type": r.event_type,
                "frame_url": r.frame_url,
            }
            for r in results
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video metadata & streaming
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/videos")
async def list_videos():
    """List all ingested videos."""
    videos = _db.list_videos()
    return {
        "videos": [
            {
                "video_id": v.id,
                "filename": v.filename,
                "duration_sec": v.duration_sec,
                "status": v.status.value,
                "created_at": v.created_at,
            }
            for v in videos
        ]
    }


@router.get("/video/{video_id}")
async def get_video(video_id: str):
    """Get video metadata and processing status."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    return {
        "video_id": meta.id,
        "filename": meta.filename,
        "duration_sec": meta.duration_sec,
        "fps": meta.fps,
        "resolution": f"{meta.width}x{meta.height}",
        "status": meta.status.value,
        "segment_count": _db.segment_count(video_id),
        "event_count": _db.event_count(video_id),
        "created_at": meta.created_at,
        "processed_at": meta.processed_at,
    }


@router.get("/video/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream the video file (supports range requests via FileResponse)."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    path = Path(meta.file_path)
    if not path.is_file():
        raise HTTPException(404, "Video file not found on disk.")

    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=meta.filename,
    )


@router.get("/video/{video_id}/frame/{frame_name}")
async def get_frame(video_id: str, frame_name: str):
    """Serve a keyframe image.

    Sanitises frame_name to prevent path traversal attacks.
    Only the filename component is used — directory separators are stripped.
    """
    # SECURITY: strip directory components to prevent traversal
    safe_name = Path(frame_name).name
    if not safe_name or safe_name != frame_name:
        raise HTTPException(400, "Invalid frame name.")

    frame_path = FRAME_DIR / video_id / safe_name

    # SECURITY: verify the resolved path is still inside FRAME_DIR
    try:
        frame_path.resolve().relative_to(FRAME_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid frame path.")

    if not frame_path.is_file():
        raise HTTPException(404, f"Frame not found: {safe_name}")

    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thumbnail
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/video/{video_id}/thumbnail")
async def get_thumbnail(video_id: str):
    """Serve the first keyframe as a thumbnail for the video card."""
    frame_dir = FRAME_DIR / video_id
    if not frame_dir.is_dir():
        raise HTTPException(404, "No keyframes available for this video.")

    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(frame_dir.glob("*.jpeg")) + sorted(frame_dir.glob("*.png"))
    if not frames:
        raise HTTPException(404, "No keyframes found.")

    return FileResponse(path=str(frames[0]), media_type="image/jpeg")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Find similar segments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/similar")
async def find_similar(req: SimilarRequest):
    """Find segments similar to a given segment using its embedding."""
    segment = _store.get_segment(req.segment_id)
    if segment is None:
        raise HTTPException(404, f"Segment not found: {req.segment_id}")

    embedding = segment.get("embedding")
    if embedding is None:
        raise HTTPException(400, "Segment has no embedding.")

    raw_results = _store.query(
        embedding=embedding,
        top_k=req.top_k + 1,  # +1 to exclude self
        video_id=req.video_id,
    )

    # Exclude the source segment itself
    results = [r for r in raw_results if r["id"] != req.segment_id][:req.top_k]

    return {
        "source_segment_id": req.segment_id,
        "results": [
            {
                "video_id": r["video_id"],
                "segment_id": r["id"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "text": r["text"],
                "source_type": r["source_type"],
                "score": r["score"],
                "frame_url": f"/video/{r['video_id']}/frame/{Path(r['frame_path']).name}"
                if r.get("frame_path") else None,
            }
            for r in results
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Clip export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/video/{video_id}/clip")
async def export_clip(video_id: str, req: ClipRequest):
    """Extract a video clip using FFmpeg and return it as a download."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    src = Path(meta.file_path)
    if not src.is_file():
        raise HTTPException(404, "Video file not found on disk.")

    if req.end_time <= req.start_time:
        raise HTTPException(400, "end_time must be greater than start_time.")

    clip_name = f"clip_{video_id[:8]}_{req.start_time:.1f}-{req.end_time:.1f}.mp4"
    clip_path = VIDEO_DIR / f"_clip_{uuid.uuid4().hex}.mp4"

    try:
        cmd = [
            str(FFMPEG_PATH), "-y",
            "-ss", str(req.start_time),
            "-to", str(req.end_time),
            "-i", str(src),
            "-c", "copy",
            "-movflags", "+faststart",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise HTTPException(500, f"FFmpeg failed: {result.stderr.decode()[:200]}")

        return FileResponse(
            path=str(clip_path),
            media_type="video/mp4",
            filename=clip_name,
            background=None,  # Don't delete before sending
        )
    except subprocess.TimeoutExpired:
        clip_path.unlink(missing_ok=True)
        raise HTTPException(504, "Clip export timed out.")
    except HTTPException:
        raise
    except Exception as exc:
        clip_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Clip export failed: {exc}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Knowledge graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/video/{video_id}/graph")
async def get_graph(video_id: str):
    """Return the knowledge graph as JSON nodes and edges for visualization."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    graph_path = GRAPH_DIR / f"{video_id}.graphml"
    if not graph_path.is_file():
        return {"nodes": [], "edges": []}

    import networkx as nx

    G = nx.read_graphml(str(graph_path))

    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "label": data.get("label", node_id),
            "type": data.get("type", "object"),
            "count": int(float(data.get("count", 1))),
            "first_seen": float(data.get("first_seen", 0)),
            "last_seen": float(data.get("last_seen", 0)),
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append({
            "source": src,
            "target": tgt,
            "action": data.get("action", ""),
            "timestamp": float(data.get("timestamp", 0)),
        })

    return {"nodes": nodes, "edges": edges}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Events & timeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/video/{video_id}/events")
async def get_events(video_id: str):
    """Return all detected events for a video timeline."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    events = _db.list_events(video_id)
    return {"video_id": video_id, "events": events}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch processing queue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/process-batch", status_code=202)
async def process_batch(req: BatchProcessRequest):
    """Queue multiple videos for sequential processing."""
    queued = []
    errors = []

    for vid in req.video_ids:
        meta = _db.get_video(vid)
        if meta is None:
            errors.append({"video_id": vid, "error": "Not found"})
            continue
        if meta.status == VideoStatus.PROCESSING:
            errors.append({"video_id": vid, "error": "Already processing"})
            continue

        with _queue_lock:
            if vid not in _process_queue:
                _process_queue.append(vid)
                queued.append(vid)

    # Start the worker if pipeline isn't busy
    if queued and not _orchestrator.is_busy:
        thread = threading.Thread(target=_queue_worker, daemon=True)
        thread.start()

    return {
        "queued": queued,
        "queue_size": len(_process_queue),
        "errors": errors,
    }


@router.get("/process-queue")
async def get_queue():
    """Return the current processing queue."""
    with _queue_lock:
        return {
            "queue": list(_process_queue),
            "queue_size": len(_process_queue),
            "is_busy": _orchestrator.is_busy,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/annotations", status_code=201)
async def create_annotation(req: AnnotationRequest):
    """Create a new annotation (bookmark/tag) for a video segment."""
    meta = _db.get_video(req.video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {req.video_id}")

    ann = {
        "id": uuid.uuid4().hex,
        "video_id": req.video_id,
        "start_time": req.start_time,
        "end_time": req.end_time,
        "label": req.label,
        "note": req.note,
        "color": req.color,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _db.save_annotation(ann)
    return ann


@router.get("/video/{video_id}/annotations")
async def list_annotations(video_id: str):
    """List all annotations for a video."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    annotations = _db.list_annotations(video_id)
    return {"video_id": video_id, "annotations": annotations}


@router.delete("/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation by ID."""
    deleted = _db.delete_annotation(annotation_id)
    if not deleted:
        raise HTTPException(404, f"Annotation not found: {annotation_id}")
    return {"message": "Annotation deleted.", "id": annotation_id}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Live video feeds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/live/start", status_code=201)
async def start_live_feed(req: LiveStartRequest):
    """Start a live video feed from an RTSP/RTMP/HTTP stream or webcam.

    The feed continuously captures and processes video in real-time chunks.
    Connect to the WebSocket at ``/ws/live/{video_id}`` to receive events.
    """
    try:
        status = _streaming_pipeline.start_live(
            url=req.url,
            video_id=req.video_id,
            chunk_sec=req.chunk_sec,
        )
        return {
            "video_id": status.video_id,
            "url": status.url,
            "state": status.state,
            "chunk_sec": req.chunk_sec,
            "message": f"Live feed started. Connect to /ws/live/{req.video_id} for real-time events.",
        }
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/live/stop")
async def stop_live_feed(req: LiveStopRequest):
    """Stop an active live feed."""
    stopped = _streaming_pipeline.stop_live(req.video_id)
    if not stopped:
        raise HTTPException(404, f"No active live feed: {req.video_id}")
    return {"video_id": req.video_id, "message": "Live feed stopping."}


@router.get("/live/status")
async def live_feed_status(video_id: Optional[str] = Query(None)):
    """Get status of all active live feeds, or a specific one."""
    feeds = _streaming_pipeline.get_live_status(video_id)
    return {
        "feeds": [
            {
                "video_id": f.video_id,
                "url": f.url,
                "state": f.state,
                "chunks_processed": f.chunks_processed,
                "total_segments": f.total_segments,
                "fps": f.fps,
                "started_at": f.started_at,
                "last_chunk_at": f.last_chunk_at,
                "error": f.error,
            }
            for f in feeds
        ],
    }


@router.websocket("/ws/live/{video_id}")
async def live_websocket(websocket: WebSocket, video_id: str):
    """WebSocket endpoint for real-time live feed events.

    Clients receive JSON messages with fields:
        - video_id: str
        - event_type: "chunk_ready" | "segment_indexed" | "event_detected" | "status_change" | "error"
        - data: dict (event-specific payload)
        - timestamp: ISO 8601

    Send ``{"action": "search", "query": "..."}`` to search indexed segments
    from this live feed in real-time.
    """
    await websocket.accept()

    # Register client
    with _ws_lock:
        if video_id not in _ws_clients:
            _ws_clients[video_id] = set()
        _ws_clients[video_id].add(websocket)

    try:
        # Send current status immediately
        feeds = _streaming_pipeline.get_live_status(video_id)
        if feeds:
            f = feeds[0]
            await websocket.send_json({
                "video_id": video_id,
                "event_type": "status_change",
                "data": {"state": f.state, "chunks_processed": f.chunks_processed},
                "timestamp": f.last_chunk_at or f.started_at,
            })

        # Listen for client messages (search queries, keepalives)
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"event_type": "ping"})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Handle live search request
            if msg.get("action") == "search" and msg.get("query"):
                results = _query_engine.search(
                    query=msg["query"],
                    top_k=msg.get("top_k", 10),
                    video_id=video_id,
                )
                await websocket.send_json({
                    "video_id": video_id,
                    "event_type": "search_results",
                    "data": {
                        "query": msg["query"],
                        "results": [
                            {
                                "segment_id": r.segment_id,
                                "start_time": r.start_time,
                                "end_time": r.end_time,
                                "text": r.text,
                                "score": r.score,
                                "source_type": r.source_type,
                            }
                            for r in results
                        ],
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket error for %s: %s", video_id, exc)
    finally:
        # Unregister client
        with _ws_lock:
            clients = _ws_clients.get(video_id)
            if clients:
                clients.discard(websocket)
                if not clients:
                    del _ws_clients[video_id]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video deletion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its associated data."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    if meta.status == VideoStatus.PROCESSING:
        raise HTTPException(409, "Cannot delete a video that is currently processing.")

    _store.purge_video(video_id)

    video_path = Path(meta.file_path)
    if video_path.is_file():
        video_path.unlink()
    frame_dir = FRAME_DIR / video_id
    if frame_dir.is_dir():
        shutil.rmtree(frame_dir)

    _db.delete_video(video_id)

    return {"message": f"Video {video_id} deleted.", "video_id": video_id}
