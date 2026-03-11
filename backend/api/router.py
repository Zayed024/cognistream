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
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    FRAME_DIR,
    MAX_VIDEO_SIZE_MB,
    VIDEO_DIR,
)
from backend.db.chroma_store import ChromaStore
from backend.db.models import VideoStatus
from backend.db.sqlite import SQLiteDB
from backend.fusion.multimodal_embedder import MultimodalEmbedder
from backend.ingestion.loader import VideoLoadError, VideoLoader
from backend.pipeline.orchestrator import PipelineOrchestrator, PipelineProgress
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


# ── Shared singletons (created once, reused across requests) ──
_db = SQLiteDB()
_store = ChromaStore()
_embedder = MultimodalEmbedder()
_query_engine = QueryEngine(embedder=_embedder, store=_store)
_orchestrator = PipelineOrchestrator(db=_db, store=_store, on_progress=_on_progress)
_loader = VideoLoader()

# Maximum allowed top_k to prevent ChromaDB abuse
_MAX_TOP_K = 100

# Chunk size for streaming uploads (1 MB)
_UPLOAD_CHUNK = 1024 * 1024


# ── Request / Response models ──────────────────────────────────

class ProcessRequest(BaseModel):
    video_id: str


class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=_MAX_TOP_K)
    source_filter: Optional[str] = None


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


@router.post("/process-video", status_code=202)
async def process_video(req: ProcessRequest):
    """Trigger the full processing pipeline (runs in background thread)."""
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

    thread = threading.Thread(
        target=_safe_process,
        args=(meta,),
        daemon=True,
    )
    thread.start()

    return {
        "video_id": meta.id,
        "status": "PROCESSING",
        "message": "Processing started.",
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
