"""
CogniStream — FastAPI Application Entry Point

Mounts the API router and configures logging.
This is the module referenced by the Dockerfile CMD:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import os
import uuid
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.router import init_event_loop, router
from backend.config import (
    API_KEY,
    API_KEYS,
    LOG_DIR,
    LOG_TO_FILE,
    LOG_FILE_LEVEL,
    LOG_LEVEL,
    RATE_LIMIT_RPM,
    get_pipeline_tuning_summary,
)
from backend.db.sqlite import SQLiteDB

# ── Logging ────────────────────────────────────────────────────
_RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_console_handler.setFormatter(logging.Formatter("%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
root_logger.addHandler(_console_handler)

if LOG_TO_FILE:
    log_path = Path(LOG_DIR) / f"run-{_RUN_ID}.log"
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setLevel(getattr(logging, LOG_FILE_LEVEL, logging.DEBUG))
    _file_handler.setFormatter(logging.Formatter("%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(_file_handler)
    logging.getLogger(__name__).info("Debug log file enabled: %s", log_path)

# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    # Capture the running event loop so background threads can use it safely
    init_event_loop(asyncio.get_running_loop())

    tuning = get_pipeline_tuning_summary()
    logging.getLogger(__name__).info("Run ID: %s", _RUN_ID)
    logging.getLogger(__name__).info("Hardware profile: %s", {
        "platform": tuning["platform"],
        "cpu_cores": tuning["cpu_cores"],
        "ram_gb": tuning["ram_gb"],
        "cuda": tuning["cuda"],
        "cuda_vram_gb": tuning["cuda_vram_gb"],
    })
    logging.getLogger(__name__).info("Pipeline tuning: %s", {
        "auto_tune": tuning["auto_tune"],
        "local_strategy_profile": tuning.get("local_strategy_profile"),
        "ollama_num_parallel_recommended": tuning.get("ollama_num_parallel_recommended"),
        "nvidia_cloud_mode": tuning["nvidia_cloud_mode"],
        "resolved_workers": tuning["resolved_workers"],
        "env_overrides": tuning["env_overrides"],
    })

    # Reset any videos stuck in PROCESSING from a previous crash
    db = SQLiteDB()
    reset_ids = db.reset_stale_processing()
    if reset_ids:
        logging.getLogger(__name__).info(
            "Reset %d interrupted processing jobs on startup", len(reset_ids)
        )
    yield
    # Shutdown: nothing needed


# ── Rate limiter (in-memory, per-IP) ──────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window rate limiter.

    Tracks request timestamps per client IP and returns 429 when
    the limit is exceeded.  Skips rate limiting for WebSocket upgrades.
    """

    def __init__(self, app, rpm: int = 120):
        super().__init__(app)
        self.rpm = rpm
        self.window = 60.0  # seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for WebSocket and health checks
        if request.url.path in ("/health", "/api/health") or "websocket" in request.headers.get("upgrade", "").lower():
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[client_ip]

        # Prune old entries outside the window
        cutoff = now - self.window
        self._hits[client_ip] = hits = [t for t in hits if t > cutoff]

        if len(hits) >= self.rpm:
            return Response(
                content='{"detail":"Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        hits.append(now)
        return await call_next(request)


# ── API key auth middleware ────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication via X-API-Key header.

    Only active when COGNISTREAM_API_KEY env var is set.
    Skips auth for health checks and CORS preflight.
    """

    def __init__(self, app, valid_keys: set[str]):
        super().__init__(app)
        self.valid_keys = valid_keys

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health, OPTIONS (CORS preflight), WebSocket, and static
        if (
            request.method == "OPTIONS"
            or request.url.path in ("/health", "/api/health", "/stats", "/api/stats")
            or "websocket" in request.headers.get("upgrade", "").lower()
        ):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided not in self.valid_keys:
            return Response(
                content='{"detail":"Invalid or missing API key."}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="CogniStream",
    description="Privacy-preserving multimodal video retrieval engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server during local development.
# In Docker, nginx handles proxying so CORS isn't needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (applied to all requests except health/WebSocket)
if RATE_LIMIT_RPM > 0:
    app.add_middleware(RateLimitMiddleware, rpm=RATE_LIMIT_RPM)

# API key auth (only when COGNISTREAM_API_KEY is set)
if API_KEYS:
    app.add_middleware(APIKeyMiddleware, valid_keys=API_KEYS)
    logging.getLogger(__name__).info("API key authentication enabled (%d keys)", len(API_KEYS))

app.include_router(router)
