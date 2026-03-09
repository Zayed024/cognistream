"""
CogniStream — FastAPI Application Entry Point

Mounts the API router and configures logging.
This is the module referenced by the Dockerfile CMD:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import router
from backend.config import LOG_LEVEL

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="CogniStream",
    description="Privacy-preserving multimodal video retrieval engine",
    version="0.1.0",
)

# CORS — allow the Vite dev server during local development.
# In Docker, nginx handles proxying so CORS isn't needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
