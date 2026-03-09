"""
CogniStream — Central Configuration

All tunable parameters live here. Values are read from environment variables
with sensible defaults for edge deployment (2-4 CPU cores, 4-6 GB RAM).
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Path roots
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(os.getenv("COGNISTREAM_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "data"

VIDEO_DIR = DATA_DIR / "videos"
FRAME_DIR = DATA_DIR / "frames"
AUDIO_DIR = DATA_DIR / "audio"
GRAPH_DIR = DATA_DIR / "graphs"
DB_DIR = DATA_DIR / "db"

SQLITE_PATH = DB_DIR / "cognistream.db"
CHROMA_DIR = DB_DIR / "chroma"

# Ensure all directories exist at import time
for _d in (VIDEO_DIR, FRAME_DIR, AUDIO_DIR, GRAPH_DIR, DB_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Video ingestion
# ──────────────────────────────────────────────
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "2048"))

# Shot detection: histogram-difference threshold (0-1 scale, lower = more sensitive)
SHOT_THRESHOLD = float(os.getenv("SHOT_THRESHOLD", "0.45"))
# Minimum segment length in frames (prevents micro-segments from noise)
MIN_SEGMENT_FRAMES = int(os.getenv("MIN_SEGMENT_FRAMES", "15"))

# Frame sampling
MAX_KEYFRAMES_PER_VIDEO = int(os.getenv("MAX_KEYFRAMES", "200"))
MIN_KEYFRAMES_PER_SEGMENT = 1
MAX_KEYFRAMES_PER_SEGMENT = int(os.getenv("MAX_KEYFRAMES_PER_SEGMENT", "10"))
KEYFRAME_IMAGE_FORMAT = "jpg"
KEYFRAME_JPEG_QUALITY = 85

# Audio extraction
AUDIO_SAMPLE_RATE = 16000  # 16 kHz mono — required by Whisper models
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# ──────────────────────────────────────────────
# Ollama (Visual Narrative Engine)
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "moondream2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ──────────────────────────────────────────────
# Faster-Whisper (Audio Transcriber)
# ──────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ──────────────────────────────────────────────
# Embeddings (Multimodal Fusion)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# ──────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────
CHROMA_COLLECTION = "cognistream_segments"
# When running in Docker, ChromaDB is a separate service.
# Set CHROMA_HOST to enable HTTP client mode instead of embedded.
CHROMA_HOST = os.getenv("CHROMA_HOST", "")  # empty = embedded/persistent mode
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ──────────────────────────────────────────────
# Multimodal fusion
# ──────────────────────────────────────────────
# Temporal overlap window (seconds) for merging visual captions with transcripts.
FUSION_WINDOW_SEC = float(os.getenv("FUSION_WINDOW_SEC", "2.0"))

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
DEFAULT_TOP_K = 10

# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────
# Only one video processes at a time on edge hardware.
# Additional submissions are rejected while processing.
MAX_CONCURRENT_JOBS = 1

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
