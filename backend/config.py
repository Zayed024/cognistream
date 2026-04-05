"""
CogniStream — Central Configuration

All tunable parameters live here. Values are read from environment variables
with sensible defaults for edge deployment (2-4 CPU cores, 4-6 GB RAM).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (if it exists)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ──────────────────────────────────────────────
# Faster-Whisper (Audio Transcriber)
# ──────────────────────────────────────────────
# Whisper model: "small" (fast, 2GB GPU), "large-v3-turbo" (best quality, 6GB GPU)
# When using large-v3-turbo, the VLM is unloaded from GPU first to free VRAM.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# ──────────────────────────────────────────────
# Embeddings (Multimodal Fusion)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # local model dimension

# SigLIP visual embedding — embeds frames directly into a text-searchable vector space.
# Enables visual similarity search without VLM captioning (much faster).
# Set to empty string to disable.
SIGLIP_MODEL = os.getenv("SIGLIP_MODEL", "google/siglip2-base-patch16-224")
SIGLIP_DIM = 768

# When NVIDIA is enabled, embeddings are 1024-dim (NV-EmbedQA-E5-V5).
# ChromaDB collections are dimension-locked, so we use a separate
# collection name to avoid conflicts when switching providers.
NVIDIA_EMBEDDING_DIM = 1024

# ──────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────
# Collection name auto-switches when NVIDIA is enabled to avoid dimension conflicts.
# Local embeddings (384-dim) and NVIDIA embeddings (1024-dim) can't share a collection.
CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "cognistream_nvidia" if os.getenv("NVIDIA_API_KEY", "") else "cognistream_segments",
)
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

# Multi-vector retrieval weights — how much each modality contributes to
# the final search score.  Set to 0 to disable a modality.
RETRIEVAL_WEIGHT_TEXT = float(os.getenv("RETRIEVAL_WEIGHT_TEXT", "0.5"))
RETRIEVAL_WEIGHT_VISUAL = float(os.getenv("RETRIEVAL_WEIGHT_VISUAL", "0.3"))
RETRIEVAL_WEIGHT_AUDIO = float(os.getenv("RETRIEVAL_WEIGHT_AUDIO", "0.2"))

# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────
# Only one video processes at a time on edge hardware.
# Additional submissions are rejected while processing.
MAX_CONCURRENT_JOBS = 1

# Processing mode: "auto" (default) = picks best strategy per model
#   moondream → 4-pass quality (small model, needs focused prompts)
#   gemma3n/gemma4/qwen → single-pass fast (larger model, handles structured output)
#   nvidia cloud → single-pass fast
# Can also be forced to "quality" (always 4-pass) or "fast" (always single-pass).
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "auto")

# Known small models that need 4-pass quality mode (can't follow combined prompts)
_SMALL_VLMS = {"moondream", "moondream2", "moondream:latest"}

# Number of parallel workers for VLM frame analysis.
# Local Ollama: keep at 1 (model handles one request at a time).
# NVIDIA cloud: set to 4-8 for concurrent API calls.
# Auto ("0") = 1 for local, 4 for NVIDIA cloud.
VLM_WORKERS = int(os.getenv("VLM_WORKERS", "0"))

# Number of parallel workers for shot detection (splits video into chunks).
SHOT_DETECTION_WORKERS = int(os.getenv("SHOT_DETECTION_WORKERS", "2"))

# Streaming / live-video chunk size in seconds
STREAM_CHUNK_SEC = int(os.getenv("STREAM_CHUNK_SEC", "30"))

# Live feed segment TTL — auto-delete segments older than this (hours).
# 0 = keep forever (default for file-based processing).
LIVE_SEGMENT_TTL_HOURS = int(os.getenv("LIVE_SEGMENT_TTL_HOURS", "24"))

# ──────────────────────────────────────────────
# NVIDIA NIM Cloud (optional — set API key to enable)
# When enabled, NVIDIA models are used for higher quality.
# When disabled (default), local models (Ollama/Whisper) are used.
# ──────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Which NVIDIA models to use (only active when NVIDIA_API_KEY is set)
NVIDIA_VLM_MODEL = os.getenv("NVIDIA_VLM_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
NVIDIA_ASR_MODEL = os.getenv("NVIDIA_ASR_MODEL", "nvidia/parakeet-ctc-1_1b-asr")
NVIDIA_ASR_FUNCTION_ID = os.getenv("NVIDIA_ASR_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081")
NVIDIA_CLIP_MODEL = os.getenv("NVIDIA_CLIP_MODEL", "nvidia/nvclip")
NVIDIA_GROUNDING_MODEL = os.getenv("NVIDIA_GROUNDING_MODEL", "nvidia/nv-grounding-dino")
NVIDIA_GROUNDING_URL = os.getenv("NVIDIA_GROUNDING_URL", "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino")

def is_nvidia_enabled() -> bool:
    """True if NVIDIA cloud mode is active (API key provided)."""
    return bool(NVIDIA_API_KEY)

# ──────────────────────────────────────────────
# Security
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Webhooks — notify external systems on events
# ──────────────────────────────────────────────
# Comma-separated list of URLs to POST event payloads to.
# Empty = disabled.  Events: video_processed, event_detected, live_chunk_ready
WEBHOOK_URLS = [u.strip() for u in os.getenv("WEBHOOK_URLS", "").split(",") if u.strip()]

# Optional API key authentication.  Set to enable — requests must include
# header "X-API-Key: <key>".  Empty = no auth (default for local dev).
# Supports multiple comma-separated keys for multi-user setups.
API_KEY = os.getenv("COGNISTREAM_API_KEY", "")
API_KEYS: set[str] = {k.strip() for k in API_KEY.split(",") if k.strip()}

# Rate limiting (requests per minute per IP)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
