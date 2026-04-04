# CogniStream

Privacy-preserving multimodal video retrieval engine. Search video content with natural language queries like *"Show me when the red car arrived"* — entirely offline, no cloud APIs.

## How It Works

CogniStream processes videos through a 10-stage pipeline:

1. **Shot detection** — HSV histogram correlation finds scene boundaries
2. **Frame sampling** — Adaptive keyframe extraction (budget per shot, max 200)
3. **Audio extraction** — FFmpeg pulls 16kHz mono WAV
4. **Visual analysis** — Local VLM (Moondream2 via Ollama) describes each keyframe in 4 passes: scene, objects, activity, anomaly
5. **Transcription** — Faster-Whisper (int8, CPU) produces timestamped speech segments with TF-IDF keywords
6. **Multimodal fusion** — Visual captions + transcripts merged by temporal overlap, each transcript assigned to nearest keyframe
7. **Embedding** — SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) encodes fused text
8. **Knowledge graph** — NetworkX directed graph of entities + temporal relationships
9. **Event detection** — Pattern matching over graph edges (car arrival, building entry, suspicious activity, etc.)
10. **Vector storage** — ChromaDB stores embeddings with metadata for cosine similarity search

Queries hit a 4-stage retrieval pipeline: embed query → ChromaDB search (2x overfetch) → temporal re-ranking (Gaussian decay) → format results.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│   SearchBar → ResultsPanel → VideoPlayer + Timeline      │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────┐
│                 Backend (FastAPI)                         │
│                                                          │
│  Ingestion ──► Visual ──► Audio ──► Fusion ──► Storage   │
│  loader        vlm_runner  extractor  embedder  chroma   │
│  shot_detect   caption     whisper    graph     sqlite   │
│  frame_sample              keywords   events             │
│                                                          │
│  Retrieval: query_engine (embed → search → rerank)       │
└──────┬──────────────┬───────────────────────────────────┘
       │              │
  ┌────▼────┐   ┌─────▼─────┐
  │ Ollama  │   │ ChromaDB  │
  │ VLM     │   │ Vectors   │
  └─────────┘   └───────────┘
```

## Project Structure

```
cognistream/
├── backend/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Central configuration
│   ├── api/router.py           # REST endpoints
│   ├── ingestion/              # Video loader, shot detector, frame sampler
│   ├── visual/                 # Ollama VLM runner, caption processor
│   ├── audio/                  # FFmpeg extractor, Faster-Whisper runner
│   ├── fusion/                 # Multimodal embedder (fusion + encoding)
│   ├── knowledge/              # Knowledge graph, event detector
│   ├── retrieval/              # Query engine (search + re-rank)
│   ├── pipeline/               # Orchestrator (wires all stages)
│   └── db/                     # SQLite + ChromaDB wrappers
├── frontend/
│   └── src/
│       ├── components/         # SearchBar, VideoPlayer, ResultsPanel, TimelineMarkers
│       ├── hooks/              # useSearch, useVideo
│       ├── api/client.ts       # Axios HTTP client
│       └── types/index.ts      # TypeScript interfaces
├── docker/
│   ├── docker-compose.yml      # 4 services with resource limits
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── nginx.conf
│   └── ollama-entrypoint.sh
├── scripts/pull_models.sh      # Pre-download models for offline use
├── requirements.txt
└── .env.example
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- FFmpeg
- [Ollama](https://ollama.com/) with `moondream2` pulled

### Local Development

```bash
# 1. Clone and install
git clone <repo-url> && cd cognistream
pip install -r requirements.txt

# 2. Pull models (run once, requires internet)
bash scripts/pull_models.sh

# 3. Start Ollama
ollama serve &
ollama pull moondream2

# 4. Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 5. Start frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### Docker (Recommended)

```bash
# Build and start all services
cd docker
docker compose up --build

# Or run detached
docker compose up --build -d
```

Services start in order: Ollama → ChromaDB → Backend → Frontend.

| Service  | Port  | Resources        |
|----------|-------|------------------|
| Frontend | 3000  | 0.5 CPU, 256 MB  |
| Backend  | 8000  | 2.0 CPU, 3 GB    |
| Ollama   | 11434 | 2.0 CPU, 3 GB    |
| ChromaDB | 8500  | 0.5 CPU, 512 MB  |

Total budget: **4 CPU cores, ~6 GB RAM** (edge deployment simulation).

## API Reference

### Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest-video` | Upload a video file (streaming, max 2 GB) |
| `POST` | `/process-video` | Trigger the 10-stage pipeline (async) |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search` | Natural language query over all videos |

Request body:
```json
{
  "query": "red car arriving at the entrance",
  "video_id": null,
  "top_k": 10,
  "source_filter": null
}
```

`source_filter` options: `"visual"`, `"audio"`, `"fused"`, `"event"`

### Video Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/videos` | List all ingested videos |
| `GET` | `/video/{id}` | Video metadata + processing status |
| `GET` | `/video/{id}/stream` | Stream video (supports range requests) |
| `GET` | `/video/{id}/frame/{name}` | Serve a keyframe image |
| `DELETE` | `/video/{id}` | Delete video and all associated data |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |

## Configuration

All settings are in `backend/config.py` and overridable via environment variables. See `.env.example` for the full list.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `moondream2` | Vision language model |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model size |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization for CPU |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model (384-dim) |
| `MAX_KEYFRAMES` | `200` | Global keyframe cap per video |
| `SHOT_THRESHOLD` | `0.45` | Shot boundary sensitivity |
| `FUSION_WINDOW_SEC` | `2.0` | Temporal window for visual-audio merge |
| `MAX_VIDEO_SIZE_MB` | `2048` | Upload size limit |
| `CHROMA_HOST` | `""` | Empty = embedded mode, set for Docker |

## Tech Stack

**Backend**: Python 3.11, FastAPI, OpenCV, FFmpeg, Faster-Whisper (CTranslate2), SentenceTransformers, ChromaDB, NetworkX, httpx

**Frontend**: React 19, TypeScript 5.9, Vite 7, Axios

**Infrastructure**: Docker Compose, Ollama, nginx

## Data Storage

```
data/
├── videos/          # Uploaded video files
├── frames/          # Extracted JPEG keyframes
├── audio/           # 16kHz mono WAV files
├── graphs/          # GraphML knowledge graphs
└── db/
    ├── cognistream.db   # SQLite (video metadata, status)
    └── chroma/          # ChromaDB (vector embeddings)
```

## License

MIT
