#!/bin/bash
# ─────────────────────────────────────────────────────────
# CogniStream — Model Pre-Pull Script
#
# Downloads all required ML models for offline operation.
# Run this ONCE on a machine with internet access before
# going fully offline.
#
# Usage:
#   bash scripts/pull_models.sh
# ─────────────────────────────────────────────────────────

set -e

# Detect script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Detect Python command - prefer venv if available
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
    echo "Using virtual environment Python"
elif [ -n "$VIRTUAL_ENV" ]; then
    # Venv is already activated
    PYTHON=python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python or create a virtual environment at .venv/"
    exit 1
fi

echo "=== CogniStream Model Setup ==="
echo "Using Python: $PYTHON"
echo ""

# 1. Ollama VLM model
echo "[1/3] Pulling Ollama model: qwen2.5vl:3b..."
if command -v ollama &> /dev/null; then
    ollama pull qwen2.5vl:3b
    echo "  Done."
else
    echo "  SKIP: ollama not found. It will be pulled inside Docker."
fi

# 2. SentenceTransformers embedding model
echo "[2/3] Downloading SentenceTransformer: BAAI/bge-small-en-v1.5..."
$PYTHON -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('  Downloaded to:', model._model_card_vars.get('model_id', 'cache'))
"
echo "  Done."

# 3. Faster-Whisper model
echo "[3/3] Downloading Faster-Whisper: small (int8)..."
$PYTHON -c "
from faster_whisper import WhisperModel
model = WhisperModel('small', device='cpu', compute_type='int8')
print('  Model loaded successfully.')
"
echo "  Done."

echo ""
echo "=== All models ready for offline operation ==="
