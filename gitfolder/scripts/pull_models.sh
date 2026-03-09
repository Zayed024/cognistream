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

echo "=== CogniStream Model Setup ==="
echo ""

# 1. Ollama model
echo "[1/3] Pulling Ollama model: moondream2..."
if command -v ollama &> /dev/null; then
    ollama pull moondream2
    echo "  Done."
else
    echo "  SKIP: ollama not found. It will be pulled inside Docker."
fi

# 2. SentenceTransformers embedding model
echo "[2/3] Downloading SentenceTransformer: all-MiniLM-L6-v2..."
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('  Downloaded to:', model._model_card_vars.get('model_id', 'cache'))
"
echo "  Done."

# 3. Faster-Whisper model
echo "[3/3] Downloading Faster-Whisper: base (int8)..."
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('base', device='cpu', compute_type='int8')
print('  Model loaded successfully.')
"
echo "  Done."

echo ""
echo "=== All models ready for offline operation ==="
