#!/bin/bash
# ─────────────────────────────────────────────────────────
# CogniStream — Ollama Entrypoint
#
# Starts the Ollama server, waits for it to be ready, then
# pulls the configured model (if not already cached).
# After pull completes, the model is warm and ready to serve.
# ─────────────────────────────────────────────────────────

MODEL="${OLLAMA_MODEL:-moondream}"

echo "[CogniStream] Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for the server to become responsive
echo "[CogniStream] Waiting for Ollama to be ready..."
for i in $(seq 1 60); do
    if ollama list >/dev/null 2>&1; then
        echo "[CogniStream] Ollama is ready."
        break
    fi
    sleep 1
done

# Pull the model if not already present (non-fatal — backend retries)
if ollama list | grep -q "$MODEL"; then
    echo "[CogniStream] Model '$MODEL' already available."
else
    echo "[CogniStream] Pulling model '$MODEL'..."
    if ollama pull "$MODEL"; then
        echo "[CogniStream] Model '$MODEL' pulled successfully."
    else
        echo "[CogniStream] WARNING: Failed to pull '$MODEL'. Backend will retry on first request."
    fi
fi

echo "[CogniStream] Ollama server is running."

# Keep the server in the foreground
wait $SERVER_PID
