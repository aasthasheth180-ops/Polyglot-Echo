#!/bin/bash
# ── Railway Port Entrypoint ────────────────────────────────────
set -e

# If PORT is not set, default to 8000
PORT=${PORT:-8080}

# Validate that PORT is an integer
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid PORT value: $PORT (must be a positive integer)"
    PORT=8080
fi

echo "[Startup] Railway detected PORT=$PORT"
echo "[Startup] Starting Uvicorn worker..."

# Execute Uvicorn using the path to the file inside the backend folder
# We use 'backend.main:app' to tell Uvicorn to look inside the backend package
exec uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info