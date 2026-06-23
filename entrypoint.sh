#!/bin/bash
# ── Railway Port Entrypoint ────────────────────────────────────
# This script explicitly handles Railway's dynamic $PORT injection
# and prevents shell expansion errors in the Uvicorn worker process.

set -e

# If PORT is not set, default to 8000
PORT=${PORT:-8000}

# Validate that PORT is an integer
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid PORT value: $PORT (must be a positive integer)"
    echo "[INFO] Defaulting to PORT=8000"
    PORT=8000
fi

echo "[Startup] Railway detected PORT=$PORT"
echo "[Startup] Starting Uvicorn worker..."

# Execute Uvicorn with the validated PORT
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info