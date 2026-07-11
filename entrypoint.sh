#!/bin/bash
set -e

# ── Two processes, one container ──────────────────────────────────────
# app.py   = GPU worker (Whisper + F5-TTS) — internal only, not exposed
# main.py  = public API + serves index.html + talks to Gemini — this is
#            what HF Space's single exposed port must point at
#
# ai_client.py / config.py read AI_SERVICE_URL from the environment, so
# pointing it at localhost here is enough — no code change needed there.

export WORKER_PORT=${WORKER_PORT:-8001}
export PORT=${PORT:-7860}
export AI_SERVICE_URL="http://127.0.0.1:${WORKER_PORT}"

echo "[Entrypoint] Starting AI worker (app.py) on internal port ${WORKER_PORT}..."
PORT=${WORKER_PORT} python3 /code/app.py &
WORKER_PID=$!

echo "[Entrypoint] Waiting for worker to come up..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${WORKER_PORT}/health" > /dev/null 2>&1; then
    echo "[Entrypoint] Worker is responding (models still load lazily on first real request)."
    break
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "[Entrypoint] Worker process died during startup — check logs above."
    exit 1
  fi
  sleep 2
done

echo "[Entrypoint] Starting main API (main.py) on public port ${PORT}..."
cd /code/backend
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "${PORT}"