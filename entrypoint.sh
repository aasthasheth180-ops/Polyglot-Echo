#!/bin/bash
# ── Railway Port Entrypoint with PYTHONPATH Fix ────────────────────────────
set -e

# ── CRITICAL: Set PYTHONPATH ────────────────────────────────────────────────
# This tells Python where to find the 'backend' module
export PYTHONPATH=/app:$PYTHONPATH

# ── Port Configuration ──────────────────────────────────────────────────────
PORT=${PORT:-8080}

# Validate PORT is a number
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "[Startup] Warning: PORT='$PORT' is invalid, using 8080"
    PORT=8080
fi

# ── Startup Information ────────────────────────────────────────────────────
echo "[Startup] ════════════════════════════════════════════════════"
echo "[Startup] Railway Polyglot Echo Backend - Startup Sequence"
echo "[Startup] ════════════════════════════════════════════════════"
echo "[Startup] Detected PORT: $PORT"
echo "[Startup] PYTHONPATH: $PYTHONPATH"
echo "[Startup] Working Directory: $(pwd)"
echo "[Startup] Python Version: $(python --version)"
echo "[Startup] App Directory Contents:"
ls -la /app/backend/ 2>/dev/null | head -10 || echo "[Startup] ⚠️  No backend directory found"

# ── Verify Module Imports ──────────────────────────────────────────────────
echo "[Startup] Verifying module imports..."

python -c "
import sys
print('[Startup] Python path:', sys.path[:3])
try:
    from backend.main import app
    print('[Startup] ✅ Successfully imported: backend.main')
    print('[Startup] ✅ FastAPI app found:', type(app).__name__)
except ImportError as e:
    print('[Startup] ❌ Failed to import backend.main')
    print('[Startup] Error:', str(e))
    sys.exit(1)
" || {
    echo "[Startup] ❌ Import verification failed!"
    echo "[Startup] Debugging information:"
    python -c "import sys; print('  Python path:', sys.path)" 2>/dev/null || true
    python -c "import os; print('  Backend files:', os.listdir('/app/backend'))" 2>/dev/null || true
    exit 1
}

# ── Start Uvicorn ──────────────────────────────────────────────────────────
echo "[Startup] ════════════════════════════════════════════════════"
echo "[Startup] Starting Uvicorn ASGI server..."
echo "[Startup] Command: python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
echo "[Startup] ════════════════════════════════════════════════════"
echo ""

exec python -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info