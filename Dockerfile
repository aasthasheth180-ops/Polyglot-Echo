# ── Production Dockerfile for Polyglot Echo ───────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# ── Install System Dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy Python Requirements and Install ───────────────────────────────────
# ── Copy Python Requirements and Install ───────────────────────────────────
# Copy only the requirements first to leverage Docker layer caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Backend Source Code ───────────────────────────────────────────────
COPY backend/ ./backend/

# ── Create __init__.py in backend (CRITICAL for imports) ─────────────────
RUN touch /app/backend/__init__.py

# ── Create Required Directories ────────────────────────────────────────────
RUN mkdir -p /data && chmod 777 /data

# ── Copy Entrypoint Script ────────────────────────────────────────────────
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# ── Set PYTHONPATH Environment Variable ────────────────────────────────────
# This tells Python where to find modules (CRITICAL for Railway)
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# ── Expose Port ────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Health Check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# ── Run Application ───────────────────────────────────────────────────────
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
