# ── Production Dockerfile for Polyglot Echo ────────────────────
# This builds a SINGLE container with:
# - Backend (FastAPI on port 8000)
# - Frontend (Static files served by FastAPI)

FROM python:3.10-slim

# ── Install system dependencies ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ──────────────────────────────────────
WORKDIR /app

# ── Copy Python requirements and install ───────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy backend source code ───────────────────────────────────
COPY backend/ /app/backend/

# ── Copy frontend static files into a 'static' directory ───────
# This is where FastAPI will serve them from
RUN mkdir -p /app/static
COPY frontend/index.html /app/static/index.html
COPY frontend/ /app/static/

# ── Copy entrypoint script ─────────────────────────────────────
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ── Expose port ────────────────────────────────────────────────
EXPOSE 8000

# ── Run the application ────────────────────────────────────────
ENTRYPOINT ["/app/entrypoint.sh"]