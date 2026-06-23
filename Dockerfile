# ── Production Dockerfile for Polyglot Echo ────────────────────
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

# ── CRITICAL FIX: Set PYTHONPATH so imports work ───────────────
ENV PYTHONPATH=/app

# ── Copy Python requirements and install ───────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy backend source code ───────────────────────────────────
COPY backend/ /app/backend/

# ── Copy frontend static files ─────────────────────────────────
# ── Copy frontend static files ─────────────────────────────────
RUN mkdir -p /app/static
# Change from 'frontend/' to 'backend/static/'
COPY backend/static/ /app/static/

# ── Copy entrypoint script ─────────────────────────────────────
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ── Expose port ────────────────────────────────────────────────
EXPOSE 8080

# ── Run the application ────────────────────────────────────────
ENTRYPOINT ["/app/entrypoint.sh"]