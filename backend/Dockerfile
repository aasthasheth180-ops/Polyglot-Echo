# ── Use a stable Python slim base image ────────────────────────
FROM python:3.10-slim

# ── Install essential Linux packages ──────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Set up container workspace ─────────────────────────────────
WORKDIR /app

# ── Copy and install Python requirements ───────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────
COPY . .

# ── Copy and make entrypoint script executable ─────────────────
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ── Use entrypoint script to handle Railway's $PORT injection ──
# This explicitly validates and passes $PORT to Uvicorn
ENTRYPOINT ["/app/entrypoint.sh"]