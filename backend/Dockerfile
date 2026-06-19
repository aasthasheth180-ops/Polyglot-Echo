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

# ── DO NOT hardcode port — Railway injects $PORT at runtime ────
# Use shell form CMD (not exec form) so $PORT variable expands
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]