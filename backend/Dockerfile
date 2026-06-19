# ── Use a stable Python slim base image ────────────────────────
FROM python:3.10-slim

# ── Install essential Linux C-compiler & audio system packages ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Set up container workspace ────────────────────────────────
WORKDIR /app

# ── Copy and install Python requirement modules ───────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy the rest of your backend execution source code ───────
COPY . .

# ── Expose port matching Railway execution arguments ──────────
EXPOSE 8000

# ── Launch production server via live Uvicorn bindings ────────
CMD uvicorn main:app --host 0.0.0.0 --port $PORT