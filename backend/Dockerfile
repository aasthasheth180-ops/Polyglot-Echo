FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11, core tools, AND the system audio/compiler libraries needed for TTS and Whisper
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    build-essential \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# Pre-download whisper small model so container starts fast
RUN python3 -c "import whisper; whisper.load_model('small')"

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]