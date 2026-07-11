FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 1. Install System Dependencies
# FIX: openai-whisper shells out to the `ffmpeg` binary at runtime to decode
# audio — without it, every /transcribe call fails inside the container.
# curl is used by entrypoint.sh to wait for the worker process to come up.
RUN apt-get update && apt-get install -y \
    python3 \
    git \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Environment Variables
WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COQUI_TOS_AGREED=1 \
    PYTHONPATH=/code

# 3. Install Python Dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# 4. Copy Code
COPY app.py /code/app.py
COPY backend/ /code/backend/
COPY entrypoint.sh /code/entrypoint.sh
RUN chmod +x /code/entrypoint.sh
RUN mkdir -p /code/logs

# 5. Launch BOTH processes (worker + public API) — see entrypoint.sh
CMD ["/code/entrypoint.sh"]
