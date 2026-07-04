FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 1. Install System Dependencies (Needed for Kafka/Postgres drivers)
RUN apt-get update && apt-get install -y \
    python3 \
    git \
    python3-pip \
    libpq-dev \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Environment Variables
WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    COQUI_TOS_AGREED=1

# 3. Install Python Dependencies
COPY requirements.hf.txt /code/requirements.txt 
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# 4. Copy Code
COPY . /code/

# 5. Launch
CMD ["python3", "app.py"]