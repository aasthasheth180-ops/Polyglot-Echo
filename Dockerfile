FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Environment Variables
WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COQUI_TOS_AGREED=1 \
    PYTHONPATH=/code

# 3. Install Python Dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.hf.txt /code/requirements.txt 
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# 4. Copy Code
COPY app.py /code/app.py
COPY backend/ /code/backend/ 
RUN mkdir -p /code/logs

# 5. Launch
CMD ["python3", "app.py"]