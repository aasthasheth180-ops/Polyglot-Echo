FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COQUI_TOS_AGREED=1 \
    PYTHONPATH=/code  # <--- CRITICAL: This makes 'backend/' visible

# ... (Keep your existing apt-get and pip install commands) ...

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.hf.txt /code/requirements.txt 
RUN pip install --no-cache-dir -r /code/requirements.txt

# Explicitly copy the application code
COPY app.py /code/app.py
COPY backend/ /code/backend/ # <--- CRITICAL: Copy the backend folder!
RUN mkdir -p /code/audio /code/logs

# Final Command to launch the app
CMD ["python", "app.py"] # <--- Ensure this is at the end