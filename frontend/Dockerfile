FROM python:3.11-slim

WORKDIR /app

# Install standard Linux system utilities for network and audio checking
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Copy from the immediate execution directory context
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all front-end interface layout assets and scripts directly
COPY . .

# Expose Streamlit's default networking port
EXPOSE 8501

# Clean internal health monitoring to track active container performance
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the data application layer securely bound to Railway's routing variables
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]