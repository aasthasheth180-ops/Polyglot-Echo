### 📄 `README.md`

```markdown
# Polyglot Echo v2.0: Low-Latency Multilingual Streaming Voice Assistant

Polyglot Echo is an enterprise-grade, real-time conversational AI voice pipeline that enables full-duplex, multilingual audio streaming. By decoupling heavy machine learning orchestration from the client web interface, the architecture achieves minimal Time-To-First-Audio (TTFA) using asynchronous worker queues, WebSockets, and edge digital signal processing (DSP).

The system seamlessly switches between processing static full-file voice uploads via standard REST endpoints and managing highly responsive, real-time audio streams via persistent WebSocket channels.

---

## 🏗️ System Architecture

The pipeline is split into a lightweight local orchestration hub (Laptop) and a high-performance cloud acceleration layer (Google Colab GPU Nodes) connected via high-speed, secure reverse-proxy tunnels.



### ⚡ Key Architectural Enhancements
Full-Duplex Async Streaming: Audio is transferred from the browser in 100ms frames using raw binary WebSockets, eliminating the overhead of standard file compilation.
Edge Audio Preprocessing (DSP Matrix): Applies an 80Hz High-Pass Butterworth filter to remove low-frequency microphone rumble and a 6000Hz IIR Notch filter to eliminate sibilance/hiss prior to transcription.
VAD Noise Gate: Built-in Root-Mean-Square (RMS) volume gate set aggressively (`0.025`) alongside text-matching regularizers to discard Whisper static hallucinations (such as ghost "Thank you" loops).
Sliding-Window Memory Vault: Implements an internal memory manager tracking the last k=5 conversation turns natively inside the context prompt to prevent token bloat while maintaining strict multi-turn conversational accuracy.
Decoupled Metric Tracking: Emits lifecycle tracking event payloads down to an Apache Kafka cluster to record sub-component processing latencies without blocking the active voice path.

---

## 🛠️ Tech Stack & Infrastructure

Backend Framework: FastAPI (Python 3.10) with Asynchronous Coroutines (`asyncio`).
Signal Processing: NumPy, SciPy (Signal Processing Sub-module).
LLM Engine: Google GenAI SDK (`gemini-2.5-flash`).
Cloud AI Inference Tunnels: OpenAI Whisper (ASR), F5-TTS (Voice Synthesis) via remote GPU worker threads.
Event Streaming: Apache Kafka.
Caching & Analytics: Redis (Audio Payload Cache), PostgreSQL (Pipeline Turn Telemetry).
Containerization: Docker & Docker Compose.

---

## 📂 Repository Structure

```text
polyglot-echo/
├── backend/
│   ├── ai_client.py        # Connects network bridges out to cloud GPU nodes
│   ├── cache.py            # Handles Redis audio payload string key/value pairs
│   ├── llm_engine.py       # Manages Gemini streaming client & memory window matrix
│   ├── main.py             # FastAPI entry point hosting REST & WebSocket gateways
│   ├── pipeline.py         # Standard full-file batch ingestion pipeline loop
│   └── stream_handler.py   # Asynchronous WebSocket workers, DSP matrix & VAD gate
├── pipeline/
│   ├── kafka_producer.py   # Fires event state packets to the streaming clusters
│   └── models.py           # Implements database tracking schemas
├── infra/
│   └── docker-compose.yml  # Spins up local Kafka, Postgres, and Redis containers
└── frontend/
    └── stream_test.html    # Raw HTML5 WebAudio API real-time testing sandbox

```

---

## 🚀 Deployment & Execution Guide

### 1. Initialize Distributed Infrastructure

Spin up the local message brokers and caching instances using Docker Compose:

```bash
docker compose -f infra/docker-compose.yml up -d

```

### 2. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GEMINI_API_KEY="your-google-gemini-api-key"
COLAB_AI_URL="[https://your-active-ngrok-tunnel-url.ngrok-free.app](https://your-active-ngrok-tunnel-url.ngrok-free.app)"
REDIS_HOST="localhost"
KAFKA_BOOTSTRAP_SERVERS="localhost:9092"

```

### 3. Setup Virtual Environment & Dependencies

```bash
python -m venv venv310
source venv310/bin/activate  # On Windows: .\venv310\Scripts\activate

# Install requirements (ensure numpy is bounded to avoid binary compatibility errors)
pip install -r requirements.txt
pip install "numpy<2.0.0"

```

### 4. Boot Up the FastApi Application Core

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

```

Look for the validation indicator: `[✓] LLM Engine: Gemini + Native Sliding-Window Memory Online`.

### 5. Open the Frontend Sandbox Interface

Expose the static testing sandbox locally using Python's built-in HTTP server module:

```bash
python -m http.server 8080 --directory frontend

```

Navigate to `http://localhost:8080/stream_test.html` in your browser, select your language parameter matrix, hit **Connect**, and hold down the microphone button to initiate a live voice stream.

---

## 📊 Performance & Latency Metrics

| Segment Pipeline Stage | Processing Infrastructure | Average Streaming Latency |
| --- | --- | --- |
| **Edge Filtering & VAD Gate** | Local Laptop CPU (Async Loop) | ~2ms - 5ms |
| **Whisper Transcription** | Remote Cloud GPU Tunnel | ~200ms - 400ms |
| **Gemini Token Generation** | Official Google GenAI API | ~50ms (First Chunk) |
| **F5-TTS Audio Matrix Synthesis** | Remote Cloud GPU Tunnel | ~350ms - 600ms (Sentence Blocks) |
| **Total System Turn (TTFA)** | **End-to-End Dynamic Stream** | **< 1.0 Second** |

```

