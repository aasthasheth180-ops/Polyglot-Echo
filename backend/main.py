# backend/main.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Internal imports
from backend.pipeline import process_audio_loop
from backend.llm_engine import llm_engine
from backend.ai_client import check_colab_health
from pipeline.models import create_tables
from backend.stream_handler import handle_voice_stream

# 1. Initialize database schema tables on framework startup
create_tables()

# 2. Instantiate the core FastAPI application object FIRST
app = FastAPI(title="Polyglot Echo", version="2.0")

# 3. Add global network middleware configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── WebSocket Live Streaming Gateway ───────────────────────────
@app.websocket("/ws/stream-audio")
async def websocket_endpoint(websocket: WebSocket):
    """Handles persistent real-time dual-streaming connections."""
    # Read config from query params so frontend can set language etc.
    target_lang     = websocket.query_params.get("target_lang", "en")
    speaker_profile = websocket.query_params.get("speaker_profile", "aastha")
    session_id      = websocket.query_params.get("session_id", None)

    await handle_voice_stream(
        websocket=websocket,
        target_lang=target_lang,
        speaker_profile=speaker_profile,
        session_id=session_id
    )

# ── Standard HTTP REST Endpoints ───────────────────────────────
@app.get("/health")
async def health():
    # Hits the updated check_colab_health that handles the ngrok security headers
    colab_ok = check_colab_health()
    return {
        "status": "ok",
        "colab_ai": "connected" if colab_ok else "disconnected"
    }

@app.post("/api/process-voice")
async def process_voice(
    audio_file: UploadFile = File(...),
    target_lang: str = Form("en"),
    speaker_profile: str = Form("aastha"),
    reference_wav: str = Form(None),
    x_session_id: Optional[str] = Header(default=None) # Extracts session id from incoming network packets
):
    session_id = x_session_id or str(uuid.uuid4())
    wav_bytes = await audio_file.read()
    
    # Passes session_id straight down into process_audio_loop
    result = process_audio_loop(
        wav_bytes=wav_bytes,
        target_lang=target_lang,
        speaker_profile=speaker_profile,
        reference_wav=reference_wav,
        session_id=session_id   
    )

    return Response(
        content=result["audio_bytes"],
        media_type="audio/wav",
        headers={
            "X-Transcript":       result["transcript"],
            "X-Response-Text":    result["response_text"],
            "X-Detected-Lang":    result["detected_lang"],
            "X-Speaker-Profile":  speaker_profile,
            "X-Session-Id":       session_id,
            "X-Whisper-MS":       str(result["latency"]["whisper_ms"]),
            "X-LLM-MS":           str(result["latency"]["llm_ms"]),
            "X-TTS-MS":           str(result["latency"]["tts_ms"]),
            "X-Latency-Total-MS": str(result["latency"]["total_ms"])
        }
    )

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    llm_engine.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}