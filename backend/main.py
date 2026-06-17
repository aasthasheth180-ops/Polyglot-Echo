# backend/main.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from backend.pipeline import process_audio_loop
from backend.llm_engine import llm_engine
from backend.ai_client import check_colab_health
from pipeline.models import create_tables

# Initialize database schema tables on framework startup
create_tables()

app = FastAPI(title="Polyglot Echo", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

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
            "X-Latency-Total-MS": str(result["latency"]["total_ms"])
        }
    )

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    llm_engine.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

#@app.post("/api/upload-guest-reference")
#async def upload_guest_ref(audio_file: UploadFile = File(...)):
    """Upload guest voice reference to Colab AI server."""
    wav_bytes = await audio_file.read()
    success = upload_guest_reference(wav_bytes)
    return {"status": "uploaded" if success else "failed"}