import os
import uuid
import time
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Internal imports (Ensure these files exist in your repo)
from pipeline import process_audio_loop
from llm_engine import llm_engine
from ai_client import check_colab_health, synthesize_speech, wait_for_hf_space_ready
from models import create_tables

# ── Setup ──────────────────────────────────────────────────────
app = FastAPI(title="Polyglot Echo", version="2.0")
PRODUCTION_URL = "https://polyglot-echo-production.up.railway.app"

# ── CORS Middleware ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PRODUCTION_URL, "http://localhost:3000", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ── CORS Helper ────────────────────────────────────────────────
def get_cors_headers():
    return {
        "Access-Control-Allow-Origin": PRODUCTION_URL,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Expose-Headers": "X-Transcript, X-Response-Text, X-Detected-Lang, X-Session-Id, X-Whisper-MS, X-LLM-MS, X-TTS-MS, X-Latency-Total-MS, X-Speaker-Profile"
    }

# ── API Endpoints ──────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str
    target_lang: str = "en"
    speaker_profile: str = "aastha"

@app.post("/api/process-text")
async def process_text(request: TextRequest, x_session_id: Optional[str] = Header(default=None)):
    session_id = x_session_id or str(uuid.uuid4())
    llm_result = llm_engine.generate(request.text, request.target_lang, session_id)
    
    tts_start = time.time()
    audio_bytes = synthesize_speech(
        text=llm_result["text"],
        target_lang=request.target_lang,
        speaker_profile=request.speaker_profile,
        ref_text="Hey, how have you been lately?"
    )
    
    headers = get_cors_headers()
    headers.update({
        "X-Response-Text": llm_result["text"],
        "X-Session-Id": session_id,
        "X-LLM-MS": str(llm_result["latency_ms"]),
        "X-TTS-MS": str(int((time.time() - tts_start) * 1000)),
        "X-Latency-Total-MS": str(llm_result["latency_ms"] + int((time.time() - tts_start) * 1000))
    })
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)

@app.post("/api/process-voice")
async def process_voice(
    audio_file: UploadFile = File(...),
    target_lang: str = Form("en"),
    speaker_profile: str = Form("aastha"),
    reference_wav: str = Form(None),
    x_session_id: Optional[str] = Header(default=None)
):
    session_id = x_session_id or str(uuid.uuid4())
    wav_bytes = await audio_file.read()
    result = process_audio_loop(wav_bytes, target_lang, speaker_profile, reference_wav, session_id)
    
    latencies = result.get("latency", {})
    headers = get_cors_headers()
    headers.update({
        "X-Transcript": str(result.get("transcript", "")),
        "X-Response-Text": str(result.get("response_text", "")),
        "X-Detected-Lang": str(result.get("detected_lang", "")),
        "X-Speaker-Profile": speaker_profile,
        "X-Session-Id": session_id,
        "X-Whisper-MS": str(latencies.get("whisper_ms", 0)),
        "X-LLM-MS": str(latencies.get("llm_ms", 0)),
        "X-TTS-MS": str(latencies.get("tts_ms", 0)),
        "X-Latency-Total-MS": str(latencies.get("total_ms", 0))
    })
    return Response(content=result["audio_bytes"], media_type="audio/wav", headers=headers)

# ── Static File Mounting ───────────────────────────────────────
# Place this at the very end. Ensure index.html exists in your root folder.
app.mount("/", StaticFiles(directory=".", html=True), name="static")