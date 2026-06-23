import os
import uuid
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from backend.pipeline import process_audio_loop
from backend.llm_engine import llm_engine
from backend.ai_client import synthesize_speech

app = FastAPI(title="Polyglot Echo", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TextRequest(BaseModel):
    text: str
    target_lang: str = "en"
    speaker_profile: str = "aastha"

# --- NEW: Process Text Endpoint ---
@app.post("/api/process-text")
async def process_text(request: TextRequest, x_session_id: Optional[str] = Header(default=None)):
    session_id = x_session_id or str(uuid.uuid4())
    llm_result = llm_engine.generate(request.text, request.target_lang, session_id)
    
    tts_start = time.time()
    audio_bytes = synthesize_speech(
        text=llm_result["text"],
        lang=request.target_lang,
        speaker_profile=request.speaker_profile
    )
    tts_ms = int((time.time() - tts_start) * 1000)
    
    return Response(content=audio_bytes, media_type="audio/wav", headers={
        "X-Response-Text": llm_result["text"],
        "X-Latency-Total-MS": str(llm_result["latency_ms"] + tts_ms)
    })

# --- NEW: Guest Voice Upload ---
@app.post("/api/upload-guest-reference")
async def upload_guest_reference(audio_file: UploadFile = File(...)):
    # Save to /tmp for F5-TTS to pick up
    with open("/tmp/guest_ref.wav", "wb") as buffer:
        buffer.write(await audio_file.read())
    return {"status": "success"}

# --- Existing Voice Processing ---
@app.post("/api/process-voice")
async def process_voice(file: UploadFile = File(...), target_lang: str = Form("en")):
    result = process_audio_loop(wav_bytes=await file.read(), target_lang=target_lang)
    return Response(content=result["audio_bytes"], media_type="audio/wav")

# --- Static Routing ---

@app.get("/", response_class=HTMLResponse)
async def read_index():
    # This points to index.html in the same directory as main.py
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()