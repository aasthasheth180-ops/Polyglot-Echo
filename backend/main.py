import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Absolute imports for the backend package
from backend.pipeline import process_audio_loop
from backend.llm_engine import llm_engine
from backend.ai_client import synthesize_speech

load_dotenv()
app = FastAPI(title="Polyglot Echo", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files Mounting ──
static_dir = "/app/static"

# Serve API routes first
@app.post("/api/process-text")
async def process_text(request: TextRequest):
    session_id = str(uuid.uuid4())
    llm_result = llm_engine.generate(request.text, request.target_lang, session_id)
    audio_bytes = synthesize_speech(
        text=llm_result["text"],
        target_lang=request.target_lang,
        speaker_profile=request.speaker_profile
    )
    return Response(content=audio_bytes, media_type="audio/wav")

@app.post("/api/process-voice")
async def process_voice(file: UploadFile = File(...), target_lang: str = Form("en")):
    wav_bytes = await file.read()
    result = process_audio_loop(wav_bytes=wav_bytes, target_lang=target_lang)
    return Response(content=result["audio_bytes"], media_type="audio/wav")

# Serve Frontend
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

app.mount("/static", StaticFiles(directory=static_dir), name="static")