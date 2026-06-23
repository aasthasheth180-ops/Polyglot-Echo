import os
import uuid
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import internal modules safely
from backend.pipeline import process_audio_loop
from backend.llm_engine import llm_engine
from backend.ai_client import synthesize_speech, check_colab_health

load_dotenv()
app = FastAPI(title="Polyglot Echo", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    target_lang: str = "en"
    speaker_profile: str = "aastha"

@app.post("/api/process-text")
async def process_text(request: TextRequest):
    session_id = str(uuid.uuid4())
    llm_result = llm_engine.generate(request.text, request.target_lang, session_id)
    audio_bytes = synthesize_speech(
        text=llm_result["text"],
        target_lang=request.target_lang,
        speaker_profile=request.speaker_profile,
        ref_text="Hey, how have you been lately?"
    )
    return Response(content=audio_bytes, media_type="audio/wav")

@app.post("/api/process-voice")
async def process_voice(
    file: UploadFile = File(...),
    target_lang: str = Form("en"),
    speaker_profile: str = Form("aastha")
):
    wav_bytes = await file.read()
    result = process_audio_loop(
        wav_bytes=wav_bytes,
        target_lang=target_lang,
        speaker_profile=speaker_profile,
        session_id=str(uuid.uuid4())
    )
    return Response(content=result["audio_bytes"], media_type="audio/wav")

# Serve Frontend
if os.path.exists("/app/static"):
    app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")