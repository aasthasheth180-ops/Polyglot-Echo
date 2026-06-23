import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Internal imports
from backend.pipeline import process_audio_loop
from backend.llm_engine import llm_engine
from backend.ai_client import synthesize_speech

app = FastAPI(title="Polyglot Echo", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = "/app/static"

class TextRequest(BaseModel):
    text: str
    target_lang: str = "en"
    speaker_profile: str = "aastha"

# 1. Text Processing Route
@app.post("/api/process-text")
async def process_text(request: TextRequest, x_session_id: Optional[str] = Header(default=None)):
    session_id = x_session_id or str(uuid.uuid4())
    llm_result = llm_engine.generate(request.text, request.target_lang, session_id)
    
    audio_bytes = synthesize_speech(
        text=llm_result["text"],
        lang=request.target_lang,
        speaker_profile=request.speaker_profile
    )
    
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Response-Text": llm_result["text"],
            "X-Latency-Total-MS": str(llm_result["latency_ms"])
        }
    )

# 2. Voice Processing Route
@app.post("/api/process-voice")
async def process_voice(file: UploadFile = File(...), target_lang: str = Form("en")):
    wav_bytes = await file.read()
    result = process_audio_loop(wav_bytes=wav_bytes, target_lang=target_lang)
    return Response(content=result["audio_bytes"], media_type="audio/wav")

# 3. Guest Voice Reference Upload
@app.post("/api/upload-guest-reference")
async def upload_guest_reference(audio_file: UploadFile = File(...)):
    # Save the reference file to a temporary location for the F5-TTS model
    with open("/tmp/guest_ref.wav", "wb") as buffer:
        buffer.write(await audio_file.read())
    return {"status": "success"}

# 4. Static Frontend Routing
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

app.mount("/static", StaticFiles(directory=static_dir), name="static")