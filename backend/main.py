# backend/main.py
import os
import uuid
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# ── Internal imports ───────────────────────────────────────────
try:
    from pipeline import process_audio_loop
except ImportError:
    print("[WARNING] Could not import process_audio_loop, will fail on voice endpoints")

try:
    from llm_engine import llm_engine
except ImportError:
    print("[WARNING] Could not import llm_engine, will fail on text endpoints")

try:
    from ai_client import synthesize_speech, check_colab_health
except ImportError:
    print("[WARNING] Could not import ai_client, will fail on synthesis")

# ── Initialize FastAPI app ────────────────────────────────────
app = FastAPI(title="Polyglot Echo", version="2.0")

# ── CORS Middleware ───────────────────────────────────────────
# Allow cross-origin requests from anywhere during development
# For production, restrict to your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify: ["https://polyglot-echo-production.up.railway.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Transcript",
        "X-Response-Text",
        "X-Detected-Lang",
        "X-Session-Id",
        "X-Whisper-MS",
        "X-LLM-MS",
        "X-TTS-MS",
        "X-Latency-Total-MS",
        "X-Speaker-Profile"
    ]
)

# ── Data Models ────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    target_lang: str = "en"
    speaker_profile: str = "aastha"

# ── Health Check Endpoint ──────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        colab_ok = check_colab_health()
    except:
        colab_ok = False
    
    return {
        "status": "ok",
        "service": "polyglot-echo",
        "colab_ai": "connected" if colab_ok else "disconnected"
    }

# ── API Endpoints ──────────────────────────────────────────────

@app.post("/api/process-text")
async def process_text(
    request: TextRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    """Process text input and return synthesized speech."""
    session_id = x_session_id or str(uuid.uuid4())
    
    try:
        # Generate LLM response
        llm_result = llm_engine.generate(
            request.text,
            request.target_lang,
            session_id
        )
        
        # Synthesize speech
        tts_start = time.time()
        audio_bytes = synthesize_speech(
            text=llm_result["text"],
            target_lang=request.target_lang,
            speaker_profile=request.speaker_profile,
            ref_text="Hey, how have you been lately?"
        )
        tts_ms = int((time.time() - tts_start) * 1000)
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Response-Text": llm_result["text"],
                "X-Session-Id": session_id,
                "X-LLM-MS": str(llm_result["latency_ms"]),
                "X-TTS-MS": str(tts_ms),
                "X-Latency-Total-MS": str(llm_result["latency_ms"] + tts_ms)
            }
        )
    except Exception as e:
        print(f"[Error] process_text failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-voice")
async def process_voice(
    audio_file: UploadFile = File(...),
    target_lang: str = Form("en"),
    speaker_profile: str = Form("aastha"),
    reference_wav: str = Form(None),
    x_session_id: Optional[str] = Header(default=None)
):
    """Process voice input and return response."""
    session_id = x_session_id or str(uuid.uuid4())
    
    try:
        wav_bytes = await audio_file.read()
        
        result = process_audio_loop(
            wav_bytes=wav_bytes,
            target_lang=target_lang,
            speaker_profile=speaker_profile,
            reference_wav=reference_wav,
            session_id=session_id
        )
        
        latencies = result.get("latency", {})
        
        return Response(
            content=result["audio_bytes"],
            media_type="audio/wav",
            headers={
                "X-Transcript": str(result.get("transcript", "")),
                "X-Response-Text": str(result.get("response_text", "")),
                "X-Detected-Lang": str(result.get("detected_lang", "")),
                "X-Speaker-Profile": speaker_profile,
                "X-Session-Id": session_id,
                "X-Whisper-MS": str(latencies.get("whisper_ms", 0)),
                "X-LLM-MS": str(latencies.get("llm_ms", 0)),
                "X-TTS-MS": str(latencies.get("tts_ms", 0)),
                "X-Latency-Total-MS": str(latencies.get("total_ms", 0))
            }
        )
    except Exception as e:
        print(f"[Error] process_voice failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Static Files Mounting ──────────────────────────────────────
# This serves the frontend index.html and any other static assets
# The path '/app/static' is where the Dockerfile copies frontend files
static_dir = "/app/static"

if os.path.exists(static_dir):
    print(f"[FastAPI] Mounting static files from {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    print(f"[WARNING] Static directory not found at {static_dir}")
    print(f"[WARNING] Frontend will not be served. Available path contents:")
    if os.path.exists("/app"):
        print(f"[WARNING] /app contents: {os.listdir('/app')}")