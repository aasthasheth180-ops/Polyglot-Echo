# backend/main.py
import os
import uuid
import time
import asyncio
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environmental variables
load_dotenv()

# Internal import
from pipeline import process_audio_loop
from llm_engine import llm_engine
from ai_client import check_colab_health, synthesize_speech, wait_for_hf_space_ready
from stream_handler import handle_voice_stream

# ── Database Import ───────────────────────────────────────────
from models import create_tables, SessionLocal

# ── Global Flag to Track DB Status ─────────────────────────────
db_initialized = False
db_error = None

# ── 2. Instantiate the core FastAPI application object FIRST ───
app = FastAPI(title="Polyglot Echo", version="2.0")

# ── 3. Professor's Exact CORS Configuration (Fixed Wildcard Bug) ──
origins = [
    "https://polyglot-echo-production.up.railway.app",  # <--- Update this to match your real URL
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Safe list without wildcard conflict
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Session-Id",
        "X-Requested-With",
        "Accept",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
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
    ],
    max_age=3600  # preflight cache — browser won't re-check for 1 hour
)

# ── Professor's Explicit OPTIONS Handler (Fixed for Credentials) ──
@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    # Update this line to your current production URL
    origin = request.headers.get("Origin", "https://polyglot-echo-production.up.railway.app")
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Session-Id, X-Requested-With, Accept, Origin",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600"
        }
    )

# ── Data Models for REST Injections ────────────────────────────
class TextRequest(BaseModel):
    text:            str
    target_lang:     str = "en"
    speaker_profile: str = "aastha"


# ── 4. STARTUP HOOKS ───────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global db_initialized, db_error
    print("\n" + "="*70)
    print("[Backend Startup] Beginning initialization sequence...")
    print("="*70 + "\n")
    
    try:
        create_tables()
        db_initialized = True
        print("[Backend] ✅ PostgreSQL database ready\n")
    except Exception as e:
        db_error = str(e)
        print(f"[Backend] ⚠️  Database initialization failed: {e}")
        print("[Backend] ⚠️  Server will start, but /api endpoints will return 503 error\n")
    
    space_ready = await asyncio.to_thread(
        wait_for_hf_space_ready,
        max_attempts=30,
        delay_between_checks=2
    )
    if not space_ready:
        print("[Backend] ⚠️  HF Space still initializing. Requests may fail initially.")
    else:
        print("[Backend] ✅ HF Space ready.\n")


def require_db():
    if not db_initialized:
        raise HTTPException(
            status_code=503,
            detail=f"Database not initialized. Error: {db_error or 'Unknown error'}",
            headers={"Access-Control-Allow-Origin": "https://frontend-production-faa5.up.railway.app"}
        )


# ── Standard HTTP REST Endpoints ───────────────────────────────

@app.get("/health")
async def health():
    colab_ok = check_colab_health()
    return {
        "status": "ok",
        "database": "connected" if db_initialized else "disconnected",
        "colab_ai": "connected" if colab_ok else "disconnected",
        "db_error": db_error if db_error else None
    }


@app.post("/api/process-text")
async def process_text(
    request: TextRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    require_db()
    session_id = x_session_id or str(uuid.uuid4())

    llm_result = llm_engine.generate(
        request.text, request.target_lang, session_id
    )
    ai_text = llm_result["text"]
    llm_ms  = llm_result["latency_ms"]

    tts_start   = time.time()
    audio_bytes = synthesize_speech(
        text=ai_text,
        target_lang=request.target_lang,
        speaker_profile=request.speaker_profile,
        ref_text="Hey, how have you been lately?"
    )
    tts_ms    = int((time.time() - tts_start) * 1000)
    total_ms  = llm_ms + tts_ms

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Access-Control-Allow-Origin": "https://frontend-production-faa5.up.railway.app",
            "X-Response-Text":    ai_text,
            "X-Session-Id":       session_id,
            "X-LLM-MS":           str(llm_ms),
            "X-TTS-MS":           str(tts_ms),
            "X-Latency-Total-MS": str(total_ms)
        }
    )


@app.post("/api/process-voice")
async def process_voice(
    audio_file: UploadFile = File(...),
    target_lang: str = Form("en"),
    speaker_profile: str = Form("aastha"),
    reference_wav: str = Form(None),
    x_session_id: Optional[str] = Header(default=None)
):
    require_db()
    session_id = x_session_id or str(uuid.uuid4())
    wav_bytes = await audio_file.read()
    
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
            "Access-Control-Allow-Origin": "https://frontend-production-faa5.up.railway.app",
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


@app.websocket("/ws/stream-audio")
async def websocket_endpoint(websocket: WebSocket):
    if not db_initialized:
        await websocket.close(code=1008, reason="Database not initialized")
        return
    
    target_lang     = websocket.query_params.get("target_lang", "en")
    speaker_profile = websocket.query_params.get("speaker_profile", "aastha")
    session_id      = websocket.query_params.get("session_id", None)

    await handle_voice_stream(
        websocket=websocket,
        target_lang=target_lang,
        speaker_profile=speaker_profile,
        session_id=session_id
    )


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    require_db()
    llm_engine.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}