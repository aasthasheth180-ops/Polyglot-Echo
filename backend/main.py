# backend/main.py
import os
import uuid
import time
import asyncio
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environmental variables
load_dotenv()

# Internal import
from pipeline import process_audio_loop
from llm_engine import llm_engine
from ai_client import check_colab_health, synthesize_speech, wait_for_hf_space_ready
from stream_handler import handle_voice_stream

# ── Database Import (but DON'T call create_tables yet) ────────
from pipeline_events.models import create_tables, SessionLocal

# ── Global Flag to Track DB Status ─────────────────────────────
db_initialized = False
db_error = None

# ── 2. Instantiate the core FastAPI application object FIRST ───
# (BEFORE any database calls)
app = FastAPI(title="Polyglot Echo", version="2.0")

# ── 3. Add global network middleware configurations ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Data Models for REST Injections ────────────────────────────
class TextRequest(BaseModel):
    text:            str
    target_lang:     str = "en"
    speaker_profile: str = "aastha"


# ── 4. STARTUP HOOKS (runs when server starts, not at import time) ──
@app.on_event("startup")
async def startup_event():
    """
    Initialize critical services at startup time (not import time).
    This prevents the entire process from crashing if a service is temporarily down.
    """
    global db_initialized, db_error
    
    print("\n" + "="*70)
    print("[Backend Startup] Beginning initialization sequence...")
    print("="*70 + "\n")
    
    # 1. Try to initialize database
    print("[Backend] Initializing PostgreSQL database schema...")
    try:
        create_tables()
        db_initialized = True
        print("[Backend] ✅ PostgreSQL database ready\n")
    except Exception as e:
        db_error = str(e)
        print(f"[Backend] ⚠️  Database initialization failed: {e}")
        print("[Backend] ⚠️  Server will start, but /api endpoints will return 503 error")
        print("[Backend] Reason: {}\n".format(db_error))
        # DO NOT raise — let the server continue running
        # This way, /health endpoint works even if DB is down
    
    # 2. Verify Hugging Face Space is ready
    print("[Backend] Checking if Hugging Face Space is initialized...")
    space_ready = await asyncio.to_thread(
        wait_for_hf_space_ready,
        max_attempts=30,
        delay_between_checks=2
    )
    if not space_ready:
        print("[Backend] ⚠️  HF Space still initializing. Requests may fail initially.")
    else:
        print("[Backend] ✅ HF Space ready.\n")
    
    print("="*70)
    print("[Backend Startup] ✅ Initialization complete. Server online.")
    print("="*70 + "\n")


# ── UTILITY: Check DB status before processing ──────────────────
def require_db():
    """Decorator helper: Returns 503 if database is not initialized."""
    if not db_initialized:
        raise HTTPException(
            status_code=503,
            detail=f"Database not initialized. Error: {db_error or 'Unknown error'}"
        )


# ── Standard HTTP REST Endpoints ───────────────────────────────

@app.get("/health")
async def health():
    """
    Health check endpoint — always responds, even if DB is down.
    Clients can use this to detect if the server is alive.
    """
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
    """Process text input — same pipeline as audio but skips Whisper."""
    # Check database is ready before processing
    require_db()
    
    session_id = x_session_id or str(uuid.uuid4())

    # Go straight to LLM — no Whisper needed
    llm_result = llm_engine.generate(
        request.text, request.target_lang, session_id
    )
    ai_text = llm_result["text"]
    llm_ms  = llm_result["latency_ms"]

    # TTS Synthesis
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
    """Process voice input — transcribe, LLM, synthesize."""
    # Check database is ready before processing
    require_db()
    
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


@app.websocket("/ws/stream-audio")
async def websocket_endpoint(websocket: WebSocket):
    """Handles persistent real-time dual-streaming connections."""
    # Check database is ready before accepting WebSocket
    if not db_initialized:
        await websocket.close(code=1008, reason="Database not initialized")
        return
    
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


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session from LLM cache."""
    # Check database is ready before processing
    require_db()
    
    llm_engine.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ── Startup Lifespan (alternative to @app.on_event for FastAPI 0.93+) ──
# If you want to use FastAPI's newer lifespan context manager, uncomment below:
"""
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    print("[Backend] Shutting down...")

app = FastAPI(title="Polyglot Echo", version="2.0", lifespan=lifespan)
"""