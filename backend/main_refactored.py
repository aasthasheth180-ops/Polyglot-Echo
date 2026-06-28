# backend/main.py (REFACTORED)
"""
FastAPI backend for Polyglot Echo with:
- Circuit breaker for TTS failures
- Guest voice cloning support
- Comprehensive error handling
- Structured logging
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import uuid
import time
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import (
    COLAB_URL, MIN_AUDIO_BYTES, DEFAULT_REF_TEXT, GUEST_REF_AUDIO_PATH,
    SPEAKER_PROFILES, get_ref_audio_path, get_ref_text
)
from logger import get_logger, set_context, clear_context, log_async_function_call
from health_check import check_endpoint_readiness
from ai_client_refactored import (
    synthesize_speech, check_colab_health, get_circuit_breaker,
    wait_for_hf_space_ready
)
from llm_engine_refactored import llm_engine

logger = get_logger(__name__)

# ── Imports ────────────────────────────────────────────────────
try:
    from pipeline import process_audio_loop
    logger.info("[Import] ✅ pipeline imported")
except ImportError as e:
    logger.warning(f"[Import] ⚠️  pipeline import failed: {e}")

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(title="Polyglot Echo", version="2.1")

# ── CORS Middleware ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Response-Text", "X-Session-Id", "X-Transcript",
        "X-Detected-Lang", "X-Latency-Total-MS", "X-LLM-MS",
        "X-TTS-MS", "X-Whisper-MS", "X-Speaker-Profile"
    ]
)

# ── Data Models ────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    target_lang: str = "en"
    speaker_profile: str = "aastha"


# ── Health Check ───────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        colab_ok = check_colab_health()
    except Exception as e:
        logger.error(f"[Health] Colab health check failed: {e}")
        colab_ok = False
    
    circuit_breaker = get_circuit_breaker()
    
    return JSONResponse({
        "status": "ok",
        "service": "polyglot-echo",
        "version": "2.1",
        "colab_ai": "connected" if colab_ok else "disconnected",
        "circuit_breaker": {
            "open": circuit_breaker.is_open,
            "failures": len(circuit_breaker.failures)
        }
    })


# ── REST API: Process Text ─────────────────────────────────────
@app.post("/api/process-text")
@log_async_function_call
async def process_text(
    request: TextRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    """
    Process text input and return synthesized speech.
    
    Returns:
        - Binary audio/wav content
        - Headers with latencies and metadata
    """
    session_id = x_session_id or str(uuid.uuid4())
    set_context(session_id=session_id, endpoint="process_text", lang=request.target_lang)
    
    try:
        logger.info(f"[API] process-text: '{request.text[:40]}'")
        
        # Pre-flight check
        ready, status = check_endpoint_readiness("process_text")
        if not ready:
            logger.error(f"[API] System not ready: {status}")
            raise HTTPException(status_code=503, detail=status)
        
        # Check circuit breaker
        circuit_breaker = get_circuit_breaker()
        if circuit_breaker.is_open:
            logger.error("[API] Circuit breaker OPEN; TTS service unhealthy")
            raise HTTPException(status_code=503, detail="TTS service temporarily unavailable")
        
        # Generate LLM response
        llm_start = time.time()
        try:
            llm_result = llm_engine.generate(
                request.text,
                request.target_lang,
                session_id
            )
        except Exception as e:
            logger.error(f"[API] LLM generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="LLM generation failed")
        
        llm_ms = int((time.time() - llm_start) * 1000)
        
        # Get reference text (SHORT, not LLM output)
        ref_text = get_ref_text(request.speaker_profile)
        logger.debug(f"[API] Using ref_text: '{ref_text}'")
        
        # Synthesize speech with retry logic
        tts_start = time.time()
        audio_bytes, tts_success = synthesize_speech(
            text=llm_result["text"],
            lang=request.target_lang,
            speaker_profile=request.speaker_profile,
            ref_text=ref_text,
            session_id=session_id
        )
        tts_ms = int((time.time() - tts_start) * 1000)
        
        if not tts_success:
            logger.error("[API] TTS synthesis failed")
            raise HTTPException(status_code=503, detail="TTS synthesis failed; please retry")
        
        total_ms = llm_ms + tts_ms
        logger.info(f"[API] ✅ Complete: LLM {llm_ms}ms + TTS {tts_ms}ms = {total_ms}ms")
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Response-Text": llm_result["text"],
                "X-Session-Id": session_id,
                "X-LLM-MS": str(llm_ms),
                "X-TTS-MS": str(tts_ms),
                "X-Latency-Total-MS": str(total_ms)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] process-text error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


# ── REST API: Process Voice ────────────────────────────────────
@app.post("/api/process-voice")
@log_async_function_call
async def process_voice(
    audio_file: UploadFile = File(...),
    target_lang: str = Form("en"),
    speaker_profile: str = Form("aastha"),
    reference_wav: str = Form(None),
    x_session_id: Optional[str] = Header(default=None)
):
    """
    Process voice input: transcribe → LLM → TTS.
    
    Returns:
        - Binary audio/wav content
        - Headers with transcript, response, and latencies
    """
    session_id = x_session_id or str(uuid.uuid4())
    set_context(session_id=session_id, endpoint="process_voice", lang=target_lang)
    
    try:
        logger.info(f"[API] process-voice: {target_lang} → {speaker_profile}")
        
        # Pre-flight checks
        ready, status = check_endpoint_readiness("process_voice", required_models=["all"])
        if not ready:
            logger.error(f"[API] System not ready: {status}")
            raise HTTPException(status_code=503, detail=status)
        
        circuit_breaker = get_circuit_breaker()
        if circuit_breaker.is_open:
            logger.error("[API] Circuit breaker OPEN")
            raise HTTPException(status_code=503, detail="TTS service temporarily unavailable")
        
        # Read audio
        wav_bytes = await audio_file.read()
        logger.debug(f"[API] Read {len(wav_bytes)} bytes from upload")
        
        # Process audio pipeline
        result = process_audio_loop(
            wav_bytes=wav_bytes,
            target_lang=target_lang,
            speaker_profile=speaker_profile,
            reference_wav=reference_wav,
            session_id=session_id
        )
        
        latencies = result.get("latency", {})
        logger.info(
            f"[API] ✅ Whisper {latencies.get('whisper_ms', 0)}ms + "
            f"LLM {latencies.get('llm_ms', 0)}ms + TTS {latencies.get('tts_ms', 0)}ms"
        )
        
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] process-voice error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


# ── REST API: Upload Guest Voice ───────────────────────────────
@app.post("/api/upload-guest-reference")
@log_async_function_call
async def upload_guest_reference(audio_file: UploadFile = File(...)):
    """
    Upload guest voice reference for F5-TTS cloning.
    Saves to persistent location.
    """
    set_context(operation="upload_guest_reference")
    
    try:
        content = await audio_file.read()
        logger.debug(f"[API] Received {len(content)} bytes")
        
        # Validate
        if len(content) < MIN_AUDIO_BYTES:
            logger.error(f"[API] File too small: {len(content)} bytes")
            raise HTTPException(status_code=400, detail="Audio file too small")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(GUEST_REF_AUDIO_PATH), exist_ok=True)
        
        # Save to persistent location
        with open(GUEST_REF_AUDIO_PATH, "wb") as f:
            f.write(content)
        
        logger.info(f"[API] ✅ Guest voice saved: {len(content)} bytes → {GUEST_REF_AUDIO_PATH}")
        
        return JSONResponse({
            "status": "success",
            "path": GUEST_REF_AUDIO_PATH,
            "bytes": len(content),
            "message": "Guest voice uploaded; use speaker_profile='guest' to activate"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Guest voice upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


# ── REST API: Session Debug Info ───────────────────────────────
@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get debugging info about a session."""
    try:
        info = llm_engine.get_session_info(session_id)
        circuit_breaker = get_circuit_breaker()
        
        return JSONResponse({
            "session": info,
            "circuit_breaker": {
                "open": circuit_breaker.is_open,
                "failures": len(circuit_breaker.failures),
                "window_seconds": circuit_breaker.window_seconds
            }
        })
    except Exception as e:
        logger.error(f"[API] Session info error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── REST API: Clear Session ────────────────────────────────────
@app.post("/api/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Explicitly clear a session."""
    try:
        llm_engine.clear_session(session_id)
        logger.info(f"[API] Session cleared: {session_id}")
        return JSONResponse({"status": "cleared", "session_id": session_id})
    except Exception as e:
        logger.error(f"[API] Clear session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket: Real-time Streaming ────────────────────────────
@app.websocket("/ws/stream-audio")
async def websocket_stream_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Flow:
    1. Client connects with query params: target_lang, speaker_profile, session_id
    2. Client sends audio chunks
    3. Server transcribes, generates LLM response, synthesizes TTS
    4. Server streams back transcript, LLM chunks, and audio
    """
    await websocket.accept()
    
    target_lang = websocket.query_params.get("target_lang", "en")
    speaker_profile = websocket.query_params.get("speaker_profile", "aastha")
    session_id = websocket.query_params.get("session_id", str(uuid.uuid4()))
    
    set_context(session_id=session_id, endpoint="ws_stream_audio", lang=target_lang)
    
    try:
        logger.info(f"[WS] Connected: {target_lang} → {speaker_profile}")
        
        audio_chunks = []
        
        while True:
            data = await websocket.receive()
            
            if "disconnect" in data:
                logger.info(f"[WS] Disconnected: {session_id[:8]}")
                break
            
            if "bytes" in data:
                audio_chunks.append(data["bytes"])
                
                # Process when we have enough audio (~0.5 seconds)
                if len(audio_chunks) >= 3:
                    audio_bytes = b"".join(audio_chunks)
                    audio_chunks = []
                    
                    try:
                        logger.debug(f"[WS] Processing {len(audio_bytes)} bytes")
                        
                        # Process pipeline
                        result = process_audio_loop(
                            wav_bytes=audio_bytes,
                            target_lang=target_lang,
                            speaker_profile=speaker_profile,
                            session_id=session_id
                        )
                        
                        # Send transcript
                        await websocket.send_json({
                            "type": "transcript",
                            "text": result.get("transcript", ""),
                            "language": result.get("detected_lang", target_lang),
                            "latency_ms": result.get("latency", {}).get("whisper_ms", 0)
                        })
                        
                        # Send LLM response in chunks
                        response_text = result.get("response_text", "")
                        words = response_text.split()
                        for word in words:
                            await websocket.send_json({"type": "llm_chunk", "text": word})
                            await asyncio.sleep(0.05)
                        
                        # Send response done
                        await websocket.send_json({
                            "type": "response_done",
                            "full_text": response_text
                        })
                        
                        # Send audio chunks
                        audio_data = result.get("audio_bytes", b"")
                        await websocket.send_json({
                            "type": "audio_chunk_start",
                            "tts_ms": result.get("latency", {}).get("tts_ms", 0)
                        })
                        
                        for i in range(0, len(audio_data), 4096):
                            chunk = audio_data[i:i+4096]
                            await websocket.send_bytes(chunk)
                            await asyncio.sleep(0.01)
                        
                        logger.debug(f"[WS] ✅ Sent transcript + LLM + {len(audio_data)} bytes audio")
                    
                    except Exception as e:
                        logger.error(f"[WS] Processing error: {e}", exc_info=True)
                        await websocket.send_json({"type": "error", "message": str(e)})
            
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    logger.debug(f"[WS] Message: {msg}")
                except json.JSONDecodeError:
                    pass
    
    except Exception as e:
        logger.error(f"[WS] Error: {e}", exc_info=True)
    
    finally:
        try:
            await websocket.close()
        except:
            pass
        clear_context()


# ── Static Files ───────────────────────────────────────────────
static_dir = "/app/static"
if os.path.exists(static_dir):
    logger.info(f"[FastAPI] Mounting static files from {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    local_static = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(local_static):
        logger.info(f"[FastAPI] Mounting static files from {local_static}")
        app.mount("/", StaticFiles(directory=local_static, html=True), name="static")
    else:
        logger.warning("[FastAPI] Static directory not found; frontend will not be served")

logger.info("[FastAPI] ✅ Application startup complete\n")