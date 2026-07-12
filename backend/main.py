# backend/main.py (REFACTORED / FIXED)
"""
Changes in this version:

1. NEW: /api/voice/enroll, /api/voice/preview, /api/voice/approve
   This is the actual "record my voice -> clone a short preview -> I approve
   it -> now chat in my voice" flow you asked for. Previously
   /api/upload-guest-reference just saved a file locally and nothing else
   in the app ever used it or referenced it consistently with the worker's
   per-session naming.

2. FIX: process_text no longer calls config.get_ref_text() for guests —
   that always returned Aastha's fixed phrase regardless of profile, which
   breaks F5-TTS cloning (it needs the REAL transcript of whichever
   reference clip is being used). Guests now get their own enrolled
   ref_text from voice_sessions.py.

3. FIX: guest requests are now rejected with a clear 400 (not a confusing
   500 or silently-wrong audio) if that session hasn't enrolled+approved a
   voice yet.

4. NEW: /internal/qstash-webhook — receives the events pipeline.py now
   publishes via QStash (events.py) and writes them into the stage_events
   table (models.py) that already existed but had no writer.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import uuid
import time
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import (
    COLAB_URL, MIN_AUDIO_BYTES, GUEST_REF_AUDIO_PATH,
    SPEAKER_PROFILES, get_ref_audio_path, get_ref_text
)
from logger import get_logger, set_context, clear_context, log_async_function_call
from health_check import check_endpoint_readiness
from ai_client import (
    synthesize_speech, check_colab_health, get_circuit_breaker,
    wait_for_hf_space_ready
)
from llm_engine import llm_engine
from voice_sessions import voice_sessions

logger = get_logger(__name__)

try:
    from pipeline import process_audio_loop
    logger.info("[Import] ✅ pipeline imported")
except ImportError as e:
    logger.warning(f"[Import] ⚠️  pipeline import failed: {e}")

app = FastAPI(title="Polyglot Echo", version="2.2")

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


class TextRequest(BaseModel):
    text: str
    target_lang: str = "en"
    speaker_profile: str = "aastha"


def _resolve_ref_text(speaker_profile: str, session_id: str) -> str:
    """Same resolution logic as pipeline.py, for the text-only chat path."""
    if speaker_profile != "guest":
        return get_ref_text(speaker_profile)

    entry = voice_sessions.get(session_id)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail="No enrolled voice for this session. Record a sample via "
                   "/api/voice/enroll, then approve it via /api/voice/approve."
        )
    return entry.ref_text


@app.get("/health")
async def health():
    try:
        colab_ok = check_colab_health()
    except Exception as e:
        logger.error(f"[Health] Colab health check failed: {e}")
        colab_ok = False

    circuit_breaker = get_circuit_breaker()

    return JSONResponse({
        "status": "ok",
        "service": "polyglot-echo",
        "version": "2.2",
        "colab_ai": "connected" if colab_ok else "disconnected",
        "circuit_breaker": {
            "open": circuit_breaker.is_open,
            "failures": len(circuit_breaker.failures)
        }
    })


# ── Voice Enrollment Flow (record -> preview -> approve) ──────────────
@app.post("/api/voice/enroll")
@log_async_function_call
async def enroll_voice(
    audio_file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default=None)
):
    """
    Step 1: user records a short (~10-15s) clean sample of their own voice.
    We forward it to the worker to be stored (locally + pushed to the
    private HF dataset), and auto-transcribe it with Whisper so we have an
    ACCURATE reference transcript — no need to ask the user to type out
    exactly what they said.
    """
    session_id = x_session_id or str(uuid.uuid4())
    set_context(session_id=session_id, endpoint="enroll_voice")

    try:
        content = await audio_file.read()
        if len(content) < MIN_AUDIO_BYTES:
            raise HTTPException(status_code=400, detail="Recording too short/small — please record a longer sample.")

        import requests
        # Save the raw sample on the worker (app.py), keyed by this session.
        # FIX: the worker now trims the clip to a safe length AND transcribes
        # exactly that trimmed audio itself, returning detected_text directly —
        # so we no longer make a second, separate /transcribe call here on the
        # raw untrimmed clip, which could disagree with what F5-TTS actually
        # uses as its reference (the root cause of the slow/"ghost" voice bug).
        upload_resp = requests.post(
            f"{COLAB_URL}/upload-reference",
            files={"audio": ("guest_sample.wav", content, "audio/wav")},
            data={"session_id": session_id},
            timeout=60
        )
        if upload_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Voice worker rejected the sample upload.")

        ref_text = upload_resp.json().get("detected_text", "").strip()

        if not ref_text:
            raise HTTPException(
                status_code=400,
                detail="Couldn't make out any speech in that recording — please try again somewhere quieter."
            )

        voice_sessions.enroll(session_id, ref_text)
        logger.info(f"[Enroll] session={session_id[:8]} ref_text='{ref_text[:50]}'")

        return JSONResponse({
            "status": "enrolled",
            "session_id": session_id,
            "detected_text": ref_text,
            "next_step": "Call /api/voice/preview to hear your cloned voice, then /api/voice/approve to confirm it."
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Enroll] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


@app.post("/api/voice/preview")
@log_async_function_call
async def preview_voice(x_session_id: Optional[str] = Header(default=None)):
    """
    Step 2: generate ONE short line in the newly cloned voice so the user
    can approve or re-record before starting the real chat.
    """
    session_id = x_session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header.")

    entry = voice_sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=400, detail="No enrolled sample for this session yet — call /api/voice/enroll first.")

    preview_line = "Hi, this is your voice — cloned. Does this sound right to you?"
    audio_bytes, success = synthesize_speech(
        text=preview_line,
        lang="en",
        speaker_profile="guest",
        ref_text=entry.ref_text,
        session_id=session_id
    )
    if not success:
        raise HTTPException(status_code=503, detail="Preview synthesis failed — please try again.")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"X-Session-Id": session_id, "X-Preview-Text": preview_line}
    )


@app.post("/api/voice/approve")
async def approve_voice(x_session_id: Optional[str] = Header(default=None)):
    """Step 3: user confirms the preview sounded right — unlocks guest chat."""
    session_id = x_session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header.")

    ok = voice_sessions.approve(session_id)
    if not ok:
        raise HTTPException(status_code=400, detail="No enrolled sample for this session — call /api/voice/enroll first.")

    logger.info(f"[Approve] session={session_id[:8]} approved for guest chat")
    return JSONResponse({"status": "approved", "session_id": session_id})


# ── REST API: Process Text ─────────────────────────────────────
@app.post("/api/process-text")
@log_async_function_call
async def process_text(
    request: TextRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    session_id = x_session_id or str(uuid.uuid4())
    set_context(session_id=session_id, endpoint="process_text", lang=request.target_lang)

    try:
        logger.info(f"[API] process-text: '{request.text[:40]}'")

        if request.speaker_profile == "guest" and not voice_sessions.is_ready_to_chat(session_id):
            raise HTTPException(
                status_code=400,
                detail="This session hasn't enrolled/approved a cloned voice yet. "
                       "Complete /api/voice/enroll then /api/voice/approve first."
            )

        ready, status = check_endpoint_readiness("process_text")
        if not ready:
            logger.error(f"[API] System not ready: {status}")
            raise HTTPException(status_code=503, detail=status)

        circuit_breaker = get_circuit_breaker()
        if circuit_breaker.is_open:
            logger.error("[API] Circuit breaker OPEN; TTS service unhealthy")
            raise HTTPException(status_code=503, detail="TTS service temporarily unavailable")

        llm_start = time.time()
        try:
            llm_result = llm_engine.generate(request.text, request.target_lang, session_id)
        except Exception as e:
            logger.error(f"[API] LLM generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="LLM generation failed")
        llm_ms = int((time.time() - llm_start) * 1000)

        # FIX: was always Aastha's fixed phrase regardless of profile.
        ref_text = _resolve_ref_text(request.speaker_profile, session_id)
        logger.debug(f"[API] Using ref_text: '{ref_text}'")

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
    x_session_id: Optional[str] = Header(default=None)
):
    session_id = x_session_id or str(uuid.uuid4())
    set_context(session_id=session_id, endpoint="process_voice", lang=target_lang)

    try:
        logger.info(f"[API] process-voice: {target_lang} -> {speaker_profile}")

        if speaker_profile == "guest" and not voice_sessions.is_ready_to_chat(session_id):
            raise HTTPException(
                status_code=400,
                detail="This session hasn't enrolled/approved a cloned voice yet. "
                       "Complete /api/voice/enroll then /api/voice/approve first."
            )

        ready, status = check_endpoint_readiness("process_voice", required_models=["all"])
        if not ready:
            logger.error(f"[API] System not ready: {status}")
            raise HTTPException(status_code=503, detail=status)

        circuit_breaker = get_circuit_breaker()
        if circuit_breaker.is_open:
            logger.error("[API] Circuit breaker OPEN")
            raise HTTPException(status_code=503, detail="TTS service temporarily unavailable")

        wav_bytes = await audio_file.read()
        logger.debug(f"[API] Read {len(wav_bytes)} bytes from upload")

        result = process_audio_loop(
            wav_bytes=wav_bytes,
            target_lang=target_lang,
            speaker_profile=speaker_profile,
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
    except ValueError as e:
        # Raised by pipeline.py when a guest session isn't enrolled/approved
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[API] process-voice error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_context()


# ── REST API: Session Debug Info ───────────────────────────────
@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    try:
        info = llm_engine.get_session_info(session_id)
        circuit_breaker = get_circuit_breaker()
        voice_entry = voice_sessions.get(session_id)

        return JSONResponse({
            "session": info,
            "voice_enrollment": {
                "enrolled": voice_entry is not None,
                "approved": voice_entry.approved if voice_entry else False
            },
            "circuit_breaker": {
                "open": circuit_breaker.is_open,
                "failures": len(circuit_breaker.failures),
                "window_seconds": circuit_breaker.window_seconds
            }
        })
    except Exception as e:
        logger.error(f"[API] Session info error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/clear")
async def clear_session(session_id: str):
    try:
        llm_engine.clear_session(session_id)
        voice_sessions.clear(session_id)
        logger.info(f"[API] Session cleared: {session_id}")
        return JSONResponse({"status": "cleared", "session_id": session_id})
    except Exception as e:
        logger.error(f"[API] Clear session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Internal: QStash Event Webhook ─────────────────────────────
@app.post("/internal/qstash-webhook")
async def qstash_webhook(request: Request):
    """
    Receives events published by events.py (via QStash) and writes them
    into the stage_events table (models.py) — that table already existed
    but previously had no writer since the old kafka_producer import was
    broken and pipeline.py silently never emitted anything real.
    """
    try:
        payload = await request.json()
        session_id = payload.get("session_id", "unknown")
        event_type = payload.get("event_type", "unknown")
        latency_ms = payload.get("latency_ms")
        metadata = payload.get("metadata", {})

        from models import SessionLocal, StageEvent
        db = SessionLocal()
        try:
            db.add(StageEvent(
                session_id=session_id,
                event_type=event_type,
                latency_ms=latency_ms or 0,
                metadata_json=json.dumps(metadata)
            ))
            db.commit()
        finally:
            db.close()

        return JSONResponse({"status": "recorded"})
    except Exception as e:
        logger.error(f"[QStash Webhook] Error: {e}", exc_info=True)
        # Return 500 so QStash retries per its normal retry policy.
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket: Real-time Streaming ────────────────────────────
@app.websocket("/ws/stream-audio")
async def websocket_stream_audio(websocket: WebSocket):
    await websocket.accept()

    target_lang = websocket.query_params.get("target_lang", "en")
    speaker_profile = websocket.query_params.get("speaker_profile", "aastha")
    session_id = websocket.query_params.get("session_id", str(uuid.uuid4()))

    set_context(session_id=session_id, endpoint="ws_stream_audio", lang=target_lang)

    if speaker_profile == "guest" and not voice_sessions.is_ready_to_chat(session_id):
        await websocket.send_json({
            "type": "error",
            "message": "This session hasn't enrolled/approved a cloned voice yet. "
                       "Complete /api/voice/enroll then /api/voice/approve first."
        })
        await websocket.close()
        return

    try:
        logger.info(f"[WS] Connected: {target_lang} -> {speaker_profile}")

        audio_chunks = []

        while True:
            data = await websocket.receive()

            if "disconnect" in data:
                logger.info(f"[WS] Disconnected: {session_id[:8]}")
                break

            if "bytes" in data:
                audio_chunks.append(data["bytes"])

                if len(audio_chunks) >= 3:
                    audio_bytes = b"".join(audio_chunks)
                    audio_chunks = []

                    try:
                        logger.debug(f"[WS] Processing {len(audio_bytes)} bytes")

                        result = process_audio_loop(
                            wav_bytes=audio_bytes,
                            target_lang=target_lang,
                            speaker_profile=speaker_profile,
                            session_id=session_id
                        )

                        await websocket.send_json({
                            "type": "transcript",
                            "text": result.get("transcript", ""),
                            "language": result.get("detected_lang", target_lang),
                            "latency_ms": result.get("latency", {}).get("whisper_ms", 0)
                        })

                        response_text = result.get("response_text", "")
                        words = response_text.split()
                        for word in words:
                            await websocket.send_json({"type": "llm_chunk", "text": word})
                            await asyncio.sleep(0.05)

                        await websocket.send_json({
                            "type": "response_done",
                            "full_text": response_text
                        })

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

                    except ValueError as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
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
        except Exception:
            pass
        clear_context()


# ── Static Files ───────────────────────────────────────────────
# FIX: index.html now lives in a top-level frontend/ folder (sibling of
# backend/), not backend/static/ — checking all plausible locations so
# this keeps working regardless of which layout is actually deployed.
_static_candidates = [
    "/app/static",
    "/code/frontend",
    os.path.join(os.path.dirname(__file__), "static"),
    os.path.join(os.path.dirname(__file__), "..", "frontend"),
]

_mounted = False
for _candidate in _static_candidates:
    if os.path.exists(_candidate):
        logger.info(f"[FastAPI] Mounting static files from {_candidate}")
        app.mount("/", StaticFiles(directory=_candidate, html=True), name="static")
        _mounted = True
        break

if not _mounted:
    logger.warning(f"[FastAPI] No static directory found in any of {_static_candidates}; frontend will not be served")

logger.info("[FastAPI] ✅ Application startup complete\n")