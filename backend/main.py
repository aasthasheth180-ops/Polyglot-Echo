# backend/main.py
import os
import json
import uuid
import time
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response, Header, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Internal imports (NO 'backend.' prefix — we're already in backend/) ───
try:
    from pipeline import process_audio_loop
    print("[Import] ✅ pipeline imported")
except ImportError as e:
    print(f"[Import] ⚠️  pipeline import failed: {e}")

try:
    from llm_engine import llm_engine
    print("[Import] ✅ llm_engine imported")
except ImportError as e:
    print(f"[Import] ⚠️  llm_engine import failed: {e}")

try:
    from ai_client import synthesize_speech, check_colab_health
    print("[Import] ✅ ai_client imported")
except ImportError as e:
    print(f"[Import] ⚠️  ai_client import failed: {e}")

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(title="Polyglot Echo", version="2.0")

# ── CORS Middleware ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Response-Text",
        "X-Session-Id",
        "X-Transcript",
        "X-Detected-Lang",
        "X-Latency-Total-MS",
        "X-LLM-MS",
        "X-TTS-MS",
        "X-Whisper-MS",
        "X-Speaker-Profile"
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
    except:
        colab_ok = False
    
    return {
        "status": "ok",
        "service": "polyglot-echo",
        "version": "2.0",
        "colab_ai": "connected" if colab_ok else "disconnected"
    }

# ── REST API: Process Text ─────────────────────────────────────
@app.post("/api/process-text")
async def process_text(
    request: TextRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    """
    Process text input and return synthesized speech.
    
    Returns:
    - Binary audio/wav content
    - Headers: X-Response-Text, X-LLM-MS, X-TTS-MS, X-Latency-Total-MS
    """
    session_id = x_session_id or str(uuid.uuid4())
    
    try:
        print(f"[API] process-text: '{request.text[:40]}' → {request.target_lang}")
        
        # Generate LLM response
        llm_start = time.time()
        llm_result = llm_engine.generate(
            request.text,
            request.target_lang,
            session_id
        )
        llm_ms = int((time.time() - llm_start) * 1000)
        
        # Synthesize speech
        tts_start = time.time()
        audio_bytes = synthesize_speech(
            text=llm_result["text"],
            lang=request.target_lang,
            speaker_profile=request.speaker_profile,
            ref_text="Hey, how have you been lately?"
        )
        tts_ms = int((time.time() - tts_start) * 1000)
        total_ms = llm_ms + tts_ms
        
        print(f"[API] ✅ LLM {llm_ms}ms + TTS {tts_ms}ms = {total_ms}ms total")
        
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
    except Exception as e:
        print(f"[API] ❌ process-text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── REST API: Process Voice ────────────────────────────────────
@app.post("/api/process-voice")
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
    - Headers: X-Transcript, X-Response-Text, X-Detected-Lang, latencies
    """
    session_id = x_session_id or str(uuid.uuid4())
    
    try:
        print(f"[API] process-voice: {target_lang} → {speaker_profile}")
        
        wav_bytes = await audio_file.read()
        result = process_audio_loop(
            wav_bytes=wav_bytes,
            target_lang=target_lang,
            speaker_profile=speaker_profile,
            reference_wav=reference_wav,
            session_id=session_id
        )
        
        latencies = result.get("latency", {})
        print(f"[API] ✅ Whisper {latencies.get('whisper_ms', 0)}ms + LLM {latencies.get('llm_ms', 0)}ms + TTS {latencies.get('tts_ms', 0)}ms")
        
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
        print(f"[API] ❌ process-voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── REST API: Upload Guest Voice ───────────────────────────────
@app.post("/api/upload-guest-reference")
async def upload_guest_reference(audio_file: UploadFile = File(...)):
    """
    Upload guest voice reference for F5-TTS voice cloning.
    Saves to /tmp/guest_ref.wav
    """
    try:
        content = await audio_file.read()
        
        # Save to standard location
        save_path = "/tmp/guest_ref.wav"
        with open(save_path, "wb") as f:
            f.write(content)
        
        print(f"[API] ✅ Guest voice saved: {len(content)} bytes → {save_path}")
        
        return {
            "status": "success",
            "path": save_path,
            "bytes": len(content)
        }
    except Exception as e:
        print(f"[API] ❌ Guest voice upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── WebSocket: Real-time Streaming ────────────────────────────
@app.websocket("/ws/stream-audio")
async def websocket_stream_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Flow:
    1. Client connects with target_lang, speaker_profile, session_id
    2. Client sends audio chunks (webm/opus)
    3. Server transcribes, generates LLM response, synthesizes TTS
    4. Server streams back: transcripts, LLM chunks, audio chunks
    
    Messages sent to client:
    - {"type": "transcript", "text": "...", "language": "en", "latency_ms": 123}
    - {"type": "llm_chunk", "text": "..."}
    - {"type": "response_done", "full_text": "..."}
    - {"type": "audio_chunk_start", "tts_ms": 234}
    - Binary audio chunks (ArrayBuffer)
    """
    await websocket.accept()
    
    try:
        # Extract query parameters
        target_lang = websocket.query_params.get("target_lang", "en")
        speaker_profile = websocket.query_params.get("speaker_profile", "aastha")
        session_id = websocket.query_params.get("session_id", str(uuid.uuid4()))
        
        print(f"[WS] Connected: {target_lang} → {speaker_profile} (session: {session_id[:8]})")
        
        # Receive audio chunks from client
        audio_chunks = []
        
        while True:
            data = await websocket.receive()
            
            # Handle client disconnect
            if "disconnect" in data:
                print(f"[WS] Disconnected: {session_id[:8]}")
                break
            
            # Handle binary audio data
            if "bytes" in data:
                audio_chunks.append(data["bytes"])
                
                # Once we have enough audio (simple heuristic: ~0.5 seconds)
                if len(audio_chunks) >= 3:
                    # Combine chunks
                    audio_bytes = b"".join(audio_chunks)
                    audio_chunks = []
                    
                    try:
                        # Process audio pipeline
                        print(f"[WS] Processing {len(audio_bytes)} bytes...")
                        
                        result = process_audio_loop(
                            wav_bytes=audio_bytes,
                            target_lang=target_lang,
                            speaker_profile=speaker_profile,
                            session_id=session_id
                        )
                        
                        # Send transcript
                        transcript_msg = {
                            "type": "transcript",
                            "text": result.get("transcript", ""),
                            "language": result.get("detected_lang", target_lang),
                            "latency_ms": result.get("latency", {}).get("whisper_ms", 0)
                        }
                        await websocket.send_json(transcript_msg)
                        
                        # Send LLM response in chunks
                        response_text = result.get("response_text", "")
                        words = response_text.split()
                        for word in words:
                            await websocket.send_json({
                                "type": "llm_chunk",
                                "text": word
                            })
                            await asyncio.sleep(0.05)  # Simulate streaming
                        
                        # Send response done
                        await websocket.send_json({
                            "type": "response_done",
                            "full_text": response_text
                        })
                        
                        # Send audio chunks
                        audio_data = result.get("audio_bytes", b"")
                        chunk_size = 4096  # 4KB chunks
                        
                        # Send audio start marker
                        await websocket.send_json({
                            "type": "audio_chunk_start",
                            "tts_ms": result.get("latency", {}).get("tts_ms", 0)
                        })
                        
                        # Send audio in chunks with slight delay for smooth playback
                        for i in range(0, len(audio_data), chunk_size):
                            chunk = audio_data[i:i+chunk_size]
                            await websocket.send_bytes(chunk)
                            await asyncio.sleep(0.01)  # 10ms delay between chunks
                        
                        print(f"[WS] ✅ Sent transcript + LLM + {len(audio_data)} bytes audio")
                        
                    except Exception as e:
                        print(f"[WS] ❌ Processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
            
            # Handle JSON text messages
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    print(f"[WS] Message: {msg}")
                except json.JSONDecodeError:
                    pass
    
    except Exception as e:
        print(f"[WS] ❌ WebSocket error: {e}")
    
    finally:
        try:
            await websocket.close()
        except:
            pass

# ── Static Files: Serve Frontend ───────────────────────────────
# Try to serve index.html from the static directory created by Dockerfile
static_dir = "/app/static"

if os.path.exists(static_dir):
    print(f"[FastAPI] ✅ Mounting static files from {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    # Fallback: try to serve from current directory
    local_static = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(local_static):
        print(f"[FastAPI] ✅ Mounting static files from {local_static}")
        app.mount("/", StaticFiles(directory=local_static, html=True), name="static")
    else:
        print(f"[FastAPI] ⚠️  Static directory not found at {static_dir} or {local_static}")
        print(f"[FastAPI] ⚠️  Frontend will not be served")

print("\n[FastAPI] ✅ Application startup complete\n")