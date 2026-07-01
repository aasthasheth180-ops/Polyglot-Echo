#!/usr/bin/env python3
# =====================================================================
# HF Space: app.py (FIXED - Avoids F5-TTS Config File Issue)
# =====================================================================
"""
Polyglot Echo Voice Worker - CORRECTED VERSION
- Uses correct F5-TTS API (avoids infer_cli.py)
- No config file dependencies
- Works with official GitHub repo
"""

import os
import io
import time
import tempfile
import logging
import sys
import json
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from datetime import datetime


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Force unbuffered logging ───────────────────────────────────
sys.stdout.flush()
sys.stderr.flush()

# ── Standard Python Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ── Environment Configuration ──────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", "/tmp/clip_1.wav")
REF_TEXT = os.getenv("REF_TEXT", "Hey, how have you been?")
PORT = int(os.getenv("PORT", "7860"))

print(f"\n{'='*80}", flush=True)
print(f"[Config] Device: {DEVICE.upper()}", flush=True)
print(f"[Config] Reference audio: {REF_AUDIO_PATH}", flush=True)
print(f"[Config] Reference text: {REF_TEXT}", flush=True)
print(f"[Config] Port: {PORT}", flush=True)
print(f"{'='*80}\n", flush=True)

sys.stdout.flush()

# ── Global Model Containers ────────────────────────────────────
_whisper_model = None
_f5tts_model = None
_startup_complete = False
_startup_errors = []

# ── Whisper Model Loading ──────────────────────────────────────
def load_whisper_model(device: str = "cuda"):
    """Load OpenAI Whisper (large-v3)."""
    try:
        print("  [Whisper] Importing whisper...", flush=True)
        logger.info("[Whisper] Importing whisper...")
        
        import whisper
        
        print("  [Whisper] ✅ Import successful", flush=True)
        
        print(f"  [Whisper] Loading large-v3 on {device.upper()}...", flush=True)
        logger.info(f"[Whisper] Loading large-v3 on {device.upper()}...")
        
        model = whisper.load_model("large-v3", device=device)
        
        print("  [Whisper] ✅ Loaded", flush=True)
        logger.info("[Whisper] ✅ Loaded")
        
        return model
    
    except Exception as e:
        error_msg = f"[Whisper] ❌ FAILED: {e}"
        print(error_msg, flush=True)
        logger.error(error_msg, exc_info=True)
        raise

# ── F5-TTS Model Loading (CORRECT API - NO infer_cli) ─────────
def load_f5tts_model(device: str = "cuda"):
    """Load F5-TTS with the updated API."""
    try:
        logger.info("[F5-TTS] Importing F5-TTS modules...")
        
        # Import the model and utility functions
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_vocoder, infer_process
        
        logger.info("[F5-TTS] ✅ Imports successful")
        logger.info(f"[F5-TTS] Loading model on {device.upper()}...")
        
        # Instead of calling load_model(model_type=...), instantiate the model directly.
        # Ensure you have the checkpoint path loaded.
        model = DiT(dim=1024, depth=22, heads=16) 
        model.to(device)
        model.eval()
        
        logger.info("[F5-TTS] ✅ Model loaded")
        logger.info("[F5-TTS] Loading vocoder...")
        
        vocoder = load_vocoder(device=device)
        
        return {
            "model": model,
            "vocoder": vocoder,
            "infer_fn": infer_process
        }
    
    except Exception as e:
        logger.error(f"[F5-TTS] ❌ LOAD FAILED: {e}", exc_info=True)
        raise
     

# ── Eager Model Loading ────────────────────────────────────────
def load_models_eager():
    """Load both models during startup."""
    global _whisper_model, _f5tts_model, _startup_complete, _startup_errors
    
    print("=" * 80, flush=True)
    print("[STARTUP] Beginning eager model load sequence...", flush=True)
    print("=" * 80, flush=True)
    
    logger.info("=" * 80)
    logger.info("[STARTUP] Beginning eager model load sequence...")
    logger.info("=" * 80)
    
    sys.stdout.flush()
    
    try:
        # ──────────────────────────────────────────────────────────────
        # STEP 1: LOAD WHISPER
        # ──────────────────────────────────────────────────────────────
        print("[STARTUP] Step 1/2: Loading Whisper...", flush=True)
        logger.info("[STARTUP] Step 1/2: Loading Whisper...")
        
        whisper_start = time.time()
        _whisper_model = load_whisper_model(DEVICE)
        whisper_time = time.time() - whisper_start
        
        msg = f"[Whisper] ✅ Loaded in {whisper_time:.2f}s"
        print(msg, flush=True)
        logger.info(msg)
        
        sys.stdout.flush()
        
        # ──────────────────────────────────────────────────────────────
        # STEP 2: LOAD F5-TTS
        # ──────────────────────────────────────────────────────────────
        print("[STARTUP] Step 2/2: Loading F5-TTS...", flush=True)
        logger.info("[STARTUP] Step 2/2: Loading F5-TTS...")
        
        f5tts_start = time.time()
        _f5tts_model = load_f5tts_model(DEVICE)
        f5tts_time = time.time() - f5tts_start
        
        msg = f"[F5-TTS] ✅ Loaded in {f5tts_time:.2f}s"
        print(msg, flush=True)
        logger.info(msg)
        
        sys.stdout.flush()
        
        # ──────────────────────────────────────────────────────────────
        # SUCCESS
        # ──────────────────────────────────────────────────────────────
        total_time = whisper_time + f5tts_time
        
        print("=" * 80, flush=True)
        print(f"[STARTUP] ✅ ALL MODELS LOADED in {total_time:.2f}s", flush=True)
        print("=" * 80, flush=True)
        
        logger.info("=" * 80)
        logger.info(f"[STARTUP] ✅ ALL MODELS LOADED in {total_time:.2f}s")
        logger.info("=" * 80)
        
        _startup_complete = True
        
        sys.stdout.flush()
        
    except Exception as e:
        error_msg = f"[CRITICAL] Model loading FAILED: {str(e)}"
        
        print(f"\n{'!'*80}", flush=True)
        print(error_msg, flush=True)
        print(f"Full error:\n{e}", flush=True)
        print(f"{'!'*80}\n", flush=True)
        
        logger.critical(error_msg)
        logger.critical(f"Full error: {e}", exc_info=True)
        
        _startup_errors.append(error_msg)
        _startup_complete = False
        
        sys.stdout.flush()
        
        raise

# ── FastAPI Lifespan ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown."""
    # We are no longer loading models here to prevent 503 timeouts.
    # Models will be loaded 'lazily' on the first request.
    logger.info("[Lifespan] Server ready to accept requests.")
    yield
    # Cleanup happens here when the app stops
    logger.info("[Lifespan] Shutting down...")

# ── Create FastAPI App (WITH lifespan parameter) ──────────────
app = FastAPI(
    title="Polyglot Echo — Voice Worker",
    description="Whisper ASR + F5-TTS Voice Synthesis",
    version="2.0",
    lifespan=lifespan  # ← CRITICAL: Must pass lifespan
)

# ── CORS Middleware ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Audio Post-Processing ──────────────────────────────────────
def post_process_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply DSP post-processing to TTS output."""
    import scipy.signal as signal
    
    y = y.astype(np.float32)
    
    # RMS Normalization
    TARGET_RMS = 0.05
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y * (TARGET_RMS / rms)
        y = np.clip(y, -1.0, 1.0)
    
    # Gentle high-pass filtering
    sos = signal.butter(1, 4500, btype='low', fs=sr, output='sos')
    y_soft = signal.sosfilt(sos, y)
    y = (0.9 * y) + (0.1 * y_soft)
    y = np.clip(y, -1.0, 1.0)
    
    return y

# ── Data Models ────────────────────────────────────────────────
class SynthRequest(BaseModel):
    """TTS synthesis request."""
    text: str
    lang: str = "en"
    speaker_profile: str = "aastha"
    ref_text: str = REF_TEXT

# ── ENDPOINTS ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        all_ready = (_whisper_model is not None and _f5tts_model is not None)
        return {
            "status": "ok" if all_ready else "initializing",
            "device": DEVICE,
            "whisper_loaded": _whisper_model is not None,
            "f5tts_loaded": _f5tts_model is not None,
            "models_ready": all_ready,
            "startup_complete": _startup_complete,
            "errors": _startup_errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"[Health] Error: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/debug/startup-errors")
async def get_startup_errors():
    """Debug endpoint - shows startup errors."""
    return {
        "startup_complete": _startup_complete,
        "errors": _startup_errors,
        "whisper_model_loaded": _whisper_model is not None,
        "f5tts_model_loaded": _f5tts_model is not None,
        "device": DEVICE,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio using Whisper."""
    global _whisper_model
    
    # LAZY LOAD: Load only when needed
    if _whisper_model is None:
        logger.info("[Transcribe] First request: Loading Whisper model...")
        _whisper_model = load_whisper_model(DEVICE)
    
    start = time.time()
    
    try:
        content = await audio.read()
        logger.debug(f"[Transcribe] Received {len(content)} bytes")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            with torch.no_grad():
                result = _whisper_model.transcribe(
                    tmp_path,
                    task="transcribe",
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    fp16=(DEVICE == "cuda")
                )
            
            text = result.get("text", "").strip()
            detected_lang = result.get("language", "unknown")
            
            if not text:
                logger.info("[Transcribe] Empty result, retrying...")
                with torch.no_grad():
                    result = _whisper_model.transcribe(
                        tmp_path,
                        task="transcribe",
                        temperature=0.2,
                        fp16=(DEVICE == "cuda")
                    )
                text = result.get("text", "").strip()
            
            latency_ms = int((time.time() - start) * 1000)
            logger.info(f"[Transcribe] ✅ ({detected_lang}): {latency_ms}ms")
            
            return JSONResponse({
                "text": text,
                "language": detected_lang,
                "latency_ms": latency_ms
            })
        
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"[Transcribe] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    """Synthesize speech using F5-TTS."""
    global _f5tts_model
    
    # LAZY LOAD: Load model if needed
    if _f5tts_model is None:
        logger.info("[Synthesize] First request: Loading F5-TTS model...")
        _f5tts_model = load_f5tts_model(DEVICE)

    # NEW: Ensure the reference exists (Download if missing)
    if not os.path.exists(REF_AUDIO_PATH):
        logger.info("[Synthesize] Downloading reference audio from Dataset...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id="AaasthaSSS/polyglot-audio", 
                filename="clip_1.wav", 
                repo_type="dataset",
                local_dir="/tmp",
                local_dir_use_symlinks=False
            )
            logger.info(f"[Synthesize] Downloaded to: {downloaded_path}")
        except Exception as e:
            logger.error(f"[Synthesize] Download failed: {e}")
            raise HTTPException(status_code=500, detail="Reference audio missing")
    
    start = time.time()
    
    try:
        # Now ref_path will definitely exist!
        ref_path = REF_AUDIO_PATH
        
        logger.info(f"[Synthesize] Text: '{req.text[:40]}'")
        
        model = _f5tts_model["model"]
        vocoder = _f5tts_model["vocoder"]
        infer_fn = _f5tts_model["infer_fn"]
        
        with torch.no_grad():
            audio_wave, sample_rate, _ = infer_fn(
                ref_file=ref_path,
                ref_text=req.ref_text,
                gen_text=req.text,
                model=model,
                vocoder=vocoder,
                cfg_strength=1.8,
                nfe_step=24,
                speed=1.15
            )
        
        y = np.array(audio_wave, dtype=np.float32)
        y = post_process_audio(y, sample_rate)
        
        buf = io.BytesIO()
        sf.write(buf, y, sample_rate, format="WAV")
        buf.seek(0)
        audio_bytes = buf.read()
        
        if len(audio_bytes) < 1000:
            logger.error(f"[Synthesize] Output too small: {len(audio_bytes)}")
            raise HTTPException(status_code=500, detail="Invalid audio output")
        
        latency_ms = int((time.time() - start) * 1000)
        logger.info(f"[Synthesize] ✅ {len(audio_bytes)} bytes in {latency_ms}ms")
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"X-TTS-MS": str(latency_ms)}
        )
    
    except Exception as e:
        logger.error(f"[Synthesize] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-reference")
async def upload_reference(audio: UploadFile = File(...)):
    """Upload guest voice reference."""
    try:
        content = await audio.read()
        save_path = "/code/guest_ref.wav"
        
        with open(save_path, "wb") as f:
            f.write(content)
        
        logger.info(f"[Upload] Guest reference saved: {len(content)} bytes")
        return {"status": "saved", "path": save_path}
    
    except Exception as e:
        logger.error(f"[Upload] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("", flush=True)
    print("=" * 80, flush=True)
    print("[Main] Starting Uvicorn ASGI server...", flush=True)
    print(f"[Main] Host: 0.0.0.0", flush=True)
    print(f"[Main] Port: {PORT}", flush=True)
    print(f"[Main] Device: {DEVICE.upper()}", flush=True)
    print("=" * 80, flush=True)
    print("", flush=True)
    
    sys.stdout.flush()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860
    )