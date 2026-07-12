#!/usr/bin/env python3
# =====================================================================
# HF Space: app.py (FIXED)
# - Loads REAL pretrained F5-TTS weights via the official high-level API
#   (previous version instantiated a random, untrained DiT — this is why
#   cloning quality/robotic-voice/language-mixing bugs happened)
# - Guest voice cloning is now actually wired: /synthesize accepts a
#   reference audio identifier and uses it, instead of always hardcoding
#   Aastha's clip
# - Private HF Dataset downloads now pass HF_TOKEN (required for private
#   repos — missing this caused intermittent silent failures / 500s)
# - Reuses the exact enhance_cloned_voice / enhance_cloned_voice_auto DSP
#   math from voice_enhancer.py / voice_enhancer_auto.py (UNCHANGED) instead
#   of a second, inconsistent post-processing function
# =====================================================================
"""
Polyglot Echo Voice Worker - CORRECTED VERSION
"""

import os
import io
import time
import tempfile
import logging
import sys
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.voice_enhancer import enhance_cloned_voice
from backend.voice_enhancer_auto import enhance_cloned_voice_auto

sys.stdout.flush()
sys.stderr.flush()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── Environment Configuration ──────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Aastha's tuned reference (kept exactly as before)
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", "/tmp/clip_1.wav")
REF_TEXT = os.getenv("REF_TEXT", "Hey, how have you been?")

# Guest reference (this was never wired up before)
GUEST_REF_AUDIO_PATH = os.getenv("GUEST_REF_AUDIO_PATH", "/tmp/guest_ref.wav")

# Private dataset that stores reference clips — needs HF_TOKEN since it's private
DATASET_REPO_ID = os.getenv("DATASET_REPO_ID", "AaasthaSSS/polyglot-audio")
HF_TOKEN = os.getenv("HF_TOKEN")

PORT = int(os.getenv("PORT", "7860"))

print(f"\n{'='*80}", flush=True)
print(f"[Config] Device: {DEVICE.upper()}", flush=True)
print(f"[Config] Dataset repo: {DATASET_REPO_ID}", flush=True)
print(f"[Config] HF_TOKEN present: {bool(HF_TOKEN)}", flush=True)
print(f"[Config] Port: {PORT}", flush=True)
print(f"{'='*80}\n", flush=True)
sys.stdout.flush()

# ── Global Model Containers ────────────────────────────────────
_whisper_model = None
_f5tts = None          # official F5TTS() wrapper instance (has .infer)
_startup_complete = False
_startup_errors = []


def load_whisper_model(device: str = "cuda"):
    """Load OpenAI Whisper (large-v3)."""
    try:
        logger.info("[Whisper] Importing whisper...")
        import whisper
        logger.info(f"[Whisper] Loading large-v3 on {device.upper()}...")
        model = whisper.load_model("large-v3", device=device)
        logger.info("[Whisper] ✅ Loaded")
        return model
    except Exception as e:
        logger.error(f"[Whisper] ❌ FAILED: {e}", exc_info=True)
        raise


def load_f5tts_model(device: str = "cuda"):
    """
    Load F5-TTS with the OFFICIAL high-level API.
    This is the fix: the previous version built a bare `DiT(...)` module
    with random weights and never loaded a checkpoint at all. Using the
    F5TTS() wrapper (same approach as tts_f5.py) downloads and loads the
    actual pretrained checkpoint + vocoder correctly.
    """
    try:
        logger.info("[F5-TTS] Importing official F5TTS API...")
        from f5_tts.api import F5TTS
        logger.info(f"[F5-TTS] Loading pretrained checkpoint on {device.upper()}...")
        instance = F5TTS(device=device)
        logger.info("[F5-TTS] ✅ Pretrained model + vocoder loaded")
        return instance
    except Exception as e:
        logger.error(f"[F5-TTS] ❌ LOAD FAILED: {e}", exc_info=True)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Lifespan] Server ready to accept requests.")
    yield
    logger.info("[Lifespan] Shutting down...")


app = FastAPI(
    title="Polyglot Echo — Voice Worker",
    description="Whisper ASR + F5-TTS Voice Synthesis",
    version="2.1",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SynthRequest(BaseModel):
    """TTS synthesis request."""
    text: str
    lang: str = "en"
    speaker_profile: str = "aastha"   # "aastha" | "guest"
    ref_text: str = ""                # IGNORED now — see _prepare_reference(). Kept only
                                        # for backward-compat with older clients; the worker
                                        # derives the real ref_text itself.
    session_id: str = ""              # which user's cloned voice to use, when speaker_profile == "guest"


# Cache: resolved (trimmed_audio_path -> auto-transcribed ref_text), so we only
# run Whisper once per reference clip, not on every single /synthesize call.
_ref_text_cache: dict = {}

# F5-TTS's own preprocessing silently clips any reference clip longer than
# ~12s. If the ref_text you give it doesn't match that clipped duration,
# F5-TTS's duration predictor gets a wildly wrong "words per second" prior
# from the mismatch and reproduces that wrong pace on EVERY generated
# sentence — this is the actual cause of slow/deep/"ghost-sounding" output.
# So: trim it ourselves to a known-safe length, then transcribe exactly
# that trimmed audio — guaranteeing the text always matches the audio.
_REF_CLIP_MAX_SECONDS = 11


def _guest_filename(session_id: str) -> str:
    safe_id = "".join(c for c in session_id if c.isalnum() or c == "-") or "default"
    return f"guest_{safe_id}.wav"


def _trim_reference_clip(raw_path: str) -> str:
    """
    Re-encode + trim a reference clip to a safe length, following F5-TTS's
    own documented guidance: cutting mid-word/mid-phrase with no trailing
    silence is what causes the model to occasionally repeat a chunk of the
    reference audio inside longer generations (a known F5-TTS behavior, not
    unique to this app). So: prefer cutting at a natural pause if one exists
    within our window, and always leave ~1s of silence at the end either way.
    """
    import subprocess
    trimmed_path = raw_path.rsplit(".", 1)[0] + "_trimmed.wav"
    if os.path.exists(trimmed_path):
        return trimmed_path

    cut_at = _REF_CLIP_MAX_SECONDS  # fallback if no natural pause is found

    try:
        detect = subprocess.run(
            ["ffmpeg", "-i", raw_path, "-af", "silencedetect=noise=-30dB:d=0.3",
             "-t", str(_REF_CLIP_MAX_SECONDS + 2), "-f", "null", "-"],
            capture_output=True, timeout=20, text=True
        )
        pause_points = []
        for line in detect.stderr.splitlines():
            if "silence_start:" in line:
                try:
                    t = float(line.split("silence_start:")[1].strip())
                    if 3.0 <= t <= _REF_CLIP_MAX_SECONDS:
                        pause_points.append(t)
                except ValueError:
                    pass
        if pause_points:
            cut_at = max(pause_points)  # the latest natural pause within our window
            logger.info(f"[Reference] Cutting at natural pause: {cut_at:.1f}s")
        else:
            logger.info(f"[Reference] No natural pause found in first {_REF_CLIP_MAX_SECONDS}s; "
                         f"using hard cutoff with trailing silence padding")
    except Exception as e:
        logger.warning(f"[Reference] Silence detection failed, using hard cutoff: {e}")

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-t", str(cut_at),
         "-af", "apad=pad_dur=1",  # ~1s trailing silence, per F5-TTS's own README guidance
         "-ar", "24000", "-ac", "1", trimmed_path],
        capture_output=True, timeout=30
    )
    if result.returncode != 0 or not os.path.exists(trimmed_path):
        logger.warning(f"[Reference] Trim failed, using raw file untrimmed: "
                        f"{result.stderr.decode(errors='ignore')[:200]}")
        return raw_path
    return trimmed_path


def _auto_transcribe_reference(trimmed_path: str) -> str:
    """Whisper-transcribe the EXACT audio F5-TTS will use as reference,
    caching the result so this only runs once per clip."""
    global _whisper_model

    if trimmed_path in _ref_text_cache:
        return _ref_text_cache[trimmed_path]

    if _whisper_model is None:
        logger.info("[Reference] Loading Whisper to auto-transcribe reference clip...")
        _whisper_model = load_whisper_model(DEVICE)

    with torch.no_grad():
        result = _whisper_model.transcribe(
            trimmed_path, task="transcribe", temperature=0.0, fp16=(DEVICE == "cuda")
        )
    text = result.get("text", "").strip()

    if not text:
        logger.warning(f"[Reference] Auto-transcription came back empty for {trimmed_path}; "
                        f"F5-TTS pacing may be off until a clearer reference is provided.")
        text = "Hello."  # non-empty fallback so F5-TTS doesn't choke on an empty ref_text

    _ref_text_cache[trimmed_path] = text
    logger.info(f"[Reference] Auto-transcribed '{trimmed_path}': '{text[:60]}'")
    return text


def _prepare_reference(speaker_profile: str, session_id: str = "") -> tuple:
    """
    Returns (trimmed_audio_path, accurate_ref_text) for whichever profile is
    requested — downloading from the dataset if needed, trimming to a safe
    duration, and auto-transcribing exactly that trimmed audio. This is the
    single source of truth for reference audio/text now; any ref_text a
    client sends is ignored on purpose, since a hand-typed guess is exactly
    what caused the slow/"ghost" voice bug.
    """
    if speaker_profile == "guest":
        filename = _guest_filename(session_id)
        target_path = f"/tmp/{filename}"
    else:
        target_path = REF_AUDIO_PATH
        filename = "clip_1.wav"

    if not os.path.exists(target_path):
        logger.info(f"[Reference] '{target_path}' missing locally — syncing from dataset...")
        try:
            from huggingface_hub import hf_hub_download
            target_path = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir="/tmp",
                local_dir_use_symlinks=False
            )
            logger.info(f"[Reference] Synced to: {target_path}")
        except Exception as e:
            logger.error(f"[Reference] Sync failed for '{speaker_profile}': {e}", exc_info=True)
            if speaker_profile == "guest":
                raise HTTPException(
                    status_code=400,
                    detail="No cloned voice found for this session yet. Record a voice sample first via /upload-reference."
                )
            raise HTTPException(status_code=500, detail="Reference audio missing and could not be synced")

    trimmed_path = _trim_reference_clip(target_path)
    ref_text = _auto_transcribe_reference(trimmed_path)
    return trimmed_path, ref_text


@app.get("/health")
async def health():
    try:
        all_ready = (_whisper_model is not None and _f5tts is not None)
        return {
            "status": "ok" if all_ready else "initializing",
            "device": DEVICE,
            "whisper_loaded": _whisper_model is not None,
            "f5tts_loaded": _f5tts is not None,
            "models_ready": all_ready,
            "startup_complete": _startup_complete,
            "errors": _startup_errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"[Health] Error: {e}")
        return {"status": "error", "detail": str(e)}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    global _whisper_model
    if _whisper_model is None:
        logger.info("[Transcribe] First request: Loading Whisper model...")
        _whisper_model = load_whisper_model(DEVICE)

    start = time.time()
    try:
        content = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            with torch.no_grad():
                result = _whisper_model.transcribe(
                    tmp_path, task="transcribe", temperature=0.0,
                    compression_ratio_threshold=2.4, no_speech_threshold=0.6,
                    fp16=(DEVICE == "cuda")
                )
            text = result.get("text", "").strip()
            detected_lang = result.get("language", "unknown")

            if not text:
                with torch.no_grad():
                    result = _whisper_model.transcribe(
                        tmp_path, task="transcribe", temperature=0.2, fp16=(DEVICE == "cuda")
                    )
                text = result.get("text", "").strip()

            latency_ms = int((time.time() - start) * 1000)
            logger.info(f"[Transcribe] ✅ ({detected_lang}): {latency_ms}ms")
            return JSONResponse({"text": text, "language": detected_lang, "latency_ms": latency_ms})
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"[Transcribe] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    """Synthesize speech using F5-TTS — now actually honors speaker_profile."""
    global _f5tts
    if _f5tts is None:
        logger.info("[Synthesize] First request: Loading F5-TTS model...")
        _f5tts = load_f5tts_model(DEVICE)

    ref_path, ref_text = _prepare_reference(req.speaker_profile, req.session_id)
    start = time.time()

    try:
        logger.info(f"[Synthesize] profile={req.speaker_profile} text='{req.text[:40]}' "
                    f"ref_text='{ref_text[:40]}'")

        with torch.no_grad():
            audio_wave, sample_rate, _ = _f5tts.infer(
                ref_file=ref_path,
                ref_text=ref_text,   # FIX: always the auto-transcribed, duration-matched text —
                                      # a mismatched ref_text is what caused the slow/"ghost" voice.
                gen_text=req.text,
                speed=1.0
            )

        y = np.array(audio_wave, dtype=np.float32)

        # Use the EXACT same DSP math you said is dialed in — unchanged.
        buf_in = io.BytesIO()
        sf.write(buf_in, y, sample_rate, format="WAV")
        buf_in.seek(0)

        if req.speaker_profile == "guest":
            # Auto-matches target pitch/RMS/brightness to the guest's own reference clip
            audio_bytes = enhance_cloned_voice_auto(buf_in.read(), reference_audio_path=ref_path)
        else:
            # Aastha's hand-tuned enhancement targets — untouched
            audio_bytes = enhance_cloned_voice(buf_in.read())

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Synthesize] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-reference")
async def upload_reference(audio: UploadFile = File(...), session_id: str = Form("")):
    """
    Upload a guest's voice sample for cloning — this is the "record my voice"
    step. Saved per-session (FIX: previously one shared global file for every
    guest) and pushed to the private HF Dataset so it survives the Space
    restarting (Spaces are ephemeral, local /tmp is wiped on restart).

    FIX: browsers record via MediaRecorder, which produces webm/opus (or
    similar), not actual WAV/PCM — even if the frontend names the file
    "something.wav". Whisper's transcribe() shells out to ffmpeg internally
    so it tolerates that fine, but F5-TTS's reference loader uses
    soundfile/librosa (not ffmpeg) to read the reference clip — feeding it
    a webm blob with a .wav name can fail or load garbage. So we always
    re-encode to real 24kHz mono PCM WAV with ffmpeg (now installed in the
    Docker image) before saving/persisting, regardless of what the browser
    actually sent.
    """
    import subprocess
    import tempfile

    try:
        content = await audio.read()
        filename = _guest_filename(session_id)
        local_path = f"/tmp/{filename}"

        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as raw_tmp:
            raw_tmp.write(content)
            raw_tmp_path = raw_tmp.name

        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", raw_tmp_path, "-ar", "24000", "-ac", "1", local_path],
                capture_output=True, timeout=30
            )
            if result.returncode != 0 or not os.path.exists(local_path):
                logger.error(f"[Upload] ffmpeg conversion failed: {result.stderr.decode(errors='ignore')[:300]}")
                raise HTTPException(status_code=400, detail="Could not process that recording — please try again.")
        finally:
            os.unlink(raw_tmp_path)

        logger.info(f"[Upload] Guest reference saved locally: {local_path} ({os.path.getsize(local_path)} bytes)")

        # FIX: transcribe the SAME trimmed audio F5-TTS will actually use, not
        # the raw untrimmed upload — that mismatch is exactly what causes the
        # slow/"ghost"-sounding voice bug for Aastha's clip, and would have
        # eventually hit guest clones too once someone recorded >12s.
        trimmed_path = _trim_reference_clip(local_path)
        detected_text = _auto_transcribe_reference(trimmed_path)

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
            )
            logger.info(f"[Upload] ✅ Persisted '{filename}' to HF dataset")
        except Exception as e:
            logger.warning(f"[Upload] Could not persist to dataset (still usable this session): {e}")

        return {
            "status": "saved", "path": local_path, "bytes": len(content),
            "session_id": session_id, "detected_text": detected_text
        }
    except Exception as e:
        logger.error(f"[Upload] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)