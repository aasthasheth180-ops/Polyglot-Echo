# backend/pipeline.py (FIXED)
"""
Fixes applied here:

1. `from backend.pipeline_events.kafka_producer import emit_event, flush`
   pointed at a file that doesn't exist. Now uses the real QStash-backed
   `events.py` module instead (you confirmed you use Upstash QStash, not
   Kafka) — events actually get published now instead of being no-ops.

2. `synthesize_speech(..., target_lang=target_lang, ...)` — ai_client's
   real signature is `synthesize_speech(text, lang=..., speaker_profile=...,
   ref_text=..., session_id=...)`. Calling with `target_lang=` raised
   TypeError on every call through this path.

3. Guest ref_text was hardcoded to Aastha's reference phrase for EVERY
   profile, including guest. F5-TTS needs the ACTUAL transcript of the
   reference clip being used, or cloning quality degrades. Now looks up
   the guest's own auto-transcribed ref_text from voice_sessions.py
   (populated during the enroll step in main.py).

4. Guests are only allowed through here once voice_sessions confirms
   they've recorded + approved their clone — defense in depth alongside
   the same check in main.py's endpoints.

Cache key is on the AI's response text, matching cache.py.
"""

import io
import wave
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

from ai_client import transcribe_audio, synthesize_speech
from llm_engine import llm_engine
from cache import get_cached_audio, set_cached_audio
from events import emit_event, flush
from voice_sessions import voice_sessions

AASTHA_REF_TEXT = "Hey, how have you been lately? It feels like it has been forever."


def _resolve_ref_text(speaker_profile: str, session_id: str) -> str:
    """Aastha uses her fixed tuned reference. Guests use the transcript of
    THEIR OWN recorded sample — using the wrong one here is what breaks
    guest cloning quality."""
    if speaker_profile != "guest":
        return AASTHA_REF_TEXT

    entry = voice_sessions.get(session_id)
    if entry is None:
        raise ValueError(
            "No enrolled voice for this session. Call /api/voice/enroll and "
            "/api/voice/approve before chatting with speaker_profile='guest'."
        )
    return entry.ref_text


def trim_audio_for_whisper(wav_bytes: bytes, max_seconds: int = 30) -> bytes:
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, 'rb') as w:
            framerate = w.getframerate()
            max_frames = max_seconds * framerate
            total_frames = w.getnframes()
            if total_frames <= max_frames:
                return wav_bytes
            w.rewind()
            frames = w.readframes(max_frames)
            params = w.getparams()
        out = io.BytesIO()
        with wave.open(out, 'wb') as w_out:
            w_out.setparams(params)
            w_out.writeframes(frames)
        out.seek(0)
        trimmed = out.read()
        print(f"[Pipeline] Audio trimmed: {total_frames/framerate:.1f}s -> {max_seconds}s")
        return trimmed
    except Exception as e:
        print(f"[Pipeline] Trim failed ({e}) — using original")
        return wav_bytes


def process_audio_loop(
    wav_bytes: bytes,
    target_lang: str = "en",
    speaker_profile: str = "aastha",
    reference_wav: str = None,   # kept for signature compatibility; unused now —
                                  # guest reference is resolved via session_id instead
    session_id: str = None
) -> dict:
    wav_bytes = trim_audio_for_whisper(wav_bytes, max_seconds=30)

    if session_id is None:
        session_id = str(uuid.uuid4())

    if speaker_profile == "guest" and not voice_sessions.is_ready_to_chat(session_id):
        raise ValueError(
            "This session hasn't enrolled/approved a cloned voice yet. "
            "Complete /api/voice/enroll then /api/voice/approve first."
        )

    pipeline_start = time.time()
    emit_event(session_id, "AUDIO_RECEIVED",
               metadata={"target_lang": target_lang, "speaker_profile": speaker_profile})

    print("[Pipeline] Stage 1: Whisper...")
    t1 = time.time()
    whisper_result = transcribe_audio(wav_bytes, session_id=session_id)
    whisper_ms = int((time.time() - t1) * 1000)

    transcript = whisper_result.get("text", "")
    detected_lang = whisper_result.get("language", "unknown")
    print(f"[Pipeline] Transcribed ({detected_lang}): '{transcript[:60]}' | {whisper_ms}ms")

    emit_event(session_id, "TRANSCRIPT_DONE", latency_ms=whisper_ms,
               metadata={"transcript": transcript, "detected_lang": detected_lang})

    ref_text = _resolve_ref_text(speaker_profile, session_id)

    if not transcript.strip():
        fallback = "I didn't catch that. Could you please speak again?"
        audio_bytes_out, _ = synthesize_speech(
            text=fallback, lang="en", speaker_profile=speaker_profile,
            ref_text=ref_text, session_id=session_id
        )
        _write_metrics(session_id, whisper_ms, 0, 0, whisper_ms, target_lang, speaker_profile, False)
        return {
            "audio_bytes": audio_bytes_out, "transcript": "", "detected_lang": detected_lang,
            "response_text": fallback, "target_lang": target_lang, "session_id": session_id,
            "latency": {"whisper_ms": whisper_ms, "llm_ms": 0, "tts_ms": 0, "total_ms": whisper_ms}
        }

    print(f"[Pipeline] Stage 2: Gemini -> '{target_lang}'...")
    llm_result = llm_engine.generate(transcript, target_lang, session_id)
    llm_ms = llm_result["latency_ms"]
    ai_text = llm_result["text"]
    print(f"[Pipeline] Gemini: '{ai_text[:60]}' | {llm_ms}ms")

    emit_event(session_id, "LLM_FIRST_CHUNK", latency_ms=llm_ms,
               metadata={"ai_response": ai_text, "target_lang": target_lang})

    # Cache lookup keyed by the RESPONSE TEXT (not the user's transcript) — fixes
    # "sometimes plays a stale/previous answer" for repeated user phrasing.
    cached = get_cached_audio(ai_text, target_lang, speaker_profile)
    if cached:
        total_ms = int((time.time() - pipeline_start) * 1000)
        emit_event(session_id, "SESSION_END", latency_ms=total_ms, metadata={"cache_hit": True})
        flush()
        print(f"[Pipeline] Cache hit! {total_ms}ms")
        _write_metrics(session_id, whisper_ms, llm_ms, 0, total_ms, target_lang, speaker_profile, True)
        return {
            "audio_bytes": cached, "transcript": transcript, "detected_lang": detected_lang,
            "response_text": ai_text, "target_lang": target_lang, "session_id": session_id,
            "latency": {"whisper_ms": whisper_ms, "llm_ms": llm_ms, "tts_ms": 0, "total_ms": total_ms}
        }

    print(f"[Pipeline] Stage 3: F5-TTS profile={speaker_profile}...")
    t3 = time.time()
    audio_bytes_out, tts_success = synthesize_speech(
        text=ai_text,
        lang=target_lang,                 # FIX: was `target_lang=` (TypeError)
        speaker_profile=speaker_profile,   # "guest" now actually reaches the worker
        ref_text=ref_text,                 # FIX: guest's own transcript, not Aastha's
        session_id=session_id
    )
    tts_ms = int((time.time() - t3) * 1000)
    print(f"[Pipeline] TTS: {len(audio_bytes_out) if audio_bytes_out else 0} bytes | {tts_ms}ms")

    emit_event(session_id, "TTS_DONE", latency_ms=tts_ms, metadata={"speaker_profile": speaker_profile})

    if tts_success:
        set_cached_audio(ai_text, target_lang, audio_bytes_out, speaker_profile)

    total_ms = int((time.time() - pipeline_start) * 1000)
    emit_event(session_id, "SESSION_END", latency_ms=total_ms,
               metadata={"cache_hit": False, "response_text": ai_text, "target_lang": target_lang})
    flush()

    _write_metrics(session_id, whisper_ms, llm_ms, tts_ms, total_ms, target_lang, speaker_profile, False)

    print(f"[Pipeline] Done in {total_ms}ms")
    return {
        "audio_bytes": audio_bytes_out, "transcript": transcript, "detected_lang": detected_lang,
        "response_text": ai_text, "target_lang": target_lang, "session_id": session_id,
        "latency": {"whisper_ms": whisper_ms, "llm_ms": llm_ms, "tts_ms": tts_ms, "total_ms": total_ms}
    }


def _write_metrics(session_id, whisper_ms, llm_ms, tts_ms, total_ms, language, speaker_profile, cache_hit):
    try:
        from metrics_writer import metrics_writer
        metrics_writer.write_turn(
            session_id=session_id, whisper_ms=whisper_ms, llm_ms=llm_ms, tts_ms=tts_ms,
            total_ms=total_ms, language=language, speaker_profile=speaker_profile, cache_hit=cache_hit
        )
    except Exception as metrics_err:
        print(f"[Metrics Bypass Notice] {metrics_err}")