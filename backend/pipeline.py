# backend/pipeline.py
import io
import wave
import struct
import os
import time
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Native direct imports
from ai_client import transcribe_audio, synthesize_speech, check_colab_health
from llm_engine import llm_engine
from cache import get_cached_audio, set_cached_audio

# ── FIXED LINE: We are now pointing to the unique folder name ──
from pipeline_events.kafka_producer import emit_event, flush

AASTHA_REF_TEXT = "Hey, how have you been lately? It feels like it has been forever."

def trim_audio_for_whisper(wav_bytes: bytes, max_seconds: int = 30) -> bytes:
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, 'rb') as w:
            framerate   = w.getframerate()
            max_frames  = max_seconds * framerate
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
        print(f"[Pipeline] Audio trimmed: {total_frames/framerate:.1f}s → {max_seconds}s")
        return trimmed
    except Exception as e:
        print(f"[Pipeline] Trim failed ({e}) — using original")
        return wav_bytes


def process_audio_loop(
    wav_bytes: bytes,
    target_lang: str = "en",
    speaker_profile: str = "aastha",
    reference_wav: str = None,
    session_id: str = None
) -> dict:
    wav_bytes = trim_audio_for_whisper(wav_bytes, max_seconds=30)

    if session_id is None:
        session_id = str(uuid.uuid4())

    pipeline_start = time.time()

    emit_event(session_id, "AUDIO_RECEIVED",
               metadata={"target_lang": target_lang,
                         "speaker_profile": speaker_profile})

    print("[Pipeline] Stage 1: Whisper (Colab)...")
    t1 = time.time()
    whisper_result = transcribe_audio(wav_bytes)
    whisper_ms = int((time.time() - t1) * 1000)

    transcript   = whisper_result.get("text", "")
    detected_lang = whisper_result.get("language", "unknown")
    print(f"[Pipeline] Transcribed ({detected_lang}): '{transcript[:60]}' | {whisper_ms}ms")

    emit_event(session_id, "TRANSCRIPT_DONE", latency_ms=whisper_ms,
               metadata={"transcript": transcript,
                         "detected_lang": detected_lang})

    if not transcript.strip():
        fallback = "I didn't catch that. Could you please speak again?"
        audio_bytes_out = synthesize_speech(fallback, "en", speaker_profile)
        
        try:
            from metrics_writer import metrics_writer
            metrics_writer.write_turn(
                session_id=session_id,
                whisper_ms=whisper_ms,
                llm_ms=0,
                tts_ms=0,
                total_ms=whisper_ms,
                language=target_lang,
                speaker_profile=speaker_profile,
                cache_hit=False
            )
        except Exception as metrics_err:
            print(f"[Metrics Bypass Notice] {metrics_err}")

        return {
            "audio_bytes": audio_bytes_out,
            "transcript": "",
            "detected_lang": detected_lang,
            "response_text": fallback,
            "target_lang": target_lang,
            "session_id": session_id,
            "latency": {"whisper_ms": whisper_ms, "llm_ms": 0, "tts_ms": 0, "total_ms": whisper_ms}
        }

    cache_key = f"{target_lang}_{speaker_profile}"
    cached = get_cached_audio(transcript, cache_key)
    if cached:
        total_ms = int((time.time() - pipeline_start) * 1000)
        emit_event(session_id, "SESSION_END", latency_ms=total_ms, metadata={"cache_hit": True})
        flush()
        print(f"[Pipeline] Cache hit! {total_ms}ms")

        try:
            from metrics_writer import metrics_writer
            metrics_writer.write_turn(
                session_id=session_id,
                whisper_ms=whisper_ms,
                llm_ms=0,
                tts_ms=0,
                total_ms=total_ms,
                language=target_lang,
                speaker_profile=speaker_profile,
                cache_hit=True
            )
        except Exception as metrics_err:
            print(f"[Metrics Bypass Notice] {metrics_err}")

        return {
            "audio_bytes": cached,
            "transcript": transcript,
            "detected_lang": detected_lang,
            "response_text": "[CACHE HIT]",
            "target_lang": target_lang,
            "session_id": session_id,
            "latency": {"whisper_ms": whisper_ms, "llm_ms": 0, "tts_ms": 0, "total_ms": total_ms}
        }

    print(f"[Pipeline] Stage 2: Gemini → '{target_lang}'...")
    llm_result = llm_engine.generate(transcript, target_lang, session_id)
    llm_ms = llm_result["latency_ms"]
    ai_text = llm_result["text"]
    print(f"[Pipeline] Gemini: '{ai_text[:60]}' | {llm_ms}ms")

    emit_event(session_id, "LLM_FIRST_CHUNK", latency_ms=llm_ms,
               metadata={"ai_response": ai_text, "target_lang": target_lang})

    print(f"[Pipeline] Stage 3: F5-TTS (Colab) lang={target_lang}...")
    t3 = time.time()
    audio_bytes_out = synthesize_speech(
        text=ai_text,
        target_lang=target_lang,
        speaker_profile=speaker_profile,
        ref_text=AASTHA_REF_TEXT
    )
    tts_ms = int((time.time() - t3) * 1000)
    print(f"[Pipeline] TTS: {len(audio_bytes_out)} bytes | {tts_ms}ms")

    emit_event(session_id, "TTS_DONE", latency_ms=tts_ms, metadata={"speaker_profile": speaker_profile})

    set_cached_audio(transcript, cache_key, audio_bytes_out)

    total_ms = int((time.time() - pipeline_start) * 1000)
    emit_event(session_id, "SESSION_END", latency_ms=total_ms,
               metadata={"cache_hit": False, "response_text": ai_text, "target_lang": target_lang})
    flush()

    try:
        from metrics_writer import metrics_writer
        metrics_writer.write_turn(
            session_id=session_id,
            whisper_ms=whisper_ms,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
            total_ms=total_ms,
            language=target_lang,
            speaker_profile=speaker_profile,
            cache_hit=False
        )
    except Exception as metrics_err:
        print(f"[Metrics Bypass Notice] Ingestion layer skipped: {metrics_err}")

    print(f"[Pipeline] ✓ Done in {total_ms}ms")
    return {
        "audio_bytes": audio_bytes_out,
        "transcript": transcript,
        "detected_lang": detected_lang,
        "response_text": ai_text,
        "target_lang": target_lang,
        "session_id": session_id,
        "latency": {
            "whisper_ms": whisper_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms
        }
    }