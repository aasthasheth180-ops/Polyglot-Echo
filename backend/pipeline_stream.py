import os
import time
import tempfile
import uuid
import asyncio
import io
import numpy as np
import soundfile as sf
from typing import AsyncGenerator

from backend.whisper_engine import whisper_engine
from backend.llm_engine import llm_engine
# 🎯 FIX: Import the singleton getter instead of the raw VoiceCloner class
from backend.tts_handler import get_cloner
from backend.voice_enhancer import enhance_cloned_voice
from backend.voice_enhancer_auto import enhance_cloned_voice_auto

# Add these dummy functions directly underneath:
def emit_event(*args, **kwargs):
    pass

def flush():
    pass

# 🎯 FIX: Fetch the shared single-instance memory block
tts_cloner = get_cloner()
AASTHA_REFERENCE_PATH = "audio/clip_1.wav"

SENTENCE_ENDINGS = {'.', '!', '?', '।', '。', '؟'}


async def process_audio_stream(
    wav_bytes: bytes,
    target_lang: str = "en",
    speaker_profile: str = "aastha",
    reference_wav: str = None
) -> AsyncGenerator[dict, None]:
    """
    Streaming pipeline. Yields dicts as each stage completes:
        {"type": "transcript", ...}
        {"type": "chunk", "audio_bytes": ..., "text": ...}
        {"type": "done", "latency": {...}}
    """
    session_id = str(uuid.uuid4())
    pipeline_start = time.time()
    emit_event(session_id, "AUDIO_RECEIVED",
               metadata={"target_lang": target_lang, "speaker_profile": speaker_profile})

    # --- STAGE 1: Whisper (blocking, run in executor) ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    loop = asyncio.get_event_loop()
    whisper_result = await loop.run_in_executor(None, whisper_engine.transcribe, tmp_path)
    os.unlink(tmp_path)

    whisper_ms = whisper_result["latency_ms"]
    transcript = whisper_result["text"]
    detected_lang = whisper_result.get("language", "unknown")

    emit_event(session_id, "TRANSCRIPT_DONE", latency_ms=whisper_ms,
               metadata={"transcript": transcript, "detected_lang": detected_lang})

    yield {
        "type": "transcript",
        "transcript": transcript,
        "detected_lang": detected_lang,
        "whisper_ms": whisper_ms
    }

    # Decide reference + enhancer based on profile
    active_reference = AASTHA_REFERENCE_PATH if speaker_profile == "aastha" else reference_wav

    # --- STAGE 2+3: Stream LLM sentence-by-sentence -> TTS each sentence ---
    sentence_buffer = ""
    full_response = ""
    first_chunk_logged = False
    llm_start = time.time()

    async for token in llm_engine.generate_stream(transcript, target_lang):
        sentence_buffer += token
        full_response += token

        if any(sentence_buffer.rstrip().endswith(e) for e in SENTENCE_ENDINGS):
            sentence = sentence_buffer.strip()
            sentence_buffer = ""
            if not sentence:
                continue

            if not first_chunk_logged:
                llm_ms = int((time.time() - llm_start) * 1000)
                emit_event(session_id, "LLM_FIRST_CHUNK", latency_ms=llm_ms,
                           metadata={"first_sentence": sentence})
                first_chunk_logged = True

            # TTS this sentence
            tts_start = time.time()
            audio_bytes = await loop.run_in_executor(
                None, _synthesize_sentence, sentence, target_lang,
                speaker_profile, active_reference
            )
            tts_ms = int((time.time() - tts_start) * 1000)

            yield {
                "type": "chunk",
                "audio_bytes": audio_bytes,
                "text": sentence,
                "tts_ms": tts_ms
            }

    # Flush any remaining text
    if sentence_buffer.strip():
        audio_bytes = await loop.run_in_executor(
            None, _synthesize_sentence, sentence_buffer.strip(), target_lang,
            speaker_profile, active_reference
        )
        yield {"type": "chunk", "audio_bytes": audio_bytes, "text": sentence_buffer.strip(), "tts_ms": 0}

    emit_event(session_id, "TTS_DONE", latency_ms=int((time.time() - llm_start) * 1000))

    total_ms = int((time.time() - pipeline_start) * 1000)
    emit_event(session_id, "SESSION_END", latency_ms=total_ms,
               metadata={"response_text": full_response, "cache_hit": False})
    flush()

    yield {
        "type": "done",
        "response_text": full_response,
        "session_id": session_id,
        "latency": {"whisper_ms": whisper_ms, "total_ms": total_ms}
    }


def _synthesize_sentence(text, target_lang, speaker_profile, reference_path) -> bytes:
    """Blocking TTS + enhancement for one sentence."""
    # 🎯 FIX: Access the model wrapper through the shared singleton instance safely
    raw_audio_list = tts_cloner.model.tts(
        text=text,
        speaker_wav=reference_path,
        language=target_lang
    )
    raw_array = np.array(raw_audio_list, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, raw_array, 24000, format="WAV")
    buf.seek(0)
    raw_bytes = buf.read()

    if speaker_profile == "aastha":
        return enhance_cloned_voice(raw_bytes)
    else:
        return enhance_cloned_voice_auto(raw_bytes, reference_audio_path=reference_path)