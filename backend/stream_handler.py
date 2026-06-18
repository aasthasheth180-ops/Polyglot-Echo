# backend/stream_handler.py
import asyncio
import json
import time
import numpy as np
import scipy.signal as signal
from fastapi import WebSocket

from backend.ai_client import transcribe_audio, synthesize_speech
from backend.llm_engine import llm_engine
from backend.pipeline import trim_audio_for_whisper
from pipeline.kafka_producer import emit_event, flush

# ── Constants & Configuration ─────────────────────────────────
SAMPLE_RATE        = 16000   # Whisper native input frequency
CHUNK_DURATION_MS  = 100     # Browser streaming frame payload sizing
SAMPLES_PER_CHUNK  = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
VAD_WINDOW_S       = 2.0     # Maximum pre-speech window length
VAD_SILENCE_THRESH = 0.025   # Aggressive noise gate threshold to kill mic hum
MIN_SPEECH_CHUNKS  = 8       # Minimum continuous speech frames (800ms)
SENTENCE_ENDINGS   = {'.', '!', '?', '।', '。', '؟'}

# Common Whisper hallucinations triggered by ambient static noise
WHISPER_HALLUCINATIONS = {
    "thank you.", "thank you", "thank you for watching.", 
    "thank you very much.", "you", "bye.", "sh"
}

# ── Audio Matrix DSP Utilities ────────────────────────────────
def apply_audio_dsp_matrix(audio_data: np.ndarray) -> np.ndarray:
    """
    Applies custom digital signal processing to clean up raw browser audio.
    1. High-Pass Filter (80Hz) to remove low-frequency room rumble.
    2. Notch Filter (6000Hz) to de-ess harsh sibilance.
    """
    try:
        # 1. 80Hz High-Pass Butterworth Filter
        b_hp, a_hp = signal.butter(4, 80 / (SAMPLE_RATE / 2), btype='high')
        filtered_audio = signal.filtfilt(b_hp, a_hp, audio_data)

        # 2. 6000Hz Notch Filter (De-esser)
        b_notch, a_notch = signal.iirnotch(6000 / (SAMPLE_RATE / 2), 30.0)
        final_audio = signal.filtfilt(b_notch, a_notch, filtered_audio)
        
        return final_audio.astype(np.float32)
    except Exception as dsp_err:
        print(f"[DSP Matrix] Warning, math optimization fell back: {dsp_err}")
        return audio_data


def convert_float32_to_wav_pcm(audio_np: np.ndarray) -> bytes:
    """Wraps clean structured float arrays into executable high-fidelity WAV bytes."""
    import io
    import wave
    
    # Scale float32 (-1.0 to 1.0) values to signed 16-bit PCM integers
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16 bits
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
        
    wav_buffer.seek(0)
    return wav_buffer.read()

# ── Worker Engine 1: Audio Aggregation & VAD ──────────────────
async def audio_transcribe_worker(
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    send_json,
    session_id: str
):
    """
    Asynchronously monitors the raw PCM queue, accumulates frames,
    applies DSP, checks noise gate levels, and dispatches to Whisper.
    """
    audio_buffer = []
    speech_detected = False
    continuous_speech_chunks = 0
    
    print("[Stream Worker] Transcription Engine Online.")

    while True:
        chunk = await audio_queue.get()
        if chunk is None:
            break
            
        audio_buffer.append(chunk)
        
        # Calculate root-mean-square energy to evaluate ambient volume levels
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0
        
        if rms > VAD_SILENCE_THRESH:
            continuous_speech_chunks += 1
            if continuous_speech_chunks >= MIN_SPEECH_CHUNKS:
                speech_detected = True
        else:
            # Decay speech frame windows slowly to accommodate pauses between words
            continuous_speech_chunks = max(0, continuous_speech_chunks - 1)

        # Process buffer window once target size is reached
        if len(audio_buffer) >= int(VAD_WINDOW_S * 1000 / CHUNK_DURATION_MS):
            if speech_detected:
                # Concatenate, process, and pass out to cloud AI network node
                full_audio = np.concatenate(audio_buffer)
                cleaned_audio = apply_audio_dsp_matrix(full_audio)
                wav_bytes = convert_float32_to_wav_pcm(cleaned_audio)
                
                # Protect input chunk constraints
                trimmed_bytes = trim_audio_for_whisper(wav_bytes, max_seconds=30)
                
                t_start = time.time()
                result = transcribe_audio(trimmed_bytes)
                text = result.get("text", "").strip()
                lang = result.get("language", "unknown")
                w_ms = int((time.time() - t_start) * 1000)

                # Check if phrase is a typical Whisper static hallucination
                clean_text_lower = text.lower().strip(" .?!,")
                is_hallucination = clean_text_lower in {h.strip(" .?!,") for h in WHISPER_HALLUCINATIONS}

                if text and not is_hallucination:
                    print(f"[Stream] Transcribed ({lang}): '{text[:60]}' | {w_ms}ms")
                    
                    emit_event(session_id, "STREAM_TRANSCRIPT_DONE", latency_ms=w_ms, 
                               metadata={"text": text, "lang": lang})
                    
                    await send_json({
                        "type": "transcript",
                        "text": text,
                        "language": lang,
                        "latency_ms": w_ms
                    })
                    # Push downstream to LLM Processing Queue
                    await text_queue.put((text, lang))
                else:
                    print(f"[Stream] Noise gate bypassed or hallucinated phrase ('{text}') — skipped")
                
                # Reset State
                speech_detected = False
                continuous_speech_chunks = 0
                
            # Clear historical sliding window state safely
            audio_buffer.clear()
            audio_queue.task_done()

# ── Worker Engine 2: LLM Streaming & TTS Orchestration ────────
async def llm_tts_worker(
    text_queue: asyncio.Queue,
    websocket: WebSocket,
    speaker_profile: str,
    session_id: str
):
    """
    Monitors valid transcripts, triggers the asynchronous token stream from Gemini, 
    and groups words into logical sentences before firing them off to F5-TTS.
    """
    print("[Stream Worker] LLM to TTS Streaming Orchestrator Online.")
    
    while True:
        queue_payload = await text_queue.get()
        if queue_payload is None:
            break
            
        user_text, target_lang = queue_payload
        current_sentence = []
        
        try:
            # Fire the active asynchronous pipeline stream generator
            async for token in llm_engine.generate_stream(user_text, target_lang, session_id):
                await websocket.send_json({
                    "type": "llm_token",
                    "text": token
                })
                
                current_sentence.append(token)
                
                # Check if current chunk contains a logical sentence ending
                if any(ending in token for ending in SENTENCE_ENDINGS):
                    sentence = "".join(current_sentence).strip()
                    if sentence:
                        print(f"[Stream] Streaming synthesis for sentence block: '{sentence}'")
                        
                        t_start = time.time()
                        audio_bytes = synthesize_speech(
                            text=sentence,
                            lang=target_lang,
                            speaker_profile=speaker_profile,
                            ref_text="Hey, how have you been lately?"
                        )
                        tts_ms = int((time.time() - t_start) * 1000)
                        
                        emit_event(session_id, "STREAM_TTS_CHUNK_DONE", latency_ms=tts_ms)
                        
                        # Send raw audio binary payload straight down the WebSocket wire
                        await websocket.send_bytes(audio_bytes)
                        
                    current_sentence.clear()
                    
            # Synthesis cleanup for any remaining tokens without trailing punctuation
            remainder = "".join(current_sentence).strip()
            if remainder:
                audio_bytes = synthesize_speech(remainder, target_lang, speaker_profile, "Hey, how have you been lately?")
                await websocket.send_bytes(audio_bytes)
                
        except Exception as stream_err:
            print(f"[Stream Worker] Error executing pipeline stream path: {stream_err}")
        finally:
            text_queue.task_done()

# ── Primary Router Connection Handler ─────────────────────────
async def handle_voice_stream(
    websocket: WebSocket,
    target_lang: str,
    speaker_profile: str,
    session_id: str = None
):
    """Main persistent lifecycle loop for client WebSocket connections."""
    await websocket.accept()
    
    if not session_id:
        import uuid
        session_id = f"stream_{uuid.uuid4().hex[:8]}"
        
    print(f"[Stream] Persistent pipeline channel established | session={session_id}")
    emit_event(session_id, "STREAM_SESSION_STARTED", metadata={"lang": target_lang, "speaker": speaker_profile})
    
    audio_queue = asyncio.Queue()
    text_queue = asyncio.Queue()
    
    # Helper lambda to avoid threading reference collisions across tasks
    def send_json_sync(data):
        return asyncio.create_task(websocket.send_json(data))

    # Spawn background co-routines
    transcribe_task = asyncio.create_task(
        audio_transcribe_worker(audio_queue, text_queue, send_json_sync, session_id)
    )
    llm_tts_task = asyncio.create_task(
        llm_tts_worker(text_queue, websocket, speaker_profile, session_id)
    )

    try:
        while True:
            # Await raw data chunks from the frontend client microphone
            message = await websocket.receive()
            
            if "bytes" in message:
                raw_bytes = message["bytes"]
                # Convert binary buffer straight into floating-point numpy vectors
                float_chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                await audio_queue.put(float_chunk)
                
            elif "text" in message:
                # Handle control commands or heartbeats
                data = json.loads(message["text"])
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
    except Exception as conn_err:
        print(f"[Stream] Session disconnected or interrupted: {conn_err}")
    finally:
        # Gracefully wind down workers and clean up memory
        print(f"[Stream] Closing backend worker channels for session: {session_id}")
        await audio_queue.put(None)
        await text_queue.put(None)
        
        # Await thread joining terminations
        await asyncio.gather(transcribe_task, llm_tts_task, return_exceptions=True)
        emit_event(session_id, "STREAM_SESSION_ENDED")
        flush()