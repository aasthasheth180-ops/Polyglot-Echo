# backend/stream_handler.py
import asyncio
import io
import time
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
from fastapi import WebSocket, WebSocketDisconnect
from backend.ai_client import transcribe_audio, synthesize_speech
from backend.llm_engine import llm_engine

# ── Constants (SYSTEMATIC FIXES APPLIED) ──────────────────────
SAMPLE_RATE        = 16000   
CHUNK_DURATION_MS  = 100     
SAMPLES_PER_CHUNK  = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
VAD_WINDOW_S       = 2.0     
VAD_SILENCE_THRESH = 0.018   # Problem 3 Fix: Higher threshold to ignore background room noise
MIN_SPEECH_CHUNKS  = 15      # Problem 3 Fix: Requires 1.5s of actual speech minimum
MIN_SPEECH_RMS     = 0.025   # Problem 3 Fix: New absolute minimum energy baseline
SENTENCE_ENDINGS   = {'.', '!', '?', '।', '。', '؟', ','} # Problem 2 Fix: Added soft comma boundary


# ── DSP Cleaning Pipeline ────────────────────────────────────
def clean_audio_chunk(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Applies high-pass filtering, normalization, noise gating, and de-essing."""
    y = y.astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9

    sos_hp = signal.butter(4, 80, btype='high', fs=sr, output='sos')
    y = signal.sosfilt(sos_hp, y)

    frame_size = int(sr * 0.01)
    for i in range(0, len(y) - frame_size, frame_size):
        frame = y[i:i + frame_size]
        if np.sqrt(np.mean(frame**2)) < VAD_SILENCE_THRESH:
            y[i:i + frame_size] = 0.0

    sos_ds = signal.butter(2, [5500, 7500], btype='bandstop', fs=sr, output='sos')
    y = signal.sosfilt(sos_ds, y)

    y = np.clip(y, -1.0, 1.0)
    return y


# ── Speech Validation Filter ──────────────────────────────────
def is_real_speech(audio_block: np.ndarray) -> bool:
    """
    Validate audio block before sending to Whisper to prevent hallucinations.
    Rejects ambient static, mouse clicks, and empty channel inputs.
    """
    if len(audio_block) < SAMPLE_RATE * 0.8:  
        return False
    
    rms = np.sqrt(np.mean(audio_block.astype(np.float32)**2))
    if rms < MIN_SPEECH_RMS:
        print(f"[VAD] Rejected block — RMS too low: {rms:.4f}")
        return False
    
    frame_size   = int(SAMPLE_RATE * 0.02)  # 20ms frames
    voiced_count = 0
    total_frames = 0
    
    for i in range(0, len(audio_block) - frame_size, frame_size):
        frame = audio_block[i:i+frame_size]
        if np.sqrt(np.mean(frame**2)) > VAD_SILENCE_THRESH:
            voiced_count += 1
        total_frames += 1
    
    voiced_ratio = voiced_count / max(total_frames, 1)
    if voiced_ratio < 0.30:
        print(f"[VAD] Rejected block — only {voiced_ratio:.0%} voiced")
        return False
    
    return True


def is_silence(chunk: np.ndarray) -> bool:
    rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
    return rms < VAD_SILENCE_THRESH


def webm_to_pcm(raw_bytes: bytes) -> np.ndarray:
    try:
        buf = io.BytesIO(raw_bytes)
        y, sr = sf.read(buf, dtype='float32')
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y
    except Exception:
        try:
            y_fallback = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            y_fallback = y_fallback / 32768.0
            return y_fallback
        except Exception:
            return np.zeros(SAMPLES_PER_CHUNK, dtype=np.float32)


def pcm_to_wav_bytes(y: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.read()


# ── Voice Activity Detection Buffer ──────────────────────────
class VADBuffer:
    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.speech_started = False
        self.silence_count  = 0
        self.SILENCE_LIMIT  = 8  

    def add(self, chunk: np.ndarray) -> bool:
        silent = is_silence(chunk)

        if not silent:
            self.speech_started = True
            self.silence_count  = 0
            self.chunks.append(chunk)
        elif self.speech_started:
            self.silence_count += 1
            self.chunks.append(chunk)

        total_duration = len(self.chunks) * CHUNK_DURATION_MS / 1000
        silence_ended  = (self.speech_started and self.silence_count >= self.SILENCE_LIMIT)
        window_full    = total_duration >= VAD_WINDOW_S

        return silence_ended or window_full

    def flush(self) -> np.ndarray | None:
        if len(self.chunks) < MIN_SPEECH_CHUNKS:
            self.reset()
            return None
        audio = np.concatenate(self.chunks)
        self.reset()
        return audio

    def reset(self):
        self.chunks        = []
        self.speech_started = False
        self.silence_count  = 0


# ── Main WebSocket Handler ────────────────────────────────────
async def handle_voice_stream(
    websocket: WebSocket,
    target_lang: str      = "en",
    speaker_profile: str  = "aastha",
    session_id: str       = None
):
    import uuid
    if session_id is None:
        session_id = str(uuid.uuid4())

    await websocket.accept()
    print(f"[Stream] Connected | session={session_id[:8]} | lang={target_lang}")

    vad_buffer    = VADBuffer()
    audio_queue   = asyncio.Queue()   
    text_queue    = asyncio.Queue()   
    pipeline_lock = asyncio.Lock() 
    is_connected  = True

    async def send_json(data: dict):
        try: await websocket.send_json(data)
        except Exception: pass

    async def send_bytes(data: bytes):
        try: await websocket.send_bytes(data)
        except Exception: pass

    # Worker 1: Gather audio packets asynchronously
    async def receive_audio():
        nonlocal is_connected
        try:
            while is_connected:
                message = await asyncio.wait_for(websocket.receive(), timeout=3600.0)
                if message.get("type") == "websocket.disconnect":
                    is_connected = False
                    break

                if "bytes" in message:
                    raw = message["bytes"]
                    if not raw: continue
                    pcm = webm_to_pcm(raw)
                    if vad_buffer.add(pcm):
                        audio_block = vad_buffer.flush()
                        if audio_block is not None:
                            await audio_queue.put(audio_block)
        except Exception:
            is_connected = False
        finally:
            leftover = vad_buffer.flush()
            if leftover is not None: await audio_queue.put(leftover)
            await audio_queue.put(None)  

    # Worker 2: DSP Processing + Whisper Decoding with Validation Check
    async def transcribe_worker():
        loop = asyncio.get_event_loop()
        try:
            while True:
                audio_block = await audio_queue.get()
                if audio_block is None: break

                # Check speech validation filter before spending processing overhead on Whisper
                is_valid = await loop.run_in_executor(None, is_real_speech, audio_block)
                if not is_valid:
                    print("[Stream] Audio block rejected by VAD validation — skipping Whisper")
                    continue

                cleaned = await loop.run_in_executor(None, clean_audio_chunk, audio_block)
                wav_bytes = await loop.run_in_executor(None, pcm_to_wav_bytes, cleaned)
                result = await loop.run_in_executor(None, transcribe_audio, wav_bytes)

                text = result.get("text", "").strip()
                lang = result.get("language", "unknown")

                if text:
                    print(f"[Stream] Transcribed ({lang}): '{text}'")
                    # Killer Feature: Send interim/final transcription payload instantly
                    await send_json({
                        "type": "transcript_interim",
                        "text": text,
                        "is_final": True
                    })
                    await text_queue.put((text, lang))
        except Exception as e:
            print(f"[Stream] Transcribe fault: {e}")
        finally:
            await text_queue.put(None)

    # Worker 3: Streamed Gemini Response Engine + Outbound TTS Generation
    async def llm_tts_worker():
        loop = asyncio.get_event_loop()
        try:
            while True:
                item = await text_queue.get()
                if item is None: break

                user_text, detected_lang = item
                
                async with pipeline_lock:
                    sentence_buffer = ""
                    full_response   = ""
                    first_chunk     = True

                    async for token in llm_engine.generate_stream(user_text, target_lang, session_id):
                        sentence_buffer += token
                        full_response   += token

                        if any(sentence_buffer.rstrip().endswith(e) for e in SENTENCE_ENDINGS):
                            sentence = sentence_buffer.strip()
                            sentence_buffer = ""  
                            if not sentence: continue

                            if first_chunk:
                                await send_json({"type": "llm_chunk", "text": sentence})
                                first_chunk = False
                            else:
                                await send_json({"type": "llm_chunk", "text": sentence})

                            voice_to_use = speaker_profile if speaker_profile else "default"
                            audio_bytes = await loop.run_in_executor(
                                None, synthesize_speech, sentence, target_lang, voice_to_use, "Hey, how have you been lately?"
                            )

                            if audio_bytes:
                                await send_json({"type": "audio_chunk_start", "size_bytes": len(audio_bytes)})
                                await send_bytes(audio_bytes)

                    if sentence_buffer.strip():
                        sentence = sentence_buffer.strip()
                        await send_json({"type": "llm_chunk", "text": sentence})
                        voice_to_use = speaker_profile if speaker_profile else "default"
                        audio_bytes = await loop.run_in_executor(
                            None, synthesize_speech, sentence, target_lang, voice_to_use, "Hey, how have you been lately?"
                        )
                        if audio_bytes:
                            await send_json({"type": "audio_chunk_start", "size_bytes": len(audio_bytes)})
                            await send_bytes(audio_bytes)

                    await send_json({"type": "response_done", "full_response": full_response})
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[Stream] LLM/TTS loop fault: {e}")
        finally:
            await send_json({"type": "stream_end"})

    try:
        await asyncio.gather(receive_audio(), transcribe_worker(), llm_tts_worker())
    finally:
        is_connected = False
        vad_buffer.reset()