import os
from dotenv import load_dotenv
load_dotenv()

# Safely check configurations defaulting seamlessly to standard setups if blank
TTS_BACKEND = os.getenv("TTS_BACKEND", "xtts").lower().strip()

print(f"[TTS Selector] Current active processing engine layer: {TTS_BACKEND.upper()}")

if TTS_BACKEND == "f5":
    from backend.tts_f5 import get_f5_engine
    _engine = get_f5_engine()

    def synthesize(text: str, lang: str, speaker_wav: str = None) -> bytes:
        return _engine.synthesize(text, lang, speaker_wav)

    def update_reference(ref_path: str, ref_text: str = None):
        _engine.update_reference(ref_path, ref_text)

elif TTS_BACKEND == "xtts":
    from backend.tts_handler import get_cloner
    _cloner = get_cloner()

    def synthesize(text: str, lang: str, speaker_wav: str = None) -> bytes:
        import io
        import numpy as np
        import soundfile as sf
        
        raw_list = _cloner.tts.tts(
            text=text,
            speaker_wav=speaker_wav or "audio/clip_1.wav",
            language=lang
        )
        raw_array = np.array(raw_list, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, raw_array, 24000, format="WAV")
        buf.seek(0)
        return buf.read()

    def update_reference(ref_path: str, ref_text: str = None):
        pass  # XTTS parses and applies profile data dynamically during evaluation

else:
    raise ValueError(f"Target backend parameter mapping error: '{TTS_BACKEND}'. Must be 'f5' or 'xtts'")