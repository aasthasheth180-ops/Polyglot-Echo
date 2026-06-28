import os
import soundfile as sf
from backend.tts_f5 import get_f5_engine

print("[*] Initiating F5-TTS isolated target pipeline evaluation...")

if not os.path.exists("audio/clip_1.wav"):
    print("[X] Execution blocked: audio/clip_1.wav reference track missing.")
else:
    engine = get_f5_engine("audio/clip_1.wav")

    audio_bytes = engine.synthesize(
        text="Hello, this is a test of F5-TTS voice cloning. How does this sound?",
        lang="en"
    )

    os.makedirs("audio", exist_ok=True)
    with open("audio/f5_test_output.wav", "wb") as f:
        f.write(audio_bytes)

    print("[✓] Execution complete! Verify target file at: audio/f5_test_output.wav")