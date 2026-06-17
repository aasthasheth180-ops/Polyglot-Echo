import torch
import torch.serialization

# PyTorch 2.6+ blocks loading TTS model weights by default.
# XTTS v2 uses the old serialization format — we allowlist it explicitly.
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.config import BaseAudioConfig, BaseDatasetConfig
    torch.serialization.add_safe_globals([
        XttsConfig,
        Xtts,
        BaseAudioConfig,
        BaseDatasetConfig
    ])
    print("[*] PyTorch safe globals registered for XTTS v2")
except Exception as e:
    print(f"[*] Safe globals registration skipped: {e}")


import os
import sys
import io
import numpy as np
import soundfile as sf
from TTS.api import TTS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.voice_enhancer import enhance_cloned_voice

os.environ["COQUI_TOS_AGREED"] = "1"

class VoiceCloner:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing XTTS Model on: {self.device.upper()}")
        try:
            self.tts = TTS(model_name).to(self.device)
            print(f"[✓] Voice Engine successfully loaded into VRAM.")
        except Exception as e:
            print(f"[X] Failed to load model: {e}")
            sys.exit(1)

    def clone_voice(self, text_to_speak: str, reference_audio_path, output_wav_path: str, language: str = "en"):
        if isinstance(reference_audio_path, list):
            for path in reference_audio_path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing: {path}")
        else:
            if not os.path.exists(reference_audio_path):
                raise FileNotFoundError(f"Missing: {reference_audio_path}")

        output_dir = os.path.dirname(output_wav_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            print("[*] Generating raw audio vectors via XTTS VRAM context...")

            raw_audio_list = self.tts.tts(
                text=text_to_speak,
                speaker_wav=reference_audio_path,
                language=language
            )
            raw_array = np.array(raw_audio_list, dtype=np.float32)
            orig_sr = 24000  # XTTS v2 native sample rate

            print("[*] Passing audio vector through DSP Calibration Pipeline...")
            mem_buffer = io.BytesIO()
            sf.write(mem_buffer, raw_array, orig_sr, format="WAV")
            mem_buffer.seek(0)

            enhanced_bytes = enhance_cloned_voice(mem_buffer.read())

            with open(output_wav_path, "wb") as f:
                f.write(enhanced_bytes)

            print(f"[✓] Complete pipeline success! Enhanced audio saved to: {output_wav_path}")
            return True

        except Exception as e:
            print(f"[X] Pipeline execution failure: {e}")
            return False


# =====================================================================
# 🎯 SINGLETON PATTERN: Prevents duplicate heavy model memory loads
# =====================================================================
_cloner_instance = None

def get_cloner():
    global _cloner_instance
    if _cloner_instance is None:
        print("[*] Memory Optimizer: Initializing VoiceCloner instance for the FIRST time...")
        _cloner_instance = VoiceCloner()
    else:
        print("[✓] Memory Optimizer: Reusing already loaded VoiceCloner memory instance.")
    return _cloner_instance


if __name__ == "__main__":
    # 🎯 FIX: Testing script updated to use the shared singleton initialization
    cloner = get_cloner()

    long_test_text = (
        "Hey! I wanted to check in and see how everything is moving along with our timeline today. "
        "Honestly, when you look at how fast modern technology is developing, it is completely crazy "
        "how much data we handle on a daily basis. Sometimes it feels like we are just moving pieces around, "
        "but when you actually build a full-stack system from the ground up, everything clicks. "
        "We need to make sure our pipeline handles long audio formats smoothly without any delay or audio clipping. "
        "Once we lock down these foundational components, I want to show you how cleanly we can stream this data "
        "across different networks, languages, and accents. Let me know what you think of this voice test, "
        "and let's check the output together!"
    )

    print("\n🚀 Running a long-form voice synthesis test (English)...")
    cloner.clone_voice(
        text_to_speak=long_test_text,
        reference_audio_path="audio/clip_1.wav",   
        output_wav_path="audio/cloned_output_multi.wav",
        language="en"
    )