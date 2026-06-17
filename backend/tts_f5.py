import os
import io
import time
import torch
import soundfile as sf

os.environ["COQUI_TOS_AGREED"] = "1"

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}

class F5TTSEngine:
    def __init__(
        self,
        speaker_ref_path: str = "audio/clip_1.wav",
        ref_text: str = "Hello, I am testing the voice cloning system today."
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_ref = speaker_ref_path
        self.ref_text = ref_text
        self.f5tts = None  # Unified API wrapper instance
        print(f"[F5-TTS] Engine initialized on {self.device.upper()} (Lazy loading enabled)")

    def _load_model(self):
        """Lazy load — loads the official F5TTS API instance on first use."""
        if self.f5tts is not None:
            return  

        print("[F5-TTS] Allocating model weights into memory...")
        start = time.time()

        try:
            # 🎯 FIX: Using the high-level official API pipeline class to handle loading safely
            from f5_tts.api import F5TTS
            self.f5tts = F5TTS(device=self.device)
            print(f"[F5-TTS] Model structurally ready in {time.time()-start:.1f}s on {self.device.upper()}")
        except Exception as e:
            print(f"[F5-TTS] Critical loading failure: {e}")
            raise

    def synthesize(
        self,
        text: str,
        lang: str = "en",
        speaker_wav: str = None
    ) -> bytes:
        """
        Drop-in structural signature replacement for pipeline modules.
        Returns clean WAV byte streams.
        """
        if lang not in SUPPORTED_LANGUAGES:
            print(f"[F5-TTS] '{lang}' target outside scope, redirecting fallback to 'en'")
            lang = "en"

        self._load_model()

        ref = speaker_wav or self.speaker_ref
        if not os.path.exists(ref):
            raise FileNotFoundError(f"[F5-TTS] Reference path target missing: {ref}")

        start = time.time()
        print(f"[F5-TTS] Synthesizing {len(text)} characters into language stream: '{lang}'...")

        try:
            audio_wave, sample_rate, _ = self.f5tts.infer(
                ref_file=ref,              # ← Changed this parameter key
                ref_text=self.ref_text,
                gen_text=text,
                speed=1.0
            )
        except Exception as e:
            print(f"[F5-TTS] Inference execution block crashed: {e}")
            raise


        latency_ms = int((time.time() - start) * 1000)
        print(f"[F5-TTS] Synthesis completed cleanly in {latency_ms}ms")

        buffer = io.BytesIO()
        sf.write(buffer, audio_wave, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def update_reference(self, new_ref_path: str, new_ref_text: str = None):
        self.speaker_ref = new_ref_path
        if new_ref_text:
            self.ref_text = new_ref_text
        print(f"[F5-TTS] Memory reference routing swapped to: {new_ref_path}")


# =====================================================================
# 🎯 SINGLETON STRUCTURE
# =====================================================================
_f5_instance = None

def get_f5_engine(speaker_ref_path: str = "audio/clip_1.wav") -> F5TTSEngine:
    global _f5_instance
    if _f5_instance is None:
        _f5_instance = F5TTSEngine(speaker_ref_path=speaker_ref_path)
    return _f5_instance