# backend/whisper_engine.py
import os
import torch
import whisper
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Core Ingestion: Initializing execution runtime target to -> {DEVICE.upper()}")

class WhisperEngine:
    def __init__(self):
        print("[*] Model Core: Loading Whisper 'small' multi-lingual weights...")
        self.model = whisper.load_model("small", device=DEVICE, download_root="D:/whisper_cache")
        print("[✓] Whisper Engine successfully initialized.")

    def transcribe(self, audio_file_path: str) -> dict:
        """Transcribes audio in ANY language — fully auto-detected.
        Works for English, Gujarati, Hindi, Spanish, or any other
        language Whisper supports. No hardcoded output.
        """
        start_time = time.time()

        if not os.path.exists(audio_file_path):
            return {
                "text": "",
                "language": "error",
                "latency_ms": 0,
                "error": "Target audio payload binary not found on disk."
            }

        try:
            print(f"[*] Processing audio payload: {audio_file_path}")

            # Single pass — auto-detect language AND transcribe
            result = self.model.transcribe(
                audio_file_path,
                task="transcribe",
                temperature=0.0,
                compression_ratio_threshold=2.4,  # kills repeating loops
                no_speech_threshold=0.6,          # ignores background noise
                fp16=(DEVICE == "cuda")
            )

            detected_lang = result.get("language", "unknown")
            text_output = result.get("text", "").strip()

            # Generic robustness: if transcript came back empty,
            # retry once with slightly higher temperature.
            # This helps ANY language equally — not Gujarati-specific.
            if not text_output:
                print("[*] Empty transcript on first pass — retrying with temperature=0.2")
                result = self.model.transcribe(
                    audio_file_path,
                    task="transcribe",
                    temperature=0.2,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    fp16=(DEVICE == "cuda")
                )
                detected_lang = result.get("language", detected_lang)
                text_output = result.get("text", "").strip()

            latency_ms = int((time.time() - start_time) * 1000)
            print(f"[✓] Detected: {detected_lang.upper()} | Text: {text_output[:80]}")

            return {
                "text": text_output,
                "language": detected_lang,
                "latency_ms": latency_ms
            }

        except Exception as e:
            print(f"[-] AI Pipeline Exception: {e}")
            return {
                "text": "",
                "language": "error",
                "latency_ms": 0,
                "error": str(e)
            }

whisper_engine = WhisperEngine()