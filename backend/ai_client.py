# backend/ai_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

COLAB_URL = os.getenv("COLAB_AI_URL", "http://localhost:5000")

def transcribe_audio(wav_bytes: bytes) -> dict:
    """Sends recorded speech to Colab Whisper with bypass headers."""
    try:
        # 🎯 The essential bypass passport header
        headers = {
            "ngrok-skip-browser-warning": "true"
        }
        response = requests.post(
            f"{COLAB_URL}/transcribe",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            headers=headers,
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[AI Client] Transcribe error code: {response.status_code}")
            return {"text": "", "language": "error", "latency_ms": 0}
    except Exception as e:
        print(f"[AI Client] Transcribe connection failed: {e}")
        return {"text": "", "language": "error", "latency_ms": 0, "error": str(e)}


def synthesize_speech(
    text: str,
    lang: str = "en",
    speaker_profile: str = "aastha",
    ref_text: str = "Hey, how have you been lately?"
) -> bytes:
    """Sends text to Colab F5-TTS with bypass headers and returns clean WAV bytes."""
    try:
        # 🎯 The essential bypass passport headers
        headers = {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{COLAB_URL}/synthesize",
            json={
                "text": text,
                "lang": lang,
                "speaker_profile": speaker_profile,
                "ref_text": ref_text,
                "apply_dsp": True  # Tells Colab server to execute your custom audio matrix tuning
            },
            headers=headers,
            timeout=180
        )
        if response.status_code == 200:
            return response.content  # Returns raw high-fidelity WAV audio data stream
        else:
            print(f"[AI Client] Synthesize error code: {response.status_code}")
            return b""
    except Exception as e:
        print(f"[AI Client] Synthesize connection failed: {e}")
        return b""


def check_colab_health() -> bool:
    """Verifies connection to Colab health diagnostic point."""
    try:
        headers = {"ngrok-skip-browser-warning": "true"}
        response = requests.get(f"{COLAB_URL}/health", headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"[AI Client] Connected to Cloud AI Nodes successfully: {response.json()}")
            return True
    except Exception as e:
        print(f"[AI Client] Cloud AI Nodes unreachable: {e}")
    return False