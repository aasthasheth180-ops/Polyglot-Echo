# backend/ai_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Safely catch either environment variable name variation
COLAB_URL = os.getenv("COLAB_AI_URL") or os.getenv("COLAB_URL") or "https://aaasthasss-polyglot-echo-ai.hf.space"

# Strip any accidental trailing slashes that break route concatenation
COLAB_URL = COLAB_URL.rstrip("/")

print(f"[AI Client] Directing pipeline traffic to: {COLAB_URL}")

def transcribe_audio(wav_bytes: bytes) -> dict:
    """Sends recorded speech to Hugging Face Whisper container."""
    try:
        # A standard User-Agent prevents Hugging Face's security layer from blocking the request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        response = requests.post(
            f"{COLAB_URL}/transcribe",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            headers=headers,
            timeout=150  # Generous headroom for cold-start weight pulls
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[AI Client] Transcribe error code: {response.status_code} | Body: {response.text}")
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
    """Sends text to Hugging Face F5-TTS and returns clean WAV bytes."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json",
            "Accept": "*/*"
        }
        
        response = requests.post(
            f"{COLAB_URL}/synthesize",
            json={
                "text": text,
                "lang": lang,
                "speaker_profile": speaker_profile,
                "ref_text": ref_text
            },
            headers=headers,
            timeout=300  # 5 full minutes to let F5-TTS download its models safely
        )
        
        if response.status_code == 200:
            return response.content  # Returns raw high-fidelity WAV audio data stream
        else:
            print(f"[AI Client] Synthesize error code: {response.status_code} | Body: {response.text}")
            return b""
            
    except Exception as e:
        print(f"[AI Client] Synthesize connection failed: {e}")
        return b""


def check_colab_health() -> bool:
    """Verifies connection to Hugging Face health diagnostic endpoint."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(f"{COLAB_URL}/health", headers=headers, timeout=15)
        if response.status_code == 200:
            print(f"[AI Client] Connected to Hugging Face AI Nodes successfully: {response.json()}")
            return True
    except Exception as e:
        print(f"[AI Client] Cloud AI Nodes unreachable at {COLAB_URL}: {e}")
    return False