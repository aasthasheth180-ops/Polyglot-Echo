# backend/ai_client.py
import os
import time
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# Safely catch either environment variable name variation
COLAB_URL = os.getenv("COLAB_AI_URL") or os.getenv("COLAB_URL") or "https://aaasthasss-polyglot-echo-ai.hf.space"

# Strip any accidental trailing slashes that break route concatenation
COLAB_URL = COLAB_URL.rstrip("/")

print(f"[AI Client] Directing pipeline traffic to: {COLAB_URL}")

# ── Connection Pool Setup ──────────────────────────────────────
# This reuses TCP connections across multiple requests
# and implements exponential backoff on failures
def create_session_with_retries():
    """
    Create a requests Session with smart retry logic:
    - Retries on connection errors, timeouts, and 5xx errors
    - Uses exponential backoff (0.1s, 0.2s, 0.4s, 0.8s, 1.6s)
    - Respects 429 (rate limit) and 503 (service unavailable) responses
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=5,                                    # Max 5 retries per request
        backoff_factor=0.5,                         # Exponential backoff multiplier
        status_forcelist=[429, 500, 502, 503, 504], # Retry on these HTTP codes
        allowed_methods=["GET", "POST", "PUT"]      # Which HTTP methods to retry
    )
    
    # Attach adapter to both HTTP and HTTPS
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Create a persistent session for the lifetime of the backend process
_session = create_session_with_retries()

# ── Warmup Detection ───────────────────────────────────────────
def wait_for_hf_space_ready(max_attempts=15, delay_between_checks=2):
    """
    Poll the /health endpoint until both models report as loaded.
    
    Args:
        max_attempts: How many times to check (15 * 2s = 30s max wait)
        delay_between_checks: Seconds to wait between health checks
    
    This handles HF Space cold-starts gracefully.
    """
    print(f"[AI Client] Waiting for Hugging Face Space to fully initialize...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = _session.get(
                f"{COLAB_URL}/health",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            
            if response.status_code == 200:
                health = response.json()
                whisper_ready = health.get("whisper_loaded", False)
                f5tts_ready = health.get("f5tts_loaded", False)
                models_ready = health.get("models_ready", False)
                
                print(f"[AI Client] Health check {attempt}/{max_attempts}: "
                      f"whisper={whisper_ready}, f5tts={f5tts_ready}, models_ready={models_ready}")
                
                # Both models fully loaded and ready
                if models_ready or (whisper_ready and f5tts_ready):
                    print(f"[AI Client] ✅ Space ready! All models loaded.")
                    return True
                else:
                    print(f"[AI Client] Models still loading, retrying in {delay_between_checks}s...")
                    time.sleep(delay_between_checks)
                    continue
                    
        except requests.RequestException as e:
            print(f"[AI Client] Health check {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay_between_checks)
                continue
    
    print(f"[AI Client] ⚠️  Space did not fully initialize within {max_attempts * delay_between_checks}s")
    return False


def transcribe_audio(wav_bytes: bytes) -> dict:
    """Sends recorded speech to Hugging Face Whisper container."""
    try:
        # Standard User-Agent prevents Hugging Face's security layer from blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        print(f"[AI Client] Transcribe: Sending {len(wav_bytes)} bytes to {COLAB_URL}/transcribe")
        
        response = _session.post(
            f"{COLAB_URL}/transcribe",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            headers=headers,
            timeout=300,  # 5 minutes for cold-start model loads
            allow_redirects=True
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[AI Client] Transcribe ✅: '{result.get('text', '')[:60]}'")
            return result
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"[AI Client] Transcribe ❌: {error_msg}")
            return {"text": "", "language": "error", "latency_ms": 0, "error": error_msg}
            
    except requests.Timeout:
        error_msg = "Request timeout (>300s). Space may be cold-starting."
        print(f"[AI Client] Transcribe ❌: {error_msg}")
        return {"text": "", "language": "error", "latency_ms": 0, "error": error_msg}
        
    except requests.RequestException as e:
        error_msg = str(e)
        print(f"[AI Client] Transcribe ❌: Connection failed: {error_msg}")
        return {"text": "", "language": "error", "latency_ms": 0, "error": error_msg}
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[AI Client] Transcribe ❌: {error_msg}")
        return {"text": "", "language": "error", "latency_ms": 0, "error": error_msg}


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
        
        print(f"[AI Client] Synthesize: Sending '{text[:40]}' to {COLAB_URL}/synthesize")
        
        response = _session.post(
            f"{COLAB_URL}/synthesize",
            json={
                "text": text,
                "lang": lang,
                "speaker_profile": speaker_profile,
                "ref_text": ref_text
            },
            headers=headers,
            timeout=300,  # 5 minutes to allow model cold-start
            allow_redirects=True
        )
        
        if response.status_code == 200:
            print(f"[AI Client] Synthesize ✅: Generated {len(response.content)} bytes")
            return response.content  # High-fidelity WAV audio data
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"[AI Client] Synthesize ❌: {error_msg}")
            return b""
            
    except requests.Timeout:
        error_msg = "Request timeout (>300s). Space may be cold-starting."
        print(f"[AI Client] Synthesize ❌: {error_msg}")
        return b""
        
    except requests.RequestException as e:
        error_msg = str(e)
        print(f"[AI Client] Synthesize ❌: Connection failed: {error_msg}")
        return b""
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[AI Client] Synthesize ❌: {error_msg}")
        return b""


def check_colab_health() -> bool:
    """
    Verifies connection to Hugging Face health diagnostic endpoint.
    Returns True if all models are fully loaded and ready.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = _session.get(
            f"{COLAB_URL}/health",
            headers=headers,
            timeout=15,
            allow_redirects=True
        )
        if response.status_code == 200:
            health = response.json()
            models_ready = health.get("models_ready", False)
            print(f"[AI Client] ✅ Health: {health}")
            return models_ready
    except Exception as e:
        print(f"[AI Client] ❌ Cloud AI Nodes unreachable at {COLAB_URL}: {e}")
    return False