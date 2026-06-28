# backend/ai_client.py (REFACTORED)
"""
AI Client with circuit breaker, exponential backoff, and robust retry logic.
"""

import os
import time
import requests
from typing import Optional, Tuple
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import (
    COLAB_URL, TTS_MAX_RETRIES, TTS_INITIAL_BACKOFF, TTS_MAX_BACKOFF, TTS_TIMEOUT,
    DEFAULT_REF_AUDIO_PATH, DEFAULT_REF_TEXT
)
from logger import get_logger, set_context, log_function_call

load_dotenv()

logger = get_logger(__name__)

print(f"[AI Client] Directing pipeline traffic to: {COLAB_URL}")


# ── Circuit Breaker ───────────────────────────────────────────
class CircuitBreaker:
    """
    Circuit breaker pattern for TTS service.
    Prevents cascading failures by failing fast when service is unhealthy.
    """
    
    def __init__(self, failure_threshold: int = 5, window_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.failures = []
        self.is_open = False
        self.last_failure_time = None
    
    def record_failure(self):
        """Record a failure."""
        now = time.time()
        self.failures.append(now)
        self.last_failure_time = now
        
        # Keep only failures within the window
        self.failures = [t for t in self.failures if now - t < self.window_seconds]
        
        # Check if threshold exceeded
        if len(self.failures) >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"[CircuitBreaker] OPEN after {len(self.failures)} failures")
    
    def record_success(self):
        """Record a success; reset failures."""
        self.failures.clear()
        self.is_open = False
        logger.debug("[CircuitBreaker] Reset after success")
    
    def should_allow_request(self) -> bool:
        """Check if circuit breaker allows request."""
        if not self.is_open:
            return True
        
        # Check if window has passed; reset if so
        if self.last_failure_time and time.time() - self.last_failure_time > self.window_seconds:
            self.failures.clear()
            self.is_open = False
            logger.info("[CircuitBreaker] Reset after window expired")
            return True
        
        return False


# ── Connection Pool Setup ──────────────────────────────────────
def create_session_with_retries():
    """
    Create a requests Session with smart retry logic:
    - Retries on connection errors, timeouts, and 5xx errors
    - Uses exponential backoff
    - Respects 429 (rate limit) and 503 (service unavailable)
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,  # Max retries for connection errors
        backoff_factor=0.5,  # 0.5s, 1s, 2s backoff
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# Global session and circuit breaker
_session = create_session_with_retries()
_circuit_breaker = CircuitBreaker(failure_threshold=5, window_seconds=60)


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance."""
    return _circuit_breaker


# ── Warmup Detection ───────────────────────────────────────────
@log_function_call
def wait_for_hf_space_ready(max_attempts: int = 15, delay_between_checks: int = 2) -> bool:
    """
    Poll the /health endpoint until both models report as loaded.
    
    Args:
        max_attempts: How many times to check
        delay_between_checks: Seconds to wait between checks
    
    Returns:
        True if ready, False if timeout
    """
    logger.info("[AI Client] Waiting for Hugging Face Space to initialize...")
    
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
                
                logger.debug(
                    f"[AI Client] Health {attempt}/{max_attempts}: "
                    f"whisper={whisper_ready}, f5tts={f5tts_ready}, ready={models_ready}"
                )
                
                if models_ready or (whisper_ready and f5tts_ready):
                    logger.info("[AI Client] ✅ Space ready; all models loaded")
                    return True
                else:
                    logger.debug(f"[AI Client] Waiting {delay_between_checks}s for models...")
                    time.sleep(delay_between_checks)
                    continue
        
        except requests.RequestException as e:
            logger.warning(f"[AI Client] Health check {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay_between_checks)
    
    logger.error(f"[AI Client] Space did not initialize within {max_attempts * delay_between_checks}s")
    return False


# ── Transcription ──────────────────────────────────────────────
@log_function_call
def transcribe_audio(wav_bytes: bytes, session_id: str = "unknown") -> dict:
    """
    Transcribe audio using remote Whisper service.
    
    Args:
        wav_bytes: WAV audio data
        session_id: For logging context
    
    Returns:
        {"text": "...", "language": "en", "latency_ms": 123}
        or {"text": "", "language": "error", "error": "..."}
    """
    set_context(session_id=session_id, operation="transcribe")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        logger.debug(f"[AI Client] Transcribing {len(wav_bytes)} bytes")
        
        response = _session.post(
            f"{COLAB_URL}/transcribe",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            headers=headers,
            timeout=TTS_TIMEOUT,
            allow_redirects=True
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "").strip()
            logger.info(f"[AI Client] Transcribe ✅: '{text[:50]}'")
            return result
        else:
            error_msg = f"HTTP {response.status_code}"
            logger.error(f"[AI Client] Transcribe ❌: {error_msg}")
            return {
                "text": "",
                "language": "error",
                "latency_ms": 0,
                "error": error_msg
            }
    
    except requests.Timeout:
        error_msg = "Timeout (>300s)"
        logger.error(f"[AI Client] Transcribe ❌: {error_msg}")
        return {
            "text": "",
            "language": "error",
            "latency_ms": 0,
            "error": error_msg
        }
    
    except Exception as e:
        logger.error(f"[AI Client] Transcribe failed: {e}", exc_info=True)
        return {
            "text": "",
            "language": "error",
            "latency_ms": 0,
            "error": str(e)
        }


# ── TTS with Exponential Backoff ───────────────────────────────
@log_function_call
def synthesize_speech(
    text: str,
    lang: str = "en",
    speaker_profile: str = "aastha",
    ref_text: str = DEFAULT_REF_TEXT,
    session_id: str = "unknown"
) -> Tuple[bytes, bool]:
    """
    Synthesize speech with exponential backoff and circuit breaker.
    
    Args:
        text: Text to synthesize
        lang: Language code
        speaker_profile: Voice profile
        ref_text: Short reference text (MUST be <15 words)
        session_id: For logging
    
    Returns:
        (audio_bytes, success_bool)
    """
    set_context(session_id=session_id, operation="synthesize")
    
    # Circuit breaker check
    if not _circuit_breaker.should_allow_request():
        logger.error("[AI Client] Circuit breaker OPEN; rejecting request")
        return b"", False
    
    backoff = TTS_INITIAL_BACKOFF
    
    for attempt in range(1, TTS_MAX_RETRIES + 1):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Content-Type": "application/json",
                "Accept": "*/*"
            }
            
            logger.debug(f"[AI Client] Synthesize attempt {attempt}/{TTS_MAX_RETRIES}: '{text[:30]}'")
            
            response = _session.post(
                f"{COLAB_URL}/synthesize",
                json={
                    "text": text,
                    "lang": lang,
                    "speaker_profile": speaker_profile,
                    "ref_text": ref_text
                },
                headers=headers,
                timeout=TTS_TIMEOUT,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                audio_bytes = response.content
                if len(audio_bytes) > 1000:  # Validate non-empty
                    logger.info(f"[AI Client] Synthesize ✅: {len(audio_bytes)} bytes")
                    _circuit_breaker.record_success()
                    return audio_bytes, True
                else:
                    logger.warning("[AI Client] TTS returned empty audio")
                    return b"", False
            
            elif response.status_code == 503:
                logger.warning(f"[AI Client] TTS busy (503), attempt {attempt}/{TTS_MAX_RETRIES}")
                _circuit_breaker.record_failure()
                
                if attempt < TTS_MAX_RETRIES:
                    logger.info(f"[AI Client] Waiting {backoff}s before retry...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, TTS_MAX_BACKOFF)  # Exponential backoff
                    continue
            
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"[AI Client] TTS error: {error_msg}")
                _circuit_breaker.record_failure()
                return b"", False
        
        except requests.Timeout:
            logger.warning(f"[AI Client] TTS timeout, attempt {attempt}/{TTS_MAX_RETRIES}")
            _circuit_breaker.record_failure()
            
            if attempt < TTS_MAX_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * 2, TTS_MAX_BACKOFF)
                continue
            
            return b"", False
        
        except Exception as e:
            logger.error(f"[AI Client] TTS exception: {e}", exc_info=True)
            _circuit_breaker.record_failure()
            return b"", False
    
    logger.error(f"[AI Client] TTS failed after {TTS_MAX_RETRIES} retries")
    return b"", False


# ── Health Check ───────────────────────────────────────────────
@log_function_call
def check_colab_health() -> bool:
    """Check if Hugging Face Space is healthy."""
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
            logger.debug(f"[AI Client] Health check: {health}")
            return models_ready
    
    except Exception as e:
        logger.error(f"[AI Client] Health check failed: {e}")
    
    return False


def get_hf_space_status() -> dict:
    """Get detailed status of HF Space."""
    try:
        response = _session.get(
            f"{COLAB_URL}/health",
            timeout=10,
            allow_redirects=True
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"[AI Client] Status check failed: {e}")
    
    return {
        "status": "unreachable",
        "error": "Cannot connect to HF Space"
    }