# backend/health_check.py
"""
Health check and model readiness validation.
Ensures models are loaded before accepting requests.
"""

import time
import requests
from typing import Dict, Tuple
from logger import get_logger
from config import COLAB_URL

logger = get_logger(__name__)


class ModelReadiness:
    """Tracks model loading state with timestamps."""
    
    def __init__(self):
        self.whisper_ready = False
        self.f5tts_ready = False
        self.whisper_load_time = None
        self.f5tts_load_time = None
        self.last_health_check = None
    
    def mark_whisper_ready(self, load_time_seconds: float):
        """Mark Whisper as loaded."""
        self.whisper_ready = True
        self.whisper_load_time = load_time_seconds
        logger.info(f"[Readiness] Whisper ready in {load_time_seconds:.2f}s")
    
    def mark_f5tts_ready(self, load_time_seconds: float):
        """Mark F5-TTS as loaded."""
        self.f5tts_ready = True
        self.f5tts_load_time = load_time_seconds
        logger.info(f"[Readiness] F5-TTS ready in {load_time_seconds:.2f}s")
    
    def all_ready(self) -> bool:
        """Check if all models are loaded."""
        return self.whisper_ready and self.f5tts_ready
    
    def get_status(self) -> Dict:
        """Get detailed readiness status."""
        return {
            "status": "ok" if self.all_ready() else "initializing",
            "whisper_ready": self.whisper_ready,
            "f5tts_ready": self.f5tts_ready,
            "models_ready": self.all_ready(),
            "whisper_load_ms": int(self.whisper_load_time * 1000) if self.whisper_load_time else None,
            "f5tts_load_ms": int(self.f5tts_load_time * 1000) if self.f5tts_load_time else None,
            "timestamp": time.time()
        }
    
    def reset(self):
        """Reset readiness flags (for testing)."""
        self.whisper_ready = False
        self.f5tts_ready = False
        self.whisper_load_time = None
        self.f5tts_load_time = None


# Global readiness tracker
_readiness = ModelReadiness()


def get_readiness() -> ModelReadiness:
    """Get the global readiness tracker."""
    return _readiness


def is_ready() -> bool:
    """Check if system is ready to serve requests."""
    return _readiness.all_ready()


def require_ready() -> Tuple[bool, Dict]:
    """
    Check readiness and return (is_ready, status_dict).
    Use this in endpoint decorators to validate before processing.
    """
    status = _readiness.get_status()
    return _readiness.all_ready(), status


def check_endpoint_readiness(endpoint_name: str, required_models: list = None) -> Tuple[bool, str]:
    """
    Checks readiness by pinging the remote Hugging Face Space health endpoint.
    """
    try:
        # Ping the remote HF Space health endpoint
        response = requests.get(f"{COLAB_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("models_ready"):
                return True, ""
            else:
                return False, f"Remote models initializing: {data.get('status')}"
        return False, "Voice worker unreachable"
    except Exception as e:
        logger.error(f"[Readiness] Failed to connect to HF Space: {e}")
        return False, "Voice worker connection failed"