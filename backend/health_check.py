# backend/health_check.py
"""
Health check and model readiness validation.
Ensures models are loaded before accepting requests.
"""

import time
from typing import Dict, Tuple
from logger import get_logger

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
    Pre-flight check for specific endpoint.
    
    Args:
        endpoint_name: Name of endpoint (for logging)
        required_models: List of required models ['whisper', 'f5tts', 'all']
    
    Returns:
        (is_ready, error_message)
    """
    if required_models is None:
        required_models = ['whisper', 'f5tts']
    
    if 'all' in required_models:
        required_models = ['whisper', 'f5tts']
    
    missing = []
    
    if 'whisper' in required_models and not _readiness.whisper_ready:
        missing.append("Whisper")
    
    if 'f5tts' in required_models and not _readiness.f5tts_ready:
        missing.append("F5-TTS")
    
    if missing:
        error_msg = f"{endpoint_name}: Missing models: {', '.join(missing)}"
        logger.warning(f"[Readiness] {error_msg}")
        return False, error_msg
    
    logger.debug(f"[Readiness] {endpoint_name} pre-flight check passed")
    return True, ""