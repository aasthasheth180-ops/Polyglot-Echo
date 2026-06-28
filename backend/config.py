# backend/config.py
"""
Centralized configuration for Polyglot Echo.
All constants, paths, and settings in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Environment ────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# ── AI Services URLs ───────────────────────────────────────────
# Safely handle environment variable name variations
COLAB_URL = (os.getenv("COLAB_AI_URL") or 
             os.getenv("COLAB_URL") or 
             "https://aaasthasss-polyglot-echo-ai.hf.space").rstrip("/")

# ── API Keys ───────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Session Management ─────────────────────────────────────────
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL", "300"))  # 5 minutes
SESSION_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL", "60"))  # Check every 60s

# ── TTS Configuration ─────────────────────────────────────────
TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "5"))
TTS_INITIAL_BACKOFF = int(os.getenv("TTS_INITIAL_BACKOFF", "2"))  # seconds
TTS_MAX_BACKOFF = int(os.getenv("TTS_MAX_BACKOFF", "32"))  # seconds
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "300"))  # 5 minutes

# ── Audio Validation ───────────────────────────────────────────
MIN_AUDIO_BYTES = int(os.getenv("MIN_AUDIO_BYTES", "1000"))
WAV_HEADER_SIZE = 44  # Standard WAV header

# ── Reference Audio ────────────────────────────────────────────
# Default voice sample (should exist in container)
DEFAULT_REF_AUDIO_PATH = "/code/clip_1.wav"
DEFAULT_REF_TEXT = "Hey, how have you been?"

# Guest voice (persistent location)
GUEST_REF_AUDIO_PATH = "/data/guest_ref.wav"

# Fallback if /data is not available
FALLBACK_GUEST_REF_PATH = "/tmp/guest_ref_persistent.wav"

# ── LLM Configuration ──────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))  # Increased from 100
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # seconds

# ── Logging ────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_STRUCTURED_LOGGING = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"

# ── Circuit Breaker ────────────────────────────────────────────
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
CIRCUIT_BREAKER_WINDOW = int(os.getenv("CIRCUIT_BREAKER_WINDOW", "60"))  # seconds

# ── Whisper Configuration ──────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # Override in config

# ── F5-TTS Configuration ──────────────────────────────────────
F5TTS_CFG_STRENGTH = float(os.getenv("F5TTS_CFG_STRENGTH", "1.8"))
F5TTS_NFE_STEP = int(os.getenv("F5TTS_NFE_STEP", "24"))
F5TTS_SPEED = float(os.getenv("F5TTS_SPEED", "1.15"))

# ── Language Configuration ─────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "default": True},
    "hi": {"name": "Hindi"},
    "gu": {"name": "Gujarati"},
    "es": {"name": "Spanish"},
    "fr": {"name": "French"},
    "de": {"name": "German"}
}

# ── Speaker Profiles ───────────────────────────────────────────
SPEAKER_PROFILES = {
    "aastha": {
        "name": "Aastha (Default)",
        "ref_audio": DEFAULT_REF_AUDIO_PATH,
        "ref_text": DEFAULT_REF_TEXT,
        "description": "Default female voice"
    },
    "guest": {
        "name": "Guest (Cloned)",
        "ref_audio": GUEST_REF_AUDIO_PATH,
        "ref_text": DEFAULT_REF_TEXT,  # Use same ref_text as default
        "description": "User-uploaded voice clone",
        "fallback": "aastha"  # If file not found, fall back to aastha
    }
}

# ── Rate Limiting ──────────────────────────────────────────────
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "30"))

# ── Validation & Defaults ──────────────────────────────────────
def get_ref_audio_path(speaker_profile: str) -> str:
    """Get reference audio path for a speaker profile, with fallback."""
    profile = SPEAKER_PROFILES.get(speaker_profile, SPEAKER_PROFILES["aastha"])
    ref_path = profile.get("ref_audio", DEFAULT_REF_AUDIO_PATH)
    
    # Check if file exists; fallback if not
    import os as os_module
    if not os_module.path.exists(ref_path):
        fallback = profile.get("fallback", "aastha")
        if fallback and fallback != speaker_profile:
            return get_ref_audio_path(fallback)
    
    return ref_path


def get_ref_text(speaker_profile: str) -> str:
    """Get reference text for a speaker profile."""
    profile = SPEAKER_PROFILES.get(speaker_profile, SPEAKER_PROFILES["aastha"])
    return profile.get("ref_text", DEFAULT_REF_TEXT)


# ── Log Configuration Summary ──────────────────────────────────
if __name__ == "__main__":
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Polyglot Echo Configuration Summary                      ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Environment:            {ENVIRONMENT:30} ║
    ║  Debug Mode:             {str(DEBUG):30} ║
    ║  AI Services URL:        {COLAB_URL[:35]:35} ║
    ║  Session TTL:            {SESSION_TTL_SECONDS} seconds{' ' * 24}║
    ║  LLM Model:              {LLM_MODEL:30} ║
    ║  LLM Max Tokens:         {LLM_MAX_TOKENS:30} ║
    ║  TTS Max Retries:        {TTS_MAX_RETRIES:30} ║
    ║  Whisper Model Size:     {WHISPER_MODEL_SIZE:30} ║
    ║  Supported Languages:    {', '.join(SUPPORTED_LANGUAGES.keys()):30} ║
    ║  Structured Logging:     {str(ENABLE_STRUCTURED_LOGGING):30} ║
    ║  Circuit Breaker:        {CIRCUIT_BREAKER_THRESHOLD} failures/{CIRCUIT_BREAKER_WINDOW}s{' ' * 20}║
    ╚═══════════════════════════════════════════════════════════╝
    """)