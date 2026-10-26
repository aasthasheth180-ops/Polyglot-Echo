# backend/config.py (FIXED)
"""
Centralized configuration for Polyglot Echo.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

AI_SERVICE_URL = (os.getenv("AI_SERVICE_URL") or
                  "https://aaasthasss-polyglot-echo-ai.hf.space").rstrip("/")
COLAB_URL = AI_SERVICE_URL  # kept for compatibility with existing imports

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATASET_REPO_ID = os.getenv("DATASET_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL", "300"))
SESSION_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL", "60"))

TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "5"))
TTS_INITIAL_BACKOFF = int(os.getenv("TTS_INITIAL_BACKOFF", "2"))
TTS_MAX_BACKOFF = int(os.getenv("TTS_MAX_BACKOFF", "32"))
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "300"))

DEFAULT_REF_AUDIO_PATH = "/tmp/clip_1.wav"
DEFAULT_REF_TEXT = "Hey, how have you been?"

GUEST_REF_AUDIO_PATH = "/tmp/guest_ref.wav"
FALLBACK_GUEST_REF_PATH = "/tmp/guest_ref_persistent.wav"

MIN_AUDIO_BYTES = int(os.getenv("MIN_AUDIO_BYTES", "8000"))

SPEAKER_PROFILES = ["aastha", "guest"]

# FIX: 256 tokens was clipping answers mid-sentence (feeds a fragment into
# TTS -> sounds broken/robotic on top of just reading "short"). Raised to
# give the model room for a complete, natural reply. Tune down if latency
# becomes an issue for your GPU tier.
# FIX: gemini-2.5-flash started returning 404 "no longer available to new
# users" — Google has restricted it for new API keys ahead of its full
# shutdown later this year (confirmed via Google's own deprecation docs,
# July 2026). gemini-3.5-flash-lite is the current direct cost/latency
# equivalent replacement. If you want noticeably smarter answers and don't
# mind higher cost/latency, gemini-3.5-flash is the step up (also GA).
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))


def get_ref_audio_path(speaker_profile: str) -> str:
    if speaker_profile == "guest":
        return GUEST_REF_AUDIO_PATH
    return DEFAULT_REF_AUDIO_PATH


def get_ref_text(speaker_profile: str) -> str:
    # Keep this SHORT (<15 words) — it's the reference transcript for cloning,
    # not the thing being said. Long ref_text degrades cloning quality.
    return "Hey, how have you been lately?"


if __name__ == "__main__":
    print(f"""
    Environment:     {ENVIRONMENT}
    Debug:           {DEBUG}
    AI Service URL:  {AI_SERVICE_URL}
    Session TTL:     {SESSION_TTL_SECONDS}s
    LLM Model:       {LLM_MODEL}
    LLM Max Tokens:  {LLM_MAX_TOKENS}
    """)
