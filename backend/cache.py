# backend/cache.py
import os
import hashlib
import redis
from dotenv import load_dotenv
load_dotenv()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)
CACHE_TTL = 7200

def _cache_key(transcript: str, lang: str) -> str:
    raw = f"{transcript.lower().strip()}:{lang}"
    return f"tts_cache:{hashlib.sha256(raw.encode()).hexdigest()}"

def get_cached_audio(transcript: str, lang: str):
    try:
        return redis_client.get(_cache_key(transcript, lang))
    except Exception as e:
        print(f"[Cache] Redis error: {e}")
        return None

def set_cached_audio(transcript: str, lang: str, audio_bytes: bytes):
    try:
        redis_client.set(_cache_key(transcript, lang), audio_bytes, ex=CACHE_TTL)
    except Exception as e:
        print(f"[Cache] Redis write error: {e}")