# backend/cache.py (FIXED)
"""
Was written against a plain TCP `redis` client using REDIS_HOST/REDIS_PORT,
which you never provisioned. You have Upstash REST credentials
(UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN) and `upstash-redis`
already in requirements.txt — this now actually uses them, so caching
stops silently no-op-ing.

Cache key is now based on the TEXT BEING SYNTHESIZED (the AI's response),
not the user's spoken transcript. Keying by transcript was the bug behind
"sometimes speaks of previous output text": with multi-turn memory, the
same/similar user phrase can legitimately deserve a different answer on
turn 5 than on turn 1, but the old key would replay whatever audio was
cached earlier for that transcript.
"""

import os
import hashlib
from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()

redis_client = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL", ""),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
)

CACHE_TTL = 7200


def _cache_key(response_text: str, lang: str, speaker_profile: str) -> str:
    # Keyed on the OUTPUT text + voice, not the user's input transcript.
    raw = f"{response_text.lower().strip()}:{lang}:{speaker_profile}"
    return f"tts_cache:{hashlib.sha256(raw.encode()).hexdigest()}"


def get_cached_audio(response_text: str, lang: str, speaker_profile: str = "aastha"):
    try:
        val = redis_client.get(_cache_key(response_text, lang, speaker_profile))
        if val is None:
            return None
        # upstash-redis REST returns str; audio is stored base64-encoded (see set_cached_audio)
        import base64
        return base64.b64decode(val)
    except Exception as e:
        print(f"[Cache] Upstash read error: {e}")
        return None


def set_cached_audio(response_text: str, lang: str, audio_bytes: bytes, speaker_profile: str = "aastha"):
    try:
        import base64
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        redis_client.set(_cache_key(response_text, lang, speaker_profile), encoded, ex=CACHE_TTL)
    except Exception as e:
        print(f"[Cache] Upstash write error: {e}")