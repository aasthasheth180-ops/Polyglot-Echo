# backend/llm_engine.py (REFACTORED / FIXED)
"""
LLM Engine with proper session isolation and TTL-based cleanup.

FIX #1: This file imports `from google import genai` and calls
`genai.Client(...)` — that's the google-genai package's API. The old
requirements.txt listed google-generativeai (a DIFFERENT SDK: genai.configure()
/ genai.GenerativeModel(), no .Client). If that old package were installed,
LLMEngine() (instantiated at import time at the bottom of this file) would
raise AttributeError immediately, taking the whole app's import chain down —
this is a likely cause of intermittent startup failures / 500s. Make sure
requirements.txt has `google-genai`.

FIX #2: The prompt used to force "1-2 sentences MAXIMUM" — that's why answers
felt clipped and curt. Loosened below to a natural conversational length
(roughly 1-4 sentences, model's judgment) while raising LLM_MAX_TOKENS so
responses aren't hard-truncated mid-sentence (a truncated sentence handed to
TTS is also a likely contributor to "robotic"/garbled-sounding audio, since
F5-TTS is fed a fragment).

gemini-2.5-flash itself is a solid, fast default for this — no model swap
was needed to fix the length issue. If you want a smarter model at similar
latency, `gemini-3-flash-lite` (GA as of 2026) is the newer step up; for
noticeably better quality at higher cost, `gemini-3.5-flash`.
"""

import os
import time
import threading
import httpx
from typing import Optional, Dict
from google import genai
from groq import Groq
from config import (
    SESSION_TTL_SECONDS, SESSION_CLEANUP_INTERVAL,
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
)
from logger import get_logger, set_context, log_function_call
from google.genai import errors

logger = get_logger(__name__)


LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "gu": "Gujarati",
    "es": "Spanish", "fr": "French", "de": "German"
}

LANGUAGE_RULES = {
    "en": "You MUST respond ONLY in English. Every single word must be English. Do NOT mix languages.",
    "hi": "आप ONLY हिंदी में जवाब दें। हर शब्द हिंदी में होना चाहिए। अंग्रेजी के साथ मिलाएं नहीं।",
    "gu": "તમે ONLY ગુજરાતીમાં જવાબ આપો। દરેક શબ્દ ગુજરાતીમાં હોવો જોઈએ। અંગ્રેજી સાથે મિશ્રણ કરશો નહીં।",
    "es": "Debes responder SOLO en español. Cada palabra debe ser en español. NO mezcles idiomas.",
    "fr": "Vous devez répondre UNIQUEMENT en français. Chaque mot doit être en français. N'mélangez pas les langues.",
    "de": "Du musst ONLY auf Deutsch antworten. Jedes Wort muss auf Deutsch sein. Keine Sprachmischung."
}


class ConversationMemory:
    def __init__(self, session_id: str, ttl_seconds: int = SESSION_TTL_SECONDS):
        self.session_id = session_id
        self.history = []
        self.ttl_seconds = ttl_seconds
        self.created_at = time.time()
        self.last_accessed = time.time()

    @property
    def buffer(self) -> str:
        if not self.history:
            return "No previous conversation."
        recent = self.history[-20:]
        return "\n".join([f"{msg['role']}: {msg['text']}" for msg in recent])

    def save_context(self, inputs: dict, outputs: dict):
        self.history.append({"role": "Human", "text": inputs.get("input", "")})
        self.history.append({"role": "AI", "text": outputs.get("output", "")})
        self.last_accessed = time.time()
        if len(self.history) > 20:
            self.history = self.history[-20:]

    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > self.ttl_seconds

    def clear(self):
        self.history.clear()
        self.last_accessed = time.time()

    def age_seconds(self) -> float:
        return time.time() - self.created_at


def _build_prompt(user_text: str, target_lang: str, history_text: str) -> str:
    lang_name = LANGUAGE_NAMES.get(target_lang, "English")
    lang_rule = LANGUAGE_RULES.get(target_lang, "")

    # FIX: no longer forces "1-2 sentences MAXIMUM" — that produced curt,
    # unhelpful answers. Now asks for a natural, complete, conversational
    # reply instead of an arbitrary sentence cap.
    return f"""You are Polyglot Echo, a helpful, friendly voice assistant.

LANGUAGE ENFORCEMENT:
{lang_rule}

RULES:
1. Respond ONLY in {lang_name} — no English mixed in unless {lang_name} is English.
2. Give a complete, natural, conversational answer — as long as it needs to be
   to actually help, but stay concise and avoid rambling (a good voice-assistant
   answer is usually 1-4 sentences, but use your judgment for the question asked).
3. Never repeat earlier messages verbatim from the conversation history.
4. Answer only the CURRENT question; use history only for context.
5. Do not restate the user's question back to them.
6. End on a complete sentence — never trail off mid-thought (this response
   is spoken aloud, so a cut-off sentence sounds broken).

Previous conversation (context only):
{history_text}

Current question from user:
{user_text}

Your fresh {lang_name} response:"""

class EmptyLLMResponseError(Exception):
    """Raised when an LLM returns an empty response."""

class LLMEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("[LLM] GEMINI_API_KEY not set!")
            raise ValueError("GEMINI_API_KEY environment variable required")

        self.client = genai.Client(
            api_key=api_key,
            http_options=genai.types.HttpOptions(
                timeout=10000,
                retry_options=genai.types.HttpRetryOptions(
                    attempts=1
                    ),
                ),
            )
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = (
            Groq(api_key=groq_api_key, timeout=15.0, max_retries=0)
            if groq_api_key
            else None
        )
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.sessions: Dict[str, ConversationMemory] = {}
        self._lock = threading.RLock()
        self._start_cleanup_thread()
        logger.info("[LLM] Engine initialized with Gemini")

    def _get_memory(self, session_id: str) -> ConversationMemory:
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationMemory(session_id)
            memory = self.sessions[session_id]
            memory.last_accessed = time.time()
            return memory

    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"[LLM] Cleared session: {session_id}")

    def _cleanup_expired_sessions(self):
        with self._lock:
            expired = [sid for sid, m in list(self.sessions.items()) if m.is_expired()]
            for sid in expired:
                del self.sessions[sid]
            if expired:
                logger.info(f"[LLM] Cleaned up {len(expired)} expired sessions")

    def _start_cleanup_thread(self):
        def cleanup_loop():
            while True:
                time.sleep(SESSION_CLEANUP_INTERVAL)
                try:
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"[LLM] Cleanup thread error: {e}", exc_info=True)
        threading.Thread(target=cleanup_loop, daemon=True).start()

    def _generate_with_groq(self, prompt: str) -> str:
        """Generate a response using Groq as the fallback provider."""

        if self.groq_client is None:
            raise RuntimeError("Groq fallback is not configured")

        logger.info("[LLM] Switching to Groq fallback")

        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_TOKENS,
        )

        ai_text = response.choices[0].message.content

        if not ai_text or not ai_text.strip():
            raise RuntimeError("Groq returned an empty response")

        return ai_text.strip()

    @log_function_call
    def generate(self, user_text: str, target_lang: str, session_id: str = "default") -> dict:
        set_context(session_id=session_id, target_lang=target_lang)

        try:
            t_start = time.time()

            memory = self._get_memory(session_id)
            prompt = _build_prompt(user_text, target_lang, memory.buffer)

            ai_text = ""
            provider = "gemini"

            try:
                max_attempts = 2

                for attempt in range(max_attempts):
                    try:
                        response = self.client.models.generate_content(
                            model=LLM_MODEL,
                            contents=prompt,
                            config={
                                "temperature": LLM_TEMPERATURE,
                                "max_output_tokens": LLM_MAX_TOKENS,
                            },
                        )

                        ai_text = (
                            response.text.strip()
                            if response.text
                            else ""
                        )

                        if not ai_text:
                            raise EmptyLLMResponseError("Gemini returned an empty response")
                        break

                    except errors.ServerError as e:
                        status_code = getattr(e, "code", None)

                        if status_code not in (500, 502, 503, 504):
                            raise

                        if attempt == max_attempts - 1:
                            raise

                        logger.warning(
                            f"[LLM] Gemini HTTP {status_code}; retrying once"
                        )
                        time.sleep(1)

            except ( errors.APIError,
                    TimeoutError,
                    ConnectionError,
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    EmptyLLMResponseError,
                ) as e:
                if isinstance(e, errors.APIError):
                    status_code = getattr(e, "code", None)

                    if status_code not in (429, 500, 502, 503, 504):
                        raise
                logger.warning(
                    f"[LLM] Gemini unavailable; switching to Groq: {type(e).__name__}"
                )

                if self.groq_client is None:
                    raise RuntimeError(
                        "Gemini failed and Groq fallback is not configured"
                    ) from e

                ai_text = self._generate_with_groq(prompt)
                provider = "groq"

            memory.save_context(
                {"input": user_text},
                {"output": ai_text},
            )

            latency_ms = int((time.time() - t_start) * 1000)

            logger.info(
                f"[LLM] Provider={provider}; latency={latency_ms}ms"
            )

            return {
                "text": ai_text,
                "latency_ms": latency_ms,
                "language": target_lang,
                "tokens_used": len(ai_text.split()),
                "provider": provider,
            }

        except Exception as e:
            logger.error(
                f"[LLM] Generation failed: {e}",
                exc_info=True,
            )
            raise

    async def generate_stream(self, user_text: str, target_lang: str, session_id: str = "default"):
        set_context(session_id=session_id, target_lang=target_lang)
        try:
            memory = self._get_memory(session_id)
            prompt = _build_prompt(user_text, target_lang, memory.buffer)
            full_response = ""

            response = self.client.models.generate_content_stream(
                model=LLM_MODEL,
                contents=prompt,
                config={
                    "temperature": LLM_TEMPERATURE,
                    "max_output_tokens": LLM_MAX_TOKENS,
                }
            )

            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            memory.save_context({"input": user_text}, {"output": full_response})
            logger.debug(f"[LLM] Streamed response complete: '{full_response[:40]}'")
        except Exception as e:
            logger.error(f"[LLM] Stream generation failed: {e}", exc_info=True)
            raise

    def get_session_info(self, session_id: str) -> Dict:
        with self._lock:
            if session_id not in self.sessions:
                return {"status": "not_found"}
            memory = self.sessions[session_id]
            return {
                "session_id": session_id,
                "history_length": len(memory.history),
                "age_seconds": memory.age_seconds(),
                "last_accessed": memory.last_accessed,
                "is_expired": memory.is_expired(),
                "preview": memory.buffer[:200]
            }


llm_engine = LLMEngine()