# backend/llm_engine.py (REFACTORED)
"""
LLM Engine with proper session isolation and TTL-based cleanup.
"""

import os
import time
import threading
from typing import Optional, Dict
from google import genai
from config import (
    SESSION_TTL_SECONDS, SESSION_CLEANUP_INTERVAL,
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    SUPPORTED_LANGUAGES
)
from logger import get_logger, set_context, log_function_call

logger = get_logger(__name__)


# ── Language Configuration ────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}

LANGUAGE_RULES = {
    "en": "You MUST respond ONLY in English. Every single word must be English. Do NOT mix languages.",
    "hi": "आप ONLY हिंदी में जवाब दें। हर शब्द हिंदी में होना चाहिए। अंग्रेजी के साथ मिलाएं नहीं।",
    "gu": "તમે ONLY ગુજરાતીમાં જવાબ આપો। દરેક શબ્દ ગુજરાતીમાં હોવો જોઈએ। અંગ્રેજી સાથે મિશ્રણ કરશો નહીં।",
    "es": "Debes responder SOLO en español. Cada palabra debe ser en español. NO mezcles idiomas.",
    "fr": "Vous devez répondre UNIQUEMENT en français. Chaque mot doit être en français. N'mélangez pas les langues.",
    "de": "Du musst ONLY auf Deutsch antworten. Jedes Wort muss auf Deutsch sein. Keine Sprachmischung."
}


# ── Session Memory Class ───────────────────────────────────────
class ConversationMemory:
    """
    Stores conversation history for a single session.
    With TTL-based expiration and bounded history.
    """
    
    def __init__(self, session_id: str, ttl_seconds: int = SESSION_TTL_SECONDS):
        self.session_id = session_id
        self.history = []
        self.ttl_seconds = ttl_seconds
        self.created_at = time.time()
        self.last_accessed = time.time()
    
    @property
    def buffer(self) -> str:
        """Format history as plain text for LLM context."""
        if not self.history:
            return "No previous conversation."
        
        # Only use last 10 exchanges (20 messages) to control token usage
        recent = self.history[-20:]
        return "\n".join([f"{msg['role']}: {msg['text']}" for msg in recent])
    
    def save_context(self, inputs: dict, outputs: dict):
        """Save user input and AI output."""
        self.history.append({"role": "Human", "text": inputs.get("input", "")})
        self.history.append({"role": "AI", "text": outputs.get("output", "")})
        self.last_accessed = time.time()
        
        # Keep sliding window of last 20 messages
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def is_expired(self) -> bool:
        """Check if session has exceeded TTL."""
        elapsed = time.time() - self.last_accessed
        return elapsed > self.ttl_seconds
    
    def clear(self):
        """Explicitly clear history."""
        self.history.clear()
        self.last_accessed = time.time()
    
    def age_seconds(self) -> float:
        """Get age of session in seconds."""
        return time.time() - self.created_at


# ── LLM Engine ─────────────────────────────────────────────────
class LLMEngine:
    """
    Gemini-powered LLM with:
    - Session isolation (per session_id)
    - TTL-based session cleanup
    - Strict language enforcement
    - Token-bounded responses
    """
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("[LLM] GEMINI_API_KEY not set!")
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        self.client = genai.Client(api_key=api_key)
        self.sessions: Dict[str, ConversationMemory] = {}
        self._lock = threading.RLock()
        
        # Start background cleanup thread
        self._start_cleanup_thread()
        
        logger.info("[LLM] Engine initialized with Gemini")
    
    def _get_memory(self, session_id: str) -> ConversationMemory:
        """Get or create memory for a session."""
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationMemory(session_id)
                logger.debug(f"[LLM] Created new session: {session_id}")
            
            memory = self.sessions[session_id]
            memory.last_accessed = time.time()
            return memory
    
    def clear_session(self, session_id: str):
        """Explicitly clear a session."""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"[LLM] Cleared session: {session_id}")
    
    def _cleanup_expired_sessions(self):
        """Remove sessions that have exceeded TTL."""
        with self._lock:
            expired = []
            for session_id, memory in list(self.sessions.items()):
                if memory.is_expired():
                    expired.append(session_id)
                    del self.sessions[session_id]
            
            if expired:
                logger.info(f"[LLM] Cleaned up {len(expired)} expired sessions")
    
    def _start_cleanup_thread(self):
        """Background thread to clean up expired sessions."""
        def cleanup_loop():
            while True:
                time.sleep(SESSION_CLEANUP_INTERVAL)
                try:
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"[LLM] Cleanup thread error: {e}", exc_info=True)
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
        logger.debug("[LLM] Started session cleanup thread")
    
    @log_function_call
    def generate(
        self,
        user_text: str,
        target_lang: str,
        session_id: str = "default"
    ) -> dict:
        """
        Generate LLM response.
        
        Returns:
            {
                "text": "...",
                "latency_ms": 123,
                "tokens_used": 45,
                "language": "en"
            }
        """
        set_context(session_id=session_id, target_lang=target_lang)
        
        try:
            t_start = time.time()
            
            lang_name = LANGUAGE_NAMES.get(target_lang, "English")
            lang_rule = LANGUAGE_RULES.get(target_lang, "")
            memory = self._get_memory(session_id)
            history_text = memory.buffer
            
            # STRICT PROMPT — enforces language purity and conciseness
            prompt = f"""You are Polyglot Echo, a multilingual voice assistant.

LANGUAGE ENFORCEMENT:
{lang_rule}

CRITICAL RULES:
1. Respond ONLY in {lang_name} — absolutely NO English, NO code, NO special characters
2. Response must be 1-2 sentences MAXIMUM (very short and natural)
3. NEVER repeat previous messages from the conversation history
4. Answer ONLY the current question — ignore the history for generating new content
5. Do NOT include the user's question in your response
6. Speak naturally and conversationally

Previous conversation (context only):
{history_text}

Current question from user:
{user_text}

Your fresh {lang_name} response (short, natural, 1-2 sentences):"""
            
            logger.debug(f"[LLM] Generating response for: '{user_text[:50]}'")
            
            response = self.client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={
                    "temperature": LLM_TEMPERATURE,
                    "max_output_tokens": LLM_MAX_TOKENS
                }
            )
            
            ai_text = response.text.strip() if response.text else ""
            
            # Validate response is not empty
            if not ai_text:
                logger.warning(f"[LLM] Empty response from Gemini for session {session_id}")
                ai_text = "Sorry, I couldn't generate a response."
            
            # Save to memory
            memory.save_context(
                {"input": user_text},
                {"output": ai_text}
            )
            
            latency_ms = int((time.time() - t_start) * 1000)
            
            logger.info(f"[LLM] Generated response in {latency_ms}ms: '{ai_text[:40]}'")
            
            return {
                "text": ai_text,
                "latency_ms": latency_ms,
                "language": target_lang,
                "tokens_used": len(ai_text.split())  # Rough estimate
            }
        
        except Exception as e:
            logger.error(f"[LLM] Generation failed: {e}", exc_info=True)
            raise
    
    async def generate_stream(
        self,
        user_text: str,
        target_lang: str,
        session_id: str = "default"
    ):
        """
        Async streaming generation (yields chunks).
        """
        set_context(session_id=session_id, target_lang=target_lang)
        
        try:
            lang_name = LANGUAGE_NAMES.get(target_lang, "English")
            lang_rule = LANGUAGE_RULES.get(target_lang, "")
            memory = self._get_memory(session_id)
            history_text = memory.buffer
            
            prompt = f"""You are Polyglot Echo, a multilingual voice assistant.

LANGUAGE ENFORCEMENT:
{lang_rule}

CRITICAL RULES:
1. Respond ONLY in {lang_name} — absolutely NO English, NO code, NO special characters
2. Response must be 1-2 sentences MAXIMUM (very short and natural)
3. NEVER repeat previous messages from the conversation history
4. Answer ONLY the current question — ignore the history for generating new content
5. Do NOT include the user's question in your response
6. Speak naturally and conversationally

Previous conversation (context only):
{history_text}

Current question from user:
{user_text}

Your fresh {lang_name} response (short, natural, 1-2 sentences):"""
            
            full_response = ""
            
            response = self.client.models.generate_content_stream(
                model=LLM_MODEL,
                contents=prompt,
                config={
                    "temperature": LLM_TEMPERATURE,
                    "max_output_tokens": LLM_MAX_TOKENS
                }
            )
            
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Save to memory after streaming complete
            memory.save_context(
                {"input": user_text},
                {"output": full_response}
            )
            
            logger.debug(f"[LLM] Streamed response complete: '{full_response[:40]}'")
        
        except Exception as e:
            logger.error(f"[LLM] Stream generation failed: {e}", exc_info=True)
            raise
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get debugging info about a session."""
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


# ── Global Engine Instance ─────────────────────────────────────
llm_engine = LLMEngine()