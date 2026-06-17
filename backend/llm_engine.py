# backend/llm_engine.py
import os
import time
from dotenv import load_dotenv
load_dotenv()

from google import genai

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish",
    "fr": "French",  "de": "German", "zh": "Chinese",
    "ja": "Japanese", "pt": "Portuguese", "ru": "Russian",
    "ar": "Arabic",  "gu": "Gujarati"
}

SYSTEM_TEMPLATE = """You are Polyglot Echo, a high-performance multilingual voice assistant.

CRITICAL: Respond ONLY in {target_language}. Never switch languages under any circumstances.
Keep responses short (exactly 2-3 sentences) — this is a voice interface.
Be natural, friendly, and conversational. Never use bullet points, lists, or markdown stars.

Conversation History so far:
{history}

User said: {input}
Your response in {target_language}:"""

class LLMEngine:
    def __init__(self):
        # Explicitly targets the new official Google GenAI standard
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # Per-session active history vault: {session_id: [{"user": "...", "assistant": "..."}]}
        self._memories = {}
        print("[✓] LLM Engine: Gemini + Native Sliding-Window Memory Online")

    def _get_history_buffer(self, session_id: str, k: int = 5) -> str:
        """Keeps only the last k turns of the conversation to prevent context bloat."""
        if session_id not in self._memories:
            self._memories[session_id] = []
        
        # Slice to get only the last 'k' interactions
        recent_turns = self._memories[session_id][-k:]
        
        if not recent_turns:
            return "No previous context."
            
        formatted_history = []
        for turn in recent_turns:
            formatted_history.append(f"User: {turn['user']}")
            formatted_history.append(f"Assistant: {turn['assistant']}")
            
        return "\n".join(formatted_history)

    def generate(self, user_text: str, target_lang: str, session_id: str = "default") -> dict:
        start = time.time()
        lang_name = LANGUAGE_NAMES.get(target_lang, "English")
        
        # Pull the history formatted strings directly from our native sliding matrix
        history_buffer = self._get_history_buffer(session_id, k=5)
        
        prompt = SYSTEM_TEMPLATE.format(
            target_language=lang_name,
            history=history_buffer,
            input=user_text
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response_text = response.text.strip()

        # Append current interaction turn straight to internal memory list state
        self._memories[session_id].append({
            "user": user_text,
            "assistant": response_text
        })

        latency_ms = int((time.time() - start) * 1000)
        print(f"[LLM Memory Node] ({target_lang}): '{response_text[:50]}...' | {latency_ms}ms")
        return {"text": response_text, "latency_ms": latency_ms}

    def clear_session(self, session_id: str):
        if session_id in self._memories:
            del self._memories[session_id]
            print(f"🧹 [LLM Memory Node] Purged context history window for session: {session_id[:8]}")

llm_engine = LLMEngine()