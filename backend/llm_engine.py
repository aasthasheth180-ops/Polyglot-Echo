# backend/llm_engine.py
import os
from google import genai

# Explicit language names mapping dictionary helper
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}

# ── 🎯 Native Memory Class (Replaces LangChain ConversationBufferMemory) ──
class LocalConversationMemory:
    def __init__(self):
        self.history = []

    @property
    def buffer(self) -> str:
        """Formats the history array into a text string matching LangChain's buffer format."""
        if not self.history:
            return ""
        return "\n".join([f"{msg['role']}: {msg['text']}" for msg in self.history])

    def save_context(self, inputs: dict, outputs: dict):
        # inputs typically {"input": "user text"}, outputs typically {"output": "ai response"}
        self.history.append({"role": "Human", "text": inputs.get("input", "")})
        self.history.append({"role": "AI", "text": outputs.get("output", "")})
        
        # Keep a sliding window of the last 10 exchanges to keep prompt tokens bounded
        if len(self.history) > 20:
            self.history = self.history[-20:]


class LLMEngine:
    def __init__(self):
        # Initializes the native Google GenAI client object structure
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.sessions = {}

    def _get_memory(self, session_id: str) -> LocalConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = LocalConversationMemory()
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"[Memory] Cleared context memory footprint for session={session_id[:8]}")

    def generate(self, user_text: str, target_lang: str, session_id: str = "default") -> dict:
        """Standard non-streaming generation for HTTP REST endpoints."""
        import time
        t_start = time.time()
        
        lang_name = LANGUAGE_NAMES.get(target_lang, "English")
        memory = self._get_memory(session_id)
        history_text = memory.buffer if memory.buffer else "No previous conversation."

        prompt = f"""You are Polyglot Echo, a multilingual voice assistant.
CRITICAL RULES:
1. Respond ONLY in {lang_name}
2. Keep response to 2-3 sentences MAXIMUM
3. NEVER repeat what was said before
4. Answer ONLY the current question below
5. Do NOT include the question in your answer

Previous conversation (for context only — do NOT repeat it):
{history_text}

Current question (answer THIS and only this):
{user_text}

Your fresh response in {lang_name}:"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 150
            }
        )
        
        ai_text = response.text.strip() if response.text else ""
        memory.save_context({"input": user_text}, {"output": ai_text})
        
        return {
            "text": ai_text,
            "latency_ms": int((time.time() - t_start) * 1000)
        }

    async def generate_stream(self, user_text: str, target_lang: str, session_id: str = "default"):
        """Asynchronous token-yielding stream iteration framework."""
        lang_name = LANGUAGE_NAMES.get(target_lang, "English")
        memory    = self._get_memory(session_id)
        history_text = memory.buffer if memory.buffer else "No previous conversation."

        prompt = f"""You are Polyglot Echo, a multilingual voice assistant.
CRITICAL RULES:
1. Respond ONLY in {lang_name}
2. Keep response to 2-3 sentences MAXIMUM
3. NEVER repeat what was said before
4. Answer ONLY the current question below
5. Do NOT include the question in your answer

Previous conversation (for context only — do NOT repeat it):
{history_text}

Current question (answer THIS and only this):
{user_text}

Your fresh response in {lang_name}:"""

        full_response = ""
        
        response = self.client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.3,      
                "max_output_tokens": 150  
            }
        )
        
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        memory.save_context({"input": user_text}, {"output": full_response})


# ── CRITICAL INSTANTIATION UNIT EXPORT ──────────────────────────
llm_engine = LLMEngine()