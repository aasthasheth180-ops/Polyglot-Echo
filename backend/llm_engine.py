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

# Language-specific instructions to FORCE correct output
LANGUAGE_RULES = {
    "en": "You MUST respond ONLY in English. Every single word must be English. Do NOT mix languages.",
    "hi": "आप ONLY हिंदी में जवाब दें। हर शब्द हिंदी में होना चाहिए। अंग्रेजी के साथ मिलाएं नहीं।",
    "gu": "તમે ONLY ગુજરાતીમાં જવાબ આપો। દરેક શબ્દ ગુજરાતીમાં હોવો જોઈએ। અંગ્રેજી સાથે મિશ્રण કરશો નહીં।",
    "es": "Debes responder SOLO en español. Cada palabra debe ser en español. NO mezcles idiomas.",
    "fr": "Vous devez répondre UNIQUEMENT en français. Chaque mot doit être en français. N'mélangez pas les langues.",
    "de": "Du musst ONLY auf Deutsch antworten. Jedes Wort muss auf Deutsch sein. Keine Sprachmischung."
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
        lang_rule = LANGUAGE_RULES.get(target_lang, "")
        memory = self._get_memory(session_id)
        history_text = memory.buffer if memory.buffer else "No previous conversation."

        # MUCH STRICTER PROMPT - emphasis on language purity and brevity
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

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.2,  # Lower temp = more focused on language rules
                "max_output_tokens": 100  # Shorter max token limit
            }
        )
        
        ai_text = response.text.strip() if response.text else ""
        
        # Post-process: clean any accidental English or special chars
        # (Optional: you could add language detection here to verify)
        
        memory.save_context({"input": user_text}, {"output": ai_text})
        
        return {
            "text": ai_text,
            "latency_ms": int((time.time() - t_start) * 1000)
        }

    async def generate_stream(self, user_text: str, target_lang: str, session_id: str = "default"):
        """Asynchronous token-yielding stream iteration framework."""
        lang_name = LANGUAGE_NAMES.get(target_lang, "English")
        lang_rule = LANGUAGE_RULES.get(target_lang, "")
        memory    = self._get_memory(session_id)
        history_text = memory.buffer if memory.buffer else "No previous conversation."

        # SAME STRICT PROMPT as generate()
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
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.2,  # Lower temp = more consistent language
                "max_output_tokens": 100  # Shorter max
            }
        )
        
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        memory.save_context({"input": user_text}, {"output": full_response})


# ── CRITICAL INSTANTIATION UNIT EXPORT ──────────────────────────
llm_engine = LLMEngine()