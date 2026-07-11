# backend/voice_sessions.py
"""
Tracks the "record my voice -> clone a preview -> I approve it -> now chat"
flow per session_id. This is what makes guest cloning behave like an actual
onboarding step instead of silently using the wrong (or no) reference.

Nothing here is persisted to a database — it's in-memory and TTL'd, same
pattern as ConversationMemory in llm_engine.py. If the process restarts,
guests re-enroll (their raw voice sample itself is still safe in the HF
dataset — this store just tracks the *chat session's* approval state and
which exact reference transcript belongs to it).
"""

import time
import threading
from typing import Dict, Optional

ENROLLMENT_TTL_SECONDS = 3600 * 6  # generous — a recorded voice shouldn't expire mid-session


class VoiceEnrollment:
    def __init__(self, ref_text: str):
        self.ref_text = ref_text      # accurate transcript of THIS person's reference clip
        self.approved = False         # user has heard the preview clone and confirmed it
        self.enrolled_at = time.time()
        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > ENROLLMENT_TTL_SECONDS


class VoiceSessionStore:
    def __init__(self):
        self._store: Dict[str, VoiceEnrollment] = {}
        self._lock = threading.RLock()

    def enroll(self, session_id: str, ref_text: str) -> None:
        with self._lock:
            self._store[session_id] = VoiceEnrollment(ref_text)

    def approve(self, session_id: str) -> bool:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None or entry.is_expired():
                return False
            entry.approved = True
            entry.last_accessed = time.time()
            return True

    def get(self, session_id: str) -> Optional[VoiceEnrollment]:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None or entry.is_expired():
                return None
            entry.last_accessed = time.time()
            return entry

    def is_ready_to_chat(self, session_id: str) -> bool:
        """True only once this session recorded a sample AND approved the preview."""
        entry = self.get(session_id)
        return entry is not None and entry.approved

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)


voice_sessions = VoiceSessionStore()