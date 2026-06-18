# frontend/app.py — complete rebuild
import streamlit as st
import requests
import os
import uuid
import base64
import time
from audio_recorder_streamlit import audio_recorder

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Polyglot Echo",
    page_icon="🎙️",
    layout="centered"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

LANGUAGES = {
    "English":    "en",
    "Hindi":      "hi",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Portuguese": "pt",
    "Russian":    "ru",
    "Arabic":     "ar",
    "Chinese":    "zh",
    "Japanese":   "ja",
    "Korean":     "ko",
}

# ── Session state ─────────────────────────────────────────────
if "session_id"  not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages"    not in st.session_state:
    st.session_state.messages = []
if "processing"  not in st.session_state:
    st.session_state.processing = False

# ── Custom CSS — dark chat UI ─────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit header */
    #MainMenu, header, footer {visibility: hidden;}

    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #E2E8F0;
    }
    .sub-title {
        text-align: center;
        font-size: 0.85rem;
        color: #718096;
        margin-bottom: 1.5rem;
    }
    .chat-bubble-user {
        background: #2D3748;
        color: #E2E8F0;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.4rem 0 0.4rem auto;
        max-width: 75%;
        font-size: 0.9rem;
        width: fit-content;
        margin-left: auto;
    }
    .chat-bubble-ai {
        background: #1A365D;
        color: #BEE3F8;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.4rem auto 0.4rem 0;
        max-width: 75%;
        font-size: 0.9rem;
        width: fit-content;
    }
    .lang-tag {
        font-size: 0.7rem;
        color: #718096;
        margin-bottom: 2px;
    }
    .metric-row {
        display: flex;
        gap: 12px;
        justify-content: center;
        margin-top: 0.5rem;
    }
    .metric-box {
        background: #2D3748;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #A0AEC0;
    }
    .status-dot-green {
        display: inline-block;
        width: 8px; height: 8px;
        background: #48BB78;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot-red {
        display: inline-block;
        width: 8px; height: 8px;
        background: #FC8181;
        border-radius: 50%;
        margin-right: 6px;
    }
    .divider {color: #2D3748; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎙️ Polyglot Echo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Speak in any language — hear back in your cloned voice</div>',
            unsafe_allow_html=True)

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    speaker_label = st.radio(
        "Speaker Profile",
        ["🧑 Primary (Aastha — Tuned)", "👤 Guest (Auto-Calibrate)"],
        index=0
    )
    speaker_profile = "aastha" if "Aastha" in speaker_label else "guest"

    target_lang_label = st.selectbox(
        "Respond in",
        list(LANGUAGES.keys()),
        index=0
    )
    target_lang = LANGUAGES[target_lang_label]

    st.markdown("---")

    # Guest voice upload — only shown if guest selected
    if speaker_profile == "guest":
        st.markdown("**Upload guest voice reference (8-15s WAV):**")
        guest_ref = st.file_uploader("Guest reference", type=["wav"], key="guest_ref")
        if guest_ref and st.button("Upload Reference"):
            resp = requests.post(
                f"{BACKEND_URL}/api/upload-guest-reference",
                files={"audio_file": ("ref.wav", guest_ref.read(), "audio/wav")}
            )
            if resp.status_code == 200:
                st.success("Reference uploaded to AI server")
            else:
                st.error("Upload failed")

    st.markdown("---")

    # Connection status
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        colab_ok = health.get("colab_ai") == "connected"
        st.markdown(
            f'<span class="status-dot-{"green" if colab_ok else "red"}"></span>'
            f'Colab AI: {"Connected" if colab_ok else "Disconnected"}',
            unsafe_allow_html=True
        )
    except:
        st.markdown('<span class="status-dot-red"></span>Backend offline',
                    unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear conversation"):
        requests.delete(
            f"{BACKEND_URL}/api/session/{st.session_state.session_id}",
            timeout=5
        )
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="lang-tag">You ({msg.get("lang","?")})</div>'
                f'<div class="chat-bubble-user">🎤 {msg["text"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="lang-tag">Echo ({msg.get("lang","?")})</div>'
                f'<div class="chat-bubble-ai">🔊 {msg["text"]}</div>',
                unsafe_allow_html=True
            )
            # Play audio inline
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/wav")

            # Latency metrics
            if "latency" in msg:
                lat = msg["latency"]
                st.markdown(
                    f'<div class="metric-row">'
                    f'<div class="metric-box">🎙 {lat.get("whisper","?")}ms</div>'
                    f'<div class="metric-box">🧠 {lat.get("llm","?")}ms</div>'
                    f'<div class="metric-box">🔊 {lat.get("tts","?")}ms</div>'
                    f'<div class="metric-box">⚡ {lat.get("total","?")}ms total</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ── Recording area ────────────────────────────────────────────
st.markdown("---")
st.markdown("**🎤 Hold to record — release to send:**")

audio_bytes = audio_recorder(
    text="Hold & speak",
    recording_color="#E53E3E",
    neutral_color="#4A5568",
    icon_name="microphone",
    icon_size="2x",
    pause_threshold=2.0,    # stop after 2s silence
    sample_rate=16000       # 16kHz — exactly what Whisper needs, no resampling
)

# ── Process when audio received ───────────────────────────────
if audio_bytes and len(audio_bytes) > 5000:  # ignore tiny noise captures
    if not st.session_state.processing:
        st.session_state.processing = True

        with st.spinner("Processing..."):
            try:
                start = time.time()
                response = requests.post(
                    f"{BACKEND_URL}/api/process-voice",
                    files={"audio_file": ("recording.wav", audio_bytes, "audio/wav")},
                    data={
                        "target_lang":      target_lang,
                        "speaker_profile":  speaker_profile,
                    },
                    headers={"X-Session-Id": st.session_state.session_id},
                    timeout=300
                )

                if response.status_code == 200:
                    transcript   = response.headers.get("X-Transcript", "")
                    ai_response  = response.headers.get("X-Response-Text", "")
                    detected     = response.headers.get("X-Detected-Lang", "?")
                    total_ms     = response.headers.get("X-Latency-Total-MS", "0")

                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "text": transcript if transcript else "(audio recorded)",
                        "lang": detected
                    })

                    # Add AI message with audio
                    st.session_state.messages.append({
                        "role": "ai",
                        "text": ai_response,
                        "lang": target_lang_label,
                        "audio": response.content,
                        "latency": {
                            "whisper": response.headers.get("X-Whisper-MS", "?"),
                            "llm":     response.headers.get("X-LLM-MS", "?"),
                            "tts":     response.headers.get("X-TTS-MS", "?"),
                            "total":   total_ms
                        }
                    })

                else:
                    st.error(f"Server error: {response.status_code}")
                    try:
                        st.error(response.json())
                    except:
                        st.error(response.text[:200])

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend — is uvicorn running on port 8000?")
            except requests.exceptions.Timeout:
                st.error("Request timed out — Colab may be slow or disconnected")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

        st.session_state.processing = False
        st.rerun()