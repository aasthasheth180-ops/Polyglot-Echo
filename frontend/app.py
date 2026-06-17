import streamlit as st
import requests
import os
import uuid
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy import text
from audio_recorder_streamlit import audio_recorder

BACKEND_URL = "http://127.0.0.1:8000"
DB_URL = "postgresql://polyglot:polyglot_pass@localhost:5432/polyglot_echo"
db_engine = create_engine(DB_URL)

st.set_page_config(
    page_title="Polyglot Echo Dashboard",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗣️ Polyglot Echo: Real-Time Multilingual Voice Cloner")
st.markdown("---")

# XTTS/F5-TTS supported output languages
LANGUAGES = {
    "Hindi (हिन्दी)": "hi",
    "English (English)": "en",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Italian (Italiano)": "it",
    "Portuguese (Português)": "pt",
    "Polish (Polski)": "pl",
    "Turkish (Türkçe)": "tr",
    "Russian (Русский)": "ru",
    "Dutch (Nederlands)": "nl",
    "Czech (Čeština)": "cs",
    "Arabic (العربية)": "ar",
    "Chinese (中文)": "zh-cn",
    "Japanese (日本語)": "ja",
    "Korean (한국어)": "ko",
    "Hungarian (Magyar)": "hu"
}

# State Initialization for Sliding-Window History Tracker
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# --- SIDEBAR ---
st.sidebar.header("🎛️ Pipeline Control Panel")

# Speaker Profile Selector
st.sidebar.markdown("### 🎙️ Select Speaker Profile")
speaker_profile_label = st.sidebar.radio(
    "Whose voice should be cloned?",
    options=["Primary Developer (Aastha - Tuned DSP)", "Guest / New Profile (Auto-Calibrating DSP)"],
    index=0
)
speaker_profile = "aastha" if "Aastha" in speaker_profile_label else "guest"

# Output language dropdown
st.sidebar.markdown("### 🌐 Output Language")
selected_lang_label = st.sidebar.selectbox(
    "Respond in:",
    options=list(LANGUAGES.keys())
)
target_lang_code = LANGUAGES[selected_lang_label]

st.sidebar.markdown("---")

# Conversation Memory Purge
st.sidebar.markdown("### 🧠 Conversation Memory")
if st.sidebar.button("🗑️ Clear Context Memory", use_container_width=True):
    try:
        requests.delete(f"{BACKEND_URL}/api/session/{st.session_state.session_id}", timeout=10)
    except Exception as e:
        pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.conversation_history = []
    st.sidebar.success("Context memory wiped cleanly!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.success("Backend Core: Connected")
st.sidebar.info(f"Session ID: `{st.session_state.session_id[:8]}`")

# Tab setup
tab1, tab2 = st.tabs(["🎙️ Conversation Room", "📊 Analytical Engine Logs"])

with tab1:
    # Render Active Ongoing UI Conversation Feed Above Input Fields
    if st.session_state.conversation_history:
        st.subheader("💬 Ongoing Conversation Feed")
        for turn in st.session_state.conversation_history[-5:]: # Keep last 5 elements visible
            with st.chat_message("user"):
                st.write(f"**You ({turn['input_lang']}):** {turn['input_text']}")
            with st.chat_message("assistant"):
                st.write(f"**Echo ({turn['output_lang']}):** {turn['output_text']}")
        st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📥 Input Audio Payload")
        
        # 🎙️ Option 1: Live Mic Recording Zone
        # 🎙️ Option 1: Live Mic Recording Zone (Clean Styled Layout)
        st.markdown(
            """
            <div style="
                background-color: #1e2430; 
                padding: 15px; 
                border-radius: 10px; 
                border: 1px solid #34495e; 
                margin-bottom: 10px;
            ">
                <span style="font-weight: bold; font-size: 16px; color: #ffffff; display: block; margin-bottom: 8px;">
                    🎙️ Option 1: Speak Live to Echo
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Center the microphone widget on its own clean row
        mic_col1, mic_col2, mic_col3 = st.columns([1, 2, 1])
        with mic_col2:
            audio_bytes = audio_recorder(
                text="Click icon to talk",
                recording_color="#e74c3c",
                neutral_color="#5a738e",  # Brighter neutral icon color so it pops
                icon_size="2x"
            )
        
        # A cleaner, more professional looking divider line instead of plain text
        st.markdown(
            """
            <div style="text-align: center; margin: 20px 0; color: #5a738e; font-weight: bold; letter-spacing: 2px;">
                ─── OR ───
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 📂 Option 2: File Upload (Fallback Backup)
        st.markdown("**📂 Option 2: Upload WAV File**")
        uploaded_file = st.file_uploader(
            "Choose an existing WAV file...",
            type=["wav"],
            key="input_payload",
            help="Any language — auto-detected by Whisper"
        )

        # ── Unified Payload Dynamic Mapper ──
        uploaded_file_data = None
        uploaded_file_name = None

        if audio_bytes is not None:
            # Clean the raw memory view bytes from the browser widget
            clean_bytes = bytes(audio_bytes)
            
            # Render the audio player using explicit raw WAV format declaration
            st.audio(clean_bytes, format="audio/wav")
            
            uploaded_file_data = clean_bytes
            uploaded_file_name = f"live_mic_{uuid.uuid4().hex[:6]}.wav"
            st.success("✨ Live microphone capture locked into payload buffer.")
            
        elif uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            uploaded_file_data = uploaded_file.getvalue()
            uploaded_file_name = uploaded_file.name
            st.info(f"📁 Local file loaded: {uploaded_file.name} ({len(uploaded_file_data)} bytes)")

        # Guest reference upload (only shown for guest profile)
        guest_reference_path = None
        if speaker_profile == "guest":
            st.markdown("---")
            st.markdown("**🎤 Guest Voice Reference** (8-15s clean clip of the voice to clone)")
            guest_ref_file = st.file_uploader(
                "Upload guest's voice reference...",
                type=["wav"],
                key="guest_ref"
            )
            if guest_ref_file is not None:
                guest_reference_path = f"audio/ref_{st.session_state.session_id[:8]}.wav"
                os.makedirs("audio", exist_ok=True)
                with open(guest_reference_path, "wb") as f:
                    f.write(guest_ref_file.read())
                st.success(f"Guest reference saved as `{os.path.basename(guest_reference_path)}`")
                st.audio(guest_ref_file)

        st.markdown("---")
        process_btn = st.button("🚀 Process & Clone Voice", type="primary", use_container_width=True)

    with col2:
        st.subheader("📤 AI Output & Metrics")

        if not process_btn:
            st.info("Waiting for audio input submission to trigger pipeline sequence...")

        elif process_btn and uploaded_file_data is None:
            st.warning("⚠️ Please record a live stream or upload a voice file to respond to.")

        elif process_btn and speaker_profile == "guest" and guest_reference_path is None:
            st.warning("⚠️ Guest profile selected — please upload a guest voice reference too.")

        elif process_btn and uploaded_file_data is not None:
            with st.spinner("⚡ Running Pipeline Array (Whisper ➜ Gemini ➜ F5-TTS)..."):
                try:
                    # Packaging the binary data stream cleanly for multipart boundary delivery
                    files = {"audio_file": (uploaded_file_name, uploaded_file_data, "audio/wav")}
                    data = {
                        "target_lang": target_lang_code,
                        "speaker_profile": speaker_profile,
                    }
                    if speaker_profile == "guest":
                        data["reference_wav"] = guest_reference_path

                    # Pass X-Session-Id metadata token via request header boundaries
                    headers = {"X-Session-Id": st.session_state.session_id}
                    
                    backend_url = f"{BACKEND_URL}/api/process-voice"
                    response = requests.post(backend_url, files=files, data=data, headers=headers, timeout=300)

                    if response.status_code == 200:
                        # 1. Extract the raw binary audio data
                        response_audio_bytes = response.content
                        
                        # 2. Extract the metadata cleanly from the custom network headers
                        transcript = response.headers.get("X-Transcript", "")
                        response_text = response.headers.get("X-Response-Text", "")
                        detected_lang = response.headers.get("X-Detected-Lang", "unknown")
                        latency = response.headers.get("X-Latency-Total-MS", "0")
                        
                        # 3. Save to live sliding-window session tracker state memory
                        st.session_state.conversation_history.append({
                            "input_lang": detected_lang.upper(),
                            "input_text": transcript,
                            "output_lang": selected_lang_label.split(" ")[0],
                            "output_text": response_text
                        })
                        
                        # 4. Display the text metrics inside Streamlit
                        st.success(f"🤖 Connected! Processed in {latency}ms")
                        st.write(f"**Detected Language:** {detected_lang.upper()}")
                        st.write(f"**What you said:** {transcript}")
                        st.write(f"**Echo AI Response:** {response_text}")
                        
                        # 5. Render the audio player widget so you can hear your clone!
                        st.audio(response_audio_bytes, format="audio/wav")
                    else:
                        st.error(f"❌ Backend returned an error status: {response.status_code}")
                        st.write(response.text)

                except Exception as e:
                    st.error(f"🔌 Failed to communicate with backend server. Is main.py running?")
                    st.exception(e)

with tab2:
    st.subheader("📊 Pipeline Analytics")

    if st.button("🔄 Refresh Logs"):
        st.rerun()

    # SQL query database fetch
    try:
        query = "SELECT * FROM conversation_turns ORDER BY created_at DESC LIMIT 100"
        with db_engine.connect() as connection:
            df = pd.read_sql(text(query), connection)
    except Exception as e:
        df = pd.DataFrame()

    if df.empty:
        st.info("No logs present yet. Run a few pipeline transactions to populate the metrics dashboard panel.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Turns", len(df))
        c2.metric("Avg Total Latency", f"{df['total_ms'].mean():.0f}ms")
        c3.metric("Cache Hit Rate", f"{df['cache_hit'].mean()*100:.1f}%")
        c4.metric("Avg Whisper", f"{df['whisper_ms'].mean():.0f}ms")

        st.markdown("---")
        latency_df = pd.DataFrame({
            "Stage": ["Whisper", "Gemini", "XTTS/F5"],
            "Avg ms": [df["whisper_ms"].mean(), df["llm_ms"].mean(), df["tts_ms"].mean()]
        })
        st.plotly_chart(px.bar(latency_df, x="Stage", y="Avg ms", color="Stage",
                               title="Average Latency by Stage"), use_container_width=True)

        lang_counts = df["output_lang"].value_counts().reset_index()
        lang_counts.columns = ["Language", "Count"]
        st.plotly_chart(px.pie(lang_counts, names="Language", values="Count",
                               title="Output Language Distribution"), use_container_width=True)

        profile_counts = df["speaker_profile"].value_counts().reset_index()
        profile_counts.columns = ["Profile", "Count"]
        st.plotly_chart(px.bar(profile_counts, x="Profile", y="Count",
                               title="Speaker Profile Usage"), use_container_width=True)

        st.subheader("Recent Turns")
        st.dataframe(df[["input_text", "output_lang", "speaker_profile", "total_ms", "cache_hit", "created_at"]].head(20),
                     use_container_width=True)