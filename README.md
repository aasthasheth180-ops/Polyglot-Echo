Polyglot Echo: Real-Time Multilingual Voice Cloner & Telemetry Pipeline
Polyglot Echo is an end-to-end, low-latency conversational AI pipeline designed to transcribe user speech, generate context-aware LLM translations, and instantly clone the speaker's voice to read back the response in multiple target languages.

![Uploading image.png…]()



Beyond core ML modeling, this project is built with a heavy emphasis on Data Engineering production standards, featuring a robust event-driven telemetry logging infrastructure, asynchronous distributed metrics streaming, and an in-memory caching layer to drop pipeline processing bottlenecks.

Architecture & Core Components
The pipeline bridges deep learning inference with microservice analytics across three core stages:

Speech-to-Text (STT): Powered by an optimized deployment of OpenAI's Whisper, which automatically detects the input language from short live microphone bursts or file uploads while dynamically suppressing silent-frame loops.

Contextual Translation (LLM): Orchestrated via the Gemini API to act as the conversational engine, maintaining a sliding-window session context memory for true conversational back-and-forths.

Zero-Shot Voice Synthesis (TTS): Utilizing F5-TTS, the pipeline extracts a vocal profile from a raw audio sample and generates natural, multilingual voice clones on the fly.

Custom Audio DSP Layer: Implemented with Librosa and SciPy to automatically run RMS volume normalization and a 1st-order 2700Hz low-pass Butterworth filter on raw audio arrays, eliminating high-frequency digital jitter and clipping before user playback.

Data Engineering & MLOps Highlights
Event-Driven Telemetry Logging: Integrated Apache Kafka to capture microservice telemetry data asynchronously, preventing pipeline execution stalls by decoupled routing of latency metrics and language classifications.

Low-Latency In-Memory Caching: Uses Redis as a distributed caching layer to drop redundant generation steps for recurring conversational states and tokens.

Analytical Engine Logs: Telemetry events streamed through Kafka are loaded into a relational PostgreSQL database, driving a real-time system monitoring dashboard that tracks component latency splits, cache hit rates, and speaker profile usage.

Tech Stack
Core Backend: Python, FastAPI, SciPy, Librosa

Data Infrastructure: Apache Kafka, Redis, PostgreSQL

Machine Learning & GenAI: F5-TTS, Whisper, Gemini API, Hugging Face

Frontend UI & Visualization: Streamlit, Plotly Express
