import json
import os
from confluent_kafka import Consumer
from pipeline.models import SessionLocal, ConversationTurn, StageEvent, create_tables

# 🌐 DYNAMIC ROUTING: Use 'kafka:29092' inside Docker mesh, fall back to localhost for manual tests
KAFKA_SERVER = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

KAFKA_CONFIG = {
    "bootstrap.servers": KAFKA_SERVER,
    "group.id": "polyglot-consumer-group",
    "auto.offset.reset": "earliest"
}

def run_consumer():
    create_tables()
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe(["session_events"])
    print(f"[Consumer] Listening on session_events at {KAFKA_SERVER}...")

    # Accumulate events per session until SESSION_END hits
    session_buffer = {}

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[Consumer] Error: {msg.error()}")
                continue

            # Safely decode the verified payload
            event = json.loads(msg.value().decode())
            session_id = event["session_id"]
            event_type = event["event_type"]
            latency_ms = event.get("latency_ms", 0)
            metadata = event.get("metadata", {})

            # 🎯 FIX: Instantiate DB session right here to prevent connection leaks on empty polls
            db = SessionLocal()
            try:
                # Always log raw stage event for deep diagnostic tracking
                db.add(StageEvent(
                    session_id=session_id,
                    event_type=event_type,
                    latency_ms=latency_ms,
                    metadata_json=json.dumps(metadata)
                ))

                # Initialize state buffer dict for this specific conversational turn
                if session_id not in session_buffer:
                    session_buffer[session_id] = {}
                buf = session_buffer[session_id]

                if event_type == "TRANSCRIPT_DONE":
                    buf["input_text"] = metadata.get("transcript", "")
                    buf["input_lang"] = metadata.get("detected_lang", "")
                    buf["whisper_ms"] = latency_ms
                elif event_type == "LLM_FIRST_CHUNK":
                    buf["llm_ms"] = latency_ms
                elif event_type == "TTS_DONE":
                    buf["tts_ms"] = latency_ms
                    buf["speaker_profile"] = metadata.get("speaker_profile", "unknown")
                elif event_type == "SESSION_END":
                    buf["total_ms"] = latency_ms
                    buf["cache_hit"] = metadata.get("cache_hit", False)
                    buf["output_text"] = metadata.get("response_text", "")
                    buf["output_lang"] = metadata.get("target_lang", "")

                    # Map state buffer variables straight into the analytical relational tables
                    turn = ConversationTurn(
                        session_id=session_id,
                        input_text=buf.get("input_text", ""),
                        input_lang=buf.get("input_lang", ""),
                        output_text=buf.get("output_text", ""),
                        output_lang=buf.get("output_lang", ""),
                        speaker_profile=buf.get("speaker_profile", "unknown"),
                        whisper_ms=buf.get("whisper_ms", 0),
                        llm_ms=buf.get("llm_ms", 0),
                        tts_ms=buf.get("tts_ms", 0),
                        total_ms=buf.get("total_ms", 0),
                        cache_hit=buf.get("cache_hit", False)
                    )
                    db.add(turn)
                    print(f"[Consumer] Stored consolidated turn transaction for session {session_id[:8]}")
                    del session_buffer[session_id]

                db.commit()
            except Exception as e:
                print(f"[Consumer DB Error] Failed to process message transaction: {e}")
                db.rollback()
            finally:
                db.close()

    except KeyboardInterrupt:
        print("[Consumer] Shutting down clean.")
    finally:
        consumer.close()

if __name__ == "__main__":
    run_consumer()