# pipeline/kafka_producer.py
import os
from kafka import KafkaProducer
import json
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Wrap the initialization in a try-except so it doesn't crash the server
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        request_timeout_ms=5000  # Give up after 5 seconds instead of hanging
    )
    print("[Kafka] ✅ Successfully connected to Kafka broker.")
except Exception as e:
    print(f"[Kafka] ⚠️ Could not connect to Kafka broker: {e}")
    print("[Kafka] Running in fallback mode (events will not be emitted).")
    producer = None

def emit_event(session_id, event_type, latency_ms=0, metadata=None):
    if producer is None:
        return # Skip quietly if Kafka isn't available
    try:
        payload = {
            "session_id": session_id,
            "event_type": event_type,
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }
        producer.send("session_events", value=payload)
    except Exception as e:
        print(f"[Kafka Warning] Failed to emit event: {e}")

def flush():
    if producer:
        try:
            producer.flush()
        except Exception:
            pass