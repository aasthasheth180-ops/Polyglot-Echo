# pipeline/kafka_producer.py
import os
from kafka import KafkaProducer
import json
from dotenv import load_dotenv

load_dotenv()

# This pulls 'localhost:9092' from your laptop's .env file perfectly!
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def emit_event(session_id, event_type, latency_ms=0, metadata=None):
    payload = {
        "session_id": session_id,
        "event_type": event_type,
        "latency_ms": latency_ms,
        "metadata": metadata or {}
    }
    producer.send("session_events", value=payload)

def flush():
    producer.flush()