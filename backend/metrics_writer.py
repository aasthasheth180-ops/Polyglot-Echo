# backend/metrics_writer.py
import os
import csv
from datetime import datetime, timezone

METRICS_FILE = "pipeline_metrics.csv"

class MetricsWriter:
    def __init__(self):
        self.bucket = "pipeline_metrics"
        # If the metrics database file doesn't exist yet, initialize it with strict headers
        if not os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "measurement", "stage", "language", "speaker_profile", "cache_hit", "latency_ms"])
        print("[Metrics Node] InfluxDB schema protocol ready (Resource-Optimized Local Stream Mode)")

    def write_turn(self, session_id: str, whisper_ms: int, llm_ms: int,
                   tts_ms: int, total_ms: int, language: str,
                   speaker_profile: str, cache_hit: bool):
        
        # Define the exact operational pipeline stages to break down
        stages = {
            "whisper": whisper_ms,
            "llm": llm_ms,
            "tts": tts_ms,
            "total": total_ms
        }
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            # Open the file stream in append mode to dynamically log data without locking memory
            with open(METRICS_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for stage, ms in stages.items():
                    # Conforms precisely to the structural layout of InfluxDB line protocol tags and fields
                    writer.writerow([
                        timestamp,
                        "pipeline_latency",   # InfluxDB Measurement
                        stage,                # Tag 1
                        language,             # Tag 2
                        speaker_profile,      # Tag 3
                        str(cache_hit).lower(), # Tag 4
                        float(ms)             # Field Value (Latency Metric)
                    ])
            print(f"[Metrics Node] Successfully streamed transaction steps for session {session_id[:8]} to time-series log.")
        except Exception as e:
            print(f"[Metrics Node] Ingestion alert (non-critical): {e}")

metrics_writer = MetricsWriter()