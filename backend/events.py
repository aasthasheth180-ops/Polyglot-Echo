# backend/events.py
"""
Real event pipeline using Upstash QStash — replaces the no-op emit_event/
flush stub that used to live in pipeline.py (which existed only because
the original code imported a `kafka_producer.py` that didn't exist).

How it works:
  pipeline.py calls emit_event(...) at each pipeline stage
    -> this publishes the event as a JSON message to QStash
    -> QStash reliably (with retries) POSTs it to YOUR OWN public webhook
       endpoint: {PUBLIC_APP_URL}/internal/qstash-webhook
    -> main.py's webhook handler writes it into the `stage_events` table
       (already defined in models.py, previously unused)

This decouples the slow "write to Postgres" step from the hot request path
— the user gets their answer immediately, and event logging happens async,
with QStash's automatic retries if Supabase hiccups.

Requires two things you already have: QSTASH_TOKEN (secret) and
PUBLIC_APP_URL (the same public URL your HF Space is reachable at, e.g.
https://aaasthasss-polyglot-echo-ai.hf.space) — add PUBLIC_APP_URL as a
Space "Variable" if it isn't already set; falls back to AI_SERVICE_URL if
you'd rather not add a new one, since in your setup they're the same host.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

QSTASH_TOKEN = os.getenv("QSTASH_TOKEN")
PUBLIC_APP_URL = (os.getenv("PUBLIC_APP_URL") or os.getenv("AI_SERVICE_URL", "")).rstrip("/")

_client = None
if QSTASH_TOKEN:
    try:
        from qstash import QStash
        _client = QStash(QSTASH_TOKEN)
    except Exception as e:
        logger.warning(f"[Events] QStash client unavailable, events will be logged locally only: {e}")


def emit_event(session_id: str, event_type: str, latency_ms: int = None, metadata: dict = None):
    """
    Fire-and-forget: publish a pipeline stage event. Never raises — a
    logging/telemetry failure must never break the actual chat/voice
    request in progress.
    """
    payload = {
        "session_id": session_id,
        "event_type": event_type,
        "latency_ms": latency_ms,
        "metadata": metadata or {},
    }

    if not _client or not PUBLIC_APP_URL:
        logger.debug(f"[Events] (local only, QStash not configured) {payload}")
        return

    try:
        _client.message.publish_json(
            url=f"{PUBLIC_APP_URL}/internal/qstash-webhook",
            body=payload,
            retries=3,
        )
    except Exception as e:
        # Non-fatal by design — never let telemetry take down the pipeline.
        logger.warning(f"[Events] QStash publish failed (non-fatal): {e}")


def flush():
    """
    Kept for compatibility with pipeline.py's existing call sites.
    QStash publishes are already async/fire-and-forget over HTTP, so
    there's nothing to flush — this is a no-op on purpose.
    """
    pass